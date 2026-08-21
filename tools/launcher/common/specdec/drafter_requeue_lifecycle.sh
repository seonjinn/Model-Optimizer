#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Opt-in lifecycle helpers for checkpoint-gated Slurm self-requeue.

set -euo pipefail

DRAFTER_REQUEUE_STEP_PID=""

drafter_latest_complete_checkpoint() {
    python3 - "$1" "${2:-}" "${EXPECTED_RNG_STATES:-8}" <<'PY'
import json
import re
import sys
from pathlib import Path

output_root = Path(sys.argv[1])
quarantine_root = Path(sys.argv[2]) if sys.argv[2] else None
expected_rng_states = int(sys.argv[3])
weight_names = (
    "model.safetensors",
    "model.safetensors.index.json",
    "pytorch_model.bin",
    "pytorch_model.bin.index.json",
)
candidates = []
if output_root.is_dir():
    for path in output_root.iterdir():
        match = re.fullmatch(r"checkpoint-([1-9][0-9]*)", path.name)
        if match and path.is_dir():
            candidates.append((int(match.group(1)), path))

for directory_step, checkpoint in sorted(candidates, reverse=True):
    state_path = checkpoint / "trainer_state.json"
    try:
        state = json.loads(state_path.read_text())
        global_step = int(state["global_step"])
    except (FileNotFoundError, KeyError, TypeError, ValueError, json.JSONDecodeError):
        continue
    if global_step != directory_step:
        continue
    if not (checkpoint / "optimizer.pt").is_file() or (checkpoint / "optimizer.pt").stat().st_size == 0:
        continue
    if not (checkpoint / "scheduler.pt").is_file() or (checkpoint / "scheduler.pt").stat().st_size == 0:
        continue
    if not any((checkpoint / name).is_file() and (checkpoint / name).stat().st_size > 0 for name in weight_names):
        continue
    if not all(
        (checkpoint / f"rng_state_{rank}.pth").is_file()
        and (checkpoint / f"rng_state_{rank}.pth").stat().st_size > 0
        for rank in range(expected_rng_states)
    ):
        continue
    if quarantine_root is not None:
        quarantine_root.mkdir(parents=True, exist_ok=False)
        for newer_step, newer_checkpoint in candidates:
            if newer_step > global_step:
                newer_checkpoint.rename(quarantine_root / newer_checkpoint.name)
    print(f"{checkpoint}\t{global_step}")
    raise SystemExit(0)
raise SystemExit(1)
PY
}

drafter_completed_training_artifact() {
    python3 - "$OUTPUT_ROOT" "$EXPORT_PATH" "$MAX_STEPS" <<'PY'
import json
import sys
from pathlib import Path

output_root = Path(sys.argv[1])
export_path = Path(sys.argv[2])
target_step = int(sys.argv[3])
weight_names = (
    "model.safetensors",
    "model.safetensors.index.json",
    "pytorch_model.bin",
    "pytorch_model.bin.index.json",
)
try:
    state = json.loads((output_root / "trainer_state.json").read_text())
    global_step = int(state["global_step"])
except (FileNotFoundError, KeyError, TypeError, ValueError, json.JSONDecodeError):
    raise SystemExit(1)
if global_step < target_step:
    raise SystemExit(1)
for directory in (output_root, export_path):
    if not any((directory / name).is_file() and (directory / name).stat().st_size > 0 for name in weight_names):
        raise SystemExit(1)
print(f"{output_root}\t{global_step}")
PY
}

drafter_write_lifecycle_record() {
    local path="$1" status="$2" checkpoint="$3" global_step="$4"
    mkdir -p "$(dirname "$path")"
    python3 - "$path" "$status" "$checkpoint" "$global_step" \
        "$SLURM_JOB_ID" "${SLURM_RESTART_COUNT:-0}" "$MAX_STEPS" <<'PY'
import json
import os
import sys
from pathlib import Path

path = Path(sys.argv[1])
record = {
    "checkpoint": sys.argv[3],
    "global_step": int(sys.argv[4]),
    "job_id": sys.argv[5],
    "restart_count": int(sys.argv[6]),
    "status": sys.argv[2],
    "target_step": int(sys.argv[7]),
}
temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
temporary.write_text(json.dumps(record, sort_keys=True) + "\n")
os.replace(temporary, path)
PY
}

drafter_stop_requeueable_step() {
    local pid="$DRAFTER_REQUEUE_STEP_PID"
    [[ -n "$pid" ]] || return 0
    kill -TERM "$pid" 2>/dev/null || true
    local waited=0 grace="${REQUEUE_TERMINATE_GRACE_SECONDS:-120}"
    while kill -0 "$pid" 2>/dev/null && (( waited < grace )); do
        sleep 1
        waited=$((waited + 1))
    done
    if kill -0 "$pid" 2>/dev/null; then
        kill -KILL "$pid" 2>/dev/null || true
    fi
    wait "$pid" 2>/dev/null || true
    DRAFTER_REQUEUE_STEP_PID=""
}

drafter_handle_usr1() {
    trap - USR1
    drafter_stop_requeueable_step

    local restart_count="${SLURM_RESTART_COUNT:-0}"
    local receipt_root="${OUTPUT_ROOT}/control/requeue/job-${SLURM_JOB_ID}"
    local receipt="${receipt_root}/attempt-${restart_count}.json"
    local checkpoint_record checkpoint global_step
    local quarantine="${receipt_root}/quarantine-${restart_count}"
    if ! checkpoint_record="$(drafter_latest_complete_checkpoint "$OUTPUT_ROOT" "$quarantine")"; then
        drafter_write_lifecycle_record "$receipt" "failed-no-complete-checkpoint" "" 0
        echo "USR1 received without a complete resumable checkpoint; refusing requeue" >&2
        exit 1
    fi
    IFS=$'\t' read -r checkpoint global_step <<<"$checkpoint_record"
    if (( restart_count >= MAX_REQUEUES )); then
        drafter_write_lifecycle_record "$receipt" "failed-restart-limit" "$checkpoint" "$global_step"
        echo "self-requeue restart limit reached: ${restart_count}/${MAX_REQUEUES}" >&2
        exit 1
    fi
    if ! scontrol requeue "$SLURM_JOB_ID"; then
        drafter_write_lifecycle_record "$receipt" "failed-scontrol-requeue" "$checkpoint" "$global_step"
        echo "scontrol refused requeue for job ${SLURM_JOB_ID}" >&2
        exit 1
    fi
    drafter_write_lifecycle_record "$receipt" "requeued" "$checkpoint" "$global_step"
    exit 0
}

drafter_requeue_init() {
    [[ "${SELF_REQUEUE:-0}" == 1 ]] || { echo "self-requeue lifecycle is not enabled" >&2; return 2; }
    [[ "${MAX_STEPS:-}" =~ ^[1-9][0-9]*$ ]] || { echo "MAX_STEPS must be positive" >&2; return 2; }
    [[ "${MAX_REQUEUES:-}" =~ ^[1-9][0-9]*$ ]] || { echo "MAX_REQUEUES must be positive" >&2; return 2; }
    [[ -n "${OUTPUT_ROOT:-}" && -n "${EXPORT_PATH:-}" && -n "${SLURM_JOB_ID:-}" ]] || {
        echo "OUTPUT_ROOT, EXPORT_PATH, and SLURM_JOB_ID are required" >&2
        return 2
    }
    [[ "${SLURM_RESTART_COUNT:-0}" =~ ^[0-9]+$ ]] || {
        echo "SLURM_RESTART_COUNT must be non-negative" >&2
        return 2
    }
    trap drafter_handle_usr1 USR1
}

drafter_run_requeueable_step() {
    "$@" &
    DRAFTER_REQUEUE_STEP_PID=$!
    local status
    if wait "$DRAFTER_REQUEUE_STEP_PID"; then
        status=0
    else
        status=$?
    fi
    DRAFTER_REQUEUE_STEP_PID=""
    return "$status"
}

drafter_mark_training_complete() {
    local checkpoint_record checkpoint global_step
    if ! checkpoint_record="$(drafter_completed_training_artifact)"; then
        echo "training exited successfully without complete final artifacts" >&2
        return 1
    fi
    IFS=$'\t' read -r checkpoint global_step <<<"$checkpoint_record"
    drafter_write_lifecycle_record \
        "${OUTPUT_ROOT}/control/training-complete-s${MAX_STEPS}.json" \
        "completed" "$checkpoint" "$global_step"
}

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
    case "${1:-}" in
        latest-complete)
            [[ $# -eq 2 ]] || exit 2
            drafter_latest_complete_checkpoint "$2"
            ;;
        *)
            echo "usage: $0 latest-complete OUTPUT_ROOT" >&2
            exit 2
            ;;
    esac
fi
