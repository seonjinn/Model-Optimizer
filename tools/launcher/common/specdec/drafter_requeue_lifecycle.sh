#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Opt-in lifecycle helpers for checkpoint-gated Slurm self-requeue.

set -euo pipefail

DRAFTER_REQUEUE_STEP_PID=""

drafter_latest_complete_checkpoint() {
    python3 - "$1" "${2:-}" "${EXPECTED_RNG_STATES:-8}" "${3:-}" <<'PY'
import json
import re
import sys
from pathlib import Path

output_root = Path(sys.argv[1])
quarantine_root = Path(sys.argv[2]) if sys.argv[2] else None
expected_rng_states = int(sys.argv[3])
maximum_step = int(sys.argv[4]) if sys.argv[4] else None
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
            step = int(match.group(1))
            if maximum_step is None or step <= maximum_step:
                candidates.append((step, path))

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

drafter_preserve_milestone() {
    case ",${MILESTONE_STEPS:-4166,25391}," in
        *",${MAX_STEPS},"*) ;;
        *) return 0 ;;
    esac

    local checkpoint_record checkpoint checkpoint_step
    if ! checkpoint_record="$(drafter_latest_complete_checkpoint "$OUTPUT_ROOT" "" "$MAX_STEPS")"; then
        echo "milestone ${MAX_STEPS} has no complete resumable checkpoint to preserve" >&2
        return 1
    fi
    IFS=$'\t' read -r checkpoint checkpoint_step <<<"$checkpoint_record"
    python3 - "$OUTPUT_ROOT" "$EXPORT_PATH" "$MAX_STEPS" "$checkpoint" "$checkpoint_step" <<'PY'
import errno
import hashlib
import json
import os
import shutil
import sys
from pathlib import Path

output_root = Path(sys.argv[1])
export_path = Path(sys.argv[2])
target_step = int(sys.argv[3])
checkpoint = Path(sys.argv[4])
checkpoint_step = int(sys.argv[5])
milestones_root = output_root / "milestones"
milestone = milestones_root / f"step-{target_step:06d}"
resume_name = f"resume-checkpoint-{checkpoint_step:06d}"
manifest_base = {
    "exact_model_path": f"../../{export_path.name}",
    "exact_model_step": target_step,
    "resume_checkpoint_path": resume_name,
    "resume_checkpoint_step": checkpoint_step,
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def regular_file_hashes(root: Path) -> dict[str, str]:
    return {
        str(path.relative_to(root)): sha256(path)
        for path in sorted(root.rglob("*"))
        if path.is_file() and not path.is_symlink()
    }


def validate_existing() -> None:
    try:
        existing = json.loads((milestone / "manifest.json").read_text())
    except (FileNotFoundError, json.JSONDecodeError) as error:
        raise RuntimeError(f"invalid existing milestone: {milestone}") from error
    for key, value in manifest_base.items():
        if existing.get(key) != value:
            raise RuntimeError(f"milestone manifest mismatch: {milestone}")
    if not existing.get("exact_model_sha256") or not existing.get("resume_checkpoint_sha256"):
        raise RuntimeError(f"milestone manifest mismatch: {milestone}")
    if not (milestone / "exact-model").is_symlink():
        raise RuntimeError(f"milestone exact-model link is missing: {milestone}")
    if (milestone / "exact-model").resolve() != export_path.resolve():
        raise RuntimeError(f"milestone exact-model link changed: {milestone}")
    if not (milestone / resume_name / "trainer_state.json").is_file():
        raise RuntimeError(f"milestone resume state is missing: {milestone}")
    if regular_file_hashes(export_path) != existing["exact_model_sha256"]:
        raise RuntimeError(f"milestone exact-model checksum mismatch: {milestone}")
    if regular_file_hashes(milestone / resume_name) != existing["resume_checkpoint_sha256"]:
        raise RuntimeError(f"milestone resume checksum mismatch: {milestone}")


if milestone.exists():
    validate_existing()
    raise SystemExit(0)

milestones_root.mkdir(parents=True, exist_ok=True)
temporary = milestones_root / f".{milestone.name}.tmp-{os.getpid()}"
temporary.mkdir()
try:
    os.symlink(manifest_base["exact_model_path"], temporary / "exact-model")
    resume = temporary / resume_name
    storage: dict[str, str] = {}
    force_copy = os.environ.get("MILESTONE_FORCE_COPY", "0") == "1"
    same_device = checkpoint.stat().st_dev == milestones_root.stat().st_dev
    for source in sorted(checkpoint.rglob("*")):
        relative = source.relative_to(checkpoint)
        destination = resume / relative
        if source.is_symlink():
            destination.parent.mkdir(parents=True, exist_ok=True)
            os.symlink(os.readlink(source), destination)
        elif source.is_dir():
            destination.mkdir(parents=True, exist_ok=True)
        elif source.is_file():
            destination.parent.mkdir(parents=True, exist_ok=True)
            try:
                if force_copy or not same_device:
                    raise OSError(errno.EXDEV, "milestone hardlink fallback requested")
                os.link(source, destination)
                storage[str(relative)] = "hardlink"
            except OSError as error:
                fallback_errnos = {errno.EXDEV, errno.EPERM, errno.EMLINK}
                if hasattr(errno, "EOPNOTSUPP"):
                    fallback_errnos.add(errno.EOPNOTSUPP)
                if error.errno not in fallback_errnos:
                    raise
                shutil.copy2(source, destination)
                if sha256(source) != sha256(destination):
                    raise RuntimeError(f"milestone copy integrity mismatch: {source}")
                storage[str(relative)] = "copy"
        else:
            raise RuntimeError(f"unsupported checkpoint entry: {source}")
    manifest = {
        **manifest_base,
        "exact_model_sha256": regular_file_hashes(export_path),
        "resume_checkpoint_sha256": regular_file_hashes(resume),
        "resume_checkpoint_storage": storage,
    }
    (temporary / "manifest.json").write_text(json.dumps(manifest, sort_keys=True) + "\n")
    os.rename(temporary, milestone)
except BaseException:
    shutil.rmtree(temporary, ignore_errors=True)
    raise
validate_existing()
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
    drafter_write_lifecycle_record "$receipt" "requeue-requested" "$checkpoint" "$global_step"
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
    drafter_preserve_milestone
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
