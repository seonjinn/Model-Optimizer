#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Pre-submit the complete 32-run paper-scale matrix as bounded afterok chains.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LAUNCHER_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
WAVE_SUBMITTER="${SCRIPT_DIR}/submit_drafter_training_wave.sh"
MANIFEST=""
RECEIPT=""
DRY_RUN=0
TEST_DEPENDENCY_JOB=""
LEGACY_SOURCE_SHA=""

usage() {
    echo "usage: $0 --manifest /home/.../manifest.json --receipt /lustre/.../submission.jsonl [--dry-run --test-dependency-job COMPLETED_JOBID]" >&2
    exit 2
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --manifest) MANIFEST="$2"; shift 2 ;;
        --receipt) RECEIPT="$2"; shift 2 ;;
        --dry-run) DRY_RUN=1; shift ;;
        --test-dependency-job) TEST_DEPENDENCY_JOB="$2"; shift 2 ;;
        --legacy-source-sha) LEGACY_SOURCE_SHA="$2"; shift 2 ;;
        *) usage ;;
    esac
done
[[ "$MANIFEST" == /home/* && -f "$MANIFEST" && "$RECEIPT" == /lustre/* ]] || usage
[[ -z "$TEST_DEPENDENCY_JOB" || "$TEST_DEPENDENCY_JOB" =~ ^[1-9][0-9]*$ ]] || usage
[[ -z "$LEGACY_SOURCE_SHA" || "$LEGACY_SOURCE_SHA" =~ ^[0-9a-f]{40}$ ]] || usage
[[ "$DRY_RUN" -eq 0 || -n "$TEST_DEPENDENCY_JOB" ]] || usage

mkdir -p "$(dirname "$RECEIPT")"
scheduler_jobs="$({
    squeue -h -u "$USER" -o "%j|%A|%k|%T"
    sacct -X -n -P -u "$USER" -S today --format=JobName%64,JobIDRaw,Comment,State
} || true)"

render_plan() {
    PYTHONPATH="$LAUNCHER_ROOT${PYTHONPATH:+:$PYTHONPATH}" python3 - "$MANIFEST" <<'PY'
import sys
from pathlib import Path

from common.specdec.build_drafter_full_manifest import readable_job_name
from common.specdec.drafter_job_manifest import load_manifest

experiments = load_manifest(Path(sys.argv[1]))
if len(experiments) != 32:
    raise SystemExit("full chain manifest must contain exactly 32 experiments")
for index, experiment in enumerate(experiments):
    for boundary in experiment.cumulative_max_steps:
        print(
            "\t".join(
                (
                    str(index),
                    experiment.experiment_id,
                    str(boundary),
                    readable_job_name(
                        experiment.target,
                        experiment.dataset,
                        experiment.method,
                        experiment.block_size,
                        boundary,
                    ),
                    experiment.paths.output_root,
                )
            )
        )
PY
}

previous_index=""
previous_job=""
legacy_step=""
while IFS=$'\t' read -r index identity boundary job_name output_root; do
    if [[ "$index" != "$previous_index" ]]; then
        previous_index="$index"
        previous_job=""
        legacy_step=""
        if [[ ! -f "$output_root/control/training-identity.json" ]]; then
            checkpoint_record="$(bash "${SCRIPT_DIR}/drafter_requeue_lifecycle.sh" latest-complete "$output_root" 2>/dev/null || true)"
            if [[ -n "$checkpoint_record" ]]; then
                [[ -n "$LEGACY_SOURCE_SHA" ]] || {
                    echo "legacy checkpoint requires --legacy-source-sha: $output_root" >&2
                    exit 1
                }
                legacy_step="${checkpoint_record##*$'\t'}"
            elif compgen -G "$output_root/checkpoint-*" >/dev/null; then
                echo "legacy output has no complete resumable checkpoint: $output_root" >&2
                exit 1
            fi
        fi
    fi
    args=(
        --manifest "$MANIFEST"
        --receipt "$RECEIPT"
        --experiment-index "$index"
        --max-steps "$boundary"
        --job-name "$job_name"
        --save-steps 50
        --self-requeue
        --max-requeues 50
        --requeue-signal-lead 300
    )
    if [[ -n "$legacy_step" ]]; then
        args+=(
            --legacy-adoption-source-sha "$LEGACY_SOURCE_SHA"
            --legacy-adoption-checkpoint-step "$legacy_step"
        )
    fi
    if [[ "$DRY_RUN" -eq 1 ]]; then
        args+=(--dry-run)
        [[ -z "$previous_job" ]] || args+=(--dependency "$TEST_DEPENDENCY_JOB")
    elif [[ -n "$previous_job" ]]; then
        args+=(--dependency "$previous_job")
    fi
    SCHEDULER_JOBS_SNAPSHOT="$scheduler_jobs" bash "$WAVE_SUBMITTER" "${args[@]}"
    if [[ "$DRY_RUN" -eq 0 ]]; then
        tuple_identity="${identity}:${boundary}"
        previous_job="$(python3 - "$RECEIPT" "$tuple_identity" <<'PY'
import json
import sys
from pathlib import Path

for line in reversed(Path(sys.argv[1]).read_text().splitlines()):
    record = json.loads(line)
    if record.get("tuple_identity") == sys.argv[2] and record.get("job_id"):
        print(record["job_id"])
        break
PY
)"
        [[ -n "$previous_job" ]] || {
            echo "submission receipt lacks job ID for ${tuple_identity}" >&2
            exit 1
        }
    else
        previous_job="$TEST_DEPENDENCY_JOB"
    fi
done < <(render_plan)
