#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Render and submit unique four-node cumulative drafter training waves.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LAUNCHER_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
RUNNER="${SCRIPT_DIR}/run_drafter_training.sbatch"
MANIFEST=""
RECEIPT=""
DEPENDENCY=""
ONLY_STEP=""
DRY_RUN=0
EXTRA_ARGS=()

usage() {
    echo "usage: $0 --manifest /home/.../manifest.json --receipt /lustre/.../receipt.jsonl [--dependency JOBID] [--max-steps N] [--dry-run] [-- dotlist args]" >&2
    exit 2
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --manifest) MANIFEST="$2"; shift 2 ;;
        --receipt) RECEIPT="$2"; shift 2 ;;
        --dependency) DEPENDENCY="$2"; shift 2 ;;
        --max-steps) ONLY_STEP="$2"; shift 2 ;;
        --dry-run) DRY_RUN=1; shift ;;
        --) shift; EXTRA_ARGS=("$@"); break ;;
        *) usage ;;
    esac
done
[[ "$MANIFEST" == /home/* && "$RECEIPT" == /lustre/* && -f "$MANIFEST" ]] || usage
[[ "$RUNNER" == /home/* ]] || { echo "runner must be in /home source" >&2; exit 2; }
[[ -z "$ONLY_STEP" || "$ONLY_STEP" =~ ^[1-9][0-9]*$ ]] || usage
EXTRA_ARGS=("${EXTRA_ARGS[@]:-}")

render_waves() {
    PYTHONPATH="$LAUNCHER_ROOT${PYTHONPATH:+:$PYTHONPATH}" python3 - "$MANIFEST" "$ONLY_STEP" <<'PY'
import sys
from pathlib import Path
from common.specdec.drafter_job_manifest import load_manifest

selected = sys.argv[2]
for index, experiment in enumerate(load_manifest(Path(sys.argv[1]))):
    for boundary in experiment.cumulative_max_steps:
        if selected and boundary != int(selected):
            continue
        print(f"{index}\t{boundary}\t{experiment.run_name}\t{experiment.slurm.account}\t{experiment.slurm.partition}")
PY
}

declare -A identities=()
while IFS=$'\t' read -r index boundary run_name account partition; do
    [[ -n "$index" ]] || continue
    identity="${index}:${boundary}"
    [[ -z "${identities[$identity]:-}" ]] || { echo "duplicate training tuple: $identity" >&2; exit 2; }
    identities[$identity]=1
    job_name="${run_name}-s${boundary}"
    existing="$(squeue -h -n "$job_name" -o "%A" | head -n 1 || true)"
    if [[ -n "$existing" ]]; then
        printf '{"job_id":"%s","status":"already-queued","max_steps":%s}\n' "$existing" "$boundary" >>"$RECEIPT"
        continue
    fi
    exports="ALL,MANIFEST_PATH=${MANIFEST},EXPERIMENT_INDEX=${index},MAX_STEPS=${boundary},EXTRA_MODELOPT_DOTLIST=${EXTRA_ARGS[*]:-}"
    args=(--account="$account" --partition="$partition" -N4 --ntasks-per-node=1 --gpus-per-node=4 --segment=4 --time=03:55:00 --job-name="$job_name" --export="$exports")
    [[ -z "$DEPENDENCY" ]] || args+=(--dependency=afterok:"$DEPENDENCY")
    sbatch --test-only "${args[@]}" "$RUNNER"
    if [[ "$DRY_RUN" -eq 1 ]]; then
        printf '{"job_name":"%s","status":"test-only","max_steps":%s}\n' "$job_name" "$boundary" >>"$RECEIPT"
        continue
    fi
    submitted="$(sbatch --parsable "${args[@]}" "$RUNNER" || true)"
    job_id="${submitted%%;*}"
    if [[ -z "$job_id" ]]; then
        job_id="$(squeue -h -n "$job_name" -o "%A" | head -n 1 || true)"
    fi
    [[ -n "$job_id" ]] || { echo "scheduler did not confirm training submission: $job_name" >&2; exit 1; }
    printf '{"job_id":"%s","status":"submitted","max_steps":%s,"job_name":"%s"}\n' "$job_id" "$boundary" "$job_name" >>"$RECEIPT"
done < <(render_waves)
