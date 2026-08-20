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
EXPERIMENT_INDEX=""
DRY_RUN=0

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
        --experiment-index) EXPERIMENT_INDEX="$2"; shift 2 ;;
        --dry-run) DRY_RUN=1; shift ;;
        --) echo "extra ModelOpt dotlist arguments are not accepted for pinned production manifests" >&2; exit 2 ;;
        *) usage ;;
    esac
done
[[ "$MANIFEST" == /home/* && "$RECEIPT" == /lustre/* && -f "$MANIFEST" ]] || usage
[[ "$RUNNER" == /home/* ]] || { echo "runner must be in /home source" >&2; exit 2; }
[[ -z "$ONLY_STEP" || "$ONLY_STEP" =~ ^[1-9][0-9]*$ ]] || usage
[[ -z "$EXPERIMENT_INDEX" || "$EXPERIMENT_INDEX" =~ ^[0-9]+$ ]] || usage
render_waves() {
    PYTHONPATH="$LAUNCHER_ROOT${PYTHONPATH:+:$PYTHONPATH}" python3 - "$MANIFEST" "$ONLY_STEP" "$EXPERIMENT_INDEX" <<'PY'
import sys
from pathlib import Path
from common.specdec.drafter_job_manifest import load_manifest

selected = sys.argv[2]
selected_index = sys.argv[3]
for index, experiment in enumerate(load_manifest(Path(sys.argv[1]))):
    if selected_index and index != int(selected_index):
        continue
    for boundary in experiment.cumulative_max_steps:
        if selected and boundary != int(selected):
            continue
        print(f"{index}\t{experiment.experiment_id}\t{boundary}\t{experiment.run_name}\t{experiment.slurm.account}\t{experiment.slurm.partition}")
PY
}

declare -A identities=()
while IFS=$'\t' read -r index identity boundary _run_name account partition; do
    [[ -n "$index" ]] || continue
    tuple_identity="${identity}:${boundary}"
    [[ -z "${identities[$tuple_identity]:-}" ]] || { echo "duplicate training tuple: $tuple_identity" >&2; exit 2; }
    identities[$tuple_identity]=1
    job_name="drafter-train-${identity}-s${boundary}"
    receipt_job="$(python3 - "$RECEIPT" "$tuple_identity" <<'PY'
import json, sys
from pathlib import Path
path = Path(sys.argv[1])
if path.exists():
    for line in reversed(path.read_text().splitlines()):
        record = json.loads(line)
        if record.get("tuple_identity") == sys.argv[2] and record.get("job_id"):
            print(record["job_id"])
            break
PY
)"
    if [[ -n "$receipt_job" ]]; then
        printf '{"job_id":"%s","status":"receipt","tuple_identity":"%s"}\n' "$receipt_job" "$tuple_identity" >>"$RECEIPT"
        continue
    fi
    existing="$(squeue -h -n "$job_name" -o "%A" | head -n 1 || true)"
    [[ -n "$existing" ]] || existing="$(sacct -X -n --name "$job_name" --format=JobIDRaw,State | awk 'NF {print $1; exit}' || true)"
    if [[ -n "$existing" ]]; then
        printf '{"job_id":"%s","status":"already-known","tuple_identity":"%s","max_steps":%s}\n' "$existing" "$tuple_identity" "$boundary" >>"$RECEIPT"
        continue
    fi
    exports="ALL,MANIFEST_PATH=${MANIFEST},EXPERIMENT_INDEX=${index},MAX_STEPS=${boundary}"
    args=(--account="$account" --partition="$partition" -N4 --ntasks-per-node=1 --gpus-per-node=4 --segment=4 --time=03:55:00 --job-name="$job_name" --comment="$tuple_identity" --export="$exports")
    [[ -z "$DEPENDENCY" ]] || args+=(--dependency=afterok:"$DEPENDENCY")
    sbatch --test-only "${args[@]}" "$RUNNER"
    if [[ "$DRY_RUN" -eq 1 ]]; then
        printf '{"job_name":"%s","status":"test-only","tuple_identity":"%s","max_steps":%s}\n' "$job_name" "$tuple_identity" "$boundary" >>"$RECEIPT"
        continue
    fi
    submitted="$(sbatch --parsable "${args[@]}" "$RUNNER" || true)"
    job_id="${submitted%%;*}"
    if [[ -z "$job_id" ]]; then
        job_id="$(squeue -h -n "$job_name" -o "%A" | head -n 1 || true)"
        [[ -n "$job_id" ]] || job_id="$(sacct -X -n --name "$job_name" --format=JobIDRaw,State | awk 'NF {print $1; exit}' || true)"
    fi
    [[ -n "$job_id" ]] || { echo "scheduler did not confirm training submission: $job_name" >&2; exit 1; }
    printf '{"job_id":"%s","status":"submitted","tuple_identity":"%s","max_steps":%s,"job_name":"%s"}\n' "$job_id" "$tuple_identity" "$boundary" "$job_name" >>"$RECEIPT"
done < <(render_waves)
