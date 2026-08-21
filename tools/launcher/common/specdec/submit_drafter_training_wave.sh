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
JOB_NAME_OVERRIDE=""
LEGACY_ADOPTION_SOURCE_SHA=""
LEGACY_ADOPTION_CHECKPOINT_STEP=""
SAVE_STEPS=""
DEFAULT_REQUEUE_SAVE_STEPS=50
SELF_REQUEUE=0
MAX_REQUEUES=50
REQUEUE_SIGNAL_LEAD=300
DRY_RUN=0

usage() {
    echo "usage: $0 --manifest /home/.../manifest.json --receipt /lustre/.../receipt.jsonl [--dependency JOBID] [--max-steps N] [--save-steps N] [--self-requeue] [--max-requeues N] [--requeue-signal-lead SECONDS] [--experiment-index N] [--job-name NAME] [--dry-run]" >&2
    exit 2
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --manifest) MANIFEST="$2"; shift 2 ;;
        --receipt) RECEIPT="$2"; shift 2 ;;
        --dependency) DEPENDENCY="$2"; shift 2 ;;
        --max-steps) ONLY_STEP="$2"; shift 2 ;;
        --save-steps) SAVE_STEPS="$2"; shift 2 ;;
        --self-requeue) SELF_REQUEUE=1; shift ;;
        --max-requeues) MAX_REQUEUES="$2"; shift 2 ;;
        --requeue-signal-lead) REQUEUE_SIGNAL_LEAD="$2"; shift 2 ;;
        --experiment-index) EXPERIMENT_INDEX="$2"; shift 2 ;;
        --job-name) JOB_NAME_OVERRIDE="$2"; shift 2 ;;
        --legacy-adoption-source-sha) LEGACY_ADOPTION_SOURCE_SHA="$2"; shift 2 ;;
        --legacy-adoption-checkpoint-step) LEGACY_ADOPTION_CHECKPOINT_STEP="$2"; shift 2 ;;
        --dry-run) DRY_RUN=1; shift ;;
        --) echo "extra ModelOpt dotlist arguments are not accepted for pinned production manifests" >&2; exit 2 ;;
        *) usage ;;
    esac
done
[[ "$MANIFEST" == /home/* && "$RECEIPT" == /lustre/* && -f "$MANIFEST" ]] || usage
[[ "$RUNNER" == /home/* ]] || { echo "runner must be in /home source" >&2; exit 2; }
[[ -z "$ONLY_STEP" || "$ONLY_STEP" =~ ^[1-9][0-9]*$ ]] || usage
[[ -z "$SAVE_STEPS" || "$SAVE_STEPS" =~ ^[1-9][0-9]*$ ]] || usage
[[ "$MAX_REQUEUES" =~ ^[1-9][0-9]*$ ]] || usage
[[ "$REQUEUE_SIGNAL_LEAD" =~ ^[1-9][0-9]*$ ]] || usage
[[ -z "$EXPERIMENT_INDEX" || "$EXPERIMENT_INDEX" =~ ^[0-9]+$ ]] || usage
[[ -z "$JOB_NAME_OVERRIDE" || ( -n "$ONLY_STEP" && -n "$EXPERIMENT_INDEX" && ${#JOB_NAME_OVERRIDE} -le 64 && "$JOB_NAME_OVERRIDE" =~ ^[a-zA-Z0-9._-]+$ ) ]] || usage
[[ ( -z "$LEGACY_ADOPTION_SOURCE_SHA" && -z "$LEGACY_ADOPTION_CHECKPOINT_STEP" ) || ( "$LEGACY_ADOPTION_SOURCE_SHA" =~ ^[0-9a-f]{40}$ && "$LEGACY_ADOPTION_CHECKPOINT_STEP" =~ ^[1-9][0-9]*$ ) ]] || usage
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
        print(f"{index}\t{experiment.experiment_id}\t{boundary}\t{experiment.run_name}\t{experiment.slurm.account}\t{experiment.slurm.partition}\t{experiment.paths.output_root}")
PY
}

declare -A identities=()
if [[ -n "${SCHEDULER_JOBS_SNAPSHOT+x}" ]]; then
    scheduler_jobs="$SCHEDULER_JOBS_SNAPSHOT"
else
    scheduler_jobs="$(
        {
            squeue -h -u "$USER" -o "%j|%A|%k|%T"
            sacct -X -n -P -u "$USER" -S today --format=JobName%64,JobIDRaw,Comment,State
        } || true
    )"
fi
while IFS=$'\t' read -r index identity boundary _run_name account partition output_root; do
    [[ -n "$index" ]] || continue
    tuple_identity="${identity}:${boundary}"
    [[ -z "${identities[$tuple_identity]:-}" ]] || { echo "duplicate training tuple: $tuple_identity" >&2; exit 2; }
    identities[$tuple_identity]=1
    job_name="${JOB_NAME_OVERRIDE:-drafter-train-${identity}-s${boundary}}"
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
        receipt_state="$(awk -F'|' -v job="$receipt_job" '$2 == job {print $4; exit}' <<<"$scheduler_jobs")"
        if [[ "$receipt_state" =~ ^(PENDING|RUNNING|COMPLETING|CONFIGURING|REQUEUED|COMPLETED)$ ]]; then
            printf '{"job_id":"%s","status":"receipt","tuple_identity":"%s"}\n' "$receipt_job" "$tuple_identity" >>"$RECEIPT"
            continue
        fi
        printf '{"job_id":"%s","status":"retrying-stale-receipt","tuple_identity":"%s","scheduler_state":"%s"}\n' "$receipt_job" "$tuple_identity" "$receipt_state" >>"$RECEIPT"
    fi
    existing="$(awk -F'|' -v tuple="$tuple_identity" '$3 == tuple && ($4 == "" || $4 ~ /^(PENDING|RUNNING|COMPLETING|CONFIGURING|REQUEUED|COMPLETED)$/) {print $2; exit}' <<<"$scheduler_jobs")"
    if [[ -n "$existing" ]]; then
        printf '{"job_id":"%s","status":"already-known","tuple_identity":"%s","max_steps":%s}\n' "$existing" "$tuple_identity" "$boundary" >>"$RECEIPT"
        continue
    fi
    if [[ -n "$SAVE_STEPS" ]]; then
        save_steps="$SAVE_STEPS"
    elif [[ "$SELF_REQUEUE" -eq 1 ]]; then
        save_steps="$DEFAULT_REQUEUE_SAVE_STEPS"
    else
        save_steps="$boundary"
    fi
    exports="ALL,SCHEDULER_JOBS_SNAPSHOT=,MANIFEST_PATH=${MANIFEST},EXPERIMENT_INDEX=${index},MAX_STEPS=${boundary},SAVE_STEPS=${save_steps},SELF_REQUEUE=${SELF_REQUEUE},LAUNCHER_ROOT=${LAUNCHER_ROOT}"
    if [[ -n "$LEGACY_ADOPTION_SOURCE_SHA" ]]; then
        exports+=",LEGACY_ADOPTION_SOURCE_SHA=${LEGACY_ADOPTION_SOURCE_SHA},LEGACY_ADOPTION_CHECKPOINT_STEP=${LEGACY_ADOPTION_CHECKPOINT_STEP}"
    fi
    requeue_args=()
    if [[ "$SELF_REQUEUE" -eq 1 ]]; then
        wandb_run_id="sd-${identity}"
        exports+=",MAX_REQUEUES=${MAX_REQUEUES},WANDB_RUN_ID=${wandb_run_id}"
        requeue_args=(--requeue "--signal=B:USR1@${REQUEUE_SIGNAL_LEAD}")
    fi
    mkdir -p "$output_root/logs"
    args=(--account="$account" --partition="$partition" -N4 --ntasks-per-node=1 --gpus-per-node=4 --segment=4 --time=03:55:00 --job-name="$job_name" --comment="$tuple_identity" --export="$exports" --output="${output_root}/logs/slurm-%j.out" --error="${output_root}/logs/slurm-%j.err" "${requeue_args[@]}")
    [[ -z "$DEPENDENCY" ]] || args+=(--dependency=afterok:"$DEPENDENCY")
    sbatch --test-only "${args[@]}" "$RUNNER"
    if [[ "$DRY_RUN" -eq 1 ]]; then
        printf '{"job_name":"%s","status":"test-only","tuple_identity":"%s","max_steps":%s}\n' "$job_name" "$tuple_identity" "$boundary" >>"$RECEIPT"
        continue
    fi
    submitted="$(sbatch --parsable "${args[@]}" "$RUNNER" || true)"
    job_id="${submitted%%;*}"
    if [[ -z "$job_id" ]]; then
        job_id="$(squeue -h -n "$job_name" -o "%A|%k" | awk -F'|' -v tuple="$tuple_identity" '$2 == tuple {print $1; exit}' || true)"
        [[ -n "$job_id" ]] || job_id="$(sacct -X -n --name "$job_name" -P --format=JobIDRaw,Comment,State | awk -F'|' -v tuple="$tuple_identity" '$2 == tuple && $3 ~ /^(PENDING|RUNNING|COMPLETING|CONFIGURING|REQUEUED|COMPLETED)$/ {print $1; exit}' || true)"
    fi
    [[ -n "$job_id" ]] || { echo "scheduler did not confirm training submission: $job_name" >&2; exit 1; }
    printf '{"job_id":"%s","status":"submitted","tuple_identity":"%s","max_steps":%s,"job_name":"%s"}\n' "$job_id" "$tuple_identity" "$boundary" "$job_name" >>"$RECEIPT"
done < <(render_waves)
