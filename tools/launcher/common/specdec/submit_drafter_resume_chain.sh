#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Submit train -> public acceptance -> next-train dependencies for one manifest.
set -euo pipefail

if [[ "${1:-}" == "--run-evaluation-container" ]]; then
    work_root="/raid/scratch/${SLURM_JOB_ID}/evaluation"
    mkdir -p "$work_root"
    tar --extract --file="$EVAL_RUNTIME_ARCHIVE" --directory="$work_root"
    old_venv="$(sed -nE "s/^[[:space:]]*(export[[:space:]]+)?VIRTUAL_ENV=[\\042\\047]?([^\\042\\047]+)[\\042\\047]?.*/\\2/p" "$work_root/bin/activate" | head -n 1)"
    [[ -n "$old_venv" ]] || { echo "runtime archive has no VIRTUAL_ENV" >&2; exit 1; }
    grep -IlZ "$old_venv" "$work_root/bin"/* "$work_root/pyvenv.cfg" 2>/dev/null | xargs -0 -r sed -i "s|$old_venv|$work_root|g"
    export VIRTUAL_ENV="$work_root" PATH="$work_root/bin:$PATH"
    export HF_HOME="$work_root/hf" HF_HUB_CACHE="$work_root/hf/hub" HF_DATASETS_CACHE="$work_root/hf/datasets"
    export XDG_CACHE_HOME="$work_root/xdg" SQLITE_TMPDIR="$work_root/sqlite" TMPDIR="$work_root/tmp"
    # shellcheck disable=SC1090
    source "$EVAL_EVALUATOR_ENV"
    export EVAL_CONFIG_PATH="$EVAL_EVALUATOR_CONFIG" EVAL_OUTPUT_ROOT="$EVAL_RUN_OUTPUT"
    export DRAFT_MODEL="$EVAL_TRAIN_EXPORT" SPEC_METHOD="$EVAL_METHOD" DFLASH_BLOCK_SIZE="$EVAL_BLOCK_SIZE" NUM_SPEC_TOKENS="$EVAL_NUM_SPEC_TOKENS"
    "$EVAL_EVALUATOR_SCRIPT"
    test -s "${EVAL_RUN_OUTPUT}/acceptance.csv"
    exit 0
fi
if [[ "${1:-}" == "--run-evaluation" ]]; then
    exec srun --nodes=1 --ntasks=1 --container-image="$EVAL_IMAGE" --container-mounts="/home:/home,/lustre:/lustre,/raid/scratch:/raid/scratch" bash "$0" --run-evaluation-container
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WAVE_SUBMITTER="${SCRIPT_DIR}/submit_drafter_training_wave.sh"
MANIFEST=""
RECEIPT_ROOT=""
EVALUATOR_SCRIPT=""
EVALUATOR_ENV=""
EVALUATOR_CONFIG=""
EVAL_OUTPUT_ROOT=""
EXPORT_ROOT=""
IMAGE=""
RUNTIME_ARCHIVE=""
ACCOUNT="nemotron_n3_post"
PARTITION="batch"
DRY_RUN=0
readonly PUBLIC_SUBSETS="HumanEval,math_reasoning,qa,question,rag,summarization,tool_call,translation,writing"

usage() {
    echo "usage: $0 --manifest /home/... --receipt-root /lustre/... --evaluator-script /home/... --evaluator-env /home/... --evaluator-config /home/... --eval-output-root /lustre/... --export-root /lustre/... --image /lustre/... --runtime-archive /lustre/... [--dry-run]" >&2
    exit 2
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --manifest) MANIFEST="$2"; shift 2 ;;
        --receipt-root) RECEIPT_ROOT="$2"; shift 2 ;;
        --evaluator-script) EVALUATOR_SCRIPT="$2"; shift 2 ;;
        --evaluator-env) EVALUATOR_ENV="$2"; shift 2 ;;
        --evaluator-config) EVALUATOR_CONFIG="$2"; shift 2 ;;
        --eval-output-root) EVAL_OUTPUT_ROOT="$2"; shift 2 ;;
        --export-root) EXPORT_ROOT="$2"; shift 2 ;;
        --image) IMAGE="$2"; shift 2 ;;
        --runtime-archive) RUNTIME_ARCHIVE="$2"; shift 2 ;;
        --account) ACCOUNT="$2"; shift 2 ;;
        --partition) PARTITION="$2"; shift 2 ;;
        --dry-run) DRY_RUN=1; shift ;;
        *) usage ;;
    esac
done
[[ "$MANIFEST" == /home/* && -f "$MANIFEST" && "$EVALUATOR_SCRIPT" == /home/* && "$EVALUATOR_ENV" == /home/* && "$EVALUATOR_CONFIG" == /home/* ]] || usage
[[ "$RECEIPT_ROOT" == /lustre/* && "$EVAL_OUTPUT_ROOT" == /lustre/* && "$EXPORT_ROOT" == /lustre/* && "$IMAGE" == /lustre/* && "$RUNTIME_ARCHIVE" == /lustre/* ]] || usage
[[ -f "$IMAGE" && -f "$RUNTIME_ARCHIVE" ]] || { echo "missing pinned evaluator artifact" >&2; exit 2; }
mkdir -p "$RECEIPT_ROOT"

boundaries() {
    PYTHONPATH="$(cd "${SCRIPT_DIR}/../.." && pwd)${PYTHONPATH:+:$PYTHONPATH}" python3 - "$MANIFEST" <<'PY'
import sys
from pathlib import Path
from common.specdec.drafter_job_manifest import load_manifest

for index, experiment in enumerate(load_manifest(Path(sys.argv[1]))):
    for boundary in experiment.cumulative_max_steps:
        print(f"{index}\t{experiment.experiment_id}\t{boundary}\t{experiment.method}\t{experiment.block_size}\t{experiment.num_speculative_tokens}\t{experiment.paths.output_root}")
PY
}

receipt_job_id() {
    python3 - "$1" <<'PY'
import json
import sys
from pathlib import Path

records = [json.loads(line) for line in Path(sys.argv[1]).read_text().splitlines() if line]
print(records[-1].get("job_id", ""))
PY
}

submit_evaluation() {
    local index="$1" identity="$2" boundary="$3" train_id="$4" method="$5" block_size="$6" num_tokens="$7" export_root="$8"
    local name="drafter-eval-${identity}-s${boundary}" train_export="${export_root}/exported-checkpoint-${boundary}"
    local run_output="${EVAL_OUTPUT_ROOT}/${identity}/step-${boundary}" receipt="${RECEIPT_ROOT}/evaluation-${identity}-s${boundary}.json"
    local exports="ALL,EVAL_IMAGE=${IMAGE},EVAL_RUNTIME_ARCHIVE=${RUNTIME_ARCHIVE},EVAL_EVALUATOR_ENV=${EVALUATOR_ENV},EVAL_EVALUATOR_CONFIG=${EVALUATOR_CONFIG},EVAL_EVALUATOR_SCRIPT=${EVALUATOR_SCRIPT},EVAL_RUN_OUTPUT=${run_output},EVAL_TRAIN_EXPORT=${train_export},EVAL_METHOD=${method},EVAL_BLOCK_SIZE=${block_size},EVAL_NUM_SPEC_TOKENS=${num_tokens},EXPERIMENT_IDENTITY=${identity}"
    local args=(--account="$ACCOUNT" --partition="$PARTITION" --nodes=1 --ntasks-per-node=1 --gpus-per-node=4 --segment=1 --time=03:55:00 --job-name="$name" --comment="${identity}:${boundary}" --dependency=afterok:"$train_id" --output="${RECEIPT_ROOT}/%x-%j.out" --export="$exports")
    existing="$(squeue -h -n "$name" -o "%A" | head -n 1 || true)"
    [[ -n "$existing" ]] || existing="$(sacct -X -n --name "$name" --format=JobIDRaw,State | awk 'NF {print $1; exit}' || true)"
    if [[ -n "$existing" ]]; then
        printf '{"job_id":"%s","status":"already-queued","max_steps":%s}\n' "$existing" "$boundary" >"$receipt"
        printf '%s\n' "$existing"
        return
    fi
    sbatch --test-only "${args[@]}" "$0" --run-evaluation
    if [[ "$DRY_RUN" -eq 1 ]]; then
        printf '{"status":"test-only","max_steps":%s,"subsets":"%s"}\n' "$boundary" "$PUBLIC_SUBSETS" >"$receipt"
        printf '\n'
        return
    fi
    submitted="$(sbatch --parsable "${args[@]}" "$0" --run-evaluation || true)"
    job_id="${submitted%%;*}"
    if [[ -z "$job_id" ]]; then
        job_id="$(squeue -h -n "$name" -o "%A" | head -n 1 || true)"
        [[ -n "$job_id" ]] || job_id="$(sacct -X -n --name "$name" --format=JobIDRaw,State | awk 'NF {print $1; exit}' || true)"
    fi
    [[ -n "$job_id" ]] || { echo "scheduler did not confirm evaluator submission" >&2; exit 1; }
    printf '{"job_id":"%s","status":"submitted","max_steps":%s,"subsets":"%s"}\n' "$job_id" "$boundary" "$PUBLIC_SUBSETS" >"$receipt"
    printf '%s\n' "$job_id"
}

declare -A previous_evaluations=()
while IFS=$'\t' read -r index identity boundary method block_size num_tokens output_root; do
    [[ -n "$boundary" ]] || continue
    train_receipt="${RECEIPT_ROOT}/training-${identity}-s${boundary}.jsonl"
    train_args=(--manifest "$MANIFEST" --receipt "$train_receipt" --experiment-index "$index" --max-steps "$boundary")
    [[ -z "${previous_evaluations[$identity]:-}" ]] || train_args+=(--dependency "${previous_evaluations[$identity]}")
    [[ "$DRY_RUN" -eq 0 ]] || train_args+=(--dry-run)
    "$WAVE_SUBMITTER" "${train_args[@]}"
    train_id="$(receipt_job_id "$train_receipt")"
    if [[ "$DRY_RUN" -eq 1 ]]; then
        continue
    fi
    [[ -n "$train_id" ]] || { echo "training receipt has no scheduler-confirmed job ID" >&2; exit 1; }
    previous_evaluations[$identity]="$(submit_evaluation "$index" "$identity" "$boundary" "$train_id" "$method" "$block_size" "$num_tokens" "$output_root")"
    [[ -n "${previous_evaluations[$identity]}" ]] || { echo "evaluation receipt has no scheduler-confirmed job ID" >&2; exit 1; }
done < <(boundaries)
