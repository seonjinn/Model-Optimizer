#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Submit train -> public acceptance -> next-train dependencies for one manifest.
set -euo pipefail

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

for experiment in load_manifest(Path(sys.argv[1])):
    for boundary in experiment.cumulative_max_steps:
        print(boundary)
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
    local boundary="$1"
    local train_id="$2"
    local name="drafter-eval-${boundary}"
    local train_export="${EXPORT_ROOT}/exported-checkpoint-${boundary}"
    local run_output="${EVAL_OUTPUT_ROOT}/step-${boundary}"
    local receipt="${RECEIPT_ROOT}/evaluation-${boundary}.json"
    local local_root="/raid/scratch/\${SLURM_JOB_ID}/evaluation"
    local command
    command="set -euo pipefail; mkdir -p ${local_root}; tar --extract --file=${RUNTIME_ARCHIVE} --directory=${local_root}; source ${local_root}/bin/activate; export HF_HOME=${local_root}/hf HF_HUB_CACHE=${local_root}/hf/hub HF_DATASETS_CACHE=${local_root}/hf/datasets XDG_CACHE_HOME=${local_root}/xdg SQLITE_TMPDIR=${local_root}/sqlite TMPDIR=${local_root}/tmp; source ${EVALUATOR_ENV}; export EVAL_CONFIG_PATH=${EVALUATOR_CONFIG} EVAL_OUTPUT_ROOT=${run_output} DRAFT_MODEL=${train_export} PUBLIC_SUBSETS=${PUBLIC_SUBSETS}; ${EVALUATOR_SCRIPT}; test -s ${run_output}/acceptance.csv"
    local args=(--account="$ACCOUNT" --partition="$PARTITION" --nodes=1 --ntasks-per-node=1 --gpus-per-node=4 --segment=1 --time=03:55:00 --job-name="$name" --dependency=afterok:"$train_id" --output="${RECEIPT_ROOT}/%x-%j.out")
    existing="$(squeue -h -n "$name" -o "%A" | head -n 1 || true)"
    if [[ -n "$existing" ]]; then
        printf '{"job_id":"%s","status":"already-queued","max_steps":%s}\n' "$existing" "$boundary" >"$receipt"
        printf '%s\n' "$existing"
        return
    fi
    sbatch --test-only "${args[@]}" --container-image="$IMAGE" --wrap "$command"
    if [[ "$DRY_RUN" -eq 1 ]]; then
        printf '{"status":"test-only","max_steps":%s,"subsets":"%s"}\n' "$boundary" "$PUBLIC_SUBSETS" >"$receipt"
        printf '\n'
        return
    fi
    submitted="$(sbatch --parsable "${args[@]}" --container-image="$IMAGE" --wrap "$command" || true)"
    job_id="${submitted%%;*}"
    if [[ -z "$job_id" ]]; then
        job_id="$(squeue -h -n "$name" -o "%A" | head -n 1 || true)"
    fi
    [[ -n "$job_id" ]] || { echo "scheduler did not confirm evaluator submission" >&2; exit 1; }
    printf '{"job_id":"%s","status":"submitted","max_steps":%s,"subsets":"%s"}\n' "$job_id" "$boundary" "$PUBLIC_SUBSETS" >"$receipt"
    printf '%s\n' "$job_id"
}

previous_evaluation=""
while read -r boundary; do
    [[ -n "$boundary" ]] || continue
    train_receipt="${RECEIPT_ROOT}/training-${boundary}.jsonl"
    train_args=(--manifest "$MANIFEST" --receipt "$train_receipt" --max-steps "$boundary")
    [[ -z "$previous_evaluation" ]] || train_args+=(--dependency "$previous_evaluation")
    [[ "$DRY_RUN" -eq 0 ]] || train_args+=(--dry-run)
    "$WAVE_SUBMITTER" "${train_args[@]}"
    train_id="$(receipt_job_id "$train_receipt")"
    if [[ "$DRY_RUN" -eq 1 ]]; then
        continue
    fi
    [[ -n "$train_id" ]] || { echo "training receipt has no scheduler-confirmed job ID" >&2; exit 1; }
    previous_evaluation="$(submit_evaluation "$boundary" "$train_id")"
    [[ -n "$previous_evaluation" ]] || { echo "evaluation receipt has no scheduler-confirmed job ID" >&2; exit 1; }
done < <(boundaries)
