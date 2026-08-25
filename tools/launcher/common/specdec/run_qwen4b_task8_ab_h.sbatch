#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

#SBATCH --partition=cpu_datamover
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=96
#SBATCH --mem=0

set -euo pipefail

[[ "${SLURM_CPUS_PER_TASK:-}" == 96 ]] || exit 2
[[ "${SLURM_JOB_ID:-}" =~ ^[0-9]+$ ]] || exit 2
[[ "${SLURM_JOB_PARTITION:-cpu_datamover}" == cpu_datamover ]] || exit 2
[[ -z "${SLURM_JOB_GPUS:-}" && -z "${SLURM_GPUS_ON_NODE:-}" ]] || exit 2
required=(REPO_ROOT SOURCE_COMMIT A_TOKENIZED_RECEIPT A_TOKENIZED_RECEIPT_SHA256
    B_TOKENIZED_RECEIPT B_TOKENIZED_RECEIPT_SHA256 STUDY_ROOT IMAGE_PATH IMAGE_SHA256)
for name in "${required[@]}"; do [[ -n "${!name:-}" ]] || exit 2; done
[[ "$SOURCE_COMMIT" =~ ^[0-9a-f]{40}$ ]] || exit 2
for name in A_TOKENIZED_RECEIPT_SHA256 B_TOKENIZED_RECEIPT_SHA256 IMAGE_SHA256; do
    [[ "${!name}" =~ ^[0-9a-f]{64}$ ]] || exit 2
done
[[ -z "${EXPECTED_COMPLETION_RECEIPT_SHA256:-}" \
    || "$EXPECTED_COMPLETION_RECEIPT_SHA256" =~ ^[0-9a-f]{64}$ ]] || exit 2
durable_prefix="/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sna/modelopt-specdec/dataset-studies"
[[ "$STUDY_ROOT" == "$durable_prefix"/* ]] || exit 2
[[ "$A_TOKENIZED_RECEIPT" == "$STUDY_ROOT/a-repair-tokenized/TOKENIZED.json" ]] || exit 2
[[ "$B_TOKENIZED_RECEIPT" == "$STUDY_ROOT/b-balanced-tokenized/TOKENIZED.json" ]] || exit 2
for pair in "$A_TOKENIZED_RECEIPT:$A_TOKENIZED_RECEIPT_SHA256" \
    "$B_TOKENIZED_RECEIPT:$B_TOKENIZED_RECEIPT_SHA256" "$IMAGE_PATH:$IMAGE_SHA256"; do
    path="${pair%:*}" expected="${pair##*:}"
    [[ -f "$path" && ! -L "$path" && "$(sha256sum "$path" | cut -d' ' -f1)" == "$expected" ]] \
        || exit 2
done
[[ "$(git -C "$REPO_ROOT" rev-parse HEAD)" == "$SOURCE_COMMIT" ]] || exit 2
[[ -z "$(git -C "$REPO_ROOT" status --porcelain)" ]] || exit 2

scratch_parent="${SLURM_TMPDIR:-/raid/scratch}"
[[ "$scratch_parent" == /* && "$scratch_parent" != /home* && "$scratch_parent" != /lustre* ]] \
    || exit 2
scratch_root="$scratch_parent/qwen4b-task8-abh-$SLURM_JOB_ID"
[[ ! -e "$scratch_root" && ! -L "$scratch_root" ]] || exit 2
mkdir -p "$scratch_root"
trap 'find "$scratch_root" -depth -delete 2>/dev/null || true' EXIT

command=(python3 -P -m common.specdec.build_qwen4b_task8 --materialize-ab-h
    --a-tokenized-receipt "$A_TOKENIZED_RECEIPT"
    --a-tokenized-receipt-sha256 "$A_TOKENIZED_RECEIPT_SHA256"
    --b-tokenized-receipt "$B_TOKENIZED_RECEIPT"
    --b-tokenized-receipt-sha256 "$B_TOKENIZED_RECEIPT_SHA256"
    --study-root "$STUDY_ROOT" --workers 96)
if [[ -n "${EXPECTED_COMPLETION_RECEIPT_SHA256:-}" ]]; then
    command+=(--expected-completion-receipt-sha256 "$EXPECTED_COMPLETION_RECEIPT_SHA256")
fi
export ARROW_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 TOKENIZERS_PARALLELISM=false
srun --nodes=1 --ntasks=1 --cpus-per-task=96 --kill-on-bad-exit=1 \
    --chdir="$REPO_ROOT" --no-container-mount-home --container-image="$IMAGE_PATH" \
    --container-mounts="/lustre:/lustre,$REPO_ROOT:$REPO_ROOT:ro,$scratch_root:$scratch_root" \
    env PYTHONDONTWRITEBYTECODE=1 PYTHONSAFEPATH=1 \
    PYTHONPATH="$REPO_ROOT/tools/launcher:$REPO_ROOT/examples/dataset" "${command[@]}"

completion="$STUDY_ROOT/ptv2-study-exposures/COMPLETE.json"
[[ -f "$completion" && ! -L "$completion" ]] || exit 1
for arm in a-repair b-balanced h-historical-cyclic; do
    for tokens in 64000000 256000000; do
        bundle="$STUDY_ROOT/$arm-exposures/$arm-$tokens-assistant-tokens-v2"
        [[ -f "$bundle/SCIENTIFIC.json" && -f "$bundle/records.jsonl" \
            && -f "$bundle/EXECUTION.json" ]] || exit 1
    done
done
sha256sum "$completion"
