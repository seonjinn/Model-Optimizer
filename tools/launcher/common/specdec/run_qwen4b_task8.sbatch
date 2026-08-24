#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

#SBATCH --partition=cpu_datamover
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=96
#SBATCH --mem=0

set -euo pipefail

[[ "${SLURM_CPUS_PER_TASK:-}" == 96 ]] || {
    echo "Qwen4B Task8 requires exactly 96 CPUs" >&2
    exit 2
}
[[ "${SLURM_JOB_ID:-}" =~ ^[0-9]+$ ]] || { echo "invalid Slurm job identity" >&2; exit 2; }
[[ "${SLURM_JOB_PARTITION:-cpu_datamover}" == cpu_datamover ]] || exit 2
[[ -z "${SLURM_JOB_GPUS:-}" && -z "${SLURM_GPUS_ON_NODE:-}" ]] || exit 2

required=(REPO_ROOT SOURCE_COMMIT TASK9_SOURCE_COMMIT SOURCE_RECEIPT SOURCE_RECEIPT_SHA256
    SELECTION_RECEIPT SELECTION_RECEIPT_SHA256 RESPONSE_RECEIPT RESPONSE_RECEIPT_SHA256
    REJECTION_RECEIPT REJECTION_RECEIPT_SHA256 TOKENIZER_RECEIPT TOKENIZER_RECEIPT_SHA256
    ASSISTANT_LOSS_TARGET_SHA256 TRAINING_CONFIG_SHA256 PUBLICATION_ROOT IMAGE_PATH IMAGE_SHA256)
for name in "${required[@]}"; do
    [[ -n "${!name:-}" ]] || { echo "missing required environment: $name" >&2; exit 2; }
done
[[ "$SOURCE_COMMIT" =~ ^[0-9a-f]{40}$ ]] || exit 2
[[ "$TASK9_SOURCE_COMMIT" =~ ^[0-9a-f]{40}$ ]] || exit 2
for name in SOURCE_RECEIPT_SHA256 SELECTION_RECEIPT_SHA256 RESPONSE_RECEIPT_SHA256 \
    REJECTION_RECEIPT_SHA256 TOKENIZER_RECEIPT_SHA256 ASSISTANT_LOSS_TARGET_SHA256 \
    TRAINING_CONFIG_SHA256; do
    [[ "${!name}" =~ ^[0-9a-f]{64}$ ]] || exit 2
done
[[ "$IMAGE_SHA256" =~ ^[0-9a-f]{64}$ ]] || exit 2
durable_prefix="/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sna/modelopt-specdec/dataset-studies"
[[ "$PUBLICATION_ROOT" == "$durable_prefix"/* ]] || {
    echo "Task8 publication must use the durable dataset-study root" >&2
    exit 2
}
case "$PUBLICATION_ROOT" in
    *//*|*/./*|*/../*|*/.|*/..)
        echo "Task8 publication root must be canonical without traversal" >&2
        exit 2
        ;;
esac
for path in "$REPO_ROOT" "$SOURCE_RECEIPT" "$SELECTION_RECEIPT" "$RESPONSE_RECEIPT" \
    "$REJECTION_RECEIPT" "$TOKENIZER_RECEIPT" "$PUBLICATION_ROOT"; do
    [[ "$path" == /* ]] || exit 2
done
for path in "$SOURCE_RECEIPT" "$SELECTION_RECEIPT" "$RESPONSE_RECEIPT" \
    "$REJECTION_RECEIPT" "$TOKENIZER_RECEIPT"; do
    [[ -f "$path" && ! -L "$path" ]] || exit 2
done
[[ "$IMAGE_PATH" == /* && -f "$IMAGE_PATH" && ! -L "$IMAGE_PATH" \
    && "$(sha256sum "$IMAGE_PATH" | cut -d' ' -f1)" == "$IMAGE_SHA256" ]] || exit 2
[[ ! -e "$PUBLICATION_ROOT" && ! -L "$PUBLICATION_ROOT" ]] || exit 2
[[ "$(git -C "$REPO_ROOT" rev-parse HEAD)" == "$SOURCE_COMMIT" ]] || exit 2
[[ -z "$(git -C "$REPO_ROOT" status --porcelain)" ]] || exit 2
PYTHONDONTWRITEBYTECODE=1 PYTHONSAFEPATH=1 python3 -P \
    "$REPO_ROOT/tools/launcher/common/specdec/build_qwen4b_task8.py" \
    --validate-publication-root "$PUBLICATION_ROOT" "$durable_prefix"

scratch_parent="${SLURM_TMPDIR:-/tmp}"
[[ "$scratch_parent" == /* && "$scratch_parent" != /home* && "$scratch_parent" != /lustre* ]] || {
    echo "Task8 scratch must be node-local" >&2
    exit 2
}
scratch_root="$scratch_parent/qwen4b-task8-${SLURM_JOB_ID}"
[[ ! -e "$scratch_root" && ! -L "$scratch_root" ]] || exit 2
mkdir -p "$scratch_root"
trap 'find "$scratch_root" -depth -delete 2>/dev/null || true' EXIT

export ARROW_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 TOKENIZERS_PARALLELISM=false

srun --nodes=1 --ntasks=1 --cpus-per-task=96 --kill-on-bad-exit=1 \
    --chdir="$REPO_ROOT" \
    --no-container-mount-home --container-image="$IMAGE_PATH" \
    --container-mounts="/lustre:/lustre,$REPO_ROOT:$REPO_ROOT:ro,$scratch_root:$scratch_root" \
    env PYTHONDONTWRITEBYTECODE=1 PYTHONSAFEPATH=1 \
    PYTHONPATH="$REPO_ROOT/tools/launcher:$REPO_ROOT/examples/dataset" \
    python3 -P -m common.specdec.build_qwen4b_task8 \
        --source-receipt "$SOURCE_RECEIPT" \
        --source-receipt-sha256 "$SOURCE_RECEIPT_SHA256" \
        --selection-receipt "$SELECTION_RECEIPT" \
        --selection-receipt-sha256 "$SELECTION_RECEIPT_SHA256" \
        --response-receipt "$RESPONSE_RECEIPT" \
        --response-receipt-sha256 "$RESPONSE_RECEIPT_SHA256" \
        --rejection-receipt "$REJECTION_RECEIPT" \
        --rejection-receipt-sha256 "$REJECTION_RECEIPT_SHA256" \
        --tokenizer-receipt "$TOKENIZER_RECEIPT" \
        --tokenizer-receipt-sha256 "$TOKENIZER_RECEIPT_SHA256" \
        --assistant-loss-target-sha256 "$ASSISTANT_LOSS_TARGET_SHA256" \
        --training-config-sha256 "$TRAINING_CONFIG_SHA256" \
        --work-root "$scratch_root/work" \
        --materialization-root "$scratch_root/materialized" \
        --publication-root "$PUBLICATION_ROOT" \
        --task9-source-commit "$TASK9_SOURCE_COMMIT" \
        --source-commit "$SOURCE_COMMIT" --job-id "$SLURM_JOB_ID" --workers 96

[[ -f "$PUBLICATION_ROOT/PUBLICATION.json" && ! -L "$PUBLICATION_ROOT/PUBLICATION.json" ]] || exit 1
sha256sum "$PUBLICATION_ROOT/PUBLICATION.json"
