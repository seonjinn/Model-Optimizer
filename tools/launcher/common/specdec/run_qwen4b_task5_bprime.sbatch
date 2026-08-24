#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=96
#SBATCH --mem=0

set -euo pipefail

required=(
    REPO_ROOT SOURCE_COMMIT SOURCE_INVENTORY TOKENIZER_RECEIPT
    TOKENIZER_RECEIPT_SHA256 BASELINE_AUDIT BASELINE_AUDIT_SHA256
    HELD_OUT_UUIDS HELD_OUT_FILE_SHA256 POLICY POLICY_FILE_SHA256 OUTPUT_DIR
    DIAGNOSTIC_RECEIPT IMAGE_PATH IMAGE_SHA256
)
for name in "${required[@]}"; do
    [[ -n "${!name:-}" ]] || { echo "missing required environment: $name" >&2; exit 2; }
done
[[ "${SLURM_CPUS_PER_TASK:-}" == 96 && "${SLURM_JOB_ID:-}" =~ ^[0-9]+$ ]] || {
    echo "Task5 B-prime requires one 96-CPU OCI data-mover allocation" >&2
    exit 2
}
[[ -z "${SLURM_JOB_GPUS:-}" && -z "${SLURM_GPUS_ON_NODE:-}" ]] || {
    echo "Task5 B-prime must not allocate GPUs" >&2
    exit 2
}
[[ "$(git -C "$REPO_ROOT" rev-parse HEAD)" == "$SOURCE_COMMIT" ]] || exit 2
[[ -z "$(git -C "$REPO_ROOT" status --porcelain)" ]] || exit 2
for path in "$SOURCE_INVENTORY" "$TOKENIZER_RECEIPT" "$BASELINE_AUDIT" \
    "$HELD_OUT_UUIDS" "$POLICY" "$IMAGE_PATH"; do
    [[ "$path" == /* && -f "$path" && ! -L "$path" ]] || exit 2
done
[[ "$IMAGE_SHA256" =~ ^[0-9a-f]{64}$ \
    && "$(sha256sum "$IMAGE_PATH" | cut -d' ' -f1)" == "$IMAGE_SHA256" ]] || exit 2
[[ "$OUTPUT_DIR" == /* && ! -e "$OUTPUT_DIR" && ! -L "$OUTPUT_DIR" ]] || exit 2
[[ "$DIAGNOSTIC_RECEIPT" == /* && ! -e "$DIAGNOSTIC_RECEIPT" \
    && ! -L "$DIAGNOSTIC_RECEIPT" ]] || exit 2
for pair in "$TOKENIZER_RECEIPT:$TOKENIZER_RECEIPT_SHA256" \
    "$BASELINE_AUDIT:$BASELINE_AUDIT_SHA256" "$HELD_OUT_UUIDS:$HELD_OUT_FILE_SHA256" \
    "$POLICY:$POLICY_FILE_SHA256"; do
    path="${pair%:*}" expected="${pair##*:}"
    [[ "$expected" =~ ^[0-9a-f]{64}$ && "$(sha256sum "$path" | cut -d' ' -f1)" == "$expected" ]] || exit 2
done

job_local_parent="${SLURM_TMPDIR:-/tmp}"
[[ "$job_local_parent" == /* && "$job_local_parent" != /home* && "$job_local_parent" != /lustre* ]] || exit 2
scratch="$job_local_parent/qwen4b-task5-bprime-${SLURM_JOB_ID}"
[[ ! -e "$scratch" && ! -L "$scratch" ]] || exit 2
mkdir -p "$scratch"
trap 'find "$scratch" -depth -delete 2>/dev/null || true' EXIT
export ARROW_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
    NUMEXPR_NUM_THREADS=1

srun --nodes=1 --ntasks=1 --cpus-per-task=96 --kill-on-bad-exit=1 \
    --no-container-mount-home --container-image="$IMAGE_PATH" \
    --container-mounts="/lustre:/lustre,$REPO_ROOT:$REPO_ROOT,$scratch:$scratch" \
    python3 "$REPO_ROOT/examples/dataset/build_qwen4b_bprime.py" \
        --source-inventory "$SOURCE_INVENTORY" \
        --tokenizer-receipt "$TOKENIZER_RECEIPT" \
        --tokenizer-receipt-sha256 "$TOKENIZER_RECEIPT_SHA256" \
        --baseline-audit "$BASELINE_AUDIT" \
        --held-out-uuids "$HELD_OUT_UUIDS" \
        --policy "$POLICY" \
        --storage-dir "$scratch" \
        --output-dir "$OUTPUT_DIR" \
        --diagnostic-receipt "$DIAGNOSTIC_RECEIPT" \
        --workers "$SLURM_CPUS_PER_TASK"
[[ -f "$OUTPUT_DIR/SELECTION_MANIFEST.json" && ! -L "$OUTPUT_DIR/SELECTION_MANIFEST.json" ]] || exit 1
[[ -f "$OUTPUT_DIR/EXECUTION_RECEIPT.json" && ! -L "$OUTPUT_DIR/EXECUTION_RECEIPT.json" ]] || exit 1
sha256sum "$OUTPUT_DIR/SELECTION_MANIFEST.json"
