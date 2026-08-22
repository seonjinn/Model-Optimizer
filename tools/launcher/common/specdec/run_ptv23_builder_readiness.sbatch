#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --segment=1
#SBATCH --time=00:30:00

set -euo pipefail

for name in SOURCE_PATH SOURCE_SHA IMAGE_PATH IMAGE_SHA256 AUDIT_RECEIPT AUDIT_RECEIPT_SHA256 PTV2_ROOT PTV2_REVISION PTV3_PROMPT_ROOT PTV3_SOURCE_MANIFEST OUTPUT_RECEIPT; do
    [[ -n "${!name:-}" ]] || { echo "missing required environment: $name" >&2; exit 2; }
done
[[ "${SLURM_NNODES:-}" == 1 && "$SOURCE_PATH" == /home/* && "$OUTPUT_RECEIPT" == /lustre/* ]] || {
    echo "invalid readiness topology or paths" >&2
    exit 2
}
[[ "$(git -C "$SOURCE_PATH" rev-parse HEAD)" == "$SOURCE_SHA" && -z "$(git -C "$SOURCE_PATH" status --porcelain)" ]] || {
    echo "source checkout identity mismatch" >&2
    exit 2
}
[[ -f "$IMAGE_PATH" && "$(sha256sum "$IMAGE_PATH" | cut -d' ' -f1)" == "$IMAGE_SHA256" ]] || {
    echo "image identity mismatch" >&2
    exit 2
}
[[ -f "$AUDIT_RECEIPT" && -d "$PTV2_ROOT" && -d "$PTV3_PROMPT_ROOT" && -f "$PTV3_SOURCE_MANIFEST" && ! -e "$OUTPUT_RECEIPT" ]] || {
    echo "readiness input missing or immutable output already exists" >&2
    exit 2
}

srun --nodes=1 --ntasks=1 --gpus=4 --cpus-per-task=8 --mem=32G --kill-on-bad-exit=1 \
    --no-container-mount-home \
    --container-image="$IMAGE_PATH" \
    --container-mounts="$SOURCE_PATH:$SOURCE_PATH,$AUDIT_RECEIPT:$AUDIT_RECEIPT,$PTV2_ROOT:$PTV2_ROOT,$PTV3_PROMPT_ROOT:$PTV3_PROMPT_ROOT,$PTV3_SOURCE_MANIFEST:$PTV3_SOURCE_MANIFEST,$(dirname "$OUTPUT_RECEIPT"):$(dirname "$OUTPUT_RECEIPT")" \
    python3 "$SOURCE_PATH/examples/dataset/ptv23_builder_readiness.py" \
        --audit-receipt "$AUDIT_RECEIPT" \
        --audit-receipt-sha256 "$AUDIT_RECEIPT_SHA256" \
        --ptv2-root "$PTV2_ROOT" \
        --ptv2-revision "$PTV2_REVISION" \
        --ptv3-prompt-root "$PTV3_PROMPT_ROOT" \
        --ptv3-source-manifest "$PTV3_SOURCE_MANIFEST" \
        --output-receipt "$OUTPUT_RECEIPT"

echo "builder readiness receipt complete: $OUTPUT_RECEIPT"
