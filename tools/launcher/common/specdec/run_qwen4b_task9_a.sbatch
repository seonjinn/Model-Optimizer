#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=96
#SBATCH --mem=0

set -euo pipefail

required=(REPO_ROOT SOURCE_COMMIT SOURCE_INVENTORY BASELINE_AUDIT HELD_OUT_UUIDS POLICY
    TASK5_MANIFEST TASK5_MANIFEST_SHA256 OUTPUT_ROOT IMAGE_PATH IMAGE_SHA256)
for name in "${required[@]}"; do
    [[ -n "${!name:-}" ]] || { echo "missing required environment: $name" >&2; exit 2; }
done
[[ "${SLURM_JOB_ID:-}" =~ ^[0-9]+$ && "${SLURM_CPUS_PER_TASK:-}" == 96 ]] || exit 2
[[ -z "${SLURM_JOB_GPUS:-}" && -z "${SLURM_GPUS_ON_NODE:-}" ]] || exit 2
[[ "$(git -C "$REPO_ROOT" rev-parse HEAD)" == "$SOURCE_COMMIT" ]] || exit 2
[[ -z "$(git -C "$REPO_ROOT" status --porcelain)" ]] || exit 2
for path in "$SOURCE_INVENTORY" "$BASELINE_AUDIT" "$HELD_OUT_UUIDS" "$POLICY" \
    "$TASK5_MANIFEST" "$IMAGE_PATH"; do
    [[ "$path" == /* && -f "$path" && ! -L "$path" ]] || exit 2
done
[[ "$TASK5_MANIFEST_SHA256" =~ ^[0-9a-f]{64}$ \
    && "$(sha256sum "$TASK5_MANIFEST" | cut -d' ' -f1)" == "$TASK5_MANIFEST_SHA256" ]] || exit 2
[[ "$IMAGE_SHA256" =~ ^[0-9a-f]{64}$ \
    && "$(sha256sum "$IMAGE_PATH" | cut -d' ' -f1)" == "$IMAGE_SHA256" ]] || exit 2
[[ "$OUTPUT_ROOT" == /* && ! -e "$OUTPUT_ROOT" && ! -L "$OUTPUT_ROOT" ]] || exit 2

scratch_parent="${SLURM_TMPDIR:-/tmp}"
[[ "$scratch_parent" == /* && "$scratch_parent" != /home* && "$scratch_parent" != /lustre* ]] || exit 2
scratch="$scratch_parent/qwen4b-task9-a-${SLURM_JOB_ID}"
[[ ! -e "$scratch" && ! -L "$scratch" ]] || exit 2
mkdir -p "$scratch"
trap 'find "$scratch" -depth -delete 2>/dev/null || true' EXIT

export ARROW_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1 PYTHONSAFEPATH=1
export PYTHONPATH="$REPO_ROOT/examples/dataset"

srun --nodes=1 --ntasks=1 --cpus-per-task=96 --kill-on-bad-exit=1 \
    --no-container-mount-home --container-image="$IMAGE_PATH" \
    --container-mounts="/lustre:/lustre,$REPO_ROOT:$REPO_ROOT,$scratch:$scratch" \
    python3 -P "$REPO_ROOT/examples/dataset/build_qwen4b_a_repair.py" \
        --source-inventory "$SOURCE_INVENTORY" --baseline-audit "$BASELINE_AUDIT" \
        --held-out-uuids "$HELD_OUT_UUIDS" --policy "$POLICY" \
        --task5-manifest "$TASK5_MANIFEST" \
        --task5-manifest-sha256 "$TASK5_MANIFEST_SHA256" \
        --work-root "$scratch/work" --selection-receipt-root "$OUTPUT_ROOT" \
        --workers 96
[[ -f "$OUTPUT_ROOT/SELECTION_RECEIPT.json" && ! -L "$OUTPUT_ROOT/SELECTION_RECEIPT.json" ]] || exit 1
sha256sum "$OUTPUT_ROOT/SELECTION_RECEIPT.json"
