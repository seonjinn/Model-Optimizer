#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G

set -euo pipefail
required=(REPO_ROOT SOURCE_COMMIT SOURCE_INVENTORY TOKENIZER_RECEIPT TOKENIZER_RECEIPT_SHA256
    BASELINE_AUDIT HELD_OUT_UUIDS IMAGE_PATH DIAGNOSTIC_RECEIPT SHARD_INDEX)
for name in "${required[@]}"; do [[ -n "${!name:-}" ]] || exit 2; done
[[ "$(git -C "$REPO_ROOT" rev-parse HEAD)" == "$SOURCE_COMMIT" \
    && -z "$(git -C "$REPO_ROOT" status --porcelain)" ]] || exit 2
[[ ! -e "$DIAGNOSTIC_RECEIPT" && ! -L "$DIAGNOSTIC_RECEIPT" ]] || exit 2
scratch="${SLURM_TMPDIR:-/tmp}/q4-candidate-diag-${SLURM_JOB_ID}"
mkdir -p "$scratch"
trap 'find "$scratch" -depth -delete 2>/dev/null || true' EXIT
srun --nodes=1 --ntasks=1 --cpus-per-task=2 --kill-on-bad-exit=1 \
    --no-container-mount-home --container-image="$IMAGE_PATH" \
    --container-mounts="/lustre:/lustre,$REPO_ROOT:$REPO_ROOT,$scratch:$scratch" \
    python3 "$REPO_ROOT/examples/dataset/diagnose_qwen4b_candidate_shard.py" \
        --source-inventory "$SOURCE_INVENTORY" --tokenizer-receipt "$TOKENIZER_RECEIPT" \
        --tokenizer-receipt-sha256 "$TOKENIZER_RECEIPT_SHA256" \
        --baseline-audit "$BASELINE_AUDIT" --held-out-uuids "$HELD_OUT_UUIDS" \
        --shard-index "$SHARD_INDEX" --scratch-root "$scratch" --receipt "$DIAGNOSTIC_RECEIPT"
