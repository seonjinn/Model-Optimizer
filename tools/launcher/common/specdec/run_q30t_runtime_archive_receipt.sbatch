#!/bin/bash -p
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

#SBATCH --account=coreai_dlalgo_llm
#SBATCH --partition=36x2-a01r
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=400G
#SBATCH --time=00:45:00
#SBATCH --export=NONE

set -euo pipefail
umask 077

[[ "$#" == 7 ]] || {
    echo "usage: $0 ARCHIVE ARCHIVE_SHA256 SOURCE_CHECKOUT SOURCE_COMMIT PRODUCER_SHA256 OUTPUT_RECEIPT OUTPUT_RECEIPT_PARENT" >&2
    exit 2
}
readonly archive="$1"
readonly archive_sha256="$2"
readonly source_checkout="$3"
readonly source_commit="$4"
readonly producer_sha256="$5"
readonly output_receipt="$6"
readonly output_receipt_parent="$7"

readonly expected_archive="/lustre/fsw/coreai_dlalgo_llm/users/sna/modelopt-qwen3-drafter-training/assets/q235-training-prereqs-vllm0271-v1/runtime/modelopt-vllm-0.27.1-py312-aarch64-symlinks.tar.zst"
readonly expected_archive_sha256="4a20aee61f290c48bed22a84b4a0ae0cbdc54e3e3910854d253188c8854f5dc9"
readonly slurm_job_id="${SLURM_JOB_ID:-}"
readonly slurm_nnodes="${SLURM_NNODES:-}"
readonly slurm_job_gpus="${SLURM_JOB_GPUS:-}"
readonly slurm_step_gpus="${SLURM_STEP_GPUS:-}"
readonly slurm_gpus_on_node="${SLURM_GPUS_ON_NODE:-}"
readonly slurm_gpus="${SLURM_GPUS:-}"
readonly slurm_gpus_per_node="${SLURM_GPUS_PER_NODE:-}"
readonly slurm_gpus_per_task="${SLURM_GPUS_PER_TASK:-}"
readonly slurm_tres_per_node="${SLURM_TRES_PER_NODE:-}"

[[ "${SLURM_EXPORT_ENV:-}" == NONE ]] || {
    echo "Q30 runtime archive receipt requires --export=NONE" >&2
    exit 2
}
[[ "$slurm_job_id" =~ ^[0-9]+$ && "$slurm_nnodes" == 1 ]] || {
    echo "Q30 runtime archive receipt requires exactly one Slurm node" >&2
    exit 2
}
[[ -z "$slurm_job_gpus" && -z "$slurm_step_gpus" \
    && ( -z "$slurm_gpus_on_node" || "$slurm_gpus_on_node" == 0 ) \
    && ( -z "$slurm_gpus" || "$slurm_gpus" == 0 ) \
    && ( -z "$slurm_gpus_per_node" || "$slurm_gpus_per_node" == 0 ) \
    && ( -z "$slurm_gpus_per_task" || "$slurm_gpus_per_task" == 0 ) \
    && "$slurm_tres_per_node" != *gpu* ]] || {
    echo "Q30 runtime archive receipt requires a CPU-only allocation" >&2
    exit 2
}
[[ "$archive" == "$expected_archive" && "$archive_sha256" == "$expected_archive_sha256" ]] || {
    echo "Q30 runtime archive identity is not approved" >&2
    exit 2
}
[[ "$source_checkout" == /home/* && -d "$source_checkout" && ! -L "$source_checkout" \
    && "$source_commit" =~ ^[0-9a-f]{40}$ && "$producer_sha256" =~ ^[0-9a-f]{64}$ ]] || {
    echo "Q30 runtime archive source identity is invalid" >&2
    exit 2
}
[[ "$output_receipt" == /* && "$output_receipt_parent" == /* \
    && "$(/usr/bin/dirname -- "$output_receipt")" == "$output_receipt_parent" ]] || {
    echo "Q30 runtime archive receipt destination is invalid" >&2
    exit 2
}

scratch_root="/raid/scratch/$USER/q30t-runtime-archive-$SLURM_JOB_ID"
readonly scratch_root
[[ ! -e "$scratch_root" && ! -L "$scratch_root" ]] || {
    echo "Q30 runtime archive scratch root is not fresh" >&2
    exit 2
}
/bin/mkdir -m 0700 -- "$scratch_root"
/bin/mkdir -p -- "$output_receipt_parent"

env -i PATH=/usr/bin:/bin PYTHONPATH="$source_checkout/tools/launcher" \
    /usr/bin/python3.12 -m common.specdec.q30t_runtime_archive_receipt produce \
    --archive "$archive" \
    --archive-sha256 "$archive_sha256" \
    --extraction-root "$scratch_root/runtime" \
    --source-commit "$source_commit" \
    --producer "$source_checkout/tools/launcher/common/specdec/q30t_runtime_archive_receipt.py" \
    --producer-sha256 "$producer_sha256" \
    --output "$output_receipt" \
    --job-id "$SLURM_JOB_ID"
