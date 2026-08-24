#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

#SBATCH --partition=cpu_datamover
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=96
#SBATCH --mem=0

set -euo pipefail

required=(
    REPO_ROOT SOURCE_COMMIT PTV2_SYMLINK_ROOT APPROVED_HAO_ROOT
    PTV2_README PTV2_MANIFEST_OUTPUT
)
for name in "${required[@]}"; do
    [[ -n "${!name:-}" ]] || { echo "missing required environment: $name" >&2; exit 2; }
done
[[ "${SLURM_JOB_ID:-}" =~ ^[0-9]+$ ]] || { echo "invalid Slurm job identity" >&2; exit 2; }
[[ "${SLURM_JOB_PARTITION:-}" == "cpu_datamover" ]] || {
    echo "source-manifest bootstrap requires cpu_datamover" >&2
    exit 2
}
[[ "${SLURM_CPUS_PER_TASK:-}" == "96" ]] || {
    echo "source-manifest bootstrap requires exactly 96 CPUs" >&2
    exit 2
}
[[ -z "${SLURM_JOB_GPUS:-}" && -z "${SLURM_GPUS_ON_NODE:-}" ]] || {
    echo "source-manifest bootstrap must not allocate GPUs" >&2
    exit 2
}
for path in "$REPO_ROOT" "$PTV2_SYMLINK_ROOT" "$APPROVED_HAO_ROOT" \
    "$PTV2_README" "$PTV2_MANIFEST_OUTPUT"; do
    [[ "$path" == /* ]] || { echo "bootstrap paths must be absolute" >&2; exit 2; }
done
[[ "$SOURCE_COMMIT" =~ ^[0-9a-f]{40}$ ]] || exit 2
[[ "$(git -C "$REPO_ROOT" rev-parse HEAD)" == "$SOURCE_COMMIT" ]] || {
    echo "source checkout commit mismatch" >&2
    exit 2
}
[[ -z "$(git -C "$REPO_ROOT" status --porcelain)" ]] || {
    echo "source checkout is dirty" >&2
    exit 2
}

export ARROW_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1

python3 "$REPO_ROOT/examples/dataset/bootstrap_ptv2_source_manifest.py" \
    --source-root "$PTV2_SYMLINK_ROOT" \
    --approved-hao-root "$APPROVED_HAO_ROOT" \
    --readme "$PTV2_README" \
    --output-root "$PTV2_MANIFEST_OUTPUT" \
    --source-commit "$SOURCE_COMMIT" \
    --workers "$SLURM_CPUS_PER_TASK"
