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
    REPO_ROOT SOURCE_COMMIT TASK9_B_VIEW B_CANARY_OUTPUT
    B_CANARY_BUILD_RECEIPT B_CANARY_SEED
)
for name in "${required[@]}"; do
    [[ -n "${!name:-}" ]] || { echo "missing required environment: $name" >&2; exit 2; }
done

[[ "${SLURM_JOB_ID:-}" =~ ^[0-9]+$ ]] || { echo "invalid Slurm job identity" >&2; exit 2; }
[[ "${SLURM_JOB_PARTITION:-}" == "cpu_datamover" ]] || {
    echo "B builder requires the cpu_datamover partition" >&2
    exit 2
}
[[ "${SLURM_CPUS_PER_TASK:-}" == "96" ]] || {
    echo "B builder requires exactly 96 allocated CPUs" >&2
    exit 2
}
[[ -z "${SLURM_JOB_GPUS:-}" && -z "${SLURM_GPUS_ON_NODE:-}" ]] || {
    echo "B builder must not allocate GPUs" >&2
    exit 2
}
[[ "$REPO_ROOT" == /* && "$TASK9_B_VIEW" == /* && "$B_CANARY_OUTPUT" == /* ]] || {
    echo "B builder paths must be absolute" >&2
    exit 2
}
[[ "$B_CANARY_BUILD_RECEIPT" == /* && "$SOURCE_COMMIT" =~ ^[0-9a-f]{40}$ ]] || {
    echo "B builder receipt or source identity is invalid" >&2
    exit 2
}
[[ "$B_CANARY_SEED" =~ ^[0-9]+$ ]] || { echo "B builder seed must be non-negative" >&2; exit 2; }
[[ "$(git -C "$REPO_ROOT" rev-parse HEAD)" == "$SOURCE_COMMIT" ]] || {
    echo "source checkout commit mismatch" >&2
    exit 2
}
[[ -z "$(git -C "$REPO_ROOT" status --porcelain)" ]] || {
    echo "source checkout is dirty" >&2
    exit 2
}
[[ -f "$TASK9_B_VIEW" ]] || { echo "Task9 B projection is absent" >&2; exit 2; }

job_local_parent="${SLURM_TMPDIR:-/tmp}"
[[ "$job_local_parent" == /* && "$job_local_parent" != /home* && "$job_local_parent" != /lustre* ]] || {
    echo "B builder scratch must use job-local storage" >&2
    exit 2
}
scratch_root="$job_local_parent/qwen4b-b-builder-${SLURM_JOB_ID}"
[[ ! -e "$scratch_root" && ! -L "$scratch_root" ]] || {
    echo "B builder scratch root already exists: $scratch_root" >&2
    exit 2
}

export ARROW_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1

launcher_root="$REPO_ROOT/tools/launcher"
PYTHONPATH="$launcher_root${PYTHONPATH:+:$PYTHONPATH}" python3 \
    "$launcher_root/common/specdec/build_qwen4b_b_canary.py" \
    --task9-view "$TASK9_B_VIEW" \
    --output "$B_CANARY_OUTPUT" \
    --receipt "$B_CANARY_BUILD_RECEIPT" \
    --scratch-root "$scratch_root" \
    --seed "$B_CANARY_SEED" \
    --source-commit "$SOURCE_COMMIT" \
    --workers "$SLURM_CPUS_PER_TASK"
