#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=144
#SBATCH --mem=0

set -euo pipefail

required=(
    REPO_ROOT SOURCE_COMMIT SOURCE_MANIFEST SOURCE_MANIFEST_SHA256
    DURABLE_ROOT
)
for name in "${required[@]}"; do
    [[ -n "${!name:-}" ]] || { echo "missing required environment: $name" >&2; exit 2; }
done
[[ "${SLURM_JOB_ID:-}" =~ ^[0-9]+$ ]] || { echo "invalid Slurm job identity" >&2; exit 2; }
[[ "${SLURM_CPUS_PER_TASK:-}" == 144 ]] || {
    echo "PTV23 source staging requires 144 CPUs on a GB200 data-mover node" >&2
    exit 2
}
[[ -z "${SLURM_JOB_GPUS:-}" && -z "${SLURM_GPUS_ON_NODE:-}" ]] || {
    echo "PTV23 source staging must not allocate GPUs" >&2
    exit 2
}
[[ "$REPO_ROOT" == /* && "$SOURCE_MANIFEST" == /* && "$DURABLE_ROOT" == /* ]] || {
    echo "source staging paths must be absolute" >&2
    exit 2
}
[[ "$SOURCE_COMMIT" =~ ^[0-9a-f]{40}$ && "$SOURCE_MANIFEST_SHA256" =~ ^[0-9a-f]{64}$ ]] || {
    echo "source staging identities are malformed" >&2
    exit 2
}
[[ "$(git -C "$REPO_ROOT" rev-parse HEAD)" == "$SOURCE_COMMIT" ]] || {
    echo "source checkout commit mismatch" >&2
    exit 2
}
[[ -z "$(git -C "$REPO_ROOT" status --porcelain)" ]] || {
    echo "source checkout is dirty" >&2
    exit 2
}
[[ -f "$SOURCE_MANIFEST" ]] || { echo "source manifest is absent" >&2; exit 2; }
actual_manifest_sha256="$(PYTHONPATH="$REPO_ROOT/examples/dataset${PYTHONPATH:+:$PYTHONPATH}" python3 - "$SOURCE_MANIFEST" <<'PY'
import sys
from pathlib import Path

from stage_ptv23_sources import load_source_inventory

print(load_source_inventory(Path(sys.argv[1])).manifest_sha256)
PY
)"
[[ "$actual_manifest_sha256" == "$SOURCE_MANIFEST_SHA256" ]] || {
    echo "source manifest SHA-256 mismatch" >&2
    exit 2
}

job_local_parent="${SLURM_TMPDIR:-/tmp}"
[[ "$job_local_parent" == /* && "$job_local_parent" != /home* && "$job_local_parent" != /lustre* ]] || {
    echo "job-local scratch must not use home or Lustre" >&2
    exit 2
}
scratch_root="$job_local_parent/ptv23-source-stage-${SLURM_JOB_ID}"
[[ ! -e "$scratch_root" && ! -L "$scratch_root" ]] || {
    echo "job-local staging root already exists: $scratch_root" >&2
    exit 2
}
mkdir -p "$scratch_root" "$DURABLE_ROOT"
trap 'find "$scratch_root" -depth -delete 2>/dev/null || true' EXIT

command=(
    python3 "$REPO_ROOT/examples/dataset/stage_ptv23_sources.py"
    --manifest "$SOURCE_MANIFEST"
    --durable-root "$DURABLE_ROOT"
    --scratch-root "$scratch_root"
)
if [[ -n "${LOCAL_SOURCE_ROOT:-}" ]]; then
    [[ "$LOCAL_SOURCE_ROOT" == /* ]] || { echo "local source root must be absolute" >&2; exit 2; }
    command+=(--local-source-root "$LOCAL_SOURCE_ROOT")
fi
"${command[@]}"
