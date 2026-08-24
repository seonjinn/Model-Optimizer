#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=96
#SBATCH --mem=0

set -euo pipefail

required=(
    REPO_ROOT SOURCE_COMMIT SOURCE_INVENTORY SOURCE_INVENTORY_SHA256
    POLICY POLICY_FILE_SHA256 HELD_OUT_UUIDS HELD_OUT_FILE_SHA256 RECEIPT_ROOT
)
for name in "${required[@]}"; do
    [[ -n "${!name:-}" ]] || { echo "missing required environment: $name" >&2; exit 2; }
done
[[ "${SLURM_JOB_ID:-}" =~ ^[0-9]+$ ]] || { echo "invalid Slurm job identity" >&2; exit 2; }
[[ "${SLURM_CPUS_PER_TASK:-}" == 96 ]] || {
    echo "Task9 B selection requires 96 CPUs on the OCI data-mover node" >&2
    exit 2
}
[[ -z "${SLURM_JOB_GPUS:-}" && -z "${SLURM_GPUS_ON_NODE:-}" ]] || {
    echo "Task9 B selection must not allocate GPUs" >&2
    exit 2
}
for path in "$REPO_ROOT" "$SOURCE_INVENTORY" "$POLICY" "$HELD_OUT_UUIDS" "$RECEIPT_ROOT"; do
    [[ "$path" == /* ]] || { echo "Task9 paths must be absolute" >&2; exit 2; }
done
[[ "$SOURCE_COMMIT" =~ ^[0-9a-f]{40}$ ]] || { echo "invalid source commit" >&2; exit 2; }
for digest in "$SOURCE_INVENTORY_SHA256" "$POLICY_FILE_SHA256" "$HELD_OUT_FILE_SHA256"; do
    [[ "$digest" =~ ^[0-9a-f]{64}$ ]] || { echo "invalid Task9 input digest" >&2; exit 2; }
done
[[ "$(git -C "$REPO_ROOT" rev-parse HEAD)" == "$SOURCE_COMMIT" ]] || {
    echo "source checkout commit mismatch" >&2
    exit 2
}
[[ -z "$(git -C "$REPO_ROOT" status --porcelain)" ]] || {
    echo "source checkout is dirty" >&2
    exit 2
}
[[ -f "$SOURCE_INVENTORY" && ! -L "$SOURCE_INVENTORY" ]] || {
    echo "source inventory must be a no-follow regular file" >&2
    exit 2
}
[[ -f "$POLICY" && ! -L "$POLICY" && -f "$HELD_OUT_UUIDS" && ! -L "$HELD_OUT_UUIDS" ]] || {
    echo "Task9 policy and held-out set must be no-follow regular files" >&2
    exit 2
}
[[ ! -e "$RECEIPT_ROOT" && ! -L "$RECEIPT_ROOT" ]] || {
    echo "immutable Task9 receipt destination already exists" >&2
    exit 2
}
export ARROW_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
    NUMEXPR_NUM_THREADS=1

readarray -t identities < <(
    PYTHONPATH="$REPO_ROOT/examples/dataset${PYTHONPATH:+:$PYTHONPATH}" python3 - \
        "$SOURCE_INVENTORY" "$POLICY" "$HELD_OUT_UUIDS" <<'PY'
from hashlib import sha256
import json
from pathlib import Path
import sys

inventory, policy, held_out = map(Path, sys.argv[1:])
print(json.loads(inventory.read_bytes())["source_manifest_sha256"])
print(sha256(policy.read_bytes()).hexdigest())
print(sha256(held_out.read_bytes()).hexdigest())
PY
)
[[ "${identities[0]}" == "$SOURCE_INVENTORY_SHA256" ]] || {
    echo "source inventory identity mismatch" >&2
    exit 2
}
[[ "${identities[1]}" == "$POLICY_FILE_SHA256" ]] || { echo "policy identity mismatch" >&2; exit 2; }
[[ "${identities[2]}" == "$HELD_OUT_FILE_SHA256" ]] || {
    echo "held-out UUID identity mismatch" >&2
    exit 2
}

job_local_parent="${SLURM_TMPDIR:-/tmp}"
[[ "$job_local_parent" == /* && "$job_local_parent" != /home* && "$job_local_parent" != /lustre* ]] || {
    echo "job-local scratch must not use home or Lustre" >&2
    exit 2
}
scratch_root="$job_local_parent/qwen4b-task9-b-${SLURM_JOB_ID}"
[[ ! -e "$scratch_root" && ! -L "$scratch_root" ]] || {
    echo "job-local Task9 root already exists" >&2
    exit 2
}
mkdir -p "$scratch_root"
trap 'find "$scratch_root" -depth -delete 2>/dev/null || true' EXIT

python3 "$REPO_ROOT/examples/dataset/qwen3_4b_ptv2_study.py" \
    --policy "$POLICY" \
    --source-inventory "$SOURCE_INVENTORY" \
    --output-root "$scratch_root/selection" \
    --held-out-uuids "$HELD_OUT_UUIDS" \
    --selection-receipt-root "$RECEIPT_ROOT" \
    --workers "$SLURM_CPUS_PER_TASK"

[[ -f "$RECEIPT_ROOT/SELECTION_RECEIPT.json" && ! -L "$RECEIPT_ROOT/SELECTION_RECEIPT.json" ]] || {
    echo "Task9 B selection did not publish its immutable receipt" >&2
    exit 1
}
[[ -f "$RECEIPT_ROOT/EXECUTION_RECEIPT.json" \
    && ! -L "$RECEIPT_ROOT/EXECUTION_RECEIPT.json" ]] || exit 1
sha256sum "$RECEIPT_ROOT/SELECTION_RECEIPT.json"
