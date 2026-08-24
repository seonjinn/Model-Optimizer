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
    REPO_ROOT SOURCE_COMMIT TASK9_B_VIEW TASK8_PUBLICATION
    TASK8_PUBLICATION_SHA256 TASK9_SELECTION_RECEIPT_SHA256
    B_CANARY_BUILD_ROOT B_CANARY_READINESS B_CANARY_MANIFEST
    B_CANARY_SEED B_CANARY_ACCOUNT
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
[[ "$REPO_ROOT" == /* && "$TASK9_B_VIEW" == /* && "$TASK8_PUBLICATION" == /* ]] || {
    echo "B builder paths must be absolute" >&2
    exit 2
}
[[ "$B_CANARY_BUILD_ROOT" == /* && "$B_CANARY_READINESS" == /* ]] || {
    echo "B builder receipt or source identity is invalid" >&2
    exit 2
}
[[ "$B_CANARY_MANIFEST" == /* && "$SOURCE_COMMIT" =~ ^[0-9a-f]{40}$ ]] || exit 2
[[ "$TASK8_PUBLICATION_SHA256" =~ ^[0-9a-f]{64}$ ]] || exit 2
[[ "$TASK9_SELECTION_RECEIPT_SHA256" =~ ^[0-9a-f]{64}$ ]] || exit 2
case "$B_CANARY_ACCOUNT" in nemotron_sw_post|nemotron_n4_post) ;; *) exit 2 ;; esac
[[ "$B_CANARY_SEED" =~ ^[0-9]+$ ]] || { echo "B builder seed must be non-negative" >&2; exit 2; }
[[ "$(git -C "$REPO_ROOT" rev-parse HEAD)" == "$SOURCE_COMMIT" ]] || {
    echo "source checkout commit mismatch" >&2
    exit 2
}
[[ -z "$(git -C "$REPO_ROOT" status --porcelain)" ]] || {
    echo "source checkout is dirty" >&2
    exit 2
}
[[ -f "$TASK9_B_VIEW" && -d "$TASK8_PUBLICATION" ]] || {
    echo "Task9 B projection or Task8 publication is absent" >&2
    exit 2
}

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
    --task8-publication "$TASK8_PUBLICATION" \
    --task8-publication-sha256 "$TASK8_PUBLICATION_SHA256" \
    --task9-selection-receipt-sha256 "$TASK9_SELECTION_RECEIPT_SHA256" \
    --task9-view "$TASK9_B_VIEW" \
    --output-root "$B_CANARY_BUILD_ROOT" \
    --scratch-root "$scratch_root" \
    --seed "$B_CANARY_SEED" \
    --source-commit "$SOURCE_COMMIT" \
    --job-id "$SLURM_JOB_ID" \
    --workers "$SLURM_CPUS_PER_TASK"

PYTHONPATH="$launcher_root${PYTHONPATH:+:$PYTHONPATH}" python3 - \
    "$TASK9_B_VIEW" "$B_CANARY_BUILD_ROOT" "$B_CANARY_READINESS" \
    "$B_CANARY_MANIFEST" "$SOURCE_COMMIT" "$B_CANARY_ACCOUNT" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

from common.specdec.qwen4b_b_canary_manifest import (
    BCanaryManifest,
    BCanaryTopology,
    write_b_canary_manifest,
)
from common.specdec.qwen4b_b_readiness import (
    BReadinessInputs,
    assess_b_readiness,
    load_task9_balanced_view,
    write_b_readiness_receipt,
)

view_path, build_root, readiness_path, manifest_path = map(Path, sys.argv[1:5])
source_commit, account = sys.argv[5:7]
receipt_path = build_root / "BUILD_RECEIPT.json"
output_path = build_root / "canary.jsonl"
receipt_sha = hashlib.sha256(receipt_path.read_bytes()).hexdigest()
output_sha = hashlib.sha256(output_path.read_bytes()).hexdigest()
readiness = assess_b_readiness(
    BReadinessInputs(
        task9_b_view=load_task9_balanced_view(view_path),
        source_commit=source_commit,
        builder_receipt_path=receipt_path,
        builder_receipt_sha256=receipt_sha,
        builder_output_path=output_path,
        builder_output_sha256=output_sha,
    )
)
if not readiness.ready:
    raise SystemExit("B builder readiness blocked: " + ",".join(readiness.blocker_codes))
write_b_readiness_receipt(readiness_path, readiness)
readiness_payload = json.loads(readiness_path.read_bytes())
write_b_canary_manifest(
    manifest_path,
    BCanaryManifest(
        readiness_receipt_sha256=readiness_payload["receipt_sha256"],
        builder_receipt_sha256=receipt_sha,
        builder_output_sha256=output_sha,
        canary_occurrence_count=102_400,
        topology=BCanaryTopology(
            cluster="oci-hsg",
            account=account,
            partition="batch",
            nodes=16,
            segment=16,
            serve_nodes=8,
            train_nodes=8,
            gpus_per_node=4,
            cpu_datamover=96,
            per_device_batch_size=4,
            gradient_accumulation_steps=4,
            max_steps=200,
            wandb_project="sna-qwen3-4b-dataset-study",
        ),
        source_commit=source_commit,
    ),
)
PY
