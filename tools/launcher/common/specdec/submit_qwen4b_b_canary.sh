#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

usage() {
    echo "usage: $0 (--test-only|--submit-prep|--submit-canary) --account ACCOUNT --repo-root PATH --source-commit SHA --task9-view PATH --task9-view-sha256 SHA --task8-publication PATH --task8-publication-sha256 SHA --task9-selection-receipt-sha256 SHA --build-root PATH --readiness PATH --manifest PATH --evidence PATH --receipt PATH --target-path PATH --target-revision SHA --container-image PATH --container-sha256 SHA --modelopt-runtime PATH --output-root PATH [--a-authorization-receipt PATH --a-authorization-receipt-sha256 SHA]" >&2
    exit 2
}

mode="" account="" repo_root="" source_commit="" task9_view="" task9_view_sha256=""
task8_publication="" task8_publication_sha256="" task9_selection_receipt_sha256=""
build_root="" readiness_path="" manifest_path="" evidence_path="" receipt_path=""
target_path="" target_revision="" container_image="" container_sha256=""
modelopt_runtime="" output_root="" a_receipt="" a_receipt_sha256=""
while (( $# )); do
    case "$1" in
        --test-only|--submit-prep|--submit-canary) [[ -z "$mode" ]] || usage; mode="$1"; shift ;;
        --account) account="${2:-}"; shift 2 ;;
        --repo-root) repo_root="${2:-}"; shift 2 ;;
        --source-commit) source_commit="${2:-}"; shift 2 ;;
        --task9-view) task9_view="${2:-}"; shift 2 ;;
        --task9-view-sha256) task9_view_sha256="${2:-}"; shift 2 ;;
        --task8-publication) task8_publication="${2:-}"; shift 2 ;;
        --task8-publication-sha256) task8_publication_sha256="${2:-}"; shift 2 ;;
        --task9-selection-receipt-sha256) task9_selection_receipt_sha256="${2:-}"; shift 2 ;;
        --build-root) build_root="${2:-}"; shift 2 ;;
        --readiness) readiness_path="${2:-}"; shift 2 ;;
        --manifest) manifest_path="${2:-}"; shift 2 ;;
        --evidence) evidence_path="${2:-}"; shift 2 ;;
        --receipt) receipt_path="${2:-}"; shift 2 ;;
        --target-path) target_path="${2:-}"; shift 2 ;;
        --target-revision) target_revision="${2:-}"; shift 2 ;;
        --container-image) container_image="${2:-}"; shift 2 ;;
        --container-sha256) container_sha256="${2:-}"; shift 2 ;;
        --modelopt-runtime) modelopt_runtime="${2:-}"; shift 2 ;;
        --output-root) output_root="${2:-}"; shift 2 ;;
        --a-authorization-receipt) a_receipt="${2:-}"; shift 2 ;;
        --a-authorization-receipt-sha256) a_receipt_sha256="${2:-}"; shift 2 ;;
        *) usage ;;
    esac
done

[[ -n "$mode" && -n "$account" && -n "$repo_root" && -n "$source_commit" ]] || usage
[[ -n "$task9_view" && -n "$task8_publication" && -n "$build_root" ]] || usage
[[ -n "$readiness_path" && -n "$manifest_path" && -n "$evidence_path" && -n "$receipt_path" ]] || usage
[[ -n "$target_path" && -n "$container_image" && -n "$modelopt_runtime" && -n "$output_root" ]] || usage
[[ "$source_commit" =~ ^[0-9a-f]{40}$ && "$target_revision" =~ ^[0-9a-f]{40}$ ]] || usage
for digest in "$task9_view_sha256" "$task8_publication_sha256" \
    "$task9_selection_receipt_sha256" "$container_sha256"; do
    [[ "$digest" =~ ^[0-9a-f]{64}$ ]] || usage
done
case "$account" in nemotron_sw_post|nemotron_n4_post) ;; *) usage ;; esac
for path in "$repo_root" "$task9_view" "$task8_publication" "$build_root" \
    "$readiness_path" "$manifest_path" "$evidence_path" "$receipt_path" "$target_path" \
    "$container_image" "$modelopt_runtime" "$output_root"; do
    [[ "$path" == /* ]] || usage
done
[[ -d "$repo_root" && -f "$task9_view" && -d "$task8_publication" ]] || usage
[[ ! -e "$receipt_path" && ! -L "$receipt_path" ]] || { echo "submission receipt already exists" >&2; exit 2; }

launcher_root="$repo_root/tools/launcher"
builder_runner="$launcher_root/common/specdec/run_qwen4b_b_builder.sbatch"
canary_runner="$launcher_root/common/specdec/run_qwen4b_b_canary.sbatch"
wandb_run_id="q4b-b-${source_commit:0:12}"

# This gate intentionally precedes even sbatch --test-only so a production request
# cannot cause any scheduler mutation before its caller-pinned A trust root passes.
if [[ "$mode" == "--submit-canary" ]]; then
    [[ -f "$a_receipt" && "$a_receipt_sha256" =~ ^[0-9a-f]{64}$ ]] || {
        echo "authenticated A-repair authorization is required" >&2
        exit 2
    }
    PYTHONPATH="$launcher_root${PYTHONPATH:+:$PYTHONPATH}" python3 - \
        "$a_receipt" "$a_receipt_sha256" "$source_commit" "$target_revision" \
        "$task9_view" "$container_sha256" "$task9_view_sha256" \
        "$container_image" <<'PY'
import hashlib
import os
import sys
from pathlib import Path
from common.specdec.qwen4b_b_canary_manifest import validate_a_authorization_receipt
from common.specdec.qwen4b_b_readiness import load_task9_balanced_view
projection = load_task9_balanced_view(Path(sys.argv[5]), sys.argv[7])
flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
descriptor = os.open(sys.argv[8], flags)
with os.fdopen(descriptor, "rb") as stream:
    digest = hashlib.sha256()
    for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
        digest.update(block)
if digest.hexdigest() != sys.argv[6]:
    raise ValueError("container image identity mismatch")
validate_a_authorization_receipt(
    Path(sys.argv[1]),
    sys.argv[2],
    source_commit=sys.argv[3],
    target_revision=sys.argv[4],
    tokenizer_sha256=projection.tokenizer_sha256,
    chat_template_sha256=projection.chat_template_sha256,
    container_sha256=sys.argv[6],
)
PY
fi

builder_exports="ALL,REPO_ROOT=$repo_root,SOURCE_COMMIT=$source_commit,TASK9_B_VIEW=$task9_view,TASK9_B_VIEW_SHA256=$task9_view_sha256,TASK8_PUBLICATION=$task8_publication,TASK8_PUBLICATION_SHA256=$task8_publication_sha256,TASK9_SELECTION_RECEIPT_SHA256=$task9_selection_receipt_sha256,B_CANARY_BUILD_ROOT=$build_root,B_CANARY_READINESS=$readiness_path,B_CANARY_MANIFEST=$manifest_path,B_CANARY_SEED=42,B_CANARY_ACCOUNT=$account,TARGET_PATH=$target_path,TARGET_REVISION=$target_revision,CONTAINER_SHA256=$container_sha256,B_CANARY_OUTPUT_ROOT=$output_root,B_CANARY_WANDB_RUN_ID=$wandb_run_id"
canary_exports="ALL,LAUNCHER_ROOT=$launcher_root,REPO_ROOT=$repo_root,MODELOPT_RUNTIME=$modelopt_runtime,B_CANARY_MANIFEST=$manifest_path,B_CANARY_READINESS=$readiness_path,B_CANARY_BUILD_RECEIPT=$build_root/BUILD_RECEIPT.json,B_CANARY_OUTPUT=$build_root/canary.jsonl,B_CANARY_EVIDENCE=$evidence_path,B_CANARY_CHECKPOINT=$output_root/checkpoint,B_CANARY_EXPORT=$output_root/export,B_CANARY_EVALUATION_RECEIPT=$output_root/evaluation/RESULT.json,B_CANARY_GPU_EVIDENCE=$output_root/control/GPU_ACTIVITY.json,B_A_AUTHORIZATION_RECEIPT=$a_receipt,B_A_AUTHORIZATION_RECEIPT_SHA256=$a_receipt_sha256"

builder_job_id="" canary_job_id=""
sbatch --test-only --account="$account" --partition=cpu_datamover --nodes=1 \
    --ntasks=1 --cpus-per-task=96 --export="$builder_exports" "$builder_runner" >/dev/null
if [[ "$mode" == "--test-only" || "$mode" == "--submit-canary" ]]; then
    sbatch --test-only --account="$account" --partition=batch --nodes=16 --segment=16 \
        --ntasks-per-node=1 --gpus-per-node=4 --cpus-per-task=96 \
        --container-image="$container_image" --export="$canary_exports" "$canary_runner" >/dev/null
fi
if [[ "$mode" == "--submit-prep" || "$mode" == "--submit-canary" ]]; then
    builder_job_id="$(sbatch --parsable --account="$account" --partition=cpu_datamover \
        --nodes=1 --ntasks=1 --cpus-per-task=96 --export="$builder_exports" "$builder_runner")"
    [[ "$builder_job_id" =~ ^[0-9]+([_;].*)?$ ]] || exit 1
fi
if [[ "$mode" == "--submit-canary" ]]; then
    dependency_id="${builder_job_id%%[_;]*}"
    canary_job_id="$(sbatch --parsable --account="$account" --partition=batch \
        --dependency=afterok:"$dependency_id" --nodes=16 --segment=16 \
        --ntasks-per-node=1 --gpus-per-node=4 --cpus-per-task=96 \
        --container-image="$container_image" --export="$canary_exports" "$canary_runner")"
    [[ "$canary_job_id" =~ ^[0-9]+([_;].*)?$ ]] || exit 1
fi

mkdir -p "$(dirname "$receipt_path")"
PYTHONPATH="$launcher_root${PYTHONPATH:+:$PYTHONPATH}" python3 - \
    "$receipt_path" "$source_commit" "$mode" "$builder_job_id" "$canary_job_id" \
    "$a_receipt_sha256" <<'PY'
import json
import sys
from pathlib import Path
from common.specdec.qwen4b_b_atomic import atomic_publish_bytes
path = Path(sys.argv[1])
record = {
    "schema_version": 2,
    "source_commit": sys.argv[2],
    "mode": sys.argv[3],
    "test_only_passed": sys.argv[3] == "--test-only",
    "canary_submitted": bool(sys.argv[5]),
    "builder_job_id": sys.argv[4] or None,
    "canary_job_id": sys.argv[5] or None,
    "dependency": None if not sys.argv[5] else f"afterok:{sys.argv[4].split(';')[0].split('_')[0]}",
    "a_authorization_receipt_sha256": sys.argv[6] or None,
    "required_training_order": "A-repair-first",
    "b_preparation_only": sys.argv[3] != "--submit-canary",
    "scientific_training_authorized": False,
}
atomic_publish_bytes(path, (json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n").encode(), job_id="submit")
PY
