#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

usage() {
    echo "usage: $0 (--test-only|--submit-canary) --account ACCOUNT --repo-root PATH --source-commit SHA --task9-view PATH --task8-publication PATH --task8-publication-sha256 SHA --task9-selection-receipt-sha256 SHA --build-root PATH --readiness PATH --manifest PATH --evidence PATH --receipt PATH" >&2
    exit 2
}

mode=""
account=""
repo_root=""
source_commit=""
task9_view=""
task8_publication=""
task8_publication_sha256=""
task9_selection_receipt_sha256=""
build_root=""
readiness_path=""
manifest_path=""
evidence_path=""
receipt_path=""
while (( $# )); do
    case "$1" in
        --test-only|--submit-canary) [[ -z "$mode" ]] || usage; mode="$1"; shift ;;
        --account) account="${2:-}"; shift 2 ;;
        --repo-root) repo_root="${2:-}"; shift 2 ;;
        --source-commit) source_commit="${2:-}"; shift 2 ;;
        --task9-view) task9_view="${2:-}"; shift 2 ;;
        --task8-publication) task8_publication="${2:-}"; shift 2 ;;
        --task8-publication-sha256) task8_publication_sha256="${2:-}"; shift 2 ;;
        --task9-selection-receipt-sha256) task9_selection_receipt_sha256="${2:-}"; shift 2 ;;
        --build-root) build_root="${2:-}"; shift 2 ;;
        --readiness) readiness_path="${2:-}"; shift 2 ;;
        --manifest) manifest_path="${2:-}"; shift 2 ;;
        --evidence) evidence_path="${2:-}"; shift 2 ;;
        --receipt) receipt_path="${2:-}"; shift 2 ;;
        *) usage ;;
    esac
done

[[ -n "$mode" && -n "$account" && -n "$repo_root" && -n "$source_commit" ]] || usage
[[ -n "$task9_view" && -n "$task8_publication" && -n "$build_root" ]] || usage
[[ -n "$readiness_path" && -n "$manifest_path" && -n "$evidence_path" && -n "$receipt_path" ]] || usage
[[ "$source_commit" =~ ^[0-9a-f]{40}$ ]] || usage
[[ "$task8_publication_sha256" =~ ^[0-9a-f]{64}$ ]] || usage
[[ "$task9_selection_receipt_sha256" =~ ^[0-9a-f]{64}$ ]] || usage
case "$account" in nemotron_sw_post|nemotron_n4_post) ;; *) usage ;; esac
for path in "$repo_root" "$task9_view" "$task8_publication" "$build_root" \
    "$readiness_path" "$manifest_path" "$evidence_path" "$receipt_path"; do
    [[ "$path" == /* ]] || usage
done
[[ -d "$repo_root" && -f "$task9_view" && -d "$task8_publication" ]] || usage
[[ ! -e "$receipt_path" && ! -L "$receipt_path" ]] || {
    echo "submission receipt already exists" >&2
    exit 2
}
: "${B_CANARY_SERVE_COMMAND:?B_CANARY_SERVE_COMMAND is required}"
: "${B_CANARY_TRAIN_COMMAND:?B_CANARY_TRAIN_COMMAND is required}"

launcher_root="$repo_root/tools/launcher"
builder_runner="$launcher_root/common/specdec/run_qwen4b_b_builder.sbatch"
canary_runner="$launcher_root/common/specdec/run_qwen4b_b_canary.sbatch"
builder_exports="ALL,REPO_ROOT=$repo_root,SOURCE_COMMIT=$source_commit,TASK9_B_VIEW=$task9_view,TASK8_PUBLICATION=$task8_publication,TASK8_PUBLICATION_SHA256=$task8_publication_sha256,TASK9_SELECTION_RECEIPT_SHA256=$task9_selection_receipt_sha256,B_CANARY_BUILD_ROOT=$build_root,B_CANARY_READINESS=$readiness_path,B_CANARY_MANIFEST=$manifest_path,B_CANARY_SEED=20260822,B_CANARY_ACCOUNT=$account"
canary_exports="ALL,LAUNCHER_ROOT=$launcher_root,B_CANARY_MANIFEST=$manifest_path,B_CANARY_READINESS=$readiness_path,B_CANARY_BUILD_RECEIPT=$build_root/BUILD_RECEIPT.json,B_CANARY_OUTPUT=$build_root/canary.jsonl,B_CANARY_EVIDENCE=$evidence_path"

builder_job_id=""
canary_job_id=""
sbatch --test-only --account="$account" --partition=cpu_datamover --nodes=1 \
    --ntasks=1 --cpus-per-task=96 --export="$builder_exports" "$builder_runner" >/dev/null
sbatch --test-only --account="$account" --partition=batch --nodes=16 --segment=16 \
    --ntasks-per-node=4 --gpus-per-node=4 --cpus-per-task=96 \
    --export="$canary_exports" "$canary_runner" >/dev/null
if [[ "$mode" == "--submit-canary" ]]; then
    builder_job_id="$(sbatch --parsable --account="$account" --partition=cpu_datamover \
        --nodes=1 --ntasks=1 --cpus-per-task=96 --export="$builder_exports" "$builder_runner")"
    [[ "$builder_job_id" =~ ^[0-9]+([_;].*)?$ ]] || {
        echo "builder submission returned an invalid job identity" >&2
        exit 1
    }
    builder_dependency_id="${builder_job_id%%[_;]*}"
    canary_job_id="$(sbatch --parsable --account="$account" --partition=batch \
        --dependency=afterok:"$builder_dependency_id" --nodes=16 --segment=16 \
        --ntasks-per-node=4 --gpus-per-node=4 --cpus-per-task=96 \
        --export="$canary_exports" "$canary_runner")"
    [[ "$canary_job_id" =~ ^[0-9]+([_;].*)?$ ]] || {
        echo "canary submission returned an invalid job identity" >&2
        exit 1
    }
fi

receipt_parent="$(dirname "$receipt_path")"
mkdir -p "$receipt_parent"
PYTHONPATH="$launcher_root${PYTHONPATH:+:$PYTHONPATH}" python3 - \
    "$receipt_path" "$source_commit" "$mode" "$builder_job_id" "$canary_job_id" <<'PY'
import json
import sys
from pathlib import Path

from common.specdec.qwen4b_b_atomic import atomic_publish_bytes

path = Path(sys.argv[1])
record = {
    "schema_version": 1,
    "source_commit": sys.argv[2],
    "test_only_passed": sys.argv[3] == "--test-only",
    "canary_submitted": sys.argv[3] == "--submit-canary",
    "builder_job_id": sys.argv[4] or None,
    "canary_job_id": sys.argv[5] or None,
    "dependency": None if not sys.argv[4] else f"afterok:{sys.argv[4].split(';')[0].split('_')[0]}",
    "required_training_order": "A-repair-first",
    "b_preparation_only": True,
    "scientific_training_authorized": False,
}
atomic_publish_bytes(
    path,
    (json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n").encode(),
    job_id="submit",
)
PY
