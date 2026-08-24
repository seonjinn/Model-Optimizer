#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

usage() {
    echo "usage: $0 (--test-only|--submit) --account ACCOUNT --repo-root PATH --source-commit SHA --task9-a-selection PATH --task9-a-selection-sha256 SHA --task8-publication PATH --task8-publication-sha256 SHA --build-root PATH --manifest PATH --authorization PATH --target-path PATH --target-revision SHA --tokenizer-sha256 SHA --chat-template-sha256 SHA --container-image PATH --container-sha256 SHA --modelopt-runtime PATH --speculators-runtime PATH --speculators-repo PATH --hf-home PATH --eval-config PATH --container-identity PATH --dataset-manifest PATH --output-root PATH --submission-receipt PATH" >&2
    exit 2
}

mode="" account="" repo_root="" source_commit="" task9_selection="" selection_sha=""
task8_publication="" publication_sha="" build_root="" manifest="" authorization=""
target_path="" target_revision="" tokenizer_sha="" template_sha="" container_image=""
container_sha="" modelopt_runtime="" output_root="" submission_receipt=""
speculators_runtime="" speculators_repo="" hf_home="" eval_config=""
container_identity="" dataset_manifest=""
while (( $# )); do
    case "$1" in
        --test-only|--submit) [[ -z "$mode" ]] || usage; mode="$1"; shift ;;
        --account) account="${2:-}"; shift 2 ;;
        --repo-root) repo_root="${2:-}"; shift 2 ;;
        --source-commit) source_commit="${2:-}"; shift 2 ;;
        --task9-a-selection) task9_selection="${2:-}"; shift 2 ;;
        --task9-a-selection-sha256) selection_sha="${2:-}"; shift 2 ;;
        --task8-publication) task8_publication="${2:-}"; shift 2 ;;
        --task8-publication-sha256) publication_sha="${2:-}"; shift 2 ;;
        --build-root) build_root="${2:-}"; shift 2 ;;
        --manifest) manifest="${2:-}"; shift 2 ;;
        --authorization) authorization="${2:-}"; shift 2 ;;
        --target-path) target_path="${2:-}"; shift 2 ;;
        --target-revision) target_revision="${2:-}"; shift 2 ;;
        --tokenizer-sha256) tokenizer_sha="${2:-}"; shift 2 ;;
        --chat-template-sha256) template_sha="${2:-}"; shift 2 ;;
        --container-image) container_image="${2:-}"; shift 2 ;;
        --container-sha256) container_sha="${2:-}"; shift 2 ;;
        --modelopt-runtime) modelopt_runtime="${2:-}"; shift 2 ;;
        --speculators-runtime) speculators_runtime="${2:-}"; shift 2 ;;
        --speculators-repo) speculators_repo="${2:-}"; shift 2 ;;
        --hf-home) hf_home="${2:-}"; shift 2 ;;
        --eval-config) eval_config="${2:-}"; shift 2 ;;
        --container-identity) container_identity="${2:-}"; shift 2 ;;
        --dataset-manifest) dataset_manifest="${2:-}"; shift 2 ;;
        --output-root) output_root="${2:-}"; shift 2 ;;
        --submission-receipt) submission_receipt="${2:-}"; shift 2 ;;
        *) usage ;;
    esac
done

[[ -n "$mode" && -n "$account" && -n "$repo_root" && -n "$source_commit" ]] || usage
[[ -n "$task9_selection" && -n "$task8_publication" && -n "$build_root" && -n "$manifest" ]] || usage
[[ -n "$authorization" && -n "$target_path" && -n "$container_image" && -n "$modelopt_runtime" ]] || usage
[[ -n "$speculators_runtime" && -n "$speculators_repo" && -n "$hf_home" ]] || usage
[[ -n "$eval_config" && -n "$container_identity" && -n "$dataset_manifest" ]] || usage
[[ -n "$output_root" && -n "$submission_receipt" ]] || usage
case "$account" in nemotron_sw_post|nemotron_n4_post) ;; *) usage ;; esac
[[ "$source_commit" =~ ^[0-9a-f]{40}$ && "$target_revision" =~ ^[0-9a-f]{40}$ ]] || usage
for digest in "$selection_sha" "$publication_sha" "$tokenizer_sha" "$template_sha" "$container_sha"; do
    [[ "$digest" =~ ^[0-9a-f]{64}$ ]] || usage
done
for path in "$repo_root" "$task9_selection" "$task8_publication" "$build_root" "$manifest" \
    "$authorization" "$target_path" "$container_image" "$modelopt_runtime" "$output_root" \
    "$submission_receipt" "$speculators_runtime" "$speculators_repo" "$hf_home" \
    "$eval_config" "$container_identity" "$dataset_manifest"; do
    [[ "$path" == /* ]] || usage
done
[[ -d "$repo_root" && -f "$task9_selection" && -d "$task8_publication" ]] || usage
[[ -d "$target_path" && -f "$container_image" && -d "$modelopt_runtime" ]] || usage
[[ -d "$speculators_runtime" && -d "$speculators_repo" && -d "$hf_home" ]] || usage
[[ -f "$eval_config" && -f "$container_identity" && -f "$dataset_manifest" ]] || usage
[[ ! -e "$submission_receipt" && ! -L "$submission_receipt" ]] || exit 2
[[ "$(git -C "$repo_root" rev-parse HEAD)" == "$source_commit" ]] || exit 2
[[ -z "$(git -C "$repo_root" status --porcelain)" ]] || exit 2

python3 - "$task9_selection" "$selection_sha" \
    "$task8_publication/PUBLICATION.json" "$publication_sha" \
    "$container_image" "$container_sha" <<'PY'
import hashlib
import sys
from pathlib import Path
for path_value, expected in zip(sys.argv[1::2], sys.argv[2::2], strict=True):
    digest = hashlib.sha256()
    with Path(path_value).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    if digest.hexdigest() != expected:
        raise ValueError(f"caller-pinned artifact identity mismatch: {path_value}")
PY

readonly LAUNCHER_ROOT="$repo_root/tools/launcher"
readonly CPU_SCRIPT="$LAUNCHER_ROOT/common/specdec/run_qwen4b_a_builder.sbatch"
readonly GPU_SCRIPT="$LAUNCHER_ROOT/common/specdec/run_qwen4b_a_canary.sbatch"
builder_exports="ALL,REPO_ROOT=$repo_root,SOURCE_COMMIT=$source_commit,TASK9_A_SELECTION=$task9_selection,TASK9_A_SELECTION_SHA256=$selection_sha,TASK8_PUBLICATION=$task8_publication,TASK8_PUBLICATION_SHA256=$publication_sha,A_CANARY_BUILD_ROOT=$build_root,A_CANARY_MANIFEST=$manifest,A_CANARY_ACCOUNT=$account,TARGET_REVISION=$target_revision,TOKENIZER_SHA256=$tokenizer_sha,CHAT_TEMPLATE_SHA256=$template_sha,CONTAINER_SHA256=$container_sha"
gpu_exports="ALL,A_CANARY_MANIFEST=$manifest,A_CANARY_CHECKPOINT=$output_root/train,A_CANARY_EXPORT=$output_root/export,A_CANARY_GPU_EVIDENCE=$output_root/control/GPU_ACTIVITY.json,A_CANARY_EVALUATION_RECEIPT=$output_root/evaluation/RESULT.json,A_CANARY_AUTHORIZATION=$authorization,A_CANARY_EVAL_OUTPUT=$output_root/evaluation/specdec,LAUNCHER_ROOT=$LAUNCHER_ROOT,REPO_ROOT=$repo_root,TARGET_PATH=$target_path,MODELOPT_RUNTIME=$modelopt_runtime,SPECULATORS_RUNTIME=$speculators_runtime,SPECULATORS_REPO=$speculators_repo,HF_HOME=$hf_home,EVAL_CONFIG_PATH=$eval_config,CONTAINER_IMAGE=$container_image,CONTAINER_IDENTITY_PATH=$container_identity,DATASET_MANIFEST_PATH=$dataset_manifest,MODELOPT_REPO=$repo_root"

# Validate both allocations before any scheduler mutation.
sbatch --account="$account" --partition=cpu_datamover --nodes=1 --ntasks=1 \
    --cpus-per-task=96 --export="$builder_exports" --test-only --parsable "$CPU_SCRIPT" >/dev/null
sbatch --account="$account" --partition=batch --nodes=16 --segment=16 --ntasks-per-node=1 \
    --gpus-per-node=4 --cpus-per-task=96 --container-image="$container_image" \
    --export="$gpu_exports" --test-only --parsable "$GPU_SCRIPT" >/dev/null

builder_job_id="" gpu_job_id=""
if [[ "$mode" == --submit ]]; then
    builder_job_id="$(sbatch --parsable --account="$account" --partition=cpu_datamover \
        --nodes=1 --ntasks=1 --cpus-per-task=96 --export="$builder_exports" "$CPU_SCRIPT")"
    dependency_id="${builder_job_id%%[_;]*}"
    [[ "$dependency_id" =~ ^[0-9]+$ ]] || exit 1
    gpu_job_id="$(sbatch --parsable --account="$account" --partition=batch \
        --dependency="afterok:${dependency_id}" --nodes=16 --segment=16 --ntasks-per-node=1 \
        --gpus-per-node=4 --cpus-per-task=96 --container-image="$container_image" \
        --export="$gpu_exports" "$GPU_SCRIPT")"
fi

mkdir -p "$(dirname "$submission_receipt")"
PYTHONPATH="$LAUNCHER_ROOT${PYTHONPATH:+:$PYTHONPATH}" python3 - \
    "$submission_receipt" "$source_commit" "$mode" "$builder_job_id" "$gpu_job_id" <<'PY'
import json
import sys
from pathlib import Path
from common.specdec.qwen4b_b_atomic import atomic_publish_bytes
path = Path(sys.argv[1])
body = {
    "schema_version": 1,
    "source_commit": sys.argv[2],
    "mode": sys.argv[3],
    "builder_job_id": sys.argv[4] or None,
    "gpu_job_id": sys.argv[5] or None,
    "dependency": None if not sys.argv[5] else f"afterok:{sys.argv[4].split(';')[0].split('_')[0]}",
    "authorization": "B-balanced-canary",
    "scientific_training_authorized": False,
}
atomic_publish_bytes(
    path,
    (json.dumps(body, sort_keys=True, separators=(",", ":")) + "\n").encode(),
    job_id="submit",
)
PY
# Stable literal retained for static authorization audits: scientific_training_authorized=false
