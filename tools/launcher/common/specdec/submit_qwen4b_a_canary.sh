#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

usage() {
    echo "usage: $0 (--test-only|--submit) --account ACCOUNT --repo-root PATH --source-commit SHA --task9-a-selection PATH --task9-a-selection-sha256 SHA --task8-publication PATH --task8-publication-sha256 SHA --build-root PATH --manifest PATH --authorization PATH --target-path PATH --target-revision SHA --container-image PATH --modelopt-runtime PATH --speculators-runtime PATH --speculators-repo PATH --hf-home PATH --eval-config PATH --container-identity PATH --dataset-manifest PATH --wandb-netrc PATH --wandb-dir PATH --wandb-cache-dir PATH --wandb-config-dir PATH --wandb-artifact-dir PATH --slurm-comment TEXT --slurm-output PATH --output-root PATH --submission-receipt PATH" >&2
    exit 2
}

mode="" account="" repo_root="" source_commit="" task9_selection="" selection_sha=""
task8_publication="" publication_sha="" build_root="" manifest="" authorization=""
target_path="" target_revision="" container_image=""
modelopt_runtime="" output_root="" submission_receipt=""
speculators_runtime="" speculators_repo="" hf_home="" eval_config=""
container_identity="" dataset_manifest=""
wandb_netrc="" wandb_dir="" wandb_cache_dir="" wandb_config_dir="" wandb_artifact_dir=""
slurm_comment="" slurm_output=""
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
        --container-image) container_image="${2:-}"; shift 2 ;;
        --modelopt-runtime) modelopt_runtime="${2:-}"; shift 2 ;;
        --speculators-runtime) speculators_runtime="${2:-}"; shift 2 ;;
        --speculators-repo) speculators_repo="${2:-}"; shift 2 ;;
        --hf-home) hf_home="${2:-}"; shift 2 ;;
        --eval-config) eval_config="${2:-}"; shift 2 ;;
        --container-identity) container_identity="${2:-}"; shift 2 ;;
        --dataset-manifest) dataset_manifest="${2:-}"; shift 2 ;;
        --wandb-netrc) wandb_netrc="${2:-}"; shift 2 ;;
        --wandb-dir) wandb_dir="${2:-}"; shift 2 ;;
        --wandb-cache-dir) wandb_cache_dir="${2:-}"; shift 2 ;;
        --wandb-config-dir) wandb_config_dir="${2:-}"; shift 2 ;;
        --wandb-artifact-dir) wandb_artifact_dir="${2:-}"; shift 2 ;;
        --slurm-comment) slurm_comment="${2:-}"; shift 2 ;;
        --slurm-output) slurm_output="${2:-}"; shift 2 ;;
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
[[ -n "$wandb_netrc" && -n "$wandb_dir" && -n "$wandb_cache_dir" ]] || usage
[[ -n "$wandb_config_dir" && -n "$wandb_artifact_dir" && -n "$slurm_comment" ]] || usage
[[ -n "$slurm_output" ]] || usage
[[ "$slurm_output" != *%* ]] || usage
[[ -n "$output_root" && -n "$submission_receipt" ]] || usage
case "$account" in nemotron_sw_post|nemotron_n4_post) ;; *) usage ;; esac
[[ "$source_commit" =~ ^[0-9a-f]{40}$ && "$target_revision" =~ ^[0-9a-f]{40}$ ]] || usage
for digest in "$selection_sha" "$publication_sha"; do
    [[ "$digest" =~ ^[0-9a-f]{64}$ ]] || usage
done
for path in "$repo_root" "$task9_selection" "$task8_publication" "$build_root" "$manifest" \
    "$authorization" "$target_path" "$container_image" "$modelopt_runtime" "$output_root" \
    "$submission_receipt" "$speculators_runtime" "$speculators_repo" "$hf_home" \
    "$eval_config" "$container_identity" "$dataset_manifest"; do
    [[ "$path" == /* ]] || usage
done
for path in "$wandb_netrc" "$wandb_dir" "$wandb_cache_dir" "$wandb_config_dir" \
    "$wandb_artifact_dir" "$slurm_output"; do
    [[ "$path" == /* ]] || usage
    case "$path" in *","*|*":"*) usage ;; esac
done
[[ -d "$repo_root" && -f "$task9_selection" && -d "$task8_publication" ]] || usage
[[ -d "$target_path" && -f "$container_image" && -d "$modelopt_runtime" ]] || usage
[[ -d "$speculators_runtime" && -d "$speculators_repo" && -d "$hf_home" ]] || usage
[[ -f "$eval_config" && -f "$container_identity" && -f "$dataset_manifest" ]] || usage
[[ -f "$wandb_netrc" && -d "$wandb_dir" && -d "$wandb_cache_dir" ]] || usage
[[ -d "$wandb_config_dir" && -d "$wandb_artifact_dir" ]] || usage
[[ ! -e "$submission_receipt" && ! -L "$submission_receipt" ]] || exit 2
[[ ! -e "$output_root" && ! -L "$output_root" ]] || exit 2
[[ -d "$(dirname "$output_root")" ]] || usage
[[ "$authorization" == "$output_root/control/A_AUTHORIZATION.json" ]] || usage
[[ -d "$(dirname "$slurm_output")" && "$slurm_output" != "$output_root"/* ]] || usage
for path in "$wandb_dir" "$wandb_cache_dir" "$wandb_config_dir" "$wandb_artifact_dir"; do
    [[ "$path" != "$repo_root" && "$path" != "$repo_root"/* ]] || exit 2
    [[ "$path" == /lustre/* ]] || exit 2
done
[[ "$wandb_cache_dir" == */scratch/* ]] || exit 2
[[ "$(git -C "$repo_root" rev-parse HEAD)" == "$source_commit" ]] || exit 2
[[ -z "$(git -C "$repo_root" status --porcelain)" ]] || exit 2

python3 - "$task9_selection" "$selection_sha" \
    "$task8_publication/PUBLICATION.json" "$publication_sha" \
    <<'PY'
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
readonly CONTROLLER_SCRIPT="$LAUNCHER_ROOT/common/specdec/finalize_qwen4b_a_canary.sh"
readonly job_completion="$output_root/control/A_JOB_COMPLETION.json"
readonly scheduler_observation="$output_root/control/A_SCHEDULER_OBSERVATION.json"
builder_exports="ALL,REPO_ROOT=$repo_root,SOURCE_COMMIT=$source_commit,TASK9_A_SELECTION=$task9_selection,TASK9_A_SELECTION_SHA256=$selection_sha,TASK8_PUBLICATION=$task8_publication,TASK8_PUBLICATION_SHA256=$publication_sha,A_CANARY_BUILD_ROOT=$build_root,A_CANARY_MANIFEST=$manifest,A_CANARY_ACCOUNT=$account,TARGET_REVISION=$target_revision,TARGET_PATH=$target_path,CONTAINER_IMAGE=$container_image,WANDB_NETRC_HOST_PATH=$wandb_netrc,WANDB_DIR=$wandb_dir,WANDB_CACHE_DIR=$wandb_cache_dir,WANDB_CONFIG_DIR=$wandb_config_dir,WANDB_ARTIFACT_DIR=$wandb_artifact_dir,MODELOPT_RUNTIME=$modelopt_runtime,SPECULATORS_RUNTIME=$speculators_runtime,SPECULATORS_REPO=$speculators_repo"
gpu_exports="ALL,A_CANARY_MANIFEST=$manifest,A_CANARY_OUTPUT_ROOT=$output_root,A_CANARY_CHECKPOINT=$output_root/train,A_CANARY_EXPORT=$output_root/export,A_CANARY_INTERMEDIATE_EXPORT=$output_root/control/parent-export,A_CANARY_GPU_EVIDENCE=$output_root/control/GPU_ACTIVITY.json,A_CANARY_EVALUATION_RECEIPT=$output_root/evaluation/RESULT.json,A_CANARY_SUPERVISOR_COMPLETION=$output_root/control/SUPERVISOR_COMPLETION.json,A_CANARY_JOB_COMPLETION=$job_completion,A_CANARY_SUBMISSION_RECEIPT=$submission_receipt,A_CANARY_EVAL_OUTPUT=$output_root/evaluation/specdec,LAUNCHER_ROOT=$LAUNCHER_ROOT,REPO_ROOT=$repo_root,MODELOPT_RUNTIME=$modelopt_runtime,SPECULATORS_RUNTIME=$speculators_runtime,SPECULATORS_REPO=$speculators_repo,HF_HOME=$hf_home,EVAL_CONFIG_PATH=$eval_config,CONTAINER_IMAGE=$container_image,CONTAINER_IDENTITY_PATH=$container_identity,DATASET_MANIFEST_PATH=$dataset_manifest,MODELOPT_REPO=$repo_root,WANDB_NETRC_PATH=/run/secrets/wandb.netrc,WANDB_DIR=$wandb_dir,WANDB_CACHE_DIR=$wandb_cache_dir,WANDB_CONFIG_DIR=$wandb_config_dir,WANDB_ARTIFACT_DIR=$wandb_artifact_dir,A_CANARY_JOB_COMMENT=$slurm_comment,A_CANARY_SLURM_OUTPUT=$slurm_output"
container_mounts="$wandb_netrc:/run/secrets/wandb.netrc:ro,$wandb_dir:$wandb_dir:rw,$wandb_cache_dir:$wandb_cache_dir:rw,$wandb_config_dir:$wandb_config_dir:rw,$wandb_artifact_dir:$wandb_artifact_dir:rw,$target_path:$target_path:ro,$container_image:$container_image:ro"

# Validate both allocations before any scheduler mutation.
sbatch --account="$account" --partition=cpu_datamover --nodes=1 --ntasks=1 \
    --cpus-per-task=96 --export="$builder_exports" --test-only --parsable "$CPU_SCRIPT" >/dev/null
sbatch --account="$account" --partition=batch --nodes=16 --segment=16 --ntasks-per-node=1 \
    --gpus-per-node=4 --cpus-per-task=96 --container-image="$container_image" \
    --container-mounts="$container_mounts" --job-name=q4b-a-repair-canary \
    --comment="$slurm_comment" --output="$slurm_output" \
    --export="$gpu_exports" --test-only --parsable "$GPU_SCRIPT" >/dev/null
sbatch --account="$account" --partition=cpu_datamover --nodes=1 --ntasks=1 \
    --cpus-per-task=2 --output="$output_root/control/controller-%j.out" \
    --test-only --parsable "$CONTROLLER_SCRIPT" \
    --repo-root "$repo_root" --submission-receipt "$submission_receipt" \
    --job-completion "$job_completion" --scheduler-observation "$scheduler_observation" \
    --authorization "$authorization" >/dev/null

builder_job_id="" gpu_job_id=""
if [[ "$mode" == --submit ]]; then
    builder_job_id="$(sbatch --parsable --account="$account" --partition=cpu_datamover \
        --nodes=1 --ntasks=1 --cpus-per-task=96 --export="$builder_exports" "$CPU_SCRIPT")"
    dependency_id="${builder_job_id%%[_;]*}"
    [[ "$dependency_id" =~ ^[0-9]+$ ]] || exit 1
    gpu_job_id="$(sbatch --parsable --account="$account" --partition=batch \
        --dependency="afterok:${dependency_id}" --nodes=16 --segment=16 --ntasks-per-node=1 \
        --gpus-per-node=4 --cpus-per-task=96 --container-image="$container_image" \
        --container-mounts="$container_mounts" --job-name=q4b-a-repair-canary \
        --comment="$slurm_comment" --output="$slurm_output" \
        --export="$gpu_exports" "$GPU_SCRIPT")"
    gpu_id="${gpu_job_id%%[_;]*}"
    [[ "$gpu_id" =~ ^[0-9]+$ ]] || exit 1
    mkdir -p "$(dirname "$submission_receipt")"
    PYTHONPATH="$LAUNCHER_ROOT${PYTHONPATH:+:$PYTHONPATH}" python3 - \
        "$submission_receipt" "$source_commit" "$dependency_id" "$gpu_id" "$account" \
        "$slurm_comment" "$slurm_output" "$manifest" <<'PY'
import sys
from pathlib import Path
from common.specdec.qwen4b_a_canary_manifest import publish_a_submission_receipt
publish_a_submission_receipt(
    Path(sys.argv[1]), source_commit=sys.argv[2], builder_job_id=sys.argv[3],
    gpu_job_id=sys.argv[4], account=sys.argv[5], job_comment=sys.argv[6],
    stdout_path=sys.argv[7], manifest_path=Path(sys.argv[8]),
)
PY
    sbatch --parsable --account="$account" --partition=cpu_datamover \
        --dependency="afterok:${gpu_id}" --nodes=1 --ntasks=1 --cpus-per-task=2 \
        --output="$output_root/control/controller-%j.out" \
        "$CONTROLLER_SCRIPT" --repo-root "$repo_root" \
        --submission-receipt "$submission_receipt" --job-completion "$job_completion" \
        --scheduler-observation "$scheduler_observation" --authorization "$authorization"
fi
# Stable literal retained for static authorization audits: scientific_training_authorized=false
