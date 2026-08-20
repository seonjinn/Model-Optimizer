#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Stage one immutable Hugging Face snapshot without exposing partial Lustre trees.
set -euo pipefail

SCRIPT_PATH="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/$(basename "${BASH_SOURCE[0]}")"
MODE="submit"
REPOSITORY=""
REVISION=""
ARTIFACT_DIR=""
SCRATCH_ROOT="/raid/scratch"
IMAGE=""
RUNTIME_ARCHIVE=""
ACCOUNT="nemotron_n3_post"
PARTITION="batch"
LOG_DIR=""

usage() {
    echo "usage: $0 --repo ID --revision SHA --artifact-dir /lustre/... --image /lustre/... --runtime-archive /lustre/... --log-dir /lustre/... [--scratch-root /raid/scratch]" >&2
    exit 2
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --repo) REPOSITORY="$2"; shift 2 ;;
        --revision) REVISION="$2"; shift 2 ;;
        --artifact-dir) ARTIFACT_DIR="$2"; shift 2 ;;
        --scratch-root) SCRATCH_ROOT="$2"; shift 2 ;;
        --image) IMAGE="$2"; shift 2 ;;
        --runtime-archive) RUNTIME_ARCHIVE="$2"; shift 2 ;;
        --account) ACCOUNT="$2"; shift 2 ;;
        --partition) PARTITION="$2"; shift 2 ;;
        --log-dir) LOG_DIR="$2"; shift 2 ;;
        --run-stage) MODE="run"; shift ;;
        *) usage ;;
    esac
done

[[ -n "$REPOSITORY" && "$REVISION" =~ ^[0-9a-f]{40}$ ]] || usage
[[ "$ARTIFACT_DIR" == /lustre/* && "$IMAGE" == /lustre/* && "$RUNTIME_ARCHIVE" == /lustre/* && "$LOG_DIR" == /lustre/* ]] || usage
[[ "$SCRATCH_ROOT" == /raid/scratch* ]] || usage

job_name() {
    printf 'stage-hf-%s-%s' "$(printf '%s' "$REPOSITORY" | tr '/_' '--')" "${REVISION:0:12}"
}

require_pinned_artifacts() {
    [[ -f "$IMAGE" ]] || { echo "missing pinned image: $IMAGE" >&2; exit 2; }
    [[ -f "$RUNTIME_ARCHIVE" ]] || { echo "missing pinned runtime archive: $RUNTIME_ARCHIVE" >&2; exit 2; }
}

run_stage() {
    require_pinned_artifacts
    local work_dir="${SCRATCH_ROOT}/${SLURM_JOB_ID:?SLURM_JOB_ID is required}"
    local local_snapshot="${work_dir}/snapshot"
    local partial="${ARTIFACT_DIR}.partial-${SLURM_JOB_ID}"
    local marker="${ARTIFACT_DIR}/completion.json"
    mkdir -p "$work_dir" "$LOG_DIR"
    export HF_HOME="${work_dir}/hf-home"
    export HF_HUB_CACHE="${HF_HOME}/hub"
    export HF_DATASETS_CACHE="${HF_HOME}/datasets"
    export LOCK_ROOT="${work_dir}/locks"
    export XDG_CACHE_HOME="${work_dir}/xdg-cache"
    export TMPDIR="${work_dir}/tmp"
    mkdir -p "$HF_HUB_CACHE" "$HF_DATASETS_CACHE" "$LOCK_ROOT" "$XDG_CACHE_HOME" "$TMPDIR"

    if [[ -f "$marker" ]] && grep -Fq "\"revision\": \"${REVISION}\"" "$marker"; then
        echo "completed snapshot already present: $ARTIFACT_DIR"
        return 0
    fi
    rm -rf "$local_snapshot" "$partial"
    mkdir -p "$local_snapshot" "$partial"
    export STAGE_REPOSITORY="$REPOSITORY" STAGE_REVISION="$REVISION"
    srun --nodes=1 --ntasks=1 --container-image="$IMAGE" --container-mounts="${work_dir}:/stage" \
        python3 -c 'from huggingface_hub import snapshot_download; import os; snapshot_download(repo_id=os.environ["STAGE_REPOSITORY"], revision=os.environ["STAGE_REVISION"], local_dir="/stage/snapshot", local_dir_use_symlinks=False)'

    # shellcheck disable=SC2016 # The child shell expands these file-specific variables.
    find "$local_snapshot" -type f -print0 | xargs -0 -r -P 4 -n 1 sh -c '
        source_file="$1"
        relative_path="${source_file#"$2"/}"
        target_file="$3/$relative_path"
        mkdir -p "$(dirname "$target_file")"
        cp --reflink=auto --preserve=mode,timestamps "$source_file" "$target_file"
    ' sh {} "$local_snapshot" "$partial"
    printf '{\n  "repository": "%s",\n  "revision": "%s"\n}\n' "$REPOSITORY" "$REVISION" >"${partial}/snapshot-manifest.json"
    mv "$partial" "$ARTIFACT_DIR"
    mv "${ARTIFACT_DIR}/snapshot-manifest.json" "$marker"
}

submit() {
    require_pinned_artifacts
    [[ "$SCRIPT_PATH" == /home/* ]] || { echo "staging script must run from /home source" >&2; exit 2; }
    mkdir -p "$LOG_DIR"
    local name
    name="$(job_name)"
    local existing
    existing="$(squeue -h -n "$name" -o "%A" | head -n 1 || true)"
    if [[ -n "$existing" ]]; then
        printf '{"job_id":"%s","status":"already-queued"}\n' "$existing"
        return 0
    fi
    local args=(--account="$ACCOUNT" --partition="$PARTITION" --nodes=1 --ntasks-per-node=1 --gpus-per-node=1 --segment=1 --job-name="$name" --output="${LOG_DIR}/%x-%j.out")
    sbatch --test-only "${args[@]}" "$SCRIPT_PATH" --run-stage --repo "$REPOSITORY" --revision "$REVISION" --artifact-dir "$ARTIFACT_DIR" --scratch-root "$SCRATCH_ROOT" --image "$IMAGE" --runtime-archive "$RUNTIME_ARCHIVE" --account "$ACCOUNT" --partition "$PARTITION" --log-dir "$LOG_DIR"
    local submitted
    submitted="$(sbatch --parsable "${args[@]}" "$SCRIPT_PATH" --run-stage --repo "$REPOSITORY" --revision "$REVISION" --artifact-dir "$ARTIFACT_DIR" --scratch-root "$SCRATCH_ROOT" --image "$IMAGE" --runtime-archive "$RUNTIME_ARCHIVE" --account "$ACCOUNT" --partition "$PARTITION" --log-dir "$LOG_DIR" || true)"
    local job_id="${submitted%%;*}"
    if [[ -z "$job_id" ]]; then
        job_id="$(squeue -h -n "$name" -o "%A" | head -n 1 || true)"
    fi
    [[ -n "$job_id" ]] || { echo "scheduler did not confirm staging submission" >&2; exit 1; }
    printf '{"job_id":"%s","status":"submitted","repository":"%s","revision":"%s"}\n' "$job_id" "$REPOSITORY" "$REVISION"
}

if [[ "$MODE" == "run" ]]; then
    run_stage
else
    submit
fi
