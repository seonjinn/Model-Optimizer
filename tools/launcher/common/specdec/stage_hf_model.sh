#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Materialize one immutable Hugging Face artifact without exposing partial Lustre trees.
set -euo pipefail

SCRIPT_PATH="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/$(basename "${BASH_SOURCE[0]}")"
MODE="submit"
REPOSITORY=""
REVISION=""
SOURCE_DIR=""
SOURCE_ID=""
SOURCE_KIND=""
SOURCE_IDENTITY=""
SOURCE_LABEL=""
ARTIFACT_DIR=""
SCRATCH_ROOT="/raid/scratch/${USER}"
IMAGE=""
RUNTIME_ARCHIVE=""
ACCOUNT="nemotron_n3_post"
PARTITION="batch"
LOG_DIR="/raid/scratch"

usage() {
    echo "usage: $0 (--repo ID --revision SHA --image /lustre/... --runtime-archive /lustre/... | --source-dir /lustre/... --source-id SHA) --artifact-dir /lustre/... [--scratch-root /raid/scratch/...] [--log-dir /raid/scratch/...]" >&2
    exit 2
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --repo) REPOSITORY="$2"; shift 2 ;;
        --revision) REVISION="$2"; shift 2 ;;
        --source-dir) SOURCE_DIR="$2"; shift 2 ;;
        --source-id) SOURCE_ID="$2"; shift 2 ;;
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

[[ "$ARTIFACT_DIR" == /lustre/* ]] || usage
[[ "$SCRATCH_ROOT" == /raid/scratch/* ]] || usage
[[ "$LOG_DIR" == /raid/scratch* ]] || usage
[[ -n "$ACCOUNT" && -n "$PARTITION" ]] || usage
ARTIFACT_CANONICAL="$(realpath -m -- "$ARTIFACT_DIR")"
SCRATCH_CANONICAL="$(realpath -m -- "$SCRATCH_ROOT")"
LOG_CANONICAL="$(realpath -m -- "$LOG_DIR")"
[[ "$ARTIFACT_CANONICAL" == /lustre/* ]] || usage
[[ "$SCRATCH_CANONICAL" == /raid/scratch/* ]] || usage
[[ "$LOG_CANONICAL" == /raid/scratch || "$LOG_CANONICAL" == /raid/scratch/* ]] || usage
ARTIFACT_DIR="$ARTIFACT_CANONICAL"
SCRATCH_ROOT="$SCRATCH_CANONICAL"
LOG_DIR="$LOG_CANONICAL"

if [[ -n "$SOURCE_DIR" || -n "$SOURCE_ID" ]]; then
    [[ -z "$REPOSITORY" && -z "$REVISION" ]] || usage
    [[ "$SOURCE_DIR" == /lustre/* ]] || usage
    [[ "$SOURCE_ID" =~ ^[0-9a-f]{40}$ ]] || usage
    SOURCE_CANONICAL="$(realpath -m -- "$SOURCE_DIR")"
    [[ "$SOURCE_CANONICAL" == /lustre/* ]] || usage
    [[ "$ARTIFACT_CANONICAL" != "$SOURCE_CANONICAL" ]] || usage
    [[ "$ARTIFACT_CANONICAL" != "$SOURCE_CANONICAL"/* ]] || usage
    [[ "$SOURCE_CANONICAL" != "$ARTIFACT_CANONICAL"/* ]] || usage
    SOURCE_DIR="$SOURCE_CANONICAL"
    SOURCE_KIND="local"
    SOURCE_IDENTITY="$SOURCE_ID"
    SOURCE_LABEL="$(basename "$SOURCE_DIR")"
else
    [[ -n "$REPOSITORY" && "$REVISION" =~ ^[0-9a-f]{40}$ && "$IMAGE" == /lustre/* && "$RUNTIME_ARCHIVE" == /lustre/* ]] || usage
    SOURCE_KIND="hub"
    SOURCE_IDENTITY="$REVISION"
    SOURCE_LABEL="${REPOSITORY//\//-}"
fi

job_name() {
    printf 'stage-hf-%s-%s' "$(printf '%s' "$SOURCE_LABEL" | tr -c '[:alnum:].-' '-')" "${SOURCE_IDENTITY:0:12}"
}

require_inputs() {
    if [[ "$SOURCE_KIND" == "local" ]]; then
        [[ -d "$SOURCE_DIR" ]] || { echo "missing local source: $SOURCE_DIR" >&2; exit 2; }
        return
    fi
    [[ -f "$IMAGE" ]] || { echo "missing pinned image: $IMAGE" >&2; exit 2; }
    [[ -f "$RUNTIME_ARCHIVE" ]] || { echo "missing pinned runtime archive: $RUNTIME_ARCHIVE" >&2; exit 2; }
}

completed_artifact_matches() {
    local marker="${ARTIFACT_DIR}/completion.json"
    [[ -f "$marker" ]] \
        && grep -Fq "\"source_kind\": \"${SOURCE_KIND}\"" "$marker" \
        && grep -Fq "\"source_identity\": \"${SOURCE_IDENTITY}\"" "$marker"
}

run_stage() {
    require_inputs
    local work_dir="${SCRATCH_ROOT}/${SLURM_JOB_ID:?SLURM_JOB_ID is required}/hf-stage"
    local local_snapshot="${work_dir}/snapshot"
    local partial="${ARTIFACT_DIR}.partial-${SLURM_JOB_ID}"

    if completed_artifact_matches; then
        echo "completed artifact already present: $ARTIFACT_DIR"
        return 0
    fi
    [[ ! -e "$ARTIFACT_DIR" && ! -L "$ARTIFACT_DIR" ]] || { echo "refusing to replace existing artifact: $ARTIFACT_DIR" >&2; return 1; }
    [[ ! -e "$partial" && ! -L "$partial" ]] || { echo "refusing to replace existing partial: $partial" >&2; return 1; }

    mkdir -p "$local_snapshot" "$(dirname "$ARTIFACT_DIR")"
    mkdir "$partial"
    cleanup() {
        find "$work_dir" -depth -delete 2>/dev/null || true
        if [[ -e "$partial" ]]; then
            find "$partial" -depth -delete 2>/dev/null || true
        fi
    }
    trap cleanup RETURN

    if [[ "$SOURCE_KIND" == "local" ]]; then
        cp -aL "$SOURCE_DIR"/. "$local_snapshot"/
    else
        export HF_HOME="${work_dir}/hf-home"
        export HF_HUB_CACHE="${HF_HOME}/hub"
        export HF_DATASETS_CACHE="${HF_HOME}/datasets"
        export LOCK_ROOT="${work_dir}/locks"
        export XDG_CACHE_HOME="${work_dir}/xdg-cache"
        export TMPDIR="${work_dir}/tmp"
        mkdir -p "$HF_HUB_CACHE" "$HF_DATASETS_CACHE" "$LOCK_ROOT" "$XDG_CACHE_HOME" "$TMPDIR"
        export STAGE_REPOSITORY="$REPOSITORY" STAGE_REVISION="$REVISION"
        srun --nodes=1 --ntasks=1 --container-image="$IMAGE" --container-mounts="${work_dir}:/stage" \
            python3 -c 'from huggingface_hub import snapshot_download; import os; snapshot_download(repo_id=os.environ["STAGE_REPOSITORY"], revision=os.environ["STAGE_REVISION"], local_dir="/stage/snapshot", local_dir_use_symlinks=False)'
    fi

    # shellcheck disable=SC2016 # The child shell expands these file-specific variables.
    find "$local_snapshot" -type f -print0 | xargs -0 -r -P 4 -n 1 sh -c '
        source_file="$1"
        relative_path="${source_file#"$2"/}"
        target_file="$3/$relative_path"
        mkdir -p "$(dirname "$target_file")"
        cp --reflink=auto --preserve=mode,timestamps "$source_file" "$target_file"
    ' sh {} "$local_snapshot" "$partial"
    printf '{\n  "source_kind": "%s",\n  "source_identity": "%s"\n}\n' "$SOURCE_KIND" "$SOURCE_IDENTITY" >"${partial}/snapshot-manifest.json"
    cp "${partial}/snapshot-manifest.json" "${partial}/completion.json"

    mv -T --no-clobber "$partial" "$ARTIFACT_DIR"
    [[ ! -e "$partial" ]] || { echo "another publisher created artifact: $ARTIFACT_DIR" >&2; return 1; }
}

submit() {
    require_inputs
    [[ "$SCRIPT_PATH" == /home/* ]] || { echo "staging script must run from /home source" >&2; exit 2; }
    local name
    name="$(job_name)"
    local existing
    existing="$(squeue -h -n "$name" -u "$USER" -o "%A" | head -n 1 || true)"
    if [[ -n "$existing" ]]; then
        printf '{"job_id":"%s","status":"already-queued"}\n' "$existing"
        return 0
    fi

    local args=(
        --account="$ACCOUNT" --partition="$PARTITION" --nodes=1 --ntasks-per-node=1
        --gpus-per-node=4 --segment=1 --time=01:00:00 --job-name="$name"
        --output="${LOG_DIR%/}/%x-%j.out"
    )
    local command=(
        "$SCRIPT_PATH" --run-stage --artifact-dir "$ARTIFACT_DIR" --scratch-root "$SCRATCH_ROOT"
        --account "$ACCOUNT" --partition "$PARTITION" --log-dir "$LOG_DIR"
    )
    if [[ "$SOURCE_KIND" == "local" ]]; then
        command+=(--source-dir "$SOURCE_DIR" --source-id "$SOURCE_ID")
    else
        command+=(--repo "$REPOSITORY" --revision "$REVISION" --image "$IMAGE" --runtime-archive "$RUNTIME_ARCHIVE")
    fi

    sbatch --test-only "${args[@]}" "${command[@]}"
    local submitted
    submitted="$(sbatch --parsable "${args[@]}" "${command[@]}" || true)"
    local job_id="${submitted%%;*}"
    if [[ -z "$job_id" ]]; then
        job_id="$(squeue -h -n "$name" -u "$USER" -o "%A" | head -n 1 || true)"
    fi
    [[ -n "$job_id" ]] || { echo "scheduler did not confirm staging submission" >&2; exit 1; }
    printf '{"job_id":"%s","status":"submitted","source_kind":"%s","source_identity":"%s"}\n' "$job_id" "$SOURCE_KIND" "$SOURCE_IDENTITY"
}

if [[ "$MODE" == "run" ]]; then
    run_stage
else
    submit
fi
