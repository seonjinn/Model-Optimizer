#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Materialize one immutable Hugging Face artifact without exposing partial Lustre trees.
set -euo pipefail

SCRIPT_PATH="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/$(basename "${BASH_SOURCE[0]}")"
LAUNCHER_ROOT="${DRAFTER_LAUNCHER_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
MODE="submit"
CLUSTER_PROFILE=""
READINESS_RECEIPT=""
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
MATERIALIZE_CHAT_TEMPLATE=0
CPU_DATAMOVER=0

usage() {
    echo "usage: $0 (--repo ID --revision SHA --image /lustre/... --runtime-archive /lustre/... | --source-dir /lustre/... --source-id SHA [--materialize-chat-template] [--cpu-datamover]) --artifact-dir /lustre/... [--cluster-profile PATH --readiness-receipt PATH] [--scratch-root PATH] [--log-dir PATH]" >&2
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
        --cluster-profile) CLUSTER_PROFILE="$2"; shift 2 ;;
        --readiness-receipt) READINESS_RECEIPT="$2"; shift 2 ;;
        --materialize-chat-template) MATERIALIZE_CHAT_TEMPLATE=1; shift ;;
        --cpu-datamover) CPU_DATAMOVER=1; shift ;;
        --run-stage) MODE="run"; shift ;;
        *) usage ;;
    esac
done

PROFILE_DURABLE_ROOT=""
PROFILE_GPU_ARGS=()
PROFILE_EVAL_NODES=1
PROFILE_EVAL_SEGMENT=1
if [[ -n "$CLUSTER_PROFILE" || -n "$READINESS_RECEIPT" ]]; then
    [[ -n "$CLUSTER_PROFILE" && -n "$READINESS_RECEIPT" ]] || usage
    mapfile -t profile_values < <(PYTHONPATH="${LAUNCHER_ROOT}${PYTHONPATH:+:${PYTHONPATH}}" python3 - \
        "$CLUSTER_PROFILE" "$READINESS_RECEIPT" <<'PY'
import json
import sys
from pathlib import Path

from common.specdec.cluster_profile import (
    load_cluster_profile,
    scheduler_gpu_args,
    validate_scratch_root,
)

profile = load_cluster_profile(Path(sys.argv[1]).resolve())
receipt = json.loads(Path(sys.argv[2]).resolve().read_text())
expected = {
    "profile": profile.name,
    "account": profile.account,
    "partition": profile.partition,
    "pyxis_available": True,
    "gpu_count": profile.gpus_per_node,
    "architecture": "aarch64",
}
if any(receipt.get(key) != value for key, value in expected.items()):
    raise ValueError("cluster readiness receipt does not match profile")
scratch = Path(receipt.get("scratch_root", ""))
validate_scratch_root(profile, scratch)
print(profile.account)
print(profile.partition)
print(profile.durable_root)
print(scratch)
print(profile.evaluation_nodes)
print(profile.evaluation_segment)
for argument in scheduler_gpu_args(profile):
    print(argument)
PY
    )
    (( ${#profile_values[@]} >= 6 )) || { echo "invalid cluster readiness contract" >&2; exit 2; }
    ACCOUNT="${profile_values[0]}"
    PARTITION="${profile_values[1]}"
    PROFILE_DURABLE_ROOT="${profile_values[2]}"
    SCRATCH_ROOT="${profile_values[3]%/}/${USER}"
    LOG_DIR="$SCRATCH_ROOT"
    PROFILE_EVAL_NODES="${profile_values[4]}"
    PROFILE_EVAL_SEGMENT="${profile_values[5]}"
    PROFILE_GPU_ARGS=("${profile_values[@]:6}")
fi

if [[ -z "$CLUSTER_PROFILE" ]]; then
    if [[ -n "$SOURCE_DIR" ]]; then
        [[ "$SOURCE_DIR" == /lustre/* ]] || usage
    fi
    [[ "$ARTIFACT_DIR" == /lustre/* ]] || usage
    [[ "$SCRATCH_ROOT" == /raid/scratch/* ]] || usage
fi

[[ -n "$ACCOUNT" && -n "$PARTITION" ]] || usage
[[ ! -L "$ARTIFACT_DIR" ]] || { echo "refusing symlink artifact path: $ARTIFACT_DIR" >&2; exit 2; }
ARTIFACT_CANONICAL="$(realpath -m -- "$ARTIFACT_DIR")"
SCRATCH_CANONICAL="$(realpath -m -- "$SCRATCH_ROOT")"
LOG_CANONICAL="$(realpath -m -- "$LOG_DIR")"
if [[ -n "$PROFILE_DURABLE_ROOT" ]]; then
    [[ "$ARTIFACT_CANONICAL" == "$PROFILE_DURABLE_ROOT"/* ]] || usage
    PYTHONPATH="${LAUNCHER_ROOT}${PYTHONPATH:+:${PYTHONPATH}}" python3 - \
        "$CLUSTER_PROFILE" "$SCRATCH_CANONICAL" <<'PY'
import sys
from pathlib import Path
from common.specdec.cluster_profile import load_cluster_profile, validate_scratch_root
validate_scratch_root(load_cluster_profile(Path(sys.argv[1]).resolve()), Path(sys.argv[2]))
PY
    [[ "$LOG_CANONICAL" == "$SCRATCH_CANONICAL" || "$LOG_CANONICAL" == "$SCRATCH_CANONICAL"/* ]] || usage
else
    [[ "$ARTIFACT_CANONICAL" == /lustre/* ]] || usage
    [[ "$SCRATCH_CANONICAL" == /raid/scratch/* ]] || usage
    [[ "$LOG_CANONICAL" == /raid/scratch || "$LOG_CANONICAL" == /raid/scratch/* ]] || usage
    PROFILE_GPU_ARGS=(--gpus-per-node=4)
fi
ARTIFACT_DIR="$ARTIFACT_CANONICAL"
SCRATCH_ROOT="$SCRATCH_CANONICAL"
LOG_DIR="$LOG_CANONICAL"

if [[ -n "$SOURCE_DIR" || -n "$SOURCE_ID" ]]; then
    [[ -z "$REPOSITORY" && -z "$REVISION" ]] || usage
    [[ "$SOURCE_ID" =~ ^[0-9a-f]{40}$ ]] || usage
    SOURCE_CANONICAL="$(realpath -m -- "$SOURCE_DIR")"
    if [[ -n "$PROFILE_DURABLE_ROOT" ]]; then
        [[ "$SOURCE_CANONICAL" == "$PROFILE_DURABLE_ROOT"/* ]] || usage
    else
        [[ "$SOURCE_CANONICAL" == /lustre/* ]] || usage
    fi
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
if (( MATERIALIZE_CHAT_TEMPLATE || CPU_DATAMOVER )); then
    [[ "$SOURCE_KIND" == "local" ]] || usage
fi
if (( CPU_DATAMOVER )); then
    [[ -z "$CLUSTER_PROFILE" ]] || usage
    PARTITION="cpu_datamover"
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

run_stage() (
    require_inputs
    if (( CPU_DATAMOVER )); then
        [[ "${SLURM_JOB_PARTITION:-}" == "cpu_datamover" ]] || {
            echo "local model staging requires cpu_datamover" >&2
            return 2
        }
        [[ "${SLURM_CPUS_PER_TASK:-}" == "96" ]] || {
            echo "local model staging requires exactly 96 CPUs" >&2
            return 2
        }
        [[ -z "${SLURM_JOB_GPUS:-}" && -z "${SLURM_GPUS_ON_NODE:-}" ]] || {
            echo "local model staging must not allocate GPUs" >&2
            return 2
        }
    fi
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
    # shellcheck disable=SC2329 # Invoked indirectly by the EXIT trap.
    cleanup() {
        find "$work_dir" -depth -delete 2>/dev/null || true
        if [[ -e "$partial" ]]; then
            find "$partial" -depth -delete 2>/dev/null || true
        fi
    }
    trap cleanup EXIT

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
    if (( MATERIALIZE_CHAT_TEMPLATE )); then
        python3 - "$local_snapshot" <<'PY'
import json
import os
import stat
import sys
from pathlib import Path

root = Path(sys.argv[1])
config_path = root / "tokenizer_config.json"
template_path = root / "chat_template.jinja"
config_metadata = os.lstat(config_path)
if stat.S_ISLNK(config_metadata.st_mode) or not stat.S_ISREG(config_metadata.st_mode):
    raise ValueError("tokenizer configuration is not a no-follow regular file")
config = json.loads(config_path.read_bytes())
template = config.get("chat_template") if isinstance(config, dict) else None
if not isinstance(template, str) or not template or "\0" in template:
    raise ValueError("tokenizer configuration has no materializable chat template")
payload = template.encode("utf-8")
if os.path.lexists(template_path):
    metadata = os.lstat(template_path)
    if (
        stat.S_ISLNK(metadata.st_mode)
        or not stat.S_ISREG(metadata.st_mode)
        or template_path.read_bytes() != payload
    ):
        raise ValueError("existing chat template differs from tokenizer configuration")
else:
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(template_path, flags, 0o600)
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
descriptor = os.open(root, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
try:
    os.fsync(descriptor)
finally:
    os.close(descriptor)
PY
    fi

    # shellcheck disable=SC2016 # The child shell expands these file-specific variables.
    find "$local_snapshot" -type f -print0 | xargs -0 -r -P 4 -n 1 sh -c '
        source_file="$3"
        relative_path="${source_file#"$1"/}"
        target_file="$2/$relative_path"
        mkdir -p "$(dirname "$target_file")"
        cp --reflink=auto --preserve=mode,timestamps "$source_file" "$target_file"
    ' sh "$local_snapshot" "$partial"
    printf '{\n  "source_kind": "%s",\n  "source_identity": "%s"\n}\n' "$SOURCE_KIND" "$SOURCE_IDENTITY" >"${partial}/snapshot-manifest.json"
    cp "${partial}/snapshot-manifest.json" "${partial}/completion.json"

    mv -T --no-clobber "$partial" "$ARTIFACT_DIR"
    [[ ! -e "$partial" ]] || { echo "another publisher created artifact: $ARTIFACT_DIR" >&2; return 1; }
)

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

    local args
    if (( CPU_DATAMOVER )); then
        args=(--account="$ACCOUNT" --partition="$PARTITION" --nodes=1
            --ntasks=1 --cpus-per-task=96)
    elif [[ -n "$CLUSTER_PROFILE" ]]; then
        args=(--account="$ACCOUNT" --partition="$PARTITION" --nodes="$PROFILE_EVAL_NODES"
            --ntasks-per-node=1 "${PROFILE_GPU_ARGS[@]}" --segment="$PROFILE_EVAL_SEGMENT")
    else
        args=(--account="$ACCOUNT" --partition="$PARTITION" --nodes=1
            --ntasks-per-node=1 --gpus-per-node=4 --segment=1)
    fi
    args+=(--time=01:00:00 --job-name="$name" --output="${LOG_DIR%/}/%x-%j.out")
    local command=(
        "$SCRIPT_PATH" --run-stage --artifact-dir "$ARTIFACT_DIR" --scratch-root "$SCRATCH_ROOT"
        --account "$ACCOUNT" --partition "$PARTITION" --log-dir "$LOG_DIR"
    )
    if [[ "$SOURCE_KIND" == "local" ]]; then
        command+=(--source-dir "$SOURCE_DIR" --source-id "$SOURCE_ID")
    else
        command+=(--repo "$REPOSITORY" --revision "$REVISION" --image "$IMAGE" --runtime-archive "$RUNTIME_ARCHIVE")
    fi
    (( MATERIALIZE_CHAT_TEMPLATE )) && command+=(--materialize-chat-template)
    (( CPU_DATAMOVER )) && command+=(--cpu-datamover)
    if [[ -n "$CLUSTER_PROFILE" ]]; then
        args+=("--export=ALL,DRAFTER_LAUNCHER_ROOT=$LAUNCHER_ROOT")
        command+=(--cluster-profile "$CLUSTER_PROFILE" --readiness-receipt "$READINESS_RECEIPT")
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
