#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --segment=1
#SBATCH --time=01:55:00

# Export mid-training drafter checkpoints into vLLM-servable drafts.
#
# dflash_online_training.sh only exports after a run finishes, so a run that is
# still training (or that was cut off at the wall clock) leaves nothing the
# speculators harness can serve. This job runs the same exporter against
# checkpoints picked by hand, without waiting for the training job to end.
#
# The trainer saves the draft beside a FakeBaseModel, so the checkpoint carries
# everything the exporter reads (draft weights, embed_tokens, lm_head) and no
# real target has to be staged.
#
# Inputs (all required):
#   EXPORT_CELLS      newline-separated "label|checkpoint|export_dir" records.
#   SOURCE_PATH       ModelOpt checkout to run from.
#   SOURCE_SHA        Expected HEAD of that checkout.
#   SOURCE_DISTRIBUTION_VERSION  Version to write into the synthesized egg-info.
#   IMAGE_PATH        Pinned container.
#   RUNTIME_ARCHIVE   Relocatable venv tarball.
#   DURABLE_ROOT      Shared root holding the checkpoints and the exports.
#   EXPORT_LOG_ROOT   Where per-cell logs land.

set -euo pipefail

for name in EXPORT_CELLS SOURCE_PATH SOURCE_SHA SOURCE_DISTRIBUTION_VERSION \
    IMAGE_PATH RUNTIME_ARCHIVE DURABLE_ROOT EXPORT_LOG_ROOT; do
    [[ -n "${!name:-}" ]] || { echo "ERROR: ${name} is required" >&2; exit 2; }
done

SCRATCH_ROOT="${SCRATCH_ROOT:-/raid/scratch}"
SCRATCH_JOB_ROOT="${SCRATCH_ROOT%/}/${SLURM_JOB_ID}"

[[ "$(git -C "$SOURCE_PATH" rev-parse HEAD)" == "$SOURCE_SHA" ]] || {
    echo "ERROR: ${SOURCE_PATH} is not at ${SOURCE_SHA}" >&2
    exit 2
}
[[ "$EXPORT_LOG_ROOT" == "$DURABLE_ROOT"/* ]] || {
    echo "ERROR: EXPORT_LOG_ROOT must live under DURABLE_ROOT" >&2
    exit 2
}
mkdir -p "$EXPORT_LOG_ROOT"

while IFS="|" read -r label checkpoint export_dir; do
    [[ -n "$label" ]] || continue
    [[ "$label" =~ ^[a-z0-9][a-z0-9._-]{0,63}$ ]] || {
        echo "ERROR: unsafe label: ${label}" >&2; exit 2; }
    [[ -r "${checkpoint}/config.json" ]] || {
        echo "ERROR: ${checkpoint} has no config.json" >&2; exit 2; }
    [[ -r "${checkpoint}/modelopt_state.pth" ]] || {
        echo "ERROR: ${checkpoint} has no modelopt_state.pth" >&2; exit 2; }
    [[ "$export_dir" == "$DURABLE_ROOT"/* ]] || {
        echo "ERROR: export_dir must live under DURABLE_ROOT: ${export_dir}" >&2; exit 2; }
    [[ ! -e "$export_dir" ]] || {
        echo "ERROR: export_dir already exists: ${export_dir}" >&2; exit 2; }
done <<<"$EXPORT_CELLS"

export SCRATCH_ROOT SCRATCH_JOB_ROOT SOURCE_PATH SOURCE_DISTRIBUTION_VERSION \
    RUNTIME_ARCHIVE EXPORT_LOG_ROOT EXPORT_CELLS

# shellcheck disable=SC2016 # The child shell expands its own node-local paths.
srun --nodes=1 --ntasks=1 --ntasks-per-node=1 bash -c '
    set -euo pipefail
    node_root="${SCRATCH_JOB_ROOT}/node-0"
    [[ "$node_root" == "${SCRATCH_ROOT%/}/${SLURM_JOB_ID}/node-0" ]] || {
        echo "refusing to clean unexpected scratch path: $node_root" >&2
        exit 2
    }
    rm -rf -- "$node_root"
    mkdir -p "$node_root/runtime" "$node_root/hf/hub" "$node_root/tmp"
    # Staging copies this tree onto every node of every arm at once. .git is three
    # quarters of the bytes and the job never reads it. Worse, any git command in
    # the pinned tree creates and removes .git/index.lock, and a copy that
    # enumerates the directory before the lock is gone but opens it after fails
    # with ENOENT, which kills the staging task and takes the whole step down with
    # it. Excluding .git removes both the cost and the race. Keep apostrophes out
    # of this block: in some callers it sits inside a single-quoted bash -c body.
    mkdir -p "$node_root/source"
    tar --create --file=- --exclude=./.git --directory="$SOURCE_PATH" . \
        | tar --extract --file=- --directory="$node_root/source"
    egg_info="$node_root/source/nvidia_modelopt.egg-info"
    mkdir -p "$egg_info"
    printf "Metadata-Version: 2.4\nName: nvidia-modelopt\nVersion: %s\n" \
        "$SOURCE_DISTRIBUTION_VERSION" >"$egg_info/PKG-INFO"
    printf "modelopt\n" >"$egg_info/top_level.txt"
    tar --extract --file="$RUNTIME_ARCHIVE" --directory="$node_root/runtime"
    old_venv=$(sed -nE "s/^[[:space:]]*export[[:space:]]+VIRTUAL_ENV=(.*)$/\1/p" "$node_root/runtime/bin/activate" | head -n 1)
    old_venv="${old_venv#\"}"
    old_venv="${old_venv%\"}"
    [[ -n "$old_venv" ]] || { echo "runtime archive has no VIRTUAL_ENV" >&2; exit 2; }
    grep -IlZ "$old_venv" "$node_root/runtime/bin"/* "$node_root/runtime/pyvenv.cfg" 2>/dev/null \
        | xargs -0 -r sed -i "s|$old_venv|$node_root/runtime|g"
    echo "staging complete"
'

CONTAINER_MOUNTS="${SCRATCH_ROOT}:${SCRATCH_ROOT},${DURABLE_ROOT}:${DURABLE_ROOT}"

# shellcheck disable=SC2016 # The child shell expands its own node-local paths.
srun --nodes=1 --ntasks=1 --ntasks-per-node=1 --kill-on-bad-exit=1 \
    --no-container-mount-home --container-image="$IMAGE_PATH" \
    --container-mounts="$CONTAINER_MOUNTS" bash -c '
    set -euo pipefail
    node_root="${SCRATCH_JOB_ROOT}/node-0"
    export VIRTUAL_ENV="$node_root/runtime"
    export PATH="$VIRTUAL_ENV/bin:$PATH"
    export PYTHONPATH="$node_root/source${PYTHONPATH:+:$PYTHONPATH}"
    export HF_HOME="$node_root/hf"
    export HF_HUB_CACHE="$HF_HOME/hub"
    export HF_HUB_OFFLINE=1
    export TRITON_CACHE_DIR="$node_root/triton"
    export XDG_CACHE_HOME="$node_root/xdg"
    export TMPDIR="$node_root/tmp"
    export CUDA_VISIBLE_DEVICES=0

    status=0
    while IFS="|" read -r label checkpoint export_dir; do
        [[ -n "$label" ]] || continue
        log="${EXPORT_LOG_ROOT}/${label}.log"
        echo "== ${label}  checkpoint=${checkpoint}  export_dir=${export_dir}"
        # Export into a staging path so a crash cannot leave a half-written draft
        # that later looks like a finished export.
        staging="${export_dir}.partial"
        rm -rf -- "$staging"
        if python3 "$node_root/source/examples/speculative_decoding/scripts/export_hf_checkpoint.py" \
            --model_path "$checkpoint" \
            --export_path "$staging" \
            --trust_remote_code >"$log" 2>&1; then
            mv -- "$staging" "$export_dir"
            ls -l "$export_dir" | tail -n 20
        else
            status=1
            rm -rf -- "$staging"
            echo "FAILED: ${label} (see ${log})" >&2
            tail -n 40 "$log" >&2
        fi
    done <<<"$EXPORT_CELLS"
    exit "$status"
'
