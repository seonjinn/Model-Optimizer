#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#SBATCH -N 4
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --segment=4
#SBATCH --time=03:55:00

# Execute one pinned streaming-drafter wave. The submitter supplies all inputs.
set -euo pipefail

readonly SERVE_NODES=2
readonly TRAINER_NODES=2
readonly GPUS_PER_NODE=4
readonly GLOBAL_BATCH_SIZE=512
readonly SCRATCH_JOB_ROOT="/raid/scratch/${SLURM_JOB_ID}"
LAUNCHER_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
readonly LAUNCHER_ROOT

for name in MANIFEST_PATH EXPERIMENT_INDEX MAX_STEPS; do
    [[ -n "${!name:-}" ]] || { echo "missing required environment: $name" >&2; exit 2; }
done
mapfile -t manifest_values < <(PYTHONPATH="$LAUNCHER_ROOT${PYTHONPATH:+:$PYTHONPATH}" python3 - "$MANIFEST_PATH" "$EXPERIMENT_INDEX" <<'PY'
import sys
from pathlib import Path

manifest = Path(sys.argv[1])
from common.specdec.drafter_job_manifest import load_manifest

experiment = load_manifest(manifest)[int(sys.argv[2])]
for value in (
    experiment.paths.source_path, experiment.paths.source_sha, experiment.paths.image_path, experiment.paths.runtime_archive_path,
    experiment.paths.runtime_archive_sha256,
    experiment.paths.target_path, experiment.paths.dataset_path, experiment.paths.output_root,
    experiment.method, experiment.block_size, experiment.run_name, experiment.sample_size,
    ",".join(map(str, experiment.topology.capture_ids)), experiment.topology.serve_tp,
    experiment.topology.per_device_train_batch_size, experiment.topology.gradient_accumulation_steps,
    experiment.topology.num_attention_heads, experiment.topology.num_key_value_heads,
    experiment.topology.head_dim, experiment.topology.intermediate_size,
):
    print(value)
PY
)
(( ${#manifest_values[@]} == 20 )) || { echo "manifest entry is unavailable" >&2; exit 2; }
SOURCE_PATH="${manifest_values[0]}"
SOURCE_SHA="${manifest_values[1]}"
IMAGE_PATH="${manifest_values[2]}"
RUNTIME_ARCHIVE="${manifest_values[3]}"
RUNTIME_ARCHIVE_SHA256="${manifest_values[4]}"
TARGET_PATH="${manifest_values[5]}"
DATASET_PATH="${manifest_values[6]}"
DATASET_LOCAL_NAME="$(basename "$DATASET_PATH")"
OUTPUT_ROOT="${manifest_values[7]}"
METHOD="${manifest_values[8]}"
BLOCK_SIZE="${manifest_values[9]}"
RUN_NAME="${manifest_values[10]}"
SAMPLE_SIZE="${manifest_values[11]}"
EAGLE_CAPTURE_IDS="[${manifest_values[12]}]"
SERVE_TP="${manifest_values[13]}"
PER_DEVICE_TRAIN_BATCH_SIZE="${manifest_values[14]}"
GRADIENT_ACCUMULATION_STEPS="${manifest_values[15]}"
NUM_ATTENTION_HEADS="${manifest_values[16]}"
NUM_KEY_VALUE_HEADS="${manifest_values[17]}"
HEAD_DIM="${manifest_values[18]}"
INTERMEDIATE_SIZE="${manifest_values[19]}"
for name in SOURCE_PATH SOURCE_SHA RUNTIME_ARCHIVE TARGET_PATH DATASET_PATH OUTPUT_ROOT METHOD BLOCK_SIZE RUN_NAME; do
    [[ -n "${!name:-}" ]] || { echo "missing manifest value: $name" >&2; exit 2; }
done
[[ "$SOURCE_PATH" == /home/* && "$IMAGE_PATH" == /lustre/* && "$RUNTIME_ARCHIVE" == /lustre/* && "$TARGET_PATH" == /lustre/* && "$DATASET_PATH" == /lustre/* && "$OUTPUT_ROOT" == /lustre/* ]] || {
    echo "source must be /home and durable artifacts must be /lustre" >&2; exit 2;
}
[[ "$SOURCE_SHA" =~ ^[0-9a-f]{40}$ ]] || { echo "SOURCE_SHA must be a full commit SHA" >&2; exit 2; }
[[ "$(git -C "$SOURCE_PATH" rev-parse HEAD)" == "$SOURCE_SHA" ]] || { echo "source SHA mismatch" >&2; exit 2; }
[[ -z "$(git -C "$SOURCE_PATH" status --porcelain)" ]] || { echo "source checkout is dirty" >&2; exit 2; }
[[ -f "$RUNTIME_ARCHIVE" ]] || { echo "runtime archive is missing" >&2; exit 2; }
[[ "$(sha256sum "$RUNTIME_ARCHIVE" | cut -d' ' -f1)" == "$RUNTIME_ARCHIVE_SHA256" ]] || { echo "runtime archive SHA-256 mismatch" >&2; exit 2; }
[[ "$MAX_STEPS" =~ ^[1-9][0-9]*$ ]] || { echo "MAX_STEPS must be positive" >&2; exit 2; }
[[ $((PER_DEVICE_TRAIN_BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS * TRAINER_NODES * GPUS_PER_NODE)) -eq "$GLOBAL_BATCH_SIZE" ]] || { echo "manifest batch arithmetic must equal GBS 512" >&2; exit 2; }
CONTROL_ROOT="${OUTPUT_ROOT}/control"
mkdir -p "$OUTPUT_ROOT" "$CONTROL_ROOT"

case "${METHOD}:${BLOCK_SIZE}" in
    dflash:8|dflash:16)
        NUM_SPEC_TOKENS=$((BLOCK_SIZE - 1))
        LOSS_OBJECTIVE=dpace
        LOSS_VALUE=0
        DFLASH_DPACE_ALPHA=0.5
        ;;
    dspark:8)
        NUM_SPEC_TOKENS=8
        LOSS_OBJECTIVE=decay
        LOSS_VALUE=4
        DFLASH_DPACE_ALPHA=""
        ;;
    dspark:16)
        NUM_SPEC_TOKENS=16
        LOSS_OBJECTIVE=decay
        LOSS_VALUE=7
        DFLASH_DPACE_ALPHA=""
        ;;
    *) echo "unsupported method/block size: ${METHOD}/${BLOCK_SIZE}" >&2; exit 2 ;;
esac
export SERVE_NODES NUM_SPEC_TOKENS

export SOURCE_PATH SOURCE_SHA IMAGE_PATH RUNTIME_ARCHIVE TARGET_PATH DATASET_PATH DATASET_LOCAL_NAME OUTPUT_ROOT SCRATCH_JOB_ROOT
# shellcheck disable=SC2016 # The child shell expands node-local placement variables.
srun --nodes=4 --ntasks=4 --ntasks-per-node=1 --no-container-mount-home --container-image="$IMAGE_PATH" --container-mounts="${SOURCE_PATH}:${SOURCE_PATH},${RUNTIME_ARCHIVE}:${RUNTIME_ARCHIVE},${TARGET_PATH}:${TARGET_PATH},${DATASET_PATH}:${DATASET_PATH},/raid/scratch:/raid/scratch" bash -c '
    set -euo pipefail
    node_root="${SCRATCH_JOB_ROOT}/node-${SLURM_NODEID}"
    mkdir -p "$node_root/runtime" "$node_root/input" "$node_root/tmp"
    cp -aL "$SOURCE_PATH" "$node_root/source"
    mkdir -p "$node_root/source/modules"
    ln -s .. "$node_root/source/modules/Model-Optimizer"
    tar --extract --file="$RUNTIME_ARCHIVE" --directory="$node_root/runtime"
    old_venv=$(sed -nE "s/^[[:space:]]*export[[:space:]]+VIRTUAL_ENV=(.*)$/\\1/p" "$node_root/runtime/bin/activate" | head -n 1)
    old_venv="${old_venv#\"}"
    old_venv="${old_venv%\"}"
    [[ -n "$old_venv" ]] || { echo "runtime archive has no VIRTUAL_ENV" >&2; exit 2; }
    grep -IlZ "$old_venv" "$node_root/runtime/bin"/* "$node_root/runtime/pyvenv.cfg" 2>/dev/null \
        | xargs -0 -r sed -i "s|$old_venv|$node_root/runtime|g"
    if (( SLURM_NODEID < SERVE_NODES )); then
        cp -aL "$TARGET_PATH" "$node_root/input/target"
    else
        cp -aL "$DATASET_PATH" "$node_root/input/$DATASET_LOCAL_NAME"
        # Fake-base trainers still need the target configuration/tokenizer sidecars.
        mkdir -p "$node_root/input/target"
        for sidecar in config.json generation_config.json tokenizer.json tokenizer_config.json special_tokens_map.json; do
            [[ -e "$TARGET_PATH/$sidecar" ]] && cp -aL "$TARGET_PATH/$sidecar" "$node_root/input/target/"
        done
    fi
'

export PER_DEVICE_TRAIN_BATCH_SIZE GRADIENT_ACCUMULATION_STEPS NUM_ATTENTION_HEADS NUM_KEY_VALUE_HEADS HEAD_DIM INTERMEDIATE_SIZE EAGLE_CAPTURE_IDS SERVE_TP LOSS_OBJECTIVE LOSS_VALUE DFLASH_DPACE_ALPHA METHOD BLOCK_SIZE MAX_STEPS RUN_NAME SAMPLE_SIZE

# shellcheck disable=SC2016,SC2026 # The child shell expands its Slurm-node environment.
srun --nodes=4 --ntasks=4 --ntasks-per-node=1 --no-container-mount-home --container-image="$IMAGE_PATH" --container-mounts="/raid/scratch:/raid/scratch,${OUTPUT_ROOT}:${OUTPUT_ROOT},${CONTROL_ROOT}:/scratchspace" bash -c '
    set -euo pipefail
    node_root="${SCRATCH_JOB_ROOT}/node-${SLURM_NODEID}"
    export MODELOPT_RUNTIME="$node_root/runtime"
    export VIRTUAL_ENV="$MODELOPT_RUNTIME"
    export PATH="$VIRTUAL_ENV/bin:$PATH"
    export PYTHONPATH="$node_root/source${PYTHONPATH:+:$PYTHONPATH}"
    "$MODELOPT_RUNTIME/bin/python" - "$node_root/source" <<'PY'
import sys
from pathlib import Path

import accelerate, datasets, modelopt, wandb

source = Path(sys.argv[1]).resolve()
origin = Path(modelopt.__file__).resolve()
if not origin.is_relative_to(source):
    raise RuntimeError(f"modelopt imported outside staged source: {origin}")
PY
    export HF_MODEL_CKPT="$node_root/input/target"
    export EAGLE_CAPTURE_IDS
    export HF_HOME="${SCRATCH_JOB_ROOT}/node-${SLURM_NODEID}/hf"
    export HF_HUB_CACHE="${HF_HOME}/hub"
    export HF_DATASETS_CACHE="${HF_HOME}/datasets"
    export TRITON_CACHE_DIR="${SCRATCH_JOB_ROOT}/node-${SLURM_NODEID}/triton"
    export XDG_CACHE_HOME="${SCRATCH_JOB_ROOT}/node-${SLURM_NODEID}/xdg"
    export SQLITE_TMPDIR="${SCRATCH_JOB_ROOT}/node-${SLURM_NODEID}/sqlite"
    export WANDB_DIR="${SCRATCH_JOB_ROOT}/node-${SLURM_NODEID}/wandb"
    export WANDB_CACHE_DIR="${SCRATCH_JOB_ROOT}/node-${SLURM_NODEID}/wandb-cache"
    export WANDB_PROJECT="sna-modelopt-specdec"
    export WANDB_RUN_GROUP="$RUN_NAME"
    export SERVE_LOG_DIR="${SCRATCH_JOB_ROOT}/node-${SLURM_NODEID}/logs"
    export TMPDIR="${SCRATCH_JOB_ROOT}/node-${SLURM_NODEID}/tmp"
    mkdir -p "$HF_HUB_CACHE" "$HF_DATASETS_CACHE" "$TRITON_CACHE_DIR" "$XDG_CACHE_HOME" "$SQLITE_TMPDIR" "$WANDB_DIR" "$WANDB_CACHE_DIR" "$SERVE_LOG_DIR" "$TMPDIR"
    export SERVE_NODES=2
    export SERVE_BLOCK_SIZE=32 SERVE_MAX_MODEL_LEN=8192 SERVE_READY_TIMEOUT=1800
    export HS_POOL_SLOTS=32 SERVE_MAX_NUM_SEQS=16
    export EXPORT_PATH="${OUTPUT_ROOT}/exported-checkpoint-${MAX_STEPS}"
    loss_args=(
        "dflash.dflash_loss_objective=${LOSS_OBJECTIVE}"
        "dflash.dflash_loss_decay_factor=${LOSS_VALUE}"
    )
    if [[ "$METHOD" == dflash ]]; then
        loss_args+=("dflash.dflash_dpace_alpha=${DFLASH_DPACE_ALPHA}")
    fi
    cd "$node_root/source"
    exec bash tools/launcher/common/eagle3/train_eagle_streaming.sh \
        --config "modules/Model-Optimizer/modelopt_recipes/general/speculative_decoding/${METHOD}.yaml" \
        model.model_name_or_path="$HF_MODEL_CKPT" \
        model.use_fake_base_for_offline=true \
        data.mode=streaming \
        data.data_path="$node_root/input/$DATASET_LOCAL_NAME" \
        data.sample_size="${SAMPLE_SIZE}" \
        training.output_dir="${OUTPUT_ROOT}" \
        training.max_steps="${MAX_STEPS}" \
        training.save_steps="${MAX_STEPS}" \
        training.save_total_limit=2 \
        training.training_seq_len=4096 \
        training.answer_only_loss=false \
        training.seed=42 \
        training.per_device_train_batch_size="${PER_DEVICE_TRAIN_BATCH_SIZE}" \
        training.gradient_accumulation_steps="${GRADIENT_ACCUMULATION_STEPS}" \
        training.report_to=wandb \
        training.run_name="${RUN_NAME}-s${MAX_STEPS}" \
        dflash.dflash_block_size="${BLOCK_SIZE}" \
        dflash.dflash_num_anchors=512 \
        dflash.dflash_mask_token_id=151669 \
        dflash.dflash_architecture_config.num_attention_heads="${NUM_ATTENTION_HEADS}" \
        dflash.dflash_architecture_config.num_key_value_heads="${NUM_KEY_VALUE_HEADS}" \
        dflash.dflash_architecture_config.head_dim="${HEAD_DIM}" \
        dflash.dflash_architecture_config.intermediate_size="${INTERMEDIATE_SIZE}" \
        "${loss_args[@]}"
'
