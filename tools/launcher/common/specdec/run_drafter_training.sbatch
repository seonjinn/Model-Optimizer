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
    experiment.paths.target_path, experiment.paths.dataset_path, experiment.paths.output_root,
    experiment.method, experiment.block_size, experiment.run_name,
    ",".join(map(str, experiment.topology.capture_ids)), experiment.topology.serve_tp,
    experiment.topology.per_device_train_batch_size, experiment.topology.gradient_accumulation_steps,
    experiment.topology.num_attention_heads, experiment.topology.num_key_value_heads,
    experiment.topology.head_dim, experiment.topology.intermediate_size,
):
    print(value)
PY
)
(( ${#manifest_values[@]} == 18 )) || { echo "manifest entry is unavailable" >&2; exit 2; }
SOURCE_PATH="${manifest_values[0]}"
SOURCE_SHA="${manifest_values[1]}"
IMAGE_PATH="${manifest_values[2]}"
RUNTIME_ARCHIVE="${manifest_values[3]}"
TARGET_PATH="${manifest_values[4]}"
DATASET_PATH="${manifest_values[5]}"
OUTPUT_ROOT="${manifest_values[6]}"
METHOD="${manifest_values[7]}"
BLOCK_SIZE="${manifest_values[8]}"
RUN_NAME="${manifest_values[9]}"
EAGLE_CAPTURE_IDS="[${manifest_values[10]}]"
SERVE_TP="${manifest_values[11]}"
PER_DEVICE_TRAIN_BATCH_SIZE="${manifest_values[12]}"
GRADIENT_ACCUMULATION_STEPS="${manifest_values[13]}"
NUM_ATTENTION_HEADS="${manifest_values[14]}"
NUM_KEY_VALUE_HEADS="${manifest_values[15]}"
HEAD_DIM="${manifest_values[16]}"
INTERMEDIATE_SIZE="${manifest_values[17]}"
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
[[ "$MAX_STEPS" =~ ^[1-9][0-9]*$ ]] || { echo "MAX_STEPS must be positive" >&2; exit 2; }
[[ $((PER_DEVICE_TRAIN_BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS * TRAINER_NODES * GPUS_PER_NODE)) -eq "$GLOBAL_BATCH_SIZE" ]] || { echo "manifest batch arithmetic must equal GBS 512" >&2; exit 2; }

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

export SOURCE_PATH SOURCE_SHA IMAGE_PATH RUNTIME_ARCHIVE TARGET_PATH DATASET_PATH OUTPUT_ROOT SCRATCH_JOB_ROOT
# shellcheck disable=SC2016 # The child shell expands node-local placement variables.
srun --nodes=4 --ntasks=4 --ntasks-per-node=1 --container-image="$IMAGE_PATH" --container-mounts="${SOURCE_PATH}:${SOURCE_PATH},${RUNTIME_ARCHIVE}:${RUNTIME_ARCHIVE},${TARGET_PATH}:${TARGET_PATH},${DATASET_PATH}:${DATASET_PATH},${SCRATCH_JOB_ROOT}:${SCRATCH_JOB_ROOT}" bash -c '
    set -euo pipefail
    node_root="${SCRATCH_JOB_ROOT}/node-${SLURM_NODEID}"
    mkdir -p "$node_root/runtime" "$node_root/input" "$node_root/tmp"
    cp -a "$SOURCE_PATH" "$node_root/source"
    tar --extract --file="$RUNTIME_ARCHIVE" --directory="$node_root/runtime"
    cp -a "$TARGET_PATH" "$node_root/input/target"
    cp -a "$DATASET_PATH" "$node_root/input/dataset"
'

export PER_DEVICE_TRAIN_BATCH_SIZE GRADIENT_ACCUMULATION_STEPS NUM_ATTENTION_HEADS NUM_KEY_VALUE_HEADS HEAD_DIM INTERMEDIATE_SIZE EAGLE_CAPTURE_IDS SERVE_TP LOSS_OBJECTIVE LOSS_VALUE DFLASH_DPACE_ALPHA METHOD BLOCK_SIZE MAX_STEPS RUN_NAME

# shellcheck disable=SC2016 # The child shell expands its Slurm-node environment.
srun --nodes=4 --ntasks=4 --ntasks-per-node=1 --container-image="$IMAGE_PATH" --container-mounts="${SCRATCH_JOB_ROOT}:${SCRATCH_JOB_ROOT}" bash -c '
    set -euo pipefail
    node_root="${SCRATCH_JOB_ROOT}/node-${SLURM_NODEID}"
    export MODELOPT_RUNTIME="$node_root/runtime"
    export HF_MODEL_CKPT="$node_root/input/target"
    export EAGLE_CAPTURE_IDS
    export HF_HOME="${SCRATCH_JOB_ROOT}/node-${SLURM_NODEID}/hf"
    export HF_HUB_CACHE="${HF_HOME}/hub"
    export HF_DATASETS_CACHE="${HF_HOME}/datasets"
    export TRITON_CACHE_DIR="${SCRATCH_JOB_ROOT}/node-${SLURM_NODEID}/triton"
    export XDG_CACHE_HOME="${SCRATCH_JOB_ROOT}/node-${SLURM_NODEID}/xdg"
    export SQLITE_TMPDIR="${SCRATCH_JOB_ROOT}/node-${SLURM_NODEID}/sqlite"
    export WANDB_DIR="${SCRATCH_JOB_ROOT}/node-${SLURM_NODEID}/wandb"
    export TMPDIR="${SCRATCH_JOB_ROOT}/node-${SLURM_NODEID}/tmp"
    mkdir -p "$HF_HUB_CACHE" "$HF_DATASETS_CACHE" "$TRITON_CACHE_DIR" "$XDG_CACHE_HOME" "$SQLITE_TMPDIR" "$WANDB_DIR" "$TMPDIR"
    export SERVE_NODES=2
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
        data.data_path="$node_root/input/dataset" \
        training.output_dir="${OUTPUT_ROOT}" \
        training.max_steps="${MAX_STEPS}" \
        training.global_batch_size=512 \
        training.per_device_train_batch_size="${PER_DEVICE_TRAIN_BATCH_SIZE}" \
        training.gradient_accumulation_steps="${GRADIENT_ACCUMULATION_STEPS}" \
        training.run_name="${RUN_NAME}" \
        dflash.dflash_block_size="${BLOCK_SIZE}" \
        dflash.dflash_num_anchors=512 \
        dflash.dflash_architecture_config.num_attention_heads="${NUM_ATTENTION_HEADS}" \
        dflash.dflash_architecture_config.num_key_value_heads="${NUM_KEY_VALUE_HEADS}" \
        dflash.dflash_architecture_config.head_dim="${HEAD_DIM}" \
        dflash.dflash_architecture_config.intermediate_size="${INTERMEDIATE_SIZE}" \
        "${loss_args[@]}" \
        ${EXTRA_MODELOPT_DOTLIST:-}
'
