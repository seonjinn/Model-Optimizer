#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

#SBATCH --nodes=16
#SBATCH --segment=16
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=96

set -euo pipefail

required=(A_CANARY_MANIFEST A_CANARY_CHECKPOINT A_CANARY_EXPORT A_CANARY_GPU_EVIDENCE
    A_CANARY_EVALUATION_RECEIPT A_CANARY_AUTHORIZATION A_CANARY_EVAL_OUTPUT
    LAUNCHER_ROOT REPO_ROOT TARGET_PATH MODELOPT_RUNTIME)
for name in "${required[@]}"; do
    [[ -n "${!name:-}" ]] || { echo "missing required environment: $name" >&2; exit 2; }
done
[[ "${SLURM_JOB_ID:-}" =~ ^[0-9]+$ && "${SLURM_NNODES:-}" == 16 ]] || exit 2
for output in "$A_CANARY_CHECKPOINT" "$A_CANARY_EXPORT" "$A_CANARY_GPU_EVIDENCE" \
    "$A_CANARY_EVALUATION_RECEIPT" "$A_CANARY_AUTHORIZATION" "$A_CANARY_EVAL_OUTPUT"; do
    [[ ! -e "$output" && ! -L "$output" ]] || {
        echo "A canary output already exists: $output" >&2
        exit 2
    }
done
readonly SUPERVISOR="$LAUNCHER_ROOT/common/eagle3/train_eagle_streaming.sh"
readonly DFLASH_CONFIG="$LAUNCHER_ROOT/modules/Model-Optimizer/modelopt_recipes/general/speculative_decoding/dflash.yaml"
readonly EVALUATOR="$LAUNCHER_ROOT/common/specdec/run_qwen4b_a_canary_eval.sh"

runtime_env="$({ PYTHONPATH="$LAUNCHER_ROOT${PYTHONPATH:+:$PYTHONPATH}" python3 - \
    "$A_CANARY_MANIFEST" "$REPO_ROOT" <<'PY'
import shlex
import sys
from pathlib import Path
from common.specdec.qwen4b_a_canary_manifest import load_a_canary_manifest, validate_bound_artifacts
manifest = load_a_canary_manifest(Path(sys.argv[1]))
validate_bound_artifacts(manifest, repository_root=Path(sys.argv[2]))
values = {
    "A_CANARY_OUTPUT": manifest.builder_output_path,
    "WANDB_PROJECT": manifest.runtime.wandb_project,
    "WANDB_RUN_ID": f"q4a-repair-{manifest.runtime.source_commit[:12]}",
}
for key, value in values.items():
    print(f"{key}={shlex.quote(value)}")
PY
} )"
eval "$runtime_env"
export A_CANARY_OUTPUT WANDB_PROJECT WANDB_RUN_ID
export HF_MODEL_CKPT="$TARGET_PATH" SERVE_NODES=8 SERVE_TP=1 SERVE_REPLICAS_PER_NODE=4
export EAGLE_CAPTURE_IDS='[2,18,33,36]' EXPORT_PATH="$A_CANARY_EXPORT"
export SERVE_READY_TIMEOUT=1800 SERVE_BLOCK_SIZE=32 SERVE_MAX_MODEL_LEN=32768
export SERVE_MAX_NUM_SEQS=16 HS_POOL_SLOTS=32
WANDB_API_KEY="$(python3 - <<'PY'
import netrc
credentials = netrc.netrc('/run/secrets/wandb.netrc').authenticators('api.wandb.ai')
if credentials is None or not credentials[2]:
    raise SystemExit('missing W&B credential')
print(credentials[2])
PY
)"
export WANDB_API_KEY

gpu_log_root="$(dirname "$A_CANARY_GPU_EVIDENCE")/pmon"
mkdir -p "$gpu_log_root" "$A_CANARY_EVAL_OUTPUT"
# shellcheck disable=SC2016
srun --overlap --nodes=16 --ntasks=16 --ntasks-per-node=1 --cpus-per-task=1 \
    bash -c 'exec nvidia-smi pmon -d 5 -s um -o DT >"$1/nvidia-smi-pmon-node-${SLURM_NODEID}-${SLURM_JOB_ID}.log" 2>&1' \
    bash "$gpu_log_root" &
monitor_pid=$!
cleanup_monitor() { kill -TERM "$monitor_pid" 2>/dev/null || true; wait "$monitor_pid" 2>/dev/null || true; }
trap cleanup_monitor EXIT INT TERM

# shellcheck disable=SC2016
srun --overlap --nodes=16 --ntasks=16 --ntasks-per-node=1 --cpus-per-task=1 \
    bash -c 'node_root="${SLURM_TMPDIR}/qwen4b-a-${SLURM_JOB_ID}"; mkdir -p "$node_root/input/corpus"; cp "$1" "$node_root/input/corpus/canary.jsonl"' \
    bash "$A_CANARY_OUTPUT"

# shellcheck disable=SC2016
srun --overlap --nodes=16 --ntasks=16 --ntasks-per-node=1 --gpus-per-task=4 --cpus-per-task=96 \
    --kill-on-bad-exit=1 bash -c '
    node_root="${SLURM_TMPDIR}/qwen4b-a-${SLURM_JOB_ID}"
    source "$1/bin/activate"
    export HF_HOME="$node_root/hf" XDG_CACHE_HOME="$node_root/xdg" TRITON_CACHE_DIR="$node_root/triton"
    export SERVE_LOG_DIR="$node_root/logs"
    mkdir -p "$HF_HOME" "$XDG_CACHE_HOME" "$TRITON_CACHE_DIR" "$SERVE_LOG_DIR"
    cd "$4"
    bash "$2" --config "$3" \
        model.model_name_or_path="$HF_MODEL_CKPT" model.use_fake_base_for_offline=true \
        data.mode=streaming data.data_path="$node_root/input/corpus" data.sample_size=102400 \
        training.output_dir="$A_CANARY_CHECKPOINT" training.max_steps=200 \
        training.save_steps=200 training.save_total_limit=1 \
        training.training_seq_len=4096 training.answer_only_loss=true training.seed=42 \
        training.per_device_train_batch_size=4 training.gradient_accumulation_steps=4 \
        training.report_to=wandb training.run_name="$WANDB_RUN_ID" \
        dflash.dflash_block_size=8 dflash.dflash_num_anchors=512 \
        dflash.dflash_mask_token_id=151669 \
        dflash.dflash_architecture_config.num_attention_heads=32 \
        dflash.dflash_architecture_config.num_key_value_heads=8 \
        dflash.dflash_architecture_config.head_dim=128 \
        dflash.dflash_architecture_config.intermediate_size=9728
    ' bash "$MODELOPT_RUNTIME" "$SUPERVISOR" "$DFLASH_CONFIG" "$LAUNCHER_ROOT"
cleanup_monitor
trap - EXIT INT TERM

PYTHONPATH="$LAUNCHER_ROOT${PYTHONPATH:+:$PYTHONPATH}" python3 - \
    "$gpu_log_root" "$A_CANARY_GPU_EVIDENCE" "$SLURM_JOB_ID" <<'PY'
import sys
from pathlib import Path
from common.specdec.qwen4b_a_canary_manifest import publish_a_gpu_activity
publish_a_gpu_activity(Path(sys.argv[1]), Path(sys.argv[2]), job_id=sys.argv[3])
PY

export A_CANARY_EXPORT A_CANARY_EVALUATION_RECEIPT A_CANARY_EVAL_OUTPUT LAUNCHER_ROOT
srun --overlap --nodes=1 --ntasks=1 --gpus=4 --cpus-per-task=72 \
    --kill-on-bad-exit=1 bash "$EVALUATOR"

PYTHONPATH="$LAUNCHER_ROOT${PYTHONPATH:+:$PYTHONPATH}" python3 - \
    "$A_CANARY_MANIFEST" "$A_CANARY_AUTHORIZATION" "$SLURM_JOB_ID" \
    "$A_CANARY_CHECKPOINT/checkpoint-200" "$A_CANARY_EXPORT" \
    "$A_CANARY_GPU_EVIDENCE" "$A_CANARY_EVALUATION_RECEIPT" <<'PY'
import sys
from pathlib import Path
from common.specdec.qwen4b_a_canary_manifest import load_a_canary_manifest, publish_a_authorization
publish_a_authorization(
    Path(sys.argv[2]), load_a_canary_manifest(Path(sys.argv[1])), job_id=sys.argv[3],
    checkpoint_path=Path(sys.argv[4]), export_path=Path(sys.argv[5]),
    gpu_evidence_path=Path(sys.argv[6]), evaluation_receipt_path=Path(sys.argv[7]),
)
PY
