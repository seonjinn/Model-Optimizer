#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

#SBATCH --nodes=16
#SBATCH --segment=16
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=96

set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1

required=(A_CANARY_MANIFEST A_CANARY_CHECKPOINT A_CANARY_EXPORT A_CANARY_GPU_EVIDENCE
    A_CANARY_EVALUATION_RECEIPT A_CANARY_JOB_COMPLETION A_CANARY_SUBMISSION_RECEIPT
    A_CANARY_EVAL_OUTPUT
    A_CANARY_SUPERVISOR_COMPLETION A_CANARY_OUTPUT_ROOT LAUNCHER_ROOT REPO_ROOT
    MODELOPT_RUNTIME WANDB_NETRC_PATH WANDB_DIR WANDB_CACHE_DIR WANDB_CONFIG_DIR
    WANDB_ARTIFACT_DIR A_CANARY_INTERMEDIATE_EXPORT)
for name in "${required[@]}"; do
    [[ -n "${!name:-}" ]] || { echo "missing required environment: $name" >&2; exit 2; }
done
[[ "${SLURM_JOB_ID:-}" =~ ^[0-9]+$ && "${SLURM_NNODES:-}" == 16 ]] || exit 2
for output in "$A_CANARY_CHECKPOINT" "$A_CANARY_EXPORT" "$A_CANARY_GPU_EVIDENCE" \
    "$A_CANARY_EVALUATION_RECEIPT" "$A_CANARY_JOB_COMPLETION" \
    "$A_CANARY_SUPERVISOR_COMPLETION" "$A_CANARY_EVAL_OUTPUT" \
    "$A_CANARY_INTERMEDIATE_EXPORT"; do
    [[ ! -e "$output" && ! -L "$output" ]] || {
        echo "A canary output already exists: $output" >&2
        exit 2
    }
done
[[ ! -e "$A_CANARY_OUTPUT_ROOT" && ! -L "$A_CANARY_OUTPUT_ROOT" ]] || exit 2
mkdir "$A_CANARY_OUTPUT_ROOT"
for output in "$A_CANARY_CHECKPOINT" "$A_CANARY_EXPORT" "$A_CANARY_GPU_EVIDENCE" \
    "$A_CANARY_EVALUATION_RECEIPT" "$A_CANARY_JOB_COMPLETION" \
    "$A_CANARY_SUPERVISOR_COMPLETION" "$A_CANARY_EVAL_OUTPUT" \
    "$A_CANARY_INTERMEDIATE_EXPORT"; do
    [[ "$output" == "$A_CANARY_OUTPUT_ROOT"/* ]] || exit 2
done
readonly SUPERVISOR="$LAUNCHER_ROOT/common/eagle3/train_eagle_streaming.sh"
readonly DFLASH_CONFIG="$REPO_ROOT/modelopt_recipes/general/speculative_decoding/dflash.yaml"
readonly EVALUATOR="$LAUNCHER_ROOT/common/specdec/run_qwen4b_a_canary_eval.sh"

runtime_env="$({ PYTHONPATH="$LAUNCHER_ROOT${PYTHONPATH:+:$PYTHONPATH}" python3 - \
    "$A_CANARY_MANIFEST" "$REPO_ROOT" <<'PY'
import shlex
import sys
import os
from pathlib import Path
from common.specdec.qwen4b_a_canary_manifest import (
    load_a_canary_manifest,
    validate_bound_artifacts,
    validate_runtime_mounts,
)
manifest = load_a_canary_manifest(Path(sys.argv[1]))
validate_bound_artifacts(manifest, repository_root=Path(sys.argv[2]))
job_id = os.environ["SLURM_JOB_ID"]
wandb_paths = {
    "WANDB_DIR": Path(manifest.runtime.wandb_dir) / "jobs" / job_id / "run",
    "WANDB_CACHE_DIR": Path(manifest.runtime.wandb_cache_dir) / "jobs" / job_id / "cache",
    "WANDB_CONFIG_DIR": Path(manifest.runtime.wandb_config_dir) / "jobs" / job_id / "config",
    "WANDB_ARTIFACT_DIR": Path(manifest.runtime.wandb_artifact_dir) / "jobs" / job_id / "artifacts",
}
for path in wandb_paths.values():
    path.mkdir(parents=True, exist_ok=False)
runtime_environment = dict(os.environ)
runtime_environment.update({name: str(path) for name, path in wandb_paths.items()})
validate_runtime_mounts(
    manifest.runtime, repository_root=Path(sys.argv[2]), environ=runtime_environment
)
values = {
    "A_CANARY_OUTPUT": manifest.builder_output_path,
    "HF_MODEL_CKPT": manifest.runtime.target_path,
    "WANDB_PROJECT": manifest.runtime.wandb_project,
    "WANDB_RUN_ID": f"q4a-repair-{manifest.runtime.source_commit[:12]}-{job_id}",
    **{name: str(path) for name, path in wandb_paths.items()},
}
for key, value in values.items():
    print(f"{key}={shlex.quote(value)}")
PY
} )"
eval "$runtime_env"
export A_CANARY_OUTPUT WANDB_PROJECT WANDB_RUN_ID HF_MODEL_CKPT
export WANDB_DIR WANDB_CACHE_DIR WANDB_CONFIG_DIR WANDB_ARTIFACT_DIR
export SERVE_NODES=8 SERVE_TP=1 SERVE_REPLICAS_PER_NODE=4
export EAGLE_CAPTURE_IDS='[2,18,33,36]' EXPORT_PATH="$A_CANARY_INTERMEDIATE_EXPORT"
export SERVE_READY_TIMEOUT=1800 SERVE_BLOCK_SIZE=32 SERVE_MAX_MODEL_LEN=32768
export SERVE_MAX_NUM_SEQS=16 HS_POOL_SLOTS=32
[[ "$WANDB_NETRC_PATH" == "/run/secrets/wandb.netrc" ]] || exit 2
readonly WANDB_NETRC_PATH="/run/secrets/wandb.netrc"
WANDB_API_KEY="$(python3 - <<'PY'
import netrc
credentials = netrc.netrc('/run/secrets/wandb.netrc').authenticators('api.wandb.ai')
if credentials is None or not credentials[2]:
    raise SystemExit('missing W&B credential')
print(credentials[2])
PY
)"
export WANDB_API_KEY
supervisor_started_at="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
readonly supervisor_started_at

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
    export QWEN4B_A_SEQUENTIAL_SAMPLER=1
    export PYTHONPATH="$4/common/specdec/qwen4b_a_site${PYTHONPATH:+:$PYTHONPATH}"
    export HF_HOME="$node_root/hf" XDG_CACHE_HOME="$node_root/xdg" TRITON_CACHE_DIR="$node_root/triton"
    export SERVE_LOG_DIR="$node_root/logs"
    mkdir -p "$HF_HOME" "$XDG_CACHE_HOME" "$TRITON_CACHE_DIR" "$SERVE_LOG_DIR"
    cd "$4"
    bash "$2" --config "$3" \
        model.model_name_or_path="$HF_MODEL_CKPT" model.use_fake_base_for_offline=true \
        data.data_path="$node_root/input/corpus" \
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

readonly EXACT_EXPORTER="$REPO_ROOT/examples/speculative_decoding/scripts/export_hf_checkpoint.py"
export EXACT_EXPORTER
# shellcheck disable=SC2016
srun --overlap --nodes=1 --ntasks=1 --gpus=4 --cpus-per-task=72 \
    --kill-on-bad-exit=1 bash -c '
    source "$1/bin/activate"
    python "$EXACT_EXPORTER" --model_path "$A_CANARY_CHECKPOINT/checkpoint-200" \
        --export_path "$A_CANARY_EXPORT"
    ' bash "$MODELOPT_RUNTIME"
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
supervisor_finished_at="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
readonly supervisor_finished_at

PYTHONPATH="$LAUNCHER_ROOT${PYTHONPATH:+:$PYTHONPATH}" python3 - \
    "$A_CANARY_MANIFEST" "$A_CANARY_SUPERVISOR_COMPLETION" "$SLURM_JOB_ID" \
    "$A_CANARY_CHECKPOINT/checkpoint-200" "$A_CANARY_EXPORT" \
    "$A_CANARY_EVALUATION_RECEIPT" "$supervisor_started_at" "$supervisor_finished_at" <<'PY'
import sys
from pathlib import Path
from common.specdec.qwen4b_a_canary_manifest import (
    load_a_canary_manifest,
    publish_a_supervisor_completion,
)
publish_a_supervisor_completion(
    Path(sys.argv[2]), load_a_canary_manifest(Path(sys.argv[1])), job_id=sys.argv[3],
    checkpoint_path=Path(sys.argv[4]), export_path=Path(sys.argv[5]),
    evaluation_receipt_path=Path(sys.argv[6]), started_at=sys.argv[7], finished_at=sys.argv[8],
)
PY

PYTHONPATH="$LAUNCHER_ROOT${PYTHONPATH:+:$PYTHONPATH}" python3 - \
    "$A_CANARY_MANIFEST" "$A_CANARY_JOB_COMPLETION" "$SLURM_JOB_ID" \
    "$A_CANARY_CHECKPOINT/checkpoint-200" "$A_CANARY_EXPORT" \
    "$A_CANARY_GPU_EVIDENCE" "$A_CANARY_EVALUATION_RECEIPT" \
    "$A_CANARY_SUPERVISOR_COMPLETION" "$REPO_ROOT" <<'PY'
import sys
from pathlib import Path
from common.specdec.qwen4b_a_canary_manifest import load_a_canary_manifest, publish_a_job_completion
publish_a_job_completion(
    Path(sys.argv[2]), load_a_canary_manifest(Path(sys.argv[1])), job_id=sys.argv[3],
    checkpoint_path=Path(sys.argv[4]), export_path=Path(sys.argv[5]),
    gpu_evidence_path=Path(sys.argv[6]), evaluation_receipt_path=Path(sys.argv[7]),
    supervisor_completion_path=Path(sys.argv[8]), manifest_path=Path(sys.argv[1]),
    repository_root=Path(sys.argv[9]),
    submission_receipt_path=Path(__import__('os').environ["A_CANARY_SUBMISSION_RECEIPT"]),
)
PY
