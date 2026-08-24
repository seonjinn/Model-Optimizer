#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

#SBATCH --nodes=16
#SBATCH --segment=16
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=96

set -euo pipefail

required=(
    B_CANARY_MANIFEST B_CANARY_READINESS B_CANARY_BUILD_RECEIPT B_CANARY_OUTPUT
    B_CANARY_EVIDENCE B_CANARY_CHECKPOINT B_CANARY_EXPORT B_CANARY_EVALUATION_RECEIPT
    B_CANARY_GPU_EVIDENCE B_CANARY_EVAL_OUTPUT LAUNCHER_ROOT REPO_ROOT MODELOPT_RUNTIME
    B_CANARY_SUPERVISOR_RECEIPT CONTAINER_IMAGE
    B_A_AUTHORIZATION_RECEIPT B_A_AUTHORIZATION_RECEIPT_SHA256
    B_A_SCHEDULER_OBSERVATION B_A_SCHEDULER_OBSERVATION_SHA256
)
for name in "${required[@]}"; do
    [[ -n "${!name:-}" ]] || { echo "missing required environment: $name" >&2; exit 2; }
done
[[ "${SLURM_JOB_ID:-}" =~ ^[0-9]+$ && "${SLURM_NNODES:-}" == 16 ]] || exit 2

readonly SUPERVISOR="$REPO_ROOT/tools/launcher/common/eagle3/train_eagle_streaming.sh"
readonly DFLASH_CONFIG="$REPO_ROOT/modelopt_recipes/general/speculative_decoding/dflash.yaml"
readonly EVALUATOR="$LAUNCHER_ROOT/common/specdec/run_qwen4b_b_canary_eval.sh"

runtime_env="$({ PYTHONPATH="$LAUNCHER_ROOT${PYTHONPATH:+:$PYTHONPATH}" python3 - \
    "$B_CANARY_MANIFEST" "$B_CANARY_READINESS" "$B_CANARY_BUILD_RECEIPT" \
    "$B_CANARY_OUTPUT" "$SUPERVISOR" "$DFLASH_CONFIG" \
    "$B_A_AUTHORIZATION_RECEIPT" "$B_A_AUTHORIZATION_RECEIPT_SHA256" \
    "$REPO_ROOT" "$B_A_SCHEDULER_OBSERVATION" \
    "$B_A_SCHEDULER_OBSERVATION_SHA256" "$CONTAINER_IMAGE" <<'PY'
import shlex
import sys
from pathlib import Path

from common.specdec.qwen4b_b_canary_manifest import (
    load_b_canary_manifest,
    validate_a_authorization_receipt,
    validate_a_scheduler_observation,
    validate_bound_artifacts,
    validate_runtime_artifacts,
)

manifest = load_b_canary_manifest(Path(sys.argv[1]))
validate_bound_artifacts(
    manifest,
    readiness_path=Path(sys.argv[2]),
    builder_receipt_path=Path(sys.argv[3]),
    builder_output_path=Path(sys.argv[4]),
)
supervisor, config = map(Path, sys.argv[5:7])
validate_runtime_artifacts(
    manifest,
    supervisor_path=supervisor,
    config_path=config,
    repo_root=Path(sys.argv[9]),
    container_path=Path(sys.argv[12]),
    require_clean_checkout=True,
)
if manifest.runtime.corpus_path != str(Path(sys.argv[4])):
    raise SystemExit("runtime corpus identity changed")
validate_a_authorization_receipt(
    Path(sys.argv[7]),
    sys.argv[8],
    source_commit=manifest.source_commit,
    target_revision=manifest.runtime.target_revision,
    tokenizer_sha256=manifest.runtime.tokenizer_sha256,
    chat_template_sha256=manifest.runtime.chat_template_sha256,
    container_sha256=manifest.runtime.container_sha256,
    repo_root=Path(sys.argv[9]),
    b_runtime=manifest.runtime,
)
validate_a_scheduler_observation(
    Path(sys.argv[10]),
    sys.argv[11],
    authorization_sha256=sys.argv[8],
)
values = {
    "HF_MODEL_CKPT": manifest.runtime.target_path,
    "SERVE_NODES": "8",
    "SERVE_TP": "1",
    "SERVE_REPLICAS_PER_NODE": "4",
    "EAGLE_CAPTURE_IDS": "[2,18,33,36]",
    "WANDB_PROJECT": manifest.runtime.wandb_project,
    "WANDB_RUN_ID": manifest.runtime.wandb_run_id,
    "B_OUTPUT_ROOT": manifest.runtime.output_root,
    "B_CANARY_OUTPUT_SHA256": manifest.builder_output_sha256,
}
for key, value in values.items():
    print(f"{key}={shlex.quote(value)}")
PY
} )"
eval "$runtime_env"
export HF_MODEL_CKPT SERVE_NODES SERVE_TP SERVE_REPLICAS_PER_NODE EAGLE_CAPTURE_IDS
export WANDB_PROJECT WANDB_RUN_ID MODELOPT_RUNTIME B_OUTPUT_ROOT B_CANARY_OUTPUT_SHA256
export EXPORT_PATH="$B_CANARY_EXPORT"

for output in "$B_CANARY_CHECKPOINT" "$B_CANARY_EXPORT" \
    "$B_CANARY_EVALUATION_RECEIPT" "$B_CANARY_EVAL_OUTPUT" \
    "$B_CANARY_GPU_EVIDENCE" "$B_CANARY_SUPERVISOR_RECEIPT"; do
    [[ "$output" == "$B_OUTPUT_ROOT"/* ]] || {
        echo "B canary output escapes current-job root: $output" >&2
        exit 2
    }
done
for output in "$B_OUTPUT_ROOT" "$B_CANARY_CHECKPOINT" "$B_CANARY_EXPORT" \
    "$B_CANARY_EVIDENCE" "$B_CANARY_EVALUATION_RECEIPT" "$B_CANARY_EVAL_OUTPUT" \
    "$B_CANARY_GPU_EVIDENCE" "$B_CANARY_SUPERVISOR_RECEIPT"; do
    [[ ! -e "$output" && ! -L "$output" ]] || {
        echo "B canary output already exists: $output" >&2
        exit 2
    }
done
mkdir -m 700 "$B_OUTPUT_ROOT"
mkdir -m 700 "$B_OUTPUT_ROOT/control" "$B_OUTPUT_ROOT/evaluation"
job_owner="$B_OUTPUT_ROOT/control/SLURM_JOB_ID"
( set -o noclobber; printf '%s\n' "$SLURM_JOB_ID" >"$job_owner" )
[[ "$(<"$job_owner")" == "$SLURM_JOB_ID" ]] || exit 2

# Capture utilization inside this job; the receipt below is derived from these logs,
# never from caller-authored booleans. The monitor is explicitly terminated and waited.
gpu_log_root="$B_OUTPUT_ROOT/control/pmon"
mkdir -p "$gpu_log_root"
srun --overlap --nodes=16 --ntasks=16 --ntasks-per-node=1 --cpus-per-task=1 \
    bash -c 'log="$1/nvidia-smi-pmon-node-${SLURM_NODEID}-${SLURM_JOB_ID}.log"; exec nvidia-smi pmon -d 5 -s um -o DT >"$log" 2>&1' \
    bash "$gpu_log_root" &
monitor_pid=$!
cleanup_monitor() {
    kill -TERM "$monitor_pid" 2>/dev/null || true
    wait "$monitor_pid" 2>/dev/null || true
}
trap cleanup_monitor EXIT INT TERM

# Stage the authenticated corpus once per node on job-local storage. Each copy is
# rehashed before the trainer can consume it; no shared-path auto-resume is allowed.
# shellcheck disable=SC2016
srun --overlap --nodes=16 --ntasks=16 --ntasks-per-node=1 --cpus-per-task=1 \
    bash -c '
    node_root="${SLURM_TMPDIR}/qwen4b-b-${SLURM_JOB_ID}"
    [[ ! -e "$node_root" && ! -L "$node_root" ]] || exit 2
    mkdir -m 700 -p "$node_root/input/corpus"
    cp "$1" "$node_root/input/corpus/canary.jsonl"
    [[ "$(sha256sum "$node_root/input/corpus/canary.jsonl" | awk '\''{print $1}'\'')" == "$2" ]]
    ' bash "$B_CANARY_OUTPUT" "$B_CANARY_OUTPUT_SHA256"

# The canonical supervisor owns each server PID, fails on early server death, and its
# EXIT trap sends TERM then wait(2)s every server after the trainer has completed.
supervisor_started_at="$(date -u +%Y-%m-%dT%H:%M:%S.%NZ)"
# shellcheck disable=SC2016
srun --overlap --nodes=16 --ntasks=16 --ntasks-per-node=1 --gpus-per-task=4 --cpus-per-task=96 \
    --kill-on-bad-exit=1 bash -c '
    node_root="${SLURM_TMPDIR}/qwen4b-b-${SLURM_JOB_ID}"
    source "$1/bin/activate"
    export HF_HOME="$node_root/hf" XDG_CACHE_HOME="$node_root/xdg" TRITON_CACHE_DIR="$node_root/triton"
    export SERVE_LOG_DIR="$node_root/logs"
    mkdir -p "$HF_HOME" "$XDG_CACHE_HOME" "$TRITON_CACHE_DIR" "$SERVE_LOG_DIR"
    cd "$LAUNCHER_ROOT"
    bash "$2" --config "$3" \
        model.model_name_or_path="$HF_MODEL_CKPT" model.use_fake_base_for_offline=true \
        data.data_path="$node_root/input/corpus" \
        training.output_dir="$B_CANARY_CHECKPOINT" training.max_steps=200 \
        training.save_strategy=steps training.save_steps=200 training.save_total_limit=1 \
        training.training_seq_len=4096 training.answer_only_loss=true training.seed=42 \
        training.per_device_train_batch_size=4 training.gradient_accumulation_steps=4 \
        training.report_to=wandb training.run_name="$WANDB_RUN_ID" \
        dflash.dflash_block_size=8 dflash.dflash_num_anchors=512 \
        dflash.dflash_mask_token_id=151669 \
        dflash.dflash_architecture_config.num_hidden_layers=5 \
        dflash.dflash_architecture_config.num_attention_heads=32 \
        dflash.dflash_architecture_config.num_key_value_heads=8 \
        dflash.dflash_architecture_config.head_dim=128 \
        dflash.dflash_architecture_config.intermediate_size=9728
    ' bash "$MODELOPT_RUNTIME" "$SUPERVISOR" "$DFLASH_CONFIG"
supervisor_finished_at="$(date -u +%Y-%m-%dT%H:%M:%S.%NZ)"
cleanup_monitor
trap - EXIT INT TERM

PYTHONPATH="$LAUNCHER_ROOT${PYTHONPATH:+:$PYTHONPATH}" python3 - \
    "$gpu_log_root" "$B_CANARY_GPU_EVIDENCE" "$SLURM_JOB_ID" <<'PY'
import sys
from pathlib import Path
from common.specdec.qwen4b_b_canary_manifest import publish_b_gpu_activity
publish_b_gpu_activity(Path(sys.argv[1]), Path(sys.argv[2]), job_id=sys.argv[3])
PY

srun --overlap --nodes=1 --ntasks=1 --gpus=4 --cpus-per-task=72 \
    --kill-on-bad-exit=1 bash "$EVALUATOR"

PYTHONPATH="$LAUNCHER_ROOT${PYTHONPATH:+:$PYTHONPATH}" python3 - \
    "$B_CANARY_MANIFEST" "$B_CANARY_SUPERVISOR_RECEIPT" "$SLURM_JOB_ID" \
    "$B_CANARY_CHECKPOINT/checkpoint-200" "$B_CANARY_EXPORT" \
    "$B_CANARY_EVALUATION_RECEIPT" "$supervisor_started_at" \
    "$supervisor_finished_at" <<'PY'
import sys
from pathlib import Path
from common.specdec.qwen4b_b_canary_manifest import (
    load_b_canary_manifest,
    publish_b_supervisor_completion,
)
publish_b_supervisor_completion(
    Path(sys.argv[2]),
    load_b_canary_manifest(Path(sys.argv[1])),
    job_id=sys.argv[3],
    checkpoint_path=Path(sys.argv[4]),
    export_path=Path(sys.argv[5]),
    evaluation_receipt_path=Path(sys.argv[6]),
    started_at=sys.argv[7],
    finished_at=sys.argv[8],
)
PY

PYTHONPATH="$LAUNCHER_ROOT${PYTHONPATH:+:$PYTHONPATH}" python3 - \
    "$B_CANARY_MANIFEST" "$B_CANARY_EVIDENCE" "$SLURM_JOB_ID" \
    "$B_CANARY_CHECKPOINT/checkpoint-200" "$B_CANARY_EXPORT" "$B_CANARY_EVALUATION_RECEIPT" \
    "$B_CANARY_GPU_EVIDENCE" "$B_CANARY_SUPERVISOR_RECEIPT" <<'PY'
import sys
from pathlib import Path

from common.specdec.qwen4b_b_canary_manifest import load_b_canary_manifest, publish_b_canary_evidence

publish_b_canary_evidence(
    Path(sys.argv[2]),
    load_b_canary_manifest(Path(sys.argv[1])),
    job_id=sys.argv[3],
    checkpoint_path=Path(sys.argv[4]),
    export_path=Path(sys.argv[5]),
    evaluation_receipt_path=Path(sys.argv[6]),
    gpu_evidence_path=Path(sys.argv[7]),
    supervisor_receipt_path=Path(sys.argv[8]),
)
PY
