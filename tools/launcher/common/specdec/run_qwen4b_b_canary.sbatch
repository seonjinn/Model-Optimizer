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
    B_CANARY_GPU_EVIDENCE LAUNCHER_ROOT REPO_ROOT MODELOPT_RUNTIME
    B_A_AUTHORIZATION_RECEIPT B_A_AUTHORIZATION_RECEIPT_SHA256
)
for name in "${required[@]}"; do
    [[ -n "${!name:-}" ]] || { echo "missing required environment: $name" >&2; exit 2; }
done
[[ "${SLURM_JOB_ID:-}" =~ ^[0-9]+$ && "${SLURM_NNODES:-}" == 16 ]] || exit 2

readonly SUPERVISOR="$REPO_ROOT/tools/launcher/common/eagle3/train_eagle_streaming.sh"
readonly DFLASH_CONFIG="$REPO_ROOT/modules/Model-Optimizer/modelopt_recipes/general/speculative_decoding/dflash.yaml"

runtime_env="$({ PYTHONPATH="$LAUNCHER_ROOT${PYTHONPATH:+:$PYTHONPATH}" python3 - \
    "$B_CANARY_MANIFEST" "$B_CANARY_READINESS" "$B_CANARY_BUILD_RECEIPT" \
    "$B_CANARY_OUTPUT" "$SUPERVISOR" "$DFLASH_CONFIG" \
    "$B_A_AUTHORIZATION_RECEIPT" "$B_A_AUTHORIZATION_RECEIPT_SHA256" <<'PY'
import hashlib
import shlex
import sys
from pathlib import Path

from common.specdec.qwen4b_b_canary_manifest import (
    load_b_canary_manifest,
    validate_a_authorization_receipt,
    validate_bound_artifacts,
)

manifest = load_b_canary_manifest(Path(sys.argv[1]))
validate_bound_artifacts(
    manifest,
    readiness_path=Path(sys.argv[2]),
    builder_receipt_path=Path(sys.argv[3]),
    builder_output_path=Path(sys.argv[4]),
)
supervisor, config = map(Path, sys.argv[5:7])
for path, expected in (
    (supervisor, manifest.runtime.supervisor_sha256),
    (config, manifest.runtime.config_sha256),
):
    if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
        raise SystemExit(f"runtime launcher identity changed: {path}")
if manifest.runtime.supervisor_path != str(supervisor) or manifest.runtime.config_path != str(config):
    raise SystemExit("runtime launcher path identity changed")
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
}
for key, value in values.items():
    print(f"{key}={shlex.quote(value)}")
PY
} )"
eval "$runtime_env"
export HF_MODEL_CKPT SERVE_NODES SERVE_TP SERVE_REPLICAS_PER_NODE EAGLE_CAPTURE_IDS
export WANDB_PROJECT WANDB_RUN_ID MODELOPT_RUNTIME B_OUTPUT_ROOT
export EXPORT_PATH="$B_CANARY_EXPORT"

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

# The canonical supervisor owns each server PID, fails on early server death, and its
# EXIT trap sends TERM then wait(2)s every server after the trainer has completed.
srun --overlap --nodes=16 --ntasks=16 --ntasks-per-node=1 --gpus-per-task=4 --cpus-per-task=96 \
    --kill-on-bad-exit=1 bash "$SUPERVISOR" --config "$DFLASH_CONFIG" \
    model.model_name_or_path="$HF_MODEL_CKPT" \
    model.draft_model_type=dflash model.draft_block_size=8 \
    data.dataset="$B_CANARY_OUTPUT" data.max_length=4096 \
    training.output_dir="$B_CANARY_CHECKPOINT" training.max_steps=200 \
    training.per_device_train_batch_size=4 training.gradient_accumulation_steps=4 \
    training.seed=42 training.report_to=wandb training.run_name="$WANDB_RUN_ID"
cleanup_monitor
trap - EXIT INT TERM

PYTHONPATH="$LAUNCHER_ROOT${PYTHONPATH:+:$PYTHONPATH}" python3 - \
    "$gpu_log_root" "$B_CANARY_GPU_EVIDENCE" "$SLURM_JOB_ID" <<'PY'
import sys
from pathlib import Path
from common.specdec.qwen4b_b_canary_manifest import publish_b_gpu_activity
publish_b_gpu_activity(Path(sys.argv[1]), Path(sys.argv[2]), job_id=sys.argv[3])
PY

PYTHONPATH="$LAUNCHER_ROOT${PYTHONPATH:+:$PYTHONPATH}" python3 - \
    "$B_CANARY_MANIFEST" "$B_CANARY_EVALUATION_RECEIPT" "$SLURM_JOB_ID" \
    "$B_CANARY_EXPORT" <<'PY'
import sys
from pathlib import Path
from common.specdec.qwen4b_b_canary_manifest import load_b_canary_manifest, publish_b_export_evaluation
publish_b_export_evaluation(
    Path(sys.argv[2]),
    load_b_canary_manifest(Path(sys.argv[1])),
    job_id=sys.argv[3],
    export_path=Path(sys.argv[4]),
)
PY

PYTHONPATH="$LAUNCHER_ROOT${PYTHONPATH:+:$PYTHONPATH}" python3 - \
    "$B_CANARY_MANIFEST" "$B_CANARY_EVIDENCE" "$SLURM_JOB_ID" \
    "$B_CANARY_CHECKPOINT" "$B_CANARY_EXPORT" "$B_CANARY_EVALUATION_RECEIPT" \
    "$B_CANARY_GPU_EVIDENCE" <<'PY'
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
)
PY
