#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --segment=2
#SBATCH --time=03:55:00

set -euo pipefail

readonly GLOBAL_BATCH_SIZE=128
readonly SERVE_NODES=1
readonly TRAINER_NODES=1
readonly SERVE_REPLICAS_PER_NODE=4
readonly EXPECTED_RNG_STATES=4

for name in MANIFEST_PATH MANIFEST_SHA256 EXPERIMENT_INDEX LAUNCHER_ROOT CLUSTER_PROFILE_PATH CLUSTER_PROFILE_SHA256 READINESS_RECEIPT READINESS_RECEIPT_SHA256 SCRATCH_ROOT MAX_STEPS SAVE_STEPS; do
    [[ -n "${!name:-}" ]] || { echo "missing required environment: $name" >&2; exit 2; }
done
[[ "${SLURM_NNODES:-}" == 2 ]] || { echo "Qwen3-4B study training requires two nodes" >&2; exit 2; }
[[ "$SCRATCH_ROOT" == /* && "$SCRATCH_ROOT" != /home* && "$SCRATCH_ROOT" != /lustre* ]] || {
    echo "scratch root must be node-local" >&2
    exit 2
}
for specification in \
    "$MANIFEST_PATH:$MANIFEST_SHA256:study manifest" \
    "$CLUSTER_PROFILE_PATH:$CLUSTER_PROFILE_SHA256:cluster profile" \
    "$READINESS_RECEIPT:$READINESS_RECEIPT_SHA256:readiness receipt"; do
    IFS=: read -r path expected label <<<"$specification"
    [[ -f "$path" && "$(sha256sum "$path" | cut -d' ' -f1)" == "$expected" ]] || {
        echo "$label identity mismatch" >&2
        exit 2
    }
done

mapfile -t values < <(PYTHONPATH="$LAUNCHER_ROOT${PYTHONPATH:+:$PYTHONPATH}" python3 - "$MANIFEST_PATH" "$EXPERIMENT_INDEX" "$CLUSTER_PROFILE_PATH" "$READINESS_RECEIPT" "$SCRATCH_ROOT" <<'PY'
import hashlib
import json
import os
import sys
from pathlib import Path

from common.specdec.cluster_profile import load_cluster_profile, validate_scratch_root
from common.specdec.build_qwen4b_synthesis_manifest import (
    tokenizer_snapshot_sha256,
    verify_data_manifest,
)
from common.specdec.qwen4b_study_manifest import (
    load_study_manifest,
    readable_study_job_name,
    validate_milestone_receipt,
    validate_readiness_receipt,
    validate_training_target,
)

experiment = load_study_manifest(Path(sys.argv[1]))[int(sys.argv[2])]
profile = load_cluster_profile(Path(sys.argv[3]))
receipt = json.loads(Path(sys.argv[4]).read_text())
scratch_root = Path(sys.argv[5])
validate_scratch_root(profile, scratch_root)
if experiment.inputs.source_sha != profile.modelopt_commit:
    raise SystemExit("source/profile commit mismatch")
if experiment.schedule.account != profile.account or experiment.schedule.partition != profile.partition:
    raise SystemExit("manifest/profile scheduler mismatch")
if profile.training_nodes < 2 or profile.training_nodes % 2 or profile.gpus_per_node != 4:
    raise SystemExit("cluster profile cannot host the pinned two-node topology")
validate_readiness_receipt(
    receipt,
    profile=profile.name,
    account=profile.account,
    partition=profile.partition,
    scratch_root=scratch_root,
)
if experiment.schedule.account == "nemotron_n3_post":
    raise SystemExit("Qwen3-4B study training is forbidden on nemotron_n3_post")
try:
    target_snapshot = json.loads(
        (Path(experiment.inputs.target_path) / "snapshot-manifest.json").read_text()
    )
except (FileNotFoundError, json.JSONDecodeError) as error:
    raise SystemExit("target snapshot manifest is missing or invalid") from error
if target_snapshot.get("source_identity") != experiment.inputs.target_revision:
    raise SystemExit("target snapshot revision mismatch")
if tokenizer_snapshot_sha256(Path(experiment.inputs.target_path)) != experiment.inputs.tokenizer_sha256:
    raise SystemExit("target tokenizer identity mismatch")
corpus_manifest = Path(experiment.inputs.corpus_manifest_path)
corpus_root = Path(experiment.inputs.corpus_path)
if corpus_manifest.parent.resolve() != corpus_root.resolve():
    raise SystemExit("corpus manifest must be rooted in the configured corpus path")
verify_data_manifest(corpus_manifest, experiment.inputs.corpus_manifest_sha256)
if json.loads(corpus_manifest.read_text()).get("tokenizer_sha256") != experiment.inputs.tokenizer_sha256:
    raise SystemExit("corpus tokenizer identity mismatch")
assistant_token_target = os.environ.get("ASSISTANT_TOKEN_TARGET")
assistant_token_target = (
    int(assistant_token_target)
    if assistant_token_target and assistant_token_target != "none"
    else None
)
validate_training_target(experiment, int(os.environ["MAX_STEPS"]), assistant_token_target)
if assistant_token_target is not None:
    milestone_index = experiment.schedule.assistant_token_milestones.index(
        assistant_token_target
    )
    if milestone_index:
        previous_tokens = experiment.schedule.assistant_token_milestones[milestone_index - 1]
        previous_path = Path(os.environ.get("PREVIOUS_MILESTONE_RECEIPT", ""))
        expected_path = (
            Path(experiment.inputs.output_root)
            / "control"
            / f"assistant-token-milestone-{previous_tokens}.json"
        )
        if previous_path != expected_path:
            raise SystemExit("previous assistant-token milestone path mismatch")
        if hashlib.sha256(previous_path.read_bytes()).hexdigest() != os.environ.get(
            "PREVIOUS_MILESTONE_RECEIPT_SHA256"
        ):
            raise SystemExit("previous assistant-token milestone identity mismatch")
        validate_milestone_receipt(previous_path, experiment, previous_tokens)

for value in (
    experiment.experiment_id,
    readable_study_job_name(experiment),
    experiment.arm,
    experiment.thinking_mode,
    experiment.method,
    experiment.block_size,
    experiment.inputs.source_path,
    experiment.inputs.source_sha,
    experiment.inputs.image_path,
    experiment.inputs.image_sha256,
    experiment.inputs.runtime_archive_path,
    experiment.inputs.runtime_archive_sha256,
    experiment.inputs.target_path,
    experiment.inputs.target_revision,
    experiment.inputs.tokenizer_sha256,
    experiment.inputs.corpus_path,
    experiment.inputs.corpus_manifest_path,
    experiment.inputs.corpus_manifest_sha256,
    experiment.inputs.output_root,
    experiment.schedule.assistant_token_budget,
    experiment.schedule.corpus_rows,
    experiment.schedule.max_steps,
    experiment.schedule.canary_steps,
    ",".join(map(str, experiment.schedule.assistant_token_milestones)),
    ",".join(map(str, experiment.schedule.assistant_token_milestone_step_ceilings)),
    ",".join(map(str, experiment.capture_ids)),
    experiment.per_device_train_batch_size,
    experiment.gradient_accumulation_steps,
    experiment.num_attention_heads,
    experiment.num_key_value_heads,
    experiment.head_dim,
    experiment.intermediate_size,
):
    print(value)
PY
)
(( ${#values[@]} == 32 )) || { echo "study manifest entry is unavailable" >&2; exit 2; }
EXPERIMENT_ID="${values[0]}"
RUN_NAME="${values[1]}"
ARM="${values[2]}"
THINKING_MODE="${values[3]}"
METHOD="${values[4]}"
BLOCK_SIZE="${values[5]}"
SOURCE_PATH="${values[6]}"
SOURCE_SHA="${values[7]}"
IMAGE_PATH="${values[8]}"
IMAGE_SHA256="${values[9]}"
RUNTIME_ARCHIVE="${values[10]}"
RUNTIME_ARCHIVE_SHA256="${values[11]}"
TARGET_PATH="${values[12]}"
TARGET_REVISION="${values[13]}"
TOKENIZER_SHA256="${values[14]}"
CORPUS_PATH="${values[15]}"
CORPUS_MANIFEST_PATH="${values[16]}"
CORPUS_MANIFEST_SHA256="${values[17]}"
STUDY_OUTPUT_ROOT="${values[18]}"
ASSISTANT_TOKEN_BUDGET="${values[19]}"
CORPUS_ROWS="${values[20]}"
CANARY_STEPS="${values[22]}"
OUTPUT_ROOT="$STUDY_OUTPUT_ROOT"
if [[ "$MAX_STEPS" == "$CANARY_STEPS" ]]; then
    OUTPUT_ROOT="$STUDY_OUTPUT_ROOT/canary"
fi
ASSISTANT_TOKEN_MILESTONES="${values[23]}"
ASSISTANT_TOKEN_MILESTONE_STEP_CEILINGS="${values[24]}"
EAGLE_CAPTURE_IDS="[${values[25]}]"
PER_DEVICE_TRAIN_BATCH_SIZE="${values[26]}"
GRADIENT_ACCUMULATION_STEPS="${values[27]}"
NUM_ATTENTION_HEADS="${values[28]}"
NUM_KEY_VALUE_HEADS="${values[29]}"
HEAD_DIM="${values[30]}"
INTERMEDIATE_SIZE="${values[31]}"
readonly EXPERIMENT_ID RUN_NAME ARM THINKING_MODE METHOD BLOCK_SIZE SOURCE_PATH SOURCE_SHA
readonly IMAGE_PATH IMAGE_SHA256 RUNTIME_ARCHIVE RUNTIME_ARCHIVE_SHA256 TARGET_PATH TARGET_REVISION TOKENIZER_SHA256
readonly CORPUS_PATH CORPUS_MANIFEST_PATH CORPUS_MANIFEST_SHA256 STUDY_OUTPUT_ROOT OUTPUT_ROOT
readonly ASSISTANT_TOKEN_BUDGET
readonly CORPUS_ROWS CANARY_STEPS ASSISTANT_TOKEN_MILESTONES EAGLE_CAPTURE_IDS
readonly ASSISTANT_TOKEN_MILESTONE_STEP_CEILINGS
readonly PER_DEVICE_TRAIN_BATCH_SIZE GRADIENT_ACCUMULATION_STEPS NUM_ATTENTION_HEADS
readonly NUM_KEY_VALUE_HEADS HEAD_DIM INTERMEDIATE_SIZE

[[ "$MAX_STEPS" =~ ^[1-9][0-9]*$ && "$SAVE_STEPS" =~ ^[1-9][0-9]*$ ]] || {
    echo "MAX_STEPS and SAVE_STEPS must be positive" >&2
    exit 2
}
case ",$ASSISTANT_TOKEN_MILESTONE_STEP_CEILINGS," in
    *",$MAX_STEPS,"*) ;;
    *)
        [[ "$MAX_STEPS" == "$CANARY_STEPS" ]] || {
            echo "MAX_STEPS must select the canary or an immutable milestone ceiling" >&2
            exit 2
        }
        ;;
esac
[[ "$SOURCE_PATH" == /home/* && "$OUTPUT_ROOT" == /lustre/* ]] || { echo "invalid source/output roots" >&2; exit 2; }
[[ "$(git -C "$SOURCE_PATH" rev-parse HEAD)" == "$SOURCE_SHA" && -z "$(git -C "$SOURCE_PATH" status --porcelain)" ]] || {
    echo "source checkout identity mismatch" >&2
    exit 2
}
for specification in \
    "$IMAGE_PATH:$IMAGE_SHA256:image" \
    "$RUNTIME_ARCHIVE:$RUNTIME_ARCHIVE_SHA256:runtime archive" \
    "$CORPUS_MANIFEST_PATH:$CORPUS_MANIFEST_SHA256:corpus manifest"; do
    IFS=: read -r path expected label <<<"$specification"
    [[ -f "$path" && "$(sha256sum "$path" | cut -d' ' -f1)" == "$expected" ]] || {
        echo "$label identity mismatch" >&2
        exit 2
    }
done
[[ -d "$TARGET_PATH" && -d "$CORPUS_PATH" ]] || { echo "target or corpus is missing" >&2; exit 2; }
if [[ "$MAX_STEPS" != "$CANARY_STEPS" ]]; then
    [[ "${ASSISTANT_TOKEN_TARGET:-}" =~ ^[1-9][0-9]*$ ]] || {
        echo "production requires an exact assistant-token target" >&2
        exit 2
    }
    [[ -n "${CANARY_RECEIPT:-}" && -f "$CANARY_RECEIPT" ]] || {
        echo "production requires a canary receipt" >&2
        exit 2
    }
    [[ "$(sha256sum "$CANARY_RECEIPT" | cut -d' ' -f1)" == "${CANARY_RECEIPT_SHA256:-}" ]] || {
        echo "canary receipt identity mismatch" >&2
        exit 2
    }
    PYTHONPATH="$LAUNCHER_ROOT${PYTHONPATH:+:$PYTHONPATH}" python3 - \
        "$MANIFEST_PATH" "$EXPERIMENT_INDEX" "$CANARY_RECEIPT" "$MAX_STEPS" <<'PY'
import sys
from pathlib import Path

from common.specdec.qwen4b_study_manifest import load_study_manifest, validate_canary_receipt

experiment = load_study_manifest(Path(sys.argv[1]))[int(sys.argv[2])]
validate_canary_receipt(Path(sys.argv[3]), experiment, int(sys.argv[4]))
PY
fi

mkdir -p "$OUTPUT_ROOT/control" "$OUTPUT_ROOT/logs"
TARGET_TOKEN_BUDGET="${ASSISTANT_TOKEN_TARGET:-$ASSISTANT_TOKEN_BUDGET}"
[[ "$TARGET_TOKEN_BUDGET" != none ]] || TARGET_TOKEN_BUDGET="$ASSISTANT_TOKEN_BUDGET"
readonly TARGET_TOKEN_BUDGET
python3 - "$CORPUS_MANIFEST_PATH" "$ASSISTANT_TOKEN_BUDGET" "$CORPUS_ROWS" "$OUTPUT_ROOT/control/assistant-token-budget.json" "$EXPERIMENT_ID" "$ASSISTANT_TOKEN_MILESTONES" "$TARGET_TOKEN_BUDGET" "$TOKENIZER_SHA256" <<'PY'
import json
import os
import sys
from pathlib import Path

manifest = json.loads(Path(sys.argv[1]).read_text())
budget = int(sys.argv[2])
rows = int(sys.argv[3])
if int(manifest.get("unique_assistant_tokens", 0)) != budget:
    raise SystemExit("corpus does not exactly match the assistant-token budget")
if int(manifest.get("row_count", 0)) != rows:
    raise SystemExit("corpus row count does not match the study schedule")
record = {
    "experiment_id": sys.argv[5],
    "requested_assistant_tokens": budget,
    "realized_unique_assistant_tokens": int(manifest["unique_assistant_tokens"]),
    "corpus_rows": rows,
    "assistant_token_milestones": [int(value) for value in sys.argv[6].split(",")],
    "current_assistant_token_target": int(sys.argv[7]),
    "tokenizer_sha256": sys.argv[8],
}
path = Path(sys.argv[4])
if path.exists() and json.loads(path.read_text()) != record:
    raise SystemExit("assistant-token budget identity mismatch")
temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
temporary.write_text(json.dumps(record, sort_keys=True) + "\n")
os.replace(temporary, path)
PY

case "${METHOD}:${BLOCK_SIZE}" in
    dflash:8) NUM_SPEC_TOKENS=7; LOSS_OBJECTIVE=dpace; LOSS_VALUE=0; DFLASH_DPACE_ALPHA=0.5 ;;
    dspark:8) NUM_SPEC_TOKENS=8; LOSS_OBJECTIVE=decay; LOSS_VALUE=4; DFLASH_DPACE_ALPHA="" ;;
    *) echo "unsupported study method/block size" >&2; exit 2 ;;
esac
readonly NUM_SPEC_TOKENS LOSS_OBJECTIVE LOSS_VALUE DFLASH_DPACE_ALPHA

SCRATCH_JOB_ROOT="${SCRATCH_ROOT%/}/${USER}/${SLURM_JOB_ID}/qwen4b-study-${EXPERIMENT_ID}"
STAGE_LOG_ROOT="$OUTPUT_ROOT/logs/stage-${SLURM_JOB_ID}"
EXPORT_PATH="$OUTPUT_ROOT/exported-checkpoint-${MAX_STEPS}"
if [[ "$MAX_STEPS" != "$CANARY_STEPS" ]]; then
    EXPORT_PATH="$OUTPUT_ROOT/exported-assistant-tokens-${ASSISTANT_TOKEN_TARGET}"
fi
TRAINING_IDENTITY="$EXPERIMENT_ID"
TRAINING_FINGERPRINT="$MANIFEST_SHA256"
readonly SCRATCH_JOB_ROOT STAGE_LOG_ROOT EXPORT_PATH TRAINING_IDENTITY TRAINING_FINGERPRINT
export SCRATCH_ROOT SCRATCH_JOB_ROOT STAGE_LOG_ROOT SOURCE_PATH SOURCE_SHA RUNTIME_ARCHIVE RUNTIME_ARCHIVE_SHA256
export TARGET_PATH CORPUS_PATH OUTPUT_ROOT EXPORT_PATH EXPECTED_RNG_STATES TRAINING_IDENTITY TRAINING_FINGERPRINT
export MAX_STEPS SAVE_STEPS SELF_REQUEUE="${SELF_REQUEUE:-1}" MAX_REQUEUES="${MAX_REQUEUES:-50}"
export MILESTONE_STEPS="${MILESTONE_STEPS:-$ASSISTANT_TOKEN_MILESTONE_STEP_CEILINGS}"
unset MODELOPT_ASSISTANT_TOKEN_TARGET
if [[ "$MAX_STEPS" != "$CANARY_STEPS" ]]; then
    export MODELOPT_ASSISTANT_TOKEN_TARGET="$ASSISTANT_TOKEN_TARGET"
fi

# shellcheck source=tools/launcher/common/specdec/drafter_requeue_lifecycle.sh
source "$LAUNCHER_ROOT/common/specdec/drafter_requeue_lifecycle.sh"
drafter_requeue_init
drafter_prepare_training_output

mkdir -p "$STAGE_LOG_ROOT"
stage_step=(srun --nodes=2 --ntasks=2 --ntasks-per-node=1 bash -c '
    set -euo pipefail
    node_root="${SCRATCH_JOB_ROOT}/node-${SLURM_NODEID}"
    [[ "$node_root" == "${SCRATCH_ROOT%/}/${USER}/${SLURM_JOB_ID}/"* ]] || exit 2
    rm -rf -- "$node_root"
    mkdir -p "$node_root/runtime" "$node_root/input" "$node_root/tmp"
    cp -rP --no-preserve=all -- "$SOURCE_PATH" "$node_root/source"
    cp -rL --no-preserve=all -- "$TARGET_PATH" "$node_root/input/target"
    if (( SLURM_NODEID == 1 )); then
        cp -rL --no-preserve=all -- "$CORPUS_PATH" "$node_root/input/corpus"
    fi
    tar --extract --file="$RUNTIME_ARCHIVE" --directory="$node_root/runtime"
    old_venv=$(sed -nE "s/^[[:space:]]*export[[:space:]]+VIRTUAL_ENV=(.*)$/\\1/p" "$node_root/runtime/bin/activate" | head -n1)
    old_venv="${old_venv#\"}"; old_venv="${old_venv%\"}"
    [[ -n "$old_venv" ]] || exit 2
    grep -IlZ "$old_venv" "$node_root/runtime/bin"/* "$node_root/runtime/pyvenv.cfg" 2>/dev/null \
        | xargs -0 -r sed -i "s|$old_venv|$node_root/runtime|g"
')
drafter_run_requeueable_step "${stage_step[@]}"

WANDB_NETRC_PATH="${WANDB_NETRC_PATH:-}"
CONTAINER_MOUNTS="${SCRATCH_ROOT}:${SCRATCH_ROOT},${OUTPUT_ROOT}:${OUTPUT_ROOT},${OUTPUT_ROOT}/control:/scratchspace"
if [[ -n "$WANDB_NETRC_PATH" ]]; then
    [[ "$WANDB_NETRC_PATH" =~ ^/home/[A-Za-z0-9._-]+/\.netrc$ && -f "$WANDB_NETRC_PATH" && ! -L "$WANDB_NETRC_PATH" ]] || {
        echo "invalid W&B netrc" >&2; exit 2;
    }
    CONTAINER_MOUNTS+=",${WANDB_NETRC_PATH}:/run/secrets/wandb.netrc:ro"
fi
export IMAGE_PATH CONTAINER_MOUNTS WANDB_NETRC_PATH RUN_NAME ARM THINKING_MODE METHOD BLOCK_SIZE
export NUM_SPEC_TOKENS LOSS_OBJECTIVE LOSS_VALUE DFLASH_DPACE_ALPHA EAGLE_CAPTURE_IDS
export PER_DEVICE_TRAIN_BATCH_SIZE GRADIENT_ACCUMULATION_STEPS NUM_ATTENTION_HEADS NUM_KEY_VALUE_HEADS HEAD_DIM INTERMEDIATE_SIZE
export SERVE_NODES TRAINER_NODES SERVE_REPLICAS_PER_NODE GLOBAL_BATCH_SIZE CORPUS_ROWS TARGET_REVISION
DEFAULT_WANDB_RUN_ID="q4b-${EXPERIMENT_ID}-production"
if [[ "$MAX_STEPS" == "$CANARY_STEPS" ]]; then
    DEFAULT_WANDB_RUN_ID="q4b-${EXPERIMENT_ID}-canary"
fi
export WANDB_RUN_ID="$DEFAULT_WANDB_RUN_ID" WANDB_RESUME=allow

rm -f "$OUTPUT_ROOT/control/.training_done.${SLURM_JOB_ID}" "$OUTPUT_ROOT/control/.trainer_addr.${SLURM_JOB_ID}"
for replica in 0 1 2 3; do rm -f "$OUTPUT_ROOT/control/.serve_addr.${SLURM_JOB_ID}.0.${replica}"; done

# shellcheck disable=SC2016
training_step=(srun --nodes=2 --ntasks=2 --ntasks-per-node=1 --exclusive --kill-on-bad-exit=1 \
    --no-container-mount-home --container-image="$IMAGE_PATH" --container-mounts="$CONTAINER_MOUNTS" bash -c '
    set -euo pipefail
    node_root="${SCRATCH_JOB_ROOT}/node-${SLURM_NODEID}"
    export VIRTUAL_ENV="$node_root/runtime" PATH="$node_root/runtime/bin:$PATH"
    export PYTHONPATH="$node_root/source${PYTHONPATH:+:$PYTHONPATH}"
    export HF_MODEL_CKPT="$node_root/input/target"
    export HF_HOME="$node_root/hf" XDG_CACHE_HOME="$node_root/xdg" TRITON_CACHE_DIR="$node_root/triton" TMPDIR="$node_root/tmp"
    export SERVE_LOG_DIR="$node_root/logs" SERVE_MODEL_NAME="qwen4b-${EXPERIMENT_ID}"
    export SERVE_TP=1 SERVE_BLOCK_SIZE=32 SERVE_MAX_MODEL_LEN=32768 SERVE_READY_TIMEOUT=1800
    export HS_POOL_SLOTS=32 SERVE_MAX_NUM_SEQS=16 EXPORT_PATH EAGLE_CAPTURE_IDS
    export WANDB_PROJECT=sna-modelopt-specdec WANDB_RUN_GROUP="$RUN_NAME" WANDB_DIR="$node_root/wandb"
    if [[ -n "$WANDB_NETRC_PATH" ]]; then
        export WANDB_API_KEY="$(python - <<'PY'
import netrc
credentials = netrc.netrc("/run/secrets/wandb.netrc").authenticators("api.wandb.ai")
if credentials is None or not credentials[2]:
    raise SystemExit("missing W&B credential")
print(credentials[2])
PY
)"
    fi
    mkdir -p "$HF_HOME" "$XDG_CACHE_HOME" "$TRITON_CACHE_DIR" "$TMPDIR" "$SERVE_LOG_DIR" "$WANDB_DIR"
    pmon="$OUTPUT_ROOT/logs/nvidia-smi-pmon-node-${SLURM_NODEID}-${SLURM_JOB_ID}.log"
    nvidia-smi pmon -d 10 -s um -o DT >"$pmon" 2>&1 &
    pmon_pid=$!
    cleanup() { kill -TERM "$pmon_pid" 2>/dev/null || true; }
    trap cleanup EXIT INT TERM
    loss_args=("dflash.dflash_loss_objective=${LOSS_OBJECTIVE}" "dflash.dflash_loss_decay_factor=${LOSS_VALUE}")
    [[ "$METHOD" != dflash ]] || loss_args+=("dflash.dflash_dpace_alpha=${DFLASH_DPACE_ALPHA}")
    cd "$node_root/source/tools/launcher"
    bash common/eagle3/train_eagle_streaming.sh \
        --config "modules/Model-Optimizer/modelopt_recipes/general/speculative_decoding/${METHOD}.yaml" \
        model.model_name_or_path="$HF_MODEL_CKPT" model.use_fake_base_for_offline=true \
        data.mode=streaming data.data_path="$node_root/input/corpus" data.sample_size="$CORPUS_ROWS" \
        training.output_dir="$OUTPUT_ROOT" training.max_steps="$MAX_STEPS" \
        training.save_steps="${SAVE_STEPS}" training.save_total_limit=2 \
        training.training_seq_len=4096 training.answer_only_loss=true training.seed=42 \
        training.per_device_train_batch_size="$PER_DEVICE_TRAIN_BATCH_SIZE" \
        training.gradient_accumulation_steps="$GRADIENT_ACCUMULATION_STEPS" \
        training.report_to=wandb training.run_name="${RUN_NAME}-s${MAX_STEPS}" \
        dflash.dflash_block_size="$BLOCK_SIZE" dflash.dflash_num_anchors=512 \
        dflash.dflash_mask_token_id=151669 \
        dflash.dflash_architecture_config.num_attention_heads="$NUM_ATTENTION_HEADS" \
        dflash.dflash_architecture_config.num_key_value_heads="$NUM_KEY_VALUE_HEADS" \
        dflash.dflash_architecture_config.head_dim="$HEAD_DIM" \
        dflash.dflash_architecture_config.intermediate_size="$INTERMEDIATE_SIZE" \
        "${loss_args[@]}"
')
drafter_run_requeueable_step "${training_step[@]}"

PYTHONPATH="$LAUNCHER_ROOT${PYTHONPATH:+:$PYTHONPATH}" python3 - \
    "$OUTPUT_ROOT/logs" "$OUTPUT_ROOT/control/gpu-ownership.json" "$SLURM_JOB_ID" <<'PY'
import hashlib
import json
import os
import sys
from pathlib import Path

from common.specdec.qwen4b_study_manifest import validate_pmon_logs

observed_gpu_indices = validate_pmon_logs(Path(sys.argv[1]), sys.argv[3])
pmon_sha256 = {
    name: hashlib.sha256((Path(sys.argv[1]) / name).read_bytes()).hexdigest()
    for name in observed_gpu_indices
}
record = {
    "nodes": 2,
    "pmon_logs": sorted(observed_gpu_indices),
    "observed_gpu_indices": observed_gpu_indices,
    "pmon_sha256": pmon_sha256,
    "nonzero_sm_activity": True,
}
path = Path(sys.argv[2])
temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
temporary.write_text(json.dumps(record, sort_keys=True) + "\n")
os.replace(temporary, path)
PY
drafter_mark_training_complete
checkpoint_record="$(drafter_latest_complete_checkpoint "$OUTPUT_ROOT")"
checkpoint_path="${checkpoint_record%%$'\t'*}"
if [[ "$MAX_STEPS" != "$CANARY_STEPS" ]]; then
    PYTHONPATH="$LAUNCHER_ROOT${PYTHONPATH:+:$PYTHONPATH}" python3 - \
        "$OUTPUT_ROOT/control/assistant-token-milestone-${ASSISTANT_TOKEN_TARGET}.json" \
        "$EXPERIMENT_ID" "$ASSISTANT_TOKEN_TARGET" "$checkpoint_path" "$EXPORT_PATH" \
        "$TRAINING_FINGERPRINT" <<'PY'
import json
import os
import sys
from pathlib import Path

from common.specdec.qwen4b_study_manifest import _directory_sha256

path = Path(sys.argv[1])
checkpoint = Path(sys.argv[4])
export_path = Path(sys.argv[5])
fingerprint = sys.argv[6]
token_state = json.loads((checkpoint / "assistant-token-state.json").read_text())
trainer_state = json.loads((checkpoint / "trainer_state.json").read_text())
target = int(sys.argv[3])
if (
    token_state.get("training_fingerprint") != fingerprint
    or token_state.get("committed_assistant_tokens") != target
    or token_state.get("global_step") != int(trainer_state.get("global_step", -1))
):
    raise SystemExit("final assistant-token checkpoint state mismatch")
if not export_path.is_dir() or not any(item.is_file() for item in export_path.iterdir()):
    raise SystemExit("assistant-token milestone export is missing")
record = {
    "schema_version": 1,
    "experiment_id": sys.argv[2],
    "committed_assistant_tokens": target,
    "global_step": token_state["global_step"],
    "checkpoint_path": str(checkpoint),
    "checkpoint_sha256": _directory_sha256(checkpoint),
    "export_path": str(export_path),
    "export_sha256": _directory_sha256(export_path),
    "checkpoint_verified": True,
    "export_verified": True,
    "status": "passed",
}
temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
temporary.write_text(json.dumps(record, sort_keys=True) + "\n")
os.replace(temporary, path)
PY
fi
if [[ "$MAX_STEPS" == "$CANARY_STEPS" ]]; then
    PYTHONPATH="$LAUNCHER_ROOT${PYTHONPATH:+:$PYTHONPATH}" python3 - \
        "$OUTPUT_ROOT/control/canary-${EXPERIMENT_ID}.json" \
        "$EXPERIMENT_ID" "$MAX_STEPS" "$checkpoint_path" "$EXPORT_PATH" \
        "$OUTPUT_ROOT/control/gpu-ownership.json" <<'PY'
import json
import os
import sys
from pathlib import Path

from common.specdec.qwen4b_study_manifest import _directory_sha256, _file_sha256

path = Path(sys.argv[1])
checkpoint = Path(sys.argv[4])
export_path = Path(sys.argv[5])
gpu_ownership = Path(sys.argv[6])
record = {
    "schema_version": 1,
    "experiment_id": sys.argv[2],
    "completed_steps": int(sys.argv[3]),
    "checkpoint_verified": True,
    "export_verified": True,
    "all_gpus_active": True,
    "checkpoint_path": str(checkpoint),
    "checkpoint_sha256": _directory_sha256(checkpoint),
    "export_path": str(export_path),
    "export_sha256": _directory_sha256(export_path),
    "gpu_ownership_path": str(gpu_ownership),
    "gpu_ownership_sha256": _file_sha256(gpu_ownership),
    "status": "passed",
}
temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
temporary.write_text(json.dumps(record, sort_keys=True) + "\n")
os.replace(temporary, path)
PY
fi
