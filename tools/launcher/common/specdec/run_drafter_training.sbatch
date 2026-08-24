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

#SBATCH --ntasks-per-node=1
#SBATCH --time=03:55:00

# Execute one pinned streaming-drafter wave. The submitter supplies all inputs.
set -euo pipefail

readonly GLOBAL_BATCH_SIZE=512

for name in MANIFEST_PATH EXPERIMENT_INDEX MAX_STEPS LAUNCHER_ROOT CLUSTER_NAME CLUSTER_PROFILE_PATH CLUSTER_PROFILE_SHA256 SCRATCH_ROOT; do
    [[ -n "${!name:-}" ]] || { echo "missing required environment: $name" >&2; exit 2; }
done
readonly LAUNCHER_ROOT
[[ "$CLUSTER_PROFILE_PATH" == /home/* && -f "$CLUSTER_PROFILE_PATH" ]] || { echo "cluster profile is missing" >&2; exit 2; }
[[ "$(sha256sum "$CLUSTER_PROFILE_PATH" | cut -d' ' -f1)" == "$CLUSTER_PROFILE_SHA256" ]] || { echo "cluster profile SHA-256 mismatch" >&2; exit 2; }
if [[ -n "${READINESS_RECEIPT:-}" ]]; then
    [[ "$READINESS_RECEIPT" == /lustre/* && -f "$READINESS_RECEIPT" && -n "${READINESS_RECEIPT_SHA256:-}" ]] || { echo "readiness receipt is missing" >&2; exit 2; }
    [[ "$(sha256sum "$READINESS_RECEIPT" | cut -d' ' -f1)" == "$READINESS_RECEIPT_SHA256" ]] || { echo "readiness receipt SHA-256 mismatch" >&2; exit 2; }
elif [[ "$CLUSTER_NAME" != "oci-hsg" ]]; then
    echo "non-OCI training requires a readiness receipt" >&2
    exit 2
fi
[[ "$SCRATCH_ROOT" == /* && "$SCRATCH_ROOT" != /lustre* && "$SCRATCH_ROOT" != /home* ]] || { echo "scratch root must be node-local" >&2; exit 2; }
readonly SCRATCH_ROOT
readonly SCRATCH_JOB_ROOT="${SCRATCH_ROOT%/}/${SLURM_JOB_ID}"
WANDB_NETRC_PATH="${WANDB_NETRC_PATH:-}"
WANDB_NETRC_CONTAINER_PATH=""
if [[ -n "$WANDB_NETRC_PATH" ]]; then
    [[ "$WANDB_NETRC_PATH" == /home/*/.netrc ]] || { echo "WANDB netrc must be under /home" >&2; exit 2; }
    [[ "$WANDB_NETRC_PATH" =~ ^/home/[A-Za-z0-9._-]+/\.netrc$ ]] || { echo "WANDB netrc path is unsafe" >&2; exit 2; }
    [[ ! -L "$WANDB_NETRC_PATH" && -f "$WANDB_NETRC_PATH" ]] || { echo "WANDB netrc must be a regular non-symlink file" >&2; exit 2; }
    [[ "$(stat -c %u "$WANDB_NETRC_PATH")" == "$(id -u)" ]] || { echo "WANDB netrc must be owned by the submitting user" >&2; exit 2; }
    case "$(stat -c %a "$WANDB_NETRC_PATH")" in
        400|600) ;;
        *) echo "WANDB netrc permissions must be 0400 or 0600" >&2; exit 2 ;;
    esac
    python3 - "$WANDB_NETRC_PATH" <<'PY'
import netrc
import sys

credentials = netrc.netrc(sys.argv[1]).authenticators("api.wandb.ai")
if credentials is None or not credentials[2]:
    raise SystemExit("WANDB netrc has no api.wandb.ai credential")
PY
    WANDB_NETRC_CONTAINER_PATH=/run/secrets/wandb.netrc
fi
readonly WANDB_NETRC_PATH WANDB_NETRC_CONTAINER_PATH

mapfile -t manifest_values < <(PYTHONPATH="$LAUNCHER_ROOT${PYTHONPATH:+:$PYTHONPATH}" python3 - "$MANIFEST_PATH" "$EXPERIMENT_INDEX" "$CLUSTER_PROFILE_PATH" "${READINESS_RECEIPT:-}" "$CLUSTER_NAME" "$SCRATCH_ROOT" <<'PY'
import json
import sys
from pathlib import Path

manifest = Path(sys.argv[1])
from common.specdec.cluster_profile import load_cluster_profile, validate_scratch_root
from common.specdec.drafter_job_manifest import (
    legacy_training_fingerprint,
    load_manifest,
    topology_v2_training_fingerprint,
)
from common.specdec.dflash2_runtime_contract import validate_artifact_receipt

experiment = load_manifest(manifest)[int(sys.argv[2])]
profile = load_cluster_profile(Path(sys.argv[3]))
receipt_path = Path(sys.argv[4]) if sys.argv[4] else None
cluster_name = sys.argv[5]
scratch_root = Path(sys.argv[6])
if profile.name != cluster_name or experiment.paths.source_sha != profile.modelopt_commit:
    raise SystemExit("cluster profile identity mismatch")
if (
    experiment.slurm.nodes > profile.training_nodes
    or profile.training_nodes % experiment.slurm.nodes
    or experiment.slurm.gpus_per_node != profile.gpus_per_node
    or experiment.slurm.segment > profile.training_segment
):
    raise SystemExit("manifest allocation is incompatible with the cluster profile")
validate_scratch_root(profile, scratch_root)
if receipt_path is not None:
    receipt = json.loads(receipt_path.read_text())
    if receipt.get("profile") != profile.name or Path(receipt["scratch_root"]) != scratch_root:
        raise SystemExit("readiness receipt identity mismatch")
if experiment.method == "dflash2":
    validate_artifact_receipt(
        Path(experiment.paths.target_receipt_path),
        expected_receipt_sha256=experiment.paths.target_receipt_sha256,
        artifact_path=Path(experiment.paths.target_path),
        expected_artifact_sha256=experiment.paths.target_sha256,
        kind="target",
    )
    validate_artifact_receipt(
        Path(experiment.paths.dataset_receipt_path),
        expected_receipt_sha256=experiment.paths.dataset_receipt_sha256,
        artifact_path=Path(experiment.paths.dataset_path),
        expected_artifact_sha256=experiment.paths.dataset_sha256,
        kind="dataset",
    )

uses_topology_v2 = (
    experiment.topology.target_kind == "qwen3-30b-a3b" or experiment.slurm.nodes != 4
    or experiment.method == "dflash2"
)
if uses_topology_v2:
    fingerprint_schema = "topology-v2"
    fingerprint = topology_v2_training_fingerprint(
        experiment,
        cluster_name,
    )
else:
    fingerprint_schema = "legacy-v1"
    fingerprint = legacy_training_fingerprint(experiment)
for value in (
    experiment.paths.source_path, experiment.paths.source_sha, experiment.paths.image_path, experiment.paths.runtime_archive_path,
    experiment.paths.runtime_archive_sha256,
    experiment.paths.image_sha256 or "",
    experiment.paths.target_path, experiment.paths.dataset_path, experiment.paths.output_root,
    experiment.paths.target_sha256 or "-", experiment.paths.target_receipt_path or "-",
    experiment.paths.target_receipt_sha256 or "-", experiment.paths.dataset_sha256 or "-",
    experiment.paths.dataset_receipt_path or "-", experiment.paths.dataset_receipt_sha256 or "-",
    experiment.method, experiment.block_size, experiment.run_name, experiment.sample_size,
    ",".join(map(str, experiment.topology.capture_ids)), experiment.topology.serve_tp,
    experiment.topology.per_device_train_batch_size, experiment.topology.gradient_accumulation_steps,
    experiment.topology.num_attention_heads, experiment.topology.num_key_value_heads,
    experiment.topology.head_dim, experiment.topology.intermediate_size,
    experiment.topology.target_kind,
    experiment.slurm.nodes, experiment.slurm.segment, experiment.slurm.gpus_per_node,
    experiment.experiment_id,
    fingerprint,
    fingerprint_schema,
    experiment.dflash2.vllm_expected_commit if experiment.dflash2 is not None else "-",
    experiment.dflash2.vllm_required_ancestor if experiment.dflash2 is not None else "-",
    experiment.dflash2.vllm_receipt_relative_path if experiment.dflash2 is not None else "-",
    experiment.dflash2.vllm_receipt_sha256 if experiment.dflash2 is not None else "-",
    experiment.dflash2.warmstart_policy if experiment.dflash2 is not None else "-",
    experiment.dflash2.kernel_projection_init if experiment.dflash2 is not None else "-",
    experiment.dflash2.conv_kernel_size if experiment.dflash2 is not None else "-",
    experiment.dflash2.conv_group_size if experiment.dflash2 is not None else "-",
    experiment.dflash2.selector_rank if experiment.dflash2 is not None else "-",
    experiment.dflash2.selector_top_k if experiment.dflash2 is not None else "-",
    experiment.dflash2.max_speculative_tokens if experiment.dflash2 is not None else "-",
):
    print(value)
PY
)
(( ${#manifest_values[@]} == 45 )) || { echo "manifest entry is unavailable" >&2; exit 2; }
SOURCE_PATH="${manifest_values[0]}"
SOURCE_SHA="${manifest_values[1]}"
IMAGE_PATH="${manifest_values[2]}"
RUNTIME_ARCHIVE="${manifest_values[3]}"
RUNTIME_ARCHIVE_SHA256="${manifest_values[4]}"
IMAGE_SHA256="${manifest_values[5]}"
TARGET_PATH="${manifest_values[6]}"
DATASET_PATH="${manifest_values[7]}"
DATASET_LOCAL_NAME="$(basename "$DATASET_PATH")"
OUTPUT_ROOT="${manifest_values[8]}"
TARGET_SHA256="${manifest_values[9]}"
DATASET_SHA256="${manifest_values[12]}"
METHOD="${manifest_values[15]}"
STAGE_TARGET_IDENTITY="$TARGET_PATH"
STAGE_DATASET_IDENTITY="$DATASET_PATH"
if [[ "$METHOD" == dflash2 ]]; then
    STAGE_TARGET_IDENTITY="$TARGET_SHA256"
    STAGE_DATASET_IDENTITY="$DATASET_SHA256"
fi
BLOCK_SIZE="${manifest_values[16]}"
RUN_NAME="${manifest_values[17]}"
SAMPLE_SIZE="${manifest_values[18]}"
EAGLE_CAPTURE_IDS="[${manifest_values[19]}]"
SERVE_TP="${manifest_values[20]}"
PER_DEVICE_TRAIN_BATCH_SIZE="${manifest_values[21]}"
GRADIENT_ACCUMULATION_STEPS="${manifest_values[22]}"
NUM_ATTENTION_HEADS="${manifest_values[23]}"
NUM_KEY_VALUE_HEADS="${manifest_values[24]}"
HEAD_DIM="${manifest_values[25]}"
INTERMEDIATE_SIZE="${manifest_values[26]}"
TARGET_KIND="${manifest_values[27]}"
ALLOCATED_NODES="${manifest_values[28]}"
ALLOCATION_SEGMENT="${manifest_values[29]}"
GPUS_PER_NODE="${manifest_values[30]}"
TRAINING_IDENTITY="${manifest_values[31]}"
TRAINING_FINGERPRINT="${manifest_values[32]}"
TRAINING_FINGERPRINT_SCHEMA="${manifest_values[33]}"
DFLASH2_VLLM_EXPECTED_COMMIT="${manifest_values[34]}"
DFLASH2_VLLM_REQUIRED_ANCESTOR="${manifest_values[35]}"
DFLASH2_VLLM_RECEIPT_RELATIVE_PATH="${manifest_values[36]}"
DFLASH2_VLLM_RECEIPT_SHA256="${manifest_values[37]}"
DFLASH2_WARMSTART_POLICY="${manifest_values[38]}"
DFLASH2_KERNEL_PROJECTION_INIT="${manifest_values[39]}"
DFLASH2_CONV_KERNEL_SIZE="${manifest_values[40]}"
DFLASH2_CONV_GROUP_SIZE="${manifest_values[41]}"
DFLASH2_SELECTOR_RANK="${manifest_values[42]}"
DFLASH2_SELECTOR_TOP_K="${manifest_values[43]}"
DFLASH2_MAX_SPECULATIVE_TOKENS="${manifest_values[44]}"
SERVE_NODES=$((ALLOCATED_NODES / 2))
TRAINER_NODES=$((ALLOCATED_NODES - SERVE_NODES))
EXPECTED_RNG_STATES=$((TRAINER_NODES * GPUS_PER_NODE))
readonly ALLOCATED_NODES ALLOCATION_SEGMENT GPUS_PER_NODE SERVE_NODES TRAINER_NODES EXPECTED_RNG_STATES TRAINING_FINGERPRINT_SCHEMA
[[ "${SLURM_NNODES:-}" == "$ALLOCATED_NODES" ]] || { echo "Slurm allocation does not match manifest nodes" >&2; exit 2; }
echo "training allocation: nodes=${ALLOCATED_NODES} segment=${ALLOCATION_SEGMENT} serve=${SERVE_NODES} trainer=${TRAINER_NODES}"
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
if [[ -n "$IMAGE_SHA256" ]]; then
    [[ "$IMAGE_SHA256" =~ ^[0-9a-f]{64}$ ]] || { echo "IMAGE_SHA256 must be exact" >&2; exit 2; }
    [[ "$(sha256sum "$IMAGE_PATH" | cut -d' ' -f1)" == "$IMAGE_SHA256" ]] || { echo "image SHA-256 mismatch" >&2; exit 2; }
fi
[[ "$MAX_STEPS" =~ ^[1-9][0-9]*$ ]] || { echo "MAX_STEPS must be positive" >&2; exit 2; }
SAVE_STEPS="${SAVE_STEPS:-$MAX_STEPS}"
[[ "$SAVE_STEPS" =~ ^[1-9][0-9]*$ ]] || { echo "SAVE_STEPS must be positive" >&2; exit 2; }
[[ "${SELF_REQUEUE:-0}" =~ ^[01]$ ]] || { echo "SELF_REQUEUE must be 0 or 1" >&2; exit 2; }
if [[ "${SELF_REQUEUE:-0}" == 1 ]]; then
    [[ "${MAX_REQUEUES:-}" =~ ^[1-9][0-9]*$ ]] || { echo "MAX_REQUEUES must be positive" >&2; exit 2; }
    [[ -n "${WANDB_RUN_ID:-}" ]] || { echo "WANDB_RUN_ID is required for self-requeue" >&2; exit 2; }
fi
[[ "$SERVE_TP" =~ ^[1-9][0-9]*$ ]] || { echo "SERVE_TP must be positive" >&2; exit 2; }
(( GPUS_PER_NODE % SERVE_TP == 0 )) || {
    echo "SERVE_TP must divide all ${GPUS_PER_NODE} GPUs on each serve node" >&2; exit 2;
}
SERVE_REPLICAS_PER_NODE=$((GPUS_PER_NODE / SERVE_TP))
(( SERVE_REPLICAS_PER_NODE * SERVE_TP == GPUS_PER_NODE )) || {
    echo "serve replicas must occupy every allocated GPU" >&2; exit 2;
}
[[ $((PER_DEVICE_TRAIN_BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS * TRAINER_NODES * GPUS_PER_NODE)) -eq "$GLOBAL_BATCH_SIZE" ]] || { echo "manifest batch arithmetic must equal GBS 512" >&2; exit 2; }
CONTROL_ROOT="${OUTPUT_ROOT}/control"
STAGE_LOG_ROOT="${OUTPUT_ROOT}/logs/stage-${SLURM_JOB_ID}"
EXPORT_PATH="${OUTPUT_ROOT}/exported-checkpoint-${MAX_STEPS}"
mkdir -p "$OUTPUT_ROOT" "$CONTROL_ROOT"
mkdir -p "$STAGE_LOG_ROOT"

case "${METHOD}:${BLOCK_SIZE}" in
    dflash:8|dflash:16)
        NUM_SPEC_TOKENS=$((BLOCK_SIZE - 1))
        LOSS_OBJECTIVE=dpace
        LOSS_VALUE=0
        DFLASH_DPACE_ALPHA=0.5
        ;;
    dflash2:8)
        NUM_SPEC_TOKENS=7
        LOSS_OBJECTIVE=decay
        LOSS_VALUE=4
        DFLASH_DPACE_ALPHA=""
        [[ "$DFLASH2_WARMSTART_POLICY" == exact-dflash2-only ]] || { echo "unsafe DFlash2 warm-start policy" >&2; exit 2; }
        [[ "$DFLASH2_KERNEL_PROJECTION_INIT" == zero ]] || { echo "unsafe DFlash2 kernel initialization" >&2; exit 2; }
        [[ "$DFLASH2_MAX_SPECULATIVE_TOKENS" == 7 ]] || { echo "DFlash2 K>7 is not validated" >&2; exit 2; }
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
export SERVE_REPLICAS_PER_NODE

export SOURCE_PATH SOURCE_SHA IMAGE_PATH RUNTIME_ARCHIVE RUNTIME_ARCHIVE_SHA256 TARGET_PATH TARGET_SHA256 STAGE_TARGET_IDENTITY DATASET_PATH DATASET_SHA256 STAGE_DATASET_IDENTITY DATASET_LOCAL_NAME OUTPUT_ROOT EXPORT_PATH SCRATCH_JOB_ROOT STAGE_LOG_ROOT EXPECTED_RNG_STATES TRAINING_IDENTITY TRAINING_FINGERPRINT METHOD
if [[ "${SELF_REQUEUE:-0}" == 1 ]]; then
    # shellcheck source=tools/launcher/common/specdec/drafter_requeue_lifecycle.sh
    source "$LAUNCHER_ROOT/common/specdec/drafter_requeue_lifecycle.sh"
    drafter_requeue_init
    drafter_prepare_training_output
    python3 - "$OUTPUT_ROOT/control/training-identity.json" "$TARGET_KIND" "$TRAINING_FINGERPRINT_SCHEMA" <<'PY'
import json
import os
import sys
from pathlib import Path

path = Path(sys.argv[1])
target_kind = sys.argv[2]
schema = sys.argv[3]
if target_kind != "qwen3-30b-a3b" and schema == "legacy-v1":
    raise SystemExit(0)
if schema != "topology-v2":
    raise SystemExit("scaled training requires topology-v2 fingerprinting")
identity = json.loads(path.read_text())
if identity.get("fingerprint_schema") not in (None, schema):
    raise SystemExit("training fingerprint schema mismatch")
if "fingerprint_schema" not in identity:
    identity["fingerprint_schema"] = schema
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(json.dumps(identity, sort_keys=True) + "\n")
    os.replace(temporary, path)
PY
fi
# shellcheck disable=SC2016 # The child shell expands node-local placement variables.
staging_step=(srun --nodes="$ALLOCATED_NODES" --ntasks="$ALLOCATED_NODES" --ntasks-per-node=1 bash -c '
    set -euo pipefail
    exec >"${STAGE_LOG_ROOT}/node-${SLURM_NODEID}.log" 2>&1
    set -x
    node_root="${SCRATCH_JOB_ROOT}/node-${SLURM_NODEID}"
    stage_marker="$node_root/staging.complete"
    if (( SLURM_NODEID >= SERVE_NODES )); then
        stage_role=trainer
    else
        stage_role=serve
    fi
    stage_fingerprint="${SOURCE_SHA}|${RUNTIME_ARCHIVE_SHA256}|${STAGE_TARGET_IDENTITY}|${STAGE_DATASET_IDENTITY}|${stage_role}"
    if [[ -f "$stage_marker" && "$(<"$stage_marker")" == "$stage_fingerprint" ]]; then
        echo "reusing completed host staging: node=${SLURM_NODEID}"
        exit 0
    fi
    [[ "$node_root" == "${SCRATCH_ROOT%/}/${SLURM_JOB_ID}/node-${SLURM_NODEID}" ]] || {
        echo "refusing to clean unexpected scratch path: $node_root" >&2
        exit 2
    }
    if [[ -e "$node_root" ]]; then
        rm -rf -- "$node_root"
    fi
    mkdir -p "$node_root/runtime" "$node_root/input" "$node_root/tmp"
    cp -rP --no-preserve=all -- "$SOURCE_PATH" "$node_root/source"
    tar --extract --file="$RUNTIME_ARCHIVE" --directory="$node_root/runtime"
    old_venv=$(sed -nE "s/^[[:space:]]*export[[:space:]]+VIRTUAL_ENV=(.*)$/\\1/p" "$node_root/runtime/bin/activate" | head -n 1)
    old_venv="${old_venv#\"}"
    old_venv="${old_venv%\"}"
    [[ -n "$old_venv" ]] || { echo "runtime archive has no VIRTUAL_ENV" >&2; exit 2; }
    grep -IlZ "$old_venv" "$node_root/runtime/bin"/* "$node_root/runtime/pyvenv.cfg" 2>/dev/null \
        | xargs -0 -r sed -i "s|$old_venv|$node_root/runtime|g"
    cp -rL --no-preserve=all -- "$TARGET_PATH" "$node_root/input/target"
    if (( SLURM_NODEID >= SERVE_NODES )); then
        cp -rL --no-preserve=all -- "$DATASET_PATH" "$node_root/input/$DATASET_LOCAL_NAME"
    fi
    if [[ "$METHOD" == dflash2 ]]; then
        PYTHONPATH="$node_root/source/tools/launcher" "$node_root/runtime/bin/python" - \
            "$node_root/input/target" "$TARGET_SHA256" \
            "$node_root/input/$DATASET_LOCAL_NAME" "$DATASET_SHA256" "$stage_role" <<'PY'
import sys
from pathlib import Path
from common.specdec.dflash2_runtime_contract import artifact_tree_sha256

if artifact_tree_sha256(Path(sys.argv[1])) != sys.argv[2]:
    raise SystemExit("staged target bytes mismatch")
if sys.argv[5] == "trainer" and artifact_tree_sha256(Path(sys.argv[3])) != sys.argv[4]:
    raise SystemExit("staged Nemotron dataset bytes mismatch")
PY
    fi
    stage_marker_tmp="${stage_marker}.tmp.$$"
    printf "%s\n" "$stage_fingerprint" >"$stage_marker_tmp"
    mv "$stage_marker_tmp" "$stage_marker"
    echo "host staging complete: node=${SLURM_NODEID}"
')
if [[ "${SELF_REQUEUE:-0}" == 1 ]]; then
    drafter_run_requeueable_step "${staging_step[@]}"
else
    "${staging_step[@]}"
fi

export PER_DEVICE_TRAIN_BATCH_SIZE GRADIENT_ACCUMULATION_STEPS NUM_ATTENTION_HEADS NUM_KEY_VALUE_HEADS HEAD_DIM INTERMEDIATE_SIZE EAGLE_CAPTURE_IDS SERVE_TP LOSS_OBJECTIVE LOSS_VALUE DFLASH_DPACE_ALPHA METHOD BLOCK_SIZE MAX_STEPS SAVE_STEPS RUN_NAME SAMPLE_SIZE
export DFLASH2_VLLM_EXPECTED_COMMIT DFLASH2_VLLM_REQUIRED_ANCESTOR DFLASH2_VLLM_RECEIPT_RELATIVE_PATH DFLASH2_VLLM_RECEIPT_SHA256 DFLASH2_CONV_KERNEL_SIZE DFLASH2_CONV_GROUP_SIZE DFLASH2_SELECTOR_RANK DFLASH2_SELECTOR_TOP_K
export SERVE_NODES CLUSTER_NAME WANDB_NETRC_CONTAINER_PATH
if [[ "${SELF_REQUEUE:-0}" == 1 ]]; then
    export WANDB_RUN_ID="${WANDB_RUN_ID}"
    export WANDB_RESUME=allow
fi

rm -f "$CONTROL_ROOT/.training_done.${SLURM_JOB_ID}"
rm -f "$CONTROL_ROOT/.trainer_addr.${SLURM_JOB_ID}"
for ((serve_node = 0; serve_node < SERVE_NODES; serve_node++)); do
    for ((replica = 0; replica < SERVE_REPLICAS_PER_NODE; replica++)); do
        rm -f "$CONTROL_ROOT/.serve_addr.${SLURM_JOB_ID}.${serve_node}.${replica}"
    done
done

CONTAINER_MOUNTS="${SCRATCH_ROOT}:${SCRATCH_ROOT},${OUTPUT_ROOT}:${OUTPUT_ROOT},${CONTROL_ROOT}:/scratchspace"
if [[ -n "$WANDB_NETRC_PATH" ]]; then
    CONTAINER_MOUNTS+=",${WANDB_NETRC_PATH}:/run/secrets/wandb.netrc:ro"
fi

# shellcheck disable=SC2016,SC2026 # The child shell expands its Slurm-node environment.
training_step=(srun --nodes="$ALLOCATED_NODES" --ntasks="$ALLOCATED_NODES" --ntasks-per-node=1 --kill-on-bad-exit=1 --no-container-mount-home --container-image="$IMAGE_PATH" --container-mounts="$CONTAINER_MOUNTS" bash -c '
    set -euo pipefail
    node_root="${SCRATCH_JOB_ROOT}/node-${SLURM_NODEID}"
    export MODELOPT_RUNTIME="$node_root/runtime"
    export VIRTUAL_ENV="$MODELOPT_RUNTIME"
    export PATH="$VIRTUAL_ENV/bin:$PATH"
    export PYTHONPATH="$node_root/source${PYTHONPATH:+:$PYTHONPATH}"
    if [[ -n "$WANDB_NETRC_CONTAINER_PATH" ]]; then
        export WANDB_API_KEY="$("$MODELOPT_RUNTIME/bin/python" - "$WANDB_NETRC_CONTAINER_PATH" <<'PY'
import netrc
import sys

credentials = netrc.netrc(sys.argv[1]).authenticators("api.wandb.ai")
if credentials is None or not credentials[2]:
    raise SystemExit("mounted WANDB netrc has no api.wandb.ai credential")
print(credentials[2])
PY
)"
    fi
    "$MODELOPT_RUNTIME/bin/python" - "$node_root/source" <<'PY'
import sys
from pathlib import Path

import accelerate, datasets, modelopt, wandb

source = Path(sys.argv[1]).resolve()
origin = Path(modelopt.__file__).resolve()
if not origin.is_relative_to(source):
    raise RuntimeError(f"modelopt imported outside staged source: {origin}")
PY
    if [[ "$METHOD" == dflash2 ]]; then
        "$MODELOPT_RUNTIME/bin/python" - "$DFLASH2_VLLM_EXPECTED_COMMIT" \
            "$DFLASH2_VLLM_REQUIRED_ANCESTOR" "$DFLASH2_VLLM_RECEIPT_RELATIVE_PATH" \
            "$DFLASH2_VLLM_RECEIPT_SHA256" <<'PY'
import sys
from pathlib import Path

import vllm

from tools.launcher.common.specdec.dflash2_runtime_contract import verify_vllm_runtime

verify_vllm_runtime(
    Path(vllm.__file__).resolve().parent,
    Path(sys.prefix) / sys.argv[3],
    sys.argv[4],
    sys.argv[1],
    sys.argv[2],
)
PY
    fi
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
    TRAINING_RUN_NAME="$RUN_NAME"
    if [[ "$CLUSTER_NAME" != "oci-hsg" ]]; then
        export WANDB_RUN_GROUP="${RUN_NAME}-${CLUSTER_NAME}"
        TRAINING_RUN_NAME="${RUN_NAME}-${CLUSTER_NAME}"
    fi
    export SERVE_LOG_DIR="${SCRATCH_JOB_ROOT}/node-${SLURM_NODEID}/logs"
    export TMPDIR="${SCRATCH_JOB_ROOT}/node-${SLURM_NODEID}/tmp"
    mkdir -p "$HF_HUB_CACHE" "$HF_DATASETS_CACHE" "$TRITON_CACHE_DIR" "$XDG_CACHE_HOME" "$SQLITE_TMPDIR" "$WANDB_DIR" "$WANDB_CACHE_DIR" "$SERVE_LOG_DIR" "$TMPDIR"
    export SERVE_MODEL_NAME="modelopt-${RUN_NAME}"
    export SERVE_BLOCK_SIZE=32 SERVE_MAX_MODEL_LEN=8192 SERVE_READY_TIMEOUT=1800
    export HS_POOL_SLOTS=32 SERVE_MAX_NUM_SEQS=16
    export EXPORT_PATH
    loss_args=(
        "dflash.dflash_loss_objective=${LOSS_OBJECTIVE}"
        "dflash.dflash_loss_decay_factor=${LOSS_VALUE}"
    )
    if [[ "$METHOD" == dflash ]]; then
        loss_args+=("dflash.dflash_dpace_alpha=${DFLASH_DPACE_ALPHA}")
    fi
    architecture_args=()
    if [[ "$METHOD" == dflash2 ]]; then
        architecture_args+=(
            "dflash.dflash_architecture_config.projector_type=dflash2"
            "dflash.dflash_architecture_config.conv_kernel_size=${DFLASH2_CONV_KERNEL_SIZE}"
            "dflash.dflash_architecture_config.conv_group_size=${DFLASH2_CONV_GROUP_SIZE}"
            "dflash.dflash_architecture_config.selector_rank=${DFLASH2_SELECTOR_RANK}"
            "dflash.dflash_architecture_config.selector_top_k=${DFLASH2_SELECTOR_TOP_K}"
        )
    fi
    cd "$node_root/source/tools/launcher"
    exec bash common/eagle3/train_eagle_streaming.sh \
        --config "modules/Model-Optimizer/modelopt_recipes/general/speculative_decoding/${METHOD}.yaml" \
        model.model_name_or_path="$HF_MODEL_CKPT" \
        model.use_fake_base_for_offline=true \
        data.mode=streaming \
        data.data_path="$node_root/input/$DATASET_LOCAL_NAME" \
        data.sample_size="${SAMPLE_SIZE}" \
        training.output_dir="${OUTPUT_ROOT}" \
        training.max_steps="${MAX_STEPS}" \
        training.save_steps="${SAVE_STEPS}" \
        training.save_total_limit=2 \
        training.training_seq_len=4096 \
        training.answer_only_loss=false \
        training.seed=42 \
        training.per_device_train_batch_size="${PER_DEVICE_TRAIN_BATCH_SIZE}" \
        training.gradient_accumulation_steps="${GRADIENT_ACCUMULATION_STEPS}" \
        training.report_to=wandb \
        training.run_name="${TRAINING_RUN_NAME}-s${MAX_STEPS}" \
        dflash.dflash_block_size="${BLOCK_SIZE}" \
        dflash.dflash_num_anchors=512 \
        dflash.dflash_mask_token_id=151669 \
        dflash.dflash_architecture_config.num_attention_heads="${NUM_ATTENTION_HEADS}" \
        dflash.dflash_architecture_config.num_key_value_heads="${NUM_KEY_VALUE_HEADS}" \
        dflash.dflash_architecture_config.head_dim="${HEAD_DIM}" \
        dflash.dflash_architecture_config.intermediate_size="${INTERMEDIATE_SIZE}" \
        "${architecture_args[@]}" \
        "${loss_args[@]}"
')

if [[ "${SELF_REQUEUE:-0}" == 1 ]]; then
    drafter_run_requeueable_step "${training_step[@]}"
    drafter_mark_training_complete
else
    "${training_step[@]}"
fi
