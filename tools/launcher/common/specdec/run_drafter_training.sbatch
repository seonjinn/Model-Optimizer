#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#SBATCH --ntasks-per-node=1
#SBATCH --time=03:55:00

# Execute one pinned streaming-drafter wave. The submitter supplies all inputs.
set -euo pipefail

readonly GLOBAL_BATCH_SIZE=512

# Where durable artifacts may live is a per-cluster fact, not a constant:
# OCI-HSG, Ptyche and Lyris mount Lustre at /lustre, AWS-CMH and OCI-AGA at
# /scratch/fsw. An allow-list of one prefix refuses to run on two of the five
# clusters, so state the rule the MARS storage guidance actually gives -- name
# the storage an artifact must NOT outlive the job on -- and let any shared
# mount satisfy it.
is_durable_path() {
    case "${1:-}" in
        /home/*|/raid/*|/tmp/*|/var/*|/cm/*|/dev/shm/*) return 1 ;;
        /*) return 0 ;;
        *) return 1 ;;
    esac
}

is_node_local_path() {
    case "${1:-}" in
        /lustre/*|/lustre|/scratch/fsw/*|/scratch/fsw|/home/*|/home) return 1 ;;
        /*) return 0 ;;
        *) return 1 ;;
    esac
}

for name in MANIFEST_PATH EXPERIMENT_INDEX MAX_STEPS LAUNCHER_ROOT CLUSTER_NAME CLUSTER_PROFILE_PATH CLUSTER_PROFILE_SHA256 SCRATCH_ROOT; do
    [[ -n "${!name:-}" ]] || { echo "missing required environment: $name" >&2; exit 2; }
done
readonly LAUNCHER_ROOT
[[ "$CLUSTER_PROFILE_PATH" == /home/* && -f "$CLUSTER_PROFILE_PATH" ]] || { echo "cluster profile is missing" >&2; exit 2; }
[[ "$(sha256sum "$CLUSTER_PROFILE_PATH" | cut -d' ' -f1)" == "$CLUSTER_PROFILE_SHA256" ]] || { echo "cluster profile SHA-256 mismatch" >&2; exit 2; }
if [[ -n "${READINESS_RECEIPT:-}" ]]; then
    is_durable_path "$READINESS_RECEIPT" && [[ -f "$READINESS_RECEIPT" && -n "${READINESS_RECEIPT_SHA256:-}" ]] || { echo "readiness receipt is missing" >&2; exit 2; }
    [[ "$(sha256sum "$READINESS_RECEIPT" | cut -d' ' -f1)" == "$READINESS_RECEIPT_SHA256" ]] || { echo "readiness receipt SHA-256 mismatch" >&2; exit 2; }
elif [[ "$CLUSTER_NAME" != "oci-hsg" ]]; then
    echo "non-OCI training requires a readiness receipt" >&2
    exit 2
fi
is_node_local_path "$SCRATCH_ROOT" || { echo "scratch root must be node-local, got: $SCRATCH_ROOT" >&2; exit 2; }
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

# training.report_to=wandb further down is unconditional, and the netrc branch
# that exports WANDB_API_KEY only runs when a netrc was configured. Without one
# the run depends on the key being inherited from the job environment, which is
# how the clusters are actually set up -- but if it ever is not there, the
# failure surfaces inside the container with a 16-node allocation already
# standing, or the run trains for a day logging nowhere. Decide it here, in the
# first seconds of the job, and say which source won. The key is never echoed;
# only its shape is checked.
if [[ -z "$WANDB_NETRC_CONTAINER_PATH" ]]; then
    [[ -n "${WANDB_API_KEY:-}" ]] || {
        echo "no wandb credential: export WANDB_API_KEY or set WANDB_NETRC_PATH" >&2
        exit 2
    }
    # Deliberately loose: printable, no whitespace, long enough. The job of
    # this check is to catch a *missing* credential. A rule tight enough to
    # reject every malformed key also rejects unusual but valid ones, which
    # blocks the study, while a malformed key merely fails at wandb init with a
    # clear message. This was ^[A-Za-z0-9]{40,}$ until it rejected a live
    # 86-character key that wandb itself accepts, killing every arm on a new
    # cluster in preflight. Do not re-tighten it to a character whitelist.
    [[ "${WANDB_API_KEY}" =~ ^[[:graph:]]{40,}$ ]] || {
        echo "WANDB_API_KEY is not shaped like a wandb key" >&2
        exit 2
    }
    echo "wandb credential source: WANDB_API_KEY from the job environment"
else
    echo "wandb credential source: $WANDB_NETRC_PATH"
fi

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

uses_topology_v2 = (
    experiment.topology.target_kind == "qwen3-30b-a3b" or experiment.slurm.nodes != 4
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
):
    print(value)
PY
)
(( ${#manifest_values[@]} == 28 )) || { echo "manifest entry is unavailable" >&2; exit 2; }
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
METHOD="${manifest_values[9]}"
BLOCK_SIZE="${manifest_values[10]}"
RUN_NAME="${manifest_values[11]}"
SAMPLE_SIZE="${manifest_values[12]}"
EAGLE_CAPTURE_IDS="[${manifest_values[13]}]"
SERVE_TP="${manifest_values[14]}"
PER_DEVICE_TRAIN_BATCH_SIZE="${manifest_values[15]}"
GRADIENT_ACCUMULATION_STEPS="${manifest_values[16]}"
NUM_ATTENTION_HEADS="${manifest_values[17]}"
NUM_KEY_VALUE_HEADS="${manifest_values[18]}"
HEAD_DIM="${manifest_values[19]}"
INTERMEDIATE_SIZE="${manifest_values[20]}"
TARGET_KIND="${manifest_values[21]}"
ALLOCATED_NODES="${manifest_values[22]}"
ALLOCATION_SEGMENT="${manifest_values[23]}"
GPUS_PER_NODE="${manifest_values[24]}"
TRAINING_IDENTITY="${manifest_values[25]}"
TRAINING_FINGERPRINT="${manifest_values[26]}"
TRAINING_FINGERPRINT_SCHEMA="${manifest_values[27]}"
SERVE_NODES=$((ALLOCATED_NODES / 2))
TRAINER_NODES=$((ALLOCATED_NODES - SERVE_NODES))
EXPECTED_RNG_STATES=$((TRAINER_NODES * GPUS_PER_NODE))
readonly ALLOCATED_NODES ALLOCATION_SEGMENT GPUS_PER_NODE SERVE_NODES TRAINER_NODES EXPECTED_RNG_STATES TRAINING_FINGERPRINT_SCHEMA
[[ "${SLURM_NNODES:-}" == "$ALLOCATED_NODES" ]] || { echo "Slurm allocation does not match manifest nodes" >&2; exit 2; }
echo "training allocation: nodes=${ALLOCATED_NODES} segment=${ALLOCATION_SEGMENT} serve=${SERVE_NODES} trainer=${TRAINER_NODES}"
for name in SOURCE_PATH SOURCE_SHA RUNTIME_ARCHIVE TARGET_PATH DATASET_PATH OUTPUT_ROOT METHOD BLOCK_SIZE RUN_NAME; do
    [[ -n "${!name:-}" ]] || { echo "missing manifest value: $name" >&2; exit 2; }
done
[[ "$SOURCE_PATH" == /home/* ]] || { echo "source checkout must be in /home, got: $SOURCE_PATH" >&2; exit 2; }
for durable in IMAGE_PATH RUNTIME_ARCHIVE TARGET_PATH DATASET_PATH OUTPUT_ROOT; do
    is_durable_path "${!durable}" || { echo "$durable must be on shared storage, got: ${!durable}" >&2; exit 2; }
done
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
# eagle_utils takes two different loading branches off the sign of sample_size.
# A positive value materializes the whole corpus into a Python list, once per
# rank and again per dataloader worker; at 44.9 KB a row the 1,179,389-row PTV3
# corpus is 54 GB a rank, which OOM-killed every task on a 900 GB node. A
# non-positive value takes the Arrow memory-mapped branch, which keeps every row
# but shares one mapping across all processes. Only -1 is accepted so this can
# never become a way to quietly train on fewer rows than the manifest pins.
TRAINING_SAMPLE_SIZE="${TRAINING_SAMPLE_SIZE:-$SAMPLE_SIZE}"
[[ "$TRAINING_SAMPLE_SIZE" == "$SAMPLE_SIZE" || "$TRAINING_SAMPLE_SIZE" == "-1" ]] || {
    echo "TRAINING_SAMPLE_SIZE must be either the pinned ${SAMPLE_SIZE} or -1" >&2; exit 2;
}
# The training window belongs to the run, not to the submitting shell. It used
# to default to 4096 whenever the caller forgot to export it, and nothing in the
# job disagreed: five PTV3-SWE legs resumed a 16384 arm at 4096 and retrained two
# thirds of two arms on a window the corpus was not built for. No error, no
# warning, no record in the receipt; it surfaced only when a benchmark disagreed
# weeks later. Derive the requirement from the pinned corpus instead, and refuse
# a caller who contradicts it.
case "$DATASET_PATH" in
    *ptv3swe*) REQUIRED_SEQ_LEN=16384 ;;
    *)         REQUIRED_SEQ_LEN=4096 ;;
esac
TRAINING_SEQ_LEN="${TRAINING_SEQ_LEN:-$REQUIRED_SEQ_LEN}"
[[ "$TRAINING_SEQ_LEN" =~ ^[1-9][0-9]*$ ]] || { echo "TRAINING_SEQ_LEN must be positive" >&2; exit 2; }
(( TRAINING_SEQ_LEN == REQUIRED_SEQ_LEN )) || {
    echo "TRAINING_SEQ_LEN ${TRAINING_SEQ_LEN} contradicts the pinned corpus" >&2
    echo "  corpus:   ${DATASET_PATH}" >&2
    echo "  requires: ${REQUIRED_SEQ_LEN}" >&2
    exit 2
}
# Each leg of the incident above was internally consistent; only the arm as a
# whole was not. A per-arm record is the one thing that could have caught it,
# so pin the window to the output root and refuse to continue an arm at a
# window it was not trained at.
WINDOW_LOCK="$OUTPUT_ROOT/training_window.json"
if [[ -f "$WINDOW_LOCK" ]]; then
    locked_seq_len="$(sed -nE 's/.*"training_seq_len"[[:space:]]*:[[:space:]]*([0-9]+).*/\1/p' "$WINDOW_LOCK" | head -n 1)"
    [[ "$locked_seq_len" == "$TRAINING_SEQ_LEN" ]] || {
        echo "arm ${RUN_NAME} was trained at ${locked_seq_len}; refusing to continue it at ${TRAINING_SEQ_LEN}" >&2
        echo "remove ${WINDOW_LOCK} only when the whole arm is being restarted" >&2
        exit 2
    }
else
    mkdir -p "$OUTPUT_ROOT"
    printf '{"training_seq_len": %s, "dataset_path": "%s", "run_name": "%s"}\n' \
        "$TRAINING_SEQ_LEN" "$DATASET_PATH" "$RUN_NAME" >"$WINDOW_LOCK"
fi
# eagle_utils right-truncates to train_len and the sidecar refuses a longer
# prompt, so a train_len above the serve limit silently trains on truncated
# rows while the capture path errors. Raise the serve limit with the window
# rather than leaving the caller to keep two exports in sync; the 8192 floor
# preserves what every pre-existing 4096 arm was served with.
if (( TRAINING_SEQ_LEN > 8192 )); then
    SERVE_MAX_MODEL_LEN="${SERVE_MAX_MODEL_LEN:-$TRAINING_SEQ_LEN}"
else
    SERVE_MAX_MODEL_LEN="${SERVE_MAX_MODEL_LEN:-8192}"
fi
(( TRAINING_SEQ_LEN <= SERVE_MAX_MODEL_LEN )) || { echo "TRAINING_SEQ_LEN ${TRAINING_SEQ_LEN} exceeds SERVE_MAX_MODEL_LEN ${SERVE_MAX_MODEL_LEN}" >&2; exit 2; }
export SERVE_MAX_MODEL_LEN
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
    dflash2:8)
        NUM_SPEC_TOKENS=7
        LOSS_OBJECTIVE=decay
        LOSS_VALUE=4
        DFLASH_DPACE_ALPHA=""
        ;;
    dflash2:16)
        NUM_SPEC_TOKENS=15
        LOSS_OBJECTIVE=decay
        LOSS_VALUE=7
        DFLASH_DPACE_ALPHA=""
        ;;
    *) echo "unsupported method/block size: ${METHOD}/${BLOCK_SIZE}" >&2; exit 2 ;;
esac
export SERVE_NODES NUM_SPEC_TOKENS
export SERVE_REPLICAS_PER_NODE

# modelopt/__init__.py resolves __version__ through importlib.metadata, so the
# package cannot be imported from PYTHONPATH alone -- it needs an installed
# distribution, and neither the container nor the runtime archive carries one.
# The staging step writes the metadata beside the staged source instead of
# installing anything, which keeps modelopt resolving to the pinned checkout
# that the preflight insists on. Derive the version the way a real build would:
# setuptools_scm reads git describe, and the checkout is cloned without tags, so
# the project's own declared fallback is the honest answer.
SOURCE_DISTRIBUTION_VERSION="$(
    describe="$(git -C "$SOURCE_PATH" describe --tags --long --match "[0-9]*" 2>/dev/null || true)"
    if [[ "$describe" =~ ^([0-9][^-]*)-([0-9]+)-g([0-9a-f]+)$ ]]; then
        printf "%s.dev%s+g%s" "${BASH_REMATCH[1]}" "${BASH_REMATCH[2]}" "${BASH_REMATCH[3]}"
    else
        sed -nE 's/^fallback_version[[:space:]]*=[[:space:]]*"([^"]+)".*/\1/p' \
            "$SOURCE_PATH/pyproject.toml" | head -n 1
    fi
)"
[[ -n "$SOURCE_DISTRIBUTION_VERSION" ]] || {
    echo "cannot determine the nvidia-modelopt distribution version" >&2
    exit 2
}
readonly SOURCE_DISTRIBUTION_VERSION
echo "staged modelopt distribution version: $SOURCE_DISTRIBUTION_VERSION"

export SOURCE_DISTRIBUTION_VERSION
export SOURCE_PATH SOURCE_SHA IMAGE_PATH RUNTIME_ARCHIVE RUNTIME_ARCHIVE_SHA256 TARGET_PATH DATASET_PATH DATASET_LOCAL_NAME OUTPUT_ROOT EXPORT_PATH SCRATCH_JOB_ROOT STAGE_LOG_ROOT EXPECTED_RNG_STATES TRAINING_IDENTITY TRAINING_FINGERPRINT
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
    stage_fingerprint="${SOURCE_SHA}|${RUNTIME_ARCHIVE_SHA256}|${TARGET_PATH}|${DATASET_PATH}|${stage_role}"
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

export TRAINING_SEQ_LEN TRAINING_SAMPLE_SIZE
export PER_DEVICE_TRAIN_BATCH_SIZE GRADIENT_ACCUMULATION_STEPS NUM_ATTENTION_HEADS NUM_KEY_VALUE_HEADS HEAD_DIM INTERMEDIATE_SIZE EAGLE_CAPTURE_IDS SERVE_TP LOSS_OBJECTIVE LOSS_VALUE DFLASH_DPACE_ALPHA METHOD BLOCK_SIZE MAX_STEPS SAVE_STEPS RUN_NAME SAMPLE_SIZE
export SERVE_NODES CLUSTER_NAME WANDB_NETRC_CONTAINER_PATH WANDB_API_KEY
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
        # Assign first, export second. An export reports its own status, not
        # the status of the command substitution feeding it, so under set -e a
        # failed read here would sail through as an empty key -- and because
        # this branch overwrites whatever was inherited, it would also destroy
        # a good key that arrived from the job environment. On a 16-node run
        # the mount only has to fail on one node for that to happen.
        # No apostrophes in this block: it lives inside a single-quoted body.
        wandb_api_key="$("$MODELOPT_RUNTIME/bin/python" - "$WANDB_NETRC_CONTAINER_PATH" <<'PY'
import netrc
import sys

credentials = netrc.netrc(sys.argv[1]).authenticators("api.wandb.ai")
if credentials is None or not credentials[2]:
    raise SystemExit("mounted WANDB netrc has no api.wandb.ai credential")
print(credentials[2])
PY
)"
        [[ -n "$wandb_api_key" ]] || { echo "netrc read produced an empty wandb key" >&2; exit 2; }
        export WANDB_API_KEY="$wandb_api_key"
    fi
    "$MODELOPT_RUNTIME/bin/python" - "$node_root/source" <<'PY'
import sys
from pathlib import Path

import accelerate, datasets, wandb
import modelopt

# modelopt/__init__.py is two lines, so `import modelopt` only proves the
# distribution metadata resolves. The trainer and the serve side both reach
# modelopt.torch, whose closure pulls omegaconf, pulp and scipy -- none of
# which the container carries. Import what the run imports, or the preflight
# passes and the job dies minutes later on a missing third-party module.
import modelopt.torch.speculative

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
    TRAINING_RUN_NAME="$RUN_NAME"
    if [[ "$CLUSTER_NAME" != "oci-hsg" ]]; then
        export WANDB_RUN_GROUP="${RUN_NAME}-${CLUSTER_NAME}"
        TRAINING_RUN_NAME="${RUN_NAME}-${CLUSTER_NAME}"
    fi
    export SERVE_LOG_DIR="${SCRATCH_JOB_ROOT}/node-${SLURM_NODEID}/logs"
    export TMPDIR="${SCRATCH_JOB_ROOT}/node-${SLURM_NODEID}/tmp"
    mkdir -p "$HF_HUB_CACHE" "$HF_DATASETS_CACHE" "$TRITON_CACHE_DIR" "$XDG_CACHE_HOME" "$SQLITE_TMPDIR" "$WANDB_DIR" "$WANDB_CACHE_DIR" "$SERVE_LOG_DIR" "$TMPDIR"
    export SERVE_MODEL_NAME="modelopt-${RUN_NAME}"
    # A longer-context phase (PTV3 agentic traces at 16384) has to raise the
    # sidecar limit in lockstep with training_seq_len, and a bigger KV pool
    # takes longer to come up. Defaults keep every PTV2 arm byte-identical.
    export SERVE_BLOCK_SIZE=32
    export SERVE_MAX_MODEL_LEN="${SERVE_MAX_MODEL_LEN:-8192}"
    export SERVE_READY_TIMEOUT="${SERVE_READY_TIMEOUT:-1800}"
    export HS_POOL_SLOTS=32 SERVE_MAX_NUM_SEQS=16
    # The hidden-state sidecar has to bind a port the kernel will never hand out
    # as an ephemeral source port. OCI-HSG and aws-cmh set ip_local_port_range to
    # 9000-65000, so the 18999 default sat inside it and a concurrent NCCL/NIXL
    # socket could win the race: one replica dies on EADDRINUSE, the trainer
    # waits forever for an address that never publishes, and the afterok chain
    # unwinds behind it. 8700 is below every low bound in the study and clear of
    # the 8765+ serve API ports. Keep apostrophes out of this block: it sits
    # inside a single-quoted bash -c body, so one would terminate the string and
    # break the script at submit time rather than at review time.
    export HS_SIDECAR_PORT="${HS_SIDECAR_PORT:-8700}"
    export EXPORT_PATH
    loss_args=(
        "dflash.dflash_loss_objective=${LOSS_OBJECTIVE}"
        "dflash.dflash_loss_decay_factor=${LOSS_VALUE}"
    )
    if [[ "$METHOD" == dflash ]]; then
        loss_args+=("dflash.dflash_dpace_alpha=${DFLASH_DPACE_ALPHA}")
    fi
    # Study hook: space-separated Hydra overrides appended last, so a parameter-sweep
    # cell can change the objective or the schedule without forking the recipe file.
    extra_args=()
    if [[ -n "${DFLASH_EXTRA_TRAIN_ARGS:-}" ]]; then
        read -ra extra_args <<< "${DFLASH_EXTRA_TRAIN_ARGS}"
    fi
    cd "$node_root/source/tools/launcher"
    exec bash common/eagle3/train_eagle_streaming.sh \
        --config "modules/Model-Optimizer/modelopt_recipes/general/speculative_decoding/${METHOD}.yaml" \
        model.model_name_or_path="$HF_MODEL_CKPT" \
        model.use_fake_base_for_offline=true \
        data.mode=streaming \
        data.data_path="$node_root/input/$DATASET_LOCAL_NAME" \
        data.sample_size="${TRAINING_SAMPLE_SIZE}" \
        training.output_dir="${OUTPUT_ROOT}" \
        training.max_steps="${MAX_STEPS}" \
        training.save_steps="${SAVE_STEPS}" \
        training.save_total_limit=2 \
        training.training_seq_len="${TRAINING_SEQ_LEN}" \
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
        "${loss_args[@]}" \
        ${extra_args[@]+"${extra_args[@]}"}
')

if [[ "${SELF_REQUEUE:-0}" == 1 ]]; then
    drafter_run_requeueable_step "${training_step[@]}"
    drafter_mark_training_complete
else
    "${training_step[@]}"
fi
