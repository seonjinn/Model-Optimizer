#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Verify that a pinned runtime archive works after node-local extraction.
set -euo pipefail

# Shared parallel storage is mounted at /lustre on OCI-HSG, Ptyche and Lyris
# and at /scratch/fsw on AWS-CMH and OCI-AGA, so requiring one prefix refuses
# to run on two of the five clusters. Name the storage a durable artifact must
# NOT live on instead -- node-local scratch that vanishes with the job, and the
# NFS home the MARS guidance reserves for source.
is_durable_path() {
    case "${1:-}" in
        /home/*|/raid/*|/tmp/*|/var/*|/cm/*|/dev/shm/*) return 1 ;;
        /*) return 0 ;;
        *) return 1 ;;
    esac
}

SCRIPT_PATH="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/$(basename "${BASH_SOURCE[0]}")"
LAUNCHER_ROOT="${DRAFTER_LAUNCHER_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
MODE="outer"
SOURCE_PATH=""
RUNTIME_ARCHIVE=""
RUNTIME_SHA256=""
IMAGE_PATH=""
SCRATCH_ROOT="/raid/scratch/${USER}/modelopt-runtime-probe-${SLURM_JOB_ID:-local}"
ACCOUNT="nemotron_n3_post"
PARTITION="batch"
PROBE_LOG=""
CLUSTER_PROFILE=""
READINESS_RECEIPT=""

usage() {
    echo "usage: $0 --source-path /home/... --runtime-archive /lustre/... --runtime-sha256 SHA256 --image /lustre/... [--cluster-profile PATH --readiness-receipt PATH] [--scratch-root PATH] [--account ACCOUNT] [--partition PARTITION]" >&2
    exit 2
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --inside) MODE="inside"; shift ;;
        --source-path) SOURCE_PATH="$2"; shift 2 ;;
        --runtime-archive) RUNTIME_ARCHIVE="$2"; shift 2 ;;
        --runtime-sha256) RUNTIME_SHA256="$2"; shift 2 ;;
        --image) IMAGE_PATH="$2"; shift 2 ;;
        --scratch-root) SCRATCH_ROOT="$2"; shift 2 ;;
        --account) ACCOUNT="$2"; shift 2 ;;
        --partition) PARTITION="$2"; shift 2 ;;
        --probe-log) PROBE_LOG="$2"; shift 2 ;;
        --cluster-profile) CLUSTER_PROFILE="$2"; shift 2 ;;
        --readiness-receipt) READINESS_RECEIPT="$2"; shift 2 ;;
        *) usage ;;
    esac
done

PROFILE_DURABLE_ROOT=""
PROFILE_SCRATCH_BASE=""
PROFILE_GPU_ARGS=(--gpus-per-node=4)
PROFILE_EVAL_NODES=1
PROFILE_EVAL_SEGMENT=1
if [[ -n "$CLUSTER_PROFILE" || -n "$READINESS_RECEIPT" ]]; then
    [[ -n "$CLUSTER_PROFILE" && -n "$READINESS_RECEIPT" ]] || usage
    mapfile -t profile_values < <(PYTHONPATH="${LAUNCHER_ROOT}${PYTHONPATH:+:${PYTHONPATH}}" python3 - \
        "$CLUSTER_PROFILE" "$READINESS_RECEIPT" <<'PY'
import json
import sys
from pathlib import Path
from common.specdec.cluster_profile import load_cluster_profile, scheduler_gpu_args, validate_scratch_root
profile = load_cluster_profile(Path(sys.argv[1]).resolve())
receipt = json.loads(Path(sys.argv[2]).resolve().read_text())
expected = {"profile": profile.name, "account": profile.account, "partition": profile.partition,
            "pyxis_available": True, "gpu_count": profile.gpus_per_node, "architecture": "aarch64"}
if any(receipt.get(key) != value for key, value in expected.items()):
    raise ValueError("cluster readiness receipt does not match profile")
scratch = Path(receipt.get("scratch_root", ""))
validate_scratch_root(profile, scratch)
print(profile.account)
print(profile.partition)
print(profile.durable_root)
print(scratch)
print(profile.evaluation_nodes)
print(profile.evaluation_segment)
for argument in scheduler_gpu_args(profile): print(argument)
PY
    )
    (( ${#profile_values[@]} >= 6 )) || { echo "invalid cluster readiness contract" >&2; exit 2; }
    ACCOUNT="${profile_values[0]}"
    PARTITION="${profile_values[1]}"
    PROFILE_DURABLE_ROOT="${profile_values[2]}"
    PROFILE_SCRATCH_BASE="${profile_values[3]}"
    SCRATCH_ROOT="${PROFILE_SCRATCH_BASE%/}/${USER}/modelopt-runtime-probe"
    PROFILE_EVAL_NODES="${profile_values[4]}"
    PROFILE_EVAL_SEGMENT="${profile_values[5]}"
    PROFILE_GPU_ARGS=("${profile_values[@]:6}")
fi

[[ "$SOURCE_PATH" == /home/* && "$RUNTIME_SHA256" =~ ^[0-9a-f]{64}$ ]] || usage
if [[ -n "$PROFILE_DURABLE_ROOT" ]]; then
    [[ "$(realpath -m -- "$RUNTIME_ARCHIVE")" == "$PROFILE_DURABLE_ROOT"/* ]] || usage
    [[ "$(realpath -m -- "$IMAGE_PATH")" == "$PROFILE_DURABLE_ROOT"/* ]] || usage
else
    is_durable_path "$RUNTIME_ARCHIVE" && is_durable_path "$IMAGE_PATH" \
        && [[ "$SCRATCH_ROOT" == /raid/scratch/* ]] || usage
fi

if [[ "$MODE" == "outer" ]]; then
    [[ "$SCRIPT_PATH" == /home/* ]] || { echo "probe script must be run from /home source" >&2; exit 2; }
    [[ -f "$RUNTIME_ARCHIVE" && -f "$IMAGE_PATH" ]] || { echo "pinned archive or image is missing" >&2; exit 2; }
    [[ "$(sha256sum "$RUNTIME_ARCHIVE" | cut -d' ' -f1)" == "$RUNTIME_SHA256" ]] || { echo "runtime archive SHA-256 mismatch" >&2; exit 2; }
    PROBE_LOG="${PROBE_LOG:-$(dirname "$RUNTIME_ARCHIVE")/probes/runtime-probe-%j.out}"
    RUNTIME_ARCHIVE_ROOT="$(dirname "$RUNTIME_ARCHIVE")"
    is_durable_path "$PROBE_LOG" || usage
    mkdir -p "$(dirname "$PROBE_LOG")"
    if [[ -n "$CLUSTER_PROFILE" ]]; then
        args=(--account="$ACCOUNT" --partition="$PARTITION" --nodes="$PROFILE_EVAL_NODES" --ntasks=1
            "${PROFILE_GPU_ARGS[@]}" --segment="$PROFILE_EVAL_SEGMENT" --time=00:10:00
            --job-name=modelopt-runtime-probe --output="$PROBE_LOG" --error="$PROBE_LOG"
            "--export=ALL,DRAFTER_LAUNCHER_ROOT=$LAUNCHER_ROOT"
            --no-container-mount-home --container-image="$IMAGE_PATH"
            --container-mounts="${SOURCE_PATH}:${SOURCE_PATH},${RUNTIME_ARCHIVE_ROOT}:${RUNTIME_ARCHIVE_ROOT},${PROFILE_SCRATCH_BASE}:${PROFILE_SCRATCH_BASE}")
        command=(bash "$SCRIPT_PATH" --inside --source-path "$SOURCE_PATH"
            --runtime-archive "$RUNTIME_ARCHIVE" --runtime-sha256 "$RUNTIME_SHA256"
            --image "$IMAGE_PATH" --scratch-root "$SCRATCH_ROOT"
            --cluster-profile "$CLUSTER_PROFILE" --readiness-receipt "$READINESS_RECEIPT")
        sbatch --test-only "${args[@]}" "${command[@]}"
        submitted="$(sbatch --parsable "${args[@]}" "${command[@]}")"
        echo "${submitted%%;*}"
    else
        srun --account="$ACCOUNT" --partition="$PARTITION" --nodes=1 --ntasks=1 --gpus-per-node=4 --segment=1 --time=00:10:00 \
            --job-name=modelopt-runtime-probe --output="$PROBE_LOG" --error="$PROBE_LOG" \
            --no-container-mount-home \
            --container-image="$IMAGE_PATH" \
            --container-mounts="${SOURCE_PATH}:${SOURCE_PATH},${RUNTIME_ARCHIVE_ROOT}:${RUNTIME_ARCHIVE_ROOT},/raid/scratch:/raid/scratch" \
            bash "$SCRIPT_PATH" --inside --source-path "$SOURCE_PATH" --runtime-archive "$RUNTIME_ARCHIVE" --runtime-sha256 "$RUNTIME_SHA256" --image "$IMAGE_PATH" --scratch-root "$SCRATCH_ROOT"
    fi
    echo "runtime probe log: $PROBE_LOG"
    exit 0
fi

TRACE_LOG="$(dirname "$RUNTIME_ARCHIVE")/probes/runtime-probe-${SLURM_JOB_ID}.trace"
exec >>"$TRACE_LOG" 2>&1
set -x
if [[ -n "$CLUSTER_PROFILE" ]]; then
    node_root="${SCRATCH_ROOT}/${SLURM_JOB_ID}/node-${SLURM_NODEID:-0}"
else
    node_root="${SCRATCH_ROOT}/node-${SLURM_NODEID:-0}"
fi
rm -rf "$node_root"
mkdir -p "$node_root/runtime"
cp -a "$SOURCE_PATH" "$node_root/source"
tar --extract --file="$RUNTIME_ARCHIVE" --directory="$node_root/runtime"
old_venv="$(sed -nE "s/^[[:space:]]*export[[:space:]]+VIRTUAL_ENV=(.*)$/\\1/p" "$node_root/runtime/bin/activate" | head -n 1)"
old_venv="${old_venv#\"}"
old_venv="${old_venv%\"}"
[[ -n "$old_venv" ]] || { echo "runtime archive has no VIRTUAL_ENV" >&2; exit 1; }
grep -IlZ "$old_venv" "$node_root/runtime/bin"/* "$node_root/runtime/pyvenv.cfg" 2>/dev/null \
    | xargs -0 -r sed -i "s|$old_venv|$node_root/runtime|g"
export MODELOPT_RUNTIME="$node_root/runtime"
export VIRTUAL_ENV="$MODELOPT_RUNTIME"
export PATH="$VIRTUAL_ENV/bin:$PATH"
export PYTHONPATH="$node_root/source${PYTHONPATH:+:$PYTHONPATH}"
"$MODELOPT_RUNTIME/bin/python" - "$node_root/source" <<'PY'
import importlib
import sys
from pathlib import Path

import accelerate, datasets, modelopt, wandb

source = Path(sys.argv[1]).resolve()
origin = Path(modelopt.__file__).resolve()
if not origin.is_relative_to(source):
    raise RuntimeError(f"modelopt imported outside staged source: {origin}")
# The archive carries its own vLLM and transformers but no torch, and its venv
# precedes the container's site-packages, so the stack that actually serves is
# the archive's Python bound to the image's libtorch. Nothing anywhere pins
# that pair. Report which side each import resolved to rather than assuming a
# matching image was mounted.
runtime = Path(sys.executable).resolve().parent.parent


def _side(module) -> str:
    location = getattr(module, "__file__", None)
    if location and Path(location).resolve().is_relative_to(runtime):
        return "archive"
    return "image"


import torch
import transformers
import vllm

for name, module in (("torch", torch), ("transformers", transformers), ("vllm", vllm)):
    print(f"probe: {name} {module.__version__} from {_side(module)}")

# vLLM's compiled extensions name libtorch by unversioned soname, so they bind
# to whatever torch the image ships. A C++ ABI break surfaces here, at import,
# as an undefined symbol -- which is the whole reason this probe exists. Other
# load failures are reported but not fatal: several .so files in the tree are
# opened through ctypes and are not importable as modules at all.
undefined = []
for library in sorted(Path(vllm.__file__).parent.glob("*.so")):
    name = library.name.split(".")[0]
    try:
        importlib.import_module(f"vllm.{name}")
    except Exception as error:
        detail = f"{type(error).__name__}: {error}"
        if "undefined symbol" in str(error):
            undefined.append(f"{name}: {detail}")
            print(f"probe: vllm.{name} ABI FAILURE {detail}")
        else:
            print(f"probe: vllm.{name} not importable ({detail})")
    else:
        print(f"probe: vllm.{name} ok")

# The served architecture registry comes from whichever vLLM is on the path, so
# verifying that the image registers the three arms says nothing about what the
# server will accept. Check the one that actually runs.
from vllm.model_executor.models.registry import ModelRegistry

architectures = sorted(ModelRegistry.get_supported_archs())
drafts = [name for name in architectures if "DFlash" in name or "DSpark" in name]
print(f"probe: draft architectures {drafts}")
wanted = {"DFlashDraftModel", "DFlash2DraftModel", "Qwen3DSparkModel"}
missing = sorted(wanted.difference(architectures))
if missing:
    raise RuntimeError(f"serving vLLM does not register: {missing}")
if undefined:
    raise RuntimeError("vLLM extensions failed against the image torch: " + "; ".join(undefined))
PY
echo "relocatable runtime probe passed: $MODELOPT_RUNTIME"
