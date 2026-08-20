#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Verify that a pinned runtime archive works after node-local extraction.
set -euo pipefail

SCRIPT_PATH="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/$(basename "${BASH_SOURCE[0]}")"
MODE="outer"
SOURCE_PATH=""
RUNTIME_ARCHIVE=""
RUNTIME_SHA256=""
IMAGE_PATH=""
SCRATCH_ROOT="/raid/scratch/${USER}/modelopt-runtime-probe-${SLURM_JOB_ID:-local}"

usage() {
    echo "usage: $0 --source-path /home/... --runtime-archive /lustre/... --runtime-sha256 SHA256 --image /lustre/... [--scratch-root /raid/scratch/...]" >&2
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
        *) usage ;;
    esac
done

[[ "$SOURCE_PATH" == /home/* && "$RUNTIME_ARCHIVE" == /lustre/* && "$RUNTIME_SHA256" =~ ^[0-9a-f]{64}$ && "$IMAGE_PATH" == /lustre/* && "$SCRATCH_ROOT" == /raid/scratch/* ]] || usage

if [[ "$MODE" == "outer" ]]; then
    [[ "$SCRIPT_PATH" == /home/* ]] || { echo "probe script must be run from /home source" >&2; exit 2; }
    [[ -f "$RUNTIME_ARCHIVE" && -f "$IMAGE_PATH" ]] || { echo "pinned archive or image is missing" >&2; exit 2; }
    [[ "$(sha256sum "$RUNTIME_ARCHIVE" | cut -d' ' -f1)" == "$RUNTIME_SHA256" ]] || { echo "runtime archive SHA-256 mismatch" >&2; exit 2; }
    srun --nodes=1 --ntasks=1 --no-container-mount-home --container-image="$IMAGE_PATH" \
        --container-mounts="${SCRIPT_PATH}:${SCRIPT_PATH},${SOURCE_PATH}:${SOURCE_PATH},${RUNTIME_ARCHIVE}:${RUNTIME_ARCHIVE},/raid/scratch:/raid/scratch" \
        bash "$SCRIPT_PATH" --inside --source-path "$SOURCE_PATH" --runtime-archive "$RUNTIME_ARCHIVE" --runtime-sha256 "$RUNTIME_SHA256" --image "$IMAGE_PATH" --scratch-root "$SCRATCH_ROOT"
    exit 0
fi

node_root="${SCRATCH_ROOT}/node-${SLURM_NODEID:-0}"
rm -rf "$node_root"
mkdir -p "$node_root/runtime"
cp -aL "$SOURCE_PATH" "$node_root/source"
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
import sys
from pathlib import Path

import accelerate, datasets, modelopt, wandb

source = Path(sys.argv[1]).resolve()
origin = Path(modelopt.__file__).resolve()
if not origin.is_relative_to(source):
    raise RuntimeError(f"modelopt imported outside staged source: {origin}")
PY
echo "relocatable runtime probe passed: $MODELOPT_RUNTIME"
