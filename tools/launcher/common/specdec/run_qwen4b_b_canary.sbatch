#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

#SBATCH --nodes=16
#SBATCH --segment=16
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=96

set -euo pipefail

: "${B_CANARY_MANIFEST:?B_CANARY_MANIFEST is required}"
: "${B_CANARY_READINESS:?B_CANARY_READINESS is required}"
: "${B_CANARY_BUILD_RECEIPT:?B_CANARY_BUILD_RECEIPT is required}"
: "${B_CANARY_OUTPUT:?B_CANARY_OUTPUT is required}"
: "${B_CANARY_EVIDENCE:?B_CANARY_EVIDENCE is required}"
: "${B_CANARY_SERVE_COMMAND:?B_CANARY_SERVE_COMMAND is required}"
: "${B_CANARY_TRAIN_COMMAND:?B_CANARY_TRAIN_COMMAND is required}"
: "${LAUNCHER_ROOT:?LAUNCHER_ROOT is required}"

readonly WANDB_PROJECT="sna-qwen3-4b-dataset-study"
export WANDB_PROJECT

mapfile -t allocated_nodes < <(scontrol show hostnames "$SLURM_NODELIST")
(( ${#allocated_nodes[@]} == 16 )) || { echo "B canary requires 16 allocated nodes" >&2; exit 2; }
serve_nodes=$(IFS=,; echo "${allocated_nodes[*]:0:8}")
train_nodes=$(IFS=,; echo "${allocated_nodes[*]:8:8}")
readonly serve_nodes train_nodes

PYTHONPATH="$LAUNCHER_ROOT${PYTHONPATH:+:$PYTHONPATH}" python3 - \
    "$B_CANARY_MANIFEST" "$B_CANARY_READINESS" "$B_CANARY_BUILD_RECEIPT" \
    "$B_CANARY_OUTPUT" <<'PY'
import sys
from pathlib import Path

from common.specdec.qwen4b_b_canary_manifest import (
    load_b_canary_manifest,
    validate_bound_artifacts,
)

manifest = load_b_canary_manifest(Path(sys.argv[1]))
validate_bound_artifacts(
    manifest,
    readiness_path=Path(sys.argv[2]),
    builder_receipt_path=Path(sys.argv[3]),
    builder_output_path=Path(sys.argv[4]),
)
assert manifest.global_batch_size == 512
assert manifest.active_gpu_ranks == 64
assert manifest.scientific_milestone is False
PY

srun --nodes=8 --nodelist="$serve_nodes" --ntasks=32 --ntasks-per-node=4 \
    --gpus-per-task=1 --exclusive bash -lc "$B_CANARY_SERVE_COMMAND" &
serve_pid=$!
trap 'kill "$serve_pid" 2>/dev/null || true' EXIT INT TERM

srun --nodes=8 --nodelist="$train_nodes" --ntasks=32 --ntasks-per-node=4 \
    --gpus-per-task=1 --exclusive bash -lc "$B_CANARY_TRAIN_COMMAND"
wait "$serve_pid"

PYTHONPATH="$LAUNCHER_ROOT${PYTHONPATH:+:$PYTHONPATH}" python3 - \
    "$B_CANARY_MANIFEST" "$B_CANARY_EVIDENCE" <<'PY'
import json
import sys
from pathlib import Path

from common.specdec.qwen4b_b_canary_manifest import (
    BCanaryEvidence,
    load_b_canary_manifest,
    validate_canary_evidence,
)

manifest = load_b_canary_manifest(Path(sys.argv[1]))
payload = json.loads(Path(sys.argv[2]).read_text())
evidence = BCanaryEvidence(
    finite_loss=payload.get("finite_loss") is True,
    checkpoint_reloaded=payload.get("checkpoint_reloaded") is True,
    drafter_exported=payload.get("drafter_exported") is True,
    evaluator_completed=payload.get("evaluator_completed") is True,
    active_gpu_ranks=tuple(payload.get("active_gpu_ranks", [])),
    scientific_milestone=payload.get("scientific_milestone") is True,
)
validate_canary_evidence(evidence, manifest)
payload["all_64_gpus_active"] = True
payload["scientific_milestone"] = False
Path(sys.argv[2]).write_text(json.dumps(payload, sort_keys=True) + "\n")
PY
