#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

#SBATCH --partition=cpu_datamover
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=96
#SBATCH --mem=0

set -euo pipefail

required=(REPO_ROOT SOURCE_COMMIT TASK9_A_SELECTION TASK9_A_SELECTION_SHA256
    TASK8_PUBLICATION TASK8_PUBLICATION_SHA256 A_CANARY_BUILD_ROOT A_CANARY_MANIFEST
    A_CANARY_ACCOUNT TARGET_REVISION TARGET_PATH CONTAINER_IMAGE WANDB_NETRC_HOST_PATH
    WANDB_DIR WANDB_CACHE_DIR WANDB_CONFIG_DIR WANDB_ARTIFACT_DIR)
for name in "${required[@]}"; do
    [[ -n "${!name:-}" ]] || { echo "missing required environment: $name" >&2; exit 2; }
done
[[ "${SLURM_JOB_ID:-}" =~ ^[0-9]+$ && "${SLURM_CPUS_PER_TASK:-}" == 96 ]] || exit 2
[[ "${SLURM_JOB_PARTITION:-}" == cpu_datamover ]] || exit 2
[[ -z "${SLURM_JOB_GPUS:-}" && -z "${SLURM_GPUS_ON_NODE:-}" ]] || exit 2
case "$A_CANARY_ACCOUNT" in nemotron_sw_post|nemotron_n4_post) ;; *) exit 2 ;; esac
[[ "$(git -C "$REPO_ROOT" rev-parse HEAD)" == "$SOURCE_COMMIT" ]] || exit 2
[[ -z "$(git -C "$REPO_ROOT" status --porcelain)" ]] || exit 2

readonly LAUNCHER_ROOT="$REPO_ROOT/tools/launcher"
readonly SCRATCH_ROOT="${SLURM_TMPDIR:?}/qwen4b-a-builder-${SLURM_JOB_ID}"
[[ ! -e "$SCRATCH_ROOT" && ! -L "$SCRATCH_ROOT" ]] || exit 2
export ARROW_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1

PYTHONPATH="$LAUNCHER_ROOT${PYTHONPATH:+:$PYTHONPATH}" python3 \
    "$LAUNCHER_ROOT/common/specdec/build_qwen4b_a_canary.py" \
    --task8-root "$TASK8_PUBLICATION" \
    --publication-sha256 "$TASK8_PUBLICATION_SHA256" \
    --selection-sha256 "$TASK9_A_SELECTION_SHA256" \
    --source-commit "$SOURCE_COMMIT" \
    --output-root "$A_CANARY_BUILD_ROOT" --scratch-root "$SCRATCH_ROOT" \
    --job-id "$SLURM_JOB_ID" --workers 96

mkdir -p "$(dirname "$A_CANARY_MANIFEST")"
PYTHONPATH="$LAUNCHER_ROOT${PYTHONPATH:+:$PYTHONPATH}" python3 - \
    "$A_CANARY_MANIFEST" "$A_CANARY_BUILD_ROOT" <<'PY'
import hashlib
import json
import os
import sys
from pathlib import Path

from common.specdec.qwen4b_a_canary_manifest import (
    ACanaryManifest,
    ACanaryRuntimeIdentity,
    ACanaryTopology,
    snapshot_target_identity,
    write_a_canary_manifest,
)

manifest_path, build_root = Path(sys.argv[1]), Path(sys.argv[2])
repo = Path(os.environ["REPO_ROOT"])
launcher = repo / "tools/launcher"
producer = launcher / "common/specdec/qwen4b_a_canary_manifest.py"
runner = launcher / "common/specdec/run_qwen4b_a_canary.sbatch"
supervisor = launcher / "common/eagle3/train_eagle_streaming.sh"
recipe = repo / "modelopt_recipes/general/speculative_decoding/dflash.yaml"
evaluator = launcher / "common/specdec/run_qwen4b_a_canary_eval.sh"
exporter = repo / "examples/speculative_decoding/scripts/export_hf_checkpoint.py"
train_script = repo / "examples/speculative_decoding/launch_train.sh"
receipt = build_root / "BUILD_RECEIPT.json"
output = build_root / "canary.jsonl"
sha = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
target = Path(os.environ["TARGET_PATH"]).resolve(strict=True)
container = Path(os.environ["CONTAINER_IMAGE"]).resolve(strict=True)
target_identity = snapshot_target_identity(
    target, container_path=container, target_revision=os.environ["TARGET_REVISION"]
)
builder_receipt = json.loads(receipt.read_bytes())
if (
    builder_receipt.get("tokenizer_sha256") != target_identity["tokenizer_sha256"]
    or builder_receipt.get("chat_template_sha256") != target_identity["chat_template_sha256"]
):
    raise ValueError("Task9 tokenization does not match the exact target snapshot")
netrc = Path(os.environ["WANDB_NETRC_HOST_PATH"]).resolve(strict=True)
runtime = ACanaryRuntimeIdentity(
    source_commit=os.environ["SOURCE_COMMIT"],
    target_revision=os.environ["TARGET_REVISION"],
    target_path=str(target), target_snapshot_sha256=target_identity["target_snapshot_sha256"],
    target_config_sha256=target_identity["target_config_sha256"],
    tokenizer_sha256=target_identity["tokenizer_sha256"],
    chat_template_sha256=target_identity["chat_template_sha256"],
    container_path=str(container), container_sha256=target_identity["container_sha256"],
    wandb_netrc_sha256=sha(netrc),
    wandb_dir=os.environ["WANDB_DIR"], wandb_cache_dir=os.environ["WANDB_CACHE_DIR"],
    wandb_config_dir=os.environ["WANDB_CONFIG_DIR"],
    wandb_artifact_dir=os.environ["WANDB_ARTIFACT_DIR"],
    producer_module_path=str(producer.relative_to(repo)), producer_module_sha256=sha(producer),
    runner_path=str(runner.relative_to(repo)), runner_sha256=sha(runner),
    supervisor_path=str(supervisor.relative_to(repo)), supervisor_sha256=sha(supervisor),
    recipe_path=str(recipe.relative_to(repo)), recipe_sha256=sha(recipe),
    evaluator_path=str(evaluator.relative_to(repo)), evaluator_sha256=sha(evaluator),
    exporter_path=str(exporter.relative_to(repo)), exporter_sha256=sha(exporter),
    train_script_path=str(train_script.relative_to(repo)), train_script_sha256=sha(train_script),
)
write_a_canary_manifest(
    manifest_path,
    ACanaryManifest(
        runtime=runtime,
        topology=ACanaryTopology(),
        task9_a_selection_path=os.environ["TASK9_A_SELECTION"],
        task9_a_selection_sha256=os.environ["TASK9_A_SELECTION_SHA256"],
        task8_publication_path=str(Path(os.environ["TASK8_PUBLICATION"]) / "PUBLICATION.json"),
        task8_publication_sha256=os.environ["TASK8_PUBLICATION_SHA256"],
        builder_receipt_path=str(receipt), builder_receipt_sha256=sha(receipt),
        builder_output_path=str(output), builder_output_sha256=sha(output),
    ),
)
PY
