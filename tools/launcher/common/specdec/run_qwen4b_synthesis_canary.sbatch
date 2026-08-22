#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --segment=1
#SBATCH --time=03:55:00

set -euo pipefail

for name in THINKING_MODE SOURCE_PATH SOURCE_SHA IMAGE_PATH IMAGE_SHA256 RUNTIME_ARCHIVE RUNTIME_ARCHIVE_SHA256 TARGET_PATH TARGET_REVISION TOKENIZER_SHA256 PROMPT_ROOT SYNTHESIS_ROOT READINESS_RECEIPT READINESS_RECEIPT_SHA256 SCRATCH_ROOT; do
    [[ -n "${!name:-}" ]] || { echo "missing required environment: $name" >&2; exit 2; }
done
[[ "$THINKING_MODE" == on || "$THINKING_MODE" == off ]] || {
    echo "thinking mode must be on or off" >&2
    exit 2
}
[[ "${SLURM_NNODES:-}" == 1 && "$SYNTHESIS_ROOT" == /lustre/* ]] || {
    echo "invalid synthesis-canary topology or output path" >&2
    exit 2
}

LAUNCHER_ROOT="$SOURCE_PATH/tools/launcher"
PROMPT_MANIFEST="$PROMPT_ROOT/target-synth/MANIFEST.json"
TRACE_MANIFEST="$PROMPT_ROOT/trace-replay/MANIFEST.json"
MANIFEST_ROOT="$SYNTHESIS_ROOT/manifests/$THINKING_MODE"
OUTPUT_ROOT="$SYNTHESIS_ROOT/outputs/$THINKING_MODE"
mkdir -p "$MANIFEST_ROOT" "$(dirname "$OUTPUT_ROOT")"
MANIFEST_PATH="$MANIFEST_ROOT/synthesis.json"
CLUSTER_PROFILE_PATH="$MANIFEST_ROOT/oci-hsg.yaml"

mapfile -t identities < <(PYTHONPATH="$LAUNCHER_ROOT${PYTHONPATH:+:$PYTHONPATH}" python3 - \
    "$SOURCE_PATH" "$SOURCE_SHA" "$IMAGE_PATH" "$IMAGE_SHA256" \
    "$RUNTIME_ARCHIVE" "$RUNTIME_ARCHIVE_SHA256" "$TARGET_PATH" "$TARGET_REVISION" \
    "$TOKENIZER_SHA256" "$PROMPT_MANIFEST" "$TRACE_MANIFEST" "$OUTPUT_ROOT" \
    "$THINKING_MODE" "$MANIFEST_PATH" "$CLUSTER_PROFILE_PATH" <<'PY'
import hashlib
import os
import sys
from pathlib import Path

import yaml
from common.specdec.build_qwen4b_synthesis_manifest import (
    Qwen4BSynthesisManifest,
    SynthesisInputs,
    SynthesisSettings,
    load_synthesis_manifest,
    verify_data_manifest,
    write_synthesis_manifest,
)

(
    source_path,
    source_sha,
    image_path,
    image_sha256,
    runtime_archive,
    runtime_archive_sha256,
    target_path,
    target_revision,
    tokenizer_sha256,
    prompt_manifest_raw,
    trace_manifest_raw,
    output_root,
    thinking_mode,
    manifest_path_raw,
    profile_path_raw,
) = sys.argv[1:]
prompt_manifest = Path(prompt_manifest_raw)
trace_manifest = Path(trace_manifest_raw)
prompt_sha256 = hashlib.sha256(prompt_manifest.read_bytes()).hexdigest()
trace_sha256 = hashlib.sha256(trace_manifest.read_bytes()).hexdigest()
verify_data_manifest(prompt_manifest, prompt_sha256)
verify_data_manifest(trace_manifest, trace_sha256)
manifest = Qwen4BSynthesisManifest(
    arm="C",
    thinking_mode=thinking_mode,
    response_source="target-synth",
    inputs=SynthesisInputs(
        source_path=source_path,
        source_sha=source_sha,
        image_path=image_path,
        image_sha256=image_sha256,
        runtime_archive_path=runtime_archive,
        runtime_archive_sha256=runtime_archive_sha256,
        target_path=target_path,
        target_revision=target_revision,
        tokenizer_sha256=tokenizer_sha256,
        prompt_manifest_path=str(prompt_manifest),
        prompt_manifest_sha256=prompt_sha256,
        trace_manifest_path=str(trace_manifest),
        trace_manifest_sha256=trace_sha256,
        output_root=output_root,
    ),
    settings=SynthesisSettings(
        account="nemotron_sw_post",
        partition="batch",
        replicas=4,
        tensor_parallel_size=1,
        num_shards=8,
        output_token_budget=1024,
        temperature=0.0,
        max_tokens=2048,
        max_total_length=32768,
    ),
)
manifest_path = Path(manifest_path_raw)
if manifest_path.exists():
    if load_synthesis_manifest(manifest_path) != [manifest]:
        raise SystemExit("existing synthesis manifest identity mismatch")
else:
    write_synthesis_manifest(manifest_path, [manifest])
profile = {
    "name": "oci-hsg",
    "modelopt_commit": source_sha,
    "ssh_host": "oci-hsg-cs-001-vscode-02",
    "account": "nemotron_sw_post",
    "partition": "batch",
    "fallback_partition": None,
    "durable_root": str(Path(output_root).parents[3]),
    "scratch_candidates": ["/raid/scratch"],
    "training_nodes": 2,
    "training_segment": 2,
    "evaluation_nodes": 1,
    "evaluation_segment": 1,
    "gpus_per_node": 4,
    "explicit_gpu_flag": True,
    "walltime": "03:55:00",
}
profile_path = Path(profile_path_raw)
profile_bytes = yaml.safe_dump(profile, sort_keys=False).encode()
if profile_path.exists():
    if profile_path.read_bytes() != profile_bytes:
        raise SystemExit("existing cluster profile identity mismatch")
else:
    temporary = profile_path.with_name(f".{profile_path.name}.tmp-{os.getpid()}")
    temporary.write_bytes(profile_bytes)
    os.replace(temporary, profile_path)
print(hashlib.sha256(profile_bytes).hexdigest())
print(hashlib.sha256(manifest_path.read_bytes()).hexdigest())
print(manifest.experiment_id)
PY
)
(( ${#identities[@]} == 3 )) || { echo "failed to build synthesis identity" >&2; exit 2; }
CLUSTER_PROFILE_SHA256="${identities[0]}"
MANIFEST_SHA256="${identities[1]}"
EXPERIMENT_ID="${identities[2]}"
readonly LAUNCHER_ROOT PROMPT_MANIFEST TRACE_MANIFEST MANIFEST_ROOT OUTPUT_ROOT MANIFEST_PATH
readonly CLUSTER_PROFILE_PATH CLUSTER_PROFILE_SHA256 MANIFEST_SHA256 EXPERIMENT_ID
export MANIFEST_PATH SYNTHESIS_INDEX=0 LAUNCHER_ROOT CLUSTER_PROFILE_PATH CLUSTER_PROFILE_SHA256
export READINESS_RECEIPT READINESS_RECEIPT_SHA256 SCRATCH_ROOT

echo "synthesis canary identity: $EXPERIMENT_ID manifest_sha256=$MANIFEST_SHA256"
exec bash "$SOURCE_PATH/tools/launcher/common/specdec/run_qwen4b_synthesis.sbatch"
