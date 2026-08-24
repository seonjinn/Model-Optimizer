#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Build a deterministic, untrained DFlash2 checkpoint-0 and prove the exact
# staged runtime can reload it, become healthy, and complete one TP2 request.
set -euo pipefail

[[ $# == 11 ]] || {
    echo "usage: $0 RUNTIME_ARCHIVE RUNTIME_ARCHIVE_SHA TARGET TARGET_RECEIPT TARGET_RECEIPT_SHA IMAGE LAUNCHER SOURCE_SHA OUTPUT_ROOT SCRATCH_ROOT PORT" >&2
    exit 2
}
RUNTIME_ARCHIVE="$(realpath "$1")"
RUNTIME_ARCHIVE_SHA="$2"
TARGET="$(realpath "$3")"
TARGET_RECEIPT="$(realpath "$4")"
TARGET_RECEIPT_SHA="$5"
IMAGE="$(realpath "$6")"
LAUNCHER="$(realpath "$7")"
SOURCE_SHA="$8"
OUTPUT_ROOT="$(realpath -m "$9")"
SCRATCH_ROOT="$(realpath -m "${10}")"
PORT="${11}"
[[ "$RUNTIME_ARCHIVE_SHA" =~ ^[0-9a-f]{64}$ && "$TARGET_RECEIPT_SHA" =~ ^[0-9a-f]{64}$ ]]
[[ "$SOURCE_SHA" =~ ^[0-9a-f]{40}$ && "$PORT" =~ ^[0-9]+$ ]]
[[ ! -e "$OUTPUT_ROOT" ]] || {
    echo "refusing to replace existing serve-gate output" >&2
    exit 2
}
[[ "$(sha256sum "$RUNTIME_ARCHIVE" | cut -d' ' -f1)" == "$RUNTIME_ARCHIVE_SHA" ]]
[[ "$(sha256sum "$TARGET_RECEIPT" | cut -d' ' -f1)" == "$TARGET_RECEIPT_SHA" ]]

work_root="${SCRATCH_ROOT%/}/${SLURM_JOB_ID:?SLURM_JOB_ID is required}/dflash2-zero-init-gate"
runtime="$work_root/runtime"
attempt="$work_root/serve-gate"
mkdir -p "$runtime" "$attempt" "$(dirname "$OUTPUT_ROOT")"
tar --extract --file="$RUNTIME_ARCHIVE" --directory="$runtime"
runtime_package="$(find "$runtime/lib" -type d -path '*/site-packages/vllm' -print -quit)"
[[ -n "$runtime_package" ]]
runtime_receipt="$runtime/dflash2-vllm-runtime-receipt.json"
runtime_receipt_sha="$(sha256sum "$runtime_receipt" | cut -d' ' -f1)"

PYTHONPATH="$LAUNCHER:$LAUNCHER/tools/launcher" python3 - \
    "$LAUNCHER" "$SOURCE_SHA" "$TARGET" "$TARGET_RECEIPT" "$TARGET_RECEIPT_SHA" \
    "$runtime_package" "$runtime_receipt" "$runtime_receipt_sha" <<'PY'
import json
import sys
from pathlib import Path

from tools.launcher.common.specdec.dflash2_runtime_contract import (
    validate_artifact_receipt,
    validate_dflash2_source_checkout,
    verify_vllm_runtime,
)

launcher, source_sha, target, target_receipt, target_receipt_sha, package, receipt, receipt_sha = sys.argv[1:]
validate_dflash2_source_checkout(Path(launcher), source_sha, "6eda6bbf54455086a54660fe7a7b06c415b87da5")
target_body = json.loads(Path(target_receipt).read_text())
validate_artifact_receipt(
    Path(target_receipt),
    expected_receipt_sha256=target_receipt_sha,
    artifact_path=Path(target),
    expected_artifact_sha256=target_body["artifact_sha256"],
    kind="target",
)
verify_vllm_runtime(
    Path(package),
    Path(receipt),
    receipt_sha,
    "b389ac29465b33f9e9c534df221ea3c129e9793f",
    "b389ac29465b33f9e9c534df221ea3c129e9793f",
    expected_flashmla_commit="a8f794d1251cbfd88a5011445dd5582289c727e4",
)
PY

mounts="${work_root}:${work_root},${TARGET}:${TARGET}:ro,${TARGET_RECEIPT}:${TARGET_RECEIPT}:ro,${RUNTIME_ARCHIVE}:${RUNTIME_ARCHIVE}:ro,${LAUNCHER}:${LAUNCHER}:ro,$(dirname "$OUTPUT_ROOT"):$(dirname "$OUTPUT_ROOT")"
srun --nodes=1 --ntasks=1 --ntasks-per-node=1 --kill-on-bad-exit=1 \
    --no-container-mount-home --container-image="$IMAGE" --container-mounts="$mounts" \
    bash -s -- "$runtime" "$runtime_package" "$TARGET" "$TARGET_RECEIPT" \
    "$TARGET_RECEIPT_SHA" "$RUNTIME_ARCHIVE" "$RUNTIME_ARCHIVE_SHA" \
    "$runtime_receipt" "$runtime_receipt_sha" "$LAUNCHER" "$SOURCE_SHA" \
    "$attempt" "$OUTPUT_ROOT" "$PORT" "$SLURM_JOB_ID" <<'INNER'
set -euo pipefail
runtime="$1"
runtime_package="$2"
target="$3"
target_receipt="$4"
target_receipt_sha="$5"
runtime_archive="$6"
runtime_archive_sha="$7"
runtime_receipt="$8"
runtime_receipt_sha="$9"
launcher="${10}"
source_sha="${11}"
attempt="${12}"
output_root="${13}"
port="${14}"
slurm_job_id="${15}"
started_at="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
export_dir="$attempt/checkpoint-0-runtime-only"
server_log="$attempt/vllm-serve.log"
request_payload="$attempt/request.json"
response_path="$attempt/response.json"
export PATH="$runtime/bin:$PATH"
export PYTHONPATH="$launcher:$launcher/tools/launcher"
export CUDA_VISIBLE_DEVICES=0,1
export VLLM_USE_V2_MODEL_RUNNER=1

"$runtime/bin/python" - <<'PY'
import vllm._flashmla_C
import vllm._flashmla_extension_C
from vllm.v1.attention.ops.flashmla import _is_flashmla_available

assert _is_flashmla_available() == (True, None)
PY

"$runtime/bin/python" - "$target" "$export_dir" <<'PY'
import json
import sys
from pathlib import Path

import torch
from transformers import AutoConfig, AutoModelForCausalLM

import modelopt.torch.speculative as mtsp

target, export_dir = sys.argv[1:]
torch.manual_seed(42)
target_config = AutoConfig.from_pretrained(target, trust_remote_code=False)
assert target_config.num_attention_heads == 32
assert target_config.num_key_value_heads == 4
assert target_config.head_dim == 128
assert target_config.intermediate_size == 6144
with torch.device("meta"):
    model = AutoModelForCausalLM.from_config(target_config, trust_remote_code=False)
mtsp.convert(
    model,
    [
        (
            "dflash",
            {
                "dflash_block_size": 8,
                "dflash_num_anchors": 256,
                "dflash_mask_token_id": 151669,
                "dflash_selector_loss_alpha": 1.0,
                "dflash_architecture_config": {
                    "num_hidden_layers": 5,
                    "num_attention_heads": 32,
                    "num_key_value_heads": 4,
                    "head_dim": 128,
                    "intermediate_size": 6144,
                    "projector_type": "dflash2",
                    "conv_kernel_size": 2,
                    "conv_group_size": 16,
                    "selector_rank": 256,
                    "selector_top_k": 16,
                },
            },
        )
    ],
)
draft = model.dflash_module.to_empty(device="cpu")
with torch.no_grad():
    for parameter in draft.parameters():
        parameter.zero_()
    for module in draft.modules():
        if hasattr(module, "base_kernel"):
            base_kernel = module.base_kernel
            base_kernel.zero_()
            base_kernel[:, 0].fill_(1.0)
model.get_exporter().export(Path(export_dir), dtype=torch.bfloat16)
config = json.loads((Path(export_dir) / "config.json").read_text())
assert config["architectures"] == ["DFlash2DraftModel"]
assert config["block_size"] == 8
assert config["num_attention_heads"] == 32
assert config["num_key_value_heads"] == 4
assert config["head_dim"] == 128
assert config["intermediate_size"] == 6144
assert config["dflash_config"]["projector_type"] == "dflash2"
PY

"$runtime/bin/python" - "$target" "$export_dir" >"$request_payload" <<'PY'
import json
import sys
print(json.dumps({"model": sys.argv[1], "prompt": "Write one short sentence about CUDA.", "max_tokens": 16, "temperature": 0}))
PY
SPEC_CONFIG="$($runtime/bin/python - "$export_dir" <<'PY'
import json
import sys
print(json.dumps({"method": "dflash", "model": sys.argv[1], "num_speculative_tokens": 7}))
PY
)"
"$runtime/bin/python" -m vllm.entrypoints.cli.main serve "$target" \
    --tensor-parallel-size 2 --max-model-len 4096 --max-num-batched-tokens 4096 \
    --gpu-memory-utilization 0.85 --port "$port" --speculative-config "$SPEC_CONFIG" \
    >"$server_log" 2>&1 &
server_pid=$!
cleanup() {
    kill "$server_pid" 2>/dev/null || true
    wait "$server_pid" 2>/dev/null || true
}
trap cleanup EXIT
ready=0
for _ in $(seq 1 900); do
    if ! kill -0 "$server_pid" 2>/dev/null; then
        wait "$server_pid"
        exit 3
    fi
    if curl -fsS "http://127.0.0.1:${port}/health" >/dev/null; then
        ready=1
        break
    fi
    sleep 1
done
[[ "$ready" == 1 ]]
curl -fsS "http://127.0.0.1:${port}/v1/completions" \
    -H 'Content-Type: application/json' --data-binary "@$request_payload" >"$response_path"
"$runtime/bin/python" - "$response_path" <<'PY'
import json
import sys
response = json.load(open(sys.argv[1]))
assert response.get("choices")
assert response.get("usage", {}).get("completion_tokens", 0) > 0
PY
grep -q 'DFlash2' "$server_log"
cleanup
trap - EXIT
finished_at="$(date -u +%Y-%m-%dT%H:%M:%SZ)"

"$runtime/bin/python" - "$attempt" "$target" "$target_receipt" "$target_receipt_sha" \
    "$runtime_archive" "$runtime_archive_sha" "$runtime_receipt" "$runtime_receipt_sha" \
    "$launcher" "$source_sha" "$runtime_package" "$slurm_job_id" "$started_at" \
    "$finished_at" "$port" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

(
    attempt_s,
    target_s,
    target_receipt_s,
    target_receipt_sha,
    runtime_archive_s,
    runtime_archive_sha,
    runtime_receipt_s,
    runtime_receipt_sha,
    launcher_s,
    source_sha,
    runtime_package_s,
    slurm_job_id,
    started_at,
    finished_at,
    port,
) = sys.argv[1:]
attempt = Path(attempt_s)
launcher = Path(launcher_s)
runtime_package = Path(runtime_package_s)

def descriptor(path: Path, name: str) -> dict[str, object]:
    raw = path.read_bytes()
    return {"path": name, "bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}

target_receipt = json.loads(Path(target_receipt_s).read_text())
config = json.loads((attempt / "checkpoint-0-runtime-only/config.json").read_text())
response = json.loads((attempt / "response.json").read_text())
body = {
    "schema_version": 1,
    "producer": "dflash2-zero-init-serve-gate-v1",
    "scientific_training_authorized": False,
    "training_quality_claim": False,
    "purpose": "checkpoint-0-runtime-only",
    "slurm_job_id": slurm_job_id,
    "started_at": started_at,
    "finished_at": finished_at,
    "source_commit": source_sha,
    "target_path": target_s,
    "target_artifact_sha256": target_receipt["artifact_sha256"],
    "target_receipt": descriptor(Path(target_receipt_s), Path(target_receipt_s).name),
    "target_receipt_sha256": target_receipt_sha,
    "runtime_archive": descriptor(Path(runtime_archive_s), Path(runtime_archive_s).name),
    "runtime_archive_sha256": runtime_archive_sha,
    "runtime_receipt": descriptor(Path(runtime_receipt_s), "dflash2-vllm-runtime-receipt.json"),
    "runtime_receipt_sha256": runtime_receipt_sha,
    "architecture": {
        "method": "dflash",
        "draft_architecture": "DFlash2DraftModel",
        "dflash_block_size": 8,
        "num_speculative_tokens": 7,
        "num_hidden_layers": 5,
        "num_attention_heads": 32,
        "num_key_value_heads": 4,
        "head_dim": 128,
        "intermediate_size": 6144,
        "conv_kernel_size": 2,
        "conv_group_size": 16,
        "selector_rank": 256,
        "selector_top_k": 16,
    },
    "initialization": "all-zero-except-identity-convolution-base-kernel",
    "serve_config": {
        "tensor_parallel_size": 2,
        "port": int(port),
        "max_model_len": 4096,
        "health_passed": True,
        "completed_requests": 1,
    },
    "exported_config": config,
    "exported_config_file": descriptor(attempt / "checkpoint-0-runtime-only/config.json", "checkpoint-0-runtime-only/config.json"),
    "exported_weights": descriptor(attempt / "checkpoint-0-runtime-only/model.safetensors", "checkpoint-0-runtime-only/model.safetensors"),
    "exporter_sha256": descriptor(launcher / "modelopt/torch/export/plugins/hf_spec_export.py", "modelopt/torch/export/plugins/hf_spec_export.py")["sha256"],
    "reload_loader_sha256": descriptor(runtime_package / "model_executor/models/qwen3_dflash2.py", "vllm/model_executor/models/qwen3_dflash2.py")["sha256"],
    "speculator_sha256": descriptor(runtime_package / "v1/worker/gpu/spec_decode/dflash2/speculator.py", "vllm/v1/worker/gpu/spec_decode/dflash2/speculator.py")["sha256"],
    "runner": descriptor(launcher / "tools/launcher/common/specdec/run_dflash2_zero_init_serve_gate.sbatch", "tools/launcher/common/specdec/run_dflash2_zero_init_serve_gate.sbatch"),
    "server_log": descriptor(attempt / "vllm-serve.log", "vllm-serve.log"),
    "request": descriptor(attempt / "request.json", "request.json"),
    "response": descriptor(attempt / "response.json", "response.json"),
    "completion_tokens": response["usage"]["completion_tokens"],
}
canonical = json.dumps(body, sort_keys=True, separators=(",", ":"))
body["receipt_sha256"] = hashlib.sha256(canonical.encode()).hexdigest()
(attempt / "serve-gate-receipt.json").write_text(
    json.dumps(body, sort_keys=True, separators=(",", ":")) + "\n"
)
PY

temporary="${output_root}.partial-${slurm_job_id}"
[[ ! -e "$temporary" && ! -e "$output_root" ]]
cp -a "$attempt" "$temporary"
mv "$temporary" "$output_root"
sha256sum "$output_root/serve-gate-receipt.json"
INNER
