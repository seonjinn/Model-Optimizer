#!/bin/bash
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

set -uo pipefail

readonly SPECULATORS_EXPECTED_SHA="0b08a89a83b92007be63f128e01497455b0209df"
readonly DFLASH2_VLLM_EXPECTED_SHA="b389ac29465b33f9e9c534df221ea3c129e9793f"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LAUNCHER_ROOT="${DRAFTER_LAUNCHER_ROOT:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"
ARTIFACT_HELPER="${SCRIPT_DIR}/speculators_eval_artifacts.py"
SERVER_PID=""
FINAL_STATUS="failed"
MODELOPT_SHA="unknown"
MODELOPT_DIRTY="unknown"
PYTHON_VERSION="unknown"
VLLM_VERSION="unknown"
GUIDELLM_VERSION="unknown"
TP="${TP_SIZE:-1}"
MAX_CONCURRENCY="${MAX_CONCURRENCY:-32}"
MAX_REQUESTS="${MAX_REQUESTS:-200}"
EVAL_MODE="${EVAL_MODE:-throughput}"
DRAFT_MODEL="${DRAFT_MODEL:-}"
DFLASH_BLOCK_SIZE="${DFLASH_BLOCK_SIZE:-0}"
NUM_SPEC_TOKENS="${NUM_SPEC_TOKENS:-0}"
SERVER_ARGS=()
EVALUATOR_ARGS=()
SPECULATORS_CLIENT_RUNTIME="${SPECULATORS_CLIENT_RUNTIME:-${SPECULATORS_RUNTIME:-}}"
VLLM_SERVER_RUNTIME="${VLLM_SERVER_RUNTIME:-${SPECULATORS_CLIENT_RUNTIME}}"
SERVER_PYTHON="${VLLM_SERVER_RUNTIME}/bin/python"
if [[ ! -x "${SERVER_PYTHON}" && -x "${VLLM_SERVER_RUNTIME}/bin/python3" ]]; then
    SERVER_PYTHON="${VLLM_SERVER_RUNTIME}/bin/python3"
fi

require_var() {
    local name="$1"
    if [[ -z "${!name:-}" ]]; then
        echo "ERROR: ${name} is required" >&2
        exit 2
    fi
}

for name in SPECULATORS_CLIENT_RUNTIME VLLM_SERVER_RUNTIME SPECULATORS_REPO HF_MODEL_CKPT SPEC_METHOD \
    HF_HOME EVAL_OUTPUT_ROOT EVAL_CONFIG_PATH CONTAINER_IMAGE \
    CONTAINER_IDENTITY_PATH DATASET_MANIFEST_PATH MODELOPT_REPO; do
    require_var "$name"
done
if [[ "${SPEC_METHOD}" != "baseline" ]]; then
    require_var DRAFT_MODEL
fi

validate_cluster_contract() {
    [[ -n "${CLUSTER_PROFILE:-}" || -n "${CLUSTER_READINESS_RECEIPT:-}" ]] || return 0
    [[ -n "${CLUSTER_PROFILE:-}" && -n "${CLUSTER_READINESS_RECEIPT:-}" ]] || {
        echo "ERROR: CLUSTER_PROFILE and CLUSTER_READINESS_RECEIPT are required together" >&2
        exit 2
    }
    PYTHONPATH="${LAUNCHER_ROOT}${PYTHONPATH:+:${PYTHONPATH}}" python3 - \
        "${CLUSTER_PROFILE}" "${CLUSTER_READINESS_RECEIPT}" \
        "${MARS_SCRATCH_ROOT:-/raid/scratch}" "${EVAL_OUTPUT_ROOT}" <<'PY'
import json
import sys
from pathlib import Path

from common.specdec.cluster_profile import load_cluster_profile, validate_scratch_root

profile = load_cluster_profile(Path(sys.argv[1]).resolve())
receipt = json.loads(Path(sys.argv[2]).resolve().read_text())
expected = {
    "profile": profile.name,
    "account": profile.account,
    "partition": profile.partition,
    "pyxis_available": True,
    "gpu_count": profile.gpus_per_node,
    "architecture": "aarch64",
}
if any(receipt.get(key) != value for key, value in expected.items()):
    raise ValueError("cluster readiness receipt does not match profile")
receipt_scratch = Path(receipt.get("scratch_root", ""))
validate_scratch_root(profile, receipt_scratch)
runtime_scratch = Path(sys.argv[3]).resolve(strict=False)
validate_scratch_root(profile, runtime_scratch)
if not runtime_scratch.is_relative_to(receipt_scratch.resolve(strict=False)):
    raise ValueError("runtime scratch is outside the readiness scratch root")
output_root = Path(sys.argv[4]).resolve(strict=False)
if not output_root.is_relative_to(profile.durable_root.resolve(strict=False)):
    raise ValueError("EVAL_OUTPUT_ROOT must be under the profile durable_root")
PY
}

if [[ ! -f "${SPECULATORS_CLIENT_RUNTIME}/bin/activate" ]]; then
    echo "ERROR: shared client runtime is missing: ${SPECULATORS_CLIENT_RUNTIME}" >&2
    exit 2
fi
[[ -x "${SERVER_PYTHON}" ]] || { echo "ERROR: vLLM server runtime is missing" >&2; exit 2; }
# shellcheck disable=SC1090,SC1091
source "${SPECULATORS_CLIENT_RUNTIME}/bin/activate"
export PATH="${SPECULATORS_CLIENT_RUNTIME}/bin:${PATH}"
export PYTHONPATH="${SPECULATORS_REPO}/src${PYTHONPATH:+:${PYTHONPATH}}"
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${HF_HOME}/datasets}"

if ! command -v python3 >/dev/null 2>&1; then
    echo "ERROR: shared runtime has no python3" >&2
    exit 2
fi
if ! command -v guidellm >/dev/null 2>&1; then
    echo "ERROR: shared Speculators runtime has no guidellm" >&2
    exit 2
fi
if ! validate_cluster_contract; then
    echo "ERROR: invalid cluster profile/readiness contract" >&2
    exit 2
fi

RUN_ID="${EVAL_RUN_ID:-${SLURM_JOB_ID:-manual}-$(date -u +%Y%m%dT%H%M%SZ)-$$-${SPEC_METHOD}-b${DFLASH_BLOCK_SIZE}}"
RUN_DIR="${EVAL_OUTPUT_ROOT%/}/${RUN_ID}"
LOCK_DIR="${RUN_DIR}/.active"
mkdir -p "${RUN_DIR}/subsets" "${RUN_DIR}/.attempts"
if ! mkdir "${LOCK_DIR}"; then
    echo "ERROR: evaluation output is active: ${RUN_DIR}" >&2
    exit 2
fi

write_manifest() {
    local helper_args=(manifest \
        --output "${RUN_DIR}/manifest.json" \
        --status "$1" \
        --method "${SPEC_METHOD}" \
        --block-size "${DFLASH_BLOCK_SIZE}" \
        --num-speculative-tokens "${NUM_SPEC_TOKENS}" \
        --target-model "${HF_MODEL_CKPT}" \
        --draft-model "${DRAFT_MODEL}" \
        --speculators-repo "${SPECULATORS_REPO}" \
        --speculators-sha "${SPECULATORS_EXPECTED_SHA}" \
        --modelopt-repo "${MODELOPT_REPO}" \
        --modelopt-sha "${MODELOPT_SHA}" \
        --modelopt-dirty "${MODELOPT_DIRTY}" \
        --client-runtime "${SPECULATORS_CLIENT_RUNTIME}" \
        --server-runtime "${VLLM_SERVER_RUNTIME}" \
        --artifact-identity "${EVAL_ARTIFACT_IDENTITY_PATH:-}" \
        --artifact-identity-sha256 "${EVAL_ARTIFACT_IDENTITY_SHA256:-}" \
        --container-image "${CONTAINER_IMAGE}" \
        --container-identity "${CONTAINER_IDENTITY_PATH}" \
        --dataset-manifest "${DATASET_MANIFEST_PATH}" \
        --hf-home "${HF_HOME}" \
        --slurm-job-id "${SLURM_JOB_ID:-none}" \
        --launcher-config "${EVAL_CONFIG_PATH}" \
        --python-version "${PYTHON_VERSION}" \
        --vllm-version "${VLLM_VERSION}" \
        --guidellm-version "${GUIDELLM_VERSION}" \
        --max-concurrency "${MAX_CONCURRENCY}" \
        --max-requests "${MAX_REQUESTS}" \
        --evaluation-mode "${EVAL_MODE}" \
        --tensor-parallel-size "${TP}")
    local arg
    if [[ ${#SERVER_ARGS[@]} -gt 0 ]]; then
        for arg in "${SERVER_ARGS[@]}"; do
            helper_args+=("--server-arg=${arg}")
        done
    fi
    if [[ ${#EVALUATOR_ARGS[@]} -gt 0 ]]; then
        for arg in "${EVALUATOR_ARGS[@]}"; do
            helper_args+=("--evaluator-arg=${arg}")
        done
    fi
    "${SPECULATORS_CLIENT_RUNTIME}/bin/python3" "${ARTIFACT_HELPER}" "${helper_args[@]}"
}

cleanup() {
    local rc=$?
    trap - EXIT INT TERM
    if [[ -n "${SERVER_PID}" ]]; then
        kill "${SERVER_PID}" 2>/dev/null || true
        wait "${SERVER_PID}" 2>/dev/null || true
    fi
    rmdir "${LOCK_DIR}" 2>/dev/null || true
    if ! write_manifest "${FINAL_STATUS}"; then
        echo "ERROR: failed to write evaluation manifest" >&2
        [[ ${rc} -ne 0 ]] || rc=1
    fi
    exit "${rc}"
}

handle_signal() {
    local signal_number="$1"
    FINAL_STATUS="failed"
    exit "$((128 + signal_number))"
}

trap cleanup EXIT
trap 'handle_signal 2' INT
trap 'handle_signal 15' TERM

required_paths=("${SPECULATORS_REPO}/scripts/evaluate/evaluate.py"
    "${HF_MODEL_CKPT}/config.json" "${EVAL_CONFIG_PATH}")
if [[ "${SPEC_METHOD}" != "baseline" ]]; then
    required_paths+=("${DRAFT_MODEL}/config.json")
fi
for path in "${required_paths[@]}"; do
    if [[ ! -f "${path}" ]]; then
        echo "ERROR: required staged file is missing: ${path}" >&2
        exit 2
    fi
done

case "${SPEC_METHOD}:${DFLASH_BLOCK_SIZE}:${NUM_SPEC_TOKENS}" in
    baseline:0:0|dflash:8:7|dflash:16:15|dflash2:8:7|dspark:8:8|dspark:16:16) ;;
    *)
        echo "ERROR: invalid method/B/K mapping: ${SPEC_METHOD}/${DFLASH_BLOCK_SIZE}/${NUM_SPEC_TOKENS}" >&2
        exit 2
        ;;
esac
case "${MAX_CONCURRENCY}" in
    1|8|32|128) ;;
    *) echo "ERROR: invalid MAX_CONCURRENCY: ${MAX_CONCURRENCY}; expected 1, 8, 32, or 128" >&2; exit 2 ;;
esac
if [[ ! "${MAX_REQUESTS}" =~ ^[1-9][0-9]*$ ]]; then
    echo "ERROR: MAX_REQUESTS must be a positive integer: ${MAX_REQUESTS}" >&2
    exit 2
fi
if [[ "${MAX_CONCURRENCY}" == "128" && ${MAX_REQUESTS} -lt 512 ]]; then
    echo "ERROR: MAX_REQUESTS must be at least 512 for MAX_CONCURRENCY=128" >&2
    exit 2
fi
case "${EVAL_MODE}" in
    throughput) ;;
    *) echo "ERROR: unsupported EVAL_MODE: ${EVAL_MODE}; expected throughput" >&2; exit 2 ;;
esac

SPECULATORS_ACTUAL_SHA="$(git -C "${SPECULATORS_REPO}" rev-parse HEAD 2>/dev/null || true)"
if [[ "${SPECULATORS_ACTUAL_SHA}" != "${SPECULATORS_EXPECTED_SHA}" ]]; then
    echo "ERROR: Speculators SHA mismatch: expected ${SPECULATORS_EXPECTED_SHA}, got ${SPECULATORS_ACTUAL_SHA}" >&2
    exit 2
fi
if [[ -n "$(git -C "${SPECULATORS_REPO}" status --porcelain 2>/dev/null)" ]]; then
    echo "ERROR: shared Speculators checkout is dirty: ${SPECULATORS_REPO}" >&2
    exit 2
fi

MODELOPT_SHA="$(git -C "${MODELOPT_REPO}" rev-parse HEAD 2>/dev/null || true)"
if [[ ! "${MODELOPT_SHA}" =~ ^[0-9a-f]{40}$ ]]; then
    echo "ERROR: ModelOpt checkout does not resolve to a full commit: ${MODELOPT_REPO}" >&2
    exit 2
fi
if [[ -n "$(git -C "${MODELOPT_REPO}" status --porcelain 2>/dev/null)" ]]; then
    echo "ERROR: ModelOpt checkout is dirty: ${MODELOPT_REPO}" >&2
    exit 2
fi
MODELOPT_DIRTY="false"
if [[ "${SPEC_METHOD}" == dflash2 && -z "${EVAL_ARTIFACT_IDENTITY_PATH:-}" ]]; then
    echo "ERROR: DFlash2 evaluation requires an authenticated artifact identity" >&2
    exit 2
fi
if [[ -n "${EVAL_ARTIFACT_IDENTITY_PATH:-}" ]]; then
    [[ "${EVAL_ARTIFACT_IDENTITY_SHA256:-}" =~ ^[0-9a-f]{64}$ \
        && "$(sha256sum "${EVAL_ARTIFACT_IDENTITY_PATH}" | cut -d' ' -f1)" == "${EVAL_ARTIFACT_IDENTITY_SHA256}" ]] || {
        echo "ERROR: evaluation artifact identity mismatch" >&2
        exit 2
    }
fi
if ! "${SPECULATORS_CLIENT_RUNTIME}/bin/python3" "${ARTIFACT_HELPER}" verify-inputs \
    --dataset-manifest "${DATASET_MANIFEST_PATH}" \
    --hf-home "${HF_HOME}" \
    --container-identity "${CONTAINER_IDENTITY_PATH}" \
    --container-image "${CONTAINER_IMAGE}" \
    --launcher-config "${EVAL_CONFIG_PATH}"; then
    echo "ERROR: staged dataset or container identity verification failed" >&2
    exit 2
fi

if [[ "${SPEC_METHOD}" == dflash2 || -n "${VLLM_SERVER_RUNTIME_RECEIPT_SHA256:-}" ]]; then
    [[ "${VLLM_SERVER_RUNTIME_RECEIPT_SHA256:-}" =~ ^[0-9a-f]{64}$ ]] || {
        echo "ERROR: exact DFlash2 server runtime receipt SHA-256 is required" >&2
        exit 2
    }
    PYTHONPATH="${LAUNCHER_ROOT}${PYTHONPATH:+:${PYTHONPATH}}" \
        "${SERVER_PYTHON}" - "${VLLM_SERVER_RUNTIME}/dflash2-vllm-runtime-receipt.json" \
        "${VLLM_SERVER_RUNTIME_RECEIPT_SHA256}" "${DFLASH2_VLLM_EXPECTED_SHA}" <<'PY'
import sys
from pathlib import Path

import vllm
from common.specdec.dflash2_runtime_contract import verify_vllm_runtime

verify_vllm_runtime(
    Path(vllm.__file__).resolve().parent,
    Path(sys.argv[1]),
    sys.argv[2],
    sys.argv[3],
    sys.argv[3],
)
PY
fi

if ! "${SPECULATORS_CLIENT_RUNTIME}/bin/python3" - "${RUN_DIR}/input-fingerprint.json" \
    "${HF_MODEL_CKPT}/config.json" \
    "$([[ "${SPEC_METHOD}" == "baseline" ]] || printf '%s' "${DRAFT_MODEL}/config.json")" \
    "${DATASET_MANIFEST_PATH}" "${CONTAINER_IDENTITY_PATH}" \
    "${SPECULATORS_CLIENT_RUNTIME}/.archive.sha256" \
    "${VLLM_SERVER_RUNTIME}/.archive.sha256" "${EVAL_ARTIFACT_IDENTITY_PATH:-}" \
    "${EVAL_CONFIG_PATH}" \
    "${MODELOPT_SHA}" "${SPECULATORS_ACTUAL_SHA}" \
    "${SPEC_METHOD}" "${DFLASH_BLOCK_SIZE}" "${NUM_SPEC_TOKENS}" \
    "${MAX_CONCURRENCY}" "${MAX_REQUESTS}" "${EVAL_MODE}" "${TP}" <<'PY'
import hashlib
import json
import os
import re
import sys
from pathlib import Path


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


(
    fingerprint_path_raw,
    target_config_raw,
    draft_config_raw,
    dataset_manifest_raw,
    image_identity_raw,
    client_runtime_marker_raw,
    server_runtime_marker_raw,
    artifact_identity_raw,
    launcher_config_raw,
    modelopt_sha,
    speculators_sha,
    method,
    block_size,
    num_spec_tokens,
    max_concurrency,
    max_requests,
    eval_mode,
    tensor_parallel_size,
) = sys.argv[1:]

fingerprint_path = Path(fingerprint_path_raw)
target_config = Path(target_config_raw)
draft_config = Path(draft_config_raw) if draft_config_raw else None
dataset_manifest_path = Path(dataset_manifest_raw)
image_identity_path = Path(image_identity_raw)
client_runtime_marker_path = Path(client_runtime_marker_raw)
server_runtime_marker_path = Path(server_runtime_marker_raw)
artifact_identity_path = Path(artifact_identity_raw) if artifact_identity_raw else None
launcher_config = Path(launcher_config_raw)
dataset_manifest = json.loads(dataset_manifest_path.read_text())
image_identity = json.loads(image_identity_path.read_text())
client_runtime_sha256 = client_runtime_marker_path.read_text().strip()
server_runtime_sha256 = server_runtime_marker_path.read_text().strip()
if not re.fullmatch(r"[0-9a-f]{64}", client_runtime_sha256):
    raise ValueError(f"invalid client runtime archive SHA marker: {client_runtime_marker_path}")
if not re.fullmatch(r"[0-9a-f]{64}", server_runtime_sha256):
    raise ValueError(f"invalid server runtime archive SHA marker: {server_runtime_marker_path}")
expected_runtime_sha256 = os.environ.get("SPECULATORS_RUNTIME_ARCHIVE_SHA256")
if expected_runtime_sha256 and client_runtime_sha256 != expected_runtime_sha256:
    raise ValueError("client runtime archive SHA marker does not match requested runtime")
expected_server_sha256 = os.environ.get("VLLM_SERVER_RUNTIME_ARCHIVE_SHA256")
if expected_server_sha256 and server_runtime_sha256 != expected_server_sha256:
    raise ValueError("server runtime archive SHA marker does not match requested runtime")

inputs = {
    "target_config_sha256": file_sha256(target_config),
    "draft_config_sha256": file_sha256(draft_config) if draft_config else None,
    "dataset": {
        "revision": dataset_manifest.get("revision"),
        "manifest_sha256": file_sha256(dataset_manifest_path),
    },
    "image": {
        "sha256": image_identity.get("sha256"),
        "identity_sha256": file_sha256(image_identity_path),
    },
    "runtimes": {
        "client": {"sha256": client_runtime_sha256},
        "server": {
            "sha256": server_runtime_sha256,
            "receipt_sha256": os.environ.get("VLLM_SERVER_RUNTIME_RECEIPT_SHA256"),
        },
    },
    "artifact_identity_sha256": (
        file_sha256(artifact_identity_path) if artifact_identity_path else None
    ),
    "source": {
        "modelopt_sha": modelopt_sha,
        "speculators_sha": speculators_sha,
    },
    "launcher_config_sha256": file_sha256(launcher_config),
    "evaluation": {
        "method": method,
        "block_size": int(block_size),
        "num_speculative_tokens": int(num_spec_tokens),
        "max_concurrency": int(max_concurrency),
        "max_requests": int(max_requests),
        "mode": eval_mode,
        "tensor_parallel_size": int(tensor_parallel_size),
    },
}
canonical_inputs = json.dumps(inputs, sort_keys=True, separators=(",", ":"))
fingerprint = {
    "schema_version": 1,
    "sha256": hashlib.sha256(canonical_inputs.encode()).hexdigest(),
    "inputs": inputs,
}
if fingerprint_path.exists():
    if json.loads(fingerprint_path.read_text()) != fingerprint:
        raise ValueError(f"immutable input fingerprint mismatch: {fingerprint_path}")
else:
    temporary = fingerprint_path.with_name(f".{fingerprint_path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(fingerprint, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, fingerprint_path)
PY
then
    echo "ERROR: immutable evaluation input fingerprint validation failed" >&2
    exit 2
fi

PYTHON_VERSION="$("${SPECULATORS_CLIENT_RUNTIME}/bin/python3" --version 2>&1)"
VLLM_VERSION="$("${SERVER_PYTHON}" -c 'from importlib.metadata import version; print(version("vllm"))')"
GUIDELLM_VERSION="$("${SPECULATORS_CLIENT_RUNTIME}/bin/python3" -c 'from importlib.metadata import version; print(version("guidellm"))')"

PORT="${VLLM_PORT:-8000}"
if curl -fsS --max-time 2 "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1; then
    echo "ERROR: port ${PORT} is already serving health before vLLM launch" >&2
    exit 5
fi
SERVER_ARGS=(-m vllm.entrypoints.cli.main serve "${HF_MODEL_CKPT}"
    --tensor-parallel-size "${TP}"
    --port "${PORT}")
if [[ "${SPEC_METHOD}" != "baseline" ]]; then
    SPEC_CONFIG="$(printf '{\"method\":\"%s\",\"model\":\"%s\",\"num_speculative_tokens\":%s}' \
        "$([[ "${SPEC_METHOD}" == dflash2 ]] && printf dflash || printf '%s' "${SPEC_METHOD}")" \
        "${DRAFT_MODEL}" "${NUM_SPEC_TOKENS}")"
    SERVER_ARGS+=(--speculative-config "${SPEC_CONFIG}")
fi
if [[ -n "${VLLM_SERVER_RUNTIME_RECEIPT_SHA256:-}" ]]; then
    export VLLM_USE_V2_MODEL_RUNNER=1
fi
"${SERVER_PYTHON}" "${SERVER_ARGS[@]}" \
    >"${RUN_DIR}/vllm.log" 2>&1 &
SERVER_PID=$!

READY=0
for ((attempt = 1; attempt <= ${SERVE_READY_TIMEOUT:-1800}; attempt++)); do
    if ! kill -0 "${SERVER_PID}" 2>/dev/null; then
        wait "${SERVER_PID}" 2>/dev/null
        echo "ERROR: owned vLLM server exited before readiness" >&2
        exit 5
    fi
    if curl -fsS "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1 \
        && curl -fsS "http://127.0.0.1:${PORT}/v1/models" \
            | "${SPECULATORS_CLIENT_RUNTIME}/bin/python3" -c 'import json,sys; data=json.load(sys.stdin).get("data", []); sys.exit(0 if any(item.get("id") == sys.argv[1] for item in data) else 1)' "${HF_MODEL_CKPT}" \
            >/dev/null 2>&1 \
        && curl -fsS "http://127.0.0.1:${PORT}/metrics" 2>/dev/null \
            | grep -q "vllm:$([[ "${SPEC_METHOD}" == "baseline" ]] && printf num_requests || printf spec_decode)" \
        && kill -0 "${SERVER_PID}" 2>/dev/null; then
        READY=1
        break
    fi
    sleep 1
done
if [[ ${READY} -ne 1 ]]; then
    echo "ERROR: vLLM health/models/metrics readiness timed out" >&2
    exit 3
fi

if [[ "${CAPTURE_EQUIVALENCE:-0}" == 1 ]]; then
    if [[ "${DIVERGENCE_PROBE:-0}" == 1 ]]; then
        "${SPECULATORS_CLIENT_RUNTIME}/bin/python3" \
            "${SCRIPT_DIR}/dflash2_speculators_eval.py" capture-probe \
            --dataset-manifest "${DATASET_MANIFEST_PATH}" --hf-home "${HF_HOME}" \
            --endpoint "http://127.0.0.1:${PORT}/v1" --model "${HF_MODEL_CKPT}" \
            --method "${SPEC_METHOD}" --output "${RUN_DIR}/divergence-probe.json"
    else
        "${SPECULATORS_CLIENT_RUNTIME}/bin/python3" \
            "${SCRIPT_DIR}/dflash2_speculators_eval.py" capture-outputs \
            --dataset-manifest "${DATASET_MANIFEST_PATH}" --hf-home "${HF_HOME}" \
            --endpoint "http://127.0.0.1:${PORT}/v1" --model "${HF_MODEL_CKPT}" \
            --output "${RUN_DIR}/output-equivalence.jsonl"
    fi
fi
if [[ "${EQUIVALENCE_ONLY:-0}" == 1 ]]; then
    [[ "${CAPTURE_EQUIVALENCE:-0}:${MAX_CONCURRENCY}:${MAX_REQUESTS}" == 1:1:200 ]] || {
        echo "ERROR: equivalence-only mode requires C1 and exactly 200 requests" >&2
        exit 2
    }
    FINAL_STATUS="success"
    if [[ "${DIVERGENCE_PROBE:-0}" == 1 ]]; then
        echo "Speculators divergence probe complete: ${RUN_DIR}"
    else
        echo "Speculators output-equivalence capture complete: ${RUN_DIR}"
    fi
    exit 0
fi

while IFS=$'\t' read -r subset dataset_path; do
    subset_dir="${RUN_DIR}/subsets/${subset}"
    subset_args_common=(--subset "${subset}" --method "${SPEC_METHOD}"
        --num-speculative-tokens "${NUM_SPEC_TOKENS}"
        --max-concurrency "${MAX_CONCURRENCY}" --max-requests "${MAX_REQUESTS}")
    if "${SPECULATORS_CLIENT_RUNTIME}/bin/python3" "${ARTIFACT_HELPER}" validate-fixed-subset \
        --dir "${subset_dir}" "${subset_args_common[@]}" >/dev/null 2>&1; then
        echo "[INFO] [${subset}] Reusing validated completed subset"
        continue
    fi
    if [[ -e "${subset_dir}" ]]; then
        echo "ERROR: invalid durable subset output requires manual quarantine: ${subset_dir}" >&2
        exit 4
    fi
    attempt_dir="${RUN_DIR}/.attempts/${subset}-${SLURM_JOB_ID:-manual}-$$"
    mkdir "${attempt_dir}"
    subset_args=("${SPECULATORS_REPO}/scripts/evaluate/evaluate.py"
        --target "http://127.0.0.1:${PORT}/v1"
        --dataset "${dataset_path}"
        --subsets "${subset}"
        --output-dir "${attempt_dir}"
        --max-concurrency "${MAX_CONCURRENCY}"
        --max-requests "${MAX_REQUESTS}"
        --gen-kwargs '{"temperature":0,"top_p":1}'
        "${EVAL_MODE}")
    EVALUATOR_ARGS+=(--invocation "${subset_args[@]}")
    "${SPECULATORS_CLIENT_RUNTIME}/bin/python3" "${subset_args[@]}"
    EVALUATOR_RC=$?
    if [[ ${EVALUATOR_RC} -ne 0 ]] \
        && ! [[ "${SPEC_METHOD}" == "baseline" && ${EVALUATOR_RC} -eq 1 ]]; then
        echo "ERROR: Speculators evaluator failed with status ${EVALUATOR_RC}" >&2
        exit "${EVALUATOR_RC}"
    fi
    if [[ "${EVAL_MODE}" == "throughput" ]]; then
        if ! "${SPECULATORS_CLIENT_RUNTIME}/bin/python3" "${ARTIFACT_HELPER}" append-fixed-perf \
            --json "${attempt_dir}/artifacts/run_${subset}.json" \
            --csv "${attempt_dir}/perf_results.csv" \
            --subset "${subset}" \
            --max-concurrency "${MAX_CONCURRENCY}" \
            --max-requests "${MAX_REQUESTS}"; then
            echo "ERROR: failed to derive fixed performance for ${subset}" >&2
            exit 4
        fi
    fi
    if ! "${SPECULATORS_CLIENT_RUNTIME}/bin/python3" "${ARTIFACT_HELPER}" validate-fixed-subset \
        --dir "${attempt_dir}" "${subset_args_common[@]}"; then
        echo "ERROR: invalid fixed subset output for ${subset}" >&2
        exit 4
    fi
    if ! mv "${attempt_dir}" "${subset_dir}"; then
        echo "ERROR: failed to publish fixed subset output for ${subset}" >&2
        exit 4
    fi
done < <("${SPECULATORS_CLIENT_RUNTIME}/bin/python3" "${ARTIFACT_HELPER}" dataset-paths \
    --dataset-manifest "${DATASET_MANIFEST_PATH}" \
    --hf-home "${HF_HOME}")

if [[ "${EVAL_MODE}" == "throughput" ]]; then
    "${SPECULATORS_CLIENT_RUNTIME}/bin/python3" "${ARTIFACT_HELPER}" consolidate-fixed \
        --run-dir "${RUN_DIR}" --method "${SPEC_METHOD}" \
        --num-speculative-tokens "${NUM_SPEC_TOKENS}" \
        --max-concurrency "${MAX_CONCURRENCY}" --max-requests "${MAX_REQUESTS}"
fi
if [[ "${SPEC_METHOD}" != "baseline" ]]; then
    if ! "${SPECULATORS_CLIENT_RUNTIME}/bin/python3" "${ARTIFACT_HELPER}" validate \
        --csv "${RUN_DIR}/acceptance.csv" \
        --num-speculative-tokens "${NUM_SPEC_TOKENS}"; then
        echo "ERROR: invalid Speculators acceptance output" >&2
        exit 4
    fi
fi
if [[ "${EVAL_MODE}" == "sweep" || "${EVAL_MODE}" == "throughput" ]]; then
    if ! "${SPECULATORS_CLIENT_RUNTIME}/bin/python3" "${ARTIFACT_HELPER}" validate-perf \
        --csv "${RUN_DIR}/perf_results.csv"; then
        echo "ERROR: invalid Speculators performance output" >&2
        exit 4
    fi
fi

FINAL_STATUS="success"
echo "Speculators evaluation complete: ${RUN_DIR}"
