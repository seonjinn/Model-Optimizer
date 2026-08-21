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
readonly STANDARD_SUBSETS="HumanEval,math_reasoning,qa,question,rag,summarization,tool_call,translation,writing"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
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

require_var() {
    local name="$1"
    if [[ -z "${!name:-}" ]]; then
        echo "ERROR: ${name} is required" >&2
        exit 2
    fi
}

for name in SPECULATORS_RUNTIME SPECULATORS_REPO HF_MODEL_CKPT SPEC_METHOD \
    HF_HOME EVAL_OUTPUT_ROOT EVAL_CONFIG_PATH CONTAINER_IMAGE \
    CONTAINER_IDENTITY_PATH DATASET_MANIFEST_PATH MODELOPT_REPO; do
    require_var "$name"
done
if [[ "${SPEC_METHOD}" != "baseline" ]]; then
    require_var DRAFT_MODEL
fi

if [[ ! -f "${SPECULATORS_RUNTIME}/bin/activate" ]]; then
    echo "ERROR: shared runtime is missing: ${SPECULATORS_RUNTIME}" >&2
    exit 2
fi
# shellcheck disable=SC1090
source "${SPECULATORS_RUNTIME}/bin/activate"
export PATH="${SPECULATORS_RUNTIME}/bin:${PATH}"
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

RUN_ID="${EVAL_RUN_ID:-${SLURM_JOB_ID:-manual}-$(date -u +%Y%m%dT%H%M%SZ)-$$-${SPEC_METHOD}-b${DFLASH_BLOCK_SIZE}}"
RUN_DIR="${EVAL_OUTPUT_ROOT%/}/${RUN_ID}"
mkdir -p "${EVAL_OUTPUT_ROOT}"
if ! mkdir "${RUN_DIR}"; then
    echo "ERROR: evaluation output already exists: ${RUN_DIR}" >&2
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
        --runtime "${SPECULATORS_RUNTIME}" \
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
    python3 "${ARTIFACT_HELPER}" "${helper_args[@]}"
}

cleanup() {
    local rc=$?
    trap - EXIT INT TERM
    if [[ -n "${SERVER_PID}" ]]; then
        kill "${SERVER_PID}" 2>/dev/null || true
        wait "${SERVER_PID}" 2>/dev/null || true
    fi
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
    baseline:0:0|dflash:8:7|dflash:16:15|dspark:8:8|dspark:16:16) ;;
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
    throughput|sweep) ;;
    *) echo "ERROR: invalid EVAL_MODE: ${EVAL_MODE}; expected throughput or sweep" >&2; exit 2 ;;
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
if ! python3 "${ARTIFACT_HELPER}" verify-inputs \
    --dataset-manifest "${DATASET_MANIFEST_PATH}" \
    --hf-home "${HF_HOME}" \
    --container-identity "${CONTAINER_IDENTITY_PATH}" \
    --container-image "${CONTAINER_IMAGE}" \
    --launcher-config "${EVAL_CONFIG_PATH}"; then
    echo "ERROR: staged dataset or container identity verification failed" >&2
    exit 2
fi

PYTHON_VERSION="$(python3 --version 2>&1)"
VLLM_VERSION="$(python3 -c 'from importlib.metadata import version; print(version("vllm"))')"
GUIDELLM_VERSION="$(python3 -c 'from importlib.metadata import version; print(version("guidellm"))')"

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
        "${SPEC_METHOD}" "${DRAFT_MODEL}" "${NUM_SPEC_TOKENS}")"
    SERVER_ARGS+=(--speculative-config "${SPEC_CONFIG}")
fi
python3 "${SERVER_ARGS[@]}" \
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
            | python3 -c 'import json,sys; data=json.load(sys.stdin).get("data", []); sys.exit(0 if any(item.get("id") == sys.argv[1] for item in data) else 1)' "${HF_MODEL_CKPT}" \
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

while IFS=$'\t' read -r subset dataset_path; do
    subset_args=("${SPECULATORS_REPO}/scripts/evaluate/evaluate.py"
        --target "http://127.0.0.1:${PORT}/v1"
        --dataset "${dataset_path}"
        --subsets "${subset}"
        --output-dir "${RUN_DIR}"
        --max-concurrency "${MAX_CONCURRENCY}"
        --max-requests "${MAX_REQUESTS}"
        --gen-kwargs '{"temperature":0,"top_p":1}'
        "${EVAL_MODE}")
    EVALUATOR_ARGS+=(--invocation "${subset_args[@]}")
    python3 "${subset_args[@]}"
    EVALUATOR_RC=$?
    if [[ ${EVALUATOR_RC} -ne 0 \
        && !( "${SPEC_METHOD}" == "baseline" && "${EVAL_MODE}" == "sweep" \
            && ${EVALUATOR_RC} -eq 1 ) ]]; then
        echo "ERROR: Speculators evaluator failed with status ${EVALUATOR_RC}" >&2
        exit "${EVALUATOR_RC}"
    fi
done < <(python3 "${ARTIFACT_HELPER}" dataset-paths \
    --dataset-manifest "${DATASET_MANIFEST_PATH}" \
    --hf-home "${HF_HOME}")

if [[ "${SPEC_METHOD}" != "baseline" ]]; then
    if ! python3 "${ARTIFACT_HELPER}" validate \
        --csv "${RUN_DIR}/acceptance.csv" \
        --num-speculative-tokens "${NUM_SPEC_TOKENS}"; then
        echo "ERROR: invalid Speculators acceptance output" >&2
        exit 4
    fi
fi
if [[ "${EVAL_MODE}" == "sweep" ]]; then
    if ! python3 "${ARTIFACT_HELPER}" validate-perf \
        --csv "${RUN_DIR}/perf_results.csv"; then
        echo "ERROR: invalid Speculators performance output" >&2
        exit 4
    fi
fi

FINAL_STATUS="success"
echo "Speculators evaluation complete: ${RUN_DIR}"
