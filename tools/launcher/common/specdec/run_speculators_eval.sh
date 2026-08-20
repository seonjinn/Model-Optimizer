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

require_var() {
    local name="$1"
    if [[ -z "${!name:-}" ]]; then
        echo "ERROR: ${name} is required" >&2
        exit 2
    fi
}

for name in SPECULATORS_RUNTIME SPECULATORS_REPO HF_MODEL_CKPT DRAFT_MODEL SPEC_METHOD \
    DFLASH_BLOCK_SIZE NUM_SPEC_TOKENS HF_HOME EVAL_OUTPUT_ROOT EVAL_CONFIG_PATH CONTAINER_IMAGE; do
    require_var "$name"
done

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
    python3 "${ARTIFACT_HELPER}" manifest \
        --output "${RUN_DIR}/manifest.json" \
        --status "$1" \
        --method "${SPEC_METHOD}" \
        --block-size "${DFLASH_BLOCK_SIZE}" \
        --num-speculative-tokens "${NUM_SPEC_TOKENS}" \
        --target-model "${HF_MODEL_CKPT}" \
        --draft-model "${DRAFT_MODEL}" \
        --speculators-repo "${SPECULATORS_REPO}" \
        --speculators-sha "${SPECULATORS_EXPECTED_SHA}" \
        --runtime "${SPECULATORS_RUNTIME}" \
        --container-image "${CONTAINER_IMAGE}" \
        --slurm-job-id "${SLURM_JOB_ID:-none}" \
        --launcher-config "${EVAL_CONFIG_PATH}"
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
trap cleanup EXIT INT TERM

for path in "${SPECULATORS_REPO}/scripts/evaluate/evaluate.py" \
    "${HF_MODEL_CKPT}/config.json" "${DRAFT_MODEL}/config.json" "${EVAL_CONFIG_PATH}"; do
    if [[ ! -f "${path}" ]]; then
        echo "ERROR: required staged file is missing: ${path}" >&2
        exit 2
    fi
done

case "${SPEC_METHOD}:${DFLASH_BLOCK_SIZE}:${NUM_SPEC_TOKENS}" in
    dflash:8:7|dflash:16:15|dspark:8:8|dspark:16:16) ;;
    *)
        echo "ERROR: invalid method/B/K mapping: ${SPEC_METHOD}/${DFLASH_BLOCK_SIZE}/${NUM_SPEC_TOKENS}" >&2
        exit 2
        ;;
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

PORT="${VLLM_PORT:-8000}"
TP="${TP_SIZE:-1}"
SPEC_CONFIG="$(printf '{\"method\":\"%s\",\"model\":\"%s\",\"num_speculative_tokens\":%s}' \
    "${SPEC_METHOD}" "${DRAFT_MODEL}" "${NUM_SPEC_TOKENS}")"
python3 -m vllm.entrypoints.cli.main serve "${HF_MODEL_CKPT}" \
    --speculative-config "${SPEC_CONFIG}" \
    --tensor-parallel-size "${TP}" \
    --port "${PORT}" \
    >"${RUN_DIR}/vllm.log" 2>&1 &
SERVER_PID=$!

READY=0
for ((attempt = 1; attempt <= ${SERVE_READY_TIMEOUT:-1800}; attempt++)); do
    if ! kill -0 "${SERVER_PID}" 2>/dev/null; then
        echo "ERROR: vLLM server exited before readiness" >&2
        wait "${SERVER_PID}" || exit $?
    fi
    if curl -fsS "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1 \
        && curl -fsS "http://127.0.0.1:${PORT}/v1/models" \
            | python3 -c 'import json,sys; assert json.load(sys.stdin).get("data")' \
            >/dev/null 2>&1 \
        && curl -fsS "http://127.0.0.1:${PORT}/metrics" 2>/dev/null \
            | grep -q 'vllm:spec_decode'; then
        READY=1
        break
    fi
    sleep 1
done
if [[ ${READY} -ne 1 ]]; then
    echo "ERROR: vLLM health/models/metrics readiness timed out" >&2
    exit 3
fi

python3 "${SPECULATORS_REPO}/scripts/evaluate/evaluate.py" \
    --target "http://127.0.0.1:${PORT}/v1" \
    --dataset RedHatAI/speculator_benchmarks \
    --subsets "${STANDARD_SUBSETS}" \
    --output-dir "${RUN_DIR}" \
    --max-concurrency 128 \
    --max-requests 200 \
    --gen-kwargs '{"temperature":0}' \
    throughput
EVALUATOR_RC=$?
if [[ ${EVALUATOR_RC} -ne 0 ]]; then
    echo "ERROR: Speculators evaluator failed with status ${EVALUATOR_RC}" >&2
    exit "${EVALUATOR_RC}"
fi

if ! python3 "${ARTIFACT_HELPER}" validate \
    --csv "${RUN_DIR}/acceptance.csv" \
    --num-speculative-tokens "${NUM_SPEC_TOKENS}"; then
    echo "ERROR: invalid Speculators acceptance output" >&2
    exit 4
fi

FINAL_STATUS="success"
echo "Speculators evaluation complete: ${RUN_DIR}"
