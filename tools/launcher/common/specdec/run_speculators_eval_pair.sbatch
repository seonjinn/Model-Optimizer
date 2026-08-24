#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

#SBATCH --account=nemotron_n3_post
#SBATCH --partition=batch
#SBATCH --job-name=specdec-eval-pair
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=72
#SBATCH --gpus-per-node=4
#SBATCH --segment=1
#SBATCH --mem=900G
#SBATCH --time=03:55:00

set -euo pipefail

CLIENT_RUNTIME_ARCHIVE="${CLIENT_RUNTIME_ARCHIVE:-${RUNTIME_ARCHIVE:-}}"
CLIENT_RUNTIME_ARCHIVE_SHA256="${CLIENT_RUNTIME_ARCHIVE_SHA256:-${RUNTIME_ARCHIVE_SHA256:-}}"
SERVER_RUNTIME_ARCHIVE="${SERVER_RUNTIME_ARCHIVE:-${CLIENT_RUNTIME_ARCHIVE}}"
SERVER_RUNTIME_ARCHIVE_SHA256="${SERVER_RUNTIME_ARCHIVE_SHA256:-${CLIENT_RUNTIME_ARCHIVE_SHA256}}"
PAIR_PHASE="${PAIR_PHASE:-legacy}"

for name in CELL_A CELL_B CONTAINER_IDENTITY_PATH CONTAINER_IMAGE \
    CLIENT_RUNTIME_ARCHIVE CLIENT_RUNTIME_ARCHIVE_SHA256 \
    CLUSTER_PROFILE CLUSTER_READINESS_RECEIPT \
    DATASET_MANIFEST_PATH EVAL_CONFIG_PATH EVAL_OUTPUT_ROOT HF_HOME_DURABLE \
    HF_MODEL_CKPT MODELOPT_REPO MODELOPT_SHA PAIR_LABEL \
    SERVER_RUNTIME_ARCHIVE SERVER_RUNTIME_ARCHIVE_SHA256 \
    SPECULATORS_REPO SPECULATORS_SHA; do
    [[ -n "${!name:-}" ]] || { echo "ERROR: ${name} is required" >&2; exit 2; }
done

validate_label() {
    local label="$1"
    [[ "${label}" =~ ^[a-z0-9][a-z0-9._-]{0,63}$ ]] || {
        echo "ERROR: unsafe label: ${label}" >&2
        return 2
    }
}

validate_label "${PAIR_LABEL}"

MARS_SCRATCH_ROOT="${MARS_SCRATCH_ROOT:-/raid/scratch}"
readonly JOB_ROOT="${MARS_SCRATCH_ROOT%/}/${SLURM_JOB_ID}"
readonly JOB_CLIENT_RUNTIME="${JOB_ROOT}/speculators-client-runtime"
JOB_SERVER_RUNTIME="${JOB_CLIENT_RUNTIME}"
readonly WRAPPER="${MODELOPT_REPO}/tools/launcher/common/specdec/run_speculators_eval.sh"
readonly CLIENT_STAGER="${MODELOPT_REPO}/tools/launcher/common/specdec/stage_speculators_eval_runtime.sh"
readonly SERVER_STAGER="${MODELOPT_REPO}/tools/launcher/common/specdec/stage_speculators_eval_server_runtime.sh"
readonly STUDY_HELPER="${MODELOPT_REPO}/tools/launcher/common/specdec/dflash2_speculators_eval.py"
readonly RESULT_ROOT="${EVAL_OUTPUT_ROOT%/}/${PAIR_LABEL}"
ARTIFACT_IDENTITY_PATH=""
ARTIFACT_IDENTITY_SHA256=""
EQUIVALENCE_ONLY_MODE=0

[[ "$(git -C "${MODELOPT_REPO}" rev-parse HEAD)" == "${MODELOPT_SHA}" ]]
[[ -z "$(git -C "${MODELOPT_REPO}" status --porcelain)" ]]
[[ "$(git -C "${SPECULATORS_REPO}" rev-parse HEAD)" == "${SPECULATORS_SHA}" ]]
[[ -z "$(git -C "${SPECULATORS_REPO}" status --porcelain)" ]]
for path in "${CONTAINER_IMAGE}" "${CONTAINER_IDENTITY_PATH}" \
    "${DATASET_MANIFEST_PATH}" "${EVAL_CONFIG_PATH}" \
    "${HF_MODEL_CKPT}/config.json"; do
    [[ -r "${path}" ]] || { echo "ERROR: missing input: ${path}" >&2; exit 2; }
done

mkdir -p "${JOB_ROOT}" "${RESULT_ROOT}"
SPECULATORS_RUNTIME_ARCHIVE="${CLIENT_RUNTIME_ARCHIVE}" \
SPECULATORS_RUNTIME_ARCHIVE_SHA256="${CLIENT_RUNTIME_ARCHIVE_SHA256}" \
SPECULATORS_RUNTIME="${JOB_CLIENT_RUNTIME}" \
CLUSTER_PROFILE="${CLUSTER_PROFILE}" \
CLUSTER_READINESS_RECEIPT="${CLUSTER_READINESS_RECEIPT}" \
MARS_SCRATCH_ROOT="${MARS_SCRATCH_ROOT}" \
bash "${CLIENT_STAGER}" >/dev/null
if [[ -n "${SERVER_RUNTIME_RECEIPT_SHA256:-}" ]]; then
    JOB_SERVER_RUNTIME="${JOB_ROOT}/dflash2-server-runtime"
    SERVER_RUNTIME_ARCHIVE="${SERVER_RUNTIME_ARCHIVE}" \
    SERVER_RUNTIME_ARCHIVE_SHA256="${SERVER_RUNTIME_ARCHIVE_SHA256}" \
    SERVER_RUNTIME_RECEIPT_SHA256="${SERVER_RUNTIME_RECEIPT_SHA256}" \
    VLLM_SERVER_RUNTIME="${JOB_SERVER_RUNTIME}" \
    MARS_SCRATCH_ROOT="${MARS_SCRATCH_ROOT}" \
    bash "${SERVER_STAGER}"
fi

validate_cell() {
    local descriptor="$1"
    local method block_size spec_tokens concurrency max_requests draft label tp_size
    IFS='|' read -r method block_size spec_tokens concurrency max_requests draft label tp_size \
        <<<"${descriptor}"
    tp_size="${tp_size:-2}"
    validate_label "${label}"
    case "${method}:${block_size}:${spec_tokens}" in
        baseline:0:0|dflash:8:7|dflash:16:15|dflash2:8:7|dspark:8:8|dspark:16:16) ;;
        *) echo "ERROR: invalid cell descriptor: ${descriptor}" >&2; return 2 ;;
    esac
    case "${concurrency}" in
        1|8|32|128) ;;
        *) echo "ERROR: invalid concurrency: ${concurrency}" >&2; return 2 ;;
    esac
    case "${tp_size}" in
        1|2|4) ;;
        *) echo "ERROR: invalid tensor parallel size: ${tp_size}" >&2; return 2 ;;
    esac
    if [[ "${concurrency}" == 128 && ${max_requests} -lt 512 ]]; then
        echo "ERROR: C128 requires at least 512 requests" >&2
        return 2
    fi
    if [[ "${method}" != baseline ]]; then
        [[ -r "${draft}/config.json" && -r "${draft}/model.safetensors" ]]
    fi
}

run_cell() {
    local descriptor="$1" port="$2"
    local method block_size spec_tokens concurrency max_requests draft label tp_size
    validate_cell "${descriptor}"
    IFS='|' read -r method block_size spec_tokens concurrency max_requests draft label tp_size \
        <<<"${descriptor}"
    tp_size="${tp_size:-2}"

    local cell_root="${JOB_ROOT}/${label}"
    local output_root="${RESULT_ROOT}/${label}"
    local resolved_config="${cell_root}/resolved.yaml"
    mkdir -p "${cell_root}/hf-datasets" "${cell_root}/triton" \
        "${cell_root}/torchinductor" "${cell_root}/xdg" "${output_root}"
    printf 'pipeline:\n  task_0:\n    slurm_config:\n      container: %s\n' \
        "${CONTAINER_IMAGE}" >"${resolved_config}"

    local mounts
    mounts="${MODELOPT_REPO}:${MODELOPT_REPO},${SPECULATORS_REPO}:${SPECULATORS_REPO}"
    mounts+=",${JOB_CLIENT_RUNTIME}:${JOB_CLIENT_RUNTIME},${JOB_SERVER_RUNTIME}:${JOB_SERVER_RUNTIME}"
    mounts+=",${HF_MODEL_CKPT}:${HF_MODEL_CKPT}"
    mounts+=",${HF_HOME_DURABLE}:${HF_HOME_DURABLE},${DATASET_MANIFEST_PATH}:${DATASET_MANIFEST_PATH}"
    mounts+=",${CONTAINER_IDENTITY_PATH}:${CONTAINER_IDENTITY_PATH},${CONTAINER_IMAGE}:${CONTAINER_IMAGE}"
    mounts+=",${EVAL_CONFIG_PATH}:${EVAL_CONFIG_PATH},${output_root}:${output_root},${cell_root}:${cell_root}"
    mounts+=",${CLUSTER_PROFILE}:${CLUSTER_PROFILE}"
    mounts+=",${CLUSTER_READINESS_RECEIPT}:${CLUSTER_READINESS_RECEIPT}"
    if [[ -n "${ARTIFACT_IDENTITY_PATH}" ]]; then
        mounts+=",${ARTIFACT_IDENTITY_PATH}:${ARTIFACT_IDENTITY_PATH}"
    fi
    if [[ "${method}" != baseline ]]; then
        mounts+=",${draft}:${draft}"
    fi

    exec srun --exclusive --nodes=1 --ntasks=1 --gpus="${tp_size}" --cpus-per-task=72 --mem=450G \
        --kill-on-bad-exit=1 \
        --container-image="${CONTAINER_IMAGE}" \
        --container-mounts="${mounts}" \
        --container-workdir="${MODELOPT_REPO}/tools/launcher" \
        --no-container-mount-home \
        env \
        SPECULATORS_CLIENT_RUNTIME="${JOB_CLIENT_RUNTIME}" \
        SPECULATORS_RUNTIME_ARCHIVE_SHA256="${CLIENT_RUNTIME_ARCHIVE_SHA256}" \
        VLLM_SERVER_RUNTIME="${JOB_SERVER_RUNTIME}" \
        VLLM_SERVER_RUNTIME_ARCHIVE_SHA256="${SERVER_RUNTIME_ARCHIVE_SHA256}" \
        VLLM_SERVER_RUNTIME_RECEIPT_SHA256="${SERVER_RUNTIME_RECEIPT_SHA256}" \
        SPECULATORS_REPO="${SPECULATORS_REPO}" \
        MODELOPT_REPO="${MODELOPT_REPO}" \
        CLUSTER_PROFILE="${CLUSTER_PROFILE}" \
        CLUSTER_READINESS_RECEIPT="${CLUSTER_READINESS_RECEIPT}" \
        MARS_SCRATCH_ROOT="${MARS_SCRATCH_ROOT}" \
        HF_MODEL_CKPT="${HF_MODEL_CKPT}" \
        DRAFT_MODEL="${draft}" \
        SPEC_METHOD="${method}" \
        DFLASH_BLOCK_SIZE="${block_size}" \
        NUM_SPEC_TOKENS="${spec_tokens}" \
        MAX_CONCURRENCY="${concurrency}" \
        MAX_REQUESTS="${max_requests}" \
        EVAL_MODE=throughput \
        TP_SIZE="${tp_size}" \
        VLLM_PORT="${port}" \
        SERVE_READY_TIMEOUT=1800 \
        HF_HOME="${HF_HOME_DURABLE}" \
        PYTHONPATH="${SPECULATORS_REPO}/hs_connectors/src" \
        HF_DATASETS_CACHE="${cell_root}/hf-datasets" \
        TRITON_CACHE_DIR="${cell_root}/triton" \
        TORCHINDUCTOR_CACHE_DIR="${cell_root}/torchinductor" \
        XDG_CACHE_HOME="${cell_root}/xdg" \
        EVAL_OUTPUT_ROOT="${output_root}" \
        EVAL_RUN_ID="${label}" \
        EVAL_CONFIG_PATH="${resolved_config}" \
        DATASET_MANIFEST_PATH="${DATASET_MANIFEST_PATH}" \
        EVAL_ARTIFACT_IDENTITY_PATH="${ARTIFACT_IDENTITY_PATH}" \
        EVAL_ARTIFACT_IDENTITY_SHA256="${ARTIFACT_IDENTITY_SHA256}" \
        CAPTURE_EQUIVALENCE="${EQUIVALENCE_ONLY_MODE}" \
        EQUIVALENCE_ONLY="${EQUIVALENCE_ONLY_MODE}" \
        CONTAINER_IMAGE="${CONTAINER_IMAGE}" \
        CONTAINER_IDENTITY_PATH="${CONTAINER_IDENTITY_PATH}" \
        bash "${WRAPPER}"
}

cell_tp() {
    local descriptor="$1"
    local method block_size spec_tokens concurrency max_requests draft label tp_size
    IFS='|' read -r method block_size spec_tokens concurrency max_requests draft label tp_size \
        <<<"${descriptor}"
    tp_size="${tp_size:-2}"
    case "${tp_size}" in
        1|2|4) printf '%s\n' "${tp_size}" ;;
        *) echo "ERROR: invalid tensor parallel size: ${tp_size}" >&2; return 2 ;;
    esac
}

CHILD_PIDS=()
terminate_children() {
    local pid
    trap - EXIT INT TERM
    for pid in "${CHILD_PIDS[@]:-}"; do
        [[ -n "${pid}" ]] || continue
        kill "${pid}" 2>/dev/null || true
    done
    for pid in "${CHILD_PIDS[@]:-}"; do
        [[ -n "${pid}" ]] || continue
        wait "${pid}" 2>/dev/null || true
    done
    CHILD_PIDS=()
}

run_supervised_cell() {
    local descriptor="$1" port="$2" log="$3" pid status
    run_cell "${descriptor}" "${port}" >"${log}" 2>&1 &
    pid=$!
    CHILD_PIDS=("${pid}")
    set +e
    wait "${pid}"
    status=$?
    set -e
    CHILD_PIDS=()
    return "${status}"
}
trap terminate_children EXIT
trap 'terminate_children; exit 130' INT
trap 'terminate_children; exit 143' TERM

validate_cell "${CELL_A}"
validate_cell "${CELL_B}"
method_a="${CELL_A%%|*}"
method_b="${CELL_B%%|*}"
if [[ "${method_a}" == dflash2 || "${method_b}" == dflash2 ]]; then
    for name in SERVER_RUNTIME_RECEIPT_SHA256 \
        DRAFT_MILESTONE_MANIFEST_PATH DRAFT_MILESTONE_MANIFEST_SHA256 \
        TARGET_RECEIPT_PATH TARGET_RECEIPT_SHA256 TARGET_SHA256; do
        [[ -n "${!name:-}" ]] || { echo "ERROR: ${name} is required for DFlash2" >&2; exit 2; }
    done
    [[ "${method_a}:${method_b}" == baseline:dflash2 \
        || "${method_a}:${method_b}" == dflash2:baseline ]] || {
        echo "ERROR: matched DFlash2 pair must contain exactly baseline and dflash2" >&2
        exit 2
    }
    for descriptor in "${CELL_A}" "${CELL_B}"; do
        IFS='|' read -r _ _ _ concurrency max_requests _ _ tp_size <<<"${descriptor}"
        tp_size="${tp_size:-2}"
        case "${PAIR_PHASE}:${concurrency}:${max_requests}:${tp_size}" in
            correctness:1:200:2|performance:32:200:2) ;;
            *) echo "ERROR: matched DFlash2 cells require C1/C32, 200 requests, and TP2" >&2; exit 2 ;;
        esac
    done
    dflash2_descriptor="${CELL_A}"
    [[ "${method_a}" == dflash2 ]] || dflash2_descriptor="${CELL_B}"
    IFS='|' read -r _ _ _ _ _ dflash2_export _ _ <<<"${dflash2_descriptor}"
    [[ "$(sha256sum "${DRAFT_MILESTONE_MANIFEST_PATH}" | cut -d' ' -f1)" == "${DRAFT_MILESTONE_MANIFEST_SHA256}" ]]
    ARTIFACT_IDENTITY_PATH="${RESULT_ROOT}/artifact-identity.json"
    PYTHONPATH="${MODELOPT_REPO}/tools/launcher${PYTHONPATH:+:${PYTHONPATH}}" \
        "${JOB_CLIENT_RUNTIME}/bin/python3" "${STUDY_HELPER}" artifact-identity \
        --output "${ARTIFACT_IDENTITY_PATH}" \
        --target "${HF_MODEL_CKPT}" --target-sha256 "${TARGET_SHA256}" \
        --target-receipt "${TARGET_RECEIPT_PATH}" \
        --target-receipt-sha256 "${TARGET_RECEIPT_SHA256}" \
        --export "${dflash2_export}" \
        --milestone-manifest "${DRAFT_MILESTONE_MANIFEST_PATH}" \
        --dataset-manifest "${DATASET_MANIFEST_PATH}" --hf-home "${HF_HOME_DURABLE}" \
        --client-runtime-archive "${CLIENT_RUNTIME_ARCHIVE}" \
        --client-runtime-archive-sha256 "${CLIENT_RUNTIME_ARCHIVE_SHA256}" \
        --server-runtime-archive "${SERVER_RUNTIME_ARCHIVE}" \
        --server-runtime-archive-sha256 "${SERVER_RUNTIME_ARCHIVE_SHA256}" \
        --server-runtime-receipt-sha256 "${SERVER_RUNTIME_RECEIPT_SHA256}"
    ARTIFACT_IDENTITY_SHA256="$(sha256sum "${ARTIFACT_IDENTITY_PATH}" | cut -d' ' -f1)"
fi
cell_label() {
    local descriptor="$1" method block_size spec_tokens concurrency max_requests draft label tp_size
    IFS='|' read -r method block_size spec_tokens concurrency max_requests draft label tp_size \
        <<<"${descriptor}"
    printf '%s\n' "${label}"
}

run_pair_cells() {
    local equivalence_only="$1" suffix=""
    local tp_a tp_b pid_a pid_b status completed_pid
    local -a active_pids remaining_pids
    EQUIVALENCE_ONLY_MODE="${equivalence_only}"
    [[ "${equivalence_only}" == 0 ]] || suffix="-equivalence"
    tp_a="$(cell_tp "${CELL_A}")"
    tp_b="$(cell_tp "${CELL_B}")"
    if ((tp_a + tp_b <= 4)); then
        run_cell "${CELL_A}" 8000 >"${RESULT_ROOT}/cell-a${suffix}.log" 2>&1 &
        pid_a=$!
        run_cell "${CELL_B}" 8010 >"${RESULT_ROOT}/cell-b${suffix}.log" 2>&1 &
        pid_b=$!
        CHILD_PIDS=("${pid_a}" "${pid_b}")
        active_pids=("${CHILD_PIDS[@]}")
        while ((${#active_pids[@]})); do
            set +e
            wait -n -p completed_pid "${active_pids[@]}"
            status=$?
            set -e
            remaining_pids=()
            for pid in "${active_pids[@]}"; do
                [[ "${pid}" == "${completed_pid}" ]] || remaining_pids+=("${pid}")
            done
            active_pids=("${remaining_pids[@]}")
            CHILD_PIDS=("${active_pids[@]}")
            [[ ${status} -eq 0 ]] || return "${status}"
        done
    else
        run_supervised_cell "${CELL_A}" 8000 "${RESULT_ROOT}/cell-a${suffix}.log"
        run_supervised_cell "${CELL_B}" 8010 "${RESULT_ROOT}/cell-b${suffix}.log"
    fi
    CHILD_PIDS=()
}

if [[ "${PAIR_PHASE}" == correctness && ( "${method_a}" == dflash2 || "${method_b}" == dflash2 ) ]]; then
    run_pair_cells 1
    if [[ "${method_a}" == baseline ]]; then
        baseline_label="$(cell_label "${CELL_A}")"
        dflash2_label="$(cell_label "${CELL_B}")"
    else
        baseline_label="$(cell_label "${CELL_B}")"
        dflash2_label="$(cell_label "${CELL_A}")"
    fi
    baseline_run="${RESULT_ROOT}/${baseline_label}/${baseline_label}"
    dflash2_run="${RESULT_ROOT}/${dflash2_label}/${dflash2_label}"
    "${JOB_CLIENT_RUNTIME}/bin/python3" "${STUDY_HELPER}" compare-outputs \
        --baseline "${baseline_run}/output-equivalence.jsonl" \
        --dflash2 "${dflash2_run}/output-equivalence.jsonl" \
        --artifact-identity "${ARTIFACT_IDENTITY_PATH}" \
        --output "${RESULT_ROOT}/correctness-receipt.json"
fi
run_pair_cells 0
trap - EXIT INT TERM

if [[ "${method_a}" == dflash2 || "${method_b}" == dflash2 ]]; then
    if [[ "${method_a}" == baseline ]]; then
        baseline_label="$(cell_label "${CELL_A}")"
        dflash2_label="$(cell_label "${CELL_B}")"
    else
        baseline_label="$(cell_label "${CELL_B}")"
        dflash2_label="$(cell_label "${CELL_A}")"
    fi
    baseline_run="${RESULT_ROOT}/${baseline_label}/${baseline_label}"
    dflash2_run="${RESULT_ROOT}/${dflash2_label}/${dflash2_label}"
    if [[ "${PAIR_PHASE}" == correctness ]]; then
        "${JOB_CLIENT_RUNTIME}/bin/python3" "${STUDY_HELPER}" summarize \
            --baseline-run "${baseline_run}" --dflash2-run "${dflash2_run}" \
            --correctness-receipt "${RESULT_ROOT}/correctness-receipt.json" \
            --artifact-identity "${ARTIFACT_IDENTITY_PATH}" \
            --concurrency 1 \
            --output "${RESULT_ROOT}/c1-performance-report.json"
        "${JOB_CLIENT_RUNTIME}/bin/python3" "${STUDY_HELPER}" verify-report \
            --report "${RESULT_ROOT}/c1-performance-report.json" >/dev/null
    else
        [[ -n "${CORRECTNESS_RECEIPT_PATH:-}" && -n "${CORRECTNESS_RECEIPT_SHA256:-}" ]]
        [[ "$(sha256sum "${CORRECTNESS_RECEIPT_PATH}" | cut -d' ' -f1)" == "${CORRECTNESS_RECEIPT_SHA256}" ]]
        "${JOB_CLIENT_RUNTIME}/bin/python3" "${STUDY_HELPER}" summarize \
            --baseline-run "${baseline_run}" --dflash2-run "${dflash2_run}" \
            --correctness-receipt "${CORRECTNESS_RECEIPT_PATH}" \
            --artifact-identity "${ARTIFACT_IDENTITY_PATH}" \
            --concurrency 32 \
            --output "${RESULT_ROOT}/c32-performance-report.json"
        "${JOB_CLIENT_RUNTIME}/bin/python3" "${STUDY_HELPER}" verify-report \
            --report "${RESULT_ROOT}/c32-performance-report.json" >/dev/null
    fi
fi
