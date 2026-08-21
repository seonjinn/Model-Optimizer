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

for name in CELL_A CELL_B CONTAINER_IDENTITY_PATH CONTAINER_IMAGE \
    CLUSTER_PROFILE CLUSTER_READINESS_RECEIPT \
    DATASET_MANIFEST_PATH EVAL_CONFIG_PATH EVAL_OUTPUT_ROOT HF_HOME_DURABLE \
    HF_MODEL_CKPT MODELOPT_REPO MODELOPT_SHA PAIR_LABEL RUNTIME_ARCHIVE \
    RUNTIME_ARCHIVE_SHA256 SPECULATORS_REPO SPECULATORS_SHA; do
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
readonly JOB_RUNTIME="${JOB_ROOT}/speculators-runtime"
readonly WRAPPER="${MODELOPT_REPO}/tools/launcher/common/specdec/run_speculators_eval.sh"
readonly STAGER="${MODELOPT_REPO}/tools/launcher/common/specdec/stage_speculators_eval_runtime.sh"
readonly RESULT_ROOT="${EVAL_OUTPUT_ROOT%/}/${PAIR_LABEL}"

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
SPECULATORS_RUNTIME_ARCHIVE="${RUNTIME_ARCHIVE}" \
SPECULATORS_RUNTIME_ARCHIVE_SHA256="${RUNTIME_ARCHIVE_SHA256}" \
SPECULATORS_RUNTIME="${JOB_RUNTIME}" \
CLUSTER_PROFILE="${CLUSTER_PROFILE}" \
CLUSTER_READINESS_RECEIPT="${CLUSTER_READINESS_RECEIPT}" \
MARS_SCRATCH_ROOT="${MARS_SCRATCH_ROOT}" \
bash "${STAGER}" >/dev/null

validate_cell() {
    local descriptor="$1"
    local method block_size spec_tokens concurrency max_requests draft label tp_size
    IFS='|' read -r method block_size spec_tokens concurrency max_requests draft label tp_size \
        <<<"${descriptor}"
    tp_size="${tp_size:-2}"
    validate_label "${label}"
    case "${method}:${block_size}:${spec_tokens}" in
        baseline:0:0|dflash:8:7|dflash:16:15|dspark:8:8|dspark:16:16) ;;
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
    mounts+=",${JOB_RUNTIME}:${JOB_RUNTIME},${HF_MODEL_CKPT}:${HF_MODEL_CKPT}"
    mounts+=",${HF_HOME_DURABLE}:${HF_HOME_DURABLE},${DATASET_MANIFEST_PATH}:${DATASET_MANIFEST_PATH}"
    mounts+=",${CONTAINER_IDENTITY_PATH}:${CONTAINER_IDENTITY_PATH},${CONTAINER_IMAGE}:${CONTAINER_IMAGE}"
    mounts+=",${EVAL_CONFIG_PATH}:${EVAL_CONFIG_PATH},${output_root}:${output_root},${cell_root}:${cell_root}"
    mounts+=",${CLUSTER_PROFILE}:${CLUSTER_PROFILE}"
    mounts+=",${CLUSTER_READINESS_RECEIPT}:${CLUSTER_READINESS_RECEIPT}"
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
        SPECULATORS_RUNTIME="${JOB_RUNTIME}" \
        SPECULATORS_RUNTIME_ARCHIVE_SHA256="${RUNTIME_ARCHIVE_SHA256}" \
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
tp_a="$(cell_tp "${CELL_A}")"
tp_b="$(cell_tp "${CELL_B}")"
if ((tp_a + tp_b <= 4)); then
    run_cell "${CELL_A}" 8000 >"${RESULT_ROOT}/cell-a.log" 2>&1 &
    pid_a=$!
    run_cell "${CELL_B}" 8010 >"${RESULT_ROOT}/cell-b.log" 2>&1 &
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
        [[ ${status} -eq 0 ]] || exit "${status}"
    done
else
    run_supervised_cell "${CELL_A}" 8000 "${RESULT_ROOT}/cell-a.log"
    run_supervised_cell "${CELL_B}" 8010 "${RESULT_ROOT}/cell-b.log"
fi
CHILD_PIDS=()
trap - EXIT INT TERM
