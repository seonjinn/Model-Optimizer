#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

for name in SERVER_RUNTIME_ARCHIVE SERVER_RUNTIME_ARCHIVE_SHA256 \
    SERVER_RUNTIME_RECEIPT_SHA256 VLLM_SERVER_RUNTIME; do
    [[ -n "${!name:-}" ]] || { echo "ERROR: ${name} is required" >&2; exit 2; }
done
[[ "${SERVER_RUNTIME_ARCHIVE_SHA256}" =~ ^[0-9a-f]{64}$ ]]
[[ "${SERVER_RUNTIME_RECEIPT_SHA256}" =~ ^[0-9a-f]{64}$ ]]
readonly JOB_ROOT="${MARS_SCRATCH_ROOT:?MARS_SCRATCH_ROOT is required}/${SLURM_JOB_ID:?SLURM_JOB_ID is required}"
case "${VLLM_SERVER_RUNTIME}" in
    "${JOB_ROOT}"/*) ;;
    *) echo "ERROR: VLLM_SERVER_RUNTIME must be job-local under ${JOB_ROOT}" >&2; exit 2 ;;
esac
[[ -f "${SERVER_RUNTIME_ARCHIVE}" ]]
actual_archive_sha="$(sha256sum "${SERVER_RUNTIME_ARCHIVE}" | cut -d' ' -f1)"
[[ "${actual_archive_sha}" == "${SERVER_RUNTIME_ARCHIVE_SHA256}" ]] || {
    echo "ERROR: server runtime archive SHA-256 mismatch" >&2
    exit 2
}

archive_marker="${VLLM_SERVER_RUNTIME}/.archive.sha256"
receipt="${VLLM_SERVER_RUNTIME}/dflash2-vllm-runtime-receipt.json"
if [[ -x "${VLLM_SERVER_RUNTIME}/bin/python" \
    && -f "${archive_marker}" && "$(<"${archive_marker}")" == "${SERVER_RUNTIME_ARCHIVE_SHA256}" \
    && -f "${receipt}" && "$(sha256sum "${receipt}" | cut -d' ' -f1)" == "${SERVER_RUNTIME_RECEIPT_SHA256}" ]]; then
    exit 0
fi
[[ ! -e "${VLLM_SERVER_RUNTIME}" ]] || {
    echo "ERROR: incomplete server runtime destination already exists" >&2
    exit 2
}

staging="${VLLM_SERVER_RUNTIME}.partial-$$"
trap 'rm -rf -- "${staging}"' EXIT
mkdir -p "${staging}"
tar --extract --file="${SERVER_RUNTIME_ARCHIVE}" --directory="${staging}"
[[ -x "${staging}/bin/python" && -f "${staging}/dflash2-vllm-runtime-receipt.json" ]] || {
    echo "ERROR: staged server runtime is incomplete" >&2
    exit 2
}
[[ "$(sha256sum "${staging}/dflash2-vllm-runtime-receipt.json" | cut -d' ' -f1)" == "${SERVER_RUNTIME_RECEIPT_SHA256}" ]] || {
    echo "ERROR: staged server runtime receipt SHA-256 mismatch" >&2
    exit 2
}

old_venv="$(sed -nE 's/^[[:space:]]*export[[:space:]]+VIRTUAL_ENV=(.*)$/\1/p' "${staging}/bin/activate" | head -n 1)"
old_venv="${old_venv#\"}"
old_venv="${old_venv%\"}"
[[ -n "${old_venv}" ]] || { echo "ERROR: server runtime activation has no VIRTUAL_ENV" >&2; exit 2; }
for candidate in "${staging}/bin/"* "${staging}/pyvenv.cfg"; do
    [[ -f "${candidate}" ]] || continue
    if grep -IqF "${old_venv}" "${candidate}"; then
        sed -i.bak "s|${old_venv}|${VLLM_SERVER_RUNTIME}|g" "${candidate}"
        rm -f "${candidate}.bak"
    fi
done
printf '%s\n' "${SERVER_RUNTIME_ARCHIVE_SHA256}" >"${staging}/.archive.sha256"
mv "${staging}" "${VLLM_SERVER_RUNTIME}"
trap - EXIT
