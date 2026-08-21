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

set -euo pipefail

: "${SLURM_JOB_ID:?SLURM_JOB_ID is required}"
: "${SPECULATORS_RUNTIME_ARCHIVE:?SPECULATORS_RUNTIME_ARCHIVE is required}"
: "${SPECULATORS_RUNTIME_ARCHIVE_SHA256:?SPECULATORS_RUNTIME_ARCHIVE_SHA256 is required}"
: "${SPECULATORS_RUNTIME:?SPECULATORS_RUNTIME is required}"

SCRATCH_ROOT="${MARS_SCRATCH_ROOT:-/raid/scratch}"
JOB_ROOT="${SCRATCH_ROOT%/}/${SLURM_JOB_ID}"
case "${SPECULATORS_RUNTIME}" in
    "${JOB_ROOT}"/*) ;;
    *)
        echo "ERROR: SPECULATORS_RUNTIME must be job-local under ${JOB_ROOT}" >&2
        exit 2
        ;;
esac

if command -v sha256sum >/dev/null 2>&1; then
    ACTUAL_SHA256="$(sha256sum "${SPECULATORS_RUNTIME_ARCHIVE}" | cut -d' ' -f1)"
else
    ACTUAL_SHA256="$(shasum -a 256 "${SPECULATORS_RUNTIME_ARCHIVE}" | cut -d' ' -f1)"
fi
if [[ "${ACTUAL_SHA256}" != "${SPECULATORS_RUNTIME_ARCHIVE_SHA256}" ]]; then
    echo "ERROR: Speculators runtime archive SHA-256 mismatch" >&2
    exit 2
fi

MARKER="${SPECULATORS_RUNTIME}/.archive.sha256"
if [[ -x "${SPECULATORS_RUNTIME}/bin/python3" \
    && -x "${SPECULATORS_RUNTIME}/bin/guidellm" \
    && -f "${MARKER}" \
    && "$(<"${MARKER}")" == "${SPECULATORS_RUNTIME_ARCHIVE_SHA256}" ]]; then
    printf '%s\n' "${SPECULATORS_RUNTIME}"
    exit 0
fi

mkdir -p "${JOB_ROOT}"
STAGING_DIR="$(mktemp -d "${JOB_ROOT}/.speculators-runtime.XXXXXX")"
cleanup() {
    rm -rf "${STAGING_DIR}"
}
trap cleanup EXIT
tar --extract --file="${SPECULATORS_RUNTIME_ARCHIVE}" --directory="${STAGING_DIR}"

ACTIVATE="${STAGING_DIR}/bin/activate"
if [[ ! -f "${ACTIVATE}" ]]; then
    echo "ERROR: staged runtime has no bin/activate" >&2
    exit 2
fi
OLD_VENV="$(sed -nE 's/^[[:space:]]*export[[:space:]]+VIRTUAL_ENV=(.*)$/\1/p' \
    "${ACTIVATE}" | head -n 1)"
OLD_VENV="${OLD_VENV%\"}"
OLD_VENV="${OLD_VENV#\"}"
OLD_VENV="${OLD_VENV%\'}"
OLD_VENV="${OLD_VENV#\'}"
if [[ -z "${OLD_VENV}" ]]; then
    echo "ERROR: staged runtime activation has no VIRTUAL_ENV" >&2
    exit 2
fi
for candidate in "${STAGING_DIR}/bin/"* "${STAGING_DIR}/pyvenv.cfg"; do
    [[ -f "${candidate}" ]] || continue
    if grep -IqF "${OLD_VENV}" "${candidate}"; then
        sed -i.bak "s|${OLD_VENV}|${SPECULATORS_RUNTIME}|g" "${candidate}"
        rm -f "${candidate}.bak"
    fi
done

if [[ ! -x "${STAGING_DIR}/bin/python3" || ! -x "${STAGING_DIR}/bin/guidellm" ]]; then
    echo "ERROR: staged runtime lacks python3 or guidellm" >&2
    exit 2
fi
printf '%s\n' "${SPECULATORS_RUNTIME_ARCHIVE_SHA256}" > "${STAGING_DIR}/.archive.sha256"
if [[ -e "${SPECULATORS_RUNTIME}" ]]; then
    echo "ERROR: incomplete runtime destination already exists: ${SPECULATORS_RUNTIME}" >&2
    exit 2
fi
mv "${STAGING_DIR}" "${SPECULATORS_RUNTIME}"
trap - EXIT
printf '%s\n' "${SPECULATORS_RUNTIME}"
