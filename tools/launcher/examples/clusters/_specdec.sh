#!/usr/bin/env bash
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

# Select a cluster profile, then delegate to one shared SpecDec implementation.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LAUNCHER_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
CLUSTER=""
CLUSTER_PROFILE=""
PRINT_COMMAND=0

usage() {
    echo "usage: $0 --cluster oci-hsg|lyris|ptyche [--cluster-profile PATH] [--print-command] probe|training-wave|full-chain|stage-runtime|upload|download [ARGS...]" >&2
    exit 2
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --cluster) CLUSTER="$2"; shift 2 ;;
        --cluster-profile) CLUSTER_PROFILE="$2"; shift 2 ;;
        --print-command) PRINT_COMMAND=1; shift ;;
        *) break ;;
    esac
done
[[ "$CLUSTER" == "oci-hsg" || "$CLUSTER" == "lyris" || "$CLUSTER" == "ptyche" ]] || usage
ACTION="${1:-}"
[[ -n "$ACTION" ]] || usage
shift
CLUSTER_PROFILE="${CLUSTER_PROFILE:-${LAUNCHER_ROOT}/common/specdec/profiles/${CLUSTER}.yaml}"
[[ -f "$CLUSTER_PROFILE" ]] || { echo "missing cluster profile: $CLUSTER_PROFILE" >&2; exit 2; }

PROFILE_NAME="$(PYTHONPATH="${LAUNCHER_ROOT}${PYTHONPATH:+:${PYTHONPATH}}" python3 - "$CLUSTER_PROFILE" <<'PY'
import sys
from pathlib import Path

from common.specdec.cluster_profile import load_cluster_profile

print(load_cluster_profile(Path(sys.argv[1]).resolve()).name)
PY
)"
[[ "$PROFILE_NAME" == "$CLUSTER" ]] || {
    echo "profile name $PROFILE_NAME does not match wrapper cluster $CLUSTER" >&2
    exit 2
}

for argument in "$@"; do
    [[ "$argument" != "--cluster-profile" && "$argument" != "--profile" ]] || {
        echo "profile overrides must precede the action" >&2
        exit 2
    }
done

requires_readiness=0
case "$ACTION" in
    probe)
        command=("${LAUNCHER_ROOT}/common/specdec/probe_cluster_profile.sh" --profile "$CLUSTER_PROFILE" "$@")
        ;;
    training-wave)
        requires_readiness=1
        command=("${LAUNCHER_ROOT}/common/specdec/submit_drafter_training_wave.sh" --cluster-profile "$CLUSTER_PROFILE" "$@")
        ;;
    full-chain)
        requires_readiness=1
        command=("${LAUNCHER_ROOT}/common/specdec/submit_drafter_full_chain.sh" --cluster-profile "$CLUSTER_PROFILE" "$@")
        ;;
    stage-runtime)
        requires_readiness=1
        command=("${LAUNCHER_ROOT}/common/specdec/stage_relocatable_runtime_archive.sh" --cluster-profile "$CLUSTER_PROFILE" "$@")
        ;;
    upload)
        command=("${LAUNCHER_ROOT}/common/specdec/transfers/submit_transfer.sh" --cluster-profile "$CLUSTER_PROFILE" upload "$@")
        ;;
    download)
        command=("${LAUNCHER_ROOT}/common/specdec/transfers/submit_transfer.sh" --cluster-profile "$CLUSTER_PROFILE" download "$@")
        ;;
    *) echo "unsupported action for portable cluster wrapper: $ACTION" >&2; usage ;;
esac

if [[ "$requires_readiness" -eq 1 ]]; then
    readiness=""
    action_arguments=("$@")
    for ((index = 0; index < ${#action_arguments[@]}; index++)); do
        if [[ "${action_arguments[$index]}" == "--readiness-receipt" ]]; then
            readiness="${action_arguments[$((index + 1))]:-}"
            break
        fi
    done
    [[ -n "$readiness" ]] || { echo "--readiness-receipt is required for $ACTION" >&2; exit 2; }
fi

if [[ "$PRINT_COMMAND" -eq 1 ]]; then
    printf '%q ' "${command[@]}"
    printf '\n'
    exit 0
fi
exec "${command[@]}"
