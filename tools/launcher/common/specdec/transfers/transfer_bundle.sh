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

# Transfer one immutable, content-addressed artifact bundle through an rclone remote.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LAUNCHER_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
MANIFEST_TOOL="${SCRIPT_DIR}/bundle_manifest.py"
DIRECTION="${1:-}"
[[ "$DIRECTION" == "upload" || "$DIRECTION" == "download" ]] || {
    echo "usage: $0 upload|download --cluster-profile PATH --artifact-id ID --artifact-source-commit COMMIT --remote-root REMOTE:PATH (--source PATH | --destination PATH) [--manifest-sha256 SHA256] [--rclone-bin PATH]" >&2
    exit 2
}
shift

CLUSTER_PROFILE=""
ARTIFACT_ID=""
SOURCE=""
DESTINATION=""
REMOTE_ROOT=""
MANIFEST_SHA256=""
LAUNCHER_COMMIT=""
ARTIFACT_SOURCE_COMMIT=""
RCLONE_BIN="rclone"

usage() {
    echo "usage: $0 $DIRECTION --cluster-profile PATH --artifact-id ID --artifact-source-commit COMMIT --remote-root REMOTE:PATH (--source PATH | --destination PATH) [--manifest-sha256 SHA256] [--rclone-bin PATH]" >&2
    exit 2
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --cluster-profile) CLUSTER_PROFILE="$2"; shift 2 ;;
        --artifact-id) ARTIFACT_ID="$2"; shift 2 ;;
        --source) SOURCE="$2"; shift 2 ;;
        --destination) DESTINATION="$2"; shift 2 ;;
        --remote-root) REMOTE_ROOT="$2"; shift 2 ;;
        --manifest-sha256) MANIFEST_SHA256="$2"; shift 2 ;;
        --launcher-commit) LAUNCHER_COMMIT="$2"; shift 2 ;;
        --launcher-commit=*) LAUNCHER_COMMIT="${1#*=}"; shift ;;
        --artifact-source-commit) ARTIFACT_SOURCE_COMMIT="$2"; shift 2 ;;
        --rclone-bin) RCLONE_BIN="$2"; shift 2 ;;
        *) usage ;;
    esac
done

[[ -n "${SLURM_JOB_ID:-}" && "$SLURM_JOB_ID" =~ ^[0-9]+$ && -n "${SLURM_JOB_NODELIST:-}" && "${SLURM_JOB_NUM_NODES:-}" == "1" ]] || {
    echo "bulk transfers must run inside an approved one-node Slurm allocation" >&2
    exit 2
}
[[ -f "$CLUSTER_PROFILE" ]] || { echo "missing cluster profile: $CLUSTER_PROFILE" >&2; exit 2; }
[[ "$ARTIFACT_ID" =~ ^[a-zA-Z0-9][a-zA-Z0-9._-]{0,79}$ ]] || usage
[[ "$REMOTE_ROOT" =~ ^[^/[:space:]:]+:.+ ]] || usage
if [[ "$DIRECTION" == "upload" ]]; then
    [[ -d "$SOURCE" && -z "$DESTINATION" && -z "$MANIFEST_SHA256" && "$ARTIFACT_SOURCE_COMMIT" =~ ^[0-9a-f]{40}$ ]] || usage
    LOCAL_PATH="$SOURCE"
else
    [[ -z "$SOURCE" && -n "$DESTINATION" && "$MANIFEST_SHA256" =~ ^[0-9a-f]{64}$ && "$ARTIFACT_SOURCE_COMMIT" =~ ^[0-9a-f]{40}$ ]] || usage
    LOCAL_PATH="$DESTINATION"
fi
[[ "$LAUNCHER_COMMIT" =~ ^[0-9a-f]{40}$ ]] || usage
if [[ "$RCLONE_BIN" == */* ]]; then
    [[ -x "$RCLONE_BIN" ]] || { echo "rclone executable is unavailable: $RCLONE_BIN" >&2; exit 2; }
else
    RCLONE_BIN="$(command -v "$RCLONE_BIN")" || { echo "rclone executable is unavailable" >&2; exit 2; }
fi

set +e
profile_data="$(PYTHONPATH="${LAUNCHER_ROOT}${PYTHONPATH:+:${PYTHONPATH}}" python3 - "$CLUSTER_PROFILE" "$LOCAL_PATH" "$DIRECTION" "$SCRIPT_DIR/profiles" <<'PY'
import os
import sys
from pathlib import Path

import yaml
from common.specdec.cluster_profile import load_cluster_profile, select_scratch_root

profile = load_cluster_profile(Path(sys.argv[1]).resolve())
config_path = Path(sys.argv[4]).resolve() / f"{profile.name}.yaml"
if not config_path.is_file():
    raise SystemExit(f"no approved transfer configuration for {profile.name}")
config = yaml.safe_load(config_path.read_text())
if not isinstance(config, dict) or config.get("name") != profile.name:
    raise SystemExit("invalid transfer configuration")
partition = config.get("partition")
if not isinstance(partition, str) or not partition:
    raise SystemExit(f"no approved transfer partition for {profile.name}")
cpu = config.get("cpus_per_task")
gpu = config.get("gpus_per_node")
if (cpu is None) == (gpu is None):
    raise SystemExit("transfer configuration must select exactly one CPU or GPU resource")
for name, value in (("cpus_per_task", cpu), ("gpus_per_node", gpu)):
    if value is not None and (not isinstance(value, int) or isinstance(value, bool) or value < 1):
        raise SystemExit(f"invalid {name}")
durable = profile.durable_root.resolve(strict=True)
requested_local = Path(sys.argv[2])
if requested_local.is_symlink():
    raise SystemExit("local bundle path cannot be a symlink")
local = requested_local.resolve(strict=False)
if local == durable:
    raise SystemExit("local bundle path cannot equal the profile durable root")
if not local.is_relative_to(durable):
    raise SystemExit("local bundle path must be under the profile durable root")
if sys.argv[3] == "upload" and not local.is_dir():
    raise SystemExit("upload source must be a directory")

def writable(path: Path) -> bool:
    return path.is_dir() and os.access(path, os.W_OK | os.X_OK)

scratch = select_scratch_root(profile.scratch_candidates, os.environ, writable)
if scratch.is_relative_to(Path("/lustre")):
    raise SystemExit("transfer scratch cannot use Lustre")
print(profile.account)
print(durable)
print(scratch)
print(local)
print(partition)
print(cpu or "-")
print(gpu or "-")
PY
)"
profile_status=$?
set -e
[[ "$profile_status" -eq 0 ]] || exit 2
profile_values=()
while IFS= read -r profile_value; do profile_values+=("$profile_value"); done <<<"$profile_data"
(( ${#profile_values[@]} == 7 )) || { echo "invalid cluster profile data" >&2; exit 2; }
TRANSFER_ACCOUNT="${profile_values[0]}"
DURABLE_ROOT="${profile_values[1]}"
SCRATCH_ROOT="${profile_values[2]}"
LOCAL_PATH="${profile_values[3]}"
TRANSFER_PARTITION="${profile_values[4]}"
TRANSFER_CPUS="${profile_values[5]}"
TRANSFER_GPUS="${profile_values[6]}"
[[ "$TRANSFER_CPUS" != "-" ]] || TRANSFER_CPUS=""
[[ "$TRANSFER_GPUS" != "-" ]] || TRANSFER_GPUS=""

[[ "${SLURM_JOB_PARTITION:-}" == "$TRANSFER_PARTITION" ]] || {
    echo "allocation partition is not approved for transfers: ${SLURM_JOB_PARTITION:-unset}" >&2
    exit 2
}
[[ "${SLURM_NTASKS:-}" == "1" ]] || {
    echo "transfer allocation must contain exactly one task" >&2
    exit 2
}
if [[ "${SLURM_JOB_ACCOUNT:-}" != "$TRANSFER_ACCOUNT" ]]; then
    echo "allocation account is not approved for transfers: ${SLURM_JOB_ACCOUNT:-unset}" >&2
    exit 2
fi
if [[ -n "$TRANSFER_CPUS" ]]; then
    [[ "${SLURM_CPUS_PER_TASK:-}" == "$TRANSFER_CPUS" ]] || {
        echo "allocation CPU count does not match approved transfer shape" >&2
        exit 2
    }
    [[ "${SLURM_GPUS:-0}" == "0" && "${SLURM_GPUS_ON_NODE:-0}" == "0" ]] || {
        echo "CPU transfer allocation unexpectedly contains GPUs" >&2
        exit 2
    }
else
    [[ "${SLURM_GPUS_ON_NODE:-0}" == "$TRANSFER_GPUS" ]] || {
        echo "GPU transfer allocation does not match approved GPU count" >&2
        exit 2
    }
fi
MODELOPT_ROOT="$(git -C "$LAUNCHER_ROOT" rev-parse --show-toplevel)"
python3 "$MANIFEST_TOOL" verify-launcher --root "$MODELOPT_ROOT" --expected-commit "$LAUNCHER_COMMIT"

TRANSFER_WORK="$(mktemp -d "${SCRATCH_ROOT%/}/modelopt-transfer-${ARTIFACT_ID}.XXXXXX")"
DOWNLOAD_PARTIAL=""
STATE_TEMP_MANIFEST=""
STATE_TEMP_COMPLETION=""
cleanup() {
    if [[ -n "$DOWNLOAD_PARTIAL" && "$DOWNLOAD_PARTIAL" == "${LOCAL_PATH}.partial-${SLURM_JOB_ID}."* ]]; then
        rm -rf -- "$DOWNLOAD_PARTIAL"
    fi
    if [[ -n "${TRANSFER_WORK:-}" && "$TRANSFER_WORK" == "${SCRATCH_ROOT%/}/modelopt-transfer-${ARTIFACT_ID}."* ]]; then
        rm -rf -- "$TRANSFER_WORK"
    fi
    [[ -z "$STATE_TEMP_MANIFEST" ]] || rm -f -- "$STATE_TEMP_MANIFEST"
    [[ -z "$STATE_TEMP_COMPLETION" ]] || rm -f -- "$STATE_TEMP_COMPLETION"
}
trap cleanup EXIT

COPY_ARGS=(
    --checksum
    --checkers="${MODELOPT_RCLONE_CHECKERS:-32}"
    --transfers="${MODELOPT_RCLONE_TRANSFERS:-32}"
    --stats=60s
)
CHECK_ARGS=(--checksum --checkers="${MODELOPT_RCLONE_CHECKERS:-32}" --stats=60s)

run_rclone() {
    TMPDIR="$TRANSFER_WORK" XDG_CACHE_HOME="$TRANSFER_WORK/cache" "$RCLONE_BIN" "$@"
}

fetch_remote() {
    local remote_path="$1"
    local local_path="$2"
    rm -f -- "$local_path"
    set +e
    run_rclone copyto "$remote_path" "$local_path" --checksum
    local status=$?
    set -e
    return "$status"
}

record_verified_state() {
    local completion="$1"
    local manifest="$2"
    local state_root="${DURABLE_ROOT%/}/transfers/${ARTIFACT_ID}/${MANIFEST_SHA256}"
    mkdir -p "$state_root"
    STATE_TEMP_MANIFEST="${state_root}/.manifest.json.${SLURM_JOB_ID}.tmp"
    STATE_TEMP_COMPLETION="${state_root}/.completion.json.${SLURM_JOB_ID}.tmp"
    cp -- "$manifest" "$STATE_TEMP_MANIFEST"
    mv -f -- "$STATE_TEMP_MANIFEST" "$state_root/manifest.json"
    STATE_TEMP_MANIFEST=""
    cp -- "$completion" "$STATE_TEMP_COMPLETION"
    mv -f -- "$STATE_TEMP_COMPLETION" "$state_root/completion.json"
    STATE_TEMP_COMPLETION=""
}

verify_remote_upload() {
    local completion="$1"
    local remote_manifest="$2"
    python3 "$MANIFEST_TOOL" verify-completion --completion "$completion" --artifact-id "$ARTIFACT_ID" --manifest-sha256 "$MANIFEST_SHA256" --artifact-source-commit "$ARTIFACT_SOURCE_COMMIT"
    fetch_remote "${REMOTE_BUNDLE}/manifest.json" "$remote_manifest" || return $?
    python3 "$MANIFEST_TOOL" verify-manifest --manifest "$remote_manifest" --expected-sha256 "$MANIFEST_SHA256" --expected-artifact-source-commit "$ARTIFACT_SOURCE_COMMIT"
    cmp -s -- "$LOCAL_MANIFEST" "$remote_manifest" || {
        echo "remote manifest conflicts with local content" >&2
        return 2
    }
    run_rclone check "$LOCAL_PATH" "${REMOTE_BUNDLE}/payload" "${CHECK_ARGS[@]}"
    python3 "$MANIFEST_TOOL" verify-tree --manifest "$LOCAL_MANIFEST" --root "$LOCAL_PATH"
}

if [[ "$DIRECTION" == "upload" ]]; then
    LOCAL_MANIFEST="${TRANSFER_WORK}/local-manifest.json"
    MANIFEST_SHA256="$(python3 "$MANIFEST_TOOL" create --source "$LOCAL_PATH" --durable-root "$DURABLE_ROOT" --output "$LOCAL_MANIFEST" --artifact-source-commit "$ARTIFACT_SOURCE_COMMIT")"
    REMOTE_BUNDLE="${REMOTE_ROOT%/}/${ARTIFACT_ID}/${MANIFEST_SHA256}"
    EXISTING_COMPLETION="${TRANSFER_WORK}/existing-completion.json"
    REMOTE_MANIFEST="${TRANSFER_WORK}/remote-manifest.json"

    if fetch_remote "${REMOTE_BUNDLE}/completion.json" "$EXISTING_COMPLETION"; then
        verify_remote_upload "$EXISTING_COMPLETION" "$REMOTE_MANIFEST"
        record_verified_state "$EXISTING_COMPLETION" "$REMOTE_MANIFEST"
        echo "reused completed bundle: ${REMOTE_BUNDLE}"
        exit 0
    else
        marker_status=$?
        case "$marker_status" in
            3|4) ;;
            *) echo "could not verify remote artifact state: $ARTIFACT_ID" >&2; exit "$marker_status" ;;
        esac
    fi

    run_rclone copy "$LOCAL_PATH" "${REMOTE_BUNDLE}/payload" "${COPY_ARGS[@]}"
    run_rclone check "$LOCAL_PATH" "${REMOTE_BUNDLE}/payload" "${CHECK_ARGS[@]}"
    python3 "$MANIFEST_TOOL" verify-tree --manifest "$LOCAL_MANIFEST" --root "$LOCAL_PATH"
    run_rclone copyto "$LOCAL_MANIFEST" "${REMOTE_BUNDLE}/manifest.json" --checksum

    if fetch_remote "${REMOTE_BUNDLE}/completion.json" "$EXISTING_COMPLETION"; then
        verify_remote_upload "$EXISTING_COMPLETION" "$REMOTE_MANIFEST"
        record_verified_state "$EXISTING_COMPLETION" "$REMOTE_MANIFEST"
        echo "reused completed bundle: ${REMOTE_BUNDLE}"
        exit 0
    else
        marker_status=$?
        case "$marker_status" in
            3|4) ;;
            *) echo "could not verify remote artifact state: $ARTIFACT_ID" >&2; exit "$marker_status" ;;
        esac
    fi

    LOCAL_COMPLETION="${TRANSFER_WORK}/completion.json"
    python3 "$MANIFEST_TOOL" write-completion --output "$LOCAL_COMPLETION" --artifact-id "$ARTIFACT_ID" --manifest-sha256 "$MANIFEST_SHA256" --artifact-source-commit "$ARTIFACT_SOURCE_COMMIT"
    run_rclone copyto "$LOCAL_COMPLETION" "${REMOTE_BUNDLE}/completion.json" --checksum
    fetch_remote "${REMOTE_BUNDLE}/completion.json" "$EXISTING_COMPLETION" || {
        status=$?; echo "could not verify published completion marker" >&2; exit "$status";
    }
    verify_remote_upload "$EXISTING_COMPLETION" "$REMOTE_MANIFEST"
    record_verified_state "$EXISTING_COMPLETION" "$REMOTE_MANIFEST"
    echo "uploaded bundle: ${REMOTE_BUNDLE}"
    exit 0
fi

REMOTE_BUNDLE="${REMOTE_ROOT%/}/${ARTIFACT_ID}/${MANIFEST_SHA256}"
REMOTE_COMPLETION="${TRANSFER_WORK}/completion.json"
REMOTE_MANIFEST="${TRANSFER_WORK}/manifest.json"
if fetch_remote "${REMOTE_BUNDLE}/completion.json" "$REMOTE_COMPLETION"; then
    :
else
    marker_status=$?
    case "$marker_status" in
        3|4) echo "remote bundle is not complete: $ARTIFACT_ID" >&2; exit 2 ;;
        *) echo "could not fetch remote completion marker: $ARTIFACT_ID" >&2; exit "$marker_status" ;;
    esac
fi
python3 "$MANIFEST_TOOL" verify-completion --completion "$REMOTE_COMPLETION" --artifact-id "$ARTIFACT_ID" --manifest-sha256 "$MANIFEST_SHA256" --artifact-source-commit "$ARTIFACT_SOURCE_COMMIT"
if fetch_remote "${REMOTE_BUNDLE}/manifest.json" "$REMOTE_MANIFEST"; then
    :
else
    status=$?; echo "could not fetch remote manifest: $ARTIFACT_ID" >&2; exit "$status"
fi
python3 "$MANIFEST_TOOL" verify-manifest --manifest "$REMOTE_MANIFEST" --expected-sha256 "$MANIFEST_SHA256" --expected-artifact-source-commit "$ARTIFACT_SOURCE_COMMIT"

if [[ -e "$LOCAL_PATH" || -L "$LOCAL_PATH" ]]; then
    if python3 "$MANIFEST_TOOL" verify-tree --manifest "$REMOTE_MANIFEST" --root "$LOCAL_PATH"; then
        record_verified_state "$REMOTE_COMPLETION" "$REMOTE_MANIFEST"
        echo "reused final destination: $LOCAL_PATH"
        exit 0
    fi
    echo "conflicting final destination: $LOCAL_PATH" >&2
    exit 2
fi

mkdir -p "$(dirname "$LOCAL_PATH")"
DOWNLOAD_PARTIAL="$(mktemp -d "${LOCAL_PATH}.partial-${SLURM_JOB_ID}.XXXXXX")"
run_rclone copy "${REMOTE_BUNDLE}/payload" "$DOWNLOAD_PARTIAL" "${COPY_ARGS[@]}"
python3 "$MANIFEST_TOOL" verify-tree --manifest "$REMOTE_MANIFEST" --root "$DOWNLOAD_PARTIAL"
install_result="$(python3 "$MANIFEST_TOOL" install --manifest "$REMOTE_MANIFEST" --partial "$DOWNLOAD_PARTIAL" --destination "$LOCAL_PATH")" || {
    echo "conflicting final destination: $LOCAL_PATH" >&2
    exit 2
}
DOWNLOAD_PARTIAL=""
record_verified_state "$REMOTE_COMPLETION" "$REMOTE_MANIFEST"
if [[ "$install_result" == "reused" ]]; then
    echo "reused final destination: $LOCAL_PATH"
else
    echo "downloaded bundle: $LOCAL_PATH"
fi
