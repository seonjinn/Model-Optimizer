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

# Submit one bulk transfer to a profile-approved one-node Slurm partition.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LAUNCHER_ROOT="$(git -C "${SCRIPT_DIR}/../../../.." rev-parse --show-toplevel)"
PROFILE=""
DRY_RUN=0

usage() {
    echo "usage: $0 --cluster-profile PATH [--dry-run] upload|download [TRANSFER_ARGS...]" >&2
    exit 2
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --cluster-profile) PROFILE="$2"; shift 2 ;;
        --dry-run) DRY_RUN=1; shift ;;
        upload|download) DIRECTION="$1"; shift; break ;;
        *) usage ;;
    esac
done
[[ -f "$PROFILE" && -n "${DIRECTION:-}" ]] || usage
TRANSFER_ARGS=()
for argument in "$@"; do
    if [[ "$argument" == "--dry-run" ]]; then
        DRY_RUN=1
    else
        TRANSFER_ARGS+=("$argument")
    fi
done

ARTIFACT_ID=""
for ((index = 0; index < ${#TRANSFER_ARGS[@]}; index++)); do
    if [[ "${TRANSFER_ARGS[$index]}" == "--artifact-id" ]]; then
        ARTIFACT_ID="${TRANSFER_ARGS[$((index + 1))]:-}"
        break
    fi
done
[[ "$ARTIFACT_ID" =~ ^[a-zA-Z0-9][a-zA-Z0-9._-]{0,79}$ ]] || usage

LAUNCHER_COMMIT="$(git -C "$LAUNCHER_ROOT" rev-parse HEAD)"
set +e
profile_data="$(PYTHONPATH="${LAUNCHER_ROOT}/tools/launcher${PYTHONPATH:+:${PYTHONPATH}}" python3 - "$PROFILE" "$SCRIPT_DIR/profiles" <<'PY'
import sys
from pathlib import Path

import yaml
from common.specdec.cluster_profile import load_cluster_profile

profile = load_cluster_profile(Path(sys.argv[1]).resolve())
config_path = Path(sys.argv[2]).resolve() / f"{profile.name}.yaml"
if not config_path.is_file():
    raise SystemExit(f"no approved transfer configuration for {profile.name}")
config = yaml.safe_load(config_path.read_text())
fields = {"name", "partition", "cpus_per_task", "gpus_per_node", "walltime"}
if not isinstance(config, dict) or set(config) != fields or config["name"] != profile.name:
    raise SystemExit("invalid transfer configuration")
partition = config["partition"]
if partition is None:
    raise SystemExit(f"no approved transfer partition for {profile.name}")
if not isinstance(partition, str) or not partition:
    raise SystemExit("invalid transfer partition")
cpu = config["cpus_per_task"]
gpu = config["gpus_per_node"]
if (cpu is None) == (gpu is None):
    raise SystemExit("transfer configuration must select exactly one CPU or GPU resource")
for name, value in (("cpus_per_task", cpu), ("gpus_per_node", gpu)):
    if value is not None and (not isinstance(value, int) or isinstance(value, bool) or value < 1):
        raise SystemExit(f"invalid {name}")
walltime = config["walltime"]
if not isinstance(walltime, str) or not walltime:
    raise SystemExit("invalid transfer walltime")
print(profile.account)
print(profile.durable_root)
print(partition)
print(cpu or "")
print(gpu or "")
print(walltime)
PY
)"
profile_status=$?
set -e
[[ "$profile_status" -eq 0 ]] || exit 2
profile_values=()
while IFS= read -r value; do profile_values+=("$value"); done <<<"$profile_data"
(( ${#profile_values[@]} == 6 )) || { echo "invalid transfer profile data" >&2; exit 2; }
ACCOUNT="${profile_values[0]}"
DURABLE_ROOT="${profile_values[1]}"
TRANSFER_PARTITION="${profile_values[2]}"
CPUS_PER_TASK="${profile_values[3]}"
GPUS_PER_NODE="${profile_values[4]}"
WALLTIME="${profile_values[5]}"
python3 "$SCRIPT_DIR/bundle_manifest.py" verify-launcher \
    --root "$LAUNCHER_ROOT" --expected-commit "$LAUNCHER_COMMIT"

LOG_DIR="${DURABLE_ROOT%/}/transfers/logs"
mkdir -p "$LOG_DIR"
SBATCH_ARGS=(
    "--account=$ACCOUNT"
    "--partition=$TRANSFER_PARTITION"
    --nodes=1
    --ntasks=1
    "--time=$WALLTIME"
    "--job-name=specdec-transfer-${ARTIFACT_ID:0:48}"
    "--output=${LOG_DIR}/%x-%j.out"
)
if [[ -n "$CPUS_PER_TASK" ]]; then
    SBATCH_ARGS+=("--cpus-per-task=$CPUS_PER_TASK")
else
    SBATCH_ARGS+=("--gpus-per-node=$GPUS_PER_NODE")
fi
TRANSFER_SCRIPT="${SCRIPT_DIR}/${DIRECTION}_bundle.sh"
INNER_ARGS=(
    --cluster-profile "$PROFILE"
    "--launcher-commit=$LAUNCHER_COMMIT"
    "${TRANSFER_ARGS[@]}"
)

sbatch --test-only "${SBATCH_ARGS[@]}" "$TRANSFER_SCRIPT" "${INNER_ARGS[@]}"
[[ "$DRY_RUN" -eq 0 ]] || exit 0
submitted="$(sbatch --parsable "${SBATCH_ARGS[@]}" "$TRANSFER_SCRIPT" "${INNER_ARGS[@]}")"
job_id="${submitted%%;*}"
[[ "$job_id" =~ ^[0-9]+$ ]] || { echo "scheduler did not confirm transfer submission" >&2; exit 1; }
printf '%s\n' "$job_id"
