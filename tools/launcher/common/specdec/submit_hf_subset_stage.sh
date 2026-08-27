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

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LAUNCHER_ROOT="${DRAFTER_LAUNCHER_ROOT:-$(cd "$SCRIPT_DIR/../.." && pwd)}"
RUNNER="$SCRIPT_DIR/run_hf_subset_stage.sbatch"
PROFILE=""
READINESS=""
PLAN=""
OUTPUT_ROOT=""
DEPENDENCY=""
DRY_RUN=0

usage() {
    echo "usage: $0 --profile PATH --readiness PATH --plan PATH --output-root PATH [--dependency JOBID] [--dry-run]" >&2
    exit 2
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --profile) PROFILE="$2"; shift 2 ;;
        --readiness) READINESS="$2"; shift 2 ;;
        --plan) PLAN="$2"; shift 2 ;;
        --output-root) OUTPUT_ROOT="$2"; shift 2 ;;
        --dependency) DEPENDENCY="$2"; shift 2 ;;
        --dry-run) DRY_RUN=1; shift ;;
        *) usage ;;
    esac
done
[[ -f "$PROFILE" && -f "$READINESS" && -f "$PLAN" && "$OUTPUT_ROOT" == /lustre/* ]] || usage
[[ -z "$DEPENDENCY" || "$DEPENDENCY" =~ ^[0-9]+$ ]] || usage

mapfile -t identity < <(PYTHONPATH="$LAUNCHER_ROOT${PYTHONPATH:+:$PYTHONPATH}" python3 - "$PROFILE" "$READINESS" "$PLAN" "$OUTPUT_ROOT" <<'PY'
import json
import sys
from pathlib import Path

from common.specdec.cluster_profile import load_cluster_profile, scheduler_gpu_args
from common.specdec.qwen4b_study_manifest import validate_readiness_receipt
from common.specdec.stage_hf_subset import load_subset_plan

profile = load_cluster_profile(Path(sys.argv[1]).resolve())
readiness_path = Path(sys.argv[2]).resolve()
receipt = json.loads(readiness_path.read_text())
validate_readiness_receipt(
    receipt,
    profile=profile.name,
    account=profile.account,
    partition=profile.partition,
    scratch_root=Path(receipt.get("scratch_root", "")),
)
plan_path = Path(sys.argv[3]).resolve()
plan = load_subset_plan(plan_path)
output = Path(sys.argv[4]).resolve()
if not output.is_relative_to(profile.durable_root):
    raise ValueError("output root must be under profile durable root")
print(profile.account)
print(profile.partition)
print(profile.walltime)
print(plan_path)
print(plan["plan_sha256"])
print(plan["container"]["path"])
print(plan["container"]["sha256"])
print(readiness_path)
print(Path(sys.argv[1]).resolve())
gpu_args = scheduler_gpu_args(profile)
print(gpu_args[0] if gpu_args else "")
PY
)
(( ${#identity[@]} == 10 )) || { echo "invalid staging identity" >&2; exit 2; }

ACCOUNT="${identity[0]}"
PARTITION="${identity[1]}"
WALLTIME="${identity[2]}"
PLAN="${identity[3]}"
PLAN_SHA256="${identity[4]}"
IMAGE="${identity[5]}"
IMAGE_SHA256="${identity[6]}"
READINESS="${identity[7]}"
PROFILE="${identity[8]}"
GPU_ARG="${identity[9]}"
REPO_ROOT="$(git -C "$LAUNCHER_ROOT" rev-parse --show-toplevel)"
SOURCE_COMMIT="$(git -C "$REPO_ROOT" rev-parse HEAD)"
[[ -z "$(git -C "$REPO_ROOT" status --porcelain)" ]] || { echo "source checkout is dirty" >&2; exit 2; }
READINESS_SHA256="$(sha256sum "$READINESS" | cut -d' ' -f1)"
PROFILE_SHA256="$(sha256sum "$PROFILE" | cut -d' ' -f1)"
LOG_DIR="$(dirname "$OUTPUT_ROOT")/logs"
mkdir -p "$LOG_DIR"

export_values="ALL,REPO_ROOT=$REPO_ROOT,SOURCE_COMMIT=$SOURCE_COMMIT,PROFILE=$PROFILE,PROFILE_SHA256=$PROFILE_SHA256,PLAN_PATH=$PLAN,PLAN_SHA256=$PLAN_SHA256,IMAGE=$IMAGE,IMAGE_SHA256=$IMAGE_SHA256,OUTPUT_ROOT=$OUTPUT_ROOT,READINESS_RECEIPT=$READINESS,READINESS_RECEIPT_SHA256=$READINESS_SHA256"
args=(
    --account="$ACCOUNT" --partition="$PARTITION" --job-name=qwen4b-ptv3-stage
    --nodes=1 --ntasks=1 --cpus-per-task=16 --mem=128G --time="$WALLTIME"
    --output="$LOG_DIR/%x-%j.out" --error="$LOG_DIR/%x-%j.err"
    --export="$export_values"
)
[[ -z "$GPU_ARG" ]] || args+=("$GPU_ARG")
[[ -z "$DEPENDENCY" ]] || args+=(--dependency="afterok:$DEPENDENCY")
sbatch --test-only "${args[@]}" "$RUNNER" >/dev/null
[[ "$DRY_RUN" -eq 0 ]] || { echo "test-only passed"; exit 0; }
job_id="$(sbatch --parsable "${args[@]}" "$RUNNER")"
job_id="${job_id%%;*}"
[[ "$job_id" =~ ^[0-9]+$ ]] || { echo "scheduler did not confirm staging submission" >&2; exit 1; }
echo "$job_id"
