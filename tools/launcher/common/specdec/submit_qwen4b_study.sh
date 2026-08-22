#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

usage() {
    echo "usage: $0 --manifest PATH --index N --profile PATH --readiness PATH --receipt PATH [--canary-receipt PATH] [--assistant-token-target N] --max-steps N --save-steps N" >&2
    exit 2
}

MANIFEST_PATH="" EXPERIMENT_INDEX="" CLUSTER_PROFILE_PATH="" READINESS_RECEIPT=""
SUBMISSION_RECEIPT="" CANARY_RECEIPT="" ASSISTANT_TOKEN_TARGET="" MAX_STEPS="" SAVE_STEPS=""
while (( $# )); do
    case "$1" in
        --manifest) MANIFEST_PATH="${2:-}"; shift 2 ;;
        --index) EXPERIMENT_INDEX="${2:-}"; shift 2 ;;
        --profile) CLUSTER_PROFILE_PATH="${2:-}"; shift 2 ;;
        --readiness) READINESS_RECEIPT="${2:-}"; shift 2 ;;
        --receipt) SUBMISSION_RECEIPT="${2:-}"; shift 2 ;;
        --canary-receipt) CANARY_RECEIPT="${2:-}"; shift 2 ;;
        --assistant-token-target) ASSISTANT_TOKEN_TARGET="${2:-}"; shift 2 ;;
        --max-steps) MAX_STEPS="${2:-}"; shift 2 ;;
        --save-steps) SAVE_STEPS="${2:-}"; shift 2 ;;
        *) usage ;;
    esac
done
for value in MANIFEST_PATH EXPERIMENT_INDEX CLUSTER_PROFILE_PATH READINESS_RECEIPT SUBMISSION_RECEIPT MAX_STEPS SAVE_STEPS; do
    [[ -n "${!value}" ]] || usage
done
[[ "$EXPERIMENT_INDEX" =~ ^[0-9]+$ && "$MAX_STEPS" =~ ^[1-9][0-9]*$ && "$SAVE_STEPS" =~ ^[1-9][0-9]*$ ]] || usage
[[ -z "$ASSISTANT_TOKEN_TARGET" || "$ASSISTANT_TOKEN_TARGET" =~ ^[1-9][0-9]*$ ]] || usage
[[ -f "$MANIFEST_PATH" && -f "$CLUSTER_PROFILE_PATH" && -f "$READINESS_RECEIPT" ]] || {
    echo "manifest, profile, and readiness receipt must exist" >&2
    exit 2
}

LAUNCHER_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
RUNNER="$LAUNCHER_ROOT/common/specdec/run_qwen4b_study_training.sbatch"
[[ -x "$RUNNER" ]] || { echo "study training runner is not executable" >&2; exit 2; }
settings=()
while IFS= read -r line; do
    settings+=("$line")
done < <(PYTHONPATH="$LAUNCHER_ROOT${PYTHONPATH:+:$PYTHONPATH}" python3 - "$MANIFEST_PATH" "$EXPERIMENT_INDEX" "$CLUSTER_PROFILE_PATH" "$READINESS_RECEIPT" "$MAX_STEPS" "$CANARY_RECEIPT" "$ASSISTANT_TOKEN_TARGET" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

from common.specdec.cluster_profile import load_cluster_profile, scheduler_gpu_args
from common.specdec.qwen4b_study_manifest import (
    load_study_manifest,
    readable_study_job_name,
    validate_canary_receipt,
    validate_milestone_receipt,
    validate_training_target,
)

manifest_path = Path(sys.argv[1])
experiment = load_study_manifest(manifest_path)[int(sys.argv[2])]
profile_path = Path(sys.argv[3])
profile = load_cluster_profile(profile_path)
receipt_path = Path(sys.argv[4])
max_steps = int(sys.argv[5])
canary_receipt_path = Path(sys.argv[6]) if sys.argv[6] else None
assistant_token_target = int(sys.argv[7]) if sys.argv[7] else None
if profile.name != "oci-hsg":
    raise SystemExit("Qwen3-4B Phase A submission is currently qualified only for oci-hsg")
if profile.account != "nemotron_sw_post" or experiment.schedule.account != profile.account:
    raise SystemExit("study submission requires nemotron_sw_post")
if experiment.schedule.partition != profile.partition:
    raise SystemExit("study partition/profile mismatch")
validate_canary_receipt(canary_receipt_path, experiment, max_steps)
validate_training_target(experiment, max_steps, assistant_token_target)
previous_milestone_path = None
if assistant_token_target is not None:
    milestone_index = experiment.schedule.assistant_token_milestones.index(
        assistant_token_target
    )
    if milestone_index:
        previous_tokens = experiment.schedule.assistant_token_milestones[milestone_index - 1]
        previous_milestone_path = (
            Path(experiment.inputs.output_root)
            / "control"
            / f"assistant-token-milestone-{previous_tokens}.json"
        )
        validate_milestone_receipt(previous_milestone_path, experiment, previous_tokens)
receipt = json.loads(receipt_path.read_text())
if receipt.get("profile") not in (None, profile.name):
    raise SystemExit("readiness receipt profile mismatch")
for value in (
    experiment.experiment_id,
    readable_study_job_name(experiment)
    + (f"-c{experiment.schedule.canary_steps}" if max_steps == experiment.schedule.canary_steps else f"-m{assistant_token_target // 1_000_000}"),
    profile.account,
    profile.partition,
    profile.walltime,
    profile.modelopt_commit,
    hashlib.sha256(profile_path.read_bytes()).hexdigest(),
    hashlib.sha256(receipt_path.read_bytes()).hexdigest(),
    hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
    hashlib.sha256(canary_receipt_path.read_bytes()).hexdigest()
    if canary_receipt_path is not None
    else "none",
    assistant_token_target or "none",
    previous_milestone_path or "none",
    hashlib.sha256(previous_milestone_path.read_bytes()).hexdigest()
    if previous_milestone_path is not None
    else "none",
    *(scheduler_gpu_args(profile) or ("",)),
):
    print(value)
PY
)
(( ${#settings[@]} == 14 )) || { echo "unable to resolve study submission" >&2; exit 2; }
EXPERIMENT_ID="${settings[0]}"
JOB_NAME="${settings[1]}"
ACCOUNT="${settings[2]}"
PARTITION="${settings[3]}"
WALLTIME="${settings[4]}"
SOURCE_SHA="${settings[5]}"
CLUSTER_PROFILE_SHA256="${settings[6]}"
READINESS_RECEIPT_SHA256="${settings[7]}"
MANIFEST_SHA256="${settings[8]}"
CANARY_RECEIPT_SHA256="${settings[9]}"
ASSISTANT_TOKEN_TARGET="${settings[10]}"
PREVIOUS_MILESTONE_RECEIPT="${settings[11]}"
PREVIOUS_MILESTONE_RECEIPT_SHA256="${settings[12]}"
GPU_ARG="${settings[13]}"
SCRATCH_ROOT="$(PYTHONPATH="$LAUNCHER_ROOT${PYTHONPATH:+:$PYTHONPATH}" python3 - "$CLUSTER_PROFILE_PATH" "$READINESS_RECEIPT" <<'PY'
import json
import sys
from pathlib import Path

from common.specdec.cluster_profile import load_cluster_profile, validate_scratch_root

profile = load_cluster_profile(Path(sys.argv[1]))
receipt = json.loads(Path(sys.argv[2]).read_text())
scratch = Path(receipt["scratch_root"])
validate_scratch_root(profile, scratch)
print(scratch)
PY
)"
readonly EXPERIMENT_ID JOB_NAME ACCOUNT PARTITION WALLTIME SOURCE_SHA CLUSTER_PROFILE_SHA256
readonly READINESS_RECEIPT_SHA256 MANIFEST_SHA256 GPU_ARG SCRATCH_ROOT
readonly CANARY_RECEIPT_SHA256
readonly ASSISTANT_TOKEN_TARGET
readonly PREVIOUS_MILESTONE_RECEIPT PREVIOUS_MILESTONE_RECEIPT_SHA256

export_list="ALL,MANIFEST_PATH=${MANIFEST_PATH},MANIFEST_SHA256=${MANIFEST_SHA256},EXPERIMENT_INDEX=${EXPERIMENT_INDEX},LAUNCHER_ROOT=${LAUNCHER_ROOT},CLUSTER_PROFILE_PATH=${CLUSTER_PROFILE_PATH},CLUSTER_PROFILE_SHA256=${CLUSTER_PROFILE_SHA256},READINESS_RECEIPT=${READINESS_RECEIPT},READINESS_RECEIPT_SHA256=${READINESS_RECEIPT_SHA256},CANARY_RECEIPT=${CANARY_RECEIPT},CANARY_RECEIPT_SHA256=${CANARY_RECEIPT_SHA256},ASSISTANT_TOKEN_TARGET=${ASSISTANT_TOKEN_TARGET},PREVIOUS_MILESTONE_RECEIPT=${PREVIOUS_MILESTONE_RECEIPT},PREVIOUS_MILESTONE_RECEIPT_SHA256=${PREVIOUS_MILESTONE_RECEIPT_SHA256},SCRATCH_ROOT=${SCRATCH_ROOT},MAX_STEPS=${MAX_STEPS},SAVE_STEPS=${SAVE_STEPS},SELF_REQUEUE=1,MAX_REQUEUES=50"
args=(
    --account="$ACCOUNT"
    --partition="$PARTITION"
    --nodes=2
    --ntasks-per-node=1
    --segment=2
    --time="$WALLTIME"
    --job-name="$JOB_NAME"
    --comment="q4b-study:${EXPERIMENT_ID}:${MAX_STEPS}:${SOURCE_SHA}"
    --signal=B:USR1@300
    --requeue
    --export="$export_list"
)
[[ -z "$GPU_ARG" ]] || args+=("$GPU_ARG")

if [[ -e "$SUBMISSION_RECEIPT" || -L "$SUBMISSION_RECEIPT" ]]; then
    [[ -f "$SUBMISSION_RECEIPT" && ! -L "$SUBMISSION_RECEIPT" ]] || {
        echo "submission receipt destination must be absent or a regular file" >&2
        exit 2
    }
fi
receipt_parent="$(dirname "$SUBMISSION_RECEIPT")"
mkdir -p "$receipt_parent"
receipt_probe="$(mktemp "$receipt_parent/.qwen4b-submission-receipt.XXXXXX")"
rm -f "$receipt_probe"
sbatch --test-only "${args[@]}" "$RUNNER" >/dev/null
job_id="$(sbatch --parsable "${args[@]}" "$RUNNER")"
job_id="${job_id%%;*}"
[[ "$job_id" =~ ^[0-9]+$ ]] || { echo "invalid sbatch job ID: $job_id" >&2; exit 1; }
python3 - "$SUBMISSION_RECEIPT" "$job_id" "$EXPERIMENT_ID" "$JOB_NAME" "$MAX_STEPS" "$MANIFEST_PATH" "$MANIFEST_SHA256" "$SOURCE_SHA" "$CANARY_RECEIPT" "$CANARY_RECEIPT_SHA256" "$ASSISTANT_TOKEN_TARGET" "$PREVIOUS_MILESTONE_RECEIPT" "$PREVIOUS_MILESTONE_RECEIPT_SHA256" <<'PY'
import json
import os
import sys
from pathlib import Path

path = Path(sys.argv[1])
record = {
    "job_id": sys.argv[2],
    "experiment_id": sys.argv[3],
    "job_name": sys.argv[4],
    "max_steps": int(sys.argv[5]),
    "manifest_path": sys.argv[6],
    "manifest_sha256": sys.argv[7],
    "source_sha": sys.argv[8],
    "canary_receipt": sys.argv[9] or None,
    "canary_receipt_sha256": sys.argv[10],
    "assistant_token_target": None if sys.argv[11] == "none" else int(sys.argv[11]),
    "previous_milestone_receipt": None if sys.argv[12] == "none" else sys.argv[12],
    "previous_milestone_receipt_sha256": sys.argv[13],
    "test_only_passed": True,
}
temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
temporary.write_text(json.dumps(record, sort_keys=True) + "\n")
os.replace(temporary, path)
PY
echo "$job_id"
