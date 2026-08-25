#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUNNER="$SCRIPT_DIR/run_qwen4b_task8_ab_h.sbatch"
MODE="" A_RECEIPT="" A_SHA256="" B_RECEIPT="" B_SHA256="" STUDY_ROOT=""
EXPECTED_COMPLETE="" IMAGE_PATH="" IMAGE_SHA256="" ACCOUNT="" PARTITION="" WALLTIME=""

usage() {
    echo "usage: $0 (--test-only|--submit) --a-tokenized-receipt PATH --a-tokenized-receipt-sha256 SHA256 --b-tokenized-receipt PATH --b-tokenized-receipt-sha256 SHA256 --study-root PATH --image-path PATH --image-sha256 SHA256 --account ACCOUNT --partition cpu_datamover --time HH:MM:SS [--expected-completion-receipt-sha256 SHA256]" >&2
    exit 2
}

while (( $# )); do
    case "$1" in
        --test-only|--submit) [[ -z "$MODE" ]] || usage; MODE="$1"; shift ;;
        --a-tokenized-receipt) A_RECEIPT="${2:-}"; shift 2 ;;
        --a-tokenized-receipt-sha256) A_SHA256="${2:-}"; shift 2 ;;
        --b-tokenized-receipt) B_RECEIPT="${2:-}"; shift 2 ;;
        --b-tokenized-receipt-sha256) B_SHA256="${2:-}"; shift 2 ;;
        --study-root) STUDY_ROOT="${2:-}"; shift 2 ;;
        --expected-completion-receipt-sha256) EXPECTED_COMPLETE="${2:-}"; shift 2 ;;
        --image-path) IMAGE_PATH="${2:-}"; shift 2 ;;
        --image-sha256) IMAGE_SHA256="${2:-}"; shift 2 ;;
        --account) ACCOUNT="${2:-}"; shift 2 ;;
        --partition) PARTITION="${2:-}"; shift 2 ;;
        --time) WALLTIME="${2:-}"; shift 2 ;;
        *) usage ;;
    esac
done
for name in MODE A_RECEIPT A_SHA256 B_RECEIPT B_SHA256 STUDY_ROOT IMAGE_PATH IMAGE_SHA256 \
    ACCOUNT PARTITION WALLTIME; do
    [[ -n "${!name}" ]] || usage
done
[[ "$PARTITION" == cpu_datamover ]] || usage
[[ "$WALLTIME" =~ ^[0-9]{2,3}:[0-5][0-9]:[0-5][0-9]$ ]] || usage
for value in "$A_SHA256" "$B_SHA256" "$IMAGE_SHA256"; do
    [[ "$value" =~ ^[0-9a-f]{64}$ ]] || usage
done
[[ -z "$EXPECTED_COMPLETE" || "$EXPECTED_COMPLETE" =~ ^[0-9a-f]{64}$ ]] || usage
for path in "$A_RECEIPT" "$B_RECEIPT" "$IMAGE_PATH"; do
    [[ "$path" == /* && -f "$path" && ! -L "$path" ]] || usage
done
[[ "$STUDY_ROOT" == /* && -d "$STUDY_ROOT" && ! -L "$STUDY_ROOT" ]] || usage
case "$STUDY_ROOT" in *//*|*/./*|*/../*|*/.|*/..) usage ;; esac
[[ "$A_RECEIPT" == "$STUDY_ROOT/a-repair-tokenized/TOKENIZED.json" ]] || usage
[[ "$B_RECEIPT" == "$STUDY_ROOT/b-balanced-tokenized/TOKENIZED.json" ]] || usage
[[ -x "$RUNNER" ]] || usage
for value in "$A_RECEIPT" "$B_RECEIPT" "$STUDY_ROOT" "$IMAGE_PATH" "$ACCOUNT"; do
    [[ "$value" != *","* && "$value" != *$'\n'* ]] || usage
done
[[ "$(sha256sum "$A_RECEIPT" | cut -d' ' -f1)" == "$A_SHA256" ]] || exit 2
[[ "$(sha256sum "$B_RECEIPT" | cut -d' ' -f1)" == "$B_SHA256" ]] || exit 2
[[ "$(sha256sum "$IMAGE_PATH" | cut -d' ' -f1)" == "$IMAGE_SHA256" ]] || exit 2

complete="$STUDY_ROOT/ptv2-study-exposures/COMPLETE.json"
if [[ -e "$complete" || -L "$complete" ]]; then
    [[ -n "$EXPECTED_COMPLETE" && -f "$complete" && ! -L "$complete" ]] || {
        echo "existing Task8 completion requires its caller-pinned identity" >&2
        exit 2
    }
    [[ "$(python3 - "$complete" <<'PY'
import json
import sys
print(json.load(open(sys.argv[1], encoding="utf-8"))["receipt_sha256"])
PY
)" == "$EXPECTED_COMPLETE" ]] || exit 2
elif [[ -n "$EXPECTED_COMPLETE" ]]; then
    echo "caller-pinned Task8 completion is missing" >&2
    exit 2
fi

durable_prefix="/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sna/modelopt-specdec/dataset-studies"
if [[ "$MODE" == --submit && "$STUDY_ROOT" != "$durable_prefix"/* ]]; then
    echo "Task8 A/B/H production root must be durable Lustre storage" >&2
    exit 2
fi

REPO_ROOT="$(git -C "$SCRIPT_DIR" rev-parse --show-toplevel)"
[[ -z "$(git -C "$REPO_ROOT" status --porcelain)" ]] || exit 2
git -C "$REPO_ROOT" pull --ff-only
[[ -z "$(git -C "$REPO_ROOT" status --porcelain)" ]] || exit 2
SOURCE_COMMIT="$(git -C "$REPO_ROOT" rev-parse HEAD)"
[[ "$SOURCE_COMMIT" =~ ^[0-9a-f]{40}$ ]] || exit 2

job_name="q4-task8-abh-${A_SHA256:0:6}-${B_SHA256:0:6}"
tuple_identity="q4-task8-abh:$A_SHA256:$B_SHA256:${EXPECTED_COMPLETE:-new}"
log_root="$STUDY_ROOT/task8-ab-h-logs"
exports="ALL,REPO_ROOT=$REPO_ROOT,SOURCE_COMMIT=$SOURCE_COMMIT,A_TOKENIZED_RECEIPT=$A_RECEIPT,A_TOKENIZED_RECEIPT_SHA256=$A_SHA256,B_TOKENIZED_RECEIPT=$B_RECEIPT,B_TOKENIZED_RECEIPT_SHA256=$B_SHA256,STUDY_ROOT=$STUDY_ROOT,EXPECTED_COMPLETION_RECEIPT_SHA256=$EXPECTED_COMPLETE,IMAGE_PATH=$IMAGE_PATH,IMAGE_SHA256=$IMAGE_SHA256"
args=(--account="$ACCOUNT" --partition="$PARTITION" --nodes=1 --ntasks=1
    --cpus-per-task=96 --mem=0 --time="$WALLTIME" --chdir="$REPO_ROOT"
    --job-name="$job_name" --comment="$tuple_identity"
    --output="$log_root/%x-%j.out" --error="$log_root/%x-%j.err" --export="$exports")

sbatch --test-only "${args[@]}" "$RUNNER" >/dev/null
if [[ "$MODE" == --test-only ]]; then
    echo "test-only passed"
    exit 0
fi
mkdir -p "$log_root"
existing="$(squeue -h -u "$USER" -n "$job_name" -o "%A|%k" | awk -F'|' -v tuple="$tuple_identity" '$2 == tuple {print $1; exit}')"
[[ -z "$existing" ]] || { echo "existing writer job $existing" >&2; exit 2; }
job_id="$(sbatch --parsable "${args[@]}" "$RUNNER")"
job_id="${job_id%%;*}"
[[ "$job_id" =~ ^[0-9]+$ ]] || exit 1
echo "$job_id"
