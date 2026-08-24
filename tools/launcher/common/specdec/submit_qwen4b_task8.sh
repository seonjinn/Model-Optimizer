#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUNNER="$SCRIPT_DIR/run_qwen4b_task8.sbatch"
SOURCE_RECEIPT="" SOURCE_RECEIPT_SHA256="" SELECTION_RECEIPT="" SELECTION_RECEIPT_SHA256=""
RESPONSE_RECEIPT="" RESPONSE_RECEIPT_SHA256="" REJECTION_RECEIPT="" REJECTION_RECEIPT_SHA256=""
TOKENIZER_RECEIPT="" TOKENIZER_RECEIPT_SHA256="" ASSISTANT_LOSS_TARGET_SHA256=""
TRAINING_CONFIG_SHA256="" PUBLICATION_ROOT="" ACCOUNT="" PARTITION="" WALLTIME="" DRY_RUN=0
IMAGE_PATH="" IMAGE_SHA256=""

usage() {
    echo "usage: $0 --source-receipt PATH --source-receipt-sha256 SHA256 --selection-receipt PATH --selection-receipt-sha256 SHA256 --response-receipt PATH --response-receipt-sha256 SHA256 --rejection-receipt PATH --rejection-receipt-sha256 SHA256 --tokenizer-receipt PATH --tokenizer-receipt-sha256 SHA256 --assistant-loss-target-sha256 SHA256 --training-config-sha256 SHA256 --image-path PATH --image-sha256 SHA256 --publication-root PATH --account NAME --partition cpu_datamover --time HH:MM:SS [--dry-run]" >&2
    exit 2
}

while (( $# )); do
    case "$1" in
        --source-receipt) SOURCE_RECEIPT="${2:-}"; shift 2 ;;
        --source-receipt-sha256) SOURCE_RECEIPT_SHA256="${2:-}"; shift 2 ;;
        --selection-receipt) SELECTION_RECEIPT="${2:-}"; shift 2 ;;
        --selection-receipt-sha256) SELECTION_RECEIPT_SHA256="${2:-}"; shift 2 ;;
        --response-receipt) RESPONSE_RECEIPT="${2:-}"; shift 2 ;;
        --response-receipt-sha256) RESPONSE_RECEIPT_SHA256="${2:-}"; shift 2 ;;
        --rejection-receipt) REJECTION_RECEIPT="${2:-}"; shift 2 ;;
        --rejection-receipt-sha256) REJECTION_RECEIPT_SHA256="${2:-}"; shift 2 ;;
        --tokenizer-receipt) TOKENIZER_RECEIPT="${2:-}"; shift 2 ;;
        --tokenizer-receipt-sha256) TOKENIZER_RECEIPT_SHA256="${2:-}"; shift 2 ;;
        --assistant-loss-target-sha256) ASSISTANT_LOSS_TARGET_SHA256="${2:-}"; shift 2 ;;
        --training-config-sha256) TRAINING_CONFIG_SHA256="${2:-}"; shift 2 ;;
        --image-path) IMAGE_PATH="${2:-}"; shift 2 ;;
        --image-sha256) IMAGE_SHA256="${2:-}"; shift 2 ;;
        --publication-root) PUBLICATION_ROOT="${2:-}"; shift 2 ;;
        --account) ACCOUNT="${2:-}"; shift 2 ;;
        --partition) PARTITION="${2:-}"; shift 2 ;;
        --time) WALLTIME="${2:-}"; shift 2 ;;
        --dry-run) DRY_RUN=1; shift ;;
        *) usage ;;
    esac
done
for name in SOURCE_RECEIPT SOURCE_RECEIPT_SHA256 SELECTION_RECEIPT SELECTION_RECEIPT_SHA256 \
    RESPONSE_RECEIPT RESPONSE_RECEIPT_SHA256 REJECTION_RECEIPT REJECTION_RECEIPT_SHA256 \
    TOKENIZER_RECEIPT TOKENIZER_RECEIPT_SHA256 ASSISTANT_LOSS_TARGET_SHA256 \
    TRAINING_CONFIG_SHA256 IMAGE_PATH IMAGE_SHA256 PUBLICATION_ROOT ACCOUNT PARTITION WALLTIME; do
    [[ -n "${!name}" ]] || usage
done
[[ "$PARTITION" == cpu_datamover ]] || usage
[[ "$WALLTIME" =~ ^[0-9]{2,3}:[0-5][0-9]:[0-5][0-9]$ ]] || usage
for name in SOURCE_RECEIPT_SHA256 SELECTION_RECEIPT_SHA256 RESPONSE_RECEIPT_SHA256 \
    REJECTION_RECEIPT_SHA256 TOKENIZER_RECEIPT_SHA256 ASSISTANT_LOSS_TARGET_SHA256 \
    TRAINING_CONFIG_SHA256; do
    [[ "${!name}" =~ ^[0-9a-f]{64}$ ]] || usage
done
[[ "$IMAGE_SHA256" =~ ^[0-9a-f]{64}$ ]] || usage
for path in "$SOURCE_RECEIPT" "$SELECTION_RECEIPT" "$RESPONSE_RECEIPT" \
    "$REJECTION_RECEIPT" "$TOKENIZER_RECEIPT"; do
    [[ "$path" == /* && -f "$path" && ! -L "$path" ]] || usage
done
[[ "$PUBLICATION_ROOT" == /* && ! -e "$PUBLICATION_ROOT" && ! -L "$PUBLICATION_ROOT" ]] || usage
durable_prefix="/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sna/modelopt-specdec/dataset-studies"
[[ "$PUBLICATION_ROOT" == "$durable_prefix"/* ]] || {
    echo "Task8 publication must use the durable dataset-study root" >&2
    exit 2
}
[[ "$IMAGE_PATH" == /* && -f "$IMAGE_PATH" && ! -L "$IMAGE_PATH" ]] || usage
[[ -x "$RUNNER" ]] || usage

REPO_ROOT="$(git -C "$SCRIPT_DIR" rev-parse --show-toplevel)"
[[ -z "$(git -C "$REPO_ROOT" status --porcelain)" ]] || {
    echo "source checkout must be clean before pull" >&2
    exit 2
}
git -C "$REPO_ROOT" pull --ff-only
[[ -z "$(git -C "$REPO_ROOT" status --porcelain)" ]] || exit 2
SOURCE_COMMIT="$(git -C "$REPO_ROOT" rev-parse HEAD)"
[[ "$SOURCE_COMMIT" =~ ^[0-9a-f]{40}$ ]] || exit 2

for pair in "$SOURCE_RECEIPT:$SOURCE_RECEIPT_SHA256" \
    "$SELECTION_RECEIPT:$SELECTION_RECEIPT_SHA256" \
    "$RESPONSE_RECEIPT:$RESPONSE_RECEIPT_SHA256" \
    "$REJECTION_RECEIPT:$REJECTION_RECEIPT_SHA256" \
    "$TOKENIZER_RECEIPT:$TOKENIZER_RECEIPT_SHA256"; do
    path="${pair%:*}"
    expected="${pair##*:}"
    [[ "$(sha256sum "$path" | cut -d' ' -f1)" == "$expected" ]] || {
        echo "pinned Task8 receipt SHA-256 mismatch: $path" >&2
        exit 2
    }
done
[[ "$(sha256sum "$IMAGE_PATH" | cut -d' ' -f1)" == "$IMAGE_SHA256" ]] || {
    echo "pinned Task8 container SHA-256 mismatch: $IMAGE_PATH" >&2
    exit 2
}
for value in "$SOURCE_RECEIPT" "$SELECTION_RECEIPT" "$RESPONSE_RECEIPT" \
    "$REJECTION_RECEIPT" "$TOKENIZER_RECEIPT" "$IMAGE_PATH" "$PUBLICATION_ROOT" "$ACCOUNT"; do
    [[ "$value" != *","* && "$value" != *$'\n'* ]] || exit 2
done

LOG_DIR="$PUBLICATION_ROOT-logs"
mkdir -p "$LOG_DIR"
exports="ALL,REPO_ROOT=$REPO_ROOT,SOURCE_COMMIT=$SOURCE_COMMIT,SOURCE_RECEIPT=$SOURCE_RECEIPT,SOURCE_RECEIPT_SHA256=$SOURCE_RECEIPT_SHA256,SELECTION_RECEIPT=$SELECTION_RECEIPT,SELECTION_RECEIPT_SHA256=$SELECTION_RECEIPT_SHA256,RESPONSE_RECEIPT=$RESPONSE_RECEIPT,RESPONSE_RECEIPT_SHA256=$RESPONSE_RECEIPT_SHA256,REJECTION_RECEIPT=$REJECTION_RECEIPT,REJECTION_RECEIPT_SHA256=$REJECTION_RECEIPT_SHA256,TOKENIZER_RECEIPT=$TOKENIZER_RECEIPT,TOKENIZER_RECEIPT_SHA256=$TOKENIZER_RECEIPT_SHA256,ASSISTANT_LOSS_TARGET_SHA256=$ASSISTANT_LOSS_TARGET_SHA256,TRAINING_CONFIG_SHA256=$TRAINING_CONFIG_SHA256,IMAGE_PATH=$IMAGE_PATH,IMAGE_SHA256=$IMAGE_SHA256,PUBLICATION_ROOT=$PUBLICATION_ROOT"
args=(--account="$ACCOUNT" --partition=cpu_datamover --nodes=1 --ntasks=1
    --cpus-per-task=96 --mem=0 --time="$WALLTIME"
    --job-name="q4-task8-${SELECTION_RECEIPT_SHA256:0:12}"
    --comment="q4-task8:$SELECTION_RECEIPT_SHA256"
    --output="$LOG_DIR/%x-%j.out" --error="$LOG_DIR/%x-%j.err" --export="$exports")

sbatch --test-only "${args[@]}" "$RUNNER" >/dev/null
if (( DRY_RUN )); then
    echo "test-only passed"
    exit 0
fi
job_id="$(sbatch --parsable "${args[@]}" "$RUNNER")"
job_id="${job_id%%;*}"
[[ "$job_id" =~ ^[0-9]+$ ]] || exit 1
echo "$job_id"
