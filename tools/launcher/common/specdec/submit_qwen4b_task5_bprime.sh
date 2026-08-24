#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUNNER="$SCRIPT_DIR/run_qwen4b_task5_bprime.sbatch"
SOURCE_INVENTORY="" TOKENIZER_RECEIPT="" BASELINE_AUDIT="" HELD_OUT_UUIDS=""
POLICY="" OUTPUT_DIR="" IMAGE_PATH="" IMAGE_SHA256="" ACCOUNT="" PARTITION="" WALLTIME="" DRY_RUN=0

usage() {
    echo "usage: $0 --source-inventory PATH --tokenizer-receipt PATH --baseline-audit PATH --held-out-uuids PATH --policy PATH --output-dir PATH --image PATH --image-sha256 SHA256 --account NAME --partition NAME --time HH:MM:SS [--dry-run]" >&2
    exit 2
}
while (( $# )); do
    case "$1" in
        --source-inventory) SOURCE_INVENTORY="${2:-}"; shift 2 ;;
        --tokenizer-receipt) TOKENIZER_RECEIPT="${2:-}"; shift 2 ;;
        --baseline-audit) BASELINE_AUDIT="${2:-}"; shift 2 ;;
        --held-out-uuids) HELD_OUT_UUIDS="${2:-}"; shift 2 ;;
        --policy) POLICY="${2:-}"; shift 2 ;;
        --output-dir) OUTPUT_DIR="${2:-}"; shift 2 ;;
        --image) IMAGE_PATH="${2:-}"; shift 2 ;;
        --image-sha256) IMAGE_SHA256="${2:-}"; shift 2 ;;
        --account) ACCOUNT="${2:-}"; shift 2 ;;
        --partition) PARTITION="${2:-}"; shift 2 ;;
        --time) WALLTIME="${2:-}"; shift 2 ;;
        --dry-run) DRY_RUN=1; shift ;;
        *) usage ;;
    esac
done
for name in SOURCE_INVENTORY TOKENIZER_RECEIPT BASELINE_AUDIT HELD_OUT_UUIDS POLICY OUTPUT_DIR IMAGE_PATH IMAGE_SHA256 ACCOUNT PARTITION WALLTIME; do
    [[ -n "${!name}" ]] || usage
done
for path in "$SOURCE_INVENTORY" "$TOKENIZER_RECEIPT" "$BASELINE_AUDIT" \
    "$HELD_OUT_UUIDS" "$POLICY" "$OUTPUT_DIR" "$IMAGE_PATH"; do
    [[ "$path" == /* ]] || usage
done
[[ "$IMAGE_SHA256" =~ ^[0-9a-f]{64}$ ]] || usage
[[ "$WALLTIME" =~ ^[0-9]{2,3}:[0-5][0-9]:[0-5][0-9]$ ]] || usage
for value in "$SOURCE_INVENTORY" "$TOKENIZER_RECEIPT" "$BASELINE_AUDIT" \
    "$HELD_OUT_UUIDS" "$POLICY" "$OUTPUT_DIR" "$ACCOUNT" "$PARTITION"; do
    [[ "$value" != *","* && "$value" != *$'\n'* ]] || exit 2
done
for path in "$SOURCE_INVENTORY" "$TOKENIZER_RECEIPT" "$BASELINE_AUDIT" \
    "$HELD_OUT_UUIDS" "$POLICY" "$IMAGE_PATH"; do
    [[ -f "$path" && ! -L "$path" ]] || usage
done
[[ ! -e "$OUTPUT_DIR" && ! -L "$OUTPUT_DIR" && -x "$RUNNER" ]] || usage

REPO_ROOT="$(git -C "$SCRIPT_DIR" rev-parse --show-toplevel)"
[[ -z "$(git -C "$REPO_ROOT" status --porcelain)" ]] || exit 2
git -C "$REPO_ROOT" pull --ff-only
[[ -z "$(git -C "$REPO_ROOT" status --porcelain)" ]] || exit 2
SOURCE_COMMIT="$(git -C "$REPO_ROOT" rev-parse HEAD)"
for name in SOURCE_INVENTORY TOKENIZER_RECEIPT BASELINE_AUDIT HELD_OUT_UUIDS POLICY; do
    printf -v "$name" '%s' "$(python3 -c 'from pathlib import Path; import sys; print(Path(sys.argv[1]).resolve(strict=True))' "${!name}")"
done
OUTPUT_DIR="$(python3 -c 'from pathlib import Path; import sys; print(Path(sys.argv[1]).resolve(strict=False))' "$OUTPUT_DIR")"
TOKENIZER_RECEIPT_SHA256="$(sha256sum "$TOKENIZER_RECEIPT" | cut -d' ' -f1)"
BASELINE_AUDIT_SHA256="$(sha256sum "$BASELINE_AUDIT" | cut -d' ' -f1)"
HELD_OUT_FILE_SHA256="$(sha256sum "$HELD_OUT_UUIDS" | cut -d' ' -f1)"
POLICY_FILE_SHA256="$(sha256sum "$POLICY" | cut -d' ' -f1)"
for digest in "$TOKENIZER_RECEIPT_SHA256" "$BASELINE_AUDIT_SHA256" \
    "$HELD_OUT_FILE_SHA256" "$POLICY_FILE_SHA256"; do
    [[ "$digest" =~ ^[0-9a-f]{64}$ ]] || exit 2
done
LOG_DIR="$OUTPUT_DIR-logs"
mkdir -p "$LOG_DIR"
exports="ALL,REPO_ROOT=$REPO_ROOT,SOURCE_COMMIT=$SOURCE_COMMIT,SOURCE_INVENTORY=$SOURCE_INVENTORY,TOKENIZER_RECEIPT=$TOKENIZER_RECEIPT,TOKENIZER_RECEIPT_SHA256=$TOKENIZER_RECEIPT_SHA256,BASELINE_AUDIT=$BASELINE_AUDIT,BASELINE_AUDIT_SHA256=$BASELINE_AUDIT_SHA256,HELD_OUT_UUIDS=$HELD_OUT_UUIDS,HELD_OUT_FILE_SHA256=$HELD_OUT_FILE_SHA256,POLICY=$POLICY,POLICY_FILE_SHA256=$POLICY_FILE_SHA256,OUTPUT_DIR=$OUTPUT_DIR,IMAGE_PATH=$IMAGE_PATH,IMAGE_SHA256=$IMAGE_SHA256"
args=(--account="$ACCOUNT" --partition="$PARTITION" --nodes=1 --ntasks=1 --cpus-per-task=96
    --mem=0 --time="$WALLTIME" --job-name="q4-task5-bprime" --comment="q4-task5-bprime:$SOURCE_COMMIT"
    --output="$LOG_DIR/%x-%j.out" --error="$LOG_DIR/%x-%j.err" --export="$exports")
sbatch --test-only "${args[@]}" "$RUNNER" >/dev/null
if (( DRY_RUN )); then echo "test-only passed"; exit 0; fi
job_id="$(sbatch --parsable "${args[@]}" "$RUNNER")"; job_id="${job_id%%;*}"
[[ "$job_id" =~ ^[0-9]+$ ]] || exit 1
echo "$job_id"
