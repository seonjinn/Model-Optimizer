#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUNNER="$SCRIPT_DIR/run_qwen4b_task9_a.sbatch"
SOURCE_INVENTORY="" BASELINE_AUDIT="" HELD_OUT_UUIDS="" POLICY="" TASK5_MANIFEST=""
TASK5_MANIFEST_SHA256="" OUTPUT_ROOT="" IMAGE_PATH="" IMAGE_SHA256=""
ACCOUNT="" PARTITION="" WALLTIME="" DRY_RUN=0

usage() {
    echo "usage: $0 --source-inventory PATH --baseline-audit PATH --held-out-uuids PATH --policy PATH --task5-manifest PATH --task5-manifest-sha256 SHA256 --output-root PATH --image PATH --image-sha256 SHA256 --account NAME --partition NAME --time HH:MM:SS [--dry-run]" >&2
    exit 2
}
while (( $# )); do
    case "$1" in
        --source-inventory) SOURCE_INVENTORY="${2:-}"; shift 2 ;;
        --baseline-audit) BASELINE_AUDIT="${2:-}"; shift 2 ;;
        --held-out-uuids) HELD_OUT_UUIDS="${2:-}"; shift 2 ;;
        --policy) POLICY="${2:-}"; shift 2 ;;
        --task5-manifest) TASK5_MANIFEST="${2:-}"; shift 2 ;;
        --task5-manifest-sha256) TASK5_MANIFEST_SHA256="${2:-}"; shift 2 ;;
        --output-root) OUTPUT_ROOT="${2:-}"; shift 2 ;;
        --image) IMAGE_PATH="${2:-}"; shift 2 ;;
        --image-sha256) IMAGE_SHA256="${2:-}"; shift 2 ;;
        --account) ACCOUNT="${2:-}"; shift 2 ;;
        --partition) PARTITION="${2:-}"; shift 2 ;;
        --time) WALLTIME="${2:-}"; shift 2 ;;
        --dry-run) DRY_RUN=1; shift ;;
        *) usage ;;
    esac
done
for name in SOURCE_INVENTORY BASELINE_AUDIT HELD_OUT_UUIDS POLICY TASK5_MANIFEST \
    TASK5_MANIFEST_SHA256 OUTPUT_ROOT IMAGE_PATH IMAGE_SHA256 ACCOUNT PARTITION WALLTIME; do
    [[ -n "${!name}" ]] || usage
done
for path in "$SOURCE_INVENTORY" "$BASELINE_AUDIT" "$HELD_OUT_UUIDS" "$POLICY" \
    "$TASK5_MANIFEST" "$OUTPUT_ROOT" "$IMAGE_PATH"; do
    [[ "$path" == /* && "$path" != *","* && "$path" != *$'\n'* ]] || usage
done
for digest in "$TASK5_MANIFEST_SHA256" "$IMAGE_SHA256"; do
    [[ "$digest" =~ ^[0-9a-f]{64}$ ]] || usage
done
[[ "$WALLTIME" =~ ^[0-9]{2,3}:[0-5][0-9]:[0-5][0-9]$ ]] || usage
for path in "$SOURCE_INVENTORY" "$BASELINE_AUDIT" "$HELD_OUT_UUIDS" "$POLICY" \
    "$TASK5_MANIFEST" "$IMAGE_PATH"; do
    [[ -f "$path" && ! -L "$path" ]] || usage
done
[[ ! -e "$OUTPUT_ROOT" && ! -L "$OUTPUT_ROOT" && -x "$RUNNER" ]] || usage

REPO_ROOT="$(git -C "$SCRIPT_DIR" rev-parse --show-toplevel)"
[[ -z "$(git -C "$REPO_ROOT" status --porcelain)" ]] || exit 2
git -C "$REPO_ROOT" pull --ff-only
[[ -z "$(git -C "$REPO_ROOT" status --porcelain)" ]] || exit 2
SOURCE_COMMIT="$(git -C "$REPO_ROOT" rev-parse HEAD)"
LOG_ROOT="$OUTPUT_ROOT-logs"
mkdir -p "$LOG_ROOT"
exports="ALL,REPO_ROOT=$REPO_ROOT,SOURCE_COMMIT=$SOURCE_COMMIT,SOURCE_INVENTORY=$SOURCE_INVENTORY,BASELINE_AUDIT=$BASELINE_AUDIT,HELD_OUT_UUIDS=$HELD_OUT_UUIDS,POLICY=$POLICY,TASK5_MANIFEST=$TASK5_MANIFEST,TASK5_MANIFEST_SHA256=$TASK5_MANIFEST_SHA256,OUTPUT_ROOT=$OUTPUT_ROOT,IMAGE_PATH=$IMAGE_PATH,IMAGE_SHA256=$IMAGE_SHA256"
args=(--account="$ACCOUNT" --partition="$PARTITION" --nodes=1 --ntasks=1 --cpus-per-task=96
    --mem=0 --time="$WALLTIME" --job-name=q4-task9-a-repair
    --comment="q4-task9-a:$SOURCE_COMMIT" --output="$LOG_ROOT/%x-%j.out"
    --error="$LOG_ROOT/%x-%j.err" --export="$exports")
sbatch --test-only "${args[@]}" "$RUNNER" >/dev/null
if (( DRY_RUN )); then echo "test-only passed"; exit 0; fi
job_id="$(sbatch --parsable "${args[@]}" "$RUNNER")"; job_id="${job_id%%;*}"
[[ "$job_id" =~ ^[0-9]+$ ]] || exit 1
echo "$job_id"
