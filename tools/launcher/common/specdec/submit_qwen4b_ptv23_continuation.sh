#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail
RUNTIME_ATTESTATION_IMPLEMENTED=0

usage() {
    echo "usage: $0 (--test-only|--submit-canary|--submit-full) --account ACCOUNT --contract PATH --contract-sha256 SHA256 --contract-tool PATH --contract-tool-sha256 SHA256 --training-entrypoint PATH --training-entrypoint-sha256 SHA256 --image PATH --image-sha256 SHA256 --output PATH --checkpoint-output-root PATH [--full-steps N --canary-receipt PATH --canary-receipt-sha256 SHA256 --canary-evidence PATH]" >&2
    exit 2
}

validate_export_value() {
    local name="$1" value="$2"
    if [[ "$value" == *","* || ! "$value" =~ ^[[:print:]]*$ ]]; then
        echo "unsafe SLURM export value: $name" >&2
        exit 2
    fi
}

MODE=""
ACCOUNT=""
CONTRACT=""
CONTRACT_SHA256=""
CONTRACT_TOOL=""
CONTRACT_TOOL_SHA256=""
TRAINING_ENTRYPOINT=""
TRAINING_ENTRYPOINT_SHA256=""
IMAGE=""
IMAGE_SHA256=""
OUTPUT=""
CHECKPOINT_OUTPUT_ROOT=""
CANARY_RECEIPT=""
CANARY_RECEIPT_SHA256=""
CANARY_EVIDENCE=""
FULL_COMPLETION_EVIDENCE=""
FULL_COMPLETION_RECEIPT=""
FULL_STEPS="0"
while (($#)); do
    case "$1" in
        --test-only|--submit-canary|--submit-full) [[ -z "$MODE" ]] || usage; MODE="$1"; shift ;;
        --account) ACCOUNT="$2"; shift 2 ;;
        --contract) CONTRACT="$2"; shift 2 ;;
        --contract-sha256) CONTRACT_SHA256="$2"; shift 2 ;;
        --contract-tool) CONTRACT_TOOL="$2"; shift 2 ;;
        --contract-tool-sha256) CONTRACT_TOOL_SHA256="$2"; shift 2 ;;
        --training-entrypoint) TRAINING_ENTRYPOINT="$2"; shift 2 ;;
        --training-entrypoint-sha256) TRAINING_ENTRYPOINT_SHA256="$2"; shift 2 ;;
        --image) IMAGE="$2"; shift 2 ;;
        --image-sha256) IMAGE_SHA256="$2"; shift 2 ;;
        --output) OUTPUT="$2"; shift 2 ;;
        --checkpoint-output-root) CHECKPOINT_OUTPUT_ROOT="$2"; shift 2 ;;
        --canary-receipt) CANARY_RECEIPT="$2"; shift 2 ;;
        --canary-receipt-sha256) CANARY_RECEIPT_SHA256="$2"; shift 2 ;;
        --canary-evidence) CANARY_EVIDENCE="$2"; shift 2 ;;
        --full-completion-evidence) FULL_COMPLETION_EVIDENCE="$2"; shift 2 ;;
        --full-completion-receipt) FULL_COMPLETION_RECEIPT="$2"; shift 2 ;;
        --full-steps) FULL_STEPS="$2"; shift 2 ;;
        *) usage ;;
    esac
done
[[ -n "$MODE" && -n "$ACCOUNT" && -f "$CONTRACT" && -f "$CONTRACT_TOOL" ]] || usage
[[ -f "$TRAINING_ENTRYPOINT" && -f "$IMAGE" && -n "$OUTPUT" ]] || usage
[[ -n "$CHECKPOINT_OUTPUT_ROOT" ]] || usage
[[ "$CONTRACT_SHA256" =~ ^[0-9a-f]{64}$ && "$IMAGE_SHA256" =~ ^[0-9a-f]{64}$ ]] || usage
[[ "$CONTRACT_TOOL_SHA256" =~ ^[0-9a-f]{64}$ ]] || usage
[[ "$TRAINING_ENTRYPOINT_SHA256" =~ ^[0-9a-f]{64}$ ]] || usage
for name in STAGE FULL_STEPS CONTRACT CONTRACT_SHA256 CONTRACT_TOOL CONTRACT_TOOL_SHA256 \
    TRAINING_ENTRYPOINT TRAINING_ENTRYPOINT_SHA256 IMAGE IMAGE_SHA256 OUTPUT \
    CHECKPOINT_OUTPUT_ROOT CANARY_RECEIPT CANARY_RECEIPT_SHA256 CANARY_EVIDENCE \
    FULL_COMPLETION_EVIDENCE FULL_COMPLETION_RECEIPT; do
    validate_export_value "$name" "${!name:-}"
done
((RUNTIME_ATTESTATION_IMPLEMENTED == 1)) || {
    echo "runtime image attestation is not implemented; submission is fail-closed" >&2
    exit 1
}
printf '%s  %s\n' "$IMAGE_SHA256" "$IMAGE" | sha256sum --check --status
if [[ "$MODE" == "--submit-full" ]]; then
    [[ -f "$CANARY_RECEIPT" && "$CANARY_RECEIPT_SHA256" =~ ^[0-9a-f]{64}$ ]] || usage
    if [[ ! "$FULL_STEPS" =~ ^[0-9]+$ ]] || ((FULL_STEPS <= 20)); then
        usage
    fi
    [[ -n "$FULL_COMPLETION_EVIDENCE" && ! -e "$FULL_COMPLETION_EVIDENCE" ]] || usage
    [[ -n "$FULL_COMPLETION_RECEIPT" && ! -e "$FULL_COMPLETION_RECEIPT" ]] || usage
fi
if [[ "$MODE" == "--submit-canary" ]]; then
    [[ -n "$CANARY_RECEIPT" && ! -e "$CANARY_RECEIPT" ]] || usage
    [[ -n "$CANARY_EVIDENCE" && ! -e "$CANARY_EVIDENCE" ]] || usage
fi
STAGE="canary"
[[ "$MODE" == "--submit-full" ]] && STAGE="full"
RUNNER="$(dirname "$0")/run_qwen4b_ptv23_continuation.sbatch"
exports="STAGE=$STAGE,FULL_STEPS=$FULL_STEPS,CONTRACT=$CONTRACT,CONTRACT_SHA256=$CONTRACT_SHA256,CONTRACT_TOOL=$CONTRACT_TOOL,CONTRACT_TOOL_SHA256=$CONTRACT_TOOL_SHA256,TRAINING_ENTRYPOINT=$TRAINING_ENTRYPOINT,TRAINING_ENTRYPOINT_SHA256=$TRAINING_ENTRYPOINT_SHA256,IMAGE=$IMAGE,IMAGE_SHA256=$IMAGE_SHA256,OUTPUT=$OUTPUT,CHECKPOINT_OUTPUT_ROOT=$CHECKPOINT_OUTPUT_ROOT,CANARY_RECEIPT=$CANARY_RECEIPT,CANARY_RECEIPT_SHA256=$CANARY_RECEIPT_SHA256,CANARY_EVIDENCE=$CANARY_EVIDENCE,FULL_COMPLETION_EVIDENCE=$FULL_COMPLETION_EVIDENCE,FULL_COMPLETION_RECEIPT=$FULL_COMPLETION_RECEIPT"
args=(--account="$ACCOUNT" --partition=batch --nodes=1 --gpus-per-node=4 --export="$exports" --output="$OUTPUT")
sbatch --test-only "${args[@]}" "$RUNNER" >/dev/null
if [[ "$MODE" == "--test-only" ]]; then
    echo "test-only passed"
    exit 0
fi
sbatch --parsable "${args[@]}" "$RUNNER"
