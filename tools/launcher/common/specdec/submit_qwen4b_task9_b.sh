#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUNNER="$SCRIPT_DIR/run_qwen4b_task9_b.sbatch"
SOURCE_INVENTORY="" POLICY="" HELD_OUT_UUIDS="" RECEIPT_ROOT=""
ACCOUNT="" PARTITION="" WALLTIME="" DEPENDENCY="" DRY_RUN=0

usage() {
    echo "usage: $0 --source-inventory PATH --policy PATH --held-out-uuids PATH --receipt-root PATH --account NAME --partition NAME --time HH:MM:SS [--dependency JOBID] [--dry-run]" >&2
    exit 2
}

while (( $# )); do
    case "$1" in
        --source-inventory) SOURCE_INVENTORY="${2:-}"; shift 2 ;;
        --policy) POLICY="${2:-}"; shift 2 ;;
        --held-out-uuids) HELD_OUT_UUIDS="${2:-}"; shift 2 ;;
        --receipt-root) RECEIPT_ROOT="${2:-}"; shift 2 ;;
        --account) ACCOUNT="${2:-}"; shift 2 ;;
        --partition) PARTITION="${2:-}"; shift 2 ;;
        --time) WALLTIME="${2:-}"; shift 2 ;;
        --dependency) DEPENDENCY="${2:-}"; shift 2 ;;
        --dry-run) DRY_RUN=1; shift ;;
        *) usage ;;
    esac
done
for value in SOURCE_INVENTORY POLICY HELD_OUT_UUIDS RECEIPT_ROOT ACCOUNT PARTITION WALLTIME; do
    [[ -n "${!value}" ]] || usage
done
for path in "$SOURCE_INVENTORY" "$POLICY" "$HELD_OUT_UUIDS" "$RECEIPT_ROOT"; do
    [[ "$path" == /* ]] || usage
done
[[ "$WALLTIME" =~ ^[0-9]{2,3}:[0-5][0-9]:[0-5][0-9]$ ]] || usage
[[ -z "$DEPENDENCY" || "$DEPENDENCY" =~ ^[0-9]+$ ]] || usage
for value in "$SOURCE_INVENTORY" "$POLICY" "$HELD_OUT_UUIDS" "$RECEIPT_ROOT" \
    "$ACCOUNT" "$PARTITION"; do
    [[ "$value" != *","* && "$value" != *$'\n'* ]] || {
        echo "submission values must not contain commas or newlines" >&2
        exit 2
    }
done
[[ -f "$SOURCE_INVENTORY" && ! -L "$SOURCE_INVENTORY" ]] || usage
[[ -f "$POLICY" && ! -L "$POLICY" && -f "$HELD_OUT_UUIDS" && ! -L "$HELD_OUT_UUIDS" ]] || usage
[[ ! -e "$RECEIPT_ROOT" && ! -L "$RECEIPT_ROOT" && -x "$RUNNER" ]] || usage

REPO_ROOT="$(git -C "$SCRIPT_DIR" rev-parse --show-toplevel)"
[[ -z "$(git -C "$REPO_ROOT" status --porcelain)" ]] || {
    echo "source checkout must be clean before pull" >&2
    exit 2
}
git -C "$REPO_ROOT" pull --ff-only
[[ -z "$(git -C "$REPO_ROOT" status --porcelain)" ]] || {
    echo "source checkout became dirty after pull" >&2
    exit 2
}
SOURCE_COMMIT="$(git -C "$REPO_ROOT" rev-parse HEAD)"
[[ "$SOURCE_COMMIT" =~ ^[0-9a-f]{40}$ ]] || { echo "invalid source commit" >&2; exit 2; }

SOURCE_INVENTORY="$(python3 -c 'from pathlib import Path; import sys; print(Path(sys.argv[1]).resolve(strict=True))' "$SOURCE_INVENTORY")"
POLICY="$(python3 -c 'from pathlib import Path; import sys; print(Path(sys.argv[1]).resolve(strict=True))' "$POLICY")"
HELD_OUT_UUIDS="$(python3 -c 'from pathlib import Path; import sys; print(Path(sys.argv[1]).resolve(strict=True))' "$HELD_OUT_UUIDS")"
RECEIPT_ROOT="$(python3 -c 'from pathlib import Path; import sys; print(Path(sys.argv[1]).resolve(strict=False))' "$RECEIPT_ROOT")"
SOURCE_INVENTORY_SHA256="$(python3 - "$SOURCE_INVENTORY" <<'PY'
import json
import sys
from pathlib import Path
print(json.loads(Path(sys.argv[1]).read_bytes())["source_manifest_sha256"])
PY
)"
POLICY_FILE_SHA256="$(sha256sum "$POLICY" | cut -d' ' -f1)"
HELD_OUT_FILE_SHA256="$(sha256sum "$HELD_OUT_UUIDS" | cut -d' ' -f1)"
for digest in "$SOURCE_INVENTORY_SHA256" "$POLICY_FILE_SHA256" "$HELD_OUT_FILE_SHA256"; do
    [[ "$digest" =~ ^[0-9a-f]{64}$ ]] || { echo "invalid Task9 input identity" >&2; exit 2; }
done
LOG_DIR="$RECEIPT_ROOT-logs"
mkdir -p "$LOG_DIR"

exports="ALL,REPO_ROOT=$REPO_ROOT,SOURCE_COMMIT=$SOURCE_COMMIT,SOURCE_INVENTORY=$SOURCE_INVENTORY,SOURCE_INVENTORY_SHA256=$SOURCE_INVENTORY_SHA256,POLICY=$POLICY,POLICY_FILE_SHA256=$POLICY_FILE_SHA256,HELD_OUT_UUIDS=$HELD_OUT_UUIDS,HELD_OUT_FILE_SHA256=$HELD_OUT_FILE_SHA256,RECEIPT_ROOT=$RECEIPT_ROOT"
args=(
    --account="$ACCOUNT" --partition="$PARTITION" --nodes=1 --ntasks=1
    --cpus-per-task=96 --mem=0 --time="$WALLTIME"
    --job-name="q4-task9-b-${SOURCE_INVENTORY_SHA256:0:12}"
    --comment="q4-task9-b:$SOURCE_INVENTORY_SHA256"
    --output="$LOG_DIR/%x-%j.out" --error="$LOG_DIR/%x-%j.err"
    --export="$exports"
)
[[ -z "$DEPENDENCY" ]] || args+=(--dependency="afterok:$DEPENDENCY")

sbatch --test-only "${args[@]}" "$RUNNER" >/dev/null
if (( DRY_RUN )); then
    echo "test-only passed"
    exit 0
fi
job_id="$(sbatch --parsable "${args[@]}" "$RUNNER")"
job_id="${job_id%%;*}"
[[ "$job_id" =~ ^[0-9]+$ ]] || { echo "scheduler did not confirm Task9 submission" >&2; exit 1; }
echo "$job_id"
