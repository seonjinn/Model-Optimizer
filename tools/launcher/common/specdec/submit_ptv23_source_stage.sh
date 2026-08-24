#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUNNER="$SCRIPT_DIR/run_ptv23_source_stage.sbatch"
SOURCE_MANIFEST=""
DURABLE_ROOT=""
LOCAL_SOURCE_ROOT=""
LOCAL_PROJECTION_RECEIPT=""
ACCOUNT=""
PARTITION=""
WALLTIME=""
DRY_RUN=0

usage() {
    echo "usage: $0 --manifest PATH --durable-root PATH --account NAME --partition NAME --time HH:MM:SS [--local-source-root PATH --local-projection-receipt PATH] [--dry-run]" >&2
    exit 2
}

while (( $# )); do
    case "$1" in
        --manifest) SOURCE_MANIFEST="${2:-}"; shift 2 ;;
        --durable-root) DURABLE_ROOT="${2:-}"; shift 2 ;;
        --local-source-root) LOCAL_SOURCE_ROOT="${2:-}"; shift 2 ;;
        --local-projection-receipt) LOCAL_PROJECTION_RECEIPT="${2:-}"; shift 2 ;;
        --account) ACCOUNT="${2:-}"; shift 2 ;;
        --partition) PARTITION="${2:-}"; shift 2 ;;
        --time) WALLTIME="${2:-}"; shift 2 ;;
        --dry-run) DRY_RUN=1; shift ;;
        *) usage ;;
    esac
done
for value in SOURCE_MANIFEST DURABLE_ROOT ACCOUNT PARTITION WALLTIME; do
    [[ -n "${!value}" ]] || usage
done
[[ "$SOURCE_MANIFEST" == /* && "$DURABLE_ROOT" == /* ]] || usage
[[ -z "$LOCAL_SOURCE_ROOT" || "$LOCAL_SOURCE_ROOT" == /* ]] || usage
[[ -z "$LOCAL_PROJECTION_RECEIPT" || "$LOCAL_PROJECTION_RECEIPT" == /* ]] || usage
[[ -z "$LOCAL_PROJECTION_RECEIPT" || -n "$LOCAL_SOURCE_ROOT" ]] || usage
[[ "$WALLTIME" =~ ^[0-9]{2,3}:[0-5][0-9]:[0-5][0-9]$ ]] || usage
for value in "$SOURCE_MANIFEST" "$DURABLE_ROOT" "$LOCAL_SOURCE_ROOT" \
    "$LOCAL_PROJECTION_RECEIPT" "$ACCOUNT" "$PARTITION"; do
    [[ "$value" != *","* && "$value" != *$'\n'* ]] || {
        echo "submission values must not contain commas or newlines" >&2
        exit 2
    }
done
[[ -f "$SOURCE_MANIFEST" && -x "$RUNNER" ]] || {
    echo "source manifest and executable runner are required" >&2
    exit 2
}

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
[[ "$SOURCE_COMMIT" =~ ^[0-9a-f]{40}$ ]] || {
    echo "source checkout is not on an exact commit" >&2
    exit 2
}
SOURCE_MANIFEST="$(python3 -c 'from pathlib import Path; import sys; print(Path(sys.argv[1]).resolve(strict=True))' "$SOURCE_MANIFEST")"
DURABLE_ROOT="$(python3 -c 'from pathlib import Path; import sys; print(Path(sys.argv[1]).resolve(strict=False))' "$DURABLE_ROOT")"
if [[ -n "$LOCAL_SOURCE_ROOT" ]]; then
    LOCAL_SOURCE_ROOT="$(python3 -c 'from pathlib import Path; import sys; print(Path(sys.argv[1]).resolve(strict=True))' "$LOCAL_SOURCE_ROOT")"
fi
if [[ -n "$LOCAL_PROJECTION_RECEIPT" ]]; then
    LOCAL_PROJECTION_RECEIPT="$(python3 -c 'from pathlib import Path; import sys; print(Path(sys.argv[1]).resolve(strict=True))' "$LOCAL_PROJECTION_RECEIPT")"
fi
SOURCE_MANIFEST_SHA256="$(PYTHONPATH="$REPO_ROOT/examples/dataset${PYTHONPATH:+:$PYTHONPATH}" python3 - "$SOURCE_MANIFEST" <<'PY'
import sys
from pathlib import Path

from stage_ptv23_sources import load_source_inventory

print(load_source_inventory(Path(sys.argv[1])).manifest_sha256)
PY
)"
[[ "$SOURCE_MANIFEST_SHA256" =~ ^[0-9a-f]{64}$ ]] || {
    echo "source manifest canonical SHA-256 is invalid" >&2
    exit 2
}
LOG_DIR="$DURABLE_ROOT/logs/$SOURCE_MANIFEST_SHA256"
mkdir -p "$LOG_DIR"

export_values="ALL,REPO_ROOT=$REPO_ROOT,SOURCE_COMMIT=$SOURCE_COMMIT,SOURCE_MANIFEST=$SOURCE_MANIFEST,SOURCE_MANIFEST_SHA256=$SOURCE_MANIFEST_SHA256,DURABLE_ROOT=$DURABLE_ROOT,LOCAL_SOURCE_ROOT=$LOCAL_SOURCE_ROOT,LOCAL_PROJECTION_RECEIPT=$LOCAL_PROJECTION_RECEIPT"
args=(
    --account="$ACCOUNT"
    --partition="$PARTITION"
    --nodes=1
    --ntasks=1
    --cpus-per-task=96
    --mem=0
    --time="$WALLTIME"
    --job-name="ptv23-src-${SOURCE_MANIFEST_SHA256:0:12}"
    --comment="ptv23-source:$SOURCE_MANIFEST_SHA256"
    --output="$LOG_DIR/%x-%j.out"
    --error="$LOG_DIR/%x-%j.err"
    --export="$export_values"
)

sbatch --test-only "${args[@]}" "$RUNNER" >/dev/null
if (( DRY_RUN )); then
    echo "test-only passed"
    exit 0
fi
job_id="$(sbatch --parsable "${args[@]}" "$RUNNER")"
job_id="${job_id%%;*}"
[[ "$job_id" =~ ^[0-9]+$ ]] || {
    echo "scheduler did not confirm source-staging submission" >&2
    exit 1
}
echo "$job_id"
