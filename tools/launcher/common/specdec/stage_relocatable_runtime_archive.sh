#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Build one verified runtime archive on scratch from an existing Lustre venv.
set -euo pipefail

SOURCE_RUNTIME=""
OUTPUT_ARCHIVE=""
SCRATCH_ROOT="/raid/scratch"
MODE="submit"

usage() {
    echo "usage: $0 --source-runtime /lustre/... --output-archive /lustre/... [--scratch-root /raid/scratch/...]" >&2
    exit 2
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --source-runtime) SOURCE_RUNTIME="$2"; shift 2 ;;
        --output-archive) OUTPUT_ARCHIVE="$2"; shift 2 ;;
        --scratch-root) SCRATCH_ROOT="$2"; shift 2 ;;
        --run-stage) MODE="run"; shift ;;
        *) usage ;;
    esac
done

[[ "$SOURCE_RUNTIME" == /lustre/* && "$OUTPUT_ARCHIVE" == /lustre/* && "$SCRATCH_ROOT" == /raid/scratch/* ]] || usage

if [[ "$MODE" == "submit" ]]; then
    [[ "${BASH_SOURCE[0]}" == /home/* ]] || { echo "staging script must run from /home source" >&2; exit 2; }
    sbatch --test-only --nodes=1 --ntasks-per-node=1 --segment=1 "$0" --run-stage --source-runtime "$SOURCE_RUNTIME" --output-archive "$OUTPUT_ARCHIVE" --scratch-root "$SCRATCH_ROOT"
    exit 0
fi

work_root="${SCRATCH_ROOT}/${SLURM_JOB_ID:?SLURM_JOB_ID is required}/runtime-archive"
archive="${work_root}/runtime.tar.zst"
checksum="${archive}.sha256"
provenance="${archive}.provenance.json"
mkdir -p "$work_root" "$(dirname "$OUTPUT_ARCHIVE")"
tar --dereference --create --use-compress-program=zstd --file="$archive" -C "$(dirname "$SOURCE_RUNTIME")" "$(basename "$SOURCE_RUNTIME")"
tar --list --use-compress-program=zstd --file="$archive" | grep -q '/bin/activate$'
sha256sum "$archive" >"$checksum"
printf '{"source_runtime":"%s","sha256":"%s"}\n' "$SOURCE_RUNTIME" "$(cut -d' ' -f1 "$checksum")" >"$provenance"
temporary="${OUTPUT_ARCHIVE}.partial-${SLURM_JOB_ID}"
cp "$archive" "$temporary"
mv "$temporary" "$OUTPUT_ARCHIVE"
mv "$checksum" "${OUTPUT_ARCHIVE}.sha256"
mv "$provenance" "${OUTPUT_ARCHIVE}.provenance.json"
