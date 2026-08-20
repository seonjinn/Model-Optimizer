#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Build one verified runtime archive on scratch from an existing Lustre venv.
set -euo pipefail

SOURCE_RUNTIME=""
OUTPUT_ARCHIVE=""
SCRATCH_ROOT="/raid/scratch"
MODE="submit"
ACCOUNT="nemotron_n3_post"
PARTITION="batch"

usage() {
    echo "usage: $0 --source-runtime /lustre/... --output-archive /lustre/... [--scratch-root /raid/scratch/...] [--account ACCOUNT] [--partition PARTITION]" >&2
    exit 2
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --source-runtime) SOURCE_RUNTIME="$2"; shift 2 ;;
        --output-archive) OUTPUT_ARCHIVE="$2"; shift 2 ;;
        --scratch-root) SCRATCH_ROOT="$2"; shift 2 ;;
        --account) ACCOUNT="$2"; shift 2 ;;
        --partition) PARTITION="$2"; shift 2 ;;
        --run-stage) MODE="run"; shift ;;
        *) usage ;;
    esac
done

[[ "$SOURCE_RUNTIME" == /lustre/* && "$OUTPUT_ARCHIVE" == /lustre/*.tar.zst && "$SCRATCH_ROOT" == /raid/scratch/* && -n "$ACCOUNT" && -n "$PARTITION" ]] || usage

if [[ "$MODE" == "submit" ]]; then
    [[ "${BASH_SOURCE[0]}" == /home/* ]] || { echo "staging script must run from /home source" >&2; exit 2; }
    if [[ -f "$OUTPUT_ARCHIVE" && -f "${OUTPUT_ARCHIVE}.sha256" && -f "${OUTPUT_ARCHIVE}.provenance.json" ]]; then
        (cd "$(dirname "$OUTPUT_ARCHIVE")" && sha256sum -c "$(basename "${OUTPUT_ARCHIVE}.sha256")")
        echo "runtime archive already verified: $OUTPUT_ARCHIVE"
        exit 0
    fi
    args=(
        --account="$ACCOUNT" --partition="$PARTITION" --nodes=1 --ntasks-per-node=1
        --gpus-per-node=4 --segment=1 --time=00:30:00 --job-name=modelopt-runtime-archive
        --output="${OUTPUT_ARCHIVE}.stage-%j.out"
    )
    command=("$0" --run-stage --source-runtime "$SOURCE_RUNTIME" --output-archive "$OUTPUT_ARCHIVE" --scratch-root "$SCRATCH_ROOT")
    sbatch --test-only "${args[@]}" "${command[@]}"
    submitted="$(sbatch --parsable "${args[@]}" "${command[@]}")"
    echo "${submitted%%;*}"
    exit 0
fi

work_root="${SCRATCH_ROOT}/${SLURM_JOB_ID:?SLURM_JOB_ID is required}/runtime-archive"
archive="${work_root}/runtime.tar.zst"
checksum="${archive}.sha256"
provenance="${archive}.provenance.json"
listing="${work_root}/runtime.list"
mkdir -p "$work_root" "$(dirname "$OUTPUT_ARCHIVE")"
tar --dereference --create --use-compress-program=zstd --file="$archive" -C "$SOURCE_RUNTIME" .
tar --list --use-compress-program=zstd --file="$archive" >"$listing"
grep -q '/bin/activate$' "$listing"
archive_sha="$(sha256sum "$archive" | cut -d' ' -f1)"
printf '%s  %s\n' "$archive_sha" "$(basename "$OUTPUT_ARCHIVE")" >"$checksum"
printf '{"source_runtime":"%s","sha256":"%s"}\n' "$SOURCE_RUNTIME" "$archive_sha" >"$provenance"
temporary="${OUTPUT_ARCHIVE}.partial-${SLURM_JOB_ID}"
cp "$archive" "$temporary"
mv "$temporary" "$OUTPUT_ARCHIVE"
mv "$checksum" "${OUTPUT_ARCHIVE}.sha256"
mv "$provenance" "${OUTPUT_ARCHIVE}.provenance.json"
