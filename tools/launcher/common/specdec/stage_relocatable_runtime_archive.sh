#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Build one verified runtime archive on scratch from an existing Lustre venv.
set -euo pipefail

# Shared parallel storage is mounted at /lustre on OCI-HSG, Ptyche and Lyris
# and at /scratch/fsw on AWS-CMH and OCI-AGA, so requiring one prefix refuses
# to run on two of the five clusters. Name the storage a durable artifact must
# NOT live on instead -- node-local scratch that vanishes with the job, and the
# NFS home the MARS guidance reserves for source.
is_durable_path() {
    case "${1:-}" in
        /home/*|/raid/*|/tmp/*|/var/*|/cm/*|/dev/shm/*) return 1 ;;
        /*) return 0 ;;
        *) return 1 ;;
    esac
}

SOURCE_RUNTIME=""
OUTPUT_ARCHIVE=""
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LAUNCHER_ROOT="${DRAFTER_LAUNCHER_ROOT:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"
SCRATCH_ROOT="/raid/scratch/${USER}"
MODE="submit"
ACCOUNT="nemotron_n3_post"
PARTITION="batch"
CLUSTER_PROFILE=""
READINESS_RECEIPT=""

usage() {
    echo "usage: $0 --source-runtime /lustre/... --output-archive /lustre/... [--cluster-profile PATH --readiness-receipt PATH] [--scratch-root PATH] [--account ACCOUNT] [--partition PARTITION]" >&2
    exit 2
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --source-runtime) SOURCE_RUNTIME="$2"; shift 2 ;;
        --output-archive) OUTPUT_ARCHIVE="$2"; shift 2 ;;
        --scratch-root) SCRATCH_ROOT="$2"; shift 2 ;;
        --account) ACCOUNT="$2"; shift 2 ;;
        --partition) PARTITION="$2"; shift 2 ;;
        --cluster-profile) CLUSTER_PROFILE="$2"; shift 2 ;;
        --readiness-receipt) READINESS_RECEIPT="$2"; shift 2 ;;
        --run-stage) MODE="run"; shift ;;
        *) usage ;;
    esac
done

PROFILE_DURABLE_ROOT=""
PROFILE_GPU_ARGS=(--gpus-per-node=4)
PROFILE_EVAL_NODES=1
PROFILE_EVAL_SEGMENT=1
if [[ -n "$CLUSTER_PROFILE" || -n "$READINESS_RECEIPT" ]]; then
    [[ -n "$CLUSTER_PROFILE" && -n "$READINESS_RECEIPT" ]] || usage
    mapfile -t profile_values < <(PYTHONPATH="${LAUNCHER_ROOT}${PYTHONPATH:+:${PYTHONPATH}}" python3 - \
        "$CLUSTER_PROFILE" "$READINESS_RECEIPT" <<'PY'
import json
import sys
from pathlib import Path
from common.specdec.cluster_profile import load_cluster_profile, scheduler_gpu_args, validate_scratch_root
profile = load_cluster_profile(Path(sys.argv[1]).resolve())
receipt = json.loads(Path(sys.argv[2]).resolve().read_text())
expected = {"profile": profile.name, "account": profile.account, "partition": profile.partition,
            "pyxis_available": True, "gpu_count": profile.gpus_per_node, "architecture": "aarch64"}
if any(receipt.get(key) != value for key, value in expected.items()):
    raise ValueError("cluster readiness receipt does not match profile")
scratch = Path(receipt.get("scratch_root", ""))
validate_scratch_root(profile, scratch)
print(profile.account)
print(profile.partition)
print(profile.durable_root)
print(scratch)
print(profile.evaluation_nodes)
print(profile.evaluation_segment)
for argument in scheduler_gpu_args(profile): print(argument)
PY
    )
    (( ${#profile_values[@]} >= 6 )) || { echo "invalid cluster readiness contract" >&2; exit 2; }
    ACCOUNT="${profile_values[0]}"
    PARTITION="${profile_values[1]}"
    PROFILE_DURABLE_ROOT="${profile_values[2]}"
    SCRATCH_ROOT="${profile_values[3]%/}/${USER}"
    PROFILE_EVAL_NODES="${profile_values[4]}"
    PROFILE_EVAL_SEGMENT="${profile_values[5]}"
    PROFILE_GPU_ARGS=("${profile_values[@]:6}")
fi

[[ -n "$ACCOUNT" && -n "$PARTITION" && "$OUTPUT_ARCHIVE" == *.tar.zst ]] || usage
if [[ -n "$PROFILE_DURABLE_ROOT" ]]; then
    [[ "$(realpath -m -- "$SOURCE_RUNTIME")" == "$PROFILE_DURABLE_ROOT"/* ]] || usage
    [[ "$(realpath -m -- "$OUTPUT_ARCHIVE")" == "$PROFILE_DURABLE_ROOT"/* ]] || usage
    PYTHONPATH="${LAUNCHER_ROOT}${PYTHONPATH:+:${PYTHONPATH}}" python3 - "$CLUSTER_PROFILE" "$SCRATCH_ROOT" <<'PY'
import sys
from pathlib import Path
from common.specdec.cluster_profile import load_cluster_profile, validate_scratch_root
validate_scratch_root(load_cluster_profile(Path(sys.argv[1]).resolve()), Path(sys.argv[2]))
PY
else
    is_durable_path "$SOURCE_RUNTIME" && is_durable_path "$OUTPUT_ARCHIVE" \
        && [[ "$OUTPUT_ARCHIVE" == *.tar.zst && "$SCRATCH_ROOT" == /raid/scratch/* ]] || usage
fi

if [[ "$MODE" == "submit" ]]; then
    [[ "${BASH_SOURCE[0]}" == /home/* ]] || { echo "staging script must run from /home source" >&2; exit 2; }
    if [[ -f "$OUTPUT_ARCHIVE" && -f "${OUTPUT_ARCHIVE}.sha256" && -f "${OUTPUT_ARCHIVE}.provenance.json" ]]; then
        (cd "$(dirname "$OUTPUT_ARCHIVE")" && sha256sum -c "$(basename "${OUTPUT_ARCHIVE}.sha256")")
        echo "runtime archive already verified: $OUTPUT_ARCHIVE"
        exit 0
    fi
    if [[ -n "$CLUSTER_PROFILE" ]]; then
        args=(--account="$ACCOUNT" --partition="$PARTITION" --nodes="$PROFILE_EVAL_NODES"
            --ntasks-per-node=1 "${PROFILE_GPU_ARGS[@]}" --segment="$PROFILE_EVAL_SEGMENT")
    else
        args=(--account="$ACCOUNT" --partition="$PARTITION" --nodes=1
            --ntasks-per-node=1 --gpus-per-node=4 --segment=1)
    fi
    args+=(--time=00:30:00 --job-name=modelopt-runtime-archive
        --output="${OUTPUT_ARCHIVE}.stage-%j.out")
    command=("$0" --run-stage --source-runtime "$SOURCE_RUNTIME" --output-archive "$OUTPUT_ARCHIVE" --scratch-root "$SCRATCH_ROOT")
    if [[ -n "$CLUSTER_PROFILE" ]]; then
        args+=("--export=ALL,DRAFTER_LAUNCHER_ROOT=$LAUNCHER_ROOT")
        command+=(--cluster-profile "$CLUSTER_PROFILE" --readiness-receipt "$READINESS_RECEIPT")
    fi
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
tar --create --use-compress-program=zstd --file="$archive" -C "$SOURCE_RUNTIME" .
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
