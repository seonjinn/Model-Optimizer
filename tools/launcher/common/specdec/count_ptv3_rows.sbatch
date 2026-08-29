#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Count PTv3 rows from the staged shards, with a node-local pyarrow.
#
# The pinned vLLM image does not carry pyarrow -- it is not a core vLLM
# dependency -- and the converted repos are stored as parquet, so the counter
# needs an arrow reader from somewhere. Build it in /raid/scratch: node-local
# and ephemeral, rather than a venv on Lustre that every later job would
# inherit and none would own.
#
# Usage:
#   STUDY_ROOT=<lustre study dir> sbatch --partition=<cpu> --account=<acct> \
#       tools/launcher/common/specdec/count_ptv3_rows.sbatch

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=96
#SBATCH --mem=0

set -euo pipefail

[[ -n "${STUDY_ROOT:-}" ]] || { echo "missing required environment: STUDY_ROOT" >&2; exit 2; }
[[ "$STUDY_ROOT" == /lustre/* ]] || { echo "staged data lives on Lustre" >&2; exit 2; }
[[ "${SLURM_JOB_ID:-}" =~ ^[0-9]+$ ]] || { echo "invalid Slurm job identity" >&2; exit 2; }

# Locate the counter, which ships beside this file in the repo so the study is
# reproducible from a clone rather than from whatever sits in a home directory.
#
# BASH_SOURCE is useless for this under sbatch: Slurm copies the batch script
# into a per-job spool directory, so at run time this file's own directory is
# /cm/local/.../job<id> and the counter is not there. SLURM_SUBMIT_DIR is the
# directory the job was submitted from, which is where the repo copy actually
# lives -- but that is the repo root when the script is submitted by its repo
# path, as the usage above shows, so the counter's own subdirectory has to be
# probed too. Keep the BASH_SOURCE probe last so the script still works when
# run directly with bash, outside Slurm.
for candidate in "${COUNTER:-}" \
                 "${SLURM_SUBMIT_DIR:-}/tools/launcher/common/specdec/count_ptv3_rows.py" \
                 "${SLURM_SUBMIT_DIR:-}/count_ptv3_rows.py" \
                 "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/count_ptv3_rows.py"; do
    [[ -n "$candidate" && -r "$candidate" ]] || continue
    counter="$candidate"
    break
done
[[ -n "${counter:-}" ]] || {
    echo "count_ptv3_rows.py not found; submit from the repo root or set COUNTER" >&2
    exit 2
}
echo "counter: $counter"

scratch="/raid/scratch/${SLURM_JOB_ID}"
mkdir -p "$scratch"
trap 'rm -rf "$scratch"' EXIT

python3 -m venv "$scratch/venv"
"$scratch/venv/bin/pip" install --quiet --no-input pyarrow
"$scratch/venv/bin/python" -c 'import pyarrow; print("pyarrow", pyarrow.__version__)'

mkdir -p "$STUDY_ROOT/receipts"

# Which repos to count. Staging finishes at different times on different
# clusters, so the caller extends this as repos verify complete rather than
# editing the file: REPOS="<space-separated names>".
REPOS="${REPOS:-Nemotron-SFT-Math-v4 Nemotron-RL-Math-v2 Nemotron-SFT-SWE-v3 Nemotron-SFT-SWE-v3.5 Nemotron-SFT-SWE-v2 Nemotron-SWE-v1}"
repo_args=()
for repo in $REPOS; do
    repo_args+=(--repo "$repo")
done
[[ ${#repo_args[@]} -gt 0 ]] || { echo "REPOS is empty" >&2; exit 2; }

# The --expect values come from the HuggingFace dataset index, not from a
# previous run of this counter. That is what makes them a cross-check: an
# agreement means the footer/newline readers and the index independently
# arrived at the same total.
#
# Two repos deliberately carry no expectation. The index reports
# Nemotron-SFT-SWE-v2 as empty, which is the reason it needs measuring at all;
# and Nemotron-SWE-v1's indexed 24,875 describes a partial conversion (1.4 G
# against the 11.1 G the pinned revision actually holds), so treating it as a
# reference would assert agreement with a number that was never the total.
"$scratch/venv/bin/python" "$counter" \
    --data-root "$STUDY_ROOT/ptv3_data" \
    --output "$STUDY_ROOT/receipts/ptv3_rows_${SLURM_JOB_ID}.json" \
    --workers 48 \
    "${repo_args[@]}" \
    --expect Nemotron-RL-Math-v2=7732 --expect Nemotron-SFT-SWE-v3.5=5115 \
    --expect Nemotron-SFT-Math-v4=545431 --expect Nemotron-SFT-SWE-v3=237970
