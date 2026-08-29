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

# The counter ships next to this script, so the study is reproducible from a
# clone rather than from whatever happens to sit in a home directory.
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
counter="$script_dir/count_ptv3_rows.py"
[[ -r "$counter" ]] || { echo "counter not found beside this script: $counter" >&2; exit 2; }

scratch="/raid/scratch/${SLURM_JOB_ID}"
mkdir -p "$scratch"
trap 'rm -rf "$scratch"' EXIT

python3 -m venv "$scratch/venv"
"$scratch/venv/bin/pip" install --quiet --no-input pyarrow
"$scratch/venv/bin/python" -c 'import pyarrow; print("pyarrow", pyarrow.__version__)'

mkdir -p "$STUDY_ROOT/receipts"

# The --expect values come from the HuggingFace dataset index, not from a
# previous run of this counter. That is what makes them a cross-check: an
# agreement means the footer/newline readers and the index independently
# arrived at the same total. Nemotron-SFT-SWE-v2 has no expectation because
# the index reports it as empty, which is the reason it needs measuring.
"$scratch/venv/bin/python" "$counter" \
    --data-root "$STUDY_ROOT/ptv3_data" \
    --output "$STUDY_ROOT/receipts/ptv3_rows_${SLURM_JOB_ID}.json" \
    --workers 48 \
    --repo Nemotron-SFT-Math-v4 --repo Nemotron-RL-Math-v2 \
    --repo Nemotron-SFT-SWE-v3 --repo Nemotron-SFT-SWE-v3.5 --repo Nemotron-SFT-SWE-v2 \
    --expect Nemotron-RL-Math-v2=7732 --expect Nemotron-SFT-SWE-v3.5=5115 \
    --expect Nemotron-SFT-Math-v4=545431 --expect Nemotron-SFT-SWE-v3=237970
