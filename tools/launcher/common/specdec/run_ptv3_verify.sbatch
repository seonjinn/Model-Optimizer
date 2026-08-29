#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Verify staged PTv3 shards against every pinned revision tree.
#
# This exists so completeness can gate a job rather than be read by eye: chain
# it after the stager with --dependency=afterok and chain the row counter after
# it, and the counter only runs against a tree the pins vouch for.
#
# Usage:
#   STUDY_ROOT=<study dir> sbatch --dependency=afterok:<stage job> \
#       --partition=<cpu> --account=<acct> \
#       tools/launcher/common/specdec/run_ptv3_verify.sbatch

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8

set -euo pipefail

[[ -n "${STUDY_ROOT:-}" ]] || { echo "missing required environment: STUDY_ROOT" >&2; exit 2; }
[[ -n "${HF_TOKEN:-}" ]] || { echo "missing required environment: HF_TOKEN" >&2; exit 2; }
# Same reasoning as run_stage_ptv3.sbatch: shared parallel storage is /lustre on
# three clusters and /scratch/fsw on the other two, so name the storage staged
# data must not be on rather than allow-listing one prefix.
case "$STUDY_ROOT" in
    /home/*|/raid/*|/tmp/*|/var/*|/cm/*|/dev/shm/*)
        echo "staged data lives on shared parallel storage, not $STUDY_ROOT" >&2; exit 2 ;;
    /*) ;;
    *) echo "STUDY_ROOT must be an absolute path, got: $STUDY_ROOT" >&2; exit 2 ;;
esac
[[ "${SLURM_JOB_ID:-}" =~ ^[0-9]+$ ]] || { echo "invalid Slurm job identity" >&2; exit 2; }

# Slurm runs a spooled copy of this script, so its own directory holds none of
# its siblings at run time; SLURM_SUBMIT_DIR is where the job was submitted from.
for candidate in "${VERIFIER:-}" \
                 "${SLURM_SUBMIT_DIR:-}/tools/launcher/common/specdec/verify_ptv3.py" \
                 "${SLURM_SUBMIT_DIR:-}/verify_ptv3.py" \
                 "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/verify_ptv3.py"; do
    [[ -n "$candidate" && -r "$candidate" ]] || continue
    verifier="$candidate"
    break
done
[[ -n "${verifier:-}" ]] || {
    echo "verify_ptv3.py not found; submit from the repo root or set VERIFIER" >&2
    exit 2
}
echo "verifier: $verifier"
echo "study root: $STUDY_ROOT"

# Non-zero exit is the point: a dependent counter must not start against an
# incomplete tree.
python3 "$verifier" "$STUDY_ROOT/ptv3_data"
