#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Stage or resume the PTv3 candidate set on a data-mover node.
#
# stage_ptv3.sh skips any file already present at non-zero size, so this is
# both the first staging run and the resume: a repo that finished costs a tree
# fetch, and a repo that stopped mid-file restarts that file. Run it again
# after a failure rather than reasoning about what to re-fetch.
#
# The download is tens of gigabytes and belongs on a data-mover node, not a
# login node.
#
# Usage:
#   STUDY_ROOT=<lustre study dir> sbatch --partition=<cpu> --account=<acct> \
#       tools/launcher/common/specdec/run_stage_ptv3.sbatch

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=0

set -euo pipefail

[[ -n "${STUDY_ROOT:-}" ]] || { echo "missing required environment: STUDY_ROOT" >&2; exit 2; }
[[ "$STUDY_ROOT" == /lustre/* ]] || { echo "staged data belongs on Lustre" >&2; exit 2; }
[[ -n "${HF_TOKEN:-}" ]] || { echo "missing required environment: HF_TOKEN" >&2; exit 2; }
[[ "${SLURM_JOB_ID:-}" =~ ^[0-9]+$ ]] || { echo "invalid Slurm job identity" >&2; exit 2; }

# Same resolution as count_ptv3_rows.sbatch, and for the same reason: Slurm
# runs a spooled copy of this script, so its own directory at run time holds
# none of its siblings. SLURM_SUBMIT_DIR is where the job was submitted from.
for candidate in "${STAGER:-}" \
                 "${SLURM_SUBMIT_DIR:-}/tools/launcher/common/specdec/stage_ptv3.sh" \
                 "${SLURM_SUBMIT_DIR:-}/stage_ptv3.sh" \
                 "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/stage_ptv3.sh"; do
    [[ -n "$candidate" && -r "$candidate" ]] || continue
    stager="$candidate"
    break
done
[[ -n "${stager:-}" ]] || {
    echo "stage_ptv3.sh not found; submit from the repo root or set STAGER" >&2
    exit 2
}
echo "stager: $stager"
echo "study root: $STUDY_ROOT"

mkdir -p "$STUDY_ROOT/ptv3_data"

# The token goes over a pipe on stdin. Passing it in argv would expose it to
# every user on the node, since /proc here is mounted without hidepid.
# Capture the status rather than letting set -e take the exit: the summary
# below is the point of the job, and a partial stage is the case where it is
# most worth reading.
status=0
printf '%s\n' "$HF_TOKEN" | bash "$stager" "$STUDY_ROOT/ptv3_data" || status=$?

echo "stager exit: $status"
cat "$STUDY_ROOT/stage_ptv3_status.txt" 2>/dev/null || true
exit "$status"
