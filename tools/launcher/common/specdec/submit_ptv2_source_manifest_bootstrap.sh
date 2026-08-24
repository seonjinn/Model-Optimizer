#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

usage() {
    echo "usage: $0 --account ACCOUNT --repo-root PATH --source-commit SHA --source-root PATH --approved-hao-root PATH --readme PATH --output-root PATH --time HH:MM:SS [--dry-run]" >&2
    exit 2
}

account=""
repo_root=""
source_commit=""
source_root=""
approved_hao_root=""
readme=""
output_root=""
walltime=""
dry_run=0
while (( $# )); do
    case "$1" in
        --account) account="${2:-}"; shift 2 ;;
        --repo-root) repo_root="${2:-}"; shift 2 ;;
        --source-commit) source_commit="${2:-}"; shift 2 ;;
        --source-root) source_root="${2:-}"; shift 2 ;;
        --approved-hao-root) approved_hao_root="${2:-}"; shift 2 ;;
        --readme) readme="${2:-}"; shift 2 ;;
        --output-root) output_root="${2:-}"; shift 2 ;;
        --time) walltime="${2:-}"; shift 2 ;;
        --dry-run) dry_run=1; shift ;;
        *) usage ;;
    esac
done
[[ -n "$account" && -n "$repo_root" && -n "$source_commit" ]] || usage
[[ -n "$source_root" && -n "$approved_hao_root" && -n "$readme" ]] || usage
[[ -n "$output_root" && -n "$walltime" ]] || usage
case "$account" in nemotron_n4_post|nemotron_sw_post) ;; *) usage ;; esac
[[ "$source_commit" =~ ^[0-9a-f]{40}$ ]] || usage
[[ "$walltime" =~ ^[0-9]{2,3}:[0-5][0-9]:[0-5][0-9]$ ]] || usage
for value in "$repo_root" "$source_root" "$approved_hao_root" "$readme" "$output_root"; do
    [[ "$value" == /* && "$value" != *","* && "$value" != *$'\n'* ]] || usage
done
[[ -d "$repo_root" && -d "$source_root" && -d "$approved_hao_root" && -f "$readme" ]] || usage
readonly REPO_ROOT="$repo_root"
readonly SOURCE_COMMIT="$source_commit"
[[ -z "$(git -C "$REPO_ROOT" status --porcelain)" ]] || {
    echo "source checkout must be clean before pull" >&2
    exit 2
}
git -C "$REPO_ROOT" pull --ff-only
[[ -z "$(git -C "$REPO_ROOT" status --porcelain)" ]] || {
    echo "source checkout became dirty after pull" >&2
    exit 2
}
[[ "$(git -C "$REPO_ROOT" rev-parse HEAD)" == "$SOURCE_COMMIT" ]] || {
    echo "pulled checkout does not match requested source commit" >&2
    exit 2
}

runner="$repo_root/tools/launcher/common/specdec/run_ptv2_source_manifest_bootstrap.sbatch"
[[ -f "$runner" ]] || { echo "bootstrap runner is absent" >&2; exit 2; }
log_root="$(dirname "$output_root")/logs/ptv2-source-manifest-${source_commit:0:12}"
mkdir -p "$log_root"
exports="ALL,REPO_ROOT=$repo_root,SOURCE_COMMIT=$source_commit,PTV2_SYMLINK_ROOT=$source_root,APPROVED_HAO_ROOT=$approved_hao_root,PTV2_README=$readme,PTV2_MANIFEST_OUTPUT=$output_root"
args=(
    --account="$account"
    --partition=cpu_datamover
    --nodes=1
    --ntasks=1
    --cpus-per-task=96
    --mem=0
    --time="$walltime"
    --job-name="ptv2-manifest-${source_commit:0:12}"
    --comment="ptv2-source-manifest:$source_commit"
    --output="$log_root/%x-%j.out"
    --error="$log_root/%x-%j.err"
    --export="$exports"
)
sbatch --test-only "${args[@]}" "$runner" >/dev/null
if (( dry_run )); then
    echo "test-only passed"
    exit 0
fi
job_id="$(sbatch --parsable "${args[@]}" "$runner")"
job_id="${job_id%%;*}"
[[ "$job_id" =~ ^[0-9]+$ ]] || { echo "invalid sbatch job ID" >&2; exit 1; }
echo "$job_id"
