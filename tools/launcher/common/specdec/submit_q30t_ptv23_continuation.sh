#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

usage() {
    echo "usage: $0 (--test-only|--submit-canary|--submit-full) --account ACCOUNT --output LOG (--canary-receipt|--full-receipt) PATH --launch-descriptor PATH --launch-descriptor-sha256 SHA256" >&2
    exit 2
}

mode="" account="" output="" completion_receipt="" launch_descriptor="" launch_descriptor_sha256=""
while (($#)); do
    case "$1" in
        --test-only|--submit-canary|--submit-full) [[ -z "$mode" ]] || usage; mode="$1"; shift ;;
        --account) account="${2:-}"; shift 2 ;;
        --output) output="${2:-}"; shift 2 ;;
        --canary-receipt|--full-receipt) [[ -z "$completion_receipt" ]] || usage; completion_receipt="${2:-}"; shift 2 ;;
        --launch-descriptor) launch_descriptor="${2:-}"; shift 2 ;;
        --launch-descriptor-sha256) launch_descriptor_sha256="${2:-}"; shift 2 ;;
        *) usage ;;
    esac
done
[[ -n "$mode" && -n "$account" && "$output" == /* && "$completion_receipt" == /* \
    && "$launch_descriptor" == /* && "$launch_descriptor_sha256" =~ ^[0-9a-f]{64}$ ]] || usage
[[ ! -e "$completion_receipt" && ! -L "$completion_receipt" \
    && -f "$launch_descriptor" && ! -L "$launch_descriptor" ]] || usage
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
repo_root="$(git -C "$script_dir" rev-parse --show-toplevel)"
[[ -z "$(git -C "$repo_root" status --porcelain --untracked-files=all)" ]] || {
    echo "Q30 submission requires an exact clean pushed commit" >&2
    exit 2
}
head_commit="$(git -C "$repo_root" rev-parse HEAD)"
upstream_commit="$(git -C "$repo_root" rev-parse '@{upstream}')"
[[ "$head_commit" == "$upstream_commit" ]] || {
    echo "Q30 submission commit is not pushed to its upstream" >&2
    exit 2
}
auth_dir="$(mktemp -d "${TMPDIR:-/tmp}/q30-submit-auth.XXXXXX")"
auth_environment="$auth_dir/environment.bin"
trap 'rm -f -- "$auth_environment"; rmdir -- "$auth_dir"' EXIT
PYTHONPATH="$repo_root/tools/launcher" python3 -m common.specdec.q30t_ptv23_continuation \
    replay-launch-descriptor \
    --descriptor "$launch_descriptor" \
    --sha256 "$launch_descriptor_sha256" \
    --source-checkout "$repo_root" \
    --environment-output "$auth_environment"
authenticated_method="" authenticated_run_name=""
while IFS= read -r -d '' name && IFS= read -r -d '' value; do
    case "$name" in
        METHOD) authenticated_method="$value" ;;
        RUN_NAME) authenticated_run_name="$value" ;;
    esac
done < "$auth_environment"
case "$authenticated_run_name" in
    q30t-dflash-b8-ptv23-swe-heavy-700k-canary) expected_method=DFlash; authenticated_stage=canary ;;
    q30t-dspark-b8-ptv23-swe-heavy-700k-canary) expected_method=DSpark; authenticated_stage=canary ;;
    q30t-dflash-b8-ptv23-swe-heavy-700k-full) expected_method=DFlash; authenticated_stage=full ;;
    q30t-dspark-b8-ptv23-swe-heavy-700k-full) expected_method=DSpark; authenticated_stage=full ;;
    *) echo "authenticated Q30 submission method is invalid" >&2; exit 2 ;;
esac
[[ "$authenticated_method" == "$expected_method" ]] || {
    echo "authenticated Q30 submission method/run mismatch" >&2
    exit 2
}
[[ "$mode" == --test-only || "$mode" == "--submit-$authenticated_stage" ]] || usage
rm -f -- "$auth_environment"
rmdir -- "$auth_dir"
trap - EXIT
runner="$script_dir/run_q30t_ptv23_continuation.sbatch"
args=(--account="$account" --nodes=16 --gpus-per-node=4 --job-name="$authenticated_run_name" \
    --output="$output" --export=NONE)
[[ "$authenticated_stage" == canary ]] || args+=(--time=48:00:00)
job_args=("$runner" "$launch_descriptor" "$launch_descriptor_sha256" "$completion_receipt" "$repo_root")
sbatch --test-only "${args[@]}" "${job_args[@]}" >/dev/null
if [[ "$mode" == --test-only ]]; then
    echo "test-only passed"
    exit 0
fi
sbatch --parsable "${args[@]}" "${job_args[@]}"
