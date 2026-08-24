#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Hash one DFlash2 target or admitted dataset using the authenticated runtime.
set -euo pipefail

[[ $# == 8 ]] || {
    echo "usage: $0 RUNTIME IMAGE LAUNCHER ARTIFACT OUTPUT KIND OCCURRENCES SCRATCH_ROOT" >&2
    exit 2
}
runtime_archive="$(realpath "$1")"
image="$(realpath "$2")"
launcher="$(realpath "$3")"
artifact="$(realpath "$4")"
output="$(realpath -m "$5")"
kind="$6"
occurrences="$7"
scratch_root="$(realpath -m "$8")"

[[ -f "$runtime_archive" && -f "$image" && -d "$launcher" ]] || exit 2
[[ "$kind" == target || "$kind" == dataset ]] || exit 2
[[ "$occurrences" == 0 || "$occurrences" == 1300000 ]] || exit 2
[[ "$kind:$occurrences" == target:0 || "$kind:$occurrences" == dataset:1300000 ]] || exit 2
[[ ! -e "$output" ]] || { echo "refusing to replace artifact receipt" >&2; exit 2; }

work_root="${scratch_root%/}/${SLURM_JOB_ID:?SLURM_JOB_ID is required}/artifact-receipt"
mkdir -p "$work_root/runtime" "$(dirname "$output")"
tar --extract --file="$runtime_archive" --directory="$work_root/runtime"

contract="$launcher/tools/launcher/common/specdec/dflash2_runtime_contract.py"
mounts="${work_root}:${work_root},${launcher}:${launcher}:ro,${artifact}:${artifact}:ro,$(dirname "$output"):$(dirname "$output")"
command=(
    "$work_root/runtime/bin/python" "$contract" artifact-receipt
    --artifact "$artifact" --output "$output" --kind "$kind"
)
if [[ "$kind" == dataset ]]; then
    command+=(--occurrence-count 1300000)
fi
srun --nodes=1 --ntasks=1 --ntasks-per-node=1 --kill-on-bad-exit=1 \
    --no-container-mount-home --container-image="$image" --container-mounts="$mounts" \
    "${command[@]}"
