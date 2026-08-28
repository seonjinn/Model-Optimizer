#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

#SBATCH --account=coreai_dlalgo_llm
#SBATCH --partition=36x2-a01r
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --time=04:00:00

set -euo pipefail
export LC_ALL=C
umask 077

fail() {
    echo "$1" >&2
    exit 2
}

[[ $# == 3 ]] || fail "usage: $0 SOURCE_PATH SOURCE_COMMIT OUTPUT"
readonly source_path="$1"
readonly source_commit="$2"
readonly output="$3"

[[ "$source_path" == /* && "$source_path" != *"/../"* && "$source_path" != *"/./"* \
    && -d "$source_path" && ! -L "$source_path" ]] || fail "source path is unsafe"
[[ "$source_commit" =~ ^[0-9a-f]{40}$ ]] || fail "source commit is invalid"
readonly receipt_root="/lustre/fsw/coreai_dlalgo_llm/users/sna/modelopt-qwen3-drafter-training/receipts/q30t-ptv23-complement-700k-v1"
[[ "$output" =~ ^${receipt_root}/row-schema/observation-[ab]\.json$ ]] \
    || fail "output is not an approved row-schema observation path"
[[ ! -e "$output" && ! -L "$output" ]] || fail "immutable observation output already exists"

[[ "${SLURM_NNODES:-}" == 1 && "${SLURM_CPUS_PER_TASK:-}" == 32 \
    && "${SLURM_JOB_ID:-}" =~ ^[0-9]+$ ]] || fail "observation requires one 32-CPU node"
[[ -z "${SLURM_JOB_GPUS:-}" \
    && ( -z "${SLURM_GPUS_ON_NODE:-}" || "${SLURM_GPUS_ON_NODE:-}" == 0 ) ]] \
    || fail "observation requires a CPU-only allocation"

readonly approved_chmod="/usr/bin/chmod"
readonly approved_git="/usr/bin/git"
readonly approved_mkdir="/usr/bin/mkdir"
readonly approved_mktemp="/usr/bin/mktemp"
readonly approved_python="/usr/bin/python3.12"
readonly approved_sha256sum="/usr/bin/sha256sum"
for executable in /usr/bin/env "$approved_chmod" "$approved_git" "$approved_mkdir" \
    "$approved_mktemp" "$approved_python" "$approved_sha256sum"; do
    [[ -f "$executable" && -x "$executable" && ! -L "$executable" ]] \
        || fail "approved executable is unavailable: $executable"
done

/usr/bin/env -i LC_ALL=C PYTHONDONTWRITEBYTECODE=1 PYTHONSAFEPATH=1 \
    "$approved_python" -I - <<'PY'
import os
import stat
import sys

approved = os.stat("/usr/bin/python3.12", follow_symlinks=False)
running = os.stat("/proc/self/exe")
if not (
    sys.version_info >= (3, 12)
    and stat.S_ISREG(approved.st_mode)
    and approved.st_uid == 0
    and stat.S_IMODE(approved.st_mode) & 0o022 == 0
    and (approved.st_dev, approved.st_ino) == (running.st_dev, running.st_ino)
):
    raise SystemExit("approved Python identity or version mismatch")
PY

readonly scratch_parent="${SLURM_TMPDIR:-}"
[[ "$scratch_parent" == /raid/scratch/* && -d "$scratch_parent" && ! -L "$scratch_parent" ]] \
    || fail "scheduler node-local scratch is unavailable"
scratch_root="$($approved_mktemp -d \
    "$scratch_parent/q30t-row-schema-${SLURM_JOB_ID}.XXXXXXXX")"
readonly scratch_root
[[ "$scratch_root" == "$scratch_parent"/* && -d "$scratch_root" && ! -L "$scratch_root" ]] \
    || fail "private node-local scratch creation failed"

readonly observer_path="examples/dataset/observe_q30t_ptv23_row_schemas.py"
readonly ptv3_plan_path="examples/dataset/qwen3_30ba3b_thinking_ptv3_stage_subset_v1.json"
readonly observer="$scratch_root/observe_q30t_ptv23_row_schemas.py"
readonly ptv3_plan="$scratch_root/qwen3_30ba3b_thinking_ptv3_stage_subset_v1.json"

# git cat-file binds both executable and plan bytes to the reviewed commit object.
/usr/bin/env -i LC_ALL=C GIT_CONFIG_NOSYSTEM=1 GIT_CONFIG_GLOBAL=/dev/null \
    GIT_NO_REPLACE_OBJECTS=1 "$approved_git" -C "$source_path" \
    cat-file -e "$source_commit^{commit}"
/usr/bin/env -i LC_ALL=C GIT_CONFIG_NOSYSTEM=1 GIT_CONFIG_GLOBAL=/dev/null \
    GIT_NO_REPLACE_OBJECTS=1 "$approved_git" -C "$source_path" \
    cat-file blob "$source_commit:$observer_path" >"$observer"
/usr/bin/env -i LC_ALL=C GIT_CONFIG_NOSYSTEM=1 GIT_CONFIG_GLOBAL=/dev/null \
    GIT_NO_REPLACE_OBJECTS=1 "$approved_git" -C "$source_path" \
    cat-file blob "$source_commit:$ptv3_plan_path" >"$ptv3_plan"
"$approved_chmod" 0400 "$observer" "$ptv3_plan"

readonly ptv2_root="/lustre/fsw/coreai_dlalgo_llm/users/sna/modelopt-qwen3-drafter-training/assets/q30t-ptv2-full201-source-v1"
readonly ptv2_plan="$ptv2_root/SOURCE_PLAN.json"
readonly ptv2_plan_sha256="96970541d0c6f5c99e74b9222b805d4a0bd2ac682837b0ad92b8bf54f7d71a3a"
readonly ptv2_completion="$ptv2_root/SOURCE_MANIFEST_COMPLETION.json"
readonly ptv2_completion_sha256="018d659170834b17967dd6c1b066e3b03e7859386eceefd9d4409dc3bc8f48c1"
readonly ptv3_root="/lustre/fsw/coreai_dlalgo_llm/users/sna/modelopt-qwen3-drafter-training/assets/q30t-ptv3-source-stage-75209087-v1"
readonly ptv3_plan_sha256="752090878ed2c5fce47683b2939f9d857be5549311b158476fce33bb79f52eac"
readonly ptv3_completion="$ptv3_root/completion.json"

observed_ptv3_plan_sha256="$(/usr/bin/env -i LC_ALL=C \
    "$approved_sha256sum" "$ptv3_plan")"
[[ "${observed_ptv3_plan_sha256%% *}" == "$ptv3_plan_sha256" ]] \
    || fail "immutable PTV3 plan blob SHA-256 mismatch"
ptv3_completion_sha256="$(/usr/bin/env -i LC_ALL=C PYTHONDONTWRITEBYTECODE=1 \
    PYTHONSAFEPATH=1 "$approved_python" -I - "$ptv3_completion" <<'PY'
import hashlib
import os
import stat
import sys

path = sys.argv[1]
before = os.stat(path, follow_symlinks=False)
descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
try:
    opened = os.fstat(descriptor)
    if (
        not stat.S_ISREG(before.st_mode)
        or before.st_nlink != 1
        or (before.st_dev, before.st_ino) != (opened.st_dev, opened.st_ino)
    ):
        raise SystemExit("PTV3 completion is not a stable single-link regular file")
    digest = hashlib.sha256()
    while chunk := os.read(descriptor, 1024 * 1024):
        digest.update(chunk)
    after = os.fstat(descriptor)
    current = os.stat(path, follow_symlinks=False)
    identity = lambda value: (
        value.st_dev,
        value.st_ino,
        value.st_mode,
        value.st_nlink,
        value.st_size,
        value.st_mtime_ns,
        value.st_ctime_ns,
    )
    if identity(before) != identity(after) or identity(after) != identity(current):
        raise SystemExit("PTV3 completion changed while binding its SHA-256")
    print(digest.hexdigest())
finally:
    os.close(descriptor)
PY
)"
readonly ptv3_completion_sha256
[[ "$ptv3_completion_sha256" =~ ^[0-9a-f]{64}$ ]] \
    || fail "PTV3 completion SHA-256 capture failed"

"$approved_mkdir" -p -- "$receipt_root/row-schema"
/usr/bin/env -i LC_ALL=C PYTHONDONTWRITEBYTECODE=1 PYTHONSAFEPATH=1 \
    "$approved_python" -I "$observer" \
    --ptv2-plan "$ptv2_plan" \
    --ptv2-plan-sha256 "$ptv2_plan_sha256" \
    --ptv2-completion "$ptv2_completion" \
    --ptv2-completion-sha256 "$ptv2_completion_sha256" \
    --ptv2-root "$ptv2_root" \
    --ptv3-plan "$ptv3_plan" \
    --ptv3-plan-sha256 "$ptv3_plan_sha256" \
    --ptv3-completion "$ptv3_completion" \
    --ptv3-completion-sha256 "$ptv3_completion_sha256" \
    --ptv3-root "$ptv3_root" \
    --output "$output"
