#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

#SBATCH --account=coreai_dlalgo_llm
#SBATCH --partition=36x2-a01r
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --time=01:00:00

set -euo pipefail
export LC_ALL=C

[[ $# == 2 ]] || {
    echo "usage: $0 SOURCE_SHA OUTPUT_PATH" >&2
    exit 2
}
source_sha="$1"
output_path="$2"
[[ "$source_sha" =~ ^[0-9a-f]{40}$ && "$output_path" == /* ]] || {
    echo "invalid transfer verification arguments" >&2
    exit 2
}
[[ "${SLURM_NNODES:-}" == 1 && "${SLURM_CPUS_PER_TASK:-}" == 32 \
    && "${SLURM_JOB_ID:-}" =~ ^[0-9]+$ ]] || {
    echo "transfer verification requires one 32-CPU node" >&2
    exit 2
}
[[ -z "${SLURM_JOB_GPUS:-}" \
    && ( -z "${SLURM_GPUS_ON_NODE:-}" || "${SLURM_GPUS_ON_NODE:-}" == 0 ) ]] || {
    echo "transfer verification must use a CPU-only allocation" >&2
    exit 2
}

readonly canonical_branch="refs/heads/sj/q30t-ptv23-complement-700k"
readonly approved_gitlab_fetch_url="ssh://git@gitlab-master.nvidia.com:12051/sna/modelopt.git"
readonly verifier_path="tools/launcher/common/specdec/verify_q30t_ptv2_full201_transfer.py"
readonly source_root="/lustre/fsw/coreai_dlalgo_llm/users/sna/modelopt-qwen3-drafter-training/assets/q30t-ptv2-full201-source-v1"
readonly approved_output="/lustre/fsw/coreai_dlalgo_llm/users/sna/modelopt-qwen3-drafter-training/receipts/q30t-ptv23-complement-700k-v1/ptv2-full201-transfer/VERIFY.json"
[[ "$output_path" == "$approved_output" ]] || {
    echo "OUTPUT_PATH is not the approved verification receipt" >&2
    exit 2
}
[[ ! -e "$output_path" && ! -L "$output_path" ]] || {
    echo "immutable transfer verification receipt already exists" >&2
    exit 2
}

git_environment=(
    env
    -u GIT_DIR
    -u GIT_WORK_TREE
    -u GIT_COMMON_DIR
    -u GIT_INDEX_FILE
    -u GIT_OBJECT_DIRECTORY
    -u GIT_ALTERNATE_OBJECT_DIRECTORIES
    -u GIT_NAMESPACE
    -u GIT_SHALLOW_FILE
    -u GIT_REPLACE_REF_BASE
    -u GIT_CEILING_DIRECTORIES
    -u GIT_DISCOVERY_ACROSS_FILESYSTEM
    -u GIT_CONFIG
    -u GIT_CONFIG_SYSTEM
    -u GIT_CONFIG_GLOBAL
    -u GIT_CONFIG_NOSYSTEM
    -u GIT_CONFIG_COUNT
    -u GIT_CONFIG_PARAMETERS
    -u GIT_TEMPLATE_DIR
    -u GIT_SSH
    -u GIT_SSH_COMMAND
    -u GIT_SSH_VARIANT
    -u GIT_PROXY_COMMAND
    GIT_CONFIG_NOSYSTEM=1
    GIT_CONFIG_GLOBAL=/dev/null
    GIT_NO_REPLACE_OBJECTS=1
)

sterile_parent="${SLURM_TMPDIR:-}"
[[ -d "$sterile_parent" && ! -L "$sterile_parent" ]] || {
    echo "scheduler-owned SLURM_TMPDIR is unavailable" >&2
    exit 2
}
umask 077
private_scratch="$(mktemp -d "${sterile_parent%/}/q30t-transfer-verifier.XXXXXX")" || {
    echo "private verifier scratch creation failed" >&2
    exit 2
}
sterile_repo="$private_scratch/repository.git"
"${git_environment[@]}" git init --bare --template=/dev/null "$sterile_repo" >/dev/null 2>&1 || {
    echo "sterile Git repository initialization failed" >&2
    exit 2
}
sterile_git() {
    "${git_environment[@]}" git --git-dir="$sterile_repo" "$@"
}
isolated_ref="refs/codex-transfer-verifier/${SLURM_JOB_ID}"
sterile_git fetch --filter=blob:none --depth=1 --no-tags --no-write-fetch-head --force \
    "$approved_gitlab_fetch_url" "+${canonical_branch}:${isolated_ref}" >/dev/null 2>&1 || {
    echo "canonical source fetch failed" >&2
    exit 2
}
fetched_commit="$(sterile_git rev-parse --verify "${isolated_ref}^{commit}" 2>/dev/null)" || {
    echo "freshly fetched source commit is unavailable" >&2
    exit 2
}
[[ "$source_sha" == "$fetched_commit" ]] || {
    echo "requested commit does not match the freshly fetched canonical commit" >&2
    exit 2
}

tree_record="$(sterile_git ls-tree "$fetched_commit" -- "$verifier_path")" || {
    echo "authenticated verifier tree entry is unavailable" >&2
    exit 2
}
[[ "$tree_record" != *$'\n'* ]] || {
    echo "authenticated verifier tree lookup returned multiple entries" >&2
    exit 2
}
IFS=$'\t' read -r tree_metadata tree_path <<<"$tree_record"
read -r tree_mode tree_type verifier_object extra <<<"${tree_metadata:-}"
[[ "$tree_mode" == 100644 && "$tree_type" == blob && -z "${extra:-}" \
    && "$verifier_object" =~ ^[0-9a-f]{40}$ && "$tree_path" == "$verifier_path" ]] || {
    echo "authenticated verifier is not the exact approved regular Git blob" >&2
    exit 2
}
verifier_size="$(sterile_git cat-file -s "$verifier_object")" || {
    echo "authenticated verifier size is unavailable" >&2
    exit 2
}
[[ "$verifier_size" =~ ^[1-9][0-9]*$ && "$verifier_size" -le 8388608 ]] || {
    echo "authenticated verifier size is invalid" >&2
    exit 2
}

sterile_git cat-file blob "$verifier_object" | \
    env -u PYTHONPATH -u PYTHONHOME PYTHONDONTWRITEBYTECODE=1 PYTHONSAFEPATH=1 \
    python3 -S -I -P /dev/fd/3 \
        "$verifier_object" "$verifier_size" "$source_root" "$output_path" 16 3<<'PY'
import errno
import fcntl
import hashlib
import os
import sys
import tempfile


def fail(message: str) -> None:
    raise RuntimeError(message)


expected_object = sys.argv[1]
expected_size = int(sys.argv[2])
payload = sys.stdin.buffer.read(expected_size + 1)
if len(payload) != expected_size:
    fail("authenticated verifier blob length mismatch")
digest = hashlib.sha1()
digest.update(f"blob {len(payload)}\0".encode("ascii"))
digest.update(payload)
if digest.hexdigest() != expected_object:
    fail("authenticated verifier bytes do not match the fetched Git blob")

if sys.platform == "linux":
    if not hasattr(os, "memfd_create") or not hasattr(os, "MFD_ALLOW_SEALING"):
        fail("Linux memfd sealing support is unavailable")
    descriptor = os.memfd_create(
        "q30t-ptv2-transfer-verifier",
        os.MFD_ALLOW_SEALING | getattr(os, "MFD_CLOEXEC", 0),
    )
else:
    if os.environ.get("Q30T_TEST_ALLOW_UNSEALED_PLATFORM") != sys.platform:
        fail("sealed verifier execution requires Linux")
    descriptor, temporary_path = tempfile.mkstemp(prefix="q30t-test-verifier-")
    os.unlink(temporary_path)

offset = 0
while offset < len(payload):
    offset += os.write(descriptor, payload[offset:])
os.lseek(descriptor, 0, os.SEEK_SET)
if sys.platform == "linux":
    required_seals = (
        fcntl.F_SEAL_WRITE | fcntl.F_SEAL_GROW | fcntl.F_SEAL_SHRINK | fcntl.F_SEAL_SEAL
    )
    fcntl.fcntl(descriptor, fcntl.F_ADD_SEALS, required_seals)
    if fcntl.fcntl(descriptor, fcntl.F_GET_SEALS) & required_seals != required_seals:
        fail("sealed verifier is missing required Linux memfd seals")
    try:
        os.write(descriptor, b"x")
    except OSError as error:
        if error.errno != errno.EPERM:
            fail(f"sealed verifier write failed with errno {error.errno}")
    else:
        fail("sealed verifier accepted a write")
    os.lseek(descriptor, 0, os.SEEK_SET)
    print("sealed-verifier-write-check: inplace=EPERM reopened=EPERM", file=sys.stderr)
else:
    print(f"sealed-verifier-write-check: test-fallback={sys.platform}", file=sys.stderr)

os.set_inheritable(descriptor, True)
descriptor_root = "/proc/self/fd" if sys.platform == "linux" else "/dev/fd"
for name in os.listdir(descriptor_root):
    if not name.isdigit():
        continue
    candidate = int(name)
    if candidate <= 2 or candidate == descriptor:
        continue
    try:
        os.set_inheritable(candidate, False)
    except OSError as error:
        if error.errno != errno.EBADF:
            raise

environment = os.environ.copy()
for name in tuple(environment):
    if name.startswith("GIT_") or name in {"PYTHONHOME", "PYTHONPATH"}:
        environment.pop(name, None)
environment.pop("Q30T_TEST_ALLOW_UNSEALED_PLATFORM", None)
environment["PYTHONDONTWRITEBYTECODE"] = "1"
environment["PYTHONSAFEPATH"] = "1"
verifier = f"/dev/fd/{descriptor}"
os.execve(
    sys.executable,
    [
        sys.executable,
        "-S",
        "-I",
        "-P",
        verifier,
        "--source-root",
        sys.argv[3],
        "--output",
        sys.argv[4],
        "--workers",
        sys.argv[5],
    ],
    environment,
)
PY
