#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

usage() {
    echo "usage: $0 (--test-only|--submit) --source-path PATH --output PATH --slurm-output PATH" >&2
    exit 2
}

mode="" source_path="" output_path="" slurm_output=""
while (($#)); do
    case "$1" in
        --test-only|--submit) [[ -z "$mode" ]] || usage; mode="$1"; shift ;;
        --source-path) source_path="${2:-}"; shift 2 ;;
        --output) output_path="${2:-}"; shift 2 ;;
        --slurm-output) slurm_output="${2:-}"; shift 2 ;;
        *) usage ;;
    esac
done
[[ -n "$mode" && "$source_path" == /* && "$output_path" == /* \
    && "$slurm_output" == /* ]] || usage

readonly canonical_remote="gitlab"
readonly canonical_branch="refs/heads/sj/q30t-ptv23-complement-700k"
readonly approved_gitlab_fetch_url="ssh://git@gitlab-master.nvidia.com:12051/sna/modelopt.git"
readonly runner_path="tools/launcher/common/specdec/run_q30t_ptv2_full201_transfer_verify.sbatch"
readonly approved_output="/lustre/fsw/coreai_dlalgo_llm/users/sna/modelopt-qwen3-drafter-training/receipts/q30t-ptv23-complement-700k-v1/ptv2-full201-transfer/VERIFY.json"
[[ "$output_path" == "$approved_output" ]] || {
    echo "OUTPUT_PATH is not the approved verification receipt" >&2
    exit 2
}
[[ -d "$source_path" && ! -L "$source_path" ]] || {
    echo "source checkout is unavailable" >&2
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
source_git() {
    "${git_environment[@]}" git -C "$source_path" "$@"
}
source_toplevel="$(source_git rev-parse --show-toplevel 2>/dev/null)" || {
    echo "SOURCE_PATH is not a Git worktree" >&2
    exit 2
}
[[ "$source_toplevel" == "$source_path" ]] || {
    echo "Git worktree is not bound to SOURCE_PATH" >&2
    exit 2
}
[[ -z "$(source_git -c core.fsmonitor=false status --porcelain=v1 --untracked-files=all)" ]] || {
    echo "transfer verifier submission requires a clean checkout" >&2
    exit 2
}
source_sha="$(source_git rev-parse --verify 'HEAD^{commit}')" || {
    echo "source HEAD is unavailable" >&2
    exit 2
}
configured_remote_url="$(source_git config --includes --get-all \
    "remote.${canonical_remote}.url" 2>/dev/null)" || {
    echo "canonical source remote URL is unavailable" >&2
    exit 2
}
[[ "$configured_remote_url" == "$approved_gitlab_fetch_url" ]] || {
    echo "gitlab remote does not use the approved GitLab fetch URL" >&2
    exit 2
}
if source_git config --includes --local --get-regexp \
    '^(url\..*\.insteadof|core\.sshcommand)$' >/dev/null 2>&1 \
    || source_git config --includes --worktree --get-regexp \
        '^(url\..*\.insteadof|core\.sshcommand)$' >/dev/null 2>&1; then
    echo "source checkout contains a disallowed Git transport override" >&2
    exit 2
fi

sterile_parent="${TMPDIR:-/tmp}"
[[ -d "$sterile_parent" && ! -L "$sterile_parent" ]] || {
    echo "sterile submitter parent is unavailable" >&2
    exit 2
}
umask 077
private_scratch="$(mktemp -d "${sterile_parent%/}/q30t-transfer-submit.XXXXXX")" || {
    echo "private submitter scratch creation failed" >&2
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
isolated_ref="refs/codex-transfer-submitter/verified"
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
    echo "source HEAD does not match the freshly fetched canonical commit" >&2
    exit 2
}

tree_record="$(sterile_git ls-tree "$fetched_commit" -- "$runner_path")" || {
    echo "authenticated runner tree entry is unavailable" >&2
    exit 2
}
IFS=$'\t' read -r tree_metadata tree_path <<<"$tree_record"
read -r tree_mode tree_type runner_object extra <<<"${tree_metadata:-}"
[[ "$tree_mode" == 100644 && "$tree_type" == blob && -z "${extra:-}" \
    && "$runner_object" =~ ^[0-9a-f]{40}$ && "$tree_path" == "$runner_path" ]] || {
    echo "authenticated runner is not the exact approved regular Git blob" >&2
    exit 2
}
runner_size="$(sterile_git cat-file -s "$runner_object")" || {
    echo "authenticated runner size is unavailable" >&2
    exit 2
}
[[ "$runner_size" =~ ^[1-9][0-9]*$ && "$runner_size" -le 8388608 ]] || {
    echo "authenticated runner size is invalid" >&2
    exit 2
}

invoke_sbatch_with_fresh_sealed_runner() {
    local scheduler_mode="$1"
    sterile_git cat-file blob "$runner_object" | \
        env -u PYTHONPATH -u PYTHONHOME PYTHONDONTWRITEBYTECODE=1 PYTHONSAFEPATH=1 \
        python3 -S -I -P /dev/fd/3 \
            "$runner_object" "$runner_size" "$scheduler_mode" "$slurm_output" \
            "$source_sha" "$output_path" 3<<'PY'
import errno
import fcntl
import hashlib
import os
import subprocess
import sys
import tempfile


def fail(message: str) -> None:
    raise RuntimeError(message)


expected_object = sys.argv[1]
expected_size = int(sys.argv[2])
payload = sys.stdin.buffer.read(expected_size + 1)
if len(payload) != expected_size:
    fail("authenticated runner blob length mismatch")
digest = hashlib.sha1()
digest.update(f"blob {len(payload)}\0".encode("ascii"))
digest.update(payload)
if digest.hexdigest() != expected_object:
    fail("authenticated runner bytes do not match the fetched Git blob")

if sys.platform == "linux":
    if not hasattr(os, "memfd_create") or not hasattr(os, "MFD_ALLOW_SEALING"):
        fail("Linux memfd sealing support is unavailable")
    descriptor = os.memfd_create(
        "q30t-ptv2-transfer-runner",
        os.MFD_ALLOW_SEALING | getattr(os, "MFD_CLOEXEC", 0),
    )
    runner_path = f"/proc/self/fd/{descriptor}"
else:
    if os.environ.get("Q30T_TEST_ALLOW_UNSEALED_PLATFORM") != sys.platform:
        fail("sealed runner submission requires Linux")
    descriptor, temporary_path = tempfile.mkstemp(prefix="q30t-test-runner-")
    os.unlink(temporary_path)
    runner_path = f"/dev/fd/{descriptor}"

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
        fail("sealed runner is missing required Linux memfd seals")
    try:
        os.write(descriptor, b"x")
    except OSError as error:
        if error.errno != errno.EPERM:
            fail(f"sealed runner write failed with errno {error.errno}")
    else:
        fail("sealed runner accepted a write")
    os.lseek(descriptor, 0, os.SEEK_SET)
if hashlib.sha256(os.read(descriptor, expected_size + 1)).digest() != hashlib.sha256(payload).digest():
    fail("sealed runner reread mismatch")
os.lseek(descriptor, 0, os.SEEK_SET)

command = [
    "sbatch",
    sys.argv[3],
    "--nodes=1",
    "--cpus-per-task=32",
    "--job-name=q30t-ptv2-full201-transfer-verify",
    f"--output={sys.argv[4]}",
    "--export=NONE",
    runner_path,
    sys.argv[5],
    sys.argv[6],
]
result = subprocess.run(command, pass_fds=(descriptor,), check=False, capture_output=True)
if result.returncode != 0:
    sys.stderr.buffer.write(result.stderr)
    raise SystemExit(result.returncode)
if sys.argv[3] == "--parsable":
    sys.stdout.buffer.write(result.stdout)
PY
}

invoke_sbatch_with_fresh_sealed_runner --test-only
if [[ "$mode" == --test-only ]]; then
    echo "test-only passed"
    exit 0
fi
job_id="$(invoke_sbatch_with_fresh_sealed_runner --parsable)"
[[ "$job_id" =~ ^[0-9]+$ ]] || {
    echo "invalid sbatch job ID" >&2
    exit 1
}
printf '%s\n' "$job_id"
