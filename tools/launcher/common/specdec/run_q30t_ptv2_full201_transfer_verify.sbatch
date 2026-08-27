#!/bin/bash
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

readonly approved_python="/usr/bin/python3"
readonly approved_git="/usr/bin/git"
readonly approved_ssh="/usr/bin/ssh"
system_python="$approved_python"
system_git="$approved_git"
system_ssh="$approved_ssh"
test_executables=0
if [[ -n "${Q30T_TEST_ALLOW_SYSTEM_EXECUTABLES:-}" \
    || -n "${Q30T_TEST_PYTHON:-}" || -n "${Q30T_TEST_GIT:-}" \
    || -n "${Q30T_TEST_SSH:-}" ]]; then
    [[ "$OSTYPE" != linux* \
        && "${Q30T_TEST_ALLOW_SYSTEM_EXECUTABLES:-}" == non-linux-test \
        && "${Q30T_TEST_PYTHON:-}" == /* && "${Q30T_TEST_GIT:-}" == /* \
        && "${Q30T_TEST_SSH:-}" == /* ]] || {
        echo "system executable test override is forbidden" >&2
        exit 2
    }
    system_python="$Q30T_TEST_PYTHON"
    system_git="$Q30T_TEST_GIT"
    system_ssh="$Q30T_TEST_SSH"
    test_executables=1
fi
for executable in "$system_python" "$system_git" "$system_ssh"; do
    [[ -f "$executable" && -x "$executable" ]] || {
        echo "approved system executable is unavailable" >&2
        exit 2
    }
done

job_user="$(/usr/bin/env -i LC_ALL=C /usr/bin/id -un 2>/dev/null)" || {
    echo "job user identity is unavailable" >&2
    exit 2
}
[[ "$job_user" =~ ^[A-Za-z0-9._-]+$ ]] || {
    echo "job user identity is invalid" >&2
    exit 2
}
node_local_root="/raid/scratch"
if [[ -n "${Q30T_TEST_ALLOW_NODE_LOCAL_SCRATCH_ROOT:-}" \
    || -n "${Q30T_TEST_NODE_LOCAL_SCRATCH_ROOT:-}" ]]; then
    [[ "$OSTYPE" != linux* \
        && "${Q30T_TEST_ALLOW_NODE_LOCAL_SCRATCH_ROOT:-}" == 1 \
        && "${Q30T_TEST_NODE_LOCAL_SCRATCH_ROOT:-}" == /* ]] || {
        echo "node-local scratch test override is forbidden" >&2
        exit 2
    }
    node_local_root="$Q30T_TEST_NODE_LOCAL_SCRATCH_ROOT"
fi

supervisor_environment=(
    /usr/bin/env -i
    LC_ALL=C
    PYTHONDONTWRITEBYTECODE=1
    PYTHONSAFEPATH=1
)
if ((test_executables)); then
    readonly test_environment_names=(
        APPROVED_TEST_GIT_URL BARE_REPOSITORY_INODE_EVIDENCE BARE_REPOSITORY_PATH
        DISABLE_APPROVED_URL_REWRITE ESCAPED_WRITE_MARKER FORBIDDEN_FD
        GIT_CALLS GIT_ENV_EVIDENCE GLOBAL_SITECUSTOMIZE_MARKER
        INJECT_AUTH_TREE_RECORD INJECT_BARE_REPOSITORY_REPLACEMENT
        INJECT_MANIFEST_PATH_SWAP INJECT_SCRATCH_REPLACEMENT INJECT_STAGE_ENTRY
        MANIFEST_SWAP_MARKER PYTHON_CALLS Q30T_TEST_ALLOW_UNSEALED_PLATFORM
        PROBE_SSH_COMMAND REAL_GIT REAL_PYTHON SCRATCH_REPLACEMENT_POINTER
        SSH_CALLS SSH_REDIRECT_MARKER TEST_GIT_FETCH_URL TEST_TREE_DUPLICATE_PATH
        TEST_TREE_OBJECT_ID
    )
    for name in "${test_environment_names[@]}"; do
        if declare -p "$name" &>/dev/null; then
            supervisor_environment+=("$name=${!name}")
        fi
    done
fi

"${supervisor_environment[@]}" "$system_python" -S -I /dev/fd/3 \
        "$source_sha" "$output_path" "$canonical_branch" "$approved_gitlab_fetch_url" \
        "$verifier_path" "$source_root" "$SLURM_JOB_ID" "$job_user" \
        "${SLURM_TMPDIR:-}" "$node_local_root" "$system_python" "$system_git" \
        "$system_ssh" "${HOME:-}" "${USER:-}" "${LOGNAME:-}" \
        "${SSH_AUTH_SOCK:-}" "$test_executables" 3<<'PY'
import errno
import fcntl
import hashlib
import os
import re
import secrets
import stat
import subprocess
import sys
import tempfile

TEST_ENVIRONMENT_NAMES = (
    "APPROVED_TEST_GIT_URL",
    "BARE_REPOSITORY_INODE_EVIDENCE",
    "BARE_REPOSITORY_PATH",
    "DISABLE_APPROVED_URL_REWRITE",
    "ESCAPED_WRITE_MARKER",
    "GIT_CALLS",
    "GIT_ENV_EVIDENCE",
    "INJECT_AUTH_TREE_RECORD",
    "INJECT_BARE_REPOSITORY_REPLACEMENT",
    "INJECT_MANIFEST_PATH_SWAP",
    "INJECT_SCRATCH_REPLACEMENT",
    "INJECT_STAGE_ENTRY",
    "MANIFEST_SWAP_MARKER",
    "PROBE_SSH_COMMAND",
    "REAL_GIT",
    "REAL_PYTHON",
    "SCRATCH_REPLACEMENT_POINTER",
    "SSH_CALLS",
    "SSH_REDIRECT_MARKER",
    "TEST_GIT_FETCH_URL",
    "TEST_TREE_DUPLICATE_PATH",
    "TEST_TREE_OBJECT_ID",
)
git_executable = ""
ssh_executable = ""
identity_environment: dict[str, str] = {}
test_mode = False


def fail(message: str) -> None:
    raise RuntimeError(message)


def directory_flags() -> int:
    return os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC


def identity(metadata: os.stat_result) -> tuple[int, int]:
    return metadata.st_dev, metadata.st_ino


def open_absolute_directory(path: str) -> int:
    path = path.rstrip("/")
    if not path.startswith("/") or path == "/" or "//" in path:
        fail("scratch root is not canonical")
    components = path.removeprefix("/").split("/")
    if any(component in {"", ".", ".."} for component in components):
        fail("scratch root has an unsafe component")
    descriptor = os.open("/", directory_flags())
    try:
        for component in components:
            next_descriptor = os.open(component, directory_flags(), dir_fd=descriptor)
            os.close(descriptor)
            descriptor = next_descriptor
        metadata = os.fstat(descriptor)
        current = os.stat(path, follow_symlinks=False)
        if not stat.S_ISDIR(metadata.st_mode) or identity(metadata) != identity(current):
            fail("scratch root identity changed")
        return descriptor
    except BaseException:
        os.close(descriptor)
        raise


def open_or_create_user_root(root_descriptor: int, user_name: str) -> int:
    try:
        os.mkdir(user_name, mode=0o700, dir_fd=root_descriptor)
    except FileExistsError:
        pass
    descriptor = os.open(user_name, directory_flags(), dir_fd=root_descriptor)
    metadata = os.fstat(descriptor)
    current = os.stat(user_name, dir_fd=root_descriptor, follow_symlinks=False)
    if (
        metadata.st_uid != os.getuid()
        or stat.S_IMODE(metadata.st_mode) != 0o700
        or identity(metadata) != identity(current)
    ):
        os.close(descriptor)
        fail("node-local user scratch is not stable owned mode 0700")
    return descriptor


def create_private_root(parent_descriptor: int, job_id: str) -> int:
    for _ in range(64):
        name = f"q30t-transfer-verifier.{job_id}.{secrets.token_hex(8)}"
        try:
            os.mkdir(name, mode=0o700, dir_fd=parent_descriptor)
        except FileExistsError:
            continue
        descriptor = os.open(name, directory_flags(), dir_fd=parent_descriptor)
        metadata = os.fstat(descriptor)
        current = os.stat(name, dir_fd=parent_descriptor, follow_symlinks=False)
        if (
            metadata.st_uid != os.getuid()
            or stat.S_IMODE(metadata.st_mode) != 0o700
            or identity(metadata) != identity(current)
        ):
            os.close(descriptor)
            fail("private verifier scratch is not stable owned mode 0700")
        return descriptor
    fail("exclusive private verifier scratch creation failed")


def git_environment() -> dict[str, str]:
    environment = {
        "LC_ALL": "C",
        "GIT_CONFIG_NOSYSTEM": "1",
        "GIT_CONFIG_GLOBAL": "/dev/null",
        "GIT_NO_REPLACE_OBJECTS": "1",
        "GIT_SSH_COMMAND": f"{ssh_executable} -F /dev/null",
        "GIT_SSH_VARIANT": "ssh",
    }
    environment.update(identity_environment)
    if test_mode:
        for name in TEST_ENVIRONMENT_NAMES:
            if name in os.environ:
                environment[name] = os.environ[name]
    return environment


def git(arguments: list[str], *, stderr: int | None = None) -> bytes:
    result = subprocess.run(
        [git_executable, *arguments],
        env=git_environment(),
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE if stderr is None else stderr,
    )
    if result.returncode != 0:
        detail = result.stderr.decode(errors="replace").strip() if result.stderr else "no stderr"
        fail(f"sterile Git command failed: {arguments[0]}: {detail}")
    return result.stdout


def parse_tree_record(record: bytes, expected_path: str) -> str:
    if record.count(b"\n") != 1 or b"\t" not in record:
        fail("authenticated verifier tree lookup is not singular")
    metadata, path = record.rstrip(b"\n").split(b"\t", 1)
    fields = metadata.split()
    if len(fields) != 3 or fields[:2] != [b"100644", b"blob"]:
        fail("authenticated verifier is not a regular Git blob")
    if path != expected_path.encode() or not re.fullmatch(rb"[0-9a-f]{40}", fields[2]):
        fail("authenticated verifier tree entry is invalid")
    return fields[2].decode()


def seal_payload(payload: bytes, expected_object: str) -> int:
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
            fcntl.F_SEAL_WRITE
            | fcntl.F_SEAL_GROW
            | fcntl.F_SEAL_SHRINK
            | fcntl.F_SEAL_SEAL
        )
        fcntl.fcntl(descriptor, fcntl.F_ADD_SEALS, required_seals)
        if fcntl.fcntl(descriptor, fcntl.F_GET_SEALS) & required_seals != required_seals:
            fail("sealed verifier is missing required Linux memfd seals")
        for candidate in (descriptor, os.open(f"/proc/self/fd/{descriptor}", os.O_RDWR)):
            try:
                os.write(candidate, b"x")
            except OSError as error:
                if error.errno != errno.EPERM:
                    fail(f"sealed verifier write failed with errno {error.errno}")
            else:
                fail("sealed verifier accepted a write")
            finally:
                if candidate != descriptor:
                    os.close(candidate)
        print(
            "sealed-verifier-write-check: inplace=EPERM reopened-write=EPERM",
            file=sys.stderr,
        )
    else:
        print(f"sealed-verifier-write-check: test-fallback={sys.platform}", file=sys.stderr)
    os.lseek(descriptor, 0, os.SEEK_SET)
    return descriptor


def main() -> None:
    global git_executable, identity_environment, ssh_executable, test_mode
    (
        source_sha,
        output_path,
        canonical_branch,
        fetch_url,
        verifier_path,
        source_root,
        job_id,
        user_name,
        slurm_tmpdir,
        node_local_root,
        python_executable,
        git_executable,
        ssh_executable,
        home,
        user,
        logname,
        ssh_auth_sock,
        test_mode_text,
    ) = sys.argv[1:]
    test_mode = test_mode_text == "1"
    if sys.platform == "linux" and test_mode:
        fail("system executable test override is forbidden on Linux")
    identity_environment = {
        name: value
        for name, value in {
            "HOME": home,
            "USER": user,
            "LOGNAME": logname,
            "SSH_AUTH_SOCK": ssh_auth_sock,
        }.items()
        if value
    }
    os.umask(0o077)
    if slurm_tmpdir:
        parent_descriptor = open_absolute_directory(slurm_tmpdir)
    else:
        root_descriptor = open_absolute_directory(node_local_root)
        root_metadata = os.fstat(root_descriptor)
        mode = stat.S_IMODE(root_metadata.st_mode)
        if root_metadata.st_uid not in {0, os.getuid()} or (
            mode & 0o022 and not mode & stat.S_ISVTX
        ):
            fail("node-local scratch root is not trusted")
        parent_descriptor = open_or_create_user_root(root_descriptor, user_name)
        os.close(root_descriptor)
    job_descriptor = create_private_root(parent_descriptor, job_id)
    os.close(parent_descriptor)
    os.fchdir(job_descriptor)
    os.mkdir("repository.git", mode=0o700)
    repository_descriptor = os.open("repository.git", directory_flags())
    repository_metadata = os.fstat(repository_descriptor)
    if repository_metadata.st_uid != os.getuid():
        fail("sterile Git repository is not job-owned")
    os.fchdir(repository_descriptor)
    git(["init", "--bare", "--template=/dev/null", "."])
    isolated_ref = f"refs/codex-transfer-verifier/{job_id}"
    git(
        [
            "--git-dir=.",
            "fetch",
            "--filter=blob:none",
            "--depth=1",
            "--no-tags",
            "--no-write-fetch-head",
            "--force",
            fetch_url,
            f"+{canonical_branch}:{isolated_ref}",
        ],
        stderr=subprocess.DEVNULL,
    )
    fetched_commit = git(
        ["--git-dir=.", "rev-parse", "--verify", f"{isolated_ref}^{{commit}}"],
        stderr=subprocess.DEVNULL,
    ).decode().strip()
    if fetched_commit != source_sha:
        fail("requested commit does not match the freshly fetched canonical commit")
    verifier_object = parse_tree_record(
        git(["--git-dir=.", "ls-tree", fetched_commit, "--", verifier_path]), verifier_path
    )
    size_text = git(["--git-dir=.", "cat-file", "-s", verifier_object]).decode().strip()
    if not size_text.isdecimal() or not 0 < int(size_text) <= 8_388_608:
        fail("authenticated verifier size is invalid")
    payload = git(["--git-dir=.", "cat-file", "blob", verifier_object])
    if len(payload) != int(size_text):
        fail("authenticated verifier blob length mismatch")
    descriptor = seal_payload(payload, verifier_object)
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
    environment = {
        "LC_ALL": "C",
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONSAFEPATH": "1",
    }
    if test_mode:
        for name in ("FORBIDDEN_FD", "PYTHON_CALLS", "REAL_PYTHON"):
            if name in os.environ:
                environment[name] = os.environ[name]
    verifier = f"/dev/fd/{descriptor}"
    os.execve(
        python_executable,
        [
            python_executable,
            "-S",
            "-I",
            verifier,
            "--source-root",
            source_root,
            "--output",
            output_path,
            "--workers",
            "16",
        ],
        environment,
    )

try:
    main()
except Exception as error:
    print(f"transfer verifier supervisor failed: {error}", file=sys.stderr)
    raise SystemExit(2)
PY
