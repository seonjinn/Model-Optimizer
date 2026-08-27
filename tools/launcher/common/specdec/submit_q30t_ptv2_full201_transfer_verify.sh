#!/bin/bash
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

readonly approved_python="/usr/bin/python3"
readonly approved_git="/usr/bin/git"
readonly approved_sbatch="/usr/bin/sbatch"
readonly approved_ssh="/usr/bin/ssh"
system_python="$approved_python"
system_git="$approved_git"
system_sbatch="$approved_sbatch"
system_ssh="$approved_ssh"
test_executables=0
if [[ -n "${Q30T_TEST_ALLOW_SYSTEM_EXECUTABLES:-}" \
    || -n "${Q30T_TEST_PYTHON:-}" || -n "${Q30T_TEST_GIT:-}" \
    || -n "${Q30T_TEST_SBATCH:-}" || -n "${Q30T_TEST_SSH:-}" ]]; then
    [[ "$OSTYPE" != linux* \
        && "${Q30T_TEST_ALLOW_SYSTEM_EXECUTABLES:-}" == non-linux-test \
        && "${Q30T_TEST_PYTHON:-}" == /* && "${Q30T_TEST_GIT:-}" == /* \
        && "${Q30T_TEST_SBATCH:-}" == /* && "${Q30T_TEST_SSH:-}" == /* ]] || {
        echo "system executable test override is forbidden" >&2
        exit 2
    }
    system_python="$Q30T_TEST_PYTHON"
    system_git="$Q30T_TEST_GIT"
    system_sbatch="$Q30T_TEST_SBATCH"
    system_ssh="$Q30T_TEST_SSH"
    test_executables=1
fi
for executable in "$system_python" "$system_git" "$system_sbatch" "$system_ssh"; do
    [[ -f "$executable" && -x "$executable" ]] || {
        echo "approved system executable is unavailable" >&2
        exit 2
    }
done

readonly submit_test_environment_names=(
    APPROVED_TEST_GIT_URL BARE_REPOSITORY_INODE_EVIDENCE BARE_REPOSITORY_PATH
    INJECT_BARE_REPOSITORY_REPLACEMENT INJECT_CLEAN_STATUS
    INJECT_SCRATCH_REPLACEMENT INJECT_SOURCE_CHECKOUT_REPLACEMENT REAL_GIT
    PROBE_SSH_COMMAND REAL_PYTHON SBATCH_CALLS SBATCH_ENV_EVIDENCE SBATCH_RUNNER_EVIDENCE
    SCRATCH_REPLACEMENT_POINTER SOURCE_CHECKOUT_PATH SOURCE_REPLACEMENT_MARKER
    SPOOLED_RUNNER SSH_CALLS SSH_REDIRECT_MARKER SUBMIT_GIT_ENV_EVIDENCE
    TEST_GIT_FETCH_URL
)

sterile_parent="${TMPDIR:-/tmp}"
[[ "$sterile_parent" == /* ]] || {
    echo "sterile submitter parent is not absolute" >&2
    exit 2
}

supervisor_environment=(
    /usr/bin/env -i
    LC_ALL=C
    PYTHONDONTWRITEBYTECODE=1
    PYTHONSAFEPATH=1
)
if ((test_executables)); then
    for name in "${submit_test_environment_names[@]}" Q30T_TEST_ALLOW_UNSEALED_PLATFORM; do
        if declare -p "$name" &>/dev/null; then
            supervisor_environment+=("$name=${!name}")
        fi
    done
fi

"${supervisor_environment[@]}" "$system_python" -S -I /dev/fd/3 \
        "$sterile_parent" "$source_path" "$mode" "$slurm_output" "$output_path" \
        "$canonical_remote" "$canonical_branch" "$approved_gitlab_fetch_url" "$runner_path" \
        "$system_git" "$system_sbatch" "$system_ssh" "${HOME:-}" "${USER:-}" \
        "${LOGNAME:-}" "${SSH_AUTH_SOCK:-}" "$test_executables" 3<<'PY'
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

TEST_GIT_ENVIRONMENT_NAMES = (
    "APPROVED_TEST_GIT_URL",
    "BARE_REPOSITORY_INODE_EVIDENCE",
    "BARE_REPOSITORY_PATH",
    "INJECT_BARE_REPOSITORY_REPLACEMENT",
    "INJECT_CLEAN_STATUS",
    "INJECT_SCRATCH_REPLACEMENT",
    "INJECT_SOURCE_CHECKOUT_REPLACEMENT",
    "PROBE_SSH_COMMAND",
    "REAL_GIT",
    "REAL_PYTHON",
    "SCRATCH_REPLACEMENT_POINTER",
    "SOURCE_CHECKOUT_PATH",
    "SOURCE_REPLACEMENT_MARKER",
    "SSH_CALLS",
    "SSH_REDIRECT_MARKER",
    "SUBMIT_GIT_ENV_EVIDENCE",
    "TEST_GIT_FETCH_URL",
)
git_executable = ""
sbatch_executable = ""
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
        fail("submitter scratch root is not canonical")
    components = path.removeprefix("/").split("/")
    if any(component in {"", ".", ".."} for component in components):
        fail("submitter scratch root has an unsafe component")
    descriptor = os.open("/", directory_flags())
    try:
        for component in components:
            next_descriptor = os.open(component, directory_flags(), dir_fd=descriptor)
            os.close(descriptor)
            descriptor = next_descriptor
        metadata = os.fstat(descriptor)
        current = os.stat(path, follow_symlinks=False)
        if not stat.S_ISDIR(metadata.st_mode) or identity(metadata) != identity(current):
            fail("submitter scratch root identity changed")
        return descriptor
    except BaseException:
        os.close(descriptor)
        raise


def create_private_root(parent_descriptor: int) -> int:
    for _ in range(64):
        name = f"q30t-transfer-submit.{secrets.token_hex(8)}"
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
            fail("private submitter scratch is not stable owned mode 0700")
        return descriptor
    fail("exclusive private submitter scratch creation failed")


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
        for name in TEST_GIT_ENVIRONMENT_NAMES:
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


def git_configuration_exists(arguments: list[str]) -> bool:
    result = subprocess.run(
        [git_executable, *arguments],
        env=git_environment(),
        check=False,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    if result.returncode not in {0, 1}:
        fail(f"source Git configuration inspection failed: {arguments[1]}")
    return result.returncode == 0


def read_worktree_blob(root_descriptor: int, path_bytes: bytes, mode: bytes) -> bytes:
    path = os.fsdecode(path_bytes)
    components = path.split("/")
    if not path or any(component in {"", ".", ".."} for component in components):
        fail("tracked path is unsafe")
    descriptor = os.dup(root_descriptor)
    try:
        for component in components[:-1]:
            next_descriptor = os.open(component, directory_flags(), dir_fd=descriptor)
            os.close(descriptor)
            descriptor = next_descriptor
        leaf = components[-1]
        metadata = os.stat(leaf, dir_fd=descriptor, follow_symlinks=False)
        if mode == b"120000":
            if not stat.S_ISLNK(metadata.st_mode):
                fail(f"tracked bytes differ from HEAD: {path}")
            return os.fsencode(os.readlink(leaf, dir_fd=descriptor))
        if mode not in {b"100644", b"100755"} or not stat.S_ISREG(metadata.st_mode):
            fail(f"tracked bytes differ from HEAD: {path}")
        file_descriptor = os.open(
            leaf,
            os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC,
            dir_fd=descriptor,
        )
        try:
            opened = os.fstat(file_descriptor)
            if identity(opened) != identity(metadata) or not stat.S_ISREG(opened.st_mode):
                fail(f"tracked bytes changed while opening: {path}")
            chunks: list[bytes] = []
            while chunk := os.read(file_descriptor, 8 * 1024 * 1024):
                chunks.append(chunk)
            if identity(os.fstat(file_descriptor)) != identity(opened):
                fail(f"tracked bytes changed while reading: {path}")
            rebound = os.stat(leaf, dir_fd=descriptor, follow_symlinks=False)
            if identity(rebound) != identity(opened):
                fail(f"tracked bytes changed while reading: {path}")
            return b"".join(chunks)
        finally:
            os.close(file_descriptor)
    except OSError as error:
        fail(f"tracked bytes are unavailable: {path}: {error}")
    finally:
        os.close(descriptor)


def authenticate_gitlink(root_descriptor: int, path_bytes: bytes, expected_commit: bytes) -> None:
    path = os.fsdecode(path_bytes)
    components = path.split("/")
    if not path or any(component in {"", ".", ".."} for component in components):
        fail("tracked gitlink path is unsafe")
    descriptor = os.dup(root_descriptor)
    try:
        for component in components:
            next_descriptor = os.open(component, directory_flags(), dir_fd=descriptor)
            os.close(descriptor)
            descriptor = next_descriptor
        metadata = os.fstat(descriptor)
        if not stat.S_ISDIR(metadata.st_mode):
            fail(f"tracked gitlink is not a directory: {path}")
        os.fchdir(descriptor)
        if git(["rev-parse", "--is-inside-work-tree"]).strip() != b"true" or git(
            ["rev-parse", "--show-prefix"]
        ) not in {b"", b"\n"}:
            fail(f"tracked gitlink is not a worktree root: {path}")
        if git(["rev-parse", "--show-object-format"]).strip() != b"sha1":
            fail(f"tracked gitlink uses an unsupported object format: {path}")
        observed_commit = git(["rev-parse", "--verify", "HEAD^{commit}"]).strip()
        if observed_commit != expected_commit:
            fail(f"tracked gitlink commit differs from HEAD: {path}")
    except OSError as error:
        fail(f"tracked gitlink is unavailable: {path}: {error}")
    finally:
        os.fchdir(root_descriptor)
        os.close(descriptor)


def require_tracked_bytes_match_head(root_descriptor: int, source_sha: str) -> None:
    records = git(["ls-tree", "-rz", "--full-tree", source_sha]).split(b"\0")
    for record in records:
        if not record:
            continue
        if b"\t" not in record:
            fail("HEAD tree record is malformed")
        metadata, path_bytes = record.split(b"\t", 1)
        fields = metadata.split()
        if len(fields) != 3 or not re.fullmatch(rb"[0-9a-f]{40}", fields[2]):
            fail("HEAD contains an unsupported tracked object")
        if fields[:2] == [b"160000", b"commit"]:
            authenticate_gitlink(root_descriptor, path_bytes, fields[2])
            continue
        if fields[1] != b"blob":
            fail("HEAD contains an unsupported tracked object")
        payload = read_worktree_blob(root_descriptor, path_bytes, fields[0])
        digest = hashlib.sha1()
        digest.update(f"blob {len(payload)}\0".encode("ascii"))
        digest.update(payload)
        if digest.hexdigest().encode() != fields[2]:
            fail(f"tracked bytes differ from HEAD: {os.fsdecode(path_bytes)}")


def inspect_source_checkout(
    source_path: str,
    canonical_remote: str,
    approved_fetch_url: str,
) -> tuple[int, str]:
    source_descriptor = open_absolute_directory(source_path)
    os.fchdir(source_descriptor)
    if git(["rev-parse", "--is-inside-work-tree"]).strip() != b"true" or git(
        ["rev-parse", "--show-prefix"]
    ) not in {b"", b"\n"}:
        fail("SOURCE_PATH is not the Git worktree root")
    if git(["rev-parse", "--show-object-format"]).strip() != b"sha1":
        fail("source checkout does not use approved SHA-1 Git objects")
    index_records = git(["ls-files", "-v", "-z"]).split(b"\0")
    for record in index_records:
        if not record:
            continue
        if len(record) < 3 or record[1:2] != b" ":
            fail("source index record is malformed")
        tag = record[:1]
        if tag == b"S" or 97 <= tag[0] <= 122:
            fail("source checkout contains a hidden index flag")
    source_sha = git(["rev-parse", "--verify", "HEAD^{commit}"]).decode().strip()
    configured_remote_url = git(
        ["config", "--local", "--get-all", f"remote.{canonical_remote}.url"]
    ).decode().strip()
    if configured_remote_url != approved_fetch_url:
        fail("gitlab remote does not use the approved GitLab fetch URL")
    override_pattern = r"^(url\..*\.insteadof|core\.sshcommand)$"
    if git_configuration_exists(
        ["config", "--local", "--get-regexp", override_pattern]
    ) or git_configuration_exists(
        ["config", "--worktree", "--get-regexp", override_pattern]
    ):
        fail("source checkout contains a disallowed Git transport override")
    require_tracked_bytes_match_head(source_descriptor, source_sha)
    if git(
        ["-c", "core.fsmonitor=false", "status", "--porcelain=v1", "--untracked-files=all"]
    ):
        fail("transfer verifier submission requires a clean checkout")
    final_head = git(["rev-parse", "--verify", "HEAD^{commit}"]).decode().strip()
    if final_head != source_sha:
        fail("source HEAD changed during clean-checkout inspection")
    final_index_records = git(["ls-files", "-v", "-z"])
    if final_index_records != b"\0".join(index_records):
        fail("source index changed during clean-checkout inspection")
    return source_descriptor, source_sha


def parse_tree_record(record: bytes, expected_path: str) -> str:
    if record.count(b"\n") != 1 or b"\t" not in record:
        fail("authenticated runner tree lookup is not singular")
    metadata, path = record.rstrip(b"\n").split(b"\t", 1)
    fields = metadata.split()
    if len(fields) != 3 or fields[:2] != [b"100644", b"blob"]:
        fail("authenticated runner is not a regular Git blob")
    if path != expected_path.encode() or not re.fullmatch(rb"[0-9a-f]{40}", fields[2]):
        fail("authenticated runner tree entry is invalid")
    return fields[2].decode()


def seal_runner(payload: bytes, expected_object: str) -> tuple[int, str]:
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
            fcntl.F_SEAL_WRITE
            | fcntl.F_SEAL_GROW
            | fcntl.F_SEAL_SHRINK
            | fcntl.F_SEAL_SEAL
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
    if hashlib.sha256(os.read(descriptor, len(payload) + 1)).digest() != hashlib.sha256(
        payload
    ).digest():
        fail("sealed runner reread mismatch")
    os.lseek(descriptor, 0, os.SEEK_SET)
    return descriptor, runner_path


def invoke_sbatch(
    scheduler_mode: str,
    payload: bytes,
    runner_object: str,
    slurm_output: str,
    source_sha: str,
    output_path: str,
) -> bytes:
    descriptor, runner_path = seal_runner(payload, runner_object)
    try:
        command = [
            sbatch_executable,
            scheduler_mode,
            "--nodes=1",
            "--cpus-per-task=32",
            "--job-name=q30t-ptv2-full201-transfer-verify",
            f"--output={slurm_output}",
            "--export=NONE",
            runner_path,
            source_sha,
            output_path,
        ]
        environment = {"LC_ALL": "C"}
        if test_mode:
            for name in (
                "REAL_PYTHON",
                "SBATCH_CALLS",
                "SBATCH_ENV_EVIDENCE",
                "SBATCH_RUNNER_EVIDENCE",
                "SPOOLED_RUNNER",
            ):
                if name in os.environ:
                    environment[name] = os.environ[name]
        result = subprocess.run(
            command,
            env=environment,
            pass_fds=(descriptor,),
            check=False,
            capture_output=True,
        )
    finally:
        os.close(descriptor)
    if result.returncode != 0:
        sys.stderr.buffer.write(result.stderr)
        raise SystemExit(result.returncode)
    return result.stdout


def main() -> None:
    global git_executable, identity_environment, sbatch_executable, ssh_executable, test_mode
    (
        scratch_parent,
        source_path,
        mode,
        slurm_output,
        output_path,
        canonical_remote,
        canonical_branch,
        fetch_url,
        runner_path,
        git_executable,
        sbatch_executable,
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
    source_descriptor, source_sha = inspect_source_checkout(
        source_path,
        canonical_remote,
        fetch_url,
    )
    parent_descriptor = open_absolute_directory(scratch_parent)
    parent_metadata = os.fstat(parent_descriptor)
    parent_mode = stat.S_IMODE(parent_metadata.st_mode)
    if parent_metadata.st_uid not in {0, os.getuid()} or (
        parent_mode & 0o022 and not parent_mode & stat.S_ISVTX
    ):
        fail("submitter scratch parent is not trusted")
    scratch_descriptor = create_private_root(parent_descriptor)
    os.close(parent_descriptor)
    os.fchdir(scratch_descriptor)
    os.close(source_descriptor)
    os.mkdir("repository.git", mode=0o700)
    repository_descriptor = os.open("repository.git", directory_flags())
    if os.fstat(repository_descriptor).st_uid != os.getuid():
        fail("sterile Git repository is not submitter-owned")
    os.fchdir(repository_descriptor)
    git(["init", "--bare", "--template=/dev/null", "."])
    isolated_ref = "refs/codex-transfer-submitter/verified"
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
        fail("source HEAD does not match the freshly fetched canonical commit")
    runner_object = parse_tree_record(
        git(["--git-dir=.", "ls-tree", fetched_commit, "--", runner_path]), runner_path
    )
    size_text = git(["--git-dir=.", "cat-file", "-s", runner_object]).decode().strip()
    if not size_text.isdecimal() or not 0 < int(size_text) <= 8_388_608:
        fail("authenticated runner size is invalid")
    payload = git(["--git-dir=.", "cat-file", "blob", runner_object])
    if len(payload) != int(size_text):
        fail("authenticated runner blob length mismatch")
    invoke_sbatch("--test-only", payload, runner_object, slurm_output, source_sha, output_path)
    if mode == "--test-only":
        print("test-only passed")
        return
    job_id = invoke_sbatch(
        "--parsable", payload, runner_object, slurm_output, source_sha, output_path
    ).decode().strip()
    if not re.fullmatch(r"[0-9]+", job_id):
        fail("invalid sbatch job ID")
    print(job_id)


try:
    main()
except Exception as error:
    print(f"transfer submitter supervisor failed: {error}", file=sys.stderr)
    raise SystemExit(2)
PY
