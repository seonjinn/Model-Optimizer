#!/bin/bash -p
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

readonly approved_git="/usr/bin/git"
readonly approved_id="/usr/bin/id"
readonly approved_mkdir="/usr/bin/mkdir"
readonly approved_python="/usr/bin/python3.12"
readonly approved_env="/usr/bin/env"
[[ "$-" == *p* ]] || fail "privileged Bash mode is required"
for name in $(compgen -e); do
    [[ "$name" != BASH_FUNC_* && "$name" != LD_* && "$name" != DYLD_* \
        && "$name" != BASH_ENV && "$name" != ENV && "$name" != CDPATH \
        && "$name" != PYTHONPATH && "$name" != PYTHONHOME ]] \
        || fail "unsafe inherited environment variable: $name"
done
unset PATH
PATH=/usr/bin:/bin
export PATH

"$approved_python" -I -S - /bin/bash "$approved_env" "$approved_git" \
    "$approved_id" "$approved_mkdir" "$approved_python" <<'PY'
# Q30T_TOOL_AUTHENTICATOR
import os
import stat
import sys

arguments = sys.argv[1:]
test_owner = arguments[:1] == ["--test-owner"]
if test_owner:
    arguments = arguments[1:]
required_uid = os.getuid() if test_owner else 0
for value in arguments:
    path = value
    if not os.path.isabs(path):
        raise SystemExit("approved executable path is not absolute")
    for _ in range(16):
        metadata = os.lstat(path)
        if metadata.st_uid != required_uid or stat.S_IMODE(metadata.st_mode) & 0o022:
            raise SystemExit(f"approved executable is not owned safely or is writable: {value}")
        if not stat.S_ISLNK(metadata.st_mode):
            if not stat.S_ISREG(metadata.st_mode) or not os.access(path, os.X_OK):
                raise SystemExit(f"approved executable is not regular and executable: {value}")
            break
        target = os.readlink(path)
        path = os.path.normpath(target if os.path.isabs(target) else os.path.join(os.path.dirname(path), target))
    else:
        raise SystemExit(f"approved executable symlink chain is too deep: {value}")
if not test_owner and sys.version_info < (3, 12):
    raise SystemExit("approved Python must be Python 3.12 or newer")
PY

scratch_user="$(/usr/bin/env -i LC_ALL=C "$approved_id" -un)"
readonly scratch_user
[[ "$scratch_user" =~ ^[a-z_][a-z0-9_-]{0,31}$ ]] \
    || fail "approved scratch user name is invalid"
scratch_record="$(/usr/bin/env -i LC_ALL=C PYTHONDONTWRITEBYTECODE=1 PYTHONSAFEPATH=1 \
    "$approved_python" -I -S - "$scratch_user" "$SLURM_JOB_ID" "${SLURM_TMPDIR:-}" <<'PY'
# Q30T_SCRATCH_ALLOCATOR
import os
import secrets
import stat
import sys

scratch_root_path = "/raid/scratch"
required_root_uids = {0}
effective_group_ids = {os.getegid(), *os.getgroups()}
open_flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW


def fail(message):
    raise SystemExit(message)


def identity(metadata):
    return (
        metadata.st_dev,
        metadata.st_ino,
        metadata.st_mode,
        metadata.st_nlink,
        metadata.st_uid,
        metadata.st_gid,
        metadata.st_ctime_ns,
    )


def require_root_component(metadata, label, *, shared_root=False):
    mode = stat.S_IMODE(metadata.st_mode)
    if not stat.S_ISDIR(metadata.st_mode) or metadata.st_uid not in required_root_uids:
        fail(f"{label} is not an approved root-owned directory")
    if shared_root:
        if metadata.st_gid != 0 or mode != 0o1777:
            fail(f"{label} has unsafe write permissions")
        return
    safe_root_group_write = (
        mode & 0o020
        and metadata.st_uid == 0
        and metadata.st_gid == 0
        and 0 not in effective_group_ids
    )
    if mode & 0o002 or mode & 0o020 and not safe_root_group_write:
        fail(f"{label} has unsafe write permissions")


def require_private_directory(metadata, label, owner_uid):
    if (
        not stat.S_ISDIR(metadata.st_mode)
        or metadata.st_uid != owner_uid
        or stat.S_IMODE(metadata.st_mode) != 0o700
    ):
        fail(f"{label} is not an exact-owner mode-0700 directory")


def require_named_binding(parent_fd, name, descriptor, label):
    try:
        named = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
        opened = os.fstat(descriptor)
    except OSError as error:
        fail(f"{label} namespace binding is unavailable: {error}")
    if identity(named) != identity(opened):
        fail(f"{label} namespace changed")
    return opened


def open_fixed_root(path):
    if not os.path.isabs(path) or os.path.normpath(path) != path:
        fail("scratch root path is invalid")
    components = [component for component in path.split("/") if component]
    if not components:
        fail("scratch root path has no components")
    descriptor = os.open("/", open_flags)
    try:
        require_root_component(os.fstat(descriptor), "scratch root ancestor")
        for ordinal, component in enumerate(components):
            try:
                child = os.open(component, open_flags, dir_fd=descriptor)
            except OSError as error:
                fail(f"scratch root component is unavailable: {error}")
            metadata = os.fstat(child)
            require_root_component(
                metadata,
                "scratch root" if ordinal == len(components) - 1 else "scratch root ancestor",
                shared_root=ordinal == len(components) - 1,
            )
            if ordinal == len(components) - 1:
                require_named_binding(descriptor, component, child, "scratch root")
                return descriptor, child, component
            os.close(descriptor)
            descriptor = child
    except BaseException:
        os.close(descriptor)
        raise
    fail("scratch root traversal failed")


def validate_name(value, label, *, username=False):
    if username:
        valid = (
            1 <= len(value) <= 32
            and (value[0].islower() or value[0] == "_")
            and value.isascii()
            and all(
                character.islower() or character.isdigit() or character in "_-"
                for character in value
            )
        )
    else:
        valid = 1 <= len(value) <= 20 and value.isascii() and value.isdecimal()
    if not valid:
        fail(f"{label} is invalid")


def open_private_relative(root_fd, relative, owner_uid, label):
    components = relative.split("/")
    if not components or any(component in {"", ".", ".."} for component in components):
        fail(f"{label} path components are invalid")
    parent_fd = os.dup(root_fd)
    try:
        for ordinal, component in enumerate(components):
            try:
                child = os.open(component, open_flags, dir_fd=parent_fd)
            except OSError as error:
                fail(f"{label} component is unavailable: {error}")
            metadata = os.fstat(child)
            if ordinal == len(components) - 1:
                require_private_directory(metadata, label, owner_uid)
                require_named_binding(parent_fd, component, child, label)
                return parent_fd, child, component
            mode = stat.S_IMODE(metadata.st_mode)
            if (
                not stat.S_ISDIR(metadata.st_mode)
                or metadata.st_uid not in required_root_uids | {owner_uid}
                or mode & 0o022
            ):
                fail(f"{label} ancestor is unsafe")
            os.close(parent_fd)
            parent_fd = child
    except BaseException:
        os.close(parent_fd)
        raise
    fail(f"{label} traversal failed")


def create_or_adopt_user(root_fd, username, owner_uid):
    try:
        os.mkdir(username, 0o700, dir_fd=root_fd)
    except FileExistsError:
        pass
    except OSError as error:
        fail(f"scratch user directory creation failed: {error}")
    return open_private_relative(root_fd, username, owner_uid, "scratch user directory")


def create_private_child(parent_fd, job_id, owner_uid):
    for _ in range(32):
        name = f"q30t-row-schema-{job_id}-{secrets.token_hex(16)}"
        try:
            os.mkdir(name, 0o700, dir_fd=parent_fd)
        except FileExistsError:
            continue
        except OSError as error:
            fail(f"private scratch creation failed: {error}")
        try:
            descriptor = os.open(name, open_flags, dir_fd=parent_fd)
        except OSError as error:
            fail(f"private scratch adoption failed: {error}")
        metadata = require_named_binding(parent_fd, name, descriptor, "private scratch")
        require_private_directory(metadata, "private scratch", owner_uid)
        return descriptor, name
    fail("private scratch random namespace was exhausted")


def main():
    if len(sys.argv) != 4:
        fail("usage: allocator USER JOB_ID SLURM_TMPDIR")
    username, job_id, slurm_tmpdir = sys.argv[1:]
    validate_name(username, "scratch user name", username=True)
    validate_name(job_id, "SLURM job id")
    owner_uid = os.getuid()
    root_parent_fd, root_fd, root_name = open_fixed_root(scratch_root_path)
    private_parent_fd = private_fd = private_name = None
    scratch_fd = None
    try:
        if slurm_tmpdir:
            if (
                not os.path.isabs(slurm_tmpdir)
                or any(ord(character) < 32 for character in slurm_tmpdir)
                or os.path.normpath(slurm_tmpdir) != slurm_tmpdir
                or os.path.commonpath((scratch_root_path, slurm_tmpdir)) != scratch_root_path
                or slurm_tmpdir == scratch_root_path
            ):
                fail("SLURM_TMPDIR is outside approved node scratch")
            relative = os.path.relpath(slurm_tmpdir, scratch_root_path)
            private_parent_fd, private_fd, private_name = open_private_relative(
                root_fd, relative, owner_uid, "SLURM_TMPDIR"
            )
            scratch_parent_fd = private_fd
            display_parent = slurm_tmpdir
        else:
            private_parent_fd, private_fd, private_name = create_or_adopt_user(
                root_fd, username, owner_uid
            )
            scratch_parent_fd = private_fd
            display_parent = os.path.join(scratch_root_path, username)
        scratch_fd, scratch_name = create_private_child(scratch_parent_fd, job_id, owner_uid)
        require_named_binding(root_parent_fd, root_name, root_fd, "scratch root")
        require_named_binding(private_parent_fd, private_name, private_fd, "scratch parent")
        scratch_metadata = require_named_binding(
            scratch_parent_fd, scratch_name, scratch_fd, "private scratch"
        )
        require_private_directory(scratch_metadata, "private scratch", owner_uid)
        print(
            f"{os.path.join(display_parent, scratch_name)}\t"
            f"{scratch_metadata.st_dev}\t{scratch_metadata.st_ino}",
            flush=True,
        )
    finally:
        if scratch_fd is not None:
            os.close(scratch_fd)
        if private_fd is not None:
            os.close(private_fd)
        if private_parent_fd is not None:
            os.close(private_parent_fd)
        os.close(root_fd)
        os.close(root_parent_fd)


main()
PY
)"
readonly scratch_record
IFS=$'\t' read -r scratch_path scratch_device scratch_inode scratch_extra <<<"$scratch_record"
readonly scratch_path scratch_device scratch_inode scratch_extra
[[ "$scratch_path" == /raid/scratch/* && "$scratch_device" =~ ^[0-9]+$ \
    && "$scratch_inode" =~ ^[0-9]+$ && -z "$scratch_extra" ]] \
    || fail "private node-local scratch allocation failed"
exec 9<"$scratch_path" || fail "private node-local scratch adoption failed"
readonly scratch_fd=9
/usr/bin/env -i LC_ALL=C PYTHONDONTWRITEBYTECODE=1 PYTHONSAFEPATH=1 \
    "$approved_python" -I -S - "$scratch_fd" "$scratch_path" \
    "$scratch_device" "$scratch_inode" <<'PY'
# Q30T_SCRATCH_ADOPTER
import os
import stat
import sys

descriptor = int(sys.argv[1])
path = sys.argv[2]
expected = (int(sys.argv[3]), int(sys.argv[4]))
opened = os.fstat(descriptor)
named = os.stat(path, follow_symlinks=False)
if (
    not stat.S_ISDIR(opened.st_mode)
    or opened.st_uid != os.getuid()
    or stat.S_IMODE(opened.st_mode) != 0o700
    or (opened.st_dev, opened.st_ino) != expected
    or (named.st_dev, named.st_ino) != expected
):
    raise SystemExit("private node-local scratch namespace changed before adoption")
PY
readonly scratch_root="/proc/self/fd/$scratch_fd"
[[ -d "$scratch_root" ]] || fail "held private node-local scratch is unavailable"

readonly observer_path="examples/dataset/observe_q30t_ptv23_row_schemas.py"
readonly ptv3_plan_path="examples/dataset/qwen3_30ba3b_thinking_ptv3_stage_subset_v1.json"
readonly atomic_path="tools/launcher/common/specdec/qwen4b_b_atomic.py"
readonly ptv3_plan="$scratch_root/$ptv3_plan_path"
materialized_record="$(/usr/bin/env -i LC_ALL=C PYTHONDONTWRITEBYTECODE=1 PYTHONSAFEPATH=1 \
    "$approved_python" -I -S - "$scratch_fd" "$approved_git" "$source_path" \
    "$source_commit" "$observer_path" "$ptv3_plan_path" "$atomic_path" <<'PY'
# Q30T_BLOB_MATERIALIZER
import hashlib
import os
import stat
import subprocess
import sys

root_fd = int(sys.argv[1])
git, source, commit = sys.argv[2:5]
paths = sys.argv[5:8]
limits = (8 * 1024 * 1024, 8 * 1024 * 1024, 8 * 1024 * 1024)
open_directory_flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
open_output_flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW | os.O_NONBLOCK
git_environment = {
    "LC_ALL": "C",
    "GIT_CONFIG_NOSYSTEM": "1",
    "GIT_CONFIG_SYSTEM": "/dev/null",
    "GIT_CONFIG_GLOBAL": "/dev/null",
    "GIT_NO_REPLACE_OBJECTS": "1",
}
git_prefix = [
    git,
    "-c",
    "core.attributesFile=/dev/null",
    "-c",
    "core.hooksPath=/dev/null",
    "-C",
    source,
]


def fail(message):
    raise SystemExit(message)


def require_private_directory(descriptor, label):
    metadata = os.fstat(descriptor)
    if (
        not stat.S_ISDIR(metadata.st_mode)
        or metadata.st_uid != os.getuid()
        or stat.S_IMODE(metadata.st_mode) != 0o700
    ):
        fail(f"{label} is not an exact-owner mode-0700 directory")
    return metadata


def open_parent(relative):
    if os.path.isabs(relative) or os.path.normpath(relative) != relative:
        fail("materialized blob path is not normalized and relative")
    components = relative.split("/")
    if len(components) < 2 or any(not value or value in (".", "..") for value in components):
        fail("materialized blob path has unsafe components")
    parent = os.dup(root_fd)
    require_private_directory(parent, "materialization root")
    try:
        for component in components[:-1]:
            try:
                os.mkdir(component, 0o700, dir_fd=parent)
            except FileExistsError:
                pass
            child = os.open(component, open_directory_flags, dir_fd=parent)
            metadata = require_private_directory(child, "materialization directory")
            named = os.stat(component, dir_fd=parent, follow_symlinks=False)
            if (metadata.st_dev, metadata.st_ino) != (named.st_dev, named.st_ino):
                os.close(child)
                fail("materialization directory namespace changed")
            os.close(parent)
            parent = child
        return parent, components[-1]
    except BaseException:
        os.close(parent)
        raise


def write_all(descriptor, data):
    offset = 0
    while offset < len(data):
        written = os.write(descriptor, data[offset:])
        if written <= 0:
            fail("materialized blob write made no progress")
        offset += written


def materialize(relative, maximum_bytes):
    parent, name = open_parent(relative)
    descriptor = None
    process = None
    try:
        descriptor = os.open(name, open_output_flags, 0o400, dir_fd=parent)
        opened = os.fstat(descriptor)
        if (
            not stat.S_ISREG(opened.st_mode)
            or opened.st_uid != os.getuid()
            or stat.S_IMODE(opened.st_mode) != 0o400
            or opened.st_nlink != 1
        ):
            fail("new materialized blob has unsafe metadata")
        process = subprocess.Popen(
            [*git_prefix, "cat-file", "blob", f"{commit}:{relative}"],
            env=git_environment,
            stdout=subprocess.PIPE,
        )
        if process.stdout is None:
            fail("Git blob stream is unavailable")
        digest = hashlib.sha256()
        counted = 0
        while True:
            block = process.stdout.read(1024 * 1024)
            if not block:
                break
            if len(block) > maximum_bytes - counted:
                process.kill()
                fail("materialized blob exceeds its byte bound")
            write_all(descriptor, block)
            digest.update(block)
            counted += len(block)
        process.stdout.close()
        if process.wait() != 0:
            fail("Git failed while streaming a reviewed blob")
        os.fsync(descriptor)
        os.fsync(parent)
        after = os.fstat(descriptor)
        named = os.stat(name, dir_fd=parent, follow_symlinks=False)
        if (
            (opened.st_dev, opened.st_ino) != (after.st_dev, after.st_ino)
            or (after.st_dev, after.st_ino) != (named.st_dev, named.st_ino)
            or after.st_size != counted
            or after.st_nlink != 1
            or stat.S_IMODE(after.st_mode) != 0o400
        ):
            fail("materialized blob changed before durable publication")
        return after.st_dev, after.st_ino, counted, digest.hexdigest()
    finally:
        if process is not None and process.poll() is None:
            process.kill()
            process.wait()
        if descriptor is not None:
            os.close(descriptor)
        os.close(parent)


if subprocess.run(
    [*git_prefix, "cat-file", "-e", f"{commit}^{{commit}}"],
    env=git_environment,
    stdin=subprocess.DEVNULL,
    stdout=subprocess.DEVNULL,
).returncode != 0:
    fail("reviewed Git commit object is unavailable")
records = [materialize(path, limit) for path, limit in zip(paths, limits, strict=True)]
print("\t".join(str(value) for record in records for value in record), flush=True)
PY
)"
readonly materialized_record
IFS=$'\t' read -r observer_device observer_inode observer_size observer_sha256 \
    ptv3_plan_device ptv3_plan_inode ptv3_plan_size observed_ptv3_plan_sha256 \
    atomic_device atomic_inode atomic_size atomic_sha256 materialized_extra <<<"$materialized_record"
readonly observer_device observer_inode observer_size observer_sha256 \
    ptv3_plan_device ptv3_plan_inode ptv3_plan_size observed_ptv3_plan_sha256 \
    atomic_device atomic_inode atomic_size atomic_sha256 materialized_extra
[[ "$observer_device" =~ ^[0-9]+$ && "$observer_inode" =~ ^[0-9]+$ \
    && "$observer_size" =~ ^[0-9]+$ && "$observer_sha256" =~ ^[0-9a-f]{64}$ \
    && "$ptv3_plan_device" =~ ^[0-9]+$ && "$ptv3_plan_inode" =~ ^[0-9]+$ \
    && "$ptv3_plan_size" =~ ^[0-9]+$ && "$observed_ptv3_plan_sha256" =~ ^[0-9a-f]{64}$ \
    && "$atomic_device" =~ ^[0-9]+$ && "$atomic_inode" =~ ^[0-9]+$ \
    && "$atomic_size" =~ ^[0-9]+$ && "$atomic_sha256" =~ ^[0-9a-f]{64}$ \
    && -z "$materialized_extra" ]] || fail "materialized blob identity capture failed"

readonly ptv2_root="/lustre/fsw/coreai_dlalgo_llm/users/sna/modelopt-qwen3-drafter-training/assets/q30t-ptv2-full201-source-v1"
readonly ptv2_plan="$ptv2_root/SOURCE_PLAN.json"
readonly ptv2_plan_sha256="96970541d0c6f5c99e74b9222b805d4a0bd2ac682837b0ad92b8bf54f7d71a3a"
readonly ptv2_completion="$ptv2_root/SOURCE_MANIFEST_COMPLETION.json"
readonly ptv2_completion_sha256="018d659170834b17967dd6c1b066e3b03e7859386eceefd9d4409dc3bc8f48c1"
readonly ptv3_root="/lustre/fsw/coreai_dlalgo_llm/users/sna/modelopt-qwen3-drafter-training/assets/q30t-ptv3-source-stage-75209087-v1"
readonly ptv3_plan_sha256="752090878ed2c5fce47683b2939f9d857be5549311b158476fce33bb79f52eac"
readonly ptv3_completion="$ptv3_root/completion.json"

[[ "$observed_ptv3_plan_sha256" == "$ptv3_plan_sha256" ]] \
    || fail "immutable PTV3 plan blob SHA-256 mismatch"
ptv3_completion_sha256="$(/usr/bin/env -i LC_ALL=C PYTHONDONTWRITEBYTECODE=1 \
    PYTHONSAFEPATH=1 "$approved_python" -I -S - "$ptv3_completion" <<'PY'
# Q30T_COMPLETION_BINDER
import hashlib
import os
import stat
import sys

path = sys.argv[1]
before = os.stat(path, follow_symlinks=False)
maximum_bytes = 1024 * 1024
if before.st_size > maximum_bytes:
    raise SystemExit("PTV3 completion exceeds its declared byte bound")
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
    counted = 0
    while counted < before.st_size:
        chunk = os.read(descriptor, min(1024 * 1024, before.st_size - counted))
        if not chunk:
            raise SystemExit("PTV3 completion shrank while binding its SHA-256")
        digest.update(chunk)
        counted += len(chunk)
    if os.read(descriptor, 1):
        raise SystemExit("PTV3 completion grew while binding its SHA-256")
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
    if (
        counted != before.st_size
        or identity(before) != identity(after)
        or identity(after) != identity(current)
    ):
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
    "$approved_python" -I -S - "$scratch_fd" \
    "$observer_path" "$observer_device" "$observer_inode" "$observer_size" "$observer_sha256" \
    "$ptv3_plan_path" "$ptv3_plan_device" "$ptv3_plan_inode" "$ptv3_plan_size" "$observed_ptv3_plan_sha256" \
    "$atomic_path" "$atomic_device" "$atomic_inode" "$atomic_size" "$atomic_sha256" \
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
    --output "$output" <<'PY'
# Q30T_IMMUTABLE_BOOTSTRAP
import hashlib
import os
import stat
import sys

MAX_BLOB_BYTES = 8 * 1024 * 1024
MAX_RUNTIME_FILES = 250_000
MAX_RUNTIME_BYTES = 64 * 1024 * 1024 * 1024


def identity(value):
    return (
        value.st_dev,
        value.st_ino,
        value.st_mode,
        value.st_nlink,
        value.st_size,
        value.st_mtime_ns,
        value.st_ctime_ns,
    )


def open_blob(path, expected_sha256, label):
    if len(expected_sha256) != 64 or any(character not in "0123456789abcdef" for character in expected_sha256):
        raise SystemExit(f"{label} blob SHA-256 is invalid")
    held_descriptor = path.startswith("fd:")
    if held_descriptor:
        try:
            descriptor = os.dup(int(path.removeprefix("fd:")))
        except (OSError, ValueError):
            raise SystemExit(f"{label} held descriptor is invalid") from None
        before = os.fstat(descriptor)
    else:
        before = os.stat(path, follow_symlinks=False)
    if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1 or before.st_size > MAX_BLOB_BYTES:
        raise SystemExit(f"{label} blob is not a bounded single-link regular file")
    if not held_descriptor:
        descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    opened = os.fstat(descriptor)
    if identity(before) != identity(opened):
        os.close(descriptor)
        raise SystemExit(f"{label} blob changed while opening")
    chunks = []
    counted = 0
    digest = hashlib.sha256()
    while counted < before.st_size:
        block = os.read(descriptor, min(1024 * 1024, before.st_size - counted))
        if not block:
            os.close(descriptor)
            raise SystemExit(f"{label} blob shrank while reading")
        chunks.append(block)
        digest.update(block)
        counted += len(block)
    if os.read(descriptor, 1):
        os.close(descriptor)
        raise SystemExit(f"{label} blob grew while reading")
    after = os.fstat(descriptor)
    rebound = after if held_descriptor else os.stat(path, follow_symlinks=False)
    if identity(before) != identity(after) or identity(after) != identity(rebound):
        os.close(descriptor)
        raise SystemExit(f"{label} blob changed while reading")
    if digest.hexdigest() != expected_sha256:
        os.close(descriptor)
        raise SystemExit(f"{label} blob SHA-256 mismatch")
    return descriptor, b"".join(chunks)


def adopt_materialized_blob(
    root_descriptor,
    relative,
    expected_device,
    expected_inode,
    expected_size,
    expected_sha256,
    label,
):
    if os.path.isabs(relative) or os.path.normpath(relative) != relative:
        raise SystemExit(f"{label} materialized path is unsafe")
    components = relative.split("/")
    if len(components) < 2 or any(not value or value in (".", "..") for value in components):
        raise SystemExit(f"{label} materialized path has unsafe components")
    parent = os.dup(root_descriptor)
    descriptor = None
    try:
        for component in components[:-1]:
            child = os.open(
                component,
                os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_NONBLOCK,
                dir_fd=parent,
            )
            os.close(parent)
            parent = child
        descriptor = os.open(
            components[-1],
            os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK,
            dir_fd=parent,
        )
        before = os.fstat(descriptor)
        named = os.stat(components[-1], dir_fd=parent, follow_symlinks=False)
        expected_identity = int(expected_device), int(expected_inode)
        size = int(expected_size)
        if (
            not stat.S_ISREG(before.st_mode)
            or before.st_uid != os.getuid()
            or stat.S_IMODE(before.st_mode) != 0o400
            or before.st_nlink != 1
            or before.st_size != size
            or (before.st_dev, before.st_ino) != expected_identity
            or identity(before) != identity(named)
            or size > MAX_BLOB_BYTES
        ):
            raise SystemExit(f"{label} changed before descriptor adoption")
        chunks = []
        digest = hashlib.sha256()
        counted = 0
        while counted < size:
            block = os.read(descriptor, min(1024 * 1024, size - counted))
            if not block:
                raise SystemExit(f"{label} shrank during descriptor adoption")
            chunks.append(block)
            digest.update(block)
            counted += len(block)
        if os.read(descriptor, 1):
            raise SystemExit(f"{label} grew during descriptor adoption")
        after = os.fstat(descriptor)
        rebound = os.stat(components[-1], dir_fd=parent, follow_symlinks=False)
        if identity(before) != identity(after) or identity(after) != identity(rebound):
            raise SystemExit(f"{label} rebound during descriptor adoption")
        if digest.hexdigest() != expected_sha256:
            raise SystemExit(f"{label} SHA-256 mismatch during descriptor adoption")
        return descriptor, b"".join(chunks)
    except BaseException:
        if descriptor is not None:
            os.close(descriptor)
        raise
    finally:
        os.close(parent)


def require_root_owned_nonwritable(metadata, label):
    if metadata.st_uid != 0 or stat.S_IMODE(metadata.st_mode) & 0o022:
        raise SystemExit(f"{label} is not root-owned and non-writable")


def hash_runtime_file(path, named, label):
    descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    try:
        opened = os.fstat(descriptor)
        if identity(named) != identity(opened):
            raise SystemExit(f"{label} changed while opening")
        digest = hashlib.sha256()
        counted = 0
        while counted < named.st_size:
            block = os.read(descriptor, min(1024 * 1024, named.st_size - counted))
            if not block:
                raise SystemExit(f"{label} shrank while hashing")
            digest.update(block)
            counted += len(block)
        if os.read(descriptor, 1):
            raise SystemExit(f"{label} grew while hashing")
        after = os.fstat(descriptor)
        rebound = os.stat(path, follow_symlinks=False)
        if identity(named) != identity(after) or identity(after) != identity(rebound):
            raise SystemExit(f"{label} changed while hashing")
        return counted, digest.hexdigest()
    finally:
        os.close(descriptor)


def inventory_runtime_tree(root, label):
    root = os.path.normpath(root)
    root_before = os.stat(root, follow_symlinks=False)
    require_root_owned_nonwritable(root_before, f"{label} root")
    if not stat.S_ISDIR(root_before.st_mode):
        raise SystemExit(f"{label} root is not a directory")
    digest = hashlib.sha256()
    file_count = 0
    total_bytes = 0
    for directory, names, filenames in os.walk(root, followlinks=False):
        names.sort()
        filenames.sort()
        directory_metadata = os.stat(directory, follow_symlinks=False)
        require_root_owned_nonwritable(directory_metadata, f"{label} directory")
        if not stat.S_ISDIR(directory_metadata.st_mode):
            raise SystemExit(f"{label} inventory contains a non-directory")
        for name in names:
            child = os.stat(os.path.join(directory, name), follow_symlinks=False)
            if stat.S_ISLNK(child.st_mode):
                raise SystemExit(f"{label} inventory contains a symlink")
        for name in filenames:
            path = os.path.join(directory, name)
            named = os.stat(path, follow_symlinks=False)
            if file_count >= MAX_RUNTIME_FILES or named.st_size > MAX_RUNTIME_BYTES - total_bytes:
                raise SystemExit(f"{label} inventory exceeds its bound")
            require_root_owned_nonwritable(named, f"{label} file")
            if not stat.S_ISREG(named.st_mode) or named.st_nlink < 1:
                raise SystemExit(f"{label} inventory contains an unsafe file")
            size, file_sha256 = hash_runtime_file(path, named, f"{label} file")
            file_count += 1
            total_bytes += size
            relative = os.path.relpath(path, root).encode()
            digest.update(len(relative).to_bytes(4, "big"))
            digest.update(relative)
            digest.update(size.to_bytes(8, "big"))
            digest.update(bytes.fromhex(file_sha256))
    root_after = os.stat(root, follow_symlinks=False)
    if identity(root_before) != identity(root_after) or file_count < 1:
        raise SystemExit(f"{label} root changed while hashing")
    return file_count, total_bytes, digest.hexdigest()


def authenticate_python_runtime():
    if sys.flags.isolated != 1 or sys.flags.no_site != 1:
        raise SystemExit("runtime bootstrap requires Python -I -S")
    if any(name == "pyarrow" or name.startswith("pyarrow.") for name in sys.modules):
        raise SystemExit("runtime bootstrap found preloaded PyArrow")
    executable = sys.executable
    if not os.path.isabs(executable):
        raise SystemExit("runtime Python executable is not absolute")
    target = executable
    for _ in range(16):
        metadata = os.lstat(target)
        require_root_owned_nonwritable(metadata, "runtime Python executable")
        if not stat.S_ISLNK(metadata.st_mode):
            if not stat.S_ISREG(metadata.st_mode) or not os.access(target, os.X_OK):
                raise SystemExit("runtime Python executable is not regular and executable")
            break
        link = os.readlink(target)
        target = os.path.normpath(link if os.path.isabs(link) else os.path.join(os.path.dirname(target), link))
    else:
        raise SystemExit("runtime Python executable symlink chain is too deep")
    running = os.stat("/proc/self/exe")
    target_metadata = os.stat(target, follow_symlinks=False)
    if (running.st_dev, running.st_ino) != (target_metadata.st_dev, target_metadata.st_ino):
        raise SystemExit("runtime Python differs from its approved executable")
    python_bytes, python_sha256 = hash_runtime_file(target, target_metadata, "runtime Python executable")
    import sysconfig

    paths = sysconfig.get_paths()
    stdlib_root = paths.get("stdlib")
    if not stdlib_root or not os.path.isabs(stdlib_root):
        raise SystemExit("runtime Python stdlib root is unavailable")
    candidates = {
        value for key in ("purelib", "platlib") if (value := paths.get(key)) is not None
    }
    site_roots = sorted(
        value for value in candidates if os.path.isabs(value) and os.path.isdir(os.path.join(value, "pyarrow"))
    )
    if len(site_roots) != 1:
        raise SystemExit("runtime approved PyArrow site-packages root is not unique")
    site_root = site_roots[0]
    stdlib_inventory = inventory_runtime_tree(stdlib_root, "Python stdlib")
    site_inventory = inventory_runtime_tree(site_root, "site-packages")
    return {
        "python_executable": target,
        "python_executable_bytes": python_bytes,
        "python_executable_sha256": python_sha256,
        "python_version": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
        "python_stdlib_root": stdlib_root,
        "python_stdlib_tree_file_count": stdlib_inventory[0],
        "python_stdlib_tree_bytes": stdlib_inventory[1],
        "python_stdlib_tree_sha256": stdlib_inventory[2],
        "site_packages_root": site_root,
        "site_packages_tree_file_count": site_inventory[0],
        "site_packages_tree_bytes": site_inventory[1],
        "site_packages_tree_sha256": site_inventory[2],
    }


# Q30T_BOOTSTRAP_ENTRY
if len(sys.argv) >= 17 and sys.argv[1].isdigit():
    scratch_fd = int(sys.argv[1])
    observer_path, observer_device, observer_inode, observer_size, observer_sha256 = sys.argv[2:7]
    ptv3_plan_path, ptv3_device, ptv3_inode, ptv3_size, ptv3_sha256 = sys.argv[7:12]
    atomic_path, atomic_device, atomic_inode, atomic_size, atomic_sha256 = sys.argv[12:17]
    arguments = sys.argv[17:]
    observer_fd, observer_bytes = adopt_materialized_blob(
        scratch_fd,
        observer_path,
        observer_device,
        observer_inode,
        observer_size,
        observer_sha256,
        "observer",
    )
    ptv3_plan_fd, _ = adopt_materialized_blob(
        scratch_fd,
        ptv3_plan_path,
        ptv3_device,
        ptv3_inode,
        ptv3_size,
        ptv3_sha256,
        "PTV3 plan",
    )
    atomic_fd, atomic_bytes = adopt_materialized_blob(
        scratch_fd,
        atomic_path,
        atomic_device,
        atomic_inode,
        atomic_size,
        atomic_sha256,
        "atomic dependency",
    )
else:
    observer_path, observer_sha256, atomic_path, atomic_sha256 = sys.argv[1:5]
    arguments = sys.argv[5:]
    observer_fd, observer_bytes = open_blob(observer_path, observer_sha256, "observer")
    atomic_fd, atomic_bytes = open_blob(atomic_path, atomic_sha256, "atomic dependency")
    ptv3_plan_fd = None
try:
    runtime_evidence = authenticate_python_runtime()
    import types

    runtime_module = types.ModuleType("_q30t_runtime_bootstrap")
    runtime_module.evidence = runtime_evidence
    sys.modules[runtime_module.__name__] = runtime_module
    for package_name in ("tools", "tools.launcher", "tools.launcher.common", "tools.launcher.common.specdec"):
        package = types.ModuleType(package_name)
        package.__path__ = []
        sys.modules[package_name] = package
    atomic_name = "tools.launcher.common.specdec.qwen4b_b_atomic"
    atomic_module = types.ModuleType(atomic_name)
    atomic_module.__file__ = f"/proc/self/fd/{atomic_fd}"
    atomic_module.__package__ = "tools.launcher.common.specdec"
    sys.modules[atomic_name] = atomic_module
    exec(compile(atomic_bytes, atomic_module.__file__, "exec"), atomic_module.__dict__)

    observer_module = types.ModuleType("q30t_materialized_observer")
    observer_module.__file__ = f"/proc/self/fd/{observer_fd}"
    observer_module.__package__ = ""
    sys.modules[observer_module.__name__] = observer_module
    exec(compile(observer_bytes, observer_module.__file__, "exec"), observer_module.__dict__)
    sys.argv = [observer_path, *arguments]
    raise SystemExit(observer_module.main())
finally:
    if ptv3_plan_fd is not None:
        os.close(ptv3_plan_fd)
    os.close(atomic_fd)
    os.close(observer_fd)
PY
