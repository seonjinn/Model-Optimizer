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

readonly approved_chmod="/usr/bin/chmod"
readonly approved_git="/usr/bin/git"
readonly approved_mkdir="/usr/bin/mkdir"
readonly approved_mktemp="/usr/bin/mktemp"
readonly approved_python="/usr/bin/python3.12"
readonly approved_sha256sum="/usr/bin/sha256sum"
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

"$approved_python" -I -S - /bin/bash "$approved_env" "$approved_chmod" "$approved_git" \
    "$approved_mkdir" "$approved_mktemp" "$approved_python" "$approved_sha256sum" <<'PY'
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
readonly atomic_path="tools/launcher/common/specdec/qwen4b_b_atomic.py"
readonly observer="$scratch_root/$observer_path"
readonly ptv3_plan="$scratch_root/$ptv3_plan_path"
readonly atomic="$scratch_root/$atomic_path"
"$approved_mkdir" -p -- "${observer%/*}" "${atomic%/*}"

# git cat-file binds both executable and plan bytes to the reviewed commit object.
/usr/bin/env -i LC_ALL=C GIT_CONFIG_NOSYSTEM=1 GIT_CONFIG_SYSTEM=/dev/null \
    GIT_CONFIG_GLOBAL=/dev/null GIT_NO_REPLACE_OBJECTS=1 "$approved_git" \
    -c core.attributesFile=/dev/null -c core.hooksPath=/dev/null -C "$source_path" \
    cat-file -e "$source_commit^{commit}"
/usr/bin/env -i LC_ALL=C GIT_CONFIG_NOSYSTEM=1 GIT_CONFIG_SYSTEM=/dev/null \
    GIT_CONFIG_GLOBAL=/dev/null GIT_NO_REPLACE_OBJECTS=1 "$approved_git" \
    -c core.attributesFile=/dev/null -c core.hooksPath=/dev/null -C "$source_path" \
    cat-file blob "$source_commit:$observer_path" >"$observer"
/usr/bin/env -i LC_ALL=C GIT_CONFIG_NOSYSTEM=1 GIT_CONFIG_SYSTEM=/dev/null \
    GIT_CONFIG_GLOBAL=/dev/null GIT_NO_REPLACE_OBJECTS=1 "$approved_git" \
    -c core.attributesFile=/dev/null -c core.hooksPath=/dev/null -C "$source_path" \
    cat-file blob "$source_commit:$ptv3_plan_path" >"$ptv3_plan"
/usr/bin/env -i LC_ALL=C GIT_CONFIG_NOSYSTEM=1 GIT_CONFIG_SYSTEM=/dev/null \
    GIT_CONFIG_GLOBAL=/dev/null GIT_NO_REPLACE_OBJECTS=1 "$approved_git" \
    -c core.attributesFile=/dev/null -c core.hooksPath=/dev/null -C "$source_path" \
    cat-file blob "$source_commit:$atomic_path" >"$atomic"
"$approved_chmod" 0400 "$observer" "$ptv3_plan" "$atomic"
observer_sha256="$(
    /usr/bin/env -i LC_ALL=C GIT_CONFIG_NOSYSTEM=1 GIT_CONFIG_SYSTEM=/dev/null \
        GIT_CONFIG_GLOBAL=/dev/null GIT_NO_REPLACE_OBJECTS=1 "$approved_git" \
        -c core.attributesFile=/dev/null -c core.hooksPath=/dev/null -C "$source_path" \
        cat-file blob "$source_commit:$observer_path" \
        | /usr/bin/env -i LC_ALL=C "$approved_sha256sum"
)"
observer_sha256="${observer_sha256%% *}"
atomic_sha256="$(
    /usr/bin/env -i LC_ALL=C GIT_CONFIG_NOSYSTEM=1 GIT_CONFIG_SYSTEM=/dev/null \
        GIT_CONFIG_GLOBAL=/dev/null GIT_NO_REPLACE_OBJECTS=1 "$approved_git" \
        -c core.attributesFile=/dev/null -c core.hooksPath=/dev/null -C "$source_path" \
        cat-file blob "$source_commit:$atomic_path" \
        | /usr/bin/env -i LC_ALL=C "$approved_sha256sum"
)"
atomic_sha256="${atomic_sha256%% *}"
readonly observer_sha256 atomic_sha256
[[ "$observer_sha256" =~ ^[0-9a-f]{64}$ && "$atomic_sha256" =~ ^[0-9a-f]{64}$ ]] \
    || fail "materialized executable SHA-256 capture failed"

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
    "$approved_python" -I -S - "$observer" "$observer_sha256" "$atomic" "$atomic_sha256" \
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
    before = os.stat(path, follow_symlinks=False)
    if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1 or before.st_size > MAX_BLOB_BYTES:
        raise SystemExit(f"{label} blob is not a bounded single-link regular file")
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
    rebound = os.stat(path, follow_symlinks=False)
    if identity(before) != identity(after) or identity(after) != identity(rebound):
        os.close(descriptor)
        raise SystemExit(f"{label} blob changed while reading")
    if digest.hexdigest() != expected_sha256:
        os.close(descriptor)
        raise SystemExit(f"{label} blob SHA-256 mismatch")
    return descriptor, b"".join(chunks)


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


observer_path, observer_sha256, atomic_path, atomic_sha256 = sys.argv[1:5]
arguments = sys.argv[5:]
observer_fd, observer_bytes = open_blob(observer_path, observer_sha256, "observer")
atomic_fd, atomic_bytes = open_blob(atomic_path, atomic_sha256, "atomic dependency")
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
    os.close(atomic_fd)
    os.close(observer_fd)
PY
