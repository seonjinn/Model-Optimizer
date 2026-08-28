# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""No-replace durable publication primitives for Task10 artifacts."""

from __future__ import annotations

import ctypes
import errno
import os
import platform
import re
import stat
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Literal

__all__ = [
    "Q30_DIRECTORY_COMPLETION_MARKER",
    "Task10PublicationError",
    "Task10RecoveryState",
    "atomic_publish_bytes",
    "atomic_publish_directory",
    "authenticate_directory_completion",
    "recovery_state",
]

Q30_DIRECTORY_COMPLETION_MARKER = ".q30-publication-incomplete"
_DIRECTORY_INCOMPLETE = b"q30-directory-publication-incomplete-v1\n"
_DIRECTORY_COMPLETE = b"q30-directory-publication-complete-v1\n"


@dataclass(frozen=True)
class PathObservation:
    """No-follow pathname evidence captured after a publication failure."""

    path: Path
    status: Literal["present", "absent", "unavailable"]
    device: int | None
    inode: int | None
    mode: int | None
    error: str | None

    @property
    def identity(self) -> tuple[int, int] | None:
        if self.device is None or self.inode is None:
            return None
        return self.device, self.inode


@dataclass(frozen=True)
class Task10RecoveryState:
    """Evidence needed to recover an ambiguous immutable publication."""

    phase: str
    partial_path: Path
    destination_path: Path
    expected_partial_identity: tuple[int, int] | None
    partial_observation: PathObservation
    destination_observation: PathObservation
    recovery_required: Literal[True] = True


class Task10PublicationError(RuntimeError):
    """A Task10 artifact could not be published without replacement."""


def recovery_state(error: BaseException) -> Task10RecoveryState:
    """Return typed recovery evidence attached to a publication failure."""
    state = getattr(error, "recovery_state", None)
    if not isinstance(state, Task10RecoveryState):
        raise ValueError("exception does not carry Task10 recovery state")
    return state


def authenticate_directory_completion(directory_fd: int) -> tuple[int, int, int, int, int]:
    """Authenticate a durable completion marker through one already-held root."""
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0)
    named = os.stat(
        Q30_DIRECTORY_COMPLETION_MARKER,
        dir_fd=directory_fd,
        follow_symlinks=False,
    )
    descriptor = os.open(Q30_DIRECTORY_COMPLETION_MARKER, flags, dir_fd=directory_fd)
    try:
        before = os.fstat(descriptor)
        identity = (
            before.st_dev,
            before.st_ino,
            before.st_size,
            before.st_mtime_ns,
            before.st_ctime_ns,
        )
        if (
            (named.st_dev, named.st_ino) != (before.st_dev, before.st_ino)
            or not stat.S_ISREG(before.st_mode)
            or before.st_nlink != 1
            or before.st_size != len(_DIRECTORY_COMPLETE)
        ):
            raise Task10PublicationError("Task10 directory completion marker is invalid")
        raw = os.read(descriptor, len(_DIRECTORY_COMPLETE) + 1)
        after = os.fstat(descriptor)
        rebound = os.stat(
            Q30_DIRECTORY_COMPLETION_MARKER,
            dir_fd=directory_fd,
            follow_symlinks=False,
        )
        if (
            raw != _DIRECTORY_COMPLETE
            or (
                after.st_dev,
                after.st_ino,
                after.st_size,
                after.st_mtime_ns,
                after.st_ctime_ns,
            )
            != identity
            or (rebound.st_dev, rebound.st_ino) != (before.st_dev, before.st_ino)
            or after.st_nlink != 1
            or rebound.st_nlink != 1
        ):
            raise Task10PublicationError("Task10 directory completion marker changed")
        return identity
    finally:
        os.close(descriptor)


def atomic_publish_bytes(destination: Path, payload: bytes, *, job_id: str) -> None:
    """Publish one durable file without replacing a prior pathname."""
    if re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", job_id) is None:
        raise ValueError("Task10 publication job_id is unsafe")
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_name(f".{destination.name}.partial-{job_id}")
    phase = "partial_setup"
    expected: tuple[int, int] | None = None
    parent_fd: int | None = None
    try:
        directory_flags = (
            os.O_RDONLY
            | getattr(os, "O_DIRECTORY", 0)
            | getattr(os, "O_NOFOLLOW", 0)
            | getattr(os, "O_NONBLOCK", 0)
        )
        parent_fd = os.open(destination.parent, directory_flags)
        parent_status = os.fstat(parent_fd)
        absolute_parent = os.stat(destination.parent, follow_symlinks=False)
        parent_identity = (parent_status.st_dev, parent_status.st_ino)
        if (absolute_parent.st_dev, absolute_parent.st_ino) != parent_identity:
            raise Task10PublicationError("Task10 publication parent changed while opening")
        try:
            os.stat(destination.name, dir_fd=parent_fd, follow_symlinks=False)
        except FileNotFoundError:
            pass
        else:
            raise FileExistsError(f"immutable Task10 artifact already exists: {destination}")
        flags = (
            os.O_WRONLY
            | os.O_CREAT
            | os.O_EXCL
            | getattr(os, "O_NOFOLLOW", 0)
            | getattr(os, "O_NONBLOCK", 0)
        )
        descriptor = os.open(partial.name, flags, 0o600, dir_fd=parent_fd)
        metadata = os.fstat(descriptor)
        expected = (metadata.st_dev, metadata.st_ino)
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        phase = "reread"
        if _reread_created_file_at(parent_fd, partial.name, expected) != payload:
            raise Task10PublicationError("Task10 partial reread identity mismatch")
        phase = "rename"
        installed_identity = _rename_no_replace(
            partial,
            destination,
            expected_file_sha256=sha256(payload).hexdigest(),
            expected_file_identity=expected,
            expected_parent_identity=parent_identity,
            parent_fd=parent_fd,
        )
        installed_status = os.stat(
            destination.name,
            dir_fd=parent_fd,
            follow_symlinks=False,
        )
        if (installed_status.st_dev, installed_status.st_ino) != installed_identity:
            raise Task10PublicationError("Task10 destination inode differs from its partial")
        if _reread_created_file_at(parent_fd, destination.name, installed_identity) != payload:
            raise Task10PublicationError("Task10 destination content differs from its partial")
        phase = "parent_fsync"
        os.fsync(parent_fd)
        final_parent = os.fstat(parent_fd)
        final_named = os.stat(
            destination.name,
            dir_fd=parent_fd,
            follow_symlinks=False,
        )
        try:
            final_absolute_parent = os.stat(destination.parent, follow_symlinks=False)
            final_absolute = os.stat(destination, follow_symlinks=False)
        except OSError as error:
            raise Task10PublicationError(
                "Task10 publication parent or destination rebound after durability"
            ) from error
        if (
            (final_parent.st_dev, final_parent.st_ino) != parent_identity
            or (final_absolute_parent.st_dev, final_absolute_parent.st_ino) != parent_identity
            or (final_named.st_dev, final_named.st_ino) != installed_identity
            or (final_absolute.st_dev, final_absolute.st_ino) != installed_identity
            or not stat.S_ISREG(final_named.st_mode)
            or final_named.st_nlink != 1
            or final_absolute.st_nlink != 1
        ):
            raise Task10PublicationError(
                "Task10 publication parent or destination rebound after durability"
            )
    except BaseException as error:
        setattr(
            error,
            "recovery_state",
            Task10RecoveryState(
                phase,
                partial,
                destination,
                expected,
                _observe(partial),
                _observe(destination),
            ),
        )
        raise
    finally:
        if parent_fd is not None:
            os.close(parent_fd)


def atomic_publish_directory(
    partial: Path,
    destination: Path,
    *,
    expected_directory_identity: tuple[int, int] | None = None,
    expected_parent_identity: tuple[int, int] | None = None,
) -> None:
    """Publish a fully fsynced sibling directory without replacement."""
    observed_partial_identity = _observe(partial).identity
    observed_parent_identity = _observe(destination.parent).identity
    expected = expected_directory_identity or observed_partial_identity
    parent_identity = expected_parent_identity or observed_parent_identity
    phase = "caller_binding"
    try:
        if (
            expected_directory_identity is not None
            and observed_partial_identity != expected_directory_identity
        ):
            raise Task10PublicationError("Task10 partial identity differs from caller binding")
        if (
            expected_parent_identity is not None
            and observed_parent_identity != expected_parent_identity
        ):
            raise Task10PublicationError("Task10 parent identity differs from caller binding")
        if (
            expected is None
            or parent_identity is None
            or partial.parent.resolve() != destination.parent.resolve()
        ):
            raise Task10PublicationError("Task10 partial must be a present sibling directory")
        phase = "directory_fsync"
        _fsync_tree(partial)
        if _observe(partial).identity != expected:
            raise Task10PublicationError("Task10 partial inode changed before rename")
        expected_tree_sha256 = _directory_content_sha256(partial)
        if _observe(partial).identity != expected:
            raise Task10PublicationError("Task10 partial root changed while digesting")
        phase = "rename"
        installed_identity = _rename_no_replace(
            partial,
            destination,
            expected_directory_sha256=expected_tree_sha256,
            expected_directory_identity=expected,
            expected_parent_identity=parent_identity,
        )
        if _observe(destination).identity != installed_identity:
            raise Task10PublicationError("Task10 destination inode differs from its partial")
        directory_flags = (
            os.O_RDONLY
            | getattr(os, "O_DIRECTORY", 0)
            | getattr(os, "O_NOFOLLOW", 0)
            | getattr(os, "O_NONBLOCK", 0)
        )
        installed_fd = os.open(destination, directory_flags)
        try:
            installed_root = os.fstat(installed_fd)
            if (installed_root.st_dev, installed_root.st_ino) != installed_identity:
                raise Task10PublicationError("Task10 destination changed before marker adoption")
            marker_identity = authenticate_directory_completion(installed_fd)
        finally:
            os.close(installed_fd)
        phase = "parent_fsync"
        _fsync_directory(destination.parent)
        final_status = os.stat(destination, follow_symlinks=False)
        if (
            _observe(destination.parent).identity != parent_identity
            or (final_status.st_dev, final_status.st_ino) != installed_identity
            or not stat.S_ISDIR(final_status.st_mode)
        ):
            raise Task10PublicationError(
                "Task10 publication parent or destination rebound after durability"
            )
        final_fd = os.open(destination, directory_flags)
        try:
            if authenticate_directory_completion(final_fd) != marker_identity:
                raise Task10PublicationError("Task10 directory completion marker rebound")
        finally:
            os.close(final_fd)
    except BaseException as error:
        setattr(
            error,
            "recovery_state",
            Task10RecoveryState(
                phase,
                partial,
                destination,
                expected,
                _observe(partial),
                _observe(destination),
            ),
        )
        raise


def _reread_created_file_at(
    parent_fd: int,
    name: str,
    expected: tuple[int, int],
) -> bytes:
    descriptor = os.open(
        name,
        os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0),
        dir_fd=parent_fd,
    )
    try:
        before = os.fstat(descriptor)
        if (before.st_dev, before.st_ino) != expected or not stat.S_ISREG(before.st_mode):
            raise Task10PublicationError("Task10 partial inode changed before reread")
        with os.fdopen(descriptor, "rb") as stream:
            payload = stream.read()
            after = os.fstat(stream.fileno())
    except BaseException:
        raise
    rebound = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
    if (
        (after.st_dev, after.st_ino) != expected
        or (rebound.st_dev, rebound.st_ino) != expected
        or after.st_nlink != 1
        or rebound.st_nlink != 1
    ):
        raise Task10PublicationError("Task10 partial inode changed during reread")
    return payload


def _observe(path: Path) -> PathObservation:
    try:
        metadata = os.lstat(path)
    except FileNotFoundError:
        return PathObservation(path, "absent", None, None, None, None)
    except OSError as error:
        return PathObservation(path, "unavailable", None, None, None, str(error))
    return PathObservation(
        path, "present", metadata.st_dev, metadata.st_ino, metadata.st_mode, None
    )


def _fsync_tree(root: Path) -> None:
    directories: list[Path] = []
    for directory, directory_names, file_names in os.walk(root, followlinks=False):
        current = Path(directory)
        directories.append(current)
        for name in directory_names:
            child = current / name
            if child.is_symlink():
                raise Task10PublicationError("Task10 partial contains a symlink directory")
        for name in file_names:
            child = current / name
            metadata = os.stat(child, follow_symlinks=False)
            if not stat.S_ISREG(metadata.st_mode):
                raise Task10PublicationError("Task10 partial contains a non-regular file")
            descriptor = os.open(child, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
            try:
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
    for directory in reversed(directories):
        _fsync_directory(directory)


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _directory_content_sha256(path: Path) -> str:
    flags = (
        os.O_RDONLY
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_NONBLOCK", 0)
    )
    descriptor = os.open(path, flags)
    try:
        return _directory_content_sha256_at(descriptor)
    finally:
        os.close(descriptor)


def _directory_content_sha256_at(directory_fd: int) -> str:
    before = os.fstat(directory_fd)
    if not stat.S_ISDIR(before.st_mode):
        raise Task10PublicationError("Task10 directory digest root is invalid")
    digest = sha256()
    names = sorted(os.listdir(directory_fd))
    read_flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0)
    directory_flags = read_flags | getattr(os, "O_DIRECTORY", 0)
    for name in names:
        encoded = os.fsencode(name)
        named = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
        digest.update(len(encoded).to_bytes(8, "big"))
        digest.update(encoded)
        digest.update(stat.S_IMODE(named.st_mode).to_bytes(4, "big"))
        if stat.S_ISREG(named.st_mode):
            if named.st_nlink != 1:
                raise Task10PublicationError("Task10 directory digest contains a hardlink")
            child = os.open(name, read_flags, dir_fd=directory_fd)
            try:
                opened = os.fstat(child)
                identity = (
                    opened.st_dev,
                    opened.st_ino,
                    opened.st_size,
                    opened.st_mtime_ns,
                    opened.st_ctime_ns,
                )
                if (named.st_dev, named.st_ino) != identity[:2]:
                    raise Task10PublicationError(
                        "Task10 directory digest file changed while opening"
                    )
                digest.update(b"F")
                digest.update(opened.st_size.to_bytes(8, "big"))
                while block := os.read(child, 1024 * 1024):
                    digest.update(block)
                after = os.fstat(child)
                rebound = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
                if (
                    after.st_dev,
                    after.st_ino,
                    after.st_size,
                    after.st_mtime_ns,
                    after.st_ctime_ns,
                ) != identity or (
                    rebound.st_dev,
                    rebound.st_ino,
                    rebound.st_size,
                    rebound.st_mtime_ns,
                    rebound.st_ctime_ns,
                ) != identity:
                    raise Task10PublicationError(
                        "Task10 directory digest file changed while reading"
                    )
            finally:
                os.close(child)
            continue
        if not stat.S_ISDIR(named.st_mode):
            raise Task10PublicationError("Task10 directory digest has an unsupported entry")
        child = os.open(name, directory_flags, dir_fd=directory_fd)
        try:
            opened = os.fstat(child)
            if (named.st_dev, named.st_ino) != (opened.st_dev, opened.st_ino):
                raise Task10PublicationError("Task10 directory digest child changed while opening")
            digest.update(b"D")
            digest.update(bytes.fromhex(_directory_content_sha256_at(child)))
        finally:
            os.close(child)
    after = os.fstat(directory_fd)
    if (before.st_dev, before.st_ino, before.st_mtime_ns, before.st_ctime_ns) != (
        after.st_dev,
        after.st_ino,
        after.st_mtime_ns,
        after.st_ctime_ns,
    ) or sorted(os.listdir(directory_fd)) != names:
        raise Task10PublicationError("Task10 directory changed while digesting")
    return digest.hexdigest()


def _native_rename_no_replace(
    source: Path,
    destination: Path,
    *,
    parent_fd: int | None = None,
) -> None:
    library = ctypes.CDLL(None, use_errno=True)
    source_bytes = os.fsencode(source.name if parent_fd is not None else source)
    destination_bytes = os.fsencode(destination.name if parent_fd is not None else destination)
    if platform.system() == "Linux":
        try:
            rename = library.renameat2
        except AttributeError as error:
            raise OSError(errno.ENOSYS, "atomic no-replace rename is unavailable") from error
        rename.argtypes = [
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_uint,
        ]
        rename.restype = ctypes.c_int
        directory_fd = -100 if parent_fd is None else parent_fd
        result = rename(directory_fd, source_bytes, directory_fd, destination_bytes, 1)
    elif platform.system() == "Darwin":
        if parent_fd is None:
            rename = library.renamex_np
            rename.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_uint]
            rename.restype = ctypes.c_int
            result = rename(source_bytes, destination_bytes, 0x00000004)
        else:
            try:
                rename = library.renameatx_np
            except AttributeError as error:
                raise OSError(errno.ENOSYS, "atomic no-replace rename is unavailable") from error
            rename.argtypes = [
                ctypes.c_int,
                ctypes.c_char_p,
                ctypes.c_int,
                ctypes.c_char_p,
                ctypes.c_uint,
            ]
            rename.restype = ctypes.c_int
            result = rename(parent_fd, source_bytes, parent_fd, destination_bytes, 0x00000004)
    else:
        raise OSError(
            errno.ENOSYS, f"atomic no-replace rename is unsupported on {platform.system()}"
        )
    if result == 0:
        return
    number = ctypes.get_errno()
    raise OSError(number, os.strerror(number), destination)


def _unsupported_no_replace_error(error: OSError) -> bool:
    unsupported = {errno.EINVAL, errno.ENOSYS}
    unsupported.update(
        number
        for number in (getattr(errno, "EOPNOTSUPP", None), getattr(errno, "ENOTSUP", None))
        if number is not None
    )
    return error.errno in unsupported


def _copy_file_with_exclusive_destination(
    source: Path,
    destination: Path,
    *,
    expected_sha256: str,
    expected_source_identity: tuple[int, int] | None = None,
    expected_parent_identity: tuple[int, int] | None = None,
    parent_fd: int | None = None,
) -> tuple[int, int]:
    """Copy a sibling partial into an exclusively created final pathname."""
    if source.parent.absolute() != destination.parent.absolute():
        raise Task10PublicationError("Task10 partial must be a destination sibling")
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0)
    directory_flags = flags | getattr(os, "O_DIRECTORY", 0)
    owns_parent_fd = parent_fd is None
    if parent_fd is None:
        parent_fd = os.open(source.parent, directory_flags)
    source_fd: int | None = None
    destination_fd: int | None = None
    try:
        parent_before = os.fstat(parent_fd)
        absolute_parent = os.stat(source.parent, follow_symlinks=False)
        parent_identity = (parent_before.st_dev, parent_before.st_ino)
        if (
            expected_parent_identity is not None and parent_identity != expected_parent_identity
        ) or parent_identity != (
            absolute_parent.st_dev,
            absolute_parent.st_ino,
        ):
            raise Task10PublicationError("Task10 publication parent changed while opening")
        source_named = os.stat(source.name, dir_fd=parent_fd, follow_symlinks=False)
        source_fd = os.open(source.name, flags, dir_fd=parent_fd)
        source_before = os.fstat(source_fd)
        source_identity = (source_before.st_dev, source_before.st_ino)
        source_full_identity = (
            source_before.st_dev,
            source_before.st_ino,
            source_before.st_size,
            source_before.st_mtime_ns,
            source_before.st_ctime_ns,
        )
        if (
            (source_named.st_dev, source_named.st_ino) != source_identity
            or (
                expected_source_identity is not None and source_identity != expected_source_identity
            )
            or not stat.S_ISREG(source_before.st_mode)
            or source_before.st_nlink != 1
        ):
            raise Task10PublicationError("Task10 partial is not a stable single-link file")
        destination_flags = (
            os.O_RDWR
            | os.O_CREAT
            | os.O_EXCL
            | getattr(os, "O_NOFOLLOW", 0)
            | getattr(os, "O_NONBLOCK", 0)
        )
        destination_fd = os.open(
            destination.name,
            destination_flags,
            stat.S_IMODE(source_before.st_mode),
            dir_fd=parent_fd,
        )
        destination_before = os.fstat(destination_fd)
        destination_identity = (destination_before.st_dev, destination_before.st_ino)
        os.lseek(source_fd, 0, os.SEEK_SET)
        copied_digest = sha256()
        while block := os.read(source_fd, 1024 * 1024):
            copied_digest.update(block)
            view = memoryview(block)
            while view:
                written = os.write(destination_fd, view)
                if written < 1:
                    raise Task10PublicationError("Task10 exclusive destination write stalled")
                view = view[written:]
        os.fsync(destination_fd)
        source_after = os.fstat(source_fd)
        source_rebound = os.stat(source.name, dir_fd=parent_fd, follow_symlinks=False)
        destination_after = os.fstat(destination_fd)
        destination_rebound = os.stat(destination.name, dir_fd=parent_fd, follow_symlinks=False)
        os.lseek(destination_fd, 0, os.SEEK_SET)
        destination_digest = sha256()
        while block := os.read(destination_fd, 1024 * 1024):
            destination_digest.update(block)
        if (
            (
                source_after.st_dev,
                source_after.st_ino,
                source_after.st_size,
                source_after.st_mtime_ns,
                source_after.st_ctime_ns,
            )
            != source_full_identity
            or (
                source_rebound.st_dev,
                source_rebound.st_ino,
                source_rebound.st_size,
                source_rebound.st_mtime_ns,
                source_rebound.st_ctime_ns,
            )
            != source_full_identity
            or (destination_after.st_dev, destination_after.st_ino) != destination_identity
            or (destination_rebound.st_dev, destination_rebound.st_ino) != destination_identity
            or destination_after.st_nlink != 1
            or destination_after.st_size != source_before.st_size
            or copied_digest.hexdigest() != expected_sha256
            or destination_digest.hexdigest() != expected_sha256
        ):
            raise Task10PublicationError("Task10 exclusive file publication changed while copying")
        os.fsync(parent_fd)
        parent_after = os.fstat(parent_fd)
        absolute_parent_after = os.stat(source.parent, follow_symlinks=False)
        final_named = os.stat(destination.name, dir_fd=parent_fd, follow_symlinks=False)
        final_absolute = os.stat(destination, follow_symlinks=False)
        final_opened = os.fstat(destination_fd)
        if (
            (parent_after.st_dev, parent_after.st_ino)
            != (parent_before.st_dev, parent_before.st_ino)
            or (absolute_parent_after.st_dev, absolute_parent_after.st_ino)
            != (parent_before.st_dev, parent_before.st_ino)
            or (final_named.st_dev, final_named.st_ino) != destination_identity
            or (final_absolute.st_dev, final_absolute.st_ino) != destination_identity
            or (final_opened.st_dev, final_opened.st_ino) != destination_identity
            or final_opened.st_nlink != 1
        ):
            raise Task10PublicationError("Task10 exclusive file publication rebound")
        return destination_identity
    finally:
        if destination_fd is not None:
            os.close(destination_fd)
        if source_fd is not None:
            os.close(source_fd)
        if owns_parent_fd:
            os.close(parent_fd)


def _copy_directory_at(source_fd: int, destination_fd: int, *, root: bool = False) -> None:
    """Copy one stable no-link tree into an exclusively created held directory."""
    source_before = os.fstat(source_fd)
    source_names = sorted(os.listdir(source_fd))
    destination_names = set(os.listdir(destination_fd))
    expected_destination_names = set(source_names)
    if root:
        expected_destination_names.add(Q30_DIRECTORY_COMPLETION_MARKER)
    if destination_names != ({Q30_DIRECTORY_COMPLETION_MARKER} if root else set()):
        raise Task10PublicationError("Task10 directory reservation is not empty")
    read_flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0)
    directory_flags = read_flags | getattr(os, "O_DIRECTORY", 0)
    for name in source_names:
        source_named = os.stat(name, dir_fd=source_fd, follow_symlinks=False)
        if stat.S_ISREG(source_named.st_mode):
            if source_named.st_nlink != 1:
                raise Task10PublicationError("Task10 directory partial contains a hardlink")
            source_child = os.open(name, read_flags, dir_fd=source_fd)
            destination_child: int | None = None
            try:
                source_opened = os.fstat(source_child)
                source_identity = (source_opened.st_dev, source_opened.st_ino)
                if (source_named.st_dev, source_named.st_ino) != source_identity:
                    raise Task10PublicationError("Task10 source file changed while opening")
                destination_child = os.open(
                    name,
                    os.O_WRONLY
                    | os.O_CREAT
                    | os.O_EXCL
                    | getattr(os, "O_NOFOLLOW", 0)
                    | getattr(os, "O_NONBLOCK", 0),
                    stat.S_IMODE(source_opened.st_mode),
                    dir_fd=destination_fd,
                )
                destination_opened = os.fstat(destination_child)
                destination_identity = (
                    destination_opened.st_dev,
                    destination_opened.st_ino,
                )
                while block := os.read(source_child, 1024 * 1024):
                    view = memoryview(block)
                    while view:
                        written = os.write(destination_child, view)
                        if written < 1:
                            raise Task10PublicationError("Task10 directory copy stalled")
                        view = view[written:]
                os.fsync(destination_child)
                source_after = os.fstat(source_child)
                source_rebound = os.stat(name, dir_fd=source_fd, follow_symlinks=False)
                destination_after = os.fstat(destination_child)
                destination_rebound = os.stat(name, dir_fd=destination_fd, follow_symlinks=False)
                if (
                    (source_after.st_dev, source_after.st_ino) != source_identity
                    or (source_rebound.st_dev, source_rebound.st_ino) != source_identity
                    or source_after.st_size != source_opened.st_size
                    or (destination_after.st_dev, destination_after.st_ino) != destination_identity
                    or (destination_rebound.st_dev, destination_rebound.st_ino)
                    != destination_identity
                    or destination_after.st_nlink != 1
                    or destination_after.st_size != source_opened.st_size
                ):
                    raise Task10PublicationError("Task10 directory file changed while copying")
            finally:
                if destination_child is not None:
                    os.close(destination_child)
                os.close(source_child)
            continue
        if not stat.S_ISDIR(source_named.st_mode):
            raise Task10PublicationError("Task10 directory partial has an unsupported entry")
        source_child = os.open(name, directory_flags, dir_fd=source_fd)
        destination_child = None
        try:
            source_opened = os.fstat(source_child)
            source_identity = (source_opened.st_dev, source_opened.st_ino)
            if (source_named.st_dev, source_named.st_ino) != source_identity:
                raise Task10PublicationError("Task10 source directory changed while opening")
            os.mkdir(name, stat.S_IMODE(source_opened.st_mode), dir_fd=destination_fd)
            destination_child = os.open(name, directory_flags, dir_fd=destination_fd)
            destination_opened = os.fstat(destination_child)
            destination_identity = (destination_opened.st_dev, destination_opened.st_ino)
            _copy_directory_at(source_child, destination_child)
            os.fsync(destination_child)
            source_after = os.fstat(source_child)
            source_rebound = os.stat(name, dir_fd=source_fd, follow_symlinks=False)
            destination_after = os.fstat(destination_child)
            destination_rebound = os.stat(name, dir_fd=destination_fd, follow_symlinks=False)
            if (
                (source_after.st_dev, source_after.st_ino) != source_identity
                or (source_rebound.st_dev, source_rebound.st_ino) != source_identity
                or (destination_after.st_dev, destination_after.st_ino) != destination_identity
                or (destination_rebound.st_dev, destination_rebound.st_ino) != destination_identity
            ):
                raise Task10PublicationError("Task10 directory changed while copying")
        finally:
            if destination_child is not None:
                os.close(destination_child)
            os.close(source_child)
    source_after = os.fstat(source_fd)
    if (
        (
            source_after.st_dev,
            source_after.st_ino,
            source_after.st_mtime_ns,
            source_after.st_ctime_ns,
        )
        != (
            source_before.st_dev,
            source_before.st_ino,
            source_before.st_mtime_ns,
            source_before.st_ctime_ns,
        )
        or sorted(os.listdir(source_fd)) != source_names
        or set(os.listdir(destination_fd)) != expected_destination_names
    ):
        raise Task10PublicationError("Task10 directory tree changed while copying")


def _trees_equal_at(source_fd: int, destination_fd: int, *, root: bool = False) -> bool:
    """Reauthenticate two no-link trees through already-held roots."""
    source_names = sorted(os.listdir(source_fd))
    expected_destination_names = set(source_names)
    if root:
        expected_destination_names.add(Q30_DIRECTORY_COMPLETION_MARKER)
    if set(os.listdir(destination_fd)) != expected_destination_names:
        return False
    read_flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0)
    directory_flags = read_flags | getattr(os, "O_DIRECTORY", 0)
    for name in source_names:
        source_named = os.stat(name, dir_fd=source_fd, follow_symlinks=False)
        destination_named = os.stat(name, dir_fd=destination_fd, follow_symlinks=False)
        if stat.S_ISREG(source_named.st_mode) and stat.S_ISREG(destination_named.st_mode):
            if source_named.st_nlink != 1 or destination_named.st_nlink != 1:
                return False
            source_child = os.open(name, read_flags, dir_fd=source_fd)
            destination_child = os.open(name, read_flags, dir_fd=destination_fd)
            try:
                source_before = os.fstat(source_child)
                destination_before = os.fstat(destination_child)
                if source_before.st_size != destination_before.st_size:
                    return False
                while True:
                    source_block = os.read(source_child, 1024 * 1024)
                    destination_block = os.read(destination_child, 1024 * 1024)
                    if source_block != destination_block:
                        return False
                    if not source_block:
                        break
                source_after = os.fstat(source_child)
                destination_after = os.fstat(destination_child)
                if (
                    (source_before.st_dev, source_before.st_ino)
                    != (source_after.st_dev, source_after.st_ino)
                    or (destination_before.st_dev, destination_before.st_ino)
                    != (destination_after.st_dev, destination_after.st_ino)
                    or (source_named.st_dev, source_named.st_ino)
                    != (source_after.st_dev, source_after.st_ino)
                    or (destination_named.st_dev, destination_named.st_ino)
                    != (destination_after.st_dev, destination_after.st_ino)
                ):
                    return False
            finally:
                os.close(destination_child)
                os.close(source_child)
            continue
        if not stat.S_ISDIR(source_named.st_mode) or not stat.S_ISDIR(destination_named.st_mode):
            return False
        source_child = os.open(name, directory_flags, dir_fd=source_fd)
        destination_child = os.open(name, directory_flags, dir_fd=destination_fd)
        try:
            if not _trees_equal_at(source_child, destination_child):
                return False
        finally:
            os.close(destination_child)
            os.close(source_child)
    return (
        sorted(os.listdir(source_fd)) == source_names
        and set(os.listdir(destination_fd)) == expected_destination_names
    )


def _rename_directory_with_reservation(
    source: Path,
    destination: Path,
    *,
    expected_sha256: str,
    expected_source_identity: tuple[int, int],
    expected_parent_identity: tuple[int, int],
) -> tuple[int, int]:
    """Claim an absent sibling directory before a portable held-parent rename."""
    if source.parent.absolute() != destination.parent.absolute():
        raise Task10PublicationError("Task10 partial must be a destination sibling")
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0)
    directory_flags = flags | getattr(os, "O_DIRECTORY", 0)
    parent_fd = os.open(source.parent, directory_flags)
    reservation_fd: int | None = None
    source_fd: int | None = None
    try:
        parent_before = os.fstat(parent_fd)
        absolute_parent = os.stat(source.parent, follow_symlinks=False)
        if (parent_before.st_dev, parent_before.st_ino) != expected_parent_identity or (
            absolute_parent.st_dev,
            absolute_parent.st_ino,
        ) != expected_parent_identity:
            raise Task10PublicationError(
                "Task10 directory reservation parent identity changed before publication"
            )
        source_named = os.stat(source.name, dir_fd=parent_fd, follow_symlinks=False)
        source_fd = os.open(source.name, directory_flags, dir_fd=parent_fd)
        source_opened = os.fstat(source_fd)
        source_identity = (source_opened.st_dev, source_opened.st_ino)
        if (
            (absolute_parent.st_dev, absolute_parent.st_ino)
            != (parent_before.st_dev, parent_before.st_ino)
            or source_identity != expected_source_identity
            or (source_named.st_dev, source_named.st_ino) != expected_source_identity
            or not stat.S_ISDIR(source_opened.st_mode)
        ):
            raise Task10PublicationError("Task10 directory partial or parent changed")
        try:
            os.mkdir(
                destination.name,
                stat.S_IMODE(source_opened.st_mode),
                dir_fd=parent_fd,
            )
        except FileExistsError:
            raise FileExistsError(errno.EEXIST, os.strerror(errno.EEXIST), destination) from None
        reservation_named = os.stat(destination.name, dir_fd=parent_fd, follow_symlinks=False)
        reservation_fd = os.open(destination.name, directory_flags, dir_fd=parent_fd)
        reservation_opened = os.fstat(reservation_fd)
        reservation_identity = (reservation_opened.st_dev, reservation_opened.st_ino)
        if (
            (reservation_named.st_dev, reservation_named.st_ino) != reservation_identity
            or not stat.S_ISDIR(reservation_opened.st_mode)
            or os.listdir(reservation_fd)
        ):
            raise Task10PublicationError("Task10 directory reservation changed after creation")
        os.fsync(parent_fd)
        parent_reserved = os.fstat(parent_fd)
        absolute_parent_reserved = os.stat(source.parent, follow_symlinks=False)
        source_rebound = os.stat(source.name, dir_fd=parent_fd, follow_symlinks=False)
        reservation_rebound = os.stat(destination.name, dir_fd=parent_fd, follow_symlinks=False)
        if (
            (parent_reserved.st_dev, parent_reserved.st_ino)
            != (parent_before.st_dev, parent_before.st_ino)
            or (absolute_parent_reserved.st_dev, absolute_parent_reserved.st_ino)
            != (parent_before.st_dev, parent_before.st_ino)
            or (source_rebound.st_dev, source_rebound.st_ino) != source_identity
            or (reservation_rebound.st_dev, reservation_rebound.st_ino) != reservation_identity
            or os.listdir(reservation_fd)
        ):
            raise Task10PublicationError("Task10 directory reservation rebound before install")
        marker_fd = os.open(
            Q30_DIRECTORY_COMPLETION_MARKER,
            os.O_RDWR | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
            0o600,
            dir_fd=reservation_fd,
        )
        try:
            marker_status = os.fstat(marker_fd)
            marker_identity = (marker_status.st_dev, marker_status.st_ino)
            os.write(marker_fd, _DIRECTORY_INCOMPLETE)
            os.fsync(marker_fd)
            _copy_directory_at(source_fd, reservation_fd, root=True)
            if _directory_content_sha256_at(source_fd) != expected_sha256:
                raise Task10PublicationError(
                    "Task10 directory source content changed while copying"
                )
            os.fsync(reservation_fd)
            marker_named = os.stat(
                Q30_DIRECTORY_COMPLETION_MARKER,
                dir_fd=reservation_fd,
                follow_symlinks=False,
            )
            marker_after = os.fstat(marker_fd)
            if (
                (marker_named.st_dev, marker_named.st_ino) != marker_identity
                or (marker_after.st_dev, marker_after.st_ino) != marker_identity
                or not stat.S_ISREG(marker_after.st_mode)
                or marker_after.st_nlink != 1
            ):
                raise Task10PublicationError("Task10 directory commit marker changed")
            os.lseek(marker_fd, 0, os.SEEK_SET)
            os.ftruncate(marker_fd, 0)
            view = memoryview(_DIRECTORY_COMPLETE)
            while view:
                written = os.write(marker_fd, view)
                if written < 1:
                    raise Task10PublicationError("Task10 directory marker update stalled")
                view = view[written:]
            os.fsync(marker_fd)
            marker_complete = os.fstat(marker_fd)
            marker_rebound = os.stat(
                Q30_DIRECTORY_COMPLETION_MARKER,
                dir_fd=reservation_fd,
                follow_symlinks=False,
            )
            if (
                (marker_complete.st_dev, marker_complete.st_ino) != marker_identity
                or (marker_rebound.st_dev, marker_rebound.st_ino) != marker_identity
                or marker_complete.st_nlink != 1
                or marker_complete.st_size != len(_DIRECTORY_COMPLETE)
            ):
                raise Task10PublicationError("Task10 directory completion marker rebound")
            os.fsync(reservation_fd)
        finally:
            os.close(marker_fd)
        if authenticate_directory_completion(reservation_fd)[:2] != marker_identity:
            raise Task10PublicationError("Task10 directory completion marker is not durable")
        if not _trees_equal_at(source_fd, reservation_fd, root=True):
            raise Task10PublicationError("Task10 reserved directory copy changed")
        os.fsync(parent_fd)
        final_parent = os.fstat(parent_fd)
        final_absolute_parent = os.stat(source.parent, follow_symlinks=False)
        final_named = os.stat(destination.name, dir_fd=parent_fd, follow_symlinks=False)
        final_absolute = os.stat(destination, follow_symlinks=False)
        if (
            (final_parent.st_dev, final_parent.st_ino)
            != (parent_before.st_dev, parent_before.st_ino)
            or (final_absolute_parent.st_dev, final_absolute_parent.st_ino)
            != (parent_before.st_dev, parent_before.st_ino)
            or (final_named.st_dev, final_named.st_ino) != reservation_identity
            or (final_absolute.st_dev, final_absolute.st_ino) != reservation_identity
            or authenticate_directory_completion(reservation_fd)[:2] != marker_identity
            or not _trees_equal_at(source_fd, reservation_fd, root=True)
        ):
            raise Task10PublicationError("Task10 reserved directory publication rebound")
        return reservation_identity
    finally:
        if source_fd is not None:
            os.close(source_fd)
        if reservation_fd is not None:
            os.close(reservation_fd)
        os.close(parent_fd)


def _rename_no_replace(
    source: Path,
    destination: Path,
    *,
    expected_file_sha256: str | None = None,
    expected_file_identity: tuple[int, int] | None = None,
    expected_directory_sha256: str | None = None,
    expected_directory_identity: tuple[int, int] | None = None,
    expected_parent_identity: tuple[int, int] | None = None,
    parent_fd: int | None = None,
) -> tuple[int, int]:
    try:
        if parent_fd is None:
            source_status = os.stat(source, follow_symlinks=False)
        else:
            opened_parent = os.fstat(parent_fd)
            absolute_parent = os.stat(source.parent, follow_symlinks=False)
            if (
                expected_parent_identity is None
                or (opened_parent.st_dev, opened_parent.st_ino) != expected_parent_identity
                or (absolute_parent.st_dev, absolute_parent.st_ino) != expected_parent_identity
            ):
                raise Task10PublicationError(
                    "Task10 file publication parent identity changed before publication"
                )
            source_status = os.stat(source.name, dir_fd=parent_fd, follow_symlinks=False)
    except Task10PublicationError:
        raise
    except OSError as error:
        raise Task10PublicationError(
            "Task10 partial cannot be inspected for publication"
        ) from error
    source_identity = (source_status.st_dev, source_status.st_ino)
    if stat.S_ISDIR(source_status.st_mode):
        if (
            expected_file_sha256 is not None
            or expected_file_identity is not None
            or expected_directory_sha256 is None
            or expected_directory_identity is None
            or expected_parent_identity is None
            or source_identity != expected_directory_identity
        ):
            raise Task10PublicationError("Task10 directory cannot use a file content identity")
        return _rename_directory_with_reservation(
            source,
            destination,
            expected_sha256=expected_directory_sha256,
            expected_source_identity=expected_directory_identity,
            expected_parent_identity=expected_parent_identity,
        )
    if stat.S_ISREG(source_status.st_mode):
        if (
            expected_file_sha256 is None
            or expected_file_identity is None
            or source_identity != expected_file_identity
            or expected_directory_sha256 is not None
            or expected_directory_identity is not None
            or expected_parent_identity is None
            or parent_fd is None
        ):
            raise Task10PublicationError("Task10 file fallback requires a content identity")
        return _copy_file_with_exclusive_destination(
            source,
            destination,
            expected_sha256=expected_file_sha256,
            expected_source_identity=expected_file_identity,
            expected_parent_identity=expected_parent_identity,
            parent_fd=parent_fd,
        )
    raise Task10PublicationError("Task10 partial has an unsupported fallback type")
