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
from pathlib import Path
from typing import Literal

__all__ = [
    "Task10PublicationError",
    "Task10RecoveryState",
    "atomic_publish_bytes",
    "atomic_publish_directory",
    "recovery_state",
]


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


def atomic_publish_bytes(destination: Path, payload: bytes, *, job_id: str) -> None:
    """Publish one durable file without replacing a prior pathname."""
    if re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", job_id) is None:
        raise ValueError("Task10 publication job_id is unsafe")
    destination.parent.mkdir(parents=True, exist_ok=True)
    if os.path.lexists(destination):
        raise FileExistsError(f"immutable Task10 artifact already exists: {destination}")
    partial = destination.with_name(f".{destination.name}.partial-{job_id}")
    phase = "partial_setup"
    expected: tuple[int, int] | None = None
    try:
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
        descriptor = os.open(partial, flags, 0o600)
        metadata = os.fstat(descriptor)
        expected = (metadata.st_dev, metadata.st_ino)
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        phase = "reread"
        if _reread_created_file(partial, expected) != payload:
            raise Task10PublicationError("Task10 partial reread identity mismatch")
        phase = "rename"
        _rename_no_replace(partial, destination)
        if _observe(destination).identity != expected:
            raise Task10PublicationError("Task10 destination inode differs from its partial")
        phase = "parent_fsync"
        _fsync_directory(destination.parent)
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


def atomic_publish_directory(partial: Path, destination: Path) -> None:
    """Publish a fully fsynced sibling directory without replacement."""
    expected = _observe(partial).identity
    phase = "directory_fsync"
    try:
        if expected is None or partial.parent.resolve() != destination.parent.resolve():
            raise Task10PublicationError("Task10 partial must be a present sibling directory")
        _fsync_tree(partial)
        if _observe(partial).identity != expected:
            raise Task10PublicationError("Task10 partial inode changed before rename")
        phase = "rename"
        _rename_no_replace(partial, destination)
        if _observe(destination).identity != expected:
            raise Task10PublicationError("Task10 destination inode differs from its partial")
        phase = "parent_fsync"
        _fsync_directory(destination.parent)
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


def _reread_created_file(path: Path, expected: tuple[int, int]) -> bytes:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    try:
        before = os.fstat(descriptor)
        if (before.st_dev, before.st_ino) != expected or not stat.S_ISREG(before.st_mode):
            raise Task10PublicationError("Task10 partial inode changed before reread")
        with os.fdopen(descriptor, "rb") as stream:
            payload = stream.read()
            after = os.fstat(stream.fileno())
    except BaseException:
        raise
    if (after.st_dev, after.st_ino) != expected or _observe(path).identity != expected:
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


def _rename_no_replace(source: Path, destination: Path) -> None:
    library = ctypes.CDLL(None, use_errno=True)
    source_bytes = os.fsencode(source)
    destination_bytes = os.fsencode(destination)
    if platform.system() == "Linux":
        try:
            rename = library.renameat2
        except AttributeError as error:
            raise Task10PublicationError("atomic no-replace rename is unavailable") from error
        rename.argtypes = [
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_uint,
        ]
        rename.restype = ctypes.c_int
        result = rename(-100, source_bytes, -100, destination_bytes, 1)
    elif platform.system() == "Darwin":
        rename = library.renamex_np
        rename.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_uint]
        rename.restype = ctypes.c_int
        result = rename(source_bytes, destination_bytes, 0x00000004)
    else:
        raise Task10PublicationError(
            f"atomic no-replace rename is unsupported on {platform.system()}"
        )
    if result == 0:
        return
    number = ctypes.get_errno()
    if number == errno.EEXIST:
        raise FileExistsError(number, os.strerror(number), destination)
    raise Task10PublicationError(f"atomic no-replace rename failed: {os.strerror(number)}")
