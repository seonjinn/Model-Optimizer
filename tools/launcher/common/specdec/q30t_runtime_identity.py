# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Canonical, descriptor-stable identity evidence for Q30 runtime trees."""

from __future__ import annotations

import hashlib
import json
import os
import stat
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from collections.abc import Callable

__all__ = [
    "RuntimeTreeEntry",
    "RuntimeTreeIdentity",
    "runtime_tree_entries",
    "runtime_tree_identity",
]

_READ_BLOCK_BYTES = 1024 * 1024


def _post_file_hash_hook() -> None:
    return None


_POST_FILE_HASH_HOOK: Callable[[], None] = _post_file_hash_hook


@dataclass(frozen=True)
class RuntimeTreeEntry:
    """One canonical regular-file or symlink entry in a runtime tree."""

    path: str
    type: Literal["regular", "symlink"]
    size: int | None
    sha256: str | None
    target: str | None


@dataclass(frozen=True)
class RuntimeTreeIdentity:
    """Canonical runtime-tree evidence and its legacy-compatible digest."""

    entries: tuple[RuntimeTreeEntry, ...]
    file_count: int
    symlink_count: int
    total_regular_bytes: int
    sha256: str


def runtime_tree_entries(root: Path) -> tuple[RuntimeTreeEntry, ...]:
    """Return sorted no-follow runtime entries, rejecting unstable tree state."""
    root_fd, root_before = _open_absolute_directory(root)
    entries: list[RuntimeTreeEntry] = []
    try:
        _walk_directory(root_fd, "", entries)
    finally:
        os.close(root_fd)
    rebound_fd, root_after = _open_absolute_directory(root)
    try:
        if _identity(root_before) != _identity(root_after):
            raise ValueError("runtime root changed while traversing")
    finally:
        os.close(rebound_fd)
    entries.sort(key=lambda entry: entry.path)
    return tuple(entries)


def runtime_tree_identity(root: Path) -> RuntimeTreeIdentity:
    """Produce legacy-compatible runtime digest evidence for ``root``."""
    entries = runtime_tree_entries(root)
    legacy = [
        [entry.path, "regular", entry.size, entry.sha256]
        if entry.type == "regular"
        else [entry.path, "symlink", entry.target]
        for entry in entries
    ]
    digest = hashlib.sha256(
        json.dumps(legacy, ensure_ascii=False, separators=(",", ":")).encode()
    ).hexdigest()
    return RuntimeTreeIdentity(
        entries=entries,
        file_count=sum(entry.type == "regular" for entry in entries),
        symlink_count=sum(entry.type == "symlink" for entry in entries),
        total_regular_bytes=sum(entry.size or 0 for entry in entries),
        sha256=digest,
    )


def _open_absolute_directory(path: Path) -> tuple[int, os.stat_result]:
    if not path.is_absolute() or path == Path("/") or ".." in path.parts:
        raise ValueError(f"runtime root must be an absolute non-root path: {path}")
    descriptor = os.open(Path("/"), _directory_open_flags())
    try:
        for component in path.parts[1:]:
            try:
                named = os.stat(component, dir_fd=descriptor, follow_symlinks=False)
                child = os.open(component, _directory_open_flags(), dir_fd=descriptor)
            except OSError as error:
                raise ValueError(f"runtime root is unreadable: {path}") from error
            try:
                opened = os.fstat(child)
                if not stat.S_ISDIR(named.st_mode) or _identity(named) != _identity(opened):
                    raise ValueError(f"runtime root changed while opening: {path}")
            except BaseException:
                os.close(child)
                raise
            os.close(descriptor)
            descriptor = child
        return descriptor, os.fstat(descriptor)
    except BaseException:
        os.close(descriptor)
        raise


def _walk_directory(directory_fd: int, relative: str, entries: list[RuntimeTreeEntry]) -> None:
    before = os.fstat(directory_fd)
    if not stat.S_ISDIR(before.st_mode):
        raise ValueError("runtime directory is invalid")
    with os.scandir(directory_fd) as scan:
        names = sorted(entry.name for entry in scan)
    for name in names:
        path = f"{relative}/{name}" if relative else name
        _require_canonical_relative_path(path)
        try:
            named = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
        except OSError as error:
            raise ValueError(f"runtime entry is unreadable: {path}") from error
        if stat.S_ISREG(named.st_mode):
            entries.append(_hash_regular_entry(directory_fd, name, path, named))
        elif stat.S_ISLNK(named.st_mode):
            entries.append(_read_symlink_entry(directory_fd, name, path, named))
        elif stat.S_ISDIR(named.st_mode):
            _walk_child_directory(directory_fd, name, path, named, entries)
        else:
            raise ValueError(f"runtime contains unsupported entry: {path}")
    _require_directory_stable(directory_fd, relative, before)


def _hash_regular_entry(
    directory_fd: int, name: str, path: str, named: os.stat_result
) -> RuntimeTreeEntry:
    if named.st_nlink != 1:
        raise ValueError(f"runtime regular entry has hard links: {path}")
    try:
        descriptor = os.open(name, _file_open_flags(), dir_fd=directory_fd)
    except OSError as error:
        raise ValueError(f"runtime regular entry is unreadable: {path}") from error
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode) or _identity(named) != _identity(before):
            raise ValueError(f"runtime regular entry changed while opening: {path}")
        digest = hashlib.sha256()
        size = 0
        while block := os.read(descriptor, _READ_BLOCK_BYTES):
            digest.update(block)
            size += len(block)
        _POST_FILE_HASH_HOOK()
        after = os.fstat(descriptor)
        try:
            named_after = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
        except OSError as error:
            raise ValueError(f"runtime regular entry changed while hashing: {path}") from error
        if (
            _identity(before) != _identity(after)
            or _identity(before) != _identity(named_after)
            or not stat.S_ISREG(named_after.st_mode)
            or size != before.st_size
        ):
            raise ValueError(f"runtime regular entry changed while hashing: {path}")
        return RuntimeTreeEntry(path, "regular", size, digest.hexdigest(), None)
    finally:
        os.close(descriptor)


def _read_symlink_entry(
    directory_fd: int, name: str, path: str, named: os.stat_result
) -> RuntimeTreeEntry:
    try:
        target = os.readlink(name, dir_fd=directory_fd)
        named_after = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
    except OSError as error:
        raise ValueError(f"runtime symlink entry is unreadable: {path}") from error
    if not stat.S_ISLNK(named_after.st_mode) or _identity(named) != _identity(named_after):
        raise ValueError(f"runtime symlink entry changed while reading: {path}")
    return RuntimeTreeEntry(path, "symlink", None, None, target)


def _walk_child_directory(
    directory_fd: int,
    name: str,
    path: str,
    named: os.stat_result,
    entries: list[RuntimeTreeEntry],
) -> None:
    try:
        child_fd = os.open(name, _directory_open_flags(), dir_fd=directory_fd)
    except OSError as error:
        raise ValueError(f"runtime directory is unreadable: {path}") from error
    try:
        opened = os.fstat(child_fd)
        if not stat.S_ISDIR(opened.st_mode) or _identity(named) != _identity(opened):
            raise ValueError(f"runtime directory changed while opening: {path}")
        _walk_directory(child_fd, path, entries)
        _require_directory_stable(child_fd, path, opened)
        try:
            named_after = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
        except OSError as error:
            raise ValueError(f"runtime directory changed while traversing: {path}") from error
        if _identity(opened) != _identity(named_after):
            raise ValueError(f"runtime directory changed while traversing: {path}")
    finally:
        os.close(child_fd)


def _require_directory_stable(directory_fd: int, path: str, before: os.stat_result) -> None:
    if _identity(before) != _identity(os.fstat(directory_fd)):
        raise ValueError(f"runtime directory changed while traversing: {path or '.'}")


def _require_canonical_relative_path(path: str) -> None:
    canonical = PurePosixPath(path)
    if (
        not path
        or canonical.is_absolute()
        or canonical.as_posix() != path
        or any(component in {"", ".", ".."} for component in canonical.parts)
    ):
        raise ValueError(f"runtime entry path is not canonical: {path!r}")


def _directory_open_flags() -> int:
    return (
        os.O_RDONLY
        | _nofollow_flag()
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_NONBLOCK", 0)
        | getattr(os, "O_CLOEXEC", 0)
    )


def _file_open_flags() -> int:
    return (
        os.O_RDONLY | _nofollow_flag() | getattr(os, "O_NONBLOCK", 0) | getattr(os, "O_CLOEXEC", 0)
    )


def _nofollow_flag() -> int:
    flag = getattr(os, "O_NOFOLLOW", 0)
    if not isinstance(flag, int) or flag == 0:
        raise ValueError("runtime identity requires O_NOFOLLOW")
    return flag


def _identity(status: os.stat_result) -> tuple[int, int, int, int, int, int]:
    return (
        status.st_dev,
        status.st_ino,
        status.st_nlink,
        status.st_size,
        status.st_mtime_ns,
        status.st_ctime_ns,
    )
