# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""One canonical regular-file tree digest shared by Q30 receipt producers."""

from __future__ import annotations

import json
import os
import re
import stat
from hashlib import sha256
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING, cast

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

__all__ = [
    "canonical_tree_sha256",
    "descriptor_stable_tree_entries",
    "reconcile_q30t_model_asset",
    "require_stable_absolute_tree_root",
]

Q30T_MODEL_REPOSITORY = "Qwen/Qwen3-30B-A3B-Thinking-2507"
Q30T_MODEL_REVISION = "144afc2f379b542fdd4e85a1fcd5e1f79112d95d"
APPROVED_Q30T_MODEL_ROOT = Path(
    "/lustre/fsw/coreai_dlalgo_llm/users/sna/modelopt-qwen3-drafter-training/assets/"
    "q30-thinking-2507-144afc2f379b542fdd4e85a1fcd5e1f79112d95d-v1/model"
)
APPROVED_Q30T_MODEL_IDENTITY_SHA256 = (
    "280341237fdfd95fec51360a1ce4fcbde2b9073e0e99730709aff236b795d20b"
)
APPROVED_Q30T_MODEL_SHA256_MANIFEST_SHA256 = (
    "bbeaafb862b4333478aca69bb4d2fb84eab89844a5bb233e3b18a9195ab23554"
)
_SHA256_LINE = re.compile(r"([0-9a-f]{64})  \./(.+)")
_READ_BLOCK_BYTES = 1024 * 1024


def _file_identity(status: os.stat_result) -> tuple[int, int, int, int, int]:
    return status.st_dev, status.st_ino, status.st_size, status.st_mtime_ns, status.st_ctime_ns


def _directory_identity(status: os.stat_result) -> tuple[int, int]:
    return status.st_dev, status.st_ino


def _open_absolute_directory(path: Path) -> tuple[int, os.stat_result]:
    if not path.is_absolute() or path == Path("/"):
        raise ValueError(f"Q30 tree root must be an absolute non-root path: {path}")
    flags = (
        os.O_RDONLY
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_NONBLOCK", 0)
    )
    try:
        descriptor = os.open(Path("/"), flags)
    except OSError as error:
        raise ValueError("Q30 tree absolute root boundary is unavailable") from error
    current = Path("/")
    try:
        for component in path.parts[1:]:
            current /= component
            try:
                expected = os.stat(component, dir_fd=descriptor, follow_symlinks=False)
                child = os.open(component, flags, dir_fd=descriptor)
            except OSError as error:
                raise ValueError(f"Q30 tree absolute root is unreadable: {current}") from error
            try:
                opened = os.fstat(child)
                if not stat.S_ISDIR(expected.st_mode) or _directory_identity(
                    expected
                ) != _directory_identity(opened):
                    raise ValueError(f"Q30 tree absolute root changed while opening: {current}")
            except BaseException:
                os.close(child)
                raise
            os.close(descriptor)
            descriptor = child
        opened = os.fstat(descriptor)
        pathname = os.stat(path, follow_symlinks=False)
        if not stat.S_ISDIR(pathname.st_mode) or _file_identity(opened) != _file_identity(pathname):
            raise ValueError("Q30 tree absolute root changed while binding")
        return descriptor, opened
    except BaseException:
        os.close(descriptor)
        raise


def require_stable_absolute_tree_root(path: Path, expected: os.stat_result) -> None:
    """Rebind an absolute tree root and require the original directory identity."""
    rebound_fd, rebound = _open_absolute_directory(path)
    try:
        if _file_identity(expected) != _file_identity(rebound):
            raise ValueError("Q30 tree absolute root changed after traversal")
    finally:
        os.close(rebound_fd)


def descriptor_stable_tree_entries(
    root: Path,
    *,
    require_single_link: bool = True,
    copy_root: Path | None = None,
    copy_predicate: Callable[[str], bool] | None = None,
) -> list[dict[str, object]]:
    """Stream a no-follow tree and optionally copy selected regular files."""
    if (copy_root is None) != (copy_predicate is None):
        raise ValueError("Q30 tree copy root and predicate must be provided together")
    flags = (
        os.O_RDONLY
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_NONBLOCK", 0)
    )
    try:
        expected_root = os.stat(root, follow_symlinks=False)
        if not stat.S_ISDIR(expected_root.st_mode):
            raise ValueError("Q30 tree root is not a no-follow directory")
        root_fd = os.open(root, flags)
    except OSError as error:
        raise ValueError(f"Q30 tree root is unreadable: {root}") from error
    root_opened = os.fstat(root_fd)
    if _file_identity(expected_root) != _file_identity(root_opened):
        os.close(root_fd)
        raise ValueError("Q30 tree root changed while opening")
    entries: list[dict[str, object]] = []

    def walk(directory_fd: int, relative: str) -> None:
        before = os.fstat(directory_fd)
        if not stat.S_ISDIR(before.st_mode):
            raise ValueError("Q30 tree directory is invalid")
        with os.scandir(directory_fd) as scan:
            names = sorted(entry.name for entry in scan)
        for name in names:
            path = f"{relative}/{name}" if relative else name
            try:
                expected = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
            except OSError as error:
                raise ValueError(f"Q30 tree entry is unreadable: {path}") from error
            if stat.S_ISREG(expected.st_mode):
                if require_single_link and expected.st_nlink != 1:
                    raise ValueError(f"Q30 tree cannot contain hardlinks: {path}")
                _stream_regular_entry(
                    directory_fd,
                    name,
                    path,
                    expected,
                    entries,
                    copy_root,
                    copy_predicate,
                )
                continue
            if not stat.S_ISDIR(expected.st_mode):
                raise ValueError(f"Q30 tree has unsupported entry: {path}")
            try:
                child_fd = os.open(name, flags, dir_fd=directory_fd)
            except OSError as error:
                raise ValueError(f"Q30 tree directory is unreadable: {path}") from error
            try:
                if _file_identity(expected) != _file_identity(os.fstat(child_fd)):
                    raise ValueError(f"Q30 tree directory changed while opening: {path}")
                walk(child_fd, path)
            finally:
                os.close(child_fd)
        if _file_identity(before) != _file_identity(os.fstat(directory_fd)):
            raise ValueError("Q30 tree changed while walking")

    try:
        walk(root_fd, "")
    finally:
        os.close(root_fd)
    require_stable_absolute_tree_root(root, root_opened)
    entries.sort(key=lambda entry: cast("str", entry["path"]))
    canonical_tree_sha256(entries)
    return entries


def _stream_regular_entry(
    directory_fd: int,
    name: str,
    path: str,
    expected: os.stat_result,
    entries: list[dict[str, object]],
    copy_root: Path | None,
    copy_predicate: Callable[[str], bool] | None,
) -> None:
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0)
    try:
        descriptor = os.open(name, flags, dir_fd=directory_fd)
    except OSError as error:
        raise ValueError(f"Q30 tree regular entry is unreadable: {path}") from error
    destination_descriptor: int | None = None
    try:
        before = os.fstat(descriptor)
        if _file_identity(expected) != _file_identity(before) or not stat.S_ISREG(before.st_mode):
            raise ValueError(f"Q30 tree regular entry identity changed while opening: {path}")
        should_copy = copy_predicate is not None and copy_predicate(path)
        if should_copy:
            assert copy_root is not None
            destination = copy_root / PurePosixPath(path)
            destination.parent.mkdir(mode=0o750, parents=True, exist_ok=True)
            destination_flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
            destination_descriptor = os.open(destination, destination_flags, 0o640)
        digest = sha256()
        size = 0
        with os.fdopen(descriptor, "rb", closefd=False) as stream:
            while block := stream.read(_READ_BLOCK_BYTES):
                digest.update(block)
                size += len(block)
                if destination_descriptor is not None:
                    remaining = memoryview(block)
                    while remaining:
                        written = os.write(destination_descriptor, remaining)
                        if written < 1:
                            raise OSError("Q30 tokenizer asset copy made no progress")
                        remaining = remaining[written:]
        if destination_descriptor is not None:
            os.fsync(destination_descriptor)
        after = os.fstat(descriptor)
        if _file_identity(before) != _file_identity(after) or size != before.st_size:
            raise ValueError(f"Q30 tree regular entry changed while reading: {path}")
        entries.append(
            {"path": path, "type": "regular", "size": size, "sha256": digest.hexdigest()}
        )
    finally:
        if destination_descriptor is not None:
            os.close(destination_descriptor)
        os.close(descriptor)


def canonical_tree_sha256(entries: Sequence[dict[str, object]]) -> str:
    """Validate and hash sorted ``path/type/size/sha256`` tree evidence."""
    previous = ""
    for entry in entries:
        if (
            set(entry) != {"path", "type", "size", "sha256"}
            or type(entry["path"]) is not str
            or not entry["path"]
            or entry["path"] <= previous
            or entry["type"] != "regular"
            or type(entry["size"]) is not int
            or entry["size"] < 0
            or type(entry["sha256"]) is not str
            or len(entry["sha256"]) != 64
            or any(character not in "0123456789abcdef" for character in entry["sha256"])
        ):
            raise ValueError("Q30 tree evidence is not canonical")
        path = PurePosixPath(entry["path"])
        if path.is_absolute() or path.as_posix() != entry["path"] or ".." in path.parts:
            raise ValueError("Q30 tree evidence path is invalid")
        previous = entry["path"]
    if not entries:
        raise ValueError("Q30 tree evidence is empty")
    raw = json.dumps(entries, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    return sha256(raw).hexdigest()


def _stable_regular_bytes(path: Path) -> tuple[bytes, str]:
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0)
    descriptor = os.open(path, flags)
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1:
            raise ValueError(f"Q30 model manifest is not a single-link regular file: {path}")
        digest = sha256()
        payload = bytearray()
        with os.fdopen(descriptor, "rb", closefd=False) as stream:
            while block := stream.read(1024 * 1024):
                digest.update(block)
                payload.extend(block)
                if len(payload) > 64 * 1024 * 1024:
                    raise ValueError(f"Q30 model manifest is too large: {path}")
        after = os.fstat(descriptor)
        identity_before = (
            before.st_dev,
            before.st_ino,
            before.st_size,
            before.st_mtime_ns,
            before.st_ctime_ns,
        )
        identity_after = (
            after.st_dev,
            after.st_ino,
            after.st_size,
            after.st_mtime_ns,
            after.st_ctime_ns,
        )
        if identity_before != identity_after:
            raise ValueError(f"Q30 model manifest changed while reading: {path}")
        return bytes(payload), digest.hexdigest()
    finally:
        os.close(descriptor)


def reconcile_q30t_model_asset(
    *,
    snapshot: Path,
    entries: Sequence[dict[str, object]],
    repository: str,
    revision: str,
    identity_path: Path,
    identity_sha256: str,
    model_sha256_path: Path,
    model_sha256_sha256: str,
) -> str:
    """Bind the reviewed Q30 asset identity and sha256sum inventory to its actual tree."""
    if (
        snapshot != APPROVED_Q30T_MODEL_ROOT
        or repository != Q30T_MODEL_REPOSITORY
        or revision != Q30T_MODEL_REVISION
        or identity_path != snapshot.parent / "manifest/identity.json"
        or model_sha256_path != snapshot.parent / "manifest/model.sha256"
        or identity_sha256 != APPROVED_Q30T_MODEL_IDENTITY_SHA256
        or model_sha256_sha256 != APPROVED_Q30T_MODEL_SHA256_MANIFEST_SHA256
    ):
        raise ValueError("Q30 model asset trust root is invalid")
    identity_raw, actual_identity_sha256 = _stable_regular_bytes(identity_path)
    model_raw, actual_model_sha256 = _stable_regular_bytes(model_sha256_path)
    if actual_identity_sha256 != identity_sha256 or actual_model_sha256 != model_sha256_sha256:
        raise ValueError("Q30 model asset manifest file SHA-256 does not reconcile")
    try:
        identity = json.loads(identity_raw)
        model_lines = model_raw.decode("utf-8").splitlines()
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("Q30 model asset manifest is invalid") from error
    if not model_lines or model_raw != ("\n".join(model_lines) + "\n").encode():
        raise ValueError("Q30 model sha256sum manifest is not canonical")
    inventory: dict[str, str] = {}
    previous = ""
    for line in model_lines:
        match = _SHA256_LINE.fullmatch(line)
        if match is None:
            raise ValueError("Q30 model sha256sum manifest line is invalid")
        digest, name = match.groups()
        path = PurePosixPath(name)
        if (
            path.is_absolute()
            or path.as_posix() != name
            or "\\" in name
            or any(part in {"", ".", ".."} for part in name.split("/"))
            or name <= previous
            or name in inventory
        ):
            raise ValueError("Q30 model sha256sum manifest path is invalid")
        inventory[name] = digest
        previous = name
    actual_inventory = {str(entry["path"]): str(entry["sha256"]) for entry in entries}
    total_bytes = sum(cast("int", entry["size"]) for entry in entries)
    if inventory != actual_inventory:
        raise ValueError("Q30 model sha256sum inventory does not match the snapshot tree")
    expected_identity = {
        "artifact": repository,
        "bytes": total_bytes,
        "files": len(entries),
        "model_sha256_manifest": model_sha256_sha256,
        "source_revision": revision,
        "symlinks": 0,
    }
    if identity != expected_identity:
        raise ValueError("Q30 model identity manifest does not match the snapshot tree")
    return canonical_tree_sha256(entries)
