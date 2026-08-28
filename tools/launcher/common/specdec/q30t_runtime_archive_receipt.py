# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Canonical Q30 runtime archive-to-extracted-tree qualification evidence."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import stat
import subprocess
import sys
import tarfile
import tempfile
import unicodedata
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING, Any, Literal, cast

from common.specdec.q30t_runtime_identity import RuntimeTreeEntry, runtime_tree_identity
from common.specdec.qwen4b_b_atomic import atomic_publish_bytes

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

__all__ = [
    "RuntimeArchiveTreeReceipt",
    "canonical_json_bytes",
    "load_runtime_archive_tree_receipt",
    "main",
    "produce_runtime_archive_tree_receipt",
    "stable_regular_file_sha256",
]

SCHEMA_VERSION: Literal["q30t-runtime-archive-tree-v1"] = "q30t-runtime-archive-tree-v1"
_HASH_LENGTH = 64
_SOURCE_COMMIT_LENGTH = 40
_READ_BLOCK_BYTES = 1024 * 1024
_MAX_RECEIPT_BYTES = 1024 * 1024
_SENTINEL_PATHS = frozenset({"bin/activate", "bin/python", "pyvenv.cfg"})
_REQUIRED_SENTINEL_PATHS = frozenset({"bin/python", "pyvenv.cfg"})


def _post_hash_hook() -> None:
    return None


_POST_HASH_HOOK: Callable[[], None] = _post_hash_hook
_PRE_EXTRACT_HOOK: Callable[[], None] = _post_hash_hook


def canonical_json_bytes(value: object) -> bytes:
    """Encode one JSON-compatible value using the receipt canonical form."""
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


@dataclass(frozen=True)
class RuntimeArchiveTreeReceipt:
    """Canonical binding between one runtime archive, tree, and producer."""

    schema_version: Literal["q30t-runtime-archive-tree-v1"]
    archive_path: str
    archive_size: int
    archive_sha256: str
    runtime_tree_sha256: str
    runtime_file_count: int
    runtime_symlink_count: int
    runtime_total_regular_bytes: int
    sentinel_paths: tuple[str, ...]
    sentinel_inventory_sha256: str
    source_commit: str
    producer_path: str
    producer_sha256: str
    receipt_sha256: str

    def body_dict(self) -> dict[str, object]:
        """Return every self-hashed receipt field in canonical key order."""
        return {
            "archive_path": self.archive_path,
            "archive_sha256": self.archive_sha256,
            "archive_size": self.archive_size,
            "producer_path": self.producer_path,
            "producer_sha256": self.producer_sha256,
            "runtime_file_count": self.runtime_file_count,
            "runtime_symlink_count": self.runtime_symlink_count,
            "runtime_total_regular_bytes": self.runtime_total_regular_bytes,
            "runtime_tree_sha256": self.runtime_tree_sha256,
            "schema_version": self.schema_version,
            "sentinel_inventory_sha256": self.sentinel_inventory_sha256,
            "sentinel_paths": list(self.sentinel_paths),
            "source_commit": self.source_commit,
        }

    def to_dict(self) -> dict[str, object]:
        """Return the complete JSON-compatible receipt record."""
        return self.body_dict() | {"receipt_sha256": self.receipt_sha256}

    def canonical_bytes(self) -> bytes:
        """Return canonical JSON without the publication newline."""
        return canonical_json_bytes(self.to_dict())

    def verify_self_hash(self) -> None:
        """Reject a receipt whose body does not reproduce its self-hash."""
        expected = hashlib.sha256(canonical_json_bytes(self.body_dict())).hexdigest()
        if self.receipt_sha256 != expected:
            raise ValueError("runtime archive receipt self-hash mismatch")

    @classmethod
    def from_dict(cls, raw: object) -> RuntimeArchiveTreeReceipt:
        """Decode and validate one exact receipt schema."""
        expected_keys = {
            "archive_path",
            "archive_sha256",
            "archive_size",
            "producer_path",
            "producer_sha256",
            "receipt_sha256",
            "runtime_file_count",
            "runtime_symlink_count",
            "runtime_total_regular_bytes",
            "runtime_tree_sha256",
            "schema_version",
            "sentinel_inventory_sha256",
            "sentinel_paths",
            "source_commit",
        }
        if not isinstance(raw, dict) or set(raw) != expected_keys:
            raise ValueError("runtime archive receipt has an unexpected schema")
        sentinels = raw["sentinel_paths"]
        if not isinstance(sentinels, list) or any(type(item) is not str for item in sentinels):
            raise ValueError("runtime archive receipt sentinel inventory is invalid")
        try:
            receipt = cls(
                schema_version=cast(
                    "Literal['q30t-runtime-archive-tree-v1']",
                    _required_text(raw, "schema_version"),
                ),
                archive_path=_required_text(raw, "archive_path"),
                archive_size=_required_integer(raw, "archive_size"),
                archive_sha256=_required_text(raw, "archive_sha256"),
                runtime_tree_sha256=_required_text(raw, "runtime_tree_sha256"),
                runtime_file_count=_required_integer(raw, "runtime_file_count"),
                runtime_symlink_count=_required_integer(raw, "runtime_symlink_count"),
                runtime_total_regular_bytes=_required_integer(raw, "runtime_total_regular_bytes"),
                sentinel_paths=tuple(sentinels),
                sentinel_inventory_sha256=_required_text(raw, "sentinel_inventory_sha256"),
                source_commit=_required_text(raw, "source_commit"),
                producer_path=_required_text(raw, "producer_path"),
                producer_sha256=_required_text(raw, "producer_sha256"),
                receipt_sha256=_required_text(raw, "receipt_sha256"),
            )
        except (TypeError, ValueError) as error:
            raise ValueError("runtime archive receipt contains invalid values") from error
        receipt._validate()
        return receipt

    def _validate(self) -> None:
        if self.schema_version != SCHEMA_VERSION:
            raise ValueError("runtime archive receipt schema version is invalid")
        if not _is_hash(self.archive_sha256) or not _is_hash(self.runtime_tree_sha256):
            raise ValueError("runtime archive receipt archive/tree hash is invalid")
        if not _is_hash(self.sentinel_inventory_sha256) or not _is_hash(self.producer_sha256):
            raise ValueError("runtime archive receipt evidence hash is invalid")
        if not _is_hash(self.receipt_sha256):
            raise ValueError("runtime archive receipt self-hash is invalid")
        if not _is_lower_hex(self.source_commit, _SOURCE_COMMIT_LENGTH):
            raise ValueError("runtime archive receipt source commit is invalid")
        if self.archive_size <= 0 or any(
            value < 0
            for value in (
                self.runtime_file_count,
                self.runtime_symlink_count,
                self.runtime_total_regular_bytes,
            )
        ):
            raise ValueError("runtime archive receipt counts are invalid")
        _require_canonical_absolute_path(Path(self.archive_path), "archive")
        _require_canonical_absolute_path(Path(self.producer_path), "producer")
        if (
            tuple(sorted(self.sentinel_paths)) != self.sentinel_paths
            or len(set(self.sentinel_paths)) != len(self.sentinel_paths)
            or not _REQUIRED_SENTINEL_PATHS.issubset(self.sentinel_paths)
            or not set(self.sentinel_paths).issubset(_SENTINEL_PATHS)
        ):
            raise ValueError("runtime archive receipt sentinel inventory is invalid")


def stable_regular_file_sha256(path: Path) -> tuple[int, str]:
    """Hash one canonical no-follow single-link file through one descriptor."""
    descriptor, parent_fd, before = _open_stable_regular(path)
    try:
        size, digest = _hash_open_regular(descriptor, path, before)
        _require_named_file_stable(path, descriptor, parent_fd, before, "hashing")
        return size, digest
    finally:
        os.close(descriptor)
        os.close(parent_fd)


def produce_runtime_archive_tree_receipt(
    *,
    archive_path: Path,
    expected_archive_sha256: str,
    extraction_root: Path,
    output_path: Path,
    source_commit: str,
    producer_path: Path,
    producer_sha256: str,
    job_id: str,
) -> RuntimeArchiveTreeReceipt:
    """Extract one authenticated archive and publish its canonical tree receipt.

    ``extraction_root`` is the runtime tree itself. Callers that want a directory
    named ``runtime`` pass that exact path; this function never adds another level.
    """
    _require_python_runtime()
    for path, label in (
        (archive_path, "archive"),
        (extraction_root, "extraction root"),
        (output_path, "output"),
        (producer_path, "producer"),
    ):
        _require_canonical_absolute_path(path, label)
    if not _is_hash(expected_archive_sha256):
        raise ValueError("runtime archive caller SHA-256 is invalid")
    if not _is_hash(producer_sha256):
        raise ValueError("runtime archive producer caller SHA-256 is invalid")
    if not _is_lower_hex(source_commit, _SOURCE_COMMIT_LENGTH):
        raise ValueError("runtime archive source commit is invalid")
    if output_path == producer_path:
        raise ValueError("runtime archive receipt cannot approve itself")
    if _is_within(output_path, extraction_root):
        raise ValueError("runtime archive output cannot be inside the extraction root")

    archive_fd, archive_parent_fd, archive_before = _open_stable_regular(archive_path)
    snapshot_fd: int | None = None
    extraction_parent_fd: int | None = None
    extraction_fd: int | None = None
    try:
        snapshot_fd, archive_size, archive_sha256 = _snapshot_open_regular(
            archive_fd,
            archive_path,
            archive_before,
            private_parent=extraction_root.parent,
        )
        _require_named_file_stable(
            archive_path, archive_fd, archive_parent_fd, archive_before, "hashing"
        )
        if archive_sha256 != expected_archive_sha256:
            raise ValueError("runtime archive SHA-256 mismatch")

        producer_size, observed_producer_sha256 = stable_regular_file_sha256(producer_path)
        if producer_size <= 0 or observed_producer_sha256 != producer_sha256:
            raise ValueError("runtime archive producer SHA-256 mismatch")

        _preflight_archive(snapshot_fd)
        _require_named_file_stable(
            archive_path,
            archive_fd,
            archive_parent_fd,
            archive_before,
            "preflighting",
        )
        extraction_parent_fd, extraction_fd, extraction_before = _create_extraction_root(
            extraction_root
        )
        _PRE_EXTRACT_HOOK()
        _require_extraction_root_stable(
            extraction_root,
            extraction_fd,
            extraction_parent_fd,
            extraction_before,
            "before extracting",
        )
        _extract_archive(snapshot_fd, extraction_fd)
        _require_extraction_root_stable(
            extraction_root,
            extraction_fd,
            extraction_parent_fd,
            extraction_before,
            "after extracting",
        )
        _require_named_file_stable(
            archive_path,
            archive_fd,
            archive_parent_fd,
            archive_before,
            "extracting",
        )
        _require_extraction_root_stable(
            extraction_root,
            extraction_fd,
            extraction_parent_fd,
            extraction_before,
            "before identity",
        )
        identity = runtime_tree_identity(extraction_root)
        _require_extraction_root_stable(
            extraction_root,
            extraction_fd,
            extraction_parent_fd,
            extraction_before,
            "after identity",
        )
    finally:
        if extraction_fd is not None:
            os.close(extraction_fd)
        if extraction_parent_fd is not None:
            os.close(extraction_parent_fd)
        if snapshot_fd is not None:
            os.close(snapshot_fd)
        os.close(archive_fd)
        os.close(archive_parent_fd)

    sentinels = tuple(entry for entry in identity.entries if entry.path in _SENTINEL_PATHS)
    sentinel_paths = tuple(entry.path for entry in sentinels)
    if not _REQUIRED_SENTINEL_PATHS.issubset(sentinel_paths):
        raise ValueError("runtime archive lacks required runtime sentinels")
    sentinel_inventory_sha256 = hashlib.sha256(
        canonical_json_bytes([_entry_dict(entry) for entry in sentinels])
    ).hexdigest()
    body: dict[str, object] = {
        "archive_path": str(archive_path),
        "archive_sha256": archive_sha256,
        "archive_size": archive_size,
        "producer_path": str(producer_path),
        "producer_sha256": observed_producer_sha256,
        "runtime_file_count": identity.file_count,
        "runtime_symlink_count": identity.symlink_count,
        "runtime_total_regular_bytes": identity.total_regular_bytes,
        "runtime_tree_sha256": identity.sha256,
        "schema_version": SCHEMA_VERSION,
        "sentinel_inventory_sha256": sentinel_inventory_sha256,
        "sentinel_paths": list(sentinel_paths),
        "source_commit": source_commit,
    }
    receipt = RuntimeArchiveTreeReceipt.from_dict(
        body | {"receipt_sha256": hashlib.sha256(canonical_json_bytes(body)).hexdigest()}
    )
    receipt.verify_self_hash()
    _publish_or_adopt(output_path, receipt.canonical_bytes() + b"\n", job_id=job_id)
    return receipt


def load_runtime_archive_tree_receipt(
    path: Path, expected_file_sha256: str
) -> RuntimeArchiveTreeReceipt:
    """Stable-read and fully replay one canonical archive/tree receipt."""
    _require_python_runtime()
    if not _is_hash(expected_file_sha256):
        raise ValueError("runtime archive receipt caller SHA-256 is invalid")
    observed_file_sha256, raw = _stable_single_link_bytes(path, maximum_bytes=_MAX_RECEIPT_BYTES)
    if observed_file_sha256 != expected_file_sha256:
        raise ValueError("runtime archive receipt whole-file SHA-256 mismatch")
    try:
        decoded: Any = json.loads(raw, object_pairs_hook=_object_without_duplicates)
    except json.JSONDecodeError as error:
        raise ValueError("runtime archive receipt JSON is invalid") from error
    receipt = RuntimeArchiveTreeReceipt.from_dict(decoded)
    if raw != receipt.canonical_bytes() + b"\n":
        raise ValueError("runtime archive receipt is not canonical")
    receipt.verify_self_hash()
    return receipt


def _required_text(record: dict[object, object], key: str) -> str:
    value = record.get(key)
    if type(value) is not str:
        raise TypeError(f"{key} must be text")
    return value


def _require_python_runtime() -> None:
    if sys.version_info < (3, 12):
        raise RuntimeError("Q30 runtime archive receipts require Python 3.12 or newer")


def _required_integer(record: dict[object, object], key: str) -> int:
    value = record.get(key)
    if type(value) is not int:
        raise TypeError(f"{key} must be an integer")
    return value


def _is_hash(value: object) -> bool:
    return type(value) is str and _is_lower_hex(value, _HASH_LENGTH)


def _is_lower_hex(value: str, length: int) -> bool:
    return len(value) == length and all(character in "0123456789abcdef" for character in value)


def _require_canonical_absolute_path(path: Path, label: str) -> None:
    if (
        not path.is_absolute()
        or path == Path("/")
        or path.parts[0] != "/"
        or ".." in path.parts
        or str(path) != path.as_posix()
    ):
        raise ValueError(f"runtime archive {label} path must be canonical and absolute")


def _is_within(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def _file_identity(status: os.stat_result) -> tuple[int, int, int, int, int, int, int]:
    return (
        status.st_dev,
        status.st_ino,
        status.st_mode,
        status.st_nlink,
        status.st_size,
        status.st_mtime_ns,
        status.st_ctime_ns,
    )


def _open_stable_regular(path: Path) -> tuple[int, int, os.stat_result]:
    _require_canonical_absolute_path(path, "input")
    directory_flags = (
        os.O_RDONLY
        | getattr(os, "O_DIRECTORY", 0)
        | _nofollow_flag()
        | getattr(os, "O_NONBLOCK", 0)
        | getattr(os, "O_CLOEXEC", 0)
    )
    file_flags = (
        os.O_RDONLY | _nofollow_flag() | getattr(os, "O_NONBLOCK", 0) | getattr(os, "O_CLOEXEC", 0)
    )
    try:
        parent_fd = _open_absolute_directory(path.parent, directory_flags)
    except OSError as error:
        raise ValueError(f"runtime archive input parent is unreadable: {path.parent}") from error
    try:
        named = os.stat(path.name, dir_fd=parent_fd, follow_symlinks=False)
        descriptor = os.open(path.name, file_flags, dir_fd=parent_fd)
    except OSError as error:
        os.close(parent_fd)
        raise ValueError(f"runtime archive input is unreadable: {path}") from error
    opened = os.fstat(descriptor)
    if (
        _file_identity(named) != _file_identity(opened)
        or not stat.S_ISREG(opened.st_mode)
        or opened.st_nlink != 1
    ):
        os.close(descriptor)
        os.close(parent_fd)
        raise ValueError(f"runtime archive input must be a single-link regular file: {path}")
    return descriptor, parent_fd, opened


def _open_absolute_directory(path: Path, flags: int) -> int:
    descriptor = os.open(Path("/"), flags)
    try:
        for component in path.parts[1:]:
            named = os.stat(component, dir_fd=descriptor, follow_symlinks=False)
            child = os.open(component, flags, dir_fd=descriptor)
            opened = os.fstat(child)
            if not stat.S_ISDIR(named.st_mode) or _directory_identity(named) != _directory_identity(
                opened
            ):
                os.close(child)
                raise ValueError(f"runtime archive input directory changed: {path}")
            os.close(descriptor)
            descriptor = child
        return descriptor
    except BaseException:
        os.close(descriptor)
        raise


def _create_extraction_root(path: Path) -> tuple[int, int, os.stat_result]:
    directory_flags = (
        os.O_RDONLY
        | getattr(os, "O_DIRECTORY", 0)
        | _nofollow_flag()
        | getattr(os, "O_NONBLOCK", 0)
        | getattr(os, "O_CLOEXEC", 0)
    )
    parent_fd = _open_absolute_directory(path.parent, directory_flags)
    try:
        os.mkdir(path.name, mode=0o700, dir_fd=parent_fd)
        named = os.stat(path.name, dir_fd=parent_fd, follow_symlinks=False)
        descriptor = os.open(path.name, directory_flags, dir_fd=parent_fd)
        opened = os.fstat(descriptor)
        absolute = os.stat(path, follow_symlinks=False)
        if (
            _directory_identity(named) != _directory_identity(opened)
            or _directory_identity(opened) != _directory_identity(absolute)
            or stat.S_IMODE(opened.st_mode) != 0o700
        ):
            os.close(descriptor)
            raise ValueError("runtime archive extraction root changed while opening")
        return parent_fd, descriptor, opened
    except BaseException:
        os.close(parent_fd)
        raise


def _directory_identity(status: os.stat_result) -> tuple[int, int, int]:
    return status.st_dev, status.st_ino, stat.S_IFMT(status.st_mode)


def _require_extraction_root_stable(
    path: Path,
    descriptor: int,
    parent_fd: int,
    expected: os.stat_result,
    phase: str,
) -> None:
    try:
        opened = os.fstat(descriptor)
        named = os.stat(path.name, dir_fd=parent_fd, follow_symlinks=False)
        absolute = os.stat(path, follow_symlinks=False)
    except OSError as error:
        raise ValueError(f"runtime archive extraction root changed {phase}") from error
    expected_identity = _directory_identity(expected)
    if (
        _directory_identity(opened) != expected_identity
        or _directory_identity(named) != expected_identity
        or _directory_identity(absolute) != expected_identity
        or stat.S_IMODE(opened.st_mode) != 0o700
    ):
        raise ValueError(f"runtime archive extraction root changed {phase}")


def _hash_open_regular(descriptor: int, path: Path, before: os.stat_result) -> tuple[int, str]:
    os.lseek(descriptor, 0, os.SEEK_SET)
    digest = hashlib.sha256()
    remaining = before.st_size
    while remaining:
        block = os.read(descriptor, min(_READ_BLOCK_BYTES, remaining))
        if not block:
            raise ValueError(f"runtime archive input changed while hashing: {path}")
        digest.update(block)
        remaining -= len(block)
    if os.read(descriptor, 1):
        raise ValueError(f"runtime archive input changed while hashing: {path}")
    _POST_HASH_HOOK()
    after = os.fstat(descriptor)
    if _file_identity(before) != _file_identity(after):
        raise ValueError(f"runtime archive input changed while hashing: {path}")
    return before.st_size, digest.hexdigest()


def _snapshot_open_regular(
    source_fd: int,
    source_path: Path,
    source_before: os.stat_result,
    *,
    private_parent: Path,
) -> tuple[int, int, str]:
    snapshot_fd = _new_snapshot_descriptor(private_parent)
    try:
        os.lseek(source_fd, 0, os.SEEK_SET)
        digest = hashlib.sha256()
        remaining = source_before.st_size
        while remaining:
            block = os.read(source_fd, min(_READ_BLOCK_BYTES, remaining))
            if not block:
                raise ValueError(f"runtime archive input changed while hashing: {source_path}")
            _write_all(snapshot_fd, block)
            digest.update(block)
            remaining -= len(block)
        if os.read(source_fd, 1):
            raise ValueError(f"runtime archive input changed while hashing: {source_path}")
        _POST_HASH_HOOK()
        source_after = os.fstat(source_fd)
        if _file_identity(source_before) != _file_identity(source_after):
            raise ValueError(f"runtime archive input changed while hashing: {source_path}")
        os.fsync(snapshot_fd)
        snapshot_fd = _make_snapshot_immutable(snapshot_fd, private_parent)
        snapshot = os.fstat(snapshot_fd)
        if not stat.S_ISREG(snapshot.st_mode) or snapshot.st_size != source_before.st_size:
            raise ValueError("runtime archive private snapshot is invalid")
        os.lseek(snapshot_fd, 0, os.SEEK_SET)
        return snapshot_fd, source_before.st_size, digest.hexdigest()
    except BaseException:
        os.close(snapshot_fd)
        raise


def _new_snapshot_descriptor(private_parent: Path) -> int:
    memfd_create = getattr(os, "memfd_create", None)
    allow_sealing = getattr(os, "MFD_ALLOW_SEALING", 0)
    if sys.platform == "linux" and callable(memfd_create) and allow_sealing:
        return memfd_create(
            "q30t-runtime-archive",
            getattr(os, "MFD_CLOEXEC", 0) | allow_sealing,
        )
    private_directory = Path(tempfile.mkdtemp(prefix=".q30t-runtime-snapshot-", dir=private_parent))
    private_directory.chmod(0o700)
    return os.open(
        private_directory / "archive.tar.zst",
        os.O_RDWR | os.O_CREAT | os.O_EXCL | _nofollow_flag() | getattr(os, "O_CLOEXEC", 0),
        0o600,
    )


def _make_snapshot_immutable(descriptor: int, private_parent: Path) -> int:
    if sys.platform == "linux" and hasattr(fcntl, "F_ADD_SEALS"):
        seals = fcntl.F_SEAL_SEAL | fcntl.F_SEAL_SHRINK | fcntl.F_SEAL_GROW | fcntl.F_SEAL_WRITE
        fcntl.fcntl(descriptor, fcntl.F_ADD_SEALS, seals)
        if fcntl.fcntl(descriptor, fcntl.F_GET_SEALS) != seals:
            raise ValueError("runtime archive private snapshot sealing failed")
        return descriptor
    status = os.fstat(descriptor)
    os.fchmod(descriptor, 0o400)
    snapshot_path = Path(_descriptor_path(descriptor)).resolve()
    readonly = os.open(snapshot_path, os.O_RDONLY | _nofollow_flag() | getattr(os, "O_CLOEXEC", 0))
    opened = os.fstat(readonly)
    if (status.st_dev, status.st_ino, status.st_size) != (
        opened.st_dev,
        opened.st_ino,
        opened.st_size,
    ):
        os.close(readonly)
        raise ValueError(f"runtime archive private snapshot changed under {private_parent}")
    os.close(descriptor)
    return readonly


def _write_all(descriptor: int, block: bytes) -> None:
    view = memoryview(block)
    while view:
        written = os.write(descriptor, view)
        if written <= 0:
            raise OSError("runtime archive private snapshot write made no progress")
        view = view[written:]


def _require_named_file_stable(
    path: Path,
    descriptor: int,
    parent_fd: int,
    expected: os.stat_result,
    phase: str,
) -> None:
    try:
        named = os.stat(path.name, dir_fd=parent_fd, follow_symlinks=False)
        absolute = os.stat(path, follow_symlinks=False)
    except OSError as error:
        raise ValueError(f"runtime archive input changed while {phase}: {path}") from error
    opened = os.fstat(descriptor)
    if (
        _file_identity(expected) != _file_identity(opened)
        or _file_identity(expected) != _file_identity(named)
        or _file_identity(expected) != _file_identity(absolute)
    ):
        raise ValueError(f"runtime archive input changed while {phase}: {path}")


def _nofollow_flag() -> int:
    flag = getattr(os, "O_NOFOLLOW", 0)
    if not isinstance(flag, int) or flag == 0:
        raise ValueError("runtime archive receipt requires O_NOFOLLOW")
    return flag


def _descriptor_path(descriptor: int) -> str:
    linux_path = f"/proc/self/fd/{descriptor}"
    if Path("/proc/self/fd").is_dir():
        return linux_path
    return f"/dev/fd/{descriptor}"


def _zstd_executable() -> str:
    approved = Path("/usr/bin/zstd")
    if approved.is_file():
        return str(approved)
    if sys.platform != "linux":
        fallback = Path("/opt/homebrew/bin/zstd")
        if fallback.is_file():
            return str(fallback)
    raise ValueError("approved zstd executable is unavailable")


def _tool_environment() -> dict[str, str]:
    if sys.platform == "linux":
        return {"LC_ALL": "C", "PATH": "/usr/bin:/bin"}
    return {
        "LC_ALL": "C",
        "PATH": f"{Path(_zstd_executable()).parent}:/usr/bin:/bin",
    }


def _preflight_archive(descriptor: int) -> None:
    os.lseek(descriptor, 0, os.SEEK_SET)
    process = subprocess.Popen(
        (
            _zstd_executable(),
            "-q",
            "--decompress",
            "--stdout",
            "--",
            _descriptor_path(descriptor),
        ),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=_tool_environment(),
        pass_fds=(descriptor,),
    )
    seen: set[str] = set()
    try:
        if process.stdout is None:
            raise RuntimeError("zstd preflight pipe is unavailable")
        with process.stdout, tarfile.open(fileobj=process.stdout, mode="r|") as archive:
            for member in archive:
                canonical = _canonical_member_path(
                    member.name, is_directory=member.type == tarfile.DIRTYPE
                )
                if canonical is None:
                    continue
                if canonical in seen:
                    raise ValueError(f"duplicate archive member path: {canonical}")
                seen.add(canonical)
                if member.type in {tarfile.REGTYPE, tarfile.AREGTYPE, tarfile.DIRTYPE}:
                    continue
                if member.type == tarfile.SYMTYPE:
                    _require_safe_link_target(canonical, member.linkname)
                    continue
                raise ValueError(f"unsupported archive member: {canonical}")
        return_code = process.wait()
        stderr = process.stderr.read(_MAX_RECEIPT_BYTES) if process.stderr is not None else b""
        if return_code != 0:
            raise ValueError(
                f"runtime archive zstd preflight failed: {stderr.decode(errors='replace')[:512]}"
            )
    except BaseException:
        if process.poll() is None:
            process.terminate()
        process.wait()
        raise
    finally:
        if process.stderr is not None:
            process.stderr.close()


def _canonical_member_path(name: str, *, is_directory: bool) -> str | None:
    try:
        name.encode("utf-8")
    except UnicodeEncodeError as error:
        raise ValueError("unsafe archive member path is not UTF-8") from error
    if _contains_unicode_control(name):
        raise ValueError(f"unsafe archive member path: {name!r}")
    if name == ".":
        if is_directory:
            return None
        raise ValueError(f"unsafe archive member path: {name!r}")
    relative = name.removeprefix("./")
    candidate = PurePosixPath(relative)
    if (
        not relative
        or candidate.is_absolute()
        or candidate.as_posix() != relative
        or any(component in {"", ".", ".."} for component in candidate.parts)
    ):
        raise ValueError(f"unsafe archive member path: {name!r}")
    return relative


def _require_safe_link_target(member_path: str, link_target: str) -> None:
    try:
        link_target.encode("utf-8")
    except UnicodeEncodeError as error:
        raise ValueError("unsafe archive link target is not UTF-8") from error
    target = PurePosixPath(link_target)
    if not link_target or target.is_absolute() or _contains_unicode_control(link_target):
        raise ValueError(f"unsafe archive link target: {link_target!r}")
    resolved = list(PurePosixPath(member_path).parent.parts)
    for component in target.parts:
        if component in {"", "."}:
            continue
        if component == "..":
            if not resolved:
                raise ValueError(f"unsafe archive link target: {link_target!r}")
            resolved.pop()
            continue
        resolved.append(component)


def _contains_unicode_control(value: str) -> bool:
    return any(unicodedata.category(character).startswith("C") for character in value)


def _extract_archive(archive_fd: int, extraction_fd: int) -> None:
    os.lseek(archive_fd, 0, os.SEEK_SET)
    subprocess.run(
        (
            "/usr/bin/tar",
            "--zstd",
            "--extract",
            "--no-same-owner",
            "--no-same-permissions",
            "--file",
            _descriptor_path(archive_fd),
            "--directory",
            ".",
        ),
        check=True,
        env=_tool_environment(),
        pass_fds=(archive_fd, extraction_fd),
        preexec_fn=lambda: os.fchdir(extraction_fd),
    )


def _entry_dict(entry: RuntimeTreeEntry) -> dict[str, object]:
    return {
        "path": entry.path,
        "sha256": entry.sha256,
        "size": entry.size,
        "target": entry.target,
        "type": entry.type,
    }


def _object_without_duplicates(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"runtime archive receipt has duplicate JSON key: {key}")
        result[key] = value
    return result


def _stable_single_link_bytes(path: Path, *, maximum_bytes: int) -> tuple[str, bytes]:
    descriptor, parent_fd, before = _open_stable_regular(path)
    try:
        if before.st_size > maximum_bytes:
            raise ValueError("runtime archive receipt is too large")
        retained = bytearray()
        digest = hashlib.sha256()
        remaining = before.st_size
        while remaining:
            block = os.read(descriptor, min(_READ_BLOCK_BYTES, remaining))
            if not block:
                raise ValueError("runtime archive receipt changed while reading")
            retained.extend(block)
            digest.update(block)
            remaining -= len(block)
        if os.read(descriptor, 1):
            raise ValueError("runtime archive receipt changed while reading")
        after = os.fstat(descriptor)
        _require_named_file_stable(path, descriptor, parent_fd, before, "reading")
        if _file_identity(before) != _file_identity(after) or len(retained) != before.st_size:
            raise ValueError("runtime archive receipt changed while reading")
        return digest.hexdigest(), bytes(retained)
    finally:
        os.close(descriptor)
        os.close(parent_fd)


def _adopt_exact(path: Path, expected: bytes) -> None:
    try:
        _, observed = _stable_single_link_bytes(path, maximum_bytes=len(expected))
    except (OSError, ValueError) as error:
        raise FileExistsError(f"runtime archive receipt cannot be adopted: {path}") from error
    if observed != expected:
        raise FileExistsError(f"runtime archive receipt differs: {path}")


def _publish_or_adopt(path: Path, payload: bytes, *, job_id: str) -> None:
    if os.path.lexists(path):
        _adopt_exact(path, payload)
        return
    try:
        atomic_publish_bytes(path, payload, job_id=job_id)
    except FileExistsError:
        if not os.path.lexists(path):
            raise
        _adopt_exact(path, payload)
        return
    _adopt_exact(path, payload)


def _arguments(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Produce or verify a Q30 runtime archive receipt.")
    commands = parser.add_subparsers(dest="command", required=True)
    produce = commands.add_parser("produce", help="extract and publish one receipt")
    produce.add_argument("--archive", type=Path, required=True)
    produce.add_argument("--archive-sha256", required=True)
    produce.add_argument("--extraction-root", type=Path, required=True)
    produce.add_argument("--output", type=Path, required=True)
    produce.add_argument("--source-commit", required=True)
    produce.add_argument("--producer", type=Path, required=True)
    produce.add_argument("--producer-sha256", required=True)
    produce.add_argument("--job-id", required=True)
    verify = commands.add_parser("verify", help="replay one canonical receipt")
    verify.add_argument("--receipt", type=Path, required=True)
    verify.add_argument("--receipt-sha256", required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Run the archive receipt producer or verifier CLI."""
    _require_python_runtime()
    arguments = _arguments(argv)
    if arguments.command == "produce":
        receipt = produce_runtime_archive_tree_receipt(
            archive_path=arguments.archive,
            expected_archive_sha256=arguments.archive_sha256,
            extraction_root=arguments.extraction_root,
            output_path=arguments.output,
            source_commit=arguments.source_commit,
            producer_path=arguments.producer,
            producer_sha256=arguments.producer_sha256,
            job_id=arguments.job_id,
        )
    else:
        receipt = load_runtime_archive_tree_receipt(arguments.receipt, arguments.receipt_sha256)
    sys.stdout.buffer.write(receipt.canonical_bytes() + b"\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
