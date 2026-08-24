# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Publish the immutable full-201 PTV2 source manifest from Hao's symlink view."""

from __future__ import annotations

import argparse
import concurrent.futures
import ctypes
import errno
import hashlib
import json
import os
import platform
import re
import stat
import sys
import time
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

__all__ = [
    "APPROVED_PTV2_REVISION",
    "PRODUCTION_SPLIT_COUNTS",
    "PTV2SourceManifestError",
    "SourceManifestBootstrapResult",
    "SourceManifestPublicationPhase",
    "SourceManifestRecoveryState",
    "bootstrap_ptv2_source_manifest",
    "source_manifest_recovery_state",
]

APPROVED_REPOSITORY = "nvidia/Nemotron-Post-Training-Dataset-v2"
APPROVED_CONFIGURATION = "default"
APPROVED_PTV2_REVISION = "5c89e01dd720ae0f4058445ed49c5fb68a03c76e"
APPROVED_LICENSE_EXPRESSION = "CC-BY-4.0"
PRODUCTION_SPLIT_COUNTS: Mapping[str, int] = {
    "chat": 12,
    "math": 2,
    "code": 2,
    "stem": 2,
    "multilingual_de": 38,
    "multilingual_ja": 37,
    "multilingual_es": 33,
    "multilingual_fr": 37,
    "multilingual_it": 38,
}
_SOURCE_COMMIT = re.compile(r"^[0-9a-f]{40}$")
_CHUNK_BYTES = 1024 * 1024


class PTV2SourceManifestError(RuntimeError):
    """The immutable source view or its publication fails the approved contract."""


class SourceManifestPublicationPhase(str, Enum):
    """Last attempted operation when bootstrap publication became ambiguous."""

    PARTIAL_SETUP = "partial_setup"
    WRITE = "write"
    DIRECTORY_FSYNC = "directory_fsync"
    RENAME = "rename"
    PARENT_FSYNC = "parent_fsync"


@dataclass(frozen=True)
class SourceManifestPathObservation:
    """A no-follow point-in-time observation used for recovery."""

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
class SourceManifestRecoveryState:
    """Typed evidence for a preserved or ambiguously installed bootstrap bundle."""

    phase: SourceManifestPublicationPhase
    partial_path: Path
    destination_path: Path
    expected_parent_identity: tuple[int, int]
    expected_partial_identity: tuple[int, int] | None
    parent_observation: SourceManifestPathObservation
    partial_observation: SourceManifestPathObservation
    destination_observation: SourceManifestPathObservation
    recovery_required: Literal[True] = True


def source_manifest_recovery_state(error: BaseException) -> SourceManifestRecoveryState:
    """Return typed bootstrap recovery evidence carried by an exception."""
    state = getattr(error, "recovery_state", None)
    if not isinstance(state, SourceManifestRecoveryState):
        raise ValueError("exception does not carry source-manifest recovery state")
    return state


@dataclass(frozen=True)
class SourceManifestBootstrapResult:
    """Paths and immutable identity of a published source-manifest bundle."""

    output_root: Path
    manifest_path: Path
    completion_path: Path
    manifest_sha256: str


@dataclass(frozen=True)
class _FileEvidence:
    split: str
    basename: str
    logical_path: str
    bytes: int
    sha256: str
    target_realpath: str
    target_device: int
    target_inode: int
    target_mtime_ns: int
    symlink_device: int | None
    symlink_inode: int | None
    symlink_text: str | None

    def target_record(self) -> dict[str, Any]:
        return {
            "basename": self.basename,
            "bytes": self.bytes,
            "sha256": self.sha256,
            "symlink_device": self.symlink_device,
            "symlink_inode": self.symlink_inode,
            "symlink_text": self.symlink_text,
            "target_device": self.target_device,
            "target_inode": self.target_inode,
            "target_mtime_ns": self.target_mtime_ns,
            "target_realpath": self.target_realpath,
        }


def _canonical_json(value: object) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode()


def _stat_identity(metadata: os.stat_result) -> tuple[int, int, int, int]:
    return metadata.st_dev, metadata.st_ino, metadata.st_size, metadata.st_mtime_ns


def _is_within(path: Path, root: Path) -> bool:
    return path != root and path.is_relative_to(root)


def _hash_approved_file(
    path: Path,
    *,
    approved_root: Path,
    split: str,
    must_be_symlink: bool,
) -> _FileEvidence:
    try:
        link_before = os.lstat(path)
    except OSError as error:
        raise PTV2SourceManifestError(f"source path is unavailable: {path.name}") from error
    is_symlink = stat.S_ISLNK(link_before.st_mode)
    if must_be_symlink and not is_symlink:
        raise PTV2SourceManifestError(f"source shard is not a symlink: {path.name}")
    if not must_be_symlink and not (is_symlink or stat.S_ISREG(link_before.st_mode)):
        raise PTV2SourceManifestError(f"metadata path is not a regular file: {path}")
    try:
        symlink_text = os.readlink(path) if is_symlink else None
        target = path.resolve(strict=True)
    except OSError as error:
        raise PTV2SourceManifestError(f"source target is unavailable: {path.name}") from error
    if not _is_within(target, approved_root):
        raise PTV2SourceManifestError(f"source target escapes approved Hao root: {path.name}")
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(target, flags)
    except OSError as error:
        raise PTV2SourceManifestError(f"unable to open source target: {path.name}") from error
    digest = hashlib.sha256()
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode) or before.st_size <= 0:
            raise PTV2SourceManifestError(
                f"source target is not a non-empty regular file: {path.name}"
            )
        while chunk := os.read(descriptor, _CHUNK_BYTES):
            digest.update(chunk)
        after = os.fstat(descriptor)
    except OSError as error:
        raise PTV2SourceManifestError(f"unable to hash source target: {path.name}") from error
    finally:
        os.close(descriptor)
    if _stat_identity(before) != _stat_identity(after):
        raise PTV2SourceManifestError(f"source target changed while hashing: {path.name}")
    try:
        link_after = os.lstat(path)
        target_after = path.resolve(strict=True)
        symlink_text_after = os.readlink(path) if is_symlink else None
    except OSError as error:
        raise PTV2SourceManifestError(
            f"source pathname changed while hashing: {path.name}"
        ) from error
    if (
        (link_before.st_dev, link_before.st_ino, link_before.st_mtime_ns)
        != (link_after.st_dev, link_after.st_ino, link_after.st_mtime_ns)
        or target_after != target
        or symlink_text_after != symlink_text
    ):
        raise PTV2SourceManifestError(f"source pathname changed while hashing: {path.name}")
    return _FileEvidence(
        split=split,
        basename=path.name,
        logical_path=f"data/{path.name}",
        bytes=before.st_size,
        sha256=digest.hexdigest(),
        target_realpath=str(target),
        target_device=before.st_dev,
        target_inode=before.st_ino,
        target_mtime_ns=before.st_mtime_ns,
        symlink_device=link_before.st_dev if is_symlink else None,
        symlink_inode=link_before.st_ino if is_symlink else None,
        symlink_text=symlink_text,
    )


def _expected_names(split_counts: Mapping[str, int]) -> list[tuple[str, str]]:
    names: list[tuple[str, str]] = []
    for split, count in split_counts.items():
        if isinstance(count, bool) or not isinstance(count, int) or count <= 0:
            raise PTV2SourceManifestError(f"invalid shard count for split: {split}")
        names.extend(
            (split, f"{split}-{index:05d}-of-{count:05d}.parquet") for index in range(count)
        )
    return names


def _validate_source_root(source_root: Path, expected_names: Sequence[str]) -> None:
    try:
        metadata = os.lstat(source_root)
    except OSError as error:
        raise PTV2SourceManifestError("exact symlink root is unavailable") from error
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode):
        raise PTV2SourceManifestError("exact symlink root must be a no-follow directory")
    actual = {entry.name for entry in os.scandir(source_root)}
    expected = set(expected_names)
    if actual != expected:
        missing = sorted(expected - actual)
        orphan = sorted(actual - expected)
        raise PTV2SourceManifestError(
            f"source root shard set mismatch; missing={missing[:3]}, orphan={orphan[:3]}"
        )


def _hash_shards(
    jobs: Sequence[tuple[str, Path]], *, approved_root: Path, workers: int
) -> tuple[list[_FileEvidence], int]:
    if isinstance(workers, bool) or not isinstance(workers, int) or workers <= 0:
        raise PTV2SourceManifestError("workers must be a positive integer")
    effective = min(workers, len(jobs))

    def run(job: tuple[str, Path]) -> _FileEvidence:
        split, path = job
        return _hash_approved_file(
            path, approved_root=approved_root, split=split, must_be_symlink=True
        )

    executor = concurrent.futures.ThreadPoolExecutor(max_workers=effective)
    futures = [executor.submit(run, job) for job in jobs]
    try:
        records = [future.result() for future in futures]
    except BaseException:
        for future in futures:
            future.cancel()
        executor.shutdown(wait=True, cancel_futures=True)
        raise
    executor.shutdown(wait=True)
    return records, effective


def _manifest_payload(
    evidence: Sequence[_FileEvidence], split_counts: Mapping[str, int]
) -> dict[str, Any]:
    by_split = {split: [] for split in split_counts}
    for record in evidence:
        by_split[record.split].append(
            {"path": record.logical_path, "bytes": record.bytes, "sha256": record.sha256}
        )
    return {
        "schema_version": 1,
        "name": "qwen3-4b-ptv2-full-source-v1",
        "sources": [
            {
                "repository_id": APPROVED_REPOSITORY,
                "configuration": APPROVED_CONFIGURATION,
                "split": split,
                "revision": APPROVED_PTV2_REVISION,
                "license_expression": APPROVED_LICENSE_EXPRESSION,
                "approved_use": True,
                "cell": split,
                "lane": "target-synth",
                "files": by_split[split],
            }
            for split in split_counts
        ],
    }


def _write_durable(path: Path, payload: bytes) -> None:
    descriptor = os.open(
        path,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
        0o440,
    )
    try:
        offset = 0
        while offset < len(payload):
            offset += os.write(descriptor, payload[offset:])
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    try:
        if os.read(descriptor, len(payload) + 1) != payload:
            raise PTV2SourceManifestError(f"durable reread mismatch: {path.name}")
    finally:
        os.close(descriptor)


def _read_regular_no_follow(path: Path) -> bytes:
    try:
        descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    except OSError as error:
        raise PTV2SourceManifestError(f"immutable artifact is unavailable: {path.name}") from error
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise PTV2SourceManifestError(f"immutable artifact is not regular: {path.name}")
        chunks: list[bytes] = []
        while chunk := os.read(descriptor, _CHUNK_BYTES):
            chunks.append(chunk)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    if _stat_identity(before) != _stat_identity(after):
        raise PTV2SourceManifestError(f"immutable artifact changed while reading: {path.name}")
    try:
        pathname = os.lstat(path)
    except OSError as error:
        raise PTV2SourceManifestError(
            f"immutable artifact pathname changed: {path.name}"
        ) from error
    if (pathname.st_dev, pathname.st_ino) != (after.st_dev, after.st_ino):
        raise PTV2SourceManifestError(f"immutable artifact pathname changed: {path.name}")
    return b"".join(chunks)


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _native_rename_no_replace(source: Path, destination: Path) -> None:
    library = ctypes.CDLL(None, use_errno=True)
    source_bytes = os.fsencode(source)
    destination_bytes = os.fsencode(destination)
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
        result = rename(-100, source_bytes, -100, destination_bytes, 1)
    elif platform.system() == "Darwin":
        rename = library.renamex_np
        rename.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_uint]
        rename.restype = ctypes.c_int
        result = rename(source_bytes, destination_bytes, 0x00000004)
    else:
        raise OSError(errno.ENOSYS, "atomic no-replace rename is unsupported")
    if result == 0:
        return
    error_number = ctypes.get_errno()
    raise OSError(error_number, os.strerror(error_number), destination)


def _observe(path: Path) -> SourceManifestPathObservation:
    try:
        metadata = os.lstat(path)
    except FileNotFoundError:
        return SourceManifestPathObservation(path, "absent", None, None, None, None)
    except OSError as error:
        return SourceManifestPathObservation(path, "unavailable", None, None, None, str(error))
    return SourceManifestPathObservation(
        path, "present", metadata.st_dev, metadata.st_ino, metadata.st_mode, None
    )


def _rename_with_directory_reservation(source: Path, destination: Path) -> None:
    """Reserve an absent sibling pathname before a portable directory rename."""
    if source.parent.absolute() != destination.parent.absolute():
        raise PTV2SourceManifestError("publication partial must be a destination sibling")
    parent = source.parent
    parent_observation = _observe(parent)
    source_observation = _observe(source)
    if (
        parent_observation.identity is None
        or parent_observation.mode is None
        or stat.S_ISLNK(parent_observation.mode)
        or not stat.S_ISDIR(parent_observation.mode)
    ):
        raise PTV2SourceManifestError("publication parent is not a no-follow directory")
    if (
        source_observation.identity is None
        or source_observation.mode is None
        or stat.S_ISLNK(source_observation.mode)
        or not stat.S_ISDIR(source_observation.mode)
    ):
        raise PTV2SourceManifestError("publication partial is not a no-follow directory")
    try:
        destination.mkdir(mode=0o700)
    except FileExistsError:
        raise FileExistsError(errno.EEXIST, os.strerror(errno.EEXIST), destination) from None
    reservation = _observe(destination)
    if (
        reservation.identity is None
        or reservation.mode is None
        or stat.S_ISLNK(reservation.mode)
        or not stat.S_ISDIR(reservation.mode)
    ):
        raise PTV2SourceManifestError("publication reservation is not a no-follow directory")
    _fsync_directory(parent)
    if _observe(parent).identity != parent_observation.identity:
        raise PTV2SourceManifestError("publication parent inode changed after reservation")
    if _observe(destination).identity != reservation.identity:
        raise PTV2SourceManifestError("publication reservation inode changed before rename")
    with os.scandir(destination) as entries:
        if next(entries, None) is not None:
            raise PTV2SourceManifestError("publication reservation changed before rename")
    try:
        os.rename(source, destination)
    except OSError as error:
        collision_numbers = {errno.EEXIST, errno.ENOTEMPTY}
        if error.errno in collision_numbers:
            raise FileExistsError(errno.EEXIST, os.strerror(errno.EEXIST), destination) from error
        raise
    if _observe(destination).identity != source_observation.identity:
        raise PTV2SourceManifestError("reserved destination inode differs from its partial")


def _rename_no_replace(source: Path, destination: Path) -> None:
    try:
        _native_rename_no_replace(source, destination)
        return
    except OSError as error:
        if error.errno == errno.EEXIST:
            raise FileExistsError(errno.EEXIST, os.strerror(errno.EEXIST), destination) from error
        unsupported = {errno.EINVAL, errno.ENOSYS}
        unsupported.update(
            number
            for number in (getattr(errno, "EOPNOTSUPP", None), getattr(errno, "ENOTSUP", None))
            if number is not None
        )
        if error.errno not in unsupported:
            raise PTV2SourceManifestError(
                f"atomic no-replace rename failed: {os.strerror(error.errno or errno.EIO)}"
            ) from error
    _rename_with_directory_reservation(source, destination)


def _result(output_root: Path, manifest_sha256: str) -> SourceManifestBootstrapResult:
    return SourceManifestBootstrapResult(
        output_root=output_root,
        manifest_path=output_root / "SOURCE_PLAN.json",
        completion_path=output_root / "SOURCE_MANIFEST_COMPLETION.json",
        manifest_sha256=manifest_sha256,
    )


def _validate_existing(
    output_root: Path,
    *,
    manifest_bytes: bytes,
    stable_receipt: Mapping[str, Any],
) -> SourceManifestBootstrapResult:
    try:
        metadata = os.lstat(output_root)
    except OSError as error:
        raise PTV2SourceManifestError("immutable publication is unavailable") from error
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode):
        raise PTV2SourceManifestError("immutable publication mismatch: output is not a directory")
    manifest_path = output_root / "SOURCE_PLAN.json"
    receipt_path = output_root / "SOURCE_MANIFEST_COMPLETION.json"
    if {entry.name for entry in os.scandir(output_root)} != {
        manifest_path.name,
        receipt_path.name,
    }:
        raise PTV2SourceManifestError("immutable publication mismatch: bundle file set")
    try:
        actual_manifest = _read_regular_no_follow(manifest_path)
        receipt = json.loads(_read_regular_no_follow(receipt_path))
    except (OSError, json.JSONDecodeError) as error:
        raise PTV2SourceManifestError(
            "immutable publication mismatch: unreadable bundle"
        ) from error
    if actual_manifest != manifest_bytes or not isinstance(receipt, dict):
        raise PTV2SourceManifestError("immutable publication mismatch: manifest identity changed")
    for key, value in stable_receipt.items():
        if receipt.get(key) != value:
            raise PTV2SourceManifestError(f"immutable publication mismatch: {key}")
    manifest_sha256 = hashlib.sha256(manifest_bytes).hexdigest()
    return _result(output_root, manifest_sha256)


def _bootstrap_source_manifest(
    *,
    source_root: Path,
    approved_hao_root: Path,
    readme_path: Path,
    output_root: Path,
    source_commit: str,
    workers: int,
    split_counts: Mapping[str, int],
) -> SourceManifestBootstrapResult:
    if _SOURCE_COMMIT.fullmatch(source_commit) is None:
        raise PTV2SourceManifestError("source_commit must be an exact lowercase commit")
    started_wall = datetime.now(UTC)
    started = time.monotonic()
    try:
        source_root_real = source_root.expanduser().resolve(strict=True)
        approved_root_real = approved_hao_root.expanduser().resolve(strict=True)
    except OSError as error:
        raise PTV2SourceManifestError("source or approved Hao root is unavailable") from error
    expected = _expected_names(split_counts)
    _validate_source_root(source_root, [name for _, name in expected])
    jobs = [(split, source_root / name) for split, name in expected]
    try:
        records, effective_workers = _hash_shards(
            jobs, approved_root=approved_root_real, workers=workers
        )
        readme = _hash_approved_file(
            readme_path,
            approved_root=approved_root_real,
            split="README",
            must_be_symlink=False,
        )
    except PTV2SourceManifestError:
        raise
    except BaseException as error:
        raise PTV2SourceManifestError("parallel source hashing failed") from error
    manifest_bytes = _canonical_json(_manifest_payload(records, split_counts))
    manifest_sha256 = hashlib.sha256(manifest_bytes).hexdigest()
    split_bytes = {
        split: sum(record.bytes for record in records if record.split == split)
        for split in split_counts
    }
    target_inventory_sha256 = hashlib.sha256(
        _canonical_json([record.target_record() for record in records])
    ).hexdigest()
    stable_receipt: dict[str, Any] = {
        "schema_version": 1,
        "complete": True,
        "repository_id": APPROVED_REPOSITORY,
        "configuration": APPROVED_CONFIGURATION,
        "revision": APPROVED_PTV2_REVISION,
        "license_expression": APPROVED_LICENSE_EXPRESSION,
        "approved_use": True,
        "source_commit": source_commit,
        "source_root_realpath": str(source_root_real),
        "approved_hao_root_realpath": str(approved_root_real),
        "manifest": {
            "path": "SOURCE_PLAN.json",
            "bytes": len(manifest_bytes),
            "sha256": manifest_sha256,
        },
        "split_counts": dict(split_counts),
        "split_bytes": split_bytes,
        "target_inventory": {"count": len(records), "sha256": target_inventory_sha256},
        "readme": {
            "path": str(readme_path.resolve(strict=True)),
            "bytes": readme.bytes,
            "sha256": readme.sha256,
            "target_device": readme.target_device,
            "target_inode": readme.target_inode,
            "target_mtime_ns": readme.target_mtime_ns,
        },
    }
    output_root = output_root.expanduser().absolute()
    if os.path.lexists(output_root):
        return _validate_existing(
            output_root, manifest_bytes=manifest_bytes, stable_receipt=stable_receipt
        )
    output_root.parent.mkdir(parents=True, exist_ok=True)
    parent_observation = _observe(output_root.parent)
    if (
        parent_observation.identity is None
        or parent_observation.mode is None
        or stat.S_ISLNK(parent_observation.mode)
        or not stat.S_ISDIR(parent_observation.mode)
    ):
        raise PTV2SourceManifestError("publication parent must be a no-follow directory")
    expected_parent_identity = parent_observation.identity
    partial = output_root.with_name(f".{output_root.name}.partial-{os.getpid()}-{uuid.uuid4().hex}")
    phase = SourceManifestPublicationPhase.PARTIAL_SETUP
    partial_identity: tuple[int, int] | None = None
    try:
        partial.mkdir(mode=0o700)
        partial_metadata = os.lstat(partial)
        if stat.S_ISLNK(partial_metadata.st_mode) or not stat.S_ISDIR(partial_metadata.st_mode):
            raise PTV2SourceManifestError("publication partial is not a no-follow directory")
        partial_identity = partial_metadata.st_dev, partial_metadata.st_ino
        receipt = {
            **stable_receipt,
            "workers": {"requested": workers, "effective": effective_workers},
            "timing": {
                "started_at_utc": started_wall.isoformat(),
                "completed_at_utc": datetime.now(UTC).isoformat(),
                "duration_seconds": round(time.monotonic() - started, 6),
            },
        }
        phase = SourceManifestPublicationPhase.WRITE
        _write_durable(partial / "SOURCE_PLAN.json", manifest_bytes)
        _write_durable(partial / "SOURCE_MANIFEST_COMPLETION.json", _canonical_json(receipt))
        phase = SourceManifestPublicationPhase.DIRECTORY_FSYNC
        _fsync_directory(partial)
        if _observe(output_root.parent).identity != expected_parent_identity:
            raise PTV2SourceManifestError("publication parent inode changed before rename")
        if _observe(partial).identity != partial_identity:
            raise PTV2SourceManifestError("publication partial inode changed before rename")
        phase = SourceManifestPublicationPhase.RENAME
        _rename_no_replace(partial, output_root)
        if _observe(output_root).identity != partial_identity:
            raise PTV2SourceManifestError("published directory inode does not match its partial")
        phase = SourceManifestPublicationPhase.PARENT_FSYNC
        _fsync_directory(output_root.parent)
    except BaseException as error:
        state = SourceManifestRecoveryState(
            phase=phase,
            partial_path=partial,
            destination_path=output_root,
            expected_parent_identity=expected_parent_identity,
            expected_partial_identity=partial_identity,
            parent_observation=_observe(output_root.parent),
            partial_observation=_observe(partial),
            destination_observation=_observe(output_root),
        )
        if isinstance(error, FileExistsError):
            message = f"concurrent publication requires recovery; retained partial: {partial}"
        else:
            message = f"source-manifest publication requires recovery; retained partial: {partial}"
        recovery_error = PTV2SourceManifestError(message)
        setattr(recovery_error, "recovery_state", state)
        raise recovery_error from error
    return _result(output_root, manifest_sha256)


def bootstrap_ptv2_source_manifest(
    *,
    source_root: Path,
    approved_hao_root: Path,
    readme_path: Path,
    output_root: Path,
    source_commit: str,
    workers: int,
) -> SourceManifestBootstrapResult:
    """Validate and publish the approved 201-shard production source bundle."""
    if sum(PRODUCTION_SPLIT_COUNTS.values()) != 201:
        raise PTV2SourceManifestError("production contract does not declare exactly 201 shards")
    try:
        actual_count = sum(
            1 for entry in os.scandir(source_root) if entry.name.endswith(".parquet")
        )
    except OSError as error:
        raise PTV2SourceManifestError("exact symlink root is unavailable") from error
    if actual_count != 201:
        raise PTV2SourceManifestError(f"expected exactly 201 source shards, found {actual_count}")
    return _bootstrap_source_manifest(
        source_root=source_root,
        approved_hao_root=approved_hao_root,
        readme_path=readme_path,
        output_root=output_root,
        source_commit=source_commit,
        workers=workers,
        split_counts=PRODUCTION_SPLIT_COUNTS,
    )


def _default_workers() -> int:
    raw = os.environ.get("SLURM_CPUS_PER_TASK", "96")
    try:
        workers = int(raw)
    except ValueError as error:
        raise PTV2SourceManifestError("SLURM_CPUS_PER_TASK must be an integer") from error
    if workers <= 0:
        raise PTV2SourceManifestError("SLURM_CPUS_PER_TASK must be positive")
    return workers


def main() -> int:
    """Run the production bootstrap from an exact 201-shard symlink root."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", required=True, type=Path)
    parser.add_argument("--approved-hao-root", required=True, type=Path)
    parser.add_argument("--readme", required=True, type=Path, dest="readme_path")
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--workers", type=int)
    arguments = parser.parse_args()
    try:
        result = bootstrap_ptv2_source_manifest(
            source_root=arguments.source_root,
            approved_hao_root=arguments.approved_hao_root,
            readme_path=arguments.readme_path,
            output_root=arguments.output_root,
            source_commit=arguments.source_commit,
            workers=arguments.workers if arguments.workers is not None else _default_workers(),
        )
    except PTV2SourceManifestError as error:
        print(str(error), file=sys.stderr)
        return 2
    print(result.manifest_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
