# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Fail-closed, node-local descriptor keeper for PTV23 runtime inputs."""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import pwd
import re
import signal
import stat
import sys
from contextlib import suppress
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import TYPE_CHECKING, Literal, cast

if TYPE_CHECKING:
    from collections.abc import Callable

SCHEMA_VERSION: Literal["ptv23-node-keeper-v1"] = "ptv23-node-keeper-v1"
__all__ = [
    "KeeperError",
    "KeeperItem",
    "KeeperPlan",
    "KeeperReceipt",
    "KeeperSource",
    "load_plan",
    "load_receipt",
    "main",
    "open_tree_root",
    "serve",
    "stage_regular_to_tmpfile",
    "validate_keeper_receipt",
]
_HASH_RE = re.compile(r"[0-9a-f]{64}\Z")
_NAME_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]*\Z")
_BLOCK_SIZE = 8 * 1024 * 1024
_CLEANUP_RACE_HOOK: Callable[[], None] | None = None


class KeeperError(RuntimeError):
    """The keeper cannot prove a safe input or a live descriptor boundary."""


class _StopKeeper(BaseException):
    pass


@dataclass(frozen=True)
class KeeperSource:
    """One immutable regular file which must be retained by the keeper."""

    name: str
    source_path: Path
    expected_sha256: str


@dataclass(frozen=True)
class KeeperPlan:
    """Canonical controller input for exactly one node-local keeper."""

    schema_version: Literal["ptv23-node-keeper-v1"]
    job_id: str
    node_name: str
    scratch_root: Path
    directory_roots: tuple[Path, ...]
    anchor_root: Path
    fifo_path: Path
    sources: tuple[KeeperSource, ...]

    def __post_init__(self) -> None:
        _validate_plan(self)

    def canonical_bytes(self) -> bytes:
        """Return the exact canonical JSON bytes accepted by ``load_plan``."""
        return _canonical(self.to_dict())

    def to_dict(self) -> dict[str, object]:
        """Return the JSON-compatible canonical plan body."""
        return {
            "anchor_root": str(self.anchor_root),
            "directory_roots": [str(path) for path in self.directory_roots],
            "fifo_path": str(self.fifo_path),
            "job_id": self.job_id,
            "node_name": self.node_name,
            "schema_version": self.schema_version,
            "scratch_root": str(self.scratch_root),
            "sources": [
                {
                    "expected_sha256": source.expected_sha256,
                    "name": source.name,
                    "source_path": str(source.source_path),
                }
                for source in self.sources
            ],
        }

    @classmethod
    def from_dict(cls, raw: object) -> KeeperPlan:
        """Decode and validate a JSON-compatible canonical plan body."""
        if not isinstance(raw, dict) or set(raw) != {
            "anchor_root",
            "directory_roots",
            "fifo_path",
            "job_id",
            "node_name",
            "schema_version",
            "scratch_root",
            "sources",
        }:
            raise KeeperError("keeper plan has an unexpected schema")
        roots = raw["directory_roots"]
        sources = raw["sources"]
        if not isinstance(roots, list) or not isinstance(sources, list):
            raise KeeperError("keeper plan roots and sources must be arrays")
        if any(not isinstance(root, str) for root in roots):
            raise KeeperError("keeper plan directory roots must be text")
        try:
            parsed_sources = tuple(
                KeeperSource(
                    name=_text(item, "name"),
                    source_path=Path(_text(item, "source_path")),
                    expected_sha256=_text(item, "expected_sha256"),
                )
                for item in sources
            )
            return cls(
                schema_version=cast(
                    "Literal['ptv23-node-keeper-v1']", _text(raw, "schema_version")
                ),
                job_id=_text(raw, "job_id"),
                node_name=_text(raw, "node_name"),
                scratch_root=Path(_text(raw, "scratch_root")),
                directory_roots=tuple(Path(root) for root in roots),
                anchor_root=Path(_text(raw, "anchor_root")),
                fifo_path=Path(_text(raw, "fifo_path")),
                sources=parsed_sources,
            )
        except (TypeError, ValueError) as error:
            raise KeeperError("keeper plan contains invalid values") from error


@dataclass(frozen=True)
class KeeperItem:
    """Descriptor-backed staged input published by one keeper."""

    name: str
    source_path: Path
    expected_sha256: str
    staged_size: int
    staged_sha256: str
    anchor_path: Path
    descriptor: int

    @property
    def source_sha256(self) -> str:
        """The source digest authenticated while copying this exact descriptor."""
        return self.expected_sha256


@dataclass(frozen=True)
class KeeperReceipt:
    """Canonical readiness or failure evidence for one node-local keeper."""

    schema_version: Literal["ptv23-node-keeper-v1"]
    job_id: str
    node_name: str
    keeper_pid: int
    keeper_start_ticks: int
    items: tuple[KeeperItem, ...]
    receipt_sha256: str

    @property
    def names(self) -> tuple[str, ...]:
        """Return staged source names in their canonical order."""
        return tuple(item.name for item in self.items)

    def body_dict(self) -> dict[str, object]:
        """Return the receipt body used to calculate its digest."""
        return {
            "items": [
                {
                    "anchor_path": str(item.anchor_path),
                    "descriptor": item.descriptor,
                    "expected_sha256": item.expected_sha256,
                    "name": item.name,
                    "source_path": str(item.source_path),
                    "staged_sha256": item.staged_sha256,
                    "staged_size": item.staged_size,
                }
                for item in self.items
            ],
            "job_id": self.job_id,
            "keeper_pid": self.keeper_pid,
            "keeper_start_ticks": self.keeper_start_ticks,
            "node_name": self.node_name,
            "schema_version": self.schema_version,
        }

    def to_dict(self) -> dict[str, object]:
        """Return the JSON-compatible canonical receipt record."""
        return self.body_dict() | {"receipt_sha256": self.receipt_sha256}


@dataclass(frozen=True)
class _OwnedAnchor:
    name: str
    device: int
    inode: int
    target: str


def _canonical(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def _text(record: object, name: str) -> str:
    if not isinstance(record, dict) or not isinstance(record.get(name), str):
        raise TypeError(f"{name} must be text")
    return record[name]


def _is_hash(value: str) -> bool:
    return bool(_HASH_RE.fullmatch(value))


def _absolute(path: Path, what: str) -> None:
    if not path.is_absolute():
        raise KeeperError(f"{what} must be an absolute path")


def _nofollow_flag() -> int:
    flag = getattr(os, "O_NOFOLLOW", 0)
    if not isinstance(flag, int) or flag == 0:
        raise KeeperError("platform lacks a usable O_NOFOLLOW flag")
    return flag


def _canonical_absolute_path(path: Path, what: str) -> None:
    _absolute(path, what)
    rendered = str(path)
    if rendered != os.path.normpath(rendered) or any(
        component in {".", ".."} for component in path.parts
    ):
        raise KeeperError(f"{what} must use canonical path components")


def _expected_scratch_root(job_id: str) -> Path:
    return (
        Path("/raid/scratch") / pwd.getpwuid(os.geteuid()).pw_name / f"ptv23-node-keeper-{job_id}"
    )


def _validate_plan(plan: KeeperPlan) -> None:
    if plan.schema_version != SCHEMA_VERSION:
        raise KeeperError("unsupported keeper plan schema")
    if not plan.job_id or not plan.node_name:
        raise KeeperError("keeper identity must be non-empty")
    if not _NAME_RE.fullmatch(plan.job_id) or not _NAME_RE.fullmatch(plan.node_name):
        raise KeeperError("keeper identity is unsafe")
    if not plan.sources:
        raise KeeperError("keeper plan must stage at least one source")
    for path, what in (
        (plan.scratch_root, "scratch root"),
        (plan.anchor_root, "anchor root"),
        (plan.fifo_path, "controller FIFO"),
    ):
        _canonical_absolute_path(path, what)
    if plan.scratch_root != _expected_scratch_root(plan.job_id):
        raise KeeperError("scratch root must be the current user's node-local job namespace")
    if not plan.directory_roots:
        raise KeeperError("keeper plan must name allowed directory roots")
    if tuple(sorted(plan.directory_roots, key=str)) != plan.directory_roots:
        raise KeeperError("directory roots must be sorted")
    if len(set(plan.directory_roots)) != len(plan.directory_roots):
        raise KeeperError("directory roots must be unique")
    for root in plan.directory_roots:
        _canonical_absolute_path(root, "directory root")
    try:
        relative_anchor = plan.anchor_root.relative_to(plan.scratch_root)
    except ValueError as error:
        raise KeeperError("anchor root must be beneath scratch root") from error
    if relative_anchor == Path(".") or len(relative_anchor.parts) != 1:
        raise KeeperError("anchor root must be a direct private scratch child")
    if tuple(sorted(plan.sources, key=lambda source: source.name)) != plan.sources:
        raise KeeperError("keeper sources must be sorted by name")
    if len({source.name for source in plan.sources}) != len(plan.sources):
        raise KeeperError("keeper source names must be unique")
    for source in plan.sources:
        if not _NAME_RE.fullmatch(source.name) or source.name in {".", ".."}:
            raise KeeperError("keeper source name is unsafe")
        _canonical_absolute_path(source.source_path, "keeper source")
        if not _is_hash(source.expected_sha256):
            raise KeeperError("keeper source SHA-256 is invalid")
        if not any(_is_under(source.source_path, root) for root in plan.directory_roots):
            raise KeeperError("keeper source is outside allowed directory roots")


def _is_under(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def open_tree_root(path: Path) -> int:
    """Open an absolute directory path one no-follow component at a time."""
    _canonical_absolute_path(path, "directory root")
    flags = os.O_RDONLY | os.O_DIRECTORY | _nofollow_flag()
    descriptor = os.open("/", flags)
    try:
        for component in path.parts[1:]:
            child = os.open(component, flags, dir_fd=descriptor)
            os.close(descriptor)
            descriptor = child
        metadata = os.fstat(descriptor)
        if not stat.S_ISDIR(metadata.st_mode):
            raise KeeperError("directory root is not a directory")
        return descriptor
    except BaseException:
        os.close(descriptor)
        raise


def _open_regular(path: Path) -> int:
    _canonical_absolute_path(path, "keeper source")
    parent = open_tree_root(path.parent)
    try:
        descriptor = os.open(
            path.name,
            os.O_RDONLY | _nofollow_flag(),
            dir_fd=parent,
        )
    finally:
        os.close(parent)
    metadata = os.fstat(descriptor)
    if not stat.S_ISREG(metadata.st_mode):
        os.close(descriptor)
        raise KeeperError("keeper source must be a regular no-follow file")
    return descriptor


def _write_all(descriptor: int, block: bytes) -> None:
    remaining = memoryview(block)
    while remaining:
        written = os.write(descriptor, remaining)
        if written <= 0:
            raise KeeperError("anonymous staged source copy made no progress")
        remaining = remaining[written:]


def _descriptor_digest(descriptor: int) -> tuple[int, str]:
    os.lseek(descriptor, 0, os.SEEK_SET)
    digest = sha256()
    size = 0
    while block := os.read(descriptor, _BLOCK_SIZE):
        digest.update(block)
        size += len(block)
    os.lseek(descriptor, 0, os.SEEK_SET)
    return size, digest.hexdigest()


def stage_regular_to_tmpfile(
    source_path: Path, expected_sha256: str, scratch_root: Path
) -> tuple[int, int, str]:
    """Copy and independently re-authenticate one no-follow file into ``O_TMPFILE``."""
    _nofollow_flag()
    if not _is_hash(expected_sha256):
        raise KeeperError("expected source SHA-256 is invalid")
    tmpfile_flag = getattr(os, "O_TMPFILE", None)
    if tmpfile_flag is None:
        raise KeeperError("node-local staging requires Python O_TMPFILE support")
    source = _open_regular(source_path)
    scratch = open_tree_root(scratch_root)
    try:
        try:
            staged = os.open(".", os.O_RDWR | tmpfile_flag, 0o600, dir_fd=scratch)
        except OSError as error:
            raise KeeperError("node-local staging root does not support O_TMPFILE") from error
    finally:
        os.close(scratch)
    try:
        before = os.fstat(source)
        source_digest = sha256()
        copied_size = 0
        while block := os.read(source, _BLOCK_SIZE):
            source_digest.update(block)
            copied_size += len(block)
            _write_all(staged, block)
        after = os.fstat(source)
        if (
            (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
            != (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
            or copied_size != before.st_size
            or source_digest.hexdigest() != expected_sha256
        ):
            raise KeeperError("staged source identity mismatch")
        os.fsync(staged)
        staged_size, staged_sha256 = _descriptor_digest(staged)
        if staged_size != before.st_size or staged_sha256 != expected_sha256:
            raise KeeperError("staged source copy mismatch")
        os.set_inheritable(staged, True)
        return staged, staged_size, staged_sha256
    except BaseException:
        os.close(staged)
        raise
    finally:
        os.close(source)


def _create_private_anchor_root(plan: KeeperPlan) -> int:
    scratch = open_tree_root(plan.scratch_root)
    name = plan.anchor_root.name
    try:
        try:
            os.mkdir(name, 0o700, dir_fd=scratch)
        except FileExistsError as error:
            raise KeeperError("private anchor root must be fresh") from error
        anchor_root = os.open(
            name,
            os.O_RDONLY | os.O_DIRECTORY | _nofollow_flag(),
            dir_fd=scratch,
        )
    finally:
        os.close(scratch)
    metadata = os.fstat(anchor_root)
    if (
        not stat.S_ISDIR(metadata.st_mode)
        or stat.S_IMODE(metadata.st_mode) != 0o700
        or metadata.st_uid != os.geteuid()
    ):
        os.close(anchor_root)
        raise KeeperError("private anchor root is not an owned 0700 directory")
    return anchor_root


def _create_anchor(directory: int, *, name: str, keeper_pid: int, descriptor: int) -> _OwnedAnchor:
    target = f"/proc/{keeper_pid}/fd/{descriptor}"
    try:
        os.symlink(target, name, dir_fd=directory)
    except FileExistsError as error:
        raise KeeperError("keeper anchor name is already occupied") from error
    metadata = os.stat(name, dir_fd=directory, follow_symlinks=False)
    if not stat.S_ISLNK(metadata.st_mode) or os.readlink(name, dir_fd=directory) != target:
        raise KeeperError("keeper anchor was replaced during publication")
    return _OwnedAnchor(name=name, device=metadata.st_dev, inode=metadata.st_ino, target=target)


def _cleanup_owned_anchors(directory: int, anchors: tuple[_OwnedAnchor, ...]) -> None:
    """Abandon private anchors without unlinking mutable names.

    Descriptor closure makes every owned anchor unusable. Removing a pathname
    after checking its inode has an unavoidable replacement race, so node-local
    scratch cleanup reclaims this private job directory after the allocation.
    """
    for anchor in anchors:
        if _CLEANUP_RACE_HOOK is not None:
            _CLEANUP_RACE_HOOK()
        try:
            metadata = os.stat(anchor.name, dir_fd=directory, follow_symlinks=False)
        except FileNotFoundError:
            continue
        if (
            stat.S_ISLNK(metadata.st_mode)
            and (metadata.st_dev, metadata.st_ino) == (anchor.device, anchor.inode)
            and os.readlink(anchor.name, dir_fd=directory) == anchor.target
        ):
            os.unlink(anchor.name, dir_fd=directory)


def _acquire_keeper_lock(plan: KeeperPlan) -> int:
    """Acquire the exclusive node-local keeper lifetime lock for this identity."""
    descriptor = open_tree_root(plan.scratch_root)
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        return descriptor
    except BlockingIOError as error:
        os.close(descriptor)
        raise KeeperError("keeper already exists for this job and node") from error
    except BaseException:
        os.close(descriptor)
        raise


def _open_controller_fifo(path: Path) -> int:
    """Open and retain an exact no-follow controller FIFO descriptor."""
    _canonical_absolute_path(path, "controller FIFO")
    parent = open_tree_root(path.parent)
    try:
        descriptor = os.open(path.name, os.O_RDONLY | _nofollow_flag(), dir_fd=parent)
    finally:
        os.close(parent)
    if not stat.S_ISFIFO(os.fstat(descriptor).st_mode):
        os.close(descriptor)
        raise KeeperError("controller path must be a FIFO")
    return descriptor


def _process_start_ticks(pid: int) -> int:
    try:
        contents = Path(f"/proc/{pid}/stat").read_text()
        end = contents.rfind(")")
        fields = contents[end + 2 :].split()
        return int(fields[19])
    except (FileNotFoundError, IndexError, ValueError) as error:
        raise KeeperError("procfs keeper identity is unavailable") from error


def _make_receipt(
    plan: KeeperPlan, *, keeper_pid: int, keeper_start_ticks: int, items: tuple[KeeperItem, ...]
) -> KeeperReceipt:
    body = {
        "items": [
            {
                "anchor_path": str(item.anchor_path),
                "descriptor": item.descriptor,
                "expected_sha256": item.expected_sha256,
                "name": item.name,
                "source_path": str(item.source_path),
                "staged_sha256": item.staged_sha256,
                "staged_size": item.staged_size,
            }
            for item in items
        ],
        "job_id": plan.job_id,
        "keeper_pid": keeper_pid,
        "keeper_start_ticks": keeper_start_ticks,
        "node_name": plan.node_name,
        "schema_version": SCHEMA_VERSION,
    }
    return KeeperReceipt(
        schema_version=SCHEMA_VERSION,
        job_id=plan.job_id,
        node_name=plan.node_name,
        keeper_pid=keeper_pid,
        keeper_start_ticks=keeper_start_ticks,
        items=items,
        receipt_sha256=sha256(_canonical(body)).hexdigest(),
    )


def _write_fresh_receipt(path: Path, receipt: KeeperReceipt) -> None:
    parent = open_tree_root(path.parent)
    temporary_name = f".{path.name}.{os.getpid()}.tmp"
    try:
        descriptor = os.open(
            temporary_name,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
            0o600,
            dir_fd=parent,
        )
        try:
            payload = receipt.to_dict()
            _write_all(descriptor, _canonical(payload) + b"\n")
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        os.link(temporary_name, path.name, src_dir_fd=parent, dst_dir_fd=parent)
        os.fsync(parent)
    except BaseException:
        with suppress(FileNotFoundError):
            os.unlink(temporary_name, dir_fd=parent)
        raise
    finally:
        os.close(parent)


def _replace_owned_receipt(path: Path, owned: os.stat_result, receipt: KeeperReceipt) -> None:
    """Publish immutable failure evidence without replacing a readiness path."""
    del owned
    _write_fresh_receipt(path.with_name(f"{path.name}.failed"), receipt)


def _stable_read(path: Path) -> bytes:
    descriptor = _open_regular(path)
    try:
        before = os.fstat(descriptor)
        chunks: list[bytes] = []
        while block := os.read(descriptor, _BLOCK_SIZE):
            chunks.append(block)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    if (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns) != (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
    ):
        raise KeeperError("receipt changed while reading")
    return b"".join(chunks)


def load_plan(path: Path) -> KeeperPlan:
    """Load one canonical, no-follow keeper plan."""
    try:
        payload = _stable_read(path)
        raw = json.loads(payload)
    except (json.JSONDecodeError, UnicodeDecodeError) as error:
        raise KeeperError("keeper plan is not canonical JSON") from error
    plan = KeeperPlan.from_dict(raw)
    if payload != plan.canonical_bytes() + b"\n":
        raise KeeperError("keeper plan is not canonical")
    return plan


def load_receipt(path: Path) -> KeeperReceipt:
    """Load and authenticate one canonical keeper receipt."""
    try:
        payload = _stable_read(path)
        raw = json.loads(payload)
    except (json.JSONDecodeError, UnicodeDecodeError) as error:
        raise KeeperError("keeper receipt is not canonical JSON") from error
    if not isinstance(raw, dict) or set(raw) != {
        "items",
        "job_id",
        "keeper_pid",
        "keeper_start_ticks",
        "node_name",
        "receipt_sha256",
        "schema_version",
    }:
        raise KeeperError("keeper receipt has an unexpected schema")
    try:
        items_raw = raw["items"]
        if not isinstance(items_raw, list):
            raise TypeError("items")
        items = tuple(
            KeeperItem(
                name=_text(item, "name"),
                source_path=Path(_text(item, "source_path")),
                expected_sha256=_text(item, "expected_sha256"),
                staged_size=item["staged_size"],
                staged_sha256=_text(item, "staged_sha256"),
                anchor_path=Path(_text(item, "anchor_path")),
                descriptor=item["descriptor"],
            )
            for item in items_raw
        )
        receipt = KeeperReceipt(
            schema_version=cast("Literal['ptv23-node-keeper-v1']", _text(raw, "schema_version")),
            job_id=_text(raw, "job_id"),
            node_name=_text(raw, "node_name"),
            keeper_pid=raw["keeper_pid"],
            keeper_start_ticks=raw["keeper_start_ticks"],
            items=items,
            receipt_sha256=_text(raw, "receipt_sha256"),
        )
    except (KeyError, TypeError, ValueError) as error:
        raise KeeperError("keeper receipt contains invalid values") from error
    if (
        receipt.schema_version != SCHEMA_VERSION
        or not isinstance(receipt.keeper_pid, int)
        or receipt.keeper_pid <= 0
        or not isinstance(receipt.keeper_start_ticks, int)
        or receipt.keeper_start_ticks < 0
        or not _is_hash(receipt.receipt_sha256)
        or tuple(sorted(receipt.items, key=lambda item: item.name)) != receipt.items
        or len({item.name for item in receipt.items}) != len(receipt.items)
        or any(
            not _NAME_RE.fullmatch(item.name)
            or not _is_hash(item.expected_sha256)
            or not _is_hash(item.staged_sha256)
            or not isinstance(item.staged_size, int)
            or item.staged_size < 0
            or not isinstance(item.descriptor, int)
            or item.descriptor < 0
            for item in receipt.items
        )
    ):
        raise KeeperError("keeper receipt fails structural validation")
    if payload != _canonical(receipt.to_dict()) + b"\n":
        raise KeeperError("keeper receipt is not canonical")
    if sha256(_canonical(receipt.body_dict())).hexdigest() != receipt.receipt_sha256:
        raise KeeperError("keeper receipt SHA-256 mismatch")
    return receipt


def _assert_keeper_alive(receipt: KeeperReceipt) -> None:
    try:
        os.kill(receipt.keeper_pid, 0)
    except ProcessLookupError as error:
        raise KeeperError("keeper is not alive") from error
    except PermissionError as error:
        raise KeeperError("keeper liveness cannot be checked") from error
    if _process_start_ticks(receipt.keeper_pid) != receipt.keeper_start_ticks:
        raise KeeperError("keeper is not alive")


def validate_keeper_receipt(path: Path) -> KeeperReceipt:
    """Validate the receipt, its live keeper identity, and exact anchor bytes."""
    receipt = load_receipt(path)
    _assert_keeper_alive(receipt)
    if not receipt.items:
        raise KeeperError("keeper reported a failed receipt")
    for item in receipt.items:
        descriptor_path = Path(f"/proc/{receipt.keeper_pid}/fd/{item.descriptor}")
        try:
            descriptor = os.open(descriptor_path, os.O_RDONLY)
        except OSError as error:
            raise KeeperError("keeper live descriptor cannot be reopened") from error
        try:
            metadata = os.fstat(descriptor)
            staged_size, staged_sha256 = _descriptor_digest(descriptor)
        finally:
            os.close(descriptor)
        if (
            not stat.S_ISREG(metadata.st_mode)
            or staged_size != item.staged_size
            or staged_sha256 != item.staged_sha256
            or item.source_sha256 != item.staged_sha256
        ):
            raise KeeperError("keeper anchor digest mismatch")
    return receipt


def serve(plan: KeeperPlan, receipt_path: Path) -> None:
    """Publish descriptor anchors, then hold them until FIFO EOF or ``stop``."""
    _canonical_absolute_path(receipt_path, "receipt path")
    keeper_pid = os.getpid()
    keeper_start_ticks = _process_start_ticks(keeper_pid)
    descriptors: list[int] = []
    anchors: list[_OwnedAnchor] = []
    anchor_directory: int | None = None
    lock_descriptor: int | None = None
    published = False
    received_stop = False
    failed_by_signal = False
    receipt_metadata: os.stat_result | None = None

    def stop_handler(_signum: int, _frame: object) -> None:
        if failed_by_signal:
            return
        raise _StopKeeper()

    previous_handlers = {
        signum: signal.signal(signum, stop_handler) for signum in (signal.SIGTERM, signal.SIGINT)
    }
    try:
        lock_descriptor = _acquire_keeper_lock(plan)
        anchor_directory = _create_private_anchor_root(plan)
        items: list[KeeperItem] = []
        for source in plan.sources:
            descriptor, staged_size, staged_sha256 = stage_regular_to_tmpfile(
                source.source_path, source.expected_sha256, plan.scratch_root
            )
            try:
                anchor = _create_anchor(
                    anchor_directory,
                    name=source.name,
                    keeper_pid=keeper_pid,
                    descriptor=descriptor,
                )
            except BaseException:
                os.close(descriptor)
                raise
            descriptors.append(descriptor)
            anchors.append(anchor)
            items.append(
                KeeperItem(
                    name=source.name,
                    source_path=source.source_path,
                    expected_sha256=source.expected_sha256,
                    staged_size=staged_size,
                    staged_sha256=staged_sha256,
                    anchor_path=plan.anchor_root / source.name,
                    descriptor=descriptor,
                )
            )
        receipt = _make_receipt(
            plan,
            keeper_pid=keeper_pid,
            keeper_start_ticks=keeper_start_ticks,
            items=tuple(items),
        )
        _write_fresh_receipt(receipt_path, receipt)
        receipt_metadata = os.lstat(receipt_path)
        published = True
        fifo = _open_controller_fifo(plan.fifo_path)
        try:
            while block := os.read(fifo, 4096):
                if b"stop" in block.splitlines():
                    received_stop = True
                    break
        finally:
            os.close(fifo)
    except _StopKeeper:
        failed_by_signal = True
    finally:
        if not failed_by_signal:
            for signum, handler in previous_handlers.items():
                signal.signal(signum, handler)
        if anchor_directory is not None:
            _cleanup_owned_anchors(anchor_directory, tuple(anchors))
            os.close(anchor_directory)
        for descriptor in descriptors:
            os.close(descriptor)
        if lock_descriptor is not None:
            os.close(lock_descriptor)
    if failed_by_signal:
        failure_receipt = _make_receipt(
            plan,
            keeper_pid=keeper_pid,
            keeper_start_ticks=keeper_start_ticks,
            items=(),
        )
        if published:
            if receipt_metadata is None:
                raise KeeperError("keeper failure receipt has no owned readiness receipt")
            _replace_owned_receipt(receipt_path, receipt_metadata, failure_receipt)
        else:
            _write_fresh_receipt(
                receipt_path.with_name(f"{receipt_path.name}.failed"), failure_receipt
            )
        for signum, handler in previous_handlers.items():
            signal.signal(signum, handler)
        return
    if not published:
        raise KeeperError("keeper failed before publishing a receipt")
    if received_stop:
        return


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    serve_parser = commands.add_parser("serve")
    serve_parser.add_argument("--plan", required=True, type=Path)
    serve_parser.add_argument("--receipt", required=True, type=Path)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Run the keeper command-line interface."""
    args = _parse_args(argv)
    try:
        if args.command == "serve":
            serve(load_plan(args.plan), args.receipt)
            return 0
    except KeeperError as error:
        print(f"ptv23 node keeper: {error}", file=sys.stderr)
        return 2
    raise AssertionError("unreachable command")


if __name__ == "__main__":
    raise SystemExit(main())
