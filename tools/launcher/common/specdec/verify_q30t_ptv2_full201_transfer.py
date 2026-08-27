# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Verify and receipt the complete Q30 Thinking PTV2 transfer on Ptyche."""

from __future__ import annotations

import argparse
import concurrent.futures
import ctypes
import errno
import fcntl
import hashlib
import json
import os
import platform
import signal
import stat
import sys
import threading
import uuid
from pathlib import Path
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, cast

if TYPE_CHECKING:
    from collections.abc import Mapping

__all__ = [
    "APPROVED_COMPLETION_FILE_SHA256",
    "APPROVED_OUTPUT_PATH",
    "APPROVED_RECEIPT_ANCESTOR",
    "APPROVED_SOURCE_PLAN_SHA256",
    "APPROVED_SOURCE_ROOT",
    "TransferVerificationError",
    "verify_and_publish_q30t_ptv2_full201_transfer",
]

APPROVED_SOURCE_ROOT = Path(
    "/lustre/fsw/coreai_dlalgo_llm/users/sna/modelopt-qwen3-drafter-training/"
    "assets/q30t-ptv2-full201-source-v1"
)
APPROVED_RECEIPT_ANCESTOR = Path(
    "/lustre/fsw/coreai_dlalgo_llm/users/sna/modelopt-qwen3-drafter-training/"
    "receipts/q30t-ptv23-complement-700k-v1"
)
APPROVED_OUTPUT_PATH = APPROVED_RECEIPT_ANCESTOR / "ptv2-full201-transfer/VERIFY.json"
APPROVED_SOURCE_PLAN_SHA256 = "96970541d0c6f5c99e74b9222b805d4a0bd2ac682837b0ad92b8bf54f7d71a3a"
APPROVED_COMPLETION_FILE_SHA256 = "018d659170834b17967dd6c1b066e3b03e7859386eceefd9d4409dc3bc8f48c1"
APPROVED_REPOSITORY = "nvidia/Nemotron-Post-Training-Dataset-v2"
APPROVED_CONFIGURATION = "default"
APPROVED_REVISION = "5c89e01dd720ae0f4058445ed49c5fb68a03c76e"
APPROVED_LICENSE = "CC-BY-4.0"
APPROVED_SPLIT_COUNTS: Mapping[str, int] = MappingProxyType(
    {
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
)
APPROVED_FILE_COUNT = 201
APPROVED_TOTAL_BYTES = 44_423_886_661

_PLAN_NAME = "SOURCE_PLAN.json"
_COMPLETION_NAME = "SOURCE_MANIFEST_COMPLETION.json"
_DATA_NAME = "data"
_READ_BLOCK_BYTES = 8 * 1024 * 1024
_MAX_JSON_BYTES = 64 * 1024 * 1024
_SHA256_LENGTH = 64
_F_SETLEASE = getattr(fcntl, "F_SETLEASE", None)
_F_GETLEASE = getattr(fcntl, "F_GETLEASE", None)
_F_RDLCK = getattr(fcntl, "F_RDLCK", None)
_F_UNLCK = getattr(fcntl, "F_UNLCK", None)
_ALLOW_NON_LINUX_OPENAT2_TEST_FALLBACK = False
_OPENAT2_SYSTEM_CALL = 437
_RESOLVE_NO_MAGICLINKS = 0x02
_RESOLVE_NO_SYMLINKS = 0x04
_RESOLVE_BENEATH = 0x08


class _OpenHow(ctypes.Structure):
    _fields_ = [
        ("flags", ctypes.c_uint64),
        ("mode", ctypes.c_uint64),
        ("resolve", ctypes.c_uint64),
    ]


class TransferVerificationError(RuntimeError):
    """The transferred source tree or receipt publication violates its contract."""


def _acquire_read_lease(descriptor: int, display_path: Path) -> None:
    if _F_SETLEASE is None or _F_RDLCK is None:
        raise TransferVerificationError("Linux read leases are unavailable")
    try:
        fcntl.fcntl(descriptor, _F_SETLEASE, _F_RDLCK)
    except OSError as error:
        raise TransferVerificationError(
            f"write-excluding read lease is unsupported: {display_path}"
        ) from error


def _require_read_lease(descriptor: int, display_path: Path) -> None:
    if _F_GETLEASE is None or _F_RDLCK is None:
        raise TransferVerificationError("Linux read lease inspection is unavailable")
    try:
        lease = fcntl.fcntl(descriptor, _F_GETLEASE)
    except OSError as error:
        raise TransferVerificationError(
            f"write-excluding read lease cannot be inspected: {display_path}"
        ) from error
    if lease != _F_RDLCK:
        raise TransferVerificationError(f"write-excluding read lease was broken: {display_path}")


def _release_read_lease(descriptor: int) -> None:
    if _F_SETLEASE is None or _F_UNLCK is None:
        raise TransferVerificationError("Linux read lease release is unavailable")
    fcntl.fcntl(descriptor, _F_SETLEASE, _F_UNLCK)


def _release_and_close_read_lease(descriptor: int) -> BaseException | None:
    try:
        _release_read_lease(descriptor)
    except BaseException as error:
        return error
    finally:
        os.close(descriptor)
    return None


class _ReadLeaseGuard:
    """Hold every authenticated input read lease through durable publication."""

    def __init__(self) -> None:
        self._descriptors: list[tuple[int, Path]] = []
        self._lock = threading.Lock()
        self._break_requested = threading.Event()
        self._previous_sigio_handler: Any = None

    def __enter__(self) -> _ReadLeaseGuard:
        if not hasattr(signal, "SIGIO"):
            raise TransferVerificationError("SIGIO lease-break detection is unavailable")
        self._previous_sigio_handler = signal.signal(signal.SIGIO, self._handle_break)
        return self

    def _handle_break(self, _signum: int, _frame: object) -> None:
        self._break_requested.set()

    def acquire(self, descriptor: int, display_path: Path) -> None:
        _acquire_read_lease(descriptor, display_path)
        with self._lock:
            self._descriptors.append((descriptor, display_path))

    def require_no_break(self) -> None:
        if self._break_requested.is_set():
            raise TransferVerificationError("conflicting writer requested a read lease break")
        with self._lock:
            descriptors = tuple(self._descriptors)
        for descriptor, display_path in descriptors:
            _require_read_lease(descriptor, display_path)
        if self._break_requested.is_set():
            raise TransferVerificationError("conflicting writer requested a read lease break")

    def __exit__(
        self,
        exception_type: type[BaseException] | None,
        _exception: BaseException | None,
        _traceback: object,
    ) -> None:
        release_error: BaseException | None = None
        with self._lock:
            descriptors = tuple(reversed(self._descriptors))
            self._descriptors.clear()
        for descriptor, _ in descriptors:
            error = _release_and_close_read_lease(descriptor)
            if release_error is None and error is not None:
                release_error = error
        signal.signal(signal.SIGIO, self._previous_sigio_handler)
        if exception_type is None and release_error is not None:
            raise release_error


def _canonical_json(value: object) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode()


def _identity(metadata: os.stat_result) -> tuple[int, int, int, int, int, int]:
    return (
        metadata.st_dev,
        metadata.st_ino,
        metadata.st_mode,
        metadata.st_size,
        metadata.st_mtime_ns,
        metadata.st_ctime_ns,
    )


def _object_identity(metadata: os.stat_result) -> tuple[int, int]:
    return metadata.st_dev, metadata.st_ino


def _is_lower_sha256(value: object) -> bool:
    return (
        type(value) is str
        and len(value) == _SHA256_LENGTH
        and all(character in "0123456789abcdef" for character in value)
    )


def _is_positive_int(value: object) -> bool:
    return type(value) is int and value > 0


def _open_directory(path: Path) -> tuple[int, os.stat_result]:
    try:
        expected = os.stat(path, follow_symlinks=False)
        if stat.S_ISLNK(expected.st_mode) or not stat.S_ISDIR(expected.st_mode):
            raise TransferVerificationError(f"transfer path is not a no-follow directory: {path}")
        descriptor = os.open(
            path,
            os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0),
        )
    except OSError as error:
        raise TransferVerificationError(f"transfer directory is unavailable: {path}") from error
    if _identity(expected) != _identity(os.fstat(descriptor)):
        os.close(descriptor)
        raise TransferVerificationError(f"transfer directory changed while opening: {path}")
    return descriptor, expected


def _open_directory_at(parent_fd: int, name: str, display_path: Path) -> tuple[int, os.stat_result]:
    try:
        expected = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
        if stat.S_ISLNK(expected.st_mode) or not stat.S_ISDIR(expected.st_mode):
            raise TransferVerificationError(
                f"transfer path is not a no-follow directory: {display_path}"
            )
        descriptor = os.open(
            name,
            os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0),
            dir_fd=parent_fd,
        )
    except OSError as error:
        raise TransferVerificationError(
            f"transfer directory is unavailable: {display_path}"
        ) from error
    if _identity(expected) != _identity(os.fstat(descriptor)):
        os.close(descriptor)
        raise TransferVerificationError(f"transfer directory changed while opening: {display_path}")
    return descriptor, expected


def _openat2_directory_from_root(root_fd: int, relative_path: str) -> int:
    components = relative_path.split("/")
    if not relative_path or any(
        not component or component in {".", ".."} for component in components
    ):
        raise TransferVerificationError("atomic directory path is invalid")
    if sys.platform != "linux":
        if not _ALLOW_NON_LINUX_OPENAT2_TEST_FALLBACK:
            raise TransferVerificationError("atomic openat2 resolution requires Linux")
        descriptor = os.dup(root_fd)
        current = Path("/")
        for component in components:
            try:
                child, _ = _open_directory_at(descriptor, component, current / component)
            finally:
                os.close(descriptor)
            descriptor = child
            current /= component
        return descriptor
    how = _OpenHow(
        flags=(
            os.O_RDONLY
            | getattr(os, "O_DIRECTORY", 0)
            | getattr(os, "O_NOFOLLOW", 0)
            | getattr(os, "O_CLOEXEC", 0)
        ),
        mode=0,
        resolve=_RESOLVE_BENEATH | _RESOLVE_NO_SYMLINKS | _RESOLVE_NO_MAGICLINKS,
    )
    library = ctypes.CDLL(None, use_errno=True)
    result = library.syscall(
        ctypes.c_long(_OPENAT2_SYSTEM_CALL),
        ctypes.c_int(root_fd),
        ctypes.c_char_p(os.fsencode(relative_path)),
        ctypes.byref(how),
        ctypes.c_size_t(ctypes.sizeof(how)),
    )
    if result < 0:
        error_number = ctypes.get_errno()
        raise TransferVerificationError(
            f"atomic openat2 directory resolution failed: {os.strerror(error_number)}"
        )
    return int(result)


def _open_absolute_directory(path: Path) -> tuple[int, os.stat_result]:
    if not path.is_absolute() or path == Path("/"):
        raise TransferVerificationError(f"directory path must be an absolute non-root path: {path}")
    root_fd, _ = _open_directory(Path("/"))
    try:
        descriptor = _openat2_directory_from_root(root_fd, path.as_posix().lstrip("/"))
    finally:
        os.close(root_fd)
    try:
        expected = os.fstat(descriptor)
        current = os.stat(path, follow_symlinks=False)
        if (
            not stat.S_ISDIR(expected.st_mode)
            or stat.S_ISLNK(current.st_mode)
            or not stat.S_ISDIR(current.st_mode)
            or _identity(expected) != _identity(current)
        ):
            raise TransferVerificationError(f"atomic directory path changed while opening: {path}")
    except BaseException:
        os.close(descriptor)
        raise
    return descriptor, expected


def _require_absolute_directory_binding(
    path: Path, expected: os.stat_result, *, message: str
) -> None:
    descriptor, current = _open_absolute_directory(path)
    try:
        if _identity(current) != _identity(expected) or _identity(
            os.fstat(descriptor)
        ) != _identity(expected):
            raise TransferVerificationError(message)
    finally:
        os.close(descriptor)


def _open_private_output_parent(output: Path) -> tuple[int, os.stat_result]:
    if output != APPROVED_OUTPUT_PATH:
        raise TransferVerificationError("output must equal the approved verification receipt path")
    expected_parent = APPROVED_RECEIPT_ANCESTOR / "ptv2-full201-transfer"
    if output.parent != expected_parent:
        raise TransferVerificationError("approved verification receipt parent mismatch")
    ancestor_fd, _ = _open_absolute_directory(APPROVED_RECEIPT_ANCESTOR)
    try:
        try:
            os.mkdir(output.parent.name, mode=0o700, dir_fd=ancestor_fd)
            os.fsync(ancestor_fd)
        except FileExistsError:
            pass
        parent_fd, parent_expected = _open_directory_at(
            ancestor_fd, output.parent.name, output.parent
        )
    finally:
        os.close(ancestor_fd)
    if parent_expected.st_uid != os.getuid() or stat.S_IMODE(parent_expected.st_mode) & 0o077:
        os.close(parent_fd)
        raise TransferVerificationError("verification receipt parent must be private and owned")
    return parent_fd, parent_expected


def _require_published_receipt_binding(
    *,
    output: Path,
    parent_expected: os.stat_result,
    receipt_expected: os.stat_result,
    payload: bytes,
    lease_guard: _ReadLeaseGuard,
) -> None:
    descriptor, current = _open_absolute_directory(output.parent)
    try:
        if (
            _object_identity(current) != _object_identity(parent_expected)
            or _object_identity(os.fstat(descriptor)) != _object_identity(parent_expected)
            or current.st_uid != os.getuid()
            or stat.S_IMODE(current.st_mode) & 0o077
        ):
            raise TransferVerificationError("approved verification receipt parent changed")
        size, digest, reread, rebound = _read_regular_at(
            descriptor, output.name, output, retain=True
        )
        if (
            _object_identity(rebound) != _object_identity(receipt_expected)
            or rebound.st_nlink != 1
            or size != len(payload)
            or digest != hashlib.sha256(payload).hexdigest()
            or reread != payload
        ):
            raise TransferVerificationError("published receipt changed after installation")
        try:
            os.fsync(descriptor)
        except OSError as error:
            raise TransferVerificationError(
                "published receipt parent final fsync failed"
            ) from error
        rebound_parent_fd, rebound_parent = _open_absolute_directory(output.parent)
        try:
            if (
                _object_identity(rebound_parent) != _object_identity(parent_expected)
                or _object_identity(os.fstat(rebound_parent_fd))
                != _object_identity(parent_expected)
                or rebound_parent.st_uid != os.getuid()
                or stat.S_IMODE(rebound_parent.st_mode) & 0o077
            ):
                raise TransferVerificationError("approved verification receipt parent changed")
            _require_unchanged_at(
                rebound_parent_fd,
                output.name,
                receipt_expected,
                output,
            )
            try:
                lease_guard.require_no_break()
            except TransferVerificationError as error:
                raise TransferVerificationError(
                    "late source lease break invalidated verification; "
                    "the no-clobber receipt remains installed but is not authenticated"
                ) from error
        finally:
            os.close(rebound_parent_fd)
    finally:
        os.close(descriptor)


def _rename_no_replace_at(parent_fd: int, source: str, destination: str) -> None:
    library = ctypes.CDLL(None, use_errno=True)
    source_bytes = os.fsencode(source)
    destination_bytes = os.fsencode(destination)
    system = platform.system()
    try:
        if system == "Linux":
            rename = library.renameat2
            rename.argtypes = [
                ctypes.c_int,
                ctypes.c_char_p,
                ctypes.c_int,
                ctypes.c_char_p,
                ctypes.c_uint,
            ]
            rename.restype = ctypes.c_int
            result = rename(parent_fd, source_bytes, parent_fd, destination_bytes, 1)
        elif system == "Darwin":
            rename = library.renameatx_np
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
            raise OSError(errno.ENOSYS, f"atomic no-replace rename is unsupported on {system}")
    except AttributeError as error:
        raise OSError(errno.ENOSYS, "atomic no-replace rename is unavailable") from error
    if result == 0:
        return
    error_number = ctypes.get_errno()
    if error_number == errno.EEXIST:
        raise FileExistsError(error_number, os.strerror(error_number), destination)
    raise OSError(error_number, os.strerror(error_number), destination)


def _read_regular_at(
    directory_fd: int,
    name: str,
    display_path: Path,
    *,
    retain: bool,
    max_retained_bytes: int = _MAX_JSON_BYTES,
    lease_guard: _ReadLeaseGuard | None = None,
) -> tuple[int, str, bytes | None, os.stat_result]:
    try:
        expected = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
        if not stat.S_ISREG(expected.st_mode):
            raise TransferVerificationError(f"transfer input is not a regular file: {display_path}")
        if expected.st_nlink != 1:
            raise TransferVerificationError(
                f"transfer input must have exactly one link: {display_path}"
            )
        descriptor = os.open(
            name,
            os.O_RDONLY
            | getattr(os, "O_NOFOLLOW", 0)
            | getattr(os, "O_NONBLOCK", 0)
            | getattr(os, "O_CLOEXEC", 0),
            dir_fd=directory_fd,
        )
    except OSError as error:
        raise TransferVerificationError(f"transfer input is unavailable: {display_path}") from error
    digest = hashlib.sha256()
    size = 0
    retained = bytearray() if retain else None
    leased = False
    try:
        before = os.fstat(descriptor)
        if (
            _identity(expected) != _identity(before)
            or not stat.S_ISREG(before.st_mode)
            or before.st_nlink != 1
        ):
            raise TransferVerificationError(f"transfer input changed while opening: {display_path}")
        if lease_guard is not None:
            lease_guard.acquire(descriptor, display_path)
            leased = True
        while block := os.read(descriptor, _READ_BLOCK_BYTES):
            digest.update(block)
            size += len(block)
            if retained is not None:
                retained.extend(block)
                if len(retained) > max_retained_bytes:
                    raise TransferVerificationError(
                        f"transfer metadata is too large: {display_path}"
                    )
        after = os.fstat(descriptor)
    except OSError as error:
        raise TransferVerificationError(f"transfer input cannot be read: {display_path}") from error
    finally:
        if not leased:
            os.close(descriptor)
    try:
        pathname = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
    except OSError as error:
        raise TransferVerificationError(
            f"transfer pathname changed while hashing: {display_path}"
        ) from error
    if (
        _identity(before) != _identity(after)
        or _identity(after) != _identity(pathname)
        or after.st_nlink != 1
        or pathname.st_nlink != 1
        or size != before.st_size
    ):
        raise TransferVerificationError(f"transfer input changed while hashing: {display_path}")
    return size, digest.hexdigest(), bytes(retained) if retained is not None else None, expected


def _require_unchanged_at(
    directory_fd: int, name: str, expected: os.stat_result, display_path: Path
) -> None:
    try:
        current = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
    except OSError as error:
        raise TransferVerificationError(f"transfer pathname disappeared: {display_path}") from error
    if _identity(current) != _identity(expected):
        raise TransferVerificationError(f"transfer pathname changed: {display_path}")


def _validate_plan(plan: object) -> list[dict[str, object]]:
    if not isinstance(plan, dict) or set(plan) != {"schema_version", "name", "sources"}:
        raise TransferVerificationError("source plan has invalid top-level fields")
    if plan["schema_version"] != 1 or plan["name"] != "qwen3-4b-ptv2-full-source-v1":
        raise TransferVerificationError("source plan identity mismatch")
    sources = plan["sources"]
    if not isinstance(sources, list) or len(sources) != len(APPROVED_SPLIT_COUNTS):
        raise TransferVerificationError("source plan split inventory mismatch")
    expected_source_fields = {
        "repository_id",
        "configuration",
        "split",
        "revision",
        "license_expression",
        "approved_use",
        "cell",
        "lane",
        "files",
    }
    expected_file_fields = {"path", "bytes", "sha256"}
    flattened: list[dict[str, object]] = []
    for source, (expected_split, expected_count) in zip(
        sources, APPROVED_SPLIT_COUNTS.items(), strict=True
    ):
        if not isinstance(source, dict) or set(source) != expected_source_fields:
            raise TransferVerificationError("source plan split has invalid fields")
        if source | {"files": None} != {
            "repository_id": APPROVED_REPOSITORY,
            "configuration": APPROVED_CONFIGURATION,
            "split": expected_split,
            "revision": APPROVED_REVISION,
            "license_expression": APPROVED_LICENSE,
            "approved_use": True,
            "cell": expected_split,
            "lane": "target-synth",
            "files": None,
        }:
            raise TransferVerificationError(f"source plan identity mismatch: {expected_split}")
        files = source["files"]
        if not isinstance(files, list) or len(files) != expected_count:
            raise TransferVerificationError(f"source plan shard count mismatch: {expected_split}")
        for index, record in enumerate(files):
            expected_name = f"{expected_split}-{index:05d}-of-{expected_count:05d}.parquet"
            if (
                not isinstance(record, dict)
                or set(record) != expected_file_fields
                or record["path"] != f"data/{expected_name}"
                or not _is_positive_int(record["bytes"])
                or not _is_lower_sha256(record["sha256"])
            ):
                raise TransferVerificationError(
                    f"source plan shard identity mismatch: {expected_split}/{index}"
                )
            flattened.append(
                {
                    "split": expected_split,
                    "path": record["path"],
                    "bytes": record["bytes"],
                    "sha256": record["sha256"],
                }
            )
    if len(flattened) != APPROVED_FILE_COUNT:
        raise TransferVerificationError("source plan file count mismatch")
    if sum(cast("int", record["bytes"]) for record in flattened) != APPROVED_TOTAL_BYTES:
        raise TransferVerificationError("source plan byte count mismatch")
    return flattened


def _validate_completion(
    completion: object,
    *,
    plan_bytes: int,
    split_bytes: Mapping[str, int],
) -> None:
    if not isinstance(completion, dict):
        raise TransferVerificationError("source completion is not an object")
    stable_identity = {
        "schema_version": 1,
        "complete": True,
        "repository_id": APPROVED_REPOSITORY,
        "configuration": APPROVED_CONFIGURATION,
        "revision": APPROVED_REVISION,
        "license_expression": APPROVED_LICENSE,
        "approved_use": True,
    }
    if any(completion.get(key) != value for key, value in stable_identity.items()):
        raise TransferVerificationError("source completion identity mismatch")
    if completion.get("manifest") != {
        "path": _PLAN_NAME,
        "bytes": plan_bytes,
        "sha256": APPROVED_SOURCE_PLAN_SHA256,
    }:
        raise TransferVerificationError("source completion plan binding mismatch")
    if completion.get("split_counts") != dict(APPROVED_SPLIT_COUNTS):
        raise TransferVerificationError("source completion split count mismatch")
    if completion.get("split_bytes") != dict(split_bytes):
        raise TransferVerificationError("source completion split byte mismatch")
    inventory = completion.get("target_inventory")
    if (
        not isinstance(inventory, dict)
        or set(inventory) != {"count", "sha256"}
        or inventory["count"] != APPROVED_FILE_COUNT
        or not _is_lower_sha256(inventory["sha256"])
    ):
        raise TransferVerificationError("source completion target inventory mismatch")
    if not (
        type(completion.get("source_commit")) is str
        and len(completion["source_commit"]) == 40
        and all(character in "0123456789abcdef" for character in completion["source_commit"])
    ):
        raise TransferVerificationError("source completion commit mismatch")
    for field in ("source_root_realpath", "approved_hao_root_realpath"):
        value = completion.get(field)
        if type(value) is not str or not Path(value).is_absolute():
            raise TransferVerificationError(f"source completion {field} mismatch")
    workers = completion.get("workers")
    if (
        not isinstance(workers, dict)
        or set(workers) != {"requested", "effective"}
        or not _is_positive_int(workers["requested"])
        or not _is_positive_int(workers["effective"])
        or workers["effective"] > APPROVED_FILE_COUNT
    ):
        raise TransferVerificationError("source completion worker identity mismatch")
    readme = completion.get("readme")
    if (
        not isinstance(readme, dict)
        or set(readme)
        != {
            "path",
            "bytes",
            "sha256",
            "target_device",
            "target_inode",
            "target_mtime_ns",
        }
        or type(readme["path"]) is not str
        or not Path(readme["path"]).is_absolute()
        or not _is_positive_int(readme["bytes"])
        or not _is_lower_sha256(readme["sha256"])
        or any(
            type(readme[field]) is not int
            for field in ("target_device", "target_inode", "target_mtime_ns")
        )
    ):
        raise TransferVerificationError("source completion README identity mismatch")
    timing = completion.get("timing")
    if (
        not isinstance(timing, dict)
        or set(timing) != {"started_at_utc", "completed_at_utc", "duration_seconds"}
        or any(type(timing[field]) is not str for field in ("started_at_utc", "completed_at_utc"))
        or type(timing["duration_seconds"]) not in {int, float}
        or timing["duration_seconds"] < 0
    ):
        raise TransferVerificationError("source completion timing identity mismatch")


def _hash_and_check_shard(
    data_fd: int, record: Mapping[str, object], lease_guard: _ReadLeaseGuard
) -> tuple[str, os.stat_result]:
    path = str(record["path"])
    name = Path(path).name
    size, digest, _, expected = _read_regular_at(
        data_fd,
        name,
        APPROVED_SOURCE_ROOT / path,
        retain=False,
        lease_guard=lease_guard,
    )
    if size != record["bytes"] or digest != record["sha256"]:
        raise TransferVerificationError(f"transferred shard identity mismatch: {path}")
    return name, expected


def _verify_tree(
    source_root: Path, *, workers: int, lease_guard: _ReadLeaseGuard
) -> dict[str, Any]:
    if source_root != APPROVED_SOURCE_ROOT:
        raise TransferVerificationError("source_root must equal the approved source root")
    if type(workers) is not int or workers <= 0 or workers > APPROVED_FILE_COUNT:
        raise TransferVerificationError("workers must be between one and the approved file count")
    root_fd, root_expected = _open_directory(source_root)
    data_fd: int | None = None
    try:
        if set(os.listdir(root_fd)) != {_PLAN_NAME, _COMPLETION_NAME, _DATA_NAME}:
            raise TransferVerificationError("transfer root file set mismatch")
        plan_size, plan_sha256, plan_raw, plan_expected = _read_regular_at(
            root_fd,
            _PLAN_NAME,
            source_root / _PLAN_NAME,
            retain=True,
            lease_guard=lease_guard,
        )
        completion_size, completion_sha256, completion_raw, completion_expected = _read_regular_at(
            root_fd,
            _COMPLETION_NAME,
            source_root / _COMPLETION_NAME,
            retain=True,
            lease_guard=lease_guard,
        )
        if plan_sha256 != APPROVED_SOURCE_PLAN_SHA256:
            raise TransferVerificationError("source plan SHA-256 mismatch")
        if completion_sha256 != APPROVED_COMPLETION_FILE_SHA256:
            raise TransferVerificationError("source completion SHA-256 mismatch")
        assert plan_raw is not None and completion_raw is not None
        try:
            plan = json.loads(plan_raw)
            completion = json.loads(completion_raw)
        except json.JSONDecodeError as error:
            raise TransferVerificationError("source metadata is invalid JSON") from error
        records = _validate_plan(plan)
        split_bytes = {
            split: sum(
                cast("int", record["bytes"]) for record in records if record["split"] == split
            )
            for split in APPROVED_SPLIT_COUNTS
        }
        _validate_completion(completion, plan_bytes=plan_size, split_bytes=split_bytes)
        data_fd, data_expected = _open_directory_at(root_fd, _DATA_NAME, source_root / _DATA_NAME)
        expected_names = {Path(str(record["path"])).name for record in records}
        if set(os.listdir(data_fd)) != expected_names:
            raise TransferVerificationError("transferred shard pathname set mismatch")
        with concurrent.futures.ThreadPoolExecutor(max_workers=min(workers, len(records))) as pool:
            futures = [
                pool.submit(_hash_and_check_shard, data_fd, record, lease_guard)
                for record in records
            ]
            try:
                verified_files = [future.result() for future in futures]
            except BaseException:
                for future in futures:
                    future.cancel()
                raise
        for name, expected in verified_files:
            _require_unchanged_at(data_fd, name, expected, source_root / _DATA_NAME / name)
        if set(os.listdir(data_fd)) != expected_names:
            raise TransferVerificationError("transferred shard pathname set changed")
        if _identity(os.fstat(data_fd)) != _identity(data_expected):
            raise TransferVerificationError("transferred data directory changed while hashing")
        _require_unchanged_at(root_fd, _PLAN_NAME, plan_expected, source_root / _PLAN_NAME)
        _require_unchanged_at(
            root_fd, _COMPLETION_NAME, completion_expected, source_root / _COMPLETION_NAME
        )
        if set(os.listdir(root_fd)) != {_PLAN_NAME, _COMPLETION_NAME, _DATA_NAME}:
            raise TransferVerificationError("transfer root file set changed")
        if _identity(os.fstat(root_fd)) != _identity(root_expected):
            raise TransferVerificationError("transfer root changed while hashing")
        _require_absolute_directory_binding(
            source_root, root_expected, message="approved source root changed after hashing"
        )
        lease_guard.require_no_break()
    finally:
        if data_fd is not None:
            os.close(data_fd)
        os.close(root_fd)
    return {
        "schema_version": "q30t-ptv2-full201-transfer-inventory-audit-v1",
        "authorization_scope": "point-in-time-inventory-only",
        "requires_consumer_live_revalidation": True,
        "observed_complete_at_verification": True,
        "repository_id": APPROVED_REPOSITORY,
        "revision": APPROVED_REVISION,
        "source_root": str(source_root),
        "source_plan": {"path": _PLAN_NAME, "sha256": plan_sha256},
        "source_manifest_completion": {
            "path": _COMPLETION_NAME,
            "sha256": completion_sha256,
        },
        "source_file_count": APPROVED_FILE_COUNT,
        "source_bytes": APPROVED_TOTAL_BYTES,
        "split_counts": dict(APPROVED_SPLIT_COUNTS),
        "split_bytes": split_bytes,
    }


def _publish_no_clobber(output: Path, payload: bytes, lease_guard: _ReadLeaseGuard) -> None:
    parent_fd, parent_expected = _open_private_output_parent(output)
    try:
        os.stat(output.name, dir_fd=parent_fd, follow_symlinks=False)
    except FileNotFoundError:
        pass
    except OSError as error:
        os.close(parent_fd)
        raise TransferVerificationError("verification receipt path is unavailable") from error
    else:
        os.close(parent_fd)
        raise TransferVerificationError("verification receipt already exists")
    temporary = f".{output.name}.partial-{os.getpid()}-{uuid.uuid4().hex}"
    installed = False
    created: os.stat_result | None = None
    published: os.stat_result | None = None
    try:
        descriptor = os.open(
            temporary,
            os.O_WRONLY
            | os.O_CREAT
            | os.O_EXCL
            | getattr(os, "O_NOFOLLOW", 0)
            | getattr(os, "O_CLOEXEC", 0),
            0o440,
            dir_fd=parent_fd,
        )
        try:
            offset = 0
            while offset < len(payload):
                offset += os.write(descriptor, payload[offset:])
            os.fsync(descriptor)
            created = os.fstat(descriptor)
        finally:
            os.close(descriptor)
        size, digest, reread, rebound = _read_regular_at(
            parent_fd, temporary, output.parent / temporary, retain=True
        )
        if (
            created is None
            or _identity(rebound) != _identity(created)
            or size != len(payload)
            or digest != hashlib.sha256(payload).hexdigest()
            or reread != payload
        ):
            raise TransferVerificationError(
                "verification receipt temporary identity or durable reread mismatch"
            )
        lease_guard.require_no_break()
        try:
            _rename_no_replace_at(parent_fd, temporary, output.name)
        except FileExistsError as error:
            raise TransferVerificationError("verification receipt already exists") from error
        installed = True
        output_size, output_digest, output_reread, output_expected = _read_regular_at(
            parent_fd, output.name, output, retain=True
        )
        published = output_expected
        if (
            _object_identity(output_expected) != _object_identity(created)
            or output_size != len(payload)
            or output_digest != hashlib.sha256(payload).hexdigest()
            or output_reread != payload
        ):
            raise TransferVerificationError("published verification receipt identity mismatch")
        os.fsync(parent_fd)
    except OSError as error:
        raise TransferVerificationError("verification receipt publication failed") from error
    finally:
        try:
            os.fsync(parent_fd)
        finally:
            os.close(parent_fd)
    if not installed or published is None:
        raise TransferVerificationError("verification receipt was not installed")
    _require_published_receipt_binding(
        output=output,
        parent_expected=parent_expected,
        receipt_expected=published,
        payload=payload,
        lease_guard=lease_guard,
    )


def verify_and_publish_q30t_ptv2_full201_transfer(
    *, source_root: Path, output: Path, workers: int
) -> bytes:
    """Rehash the approved transfer and atomically publish its canonical receipt."""
    with _ReadLeaseGuard() as lease_guard:
        receipt = _canonical_json(
            _verify_tree(source_root, workers=workers, lease_guard=lease_guard)
        )
        lease_guard.require_no_break()
        _publish_no_clobber(output, receipt, lease_guard)
    return receipt


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=16)
    return parser.parse_args()


def main() -> int:
    """Run the exact production transfer verification contract."""
    arguments = _parse_args()
    verify_and_publish_q30t_ptv2_full201_transfer(
        source_root=arguments.source_root,
        output=arguments.output,
        workers=arguments.workers,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
