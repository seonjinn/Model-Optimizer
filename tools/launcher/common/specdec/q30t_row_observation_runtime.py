# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Canonical deterministic runtime evidence for Q30T row observations."""

from __future__ import annotations

import hashlib
import json
import os
import stat
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final, cast

__all__ = [
    "RowObservationRuntimeEvidence",
    "canonical_json_bytes",
    "load_row_observation_runtime_evidence",
]

_SCHEMA_VERSION: Final = "q30t-row-observation-runtime-evidence-v1"
_HASH_LENGTH: Final = 64
_COMMIT_LENGTH: Final = 40
_MAX_EVIDENCE_BYTES: Final = 1024 * 1024
_READ_BLOCK_BYTES: Final = 1024 * 1024
_RUNTIME_ROOT: Final = "/opt/q30t-runtime"
_EXPECTED_KEYS: Final = frozenset(
    {
        "approved_tool_sha256s",
        "archive_sha256",
        "archive_tree_receipt_file_sha256",
        "base_image_sha256",
        "derived_image_receipt_file_sha256",
        "derived_image_sha256",
        "profile_file_sha256",
        "pyarrow_relative_path",
        "pyarrow_tree_sha256",
        "pyarrow_version",
        "python_relative_path",
        "python_version",
        "qualification_receipt_file_sha256",
        "qualification_receipt_sha256",
        "qualification_source_commit",
        "runtime_evidence_sha256",
        "runtime_tree_sha256",
        "schema_version",
    }
)


def canonical_json_bytes(value: object) -> bytes:
    """Encode JSON-compatible evidence in the sole canonical form."""
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode(
        "utf-8"
    )


@dataclass(frozen=True)
class RowObservationRuntimeEvidence:
    """Deterministic qualified-runtime evidence embedded in one observation."""

    approved_tool_sha256s: tuple[tuple[str, str], ...]
    archive_sha256: str
    archive_tree_receipt_file_sha256: str
    base_image_sha256: str
    derived_image_receipt_file_sha256: str
    derived_image_sha256: str
    profile_file_sha256: str
    pyarrow_relative_path: str
    pyarrow_tree_sha256: str
    pyarrow_version: str
    python_relative_path: str
    python_version: str
    qualification_receipt_file_sha256: str
    qualification_receipt_sha256: str
    qualification_source_commit: str
    runtime_tree_sha256: str
    runtime_evidence_sha256: str

    def body_dict(self) -> dict[str, object]:
        """Return every evidence field protected by the self-hash."""
        return {
            "approved_tool_sha256s": dict(self.approved_tool_sha256s),
            "archive_sha256": self.archive_sha256,
            "archive_tree_receipt_file_sha256": self.archive_tree_receipt_file_sha256,
            "base_image_sha256": self.base_image_sha256,
            "derived_image_receipt_file_sha256": self.derived_image_receipt_file_sha256,
            "derived_image_sha256": self.derived_image_sha256,
            "profile_file_sha256": self.profile_file_sha256,
            "pyarrow_relative_path": self.pyarrow_relative_path,
            "pyarrow_tree_sha256": self.pyarrow_tree_sha256,
            "pyarrow_version": self.pyarrow_version,
            "python_relative_path": self.python_relative_path,
            "python_version": self.python_version,
            "qualification_receipt_file_sha256": self.qualification_receipt_file_sha256,
            "qualification_receipt_sha256": self.qualification_receipt_sha256,
            "qualification_source_commit": self.qualification_source_commit,
            "runtime_tree_sha256": self.runtime_tree_sha256,
            "schema_version": _SCHEMA_VERSION,
        }

    def to_dict(self) -> dict[str, object]:
        """Return the complete JSON-compatible runtime evidence record."""
        return self.body_dict() | {"runtime_evidence_sha256": self.runtime_evidence_sha256}

    def canonical_bytes(self) -> bytes:
        """Return canonical evidence bytes without the required publication newline."""
        return canonical_json_bytes(self.to_dict())

    def verify_self_hash(self) -> None:
        """Require the evidence body to reproduce its declared self-hash."""
        expected = hashlib.sha256(canonical_json_bytes(self.body_dict())).hexdigest()
        if self.runtime_evidence_sha256 != expected:
            raise ValueError("runtime evidence self-hash mismatch")

    @classmethod
    def from_dict(cls, raw: object) -> RowObservationRuntimeEvidence:
        """Decode one exact typed runtime-evidence record."""
        if not isinstance(raw, dict) or set(raw) != _EXPECTED_KEYS:
            raise ValueError("runtime evidence has an unexpected schema")
        tools = _required_tool_hashes(raw)
        evidence = cls(
            approved_tool_sha256s=tools,
            archive_sha256=_required_hash(raw, "archive_sha256"),
            archive_tree_receipt_file_sha256=_required_hash(
                raw, "archive_tree_receipt_file_sha256"
            ),
            base_image_sha256=_required_hash(raw, "base_image_sha256"),
            derived_image_receipt_file_sha256=_required_hash(
                raw, "derived_image_receipt_file_sha256"
            ),
            derived_image_sha256=_required_hash(raw, "derived_image_sha256"),
            profile_file_sha256=_required_hash(raw, "profile_file_sha256"),
            pyarrow_relative_path=_required_relative_path(raw, "pyarrow_relative_path"),
            pyarrow_tree_sha256=_required_hash(raw, "pyarrow_tree_sha256"),
            pyarrow_version=_required_text(raw, "pyarrow_version"),
            python_relative_path=_required_relative_path(raw, "python_relative_path"),
            python_version=_required_text(raw, "python_version"),
            qualification_receipt_file_sha256=_required_hash(
                raw, "qualification_receipt_file_sha256"
            ),
            qualification_receipt_sha256=_required_hash(raw, "qualification_receipt_sha256"),
            qualification_source_commit=_required_commit(raw, "qualification_source_commit"),
            runtime_tree_sha256=_required_hash(raw, "runtime_tree_sha256"),
            runtime_evidence_sha256=_required_hash(raw, "runtime_evidence_sha256"),
        )
        if evidence.python_relative_path != "bin/python":
            raise ValueError("runtime evidence Python path must be bin/python")
        evidence.verify_self_hash()
        return evidence


def load_row_observation_runtime_evidence(
    path: Path, *, expected_sha256: str
) -> RowObservationRuntimeEvidence:
    """Stable-read, canonical-replay, and self-hash one runtime-evidence record."""
    if not isinstance(path, Path) or not _is_canonical_absolute_path(path):
        raise ValueError("runtime evidence path must be canonical and absolute")
    if not _is_hash(expected_sha256):
        raise ValueError("runtime evidence caller SHA-256 is invalid")
    observed_sha256, raw = _stable_single_link_bytes(path)
    if observed_sha256 != expected_sha256:
        raise ValueError("runtime evidence whole-file SHA-256 mismatch")
    if not raw.endswith(b"\n") or raw.count(b"\n") != 1:
        raise ValueError("runtime evidence must have one terminal newline")
    try:
        decoded: Any = json.loads(raw[:-1], object_pairs_hook=_object_without_duplicates)
    except UnicodeDecodeError as error:
        raise ValueError("runtime evidence is not UTF-8 JSON") from error
    except json.JSONDecodeError as error:
        raise ValueError("runtime evidence JSON is invalid") from error
    evidence = RowObservationRuntimeEvidence.from_dict(decoded)
    if raw != evidence.canonical_bytes() + b"\n":
        raise ValueError("runtime evidence is not canonical")
    return evidence


def _stable_single_link_bytes(path: Path) -> tuple[str, bytes]:
    parent_fd = _open_parent(path)
    descriptor: int | None = None
    try:
        descriptor, before = _open_stable_regular(parent_fd, path.name)
        digest = hashlib.sha256()
        raw = bytearray()
        while len(raw) < before.st_size:
            block = os.read(descriptor, min(_READ_BLOCK_BYTES, before.st_size - len(raw)))
            if not block:
                raise ValueError("runtime evidence shrank while reading")
            raw.extend(block)
            digest.update(block)
        if os.read(descriptor, 1):
            raise ValueError("runtime evidence grew while reading")
        after = os.fstat(descriptor)
        rebound = os.stat(path.name, dir_fd=parent_fd, follow_symlinks=False)
        absolute = _require_absolute_path_binding(path, parent_fd, descriptor)
        if (
            _file_identity(before) != _file_identity(after)
            or _file_identity(after) != _file_identity(rebound)
            or _file_identity(after) != _file_identity(absolute)
        ):
            raise ValueError("runtime evidence changed while reading")
        return digest.hexdigest(), bytes(raw)
    finally:
        if descriptor is not None:
            os.close(descriptor)
        os.close(parent_fd)


def _open_parent(path: Path) -> int:
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        return os.open(path.parent, flags)
    except OSError as error:
        raise ValueError("runtime evidence parent is unavailable") from error


def _require_absolute_path_binding(path: Path, parent_fd: int, descriptor: int) -> os.stat_result:
    """Require the held file and parent to remain reachable through ``path``."""
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
    absolute_parent_fd = os.open("/", flags)
    try:
        for component in path.parent.parts[1:]:
            child_fd = os.open(component, flags, dir_fd=absolute_parent_fd)
            os.close(absolute_parent_fd)
            absolute_parent_fd = child_fd
        held_parent = os.fstat(parent_fd)
        absolute_parent = os.fstat(absolute_parent_fd)
        absolute_file = os.stat(path.name, dir_fd=absolute_parent_fd, follow_symlinks=False)
        held_file = os.fstat(descriptor)
        if _file_identity(held_parent) != _file_identity(absolute_parent) or _file_identity(
            held_file
        ) != _file_identity(absolute_file):
            raise ValueError("runtime evidence absolute path changed while reading")
        return absolute_file
    except ValueError:
        raise
    except OSError as error:
        raise ValueError("runtime evidence absolute path changed while reading") from error
    finally:
        os.close(absolute_parent_fd)


def _open_stable_regular(parent_fd: int, name: str) -> tuple[int, os.stat_result]:
    flags = (
        os.O_RDONLY
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_NONBLOCK", 0)
        | getattr(os, "O_CLOEXEC", 0)
    )
    try:
        named = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
        if named.st_size > _MAX_EVIDENCE_BYTES:
            raise ValueError("runtime evidence exceeds the size limit")
        descriptor = os.open(name, flags, dir_fd=parent_fd)
    except OSError as error:
        raise ValueError("runtime evidence is unavailable") from error
    opened = os.fstat(descriptor)
    if (
        not stat.S_ISREG(named.st_mode)
        or stat.S_ISLNK(named.st_mode)
        or named.st_nlink != 1
        or opened.st_nlink != 1
        or _file_identity(named) != _file_identity(opened)
    ):
        os.close(descriptor)
        raise ValueError("runtime evidence must be a stable single-link regular file")
    return descriptor, named


def _object_without_duplicates(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("runtime evidence has a duplicate JSON key")
        result[key] = value
    return result


def _required_tool_hashes(raw: dict[object, object]) -> tuple[tuple[str, str], ...]:
    value = raw.get("approved_tool_sha256s")
    if not isinstance(value, dict) or not value:
        raise ValueError("runtime evidence approved tools are invalid")
    items = tuple(sorted(value.items()))
    if any(
        type(name) is not str or not _is_relative_path(name) or not _is_hash(digest)
        for name, digest in items
    ):
        raise ValueError("runtime evidence approved tools are invalid")
    return tuple((name, digest) for name, digest in items)


def _required_hash(raw: dict[object, object], key: str) -> str:
    value = raw.get(key)
    if not _is_hash(value):
        raise ValueError(f"runtime evidence {key} is invalid")
    return cast("str", value)


def _required_commit(raw: dict[object, object], key: str) -> str:
    value = raw.get(key)
    if type(value) is not str or not _is_lower_hex(value, _COMMIT_LENGTH):
        raise ValueError(f"runtime evidence {key} is invalid")
    return cast("str", value)


def _required_relative_path(raw: dict[object, object], key: str) -> str:
    value = raw.get(key)
    if not _is_relative_path(value):
        raise ValueError(f"runtime evidence {key} must be a safe relative path")
    return cast("str", value)


def _required_text(raw: dict[object, object], key: str) -> str:
    value = raw.get(key)
    if type(value) is not str or not value or _has_control(value):
        raise ValueError(f"runtime evidence {key} is invalid")
    return value


def _is_hash(value: object) -> bool:
    return type(value) is str and _is_lower_hex(value, _HASH_LENGTH)


def _is_lower_hex(value: str, length: int) -> bool:
    return len(value) == length and all(character in "0123456789abcdef" for character in value)


def _is_relative_path(value: object) -> bool:
    if type(value) is not str or not value or _has_control(value) or "\\" in value:
        return False
    path = Path(value)
    return not path.is_absolute() and all(part not in {"", ".", ".."} for part in path.parts)


def _has_control(value: str) -> bool:
    return any(ord(character) < 32 or ord(character) == 127 for character in value)


def _is_canonical_absolute_path(path: Path) -> bool:
    return (
        path.is_absolute()
        and path != Path("/")
        and ".." not in path.parts
        and str(path) == path.as_posix()
    )


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
