# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Authenticate staged Q30 PTV2/PTV3 inputs and observe their row schemas."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import stat
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

if TYPE_CHECKING:
    from collections.abc import Iterator, Mapping

__all__ = ["ObservationError", "StageInputs", "observe_stage_schemas"]

_SCHEMA_VERSION = "q30t-ptv23-row-schema-observation-v1"
_SHA256_LENGTH = 64
_READ_BLOCK_BYTES = 8 * 1024 * 1024
_MAX_METADATA_BYTES = 64 * 1024 * 1024
_PTV2_PLAN_NAME = "SOURCE_PLAN.json"
_PTV2_COMPLETION_NAME = "SOURCE_MANIFEST_COMPLETION.json"
_PTV3_MANIFEST_NAME = "MANIFEST.json"
_PTV3_COMPLETION_NAME = "completion.json"


class ObservationError(RuntimeError):
    """An authenticated stage input or observed row violates the contract."""


@dataclass(frozen=True)
class StageInputs:
    """Exact PTV2/PTV3 metadata and staged-root bindings for one observation."""

    ptv2_plan_path: Path
    ptv2_plan_sha256: str
    ptv2_completion_path: Path
    ptv2_completion_sha256: str
    ptv2_root: Path
    ptv3_plan_path: Path
    ptv3_plan_sha256: str
    ptv3_completion_path: Path
    ptv3_completion_sha256: str
    ptv3_root: Path


@dataclass(frozen=True)
class _FileRecord:
    source: str
    source_id: str
    revision: str
    split: str
    logical_path: str
    staged_path: str
    physical_bytes: int
    physical_sha256: str
    expected_row_count: int | None
    file_number: int


@dataclass
class _AuthenticatedStage:
    ptv2_root_fd: int
    ptv2_root_identity: tuple[int, ...]
    ptv2_data_fd: int
    ptv2_data_identity: tuple[int, ...]
    ptv3_root_fd: int
    ptv3_root_identity: tuple[int, ...]
    records: tuple[_FileRecord, ...]
    ptv2_count: int
    ptv3_count: int

    def close(self) -> None:
        os.close(self.ptv3_root_fd)
        os.close(self.ptv2_data_fd)
        os.close(self.ptv2_root_fd)


def _canonical_json(value: object, *, newline: bool = True) -> bytes:
    suffix = "\n" if newline else ""
    return (json.dumps(value, sort_keys=True, separators=(",", ":")) + suffix).encode()


def _is_sha256(value: object) -> bool:
    return (
        type(value) is str
        and len(value) == _SHA256_LENGTH
        and all(character in "0123456789abcdef" for character in value)
    )


def _is_positive_int(value: object) -> bool:
    return type(value) is int and value > 0


def _identity(metadata: os.stat_result) -> tuple[int, ...]:
    return (
        metadata.st_dev,
        metadata.st_ino,
        metadata.st_mode,
        metadata.st_nlink,
        metadata.st_size,
        metadata.st_mtime_ns,
        metadata.st_ctime_ns,
    )


def _require_absolute_paths(inputs: StageInputs) -> None:
    paths = (
        inputs.ptv2_plan_path,
        inputs.ptv2_completion_path,
        inputs.ptv2_root,
        inputs.ptv3_plan_path,
        inputs.ptv3_completion_path,
        inputs.ptv3_root,
    )
    if any(not isinstance(path, Path) or not path.is_absolute() for path in paths):
        raise ObservationError("stage input paths must be absolute")
    if not all(
        _is_sha256(value)
        for value in (
            inputs.ptv2_plan_sha256,
            inputs.ptv2_completion_sha256,
            inputs.ptv3_plan_sha256,
            inputs.ptv3_completion_sha256,
        )
    ):
        raise ObservationError("stage input SHA-256 bindings are invalid")
    if inputs.ptv2_root == inputs.ptv3_root:
        raise ObservationError("PTV2 and PTV3 staged roots must be distinct")
    if inputs.ptv2_plan_path != inputs.ptv2_root / _PTV2_PLAN_NAME:
        raise ObservationError("PTV2 plan path is not bound to the staged root")
    if inputs.ptv2_completion_path != inputs.ptv2_root / _PTV2_COMPLETION_NAME:
        raise ObservationError("PTV2 completion path is not bound to the staged root")
    if inputs.ptv3_completion_path != inputs.ptv3_root / _PTV3_COMPLETION_NAME:
        raise ObservationError("PTV3 completion path is not bound to the staged root")


def _open_root(path: Path, label: str) -> tuple[int, tuple[int, ...]]:
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        pathname = os.stat(path, follow_symlinks=False)
        descriptor = os.open(path, flags)
    except OSError as error:
        raise ObservationError(f"{label} staged root is unavailable") from error
    observed = os.fstat(descriptor)
    if (
        not stat.S_ISDIR(pathname.st_mode)
        or stat.S_ISLNK(pathname.st_mode)
        or _identity(pathname) != _identity(observed)
    ):
        os.close(descriptor)
        raise ObservationError(f"{label} staged root is not a stable no-follow directory")
    return descriptor, _identity(observed)


def _open_directory_at(parent_fd: int, name: str, label: str) -> tuple[int, tuple[int, ...]]:
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        pathname = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
        descriptor = os.open(name, flags, dir_fd=parent_fd)
    except OSError as error:
        raise ObservationError(f"{label} directory is unavailable") from error
    observed = os.fstat(descriptor)
    if (
        not stat.S_ISDIR(pathname.st_mode)
        or stat.S_ISLNK(pathname.st_mode)
        or _identity(pathname) != _identity(observed)
    ):
        os.close(descriptor)
        raise ObservationError(f"{label} is not a stable no-follow directory")
    return descriptor, _identity(observed)


def _read_descriptor(descriptor: int, *, retain: bool) -> tuple[int, str, bytes | None]:
    os.lseek(descriptor, 0, os.SEEK_SET)
    digest = hashlib.sha256()
    size = 0
    value = bytearray() if retain else None
    while True:
        chunk = os.read(descriptor, _READ_BLOCK_BYTES)
        if not chunk:
            break
        size += len(chunk)
        digest.update(chunk)
        if value is not None:
            if size > _MAX_METADATA_BYTES:
                raise ObservationError("authenticated metadata exceeds the size limit")
            value.extend(chunk)
    return size, digest.hexdigest(), bytes(value) if value is not None else None


def _open_regular_at(parent_fd: int, name: str, label: str) -> tuple[int, os.stat_result]:
    flags = (
        os.O_RDONLY
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_NONBLOCK", 0)
        | getattr(os, "O_CLOEXEC", 0)
    )
    try:
        pathname = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
        descriptor = os.open(name, flags, dir_fd=parent_fd)
    except OSError as error:
        raise ObservationError(f"{label} is unavailable") from error
    observed = os.fstat(descriptor)
    if (
        not stat.S_ISREG(pathname.st_mode)
        or pathname.st_nlink != 1
        or observed.st_nlink != 1
        or _identity(pathname) != _identity(observed)
    ):
        os.close(descriptor)
        raise ObservationError(f"{label} must be a stable single-link regular file")
    return descriptor, observed


def _read_bound_path(path: Path, expected_sha256: str, label: str) -> bytes:
    parent_fd, _ = _open_root(path.parent, f"{label} parent")
    try:
        descriptor, before = _open_regular_at(parent_fd, path.name, label)
        try:
            size, digest, raw = _read_descriptor(descriptor, retain=True)
            after = os.fstat(descriptor)
            pathname = os.stat(path.name, dir_fd=parent_fd, follow_symlinks=False)
        finally:
            os.close(descriptor)
    finally:
        os.close(parent_fd)
    if (
        digest != expected_sha256
        or size != before.st_size
        or _identity(before) != _identity(after)
        or _identity(after) != _identity(pathname)
    ):
        raise ObservationError(f"{label} SHA-256 or stable identity mismatch")
    assert raw is not None
    return raw


def _read_bound_at(parent_fd: int, name: str, expected_sha256: str, label: str) -> bytes:
    descriptor, before = _open_regular_at(parent_fd, name, label)
    try:
        size, digest, raw = _read_descriptor(descriptor, retain=True)
        after = os.fstat(descriptor)
        pathname = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
    finally:
        os.close(descriptor)
    if (
        digest != expected_sha256
        or size != before.st_size
        or _identity(before) != _identity(after)
        or _identity(after) != _identity(pathname)
    ):
        raise ObservationError(f"{label} SHA-256 or stable identity mismatch")
    assert raw is not None
    return raw


def _load_object(raw: bytes, label: str) -> dict[str, Any]:
    try:
        value = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ObservationError(f"{label} is invalid JSON") from error
    if not isinstance(value, dict):
        raise ObservationError(f"{label} is not a JSON object")
    return value


def _valid_relative_path(value: object) -> bool:
    if type(value) is not str or not value or "\\" in value:
        return False
    components = value.split("/")
    return not value.startswith("/") and all(
        component not in {"", ".", ".."} for component in components
    )


def _parse_ptv2_plan(plan: Mapping[str, Any]) -> list[dict[str, Any]]:
    if set(plan) != {"schema_version", "name", "sources"} or plan.get("schema_version") != 1:
        raise ObservationError("PTV2 plan fields are invalid")
    if type(plan.get("name")) is not str or not plan["name"]:
        raise ObservationError("PTV2 plan identity is invalid")
    sources = plan.get("sources")
    if not isinstance(sources, list) or not sources:
        raise ObservationError("PTV2 plan has no source records")
    source_fields = {
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
    records: list[dict[str, Any]] = []
    paths: set[str] = set()
    for source in sources:
        if not isinstance(source, dict) or set(source) != source_fields:
            raise ObservationError("PTV2 source record fields are invalid")
        if not all(
            type(source.get(field)) is str and source[field]
            for field in ("repository_id", "configuration", "split", "revision")
        ):
            raise ObservationError("PTV2 source identity is invalid")
        files = source.get("files")
        if not isinstance(files, list) or not files:
            raise ObservationError("PTV2 source file inventory is empty")
        for record in files:
            if not isinstance(record, dict) or set(record) != {"path", "bytes", "sha256"}:
                raise ObservationError("PTV2 plan file fields are invalid")
            path = record.get("path")
            if (
                not _valid_relative_path(path)
                or str(path).split("/") != ["data", Path(str(path)).name]
                or not str(path).endswith(".parquet")
            ):
                raise ObservationError("PTV2 plan path is invalid")
            if path in paths:
                raise ObservationError("duplicate PTV2 plan path")
            if not _is_positive_int(record.get("bytes")) or not _is_sha256(record.get("sha256")):
                raise ObservationError("PTV2 plan file identity is invalid")
            paths.add(str(path))
            records.append(
                {
                    "source_id": source["repository_id"],
                    "revision": source["revision"],
                    "split": source["split"],
                    "logical_path": path,
                    "staged_path": path,
                    "physical_bytes": record["bytes"],
                    "physical_sha256": record["sha256"],
                    "expected_row_count": None,
                }
            )
    return records


def _validate_ptv2_completion(
    completion: Mapping[str, Any],
    *,
    plan: Mapping[str, Any],
    plan_bytes: int,
    plan_sha256: str,
    records: list[dict[str, Any]],
) -> None:
    if completion.get("schema_version") != 1 or completion.get("complete") is not True:
        raise ObservationError("PTV2 completion identity is invalid")
    manifest = completion.get("manifest")
    if manifest != {
        "path": _PTV2_PLAN_NAME,
        "bytes": plan_bytes,
        "sha256": plan_sha256,
    }:
        raise ObservationError("PTV2 completion plan binding mismatch")
    sources = plan["sources"]
    source = sources[0]
    for field in ("repository_id", "configuration", "revision"):
        if (
            any(item[field] != source[field] for item in sources)
            or completion.get(field) != source[field]
        ):
            raise ObservationError("PTV2 plan/completion source identity mismatch")
    split_counts = {
        item["split"]: sum(
            len(candidate["files"]) for candidate in sources if candidate["split"] == item["split"]
        )
        for item in sources
    }
    split_bytes = {
        split: sum(record["physical_bytes"] for record in records if record["split"] == split)
        for split in split_counts
    }
    if (
        completion.get("split_counts") != split_counts
        or completion.get("split_bytes") != split_bytes
    ):
        raise ObservationError("PTV2 completion inventory totals mismatch")
    inventory = completion.get("target_inventory")
    if (
        not isinstance(inventory, dict)
        or set(inventory) != {"count", "sha256"}
        or inventory.get("count") != len(records)
        or not _is_sha256(inventory.get("sha256"))
    ):
        raise ObservationError("PTV2 completion target inventory mismatch")


def _parse_ptv3_plan(plan: Mapping[str, Any]) -> list[dict[str, Any]]:
    if set(plan) != {"schema_version", "name", "container", "files"}:
        raise ObservationError("PTV3 plan fields are invalid")
    if plan.get("schema_version") != 1 or type(plan.get("name")) is not str:
        raise ObservationError("PTV3 plan identity is invalid")
    files = plan.get("files")
    if not isinstance(files, list) or not files:
        raise ObservationError("PTV3 plan has no file records")
    required = {
        "source_id",
        "revision",
        "license",
        "pool",
        "category",
        "response_source",
        "tool_lane",
        "path",
        "bytes",
        "sha256",
    }
    records: list[dict[str, Any]] = []
    identities: set[tuple[str, str, str]] = set()
    for record in files:
        if not isinstance(record, dict) or set(record) != required:
            raise ObservationError("PTV3 plan file fields are invalid")
        path = record.get("path")
        if not _valid_relative_path(path) or Path(str(path)).suffix not in {".jsonl", ".parquet"}:
            raise ObservationError("PTV3 plan path is invalid")
        identity = (str(record.get("source_id")), str(record.get("revision")), str(path))
        if identity in identities:
            raise ObservationError("duplicate PTV3 plan source file")
        if (
            not all(
                type(record.get(field)) is str and record[field]
                for field in ("source_id", "revision")
            )
            or not _is_positive_int(record.get("bytes"))
            or not _is_sha256(record.get("sha256"))
        ):
            raise ObservationError("PTV3 plan source identity is invalid")
        identities.add(identity)
        records.append(record)
    return records


def _logical_split(path: str) -> str:
    name = Path(path).name
    if "-" in name:
        return name.split("-", maxsplit=1)[0]
    return Path(name).stem


def _reconcile_ptv3_manifest(
    manifest: Mapping[str, Any],
    *,
    plan: Mapping[str, Any],
    plan_sha256: str,
    plan_records: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    files = manifest.get("files")
    if (
        manifest.get("schema_version") != 1
        or manifest.get("name") != plan.get("name")
        or manifest.get("plan_sha256") != plan_sha256
        or manifest.get("container") != plan.get("container")
        or manifest.get("format") != "parquet"
        or not isinstance(files, list)
        or manifest.get("file_count") != len(files)
    ):
        raise ObservationError("PTV3 manifest identity mismatch")
    manifest_paths: set[str] = set()
    manifest_by_source: dict[tuple[str, str, str], dict[str, Any]] = {}
    expected_fields = {
        "path",
        "bytes",
        "sha256",
        "row_count",
        "source_id",
        "source_revision",
        "source_path",
        "source_bytes",
        "source_sha256",
        "license",
        "pool",
        "category",
        "response_source",
        "tool_lane",
    }
    for record in files:
        if not isinstance(record, dict) or set(record) != expected_fields:
            raise ObservationError("PTV3 manifest file fields are invalid")
        path = record.get("path")
        if not _valid_relative_path(path) or len(str(path).split("/")) != 1:
            raise ObservationError("PTV3 manifest path is invalid")
        if path in manifest_paths:
            raise ObservationError("duplicate PTV3 manifest path")
        identity = (
            str(record.get("source_id")),
            str(record.get("source_revision")),
            str(record.get("source_path")),
        )
        if identity in manifest_by_source:
            raise ObservationError("duplicate PTV3 manifest source record")
        if (
            not _is_positive_int(record.get("bytes"))
            or not _is_sha256(record.get("sha256"))
            or not _is_positive_int(record.get("row_count"))
        ):
            raise ObservationError("PTV3 manifest physical identity is invalid")
        manifest_paths.add(str(path))
        manifest_by_source[identity] = record
    expected_identities = {
        (record["source_id"], record["revision"], record["path"]): record for record in plan_records
    }
    if set(manifest_by_source) != set(expected_identities):
        raise ObservationError("PTV3 plan/manifest inventory mismatch")
    reconciled: list[dict[str, Any]] = []
    source_pairs = (
        ("source_bytes", "bytes"),
        ("source_sha256", "sha256"),
        ("license", "license"),
        ("pool", "pool"),
        ("category", "category"),
        ("response_source", "response_source"),
        ("tool_lane", "tool_lane"),
    )
    for source in plan_records:
        record = manifest_by_source[(source["source_id"], source["revision"], source["path"])]
        if any(record[manifest_key] != source[plan_key] for manifest_key, plan_key in source_pairs):
            raise ObservationError("PTV3 plan/manifest source binding mismatch")
        reconciled.append(
            {
                "source_id": source["source_id"],
                "revision": source["revision"],
                "split": _logical_split(source["path"]),
                "logical_path": source["path"],
                "staged_path": record["path"],
                "physical_bytes": record["bytes"],
                "physical_sha256": record["sha256"],
                "expected_row_count": record["row_count"],
            }
        )
    if manifest.get("row_count") != sum(record["expected_row_count"] for record in reconciled):
        raise ObservationError("PTV3 manifest row total mismatch")
    return reconciled


def _authenticate_stage(inputs: StageInputs) -> _AuthenticatedStage:
    _require_absolute_paths(inputs)
    ptv2_plan_raw = _read_bound_path(inputs.ptv2_plan_path, inputs.ptv2_plan_sha256, "PTV2 plan")
    ptv2_completion_raw = _read_bound_path(
        inputs.ptv2_completion_path, inputs.ptv2_completion_sha256, "PTV2 completion"
    )
    ptv3_plan_raw = _read_bound_path(inputs.ptv3_plan_path, inputs.ptv3_plan_sha256, "PTV3 plan")
    ptv3_completion_raw = _read_bound_path(
        inputs.ptv3_completion_path, inputs.ptv3_completion_sha256, "PTV3 completion"
    )
    ptv2_plan = _load_object(ptv2_plan_raw, "PTV2 plan")
    ptv2_completion = _load_object(ptv2_completion_raw, "PTV2 completion")
    ptv3_plan = _load_object(ptv3_plan_raw, "PTV3 plan")
    ptv3_completion = _load_object(ptv3_completion_raw, "PTV3 completion")
    ptv2_records = _parse_ptv2_plan(ptv2_plan)
    _validate_ptv2_completion(
        ptv2_completion,
        plan=ptv2_plan,
        plan_bytes=len(ptv2_plan_raw),
        plan_sha256=inputs.ptv2_plan_sha256,
        records=ptv2_records,
    )
    ptv3_plan_records = _parse_ptv3_plan(ptv3_plan)
    if ptv3_completion != {
        "schema_version": 1,
        "complete": True,
        "plan_sha256": inputs.ptv3_plan_sha256,
        "manifest_sha256": ptv3_completion.get("manifest_sha256"),
    } or not _is_sha256(ptv3_completion.get("manifest_sha256")):
        raise ObservationError("PTV3 completion plan binding mismatch")

    ptv2_root_fd, ptv2_root_identity = _open_root(inputs.ptv2_root, "PTV2")
    ptv2_data_fd: int | None = None
    ptv3_root_fd: int | None = None
    try:
        if set(os.listdir(ptv2_root_fd)) != {
            _PTV2_PLAN_NAME,
            _PTV2_COMPLETION_NAME,
            "data",
        }:
            raise ObservationError("PTV2 staged pathname set mismatch")
        ptv2_data_fd, ptv2_data_identity = _open_directory_at(ptv2_root_fd, "data", "PTV2 data")
        expected_ptv2_names = {Path(record["staged_path"]).name for record in ptv2_records}
        if set(os.listdir(ptv2_data_fd)) != expected_ptv2_names:
            raise ObservationError("PTV2 staged pathname set mismatch")

        ptv3_root_fd, ptv3_root_identity = _open_root(inputs.ptv3_root, "PTV3")
        manifest_raw = _read_bound_at(
            ptv3_root_fd,
            _PTV3_MANIFEST_NAME,
            str(ptv3_completion["manifest_sha256"]),
            "PTV3 manifest",
        )
        ptv3_manifest = _load_object(manifest_raw, "PTV3 manifest")
        ptv3_records = _reconcile_ptv3_manifest(
            ptv3_manifest,
            plan=ptv3_plan,
            plan_sha256=inputs.ptv3_plan_sha256,
            plan_records=ptv3_plan_records,
        )
        expected_ptv3_names = {
            _PTV3_MANIFEST_NAME,
            _PTV3_COMPLETION_NAME,
            *(record["staged_path"] for record in ptv3_records),
        }
        if set(os.listdir(ptv3_root_fd)) != expected_ptv3_names:
            raise ObservationError("PTV3 staged pathname set mismatch")

        numbered: list[_FileRecord] = []
        for number, record in enumerate([*ptv2_records, *ptv3_records], start=1):
            numbered.append(
                _FileRecord(
                    source="ptv2" if number <= len(ptv2_records) else "ptv3",
                    file_number=number,
                    **record,
                )
            )
        return _AuthenticatedStage(
            ptv2_root_fd=ptv2_root_fd,
            ptv2_root_identity=ptv2_root_identity,
            ptv2_data_fd=ptv2_data_fd,
            ptv2_data_identity=ptv2_data_identity,
            ptv3_root_fd=ptv3_root_fd,
            ptv3_root_identity=ptv3_root_identity,
            records=tuple(numbered),
            ptv2_count=len(ptv2_records),
            ptv3_count=len(ptv3_records),
        )
    except BaseException:
        if ptv3_root_fd is not None:
            os.close(ptv3_root_fd)
        if ptv2_data_fd is not None:
            os.close(ptv2_data_fd)
        os.close(ptv2_root_fd)
        raise


def _parquet_schema(descriptor: int) -> tuple[str, list[str]]:
    try:
        import pyarrow.parquet as pq  # pyright: ignore[reportMissingImports]
    except ImportError as error:
        raise ObservationError("PyArrow is required to observe Parquet inputs") from error
    os.lseek(descriptor, 0, os.SEEK_SET)
    with os.fdopen(os.dup(descriptor), "rb") as stream:
        parquet = pq.ParquetFile(stream)
        schema = parquet.schema_arrow
        return str(schema), list(schema.names)


def _iter_parquet_rows(descriptor: int) -> Iterator[dict[str, object]]:
    try:
        import pyarrow.parquet as pq  # pyright: ignore[reportMissingImports]
    except ImportError as error:
        raise ObservationError("PyArrow is required to observe Parquet inputs") from error
    os.lseek(descriptor, 0, os.SEEK_SET)
    with os.fdopen(os.dup(descriptor), "rb") as stream:
        parquet = pq.ParquetFile(stream)
        for batch in parquet.iter_batches(batch_size=8192, use_threads=False):
            for row in batch.to_pylist():
                if not isinstance(row, dict):
                    raise ObservationError("Parquet row is not an object")
                yield row


def _iter_jsonl_rows(descriptor: int) -> Iterator[dict[str, object]]:
    os.lseek(descriptor, 0, os.SEEK_SET)
    with os.fdopen(os.dup(descriptor), "r", encoding="utf-8") as stream:
        for line in stream:
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise ObservationError("JSONL row is invalid JSON") from error
            if not isinstance(row, dict):
                raise ObservationError("JSONL row is not an object")
            yield row


def _decode_row(row: dict[str, object]) -> dict[str, object]:
    if set(row) != {"raw_json"}:
        return row
    raw = row["raw_json"]
    if not isinstance(raw, str):
        raise ObservationError("raw_json row value is not text")
    try:
        decoded = json.loads(raw)
    except json.JSONDecodeError as error:
        raise ObservationError("raw_json row value is invalid JSON") from error
    if not isinstance(decoded, dict):
        raise ObservationError("raw_json row value is not an object")
    return decoded


def _decode_json_value(value: object, label: str) -> object:
    if not isinstance(value, str):
        return value
    try:
        return json.loads(value)
    except json.JSONDecodeError as error:
        raise ObservationError(f"{label} value is invalid JSON text") from error


def _shape(value: object) -> object:
    if value is None:
        return {"type": "null"}
    if type(value) is bool:
        return {"type": "boolean"}
    if type(value) is int:
        return {"type": "integer"}
    if type(value) is float:
        return {"type": "number"}
    if isinstance(value, str):
        return {"type": "string"}
    if isinstance(value, list):
        unique = {_canonical_json(_shape(item), newline=False) for item in value}
        return {
            "type": "array",
            "empty": not value,
            "item_shapes": [json.loads(item) for item in sorted(unique)],
        }
    if isinstance(value, dict):
        return {
            "type": "object",
            "fields": {key: _shape(value[key]) for key in sorted(value)},
        }
    raise ObservationError(f"unsupported JSON value type: {type(value).__name__}")


def _count_shapes(shapes: list[object]) -> list[dict[str, object]]:
    counts: dict[bytes, int] = {}
    for shape in shapes:
        key = _canonical_json(shape, newline=False)
        counts[key] = counts.get(key, 0) + 1
    return [{"shape": json.loads(key), "row_count": count} for key, count in sorted(counts.items())]


def _observe_rows(
    descriptor: int, record: _FileRecord
) -> tuple[int, str | None, list[str], dict[str, object]]:
    suffix = Path(record.staged_path).suffix
    if suffix == ".parquet":
        arrow_schema, physical_fields = _parquet_schema(descriptor)
        rows = _iter_parquet_rows(descriptor)
        physical_format = "parquet"
    elif suffix == ".jsonl":
        arrow_schema = None
        physical_fields = []
        rows = _iter_jsonl_rows(descriptor)
        physical_format = "jsonl"
    else:
        raise ObservationError(f"file {record.file_number} has an unsupported physical format")
    row_count = 0
    field_sets: list[object] = []
    messages_shapes: list[object] = []
    tools_shapes: list[object] = []
    row_shapes: list[object] = []
    tools_presence: set[bool] = set()
    try:
        iterator = iter(rows)
        while True:
            row_count += 1
            try:
                physical_row = next(iterator)
            except StopIteration:
                row_count -= 1
                break
            row = _decode_row(physical_row)
            fields = sorted(row)
            field_sets.append(fields)
            if "messages" not in row:
                raise ObservationError("messages field is absent")
            messages = _decode_json_value(row["messages"], "messages")
            if (
                not isinstance(messages, list)
                or not messages
                or not all(isinstance(message, dict) for message in messages)
            ):
                raise ObservationError("messages value is not a nonempty array of objects")
            tools_present = "tools" in row
            tools_presence.add(tools_present)
            tools = _decode_json_value(row.get("tools"), "tools") if tools_present else None
            if tools is not None and (
                not isinstance(tools, list) or not all(isinstance(tool, dict) for tool in tools)
            ):
                raise ObservationError("tools value is not null or an array of objects")
            messages_shape = _shape(messages)
            tools_shape = _shape(tools) if tools_present else {"type": "absent"}
            messages_shapes.append(messages_shape)
            tools_shapes.append(tools_shape)
            row_shapes.append(
                {
                    "fields": fields,
                    "messages": messages_shape,
                    "tools": tools_shape,
                }
            )
    except ObservationError as error:
        raise ObservationError(
            f"file {record.file_number} ({record.logical_path}) row {row_count or 1}: {error}"
        ) from error
    if row_count < 1:
        raise ObservationError(f"file {record.file_number} contains no rows")
    if len(tools_presence) != 1:
        raise ObservationError(f"file {record.file_number} has inconsistent tools fields")
    tools_field = "tools" if True in tools_presence else "__none__"
    schema_descriptor = {
        "physical_format": physical_format,
        "physical_field_names": physical_fields,
        "arrow_schema": arrow_schema,
        "json_field_sets": _count_shapes(field_sets),
        "messages_field": "messages",
        "tools_field": tools_field,
        "messages_value_shapes": _count_shapes(messages_shapes),
        "tools_value_shapes": _count_shapes(tools_shapes),
        "ordered_row_shapes_sha256": hashlib.sha256(
            _canonical_json(row_shapes, newline=False)
        ).hexdigest(),
    }
    return row_count, arrow_schema, physical_fields, schema_descriptor


def _observe_file(record: _FileRecord, stage: _AuthenticatedStage) -> dict[str, object]:
    parent_fd = stage.ptv2_data_fd if record.source == "ptv2" else stage.ptv3_root_fd
    name = Path(record.staged_path).name
    descriptor, before = _open_regular_at(parent_fd, name, f"file {record.file_number}")
    try:
        size, digest, _ = _read_descriptor(descriptor, retain=False)
        if size != record.physical_bytes or digest != record.physical_sha256:
            raise ObservationError(f"file {record.file_number} physical identity mismatch")
        row_count, _, _, row_schema = _observe_rows(descriptor, record)
        after = os.fstat(descriptor)
        pathname = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
    except OSError as error:
        raise ObservationError(f"file {record.file_number} changed while observing") from error
    finally:
        os.close(descriptor)
    if (
        _identity(before) != _identity(after)
        or _identity(after) != _identity(pathname)
        or after.st_nlink != 1
        or size != after.st_size
    ):
        raise ObservationError(f"file {record.file_number} changed while observing")
    if record.expected_row_count is not None and row_count != record.expected_row_count:
        raise ObservationError(
            f"file {record.file_number} row count mismatch: {row_count}/{record.expected_row_count}"
        )
    return {
        "source": record.source,
        "source_id": record.source_id,
        "revision": record.revision,
        "split": record.split,
        "logical_path": record.logical_path,
        "staged_path": record.staged_path,
        "physical_bytes": size,
        "physical_sha256": digest,
        "row_count": row_count,
        **row_schema,
        "row_schema_sha256": hashlib.sha256(_canonical_json(row_schema, newline=False)).hexdigest(),
    }


def _require_stage_stable(stage: _AuthenticatedStage, inputs: StageInputs) -> None:
    expected_ptv2_names = {
        _PTV2_PLAN_NAME,
        _PTV2_COMPLETION_NAME,
        "data",
    }
    expected_ptv2_data = {
        Path(record.staged_path).name for record in stage.records if record.source == "ptv2"
    }
    expected_ptv3_names = {
        _PTV3_MANIFEST_NAME,
        _PTV3_COMPLETION_NAME,
        *(record.staged_path for record in stage.records if record.source == "ptv3"),
    }
    try:
        ptv2_pathname = os.stat(inputs.ptv2_root, follow_symlinks=False)
        ptv3_pathname = os.stat(inputs.ptv3_root, follow_symlinks=False)
        if (
            _identity(os.fstat(stage.ptv2_root_fd)) != stage.ptv2_root_identity
            or _identity(ptv2_pathname) != stage.ptv2_root_identity
            or _identity(os.fstat(stage.ptv2_data_fd)) != stage.ptv2_data_identity
            or _identity(os.fstat(stage.ptv3_root_fd)) != stage.ptv3_root_identity
            or _identity(ptv3_pathname) != stage.ptv3_root_identity
            or set(os.listdir(stage.ptv2_root_fd)) != expected_ptv2_names
            or set(os.listdir(stage.ptv2_data_fd)) != expected_ptv2_data
            or set(os.listdir(stage.ptv3_root_fd)) != expected_ptv3_names
        ):
            raise ObservationError("staged trees changed while observing")
    except OSError as error:
        raise ObservationError("staged trees changed while observing") from error


def observe_stage_schemas(inputs: StageInputs) -> bytes:
    """Return a canonical, self-hashed physical row-schema observation."""
    if not isinstance(inputs, StageInputs):
        raise ObservationError("inputs must be a StageInputs value")
    stage = _authenticate_stage(inputs)
    try:
        files = [_observe_file(record, stage) for record in stage.records]
        _require_stage_stable(stage, inputs)
        _read_bound_path(inputs.ptv2_plan_path, inputs.ptv2_plan_sha256, "PTV2 plan")
        _read_bound_path(
            inputs.ptv2_completion_path,
            inputs.ptv2_completion_sha256,
            "PTV2 completion",
        )
        _read_bound_path(inputs.ptv3_plan_path, inputs.ptv3_plan_sha256, "PTV3 plan")
        _read_bound_path(
            inputs.ptv3_completion_path,
            inputs.ptv3_completion_sha256,
            "PTV3 completion",
        )
    finally:
        stage.close()
    payload: dict[str, object] = {
        "schema_version": _SCHEMA_VERSION,
        "inputs": {
            "ptv2": {
                "plan_path": inputs.ptv2_plan_path.name,
                "plan_sha256": inputs.ptv2_plan_sha256,
                "completion_path": inputs.ptv2_completion_path.name,
                "completion_sha256": inputs.ptv2_completion_sha256,
                "root": str(inputs.ptv2_root),
            },
            "ptv3": {
                "plan_path": inputs.ptv3_plan_path.name,
                "plan_sha256": inputs.ptv3_plan_sha256,
                "completion_path": inputs.ptv3_completion_path.name,
                "completion_sha256": inputs.ptv3_completion_sha256,
                "root": str(inputs.ptv3_root),
            },
        },
        "source_file_counts": {"ptv2": stage.ptv2_count, "ptv3": stage.ptv3_count},
        "file_count": len(files),
        "row_count": sum(cast("int", record["row_count"]) for record in files),
        "files": files,
    }
    observation_sha256 = hashlib.sha256(_canonical_json(payload)).hexdigest()
    return _canonical_json({**payload, "observation_sha256": observation_sha256})


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ptv2-plan", type=Path, required=True)
    parser.add_argument("--ptv2-plan-sha256", required=True)
    parser.add_argument("--ptv2-completion", type=Path, required=True)
    parser.add_argument("--ptv2-completion-sha256", required=True)
    parser.add_argument("--ptv2-root", type=Path, required=True)
    parser.add_argument("--ptv3-plan", type=Path, required=True)
    parser.add_argument("--ptv3-plan-sha256", required=True)
    parser.add_argument("--ptv3-completion", type=Path, required=True)
    parser.add_argument("--ptv3-completion-sha256", required=True)
    parser.add_argument("--ptv3-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def _publish_observation(output: Path, payload: bytes) -> None:
    if not output.is_absolute() or output.name in {"", ".", ".."}:
        raise ObservationError("output path must be absolute")
    parent_fd, _ = _open_root(output.parent, "observation output parent")
    descriptor: int | None = None
    try:
        flags = (
            os.O_WRONLY
            | os.O_CREAT
            | os.O_EXCL
            | getattr(os, "O_NOFOLLOW", 0)
            | getattr(os, "O_CLOEXEC", 0)
        )
        descriptor = os.open(output.name, flags, 0o440, dir_fd=parent_fd)
        offset = 0
        while offset < len(payload):
            offset += os.write(descriptor, payload[offset:])
        os.fsync(descriptor)
        created = os.fstat(descriptor)
        os.close(descriptor)
        descriptor = None
        current = os.stat(output.name, dir_fd=parent_fd, follow_symlinks=False)
        if (
            not stat.S_ISREG(created.st_mode)
            or created.st_nlink != 1
            or _identity(created) != _identity(current)
            or created.st_size != len(payload)
        ):
            raise ObservationError("published observation identity mismatch")
        os.fsync(parent_fd)
    except FileExistsError as error:
        raise ObservationError("observation output already exists") from error
    finally:
        if descriptor is not None:
            os.close(descriptor)
        os.close(parent_fd)


def main() -> int:
    """Run the authenticated row-schema observer CLI."""
    arguments = _parse_args()
    payload = observe_stage_schemas(
        StageInputs(
            ptv2_plan_path=arguments.ptv2_plan,
            ptv2_plan_sha256=arguments.ptv2_plan_sha256,
            ptv2_completion_path=arguments.ptv2_completion,
            ptv2_completion_sha256=arguments.ptv2_completion_sha256,
            ptv2_root=arguments.ptv2_root,
            ptv3_plan_path=arguments.ptv3_plan,
            ptv3_plan_sha256=arguments.ptv3_plan_sha256,
            ptv3_completion_path=arguments.ptv3_completion,
            ptv3_completion_sha256=arguments.ptv3_completion_sha256,
            ptv3_root=arguments.ptv3_root,
        )
    )
    _publish_observation(arguments.output, payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
