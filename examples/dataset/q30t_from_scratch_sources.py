# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Authenticate the physical PTV2/PTV3 sources for Q30 from-scratch prompts."""

from __future__ import annotations

import hashlib
import json
import os
import re
import stat
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Any, Literal, cast

from q30t_from_scratch_policy import _read_stable_regular_file
from specdec_corpus_contracts import canonical_json, sha256_bytes
from stage_ptv23_sources import (
    SourceIdentity,
    SourceInventory,
    SourceManifestError,
    load_source_inventory,
)

__all__ = [
    "AuthenticatedSourceRegistry",
    "SourceRegistryEntry",
    "SourceRegistryError",
    "load_authenticated_source_registry",
    "resolve_source_path",
]

_REQUIREMENTS_SCHEMA = "q30t-ptv23-from-scratch-source-requirements-v1"
_SCIENTIFIC_IDENTITY = "q30t-ptv23-from-scratch-drafter-study-v1"
_STAGE_RECEIPT_SCHEMA = "q30t-ptv23-from-scratch-stage-receipt-v1"
_APPROVED_USE = "specdec-drafter-training"
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_REVISION = re.compile(r"^[0-9a-f]{40}$")
_REQUIREMENTS_KEYS = frozenset(
    {"schema_version", "scientific_identity", "required_source_fields", "source_families", "adapters"}
)
_REQUIRED_SOURCE_FIELDS = (
    "repository",
    "revision",
    "configuration",
    "split",
    "relative_path",
    "bytes",
    "sha256",
    "row_count",
    "row_schema_sha256",
    "license_expression",
    "approved_use",
    "lane",
    "adapter",
)
_LANES = (
    "agentless_swe",
    "interactive_swe",
    "general_tool",
    "math",
    "code",
    "stem",
    "instruction",
    "multilingual",
)
_ADAPTER_KEYS = frozenset({"messages_field", "tools_field"})
_STAGE_RECEIPT_KEYS = frozenset({"schema_version", "source_inventory", "workers", "files"})
_SOURCE_INVENTORY_KEYS = frozenset({"path", "sha256"})
_WORKER_KEYS = frozenset({"requested", "effective"})
_STAGE_FILE_KEYS = frozenset(
    {
        "source_id",
        "repository",
        "configuration",
        "split",
        "revision",
        "relative_path",
        "staged_path",
        "bytes",
        "sha256",
        "row_count",
        "row_schema",
        "row_schema_sha256",
        "adapter",
        "lane",
    }
)


class SourceRegistryError(ValueError):
    """A source requirement, receipt, or physical file is not authenticated."""


@dataclass(frozen=True)
class SourceRegistryEntry:
    """One complete, policy-mapped physical source file identity."""

    source_id: str
    repository: str
    revision: str
    configuration: str
    split: str
    relative_path: str
    bytes: int
    file_sha256: str
    row_count: int
    row_schema_sha256: str
    license_expression: str
    approved_use: Literal["specdec-drafter-training"]
    lane: str
    adapter: str


@dataclass(frozen=True)
class AuthenticatedSourceRegistry:
    """Deterministically ordered physical source entries and their evidence."""

    entries: tuple[SourceRegistryEntry, ...]
    requirements_sha256: str
    stage_receipt_sha256s: tuple[str, ...]
    registry_sha256: str
    _entry_paths: Mapping[SourceRegistryEntry, Path] = field(
        default_factory=lambda: MappingProxyType({}), repr=False, compare=False
    )


def load_authenticated_source_registry(
    requirements_path: Path,
    *,
    expected_sha256: str,
    stage_receipts: Mapping[str, Path],
) -> AuthenticatedSourceRegistry:
    """Load complete staged source evidence against an independently supplied digest."""
    requirements_raw = _read_bound_file(requirements_path, expected_sha256, "requirements")
    allowed_sources, adapters = _parse_requirements(requirements_raw)
    if not isinstance(stage_receipts, Mapping) or not stage_receipts:
        raise SourceRegistryError("stage receipts must be a non-empty digest-to-path mapping")
    if any(
        not isinstance(receipt_sha256, str)
        or _SHA256.fullmatch(receipt_sha256) is None
        or not isinstance(receipt_path, Path)
        for receipt_sha256, receipt_path in stage_receipts.items()
    ):
        raise SourceRegistryError("stage receipts must have SHA-256 keys and Path values")

    entries: list[SourceRegistryEntry] = []
    paths: dict[SourceRegistryEntry, Path] = {}
    receipt_paths: set[Path] = set()
    for receipt_sha256, receipt_path in sorted(stage_receipts.items()):
        if not isinstance(receipt_path, Path):
            raise SourceRegistryError("stage receipt paths must be Path values")
        resolved_receipt_path = receipt_path.resolve(strict=False)
        if resolved_receipt_path in receipt_paths:
            raise SourceRegistryError("stage receipt paths must be unique")
        receipt_paths.add(resolved_receipt_path)
        receipt = _load_stage_receipt(receipt_path, receipt_sha256)
        receipt_entries, receipt_paths_by_entry = _reconcile_stage_receipt(
            receipt_path,
            receipt,
            allowed_sources=allowed_sources,
            adapters=adapters,
        )
        for entry, physical_path in zip(receipt_entries, receipt_paths_by_entry, strict=True):
            if entry in paths:
                raise SourceRegistryError("duplicate physical source entry")
            entries.append(entry)
            paths[entry] = physical_path

    declared_source_ids = frozenset().union(*allowed_sources.values())
    observed_source_ids = {entry.source_id for entry in entries}
    if observed_source_ids != declared_source_ids:
        raise SourceRegistryError("stage receipts do not cover every declared source ID")

    ordered_entries = tuple(sorted(entries, key=_entry_sort_key))
    logical_identities = {
        (entry.repository, entry.configuration, entry.split, entry.relative_path)
        for entry in ordered_entries
    }
    if len(logical_identities) != len(ordered_entries):
        raise SourceRegistryError("duplicate source logical identity")
    registry_sha256 = sha256_bytes(
        canonical_json(
            {
                "requirements_sha256": expected_sha256,
                "entries": [asdict(entry) for entry in ordered_entries],
            }
        )
    )
    return AuthenticatedSourceRegistry(
        entries=ordered_entries,
        requirements_sha256=expected_sha256,
        stage_receipt_sha256s=tuple(sorted(stage_receipts)),
        registry_sha256=registry_sha256,
        _entry_paths=MappingProxyType(paths),
    )


def resolve_source_path(registry: AuthenticatedSourceRegistry, entry: SourceRegistryEntry) -> Path:
    """Return a revalidated physical path for one registry entry."""
    if not isinstance(registry, AuthenticatedSourceRegistry) or not isinstance(entry, SourceRegistryEntry):
        raise SourceRegistryError("source resolution requires registry entry values")
    path = registry._entry_paths.get(entry)
    if path is None:
        raise SourceRegistryError("registered physical source path is unavailable")
    size, digest = _stable_file_identity(path, "registered physical source path")
    if size != entry.bytes or digest != entry.file_sha256:
        raise SourceRegistryError("registered physical source path no longer matches its identity")
    return path


def _read_bound_file(path: Path, expected_sha256: object, label: str) -> bytes:
    if not isinstance(path, Path) or not isinstance(expected_sha256, str) or not _SHA256.fullmatch(
        expected_sha256
    ):
        raise SourceRegistryError(f"{label} SHA-256 binding is invalid")
    try:
        raw = _read_stable_regular_file(path)
    except ValueError as error:
        raise SourceRegistryError(f"{label} is not a stable regular file") from error
    if sha256_bytes(raw) != expected_sha256:
        raise SourceRegistryError(f"{label} SHA-256 binding does not match")
    return raw


def _parse_requirements(raw: bytes) -> tuple[dict[str, frozenset[str]], dict[str, tuple[str, str]]]:
    root = _load_canonical_mapping(raw, "source requirements")
    _require_exact_keys(root, _REQUIREMENTS_KEYS, "source requirements")
    if root.get("schema_version") != _REQUIREMENTS_SCHEMA:
        raise SourceRegistryError("source requirements schema version is invalid")
    if root.get("scientific_identity") != _SCIENTIFIC_IDENTITY:
        raise SourceRegistryError("source requirements scientific identity is invalid")
    if tuple(root.get("required_source_fields", ())) != _REQUIRED_SOURCE_FIELDS:
        raise SourceRegistryError("source requirements fields are invalid")
    source_families = _mapping(root.get("source_families"), "source requirements source_families")
    if tuple(source_families) != tuple(sorted(source_families)) or set(source_families) != set(_LANES):
        raise SourceRegistryError("source requirements lanes are invalid")
    allowed_sources: dict[str, frozenset[str]] = {}
    for lane in _LANES:
        values = source_families[lane]
        if not isinstance(values, list) or not all(
            isinstance(value, str) and value for value in values
        ) or len(values) != len(set(values)):
            raise SourceRegistryError(f"source requirements lane {lane} is invalid")
        allowed_sources[lane] = frozenset(values)

    raw_adapters = _mapping(root.get("adapters"), "source requirements adapters")
    if not raw_adapters:
        raise SourceRegistryError("source requirements has no adapters")
    adapters: dict[str, tuple[str, str]] = {}
    for name, value in raw_adapters.items():
        if not isinstance(name, str) or not name:
            raise SourceRegistryError("source adapter name is invalid")
        adapter = _mapping(value, f"source adapter {name}")
        _require_exact_keys(adapter, _ADAPTER_KEYS, f"source adapter {name}")
        fields = (adapter.get("messages_field"), adapter.get("tools_field"))
        if not all(isinstance(field, str) and field for field in fields):
            raise SourceRegistryError(f"source adapter {name} is invalid")
        adapters[name] = cast("tuple[str, str]", fields)
    return allowed_sources, adapters


def _load_stage_receipt(path: Path, expected_sha256: object) -> dict[str, Any]:
    raw = _read_bound_file(path, expected_sha256, "stage receipt")
    root = _load_canonical_mapping(raw, "stage receipt")
    _require_exact_keys(root, _STAGE_RECEIPT_KEYS, "stage receipt")
    if root.get("schema_version") != _STAGE_RECEIPT_SCHEMA:
        raise SourceRegistryError("stage receipt schema version is invalid")
    return root


def _reconcile_stage_receipt(
    receipt_path: Path,
    receipt: Mapping[str, Any],
    *,
    allowed_sources: Mapping[str, frozenset[str]],
    adapters: Mapping[str, tuple[str, str]],
) -> tuple[tuple[SourceRegistryEntry, ...], tuple[Path, ...]]:
    inventory_path = _source_inventory_path(receipt_path, receipt.get("source_inventory"))
    inventory_raw = _read_bound_file(
        inventory_path,
        _mapping(receipt["source_inventory"], "stage receipt source inventory").get("sha256"),
        "source inventory",
    )
    try:
        inventory = load_source_inventory(inventory_path)
    except SourceManifestError as error:
        raise SourceRegistryError("source inventory is not an authenticated staged inventory") from error
    inventory_records = _inventory_records(inventory_raw, inventory)
    _validate_workers(receipt.get("workers"))
    files = receipt.get("files")
    if not isinstance(files, list) or not files:
        raise SourceRegistryError("stage receipt files are invalid")

    entries: list[SourceRegistryEntry] = []
    physical_paths: list[Path] = []
    seen: set[tuple[str, str, str, str, str]] = set()
    for index, raw_file in enumerate(files):
        record = _mapping(raw_file, f"stage receipt files[{index}]")
        _require_exact_keys(record, _STAGE_FILE_KEYS, f"stage receipt files[{index}]")
        identity = _stage_identity(record, f"stage receipt files[{index}]")
        if identity in seen:
            raise SourceRegistryError("duplicate stage receipt source file")
        seen.add(identity)
        source, inventory_record = inventory_records.get(identity, (None, None))
        if source is None or inventory_record is None:
            raise SourceRegistryError("stage receipt contains an unmapped source file")
        entry, physical_path = _entry_from_record(
            receipt_path.parent,
            record,
            source=source,
            inventory_record=inventory_record,
            allowed_sources=allowed_sources,
            adapters=adapters,
        )
        entries.append(entry)
        physical_paths.append(physical_path)
    if seen != set(inventory_records):
        raise SourceRegistryError("stage receipt does not cover every staged source file")
    return tuple(entries), tuple(physical_paths)


def _source_inventory_path(receipt_path: Path, value: object) -> Path:
    source_inventory = _mapping(value, "stage receipt source inventory")
    _require_exact_keys(source_inventory, _SOURCE_INVENTORY_KEYS, "stage receipt source inventory")
    relative = _relative_path(source_inventory.get("path"), "stage receipt source inventory.path")
    if relative != Path("SOURCE_INVENTORY.json"):
        raise SourceRegistryError("stage receipt source inventory path is invalid")
    return receipt_path.parent / relative


def _inventory_records(
    raw: bytes, inventory: SourceInventory
) -> dict[tuple[str, str, str, str, str], tuple[SourceIdentity, Mapping[str, Any]]]:
    root = _load_mapping(raw, "source inventory")
    files = root.get("files")
    if not isinstance(files, list):
        raise SourceRegistryError("source inventory files are invalid")
    sources: dict[tuple[str, str, str, str, str], SourceIdentity] = {}
    for source in inventory.sources:
        for source_file in source.files:
            identity = (
                source.repository_id,
                source.configuration,
                source.split,
                source.revision,
                source_file.path,
            )
            sources[identity] = source
    records: dict[tuple[str, str, str, str, str], tuple[SourceIdentity, Mapping[str, Any]]] = {}
    for raw_record in files:
        record = _mapping(raw_record, "source inventory file")
        identity = (
            record.get("repository_id"),
            record.get("configuration"),
            record.get("split"),
            record.get("revision"),
            record.get("source_path"),
        )
        if not all(isinstance(value, str) for value in identity):
            raise SourceRegistryError("source inventory identity is invalid")
        typed_identity = cast("tuple[str, str, str, str, str]", identity)
        source = sources.get(typed_identity)
        if source is None or typed_identity in records:
            raise SourceRegistryError("source inventory identity is invalid")
        records[typed_identity] = (source, record)
    if set(records) != set(sources):
        raise SourceRegistryError("source inventory does not cover every source file")
    return records


def _entry_from_record(
    stage_root: Path,
    record: Mapping[str, Any],
    *,
    source: SourceIdentity,
    inventory_record: Mapping[str, Any],
    allowed_sources: Mapping[str, frozenset[str]],
    adapters: Mapping[str, tuple[str, str]],
) -> tuple[SourceRegistryEntry, Path]:
    source_id = _nonempty_string(record.get("source_id"), "stage source id")
    lane = _nonempty_string(record.get("lane"), "stage source lane")
    if source_id != source.cell or lane not in allowed_sources or source_id not in allowed_sources[lane]:
        raise SourceRegistryError("stage source lane is not explicitly declared")
    revision = _nonempty_string(record.get("revision"), "stage source revision")
    if _REVISION.fullmatch(revision) is None or revision != source.revision:
        raise SourceRegistryError("stage source revision is not exact")
    repository, configuration, split, _, relative_path = _stage_identity(record, "stage source")
    if (
        repository != source.repository_id
        or configuration != source.configuration
        or split != source.split
    ):
        raise SourceRegistryError("stage source identity does not match the inventory")
    bytes_ = _positive_int(record.get("bytes"), "stage source bytes")
    file_sha256 = _sha256(record.get("sha256"), "stage source SHA-256")
    if bytes_ != inventory_record.get("bytes") or file_sha256 != inventory_record.get("sha256"):
        raise SourceRegistryError("stage source file identity does not match the inventory")
    row_count = _positive_int(record.get("row_count"), "stage source row count")
    if row_count != inventory_record.get("raw_count"):
        raise SourceRegistryError("stage source row count does not match the inventory")
    row_schema = _mapping(record.get("row_schema"), "stage source row schema")
    row_schema_sha256 = _sha256(record.get("row_schema_sha256"), "stage source row schema SHA-256")
    if sha256_bytes(canonical_json(row_schema)) != row_schema_sha256:
        raise SourceRegistryError("stage source row schema does not match its digest")
    adapter = _nonempty_string(record.get("adapter"), "stage source adapter")
    adapter_fields = adapters.get(adapter)
    if adapter_fields is None or (
        row_schema.get("messages_field"),
        row_schema.get("tools_field"),
    ) != adapter_fields:
        raise SourceRegistryError("stage source adapter does not match its row schema")
    staged_relative = _relative_path(record.get("staged_path"), "stage source staged path")
    if staged_relative.as_posix() != inventory_record.get("staged_path"):
        raise SourceRegistryError("stage source staged path does not match the inventory")
    physical_path = stage_root / staged_relative
    size, digest = _stable_file_identity(physical_path, "stage source physical file")
    if size != bytes_ or digest != file_sha256:
        raise SourceRegistryError("stage source physical file does not match its identity")
    if not source.license_expression.strip() or source.approved_use is not True:
        raise SourceRegistryError("stage source lacks approved license use")
    return (
        SourceRegistryEntry(
            source_id=source_id,
            repository=repository,
            revision=revision,
            configuration=configuration,
            split=split,
            relative_path=relative_path,
            bytes=bytes_,
            file_sha256=file_sha256,
            row_count=row_count,
            row_schema_sha256=row_schema_sha256,
            license_expression=source.license_expression,
            approved_use=_APPROVED_USE,
            lane=lane,
            adapter=adapter,
        ),
        physical_path,
    )


def _stage_identity(record: Mapping[str, Any], label: str) -> tuple[str, str, str, str, str]:
    repository = _nonempty_string(record.get("repository"), f"{label} repository")
    configuration = _nonempty_string(record.get("configuration"), f"{label} configuration")
    split = _nonempty_string(record.get("split"), f"{label} split")
    revision = _nonempty_string(record.get("revision"), f"{label} revision")
    relative_path = _relative_path(record.get("relative_path"), f"{label} relative path").as_posix()
    return repository, configuration, split, revision, relative_path


def _validate_workers(value: object) -> None:
    workers = _mapping(value, "stage receipt workers")
    _require_exact_keys(workers, _WORKER_KEYS, "stage receipt workers")
    for key in _WORKER_KEYS:
        _positive_int(workers.get(key), f"stage receipt workers.{key}")


def _entry_sort_key(entry: SourceRegistryEntry) -> tuple[str, str, str, str]:
    return entry.repository, entry.configuration, entry.split, entry.relative_path


def _load_canonical_mapping(raw: bytes, label: str) -> dict[str, Any]:
    root = _load_mapping(raw, label)
    if raw != canonical_json(root) + b"\n":
        raise SourceRegistryError(f"{label} must use canonical JSON bytes")
    return root


def _load_mapping(raw: bytes, label: str) -> dict[str, Any]:
    try:
        value = json.loads(raw, object_pairs_hook=_reject_duplicate_keys)
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
        raise SourceRegistryError(f"{label} must be valid JSON with unique keys") from error
    return _mapping(value, label)


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON object key")
        result[key] = value
    return result


def _mapping(value: object, label: str) -> dict[str, Any]:
    if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
        raise SourceRegistryError(f"{label} must be a mapping with string keys")
    return value


def _require_exact_keys(value: Mapping[str, Any], expected: frozenset[str], label: str) -> None:
    if set(value) != expected:
        raise SourceRegistryError(f"{label} has unknown or missing keys")


def _nonempty_string(value: object, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise SourceRegistryError(f"{label} must be a non-empty string")
    return value


def _relative_path(value: object, label: str) -> Path:
    path = Path(_nonempty_string(value, label))
    if path.is_absolute() or ".." in path.parts or path == Path("."):
        raise SourceRegistryError(f"{label} must be a safe relative path")
    return path


def _positive_int(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise SourceRegistryError(f"{label} must be a positive integer")
    return value


def _sha256(value: object, label: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise SourceRegistryError(f"{label} must be a lowercase SHA-256")
    return value


def _stable_file_identity(path: Path, label: str) -> tuple[int, str]:
    if not hasattr(os, "O_NOFOLLOW"):
        raise SourceRegistryError(f"{label} requires no-follow file descriptors")
    flags = os.O_RDONLY | os.O_NOFOLLOW | getattr(os, "O_CLOEXEC", 0)
    try:
        before_path = os.lstat(path)
        descriptor = os.open(path, flags)
    except OSError as error:
        raise SourceRegistryError(f"{label} is unavailable or changed while hashing") from error
    try:
        before = os.fstat(descriptor)
        if (
            not stat.S_ISREG(before_path.st_mode)
            or not stat.S_ISREG(before.st_mode)
            or _file_identity(before_path) != _file_identity(before)
        ):
            raise SourceRegistryError(f"{label} is not a stable regular file")
        digest = hashlib.sha256()
        size = 0
        while chunk := os.read(descriptor, 8 * 1024 * 1024):
            digest.update(chunk)
            size += len(chunk)
        after = os.fstat(descriptor)
    except OSError as error:
        raise SourceRegistryError(f"{label} changed while hashing") from error
    finally:
        os.close(descriptor)
    try:
        after_path = os.lstat(path)
    except OSError as error:
        raise SourceRegistryError(f"{label} changed while hashing") from error
    if (
        size != after.st_size
        or _file_identity(before) != _file_identity(after)
        or _file_identity(after_path) != _file_identity(after)
    ):
        raise SourceRegistryError(f"{label} changed while hashing")
    return size, digest.hexdigest()


def _file_identity(observation: os.stat_result) -> tuple[int, int, int, int, int, int, int]:
    return (
        observation.st_dev,
        observation.st_ino,
        observation.st_mode,
        observation.st_nlink,
        observation.st_size,
        observation.st_mtime_ns,
        observation.st_ctime_ns,
    )
