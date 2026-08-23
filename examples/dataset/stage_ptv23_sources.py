# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Validate and atomically stage the pinned B-prime/C/D source inventory."""

from __future__ import annotations

import argparse
import ctypes
import errno
import hashlib
import json
import os
import platform
import re
import shutil
import tempfile
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import TYPE_CHECKING, Any

import yaml

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

__all__ = [
    "SourceFile",
    "SourceIdentity",
    "SourceInventory",
    "SourceManifestError",
    "load_source_inventory",
    "stage_source_inventory",
    "validate_ptv3_config_pins",
]

_REVISION = re.compile(r"^[0-9a-f]{40}$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_REPOSITORY = re.compile(r"^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+$")
_ROOT_KEYS = frozenset({"schema_version", "name", "sources"})
_SOURCE_KEYS = frozenset(
    {
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
)
_FILE_KEYS = frozenset({"path", "bytes", "sha256"})
_PTV3_SOURCE_KEYS = frozenset(
    {
        "repo_id",
        "configuration",
        "split",
        "revision",
        "license_expression",
        "approved_use",
        "files",
    }
)
_RECEIPT_KEYS = frozenset(
    {"schema_version", "name", "source_manifest_sha256", "complete", "raw_counts", "files"}
)
_RECEIPT_FILE_KEYS = frozenset(
    {
        "repository_id",
        "configuration",
        "split",
        "revision",
        "license_expression",
        "approved_use",
        "cell",
        "lane",
        "source_path",
        "staged_path",
        "bytes",
        "sha256",
        "raw_count",
    }
)


class SourceManifestError(ValueError):
    """A source descriptor or staged artifact is incomplete or stale."""


@dataclass(frozen=True)
class SourceFile:
    """One immutable physical file in a source repository."""

    path: str
    bytes: int
    sha256: str


@dataclass(frozen=True)
class SourceIdentity:
    """One approved repository configuration and split."""

    repository_id: str
    configuration: str
    split: str
    revision: str
    license_expression: str
    approved_use: bool
    cell: str
    lane: str
    files: Sequence[SourceFile]


@dataclass(frozen=True)
class SourceInventory:
    """Validated source identities plus raw counts after physical staging."""

    schema_version: int
    name: str
    sources: Sequence[SourceIdentity]
    manifest_sha256: str
    canonical_manifest: bytes
    raw_counts: Mapping[str, int]
    staged_root: Path | None = None


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_json(value: object) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode()


def _require_exact_keys(value: Mapping[str, Any], expected: frozenset[str], label: str) -> None:
    if set(value) != expected:
        difference = sorted(set(value) ^ expected)
        raise SourceManifestError(f"{label} has unknown or missing keys: {difference}")


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
        raise SourceManifestError(f"{label} must be a mapping with string keys")
    return value


def _nonempty_string(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise SourceManifestError(f"{label} must be a non-empty string")
    return value


def _parse_file(value: Any, label: str) -> SourceFile:
    record = _mapping(value, label)
    _require_exact_keys(record, _FILE_KEYS, label)
    relative = Path(_nonempty_string(record["path"], f"{label}.path"))
    if relative.is_absolute() or ".." in relative.parts or relative == Path("."):
        raise SourceManifestError(f"{label}.path must be a safe relative path")
    size = record["bytes"]
    if isinstance(size, bool) or not isinstance(size, int) or size <= 0:
        raise SourceManifestError(f"{label}.bytes must be a positive byte count")
    digest = record["sha256"]
    if not isinstance(digest, str) or _SHA256.fullmatch(digest) is None:
        raise SourceManifestError(f"{label}.sha256 must be a lowercase SHA-256")
    return SourceFile(relative.as_posix(), size, digest)


def _parse_source(value: Any, index: int) -> SourceIdentity:
    label = f"sources[{index}]"
    record = _mapping(value, label)
    _require_exact_keys(record, _SOURCE_KEYS, label)
    repository_id = _nonempty_string(record["repository_id"], f"{label}.repository_id")
    if _REPOSITORY.fullmatch(repository_id) is None:
        raise SourceManifestError(f"{label}.repository_id must be an owner/name identifier")
    revision = record["revision"]
    if not isinstance(revision, str) or _REVISION.fullmatch(revision) is None:
        raise SourceManifestError(
            f"{label}.revision must be an exact lowercase 40-character commit"
        )
    license_expression = record["license_expression"]
    if not isinstance(license_expression, str) or not license_expression.strip():
        raise SourceManifestError(f"{label} requires a non-empty license expression")
    approved_use = record["approved_use"]
    if not isinstance(approved_use, bool):
        raise SourceManifestError(f"{label}.approved_use must be a boolean")
    if not approved_use:
        raise SourceManifestError(f"{label} does not have approved use")
    files = record["files"]
    if not isinstance(files, list) or not files:
        raise SourceManifestError(f"{label}.files must be a non-empty list")
    return SourceIdentity(
        repository_id=repository_id,
        configuration=_nonempty_string(record["configuration"], f"{label}.configuration"),
        split=_nonempty_string(record["split"], f"{label}.split"),
        revision=revision,
        license_expression=license_expression,
        approved_use=approved_use,
        cell=_nonempty_string(record["cell"], f"{label}.cell"),
        lane=_nonempty_string(record["lane"], f"{label}.lane"),
        files=tuple(
            _parse_file(file, f"{label}.files[{file_index}]")
            for file_index, file in enumerate(files)
        ),
    )


def load_source_inventory(path: Path) -> SourceInventory:
    """Load a pinned source plan or verify and load its published inventory receipt."""
    try:
        raw = path.read_bytes()
        payload = json.loads(raw)
    except (OSError, json.JSONDecodeError) as error:
        raise SourceManifestError(f"unable to read source manifest: {path}") from error
    root = _mapping(payload, "source manifest")
    if set(root) == _RECEIPT_KEYS:
        return _load_published_inventory(path, root)
    _require_exact_keys(root, _ROOT_KEYS, "source manifest")
    if root["schema_version"] != 1:
        raise SourceManifestError("source manifest schema_version must be 1")
    name = _nonempty_string(root["name"], "source manifest name")
    records = root["sources"]
    if not isinstance(records, list) or not records:
        raise SourceManifestError("source manifest must contain at least one source")
    sources = tuple(_parse_source(record, index) for index, record in enumerate(records))
    logical_files: set[tuple[str, str, str]] = set()
    source_splits: set[tuple[str, str, str, str]] = set()
    for source in sources:
        source_split = (
            source.repository_id,
            source.configuration,
            source.split,
            source.revision,
        )
        if source_split in source_splits:
            raise SourceManifestError(
                f"duplicate source split: {source.repository_id}:{source.split}"
            )
        source_splits.add(source_split)
        for file in source.files:
            logical = (source.repository_id, source.revision, file.path)
            if logical in logical_files:
                raise SourceManifestError(
                    f"duplicate logical file: {source.repository_id}@{source.revision}:{file.path}"
                )
            logical_files.add(logical)
    canonical_manifest = _canonical_json(root)
    return SourceInventory(
        schema_version=1,
        name=name,
        sources=sources,
        manifest_sha256=hashlib.sha256(canonical_manifest).hexdigest(),
        canonical_manifest=canonical_manifest,
        raw_counts=MappingProxyType({}),
    )


def _load_published_inventory(path: Path, receipt: Mapping[str, Any]) -> SourceInventory:
    if path.is_symlink() or path.name != "SOURCE_INVENTORY.json":
        raise SourceManifestError("staged inventory receipt path is invalid")
    if receipt["schema_version"] != 1 or receipt["complete"] is not True:
        raise SourceManifestError("staged inventory receipt is incomplete")
    staged_root = path.parent.resolve(strict=True)
    preserved_plan = staged_root / "SOURCE_PLAN.json"
    if preserved_plan.is_symlink() or not preserved_plan.is_file():
        raise SourceManifestError("preserved source plan is missing or invalid")
    inventory = load_source_inventory(preserved_plan)
    if staged_root.name != inventory.manifest_sha256:
        raise SourceManifestError("published source plan content address mismatch")
    return _verify_staged(inventory, staged_root)


def validate_ptv3_config_pins(path: Path) -> None:
    """Reject convenience dataset YAML unless every entry is production-pinned."""
    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as error:
        raise SourceManifestError(f"unable to read PTV3 dataset config: {path}") from error
    root = _mapping(payload, "PTV3 dataset config")
    if set(root) != {"datasets"}:
        raise SourceManifestError("PTV3 dataset config has unknown or missing keys")
    datasets = root.get("datasets")
    if not isinstance(datasets, list) or not datasets:
        raise SourceManifestError("PTV3 dataset config has no datasets")
    for index, value in enumerate(datasets):
        label = f"datasets[{index}]"
        record = _mapping(value, label)
        repository_id = record.get("repo_id", label)
        revision = record.get("revision")
        if not isinstance(revision, str) or _REVISION.fullmatch(revision) is None:
            raise SourceManifestError(f"unpinned dataset revision: {repository_id}")
        _require_exact_keys(record, _PTV3_SOURCE_KEYS, label)
        repository_id = _nonempty_string(record["repo_id"], f"{label}.repo_id")
        if _REPOSITORY.fullmatch(repository_id) is None:
            raise SourceManifestError(f"{label}.repo_id must be an owner/name identifier")
        _nonempty_string(record["configuration"], f"{label}.configuration")
        _nonempty_string(record["split"], f"{label}.split")
        if (
            not isinstance(record.get("license_expression"), str)
            or not record["license_expression"].strip()
        ):
            raise SourceManifestError(f"unpinned dataset license: {repository_id}")
        if record.get("approved_use") is not True:
            raise SourceManifestError(f"unapproved dataset use: {repository_id}")
        if not isinstance(record.get("files"), list) or not record["files"]:
            raise SourceManifestError(f"unpinned dataset files: {repository_id}")
        for file_index, file in enumerate(record["files"]):
            _parse_file(file, f"{label}.files[{file_index}]")


def _local_candidates(root: Path, source: SourceIdentity, file: SourceFile) -> tuple[Path, ...]:
    cache_name = f"datasets--{source.repository_id.replace('/', '--')}"
    return (
        root / source.repository_id / source.revision / file.path,
        root / cache_name / "snapshots" / source.revision / file.path,
    )


def _copy_stream(source: Path, destination: Path) -> tuple[int, str]:
    digest = hashlib.sha256()
    size = 0
    destination.parent.mkdir(parents=True, exist_ok=True)
    with source.open("rb") as input_stream, destination.open("xb") as output_stream:
        for chunk in iter(lambda: input_stream.read(8 * 1024 * 1024), b""):
            output_stream.write(chunk)
            digest.update(chunk)
            size += len(chunk)
        output_stream.flush()
        os.fsync(output_stream.fileno())
    return size, digest.hexdigest()


def _download_stream(
    source: SourceIdentity, file: SourceFile, destination: Path
) -> tuple[int, str]:
    quoted_path = urllib.parse.quote(file.path, safe="/")
    quoted_repository = urllib.parse.quote(source.repository_id, safe="/")
    url = (
        f"https://huggingface.co/datasets/{quoted_repository}/resolve/"
        f"{source.revision}/{quoted_path}?download=true"
    )
    digest = hashlib.sha256()
    size = 0
    destination.parent.mkdir(parents=True, exist_ok=True)
    with (
        urllib.request.urlopen(url, timeout=120) as response,  # nosec B310: fixed HTTPS host.
        destination.open("xb") as output_stream,
    ):
        for chunk in iter(lambda: response.read(8 * 1024 * 1024), b""):
            output_stream.write(chunk)
            digest.update(chunk)
            size += len(chunk)
        output_stream.flush()
        os.fsync(output_stream.fileno())
    return size, digest.hexdigest()


def _raw_count(path: Path) -> int:
    if path.suffix == ".parquet":
        try:
            import pyarrow.parquet as pq  # pyright: ignore[reportMissingImports]
        except ImportError as error:
            raise SourceManifestError("pyarrow is required to count Parquet source rows") from error
        count = pq.ParquetFile(path).metadata.num_rows
    else:
        count = 0
        with path.open("rb") as stream:
            for line in stream:
                if line.strip():
                    count += 1
    if count <= 0:
        raise SourceManifestError(f"source file contains no raw rows: {path.name}")
    return count


def _relative_stage_path(source: SourceIdentity, file: SourceFile) -> Path:
    return Path("sources") / source.repository_id / source.revision / file.path


def _write_json_durable(path: Path, payload: Mapping[str, Any]) -> None:
    with path.open("x", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def _write_bytes_durable(path: Path, payload: bytes) -> None:
    with path.open("xb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _rename_no_replace(source: Path, destination: Path) -> None:
    """Atomically publish a sibling directory without replacing any destination."""
    if source.parent.resolve() != destination.parent.resolve():
        raise SourceManifestError("publication partial must be a sibling of its destination")
    library = ctypes.CDLL(None, use_errno=True)
    source_bytes = os.fsencode(source)
    destination_bytes = os.fsencode(destination)
    system = platform.system()
    if system == "Linux":
        try:
            rename = library.renameat2
        except AttributeError as error:
            raise SourceManifestError("atomic no-replace rename is unavailable") from error
        rename.argtypes = [
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_uint,
        ]
        rename.restype = ctypes.c_int
        result = rename(-100, source_bytes, -100, destination_bytes, 1)
    elif system == "Darwin":
        rename = library.renamex_np
        rename.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_uint]
        rename.restype = ctypes.c_int
        result = rename(source_bytes, destination_bytes, 0x00000004)
    else:
        raise SourceManifestError(f"atomic no-replace rename is unsupported on {system}")
    if result == 0:
        return
    error_number = ctypes.get_errno()
    if error_number == errno.EEXIST:
        raise FileExistsError(error_number, os.strerror(error_number), destination)
    raise SourceManifestError(f"atomic no-replace rename failed: {os.strerror(error_number)}")


def _receipt_payload(
    inventory: SourceInventory, records: Sequence[Mapping[str, Any]], raw_counts: Mapping[str, int]
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "name": inventory.name,
        "source_manifest_sha256": inventory.manifest_sha256,
        "complete": True,
        "raw_counts": dict(sorted(raw_counts.items())),
        "files": list(records),
    }


def _verify_staged(
    inventory: SourceInventory, root: Path, *, enforce_content_address: bool = True
) -> SourceInventory:
    if root.is_symlink() or not root.is_dir():
        raise SourceManifestError("stale staged inventory root")
    if enforce_content_address and root.name != inventory.manifest_sha256:
        raise SourceManifestError("published source plan content address mismatch")
    preserved_plan = root / "SOURCE_PLAN.json"
    if (
        preserved_plan.is_symlink()
        or not preserved_plan.is_file()
        or preserved_plan.read_bytes() != inventory.canonical_manifest
        or _sha256_file(preserved_plan) != inventory.manifest_sha256
    ):
        raise SourceManifestError("preserved source plan identity mismatch")
    receipt_path = root / "SOURCE_INVENTORY.json"
    if receipt_path.is_symlink() or not receipt_path.is_file():
        raise SourceManifestError("stale staged inventory receipt")
    try:
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise SourceManifestError("stale staged inventory receipt") from error
    if not isinstance(receipt, dict):
        raise SourceManifestError("stale staged inventory receipt")
    files = receipt.get("files")
    raw_counts = receipt.get("raw_counts")
    if (
        set(receipt) != _RECEIPT_KEYS
        or receipt.get("schema_version") != 1
        or receipt.get("name") != inventory.name
        or receipt.get("source_manifest_sha256") != inventory.manifest_sha256
        or receipt.get("complete") is not True
        or not isinstance(files, list)
        or not isinstance(raw_counts, dict)
    ):
        raise SourceManifestError("stale staged inventory receipt")
    expected = {
        (source.repository_id, source.revision, file.path): (source, file)
        for source in inventory.sources
        for file in source.files
    }
    if len(files) != len(expected):
        raise SourceManifestError("stale staged inventory file set")
    seen: set[tuple[str, str, str]] = set()
    recorded_counts: dict[str, int] = {}
    for record in files:
        if not isinstance(record, dict):
            raise SourceManifestError("stale staged inventory file record")
        _require_exact_keys(record, _RECEIPT_FILE_KEYS, "staged inventory file record")
        repository_id = record.get("repository_id")
        revision = record.get("revision")
        source_path = record.get("source_path")
        if (
            not isinstance(repository_id, str)
            or not isinstance(revision, str)
            or not isinstance(source_path, str)
        ):
            raise SourceManifestError("stale staged inventory file record")
        key = (repository_id, revision, source_path)
        expected_record = expected.get(key)
        relative = record.get("staged_path")
        if expected_record is None or key in seen or not isinstance(relative, str):
            raise SourceManifestError("stale staged inventory file record")
        seen.add(key)
        source, descriptor = expected_record
        if (
            record.get("configuration") != source.configuration
            or record.get("split") != source.split
            or record.get("license_expression") != source.license_expression
            or record.get("approved_use") is not True
            or record.get("cell") != source.cell
            or record.get("lane") != source.lane
            or record.get("bytes") != descriptor.bytes
            or record.get("sha256") != descriptor.sha256
        ):
            raise SourceManifestError("stale staged source identity")
        raw_count = record.get("raw_count")
        if isinstance(raw_count, bool) or not isinstance(raw_count, int) or raw_count < 1:
            raise SourceManifestError("stale staged inventory raw counts")
        path = (root / relative).resolve(strict=False)
        if (
            not path.is_relative_to(root.resolve())
            or not path.is_file()
            or path.stat().st_size != descriptor.bytes
            or _sha256_file(path) != descriptor.sha256
        ):
            raise SourceManifestError(f"stale staged file: {relative}")
        actual_raw_count = _raw_count(path)
        if raw_count != actual_raw_count:
            raise SourceManifestError("stale staged inventory raw counts")
        recorded_counts[source.cell] = recorded_counts.get(source.cell, 0) + actual_raw_count
    if seen != set(expected):
        raise SourceManifestError("stale staged inventory file set")
    parsed_counts: dict[str, int] = {}
    for cell, count in raw_counts.items():
        if (
            not isinstance(cell, str)
            or isinstance(count, bool)
            or not isinstance(count, int)
            or count < 1
        ):
            raise SourceManifestError("stale staged inventory raw counts")
        parsed_counts[cell] = count
    if parsed_counts != recorded_counts:
        raise SourceManifestError("stale staged inventory raw counts")
    return SourceInventory(
        schema_version=inventory.schema_version,
        name=inventory.name,
        sources=inventory.sources,
        manifest_sha256=inventory.manifest_sha256,
        canonical_manifest=inventory.canonical_manifest,
        raw_counts=MappingProxyType(parsed_counts),
        staged_root=root,
    )


def stage_source_inventory(
    inventory: SourceInventory,
    *,
    durable_root: Path,
    scratch_root: Path,
    local_source_root: Path | None = None,
) -> SourceInventory:
    """Stream, verify, count, and atomically publish one content-bound source tree."""
    durable_root = durable_root.expanduser().resolve(strict=False)
    scratch_root = scratch_root.expanduser().resolve(strict=False)
    local_source_root = (
        local_source_root.expanduser().resolve(strict=True)
        if local_source_root is not None
        else None
    )
    output_root = durable_root / inventory.manifest_sha256
    if output_root.is_symlink():
        raise SourceManifestError("stale staged inventory root")
    if output_root.exists():
        return _verify_staged(inventory, output_root)
    durable_root.mkdir(parents=True, exist_ok=True)
    scratch_root.mkdir(parents=True, exist_ok=True)
    scratch_partial = Path(tempfile.mkdtemp(prefix=".ptv23-source-stage.", dir=scratch_root))
    durable_partial = durable_root / f".{inventory.manifest_sha256}.partial-{os.getpid()}"
    if durable_partial.exists() or durable_partial.is_symlink():
        shutil.rmtree(scratch_partial)
        raise SourceManifestError(f"durable partial already exists: {durable_partial}")
    records: list[dict[str, Any]] = []
    raw_counts: dict[str, int] = {}
    try:
        for source in inventory.sources:
            for file in source.files:
                relative = _relative_stage_path(source, file)
                destination = scratch_partial / relative
                if local_source_root is None:
                    size, digest = _download_stream(source, file, destination)
                else:
                    candidates = _local_candidates(local_source_root, source, file)
                    local = next(
                        (candidate for candidate in candidates if candidate.is_file()), None
                    )
                    if local is None:
                        raise SourceManifestError(
                            f"local source file is absent: {source.repository_id}:{file.path}"
                        )
                    size, digest = _copy_stream(local, destination)
                if size != file.bytes or digest != file.sha256:
                    raise SourceManifestError(
                        f"source file identity mismatch: {source.repository_id}:{file.path}"
                    )
                count = _raw_count(destination)
                raw_counts[source.cell] = raw_counts.get(source.cell, 0) + count
                records.append(
                    {
                        "repository_id": source.repository_id,
                        "configuration": source.configuration,
                        "split": source.split,
                        "revision": source.revision,
                        "license_expression": source.license_expression,
                        "approved_use": source.approved_use,
                        "cell": source.cell,
                        "lane": source.lane,
                        "source_path": file.path,
                        "staged_path": relative.as_posix(),
                        "bytes": size,
                        "sha256": digest,
                        "raw_count": count,
                    }
                )
        _write_bytes_durable(scratch_partial / "SOURCE_PLAN.json", inventory.canonical_manifest)
        _write_json_durable(
            scratch_partial / "SOURCE_INVENTORY.json",
            _receipt_payload(inventory, records, raw_counts),
        )
        _fsync_directory(scratch_partial)
        shutil.copytree(scratch_partial, durable_partial, copy_function=shutil.copyfile)
        _verify_staged(inventory, durable_partial, enforce_content_address=False)
        _fsync_directory(durable_partial)
        try:
            _rename_no_replace(durable_partial, output_root)
        except FileExistsError:
            shutil.rmtree(durable_partial, ignore_errors=True)
        _fsync_directory(durable_root)
        return _verify_staged(inventory, output_root)
    except BaseException:
        shutil.rmtree(durable_partial, ignore_errors=True)
        raise
    finally:
        shutil.rmtree(scratch_partial, ignore_errors=True)


def main() -> int:
    """Stage the pinned source plan from an HF cache/local tree or Hugging Face."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--durable-root", type=Path, required=True)
    parser.add_argument("--scratch-root", type=Path, required=True)
    parser.add_argument("--local-source-root", type=Path)
    parser.add_argument("--ptv3-config", type=Path)
    args = parser.parse_args()
    if args.ptv3_config is not None:
        validate_ptv3_config_pins(args.ptv3_config)
    inventory = stage_source_inventory(
        load_source_inventory(args.manifest),
        durable_root=args.durable_root,
        scratch_root=args.scratch_root,
        local_source_root=args.local_source_root,
    )
    assert inventory.staged_root is not None
    print(inventory.staged_root / "SOURCE_INVENTORY.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
