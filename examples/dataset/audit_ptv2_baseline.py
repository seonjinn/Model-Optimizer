# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Audit the exact prior 1.3M PTV2 corpus and emit its UUID exclusion proof."""

from __future__ import annotations

import argparse
import json
import os
import re
import stat
from collections import Counter
from dataclasses import asdict, dataclass, replace
from hashlib import sha256
from pathlib import Path
from typing import TYPE_CHECKING, cast

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

from specdec_corpus_contracts import SourceFile, canonical_json, sha256_bytes, sha256_canonical_json
from specdec_identity import prompt_uuid

__all__ = [
    "CANONICAL_BASELINE_AUDIT_FILE_SHA256",
    "CANONICAL_BASELINE_OCCURRENCE_PROMPT_IDS_SHA256",
    "EXPECTED_BASELINE",
    "AuditError",
    "BaselineAudit",
    "BaselineExpectation",
    "audit_baseline",
    "load_baseline_audit_receipt",
]

CANONICAL_BASELINE_AUDIT_FILE_SHA256 = (
    "2469430c144d9b86850901df0a28cb810b437555ee894771b796b1386d1c18b5"
)
CANONICAL_BASELINE_OCCURRENCE_PROMPT_IDS_SHA256 = (
    "863470b22925d74228d31b1c2433d9461d25e8d02a298cb3a08a6bc99be55060"
)


class AuditError(ValueError):
    pass


@dataclass(frozen=True)
class BaselineExpectation:
    source_revision: str
    shard_count: int
    split_rows: dict[str, int]
    unique_prompt_count: int | None = None
    physical_row_count: int | None = None
    excluded_tail_rows: int | None = None
    excluded_tail_split: str | None = None
    source_manifest_required: bool = False
    source_manifest_sha256: str | None = None


@dataclass(frozen=True)
class SelectionBoundary:
    file: str
    rows_selected: int
    rows_available: int
    excluded_tail_rows: int


@dataclass(frozen=True)
class BaselineAudit:
    source_revision: str
    source_manifest_sha256: str | None
    row_count: int
    split_rows: dict[str, int]
    files: tuple[SourceFile, ...]
    occurrence_count: int
    unique_prompt_count: int
    occurrence_prompt_ids: Sequence[str]
    occurrence_prompt_ids_sha256: str
    exclusion_prompt_ids: Sequence[str]
    exclusion_prompt_ids_sha256: str
    duplicate_uuid_multiplicity: dict[str, int]
    physical_row_count: int
    selection_policy: str
    selection_boundary: SelectionBoundary


EXPECTED_BASELINE = BaselineExpectation(
    "5c89e01dd720ae0f4058445ed49c5fb68a03c76e",
    26,
    {"chat": 627720, "code": 175000, "math": 239467, "multilingual_de": 257813},
    931363,
    1309377,
    9377,
    "multilingual_de",
    True,
)

_MAX_BASELINE_AUDIT_BYTES = 512 * 1024 * 1024
_SELECTION_POLICY = "hf-streaming-sorted-parquet-take"
_BASELINE_AUDIT_KEYS = frozenset(BaselineAudit.__dataclass_fields__)
_SOURCE_FILE_KEYS = frozenset(SourceFile.__dataclass_fields__)
_SELECTION_BOUNDARY_KEYS = frozenset(SelectionBoundary.__dataclass_fields__)


def _stat_identity(value: os.stat_result) -> tuple[int, int, int, int, int]:
    return value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns, value.st_ctime_ns


def _stable_single_link_regular_bytes(path: Path, *, max_bytes: int) -> bytes:
    if max_bytes < 0:
        raise ValueError("baseline audit retention limit is invalid")
    absolute_parent = path.parent.absolute()
    absolute_path = absolute_parent / path.name
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0)
    try:
        parent_descriptor = os.open(absolute_parent, flags | getattr(os, "O_DIRECTORY", 0))
    except OSError as error:
        raise AuditError(f"baseline audit parent is unreadable: {absolute_parent}") from error
    try:
        parent_before = os.fstat(parent_descriptor)
        try:
            named_before = os.stat(path.name, dir_fd=parent_descriptor, follow_symlinks=False)
            descriptor = os.open(path.name, flags, dir_fd=parent_descriptor)
        except OSError as error:
            raise AuditError(f"baseline audit is unreadable: {path}") from error
        try:
            before = os.fstat(descriptor)
            if (
                not stat.S_ISREG(before.st_mode)
                or before.st_nlink != 1
                or (named_before.st_dev, named_before.st_ino) != (before.st_dev, before.st_ino)
            ):
                raise AuditError(f"baseline audit is not a single-link regular file: {path}")
            if before.st_size > max_bytes:
                raise AuditError(f"baseline audit is too large to retain: {path}")
            payload = bytearray()
            while block := os.read(descriptor, min(8 * 1024 * 1024, max_bytes - len(payload) + 1)):
                payload.extend(block)
                if len(payload) > max_bytes:
                    raise AuditError(f"baseline audit is too large to retain: {path}")
            after = os.fstat(descriptor)
            try:
                named_after = os.stat(path.name, dir_fd=parent_descriptor, follow_symlinks=False)
                absolute_named = os.stat(absolute_path, follow_symlinks=False)
                absolute_parent_after = os.stat(absolute_parent, follow_symlinks=False)
            except OSError as error:
                raise AuditError(
                    f"baseline audit pathname changed while reading: {path}"
                ) from error
            parent_after = os.fstat(parent_descriptor)
            if (
                _stat_identity(before) != _stat_identity(after)
                or len(payload) != before.st_size
                or after.st_nlink != 1
                or named_after.st_nlink != 1
                or absolute_named.st_nlink != 1
                or (after.st_dev, after.st_ino) != (named_after.st_dev, named_after.st_ino)
                or (after.st_dev, after.st_ino) != (absolute_named.st_dev, absolute_named.st_ino)
                or _stat_identity(parent_before) != _stat_identity(parent_after)
                or (parent_before.st_dev, parent_before.st_ino)
                != (absolute_parent_after.st_dev, absolute_parent_after.st_ino)
            ):
                raise AuditError(f"baseline audit changed or rebound while reading: {path}")
            return bytes(payload)
        finally:
            os.close(descriptor)
    finally:
        os.close(parent_descriptor)


def _is_lower_hex(value: object, length: int = 64) -> bool:
    return (
        isinstance(value, str)
        and len(value) == length
        and all(character in "0123456789abcdef" for character in value)
    )


def _is_nonnegative_int(value: object) -> bool:
    return type(value) is int and value >= 0


def _exact_mapping(value: object, keys: frozenset[str], label: str) -> Mapping[str, object]:
    if not isinstance(value, dict) or set(value) != keys:
        raise AuditError(f"baseline audit {label} key set is invalid")
    return value


def _baseline_audit_from_payload(payload: object) -> BaselineAudit:
    record = _exact_mapping(payload, _BASELINE_AUDIT_KEYS, "top-level")
    files_value = record["files"]
    boundary_value = record["selection_boundary"]
    if not isinstance(files_value, list):
        raise AuditError("baseline audit files must be a list")
    files: list[SourceFile] = []
    for value in files_value:
        source = _exact_mapping(value, _SOURCE_FILE_KEYS, "source file")
        if (
            not isinstance(source["path"], str)
            or not source["path"]
            or type(source["bytes"]) is not int
            or source["bytes"] < 0
            or not _is_lower_hex(source["sha256"])
        ):
            raise AuditError("baseline audit source file record is invalid")
        files.append(
            SourceFile(
                cast("str", source["path"]),
                cast("int", source["bytes"]),
                cast("str", source["sha256"]),
            )
        )
    boundary = _exact_mapping(boundary_value, _SELECTION_BOUNDARY_KEYS, "selection boundary")
    if (
        not isinstance(boundary["file"], str)
        or not boundary["file"]
        or any(type(boundary[key]) is not int for key in _SELECTION_BOUNDARY_KEYS - {"file"})
    ):
        raise AuditError("baseline audit selection boundary is invalid")
    split_rows = record["split_rows"]
    duplicate_multiplicity = record["duplicate_uuid_multiplicity"]
    occurrences = record["occurrence_prompt_ids"]
    exclusions = record["exclusion_prompt_ids"]
    if (
        not isinstance(split_rows, dict)
        or any(
            not isinstance(key, str) or not key or type(value) is not int or value < 0
            for key, value in split_rows.items()
        )
        or not isinstance(duplicate_multiplicity, dict)
        or any(
            not _is_lower_hex(key) or type(value) is not int or value < 2
            for key, value in duplicate_multiplicity.items()
        )
        or not isinstance(occurrences, list)
        or not isinstance(exclusions, list)
    ):
        raise AuditError("baseline audit collection schema is invalid")
    integer_fields = (
        "row_count",
        "occurrence_count",
        "unique_prompt_count",
        "physical_row_count",
    )
    if (
        not isinstance(record["source_revision"], str)
        or (
            record["source_manifest_sha256"] is not None
            and not _is_lower_hex(record["source_manifest_sha256"])
        )
        or any(not _is_nonnegative_int(record[key]) for key in integer_fields)
        or not _is_lower_hex(record["occurrence_prompt_ids_sha256"])
        or not _is_lower_hex(record["exclusion_prompt_ids_sha256"])
        or not isinstance(record["selection_policy"], str)
    ):
        raise AuditError("baseline audit scalar schema is invalid")
    return BaselineAudit(
        source_revision=cast("str", record["source_revision"]),
        source_manifest_sha256=cast("str | None", record["source_manifest_sha256"]),
        row_count=cast("int", record["row_count"]),
        split_rows=dict(split_rows),
        files=tuple(files),
        occurrence_count=cast("int", record["occurrence_count"]),
        unique_prompt_count=cast("int", record["unique_prompt_count"]),
        occurrence_prompt_ids=tuple(occurrences),
        occurrence_prompt_ids_sha256=cast("str", record["occurrence_prompt_ids_sha256"]),
        exclusion_prompt_ids=tuple(exclusions),
        exclusion_prompt_ids_sha256=cast("str", record["exclusion_prompt_ids_sha256"]),
        duplicate_uuid_multiplicity=dict(duplicate_multiplicity),
        physical_row_count=cast("int", record["physical_row_count"]),
        selection_policy=cast("str", record["selection_policy"]),
        selection_boundary=SelectionBoundary(
            cast("str", boundary["file"]),
            cast("int", boundary["rows_selected"]),
            cast("int", boundary["rows_available"]),
            cast("int", boundary["excluded_tail_rows"]),
        ),
    )


def _validate_baseline_audit(audit: BaselineAudit, expected: BaselineExpectation) -> None:
    occurrences = list(audit.occurrence_prompt_ids)
    exclusions = list(audit.exclusion_prompt_ids)
    if audit.source_revision != expected.source_revision:
        raise AuditError("baseline audit source revision mismatch")
    if (expected.source_manifest_required and audit.source_manifest_sha256 is None) or (
        expected.source_manifest_sha256 is not None
        and audit.source_manifest_sha256 != expected.source_manifest_sha256
    ):
        raise AuditError("baseline audit source manifest identity mismatch")
    if len(audit.files) != expected.shard_count:
        raise AuditError("baseline audit shard count mismatch")
    if audit.split_rows != dict(sorted(expected.split_rows.items())):
        raise AuditError("baseline audit split row totals mismatch")
    if (
        audit.row_count != sum(audit.split_rows.values())
        or audit.occurrence_count != audit.row_count
        or len(occurrences) != audit.occurrence_count
        or any(not _is_lower_hex(value) for value in occurrences)
    ):
        raise AuditError("baseline audit occurrence count mismatch")
    expected_exclusions = sorted(set(occurrences))
    if (
        exclusions != expected_exclusions
        or audit.unique_prompt_count != len(expected_exclusions)
        or any(not _is_lower_hex(value) for value in exclusions)
    ):
        raise AuditError("baseline audit exclusion prompt UUIDs do not reconcile")
    if (
        expected.unique_prompt_count is not None
        and audit.unique_prompt_count != expected.unique_prompt_count
    ):
        raise AuditError("baseline audit unique prompt count mismatch")
    counts = Counter(occurrences)
    multiplicity = {prompt_id: count for prompt_id, count in sorted(counts.items()) if count > 1}
    if audit.duplicate_uuid_multiplicity != multiplicity:
        raise AuditError("baseline audit duplicate UUID multiplicity mismatch")
    if audit.occurrence_prompt_ids_sha256 != sha256_bytes(canonical_json(occurrences)):
        raise AuditError("baseline audit occurrence prompt UUID digest mismatch")
    if (
        expected is EXPECTED_BASELINE
        and audit.occurrence_prompt_ids_sha256 != CANONICAL_BASELINE_OCCURRENCE_PROMPT_IDS_SHA256
    ):
        raise AuditError("canonical baseline occurrence prompt UUID digest mismatch")
    if audit.exclusion_prompt_ids_sha256 != sha256_bytes(canonical_json(exclusions)):
        raise AuditError("baseline audit exclusion prompt UUID digest mismatch")
    if (
        expected.physical_row_count is not None
        and audit.physical_row_count != expected.physical_row_count
    ):
        raise AuditError("baseline audit physical row count mismatch")
    boundary = audit.selection_boundary
    boundary_split = re.sub(r"-\d+(?:-of-\d+)?\.parquet$", "", boundary.file)
    if (
        audit.selection_policy != _SELECTION_POLICY
        or boundary.rows_selected < 1
        or boundary.rows_available < boundary.rows_selected
        or boundary.excluded_tail_rows != boundary.rows_available - boundary.rows_selected
        or audit.physical_row_count != audit.occurrence_count + boundary.excluded_tail_rows
    ):
        raise AuditError("baseline audit selection policy or boundary mismatch")
    if (
        expected.excluded_tail_rows is not None
        and boundary.excluded_tail_rows != expected.excluded_tail_rows
    ) or (
        expected.excluded_tail_split is not None and boundary_split != expected.excluded_tail_split
    ):
        raise AuditError("baseline audit expected selection boundary mismatch")


def load_baseline_audit_receipt(
    path: Path,
    expected_file_sha256: str,
    expected: BaselineExpectation = EXPECTED_BASELINE,
) -> tuple[BaselineAudit, str]:
    """Authenticate and replay one original canonical BaselineAudit receipt."""
    if (
        expected is EXPECTED_BASELINE
        and expected_file_sha256 != CANONICAL_BASELINE_AUDIT_FILE_SHA256
    ):
        raise AuditError("canonical baseline whole-file lineage SHA-256 mismatch")
    raw = _stable_single_link_regular_bytes(path, max_bytes=_MAX_BASELINE_AUDIT_BYTES)
    if sha256(raw).hexdigest() != expected_file_sha256:
        raise AuditError("baseline audit whole-file SHA-256 mismatch")
    try:
        decoded = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise AuditError("baseline audit JSON is invalid") from error
    if not isinstance(decoded, dict) or raw != canonical_json(decoded) + b"\n":
        raise AuditError("baseline audit must use canonical JSON bytes")
    payload = dict(decoded)
    receipt_sha256 = payload.pop("receipt_sha256", None)
    if receipt_sha256 != sha256_bytes(canonical_json(payload)):
        raise AuditError("baseline audit self-hash does not reconcile")
    audit = _baseline_audit_from_payload(payload)
    _validate_baseline_audit(audit, expected)
    return audit, expected_file_sha256


def _sha256_file(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _decode(value: object) -> object:
    if isinstance(value, str):
        return json.loads(value)
    return value


def _validate_source_manifest(
    path: Path | None,
    expected: BaselineExpectation,
    files: list[tuple[str, SourceFile]],
) -> str | None:
    if path is None:
        if expected.source_manifest_required:
            raise AuditError("source manifest is required")
        return None
    try:
        payload = json.loads(path.read_bytes())
    except (OSError, json.JSONDecodeError) as error:
        raise AuditError(f"invalid source manifest: {path}") from error
    if not isinstance(payload, dict) or payload.get("source_revision") != expected.source_revision:
        raise AuditError("source manifest revision mismatch")
    records = payload.get("files")
    if not isinstance(records, list):
        raise AuditError("source manifest files must be a list")
    expected_files = {
        name: {"bytes": record.bytes, "sha256": record.sha256} for name, record in files
    }
    manifest_files: dict[str, dict[str, object]] = {}
    for item in records:
        if not isinstance(item, dict) or not isinstance(item.get("name"), str):
            raise AuditError("source manifest file record is invalid")
        name = item["name"]
        if name in manifest_files:
            raise AuditError(f"source manifest duplicates file: {name}")
        manifest_files[name] = item
    if set(manifest_files) != set(expected_files):
        raise AuditError("source manifest file inventory mismatch")
    for name, actual in expected_files.items():
        manifest = manifest_files[name]
        if manifest.get("bytes") != actual["bytes"] or manifest.get("sha256") != actual["sha256"]:
            raise AuditError(f"source manifest file mismatch: {name}")
    actual_sha256 = sha256_bytes(path.read_bytes())
    if expected.source_manifest_required and expected.source_manifest_sha256 is None:
        raise AuditError("expected source manifest SHA-256 is required")
    if (
        expected.source_manifest_sha256 is not None
        and actual_sha256 != expected.source_manifest_sha256
    ):
        raise AuditError(
            "source manifest SHA-256 mismatch: "
            f"{actual_sha256} != {expected.source_manifest_sha256}"
        )
    return actual_sha256


def audit_baseline(
    root: Path,
    expected: BaselineExpectation = EXPECTED_BASELINE,
    *,
    source_manifest: Path | None = None,
) -> BaselineAudit:
    """Verify the selected stream and reconstruct occurrence and exclusion identities."""
    import pyarrow.parquet as pq

    links = sorted(path for path in root.iterdir() if path.is_symlink())
    if len(links) != expected.shard_count:
        raise AuditError(f"shard count mismatch: {len(links)} != {expected.shard_count}")
    named_files: list[tuple[str, SourceFile]] = []
    splits: Counter[str] = Counter()
    uuid_counts: Counter[str] = Counter()
    occurrence_prompt_ids: list[str] = []
    selected_total = 0
    physical_total = 0
    target_total = sum(expected.split_rows.values())
    boundary: SelectionBoundary | None = None
    for link in links:
        try:
            resolved = link.resolve(strict=True)
        except FileNotFoundError as error:
            raise AuditError(f"broken symlink: {link.name}") from error
        named_files.append(
            (link.name, SourceFile(str(resolved), resolved.stat().st_size, _sha256_file(resolved)))
        )
        split = re.sub(r"-\d+(?:-of-\d+)?\.parquet$", "", link.name)
        parquet = pq.ParquetFile(resolved)
        available_rows = parquet.metadata.num_rows
        physical_total += available_rows
        selected_rows = min(available_rows, max(target_total - selected_total, 0))
        if selected_rows == 0:
            continue
        selected_total += selected_rows
        splits[split] += selected_rows
        boundary = SelectionBoundary(
            link.name, selected_rows, available_rows, available_rows - selected_rows
        )
        names = set(parquet.schema_arrow.names)
        if "messages" not in names and "conversations" not in names:
            raise AuditError(f"unsupported Parquet schema: {sorted(names)}")
        message_column = "messages" if "messages" in names else "conversations"
        columns = [message_column] + (["tools"] if "tools" in names else [])
        remaining_rows = selected_rows
        for batch in parquet.iter_batches(columns=columns, batch_size=8192):
            rows = batch.to_pylist()[:remaining_rows]
            for row in rows:
                tools = _decode(row.get("tools")) if row.get("tools") is not None else None
                uuid = prompt_uuid(_decode(row[message_column]), tools)
                occurrence_prompt_ids.append(uuid)
                uuid_counts[uuid] += 1
            remaining_rows -= len(rows)
            if remaining_rows == 0:
                break
    actual = dict(sorted(splits.items()))
    source_manifest_sha256 = _validate_source_manifest(source_manifest, expected, named_files)
    wanted = {key: value for key, value in sorted(expected.split_rows.items()) if value}
    if actual != wanted:
        raise AuditError(f"histogram mismatch: {actual} != {wanted}")
    if boundary is None or selected_total != target_total:
        raise AuditError("sorted stream is shorter than sample_size")
    if expected.physical_row_count is not None and physical_total != expected.physical_row_count:
        raise AuditError(
            f"physical row count mismatch: {physical_total} != {expected.physical_row_count}"
        )
    boundary_split = re.sub(r"-\d+(?:-of-\d+)?\.parquet$", "", boundary.file)
    if (
        expected.excluded_tail_rows is not None
        and boundary.excluded_tail_rows != expected.excluded_tail_rows
    ):
        raise AuditError(
            "excluded tail count mismatch: "
            f"{boundary.excluded_tail_rows} != {expected.excluded_tail_rows}"
        )
    if expected.excluded_tail_split is not None and boundary_split != expected.excluded_tail_split:
        raise AuditError(
            f"excluded tail split mismatch: {boundary_split} != {expected.excluded_tail_split}"
        )
    exclusion_prompt_ids = tuple(sorted(uuid_counts))
    duplicates = {uuid: count for uuid, count in sorted(uuid_counts.items()) if count > 1}
    if len(occurrence_prompt_ids) != sum(actual.values()):
        raise AuditError("UUID occurrence count mismatch")
    if len(exclusion_prompt_ids) + sum(count - 1 for count in duplicates.values()) != len(
        occurrence_prompt_ids
    ):
        raise AuditError("UUID occurrence reconciliation mismatch")
    if (
        expected.unique_prompt_count is not None
        and len(exclusion_prompt_ids) != expected.unique_prompt_count
    ):
        raise AuditError(
            "unique prompt count mismatch: "
            f"{len(exclusion_prompt_ids)} != {expected.unique_prompt_count}"
        )
    return BaselineAudit(
        expected.source_revision,
        source_manifest_sha256,
        sum(actual.values()),
        actual,
        tuple(record for _, record in named_files),
        len(occurrence_prompt_ids),
        len(exclusion_prompt_ids),
        tuple(occurrence_prompt_ids),
        sha256_canonical_json(occurrence_prompt_ids),
        exclusion_prompt_ids,
        sha256_canonical_json(exclusion_prompt_ids),
        duplicates,
        physical_total,
        "hf-streaming-sorted-parquet-take",
        boundary,
    )


def write_audit_receipt(audit: BaselineAudit, output: Path) -> None:
    """Publish a receipt exactly once without replacing a prior file."""
    if output.exists():
        raise FileExistsError(f"immutable audit receipt already exists: {output}")
    payload = asdict(audit)
    payload["receipt_sha256"] = sha256_bytes(canonical_json(payload))
    output.parent.mkdir(parents=True, exist_ok=True)
    partial = output.with_name(f".{output.name}.partial-{os.getpid()}")
    try:
        with partial.open("xb") as destination:
            destination.write(canonical_json(payload) + b"\n")
            destination.flush()
            os.fsync(destination.fileno())
        os.link(partial, output)
        directory_fd = os.open(output.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    except BaseException:
        partial.unlink(missing_ok=True)
        raise
    partial.unlink()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--source-manifest-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if re.fullmatch(r"[0-9a-f]{64}", args.source_manifest_sha256) is None:
        raise AuditError("source manifest SHA-256 must be lowercase hex")
    audit = audit_baseline(
        args.root,
        replace(EXPECTED_BASELINE, source_manifest_sha256=args.source_manifest_sha256),
        source_manifest=args.source_manifest,
    )
    write_audit_receipt(audit, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
