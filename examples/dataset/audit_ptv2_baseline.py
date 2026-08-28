# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Audit the exact prior 1.3M PTV2 corpus and emit its UUID exclusion proof."""

from __future__ import annotations

import argparse
import json
import os
import re
from collections import Counter
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Sequence

from specdec_corpus_contracts import SourceFile, canonical_json, sha256_bytes, sha256_canonical_json
from specdec_identity import prompt_uuid

__all__ = [
    "EXPECTED_BASELINE",
    "AuditError",
    "BaselineAudit",
    "BaselineExpectation",
    "audit_baseline",
]


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
