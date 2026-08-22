# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Audit the exact prior 1.3M PTV2 corpus and emit its UUID exclusion proof."""

from __future__ import annotations

import argparse
import json
import os
import re
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path

from specdec_corpus_contracts import SourceFile, canonical_json, sha256_bytes
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


@dataclass(frozen=True)
class SelectionBoundary:
    file: str
    rows_selected: int
    rows_available: int
    excluded_tail_rows: int


@dataclass(frozen=True)
class BaselineAudit:
    source_revision: str
    row_count: int
    split_rows: dict[str, int]
    files: tuple[SourceFile, ...]
    prompt_uuids: tuple[str, ...]
    duplicate_uuid_multiplicity: dict[str, int]
    physical_row_count: int
    selection_policy: str
    selection_boundary: SelectionBoundary


EXPECTED_BASELINE = BaselineExpectation(
    "5c89e01dd720ae0f4058445ed49c5fb68a03c76e",
    26,
    {"chat": 627720, "code": 175000, "math": 239467, "multilingual_de": 257813},
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


def _prompt_messages(messages: object) -> list[dict[str, object]]:
    if not isinstance(messages, list) or any(not isinstance(item, dict) for item in messages):
        raise AuditError("messages must be a list of mappings")
    prompt = [item for item in messages if item.get("role") in {"system", "developer", "user"}]
    if not prompt:
        raise AuditError("conversation has no prompt-bearing messages")
    return prompt


def audit_baseline(root: Path, expected: BaselineExpectation = EXPECTED_BASELINE) -> BaselineAudit:
    """Verify shard identity/histogram and reconstruct sorted canonical prompt UUIDs."""
    import pyarrow.parquet as pq

    links = sorted(path for path in root.iterdir() if path.is_symlink())
    if len(links) != expected.shard_count:
        raise AuditError(f"shard count mismatch: {len(links)} != {expected.shard_count}")
    files: list[SourceFile] = []
    splits: Counter[str] = Counter()
    uuid_counts: Counter[str] = Counter()
    selected_total = 0
    physical_total = 0
    target_total = sum(expected.split_rows.values())
    boundary: SelectionBoundary | None = None
    for link in links:
        try:
            resolved = link.resolve(strict=True)
        except FileNotFoundError as error:
            raise AuditError(f"broken symlink: {link.name}") from error
        files.append(SourceFile(str(resolved), resolved.stat().st_size, _sha256_file(resolved)))
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
                messages = _prompt_messages(_decode(row[message_column]))
                tools = _decode(row.get("tools")) if row.get("tools") is not None else None
                uuid = prompt_uuid(messages, tools)
                uuid_counts[uuid] += 1
            remaining_rows -= len(rows)
            if remaining_rows == 0:
                break
    actual = dict(sorted(splits.items()))
    wanted = {key: value for key, value in sorted(expected.split_rows.items()) if value}
    if actual != wanted:
        raise AuditError(f"histogram mismatch: {actual} != {wanted}")
    if boundary is None or selected_total != target_total:
        raise AuditError("sorted stream is shorter than sample_size")
    duplicates = {uuid: count for uuid, count in sorted(uuid_counts.items()) if count > 1}
    if len(uuid_counts) + sum(count - 1 for count in duplicates.values()) != sum(actual.values()):
        raise AuditError("UUID occurrence reconciliation mismatch")
    return BaselineAudit(
        expected.source_revision,
        sum(actual.values()),
        actual,
        tuple(files),
        tuple(sorted(uuid_counts)),
        duplicates,
        physical_total,
        "hf-streaming-sorted-parquet-take",
        boundary,
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    audit = audit_baseline(args.root)
    payload = asdict(audit)
    encoded = canonical_json(payload)
    payload["receipt_sha256"] = sha256_bytes(encoded)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    partial = args.output.with_name(f".{args.output.name}.partial-{os.getpid()}")
    partial.write_bytes(canonical_json(payload) + b"\n")
    os.replace(partial, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
