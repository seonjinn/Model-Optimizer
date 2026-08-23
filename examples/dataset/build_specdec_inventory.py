# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Build a hash-bound, pretokenized SpecDec inventory from synthesis or trace shards."""

import argparse
import hashlib
import json
import os
import re
import shutil
import sqlite3
import tempfile
import weakref
from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, overload

import yaml
from specdec_corpus_contracts import CanonicalPrompt, canonical_json, sha256_bytes
from specdec_identity import UUIDCollisionError, canonicalize_prompt
from stage_ptv23_sources import SourceFile, SourceIdentity, SourceInventory
from trajectory_schema import TrajectoryValidationError, validate_trajectory

__all__ = [
    "CandidateCell",
    "CandidateInventory",
    "CandidatePrompt",
    "DiskBackedCandidateRows",
    "InventorySource",
    "SourceFile",
    "SourceIdentity",
    "SourceInventory",
    "UUIDCollisionError",
    "build_candidate_inventory",
    "build_inventory_rows",
    "candidate_inventory_bytes",
    "iter_candidate_inventory_bytes",
    "sha256_file",
    "tokenizer_snapshot_sha256",
]

_SHA = re.compile(r"^[0-9a-f]{40}(?:[0-9a-f]{24})?$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_REVISION = re.compile(r"^[0-9a-f]{40}$")
_LANGUAGE_ALIASES = {
    "english": "en",
    "french": "fr",
    "german": "de",
    "italian": "it",
    "japanese": "ja",
    "spanish": "es",
}
_LANGUAGE_CODES = frozenset({*_LANGUAGE_ALIASES.values(), "zh"})
_APPROVED_SOURCE_LANGUAGES = {
    ("nvidia/Nemotron-SFT-SWE-v2", "agentless"): "en",
    ("nvidia/Nemotron-SFT-SWE-v2", "openhands_swe"): "en",
    ("nvidia/Nemotron-SWE-v1", "r2e_gym"): "en",
    ("nvidia/Nemotron-Agentic-v1", "interactive_agent"): "en",
    ("nvidia/Nemotron-Agentic-v1", "tool_calling"): "en",
    ("nvidia/Nemotron-SFT-Competitive-Programming-v2", "exercism"): "en",
    ("nvidia/Nemotron-Math-v2", "low"): "en",
    ("nvidia/Nemotron-Science-v1", "RQA"): "en",
    ("nvidia/Nemotron-SFT-Instruction-Following-Chat-v2", "reasoning_off"): "en",
    ("nvidia/Nemotron-SFT-Multilingual-v1", "stem_zh"): "zh",
}
_CANDIDATE_QUARANTINE_CODES = frozenset(
    {
        "context_too_long",
        "invalid_language",
        "invalid_row",
        "invalid_tokenization",
        "invalid_tools",
        "missing_messages",
        "wrong_lane",
    }
)


@dataclass(frozen=True)
class CandidateCell:
    """One capacity cell after canonical exclusion and replay validation."""

    arm_domain: str
    lane: str
    language: str
    context_bucket: str


@dataclass(frozen=True)
class CandidatePrompt(CanonicalPrompt):
    """A canonical prompt plus deterministic selection and tokenization metadata."""

    lane: str
    context_bucket: str
    full_token_count: int
    source_manifest_sha256: str
    source_file_path: str
    input_ids: tuple[int, ...]
    tokenizer_sha256: str
    replay_valid: bool

    @property
    def arm_domain(self) -> str:
        return self.domain


_CANDIDATE_COLUMNS = (
    "prompt_uuid",
    "canonical_bytes",
    "source_id",
    "source_revision",
    "source_file_sha256",
    "source_row_index",
    "domain",
    "language",
    "lane",
    "context_bucket",
    "full_token_count",
    "source_manifest_sha256",
    "source_file_path",
    "input_ids",
    "tokenizer_sha256",
    "replay_valid",
)


class DiskBackedCandidateRows(Sequence[CandidatePrompt]):
    """A deterministic, lazy prompt sequence backed by an ordered SQLite index."""

    resident_row_count = 0

    def __init__(
        self,
        storage_path: Path,
        count: int,
        source_manifest_sha256: str,
        tokenizer_sha256: str,
        storage_root: Path,
    ) -> None:
        self.storage_path = storage_path
        self._count = count
        self.source_manifest_sha256 = source_manifest_sha256
        self.tokenizer_sha256 = tokenizer_sha256
        self._closed = False
        self._finalizer = weakref.finalize(self, shutil.rmtree, storage_root, True)

    def close(self) -> None:
        """Release and remove the temporary storage owned by this sequence."""
        if self._closed:
            return
        self._closed = True
        self._finalizer()

    def _require_open(self) -> None:
        if self._closed:
            raise RuntimeError("candidate inventory is closed")

    def __len__(self) -> int:
        return self._count

    @staticmethod
    def _candidate(record: tuple[Any, ...]) -> CandidatePrompt:
        values = dict(zip(_CANDIDATE_COLUMNS, record))
        return CandidatePrompt(
            prompt_uuid=values["prompt_uuid"],
            canonical_bytes=values["canonical_bytes"],
            source_id=values["source_id"],
            source_revision=values["source_revision"],
            source_file_sha256=values["source_file_sha256"],
            source_row_index=values["source_row_index"],
            domain=values["domain"],
            language=values["language"],
            lane=values["lane"],
            context_bucket=values["context_bucket"],
            full_token_count=values["full_token_count"],
            source_manifest_sha256=values["source_manifest_sha256"],
            source_file_path=values["source_file_path"],
            input_ids=tuple(json.loads(values["input_ids"])),
            tokenizer_sha256=values["tokenizer_sha256"],
            replay_valid=bool(values["replay_valid"]),
        )

    def __iter__(self):
        self._require_open()
        query = f"SELECT {', '.join(_CANDIDATE_COLUMNS)} FROM candidates ORDER BY prompt_uuid"
        with sqlite3.connect(self.storage_path) as connection:
            for record in connection.execute(query):
                yield self._candidate(record)

    def _at(self, index: int) -> CandidatePrompt:
        self._require_open()
        if index < 0:
            index += self._count
        if index < 0 or index >= self._count:
            raise IndexError(index)
        query = (
            f"SELECT {', '.join(_CANDIDATE_COLUMNS)} FROM candidates "
            "ORDER BY prompt_uuid LIMIT 1 OFFSET ?"
        )
        with sqlite3.connect(self.storage_path) as connection:
            record = connection.execute(query, (index,)).fetchone()
        if record is None:
            raise IndexError(index)
        return self._candidate(record)

    @overload
    def __getitem__(self, index: int) -> CandidatePrompt: ...

    @overload
    def __getitem__(self, index: slice) -> tuple[CandidatePrompt, ...]: ...

    def __getitem__(self, index: int | slice) -> CandidatePrompt | tuple[CandidatePrompt, ...]:
        if isinstance(index, slice):
            start, stop, step = index.indices(self._count)
            return tuple(self._at(position) for position in range(start, stop, step))
        return self._at(index)


@dataclass(frozen=True, eq=False)
class CandidateInventory:
    """Canonical candidates, usable capacity, and stable rejection receipts."""

    rows: Sequence[CanonicalPrompt]
    capacity: Mapping[CandidateCell, int]
    quarantine_counts: Mapping[str, int]
    inventory_sha256: str

    def close(self) -> None:
        """Release temporary row storage after all sequence consumers finish."""
        if isinstance(self.rows, DiskBackedCandidateRows):
            self.rows.close()

    def __enter__(self) -> "CandidateInventory":
        if isinstance(self.rows, DiskBackedCandidateRows):
            self.rows._require_open()
        return self

    def __exit__(self, exc_type: object, exc_value: object, traceback: object) -> None:
        self.close()

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, CandidateInventory):
            return NotImplemented
        return (
            self.inventory_sha256 == other.inventory_sha256
            and self.capacity == other.capacity
            and self.quarantine_counts == other.quarantine_counts
        )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tokenizer_snapshot_sha256(root: Path) -> str:
    """Hash tokenizer serialization files without hashing model weights."""
    names = {
        "added_tokens.json",
        "chat_template.jinja",
        "merges.txt",
        "special_tokens_map.json",
        "tokenizer.json",
        "tokenizer.model",
        "tokenizer_config.json",
        "vocab.json",
        "vocab.txt",
    }
    files = sorted(path for path in root.iterdir() if path.is_file() and path.name in names)
    if not files:
        raise ValueError("tokenizer snapshot contains no serialization files")
    digest = hashlib.sha256()
    for path in files:
        digest.update(path.name.encode())
        digest.update(b"\0")
        digest.update(bytes.fromhex(sha256_file(path)))
    return digest.hexdigest()


@dataclass(frozen=True)
class InventorySource:
    """One immutable raw/synthesis or recorded-trace input lane."""

    source_id: str
    source_revision: str
    license: str
    pool: str
    category: str
    response_source: str
    tool_lane: str
    manifest_path: Path

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "manifest_path", Path(self.manifest_path).expanduser().resolve(strict=False)
        )
        if not self.source_id or not self.license or not self.category:
            raise ValueError("source identity, license, and category are required")
        if _SHA.fullmatch(self.source_revision) is None:
            raise ValueError("source_revision must be a pinned commit digest")
        if self.pool not in {"ptv2", "ptv3"}:
            raise ValueError("inventory pool must be ptv2 or ptv3")
        if self.response_source not in {"target-synth", "trace-replay"}:
            raise ValueError("response_source must be target-synth or trace-replay")
        expected_lane = "none" if self.response_source == "target-synth" else "recorded-trace"
        if self.tool_lane != expected_lane:
            raise ValueError("tool lane does not match the response source")


def _verified_files(manifest_path: Path) -> tuple[str, list[Path]]:
    raw = manifest_path.read_bytes()
    manifest_sha256 = hashlib.sha256(raw).hexdigest()
    payload = json.loads(raw)
    records = payload.get("files") if isinstance(payload, dict) else None
    if (
        not isinstance(payload, dict)
        or payload.get("schema_version") != 1
        or not isinstance(records, list)
        or not records
    ):
        raise ValueError("source manifest must contain a non-empty version-1 file set")
    root = manifest_path.parent.resolve(strict=True)
    files: list[Path] = []
    for record in records:
        relative = record.get("path") if isinstance(record, dict) else None
        if not isinstance(relative, str) or not relative:
            raise ValueError("source manifest has an invalid file record")
        path = (manifest_path.parent / relative).resolve(strict=True)
        if not path.is_relative_to(root) or not path.is_file():
            raise ValueError(f"source file escapes manifest root: {relative}")
        if path.stat().st_size != record.get("bytes") or sha256_file(path) != record.get("sha256"):
            raise ValueError(f"source file identity mismatch: {relative}")
        files.append(path)
    return manifest_sha256, files


def _iter_rows(path: Path):
    if path.suffix == ".parquet":
        import pyarrow.parquet as pq  # pyright: ignore[reportMissingImports]

        for row in pq.read_table(path).to_pylist():
            yield json.loads(row["raw_json"]) if set(row) == {"raw_json"} else row
        return
    with path.open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            if line.strip():
                try:
                    yield json.loads(line)
                except json.JSONDecodeError as error:
                    raise ValueError(f"invalid JSON row: {path}:{line_number}") from error


def _iter_candidate_rows(path: Path):
    if path.suffix == ".parquet":
        import pyarrow.parquet as pq  # pyright: ignore[reportMissingImports]

        row_index = 0
        for batch in pq.ParquetFile(path).iter_batches(batch_size=1_024):
            for raw_row in batch.to_pylist():
                value: Any = raw_row
                if isinstance(raw_row, dict) and set(raw_row) == {"raw_json"}:
                    encoded = raw_row["raw_json"]
                    try:
                        value = json.loads(encoded) if isinstance(encoded, str | bytes) else None
                    except (UnicodeDecodeError, json.JSONDecodeError):
                        value = None
                yield row_index, value if isinstance(value, dict) else None
                row_index += 1
        return
    with path.open("rb") as source:
        row_index = 0
        for raw_line in source:
            if not raw_line.strip():
                continue
            try:
                value = json.loads(raw_line.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError):
                value = None
            yield row_index, value if isinstance(value, dict) else None
            row_index += 1


def _context_bucket(token_count: int) -> str:
    if token_count <= 4096:
        return "le4k"
    if token_count <= 16384:
        return "4k_16k"
    if token_count <= 32768:
        return "16k_32k"
    raise ValueError(f"conversation exceeds the 32K inventory limit: {token_count}")


def _source_key(source: SourceIdentity) -> tuple[str, ...]:
    return (
        source.repository_id,
        source.configuration,
        source.split,
        source.revision,
        source.cell,
        source.lane,
    )


def _staged_path(root: Path, source: SourceIdentity, file: SourceFile) -> Path:
    return root / "sources" / source.repository_id / source.revision / file.path


def _verified_candidate_files(
    inventory: SourceInventory,
) -> list[tuple[SourceIdentity, SourceFile, Path]]:
    if inventory.schema_version != 1 or inventory.staged_root is None:
        raise ValueError("candidate inventory requires a staged version-1 SourceInventory")
    root = inventory.staged_root.resolve(strict=True)
    if not root.is_dir():
        raise ValueError("staged source root is not a directory")
    if (
        _SHA256.fullmatch(inventory.manifest_sha256) is None
        or hashlib.sha256(inventory.canonical_manifest).hexdigest() != inventory.manifest_sha256
    ):
        raise ValueError("source inventory manifest identity mismatch")
    verified: list[tuple[SourceIdentity, SourceFile, Path]] = []
    logical_files: set[tuple[str, str, str]] = set()
    for source in sorted(inventory.sources, key=_source_key):
        if not source.approved_use or _REVISION.fullmatch(source.revision) is None:
            raise ValueError(f"unapproved or unpinned source identity: {source.repository_id}")
        if source.lane not in {
            "target-synth",
            "agentless-swe",
            "interactive-swe-replay",
            "generic-tool-replay",
        }:
            raise ValueError(f"unsupported source lane: {source.lane}")
        for file in sorted(source.files, key=lambda descriptor: descriptor.path):
            key = (source.repository_id, source.revision, file.path)
            if key in logical_files:
                raise ValueError(f"duplicate logical source file: {key}")
            logical_files.add(key)
            path = _staged_path(root, source, file).resolve(strict=False)
            if (
                not path.is_relative_to(root)
                or not path.is_file()
                or path.stat().st_size != file.bytes
                or sha256_file(path) != file.sha256
            ):
                raise ValueError(
                    f"source file identity mismatch: {source.repository_id}:{file.path}"
                )
            verified.append((source, file, path))
    return verified


def _normalize_language(value: Any, source: SourceIdentity) -> str:
    if value is None:
        value = _APPROVED_SOURCE_LANGUAGES.get((source.repository_id, source.split))
    if not isinstance(value, str) or not value.strip():
        raise ValueError("invalid_language")
    normalized = value.strip().lower().replace("_", "-")
    normalized = _LANGUAGE_ALIASES.get(normalized, normalized.split("-", maxsplit=1)[0])
    if normalized not in _LANGUAGE_CODES:
        raise ValueError("invalid_language")
    return normalized


def _row_messages(row: dict[str, Any]) -> list[dict[str, Any]]:
    messages = row.get("messages") or row.get("conversations")
    if (
        not isinstance(messages, list)
        or not messages
        or not all(isinstance(message, dict) for message in messages)
    ):
        raise ValueError("missing_messages")
    return deepcopy(messages)


def _target_prompt(row: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    messages = _row_messages(row)
    tools = deepcopy(row.get("tools") or [])
    if not isinstance(tools, list):
        raise ValueError("invalid_tools")
    if messages[-1].get("role") == "assistant":
        messages.pop()
    if not messages:
        raise ValueError("missing_messages")
    has_tool_exchange = bool(tools) or any(
        message.get("role") == "tool" or message.get("tool_calls") for message in messages
    )
    if has_tool_exchange:
        raise ValueError("wrong_lane")
    return messages, tools


def _candidate_tokenize(
    tokenizer: Any,
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]],
    *,
    add_generation_prompt: bool,
) -> tuple[int, ...]:
    encoded = tokenizer.apply_chat_template(
        messages,
        tools=tools or None,
        tokenize=True,
        add_generation_prompt=add_generation_prompt,
        return_dict=True,
    )
    input_ids = encoded.get("input_ids") if isinstance(encoded, dict) else encoded
    if not isinstance(input_ids, list) or any(
        isinstance(token, bool) or not isinstance(token, int) for token in input_ids
    ):
        raise ValueError("invalid_tokenization")
    return tuple(input_ids)


def _quarantine(counts: dict[str, int], reason: str) -> None:
    counts[reason] = counts.get(reason, 0) + 1


def _create_candidate_database(path: Path) -> sqlite3.Connection:
    connection = sqlite3.connect(path)
    connection.execute("PRAGMA journal_mode=DELETE")
    connection.execute("PRAGMA synchronous=NORMAL")
    connection.execute(
        """
        CREATE TABLE candidates (
            prompt_uuid TEXT PRIMARY KEY,
            canonical_bytes BLOB NOT NULL,
            source_id TEXT NOT NULL,
            source_revision TEXT NOT NULL,
            source_file_sha256 TEXT NOT NULL,
            source_row_index INTEGER NOT NULL,
            domain TEXT NOT NULL,
            language TEXT NOT NULL,
            lane TEXT NOT NULL,
            context_bucket TEXT NOT NULL,
            full_token_count INTEGER NOT NULL,
            source_manifest_sha256 TEXT NOT NULL,
            source_file_path TEXT NOT NULL,
            input_ids TEXT NOT NULL,
            tokenizer_sha256 TEXT NOT NULL,
            replay_valid INTEGER NOT NULL
        ) WITHOUT ROWID
        """
    )
    return connection


def _insert_candidate(connection: sqlite3.Connection, candidate: CandidatePrompt) -> None:
    connection.execute(
        f"INSERT INTO candidates ({', '.join(_CANDIDATE_COLUMNS)}) "
        f"VALUES ({', '.join('?' for _ in _CANDIDATE_COLUMNS)})",
        (
            candidate.prompt_uuid,
            candidate.canonical_bytes,
            candidate.source_id,
            candidate.source_revision,
            candidate.source_file_sha256,
            candidate.source_row_index,
            candidate.domain,
            candidate.language,
            candidate.lane,
            candidate.context_bucket,
            candidate.full_token_count,
            candidate.source_manifest_sha256,
            candidate.source_file_path,
            json.dumps(candidate.input_ids, separators=(",", ":")),
            candidate.tokenizer_sha256,
            int(candidate.replay_valid),
        ),
    )


def _candidate_record(prompt: CandidatePrompt) -> dict[str, Any]:
    return {
        "type": "candidate",
        "prompt_uuid": prompt.prompt_uuid,
        "canonical_prompt": json.loads(prompt.canonical_bytes),
        "source_id": prompt.source_id,
        "source_revision": prompt.source_revision,
        "source_file_sha256": prompt.source_file_sha256,
        "source_row_index": prompt.source_row_index,
        "source_manifest_sha256": prompt.source_manifest_sha256,
        "source_file_path": prompt.source_file_path,
        "arm_domain": prompt.arm_domain,
        "lane": prompt.lane,
        "language": prompt.language,
        "context_bucket": prompt.context_bucket,
        "full_token_count": prompt.full_token_count,
        "input_ids": prompt.input_ids,
        "tokenizer_sha256": prompt.tokenizer_sha256,
        "replay_valid": prompt.replay_valid,
    }


def _capacity_records(capacity: Mapping[CandidateCell, int]) -> list[dict[str, Any]]:
    return [
        {
            "arm_domain": cell.arm_domain,
            "lane": cell.lane,
            "language": cell.language,
            "context_bucket": cell.context_bucket,
            "count": count,
        }
        for cell, count in capacity.items()
    ]


def _identity_chunks(
    rows: DiskBackedCandidateRows,
    capacity: Mapping[CandidateCell, int],
    quarantine_counts: Mapping[str, int],
):
    yield (
        canonical_json(
            {
                "type": "metadata",
                "schema_version": 1,
                "source_manifest_sha256": rows.source_manifest_sha256,
                "tokenizer_sha256": rows.tokenizer_sha256,
            }
        )
        + b"\n"
    )
    for prompt in rows:
        yield canonical_json(_candidate_record(prompt)) + b"\n"
    yield canonical_json({"type": "capacity", "cells": _capacity_records(capacity)}) + b"\n"
    yield canonical_json({"type": "quarantine", "counts": dict(quarantine_counts)}) + b"\n"


def build_candidate_inventory(
    source_inventory: SourceInventory,
    *,
    tokenizer: Any,
    tokenizer_sha256: str,
    historical_prompt_ids: set[str],
    held_out_prompt_ids: set[str],
    training_seq_len: int = 4_096,
    storage_dir: Path | None = None,
) -> CandidateInventory:
    """Build canonical B-prime/C/D candidates from one verified staged inventory."""
    if _SHA256.fullmatch(tokenizer_sha256) is None or training_seq_len < 1:
        raise ValueError("tokenizer digest and training sequence length must be pinned")
    files = _verified_candidate_files(source_inventory)
    capacity: dict[CandidateCell, int] = {}
    quarantine_counts: dict[str, int] = {}
    storage_parent = storage_dir or Path(tempfile.gettempdir())
    storage_parent.mkdir(parents=True, exist_ok=True)
    storage_root = Path(tempfile.mkdtemp(prefix="specdec-candidates-", dir=storage_parent))
    database_path = storage_root / "candidates.sqlite3"
    connection = _create_candidate_database(database_path)
    accepted_count = 0
    try:
        for source, descriptor, path in files:
            source_id = f"{source.repository_id}:{source.configuration}:{source.split}"
            for row_index, raw_row in _iter_candidate_rows(path):
                if raw_row is None:
                    _quarantine(quarantine_counts, "invalid_row")
                    continue
                row = dict(raw_row)
                try:
                    if source.lane in {"interactive-swe-replay", "generic-tool-replay"}:
                        validation = validate_trajectory(
                            row,
                            source_id=f"{source_id}:{row_index}",
                            lane=source.lane,
                            tokenizer=tokenizer,
                            training_seq_len=training_seq_len,
                        )
                        messages = validation.canonical["messages"]
                        tools = validation.canonical["tools"]
                        canonical_bytes = validation.canonical_bytes
                        replay_valid = True
                    else:
                        messages, tools = _target_prompt(row)
                        canonical_bytes = canonicalize_prompt(messages, tools)
                        replay_valid = False
                    canonical_prompt = json.loads(canonical_bytes)
                    messages = canonical_prompt["messages"]
                    tools = canonical_prompt["tools"]
                    uuid = sha256_bytes(canonical_bytes)
                    if uuid in historical_prompt_ids:
                        _quarantine(quarantine_counts, "historical_exclusion")
                        continue
                    if uuid in held_out_prompt_ids:
                        _quarantine(quarantine_counts, "heldout_exclusion")
                        continue
                    previous = connection.execute(
                        "SELECT canonical_bytes FROM candidates WHERE prompt_uuid = ?", (uuid,)
                    ).fetchone()
                    if previous is not None:
                        if previous[0] != canonical_bytes:
                            raise UUIDCollisionError(f"UUID collision: {uuid}")
                        _quarantine(quarantine_counts, "duplicate_prompt_uuid")
                        continue
                    language = _normalize_language(row.get("language"), source)
                    input_ids = _candidate_tokenize(
                        tokenizer,
                        messages,
                        tools,
                        add_generation_prompt=not replay_valid,
                    )
                    context_bucket = _context_bucket(len(input_ids))
                except TrajectoryValidationError as error:
                    _quarantine(quarantine_counts, error.reason)
                    continue
                except ValueError as error:
                    reason = str(error)
                    if reason.startswith("conversation exceeds the 32K inventory limit"):
                        reason = "context_too_long"
                    if reason not in _CANDIDATE_QUARANTINE_CODES:
                        raise
                    _quarantine(quarantine_counts, reason)
                    continue
                candidate = CandidatePrompt(
                    prompt_uuid=uuid,
                    canonical_bytes=canonical_bytes,
                    source_id=source_id,
                    source_revision=source.revision,
                    source_file_sha256=descriptor.sha256,
                    source_row_index=row_index,
                    domain=source.cell,
                    language=language,
                    lane=source.lane,
                    context_bucket=context_bucket,
                    full_token_count=len(input_ids),
                    source_manifest_sha256=source_inventory.manifest_sha256,
                    source_file_path=descriptor.path,
                    input_ids=input_ids,
                    tokenizer_sha256=tokenizer_sha256,
                    replay_valid=replay_valid,
                )
                _insert_candidate(connection, candidate)
                accepted_count += 1
                if accepted_count % 10_000 == 0:
                    connection.commit()
                cell = CandidateCell(source.cell, source.lane, language, context_bucket)
                capacity[cell] = capacity.get(cell, 0) + 1
        connection.commit()
    except BaseException:
        connection.close()
        shutil.rmtree(storage_root)
        raise
    connection.close()
    capacity = dict(
        sorted(
            capacity.items(),
            key=lambda item: (
                item[0].arm_domain,
                item[0].lane,
                item[0].language,
                item[0].context_bucket,
            ),
        )
    )
    quarantine_counts = dict(sorted(quarantine_counts.items()))
    rows = DiskBackedCandidateRows(
        database_path,
        accepted_count,
        source_inventory.manifest_sha256,
        tokenizer_sha256,
        storage_root,
    )
    digest = hashlib.sha256()
    for chunk in _identity_chunks(rows, capacity, quarantine_counts):
        digest.update(chunk)
    return CandidateInventory(
        rows=rows,
        capacity=MappingProxyType(capacity),
        quarantine_counts=MappingProxyType(quarantine_counts),
        inventory_sha256=digest.hexdigest(),
    )


def iter_candidate_inventory_bytes(inventory: CandidateInventory):
    """Yield deterministic inventory bytes without materializing the inventory."""
    if not isinstance(inventory.rows, DiskBackedCandidateRows):
        raise TypeError("candidate inventory rows are not disk-backed")
    yield from _identity_chunks(inventory.rows, inventory.capacity, inventory.quarantine_counts)


def candidate_inventory_bytes(inventory: CandidateInventory) -> bytes:
    """Materialize deterministic bytes for small tests; production callers should iterate."""
    return b"".join(iter_candidate_inventory_bytes(inventory))


def _tokenize(tokenizer: Any, row: dict[str, Any]) -> tuple[list[int], list[int]]:
    messages = row.get("messages") or row.get("conversations")
    if not isinstance(messages, list) or not messages:
        raise ValueError("inventory row has no messages")
    encoded = tokenizer.apply_chat_template(
        messages,
        tools=row.get("tools") or None,
        tokenize=True,
        add_generation_prompt=False,
        return_dict=True,
        return_assistant_tokens_mask=True,
    )
    input_ids = encoded.get("input_ids")
    loss_mask = encoded.get("assistant_masks", encoded.get("assistant_mask"))
    if (
        not isinstance(input_ids, list)
        or not isinstance(loss_mask, list)
        or len(input_ids) != len(loss_mask)
        or any(value not in (0, 1) for value in loss_mask)
    ):
        raise ValueError("tokenizer did not return aligned IDs and assistant mask")
    return input_ids, loss_mask


def build_inventory_rows(
    sources: list[InventorySource],
    *,
    tokenizer: Any,
    tokenizer_sha256: str,
    training_seq_len: int,
) -> list[dict[str, Any]]:
    """Verify and pretokenize all source rows without trusting caller metadata."""
    if _SHA256.fullmatch(tokenizer_sha256) is None or training_seq_len < 1:
        raise ValueError("tokenizer digest and training sequence length must be pinned")
    inventory: list[dict[str, Any]] = []
    seen: set[str] = set()
    for spec in sources:
        manifest_sha256, files = _verified_files(spec.manifest_path)
        for source_file in files:
            source_file_sha256 = sha256_file(source_file)
            for index, raw_row in enumerate(_iter_rows(source_file)):
                row = dict(raw_row)
                prompt_id = str(
                    row.get("prompt_id")
                    or hashlib.sha256(
                        json.dumps(
                            row.get("messages") or row.get("conversations"),
                            ensure_ascii=False,
                            separators=(",", ":"),
                            sort_keys=True,
                        ).encode()
                    ).hexdigest()
                )
                if prompt_id in seen:
                    raise ValueError(f"duplicate prompt_id {prompt_id}")
                seen.add(prompt_id)
                has_tools = bool(row.get("tools")) or any(
                    message.get("role") == "tool" or message.get("tool_calls")
                    for message in row.get("messages", [])
                )
                if has_tools != (spec.tool_lane == "recorded-trace"):
                    raise ValueError(f"prompt {prompt_id} is routed to the wrong tool lane")
                input_ids, full_mask = _tokenize(tokenizer, row)
                full_token_count = len(input_ids)
                training_ids = input_ids[:training_seq_len]
                training_mask = full_mask[:training_seq_len]
                assistant_tokens = sum(training_mask)
                if assistant_tokens < 1:
                    continue
                inventory.append(
                    {
                        "prompt_id": prompt_id,
                        "pool": spec.pool,
                        "category": spec.category,
                        "context_bucket": _context_bucket(full_token_count),
                        "full_token_count": full_token_count,
                        "full_assistant_tokens": sum(full_mask),
                        "assistant_tokens": assistant_tokens,
                        "source_id": spec.source_id,
                        "source_revision": spec.source_revision,
                        "license": spec.license,
                        "source_manifest_sha256": manifest_sha256,
                        "source_file_path": str(source_file.relative_to(spec.manifest_path.parent)),
                        "source_file_sha256": source_file_sha256,
                        "source_row_index": index,
                        "response_source": spec.response_source,
                        "tool_lane": spec.tool_lane,
                        "tokenizer_sha256": tokenizer_sha256,
                        "input_ids": training_ids,
                        "loss_mask": training_mask,
                        "messages": row.get("messages") or row.get("conversations"),
                        "tools": row.get("tools"),
                    }
                )
    return inventory


def _write_jsonl_atomic(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(f".{path.name}.partial-{os.getpid()}")
    with partial.open("w", encoding="utf-8") as output:
        for row in rows:
            output.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        output.flush()
        os.fsync(output.fileno())
    os.replace(partial, path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sources-config", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--tokenizer-sha256", required=True)
    parser.add_argument("--training-seq-len", type=int, default=4096)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    from transformers import AutoTokenizer  # pyright: ignore[reportMissingImports]

    config = yaml.safe_load(args.sources_config.read_text(encoding="utf-8"))
    sources = [
        InventorySource(
            **(record | {"manifest_path": args.sources_config.parent / record["manifest_path"]})
        )
        for record in config["sources"]
    ]
    actual_tokenizer_sha256 = tokenizer_snapshot_sha256(args.tokenizer)
    if actual_tokenizer_sha256 != args.tokenizer_sha256:
        raise ValueError("tokenizer snapshot SHA-256 mismatch")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True)
    rows = build_inventory_rows(
        sources,
        tokenizer=tokenizer,
        tokenizer_sha256=args.tokenizer_sha256,
        training_seq_len=args.training_seq_len,
    )
    _write_jsonl_atomic(args.output, rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
