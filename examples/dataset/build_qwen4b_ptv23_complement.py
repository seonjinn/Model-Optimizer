# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Build and verify the authenticated PTV2/PTV3 700K continuation corpus."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sqlite3
import stat
import sys
import tempfile
from collections import Counter
from collections.abc import Iterable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any, BinaryIO, overload

from ptv23_complement_target_policy import (
    ComplementError,
    TargetTokenizerPolicy,
    load_target_policy,
)
from trajectory_schema import TrajectoryValidationError, validate_trajectory

SCHEMA_VERSION = "ptv2-ptv3-complement-config-v1"
Q4_TARGET_POLICY_PATH = Path(__file__).with_name("qwen3_4b_ptv23_target_v1.json")
Q4_TARGET_POLICY_SHA256 = "028830a9bde3f9353809894fc2650ee734c55a7b6860d6ec3ac32f55ff37b4df"
Q4_TARGET_POLICY = load_target_policy(Q4_TARGET_POLICY_PATH, Q4_TARGET_POLICY_SHA256)
SCIENTIFIC_IDENTITY = Q4_TARGET_POLICY.scientific_identity
TRAINING_SEQUENCE_LENGTH = Q4_TARGET_POLICY.training_sequence_length
PTV2_REVISION = "5c89e01dd720ae0f4058445ed49c5fb68a03c76e"
QWEN3_4B_REVISION = Q4_TARGET_POLICY.tokenizer_revision
TOKENIZER_POLICY = {
    "repository": Q4_TARGET_POLICY.tokenizer_repository,
    "revision": QWEN3_4B_REVISION,
    "training_sequence_length": TRAINING_SEQUENCE_LENGTH,
    "trust_schema": Q4_TARGET_POLICY.tokenizer_trust_schema,
}
SOURCE_REQUIREMENTS_POLICY = {
    "path": Q4_TARGET_POLICY.source_requirements_path,
    "sha256": Q4_TARGET_POLICY.source_requirements_sha256,
}
APPROVED_SOURCE_REQUIREMENT_BLOCKERS: dict[str, tuple[str, ...]] = {
    Q4_TARGET_POLICY.source_requirements_sha256: (
        "ptv2_candidate_file_inventory_and_sha256",
        "ptv3_swe_v3_authoritative_revision_file_inventory_and_sha256",
        "source_row_schema_sha256_for_every_file",
        "post_exclusion_post_tokenization_capacity_receipt_for_every_category",
    ),
    "8cb600a2a839148814b0d26a6a80a7f061db5456291cd5d708be4520521d8e6b": (
        "qwen3_30ba3b_thinking_tokenizer_revision_snapshot_template_and_special_tokens",
        "ptv2_candidate_file_inventory_and_sha256",
        "ptv3_swe_v3_authoritative_revision_file_inventory_and_sha256",
        "source_row_schema_sha256_for_every_file",
        "post_exclusion_post_tokenization_capacity_receipt_for_every_category",
    ),
}
APPROVED_QUOTAS = {
    "ptv2_stem": 300_000,
    "ptv2_multilingual_ja": 50_000,
    "ptv2_multilingual_es": 50_000,
    "ptv2_multilingual_fr": 50_000,
    "ptv2_multilingual_it": 50_000,
    "ptv3_swe_v3": 100_000,
    "ptv3_interactive_agentic_swe": 19_000,
    "ptv3_general_tool_trajectories": 81_000,
}
REQUIRED_HELD_OUT_RECEIPT_NAMES = frozenset({"speed", "math", "code", "swe", "tool"})
APPROVED_PTV3_SWE_SOURCE_PINS: tuple[tuple[str, str, str], ...] = ()
APPROVED_ROW_SCHEMA_SHA256S: frozenset[str] = frozenset()
APPROVED_SOURCE_INVENTORY_FILE_SHA256 = ""
APPROVED_HISTORICAL_RECEIPT_FILE_SHA256 = ""
APPROVED_HELD_OUT_RECEIPT_FILE_SHA256S: dict[str, str] = {}


def _require_external_approval_roots() -> None:
    if (
        not APPROVED_PTV3_SWE_SOURCE_PINS
        or not APPROVED_ROW_SCHEMA_SHA256S
        or not APPROVED_SOURCE_INVENTORY_FILE_SHA256
        or not APPROVED_HISTORICAL_RECEIPT_FILE_SHA256
        or set(APPROVED_HELD_OUT_RECEIPT_FILE_SHA256S) != set(REQUIRED_HELD_OUT_RECEIPT_NAMES)
    ):
        raise ComplementError("authoritative external approval root is unresolved")


def _authenticate_provenance_roots(
    inventory: SourceInventory,
    historical: HistoricalExclusion,
    held_out: HeldOutUnion,
) -> None:
    observed_held_out = dict(
        zip(held_out.receipt_names, held_out.receipt_file_sha256s, strict=True)
    )
    if (
        inventory.file_sha256 != APPROVED_SOURCE_INVENTORY_FILE_SHA256
        or historical.file_sha256 != APPROVED_HISTORICAL_RECEIPT_FILE_SHA256
        or observed_held_out != APPROVED_HELD_OUT_RECEIPT_FILE_SHA256S
    ):
        raise ComplementError("provenance receipt is not in the external approval root")


@dataclass(frozen=True)
class SelectedRow:
    """One source-native selected row and its selected-only token evidence."""

    prompt_uuid: str
    category: str
    source_id: str
    source_file_sha256: str
    source_row_index: int
    messages: list[dict[str, Any]]
    tools: list[dict[str, Any]]
    full_tokens: int
    assistant_tokens: int


@dataclass(frozen=True)
class ContinuationSelection:
    """Exact quota selection in category and source occurrence order."""

    rows: Sequence[SelectedRow]
    exclusions: dict[str, int]
    capacity_receipt_file_sha256: str = ""
    capacity_receipt_path: Path | None = None


@dataclass(frozen=True)
class SpoolRows(Sequence[SelectedRow]):
    """Replayable selected rows stored on job-local scratch."""

    path: Path
    row_count: int

    def __len__(self) -> int:
        return self.row_count

    @overload
    def __getitem__(self, index: int) -> SelectedRow: ...

    @overload
    def __getitem__(self, index: slice) -> Sequence[SelectedRow]: ...

    def __getitem__(self, index: int | slice) -> SelectedRow | Sequence[SelectedRow]:
        rows = tuple(self)
        return rows[index]

    def __iter__(self) -> Iterator[SelectedRow]:
        connection = sqlite3.connect(f"file:{self.path}?mode=ro", uri=True)
        try:
            for record in connection.execute(
                "SELECT prompt_uuid, category, source_id, source_file_sha256, "
                "source_row_index, messages_json, tools_json, full_tokens, assistant_tokens "
                "FROM selected ORDER BY ordinal"
            ):
                yield SelectedRow(
                    prompt_uuid=record[0],
                    category=record[1],
                    source_id=record[2],
                    source_file_sha256=record[3],
                    source_row_index=record[4],
                    messages=json.loads(record[5]),
                    tools=json.loads(record[6]),
                    full_tokens=record[7],
                    assistant_tokens=record[8],
                )
        finally:
            connection.close()


@dataclass(frozen=True)
class InventoryFile:
    """One immutable physical source file in authenticated iteration order."""

    logical_path: str
    path: Path
    bytes: int
    sha256: str
    row_count: int


@dataclass(frozen=True)
class InventorySource:
    """One exact repository, split, schema, and ordered file list."""

    category: str
    source_id: str
    revision: str
    split: str
    license_expression: str
    replay_lane: str
    row_schema: dict[str, str]
    row_schema_sha256: str
    files: tuple[InventoryFile, ...]


@dataclass(frozen=True)
class SourceInventory:
    """Caller-pinned source inventory."""

    scientific_identity: str
    file_sha256: str
    sources: tuple[InventorySource, ...]


@dataclass(frozen=True)
class HistoricalExclusion:
    """Externally pinned exact historical occurrence and exclusion identity."""

    file_sha256: str
    receipt_sha256: str
    occurrence_count: int
    ordered_prompt_uuids_sha256: str
    prompt_uuids: frozenset[str]
    unique_prompt_uuids_sha256: str
    duplicate_uuid_multiplicity: dict[str, int]


@dataclass(frozen=True)
class HeldOutUnion:
    """Union of independently caller-authenticated held-out receipts."""

    prompt_uuids: frozenset[str]
    prompt_uuids_sha256: str
    receipt_file_sha256s: tuple[str, ...]
    receipt_names: tuple[str, ...]


@dataclass(frozen=True)
class ComplementCompletion:
    """One immutable published bundle identity."""

    output_root: Path
    row_count: int
    manifest_file_sha256: str
    manifest_sha256: str
    scratch_root: Path


@dataclass(frozen=True)
class TokenizerTrust:
    """Caller-pinned official Qwen3-4B tokenizer and training-template identity."""

    file_sha256: str
    receipt_sha256: str
    repository: str
    revision: str
    snapshot_path: Path
    snapshot_tree_sha256: str
    chat_template_sha256: str
    training_chat_template_sha256: str
    im_start_token_id: int
    im_end_token_id: int


def _canonical_json(value: object) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode(
        "utf-8"
    )


def _is_lower_hex(value: object, length: int) -> bool:
    return (
        isinstance(value, str)
        and len(value) == length
        and all(character in "0123456789abcdef" for character in value)
    )


def _stable_regular_bytes(path: Path) -> bytes:
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        raise ComplementError(f"authenticated input is unreadable: {path}") from error
    with os.fdopen(descriptor, "rb") as stream:
        before = os.fstat(stream.fileno())
        if not stat.S_ISREG(before.st_mode):
            raise ComplementError(f"authenticated input is not a regular file: {path}")
        raw = stream.read()
        after = os.fstat(stream.fileno())
    before_identity = (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
    after_identity = (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
    if before_identity != after_identity or len(raw) != before.st_size:
        raise ComplementError(f"authenticated input changed while reading: {path}")
    return raw


def _stable_regular_evidence(path: Path) -> tuple[int, str]:
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        raise ComplementError(f"authenticated input is unreadable: {path}") from error
    with os.fdopen(descriptor, "rb") as stream:
        before = os.fstat(stream.fileno())
        if not stat.S_ISREG(before.st_mode):
            raise ComplementError(f"authenticated input is not a regular file: {path}")
        digest = sha256()
        size = 0
        while block := stream.read(8 * 1024 * 1024):
            digest.update(block)
            size += len(block)
        after = os.fstat(stream.fileno())
    if (
        before.st_dev,
        before.st_ino,
        before.st_size,
        before.st_mtime_ns,
    ) != (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns) or size != before.st_size:
        raise ComplementError(f"authenticated input changed while hashing: {path}")
    return size, digest.hexdigest()


def _physical_row_count(path: Path, source_format: str) -> int:
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        raise ComplementError("source row-count input cannot be opened no-follow") from error
    before = os.fstat(descriptor)
    if not stat.S_ISREG(before.st_mode):
        os.close(descriptor)
        raise ComplementError("source row-count input must be a regular file")
    with os.fdopen(descriptor, "rb") as stream:
        if source_format == "jsonl":
            count = 0
            final = b""
            while block := stream.read(8 * 1024 * 1024):
                count += block.count(b"\n")
                final = block[-1:]
            row_count = count + int(bool(final) and final != b"\n")
        elif source_format == "parquet":
            try:
                import pyarrow.parquet as pq  # pyright: ignore[reportMissingImports]
            except ImportError as error:
                raise ComplementError("PyArrow is required for pinned Parquet sources") from error
            row_count = pq.ParquetFile(f"/dev/fd/{stream.fileno()}").metadata.num_rows
        else:
            raise ComplementError("source row format is unsupported")
        after = os.fstat(stream.fileno())
    if (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns) != (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
    ):
        raise ComplementError("source row-count input changed while reading")
    return row_count


def load_source_inventory(
    path: Path,
    *,
    expected_sha256: str,
    policy: TargetTokenizerPolicy = Q4_TARGET_POLICY,
) -> SourceInventory:
    """Authenticate exact source repositories, revisions, files, schema, and order."""
    if not _is_lower_hex(expected_sha256, 64):
        raise ComplementError("source inventory caller SHA-256 is invalid")
    raw = _stable_regular_bytes(path)
    if sha256(raw).hexdigest() != expected_sha256:
        raise ComplementError("source inventory caller SHA-256 mismatch")
    try:
        payload: Any = json.loads(raw)
    except json.JSONDecodeError as error:
        raise ComplementError("source inventory JSON is invalid") from error
    if not isinstance(payload, dict) or raw != _canonical_json(payload) + b"\n":
        raise ComplementError("source inventory must use canonical JSON bytes")
    if set(payload) != {"schema_version", "scientific_identity", "sources"} or (
        payload["schema_version"] != "ptv2-ptv3-complement-source-inventory-v1"
        or payload["scientific_identity"] != policy.scientific_identity
        or not isinstance(payload["sources"], list)
        or not payload["sources"]
    ):
        raise ComplementError("source inventory identity is invalid")
    parsed_sources: list[InventorySource] = []
    occurrences: set[tuple[str, str]] = set()
    for source in payload["sources"]:
        expected_source_keys = {
            "category",
            "source_id",
            "revision",
            "split",
            "license_expression",
            "approved_use",
            "replay_lane",
            "row_schema",
            "row_schema_sha256",
            "files",
        }
        if not isinstance(source, dict) or set(source) != expected_source_keys:
            raise ComplementError("source inventory source schema is invalid")
        strings = (
            source["category"],
            source["source_id"],
            source["split"],
            source["license_expression"],
            source["replay_lane"],
        )
        if (
            any(not isinstance(value, str) or not value for value in strings)
            or not _is_lower_hex(source["revision"], 40)
            or source["approved_use"] is not True
            or not isinstance(source["row_schema"], dict)
            or any(
                not isinstance(key, str) or not isinstance(value, str)
                for key, value in source["row_schema"].items()
            )
            or source["row_schema_sha256"]
            != sha256(_canonical_json(source["row_schema"])).hexdigest()
            or not isinstance(source["files"], list)
            or not source["files"]
        ):
            raise ComplementError("source inventory source pin is invalid")
        parsed_files: list[InventoryFile] = []
        source_format = source["row_schema"].get("format")
        if source_format not in {"jsonl", "parquet"}:
            raise ComplementError("source row format pin is invalid")
        for record in source["files"]:
            if not isinstance(record, dict) or set(record) != {
                "logical_path",
                "path",
                "bytes",
                "sha256",
                "row_count",
            }:
                raise ComplementError("source inventory file schema is invalid")
            logical_path = record["logical_path"]
            file_path = record["path"]
            expected_bytes = record["bytes"]
            expected_file_sha256 = record["sha256"]
            row_count = record["row_count"]
            occurrence = (source["source_id"], logical_path)
            if (
                not isinstance(logical_path, str)
                or not logical_path
                or not isinstance(file_path, str)
                or not file_path
                or type(expected_bytes) is not int
                or expected_bytes < 1
                or not _is_lower_hex(expected_file_sha256, 64)
                or type(row_count) is not int
                or row_count < 1
                or occurrence in occurrences
            ):
                raise ComplementError("source inventory file pin is invalid")
            actual_bytes, actual_sha256 = _stable_regular_evidence(Path(file_path))
            if actual_bytes != expected_bytes or actual_sha256 != expected_file_sha256:
                raise ComplementError("source physical file identity mismatch")
            if _physical_row_count(Path(file_path), source_format) != row_count:
                raise ComplementError("source physical row count mismatch")
            occurrences.add(occurrence)
            parsed_files.append(
                InventoryFile(
                    logical_path,
                    Path(file_path),
                    expected_bytes,
                    expected_file_sha256,
                    row_count,
                )
            )
        parsed_sources.append(
            InventorySource(
                category=source["category"],
                source_id=source["source_id"],
                revision=source["revision"],
                split=source["split"],
                license_expression=source["license_expression"],
                replay_lane=source["replay_lane"],
                row_schema=dict(source["row_schema"]),
                row_schema_sha256=source["row_schema_sha256"],
                files=tuple(parsed_files),
            )
        )
    return SourceInventory(policy.scientific_identity, expected_sha256, tuple(parsed_sources))


def load_historical_exclusion(
    path: Path,
    *,
    expected_sha256: str,
    expected_occurrence_count: int = 1_300_000,
) -> HistoricalExclusion:
    """Authenticate the exact historical PTV2 occurrence stream and UUID exclusion."""
    if not _is_lower_hex(expected_sha256, 64):
        raise ComplementError("historical receipt caller SHA-256 is invalid")
    raw = _stable_regular_bytes(path)
    if sha256(raw).hexdigest() != expected_sha256:
        raise ComplementError("historical receipt caller SHA-256 mismatch")
    try:
        payload: Any = json.loads(raw)
    except json.JSONDecodeError as error:
        raise ComplementError("historical receipt JSON is invalid") from error
    expected_keys = {
        "schema_version",
        "source_revision",
        "occurrence_count",
        "ordered_prompt_uuids",
        "ordered_prompt_uuids_sha256",
        "unique_prompt_uuids",
        "unique_prompt_uuids_sha256",
        "duplicate_uuid_multiplicity",
        "receipt_sha256",
    }
    if (
        not isinstance(payload, dict)
        or raw != _canonical_json(payload) + b"\n"
        or set(payload) != expected_keys
        or payload["schema_version"] != "ptv2-historical-occurrence-receipt-v1"
        or payload["source_revision"] != PTV2_REVISION
        or payload["occurrence_count"] != expected_occurrence_count
    ):
        raise ComplementError("historical receipt identity is invalid")
    occurrences = payload["ordered_prompt_uuids"]
    unique = payload["unique_prompt_uuids"]
    multiplicity = payload["duplicate_uuid_multiplicity"]
    if (
        not isinstance(occurrences, list)
        or len(occurrences) != expected_occurrence_count
        or any(not _is_lower_hex(value, 64) for value in occurrences)
        or not isinstance(unique, list)
        or unique != sorted(set(occurrences))
        or not isinstance(multiplicity, dict)
    ):
        raise ComplementError("historical occurrence evidence is invalid")
    counts = Counter(occurrences)
    expected_multiplicity = {
        prompt_uuid: count for prompt_uuid, count in sorted(counts.items()) if count > 1
    }
    body = {key: value for key, value in payload.items() if key != "receipt_sha256"}
    if (
        multiplicity != expected_multiplicity
        or payload["ordered_prompt_uuids_sha256"]
        != sha256(_canonical_json(occurrences)).hexdigest()
        or payload["unique_prompt_uuids_sha256"] != sha256(_canonical_json(unique)).hexdigest()
        or payload["receipt_sha256"] != sha256(_canonical_json(body)).hexdigest()
    ):
        raise ComplementError("historical receipt hashes or multiplicity do not reconcile")
    return HistoricalExclusion(
        file_sha256=expected_sha256,
        receipt_sha256=payload["receipt_sha256"],
        occurrence_count=expected_occurrence_count,
        ordered_prompt_uuids_sha256=payload["ordered_prompt_uuids_sha256"],
        prompt_uuids=frozenset(unique),
        unique_prompt_uuids_sha256=payload["unique_prompt_uuids_sha256"],
        duplicate_uuid_multiplicity=dict(multiplicity),
    )


def load_held_out_union(
    receipts: Iterable[tuple[Path, str] | tuple[str, Path, str]],
    *,
    required_names: frozenset[str] = REQUIRED_HELD_OUT_RECEIPT_NAMES,
) -> HeldOutUnion:
    """Authenticate each held-out receipt before forming the global exclusion union."""
    combined: set[str] = set()
    file_sha256s: list[str] = []
    normalized: list[tuple[str, Path, str]] = []
    for receipt in receipts:
        if len(receipt) == 2 and not required_names:
            path, expected_sha256 = receipt
            normalized.append(("", path, expected_sha256))
        elif len(receipt) == 3:
            name, path, expected_sha256 = receipt
            normalized.append((name, path, expected_sha256))
        else:
            raise ComplementError("held-out receipt argument schema is invalid")
    names = [name for name, _, _ in normalized]
    if required_names and (set(names) != set(required_names) or len(names) != len(required_names)):
        raise ComplementError("required held-out receipt set is missing or has extras")
    for _, path, expected_sha256 in normalized:
        if not _is_lower_hex(expected_sha256, 64):
            raise ComplementError("held-out receipt caller SHA-256 is invalid")
        raw = _stable_regular_bytes(path)
        if sha256(raw).hexdigest() != expected_sha256:
            raise ComplementError("held-out receipt caller SHA-256 mismatch")
        try:
            payload: Any = json.loads(raw)
        except json.JSONDecodeError as error:
            raise ComplementError("held-out receipt JSON is invalid") from error
        if (
            not isinstance(payload, dict)
            or raw != _canonical_json(payload) + b"\n"
            or set(payload)
            != {
                "schema_version",
                "prompt_uuids",
                "prompt_uuids_sha256",
                "receipt_sha256",
            }
            or payload["schema_version"] != "specdec-held-out-uuid-receipt-v1"
        ):
            raise ComplementError("held-out receipt identity is invalid")
        prompt_uuids = payload["prompt_uuids"]
        body = {key: value for key, value in payload.items() if key != "receipt_sha256"}
        if (
            not isinstance(prompt_uuids, list)
            or prompt_uuids != sorted(set(prompt_uuids))
            or any(not _is_lower_hex(value, 64) for value in prompt_uuids)
            or payload["prompt_uuids_sha256"] != sha256(_canonical_json(prompt_uuids)).hexdigest()
            or payload["receipt_sha256"] != sha256(_canonical_json(body)).hexdigest()
        ):
            raise ComplementError("held-out receipt does not reconcile")
        combined.update(prompt_uuids)
        file_sha256s.append(expected_sha256)
    ordered = sorted(combined)
    return HeldOutUnion(
        prompt_uuids=frozenset(combined),
        prompt_uuids_sha256=sha256(_canonical_json(ordered)).hexdigest(),
        receipt_file_sha256s=tuple(file_sha256s),
        receipt_names=tuple(names),
    )


def _snapshot_tree_sha256(root: Path) -> str:
    if root.is_symlink() or not root.is_dir():
        raise ComplementError("tokenizer snapshot must be a no-follow directory")
    entries: list[list[str]] = []
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise ComplementError("tokenizer snapshot cannot contain symlinks")
        if path.is_file():
            entries.append(
                [path.relative_to(root).as_posix(), sha256(_stable_regular_bytes(path)).hexdigest()]
            )
    if not entries:
        raise ComplementError("tokenizer snapshot is empty")
    return sha256(_canonical_json(entries)).hexdigest()


def load_tokenizer_trust(
    path: Path,
    *,
    expected_sha256: str,
    policy: TargetTokenizerPolicy = Q4_TARGET_POLICY,
) -> TokenizerTrust:
    """Authenticate the target-bound tokenizer/template trust receipt."""
    if not _is_lower_hex(expected_sha256, 64):
        raise ComplementError("tokenizer trust caller SHA-256 is invalid")
    raw = _stable_regular_bytes(path)
    if sha256(raw).hexdigest() != expected_sha256:
        raise ComplementError("tokenizer trust caller SHA-256 mismatch")
    try:
        payload: Any = json.loads(raw)
    except json.JSONDecodeError as error:
        raise ComplementError("tokenizer trust JSON is invalid") from error
    expected_keys = {
        "schema_version",
        "repository",
        "revision",
        "snapshot_path",
        "snapshot_tree_sha256",
        "chat_template_sha256",
        "training_chat_template_sha256",
        "im_start_token_id",
        "im_end_token_id",
        "receipt_sha256",
    }
    if (
        not isinstance(payload, dict)
        or raw != _canonical_json(payload) + b"\n"
        or set(payload) != expected_keys
        or payload["schema_version"] != policy.tokenizer_trust_schema
        or payload["repository"] != policy.tokenizer_repository
        or payload["revision"] != policy.tokenizer_revision
        or payload["im_start_token_id"] != 151_644
        or payload["im_end_token_id"] != 151_645
    ):
        raise ComplementError("target tokenizer identity is invalid")
    digest_fields = (
        payload["snapshot_tree_sha256"],
        payload["chat_template_sha256"],
        payload["training_chat_template_sha256"],
    )
    snapshot = Path(payload["snapshot_path"])
    body = {key: value for key, value in payload.items() if key != "receipt_sha256"}
    if (
        any(not _is_lower_hex(value, 64) for value in digest_fields)
        or payload["receipt_sha256"] != sha256(_canonical_json(body)).hexdigest()
        or _snapshot_tree_sha256(snapshot) != payload["snapshot_tree_sha256"]
    ):
        raise ComplementError("tokenizer trust evidence does not reconcile")
    return TokenizerTrust(
        file_sha256=expected_sha256,
        receipt_sha256=payload["receipt_sha256"],
        repository=payload["repository"],
        revision=payload["revision"],
        snapshot_path=snapshot,
        snapshot_tree_sha256=payload["snapshot_tree_sha256"],
        chat_template_sha256=payload["chat_template_sha256"],
        training_chat_template_sha256=payload["training_chat_template_sha256"],
        im_start_token_id=payload["im_start_token_id"],
        im_end_token_id=payload["im_end_token_id"],
    )


def _qwen_training_chat_template(official_template: str) -> str:
    assistant = '    {%- elif message.role == "assistant" %}\n'
    header = "'<|im_start|>' + message.role + '\\n' + "
    thinking_header = "'<|im_start|>' + message.role + '\\n<think>\\n' + "
    im_end = "        {{- '<|im_end|>\\n' }}\n"
    tool = '    {%- elif message.role == "tool" %}'
    boundary = im_end + tool
    if official_template.count(assistant) != 1 or official_template.count(boundary) != 1:
        raise ComplementError("pinned Qwen3 template lacks approved assistant anchors")
    before, remainder = official_template.split(assistant, 1)
    assistant_block, after = remainder.split(boundary, 1)
    if assistant_block.count(header) < 1 or assistant_block.count(thinking_header) > 1:
        raise ComplementError("pinned Qwen3 template lacks the approved assistant header")
    assistant_block = assistant_block.replace(thinking_header, "'<think>\\n' + ", 1)
    assistant_block = assistant_block.replace(header, "")
    return (
        before
        + assistant
        + "        {{- '<|im_start|>' + message.role + '\\n' }}\n"
        + "        {%- generation %}\n"
        + assistant_block
        + im_end
        + "        {%- endgeneration %}\n"
        + tool
        + after
    )


def _stage_tokenizer_snapshot(trust: TokenizerTrust, scratch_root: Path) -> Path:
    scratch_root.mkdir(parents=True, exist_ok=True)
    stage_root = Path(tempfile.mkdtemp(prefix="qwen-tokenizer-", dir=scratch_root))
    staged_snapshot = stage_root / "snapshot"
    staged_snapshot.mkdir()
    try:
        before = os.lstat(trust.snapshot_path)
        if not stat.S_ISDIR(before.st_mode) or stat.S_ISLNK(before.st_mode):
            raise ComplementError("tokenizer snapshot must be a no-follow directory")
        for source in sorted(trust.snapshot_path.rglob("*")):
            relative = source.relative_to(trust.snapshot_path)
            if source.is_symlink():
                raise ComplementError("tokenizer snapshot cannot contain symlinks")
            if source.is_dir():
                (staged_snapshot / relative).mkdir()
            elif source.is_file():
                raw = _stable_regular_bytes(source)
                destination = staged_snapshot / relative
                destination.parent.mkdir(parents=True, exist_ok=True)
                _write_durable(destination, raw)
            else:
                raise ComplementError("tokenizer snapshot contains a non-regular entry")
        after = os.lstat(trust.snapshot_path)
        if (
            before.st_dev,
            before.st_ino,
            before.st_mtime_ns,
        ) != (after.st_dev, after.st_ino, after.st_mtime_ns) or (
            _snapshot_tree_sha256(staged_snapshot) != trust.snapshot_tree_sha256
        ):
            raise ComplementError("tokenizer snapshot changed while staging")
        return staged_snapshot
    except BaseException:
        shutil.rmtree(stage_root, ignore_errors=True)
        raise


def _load_qwen_tokenizer(trust: TokenizerTrust, scratch_root: Path) -> Any:
    staged_snapshot = _stage_tokenizer_snapshot(trust, scratch_root)
    try:
        try:
            from transformers import AutoTokenizer  # pyright: ignore[reportMissingImports]

            tokenizer = AutoTokenizer.from_pretrained(
                staged_snapshot,
                local_files_only=True,
                trust_remote_code=False,
            )
            official_template = tokenizer.chat_template
            if not isinstance(official_template, str):
                raise ComplementError("pinned Qwen3 tokenizer has no chat template")
            training_template = _qwen_training_chat_template(official_template)
            identities = (
                sha256(official_template.encode()).hexdigest(),
                sha256(training_template.encode()).hexdigest(),
                tokenizer.convert_tokens_to_ids("<|im_start|>"),
                tokenizer.convert_tokens_to_ids("<|im_end|>"),
                tokenizer.eos_token_id,
            )
        except ComplementError:
            raise
        except Exception as error:
            raise ComplementError("unable to load authenticated Qwen3-4B tokenizer") from error
        if identities != (
            trust.chat_template_sha256,
            trust.training_chat_template_sha256,
            trust.im_start_token_id,
            trust.im_end_token_id,
            trust.im_end_token_id,
        ):
            raise ComplementError("authenticated tokenizer violates the Qwen3-4B contract")
        if _snapshot_tree_sha256(staged_snapshot) != trust.snapshot_tree_sha256:
            raise ComplementError("staged tokenizer changed while loading")
        setattr(tokenizer, "_continuation_training_chat_template", training_template)
        return tokenizer
    finally:
        shutil.rmtree(staged_snapshot.parent, ignore_errors=True)


@contextmanager
def _authenticated_source_stream(file: InventoryFile) -> Iterator[BinaryIO]:
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(file.path, flags)
    except OSError as error:
        raise ComplementError(
            f"source file cannot be opened no-follow: {file.logical_path}"
        ) from error
    with os.fdopen(descriptor, "rb") as stream:
        before = os.fstat(stream.fileno())

        def evidence() -> tuple[int, str]:
            stream.seek(0)
            digest = sha256()
            size = 0
            while block := stream.read(8 * 1024 * 1024):
                digest.update(block)
                size += len(block)
            return size, digest.hexdigest()

        if evidence() != (file.bytes, file.sha256):
            raise ComplementError(f"source file identity changed: {file.logical_path}")
        stream.seek(0)
        try:
            yield stream
        finally:
            after = os.fstat(stream.fileno())
            identity_before = (
                before.st_dev,
                before.st_ino,
                before.st_size,
                before.st_mtime_ns,
            )
            identity_after = (
                after.st_dev,
                after.st_ino,
                after.st_size,
                after.st_mtime_ns,
            )
            if identity_before != identity_after or evidence() != (file.bytes, file.sha256):
                raise ComplementError(f"source file changed during iteration: {file.logical_path}")


def _iter_source_file(source: InventorySource, file: InventoryFile) -> Iterable[dict[str, object]]:
    schema = source.row_schema
    messages_field = schema.get("messages_field")
    tools_field = schema.get("tools_field")
    if not messages_field or not tools_field:
        raise ComplementError("source row schema lacks message/tool field pins")
    source_columns = [messages_field]
    if tools_field != "__none__":
        source_columns.append(tools_field)
    observed = 0
    if schema.get("format") == "jsonl":
        with _authenticated_source_stream(file) as stream:
            for observed, line in enumerate(stream, start=1):
                try:
                    raw: Any = json.loads(line)
                except json.JSONDecodeError as error:
                    raise ComplementError(
                        f"invalid JSONL at {file.logical_path}:{observed}"
                    ) from error
                if not isinstance(raw, dict):
                    raise ComplementError(f"invalid source row at {file.logical_path}:{observed}")
                yield {
                    "category": source.category,
                    "source_id": source.source_id,
                    "source_file_sha256": file.sha256,
                    "source_row_index": observed - 1,
                    "messages": raw.get(messages_field),
                    "tools": raw.get(tools_field, []) if tools_field != "__none__" else [],
                    "replay_lane": source.replay_lane,
                }
    elif schema.get("format") == "parquet":
        try:
            import pyarrow.parquet as pq  # pyright: ignore[reportMissingImports]
        except ImportError as error:
            raise ComplementError("PyArrow is required for pinned Parquet sources") from error
        with _authenticated_source_stream(file) as stream:
            parquet = pq.ParquetFile(stream)
            for batch in parquet.iter_batches(
                batch_size=128, columns=source_columns, use_threads=False
            ):
                for raw in batch.to_pylist():
                    yield {
                        "category": source.category,
                        "source_id": source.source_id,
                        "source_file_sha256": file.sha256,
                        "source_row_index": observed,
                        "messages": raw.get(messages_field),
                        "tools": raw.get(tools_field, []) if tools_field != "__none__" else [],
                        "replay_lane": source.replay_lane,
                    }
                    observed += 1
    else:
        raise ComplementError("source row format is unsupported")
    if observed != file.row_count:
        raise ComplementError(
            f"source row count mismatch for {file.logical_path}: {observed}/{file.row_count}"
        )


def _category_rows(inventory: SourceInventory) -> dict[str, Iterable[Mapping[str, object]]]:
    grouped: dict[str, list[InventorySource]] = {}
    for source in inventory.sources:
        grouped.setdefault(source.category, []).append(source)
    if tuple(grouped) != tuple(APPROVED_QUOTAS):
        raise ComplementError("source inventory category order does not match approved quotas")

    def rows(sources: list[InventorySource]) -> Iterable[Mapping[str, object]]:
        for source in sources:
            for file in source.files:
                yield from _iter_source_file(source, file)

    return {category: rows(sources) for category, sources in grouped.items()}


def reconcile_source_requirements(
    path: Path,
    *,
    inventory: SourceInventory,
    capacity_receipt_path: Path,
    policy: TargetTokenizerPolicy = Q4_TARGET_POLICY,
) -> None:
    """Resolve every checked source requirement against authenticated live evidence."""
    raw = _stable_regular_bytes(path)
    if sha256(raw).hexdigest() != policy.source_requirements_sha256:
        raise ComplementError("approved source requirements identity does not match")
    try:
        requirements: Any = json.loads(raw)
    except json.JSONDecodeError as error:
        raise ComplementError("approved source requirements JSON is invalid") from error
    if (
        not isinstance(requirements, dict)
        or set(requirements)
        != {
            "schema_version",
            "scientific_identity",
            "known_pinned_sources",
            "blocking_external_pins",
        }
        or requirements["schema_version"] != "ptv2-ptv3-complement-source-requirements-v1"
        or requirements["scientific_identity"] != policy.scientific_identity
        or not isinstance(requirements["known_pinned_sources"], list)
        or not isinstance(requirements["blocking_external_pins"], list)
    ):
        raise ComplementError("approved source requirements schema is invalid")
    _require_external_approval_roots()
    inventory_files = {
        (
            source.category,
            source.source_id,
            source.revision,
            source.split,
            file.logical_path,
            file.bytes,
            file.sha256,
        )
        for source in inventory.sources
        for file in source.files
    }
    for pin in requirements["known_pinned_sources"]:
        if not isinstance(pin, dict) or set(pin) != {
            "category",
            "repository",
            "revision",
            "split",
            "path",
            "bytes",
            "sha256",
        }:
            raise ComplementError("known source pin schema is invalid")
        identity = (
            pin["category"],
            pin["repository"],
            pin["revision"],
            pin["split"],
            pin["path"],
            pin["bytes"],
            pin["sha256"],
        )
        if identity not in inventory_files:
            raise ComplementError("known source pin is missing from authenticated inventory")
    blockers = requirements["blocking_external_pins"]
    expected_blockers = APPROVED_SOURCE_REQUIREMENT_BLOCKERS.get(
        policy.source_requirements_sha256
    )
    if expected_blockers is None or blockers != list(expected_blockers):
        raise ComplementError("source requirement blocker vocabulary is invalid")
    grouped: dict[str, list[InventorySource]] = {}
    for source in inventory.sources:
        grouped.setdefault(source.category, []).append(source)
    ptv2_categories = tuple(
        category for category in APPROVED_QUOTAS if category.startswith("ptv2_")
    )
    if any(
        category not in grouped
        or any(source.revision != PTV2_REVISION for source in grouped[category])
        for category in ptv2_categories
    ):
        raise ComplementError("PTV2 candidate inventory blocker is unresolved")
    swe_sources = grouped.get("ptv3_swe_v3", [])
    observed_swe_pins = {
        (source.source_id, source.revision, source.row_schema_sha256) for source in swe_sources
    }
    if observed_swe_pins != set(APPROVED_PTV3_SWE_SOURCE_PINS):
        raise ComplementError("PTV3 SWE-v3 inventory blocker is unresolved")
    if any(
        source.row_schema_sha256 not in APPROVED_ROW_SCHEMA_SHA256S for source in inventory.sources
    ):
        raise ComplementError("source row schema blocker is unresolved")
    capacity_raw = _stable_regular_bytes(capacity_receipt_path)
    try:
        capacity: Any = json.loads(capacity_raw)
    except json.JSONDecodeError as error:
        raise ComplementError("capacity receipt JSON is invalid") from error
    expected_categories = [
        {
            "category": category,
            "required": quota,
            "selected": quota,
        }
        for category, quota in APPROVED_QUOTAS.items()
    ]
    if (
        not isinstance(capacity, dict)
        or capacity_raw != _canonical_json(capacity) + b"\n"
        or capacity.get("schema_version") != "ptv2-ptv3-complement-capacity-v1"
        or capacity.get("scientific_identity") != policy.scientific_identity
        or capacity.get("status") != "sufficient"
        or capacity.get("redistribution") != "forbidden"
        or not isinstance(capacity.get("categories"), list)
        or [
            {
                "category": record.get("category"),
                "required": record.get("required"),
                "selected": record.get("selected"),
            }
            for record in capacity["categories"]
            if isinstance(record, dict)
        ]
        != expected_categories
    ):
        raise ComplementError("post-exclusion capacity blocker is unresolved")


def _row_payload(row: Mapping[str, object]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    messages = row.get("messages")
    tools = row.get("tools", [])
    if (
        not isinstance(messages, list)
        or not messages
        or not all(isinstance(message, dict) for message in messages)
        or not isinstance(tools, list)
        or not all(isinstance(tool, dict) for tool in tools)
    ):
        raise ComplementError("invalid row schema")
    normalized_messages = deepcopy(messages)
    normalized_tools = deepcopy(tools)
    roles = [message.get("role") for message in normalized_messages]
    if any(not isinstance(role, str) for role in roles) or "assistant" not in roles:
        raise ComplementError("invalid row schema")
    return normalized_messages, normalized_tools


def prompt_uuid_from_row(row: Mapping[str, object]) -> str:
    """Derive the prompt identity from the full messages and tool schema."""
    messages, tools = _row_payload(row)
    return sha256(_canonical_json({"messages": messages, "tools": tools})).hexdigest()


def _token_evidence(
    tokenizer: Any,
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]],
) -> tuple[int, int]:
    encoded = tokenizer.apply_chat_template(
        messages,
        tools=tools or None,
        tokenize=True,
        add_generation_prompt=False,
        return_dict=True,
        return_assistant_tokens_mask=True,
        chat_template=getattr(tokenizer, "_continuation_training_chat_template", None),
    )
    if not isinstance(encoded, Mapping):
        raise ComplementError("tokenizer evidence is invalid")
    input_ids = encoded.get("input_ids")
    masks = encoded.get("assistant_masks")
    if (
        not isinstance(input_ids, list)
        or not isinstance(masks, list)
        or len(input_ids) != len(masks)
        or any(type(value) is not int for value in input_ids)
        or any(value not in (0, 1) for value in masks)
    ):
        raise ComplementError("tokenizer evidence is invalid")
    return len(input_ids), sum(masks)


def select_continuation_rows(
    rows_by_category: Mapping[str, Iterable[Mapping[str, object]]],
    *,
    quotas: Mapping[str, int],
    prior_prompt_uuids: set[str],
    held_out_prompt_uuids: set[str],
    tokenizer: Any,
    training_sequence_length: int,
    replay_categories: frozenset[str],
    spool_path: Path | None = None,
    capacity_receipt_path: Path | None = None,
    policy: TargetTokenizerPolicy = Q4_TARGET_POLICY,
) -> ContinuationSelection:
    """Select exact per-category quotas without redistribution, replacement, or cycling."""
    if tuple(rows_by_category) != tuple(quotas):
        raise ComplementError("source categories do not match quota order")
    if any(type(value) is not int or value < 1 for value in quotas.values()):
        raise ComplementError("category quotas must be positive strict integers")
    if training_sequence_length < 1:
        raise ComplementError("training sequence length must be positive")
    admitted: dict[str, bytes] = {}
    selected: list[SelectedRow] = []
    connection: sqlite3.Connection | None = None
    if spool_path is not None:
        spool_path.parent.mkdir(parents=True, exist_ok=True)
        if spool_path.exists():
            raise ComplementError("selection spool path already exists")
        connection = sqlite3.connect(spool_path)
        connection.execute("PRAGMA temp_store=FILE")
        connection.execute("PRAGMA cache_size=-65536")
        connection.execute(
            "CREATE TABLE admitted (prompt_uuid TEXT PRIMARY KEY, canonical BLOB NOT NULL)"
        )
        connection.execute(
            "CREATE TABLE selected (ordinal INTEGER PRIMARY KEY, prompt_uuid TEXT NOT NULL, "
            "category TEXT NOT NULL, source_id TEXT NOT NULL, source_file_sha256 TEXT NOT NULL, "
            "source_row_index INTEGER NOT NULL, messages_json TEXT NOT NULL, tools_json TEXT NOT NULL, "
            "full_tokens INTEGER NOT NULL, assistant_tokens INTEGER NOT NULL)"
        )
    reasons: Counter[str] = Counter()
    capacities: list[dict[str, object]] = []
    selected_ordinal = 0
    for category, quota in quotas.items():
        category_count = 0
        examined = 0
        reasons_before = reasons.copy()
        source_iterator = iter(rows_by_category[category])
        source_occurrence = 0
        while category_count < quota:
            try:
                raw = next(source_iterator)
            except StopIteration:
                break
            examined += 1
            current_source_occurrence = source_occurrence
            source_occurrence += 1
            try:
                messages, tools = _row_payload(raw)
            except ComplementError:
                reasons["invalid"] += 1
                continue
            if category in replay_categories:
                source_id = raw.get("source_id")
                if not isinstance(source_id, str) or not source_id:
                    reasons["invalid"] += 1
                    continue
                lane = (
                    "interactive-swe-replay" if "interactive" in category else "generic-tool-replay"
                )
                try:
                    validate_trajectory(
                        {"messages": messages, "tools": tools},
                        source_id=source_id,
                        lane=lane,
                        tokenizer=tokenizer,
                        training_seq_len=training_sequence_length,
                    )
                except TrajectoryValidationError as error:
                    reasons[error.reason] += 1
                    continue
            canonical = _canonical_json({"messages": messages, "tools": tools})
            prompt_uuid = sha256(canonical).hexdigest()
            if prompt_uuid in prior_prompt_uuids:
                reasons["historical"] += 1
                continue
            if prompt_uuid in held_out_prompt_uuids:
                reasons["held_out"] += 1
                continue
            previous = (
                admitted.get(prompt_uuid)
                if connection is None
                else next(
                    (
                        bytes(record[0])
                        for record in connection.execute(
                            "SELECT canonical FROM admitted WHERE prompt_uuid = ?", (prompt_uuid,)
                        )
                    ),
                    None,
                )
            )
            if previous is not None:
                if previous != canonical:
                    raise ComplementError(f"fatal prompt UUID collision: {prompt_uuid}")
                reasons["duplicate"] += 1
                continue
            full_tokens, assistant_tokens = _token_evidence(tokenizer, messages, tools)
            if full_tokens > training_sequence_length:
                reasons["overlength"] += 1
                continue
            if assistant_tokens < 1:
                reasons["no_assistant_tokens"] += 1
                continue
            source_id = raw.get("source_id")
            source_file_sha256 = raw.get("source_file_sha256")
            source_row_index = raw.get("source_row_index", current_source_occurrence)
            if (
                not isinstance(source_id, str)
                or not source_id
                or not isinstance(source_file_sha256, str)
                or len(source_file_sha256) != 64
                or type(source_row_index) is not int
                or source_row_index < 0
            ):
                reasons["invalid"] += 1
                continue
            selected_row = SelectedRow(
                prompt_uuid=prompt_uuid,
                category=category,
                source_id=source_id,
                source_file_sha256=source_file_sha256,
                source_row_index=source_row_index,
                messages=messages,
                tools=tools,
                full_tokens=full_tokens,
                assistant_tokens=assistant_tokens,
            )
            if connection is None:
                admitted[prompt_uuid] = canonical
                selected.append(selected_row)
            else:
                connection.execute("INSERT INTO admitted VALUES (?, ?)", (prompt_uuid, canonical))
                connection.execute(
                    "INSERT INTO selected VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        selected_ordinal,
                        prompt_uuid,
                        category,
                        source_id,
                        source_file_sha256,
                        source_row_index,
                        _canonical_json(messages).decode(),
                        _canonical_json(tools).decode(),
                        full_tokens,
                        assistant_tokens,
                    ),
                )
                selected_ordinal += 1
            category_count += 1
        close = getattr(source_iterator, "close", None)
        if close is not None:
            close()
        if category_count != quota:
            category_exclusions = reasons - reasons_before
            capacity_payload = {
                "schema_version": "ptv2-ptv3-complement-capacity-v1",
                "scientific_identity": policy.scientific_identity,
                "status": "insufficient-capacity",
                "blocking_category": category,
                "required": quota,
                "selected": category_count,
                "examined": examined,
                "exclusions": dict(sorted(category_exclusions.items())),
                "redistribution": "forbidden",
                "completed_categories": capacities,
            }
            if connection is not None:
                connection.commit()
                connection.close()
            if capacity_receipt_path is not None:
                capacity_receipt_path.parent.mkdir(parents=True, exist_ok=True)
                _write_durable(capacity_receipt_path, _canonical_json(capacity_payload) + b"\n")
            raise ComplementError(
                f"insufficient eligible capacity for {category}: {category_count}/{quota}"
            )
        capacities.append(
            {
                "category": category,
                "required": quota,
                "selected": category_count,
                "examined": examined,
                "exclusions": dict(sorted((reasons - reasons_before).items())),
            }
        )
    if connection is None:
        rows: Sequence[SelectedRow] = tuple(selected)
    else:
        connection.commit()
        connection.close()
        assert spool_path is not None
        rows = SpoolRows(spool_path, sum(quotas.values()))
    capacity_receipt_file_sha256 = ""
    if capacity_receipt_path is not None:
        capacity_payload = {
            "schema_version": "ptv2-ptv3-complement-capacity-v1",
            "scientific_identity": policy.scientific_identity,
            "status": "sufficient",
            "redistribution": "forbidden",
            "categories": capacities,
        }
        raw = _canonical_json(capacity_payload) + b"\n"
        capacity_receipt_path.parent.mkdir(parents=True, exist_ok=True)
        _write_durable(capacity_receipt_path, raw)
        capacity_receipt_file_sha256 = sha256(raw).hexdigest()
    return ContinuationSelection(
        rows,
        dict(sorted(reasons.items())),
        capacity_receipt_file_sha256,
        capacity_receipt_path,
    )


def _write_durable(path: Path, raw: bytes) -> None:
    with path.open("xb") as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _rename_noreplace(source: Path, destination: Path) -> None:
    import ctypes

    libc = ctypes.CDLL(None, use_errno=True)
    if sys.platform.startswith("linux"):
        result = libc.renameat2(
            -100,
            os.fsencode(source),
            -100,
            os.fsencode(destination),
            1,
        )
    elif sys.platform == "darwin":
        result = libc.renamex_np(os.fsencode(source), os.fsencode(destination), 0x00000004)
    else:
        raise ComplementError("atomic no-replace publication is unsupported on this platform")
    if result != 0:
        error = ctypes.get_errno()
        if error in {17, 39}:
            raise FileExistsError(error, os.strerror(error), str(destination))
        raise OSError(error, os.strerror(error), str(destination))


def _published_bundle_matches(
    root: Path, manifest_raw: bytes, data_sha256: str, capacity_sha256: str
) -> bool:
    try:
        expected_names = ["DATA.jsonl", "MANIFEST.json"]
        capacity_matches = True
        if capacity_sha256:
            expected_names.append("CAPACITY.json")
            capacity_matches = (
                sha256(_stable_regular_bytes(root / "CAPACITY.json")).hexdigest() == capacity_sha256
            )
        return (
            not root.is_symlink()
            and root.is_dir()
            and _stable_regular_bytes(root / "MANIFEST.json") == manifest_raw
            and _stable_regular_evidence(root / "DATA.jsonl")[1] == data_sha256
            and capacity_matches
            and tuple(sorted(item.name for item in root.iterdir())) == tuple(sorted(expected_names))
        )
    except (ComplementError, OSError):
        return False


def publish_selection_bundle(
    selection: ContinuationSelection,
    *,
    output_root: Path,
    scratch_root: Path,
    quotas: Mapping[str, int],
    config_file_sha256: str,
    source_inventory_file_sha256: str,
    historical: HistoricalExclusion,
    held_out: HeldOutUnion,
    tokenizer_trust_file_sha256: str,
    runtime_sha256: str,
    source_commit: str,
    enforce_production_paths: bool = True,
    policy: TargetTokenizerPolicy = Q4_TARGET_POLICY,
) -> ComplementCompletion:
    """Write canonical bytes on scratch and atomically install or safely adopt on Lustre."""
    hashes = (
        config_file_sha256,
        source_inventory_file_sha256,
        tokenizer_trust_file_sha256,
        runtime_sha256,
    )
    if any(not _is_lower_hex(value, 64) for value in hashes) or not _is_lower_hex(
        source_commit, 40
    ):
        raise ComplementError("publication trust identity is invalid")
    if enforce_production_paths and (
        not str(scratch_root.resolve(strict=False)).startswith("/raid/")
        or not str(output_root.resolve(strict=False)).startswith("/lustre/")
    ):
        raise ComplementError("production publication requires /raid scratch and /lustre output")
    expected_rows = sum(quotas.values())
    if len(selection.rows) != expected_rows:
        raise ComplementError("selection row count does not match exact quotas")
    counts = Counter(row.category for row in selection.rows)
    if dict(counts) != dict(quotas):
        raise ComplementError("selection category counts do not match exact quotas")
    output_root.parent.mkdir(parents=True, exist_ok=True)
    scratch_root.mkdir(parents=True, exist_ok=True)
    scratch = Path(tempfile.mkdtemp(prefix="ptv23-complement-", dir=scratch_root))
    data_path = scratch / "DATA.jsonl"
    data_hash = sha256()
    data_bytes = 0
    ordered_prompt_uuids_sha256 = sha256(b"[")
    source_occurrences_sha256 = sha256(b"[")
    selected_token_evidence_sha256 = sha256(b"[")
    evidence_count = 0
    with data_path.open("xb") as stream:
        for row in selection.rows:
            record = {
                "assistant_tokens": row.assistant_tokens,
                "category": row.category,
                "full_tokens": row.full_tokens,
                "messages": row.messages,
                "prompt_uuid": row.prompt_uuid,
                "source_file_sha256": row.source_file_sha256,
                "source_id": row.source_id,
                "source_row_index": row.source_row_index,
                "tools": row.tools,
            }
            line = _canonical_json(record) + b"\n"
            stream.write(line)
            data_hash.update(line)
            data_bytes += len(line)
            separator = b"" if evidence_count == 0 else b","
            ordered_prompt_uuids_sha256.update(separator + _canonical_json(row.prompt_uuid))
            source_occurrences_sha256.update(
                separator
                + _canonical_json([row.source_id, row.source_file_sha256, row.source_row_index])
            )
            selected_token_evidence_sha256.update(
                separator
                + _canonical_json([row.prompt_uuid, row.full_tokens, row.assistant_tokens])
            )
            evidence_count += 1
        stream.flush()
        os.fsync(stream.fileno())
    data_sha256 = data_hash.hexdigest()
    capacity_raw = b""
    if selection.capacity_receipt_path is not None:
        capacity_raw = _stable_regular_bytes(selection.capacity_receipt_path)
        if sha256(capacity_raw).hexdigest() != selection.capacity_receipt_file_sha256:
            raise ComplementError("capacity receipt changed before publication")
        _write_durable(scratch / "CAPACITY.json", capacity_raw)
    ordered_prompt_uuids_sha256.update(b"]")
    source_occurrences_sha256.update(b"]")
    selected_token_evidence_sha256.update(b"]")
    manifest_body: dict[str, object] = {
        "schema_version": "ptv2-ptv3-complement-bundle-v1",
        "scientific_identity": policy.scientific_identity,
        "row_count": len(selection.rows),
        "quotas": dict(quotas),
        "category_counts": dict(counts),
        "data": {"path": "DATA.jsonl", "bytes": data_bytes, "sha256": data_sha256},
        "ordered_prompt_uuids_sha256": ordered_prompt_uuids_sha256.hexdigest(),
        "duplicate_uuid_multiplicity": {},
        "source_occurrences_sha256": source_occurrences_sha256.hexdigest(),
        "selected_token_evidence_sha256": selected_token_evidence_sha256.hexdigest(),
        "capacity_receipt_file_sha256": selection.capacity_receipt_file_sha256,
        "capacity": (
            {
                "path": "CAPACITY.json",
                "bytes": len(capacity_raw),
                "sha256": selection.capacity_receipt_file_sha256,
            }
            if capacity_raw
            else None
        ),
        "exclusions": selection.exclusions,
        "historical": {
            "file_sha256": historical.file_sha256,
            "receipt_sha256": historical.receipt_sha256,
            "occurrence_count": historical.occurrence_count,
            "ordered_prompt_uuids_sha256": historical.ordered_prompt_uuids_sha256,
            "unique_prompt_uuids_sha256": historical.unique_prompt_uuids_sha256,
            "duplicate_uuid_multiplicity": historical.duplicate_uuid_multiplicity,
        },
        "held_out": {
            "receipt_names": list(held_out.receipt_names),
            "receipt_file_sha256s": list(held_out.receipt_file_sha256s),
            "prompt_uuids_sha256": held_out.prompt_uuids_sha256,
            "count": len(held_out.prompt_uuids),
        },
        "trust": {
            "config_file_sha256": config_file_sha256,
            "source_inventory_file_sha256": source_inventory_file_sha256,
            "tokenizer_trust_file_sha256": tokenizer_trust_file_sha256,
            "target_policy_file_sha256": policy.file_sha256,
            "runtime_sha256": runtime_sha256,
            "source_commit": source_commit,
        },
    }
    manifest_identity = sha256(_canonical_json(manifest_body)).hexdigest()
    manifest = manifest_body | {"manifest_sha256": manifest_identity}
    manifest_raw = _canonical_json(manifest) + b"\n"
    _write_durable(scratch / "MANIFEST.json", manifest_raw)
    _fsync_directory(scratch)
    manifest_file_sha256 = sha256(manifest_raw).hexdigest()
    partial = output_root.with_name(f".{output_root.name}.partial-{manifest_file_sha256[:16]}")
    if partial.exists():
        if not _published_bundle_matches(
            partial, manifest_raw, data_sha256, selection.capacity_receipt_file_sha256
        ):
            raise ComplementError("publication partial collision is foreign")
    else:
        partial.mkdir(mode=0o700)
        shutil.copyfile(data_path, partial / "DATA.jsonl")
        shutil.copyfile(scratch / "MANIFEST.json", partial / "MANIFEST.json")
        if capacity_raw:
            shutil.copyfile(scratch / "CAPACITY.json", partial / "CAPACITY.json")
        for child in partial.iterdir():
            with child.open("rb") as stream:
                os.fsync(stream.fileno())
        _fsync_directory(partial)
        if not _published_bundle_matches(
            partial, manifest_raw, data_sha256, selection.capacity_receipt_file_sha256
        ):
            raise ComplementError("publication partial changed while copying")
    if output_root.exists():
        if not _published_bundle_matches(
            output_root, manifest_raw, data_sha256, selection.capacity_receipt_file_sha256
        ):
            raise ComplementError("publication destination collision is foreign")
    else:
        try:
            _rename_noreplace(partial, output_root)
        except FileExistsError:
            if not _published_bundle_matches(
                output_root,
                manifest_raw,
                data_sha256,
                selection.capacity_receipt_file_sha256,
            ):
                raise ComplementError("publication destination collision is foreign") from None
    if not _published_bundle_matches(
        output_root,
        manifest_raw,
        data_sha256,
        selection.capacity_receipt_file_sha256,
    ):
        raise ComplementError("publication destination changed after install")
    _fsync_directory(output_root.parent)
    return ComplementCompletion(
        output_root=output_root,
        row_count=len(selection.rows),
        manifest_file_sha256=manifest_file_sha256,
        manifest_sha256=manifest_identity,
        scratch_root=scratch,
    )


def verify_selection_bundle(
    output_root: Path,
    *,
    expected_manifest_file_sha256: str,
    tokenizer: Any,
    config_path: Path,
    inventory: SourceInventory,
    historical: HistoricalExclusion,
    held_out: HeldOutUnion,
    tokenizer_trust_file_sha256: str,
    scratch_root: Path,
    expected_runtime_sha256: str,
    expected_source_commit: str,
    policy: TargetTokenizerPolicy = Q4_TARGET_POLICY,
) -> dict[str, Any]:
    """Replay one bundle and clean its scratch synchronously on every exit."""
    scratch_root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="ptv23-verify-", dir=scratch_root) as replay_name:
        return _verify_selection_bundle_with_replay_root(
            output_root,
            expected_manifest_file_sha256=expected_manifest_file_sha256,
            tokenizer=tokenizer,
            config_path=config_path,
            inventory=inventory,
            historical=historical,
            held_out=held_out,
            tokenizer_trust_file_sha256=tokenizer_trust_file_sha256,
            replay_root=Path(replay_name),
            expected_runtime_sha256=expected_runtime_sha256,
            expected_source_commit=expected_source_commit,
            policy=policy,
        )


def _verify_selection_bundle_with_replay_root(
    output_root: Path,
    *,
    expected_manifest_file_sha256: str,
    tokenizer: Any,
    config_path: Path,
    inventory: SourceInventory,
    historical: HistoricalExclusion,
    held_out: HeldOutUnion,
    tokenizer_trust_file_sha256: str,
    replay_root: Path,
    expected_runtime_sha256: str,
    expected_source_commit: str,
    policy: TargetTokenizerPolicy = Q4_TARGET_POLICY,
) -> dict[str, Any]:
    """Replay one caller-pinned installed bundle through schema, UUID, and token gates."""
    if (
        not _is_lower_hex(expected_manifest_file_sha256, 64)
        or not _is_lower_hex(tokenizer_trust_file_sha256, 64)
        or not _is_lower_hex(expected_runtime_sha256, 64)
        or not _is_lower_hex(expected_source_commit, 40)
        or output_root.is_symlink()
        or not output_root.is_dir()
    ):
        raise ComplementError("verification trust identity is invalid")
    manifest_raw = _stable_regular_bytes(output_root / "MANIFEST.json")
    if sha256(manifest_raw).hexdigest() != expected_manifest_file_sha256:
        raise ComplementError("manifest caller SHA-256 mismatch")
    try:
        manifest: Any = json.loads(manifest_raw)
    except json.JSONDecodeError as error:
        raise ComplementError("manifest JSON is invalid") from error
    expected_manifest_keys = {
        "schema_version",
        "scientific_identity",
        "row_count",
        "quotas",
        "category_counts",
        "data",
        "ordered_prompt_uuids_sha256",
        "duplicate_uuid_multiplicity",
        "source_occurrences_sha256",
        "selected_token_evidence_sha256",
        "capacity_receipt_file_sha256",
        "capacity",
        "exclusions",
        "historical",
        "held_out",
        "trust",
        "manifest_sha256",
    }
    if (
        not isinstance(manifest, dict)
        or manifest_raw != _canonical_json(manifest) + b"\n"
        or set(manifest) != expected_manifest_keys
        or manifest.get("schema_version") != "ptv2-ptv3-complement-bundle-v1"
        or manifest.get("scientific_identity") != policy.scientific_identity
        or manifest.get("quotas") != APPROVED_QUOTAS
        or manifest.get("category_counts") != APPROVED_QUOTAS
        or manifest.get("row_count") != sum(APPROVED_QUOTAS.values())
    ):
        raise ComplementError("manifest schema, canonical bytes, or approved quotas are invalid")
    _require_external_approval_roots()
    manifest_body = {key: value for key, value in manifest.items() if key != "manifest_sha256"}
    if manifest.get("manifest_sha256") != sha256(_canonical_json(manifest_body)).hexdigest():
        raise ComplementError("manifest identity does not reconcile")
    config = load_complement_config(config_path, policy=policy)
    if config["quotas"] != APPROVED_QUOTAS:
        raise ComplementError("verification config quotas are invalid")
    _authenticate_provenance_roots(inventory, historical, held_out)
    trust = manifest.get("trust")
    if (
        not isinstance(trust, dict)
        or set(trust)
        != {
            "config_file_sha256",
            "source_inventory_file_sha256",
            "tokenizer_trust_file_sha256",
            "target_policy_file_sha256",
            "runtime_sha256",
            "source_commit",
        }
        or trust.get("config_file_sha256") != sha256(_stable_regular_bytes(config_path)).hexdigest()
        or trust.get("source_inventory_file_sha256") != inventory.file_sha256
        or trust.get("tokenizer_trust_file_sha256") != tokenizer_trust_file_sha256
        or trust.get("target_policy_file_sha256") != policy.file_sha256
        or trust.get("runtime_sha256") != expected_runtime_sha256
        or trust.get("source_commit") != expected_source_commit
    ):
        raise ComplementError("manifest runtime or source commit trust mismatch")
    held_out_manifest = manifest.get("held_out")
    if (
        not isinstance(held_out_manifest, dict)
        or set(held_out_manifest)
        != {
            "receipt_names",
            "receipt_file_sha256s",
            "prompt_uuids_sha256",
            "count",
        }
        or set(held_out_manifest["receipt_names"]) != set(REQUIRED_HELD_OUT_RECEIPT_NAMES)
        or len(held_out_manifest["receipt_names"]) != len(REQUIRED_HELD_OUT_RECEIPT_NAMES)
        or len(held_out_manifest["receipt_file_sha256s"]) != len(REQUIRED_HELD_OUT_RECEIPT_NAMES)
        or any(not _is_lower_hex(value, 64) for value in held_out_manifest["receipt_file_sha256s"])
    ):
        raise ComplementError("manifest required held-out receipt set is invalid")
    expected_historical = {
        "file_sha256": historical.file_sha256,
        "receipt_sha256": historical.receipt_sha256,
        "occurrence_count": historical.occurrence_count,
        "ordered_prompt_uuids_sha256": historical.ordered_prompt_uuids_sha256,
        "unique_prompt_uuids_sha256": historical.unique_prompt_uuids_sha256,
        "duplicate_uuid_multiplicity": historical.duplicate_uuid_multiplicity,
    }
    expected_held_out = {
        "receipt_names": list(held_out.receipt_names),
        "receipt_file_sha256s": list(held_out.receipt_file_sha256s),
        "prompt_uuids_sha256": held_out.prompt_uuids_sha256,
        "count": len(held_out.prompt_uuids),
    }
    if manifest.get("historical") != expected_historical:
        raise ComplementError("manifest exclusion roots do not reconcile")
    if manifest.get("held_out") != expected_held_out:
        raise ComplementError("manifest held-out roots do not reconcile")
    data_descriptor = manifest.get("data")
    if (
        not isinstance(data_descriptor, dict)
        or set(data_descriptor) != {"path", "bytes", "sha256"}
        or data_descriptor.get("path") != "DATA.jsonl"
    ):
        raise ComplementError("manifest data descriptor is invalid")
    capacity = manifest.get("capacity")
    if (
        not isinstance(capacity, dict)
        or capacity.get("path") != "CAPACITY.json"
        or capacity.get("sha256") != manifest.get("capacity_receipt_file_sha256")
    ):
        raise ComplementError("capacity descriptor is invalid")
    capacity_path = output_root / "CAPACITY.json"
    capacity_raw = _stable_regular_bytes(capacity_path)
    if (
        capacity.get("bytes") != len(capacity_raw)
        or capacity.get("sha256") != sha256(capacity_raw).hexdigest()
    ):
        raise ComplementError("capacity receipt identity does not reconcile")
    reconcile_source_requirements(
        config_path.parent / policy.source_requirements_path,
        inventory=inventory,
        capacity_receipt_path=capacity_path,
        policy=policy,
    )
    replay_capacity_path = replay_root / "CAPACITY.json"
    replay = select_continuation_rows(
        _category_rows(inventory),
        quotas=APPROVED_QUOTAS,
        prior_prompt_uuids=set(historical.prompt_uuids),
        held_out_prompt_uuids=set(held_out.prompt_uuids),
        tokenizer=tokenizer,
        training_sequence_length=policy.training_sequence_length,
        replay_categories=frozenset(
            {"ptv3_interactive_agentic_swe", "ptv3_general_tool_trajectories"}
        ),
        spool_path=replay_root / "selection.sqlite",
        capacity_receipt_path=replay_capacity_path,
        policy=policy,
    )
    if (
        replay.exclusions != manifest.get("exclusions")
        or _stable_regular_bytes(replay_capacity_path) != capacity_raw
    ):
        raise ComplementError("selection replay capacity or exclusions do not reconcile")
    replay_rows = iter(replay.rows)
    prompt_uuids_sha256 = sha256(b"[")
    occurrences_sha256 = sha256(b"[")
    token_evidence_sha256 = sha256(b"[")
    counts: Counter[str] = Counter()
    category_order: list[str] = []
    admitted: set[str] = set()
    data_digest = sha256()
    data_bytes = 0
    row_count = 0
    data_path = output_root / "DATA.jsonl"
    if data_path.is_symlink() or not data_path.is_file():
        raise ComplementError("data identity does not reconcile")
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(data_path, flags)
    except OSError as error:
        raise ComplementError("data identity does not reconcile") from error
    with os.fdopen(descriptor, "rb") as stream:
        before = os.fstat(stream.fileno())
        if data_descriptor.get("bytes") != before.st_size:
            raise ComplementError("data identity does not reconcile")
        for index, raw_line in enumerate(stream):
            data_digest.update(raw_line)
            data_bytes += len(raw_line)
            if not raw_line.endswith(b"\n"):
                raise ComplementError(f"data row {index} lacks canonical newline")
            line = raw_line[:-1]
            try:
                record: Any = json.loads(line)
            except json.JSONDecodeError as error:
                raise ComplementError(f"data row {index} JSON is invalid") from error
            expected_keys = {
                "assistant_tokens",
                "category",
                "full_tokens",
                "messages",
                "prompt_uuid",
                "source_file_sha256",
                "source_id",
                "source_row_index",
                "tools",
            }
            if (
                not isinstance(record, dict)
                or set(record) != expected_keys
                or line != _canonical_json(record)
            ):
                raise ComplementError(f"data row {index} schema or canonical bytes are invalid")
            try:
                replay_row = next(replay_rows)
            except StopIteration as error:
                raise ComplementError("data contains rows absent from selection replay") from error
            replay_record = {
                "assistant_tokens": replay_row.assistant_tokens,
                "category": replay_row.category,
                "full_tokens": replay_row.full_tokens,
                "messages": replay_row.messages,
                "prompt_uuid": replay_row.prompt_uuid,
                "source_file_sha256": replay_row.source_file_sha256,
                "source_id": replay_row.source_id,
                "source_row_index": replay_row.source_row_index,
                "tools": replay_row.tools,
            }
            if record != replay_record:
                raise ComplementError("data row does not match authenticated selection replay")
            messages, tools = _row_payload(record)
            category = record["category"]
            if category in {
                "ptv3_interactive_agentic_swe",
                "ptv3_general_tool_trajectories",
            }:
                lane = (
                    "interactive-swe-replay"
                    if category == "ptv3_interactive_agentic_swe"
                    else "generic-tool-replay"
                )
                try:
                    validate_trajectory(
                        {"messages": messages, "tools": tools},
                        source_id=record["source_id"],
                        lane=lane,
                        tokenizer=tokenizer,
                        training_seq_len=policy.training_sequence_length,
                    )
                except TrajectoryValidationError as error:
                    raise ComplementError(f"data trajectory is invalid at row {index}") from error
            canonical = _canonical_json({"messages": messages, "tools": tools})
            prompt_uuid = sha256(canonical).hexdigest()
            if prompt_uuid != record["prompt_uuid"] or prompt_uuid in admitted:
                raise ComplementError("data prompt UUID or multiplicity is invalid")
            full_tokens, assistant_tokens = _token_evidence(tokenizer, messages, tools)
            if (
                full_tokens != record["full_tokens"]
                or assistant_tokens != record["assistant_tokens"]
                or full_tokens > policy.training_sequence_length
                or assistant_tokens < 1
            ):
                raise ComplementError("selected tokenizer evidence does not reconcile")
            source_id = record["source_id"]
            source_file_sha256 = record["source_file_sha256"]
            source_row_index = record["source_row_index"]
            if (
                not isinstance(category, str)
                or not isinstance(source_id, str)
                or not _is_lower_hex(source_file_sha256, 64)
                or type(source_row_index) is not int
                or source_row_index < 0
            ):
                raise ComplementError("data source occurrence is invalid")
            admitted.add(prompt_uuid)
            separator = b"" if row_count == 0 else b","
            prompt_uuids_sha256.update(separator + _canonical_json(prompt_uuid))
            occurrences_sha256.update(
                separator + _canonical_json([source_id, source_file_sha256, source_row_index])
            )
            token_evidence_sha256.update(
                separator + _canonical_json([prompt_uuid, full_tokens, assistant_tokens])
            )
            counts[category] += 1
            if not category_order or category_order[-1] != category:
                category_order.append(category)
            row_count += 1
        after = os.fstat(stream.fileno())
    try:
        next(replay_rows)
    except StopIteration:
        pass
    else:
        raise ComplementError("selection replay contains rows absent from data")
    if (
        (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
        != (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
        or data_descriptor.get("bytes") != data_bytes
        or data_descriptor.get("sha256") != data_digest.hexdigest()
    ):
        raise ComplementError("data identity does not reconcile")
    prompt_uuids_sha256.update(b"]")
    occurrences_sha256.update(b"]")
    token_evidence_sha256.update(b"]")
    if (
        row_count != manifest.get("row_count")
        or tuple(category_order) != tuple(APPROVED_QUOTAS)
        or dict(counts) != manifest.get("category_counts")
        or dict(counts) != manifest.get("quotas")
        or manifest.get("duplicate_uuid_multiplicity") != {}
        or manifest.get("ordered_prompt_uuids_sha256") != prompt_uuids_sha256.hexdigest()
        or manifest.get("source_occurrences_sha256") != occurrences_sha256.hexdigest()
        or manifest.get("selected_token_evidence_sha256") != token_evidence_sha256.hexdigest()
    ):
        raise ComplementError("manifest row evidence does not reconcile")
    return manifest


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build or verify ptv2-ptv3-complement-700k-v1",
        formatter_class=argparse.RawTextHelpFormatter,
        epilog=(
            "Authenticated inputs:\n"
            "  --historical-receipt-sha256\n"
            "  --source-inventory-sha256\n"
            "  --tokenizer-trust-sha256\n"
            "  --runtime-sha256"
        ),
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    build = subparsers.add_parser("build")
    build.add_argument("--target-policy", type=Path, required=True)
    build.add_argument("--target-policy-sha256", required=True)
    build.add_argument("--config", type=Path, required=True)
    build.add_argument("--source-inventory", type=Path, required=True)
    build.add_argument("--source-inventory-sha256", required=True)
    build.add_argument("--historical-receipt", type=Path, required=True)
    build.add_argument("--historical-receipt-sha256", required=True)
    build.add_argument(
        "--held-out-receipt",
        action="append",
        default=[],
        metavar="NAME=PATH:SHA256",
    )
    build.add_argument("--tokenizer-trust", type=Path, required=True)
    build.add_argument("--tokenizer-trust-sha256", required=True)
    build.add_argument("--runtime-sha256", required=True)
    build.add_argument("--source-commit", required=True)
    build.add_argument("--scratch-root", type=Path, required=True)
    build.add_argument("--output-root", type=Path, required=True)
    build.add_argument("--completion-receipt", type=Path, required=True)
    verify = subparsers.add_parser("verify")
    verify.add_argument("--target-policy", type=Path, required=True)
    verify.add_argument("--target-policy-sha256", required=True)
    verify.add_argument("--config", type=Path, required=True)
    verify.add_argument("--source-inventory", type=Path, required=True)
    verify.add_argument("--source-inventory-sha256", required=True)
    verify.add_argument("--historical-receipt", type=Path, required=True)
    verify.add_argument("--historical-receipt-sha256", required=True)
    verify.add_argument(
        "--held-out-receipt", action="append", default=[], metavar="NAME=PATH:SHA256"
    )
    verify.add_argument("--output-root", type=Path, required=True)
    verify.add_argument("--completion-receipt", type=Path, required=True)
    verify.add_argument("--completion-receipt-sha256", required=True)
    verify.add_argument("--tokenizer-trust", type=Path, required=True)
    verify.add_argument("--tokenizer-trust-sha256", required=True)
    verify.add_argument("--runtime-sha256", required=True)
    verify.add_argument("--source-commit", required=True)
    verify.add_argument("--scratch-root", type=Path, required=True)
    return parser.parse_args(argv)


def _completion_payload(
    completion: ComplementCompletion,
    *,
    runtime_sha256: str,
    source_commit: str,
    policy: TargetTokenizerPolicy = Q4_TARGET_POLICY,
) -> dict[str, object]:
    body: dict[str, object] = {
        "schema_version": "ptv2-ptv3-complement-completion-v1",
        "scientific_identity": policy.scientific_identity,
        "output_root": str(completion.output_root),
        "row_count": completion.row_count,
        "manifest_file_sha256": completion.manifest_file_sha256,
        "manifest_sha256": completion.manifest_sha256,
        "runtime_sha256": runtime_sha256,
        "source_commit": source_commit,
    }
    return body | {"receipt_sha256": sha256(_canonical_json(body)).hexdigest()}


def _write_completion_receipt(
    path: Path,
    completion: ComplementCompletion,
    *,
    runtime_sha256: str,
    source_commit: str,
    policy: TargetTokenizerPolicy = Q4_TARGET_POLICY,
) -> str:
    payload = _completion_payload(
        completion, runtime_sha256=runtime_sha256, source_commit=source_commit, policy=policy
    )
    raw = _canonical_json(payload) + b"\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    _write_durable(path, raw)
    _fsync_directory(path.parent)
    return sha256(raw).hexdigest()


def _load_completion_receipt(
    path: Path,
    *,
    expected_sha256: str,
    output_root: Path,
    runtime_sha256: str,
    source_commit: str,
    policy: TargetTokenizerPolicy = Q4_TARGET_POLICY,
) -> dict[str, Any]:
    raw = _stable_regular_bytes(path)
    if sha256(raw).hexdigest() != expected_sha256:
        raise ComplementError("completion receipt caller SHA-256 mismatch")
    try:
        payload: Any = json.loads(raw)
    except json.JSONDecodeError as error:
        raise ComplementError("completion receipt JSON is invalid") from error
    if not isinstance(payload, dict):
        raise ComplementError("completion receipt identity does not reconcile")
    body = {key: value for key, value in payload.items() if key != "receipt_sha256"}
    if (
        raw != _canonical_json(payload) + b"\n"
        or payload.get("schema_version") != "ptv2-ptv3-complement-completion-v1"
        or payload.get("scientific_identity") != policy.scientific_identity
        or payload.get("output_root") != str(output_root)
        or payload.get("runtime_sha256") != runtime_sha256
        or payload.get("source_commit") != source_commit
        or payload.get("receipt_sha256") != sha256(_canonical_json(body)).hexdigest()
    ):
        raise ComplementError("completion receipt identity does not reconcile")
    return payload


def _held_out_args(values: Iterable[str]) -> list[tuple[str, Path, str]]:
    receipts: list[tuple[str, Path, str]] = []
    for value in values:
        name, name_separator, remainder = value.partition("=")
        path_text, digest_separator, digest = remainder.rpartition(":")
        if (
            not name_separator
            or name not in REQUIRED_HELD_OUT_RECEIPT_NAMES
            or not digest_separator
            or not path_text
            or not _is_lower_hex(digest, 64)
        ):
            raise ComplementError("held-out receipt argument must be NAME=PATH:SHA256")
        receipts.append((name, Path(path_text), digest))
    return receipts


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    policy = load_target_policy(args.target_policy, args.target_policy_sha256)
    tokenizer_trust = load_tokenizer_trust(
        args.tokenizer_trust, expected_sha256=args.tokenizer_trust_sha256, policy=policy
    )
    tokenizer = _load_qwen_tokenizer(tokenizer_trust, args.scratch_root)
    if args.command == "build":
        _require_external_approval_roots()
        config = load_complement_config(args.config, policy=policy)
        inventory = load_source_inventory(
            args.source_inventory, expected_sha256=args.source_inventory_sha256, policy=policy
        )
        historical = load_historical_exclusion(
            args.historical_receipt,
            expected_sha256=args.historical_receipt_sha256,
        )
        held_out = load_held_out_union(_held_out_args(args.held_out_receipt))
        _authenticate_provenance_roots(inventory, historical, held_out)
        if not str(args.scratch_root.resolve(strict=False)).startswith("/raid/"):
            raise ComplementError("production selection requires /raid scratch")
        args.scratch_root.mkdir(parents=True, exist_ok=True)
        selection_scratch = Path(tempfile.mkdtemp(prefix="ptv23-selection-", dir=args.scratch_root))
        selection = select_continuation_rows(
            _category_rows(inventory),
            quotas=config["quotas"],
            prior_prompt_uuids=set(historical.prompt_uuids),
            held_out_prompt_uuids=set(held_out.prompt_uuids),
            tokenizer=tokenizer,
            training_sequence_length=policy.training_sequence_length,
            replay_categories=frozenset(
                {
                    "ptv3_interactive_agentic_swe",
                    "ptv3_general_tool_trajectories",
                }
            ),
            spool_path=selection_scratch / "selection.sqlite",
            capacity_receipt_path=selection_scratch / "CAPACITY.json",
            policy=policy,
        )
        reconcile_source_requirements(
            args.config.parent / policy.source_requirements_path,
            inventory=inventory,
            capacity_receipt_path=selection_scratch / "CAPACITY.json",
            policy=policy,
        )
        completion = publish_selection_bundle(
            selection,
            output_root=args.output_root,
            scratch_root=args.scratch_root,
            quotas=config["quotas"],
            config_file_sha256=sha256(_stable_regular_bytes(args.config)).hexdigest(),
            source_inventory_file_sha256=inventory.file_sha256,
            historical=historical,
            held_out=held_out,
            tokenizer_trust_file_sha256=tokenizer_trust.file_sha256,
            runtime_sha256=args.runtime_sha256,
            source_commit=args.source_commit,
            policy=policy,
        )
        completion_receipt_sha256 = _write_completion_receipt(
            args.completion_receipt,
            completion,
            runtime_sha256=args.runtime_sha256,
            source_commit=args.source_commit,
            policy=policy,
        )
        print(
            json.dumps(
                {
                    "completion_receipt": str(args.completion_receipt),
                    "completion_receipt_sha256": completion_receipt_sha256,
                    "manifest_file_sha256": completion.manifest_file_sha256,
                    "output_root": str(completion.output_root),
                    "row_count": completion.row_count,
                },
                sort_keys=True,
            )
        )
        return 0
    _require_external_approval_roots()
    inventory = load_source_inventory(
        args.source_inventory, expected_sha256=args.source_inventory_sha256, policy=policy
    )
    historical = load_historical_exclusion(
        args.historical_receipt, expected_sha256=args.historical_receipt_sha256
    )
    held_out = load_held_out_union(_held_out_args(args.held_out_receipt))
    completion_receipt = _load_completion_receipt(
        args.completion_receipt,
        expected_sha256=args.completion_receipt_sha256,
        output_root=args.output_root,
        runtime_sha256=args.runtime_sha256,
        source_commit=args.source_commit,
        policy=policy,
    )
    verify_selection_bundle(
        args.output_root,
        expected_manifest_file_sha256=completion_receipt["manifest_file_sha256"],
        tokenizer=tokenizer,
        config_path=args.config,
        inventory=inventory,
        historical=historical,
        held_out=held_out,
        tokenizer_trust_file_sha256=tokenizer_trust.file_sha256,
        scratch_root=args.scratch_root,
        expected_runtime_sha256=args.runtime_sha256,
        expected_source_commit=args.source_commit,
        policy=policy,
    )
    print(json.dumps(completion_receipt, sort_keys=True))
    return 0


def load_complement_config(
    path: Path, *, policy: TargetTokenizerPolicy = Q4_TARGET_POLICY
) -> dict[str, Any]:
    """Load the immutable quota and source policy bound to one target policy."""
    try:
        raw = _stable_regular_bytes(path)
        payload: Any = json.loads(raw)
    except (OSError, json.JSONDecodeError) as error:
        raise ComplementError("approved continuation policy is unreadable") from error
    expected = {
        "schema_version": SCHEMA_VERSION,
        "scientific_identity": policy.scientific_identity,
        "training_sequence_length": policy.training_sequence_length,
        "tokenizer": {
            "repository": policy.tokenizer_repository,
            "revision": policy.tokenizer_revision,
            "training_sequence_length": policy.training_sequence_length,
            "trust_schema": policy.tokenizer_trust_schema,
        },
        "source_requirements": {
            "path": policy.source_requirements_path,
            "sha256": policy.source_requirements_sha256,
        },
        "quotas": APPROVED_QUOTAS,
    }
    if sha256(raw).hexdigest() != policy.quota_config_sha256 or payload != expected:
        raise ComplementError("approved continuation policy does not match")
    requirements = path.parent / policy.source_requirements_path
    if (
        sha256(_stable_regular_bytes(requirements)).hexdigest()
        != policy.source_requirements_sha256
    ):
        raise ComplementError("approved source requirements identity does not match")
    return expected


if __name__ == "__main__":
    raise SystemExit(main())
