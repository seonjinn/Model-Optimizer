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

"""Stage exact assistant-loss-token exposure views from promoted responses."""

from __future__ import annotations

import json
import os
import re
import sqlite3
import uuid
from collections import Counter
from collections.abc import Iterator, Mapping
from dataclasses import asdict, dataclass
from hashlib import sha256
from pathlib import Path
from types import MappingProxyType
from typing import Any, Literal

from specdec_corpus_contracts import canonical_json
from trajectory_schema import TrajectoryValidationError, validate_trajectory

__all__ = [
    "ExposureView",
    "ExposureViewError",
    "PTV2OnePassCorpus",
    "PTV2StudyExposureViews",
    "PairedExposureViews",
    "build_exposure_views",
    "build_paired_exposure_views",
    "build_ptv2_study_exposures",
    "derive_ptv2_one_pass_corpus",
]

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_SAFE_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
_PRODUCTION_BOUNDARIES = (256_000_000, 1_000_000_000)
_RUNTIME_SCREEN_TOKENS = 64_000_000
_EXPOSURE_POLICY_SHA256 = sha256(
    canonical_json(
        {
            "production_assistant_token_boundaries": list(_PRODUCTION_BOUNDARIES),
            "runtime_screen_assistant_tokens": _RUNTIME_SCREEN_TOKENS,
        }
    )
).hexdigest()


class ExposureViewError(ValueError):
    """An exact-token staging identity or record contract is invalid."""


@dataclass(frozen=True)
class ExposureView:
    """Authenticated, consumable row order for one assistant-token boundary."""

    schema_version: int
    name: str
    arm: str
    response_corpus_sha256: str
    tokenizer_sha256: str
    chat_template_sha256: str
    selection_sha256: str
    training_sequence_length: int
    purpose: str
    policy_sha256: str
    source_policy_sha256: str
    seed: int
    assistant_tokens: int
    full_prompt_assistant_tokens: int
    row_count: int
    unique_prompt_count: int
    repeated_prompt_count: int
    completed_epochs: int
    records_path: str
    records_bytes: int
    records_sha256: str
    row_order_sha256: str
    input_ids_sha256: str
    loss_masks_sha256: str
    cumulative_tokens_sha256: str
    paired_selection_bucket_sha256: str
    task5_paired_selection_bucket_sha256: str
    task6_completion_sha256: str
    final_paired_response_sha256: str
    resume_fingerprint: str
    receipt_path: str
    receipt_sha256: str


@dataclass(frozen=True)
class PairedExposureViews:
    """C/D exact-token views bound to their common selection and bucket proof."""

    C: Mapping[str, ExposureView]
    D: Mapping[str, ExposureView]
    common_boundary: int
    C_full_prompt_assistant_tokens: int
    D_full_prompt_assistant_tokens: int
    paired_selection_bucket_sha256: str
    final_paired_response_sha256: str
    C_task6_completion_sha256: str
    D_task6_completion_sha256: str
    source_policy_sha256: str
    exposure_policy_sha256: str
    receipt_path: str
    receipt_sha256: str


@dataclass(frozen=True)
class PTV2OnePassCorpus:
    """Authenticated token totals for one exact 2M-occurrence PTV2 arm."""

    strategy: Literal["A-repair", "B-balanced"]
    occurrence_count: int
    trainer_epochs: int
    assistant_tokens: int
    tokenizer_sha256: str
    chat_template_sha256: str
    assistant_loss_target_sha256: str
    training_config_sha256: str
    source_response_root_sha256: str
    ordered_occurrences_sha256: str
    unique_prompt_count: int = 2_000_000
    natural_duplicate_count: int = 0
    constructed_repeat_count: int = 0
    serialized_tokens: int = 0
    # This is a lower bound until a concrete packer receipt is attached.  It is
    # deliberately not described as a measured packed-sequence count.
    packed_sequences: int = 0
    milestone_occurrences: tuple[int, ...] = (500_224, 1_000_448, 1_300_480, 2_000_000)
    milestone_steps: tuple[int, ...] = (977, 1_954, 2_540, 3_908)
    segment_occurrences: tuple[int, int] = (1_300_000, 700_000)
    segment_steps: tuple[int, int] = (2_540, 1_368)
    cumulative_segment_steps: tuple[int, int] = (2_540, 3_908)
    segment_final_valid_occurrences: tuple[int, int] = (32, 96)
    tokenized_path: str = ""
    tokenized_sha256: str = "0" * 64
    receipt_path: str = ""
    receipt_sha256: str = "0" * 64


@dataclass(frozen=True)
class PTV2StudyExposureViews:
    """One-pass and reachable paired scientific receipt boundaries for A/B."""

    runtime_screen_tokens: int
    scientific_tokens: int
    prefix_u2m: int
    balanced_u2m: int
    paired_scientific_reached: bool
    runtime_artifacts: Mapping[str, str] = MappingProxyType({})
    scientific_artifacts: Mapping[str, str] = MappingProxyType({})


def derive_ptv2_one_pass_corpus(
    view: Any,
    *,
    tokenizer_sha256: str,
    chat_template_sha256: str,
    assistant_loss_target_sha256: str,
    training_config_sha256: str,
    tokenizer: Any,
    output_root: Path,
    sequence_length: int = 4_096,
    milestone_occurrences: tuple[int, ...] = (500_224, 1_000_448, 1_300_480, 2_000_000),
    milestone_steps: tuple[int, ...] = (977, 1_954, 2_540, 3_908),
) -> PTV2OnePassCorpus:
    """Materialize authenticated assistant masks from the selected SQLite responses."""
    for digest_name, digest in (
        ("tokenizer", tokenizer_sha256),
        ("chat template", chat_template_sha256),
        ("assistant loss target", assistant_loss_target_sha256),
        ("training config", training_config_sha256),
    ):
        _require_digest(digest_name, digest)
    actual_tokenizer_sha256 = getattr(tokenizer, "tokenizer_sha256", None)
    actual_template = getattr(tokenizer, "chat_template", None)
    actual_mask_sha256 = getattr(tokenizer, "assistant_loss_target_sha256", None)
    if (
        actual_tokenizer_sha256 != tokenizer_sha256
        or not isinstance(actual_template, str)
        or sha256(actual_template.encode("utf-8")).hexdigest() != chat_template_sha256
        or actual_mask_sha256 != assistant_loss_target_sha256
    ):
        raise ExposureViewError("PTV2 tokenizer, template, or assistant-mask identity mismatch")
    strategy = getattr(view, "strategy", None)
    if strategy not in {"A-repair", "B-balanced"}:
        raise ExposureViewError("PTV2 materialized view strategy is invalid")
    index_path = Path(getattr(view, "index_path", ""))
    if not index_path.is_file() or index_path.is_symlink():
        raise ExposureViewError("PTV2 selection SQLite index is missing or unsafe")
    if (
        not isinstance(sequence_length, int)
        or isinstance(sequence_length, bool)
        or sequence_length < 1
        or len(milestone_occurrences) != len(milestone_steps)
        or any(
            not isinstance(value, int) or isinstance(value, bool) or value < 1
            for value in (*milestone_occurrences, *milestone_steps)
        )
    ):
        raise ExposureViewError("PTV2 sequence and milestone policy is invalid")
    root = Path(output_root)
    root.mkdir(parents=True, exist_ok=True)
    if root.is_symlink() or not root.is_dir():
        raise ExposureViewError("PTV2 tokenized output root is unsafe")
    database_path = root / f"{strategy.lower()}-tokenized.sqlite3"
    receipt_path = root / f"{strategy.lower()}-TOKENIZED.json"
    temporary_database = _prepare_ptv2_tokenized_database(root, database_path, receipt_path)
    occurrence_digest = sha256()
    response_digest = sha256()
    count = assistant_tokens = serialized_tokens = 0
    connection_out = sqlite3.connect(temporary_database)
    connection_out.execute("PRAGMA synchronous=FULL")
    connection_out.execute(
        "CREATE TABLE records(ordinal INTEGER PRIMARY KEY,prompt_uuid TEXT NOT NULL,"
        "input_ids_json TEXT NOT NULL,loss_mask_json TEXT NOT NULL,assistant_tokens INTEGER NOT NULL)"
    )
    connection = sqlite3.connect(f"file:{index_path}?mode=ro", uri=True)
    try:
        cursor = connection.execute(
            "SELECT occurrences.ordinal,occurrences.prompt_uuid,occurrences.source_identity_sha256,"
            "occurrences.source_row,occurrences.cell,occurrences.reuse_index,"
            "occurrences.conversation_sha256,occurrences.assistant_response_sha256,"
            "source_rows.canonical_conversation,source_rows.assistant_response "
            "FROM occurrences JOIN source_rows "
            "ON occurrences.source_identity_sha256=source_rows.source_identity_sha256 "
            "AND occurrences.source_row=source_rows.source_row "
            "WHERE occurrences.strategy=? ORDER BY occurrences.ordinal",
            (strategy,),
        )
        for row in cursor:
            occurrence = row[:8]
            conversation = row[8]
            response = row[9]
            if sha256(conversation.encode("utf-8")).hexdigest() != occurrence[6]:
                raise ExposureViewError("PTV2 selected conversation hash mismatch")
            if sha256(response.encode("utf-8")).hexdigest() != occurrence[7]:
                raise ExposureViewError("PTV2 selected assistant response hash mismatch")
            try:
                canonical = json.loads(conversation)
            except json.JSONDecodeError as error:
                raise ExposureViewError("PTV2 selected conversation is invalid JSON") from error
            if not isinstance(canonical, dict) or not isinstance(canonical.get("messages"), list):
                raise ExposureViewError("PTV2 selected conversation has no messages")
            assistants = [
                message
                for message in canonical["messages"]
                if isinstance(message, dict) and message.get("role") == "assistant"
            ]
            if not assistants or canonical_json(assistants[-1]).decode("utf-8") != response:
                raise ExposureViewError(
                    "PTV2 selected response is not the final assistant in its conversation"
                )
            encoded = tokenizer.apply_chat_template(
                canonical["messages"],
                tools=canonical.get("tools") or None,
                tokenize=True,
                add_generation_prompt=False,
                return_dict=True,
                return_assistant_tokens_mask=True,
            )
            input_ids = encoded.get("input_ids") if isinstance(encoded, Mapping) else None
            loss_mask = encoded.get("assistant_masks") if isinstance(encoded, Mapping) else None
            if loss_mask is None and isinstance(encoded, Mapping):
                loss_mask = encoded.get("assistant_tokens_mask")
            if (
                not isinstance(input_ids, list)
                or not input_ids
                or not isinstance(loss_mask, list)
                or len(input_ids) != len(loss_mask)
                or any(not isinstance(value, int) or isinstance(value, bool) for value in input_ids)
                or any(value not in (0, 1) for value in loss_mask)
            ):
                raise ExposureViewError(
                    "PTV2 tokenizer did not return aligned IDs and assistant mask"
                )
            input_ids = input_ids[:sequence_length]
            loss_mask = loss_mask[:sequence_length]
            token_count = sum(loss_mask)
            if token_count < 1:
                raise ExposureViewError("PTV2 final training boundary has no assistant tokens")
            occurrence_digest.update(canonical_json(list(occurrence)))
            occurrence_digest.update(b"\n")
            response_digest.update(
                canonical_json([occurrence[2], occurrence[3], occurrence[6], occurrence[7]])
            )
            response_digest.update(b"\n")
            count += 1
            assistant_tokens += token_count
            serialized_tokens += len(input_ids)
            connection_out.execute(
                "INSERT INTO records VALUES(?,?,?,?,?)",
                (
                    occurrence[0],
                    occurrence[1],
                    canonical_json(input_ids).decode("utf-8"),
                    canonical_json(loss_mask).decode("utf-8"),
                    token_count,
                ),
            )
        connection_out.commit()
    except Exception:
        connection_out.rollback()
        raise
    finally:
        connection.close()
        connection_out.close()
    if count != getattr(view, "occurrence_count", None):
        raise ExposureViewError(
            "PTV2 materialized occurrence count does not match selection receipt"
        )
    if occurrence_digest.hexdigest() != getattr(view, "ordered_occurrences_sha256", None):
        raise ExposureViewError(
            "PTV2 materialized occurrence order does not match selection receipt"
        )
    if response_digest.hexdigest() != getattr(view, "source_response_root_sha256", None):
        raise ExposureViewError("PTV2 materialized response root does not match selection receipt")
    unique = int(getattr(view, "unique_prompt_count", 0))
    natural = int(getattr(view, "natural_duplicate_count", 0))
    constructed = int(getattr(view, "constructed_repeat_count", 0))
    if unique < 1 or natural < 0 or constructed < 0 or unique + natural + constructed != count:
        raise ExposureViewError("PTV2 selection multiplicity summary does not reconcile")
    _fsync_file(temporary_database)
    _publish_ptv2_tokenized_database(temporary_database, database_path)
    database_sha256 = _sha256_file(database_path)
    receipt = {
        "schema_version": 1,
        "strategy": strategy,
        "occurrence_count": count,
        "trainer_epochs": getattr(view, "trainer_epochs", 0),
        "assistant_tokens": assistant_tokens,
        "serialized_tokens": serialized_tokens,
        "packed_sequence_lower_bound": (serialized_tokens + sequence_length - 1) // sequence_length,
        "unique_prompt_count": unique,
        "natural_duplicate_count": natural,
        "constructed_repeat_count": constructed,
        "milestone_occurrences": list(milestone_occurrences),
        "milestone_steps": list(milestone_steps),
        "segment_occurrences": [1_300_000, 700_000],
        "segment_steps": [2_540, 1_368],
        "cumulative_segment_steps": [2_540, 3_908],
        "segment_final_valid_occurrences": [32, 96],
        "tokenizer_sha256": tokenizer_sha256,
        "chat_template_sha256": chat_template_sha256,
        "assistant_loss_target_sha256": assistant_loss_target_sha256,
        "training_config_sha256": training_config_sha256,
        "source_response_root_sha256": response_digest.hexdigest(),
        "ordered_occurrences_sha256": occurrence_digest.hexdigest(),
        "database_path": str(database_path),
        "database_sha256": database_sha256,
        "database_bytes": database_path.stat().st_size,
    }
    receipt_sha256 = sha256(canonical_json(receipt)).hexdigest()
    _write_exclusive(
        receipt_path, canonical_json(receipt | {"receipt_sha256": receipt_sha256}) + b"\n"
    )
    _fsync_directory(root)
    return PTV2OnePassCorpus(
        strategy=strategy,
        occurrence_count=count,
        trainer_epochs=getattr(view, "trainer_epochs", 0),
        assistant_tokens=assistant_tokens,
        tokenizer_sha256=tokenizer_sha256,
        chat_template_sha256=chat_template_sha256,
        assistant_loss_target_sha256=assistant_loss_target_sha256,
        training_config_sha256=training_config_sha256,
        source_response_root_sha256=response_digest.hexdigest(),
        ordered_occurrences_sha256=occurrence_digest.hexdigest(),
        unique_prompt_count=unique,
        natural_duplicate_count=natural,
        constructed_repeat_count=constructed,
        serialized_tokens=serialized_tokens,
        packed_sequences=receipt["packed_sequence_lower_bound"],
        milestone_occurrences=milestone_occurrences,
        milestone_steps=milestone_steps,
        tokenized_path=str(database_path),
        tokenized_sha256=database_sha256,
        receipt_path=str(receipt_path),
        receipt_sha256=receipt_sha256,
    )


def build_ptv2_study_exposures(
    prefix: PTV2OnePassCorpus,
    balanced: PTV2OnePassCorpus,
    *,
    scientific_tokens: int = 256_000_000,
    runtime_screen_tokens: int = 64_000_000,
) -> PTV2StudyExposureViews:
    """Authorize paired PTV2 exposure receipts only within the exact one-pass totals."""
    if scientific_tokens != 256_000_000:
        raise ExposureViewError("PTV2 scientific tokens must be exactly 256M")
    if runtime_screen_tokens != 64_000_000:
        raise ExposureViewError("PTV2 runtime screen must be exactly 64M")
    # Report unreachable science boundaries before requiring local artifacts;
    # this is useful for planning receipts that have not yet been materialized.
    if min(prefix.assistant_tokens, balanced.assistant_tokens) < runtime_screen_tokens:
        raise ExposureViewError("one-pass does not reach 64M runtime screen")
    if min(prefix.assistant_tokens, balanced.assistant_tokens) < scientific_tokens:
        raise ExposureViewError("one-pass does not reach 256M")
    _validate_ptv2_one_pass(prefix, "A-repair")
    _validate_ptv2_one_pass(balanced, "B-balanced")
    identity_fields = (
        "tokenizer_sha256",
        "chat_template_sha256",
        "assistant_loss_target_sha256",
        "training_config_sha256",
    )
    if any(getattr(prefix, field) != getattr(balanced, field) for field in identity_fields):
        raise ExposureViewError(
            "A/B tokenizer, template, target mask, and training config must match"
        )
    runtime = {
        prefix.strategy: _materialize_ptv2_exposure(prefix, runtime_screen_tokens),
        balanced.strategy: _materialize_ptv2_exposure(balanced, runtime_screen_tokens),
    }
    scientific = {
        prefix.strategy: _materialize_ptv2_exposure(prefix, scientific_tokens),
        balanced.strategy: _materialize_ptv2_exposure(balanced, scientific_tokens),
    }
    return PTV2StudyExposureViews(
        runtime_screen_tokens,
        scientific_tokens,
        prefix.assistant_tokens,
        balanced.assistant_tokens,
        True,
        MappingProxyType(runtime),
        MappingProxyType(scientific),
    )


def _materialize_ptv2_exposure(corpus: PTV2OnePassCorpus, target_tokens: int) -> str:
    """Persist an exact mask-trimmed prefix rather than reporting a scalar claim."""
    root = Path(corpus.tokenized_path).parent
    stem = f"{corpus.strategy.lower()}-{target_tokens}-assistant-tokens"
    records_path = root / f"{stem}.jsonl"
    receipt_path = root / f"{stem}.json"
    if os.path.lexists(records_path) or os.path.lexists(receipt_path):
        raise ExposureViewError("PTV2 exposure artifact is immutable and already exists")
    digest = sha256()
    cumulative = 0
    rows = 0
    connection = sqlite3.connect(f"file:{corpus.tokenized_path}?mode=ro", uri=True)
    try:
        with records_path.open("xb") as output:
            for ordinal, ids_json, mask_json, available in connection.execute(
                "SELECT ordinal,input_ids_json,loss_mask_json,assistant_tokens FROM records ORDER BY ordinal"
            ):
                remaining = target_tokens - cumulative
                retained = min(int(available), remaining)
                mask = json.loads(mask_json)
                if retained != int(available):
                    mask = _trim_mask(mask, retained)
                record = {
                    "ordinal": ordinal,
                    "input_ids": json.loads(ids_json),
                    "loss_mask": mask,
                    "assistant_tokens": retained,
                    "cumulative_assistant_tokens": cumulative + retained,
                }
                encoded = canonical_json(record) + b"\n"
                output.write(encoded)
                digest.update(encoded)
                cumulative += retained
                rows += 1
                if cumulative == target_tokens:
                    break
    finally:
        connection.close()
    if cumulative != target_tokens:
        raise ExposureViewError("PTV2 tokenized corpus cannot materialize the exact exposure")
    _fsync_file(records_path)
    receipt = {
        "schema_version": 1,
        "strategy": corpus.strategy,
        "target_assistant_tokens": target_tokens,
        "records_path": str(records_path),
        "records_sha256": digest.hexdigest(),
        "row_count": rows,
        "tokenized_sha256": corpus.tokenized_sha256,
        "source_response_root_sha256": corpus.source_response_root_sha256,
        "tokenizer_sha256": corpus.tokenizer_sha256,
        "chat_template_sha256": corpus.chat_template_sha256,
        "assistant_loss_target_sha256": corpus.assistant_loss_target_sha256,
    }
    receipt_sha256 = sha256(canonical_json(receipt)).hexdigest()
    _write_exclusive(
        receipt_path, canonical_json(receipt | {"receipt_sha256": receipt_sha256}) + b"\n"
    )
    _fsync_directory(root)
    return receipt_sha256


def _validate_ptv2_one_pass(corpus: PTV2OnePassCorpus, strategy: str) -> None:
    if not isinstance(corpus, PTV2OnePassCorpus) or corpus.strategy != strategy:
        raise ExposureViewError(f"PTV2 one-pass receipt must be {strategy}")
    if corpus.occurrence_count != 2_000_000:
        raise ExposureViewError("PTV2 one-pass receipt must bind exactly 2M occurrences")
    if corpus.trainer_epochs != 1:
        raise ExposureViewError("PTV2 one-pass receipt must bind exactly one trainer epoch")
    if isinstance(corpus.assistant_tokens, bool) or corpus.assistant_tokens < 1:
        raise ExposureViewError("PTV2 one-pass assistant tokens must be positive")
    if (
        corpus.unique_prompt_count < 1
        or corpus.natural_duplicate_count < 0
        or corpus.constructed_repeat_count < 0
        or corpus.unique_prompt_count
        + corpus.natural_duplicate_count
        + corpus.constructed_repeat_count
        != corpus.occurrence_count
    ):
        raise ExposureViewError("PTV2 one-pass multiplicity summary does not reconcile")
    if corpus.milestone_occurrences != (
        500_224,
        1_000_448,
        1_300_480,
        2_000_000,
    ) or corpus.milestone_steps != (
        977,
        1_954,
        2_540,
        3_908,
    ):
        raise ExposureViewError("PTV2 one-pass receipt has the wrong optimizer milestones")
    if (
        corpus.segment_occurrences != (1_300_000, 700_000)
        or corpus.segment_steps != (2_540, 1_368)
        or corpus.cumulative_segment_steps != (2_540, 3_908)
        or corpus.segment_final_valid_occurrences != (32, 96)
    ):
        raise ExposureViewError("PTV2 one-pass receipt has the wrong two-segment schedule")
    for field in (
        "tokenizer_sha256",
        "chat_template_sha256",
        "assistant_loss_target_sha256",
        "training_config_sha256",
        "source_response_root_sha256",
        "ordered_occurrences_sha256",
        "tokenized_sha256",
        "receipt_sha256",
    ):
        _require_digest(field, getattr(corpus, field))
    receipt_path = Path(corpus.receipt_path)
    database_path = Path(corpus.tokenized_path)
    if (
        not receipt_path.is_file()
        or receipt_path.is_symlink()
        or not database_path.is_file()
        or database_path.is_symlink()
        or _sha256_file(database_path) != corpus.tokenized_sha256
    ):
        raise ExposureViewError("PTV2 one-pass artifacts are missing or unauthenticated")
    try:
        receipt = json.loads(receipt_path.read_bytes())
    except (OSError, json.JSONDecodeError) as error:
        raise ExposureViewError("PTV2 one-pass receipt is unreadable") from error
    if not isinstance(receipt, dict):
        raise ExposureViewError("PTV2 one-pass receipt is malformed")
    claimed = receipt.pop("receipt_sha256", None)
    if claimed != corpus.receipt_sha256 or claimed != sha256(canonical_json(receipt)).hexdigest():
        raise ExposureViewError("PTV2 one-pass receipt digest mismatch")
    expected = {
        "strategy": corpus.strategy,
        "occurrence_count": corpus.occurrence_count,
        "assistant_tokens": corpus.assistant_tokens,
        "database_sha256": corpus.tokenized_sha256,
        "ordered_occurrences_sha256": corpus.ordered_occurrences_sha256,
    }
    if any(receipt.get(key) != value for key, value in expected.items()):
        raise ExposureViewError("PTV2 one-pass receipt does not match the claimed corpus")


@dataclass(frozen=True)
class _TokenizedCorpus:
    schema_version: int
    arm: str
    corpus_sha256: str
    tokenizer_sha256: str
    chat_template_sha256: str
    selection_sha256: str
    training_sequence_length: int
    purpose: str
    policy_sha256: str
    source_policy_sha256: str
    seed: int
    record_count: int
    total_assistant_tokens: int
    paired_selection_bucket_sha256: str
    task5_paired_selection_bucket_sha256: str
    task6_completion_sha256: str
    final_paired_response_sha256: str
    database_path: str
    database_bytes: int
    database_sha256: str
    resume_fingerprint: str


@dataclass(frozen=True)
class _ProductionProof:
    selection_sha256: str
    task5_paired_sha256: str
    record_paired_sha256: str
    source_policy_sha256: str
    task6_completion_sha256: str
    final_paired_response_sha256: str


@dataclass(frozen=True)
class _AuthenticatedResponse:
    selection_sha256: str
    paired_sha256: str
    index_path: Path
    completion_sha256: str


def _require_digest(name: str, value: str) -> None:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ExposureViewError(f"{name} must be an exact lowercase SHA-256")


def _sha256_file(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_exclusive(path: Path, payload: bytes) -> None:
    with path.open("xb") as output:
        output.write(payload)
        output.flush()
        os.fsync(output.fileno())


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _fsync_file(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _prepare_ptv2_tokenized_database(root: Path, destination: Path, receipt: Path) -> Path:
    """Reserve a private no-follow SQLite staging file without touching the final name."""
    if os.path.lexists(destination) or os.path.lexists(receipt):
        raise ExposureViewError("PTV2 tokenized output is immutable and already exists")
    partials = tuple(root.glob(f".{destination.name}.partial-*"))
    if partials:
        raise ExposureViewError("PTV2 tokenized partial requires explicit recovery")
    temporary = root / f".{destination.name}.partial-{uuid.uuid4().hex}"
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(temporary, flags, 0o600)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    return temporary


def _publish_ptv2_tokenized_database(temporary: Path, destination: Path) -> None:
    """Durably install a private tokenized index without replacing prior evidence."""
    metadata = os.lstat(temporary)
    if not temporary.is_file() or temporary.is_symlink() or metadata.st_nlink != 1:
        raise ExposureViewError("PTV2 tokenized partial is not a private regular file")
    try:
        os.link(temporary, destination, follow_symlinks=False)
    except FileExistsError as error:
        raise ExposureViewError("PTV2 tokenized output is immutable and already exists") from error
    _fsync_directory(destination.parent)
    temporary.unlink()
    _fsync_directory(destination.parent)


def _identity_payload(
    corpus: Any,
    *,
    tokenizer_sha256: str,
    chat_template_sha256: str,
    seed: int,
    training_sequence_length: int,
    purpose: str,
    policy_sha256: str,
    source_policy_sha256: str,
    task5_paired_sha256: str,
    task6_completion_sha256: str,
    final_paired_response_sha256: str,
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "arm": corpus.arm,
        "response_corpus_sha256": corpus.corpus_sha256,
        "source_selection_sha256": corpus.source_selection_sha256,
        "generation_identity_sha256": corpus.generation_identity_sha256,
        "tokenizer_sha256": tokenizer_sha256,
        "chat_template_sha256": chat_template_sha256,
        "seed": seed,
        "training_sequence_length": training_sequence_length,
        "purpose": purpose,
        "policy_sha256": policy_sha256,
        "source_policy_sha256": source_policy_sha256,
        "task5_paired_selection_bucket_sha256": task5_paired_sha256,
        "task6_completion_sha256": task6_completion_sha256,
        "final_paired_response_sha256": final_paired_response_sha256,
    }


def _prepare_root(root: Path, identity: dict[str, Any]) -> str:
    fingerprint = sha256(canonical_json(identity)).hexdigest()
    identity_path = root / "STAGING_IDENTITY.json"
    if root.exists():
        if not root.is_dir() or not identity_path.is_file():
            raise ExposureViewError("existing staging path has no authenticated identity")
        try:
            prior = json.loads(identity_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise ExposureViewError("existing staging identity is malformed") from error
        if prior != identity | {"resume_fingerprint": fingerprint}:
            raise ExposureViewError("resume fingerprint mismatch")
        return fingerprint
    root.mkdir(parents=True)
    _write_exclusive(
        identity_path,
        canonical_json(identity | {"resume_fingerprint": fingerprint}) + b"\n",
    )
    _fsync_directory(root)
    return fingerprint


def _validated_record_sha256(record: Any) -> str:
    try:
        payload = asdict(record)
    except TypeError as error:
        raise ExposureViewError("response corpus contains a non-record value") from error
    claimed = payload.pop("record_sha256", None)
    if not isinstance(claimed, str) or claimed != sha256(canonical_json(payload)).hexdigest():
        raise ExposureViewError(f"promoted record digest mismatch: {record.prompt_uuid}")
    return claimed


def _message_prefix_tokens(
    tokenizer: Any,
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]],
    end: int,
) -> int:
    encoded = tokenizer.apply_chat_template(
        messages[:end],
        tools=tools or None,
        tokenize=True,
        add_generation_prompt=False,
        return_dict=True,
    )
    input_ids = encoded.get("input_ids") if isinstance(encoded, Mapping) else None
    if not isinstance(input_ids, list):
        raise ExposureViewError("tokenizer did not return IDs for a replay boundary")
    return len(input_ids)


def _validate_replay_training_cut(
    tokenizer: Any,
    canonical: dict[str, Any],
    record: Any,
    *,
    training_sequence_length: int,
    full_token_count: int,
) -> None:
    try:
        validate_trajectory(
            canonical,
            source_id=record.prompt_uuid,
            lane=record.lane,
        )
    except TrajectoryValidationError as error:
        raise ExposureViewError(
            f"trainer view rejects replay {record.prompt_uuid}: {error.reason}"
        ) from error
    if full_token_count <= training_sequence_length:
        return
    messages = canonical["messages"]
    tools = canonical.get("tools") or []
    lower = 1
    upper = len(messages)
    while lower < upper:
        middle = (lower + upper) // 2
        if _message_prefix_tokens(tokenizer, messages, tools, middle) > training_sequence_length:
            upper = middle
        else:
            lower = middle + 1
    cut_message = lower - 1
    pending: set[str] = set()
    transaction_start = -1
    for index, message in enumerate(messages):
        calls = message.get("tool_calls") if message.get("role") == "assistant" else None
        if isinstance(calls, list) and calls:
            transaction_start = index
            pending = {
                str(call.get("id") or call.get("tool_call_id"))
                for call in calls
                if isinstance(call, dict)
            }
        if message.get("role") != "tool":
            continue
        pending.remove(str(message.get("tool_call_id") or message.get("id")))
        if pending or not transaction_start <= cut_message <= index:
            continue
        before = _message_prefix_tokens(tokenizer, messages, tools, transaction_start)
        after = _message_prefix_tokens(tokenizer, messages, tools, index + 1)
        if before < training_sequence_length < after:
            raise ExposureViewError(
                f"trainer view rejects replay {record.prompt_uuid}: split_tool_transaction"
            )


def _tokenize_record(
    tokenizer: Any,
    record: Any,
    *,
    training_sequence_length: int,
) -> tuple[list[int], list[int]]:
    try:
        canonical = json.loads(record.canonical_record_json)
    except json.JSONDecodeError as error:
        raise ExposureViewError(f"record is not canonical JSON: {record.prompt_uuid}") from error
    if canonical_json(canonical).decode("utf-8") != record.canonical_record_json:
        raise ExposureViewError(f"record is not canonically encoded: {record.prompt_uuid}")
    if not isinstance(canonical, dict) or not isinstance(canonical.get("messages"), list):
        raise ExposureViewError(f"record has no canonical messages: {record.prompt_uuid}")
    encoded = tokenizer.apply_chat_template(
        canonical["messages"],
        tools=canonical.get("tools") or None,
        tokenize=True,
        add_generation_prompt=False,
        return_dict=True,
        return_assistant_tokens_mask=True,
    )
    if not isinstance(encoded, Mapping):
        raise ExposureViewError("tokenizer did not return a mapping")
    input_ids = encoded.get("input_ids")
    loss_mask = encoded.get("assistant_masks")
    if loss_mask is None:
        loss_mask = encoded.get("assistant_tokens_mask")
    if (
        not isinstance(input_ids, list)
        or not input_ids
        or not all(isinstance(token, int) and not isinstance(token, bool) for token in input_ids)
        or not isinstance(loss_mask, list)
        or len(input_ids) != len(loss_mask)
        or any(value not in (0, 1) for value in loss_mask)
    ):
        raise ExposureViewError("tokenizer did not return aligned IDs and a binary assistant mask")
    if record.lane in {"interactive-swe-replay", "generic-tool-replay"}:
        _validate_replay_training_cut(
            tokenizer,
            canonical,
            record,
            training_sequence_length=training_sequence_length,
            full_token_count=len(input_ids),
        )
    input_ids = input_ids[:training_sequence_length]
    loss_mask = loss_mask[:training_sequence_length]
    if sum(loss_mask) < 1:
        raise ExposureViewError(
            f"record has no assistant-owned tokens after exact retokenization: {record.prompt_uuid}"
        )
    return input_ids, loss_mask


def _load_tokenized_receipt(root: Path, resume_fingerprint: str) -> _TokenizedCorpus | None:
    receipt_path = root / "TOKENIZED.json"
    database_path = root / "tokenized.sqlite3"
    if not receipt_path.exists():
        if database_path.exists():
            raise ExposureViewError("incomplete tokenized staging state requires recovery")
        return None
    try:
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        receipt_sha256 = receipt.pop("receipt_sha256")
    except (OSError, json.JSONDecodeError, KeyError) as error:
        raise ExposureViewError("tokenized staging receipt is malformed") from error
    if (
        receipt_sha256 != sha256(canonical_json(receipt)).hexdigest()
        or receipt.get("resume_fingerprint") != resume_fingerprint
        or receipt.get("database_path") != str(database_path)
        or not database_path.is_file()
        or receipt.get("database_bytes") != database_path.stat().st_size
        or receipt.get("database_sha256") != _sha256_file(database_path)
    ):
        raise ExposureViewError("tokenized staging authentication failed")
    try:
        return _TokenizedCorpus(**receipt)
    except TypeError as error:
        raise ExposureViewError("tokenized staging receipt has an invalid schema") from error


def _stage_tokenized_corpus(
    corpus: Any,
    tokenizer: Any,
    root: Path,
    *,
    tokenizer_sha256: str,
    chat_template_sha256: str,
    seed: int,
    training_sequence_length: int,
    purpose: str,
    policy_sha256: str,
    source_policy_sha256: str,
    task5_paired_sha256: str,
    task6_completion_sha256: str,
    final_paired_response_sha256: str,
    resume_fingerprint: str,
    reconcile_resume: bool,
) -> _TokenizedCorpus:
    resumed = _load_tokenized_receipt(root, resume_fingerprint)
    if resumed is not None:
        expected_identity = (
            corpus.arm,
            corpus.corpus_sha256,
            tokenizer_sha256,
            chat_template_sha256,
            seed,
            training_sequence_length,
            purpose,
            policy_sha256,
            source_policy_sha256,
            task5_paired_sha256,
            task6_completion_sha256,
            final_paired_response_sha256,
        )
        actual_identity = (
            resumed.arm,
            resumed.corpus_sha256,
            resumed.tokenizer_sha256,
            resumed.chat_template_sha256,
            resumed.seed,
            resumed.training_sequence_length,
            resumed.purpose,
            resumed.policy_sha256,
            resumed.source_policy_sha256,
            resumed.task5_paired_selection_bucket_sha256,
            resumed.task6_completion_sha256,
            resumed.final_paired_response_sha256,
        )
        if actual_identity != expected_identity:
            raise ExposureViewError("tokenized staging identity reconciliation failed")
        if reconcile_resume:
            _reconcile_tokenized_corpus(
                resumed,
                corpus,
                tokenizer,
                training_sequence_length=training_sequence_length,
            )
        return resumed
    database_path = root / "tokenized.sqlite3"
    connection = sqlite3.connect(database_path)
    total_tokens = 0
    record_count = 0
    paired_digest: str | None = None
    selection_digest: str | None = None
    try:
        connection.execute("PRAGMA synchronous=FULL")
        connection.execute(
            """
            CREATE TABLE records(
                source_ordinal INTEGER PRIMARY KEY,
                prompt_uuid TEXT NOT NULL UNIQUE,
                domain TEXT NOT NULL,
                lane TEXT NOT NULL,
                context_bucket TEXT NOT NULL,
                input_ids_json TEXT NOT NULL,
                loss_mask_json TEXT NOT NULL,
                assistant_tokens INTEGER NOT NULL,
                promoted_record_sha256 TEXT NOT NULL
            )
            """
        )
        for ordinal, record in enumerate(corpus.records):
            record_sha256 = _validated_record_sha256(record)
            if (
                record.arm != corpus.arm
                or record.generation_identity_sha256 != corpus.generation_identity_sha256
            ):
                raise ExposureViewError(f"record provenance mismatch: {record.prompt_uuid}")
            _require_digest("selection digest", record.selection_sha256)
            if selection_digest is None:
                selection_digest = record.selection_sha256
            elif selection_digest != record.selection_sha256:
                raise ExposureViewError("record selection digest is not uniform")
            _require_digest("paired C/D digest", record.paired_cd_sha256)
            if corpus.arm in {"C", "D"}:
                if record.paired_cd_sha256 == "0" * 64:
                    raise ExposureViewError("C/D record has no paired selection/bucket proof")
                if paired_digest is None:
                    paired_digest = record.paired_cd_sha256
                elif paired_digest != record.paired_cd_sha256:
                    raise ExposureViewError("C/D paired selection/bucket proof is not uniform")
            input_ids, loss_mask = _tokenize_record(
                tokenizer,
                record,
                training_sequence_length=training_sequence_length,
            )
            assistant_tokens = sum(loss_mask)
            connection.execute(
                "INSERT INTO records VALUES(?,?,?,?,?,?,?,?,?)",
                (
                    ordinal,
                    record.prompt_uuid,
                    record.domain,
                    record.lane,
                    record.context_bucket,
                    canonical_json(input_ids).decode("utf-8"),
                    canonical_json(loss_mask).decode("utf-8"),
                    assistant_tokens,
                    record_sha256,
                ),
            )
            total_tokens += assistant_tokens
            record_count += 1
        if record_count == 0:
            raise ExposureViewError("response corpus contains no promoted records")
        connection.commit()
    except Exception:
        connection.close()
        raise
    finally:
        if connection:
            connection.close()
    database_bytes = database_path.stat().st_size
    database_sha256 = _sha256_file(database_path)
    tokenized = _TokenizedCorpus(
        schema_version=1,
        arm=corpus.arm,
        corpus_sha256=corpus.corpus_sha256,
        tokenizer_sha256=tokenizer_sha256,
        chat_template_sha256=chat_template_sha256,
        selection_sha256=selection_digest or "0" * 64,
        training_sequence_length=training_sequence_length,
        purpose=purpose,
        policy_sha256=policy_sha256,
        source_policy_sha256=source_policy_sha256,
        seed=seed,
        record_count=record_count,
        total_assistant_tokens=total_tokens,
        paired_selection_bucket_sha256=paired_digest or "0" * 64,
        task5_paired_selection_bucket_sha256=task5_paired_sha256,
        task6_completion_sha256=task6_completion_sha256,
        final_paired_response_sha256=final_paired_response_sha256,
        database_path=str(database_path),
        database_bytes=database_bytes,
        database_sha256=database_sha256,
        resume_fingerprint=resume_fingerprint,
    )
    receipt = asdict(tokenized)
    receipt_sha256 = sha256(canonical_json(receipt)).hexdigest()
    _write_exclusive(
        root / "TOKENIZED.json",
        canonical_json(receipt | {"receipt_sha256": receipt_sha256}) + b"\n",
    )
    _fsync_directory(root)
    return tokenized


def _reconcile_tokenized_corpus(
    tokenized: _TokenizedCorpus,
    corpus: Any,
    tokenizer: Any,
    *,
    training_sequence_length: int,
) -> None:
    total_tokens = 0
    record_count = 0
    selection_digest: str | None = None
    paired_digest: str | None = None
    records = iter(corpus.records)
    connection = sqlite3.connect(f"file:{tokenized.database_path}?mode=ro", uri=True)
    try:
        query = """
            SELECT source_ordinal,prompt_uuid,domain,lane,context_bucket,input_ids_json,
                   loss_mask_json,assistant_tokens,promoted_record_sha256
            FROM records ORDER BY source_ordinal
        """
        for database_row in connection.execute(query):
            try:
                record = next(records)
            except StopIteration as error:
                raise ExposureViewError(
                    "tokenized corpus reconciliation found an extra row"
                ) from error
            record_sha256 = _validated_record_sha256(record)
            input_ids, loss_mask = _tokenize_record(
                tokenizer,
                record,
                training_sequence_length=training_sequence_length,
            )
            if (
                int(database_row[0]) != record_count
                or database_row[1] != record.prompt_uuid
                or database_row[2] != record.domain
                or database_row[3] != record.lane
                or database_row[4] != record.context_bucket
                or json.loads(database_row[5]) != input_ids
                or json.loads(database_row[6]) != loss_mask
                or int(database_row[7]) != sum(loss_mask)
                or database_row[8] != record_sha256
            ):
                raise ExposureViewError("tokenized corpus reconciliation failed")
            selection_digest = selection_digest or record.selection_sha256
            if selection_digest != record.selection_sha256:
                raise ExposureViewError("tokenized corpus selection reconciliation failed")
            if corpus.arm in {"C", "D"}:
                paired_digest = paired_digest or record.paired_cd_sha256
                if paired_digest != record.paired_cd_sha256:
                    raise ExposureViewError("tokenized corpus pair reconciliation failed")
            total_tokens += sum(loss_mask)
            record_count += 1
        try:
            next(records)
        except StopIteration:
            pass
        else:
            raise ExposureViewError("tokenized corpus reconciliation found a missing row")
    finally:
        connection.close()
    expected = (
        record_count,
        total_tokens,
        selection_digest or "0" * 64,
        paired_digest or "0" * 64,
    )
    actual = (
        tokenized.record_count,
        tokenized.total_assistant_tokens,
        tokenized.selection_sha256,
        tokenized.paired_selection_bucket_sha256,
    )
    if actual != expected:
        raise ExposureViewError("tokenized corpus receipt reconciliation failed")


def _trim_mask(loss_mask: list[int], keep: int) -> list[int]:
    remaining = keep
    trimmed: list[int] = []
    for value in loss_mask:
        retained = int(value == 1 and remaining > 0)
        trimmed.append(retained)
        remaining -= retained
    if remaining != 0:
        raise ExposureViewError("final assistant mask cannot satisfy the exact boundary")
    return trimmed


def _epoch_rank(
    corpus_sha256: str,
    epoch: int,
    domain: str,
    lane: str,
    prompt_uuid: str,
    seed: int,
) -> str:
    identity = "\0".join((corpus_sha256, str(epoch), domain, lane, prompt_uuid, str(seed)))
    return sha256(identity.encode()).hexdigest()


def _digest_item(digest: Any, value: Any) -> None:
    digest.update(canonical_json(value))
    digest.update(b"\n")


def _reconcile_view(view: ExposureView, tokenized: _TokenizedCorpus) -> None:
    if (
        view.schema_version,
        view.arm,
        view.response_corpus_sha256,
        view.tokenizer_sha256,
        view.chat_template_sha256,
        view.selection_sha256,
        view.training_sequence_length,
        view.purpose,
        view.policy_sha256,
        view.source_policy_sha256,
        view.seed,
        view.paired_selection_bucket_sha256,
        view.task5_paired_selection_bucket_sha256,
        view.task6_completion_sha256,
        view.final_paired_response_sha256,
        view.resume_fingerprint,
    ) != (
        1,
        tokenized.arm,
        tokenized.corpus_sha256,
        tokenized.tokenizer_sha256,
        tokenized.chat_template_sha256,
        tokenized.selection_sha256,
        tokenized.training_sequence_length,
        tokenized.purpose,
        tokenized.policy_sha256,
        tokenized.source_policy_sha256,
        tokenized.seed,
        tokenized.paired_selection_bucket_sha256,
        tokenized.task5_paired_selection_bucket_sha256,
        tokenized.task6_completion_sha256,
        tokenized.final_paired_response_sha256,
        tokenized.resume_fingerprint,
    ):
        raise ExposureViewError("exposure receipt identity reconciliation failed")
    records_digest = sha256()
    row_order_digest = sha256()
    input_ids_digest = sha256()
    loss_masks_digest = sha256()
    cumulative_digest = sha256()
    cumulative = 0
    row_count = 0
    previous_epoch = -1
    rows_in_epoch = 0
    previous_rank = ""
    partial_seen = False
    required = {
        "view_ordinal",
        "epoch",
        "source_ordinal",
        "prompt_uuid",
        "domain",
        "lane",
        "context_bucket",
        "input_ids",
        "loss_mask",
        "assistant_tokens",
        "cumulative_assistant_tokens",
        "promoted_record_sha256",
    }
    connection = sqlite3.connect(f"file:{tokenized.database_path}?mode=ro", uri=True)
    try:
        with Path(view.records_path).open("rb") as source:
            for raw_line in source:
                records_digest.update(raw_line)
                try:
                    row = json.loads(raw_line)
                except json.JSONDecodeError as error:
                    raise ExposureViewError(
                        "exposure row reconciliation found invalid JSON"
                    ) from error
                if set(row) != required or canonical_json(row) + b"\n" != raw_line:
                    raise ExposureViewError("exposure row reconciliation found a schema mismatch")
                if partial_seen:
                    raise ExposureViewError("exposure row reconciliation found data after a trim")
                epoch = row["epoch"]
                source_ordinal = row["source_ordinal"]
                if (
                    not isinstance(epoch, int)
                    or isinstance(epoch, bool)
                    or not isinstance(source_ordinal, int)
                    or isinstance(source_ordinal, bool)
                    or row["view_ordinal"] != row_count
                ):
                    raise ExposureViewError("exposure row reconciliation found invalid ordering")
                if epoch != previous_epoch:
                    if previous_epoch >= 0 and rows_in_epoch != tokenized.record_count:
                        raise ExposureViewError("exposure row reconciliation skipped an epoch row")
                    if epoch != previous_epoch + 1:
                        raise ExposureViewError("exposure row reconciliation skipped an epoch")
                    previous_epoch = epoch
                    rows_in_epoch = 0
                    previous_rank = ""
                expected = connection.execute(
                    "SELECT prompt_uuid,domain,lane,context_bucket,input_ids_json,loss_mask_json,"
                    "promoted_record_sha256 FROM records WHERE source_ordinal=?",
                    (source_ordinal,),
                ).fetchone()
                if expected is None:
                    raise ExposureViewError("exposure row reconciliation found an unknown source")
                input_ids = json.loads(expected[4])
                full_mask = json.loads(expected[5])
                loss_mask = row["loss_mask"]
                rank = _epoch_rank(
                    tokenized.corpus_sha256,
                    epoch,
                    str(expected[1]),
                    str(expected[2]),
                    str(expected[0]),
                    tokenized.seed,
                )
                if rank <= previous_rank:
                    raise ExposureViewError(
                        "exposure row reconciliation found a shuffled-order error"
                    )
                previous_rank = rank
                retained = sum(loss_mask) if isinstance(loss_mask, list) else -1
                if (
                    row["prompt_uuid"] != expected[0]
                    or row["domain"] != expected[1]
                    or row["lane"] != expected[2]
                    or row["context_bucket"] != expected[3]
                    or row["input_ids"] != input_ids
                    or row["promoted_record_sha256"] != expected[6]
                    or not isinstance(loss_mask, list)
                    or len(loss_mask) != len(input_ids)
                    or any(value not in (0, 1) for value in loss_mask)
                    or row["assistant_tokens"] != retained
                ):
                    raise ExposureViewError("exposure row reconciliation found a content mismatch")
                if loss_mask != full_mask:
                    if retained < 1 or loss_mask != _trim_mask(full_mask, retained):
                        raise ExposureViewError("exposure row reconciliation found an invalid trim")
                    partial_seen = True
                cumulative += retained
                if row["cumulative_assistant_tokens"] != cumulative:
                    raise ExposureViewError(
                        "exposure row reconciliation found a cumulative mismatch"
                    )
                _digest_item(
                    row_order_digest, [row_count, epoch, row["prompt_uuid"], source_ordinal]
                )
                _digest_item(input_ids_digest, [row_count, input_ids])
                _digest_item(loss_masks_digest, [row_count, loss_mask])
                _digest_item(cumulative_digest, [row_count, cumulative])
                row_count += 1
                rows_in_epoch += 1
    finally:
        connection.close()
    actual = (
        records_digest.hexdigest(),
        row_order_digest.hexdigest(),
        input_ids_digest.hexdigest(),
        loss_masks_digest.hexdigest(),
        cumulative_digest.hexdigest(),
        cumulative,
        row_count,
        min(row_count, tokenized.record_count),
        row_count - min(row_count, tokenized.record_count),
        cumulative // tokenized.total_assistant_tokens,
    )
    declared = (
        view.records_sha256,
        view.row_order_sha256,
        view.input_ids_sha256,
        view.loss_masks_sha256,
        view.cumulative_tokens_sha256,
        view.assistant_tokens,
        view.row_count,
        view.unique_prompt_count,
        view.repeated_prompt_count,
        view.completed_epochs,
    )
    if actual != declared:
        raise ExposureViewError("exposure receipt reconciliation failed")


def _load_view(root: Path, name: str, tokenized: _TokenizedCorpus) -> ExposureView | None:
    receipt_path = root / f"{name}.json"
    records_path = root / f"{name}.jsonl"
    if not receipt_path.exists():
        if records_path.exists():
            raise ExposureViewError(f"incomplete {name!r} exposure staging requires recovery")
        return None
    try:
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        receipt_sha256 = receipt.pop("receipt_sha256")
    except (OSError, json.JSONDecodeError, KeyError) as error:
        raise ExposureViewError(f"{name!r} exposure receipt is malformed") from error
    if (
        receipt_sha256 != sha256(canonical_json(receipt)).hexdigest()
        or receipt.get("schema_version") != 1
        or receipt.get("name") != name
        or receipt.get("resume_fingerprint") != tokenized.resume_fingerprint
        or receipt.get("receipt_path") != str(receipt_path)
        or receipt.get("records_path") != str(records_path)
        or not records_path.is_file()
        or receipt.get("records_bytes") != records_path.stat().st_size
        or receipt.get("records_sha256") != _sha256_file(records_path)
    ):
        raise ExposureViewError(f"{name!r} exposure staging authentication failed")
    try:
        view = ExposureView(**receipt, receipt_sha256=receipt_sha256)
    except TypeError as error:
        raise ExposureViewError(f"{name!r} exposure receipt has an invalid schema") from error
    _reconcile_view(view, tokenized)
    return view


def _build_view(
    tokenized: _TokenizedCorpus,
    root: Path,
    *,
    name: str,
    target_tokens: int,
) -> ExposureView:
    if _SAFE_NAME.fullmatch(name) is None:
        raise ExposureViewError(f"unsafe exposure view name: {name!r}")
    if not isinstance(target_tokens, int) or isinstance(target_tokens, bool) or target_tokens < 1:
        raise ExposureViewError("assistant-token boundary must be a positive integer")
    resumed = _load_view(root, name, tokenized)
    if resumed is not None:
        if resumed.assistant_tokens != target_tokens:
            raise ExposureViewError(f"{name!r} exposure boundary mismatch")
        return resumed

    records_path = root / f"{name}.jsonl"
    receipt_path = root / f"{name}.json"
    records_digest = sha256()
    row_order_digest = sha256()
    input_ids_digest = sha256()
    loss_masks_digest = sha256()
    cumulative_digest = sha256()
    cumulative = 0
    row_count = 0
    epoch = 0
    connection = sqlite3.connect(f"file:{tokenized.database_path}?mode=ro", uri=True)
    connection.execute("PRAGMA temp_store=FILE")
    try:
        connection.create_function(
            "epoch_rank",
            3,
            lambda domain, lane, prompt_uuid: _epoch_rank(
                tokenized.corpus_sha256,
                epoch,
                str(domain),
                str(lane),
                str(prompt_uuid),
                tokenized.seed,
            ),
            deterministic=True,
        )
        with records_path.open("xb") as output:
            while cumulative < target_tokens:
                rows_this_epoch = 0
                query = """
                    SELECT source_ordinal,prompt_uuid,domain,lane,context_bucket,
                           input_ids_json,loss_mask_json,assistant_tokens,promoted_record_sha256
                    FROM records
                    ORDER BY epoch_rank(domain,lane,prompt_uuid),prompt_uuid
                """
                for row in connection.execute(query):
                    remaining = target_tokens - cumulative
                    input_ids = json.loads(row[5])
                    full_mask = json.loads(row[6])
                    retained = min(int(row[7]), remaining)
                    loss_mask = (
                        full_mask if retained == int(row[7]) else _trim_mask(full_mask, retained)
                    )
                    cumulative += retained
                    record = {
                        "view_ordinal": row_count,
                        "epoch": epoch,
                        "source_ordinal": int(row[0]),
                        "prompt_uuid": row[1],
                        "domain": row[2],
                        "lane": row[3],
                        "context_bucket": row[4],
                        "input_ids": input_ids,
                        "loss_mask": loss_mask,
                        "assistant_tokens": retained,
                        "cumulative_assistant_tokens": cumulative,
                        "promoted_record_sha256": row[8],
                    }
                    encoded = canonical_json(record) + b"\n"
                    output.write(encoded)
                    records_digest.update(encoded)
                    _digest_item(row_order_digest, [row_count, epoch, row[1], int(row[0])])
                    _digest_item(input_ids_digest, [row_count, input_ids])
                    _digest_item(loss_masks_digest, [row_count, loss_mask])
                    _digest_item(cumulative_digest, [row_count, cumulative])
                    row_count += 1
                    rows_this_epoch += 1
                    if cumulative == target_tokens:
                        break
                if rows_this_epoch == 0:
                    raise ExposureViewError("tokenized corpus cannot advance exposure")
                if cumulative < target_tokens:
                    epoch += 1
            output.flush()
            os.fsync(output.fileno())
    finally:
        connection.close()

    unique_prompt_count = min(row_count, tokenized.record_count)
    view_without_receipt: dict[str, Any] = {
        "schema_version": 1,
        "name": name,
        "arm": tokenized.arm,
        "response_corpus_sha256": tokenized.corpus_sha256,
        "tokenizer_sha256": tokenized.tokenizer_sha256,
        "chat_template_sha256": tokenized.chat_template_sha256,
        "selection_sha256": tokenized.selection_sha256,
        "training_sequence_length": tokenized.training_sequence_length,
        "purpose": tokenized.purpose,
        "policy_sha256": tokenized.policy_sha256,
        "source_policy_sha256": tokenized.source_policy_sha256,
        "seed": tokenized.seed,
        "assistant_tokens": cumulative,
        "full_prompt_assistant_tokens": tokenized.total_assistant_tokens,
        "row_count": row_count,
        "unique_prompt_count": unique_prompt_count,
        "repeated_prompt_count": row_count - unique_prompt_count,
        "completed_epochs": target_tokens // tokenized.total_assistant_tokens,
        "records_path": str(records_path),
        "records_bytes": records_path.stat().st_size,
        "records_sha256": records_digest.hexdigest(),
        "row_order_sha256": row_order_digest.hexdigest(),
        "input_ids_sha256": input_ids_digest.hexdigest(),
        "loss_masks_sha256": loss_masks_digest.hexdigest(),
        "cumulative_tokens_sha256": cumulative_digest.hexdigest(),
        "paired_selection_bucket_sha256": tokenized.paired_selection_bucket_sha256,
        "task5_paired_selection_bucket_sha256": tokenized.task5_paired_selection_bucket_sha256,
        "task6_completion_sha256": tokenized.task6_completion_sha256,
        "final_paired_response_sha256": tokenized.final_paired_response_sha256,
        "resume_fingerprint": tokenized.resume_fingerprint,
        "receipt_path": str(receipt_path),
    }
    receipt_sha256 = sha256(canonical_json(view_without_receipt)).hexdigest()
    _write_exclusive(
        receipt_path,
        canonical_json(view_without_receipt | {"receipt_sha256": receipt_sha256}) + b"\n",
    )
    _fsync_directory(root)
    return ExposureView(**view_without_receipt, receipt_sha256=receipt_sha256)


def _canonical_document(path: Path, description: str) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise ExposureViewError(f"{description} is not a regular file")
    raw = path.read_bytes()
    try:
        value = json.loads(raw)
    except json.JSONDecodeError as error:
        raise ExposureViewError(f"{description} is not JSON") from error
    if not isinstance(value, dict) or canonical_json(value) + b"\n" != raw:
        raise ExposureViewError(f"{description} is not canonical JSON")
    return value


def _declared_file(root: Path, descriptor: Any, description: str) -> Path:
    if not isinstance(descriptor, dict) or not isinstance(descriptor.get("path"), str):
        raise ExposureViewError(f"{description} descriptor is malformed")
    root_resolved = root.resolve(strict=True)
    unresolved = root / descriptor["path"]
    if unresolved.is_symlink():
        raise ExposureViewError(f"{description} authentication failed")
    path = unresolved.resolve(strict=True)
    expected_size = descriptor.get("bytes", descriptor.get("byte_count"))
    if (
        not path.is_relative_to(root_resolved)
        or not path.is_file()
        or (expected_size is not None and path.stat().st_size != expected_size)
        or _sha256_file(path) != descriptor.get("sha256")
    ):
        raise ExposureViewError(f"{description} authentication failed")
    return path


def _authenticate_selection_manifest(
    manifest_path: Path,
    *,
    expected_manifest_sha256: str,
) -> tuple[str, str, str]:
    if _sha256_file(manifest_path) != expected_manifest_sha256:
        raise ExposureViewError("Task 5 selection manifest identity mismatch")
    manifest = _canonical_document(manifest_path, "Task 5 selection manifest")
    root_record = {key: value for key, value in manifest.items() if key != "root_sha256"}
    if (
        manifest.get("schema_version") != 2
        or manifest.get("root_sha256") != sha256(canonical_json(root_record)).hexdigest()
    ):
        raise ExposureViewError("Task 5 selection root authentication failed")
    root = manifest_path.parent
    shard_rows = 0
    for descriptor in manifest.get("shards", []):
        _declared_file(root, descriptor, "Task 5 selection shard")
        if not isinstance(descriptor.get("row_count"), int):
            raise ExposureViewError("Task 5 selection shard row count is malformed")
        shard_rows += descriptor["row_count"]
    index_path = _declared_file(root, manifest.get("index"), "Task 5 selection index")
    arms = manifest.get("arms")
    identity = manifest.get("identity")
    if not isinstance(arms, dict) or not isinstance(identity, dict):
        raise ExposureViewError("Task 5 selection identity is incomplete")
    connection = sqlite3.connect(f"file:{index_path}?mode=ro", uri=True)
    try:
        row_count = int(connection.execute("SELECT count(*) FROM rows").fetchone()[0])
        if row_count != manifest.get("row_count") or row_count != shard_rows:
            raise ExposureViewError("Task 5 selection row count reconciliation failed")
        for left, right in (("C", "D"), ("D", "C")):
            mismatch = int(
                connection.execute(
                    "SELECT count(*) FROM ("
                    "SELECT prompt_uuid,status FROM rows WHERE arm=? AND domain!='swe-agentic-tool' "
                    "EXCEPT SELECT prompt_uuid,status FROM rows "
                    "WHERE arm=? AND domain!='swe-agentic-tool')",
                    (left, right),
                ).fetchone()[0]
            )
            if mismatch:
                raise ExposureViewError("Task 5 C/D paired UUID proof mismatch")
        try:
            paired_seed = {
                "bucket_floors": arms["C"]["non_agentic_bucket_floors"],
                "C_count_proof_sha256": arms["C"]["count_proof_sha256"],
                "D_count_proof_sha256": arms["D"]["count_proof_sha256"],
            }
        except (KeyError, TypeError) as error:
            raise ExposureViewError("Task 5 paired bucket proof is incomplete") from error
        paired = sha256(canonical_json(paired_seed))
        for row in connection.execute(
            "SELECT status,prompt_uuid FROM rows "
            "WHERE arm='C' AND domain!='swe-agentic-tool' ORDER BY status,prompt_uuid"
        ):
            _digest_item(paired, list(row))
        paired_sha256 = paired.hexdigest()
        if paired_sha256 != manifest.get("paired_cd_sha256"):
            raise ExposureViewError("Task 5 paired selection/bucket proof mismatch")
        metadata = {
            "schema_version": 2,
            "policy_sha256": identity.get("policy_sha256"),
            "seed": identity.get("seed"),
            "source_inventory_sha256": identity.get("source_inventory_sha256"),
            "baseline_receipt_sha256": identity.get("baseline_receipt_sha256"),
            "held_out_receipt_sha256": identity.get("held_out_receipt_sha256"),
            "ptv2_revision": identity.get("ptv2_revision"),
            "ptv2_allowlist_sha256": identity.get("ptv2_allowlist_sha256"),
            "paired_cd_sha256": paired_sha256,
            "arms": arms,
        }
        selection = sha256(canonical_json(metadata))
        for row in connection.execute(
            "SELECT arm,status,selection_index,prompt_uuid FROM rows "
            "ORDER BY arm,status,selection_index"
        ):
            _digest_item(selection, list(row))
        selection_sha256 = selection.hexdigest()
        if selection_sha256 != manifest.get("selection_sha256"):
            raise ExposureViewError("Task 5 selection stream proof mismatch")
    finally:
        connection.close()
    source_policy_sha256 = str(identity.get("policy_sha256"))
    _require_digest("Task 5 source policy", source_policy_sha256)
    return selection_sha256, paired_sha256, source_policy_sha256


def _authenticate_generation_identity(corpus: Any) -> dict[str, Any]:
    try:
        payload = asdict(corpus.generation_identity)
    except TypeError as error:
        raise ExposureViewError("generation identity must be a frozen Task 6 record") from error
    required = {
        "target_revision",
        "tokenizer_sha256",
        "chat_template_sha256",
        "runtime_sha256",
        "container_sha256",
        "source_selection_sha256",
        "thinking_mode",
        "temperature",
        "max_tokens",
        "max_total_length",
    }
    if set(payload) != required:
        raise ExposureViewError("generation identity schema mismatch")
    actual = sha256(canonical_json(payload)).hexdigest()
    if actual != corpus.generation_identity_sha256:
        raise ExposureViewError("generation identity digest mismatch")
    if payload["source_selection_sha256"] != corpus.source_selection_sha256:
        raise ExposureViewError("generation identity source selection mismatch")
    return payload


def _authenticate_task6_completion(root: Path, *, expected_sha256: str) -> dict[str, Any]:
    _require_digest("expected Task 6 completion", expected_sha256)
    completion_path = root.parent / "completion.json"
    if (
        completion_path.is_symlink()
        or not completion_path.is_file()
        or _sha256_file(completion_path) != expected_sha256
    ):
        raise ExposureViewError("Task 6 completion identity mismatch")
    try:
        completion = json.loads(completion_path.read_bytes())
    except json.JSONDecodeError as error:
        raise ExposureViewError("Task 6 completion is not JSON") from error
    if not isinstance(completion, dict):
        raise ExposureViewError("Task 6 completion is malformed")
    return completion


def _authenticate_response_corpus(
    corpus: Any,
    root: Path,
    *,
    expected_completion_sha256: str,
) -> _AuthenticatedResponse:
    completion = _authenticate_task6_completion(root, expected_sha256=expected_completion_sha256)
    receipt = _canonical_document(root / "PROMOTION.json", "Task 6 promotion receipt")
    promotion_sha256 = _sha256_file(root / "PROMOTION.json")
    identity = _authenticate_generation_identity(corpus)
    completion_expected = {
        "generation_identity_sha256": corpus.generation_identity_sha256,
        "target_revision": identity["target_revision"],
        "tokenizer_sha256": identity["tokenizer_sha256"],
        "chat_template_sha256": identity["chat_template_sha256"],
        "runtime_sha256": identity["runtime_sha256"],
        "container_sha256": identity["container_sha256"],
        "source_selection_sha256": identity["source_selection_sha256"],
        "thinking_mode": identity["thinking_mode"],
        "promotion_status": "passed",
        "corpus_sha256": corpus.corpus_sha256,
        "promotion_receipt_sha256": promotion_sha256,
    }
    if any(completion.get(key) != value for key, value in completion_expected.items()):
        raise ExposureViewError("Task 6 completion reconciliation failed")
    completion_files = completion.get("files")
    if not isinstance(completion_files, list) or completion.get("file_count") != len(
        completion_files
    ):
        raise ExposureViewError("Task 6 completion file manifest is malformed")
    for descriptor in completion_files:
        _declared_file(root.parent, descriptor, "Task 6 published file")
    for name, expected in (
        ("corpus_sha256", corpus.corpus_sha256),
        ("generation_identity_sha256", corpus.generation_identity_sha256),
        ("source_selection_sha256", corpus.source_selection_sha256),
    ):
        if receipt.get(name) != expected:
            raise ExposureViewError(f"Task 6 {name} mismatch")
    descriptors = receipt.get("files")
    if not isinstance(descriptors, list):
        raise ExposureViewError("Task 6 promotion file manifest is missing")
    files = {
        descriptor.get("path"): descriptor
        for descriptor in descriptors
        if isinstance(descriptor, dict)
    }
    response_path = _declared_file(root, files.get("responses.jsonl"), "Task 6 response stream")
    index_path = _declared_file(root, files.get("response-index.sqlite3"), "Task 6 response index")
    for descriptor in descriptors:
        _declared_file(root, descriptor, "Task 6 promoted file")

    promoted_digest = sha256()
    cell_counts: Counter[str] = Counter()
    histograms: dict[str, Counter[int]] = {}
    records = iter(corpus.records)
    promoted_count = 0
    connection = sqlite3.connect(f"file:{index_path}?mode=ro", uri=True)
    try:
        with response_path.open("rb") as response_stream:
            query = "SELECT ordinal,prompt_uuid,cell,assistant_tokens,payload FROM promoted ORDER BY ordinal"
            for ordinal, prompt_uuid, cell, assistant_tokens, payload in connection.execute(query):
                try:
                    record = next(records)
                except StopIteration as error:
                    raise ExposureViewError("Task 6 corpus has an extra indexed record") from error
                line = response_stream.readline()
                expected_payload = canonical_json(asdict(record)).decode("utf-8")
                if (
                    ordinal != promoted_count
                    or prompt_uuid != record.prompt_uuid
                    or payload != expected_payload
                    or line != payload.encode() + b"\n"
                ):
                    raise ExposureViewError("Task 6 promoted record reconciliation failed")
                promoted_digest.update(line)
                cell_counts[str(cell)] += 1
                histograms.setdefault(str(cell), Counter())[int(assistant_tokens)] += 1
                promoted_count += 1
            if response_stream.read(1):
                raise ExposureViewError("Task 6 response stream has unindexed data")
        try:
            next(records)
        except StopIteration:
            pass
        else:
            raise ExposureViewError("Task 6 corpus is missing an indexed record")
        attempt_history = sha256()
        for (payload,) in connection.execute("SELECT payload FROM attempts ORDER BY ordinal"):
            attempt_history.update(payload.encode() + b"\n")
        unused_count = int(connection.execute("SELECT count(*) FROM unused_reserve").fetchone()[0])
    finally:
        connection.close()
    normalized_histograms = {
        cell: dict(sorted(values.items())) for cell, values in sorted(histograms.items())
    }
    if (
        promoted_count != receipt.get("promoted_count")
        or dict(cell_counts) != receipt.get("cell_counts")
        or promoted_digest.hexdigest() != receipt.get("promoted_record_stream_sha256")
        or attempt_history.hexdigest() != receipt.get("attempt_history_stream_sha256")
        or unused_count != receipt.get("unused_reserve_count")
        or canonical_json(normalized_histograms)
        != canonical_json(receipt.get("response_token_histograms"))
    ):
        raise ExposureViewError("Task 6 promotion receipt reconciliation failed")
    corpus_record = {
        "arm": corpus.arm,
        "generation_identity_sha256": corpus.generation_identity_sha256,
        "selection_sha256": receipt.get("selection_sha256"),
        "paired_cd_sha256": receipt.get("paired_cd_sha256"),
        "cell_counts": receipt.get("cell_counts"),
        "promoted_record_stream_sha256": promoted_digest.hexdigest(),
        "attempt_history_stream_sha256": attempt_history.hexdigest(),
        "unused_reserve_count": unused_count,
        "response_token_histograms": normalized_histograms,
    }
    if sha256(canonical_json(corpus_record)).hexdigest() != corpus.corpus_sha256:
        raise ExposureViewError("Task 6 response corpus root mismatch")
    selection_sha256 = str(receipt.get("selection_sha256"))
    paired_sha256 = str(receipt.get("paired_cd_sha256"))
    _require_digest("Task 6 selection", selection_sha256)
    _require_digest("Task 6 paired C/D", paired_sha256)
    return _AuthenticatedResponse(
        selection_sha256=selection_sha256,
        paired_sha256=paired_sha256,
        index_path=index_path,
        completion_sha256=expected_completion_sha256,
    )


def _authenticate_final_paired_rows(c_index: Path, d_index: Path) -> str:
    def rows(connection: sqlite3.Connection) -> Iterator[tuple[str, str]]:
        for (payload,) in connection.execute("SELECT payload FROM promoted ORDER BY prompt_uuid"):
            record = json.loads(payload)
            if record.get("domain") != "swe-agentic-tool":
                yield str(record.get("prompt_uuid")), str(record.get("context_bucket"))

    c_connection = sqlite3.connect(f"file:{c_index}?mode=ro", uri=True)
    d_connection = sqlite3.connect(f"file:{d_index}?mode=ro", uri=True)
    try:
        c_rows = iter(rows(c_connection))
        d_rows = iter(rows(d_connection))
        digest = sha256(
            canonical_json({"schema_version": 1, "scope": "final-promoted-non-agentic-cd"})
        )
        while True:
            c_row = next(c_rows, None)
            d_row = next(d_rows, None)
            if c_row is None and d_row is None:
                break
            if c_row is None or d_row is None or c_row != d_row:
                raise ExposureViewError("final C/D promoted rows differ by UUID or context bucket")
            _digest_item(digest, list(c_row))
        return digest.hexdigest()
    finally:
        c_connection.close()
        d_connection.close()


def _authenticate_tokenizer_root(
    root: Path,
    *,
    tokenizer_sha256: str,
    chat_template_sha256: str,
) -> Any:
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
    if not files or any(path.is_symlink() for path in files):
        raise ExposureViewError("trainer tokenizer snapshot is missing or contains symlinks")
    digest = sha256()
    for path in files:
        digest.update(path.name.encode())
        digest.update(b"\0")
        digest.update(bytes.fromhex(_sha256_file(path)))
    if digest.hexdigest() != tokenizer_sha256:
        raise ExposureViewError("trainer tokenizer snapshot digest mismatch")
    template_path = root / "chat_template.jinja"
    if not template_path.is_file() or _sha256_file(template_path) != chat_template_sha256:
        raise ExposureViewError("trainer chat-template file digest mismatch")
    # Production must use the exact local snapshot just authenticated above.
    from transformers import AutoTokenizer

    try:
        tokenizer = AutoTokenizer.from_pretrained(
            root,
            local_files_only=True,
            trust_remote_code=False,
        )
    except (OSError, ValueError) as error:
        raise ExposureViewError(
            "authenticated trainer tokenizer snapshot cannot be loaded"
        ) from error
    template = getattr(tokenizer, "chat_template", None)
    if (
        not isinstance(template, str)
        or sha256(template.encode()).hexdigest() != chat_template_sha256
    ):
        raise ExposureViewError("loaded trainer chat template digest mismatch")
    return tokenizer


def _build_exposure_views(
    corpus: Any,
    tokenizer: Any,
    *,
    output_root: Path,
    boundaries: Mapping[str, int | None],
    seed: int,
    tokenizer_sha256: str,
    chat_template_sha256: str,
    training_sequence_length: int = 4_096,
    production: bool = False,
    policy_sha256: str | None = None,
    response_artifact_root: Path | None = None,
    expected_task6_completion_sha256: str | None = None,
    selection_manifest_path: Path | None = None,
    tokenizer_root: Path | None = None,
    _production_proof: _ProductionProof | None = None,
    _allow_partial_production_suite: bool = False,
    _allow_common_production: bool = False,
    _reconcile_resume: bool = True,
) -> Mapping[str, ExposureView]:
    """Retokenize a response corpus and stage exact consumable exposure views."""
    _require_digest("response corpus", corpus.corpus_sha256)
    _require_digest("source selection", corpus.source_selection_sha256)
    _require_digest("generation identity", corpus.generation_identity_sha256)
    _require_digest("tokenizer", tokenizer_sha256)
    _require_digest("chat template", chat_template_sha256)
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ExposureViewError("seed must be an integer")
    if (
        isinstance(training_sequence_length, bool)
        or not isinstance(training_sequence_length, int)
        or training_sequence_length < 1
    ):
        raise ExposureViewError("training sequence length must be a positive integer")
    if not boundaries:
        raise ExposureViewError("at least one exposure boundary is required")
    purpose = "production-comparison" if production else "content-view"
    if production:
        if training_sequence_length != 4_096:
            raise ExposureViewError("production training sequence length must be exactly 4096")
        if boundaries.get("one-pass") is not None:
            raise ExposureViewError("production one-pass view must use the full corpus")
        if "common" in boundaries and not _allow_common_production:
            raise ExposureViewError("common exposure is available only from the paired builder")
        required_names = {str(value) for value in _PRODUCTION_BOUNDARIES} | {"one-pass"}
        valid_suites = (
            (required_names, required_names | {"common"})
            if _allow_common_production
            else (required_names,)
        )
        if not _allow_partial_production_suite and set(boundaries) not in valid_suites:
            raise ExposureViewError(
                "production boundaries must include exact 256M, 1B, and one-pass views; "
                "64M is runtime-screen only"
            )
        if _allow_partial_production_suite and set(boundaries) - (required_names | {"common"}):
            raise ExposureViewError("production boundaries cannot include runtime screens")
        for value in _PRODUCTION_BOUNDARIES:
            name = str(value)
            if name in boundaries and boundaries[name] != value:
                raise ExposureViewError("production views require exact boundary values")
        if policy_sha256 is not None and policy_sha256 != _EXPOSURE_POLICY_SHA256:
            raise ExposureViewError("production exposure policy digest mismatch")
    resolved_policy_sha256 = policy_sha256 or (_EXPOSURE_POLICY_SHA256 if production else "0" * 64)
    resolved_source_policy_sha256 = "0" * 64
    task5_paired_sha256 = "0" * 64
    task6_completion_sha256 = "0" * 64
    final_paired_response_sha256 = "0" * 64
    proof = _production_proof
    if production:
        _authenticate_generation_identity(corpus)
        if proof is None:
            if (
                response_artifact_root is None
                or expected_task6_completion_sha256 is None
                or selection_manifest_path is None
                or tokenizer_root is None
            ):
                raise ExposureViewError(
                    "production views require Task 5, Task 6, and tokenizer artifact roots"
                )
            selection_sha256, paired_sha256, source_policy = _authenticate_selection_manifest(
                selection_manifest_path,
                expected_manifest_sha256=corpus.source_selection_sha256,
            )
            response = _authenticate_response_corpus(
                corpus,
                response_artifact_root,
                expected_completion_sha256=expected_task6_completion_sha256,
            )
            if (response.selection_sha256, response.paired_sha256) != (
                selection_sha256,
                paired_sha256,
            ):
                raise ExposureViewError("Task 5/Task 6 selection or paired proof mismatch")
            tokenizer = _authenticate_tokenizer_root(
                tokenizer_root,
                tokenizer_sha256=tokenizer_sha256,
                chat_template_sha256=chat_template_sha256,
            )
            record_paired_sha256 = paired_sha256 if corpus.arm in {"C", "D"} else "0" * 64
            proof = _ProductionProof(
                selection_sha256,
                paired_sha256,
                record_paired_sha256,
                source_policy,
                response.completion_sha256,
                "0" * 64,
            )
        resolved_source_policy_sha256 = proof.source_policy_sha256
        task5_paired_sha256 = proof.task5_paired_sha256
        task6_completion_sha256 = proof.task6_completion_sha256
        final_paired_response_sha256 = proof.final_paired_response_sha256
    _require_digest("policy", resolved_policy_sha256)
    _require_digest("source policy", resolved_source_policy_sha256)
    identity = corpus.generation_identity
    if identity.tokenizer_sha256 != tokenizer_sha256:
        raise ExposureViewError("trainer tokenizer does not match the response identity")
    if identity.chat_template_sha256 != chat_template_sha256:
        raise ExposureViewError("trainer chat template does not match the response identity")
    template = getattr(tokenizer, "chat_template", None)
    if (
        not isinstance(template, str)
        or sha256(template.encode()).hexdigest() != chat_template_sha256
    ):
        raise ExposureViewError("loaded trainer chat template digest mismatch")

    resume_fingerprint = _prepare_root(
        output_root,
        _identity_payload(
            corpus,
            tokenizer_sha256=tokenizer_sha256,
            chat_template_sha256=chat_template_sha256,
            seed=seed,
            training_sequence_length=training_sequence_length,
            purpose=purpose,
            policy_sha256=resolved_policy_sha256,
            source_policy_sha256=resolved_source_policy_sha256,
            task5_paired_sha256=task5_paired_sha256,
            task6_completion_sha256=task6_completion_sha256,
            final_paired_response_sha256=final_paired_response_sha256,
        ),
    )
    tokenized = _stage_tokenized_corpus(
        corpus,
        tokenizer,
        output_root,
        tokenizer_sha256=tokenizer_sha256,
        chat_template_sha256=chat_template_sha256,
        seed=seed,
        training_sequence_length=training_sequence_length,
        purpose=purpose,
        policy_sha256=resolved_policy_sha256,
        source_policy_sha256=resolved_source_policy_sha256,
        task5_paired_sha256=task5_paired_sha256,
        task6_completion_sha256=task6_completion_sha256,
        final_paired_response_sha256=final_paired_response_sha256,
        resume_fingerprint=resume_fingerprint,
        reconcile_resume=_reconcile_resume,
    )
    if production:
        if proof is None:
            raise ExposureViewError("production proof construction failed")
        if (
            tokenized.selection_sha256 != proof.selection_sha256
            or tokenized.paired_selection_bucket_sha256 != proof.record_paired_sha256
            or tokenized.task5_paired_selection_bucket_sha256 != proof.task5_paired_sha256
            or tokenized.task6_completion_sha256 != proof.task6_completion_sha256
            or tokenized.final_paired_response_sha256 != proof.final_paired_response_sha256
        ):
            raise ExposureViewError(
                "authenticated production proof does not match tokenized records"
            )
    views: dict[str, ExposureView] = {}
    for name, requested_tokens in boundaries.items():
        target_tokens = (
            tokenized.total_assistant_tokens if requested_tokens is None else requested_tokens
        )
        views[name] = _build_view(tokenized, output_root, name=name, target_tokens=target_tokens)
    return MappingProxyType(views)


def build_exposure_views(
    corpus: Any,
    tokenizer: Any,
    *,
    output_root: Path,
    boundaries: Mapping[str, int | None],
    seed: int,
    tokenizer_sha256: str,
    chat_template_sha256: str,
    training_sequence_length: int = 4_096,
    production: bool = False,
    response_artifact_root: Path | None = None,
    expected_task6_completion_sha256: str | None = None,
    selection_manifest_path: Path | None = None,
    tokenizer_root: Path | None = None,
) -> Mapping[str, ExposureView]:
    """Retokenize a response corpus into a complete content or production view suite."""
    return _build_exposure_views(
        corpus,
        tokenizer,
        output_root=output_root,
        boundaries=boundaries,
        seed=seed,
        tokenizer_sha256=tokenizer_sha256,
        chat_template_sha256=chat_template_sha256,
        training_sequence_length=training_sequence_length,
        production=production,
        response_artifact_root=response_artifact_root,
        expected_task6_completion_sha256=expected_task6_completion_sha256,
        selection_manifest_path=selection_manifest_path,
        tokenizer_root=tokenizer_root,
    )


def _write_paired_receipt(
    root: Path,
    *,
    views_c: Mapping[str, ExposureView],
    views_d: Mapping[str, ExposureView],
    common_boundary: int,
    paired_digest: str,
    source_policy_sha256: str,
    exposure_policy_sha256: str,
    final_paired_response_sha256: str,
    c_task6_completion_sha256: str,
    d_task6_completion_sha256: str,
) -> tuple[str, str]:
    receipt_path = root / "PAIRED_EXPOSURE.json"
    payload = {
        "schema_version": 1,
        "common_boundary": common_boundary,
        "C_full_prompt_assistant_tokens": views_c["one-pass"].full_prompt_assistant_tokens,
        "D_full_prompt_assistant_tokens": views_d["one-pass"].full_prompt_assistant_tokens,
        "paired_selection_bucket_sha256": paired_digest,
        "source_policy_sha256": source_policy_sha256,
        "exposure_policy_sha256": exposure_policy_sha256,
        "final_paired_response_sha256": final_paired_response_sha256,
        "C_task6_completion_sha256": c_task6_completion_sha256,
        "D_task6_completion_sha256": d_task6_completion_sha256,
        "C_view_receipts": {name: view.receipt_sha256 for name, view in views_c.items()},
        "D_view_receipts": {name: view.receipt_sha256 for name, view in views_d.items()},
    }
    receipt_sha256 = sha256(canonical_json(payload)).hexdigest()
    expected = canonical_json(payload | {"receipt_sha256": receipt_sha256}) + b"\n"
    if receipt_path.exists():
        if receipt_path.read_bytes() != expected:
            raise ExposureViewError("paired exposure receipt conflicts with staged views")
    else:
        _write_exclusive(receipt_path, expected)
        _fsync_directory(root)
    return str(receipt_path), receipt_sha256


def build_paired_exposure_views(
    corpus_c: Any,
    corpus_d: Any,
    tokenizer: Any,
    *,
    output_root: Path,
    fixed_boundaries: tuple[int, ...] = (256_000_000, 1_000_000_000),
    seed: int,
    tokenizer_sha256: str,
    chat_template_sha256: str,
    training_sequence_length: int = 4_096,
    production: bool = True,
    response_artifact_roots: Mapping[str, Path] | None = None,
    expected_task6_completion_sha256s: Mapping[str, str] | None = None,
    selection_manifest_path: Path | None = None,
    tokenizer_root: Path | None = None,
) -> PairedExposureViews:
    """Stage C/D fixed, largest-common, and separate one-pass exposure views."""
    if corpus_c.arm != "C" or corpus_d.arm != "D":
        raise ExposureViewError("paired exposure views require C and D response corpora")
    if (
        not fixed_boundaries
        or len(set(fixed_boundaries)) != len(fixed_boundaries)
        or any(
            not isinstance(value, int) or isinstance(value, bool) or value < 1
            for value in fixed_boundaries
        )
    ):
        raise ExposureViewError("fixed exposure boundaries must be nonempty and unique")
    if production and fixed_boundaries != _PRODUCTION_BOUNDARIES:
        raise ExposureViewError(
            "production boundaries must be exactly 256M and 1B; 64M is runtime-screen only"
        )
    policy_sha256 = _EXPOSURE_POLICY_SHA256
    source_policy_sha256 = "0" * 64
    production_proof: _ProductionProof | None = None
    if production:
        if (
            response_artifact_roots is None
            or set(response_artifact_roots) != {"C", "D"}
            or expected_task6_completion_sha256s is None
            or set(expected_task6_completion_sha256s) != {"C", "D"}
            or selection_manifest_path is None
            or tokenizer_root is None
        ):
            raise ExposureViewError(
                "production views require Task 5, Task 6, and tokenizer artifact roots"
            )
        if corpus_c.source_selection_sha256 != corpus_d.source_selection_sha256:
            raise ExposureViewError("C/D source selection manifest identities differ")
        selection_sha256, paired_sha256, source_policy_sha256 = _authenticate_selection_manifest(
            selection_manifest_path,
            expected_manifest_sha256=corpus_c.source_selection_sha256,
        )
        c_response = _authenticate_response_corpus(
            corpus_c,
            response_artifact_roots["C"],
            expected_completion_sha256=expected_task6_completion_sha256s["C"],
        )
        d_response = _authenticate_response_corpus(
            corpus_d,
            response_artifact_roots["D"],
            expected_completion_sha256=expected_task6_completion_sha256s["D"],
        )
        if (c_response.selection_sha256, d_response.selection_sha256) != (
            selection_sha256,
            selection_sha256,
        ) or (
            c_response.paired_sha256,
            d_response.paired_sha256,
        ) != (paired_sha256, paired_sha256):
            raise ExposureViewError("Task 5/Task 6 selection or paired proof mismatch")
        final_paired_response_sha256 = _authenticate_final_paired_rows(
            c_response.index_path,
            d_response.index_path,
        )
        tokenizer = _authenticate_tokenizer_root(
            tokenizer_root,
            tokenizer_sha256=tokenizer_sha256,
            chat_template_sha256=chat_template_sha256,
        )
        c_production_proof = _ProductionProof(
            selection_sha256,
            paired_sha256,
            paired_sha256,
            source_policy_sha256,
            c_response.completion_sha256,
            final_paired_response_sha256,
        )
        d_production_proof = _ProductionProof(
            selection_sha256,
            paired_sha256,
            paired_sha256,
            source_policy_sha256,
            d_response.completion_sha256,
            final_paired_response_sha256,
        )
    else:
        c_production_proof = production_proof
        d_production_proof = production_proof
        final_paired_response_sha256 = "0" * 64
    output_root.mkdir(parents=True, exist_ok=True)
    one_pass = {"one-pass": None}
    initial_c = _build_exposure_views(
        corpus_c,
        tokenizer,
        output_root=output_root / "C",
        boundaries=one_pass,
        seed=seed,
        tokenizer_sha256=tokenizer_sha256,
        chat_template_sha256=chat_template_sha256,
        training_sequence_length=training_sequence_length,
        production=production,
        policy_sha256=policy_sha256,
        _production_proof=c_production_proof,
        _allow_partial_production_suite=True,
    )
    initial_d = _build_exposure_views(
        corpus_d,
        tokenizer,
        output_root=output_root / "D",
        boundaries=one_pass,
        seed=seed,
        tokenizer_sha256=tokenizer_sha256,
        chat_template_sha256=chat_template_sha256,
        training_sequence_length=training_sequence_length,
        production=production,
        policy_sha256=policy_sha256,
        _production_proof=d_production_proof,
        _allow_partial_production_suite=True,
    )
    paired_digest = initial_c["one-pass"].paired_selection_bucket_sha256
    if (
        paired_digest == "0" * 64
        or initial_d["one-pass"].paired_selection_bucket_sha256 != paired_digest
        or initial_c["one-pass"].selection_sha256 != initial_d["one-pass"].selection_sha256
    ):
        raise ExposureViewError("C/D paired selection/bucket proof mismatch")
    common_boundary = min(
        initial_c["one-pass"].full_prompt_assistant_tokens,
        initial_d["one-pass"].full_prompt_assistant_tokens,
    )
    boundaries: dict[str, int | None] = {str(value): value for value in fixed_boundaries}
    boundaries.update({"common": common_boundary, "one-pass": None})
    views_c = _build_exposure_views(
        corpus_c,
        tokenizer,
        output_root=output_root / "C",
        boundaries=boundaries,
        seed=seed,
        tokenizer_sha256=tokenizer_sha256,
        chat_template_sha256=chat_template_sha256,
        training_sequence_length=training_sequence_length,
        production=production,
        policy_sha256=policy_sha256,
        _production_proof=c_production_proof,
        _reconcile_resume=False,
        _allow_common_production=True,
    )
    views_d = _build_exposure_views(
        corpus_d,
        tokenizer,
        output_root=output_root / "D",
        boundaries=boundaries,
        seed=seed,
        tokenizer_sha256=tokenizer_sha256,
        chat_template_sha256=chat_template_sha256,
        training_sequence_length=training_sequence_length,
        production=production,
        policy_sha256=policy_sha256,
        _production_proof=d_production_proof,
        _reconcile_resume=False,
        _allow_common_production=True,
    )
    receipt_path, receipt_sha256 = _write_paired_receipt(
        output_root,
        views_c=views_c,
        views_d=views_d,
        common_boundary=common_boundary,
        paired_digest=paired_digest,
        source_policy_sha256=source_policy_sha256,
        exposure_policy_sha256=policy_sha256,
        final_paired_response_sha256=final_paired_response_sha256,
        c_task6_completion_sha256=views_c["one-pass"].task6_completion_sha256,
        d_task6_completion_sha256=views_d["one-pass"].task6_completion_sha256,
    )
    return PairedExposureViews(
        C=views_c,
        D=views_d,
        common_boundary=common_boundary,
        C_full_prompt_assistant_tokens=views_c["one-pass"].full_prompt_assistant_tokens,
        D_full_prompt_assistant_tokens=views_d["one-pass"].full_prompt_assistant_tokens,
        paired_selection_bucket_sha256=paired_digest,
        final_paired_response_sha256=final_paired_response_sha256,
        C_task6_completion_sha256=views_c["one-pass"].task6_completion_sha256,
        D_task6_completion_sha256=views_d["one-pass"].task6_completion_sha256,
        source_policy_sha256=source_policy_sha256,
        exposure_policy_sha256=policy_sha256,
        receipt_path=receipt_path,
        receipt_sha256=receipt_sha256,
    )
