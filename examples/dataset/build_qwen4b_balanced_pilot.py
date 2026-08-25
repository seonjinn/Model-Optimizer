# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Deterministically select and publish the Qwen3-4B paired PTV2 pilot."""

from __future__ import annotations

import errno
import hashlib
import heapq
import json
import os
import shutil
import sqlite3
import stat
import sys
import tempfile
import uuid
from collections.abc import Iterable, Mapping
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from typing import Any

HISTORICAL_PROPORTION = "historical-proportion"
BALANCED = "balanced"
SEED = 20_260_822
TRAINING_SEQUENCE_LENGTH = 4_096


class PilotError(ValueError):
    """The pilot's caller-pinned data or publication contract is invalid."""


@dataclass(frozen=True)
class PilotQuota:
    """An exact row target from one source split for one pilot arm."""

    category: str
    split: str
    rows: int


@dataclass(frozen=True)
class PilotConfig:
    """Immutable scientific identity and exact selection quotas for the pilot."""

    seed: int = SEED
    workers: int = 1
    minimum_assistant_tokens: int = 16_000_000
    training_sequence_length: int = TRAINING_SEQUENCE_LENGTH
    source_repository: str = "nvidia/Nemotron-Post-Training-Dataset-v2"
    source_revision: str = ""
    historical_quotas: tuple[PilotQuota, ...] = (
        PilotQuota("chat", "chat", 48_286),
        PilotQuota("math", "math", 18_420),
        PilotQuota("code", "code", 13_462),
        PilotQuota("stem", "stem", 19_832),
    )
    balanced_quotas: tuple[PilotQuota, ...] = (
        PilotQuota("math", "math", 25_000),
        PilotQuota("code", "code", 20_000),
        PilotQuota("stem", "stem", 25_000),
        PilotQuota("chat", "chat", 20_000),
        PilotQuota("multilingual", "multilingual_de", 2_000),
        PilotQuota("multilingual", "multilingual_es", 2_000),
        PilotQuota("multilingual", "multilingual_fr", 2_000),
        PilotQuota("multilingual", "multilingual_it", 2_000),
        PilotQuota("multilingual", "multilingual_ja", 2_000),
    )

    def __post_init__(self) -> None:
        if isinstance(self.seed, bool) or not isinstance(self.seed, int):
            raise PilotError("seed must be an integer")
        if isinstance(self.workers, bool) or not isinstance(self.workers, int) or self.workers < 1:
            raise PilotError("workers must be a positive integer")
        _validate_quotas(HISTORICAL_PROPORTION, self.historical_quotas)
        _validate_quotas(BALANCED, self.balanced_quotas)
        if sum(quota.rows for quota in self.historical_quotas) != sum(
            quota.rows for quota in self.balanced_quotas
        ):
            raise PilotError("both pilot arms must have the same row quota")
        if (
            isinstance(self.minimum_assistant_tokens, bool)
            or not isinstance(self.minimum_assistant_tokens, int)
            or self.minimum_assistant_tokens < 1
        ):
            raise PilotError("minimum_assistant_tokens must be a positive integer")
        if (
            isinstance(self.training_sequence_length, bool)
            or not isinstance(self.training_sequence_length, int)
            or self.training_sequence_length < 1
        ):
            raise PilotError("training_sequence_length must be a positive integer")


@dataclass(frozen=True)
class PilotRow:
    """A normalized, selected source row with a canonical prompt identity."""

    prompt_uuid: str
    split: str
    category: str
    messages_json: str

    @property
    def messages(self) -> list[dict[str, Any]]:
        value = json.loads(self.messages_json)
        if not isinstance(value, list):  # Defensive: this value is created below.
            raise PilotError("stored pilot messages are malformed")
        return value


@dataclass(frozen=True)
class PilotSelection:
    """The two paired deterministic selections and their filtering evidence."""

    historical_proportion: tuple[PilotRow, ...]
    balanced: tuple[PilotRow, ...]
    excluded_held_out: int
    excluded_invalid: int
    excluded_duplicate_candidates: int

    def rows_for(self, arm: str) -> tuple[PilotRow, ...]:
        if arm == HISTORICAL_PROPORTION:
            return self.historical_proportion
        if arm == BALANCED:
            return self.balanced
        raise PilotError(f"unknown pilot arm: {arm!r}")


@dataclass(frozen=True)
class PilotArmCompletion:
    """Receipt-bound immutable completion information for one published arm."""

    arm: str
    row_count: int
    assistant_tokens: int
    manifest_sha256: str


@dataclass(frozen=True)
class PilotVerificationTrust:
    """Caller-owned identity that must be supplied to verify an installed bundle."""

    complete_file_sha256: str
    historical_manifest_sha256: str
    balanced_manifest_sha256: str
    source_repository: str
    source_revision: str
    seed: int
    historical_quotas: tuple[PilotQuota, ...]
    balanced_quotas: tuple[PilotQuota, ...]
    tokenizer_sha256: str
    chat_template_sha256: str
    producer_source_commit: str
    minimum_assistant_tokens: int
    training_sequence_length: int

    def __post_init__(self) -> None:
        for label, value in (
            ("complete file", self.complete_file_sha256),
            ("historical manifest", self.historical_manifest_sha256),
            ("balanced manifest", self.balanced_manifest_sha256),
            ("tokenizer", self.tokenizer_sha256),
            ("chat template", self.chat_template_sha256),
        ):
            if not _is_sha256(value):
                raise PilotError(f"{label} trust identity must be an exact lowercase SHA-256")
        if not self.source_repository:
            raise PilotError("source repository trust identity is required")
        for label, value in (
            ("source revision", self.source_revision),
            ("producer source commit", self.producer_source_commit),
        ):
            if (
                type(value) is not str
                or len(value) not in (40, 64)
                or any(character not in "0123456789abcdef" for character in value)
            ):
                raise PilotError(f"{label} trust identity is invalid")
        if type(self.seed) is not int:
            raise PilotError("seed trust identity must be a strict integer")
        _validate_quotas(HISTORICAL_PROPORTION, self.historical_quotas)
        _validate_quotas(BALANCED, self.balanced_quotas)
        for label, value in (
            ("minimum assistant tokens", self.minimum_assistant_tokens),
            ("training sequence length", self.training_sequence_length),
        ):
            if type(value) is not int or value < 1:
                raise PilotError(f"{label} trust identity must be a positive integer")


@dataclass(frozen=True)
class PilotCompletion:
    """Receipt-bound aggregate completion for the two-arm publication root."""

    output_root: Path
    historical_proportion: PilotArmCompletion
    balanced: PilotArmCompletion
    complete_sha256: str
    verification_trust: PilotVerificationTrust


@dataclass(frozen=True)
class QwenTokenizerPreflight:
    """Authenticated offline identity required before an OCI production build."""

    sha256: str
    chat_template_sha256: str
    im_start_token_id: int
    im_end_token_id: int
    probe_assistant_tokens: int


@dataclass(frozen=True)
class _DataEvidence:
    row_count: int
    split_counts: dict[str, int]
    category_counts: dict[str, int]
    prompt_uuid_sha256: str
    assistant_tokens: int
    token_evidence_sha256: str
    data_bytes: int
    data_sha256: str


@dataclass(frozen=True)
class _SelectionStats:
    excluded_held_out: int
    excluded_invalid: int
    excluded_duplicate_candidates: int


def canonical_json(value: object) -> str:
    """Return the unambiguous canonical JSON representation used for identities."""
    try:
        return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    except (TypeError, ValueError) as error:
        raise PilotError("value is not canonical-JSON serializable") from error


def _is_sha256(value: object) -> bool:
    return (
        type(value) is str
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def prompt_uuid_from_messages(messages: object) -> str:
    """Hash only the prompt context, removing exactly the terminal assistant turn."""
    try:
        normalized = _normalize_messages(messages)
        if normalized[-1]["role"] != "assistant":
            raise PilotError("row must have a terminal assistant response")
        return hashlib.sha256(canonical_json(normalized[:-1]).encode("utf-8")).hexdigest()
    except PilotError:
        raise
    except Exception as error:
        raise PilotError("prompt UUID normalization failed") from error


def load_held_out_prompt_uuids(path: Path) -> set[str]:
    """Load the caller-provided held-out prompt UUID JSON array fail-closed."""
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, TypeError, ValueError) as error:
        raise PilotError("held-out UUID receipt must be a JSON UUID array") from error
    if not isinstance(payload, list):
        raise PilotError("held-out UUID receipt must be a JSON UUID array")
    try:
        return _validate_held_out(payload)
    except PilotError:
        raise
    except Exception as error:
        raise PilotError("held-out UUID receipt validation failed") from error


def select_pilot_rows(
    split_rows: Mapping[str, Iterable[Mapping[str, object]]],
    *,
    config: PilotConfig | None = None,
    held_out_prompt_uuids: Iterable[str] = (),
    scratch_root: Path | None = None,
) -> PilotSelection:
    """Normalize failures at the public deterministic selection boundary."""
    try:
        return _select_pilot_rows_impl(
            split_rows,
            config=config,
            held_out_prompt_uuids=held_out_prompt_uuids,
            scratch_root=scratch_root,
        )
    except PilotError:
        raise
    except Exception as error:
        raise PilotError("pilot row selection failed") from error


def _select_pilot_rows_impl(
    split_rows: Mapping[str, Iterable[Mapping[str, object]]],
    *,
    config: PilotConfig | None = None,
    held_out_prompt_uuids: Iterable[str] = (),
    scratch_root: Path | None = None,
) -> PilotSelection:
    """Select exact paired quotas from a private SQLite spool without corpus materialization."""
    config = config or PilotConfig()
    if not isinstance(split_rows, Mapping):
        raise PilotError("split_rows must map PTV2 split names to iterables")
    held_out = _validate_held_out(held_out_prompt_uuids)
    scratch_parent = Path(scratch_root) if scratch_root is not None else Path(tempfile.gettempdir())
    if not scratch_parent.is_dir():
        raise PilotError("selection scratch root must be a directory")
    spool_root = Path(tempfile.mkdtemp(prefix=".qwen4b-selection-", dir=scratch_parent))
    os.chmod(spool_root, 0o700)
    database = spool_root / "selection.sqlite"
    try:
        connection = sqlite3.connect(database)
        try:
            _prepare_selection_database(connection)
            invalid, held_out_count, eligible_count = _spool_candidates(
                connection, split_rows, config, held_out
            )
            duplicate_count = _duplicate_count(connection, eligible_count)
            historical = _select_arm_from_spool(
                connection, HISTORICAL_PROPORTION, config.historical_quotas, config
            )
            balanced = _select_arm_from_spool(connection, BALANCED, config.balanced_quotas, config)
        finally:
            connection.close()
        return PilotSelection(
            historical_proportion=historical,
            balanced=balanced,
            excluded_held_out=held_out_count,
            excluded_invalid=invalid,
            excluded_duplicate_candidates=duplicate_count,
        )
    finally:
        shutil.rmtree(spool_root, ignore_errors=True)


def _validate_quotas(arm: str, quotas: tuple[PilotQuota, ...]) -> None:
    if not quotas:
        raise PilotError(f"{arm} quotas cannot be empty")
    if any(
        not isinstance(quota, PilotQuota)
        or isinstance(quota.rows, bool)
        or not isinstance(quota.rows, int)
        or quota.rows < 1
        for quota in quotas
    ):
        raise PilotError(f"{arm} quota rows must be positive integers")
    if len({quota.split for quota in quotas}) != len(quotas):
        raise PilotError(f"{arm} cannot declare a split more than once")


def _known_splits(config: PilotConfig) -> set[str]:
    return {quota.split for quota in config.historical_quotas + config.balanced_quotas}


def _validate_held_out(values: Iterable[str]) -> set[str]:
    try:
        result = set(values)
    except TypeError as error:
        raise PilotError("held-out UUIDs must be iterable strings") from error
    if any(
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
        for value in result
    ):
        raise PilotError("held-out UUIDs must be SHA-256 strings")
    return result


def _normalize_row(split: str, raw_row: Mapping[str, object]) -> PilotRow:
    if not isinstance(raw_row, Mapping):
        raise PilotError("PTV2 row must be a mapping")
    if "tools" in raw_row or "tool_calls" in raw_row:
        raise PilotError("tool/tool-call rows are excluded from the raw pilot")
    messages = _normalize_messages(raw_row.get("messages"))
    messages_json = canonical_json(messages)
    return PilotRow(
        prompt_uuid=prompt_uuid_from_messages(messages),
        split=split,
        category="multilingual" if split.startswith("multilingual_") else split,
        messages_json=messages_json,
    )


def _try_normalize_row(split: str, raw_row: Mapping[str, object]) -> PilotRow | None:
    try:
        return _normalize_row(split, raw_row)
    except PilotError:
        return None


def _normalize_messages(value: object) -> list[dict[str, Any]]:
    if not isinstance(value, list) or not value:
        raise PilotError("row messages must be a non-empty list")
    normalized: list[dict[str, Any]] = []
    for message in value:
        if not isinstance(message, Mapping):
            raise PilotError("each message must be a mapping")
        role = message.get("role")
        if not isinstance(role, str) or role not in {"system", "user", "assistant"}:
            raise PilotError("tool or malformed message role is excluded")
        if "tool_calls" in message or "tool_call_id" in message:
            raise PilotError("tool/tool-call rows are excluded from the raw pilot")
        content = message.get("content")
        if not isinstance(content, str):
            raise PilotError("message content must be a string")
        normalized.append({"role": role, "content": content})
    if normalized[-1]["role"] != "assistant" or not normalized[-1]["content"]:
        raise PilotError("row must have a non-empty terminal assistant response")
    return normalized


def _rank(row: PilotRow, seed: int) -> str:
    return hashlib.sha256(
        canonical_json([seed, row.split, row.prompt_uuid]).encode("utf-8")
    ).hexdigest()


def _prepare_selection_database(connection: sqlite3.Connection) -> None:
    connection.execute(
        "CREATE TABLE candidates (prompt_uuid TEXT NOT NULL, split TEXT NOT NULL, "
        "category TEXT NOT NULL, messages_json TEXT NOT NULL, rank TEXT NOT NULL, "
        "PRIMARY KEY(prompt_uuid, split)) WITHOUT ROWID"
    )
    connection.execute("CREATE INDEX candidates_split_rank ON candidates(split, rank, prompt_uuid)")
    connection.execute(
        "CREATE TABLE selected (arm TEXT NOT NULL, ordinal INTEGER NOT NULL, "
        "prompt_uuid TEXT NOT NULL, split TEXT NOT NULL, category TEXT NOT NULL, "
        "messages_json TEXT NOT NULL, assistant_tokens INTEGER, "
        "PRIMARY KEY(arm, ordinal), UNIQUE(arm, prompt_uuid)) WITHOUT ROWID"
    )


def _spool_candidates(
    connection: sqlite3.Connection,
    split_rows: Mapping[str, Iterable[Mapping[str, object]]],
    config: PilotConfig,
    held_out: set[str],
) -> tuple[int, int, int]:
    invalid = 0
    held_out_count = 0
    eligible_count = 0
    for split, rows in split_rows.items():
        if not isinstance(split, str) or split not in _known_splits(config):
            raise PilotError(f"unapproved PTV2 split: {split!r}")
        try:
            iterator = iter(rows)
        except TypeError as error:
            raise PilotError(f"split {split!r} is not iterable") from error
        batch: list[tuple[str, str, str, str, str]] = []
        for raw_row in iterator:
            normalized = _try_normalize_row(split, raw_row)
            if normalized is None:
                invalid += 1
                continue
            if normalized.prompt_uuid in held_out:
                held_out_count += 1
                continue
            eligible_count += 1
            batch.append(
                (
                    normalized.prompt_uuid,
                    normalized.split,
                    normalized.category,
                    normalized.messages_json,
                    _rank(normalized, config.seed),
                )
            )
            if len(batch) == 1_024:
                _insert_candidate_batch(connection, batch)
                batch.clear()
        if batch:
            _insert_candidate_batch(connection, batch)
    connection.commit()
    return invalid, held_out_count, eligible_count


def _insert_candidate_batch(
    connection: sqlite3.Connection, batch: list[tuple[str, str, str, str, str]]
) -> None:
    connection.executemany(
        "INSERT INTO candidates VALUES (?, ?, ?, ?, ?) "
        "ON CONFLICT(prompt_uuid, split) DO UPDATE SET "
        "messages_json = MIN(messages_json, excluded.messages_json)",
        batch,
    )


def _duplicate_count(connection: sqlite3.Connection, eligible_count: int) -> int:
    unique = int(
        connection.execute("SELECT COUNT(DISTINCT prompt_uuid) FROM candidates").fetchone()[0]
    )
    return eligible_count - unique


def _select_arm_from_spool(
    connection: sqlite3.Connection,
    arm: str,
    quotas: tuple[PilotQuota, ...],
    config: PilotConfig,
    *,
    materialize: bool = True,
) -> tuple[PilotRow, ...]:
    capacity = {quota.split: quota.rows for quota in quotas}
    category = {quota.split: quota.category for quota in quotas}
    members: dict[str, set[str]] = {split: set() for split in capacity}
    assignment: dict[str, str] = {}
    edge_cache: dict[str, tuple[tuple[str, str], ...]] = {}
    residual_heaps: dict[tuple[str, str], list[tuple[int, str]]] = {}

    def cache_residual_edges(prompt_uuid: str, assigned_split: str) -> None:
        rank_by_split = dict(edge_cache[prompt_uuid])
        displacement_key = -int(rank_by_split[assigned_split], 16)
        for target_split in rank_by_split:
            if target_split != assigned_split:
                heapq.heappush(
                    residual_heaps.setdefault((assigned_split, target_split), []),
                    (displacement_key, prompt_uuid),
                )

    def move(prompt_uuid: str, destination: str) -> None:
        previous = assignment.get(prompt_uuid)
        if previous is not None:
            members[previous].remove(prompt_uuid)
        members[destination].add(prompt_uuid)
        assignment[prompt_uuid] = destination
        cache_residual_edges(prompt_uuid, destination)

    def movable_occupant(source: str, destination: str) -> str | None:
        heap = residual_heaps.get((source, destination), [])
        while heap and assignment.get(heap[0][1]) != source:
            heapq.heappop(heap)
        return heap[0][1] if heap else None

    def place(prompt_uuid: str, candidate_edges: tuple[tuple[str, str], ...]) -> bool:
        edge_cache[prompt_uuid] = candidate_edges
        queue = [split for split, _rank_value in candidate_edges]
        parent: dict[str, tuple[str, str] | None] = dict.fromkeys(queue)
        cursor_index = 0
        free_split: str | None = None
        while cursor_index < len(queue):
            source = queue[cursor_index]
            cursor_index += 1
            if len(members[source]) < capacity[source]:
                free_split = source
                break
            for destination in sorted(capacity):
                if destination in parent or destination == source:
                    continue
                occupant = movable_occupant(source, destination)
                if occupant is None:
                    continue
                parent[destination] = (source, occupant)
                queue.append(destination)
        if free_split is None:
            edge_cache.pop(prompt_uuid)
            return False
        destination = free_split
        link = parent[destination]
        while link is not None:
            source, occupant = link
            move(occupant, destination)
            destination = source
            link = parent[destination]
        move(prompt_uuid, destination)
        return True

    target = sum(capacity.values())
    placeholders = ",".join("?" for _ in capacity)
    ranked_edges = connection.execute(
        "WITH candidate_order AS ("
        "SELECT prompt_uuid, MIN(rank) AS first_rank FROM candidates "
        f"WHERE split IN ({placeholders}) GROUP BY prompt_uuid) "
        "SELECT candidates.prompt_uuid, candidates.split, candidates.rank "
        "FROM candidate_order JOIN candidates USING(prompt_uuid) "
        f"WHERE candidates.split IN ({placeholders}) "
        "ORDER BY candidate_order.first_rank, candidates.prompt_uuid, "
        "candidates.rank, candidates.split",
        (*capacity, *capacity),
    )
    current_uuid: str | None = None
    current_edges: list[tuple[str, str]] = []
    try:
        for raw_uuid, raw_split, raw_rank in ranked_edges:
            prompt_uuid = str(raw_uuid)
            if current_uuid is not None and prompt_uuid != current_uuid:
                place(current_uuid, tuple(current_edges))
                if len(assignment) == target:
                    break
                current_edges.clear()
            current_uuid = prompt_uuid
            current_edges.append((str(raw_split), str(raw_rank)))
        else:
            if current_uuid is not None and len(assignment) < target:
                place(current_uuid, tuple(current_edges))
    finally:
        ranked_edges.close()
    for quota in quotas:
        filled = len(members[quota.split])
        if filled != quota.rows:
            raise PilotError(
                f"{arm}/{quota.category}/{quota.split} received {filled} rows after global "
                f"prompt dedup; exact quota {quota.rows} is globally infeasible"
            )
    selected_identities: list[tuple[str, str, str]] = []
    for prompt_uuid, split in assignment.items():
        rank = next(value for edge, value in edge_cache[prompt_uuid] if edge == split)
        selected_identities.append((rank, prompt_uuid, split))
    selected_identities.sort()
    connection.execute(
        "CREATE TEMP TABLE IF NOT EXISTS current_assignment ("
        "ordinal INTEGER PRIMARY KEY, prompt_uuid TEXT NOT NULL, split TEXT NOT NULL, "
        "category TEXT NOT NULL)"
    )
    connection.execute("DELETE FROM current_assignment")
    batch: list[tuple[int, str, str, str]] = []
    for ordinal, (_rank_value, prompt_uuid, split) in enumerate(selected_identities):
        batch.append((ordinal, prompt_uuid, split, category[split]))
        if len(batch) == 1_024:
            connection.executemany("INSERT INTO current_assignment VALUES (?, ?, ?, ?)", batch)
            batch.clear()
    if batch:
        connection.executemany("INSERT INTO current_assignment VALUES (?, ?, ?, ?)", batch)
    connection.execute(
        "INSERT INTO selected "
        "SELECT ?, assignment.ordinal, assignment.prompt_uuid, assignment.split, "
        "assignment.category, candidates.messages_json, NULL "
        "FROM current_assignment AS assignment JOIN candidates "
        "ON candidates.prompt_uuid = assignment.prompt_uuid "
        "AND candidates.split = assignment.split ORDER BY assignment.ordinal",
        (arm,),
    )
    inserted = connection.execute("SELECT COUNT(*) FROM selected WHERE arm = ?", (arm,)).fetchone()[
        0
    ]
    if inserted != target:
        raise PilotError("quota-aware assignment lost a selected candidate")
    connection.commit()
    if not materialize:
        return ()
    return tuple(
        PilotRow(str(prompt_uuid), str(split), str(row_category), str(messages_json))
        for prompt_uuid, split, row_category, messages_json in connection.execute(
            "SELECT prompt_uuid, split, category, messages_json FROM selected "
            "WHERE arm = ? ORDER BY ordinal",
            (arm,),
        )
    )


def count_assistant_tokens(
    tokenizer: Any,
    messages: list[dict[str, Any]],
    *,
    training_sequence_length: int = TRAINING_SEQUENCE_LENGTH,
) -> int:
    """Count trainer-visible assistant positions after left-to-right truncation."""
    if (
        isinstance(training_sequence_length, bool)
        or not isinstance(training_sequence_length, int)
        or training_sequence_length < 1
    ):
        raise PilotError("training sequence length must be a positive integer")
    try:
        encoded = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            return_dict=True,
            return_assistant_tokens_mask=True,
        )
    except Exception as error:
        raise PilotError("token worker failed while applying the chat template") from error
    if not isinstance(encoded, Mapping):
        raise PilotError("tokenizer did not return a token mapping")
    input_ids = encoded.get("input_ids")
    mask = encoded.get("assistant_masks", encoded.get("assistant_tokens_mask"))
    if (
        not isinstance(input_ids, list)
        or not isinstance(mask, list)
        or len(input_ids) != len(mask)
        or any(type(value) is not int or value not in (0, 1) for value in mask)
    ):
        raise PilotError("tokenizer did not return an aligned exact assistant mask")
    assistant_tokens = sum(mask[:training_sequence_length])
    if assistant_tokens < 1:
        raise PilotError("row has no trainable assistant tokens")
    return assistant_tokens


def build_pilot_bundles(
    split_rows: Mapping[str, Iterable[Mapping[str, object]]],
    *,
    config: PilotConfig | None = None,
    held_out_prompt_uuids: Iterable[str] = (),
    tokenizer_path: str | Path,
    tokenizer_sha256: str,
    output_root: Path,
    producer_source_commit: str,
    workers: int | None = None,
    scratch_root: Path | None = None,
    execution: Mapping[str, object] | None = None,
) -> PilotCompletion:
    """Publish only the frozen 100K-row, 16M-token Qwen3-4B pilot identity."""
    staged: Path | None = None
    try:
        config = config or PilotConfig()
        _validate_production_identity(config)
        output_root = Path(output_root)
        scratch_parent = Path(scratch_root) if scratch_root is not None else output_root.parent
        staged, _digest = _stage_verified_tokenizer_snapshot(
            Path(tokenizer_path), tokenizer_sha256, scratch_parent
        )
        tokenizer = _load_verified_qwen_tokenizer(staged)
        _qwen3_tokenizer_preflight_evidence(tokenizer, tokenizer_sha256)
        return _build_pilot_bundles_for_test(
            split_rows,
            config=config,
            held_out_prompt_uuids=held_out_prompt_uuids,
            tokenizer=tokenizer,
            tokenizer_path=tokenizer_path,
            tokenizer_sha256=tokenizer_sha256,
            output_root=output_root,
            producer_source_commit=producer_source_commit,
            workers=workers,
            scratch_root=scratch_root,
            execution=execution,
            _production=True,
        )
    except PilotError:
        raise
    except Exception as error:
        raise PilotError("pilot publication failed") from error
    finally:
        if staged is not None and staged.exists():
            shutil.rmtree(staged)


def _build_pilot_bundles_for_test(
    split_rows: Mapping[str, Iterable[Mapping[str, object]]],
    *,
    config: PilotConfig | None = None,
    held_out_prompt_uuids: Iterable[str] = (),
    tokenizer: Any,
    tokenizer_path: str | Path,
    tokenizer_sha256: str,
    output_root: Path,
    producer_source_commit: str,
    workers: int | None = None,
    scratch_root: Path | None = None,
    execution: Mapping[str, object] | None = None,
    _production: bool = False,
) -> PilotCompletion:
    """Test-only generic publication helper; production callers use ``build_pilot_bundles``."""
    config = config or PilotConfig()
    output_root = Path(output_root)
    _validate_build_inputs(
        config=config,
        tokenizer_path=tokenizer_path,
        tokenizer_sha256=tokenizer_sha256,
        producer_source_commit=producer_source_commit,
        output_root=output_root,
        workers=workers,
    )
    if output_root.exists():
        raise PilotError(f"publication destination already exists: {output_root}")
    scratch_parent = Path(scratch_root) if scratch_root is not None else output_root.parent
    if not scratch_parent.is_dir():
        raise PilotError(f"scratch parent is not a directory: {scratch_parent}")
    selection_root = Path(tempfile.mkdtemp(prefix=".qwen4b-selection-", dir=scratch_parent))
    os.chmod(selection_root, 0o700)
    connection: sqlite3.Connection | None = None
    partial = Path(tempfile.mkdtemp(prefix=f".{output_root.name}.scratch-", dir=scratch_parent))
    os.chmod(partial, 0o700)
    local_partial: Path | None = None
    try:
        connection = sqlite3.connect(selection_root / "selection.sqlite")
        _prepare_selection_database(connection)
        held_out = _validate_held_out(held_out_prompt_uuids)
        invalid, held_out_count, eligible_count = _spool_candidates(
            connection, split_rows, config, held_out
        )
        stats = _SelectionStats(
            excluded_held_out=held_out_count,
            excluded_invalid=invalid,
            excluded_duplicate_candidates=_duplicate_count(connection, eligible_count),
        )
        _select_arm_from_spool(
            connection,
            HISTORICAL_PROPORTION,
            config.historical_quotas,
            config,
            materialize=False,
        )
        _select_arm_from_spool(
            connection,
            BALANCED,
            config.balanced_quotas,
            config,
            materialize=False,
        )
        effective_workers = workers if workers is not None else config.workers
        tokens_by_arm = {
            arm: _count_spooled_rows(
                connection,
                tokenizer,
                arm,
                effective_workers,
                config.training_sequence_length,
            )
            for arm in (HISTORICAL_PROPORTION, BALANCED)
        }
        for arm, total in tokens_by_arm.items():
            if total < config.minimum_assistant_tokens:
                raise PilotError(
                    f"{arm} assistant-token minimum is {config.minimum_assistant_tokens}, "
                    f"but selected rows contain {total}"
                )
        template_sha256 = hashlib.sha256(
            str(getattr(tokenizer, "chat_template", "")).encode("utf-8")
        ).hexdigest()
        arms = _write_spooled_bundle_contents(
            partial,
            connection=connection,
            selection_stats=stats,
            tokens_by_arm=tokens_by_arm,
            config=config,
            tokenizer_path=tokenizer_path,
            tokenizer_sha256=tokenizer_sha256,
            tokenizer_template_sha256=template_sha256,
            producer_source_commit=producer_source_commit,
            effective_workers=effective_workers,
            execution=execution,
        )
        verification_trust = _build_verification_trust(
            partial,
            config=config,
            arms=arms,
            tokenizer_sha256=tokenizer_sha256,
            chat_template_sha256=template_sha256,
            producer_source_commit=producer_source_commit,
        )
        bundle_evidence = _bundle_file_evidence(partial)
        _verify_bundle_root(
            partial,
            production=_production,
            tokenizer=tokenizer,
            verification_trust=verification_trust,
        )
        _verify_bundle_file_evidence(partial, bundle_evidence)
        local_partial = output_root.parent / f".{output_root.name}.publish-{uuid.uuid4().hex}"
        shutil.copytree(partial, local_partial)
        os.chmod(local_partial, 0o700)
        _verify_bundle_file_evidence(local_partial, bundle_evidence)
        _fsync_tree(local_partial)
        if output_root.exists():
            raise PilotError(f"publication destination already exists: {output_root}")
        copied_stat = local_partial.lstat()
        copied_identity = (copied_stat.st_dev, copied_stat.st_ino)
        try:
            _rename_noreplace(local_partial, output_root)
        except OSError as error:
            if error.errno == errno.EEXIST:
                raise PilotError(
                    f"publication destination already exists: {output_root}"
                ) from error
            raise PilotError("atomic pilot publication failed") from error
        local_partial = None
        _fsync_directory(output_root.parent)
        installed_stat = output_root.lstat()
        if (installed_stat.st_dev, installed_stat.st_ino) != copied_identity:
            raise PilotError("installed publication inode differs from its authenticated copy")
        _verify_bundle_file_evidence(output_root, bundle_evidence)
        complete, complete_raw = _read_json_stable(output_root / "COMPLETE.json")
        if hashlib.sha256(complete_raw).hexdigest() != verification_trust.complete_file_sha256:
            raise PilotError("installed bundle differs from caller-owned verification trust")
        return PilotCompletion(
            output_root=output_root,
            historical_proportion=arms[HISTORICAL_PROPORTION],
            balanced=arms[BALANCED],
            complete_sha256=str(complete["complete_sha256"]),
            verification_trust=verification_trust,
        )
    finally:
        if connection is not None:
            connection.close()
        if selection_root.exists():
            shutil.rmtree(selection_root)
        if partial.exists():
            shutil.rmtree(partial)
        if local_partial is not None and local_partial.exists():
            shutil.rmtree(local_partial)


def verify_pilot_completion(
    output_root: Path,
    *,
    verification_trust: PilotVerificationTrust,
    scratch_root: Path | None = None,
) -> PilotCompletion:
    """Replay every file descriptor and receipt in an installed pilot root."""
    try:
        output_root = Path(output_root)
        arms = _verify_bundle_root(
            output_root,
            production=True,
            scratch_root=scratch_root,
            verification_trust=verification_trust,
        )
        complete, complete_raw = _read_json_stable(output_root / "COMPLETE.json")
        if hashlib.sha256(complete_raw).hexdigest() != verification_trust.complete_file_sha256:
            raise PilotError("installed bundle differs from caller-owned verification trust")
        return PilotCompletion(
            output_root=output_root,
            historical_proportion=arms[HISTORICAL_PROPORTION],
            balanced=arms[BALANCED],
            complete_sha256=str(complete["complete_sha256"]),
            verification_trust=verification_trust,
        )
    except PilotError:
        raise
    except Exception as error:
        raise PilotError("production pilot completion verification failed") from error


def _validate_build_inputs(
    *,
    config: PilotConfig,
    tokenizer_path: str | Path,
    tokenizer_sha256: str,
    producer_source_commit: str,
    output_root: Path,
    workers: int | None,
) -> None:
    if not str(tokenizer_path):
        raise PilotError("tokenizer path is required")
    if len(config.source_revision) not in (40, 64) or any(
        char not in "0123456789abcdef" for char in config.source_revision
    ):
        raise PilotError("source_revision must be an immutable hexadecimal commit")
    if len(tokenizer_sha256) != 64 or any(
        char not in "0123456789abcdef" for char in tokenizer_sha256
    ):
        raise PilotError("tokenizer_sha256 must be an exact lowercase SHA-256")
    if len(producer_source_commit) not in (40, 64) or any(
        char not in "0123456789abcdef" for char in producer_source_commit
    ):
        raise PilotError("producer_source_commit must be an immutable hexadecimal commit")
    if output_root.parent == output_root or not output_root.parent.is_dir():
        raise PilotError("output root must have an existing parent directory")
    if workers is not None and (
        isinstance(workers, bool) or not isinstance(workers, int) or workers < 1
    ):
        raise PilotError("workers must be a positive integer")


def _validate_production_identity(config: PilotConfig) -> None:
    approved = PilotConfig()
    if (
        config.seed != SEED
        or config.minimum_assistant_tokens != 16_000_000
        or config.training_sequence_length != TRAINING_SEQUENCE_LENGTH
        or config.source_repository != approved.source_repository
        or config.historical_quotas != approved.historical_quotas
        or config.balanced_quotas != approved.balanced_quotas
        or len(config.source_revision) not in (40, 64)
        or any(char not in "0123456789abcdef" for char in config.source_revision)
    ):
        raise PilotError("public builder requires the frozen production identity")
    if not sys.platform.startswith("linux"):
        raise PilotError("public builder requires Linux atomic no-replace publication")


def _open_nofollow_directory(path: Path) -> int:
    if not hasattr(os, "O_NOFOLLOW") or not hasattr(os, "O_DIRECTORY"):
        raise PilotError("tokenizer snapshot requires componentwise no-follow support")
    lexical = path.absolute()
    flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
    descriptor: int | None = None
    try:
        descriptor = os.open(lexical.anchor, flags)
        for component in lexical.parts[1:]:
            child = os.open(component, flags, dir_fd=descriptor)
            os.close(descriptor)
            descriptor = child
        return descriptor
    except OSError as error:
        if descriptor is not None:
            with suppress(OSError):
                os.close(descriptor)
        raise PilotError("tokenizer path contains a symlink or non-directory component") from error


def _copy_snapshot_tree(source_fd: int, destination: Path) -> None:
    try:
        initial_names = sorted(os.listdir(source_fd))
    except OSError as error:
        raise PilotError(
            "unable to enumerate tokenizer snapshot without following links"
        ) from error
    for name in initial_names:
        try:
            observed = os.stat(name, dir_fd=source_fd, follow_symlinks=False)
        except OSError as error:
            raise PilotError("tokenizer snapshot component changed during staging") from error
        if stat.S_ISLNK(observed.st_mode):
            raise PilotError("tokenizer snapshot contains a symlink; no-follow staging refused it")
        target = destination / name
        if stat.S_ISDIR(observed.st_mode):
            target.mkdir(mode=0o700)
            try:
                child_fd = os.open(
                    name,
                    os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
                    dir_fd=source_fd,
                )
            except OSError as error:
                raise PilotError("tokenizer directory changed during no-follow staging") from error
            try:
                _copy_snapshot_tree(child_fd, target)
            finally:
                os.close(child_fd)
            continue
        if not stat.S_ISREG(observed.st_mode):
            raise PilotError("tokenizer snapshot contains a non-regular component")
        try:
            source_file = os.open(name, os.O_RDONLY | os.O_NOFOLLOW, dir_fd=source_fd)
        except OSError as error:
            raise PilotError("tokenizer file changed during no-follow staging") from error
        try:
            initial = os.fstat(source_file)
            with target.open("xb") as output:
                os.chmod(target, 0o600)
                while chunk := os.read(source_file, 1024 * 1024):
                    output.write(chunk)
                output.flush()
                os.fsync(output.fileno())
            final = os.fstat(source_file)

            def identity(value: os.stat_result) -> tuple[int, int, int, int, int]:
                return (
                    value.st_dev,
                    value.st_ino,
                    value.st_size,
                    value.st_mtime_ns,
                    value.st_ctime_ns,
                )

            if identity(initial) != identity(final):
                raise PilotError("tokenizer file changed during private snapshot staging")
        finally:
            os.close(source_file)
    if sorted(os.listdir(source_fd)) != initial_names:
        raise PilotError("tokenizer directory changed during private snapshot staging")


def _tokenizer_tree_sha256(root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(candidate for candidate in root.rglob("*") if candidate.is_file()):
        if path.is_symlink():
            raise PilotError("private tokenizer snapshot unexpectedly contains a symlink")
        digest.update(path.relative_to(root).as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(bytes.fromhex(_sha256_file(path)))
    return digest.hexdigest()


def _stage_verified_tokenizer_snapshot(
    source: Path, expected_sha256: str, scratch_root: Path
) -> tuple[Path, str]:
    """Copy a tokenizer through no-follow descriptors, then authenticate that exact copy."""
    if not _is_sha256(expected_sha256):
        raise PilotError("tokenizer_sha256 must be an exact lowercase SHA-256")
    scratch_root = Path(scratch_root)
    if not scratch_root.is_dir() or scratch_root.is_symlink():
        raise PilotError("tokenizer scratch root must be an existing non-symlink directory")
    snapshot = Path(tempfile.mkdtemp(prefix=".qwen4b-tokenizer-", dir=scratch_root))
    os.chmod(snapshot, 0o700)
    source_fd: int | None = None
    try:
        source_fd = _open_nofollow_directory(Path(source))
        _copy_snapshot_tree(source_fd, snapshot)
        digest = _tokenizer_tree_sha256(snapshot)
        if digest != expected_sha256:
            raise PilotError("private tokenizer snapshot does not match the pinned SHA-256")
        return snapshot, digest
    except PilotError:
        shutil.rmtree(snapshot, ignore_errors=True)
        raise
    except Exception as error:
        shutil.rmtree(snapshot, ignore_errors=True)
        raise PilotError("unable to stage the private tokenizer snapshot") from error
    finally:
        if source_fd is not None:
            os.close(source_fd)


def _load_verified_qwen_tokenizer(path: Path) -> Any:
    try:
        from transformers import AutoTokenizer  # pyright: ignore[reportMissingImports]

        tokenizer = AutoTokenizer.from_pretrained(
            path,
            local_files_only=True,
            trust_remote_code=False,
        )
    except Exception as error:
        raise PilotError("unable to load the pinned Qwen3-4B tokenizer") from error
    if not isinstance(getattr(tokenizer, "chat_template", None), str):
        raise PilotError("pinned Qwen3-4B tokenizer has no chat template")
    return tokenizer


def _qwen3_tokenizer_preflight_evidence(
    tokenizer: Any, authenticated_sha256: str
) -> QwenTokenizerPreflight:
    try:
        im_start_token_id = tokenizer.convert_tokens_to_ids("<|im_start|>")
        im_end_token_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
        eos_token_id = tokenizer.eos_token_id
    except Exception as error:
        raise PilotError(
            "pinned tokenizer does not expose the Qwen3 special-token contract"
        ) from error
    if (
        type(im_start_token_id) is not int
        or im_start_token_id != 151_644
        or type(im_end_token_id) is not int
        or im_end_token_id != 151_645
        or type(eos_token_id) is not int
        or eos_token_id != im_end_token_id
    ):
        raise PilotError("pinned tokenizer is not the approved Qwen3 tokenizer identity")
    probe_assistant_tokens = count_assistant_tokens(
        tokenizer,
        [
            {"role": "system", "content": "You are useful."},
            {"role": "user", "content": "question"},
            {"role": "assistant", "content": "answer"},
        ],
    )
    template = getattr(tokenizer, "chat_template", None)
    if not isinstance(template, str) or not template:
        raise PilotError("pinned Qwen3 tokenizer has no chat template")
    return QwenTokenizerPreflight(
        sha256=authenticated_sha256,
        chat_template_sha256=hashlib.sha256(template.encode("utf-8")).hexdigest(),
        im_start_token_id=im_start_token_id,
        im_end_token_id=im_end_token_id,
        probe_assistant_tokens=probe_assistant_tokens,
    )


def preflight_qwen3_4b_tokenizer_snapshot(
    tokenizer_path: str | Path,
    tokenizer_sha256: str,
    scratch_root: Path,
) -> QwenTokenizerPreflight:
    """Authenticate and exercise the exact offline Qwen3-4B tokenizer used on OCI."""
    snapshot: Path | None = None
    try:
        snapshot, digest = _stage_verified_tokenizer_snapshot(
            Path(tokenizer_path), tokenizer_sha256, Path(scratch_root)
        )
        tokenizer = _load_verified_qwen_tokenizer(snapshot)
        return _qwen3_tokenizer_preflight_evidence(tokenizer, digest)
    finally:
        if snapshot is not None and snapshot.exists():
            shutil.rmtree(snapshot)


def _rename_noreplace(source: Path, destination: Path) -> None:
    """Install a directory without replacement; Linux gets the kernel primitive."""
    if sys.platform.startswith("linux"):
        import ctypes

        libc = ctypes.CDLL(None, use_errno=True)
        try:
            renameat2 = libc.renameat2
        except AttributeError as error:
            raise OSError(errno.ENOSYS, "libc does not expose renameat2") from error
        renameat2.argtypes = (
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_uint,
        )
        renameat2.restype = ctypes.c_int
        at_fdcwd = getattr(os, "AT_FDCWD", -100)
        result = renameat2(
            at_fdcwd,
            os.fsencode(source),
            at_fdcwd,
            os.fsencode(destination),
            1,
        )
        if result != 0:
            error_number = ctypes.get_errno()
            raise OSError(error_number, os.strerror(error_number), destination)
        return
    if destination.exists():
        raise OSError(errno.EEXIST, "destination exists", destination)
    os.rename(source, destination)


def _count_spooled_rows(
    connection: sqlite3.Connection,
    tokenizer: Any,
    arm: str,
    workers: int,
    training_sequence_length: int,
) -> int:
    """Tokenize selected messages with bounded futures and persist only scalar evidence."""
    cursor = connection.execute(
        "SELECT ordinal, prompt_uuid, split, category, messages_json FROM selected "
        "WHERE arm = ? ORDER BY ordinal",
        (arm,),
    )

    def count(payload: tuple[object, ...]) -> tuple[int, int]:
        ordinal, prompt_uuid, split, category, messages_json = payload
        if type(ordinal) is not int:
            raise PilotError("selected ordinal is not a strict integer")
        row = PilotRow(str(prompt_uuid), str(split), str(category), str(messages_json))
        return ordinal, count_assistant_tokens(
            tokenizer,
            row.messages,
            training_sequence_length=training_sequence_length,
        )

    try:
        total = 0
        updates: list[tuple[int, str, int]] = []
        with ThreadPoolExecutor(max_workers=workers) as executor:
            pending: dict[Future[tuple[int, int]], None] = {}
            exhausted = False
            limit = max(1, workers * 2)
            while not exhausted or pending:
                while not exhausted and len(pending) < limit:
                    payload = cursor.fetchone()
                    if payload is None:
                        exhausted = True
                        break
                    pending[executor.submit(count, payload)] = None
                if not pending:
                    continue
                completed, _ = wait(pending, return_when=FIRST_COMPLETED)
                for future in completed:
                    pending.pop(future)
                    ordinal, assistant_tokens = future.result()
                    updates.append((assistant_tokens, arm, ordinal))
                    total += assistant_tokens
                    if len(updates) == 1_024:
                        connection.executemany(
                            "UPDATE selected SET assistant_tokens = ? WHERE arm = ? AND ordinal = ?",
                            updates,
                        )
                        updates.clear()
        if updates:
            connection.executemany(
                "UPDATE selected SET assistant_tokens = ? WHERE arm = ? AND ordinal = ?",
                updates,
            )
        missing = connection.execute(
            "SELECT COUNT(*) FROM selected WHERE arm = ? AND assistant_tokens IS NULL", (arm,)
        ).fetchone()[0]
        if missing:
            raise PilotError("token evidence spool is incomplete")
        connection.commit()
        return total
    except PilotError:
        raise
    except Exception as error:
        raise PilotError("token worker failed") from error


def _spooled_prompt_digest(connection: sqlite3.Connection, arm: str) -> str:
    digest = hashlib.sha256()
    first = True
    for (prompt_uuid,) in connection.execute(
        "SELECT prompt_uuid FROM selected WHERE arm = ? ORDER BY prompt_uuid", (arm,)
    ):
        if not first:
            digest.update(b"\n")
        digest.update(str(prompt_uuid).encode("utf-8"))
        first = False
    return digest.hexdigest()


def _spooled_token_digest(connection: sqlite3.Connection, arm: str) -> str:
    digest = hashlib.sha256()
    for prompt_uuid, assistant_tokens in connection.execute(
        "SELECT prompt_uuid, assistant_tokens FROM selected WHERE arm = ? ORDER BY prompt_uuid",
        (arm,),
    ):
        digest.update(canonical_json([prompt_uuid, assistant_tokens]).encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def _write_spooled_bundle_contents(
    root: Path,
    *,
    connection: sqlite3.Connection,
    selection_stats: _SelectionStats,
    tokens_by_arm: Mapping[str, int],
    config: PilotConfig,
    tokenizer_path: str | Path,
    tokenizer_sha256: str,
    tokenizer_template_sha256: str,
    producer_source_commit: str,
    effective_workers: int,
    execution: Mapping[str, object] | None,
) -> dict[str, PilotArmCompletion]:
    arms: dict[str, PilotArmCompletion] = {}
    for arm in (HISTORICAL_PROPORTION, BALANCED):
        arm_root = root / arm
        arm_root.mkdir(mode=0o700)
        data_path = arm_root / "data.jsonl"
        row_count = 0
        category_counts: dict[str, int] = {}
        with data_path.open("w", encoding="utf-8", newline="\n") as destination:
            for prompt_uuid, split, category, messages_json, assistant_tokens in connection.execute(
                "SELECT prompt_uuid, split, category, messages_json, assistant_tokens "
                "FROM selected WHERE arm = ? ORDER BY ordinal",
                (arm,),
            ):
                if type(assistant_tokens) is not int:
                    raise PilotError("selected token evidence is incomplete")
                try:
                    messages = json.loads(messages_json)
                except json.JSONDecodeError as error:
                    raise PilotError("selected message spool is malformed") from error
                destination.write(
                    canonical_json(
                        {
                            "prompt_uuid": prompt_uuid,
                            "split": split,
                            "category": category,
                            "messages": messages,
                            "assistant_tokens": assistant_tokens,
                        }
                    )
                    + "\n"
                )
                row_count += 1
                category_counts[str(category)] = category_counts.get(str(category), 0) + 1
            destination.flush()
            os.fsync(destination.fileno())
        execution_payload = {
            "schema_version": "qwen3-4b-balanced-pilot-execution-v1",
            "requested_workers": config.workers,
            "effective_workers": effective_workers,
            "metadata": dict(execution) if execution is not None else {},
        }
        execution_payload["execution_sha256"] = _self_hash(execution_payload, "execution_sha256")
        _write_json(arm_root / "EXECUTION.json", execution_payload)
        manifest = {
            "schema_version": "qwen3-4b-balanced-pilot-v1",
            "arm": arm,
            "selection_mode": "pilot-row-quota-v1",
            "source": {
                "repository": config.source_repository,
                "revision": config.source_revision,
            },
            "seed": config.seed,
            "quotas": _quota_counts(config, arm),
            "row_count": row_count,
            "category_counts": dict(sorted(category_counts.items())),
            "prompt_uuid_sha256": _spooled_prompt_digest(connection, arm),
            "token_evidence_sha256": _spooled_token_digest(connection, arm),
            "data_file": {
                "path": "data.jsonl",
                "bytes": data_path.stat().st_size,
                "sha256": _sha256_file(data_path),
            },
            "tokenizer": {
                "path": str(tokenizer_path),
                "sha256": tokenizer_sha256,
                "chat_template_sha256": tokenizer_template_sha256,
            },
            "assistant_tokens": tokens_by_arm[arm],
            "minimum_assistant_tokens": config.minimum_assistant_tokens,
            "training_sequence_length": config.training_sequence_length,
            "exclusions": {
                "held_out": selection_stats.excluded_held_out,
                "invalid": selection_stats.excluded_invalid,
                "duplicate_candidates": selection_stats.excluded_duplicate_candidates,
            },
            "producer_source_commit": producer_source_commit,
        }
        manifest["manifest_sha256"] = _self_hash(manifest, "manifest_sha256")
        _write_json(arm_root / "MANIFEST.json", manifest)
        arms[arm] = PilotArmCompletion(
            arm=arm,
            row_count=row_count,
            assistant_tokens=tokens_by_arm[arm],
            manifest_sha256=str(manifest["manifest_sha256"]),
        )
    complete = {
        "schema_version": "qwen3-4b-balanced-pilot-v1",
        "arms": {
            arm: {
                "manifest_path": f"{arm}/MANIFEST.json",
                "manifest_sha256": arms[arm].manifest_sha256,
            }
            for arm in (HISTORICAL_PROPORTION, BALANCED)
        },
    }
    complete["complete_sha256"] = _self_hash(complete, "complete_sha256")
    _write_json(root / "COMPLETE.json", complete)
    return arms


def _build_verification_trust(
    root: Path,
    *,
    config: PilotConfig,
    arms: Mapping[str, PilotArmCompletion],
    tokenizer_sha256: str,
    chat_template_sha256: str,
    producer_source_commit: str,
) -> PilotVerificationTrust:
    """Capture the producer's immutable preimage for a separate verification caller."""
    return PilotVerificationTrust(
        complete_file_sha256=_sha256_file(root / "COMPLETE.json"),
        historical_manifest_sha256=arms[HISTORICAL_PROPORTION].manifest_sha256,
        balanced_manifest_sha256=arms[BALANCED].manifest_sha256,
        source_repository=config.source_repository,
        source_revision=config.source_revision,
        seed=config.seed,
        historical_quotas=config.historical_quotas,
        balanced_quotas=config.balanced_quotas,
        tokenizer_sha256=tokenizer_sha256,
        chat_template_sha256=chat_template_sha256,
        producer_source_commit=producer_source_commit,
        minimum_assistant_tokens=config.minimum_assistant_tokens,
        training_sequence_length=config.training_sequence_length,
    )


_MANIFEST_FIELDS = {
    "schema_version",
    "arm",
    "selection_mode",
    "source",
    "seed",
    "quotas",
    "row_count",
    "category_counts",
    "prompt_uuid_sha256",
    "token_evidence_sha256",
    "data_file",
    "tokenizer",
    "assistant_tokens",
    "minimum_assistant_tokens",
    "training_sequence_length",
    "exclusions",
    "producer_source_commit",
    "manifest_sha256",
}


def _verify_bundle_root(
    root: Path,
    *,
    production: bool = True,
    tokenizer: Any | None = None,
    scratch_root: Path | None = None,
    verification_trust: PilotVerificationTrust | None = None,
) -> dict[str, PilotArmCompletion]:
    try:
        return _verify_bundle_root_impl(
            Path(root),
            production=production,
            tokenizer=tokenizer,
            scratch_root=scratch_root,
            verification_trust=verification_trust,
        )
    except PilotError:
        raise
    except Exception as error:
        scope = "production " if production else ""
        raise PilotError(f"{scope}pilot completion verification failed") from error


def _verify_bundle_root_impl(
    root: Path,
    *,
    production: bool,
    tokenizer: Any | None,
    scratch_root: Path | None,
    verification_trust: PilotVerificationTrust | None,
) -> dict[str, PilotArmCompletion]:
    if production and not isinstance(verification_trust, PilotVerificationTrust):
        raise PilotError("production verification requires caller-owned verification trust")
    if verification_trust is not None and not isinstance(
        verification_trust, PilotVerificationTrust
    ):
        raise PilotError("caller-owned verification trust has an invalid type")
    complete, complete_raw = _read_json_stable(root / "COMPLETE.json")
    if (
        verification_trust is not None
        and hashlib.sha256(complete_raw).hexdigest() != verification_trust.complete_file_sha256
    ):
        raise PilotError("bundle differs from caller-owned verification trust")
    if (
        set(complete) != {"schema_version", "arms", "complete_sha256"}
        or complete.get("schema_version") != "qwen3-4b-balanced-pilot-v1"
        or not _is_sha256(complete.get("complete_sha256"))
        or complete["complete_sha256"] != _self_hash(complete, "complete_sha256")
    ):
        raise PilotError("aggregate completion schema or self-hash does not reconcile")
    descriptors = complete.get("arms")
    if type(descriptors) is not dict or set(descriptors) != {
        HISTORICAL_PROPORTION,
        BALANCED,
    }:
        raise PilotError("aggregate completion arm descriptors are invalid")
    manifests: dict[str, dict[str, Any]] = {}
    for arm in (HISTORICAL_PROPORTION, BALANCED):
        descriptor = descriptors[arm]
        if (
            type(descriptor) is not dict
            or set(descriptor) != {"manifest_path", "manifest_sha256"}
            or descriptor.get("manifest_path") != f"{arm}/MANIFEST.json"
            or not _is_sha256(descriptor.get("manifest_sha256"))
        ):
            raise PilotError("aggregate completion manifest descriptor is invalid")
        manifest_path = root / f"{arm}/MANIFEST.json"
        manifest, _manifest_raw = _read_json_stable(manifest_path)
        _validate_manifest_schema(manifest, arm=arm, production=production)
        if manifest["manifest_sha256"] != _self_hash(manifest, "manifest_sha256"):
            raise PilotError(f"{arm} manifest self-hash does not reconcile")
        if descriptor["manifest_sha256"] != manifest["manifest_sha256"]:
            raise PilotError(f"{arm} manifest descriptor does not reconcile")
        _verify_execution_receipt(manifest_path.parent / "EXECUTION.json")
        manifests[arm] = manifest
    _validate_cross_arm_pins(manifests)
    if verification_trust is not None:
        _validate_verification_trust(manifests, verification_trust)
    staged: Path | None = None
    owned_scratch: Path | None = None
    verified_tokenizer = tokenizer
    try:
        tokenizer_pin = manifests[HISTORICAL_PROPORTION]["tokenizer"]
        if verified_tokenizer is None:
            if scratch_root is None:
                owned_scratch = Path(
                    tempfile.mkdtemp(prefix=".qwen4b-verify-", dir=tempfile.gettempdir())
                )
                os.chmod(owned_scratch, 0o700)
                verifier_scratch = owned_scratch
            else:
                verifier_scratch = _validate_private_scratch_root(Path(scratch_root))
            staged, digest = _stage_verified_tokenizer_snapshot(
                Path(tokenizer_pin["path"]),
                tokenizer_pin["sha256"],
                verifier_scratch,
            )
            verified_tokenizer = _load_verified_qwen_tokenizer(staged)
            preflight = _qwen3_tokenizer_preflight_evidence(verified_tokenizer, digest)
            if preflight.chat_template_sha256 != tokenizer_pin["chat_template_sha256"]:
                raise PilotError("authenticated Qwen3 tokenizer template does not reconcile")
        else:
            template = getattr(verified_tokenizer, "chat_template", None)
            if (
                not isinstance(template, str)
                or hashlib.sha256(template.encode("utf-8")).hexdigest()
                != tokenizer_pin["chat_template_sha256"]
            ):
                raise PilotError("test tokenizer template does not reconcile")
        arms: dict[str, PilotArmCompletion] = {}
        for arm in (HISTORICAL_PROPORTION, BALANCED):
            manifest = manifests[arm]
            evidence = _read_data_evidence(
                root / arm / "data.jsonl",
                training_sequence_length=manifest["training_sequence_length"],
                tokenizer=verified_tokenizer,
            )
            _reconcile_manifest_evidence(manifest, evidence, arm=arm, production=production)
            arms[arm] = PilotArmCompletion(
                arm=arm,
                row_count=evidence.row_count,
                assistant_tokens=evidence.assistant_tokens,
                manifest_sha256=manifest["manifest_sha256"],
            )
    finally:
        if staged is not None and staged.exists():
            shutil.rmtree(staged)
        if owned_scratch is not None and owned_scratch.exists():
            shutil.rmtree(owned_scratch)
    return arms


def _validate_manifest_schema(manifest: dict[str, Any], *, arm: str, production: bool) -> None:
    if set(manifest) != _MANIFEST_FIELDS:
        raise PilotError(f"{arm} manifest schema is invalid")
    if (
        manifest.get("schema_version") != "qwen3-4b-balanced-pilot-v1"
        or manifest.get("arm") != arm
        or manifest.get("selection_mode") != "pilot-row-quota-v1"
    ):
        raise PilotError(f"{arm} manifest arm/schema identity is invalid")
    if type(manifest.get("seed")) is not int:
        raise PilotError(f"{arm} seed must be a strict integer")
    source = manifest.get("source")
    if (
        type(source) is not dict
        or set(source) != {"repository", "revision"}
        or source.get("repository") != "nvidia/Nemotron-Post-Training-Dataset-v2"
        or type(source.get("revision")) is not str
        or len(source["revision"]) not in (40, 64)
        or any(character not in "0123456789abcdef" for character in source["revision"])
    ):
        raise PilotError(f"{arm} pinned source identity is invalid")
    quotas = manifest.get("quotas")
    if (
        type(quotas) is not dict
        or not quotas
        or any(
            type(key) is not str or type(value) is not int or value < 1
            for key, value in quotas.items()
        )
    ):
        raise PilotError(f"{arm} exact quotas are invalid")
    category_counts = manifest.get("category_counts")
    if (
        type(category_counts) is not dict
        or not category_counts
        or any(
            type(key) is not str or type(value) is not int or value < 1
            for key, value in category_counts.items()
        )
    ):
        raise PilotError(f"{arm} category counts are invalid")
    data = manifest.get("data_file")
    if (
        type(data) is not dict
        or set(data) != {"path", "bytes", "sha256"}
        or data.get("path") != "data.jsonl"
        or type(data.get("bytes")) is not int
        or data["bytes"] < 1
        or not _is_sha256(data.get("sha256"))
    ):
        raise PilotError(f"{arm} data file descriptor is invalid")
    tokenizer = manifest.get("tokenizer")
    if (
        type(tokenizer) is not dict
        or set(tokenizer) != {"path", "sha256", "chat_template_sha256"}
        or type(tokenizer.get("path")) is not str
        or not tokenizer["path"]
        or not _is_sha256(tokenizer.get("sha256"))
        or not _is_sha256(tokenizer.get("chat_template_sha256"))
    ):
        raise PilotError(f"{arm} tokenizer/template identity is invalid")
    for field in (
        "row_count",
        "assistant_tokens",
        "minimum_assistant_tokens",
        "training_sequence_length",
    ):
        if type(manifest.get(field)) is not int or manifest[field] < 1:
            label = "assistant-token" if "assistant" in field else field.replace("_", " ")
            raise PilotError(f"{arm} {label} evidence is invalid")
    exclusions = manifest.get("exclusions")
    if (
        type(exclusions) is not dict
        or set(exclusions) != {"held_out", "invalid", "duplicate_candidates"}
        or any(type(value) is not int or value < 0 for value in exclusions.values())
    ):
        raise PilotError(f"{arm} exclusion evidence is invalid")
    if not _is_sha256(manifest.get("prompt_uuid_sha256")) or not _is_sha256(
        manifest.get("token_evidence_sha256")
    ):
        raise PilotError(f"{arm} token/prompt evidence digest is invalid")
    producer = manifest.get("producer_source_commit")
    if (
        type(producer) is not str
        or len(producer) not in (40, 64)
        or any(character not in "0123456789abcdef" for character in producer)
        or not _is_sha256(manifest.get("manifest_sha256"))
    ):
        raise PilotError(f"{arm} producer or manifest identity is invalid")
    if production:
        approved = PilotConfig(source_revision=source["revision"])
        if (
            manifest["seed"] != SEED
            or manifest["quotas"] != _quota_counts(approved, arm)
            or manifest["row_count"] != 100_000
            or manifest["minimum_assistant_tokens"] != 16_000_000
            or manifest["training_sequence_length"] != TRAINING_SEQUENCE_LENGTH
        ):
            raise PilotError(f"{arm} production scientific identity is invalid")


def _verify_execution_receipt(path: Path) -> None:
    receipt = _read_json(path)
    if (
        set(receipt)
        != {
            "schema_version",
            "requested_workers",
            "effective_workers",
            "metadata",
            "execution_sha256",
        }
        or receipt.get("schema_version") != "qwen3-4b-balanced-pilot-execution-v1"
        or type(receipt.get("requested_workers")) is not int
        or receipt["requested_workers"] < 1
        or type(receipt.get("effective_workers")) is not int
        or receipt["effective_workers"] < 1
        or type(receipt.get("metadata")) is not dict
        or not _is_sha256(receipt.get("execution_sha256"))
        or receipt["execution_sha256"] != _self_hash(receipt, "execution_sha256")
    ):
        raise PilotError("execution receipt schema or self-hash does not reconcile")


def _validate_private_scratch_root(path: Path) -> Path:
    try:
        observed = path.lstat()
    except OSError as error:
        raise PilotError("verifier scratch root is unavailable") from error
    if (
        not stat.S_ISDIR(observed.st_mode)
        or stat.S_ISLNK(observed.st_mode)
        or observed.st_mode & 0o077
        or (hasattr(os, "geteuid") and observed.st_uid != os.geteuid())
    ):
        raise PilotError("verifier scratch root must be a caller-owned private directory")
    return path


def _stable_stat_identity(value: os.stat_result) -> tuple[int, int, int, int, int, int, int]:
    return (
        value.st_dev,
        value.st_ino,
        value.st_mode,
        value.st_nlink,
        value.st_size,
        value.st_mtime_ns,
        value.st_ctime_ns,
    )


def _read_data_evidence(
    path: Path, *, training_sequence_length: int, tokenizer: Any
) -> _DataEvidence:
    row_count = 0
    split_counts: dict[str, int] = {}
    category_counts: dict[str, int] = {}
    prompt_uuids: set[str] = set()
    token_pairs: list[tuple[str, int]] = []
    assistant_tokens = 0
    data_digest = hashlib.sha256()
    data_bytes = 0
    parent_fd: int | None = None
    data_fd: int | None = None
    try:
        parent_fd = _open_nofollow_directory(path.parent)
        parent_initial = os.fstat(parent_fd)
        data_fd = os.open(
            path.name,
            os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0),
            dir_fd=parent_fd,
        )
        data_initial = os.fstat(data_fd)
        if not stat.S_ISREG(data_initial.st_mode):
            raise PilotError("data file is not a regular file")
        with os.fdopen(data_fd, "rb", closefd=False) as source:
            for raw_line in source:
                data_digest.update(raw_line)
                data_bytes += len(raw_line)
                try:
                    line = raw_line.decode("utf-8")
                except UnicodeDecodeError as error:
                    raise PilotError("data file JSONL row is malformed") from error
                try:
                    payload = json.loads(line)
                    if type(payload) is not dict or set(payload) != {
                        "prompt_uuid",
                        "split",
                        "category",
                        "messages",
                        "assistant_tokens",
                    }:
                        raise PilotError("data JSONL row schema is invalid")
                    split = payload["split"]
                    if type(split) is not str:
                        raise PilotError("data JSONL split is invalid")
                    row = _normalize_row(split, payload)
                    declared_count = payload["assistant_tokens"]
                    if (
                        type(declared_count) is not int
                        or declared_count < 1
                        or declared_count > training_sequence_length
                    ):
                        raise PilotError("data JSONL assistant-token evidence is invalid")
                except (KeyError, TypeError, json.JSONDecodeError, PilotError) as error:
                    raise PilotError("data file JSONL row is malformed") from error
                if (
                    row.category != payload["category"]
                    or row.prompt_uuid != payload["prompt_uuid"]
                    or row.prompt_uuid in prompt_uuids
                ):
                    raise PilotError(
                        "data JSONL identity or global UUID uniqueness does not reconcile"
                    )
                count = count_assistant_tokens(
                    tokenizer,
                    row.messages,
                    training_sequence_length=training_sequence_length,
                )
                if declared_count != count:
                    raise PilotError(
                        "data JSONL assistant-token evidence does not match authenticated tokenizer"
                    )
                prompt_uuids.add(row.prompt_uuid)
                token_pairs.append((row.prompt_uuid, count))
                row_count += 1
                assistant_tokens += count
                split_counts[row.split] = split_counts.get(row.split, 0) + 1
                category_counts[row.category] = category_counts.get(row.category, 0) + 1
        data_final = os.fstat(data_fd)
        parent_final = os.fstat(parent_fd)
        named_final = os.stat(path.name, dir_fd=parent_fd, follow_symlinks=False)
        if (
            _stable_stat_identity(data_initial) != _stable_stat_identity(data_final)
            or _stable_stat_identity(data_initial) != _stable_stat_identity(named_final)
            or _stable_stat_identity(parent_initial) != _stable_stat_identity(parent_final)
        ):
            raise PilotError("data file changed during verification")
    except OSError as error:
        raise PilotError("data file changed during verification") from error
    finally:
        if data_fd is not None:
            os.close(data_fd)
        if parent_fd is not None:
            os.close(parent_fd)
    prompt_digest = hashlib.sha256("\n".join(sorted(prompt_uuids)).encode("utf-8")).hexdigest()
    token_digest = hashlib.sha256()
    for prompt_uuid, count in sorted(token_pairs):
        token_digest.update(canonical_json([prompt_uuid, count]).encode("utf-8"))
        token_digest.update(b"\n")
    return _DataEvidence(
        row_count=row_count,
        split_counts=dict(sorted(split_counts.items())),
        category_counts=dict(sorted(category_counts.items())),
        prompt_uuid_sha256=prompt_digest,
        assistant_tokens=assistant_tokens,
        token_evidence_sha256=token_digest.hexdigest(),
        data_bytes=data_bytes,
        data_sha256=data_digest.hexdigest(),
    )


def _reconcile_manifest_evidence(
    manifest: dict[str, Any], evidence: _DataEvidence, *, arm: str, production: bool
) -> None:
    data_file = manifest["data_file"]
    if data_file["bytes"] != evidence.data_bytes or data_file["sha256"] != evidence.data_sha256:
        raise PilotError(f"{arm} data file descriptor does not reconcile")
    if manifest["row_count"] != evidence.row_count:
        raise PilotError(f"{arm} row count does not reconcile")
    if manifest["quotas"] != evidence.split_counts:
        raise PilotError(f"{arm} exact split quotas do not reconcile")
    if manifest["category_counts"] != evidence.category_counts:
        raise PilotError(f"{arm} category counts do not reconcile")
    if manifest["prompt_uuid_sha256"] != evidence.prompt_uuid_sha256:
        raise PilotError(f"{arm} prompt UUID digest does not reconcile")
    if (
        manifest["assistant_tokens"] != evidence.assistant_tokens
        or manifest["token_evidence_sha256"] != evidence.token_evidence_sha256
        or evidence.assistant_tokens < manifest["minimum_assistant_tokens"]
        or (production and evidence.assistant_tokens < 16_000_000)
    ):
        raise PilotError(f"{arm} assistant-token evidence does not reconcile")


def _validate_cross_arm_pins(manifests: Mapping[str, dict[str, Any]]) -> None:
    fields = (
        "schema_version",
        "selection_mode",
        "source",
        "seed",
        "tokenizer",
        "minimum_assistant_tokens",
        "training_sequence_length",
        "producer_source_commit",
    )
    historical = manifests[HISTORICAL_PROPORTION]
    balanced = manifests[BALANCED]
    if any(historical[field] != balanced[field] for field in fields):
        raise PilotError("cross-arm scientific pins do not reconcile")


def _validate_verification_trust(
    manifests: Mapping[str, dict[str, Any]], trust: PilotVerificationTrust
) -> None:
    historical = manifests[HISTORICAL_PROPORTION]
    balanced = manifests[BALANCED]
    expected_common = {
        "source": {
            "repository": trust.source_repository,
            "revision": trust.source_revision,
        },
        "seed": trust.seed,
        "tokenizer": {
            "path": historical["tokenizer"]["path"],
            "sha256": trust.tokenizer_sha256,
            "chat_template_sha256": trust.chat_template_sha256,
        },
        "minimum_assistant_tokens": trust.minimum_assistant_tokens,
        "training_sequence_length": trust.training_sequence_length,
        "producer_source_commit": trust.producer_source_commit,
    }
    for manifest in (historical, balanced):
        if any(manifest[field] != value for field, value in expected_common.items()):
            raise PilotError("bundle differs from caller-owned verification trust")
    if (
        historical["manifest_sha256"] != trust.historical_manifest_sha256
        or balanced["manifest_sha256"] != trust.balanced_manifest_sha256
        or historical["quotas"] != {quota.split: quota.rows for quota in trust.historical_quotas}
        or balanced["quotas"] != {quota.split: quota.rows for quota in trust.balanced_quotas}
    ):
        raise PilotError("bundle differs from caller-owned verification trust")


def _quota_counts(config: PilotConfig, arm: str) -> dict[str, int]:
    quotas = config.historical_quotas if arm == HISTORICAL_PROPORTION else config.balanced_quotas
    return {quota.split: quota.rows for quota in quotas}


def _write_json(path: Path, payload: Mapping[str, object]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as destination:
        destination.write(canonical_json(payload) + "\n")
        destination.flush()
        os.fsync(destination.fileno())


def _read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise PilotError(f"unable to read receipt: {path}") from error
    if not isinstance(payload, dict):
        raise PilotError(f"receipt must be a JSON object: {path}")
    return payload


def _read_file_stable(path: Path) -> bytes:
    parent_fd: int | None = None
    descriptor: int | None = None
    try:
        parent_fd = _open_nofollow_directory(path.parent)
        parent_initial = os.fstat(parent_fd)
        descriptor = os.open(
            path.name,
            os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0),
            dir_fd=parent_fd,
        )
        initial = os.fstat(descriptor)
        if not stat.S_ISREG(initial.st_mode):
            raise PilotError(f"receipt is not a regular file: {path}")
        chunks: list[bytes] = []
        while chunk := os.read(descriptor, 1024 * 1024):
            chunks.append(chunk)
        final = os.fstat(descriptor)
        named = os.stat(path.name, dir_fd=parent_fd, follow_symlinks=False)
        parent_final = os.fstat(parent_fd)
        if (
            _stable_stat_identity(initial) != _stable_stat_identity(final)
            or _stable_stat_identity(initial) != _stable_stat_identity(named)
            or _stable_stat_identity(parent_initial) != _stable_stat_identity(parent_final)
        ):
            raise PilotError(f"file changed during verification: {path}")
        return b"".join(chunks)
    except OSError as error:
        raise PilotError(f"file changed during verification: {path}") from error
    finally:
        if descriptor is not None:
            os.close(descriptor)
        if parent_fd is not None:
            os.close(parent_fd)


def _read_json_stable(path: Path) -> tuple[dict[str, Any], bytes]:
    raw = _read_file_stable(path)
    try:
        payload = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise PilotError(f"unable to read receipt: {path}") from error
    if not isinstance(payload, dict):
        raise PilotError(f"receipt must be a JSON object: {path}")
    return payload, raw


def _file_evidence_stable(path: Path) -> tuple[int, str]:
    parent_fd: int | None = None
    descriptor: int | None = None
    try:
        parent_fd = _open_nofollow_directory(path.parent)
        parent_initial = os.fstat(parent_fd)
        descriptor = os.open(
            path.name,
            os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0),
            dir_fd=parent_fd,
        )
        initial = os.fstat(descriptor)
        if not stat.S_ISREG(initial.st_mode):
            raise PilotError(f"bundle component is not a regular file: {path}")
        digest = hashlib.sha256()
        size = 0
        while chunk := os.read(descriptor, 1024 * 1024):
            digest.update(chunk)
            size += len(chunk)
        final = os.fstat(descriptor)
        named = os.stat(path.name, dir_fd=parent_fd, follow_symlinks=False)
        parent_final = os.fstat(parent_fd)
        if (
            _stable_stat_identity(initial) != _stable_stat_identity(final)
            or _stable_stat_identity(initial) != _stable_stat_identity(named)
            or _stable_stat_identity(parent_initial) != _stable_stat_identity(parent_final)
        ):
            raise PilotError(f"bundle component changed during verification: {path}")
        return size, digest.hexdigest()
    except OSError as error:
        raise PilotError(f"bundle component changed during verification: {path}") from error
    finally:
        if descriptor is not None:
            os.close(descriptor)
        if parent_fd is not None:
            os.close(parent_fd)


_BUNDLE_FILES = tuple(
    [Path("COMPLETE.json")]
    + [
        Path(arm) / name
        for arm in (HISTORICAL_PROPORTION, BALANCED)
        for name in ("data.jsonl", "EXECUTION.json", "MANIFEST.json")
    ]
)


def _stable_directory_names(path: Path) -> tuple[str, ...]:
    descriptor = _open_nofollow_directory(path)
    try:
        initial = os.fstat(descriptor)
        names = tuple(sorted(os.listdir(descriptor)))
        final = os.fstat(descriptor)
        if _stable_stat_identity(initial) != _stable_stat_identity(final):
            raise PilotError(f"bundle directory changed during verification: {path}")
        return names
    finally:
        os.close(descriptor)


def _bundle_file_evidence(root: Path) -> dict[str, tuple[int, str]]:
    if _stable_directory_names(root) != (
        "COMPLETE.json",
        BALANCED,
        HISTORICAL_PROPORTION,
    ):
        raise PilotError("bundle root file set is invalid")
    for arm in (HISTORICAL_PROPORTION, BALANCED):
        if _stable_directory_names(root / arm) != (
            "EXECUTION.json",
            "MANIFEST.json",
            "data.jsonl",
        ):
            raise PilotError(f"{arm} bundle file set is invalid")
    evidence: dict[str, tuple[int, str]] = {}
    for relative in _BUNDLE_FILES:
        evidence[relative.as_posix()] = _file_evidence_stable(root / relative)
    return evidence


def _verify_bundle_file_evidence(root: Path, expected: Mapping[str, tuple[int, str]]) -> None:
    if _bundle_file_evidence(root) != dict(expected):
        raise PilotError("bundle copy differs from authenticated producer evidence")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _self_hash(payload: Mapping[str, object], field: str) -> str:
    stripped = dict(payload)
    stripped.pop(field, None)
    return hashlib.sha256(canonical_json(stripped).encode("utf-8")).hexdigest()


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _fsync_tree(root: Path) -> None:
    """Make every copied file and directory durable before the root is renamed."""
    directories = [root]
    for path in root.rglob("*"):
        if path.is_symlink():
            raise PilotError("publication copy unexpectedly contains a symlink")
        if path.is_dir():
            directories.append(path)
            continue
        if not path.is_file():
            raise PilotError("publication copy contains a non-regular component")
        descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    for directory in sorted(directories, key=lambda value: len(value.parts), reverse=True):
        _fsync_directory(directory)
