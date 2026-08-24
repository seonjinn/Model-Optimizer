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

"""Validate target generations and deterministically promote prompt reserves."""

from __future__ import annotations

import argparse
import json
import os
import re
import sqlite3
import urllib.error
import urllib.request
from collections import Counter, OrderedDict, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from hashlib import sha256
from itertools import chain
from pathlib import Path
from types import MappingProxyType
from typing import Any, Literal

from select_bprime_cd_prompts import PromptView, SelectedPrompt
from specdec_corpus_contracts import canonical_json
from trajectory_schema import TrajectoryValidationError, validate_trajectory

__all__ = [
    "GenerationAttempt",
    "GenerationIdentity",
    "PromotedResponseRecord",
    "ResponseCorpus",
    "ResponsePromotionError",
    "ValidatedReplayRecord",
    "promote_response_artifacts",
    "promote_responses",
    "validated_replay_record",
]

_SHA = re.compile(r"^[0-9a-f]{40}$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_LANGUAGE_CELLS = {"ja": "japanese", "es": "spanish", "fr": "french", "it": "italian"}


class ResponsePromotionError(ValueError):
    """An immutable attempt or exact cell quota failed validation."""


def _require_digest(name: str, value: str, pattern: re.Pattern[str] = _SHA256) -> None:
    if not pattern.fullmatch(value):
        raise ValueError(f"{name} must be an exact lowercase digest")


def _canonical_text(value: Any) -> str:
    return canonical_json(value).decode("utf-8")


@dataclass(frozen=True)
class GenerationIdentity:
    """Complete target, template, runtime, mode, and generation identity."""

    target_revision: str
    tokenizer_sha256: str
    chat_template_sha256: str
    runtime_sha256: str
    container_sha256: str
    source_selection_sha256: str
    thinking_mode: Literal["on", "off"]
    temperature: float
    max_tokens: int
    max_total_length: int

    def __post_init__(self) -> None:
        _require_digest("target_revision", self.target_revision, _SHA)
        for name in (
            "tokenizer_sha256",
            "chat_template_sha256",
            "runtime_sha256",
            "container_sha256",
            "source_selection_sha256",
        ):
            _require_digest(name, getattr(self, name))
        if self.thinking_mode not in {"on", "off"}:
            raise ValueError("thinking_mode must be on or off")
        if (
            not isinstance(self.temperature, (int, float))
            or isinstance(self.temperature, bool)
            or self.temperature < 0
        ):
            raise ValueError("temperature must be non-negative")
        if (
            not isinstance(self.max_tokens, int)
            or isinstance(self.max_tokens, bool)
            or not isinstance(self.max_total_length, int)
            or isinstance(self.max_total_length, bool)
            or self.max_tokens < 1
            or self.max_total_length < self.max_tokens
        ):
            raise ValueError("generation token limits are invalid")

    @property
    def sha256(self) -> str:
        """Return the canonical digest of the complete generation contract."""
        return sha256(canonical_json(asdict(self))).hexdigest()


@dataclass(frozen=True)
class GenerationAttempt:
    """One immutable request/response attempt with content hashes."""

    attempt_id: str
    prompt_uuid: str
    generation_identity_sha256: str
    request_json: str
    request_sha256: str
    response_json: str
    response_sha256: str


@dataclass(frozen=True)
class ValidatedReplayRecord:
    """One canonical D replay record carrying its prior validation proof."""

    prompt_uuid: str
    canonical_record_json: str
    record_sha256: str
    trajectory_validation_sha256: str
    assistant_tokens: int


@dataclass(frozen=True)
class PromotedResponseRecord:
    """One Task 7-ready promoted row with authenticated content and provenance."""

    prompt_uuid: str
    arm: str
    domain: str
    lane: str
    language: str
    context_bucket: str
    canonical_record_json: str
    request_sha256: str | None
    response_sha256: str | None
    selection_sha256: str
    paired_cd_sha256: str
    generation_identity_sha256: str
    attempt_or_replay_validation_sha256: str
    record_sha256: str


@dataclass(frozen=True)
class ResponseCorpus:
    """An exact promoted corpus with attempt receipts and token histograms."""

    arm: str
    generation_identity: GenerationIdentity
    generation_identity_sha256: str
    source_selection_sha256: str
    promoted_ids: Sequence[str]
    cell_counts: Mapping[str, int]
    successful_attempts: Sequence[GenerationAttempt]
    failed_attempts: Sequence[GenerationAttempt]
    replay_records: Sequence[ValidatedReplayRecord]
    failed_prompt_ids: Sequence[str]
    unused_reserve_ids: Sequence[str]
    attempt_history: Mapping[str, tuple[GenerationAttempt, ...]]
    response_token_histograms: Mapping[str, Mapping[int, int]]
    records: Sequence[PromotedResponseRecord]
    corpus_sha256: str


@dataclass(frozen=True)
class _ValidatedSuccess:
    attempt: GenerationAttempt
    assistant_tokens: int


def _replay_validation_sha256(row: SelectedPrompt, canonical_bytes: bytes) -> str:
    proof = {
        "prompt_uuid": row.prompt_uuid,
        "lane": row.lane,
        "source_id": row.source_id,
        "canonical_sha256": sha256(canonical_bytes).hexdigest(),
    }
    return sha256(canonical_json(proof)).hexdigest()


def validated_replay_record(row: SelectedPrompt, *, assistant_tokens: int) -> ValidatedReplayRecord:
    """Validate a selected D replay row and return its authenticated receipt."""
    try:
        value = json.loads(row.canonical_prompt_json)
        validation = validate_trajectory(value, source_id=row.source_id, lane=row.lane)
    except (json.JSONDecodeError, TrajectoryValidationError) as error:
        raise ResponsePromotionError(
            f"replay trajectory validation failed: {row.prompt_uuid}"
        ) from error
    if validation.canonical_bytes.decode("utf-8") != row.canonical_prompt_json:
        raise ResponsePromotionError(f"replay canonical identity mismatch: {row.prompt_uuid}")
    if (
        not isinstance(assistant_tokens, int)
        or isinstance(assistant_tokens, bool)
        or assistant_tokens < 1
    ):
        raise ResponsePromotionError("replay assistant token count must be positive")
    return ValidatedReplayRecord(
        prompt_uuid=row.prompt_uuid,
        canonical_record_json=row.canonical_prompt_json,
        record_sha256=sha256(row.canonical_prompt_json.encode()).hexdigest(),
        trajectory_validation_sha256=_replay_validation_sha256(row, validation.canonical_bytes),
        assistant_tokens=assistant_tokens,
    )


def _prompt_cell(view: PromptView, row: SelectedPrompt) -> str:
    if row.domain in view.cell_counts:
        return row.domain
    if row.domain == "stem-science" and "stem" in view.cell_counts:
        return "stem"
    if row.domain == "multilingual":
        cell = _LANGUAGE_CELLS.get(row.language)
        if cell is not None and cell in view.cell_counts:
            return cell
    raise ResponsePromotionError(f"prompt {row.prompt_uuid} has no frozen cell")


def _expected_messages(row: SelectedPrompt, identity: GenerationIdentity) -> list[dict[str, Any]]:
    try:
        prompt = json.loads(row.canonical_prompt_json)
    except json.JSONDecodeError as error:
        raise ResponsePromotionError(f"prompt {row.prompt_uuid} is not canonical JSON") from error
    if not isinstance(prompt, dict) or not isinstance(prompt.get("messages"), list):
        raise ResponsePromotionError(f"prompt {row.prompt_uuid} has no message list")
    messages: list[dict[str, Any]] = []
    for message in prompt["messages"]:
        if not isinstance(message, dict) or not isinstance(message.get("role"), str):
            raise ResponsePromotionError(f"prompt {row.prompt_uuid} has a malformed message")
        if message["role"] == "assistant":
            raise ResponsePromotionError(
                f"prompt {row.prompt_uuid} contains a source assistant completion"
            )
        if message["role"] == "tool" or message.get("tool_calls") or prompt.get("tools"):
            raise ResponsePromotionError(
                f"prompt {row.prompt_uuid} contains target-ineligible tools"
            )
        normalized = dict(message)
        if identity.thinking_mode == "off" and message["role"] == "user":
            content = message.get("content")
            if not isinstance(content, str):
                raise ResponsePromotionError(f"prompt {row.prompt_uuid} has non-text user content")
            normalized["content"] = f"{content} /no_think"
        messages.append(normalized)
    return messages


def _validate_attempt(
    attempt: GenerationAttempt,
    row: SelectedPrompt,
    identity: GenerationIdentity,
) -> _ValidatedSuccess | None:
    if not attempt.attempt_id or not _SHA256.fullmatch(attempt.attempt_id):
        raise ResponsePromotionError("missing or malformed attempt identity")
    if attempt.prompt_uuid != row.prompt_uuid:
        raise ResponsePromotionError("attempt prompt identity mismatch")
    if attempt.generation_identity_sha256 != identity.sha256:
        raise ResponsePromotionError("attempt generation identity mismatch")
    for name, payload, claimed in (
        ("request", attempt.request_json, attempt.request_sha256),
        ("response", attempt.response_json, attempt.response_sha256),
    ):
        if sha256(payload.encode("utf-8")).hexdigest() != claimed:
            raise ResponsePromotionError(f"attempt {name} SHA-256 mismatch")
        try:
            decoded = json.loads(payload)
        except json.JSONDecodeError as error:
            raise ResponsePromotionError(f"attempt {name} is not JSON") from error
        if payload != _canonical_text(decoded):
            raise ResponsePromotionError(f"attempt {name} is not canonical JSON")
    request = json.loads(attempt.request_json)
    if not isinstance(request, dict) or set(request) != {
        "messages",
        "generation_identity_sha256",
    }:
        raise ResponsePromotionError("attempt request contract mismatch")
    if request["generation_identity_sha256"] != identity.sha256:
        raise ResponsePromotionError("attempt request generation identity mismatch")
    if request["messages"] != _expected_messages(row, identity):
        raise ResponsePromotionError("attempt request does not match the frozen prompt")

    response = json.loads(attempt.response_json)
    if not isinstance(response, dict):
        return None
    message = response.get("message")
    tokens = response.get("completion_tokens")
    if (
        set(response) != {"message", "finish_reason", "completion_tokens"}
        or response.get("finish_reason") != "stop"
        or not isinstance(message, dict)
        or set(message) != {"role", "content"}
        or message.get("role") != "assistant"
        or not isinstance(message.get("content"), str)
        or not message["content"].strip()
        or not isinstance(tokens, int)
        or isinstance(tokens, bool)
        or tokens < 1
    ):
        return None
    return _ValidatedSuccess(attempt, tokens)


def _indexed_rows(
    view: PromptView,
) -> tuple[dict[str, SelectedPrompt], dict[str, list[SelectedPrompt]]]:
    by_id: dict[str, SelectedPrompt] = {}
    reserves: dict[str, list[SelectedPrompt]] = defaultdict(list)
    for row in (*view.primary_rows, *view.reserve_rows):
        if row.prompt_uuid in by_id:
            raise ResponsePromotionError(f"duplicate selected prompt: {row.prompt_uuid}")
        by_id[row.prompt_uuid] = row
        if row.status == "reserve":
            reserves[_prompt_cell(view, row)].append(row)
    return by_id, reserves


def _promoted_record(
    row: SelectedPrompt,
    identity: GenerationIdentity,
    canonical: Mapping[str, Any],
    *,
    selection_sha256: str,
    paired_cd_sha256: str,
    request_sha256: str | None,
    response_sha256: str | None,
    validation_sha256: str,
) -> PromotedResponseRecord:
    record_without_digest = {
        "prompt_uuid": row.prompt_uuid,
        "arm": row.arm,
        "domain": row.domain,
        "lane": row.lane,
        "language": row.language,
        "context_bucket": row.context_bucket,
        "canonical_record_json": _canonical_text(canonical),
        "request_sha256": request_sha256,
        "response_sha256": response_sha256,
        "selection_sha256": selection_sha256,
        "paired_cd_sha256": paired_cd_sha256,
        "generation_identity_sha256": identity.sha256,
        "attempt_or_replay_validation_sha256": validation_sha256,
    }
    return PromotedResponseRecord(
        **record_without_digest,
        record_sha256=sha256(canonical_json(record_without_digest)).hexdigest(),
    )


def promote_responses(
    view: PromptView,
    attempts: Sequence[GenerationAttempt],
    identity: GenerationIdentity,
    *,
    replay_records: Sequence[ValidatedReplayRecord] = (),
) -> ResponseCorpus:
    """Validate attempts and fill each failed primary from its same-cell reserve."""
    by_id, reserves = _indexed_rows(view)
    for row in by_id.values():
        if row.lane == "target-synth":
            _expected_messages(row, identity)
    histories: dict[str, list[GenerationAttempt]] = defaultdict(list)
    seen_attempts: dict[str, GenerationAttempt] = {}
    successes: dict[str, _ValidatedSuccess] = {}
    failures: list[GenerationAttempt] = []
    replay_by_id: dict[str, ValidatedReplayRecord] = {}

    if replay_records and view.arm != "D":
        raise ResponsePromotionError("validated replay records may join only the D arm")
    for replay in replay_records:
        if replay.prompt_uuid in replay_by_id:
            raise ResponsePromotionError(f"duplicate validated replay: {replay.prompt_uuid}")
        row = by_id.get(replay.prompt_uuid)
        if row is None or row.lane == "target-synth" or row.status != "primary":
            raise ResponsePromotionError(
                f"validated replay is not a frozen D primary: {replay.prompt_uuid}"
            )
        try:
            replay_value = json.loads(replay.canonical_record_json)
        except json.JSONDecodeError as error:
            raise ResponsePromotionError(
                f"validated replay is not canonical JSON: {replay.prompt_uuid}"
            ) from error
        try:
            validation = validate_trajectory(
                replay_value,
                source_id=row.source_id,
                lane=row.lane,
            )
        except TrajectoryValidationError as error:
            raise ResponsePromotionError(
                f"replay trajectory validation failed: {replay.prompt_uuid}"
            ) from error
        if (
            not _SHA256.fullmatch(replay.trajectory_validation_sha256)
            or sha256(replay.canonical_record_json.encode()).hexdigest() != replay.record_sha256
            or replay.canonical_record_json != _canonical_text(replay_value)
            or replay.canonical_record_json != row.canonical_prompt_json
            or validation.canonical_bytes.decode("utf-8") != replay.canonical_record_json
            or replay.trajectory_validation_sha256
            != _replay_validation_sha256(row, validation.canonical_bytes)
            or isinstance(replay.assistant_tokens, bool)
            or replay.assistant_tokens < 1
        ):
            raise ResponsePromotionError(f"validated replay proof mismatch: {replay.prompt_uuid}")
        replay_by_id[replay.prompt_uuid] = replay

    for attempt in attempts:
        prior = seen_attempts.get(attempt.attempt_id)
        if prior is not None:
            if prior != attempt:
                raise ResponsePromotionError("attempt identity is bound to conflicting content")
            continue
        seen_attempts[attempt.attempt_id] = attempt
        row = by_id.get(attempt.prompt_uuid)
        if row is None:
            raise ResponsePromotionError(
                f"attempt references unselected prompt: {attempt.prompt_uuid}"
            )
        if row.lane != "target-synth":
            raise ResponsePromotionError(
                f"target attempt references replay prompt: {attempt.prompt_uuid}"
            )
        histories[attempt.prompt_uuid].append(attempt)
        validated = _validate_attempt(attempt, row, identity)
        if validated is None:
            failures.append(attempt)
            continue
        prior_success = successes.get(attempt.prompt_uuid)
        if (
            prior_success is not None
            and prior_success.attempt.response_sha256 != attempt.response_sha256
        ):
            raise ResponsePromotionError(f"conflicting successful responses: {attempt.prompt_uuid}")
        successes.setdefault(attempt.prompt_uuid, validated)

    target_primaries = [row for row in view.primary_rows if row.lane == "target-synth"]
    replay_primaries = [row for row in view.primary_rows if row.lane != "target-synth"]
    missing_replay = [
        row.prompt_uuid for row in replay_primaries if row.prompt_uuid not in replay_by_id
    ]
    if missing_replay:
        raise ResponsePromotionError(f"validated replay is missing: {missing_replay[0]}")
    promoted_primary = [row for row in target_primaries if row.prompt_uuid in successes]
    needed = Counter(
        _prompt_cell(view, row) for row in target_primaries if row.prompt_uuid not in successes
    )
    replacements: list[SelectedPrompt] = []
    unused: list[str] = []
    for cell, rows in reserves.items():
        for row in rows:
            if row.lane != "target-synth":
                unused.append(row.prompt_uuid)
                continue
            if needed[cell] and row.prompt_uuid in successes:
                replacements.append(row)
                needed[cell] -= 1
            else:
                unused.append(row.prompt_uuid)
    shortfalls = {cell: count for cell, count in needed.items() if count}
    if shortfalls:
        cell, count = next(iter(shortfalls.items()))
        raise ResponsePromotionError(f"{cell} cell quota has {count} unresolved responses")

    promoted_targets = [*promoted_primary, *replacements]
    promoted = [*promoted_targets, *replay_primaries]
    counts = Counter(_prompt_cell(view, row) for row in promoted)
    if dict(counts) != dict(view.cell_counts):
        raise ResponsePromotionError("promoted response cell counts do not match the frozen quota")
    histograms: dict[str, Counter[int]] = defaultdict(Counter)
    for row in promoted_targets:
        success = successes[row.prompt_uuid]
        histograms[_prompt_cell(view, row)][success.assistant_tokens] += 1
    for row in replay_primaries:
        histograms[_prompt_cell(view, row)][replay_by_id[row.prompt_uuid].assistant_tokens] += 1
    promoted_ids = tuple(row.prompt_uuid for row in promoted)
    records: list[PromotedResponseRecord] = []
    for row in promoted:
        canonical = json.loads(row.canonical_prompt_json)
        if row.lane == "target-synth":
            attempt = successes[row.prompt_uuid].attempt
            canonical["messages"] = [
                *canonical["messages"],
                json.loads(attempt.response_json)["message"],
            ]
            request_sha256 = attempt.request_sha256
            response_sha256 = attempt.response_sha256
            validation_sha256 = attempt.attempt_id
        else:
            replay = replay_by_id[row.prompt_uuid]
            request_sha256 = None
            response_sha256 = None
            validation_sha256 = replay.trajectory_validation_sha256
        records.append(
            _promoted_record(
                row,
                identity,
                canonical,
                selection_sha256=identity.source_selection_sha256,
                paired_cd_sha256="0" * 64,
                request_sha256=request_sha256,
                response_sha256=response_sha256,
                validation_sha256=validation_sha256,
            )
        )
    corpus_record = {
        "arm": view.arm,
        "generation_identity_sha256": identity.sha256,
        "source_selection_sha256": identity.source_selection_sha256,
        "promoted_ids": promoted_ids,
        "cell_counts": dict(view.cell_counts),
        "successful_response_sha256": [
            success.attempt.response_sha256 for success in successes.values()
        ],
        "attempt_history": [asdict(attempt) for attempt in seen_attempts.values()],
        "failed_attempts": [asdict(attempt) for attempt in failures],
        "replay_records": [asdict(replay_by_id[row.prompt_uuid]) for row in replay_primaries],
        "promoted_record_sha256": [record.record_sha256 for record in records],
        "unused_reserve_ids": unused,
        "response_token_histograms": {
            cell: dict(sorted(histogram.items())) for cell, histogram in sorted(histograms.items())
        },
    }
    return ResponseCorpus(
        arm=view.arm,
        generation_identity=identity,
        generation_identity_sha256=identity.sha256,
        source_selection_sha256=identity.source_selection_sha256,
        promoted_ids=promoted_ids,
        cell_counts=MappingProxyType(dict(view.cell_counts)),
        successful_attempts=tuple(success.attempt for success in successes.values()),
        failed_attempts=tuple(failures),
        replay_records=tuple(replay_by_id[row.prompt_uuid] for row in replay_primaries),
        failed_prompt_ids=tuple(
            dict.fromkeys(
                [attempt.prompt_uuid for attempt in failures]
                + [row.prompt_uuid for row in target_primaries if row.prompt_uuid not in successes]
            )
        ),
        unused_reserve_ids=tuple(unused),
        attempt_history=MappingProxyType(
            {prompt_id: tuple(history) for prompt_id, history in sorted(histories.items())}
        ),
        response_token_histograms=MappingProxyType(
            {
                cell: MappingProxyType(dict(sorted(histogram.items())))
                for cell, histogram in sorted(histograms.items())
            }
        ),
        records=tuple(records),
        corpus_sha256=sha256(canonical_json(corpus_record)).hexdigest(),
    )


def _sha256_file(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_bytes_durable(path: Path, payload: bytes) -> None:
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


def _freeze_nested(value: Mapping[str, Mapping[str, int]]) -> Mapping[str, Mapping[str, int]]:
    return MappingProxyType(
        {
            name: MappingProxyType({key: int(count) for key, count in counts.items()})
            for name, counts in value.items()
        }
    )


class _PublishedSelectedRows(Sequence[SelectedPrompt]):
    """Lazy authenticated rows backed by Task 5's publication index."""

    def __init__(
        self,
        connection: sqlite3.Connection,
        root: Path,
        arm: str,
        status: str,
    ) -> None:
        self._connection = connection
        self._root = root
        self._arm = arm
        self._status = status

    def __len__(self) -> int:
        return int(
            self._connection.execute(
                "SELECT count(*) FROM rows WHERE arm=? AND status=?",
                (self._arm, self._status),
            ).fetchone()[0]
        )

    def __getitem__(self, index: int | slice) -> SelectedPrompt | Sequence[SelectedPrompt]:
        if isinstance(index, slice):
            return tuple(self[position] for position in range(*index.indices(len(self))))
        if index < 0:
            index += len(self)
        pointer = self._connection.execute(
            "SELECT shard_path,byte_offset,byte_length,row_sha256 FROM rows "
            "WHERE arm=? AND status=? ORDER BY selection_index LIMIT 1 OFFSET ?",
            (self._arm, self._status, index),
        ).fetchone()
        if pointer is None:
            raise IndexError(index)
        return self._read_pointer(pointer)

    def __iter__(self):
        pointers = self._connection.execute(
            "SELECT shard_path,byte_offset,byte_length,row_sha256 FROM rows "
            "WHERE arm=? AND status=? ORDER BY selection_index",
            (self._arm, self._status),
        )
        yield from self._iter_pointer_rows(pointers)

    def iter_target_by_cell_rank(self):
        cell = (
            "CASE WHEN domain='stem-science' THEN 'stem' "
            "WHEN domain='multilingual' THEN CASE language "
            "WHEN 'ja' THEN 'japanese' WHEN 'es' THEN 'spanish' "
            "WHEN 'fr' THEN 'french' WHEN 'it' THEN 'italian' ELSE domain END "
            "ELSE domain END"
        )
        pointers = self._connection.execute(
            "SELECT shard_path,byte_offset,byte_length,row_sha256 FROM rows "
            f"WHERE arm=? AND status=? AND lane='target-synth' ORDER BY {cell},candidate_rank,prompt_uuid",
            (self._arm, self._status),
        )
        yield from self._iter_pointer_rows(pointers)

    def find(self, prompt_uuid: str) -> SelectedPrompt | None:
        pointer = self._connection.execute(
            "SELECT shard_path,byte_offset,byte_length,row_sha256 FROM rows "
            "WHERE arm=? AND prompt_uuid=? LIMIT 1",
            (self._arm, prompt_uuid),
        ).fetchone()
        return None if pointer is None else self._read_pointer(pointer)

    def _read_pointer(self, pointer: Sequence[Any]) -> SelectedPrompt:
        path = self._root / str(pointer[0])
        with path.open("rb") as shard:
            shard.seek(int(pointer[1]))
            line = shard.read(int(pointer[2]))
        if sha256(line).hexdigest() != pointer[3]:
            raise ResponsePromotionError(f"selection row identity mismatch: {path.name}")
        record = json.loads(line)
        canonical_prompt = _canonical_text(record.pop("canonical_prompt"))
        return SelectedPrompt(canonical_prompt_json=canonical_prompt, **record)

    def _iter_pointer_rows(self, pointers: Any):
        handles: OrderedDict[Path, Any] = OrderedDict()
        try:
            for pointer in pointers:
                path = self._root / str(pointer[0])
                shard = handles.pop(path, None)
                if shard is None:
                    shard = path.open("rb")
                handles[path] = shard
                if len(handles) > 64:
                    _, oldest = handles.popitem(last=False)
                    oldest.close()
                shard.seek(int(pointer[1]))
                line = shard.read(int(pointer[2]))
                if sha256(line).hexdigest() != pointer[3]:
                    raise ResponsePromotionError(f"selection row identity mismatch: {path.name}")
                record = json.loads(line)
                canonical_prompt = _canonical_text(record.pop("canonical_prompt"))
                yield SelectedPrompt(canonical_prompt_json=canonical_prompt, **record)
        finally:
            for handle in handles.values():
                handle.close()


class _SQLitePayloadSequence(Sequence[Any]):
    """Read-only lazy sequence over a payload column in a published response index."""

    def __init__(self, path: Path, query: str, parser: Any) -> None:
        self._path = path
        self._query = query
        self._parser = parser

    @property
    def resident_record_count(self) -> int:
        """Report that no response rows are retained in Python memory."""
        return 0

    def __len__(self) -> int:
        with sqlite3.connect(f"file:{self._path}?mode=ro", uri=True) as connection:
            return int(connection.execute(f"SELECT count(*) FROM ({self._query})").fetchone()[0])

    def __getitem__(self, index: int | slice) -> Any:
        if isinstance(index, slice):
            return tuple(self[position] for position in range(*index.indices(len(self))))
        if index < 0:
            index += len(self)
        with sqlite3.connect(f"file:{self._path}?mode=ro", uri=True) as connection:
            row = connection.execute(f"{self._query} LIMIT 1 OFFSET ?", (index,)).fetchone()
        if row is None:
            raise IndexError(index)
        return self._parser(row[0])

    def __iter__(self):
        with sqlite3.connect(f"file:{self._path}?mode=ro", uri=True) as connection:
            for row in connection.execute(self._query):
                yield self._parser(row[0])


class _SQLiteAttemptHistory(Mapping[str, tuple[GenerationAttempt, ...]]):
    def __init__(self, path: Path) -> None:
        self._path = path

    def __len__(self) -> int:
        with sqlite3.connect(f"file:{self._path}?mode=ro", uri=True) as connection:
            return int(
                connection.execute("SELECT count(DISTINCT prompt_uuid) FROM attempts").fetchone()[0]
            )

    def __iter__(self):
        with sqlite3.connect(f"file:{self._path}?mode=ro", uri=True) as connection:
            for (prompt_uuid,) in connection.execute(
                "SELECT DISTINCT prompt_uuid FROM attempts ORDER BY prompt_uuid"
            ):
                yield str(prompt_uuid)

    def __getitem__(self, prompt_uuid: str) -> tuple[GenerationAttempt, ...]:
        with sqlite3.connect(f"file:{self._path}?mode=ro", uri=True) as connection:
            rows = connection.execute(
                "SELECT payload FROM attempts WHERE prompt_uuid=? ORDER BY ordinal",
                (prompt_uuid,),
            ).fetchall()
        if not rows:
            raise KeyError(prompt_uuid)
        return tuple(GenerationAttempt(**json.loads(row[0])) for row in rows)


def load_prompt_view(
    manifest_path: Path,
    *,
    expected_manifest_sha256: str,
    arm: str,
) -> PromptView:
    """Authenticate a Task 5 selection artifact and load one frozen arm."""
    if _sha256_file(manifest_path) != expected_manifest_sha256:
        raise ResponsePromotionError("selection manifest identity mismatch")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict) or manifest.get("schema_version") != 2:
        raise ResponsePromotionError("unsupported selection manifest")
    manifest_root = manifest_path.parent.resolve(strict=True)
    declared_root = manifest.get("root_sha256")
    root_record = {key: value for key, value in manifest.items() if key != "root_sha256"}
    if declared_root != sha256(canonical_json(root_record)).hexdigest():
        raise ResponsePromotionError("selection root identity mismatch")
    arm_record = manifest.get("arms", {}).get(arm)
    if not isinstance(arm_record, dict):
        raise ResponsePromotionError(f"selection manifest has no {arm} arm")
    for descriptor in manifest.get("shards", []):
        if not isinstance(descriptor, dict) or not isinstance(descriptor.get("path"), str):
            raise ResponsePromotionError("invalid selection shard descriptor")
        shard = (manifest_path.parent / descriptor["path"]).resolve(strict=True)
        if not shard.is_relative_to(manifest_root) or not shard.is_file():
            raise ResponsePromotionError("selection shard escapes its publication root")
        if _sha256_file(shard) != descriptor.get(
            "sha256"
        ) or shard.stat().st_size != descriptor.get("byte_count"):
            raise ResponsePromotionError(f"selection shard identity mismatch: {shard.name}")
    index_descriptor = manifest.get("index")
    if not isinstance(index_descriptor, dict) or not isinstance(index_descriptor.get("path"), str):
        raise ResponsePromotionError("selection index descriptor is missing")
    index_path = (manifest_path.parent / index_descriptor["path"]).resolve(strict=True)
    if not index_path.is_relative_to(manifest_root) or not index_path.is_file():
        raise ResponsePromotionError("selection index escapes its publication root")
    if _sha256_file(index_path) != index_descriptor.get("sha256"):
        raise ResponsePromotionError("selection index identity mismatch")
    connection = sqlite3.connect(f"file:{index_path}?mode=ro", uri=True)
    primary = _PublishedSelectedRows(connection, manifest_path.parent, arm, "primary")
    reserve = _PublishedSelectedRows(connection, manifest_path.parent, arm, "reserve")
    if len(primary) != arm_record.get("primary_count") or len(reserve) != arm_record.get(
        "reserve_count"
    ):
        raise ResponsePromotionError("selection arm count mismatch")
    view = PromptView(
        arm=arm,
        primary_rows=primary,
        reserve_rows=reserve,
        cell_counts=MappingProxyType(
            {name: int(count) for name, count in arm_record["cell_counts"].items()}
        ),
        lane_counts=MappingProxyType(
            {name: int(count) for name, count in arm_record["lane_counts"].items()}
        ),
        bucket_floors=_freeze_nested(arm_record["bucket_floors"]),
        non_agentic_bucket_floors=_freeze_nested(arm_record["non_agentic_bucket_floors"]),
        lane_bucket_floors=_freeze_nested(arm_record["lane_bucket_floors"]),
        count_proof_sha256=str(arm_record["count_proof_sha256"]),
        tokenizer_sha256=(
            str(manifest["identity"]["tokenizer_sha256"])
            if "tokenizer_sha256" in manifest.get("identity", {})
            else None
        ),
        chat_template_sha256=(
            str(manifest["identity"]["chat_template_sha256"])
            if "chat_template_sha256" in manifest.get("identity", {})
            else None
        ),
    )
    object.__setattr__(view, "selection_sha256", str(manifest["selection_sha256"]))
    object.__setattr__(view, "paired_cd_sha256", str(manifest["paired_cd_sha256"]))
    object.__setattr__(view, "selection_manifest_sha256", expected_manifest_sha256)
    return view


def _load_identity(path: Path) -> GenerationIdentity:
    return GenerationIdentity(**json.loads(path.read_text(encoding="utf-8")))


def _closed_wire_assistant_message(value: Any) -> dict[str, str]:
    if not isinstance(value, dict):
        raise ValueError("wire assistant message is not an object")
    default_fields = {"reasoning_content", "reasoning", "tool_calls", "function_call"}
    if set(value) - {"role", "content", *default_fields}:
        raise ValueError("wire assistant message has unknown fields")
    if value.get("role") != "assistant" or not isinstance(value.get("content"), str):
        raise ValueError("wire assistant role/content is invalid")
    for field in {"reasoning_content", "reasoning"}.intersection(value):
        if value[field] is not None and value[field] != "":
            raise ValueError(f"wire assistant {field} is invalid")
    if "tool_calls" in value and value["tool_calls"] is not None and value["tool_calls"] != []:
        raise ValueError("wire assistant tool_calls is invalid")
    if "function_call" in value and value["function_call"] is not None:
        raise ValueError("wire assistant function_call is invalid")
    return {"role": "assistant", "content": value["content"]}


def _attempt_from_api(
    row: SelectedPrompt,
    identity: GenerationIdentity,
    *,
    base_url: str,
    model: str,
    attempt_namespace: str,
) -> GenerationAttempt:
    messages = _expected_messages(row, identity)
    request_record = {
        "messages": messages,
        "generation_identity_sha256": identity.sha256,
    }
    api_payload = {
        "model": model,
        "messages": messages,
        "temperature": identity.temperature,
        "max_tokens": identity.max_tokens,
    }
    request = urllib.request.Request(
        f"{base_url.rstrip('/')}/chat/completions",
        data=canonical_json(api_payload),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=3600) as response:
            api_response = json.loads(response.read())
        choice = api_response["choices"][0]
        response_record = {
            "message": _closed_wire_assistant_message(choice["message"]),
            "finish_reason": choice["finish_reason"],
            "completion_tokens": api_response["usage"]["completion_tokens"],
        }
    except (OSError, KeyError, IndexError, TypeError, ValueError, urllib.error.URLError):
        response_record = {
            "message": {"role": "assistant", "content": ""},
            "finish_reason": "error",
            "completion_tokens": 0,
        }
    request_json = _canonical_text(request_record)
    response_json = _canonical_text(response_record)
    attempt_id = sha256(
        canonical_json(
            {
                "namespace": attempt_namespace,
                "prompt_uuid": row.prompt_uuid,
                "generation_identity_sha256": identity.sha256,
            }
        )
    ).hexdigest()
    return GenerationAttempt(
        attempt_id=attempt_id,
        prompt_uuid=row.prompt_uuid,
        generation_identity_sha256=identity.sha256,
        request_json=request_json,
        request_sha256=sha256(request_json.encode()).hexdigest(),
        response_json=response_json,
        response_sha256=sha256(response_json.encode()).hexdigest(),
    )


def generate_attempt_artifact(
    view: PromptView,
    identity: GenerationIdentity,
    *,
    base_url: str,
    model: str,
    output_path: Path,
    num_shards: int,
    shard_id_begin: int,
    shard_id_step: int,
    attempt_namespace: str,
) -> None:
    """Generate immutable attempts in prompt-cell and frozen-rank order."""
    if output_path.exists():
        with output_path.open(encoding="utf-8") as existing:
            for line in existing:
                value = json.loads(line)
                if not isinstance(value, dict) or set(value) != set(
                    GenerationAttempt.__annotations__
                ):
                    raise ResponsePromotionError("existing attempt artifact is malformed")
                attempt = GenerationAttempt(**value)
                if attempt.generation_identity_sha256 != identity.sha256:
                    raise ResponsePromotionError("existing attempt artifact identity mismatch")
        return
    if isinstance(view.primary_rows, _PublishedSelectedRows) and isinstance(
        view.reserve_rows, _PublishedSelectedRows
    ):
        rows = chain(
            view.primary_rows.iter_target_by_cell_rank(),
            view.reserve_rows.iter_target_by_cell_rank(),
        )
    else:
        rows = iter(
            sorted(
                (
                    row
                    for row in chain(view.primary_rows, view.reserve_rows)
                    if row.lane == "target-synth"
                ),
                key=lambda row: (_prompt_cell(view, row), row.candidate_rank, row.prompt_uuid),
            )
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    partial = output_path.with_name(f".{output_path.name}.partial-{os.getpid()}")
    with partial.open("xb") as output:
        for index, row in enumerate(rows):
            if (index % num_shards) % shard_id_step != shard_id_begin:
                continue
            attempt = _attempt_from_api(
                row,
                identity,
                base_url=base_url,
                model=model,
                attempt_namespace=attempt_namespace,
            )
            output.write(canonical_json(asdict(attempt)) + b"\n")
        output.flush()
        os.fsync(output.fileno())
    os.replace(partial, output_path)
    _fsync_directory(output_path.parent)


def _iter_attempts(root: Path):
    found = False
    for path in sorted(root.glob("replicas/*/attempts-*.jsonl")):
        with path.open(encoding="utf-8") as rows:
            for line in rows:
                found = True
                value = json.loads(line)
                if not isinstance(value, dict) or set(value) != set(
                    GenerationAttempt.__annotations__
                ):
                    raise ResponsePromotionError(f"malformed generation attempt: {path}")
                yield GenerationAttempt(**value)
    if not found:
        raise ResponsePromotionError("generation produced no immutable attempts")


def _replay_assistant_tokens(tokenizer: Any, row: SelectedPrompt) -> int:
    prompt = json.loads(row.canonical_prompt_json)
    encoded = tokenizer.apply_chat_template(
        prompt["messages"],
        tools=prompt.get("tools") or None,
        tokenize=True,
        add_generation_prompt=False,
        return_dict=True,
        return_assistant_tokens_mask=True,
    )
    mask = encoded.get("assistant_masks") or encoded.get("assistant_tokens_mask")
    if not isinstance(mask, list) or not all(isinstance(value, int) for value in mask):
        raise ResponsePromotionError("tokenizer did not return an exact assistant-token mask")
    count = sum(mask)
    if count < 1:
        raise ResponsePromotionError("validated replay has no assistant tokens")
    return count


def promote_response_artifacts(
    view: PromptView,
    attempts_root: Path,
    identity: GenerationIdentity,
    output_root: Path,
    *,
    tokenizer: Any | None = None,
) -> ResponseCorpus:
    """Promote immutable attempt files and materialize the authenticated corpus."""
    output_root.mkdir(parents=True, exist_ok=False)
    index_path = output_root / "response-index.sqlite3"
    index = sqlite3.connect(index_path)
    index.executescript(
        """
        PRAGMA journal_mode=OFF;
        PRAGMA synchronous=FULL;
        CREATE TABLE attempts(
          ordinal INTEGER PRIMARY KEY, attempt_id TEXT UNIQUE NOT NULL,
          prompt_uuid TEXT NOT NULL, payload TEXT NOT NULL, successful INTEGER NOT NULL
        );
        CREATE INDEX attempts_prompt ON attempts(prompt_uuid,ordinal);
        CREATE TABLE successes(
          prompt_uuid TEXT PRIMARY KEY, attempt_id TEXT NOT NULL,
          response_sha256 TEXT NOT NULL, assistant_tokens INTEGER NOT NULL,
          payload TEXT NOT NULL
        );
        CREATE TABLE promoted(
          ordinal INTEGER PRIMARY KEY, prompt_uuid TEXT UNIQUE NOT NULL,
          cell TEXT NOT NULL, assistant_tokens INTEGER NOT NULL, payload TEXT NOT NULL
        );
        CREATE TABLE unused_reserve(ordinal INTEGER PRIMARY KEY, prompt_uuid TEXT NOT NULL);
        CREATE TABLE failed_prompts(ordinal INTEGER PRIMARY KEY, prompt_uuid TEXT UNIQUE NOT NULL);
        """
    )
    if not isinstance(view.primary_rows, _PublishedSelectedRows):
        raise ResponsePromotionError("production promotion requires a disk-backed prompt view")
    selection_sha256 = str(getattr(view, "selection_sha256", ""))
    paired_cd_sha256 = str(getattr(view, "paired_cd_sha256", ""))
    _require_digest("selection_sha256", selection_sha256)
    _require_digest("paired_cd_sha256", paired_cd_sha256)
    if getattr(view, "selection_manifest_sha256", None) != identity.source_selection_sha256:
        raise ResponsePromotionError("generation identity uses the wrong source selection")

    for ordinal, attempt in enumerate(_iter_attempts(attempts_root)):
        existing = index.execute(
            "SELECT payload FROM attempts WHERE attempt_id=?", (attempt.attempt_id,)
        ).fetchone()
        payload = _canonical_text(asdict(attempt))
        if existing is not None:
            if existing[0] != payload:
                raise ResponsePromotionError("attempt identity is bound to conflicting content")
            continue
        row = view.primary_rows.find(attempt.prompt_uuid)
        if row is None or row.lane != "target-synth":
            raise ResponsePromotionError(
                f"attempt references an ineligible prompt: {attempt.prompt_uuid}"
            )
        success = _validate_attempt(attempt, row, identity)
        index.execute(
            "INSERT INTO attempts VALUES(?,?,?,?,?)",
            (ordinal, attempt.attempt_id, attempt.prompt_uuid, payload, int(success is not None)),
        )
        if success is None:
            continue
        prior = index.execute(
            "SELECT response_sha256 FROM successes WHERE prompt_uuid=?", (attempt.prompt_uuid,)
        ).fetchone()
        if prior is not None and prior[0] != attempt.response_sha256:
            raise ResponsePromotionError(f"conflicting successful responses: {attempt.prompt_uuid}")
        index.execute(
            "INSERT OR IGNORE INTO successes VALUES(?,?,?,?,?)",
            (
                attempt.prompt_uuid,
                attempt.attempt_id,
                attempt.response_sha256,
                success.assistant_tokens,
                payload,
            ),
        )
    index.commit()

    histograms: dict[str, Counter[int]] = defaultdict(Counter)
    counts: Counter[str] = Counter()
    needed: Counter[str] = Counter()
    promoted_ordinal = 0
    unused_ordinal = 0
    failed_prompt_ordinal = 0

    def add_target(row: SelectedPrompt, success_row: Sequence[Any]) -> None:
        nonlocal promoted_ordinal
        attempt = GenerationAttempt(**json.loads(success_row[3]))
        canonical = json.loads(row.canonical_prompt_json)
        canonical["messages"] = [
            *canonical["messages"],
            json.loads(attempt.response_json)["message"],
        ]
        cell = _prompt_cell(view, row)
        record = _promoted_record(
            row,
            identity,
            canonical,
            selection_sha256=selection_sha256,
            paired_cd_sha256=paired_cd_sha256,
            request_sha256=attempt.request_sha256,
            response_sha256=attempt.response_sha256,
            validation_sha256=attempt.attempt_id,
        )
        tokens = int(success_row[2])
        index.execute(
            "INSERT INTO promoted VALUES(?,?,?,?,?)",
            (promoted_ordinal, row.prompt_uuid, cell, tokens, _canonical_text(asdict(record))),
        )
        promoted_ordinal += 1
        counts[cell] += 1
        histograms[cell][tokens] += 1

    def add_replay(row: SelectedPrompt) -> None:
        nonlocal promoted_ordinal
        if tokenizer is None:
            raise ResponsePromotionError("D replay promotion requires the pinned tokenizer")
        replay = validated_replay_record(
            row, assistant_tokens=_replay_assistant_tokens(tokenizer, row)
        )
        cell = _prompt_cell(view, row)
        record = _promoted_record(
            row,
            identity,
            json.loads(row.canonical_prompt_json),
            selection_sha256=selection_sha256,
            paired_cd_sha256=paired_cd_sha256,
            request_sha256=None,
            response_sha256=None,
            validation_sha256=replay.trajectory_validation_sha256,
        )
        index.execute(
            "INSERT INTO promoted VALUES(?,?,?,?,?)",
            (
                promoted_ordinal,
                row.prompt_uuid,
                cell,
                replay.assistant_tokens,
                _canonical_text(asdict(record)),
            ),
        )
        promoted_ordinal += 1
        counts[cell] += 1
        histograms[cell][replay.assistant_tokens] += 1

    for row in view.primary_rows:
        if row.lane != "target-synth":
            add_replay(row)
            continue
        success_row = index.execute(
            "SELECT attempt_id,response_sha256,assistant_tokens,payload FROM successes "
            "WHERE prompt_uuid=?",
            (row.prompt_uuid,),
        ).fetchone()
        if success_row is None:
            needed[_prompt_cell(view, row)] += 1
            index.execute(
                "INSERT INTO failed_prompts VALUES(?,?)",
                (failed_prompt_ordinal, row.prompt_uuid),
            )
            failed_prompt_ordinal += 1
        else:
            add_target(row, success_row)
    for row in view.reserve_rows:
        cell = _prompt_cell(view, row)
        success_row = index.execute(
            "SELECT attempt_id,response_sha256,assistant_tokens,payload FROM successes "
            "WHERE prompt_uuid=?",
            (row.prompt_uuid,),
        ).fetchone()
        if row.lane == "target-synth" and needed[cell] and success_row is not None:
            add_target(row, success_row)
            needed[cell] -= 1
        else:
            index.execute(
                "INSERT INTO unused_reserve VALUES(?,?)", (unused_ordinal, row.prompt_uuid)
            )
            unused_ordinal += 1
    shortfalls = {cell: count for cell, count in needed.items() if count}
    if shortfalls:
        cell, count = next(iter(shortfalls.items()))
        raise ResponsePromotionError(f"{cell} cell quota has {count} unresolved responses")
    if dict(counts) != dict(view.cell_counts):
        raise ResponsePromotionError("promoted response cell counts do not match the frozen quota")
    index.commit()

    corpus_path = output_root / "responses.jsonl"
    with corpus_path.open("xb") as output:
        record_sha256 = sha256()
        for (payload,) in index.execute("SELECT payload FROM promoted ORDER BY ordinal"):
            line = payload.encode() + b"\n"
            output.write(line)
            record_sha256.update(line)
        output.flush()
        os.fsync(output.fileno())
    history_sha256 = sha256()
    for (payload,) in index.execute("SELECT payload FROM attempts ORDER BY ordinal"):
        history_sha256.update(payload.encode() + b"\n")
    corpus_record = {
        "arm": view.arm,
        "generation_identity_sha256": identity.sha256,
        "selection_sha256": selection_sha256,
        "paired_cd_sha256": paired_cd_sha256,
        "cell_counts": dict(view.cell_counts),
        "promoted_record_stream_sha256": record_sha256.hexdigest(),
        "attempt_history_stream_sha256": history_sha256.hexdigest(),
        "unused_reserve_count": unused_ordinal,
        "response_token_histograms": {
            cell: dict(sorted(histogram.items())) for cell, histogram in sorted(histograms.items())
        },
    }
    corpus_sha256 = sha256(canonical_json(corpus_record)).hexdigest()
    index.execute("VACUUM")
    index.close()
    with index_path.open("rb") as index_file:
        os.fsync(index_file.fileno())
    receipt = {
        "schema_version": 1,
        "corpus_sha256": corpus_sha256,
        "generation_identity_sha256": identity.sha256,
        "source_selection_sha256": identity.source_selection_sha256,
        "selection_sha256": selection_sha256,
        "paired_cd_sha256": paired_cd_sha256,
        "promoted_count": promoted_ordinal,
        "cell_counts": dict(view.cell_counts),
        "unused_reserve_count": unused_ordinal,
        "response_token_histograms": {
            cell: dict(histogram) for cell, histogram in histograms.items()
        },
        "promoted_record_stream_sha256": record_sha256.hexdigest(),
        "attempt_history_stream_sha256": history_sha256.hexdigest(),
        "files": [
            {
                "path": corpus_path.name,
                "bytes": corpus_path.stat().st_size,
                "sha256": _sha256_file(corpus_path),
            },
            {
                "path": index_path.name,
                "bytes": index_path.stat().st_size,
                "sha256": _sha256_file(index_path),
            },
        ],
    }
    utilization_files: list[dict[str, Any]] = []
    for source in sorted(attempts_root.glob("replicas/*/gpu-utilization-replica-*.json")):
        value = json.loads(source.read_text(encoding="utf-8"))
        if (
            not isinstance(value, dict)
            or not isinstance(value.get("replica"), int)
            or not isinstance(value.get("gpu_utilization"), list)
            or not value["gpu_utilization"]
        ):
            raise ResponsePromotionError(f"malformed GPU utilization metadata: {source}")
        destination = output_root / source.name
        _write_bytes_durable(destination, source.read_bytes())
        utilization_files.append(
            {
                "path": destination.name,
                "bytes": destination.stat().st_size,
                "sha256": _sha256_file(destination),
            }
        )
    if len(utilization_files) != 4:
        raise ResponsePromotionError(
            "promotion requires GPU utilization metadata from four replicas"
        )
    receipt["files"].extend(utilization_files)
    receipt["gpu_utilization_files"] = utilization_files
    _write_bytes_durable(output_root / "PROMOTION.json", canonical_json(receipt) + b"\n")
    _fsync_directory(output_root)
    return ResponseCorpus(
        arm=view.arm,
        generation_identity=identity,
        generation_identity_sha256=identity.sha256,
        source_selection_sha256=identity.source_selection_sha256,
        promoted_ids=_SQLitePayloadSequence(
            index_path, "SELECT prompt_uuid FROM promoted ORDER BY ordinal", str
        ),
        cell_counts=MappingProxyType(dict(view.cell_counts)),
        successful_attempts=_SQLitePayloadSequence(
            index_path,
            "SELECT payload FROM successes ORDER BY prompt_uuid",
            lambda value: GenerationAttempt(**json.loads(value)),
        ),
        failed_attempts=_SQLitePayloadSequence(
            index_path,
            "SELECT payload FROM attempts WHERE successful=0 ORDER BY ordinal",
            lambda value: GenerationAttempt(**json.loads(value)),
        ),
        replay_records=(),
        failed_prompt_ids=_SQLitePayloadSequence(
            index_path, "SELECT prompt_uuid FROM failed_prompts ORDER BY ordinal", str
        ),
        unused_reserve_ids=_SQLitePayloadSequence(
            index_path, "SELECT prompt_uuid FROM unused_reserve ORDER BY ordinal", str
        ),
        attempt_history=_SQLiteAttemptHistory(index_path),
        response_token_histograms=MappingProxyType(
            {cell: MappingProxyType(dict(values)) for cell, values in histograms.items()}
        ),
        records=_SQLitePayloadSequence(
            index_path,
            "SELECT payload FROM promoted ORDER BY ordinal",
            lambda value: PromotedResponseRecord(**json.loads(value)),
        ),
        corpus_sha256=corpus_sha256,
    )


def _parse_cli(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("generate-attempts", "promote-attempts"):
        command = commands.add_parser(name)
        command.add_argument("--selection-manifest", type=Path, required=True)
        command.add_argument("--selection-manifest-sha256", required=True)
        command.add_argument("--arm", required=True)
        command.add_argument("--identity", type=Path, required=True)
    generate = commands.choices["generate-attempts"]
    generate.add_argument("--base-url", required=True)
    generate.add_argument("--model", required=True)
    generate.add_argument("--output", type=Path, required=True)
    generate.add_argument("--num-shards", type=int, required=True)
    generate.add_argument("--shard-id-begin", type=int, required=True)
    generate.add_argument("--shard-id-step", type=int, required=True)
    generate.add_argument("--attempt-namespace", required=True)
    promote = commands.choices["promote-attempts"]
    promote.add_argument("--attempts-root", type=Path, required=True)
    promote.add_argument("--output", type=Path, required=True)
    promote.add_argument("--tokenizer", type=Path)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Generate or promote production response artifacts."""
    arguments = _parse_cli(argv)
    view = load_prompt_view(
        arguments.selection_manifest,
        expected_manifest_sha256=arguments.selection_manifest_sha256,
        arm=arguments.arm,
    )
    identity = _load_identity(arguments.identity)
    if arguments.command == "generate-attempts":
        generate_attempt_artifact(
            view,
            identity,
            base_url=arguments.base_url,
            model=arguments.model,
            output_path=arguments.output,
            num_shards=arguments.num_shards,
            shard_id_begin=arguments.shard_id_begin,
            shard_id_step=arguments.shard_id_step,
            attempt_namespace=arguments.attempt_namespace,
        )
        return 0
    tokenizer = None
    if arguments.tokenizer is not None:
        from transformers import AutoTokenizer  # optional production runtime dependency

        tokenizer = AutoTokenizer.from_pretrained(arguments.tokenizer, local_files_only=True)
    promote_response_artifacts(
        view,
        arguments.attempts_root,
        identity,
        arguments.output,
        tokenizer=tokenizer,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
