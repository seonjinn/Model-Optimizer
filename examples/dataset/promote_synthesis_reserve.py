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

import json
import re
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from hashlib import sha256
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Literal

from specdec_corpus_contracts import canonical_json

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from select_bprime_cd_prompts import PromptView, SelectedPrompt

__all__ = [
    "GenerationAttempt",
    "GenerationIdentity",
    "ResponseCorpus",
    "ResponsePromotionError",
    "ValidatedReplayRecord",
    "promote_responses",
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
class ResponseCorpus:
    """An exact promoted corpus with attempt receipts and token histograms."""

    arm: str
    generation_identity: GenerationIdentity
    generation_identity_sha256: str
    source_selection_sha256: str
    promoted_ids: tuple[str, ...]
    cell_counts: Mapping[str, int]
    successful_attempts: tuple[GenerationAttempt, ...]
    failed_attempts: tuple[GenerationAttempt, ...]
    replay_records: tuple[ValidatedReplayRecord, ...]
    failed_prompt_ids: tuple[str, ...]
    unused_reserve_ids: tuple[str, ...]
    attempt_history: Mapping[str, tuple[GenerationAttempt, ...]]
    response_token_histograms: Mapping[str, Mapping[int, int]]
    corpus_sha256: str


@dataclass(frozen=True)
class _ValidatedSuccess:
    attempt: GenerationAttempt
    assistant_tokens: int


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
        or message.get("role") != "assistant"
        or not isinstance(message.get("content"), str)
        or not message["content"].strip()
        or message.get("tool_calls")
        or message.get("tool_call_id")
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
        if (
            not _SHA256.fullmatch(replay.trajectory_validation_sha256)
            or sha256(replay.canonical_record_json.encode()).hexdigest() != replay.record_sha256
            or replay.canonical_record_json != _canonical_text(replay_value)
            or replay.canonical_record_json != row.canonical_prompt_json
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
    corpus_record = {
        "arm": view.arm,
        "generation_identity_sha256": identity.sha256,
        "source_selection_sha256": identity.source_selection_sha256,
        "promoted_ids": promoted_ids,
        "cell_counts": dict(view.cell_counts),
        "successful_response_sha256": [
            success.attempt.response_sha256 for success in successes.values()
        ],
        "replay_record_sha256": [
            replay_by_id[row.prompt_uuid].record_sha256 for row in replay_primaries
        ],
        "failed_attempt_ids": [attempt.attempt_id for attempt in failures],
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
        corpus_sha256=sha256(canonical_json(corpus_record)).hexdigest(),
    )
