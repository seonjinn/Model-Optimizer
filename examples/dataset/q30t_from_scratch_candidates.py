# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Construct response-safe Q30 prompt candidates from authenticated source rows."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal, cast

from q30t_generation_prompt_identity import (
    GenerationPromptIdentity,
    PromptProvenance,
    generation_prompt_identity,
)
from specdec_corpus_contracts import canonical_json, sha256_bytes
from specdec_identity import normalize_storage_fields, prompt_uuid

if TYPE_CHECKING:
    from collections.abc import Mapping

    from q30t_from_scratch_sources import SourceRegistryEntry

__all__ = [
    "CandidateAdapterError",
    "PromptCandidate",
    "adapt_historical_occurrence",
    "adapt_target_synthesis_row",
]

_ADAPTER_FIELDS = {
    "ptv2_messages_tools_v1": ("messages", "tools"),
    "ptv3_messages_tools_v1": ("messages", "tools"),
}


class CandidateAdapterError(ValueError):
    """An authenticated source row cannot be adapted without inference."""


@dataclass(frozen=True)
class PromptCandidate:
    """One immutable prompt candidate for a historical or synthesis arm."""

    arm: Literal["H-native", "H-synth", "M-synth"]
    lane: str
    language: str
    identity: GenerationPromptIdentity
    request_bytes: bytes
    source_assistant: dict[str, object] | None
    normalized_row_sha256: str


@dataclass(frozen=True)
class _AdaptedRow:
    messages: list[dict[str, object]]
    tools: list[dict[str, object]]
    source_row_index: int
    reasoning_mode: Literal["reasoning_on", "reasoning_off"]
    language: str
    normalized_row_sha256: str


def adapt_target_synthesis_row(
    row: Mapping[str, object], source: SourceRegistryEntry
) -> PromptCandidate:
    """Remove a source completion and construct one M-synth request."""
    adapted = _adapt_row(row, source)
    prefix, _ = _remove_source_assistant(adapted.messages)
    identity = _identity(prefix, adapted, source)
    return PromptCandidate(
        arm="M-synth",
        lane=source.lane,
        language=adapted.language,
        identity=identity,
        request_bytes=_request_bytes(identity),
        source_assistant=None,
        normalized_row_sha256=adapted.normalized_row_sha256,
    )


def adapt_historical_occurrence(
    row: Mapping[str, object],
    source: SourceRegistryEntry,
    *,
    expected_prompt_uuid: str,
) -> tuple[PromptCandidate, PromptCandidate]:
    """Reconcile one ordered audit occurrence and return H-native then H-synth."""
    adapted = _adapt_row(row, source)
    actual_prompt_uuid = prompt_uuid(adapted.messages, adapted.tools)
    if actual_prompt_uuid != expected_prompt_uuid:
        raise CandidateAdapterError(
            "historical occurrence prompt UUID does not match the authenticated audit"
        )
    prefix, source_assistant = _remove_source_assistant(adapted.messages)
    if source_assistant is None:
        raise CandidateAdapterError("historical occurrence has no final source assistant")
    identity = _identity(prefix, adapted, source)
    shared = {
        "lane": source.lane,
        "language": adapted.language,
        "identity": identity,
        "normalized_row_sha256": adapted.normalized_row_sha256,
    }
    native = PromptCandidate(
        arm="H-native",
        request_bytes=b"",
        source_assistant=source_assistant,
        **shared,
    )
    synth = PromptCandidate(
        arm="H-synth",
        request_bytes=_request_bytes(identity),
        source_assistant=None,
        **shared,
    )
    return native, synth


def _adapt_row(row: Mapping[str, object], source: SourceRegistryEntry) -> _AdaptedRow:
    fields = _ADAPTER_FIELDS.get(source.adapter)
    if fields is None:
        raise CandidateAdapterError(f"unsupported source adapter: {source.adapter}")
    messages_field, tools_field = fields
    if messages_field not in row:
        raise CandidateAdapterError(
            f"source adapter {source.adapter} requires {messages_field!r} messages"
        )
    messages = _mapping_list(_decode(row[messages_field], messages_field), messages_field)
    if not messages:
        raise CandidateAdapterError("source messages must not be empty")
    tools_value = row.get(tools_field, [])
    tools = _mapping_list(_decode(tools_value, tools_field), tools_field, allow_none=True)

    source_row_index = row.get("source_row_index")
    if type(source_row_index) is not int or source_row_index < 0:
        raise CandidateAdapterError("source_row_index must be a nonnegative integer")
    reasoning_mode = row.get("reasoning_mode", "reasoning_on")
    if reasoning_mode not in {"reasoning_on", "reasoning_off"}:
        raise CandidateAdapterError("reasoning_mode must be reasoning_on or reasoning_off")
    language = row.get("language", "")
    if not isinstance(language, str):
        raise CandidateAdapterError("language must be a string")

    normalized_row = {"messages": messages, "tools": tools}
    return _AdaptedRow(
        messages=messages,
        tools=tools,
        source_row_index=source_row_index,
        reasoning_mode=cast('Literal["reasoning_on", "reasoning_off"]', reasoning_mode),
        language=language,
        normalized_row_sha256=sha256_bytes(canonical_json(normalized_row)),
    )


def _decode(value: object, field: str) -> object:
    if not isinstance(value, str):
        return value
    try:
        return cast("object", json.loads(value))
    except json.JSONDecodeError as error:
        raise CandidateAdapterError(f"source {field} is not valid JSON") from error


def _mapping_list(
    value: object, field: str, *, allow_none: bool = False
) -> list[dict[str, object]]:
    if value is None and allow_none:
        return []
    if not isinstance(value, list) or any(not isinstance(item, dict) for item in value):
        raise CandidateAdapterError(f"source {field} must be a list of mappings")
    normalized = normalize_storage_fields(value)
    if not isinstance(normalized, list) or any(not isinstance(item, dict) for item in normalized):
        raise AssertionError(f"normalized {field} must be a list of mappings")
    return [cast("dict[str, object]", item) for item in normalized]


def _remove_source_assistant(
    messages: list[dict[str, object]],
) -> tuple[list[dict[str, object]], dict[str, object] | None]:
    if messages[-1].get("role") != "assistant":
        return messages, None
    return messages[:-1], dict(messages[-1])


def _identity(
    messages: list[dict[str, object]],
    adapted: _AdaptedRow,
    source: SourceRegistryEntry,
) -> GenerationPromptIdentity:
    provenance = PromptProvenance(
        source_repository=source.repository,
        source_revision=source.revision,
        source_file_sha256=source.file_sha256,
        source_split=source.split,
        source_row_index=adapted.source_row_index,
        reasoning_mode=adapted.reasoning_mode,
        response_lane=source.lane,
    )
    return generation_prompt_identity(messages, adapted.tools, provenance)


def _request_bytes(identity: GenerationPromptIdentity) -> bytes:
    payload = json.loads(identity.canonical_bytes)
    if not isinstance(payload, dict) or set(payload) != {"messages", "tools", "provenance"}:
        raise AssertionError("generation prompt identity payload is invalid")
    return canonical_json({"messages": payload["messages"], "tools": payload["tools"]})
