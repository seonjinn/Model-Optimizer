# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Provenance-bound identity for immutable Q30 generation prefixes."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Literal

from specdec_corpus_contracts import canonical_json, sha256_bytes
from specdec_identity import normalize_storage_fields
from trajectory_schema import canonicalize_trajectory

__all__ = [
    "GenerationPromptIdentity",
    "PromptProvenance",
    "generation_prompt_identity",
]

_IDENTITY_SOURCE_ID = "q30t-generation-prompt-identity"


@dataclass(frozen=True)
class PromptProvenance:
    """Immutable source lineage for one generation prefix."""

    source_repository: str
    source_revision: str
    source_file_sha256: str
    source_split: str
    source_row_index: int
    reasoning_mode: Literal["reasoning_on", "reasoning_off"]
    response_lane: str


@dataclass(frozen=True)
class GenerationPromptIdentity:
    """Canonical immutable-prefix bytes and their SHA-256 identifier."""

    prompt_uuid: str
    canonical_bytes: bytes
    provenance: PromptProvenance


def generation_prompt_identity(
    messages: object,
    tools: object,
    provenance: PromptProvenance,
) -> GenerationPromptIdentity:
    """Return the identity for a validated, response-free generation prefix."""
    validated_messages, validated_tools = _validated_immutable_prefix(messages, tools)
    canonical_bytes = canonical_json(
        {
            "messages": validated_messages,
            "tools": validated_tools,
            "provenance": asdict(provenance),
        }
    )
    return GenerationPromptIdentity(
        prompt_uuid=sha256_bytes(canonical_bytes),
        canonical_bytes=canonical_bytes,
        provenance=provenance,
    )


def _validated_immutable_prefix(messages: object, tools: object) -> tuple[object, object]:
    normalized = normalize_storage_fields({"messages": messages, "tools": tools})
    if not isinstance(normalized, dict):
        raise AssertionError("normalized identity input must be a mapping")
    canonical = canonicalize_trajectory(normalized, source_id=_IDENTITY_SOURCE_ID)
    return canonical["messages"], canonical["tools"]
