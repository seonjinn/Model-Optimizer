# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Canonical prompt UUID and contamination exclusion enforcement."""

from __future__ import annotations

from dataclasses import dataclass, field

from specdec_corpus_contracts import CanonicalPrompt, canonical_json, sha256_bytes

__all__ = [
    "PROMPT_ROLES",
    "ContaminationError",
    "DuplicatePromptError",
    "ExclusionIndex",
    "UUIDCollisionError",
    "canonicalize_prompt",
    "normalize_storage_fields",
    "prompt_messages",
    "prompt_uuid",
    "validate_and_admit",
]

STORAGE_ONLY_FIELDS = frozenset(
    {
        "row_index",
        "shard_index",
        "download_path",
        "cache_path",
        "ingested_at",
        "source_completion",
        "generated_assistant_message",
    }
)

PROMPT_ROLES = frozenset({"system", "developer", "user"})


class ContaminationError(ValueError):
    pass


class DuplicatePromptError(ValueError):
    pass


class UUIDCollisionError(ValueError):
    pass


def normalize_storage_fields(value: object) -> object:
    if isinstance(value, dict):
        return {
            key: normalize_storage_fields(item)
            for key, item in value.items()
            if key not in STORAGE_ONLY_FIELDS
        }
    if isinstance(value, list):
        return [normalize_storage_fields(item) for item in value]
    return value


def prompt_messages(messages: object) -> list[dict[str, object]]:
    if not isinstance(messages, list) or any(not isinstance(item, dict) for item in messages):
        raise ValueError("messages must be a list of mappings")
    retained = [dict(item) for item in messages if item.get("role") in PROMPT_ROLES]
    if not retained:
        raise ValueError("conversation has no prompt-bearing messages")
    return retained


def _prompt_tools(tools: object) -> list[dict[str, object]]:
    if tools is None:
        return []
    if not isinstance(tools, list) or any(not isinstance(item, dict) for item in tools):
        raise ValueError("tools must be a list of mappings")
    return [dict(item) for item in tools]


def canonicalize_prompt(messages: object, tools: object) -> bytes:
    """Preserve semantic conversation structure while removing storage metadata."""
    return canonical_json(
        {
            "messages": normalize_storage_fields(prompt_messages(messages)),
            "tools": normalize_storage_fields(_prompt_tools(tools)),
        }
    )


def prompt_uuid(messages: object, tools: object) -> str:
    return sha256_bytes(canonicalize_prompt(messages, tools))


@dataclass
class ExclusionIndex:
    prior: set[str] = field(default_factory=set)
    held_out: set[str] = field(default_factory=set)
    admitted: dict[str, bytes] = field(default_factory=dict)

    def admit(self, candidate: CanonicalPrompt) -> None:
        if candidate.prompt_uuid in self.prior or candidate.prompt_uuid in self.held_out:
            raise ContaminationError(f"excluded prompt UUID: {candidate.prompt_uuid}")
        previous = self.admitted.get(candidate.prompt_uuid)
        if previous is not None:
            if previous != candidate.canonical_bytes:
                raise UUIDCollisionError(f"UUID collision: {candidate.prompt_uuid}")
            raise DuplicatePromptError(f"duplicate prompt UUID: {candidate.prompt_uuid}")
        self.admitted[candidate.prompt_uuid] = candidate.canonical_bytes


def validate_and_admit(candidate: CanonicalPrompt, exclusions: ExclusionIndex) -> None:
    exclusions.admit(candidate)
