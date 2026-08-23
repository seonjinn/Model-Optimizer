# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Typed, deterministic contracts shared by SpecDec corpus builders."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Sequence

__all__ = [
    "CanonicalPrompt",
    "SourceFile",
    "canonical_json",
    "sha256_bytes",
    "sha256_canonical_json",
]


def canonical_json(value: object) -> bytes:
    """Serialize a corpus identity payload deterministically."""
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()


def sha256_bytes(value: bytes) -> str:
    """Return the lowercase SHA-256 digest of bytes."""
    return hashlib.sha256(value).hexdigest()


def sha256_canonical_json(value: Sequence[str]) -> str:
    """Return the deterministic digest of an ordered identifier sequence."""
    return sha256_bytes(canonical_json(value))


@dataclass(frozen=True)
class SourceFile:
    path: str
    bytes: int
    sha256: str


@dataclass(frozen=True)
class CanonicalPrompt:
    prompt_uuid: str
    canonical_bytes: bytes
    source_id: str
    source_revision: str
    source_file_sha256: str
    source_row_index: int
    domain: str
    language: str
