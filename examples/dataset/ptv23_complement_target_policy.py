# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Immutable target-tokenizer policies for PTV2/PTV3 complement data."""

from __future__ import annotations

import json
import os
import stat
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any

__all__ = ["APPROVED_TARGET_POLICY_SHA256S", "ComplementError", "TargetTokenizerPolicy", "load_target_policy"]

TARGET_POLICY_SCHEMA_VERSION = "ptv23-complement-target-policy-v1"
APPROVED_TARGET_POLICY_SHA256S: frozenset[str] = frozenset(
    {"028830a9bde3f9353809894fc2650ee734c55a7b6860d6ec3ac32f55ff37b4df"}
)


class ComplementError(ValueError):
    """An authenticated complement input is invalid."""


@dataclass(frozen=True)
class TargetTokenizerPolicy:
    """One immutable target, tokenizer, and dataset-policy binding."""

    schema_version: str
    scientific_identity: str
    target_repository: str
    target_revision: str
    tokenizer_repository: str
    tokenizer_revision: str
    tokenizer_trust_schema: str
    training_sequence_length: int
    quota_config_path: str
    quota_config_sha256: str
    source_requirements_path: str
    source_requirements_sha256: str
    file_sha256: str


def _canonical_json(value: object) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _is_lower_hex(value: object, length: int) -> bool:
    return isinstance(value, str) and len(value) == length and all(
        character in "0123456789abcdef" for character in value
    )


def _stable_regular_bytes(path: Path) -> bytes:
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        raise ComplementError(f"target policy is unreadable: {path}") from error
    with os.fdopen(descriptor, "rb") as stream:
        before = os.fstat(stream.fileno())
        if not stat.S_ISREG(before.st_mode):
            raise ComplementError(f"target policy is not a regular file: {path}")
        raw = stream.read()
        after = os.fstat(stream.fileno())
    if (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns) != (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
    ) or len(raw) != before.st_size:
        raise ComplementError(f"target policy changed while reading: {path}")
    return raw


def _bound_file(payload: dict[str, Any], name: str) -> tuple[str, str]:
    value = payload[name]
    if (
        not isinstance(value, dict)
        or set(value) != {"path", "sha256"}
        or not isinstance(value["path"], str)
        or Path(value["path"]).name != value["path"]
        or not _is_lower_hex(value["sha256"], 64)
    ):
        raise ComplementError("target policy file binding is invalid")
    return value["path"], value["sha256"]


def load_target_policy(path: Path, expected_sha256: str) -> TargetTokenizerPolicy:
    """Load an approved canonical target policy through one no-follow descriptor."""
    if not _is_lower_hex(expected_sha256, 64):
        raise ComplementError("target policy caller SHA-256 is invalid")
    if expected_sha256 not in APPROVED_TARGET_POLICY_SHA256S:
        raise ComplementError("target policy is not approved")
    raw = _stable_regular_bytes(path)
    if sha256(raw).hexdigest() != expected_sha256:
        raise ComplementError("target policy caller SHA-256 mismatch")
    try:
        payload: Any = json.loads(raw)
    except json.JSONDecodeError as error:
        raise ComplementError("target policy JSON is invalid") from error
    expected_keys = {
        "quota_config",
        "schema_version",
        "scientific_identity",
        "source_requirements",
        "target_repository",
        "target_revision",
        "tokenizer_repository",
        "tokenizer_revision",
        "tokenizer_trust_schema",
        "training_sequence_length",
    }
    if (
        not isinstance(payload, dict)
        or raw != _canonical_json(payload) + b"\n"
        or set(payload) != expected_keys
        or payload["schema_version"] != TARGET_POLICY_SCHEMA_VERSION
        or any(
            not isinstance(payload[name], str) or not payload[name]
            for name in (
                "scientific_identity",
                "target_repository",
                "tokenizer_repository",
                "tokenizer_trust_schema",
            )
        )
        or any(
            not isinstance(payload[name], str)
            or (payload[name] and not _is_lower_hex(payload[name], 40))
            for name in ("target_revision", "tokenizer_revision")
        )
        or type(payload["training_sequence_length"]) is not int
        or payload["training_sequence_length"] < 1
    ):
        raise ComplementError("target policy identity is invalid")
    quota_config_path, quota_config_sha256 = _bound_file(payload, "quota_config")
    source_requirements_path, source_requirements_sha256 = _bound_file(payload, "source_requirements")
    return TargetTokenizerPolicy(
        schema_version=payload["schema_version"],
        scientific_identity=payload["scientific_identity"],
        target_repository=payload["target_repository"],
        target_revision=payload["target_revision"],
        tokenizer_repository=payload["tokenizer_repository"],
        tokenizer_revision=payload["tokenizer_revision"],
        tokenizer_trust_schema=payload["tokenizer_trust_schema"],
        training_sequence_length=payload["training_sequence_length"],
        quota_config_path=quota_config_path,
        quota_config_sha256=quota_config_sha256,
        source_requirements_path=source_requirements_path,
        source_requirements_sha256=source_requirements_sha256,
        file_sha256=expected_sha256,
    )
