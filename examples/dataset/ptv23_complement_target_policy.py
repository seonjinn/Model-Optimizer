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
from types import MappingProxyType
from typing import Any

__all__ = [
    "APPROVED_TARGET_POLICY_SHA256S",
    "ComplementError",
    "TargetTokenizerPolicy",
    "load_target_policy",
    "source_requirements_contract_is_compatible",
]

TARGET_POLICY_SCHEMA_VERSION = "ptv23-complement-target-policy-v1"
_MAX_TARGET_POLICY_BYTES = 1 * 1024 * 1024
APPROVED_TARGET_POLICY_SHA256S: frozenset[str] = frozenset(
    {
        "028830a9bde3f9353809894fc2650ee734c55a7b6860d6ec3ac32f55ff37b4df",
    }
)
APPROVED_SHARED_SOURCE_CONTRACT_IDENTITIES: frozenset[tuple[str, str, str, str]] = frozenset(
    {
        (
            "8cb600a2a839148814b0d26a6a80a7f061db5456291cd5d708be4520521d8e6b",
            "ptv2-ptv3-complement-700k-qwen3-30ba3b-thinking-v1",
            "ptv2-ptv3-complement-700k-qwen3-30ba3b-thinking-swe-heavy-v1",
            "940a3cfdd04a8cff94fff8a6980110dcb2e877ebd6f1897da75acff7d2023925",
        )
    }
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


APPROVED_TARGET_POLICIES = MappingProxyType(
    {
        "028830a9bde3f9353809894fc2650ee734c55a7b6860d6ec3ac32f55ff37b4df": TargetTokenizerPolicy(
            schema_version=TARGET_POLICY_SCHEMA_VERSION,
            scientific_identity="ptv2-ptv3-complement-700k-v1",
            target_repository="Qwen/Qwen3-4B",
            target_revision="1cfa9a7208912126459214e8b04321603b3df60c",
            tokenizer_repository="Qwen/Qwen3-4B",
            tokenizer_revision="1cfa9a7208912126459214e8b04321603b3df60c",
            tokenizer_trust_schema="qwen3-4b-tokenizer-trust-v1",
            training_sequence_length=4_096,
            quota_config_path="qwen3_4b_ptv23_complement_700k_v1.json",
            quota_config_sha256="9d70203bb00623d8e2c7bdfb15dfbb3b0d717779a238fc2e87292f88fe1e7ed9",
            source_requirements_path="qwen3_4b_ptv23_complement_sources_v1.json",
            source_requirements_sha256="e61ec87c2aba19c67c4a4549dba11f33abe8a3126f76232df0d72da43e8c81e1",
            file_sha256="028830a9bde3f9353809894fc2650ee734c55a7b6860d6ec3ac32f55ff37b4df",
        ),
    }
)


def _target_policy_identity(policy: TargetTokenizerPolicy) -> tuple[object, ...]:
    return (
        policy.schema_version,
        policy.scientific_identity,
        policy.target_repository,
        policy.target_revision,
        policy.tokenizer_repository,
        policy.tokenizer_revision,
        policy.tokenizer_trust_schema,
        policy.training_sequence_length,
        policy.quota_config_path,
        policy.quota_config_sha256,
        policy.source_requirements_path,
        policy.source_requirements_sha256,
        policy.file_sha256,
    )


def require_approved_target_policy(policy: TargetTokenizerPolicy) -> None:
    """Reject reconstructed policy objects that do not match one reviewed complete tuple."""
    approved = APPROVED_TARGET_POLICIES.get(policy.file_sha256)
    if approved is None or _target_policy_identity(approved) != _target_policy_identity(policy):
        raise ComplementError("target policy object is not approved")


def source_requirements_contract_is_compatible(
    policy: TargetTokenizerPolicy,
    *,
    observed_sha256: str,
    embedded_scientific_identity: object,
) -> bool:
    """Authenticate exact same-identity or explicitly reviewed shared source contracts."""
    approved = APPROVED_TARGET_POLICIES.get(policy.file_sha256)
    if approved is None or _target_policy_identity(approved) != _target_policy_identity(policy):
        return False
    if observed_sha256 != policy.source_requirements_sha256:
        return False
    if not isinstance(embedded_scientific_identity, str):
        return False
    if embedded_scientific_identity == policy.scientific_identity:
        return True
    return (
        observed_sha256,
        embedded_scientific_identity,
        policy.scientific_identity,
        policy.file_sha256,
    ) in APPROVED_SHARED_SOURCE_CONTRACT_IDENTITIES


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
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        raise ComplementError(f"target policy is unreadable: {path}") from error
    before = os.fstat(descriptor)
    if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1:
        os.close(descriptor)
        raise ComplementError(f"target policy is not a single-link regular file: {path}")
    if before.st_size > _MAX_TARGET_POLICY_BYTES:
        os.close(descriptor)
        raise ComplementError(f"target policy is too large: {path}")
    try:
        stream = os.fdopen(descriptor, "rb")
    except BaseException:
        os.close(descriptor)
        raise
    with stream:
        raw = stream.read(_MAX_TARGET_POLICY_BYTES + 1)
        after = os.fstat(stream.fileno())
    try:
        rebound = os.stat(path, follow_symlinks=False)
    except OSError as error:
        raise ComplementError(f"target policy changed while reading: {path}") from error
    before_identity = (
        before.st_dev,
        before.st_ino,
        before.st_size,
        before.st_mtime_ns,
        before.st_ctime_ns,
    )
    if (
        before_identity
        != (
            after.st_dev,
            after.st_ino,
            after.st_size,
            after.st_mtime_ns,
            after.st_ctime_ns,
        )
        or before_identity
        != (
            rebound.st_dev,
            rebound.st_ino,
            rebound.st_size,
            rebound.st_mtime_ns,
            rebound.st_ctime_ns,
        )
        or after.st_nlink != 1
        or rebound.st_nlink != 1
        or len(raw) != before.st_size
        or len(raw) > _MAX_TARGET_POLICY_BYTES
    ):
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
    source_requirements_path, source_requirements_sha256 = _bound_file(
        payload, "source_requirements"
    )
    policy = TargetTokenizerPolicy(
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
    require_approved_target_policy(policy)
    return policy
