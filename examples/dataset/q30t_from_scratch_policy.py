# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Fail-closed scientific policy boundary for the Q30 from-scratch study."""

from __future__ import annotations

import json
import os
import re
import stat
from dataclasses import dataclass
from fractions import Fraction
from hashlib import sha256
from typing import TYPE_CHECKING, Any, Literal, cast

from specdec_corpus_contracts import canonical_json

if TYPE_CHECKING:
    from collections.abc import Mapping
    from pathlib import Path

__all__ = [
    "APPROVED_FROM_SCRATCH_POLICY_SHA256S",
    "CANARY_ROWS",
    "LANE_ORDER",
    "TOKEN_SHARES",
    "FromScratchStudyPolicy",
    "LanePolicy",
    "LanguageAllocation",
    "TokenGate",
    "load_from_scratch_policy",
]

LANE_ORDER = (
    "agentless_swe",
    "interactive_swe",
    "general_tool",
    "math",
    "code",
    "stem",
    "instruction",
    "multilingual",
)
CANARY_ROWS = (1_536, 1_024, 1_024, 2_560, 1_536, 1_536, 512, 512)
TOKEN_SHARES = (
    Fraction(15, 100),
    Fraction(10, 100),
    Fraction(10, 100),
    Fraction(25, 100),
    Fraction(15, 100),
    Fraction(15, 100),
    Fraction(5, 100),
    Fraction(5, 100),
)
APPROVED_FROM_SCRATCH_POLICY_SHA256S: frozenset[str] = frozenset()

_POLICY_SCHEMA = "q30t-ptv23-from-scratch-policy-v1"
_SCIENTIFIC_IDENTITY = "q30t-ptv23-from-scratch-drafter-study-v1"
_TARGET_REPOSITORY = "Qwen/Qwen3-30B-A3B-Thinking-2507"
_TARGET_REVISION = "144afc2f379b542fdd4e85a1fcd5e1f79112d95d"
_SOURCE_REQUIREMENTS_PATH = "qwen3_30ba3b_thinking_ptv23_from_scratch_sources_v1.json"
_SEED = 922_609_729
_POLICY_KEYS = frozenset(
    {
        "schema_version",
        "scientific_identity",
        "target_repository",
        "target_revision",
        "seed",
        "lanes",
        "gates",
        "source_requirements",
    }
)
_LANE_KEYS = frozenset(
    {
        "name",
        "token_numerator",
        "token_denominator",
        "canary_rows",
        "languages",
        "language_allocations",
    }
)
_LANGUAGE_ALLOCATION_KEYS = frozenset({"language", "token_numerator", "token_denominator"})
_GATE_KEYS = frozenset({"name", "assistant_loss_tokens"})
_SOURCE_REQUIREMENTS_KEYS = frozenset({"path", "sha256"})
_EXPECTED_FRACTIONS = (
    (15, 100),
    (10, 100),
    (10, 100),
    (25, 100),
    (15, 100),
    (15, 100),
    (5, 100),
    (5, 100),
)
_EXPECTED_LANGUAGES = ((), (), (), (), (), (), (), ("de", "ja", "es", "fr", "it"))
_EXPECTED_LANGUAGE_ALLOCATION_FRACTIONS = (
    (),
    (),
    (),
    (),
    (),
    (),
    (),
    ((1, 5), (1, 5), (1, 5), (1, 5), (1, 5)),
)
_GATE_NAMES = ("pilot", "primary", "scale")
_GATE_TOKENS = (256_000_000, 1_000_000_000, 4_000_000_000)
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


@dataclass(frozen=True)
class LanePolicy:
    """Exact assistant-loss-token share and canary size for one M-synth lane."""

    name: str
    token_numerator: int
    token_denominator: int
    canary_rows: int
    languages: tuple[str, ...]
    language_allocations: tuple[LanguageAllocation, ...]

    @property
    def token_share(self) -> Fraction:
        """Return this lane's exact assistant-loss-token share."""
        return Fraction(self.token_numerator, self.token_denominator)


@dataclass(frozen=True)
class LanguageAllocation:
    """Exact token-capacity share for one language within a multilingual lane."""

    language: str
    token_numerator: int
    token_denominator: int

    @property
    def token_share(self) -> Fraction:
        """Return this language's exact share of the enclosing lane."""
        return Fraction(self.token_numerator, self.token_denominator)


@dataclass(frozen=True)
class TokenGate:
    """One cumulative assistant-loss-token gate."""

    name: Literal["pilot", "primary", "scale"]
    assistant_loss_tokens: int


@dataclass(frozen=True)
class FromScratchStudyPolicy:
    """Validated immutable policy consumed by every from-scratch corpus component."""

    schema_version: Literal["q30t-ptv23-from-scratch-policy-v1"]
    scientific_identity: str
    target_repository: str
    target_revision: str
    seed: int
    lanes: tuple[LanePolicy, ...]
    gates: tuple[TokenGate, ...]
    source_requirements_path: str
    source_requirements_sha256: str
    file_sha256: str


def load_from_scratch_policy(path: Path, *, expected_sha256: str) -> FromScratchStudyPolicy:
    """Load an externally approved, immutable Q30 from-scratch policy file."""
    if expected_sha256 not in APPROVED_FROM_SCRATCH_POLICY_SHA256S:
        raise ValueError("from-scratch policy digest is not approved")
    return _parse_from_scratch_policy(path, expected_sha256=expected_sha256)


def _parse_from_scratch_policy_for_test(path: Path) -> FromScratchStudyPolicy:
    """Parse checked-in fixtures without treating their digest as production approval."""
    raw = _read_stable_regular_file(path)
    return _parse_policy_bytes(path, raw)


def _parse_from_scratch_policy(path: Path, *, expected_sha256: str) -> FromScratchStudyPolicy:
    if not _SHA256.fullmatch(expected_sha256):
        raise ValueError("expected policy SHA-256 must be lowercase hexadecimal")
    raw = _read_stable_regular_file(path)
    actual_sha256 = sha256(raw).hexdigest()
    if actual_sha256 != expected_sha256:
        raise ValueError("from-scratch policy digest does not match expected SHA-256")
    return _parse_policy_bytes(path, raw)


def _parse_policy_bytes(path: Path, raw: bytes) -> FromScratchStudyPolicy:
    root = _load_canonical_mapping(raw, "policy")
    _require_exact_keys(root, _POLICY_KEYS, "policy")

    schema_version = cast(
        "Literal['q30t-ptv23-from-scratch-policy-v1']",
        _literal(root["schema_version"], _POLICY_SCHEMA, "policy.schema_version"),
    )
    scientific_identity = _literal(
        root["scientific_identity"], _SCIENTIFIC_IDENTITY, "policy.scientific_identity"
    )
    target_repository = _literal(
        root["target_repository"], _TARGET_REPOSITORY, "policy.target_repository"
    )
    target_revision = _literal(root["target_revision"], _TARGET_REVISION, "policy.target_revision")
    seed = _positive_int(root["seed"], "policy.seed")
    if seed != _SEED:
        raise ValueError(f"policy.seed must be {_SEED}")

    source_requirements = _mapping(root["source_requirements"], "policy.source_requirements")
    _require_exact_keys(source_requirements, _SOURCE_REQUIREMENTS_KEYS, "policy.source_requirements")
    source_requirements_path = _literal(
        source_requirements["path"], _SOURCE_REQUIREMENTS_PATH, "policy.source_requirements.path"
    )
    source_requirements_sha256 = _sha256(
        source_requirements["sha256"], "policy.source_requirements.sha256"
    )
    _verify_source_requirements_binding(path, source_requirements_path, source_requirements_sha256)

    lanes = _parse_lanes(root["lanes"])
    gates = _parse_gates(root["gates"])
    return FromScratchStudyPolicy(
        schema_version=schema_version,
        scientific_identity=scientific_identity,
        target_repository=target_repository,
        target_revision=target_revision,
        seed=seed,
        lanes=lanes,
        gates=gates,
        source_requirements_path=source_requirements_path,
        source_requirements_sha256=source_requirements_sha256,
        file_sha256=sha256(raw).hexdigest(),
    )


def _parse_lanes(value: object) -> tuple[LanePolicy, ...]:
    entries = _list(value, "policy.lanes")
    if len(entries) != len(LANE_ORDER):
        raise ValueError("policy.lanes must contain every exact lane once")
    lanes: list[LanePolicy] = []
    for index, entry in enumerate(entries):
        lane = _mapping(entry, f"policy.lanes[{index}]")
        _require_exact_keys(lane, _LANE_KEYS, f"policy.lanes[{index}]")
        name = _literal(lane["name"], LANE_ORDER[index], f"policy.lanes[{index}].name")
        numerator = _positive_int(lane["token_numerator"], f"policy.lanes[{index}].token_numerator")
        denominator = _positive_int(
            lane["token_denominator"], f"policy.lanes[{index}].token_denominator"
        )
        if (numerator, denominator) != _EXPECTED_FRACTIONS[index]:
            raise ValueError(f"policy.lanes[{index}] does not match its exact token share")
        canary_rows = _positive_int(lane["canary_rows"], f"policy.lanes[{index}].canary_rows")
        if canary_rows != CANARY_ROWS[index]:
            raise ValueError(f"policy.lanes[{index}] does not match its exact canary rows")
        languages = _string_tuple(lane["languages"], f"policy.lanes[{index}].languages")
        if languages != _EXPECTED_LANGUAGES[index]:
            raise ValueError(f"policy.lanes[{index}] does not match its exact languages")
        language_allocations = _parse_language_allocations(
            lane["language_allocations"], languages, index
        )
        lanes.append(
            LanePolicy(name, numerator, denominator, canary_rows, languages, language_allocations)
        )
    parsed = tuple(lanes)
    if sum((lane.token_share for lane in parsed), Fraction()) != Fraction(1, 1):
        raise ValueError("policy lane token shares must sum to one")
    if sum(lane.canary_rows for lane in parsed) != 10_240:
        raise ValueError("policy lane canary rows must sum to 10240")
    return parsed


def _parse_language_allocations(
    value: object, languages: tuple[str, ...], lane_index: int
) -> tuple[LanguageAllocation, ...]:
    entries = _list(value, f"policy.lanes[{lane_index}].language_allocations")
    expected_fractions = _EXPECTED_LANGUAGE_ALLOCATION_FRACTIONS[lane_index]
    if len(entries) != len(languages):
        raise ValueError(f"policy.lanes[{lane_index}] does not match exact language allocations")
    allocations: list[LanguageAllocation] = []
    for language_index, entry in enumerate(entries):
        allocation = _mapping(
            entry, f"policy.lanes[{lane_index}].language_allocations[{language_index}]"
        )
        _require_exact_keys(
            allocation,
            _LANGUAGE_ALLOCATION_KEYS,
            f"policy.lanes[{lane_index}].language_allocations[{language_index}]",
        )
        language = _literal(
            allocation["language"],
            languages[language_index],
            f"policy.lanes[{lane_index}].language_allocations[{language_index}].language",
        )
        numerator = _positive_int(
            allocation["token_numerator"],
            f"policy.lanes[{lane_index}].language_allocations[{language_index}].token_numerator",
        )
        denominator = _positive_int(
            allocation["token_denominator"],
            f"policy.lanes[{lane_index}].language_allocations[{language_index}].token_denominator",
        )
        if (numerator, denominator) != expected_fractions[language_index]:
            raise ValueError(f"policy.lanes[{lane_index}] does not match exact language allocation")
        allocations.append(LanguageAllocation(language, numerator, denominator))
    parsed = tuple(allocations)
    if languages and sum((allocation.token_share for allocation in parsed), Fraction()) != Fraction(
        1, 1
    ):
        raise ValueError(f"policy.lanes[{lane_index}] language allocations must sum to one")
    return parsed


def _parse_gates(value: object) -> tuple[TokenGate, ...]:
    entries = _list(value, "policy.gates")
    if len(entries) != len(_GATE_NAMES):
        raise ValueError("policy.gates must contain pilot, primary, and scale")
    gates: list[TokenGate] = []
    for index, entry in enumerate(entries):
        gate = _mapping(entry, f"policy.gates[{index}]")
        _require_exact_keys(gate, _GATE_KEYS, f"policy.gates[{index}]")
        name = cast(
            "Literal['pilot', 'primary', 'scale']",
            _literal(gate["name"], _GATE_NAMES[index], f"policy.gates[{index}].name"),
        )
        tokens = _positive_int(
            gate["assistant_loss_tokens"], f"policy.gates[{index}].assistant_loss_tokens"
        )
        if tokens != _GATE_TOKENS[index]:
            raise ValueError(f"policy.gates[{index}] does not match its exact token gate")
        gates.append(TokenGate(name, tokens))
    return tuple(gates)


def _verify_source_requirements_binding(
    policy_path: Path, source_requirements_path: str, expected_sha256: str
) -> None:
    source_path = policy_path.parent / source_requirements_path
    if source_path.parent != policy_path.parent:
        raise ValueError("policy.source_requirements.path must name a sibling file")
    actual_sha256 = sha256(_read_stable_regular_file(source_path)).hexdigest()
    if actual_sha256 != expected_sha256:
        raise ValueError("policy source-requirements digest does not match its bound file")


def _read_stable_regular_file(path: Path) -> bytes:
    if not hasattr(os, "O_NOFOLLOW"):
        raise ValueError("stable policy loading requires O_NOFOLLOW")
    flags = os.O_RDONLY | os.O_NOFOLLOW | getattr(os, "O_CLOEXEC", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        raise ValueError(f"policy input is not a no-follow regular file: {path}") from error
    try:
        initial = os.fstat(descriptor)
        if not stat.S_ISREG(initial.st_mode) or initial.st_nlink != 1:
            raise ValueError(f"policy input is not a single-link regular file: {path}")
        chunks: list[bytes] = []
        while chunk := os.read(descriptor, 1 << 20):
            chunks.append(chunk)
        final = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    try:
        pathname = os.lstat(path)
    except OSError as error:
        raise ValueError(f"policy input changed while loading: {path}") from error
    if (
        _file_identity(initial) != _file_identity(final)
        or not stat.S_ISREG(pathname.st_mode)
        or pathname.st_nlink != 1
        or (pathname.st_dev, pathname.st_ino) != (final.st_dev, final.st_ino)
    ):
        raise ValueError(f"policy input changed while loading: {path}")
    return b"".join(chunks)


def _file_identity(observation: os.stat_result) -> tuple[int, int, int, int, int]:
    return (
        observation.st_dev,
        observation.st_ino,
        observation.st_size,
        observation.st_mtime_ns,
        observation.st_ctime_ns,
    )


def _load_canonical_mapping(raw: bytes, name: str) -> dict[str, Any]:
    try:
        value = json.loads(raw, object_pairs_hook=_reject_duplicate_keys)
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
        raise ValueError(f"{name} must be valid JSON with unique object keys") from error
    root = _mapping(value, name)
    if raw != canonical_json(root) + b"\n":
        raise ValueError(f"{name} must use canonical JSON bytes")
    return root


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON object key: {key}")
        result[key] = value
    return result


def _mapping(value: object, name: str) -> dict[str, Any]:
    if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
        raise ValueError(f"{name} must be a JSON object with string keys")
    return value


def _require_exact_keys(value: Mapping[str, Any], expected: frozenset[str], name: str) -> None:
    unknown = set(value) - expected
    missing = expected - set(value)
    if unknown or missing:
        raise ValueError(f"{name} has unknown or missing keys: {sorted(unknown | missing)}")


def _list(value: object, name: str) -> list[object]:
    if not isinstance(value, list):
        raise ValueError(f"{name} must be a list")
    return value


def _string_tuple(value: object, name: str) -> tuple[str, ...]:
    entries = _list(value, name)
    strings: list[str] = []
    for entry in entries:
        if not isinstance(entry, str):
            raise ValueError(f"{name} must contain only strings")
        strings.append(entry)
    return tuple(strings)


def _positive_int(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _literal(value: object, expected: str, name: str) -> str:
    if not isinstance(value, str) or value != expected:
        raise ValueError(f"{name} must be {expected!r}")
    return value


def _sha256(value: object, name: str) -> str:
    if not isinstance(value, str) or not _SHA256.fullmatch(value):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return value
