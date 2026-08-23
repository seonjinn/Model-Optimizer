# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Strict, immutable prompt-count policy for the B-prime/C/D study arms."""

from __future__ import annotations

import json
from dataclasses import dataclass
from hashlib import sha256
from types import MappingProxyType
from typing import TYPE_CHECKING, Any

import yaml

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence
    from pathlib import Path

__all__ = ["ArmPolicy", "PromptCell", "PromptPolicy", "load_prompt_policy"]

_ROOT_KEYS = frozenset(
    {
        "schema_version",
        "seed",
        "reserve",
        "exposure_tokens",
        "sequence_length",
        "full_context_maximum",
        "languages",
        "arms",
    }
)
_ARM_KEYS = frozenset({"prompt_count", "cells", "lanes"})
_LANGUAGE_KEYS = frozenset({"allowed", "denied"})
_RESERVE_KEYS = frozenset({"numerator", "denominator"})
_ARM_CELLS = {
    "B-prime": frozenset({"math", "code", "stem", "chat", "multilingual"}),
    "C": frozenset({"math", "code", "stem", "chat", "multilingual", "swe"}),
    "D": frozenset({"math", "code", "stem", "chat", "multilingual", "swe"}),
}
_D_LANES = frozenset({"agentless-swe", "interactive-swe-replay", "generic-tool-replay"})


@dataclass(frozen=True)
class PromptCell:
    """An exact prompt count for one arm/category cell."""

    prompt_count: int


@dataclass(frozen=True)
class ArmPolicy:
    """Exact counts for one study arm."""

    prompt_count: int
    cells: Mapping[str, PromptCell]
    lanes: Mapping[str, int]


@dataclass(frozen=True)
class PromptPolicy:
    """Validated prompt policy with exact count and reserve lookups."""

    schema_version: int
    seed: int
    reserve_numerator: int
    reserve_denominator: int
    arms: Mapping[str, ArmPolicy]
    exposure_tokens: Sequence[int]
    sequence_length: int
    full_context_maximum: int
    allowed_languages: frozenset[str]
    denied_languages: frozenset[str]
    policy_sha256: str

    def count_for(self, arm: str, cell_or_lane: str | None = None) -> int:
        """Return the exact arm, cell, or D-lane prompt count."""
        arm_policy = self.arms.get(arm)
        if arm_policy is None:
            raise ValueError(f"unknown arm: {arm}")
        if cell_or_lane is None:
            return arm_policy.prompt_count
        cell = arm_policy.cells.get(cell_or_lane)
        if cell is not None:
            return cell.prompt_count
        lane = arm_policy.lanes.get(cell_or_lane)
        if lane is not None:
            return lane
        raise ValueError(f"unknown cell or lane for {arm}: {cell_or_lane}")

    def reserve_count(self, arm: str, cell_or_lane: str | None = None) -> int:
        """Return the 20-percent rounded-up acquisition reserve for a quota."""
        count = self.count_for(arm, cell_or_lane)
        return (
            count * self.reserve_numerator + self.reserve_denominator - 1
        ) // self.reserve_denominator

    def is_language_allowed(self, language: str) -> bool:
        """Return whether a language is permitted by the policy boundary."""
        return language in self.allowed_languages and language not in self.denied_languages


def load_prompt_policy(path: Path) -> PromptPolicy:
    """Load and validate the approved B-prime/C/D prompt-count policy."""
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as error:
        raise ValueError(f"invalid YAML policy: {error}") from error
    root = _mapping(raw, "policy")
    _require_exact_keys(root, _ROOT_KEYS, "policy")

    reserve = _mapping(root["reserve"], "reserve")
    _require_exact_keys(reserve, _RESERVE_KEYS, "reserve")
    numerator = _positive_int(reserve["numerator"], "reserve.numerator")
    denominator = _positive_int(reserve["denominator"], "reserve.denominator")
    if (numerator, denominator) != (6, 5):
        raise ValueError("reserve must use numerator/denominator 6/5")

    languages = _mapping(root["languages"], "languages")
    _require_exact_keys(languages, _LANGUAGE_KEYS, "languages")
    allowed = _language_set(languages["allowed"], "languages.allowed")
    denied = _language_set(languages["denied"], "languages.denied")
    if "de" not in denied:
        raise ValueError("languages.denied must include de")
    if allowed & denied:
        raise ValueError("allowed and denied languages must not overlap")

    raw_arms = _mapping(root["arms"], "arms")
    if set(raw_arms) != set(_ARM_CELLS):
        raise ValueError("arms must contain exactly B-prime, C, and D")
    arms = {name: _parse_arm(name, raw_arms[name]) for name in _ARM_CELLS}

    exposure_tokens = _positive_int_tuple(root["exposure_tokens"], "exposure_tokens")
    if exposure_tokens != (256_000_000, 1_000_000_000):
        raise ValueError("exposure_tokens must be 256000000 and 1000000000")
    sequence_length = _positive_int(root["sequence_length"], "sequence_length")
    if sequence_length != 4_096:
        raise ValueError("sequence_length must be 4096")
    full_context_maximum = _positive_int(root["full_context_maximum"], "full_context_maximum")
    if full_context_maximum != 32_768:
        raise ValueError("full_context_maximum must be 32768")

    canonical = _canonical_json(root)
    return PromptPolicy(
        schema_version=_positive_int(root["schema_version"], "schema_version"),
        seed=_positive_int(root["seed"], "seed"),
        reserve_numerator=numerator,
        reserve_denominator=denominator,
        arms=MappingProxyType(arms),
        exposure_tokens=exposure_tokens,
        sequence_length=sequence_length,
        full_context_maximum=full_context_maximum,
        allowed_languages=allowed,
        denied_languages=denied,
        policy_sha256=sha256(canonical).hexdigest(),
    )


def _parse_arm(name: str, raw: Any) -> ArmPolicy:
    arm = _mapping(raw, f"arms.{name}")
    _require_exact_keys(arm, _ARM_KEYS, f"arms.{name}")
    cells = _mapping(arm["cells"], f"arms.{name}.cells")
    if set(cells) != _ARM_CELLS[name]:
        raise ValueError(f"arms.{name}.cells must contain its exact policy cells")
    parsed_cells = {
        cell_name: PromptCell(_positive_int(value, f"arms.{name}.cells.{cell_name}"))
        for cell_name, value in cells.items()
    }
    prompt_count = _positive_int(arm["prompt_count"], f"arms.{name}.prompt_count")
    if sum(cell.prompt_count for cell in parsed_cells.values()) != prompt_count:
        raise ValueError(f"arms.{name}.prompt_count does not match its cells")

    lanes = _mapping(arm["lanes"], f"arms.{name}.lanes")
    expected_lanes = _D_LANES if name == "D" else frozenset()
    if set(lanes) != expected_lanes:
        raise ValueError(f"arms.{name}.lanes must contain its exact policy lanes")
    parsed_lanes = {
        lane_name: _positive_int(value, f"arms.{name}.lanes.{lane_name}")
        for lane_name, value in lanes.items()
    }
    if name == "D" and sum(parsed_lanes.values()) != 600_000:
        raise ValueError("arms.D lanes must total 600000")
    return ArmPolicy(prompt_count, MappingProxyType(parsed_cells), MappingProxyType(parsed_lanes))


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
        raise ValueError(f"{name} must be a mapping with string keys")
    return value


def _require_exact_keys(value: Mapping[str, Any], expected: frozenset[str], name: str) -> None:
    unknown = set(value) - expected
    missing = expected - set(value)
    if unknown or missing:
        raise ValueError(f"{name} has unknown or missing keys: {sorted(unknown | missing)}")


def _positive_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _positive_int_tuple(value: Any, name: str) -> tuple[int, ...]:
    if not isinstance(value, list):
        raise ValueError(f"{name} must be a list of positive integers")
    return tuple(_positive_int(item, f"{name}[{index}]") for index, item in enumerate(value))


def _language_set(value: Any, name: str) -> frozenset[str]:
    if not isinstance(value, list) or not value or any(not isinstance(item, str) for item in value):
        raise ValueError(f"{name} must be a non-empty list of strings")
    languages = frozenset(value)
    if len(languages) != len(value):
        raise ValueError(f"{name} must not contain duplicates")
    return languages


def _canonical_json(value: Mapping[str, Any]) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode(
        "utf-8"
    )
