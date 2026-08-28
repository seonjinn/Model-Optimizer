# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Contracts for the Q30 from-scratch scientific policy boundary."""

from __future__ import annotations

import json
import sys
from fractions import Fraction
from hashlib import sha256
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
MODULE_DIR = ROOT / "examples/dataset"
POLICY_PATH = MODULE_DIR / "qwen3_30ba3b_thinking_ptv23_from_scratch_v1.json"
SOURCES_PATH = MODULE_DIR / "qwen3_30ba3b_thinking_ptv23_from_scratch_sources_v1.json"

sys.path.insert(0, str(MODULE_DIR))
try:
    from q30t_from_scratch_policy import (
        APPROVED_FROM_SCRATCH_POLICY_SHA256S,
        CANARY_ROWS,
        LANE_ORDER,
        TOKEN_SHARES,
        FromScratchStudyPolicy,
        _parse_from_scratch_policy_for_test,
        load_from_scratch_policy,
    )
finally:
    sys.path.pop(0)


def load_fixture_policy_for_test() -> FromScratchStudyPolicy:
    """Load the checked-in fixture without granting it production approval."""
    return _parse_from_scratch_policy_for_test(POLICY_PATH)


def test_policy_freezes_lane_order_and_arithmetic() -> None:
    """A changed lane name, quota, canary, or gate cannot enter the study."""
    policy = load_fixture_policy_for_test()

    assert tuple(lane.name for lane in policy.lanes) == (
        "agentless_swe",
        "interactive_swe",
        "general_tool",
        "math",
        "code",
        "stem",
        "instruction",
        "multilingual",
    )
    assert tuple(lane.name for lane in policy.lanes) == LANE_ORDER
    assert CANARY_ROWS == (1_536, 1_024, 1_024, 2_560, 1_536, 1_536, 512, 512)
    assert (
        Fraction(15, 100),
        Fraction(10, 100),
        Fraction(10, 100),
        Fraction(25, 100),
        Fraction(15, 100),
        Fraction(15, 100),
        Fraction(5, 100),
        Fraction(5, 100),
    ) == TOKEN_SHARES
    assert sum(lane.canary_rows for lane in policy.lanes) == 10_240
    assert sum((lane.token_share for lane in policy.lanes), Fraction()) == Fraction(1, 1)
    assert tuple(gate.assistant_loss_tokens for gate in policy.gates) == (
        256_000_000,
        1_000_000_000,
        4_000_000_000,
    )


def test_policy_binds_the_expected_target_and_source_requirements() -> None:
    """A policy cannot silently select another Q30 revision or source registry."""
    policy = load_fixture_policy_for_test()

    assert policy.target_repository == "Qwen/Qwen3-30B-A3B-Thinking-2507"
    assert policy.target_revision == "144afc2f379b542fdd4e85a1fcd5e1f79112d95d"
    assert policy.source_requirements_path == SOURCES_PATH.name
    assert policy.source_requirements_sha256 == sha256(SOURCES_PATH.read_bytes()).hexdigest()


def test_multilingual_policy_allocates_equal_capacity_to_each_language() -> None:
    """The 5% multilingual lane cannot hide an unequal language allocation."""
    policy = load_fixture_policy_for_test()
    multilingual = next(lane for lane in policy.lanes if lane.name == "multilingual")

    assert multilingual.token_share == Fraction(5, 100)
    assert tuple(allocation.language for allocation in multilingual.language_allocations) == (
        "de",
        "ja",
        "es",
        "fr",
        "it",
    )
    assert tuple(allocation.token_share for allocation in multilingual.language_allocations) == (
        Fraction(1, 5),
        Fraction(1, 5),
        Fraction(1, 5),
        Fraction(1, 5),
        Fraction(1, 5),
    )


def test_policy_rejects_unequal_but_self_consistent_multilingual_allocations(
    tmp_path: Path,
) -> None:
    """Five allocations summing to one still fail unless each language receives one fifth."""
    payload = json.loads(POLICY_PATH.read_text(encoding="utf-8"))
    payload["lanes"][-1]["language_allocations"] = [
        {"language": "de", "token_numerator": 2, "token_denominator": 5},
        {"language": "ja", "token_numerator": 3, "token_denominator": 20},
        {"language": "es", "token_numerator": 3, "token_denominator": 20},
        {"language": "fr", "token_numerator": 3, "token_denominator": 20},
        {"language": "it", "token_numerator": 3, "token_denominator": 20},
    ]
    policy_path = tmp_path / POLICY_PATH.name
    policy_path.write_text(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    (tmp_path / SOURCES_PATH.name).write_bytes(SOURCES_PATH.read_bytes())

    with pytest.raises(ValueError, match="exact language allocation"):
        _parse_from_scratch_policy_for_test(policy_path)


def test_policy_has_no_parent_checkpoint_field() -> None:
    """The from-scratch policy cannot carry a continuation checkpoint input."""
    payload = json.loads(POLICY_PATH.read_text(encoding="utf-8"))
    forbidden = {"parent_checkpoint", "dflash_init_checkpoint", "resume_from_checkpoint"}

    assert forbidden.isdisjoint(payload)


def test_production_loader_rejects_the_unapproved_checked_in_digest() -> None:
    """A checked-in policy never authorizes itself before external approval."""
    digest = sha256(POLICY_PATH.read_bytes()).hexdigest()

    assert frozenset() == APPROVED_FROM_SCRATCH_POLICY_SHA256S
    with pytest.raises(ValueError, match="not approved"):
        load_from_scratch_policy(POLICY_PATH, expected_sha256=digest)
