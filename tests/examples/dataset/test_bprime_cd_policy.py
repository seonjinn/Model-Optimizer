# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the immutable B-prime/C/D prompt-count policy."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
POLICY = ROOT / "examples/dataset/bprime_cd_prompt_policy.yaml"
MODULE_DIR = ROOT / "examples/dataset"

sys.path.insert(0, str(MODULE_DIR))
try:
    from bprime_cd_policy import load_prompt_policy
finally:
    sys.path.pop(0)


def test_approved_policy_has_exact_integer_counts() -> None:
    """Changing a quota, lane split, reserve multiplier, or language gate fails here."""
    policy = load_prompt_policy(POLICY)

    assert policy.arms["B-prime"].prompt_count == 700_000
    assert policy.arms["C"].prompt_count == 2_000_000
    assert policy.arms["D"].prompt_count == 2_000_000
    assert policy.arms["D"].lanes == {
        "agentless-swe": 200_000,
        "generic-tool-replay": 200_000,
        "interactive-swe-replay": 200_000,
    }
    assert sum(policy.arms["D"].lanes.values()) == 600_000
    assert policy.reserve_count("D", "interactive-swe-replay") == 240_000
    assert not policy.is_language_allowed("de")
    assert policy.exposure_tokens == (256_000_000, 1_000_000_000)


@pytest.mark.parametrize(
    ("replacement", "message"),
    [
        (("math: 200000", "math: 0.2"), "positive integer"),
        (("prompt_count: 700000", "prompt_count: 700001"), "does not match"),
    ],
)
def test_policy_rejects_float_weights_and_mismatched_totals(
    tmp_path: Path, replacement: tuple[str, str], message: str
) -> None:
    """Changing count semantics from exact integers or breaking an aggregate is rejected."""
    source, target = replacement
    invalid = tmp_path / "invalid-policy.yaml"
    invalid.write_text(
        POLICY.read_text(encoding="utf-8").replace(source, target, 1), encoding="utf-8"
    )

    with pytest.raises(ValueError, match=message):
        load_prompt_policy(invalid)
