# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Contracts for response-safe Q30 prompt candidates."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
MODULE_DIR = ROOT / "examples/dataset"

sys.path.insert(0, str(MODULE_DIR))
try:
    from q30t_from_scratch_candidates import (  # pyright: ignore[reportMissingImports]
        CandidateAdapterError,
        adapt_historical_occurrence,
        adapt_target_synthesis_row,
    )
    from q30t_from_scratch_sources import (  # pyright: ignore[reportMissingImports]
        SourceRegistryEntry,
    )
    from specdec_identity import prompt_uuid  # pyright: ignore[reportMissingImports]
finally:
    sys.path.pop(0)


def _source_entry(
    *,
    adapter: str = "ptv2_messages_tools_v1",
    lane: str = "math",
) -> SourceRegistryEntry:
    return SourceRegistryEntry(
        source_id="PTV2-math",
        repository="nvidia/PTV2",
        revision="a" * 40,
        configuration="default",
        split="train",
        relative_path="math-0.parquet",
        bytes=1024,
        file_sha256="b" * 64,
        row_count=2,
        row_schema_sha256="c" * 64,
        license_expression="NVIDIA-Nemotron",
        approved_use="specdec-drafter-training",
        lane=lane,
        adapter=adapter,
    )


def _historical_row(*, answer: str = "source answer") -> dict[str, object]:
    return {
        "messages": [
            {"role": "system", "content": "Answer carefully."},
            {"role": "user", "content": "Solve 1 + 1."},
            {"role": "assistant", "content": answer},
        ],
        "tools": [],
        "source_row_index": 17,
        "reasoning_mode": "reasoning_on",
        "language": "",
    }


def _expected_historical_uuid(row: dict[str, object]) -> str:
    return prompt_uuid(row["messages"], row["tools"])


def test_h_native_and_h_synth_share_prompt_identity_but_not_response() -> None:
    row = _historical_row()

    native, synth = adapt_historical_occurrence(
        row,
        _source_entry(),
        expected_prompt_uuid=_expected_historical_uuid(row),
    )

    assert native.arm == "H-native"
    assert synth.arm == "H-synth"
    assert native.identity == synth.identity
    assert native.source_assistant == {"role": "assistant", "content": "source answer"}
    assert synth.source_assistant is None
    assert native.request_bytes == b""
    assert b"source answer" not in synth.request_bytes


def test_m_synth_strips_source_completion_before_identity_and_request() -> None:
    row = _historical_row(answer="secret answer")

    candidate = adapt_target_synthesis_row(row, _source_entry(adapter="ptv3_messages_tools_v1"))

    assert candidate.arm == "M-synth"
    assert candidate.source_assistant is None
    assert b"secret answer" not in candidate.identity.canonical_bytes
    assert b"secret answer" not in candidate.request_bytes
    assert candidate.request_bytes == (
        b'{"messages":[{"content":"Answer carefully.","role":"system"},'
        b'{"content":"Solve 1 + 1.","role":"user"}],"tools":[]}'
    )


def test_prior_assistant_context_remains_while_only_final_completion_is_removed() -> None:
    row = _historical_row()
    row["messages"] = [
        {"role": "user", "content": "First question"},
        {"role": "assistant", "content": "First answer"},
        {"role": "user", "content": "Follow-up"},
        {"role": "assistant", "content": "secret answer"},
    ]

    candidate = adapt_target_synthesis_row(row, _source_entry(adapter="ptv3_messages_tools_v1"))

    assert b"First answer" in candidate.identity.canonical_bytes
    assert b"First answer" in candidate.request_bytes
    assert b"secret answer" not in candidate.identity.canonical_bytes
    assert b"secret answer" not in candidate.request_bytes


def test_historical_occurrence_rejects_a_different_ordered_audit_uuid() -> None:
    row = _historical_row()

    with pytest.raises(CandidateAdapterError, match="historical occurrence prompt UUID"):
        adapt_historical_occurrence(row, _source_entry(), expected_prompt_uuid="0" * 64)


def test_adapter_name_controls_schema_without_fallback_heuristics() -> None:
    row = _historical_row()
    row["conversations"] = row.pop("messages")

    with pytest.raises(CandidateAdapterError, match="messages"):
        adapt_target_synthesis_row(row, _source_entry(adapter="ptv3_messages_tools_v1"))

    with pytest.raises(CandidateAdapterError, match="unsupported source adapter"):
        adapt_target_synthesis_row(_historical_row(), _source_entry(adapter="unknown-v1"))


def test_normalized_row_hash_authenticates_the_removed_source_response() -> None:
    first = _historical_row(answer="answer one")
    second = _historical_row(answer="answer two")

    first_candidate = adapt_target_synthesis_row(
        first, _source_entry(adapter="ptv3_messages_tools_v1")
    )
    second_candidate = adapt_target_synthesis_row(
        second, _source_entry(adapter="ptv3_messages_tools_v1")
    )

    assert first_candidate.identity == second_candidate.identity
    assert first_candidate.request_bytes == second_candidate.request_bytes
    assert first_candidate.normalized_row_sha256 != second_candidate.normalized_row_sha256


def test_source_lane_is_metadata_and_never_replaced_by_the_arm() -> None:
    candidate = adapt_target_synthesis_row(
        _historical_row(),
        _source_entry(adapter="ptv3_messages_tools_v1", lane="stem"),
    )

    assert candidate.lane == "stem"
    assert candidate.identity.provenance.response_lane == "stem"
    assert candidate.identity.provenance.response_lane != candidate.arm
