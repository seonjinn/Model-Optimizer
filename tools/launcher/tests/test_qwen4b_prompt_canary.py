# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for immutable Qwen3-4B prompt-canary construction."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from common.specdec.build_qwen4b_prompt_canary import (
    _context_bucket,
    _decode_raw_row,
    _has_tool_trajectory,
    _prompt_identity,
    _prompt_view,
    _tokenize_with_assistant_mask,
)


def test_raw_json_union_row_is_decoded_without_losing_tools() -> None:
    """The explicit wrapper must retain heterogeneous tool fields."""
    payload = {
        "messages": [{"role": "user", "content": "run it"}],
        "tools": [{"type": "function", "function": {"name": "shell"}}],
    }

    assert _decode_raw_row({"raw_json": json.dumps(payload)}) == payload
    assert _has_tool_trajectory(payload)


def test_non_tool_prompt_view_removes_source_answers_but_keeps_turns() -> None:
    """Target synthesis uses source prompts but never source assistant responses."""
    row = {
        "messages": [
            {"role": "system", "content": "be exact"},
            {"role": "user", "content": "first"},
            {"role": "assistant", "content": "source answer"},
            {"role": "user", "content": "second"},
        ]
    }

    prompt = _prompt_view(row)

    assert [message["role"] for message in prompt["messages"]] == ["system", "user", "user"]
    assert "tools" not in prompt
    assert _prompt_identity(prompt) == _prompt_identity(prompt)


def test_prompt_view_rejects_tool_trajectories() -> None:
    """Executed tool traces cannot silently enter target-only generation."""
    row = {
        "messages": [
            {"role": "user", "content": "run"},
            {"role": "assistant", "content": "", "tool_calls": [{"id": "1"}]},
        ]
    }

    with pytest.raises(ValueError, match="tool trajectory"):
        _prompt_view(row)


def test_trace_tokenization_preserves_tools_and_assistant_loss_mask() -> None:
    """Tool-aware chat templating returns an aligned assistant-only loss mask."""

    class FakeTokenizer:
        def apply_chat_template(self, messages: list[dict], **kwargs: object) -> dict:
            assert messages[-1]["role"] == "tool"
            assert kwargs["tools"] == [{"type": "function", "function": {"name": "shell"}}]
            assert kwargs["return_assistant_tokens_mask"] is True
            return {"input_ids": [1, 2, 3, 4], "assistant_masks": [0, 0, 1, 0]}

    row = {
        "tools": [{"type": "function", "function": {"name": "shell"}}],
        "messages": [
            {"role": "user", "content": "run"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [{"id": "call-1", "function": {"name": "shell"}}],
            },
            {"role": "tool", "tool_call_id": "call-1", "content": "ok"},
        ],
    }

    assert _tokenize_with_assistant_mask(FakeTokenizer(), row) == (
        [1, 2, 3, 4],
        [0, 0, 1, 0],
    )


@pytest.mark.parametrize(
    ("tokens", "expected"),
    [(4096, "le4k"), (4097, "4k_16k"), (16384, "4k_16k"), (16385, "16k_32k")],
)
def test_context_bucket_uses_measured_full_context(tokens: int, expected: str) -> None:
    """Measured token lengths determine every long-context bucket."""
    assert _context_bucket(tokens) == expected


def test_context_bucket_rejects_over_32k() -> None:
    """The canary does not silently truncate unsupported contexts."""
    with pytest.raises(ValueError, match="32K"):
        _context_bucket(32769)


def test_prompt_canary_runner_is_hash_bound_and_uses_compute() -> None:
    """Construction runs under SLURM with exact source and artifact identities."""
    runner = (
        Path(__file__).resolve().parents[1] / "common/specdec/run_qwen4b_prompt_canary.sbatch"
    ).read_text()

    assert "#SBATCH --gpus-per-node=4" in runner
    assert "SOURCE_MANIFEST_SHA256" in runner
    assert "TOKENIZER_SHA256" in runner
    assert "RUNTIME_ARCHIVE_SHA256" in runner
    assert 'tar --extract --file="$RUNTIME_ARCHIVE"' in runner
    assert 'export VIRTUAL_ENV="$JOB_ROOT/runtime"' in runner
    assert 'git -C "$SOURCE_PATH" rev-parse HEAD' in runner
    assert "srun --nodes=1" in runner


def test_synthesis_canary_wrapper_binds_both_lanes_and_mode() -> None:
    """Dependent synthesis binds prompts, traces, tokenizer, and explicit mode."""
    wrapper = (
        Path(__file__).resolve().parents[1] / "common/specdec/run_qwen4b_synthesis_canary.sbatch"
    ).read_text()

    assert 'THINKING_MODE" == on || "$THINKING_MODE" == off' in wrapper
    assert "PROMPT_ROOT/target-synth/MANIFEST.json" in wrapper
    assert "PROMPT_ROOT/trace-replay/MANIFEST.json" in wrapper
    assert "verify_data_manifest(prompt_manifest" in wrapper
    assert "verify_data_manifest(trace_manifest" in wrapper
    assert "num_shards=8" in wrapper
    assert (
        'exec bash "$SOURCE_PATH/tools/launcher/common/specdec/run_qwen4b_synthesis.sbatch"'
        in wrapper
    )
