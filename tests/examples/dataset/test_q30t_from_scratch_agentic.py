# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Contracts for safe target-generated Q30 agentic prefixes."""

from __future__ import annotations

import json
import sys
from copy import deepcopy
from pathlib import Path
from typing import TYPE_CHECKING, cast

import pytest

if TYPE_CHECKING:
    from collections.abc import Mapping

ROOT = Path(__file__).resolve().parents[3]
MODULE_DIR = ROOT / "examples/dataset"

sys.path.insert(0, str(MODULE_DIR))
try:
    from q30t_from_scratch_agentic import (  # pyright: ignore[reportMissingImports]
        AgenticPrefixError,
        accept_target_action,
        agentic_generation_prefixes,
        canonical_tool_call,
        recorded_result_matches,
    )
    from q30t_from_scratch_candidates import (  # pyright: ignore[reportMissingImports]
        PromptCandidate,
        adapt_target_synthesis_row,
    )
    from q30t_from_scratch_sources import (  # pyright: ignore[reportMissingImports]
        SourceRegistryEntry,
    )
finally:
    sys.path.pop(0)


def _call(call_id: str, name: str, arguments: object) -> dict[str, object]:
    return {
        "id": call_id,
        "type": "function",
        "function": {"name": name, "arguments": arguments},
    }


def _shell_tool() -> dict[str, object]:
    return {
        "type": "function",
        "function": {
            "name": "shell",
            "parameters": {"type": "object", "properties": {"cmd": {"type": "string"}}},
        },
    }


def _source_entry() -> SourceRegistryEntry:
    return SourceRegistryEntry(
        source_id="OpenHands-train",
        repository="OpenHands/OpenHands",
        revision="a" * 40,
        configuration="default",
        split="train",
        relative_path="openhands.parquet",
        bytes=1024,
        file_sha256="b" * 64,
        row_count=1,
        row_schema_sha256="c" * 64,
        license_expression="MIT",
        approved_use="specdec-drafter-training",
        lane="interactive_swe",
        adapter="ptv3_messages_tools_v1",
    )


def _trajectory() -> dict[str, object]:
    return {
        "messages": [
            {"role": "user", "content": "Inspect and test the repository."},
            {
                "role": "assistant",
                "reasoning_content": "I should inspect the tree.",
                "tool_calls": [_call("call-1", "shell", '{"cmd":"ls"}')],
            },
            {
                "role": "tool",
                "tool_call_id": "call-1",
                "name": "shell",
                "content": "a.py",
            },
            {
                "role": "assistant",
                "reasoning_content": "I should run the tests.",
                "tool_calls": [_call("call-2", "shell", {"cmd": "pytest"})],
            },
            {
                "role": "tool",
                "tool_call_id": "call-2",
                "name": "shell",
                "content": "1 passed",
            },
            {"role": "assistant", "content": "The repository is healthy."},
        ],
        "tools": [_shell_tool()],
        "source_row_index": 17,
        "reasoning_mode": "reasoning_on",
        "language": "",
    }


def _candidate(row: Mapping[str, object] | None = None) -> PromptCandidate:
    return adapt_target_synthesis_row(row or _trajectory(), _source_entry())


def test_json_string_and_object_arguments_canonicalize_identically() -> None:
    left = _call("1", "shell", '{"cmd":"ls","timeout":10}')
    right = _call("x", "shell", {"timeout": 10, "cmd": "ls"})

    assert canonical_tool_call(left).function_bytes == canonical_tool_call(right).function_bytes


def test_call_ids_never_establish_or_prevent_recorded_result_equivalence() -> None:
    recorded = canonical_tool_call(_call("same", "shell", {"cmd": "ls"}))
    divergent = canonical_tool_call(_call("same", "shell", {"cmd": "rm file"}))
    renumbered = canonical_tool_call(_call("different", "shell", {"cmd": "ls"}))

    assert not recorded_result_matches(divergent, recorded)
    assert recorded_result_matches(renumbered, recorded)


def test_divergent_non_executable_call_fails_closed_but_executable_call_may_run() -> None:
    source = canonical_tool_call(_call("1", "shell", {"cmd": "ls"}))
    target = canonical_tool_call(_call("7", "shell", {"cmd": "rm file"}))

    with pytest.raises(AgenticPrefixError, match="executable environment"):
        accept_target_action(target, source, executable=False)
    assert accept_target_action(target, source, executable=True) == "execute"
    assert accept_target_action(source, source, executable=False) == "reuse-recorded"


@pytest.mark.parametrize("executable", ["false", 1, 0, None])
def test_divergent_call_rejects_non_boolean_execution_modes(executable: object) -> None:
    source = canonical_tool_call(_call("1", "shell", {"cmd": "ls"}))
    target = canonical_tool_call(_call("7", "shell", {"cmd": "rm file"}))

    with pytest.raises(AgenticPrefixError, match="executable must be a boolean"):
        accept_target_action(target, source, executable=cast("bool", executable))


def test_prefixes_exclude_the_recorded_next_transaction_from_request_bytes() -> None:
    source = _trajectory()
    candidate = _candidate(source)

    prefixes = agentic_generation_prefixes(
        candidate, source, executable_environment=False
    )

    assert len(prefixes) == 3
    first, second, final = prefixes
    assert first.source_next_call == canonical_tool_call(
        _call("call-1", "shell", {"cmd": "ls"})
    )
    assert first.recorded_result == {
        "content": "a.py",
        "name": "shell",
        "role": "tool",
        "tool_call_id": "call-1",
    }
    assert b'"cmd":"ls"' not in first.candidate.request_bytes
    assert b"a.py" not in first.candidate.request_bytes

    assert second.source_next_call == canonical_tool_call(
        _call("call-2", "shell", {"cmd": "pytest"})
    )
    second_request = json.loads(second.candidate.request_bytes)
    assert second_request["messages"][1]["tool_calls"][0]["function"]["arguments"] == (
        '{"cmd":"ls"}'
    )
    assert b"a.py" in second.candidate.request_bytes
    assert b'"cmd":"pytest"' not in second.candidate.request_bytes
    assert b"1 passed" not in second.candidate.request_bytes

    assert final.source_next_call is None
    assert final.recorded_result is None
    assert final.candidate == candidate
    assert b"The repository is healthy." not in final.candidate.request_bytes
    assert all(prefix.candidate.identity.provenance.response_lane == "interactive_swe" for prefix in prefixes)
    assert all(prefix.candidate.arm == "M-synth" for prefix in prefixes)


def test_multi_call_action_is_rejected_as_ambiguous_for_single_call_reuse() -> None:
    row = _trajectory()
    messages = row["messages"]
    assert isinstance(messages, list)
    first_action = messages[1]
    assert isinstance(first_action, dict)
    calls = first_action["tool_calls"]
    assert isinstance(calls, list)
    calls.append(_call("call-extra", "shell", {"cmd": "pwd"}))
    messages.insert(
        3,
        {
            "role": "tool",
            "tool_call_id": "call-extra",
            "name": "shell",
            "content": "/repo",
        },
    )

    with pytest.raises(AgenticPrefixError, match=r"ambiguous.*multiple tool calls"):
        agentic_generation_prefixes(
            _candidate(row), row, executable_environment=False
        )


def test_prefix_construction_rejects_source_bytes_not_bound_by_the_candidate() -> None:
    source = _trajectory()
    candidate = _candidate(source)
    changed = deepcopy(source)
    changed_messages = changed["messages"]
    assert isinstance(changed_messages, list)
    result = changed_messages[2]
    assert isinstance(result, dict)
    result["content"] = "different recorded output"

    with pytest.raises(AgenticPrefixError, match=r"candidate.*source trajectory"):
        agentic_generation_prefixes(
            candidate, changed, executable_environment=False
        )


@pytest.mark.parametrize(
    "arguments",
    ["{", "[]", ["ls"], 7, None],
)
def test_canonical_tool_call_rejects_malformed_arguments(arguments: object) -> None:
    with pytest.raises(AgenticPrefixError, match="arguments"):
        canonical_tool_call(_call("1", "shell", arguments))


def test_canonical_tool_call_does_not_mutate_object_arguments() -> None:
    call = _call("1", "shell", {"timeout": 10, "cmd": "ls"})
    before = json.loads(json.dumps(call))

    canonical_tool_call(call)

    assert call == before
