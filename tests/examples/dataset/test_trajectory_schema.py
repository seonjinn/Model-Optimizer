from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
MODULE_PATH = REPO_ROOT / "examples/dataset/trajectory_schema.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("trajectory_schema", MODULE_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _load_conversation_utils():
    module_path = REPO_ROOT / "examples/dataset/conversation_utils.py"
    spec = importlib.util.spec_from_file_location("conversation_utils_for_test", module_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(module_path.parent))
    try:
        spec.loader.exec_module(module)
    finally:
        sys.path.pop(0)
    return module


def _trajectory() -> dict:
    return {
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "shell",
                    "description": "Run a command",
                    "parameters": {"type": "object"},
                },
            }
        ],
        "messages": [
            {"role": "system", "content": "You edit repositories."},
            {"role": "user", "content": "Fix the test."},
            {
                "role": "assistant",
                "content": "",
                "reasoning_content": "I should inspect failures.",
                "tool_calls": [
                    {
                        "id": "call-1",
                        "type": "function",
                        "function": {"name": "shell", "arguments": '{"cmd":"pytest"}'},
                    }
                ],
            },
            {
                "role": "tool",
                "name": "shell",
                "id": "call-1",
                "content": "1 failed",
            },
            {"role": "assistant", "content": "I fixed it."},
        ],
    }


def test_canonicalize_preserves_tool_schema_calls_results_and_reasoning() -> None:
    module = _load_module()

    canonical = module.canonicalize_trajectory(_trajectory(), source_id="swe-v1:train:7")

    assert canonical["source_id"] == "swe-v1:train:7"
    assert canonical["tools"][0]["function"]["name"] == "shell"
    assert canonical["messages"][2]["reasoning_content"] == "I should inspect failures."
    assert canonical["messages"][2]["tool_calls"][0]["id"] == "call-1"
    assert canonical["messages"][3] == {
        "role": "tool",
        "content": "1 failed",
        "name": "shell",
        "tool_call_id": "call-1",
    }


def test_dangling_tool_result_is_rejected_with_source_context() -> None:
    module = _load_module()
    row = _trajectory()
    row["messages"][3]["id"] = "missing"

    with pytest.raises(ValueError, match=r"swe-v1:train:9.*missing"):
        module.canonicalize_trajectory(row, source_id="swe-v1:train:9")


def test_unresolved_tool_call_is_rejected() -> None:
    module = _load_module()
    row = _trajectory()
    del row["messages"][3]

    with pytest.raises(ValueError, match=r"unresolved tool call.*call-1"):
        module.canonicalize_trajectory(row, source_id="swe-v1:train:10")


def test_duplicate_tool_call_id_is_rejected() -> None:
    module = _load_module()
    row = _trajectory()
    row["messages"][2]["tool_calls"].append(
        {
            "id": "call-1",
            "type": "function",
            "function": {"name": "shell", "arguments": "{}"},
        }
    )

    with pytest.raises(ValueError, match=r"duplicate tool call ID.*call-1"):
        module.canonicalize_trajectory(row, source_id="swe-v1:train:11")


def test_undeclared_function_and_malformed_arguments_are_rejected() -> None:
    module = _load_module()
    undeclared = _trajectory()
    undeclared["messages"][2]["tool_calls"][0]["function"]["name"] = "python"
    with pytest.raises(ValueError, match="undeclared function"):
        module.canonicalize_trajectory(undeclared, source_id="trace:12")

    malformed = _trajectory()
    malformed["messages"][2]["tool_calls"][0]["function"]["arguments"] = "{"
    with pytest.raises(ValueError, match="malformed arguments"):
        module.canonicalize_trajectory(malformed, source_id="trace:13")


def test_digest_is_independent_of_input_mapping_order() -> None:
    module = _load_module()
    row = _trajectory()
    reordered = {"messages": row["messages"], "tools": row["tools"]}
    canonical_a = module.canonicalize_trajectory(row, source_id="same")
    canonical_b = module.canonicalize_trajectory(reordered, source_id="same")

    assert module.trajectory_digest(canonical_a) == module.trajectory_digest(canonical_b)


def test_normalize_messages_preserves_top_level_tool_schema() -> None:
    module = _load_module()

    normalized = module.normalize_trajectory_for_dataset(_trajectory(), source_id="agentic-v2:3")

    assert set(normalized) == {"messages", "tools", "source_id", "trajectory_sha256"}
    assert normalized["messages"][3]["tool_call_id"] == "call-1"


def test_conversation_normalizer_routes_tool_trajectories_through_validator() -> None:
    module = _load_conversation_utils()
    row = _trajectory() | {"uuid": "row-17"}

    normalized = module.normalize_messages(row, 17)

    assert normalized["source_id"] == "row-17"
    assert normalized["tools"] == row["tools"]
    assert normalized["messages"][3]["tool_call_id"] == "call-1"
    assert normalized["trajectory_sha256"]


def _multi_call_trajectory() -> dict:
    row = _trajectory()
    row["messages"][2]["tool_calls"].append(
        {
            "id": "call-2",
            "type": "function",
            "function": {"name": "shell", "arguments": '{"cmd":"ruff check"}'},
        }
    )
    row["messages"].insert(
        4,
        {
            "role": "tool",
            "name": "shell",
            "tool_call_id": "call-2",
            "content": "All checks passed",
        },
    )
    return row


def _replay_fixture(mutation: str) -> dict:
    row = _multi_call_trajectory()
    if mutation == "declarations_only":
        row["messages"] = row["messages"][:2] + row["messages"][-1:]
    elif mutation == "undeclared_function":
        row["messages"][2]["tool_calls"][0]["function"]["name"] = "python"
    elif mutation == "duplicate_call_id":
        row["messages"][2]["tool_calls"][1]["id"] = "call-1"
    elif mutation == "orphan_result":
        row["messages"][3]["tool_call_id"] = "missing"
        row["messages"][3].pop("id", None)
    elif mutation == "open_call":
        del row["messages"][3:5]
    elif mutation == "malformed_arguments":
        row["messages"][2]["tool_calls"][0]["function"]["arguments"] = "[]"
    elif mutation == "unsupported_role":
        row["messages"][1]["role"] = "function"
    elif mutation == "reasoning_mode_mismatch":
        row["reasoning_mode"] = "reasoning_off"
    elif mutation == "wrong_result_name":
        row["messages"][3]["name"] = "python"
    else:
        raise AssertionError(f"unknown mutation: {mutation}")
    return row


@pytest.mark.parametrize("lane", ["interactive-swe-replay", "generic-tool-replay"])
def test_validate_trajectory_accepts_multi_call_replay_lanes(lane: str) -> None:
    module = _load_module()

    validation = module.validate_trajectory(
        _multi_call_trajectory(), source_id=f"fixture:{lane}", lane=lane
    )

    assert validation.tool_call_count == 2
    assert validation.canonical["messages"][3]["tool_call_id"] == "call-1"
    assert validation.canonical["messages"][4]["tool_call_id"] == "call-2"


@pytest.mark.parametrize(
    ("mutation", "reason"),
    [
        ("declarations_only", "no_assistant_tool_call"),
        ("undeclared_function", "undeclared_function"),
        ("duplicate_call_id", "duplicate_tool_call_id"),
        ("orphan_result", "orphan_tool_result"),
        ("open_call", "unresolved_tool_call"),
        ("malformed_arguments", "malformed_arguments"),
        ("unsupported_role", "unsupported_role"),
        ("reasoning_mode_mismatch", "reasoning_mode_mismatch"),
        ("wrong_result_name", "tool_result_name_mismatch"),
    ],
)
def test_invalid_replay_is_quarantined(mutation: str, reason: str) -> None:
    module = _load_module()

    assert (
        module.quarantine_reason(
            _replay_fixture(mutation), source_id="fixture:replay", lane="generic-tool-replay"
        )
        == reason
    )


class _BoundaryTokenizer:
    def apply_chat_template(self, messages, **kwargs):
        del kwargs
        return {"input_ids": list(range(1_500 * len(messages)))}


def test_4k_cut_splitting_tool_call_and_results_is_quarantined() -> None:
    module = _load_module()

    reason = module.quarantine_reason(
        _multi_call_trajectory(),
        source_id="fixture:cut",
        lane="interactive-swe-replay",
        tokenizer=_BoundaryTokenizer(),
        training_seq_len=4_096,
    )

    assert reason == "split_tool_transaction"


def test_expected_reasoning_mode_must_match_replay_content() -> None:
    module = _load_module()

    reason = module.quarantine_reason(
        _multi_call_trajectory(),
        source_id="fixture:reasoning-off",
        lane="generic-tool-replay",
        reasoning_mode="reasoning_off",
    )

    assert reason == "reasoning_mode_mismatch"
