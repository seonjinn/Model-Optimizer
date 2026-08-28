from __future__ import annotations

import importlib.util
import json
import sys
from copy import deepcopy
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
MODULE_PATH = REPO_ROOT / "examples/dataset/q30t_generation_prompt_identity.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("q30t_generation_prompt_identity", MODULE_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    sys.path.insert(0, str(MODULE_PATH.parent))
    try:
        spec.loader.exec_module(module)
    finally:
        sys.path.pop(0)
    return module


def _call(call_id: str, name: str, arguments: dict[str, object]) -> dict[str, object]:
    return {
        "id": call_id,
        "type": "function",
        "function": {"name": name, "arguments": json.dumps(arguments)},
    }


def _shell_tool() -> dict[str, object]:
    return {
        "type": "function",
        "function": {
            "name": "shell",
            "parameters": {"type": "object", "properties": {"cmd": {"type": "string"}}},
        },
    }


def _provenance(module, *, response_lane: str = "interactive_swe"):
    return module.PromptProvenance(
        source_repository="OpenHands/OpenHands",
        source_revision="a" * 40,
        source_file_sha256="b" * 64,
        source_split="train",
        source_row_index=17,
        reasoning_mode="reasoning_on",
        response_lane=response_lane,
    )


def test_agentic_prefix_binds_prior_action_and_tool_result() -> None:
    module = _load_module()
    prefix = [
        {"role": "user", "content": "inspect the repo"},
        {"role": "assistant", "tool_calls": [_call("1", "shell", {"cmd": "ls"})]},
        {"role": "tool", "tool_call_id": "1", "name": "shell", "content": "a.py"},
    ]

    identity = module.generation_prompt_identity(prefix, [_shell_tool()], _provenance(module))
    changed = deepcopy(prefix)
    changed[-1]["content"] = "b.py"

    assert identity.prompt_uuid != module.generation_prompt_identity(
        changed, [_shell_tool()], _provenance(module)
    ).prompt_uuid


def test_identity_excludes_generated_response_and_selection_rank() -> None:
    module = _load_module()

    base = module.generation_prompt_identity(
        [{"role": "user", "content": "solve"}], [], _provenance(module)
    )
    row = {"identity": base, "generated_response": "answer", "selection_rank": 7}

    assert row["identity"].prompt_uuid == base.prompt_uuid


def test_identity_strips_storage_metadata_without_changing_prefix() -> None:
    module = _load_module()
    prefix = [{"role": "user", "content": "solve"}]
    stored = [{"role": "user", "content": "solve", "row_index": 3, "cache_path": "/tmp"}]

    assert module.generation_prompt_identity(
        stored, [], _provenance(module)
    ) == module.generation_prompt_identity(prefix, [], _provenance(module))


def test_response_lane_is_provenance_not_generation_arm() -> None:
    module = _load_module()
    messages = [{"role": "user", "content": "solve"}]

    native = module.generation_prompt_identity(messages, [], _provenance(module, response_lane="math"))
    synth = module.generation_prompt_identity(messages, [], _provenance(module, response_lane="math"))
    other_lane = module.generation_prompt_identity(
        messages, [], _provenance(module, response_lane="interactive_swe")
    )

    assert native == synth
    assert native.prompt_uuid != other_lane.prompt_uuid


def test_identity_rejects_unresolved_prior_tool_call() -> None:
    module = _load_module()
    prefix = [
        {"role": "user", "content": "inspect the repo"},
        {"role": "assistant", "tool_calls": [_call("1", "shell", {"cmd": "ls"})]},
    ]

    with pytest.raises(ValueError, match=r"unresolved tool call.*1"):
        module.generation_prompt_identity(prefix, [_shell_tool()], _provenance(module))
