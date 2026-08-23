# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
MODULE_PATH = REPO_ROOT / "examples/dataset/build_specdec_inventory.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("candidate_inventory", MODULE_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    sys.path.insert(0, str(MODULE_PATH.parent))
    try:
        spec.loader.exec_module(module)
    finally:
        sys.path.pop(0)
    return module


class _Tokenizer:
    def apply_chat_template(self, messages, **kwargs):
        del kwargs
        count = sum(len(str(message.get("content", ""))) for message in messages)
        return {"input_ids": list(range(count))}


def _canonical_uuid(messages: list[dict], tools: list[dict] | None = None) -> str:
    payload = json.dumps(
        {"messages": messages, "tools": tools or []},
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode()
    return hashlib.sha256(payload).hexdigest()


def _source(module, *, split: str, path: Path, lane: str = "target-synth"):
    return module.SourceIdentity(
        repository_id="fixture/source",
        configuration="default",
        split=split,
        revision=("a" if split == "first" else "b") * 40,
        license_expression="Apache-2.0",
        approved_use=True,
        cell="math" if lane == "target-synth" else "swe-agentic-tool",
        lane=lane,
        files=(module.SourceFile(path.name, path.stat().st_size, module.sha256_file(path)),),
    )


def _inventory(module, tmp_path: Path, sources: tuple):
    for source in sources:
        for descriptor in source.files:
            destination = (
                tmp_path / "sources" / source.repository_id / source.revision / descriptor.path
            )
            destination.parent.mkdir(parents=True, exist_ok=True)
            source_path = tmp_path / source.split / descriptor.path
            destination.write_bytes(source_path.read_bytes())
    canonical = b'{"fixture":true}\n'
    return module.SourceInventory(
        schema_version=1,
        name="fixture",
        sources=sources,
        manifest_sha256=hashlib.sha256(canonical).hexdigest(),
        canonical_manifest=canonical,
        raw_counts={source.cell: 1 for source in sources},
        staged_root=tmp_path,
    )


def _write_rows(root: Path, split: str, rows: list[dict]) -> Path:
    directory = root / split
    directory.mkdir(parents=True)
    path = directory / f"{split}.jsonl"
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    return path


def _build(
    module,
    inventory,
    *,
    historical: frozenset[str] | set[str] = frozenset(),
    held_out: frozenset[str] | set[str] = frozenset(),
):
    return module.build_candidate_inventory(
        inventory,
        tokenizer=_Tokenizer(),
        tokenizer_sha256="f" * 64,
        historical_prompt_ids=set(historical),
        held_out_prompt_ids=set(held_out),
    )


def test_exclusions_and_cross_source_duplicates_precede_capacity(tmp_path: Path) -> None:
    module = _load_module()
    historical = [{"role": "user", "content": "historical"}]
    held_out = [{"role": "user", "content": "held-out"}]
    admitted = [{"role": "user", "content": "admitted"}]
    first = _write_rows(
        tmp_path,
        "first",
        [
            {"messages": historical, "language": "English"},
            {"messages": admitted, "language": "EN_us"},
        ],
    )
    second = _write_rows(
        tmp_path,
        "second",
        [
            {"messages": held_out, "language": "en-US"},
            {"messages": admitted, "language": "en"},
        ],
    )
    sources = (
        _source(module, split="first", path=first),
        _source(module, split="second", path=second),
    )

    candidates = _build(
        module,
        _inventory(module, tmp_path, sources),
        historical={_canonical_uuid(historical)},
        held_out={_canonical_uuid(historical), _canonical_uuid(held_out)},
    )

    cell = module.CandidateCell("math", "target-synth", "en", "le4k")
    assert len(candidates.rows) == 1
    assert candidates.rows[0].language == "en"
    assert candidates.capacity == {cell: 1}
    assert candidates.quarantine_counts == {
        "duplicate_prompt_uuid": 1,
        "heldout_exclusion": 1,
        "historical_exclusion": 1,
    }


def test_target_completion_is_stripped_before_uuid_and_full_context_bucket(
    tmp_path: Path,
) -> None:
    module = _load_module()
    prompt = [{"role": "user", "content": "x" * 4_097}]
    path = _write_rows(
        tmp_path,
        "first",
        [
            {
                "messages": [*prompt, {"role": "assistant", "content": "source answer"}],
                "source_completion": "source answer",
                "language": "en",
            }
        ],
    )
    source = _source(module, split="first", path=path)

    candidates = _build(module, _inventory(module, tmp_path, (source,)))

    row = candidates.rows[0]
    assert row.prompt_uuid == _canonical_uuid(prompt)
    assert json.loads(row.canonical_bytes)["messages"] == prompt
    assert row.full_token_count == 4_097
    assert row.context_bucket == "4k_16k"
    assert row.lane == "target-synth"
    assert row.arm_domain == "math"


def test_uuid_collision_is_quarantined_with_stable_code(tmp_path: Path, monkeypatch) -> None:
    module = _load_module()
    path = _write_rows(
        tmp_path,
        "first",
        [
            {"messages": [{"role": "user", "content": "one"}]},
            {"messages": [{"role": "user", "content": "two"}]},
        ],
    )
    source = _source(module, split="first", path=path)
    original_sha256 = module.sha256_bytes
    monkeypatch.setattr(
        module,
        "sha256_bytes",
        lambda value: "c" * 64 if value.startswith(b'{"messages"') else original_sha256(value),
    )

    candidates = _build(module, _inventory(module, tmp_path, (source,)))

    assert len(candidates.rows) == 1
    assert candidates.quarantine_counts == {"prompt_uuid_collision": 1}


def test_replay_validation_precedes_capacity_counting(tmp_path: Path) -> None:
    module = _load_module()
    tools = [{"type": "function", "function": {"name": "shell"}}]
    messages = [
        {"role": "user", "content": "run tests"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "call-1",
                    "type": "function",
                    "function": {"name": "shell", "arguments": '{"cmd":"pytest"}'},
                }
            ],
        },
        {"role": "tool", "tool_call_id": "call-1", "name": "shell", "content": "ok"},
        {"role": "assistant", "content": "done"},
    ]
    path = _write_rows(
        tmp_path,
        "first",
        [
            {"messages": messages, "tools": tools, "language": "English"},
            {"messages": messages[:1], "tools": tools, "language": "en"},
        ],
    )
    source = _source(module, split="first", path=path, lane="generic-tool-replay")

    candidates = _build(module, _inventory(module, tmp_path, (source,)))

    cell = module.CandidateCell("swe-agentic-tool", "generic-tool-replay", "en", "le4k")
    assert candidates.capacity == {cell: 1}
    assert candidates.rows[0].replay_valid is True
    assert candidates.quarantine_counts == {"no_assistant_tool_call": 1}


def test_candidate_inventory_is_source_order_invariant(tmp_path: Path) -> None:
    module = _load_module()
    first = _write_rows(
        tmp_path,
        "first",
        [{"messages": [{"role": "user", "content": "first"}], "language": "EN"}],
    )
    second = _write_rows(
        tmp_path,
        "second",
        [{"messages": [{"role": "user", "content": "second"}], "language": "english"}],
    )
    source_a = _source(module, split="first", path=first)
    source_b = _source(module, split="second", path=second)

    forward = _build(module, _inventory(module, tmp_path, (source_a, source_b)))
    reverse = _build(module, _inventory(module, tmp_path, (source_b, source_a)))

    assert forward == reverse
    assert forward.inventory_sha256 == reverse.inventory_sha256
    assert module.candidate_inventory_bytes(forward) == module.candidate_inventory_bytes(reverse)
    assert [row.prompt_uuid for row in forward.rows] == sorted(
        row.prompt_uuid for row in forward.rows
    )
