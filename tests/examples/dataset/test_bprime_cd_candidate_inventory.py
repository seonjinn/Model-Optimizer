# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

import gc
import hashlib
import importlib.util
import json
import sys
import tracemalloc
from dataclasses import replace
from pathlib import Path

import pytest

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
    is_ptv2 = split == "math"
    return module.SourceIdentity(
        repository_id=("nvidia/Nemotron-Post-Training-Dataset-v2" if is_ptv2 else "fixture/source"),
        configuration="default",
        split=split,
        revision=("5c89e01dd720ae0f4058445ed49c5fb68a03c76e" if is_ptv2 else "b" * 40),
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
    storage_dir: Path | None = None,
):
    return module.build_candidate_inventory(
        inventory,
        tokenizer=_Tokenizer(),
        tokenizer_sha256="f" * 64,
        baseline_exclusion=module.make_exclusion_receipt("baseline", historical),
        held_out_exclusion=module.make_exclusion_receipt("held-out", held_out),
        storage_dir=storage_dir or inventory.staged_root / "candidate-storage",
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


def test_inventory_binds_exclusion_receipts_and_pinned_ptv2_family(tmp_path: Path) -> None:
    module = _load_module()
    excluded = [{"role": "user", "content": "historical"}]
    admitted = [{"role": "user", "content": "admitted"}]
    path = _write_rows(
        tmp_path,
        "math",
        [
            {"messages": excluded, "language": "en"},
            {"messages": admitted, "language": "en"},
        ],
    )
    source = _source(module, split="math", path=path)
    historical = {_canonical_uuid(excluded)}

    candidates = module.build_candidate_inventory(
        _inventory(module, tmp_path, (source,)),
        tokenizer=_Tokenizer(),
        tokenizer_sha256="f" * 64,
        baseline_exclusion=module.make_exclusion_receipt("baseline", historical),
        held_out_exclusion=module.make_exclusion_receipt("held-out", set()),
        storage_dir=tmp_path / "candidate-storage",
    )

    assert candidates.baseline_exclusion.receipt_sha256 != "0" * 64
    assert candidates.baseline_exclusion.prompt_id_count == 1
    assert candidates.baseline_exclusion.excluded_candidate_count == 1
    assert candidates.held_out_exclusion.receipt_sha256 != "0" * 64
    assert candidates.held_out_exclusion.prompt_id_count == 0
    assert candidates.held_out_exclusion.excluded_candidate_count == 0
    assert candidates.ptv2_revision == "5c89e01dd720ae0f4058445ed49c5fb68a03c76e"
    assert candidates.rows[0].source_family == "ptv2"


def test_inventory_rejects_receipt_content_mismatch_and_revision_spoof(tmp_path: Path) -> None:
    module = _load_module()
    path = _write_rows(
        tmp_path,
        "first",
        [{"messages": [{"role": "user", "content": "x"}], "language": "en"}],
    )
    source = replace(
        _source(module, split="first", path=path),
        revision="5c89e01dd720ae0f4058445ed49c5fb68a03c76e",
    )
    baseline = module.make_exclusion_receipt("baseline", {"1" * 64})
    mismatched = replace(baseline, prompt_ids=("2" * 64,))

    with pytest.raises(ValueError, match="receipt content"):
        module.build_candidate_inventory(
            _inventory(module, tmp_path, (source,)),
            tokenizer=_Tokenizer(),
            tokenizer_sha256="f" * 64,
            baseline_exclusion=mismatched,
            held_out_exclusion=module.make_exclusion_receipt("held-out", set()),
        )

    candidates = module.build_candidate_inventory(
        _inventory(module, tmp_path, (source,)),
        tokenizer=_Tokenizer(),
        tokenizer_sha256="f" * 64,
        baseline_exclusion=module.make_exclusion_receipt("baseline", set()),
        held_out_exclusion=module.make_exclusion_receipt("held-out", set()),
        storage_dir=tmp_path / "spoof-storage",
    )
    assert candidates.rows[0].source_family == "ptv3"


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


def test_uuid_collision_is_fatal(tmp_path: Path, monkeypatch) -> None:
    module = _load_module()
    path = _write_rows(
        tmp_path,
        "first",
        [
            {"messages": [{"role": "user", "content": "one"}], "language": "en"},
            {"messages": [{"role": "user", "content": "two"}], "language": "en"},
        ],
    )
    source = _source(module, split="first", path=path)
    original_sha256 = module.sha256_bytes
    monkeypatch.setattr(
        module,
        "sha256_bytes",
        lambda value: "c" * 64 if value.startswith(b'{"messages"') else original_sha256(value),
    )

    with pytest.raises(module.UUIDCollisionError, match="UUID collision"):
        _build(module, _inventory(module, tmp_path, (source,)))


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
    assert candidates.rows[0].prompt_uuid == _canonical_uuid(messages, tools)
    assert json.loads(candidates.rows[0].canonical_bytes) == {
        "messages": messages,
        "tools": tools,
    }
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


@pytest.mark.parametrize(
    ("repository_id", "split", "expected"),
    [
        ("nvidia/Nemotron-Math-v2", "low", "en"),
        ("nvidia/Nemotron-Science-v1", "RQA", "en"),
        ("nvidia/Nemotron-SFT-Instruction-Following-Chat-v2", "reasoning_off", "en"),
        ("nvidia/Nemotron-SFT-Multilingual-v1", "stem_zh", "zh"),
    ],
)
def test_missing_language_uses_only_approved_source_split_mapping(
    repository_id: str, split: str, expected: str
) -> None:
    module = _load_module()
    source = module.SourceIdentity(
        repository_id=repository_id,
        configuration="default",
        split=split,
        revision="a" * 40,
        license_expression="Apache-2.0",
        approved_use=True,
        cell="math",
        lane="target-synth",
        files=(),
    )

    assert module._normalize_language(None, source) == expected


@pytest.mark.parametrize("split", ["low", "RQA", "reasoning_off", "xy"])
def test_missing_language_rejects_unmapped_source_suffix(split: str) -> None:
    module = _load_module()
    source = module.SourceIdentity(
        repository_id="fixture/unknown",
        configuration="default",
        split=split,
        revision="a" * 40,
        license_expression="Apache-2.0",
        approved_use=True,
        cell="math",
        lane="target-synth",
        files=(),
    )

    with pytest.raises(ValueError, match="invalid_language"):
        module._normalize_language(None, source)


def _write_many_rows(root: Path, split: str, count: int) -> Path:
    directory = root / split
    directory.mkdir(parents=True)
    path = directory / f"{split}.jsonl"
    with path.open("w", encoding="utf-8") as stream:
        for index in range(count):
            stream.write(
                json.dumps(
                    {
                        "messages": [{"role": "user", "content": f"prompt-{index:06d}"}],
                        "language": "en",
                    }
                )
                + "\n"
            )
    return path


def _measured_build(module, root: Path, count: int):
    path = _write_many_rows(root, "first", count)
    source = _source(module, split="first", path=path)
    tracemalloc.start()
    inventory = _build(
        module,
        _inventory(module, root, (source,)),
        storage_dir=root / "candidate-storage",
    )
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return inventory, peak


def test_candidate_rows_are_disk_backed_and_memory_bounded_at_scale(tmp_path: Path) -> None:
    module = _load_module()
    small, small_peak = _measured_build(module, tmp_path / "small", 100)
    large, large_peak = _measured_build(module, tmp_path / "large", 4_000)

    assert isinstance(large.rows, module.DiskBackedCandidateRows)
    assert large.rows.resident_row_count == 0
    assert len(large.rows) == 4_000
    assert large_peak < small_peak + 6_000_000
    digest = hashlib.sha256()
    for chunk in module.iter_candidate_inventory_bytes(large):
        digest.update(chunk)
    assert digest.hexdigest() == large.inventory_sha256
    expected_prompt_uuids = sorted(
        _canonical_uuid([{"role": "user", "content": f"prompt-{index:06d}"}])
        for index in range(4_000)
    )
    assert [row.prompt_uuid for row in large.rows] == expected_prompt_uuids


def test_malformed_jsonl_record_is_quarantined_without_aborting(tmp_path: Path) -> None:
    module = _load_module()
    directory = tmp_path / "first"
    directory.mkdir()
    path = directory / "first.jsonl"
    path.write_text('not-json\n{"messages":[{"role":"user","content":"valid"}],"language":"en"}\n')
    source = _source(module, split="first", path=path)

    candidates = _build(module, _inventory(module, tmp_path, (source,)))

    assert len(candidates.rows) == 1
    assert candidates.quarantine_counts == {"invalid_row": 1}


def test_invalid_utf8_jsonl_record_is_quarantined_without_aborting(tmp_path: Path) -> None:
    module = _load_module()
    directory = tmp_path / "first"
    directory.mkdir()
    path = directory / "first.jsonl"
    path.write_bytes(b'\xff\n{"messages":[{"role":"user","content":"valid"}],"language":"en"}\n')
    source = _source(module, split="first", path=path)

    candidates = _build(module, _inventory(module, tmp_path, (source,)))

    assert len(candidates.rows) == 1
    assert candidates.rows[0].canonical_bytes
    assert candidates.quarantine_counts == {"invalid_row": 1}


def test_malformed_parquet_raw_json_is_quarantined_without_aborting(tmp_path: Path) -> None:
    module = _load_module()
    import pyarrow as pa  # pyright: ignore[reportMissingImports]
    import pyarrow.parquet as pq  # pyright: ignore[reportMissingImports]

    directory = tmp_path / "first"
    directory.mkdir()
    path = directory / "first.parquet"
    pq.write_table(
        pa.table(
            {
                "raw_json": [
                    "not-json",
                    json.dumps(
                        {
                            "messages": [{"role": "user", "content": "valid"}],
                            "language": "en",
                        }
                    ),
                ]
            }
        ),
        path,
    )
    source = _source(module, split="first", path=path)

    candidates = _build(module, _inventory(module, tmp_path, (source,)))

    assert len(candidates.rows) == 1
    assert candidates.quarantine_counts == {"invalid_row": 1}


def test_candidate_inventory_storage_lives_until_explicit_close(tmp_path: Path) -> None:
    module = _load_module()
    path = _write_rows(
        tmp_path,
        "first",
        [{"messages": [{"role": "user", "content": "available"}], "language": "en"}],
    )
    source = _source(module, split="first", path=path)
    candidates = _build(module, _inventory(module, tmp_path, (source,)))
    assert isinstance(candidates.rows, module.DiskBackedCandidateRows)
    storage_root = candidates.rows.storage_path.parent

    assert storage_root.is_dir()
    assert candidates.rows[0].canonical_bytes
    assert module.candidate_inventory_bytes(candidates)

    candidates.close()
    candidates.close()
    assert not storage_root.exists()
    with pytest.raises(RuntimeError, match="closed"):
        _ = candidates.rows[0]


def test_candidate_inventory_context_manager_cleans_owned_storage(tmp_path: Path) -> None:
    module = _load_module()
    path = _write_rows(
        tmp_path,
        "first",
        [{"messages": [{"role": "user", "content": "context"}], "language": "en"}],
    )
    source = _source(module, split="first", path=path)

    with _build(module, _inventory(module, tmp_path, (source,))) as candidates:
        assert isinstance(candidates.rows, module.DiskBackedCandidateRows)
        storage_root = candidates.rows.storage_path.parent
        assert storage_root.is_dir()
        assert list(candidates.rows)

    assert not storage_root.exists()


def test_candidate_inventory_finalizer_cleans_abandoned_storage(tmp_path: Path) -> None:
    module = _load_module()
    path = _write_rows(
        tmp_path,
        "first",
        [{"messages": [{"role": "user", "content": "abandoned"}], "language": "en"}],
    )
    source = _source(module, split="first", path=path)

    candidates = _build(module, _inventory(module, tmp_path, (source,)))
    assert isinstance(candidates.rows, module.DiskBackedCandidateRows)
    storage_root = candidates.rows.storage_path.parent
    del candidates
    gc.collect()

    assert not storage_root.exists()
