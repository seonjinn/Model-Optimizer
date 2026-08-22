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

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

import pytest

REPO_ROOT = Path(__file__).resolve().parents[4]
QUERY_PATH = REPO_ROOT / "tools/launcher/common/query.py"


def _load_query_module() -> ModuleType:
    class _Completions:
        @staticmethod
        def create(**_kwargs: object) -> object:
            message = type("Message", (), {"content": "ok"})()
            choice = type("Choice", (), {"message": message})()
            return type("Completion", (), {"choices": [choice]})()

    class _OpenAI:
        def __init__(self, **_kwargs: object) -> None:
            self.chat = type("Chat", (), {"completions": _Completions()})()

    openai_stub = ModuleType("openai")
    openai_stub.OpenAI = _OpenAI
    datasets_stub = ModuleType("datasets")
    datasets_stub.load_dataset = lambda *_args, **_kwargs: None
    spec = importlib.util.spec_from_file_location("modelopt_query_for_test", QUERY_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    with (
        patch.dict(sys.modules, {"datasets": datasets_stub, "openai": openai_stub}),
        patch.object(sys, "argv", ["query.py", "http://localhost", "target"]),
    ):
        spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def query_module() -> ModuleType:
    return _load_query_module()


@pytest.mark.parametrize(
    ("mode", "source_value", "expected"),
    [
        ("on", False, True),
        ("off", True, False),
        ("source", True, True),
        ("source", False, False),
        ("source", None, True),
    ],
)
def test_resolve_thinking_mode_is_explicit(
    query_module: ModuleType,
    mode: str,
    source_value: bool | None,
    expected: bool,
) -> None:
    row = {} if source_value is None else {"enable_thinking": source_value}
    assert query_module.resolve_thinking_mode(mode, row) is expected


def test_prepare_generation_messages_never_depends_on_shard_id(
    query_module: ModuleType,
) -> None:
    row = {"messages": [{"role": "user", "content": "Solve this"}]}

    shard_zero = query_module.prepare_generation_messages(row, "on", shard_id=0)
    shard_one = query_module.prepare_generation_messages(row, "on", shard_id=1)

    assert shard_zero == shard_one
    assert shard_zero[0]["content"] == "Solve this"


def test_thinking_off_adds_no_think_without_mutating_source(
    query_module: ModuleType,
) -> None:
    row = {"messages": [{"role": "user", "content": "Solve this"}]}

    prepared = query_module.prepare_generation_messages(row, "off", shard_id=4)

    assert prepared[0]["content"] == "Solve this /no_think"
    assert row["messages"][0]["content"] == "Solve this"


def test_shard_metadata_records_reproducibility_inputs(
    query_module: ModuleType,
    tmp_path: Path,
) -> None:
    output = tmp_path / "shard_3.jsonl"
    output.write_text('{"messages": []}\n', encoding="utf-8")

    metadata = query_module.build_shard_metadata(
        output_path=output,
        shard_id=3,
        num_shards=8,
        thinking_mode="on",
        source_id="ptv2-balanced-v1",
        target_revision="0123456789abcdef",
        temperature=0.0,
        max_tokens=4096,
        max_total_length=32768,
    )

    assert metadata["schema_version"] == 1
    assert metadata["thinking_mode"] == "on"
    assert metadata["source_id"] == "ptv2-balanced-v1"
    assert metadata["target_revision"] == "0123456789abcdef"
    assert metadata["output_sha256"] == query_module.sha256_file(output)
    json.dumps(metadata, sort_keys=True)


def test_local_parquet_directory_resolves_all_shards_in_stable_order(
    query_module: ModuleType, tmp_path: Path
) -> None:
    """A materialized corpus directory is loaded as one deterministic dataset."""
    (tmp_path / "train-00002.parquet").touch()
    (tmp_path / "train-00001.parquet").touch()
    (tmp_path / "MANIFEST.json").write_text("{}\n", encoding="utf-8")

    dataset_format, data_files = query_module.resolve_local_dataset(tmp_path)

    assert dataset_format == "parquet"
    assert data_files == [
        str(tmp_path / "train-00001.parquet"),
        str(tmp_path / "train-00002.parquet"),
    ]


def test_local_manifest_resolves_only_listed_verified_shards(
    query_module: ModuleType, tmp_path: Path
) -> None:
    selected = tmp_path / "selected.jsonl"
    selected.write_text('{"messages": []}\n')
    (tmp_path / "unselected.jsonl").write_text('{"messages": []}\n')
    manifest = tmp_path / "PROMPTS.json"
    manifest.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "format": "json",
                "file_count": 1,
                "row_count": 1,
                "files": [
                    {
                        "path": selected.name,
                        "bytes": selected.stat().st_size,
                        "sha256": query_module.sha256_file(selected),
                    }
                ],
            }
        )
        + "\n"
    )

    dataset_format, data_files = query_module.resolve_local_dataset(manifest)

    assert dataset_format == "json"
    assert data_files == [str(selected)]


def test_target_synthesis_rejects_tool_trajectory(query_module: ModuleType) -> None:
    row = {
        "tools": [{"type": "function", "function": {"name": "shell"}}],
        "messages": [{"role": "user", "content": "run it"}],
    }

    with pytest.raises(ValueError, match="tool trajectory"):
        query_module.prepare_generation_messages(
            row, "on", shard_id=0, reject_tool_trajectories=True
        )


def test_synthesis_preserves_generated_turns_and_records_exact_api_tokens(
    query_module: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    class FakeLLM:
        def __init__(self) -> None:
            self.calls = 0
            self.last_completion_tokens: int | None = None

        def generate(self, _messages: list[dict], verbose: bool = False) -> dict:
            del verbose
            self.calls += 1
            self.last_completion_tokens = self.calls + 2
            return {
                "role": "assistant",
                "content": f"<think>reason-{self.calls}</think>answer-{self.calls}",
            }

    fake = FakeLLM()
    monkeypatch.setattr(query_module, "llm", fake, raising=False)
    monkeypatch.setattr(
        query_module,
        "args",
        SimpleNamespace(
            thinking_mode="on",
            reject_tool_trajectories=True,
            max_total_length=None,
            max_tokens=None,
            record_assistant_tokens=True,
        ),
        raising=False,
    )

    result = query_module.synthesize(
        {
            "messages": [
                {"role": "user", "content": "first"},
                {"role": "assistant", "content": "source answer"},
                {"role": "user", "content": "second"},
            ]
        }
    )

    assistants = [message for message in result["messages"] if message["role"] == "assistant"]
    assert [message["content"] for message in assistants] == [
        "<think>reason-1</think>answer-1",
        "<think>reason-2</think>answer-2",
    ]
    assert result["_synthesis_assistant_tokens"] == 7


def test_completed_shard_is_rehashed_before_resume(
    query_module: ModuleType, tmp_path: Path
) -> None:
    output = tmp_path / "shard.jsonl"
    output.write_text('{"messages": []}\n')
    metadata = query_module.build_shard_metadata(
        output_path=output,
        shard_id=0,
        num_shards=4,
        thinking_mode="off",
        source_id="source",
        target_revision="revision",
        temperature=0.0,
        max_tokens=64,
        max_total_length=128,
    )
    metadata_path = output.with_suffix(".jsonl.metadata.json")
    metadata_path.write_text(json.dumps(metadata) + "\n")
    done = output.with_suffix(".jsonl.done")
    done.write_text("done\n")

    query_module.verify_completed_shard(
        output,
        metadata_path,
        done,
        expected={
            "shard_id": 0,
            "num_shards": 4,
            "thinking_mode": "off",
            "source_id": "source",
            "target_revision": "revision",
            "temperature": 0.0,
            "max_tokens": 64,
            "max_total_length": 128,
        },
    )

    output.write_text("corrupt\n")
    with pytest.raises(ValueError, match="completed shard identity mismatch"):
        query_module.verify_completed_shard(
            output,
            metadata_path,
            done,
            expected={"shard_id": 0},
        )


def test_incomplete_shard_is_recovered_before_retry(
    query_module: ModuleType, tmp_path: Path
) -> None:
    """A crash before the commit marker leaves no state that blocks a retry."""
    output = tmp_path / "shard.jsonl"
    metadata = output.with_suffix(".jsonl.metadata.json")
    done = output.with_suffix(".jsonl.done")
    output.write_text("partial\n")
    metadata.write_text("{}\n")
    stale = tmp_path / ".shard.jsonl.partial-123"
    stale.write_text("partial\n")

    assert (
        query_module.recover_or_verify_shard(output, metadata, done, expected={"shard_id": 0})
        is False
    )

    assert not output.exists()
    assert not metadata.exists()
    assert not done.exists()
    assert not stale.exists()


def test_done_marker_with_missing_payload_fails_closed(
    query_module: ModuleType, tmp_path: Path
) -> None:
    """The durable commit marker is never silently downgraded to retryable state."""
    output = tmp_path / "shard.jsonl"
    metadata = output.with_suffix(".jsonl.metadata.json")
    done = output.with_suffix(".jsonl.done")
    done.write_text("done\n")

    with pytest.raises(ValueError, match="incomplete synthesis shard state"):
        query_module.recover_or_verify_shard(output, metadata, done, expected={"shard_id": 0})
