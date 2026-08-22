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
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
MODULE_PATH = REPO_ROOT / "examples/dataset/build_specdec_inventory.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("build_specdec_inventory", MODULE_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class FakeTokenizer:
    def apply_chat_template(self, messages, **kwargs):
        assert kwargs["return_assistant_tokens_mask"] is True
        assert kwargs["return_dict"] is True
        tools = kwargs.get("tools")
        input_ids: list[int] = []
        mask: list[int] = []
        for message in messages:
            tokens = [len(str(message.get("content", ""))) + 1]
            if message.get("tool_calls"):
                tokens.append(99)
            input_ids.extend(tokens)
            mask.extend([int(message["role"] == "assistant")] * len(tokens))
        if tools:
            input_ids.insert(0, 77)
            mask.insert(0, 0)
        return {"input_ids": input_ids, "assistant_masks": mask}


def _write_bound_source(tmp_path: Path, rows: list[dict]) -> Path:
    module = _load_module()
    data = tmp_path / "source.jsonl"
    data.write_text("".join(json.dumps(row) + "\n" for row in rows))
    manifest = tmp_path / "source-manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "files": [
                    {
                        "path": data.name,
                        "bytes": data.stat().st_size,
                        "sha256": module.sha256_file(data),
                    }
                ],
            }
        )
        + "\n"
    )
    return manifest


def test_inventory_pretokenizes_synthesis_and_binds_source_hashes(tmp_path: Path) -> None:
    module = _load_module()
    manifest = _write_bound_source(
        tmp_path,
        [
            {
                "prompt_id": "p1",
                "messages": [
                    {"role": "user", "content": "question"},
                    {"role": "assistant", "content": "answer"},
                ],
                "_synthesis_assistant_tokens": 1,
            }
        ],
    )
    spec = module.InventorySource(
        source_id="nvidia/Nemotron-Post-Training-Dataset-v2",
        source_revision="a" * 40,
        license="ODC-BY-1.0",
        pool="ptv2",
        category="math",
        response_source="target-synth",
        tool_lane="none",
        manifest_path=manifest,
    )

    rows = module.build_inventory_rows(
        [spec], tokenizer=FakeTokenizer(), tokenizer_sha256="f" * 64, training_seq_len=4096
    )

    assert rows[0]["assistant_tokens"] == 1
    assert rows[0]["full_token_count"] == 2
    assert rows[0]["context_bucket"] == "le4k"
    assert rows[0]["source_manifest_sha256"] == module.sha256_file(manifest)
    assert rows[0]["source_file_sha256"]
    assert rows[0]["response_source"] == "target-synth"


def test_inventory_preserves_tool_lane_and_exact_assistant_mask(tmp_path: Path) -> None:
    module = _load_module()
    manifest = _write_bound_source(
        tmp_path,
        [
            {
                "prompt_id": "tool-1",
                "tools": [{"type": "function", "function": {"name": "shell"}}],
                "messages": [
                    {"role": "user", "content": "run"},
                    {"role": "assistant", "content": "", "tool_calls": [{"id": "c1"}]},
                    {"role": "tool", "content": "ok", "tool_call_id": "c1"},
                    {"role": "assistant", "content": "done"},
                ],
            }
        ],
    )
    spec = module.InventorySource(
        source_id="nvidia/Nemotron-SFT-Agentic-v2",
        source_revision="b" * 40,
        license="NVIDIA-Open-Model-License",
        pool="ptv3",
        category="agentic_tool",
        response_source="trace-replay",
        tool_lane="recorded-trace",
        manifest_path=manifest,
    )

    rows = module.build_inventory_rows(
        [spec], tokenizer=FakeTokenizer(), tokenizer_sha256="e" * 64, training_seq_len=4096
    )

    assert rows[0]["tools"][0]["function"]["name"] == "shell"
    assert rows[0]["tool_lane"] == "recorded-trace"
    assert rows[0]["assistant_tokens"] == 3
    assert sum(rows[0]["loss_mask"]) == 3


def test_inventory_rejects_mutated_manifested_source(tmp_path: Path) -> None:
    module = _load_module()
    manifest = _write_bound_source(
        tmp_path,
        [{"prompt_id": "p1", "messages": [{"role": "assistant", "content": "a"}]}],
    )
    (tmp_path / "source.jsonl").write_text("mutated\n")
    spec = module.InventorySource(
        source_id="source",
        source_revision="c" * 40,
        license="license",
        pool="ptv2",
        category="math",
        response_source="target-synth",
        tool_lane="none",
        manifest_path=manifest,
    )

    with pytest.raises(ValueError, match="source file identity mismatch"):
        module.build_inventory_rows(
            [spec], tokenizer=FakeTokenizer(), tokenizer_sha256="d" * 64, training_seq_len=4096
        )


def test_tokenizer_digest_binds_exact_serialization_files(tmp_path: Path) -> None:
    module = _load_module()
    (tmp_path / "tokenizer.json").write_text('{"version":"1"}\n')
    (tmp_path / "tokenizer_config.json").write_text('{"chat_template":"x"}\n')
    (tmp_path / "model.safetensors").write_text("ignored weights\n")

    first = module.tokenizer_snapshot_sha256(tmp_path)
    (tmp_path / "model.safetensors").write_text("changed weights\n")
    assert module.tokenizer_snapshot_sha256(tmp_path) == first
    (tmp_path / "tokenizer.json").write_text('{"version":"2"}\n')
    assert module.tokenizer_snapshot_sha256(tmp_path) != first
