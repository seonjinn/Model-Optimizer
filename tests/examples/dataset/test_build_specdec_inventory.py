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

import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
MODULE_PATH = REPO_ROOT / "examples/dataset/build_specdec_inventory.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("build_specdec_inventory", MODULE_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    sys.path.insert(0, str(MODULE_PATH.parent))
    try:
        spec.loader.exec_module(module)
    finally:
        sys.path.pop(0)
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


def test_inventory_reads_explicit_raw_json_parquet_shards(tmp_path: Path) -> None:
    module = _load_module()
    import pyarrow as pa  # pyright: ignore[reportMissingImports]
    import pyarrow.parquet as pq  # pyright: ignore[reportMissingImports]

    path = tmp_path / "raw.parquet"
    pq.write_table(
        pa.table(
            {
                "raw_json": [
                    json.dumps(
                        {
                            "prompt_id": "raw-1",
                            "messages": [
                                {"role": "user", "content": "q"},
                                {"role": "assistant", "content": "a"},
                            ],
                        }
                    )
                ]
            }
        ),
        path,
        compression="zstd",
    )

    assert next(iter(module._iter_rows(path)))["prompt_id"] == "raw-1"


def test_candidate_inventory_verifies_every_source_before_parsing_rows(tmp_path: Path) -> None:
    module = _load_module()
    valid = tmp_path / "valid.jsonl"
    valid.write_text("not-json\n")
    malformed = tmp_path / "malformed.jsonl"
    malformed.write_text('{"messages":[{"role":"user","content":"valid"}]}\n')
    first_staged = tmp_path / "sources/fixture/source" / ("a" * 40) / valid.name
    first_staged.parent.mkdir(parents=True)
    first_staged.write_bytes(valid.read_bytes())
    second_staged = tmp_path / "sources/fixture/source" / ("b" * 40) / malformed.name
    second_staged.parent.mkdir(parents=True)
    second_staged.write_bytes(malformed.read_bytes())
    canonical_manifest = b"{}\n"
    inventory = module.SourceInventory(
        schema_version=1,
        name="fixture",
        sources=(
            module.SourceIdentity(
                repository_id="fixture/source",
                configuration="default",
                split="valid",
                revision="a" * 40,
                license_expression="Apache-2.0",
                approved_use=True,
                cell="math",
                lane="target-synth",
                files=(
                    module.SourceFile(
                        "valid.jsonl", valid.stat().st_size, module.sha256_file(valid)
                    ),
                ),
            ),
            module.SourceIdentity(
                repository_id="fixture/source",
                configuration="default",
                split="malformed",
                revision="b" * 40,
                license_expression="Apache-2.0",
                approved_use=True,
                cell="math",
                lane="target-synth",
                files=(
                    module.SourceFile(
                        "malformed.jsonl",
                        malformed.stat().st_size + 1,
                        module.sha256_file(malformed),
                    ),
                ),
            ),
        ),
        manifest_sha256=hashlib.sha256(canonical_manifest).hexdigest(),
        canonical_manifest=canonical_manifest,
        raw_counts={"math": 2},
        staged_root=tmp_path,
    )

    with pytest.raises(ValueError, match="source file identity mismatch"):
        module.build_candidate_inventory(
            inventory,
            tokenizer=FakeTokenizer(),
            tokenizer_sha256="f" * 64,
            baseline_exclusion=module.make_exclusion_receipt("baseline", set()),
            held_out_exclusion=module.make_exclusion_receipt("held-out", set()),
        )


@pytest.mark.parametrize(
    ("token_count", "bucket"),
    [
        (4_096, "le4k"),
        (4_097, "4k_16k"),
        (16_384, "4k_16k"),
        (16_385, "16k_32k"),
        (32_768, "16k_32k"),
    ],
)
def test_context_bucket_exact_boundaries(token_count: int, bucket: str) -> None:
    module = _load_module()

    assert module._context_bucket(token_count) == bucket


def test_context_bucket_rejects_above_32k() -> None:
    module = _load_module()

    with pytest.raises(ValueError, match="32K inventory limit"):
        module._context_bucket(32_769)
