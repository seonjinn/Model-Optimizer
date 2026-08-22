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
import random
import sys
from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[3]
MODULE_PATH = REPO_ROOT / "examples/dataset/build_specdec_study_corpus.py"
CONFIG_PATH = REPO_ROOT / "examples/dataset/qwen3_4b_hybrid_study.yaml"
TOKENIZER_SHA256 = "f" * 64


def _load_module():
    spec = importlib.util.spec_from_file_location("build_specdec_study_corpus", MODULE_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _config() -> dict:
    config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    config["minimum_prompts_per_category"] = 0
    config["minimum_prompts_per_populated_cell"] = 0
    config["tokenizer_sha256"] = TOKENIZER_SHA256
    return config


def _candidates(count_per_category: int = 1000) -> list[dict]:
    config = _config()
    rows = []
    for pool, pool_config in config["pools"].items():
        for category in pool_config["categories"]:
            rows.extend(
                [
                    {
                        "prompt_id": f"{pool}:{category}:{index}",
                        "pool": pool,
                        "category": category,
                        "context_bucket": ["le4k", "4k_16k", "16k_32k"][index % 3],
                        "full_token_count": [2, 5000, 20000][index % 3],
                        "assistant_tokens": 1,
                        "source_id": f"{pool}-{category}",
                        "source_revision": "a" * 40,
                        "license": "Apache-2.0",
                        "source_manifest_sha256": "b" * 64,
                        "source_file_path": f"{pool}-{category}.jsonl",
                        "source_file_sha256": "c" * 64,
                        "response_source": "target-synth",
                        "tool_lane": "none",
                        "tokenizer_sha256": TOKENIZER_SHA256,
                        "input_ids": [10, 11],
                        "loss_mask": [0, 1],
                    }
                    for index in range(count_per_category)
                ]
            )
    return rows


@pytest.mark.parametrize(
    ("arm", "expected_ptv2", "expected_ptv3"),
    [("B", 1000, 0), ("C", 750, 250), ("D", 500, 500)],
)
def test_arm_pool_weights_are_assistant_token_quotas(
    arm: str, expected_ptv2: int, expected_ptv3: int
) -> None:
    module = _load_module()

    selected = module.select_arm(
        _candidates(), config=_config(), arm=arm, target_assistant_tokens=1000
    )
    totals = module.token_totals_by(selected, "pool")

    assert totals.get("ptv2", 0) == expected_ptv2
    assert totals.get("ptv3", 0) == expected_ptv3


def test_selection_is_deterministic_and_input_order_independent() -> None:
    module = _load_module()
    candidates = _candidates(300)
    shuffled = list(candidates)
    random.Random(99).shuffle(shuffled)

    first = module.select_arm(candidates, config=_config(), arm="C", target_assistant_tokens=400)
    second = module.select_arm(shuffled, config=_config(), arm="C", target_assistant_tokens=400)

    assert [row["prompt_id"] for row in first] == [row["prompt_id"] for row in second]


def test_selection_does_not_take_lexical_prefix() -> None:
    module = _load_module()
    selected = module.select_arm(
        _candidates(500), config=_config(), arm="B", target_assistant_tokens=100
    )

    numeric_ids = {int(row["prompt_id"].rsplit(":", 1)[1]) for row in selected}
    assert max(numeric_ids) > 100


def test_held_out_prompt_is_rejected() -> None:
    module = _load_module()
    candidates = _candidates(50)

    with pytest.raises(ValueError, match="held-out contamination"):
        module.select_arm(
            candidates,
            config=_config(),
            arm="B",
            target_assistant_tokens=50,
            held_out_prompt_ids={candidates[0]["prompt_id"]},
        )


def test_duplicate_prompt_ids_are_rejected() -> None:
    module = _load_module()
    candidates = _candidates(50)
    candidates.append(dict(candidates[0]))

    with pytest.raises(ValueError, match="duplicate prompt_id"):
        module.select_arm(candidates, config=_config(), arm="B", target_assistant_tokens=50)


def test_candidate_assistant_count_must_match_binary_loss_mask() -> None:
    module = _load_module()
    candidates = _candidates(10)
    candidates[0]["assistant_tokens"] = 2

    with pytest.raises(ValueError, match=r"assistant_tokens.*loss_mask"):
        module.select_arm(candidates, config=_config(), arm="B", target_assistant_tokens=10)


def test_candidate_requires_pinned_source_revision_and_license() -> None:
    module = _load_module()
    candidates = _candidates(10)
    candidates[0]["source_revision"] = "main"

    with pytest.raises(ValueError, match="pinned source revision"):
        module.select_arm(candidates, config=_config(), arm="B", target_assistant_tokens=10)


def test_candidate_token_ids_must_match_the_pinned_target_tokenizer() -> None:
    module = _load_module()
    candidates = _candidates(10)
    candidates[0]["tokenizer_sha256"] = "e" * 64

    with pytest.raises(ValueError, match="tokenizer identity"):
        module.select_arm(candidates, config=_config(), arm="B", target_assistant_tokens=10)


def test_candidate_bucket_is_derived_from_full_length_before_training_truncation() -> None:
    module = _load_module()
    config = _config()
    row = _candidates(10)[0]
    row["full_token_count"] = 5000
    row["context_bucket"] = "le4k"

    with pytest.raises(ValueError, match="derived context bucket"):
        module.select_arm([row], config=config, arm="B", target_assistant_tokens=1)


def test_selection_trims_final_loss_mask_to_exact_token_quota() -> None:
    module = _load_module()
    config = {
        "seed": 42,
        "training_seq_len": 4096,
        "maximum_epochs": 2,
        "minimum_prompts_per_category": 0,
        "minimum_prompts_per_populated_cell": 0,
        "context_buckets": ["le4k"],
        "tokenizer_sha256": TOKENIZER_SHA256,
        "arms": {"B": {"ptv2": 1.0}},
        "pools": {"ptv2": {"categories": {"math": 1.0}}},
    }
    candidates = [
        {
            "prompt_id": f"row-{index}",
            "pool": "ptv2",
            "category": "math",
            "context_bucket": "le4k",
            "assistant_tokens": 6,
            "source_id": "source",
            "source_revision": "a" * 40,
            "license": "Apache-2.0",
            "source_manifest_sha256": "b" * 64,
            "source_file_path": "source.jsonl",
            "source_file_sha256": "c" * 64,
            "full_token_count": 8,
            "response_source": "target-synth",
            "tool_lane": "none",
            "tokenizer_sha256": TOKENIZER_SHA256,
            "input_ids": list(range(8)),
            "loss_mask": [0, 0, 1, 1, 1, 1, 1, 1],
        }
        for index in range(2)
    ]

    selected = module.select_arm(candidates, config=config, arm="B", target_assistant_tokens=10)

    assert sum(row["assistant_tokens"] for row in selected) == 10
    assert all(row["assistant_tokens"] == sum(row["loss_mask"]) for row in selected)


def test_manifest_records_unique_and_exposure_tokens() -> None:
    module = _load_module()
    selected = module.select_arm(
        _candidates(300), config=_config(), arm="D", target_assistant_tokens=300
    )

    manifest = module.build_arm_manifest(
        selected,
        config=_config(),
        arm="D",
        target_assistant_tokens=300,
        exposure_epochs=2,
    )

    assert manifest["unique_assistant_tokens"] == 300
    assert manifest["exposure_assistant_tokens"] == 600
    assert manifest["exposure_epochs"] == 2
    assert manifest["source_revisions"]
    assert manifest["category_assistant_tokens"]


def test_manifest_rejects_more_than_two_epochs() -> None:
    module = _load_module()
    selected = module.select_arm(
        _candidates(100), config=_config(), arm="B", target_assistant_tokens=50
    )

    with pytest.raises(ValueError, match="maximum_epochs"):
        module.build_arm_manifest(
            selected,
            config=_config(),
            arm="B",
            target_assistant_tokens=50,
            exposure_epochs=3,
        )


def test_materialize_parquet_is_atomic_and_checksum_manifested(tmp_path: Path) -> None:
    module = _load_module()
    selected = _candidates(10)[:5]
    for row in selected:
        row["messages"] = [{"role": "user", "content": row["prompt_id"]}]
    output = tmp_path / "arm-b"

    manifest = module.materialize_parquet(
        selected, output, rows_per_shard=2, tokenizer_sha256=TOKENIZER_SHA256
    )

    assert output.is_dir()
    assert not list(tmp_path.glob(".arm-b.partial-*"))
    assert manifest["file_count"] == 3
    assert manifest["row_count"] == 5
    assert manifest["unique_assistant_tokens"] == 5
    assert manifest["tokenizer_sha256"] == TOKENIZER_SHA256
    for entry in manifest["files"]:
        shard = output / entry["path"]
        assert shard.stat().st_size == entry["bytes"]
        assert module.sha256_file(shard) == entry["sha256"]
    assert json.loads((output / "MANIFEST.json").read_text()) == manifest


def test_materialize_parquet_uses_union_schema_for_tool_rows(tmp_path: Path) -> None:
    """A non-tool first row cannot erase top-level tools from a later row or shard."""
    module = _load_module()
    selected = _candidates(10)[:2]
    selected[0]["messages"] = [{"role": "user", "content": "plain"}]
    selected[1]["messages"] = [
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "call-1",
                    "type": "function",
                    "function": {"name": "shell", "arguments": "{}"},
                }
            ],
        },
        {"role": "tool", "content": "ok", "tool_call_id": "call-1"},
    ]
    selected[1]["tools"] = [
        {
            "type": "function",
            "function": {"name": "shell", "description": "run", "parameters": {"type": "object"}},
        }
    ]
    output = tmp_path / "corpus"

    module.materialize_parquet(
        selected, output, rows_per_shard=1, tokenizer_sha256=TOKENIZER_SHA256
    )

    import pyarrow.parquet as pq

    tables = [pq.read_table(path) for path in sorted(output.glob("*.parquet"))]
    assert all("tools" in table.column_names for table in tables)
    assert json.loads(tables[1].to_pylist()[0]["tools"])[0]["function"]["name"] == "shell"


def test_materialize_publishes_selection_with_corpus_in_one_rename(tmp_path: Path) -> None:
    module = _load_module()
    selected = _candidates(10)[:1]
    output = tmp_path / "corpus"

    module.materialize_parquet(
        selected,
        output,
        rows_per_shard=1,
        tokenizer_sha256=TOKENIZER_SHA256,
        selection_manifest={"schema_version": 1, "arm": "B"},
        selection_manifest_name="SELECTION.json",
    )

    selection = json.loads((output / "SELECTION.json").read_text())
    assert selection["corpus_manifest_path"] == str(output / "MANIFEST.json")
    assert selection["corpus_manifest_sha256"] == module.sha256_file(output / "MANIFEST.json")
    assert not list(tmp_path.glob(".corpus.partial-*"))


def test_manifest_binds_verified_input_file_identity() -> None:
    module = _load_module()
    selected = _candidates(10)[:1]
    selected[0]["source_manifest_sha256"] = "1" * 64
    selected[0]["source_file_sha256"] = "2" * 64
    selected[0]["source_file_path"] = "ptv2/train-000.parquet"

    manifest = module.build_arm_manifest(
        selected,
        config=_config(),
        arm="B",
        target_assistant_tokens=1,
        exposure_epochs=1,
    )

    assert manifest["source_revisions"][selected[0]["source_id"]]["manifest_sha256"] == "1" * 64
    assert manifest["source_files"] == [{"path": "ptv2/train-000.parquet", "sha256": "2" * 64}]


def test_cli_materializes_selected_rows_as_pinned_parquet(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _load_module()
    inventory = tmp_path / "inventory.jsonl"
    inventory.write_text("".join(json.dumps(row) + "\n" for row in _candidates(10)))
    config = tmp_path / "config.yaml"
    config.write_text(yaml.safe_dump(_config()))
    output_corpus = tmp_path / "arm-b"
    output_manifest = output_corpus / "SELECTION.json"
    observed: dict[str, object] = {}

    def fake_materialize(
        selected: list[dict],
        output: Path,
        *,
        rows_per_shard: int,
        tokenizer_sha256: str,
        selection_manifest: dict,
        selection_manifest_name: str,
    ) -> dict:
        observed.update(
            selected=selected,
            output=output,
            rows_per_shard=rows_per_shard,
            tokenizer_sha256=tokenizer_sha256,
        )
        output.mkdir()
        (output / "MANIFEST.json").write_text('{"schema_version": 1}\n')
        payload = selection_manifest | {
            "corpus_manifest_path": str(output / "MANIFEST.json"),
            "corpus_manifest_sha256": module.sha256_file(output / "MANIFEST.json"),
            "tokenizer_sha256": tokenizer_sha256,
        }
        (output / selection_manifest_name).write_text(json.dumps(payload) + "\n")
        return {"schema_version": 1}

    monkeypatch.setattr(module, "materialize_parquet", fake_materialize)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            str(MODULE_PATH),
            "--inventory",
            str(inventory),
            "--config",
            str(config),
            "--arm",
            "B",
            "--target-assistant-tokens",
            "10",
            "--tokenizer-sha256",
            TOKENIZER_SHA256,
            "--output-manifest",
            str(output_manifest),
            "--output-corpus",
            str(output_corpus),
            "--rows-per-shard",
            "3",
        ],
    )

    assert module.main() == 0
    assert observed["output"] == output_corpus
    assert observed["rows_per_shard"] == 3
    assert observed["tokenizer_sha256"] == TOKENIZER_SHA256
    assert sum(row["assistant_tokens"] for row in observed["selected"]) == 10
    selection = json.loads(output_manifest.read_text())
    assert selection["corpus_manifest_path"] == str(output_corpus / "MANIFEST.json")
    assert selection["corpus_manifest_sha256"] == module.sha256_file(
        output_corpus / "MANIFEST.json"
    )
    assert selection["tokenizer_sha256"] == TOKENIZER_SHA256
