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
                        "assistant_tokens": 1,
                        "source_id": f"{pool}-{category}",
                        "source_revision": "a" * 40,
                        "license": "Apache-2.0",
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


def test_candidate_must_already_fit_the_training_sequence_length() -> None:
    module = _load_module()
    config = _config()
    config["training_seq_len"] = 1

    with pytest.raises(ValueError, match="training sequence length"):
        module.select_arm(_candidates(10), config=config, arm="B", target_assistant_tokens=10)


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


def test_cli_materializes_selected_rows_as_pinned_parquet(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _load_module()
    inventory = tmp_path / "inventory.jsonl"
    inventory.write_text("".join(json.dumps(row) + "\n" for row in _candidates(10)))
    config = tmp_path / "config.yaml"
    config.write_text(yaml.safe_dump(_config()))
    output_manifest = tmp_path / "selection.json"
    output_corpus = tmp_path / "arm-b"
    observed: dict[str, object] = {}

    def fake_materialize(
        selected: list[dict], output: Path, *, rows_per_shard: int, tokenizer_sha256: str
    ) -> dict:
        observed.update(
            selected=selected,
            output=output,
            rows_per_shard=rows_per_shard,
            tokenizer_sha256=tokenizer_sha256,
        )
        output.mkdir()
        (output / "MANIFEST.json").write_text('{"schema_version": 1}\n')
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
