# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Structural contracts for Qwen3 DFlash and DSpark streaming launchers."""

from pathlib import Path

import pytest
import yaml


_LAUNCHER_DIR = Path(__file__).resolve().parents[1]
_DFLASH_RECIPE = "--config modules/Model-Optimizer/modelopt_recipes/general/speculative_decoding/dflash.yaml"
_DSPARK_RECIPE = "--config modules/Model-Optimizer/modelopt_recipes/general/speculative_decoding/dspark.yaml"


@pytest.mark.parametrize(
    ("relative_path", "hf_model", "capture_ids", "recipe", "dspark_dimensions"),
    [
        (
            "examples/Qwen/Qwen3-30B-A3B/hf_streaming_dflash_multi_node.yaml",
            "/hf-local/Qwen/Qwen3-30B-A3B",
            "[2,13,24,35,46,48]",
            _DFLASH_RECIPE,
            (),
        ),
        (
            "examples/Qwen/Qwen3-30B-A3B/hf_streaming_dspark_multi_node.yaml",
            "/hf-local/Qwen/Qwen3-30B-A3B",
            "[2,13,24,35,46,48]",
            _DSPARK_RECIPE,
            (
                "dflash.dflash_architecture_config.num_attention_heads=16",
                "dflash.dflash_architecture_config.num_key_value_heads=4",
                "dflash.dflash_architecture_config.head_dim=128",
                "dflash.dflash_architecture_config.intermediate_size=6144",
            ),
        ),
        (
            "examples/Qwen/Qwen3-235B-A22B/hf_streaming_dflash_multi_node.yaml",
            "/hf-local/Qwen/Qwen3-235B-A22B",
            "[2,25,47,69,92,94]",
            _DFLASH_RECIPE,
            (),
        ),
        (
            "examples/Qwen/Qwen3-235B-A22B/hf_streaming_dspark_multi_node.yaml",
            "/hf-local/Qwen/Qwen3-235B-A22B",
            "[2,25,47,69,92,94]",
            _DSPARK_RECIPE,
            (
                "dflash.dflash_architecture_config.num_attention_heads=32",
                "dflash.dflash_architecture_config.num_key_value_heads=8",
                "dflash.dflash_architecture_config.head_dim=128",
                "dflash.dflash_architecture_config.intermediate_size=12288",
            ),
        ),
    ],
)
def test_qwen_drafter_launcher_contract(
    relative_path: str,
    hf_model: str,
    capture_ids: str,
    recipe: str,
    dspark_dimensions: tuple[str, ...],
) -> None:
    """Keep Qwen3 streaming draft launchers aligned with their target backbones."""
    with (_LAUNCHER_DIR / relative_path).open() as yaml_file:
        config = yaml.safe_load(yaml_file)

    global_vars = config["pipeline"]["global_vars"]
    task = config["pipeline"]["task_1"]
    environment = {
        key: value for item in task["environment"] for key, value in item.items()
    }

    assert global_vars["hf_model"] == hf_model
    assert global_vars["dflash_block_size"] == "8"
    assert global_vars["dflash_loss_decay_factor"] == "4"
    assert "dflash.dflash_block_size=<<global_vars.dflash_block_size>>" in task["args"]
    assert "dflash.dflash_loss_decay_factor=<<global_vars.dflash_loss_decay_factor>>" in task["args"]
    assert environment["EAGLE_CAPTURE_IDS"] == capture_ids
    assert recipe in task["args"]
    for dimension in dspark_dimensions:
        assert dimension in task["args"]


def test_make_dataset_uses_python3_for_vllm_images() -> None:
    script = (_LAUNCHER_DIR / "common" / "eagle3" / "make_dataset.sh").read_text()

    assert "${PYTHON_BIN:-python3}" in script
    assert '-m pip install --no-cache-dir "datasets"' in script


def test_qwen30_dflash_training_data_is_overridable() -> None:
    path = "examples/Qwen/Qwen3-30B-A3B/hf_streaming_dflash_multi_node.yaml"
    with (_LAUNCHER_DIR / path).open() as yaml_file:
        config = yaml.safe_load(yaml_file)

    assert config["pipeline"]["global_vars"]["hf_data"] == "/scratchspace/data/train.jsonl"
    assert "data.data_path=<<global_vars.hf_data>>" in config["pipeline"]["task_1"]["args"]
    environment = {
        key: value
        for item in config["pipeline"]["task_1"]["environment"]
        for key, value in item.items()
    }
    assert environment["MODELOPT_RUNTIME"] == "<<global_vars.modelopt_runtime>>"


def test_streaming_training_can_reuse_shared_runtime() -> None:
    script = (_LAUNCHER_DIR / "common" / "eagle3" / "train_eagle_streaming.sh").read_text()

    assert 'source "$MODELOPT_RUNTIME/bin/activate"' in script
    assert 'if [ -n "${MODELOPT_RUNTIME:-}" ]; then' in script
