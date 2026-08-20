# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Structural contracts for Qwen3 DFlash and DSpark streaming launchers."""

import os
from pathlib import Path
import subprocess

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
    assert "data.sample_size=50000" in config["pipeline"]["task_1"]["args"]
    assert "training.seed=42" in config["pipeline"]["task_1"]["args"]
    environment = {
        key: value
        for item in config["pipeline"]["task_1"]["environment"]
        for key, value in item.items()
    }
    assert environment["MODELOPT_RUNTIME"] == "<<global_vars.modelopt_runtime>>"


def test_qwen30_dspark_uses_shared_opb_runtime_and_paper_loss() -> None:
    path = "examples/Qwen/Qwen3-30B-A3B/hf_streaming_dspark_multi_node.yaml"
    with (_LAUNCHER_DIR / path).open() as yaml_file:
        config = yaml.safe_load(yaml_file)

    global_vars = config["pipeline"]["global_vars"]
    task = config["pipeline"]["task_1"]
    environment = {
        key: value
        for item in task["environment"]
        for key, value in item.items()
    }

    assert global_vars["hf_data"] == "/scratchspace/data/train.jsonl"
    assert global_vars["modelopt_runtime"] == ""
    assert "data.data_path=<<global_vars.hf_data>>" in task["args"]
    assert "training.seed=42" in task["args"]
    assert "dflash.dflash_loss_objective=decay" in task["args"]
    assert environment["MODELOPT_RUNTIME"] == "<<global_vars.modelopt_runtime>>"
    assert environment["SERVE_BLOCK_SIZE"] == "32"
    assert environment["SERVE_READY_TIMEOUT"] == "1800"


def test_qwen30_dspark_wandb_is_overrideable_and_inode_safe() -> None:
    """Keep W&B metadata overrideable without writing its cache to Lustre."""
    path = "examples/Qwen/Qwen3-30B-A3B/hf_streaming_dspark_multi_node.yaml"
    with (_LAUNCHER_DIR / path).open() as yaml_file:
        config = yaml.safe_load(yaml_file)

    global_vars = config["pipeline"]["global_vars"]
    task = config["pipeline"]["task_1"]
    environment = {
        key: value
        for item in task["environment"]
        for key, value in item.items()
    }

    assert global_vars["report_to"] == "wandb"
    assert global_vars["run_name"] == "Qwen3-30B-A3B_DSpark_streaming_multi_node"
    assert "training.report_to=<<global_vars.report_to>>" in task["args"]
    assert "training.run_name=<<global_vars.run_name>>" in task["args"]
    assert environment["WANDB_PROJECT"] == "sna-modelopt-specdec"
    assert environment["WANDB_LOG_MODEL"] == "false"
    assert environment["WANDB_WATCH"] == "false"
    assert environment["WANDB_DIR"] == "/tmp/wandb"
    assert "WANDB_API_KEY" not in environment


def test_streaming_training_can_reuse_shared_runtime() -> None:
    script = (_LAUNCHER_DIR / "common" / "eagle3" / "train_eagle_streaming.sh").read_text()

    assert 'source "$MODELOPT_RUNTIME/bin/activate"' in script
    assert 'if [ -n "${MODELOPT_RUNTIME:-}" ]; then' in script


def _write_executable(path: Path, body: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(body)
    path.chmod(0o755)


def test_streaming_serve_uses_activated_runtime_python(tmp_path: Path) -> None:
    """The vLLM console script must not bypass the activated shared runtime."""
    runtime = tmp_path / "runtime"
    invocation_log = tmp_path / "invocations.log"
    activate = runtime / "bin" / "activate"
    activate.parent.mkdir(parents=True)
    activate.write_text(f'export PATH="{runtime / "bin"}:$PATH"\n')
    fake_python = f"""#!/bin/sh
printf 'runtime-python %s\\n' "$*" >> "{invocation_log}"
[ "$1" = "-m" ] && sleep 1
exit 0
"""
    _write_executable(runtime / "bin" / "python", fake_python)
    _write_executable(runtime / "bin" / "python3", fake_python)
    _write_executable(
        runtime / "bin" / "vllm",
        f"""#!/bin/sh
printf 'console-script %s\\n' "$*" >> "{invocation_log}"
exit 0
""",
    )
    _write_executable(
        runtime / "bin" / "curl",
        """#!/bin/sh
exit 0
""",
    )
    _write_executable(
        runtime / "bin" / "nvidia-smi",
        """#!/bin/sh
printf '2\\n'
""",
    )

    trainer = tmp_path / "modules/Model-Optimizer/examples/speculative_decoding/launch_train.sh"
    _write_executable(trainer, "#!/bin/sh\nexit 0\n")

    env = {
        **os.environ,
        "MODELOPT_RUNTIME": str(runtime),
        "HF_MODEL_CKPT": "target-model",
        "EAGLE_CAPTURE_IDS": "[2,10,18,26,34,36]",
        "SLURM_NNODES": "1",
        "SLURM_NODEID": "0",
        "SERVE_GPU": "0",
        "SERVE_TP": "1",
        "TRAIN_GPUS": "1",
        "SERVE_READY_TIMEOUT": "5",
    }
    result = subprocess.run(
        [
            "bash",
            str(_LAUNCHER_DIR / "common" / "eagle3" / "train_eagle_streaming.sh"),
            "training.output_dir=/scratchspace/dflash",
        ],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=15,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "runtime-python -m vllm.entrypoints.cli.main serve target-model" in invocation_log.read_text()


def test_training_launcher_uses_activated_python_for_accelerate(tmp_path: Path) -> None:
    """The Accelerate console script must not bypass the activated shared runtime."""
    runtime_bin = tmp_path / "runtime/bin"
    invocation_log = tmp_path / "invocations.log"
    runtime_bin.mkdir(parents=True)
    _write_executable(
        runtime_bin / "python3",
        f"""#!/bin/sh
printf 'runtime-python3 %s\\n' "$*" >> "{invocation_log}"
exit 0
""",
    )
    _write_executable(
        runtime_bin / "accelerate",
        f"""#!/bin/sh
printf 'console-script %s\\n' "$*" >> "{invocation_log}"
exit 0
""",
    )

    config = tmp_path / "config.yaml"
    config.write_text("model: {}\n")
    env = {
        **os.environ,
        "PATH": f"{runtime_bin}:{os.environ['PATH']}",
        "GPU_PER_NODE": "1",
        "SLURM_PROCID": "0",
    }
    result = subprocess.run(
        [
            "bash",
            str(_LAUNCHER_DIR.parents[1] / "examples/speculative_decoding/launch_train.sh"),
            "--config",
            str(config),
            "--num_nodes",
            "2",
            "--head_node_ip",
            "127.0.0.1",
        ],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    invocations = invocation_log.read_text()
    assert "runtime-python3 -m accelerate.commands.launch" in invocations
    assert "console-script" not in invocations
