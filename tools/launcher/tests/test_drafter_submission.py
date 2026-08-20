# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Contracts for the reusable OCI-HSG drafter workflow."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from common.specdec.drafter_job_manifest import (
    DrafterExperiment,
    PinnedPaths,
    SlurmSettings,
    TargetTopology,
    canonical_manifest,
    speculative_tokens,
    validate_topology,
    write_manifest,
)

_LAUNCHER_DIR = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize(
    ("nodes", "segment", "role"),
    [(4, 4, "training"), (1, 1, "evaluation")],
)
def test_validate_topology_accepts_supported_oci_segments(
    nodes: int, segment: int, role: str
) -> None:
    """Training and evaluation allocations stay within one NVL72 segment."""
    validate_topology(nodes, segment, role)


@pytest.mark.parametrize(
    ("nodes", "segment", "role"),
    [(4, 3, "training"), (19, 19, "training"), (1, 2, "evaluation")],
)
def test_validate_topology_rejects_invalid_segments(nodes: int, segment: int, role: str) -> None:
    """A segment must divide nodes and cannot exceed the 18-node block size."""
    with pytest.raises(ValueError):
        validate_topology(nodes, segment, role)


@pytest.mark.parametrize(
    ("method", "block_size", "expected"),
    [("dflash", 8, 7), ("dflash", 16, 15), ("dspark", 8, 8), ("dspark", 16, 16)],
)
def test_speculative_tokens_uses_the_pinned_b_k_matrix(
    method: str, block_size: int, expected: int
) -> None:
    """Every public evaluator run uses the corresponding DFlash/DSpark horizon."""
    assert speculative_tokens(method, block_size) == expected


def _experiment() -> DrafterExperiment:
    return DrafterExperiment(
        target="Qwen/Qwen3-30B-A3B",
        dataset="open-perfectblend",
        method="dflash",
        block_size=8,
        cumulative_max_steps=(500, 1000),
        run_name="qwen3-30b-dflash-b8",
        topology=TargetTopology(
            target_kind="qwen3-30b-a3b",
            capture_ids=(2, 13, 24, 35, 46, 48),
            serve_tp=2,
            per_device_train_batch_size=4,
            gradient_accumulation_steps=16,
            num_attention_heads=32,
            num_key_value_heads=4,
            head_dim=128,
            intermediate_size=6144,
        ),
        paths=PinnedPaths(
            source_path="/home/user/ModelOpt",
            source_sha="a" * 40,
            image_path="/lustre/images/vllm.sqsh",
            runtime_archive_path="/lustre/runtimes/modelopt-runtime.tar.zst",
            target_path="/lustre/models/Qwen3-30B-A3B",
            dataset_path="/lustre/datasets/open-perfectblend.parquet",
            output_root="/lustre/results/qwen3-30b-dflash-b8",
        ),
        slurm=SlurmSettings(
            account="nemotron_n3_post",
            partition="batch",
            nodes=4,
            gpus_per_node=4,
            segment=4,
        ),
    )


def test_canonical_manifest_is_stable_and_written_atomically(tmp_path: Path) -> None:
    """Canonical manifests are deterministic durable inputs for later submitters."""
    experiment = _experiment()
    expected = canonical_manifest((experiment,))
    output = tmp_path / "manifest.json"

    write_manifest(output, (experiment,))

    assert output.read_text() == expected
    assert json.loads(expected)["experiments"][0]["num_speculative_tokens"] == 7
    assert not list(tmp_path.glob(".manifest.json.*"))


@pytest.mark.parametrize(
    ("field", "value"),
    [("source_path", "relative/source"), ("source_sha", "short"), ("image_path", "")],
)
def test_pinned_paths_require_exact_absolute_values(field: str, value: str) -> None:
    """A mutable path or non-commit source revision cannot enter a manifest."""
    values = _experiment().paths.__dict__.copy()
    values[field] = value
    with pytest.raises(ValueError):
        PinnedPaths(**values)


def test_pinned_paths_reject_normalized_traversal_outside_the_declared_root() -> None:
    """Canonical path validation closes lexical ``..`` containment escapes."""
    values = _experiment().paths.__dict__.copy()
    values["target_path"] = "/lustre/models/../../home/other"
    with pytest.raises(ValueError):
        PinnedPaths(**values)


@pytest.mark.parametrize(
    ("kind", "per_device", "accumulation", "capture_ids", "serve_tp"),
    [
        ("qwen3-30b-a3b", 4, 16, (2, 13, 24, 35, 46, 48), 2),
        ("qwen3-235b-a22b", 2, 32, (2, 25, 47, 69, 92, 94), 4),
    ],
)
def test_target_topology_pins_capture_ids_and_global_batch_arithmetic(
    kind: str,
    per_device: int,
    accumulation: int,
    capture_ids: tuple[int, ...],
    serve_tp: int,
) -> None:
    """Both targets resolve to world-size eight and an exact global batch of 512."""
    topology = TargetTopology.for_kind(kind)

    assert topology.capture_ids == capture_ids
    assert topology.serve_tp == serve_tp
    assert topology.per_device_train_batch_size == per_device
    assert topology.gradient_accumulation_steps == accumulation
    assert topology.per_device_train_batch_size * topology.gradient_accumulation_steps * 8 == 512


def test_run_name_cannot_be_blank() -> None:
    """Receipts and output namespaces always have an explicit identity."""
    values = _experiment().__dict__.copy()
    values["run_name"] = "  "
    with pytest.raises(ValueError):
        DrafterExperiment(**values)


def test_model_staging_is_pinned_and_node_local_until_completion() -> None:
    """Staging uses one segment, a pinned revision, and only durable Lustre output."""
    script = (_LAUNCHER_DIR / "common/specdec/stage_hf_model.sh").read_text()

    for required in (
        "--segment=1",
        "--revision",
        "/raid/scratch",
        "HF_HOME",
        "HF_HUB_CACHE",
        "LOCK",
        "snapshot_download",
        "sbatch --test-only",
        "squeue -h -n",
        "xargs -0 -r -P 4",
        "snapshot-manifest.json",
        "completion.json",
    ):
        assert required in script
    assert "pip install" not in script
    assert "git clone" not in script
    assert "find /lustre" not in script
    assert "rm -rf /lustre" not in script


def test_training_wave_uses_the_fixed_four_node_streaming_topology() -> None:
    """A wave renders unique tuple jobs with node-local mutable runtime state."""
    submitter = (_LAUNCHER_DIR / "common/specdec/submit_drafter_training_wave.sh").read_text()
    runner = (_LAUNCHER_DIR / "common/specdec/run_drafter_training.sbatch").read_text()

    for required in (
        "-N4",
        "--segment=4",
        "sbatch --test-only",
        "squeue -h -n",
        "duplicate training tuple",
        "sacct -X -n --name",
        "identity",
        "--dependency=afterok:",
        "receipt",
    ):
        assert required in submitter
    for required in (
        "#SBATCH -N 4",
        "#SBATCH --segment=4",
        "SERVE_NODES=2",
        "EAGLE_CAPTURE_IDS",
        "IMAGE_PATH",
        "--container-image",
        "gradient_accumulation_steps",
        "num_attention_heads",
        "GLOBAL_BATCH_SIZE=512",
        "TRAINER_NODES=2",
        "GPUS_PER_NODE=4",
        "PER_DEVICE_TRAIN_BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS * TRAINER_NODES * GPUS_PER_NODE",
        "/home",
        "rev-parse HEAD",
        "status --porcelain",
        "/raid/scratch/${SLURM_JOB_ID}",
        "RUNTIME_ARCHIVE",
        "tar --extract",
        "HF_HOME",
        "TRITON_CACHE_DIR",
        "XDG_CACHE_HOME",
        "SQLITE_TMPDIR",
        "srun --nodes=4 --ntasks=4 --ntasks-per-node=1",
        "training.output_dir",
        "LOSS_OBJECTIVE=dpace",
        "LOSS_OBJECTIVE=decay",
    ):
        assert required in runner
    assert "/lustre/.cache" not in runner
    assert "pip install" not in runner
    assert "git clone" not in runner


def test_resume_chain_gates_each_cumulative_wave_on_public_acceptance() -> None:
    """Only a successful nine-subset evaluation releases the next stable-output wave."""
    script = (_LAUNCHER_DIR / "common/specdec/submit_drafter_resume_chain.sh").read_text()
    readme = (_LAUNCHER_DIR / "examples/Qwen/Qwen3-30B-A3B/README.md").read_text()

    for required in (
        "afterok",
        "--segment=1",
        "--nodes=1",
        "sbatch --test-only",
        "acceptance.csv",
        "HumanEval,math_reasoning,qa,question,rag,summarization,tool_call,translation,writing",
        "--time=03:55:00",
        "squeue -h -n",
        "sacct -X -n --name",
        "--run-evaluation",
        "EXPERIMENT_IDENTITY",
        "receipt",
    ):
        assert required in script
    assert "--wrap" not in script
    for required in (
        "Qwen3-30B-A3B",
        "Qwen3-235B-A22B",
        "Thinking",
        "DFlash B8",
        "DSpark B16",
        "--segment=4",
        "--segment=1",
        "/raid/scratch",
        "receipt",
        "Q30 uses per-device batch 4 with gradient accumulation 16",
        "Q235 uses per-device batch 2 with gradient accumulation 32",
    ):
        assert required in readme
    assert "512 / (2 * 4) = 64" not in readme
