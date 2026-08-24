# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Contracts for the B-balanced OCI-HSG canary path."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from common.specdec.build_qwen4b_b_canary import (
    CANARY_CELL_QUOTAS,
    CANARY_LANGUAGE_QUOTAS,
    build_canary_occurrences,
)
from common.specdec.qwen4b_b_canary_manifest import (
    BCanaryManifest,
    BCanaryTopology,
    load_b_canary_manifest,
    write_b_canary_manifest,
)
from common.specdec.qwen4b_b_readiness import (
    BReadinessInputs,
    Task9BalancedView,
    assess_b_readiness,
)


def _task9_view(shard_root: Path) -> Task9BalancedView:
    shards = tuple(f"shard-{index:03d}.jsonl" for index in range(201))
    for shard in shards:
        (shard_root / shard).write_text("{}\n")
    return Task9BalancedView(
        strategy="B-balanced",
        occurrence_count=2_000_000,
        cell_occurrence_counts={
            "math": 500_000,
            "code": 400_000,
            "stem": 500_000,
            "chat": 400_000,
            "multilingual": 200_000,
        },
        multilingual_occurrence_counts={language: 40_000 for language in ("de", "ja", "es", "fr", "it")},
        declared_shards=shards,
        shard_root=shard_root,
        trainer_epochs=1,
        global_batch_size=512,
        segment_occurrences=(1_300_000, 700_000),
        segment_steps=(2_540, 1_368),
        cumulative_steps=(2_540, 3_908),
        segment_final_valid_occurrences=(32, 96),
        source_native_responses=True,
        publication_sha256="a" * 64,
        destination_sha256="b" * 64,
        tokenizer_sha256="c" * 64,
        chat_template_sha256="d" * 64,
        assistant_loss_mask_sha256="e" * 64,
    )


def test_b_readiness_is_independent_and_rejects_orphan_shards(tmp_path: Path) -> None:
    """B accepts its complete Task9 projection without any A-repair receipt."""
    shard_root = tmp_path / "b-shards"
    shard_root.mkdir()
    inputs = BReadinessInputs(task9_b_view=_task9_view(shard_root), source_commit="f" * 40)

    receipt = assess_b_readiness(inputs)

    assert receipt.ready
    assert receipt.blocker_codes == ()
    assert receipt.declared_shard_count == 201
    (shard_root / "orphan.jsonl").write_text("{}\n")
    rejected = assess_b_readiness(inputs)
    assert not rejected.ready
    assert rejected.blocker_codes == ("B_ORPHAN_SHARD",)


def test_b_readiness_rejects_renormalized_quota_and_wrong_schedule(tmp_path: Path) -> None:
    """B fails closed when a Task9 projection diverges from the scientific policy."""
    shard_root = tmp_path / "b-shards"
    shard_root.mkdir()
    view = _task9_view(shard_root)
    inputs = BReadinessInputs(
        task9_b_view=Task9BalancedView(
            **{
                **view.__dict__,
                "cell_occurrence_counts": {**view.cell_occurrence_counts, "math": 499_999},
                "cumulative_steps": (2_540, 25_391),
            }
        ),
        source_commit="f" * 40,
    )

    receipt = assess_b_readiness(inputs)

    assert not receipt.ready
    assert receipt.blocker_codes == ("B_CELL_QUOTA", "B_SCHEDULE")


def test_b_canary_builder_preserves_exact_scaled_cell_and_language_quotas() -> None:
    """The B canary uses exactly 102,400 source-native occurrences."""
    rows: list[dict[str, object]] = []
    for cell, quota in CANARY_CELL_QUOTAS.items():
        if cell == "multilingual":
            for language, language_quota in CANARY_LANGUAGE_QUOTAS.items():
                rows.extend(
                    {"uuid": f"{language}-{index}", "cell": cell, "language": language, "source_native": True}
                    for index in range(language_quota)
                )
        else:
            rows.extend(
                {"uuid": f"{cell}-{index}", "cell": cell, "language": "", "source_native": True}
                for index in range(quota)
            )

    selected = build_canary_occurrences(rows, seed=17)

    assert len(selected) == 102_400
    assert {cell: sum(row["cell"] == cell for row in selected) for cell in CANARY_CELL_QUOTAS} == CANARY_CELL_QUOTAS
    assert {
        language: sum(row["language"] == language for row in selected)
        for language in CANARY_LANGUAGE_QUOTAS
    } == CANARY_LANGUAGE_QUOTAS


def test_manifest_binds_oci_16_node_200_step_runtime_contract(tmp_path: Path) -> None:
    """The immutable canary receipt describes all 64 working GPUs and GBS512."""
    manifest = BCanaryManifest(
        readiness_receipt_sha256="a" * 64,
        canary_occurrence_count=102_400,
        topology=BCanaryTopology(
            cluster="oci-hsg",
            account="nemotron_sw_post",
            partition="batch",
            nodes=16,
            segment=16,
            serve_nodes=8,
            train_nodes=8,
            gpus_per_node=4,
            cpu_datamover=96,
            per_device_batch_size=4,
            gradient_accumulation_steps=4,
            max_steps=200,
            wandb_project="sna-qwen3-4b-dataset-study",
        ),
        source_commit="f" * 40,
    )
    path = tmp_path / "b-canary.json"

    write_b_canary_manifest(path, manifest)

    assert load_b_canary_manifest(path) == manifest
    assert manifest.global_batch_size == 512
    assert manifest.active_gpu_ranks == 64
    assert manifest.scientific_milestone is False
    assert json.loads(path.read_text())["topology"]["cpu_datamover"] == 96


def test_canary_runner_and_submitter_enforce_bounded_evidence_contract() -> None:
    """The launch surface runs only B's bounded test-only canary."""
    root = Path(__file__).resolve().parents[1] / "common/specdec"
    runner = (root / "run_qwen4b_b_canary.sbatch").read_text()
    submitter = (root / "submit_qwen4b_b_canary.sh").read_text()

    assert "#SBATCH --nodes=16" in runner
    assert "#SBATCH --gpus-per-node=4" in runner
    assert "--cpus-per-task=96" in runner
    assert "--nodes=8" in runner
    assert "finite_loss" in runner
    assert "checkpoint_reloaded" in runner
    assert "drafter_exported" in runner
    assert "evaluator_completed" in runner
    assert "all_64_gpus_active" in runner
    assert 'payload["scientific_milestone"] = False' in runner
    assert "sbatch --test-only" in submitter
    assert "--test-only" in submitter
    assert "nemotron_sw_post" in submitter
    assert "nemotron_n4_post" in submitter
    assert "production submission is not supported" in submitter
