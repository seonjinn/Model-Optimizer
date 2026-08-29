# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Pin the three-arm plumbing canary manifest.

The canary exists to attribute a startup failure to one drafter architecture,
which only works if the three entries differ in exactly one field. These tests
hold that: same corpus, same target, same allocation, same 20-step boundary,
distinct output roots and identities, and a horizon that follows the method
rather than being restated per arm.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from common.specdec.build_q30t_canary_manifest import (
    CANARY_METHODS,
    build_canary_experiments,
)
from common.specdec.drafter_job_manifest import (
    PinnedPaths,
    SlurmSettings,
    load_manifest,
    write_manifest,
)

LUSTRE = "/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sna"
OUTPUT_ROOT = Path(f"{LUSTRE}/specdec_ptv23/training")


def _paths() -> PinnedPaths:
    return PinnedPaths(
        source_path="/home/sna/modelopt-q30t-design",
        source_sha="b549c2d338949e90c33f60d789a42bd375c0bd7b",
        image_path=f"{LUSTRE}/modelopt-specdec/image/vllm.sqsh",
        image_sha256="44b75976cc6583f890aff59031a2b5197eda69326e4b071994572f9a614387ba",
        runtime_archive_path=f"{LUSTRE}/modelopt-specdec/runtime/runtime.tar.zst",
        runtime_archive_sha256=(
            "72aef280291bedaba40b7f33185b6103716427dd4fbe4c7d65c5301cd615c379"
        ),
        target_path=f"{LUSTRE}/models/Qwen3-30B-A3B-Thinking-2507-144afc2f",
        dataset_path=f"{LUSTRE}/specdec_ptv23/assets/canary/data",
        output_root=str(OUTPUT_ROOT),
    )


def _build(nodes: int = 16):
    return build_canary_experiments(
        paths=_paths(),
        slurm=SlurmSettings(
            account="nemotron_sw_post",
            partition="batch",
            nodes=nodes,
            gpus_per_node=4,
            segment=nodes,
        ),
        output_root=OUTPUT_ROOT,
        target_label="q30-thinking",
        dataset_label="ptv23-canary",
        run_prefix="q30t-ptv23-canary",
        canary_steps=20,
    )


def test_the_canary_covers_all_three_architectures_and_nothing_else() -> None:
    experiments = _build()
    assert {experiment.method for experiment in experiments} == set(CANARY_METHODS)
    assert len(experiments) == 3


def test_the_arms_differ_only_by_method() -> None:
    """Anything else varying would make a failure unattributable."""
    experiments = _build()
    shared = {
        (
            experiment.target,
            experiment.dataset,
            experiment.block_size,
            experiment.sample_size,
            experiment.cumulative_max_steps,
            experiment.topology,
            experiment.slurm,
            experiment.paths.dataset_path,
            experiment.paths.source_sha,
            experiment.paths.image_sha256,
        )
        for experiment in experiments
    }
    assert len(shared) == 1


def test_each_arm_owns_a_distinct_output_root_and_identity() -> None:
    """A shared output root would have the second arm adopt the first's state."""
    experiments = _build()
    assert len({experiment.paths.output_root for experiment in experiments}) == 3
    assert len({experiment.experiment_id for experiment in experiments}) == 3
    assert len({experiment.run_name for experiment in experiments}) == 3


def test_the_horizon_follows_the_method() -> None:
    """DSpark's shift-label alignment keeps position 0, so it accepts K = B."""
    horizons = {
        experiment.method: experiment.num_speculative_tokens for experiment in _build()
    }
    assert horizons == {"dflash": 7, "dflash2": 7, "dspark": 8}


def test_sixteen_nodes_hold_the_global_batch_size_at_512() -> None:
    """Half the allocation trains and half serves, so 16 nodes is 32 trainer ranks."""
    topology = _build()[0].topology
    assert topology.gradient_accumulation_steps == 4
    assert topology.per_device_train_batch_size * topology.gradient_accumulation_steps * 32 == 512


def test_a_two_node_allocation_reaccumulates_to_the_same_batch() -> None:
    topology = _build(nodes=2)[0].topology
    assert topology.gradient_accumulation_steps == 32


def test_an_odd_allocation_is_refused_rather_than_rounded() -> None:
    with pytest.raises(ValueError):
        _build(nodes=6)


def test_the_manifest_round_trips_through_the_canonical_loader(tmp_path: Path) -> None:
    """The training job reloads by index, so the written order is the contract."""
    manifest = tmp_path / "canary.json"
    write_manifest(manifest, _build())
    reloaded = load_manifest(manifest)
    assert [experiment.method for experiment in reloaded] == [
        experiment["method"] for experiment in json.loads(manifest.read_text())["experiments"]
    ]
    assert len(reloaded) == 3
