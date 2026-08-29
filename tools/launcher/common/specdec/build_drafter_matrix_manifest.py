# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Build a drafter training manifest for a method/block/target matrix.

``build_drafter_full_manifest.py`` renders one specific 32-run matrix whose
shape, methods and Slurm account are all fixed. This builder takes the matrix
shape on the command line and the per-cluster asset paths from a JSON file, so
the same study can be rendered against whichever cluster has capacity without
editing code. That matters because the five clusters mount shared storage at
different roots, so every pinned path in the manifest changes per cluster.

Run names and output roots are derived from the matrix coordinates, which keeps
two renderings of the same arm pointing at the same checkpoints and lets a
fresh study be started simply by choosing a new ``--run-prefix``.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

from common.specdec.drafter_job_manifest import (
    DrafterExperiment,
    PinnedPaths,
    SlurmSettings,
    TargetTopology,
    write_manifest,
)

_GLOBAL_BATCH_SIZE = 512

_TARGET_KINDS = {
    "q30": "qwen3-30b-a3b",
    "q235": "qwen3-235b-a22b",
}

_ASSET_KEYS = (
    "source_path",
    "source_sha",
    "image_path",
    "image_sha256",
    "runtime_archive_path",
    "runtime_archive_sha256",
    "dataset_path",
    "output_root",
    "targets",
)


def accumulation_steps(topology: TargetTopology, nodes: int) -> int:
    """Return the accumulation that holds the global batch size at 512.

    Half of every allocation serves the target model, so only ``nodes // 2``
    nodes contribute trainer GPUs.
    """
    trainer_world_size = (nodes // 2) * 4
    divisor = topology.per_device_train_batch_size * trainer_world_size
    if divisor <= 0 or _GLOBAL_BATCH_SIZE % divisor:
        raise ValueError(f"{nodes} nodes cannot hold a global batch size of {_GLOBAL_BATCH_SIZE}")
    return _GLOBAL_BATCH_SIZE // divisor


def load_assets(path: Path) -> dict:
    assets = json.loads(path.read_text())
    missing = [key for key in _ASSET_KEYS if key not in assets]
    if missing:
        raise ValueError(f"asset file is missing required keys: {', '.join(missing)}")
    if not isinstance(assets["targets"], dict) or not assets["targets"]:
        raise ValueError("asset file must map at least one target label to a model path")
    return assets


def build_experiments(
    assets: dict,
    *,
    family: str,
    variants: tuple[str, ...],
    methods: tuple[str, ...],
    block_sizes: tuple[int, ...],
    dataset: str,
    boundaries: tuple[int, ...],
    nodes: int,
    account: str,
    partition: str,
    run_prefix: str,
) -> tuple[DrafterExperiment, ...]:
    topology = TargetTopology.for_kind(_TARGET_KINDS[family])
    topology = replace(topology, gradient_accumulation_steps=accumulation_steps(topology, nodes))
    output_root = assets["output_root"].rstrip("/")

    experiments = []
    for variant in variants:
        label = f"{family}-{variant}"
        target_path = assets["targets"].get(label)
        if target_path is None:
            raise ValueError(f"asset file has no model path for target {label}")
        for method in methods:
            for block_size in block_sizes:
                run_name = (
                    f"{run_prefix}-{label}-{dataset.split('-')[0]}-{method}-b{block_size}-{nodes}n"
                )
                experiments.append(
                    DrafterExperiment(
                        target=label,
                        dataset=dataset,
                        method=method,
                        block_size=block_size,
                        cumulative_max_steps=boundaries,
                        run_name=run_name,
                        topology=topology,
                        paths=PinnedPaths(
                            source_path=assets["source_path"],
                            source_sha=assets["source_sha"],
                            image_path=assets["image_path"],
                            image_sha256=assets["image_sha256"],
                            runtime_archive_path=assets["runtime_archive_path"],
                            runtime_archive_sha256=assets["runtime_archive_sha256"],
                            target_path=target_path,
                            dataset_path=assets["dataset_path"],
                            output_root=f"{output_root}/{run_name}",
                        ),
                        slurm=SlurmSettings(
                            account=account,
                            partition=partition,
                            nodes=nodes,
                            gpus_per_node=4,
                            segment=nodes,
                        ),
                    )
                )
    return tuple(experiments)


def _int_list(value: str) -> tuple[int, ...]:
    return tuple(int(item) for item in value.split(",") if item)


def _str_list(value: str) -> tuple[str, ...]:
    return tuple(item for item in value.split(",") if item)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assets", type=Path, required=True, help="per-cluster asset path JSON")
    parser.add_argument("--output", type=Path, required=True, help="manifest path under /home")
    parser.add_argument("--family", choices=sorted(_TARGET_KINDS), default="q30")
    parser.add_argument("--variants", type=_str_list, default=("base", "thinking"))
    parser.add_argument("--methods", type=_str_list, default=("dflash", "dspark", "dflash2"))
    parser.add_argument("--block-sizes", type=_int_list, default=(8, 16))
    parser.add_argument("--dataset", default="nemo-direct")
    parser.add_argument("--boundaries", type=_int_list, required=True)
    parser.add_argument("--nodes", type=int, required=True)
    parser.add_argument("--account", required=True)
    parser.add_argument("--partition", required=True)
    parser.add_argument("--run-prefix", required=True, help="distinguishes one study from another")
    args = parser.parse_args()

    experiments = build_experiments(
        load_assets(args.assets),
        family=args.family,
        variants=args.variants,
        methods=args.methods,
        block_sizes=args.block_sizes,
        dataset=args.dataset,
        boundaries=args.boundaries,
        nodes=args.nodes,
        account=args.account,
        partition=args.partition,
        run_prefix=args.run_prefix,
    )
    write_manifest(args.output, experiments)

    # The submitter selects arms by their index in the written manifest, and
    # write_manifest reorders experiments into canonical order, so print the
    # stored order rather than the order the matrix was generated in.
    stored = json.loads(args.output.read_text())["experiments"]
    for index, entry in enumerate(stored):
        print(f"{index}\t{entry['experiment_id']}\t{entry['run_name']}")
    print(f"wrote {len(stored)} experiments to {args.output}")


if __name__ == "__main__":
    main()
