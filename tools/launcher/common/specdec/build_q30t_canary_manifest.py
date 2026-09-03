# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Build a three-arm plumbing canary manifest for one pinned target.

The from-scratch study compares DFlash, DSpark and DFlash2 on one corpus, but
before any of that is worth scheduling the three architectures have to prove
they start at all on this cluster, at this source commit, in this image, on a
16-node allocation. That is what this manifest is for: one entry per method at
block size 8, a single 20-step boundary, everything else pinned identically so
a failure names the architecture rather than the environment.

Every path is a required flag. Nothing here is defaulted to a cluster location,
because a manifest that guesses an asset path fails at hour three of a queue
wait instead of at submission.

The 32-run historical matrix has its own builder in
`build_drafter_full_manifest.py`; this one deliberately does not reuse it,
because that builder pins the four-target/two-dataset cross product and rejects
DFlash2 outright.
"""

from __future__ import annotations

import argparse
import dataclasses
import re
from pathlib import Path

from common.specdec.drafter_job_manifest import (
    DrafterExperiment,
    PinnedPaths,
    SlurmSettings,
    TargetTopology,
    load_manifest,
    write_manifest,
)

__all__ = ["CANARY_METHODS", "build_canary_experiments"]

# Block size 8 for all three: it is the horizon the prior wave ran, so a
# regression here is attributable to this study's changes rather than to an
# untried block size. DSpark accepts K = B, DFlash and DFlash2 K = B - 1; the
# manifest schema derives that, so it is not restated here.
CANARY_METHODS = ("dflash", "dspark", "dflash2")
# 512 = per_device_train_batch_size * accumulation * trainer ranks, and the
# trainer holds half the allocation while the other half serves the target.
_GLOBAL_BATCH_SIZE = 512
_FULL_SHA = re.compile(r"^[0-9a-f]{40}$")
_FULL_SHA256 = re.compile(r"^[0-9a-f]{64}$")


def _accumulation_steps(nodes: int, per_device_train_batch_size: int) -> int:
    """Return the accumulation that holds global batch size at 512."""
    trainer_ranks = (nodes // 2) * 4
    if trainer_ranks < 1 or _GLOBAL_BATCH_SIZE % (per_device_train_batch_size * trainer_ranks):
        raise ValueError(f"{nodes} nodes cannot hold a global batch size of {_GLOBAL_BATCH_SIZE}")
    return _GLOBAL_BATCH_SIZE // (per_device_train_batch_size * trainer_ranks)


def build_canary_experiments(
    *,
    paths: PinnedPaths,
    slurm: SlurmSettings,
    output_root: Path,
    target_label: str,
    dataset_label: str,
    run_prefix: str,
    canary_steps: int,
    target_kind: str,
    sample_size: int,
    block_sizes: tuple[int, ...],
) -> tuple[DrafterExperiment, ...]:
    """Return one short canary experiment per architecture and block size.

    `paths.output_root` is replaced per arm: the schema requires a unique
    output root per experiment, and two arms sharing one would have the second
    adopt the first's checkpoints.

    Block size is swept rather than pinned because it is the largest single
    cost multiplier in the recipe -- a block-16 step costs roughly twice a
    block-8 one -- so an arm that starts at 8 says nothing about whether 16
    fits in HBM on the same allocation.
    """
    base = TargetTopology.for_kind(target_kind)
    topology = dataclasses.replace(
        base,
        gradient_accumulation_steps=_accumulation_steps(
            slurm.nodes, base.per_device_train_batch_size
        ),
    )
    experiments = []
    for method in CANARY_METHODS:
        for block_size in block_sizes:
            run_name = f"{run_prefix}-{method}-b{block_size}-{slurm.nodes}n"
            experiments.append(
                DrafterExperiment(
                    target=target_label,
                    dataset=dataset_label,
                    method=method,
                    block_size=block_size,
                    cumulative_max_steps=(canary_steps,),
                    run_name=run_name,
                    topology=topology,
                    paths=dataclasses.replace(paths, output_root=str(output_root / run_name)),
                    slurm=slurm,
                    sample_size=sample_size,
                )
            )
    return tuple(experiments)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-path", required=True)
    parser.add_argument("--source-sha", required=True)
    parser.add_argument("--image-path", required=True)
    parser.add_argument("--image-sha256", required=True)
    parser.add_argument("--runtime-archive-path", required=True)
    parser.add_argument("--runtime-archive-sha256", required=True)
    parser.add_argument("--target-path", required=True)
    parser.add_argument("--dataset-path", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--account", required=True)
    parser.add_argument("--partition", required=True)
    parser.add_argument("--nodes", type=int, default=16)
    parser.add_argument("--canary-steps", type=int, default=20)
    parser.add_argument("--block-sizes", default="8")
    parser.add_argument("--target-kind", default="qwen3-30b-a3b")
    parser.add_argument("--sample-size", type=int, default=1_300_000)
    parser.add_argument("--target-label", default="q30-thinking")
    parser.add_argument("--dataset-label", default="ptv23-canary")
    parser.add_argument("--run-prefix", default="q30t-ptv23-canary")
    parser.add_argument("--manifest", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if not _FULL_SHA.match(args.source_sha):
        raise SystemExit("source-sha must be a 40-character commit")
    for digest in (args.image_sha256, args.runtime_archive_sha256):
        if not _FULL_SHA256.match(digest):
            raise SystemExit("artifact digests must be 64-character SHA-256 values")
    try:
        block_sizes = tuple(int(value) for value in args.block_sizes.split(","))
    except ValueError as error:
        raise SystemExit("--block-sizes must be a comma-separated list of integers") from error
    if len(set(block_sizes)) != len(block_sizes):
        raise SystemExit("--block-sizes must not repeat a block size")
    paths = PinnedPaths(
        source_path=args.source_path,
        source_sha=args.source_sha,
        image_path=args.image_path,
        image_sha256=args.image_sha256,
        runtime_archive_path=args.runtime_archive_path,
        runtime_archive_sha256=args.runtime_archive_sha256,
        target_path=args.target_path,
        dataset_path=args.dataset_path,
        # Replaced per arm in build_canary_experiments; a placeholder here
        # would have to satisfy the same validation, so pass the real parent.
        output_root=str(args.output_root),
    )
    slurm = SlurmSettings(
        account=args.account,
        partition=args.partition,
        nodes=args.nodes,
        gpus_per_node=4,
        segment=args.nodes,
    )
    experiments = build_canary_experiments(
        paths=paths,
        slurm=slurm,
        output_root=args.output_root,
        target_label=args.target_label,
        dataset_label=args.dataset_label,
        run_prefix=args.run_prefix,
        canary_steps=args.canary_steps,
        target_kind=args.target_kind,
        sample_size=args.sample_size,
        block_sizes=block_sizes,
    )
    write_manifest(args.manifest, experiments)
    # Print the order the file actually holds: write_manifest sorts entries
    # canonically, and --experiment-index addresses that order, not this one.
    for index, experiment in enumerate(load_manifest(args.manifest)):
        print(f"{index}\t{experiment.method}\t{experiment.experiment_id}\t{experiment.run_name}")


if __name__ == "__main__":
    main()
