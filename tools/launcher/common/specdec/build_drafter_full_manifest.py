# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Build the pinned 32-run paper-scale drafter training manifest."""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

from common.specdec.drafter_job_manifest import (
    DrafterExperiment,
    legacy_training_fingerprint,
    load_manifest,
    write_manifest,
)


def validate_legacy_seed_identity(
    identity: dict[str, object],
    expected_experiment_id: str,
    expected_fingerprint: str,
    expected_source_sha: str,
) -> None:
    """Reject a legacy checkpoint whose immutable training identity is not selected."""
    actual_source_sha = identity.get("adopted_from_source_sha", identity.get("source_sha"))
    expected = {
        "experiment_id": expected_experiment_id,
        "training_fingerprint": expected_fingerprint,
    }
    if actual_source_sha != expected_source_sha or any(
        identity.get(key) != value for key, value in expected.items()
    ):
        raise ValueError("legacy checkpoint training identity does not match selected experiment")


def legacy_q30_seed_expectations(experiment: DrafterExperiment) -> tuple[str, str, str]:
    """Derive the exact legacy identity and tuple allowed to seed a Q30 ``-2n`` run."""
    suffix = "-2n"
    if (
        experiment.topology.target_kind != "qwen3-30b-a3b"
        or not experiment.run_name.endswith(suffix)
        or not experiment.paths.output_root.endswith(suffix)
    ):
        raise ValueError("legacy seed expectations require a Q30 -2n experiment")
    legacy = replace(
        experiment,
        run_name=experiment.run_name[: -len(suffix)],
        paths=replace(
            experiment.paths,
            output_root=experiment.paths.output_root[: -len(suffix)],
        ),
    )
    selected_tuple = json.dumps(
        (experiment.target, experiment.dataset, experiment.method, experiment.block_size),
        separators=(",", ":"),
    )
    return legacy.experiment_id, legacy_training_fingerprint(legacy), selected_tuple


def full_convergence_boundaries(target_kind: str, block_size: int) -> tuple[int, ...]:
    """Return bounded self-requeue stages ending at the paper-scale optimizer step."""
    if target_kind == "qwen3-30b-a3b" or block_size == 8:
        return (4166, 14500, 25391)
    if target_kind == "qwen3-235b-a22b" and block_size == 16:
        return (4166, 10500, 17000, 23500, 25391)
    raise ValueError(f"unsupported target/block pair: {target_kind}/{block_size}")


def readable_job_name(
    target: str,
    dataset: str,
    method: str,
    block_size: int,
    boundary: int,
    *,
    nodes: int | None = None,
) -> str:
    """Render a unique scheduler name that remains understandable in ``squeue``."""
    targets = {
        "q30-base": "q30b",
        "q30-thinking": "q30t",
        "q235-base": "q235b",
        "q235-thinking": "q235t",
    }
    datasets = {"opb-direct": "opb", "nemo-direct": "nemo"}
    methods = {"dflash": "df", "dspark": "ds"}
    try:
        name = (
            f"dr-{targets[target]}-{datasets[dataset]}-{methods[method]}-b{block_size}-s{boundary}"
        )
    except KeyError as error:
        raise ValueError(f"unsupported readable job identity: {error.args[0]}") from error
    if nodes not in (None, 2, 4):
        name = f"{name}-n{nodes}"
    if len(name) > 64:
        raise ValueError(f"job name exceeds Slurm limit: {name}")
    return name


def select_experiments(
    experiments: tuple[DrafterExperiment, ...], selected_ids: set[str]
) -> tuple[DrafterExperiment, ...]:
    """Select exact manifest identities while preserving canonical manifest order."""
    if not selected_ids:
        return experiments
    available = {experiment.experiment_id for experiment in experiments}
    unknown = selected_ids - available
    if unknown:
        raise ValueError(f"unknown experiment IDs: {','.join(sorted(unknown))}")
    return tuple(
        experiment for experiment in experiments if experiment.experiment_id in selected_ids
    )


def build_full_manifest(
    template: Path,
    output: Path,
    source_path: str,
    source_sha: str,
    image_sha256: str,
    q30_nodes: int = 2,
    q235_nodes: int = 4,
) -> tuple[DrafterExperiment, ...]:
    """Rewrite only source provenance and convergence stages of the trusted template."""

    def rewrite(experiment: DrafterExperiment) -> DrafterExperiment:
        suffix = f"-{q30_nodes}n"
        is_q30 = experiment.topology.target_kind == "qwen3-30b-a3b"
        run_name = experiment.run_name
        output_root = experiment.paths.output_root
        if is_q30 and not run_name.endswith(suffix):
            run_name = f"{run_name}{suffix}"
        if is_q30 and not output_root.endswith(suffix):
            output_root = f"{output_root}{suffix}"
        topology = experiment.topology
        slurm = experiment.slurm
        if is_q30:
            topology = replace(
                topology,
                gradient_accumulation_steps=64 // q30_nodes,
            )
            slurm = replace(slurm, nodes=q30_nodes, segment=q30_nodes)
        elif experiment.topology.target_kind == "qwen3-235b-a22b" and q235_nodes == 16:
            suffix = f"-{q235_nodes}n"
            if not run_name.endswith(suffix):
                run_name = f"{run_name}{suffix}"
            if not output_root.endswith(suffix):
                output_root = f"{output_root}{suffix}"
            topology = replace(
                topology,
                gradient_accumulation_steps=128 // q235_nodes,
            )
            slurm = replace(slurm, nodes=q235_nodes, segment=q235_nodes)
        return replace(
            experiment,
            run_name=run_name,
            topology=topology,
            slurm=slurm,
            cumulative_max_steps=full_convergence_boundaries(
                experiment.topology.target_kind, experiment.block_size
            ),
            paths=replace(
                experiment.paths,
                source_path=source_path,
                source_sha=source_sha,
                output_root=output_root,
                image_sha256=image_sha256,
            ),
        )

    if q30_nodes not in (2, 16):
        raise ValueError("Q30 training nodes must be 2 or 16")
    if q235_nodes not in (4, 16):
        raise ValueError("Q235 training nodes must be 4 or 16")
    experiments = tuple(
        rewrite(experiment)
        for experiment in load_manifest(template, migrate_legacy_q30_topology=True)
    )
    matrix = {
        (experiment.target, experiment.dataset, experiment.method, experiment.block_size)
        for experiment in experiments
    }
    expected = {
        (target, dataset, method, block_size)
        for target in ("q30-base", "q30-thinking", "q235-base", "q235-thinking")
        for dataset in ("opb-direct", "nemo-direct")
        for method in ("dflash", "dspark")
        for block_size in (8, 16)
    }
    if len(experiments) != 32 or matrix != expected:
        raise ValueError("template must contain the exact 32-run target/data/method/block matrix")
    if any(experiment.sample_size != 1_300_000 for experiment in experiments):
        raise ValueError("every full-convergence run must use exactly 1,300,000 samples")
    expected_kinds = {
        "q30-base": "qwen3-30b-a3b",
        "q30-thinking": "qwen3-30b-a3b",
        "q235-base": "qwen3-235b-a22b",
        "q235-thinking": "qwen3-235b-a22b",
    }
    if any(
        experiment.topology.target_kind != expected_kinds[experiment.target]
        for experiment in experiments
    ):
        raise ValueError("target label does not match its pinned topology family")
    expected_slurm = {
        "qwen3-30b-a3b": ("nemotron_n3_post", "batch", q30_nodes, 4, q30_nodes),
        "qwen3-235b-a22b": ("nemotron_n3_post", "batch", q235_nodes, 4, q235_nodes),
    }
    if any(
        (
            experiment.slurm.account,
            experiment.slurm.partition,
            experiment.slurm.nodes,
            experiment.slurm.gpus_per_node,
            experiment.slurm.segment,
        )
        != expected_slurm[experiment.topology.target_kind]
        for experiment in experiments
    ):
        raise ValueError("full convergence Slurm topology must match each target family")
    for label, values in (
        ("output_root", [experiment.paths.output_root for experiment in experiments]),
        ("run_name", [experiment.run_name for experiment in experiments]),
        ("experiment_id", [experiment.experiment_id for experiment in experiments]),
    ):
        if len(values) != len(set(values)):
            raise ValueError(f"full convergence requires unique {label} values")
    write_manifest(output, experiments)
    return experiments


def main() -> None:
    """Parse CLI arguments and write the canonical full-convergence manifest."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--template", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-path", required=True)
    parser.add_argument("--source-sha", required=True)
    parser.add_argument("--image-sha256", required=True)
    parser.add_argument("--q30-nodes", type=int, choices=(2, 16), default=2)
    parser.add_argument("--q235-nodes", type=int, choices=(4, 16), default=4)
    args = parser.parse_args()
    experiments = build_full_manifest(
        args.template,
        args.output,
        args.source_path,
        args.source_sha,
        args.image_sha256,
        args.q30_nodes,
        args.q235_nodes,
    )
    print(f"wrote {len(experiments)} experiments to {args.output}")


if __name__ == "__main__":
    main()
