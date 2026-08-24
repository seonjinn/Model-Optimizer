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

"""Build the Q30/Q235 Base Nemotron DFlash2 B8 training manifest."""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

from common.specdec.drafter_job_manifest import (
    DFlash2RuntimeContract,
    DrafterExperiment,
    load_manifest,
    write_manifest,
)

__all__ = ["build_dflash2_nemotron_manifest"]


def build_dflash2_nemotron_manifest(
    template: Path,
    output: Path,
    source_path: str,
    source_sha: str,
    image_sha256: str,
    vllm_expected_commit: str,
    *,
    q30_nodes: int = 16,
    q235_nodes: int = 16,
) -> tuple[DrafterExperiment, ...]:
    """Build the two-model Base/Nemotron DFlash2 B8 training manifest."""
    if q30_nodes not in (2, 16):
        raise ValueError("Q30 DFlash2 training nodes must be 2 or 16")
    if q235_nodes not in (4, 16):
        raise ValueError("Q235 DFlash2 training nodes must be 4 or 16")

    seeds = tuple(
        experiment
        for experiment in load_manifest(template, migrate_legacy_q30_topology=True)
        if (
            experiment.target in {"q30-base", "q235-base"}
            and experiment.dataset == "nemo-direct"
            and experiment.method == "dflash"
            and experiment.block_size == 8
        )
    )
    if {experiment.target for experiment in seeds} != {"q30-base", "q235-base"} or len(seeds) != 2:
        raise ValueError("template must contain the exact Q30/Q235 Base Nemotron DFlash B8 seeds")

    contract = DFlash2RuntimeContract(vllm_expected_commit=vllm_expected_commit)

    def rewrite(experiment: DrafterExperiment) -> DrafterExperiment:
        nodes = q30_nodes if experiment.target == "q30-base" else q235_nodes
        accumulation = (64 if experiment.target == "q30-base" else 128) // nodes
        suffix = f"-dflash2-{nodes}n"
        return replace(
            experiment,
            method="dflash2",
            run_name=f"{experiment.run_name}{suffix}",
            topology=replace(
                experiment.topology,
                gradient_accumulation_steps=accumulation,
            ),
            paths=replace(
                experiment.paths,
                source_path=source_path,
                source_sha=source_sha,
                image_sha256=image_sha256,
                output_root=f"{experiment.paths.output_root}{suffix}",
            ),
            slurm=replace(experiment.slurm, nodes=nodes, segment=nodes),
            dflash2=contract,
        )

    experiments = tuple(rewrite(experiment) for experiment in seeds)
    write_manifest(output, experiments)
    return experiments


def main() -> None:
    """Parse trusted inputs and write the canonical DFlash2 manifest."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--template", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-path", required=True)
    parser.add_argument("--source-sha", required=True)
    parser.add_argument("--image-sha256", required=True)
    parser.add_argument("--vllm-expected-commit", required=True)
    parser.add_argument("--q30-nodes", type=int, choices=(2, 16), default=16)
    parser.add_argument("--q235-nodes", type=int, choices=(4, 16), default=16)
    args = parser.parse_args()
    experiments = build_dflash2_nemotron_manifest(
        args.template,
        args.output,
        args.source_path,
        args.source_sha,
        args.image_sha256,
        args.vllm_expected_commit,
        q30_nodes=args.q30_nodes,
        q235_nodes=args.q235_nodes,
    )
    print(f"wrote {len(experiments)} DFlash2 experiments to {args.output}")


if __name__ == "__main__":
    main()
