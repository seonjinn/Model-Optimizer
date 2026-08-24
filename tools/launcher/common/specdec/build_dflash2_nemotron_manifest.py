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

"""Build a Q30/Q235 target-variant Nemotron DFlash2 B8 manifest."""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

from common.specdec.cluster_profile import load_cluster_profile
from common.specdec.dflash2_runtime_contract import validate_dflash2_source_checkout
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
    vllm_receipt_sha256: str,
    *,
    cluster_profile: Path,
    q30_nodes: int = 16,
    q235_nodes: int = 16,
    target_variant: str = "base",
) -> tuple[DrafterExperiment, ...]:
    """Build one exact two-model Base or Thinking DFlash2 B8 matrix."""
    if q30_nodes != 16 or q235_nodes != 16:
        raise ValueError("DFlash2 production requires native 16-node/segment-16 jobs")
    if target_variant not in {"base", "thinking"}:
        raise ValueError("DFlash2 target variant must be base or thinking")
    expected_targets = {f"q30-{target_variant}", f"q235-{target_variant}"}
    profile = load_cluster_profile(cluster_profile)
    if profile.modelopt_feature_base is None:
        if profile.modelopt_commit != source_sha:
            raise ValueError("DFlash2 profile source commit does not match the manifest")
    else:
        validate_dflash2_source_checkout(
            Path(source_path), source_sha, profile.modelopt_feature_base
        )
    if profile.training_nodes != 16 or profile.training_segment != 16 or profile.gpus_per_node != 4:
        raise ValueError("DFlash2 profile must expose native 16-node/segment-16 training")

    seeds = tuple(
        experiment
        for experiment in load_manifest(template, migrate_legacy_q30_topology=True)
        if (
            experiment.target in expected_targets
            and experiment.dataset == "nemo-direct"
            and experiment.method == "dflash"
            and experiment.block_size == 8
        )
    )
    if {experiment.target for experiment in seeds} != expected_targets or len(seeds) != 2:
        label = target_variant.capitalize()
        raise ValueError(
            f"template must contain the exact Q30/Q235 {label} Nemotron DFlash B8 seeds"
        )

    contract = DFlash2RuntimeContract(
        vllm_expected_commit=vllm_expected_commit,
        vllm_receipt_sha256=vllm_receipt_sha256,
    )

    def rewrite(experiment: DrafterExperiment) -> DrafterExperiment:
        is_q30 = experiment.target.startswith("q30-")
        nodes = q30_nodes if is_q30 else q235_nodes
        accumulation = (64 if is_q30 else 128) // nodes
        suffix = f"-dflash2-{nodes}n"
        run_name = f"{experiment.run_name}{suffix}"
        if target_variant == "thinking":
            family = "q30t" if is_q30 else "q235t"
            run_name = f"dfl2-{family}-nemo1p3m-b8k7-n16"
        return replace(
            experiment,
            method="dflash2",
            cumulative_max_steps=(20, 4166, 14500, 25391),
            run_name=run_name,
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
            slurm=replace(
                experiment.slurm,
                account=profile.account,
                partition=profile.partition,
                nodes=nodes,
                segment=nodes,
            ),
            dflash2=contract,
        )

    experiments = tuple(rewrite(experiment) for experiment in seeds)
    if len({experiment.paths.output_root for experiment in experiments}) != len(experiments):
        raise ValueError("DFlash2 experiments require unique output roots")
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
    parser.add_argument("--vllm-receipt-sha256", required=True)
    parser.add_argument("--cluster-profile", type=Path, required=True)
    parser.add_argument("--target-variant", choices=("base", "thinking"), default="base")
    args = parser.parse_args()
    experiments = build_dflash2_nemotron_manifest(
        args.template,
        args.output,
        args.source_path,
        args.source_sha,
        args.image_sha256,
        args.vllm_expected_commit,
        args.vllm_receipt_sha256,
        cluster_profile=args.cluster_profile,
        target_variant=args.target_variant,
    )
    print(f"wrote {len(experiments)} DFlash2 experiments to {args.output}")


if __name__ == "__main__":
    main()
