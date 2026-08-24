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

"""Fail-closed contracts for Q30/Q235 Nemotron DFlash2 training."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest
from common.specdec.build_dflash2_nemotron_manifest import build_dflash2_nemotron_manifest
from common.specdec.dflash2_runtime_contract import verify_vllm_checkout
from common.specdec.drafter_job_manifest import (
    DrafterExperiment,
    PinnedPaths,
    SlurmSettings,
    TargetTopology,
    load_manifest,
    write_manifest,
)

_SHA256 = "d" * 64
_SOURCE_SHA = "c" * 40


def _git(repo: Path, *arguments: str) -> str:
    completed = subprocess.run(
        ["git", "-c", "commit.gpgsign=false", "-C", str(repo), *arguments],
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def _vllm_checkout(tmp_path: Path) -> tuple[Path, str, str]:
    repo = tmp_path / "vllm"
    repo.mkdir()
    _git(repo, "init", "-q")
    _git(repo, "config", "user.name", "Test User")
    _git(repo, "config", "user.email", "test@example.com")
    package = repo / "vllm"
    package.mkdir()
    (package / "__init__.py").write_text('__version__ = "test"\n')
    _git(repo, "add", "vllm/__init__.py")
    _git(repo, "commit", "-q", "-m", "required DFlash2 runtime")
    required = _git(repo, "rev-parse", "HEAD")
    (package / "runtime.py").write_text("SUPPORTED = True\n")
    _git(repo, "add", "vllm/runtime.py")
    _git(repo, "commit", "-q", "-m", "verified descendant")
    return package, required, _git(repo, "rev-parse", "HEAD")


def test_vllm_runtime_accepts_the_exact_pinned_descendant(tmp_path: Path) -> None:
    """The runtime checkout must match the manifest head and contain PR 52816."""
    package, required, expected = _vllm_checkout(tmp_path)

    assert verify_vllm_checkout(package, expected, required) == expected


def test_vllm_runtime_rejects_a_manifest_head_mismatch(tmp_path: Path) -> None:
    """A different checkout cannot reuse another image's DFlash2 attestation."""
    package, required, _expected = _vllm_checkout(tmp_path)

    with pytest.raises(ValueError, match="expected commit"):
        verify_vllm_checkout(package, "a" * 40, required)


def test_vllm_runtime_rejects_a_missing_required_ancestor(tmp_path: Path) -> None:
    """An exact runtime head without the merged DFlash2 serving change is unsafe."""
    package, _required, expected = _vllm_checkout(tmp_path)

    with pytest.raises(ValueError, match="required DFlash2 commit"):
        verify_vllm_checkout(package, expected, "b" * 40)


def test_vllm_runtime_rejects_a_non_git_install(tmp_path: Path) -> None:
    """Version strings alone cannot prove that PR 52816 is present."""
    package = tmp_path / "site-packages" / "vllm"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text('__version__ = "0.20.0"\n')

    with pytest.raises(ValueError, match="Git checkout"):
        verify_vllm_checkout(package, "a" * 40, "b" * 40)


def _template_experiment(target: str) -> DrafterExperiment:
    is_q30 = target == "q30-base"
    kind = "qwen3-30b-a3b" if is_q30 else "qwen3-235b-a22b"
    nodes = 2 if is_q30 else 4
    return DrafterExperiment(
        target=target,
        dataset="nemo-direct",
        method="dflash",
        block_size=8,
        cumulative_max_steps=(4166, 14500, 25391),
        run_name=f"{target}-nemo-dflash-b8",
        topology=TargetTopology.for_kind(kind),
        paths=PinnedPaths(
            source_path="/home/user/ModelOpt",
            source_sha="a" * 40,
            image_path="/lustre/images/vllm.sqsh",
            image_sha256="b" * 64,
            runtime_archive_path="/lustre/runtimes/modelopt.tar.zst",
            runtime_archive_sha256="e" * 64,
            target_path=f"/lustre/models/{target}",
            dataset_path="/lustre/datasets/nemotron-1.3m.parquet",
            output_root=f"/lustre/results/{target}-nemo-dflash-b8",
        ),
        slurm=SlurmSettings(
            account="nemotron_n3_post",
            partition="batch",
            nodes=nodes,
            gpus_per_node=4,
            segment=nodes,
        ),
    )


def test_builder_emits_exact_q30_q235_base_nemotron_dflash2_matrix(tmp_path: Path) -> None:
    """One shared builder pins both Base targets, exact dims, B8/K7, and runtime proof."""
    template = tmp_path / "template.json"
    output = tmp_path / "dflash2.json"
    write_manifest(
        template,
        (_template_experiment("q30-base"), _template_experiment("q235-base")),
    )

    experiments = build_dflash2_nemotron_manifest(
        template,
        output,
        "/home/user/ModelOpt-dflash2",
        _SOURCE_SHA,
        _SHA256,
        "f" * 40,
    )

    assert load_manifest(output) == experiments
    assert {(item.target, item.dataset, item.method, item.block_size) for item in experiments} == {
        ("q30-base", "nemo-direct", "dflash2", 8),
        ("q235-base", "nemo-direct", "dflash2", 8),
    }
    expected = {
        "q30-base": (32, 4, 128, 6144, 4),
        "q235-base": (64, 4, 128, 12288, 8),
    }
    for experiment in experiments:
        topology = experiment.topology
        assert (
            topology.num_attention_heads,
            topology.num_key_value_heads,
            topology.head_dim,
            topology.intermediate_size,
            topology.gradient_accumulation_steps,
        ) == expected[experiment.target]
        assert experiment.slurm.nodes == 16
        assert experiment.num_speculative_tokens == 7
        assert experiment.dflash2 is not None
        assert experiment.dflash2.vllm_expected_commit == "f" * 40
        assert experiment.paths.source_sha == _SOURCE_SHA
        assert experiment.paths.image_sha256 == _SHA256


def test_builder_rejects_a_template_without_both_base_nemotron_seeds(tmp_path: Path) -> None:
    """Thinking, OPB, or incomplete templates cannot silently redefine the study."""
    template = tmp_path / "template.json"
    write_manifest(template, (_template_experiment("q30-base"),))

    with pytest.raises(ValueError, match="exact Q30/Q235 Base Nemotron DFlash B8 seeds"):
        build_dflash2_nemotron_manifest(
            template,
            tmp_path / "output.json",
            "/home/user/ModelOpt-dflash2",
            _SOURCE_SHA,
            _SHA256,
            "f" * 40,
        )


def test_shared_runner_enforces_and_consumes_the_dflash2_contract() -> None:
    """The manifest contract reaches runtime verification and the training recipe."""
    runner = (
        Path(__file__).resolve().parents[1] / "common/specdec/run_drafter_training.sbatch"
    ).read_text()

    for required in (
        "dflash2:8)",
        '[[ "$DFLASH2_WARMSTART_POLICY" == exact-dflash2-only ]]',
        '[[ "$DFLASH2_KERNEL_PROJECTION_INIT" == zero ]]',
        '[[ "$DFLASH2_MAX_SPECULATIVE_TOKENS" == 7 ]]',
        "verify_vllm_checkout",
        '"dflash.dflash_architecture_config.projector_type=dflash2"',
        '"dflash.dflash_architecture_config.conv_kernel_size=${DFLASH2_CONV_KERNEL_SIZE}"',
        '"dflash.dflash_architecture_config.conv_group_size=${DFLASH2_CONV_GROUP_SIZE}"',
        '"dflash.dflash_architecture_config.selector_rank=${DFLASH2_SELECTOR_RANK}"',
        '"dflash.dflash_architecture_config.selector_top_k=${DFLASH2_SELECTOR_TOP_K}"',
    ):
        assert required in runner
