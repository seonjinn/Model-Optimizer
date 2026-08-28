# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Validate the immutable CPU-only Q30 row-schema launcher boundary."""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RUNNER = ROOT / "launcher/common/specdec/run_q30t_row_schema_observation.sbatch"
SUBMITTER = ROOT / "launcher/common/specdec/submit_q30t_row_schema_observation.sh"


def test_observation_runner_uses_immutable_git_and_cpu_only() -> None:
    """The runner must materialize commit objects without requesting GPUs."""
    runner = RUNNER.read_text()

    assert "#SBATCH --nodes=1" in runner
    assert "#SBATCH --ntasks=1" in runner
    assert "#SBATCH --cpus-per-task=32" in runner
    assert "--gpus" not in runner
    assert "git cat-file" in runner
    assert "git archive" not in runner
    assert "examples/dataset/observe_q30t_ptv23_row_schemas.py" in runner
    assert "examples/dataset/qwen3_30ba3b_thinking_ptv3_stage_subset_v1.json" in runner
    assert '"$source_path/examples/' not in runner


def test_observation_runner_authenticates_the_approved_python_in_a_sterile_environment() -> None:
    """The runner must execute only the approved Python in an isolated environment."""
    runner = RUNNER.read_text()

    assert 'readonly approved_python="/usr/bin/python3.12"' in runner
    assert "sys.version_info >= (3, 12)" in runner
    assert "/usr/bin/env -i" in runner
    assert "PYTHONSAFEPATH=1" in runner
    assert "PYTHONDONTWRITEBYTECODE=1" in runner
    assert "Q30T_TEST_" not in runner
    assert "command -v" not in runner


def test_observation_runner_binds_exact_stage_roots_and_metadata() -> None:
    """The runner must pin both staged roots and all known metadata digests."""
    runner = RUNNER.read_text()

    assert "q30t-ptv2-full201-source-v1" in runner
    assert "q30t-ptv3-source-stage-75209087-v1" in runner
    assert "96970541d0c6f5c99e74b9222b805d4a0bd2ac682837b0ad92b8bf54f7d71a3a" in runner
    assert "018d659170834b17967dd6c1b066e3b03e7859386eceefd9d4409dc3bc8f48c1" in runner
    assert "752090878ed2c5fce47683b2939f9d857be5549311b158476fce33bb79f52eac" in runner
    assert "--ptv2-plan-sha256" in runner
    assert "--ptv2-completion-sha256" in runner
    assert "--ptv3-plan-sha256" in runner
    assert "--ptv3-completion-sha256" in runner
    assert "O_NONBLOCK" in runner
    assert '"$approved_sha256sum" "$ptv3_completion"' not in runner


def test_observation_runner_leaves_private_scratch_to_scheduler_cleanup() -> None:
    """The runner must never recursively clean or unlink shared paths."""
    runner = RUNNER.read_text()

    assert "/raid/scratch" in runner
    assert "mktemp -d" in runner
    assert "umask 077" in runner
    assert "rm -" not in runner
    assert "unlink" not in runner
    assert "shutil.rmtree" not in runner


def test_submitter_requires_clean_pushed_source_and_test_only_preflight() -> None:
    """The submitter must authenticate remote source and preflight every job."""
    submitter = SUBMITTER.read_text()

    assert "status --porcelain=v1 --untracked-files=all" in submitter
    assert "@{upstream}^{commit}" in submitter
    assert "ls-remote --exit-code" in submitter
    assert "reviewed source commit is not pushed" in submitter
    assert "source checkout is not clean" in submitter
    assert 'if [[ "$mode" == "--test-only" ]]' in submitter
    assert "--test-only" in submitter
    assert 'if [[ "$mode" == "--submit" ]]' in submitter


def test_submitter_rejects_path_and_environment_command_substitution() -> None:
    """The submitter must use fixed absolute executables under a sterile env."""
    submitter = SUBMITTER.read_text()

    assert 'readonly approved_git="/usr/bin/git"' in submitter
    assert 'readonly approved_sbatch="/usr/bin/sbatch"' in submitter
    assert "/usr/bin/env -i" in submitter
    assert "Q30T_TEST_" not in submitter
    assert "command -v" not in submitter
    assert "eval " not in submitter
    assert "bash -c" not in submitter


def test_submitter_exposes_only_the_approved_interface() -> None:
    """The submitter must expose only mode, source, output, and log bindings."""
    submitter = SUBMITTER.read_text()

    assert (
        "(--test-only|--submit) --source-path PATH --output PATH --slurm-output PATH" in submitter
    )
    assert "--source-path" in submitter
    assert "--output" in submitter
    assert "--slurm-output" in submitter
    assert "--export=NONE" in submitter
