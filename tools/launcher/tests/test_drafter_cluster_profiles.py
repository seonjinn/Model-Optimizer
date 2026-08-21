# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Contracts for immutable drafter cluster profiles."""

from __future__ import annotations

import os
import shutil
import subprocess
from dataclasses import replace
from pathlib import Path

import pytest
from common.specdec.cluster_profile import (
    ClusterProfile,
    load_cluster_profile,
    render_probe_sbatch,
    scheduler_gpu_args,
    select_scratch_root,
    validate_scratch_root,
)

_LAUNCHER_DIR = Path(__file__).resolve().parents[1]
PROFILES = _LAUNCHER_DIR / "common/specdec/profiles"
_MODELOPT_PIN = "e3febcbe1319f018eea81fa4d42e2e36cb54494e"
_PROBE = _LAUNCHER_DIR / "common/specdec/probe_cluster_profile.sh"
_BASH = shutil.which("bash")


@pytest.mark.parametrize("profile_path", sorted(PROFILES.glob("*.yaml")))
def test_profiles_pin_the_required_modelopt_commit(profile_path: Path) -> None:
    """Every cluster binds the immutable source revision used for standby qualification."""
    profile = load_cluster_profile(profile_path)

    assert profile.modelopt_commit == _MODELOPT_PIN


def test_profile_rejects_a_different_modelopt_commit() -> None:
    """The standby profile cannot silently advance to another source revision."""
    profile = load_cluster_profile(PROFILES / "oci-hsg.yaml")

    with pytest.raises(ValueError, match="pinned ModelOpt commit"):
        replace(profile, modelopt_commit="a" * 40)


def test_ptyche_profile_uses_exclusive_four_gpu_nodes() -> None:
    """Pre-Tyche relies on an exclusive allocation instead of a GPU request flag."""
    profile = load_cluster_profile(PROFILES / "ptyche.yaml")

    assert profile.account == "coreai_dlalgo_llm"
    assert profile.partition == "36x2-a01r"
    assert profile.fallback_partition == "batch"
    assert profile.training_nodes == 4
    assert profile.training_segment == 4
    assert profile.evaluation_nodes == 1
    assert profile.evaluation_segment == 1
    assert scheduler_gpu_args(profile) == ()


def test_oci_profile_preserves_gpus_per_node() -> None:
    """OCI keeps its compatible explicit four-GPU scheduler request."""
    profile = load_cluster_profile(PROFILES / "oci-hsg.yaml")

    assert profile.durable_root == Path(
        "/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sna/modelopt-specdec"
    )
    assert profile.scratch_candidates == (Path("/raid/scratch"),)
    assert scheduler_gpu_args(profile) == ("--gpus-per-node=4",)


def test_lyris_profile_uses_verified_scratch_candidates() -> None:
    """Lyris supplies only absolute fallback scratch locations after SLURM_TMPDIR."""
    profile = load_cluster_profile(PROFILES / "lyris.yaml")

    assert profile.scratch_candidates == (
        Path("$SLURM_TMPDIR"),
        Path("/raid/scratch"),
        Path("/tmp"),
    )
    assert scheduler_gpu_args(profile) == ()


def test_select_scratch_prefers_slurm_tmpdir(tmp_path: Path) -> None:
    """An allocated node uses SLURM_TMPDIR before static scratch fallbacks."""
    selected = select_scratch_root(
        candidates=(Path("$SLURM_TMPDIR"), Path("/raid/scratch"), Path("/tmp")),
        environ={"SLURM_TMPDIR": str(tmp_path)},
        writable=lambda path: path == tmp_path,
    )

    assert selected == tmp_path


def test_select_scratch_rejects_relative_slurm_tmpdir() -> None:
    """A relative allocation scratch value cannot be made absolute from the login cwd."""
    with pytest.raises(ValueError, match="no configured scratch candidate"):
        select_scratch_root(
            candidates=(Path("$SLURM_TMPDIR"),),
            environ={"SLURM_TMPDIR": "relative-scratch"},
            writable=lambda path: True,
        )


def test_lyris_render_has_no_gres_or_gpu_flag() -> None:
    """Lyris requests its exclusive allocation without incompatible GPU flags."""
    argv = render_probe_sbatch(load_cluster_profile(PROFILES / "lyris.yaml"))

    assert "--segment=1" in argv
    assert not any(arg.startswith(("--gres", "--gpus-per-node")) for arg in argv)


def test_probe_submits_once_after_test_only_when_sbatch_output_is_blank(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A blank scheduler response is recovered without submitting a duplicate probe."""
    command_dir = tmp_path / "bin"
    command_dir.mkdir()
    calls = tmp_path / "sbatch-calls"
    _write_command(command_dir / "sacctmgr", 'printf "coreai_dlalgo_llm|36x2-a01r|\\n"\n')
    _write_command(command_dir / "scontrol", "exit 0\n")
    _write_command(
        command_dir / "sbatch",
        'printf "%s\\n" "$*" >> "$SBATCH_CALLS"\n[[ "$1" == "--test-only" ]] && exit 0\n',
    )
    _write_command(command_dir / "squeue", 'printf "4242\\n"\n')
    monkeypatch.setenv("SBATCH_CALLS", str(calls))
    monkeypatch.setenv("PATH", f"{command_dir}{os.pathsep}{os.environ['PATH']}")

    result = subprocess.run(
        [
            _BASH,
            str(_PROBE),
            "--profile",
            str(PROFILES / "ptyche.yaml"),
            "--output",
            "/lustre/fsw/coreai_dlalgo_llm/users/sna/modelopt-qwen3-drafter-training/readiness.json",
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "4242"
    expected_common = (
        "--account=coreai_dlalgo_llm --partition=36x2-a01r --nodes=1 "
        "--ntasks-per-node=1 --segment=1 --time=00:10:00 "
        "--job-name=drafter-profile-probe-ptyche "
        f"{_PROBE} --inside --profile {PROFILES / 'ptyche.yaml'} "
        "--output /lustre/fsw/coreai_dlalgo_llm/users/sna/"
        "modelopt-qwen3-drafter-training/readiness.json"
    )
    assert calls.read_text().splitlines() == [
        f"--test-only {expected_common}",
        f"--parsable {expected_common}",
    ]


def test_probe_dry_run_stops_after_test_only(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The explicit dry-run gate renders the allocation without submitting a probe job."""
    command_dir = tmp_path / "bin"
    command_dir.mkdir()
    calls = tmp_path / "sbatch-calls"
    _write_command(command_dir / "sacctmgr", 'printf "coreai_dlalgo_llm|36x2-a01r|\\n"\n')
    _write_command(command_dir / "scontrol", "exit 0\n")
    _write_command(command_dir / "sbatch", 'printf "%s\\n" "$*" >> "$SBATCH_CALLS"\n')
    monkeypatch.setenv("SBATCH_CALLS", str(calls))
    monkeypatch.setenv("PATH", f"{command_dir}{os.pathsep}{os.environ['PATH']}")

    result = subprocess.run(
        [
            _BASH,
            str(_PROBE),
            "--dry-run",
            "--profile",
            str(PROFILES / "ptyche.yaml"),
            "--output",
            "/lustre/fsw/coreai_dlalgo_llm/users/sna/modelopt-qwen3-drafter-training/readiness.json",
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert calls.read_text().splitlines()[0].startswith("--test-only ")
    assert len(calls.read_text().splitlines()) == 1


def test_probe_rejects_non_aarch64_compute_node(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """ARM aliases cannot qualify a node whose kernel architecture is not aarch64."""
    profile = tmp_path / "profile.yaml"
    scratch = tmp_path / "scratch"
    durable = tmp_path / "durable"
    profile.write_text(
        "\n".join(
            (
                "name: test",
                f"modelopt_commit: {_MODELOPT_PIN}",
                "ssh_host: login-test",
                "account: account",
                "partition: batch",
                "fallback_partition: null",
                f"durable_root: {durable}",
                "scratch_candidates:",
                f"  - {scratch}",
                "training_nodes: 4",
                "training_segment: 4",
                "evaluation_nodes: 1",
                "evaluation_segment: 1",
                "gpus_per_node: 4",
                "explicit_gpu_flag: false",
                'walltime: "00:10:00"',
                "",
            )
        )
    )
    command_dir = tmp_path / "bin"
    command_dir.mkdir()
    _write_command(command_dir / "nvidia-smi", "printf 'GPU 0\\nGPU 1\\nGPU 2\\nGPU 3\\n'\n")
    _write_command(command_dir / "uname", "printf 'arm64\\n'\n")
    _write_command(command_dir / "srun", "printf '%s\\n' '--container-image'\n")
    monkeypatch.setenv("PATH", f"{command_dir}{os.pathsep}{os.environ['PATH']}")

    result = subprocess.run(
        [
            _BASH,
            str(_PROBE),
            "--inside",
            "--profile",
            str(profile),
            "--output",
            str(durable / "readiness.json"),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode != 0
    assert "expected aarch64 compute node" in result.stderr


def _write_command(path: Path, body: str) -> None:
    path.write_text(f"#!/usr/bin/env bash\nset -euo pipefail\n{body}")
    path.chmod(0o755)


def test_profile_rejects_nonexclusive_gpu_semantics() -> None:
    """A profile cannot request a topology other than four exclusive GPUs per node."""
    with pytest.raises(ValueError, match="four exclusive GPUs"):
        ClusterProfile(
            name="invalid",
            modelopt_commit=_MODELOPT_PIN,
            ssh_host="login-invalid",
            account="account",
            partition="batch",
            fallback_partition=None,
            durable_root=Path("/lustre/durable"),
            scratch_candidates=(Path("/tmp"),),
            training_nodes=4,
            training_segment=4,
            evaluation_nodes=1,
            evaluation_segment=1,
            gpus_per_node=8,
            explicit_gpu_flag=False,
            walltime="5:00:00",
        )


def test_validate_scratch_root_rejects_path_outside_profile_candidates() -> None:
    """A resolved scratch location must derive from an approved profile candidate."""
    profile = load_cluster_profile(PROFILES / "oci-hsg.yaml")

    with pytest.raises(ValueError, match="not an approved scratch root"):
        validate_scratch_root(profile, Path("/tmp/modelopt-drafter"))


def test_validate_scratch_root_rejects_lexical_traversal_outside_candidate() -> None:
    """A lexical parent traversal cannot escape an approved scratch root."""
    profile = load_cluster_profile(PROFILES / "oci-hsg.yaml")

    with pytest.raises(ValueError, match="not an approved scratch root"):
        validate_scratch_root(profile, Path("/raid/scratch/../tmp/modelopt-drafter"))


def test_validate_scratch_root_accepts_resolved_slurm_tmpdir(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A compute-node SLURM_TMPDIR expansion is an approved Lyris scratch root."""
    monkeypatch.setenv("SLURM_TMPDIR", "/private/slurm-tmp/job-123")
    profile = load_cluster_profile(PROFILES / "lyris.yaml")

    validate_scratch_root(profile, Path("/private/slurm-tmp/job-123/modelopt-drafter"))
