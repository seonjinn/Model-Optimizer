# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Contracts for immutable drafter cluster profiles."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest
from common.specdec.cluster_profile import (
    ClusterProfile,
    load_cluster_profile,
    scheduler_gpu_args,
    validate_scratch_root,
)

_LAUNCHER_DIR = Path(__file__).resolve().parents[1]
PROFILES = _LAUNCHER_DIR / "common/specdec/profiles"
_MODELOPT_PIN = "e3febcbe1319f018eea81fa4d42e2e36cb54494e"


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
