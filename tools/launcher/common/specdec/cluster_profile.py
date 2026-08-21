# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Load and validate immutable scheduler profiles for drafter workflows."""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

__all__ = [
    "ClusterProfile",
    "load_cluster_profile",
    "scheduler_gpu_args",
    "validate_scratch_root",
]

_MAX_SEGMENT_NODES = 18
_SLURM_TMPDIR = Path("$SLURM_TMPDIR")
_COMMIT_SHA = re.compile(r"^[0-9a-f]{40}$")
_MODELOPT_COMMIT = "e3febcbe1319f018eea81fa4d42e2e36cb54494e"
_REQUIRED_FIELDS = frozenset(
    {
        "name",
        "modelopt_commit",
        "ssh_host",
        "account",
        "partition",
        "fallback_partition",
        "durable_root",
        "scratch_candidates",
        "training_nodes",
        "training_segment",
        "evaluation_nodes",
        "evaluation_segment",
        "gpus_per_node",
        "explicit_gpu_flag",
        "walltime",
    }
)


@dataclass(frozen=True)
class ClusterProfile:
    """Immutable scheduler and filesystem settings for one supported cluster."""

    name: str
    modelopt_commit: str
    ssh_host: str
    account: str
    partition: str
    fallback_partition: str | None
    durable_root: Path
    scratch_candidates: tuple[Path, ...]
    training_nodes: int
    training_segment: int
    evaluation_nodes: int
    evaluation_segment: int
    gpus_per_node: int
    explicit_gpu_flag: bool
    walltime: str

    def __post_init__(self) -> None:
        for field in ("name", "ssh_host", "account", "partition", "walltime"):
            if not getattr(self, field).strip():
                raise ValueError(f"{field} must be non-empty")
        if not _COMMIT_SHA.fullmatch(self.modelopt_commit):
            raise ValueError("modelopt_commit must be an exact 40-character lowercase commit SHA")
        if self.modelopt_commit != _MODELOPT_COMMIT:
            raise ValueError("profiles must use the pinned ModelOpt commit")
        if self.fallback_partition is not None and not self.fallback_partition.strip():
            raise ValueError("fallback_partition must be non-empty when provided")
        if not self.durable_root.is_absolute():
            raise ValueError("durable_root must be an absolute path")
        if not self.scratch_candidates:
            raise ValueError("scratch_candidates must not be empty")
        if any(
            candidate != _SLURM_TMPDIR and not candidate.is_absolute()
            for candidate in self.scratch_candidates
        ):
            raise ValueError("scratch_candidates must be absolute paths or $SLURM_TMPDIR")
        if self.gpus_per_node != 4:
            raise ValueError("profiles require four exclusive GPUs per node")
        for role, nodes, segment in (
            ("training", self.training_nodes, self.training_segment),
            ("evaluation", self.evaluation_nodes, self.evaluation_segment),
        ):
            _validate_topology(role, nodes, segment)


def load_cluster_profile(path: Path) -> ClusterProfile:
    """Load one complete cluster profile from a YAML mapping."""
    with path.open() as file:
        values = yaml.safe_load(file)
    if not isinstance(values, dict) or set(values) != _REQUIRED_FIELDS:
        raise ValueError(f"profile must contain exactly {sorted(_REQUIRED_FIELDS)}")
    return ClusterProfile(
        name=_string(values, "name"),
        modelopt_commit=_string(values, "modelopt_commit"),
        ssh_host=_string(values, "ssh_host"),
        account=_string(values, "account"),
        partition=_string(values, "partition"),
        fallback_partition=_optional_string(values, "fallback_partition"),
        durable_root=_absolute_path(values, "durable_root"),
        scratch_candidates=_scratch_candidates(values),
        training_nodes=_positive_integer(values, "training_nodes"),
        training_segment=_positive_integer(values, "training_segment"),
        evaluation_nodes=_positive_integer(values, "evaluation_nodes"),
        evaluation_segment=_positive_integer(values, "evaluation_segment"),
        gpus_per_node=_positive_integer(values, "gpus_per_node"),
        explicit_gpu_flag=_boolean(values, "explicit_gpu_flag"),
        walltime=_string(values, "walltime"),
    )


def scheduler_gpu_args(profile: ClusterProfile) -> tuple[str, ...]:
    """Return the profile's compatible explicit scheduler GPU request, if any."""
    if profile.explicit_gpu_flag:
        return (f"--gpus-per-node={profile.gpus_per_node}",)
    return ()


def validate_scratch_root(profile: ClusterProfile, path: Path) -> None:
    """Reject a selected scratch path that is not rooted in the profile candidates."""
    if not path.is_absolute():
        raise ValueError("scratch root must be an absolute path")
    resolved_path = path.resolve(strict=False)
    if any(
        resolved_path.is_relative_to(candidate)
        for candidate in _resolved_scratch_candidates(profile)
    ):
        return
    raise ValueError(f"scratch root is not an approved scratch root for {profile.name}")


def _validate_topology(role: str, nodes: int, segment: int) -> None:
    if nodes < 1 or segment < 1:
        raise ValueError(f"{role}: nodes and segment must be positive")
    if segment > _MAX_SEGMENT_NODES:
        raise ValueError(f"{role}: segment {segment} exceeds the {_MAX_SEGMENT_NODES}-node limit")
    if nodes % segment:
        raise ValueError(f"{role}: segment {segment} must divide nodes {nodes}")


def _resolved_scratch_candidates(profile: ClusterProfile) -> tuple[Path, ...]:
    candidates: list[Path] = []
    for candidate in profile.scratch_candidates:
        expanded = Path(os.path.expandvars(str(candidate)))
        if expanded.is_absolute():
            candidates.append(expanded.resolve(strict=False))
    return tuple(candidates)


def _string(values: dict[str, Any], field: str) -> str:
    value = values[field]
    if not isinstance(value, str):
        raise ValueError(f"{field} must be a string")
    return value


def _optional_string(values: dict[str, Any], field: str) -> str | None:
    value = values[field]
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError(f"{field} must be a string or null")
    return value


def _absolute_path(values: dict[str, Any], field: str) -> Path:
    value = _string(values, field)
    path = Path(value)
    if not path.is_absolute():
        raise ValueError(f"{field} must be an absolute path")
    return path


def _scratch_candidates(values: dict[str, Any]) -> tuple[Path, ...]:
    raw_candidates = values["scratch_candidates"]
    if not isinstance(raw_candidates, list) or not raw_candidates:
        raise ValueError("scratch_candidates must be a non-empty list")
    candidates = tuple(Path(value) for value in raw_candidates if isinstance(value, str))
    if len(candidates) != len(raw_candidates):
        raise ValueError("scratch_candidates must contain only strings")
    return candidates


def _positive_integer(values: dict[str, Any], field: str) -> int:
    value = values[field]
    if not isinstance(value, int) or isinstance(value, bool) or value < 1:
        raise ValueError(f"{field} must be a positive integer")
    return value


def _boolean(values: dict[str, Any], field: str) -> bool:
    value = values[field]
    if not isinstance(value, bool):
        raise ValueError(f"{field} must be a boolean")
    return value
