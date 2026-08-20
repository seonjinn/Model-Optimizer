# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Canonical, validated job inputs for pinned drafter SLURM workflows."""

from __future__ import annotations

import json
import os
import re
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

__all__ = [
    "DrafterExperiment",
    "PinnedPaths",
    "SlurmSettings",
    "canonical_manifest",
    "load_manifest",
    "speculative_tokens",
    "validate_topology",
    "write_manifest",
]

_FULL_SHA = re.compile(r"^[0-9a-f]{40}$")
_HORIZONS = {("dflash", 8): 7, ("dflash", 16): 15, ("dspark", 8): 8, ("dspark", 16): 16}


def validate_topology(nodes: int, segment: int, role: str) -> None:
    """Validate that an allocation fits OCI-HSG segment placement constraints."""
    if nodes < 1 or segment < 1:
        raise ValueError(f"{role}: nodes and segment must be positive")
    if segment > 18:
        raise ValueError(f"{role}: segment {segment} exceeds the 18-node limit")
    if nodes % segment:
        raise ValueError(f"{role}: segment {segment} must divide nodes {nodes}")


def speculative_tokens(method: str, block_size: int) -> int:
    """Return the pinned evaluator horizon for a supported drafter method."""
    try:
        return _HORIZONS[(method.lower(), block_size)]
    except KeyError as error:
        raise ValueError(f"unsupported method/block-size pair: {method}/{block_size}") from error


def _require_path(name: str, value: str, root: str) -> None:
    if not value or not Path(value).is_absolute() or not Path(value).is_relative_to(root):
        raise ValueError(f"{name} must be an absolute path under {root}")


@dataclass(frozen=True)
class PinnedPaths:
    """Immutable source and large-artifact locations recorded for a job."""

    source_path: str
    source_sha: str
    image_path: str
    runtime_archive_path: str
    target_path: str
    dataset_path: str
    output_root: str

    def __post_init__(self) -> None:
        _require_path("source_path", self.source_path, "/home")
        if not _FULL_SHA.fullmatch(self.source_sha):
            raise ValueError("source_sha must be an exact 40-character lowercase commit SHA")
        for name in (
            "image_path",
            "runtime_archive_path",
            "target_path",
            "dataset_path",
            "output_root",
        ):
            _require_path(name, getattr(self, name), "/lustre")


@dataclass(frozen=True)
class SlurmSettings:
    """Pinned scheduler settings shared by a submitted experiment."""

    account: str
    partition: str
    nodes: int
    gpus_per_node: int
    segment: int

    def __post_init__(self) -> None:
        if not self.account.strip() or not self.partition.strip():
            raise ValueError("account and partition must be non-empty")
        if self.gpus_per_node < 1:
            raise ValueError("gpus_per_node must be positive")
        validate_topology(self.nodes, self.segment, "slurm")


@dataclass(frozen=True)
class DrafterExperiment:
    """One cumulative drafter-training experiment specification."""

    target: str
    dataset: str
    method: str
    block_size: int
    cumulative_max_steps: tuple[int, ...]
    run_name: str
    paths: PinnedPaths
    slurm: SlurmSettings

    def __post_init__(self) -> None:
        if not self.target.strip() or not self.dataset.strip() or not self.run_name.strip():
            raise ValueError("target, dataset, and run_name must be non-empty")
        method = self.method.lower()
        object.__setattr__(self, "method", method)
        speculative_tokens(method, self.block_size)
        if self.slurm.nodes != 4 or self.slurm.segment != 4:
            raise ValueError("training requires exactly four nodes with --segment=4")
        if not self.cumulative_max_steps or any(step < 1 for step in self.cumulative_max_steps):
            raise ValueError("cumulative_max_steps must contain positive boundaries")
        if tuple(sorted(set(self.cumulative_max_steps))).__len__() != len(
            self.cumulative_max_steps
        ):
            raise ValueError("cumulative_max_steps must be strictly increasing")

    @property
    def num_speculative_tokens(self) -> int:
        """Return the evaluator horizon associated with this experiment."""
        return speculative_tokens(self.method, self.block_size)

    @property
    def identity(self) -> tuple[str, str, str, int, int]:
        """Return the unique identity for each submitted cumulative training wave."""
        return (
            self.target,
            self.dataset,
            self.method,
            self.block_size,
            self.cumulative_max_steps[-1],
        )


def _manifest_entry(experiment: DrafterExperiment) -> dict[str, Any]:
    entry = asdict(experiment)
    entry["num_speculative_tokens"] = experiment.num_speculative_tokens
    return entry


def canonical_manifest(experiments: tuple[DrafterExperiment, ...]) -> str:
    """Serialize an experiment matrix as sorted canonical JSON with a trailing newline."""
    identities = [
        (experiment.target, experiment.dataset, experiment.method, experiment.block_size, boundary)
        for experiment in experiments
        for boundary in experiment.cumulative_max_steps
    ]
    if len(set(identities)) != len(identities):
        raise ValueError("duplicate training tuple in manifest")
    entries = sorted(
        (_manifest_entry(experiment) for experiment in experiments),
        key=lambda item: json.dumps(item, sort_keys=True),
    )
    return json.dumps({"experiments": entries}, indent=2, sort_keys=True) + "\n"


def write_manifest(output: Path, experiments: tuple[DrafterExperiment, ...]) -> None:
    """Atomically write a canonical manifest after validation."""
    output.parent.mkdir(parents=True, exist_ok=True)
    content = canonical_manifest(experiments)
    with tempfile.NamedTemporaryFile(
        "w", dir=output.parent, prefix=f".{output.name}.", delete=False
    ) as file:
        file.write(content)
        file.flush()
        os.fsync(file.fileno())
        temporary = Path(file.name)
    try:
        temporary.replace(output)
    finally:
        temporary.unlink(missing_ok=True)


def load_manifest(path: Path) -> tuple[DrafterExperiment, ...]:
    """Load and validate a canonical manifest before submission."""
    try:
        document = json.loads(path.read_text())
        raw_experiments = document["experiments"]
    except (OSError, TypeError, json.JSONDecodeError, KeyError) as error:
        raise ValueError(f"invalid manifest: {path}") from error
    if not isinstance(raw_experiments, list) or not raw_experiments:
        raise ValueError("manifest must contain a non-empty experiments list")
    experiments = tuple(
        DrafterExperiment(
            target=entry["target"],
            dataset=entry["dataset"],
            method=entry["method"],
            block_size=entry["block_size"],
            cumulative_max_steps=tuple(entry["cumulative_max_steps"]),
            run_name=entry["run_name"],
            paths=PinnedPaths(**entry["paths"]),
            slurm=SlurmSettings(**entry["slurm"]),
        )
        for entry in raw_experiments
    )
    if canonical_manifest(experiments) != path.read_text():
        raise ValueError("manifest is not canonical")
    return experiments
