# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Canonical, validated job inputs for pinned drafter SLURM workflows."""

from __future__ import annotations

import json
import os
import re
import tempfile
from dataclasses import asdict, dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any

__all__ = [
    "DrafterExperiment",
    "PinnedPaths",
    "SlurmSettings",
    "TargetTopology",
    "canonical_manifest",
    "load_manifest",
    "speculative_tokens",
    "validate_topology",
    "write_manifest",
]

_FULL_SHA = re.compile(r"^[0-9a-f]{40}$")
_FULL_SHA256 = re.compile(r"^[0-9a-f]{64}$")
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


def _normalized_path(name: str, value: str, root: str) -> str:
    if not value or not Path(value).is_absolute():
        raise ValueError(f"{name} must be an absolute path under {root}")
    lexical = Path(os.path.abspath(value))
    if not lexical.resolve(strict=False).is_relative_to(Path(root).resolve(strict=False)):
        raise ValueError(f"{name} must be an absolute path under {root}")
    return str(lexical)


@dataclass(frozen=True)
class PinnedPaths:
    """Immutable source and large-artifact locations recorded for a job."""

    source_path: str
    source_sha: str
    image_path: str
    runtime_archive_path: str
    runtime_archive_sha256: str
    target_path: str
    dataset_path: str
    output_root: str

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "source_path", _normalized_path("source_path", self.source_path, "/home")
        )
        if not _FULL_SHA.fullmatch(self.source_sha):
            raise ValueError("source_sha must be an exact 40-character lowercase commit SHA")
        if not _FULL_SHA256.fullmatch(self.runtime_archive_sha256):
            raise ValueError("runtime_archive_sha256 must be an exact lowercase SHA-256")
        for name in (
            "image_path",
            "runtime_archive_path",
            "target_path",
            "dataset_path",
            "output_root",
        ):
            object.__setattr__(self, name, _normalized_path(name, getattr(self, name), "/lustre"))


@dataclass(frozen=True)
class TargetTopology:
    """Target-specific streaming and batch settings that are safe to execute verbatim."""

    target_kind: str
    capture_ids: tuple[int, ...]
    serve_tp: int
    per_device_train_batch_size: int
    gradient_accumulation_steps: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    intermediate_size: int

    @classmethod
    def for_kind(cls, target_kind: str) -> TargetTopology:
        """Return the only supported topology for a public Qwen3 target family."""
        try:
            return cls(target_kind=target_kind, **_TARGET_DEFAULTS[target_kind])
        except KeyError as error:
            raise ValueError(f"unsupported target kind: {target_kind}") from error

    def __post_init__(self) -> None:
        expected = _TARGET_DEFAULTS.get(self.target_kind)
        if expected is None or any(
            getattr(self, field) != value for field, value in expected.items()
        ):
            raise ValueError(f"target topology must use the pinned defaults for {self.target_kind}")
        if self.per_device_train_batch_size * self.gradient_accumulation_steps * 2 * 4 != 512:
            raise ValueError(
                "target topology must produce global batch size 512 over eight trainer GPUs"
            )


_TARGET_DEFAULTS = {
    "qwen3-30b-a3b": {
        "capture_ids": (2, 13, 24, 35, 46, 48),
        "serve_tp": 2,
        "per_device_train_batch_size": 4,
        "gradient_accumulation_steps": 16,
        "num_attention_heads": 32,
        "num_key_value_heads": 4,
        "head_dim": 128,
        "intermediate_size": 6144,
    },
    "qwen3-235b-a22b": {
        "capture_ids": (2, 25, 47, 69, 92, 94),
        "serve_tp": 4,
        "per_device_train_batch_size": 2,
        "gradient_accumulation_steps": 32,
        "num_attention_heads": 64,
        "num_key_value_heads": 4,
        "head_dim": 128,
        "intermediate_size": 12288,
    },
}


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
    topology: TargetTopology
    paths: PinnedPaths
    slurm: SlurmSettings
    sample_size: int = 1_300_000

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
        if self.sample_size != 1_300_000:
            raise ValueError("production sample_size must be exactly 1,300,000")
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

    @property
    def experiment_id(self) -> str:
        """Return a stable short identifier for receipts and scheduler job names."""
        payload = json.dumps(
            (
                self.target,
                self.dataset,
                self.method,
                self.block_size,
                self.run_name,
                self.paths.output_root,
            ),
            separators=(",", ":"),
        )
        return sha256(payload.encode()).hexdigest()[:16]


def _manifest_entry(experiment: DrafterExperiment) -> dict[str, Any]:
    entry = asdict(experiment)
    entry["experiment_id"] = experiment.experiment_id
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
            sample_size=entry["sample_size"],
            topology=TargetTopology(**entry["topology"]),
            paths=PinnedPaths(**entry["paths"]),
            slurm=SlurmSettings(**entry["slurm"]),
        )
        for entry in raw_experiments
    )
    if canonical_manifest(experiments) != path.read_text():
        raise ValueError("manifest is not canonical")
    return experiments
