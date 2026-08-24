# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Immutable OCI-HSG B-balanced canary manifest and evidence validation."""

from __future__ import annotations

import hashlib
import json
import os
import re
from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING, Any

from common.specdec.qwen4b_b_atomic import atomic_publish_bytes
from common.specdec.qwen4b_b_readiness import load_b_readiness_receipt, validate_builder_artifacts

if TYPE_CHECKING:
    from pathlib import Path

__all__ = [
    "BCanaryEvidence",
    "BCanaryManifest",
    "BCanaryTopology",
    "load_b_canary_manifest",
    "validate_bound_artifacts",
    "validate_canary_evidence",
    "write_b_canary_manifest",
]

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_COMMIT = re.compile(r"^[0-9a-f]{40}$")
_CANARY_OCCURRENCE_COUNT = 102_400


@dataclass(frozen=True)
class BCanaryTopology:
    """The fixed OCI-HSG topology for the B-only bounded runtime screen."""

    cluster: str
    account: str
    partition: str
    nodes: int
    segment: int
    serve_nodes: int
    train_nodes: int
    gpus_per_node: int
    cpu_datamover: int
    per_device_batch_size: int
    gradient_accumulation_steps: int
    max_steps: int
    wandb_project: str

    def __post_init__(self) -> None:
        if self.cluster != "oci-hsg" or self.account not in {
            "nemotron_sw_post",
            "nemotron_n4_post",
        }:
            raise ValueError("B canary is restricted to approved OCI-HSG accounts")
        if self.partition != "batch":
            raise ValueError("B canary requires the OCI-HSG batch partition")
        if (self.nodes, self.segment, self.serve_nodes, self.train_nodes) != (16, 16, 8, 8):
            raise ValueError(
                "B canary requires 16 nodes split into eight serve and eight train nodes"
            )
        if self.gpus_per_node != 4 or self.cpu_datamover != 96:
            raise ValueError("B canary requires four GPUs per node and cpu_datamover=96")
        if (self.per_device_batch_size, self.gradient_accumulation_steps, self.max_steps) != (
            4,
            4,
            200,
        ):
            raise ValueError("B canary requires PDB4, GA4, and 200 steps")
        if self.wandb_project != "sna-qwen3-4b-dataset-study":
            raise ValueError("B canary requires the study W&B project")


@dataclass(frozen=True)
class BCanaryManifest:
    """All identity and resource facts needed to run the bounded B canary."""

    readiness_receipt_sha256: str
    builder_receipt_sha256: str
    builder_output_sha256: str
    canary_occurrence_count: int
    topology: BCanaryTopology
    source_commit: str
    scientific_milestone: bool = False

    def __post_init__(self) -> None:
        for label, digest in (
            ("readiness receipt", self.readiness_receipt_sha256),
            ("builder receipt", self.builder_receipt_sha256),
            ("builder output", self.builder_output_sha256),
        ):
            if _SHA256.fullmatch(digest) is None:
                raise ValueError(f"B {label} must have a SHA-256 identity")
        if self.canary_occurrence_count != _CANARY_OCCURRENCE_COUNT:
            raise ValueError(
                f"B canary must contain exactly {_CANARY_OCCURRENCE_COUNT} occurrences"
            )
        if _COMMIT.fullmatch(self.source_commit) is None:
            raise ValueError("B canary source commit must be exact")
        if self.scientific_milestone:
            raise ValueError("a 200-step canary cannot be a scientific milestone")

    @property
    def global_batch_size(self) -> int:
        """Compute GBS from the 32 trainer ranks, PDB4, and GA4."""
        return (
            self.topology.train_nodes
            * self.topology.gpus_per_node
            * self.topology.per_device_batch_size
            * self.topology.gradient_accumulation_steps
        )

    @property
    def active_gpu_ranks(self) -> int:
        """Return all allocated serve plus trainer GPU ranks."""
        return self.topology.nodes * self.topology.gpus_per_node

    def as_dict(self) -> dict[str, Any]:
        """Return the canonical JSON-compatible manifest body."""
        return asdict(self)


@dataclass(frozen=True)
class BCanaryEvidence:
    """Runtime facts that must be observed before the canary receipt is accepted."""

    finite_loss: bool
    checkpoint_reloaded: bool
    drafter_exported: bool
    evaluator_completed: bool
    active_gpu_ranks: tuple[int, ...]
    scientific_milestone: bool


def write_b_canary_manifest(path: Path, manifest: BCanaryManifest) -> None:
    """Atomically write an immutable manifest with its body digest."""
    if path.exists():
        raise FileExistsError(f"B canary manifest already exists: {path}")
    payload = manifest.as_dict()
    payload["manifest_sha256"] = _sha256_json(payload)
    atomic_publish_bytes(
        path,
        (json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n").encode(),
        job_id=str(os.getpid()),
    )


def load_b_canary_manifest(path: Path) -> BCanaryManifest:
    """Load and verify a canonical B canary manifest."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("B canary manifest must be a JSON object")
    digest = payload.pop("manifest_sha256", None)
    if not isinstance(digest, str) or digest != _sha256_json(payload):
        raise ValueError("B canary manifest identity mismatch")
    try:
        topology = payload.pop("topology")
        if not isinstance(topology, dict):
            raise ValueError("B canary topology must be an object")
        return BCanaryManifest(topology=BCanaryTopology(**topology), **payload)
    except TypeError as error:
        raise ValueError("invalid B canary manifest fields") from error


def validate_canary_evidence(evidence: BCanaryEvidence, manifest: BCanaryManifest) -> None:
    """Reject a canary that lacks required runtime, export, or evaluator evidence."""
    if not evidence.finite_loss:
        raise ValueError("canary evidence lacks finite_loss")
    if not evidence.checkpoint_reloaded:
        raise ValueError("canary evidence lacks checkpoint_reloaded")
    if not evidence.drafter_exported:
        raise ValueError("canary evidence lacks drafter_exported")
    if not evidence.evaluator_completed:
        raise ValueError("canary evidence lacks evaluator_completed")
    if tuple(sorted(evidence.active_gpu_ranks)) != tuple(range(manifest.active_gpu_ranks)):
        raise ValueError("canary evidence lacks all_64_gpus_active")
    if evidence.scientific_milestone:
        raise ValueError("canary evidence cannot set scientific_milestone")


def validate_bound_artifacts(
    manifest: BCanaryManifest,
    *,
    readiness_path: Path,
    builder_receipt_path: Path,
    builder_output_path: Path,
) -> None:
    """Revalidate readiness and builder bytes immediately before GPU work."""
    readiness = load_b_readiness_receipt(readiness_path)
    claim = readiness.pop("receipt_sha256", None)
    if (
        claim != manifest.readiness_receipt_sha256
        or claim != _sha256_json(readiness)
        or readiness.get("ready") is not True
        or readiness.get("builder_receipt_sha256") != manifest.builder_receipt_sha256
        or readiness.get("builder_output_sha256") != manifest.builder_output_sha256
        or readiness.get("source_commit") != manifest.source_commit
    ):
        raise ValueError("B readiness transitive identity mismatch")
    validate_builder_artifacts(
        receipt_path=builder_receipt_path,
        receipt_sha256=manifest.builder_receipt_sha256,
        output_path=builder_output_path,
        output_sha256=manifest.builder_output_sha256,
        source_commit=manifest.source_commit,
    )


def _sha256_json(payload: dict[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
