# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Fail-closed manifest and evidence for the genuine A-repair canary."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import stat
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from common.specdec.qwen4b_b_atomic import atomic_publish_bytes

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_COMMIT = re.compile(r"^[0-9a-f]{40}$")


@dataclass(frozen=True)
class ACanaryTopology:
    """Pinned 16-node GBS-512 DFlash-B8 canary topology."""

    nodes: int = 16
    serve_nodes: int = 8
    trainer_nodes: int = 8
    gpus_per_node: int = 4
    serve_tp: int = 1
    serve_replicas_per_node: int = 4
    per_device_train_batch_size: int = 4
    gradient_accumulation_steps: int = 4
    max_steps: int = 200
    training_seq_len: int = 4096
    num_attention_heads: int = 32
    num_key_value_heads: int = 8
    head_dim: int = 128
    intermediate_size: int = 9728

    def __post_init__(self) -> None:
        pinned = (16, 8, 8, 4, 1, 4, 4, 4, 200, 4096, 32, 8, 128, 9728)
        if tuple(asdict(self).values()) != pinned:
            raise ValueError("A canary topology must use pinned 16-node defaults")
        if self.global_batch_size != 512 or self.active_gpu_ranks != 64:
            raise ValueError("A canary topology must use GBS512 and 64 GPUs")

    @property
    def global_batch_size(self) -> int:
        """Return the sequence-level global batch size."""
        return (
            self.trainer_nodes
            * self.gpus_per_node
            * self.per_device_train_batch_size
            * self.gradient_accumulation_steps
        )

    @property
    def active_gpu_ranks(self) -> int:
        """Return the expected number of GPU ranks across all nodes."""
        return self.nodes * self.gpus_per_node

    @property
    def dflash_dims(self) -> tuple[int, int, int, int]:
        """Return the exact Qwen3-4B DFlash draft dimensions."""
        return (
            self.num_attention_heads,
            self.num_key_value_heads,
            self.head_dim,
            self.intermediate_size,
        )


@dataclass(frozen=True)
class ACanaryRuntimeIdentity:
    """Pinned source, model, container, producer, supervisor, and recipe bytes."""

    source_commit: str
    target_revision: str
    tokenizer_sha256: str
    chat_template_sha256: str
    container_sha256: str
    producer_module_path: str
    producer_module_sha256: str
    runner_path: str
    runner_sha256: str
    supervisor_path: str
    supervisor_sha256: str
    recipe_path: str
    recipe_sha256: str
    evaluator_path: str
    evaluator_sha256: str
    target_model_id: str = "Qwen/Qwen3-4B"
    method: str = "dflash"
    block_size: int = 8
    thinking_mode: str = "off"
    seed: int = 42
    wandb_project: str = "sna-qwen3-4b-dataset-study"
    chat_template_mode: str = "source-native-thinking-off"

    def __post_init__(self) -> None:
        if (
            _COMMIT.fullmatch(self.source_commit) is None
            or _COMMIT.fullmatch(self.target_revision) is None
        ):
            raise ValueError("A runtime commits must be exact")
        digests = (
            self.tokenizer_sha256,
            self.chat_template_sha256,
            self.container_sha256,
            self.producer_module_sha256,
            self.runner_sha256,
            self.supervisor_sha256,
            self.recipe_sha256,
            self.evaluator_sha256,
        )
        if any(_SHA256.fullmatch(value) is None for value in digests):
            raise ValueError("A runtime digests must be exact")
        if (self.target_model_id, self.method, self.block_size, self.thinking_mode, self.seed) != (
            "Qwen/Qwen3-4B",
            "dflash",
            8,
            "off",
            42,
        ):
            raise ValueError("A runtime identity is not the authorized comparison")
        if self.wandb_project != "sna-qwen3-4b-dataset-study":
            raise ValueError("A runtime must use the shared Qwen3-4B study project")


@dataclass(frozen=True)
class ACanaryManifest:
    """Immutable input and runtime contract for A-repair."""

    runtime: ACanaryRuntimeIdentity
    topology: ACanaryTopology
    task9_a_selection_path: str
    task9_a_selection_sha256: str
    task8_publication_path: str
    task8_publication_sha256: str
    builder_receipt_path: str
    builder_receipt_sha256: str
    builder_output_path: str
    builder_output_sha256: str
    occurrence_count: int = 102_400
    scientific_training_authorized: bool = False

    def __post_init__(self) -> None:
        if self.occurrence_count != 102_400:
            raise ValueError("A canary requires exactly 102400 occurrences")
        if self.scientific_training_authorized:
            raise ValueError("A canary cannot authorize scientific training")
        for value in (
            self.task9_a_selection_sha256,
            self.task8_publication_sha256,
            self.builder_receipt_sha256,
            self.builder_output_sha256,
        ):
            if _SHA256.fullmatch(value) is None:
                raise ValueError("A manifest lineage digest is invalid")

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable manifest body."""
        return asdict(self)


def write_a_canary_manifest(path: Path, manifest: ACanaryManifest) -> None:
    """Write an immutable self-hashed A manifest."""
    body = manifest.as_dict()
    body["manifest_sha256"] = _sha256_json(body)
    atomic_publish_bytes(path, (_canonical_json(body) + "\n").encode(), job_id=str(os.getpid()))


def load_a_canary_manifest(path: Path) -> ACanaryManifest:
    """Load and verify an immutable A manifest."""
    body = json.loads(_stable_file_bytes(path))
    if not isinstance(body, dict):
        raise ValueError("A manifest must be an object")
    claim = body.pop("manifest_sha256", None)
    if claim != _sha256_json(body):
        raise ValueError("A manifest identity mismatch")
    topology = body.pop("topology", None)
    runtime = body.pop("runtime", None)
    if not isinstance(topology, dict) or not isinstance(runtime, dict):
        raise ValueError("A manifest nested identities are invalid")
    try:
        return ACanaryManifest(
            topology=ACanaryTopology(**topology),
            runtime=ACanaryRuntimeIdentity(**runtime),
            **body,
        )
    except TypeError as error:
        raise ValueError("A manifest fields are invalid") from error


def validate_bound_artifacts(manifest: ACanaryManifest, *, repository_root: Path) -> None:
    """Rehash every source/input artifact immediately before GPU work."""
    runtime_files = (
        (manifest.runtime.producer_module_path, manifest.runtime.producer_module_sha256),
        (manifest.runtime.runner_path, manifest.runtime.runner_sha256),
        (manifest.runtime.supervisor_path, manifest.runtime.supervisor_sha256),
        (manifest.runtime.recipe_path, manifest.runtime.recipe_sha256),
        (manifest.runtime.evaluator_path, manifest.runtime.evaluator_sha256),
    )
    for relative, digest in runtime_files:
        path = repository_root / relative
        if hashlib.sha256(_stable_file_bytes(path)).hexdigest() != digest:
            raise ValueError("A runtime source identity mismatch")
    for path_value, digest in (
        (manifest.task9_a_selection_path, manifest.task9_a_selection_sha256),
        (manifest.task8_publication_path, manifest.task8_publication_sha256),
        (manifest.builder_receipt_path, manifest.builder_receipt_sha256),
        (manifest.builder_output_path, manifest.builder_output_sha256),
    ):
        if hashlib.sha256(_stable_file_bytes(Path(path_value))).hexdigest() != digest:
            raise ValueError("A input artifact identity mismatch")
    selection = json.loads(_stable_file_bytes(Path(manifest.task9_a_selection_path)))
    identity = selection.get("selection_identity") if isinstance(selection, dict) else None
    if (
        selection.get("schema_version") != 3
        or not isinstance(identity, dict)
        or identity.get("strategy") != "A-repair"
    ):
        raise ValueError("A Task9 selection semantics mismatch")
    publication = json.loads(_stable_file_bytes(Path(manifest.task8_publication_path)))
    if (
        publication.get("selection_manifest_sha256") != manifest.task9_a_selection_sha256
        or publication.get("artifact_source_commit") != manifest.runtime.source_commit
    ):
        raise ValueError("A Task8 publication lineage mismatch")
    builder = json.loads(_stable_file_bytes(Path(manifest.builder_receipt_path)))
    builder_claim = builder.pop("receipt_sha256", None) if isinstance(builder, dict) else None
    if (
        not isinstance(builder, dict)
        or builder_claim != _sha256_json(builder)
        or builder.get("source_projection_sha256") != manifest.task9_a_selection_sha256
        or builder.get("source_task8_publication_sha256") != manifest.task8_publication_sha256
        or builder.get("output_sha256") != manifest.builder_output_sha256
        or builder.get("occurrence_count") != 102_400
    ):
        raise ValueError("A builder lineage mismatch")


def last_finite_training_loss(state: object, *, expected_step: int) -> float:
    """Return HF Trainer's last finite train loss at the expected global step."""
    if not isinstance(state, dict) or state.get("global_step") != expected_step:
        raise ValueError("trainer state global step mismatch")
    history = state.get("log_history")
    if not isinstance(history, list):
        raise ValueError("trainer state lacks finite log_history loss")
    candidates = [
        float(entry["loss"])
        for entry in history
        if isinstance(entry, dict)
        and isinstance(entry.get("loss"), (int, float))
        and not isinstance(entry.get("loss"), bool)
        and math.isfinite(float(entry["loss"]))
    ]
    if not candidates:
        raise ValueError("trainer state lacks finite training loss")
    return candidates[-1]


def validate_evaluator_receipt(path: Path, *, job_id: str, export_sha256: str) -> dict[str, Any]:
    """Require a self-hashed, current-job evaluator with real completed requests."""
    raw = _stable_file_bytes(path)
    body = json.loads(raw)
    if not isinstance(body, dict):
        raise ValueError("A evaluator receipt must be an object")
    claim = body.pop("receipt_sha256", None)
    metrics = body.get("metrics")
    if (
        claim != _sha256_json(body)
        or str(body.get("slurm_job_id")) != job_id
        or body.get("status") != "passed"
        or body.get("evaluator") != "specdec-bench-v1"
        or body.get("export_sha256") != export_sha256
        or not isinstance(body.get("completed_requests"), int)
        or body["completed_requests"] <= 0
    ):
        raise ValueError("A evaluator lacks completed requests or current-job identity")
    if (
        not isinstance(metrics, dict)
        or not metrics
        or any(
            not isinstance(value, (int, float))
            or isinstance(value, bool)
            or not math.isfinite(float(value))
            for value in metrics.values()
        )
    ):
        raise ValueError("A evaluator metrics must be finite")
    return body


def publish_a_authorization(
    path: Path,
    manifest: ACanaryManifest,
    *,
    job_id: str,
    checkpoint_path: Path,
    export_path: Path,
    gpu_evidence_path: Path,
    evaluation_receipt_path: Path,
) -> None:
    """Publish the only receipt allowed to authorize the B-balanced canary."""
    if not job_id.isdigit():
        raise ValueError("A authorization requires a numeric Slurm job ID")
    trainer_state = json.loads(_stable_file_bytes(checkpoint_path / "trainer_state.json"))
    finite_loss = last_finite_training_loss(trainer_state, expected_step=200)
    checkpoint_sha = _directory_sha256(checkpoint_path)
    export_sha = _directory_sha256(export_path)
    evaluation_raw = _stable_file_bytes(evaluation_receipt_path)
    validate_evaluator_receipt(evaluation_receipt_path, job_id=job_id, export_sha256=export_sha)
    gpu_raw = _stable_file_bytes(gpu_evidence_path)
    gpu = json.loads(gpu_raw)
    if (
        not isinstance(gpu, dict)
        or str(gpu.get("slurm_job_id")) != job_id
        or tuple(gpu.get("active_gpu_ranks", ())) != tuple(range(64))
    ):
        raise ValueError("A GPU evidence does not cover all 64 ranks")
    runtime = manifest.runtime
    payload: dict[str, Any] = {
        "schema_version": 1,
        "producer": "qwen4b-a-repair-canary-job-v1",
        "arm": "A-repair",
        "authorization": "B-balanced-canary",
        "complete": True,
        "canary_completed": True,
        "source_commit": runtime.source_commit,
        "target_model_id": runtime.target_model_id,
        "target_revision": runtime.target_revision,
        "tokenizer_sha256": runtime.tokenizer_sha256,
        "chat_template_sha256": runtime.chat_template_sha256,
        "container_sha256": runtime.container_sha256,
        "thinking_mode": runtime.thinking_mode,
        "method": runtime.method,
        "block_size": runtime.block_size,
        "seed": runtime.seed,
        "global_batch_size": manifest.topology.global_batch_size,
        "max_steps": manifest.topology.max_steps,
        "scientific_training_authorized": False,
        "checkpoint_reloaded": True,
        "drafter_exported": True,
        "all_gpus_active": True,
        "finite_loss": finite_loss,
        "slurm_job_id": job_id,
        "task9_a_selection_path": manifest.task9_a_selection_path,
        "task9_a_selection_sha256": manifest.task9_a_selection_sha256,
        "task8_publication_path": manifest.task8_publication_path,
        "task8_publication_sha256": manifest.task8_publication_sha256,
        "builder_receipt_path": manifest.builder_receipt_path,
        "builder_receipt_sha256": manifest.builder_receipt_sha256,
        "builder_output_path": manifest.builder_output_path,
        "builder_output_sha256": manifest.builder_output_sha256,
        "producer_module_path": runtime.producer_module_path,
        "producer_module_sha256": runtime.producer_module_sha256,
        "runner_path": runtime.runner_path,
        "runner_sha256": runtime.runner_sha256,
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_sha256": checkpoint_sha,
        "export_path": str(export_path),
        "export_sha256": export_sha,
        "gpu_evidence_path": str(gpu_evidence_path),
        "gpu_evidence_sha256": hashlib.sha256(gpu_raw).hexdigest(),
        "evaluation_receipt_path": str(evaluation_receipt_path),
        "evaluation_receipt_sha256": hashlib.sha256(evaluation_raw).hexdigest(),
    }
    payload["receipt_sha256"] = _sha256_json(payload)
    atomic_publish_bytes(path, (_canonical_json(payload) + "\n").encode(), job_id=job_id)


def publish_a_gpu_activity(log_root: Path, path: Path, *, job_id: str) -> None:
    """Derive exact 64-rank activity from current-job node-local pmon logs."""
    if not job_id.isdigit():
        raise ValueError("A GPU evidence requires a numeric Slurm job ID")
    active_ranks: list[int] = []
    log_sha256: dict[str, str] = {}
    for node in range(16):
        log_path = log_root / f"nvidia-smi-pmon-node-{node}-{job_id}.log"
        raw = _stable_file_bytes(log_path)
        columns: dict[str, int] | None = None
        active: set[int] = set()
        for line in raw.decode(errors="replace").splitlines():
            if line.startswith("#"):
                names = line.lstrip("# ").split()
                if {"gpu", "pid", "sm"}.issubset(names):
                    columns = {name: names.index(name) for name in ("gpu", "pid", "sm")}
                continue
            fields = line.split()
            if columns is None or len(fields) <= max(columns.values()):
                continue
            gpu, pid, sm = (fields[columns["gpu"]], fields[columns["pid"]], fields[columns["sm"]])
            try:
                if gpu.isdigit() and pid != "-" and float(sm) > 0:
                    active.add(int(gpu))
            except ValueError:
                continue
        if active != {0, 1, 2, 3}:
            raise ValueError(f"A GPU evidence missing activity on node {node}")
        active_ranks.extend(node * 4 + gpu for gpu in sorted(active))
        log_sha256[log_path.name] = hashlib.sha256(raw).hexdigest()
    body: dict[str, Any] = {
        "schema_version": 1,
        "slurm_job_id": job_id,
        "active_gpu_ranks": active_ranks,
        "pmon_sha256": log_sha256,
    }
    body["receipt_sha256"] = _sha256_json(body)
    atomic_publish_bytes(path, (_canonical_json(body) + "\n").encode(), job_id=job_id)


def _stable_file_bytes(path: Path) -> bytes:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    with os.fdopen(descriptor, "rb") as stream:
        before = os.fstat(stream.fileno())
        namespace = os.lstat(path)
        raw = stream.read()
        after = os.fstat(stream.fileno())
    if (
        not stat.S_ISREG(before.st_mode)
        or (namespace.st_dev, namespace.st_ino) != (before.st_dev, before.st_ino)
        or (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
        != (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
    ):
        raise ValueError("artifact is not an inode-stable regular file")
    return raw


def _directory_sha256(root: Path) -> str:
    if not root.is_dir() or root.is_symlink() or any(path.is_symlink() for path in root.rglob("*")):
        raise ValueError("artifact directory is invalid")
    files = sorted(path for path in root.rglob("*") if path.is_file())
    if not files:
        raise ValueError("artifact directory is empty")
    digest = hashlib.sha256()
    for artifact in files:
        digest.update(artifact.relative_to(root).as_posix().encode())
        digest.update(b"\0")
        digest.update(bytes.fromhex(hashlib.sha256(_stable_file_bytes(artifact)).hexdigest()))
    return digest.hexdigest()


def _canonical_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _sha256_json(value: dict[str, Any]) -> str:
    return hashlib.sha256(_canonical_json(value).encode()).hexdigest()
