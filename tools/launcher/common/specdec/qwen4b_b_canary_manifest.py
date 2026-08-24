# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Immutable OCI-HSG B-balanced canary manifest and evidence validation."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from common.specdec.qwen4b_b_atomic import atomic_publish_bytes
from common.specdec.qwen4b_b_readiness import load_b_readiness_receipt, validate_builder_artifacts

__all__ = [
    "ACanaryAuthorization",
    "BCanaryEvidence",
    "BCanaryManifest",
    "BCanaryRuntimeIdentity",
    "BCanaryTopology",
    "load_b_canary_manifest",
    "publish_b_canary_evidence",
    "publish_b_gpu_activity",
    "publish_b_supervisor_completion",
    "validate_a_authorization_receipt",
    "validate_bound_artifacts",
    "validate_canary_evidence",
    "validate_runtime_artifacts",
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
class BCanaryRuntimeIdentity:
    """Canonical, digest-pinned scientific and launcher identity."""

    target_model_id: str
    target_path: str
    target_revision: str
    tokenizer_sha256: str
    chat_template_sha256: str
    container_sha256: str
    thinking_mode: str
    method: str
    block_size: int
    training_seq_len: int
    seed: int
    supervisor_path: str
    supervisor_sha256: str
    config_path: str
    config_sha256: str
    corpus_arm: str
    corpus_path: str
    output_root: str
    wandb_project: str
    wandb_run_id: str

    def __post_init__(self) -> None:
        expected = {
            "target_model_id": "Qwen/Qwen3-4B",
            "thinking_mode": "off",
            "method": "dflash",
            "block_size": 8,
            "training_seq_len": 4096,
            "seed": 42,
            "corpus_arm": "B-balanced",
            "wandb_project": "sna-qwen3-4b-dataset-study",
        }
        if any(getattr(self, key) != value for key, value in expected.items()):
            raise ValueError("B canary runtime identity diverges from the canonical study")
        for name in ("target_path", "supervisor_path", "config_path", "corpus_path", "output_root"):
            if not Path(getattr(self, name)).is_absolute():
                raise ValueError(f"B canary {name} must be absolute")
        for name in (
            "tokenizer_sha256",
            "chat_template_sha256",
            "container_sha256",
            "supervisor_sha256",
            "config_sha256",
        ):
            if _SHA256.fullmatch(getattr(self, name)) is None:
                raise ValueError(f"B canary {name} must be exact")
        if _COMMIT.fullmatch(self.target_revision) is None:
            raise ValueError("B canary target revision must be exact")
        if not re.fullmatch(r"q4b-b-[a-z0-9-]+", self.wandb_run_id):
            raise ValueError("B canary W&B run ID must be stable")


@dataclass(frozen=True)
class BCanaryManifest:
    """All identity and resource facts needed to run the bounded B canary."""

    readiness_receipt_sha256: str
    builder_receipt_sha256: str
    builder_output_sha256: str
    task9_projection_sha256: str
    task8_publication_sha256: str
    corpus_manifest_sha256: str
    selection_receipt_sha256: str
    shard_inventory_sha256: str
    canary_occurrence_count: int
    topology: BCanaryTopology
    runtime: BCanaryRuntimeIdentity
    source_commit: str
    scientific_milestone: bool = False

    def __post_init__(self) -> None:
        for label, digest in (
            ("readiness receipt", self.readiness_receipt_sha256),
            ("builder receipt", self.builder_receipt_sha256),
            ("builder output", self.builder_output_sha256),
            ("Task9 projection", self.task9_projection_sha256),
            ("Task8 publication", self.task8_publication_sha256),
            ("corpus manifest", self.corpus_manifest_sha256),
            ("selection receipt", self.selection_receipt_sha256),
            ("shard inventory", self.shard_inventory_sha256),
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


@dataclass(frozen=True)
class ACanaryAuthorization:
    """Authenticated A-repair gate required before B GPU scheduling."""

    receipt_sha256: str
    source_commit: str
    task9_a_selection_sha256: str
    task8_publication_sha256: str
    builder_receipt_sha256: str
    builder_output_sha256: str
    tokenizer_sha256: str
    chat_template_sha256: str
    container_sha256: str
    target_revision: str
    checkpoint_sha256: str
    export_sha256: str
    gpu_evidence_sha256: str
    evaluation_receipt_sha256: str


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
    payload = json.loads(_stable_file_bytes(path))
    if not isinstance(payload, dict):
        raise ValueError("B canary manifest must be a JSON object")
    digest = payload.pop("manifest_sha256", None)
    if not isinstance(digest, str) or digest != _sha256_json(payload):
        raise ValueError("B canary manifest identity mismatch")
    try:
        topology = payload.pop("topology")
        if not isinstance(topology, dict):
            raise ValueError("B canary topology must be an object")
        runtime = payload.pop("runtime")
        if not isinstance(runtime, dict):
            raise ValueError("B canary runtime must be an object")
        return BCanaryManifest(
            topology=BCanaryTopology(**topology),
            runtime=BCanaryRuntimeIdentity(**runtime),
            **payload,
        )
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
        or readiness.get("task9_projection_sha256") != manifest.task9_projection_sha256
        or readiness.get("task8_publication_sha256") != manifest.task8_publication_sha256
        or readiness.get("corpus_manifest_sha256") != manifest.corpus_manifest_sha256
        or readiness.get("selection_receipt_sha256") != manifest.selection_receipt_sha256
        or readiness.get("shard_inventory_sha256") != manifest.shard_inventory_sha256
    ):
        raise ValueError("B readiness transitive identity mismatch")
    validate_builder_artifacts(
        receipt_path=builder_receipt_path,
        receipt_sha256=manifest.builder_receipt_sha256,
        output_path=builder_output_path,
        output_sha256=manifest.builder_output_sha256,
        source_commit=manifest.source_commit,
        source_projection_sha256=manifest.task9_projection_sha256,
        source_task8_publication_sha256=manifest.task8_publication_sha256,
        source_corpus_manifest_sha256=manifest.corpus_manifest_sha256,
        source_selection_receipt_sha256=manifest.selection_receipt_sha256,
        source_shard_inventory_sha256=manifest.shard_inventory_sha256,
    )


def validate_runtime_artifacts(
    manifest: BCanaryManifest,
    *,
    supervisor_path: Path,
    config_path: Path,
    container_path: Path | None = None,
) -> None:
    """Rehash the exact launcher, target snapshot, tokenizer, template, and container."""
    runtime = manifest.runtime
    for actual, declared_path, declared_sha256 in (
        (supervisor_path, runtime.supervisor_path, runtime.supervisor_sha256),
        (config_path, runtime.config_path, runtime.config_sha256),
    ):
        if (
            str(actual) != declared_path
            or hashlib.sha256(_stable_file_bytes(actual)).hexdigest() != declared_sha256
        ):
            raise ValueError("B runtime launcher identity changed")
    target = Path(runtime.target_path)
    if target.resolve().name != runtime.target_revision:
        raise ValueError("B target snapshot path does not bind the exact revision")
    config = json.loads(_stable_file_bytes(target / "config.json"))
    expected_target = {
        "model_type": "qwen3",
        "hidden_size": 2560,
        "intermediate_size": 9728,
        "num_hidden_layers": 36,
        "num_attention_heads": 32,
        "num_key_value_heads": 8,
        "head_dim": 128,
    }
    if not isinstance(config, dict) or any(
        config.get(key) != value for key, value in expected_target.items()
    ):
        raise ValueError("B target snapshot is not the exact Qwen3-4B architecture")
    tokenizer_names = {
        "added_tokens.json",
        "chat_template.jinja",
        "merges.txt",
        "special_tokens_map.json",
        "tokenizer.json",
        "tokenizer.model",
        "tokenizer_config.json",
        "vocab.json",
        "vocab.txt",
    }
    tokenizer_files = sorted(
        path for path in target.iterdir() if path.name in tokenizer_names and path.is_file()
    )
    if not tokenizer_files or any(path.is_symlink() for path in tokenizer_files):
        raise ValueError("B tokenizer snapshot is missing or contains symlinks")
    tokenizer_digest = hashlib.sha256()
    for path in tokenizer_files:
        tokenizer_digest.update(path.name.encode())
        tokenizer_digest.update(b"\0")
        tokenizer_digest.update(bytes.fromhex(hashlib.sha256(_stable_file_bytes(path)).hexdigest()))
    if tokenizer_digest.hexdigest() != runtime.tokenizer_sha256:
        raise ValueError("B tokenizer snapshot identity changed")
    if (
        hashlib.sha256(_stable_file_bytes(target / "chat_template.jinja")).hexdigest()
        != runtime.chat_template_sha256
    ):
        raise ValueError("B chat-template identity changed")
    if (
        container_path is not None
        and hashlib.sha256(_stable_file_bytes(container_path)).hexdigest()
        != runtime.container_sha256
    ):
        raise ValueError("B container identity changed")


def validate_a_authorization_receipt(
    path: Path,
    expected_sha256: str,
    *,
    source_commit: str,
    target_revision: str,
    tokenizer_sha256: str,
    chat_template_sha256: str,
    container_sha256: str,
    repo_root: Path,
) -> ACanaryAuthorization:
    """Verify a caller-pinned, job-issued A-repair authorization trust root."""
    raw = _stable_file_bytes(path)
    if hashlib.sha256(raw).hexdigest() != expected_sha256:
        raise ValueError("A authorization receipt file identity mismatch")
    payload = json.loads(raw)
    if not isinstance(payload, dict):
        raise ValueError("A authorization receipt must be an object")
    self_digest = payload.pop("receipt_sha256", None)
    if self_digest != _sha256_json(payload):
        raise ValueError("A authorization receipt self identity mismatch")
    required = {
        "schema_version": 1,
        "producer": "qwen4b-a-repair-canary-job-v1",
        "arm": "A-repair",
        "authorization": "B-balanced-canary",
        "complete": True,
        "canary_completed": True,
        "source_commit": source_commit,
        "target_model_id": "Qwen/Qwen3-4B",
        "thinking_mode": "off",
        "method": "dflash",
        "block_size": 8,
        "seed": 42,
        "global_batch_size": 512,
        "max_steps": 200,
        "checkpoint_reloaded": True,
        "drafter_exported": True,
        "all_gpus_active": True,
    }
    if any(payload.get(key) != value for key, value in required.items()):
        raise ValueError("A authorization receipt semantics mismatch")
    genuine_sources = {
        "producer_module_path": "tools/launcher/common/specdec/qwen4b_a_canary_manifest.py",
        "runner_path": "tools/launcher/common/specdec/run_qwen4b_a_canary.sbatch",
    }
    if any(payload.get(key) != value for key, value in genuine_sources.items()):
        raise ValueError("A authorization lacks genuine A producer identity")
    for path_field, digest_field in (
        ("producer_module_path", "producer_module_sha256"),
        ("runner_path", "runner_sha256"),
    ):
        relative = Path(str(payload[path_field]))
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("A authorization genuine producer path is invalid")
        source_path = repo_root / relative
        try:
            source_raw = _stable_file_bytes(source_path)
        except (FileNotFoundError, OSError) as error:
            raise ValueError("A authorization genuine A producer is unavailable") from error
        if hashlib.sha256(source_raw).hexdigest() != payload.get(digest_field):
            raise ValueError("A authorization genuine A producer identity mismatch")
        committed = subprocess.run(
            ["git", "-C", str(repo_root), "show", f"{source_commit}:{relative.as_posix()}"],
            check=False,
            capture_output=True,
        )
        if committed.returncode != 0 or committed.stdout != source_raw:
            raise ValueError("A authorization genuine producer is not source-commit exact")
    identical_parent = {
        "target_revision": target_revision,
        "tokenizer_sha256": tokenizer_sha256,
        "chat_template_sha256": chat_template_sha256,
        "container_sha256": container_sha256,
    }
    if any(payload.get(key) != value for key, value in identical_parent.items()):
        raise ValueError("A/B identical-parent identity mismatch")
    if not isinstance(payload.get("finite_loss"), (int, float)) or not math.isfinite(
        float(payload["finite_loss"])
    ):
        raise ValueError("A authorization receipt lacks finite loss")
    digest_fields = (
        "task9_a_selection_sha256",
        "task8_publication_sha256",
        "builder_receipt_sha256",
        "builder_output_sha256",
        "tokenizer_sha256",
        "chat_template_sha256",
        "container_sha256",
        "checkpoint_sha256",
        "export_sha256",
        "gpu_evidence_sha256",
        "evaluation_receipt_sha256",
        "producer_module_sha256",
        "runner_sha256",
    )
    if any(_SHA256.fullmatch(str(payload.get(name, ""))) is None for name in digest_fields):
        raise ValueError("A authorization digest lineage is incomplete")
    if _COMMIT.fullmatch(str(payload.get("target_revision", ""))) is None:
        raise ValueError("A authorization target revision is invalid")
    if not str(payload.get("slurm_job_id", "")).isdigit():
        raise ValueError("A authorization is not job-generated")
    for path_field, digest_field in (
        ("task9_a_selection_path", "task9_a_selection_sha256"),
        ("task8_publication_path", "task8_publication_sha256"),
        ("builder_receipt_path", "builder_receipt_sha256"),
        ("builder_output_path", "builder_output_sha256"),
    ):
        artifact = Path(str(payload.get(path_field, "")))
        if hashlib.sha256(_stable_file_bytes(artifact)).hexdigest() != payload[digest_field]:
            raise ValueError("A authorization source lineage mismatch")
    selection = json.loads(_stable_file_bytes(Path(str(payload["task9_a_selection_path"]))))
    selection_identity = (
        selection.get("selection_identity") if isinstance(selection, dict) else None
    )
    if (
        not isinstance(selection, dict)
        or selection.get("schema_version") != 3
        or not isinstance(selection_identity, dict)
        or selection_identity.get("strategy") != "A-repair"
    ):
        raise ValueError("A authorization Task9 selection semantics mismatch")
    publication = json.loads(_stable_file_bytes(Path(str(payload["task8_publication_path"]))))
    if (
        not isinstance(publication, dict)
        or publication.get("selection_manifest_sha256") != payload["task9_a_selection_sha256"]
        or publication.get("artifact_source_commit") != source_commit
    ):
        raise ValueError("A authorization Task8 publication lineage mismatch")
    builder = json.loads(_stable_file_bytes(Path(str(payload["builder_receipt_path"]))))
    builder_claim = builder.pop("receipt_sha256", None) if isinstance(builder, dict) else None
    if (
        not isinstance(builder, dict)
        or builder_claim != _sha256_json(builder)
        or builder.get("source_commit") != source_commit
        or builder.get("source_projection_sha256") != payload["task9_a_selection_sha256"]
        or builder.get("source_task8_publication_sha256") != payload["task8_publication_sha256"]
        or builder.get("output_sha256") != payload["builder_output_sha256"]
    ):
        raise ValueError("A authorization builder lineage mismatch")
    for path_field, digest_field in (
        ("checkpoint_path", "checkpoint_sha256"),
        ("export_path", "export_sha256"),
    ):
        artifact = Path(str(payload.get(path_field, "")))
        if _directory_sha256(artifact) != payload[digest_field]:
            raise ValueError("A authorization artifact identity mismatch")
    gpu_path = Path(str(payload.get("gpu_evidence_path", "")))
    gpu_raw = _stable_file_bytes(gpu_path)
    if hashlib.sha256(gpu_raw).hexdigest() != payload["gpu_evidence_sha256"]:
        raise ValueError("A authorization GPU evidence identity mismatch")
    gpu = json.loads(gpu_raw)
    if (
        not isinstance(gpu, dict)
        or str(gpu.get("slurm_job_id")) != str(payload["slurm_job_id"])
        or tuple(gpu.get("active_gpu_ranks", ())) != tuple(range(64))
    ):
        raise ValueError("A authorization GPU evidence does not cover all 64 ranks")
    trainer_state = json.loads(
        _stable_file_bytes(Path(str(payload["checkpoint_path"])) / "trainer_state.json")
    )
    if float(payload["finite_loss"]) != _last_finite_training_loss(
        trainer_state, expected_step=200
    ):
        raise ValueError("A authorization finite-loss evidence mismatch")
    evaluation_path = Path(str(payload.get("evaluation_receipt_path", "")))
    evaluation_raw = _stable_file_bytes(evaluation_path)
    if hashlib.sha256(evaluation_raw).hexdigest() != payload["evaluation_receipt_sha256"]:
        raise ValueError("A authorization evaluator identity mismatch")
    _validate_evaluator_receipt(
        evaluation_raw,
        job_id=str(payload["slurm_job_id"]),
        export_sha256=str(payload["export_sha256"]),
    )
    return ACanaryAuthorization(
        receipt_sha256=expected_sha256,
        **{
            field: payload[field]
            for field in ACanaryAuthorization.__dataclass_fields__
            if field != "receipt_sha256"
        },
    )


def publish_b_canary_evidence(
    path: Path,
    manifest: BCanaryManifest,
    *,
    job_id: str,
    checkpoint_path: Path,
    export_path: Path,
    evaluation_receipt_path: Path,
    gpu_evidence_path: Path,
    supervisor_receipt_path: Path,
) -> None:
    """Derive immutable B evidence from artifacts produced by the current job."""
    if not job_id.isdigit():
        raise ValueError("B evidence requires the current Slurm job ID")
    trainer_state = json.loads(_stable_file_bytes(checkpoint_path / "trainer_state.json"))
    finite_loss = _last_finite_training_loss(
        trainer_state, expected_step=manifest.topology.max_steps
    )
    checkpoint_sha = _directory_sha256(checkpoint_path)
    export_sha = _directory_sha256(export_path)
    supervisor_raw = _stable_file_bytes(supervisor_receipt_path)
    supervisor = json.loads(supervisor_raw)
    supervisor_claim = (
        supervisor.pop("receipt_sha256", None) if isinstance(supervisor, dict) else None
    )
    if (
        not isinstance(supervisor, dict)
        or supervisor_claim != _sha256_json(supervisor)
        or supervisor.get("producer") != "canonical-dflash-supervisor-v1"
        or str(supervisor.get("slurm_job_id")) != job_id
        or supervisor.get("status") != "completed"
        or supervisor.get("supervisor_sha256") != manifest.runtime.supervisor_sha256
        or supervisor.get("config_sha256") != manifest.runtime.config_sha256
        or supervisor.get("checkpoint_sha256") != checkpoint_sha
        or supervisor.get("export_sha256") != export_sha
        or supervisor.get("checkpoint_reloaded_by_exporter") is not True
        or supervisor.get("final_training_loss") != finite_loss
    ):
        raise ValueError("B canonical supervisor completion identity mismatch")
    evaluation_raw = _stable_file_bytes(evaluation_receipt_path)
    _validate_evaluator_receipt(evaluation_raw, job_id=job_id, export_sha256=export_sha)
    gpu_raw = _stable_file_bytes(gpu_evidence_path)
    gpu = json.loads(gpu_raw)
    if (
        not isinstance(gpu, dict)
        or str(gpu.get("slurm_job_id")) != job_id
        or tuple(gpu.get("active_gpu_ranks", ())) != tuple(range(manifest.active_gpu_ranks))
    ):
        raise ValueError("B GPU evidence does not cover all 64 ranks")
    payload = {
        "schema_version": 1,
        "slurm_job_id": job_id,
        "manifest_sha256": _sha256_json(manifest.as_dict()),
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_sha256": checkpoint_sha,
        "export_path": str(export_path),
        "export_sha256": export_sha,
        "evaluation_receipt_sha256": hashlib.sha256(evaluation_raw).hexdigest(),
        "supervisor_receipt_sha256": hashlib.sha256(supervisor_raw).hexdigest(),
        "gpu_evidence_sha256": hashlib.sha256(gpu_raw).hexdigest(),
        "finite_loss": True,
        "final_training_loss": finite_loss,
        "checkpoint_reloaded": True,
        "drafter_exported": True,
        "evaluator_completed": True,
        "all_64_gpus_active": True,
        "scientific_milestone": False,
    }
    payload["receipt_sha256"] = _sha256_json(payload)
    atomic_publish_bytes(
        path,
        (json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n").encode(),
        job_id=job_id,
    )


def publish_b_supervisor_completion(
    path: Path,
    manifest: BCanaryManifest,
    *,
    job_id: str,
    checkpoint_path: Path,
    export_path: Path,
) -> None:
    """Record successful canonical trainer/exporter completion in the current job."""
    if not job_id.isdigit():
        raise ValueError("B supervisor completion requires a numeric Slurm job ID")
    state = json.loads(_stable_file_bytes(checkpoint_path / "trainer_state.json"))
    final_loss = _last_finite_training_loss(state, expected_step=manifest.topology.max_steps)
    export_config = json.loads(_stable_file_bytes(export_path / "config.json"))
    if not isinstance(export_config, dict) or not tuple(export_path.glob("*.safetensors")):
        raise ValueError("B canonical exporter did not produce a loadable HF artifact")
    payload = {
        "schema_version": 1,
        "producer": "canonical-dflash-supervisor-v1",
        "slurm_job_id": job_id,
        "status": "completed",
        "supervisor_sha256": manifest.runtime.supervisor_sha256,
        "config_sha256": manifest.runtime.config_sha256,
        "checkpoint_sha256": _directory_sha256(checkpoint_path),
        "export_sha256": _directory_sha256(export_path),
        "checkpoint_reloaded_by_exporter": True,
        "final_training_loss": final_loss,
    }
    payload["receipt_sha256"] = _sha256_json(payload)
    atomic_publish_bytes(
        path,
        (json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n").encode(),
        job_id=job_id,
    )


def _last_finite_training_loss(state: object, *, expected_step: int) -> float:
    if not isinstance(state, dict) or state.get("global_step") != expected_step:
        raise ValueError("training output global step mismatch")
    history = state.get("log_history")
    if not isinstance(history, list):
        raise ValueError("training output lacks trainer_state.log_history")
    losses = [
        float(entry["loss"])
        for entry in history
        if isinstance(entry, dict)
        and isinstance(entry.get("loss"), (int, float))
        and not isinstance(entry.get("loss"), bool)
        and math.isfinite(float(entry["loss"]))
    ]
    if not losses:
        raise ValueError("training output lacks finite loss")
    return losses[-1]


def _validate_evaluator_receipt(raw: bytes, *, job_id: str, export_sha256: str) -> None:
    evaluation = json.loads(raw)
    if not isinstance(evaluation, dict):
        raise ValueError("evaluator receipt must be an object")
    claim = evaluation.pop("receipt_sha256", None)
    metrics = evaluation.get("metrics")
    if (
        claim != _sha256_json(evaluation)
        or str(evaluation.get("slurm_job_id")) != job_id
        or evaluation.get("status") != "passed"
        or evaluation.get("evaluator") != "specdec-bench-v1"
        or evaluation.get("export_sha256") != export_sha256
        or not isinstance(evaluation.get("completed_requests"), int)
        or evaluation["completed_requests"] <= 0
        or not isinstance(metrics, dict)
        or not metrics
        or any(
            not isinstance(value, (int, float))
            or isinstance(value, bool)
            or not math.isfinite(float(value))
            for value in metrics.values()
        )
    ):
        raise ValueError("evaluator output lacks genuine completed metrics")


def publish_b_gpu_activity(log_root: Path, path: Path, *, job_id: str) -> None:
    """Publish current-job, all-rank GPU activity from 16 node-local pmon logs."""
    if not job_id.isdigit():
        raise ValueError("B GPU evidence requires a numeric Slurm job ID")
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
            gpu, pid, sm = (
                fields[columns["gpu"]],
                fields[columns["pid"]],
                fields[columns["sm"]],
            )
            if gpu.isdigit() and pid != "-":
                try:
                    if float(sm) > 0:
                        active.add(int(gpu))
                except ValueError:
                    continue
        if active != {0, 1, 2, 3}:
            raise ValueError(f"B GPU evidence missing activity on node {node}: {sorted(active)}")
        active_ranks.extend(node * 4 + gpu for gpu in sorted(active))
        log_sha256[log_path.name] = hashlib.sha256(raw).hexdigest()
    payload = {
        "schema_version": 1,
        "slurm_job_id": job_id,
        "active_gpu_ranks": active_ranks,
        "pmon_sha256": log_sha256,
    }
    payload["receipt_sha256"] = _sha256_json(payload)
    atomic_publish_bytes(
        path,
        (json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n").encode(),
        job_id=job_id,
    )


def _stable_file_bytes(path: Path) -> bytes:
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags)
    with os.fdopen(descriptor, "rb") as stream:
        before = os.fstat(stream.fileno())
        namespace_before = os.lstat(path)
        raw = stream.read()
        after = os.fstat(stream.fileno())
    namespace_after = os.lstat(path)
    if (
        not path.is_file()
        or path.is_symlink()
        or (namespace_before.st_dev, namespace_before.st_ino) != (before.st_dev, before.st_ino)
        or (namespace_after.st_dev, namespace_after.st_ino) != (before.st_dev, before.st_ino)
        or (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
        != (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
    ):
        raise ValueError("artifact is not an inode-stable regular file")
    return raw


def _directory_sha256(root: Path) -> str:
    if not root.is_dir() or root.is_symlink():
        raise ValueError("artifact directory is missing")
    if any(path.is_symlink() for path in root.rglob("*")):
        raise ValueError("artifact directory contains a symlink")
    files = sorted(path for path in root.rglob("*") if path.is_file() and not path.is_symlink())
    if not files:
        raise ValueError("artifact directory is empty")
    digest = hashlib.sha256()
    for artifact in files:
        digest.update(artifact.relative_to(root).as_posix().encode())
        digest.update(b"\0")
        digest.update(bytes.fromhex(hashlib.sha256(_stable_file_bytes(artifact)).hexdigest()))
    return digest.hexdigest()


def _sha256_json(payload: dict[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
