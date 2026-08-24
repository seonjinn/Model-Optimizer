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
from datetime import UTC, datetime
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
    "publish_a_scheduler_observation",
    "publish_b_canary_evidence",
    "publish_b_exporter_invocation",
    "publish_b_gpu_activity",
    "publish_b_supervisor_completion",
    "validate_a_authorization_receipt",
    "validate_a_scheduler_observation",
    "validate_b_exporter_invocation",
    "validate_bound_artifacts",
    "validate_canary_evidence",
    "validate_runtime_artifacts",
    "validate_wandb_runtime",
    "write_b_canary_manifest",
]

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_COMMIT = re.compile(r"^[0-9a-f]{40}$")
_CANARY_OCCURRENCE_COUNT = 102_400
_WANDB_NETRC_PATH = Path("/run/secrets/wandb.netrc")


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
    target_snapshot_sha256: str
    tokenizer_sha256: str
    chat_template_sha256: str
    container_path: str
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
    runner_sha256: str
    evaluator_sha256: str
    train_script_sha256: str
    exporter_sha256: str
    wandb_netrc_sha256: str
    wandb_durable_root: str
    wandb_scratch_namespace: str
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
        for name in (
            "target_path",
            "container_path",
            "supervisor_path",
            "config_path",
            "corpus_path",
            "output_root",
        ):
            if not Path(getattr(self, name)).is_absolute():
                raise ValueError(f"B canary {name} must be absolute")
        for name in (
            "tokenizer_sha256",
            "chat_template_sha256",
            "target_snapshot_sha256",
            "container_sha256",
            "supervisor_sha256",
            "config_sha256",
            "runner_sha256",
            "evaluator_sha256",
            "train_script_sha256",
            "exporter_sha256",
            "wandb_netrc_sha256",
        ):
            if _SHA256.fullmatch(getattr(self, name)) is None:
                raise ValueError(f"B canary {name} must be exact")
        if _COMMIT.fullmatch(self.target_revision) is None:
            raise ValueError("B canary target revision must be exact")
        if not re.fullmatch(r"q4b-b-[a-z0-9-]+", self.wandb_run_id):
            raise ValueError("B canary W&B run ID must be stable")
        durable = Path(self.wandb_durable_root)
        if not durable.as_posix().startswith("/lustre/"):
            raise ValueError("B canary W&B durable root must be on Lustre")
        if self.wandb_scratch_namespace != "qwen4b-b":
            raise ValueError("B canary W&B scratch namespace is invalid")


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
    slurm_job_id: str


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
    repo_root: Path,
    container_path: Path | None = None,
    require_clean_checkout: bool = False,
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
    if _directory_sha256(target) != runtime.target_snapshot_sha256:
        raise ValueError("B full target snapshot identity changed")
    config = json.loads(_stable_file_bytes(target / "config.json"))
    expected_target = {
        "model_type": "qwen3",
        "vocab_size": 151936,
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
    if container_path is not None and str(container_path) != runtime.container_path:
        raise ValueError("B container path identity changed")
    if (
        container_path is not None
        and hashlib.sha256(_stable_file_bytes(container_path)).hexdigest()
        != runtime.container_sha256
    ):
        raise ValueError("B container identity changed")
    closure = {
        "tools/launcher/common/specdec/run_qwen4b_b_canary.sbatch": runtime.runner_sha256,
        "tools/launcher/common/specdec/run_qwen4b_b_canary_eval.sh": runtime.evaluator_sha256,
        "examples/speculative_decoding/launch_train.sh": runtime.train_script_sha256,
        "examples/speculative_decoding/scripts/export_hf_checkpoint.py": runtime.exporter_sha256,
    }
    for relative, expected in closure.items():
        if hashlib.sha256(_stable_file_bytes(repo_root / relative)).hexdigest() != expected:
            raise ValueError("B launcher/train/export closure identity changed")
    if require_clean_checkout:
        head = subprocess.run(
            ["git", "-C", str(repo_root), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "-C", str(repo_root), "status", "--porcelain"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout
        if head != manifest.source_commit or dirty:
            raise ValueError("B runtime requires the exact clean source commit")


def validate_wandb_runtime(
    manifest: BCanaryManifest,
    *,
    netrc_path: Path,
    slurm_tmpdir: Path,
    job_id: str,
    repository_root: Path,
) -> dict[str, str]:
    """Validate the RO W&B secret and derive durable/job-local state paths."""
    runtime = manifest.runtime
    if not job_id.isdigit() or not slurm_tmpdir.is_absolute():
        raise ValueError("B W&B runtime requires a current Slurm job")
    if netrc_path != _WANDB_NETRC_PATH:
        raise ValueError("B W&B netrc mount path mismatch")
    if hashlib.sha256(_stable_file_bytes(netrc_path)).hexdigest() != runtime.wandb_netrc_sha256:
        raise ValueError("B W&B netrc identity mismatch")
    if not _path_on_read_only_mount(netrc_path):
        raise ValueError("B W&B netrc mount must be read-only")
    durable_root = Path(runtime.wandb_durable_root)
    try:
        durable_root.resolve().relative_to(repository_root.resolve())
    except ValueError:
        pass
    else:
        raise ValueError("B W&B durable state must remain outside the repository")
    durable_job = durable_root / "jobs" / job_id
    scratch_cache = slurm_tmpdir / f"{runtime.wandb_scratch_namespace}-{job_id}" / "wandb-cache"
    return {
        "WANDB_DIR": str(durable_job / "run"),
        "WANDB_CONFIG_DIR": str(durable_job / "config"),
        "WANDB_ARTIFACT_DIR": str(durable_job / "artifacts"),
        "WANDB_CACHE_DIR": str(scratch_cache),
    }


def _path_on_read_only_mount(path: Path) -> bool:
    mountinfo = Path("/proc/self/mountinfo")
    if not mountinfo.is_file():
        return False
    resolved = path.resolve(strict=True)
    candidates: list[tuple[int, bool]] = []
    for line in mountinfo.read_text().splitlines():
        fields = line.split()
        if len(fields) < 6:
            continue
        mount_point = Path(
            fields[4]
            .replace("\\040", " ")
            .replace("\\011", "\t")
            .replace("\\012", "\n")
            .replace("\\134", "\\")
        )
        try:
            resolved.relative_to(mount_point)
        except ValueError:
            continue
        candidates.append((len(mount_point.parts), "ro" in fields[5].split(",")))
    return max(candidates, default=(0, False))[1]


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
    b_runtime: BCanaryRuntimeIdentity | None = None,
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
        "schema_version": 2,
        "producer": "qwen4b-a-repair-controller-authorization-v2",
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
        "scientific_training_authorized": False,
    }
    if any(payload.get(key) != value for key, value in required.items()):
        raise ValueError("A authorization receipt semantics mismatch")
    _validate_a_controller_chain(payload, authorization_sha256=expected_sha256)
    from common.specdec.qwen4b_a_canary_manifest import load_a_canary_manifest
    from common.specdec.qwen4b_a_canary_manifest import (
        validate_bound_artifacts as validate_a_bound_artifacts,
    )
    from common.specdec.qwen4b_a_canary_manifest import (
        validate_supervisor_completion as validate_a_supervisor_completion,
    )

    a_manifest_path = Path(str(payload.get("a_manifest_path", "")))
    if not a_manifest_path.is_absolute():
        raise ValueError("A authorization manifest path must be absolute")
    try:
        a_manifest_raw = _stable_file_bytes(a_manifest_path)
    except OSError as error:
        raise ValueError("A authorization manifest is unavailable") from error
    if hashlib.sha256(a_manifest_raw).hexdigest() != payload.get("a_manifest_sha256"):
        raise ValueError("A authorization manifest identity mismatch")
    a_manifest = load_a_canary_manifest(a_manifest_path)
    validate_a_bound_artifacts(a_manifest, repository_root=repo_root)
    runtime = a_manifest.runtime
    manifest_identity = {
        "source_commit": runtime.source_commit,
        "target_revision": runtime.target_revision,
        "target_path": runtime.target_path,
        "target_snapshot_sha256": runtime.target_snapshot_sha256,
        "target_config_sha256": runtime.target_config_sha256,
        "tokenizer_sha256": runtime.tokenizer_sha256,
        "chat_template_sha256": runtime.chat_template_sha256,
        "container_path": runtime.container_path,
        "container_sha256": runtime.container_sha256,
        "producer_module_path": runtime.producer_module_path,
        "producer_module_sha256": runtime.producer_module_sha256,
        "runner_path": runtime.runner_path,
        "runner_sha256": runtime.runner_sha256,
        "supervisor_path": runtime.supervisor_path,
        "supervisor_sha256": runtime.supervisor_sha256,
        "recipe_path": runtime.recipe_path,
        "recipe_sha256": runtime.recipe_sha256,
        "evaluator_path": runtime.evaluator_path,
        "evaluator_sha256": runtime.evaluator_sha256,
        "exporter_path": runtime.exporter_path,
        "exporter_sha256": runtime.exporter_sha256,
        "train_script_path": runtime.train_script_path,
        "train_script_sha256": runtime.train_script_sha256,
        "task9_a_selection_path": a_manifest.task9_a_selection_path,
        "task9_a_selection_sha256": a_manifest.task9_a_selection_sha256,
        "task8_publication_path": a_manifest.task8_publication_path,
        "task8_publication_sha256": a_manifest.task8_publication_sha256,
        "builder_receipt_path": a_manifest.builder_receipt_path,
        "builder_receipt_sha256": a_manifest.builder_receipt_sha256,
        "builder_output_path": a_manifest.builder_output_path,
        "builder_output_sha256": a_manifest.builder_output_sha256,
    }
    if any(payload.get(key) != value for key, value in manifest_identity.items()):
        raise ValueError("A authorization does not transitively bind its manifest")
    completion_path = Path(str(payload.get("supervisor_completion_path", "")))
    completion_raw = _stable_file_bytes(completion_path)
    if hashlib.sha256(completion_raw).hexdigest() != payload.get("supervisor_completion_sha256"):
        raise ValueError("A authorization supervisor completion identity mismatch")
    completion = validate_a_supervisor_completion(
        completion_path,
        manifest=a_manifest,
        job_id=str(payload.get("slurm_job_id", "")),
        checkpoint_path=Path(str(payload.get("checkpoint_path", ""))),
        export_path=Path(str(payload.get("export_path", ""))),
        evaluation_receipt_path=Path(str(payload.get("evaluation_receipt_path", ""))),
    )
    if payload.get("started_at") != completion.get("started_at") or payload.get(
        "finished_at"
    ) != completion.get("finished_at"):
        raise ValueError("A authorization timing is not supervisor-issued")
    if b_runtime is not None:
        shared_parent = {
            "target_revision": b_runtime.target_revision,
            "target_path": b_runtime.target_path,
            "target_snapshot_sha256": b_runtime.target_snapshot_sha256,
            "tokenizer_sha256": b_runtime.tokenizer_sha256,
            "chat_template_sha256": b_runtime.chat_template_sha256,
            "container_path": b_runtime.container_path,
            "container_sha256": b_runtime.container_sha256,
            "supervisor_sha256": b_runtime.supervisor_sha256,
            "recipe_sha256": b_runtime.config_sha256,
            "train_script_sha256": b_runtime.train_script_sha256,
            "exporter_sha256": b_runtime.exporter_sha256,
        }
        if any(payload.get(key) != value for key, value in shared_parent.items()):
            raise ValueError("A/B canonical parent or launcher identity mismatch")
    checkout_head = subprocess.run(
        ["git", "-C", str(repo_root), "rev-parse", "HEAD"],
        check=False,
        capture_output=True,
        text=True,
    )
    checkout_dirty = subprocess.run(
        ["git", "-C", str(repo_root), "status", "--porcelain"],
        check=False,
        capture_output=True,
        text=True,
    )
    if (
        checkout_head.returncode != 0
        or checkout_dirty.returncode != 0
        or checkout_head.stdout.strip() != source_commit
        or checkout_dirty.stdout
    ):
        raise ValueError("A authorization requires the exact clean source checkout")
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
        "a_manifest_sha256",
        "supervisor_completion_sha256",
        "target_snapshot_sha256",
        "target_config_sha256",
        "supervisor_sha256",
        "recipe_sha256",
        "evaluator_sha256",
        "exporter_sha256",
        "train_script_sha256",
    )
    if any(_SHA256.fullmatch(str(payload.get(name, ""))) is None for name in digest_fields):
        raise ValueError("A authorization digest lineage is incomplete")
    if _COMMIT.fullmatch(str(payload.get("target_revision", ""))) is None:
        raise ValueError("A authorization target revision is invalid")
    if not str(payload.get("slurm_job_id", "")).isdigit():
        raise ValueError("A authorization is not job-generated")
    if (
        payload.get("slurm_account") not in {"nemotron_sw_post", "nemotron_n4_post"}
        or payload.get("slurm_job_name") != "q4b-a-repair-canary"
        or not isinstance(payload.get("slurm_job_comment"), str)
        or not payload["slurm_job_comment"]
        or not Path(str(payload.get("slurm_output_path", ""))).is_absolute()
    ):
        raise ValueError("A authorization Slurm identity is incomplete")
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


def _load_self_hashed_object(raw: bytes, label: str) -> dict[str, Any]:
    payload = json.loads(raw)
    if not isinstance(payload, dict):
        raise ValueError(f"{label} must be an object")
    claim = payload.pop("receipt_sha256", None)
    if claim != _sha256_json(payload):
        raise ValueError(f"{label} self identity mismatch")
    return payload


def _validate_a_controller_chain(
    authorization: dict[str, Any],
    *,
    authorization_sha256: str,
) -> dict[str, Any]:
    """Reconcile genuine A submission, GPU completion, and controller bytes."""
    if _SHA256.fullmatch(authorization_sha256) is None:
        raise ValueError("A authorization pinned digest is invalid")
    completion_path = Path(str(authorization.get("job_completion_path", "")))
    controller_path = Path(str(authorization.get("scheduler_observation_path", "")))
    completion_raw = _stable_file_bytes(completion_path)
    controller_raw = _stable_file_bytes(controller_path)
    completion_sha = hashlib.sha256(completion_raw).hexdigest()
    controller_sha = hashlib.sha256(controller_raw).hexdigest()
    if completion_sha != authorization.get(
        "job_completion_sha256"
    ) or controller_sha != authorization.get("scheduler_observation_sha256"):
        raise ValueError("A controller authorization artifact identity mismatch")
    completion = _load_self_hashed_object(completion_raw, "A job completion")
    controller = _load_self_hashed_object(controller_raw, "A controller observation")
    submission_path = Path(str(completion.get("submission_receipt_path", "")))
    submission_raw = _stable_file_bytes(submission_path)
    submission_sha = hashlib.sha256(submission_raw).hexdigest()
    submission = _load_self_hashed_object(submission_raw, "A submission")
    if (
        submission.get("schema_version") != 2
        or submission.get("producer") != "qwen4b-a-submit-v2"
        or completion.get("schema_version") != 1
        or completion.get("producer") != "qwen4b-a-repair-job-completion-v1"
        or controller.get("schema_version") != 2
        or controller.get("producer") != "sacct-a-repair-controller-v2"
        or completion.get("submission_receipt_sha256") != submission_sha
        or controller.get("submission_receipt_sha256") != submission_sha
        or controller.get("a_job_completion_sha256") != completion_sha
    ):
        raise ValueError("A controller authorization lineage mismatch")
    job_fields = {
        "account": submission.get("slurm_account"),
        "job_name": submission.get("slurm_job_name"),
        "comment": submission.get("slurm_job_comment"),
        "stdout": submission.get("slurm_output_path"),
    }
    exact_job = {
        "slurm_job_id": submission.get("expected_gpu_job_id"),
        "slurm_account": job_fields["account"],
        "slurm_job_name": job_fields["job_name"],
        "slurm_job_comment": job_fields["comment"],
        "slurm_output_path": job_fields["stdout"],
    }
    if any(completion.get(key) != value for key, value in exact_job.items()) or any(
        authorization.get(key) != value for key, value in exact_job.items()
    ):
        raise ValueError("A controller authorization job identity mismatch")
    controller_exact = {
        "slurm_job_id": exact_job["slurm_job_id"],
        "state": "COMPLETED",
        "exit_code": "0:0",
        **job_fields,
    }
    if any(controller.get(key) != value for key, value in controller_exact.items()):
        raise ValueError("A controller observation job identity mismatch")
    if authorization.get("started_at") != completion.get("started_at") or authorization.get(
        "finished_at"
    ) != completion.get("finished_at"):
        raise ValueError("A authorization timing differs from job completion")
    return {
        "submission_sha256": submission_sha,
        "completion_sha256": completion_sha,
        "controller_observation_sha256": controller_sha,
        "controller_observation": controller,
        "job_fields": job_fields,
    }


def publish_a_scheduler_observation(
    path: Path,
    *,
    authorization_path: Path,
    authorization_sha256: str,
    sacct_output: str,
) -> str:
    """Publish a fresh no-replace scheduler observation for a completed A job."""
    authorization_raw = _stable_file_bytes(authorization_path)
    if hashlib.sha256(authorization_raw).hexdigest() != authorization_sha256:
        raise ValueError("A scheduler observation authorization identity mismatch")
    authorization = json.loads(authorization_raw)
    if not isinstance(authorization, dict):
        raise ValueError("A scheduler observation authorization must be an object")
    authorization_claim = authorization.pop("receipt_sha256", None)
    if authorization_claim != _sha256_json(authorization):
        raise ValueError("A scheduler observation authorization self identity mismatch")
    chain = _validate_a_controller_chain(
        authorization,
        authorization_sha256=authorization_sha256,
    )
    rows = [line.split("|") for line in sacct_output.splitlines() if line.strip()]
    parent_rows = [row for row in rows if row and row[0] == str(authorization.get("slurm_job_id"))]
    if len(parent_rows) != 1 or len(parent_rows[0]) != 9:
        raise ValueError("sacct must return exactly one A parent job row")
    job_id, state, exit_code, account, job_name, start, end, comment, stdout = parent_rows[0]
    expected = chain["job_fields"]
    observed = {"account": account, "job_name": job_name, "comment": comment, "stdout": stdout}

    def timestamp(value: str) -> datetime:
        parsed = datetime.fromisoformat(value)
        return parsed.replace(tzinfo=UTC) if parsed.tzinfo is None else parsed.astimezone(UTC)

    try:
        scheduler_start = timestamp(start)
        scheduler_end = timestamp(end)
        supervisor_start = timestamp(str(authorization["started_at"]))
        supervisor_end = timestamp(str(authorization["finished_at"]))
    except (KeyError, ValueError) as error:
        raise ValueError("A scheduler observation timestamps are invalid") from error
    if (
        state != "COMPLETED"
        or exit_code != "0:0"
        or account not in {"nemotron_sw_post", "nemotron_n4_post"}
        or observed != expected
        or start != chain["controller_observation"].get("start")
        or end != chain["controller_observation"].get("end")
        or not start
        or not end
        or not scheduler_start <= supervisor_start <= supervisor_end <= scheduler_end
    ):
        raise ValueError("A scheduler observation does not prove a successful canonical job")
    payload = {
        "schema_version": 1,
        "producer": "sacct-a-repair-observation-v1",
        "observed_at": datetime.now(UTC).isoformat(),
        "a_authorization_receipt_sha256": authorization_sha256,
        "a_submission_receipt_sha256": chain["submission_sha256"],
        "a_job_completion_sha256": chain["completion_sha256"],
        "a_controller_observation_sha256": chain["controller_observation_sha256"],
        "slurm_job_id": job_id,
        "state": state,
        "exit_code": exit_code,
        **observed,
        "start": start,
        "end": end,
    }
    payload["receipt_sha256"] = _sha256_json(payload)
    atomic_publish_bytes(
        path,
        (json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n").encode(),
        job_id="sacct",
    )
    return hashlib.sha256(_stable_file_bytes(path)).hexdigest()


def validate_a_scheduler_observation(
    path: Path,
    expected_sha256: str,
    *,
    authorization_path: Path,
    authorization_sha256: str,
) -> None:
    """Verify a caller-pinned fresh scheduler observation of the A parent job."""
    raw = _stable_file_bytes(path)
    if hashlib.sha256(raw).hexdigest() != expected_sha256:
        raise ValueError("A scheduler observation file identity mismatch")
    payload = json.loads(raw)
    if not isinstance(payload, dict):
        raise ValueError("A scheduler observation must be an object")
    claim = payload.pop("receipt_sha256", None)
    authorization_raw = _stable_file_bytes(authorization_path)
    if hashlib.sha256(authorization_raw).hexdigest() != authorization_sha256:
        raise ValueError("A authorization receipt file identity mismatch")
    authorization = _load_self_hashed_object(authorization_raw, "A authorization")
    chain = _validate_a_controller_chain(
        authorization,
        authorization_sha256=authorization_sha256,
    )
    expected = {
        "slurm_job_id": str(authorization.get("slurm_job_id")),
        "account": chain["job_fields"]["account"],
        "job_name": chain["job_fields"]["job_name"],
        "comment": chain["job_fields"]["comment"],
        "stdout": chain["job_fields"]["stdout"],
        "start": chain["controller_observation"]["start"],
        "end": chain["controller_observation"]["end"],
        "a_submission_receipt_sha256": chain["submission_sha256"],
        "a_job_completion_sha256": chain["completion_sha256"],
        "a_controller_observation_sha256": chain["controller_observation_sha256"],
    }
    if (
        claim != _sha256_json(payload)
        or payload.get("producer") != "sacct-a-repair-observation-v1"
        or payload.get("a_authorization_receipt_sha256") != authorization_sha256
        or payload.get("state") != "COMPLETED"
        or payload.get("exit_code") != "0:0"
        or payload.get("account") not in {"nemotron_sw_post", "nemotron_n4_post"}
        or any(payload.get(key) != value for key, value in expected.items())
    ):
        raise ValueError("A scheduler observation semantics mismatch")


def publish_b_canary_evidence(
    path: Path,
    manifest: BCanaryManifest,
    *,
    job_id: str,
    checkpoint_path: Path,
    export_path: Path,
    exporter_receipt_path: Path,
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
    exporter_receipt_raw = _stable_file_bytes(exporter_receipt_path)
    exporter_receipt_sha = hashlib.sha256(exporter_receipt_raw).hexdigest()
    exporter_receipt = json.loads(exporter_receipt_raw)
    if not isinstance(exporter_receipt, dict):
        raise ValueError("B exporter invocation must be an object")
    validate_b_exporter_invocation(
        exporter_receipt_path,
        exporter_receipt_sha,
        manifest,
        job_id=job_id,
        exporter_path=Path(str(exporter_receipt.get("exporter_path", ""))),
        checkpoint_path=checkpoint_path,
        export_path=export_path,
    )
    supervisor_raw = _stable_file_bytes(supervisor_receipt_path)
    supervisor = json.loads(supervisor_raw)
    supervisor_claim = (
        supervisor.pop("receipt_sha256", None) if isinstance(supervisor, dict) else None
    )
    evaluation_raw = _stable_file_bytes(evaluation_receipt_path)
    if (
        not isinstance(supervisor, dict)
        or supervisor_claim != _sha256_json(supervisor)
        or supervisor.get("producer") != "canonical-dflash-exporter-reload-v1"
        or str(supervisor.get("slurm_job_id")) != job_id
        or supervisor.get("status") != "completed"
        or supervisor.get("supervisor_sha256") != manifest.runtime.supervisor_sha256
        or supervisor.get("config_sha256") != manifest.runtime.config_sha256
        or supervisor.get("exporter_sha256") != manifest.runtime.exporter_sha256
        or supervisor.get("exporter_invocation_sha256") != exporter_receipt_sha
        or supervisor.get("evaluation_receipt_sha256") != hashlib.sha256(evaluation_raw).hexdigest()
        or supervisor.get("checkpoint_sha256") != checkpoint_sha
        or supervisor.get("export_sha256") != export_sha
        or supervisor.get("checkpoint_reloaded_by_exporter") is not True
        or supervisor.get("final_training_loss") != finite_loss
    ):
        raise ValueError("B canonical supervisor completion identity mismatch")
    _validate_evaluator_receipt(
        evaluation_raw,
        job_id=job_id,
        export_sha256=export_sha,
        manifest=manifest,
        checkpoint_path=checkpoint_path,
        exporter_receipt_sha256=exporter_receipt_sha,
    )
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
        "exporter_invocation_sha256": exporter_receipt_sha,
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


def publish_b_exporter_invocation(
    path: Path,
    manifest: BCanaryManifest,
    *,
    job_id: str,
    exporter_path: Path,
    checkpoint_path: Path,
    export_path: Path,
    argv: list[str],
    started_at: str,
    finished_at: str,
) -> None:
    """Record the exact current-job exporter invocation after its output exists."""
    if not job_id.isdigit():
        raise ValueError("B exporter invocation requires a numeric Slurm job ID")
    expected_checkpoint = Path(manifest.runtime.output_root) / "checkpoint" / "checkpoint-200"
    expected_export = Path(manifest.runtime.output_root) / "export"
    expected_exporter = _canonical_b_exporter_path(manifest.runtime)
    if checkpoint_path != expected_checkpoint or checkpoint_path.name != "checkpoint-200":
        raise ValueError("B exporter must reload the exact checkpoint-200 child")
    if export_path != expected_export:
        raise ValueError("B exporter output path identity mismatch")
    if exporter_path != expected_exporter:
        raise ValueError("B exporter source path is not canonical")
    expected_argv = [
        str(exporter_path),
        "--model_path",
        str(checkpoint_path),
        "--export_path",
        str(export_path),
    ]
    if argv != expected_argv:
        raise ValueError("B exporter argv identity mismatch")
    try:
        started = datetime.fromisoformat(started_at)
        finished = datetime.fromisoformat(finished_at)
    except ValueError as error:
        raise ValueError("B exporter timestamps are invalid") from error
    if started.tzinfo is None or finished.tzinfo is None or started >= finished:
        raise ValueError("B exporter timestamps are not ordered")
    exporter_sha = hashlib.sha256(_stable_file_bytes(exporter_path)).hexdigest()
    if exporter_sha != manifest.runtime.exporter_sha256:
        raise ValueError("B exporter source identity mismatch")
    export_config = json.loads(_stable_file_bytes(export_path / "config.json"))
    if not isinstance(export_config, dict) or not tuple(export_path.glob("*.safetensors")):
        raise ValueError("B exporter did not produce a loadable HF artifact")
    payload = {
        "schema_version": 1,
        "producer": "qwen4b-b-exporter-invocation-v1",
        "slurm_job_id": job_id,
        "status": "completed",
        "exporter_path": str(exporter_path),
        "exporter_sha256": exporter_sha,
        "argv": argv,
        "model_path": str(checkpoint_path),
        "checkpoint_sha256": _directory_sha256(checkpoint_path),
        "export_path": str(export_path),
        "export_sha256": _directory_sha256(export_path),
        "started_at": started_at,
        "finished_at": finished_at,
    }
    payload["receipt_sha256"] = _sha256_json(payload)
    atomic_publish_bytes(
        path,
        (json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n").encode(),
        job_id=job_id,
    )


def validate_b_exporter_invocation(
    path: Path,
    expected_sha256: str,
    manifest: BCanaryManifest,
    *,
    job_id: str,
    exporter_path: Path,
    checkpoint_path: Path,
    export_path: Path,
) -> dict[str, Any]:
    """Revalidate a caller-pinned exact checkpoint-200 exporter invocation."""
    raw = _stable_file_bytes(path)
    if hashlib.sha256(raw).hexdigest() != expected_sha256:
        raise ValueError("B exporter invocation file identity mismatch")
    payload = json.loads(raw)
    if not isinstance(payload, dict):
        raise ValueError("B exporter invocation must be an object")
    claim = payload.pop("receipt_sha256", None)
    if exporter_path != _canonical_b_exporter_path(manifest.runtime):
        raise ValueError("B exporter source path is not canonical")
    expected = {
        "schema_version": 1,
        "producer": "qwen4b-b-exporter-invocation-v1",
        "slurm_job_id": job_id,
        "status": "completed",
        "exporter_path": str(exporter_path),
        "exporter_sha256": manifest.runtime.exporter_sha256,
        "argv": [
            str(exporter_path),
            "--model_path",
            str(checkpoint_path),
            "--export_path",
            str(export_path),
        ],
        "model_path": str(checkpoint_path),
        "checkpoint_sha256": _directory_sha256(checkpoint_path),
        "export_path": str(export_path),
        "export_sha256": _directory_sha256(export_path),
    }
    if claim != _sha256_json(payload) or any(
        payload.get(key) != value for key, value in expected.items()
    ):
        raise ValueError("B exporter invocation semantics mismatch")
    if (
        hashlib.sha256(_stable_file_bytes(exporter_path)).hexdigest()
        != manifest.runtime.exporter_sha256
    ):
        raise ValueError("B exporter source identity changed")
    return payload


def _canonical_b_exporter_path(runtime: BCanaryRuntimeIdentity) -> Path:
    supervisor = Path(runtime.supervisor_path)
    suffix = Path("tools/launcher/common/eagle3/train_eagle_streaming.sh")
    if (
        not supervisor.is_absolute()
        or tuple(supervisor.parts[-len(suffix.parts) :]) != suffix.parts
    ):
        raise ValueError("B supervisor path cannot derive the canonical exporter")
    repo_root = supervisor.parents[len(suffix.parts) - 1]
    return repo_root / "examples/speculative_decoding/scripts/export_hf_checkpoint.py"


def publish_b_supervisor_completion(
    path: Path,
    manifest: BCanaryManifest,
    *,
    job_id: str,
    checkpoint_path: Path,
    export_path: Path,
    exporter_receipt_path: Path,
    evaluation_receipt_path: Path,
    started_at: str,
    finished_at: str,
) -> None:
    """Record successful canonical trainer/exporter completion in the current job."""
    if not job_id.isdigit():
        raise ValueError("B supervisor completion requires a numeric Slurm job ID")
    try:
        started = datetime.fromisoformat(started_at)
        finished = datetime.fromisoformat(finished_at)
    except ValueError as error:
        raise ValueError("B supervisor completion timestamps are invalid") from error
    if started.tzinfo is None or finished.tzinfo is None or started >= finished:
        raise ValueError("B supervisor completion timestamps are not ordered")
    state = json.loads(_stable_file_bytes(checkpoint_path / "trainer_state.json"))
    final_loss = _last_finite_training_loss(state, expected_step=manifest.topology.max_steps)
    export_config = json.loads(_stable_file_bytes(export_path / "config.json"))
    if not isinstance(export_config, dict) or not tuple(export_path.glob("*.safetensors")):
        raise ValueError("B canonical exporter did not produce a loadable HF artifact")
    exporter_receipt_raw = _stable_file_bytes(exporter_receipt_path)
    exporter_receipt_sha = hashlib.sha256(exporter_receipt_raw).hexdigest()
    exporter_receipt = json.loads(exporter_receipt_raw)
    if not isinstance(exporter_receipt, dict):
        raise ValueError("B exporter invocation must be an object")
    invocation = validate_b_exporter_invocation(
        exporter_receipt_path,
        exporter_receipt_sha,
        manifest,
        job_id=job_id,
        exporter_path=Path(str(exporter_receipt.get("exporter_path", ""))),
        checkpoint_path=checkpoint_path,
        export_path=export_path,
    )
    try:
        exporter_started = datetime.fromisoformat(str(invocation["started_at"]))
        exporter_finished = datetime.fromisoformat(str(invocation["finished_at"]))
    except (KeyError, ValueError) as error:
        raise ValueError("B exporter invocation timestamps are invalid") from error
    if not started <= exporter_started <= exporter_finished <= finished:
        raise ValueError("B supervisor completion does not enclose the exporter invocation")
    evaluation_raw = _stable_file_bytes(evaluation_receipt_path)
    _validate_evaluator_receipt(
        evaluation_raw,
        job_id=job_id,
        export_sha256=_directory_sha256(export_path),
        manifest=manifest,
        checkpoint_path=checkpoint_path,
        exporter_receipt_sha256=exporter_receipt_sha,
    )
    payload = {
        "schema_version": 1,
        "producer": "canonical-dflash-exporter-reload-v1",
        "slurm_job_id": job_id,
        "status": "completed",
        "supervisor_sha256": manifest.runtime.supervisor_sha256,
        "config_sha256": manifest.runtime.config_sha256,
        "exporter_sha256": manifest.runtime.exporter_sha256,
        "exporter_invocation_path": str(exporter_receipt_path),
        "exporter_invocation_sha256": exporter_receipt_sha,
        "evaluation_receipt_path": str(evaluation_receipt_path),
        "evaluation_receipt_sha256": hashlib.sha256(evaluation_raw).hexdigest(),
        "started_at": started_at,
        "finished_at": finished_at,
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
        and entry.get("step") == expected_step
        and isinstance(entry.get("loss"), (int, float))
        and not isinstance(entry.get("loss"), bool)
        and math.isfinite(float(entry["loss"]))
    ]
    if not losses:
        raise ValueError(f"training output lacks finite loss at step {expected_step}")
    return losses[-1]


def _validate_evaluator_receipt(
    raw: bytes,
    *,
    job_id: str,
    export_sha256: str,
    manifest: BCanaryManifest | None = None,
    checkpoint_path: Path | None = None,
    exporter_receipt_sha256: str | None = None,
) -> None:
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
    if manifest is not None:
        expected_reload = {
            "checkpoint_path": str(checkpoint_path),
            "checkpoint_sha256": (
                None if checkpoint_path is None else _directory_sha256(checkpoint_path)
            ),
            "exporter_path": "examples/speculative_decoding/scripts/export_hf_checkpoint.py",
            "exporter_sha256": manifest.runtime.exporter_sha256,
            "exporter_invocation_sha256": exporter_receipt_sha256,
            "checkpoint_reloaded_by_exporter": True,
        }
        if (
            checkpoint_path is None
            or exporter_receipt_sha256 is None
            or any(evaluation.get(key) != value for key, value in expected_reload.items())
        ):
            raise ValueError("B evaluator exporter/reload evidence mismatch")


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
