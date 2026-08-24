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
from datetime import datetime
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
    target_path: str
    target_snapshot_sha256: str
    target_weight_set_sha256: str
    target_config_sha256: str
    tokenizer_sha256: str
    chat_template_sha256: str
    container_path: str
    container_sha256: str
    wandb_netrc_sha256: str
    wandb_dir: str
    wandb_cache_dir: str
    wandb_config_dir: str
    wandb_artifact_dir: str
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
    exporter_path: str
    exporter_sha256: str
    train_script_path: str
    train_script_sha256: str
    sequential_sampler_path: str
    sequential_sampler_sha256: str
    modelopt_runtime_path: str
    modelopt_runtime_sha256: str
    speculators_runtime_path: str
    speculators_runtime_sha256: str
    speculators_repo_path: str
    speculators_repo_sha256: str
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
            self.target_snapshot_sha256,
            self.target_weight_set_sha256,
            self.target_config_sha256,
            self.tokenizer_sha256,
            self.chat_template_sha256,
            self.container_sha256,
            self.wandb_netrc_sha256,
            self.producer_module_sha256,
            self.runner_sha256,
            self.supervisor_sha256,
            self.recipe_sha256,
            self.evaluator_sha256,
            self.exporter_sha256,
            self.train_script_sha256,
            self.sequential_sampler_sha256,
            self.modelopt_runtime_sha256,
            self.speculators_runtime_sha256,
            self.speculators_repo_sha256,
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
        absolute_paths = (
            self.target_path,
            self.container_path,
            self.wandb_dir,
            self.wandb_cache_dir,
            self.wandb_config_dir,
            self.wandb_artifact_dir,
            self.modelopt_runtime_path,
            self.speculators_runtime_path,
            self.speculators_repo_path,
        )
        if any(not Path(value).is_absolute() for value in absolute_paths):
            raise ValueError("A runtime host paths must be absolute")
        if len(set(absolute_paths)) != len(absolute_paths):
            raise ValueError("A runtime host paths must be distinct")


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
        (manifest.runtime.exporter_path, manifest.runtime.exporter_sha256),
        (manifest.runtime.train_script_path, manifest.runtime.train_script_sha256),
        (manifest.runtime.sequential_sampler_path, manifest.runtime.sequential_sampler_sha256),
    )
    for relative, digest in runtime_files:
        path = repository_root / relative
        if hashlib.sha256(_stable_file_bytes(path)).hexdigest() != digest:
            raise ValueError("A runtime source identity mismatch")
    snapshot_target_identity(
        Path(manifest.runtime.target_path),
        container_path=Path(manifest.runtime.container_path),
        expected=manifest.runtime,
    )
    for path_value, expected_digest in (
        (manifest.runtime.modelopt_runtime_path, manifest.runtime.modelopt_runtime_sha256),
        (manifest.runtime.speculators_runtime_path, manifest.runtime.speculators_runtime_sha256),
        (manifest.runtime.speculators_repo_path, manifest.runtime.speculators_repo_sha256),
    ):
        if _tree_sha256(Path(path_value)) != expected_digest:
            raise ValueError("A bound runtime tree identity mismatch")
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
        or builder.get("tokenizer_sha256") != manifest.runtime.tokenizer_sha256
        or builder.get("chat_template_sha256") != manifest.runtime.chat_template_sha256
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
        and entry.get("step") == expected_step
        and isinstance(entry.get("loss"), (int, float))
        and not isinstance(entry.get("loss"), bool)
        and math.isfinite(float(entry["loss"]))
    ]
    if not candidates:
        raise ValueError(f"trainer state lacks finite training loss at step {expected_step}")
    return candidates[-1]


def training_phase_boundaries(topology: ACanaryTopology) -> dict[str, int]:
    """Return the exact source-order phase boundary for the 200-step A canary."""
    historical_occurrences = 66_560
    complement_occurrences = 35_840
    if historical_occurrences % topology.global_batch_size or complement_occurrences % topology.global_batch_size:
        raise ValueError("A phase quotas must align to global batches")
    historical_steps = historical_occurrences // topology.global_batch_size
    complement_steps = complement_occurrences // topology.global_batch_size
    if historical_steps + complement_steps != topology.max_steps:
        raise ValueError("A phase steps do not cover the canary")
    return {
        "historical_occurrences": historical_occurrences,
        "historical_steps": historical_steps,
        "complement_occurrences": complement_occurrences,
        "complement_steps": complement_steps,
        "total_steps": topology.max_steps,
    }


def snapshot_target_identity(
    target_path: Path,
    *,
    container_path: Path,
    target_revision: str | None = None,
    expected: ACanaryRuntimeIdentity | None = None,
) -> dict[str, str]:
    """Hash the exact local Qwen3-4B snapshot, tokenizer, template, and container bytes."""
    if not target_path.is_absolute() or not container_path.is_absolute():
        raise ValueError("target and container paths must be absolute")
    config_path = target_path / "config.json"
    tokenizer_config_path = target_path / "tokenizer_config.json"
    config_raw = _stable_file_bytes(config_path)
    tokenizer_config_raw = _stable_file_bytes(tokenizer_config_path)
    try:
        config = json.loads(config_raw)
        tokenizer_config = json.loads(tokenizer_config_raw)
    except json.JSONDecodeError as error:
        raise ValueError("target snapshot JSON is invalid") from error
    exact_dims = {
        "model_type": "qwen3",
        "hidden_size": 2560,
        "num_hidden_layers": 36,
        "num_attention_heads": 32,
        "num_key_value_heads": 8,
        "head_dim": 128,
        "intermediate_size": 9728,
        "vocab_size": 151936,
    }
    if not isinstance(config, dict) or any(config.get(key) != value for key, value in exact_dims.items()):
        raise ValueError("target snapshot is not the exact Qwen3-4B configuration")
    revision = expected.target_revision if expected is not None else target_revision
    if revision is not None:
        if _COMMIT.fullmatch(revision) is None:
            raise ValueError("target revision must be exact")
        snapshot_revision = target_path.resolve().name
        if snapshot_revision != revision:
            raise ValueError("target snapshot revision mismatch")
    if not isinstance(tokenizer_config, dict):
        raise ValueError("target snapshot tokenizer configuration is invalid")
    template_raw = _stable_file_bytes(target_path / "chat_template.jinja")
    tokenizer_names = {
        "chat_template.jinja",
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "added_tokens.json",
        "vocab.json",
        "vocab.txt",
        "tokenizer.model",
        "merges.txt",
    }
    tokenizer_files = sorted(
        path for path in target_path.iterdir() if path.is_file() and path.name in tokenizer_names
    )
    if not tokenizer_files or not any(path.name == "tokenizer.json" for path in tokenizer_files):
        raise ValueError("target snapshot has no complete tokenizer bytes")
    tokenizer_digest = hashlib.sha256()
    for artifact in tokenizer_files:
        tokenizer_digest.update(artifact.name.encode())
        tokenizer_digest.update(b"\0")
        tokenizer_digest.update(bytes.fromhex(hashlib.sha256(_stable_file_bytes(artifact)).hexdigest()))
    weight_digest = _target_weight_set_sha256(target_path)
    actual = {
        "target_snapshot_sha256": _directory_sha256(target_path),
        "target_weight_set_sha256": weight_digest,
        "target_config_sha256": hashlib.sha256(config_raw).hexdigest(),
        "tokenizer_sha256": tokenizer_digest.hexdigest(),
        "chat_template_sha256": hashlib.sha256(template_raw).hexdigest(),
        "container_sha256": hashlib.sha256(_stable_file_bytes(container_path)).hexdigest(),
    }
    if expected is not None and any(getattr(expected, key) != value for key, value in actual.items()):
        raise ValueError("target snapshot or container identity mismatch")
    return actual


def _target_weight_set_sha256(target_path: Path) -> str:
    """Hash an exact single-file or indexed safetensors weight set."""
    single = target_path / "model.safetensors"
    index = target_path / "model.safetensors.index.json"
    all_weights = {path.name for path in target_path.glob("*.safetensors") if path.is_file()}
    if single.is_file() and not index.exists():
        if all_weights != {single.name}:
            raise ValueError("target snapshot has extra model weight files")
        declared = [single]
        index_raw = b""
    elif index.is_file() and not single.exists():
        index_raw = _stable_file_bytes(index)
        try:
            body = json.loads(index_raw)
        except json.JSONDecodeError as error:
            raise ValueError("target weight index is invalid") from error
        weight_map = body.get("weight_map") if isinstance(body, dict) else None
        if not isinstance(weight_map, dict) or not weight_map:
            raise ValueError("target weight index is invalid")
        shard_names = sorted(set(weight_map.values()))
        if any(
            not isinstance(name, str)
            or Path(name).name != name
            or not name.endswith(".safetensors")
            for name in shard_names
        ):
            raise ValueError("target weight shard declaration is invalid")
        if any(not (target_path / name).is_file() for name in shard_names):
            raise ValueError("target weight shard is missing")
        if all_weights != set(shard_names):
            raise ValueError("target snapshot has extra model weight files")
        declared = [target_path / name for name in shard_names]
    else:
        raise ValueError("target snapshot must contain one exact safetensors weight set")
    digest = hashlib.sha256()
    if index_raw:
        digest.update(index.name.encode())
        digest.update(b"\0")
        digest.update(bytes.fromhex(hashlib.sha256(index_raw).hexdigest()))
    for artifact in declared:
        digest.update(artifact.name.encode())
        digest.update(b"\0")
        digest.update(bytes.fromhex(hashlib.sha256(_stable_file_bytes(artifact)).hexdigest()))
    return digest.hexdigest()


def validate_runtime_mounts(
    runtime: ACanaryRuntimeIdentity,
    *,
    repository_root: Path,
    environ: dict[str, str] | None = None,
) -> None:
    """Fail unless W&B secret and durable state mounts match the signed manifest."""
    environment = os.environ if environ is None else environ
    netrc_path = Path(environment.get("WANDB_NETRC_PATH", ""))
    if netrc_path != Path("/run/secrets/wandb.netrc"):
        raise ValueError("W&B netrc must be mounted at the pinned secret path")
    if hashlib.sha256(_stable_file_bytes(netrc_path)).hexdigest() != runtime.wandb_netrc_sha256:
        raise ValueError("W&B netrc identity mismatch")
    if not _path_on_read_only_mount(netrc_path):
        raise ValueError("W&B netrc mount must be read-only")
    job_id = environment.get("SLURM_JOB_ID", "")
    if not job_id.isdigit():
        raise ValueError("W&B paths require the current Slurm job ID")
    roots = {
        "WANDB_DIR": (runtime.wandb_dir, "run"),
        "WANDB_CACHE_DIR": (runtime.wandb_cache_dir, "cache"),
        "WANDB_CONFIG_DIR": (runtime.wandb_config_dir, "config"),
        "WANDB_ARTIFACT_DIR": (runtime.wandb_artifact_dir, "artifacts"),
    }
    if any(not Path(root).as_posix().startswith("/lustre/") for root, _ in roots.values()):
        raise ValueError("W&B roots must be pinned to Lustre")
    if "/scratch/" not in Path(runtime.wandb_cache_dir).as_posix():
        raise ValueError("W&B cache must use the separately bound scratch root")
    expected_dirs = {
        name: str(Path(root) / "jobs" / job_id / leaf)
        for name, (root, leaf) in roots.items()
    }
    repo = repository_root.resolve()
    for name, expected_path in expected_dirs.items():
        path = Path(environment.get(name, ""))
        if path != Path(expected_path) or not path.is_absolute() or not path.is_dir():
            raise ValueError(f"{name} durable mount mismatch")
        try:
            path.resolve().relative_to(repo)
        except ValueError:
            pass
        else:
            raise ValueError(f"{name} must be outside the repository")
        probe = path / f".qwen4b-a-mount-probe-{os.getpid()}"
        descriptor = os.open(probe, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        os.close(descriptor)
        probe.unlink()


def _path_on_read_only_mount(path: Path) -> bool:
    """Return whether Linux mountinfo resolves the path to a read-only mount."""
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


def publish_a_supervisor_completion(
    path: Path,
    manifest: ACanaryManifest,
    *,
    job_id: str,
    checkpoint_path: Path,
    export_path: Path,
    evaluation_receipt_path: Path,
    started_at: str,
    finished_at: str,
) -> None:
    """Issue current-job proof that the canonical supervisor exported and reloaded step 200."""
    if not job_id.isdigit():
        raise ValueError("A supervisor completion requires a numeric Slurm job ID")
    state = json.loads(_stable_file_bytes(checkpoint_path / "trainer_state.json"))
    loss = last_finite_training_loss(state, expected_step=manifest.topology.max_steps)
    checkpoint_sha = _directory_sha256(checkpoint_path)
    export_sha = _directory_sha256(export_path)
    evaluation_raw = _stable_file_bytes(evaluation_receipt_path)
    validate_evaluator_receipt(
        evaluation_receipt_path,
        job_id=job_id,
        export_sha256=export_sha,
        manifest=manifest,
        checkpoint_path=checkpoint_path,
    )
    _validate_job_times(started_at, finished_at)
    body: dict[str, Any] = {
        "schema_version": 1,
        "producer": "qwen4b-a-canonical-supervisor-completion-v1",
        "status": "completed",
        "slurm_job_id": job_id,
        "started_at": started_at,
        "finished_at": finished_at,
        "source_commit": manifest.runtime.source_commit,
        "supervisor_path": manifest.runtime.supervisor_path,
        "supervisor_sha256": manifest.runtime.supervisor_sha256,
        "recipe_path": manifest.runtime.recipe_path,
        "recipe_sha256": manifest.runtime.recipe_sha256,
        "runner_sha256": manifest.runtime.runner_sha256,
        "exporter_path": manifest.runtime.exporter_path,
        "exporter_sha256": manifest.runtime.exporter_sha256,
        "train_script_path": manifest.runtime.train_script_path,
        "train_script_sha256": manifest.runtime.train_script_sha256,
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_sha256": checkpoint_sha,
        "checkpoint_step": manifest.topology.max_steps,
        "last_finite_loss": loss,
        "export_path": str(export_path),
        "export_sha256": export_sha,
        "export_model_path": str(checkpoint_path),
        "checkpoint_reloaded_by_exporter": True,
        "evaluation_receipt_path": str(evaluation_receipt_path),
        "evaluation_receipt_sha256": hashlib.sha256(evaluation_raw).hexdigest(),
    }
    body["receipt_sha256"] = _sha256_json(body)
    atomic_publish_bytes(path, (_canonical_json(body) + "\n").encode(), job_id=job_id)


def validate_supervisor_completion(
    path: Path,
    *,
    manifest: ACanaryManifest,
    job_id: str,
    checkpoint_path: Path,
    export_path: Path,
    evaluation_receipt_path: Path,
) -> dict[str, Any]:
    """Rehash and validate canonical supervisor/export/reload completion evidence."""
    body = json.loads(_stable_file_bytes(path))
    if not isinstance(body, dict):
        raise ValueError("A supervisor completion receipt is invalid")
    claim = body.pop("receipt_sha256", None)
    state = json.loads(_stable_file_bytes(checkpoint_path / "trainer_state.json"))
    loss = last_finite_training_loss(state, expected_step=manifest.topology.max_steps)
    expected = {
        "producer": "qwen4b-a-canonical-supervisor-completion-v1",
        "status": "completed",
        "slurm_job_id": job_id,
        "source_commit": manifest.runtime.source_commit,
        "supervisor_path": manifest.runtime.supervisor_path,
        "supervisor_sha256": manifest.runtime.supervisor_sha256,
        "recipe_path": manifest.runtime.recipe_path,
        "recipe_sha256": manifest.runtime.recipe_sha256,
        "runner_sha256": manifest.runtime.runner_sha256,
        "exporter_path": manifest.runtime.exporter_path,
        "exporter_sha256": manifest.runtime.exporter_sha256,
        "train_script_path": manifest.runtime.train_script_path,
        "train_script_sha256": manifest.runtime.train_script_sha256,
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_sha256": _directory_sha256(checkpoint_path),
        "checkpoint_step": manifest.topology.max_steps,
        "last_finite_loss": loss,
        "export_path": str(export_path),
        "export_sha256": _directory_sha256(export_path),
        "export_model_path": str(checkpoint_path),
        "checkpoint_reloaded_by_exporter": True,
        "evaluation_receipt_path": str(evaluation_receipt_path),
        "evaluation_receipt_sha256": hashlib.sha256(_stable_file_bytes(evaluation_receipt_path)).hexdigest(),
    }
    started_at, finished_at = body.get("started_at"), body.get("finished_at")
    if not isinstance(started_at, str) or not isinstance(finished_at, str):
        raise ValueError("A supervisor completion timing is missing")
    _validate_job_times(started_at, finished_at)
    if claim != _sha256_json(body) or any(body.get(key) != value for key, value in expected.items()):
        raise ValueError("A supervisor completion receipt mismatch")
    validate_evaluator_receipt(
        evaluation_receipt_path,
        job_id=job_id,
        export_sha256=expected["export_sha256"],
        manifest=manifest,
        checkpoint_path=checkpoint_path,
    )
    return body


def _validate_job_times(started_at: str, finished_at: str) -> None:
    try:
        started = datetime.fromisoformat(started_at.replace("Z", "+00:00"))
        finished = datetime.fromisoformat(finished_at.replace("Z", "+00:00"))
    except ValueError as error:
        raise ValueError("A supervisor completion timing is invalid") from error
    if started.tzinfo is None or finished.tzinfo is None or finished < started:
        raise ValueError("A supervisor completion timing is invalid")


def validate_evaluator_receipt(
    path: Path,
    *,
    job_id: str,
    export_sha256: str,
    manifest: ACanaryManifest | None = None,
    checkpoint_path: Path | None = None,
) -> dict[str, Any]:
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
    if manifest is not None:
        exporter_evidence = {
            "checkpoint_path": str(checkpoint_path),
            "checkpoint_sha256": (
                None if checkpoint_path is None else _directory_sha256(checkpoint_path)
            ),
            "exporter_path": manifest.runtime.exporter_path,
            "exporter_sha256": manifest.runtime.exporter_sha256,
            "checkpoint_reloaded_by_exporter": True,
        }
        if checkpoint_path is None or any(
            body.get(key) != value for key, value in exporter_evidence.items()
        ):
            raise ValueError("A evaluator exporter/reload evidence mismatch")
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


def publish_a_job_completion(
    path: Path,
    manifest: ACanaryManifest,
    *,
    job_id: str,
    checkpoint_path: Path,
    export_path: Path,
    gpu_evidence_path: Path,
    evaluation_receipt_path: Path,
    supervisor_completion_path: Path,
    manifest_path: Path,
    repository_root: Path,
    submission_receipt_path: Path,
) -> None:
    """Publish current-job evidence; final authorization is controller-issued post-exit."""
    if not job_id.isdigit():
        raise ValueError("A authorization requires a numeric Slurm job ID")
    if os.environ.get("SLURM_JOB_ID") != job_id or os.environ.get("SLURM_NNODES") != "16":
        raise ValueError("A authorization requires the current 16-node Slurm context")
    slurm_account = os.environ.get("SLURM_JOB_ACCOUNT")
    slurm_job_name = os.environ.get("SLURM_JOB_NAME")
    slurm_job_comment = os.environ.get("A_CANARY_JOB_COMMENT")
    slurm_output_path = os.environ.get("A_CANARY_SLURM_OUTPUT")
    if (
        slurm_account not in {"nemotron_sw_post", "nemotron_n4_post"}
        or slurm_job_name != "q4b-a-repair-canary"
        or not slurm_job_comment
        or not slurm_output_path
        or not Path(slurm_output_path).is_absolute()
    ):
        raise ValueError("A authorization Slurm identity is invalid")
    manifest_raw = _stable_file_bytes(manifest_path)
    if load_a_canary_manifest(manifest_path) != manifest:
        raise ValueError("A authorization manifest object mismatch")
    validate_bound_artifacts(manifest, repository_root=repository_root)
    validate_runtime_mounts(manifest.runtime, repository_root=repository_root)
    trainer_state = json.loads(_stable_file_bytes(checkpoint_path / "trainer_state.json"))
    finite_loss = last_finite_training_loss(trainer_state, expected_step=200)
    checkpoint_sha = _directory_sha256(checkpoint_path)
    export_sha = _directory_sha256(export_path)
    evaluation_raw = _stable_file_bytes(evaluation_receipt_path)
    validate_evaluator_receipt(
        evaluation_receipt_path,
        job_id=job_id,
        export_sha256=export_sha,
        manifest=manifest,
        checkpoint_path=checkpoint_path,
    )
    completion_raw = _stable_file_bytes(supervisor_completion_path)
    completion = validate_supervisor_completion(
        supervisor_completion_path,
        manifest=manifest,
        job_id=job_id,
        checkpoint_path=checkpoint_path,
        export_path=export_path,
        evaluation_receipt_path=evaluation_receipt_path,
    )
    gpu_raw = _stable_file_bytes(gpu_evidence_path)
    gpu = json.loads(gpu_raw)
    if (
        not isinstance(gpu, dict)
        or str(gpu.get("slurm_job_id")) != job_id
        or tuple(gpu.get("active_gpu_ranks", ())) != tuple(range(64))
    ):
        raise ValueError("A GPU evidence does not cover all 64 ranks")
    runtime = manifest.runtime
    submission_raw = _stable_file_bytes(submission_receipt_path)
    submission = _load_self_hashed(submission_receipt_path, "A submission")
    expected_submission = {
        "producer": "qwen4b-a-submit-v2",
        "expected_gpu_job_id": job_id,
        "slurm_account": slurm_account,
        "slurm_job_name": slurm_job_name,
        "slurm_job_comment": slurm_job_comment,
        "slurm_output_path": slurm_output_path,
        "source_commit": runtime.source_commit,
        "manifest_path": str(manifest_path),
    }
    if any(submission.get(key) != value for key, value in expected_submission.items()):
        raise ValueError("A submission receipt does not authorize this GPU job")
    payload: dict[str, Any] = {
        "schema_version": 1,
        "producer": "qwen4b-a-repair-job-completion-v1",
        "arm": "A-repair",
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
        "checkpoint_reloaded": completion["checkpoint_reloaded_by_exporter"],
        "drafter_exported": True,
        "all_gpus_active": True,
        "finite_loss": finite_loss,
        "slurm_job_id": job_id,
        "slurm_account": slurm_account,
        "slurm_job_name": slurm_job_name,
        "slurm_job_comment": slurm_job_comment,
        "slurm_output_path": slurm_output_path,
        "started_at": completion["started_at"],
        "finished_at": completion["finished_at"],
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
        "sequential_sampler_path": runtime.sequential_sampler_path,
        "sequential_sampler_sha256": runtime.sequential_sampler_sha256,
        "modelopt_runtime_path": runtime.modelopt_runtime_path,
        "modelopt_runtime_sha256": runtime.modelopt_runtime_sha256,
        "speculators_runtime_path": runtime.speculators_runtime_path,
        "speculators_runtime_sha256": runtime.speculators_runtime_sha256,
        "speculators_repo_path": runtime.speculators_repo_path,
        "speculators_repo_sha256": runtime.speculators_repo_sha256,
        "target_path": runtime.target_path,
        "target_snapshot_sha256": runtime.target_snapshot_sha256,
        "target_weight_set_sha256": runtime.target_weight_set_sha256,
        "target_config_sha256": runtime.target_config_sha256,
        "container_path": runtime.container_path,
        "wandb_netrc_sha256": runtime.wandb_netrc_sha256,
        "wandb_dir": os.environ.get("WANDB_DIR"),
        "wandb_cache_dir": os.environ.get("WANDB_CACHE_DIR"),
        "wandb_config_dir": os.environ.get("WANDB_CONFIG_DIR"),
        "wandb_artifact_dir": os.environ.get("WANDB_ARTIFACT_DIR"),
        "a_manifest_path": str(manifest_path),
        "a_manifest_sha256": hashlib.sha256(manifest_raw).hexdigest(),
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_sha256": checkpoint_sha,
        "export_path": str(export_path),
        "export_sha256": export_sha,
        "gpu_evidence_path": str(gpu_evidence_path),
        "gpu_evidence_sha256": hashlib.sha256(gpu_raw).hexdigest(),
        "evaluation_receipt_path": str(evaluation_receipt_path),
        "evaluation_receipt_sha256": hashlib.sha256(evaluation_raw).hexdigest(),
        "supervisor_completion_path": str(supervisor_completion_path),
        "supervisor_completion_sha256": hashlib.sha256(completion_raw).hexdigest(),
        "export_model_path": str(checkpoint_path),
        "checkpoint_step": 200,
        "loss_step": 200,
    }
    payload["submission_receipt_path"] = str(submission_receipt_path)
    payload["submission_receipt_sha256"] = hashlib.sha256(submission_raw).hexdigest()
    payload["receipt_sha256"] = _sha256_json(payload)
    atomic_publish_bytes(path, (_canonical_json(payload) + "\n").encode(), job_id=job_id)


def publish_a_submission_receipt(
    path: Path,
    *,
    source_commit: str,
    builder_job_id: str,
    gpu_job_id: str,
    account: str,
    job_comment: str,
    stdout_path: str,
    manifest_path: Path,
) -> None:
    """Pin the scheduler-returned GPU identity before that job can emit evidence."""
    if (
        _COMMIT.fullmatch(source_commit) is None
        or not builder_job_id.isdigit()
        or not gpu_job_id.isdigit()
        or account not in {"nemotron_sw_post", "nemotron_n4_post"}
        or not job_comment
        or not Path(stdout_path).is_absolute()
        or not manifest_path.is_absolute()
    ):
        raise ValueError("A submission identity is invalid")
    body: dict[str, Any] = {
        "schema_version": 2,
        "producer": "qwen4b-a-submit-v2",
        "source_commit": source_commit,
        "builder_job_id": builder_job_id,
        "expected_gpu_job_id": gpu_job_id,
        "dependency": f"afterok:{builder_job_id}",
        "slurm_account": account,
        "slurm_job_name": "q4b-a-repair-canary",
        "slurm_job_comment": job_comment,
        "slurm_output_path": stdout_path,
        "manifest_path": str(manifest_path),
        "scientific_training_authorized": False,
    }
    body["receipt_sha256"] = _sha256_json(body)
    atomic_publish_bytes(path, (_canonical_json(body) + "\n").encode(), job_id="submit")


def finalize_a_authorization(
    authorization_path: Path,
    observation_path: Path,
    *,
    submission_receipt_path: Path,
    job_completion_path: Path,
    sacct_output: str,
) -> None:
    """Issue final authorization only after exact parent-job sacct completion."""
    submission_raw = _stable_file_bytes(submission_receipt_path)
    completion_raw = _stable_file_bytes(job_completion_path)
    submission = _load_self_hashed(submission_receipt_path, "A submission")
    completion = _load_self_hashed(job_completion_path, "A job completion")
    if completion.get("producer") != "qwen4b-a-repair-job-completion-v1":
        raise ValueError("A controller requires genuine GPU job completion")
    expected_job = submission.get("expected_gpu_job_id")
    expected = {
        "slurm_job_id": expected_job,
        "slurm_account": submission.get("slurm_account"),
        "slurm_job_name": submission.get("slurm_job_name"),
        "slurm_job_comment": submission.get("slurm_job_comment"),
        "slurm_output_path": submission.get("slurm_output_path"),
        "submission_receipt_path": str(submission_receipt_path),
        "submission_receipt_sha256": hashlib.sha256(submission_raw).hexdigest(),
    }
    if any(completion.get(key) != value for key, value in expected.items()):
        raise ValueError("A completion and submission identities differ")
    fields = sacct_output.strip().splitlines()
    if len(fields) != 1:
        raise ValueError("A controller requires one parent sacct row")
    values = fields[0].split("|")
    if len(values) != 9:
        raise ValueError("A controller sacct row has an invalid schema")
    job_id, state, exit_code, account, job_name, start, end, comment, stdout = values
    if (job_id, state, exit_code, account, job_name, comment, stdout) != (
        expected_job,
        "COMPLETED",
        "0:0",
        submission.get("slurm_account"),
        submission.get("slurm_job_name"),
        submission.get("slurm_job_comment"),
        submission.get("slurm_output_path"),
    ):
        raise ValueError("A controller sacct identity or terminal state mismatch")
    _validate_enclosing_times(start, str(completion.get("started_at")), str(completion.get("finished_at")), end)
    observation: dict[str, Any] = {
        "schema_version": 2,
        "producer": "sacct-a-repair-controller-v2",
        "observed_at": datetime.now().astimezone().isoformat(),
        "submission_receipt_sha256": hashlib.sha256(submission_raw).hexdigest(),
        "a_job_completion_sha256": hashlib.sha256(completion_raw).hexdigest(),
        "slurm_job_id": job_id,
        "state": state,
        "exit_code": exit_code,
        "account": account,
        "job_name": job_name,
        "start": start,
        "end": end,
        "comment": comment,
        "stdout": stdout,
    }
    observation["receipt_sha256"] = _sha256_json(observation)
    atomic_publish_bytes(
        observation_path,
        (_canonical_json(observation) + "\n").encode(),
        job_id="controller",
    )
    observation_raw = _stable_file_bytes(observation_path)
    authorization = dict(completion)
    authorization.update(
        {
            "schema_version": 2,
            "producer": "qwen4b-a-repair-controller-authorization-v2",
            "authorization": "B-balanced-canary",
            "scheduler_observation_path": str(observation_path),
            "scheduler_observation_sha256": hashlib.sha256(observation_raw).hexdigest(),
            "job_completion_path": str(job_completion_path),
            "job_completion_sha256": hashlib.sha256(completion_raw).hexdigest(),
        }
    )
    authorization.pop("receipt_sha256", None)
    authorization["receipt_sha256"] = _sha256_json(authorization)
    atomic_publish_bytes(
        authorization_path,
        (_canonical_json(authorization) + "\n").encode(),
        job_id="controller",
    )


def _load_self_hashed(path: Path, label: str) -> dict[str, Any]:
    body = json.loads(_stable_file_bytes(path))
    if not isinstance(body, dict):
        raise ValueError(f"{label} receipt is invalid")
    claim = body.pop("receipt_sha256", None)
    if claim != _sha256_json(body):
        raise ValueError(f"{label} receipt identity mismatch")
    body["receipt_sha256"] = claim
    return body


def _validate_enclosing_times(start: str, work_start: str, work_end: str, end: str) -> None:
    try:
        values = [datetime.fromisoformat(value.replace("Z", "+00:00")) for value in (start, work_start, work_end, end)]
    except ValueError as error:
        raise ValueError("A controller timing is invalid") from error
    if any(value.tzinfo is None for value in values) or values != sorted(values):
        raise ValueError("A controller timing does not enclose GPU work")


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


def _tree_sha256(root: Path) -> str:
    """Hash regular files and symlink texts without following runtime-tree links."""
    if not root.is_dir() or root.is_symlink():
        raise ValueError("runtime tree is invalid")
    entries = sorted(root.rglob("*"), key=lambda path: path.relative_to(root).as_posix())
    if not entries:
        raise ValueError("runtime tree is empty")
    digest = hashlib.sha256()
    for artifact in entries:
        relative = artifact.relative_to(root).as_posix().encode()
        if artifact.is_symlink():
            kind, payload = b"L", os.readlink(artifact).encode()
        elif artifact.is_file():
            kind = b"F"
            payload = bytes.fromhex(hashlib.sha256(_stable_file_bytes(artifact)).hexdigest())
        elif artifact.is_dir():
            kind, payload = b"D", b""
        else:
            raise ValueError("runtime tree contains unsupported entries")
        digest.update(relative + b"\0" + kind + b"\0" + payload)
    return digest.hexdigest()


def _canonical_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _sha256_json(value: dict[str, Any]) -> str:
    return hashlib.sha256(_canonical_json(value).encode()).hexdigest()
