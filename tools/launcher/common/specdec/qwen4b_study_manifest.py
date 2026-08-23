# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Typed Qwen3-4B speculative-decoding dataset-study experiments."""

from __future__ import annotations

import json
import os
import re
import tempfile
from dataclasses import asdict, dataclass
from hashlib import sha256
from pathlib import Path

__all__ = [
    "Qwen4BStudyExperiment",
    "Qwen4BStudyInputs",
    "Qwen4BStudySchedule",
    "WandbStudyRun",
    "build_wandb_study_run",
    "load_study_manifest",
    "readable_study_job_name",
    "validate_canary_receipt",
    "validate_milestone_receipt",
    "validate_pmon_logs",
    "validate_readiness_receipt",
    "validate_training_target",
    "write_study_manifest",
    "write_wandb_run_receipt",
]

_SHA = re.compile(r"^[0-9a-f]{40}$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


def validate_readiness_receipt(
    receipt: dict,
    *,
    profile: str,
    account: str,
    partition: str,
    scratch_root: Path,
) -> None:
    """Validate the canonical readiness contract shared by all launchers."""
    expected = {
        "profile": profile,
        "account": account,
        "partition": partition,
        "scratch_root": str(scratch_root),
        "pyxis_available": True,
        "architecture": "aarch64",
        "gpu_count": 4,
    }
    if not isinstance(receipt, dict) or any(
        receipt.get(field) != value for field, value in expected.items()
    ):
        raise ValueError("cluster readiness receipt does not match the canonical contract")


def _path_under(name: str, value: str, root: str) -> str:
    if not value or not Path(value).is_absolute():
        raise ValueError(f"{name} must be an absolute path under {root}")
    normalized = Path(os.path.abspath(value))
    resolved = normalized.resolve(strict=False)
    resolved_root = Path(root).resolve(strict=False)
    if not resolved.is_relative_to(resolved_root) or resolved == resolved_root:
        raise ValueError(f"{name} must be a strict descendant of {root}")
    return str(normalized)


def _file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _directory_sha256(root: Path) -> str:
    """Hash a non-empty regular-file tree including every relative path."""
    if not root.is_dir():
        raise ValueError(f"artifact directory is missing: {root}")
    files = sorted(path for path in root.rglob("*") if path.is_file() and not path.is_symlink())
    if not files:
        raise ValueError(f"artifact directory is empty: {root}")
    digest = sha256()
    for path in files:
        digest.update(str(path.relative_to(root)).encode())
        digest.update(b"\0")
        digest.update(bytes.fromhex(_file_sha256(path)))
    return digest.hexdigest()


def _validate_directory_artifact(
    payload: dict, *, path_field: str, digest_field: str, root: Path
) -> None:
    raw_path = payload.get(path_field)
    if not isinstance(raw_path, str) or not Path(raw_path).is_absolute():
        raise ValueError("receipt artifact path is invalid")
    path = Path(raw_path).resolve(strict=False)
    if not path.is_relative_to(root) or path == root:
        raise ValueError("receipt artifact escapes the experiment output root")
    expected_digest = payload.get(digest_field)
    if not isinstance(expected_digest, str) or _directory_sha256(path) != expected_digest:
        raise ValueError("receipt artifact identity mismatch")


@dataclass(frozen=True)
class Qwen4BStudyInputs:
    """Immutable execution and corpus inputs for one training arm."""

    source_path: str
    source_sha: str
    image_path: str
    image_sha256: str
    runtime_archive_path: str
    runtime_archive_sha256: str
    target_path: str
    target_revision: str
    tokenizer_sha256: str
    corpus_path: str
    corpus_manifest_path: str
    corpus_manifest_sha256: str
    output_root: str

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "source_path", _path_under("source_path", self.source_path, "/home")
        )
        for name in (
            "image_path",
            "runtime_archive_path",
            "target_path",
            "corpus_path",
            "corpus_manifest_path",
            "output_root",
        ):
            object.__setattr__(self, name, _path_under(name, getattr(self, name), "/lustre"))
        for name, pattern in (
            ("source_sha", _SHA),
            ("target_revision", _SHA),
            ("tokenizer_sha256", _SHA256),
            ("image_sha256", _SHA256),
            ("runtime_archive_sha256", _SHA256),
            ("corpus_manifest_sha256", _SHA256),
        ):
            if not pattern.fullmatch(getattr(self, name)):
                raise ValueError(f"{name} must be an exact lowercase digest")


@dataclass(frozen=True)
class Qwen4BStudySchedule:
    """Comparable token exposure and scheduler policy for a study arm."""

    account: str
    partition: str
    assistant_token_budget: int
    corpus_rows: int
    max_steps: int
    canary_steps: int
    assistant_token_milestones: tuple[int, ...]
    assistant_token_milestone_step_ceilings: tuple[int, ...]

    def __post_init__(self) -> None:
        if self.account != "nemotron_sw_post":
            raise ValueError("Qwen3-4B study requires account nemotron_sw_post")
        if not self.partition.strip():
            raise ValueError("partition must be non-empty")
        if (
            min(
                self.assistant_token_budget,
                self.corpus_rows,
                self.max_steps,
                self.canary_steps,
            )
            < 1
        ):
            raise ValueError("study token, row, and step counts must be positive")
        if self.canary_steps >= self.max_steps:
            raise ValueError("canary_steps must be smaller than max_steps")
        if not self.assistant_token_milestones:
            raise ValueError("assistant-token milestones must not be empty")
        if tuple(sorted(set(self.assistant_token_milestones))) != self.assistant_token_milestones:
            raise ValueError("assistant-token milestones must be strictly increasing")
        if self.assistant_token_milestones[-1] != self.assistant_token_budget:
            raise ValueError("final assistant-token milestone must equal the budget")
        ceilings = self.assistant_token_milestone_step_ceilings
        if len(ceilings) != len(self.assistant_token_milestones):
            raise ValueError("assistant-token milestones and step ceilings must align")
        if tuple(sorted(set(ceilings))) != ceilings or ceilings[0] <= self.canary_steps:
            raise ValueError("assistant-token milestone step ceilings must be strictly increasing")
        if ceilings[-1] != self.max_steps:
            raise ValueError("final assistant-token step ceiling must equal max_steps")


@dataclass(frozen=True)
class Qwen4BStudyExperiment:
    """One Qwen3-4B hybrid-mode corpus and drafter-method training run."""

    arm: str
    thinking_mode: str
    method: str
    block_size: int
    inputs: Qwen4BStudyInputs
    schedule: Qwen4BStudySchedule

    nodes: int = 2
    serve_nodes: int = 1
    trainer_nodes: int = 1
    gpus_per_node: int = 4
    serve_tp: int = 1
    serve_replicas_per_node: int = 4
    per_device_train_batch_size: int = 4
    gradient_accumulation_steps: int = 8
    training_seq_len: int = 4096
    capture_ids: tuple[int, ...] = (2, 18, 33, 36)
    num_attention_heads: int = 32
    num_key_value_heads: int = 8
    head_dim: int = 128
    intermediate_size: int = 9728

    def __post_init__(self) -> None:
        if self.arm not in {"B", "C", "D"}:
            raise ValueError(f"unsupported study arm: {self.arm}")
        if self.thinking_mode not in {"on", "off"}:
            raise ValueError(f"unsupported thinking mode: {self.thinking_mode}")
        if (self.method, self.block_size) not in {("dflash", 8), ("dspark", 8)}:
            raise ValueError("study methods are limited to DFlash/DSpark block size 8")
        pinned = {
            "nodes": 2,
            "serve_nodes": 1,
            "trainer_nodes": 1,
            "gpus_per_node": 4,
            "serve_tp": 1,
            "serve_replicas_per_node": 4,
            "per_device_train_batch_size": 4,
            "gradient_accumulation_steps": 8,
            "training_seq_len": 4096,
            "capture_ids": (2, 18, 33, 36),
            "num_attention_heads": 32,
            "num_key_value_heads": 8,
            "head_dim": 128,
            "intermediate_size": 9728,
        }
        if any(getattr(self, field) != value for field, value in pinned.items()):
            raise ValueError("Qwen3-4B study topology must use the pinned full-GPU defaults")
        if self.global_batch_size != 128:
            raise ValueError("Qwen3-4B study topology must produce global batch size 128")

    @property
    def trainer_world_size(self) -> int:
        """Return the number of DDP trainer ranks."""
        return self.trainer_nodes * self.gpus_per_node

    @property
    def global_batch_size(self) -> int:
        """Return the fixed sequence-level global batch size."""
        return (
            self.per_device_train_batch_size
            * self.gradient_accumulation_steps
            * self.trainer_world_size
        )

    @property
    def experiment_id(self) -> str:
        """Return a stable identity over corpus, target, method, and topology."""
        payload = json.dumps(asdict(self), sort_keys=True, separators=(",", ":"))
        return sha256(payload.encode()).hexdigest()[:16]


def readable_study_job_name(experiment: Qwen4BStudyExperiment) -> str:
    """Render a scheduler name containing every human-relevant study dimension."""
    method = {"dflash": "df", "dspark": "ds"}[experiment.method]
    mode = {"on": "thon", "off": "thoff"}[experiment.thinking_mode]
    budget_millions = experiment.schedule.assistant_token_budget // 1_000_000
    return (
        f"q4b-{mode}-{experiment.arm.lower()}-{method}-b{experiment.block_size}"
        f"-n{experiment.nodes}-t{budget_millions}m"
    )


@dataclass(frozen=True)
class WandbStudyRun:
    """Durable, receipt-bound W&B identity for one scientific exposure boundary."""

    schema_version: int
    project: str
    group: str
    job_type: str
    run_name: str
    run_id: str
    mode: str
    resume: str
    wandb_dir: str
    cache_dir: str
    config_dir: str
    artifact_dir: str
    experiment_id: str
    arm: str
    thinking_mode: str
    method: str
    block_size: int
    assistant_token_target: int | None
    requested_steps: int
    corpus_manifest_sha256: str
    tokenizer_sha256: str
    target_revision: str
    source_sha: str

    @property
    def receipt(self) -> dict[str, object]:
        """Return the canonical non-secret identity recorded beside checkpoints."""
        return asdict(self)


def build_wandb_study_run(
    experiment: Qwen4BStudyExperiment,
    requested_steps: int,
    assistant_token_target: int | None,
) -> WandbStudyRun:
    """Resolve a stable W&B run without inheriting a parent training run."""
    validate_training_target(experiment, requested_steps, assistant_token_target)
    if assistant_token_target is None:
        boundary = f"canary-s{requested_steps}"
        name_boundary = f"canary-s{requested_steps}"
    else:
        boundary = f"assistant-tokens-{assistant_token_target}"
        name_boundary = (
            f"t{assistant_token_target // 1_000_000}m"
            if assistant_token_target % 1_000_000 == 0
            else f"t{assistant_token_target}"
        )
    identity = {
        "schema_version": 1,
        "experiment_id": experiment.experiment_id,
        "boundary": boundary,
    }
    identity_sha256 = sha256(
        json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    run_id = f"q4b-{experiment.experiment_id}-{identity_sha256[:12]}"
    durable_root = Path(experiment.inputs.output_root) / "wandb" / run_id
    return WandbStudyRun(
        schema_version=1,
        project="sna-qwen3-4b-dataset-study",
        group=(
            f"q4b-{experiment.arm.lower()}-thinking-{experiment.thinking_mode}-"
            f"{experiment.method}-b{experiment.block_size}"
        ),
        job_type="dataset-study-train",
        run_name=f"{readable_study_job_name(experiment)}-{name_boundary}",
        run_id=run_id,
        mode="online",
        resume="allow",
        wandb_dir=str(durable_root / "runs"),
        cache_dir=str(durable_root / "cache"),
        config_dir=str(durable_root / "config"),
        artifact_dir=str(durable_root / "artifacts"),
        experiment_id=experiment.experiment_id,
        arm=experiment.arm,
        thinking_mode=experiment.thinking_mode,
        method=experiment.method,
        block_size=experiment.block_size,
        assistant_token_target=assistant_token_target,
        requested_steps=requested_steps,
        corpus_manifest_sha256=experiment.inputs.corpus_manifest_sha256,
        tokenizer_sha256=experiment.inputs.tokenizer_sha256,
        target_revision=experiment.inputs.target_revision,
        source_sha=experiment.inputs.source_sha,
    )


def write_wandb_run_receipt(path: Path, run: WandbStudyRun) -> None:
    """Publish one immutable W&B identity, allowing byte-identical requeues."""
    content = (json.dumps(run.receipt, sort_keys=True, separators=(",", ":")) + "\n").encode()
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() or path.is_symlink():
        if not path.is_file() or path.is_symlink() or path.read_bytes() != content:
            raise ValueError("W&B run receipt identity mismatch")
        return
    with tempfile.NamedTemporaryFile(
        mode="wb", dir=path.parent, prefix=f".{path.name}.", delete=False
    ) as output:
        output.write(content)
        output.flush()
        os.fsync(output.fileno())
        temporary = Path(output.name)
    try:
        try:
            os.link(temporary, path, follow_symlinks=False)
        except FileExistsError:
            if not path.is_file() or path.is_symlink() or path.read_bytes() != content:
                raise ValueError("W&B run receipt identity mismatch") from None
        directory_fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        temporary.unlink(missing_ok=True)


def validate_canary_receipt(
    path: Path | None,
    experiment: Qwen4BStudyExperiment,
    requested_steps: int,
) -> None:
    """Require a successful, identity-bound canary before a production run."""
    if requested_steps == experiment.schedule.canary_steps:
        return
    if requested_steps not in experiment.schedule.assistant_token_milestone_step_ceilings:
        raise ValueError("requested steps must be the canary or a milestone step ceiling")
    if path is None:
        raise ValueError("canary receipt is required for production steps")
    if not path.is_file():
        raise ValueError("canary receipt is required for production steps")

    payload = json.loads(path.read_text(encoding="utf-8"))
    expected = {
        "schema_version": 1,
        "experiment_id": experiment.experiment_id,
        "completed_steps": experiment.schedule.canary_steps,
        "checkpoint_verified": True,
        "export_verified": True,
        "all_gpus_active": True,
        "status": "passed",
    }
    if not isinstance(payload, dict) or any(
        payload.get(field) != value for field, value in expected.items()
    ):
        raise ValueError("canary receipt failed validation")
    canary_root = (Path(experiment.inputs.output_root) / "canary").resolve(strict=False)
    _validate_directory_artifact(
        payload,
        path_field="checkpoint_path",
        digest_field="checkpoint_sha256",
        root=canary_root,
    )
    _validate_directory_artifact(
        payload,
        path_field="export_path",
        digest_field="export_sha256",
        root=canary_root,
    )
    ownership_path = Path(str(payload.get("gpu_ownership_path", ""))).resolve(strict=False)
    if (
        not ownership_path.is_file()
        or not ownership_path.is_relative_to(canary_root)
        or payload.get("gpu_ownership_sha256") != _file_sha256(ownership_path)
    ):
        raise ValueError("canary receipt GPU evidence mismatch")
    ownership = json.loads(ownership_path.read_text(encoding="utf-8"))
    if ownership.get("nodes") != 2 or ownership.get("nonzero_sm_activity") is not True:
        raise ValueError("canary receipt GPU evidence mismatch")
    pmon_hashes = ownership.get("pmon_sha256")
    if not isinstance(pmon_hashes, dict) or len(pmon_hashes) != 2:
        raise ValueError("canary receipt GPU evidence mismatch")
    logs_root = canary_root / "logs"
    for name, digest in pmon_hashes.items():
        if (
            not isinstance(name, str)
            or Path(name).name != name
            or not isinstance(digest, str)
            or not (logs_root / name).is_file()
            or _file_sha256(logs_root / name) != digest
        ):
            raise ValueError("canary receipt GPU evidence mismatch")


def validate_pmon_logs(log_root: Path, job_id: str) -> dict[str, list[int]]:
    """Require every GPU on both nodes to show nonzero SM activity for one job."""
    if not job_id.isdigit():
        raise ValueError("Slurm job ID must be numeric")
    files = sorted(log_root.glob(f"nvidia-smi-pmon-node-*-{job_id}.log"))
    if len(files) != 2:
        raise ValueError("missing per-node GPU utilization evidence")
    expected_gpus = {0, 1, 2, 3}
    evidence: dict[str, list[int]] = {}
    for path in files:
        columns: dict[str, int] | None = None
        active: set[int] = set()
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
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
        if active != expected_gpus:
            raise ValueError(f"nonzero SM activity missing in {path.name}: {sorted(active)}")
        evidence[path.name] = sorted(active)
    return evidence


def validate_training_target(
    experiment: Qwen4BStudyExperiment,
    requested_steps: int,
    assistant_token_target: int | None,
) -> None:
    """Bind canary or exact-token production work to its immutable step ceiling."""
    if requested_steps == experiment.schedule.canary_steps:
        if assistant_token_target is not None:
            raise ValueError("canary must not declare an assistant-token target")
        return
    if assistant_token_target is None:
        raise ValueError("production requires an assistant-token target")
    mapping = dict(
        zip(
            experiment.schedule.assistant_token_milestones,
            experiment.schedule.assistant_token_milestone_step_ceilings,
            strict=True,
        )
    )
    if mapping.get(assistant_token_target) != requested_steps:
        raise ValueError("assistant-token milestone/step ceiling mismatch")


def validate_milestone_receipt(
    path: Path,
    experiment: Qwen4BStudyExperiment,
    expected_tokens: int,
) -> None:
    """Validate a durable exact-token checkpoint/export receipt."""
    if expected_tokens not in experiment.schedule.assistant_token_milestones or not path.is_file():
        raise ValueError("assistant-token milestone receipt failed validation")
    payload = json.loads(path.read_text(encoding="utf-8"))
    expected = {
        "schema_version": 1,
        "experiment_id": experiment.experiment_id,
        "committed_assistant_tokens": expected_tokens,
        "checkpoint_verified": True,
        "export_verified": True,
        "status": "passed",
    }
    if not isinstance(payload, dict):
        raise ValueError("assistant-token milestone receipt failed validation")
    if (
        any(payload.get(field) != value for field, value in expected.items())
        or type(payload.get("global_step")) is not int
        or payload["global_step"] < 1
    ):
        raise ValueError("assistant-token milestone receipt failed validation")
    output_root = Path(experiment.inputs.output_root).resolve(strict=False)
    _validate_directory_artifact(
        payload,
        path_field="checkpoint_path",
        digest_field="checkpoint_sha256",
        root=output_root,
    )
    _validate_directory_artifact(
        payload,
        path_field="export_path",
        digest_field="export_sha256",
        root=output_root,
    )


def write_study_manifest(path: Path, experiments: list[Qwen4BStudyExperiment]) -> None:
    """Atomically write a canonical list of study experiments."""
    if not experiments:
        raise ValueError("at least one study experiment is required")
    identities = [experiment.experiment_id for experiment in experiments]
    if len(set(identities)) != len(identities):
        raise ValueError("study experiment IDs must be unique")
    payload = [
        asdict(experiment) | {"experiment_id": experiment.experiment_id}
        for experiment in experiments
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", dir=path.parent, prefix=f".{path.name}.", delete=False, encoding="utf-8"
    ) as output:
        json.dump(payload, output, indent=2, sort_keys=True)
        output.write("\n")
        temporary = Path(output.name)
    os.replace(temporary, path)


def load_study_manifest(path: Path) -> list[Qwen4BStudyExperiment]:
    """Load a canonical study manifest and verify every derived identity."""
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, list) or not raw:
        raise ValueError("study manifest must be a non-empty list")
    experiments: list[Qwen4BStudyExperiment] = []
    for item in raw:
        entry = dict(item)
        expected_identity = entry.pop("experiment_id", None)
        schedule = dict(entry.pop("schedule"))
        for field in (
            "assistant_token_milestones",
            "assistant_token_milestone_step_ceilings",
        ):
            schedule[field] = tuple(schedule[field])
        for field in ("capture_ids",):
            if field in entry:
                entry[field] = tuple(entry[field])
        inputs = Qwen4BStudyInputs(**entry.pop("inputs"))
        experiment = Qwen4BStudyExperiment(
            **entry,
            inputs=inputs,
            schedule=Qwen4BStudySchedule(**schedule),
        )
        if expected_identity != experiment.experiment_id:
            raise ValueError("study experiment identity mismatch")
        experiments.append(experiment)
    return experiments
