# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Typed authenticated checkpoint adoption for the Qwen3-4B 700K continuation."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import shutil
import stat
import subprocess
import sys
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any, Literal

Method = Literal["DFlash", "DSpark"]
DATASET_IDENTITY = "ptv2-ptv3-complement-700k-v1"
APPROVED_QUOTAS = {
    "ptv2_stem": 300_000,
    "ptv2_multilingual_ja": 50_000,
    "ptv2_multilingual_es": 50_000,
    "ptv2_multilingual_fr": 50_000,
    "ptv2_multilingual_it": 50_000,
    "ptv3_swe_v3": 100_000,
    "ptv3_interactive_agentic_swe": 19_000,
    "ptv3_general_tool_trajectories": 81_000,
}
TRUSTED_DATASET_BUILDER_SHA256 = "eb92dd9f5fe6728a3af284164a4553317b13800c5f4f143b46aea24d6299439f"
TRUSTED_TRAJECTORY_SCHEMA_SHA256 = (
    "0a61ebae867b347000158c5226e2d64397aa64e8c5bf14512f5fe62485c16e4a"
)
TRUSTED_SPECDEC_IDENTITY_SHA256 = "d2c2958ede5304da3d2d765dda697f265dfaebd3f4be1a834349e6428b37f0f6"
TRUSTED_SPECDEC_CORPUS_CONTRACTS_SHA256 = (
    "3ccdc6a6ee8b392067042ceabcdbda6d1a4c6993e1e92587bb91fa26dd1d6f44"
)
TRUSTED_HISTORICAL_AUTHENTICATOR_SHA256 = (
    "125cfe40db803517ff2c218e1d67f6db93b4e38175f558fb17ca47c6a0c45c88"
)
TRUSTED_TOKENIZER_TRUST_FILE_SHA256 = ""
TRUSTED_PARENT_CHECKPOINT_RECEIPT_SHA256S: frozenset[str] = frozenset()
TRUSTED_CANARY_RECEIPT_SHA256S: frozenset[str] = frozenset()
TRUSTED_TRAINING_ENTRYPOINT_SHA256 = ""
TRUSTED_RUNTIME_IMAGE_SHA256 = ""
RUNTIME_ATTESTATION_IMPLEMENTED = False


@dataclass(frozen=True)
class AuthenticatedParentCheckpoint:
    """An exact existing 1.3M checkpoint authenticated for weights-only adoption."""

    method: Method
    checkpoint_path: Path
    checkpoint_tree_sha256: str
    historical_receipt_file_sha256: str
    historical_ordered_prompt_uuids_sha256: str
    runtime_sha256: str
    source_commit: str
    receipt_path: Path
    receipt_file_sha256: str
    receipt_sha256: str


@dataclass(frozen=True)
class ContinuationContract:
    """Fresh-training identity rooted in one authenticated existing parent."""

    method: Method
    run_identity: str
    dataset_identity: str
    dataset_manifest_path: Path
    dataset_manifest_file_sha256: str
    dataset_completion_receipt_path: Path
    dataset_completion_receipt_file_sha256: str
    dataset_builder_tool_path: Path
    dataset_builder_tool_sha256: str
    trajectory_schema_path: Path
    trajectory_schema_sha256: str
    specdec_identity_path: Path
    specdec_identity_sha256: str
    specdec_corpus_contracts_path: Path
    specdec_corpus_contracts_sha256: str
    historical_authenticator_path: Path
    historical_authenticator_sha256: str
    config_path: Path
    config_file_sha256: str
    source_inventory_path: Path
    source_inventory_file_sha256: str
    historical_receipt_path: Path
    historical_receipt_file_sha256: str
    historical_ordered_prompt_uuids_sha256: str
    held_out_receipts: tuple[tuple[str, Path, str], ...]
    tokenizer_trust_path: Path
    tokenizer_trust_file_sha256: str
    verification_scratch_root: Path
    canary_execution_stage_root: Path
    canary_checkpoint_output_root: Path
    canary_evidence_path: Path
    canary_receipt_path: Path
    full_execution_stage_root: Path
    full_checkpoint_output_root: Path
    full_evidence_path: Path
    full_receipt_path: Path
    parent_checkpoint_path: Path
    parent_checkpoint_tree_sha256: str
    parent_receipt_path: Path
    parent_receipt_file_sha256: str
    training_entrypoint_path: Path
    training_entrypoint_sha256: str
    runtime_image_path: Path
    runtime_image_sha256: str
    full_steps: int
    global_batch_size: int
    full_example_exposure: Literal[700000] = 700_000
    resume_mode: Literal["weights-only"] = "weights-only"
    optimizer_state: Literal["fresh"] = "fresh"
    scheduler_state: Literal["fresh"] = "fresh"
    canary_steps: Literal[20] = 20
    test_only_required: Literal[True] = True


def _canonical(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def _is_hash(value: object, length: int) -> bool:
    return (
        isinstance(value, str)
        and len(value) == length
        and all(character in "0123456789abcdef" for character in value)
    )


def _validate_fresh_disjoint_roots(
    *,
    protected: tuple[Path, ...],
    writable: tuple[Path, ...],
    writable_files: tuple[Path, ...] = (),
    require_fresh: bool = True,
    allow_empty_directories: bool = False,
) -> None:
    if any(path.is_symlink() or (path.exists() and not path.is_dir()) for path in writable):
        raise ValueError("writable execution roots must be no-follow directories")
    if any(path.is_symlink() or path.parent.is_symlink() for path in writable_files):
        raise ValueError("writable evidence paths cannot be symlinks")
    protected_resolved = tuple(path.resolve(strict=False) for path in protected)
    writable_resolved = tuple(path.resolve(strict=False) for path in writable)
    writable_file_resolved = tuple(path.resolve(strict=False) for path in writable_files)
    if require_fresh:
        for path in writable_resolved + writable_file_resolved:
            if not path.exists():
                continue
            if allow_empty_directories and path.is_dir() and not any(path.iterdir()):
                continue
            raise ValueError("writable execution roots must be fresh")
    effective_writable = tuple(
        dict.fromkeys(writable_resolved + tuple(path.parent for path in writable_file_resolved))
    )
    for index, root in enumerate(effective_writable):
        for other in protected_resolved + effective_writable[index + 1 :]:
            if root == other or root in other.parents or other in root.parents:
                raise ValueError("protected and writable execution roots must be disjoint")


def _validate_stage_isolation(*, canary: tuple[Path, ...], full: tuple[Path, ...]) -> None:
    canary_roots = tuple(dict.fromkeys(path.resolve(strict=False) for path in canary))
    full_roots = tuple(dict.fromkeys(path.resolve(strict=False) for path in full))
    for root in canary_roots:
        for other in full_roots:
            if root == other or root in other.parents or other in root.parents:
                raise ValueError("canary and full writable mount roots must be disjoint")


def _stable_file(path: Path) -> bytes:
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        raise ValueError("authenticated receipt must be a no-follow regular file") from error
    with os.fdopen(descriptor, "rb") as stream:
        before = os.fstat(stream.fileno())
        if not stat.S_ISREG(before.st_mode):
            raise ValueError("authenticated receipt must be a regular file")
        blocks: list[bytes] = []
        while block := stream.read(1024 * 1024):
            blocks.append(block)
        after = os.fstat(stream.fileno())
    if (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns) != (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
    ):
        raise ValueError("checkpoint receipt changed while reading")
    return b"".join(blocks)


def _regular_file_evidence(path: Path) -> tuple[int, str]:
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        raise ValueError("authenticated artifact is not a readable no-follow file") from error
    with os.fdopen(descriptor, "rb") as stream:
        before = os.fstat(stream.fileno())
        if not stat.S_ISREG(before.st_mode):
            raise ValueError("authenticated artifact must be a regular file")
        digest = sha256()
        size = 0
        while block := stream.read(8 * 1024 * 1024):
            digest.update(block)
            size += len(block)
        after = os.fstat(stream.fileno())
    if (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns) != (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
    ) or size != before.st_size:
        raise ValueError("authenticated artifact changed while hashing")
    return size, digest.hexdigest()


def _checkpoint_tree_sha256(root: Path) -> str:
    if root.is_symlink() or not root.is_dir():
        raise ValueError("parent checkpoint must be an existing no-follow directory")
    entries: list[list[str]] = []
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise ValueError("parent checkpoint tree cannot contain symlinks")
        if path.is_file():
            entries.append([path.relative_to(root).as_posix(), _regular_file_evidence(path)[1]])
    if not entries or not any(path.endswith((".safetensors", ".bin")) for path, _ in entries):
        raise ValueError("parent checkpoint has no model weight artifact")
    return sha256(_canonical(entries)).hexdigest()


def _stage_authenticated_file(source: Path, destination: Path, *, expected_sha256: str) -> None:
    if destination.exists() or destination.is_symlink():
        raise ValueError("execution stage destination must be fresh")
    destination.parent.mkdir(parents=True, exist_ok=True)
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        source_descriptor = os.open(source, flags)
        with (
            os.fdopen(source_descriptor, "rb") as source_stream,
            destination.open("xb") as destination_stream,
        ):
            before = os.fstat(source_stream.fileno())
            if not stat.S_ISREG(before.st_mode):
                raise ValueError("authenticated source must be a regular file")
            source_digest = sha256()
            while block := source_stream.read(8 * 1024 * 1024):
                source_digest.update(block)
                destination_stream.write(block)
            destination_stream.flush()
            os.fsync(destination_stream.fileno())
            after = os.fstat(source_stream.fileno())
        if (
            (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
            != (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
            or source_digest.hexdigest() != expected_sha256
            or _regular_file_evidence(destination)[1] != expected_sha256
        ):
            raise ValueError("authenticated source changed while staging")
    except BaseException:
        destination.unlink(missing_ok=True)
        raise


def _stage_authenticated_tree(
    source: Path, destination: Path, *, expected_tree_sha256: str
) -> None:
    """Copy one authenticated tree into a fresh execution-only directory."""
    if destination.is_symlink() or (destination.exists() and any(destination.iterdir())):
        raise ValueError("execution stage destination must be fresh")
    if _checkpoint_tree_sha256(source) != expected_tree_sha256:
        raise ValueError("authenticated source tree SHA-256 mismatch")
    destination.mkdir(parents=True, exist_ok=True)
    try:
        for path in sorted(source.rglob("*")):
            relative = path.relative_to(source)
            if path.is_symlink():
                raise ValueError("authenticated source tree cannot contain symlinks")
            if path.is_dir():
                (destination / relative).mkdir(parents=True, exist_ok=True)
            elif path.is_file():
                digest = _regular_file_evidence(path)[1]
                _stage_authenticated_file(path, destination / relative, expected_sha256=digest)
            else:
                raise ValueError("authenticated source tree must contain only regular files")
        if (
            _checkpoint_tree_sha256(source) != expected_tree_sha256
            or _checkpoint_tree_sha256(destination) != expected_tree_sha256
        ):
            raise ValueError("authenticated source tree changed while staging")
    except BaseException:
        shutil.rmtree(destination, ignore_errors=True)
        raise


def _stage_authenticated_dataset(contract: ContinuationContract, destination: Path) -> Path:
    if destination.exists() or destination.is_symlink():
        raise ValueError("dataset execution stage must be fresh")
    manifest_raw = _stable_file(contract.dataset_manifest_path)
    if sha256(manifest_raw).hexdigest() != contract.dataset_manifest_file_sha256:
        raise ValueError("dataset manifest changed before execution staging")
    manifest = json.loads(manifest_raw)
    data_name = manifest.get("data", {}).get("path")
    data_sha256 = manifest.get("data", {}).get("sha256")
    if not isinstance(data_name, str) or not _is_hash(data_sha256, 64):
        raise ValueError("dataset manifest file identity is invalid")
    destination.mkdir(parents=True, exist_ok=False)
    try:
        staged_manifest = destination / contract.dataset_manifest_path.name
        _stage_authenticated_file(
            contract.dataset_manifest_path,
            staged_manifest,
            expected_sha256=contract.dataset_manifest_file_sha256,
        )
        _stage_authenticated_file(
            contract.dataset_manifest_path.parent / data_name,
            destination / data_name,
            expected_sha256=data_sha256,
        )
        return staged_manifest
    except BaseException:
        shutil.rmtree(destination, ignore_errors=True)
        raise


def _authenticate_dataset_bundle(
    manifest_path: Path,
    *,
    manifest_file_sha256: str,
    completion_receipt_path: Path,
    completion_receipt_file_sha256: str,
    historical_receipt_file_sha256: str,
    historical_ordered_prompt_uuids_sha256: str,
) -> None:
    """Authenticate a completed exact 700K bundle and its historical lineage."""
    if any(
        not _is_hash(value, 64)
        for value in (
            manifest_file_sha256,
            completion_receipt_file_sha256,
            historical_receipt_file_sha256,
            historical_ordered_prompt_uuids_sha256,
        )
    ):
        raise ValueError("dataset bundle caller identity is invalid")
    manifest_raw = _stable_file(manifest_path)
    receipt_raw = _stable_file(completion_receipt_path)
    if (
        sha256(manifest_raw).hexdigest() != manifest_file_sha256
        or sha256(receipt_raw).hexdigest() != completion_receipt_file_sha256
    ):
        raise ValueError("dataset bundle caller SHA-256 mismatch")
    try:
        manifest: Any = json.loads(manifest_raw)
        receipt: Any = json.loads(receipt_raw)
    except json.JSONDecodeError as error:
        raise ValueError("dataset bundle JSON is invalid") from error
    manifest_keys = {
        "schema_version",
        "scientific_identity",
        "row_count",
        "quotas",
        "category_counts",
        "data",
        "ordered_prompt_uuids_sha256",
        "duplicate_uuid_multiplicity",
        "source_occurrences_sha256",
        "selected_token_evidence_sha256",
        "capacity_receipt_file_sha256",
        "capacity",
        "exclusions",
        "historical",
        "held_out",
        "trust",
        "manifest_sha256",
    }
    if (
        not isinstance(manifest, dict)
        or manifest_raw != _canonical(manifest) + b"\n"
        or set(manifest) != manifest_keys
        or manifest["schema_version"] != "ptv2-ptv3-complement-bundle-v1"
        or manifest["scientific_identity"] != DATASET_IDENTITY
        or manifest["row_count"] != 700_000
        or manifest["quotas"] != APPROVED_QUOTAS
        or manifest["category_counts"] != APPROVED_QUOTAS
        or manifest["duplicate_uuid_multiplicity"] != {}
    ):
        raise ValueError("dataset manifest identity is invalid")
    manifest_body = {key: value for key, value in manifest.items() if key != "manifest_sha256"}
    if manifest["manifest_sha256"] != sha256(_canonical(manifest_body)).hexdigest():
        raise ValueError("dataset manifest self identity does not reconcile")
    historical = manifest["historical"]
    if (
        not isinstance(historical, dict)
        or historical.get("occurrence_count") != 1_300_000
        or historical.get("file_sha256") != historical_receipt_file_sha256
        or historical.get("ordered_prompt_uuids_sha256") != historical_ordered_prompt_uuids_sha256
    ):
        raise ValueError("dataset historical lineage does not match parent checkpoint")
    data = manifest["data"]
    if (
        not isinstance(data, dict)
        or set(data) != {"path", "bytes", "sha256"}
        or data["path"] != "DATA.jsonl"
        or type(data["bytes"]) is not int
        or data["bytes"] < 1
        or not _is_hash(data["sha256"], 64)
    ):
        raise ValueError("dataset data descriptor is invalid")
    data_path = manifest_path.parent / "DATA.jsonl"
    data_size, data_sha256 = _regular_file_evidence(data_path)
    if data_size != data["bytes"] or data_sha256 != data["sha256"]:
        raise ValueError("dataset data identity does not reconcile")
    counts: dict[str, int] = {}
    category_order: list[str] = []
    row_count = 0
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(data_path, flags)
    with os.fdopen(descriptor, "rb") as stream:
        for raw_line in stream:
            try:
                record = json.loads(raw_line)
            except json.JSONDecodeError as error:
                raise ValueError("dataset 700K semantic precheck failed") from error
            if not isinstance(record, dict) or not isinstance(record.get("category"), str):
                raise ValueError("dataset 700K semantic precheck failed")
            category = record["category"]
            counts[category] = counts.get(category, 0) + 1
            if not category_order or category_order[-1] != category:
                category_order.append(category)
            row_count += 1
    if (
        row_count != 700_000
        or counts != APPROVED_QUOTAS
        or tuple(category_order) != tuple(APPROVED_QUOTAS)
    ):
        raise ValueError("dataset 700K semantic precheck failed")
    receipt_keys = {
        "schema_version",
        "scientific_identity",
        "output_root",
        "row_count",
        "manifest_file_sha256",
        "manifest_sha256",
        "runtime_sha256",
        "source_commit",
        "receipt_sha256",
    }
    if (
        not isinstance(receipt, dict)
        or receipt_raw != _canonical(receipt) + b"\n"
        or set(receipt) != receipt_keys
        or receipt["schema_version"] != "ptv2-ptv3-complement-completion-v1"
        or receipt["scientific_identity"] != DATASET_IDENTITY
        or receipt["output_root"] != str(manifest_path.parent)
        or receipt["row_count"] != 700_000
        or receipt["manifest_file_sha256"] != manifest_file_sha256
        or receipt["manifest_sha256"] != manifest["manifest_sha256"]
    ):
        raise ValueError("dataset completion receipt identity is invalid")
    receipt_body = {key: value for key, value in receipt.items() if key != "receipt_sha256"}
    if receipt["receipt_sha256"] != sha256(_canonical(receipt_body)).hexdigest():
        raise ValueError("dataset completion receipt self identity does not reconcile")


def _semantic_protected_paths(
    *,
    source_inventory_path: Path,
    source_inventory_file_sha256: str,
    tokenizer_trust_path: Path,
    tokenizer_trust_file_sha256: str,
    held_out_receipts: tuple[tuple[str, Path, str], ...],
    dataset_builder_tool_path: Path,
    trajectory_schema_path: Path,
    specdec_identity_path: Path,
    specdec_corpus_contracts_path: Path,
    historical_authenticator_path: Path,
    contract_path: Path | None = None,
) -> tuple[Path, ...]:
    """Expand every physical semantic input before approving writable mounts."""
    inventory_raw = _stable_file(source_inventory_path)
    tokenizer_raw = _stable_file(tokenizer_trust_path)
    if (
        sha256(inventory_raw).hexdigest() != source_inventory_file_sha256
        or sha256(tokenizer_raw).hexdigest() != tokenizer_trust_file_sha256
    ):
        raise ValueError("semantic protected-root receipt SHA-256 mismatch")
    try:
        inventory: Any = json.loads(inventory_raw)
        tokenizer: Any = json.loads(tokenizer_raw)
        source_paths = tuple(
            Path(record["path"]) for source in inventory["sources"] for record in source["files"]
        )
        snapshot_path = Path(tokenizer["snapshot_path"])
    except (KeyError, TypeError, json.JSONDecodeError) as error:
        raise ValueError("semantic protected-root receipt schema is invalid") from error
    explicit = (
        source_inventory_path,
        tokenizer_trust_path,
        snapshot_path,
        dataset_builder_tool_path,
        trajectory_schema_path,
        specdec_identity_path,
        specdec_corpus_contracts_path,
        historical_authenticator_path,
        *(path for _, path, _ in held_out_receipts),
        *source_paths,
    )
    return explicit if contract_path is None else (*explicit, contract_path)


def _run_builder_semantic_verifier(
    *,
    builder_tool_path: Path,
    trajectory_schema_path: Path,
    specdec_identity_path: Path,
    specdec_corpus_contracts_path: Path,
    historical_authenticator_path: Path,
    config_path: Path,
    source_inventory_path: Path,
    source_inventory_file_sha256: str,
    historical_receipt_path: Path,
    historical_receipt_file_sha256: str,
    held_out_receipts: tuple[tuple[str, Path, str], ...],
    tokenizer_trust_path: Path,
    tokenizer_trust_file_sha256: str,
    verification_scratch_root: Path,
    manifest_path: Path,
    manifest_file_sha256: str,
) -> None:
    if not sys.flags.isolated:
        raise ValueError("dataset semantic replay requires isolated Python import resolution")
    if (
        not TRUSTED_DATASET_BUILDER_SHA256
        or _regular_file_evidence(builder_tool_path)[1] != TRUSTED_DATASET_BUILDER_SHA256
        or not TRUSTED_TRAJECTORY_SCHEMA_SHA256
        or _regular_file_evidence(trajectory_schema_path)[1] != TRUSTED_TRAJECTORY_SCHEMA_SHA256
        or _regular_file_evidence(specdec_identity_path)[1] != TRUSTED_SPECDEC_IDENTITY_SHA256
        or _regular_file_evidence(specdec_corpus_contracts_path)[1]
        != TRUSTED_SPECDEC_CORPUS_CONTRACTS_SHA256
        or not TRUSTED_HISTORICAL_AUTHENTICATOR_SHA256
        or _regular_file_evidence(historical_authenticator_path)[1]
        != TRUSTED_HISTORICAL_AUTHENTICATOR_SHA256
        or not TRUSTED_TOKENIZER_TRUST_FILE_SHA256
        or tokenizer_trust_file_sha256 != TRUSTED_TOKENIZER_TRUST_FILE_SHA256
    ):
        raise ValueError("dataset semantic verifier dependency trust is unresolved")
    trajectory_spec = importlib.util.spec_from_file_location(
        "trajectory_schema", trajectory_schema_path
    )
    identity_spec = importlib.util.spec_from_file_location(
        "specdec_identity", specdec_identity_path
    )
    corpus_spec = importlib.util.spec_from_file_location(
        "specdec_corpus_contracts", specdec_corpus_contracts_path
    )
    spec = importlib.util.spec_from_file_location("q4_ptv23_trusted_builder", builder_tool_path)
    if (
        trajectory_spec is None
        or trajectory_spec.loader is None
        or identity_spec is None
        or identity_spec.loader is None
        or corpus_spec is None
        or corpus_spec.loader is None
        or spec is None
        or spec.loader is None
    ):
        raise ValueError("dataset semantic verifier cannot be loaded")
    trajectory_module = importlib.util.module_from_spec(trajectory_spec)
    identity_module = importlib.util.module_from_spec(identity_spec)
    corpus_module = importlib.util.module_from_spec(corpus_spec)
    module = importlib.util.module_from_spec(spec)
    previous_modules = {
        name: sys.modules.get(name)
        for name in ("specdec_corpus_contracts", "specdec_identity", "trajectory_schema", spec.name)
    }
    try:
        sys.modules["specdec_corpus_contracts"] = corpus_module
        corpus_spec.loader.exec_module(corpus_module)
        sys.modules["specdec_identity"] = identity_module
        identity_spec.loader.exec_module(identity_module)
        sys.modules["trajectory_schema"] = trajectory_module
        trajectory_spec.loader.exec_module(trajectory_module)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        trust = module.load_tokenizer_trust(
            tokenizer_trust_path, expected_sha256=tokenizer_trust_file_sha256
        )
        tokenizer = module._load_qwen_tokenizer(trust, verification_scratch_root)
        manifest = json.loads(_stable_file(manifest_path))
        if manifest.get("trust", {}).get("tokenizer_trust_file_sha256") != trust.file_sha256:
            raise ValueError("manifest tokenizer trust mismatch")
        inventory = module.load_source_inventory(
            source_inventory_path, expected_sha256=source_inventory_file_sha256
        )
        historical = module.load_historical_exclusion(
            historical_receipt_path, expected_sha256=historical_receipt_file_sha256
        )
        held_out = module.load_held_out_union(held_out_receipts)
        module.verify_selection_bundle(
            manifest_path.parent,
            expected_manifest_file_sha256=manifest_file_sha256,
            tokenizer=tokenizer,
            config_path=config_path,
            inventory=inventory,
            historical=historical,
            held_out=held_out,
            tokenizer_trust_file_sha256=trust.file_sha256,
            scratch_root=verification_scratch_root,
            expected_runtime_sha256=manifest["trust"]["runtime_sha256"],
            expected_source_commit=manifest["trust"]["source_commit"],
        )
    finally:
        for name, previous in previous_modules.items():
            if previous is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = previous


def load_authenticated_parent_checkpoint(
    path: Path,
    *,
    expected_sha256: str,
    expected_occurrences: int = 1_300_000,
) -> AuthenticatedParentCheckpoint:
    """Authenticate one exact existing DFlash/DSpark historical checkpoint receipt."""
    if not _is_hash(expected_sha256, 64):
        raise ValueError("checkpoint receipt caller SHA-256 is invalid")
    raw = _stable_file(path)
    if sha256(raw).hexdigest() != expected_sha256:
        raise ValueError("checkpoint receipt caller SHA-256 mismatch")
    try:
        payload: Any = json.loads(raw)
    except json.JSONDecodeError as error:
        raise ValueError("checkpoint receipt JSON is invalid") from error
    expected_keys = {
        "schema_version",
        "method",
        "checkpoint_path",
        "checkpoint_tree_sha256",
        "historical_receipt_file_sha256",
        "historical_ordered_prompt_uuids_sha256",
        "completed_occurrences",
        "runtime_sha256",
        "source_commit",
        "receipt_sha256",
    }
    if (
        not isinstance(payload, dict)
        or raw != _canonical(payload) + b"\n"
        or set(payload) != expected_keys
        or payload["schema_version"] != "specdec-authenticated-1.3m-checkpoint-v1"
        or payload["method"] not in {"DFlash", "DSpark"}
        or payload["completed_occurrences"] != expected_occurrences
    ):
        raise ValueError("checkpoint receipt identity is invalid")
    digest_fields = (
        payload["checkpoint_tree_sha256"],
        payload["historical_receipt_file_sha256"],
        payload["historical_ordered_prompt_uuids_sha256"],
        payload["runtime_sha256"],
    )
    body = {key: value for key, value in payload.items() if key != "receipt_sha256"}
    checkpoint_path = Path(payload["checkpoint_path"])
    if (
        any(not _is_hash(value, 64) for value in digest_fields)
        or not _is_hash(payload["source_commit"], 40)
        or payload["receipt_sha256"] != sha256(_canonical(body)).hexdigest()
        or _checkpoint_tree_sha256(checkpoint_path) != payload["checkpoint_tree_sha256"]
    ):
        raise ValueError("checkpoint receipt evidence does not reconcile")
    return AuthenticatedParentCheckpoint(
        method=payload["method"],
        checkpoint_path=checkpoint_path,
        checkpoint_tree_sha256=payload["checkpoint_tree_sha256"],
        historical_receipt_file_sha256=payload["historical_receipt_file_sha256"],
        historical_ordered_prompt_uuids_sha256=payload["historical_ordered_prompt_uuids_sha256"],
        runtime_sha256=payload["runtime_sha256"],
        source_commit=payload["source_commit"],
        receipt_path=path,
        receipt_file_sha256=expected_sha256,
        receipt_sha256=payload["receipt_sha256"],
    )


def _validate_full_schedule(full_steps: object, global_batch_size: object) -> None:
    if (
        type(full_steps) is not int
        or type(global_batch_size) is not int
        or full_steps <= 20
        or global_batch_size < 1
        or full_steps * global_batch_size != 700_000
    ):
        raise ValueError("full schedule must exceed the 20-step canary with exact exposure")


def make_continuation_contract(
    parent: AuthenticatedParentCheckpoint,
    *,
    dataset_manifest_path: Path,
    dataset_manifest_file_sha256: str,
    dataset_completion_receipt_path: Path,
    dataset_completion_receipt_file_sha256: str,
    dataset_builder_tool_path: Path,
    trajectory_schema_path: Path,
    specdec_identity_path: Path,
    specdec_corpus_contracts_path: Path,
    historical_authenticator_path: Path,
    config_path: Path,
    source_inventory_path: Path,
    source_inventory_file_sha256: str,
    historical_receipt_path: Path,
    held_out_receipts: tuple[tuple[str, Path, str], ...],
    tokenizer_trust_path: Path,
    tokenizer_trust_file_sha256: str,
    verification_scratch_root: Path,
    canary_execution_stage_root: Path,
    canary_checkpoint_output_root: Path,
    canary_evidence_path: Path,
    canary_receipt_path: Path,
    full_execution_stage_root: Path,
    full_checkpoint_output_root: Path,
    full_evidence_path: Path,
    full_receipt_path: Path,
    training_entrypoint_path: Path,
    training_entrypoint_sha256: str,
    runtime_image_path: Path,
    runtime_image_sha256: str,
    full_steps: int,
    global_batch_size: int,
    dataset_identity: str,
    run_identity: str,
) -> ContinuationContract:
    """Create a canary-gated weights-only continuation with fresh training state."""
    _validate_full_schedule(full_steps, global_batch_size)
    if (
        dataset_identity != DATASET_IDENTITY
        or not _is_hash(dataset_manifest_file_sha256, 64)
        or not isinstance(run_identity, str)
        or not run_identity
        or dataset_manifest_path.is_symlink()
        or not dataset_manifest_path.is_file()
    ):
        raise ValueError("continuation run identity is invalid")
    _authenticate_dataset_bundle(
        dataset_manifest_path,
        manifest_file_sha256=dataset_manifest_file_sha256,
        completion_receipt_path=dataset_completion_receipt_path,
        completion_receipt_file_sha256=dataset_completion_receipt_file_sha256,
        historical_receipt_file_sha256=parent.historical_receipt_file_sha256,
        historical_ordered_prompt_uuids_sha256=parent.historical_ordered_prompt_uuids_sha256,
    )
    _validate_fresh_disjoint_roots(
        protected=(
            dataset_manifest_path.parent,
            dataset_completion_receipt_path,
            parent.checkpoint_path,
            parent.receipt_path,
            config_path,
            source_inventory_path,
            historical_receipt_path,
            tokenizer_trust_path,
            training_entrypoint_path,
            runtime_image_path,
            *_semantic_protected_paths(
                source_inventory_path=source_inventory_path,
                source_inventory_file_sha256=source_inventory_file_sha256,
                tokenizer_trust_path=tokenizer_trust_path,
                tokenizer_trust_file_sha256=tokenizer_trust_file_sha256,
                held_out_receipts=held_out_receipts,
                dataset_builder_tool_path=dataset_builder_tool_path,
                trajectory_schema_path=trajectory_schema_path,
                specdec_identity_path=specdec_identity_path,
                specdec_corpus_contracts_path=specdec_corpus_contracts_path,
                historical_authenticator_path=historical_authenticator_path,
            ),
        ),
        writable=(
            verification_scratch_root,
            canary_execution_stage_root,
            canary_checkpoint_output_root,
            full_execution_stage_root,
            full_checkpoint_output_root,
        ),
        writable_files=(
            canary_evidence_path,
            canary_receipt_path,
            full_evidence_path,
            full_receipt_path,
        ),
        allow_empty_directories=True,
    )
    _validate_stage_isolation(
        canary=(
            canary_execution_stage_root,
            canary_checkpoint_output_root,
            canary_evidence_path.parent,
            canary_receipt_path.parent,
        ),
        full=(
            full_execution_stage_root,
            full_checkpoint_output_root,
            full_evidence_path.parent,
            full_receipt_path.parent,
        ),
    )
    _run_builder_semantic_verifier(
        builder_tool_path=dataset_builder_tool_path,
        trajectory_schema_path=trajectory_schema_path,
        specdec_identity_path=specdec_identity_path,
        specdec_corpus_contracts_path=specdec_corpus_contracts_path,
        historical_authenticator_path=historical_authenticator_path,
        config_path=config_path,
        source_inventory_path=source_inventory_path,
        source_inventory_file_sha256=source_inventory_file_sha256,
        historical_receipt_path=historical_receipt_path,
        historical_receipt_file_sha256=parent.historical_receipt_file_sha256,
        held_out_receipts=held_out_receipts,
        tokenizer_trust_path=tokenizer_trust_path,
        tokenizer_trust_file_sha256=tokenizer_trust_file_sha256,
        verification_scratch_root=verification_scratch_root,
        manifest_path=dataset_manifest_path,
        manifest_file_sha256=dataset_manifest_file_sha256,
    )
    if (
        not TRUSTED_TRAINING_ENTRYPOINT_SHA256
        or not TRUSTED_RUNTIME_IMAGE_SHA256
        or parent.receipt_file_sha256 not in TRUSTED_PARENT_CHECKPOINT_RECEIPT_SHA256S
        or _regular_file_evidence(training_entrypoint_path)[1] != TRUSTED_TRAINING_ENTRYPOINT_SHA256
        or _regular_file_evidence(runtime_image_path)[1] != TRUSTED_RUNTIME_IMAGE_SHA256
        or training_entrypoint_sha256 != TRUSTED_TRAINING_ENTRYPOINT_SHA256
        or runtime_image_sha256 != TRUSTED_RUNTIME_IMAGE_SHA256
    ):
        raise ValueError("continuation runtime or exact full exposure is not approved")
    return ContinuationContract(
        method=parent.method,
        run_identity=run_identity,
        dataset_identity=dataset_identity,
        dataset_manifest_path=dataset_manifest_path,
        dataset_manifest_file_sha256=dataset_manifest_file_sha256,
        dataset_completion_receipt_path=dataset_completion_receipt_path,
        dataset_completion_receipt_file_sha256=dataset_completion_receipt_file_sha256,
        dataset_builder_tool_path=dataset_builder_tool_path,
        dataset_builder_tool_sha256=TRUSTED_DATASET_BUILDER_SHA256,
        trajectory_schema_path=trajectory_schema_path,
        trajectory_schema_sha256=TRUSTED_TRAJECTORY_SCHEMA_SHA256,
        specdec_identity_path=specdec_identity_path,
        specdec_identity_sha256=TRUSTED_SPECDEC_IDENTITY_SHA256,
        specdec_corpus_contracts_path=specdec_corpus_contracts_path,
        specdec_corpus_contracts_sha256=TRUSTED_SPECDEC_CORPUS_CONTRACTS_SHA256,
        historical_authenticator_path=historical_authenticator_path,
        historical_authenticator_sha256=TRUSTED_HISTORICAL_AUTHENTICATOR_SHA256,
        config_path=config_path,
        config_file_sha256=sha256(_stable_file(config_path)).hexdigest(),
        source_inventory_path=source_inventory_path,
        source_inventory_file_sha256=source_inventory_file_sha256,
        historical_receipt_path=historical_receipt_path,
        historical_receipt_file_sha256=parent.historical_receipt_file_sha256,
        historical_ordered_prompt_uuids_sha256=parent.historical_ordered_prompt_uuids_sha256,
        held_out_receipts=held_out_receipts,
        tokenizer_trust_path=tokenizer_trust_path,
        tokenizer_trust_file_sha256=tokenizer_trust_file_sha256,
        verification_scratch_root=verification_scratch_root,
        canary_execution_stage_root=canary_execution_stage_root,
        canary_checkpoint_output_root=canary_checkpoint_output_root,
        canary_evidence_path=canary_evidence_path,
        canary_receipt_path=canary_receipt_path,
        full_execution_stage_root=full_execution_stage_root,
        full_checkpoint_output_root=full_checkpoint_output_root,
        full_evidence_path=full_evidence_path,
        full_receipt_path=full_receipt_path,
        parent_checkpoint_path=parent.checkpoint_path,
        parent_checkpoint_tree_sha256=parent.checkpoint_tree_sha256,
        parent_receipt_path=parent.receipt_path,
        parent_receipt_file_sha256=parent.receipt_file_sha256,
        training_entrypoint_path=training_entrypoint_path,
        training_entrypoint_sha256=training_entrypoint_sha256,
        runtime_image_path=runtime_image_path,
        runtime_image_sha256=runtime_image_sha256,
        full_steps=full_steps,
        global_batch_size=global_batch_size,
    )


def _contract_payload(contract: ContinuationContract) -> dict[str, object]:
    body: dict[str, object] = {
        "schema_version": "qwen4b-ptv23-continuation-contract-v1",
        "method": contract.method,
        "run_identity": contract.run_identity,
        "dataset_identity": contract.dataset_identity,
        "dataset_manifest_path": str(contract.dataset_manifest_path),
        "dataset_manifest_file_sha256": contract.dataset_manifest_file_sha256,
        "dataset_completion_receipt_path": str(contract.dataset_completion_receipt_path),
        "dataset_completion_receipt_file_sha256": contract.dataset_completion_receipt_file_sha256,
        "dataset_builder_tool_path": str(contract.dataset_builder_tool_path),
        "dataset_builder_tool_sha256": contract.dataset_builder_tool_sha256,
        "trajectory_schema_path": str(contract.trajectory_schema_path),
        "trajectory_schema_sha256": contract.trajectory_schema_sha256,
        "specdec_identity_path": str(contract.specdec_identity_path),
        "specdec_identity_sha256": contract.specdec_identity_sha256,
        "specdec_corpus_contracts_path": str(contract.specdec_corpus_contracts_path),
        "specdec_corpus_contracts_sha256": contract.specdec_corpus_contracts_sha256,
        "historical_authenticator_path": str(contract.historical_authenticator_path),
        "historical_authenticator_sha256": contract.historical_authenticator_sha256,
        "config_path": str(contract.config_path),
        "config_file_sha256": contract.config_file_sha256,
        "source_inventory_path": str(contract.source_inventory_path),
        "source_inventory_file_sha256": contract.source_inventory_file_sha256,
        "historical_receipt_path": str(contract.historical_receipt_path),
        "historical_receipt_file_sha256": contract.historical_receipt_file_sha256,
        "historical_ordered_prompt_uuids_sha256": (contract.historical_ordered_prompt_uuids_sha256),
        "held_out_receipts": [
            {"name": name, "path": str(path), "sha256": digest}
            for name, path, digest in contract.held_out_receipts
        ],
        "tokenizer_trust_path": str(contract.tokenizer_trust_path),
        "tokenizer_trust_file_sha256": contract.tokenizer_trust_file_sha256,
        "verification_scratch_root": str(contract.verification_scratch_root),
        "canary_execution_stage_root": str(contract.canary_execution_stage_root),
        "canary_checkpoint_output_root": str(contract.canary_checkpoint_output_root),
        "canary_evidence_path": str(contract.canary_evidence_path),
        "canary_receipt_path": str(contract.canary_receipt_path),
        "full_execution_stage_root": str(contract.full_execution_stage_root),
        "full_checkpoint_output_root": str(contract.full_checkpoint_output_root),
        "full_evidence_path": str(contract.full_evidence_path),
        "full_receipt_path": str(contract.full_receipt_path),
        "parent_checkpoint_path": str(contract.parent_checkpoint_path),
        "parent_checkpoint_tree_sha256": contract.parent_checkpoint_tree_sha256,
        "parent_receipt_path": str(contract.parent_receipt_path),
        "parent_receipt_file_sha256": contract.parent_receipt_file_sha256,
        "training_entrypoint_path": str(contract.training_entrypoint_path),
        "training_entrypoint_sha256": contract.training_entrypoint_sha256,
        "runtime_image_path": str(contract.runtime_image_path),
        "runtime_image_sha256": contract.runtime_image_sha256,
        "full_steps": contract.full_steps,
        "global_batch_size": contract.global_batch_size,
        "full_example_exposure": contract.full_example_exposure,
        "resume_mode": contract.resume_mode,
        "optimizer_state": contract.optimizer_state,
        "scheduler_state": contract.scheduler_state,
        "canary_steps": contract.canary_steps,
        "test_only_required": contract.test_only_required,
    }
    return body | {"contract_sha256": sha256(_canonical(body)).hexdigest()}


def write_continuation_contract(contract: ContinuationContract, path: Path) -> str:
    """Exclusively write one canonical continuation contract."""
    payload = _contract_payload(contract)
    raw = _canonical(payload) + b"\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())
    return sha256(raw).hexdigest()


def _contract_mount_plan(contract: ContinuationContract, stage: Literal["canary", "full"]) -> bytes:
    semantic_paths = _semantic_protected_paths(
        source_inventory_path=contract.source_inventory_path,
        source_inventory_file_sha256=contract.source_inventory_file_sha256,
        tokenizer_trust_path=contract.tokenizer_trust_path,
        tokenizer_trust_file_sha256=contract.tokenizer_trust_file_sha256,
        held_out_receipts=contract.held_out_receipts,
        dataset_builder_tool_path=contract.dataset_builder_tool_path,
        trajectory_schema_path=contract.trajectory_schema_path,
        specdec_identity_path=contract.specdec_identity_path,
        specdec_corpus_contracts_path=contract.specdec_corpus_contracts_path,
        historical_authenticator_path=contract.historical_authenticator_path,
    )
    readonly_paths = (
        contract.dataset_manifest_path.parent,
        contract.dataset_completion_receipt_path,
        contract.parent_checkpoint_path,
        contract.parent_receipt_path,
        contract.config_path,
        contract.historical_receipt_path,
        contract.training_entrypoint_path,
        contract.runtime_image_path,
        *semantic_paths,
    )
    if stage == "full":
        readonly_paths += (
            contract.canary_checkpoint_output_root,
            contract.canary_evidence_path,
            contract.canary_receipt_path,
        )
    readonly_roots = sorted(
        {
            str((path if path.is_dir() else path.parent).resolve(strict=False))
            for path in readonly_paths
        }
    )
    if stage == "canary":
        execution_root = contract.canary_execution_stage_root
        checkpoint_root = contract.canary_checkpoint_output_root
        evidence_path = contract.canary_evidence_path
        receipt_path = contract.canary_receipt_path
    else:
        execution_root = contract.full_execution_stage_root
        checkpoint_root = contract.full_checkpoint_output_root
        evidence_path = contract.full_evidence_path
        receipt_path = contract.full_receipt_path
    plan = {
        "schema_version": "qwen4b-ptv23-mount-plan-v1",
        "readonly_roots": readonly_roots,
        "writable_roots": sorted(
            {
                str(path.resolve(strict=False))
                for path in (
                    contract.verification_scratch_root,
                    execution_root,
                    checkpoint_root,
                    evidence_path.parent,
                    receipt_path.parent,
                )
            }
        ),
        "execution_stage_root": str(execution_root),
        "checkpoint_output_root": str(checkpoint_root),
        "evidence_path": str(evidence_path),
        "receipt_path": str(receipt_path),
        "training_entrypoint_path": str(contract.training_entrypoint_path),
        "runtime_image_path": str(contract.runtime_image_path),
    }
    return _canonical(plan)


def load_continuation_contract(
    path: Path, *, expected_sha256: str, semantic_replay: bool = True
) -> ContinuationContract:
    """Authenticate a caller-pinned continuation contract and all live artifact roots."""
    if not _is_hash(expected_sha256, 64):
        raise ValueError("continuation contract caller SHA-256 is invalid")
    raw = _stable_file(path)
    if sha256(raw).hexdigest() != expected_sha256:
        raise ValueError("continuation contract caller SHA-256 mismatch")
    try:
        payload: Any = json.loads(raw)
    except json.JSONDecodeError as error:
        raise ValueError("continuation contract JSON is invalid") from error
    expected_keys = {
        "schema_version",
        "method",
        "run_identity",
        "dataset_identity",
        "dataset_manifest_path",
        "dataset_manifest_file_sha256",
        "dataset_completion_receipt_path",
        "dataset_completion_receipt_file_sha256",
        "dataset_builder_tool_path",
        "dataset_builder_tool_sha256",
        "trajectory_schema_path",
        "trajectory_schema_sha256",
        "specdec_identity_path",
        "specdec_identity_sha256",
        "specdec_corpus_contracts_path",
        "specdec_corpus_contracts_sha256",
        "historical_authenticator_path",
        "historical_authenticator_sha256",
        "config_path",
        "config_file_sha256",
        "source_inventory_path",
        "source_inventory_file_sha256",
        "historical_receipt_path",
        "historical_receipt_file_sha256",
        "historical_ordered_prompt_uuids_sha256",
        "held_out_receipts",
        "tokenizer_trust_path",
        "tokenizer_trust_file_sha256",
        "verification_scratch_root",
        "canary_execution_stage_root",
        "canary_checkpoint_output_root",
        "canary_evidence_path",
        "canary_receipt_path",
        "full_execution_stage_root",
        "full_checkpoint_output_root",
        "full_evidence_path",
        "full_receipt_path",
        "parent_checkpoint_path",
        "parent_checkpoint_tree_sha256",
        "parent_receipt_path",
        "parent_receipt_file_sha256",
        "training_entrypoint_path",
        "training_entrypoint_sha256",
        "runtime_image_path",
        "runtime_image_sha256",
        "full_steps",
        "global_batch_size",
        "full_example_exposure",
        "resume_mode",
        "optimizer_state",
        "scheduler_state",
        "canary_steps",
        "test_only_required",
        "contract_sha256",
    }
    if (
        not isinstance(payload, dict)
        or raw != _canonical(payload) + b"\n"
        or set(payload) != expected_keys
        or payload.get("schema_version") != "qwen4b-ptv23-continuation-contract-v1"
    ):
        raise ValueError("continuation contract identity is invalid")
    body = {key: value for key, value in payload.items() if key != "contract_sha256"}
    if payload.get("contract_sha256") != sha256(_canonical(body)).hexdigest():
        raise ValueError("continuation contract self identity does not reconcile")
    fixed = {
        "dataset_identity": DATASET_IDENTITY,
        "resume_mode": "weights-only",
        "optimizer_state": "fresh",
        "scheduler_state": "fresh",
        "canary_steps": 20,
        "test_only_required": True,
        "full_example_exposure": 700_000,
    }
    if any(payload.get(key) != value for key, value in fixed.items()) or payload.get(
        "method"
    ) not in {"DFlash", "DSpark"}:
        raise ValueError("continuation contract training state is invalid")
    full_steps = payload.get("full_steps")
    global_batch_size = payload.get("global_batch_size")
    try:
        _validate_full_schedule(full_steps, global_batch_size)
    except ValueError as error:
        raise ValueError("continuation contract full exposure is invalid") from error
    if (
        not _is_hash(payload.get("training_entrypoint_sha256"), 64)
        or not _is_hash(payload.get("runtime_image_sha256"), 64)
        or payload.get("training_entrypoint_sha256") != TRUSTED_TRAINING_ENTRYPOINT_SHA256
        or payload.get("runtime_image_sha256") != TRUSTED_RUNTIME_IMAGE_SHA256
        or payload.get("parent_receipt_file_sha256")
        not in TRUSTED_PARENT_CHECKPOINT_RECEIPT_SHA256S
    ):
        raise ValueError("continuation contract full exposure is invalid")
    dataset_path = Path(payload["dataset_manifest_path"])
    completion_receipt_path = Path(payload["dataset_completion_receipt_path"])
    parent_path = Path(payload["parent_checkpoint_path"])
    held_out_payload = payload.get("held_out_receipts")
    if (
        not isinstance(held_out_payload, list)
        or len(held_out_payload) != 5
        or any(
            not isinstance(record, dict)
            or set(record) != {"name", "path", "sha256"}
            or not isinstance(record["name"], str)
            or not isinstance(record["path"], str)
            or not _is_hash(record["sha256"], 64)
            for record in held_out_payload
        )
        or {record["name"] for record in held_out_payload}
        != {"speed", "math", "code", "swe", "tool"}
    ):
        raise ValueError("continuation contract held-out roots are invalid")
    held_out_receipts = tuple(
        (record["name"], Path(record["path"]), record["sha256"]) for record in held_out_payload
    )
    _validate_fresh_disjoint_roots(
        protected=(
            dataset_path.parent,
            completion_receipt_path,
            parent_path,
            Path(payload["parent_receipt_path"]),
            Path(payload["config_path"]),
            Path(payload["source_inventory_path"]),
            Path(payload["historical_receipt_path"]),
            Path(payload["tokenizer_trust_path"]),
            Path(payload["training_entrypoint_path"]),
            Path(payload["runtime_image_path"]),
            *_semantic_protected_paths(
                source_inventory_path=Path(payload["source_inventory_path"]),
                source_inventory_file_sha256=payload["source_inventory_file_sha256"],
                tokenizer_trust_path=Path(payload["tokenizer_trust_path"]),
                tokenizer_trust_file_sha256=payload["tokenizer_trust_file_sha256"],
                held_out_receipts=held_out_receipts,
                dataset_builder_tool_path=Path(payload["dataset_builder_tool_path"]),
                trajectory_schema_path=Path(payload["trajectory_schema_path"]),
                specdec_identity_path=Path(payload["specdec_identity_path"]),
                specdec_corpus_contracts_path=Path(payload["specdec_corpus_contracts_path"]),
                historical_authenticator_path=Path(payload["historical_authenticator_path"]),
                contract_path=path,
            ),
        ),
        writable=(
            Path(payload["verification_scratch_root"]),
            Path(payload["canary_execution_stage_root"]),
            Path(payload["canary_checkpoint_output_root"]),
            Path(payload["full_execution_stage_root"]),
            Path(payload["full_checkpoint_output_root"]),
        ),
        writable_files=(
            Path(payload["canary_evidence_path"]),
            Path(payload["canary_receipt_path"]),
            Path(payload["full_evidence_path"]),
            Path(payload["full_receipt_path"]),
        ),
        require_fresh=False,
    )
    _validate_stage_isolation(
        canary=(
            Path(payload["canary_execution_stage_root"]),
            Path(payload["canary_checkpoint_output_root"]),
            Path(payload["canary_evidence_path"]).parent,
            Path(payload["canary_receipt_path"]).parent,
        ),
        full=(
            Path(payload["full_execution_stage_root"]),
            Path(payload["full_checkpoint_output_root"]),
            Path(payload["full_evidence_path"]).parent,
            Path(payload["full_receipt_path"]).parent,
        ),
    )
    if (
        dataset_path.is_symlink()
        or not dataset_path.is_file()
        or sha256(_stable_file(dataset_path)).hexdigest()
        != payload.get("dataset_manifest_file_sha256")
        or _checkpoint_tree_sha256(parent_path) != payload.get("parent_checkpoint_tree_sha256")
        or payload.get("dataset_builder_tool_sha256") != TRUSTED_DATASET_BUILDER_SHA256
        or payload.get("trajectory_schema_sha256") != TRUSTED_TRAJECTORY_SCHEMA_SHA256
        or payload.get("specdec_identity_sha256") != TRUSTED_SPECDEC_IDENTITY_SHA256
        or payload.get("specdec_corpus_contracts_sha256") != TRUSTED_SPECDEC_CORPUS_CONTRACTS_SHA256
        or payload.get("historical_authenticator_sha256") != TRUSTED_HISTORICAL_AUTHENTICATOR_SHA256
        or payload.get("tokenizer_trust_file_sha256") != TRUSTED_TOKENIZER_TRUST_FILE_SHA256
        or _regular_file_evidence(Path(payload["training_entrypoint_path"]))[1]
        != payload.get("training_entrypoint_sha256")
        or _regular_file_evidence(Path(payload["runtime_image_path"]))[1]
        != payload.get("runtime_image_sha256")
        or sha256(_stable_file(Path(payload["config_path"]))).hexdigest()
        != payload.get("config_file_sha256")
    ):
        raise ValueError("continuation contract live artifact mismatch")
    _authenticate_dataset_bundle(
        dataset_path,
        manifest_file_sha256=payload["dataset_manifest_file_sha256"],
        completion_receipt_path=completion_receipt_path,
        completion_receipt_file_sha256=payload["dataset_completion_receipt_file_sha256"],
        historical_receipt_file_sha256=payload["historical_receipt_file_sha256"],
        historical_ordered_prompt_uuids_sha256=payload["historical_ordered_prompt_uuids_sha256"],
    )
    parent = load_authenticated_parent_checkpoint(
        Path(payload["parent_receipt_path"]),
        expected_sha256=payload["parent_receipt_file_sha256"],
    )
    if (
        parent.method != payload["method"]
        or parent.checkpoint_path != parent_path
        or parent.checkpoint_tree_sha256 != payload["parent_checkpoint_tree_sha256"]
        or parent.historical_receipt_file_sha256 != payload["historical_receipt_file_sha256"]
        or parent.historical_ordered_prompt_uuids_sha256
        != payload["historical_ordered_prompt_uuids_sha256"]
    ):
        raise ValueError("continuation contract parent receipt mismatch")
    if semantic_replay:
        _run_builder_semantic_verifier(
            builder_tool_path=Path(payload["dataset_builder_tool_path"]),
            trajectory_schema_path=Path(payload["trajectory_schema_path"]),
            specdec_identity_path=Path(payload["specdec_identity_path"]),
            specdec_corpus_contracts_path=Path(payload["specdec_corpus_contracts_path"]),
            historical_authenticator_path=Path(payload["historical_authenticator_path"]),
            config_path=Path(payload["config_path"]),
            source_inventory_path=Path(payload["source_inventory_path"]),
            source_inventory_file_sha256=payload["source_inventory_file_sha256"],
            historical_receipt_path=Path(payload["historical_receipt_path"]),
            historical_receipt_file_sha256=payload["historical_receipt_file_sha256"],
            held_out_receipts=held_out_receipts,
            tokenizer_trust_path=Path(payload["tokenizer_trust_path"]),
            tokenizer_trust_file_sha256=payload["tokenizer_trust_file_sha256"],
            verification_scratch_root=Path(payload["verification_scratch_root"]),
            manifest_path=dataset_path,
            manifest_file_sha256=payload["dataset_manifest_file_sha256"],
        )
    return ContinuationContract(
        method=payload["method"],
        run_identity=payload["run_identity"],
        dataset_identity=payload["dataset_identity"],
        dataset_manifest_path=dataset_path,
        dataset_manifest_file_sha256=payload["dataset_manifest_file_sha256"],
        dataset_completion_receipt_path=completion_receipt_path,
        dataset_completion_receipt_file_sha256=payload["dataset_completion_receipt_file_sha256"],
        dataset_builder_tool_path=Path(payload["dataset_builder_tool_path"]),
        dataset_builder_tool_sha256=payload["dataset_builder_tool_sha256"],
        trajectory_schema_path=Path(payload["trajectory_schema_path"]),
        trajectory_schema_sha256=payload["trajectory_schema_sha256"],
        specdec_identity_path=Path(payload["specdec_identity_path"]),
        specdec_identity_sha256=payload["specdec_identity_sha256"],
        specdec_corpus_contracts_path=Path(payload["specdec_corpus_contracts_path"]),
        specdec_corpus_contracts_sha256=payload["specdec_corpus_contracts_sha256"],
        historical_authenticator_path=Path(payload["historical_authenticator_path"]),
        historical_authenticator_sha256=payload["historical_authenticator_sha256"],
        config_path=Path(payload["config_path"]),
        config_file_sha256=payload["config_file_sha256"],
        source_inventory_path=Path(payload["source_inventory_path"]),
        source_inventory_file_sha256=payload["source_inventory_file_sha256"],
        historical_receipt_path=Path(payload["historical_receipt_path"]),
        historical_receipt_file_sha256=payload["historical_receipt_file_sha256"],
        historical_ordered_prompt_uuids_sha256=payload["historical_ordered_prompt_uuids_sha256"],
        held_out_receipts=held_out_receipts,
        tokenizer_trust_path=Path(payload["tokenizer_trust_path"]),
        tokenizer_trust_file_sha256=payload["tokenizer_trust_file_sha256"],
        verification_scratch_root=Path(payload["verification_scratch_root"]),
        canary_execution_stage_root=Path(payload["canary_execution_stage_root"]),
        canary_checkpoint_output_root=Path(payload["canary_checkpoint_output_root"]),
        canary_evidence_path=Path(payload["canary_evidence_path"]),
        canary_receipt_path=Path(payload["canary_receipt_path"]),
        full_execution_stage_root=Path(payload["full_execution_stage_root"]),
        full_checkpoint_output_root=Path(payload["full_checkpoint_output_root"]),
        full_evidence_path=Path(payload["full_evidence_path"]),
        full_receipt_path=Path(payload["full_receipt_path"]),
        parent_checkpoint_path=parent_path,
        parent_checkpoint_tree_sha256=payload["parent_checkpoint_tree_sha256"],
        parent_receipt_path=Path(payload["parent_receipt_path"]),
        parent_receipt_file_sha256=payload["parent_receipt_file_sha256"],
        training_entrypoint_path=Path(payload["training_entrypoint_path"]),
        training_entrypoint_sha256=payload["training_entrypoint_sha256"],
        runtime_image_path=Path(payload["runtime_image_path"]),
        runtime_image_sha256=payload["runtime_image_sha256"],
        full_steps=payload["full_steps"],
        global_batch_size=payload["global_batch_size"],
    )


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter,
        description=(
            "Create or launch a weights-only Qwen3-4B continuation with fresh optimizer and "
            "scheduler state."
        ),
    )
    commands = parser.add_subparsers(dest="command", required=True)
    preflight = commands.add_parser("preflight-contract")
    preflight.add_argument("--contract", type=Path, required=True)
    preflight.add_argument("--contract-sha256", required=True)
    preflight.add_argument("--stage", choices=("canary", "full"), required=True)
    make = commands.add_parser("make-contract")
    make.add_argument("--parent-receipt", type=Path, required=True)
    make.add_argument("--parent-receipt-sha256", required=True)
    make.add_argument("--dataset-manifest", type=Path, required=True)
    make.add_argument("--dataset-manifest-sha256", required=True)
    make.add_argument("--dataset-completion-receipt", type=Path, required=True)
    make.add_argument("--dataset-completion-receipt-sha256", required=True)
    make.add_argument("--dataset-builder-tool", type=Path, required=True)
    make.add_argument("--trajectory-schema", type=Path, required=True)
    make.add_argument("--specdec-identity", type=Path, required=True)
    make.add_argument("--specdec-corpus-contracts", type=Path, required=True)
    make.add_argument("--historical-authenticator", type=Path, required=True)
    make.add_argument("--config", type=Path, required=True)
    make.add_argument("--source-inventory", type=Path, required=True)
    make.add_argument("--source-inventory-sha256", required=True)
    make.add_argument("--historical-receipt", type=Path, required=True)
    make.add_argument("--held-out-receipt", metavar="NAME=PATH:SHA256", action="append", default=[])
    make.add_argument("--tokenizer-trust", type=Path, required=True)
    make.add_argument("--tokenizer-trust-sha256", required=True)
    make.add_argument("--verification-scratch-root", type=Path, required=True)
    make.add_argument("--canary-execution-stage-root", type=Path, required=True)
    make.add_argument("--canary-checkpoint-output-root", type=Path, required=True)
    make.add_argument("--canary-evidence", type=Path, required=True)
    make.add_argument("--canary-receipt", type=Path, required=True)
    make.add_argument("--full-execution-stage-root", type=Path, required=True)
    make.add_argument("--full-checkpoint-output-root", type=Path, required=True)
    make.add_argument("--full-completion-evidence", type=Path, required=True)
    make.add_argument("--full-completion-receipt", type=Path, required=True)
    make.add_argument("--training-entrypoint", type=Path, required=True)
    make.add_argument("--training-entrypoint-sha256", required=True)
    make.add_argument("--runtime-image", type=Path, required=True)
    make.add_argument("--runtime-image-sha256", required=True)
    make.add_argument("--full-steps", type=int, required=True)
    make.add_argument("--global-batch-size", type=int, required=True)
    make.add_argument("--run-identity", required=True)
    make.add_argument("--output", type=Path, required=True)
    launch = commands.add_parser("launch")
    launch.add_argument("--contract", type=Path, required=True)
    launch.add_argument("--contract-sha256", required=True)
    launch.add_argument("--training-entrypoint", type=Path, required=True)
    launch.add_argument("--training-entrypoint-sha256", required=True)
    launch.add_argument("--runtime-image", type=Path, required=True)
    launch.add_argument("--runtime-image-sha256", required=True)
    launch.add_argument("--checkpoint-output-root", type=Path, required=True)
    launch.add_argument("--stage", choices=("canary", "full"), required=True)
    launch.add_argument("--max-steps", type=int, required=True)
    launch.add_argument("--resume-mode", choices=("weights-only",), required=True)
    launch.add_argument("--optimizer-state", choices=("fresh",), required=True)
    launch.add_argument("--scheduler-state", choices=("fresh",), required=True)
    launch.add_argument("--canary-receipt", type=Path)
    launch.add_argument("--canary-receipt-sha256")
    launch.add_argument("--canary-evidence", type=Path)
    launch.add_argument("--full-completion-evidence", type=Path)
    launch.add_argument("--full-completion-receipt", type=Path)
    return parser.parse_args(argv)


def _parse_held_out_roots(values: list[str]) -> tuple[tuple[str, Path, str], ...]:
    roots: list[tuple[str, Path, str]] = []
    for value in values:
        name, separator, remainder = value.partition("=")
        path, digest_separator, digest = remainder.rpartition(":")
        if (
            not separator
            or name not in {"speed", "math", "code", "swe", "tool"}
            or not digest_separator
            or not path
            or not _is_hash(digest, 64)
        ):
            raise ValueError("held-out root must be NAME=PATH:SHA256")
        roots.append((name, Path(path), digest))
    if len(roots) != 5 or {name for name, _, _ in roots} != {
        "speed",
        "math",
        "code",
        "swe",
        "tool",
    }:
        raise ValueError("exact five held-out roots are required")
    return tuple(roots)


def _validate_canary_receipt(
    path: Path, expected_sha256: str, contract_sha256: str, contract: ContinuationContract
) -> None:
    if expected_sha256 not in TRUSTED_CANARY_RECEIPT_SHA256S:
        raise ValueError("canary receipt is not immutable-approved")
    raw = _stable_file(path)
    if sha256(raw).hexdigest() != expected_sha256:
        raise ValueError("canary receipt caller SHA-256 mismatch")
    payload = json.loads(raw)
    if not isinstance(payload, dict):
        raise ValueError("canary receipt is invalid")
    evidence_path = Path(payload.get("training_evidence_path", ""))
    checkpoint_path = Path(payload.get("output_checkpoint_path", ""))
    checkpoint_root = contract.canary_checkpoint_output_root.resolve(strict=False)
    if (
        path.resolve(strict=False) != contract.canary_receipt_path.resolve(strict=False)
        or evidence_path.resolve(strict=False)
        != contract.canary_evidence_path.resolve(strict=False)
        or checkpoint_path.resolve(strict=False) == checkpoint_root
        or not checkpoint_path.resolve(strict=False).is_relative_to(checkpoint_root)
    ):
        raise ValueError("canary receipt does not use contract-bound paths")
    if (
        raw != _canonical(payload) + b"\n"
        or set(payload)
        != {
            "schema_version",
            "contract_file_sha256",
            "completed_steps",
            "status",
            "training_evidence_path",
            "training_evidence_file_sha256",
            "training_evidence_sha256",
            "training_entrypoint_sha256",
            "runtime_image_sha256",
            "parent_checkpoint_tree_sha256",
            "output_checkpoint_path",
            "output_checkpoint_tree_sha256",
        }
        or payload["schema_version"] != "qwen4b-ptv23-continuation-canary-v1"
        or payload["contract_file_sha256"] != contract_sha256
        or payload["completed_steps"] != 20
        or payload["status"] != "passed"
        or not isinstance(payload["training_evidence_path"], str)
        or not _is_hash(payload["training_evidence_file_sha256"], 64)
        or not _is_hash(payload["training_evidence_sha256"], 64)
        or payload["training_entrypoint_sha256"] != contract.training_entrypoint_sha256
        or payload["runtime_image_sha256"] != contract.runtime_image_sha256
        or payload["parent_checkpoint_tree_sha256"] != contract.parent_checkpoint_tree_sha256
        or _checkpoint_tree_sha256(checkpoint_path) != payload["output_checkpoint_tree_sha256"]
    ):
        raise ValueError("canary receipt is invalid")
    evidence_file_sha256, evidence_sha256, evidence = _validate_training_canary_evidence(
        evidence_path,
        contract_sha256=contract_sha256,
        run_identity=contract.run_identity,
        training_entrypoint_sha256=contract.training_entrypoint_sha256,
        runtime_image_sha256=contract.runtime_image_sha256,
        parent_checkpoint_tree_sha256=contract.parent_checkpoint_tree_sha256,
    )
    if (
        evidence_file_sha256 != payload["training_evidence_file_sha256"]
        or evidence_sha256 != payload["training_evidence_sha256"]
        or evidence["output_checkpoint_path"] != payload["output_checkpoint_path"]
        or evidence["output_checkpoint_tree_sha256"] != payload["output_checkpoint_tree_sha256"]
    ):
        raise ValueError("canary receipt training evidence mismatch")


def _validate_training_canary_evidence(
    path: Path,
    *,
    contract_sha256: str,
    run_identity: str,
    training_entrypoint_sha256: str,
    runtime_image_sha256: str,
    parent_checkpoint_tree_sha256: str,
) -> tuple[str, str, dict[str, Any]]:
    if path.is_symlink() or not path.is_file():
        raise ValueError("training canary evidence is missing or not a regular file")
    raw = _stable_file(path)
    try:
        payload: Any = json.loads(raw)
    except json.JSONDecodeError as error:
        raise ValueError("training canary evidence JSON is invalid") from error
    expected_keys = {
        "schema_version",
        "contract_file_sha256",
        "run_identity",
        "observed_global_step",
        "canary_status",
        "training_entrypoint_sha256",
        "runtime_image_sha256",
        "parent_checkpoint_tree_sha256",
        "output_checkpoint_path",
        "output_checkpoint_tree_sha256",
        "evidence_sha256",
    }
    if not isinstance(payload, dict):
        raise ValueError("training canary evidence is invalid")
    body = {key: value for key, value in payload.items() if key != "evidence_sha256"}
    if (
        raw != _canonical(payload) + b"\n"
        or set(payload) != expected_keys
        or payload["schema_version"] != "qwen4b-ptv23-training-canary-evidence-v1"
        or payload["contract_file_sha256"] != contract_sha256
        or payload["run_identity"] != run_identity
        or payload["observed_global_step"] != 20
        or payload["canary_status"] != "passed"
        or payload["training_entrypoint_sha256"] != training_entrypoint_sha256
        or payload["runtime_image_sha256"] != runtime_image_sha256
        or payload["parent_checkpoint_tree_sha256"] != parent_checkpoint_tree_sha256
        or not isinstance(payload["output_checkpoint_path"], str)
        or _checkpoint_tree_sha256(Path(payload["output_checkpoint_path"]))
        != payload["output_checkpoint_tree_sha256"]
        or payload["evidence_sha256"] != sha256(_canonical(body)).hexdigest()
    ):
        raise ValueError("training canary evidence is invalid")
    return sha256(raw).hexdigest(), payload["evidence_sha256"], payload


def _validate_full_completion_evidence(
    path: Path, *, contract: ContinuationContract, contract_sha256: str
) -> tuple[dict[str, Any], str]:
    raw = _stable_file(path)
    try:
        payload: Any = json.loads(raw)
    except json.JSONDecodeError as error:
        raise ValueError("full completion evidence JSON is invalid") from error
    expected_keys = {
        "schema_version",
        "contract_file_sha256",
        "run_identity",
        "training_entrypoint_sha256",
        "runtime_image_sha256",
        "parent_checkpoint_tree_sha256",
        "observed_global_step",
        "consumed_examples",
        "status",
        "output_checkpoint_path",
        "output_checkpoint_tree_sha256",
        "evidence_sha256",
    }
    if not isinstance(payload, dict):
        raise ValueError("full completion evidence is invalid")
    body = {key: value for key, value in payload.items() if key != "evidence_sha256"}
    if (
        raw != _canonical(payload) + b"\n"
        or set(payload) != expected_keys
        or payload["schema_version"] != "qwen4b-ptv23-full-completion-evidence-v1"
        or payload["contract_file_sha256"] != contract_sha256
        or payload["run_identity"] != contract.run_identity
        or payload["training_entrypoint_sha256"] != contract.training_entrypoint_sha256
        or payload["runtime_image_sha256"] != contract.runtime_image_sha256
        or payload["parent_checkpoint_tree_sha256"] != contract.parent_checkpoint_tree_sha256
        or payload["observed_global_step"] != contract.full_steps
        or payload["consumed_examples"] != contract.full_example_exposure
        or payload["status"] != "passed"
        or _checkpoint_tree_sha256(Path(payload["output_checkpoint_path"]))
        != payload["output_checkpoint_tree_sha256"]
        or payload["evidence_sha256"] != sha256(_canonical(body)).hexdigest()
    ):
        raise ValueError("full completion evidence is invalid")
    return payload, sha256(raw).hexdigest()


def _trainer_environment() -> dict[str, str]:
    environment = {
        "HOME": "/tmp",
        "PATH": "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin",
        "PYTHONNOUSERSITE": "1",
        "TMPDIR": "/tmp",
    }
    scheduler_keys = {
        "CUDA_DEVICE_ORDER",
        "CUDA_VISIBLE_DEVICES",
        "LOCAL_RANK",
        "LOCAL_WORLD_SIZE",
        "MASTER_ADDR",
        "MASTER_PORT",
        "NCCL_SOCKET_IFNAME",
        "NVIDIA_VISIBLE_DEVICES",
        "NODE_RANK",
        "RANK",
        "WORLD_SIZE",
    }
    environment.update(
        (key, value)
        for key, value in os.environ.items()
        if key.startswith("SLURM_") or key in scheduler_keys
    )
    return environment


def main(argv: list[str] | None = None) -> int:
    """Create a contract or launch its authenticated canary/full stage."""
    args = _parse_args(argv)
    if args.command == "preflight-contract":
        if not sys.flags.isolated:
            raise ValueError("contract preflight requires isolated Python import resolution")
        contract = load_continuation_contract(
            args.contract, expected_sha256=args.contract_sha256, semantic_replay=False
        )
        print(_contract_mount_plan(contract, args.stage).decode())
        return 0
    if args.command == "make-contract":
        parent = load_authenticated_parent_checkpoint(
            args.parent_receipt, expected_sha256=args.parent_receipt_sha256
        )
        contract = make_continuation_contract(
            parent,
            dataset_manifest_path=args.dataset_manifest,
            dataset_manifest_file_sha256=args.dataset_manifest_sha256,
            dataset_completion_receipt_path=args.dataset_completion_receipt,
            dataset_completion_receipt_file_sha256=args.dataset_completion_receipt_sha256,
            dataset_builder_tool_path=args.dataset_builder_tool,
            trajectory_schema_path=args.trajectory_schema,
            specdec_identity_path=args.specdec_identity,
            specdec_corpus_contracts_path=args.specdec_corpus_contracts,
            historical_authenticator_path=args.historical_authenticator,
            config_path=args.config,
            source_inventory_path=args.source_inventory,
            source_inventory_file_sha256=args.source_inventory_sha256,
            historical_receipt_path=args.historical_receipt,
            held_out_receipts=_parse_held_out_roots(args.held_out_receipt),
            tokenizer_trust_path=args.tokenizer_trust,
            tokenizer_trust_file_sha256=args.tokenizer_trust_sha256,
            verification_scratch_root=args.verification_scratch_root,
            canary_execution_stage_root=args.canary_execution_stage_root,
            canary_checkpoint_output_root=args.canary_checkpoint_output_root,
            canary_evidence_path=args.canary_evidence,
            canary_receipt_path=args.canary_receipt,
            full_execution_stage_root=args.full_execution_stage_root,
            full_checkpoint_output_root=args.full_checkpoint_output_root,
            full_evidence_path=args.full_completion_evidence,
            full_receipt_path=args.full_completion_receipt,
            training_entrypoint_path=args.training_entrypoint,
            training_entrypoint_sha256=args.training_entrypoint_sha256,
            runtime_image_path=args.runtime_image,
            runtime_image_sha256=args.runtime_image_sha256,
            full_steps=args.full_steps,
            global_batch_size=args.global_batch_size,
            dataset_identity=DATASET_IDENTITY,
            run_identity=args.run_identity,
        )
        print(write_continuation_contract(contract, args.output))
        return 0
    if not RUNTIME_ATTESTATION_IMPLEMENTED:
        raise ValueError("runtime image attestation is not implemented; launch is fail-closed")
    contract = load_continuation_contract(args.contract, expected_sha256=args.contract_sha256)
    if (
        args.training_entrypoint != contract.training_entrypoint_path
        or args.runtime_image != contract.runtime_image_path
        or args.training_entrypoint_sha256 != contract.training_entrypoint_sha256
        or args.runtime_image_sha256 != contract.runtime_image_sha256
        or _regular_file_evidence(args.training_entrypoint)[1]
        != contract.training_entrypoint_sha256
    ):
        raise ValueError("training entrypoint caller SHA-256 mismatch")
    if args.stage == "canary" and args.max_steps != 20:
        raise ValueError("canary must run exactly 20 steps")
    if args.stage == "canary":
        execution_stage_root = contract.canary_execution_stage_root
        checkpoint_output_root = contract.canary_checkpoint_output_root
        expected_evidence_path = contract.canary_evidence_path
        expected_receipt_path = contract.canary_receipt_path
        if (
            args.checkpoint_output_root != checkpoint_output_root
            or args.canary_evidence != expected_evidence_path
            or args.canary_receipt != expected_receipt_path
        ):
            raise ValueError("canary writable roots do not match the authenticated contract")
    else:
        execution_stage_root = contract.full_execution_stage_root
        checkpoint_output_root = contract.full_checkpoint_output_root
        expected_evidence_path = contract.full_evidence_path
        expected_receipt_path = contract.full_receipt_path
        if (
            args.checkpoint_output_root != checkpoint_output_root
            or args.full_completion_evidence != expected_evidence_path
            or args.full_completion_receipt != expected_receipt_path
            or args.canary_receipt != contract.canary_receipt_path
        ):
            raise ValueError("full writable roots do not match the authenticated contract")
    if args.stage == "full":
        if (
            args.max_steps != contract.full_steps
            or args.canary_receipt is None
            or args.canary_receipt_sha256 is None
            or args.full_completion_evidence is None
            or args.full_completion_receipt is None
            or args.full_completion_evidence.exists()
            or args.full_completion_receipt.exists()
        ):
            raise ValueError("full continuation requires exact exposure and fresh completion paths")
        _validate_canary_receipt(
            args.canary_receipt, args.canary_receipt_sha256, args.contract_sha256, contract
        )
    _validate_fresh_disjoint_roots(
        protected=(
            contract.dataset_manifest_path.parent,
            contract.dataset_completion_receipt_path,
            contract.parent_checkpoint_path,
            contract.parent_receipt_path,
            contract.config_path,
            contract.source_inventory_path,
            contract.historical_receipt_path,
            contract.tokenizer_trust_path,
            contract.training_entrypoint_path,
            contract.runtime_image_path,
            contract.verification_scratch_root,
            *_semantic_protected_paths(
                source_inventory_path=contract.source_inventory_path,
                source_inventory_file_sha256=contract.source_inventory_file_sha256,
                tokenizer_trust_path=contract.tokenizer_trust_path,
                tokenizer_trust_file_sha256=contract.tokenizer_trust_file_sha256,
                held_out_receipts=contract.held_out_receipts,
                dataset_builder_tool_path=contract.dataset_builder_tool_path,
                trajectory_schema_path=contract.trajectory_schema_path,
                specdec_identity_path=contract.specdec_identity_path,
                specdec_corpus_contracts_path=contract.specdec_corpus_contracts_path,
                historical_authenticator_path=contract.historical_authenticator_path,
                contract_path=args.contract,
            ),
        ),
        writable=(execution_stage_root, checkpoint_output_root),
        writable_files=(expected_evidence_path, expected_receipt_path),
        allow_empty_directories=True,
    )
    execution_stage_root.mkdir(parents=True, exist_ok=True)
    staged_parent = execution_stage_root / "parent"
    staged_dataset_manifest = execution_stage_root / "dataset" / contract.dataset_manifest_path.name
    staged_training_entrypoint = execution_stage_root / "trainer" / args.training_entrypoint.name
    try:
        _stage_authenticated_tree(
            contract.parent_checkpoint_path,
            staged_parent,
            expected_tree_sha256=contract.parent_checkpoint_tree_sha256,
        )
        staged_dataset_manifest = _stage_authenticated_dataset(
            contract, execution_stage_root / "dataset"
        )
        _stage_authenticated_file(
            args.training_entrypoint,
            staged_training_entrypoint,
            expected_sha256=contract.training_entrypoint_sha256,
        )
    except BaseException:
        shutil.rmtree(execution_stage_root, ignore_errors=True)
        raise
    command = [
        sys.executable,
        "-I",
        str(staged_training_entrypoint),
        "--model-name-or-path",
        str(staged_parent),
        "--dataset-manifest",
        str(staged_dataset_manifest),
        "--max-steps",
        str(args.max_steps),
        "--global-batch-size",
        str(contract.global_batch_size),
        "--max-examples",
        str(args.max_steps * contract.global_batch_size),
        "--dataset-repeat",
        "1",
        "--disable-dataset-cycling",
        "--resume-mode",
        "weights-only",
        "--reset-optimizer",
        "--reset-scheduler",
        "--run-name",
        contract.run_identity,
        "--output-dir",
        str(checkpoint_output_root),
    ]
    if args.stage == "canary":
        if args.canary_evidence is None or args.canary_evidence.exists():
            raise ValueError("canary launch requires a fresh training evidence path")
        command.extend(["--completion-evidence", str(args.canary_evidence)])
    else:
        command.extend(["--completion-evidence", str(args.full_completion_evidence)])
    try:
        subprocess.run(command, check=True, env=_trainer_environment())
    finally:
        shutil.rmtree(execution_stage_root, ignore_errors=True)
    if args.stage == "canary":
        if args.canary_receipt is None:
            raise ValueError("canary launch requires an exclusive output receipt path")
        evidence_file_sha256, evidence_sha256, evidence = _validate_training_canary_evidence(
            args.canary_evidence,
            contract_sha256=args.contract_sha256,
            run_identity=contract.run_identity,
            training_entrypoint_sha256=contract.training_entrypoint_sha256,
            runtime_image_sha256=contract.runtime_image_sha256,
            parent_checkpoint_tree_sha256=contract.parent_checkpoint_tree_sha256,
        )
        if (
            not Path(evidence["output_checkpoint_path"])
            .resolve(strict=False)
            .is_relative_to(args.checkpoint_output_root.resolve(strict=False))
        ):
            raise ValueError("canary checkpoint escaped the approved output root")
        payload = {
            "schema_version": "qwen4b-ptv23-continuation-canary-v1",
            "contract_file_sha256": args.contract_sha256,
            "completed_steps": 20,
            "status": "passed",
            "training_evidence_path": str(args.canary_evidence),
            "training_evidence_file_sha256": evidence_file_sha256,
            "training_evidence_sha256": evidence_sha256,
            "training_entrypoint_sha256": contract.training_entrypoint_sha256,
            "runtime_image_sha256": contract.runtime_image_sha256,
            "parent_checkpoint_tree_sha256": contract.parent_checkpoint_tree_sha256,
            "output_checkpoint_path": evidence["output_checkpoint_path"],
            "output_checkpoint_tree_sha256": evidence["output_checkpoint_tree_sha256"],
        }
        raw = _canonical(payload) + b"\n"
        args.canary_receipt.parent.mkdir(parents=True, exist_ok=True)
        with args.canary_receipt.open("xb") as stream:
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
        print(sha256(raw).hexdigest())
    else:
        evidence, evidence_file_sha256 = _validate_full_completion_evidence(
            args.full_completion_evidence,
            contract=contract,
            contract_sha256=args.contract_sha256,
        )
        if (
            not Path(evidence["output_checkpoint_path"])
            .resolve(strict=False)
            .is_relative_to(args.checkpoint_output_root.resolve(strict=False))
        ):
            raise ValueError("full checkpoint escaped the approved output root")
        body = {
            "schema_version": "qwen4b-ptv23-full-completion-receipt-v1",
            "contract_file_sha256": args.contract_sha256,
            "training_evidence_file_sha256": evidence_file_sha256,
            "training_evidence_sha256": evidence["evidence_sha256"],
            "completed_steps": contract.full_steps,
            "consumed_examples": contract.full_example_exposure,
            "output_checkpoint_path": evidence["output_checkpoint_path"],
            "output_checkpoint_tree_sha256": evidence["output_checkpoint_tree_sha256"],
            "status": "passed",
        }
        payload = body | {"receipt_sha256": sha256(_canonical(body)).hexdigest()}
        raw = _canonical(payload) + b"\n"
        args.full_completion_receipt.parent.mkdir(parents=True, exist_ok=True)
        with args.full_completion_receipt.open("xb") as stream:
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
        print(sha256(raw).hexdigest())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
