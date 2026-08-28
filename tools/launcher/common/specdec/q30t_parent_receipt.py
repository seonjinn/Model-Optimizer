# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Build and verify canonical receipts for the approved Q30 Thinking parents."""

from __future__ import annotations

import argparse
import json
import os
import stat
from collections import Counter
from contextlib import suppress
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path, PurePosixPath
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Literal

from common.specdec.q30t_tree_digest import (
    canonical_tree_sha256,
    reconcile_q30t_model_asset,
    require_stable_absolute_tree_root,
)
from common.specdec.qwen4b_b_atomic import (
    Q30_DIRECTORY_COMPLETION_MARKER,
    Task10PublicationError,
    atomic_publish_directory,
    authenticate_directory_completion,
)

if TYPE_CHECKING:
    from collections.abc import Mapping

__all__ = [
    "APPROVED_Q30T_PARENT_RECEIPT_FILE_SHA256S",
    "Q30T_PARENT_IDENTITIES",
    "ParentIdentity",
    "Q30TParentReceipt",
    "build_q30t_parent_receipt",
    "load_q30t_parent_receipt",
    "require_matching_historical_lineage",
]

Method = Literal["DFlash", "DSpark"]
Q30T_PARENT_RECEIPT_SCHEMA = "qwen3-30ba3b-thinking-parent-receipt-v1"
Q30T_TARGET_REPOSITORY = "Qwen/Qwen3-30B-A3B-Thinking-2507"
Q30T_VARIANT = "Thinking-2507"
Q30T_BLOCK_SIZE = 8
Q30T_GLOBAL_STEP = 25_391
Q30T_HISTORICAL_OCCURRENCES = 1_300_000
APPROVED_Q30T_HISTORICAL_AUDIT_FILE_SHA256 = (
    "2469430c144d9b86850901df0a28cb810b437555ee894771b796b1386d1c18b5"
)
APPROVED_Q30T_HISTORICAL_ORDERED_OCCURRENCES_SHA256 = (
    "863470b22925d74228d31b1c2433d9461d25e8d02a298cb3a08a6bc99be55060"
)
Q30T_HISTORICAL_SPLIT_ROWS: Mapping[str, int] = MappingProxyType(
    {
        "chat": 627_720,
        "code": 175_000,
        "math": 239_467,
        "multilingual_de": 257_813,
    }
)
_READ_BLOCK_BYTES = 8 * 1024 * 1024
_MAX_JSON_BYTES = 64 * 1024 * 1024
_MAX_AUDIT_JSON_BYTES = 512 * 1024 * 1024
_MONOLITHIC_WEIGHT_NAMES = frozenset({"model.safetensors", "pytorch_model.bin"})
_WEIGHT_INDEX_NAMES = frozenset({"model.safetensors.index.json", "pytorch_model.bin.index.json"})


@dataclass(frozen=True)
class ParentIdentity:
    """One exact completed Ptyche parent selected by the approved design."""

    method: Method
    completion_job_id: int
    node_count: int
    parent_run_identity: str
    checkpoint_path: Path


@dataclass(frozen=True)
class Q30TParentReceipt:
    """Authenticated parent fields consumed by the Q30 continuation contract."""

    method: Method
    target_repository: str
    target_revision: str
    target_tree_sha256: str
    variant: Literal["Thinking-2507"]
    block_size: Literal[8]
    global_step: Literal[25391]
    checkpoint_path: Path
    checkpoint_tree_sha256: str
    modelopt_state_sha256: str
    historical_receipt_file_sha256: str
    historical_ordered_prompt_uuids_sha256: str
    source_commit: str
    runtime_sha256: str
    receipt_file_sha256: str


_PARENT_ROOT = Path(
    "/lustre/fsw/coreai_dlalgo_llm/users/sna/modelopt-qwen3-drafter-training/training"
)
Q30T_PARENT_IDENTITIES: Mapping[str, ParentIdentity] = MappingProxyType(
    {
        "DFlash": ParentIdentity(
            method="DFlash",
            completion_job_id=2638009,
            node_count=16,
            parent_run_identity="q30t-nemo-dflash-b8-16n-922609729",
            checkpoint_path=(
                _PARENT_ROOT
                / "q30t-nemo-dflash-b8-16n-922609729"
                / "milestones/step-025391/resume-checkpoint-025391"
            ),
        ),
        "DSpark": ParentIdentity(
            method="DSpark",
            completion_job_id=2638015,
            node_count=16,
            parent_run_identity="q30t-nemo-dspark-b8-16n-922609729",
            checkpoint_path=(
                _PARENT_ROOT
                / "q30t-nemo-dspark-b8-16n-922609729"
                / "milestones/step-025391/resume-checkpoint-025391"
            ),
        ),
    }
)

APPROVED_Q30T_PARENT_RECEIPT_FILE_SHA256S: frozenset[str] = frozenset(
    {
        "393a2b7c5cbe2037914cbfa531d1d6fdce2bcac2f4204aa001db89d345c3a500",
        "d5416b8fd9644802b42f04511791418bf00dd023f8c5df96593010be95ff2571",
    }
)


def _canonical_json(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def _is_lower_hex(value: object, length: int) -> bool:
    return (
        type(value) is str
        and len(value) == length
        and all(character in "0123456789abcdef" for character in value)
    )


def _identity(status: os.stat_result) -> tuple[int, int, int, int, int]:
    return status.st_dev, status.st_ino, status.st_size, status.st_mtime_ns, status.st_ctime_ns


def _read_regular_at(
    directory_fd: int,
    name: str,
    display_path: str,
    expected: os.stat_result,
    *,
    retain: bool,
    require_single_link: bool,
    max_retained_bytes: int = _MAX_JSON_BYTES,
) -> tuple[int, str, bytes | None]:
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0)
    try:
        descriptor = os.open(name, flags, dir_fd=directory_fd)
    except OSError as error:
        raise ValueError(f"Q30 parent input is unreadable: {display_path}") from error
    try:
        stream = os.fdopen(descriptor, "rb")
    except BaseException as error:
        with suppress(OSError):
            os.close(descriptor)
        if isinstance(error, OSError):
            raise ValueError(f"Q30 parent input cannot be read: {display_path}") from error
        raise
    with stream:
        before = os.fstat(stream.fileno())
        if (
            _identity(expected) != _identity(before)
            or not stat.S_ISREG(before.st_mode)
            or (require_single_link and before.st_nlink != 1)
        ):
            raise ValueError(f"Q30 parent input is not a permitted regular file: {display_path}")
        digest = sha256()
        size = 0
        retained = bytearray() if retain else None
        while block := stream.read(_READ_BLOCK_BYTES):
            digest.update(block)
            size += len(block)
            if retained is not None:
                retained.extend(block)
                if len(retained) > max_retained_bytes:
                    raise ValueError(f"Q30 parent JSON is too large: {display_path}")
        after = os.fstat(stream.fileno())
    if _identity(before) != _identity(after) or size != before.st_size:
        raise ValueError(f"Q30 parent input changed while reading: {display_path}")
    return size, digest.hexdigest(), bytes(retained) if retained is not None else None


def _read_stable_file(
    path: Path, *, max_retained_bytes: int = _MAX_JSON_BYTES
) -> tuple[str, bytes]:
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0)
    try:
        parent_fd = os.open(path.parent, flags | getattr(os, "O_DIRECTORY", 0))
    except OSError as error:
        raise ValueError(f"Q30 parent file directory is unreadable: {path.parent}") from error
    try:
        expected = os.stat(path.name, dir_fd=parent_fd, follow_symlinks=False)
        _, digest, raw = _read_regular_at(
            parent_fd,
            path.name,
            str(path),
            expected,
            retain=True,
            require_single_link=True,
            max_retained_bytes=max_retained_bytes,
        )
    except OSError as error:
        raise ValueError(f"Q30 parent file is unreadable: {path}") from error
    finally:
        os.close(parent_fd)
    assert raw is not None
    return digest, raw


def _hash_stable_file(path: Path) -> str:
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0)
    try:
        parent_fd = os.open(path.parent, flags | getattr(os, "O_DIRECTORY", 0))
    except OSError as error:
        raise ValueError(f"Q30 parent file directory is unreadable: {path.parent}") from error
    try:
        expected = os.stat(path.name, dir_fd=parent_fd, follow_symlinks=False)
        _, digest, _ = _read_regular_at(
            parent_fd,
            path.name,
            str(path),
            expected,
            retain=False,
            require_single_link=True,
        )
    except OSError as error:
        raise ValueError(f"Q30 parent file is unreadable: {path}") from error
    finally:
        os.close(parent_fd)
    return digest


def _walk_checkpoint(
    root: Path, *, require_single_link: bool = False
) -> tuple[list[dict[str, object]], dict[str, bytes]]:
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        expected_root = os.stat(root, follow_symlinks=False)
        if not stat.S_ISDIR(expected_root.st_mode):
            raise ValueError("Q30 parent checkpoint is not a directory")
        root_fd = os.open(root, flags)
    except OSError as error:
        raise ValueError(f"Q30 parent checkpoint is unreadable: {root}") from error
    root_opened = os.fstat(root_fd)
    if _identity(expected_root) != _identity(root_opened):
        os.close(root_fd)
        raise ValueError("Q30 parent checkpoint changed while opening")
    files: list[dict[str, object]] = []
    retained: dict[str, bytes] = {}

    def walk(directory_fd: int, relative: str) -> None:
        before = os.fstat(directory_fd)
        if not stat.S_ISDIR(before.st_mode):
            raise ValueError("Q30 parent checkpoint directory is invalid")
        with os.scandir(directory_fd) as scan:
            names = sorted(entry.name for entry in scan)
        for name in names:
            path = f"{relative}/{name}" if relative else name
            try:
                expected = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
            except OSError as error:
                raise ValueError(f"Q30 parent checkpoint entry is unreadable: {path}") from error
            if stat.S_ISREG(expected.st_mode):
                size, digest, raw = _read_regular_at(
                    directory_fd,
                    name,
                    path,
                    expected,
                    retain=path == "trainer_state.json" or path in _WEIGHT_INDEX_NAMES,
                    require_single_link=require_single_link,
                )
                files.append({"path": path, "type": "regular", "size": size, "sha256": digest})
                if raw is not None:
                    retained[path] = raw
                continue
            if not stat.S_ISDIR(expected.st_mode):
                raise ValueError(f"Q30 parent has unsupported checkpoint entry: {path}")
            try:
                child_fd = os.open(name, flags, dir_fd=directory_fd)
            except OSError as error:
                raise ValueError(
                    f"Q30 parent checkpoint directory is unreadable: {path}"
                ) from error
            try:
                if _identity(expected) != _identity(os.fstat(child_fd)):
                    raise ValueError(
                        f"Q30 parent checkpoint directory changed while opening: {path}"
                    )
                walk(child_fd, path)
            finally:
                os.close(child_fd)
        if _identity(before) != _identity(os.fstat(directory_fd)):
            raise ValueError("Q30 parent checkpoint changed while walking")

    try:
        walk(root_fd, "")
    finally:
        os.close(root_fd)
    require_stable_absolute_tree_root(root, root_opened)
    if not files:
        raise ValueError("Q30 parent checkpoint is empty")
    files.sort(key=lambda entry: str(entry["path"]))
    return files, retained


def _json_object(raw: bytes, label: str) -> dict[str, Any]:
    try:
        payload: Any = json.loads(raw)
    except json.JSONDecodeError as error:
        raise ValueError(f"Q30 parent {label} JSON is invalid") from error
    if not isinstance(payload, dict):
        raise ValueError(f"Q30 parent {label} JSON is not an object")
    return payload


def _json_object_without_duplicate_keys(raw: bytes, label: str) -> dict[str, Any]:
    duplicate_key = False

    def object_from_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        nonlocal duplicate_key
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                duplicate_key = True
            result[key] = value
        return result

    try:
        payload: Any = json.loads(raw, object_pairs_hook=object_from_pairs)
    except json.JSONDecodeError as error:
        raise ValueError(f"Q30 parent {label} JSON is invalid") from error
    if duplicate_key or not isinstance(payload, dict):
        raise ValueError(f"Q30 parent {label} JSON is not an exact object")
    return payload


def _has_usable_weights(files: list[dict[str, object]], retained: dict[str, bytes]) -> bool:
    by_path = {entry["path"]: entry for entry in files}
    for name in _MONOLITHIC_WEIGHT_NAMES:
        monolithic = by_path.get(name)
        if type(monolithic) is dict:
            monolithic_size = monolithic.get("size")
            if type(monolithic_size) is int and monolithic_size > 0:
                return True
    indexes = [name for name in _WEIGHT_INDEX_NAMES if name in by_path]
    if len(indexes) != 1:
        return False
    index_name = indexes[0]
    index_entry = by_path[index_name]
    index_size = index_entry.get("size")
    if type(index_size) is not int or index_size < 1:
        return False
    try:
        index = _json_object_without_duplicate_keys(retained[index_name], "weight index")
    except (KeyError, ValueError):
        return False
    weight_map = index.get("weight_map")
    if not isinstance(weight_map, dict) or not weight_map:
        return False
    suffix = ".safetensors" if index_name == "model.safetensors.index.json" else ".bin"
    for parameter_name, reference in weight_map.items():
        if type(parameter_name) is not str or not parameter_name or type(reference) is not str:
            return False
        path = PurePosixPath(reference)
        if (
            not reference
            or "\\" in reference
            or path.is_absolute()
            or any(part in {"", ".", ".."} for part in reference.split("/"))
            or path.as_posix() != reference
            or not reference.endswith(suffix)
        ):
            return False
        shard = by_path.get(reference)
        if type(shard) is not dict or shard.get("type") != "regular":
            return False
        shard_size = shard.get("size")
        if type(shard_size) is not int or shard_size < 1:
            return False
    return True


def _checkpoint_evidence(
    checkpoint: Path,
) -> tuple[list[dict[str, object]], str, str, dict[str, bytes]]:
    files, retained = _walk_checkpoint(checkpoint)
    by_path = {entry["path"]: entry for entry in files}
    modelopt = by_path.get("modelopt_state.pth")
    modelopt_size = modelopt.get("size") if modelopt is not None else None
    if (
        not _has_usable_weights(files, retained)
        or modelopt is None
        or type(modelopt_size) is not int
        or modelopt_size < 1
    ):
        raise ValueError("Q30 parent checkpoint must bind weights and ModelOpt state")
    try:
        trainer_state = _json_object(retained["trainer_state.json"], "trainer state")
    except KeyError as error:
        raise ValueError("Q30 parent checkpoint lacks trainer state") from error
    if type(trainer_state.get("global_step")) is not int or trainer_state["global_step"] != 25_391:
        raise ValueError("Q30 Thinking parent identity does not match step 25391")
    tree_sha256 = canonical_tree_sha256(files)
    modelopt_sha256 = modelopt["sha256"]
    assert isinstance(modelopt_sha256, str)
    return files, tree_sha256, modelopt_sha256, retained


def _manifest_evidence(checkpoint: Path, files: list[dict[str, object]]) -> tuple[Path, str, bytes]:
    manifest_path = checkpoint.parent / "manifest.json"
    manifest_file_sha256, raw = _read_stable_file(manifest_path)
    manifest = _json_object(raw, "milestone manifest")
    expected_keys = {
        "exact_model_path",
        "exact_model_step",
        "exact_model_sha256",
        "resume_checkpoint_path",
        "resume_checkpoint_step",
        "resume_checkpoint_sha256",
        "resume_checkpoint_storage",
    }
    file_hashes = {entry["path"]: entry["sha256"] for entry in files}
    exact_model_hashes = manifest.get("exact_model_sha256")
    storage = manifest.get("resume_checkpoint_storage")
    if (
        set(manifest) != expected_keys
        or type(manifest.get("exact_model_path")) is not str
        or not manifest["exact_model_path"]
        or type(manifest.get("exact_model_step")) is not int
        or manifest["exact_model_step"] != Q30T_GLOBAL_STEP
        or not isinstance(exact_model_hashes, dict)
        or not exact_model_hashes
        or any(
            type(name) is not str or not _is_lower_hex(value, 64)
            for name, value in exact_model_hashes.items()
        )
        or manifest.get("resume_checkpoint_path") != checkpoint.name
        or type(manifest.get("resume_checkpoint_step")) is not int
        or manifest["resume_checkpoint_step"] != Q30T_GLOBAL_STEP
        or manifest.get("resume_checkpoint_sha256") != file_hashes
        or not isinstance(storage, dict)
        or set(storage) != set(file_hashes)
        or any(value not in {"copy", "hardlink"} for value in storage.values())
    ):
        raise ValueError("Q30 parent milestone manifest does not reconcile")
    return manifest_path, manifest_file_sha256, raw


def _completion_evidence(identity: ParentIdentity) -> tuple[Path, str]:
    run_root = identity.checkpoint_path.parents[2]
    completion_path = run_root / "control/training-complete-s25391.json"
    try:
        completion_file_sha256, raw = _read_stable_file(completion_path)
    except ValueError as error:
        raise ValueError("Q30 parent completion evidence is missing or unreadable") from error
    completion = _json_object(raw, "completion evidence")
    if (
        set(completion)
        != {"checkpoint", "global_step", "job_id", "restart_count", "status", "target_step"}
        or completion.get("checkpoint") != str(run_root)
        or type(completion.get("global_step")) is not int
        or completion["global_step"] != Q30T_GLOBAL_STEP
        or completion.get("job_id") != str(identity.completion_job_id)
        or type(completion.get("restart_count")) is not int
        or completion["restart_count"] < 0
        or completion.get("status") != "completed"
        or type(completion.get("target_step")) is not int
        or completion["target_step"] != Q30T_GLOBAL_STEP
    ):
        raise ValueError("Q30 parent completion evidence does not reconcile")
    return completion_path, completion_file_sha256


def _selected_identity(method: object, checkpoint: object) -> ParentIdentity:
    if type(method) is not str or method not in Q30T_PARENT_IDENTITIES:
        raise ValueError("Q30 Thinking parent identity is invalid")
    identity = Q30T_PARENT_IDENTITIES[method]
    if type(checkpoint) is not str or Path(checkpoint) != identity.checkpoint_path:
        raise ValueError("Q30 Thinking parent identity is invalid")
    return identity


def build_q30t_parent_receipt(
    checkpoint: Path,
    method: Method,
    *,
    target_revision: str,
    target_tree_sha256: str,
    historical_receipt_file_sha256: str,
    historical_ordered_prompt_uuids_sha256: str,
    source_commit: str,
    runtime_sha256: str,
) -> bytes:
    """Build canonical bytes for one exact completed Q30 Thinking parent."""
    identity = _selected_identity(method, str(checkpoint))
    if (
        not _is_lower_hex(target_revision, 40)
        or not _is_lower_hex(source_commit, 40)
        or any(
            not _is_lower_hex(value, 64)
            for value in (
                target_tree_sha256,
                historical_receipt_file_sha256,
                historical_ordered_prompt_uuids_sha256,
                runtime_sha256,
            )
        )
    ):
        raise ValueError("Q30 Thinking parent lineage identity is invalid")
    files, tree_sha256, modelopt_sha256, _ = _checkpoint_evidence(checkpoint)
    manifest_path, manifest_file_sha256, _ = _manifest_evidence(checkpoint, files)
    completion_path, completion_file_sha256 = _completion_evidence(identity)
    body: dict[str, object] = {
        "schema_version": Q30T_PARENT_RECEIPT_SCHEMA,
        "method": method,
        "target_repository": Q30T_TARGET_REPOSITORY,
        "target_revision": target_revision,
        "target_tree_sha256": target_tree_sha256,
        "variant": Q30T_VARIANT,
        "block_size": Q30T_BLOCK_SIZE,
        "global_step": Q30T_GLOBAL_STEP,
        "completion_job_id": identity.completion_job_id,
        "completion_state": "COMPLETED",
        "completion_receipt_path": str(completion_path),
        "completion_receipt_file_sha256": completion_file_sha256,
        "node_count": identity.node_count,
        "parent_run_identity": identity.parent_run_identity,
        "checkpoint_path": str(checkpoint),
        "checkpoint_files": files,
        "checkpoint_tree_sha256": tree_sha256,
        "modelopt_state_sha256": modelopt_sha256,
        "milestone_manifest_path": str(manifest_path),
        "milestone_manifest_file_sha256": manifest_file_sha256,
        "historical_occurrence_count": Q30T_HISTORICAL_OCCURRENCES,
        "historical_receipt_file_sha256": historical_receipt_file_sha256,
        "historical_ordered_prompt_uuids_sha256": historical_ordered_prompt_uuids_sha256,
        "source_commit": source_commit,
        "runtime_sha256": runtime_sha256,
    }
    return (
        _canonical_json(body | {"receipt_sha256": sha256(_canonical_json(body)).hexdigest()})
        + b"\n"
    )


def _validate_checkpoint_files(value: object) -> list[dict[str, object]]:
    if not isinstance(value, list) or not value:
        raise ValueError("Q30 parent checkpoint evidence is invalid")
    files: list[dict[str, object]] = []
    previous = ""
    for item in value:
        if (
            not isinstance(item, dict)
            or set(item) != {"path", "type", "size", "sha256"}
            or type(item["path"]) is not str
            or not item["path"]
            or item["path"] <= previous
            or item["type"] != "regular"
            or type(item["size"]) is not int
            or item["size"] < 0
            or not _is_lower_hex(item["sha256"], 64)
        ):
            raise ValueError("Q30 parent checkpoint evidence is invalid")
        path = Path(item["path"])
        if path.is_absolute() or ".." in path.parts:
            raise ValueError("Q30 parent checkpoint evidence is invalid")
        files.append(item)
        previous = item["path"]
    return files


def load_q30t_parent_receipt(path: Path, expected_sha256: str) -> Q30TParentReceipt:
    """Load one reviewed parent receipt and independently recompute its local evidence."""
    if not _is_lower_hex(expected_sha256, 64):
        raise ValueError("Q30 parent receipt caller SHA-256 is invalid")
    file_sha256, raw = _read_stable_file(path)
    if file_sha256 != expected_sha256:
        raise ValueError("Q30 parent receipt caller SHA-256 mismatch")
    payload = _json_object(raw, "receipt")
    expected_keys = {
        "schema_version",
        "method",
        "target_repository",
        "target_revision",
        "target_tree_sha256",
        "variant",
        "block_size",
        "global_step",
        "completion_job_id",
        "completion_state",
        "completion_receipt_path",
        "completion_receipt_file_sha256",
        "node_count",
        "parent_run_identity",
        "checkpoint_path",
        "checkpoint_files",
        "checkpoint_tree_sha256",
        "modelopt_state_sha256",
        "milestone_manifest_path",
        "milestone_manifest_file_sha256",
        "historical_occurrence_count",
        "historical_receipt_file_sha256",
        "historical_ordered_prompt_uuids_sha256",
        "source_commit",
        "runtime_sha256",
        "receipt_sha256",
    }
    if set(payload) != expected_keys or raw != _canonical_json(payload) + b"\n":
        raise ValueError("Q30 Thinking parent identity is invalid")
    identity = _selected_identity(payload.get("method"), payload.get("checkpoint_path"))
    expected_manifest_path = identity.checkpoint_path.parent / "manifest.json"
    expected_completion_path = (
        identity.checkpoint_path.parents[2] / "control/training-complete-s25391.json"
    )
    if (
        payload.get("schema_version") != Q30T_PARENT_RECEIPT_SCHEMA
        or payload.get("target_repository") != Q30T_TARGET_REPOSITORY
        or payload.get("variant") != Q30T_VARIANT
        or type(payload.get("block_size")) is not int
        or payload["block_size"] != Q30T_BLOCK_SIZE
        or type(payload.get("global_step")) is not int
        or payload["global_step"] != Q30T_GLOBAL_STEP
        or type(payload.get("completion_job_id")) is not int
        or payload["completion_job_id"] != identity.completion_job_id
        or payload.get("completion_state") != "COMPLETED"
        or payload.get("completion_receipt_path") != str(expected_completion_path)
        or type(payload.get("node_count")) is not int
        or payload["node_count"] != identity.node_count
        or payload.get("parent_run_identity") != identity.parent_run_identity
        or payload.get("milestone_manifest_path") != str(expected_manifest_path)
        or type(payload.get("historical_occurrence_count")) is not int
        or payload["historical_occurrence_count"] != Q30T_HISTORICAL_OCCURRENCES
        or not _is_lower_hex(payload.get("target_revision"), 40)
        or not _is_lower_hex(payload.get("source_commit"), 40)
        or any(
            not _is_lower_hex(payload.get(name), 64)
            for name in (
                "target_tree_sha256",
                "completion_receipt_file_sha256",
                "checkpoint_tree_sha256",
                "modelopt_state_sha256",
                "milestone_manifest_file_sha256",
                "historical_receipt_file_sha256",
                "historical_ordered_prompt_uuids_sha256",
                "runtime_sha256",
                "receipt_sha256",
            )
        )
    ):
        raise ValueError("Q30 Thinking parent identity is invalid")
    claimed_files = _validate_checkpoint_files(payload["checkpoint_files"])
    body = {name: value for name, value in payload.items() if name != "receipt_sha256"}
    if payload["receipt_sha256"] != sha256(_canonical_json(body)).hexdigest():
        raise ValueError("Q30 parent receipt hash does not reconcile")
    if expected_sha256 not in APPROVED_Q30T_PARENT_RECEIPT_FILE_SHA256S:
        raise ValueError("Q30 parent is not a reviewed Q30 Thinking parent receipt")
    files, tree_sha256, modelopt_sha256, _ = _checkpoint_evidence(identity.checkpoint_path)
    if (
        files != claimed_files
        or tree_sha256 != payload["checkpoint_tree_sha256"]
        or modelopt_sha256 != payload["modelopt_state_sha256"]
    ):
        raise ValueError("Q30 parent checkpoint evidence does not reconcile")
    manifest_path, manifest_file_sha256, _ = _manifest_evidence(identity.checkpoint_path, files)
    if (
        manifest_path != expected_manifest_path
        or manifest_file_sha256 != payload["milestone_manifest_file_sha256"]
    ):
        raise ValueError("Q30 parent milestone manifest evidence does not reconcile")
    completion_path, completion_file_sha256 = _completion_evidence(identity)
    if (
        completion_path != expected_completion_path
        or completion_file_sha256 != payload["completion_receipt_file_sha256"]
    ):
        raise ValueError("Q30 parent completion evidence does not reconcile")
    method: Method = identity.method
    return Q30TParentReceipt(
        method=method,
        target_repository=payload["target_repository"],
        target_revision=payload["target_revision"],
        target_tree_sha256=payload["target_tree_sha256"],
        variant="Thinking-2507",
        block_size=8,
        global_step=25391,
        checkpoint_path=identity.checkpoint_path,
        checkpoint_tree_sha256=payload["checkpoint_tree_sha256"],
        modelopt_state_sha256=payload["modelopt_state_sha256"],
        historical_receipt_file_sha256=payload["historical_receipt_file_sha256"],
        historical_ordered_prompt_uuids_sha256=payload["historical_ordered_prompt_uuids_sha256"],
        source_commit=payload["source_commit"],
        runtime_sha256=payload["runtime_sha256"],
        receipt_file_sha256=expected_sha256,
    )


def require_matching_historical_lineage(
    dflash: Q30TParentReceipt, dspark: Q30TParentReceipt
) -> None:
    """Require one parent per method with the identical authenticated 1.3M ordering."""
    if dflash.method != "DFlash" or dspark.method != "DSpark":
        raise ValueError("historical lineage comparison requires DFlash and DSpark parents")
    if (
        dflash.historical_receipt_file_sha256 != dspark.historical_receipt_file_sha256
        or dflash.historical_ordered_prompt_uuids_sha256
        != dspark.historical_ordered_prompt_uuids_sha256
    ):
        raise ValueError("DFlash and DSpark historical lineage does not match")


def _historical_audit_identity(path: Path) -> tuple[str, str]:
    try:
        file_sha256, raw = _read_stable_file(path, max_retained_bytes=_MAX_AUDIT_JSON_BYTES)
        payload = _json_object(raw, "historical AUDIT")
        expected_keys = {
            "source_revision",
            "source_manifest_sha256",
            "row_count",
            "split_rows",
            "files",
            "occurrence_count",
            "unique_prompt_count",
            "occurrence_prompt_ids",
            "occurrence_prompt_ids_sha256",
            "exclusion_prompt_ids",
            "exclusion_prompt_ids_sha256",
            "duplicate_uuid_multiplicity",
            "physical_row_count",
            "selection_policy",
            "selection_boundary",
            "receipt_sha256",
        }
        if set(payload) != expected_keys or raw != _canonical_json(payload) + b"\n":
            raise ValueError("historical AUDIT is not canonical")
        occurrences = payload["occurrence_prompt_ids"]
        exclusions = payload["exclusion_prompt_ids"]
        split_rows = payload["split_rows"]
        files = payload["files"]
        multiplicity = payload["duplicate_uuid_multiplicity"]
        boundary = payload["selection_boundary"]
        if (
            payload["source_revision"] != "5c89e01dd720ae0f4058445ed49c5fb68a03c76e"
            or not _is_lower_hex(payload["source_manifest_sha256"], 64)
            or type(payload["row_count"]) is not int
            or payload["row_count"] != Q30T_HISTORICAL_OCCURRENCES
            or type(payload["occurrence_count"]) is not int
            or payload["occurrence_count"] != Q30T_HISTORICAL_OCCURRENCES
            or not isinstance(split_rows, dict)
            or split_rows != Q30T_HISTORICAL_SPLIT_ROWS
            or any(
                type(name) is not str or not name or type(count) is not int or count < 0
                for name, count in split_rows.items()
            )
            or sum(split_rows.values()) != Q30T_HISTORICAL_OCCURRENCES
            or not isinstance(files, list)
            or not files
            or type(payload["physical_row_count"]) is not int
            or payload["physical_row_count"] < Q30T_HISTORICAL_OCCURRENCES
            or payload["selection_policy"] != "hf-streaming-sorted-parquet-take"
            or not isinstance(boundary, dict)
            or set(boundary) != {"file", "rows_selected", "rows_available", "excluded_tail_rows"}
        ):
            raise ValueError("historical AUDIT identity is invalid")
        for record in files:
            if (
                not isinstance(record, dict)
                or set(record) != {"path", "bytes", "sha256"}
                or type(record["path"]) is not str
                or not Path(record["path"]).is_absolute()
                or type(record["bytes"]) is not int
                or record["bytes"] < 0
                or not _is_lower_hex(record["sha256"], 64)
            ):
                raise ValueError("historical AUDIT file identity is invalid")
        if (
            type(boundary["file"]) is not str
            or not boundary["file"]
            or any(
                type(boundary[name]) is not int or boundary[name] < 0
                for name in ("rows_selected", "rows_available", "excluded_tail_rows")
            )
            or boundary["rows_selected"] + boundary["excluded_tail_rows"]
            != boundary["rows_available"]
        ):
            raise ValueError("historical AUDIT selection boundary is invalid")
        if (
            not isinstance(occurrences, list)
            or len(occurrences) != Q30T_HISTORICAL_OCCURRENCES
            or any(not _is_lower_hex(value, 64) for value in occurrences)
            or not isinstance(exclusions, list)
            or any(not _is_lower_hex(value, 64) for value in exclusions)
            or exclusions != sorted(set(occurrences))
            or type(payload["unique_prompt_count"]) is not int
            or payload["unique_prompt_count"] != len(exclusions)
            or not isinstance(multiplicity, dict)
        ):
            raise ValueError("historical AUDIT occurrence identity is invalid")
        counts = Counter(occurrences)
        expected_multiplicity = {
            prompt_uuid: count for prompt_uuid, count in sorted(counts.items()) if count > 1
        }
        body = {name: value for name, value in payload.items() if name != "receipt_sha256"}
        ordered_sha256 = sha256(_canonical_json(occurrences)).hexdigest()
        if (
            multiplicity != expected_multiplicity
            or payload["occurrence_prompt_ids_sha256"] != ordered_sha256
            or payload["exclusion_prompt_ids_sha256"]
            != sha256(_canonical_json(exclusions)).hexdigest()
            or payload["receipt_sha256"] != sha256(_canonical_json(body)).hexdigest()
        ):
            raise ValueError("historical AUDIT hashes do not reconcile")
    except (KeyError, TypeError, ValueError) as error:
        if isinstance(error, ValueError) and "historical AUDIT" in str(error):
            raise
        raise ValueError("historical AUDIT is invalid") from error
    if (
        file_sha256 != APPROVED_Q30T_HISTORICAL_AUDIT_FILE_SHA256
        or ordered_sha256 != APPROVED_Q30T_HISTORICAL_ORDERED_OCCURRENCES_SHA256
    ):
        raise ValueError("Q30 parent does not use the reviewed historical AUDIT ordering")
    return file_sha256, ordered_sha256


def _tree_sha256(root: Path, label: str) -> str:
    try:
        files, _ = _walk_checkpoint(root)
    except ValueError as error:
        raise ValueError(f"Q30 {label} tree is invalid") from error
    return canonical_tree_sha256(files)


def _publish_receipt_pair(
    output_root: Path, receipts: Mapping[str, bytes]
) -> dict[str, dict[str, str]]:
    filenames = {method: f"q30t-{method.lower()}-parent.json" for method in receipts}
    expected_names = frozenset({*filenames.values(), Q30_DIRECTORY_COMPLETION_MARKER})
    result = {
        method: {
            "path": str(output_root / filename),
            "sha256": sha256(receipts[method]).hexdigest(),
        }
        for method, filename in filenames.items()
    }

    def adopt() -> dict[str, dict[str, str]]:
        flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0)
        directory_flags = flags | getattr(os, "O_DIRECTORY", 0)
        try:
            parent_fd = os.open(output_root.parent, directory_flags)
        except OSError as error:
            raise FileExistsError(
                f"Q30 parent output parent cannot be adopted: {output_root.parent}"
            ) from error
        try:
            parent_before = os.fstat(parent_fd)
            try:
                expected_root = os.stat(output_root.name, dir_fd=parent_fd, follow_symlinks=False)
                root_fd = os.open(output_root.name, directory_flags, dir_fd=parent_fd)
            except OSError as error:
                raise FileExistsError(
                    f"Q30 parent output root cannot be adopted: {output_root}"
                ) from error
            try:
                root_before = os.fstat(root_fd)
                if _identity(expected_root) != _identity(root_before) or not stat.S_ISDIR(
                    root_before.st_mode
                ):
                    raise FileExistsError(
                        f"Q30 parent output root changed while opening: {output_root}"
                    )
                if frozenset(os.listdir(root_fd)) != expected_names:
                    raise FileExistsError(f"Q30 parent output root is incomplete: {output_root}")
                try:
                    marker_identity = authenticate_directory_completion(root_fd)
                except (OSError, Task10PublicationError) as error:
                    raise FileExistsError(
                        f"Q30 parent output completion marker is invalid: {output_root}"
                    ) from error
                receipt_identities: dict[str, tuple[int, int, int, int, int]] = {}
                for method, filename in filenames.items():
                    expected_file = os.stat(filename, dir_fd=root_fd, follow_symlinks=False)
                    try:
                        receipt_fd = os.open(filename, flags, dir_fd=root_fd)
                    except OSError as error:
                        raise FileExistsError(
                            f"Q30 parent output root has unreadable receipt: {filename}"
                        ) from error
                    try:
                        before = os.fstat(receipt_fd)
                        if (
                            _identity(expected_file) != _identity(before)
                            or not stat.S_ISREG(before.st_mode)
                            or before.st_nlink != 1
                        ):
                            raise FileExistsError(
                                f"Q30 parent output receipt changed while opening: {filename}"
                            )
                        receipt_identities[filename] = _identity(before)
                        retained = bytearray()
                        with os.fdopen(receipt_fd, "rb", closefd=False) as stream:
                            while block := stream.read(_READ_BLOCK_BYTES):
                                retained.extend(block)
                                if len(retained) > len(receipts[method]):
                                    break
                        after = os.fstat(receipt_fd)
                        rebound = os.stat(filename, dir_fd=root_fd, follow_symlinks=False)
                        if _identity(before) != _identity(after) or _identity(after) != _identity(
                            rebound
                        ):
                            raise FileExistsError(
                                f"Q30 parent output receipt rebound while adopting: {filename}"
                            )
                        if bytes(retained) != receipts[method]:
                            raise FileExistsError(f"Q30 parent output root differs: {output_root}")
                        os.fsync(receipt_fd)
                    finally:
                        os.close(receipt_fd)
                if frozenset(os.listdir(root_fd)) != expected_names:
                    raise FileExistsError(f"Q30 parent output root changed: {output_root}")
                root_after = os.fstat(root_fd)
                rebound_root = os.stat(output_root.name, dir_fd=parent_fd, follow_symlinks=False)
                if _identity(root_before) != _identity(root_after) or _identity(
                    root_after
                ) != _identity(rebound_root):
                    raise FileExistsError(
                        f"Q30 parent output root rebound while adopting: {output_root}"
                    )
                os.fsync(root_fd)
                if authenticate_directory_completion(root_fd) != marker_identity:
                    raise FileExistsError(
                        f"Q30 parent output completion marker changed: {output_root}"
                    )
            finally:
                os.close(root_fd)
            parent_after = os.fstat(parent_fd)
            if _identity(parent_before) != _identity(parent_after):
                raise FileExistsError(
                    f"Q30 parent output parent changed while adopting: {output_root.parent}"
                )
            os.fsync(parent_fd)
            try:
                final_parent = os.fstat(parent_fd)
                final_named_root = os.stat(
                    output_root.name, dir_fd=parent_fd, follow_symlinks=False
                )
                final_root_fd = os.open(output_root.name, directory_flags, dir_fd=parent_fd)
            except OSError as error:
                raise FileExistsError(
                    f"Q30 parent output root cannot be rebound after durability: {output_root}"
                ) from error
            try:
                final_opened_root = os.fstat(final_root_fd)
                absolute_root = os.stat(output_root, follow_symlinks=False)
                if (
                    _identity(final_parent) != _identity(parent_before)
                    or _identity(final_named_root) != _identity(root_before)
                    or _identity(final_opened_root) != _identity(root_before)
                    or _identity(absolute_root) != _identity(root_before)
                    or frozenset(os.listdir(final_root_fd)) != expected_names
                ):
                    raise FileExistsError(
                        f"Q30 parent output root changed or rebound after durability: {output_root}"
                    )
                for filename, expected_identity in receipt_identities.items():
                    try:
                        final_named = os.stat(filename, dir_fd=final_root_fd, follow_symlinks=False)
                        final_receipt_fd = os.open(filename, flags, dir_fd=final_root_fd)
                    except OSError as error:
                        raise FileExistsError(
                            f"Q30 parent output receipt cannot be rebound: {filename}"
                        ) from error
                    try:
                        final_opened = os.fstat(final_receipt_fd)
                        final_absolute = os.stat(output_root / filename, follow_symlinks=False)
                        if (
                            _identity(final_named) != expected_identity
                            or _identity(final_opened) != expected_identity
                            or _identity(final_absolute) != expected_identity
                            or not stat.S_ISREG(final_opened.st_mode)
                            or final_opened.st_nlink != 1
                        ):
                            raise FileExistsError(
                                f"Q30 parent output receipt changed or rebound: {filename}"
                            )
                    finally:
                        os.close(final_receipt_fd)
                final_marker_identity = authenticate_directory_completion(final_root_fd)
                final_absolute_marker = os.stat(
                    output_root / Q30_DIRECTORY_COMPLETION_MARKER,
                    follow_symlinks=False,
                )
                if (
                    final_marker_identity != marker_identity
                    or _identity(final_absolute_marker) != marker_identity
                    or final_absolute_marker.st_nlink != 1
                ):
                    raise FileExistsError(
                        f"Q30 parent output completion marker rebound: {output_root}"
                    )
                if frozenset(os.listdir(final_root_fd)) != expected_names:
                    raise FileExistsError(
                        f"Q30 parent output root changed after final enumeration: {output_root}"
                    )
                last_parent = os.fstat(parent_fd)
                last_named_root = os.stat(output_root.name, dir_fd=parent_fd, follow_symlinks=False)
                last_absolute_root = os.stat(output_root, follow_symlinks=False)
                if (
                    _identity(last_parent) != _identity(parent_before)
                    or _identity(last_named_root) != _identity(root_before)
                    or _identity(last_absolute_root) != _identity(root_before)
                ):
                    raise FileExistsError(
                        f"Q30 parent output root rebound after final enumeration: {output_root}"
                    )
                for filename, expected_identity in receipt_identities.items():
                    last_named = os.stat(filename, dir_fd=final_root_fd, follow_symlinks=False)
                    last_absolute = os.stat(output_root / filename, follow_symlinks=False)
                    if (
                        _identity(last_named) != expected_identity
                        or _identity(last_absolute) != expected_identity
                        or last_named.st_nlink != 1
                        or last_absolute.st_nlink != 1
                    ):
                        raise FileExistsError(
                            f"Q30 parent output receipt rebound after final enumeration: {filename}"
                        )
                if authenticate_directory_completion(final_root_fd) != marker_identity:
                    raise FileExistsError(
                        f"Q30 parent output completion marker changed after enumeration: {output_root}"
                    )
            finally:
                os.close(final_root_fd)
            return result
        finally:
            os.close(parent_fd)

    if os.path.lexists(output_root):
        return adopt()
    try:
        output_root.parent.mkdir(mode=0o750, parents=True, exist_ok=True)
    except OSError as error:
        raise ValueError(
            f"Q30 parent output parent cannot be created: {output_root.parent}"
        ) from error
    job_id = os.environ.get("SLURM_JOB_ID", str(os.getpid()))
    partial = output_root.with_name(f".{output_root.name}.partial-{job_id}-{os.getpid()}")
    partial.mkdir(mode=0o750)
    for method, raw in receipts.items():
        destination = partial / filenames[method]
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
        descriptor = os.open(destination, flags, 0o640)
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
    try:
        atomic_publish_directory(partial, output_root)
    except Exception:
        if not os.path.lexists(output_root):
            raise
        return adopt()
    return adopt()


def publish_q30t_parent_receipts(
    *,
    historical_audit: Path,
    target_model_root: Path,
    target_revision: str,
    target_identity_path: Path,
    target_identity_sha256: str,
    target_model_sha256_path: Path,
    target_model_sha256_sha256: str,
    dflash_parent_root: Path,
    dspark_parent_root: Path,
    runtime_archive: Path,
    source_commit: str,
    output_root: Path,
) -> dict[str, dict[str, str]]:
    """Derive, cross-bind, and publish the two production parent receipts exactly once."""
    historical_file_sha256, historical_ordered_sha256 = _historical_audit_identity(historical_audit)
    try:
        target_files, _ = _walk_checkpoint(target_model_root, require_single_link=True)
    except ValueError as error:
        raise ValueError("Q30 target model tree is invalid") from error
    target_tree_sha256 = reconcile_q30t_model_asset(
        snapshot=target_model_root,
        entries=target_files,
        repository=Q30T_TARGET_REPOSITORY,
        revision=target_revision,
        identity_path=target_identity_path,
        identity_sha256=target_identity_sha256,
        model_sha256_path=target_model_sha256_path,
        model_sha256_sha256=target_model_sha256_sha256,
    )
    runtime_sha256 = _hash_stable_file(runtime_archive)
    roots: tuple[tuple[Method, Path], ...] = (
        ("DFlash", dflash_parent_root),
        ("DSpark", dspark_parent_root),
    )
    receipts: dict[str, bytes] = {}
    for method, checkpoint in roots:
        first = build_q30t_parent_receipt(
            checkpoint=checkpoint,
            method=method,
            target_revision=target_revision,
            target_tree_sha256=target_tree_sha256,
            historical_receipt_file_sha256=historical_file_sha256,
            historical_ordered_prompt_uuids_sha256=historical_ordered_sha256,
            source_commit=source_commit,
            runtime_sha256=runtime_sha256,
        )
        second = build_q30t_parent_receipt(
            checkpoint=checkpoint,
            method=method,
            target_revision=target_revision,
            target_tree_sha256=target_tree_sha256,
            historical_receipt_file_sha256=historical_file_sha256,
            historical_ordered_prompt_uuids_sha256=historical_ordered_sha256,
            source_commit=source_commit,
            runtime_sha256=runtime_sha256,
        )
        if first != second:
            raise ValueError(f"Q30 {method} parent receipt is not byte-deterministic")
        receipts[method] = first
    payloads = {
        method: _json_object(raw, f"{method} generated receipt") for method, raw in receipts.items()
    }
    bound_names = (
        "target_repository",
        "target_revision",
        "target_tree_sha256",
        "historical_occurrence_count",
        "historical_receipt_file_sha256",
        "historical_ordered_prompt_uuids_sha256",
        "source_commit",
        "runtime_sha256",
    )
    if any(payloads["DFlash"][name] != payloads["DSpark"][name] for name in bound_names):
        raise ValueError("Q30 parent pair does not share target and historical lineage")
    return _publish_receipt_pair(output_root, receipts)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--historical-audit", type=Path, required=True)
    parser.add_argument("--target-model-root", type=Path, required=True)
    parser.add_argument("--target-revision", required=True)
    parser.add_argument("--target-identity-path", type=Path, required=True)
    parser.add_argument("--target-identity-sha256", required=True)
    parser.add_argument("--target-model-sha256-path", type=Path, required=True)
    parser.add_argument("--target-model-sha256-sha256", required=True)
    parser.add_argument("--dflash-parent-root", type=Path, required=True)
    parser.add_argument("--dspark-parent-root", type=Path, required=True)
    parser.add_argument("--runtime-archive", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args(argv)
    result = publish_q30t_parent_receipts(
        historical_audit=args.historical_audit,
        target_model_root=args.target_model_root,
        target_revision=args.target_revision,
        target_identity_path=args.target_identity_path,
        target_identity_sha256=args.target_identity_sha256,
        target_model_sha256_path=args.target_model_sha256_path,
        target_model_sha256_sha256=args.target_model_sha256_sha256,
        dflash_parent_root=args.dflash_parent_root,
        dspark_parent_root=args.dspark_parent_root,
        runtime_archive=args.runtime_archive,
        source_commit=args.source_commit,
        output_root=args.output_root,
    )
    print(json.dumps(result, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
