# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Canonical per-node and aggregate Q30 Ptyche runtime attestation."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import stat
import sys
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, cast

from common.specdec.ptv23_node_keeper import KeeperItem, KeeperReceipt
from common.specdec.q30t_runtime_archive_receipt import (
    RuntimeArchiveTreeReceipt,
    load_runtime_archive_tree_receipt,
)
from common.specdec.qwen4b_b_atomic import atomic_publish_bytes

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

__all__ = [
    "AttestationError",
    "RuntimeKeeperLossEvidence",
    "RuntimeNodeAttestationInput",
    "RuntimeNodeAttestationReceipt",
    "RuntimeNodeObservation",
    "RuntimeNodeReuseEvidence",
    "RuntimeQualificationContext",
    "RuntimeQualificationReceipt",
    "attest_runtime_node",
    "load_runtime_archive_tree_receipt",
    "load_runtime_node_receipt",
    "load_runtime_qualification_receipt",
    "main",
    "reconcile_runtime_receipts",
    "stable_single_link_bytes",
]

_HASH_LENGTH = 64
_SOURCE_COMMIT_LENGTH = 40
_MAX_METADATA_BYTES = 4 * 1024 * 1024
_MAX_TEXT_BYTES = 4096
_MAX_TOOL_BYTES = 16 * 1024 * 1024
_READ_BLOCK_BYTES = 1024 * 1024
_REQUIRED_IMPORTS = ("accelerate", "datasets", "modelopt", "wandb")
_NODE_OBSERVATION_SCHEMA = "q30t-runtime-node-observation-v1"
_NODE_REUSE_SCHEMA = "q30t-runtime-node-reuse-v1"
_KEEPER_LOSS_SCHEMA = "q30t-runtime-node-keeper-loss-v1"
_NODE_RECEIPT_SCHEMA = "q30t-runtime-node-attestation-v1"
_QUALIFICATION_SCHEMA = "q30t-runtime-qualification-v1"


class AttestationError(ValueError):
    """Runtime qualification evidence is incomplete, ambiguous, or mutable."""


def _read_hook() -> None:
    return None


_READ_HOOK: Callable[[], None] = _read_hook


def _canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _digest(value: object) -> str:
    return hashlib.sha256(_canonical_json_bytes(value)).hexdigest()


def _required_text(record: dict[str, object], key: str) -> str:
    value = record.get(key)
    if type(value) is not str:
        raise AttestationError(f"{key} must be text")
    return value


def _required_integer(record: dict[str, object], key: str) -> int:
    value = record.get(key)
    if type(value) is not int:
        raise AttestationError(f"{key} must be an integer")
    return value


def _required_true(record: dict[str, object], key: str) -> Literal[True]:
    if record.get(key) is not True:
        raise AttestationError(f"{key} must be literal true evidence")
    return True


def _optional_text(record: dict[str, object], key: str) -> str | None:
    value = record.get(key)
    if value is not None and type(value) is not str:
        raise AttestationError(f"{key} must be text or null")
    return cast("str | None", value)


def _exact_record(raw: object, expected_keys: frozenset[str], label: str) -> dict[str, object]:
    if not isinstance(raw, dict) or any(type(key) is not str for key in raw):
        raise AttestationError(f"{label} must be a JSON object")
    record = cast("dict[str, object]", raw)
    if frozenset(record) != expected_keys:
        raise AttestationError(f"{label} has an unexpected schema")
    return record


def _is_lower_hex(value: object, length: int) -> bool:
    return (
        type(value) is str
        and len(value) == length
        and all(character in "0123456789abcdef" for character in value)
    )


def _require_hash(value: object, label: str) -> None:
    if not _is_lower_hex(value, _HASH_LENGTH):
        raise AttestationError(f"{label} SHA-256 is invalid")


def _require_source_commit(value: object) -> None:
    if not _is_lower_hex(value, _SOURCE_COMMIT_LENGTH):
        raise AttestationError("source commit is invalid")


def _contains_control(value: str) -> bool:
    return any(unicodedata.category(character).startswith("C") for character in value)


def _require_bounded_text(value: str, label: str, *, allow_empty: bool = False) -> None:
    try:
        encoded = value.encode("utf-8")
    except UnicodeEncodeError as error:
        raise AttestationError(f"{label} is not valid UTF-8") from error
    if (not allow_empty and not value) or len(encoded) > _MAX_TEXT_BYTES:
        raise AttestationError(f"{label} is not bounded non-empty text")
    if _contains_control(value):
        raise AttestationError(f"{label} contains a control character")


def _require_canonical_absolute_path(value: str | Path, label: str) -> Path:
    path = Path(value)
    if (
        not path.is_absolute()
        or path == Path("/")
        or path.parts[0] != "/"
        or any(part in {".", ".."} for part in path.parts)
        or str(path) != path.as_posix()
    ):
        raise AttestationError(f"{label} path must be canonical and absolute")
    if _contains_control(str(path)):
        raise AttestationError(f"{label} path contains a control character")
    return path


def _is_within(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


@dataclass(frozen=True)
class RuntimeNodeObservation:
    """Self-hashed initial container observation for one allocated node."""

    schema_version: Literal["q30t-runtime-node-observation-v1"]
    job_id: str
    phase: Literal["one-node", "two-node"]
    cluster: Literal["ptyche"]
    profile_file_sha256: str
    node_name: str
    keeper_pid: int
    keeper_start_ticks: int
    keeper_receipt_sha256: str
    anchor_path: str
    anchor_target: str
    anchor_device: int
    anchor_inode: int
    archive_sha256: str
    archive_tree_receipt_file_sha256: str
    runtime_tree_sha256: str
    expected_image_sha256: str
    observed_image_sha256: str
    sentinel_inventory_sha256: str
    source_commit: str
    contract_file_sha256: str
    runner_sha256: str
    keeper_tool_sha256: str
    attestation_tool_sha256: str
    machine: Literal["aarch64"]
    python_executable: str
    python_version: str
    python_import_origins: tuple[tuple[str, str], ...]
    cuda_visible_devices: str
    visible_gpu_identities: tuple[str, str, str, str]
    visible_gpu_count: Literal[4]
    pyxis_version: str
    enroot_version: str
    container_name: str
    container_root_device: int
    container_root_inode: int
    container_sentinel_sha256: str
    observation_sha256: str

    def body_dict(self) -> dict[str, object]:
        """Return every self-hashed observation field."""
        return {
            "anchor_device": self.anchor_device,
            "anchor_inode": self.anchor_inode,
            "anchor_path": self.anchor_path,
            "anchor_target": self.anchor_target,
            "archive_sha256": self.archive_sha256,
            "archive_tree_receipt_file_sha256": self.archive_tree_receipt_file_sha256,
            "attestation_tool_sha256": self.attestation_tool_sha256,
            "cluster": self.cluster,
            "container_name": self.container_name,
            "container_root_device": self.container_root_device,
            "container_root_inode": self.container_root_inode,
            "container_sentinel_sha256": self.container_sentinel_sha256,
            "contract_file_sha256": self.contract_file_sha256,
            "cuda_visible_devices": self.cuda_visible_devices,
            "enroot_version": self.enroot_version,
            "expected_image_sha256": self.expected_image_sha256,
            "job_id": self.job_id,
            "keeper_pid": self.keeper_pid,
            "keeper_receipt_sha256": self.keeper_receipt_sha256,
            "keeper_start_ticks": self.keeper_start_ticks,
            "keeper_tool_sha256": self.keeper_tool_sha256,
            "machine": self.machine,
            "node_name": self.node_name,
            "observed_image_sha256": self.observed_image_sha256,
            "phase": self.phase,
            "profile_file_sha256": self.profile_file_sha256,
            "python_executable": self.python_executable,
            "python_import_origins": [list(item) for item in self.python_import_origins],
            "python_version": self.python_version,
            "pyxis_version": self.pyxis_version,
            "runner_sha256": self.runner_sha256,
            "runtime_tree_sha256": self.runtime_tree_sha256,
            "schema_version": self.schema_version,
            "sentinel_inventory_sha256": self.sentinel_inventory_sha256,
            "source_commit": self.source_commit,
            "visible_gpu_count": self.visible_gpu_count,
            "visible_gpu_identities": list(self.visible_gpu_identities),
        }

    def to_dict(self) -> dict[str, object]:
        """Return the complete JSON-compatible observation."""
        return self.body_dict() | {"observation_sha256": self.observation_sha256}

    def canonical_bytes(self) -> bytes:
        """Encode the observation in its canonical JSON form."""
        return _canonical_json_bytes(self.to_dict())

    def verify_self_hash(self) -> None:
        """Reject an observation whose body does not reproduce its hash."""
        if self.observation_sha256 != _digest(self.body_dict()):
            raise AttestationError("runtime observation self-hash mismatch")

    def validate(self) -> None:
        """Reject structurally or semantically invalid observation fields."""
        _validate_observation(self)

    @classmethod
    def from_dict(cls, raw: object) -> RuntimeNodeObservation:
        """Decode and validate one exact observation schema."""
        record = _exact_record(raw, _OBSERVATION_KEYS, "runtime observation")
        origins = _text_pairs(record.get("python_import_origins"), "python import origins")
        gpu_identities = _four_texts(record.get("visible_gpu_identities"), "GPU identities")
        observation = cls(
            schema_version=cast(
                "Literal['q30t-runtime-node-observation-v1']",
                _required_text(record, "schema_version"),
            ),
            job_id=_required_text(record, "job_id"),
            phase=cast("Literal['one-node', 'two-node']", _required_text(record, "phase")),
            cluster=cast("Literal['ptyche']", _required_text(record, "cluster")),
            profile_file_sha256=_required_text(record, "profile_file_sha256"),
            node_name=_required_text(record, "node_name"),
            keeper_pid=_required_integer(record, "keeper_pid"),
            keeper_start_ticks=_required_integer(record, "keeper_start_ticks"),
            keeper_receipt_sha256=_required_text(record, "keeper_receipt_sha256"),
            anchor_path=_required_text(record, "anchor_path"),
            anchor_target=_required_text(record, "anchor_target"),
            anchor_device=_required_integer(record, "anchor_device"),
            anchor_inode=_required_integer(record, "anchor_inode"),
            archive_sha256=_required_text(record, "archive_sha256"),
            archive_tree_receipt_file_sha256=_required_text(
                record, "archive_tree_receipt_file_sha256"
            ),
            runtime_tree_sha256=_required_text(record, "runtime_tree_sha256"),
            expected_image_sha256=_required_text(record, "expected_image_sha256"),
            observed_image_sha256=_required_text(record, "observed_image_sha256"),
            sentinel_inventory_sha256=_required_text(record, "sentinel_inventory_sha256"),
            source_commit=_required_text(record, "source_commit"),
            contract_file_sha256=_required_text(record, "contract_file_sha256"),
            runner_sha256=_required_text(record, "runner_sha256"),
            keeper_tool_sha256=_required_text(record, "keeper_tool_sha256"),
            attestation_tool_sha256=_required_text(record, "attestation_tool_sha256"),
            machine=cast("Literal['aarch64']", _required_text(record, "machine")),
            python_executable=_required_text(record, "python_executable"),
            python_version=_required_text(record, "python_version"),
            python_import_origins=origins,
            cuda_visible_devices=_required_text(record, "cuda_visible_devices"),
            visible_gpu_identities=gpu_identities,
            visible_gpu_count=cast("Literal[4]", _required_integer(record, "visible_gpu_count")),
            pyxis_version=_required_text(record, "pyxis_version"),
            enroot_version=_required_text(record, "enroot_version"),
            container_name=_required_text(record, "container_name"),
            container_root_device=_required_integer(record, "container_root_device"),
            container_root_inode=_required_integer(record, "container_root_inode"),
            container_sentinel_sha256=_required_text(record, "container_sentinel_sha256"),
            observation_sha256=_required_text(record, "observation_sha256"),
        )
        observation.validate()
        observation.verify_self_hash()
        return observation


_OBSERVATION_KEYS = frozenset(
    {
        "anchor_device",
        "anchor_inode",
        "anchor_path",
        "anchor_target",
        "archive_sha256",
        "archive_tree_receipt_file_sha256",
        "attestation_tool_sha256",
        "cluster",
        "container_name",
        "container_root_device",
        "container_root_inode",
        "container_sentinel_sha256",
        "contract_file_sha256",
        "cuda_visible_devices",
        "enroot_version",
        "expected_image_sha256",
        "job_id",
        "keeper_pid",
        "keeper_receipt_sha256",
        "keeper_start_ticks",
        "keeper_tool_sha256",
        "machine",
        "node_name",
        "observation_sha256",
        "observed_image_sha256",
        "phase",
        "profile_file_sha256",
        "python_executable",
        "python_import_origins",
        "python_version",
        "pyxis_version",
        "runner_sha256",
        "runtime_tree_sha256",
        "schema_version",
        "sentinel_inventory_sha256",
        "source_commit",
        "visible_gpu_count",
        "visible_gpu_identities",
    }
)


def _text_pairs(raw: object, label: str) -> tuple[tuple[str, str], ...]:
    if not isinstance(raw, list):
        raise AttestationError(f"{label} must be an array")
    result: list[tuple[str, str]] = []
    for item in raw:
        if (
            not isinstance(item, list)
            or len(item) != 2
            or any(type(value) is not str for value in item)
        ):
            raise AttestationError(f"{label} must contain text pairs")
        result.append((cast("str", item[0]), cast("str", item[1])))
    return tuple(result)


def _four_texts(raw: object, label: str) -> tuple[str, str, str, str]:
    if not isinstance(raw, list) or len(raw) != 4 or any(type(value) is not str for value in raw):
        raise AttestationError(f"{label} must contain exactly four text records")
    values = cast("list[str]", raw)
    return values[0], values[1], values[2], values[3]


def _validate_observation(value: RuntimeNodeObservation) -> None:
    if value.schema_version != _NODE_OBSERVATION_SCHEMA:
        raise AttestationError("runtime observation schema version is invalid")
    if value.phase not in {"one-node", "two-node"} or value.cluster != "ptyche":
        raise AttestationError("runtime observation phase or cluster is invalid")
    for text, label in (
        (value.job_id, "job identity"),
        (value.node_name, "node identity"),
        (value.container_name, "container identity"),
        (value.python_version, "Python version"),
        (value.pyxis_version, "Pyxis version"),
        (value.enroot_version, "enroot version"),
        (value.cuda_visible_devices, "CUDA visibility"),
    ):
        _require_bounded_text(text, label)
    for digest, label in (
        (value.profile_file_sha256, "profile"),
        (value.keeper_receipt_sha256, "keeper receipt"),
        (value.archive_sha256, "archive"),
        (value.archive_tree_receipt_file_sha256, "archive receipt file"),
        (value.runtime_tree_sha256, "runtime tree"),
        (value.expected_image_sha256, "expected image"),
        (value.observed_image_sha256, "observed image"),
        (value.sentinel_inventory_sha256, "sentinel inventory"),
        (value.contract_file_sha256, "contract file"),
        (value.runner_sha256, "runner"),
        (value.keeper_tool_sha256, "keeper tool"),
        (value.attestation_tool_sha256, "attestation tool"),
        (value.container_sentinel_sha256, "container sentinel"),
        (value.observation_sha256, "observation"),
    ):
        _require_hash(digest, label)
    _require_source_commit(value.source_commit)
    if value.machine != "aarch64":
        raise AttestationError("runtime observation machine must be aarch64")
    if (
        value.keeper_pid <= 0
        or value.keeper_start_ticks < 0
        or value.anchor_device <= 0
        or value.anchor_inode <= 0
        or value.container_root_device <= 0
        or value.container_root_inode <= 0
    ):
        raise AttestationError(
            "runtime observation process, anchor, or container identity is invalid"
        )
    _require_canonical_absolute_path(value.anchor_path, "anchor")
    _require_canonical_absolute_path(value.anchor_target, "anchor target")
    _require_canonical_absolute_path(value.python_executable, "Python executable")
    anchor_prefix = f"/proc/{value.keeper_pid}/fd/"
    if (
        not value.anchor_target.startswith(anchor_prefix)
        or not value.anchor_target[len(anchor_prefix) :].isdigit()
    ):
        raise AttestationError("runtime observation anchor target does not match keeper PID")
    if value.expected_image_sha256 != value.observed_image_sha256:
        raise AttestationError("runtime observation expected and observed image identities differ")
    visible_devices = value.cuda_visible_devices.split(",")
    if (
        len(visible_devices) != 4
        or len(set(visible_devices)) != 4
        or any(not device.strip() for device in visible_devices)
    ):
        raise AttestationError("runtime observation CUDA visibility must name four unique devices")
    if value.visible_gpu_count != 4:
        raise AttestationError("runtime observation GPU count must be exactly four")
    if len(set(value.visible_gpu_identities)) != 4:
        raise AttestationError("runtime observation GPU identities must be unique")
    for identity in value.visible_gpu_identities:
        _require_bounded_text(identity, "GPU identity")
        if "|" not in identity or any(not part.strip() for part in identity.split("|", 1)):
            raise AttestationError("GPU identity must contain a non-empty UUID and name")
    if tuple(name for name, _ in value.python_import_origins) != _REQUIRED_IMPORTS:
        raise AttestationError("Python import origins must name the exact required imports")
    if len(set(value.python_import_origins)) != len(value.python_import_origins):
        raise AttestationError("Python import origins must be unique")
    for name, origin in value.python_import_origins:
        _require_bounded_text(name, "Python import name")
        _require_canonical_absolute_path(origin, f"{name} import origin")


@dataclass(frozen=True)
class RuntimeNodeReuseEvidence:
    """Self-hashed evidence that one exact named container was reused."""

    schema_version: Literal["q30t-runtime-node-reuse-v1"]
    job_id: str
    phase: Literal["one-node", "two-node"]
    node_name: str
    container_name: str
    container_root_device: int
    container_root_inode: int
    container_sentinel_sha256: str
    observation_sha256: str
    named_container_reused: Literal[True]
    reuse_sha256: str

    def body_dict(self) -> dict[str, object]:
        """Return every self-hashed reuse field."""
        return {
            "container_name": self.container_name,
            "container_root_device": self.container_root_device,
            "container_root_inode": self.container_root_inode,
            "container_sentinel_sha256": self.container_sentinel_sha256,
            "job_id": self.job_id,
            "named_container_reused": self.named_container_reused,
            "node_name": self.node_name,
            "observation_sha256": self.observation_sha256,
            "phase": self.phase,
            "schema_version": self.schema_version,
        }

    def to_dict(self) -> dict[str, object]:
        """Return the complete JSON-compatible reuse evidence."""
        return self.body_dict() | {"reuse_sha256": self.reuse_sha256}

    def canonical_bytes(self) -> bytes:
        """Encode the reuse evidence in canonical JSON form."""
        return _canonical_json_bytes(self.to_dict())

    def verify_self_hash(self) -> None:
        """Reject reuse evidence whose body does not reproduce its hash."""
        if self.reuse_sha256 != _digest(self.body_dict()):
            raise AttestationError("runtime reuse evidence self-hash mismatch")

    def validate(self) -> None:
        """Reject structurally or semantically invalid reuse evidence."""
        if self.schema_version != _NODE_REUSE_SCHEMA:
            raise AttestationError("runtime reuse evidence schema is invalid")
        if self.phase not in {"one-node", "two-node"}:
            raise AttestationError("runtime reuse evidence phase is invalid")
        for value, label in (
            (self.job_id, "reuse job identity"),
            (self.node_name, "reuse node identity"),
            (self.container_name, "reuse container identity"),
        ):
            _require_bounded_text(value, label)
        if self.container_root_device <= 0 or self.container_root_inode <= 0:
            raise AttestationError("reuse container root identity is invalid")
        for value, label in (
            (self.container_sentinel_sha256, "reuse sentinel"),
            (self.observation_sha256, "reuse observation"),
            (self.reuse_sha256, "reuse evidence"),
        ):
            _require_hash(value, label)
        if self.named_container_reused is not True:
            raise AttestationError("named_container_reused must be literal true evidence")

    @classmethod
    def from_dict(cls, raw: object) -> RuntimeNodeReuseEvidence:
        """Decode and validate one exact reuse evidence schema."""
        record = _exact_record(raw, _REUSE_KEYS, "runtime reuse evidence")
        evidence = cls(
            schema_version=cast(
                "Literal['q30t-runtime-node-reuse-v1']", _required_text(record, "schema_version")
            ),
            job_id=_required_text(record, "job_id"),
            phase=cast("Literal['one-node', 'two-node']", _required_text(record, "phase")),
            node_name=_required_text(record, "node_name"),
            container_name=_required_text(record, "container_name"),
            container_root_device=_required_integer(record, "container_root_device"),
            container_root_inode=_required_integer(record, "container_root_inode"),
            container_sentinel_sha256=_required_text(record, "container_sentinel_sha256"),
            observation_sha256=_required_text(record, "observation_sha256"),
            named_container_reused=_required_true(record, "named_container_reused"),
            reuse_sha256=_required_text(record, "reuse_sha256"),
        )
        evidence.validate()
        evidence.verify_self_hash()
        return evidence


_REUSE_KEYS = frozenset(
    {
        "container_name",
        "container_root_device",
        "container_root_inode",
        "container_sentinel_sha256",
        "job_id",
        "named_container_reused",
        "node_name",
        "observation_sha256",
        "phase",
        "reuse_sha256",
        "schema_version",
    }
)


@dataclass(frozen=True)
class RuntimeKeeperLossEvidence:
    """Self-hashed expected failure after the exact keeper exits."""

    schema_version: Literal["q30t-runtime-node-keeper-loss-v1"]
    job_id: str
    phase: Literal["one-node", "two-node"]
    node_name: str
    keeper_pid: int
    keeper_start_ticks: int
    anchor_path: str
    anchor_device: int
    anchor_inode: int
    error_classification: Literal["anchor-unreadable"]
    keeper_loss_fresh_use_failed: Literal[True]
    keeper_loss_sha256: str

    def body_dict(self) -> dict[str, object]:
        """Return every self-hashed keeper-loss field."""
        return {
            "anchor_device": self.anchor_device,
            "anchor_inode": self.anchor_inode,
            "anchor_path": self.anchor_path,
            "error_classification": self.error_classification,
            "job_id": self.job_id,
            "keeper_loss_fresh_use_failed": self.keeper_loss_fresh_use_failed,
            "keeper_pid": self.keeper_pid,
            "keeper_start_ticks": self.keeper_start_ticks,
            "node_name": self.node_name,
            "phase": self.phase,
            "schema_version": self.schema_version,
        }

    def to_dict(self) -> dict[str, object]:
        """Return the complete JSON-compatible keeper-loss evidence."""
        return self.body_dict() | {"keeper_loss_sha256": self.keeper_loss_sha256}

    def canonical_bytes(self) -> bytes:
        """Encode the keeper-loss evidence in canonical JSON form."""
        return _canonical_json_bytes(self.to_dict())

    def verify_self_hash(self) -> None:
        """Reject keeper-loss evidence with an invalid self-hash."""
        if self.keeper_loss_sha256 != _digest(self.body_dict()):
            raise AttestationError("runtime keeper-loss evidence self-hash mismatch")

    def validate(self) -> None:
        """Reject structurally or semantically invalid keeper-loss evidence."""
        if self.schema_version != _KEEPER_LOSS_SCHEMA:
            raise AttestationError("runtime keeper-loss evidence schema is invalid")
        if self.phase not in {"one-node", "two-node"}:
            raise AttestationError("runtime keeper-loss evidence phase is invalid")
        if self.error_classification != "anchor-unreadable":
            raise AttestationError("keeper-loss error classification must be anchor-unreadable")
        if self.keeper_loss_fresh_use_failed is not True:
            raise AttestationError("keeper_loss_fresh_use_failed must be literal true evidence")
        if (
            self.keeper_pid <= 0
            or self.keeper_start_ticks < 0
            or self.anchor_device <= 0
            or self.anchor_inode <= 0
        ):
            raise AttestationError("keeper-loss process or anchor identity is invalid")
        for value, label in (
            (self.job_id, "keeper-loss job identity"),
            (self.node_name, "keeper-loss node identity"),
        ):
            _require_bounded_text(value, label)
        _require_canonical_absolute_path(self.anchor_path, "keeper-loss anchor")
        _require_hash(self.keeper_loss_sha256, "keeper-loss evidence")

    @classmethod
    def from_dict(cls, raw: object) -> RuntimeKeeperLossEvidence:
        """Decode and validate one exact keeper-loss evidence schema."""
        record = _exact_record(raw, _KEEPER_LOSS_KEYS, "runtime keeper-loss evidence")
        evidence = cls(
            schema_version=cast(
                "Literal['q30t-runtime-node-keeper-loss-v1']",
                _required_text(record, "schema_version"),
            ),
            job_id=_required_text(record, "job_id"),
            phase=cast("Literal['one-node', 'two-node']", _required_text(record, "phase")),
            node_name=_required_text(record, "node_name"),
            keeper_pid=_required_integer(record, "keeper_pid"),
            keeper_start_ticks=_required_integer(record, "keeper_start_ticks"),
            anchor_path=_required_text(record, "anchor_path"),
            anchor_device=_required_integer(record, "anchor_device"),
            anchor_inode=_required_integer(record, "anchor_inode"),
            error_classification=cast(
                "Literal['anchor-unreadable']", _required_text(record, "error_classification")
            ),
            keeper_loss_fresh_use_failed=_required_true(record, "keeper_loss_fresh_use_failed"),
            keeper_loss_sha256=_required_text(record, "keeper_loss_sha256"),
        )
        evidence.validate()
        evidence.verify_self_hash()
        return evidence


_KEEPER_LOSS_KEYS = frozenset(
    {
        "anchor_device",
        "anchor_inode",
        "anchor_path",
        "error_classification",
        "job_id",
        "keeper_loss_fresh_use_failed",
        "keeper_loss_sha256",
        "keeper_pid",
        "keeper_start_ticks",
        "node_name",
        "phase",
        "schema_version",
    }
)


@dataclass(frozen=True)
class RuntimeNodeAttestationInput:
    """Exact replayed inputs needed to publish one node receipt."""

    profile_path: Path
    profile_file_sha256: str
    archive_tree_receipt_path: Path
    archive_tree_receipt_file_sha256: str
    archive_tree_receipt: RuntimeArchiveTreeReceipt
    keeper_receipt_path: Path
    keeper_receipt_file_sha256: str
    keeper_receipt: KeeperReceipt
    mounted_sqsh_path: Path
    extracted_runtime_path: Path
    source_checkout: Path
    contract_path: Path
    runner_path: Path
    keeper_tool_path: Path
    attestation_tool_path: Path
    observation: RuntimeNodeObservation
    reuse_evidence: RuntimeNodeReuseEvidence
    keeper_loss_evidence: RuntimeKeeperLossEvidence


@dataclass(frozen=True)
class RuntimeNodeAttestationReceipt:
    """Canonical final attestation for one exact allocated node."""

    schema_version: Literal["q30t-runtime-node-attestation-v1"]
    job_id: str
    phase: Literal["one-node", "two-node"]
    cluster: Literal["ptyche"]
    profile_file_sha256: str
    node_name: str
    keeper_pid: int
    keeper_start_ticks: int
    keeper_receipt_sha256: str
    anchor_path: str
    anchor_target: str
    anchor_device: int
    anchor_inode: int
    archive_sha256: str
    archive_tree_receipt_file_sha256: str
    runtime_tree_sha256: str
    expected_image_sha256: str
    observed_image_sha256: str
    sentinel_inventory_sha256: str
    source_commit: str
    contract_file_sha256: str
    runner_sha256: str
    keeper_tool_sha256: str
    attestation_tool_sha256: str
    machine: Literal["aarch64"]
    python_executable: str
    python_version: str
    python_import_origins: tuple[tuple[str, str], ...]
    cuda_visible_devices: str
    visible_gpu_identities: tuple[str, str, str, str]
    visible_gpu_count: Literal[4]
    pyxis_version: str
    enroot_version: str
    container_name: str
    container_root_device: int
    container_root_inode: int
    container_sentinel_sha256: str
    named_container_reused: Literal[True]
    keeper_loss_fresh_use_failed: Literal[True]
    receipt_sha256: str

    def body_dict(self) -> dict[str, object]:
        """Return every self-hashed node receipt field."""
        observation = _receipt_as_observation(self)
        body = observation.body_dict()
        body["schema_version"] = self.schema_version
        body["named_container_reused"] = self.named_container_reused
        body["keeper_loss_fresh_use_failed"] = self.keeper_loss_fresh_use_failed
        return body

    def to_dict(self) -> dict[str, object]:
        """Return the complete JSON-compatible node receipt."""
        return self.body_dict() | {"receipt_sha256": self.receipt_sha256}

    def canonical_bytes(self) -> bytes:
        """Encode the node receipt in canonical JSON form."""
        return _canonical_json_bytes(self.to_dict())

    def verify_self_hash(self) -> None:
        """Reject a node receipt whose body does not reproduce its hash."""
        if self.receipt_sha256 != _digest(self.body_dict()):
            raise AttestationError("runtime node receipt self-hash mismatch")

    def validate(self) -> None:
        """Reject structurally or semantically invalid node receipt fields."""
        if self.schema_version != _NODE_RECEIPT_SCHEMA:
            raise AttestationError("runtime node receipt schema version is invalid")
        _validate_observation(_receipt_as_observation(self))
        if self.named_container_reused is not True:
            raise AttestationError("runtime node receipt has false reuse proof")
        if self.keeper_loss_fresh_use_failed is not True:
            raise AttestationError("runtime node receipt has false keeper-loss proof")
        _require_hash(self.receipt_sha256, "runtime node receipt")

    @classmethod
    def from_dict(cls, raw: object) -> RuntimeNodeAttestationReceipt:
        """Decode and validate one exact node receipt schema."""
        record = _exact_record(raw, _NODE_RECEIPT_KEYS, "runtime node receipt")
        observation_record = {
            key: value for key, value in record.items() if key in _OBSERVATION_KEYS
        }
        observation_record["schema_version"] = _NODE_OBSERVATION_SCHEMA
        observation_record["observation_sha256"] = "0" * 64
        observation = _observation_without_self_hash(observation_record)
        receipt = cls(
            **_observation_receipt_arguments(observation),
            named_container_reused=_required_true(record, "named_container_reused"),
            keeper_loss_fresh_use_failed=_required_true(record, "keeper_loss_fresh_use_failed"),
            receipt_sha256=_required_text(record, "receipt_sha256"),
        )
        receipt.validate()
        return receipt


_NODE_RECEIPT_KEYS = (_OBSERVATION_KEYS - {"observation_sha256"}) | frozenset(
    {"named_container_reused", "keeper_loss_fresh_use_failed", "receipt_sha256"}
)


def _observation_without_self_hash(raw: object) -> RuntimeNodeObservation:
    record = _exact_record(raw, _OBSERVATION_KEYS, "runtime observation")
    observation_sha256 = _required_text(record, "observation_sha256")
    record["observation_sha256"] = _digest(
        {key: value for key, value in record.items() if key != "observation_sha256"}
    )
    observation = RuntimeNodeObservation.from_dict(record)
    return RuntimeNodeObservation(
        **{**observation.__dict__, "observation_sha256": observation_sha256}
    )


def _receipt_as_observation(receipt: RuntimeNodeAttestationReceipt) -> RuntimeNodeObservation:
    return RuntimeNodeObservation(
        schema_version=cast(
            "Literal['q30t-runtime-node-observation-v1']", _NODE_OBSERVATION_SCHEMA
        ),
        **{
            key: value
            for key, value in receipt.__dict__.items()
            if key
            not in {
                "schema_version",
                "named_container_reused",
                "keeper_loss_fresh_use_failed",
                "receipt_sha256",
            }
        },
        observation_sha256="0" * 64,
    )


def _observation_receipt_arguments(observation: RuntimeNodeObservation) -> dict[str, Any]:
    values = dict(observation.__dict__)
    values.pop("observation_sha256")
    values["schema_version"] = cast(
        "Literal['q30t-runtime-node-attestation-v1']", _NODE_RECEIPT_SCHEMA
    )
    return values


@dataclass(frozen=True)
class RuntimeQualificationContext:
    """Authenticated common identities for one aggregate reconciliation."""

    job_id: str
    phase: Literal["one-node", "two-node"]
    cluster: Literal["ptyche"]
    profile_path: Path
    profile_file_sha256: str
    archive_tree_receipt_path: Path
    archive_tree_receipt_file_sha256: str
    archive_tree_receipt: RuntimeArchiveTreeReceipt
    archive_sha256: str
    image_sha256: str
    runtime_tree_sha256: str
    source_commit: str
    contract_file_sha256: str
    runner_sha256: str
    keeper_tool_sha256: str
    attestation_tool_sha256: str
    prerequisite_one_node_receipt_path: Path | None
    prerequisite_one_node_receipt_file_sha256: str | None


@dataclass(frozen=True)
class RuntimeQualificationReceipt:
    """Canonical one-node or two-node Q30 runtime qualification."""

    schema_version: Literal["q30t-runtime-qualification-v1"]
    job_id: str
    phase: Literal["one-node", "two-node"]
    cluster: Literal["ptyche"]
    profile_path: str
    profile_file_sha256: str
    archive_tree_receipt_path: str
    archive_tree_receipt_file_sha256: str
    archive_sha256: str
    image_sha256: str
    runtime_tree_sha256: str
    source_commit: str
    contract_file_sha256: str
    runner_sha256: str
    keeper_tool_sha256: str
    attestation_tool_sha256: str
    expected_node_count: Literal[1, 2]
    ordered_nodes: tuple[str, ...]
    ordered_node_receipt_file_sha256s: tuple[str, ...]
    ordered_node_receipts_sha256: str
    prerequisite_one_node_receipt_path: str | None
    prerequisite_one_node_receipt_file_sha256: str | None
    named_container_reuse_proven: Literal[True]
    keeper_loss_failure_proven: Literal[True]
    receipt_sha256: str

    def body_dict(self) -> dict[str, object]:
        """Return every self-hashed qualification field."""
        return {
            "archive_sha256": self.archive_sha256,
            "archive_tree_receipt_file_sha256": self.archive_tree_receipt_file_sha256,
            "archive_tree_receipt_path": self.archive_tree_receipt_path,
            "attestation_tool_sha256": self.attestation_tool_sha256,
            "cluster": self.cluster,
            "contract_file_sha256": self.contract_file_sha256,
            "expected_node_count": self.expected_node_count,
            "image_sha256": self.image_sha256,
            "job_id": self.job_id,
            "keeper_loss_failure_proven": self.keeper_loss_failure_proven,
            "keeper_tool_sha256": self.keeper_tool_sha256,
            "named_container_reuse_proven": self.named_container_reuse_proven,
            "ordered_node_receipt_file_sha256s": list(self.ordered_node_receipt_file_sha256s),
            "ordered_node_receipts_sha256": self.ordered_node_receipts_sha256,
            "ordered_nodes": list(self.ordered_nodes),
            "phase": self.phase,
            "prerequisite_one_node_receipt_file_sha256": (
                self.prerequisite_one_node_receipt_file_sha256
            ),
            "prerequisite_one_node_receipt_path": self.prerequisite_one_node_receipt_path,
            "profile_file_sha256": self.profile_file_sha256,
            "profile_path": self.profile_path,
            "runner_sha256": self.runner_sha256,
            "runtime_tree_sha256": self.runtime_tree_sha256,
            "schema_version": self.schema_version,
            "source_commit": self.source_commit,
        }

    def to_dict(self) -> dict[str, object]:
        """Return the complete JSON-compatible qualification receipt."""
        return self.body_dict() | {"receipt_sha256": self.receipt_sha256}

    def canonical_bytes(self) -> bytes:
        """Encode the qualification receipt in canonical JSON form."""
        return _canonical_json_bytes(self.to_dict())

    def verify_self_hash(self) -> None:
        """Reject a qualification whose body does not reproduce its hash."""
        if self.receipt_sha256 != _digest(self.body_dict()):
            raise AttestationError("qualification receipt self-hash mismatch")

    def validate(self) -> None:
        """Reject structurally or semantically invalid qualification fields."""
        if self.schema_version != _QUALIFICATION_SCHEMA:
            raise AttestationError("qualification receipt schema version is invalid")
        if self.phase not in {"one-node", "two-node"} or self.cluster != "ptyche":
            raise AttestationError("qualification phase or cluster is invalid")
        expected_count = 1 if self.phase == "one-node" else 2
        if self.expected_node_count != expected_count:
            raise AttestationError("qualification node count does not match phase")
        if (
            len(self.ordered_nodes) != expected_count
            or tuple(sorted(self.ordered_nodes)) != self.ordered_nodes
        ):
            raise AttestationError("qualification ordered node set is invalid")
        if len(set(self.ordered_nodes)) != expected_count:
            raise AttestationError("qualification ordered nodes contain duplicates")
        if len(self.ordered_node_receipt_file_sha256s) != expected_count:
            raise AttestationError("qualification node receipt inventory is invalid")
        if len(set(self.ordered_node_receipt_file_sha256s)) != expected_count:
            raise AttestationError("qualification node receipt hashes contain duplicates")
        for value, label in (
            (self.profile_file_sha256, "profile"),
            (self.archive_tree_receipt_file_sha256, "archive receipt"),
            (self.archive_sha256, "archive"),
            (self.image_sha256, "image"),
            (self.runtime_tree_sha256, "runtime tree"),
            (self.contract_file_sha256, "contract"),
            (self.runner_sha256, "runner"),
            (self.keeper_tool_sha256, "keeper tool"),
            (self.attestation_tool_sha256, "attestation tool"),
            (self.ordered_node_receipts_sha256, "ordered node receipts"),
            (self.receipt_sha256, "qualification receipt"),
        ):
            _require_hash(value, label)
        for value in self.ordered_node_receipt_file_sha256s:
            _require_hash(value, "node receipt file")
        expected_ordered_digest = _digest(
            [
                {"node_name": node, "receipt_file_sha256": file_sha256}
                for node, file_sha256 in zip(
                    self.ordered_nodes,
                    self.ordered_node_receipt_file_sha256s,
                    strict=True,
                )
            ]
        )
        if self.ordered_node_receipts_sha256 != expected_ordered_digest:
            raise AttestationError("qualification ordered node receipt digest is invalid")
        _require_source_commit(self.source_commit)
        _require_canonical_absolute_path(self.profile_path, "profile")
        _require_canonical_absolute_path(self.archive_tree_receipt_path, "archive receipt")
        if self.named_container_reuse_proven is not True:
            raise AttestationError("qualification has false reuse proof")
        if self.keeper_loss_failure_proven is not True:
            raise AttestationError("qualification has false keeper-loss proof")
        if self.phase == "one-node":
            if (
                self.prerequisite_one_node_receipt_path is not None
                or self.prerequisite_one_node_receipt_file_sha256 is not None
            ):
                raise AttestationError("one-node qualification cannot have a prerequisite")
        else:
            if (
                self.prerequisite_one_node_receipt_path is None
                or self.prerequisite_one_node_receipt_file_sha256 is None
            ):
                raise AttestationError("two-node qualification requires a one-node prerequisite")
            _require_canonical_absolute_path(
                self.prerequisite_one_node_receipt_path, "one-node prerequisite"
            )
            _require_hash(
                self.prerequisite_one_node_receipt_file_sha256,
                "one-node prerequisite file",
            )

    @classmethod
    def from_dict(cls, raw: object) -> RuntimeQualificationReceipt:
        """Decode and validate one exact qualification schema."""
        record = _exact_record(raw, _QUALIFICATION_KEYS, "qualification receipt")
        ordered_nodes = _text_list(record.get("ordered_nodes"), "ordered nodes")
        node_hashes = _text_list(
            record.get("ordered_node_receipt_file_sha256s"), "node receipt hashes"
        )
        receipt = cls(
            schema_version=cast(
                "Literal['q30t-runtime-qualification-v1']",
                _required_text(record, "schema_version"),
            ),
            job_id=_required_text(record, "job_id"),
            phase=cast("Literal['one-node', 'two-node']", _required_text(record, "phase")),
            cluster=cast("Literal['ptyche']", _required_text(record, "cluster")),
            profile_path=_required_text(record, "profile_path"),
            profile_file_sha256=_required_text(record, "profile_file_sha256"),
            archive_tree_receipt_path=_required_text(record, "archive_tree_receipt_path"),
            archive_tree_receipt_file_sha256=_required_text(
                record, "archive_tree_receipt_file_sha256"
            ),
            archive_sha256=_required_text(record, "archive_sha256"),
            image_sha256=_required_text(record, "image_sha256"),
            runtime_tree_sha256=_required_text(record, "runtime_tree_sha256"),
            source_commit=_required_text(record, "source_commit"),
            contract_file_sha256=_required_text(record, "contract_file_sha256"),
            runner_sha256=_required_text(record, "runner_sha256"),
            keeper_tool_sha256=_required_text(record, "keeper_tool_sha256"),
            attestation_tool_sha256=_required_text(record, "attestation_tool_sha256"),
            expected_node_count=cast(
                "Literal[1, 2]", _required_integer(record, "expected_node_count")
            ),
            ordered_nodes=ordered_nodes,
            ordered_node_receipt_file_sha256s=node_hashes,
            ordered_node_receipts_sha256=_required_text(record, "ordered_node_receipts_sha256"),
            prerequisite_one_node_receipt_path=_optional_text(
                record, "prerequisite_one_node_receipt_path"
            ),
            prerequisite_one_node_receipt_file_sha256=_optional_text(
                record, "prerequisite_one_node_receipt_file_sha256"
            ),
            named_container_reuse_proven=_required_true(record, "named_container_reuse_proven"),
            keeper_loss_failure_proven=_required_true(record, "keeper_loss_failure_proven"),
            receipt_sha256=_required_text(record, "receipt_sha256"),
        )
        receipt.validate()
        return receipt


_QUALIFICATION_KEYS = frozenset(
    {
        "archive_sha256",
        "archive_tree_receipt_file_sha256",
        "archive_tree_receipt_path",
        "attestation_tool_sha256",
        "cluster",
        "contract_file_sha256",
        "expected_node_count",
        "image_sha256",
        "job_id",
        "keeper_loss_failure_proven",
        "keeper_tool_sha256",
        "named_container_reuse_proven",
        "ordered_node_receipt_file_sha256s",
        "ordered_node_receipts_sha256",
        "ordered_nodes",
        "phase",
        "prerequisite_one_node_receipt_file_sha256",
        "prerequisite_one_node_receipt_path",
        "profile_file_sha256",
        "profile_path",
        "receipt_sha256",
        "runner_sha256",
        "runtime_tree_sha256",
        "schema_version",
        "source_commit",
    }
)


def _text_list(raw: object, label: str) -> tuple[str, ...]:
    if not isinstance(raw, list) or any(type(value) is not str for value in raw):
        raise AttestationError(f"{label} must be an array of text")
    return tuple(cast("list[str]", raw))


def attest_runtime_node(
    inputs: RuntimeNodeAttestationInput,
    *,
    output_path: Path,
    publication_job_id: str,
) -> RuntimeNodeAttestationReceipt:
    """Replay exact node inputs and publish one immutable final receipt."""
    _require_python_runtime()
    observation = inputs.observation
    _validate_attestation_paths(inputs, output_path)
    _cross_replay_node_inputs(inputs)
    _match_reuse_evidence(observation, inputs.reuse_evidence)
    _match_keeper_loss_evidence(observation, inputs.keeper_loss_evidence)
    observation.validate()
    observation.verify_self_hash()
    inputs.reuse_evidence.validate()
    inputs.reuse_evidence.verify_self_hash()
    inputs.keeper_loss_evidence.validate()
    inputs.keeper_loss_evidence.verify_self_hash()
    body = observation.body_dict()
    body["schema_version"] = _NODE_RECEIPT_SCHEMA
    body["named_container_reused"] = True
    body["keeper_loss_fresh_use_failed"] = True
    receipt = RuntimeNodeAttestationReceipt.from_dict(body | {"receipt_sha256": _digest(body)})
    receipt.verify_self_hash()
    _publish_or_adopt(
        output_path,
        receipt.canonical_bytes() + b"\n",
        publication_job_id=publication_job_id,
    )
    return receipt


def _validate_attestation_paths(inputs: RuntimeNodeAttestationInput, output_path: Path) -> None:
    for path, label in (
        (inputs.profile_path, "profile"),
        (inputs.archive_tree_receipt_path, "archive receipt"),
        (inputs.keeper_receipt_path, "keeper receipt"),
        (inputs.mounted_sqsh_path, "mounted image"),
        (inputs.extracted_runtime_path, "extracted runtime"),
        (inputs.source_checkout, "source checkout"),
        (inputs.contract_path, "contract"),
        (inputs.runner_path, "runner"),
        (inputs.keeper_tool_path, "keeper tool"),
        (inputs.attestation_tool_path, "attestation tool"),
        (output_path, "node receipt output"),
    ):
        _require_canonical_absolute_path(path, label)
    for tool in (
        inputs.contract_path,
        inputs.runner_path,
        inputs.keeper_tool_path,
        inputs.attestation_tool_path,
    ):
        if not _is_within(tool, inputs.source_checkout):
            raise AttestationError("runtime attestation tool is outside the source checkout")
    if output_path in {
        inputs.profile_path,
        inputs.archive_tree_receipt_path,
        inputs.keeper_receipt_path,
        inputs.mounted_sqsh_path,
        inputs.contract_path,
        inputs.runner_path,
        inputs.keeper_tool_path,
        inputs.attestation_tool_path,
    }:
        raise AttestationError("runtime node receipt cannot approve an input path")


def _cross_replay_node_inputs(inputs: RuntimeNodeAttestationInput) -> None:
    observation = inputs.observation
    _require_hash(inputs.profile_file_sha256, "profile caller")
    _require_hash(inputs.archive_tree_receipt_file_sha256, "archive receipt caller")
    _require_hash(inputs.keeper_receipt_file_sha256, "keeper receipt caller")
    profile_sha256 = _stable_file_sha256(inputs.profile_path, maximum_bytes=_MAX_METADATA_BYTES)
    if (
        profile_sha256 != inputs.profile_file_sha256
        or observation.profile_file_sha256 != profile_sha256
    ):
        raise AttestationError("profile whole-file SHA-256 mismatch")
    archive_receipt = load_runtime_archive_tree_receipt(
        inputs.archive_tree_receipt_path, inputs.archive_tree_receipt_file_sha256
    )
    if archive_receipt != inputs.archive_tree_receipt:
        raise AttestationError("archive receipt object does not match replayed bytes")
    if (
        observation.archive_tree_receipt_file_sha256 != inputs.archive_tree_receipt_file_sha256
        or observation.archive_sha256 != archive_receipt.archive_sha256
        or observation.runtime_tree_sha256 != archive_receipt.runtime_tree_sha256
        or observation.sentinel_inventory_sha256 != archive_receipt.sentinel_inventory_sha256
        or observation.source_commit != archive_receipt.source_commit
    ):
        raise AttestationError("archive/tree/source identity mismatch")
    keeper_receipt = _load_keeper_receipt(
        inputs.keeper_receipt_path, inputs.keeper_receipt_file_sha256
    )
    if keeper_receipt != inputs.keeper_receipt:
        raise AttestationError("keeper receipt object does not match replayed bytes")
    image_item = tuple(item for item in keeper_receipt.items if item.name == "runtime-image")
    if len(image_item) != 1:
        raise AttestationError("keeper receipt must contain exactly one runtime-image item")
    item = image_item[0]
    expected_anchor_target = f"/proc/{keeper_receipt.keeper_pid}/fd/{item.descriptor}"
    if (
        keeper_receipt.job_id != observation.job_id
        or keeper_receipt.node_name != observation.node_name
        or keeper_receipt.keeper_pid != observation.keeper_pid
        or keeper_receipt.keeper_start_ticks != observation.keeper_start_ticks
        or keeper_receipt.receipt_sha256 != observation.keeper_receipt_sha256
        or str(item.anchor_path) != observation.anchor_path
        or observation.anchor_target != expected_anchor_target
    ):
        raise AttestationError("keeper process or anchor identity mismatch")
    observed_image_sha256 = _stable_file_sha256(inputs.mounted_sqsh_path)
    if (
        observed_image_sha256 != item.expected_sha256
        or observed_image_sha256 != item.staged_sha256
        or observed_image_sha256 != observation.expected_image_sha256
        or observed_image_sha256 != observation.observed_image_sha256
    ):
        raise AttestationError("mounted image identity mismatch")
    for path, expected, label in (
        (
            inputs.contract_path,
            observation.contract_file_sha256,
            "contract",
        ),
        (inputs.runner_path, observation.runner_sha256, "runner"),
        (
            inputs.keeper_tool_path,
            observation.keeper_tool_sha256,
            "keeper tool",
        ),
        (
            inputs.attestation_tool_path,
            observation.attestation_tool_sha256,
            "attestation tool",
        ),
    ):
        if _stable_file_sha256(path, maximum_bytes=_MAX_TOOL_BYTES) != expected:
            raise AttestationError(f"{label} whole-file SHA-256 mismatch")
    runtime_root = _require_canonical_absolute_path(
        inputs.extracted_runtime_path, "extracted runtime"
    )
    source_root = _require_canonical_absolute_path(inputs.source_checkout, "source checkout")
    if not _is_within(Path(observation.python_executable), runtime_root):
        raise AttestationError("Python executable is outside the extracted runtime")
    origins = dict(observation.python_import_origins)
    if not _is_within(Path(origins["modelopt"]), source_root):
        raise AttestationError("modelopt import origin is outside the source checkout")
    for name in ("accelerate", "datasets", "wandb"):
        if not _is_within(Path(origins[name]), runtime_root):
            raise AttestationError(f"{name} import origin is outside the extracted runtime")


def _match_reuse_evidence(
    observation: RuntimeNodeObservation, reuse: RuntimeNodeReuseEvidence
) -> None:
    if (
        reuse.job_id != observation.job_id
        or reuse.phase != observation.phase
        or reuse.node_name != observation.node_name
        or reuse.container_name != observation.container_name
        or reuse.container_root_device != observation.container_root_device
        or reuse.container_root_inode != observation.container_root_inode
        or reuse.container_sentinel_sha256 != observation.container_sentinel_sha256
        or reuse.observation_sha256 != observation.observation_sha256
    ):
        raise AttestationError("named-container reuse identity mismatch")
    if reuse.named_container_reused is not True:
        raise AttestationError("named-container reuse evidence is false")


def _match_keeper_loss_evidence(
    observation: RuntimeNodeObservation, evidence: RuntimeKeeperLossEvidence
) -> None:
    if (
        evidence.job_id != observation.job_id
        or evidence.phase != observation.phase
        or evidence.node_name != observation.node_name
        or evidence.keeper_pid != observation.keeper_pid
        or evidence.keeper_start_ticks != observation.keeper_start_ticks
        or evidence.anchor_path != observation.anchor_path
        or evidence.anchor_device != observation.anchor_device
        or evidence.anchor_inode != observation.anchor_inode
    ):
        raise AttestationError("keeper-loss process or anchor identity mismatch")
    if (
        evidence.error_classification != "anchor-unreadable"
        or evidence.keeper_loss_fresh_use_failed is not True
    ):
        raise AttestationError("keeper-loss failure evidence is false")


def reconcile_runtime_receipts(
    *,
    receipts: tuple[RuntimeNodeAttestationReceipt, ...],
    receipt_file_sha256s: tuple[str, ...],
    expected_nodes: tuple[str, ...],
    context: RuntimeQualificationContext,
    output_path: Path,
    publication_job_id: str,
) -> RuntimeQualificationReceipt:
    """Reconcile exact node receipts into one immutable qualification receipt."""
    _require_python_runtime()
    _validate_context(context, output_path)
    if len(receipts) != len(receipt_file_sha256s):
        raise AttestationError("node receipt and file SHA inventories differ")
    expected_count = 1 if context.phase == "one-node" else 2
    if len(receipts) != expected_count or len(expected_nodes) != expected_count:
        raise AttestationError("qualification node set has a missing or extra node")
    if len(set(expected_nodes)) != expected_count:
        raise AttestationError("qualification expected node set contains duplicates")
    if {receipt.node_name for receipt in receipts} != set(expected_nodes):
        raise AttestationError("qualification node set mismatch")
    if len({receipt.node_name for receipt in receipts}) != expected_count:
        raise AttestationError("qualification node receipts contain duplicate nodes")
    _validate_common_receipt_identities(receipts, context)
    _validate_unique_node_identities(receipts)
    for receipt, file_sha256 in zip(receipts, receipt_file_sha256s, strict=True):
        _require_hash(file_sha256, "node receipt file")
        receipt.validate()
        receipt.verify_self_hash()
        canonical_file_sha256 = hashlib.sha256(receipt.canonical_bytes() + b"\n").hexdigest()
        if file_sha256 != canonical_file_sha256:
            raise AttestationError("node receipt file SHA does not bind its canonical bytes")
    prerequisite = _replay_prerequisite(context)
    if prerequisite is not None:
        _match_prerequisite_common_identities(prerequisite, context)
    ordered = sorted(
        zip(receipts, receipt_file_sha256s, strict=True), key=lambda pair: pair[0].node_name
    )
    ordered_nodes = tuple(receipt.node_name for receipt, _ in ordered)
    ordered_file_sha256s = tuple(file_sha256 for _, file_sha256 in ordered)
    ordered_digest = _digest(
        [
            {"node_name": node, "receipt_file_sha256": file_sha256}
            for node, file_sha256 in zip(ordered_nodes, ordered_file_sha256s, strict=True)
        ]
    )
    body: dict[str, object] = {
        "archive_sha256": context.archive_sha256,
        "archive_tree_receipt_file_sha256": context.archive_tree_receipt_file_sha256,
        "archive_tree_receipt_path": str(context.archive_tree_receipt_path),
        "attestation_tool_sha256": context.attestation_tool_sha256,
        "cluster": context.cluster,
        "contract_file_sha256": context.contract_file_sha256,
        "expected_node_count": expected_count,
        "image_sha256": context.image_sha256,
        "job_id": context.job_id,
        "keeper_loss_failure_proven": True,
        "keeper_tool_sha256": context.keeper_tool_sha256,
        "named_container_reuse_proven": True,
        "ordered_node_receipt_file_sha256s": list(ordered_file_sha256s),
        "ordered_node_receipts_sha256": ordered_digest,
        "ordered_nodes": list(ordered_nodes),
        "phase": context.phase,
        "prerequisite_one_node_receipt_file_sha256": (
            context.prerequisite_one_node_receipt_file_sha256
        ),
        "prerequisite_one_node_receipt_path": (
            str(context.prerequisite_one_node_receipt_path)
            if context.prerequisite_one_node_receipt_path is not None
            else None
        ),
        "profile_file_sha256": context.profile_file_sha256,
        "profile_path": str(context.profile_path),
        "runner_sha256": context.runner_sha256,
        "runtime_tree_sha256": context.runtime_tree_sha256,
        "schema_version": _QUALIFICATION_SCHEMA,
        "source_commit": context.source_commit,
    }
    qualification = RuntimeQualificationReceipt.from_dict(body | {"receipt_sha256": _digest(body)})
    qualification.verify_self_hash()
    _publish_or_adopt(
        output_path,
        qualification.canonical_bytes() + b"\n",
        publication_job_id=publication_job_id,
    )
    return qualification


def _validate_context(context: RuntimeQualificationContext, output_path: Path) -> None:
    if context.phase not in {"one-node", "two-node"} or context.cluster != "ptyche":
        raise AttestationError("qualification context phase or cluster is invalid")
    _require_bounded_text(context.job_id, "qualification job identity")
    for path, label in (
        (context.profile_path, "profile"),
        (context.archive_tree_receipt_path, "archive receipt"),
        (output_path, "qualification output"),
    ):
        _require_canonical_absolute_path(path, label)
    if output_path in {context.profile_path, context.archive_tree_receipt_path}:
        raise AttestationError("qualification output cannot approve an input path")
    for value, label in (
        (context.profile_file_sha256, "profile"),
        (context.archive_tree_receipt_file_sha256, "archive receipt"),
        (context.archive_sha256, "archive"),
        (context.image_sha256, "image"),
        (context.runtime_tree_sha256, "runtime tree"),
        (context.contract_file_sha256, "contract"),
        (context.runner_sha256, "runner"),
        (context.keeper_tool_sha256, "keeper tool"),
        (context.attestation_tool_sha256, "attestation tool"),
    ):
        _require_hash(value, label)
    _require_source_commit(context.source_commit)
    if (
        _stable_file_sha256(context.profile_path, maximum_bytes=_MAX_METADATA_BYTES)
        != context.profile_file_sha256
    ):
        raise AttestationError("qualification profile whole-file SHA-256 mismatch")
    archive_receipt = load_runtime_archive_tree_receipt(
        context.archive_tree_receipt_path, context.archive_tree_receipt_file_sha256
    )
    if archive_receipt != context.archive_tree_receipt:
        raise AttestationError("qualification archive receipt object does not match replayed bytes")
    if (
        archive_receipt.archive_sha256 != context.archive_sha256
        or archive_receipt.runtime_tree_sha256 != context.runtime_tree_sha256
        or archive_receipt.source_commit != context.source_commit
    ):
        raise AttestationError("qualification archive/tree/source context mismatch")
    if context.phase == "one-node":
        if (
            context.prerequisite_one_node_receipt_path is not None
            or context.prerequisite_one_node_receipt_file_sha256 is not None
        ):
            raise AttestationError("one-node context cannot contain a prerequisite")
    elif (
        context.prerequisite_one_node_receipt_path is None
        or context.prerequisite_one_node_receipt_file_sha256 is None
    ):
        raise AttestationError("two-node context requires a one-node prerequisite")


_COMMON_NODE_FIELDS = (
    "job_id",
    "phase",
    "cluster",
    "profile_file_sha256",
    "archive_sha256",
    "archive_tree_receipt_file_sha256",
    "runtime_tree_sha256",
    "expected_image_sha256",
    "observed_image_sha256",
    "sentinel_inventory_sha256",
    "source_commit",
    "contract_file_sha256",
    "runner_sha256",
    "keeper_tool_sha256",
    "attestation_tool_sha256",
    "machine",
    "python_executable",
    "python_version",
    "python_import_origins",
    "pyxis_version",
    "enroot_version",
    "container_name",
)


def _validate_common_receipt_identities(
    receipts: tuple[RuntimeNodeAttestationReceipt, ...], context: RuntimeQualificationContext
) -> None:
    first = receipts[0]
    for receipt in receipts[1:]:
        if any(getattr(receipt, field) != getattr(first, field) for field in _COMMON_NODE_FIELDS):
            raise AttestationError("node receipts have a common runtime identity mismatch")
    expected = {
        "job_id": context.job_id,
        "phase": context.phase,
        "cluster": context.cluster,
        "profile_file_sha256": context.profile_file_sha256,
        "archive_sha256": context.archive_sha256,
        "archive_tree_receipt_file_sha256": context.archive_tree_receipt_file_sha256,
        "runtime_tree_sha256": context.runtime_tree_sha256,
        "expected_image_sha256": context.image_sha256,
        "observed_image_sha256": context.image_sha256,
        "source_commit": context.source_commit,
        "contract_file_sha256": context.contract_file_sha256,
        "runner_sha256": context.runner_sha256,
        "keeper_tool_sha256": context.keeper_tool_sha256,
        "attestation_tool_sha256": context.attestation_tool_sha256,
    }
    if any(getattr(first, field) != value for field, value in expected.items()):
        raise AttestationError("node receipt common identity does not match context")
    if any(
        receipt.named_container_reused is not True
        or receipt.keeper_loss_fresh_use_failed is not True
        for receipt in receipts
    ):
        raise AttestationError("node receipt contains a false proof boolean")


def _validate_unique_node_identities(
    receipts: tuple[RuntimeNodeAttestationReceipt, ...],
) -> None:
    if len({receipt.keeper_pid for receipt in receipts}) != len(receipts):
        raise AttestationError("node receipts contain a duplicate keeper PID")
    anchors = {(receipt.anchor_device, receipt.anchor_inode) for receipt in receipts}
    if len(anchors) != len(receipts):
        raise AttestationError("node receipts contain a duplicate anchor identity")
    all_gpus = tuple(gpu for receipt in receipts for gpu in receipt.visible_gpu_identities)
    if len(set(all_gpus)) != len(all_gpus):
        raise AttestationError("node receipts contain a duplicate GPU identity")


def _replay_prerequisite(
    context: RuntimeQualificationContext,
) -> RuntimeQualificationReceipt | None:
    if context.phase == "one-node":
        return None
    path = context.prerequisite_one_node_receipt_path
    expected = context.prerequisite_one_node_receipt_file_sha256
    if path is None or expected is None:
        raise AttestationError("two-node qualification requires a one-node prerequisite")
    prerequisite = load_runtime_qualification_receipt(path, expected)
    if prerequisite.phase != "one-node" or prerequisite.expected_node_count != 1:
        raise AttestationError("two-node prerequisite is not a one-node qualification")
    return prerequisite


def _match_prerequisite_common_identities(
    prerequisite: RuntimeQualificationReceipt, context: RuntimeQualificationContext
) -> None:
    expected = (
        (prerequisite.cluster, context.cluster),
        (prerequisite.profile_path, str(context.profile_path)),
        (prerequisite.profile_file_sha256, context.profile_file_sha256),
        (prerequisite.archive_tree_receipt_path, str(context.archive_tree_receipt_path)),
        (
            prerequisite.archive_tree_receipt_file_sha256,
            context.archive_tree_receipt_file_sha256,
        ),
        (prerequisite.archive_sha256, context.archive_sha256),
        (prerequisite.image_sha256, context.image_sha256),
        (prerequisite.runtime_tree_sha256, context.runtime_tree_sha256),
        (prerequisite.source_commit, context.source_commit),
        (prerequisite.contract_file_sha256, context.contract_file_sha256),
        (prerequisite.runner_sha256, context.runner_sha256),
        (prerequisite.keeper_tool_sha256, context.keeper_tool_sha256),
        (prerequisite.attestation_tool_sha256, context.attestation_tool_sha256),
    )
    if any(left != right for left, right in expected):
        raise AttestationError("one-node prerequisite common identity mismatch")


def load_runtime_node_receipt(
    path: Path, expected_file_sha256: str
) -> RuntimeNodeAttestationReceipt:
    """Stable-read and replay one canonical node attestation receipt."""
    _require_python_runtime()
    _require_hash(expected_file_sha256, "node receipt caller")
    raw, observed = stable_single_link_bytes(path, maximum_bytes=_MAX_METADATA_BYTES)
    if observed != expected_file_sha256:
        raise AttestationError("node receipt whole-file SHA-256 mismatch")
    decoded = _decode_json(raw, "node receipt")
    receipt = RuntimeNodeAttestationReceipt.from_dict(decoded)
    if raw != receipt.canonical_bytes() + b"\n":
        raise AttestationError("node receipt is not canonical")
    receipt.verify_self_hash()
    return receipt


def load_runtime_qualification_receipt(
    path: Path, expected_file_sha256: str
) -> RuntimeQualificationReceipt:
    """Stable-read and replay one canonical aggregate qualification receipt."""
    _require_python_runtime()
    _require_hash(expected_file_sha256, "qualification receipt caller")
    raw, observed = stable_single_link_bytes(path, maximum_bytes=_MAX_METADATA_BYTES)
    if observed != expected_file_sha256:
        raise AttestationError("qualification receipt whole-file SHA-256 mismatch")
    decoded = _decode_json(raw, "qualification receipt")
    receipt = RuntimeQualificationReceipt.from_dict(decoded)
    if raw != receipt.canonical_bytes() + b"\n":
        raise AttestationError("qualification receipt is not canonical")
    receipt.verify_self_hash()
    return receipt


def _decode_json(raw: bytes, label: str) -> object:
    try:
        return json.loads(
            raw, object_pairs_hook=lambda pairs: _object_without_duplicates(pairs, label)
        )
    except (json.JSONDecodeError, UnicodeDecodeError) as error:
        raise AttestationError(f"{label} JSON is invalid") from error


def _object_without_duplicates(pairs: list[tuple[str, object]], label: str) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise AttestationError(f"{label} has duplicate JSON key: {key}")
        result[key] = value
    return result


def _load_keeper_receipt(path: Path, expected_file_sha256: str) -> KeeperReceipt:
    _require_hash(expected_file_sha256, "keeper receipt caller")
    raw, observed = stable_single_link_bytes(path, maximum_bytes=_MAX_METADATA_BYTES)
    if observed != expected_file_sha256:
        raise AttestationError("keeper receipt whole-file SHA-256 mismatch")
    record = _exact_record(
        _decode_json(raw, "keeper receipt"),
        frozenset(
            {
                "items",
                "job_id",
                "keeper_pid",
                "keeper_start_ticks",
                "node_name",
                "receipt_sha256",
                "schema_version",
            }
        ),
        "keeper receipt",
    )
    items_raw = record.get("items")
    if not isinstance(items_raw, list):
        raise AttestationError("keeper receipt items must be an array")
    items: list[KeeperItem] = []
    for item_raw in items_raw:
        item = _exact_record(
            item_raw,
            frozenset(
                {
                    "anchor_path",
                    "descriptor",
                    "expected_sha256",
                    "name",
                    "source_path",
                    "staged_sha256",
                    "staged_size",
                }
            ),
            "keeper item",
        )
        items.append(
            KeeperItem(
                name=_required_text(item, "name"),
                source_path=Path(_required_text(item, "source_path")),
                expected_sha256=_required_text(item, "expected_sha256"),
                staged_size=_required_integer(item, "staged_size"),
                staged_sha256=_required_text(item, "staged_sha256"),
                anchor_path=Path(_required_text(item, "anchor_path")),
                descriptor=_required_integer(item, "descriptor"),
            )
        )
    receipt = KeeperReceipt(
        schema_version=cast(
            "Literal['ptv23-node-keeper-v1']", _required_text(record, "schema_version")
        ),
        job_id=_required_text(record, "job_id"),
        node_name=_required_text(record, "node_name"),
        keeper_pid=_required_integer(record, "keeper_pid"),
        keeper_start_ticks=_required_integer(record, "keeper_start_ticks"),
        items=tuple(items),
        receipt_sha256=_required_text(record, "receipt_sha256"),
    )
    if receipt.schema_version != "ptv23-node-keeper-v1":
        raise AttestationError("keeper receipt schema is invalid")
    if receipt.keeper_pid <= 0 or receipt.keeper_start_ticks < 0:
        raise AttestationError("keeper receipt process identity is invalid")
    _require_hash(receipt.receipt_sha256, "keeper receipt")
    if tuple(sorted(receipt.items, key=lambda item: item.name)) != receipt.items:
        raise AttestationError("keeper receipt items are not ordered")
    for item in receipt.items:
        _require_canonical_absolute_path(item.source_path, "keeper source")
        _require_canonical_absolute_path(item.anchor_path, "keeper anchor")
        _require_hash(item.expected_sha256, "keeper expected source")
        _require_hash(item.staged_sha256, "keeper staged source")
        if item.staged_size < 0 or item.descriptor < 0:
            raise AttestationError("keeper item size or descriptor is invalid")
    canonical = _canonical_json_bytes(receipt.to_dict()) + b"\n"
    if raw != canonical:
        raise AttestationError("keeper receipt is not canonical")
    if receipt.receipt_sha256 != _digest(receipt.body_dict()):
        raise AttestationError("keeper receipt self-hash mismatch")
    return receipt


def _stable_file_sha256(path: Path, *, maximum_bytes: int | None = None) -> str:
    descriptor, parent_fd, before = _open_stable_regular(path)
    try:
        if maximum_bytes is not None and before.st_size > maximum_bytes:
            raise AttestationError(f"attestation input is too large: {path}")
        remaining = before.st_size
        digest = hashlib.sha256()
        while remaining:
            block = os.read(descriptor, min(_READ_BLOCK_BYTES, remaining))
            if not block:
                raise AttestationError(f"attestation input changed while hashing: {path}")
            digest.update(block)
            remaining -= len(block)
        if os.read(descriptor, 1):
            raise AttestationError(f"attestation input grew while hashing: {path}")
        _READ_HOOK()
        _require_stable_name(path, descriptor, parent_fd, before, "hashing")
        return digest.hexdigest()
    finally:
        os.close(descriptor)
        os.close(parent_fd)


def stable_single_link_bytes(path: Path, *, maximum_bytes: int) -> tuple[bytes, str]:
    """Read bounded bytes through one no-follow single-link descriptor."""
    descriptor, parent_fd, before = _open_stable_regular(path)
    try:
        if before.st_size > maximum_bytes:
            raise AttestationError("attestation metadata is too large")
        retained = bytearray()
        digest = hashlib.sha256()
        remaining = before.st_size
        while remaining:
            block = os.read(descriptor, min(_READ_BLOCK_BYTES, remaining))
            if not block:
                raise AttestationError("attestation metadata changed while reading")
            retained.extend(block)
            digest.update(block)
            remaining -= len(block)
        if os.read(descriptor, 1):
            raise AttestationError("attestation metadata grew while reading")
        _READ_HOOK()
        after = os.fstat(descriptor)
        _require_stable_name(path, descriptor, parent_fd, before, "reading")
        if _file_identity(before) != _file_identity(after) or len(retained) != before.st_size:
            raise AttestationError("attestation metadata changed while reading")
        return bytes(retained), digest.hexdigest()
    finally:
        os.close(descriptor)
        os.close(parent_fd)


def _open_stable_regular(path: Path) -> tuple[int, int, os.stat_result]:
    _require_canonical_absolute_path(path, "attestation input")
    nofollow = getattr(os, "O_NOFOLLOW", 0)
    if not isinstance(nofollow, int) or nofollow == 0:
        raise AttestationError("attestation requires O_NOFOLLOW")
    directory_flags = (
        os.O_RDONLY
        | getattr(os, "O_DIRECTORY", 0)
        | nofollow
        | getattr(os, "O_NONBLOCK", 0)
        | getattr(os, "O_CLOEXEC", 0)
    )
    file_flags = os.O_RDONLY | nofollow | getattr(os, "O_NONBLOCK", 0) | getattr(os, "O_CLOEXEC", 0)
    parent_fd = _open_absolute_directory(path.parent, directory_flags)
    try:
        named = os.stat(path.name, dir_fd=parent_fd, follow_symlinks=False)
        descriptor = os.open(path.name, file_flags, dir_fd=parent_fd)
    except OSError as error:
        os.close(parent_fd)
        raise AttestationError(f"attestation input is unreadable: {path}") from error
    opened = os.fstat(descriptor)
    if (
        _file_identity(named) != _file_identity(opened)
        or not stat.S_ISREG(opened.st_mode)
        or opened.st_nlink != 1
    ):
        os.close(descriptor)
        os.close(parent_fd)
        raise AttestationError(f"attestation input must be a single-link regular file: {path}")
    return descriptor, parent_fd, opened


def _open_absolute_directory(path: Path, flags: int) -> int:
    descriptor = os.open(Path("/"), flags)
    try:
        for component in path.parts[1:]:
            named = os.stat(component, dir_fd=descriptor, follow_symlinks=False)
            child = os.open(component, flags, dir_fd=descriptor)
            opened = os.fstat(child)
            if not stat.S_ISDIR(named.st_mode) or _directory_identity(named) != _directory_identity(
                opened
            ):
                os.close(child)
                raise AttestationError(f"attestation input directory changed: {path}")
            os.close(descriptor)
            descriptor = child
        return descriptor
    except BaseException:
        os.close(descriptor)
        raise


def _file_identity(status: os.stat_result) -> tuple[int, int, int, int, int, int, int]:
    return (
        status.st_dev,
        status.st_ino,
        status.st_mode,
        status.st_nlink,
        status.st_size,
        status.st_mtime_ns,
        status.st_ctime_ns,
    )


def _directory_identity(status: os.stat_result) -> tuple[int, int, int]:
    return status.st_dev, status.st_ino, stat.S_IFMT(status.st_mode)


def _require_stable_name(
    path: Path,
    descriptor: int,
    parent_fd: int,
    before: os.stat_result,
    phase: str,
) -> None:
    try:
        named = os.stat(path.name, dir_fd=parent_fd, follow_symlinks=False)
        absolute = os.stat(path, follow_symlinks=False)
    except OSError as error:
        raise AttestationError(f"attestation input changed while {phase}: {path}") from error
    opened = os.fstat(descriptor)
    if not (
        _file_identity(before)
        == _file_identity(opened)
        == _file_identity(named)
        == _file_identity(absolute)
    ):
        raise AttestationError(f"attestation input changed while {phase}: {path}")


def _adopt_exact(path: Path, expected: bytes) -> None:
    try:
        observed, _ = stable_single_link_bytes(path, maximum_bytes=len(expected))
    except (AttestationError, OSError) as error:
        raise FileExistsError(f"runtime attestation cannot adopt existing path: {path}") from error
    if observed != expected:
        raise FileExistsError(f"runtime attestation existing bytes differ: {path}")


def _publish_or_adopt(path: Path, payload: bytes, *, publication_job_id: str) -> None:
    if re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", publication_job_id) is None:
        raise ValueError("runtime attestation publication job identity is unsafe")
    if os.path.lexists(path):
        _adopt_exact(path, payload)
        return
    try:
        atomic_publish_bytes(path, payload, job_id=publication_job_id)
    except FileExistsError:
        if not os.path.lexists(path):
            raise
        _adopt_exact(path, payload)
        return
    _adopt_exact(path, payload)


def _require_python_runtime() -> None:
    if sys.version_info < (3, 12):
        raise RuntimeError("Q30 runtime attestation requires Python 3.12 or newer")


def _load_evidence(
    path: Path,
    expected_file_sha256: str,
    loader: Callable[[object], Any],
    label: str,
) -> Any:
    _require_hash(expected_file_sha256, f"{label} caller")
    raw, observed = stable_single_link_bytes(path, maximum_bytes=_MAX_METADATA_BYTES)
    if observed != expected_file_sha256:
        raise AttestationError(f"{label} whole-file SHA-256 mismatch")
    value = loader(_decode_json(raw, label))
    if raw != value.canonical_bytes() + b"\n":
        raise AttestationError(f"{label} is not canonical")
    value.verify_self_hash()
    return value


def _arguments(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Attest and reconcile Q30 Ptyche runtime evidence."
    )
    commands = parser.add_subparsers(dest="command", required=True)
    attest = commands.add_parser("attest-node", help="replay and publish one node receipt")
    attest.add_argument("--input", type=Path, required=True)
    attest.add_argument("--input-sha256")
    attest.add_argument("--output", type=Path, required=True)
    attest.add_argument("--job-id", required=True)
    reconcile = commands.add_parser("reconcile", help="reconcile explicit node receipts")
    reconcile.add_argument("--context", type=Path, required=True)
    reconcile.add_argument("--context-sha256")
    reconcile.add_argument("--node-receipt-list", type=Path, required=True)
    reconcile.add_argument("--node-receipt-list-sha256")
    reconcile.add_argument("--output", type=Path, required=True)
    reconcile.add_argument("--job-id", required=True)
    verify = commands.add_parser("verify", help="replay one node or aggregate receipt")
    verify.add_argument("--receipt", type=Path, required=True)
    verify.add_argument("--sha256", required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Run the runtime node attestation, reconciliation, or replay CLI."""
    _require_python_runtime()
    arguments = _arguments(argv)
    if arguments.command == "verify":
        raw, observed = stable_single_link_bytes(
            arguments.receipt, maximum_bytes=_MAX_METADATA_BYTES
        )
        if observed != arguments.sha256:
            raise AttestationError("receipt whole-file SHA-256 mismatch")
        decoded = _decode_json(raw, "runtime receipt")
        if not isinstance(decoded, dict):
            raise AttestationError("runtime receipt must be a JSON object")
        schema = decoded.get("schema_version")
        if schema == _NODE_RECEIPT_SCHEMA:
            receipt: RuntimeNodeAttestationReceipt | RuntimeQualificationReceipt = (
                load_runtime_node_receipt(arguments.receipt, arguments.sha256)
            )
        elif schema == _QUALIFICATION_SCHEMA:
            receipt = load_runtime_qualification_receipt(arguments.receipt, arguments.sha256)
        else:
            raise AttestationError("runtime receipt schema is unsupported")
    elif arguments.command == "attest-node":
        inputs = _load_attestation_input(arguments.input, arguments.input_sha256)
        receipt = attest_runtime_node(
            inputs,
            output_path=arguments.output,
            publication_job_id=arguments.job_id,
        )
    else:
        context = _load_context(arguments.context, arguments.context_sha256)
        receipts, receipt_hashes, expected_nodes = _load_node_receipt_list(
            arguments.node_receipt_list, arguments.node_receipt_list_sha256
        )
        receipt = reconcile_runtime_receipts(
            receipts=receipts,
            receipt_file_sha256s=receipt_hashes,
            expected_nodes=expected_nodes,
            context=context,
            output_path=arguments.output,
            publication_job_id=arguments.job_id,
        )
    sys.stdout.buffer.write(receipt.canonical_bytes() + b"\n")
    return 0


def _load_attestation_input(path: Path, expected_sha256: str | None) -> RuntimeNodeAttestationInput:
    record = _load_reference_record(path, expected_sha256, _ATTESTATION_INPUT_KEYS, "node input")
    archive_path = Path(_required_text(record, "archive_tree_receipt_path"))
    archive_sha = _required_text(record, "archive_tree_receipt_file_sha256")
    keeper_path = Path(_required_text(record, "keeper_receipt_path"))
    keeper_sha = _required_text(record, "keeper_receipt_file_sha256")
    observation = _load_evidence(
        Path(_required_text(record, "observation_path")),
        _required_text(record, "observation_file_sha256"),
        RuntimeNodeObservation.from_dict,
        "runtime observation",
    )
    reuse = _load_evidence(
        Path(_required_text(record, "reuse_evidence_path")),
        _required_text(record, "reuse_evidence_file_sha256"),
        RuntimeNodeReuseEvidence.from_dict,
        "runtime reuse evidence",
    )
    keeper_loss = _load_evidence(
        Path(_required_text(record, "keeper_loss_evidence_path")),
        _required_text(record, "keeper_loss_evidence_file_sha256"),
        RuntimeKeeperLossEvidence.from_dict,
        "runtime keeper-loss evidence",
    )
    return RuntimeNodeAttestationInput(
        profile_path=Path(_required_text(record, "profile_path")),
        profile_file_sha256=_required_text(record, "profile_file_sha256"),
        archive_tree_receipt_path=archive_path,
        archive_tree_receipt_file_sha256=archive_sha,
        archive_tree_receipt=load_runtime_archive_tree_receipt(archive_path, archive_sha),
        keeper_receipt_path=keeper_path,
        keeper_receipt_file_sha256=keeper_sha,
        keeper_receipt=_load_keeper_receipt(keeper_path, keeper_sha),
        mounted_sqsh_path=Path(_required_text(record, "mounted_sqsh_path")),
        extracted_runtime_path=Path(_required_text(record, "extracted_runtime_path")),
        source_checkout=Path(_required_text(record, "source_checkout")),
        contract_path=Path(_required_text(record, "contract_path")),
        runner_path=Path(_required_text(record, "runner_path")),
        keeper_tool_path=Path(_required_text(record, "keeper_tool_path")),
        attestation_tool_path=Path(_required_text(record, "attestation_tool_path")),
        observation=observation,
        reuse_evidence=reuse,
        keeper_loss_evidence=keeper_loss,
    )


_ATTESTATION_INPUT_KEYS = frozenset(
    {
        "archive_tree_receipt_file_sha256",
        "archive_tree_receipt_path",
        "attestation_tool_path",
        "contract_path",
        "extracted_runtime_path",
        "keeper_loss_evidence_file_sha256",
        "keeper_loss_evidence_path",
        "keeper_receipt_file_sha256",
        "keeper_receipt_path",
        "keeper_tool_path",
        "mounted_sqsh_path",
        "observation_file_sha256",
        "observation_path",
        "profile_file_sha256",
        "profile_path",
        "reuse_evidence_file_sha256",
        "reuse_evidence_path",
        "runner_path",
        "source_checkout",
    }
)


def _load_context(path: Path, expected_sha256: str | None) -> RuntimeQualificationContext:
    record = _load_reference_record(path, expected_sha256, _CONTEXT_KEYS, "qualification context")
    archive_path = Path(_required_text(record, "archive_tree_receipt_path"))
    archive_sha = _required_text(record, "archive_tree_receipt_file_sha256")
    prerequisite_path = _optional_text(record, "prerequisite_one_node_receipt_path")
    return RuntimeQualificationContext(
        job_id=_required_text(record, "job_id"),
        phase=cast("Literal['one-node', 'two-node']", _required_text(record, "phase")),
        cluster=cast("Literal['ptyche']", _required_text(record, "cluster")),
        profile_path=Path(_required_text(record, "profile_path")),
        profile_file_sha256=_required_text(record, "profile_file_sha256"),
        archive_tree_receipt_path=archive_path,
        archive_tree_receipt_file_sha256=archive_sha,
        archive_tree_receipt=load_runtime_archive_tree_receipt(archive_path, archive_sha),
        archive_sha256=_required_text(record, "archive_sha256"),
        image_sha256=_required_text(record, "image_sha256"),
        runtime_tree_sha256=_required_text(record, "runtime_tree_sha256"),
        source_commit=_required_text(record, "source_commit"),
        contract_file_sha256=_required_text(record, "contract_file_sha256"),
        runner_sha256=_required_text(record, "runner_sha256"),
        keeper_tool_sha256=_required_text(record, "keeper_tool_sha256"),
        attestation_tool_sha256=_required_text(record, "attestation_tool_sha256"),
        prerequisite_one_node_receipt_path=(
            Path(prerequisite_path) if prerequisite_path is not None else None
        ),
        prerequisite_one_node_receipt_file_sha256=_optional_text(
            record, "prerequisite_one_node_receipt_file_sha256"
        ),
    )


_CONTEXT_KEYS = frozenset(
    {
        "archive_sha256",
        "archive_tree_receipt_file_sha256",
        "archive_tree_receipt_path",
        "attestation_tool_sha256",
        "cluster",
        "contract_file_sha256",
        "image_sha256",
        "job_id",
        "keeper_tool_sha256",
        "phase",
        "prerequisite_one_node_receipt_file_sha256",
        "prerequisite_one_node_receipt_path",
        "profile_file_sha256",
        "profile_path",
        "runner_sha256",
        "runtime_tree_sha256",
        "source_commit",
    }
)


def _load_reference_record(
    path: Path,
    expected_sha256: str | None,
    keys: frozenset[str],
    label: str,
) -> dict[str, object]:
    if expected_sha256 is not None:
        _require_hash(expected_sha256, f"{label} caller")
    raw, observed = stable_single_link_bytes(path, maximum_bytes=_MAX_METADATA_BYTES)
    if expected_sha256 is not None and observed != expected_sha256:
        raise AttestationError(f"{label} whole-file SHA-256 mismatch")
    record = _exact_record(_decode_json(raw, label), keys, label)
    if raw != _canonical_json_bytes(record) + b"\n":
        raise AttestationError(f"{label} is not canonical")
    return record


def _load_node_receipt_list(
    path: Path, expected_sha256: str | None
) -> tuple[tuple[RuntimeNodeAttestationReceipt, ...], tuple[str, ...], tuple[str, ...]]:
    raw, observed = stable_single_link_bytes(path, maximum_bytes=_MAX_METADATA_BYTES)
    if expected_sha256 is not None:
        _require_hash(expected_sha256, "node receipt list caller")
    if expected_sha256 is not None and observed != expected_sha256:
        raise AttestationError("node receipt list whole-file SHA-256 mismatch")
    decoded = _decode_json(raw, "node receipt list")
    if not isinstance(decoded, list):
        raise AttestationError("node receipt list must be an array")
    receipts: list[RuntimeNodeAttestationReceipt] = []
    hashes: list[str] = []
    nodes: list[str] = []
    for item_raw in decoded:
        item = _exact_record(
            item_raw,
            frozenset({"node_name", "path", "sha256"}),
            "node receipt reference",
        )
        node = _required_text(item, "node_name")
        receipt_path = Path(_required_text(item, "path"))
        file_sha256 = _required_text(item, "sha256")
        receipt = load_runtime_node_receipt(receipt_path, file_sha256)
        if receipt.node_name != node:
            raise AttestationError("node receipt reference node mismatch")
        receipts.append(receipt)
        hashes.append(file_sha256)
        nodes.append(node)
    if raw != _canonical_json_bytes(decoded) + b"\n":
        raise AttestationError("node receipt list is not canonical")
    return tuple(receipts), tuple(hashes), tuple(nodes)


if __name__ == "__main__":
    raise SystemExit(main())
