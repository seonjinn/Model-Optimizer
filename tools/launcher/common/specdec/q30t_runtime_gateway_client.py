# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Fail-closed client and validators for the F-bound administrator services."""

from __future__ import annotations

import base64
import binascii
import dataclasses
import hashlib
import json
import os
import re
import shutil
import socket
import stat
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Literal, Protocol, cast

__all__ = [
    "AuthenticatedController",
    "AuthenticatedWrapper",
    "GatewayBarrierSession",
    "GatewayBlockedError",
    "GatewayProtocolError",
    "GatewayRPC",
    "GlobalOperationRegistration",
    "KeeperSignalTrace",
    "OperationBarrierDecision",
    "ProtectedBuildServiceEndpoint",
    "ProtectedControlEntry",
    "ProtectedControlPublication",
    "ProtectedImagePublication",
    "ProtectedParentIdentity",
    "RuntimeGatewayClient",
    "RuntimeGatewayEndpoint",
    "RuntimeServiceEndpointIdentity",
    "ServiceEndpointObservation",
    "SlurmTaskReady",
    "canonical_sha256",
    "connect_protected_build_service",
    "connect_runtime_gateway",
    "operation_runtime_path",
    "probe_mandatory_service",
    "reconstruct_protected_build_service_endpoint",
    "reconstruct_runtime_gateway_endpoint",
    "validate_control_publications",
    "validate_endpoint_observation",
    "validate_global_operation_registration",
    "validate_protected_image_publication",
    "validate_sigio_setup_order",
]

_HASH_RE = re.compile(r"[0-9a-f]{64}\Z")
_COMMIT_RE = re.compile(r"[0-9a-f]{40}\Z")
_JOB_RE = re.compile(r"(?:0|[1-9][0-9]*)\Z")
_OPERATION_RE = _HASH_RE


class GatewayProtocolError(RuntimeError):
    """The gateway protocol or a signed gateway claim is invalid."""


class GatewayBlockedError(GatewayProtocolError):
    """A mandatory installed administrator service is unavailable."""


class GatewayRPC(str, Enum):
    """The complete non-destructive administrator gateway RPC surface."""

    PUBLISH_LEASED_IMAGE = "publish-leased-image"
    PUBLISH_CONTROL_TREE = "publish-control-tree"
    REGISTER_GLOBAL_OPERATION = "register-global-operation"
    AWAIT_ALL_READY = "await-all-ready"
    DECIDE_GLOBAL_OPERATION = "decide-global-operation"
    OBSERVE_REMOTE_EXIT = "observe-remote-exit"


def _jsonable(value: object) -> object:
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return {
            field.name: _jsonable(getattr(value, field.name)) for field in dataclasses.fields(value)
        }
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Enum):
        return value.value
    return value


def _canonical(value: object) -> bytes:
    return json.dumps(
        _jsonable(value),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode()


def canonical_sha256(value: object) -> str:
    """Hash raw bytes or one canonically encoded JSON-compatible value."""
    payload = value if isinstance(value, bytes) else _canonical(value)
    return hashlib.sha256(payload).hexdigest()


def _is_hash(value: object) -> bool:
    return isinstance(value, str) and _HASH_RE.fullmatch(value) is not None


def _require_absolute(path: str, label: str) -> Path:
    parsed = Path(path)
    if not parsed.is_absolute() or str(parsed) != path or ".." in parsed.parts:
        raise GatewayProtocolError(f"{label} must be one canonical absolute path")
    return parsed


@dataclass(frozen=True)
class RuntimeServiceEndpointIdentity:
    """Reviewed immutable identity of one installed administrator service."""

    service_kind: Literal["gateway", "build-service"]
    endpoint_node_name: str
    service_identity: Literal[
        "systemd:q30t-runtime-gateway.service",
        "systemd:q30t-runtime-build.service",
    ]
    executable_path: str
    executable_sha256: str
    expected_service_uid: Literal[0]
    expected_service_gid: int
    control_socket_path: str
    control_socket_device: int
    control_socket_inode: int
    control_socket_uid: Literal[0]
    control_socket_gid: int
    control_socket_mode: Literal[432]
    public_key_path: str
    public_key_sha256: str
    identity_sha256: str

    def body_dict(self) -> dict[str, object]:
        """Return the canonical self-hashed endpoint body."""
        return {
            field.name: _jsonable(getattr(self, field.name))
            for field in dataclasses.fields(self)
            if field.name != "identity_sha256"
        }

    def to_dict(self) -> dict[str, object]:
        """Return the complete JSON-compatible endpoint identity."""
        return self.body_dict() | {"identity_sha256": self.identity_sha256}

    def validate(self) -> None:
        """Validate exact service literals, ownership, paths, and self hash."""
        expected_identity = {
            "gateway": "systemd:q30t-runtime-gateway.service",
            "build-service": "systemd:q30t-runtime-build.service",
        }
        if (
            self.service_kind not in expected_identity
            or self.service_identity != expected_identity[self.service_kind]
        ):
            raise GatewayProtocolError("endpoint identity has a swapped service identity")
        if not self.endpoint_node_name:
            raise GatewayProtocolError("endpoint identity node name is empty")
        _require_absolute(self.executable_path, "endpoint executable")
        _require_absolute(self.control_socket_path, "endpoint socket")
        _require_absolute(self.public_key_path, "endpoint public key")
        if not all(_is_hash(value) for value in (self.executable_sha256, self.public_key_sha256)):
            raise GatewayProtocolError("endpoint identity contains an invalid file hash")
        if self.expected_service_uid != 0 or self.control_socket_uid != 0:
            raise GatewayProtocolError("endpoint identity must be root-owned")
        if (
            self.expected_service_gid < 0
            or self.control_socket_gid != self.expected_service_gid
            or self.control_socket_mode != 0o660
            or self.control_socket_device < 0
            or self.control_socket_inode <= 0
        ):
            raise GatewayProtocolError("endpoint identity socket tuple is invalid")
        if self.identity_sha256 != canonical_sha256(self.body_dict()):
            raise GatewayProtocolError("endpoint identity self hash mismatch")

    @classmethod
    def from_dict(cls, raw: object) -> RuntimeServiceEndpointIdentity:
        """Decode one exact endpoint identity."""
        names = {field.name for field in dataclasses.fields(cls)}
        if not isinstance(raw, dict) or set(raw) != names:
            raise GatewayProtocolError("endpoint identity has an unexpected schema")
        try:
            identity = cls(**raw)
        except (TypeError, ValueError) as error:
            raise GatewayProtocolError("endpoint identity contains invalid values") from error
        identity.validate()
        return identity


@dataclass(frozen=True)
class ServiceEndpointObservation:
    """Post-connect live service and socket observation."""

    socket_device: int
    socket_inode: int
    socket_uid: int
    socket_gid: int
    socket_mode: int
    peer_pid: int
    peer_uid: int
    peer_gid: int
    peer_start_ticks: int
    executable_path: str
    executable_sha256: str
    systemd_identity: str
    public_key_sha256: str
    socket_parent_root_owned: bool
    socket_parent_nofollow: bool


@dataclass(frozen=True)
class RuntimeGatewayEndpoint:
    """F-bound gateway endpoint and its protected publication root."""

    identity: RuntimeServiceEndpointIdentity
    protected_runtime_root: Path


@dataclass(frozen=True)
class ProtectedBuildServiceEndpoint:
    """F-bound build service endpoint and hermetic builder closure."""

    identity: RuntimeServiceEndpointIdentity
    hermetic_builder_image_path: Path
    hermetic_builder_image_sha256: str
    hermetic_builder_closure_manifest_sha256: str


@dataclass(frozen=True)
class ProtectedParentIdentity:
    """Observed protection-domain properties of one publication parent."""

    uid: int
    gid: int
    mode: int
    replacable_by_producer: bool


@dataclass(frozen=True)
class ProtectedImagePublication:
    """Signed same-inode protected image publication."""

    schema_version: Literal["q30t-protected-image-publication-v1"]
    job_id: str
    node_name: str
    operation_id: str
    producer_uid: int
    protected_path: str
    device: int
    inode: int
    nlink: Literal[1]
    mode: Literal[256]
    size: int
    sha256: str
    same_ofd_lease_validated: Literal[True]
    gateway_signature: str

    def body_dict(self) -> dict[str, object]:
        """Return the exact gateway-signed publication body."""
        return {
            field.name: _jsonable(getattr(self, field.name))
            for field in dataclasses.fields(self)
            if field.name != "gateway_signature"
        }

    @classmethod
    def from_dict(cls, raw: object) -> ProtectedImagePublication:
        """Decode one exact protected-image publication."""
        return _from_exact_dict(cls, raw, "protected image publication")


@dataclass(frozen=True)
class ProtectedControlEntry:
    """One root-owned file in a protected committed-control tree."""

    entry_kind: Literal["commit-blob", "generated-manifest"]
    repo_relative_path: str | None
    control_relative_path: str
    device: int
    inode: int
    nlink: Literal[1]
    uid: Literal[0]
    gid: Literal[0]
    mode: Literal[292]
    size: int
    sha256: str

    @classmethod
    def from_dict(cls, raw: object) -> ProtectedControlEntry:
        """Decode one exact protected-control entry."""
        return _from_exact_dict(cls, raw, "protected control entry")


@dataclass(frozen=True)
class ProtectedControlPublication:
    """One signed, fresh, root-owned protected committed-control tree."""

    schema_version: Literal["q30t-protected-control-publication-v1"]
    job_id: str
    node_name: str
    operation_id: str
    source_commit: str
    protected_control_root: str
    root_device: int
    root_inode: int
    root_uid: Literal[0]
    root_gid: Literal[0]
    root_mode: Literal[365]
    ordered_entries: tuple[ProtectedControlEntry, ...]
    control_manifest_sha256: str
    publication_sha256: str
    gateway_signature: str

    def publication_body_dict(self) -> dict[str, object]:
        """Return the self-hashed publication body."""
        return {
            field.name: _jsonable(getattr(self, field.name))
            for field in dataclasses.fields(self)
            if field.name not in ("publication_sha256", "gateway_signature")
        }

    def signed_body_dict(self) -> dict[str, object]:
        """Return the complete body authenticated by the gateway."""
        return self.publication_body_dict() | {"publication_sha256": self.publication_sha256}

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-compatible publication."""
        return cast("dict[str, object]", _jsonable(self))

    @classmethod
    def from_dict(cls, raw: object) -> ProtectedControlPublication:
        """Decode one exact protected-control publication."""
        if not isinstance(raw, dict):
            raise GatewayProtocolError("protected control publication must be an object")
        converted = dict(raw)
        entries = converted.get("ordered_entries")
        if not isinstance(entries, list):
            raise GatewayProtocolError("protected control publication entries must be an array")
        converted["ordered_entries"] = tuple(
            ProtectedControlEntry.from_dict(item) for item in entries
        )
        return _from_exact_dict(cls, converted, "protected control publication")


@dataclass(frozen=True)
class KeeperSignalTrace:
    """Ordered live keeper signal/lease setup trace."""

    ordered_events: tuple[str, ...]
    unblocked_task_ids: tuple[int, ...]
    waiter_count: int


@dataclass(frozen=True)
class SlurmTaskReady:
    """Authenticated global-step wrapper READY frame."""

    schema_version: Literal["q30t-slurm-task-ready-v2"]
    job_id: str
    step_id: str
    node_name: str
    operation_id: str
    task_pid: int
    task_start_ticks: int
    cgroup_relative_path: str
    cgroup_device: int
    cgroup_inode: int
    image_device: int
    image_inode: int
    root_mount_readonly: Literal[True]
    frame_sha256: str

    def body_dict(self) -> dict[str, object]:
        """Return the READY frame body."""
        return {
            field.name: _jsonable(getattr(self, field.name))
            for field in dataclasses.fields(self)
            if field.name != "frame_sha256"
        }

    @classmethod
    def from_dict(cls, raw: object) -> SlurmTaskReady:
        """Decode one exact READY frame."""
        return _from_exact_dict(cls, raw, "READY frame")


@dataclass(frozen=True)
class AuthenticatedController:
    """Registered controller identity authenticated by the gateway."""

    pid: int
    start_ticks: int
    cgroup_relative_path: str
    job_id: str
    step_id: str
    executable_sha256: str
    peer_pid: int
    peer_uid: int
    peer_gid: int


@dataclass(frozen=True)
class AuthenticatedWrapper:
    """Registered global-step wrapper identity authenticated by the gateway."""

    node_name: str
    pid: int
    start_ticks: int
    cgroup_relative_path: str
    job_id: str
    step_id: str
    executable_sha256: str
    peer_pid: int
    peer_uid: int
    peer_gid: int


@dataclass(frozen=True)
class OperationBarrierDecision:
    """One immutable global GO/ABORT transaction."""

    schema_version: Literal["q30t-operation-barrier-decision-v1"]
    operation_id: str
    expected_nodes: tuple[str, ...]
    ordered_ready_frame_sha256s: tuple[str, ...]
    decision: Literal["GO", "ABORT"]
    attack_session: bool
    gateway_transaction_sha256: str
    gateway_signature: str


@dataclass(frozen=True)
class ModeledBarrierDecision:
    """Test-only barrier-model output which cannot authenticate a live decision."""

    operation_id: str
    expected_nodes: tuple[str, ...]
    ordered_ready_frame_sha256s: tuple[str, ...]
    decision: Literal["GO", "ABORT"]
    attack_session: bool
    test_only: Literal[True]
    cryptographically_usable: Literal[False]


@dataclass(frozen=True)
class GlobalOperationRegistration:
    """Signed gateway registration for one global execution operation."""

    schema_version: Literal["q30t-global-operation-registration-v1"]
    job_id: str
    operation_id: str
    expected_nodes: tuple[str, ...]
    protected_runtime_path: str
    derived_image_sha256: str
    policy: Literal["GO_AFTER_ALL_READY", "ABORT_ONLY"]
    controller_pid: int
    controller_start_ticks: int
    registration_sha256: str
    gateway_signature: str

    def registration_body_dict(self) -> dict[str, object]:
        """Return the self-hashed registration body."""
        return {
            field.name: _jsonable(getattr(self, field.name))
            for field in dataclasses.fields(self)
            if field.name not in ("registration_sha256", "gateway_signature")
        }

    def signed_body_dict(self) -> dict[str, object]:
        """Return the complete gateway-authenticated registration body."""
        return self.registration_body_dict() | {"registration_sha256": self.registration_sha256}


class RuntimeGatewayClient(Protocol):
    """Descriptor-only client surface of the installed gateway."""

    def request(self, rpc: GatewayRPC, payload: object, *, fds: tuple[int, ...] = ()) -> object:
        """Send one exact seqpacket request and return its decoded response."""


class ProtectedBuildServiceClient(Protocol):
    """Descriptor-only client surface of the installed build service."""

    def request(self, payload: object, *, fds: tuple[int, ...]) -> object:
        """Send one exact build request and return its decoded response."""


class _SeqpacketClient:
    def __init__(self, connection: socket.socket) -> None:
        self._connection = connection

    def request(self, rpc: GatewayRPC, payload: object, *, fds: tuple[int, ...] = ()) -> object:
        packet = _canonical({"payload": payload, "rpc": rpc.value})
        ancillary: list[tuple[int, int, bytes]] = []
        if fds:
            ancillary.append((socket.SOL_SOCKET, socket.SCM_RIGHTS, _pack_fds(fds)))
        self._connection.sendmsg([packet], ancillary)
        response = self._connection.recv(1024 * 1024 + 1)
        if len(response) > 1024 * 1024:
            raise GatewayProtocolError("gateway response exceeds the bounded packet size")
        try:
            decoded = json.loads(response)
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise GatewayProtocolError("gateway returned malformed JSON") from error
        if not isinstance(decoded, dict) or set(decoded) != {"ok", "payload"}:
            raise GatewayProtocolError("gateway returned an unexpected response schema")
        if decoded["ok"] is not True:
            raise GatewayProtocolError("gateway rejected the request")
        return decoded["payload"]


class _BuildSeqpacketClient:
    def __init__(self, connection: socket.socket) -> None:
        self._connection = connection

    def request(self, payload: object, *, fds: tuple[int, ...]) -> object:
        packet = _canonical({"payload": payload, "rpc": "protected-build"})
        self._connection.sendmsg(
            [packet],
            [(socket.SOL_SOCKET, socket.SCM_RIGHTS, _pack_fds(fds))],
        )
        response = self._connection.recv(1024 * 1024 + 1)
        if len(response) > 1024 * 1024:
            raise GatewayProtocolError("build response exceeds the bounded packet size")
        try:
            decoded = json.loads(response)
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise GatewayProtocolError("build service returned malformed JSON") from error
        if (
            not isinstance(decoded, dict)
            or decoded.get("ok") is not True
            or set(decoded)
            != {
                "ok",
                "payload",
            }
        ):
            raise GatewayProtocolError("build service rejected the request")
        return decoded["payload"]


class GatewayBarrierSession:
    """Fail-closed model of the installed gateway's registered barrier."""

    def __init__(
        self,
        *,
        operation_id: str,
        expected_nodes: tuple[str, ...],
        controller: AuthenticatedController,
        wrappers: tuple[AuthenticatedWrapper, ...],
        attack_session: bool,
        expected_uid: int,
        expected_gid: int,
    ) -> None:
        """Create one exact registered-controller barrier session."""
        if not _is_hash(operation_id) or len(expected_nodes) not in (1, 2, 16):
            raise GatewayProtocolError("barrier operation or node tuple is invalid")
        if len(set(expected_nodes)) != len(expected_nodes):
            raise GatewayProtocolError("barrier node tuple contains duplicates")
        if tuple(wrapper.node_name for wrapper in wrappers) != expected_nodes:
            raise GatewayProtocolError("barrier wrapper node tuple mismatch")
        self.operation_id = operation_id
        self.expected_nodes = expected_nodes
        self.controller = controller
        self.wrappers = wrappers
        self.attack_session = attack_session
        self.expected_uid = expected_uid
        self.expected_gid = expected_gid
        self._ready: dict[str, SlurmTaskReady] = {}
        self._abort_only = attack_session
        self._decision: ModeledBarrierDecision | None = None
        self._event_channel_protected = True
        self.authenticate_controller(controller)
        for wrapper in wrappers:
            self._validate_wrapper_identity(wrapper)

    def authenticate_controller(self, observed: AuthenticatedController) -> None:
        """Require the exact persistent registered-controller identity."""
        if observed.executable_sha256 != self.controller.executable_sha256:
            raise GatewayProtocolError("controller executable identity mismatch")
        if observed != self.controller:
            raise GatewayProtocolError("controller PID/start/cgroup/step identity mismatch")
        if (
            observed.peer_pid != observed.pid
            or observed.peer_uid != self.expected_uid
            or observed.peer_gid != self.expected_gid
        ):
            raise GatewayProtocolError("controller peer credentials mismatch")

    def _validate_wrapper_identity(self, wrapper: AuthenticatedWrapper) -> None:
        if (
            wrapper.peer_pid != wrapper.pid
            or wrapper.peer_uid != self.expected_uid
            or wrapper.peer_gid != self.expected_gid
        ):
            raise GatewayProtocolError("wrapper peer credentials mismatch")
        if wrapper.job_id != self.controller.job_id or wrapper.step_id != self.controller.step_id:
            raise GatewayProtocolError("wrapper step identity mismatch")
        if not _is_hash(wrapper.executable_sha256):
            raise GatewayProtocolError("wrapper executable identity mismatch")

    def accept_ready(self, frame: SlurmTaskReady) -> None:
        """Accept one authenticated READY frame exactly once."""
        if frame.node_name not in self.expected_nodes:
            raise GatewayProtocolError("READY node is not registered")
        if frame.node_name in self._ready:
            raise GatewayProtocolError("duplicate READY frame")
        wrapper = self.wrappers[self.expected_nodes.index(frame.node_name)]
        if frame.task_pid != wrapper.pid:
            raise GatewayProtocolError("READY PID mismatch")
        if frame.task_start_ticks != wrapper.start_ticks:
            raise GatewayProtocolError("READY start ticks mismatch")
        if frame.cgroup_relative_path != wrapper.cgroup_relative_path:
            raise GatewayProtocolError("READY cgroup mismatch")
        if frame.step_id != wrapper.step_id or frame.job_id != wrapper.job_id:
            raise GatewayProtocolError("READY step mismatch")
        if frame.operation_id != self.operation_id:
            raise GatewayProtocolError("READY operation mismatch")
        if not frame.root_mount_readonly or frame.frame_sha256 != canonical_sha256(
            frame.body_dict()
        ):
            raise GatewayProtocolError("READY frame self hash or mount identity mismatch")
        self._ready[frame.node_name] = frame

    def controller_eof(self) -> None:
        """Make the transaction permanently ABORT-only on controller EOF."""
        self._abort_only = True

    def mark_event_channel_drained(self) -> None:
        """Reject a drained or externally consumable barrier event channel."""
        self._event_channel_protected = False
        self._abort_only = True
        raise GatewayProtocolError("protected event channel was drained or withheld")

    def decide(self, decision: Literal["GO", "ABORT"]) -> ModeledBarrierDecision:
        """Commit exactly one global GO/ABORT decision."""
        if decision not in ("GO", "ABORT"):
            raise GatewayProtocolError("barrier decision is invalid")
        if decision == "GO" and self._abort_only:
            raise GatewayProtocolError("barrier session is permanently ABORT-only")
        if self._decision is not None:
            raise GatewayProtocolError("barrier transaction was already decided")
        if decision == "GO" and len(self._ready) != len(self.expected_nodes):
            raise GatewayProtocolError("not all nodes ready")
        if not self._event_channel_protected:
            raise GatewayProtocolError("protected event channel is unavailable")
        ordered_hashes = tuple(
            self._ready[node].frame_sha256 for node in self.expected_nodes if node in self._ready
        )
        self._decision = ModeledBarrierDecision(
            operation_id=self.operation_id,
            expected_nodes=self.expected_nodes,
            ordered_ready_frame_sha256s=ordered_hashes,
            decision=decision,
            attack_session=self.attack_session,
            test_only=True,
            cryptographically_usable=False,
        )
        return self._decision


def _from_exact_dict(cls: type[Any], raw: object, label: str) -> Any:
    names = {field.name for field in dataclasses.fields(cls)}
    if not isinstance(raw, dict) or set(raw) != names:
        raise GatewayProtocolError(f"{label} has an unexpected schema")
    try:
        return cls(**raw)
    except (TypeError, ValueError) as error:
        raise GatewayProtocolError(f"{label} contains invalid values") from error


def operation_runtime_path(
    protected_runtime_root: Path,
    job_id: str,
    operation_id: str,
    *,
    previously_used: frozenset[str] = frozenset(),
) -> Path:
    """Construct one fresh F-bound operation image path."""
    if not protected_runtime_root.is_absolute() or not _JOB_RE.fullmatch(job_id):
        raise GatewayProtocolError("operation path root or job ID is invalid")
    if _OPERATION_RE.fullmatch(operation_id) is None:
        raise GatewayProtocolError("operation path operation ID is invalid")
    path = protected_runtime_root / job_id / operation_id / "runtime.sqsh"
    if str(path) in previously_used:
        raise GatewayProtocolError("operation path reuse is forbidden")
    return path


def validate_endpoint_observation(
    identity: RuntimeServiceEndpointIdentity,
    observation: ServiceEndpointObservation,
    *,
    producer_uid: int,
    reviewed_peer_pid: int | None = None,
    reviewed_peer_start_ticks: int | None = None,
) -> ServiceEndpointObservation:
    """Join a live socket/process observation to its reviewed endpoint identity."""
    identity.validate()
    expected = (
        identity.control_socket_device,
        identity.control_socket_inode,
        identity.control_socket_uid,
        identity.control_socket_gid,
        identity.control_socket_mode,
        identity.expected_service_uid,
        identity.expected_service_gid,
        identity.executable_path,
        identity.executable_sha256,
        identity.service_identity,
        identity.public_key_sha256,
        True,
        True,
    )
    observed = (
        observation.socket_device,
        observation.socket_inode,
        observation.socket_uid,
        observation.socket_gid,
        observation.socket_mode,
        observation.peer_uid,
        observation.peer_gid,
        observation.executable_path,
        observation.executable_sha256,
        observation.systemd_identity,
        observation.public_key_sha256,
        observation.socket_parent_root_owned,
        observation.socket_parent_nofollow,
    )
    if observed != expected or observation.peer_pid <= 0 or observation.peer_start_ticks <= 0:
        raise GatewayProtocolError("endpoint identity does not match the live service")
    if producer_uid == identity.expected_service_uid:
        raise GatewayProtocolError("endpoint identity is producer-owned")
    if reviewed_peer_pid is not None and (
        observation.peer_pid != reviewed_peer_pid
        or observation.peer_start_ticks != reviewed_peer_start_ticks
    ):
        raise GatewayProtocolError("service restart invalidates the reviewed endpoint identity")
    return observation


def probe_mandatory_service(
    identity: RuntimeServiceEndpointIdentity,
) -> object:
    """Return PASS only after a live service matches every reviewed identity field."""
    from common.specdec.q30t_runtime_platform_feasibility import (  # circular type boundary
        FeasibilityProbeResult,
        FeasibilityStatus,
    )

    try:
        identity.validate()
        connection, observation = _connect_identity_with_observation(
            identity,
            producer_uid=os.getuid(),
        )
    except (GatewayProtocolError, OSError, NotImplementedError) as error:
        return FeasibilityProbeResult(
            status=FeasibilityStatus.BLOCKED,
            reason=f"mandatory administrator service live identity is unavailable: {error}",
            receipt=None,
        )
    try:
        return FeasibilityProbeResult(
            status=FeasibilityStatus.PASS,
            reason="mandatory administrator service live identity verified",
            receipt=observation,
        )
    finally:
        connection.close()


def validate_protected_image_publication(
    publication: ProtectedImagePublication,
    ready: Any,
    *,
    protected_runtime_root: Path,
    parent: ProtectedParentIdentity,
    gateway_identity: RuntimeServiceEndpointIdentity,
) -> ProtectedImagePublication:
    """Require a signed same-inode single-link mode-0400 protected image."""
    expected_path = operation_runtime_path(protected_runtime_root, ready.job_id, ready.operation_id)
    if publication.schema_version != "q30t-protected-image-publication-v1":
        raise GatewayProtocolError("protected image publication schema is invalid")
    if publication.protected_path != str(expected_path):
        raise GatewayProtocolError("protected image operation path mismatch")
    if (publication.device, publication.inode) != (ready.staged_device, ready.staged_inode):
        raise GatewayProtocolError("protected link is not the same inode")
    if publication.mode != 0o400:
        raise GatewayProtocolError("protected image must retain mode 0400")
    if publication.nlink != 1:
        raise GatewayProtocolError("protected image must have one single link")
    if not publication.same_ofd_lease_validated:
        raise GatewayProtocolError("protected image lacks same open file description proof")
    if (
        publication.job_id != ready.job_id
        or publication.node_name != ready.node_name
        or publication.operation_id != ready.operation_id
        or publication.producer_uid != ready.keeper_uid
        or publication.size != ready.staged_size
        or publication.sha256 != ready.staged_sha256
    ):
        raise GatewayProtocolError("protected image publication identity mismatch")
    if parent.uid != 0 or parent.replacable_by_producer or parent.mode & 0o222:
        raise GatewayProtocolError("protected parent is writable or replacable")
    if (
        gateway_identity.service_kind != "gateway"
        or gateway_identity.endpoint_node_name != publication.node_name
    ):
        raise GatewayProtocolError("protected image gateway identity was swapped")
    _verify_gateway_signature(publication.body_dict(), publication.gateway_signature, gateway_identity)
    return publication


def validate_sigio_setup_order(trace: KeeperSignalTrace) -> KeeperSignalTrace:
    """Require the fixed no-window SIGIO waiter and lease setup order."""
    required = (
        "BLOCK_SIGIO_INITIAL_THREAD",
        "CREATE_THREADS",
        "F_SETOWN_EX_F_OWNER_PID",
        "F_SETSIG_SIGIO",
        "ARM_SIGWAITINFO_WAITER",
        "VERIFY_EVERY_TASK_MASK",
        "F_SETLEASE_F_RDLCK",
    )
    try:
        positions = tuple(trace.ordered_events.index(event) for event in required)
    except ValueError as error:
        raise GatewayProtocolError("signal boundary trace is incomplete") from error
    if positions != tuple(sorted(positions)) or trace.unblocked_task_ids or trace.waiter_count != 1:
        raise GatewayProtocolError("signal boundary order or task masks are invalid")
    if "WRITER_OPEN_COMPLETED" in trace.ordered_events:
        writer_index = trace.ordered_events.index("WRITER_OPEN_COMPLETED")
        lease_index = trace.ordered_events.index("F_SETLEASE_F_RDLCK")
        if writer_index > lease_index and (
            "LEASE_BREAK_PACKET" not in trace.ordered_events
            or trace.ordered_events.index("LEASE_BREAK_PACKET") > writer_index
        ):
            raise GatewayProtocolError("signal boundary lost the kernel lease break packet")
    return trace


def _entry_manifest(publication: ProtectedControlPublication) -> tuple[tuple[object, ...], ...]:
    return tuple(
        (
            entry.entry_kind,
            entry.repo_relative_path,
            entry.control_relative_path,
            entry.mode,
            entry.size,
            entry.sha256,
        )
        for entry in publication.ordered_entries
    )


def validate_control_publications(
    publications: tuple[ProtectedControlPublication, ...],
    *,
    expected_nodes: tuple[str, ...],
    gateway_identities: tuple[RuntimeServiceEndpointIdentity, ...],
    expected_source_commit: str,
) -> tuple[ProtectedControlPublication, ...]:
    """Join fresh root-owned per-node committed-control publications."""
    if (
        not _COMMIT_RE.fullmatch(expected_source_commit)
        or len(publications) != len(expected_nodes)
        or len(gateway_identities) != len(expected_nodes)
        or tuple(item.node_name for item in publications) != expected_nodes
    ):
        raise GatewayProtocolError("control publication node tuple mismatch")
    if not publications:
        raise GatewayProtocolError("control publication tuple is empty")
    reference = publications[0]
    reference_manifest = _entry_manifest(reference)
    root_identities: set[tuple[int, int]] = set()
    file_identities: set[tuple[int, int]] = set()
    for publication, gateway_identity in zip(publications, gateway_identities, strict=True):
        if (
            gateway_identity.service_kind != "gateway"
            or gateway_identity.endpoint_node_name != publication.node_name
        ):
            raise GatewayProtocolError("control publication gateway identity was swapped")
        if publication.schema_version != "q30t-protected-control-publication-v1":
            raise GatewayProtocolError("control publication schema is invalid")
        if publication.root_uid != 0 or publication.root_gid != 0:
            raise GatewayProtocolError("control publication tree must be root-owned")
        if publication.root_mode != 0o555:
            raise GatewayProtocolError("control publication root must have mode 0555")
        if (
            not _COMMIT_RE.fullmatch(publication.source_commit)
            or publication.source_commit != expected_source_commit
        ):
            raise GatewayProtocolError("control publication source commit is invalid")
        if (
            publication.protected_control_root != reference.protected_control_root
            or publication.job_id != reference.job_id
            or publication.operation_id != reference.operation_id
            or publication.source_commit != reference.source_commit
        ):
            raise GatewayProtocolError(
                "control publications do not have the same path and operation"
            )
        if _entry_manifest(publication) != reference_manifest:
            raise GatewayProtocolError("control publication ordered manifest mismatch")
        if (
            not publication.ordered_entries
            or publication.ordered_entries[-1].entry_kind != "generated-manifest"
        ):
            raise GatewayProtocolError("control publication ordered entries are invalid")
        commit_paths = [
            entry.control_relative_path
            for entry in publication.ordered_entries
            if entry.entry_kind == "commit-blob"
        ]
        if commit_paths != sorted(commit_paths) or len(set(commit_paths)) != len(commit_paths):
            raise GatewayProtocolError("control publication ordered commit blobs are invalid")
        if publication.control_manifest_sha256 != canonical_sha256(_entry_manifest(publication)):
            raise GatewayProtocolError("control publication manifest hash mismatch")
        root_identity = (publication.root_device, publication.root_inode)
        if root_identity in root_identities:
            raise GatewayProtocolError(
                "control publications must use a distinct local inode per root"
            )
        root_identities.add(root_identity)
        for entry in publication.ordered_entries:
            if entry.uid != 0 or entry.gid != 0:
                raise GatewayProtocolError("control publication files must be root-owned")
            if entry.mode != 0o444:
                raise GatewayProtocolError("control publication files must have mode 0444")
            if entry.nlink != 1:
                raise GatewayProtocolError("control publication files require a single link")
            if not _is_hash(entry.sha256) or entry.size < 0:
                raise GatewayProtocolError("control publication file hash or size is invalid")
            identity = (entry.device, entry.inode)
            if identity in file_identities:
                raise GatewayProtocolError(
                    "control publications must use distinct local inode identities"
                )
            file_identities.add(identity)
        if publication.publication_sha256 != canonical_sha256(publication.publication_body_dict()):
            raise GatewayProtocolError("control publication self hash mismatch")
        _verify_gateway_signature(
            publication.signed_body_dict(),
            publication.gateway_signature,
            gateway_identity,
        )
    return publications


def validate_global_operation_registration(
    registration: GlobalOperationRegistration,
    *,
    gateway_identity: RuntimeServiceEndpointIdentity,
    expected_job_id: str,
    expected_nodes: tuple[str, ...],
    protected_runtime_root: Path,
) -> GlobalOperationRegistration:
    """Authenticate one exact fresh global-operation registration."""
    if (
        registration.schema_version != "q30t-global-operation-registration-v1"
        or registration.job_id != expected_job_id
        or registration.expected_nodes != expected_nodes
        or len(expected_nodes) not in (1, 2, 16)
        or len(set(expected_nodes)) != len(expected_nodes)
        or not _is_hash(registration.operation_id)
        or not _is_hash(registration.derived_image_sha256)
        or registration.policy not in ("GO_AFTER_ALL_READY", "ABORT_ONLY")
        or registration.controller_pid <= 0
        or registration.controller_start_ticks <= 0
        or registration.protected_runtime_path
        != str(
            operation_runtime_path(
                protected_runtime_root,
                expected_job_id,
                registration.operation_id,
            )
        )
    ):
        raise GatewayProtocolError("global operation registration identity is invalid")
    if registration.registration_sha256 != canonical_sha256(
        registration.registration_body_dict()
    ):
        raise GatewayProtocolError("global operation registration self hash mismatch")
    _verify_gateway_signature(
        registration.signed_body_dict(),
        registration.gateway_signature,
        gateway_identity,
    )
    return registration


def _verify_gateway_signature(
    body: object,
    encoded_signature: str,
    identity: RuntimeServiceEndpointIdentity,
) -> None:
    """Verify an Ed25519 signature with the exact F-bound gateway public key."""
    identity.validate()
    if identity.service_kind != "gateway":
        raise GatewayProtocolError("signature key is not an F-bound gateway identity")
    key_path = Path(identity.public_key_path)
    if _hash_regular(key_path) != identity.public_key_sha256:
        raise GatewayProtocolError("F-bound gateway public key hash mismatch")
    try:
        signature = base64.b64decode(encoded_signature, validate=True)
    except (binascii.Error, ValueError) as error:
        raise GatewayProtocolError("gateway signature encoding is invalid") from error
    search_path = "/opt/homebrew/bin:/usr/bin:/bin" if sys.platform == "darwin" else "/usr/bin:/bin"
    openssl = shutil.which("openssl", path=search_path)
    if openssl is None:
        raise GatewayProtocolError("gateway signature verifier is unavailable")
    with tempfile.NamedTemporaryFile() as payload_file, tempfile.NamedTemporaryFile() as signature_file:
        payload_file.write(_canonical(body))
        payload_file.flush()
        signature_file.write(signature)
        signature_file.flush()
        completed = subprocess.run(
            (
                openssl,
                "pkeyutl",
                "-verify",
                "-pubin",
                "-inkey",
                str(key_path),
                "-rawin",
                "-in",
                payload_file.name,
                "-sigfile",
                signature_file.name,
            ),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
            close_fds=True,
            env={"LC_ALL": "C", "PATH": "/usr/bin:/bin"},
        )
    if completed.returncode != 0:
        raise GatewayProtocolError("gateway signature is invalid")


def _require_stable_loaded(feasibility: object) -> None:
    from common.specdec.q30t_runtime_platform_feasibility import (
        FeasibilityError,
        require_verified_platform_feasibility,
    )

    try:
        require_verified_platform_feasibility(feasibility)
    except FeasibilityError as error:
        raise GatewayProtocolError(
            "service clients require a stable-loaded reviewed F receipt"
        ) from error


def reconstruct_runtime_gateway_endpoint(
    feasibility: Any,
    *,
    node_name: str,
) -> RuntimeGatewayEndpoint:
    """Reconstruct an exact reviewed gateway endpoint for one F-approved node."""
    _require_stable_loaded(feasibility)
    if node_name not in feasibility.ordered_nodes:
        raise GatewayProtocolError("node is not in the F-approved tuple")
    index = feasibility.ordered_nodes.index(node_name)
    identities = feasibility.ordered_gateway_endpoint_identities
    if len(identities) != len(feasibility.ordered_nodes):
        raise GatewayProtocolError("reviewed endpoint tuple cardinality mismatch")
    identity = identities[index]
    if identity.endpoint_node_name != node_name or identity.service_kind != "gateway":
        raise GatewayProtocolError("reviewed gateway endpoint identity was swapped")
    identity.validate()
    return RuntimeGatewayEndpoint(identity, Path(feasibility.protected_runtime_root))


def reconstruct_protected_build_service_endpoint(feasibility: Any) -> ProtectedBuildServiceEndpoint:
    """Reconstruct the exact reviewed protected-build endpoint."""
    _require_stable_loaded(feasibility)
    identity = feasibility.build_service_endpoint_identity
    if identity.service_kind != "build-service":
        raise GatewayProtocolError("reviewed build-service endpoint identity was swapped")
    identity.validate()
    return ProtectedBuildServiceEndpoint(
        identity=identity,
        hermetic_builder_image_path=Path(feasibility.hermetic_builder_image_path),
        hermetic_builder_image_sha256=feasibility.hermetic_builder_image_sha256,
        hermetic_builder_closure_manifest_sha256=feasibility.hermetic_builder_closure_manifest_sha256,
    )


def _hash_regular(path: Path) -> str:
    descriptor = os.open(path, os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW)
    try:
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode):
            raise GatewayProtocolError(f"endpoint file is not regular: {path}")
        digest = hashlib.sha256()
        while block := os.read(descriptor, 1024 * 1024):
            digest.update(block)
        return digest.hexdigest()
    finally:
        os.close(descriptor)


def _live_socket_observation(
    identity: RuntimeServiceEndpointIdentity,
    connection: socket.socket,
    *,
    peer_pid: int,
    peer_uid: int,
    peer_gid: int,
    peer_start_ticks: int,
) -> ServiceEndpointObservation:
    metadata = os.lstat(identity.control_socket_path)
    if not stat.S_ISSOCK(metadata.st_mode):
        raise GatewayBlockedError("mandatory service socket is not a Unix socket")
    parent = Path(identity.control_socket_path).parent
    parent_metadata = os.lstat(parent)
    executable_path = os.readlink(f"/proc/{peer_pid}/exe")
    cgroup = Path(f"/proc/{peer_pid}/cgroup").read_text()
    systemd_identity = (
        identity.service_identity if identity.service_identity.split(":", 1)[1] in cgroup else ""
    )
    return ServiceEndpointObservation(
        socket_device=metadata.st_dev,
        socket_inode=metadata.st_ino,
        socket_uid=metadata.st_uid,
        socket_gid=metadata.st_gid,
        socket_mode=stat.S_IMODE(metadata.st_mode),
        peer_pid=peer_pid,
        peer_uid=peer_uid,
        peer_gid=peer_gid,
        peer_start_ticks=peer_start_ticks,
        executable_path=executable_path,
        executable_sha256=_hash_regular(Path(executable_path)),
        systemd_identity=systemd_identity,
        public_key_sha256=_hash_regular(Path(identity.public_key_path)),
        socket_parent_root_owned=parent_metadata.st_uid == 0,
        socket_parent_nofollow=not stat.S_ISLNK(parent_metadata.st_mode),
    )


def _peer_credentials(connection: socket.socket) -> tuple[int, int, int]:
    if sys.platform != "linux":
        raise GatewayBlockedError("mandatory service identity requires Linux SO_PEERCRED")
    peer_raw = connection.getsockopt(socket.SOL_SOCKET, socket.SO_PEERCRED, 12)
    return (
        int.from_bytes(peer_raw[0:4], byteorder=sys.byteorder, signed=True),
        int.from_bytes(peer_raw[4:8], byteorder=sys.byteorder),
        int.from_bytes(peer_raw[8:12], byteorder=sys.byteorder),
    )


def _connect_identity_with_observation(
    identity: RuntimeServiceEndpointIdentity, *, producer_uid: int
) -> tuple[socket.socket, ServiceEndpointObservation]:
    if sys.platform != "linux":
        raise GatewayBlockedError("mandatory service identity requires Linux pidfds")
    for path in (
        Path(identity.control_socket_path),
        Path(identity.executable_path),
        Path(identity.public_key_path),
    ):
        if not path.exists():
            raise GatewayBlockedError(
                f"mandatory administrator service primitive is missing: {path}"
            )
    connection = socket.socket(
        socket.AF_UNIX,
        socket.SOCK_SEQPACKET | getattr(socket, "SOCK_CLOEXEC", 0),
    )
    try:
        connection.connect(identity.control_socket_path)
        peer_pid, peer_uid, peer_gid = _peer_credentials(connection)
        from common.specdec.q30t_runtime_platform_feasibility import (
            read_process_start_ticks,
            require_pidfd_identity,
        )

        peer_start_ticks = read_process_start_ticks(peer_pid)
        try:
            pidfd = os.pidfd_open(peer_pid, 0)
        except OSError as error:
            raise GatewayProtocolError("service pidfd identity is unavailable") from error
        try:
            require_pidfd_identity(pidfd, peer_pid, peer_start_ticks)
            observation = _live_socket_observation(
                identity,
                connection,
                peer_pid=peer_pid,
                peer_uid=peer_uid,
                peer_gid=peer_gid,
                peer_start_ticks=peer_start_ticks,
            )
            require_pidfd_identity(pidfd, peer_pid, peer_start_ticks)
            if _peer_credentials(connection) != (peer_pid, peer_uid, peer_gid):
                raise GatewayProtocolError("service peer restarted during identity review")
            repeated = _live_socket_observation(
                identity,
                connection,
                peer_pid=peer_pid,
                peer_uid=peer_uid,
                peer_gid=peer_gid,
                peer_start_ticks=peer_start_ticks,
            )
            if repeated != observation:
                raise GatewayProtocolError("service identity changed during live review")
        finally:
            os.close(pidfd)
        validate_endpoint_observation(
            identity,
            observation,
            producer_uid=producer_uid,
            reviewed_peer_pid=peer_pid,
            reviewed_peer_start_ticks=peer_start_ticks,
        )
    except Exception:
        connection.close()
        raise
    return connection, observation


def _connect_identity(
    identity: RuntimeServiceEndpointIdentity, *, producer_uid: int
) -> socket.socket:
    connection, _ = _connect_identity_with_observation(identity, producer_uid=producer_uid)
    return connection


def connect_runtime_gateway(feasibility: Any, *, node_name: str) -> RuntimeGatewayClient:
    """Connect only to the exact stable-loaded F-bound gateway endpoint."""
    endpoint = reconstruct_runtime_gateway_endpoint(feasibility, node_name=node_name)
    return _SeqpacketClient(_connect_identity(endpoint.identity, producer_uid=os.getuid()))


def connect_protected_build_service(feasibility: Any) -> ProtectedBuildServiceClient:
    """Connect only to the exact stable-loaded F-bound build endpoint."""
    endpoint = reconstruct_protected_build_service_endpoint(feasibility)
    return _BuildSeqpacketClient(_connect_identity(endpoint.identity, producer_uid=os.getuid()))


def _pack_fds(fds: tuple[int, ...]) -> bytes:
    import array

    packed = array.array("i", fds)
    return packed.tobytes()
