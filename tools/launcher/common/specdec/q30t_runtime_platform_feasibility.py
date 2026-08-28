# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Feasibility-only F receipts, syscall probes, and fail-closed validators."""

from __future__ import annotations

import argparse
import array
import dataclasses
import errno
import fcntl
import hashlib
import json
import os
import re
import socket
import stat
import struct
import subprocess
import sys
import weakref
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Literal, cast

from common.specdec.q30t_runtime_gateway_client import (
    ProtectedImagePublication,
    RuntimeServiceEndpointIdentity,
    _verify_gateway_signature,
    canonical_sha256,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence
    from typing import Any

__all__ = [
    "CONTROL_LOG_CAP_BYTES",
    "DATA_LOG_CAP_BYTES",
    "NODE_LOCAL_LOG_ROOT_TEMPLATE",
    "BoundedArtifact",
    "BoundedArtifactCollector",
    "ConfigManifestEntry",
    "DirectPyxisTrace",
    "FeasibilityError",
    "FeasibilityProbeResult",
    "FeasibilityStatus",
    "KeeperSameOFDReady",
    "MultinodeFeasibilityTrace",
    "PyxisDirectMountAttestation",
    "RecursiveConfigManifest",
    "RemoteStepExitProof",
    "RuntimePlatformFeasibilityReceipt",
    "RuntimePlatformFeasibilityRunReceipt",
    "StagingAgentTrace",
    "assert_sibling_pidfd_getfd_denied",
    "build_global_srun_argv",
    "build_platform_feasibility_submission_argv",
    "canonical_sha256",
    "execute_test_only_first",
    "join_platform_feasibility_runs",
    "load_platform_feasibility_receipt",
    "load_platform_feasibility_run_receipt",
    "main",
    "prove_keeper_same_ofd",
    "read_process_start_ticks",
    "receive_keeper_same_ofd",
    "require_pidfd_identity",
    "validate_direct_mount_attestations",
    "validate_direct_pyxis_trace",
    "validate_multinode_feasibility_trace",
    "validate_recursive_config_manifest",
    "validate_remote_step_exit_proof",
]

CONTROL_LOG_CAP_BYTES = 1024 * 1024
DATA_LOG_CAP_BYTES = 8 * 1024 * 1024
NODE_LOCAL_LOG_ROOT_TEMPLATE = "/raid/scratch/$USER/q30t-runtime-logs"
RUN_SCHEMA = "q30t-runtime-platform-feasibility-run-v1"
AGGREGATE_SCHEMA = "q30t-runtime-platform-feasibility-v3"
_HASH_RE = re.compile(r"[0-9a-f]{64}\Z")
_COMMIT_RE = re.compile(r"[0-9a-f]{40}\Z")
_JOB_RE = re.compile(r"(?:0|[1-9][0-9]*)\Z")
_MAX_RECEIPT_BYTES = 1024 * 1024


class FeasibilityError(RuntimeError):
    """F cannot prove a mandatory platform property."""


class FeasibilityStatus(str, Enum):
    """Normative outcome of one feasibility boundary."""

    PASS = "PASS"
    BLOCKED = "BLOCKED"


@dataclass(frozen=True)
class FeasibilityProbeResult:
    """Typed result which never carries a receipt on BLOCKED."""

    status: FeasibilityStatus
    reason: str
    receipt: object | None

    def __post_init__(self) -> None:
        if self.status is FeasibilityStatus.BLOCKED and self.receipt is not None:
            raise FeasibilityError("BLOCKED feasibility cannot carry a receipt")


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


def _is_hash(value: object) -> bool:
    return isinstance(value, str) and _HASH_RE.fullmatch(value) is not None


def _is_absolute(path: str) -> bool:
    parsed = Path(path)
    return parsed.is_absolute() and str(parsed) == path and ".." not in parsed.parts


@dataclass(frozen=True)
class KeeperSameOFDReady:
    """Credentialed keeper READY packet accompanying one leased OFD."""

    schema_version: Literal["ptv23-keeper-same-ofd-ready-v1"]
    job_id: str
    node_name: str
    operation_id: str
    keeper_pid: int
    keeper_start_ticks: int
    keeper_uid: int
    keeper_gid: int
    staged_device: int
    staged_inode: int
    staged_size: int
    staged_sha256: str
    staged_mode: Literal[256]
    staged_nlink_before_gateway: Literal[0]
    all_writer_descriptors_closed: Literal[True]
    all_threads_sigio_blocked: Literal[True]
    signal_owner_type: Literal["F_OWNER_PID"]
    signal_owner_pid: int
    sigwait_thread_armed_before_lease: Literal[True]
    lease_type: Literal["F_RDLCK"]
    lease_signal: Literal["SIGIO"]
    lease_break_time_seconds: int
    packet_sha256: str

    def body_dict(self) -> dict[str, object]:
        """Return the self-hashed READY body."""
        return {
            field.name: _jsonable(getattr(self, field.name))
            for field in dataclasses.fields(self)
            if field.name != "packet_sha256"
        }

    def to_dict(self) -> dict[str, object]:
        """Return the complete JSON-compatible READY packet."""
        return self.body_dict() | {"packet_sha256": self.packet_sha256}

    def canonical_bytes(self) -> bytes:
        """Return exact canonical READY bytes."""
        return _canonical(self.to_dict())

    def validate(self) -> None:
        """Validate exact lease, signal, inode, and self-hash claims."""
        if self.schema_version != "ptv23-keeper-same-ofd-ready-v1":
            raise FeasibilityError("keeper READY schema is invalid")
        if (
            not _JOB_RE.fullmatch(self.job_id)
            or not self.node_name
            or not _is_hash(self.operation_id)
            or not _is_hash(self.staged_sha256)
            or self.keeper_pid <= 0
            or self.keeper_start_ticks <= 0
            or self.keeper_uid < 0
            or self.keeper_gid < 0
            or self.staged_device < 0
            or self.staged_inode <= 0
            or self.staged_size < 0
        ):
            raise FeasibilityError("keeper READY identity is invalid")
        if (
            self.staged_mode != 0o400
            or self.staged_nlink_before_gateway != 0
            or not self.all_writer_descriptors_closed
            or not self.all_threads_sigio_blocked
            or self.signal_owner_type != "F_OWNER_PID"
            or self.signal_owner_pid != self.keeper_pid
            or not self.sigwait_thread_armed_before_lease
            or self.lease_type != "F_RDLCK"
            or self.lease_signal != "SIGIO"
            or self.lease_break_time_seconds < 15
        ):
            raise FeasibilityError("keeper READY lease/signal proof is invalid")
        if self.packet_sha256 != canonical_sha256(self.body_dict()):
            raise FeasibilityError("keeper READY self hash mismatch")

    @classmethod
    def from_dict(cls, raw: object) -> KeeperSameOFDReady:
        """Decode one exact READY packet."""
        names = {field.name for field in dataclasses.fields(cls)}
        if not isinstance(raw, dict) or set(raw) != names:
            raise FeasibilityError("keeper READY has an unexpected schema")
        try:
            ready = cls(**raw)
        except (TypeError, ValueError) as error:
            raise FeasibilityError("keeper READY contains invalid values") from error
        ready.validate()
        return ready


def read_process_start_ticks(pid: int) -> int:
    """Read Linux process start ticks without misparsing a parenthesized comm."""
    if sys.platform != "linux":
        raise NotImplementedError("process start ticks require Linux /proc")
    try:
        raw = Path(f"/proc/{pid}/stat").read_text()
    except OSError as error:
        raise FeasibilityError(f"cannot read process start ticks for PID {pid}") from error
    close = raw.rfind(")")
    fields_after_comm = raw[close + 2 :].split()
    if close < 0 or len(fields_after_comm) <= 19:
        raise FeasibilityError("process stat has an unexpected format")
    try:
        ticks = int(fields_after_comm[19])
    except ValueError as error:
        raise FeasibilityError("process start ticks are invalid") from error
    if ticks <= 0:
        raise FeasibilityError("process start ticks are invalid")
    return ticks


def _pid_from_pidfd(pidfd: int) -> int:
    if sys.platform != "linux":
        raise NotImplementedError("pidfd identity requires Linux")
    try:
        target = os.readlink(f"/proc/self/fd/{pidfd}")
    except OSError as error:
        raise FeasibilityError("pidfd is not a live process descriptor") from error
    match = re.fullmatch(r"anon_inode:\[pidfd\]", target)
    if match is None:
        raise FeasibilityError("pidfd is not a process descriptor")
    # Linux exposes the referenced PID in fdinfo, avoiding a caller assertion.
    try:
        fdinfo = Path(f"/proc/self/fdinfo/{pidfd}").read_text()
        pid_line = next(line for line in fdinfo.splitlines() if line.startswith("Pid:\t"))
        return int(pid_line.split("\t", 1)[1])
    except (OSError, StopIteration, ValueError) as error:
        raise FeasibilityError("pidfd does not expose a stable PID identity") from error


def require_pidfd_identity(pidfd: int, expected_pid: int, expected_start_ticks: int) -> None:
    """Join a pidfd to the exact PID and non-reused start-ticks identity."""
    observed_pid = _pid_from_pidfd(pidfd)
    if observed_pid != expected_pid:
        raise FeasibilityError("pidfd PID does not match the credentialed process")
    if read_process_start_ticks(observed_pid) != expected_start_ticks:
        raise FeasibilityError("pidfd process start ticks indicate PID reuse")


def prove_keeper_same_ofd(
    descriptor: int,
    ready: KeeperSameOFDReady,
    *,
    require_lease: bool = True,
) -> KeeperSameOFDReady:
    """Require the received descriptor to retain the keeper's exact read lease."""
    ready.validate()
    metadata = os.fstat(descriptor)
    flags = fcntl.fcntl(descriptor, fcntl.F_GETFL)
    if flags & os.O_ACCMODE != os.O_RDONLY:
        raise FeasibilityError("keeper descriptor is not O_RDONLY")
    if (metadata.st_dev, metadata.st_ino, metadata.st_size) != (
        ready.staged_device,
        ready.staged_inode,
        ready.staged_size,
    ):
        raise FeasibilityError("keeper descriptor inode identity mismatch")
    if require_lease and fcntl.fcntl(descriptor, fcntl.F_GETLEASE) != fcntl.F_RDLCK:
        raise FeasibilityError("descriptor does not prove the same open file description lease")
    return ready


def _decode_ancillary_for_test(
    *,
    credential_records: tuple[tuple[int, int, int], ...],
    rights_records: tuple[tuple[int, ...], ...],
    expected_pid: int,
    expected_uid: int,
    expected_gid: int,
) -> tuple[int, ...]:
    if len(credential_records) != 1:
        raise FeasibilityError("exactly one kernel SCM_CREDENTIALS record is required")
    if credential_records[0] != (expected_pid, expected_uid, expected_gid):
        raise FeasibilityError("SCM_CREDENTIALS process identity mismatch")
    if len(rights_records) != 1 or len(rights_records[0]) != 1:
        raise FeasibilityError("exactly one SCM_RIGHTS descriptor is required")
    return rights_records[0]


def receive_keeper_same_ofd(
    control: socket.socket,
    *,
    expected_keeper_pidfd: int,
    expected_keeper_start_ticks: int,
    expected_keeper_uid: int | None = None,
    expected_keeper_gid: int | None = None,
    require_lease: bool = True,
) -> tuple[KeeperSameOFDReady, int]:
    """Receive one credentialed READY packet and the exact leased OFD."""
    if sys.platform != "linux":
        raise FeasibilityError("SCM_CREDENTIALS feasibility requires Linux")
    expected_pid = _pid_from_pidfd(expected_keeper_pidfd)
    uid = os.getuid() if expected_keeper_uid is None else expected_keeper_uid
    gid = os.getgid() if expected_keeper_gid is None else expected_keeper_gid
    payload, ancillary, flags, _ = control.recvmsg(
        _MAX_RECEIPT_BYTES + 1,
        socket.CMSG_SPACE(struct.calcsize("3i")) + socket.CMSG_SPACE(array.array("i").itemsize),
    )
    if flags & (socket.MSG_TRUNC | socket.MSG_CTRUNC) or len(payload) > _MAX_RECEIPT_BYTES:
        raise FeasibilityError("credentialed keeper packet is truncated or oversized")
    credential_records: list[tuple[int, int, int]] = []
    rights_records: list[tuple[int, ...]] = []
    received_to_close: list[int] = []
    try:
        for level, kind, data in ancillary:
            if (level, kind) == (socket.SOL_SOCKET, socket.SCM_CREDENTIALS):
                if len(data) < struct.calcsize("3i"):
                    raise FeasibilityError("SCM_CREDENTIALS record is malformed")
                credential_records.append(struct.unpack("3i", data[: struct.calcsize("3i")]))
            elif (level, kind) == (socket.SOL_SOCKET, socket.SCM_RIGHTS):
                fds = array.array("i")
                fds.frombytes(data[: len(data) - (len(data) % fds.itemsize)])
                values = tuple(fds)
                received_to_close.extend(values)
                rights_records.append(values)
            else:
                raise FeasibilityError("unknown keeper ancillary record")
        rights = _decode_ancillary_for_test(
            credential_records=tuple(credential_records),
            rights_records=tuple(rights_records),
            expected_pid=expected_pid,
            expected_uid=uid,
            expected_gid=gid,
        )
        try:
            raw = json.loads(payload)
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise FeasibilityError("keeper READY payload is malformed") from error
        ready = KeeperSameOFDReady.from_dict(raw)
        if (ready.keeper_pid, ready.keeper_uid, ready.keeper_gid) != (expected_pid, uid, gid):
            raise FeasibilityError("SCM_CREDENTIALS do not join the READY identity")
        require_pidfd_identity(expected_keeper_pidfd, expected_pid, expected_keeper_start_ticks)
        if ready.keeper_start_ticks != expected_keeper_start_ticks:
            raise FeasibilityError("keeper READY start ticks indicate PID reuse")
        prove_keeper_same_ofd(rights[0], ready, require_lease=require_lease)
        received_to_close.remove(rights[0])
        return ready, rights[0]
    finally:
        for descriptor in received_to_close:
            os.close(descriptor)


receive_keeper_same_ofd.decode_ancillary_for_test = _decode_ancillary_for_test  # type: ignore[attr-defined]


def _pidfd_getfd(pidfd: int, target_fd: int) -> int:
    import ctypes

    libc = ctypes.CDLL(None, use_errno=True)
    result = libc.syscall(438, pidfd, target_fd, 0)
    if result < 0:
        error = ctypes.get_errno()
        if error == errno.ENOSYS:
            raise NotImplementedError("host kernel lacks pidfd_getfd")
        raise OSError(error, os.strerror(error))
    return int(result)


def assert_sibling_pidfd_getfd_denied() -> None:
    """Run a real sibling same-UID pidfd_getfd attack and require EPERM."""
    if sys.platform != "linux" or not hasattr(os, "pidfd_open"):
        raise NotImplementedError("host lacks Linux pidfd support")
    ready_read, ready_write = os.pipe()
    release_read, release_write = os.pipe()
    target_pid = os.fork()
    if target_pid == 0:
        try:
            os.close(ready_read)
            os.close(release_write)
            target_fd = os.open("/dev/null", os.O_RDONLY)
            os.write(ready_write, f"{target_fd}\n".encode())
            os.read(release_read, 1)
        finally:
            os._exit(0)
    os.close(ready_write)
    os.close(release_read)
    try:
        target_fd = int(os.read(ready_read, 32).strip())
        attack_read, attack_write = os.pipe()
        attacker_pid = os.fork()
        if attacker_pid == 0:
            os.close(attack_read)
            try:
                pidfd = os.pidfd_open(target_pid)
                try:
                    duplicated = _pidfd_getfd(pidfd, target_fd)
                except NotImplementedError:
                    os.write(attack_write, b"ENOSYS")
                except OSError as error:
                    os.write(attack_write, str(error.errno).encode())
                else:
                    os.close(duplicated)
                    os.write(attack_write, b"SUCCESS")
                finally:
                    os.close(pidfd)
            finally:
                os._exit(0)
        os.close(attack_write)
        result = os.read(attack_read, 32).decode()
        os.close(attack_read)
        os.waitpid(attacker_pid, 0)
        if result == "ENOSYS":
            raise NotImplementedError("host kernel lacks pidfd_getfd")
        if result != str(errno.EPERM):
            raise FeasibilityError(f"sibling pidfd_getfd was not denied with EPERM: {result}")
    finally:
        os.write(release_write, b"x")
        os.close(release_write)
        os.close(ready_read)
        os.waitpid(target_pid, 0)


@dataclass(frozen=True)
class DirectPyxisTrace:
    """Captured direct-mode Pyxis/enroot evidence."""

    ordered_log_lines: tuple[str, ...]
    container_image: str
    container_name: str | None
    container_save: str | None
    enroot_data_path: str | None
    persistent_rootfs_paths: tuple[str, ...]
    nested_execution_count: int


@dataclass(frozen=True)
class PyxisDirectMountAttestation:
    """Gateway-signed live SquashFUSE actual-open record."""

    schema_version: Literal["q30t-pyxis-direct-mount-attestation-v1"]
    job_id: str
    step_id: str
    node_name: str
    operation_id: str
    protected_path: str
    image_device: int
    image_inode: int
    squashfuse_pid: int
    squashfuse_start_ticks: int
    squashfuse_cgroup_relative_path: str
    backing_fd_number: int
    backing_fd_access_mode: Literal["O_RDONLY"]
    backing_fd_device: int
    backing_fd_inode: int
    root_mount_id: int
    root_mount_readonly: Literal[True]
    temporary_rootfs: Literal[True]
    container_creation_absent: Literal[True]
    persistent_rootfs_absent: Literal[True]
    gateway_signature: str

    def body_dict(self) -> dict[str, object]:
        """Return the exact gateway-signed live mount body."""
        return {
            field.name: _jsonable(getattr(self, field.name))
            for field in dataclasses.fields(self)
            if field.name != "gateway_signature"
        }


@dataclass(frozen=True)
class StagingAgentTrace:
    """One node-local staging agent/keeper/publication identity."""

    node_name: str
    agent_pid: int
    keeper_pid: int
    leased_fd: int
    device: int
    inode: int
    endpoint_identity_sha256: str
    protected_path: str


@dataclass(frozen=True)
class MultinodeFeasibilityTrace:
    """One exact 2- or 16-node staging plus global-execution trace."""

    node_count: int
    ordered_nodes: tuple[str, ...]
    ordered_staging_agents: tuple[StagingAgentTrace, ...]
    global_srun_argv: tuple[str, ...]
    global_execution_step_count: int
    global_rank_count: int
    ordered_global_ranks: tuple[int, ...]
    collective_token_sha256: str
    segment_count: int | None
    canary_full_feasibility: bool


@dataclass(frozen=True)
class ConfigManifestEntry:
    """One entry in the recursive site configuration manifest."""

    relative_path: str
    entry_type: str
    mode: int
    uid: int
    gid: int
    size: int
    sha256: str | None


@dataclass(frozen=True)
class RecursiveConfigManifest:
    """Exact system configuration plus protected empty user roots."""

    ordered_entries: tuple[ConfigManifestEntry, ...]
    manifest_sha256: str
    empty_user_config_root_device: int
    empty_user_config_root_inode: int
    empty_user_config_root_uid: int
    empty_user_config_root_gid: int
    empty_user_config_root_mode: int
    home: str
    xdg_config_home: str

    def body_dict(self) -> dict[str, object]:
        """Return the exact recursive manifest and empty-root identity body."""
        return {
            field.name: _jsonable(getattr(self, field.name))
            for field in dataclasses.fields(self)
            if field.name != "manifest_sha256"
        }


@dataclass(frozen=True)
class RemoteStepExitProof:
    """Joined remote-death proof for one exact global step."""

    schema_version: Literal["q30t-remote-step-exit-proof-v1"]
    job_id: str
    step_id: str
    ordered_nodes: tuple[str, ...]
    ordered_wrapper_pid_start: tuple[tuple[str, int, int], ...]
    authenticated_exit_nodes: tuple[str, ...]
    ordered_cgroup_identities: tuple[tuple[str, int, int], ...]
    every_cgroup_populated_zero: Literal[True]
    local_srun_returncode: int
    exact_step_terminal_state: Literal["COMPLETED", "CANCELLED", "FAILED", "TIMEOUT", "NODE_FAIL"]
    allocation_state: Literal[
        "RUNNING", "COMPLETING", "COMPLETED", "CANCELLED", "FAILED", "NODE_FAIL"
    ]
    cancel_elapsed_ms: int
    proof_completed_monotonic_ns: int
    proof_sha256: str

    def body_dict(self) -> dict[str, object]:
        """Return the remote-exit self-hashed body."""
        return {
            field.name: _jsonable(getattr(self, field.name))
            for field in dataclasses.fields(self)
            if field.name != "proof_sha256"
        }


def validate_direct_pyxis_trace(trace: DirectPyxisTrace) -> DirectPyxisTrace:
    """Require exact unnamed direct SquashFUSE log and path behavior."""
    positive = "skipping container creation (squashfuse enabled)"
    if trace.ordered_log_lines.count(positive) != 1:
        raise FeasibilityError("direct SquashFUSE positive line is absent or duplicated")
    if any("creating container filesystem" in line for line in trace.ordered_log_lines):
        raise FeasibilityError("container creation is forbidden")
    if trace.container_name is not None or trace.container_save is not None:
        raise FeasibilityError("named root or saved container is forbidden")
    if trace.enroot_data_path is not None:
        raise FeasibilityError("persistent ENROOT_DATA_PATH is forbidden")
    if trace.container_image.startswith("/proc/") or "/fd/" in trace.container_image:
        raise FeasibilityError("procfd image is forbidden")
    if not _is_absolute(trace.container_image):
        raise FeasibilityError("direct image path must be absolute")
    if trace.persistent_rootfs_paths:
        raise FeasibilityError("persistent rootfs is forbidden")
    if trace.nested_execution_count:
        raise FeasibilityError("nested execution is forbidden")
    return trace


def validate_direct_mount_attestations(
    attestations: tuple[PyxisDirectMountAttestation, ...],
    publications: tuple[ProtectedImagePublication, ...],
    gateway_identities: tuple[RuntimeServiceEndpointIdentity, ...],
) -> tuple[PyxisDirectMountAttestation, ...]:
    """Join every live actual-open fd to its node-local protected inode."""
    if sys.platform != "linux":
        raise FeasibilityError("live direct-mount attestation requires Linux /proc and pidfds")
    if (
        len(attestations) != len(publications)
        or len(gateway_identities) != len(attestations)
        or not attestations
    ):
        raise FeasibilityError("direct mount attestation cardinality mismatch")
    process_ids: set[tuple[int, int]] = set()
    process_pids: set[int] = set()
    mount_ids: set[int] = set()
    backing_fds: set[int] = set()
    backing_inodes: set[tuple[int, int]] = set()
    cgroups: set[str] = set()
    for attestation, publication, gateway_identity in zip(
        attestations, publications, gateway_identities, strict=True
    ):
        if (
            attestation.schema_version != "q30t-pyxis-direct-mount-attestation-v1"
            or attestation.node_name != publication.node_name
            or attestation.job_id != publication.job_id
            or attestation.operation_id != publication.operation_id
            or attestation.protected_path != publication.protected_path
        ):
            raise FeasibilityError("direct mount attestation identity mismatch")
        if (
            gateway_identity.endpoint_node_name != attestation.node_name
            or gateway_identity.service_kind != "gateway"
        ):
            raise FeasibilityError("direct mount gateway identity was swapped")
        if (
            (attestation.image_device, attestation.image_inode)
            != (publication.device, publication.inode)
            or (attestation.backing_fd_device, attestation.backing_fd_inode)
            != (publication.device, publication.inode)
            or attestation.backing_fd_access_mode != "O_RDONLY"
            or attestation.backing_fd_number < 0
        ):
            raise FeasibilityError("live backing fd does not join the protected inode")
        if (
            not attestation.root_mount_readonly
            or not attestation.temporary_rootfs
            or not attestation.container_creation_absent
            or not attestation.persistent_rootfs_absent
        ):
            raise FeasibilityError("direct mount attestation is incomplete")
        try:
            pidfd = os.pidfd_open(attestation.squashfuse_pid, 0)
        except OSError as error:
            raise FeasibilityError("live SquashFUSE process identity is unavailable") from error
        try:
            require_pidfd_identity(
                pidfd,
                attestation.squashfuse_pid,
                attestation.squashfuse_start_ticks,
            )
            cgroup_lines = Path(f"/proc/{attestation.squashfuse_pid}/cgroup").read_text().splitlines()
            if attestation.squashfuse_cgroup_relative_path not in {
                line.split(":", 2)[2] for line in cgroup_lines if line.count(":") >= 2
            }:
                raise FeasibilityError("live SquashFUSE cgroup identity mismatch")
            mountinfo_lines = Path(
                f"/proc/{attestation.squashfuse_pid}/mountinfo"
            ).read_text().splitlines()
            matching_mounts = [
                line.split()
                for line in mountinfo_lines
                if line.split() and line.split()[0] == str(attestation.root_mount_id)
            ]
            if len(matching_mounts) != 1 or "ro" not in matching_mounts[0][5].split(","):
                raise FeasibilityError("live root mount identity or readonly state mismatch")
            named_fd = os.open(
                attestation.protected_path,
                os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW,
            )
            try:
                named_metadata = os.fstat(named_fd)
                if (named_metadata.st_dev, named_metadata.st_ino) != (
                    publication.device,
                    publication.inode,
                ):
                    raise FeasibilityError("protected image live inode identity mismatch")
            finally:
                os.close(named_fd)
            live_fd = os.open(
                f"/proc/{attestation.squashfuse_pid}/fd/{attestation.backing_fd_number}",
                os.O_RDONLY | os.O_CLOEXEC,
            )
            try:
                live_metadata = os.fstat(live_fd)
                if (live_metadata.st_dev, live_metadata.st_ino) != (
                    attestation.backing_fd_device,
                    attestation.backing_fd_inode,
                ):
                    raise FeasibilityError("live backing fd inode identity mismatch")
                fdinfo = Path(
                    f"/proc/{attestation.squashfuse_pid}/fdinfo/{attestation.backing_fd_number}"
                ).read_text()
                flags_line = next(
                    (line for line in fdinfo.splitlines() if line.startswith("flags:\t")),
                    None,
                )
                if flags_line is None or int(flags_line.split("\t", 1)[1], 8) & os.O_ACCMODE:
                    raise FeasibilityError("live backing fd is not O_RDONLY")
            finally:
                os.close(live_fd)
            require_pidfd_identity(
                pidfd,
                attestation.squashfuse_pid,
                attestation.squashfuse_start_ticks,
            )
        except (OSError, StopIteration, ValueError) as error:
            raise FeasibilityError("live SquashFUSE identity is unavailable") from error
        finally:
            os.close(pidfd)
        try:
            _verify_gateway_signature(
                attestation.body_dict(),
                attestation.gateway_signature,
                gateway_identity,
            )
        except Exception as error:
            raise FeasibilityError("direct mount gateway signature is invalid") from error
        process = (attestation.squashfuse_pid, attestation.squashfuse_start_ticks)
        if (
            process in process_ids
            or attestation.squashfuse_pid in process_pids
            or attestation.root_mount_id in mount_ids
            or attestation.backing_fd_number in backing_fds
            or (attestation.backing_fd_device, attestation.backing_fd_inode) in backing_inodes
            or attestation.squashfuse_cgroup_relative_path in cgroups
        ):
            raise FeasibilityError("fresh SquashFUSE process and mount are required")
        process_ids.add(process)
        process_pids.add(attestation.squashfuse_pid)
        mount_ids.add(attestation.root_mount_id)
        backing_fds.add(attestation.backing_fd_number)
        backing_inodes.add((attestation.backing_fd_device, attestation.backing_fd_inode))
        cgroups.add(attestation.squashfuse_cgroup_relative_path)
    return attestations


def validate_multinode_feasibility_trace(
    trace: MultinodeFeasibilityTrace,
) -> MultinodeFeasibilityTrace:
    """Require exact distinct per-node staging and one global 2/16-node step."""
    if trace.node_count not in (2, 16):
        raise FeasibilityError("multinode node count must be exactly 2 or 16")
    if (
        len(trace.ordered_nodes) != trace.node_count
        or len(set(trace.ordered_nodes)) != trace.node_count
    ):
        raise FeasibilityError("multinode node count or node tuple mismatch")
    if (
        len(trace.ordered_staging_agents) != trace.node_count
        or tuple(agent.node_name for agent in trace.ordered_staging_agents) != trace.ordered_nodes
    ):
        raise FeasibilityError("multinode staging-agent tuple mismatch")
    if trace.global_execution_step_count != 1:
        raise FeasibilityError("exactly one global step is required")
    if trace.global_rank_count != trace.node_count or trace.ordered_global_ranks != tuple(
        range(trace.node_count)
    ):
        raise FeasibilityError("global rank count or ordering mismatch")
    required_flags = {
        "--nodes": str(trace.node_count),
        "--ntasks": str(trace.node_count),
        "--ntasks-per-node": "1",
        "--gpus-per-node": "4",
    }
    for flag, expected in required_flags.items():
        matches = tuple(
            argument
            for argument in trace.global_srun_argv
            if argument == flag or argument.startswith(f"{flag}=")
        )
        if matches != (f"{flag}={expected}",):
            raise FeasibilityError("global step argv topology or flag cardinality mismatch")
    if not _is_hash(trace.collective_token_sha256):
        raise FeasibilityError("global collective token is invalid")
    paths = {agent.protected_path for agent in trace.ordered_staging_agents}
    endpoint_ids = {agent.endpoint_identity_sha256 for agent in trace.ordered_staging_agents}
    if len(paths) != 1:
        raise FeasibilityError("nodes do not share one identical protected path")
    if len(endpoint_ids) != trace.node_count:
        raise FeasibilityError("gateway endpoint reuse is forbidden")
    uniqueness_columns = (
        {agent.agent_pid for agent in trace.ordered_staging_agents},
        {agent.keeper_pid for agent in trace.ordered_staging_agents},
        {agent.leased_fd for agent in trace.ordered_staging_agents},
        {(agent.device, agent.inode) for agent in trace.ordered_staging_agents},
    )
    if any(len(column) != trace.node_count for column in uniqueness_columns):
        raise FeasibilityError("node-local keeper/fd/inode reuse is forbidden")
    if trace.node_count == 16:
        if trace.segment_count != 16 or not trace.canary_full_feasibility:
            raise FeasibilityError(
                "sixteen-node trace must prove segment-16 canary/full feasibility"
            )
    elif trace.segment_count is not None or trace.canary_full_feasibility:
        raise FeasibilityError("two-node trace cannot be relabelled as segment-16 proof")
    return trace


def validate_recursive_config_manifest(
    observed: RecursiveConfigManifest,
    reviewed: RecursiveConfigManifest,
) -> RecursiveConfigManifest:
    """Require exact recursive system config and protected empty user config."""
    if (
        observed.ordered_entries != reviewed.ordered_entries
        or observed.manifest_sha256 != reviewed.manifest_sha256
    ):
        raise FeasibilityError("recursive config manifest differs from reviewed F")
    required_prefixes = (
        "etc/enroot/hooks.d/",
        "etc/enroot/mounts.d/",
        "etc/enroot/environ.d/",
    )
    paths = tuple(entry.relative_path for entry in observed.ordered_entries)
    if paths != tuple(sorted(paths)) or len(set(paths)) != len(paths):
        raise FeasibilityError("recursive config paths must be canonical and bytewise sorted")
    for entry in observed.ordered_entries:
        parsed = Path(entry.relative_path)
        if (
            parsed.is_absolute()
            or str(parsed) != entry.relative_path
            or ".." in parsed.parts
            or entry.entry_type not in ("regular", "directory")
            or entry.uid != 0
            or entry.gid != 0
            or entry.size < 0
            or entry.mode & 0o022
        ):
            raise FeasibilityError("recursive config entry identity is invalid")
        if entry.entry_type == "regular" and not _is_hash(entry.sha256):
            raise FeasibilityError("recursive config regular-file hash is invalid")
        if entry.entry_type == "directory" and (entry.sha256 is not None or entry.size != 0):
            raise FeasibilityError("recursive config directory identity is invalid")
    if not all(any(path.startswith(prefix) for path in paths) for prefix in required_prefixes):
        raise FeasibilityError("recursive config manifest omits an enroot site directory")
    if not any(path.endswith("spank_pyxis.so") for path in paths):
        raise FeasibilityError("recursive config manifest omits the Pyxis plugin")
    if not any(path.endswith("/enroot") for path in paths) or not any(
        path.endswith("/squashfuse") for path in paths
    ):
        raise FeasibilityError("recursive config manifest omits enroot or SquashFUSE")
    if (
        observed.home != observed.xdg_config_home
        or not observed.home.startswith("/run/q30t-empty-config")
        or observed.empty_user_config_root_device < 0
        or observed.empty_user_config_root_inode <= 0
        or observed.empty_user_config_root_uid != 0
        or observed.empty_user_config_root_gid != 0
        or observed.empty_user_config_root_mode != 0o555
    ):
        raise FeasibilityError("gateway-owned empty user config root is invalid")
    if observed.manifest_sha256 != canonical_sha256(observed.body_dict()):
        raise FeasibilityError("recursive config manifest self hash mismatch")
    return observed


def validate_remote_step_exit_proof(
    proof: RemoteStepExitProof,
    *,
    expected_job_id: str,
    expected_step_id: str,
    expected_nodes: tuple[str, ...],
    lease_acquired_monotonic_ns: int,
    lease_break_time_seconds: int,
) -> RemoteStepExitProof:
    """Require exact-step remote death strictly before every deadline."""
    if proof.schema_version != "q30t-remote-step-exit-proof-v1":
        raise FeasibilityError("remote exit proof schema is invalid")
    if proof.job_id != expected_job_id or proof.step_id != expected_step_id:
        raise FeasibilityError("remote exit proof does not name the exact step")
    if proof.ordered_nodes != expected_nodes:
        raise FeasibilityError("remote exit proof node tuple mismatch")
    if not all(len(item) == 3 for item in proof.ordered_wrapper_pid_start) or not all(
        len(item) == 3 for item in proof.ordered_cgroup_identities
    ):
        raise FeasibilityError("remote exit proof ordered identity joins are invalid")
    if (
        len(expected_nodes) not in (1, 2, 16)
        or len(set(expected_nodes)) != len(expected_nodes)
        or len(proof.ordered_wrapper_pid_start) != len(expected_nodes)
        or tuple(item[0] for item in proof.ordered_wrapper_pid_start) != expected_nodes
        or len(proof.ordered_cgroup_identities) != len(expected_nodes)
        or tuple(item[0] for item in proof.ordered_cgroup_identities) != expected_nodes
        or proof.authenticated_exit_nodes != expected_nodes
    ):
        raise FeasibilityError("remote exit proof ordered identity joins are invalid")
    wrapper_identities = tuple(item[1:] for item in proof.ordered_wrapper_pid_start)
    cgroup_identities = tuple(item[1:] for item in proof.ordered_cgroup_identities)
    if (
        any(pid <= 0 or ticks <= 0 for pid, ticks in wrapper_identities)
        or len(set(wrapper_identities)) != len(expected_nodes)
        or any(device < 0 or inode <= 0 for device, inode in cgroup_identities)
        or len(set(cgroup_identities)) != len(expected_nodes)
    ):
        raise FeasibilityError("remote exit proof wrapper or cgroup identity is invalid")
    if not proof.every_cgroup_populated_zero:
        raise FeasibilityError("remote exit proof has a populated cgroup")
    if proof.local_srun_returncode >= 0:
        raise FeasibilityError("global srun reaped status is not a cancelled process")
    if proof.exact_step_terminal_state not in (
        "COMPLETED",
        "CANCELLED",
        "FAILED",
        "TIMEOUT",
        "NODE_FAIL",
    ):
        raise FeasibilityError("scheduler step is not terminal")
    if proof.allocation_state not in (
        "RUNNING",
        "COMPLETING",
        "COMPLETED",
        "CANCELLED",
        "FAILED",
        "NODE_FAIL",
    ):
        raise FeasibilityError("allocation state is missing or invalid")
    if proof.cancel_elapsed_ms >= 5000:
        raise FeasibilityError("remote cancellation must finish in less than five seconds")
    deadline_ns = lease_acquired_monotonic_ns + (lease_break_time_seconds - 5) * 1_000_000_000
    if proof.proof_completed_monotonic_ns >= deadline_ns:
        raise FeasibilityError("remote exit proof must finish strictly before lease deadline")
    if proof.proof_sha256 != canonical_sha256(proof.body_dict()):
        raise FeasibilityError("remote exit proof self hash mismatch")
    return proof


@dataclass(frozen=True)
class BoundedArtifact:
    """Stable node-local log closed before any durable publication."""

    path: Path
    size: int
    sha256: str
    closed_before_publication: Literal[True]
    fsynced_before_publication: Literal[True]
    stable_before_publication: Literal[True]


class BoundedArtifactCollector:
    """Collect closed node-local logs under exact F caps."""

    def __init__(self, *, node_root: Path) -> None:
        """Bind collection to one canonical node-local log root."""
        if not node_root.is_absolute() or str(node_root) != os.path.normpath(str(node_root)):
            raise FeasibilityError("node-local log root must be canonical and absolute")
        self._node_root = node_root

    def collect(self, path: Path, *, kind: str) -> BoundedArtifact:
        """Stable-open, cap, fsync, and hash one node-local log."""
        if kind not in ("control", "batch", "staging", "pyxis", "task-stdout", "task-stderr"):
            raise FeasibilityError("bounded log kind is invalid")
        if not path.is_absolute() or str(path) != os.path.normpath(str(path)):
            raise FeasibilityError("log path must be canonical and absolute")
        try:
            relative = path.relative_to(self._node_root)
        except ValueError as error:
            raise FeasibilityError("log is outside the node-local bounded root") from error
        if not relative.parts:
            raise FeasibilityError("log path must name a file below the bounded root")
        cap = CONTROL_LOG_CAP_BYTES if kind == "control" else DATA_LOG_CAP_BYTES
        root_descriptor = os.open(
            self._node_root,
            os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW | os.O_DIRECTORY,
        )
        root_before = os.fstat(root_descriptor)
        directory = os.dup(root_descriptor)
        try:
            for component in relative.parts[:-1]:
                child = os.open(
                    component,
                    os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW | os.O_DIRECTORY,
                    dir_fd=directory,
                )
                os.close(directory)
                directory = child
            descriptor = os.open(
                relative.parts[-1],
                os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW,
                dir_fd=directory,
            )
        except OSError as error:
            os.close(directory)
            os.close(root_descriptor)
            raise FeasibilityError("bounded log nofollow stable open failed") from error
        except Exception:
            os.close(directory)
            os.close(root_descriptor)
            raise
        try:
            before = os.fstat(descriptor)
            if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1:
                raise FeasibilityError(f"{kind} log cap or regular-file invariant failed")
            os.fsync(descriptor)
            digest = hashlib.sha256()
            size = 0
            while block := os.read(descriptor, min(1024 * 1024, cap + 1 - size)):
                size += len(block)
                if size > cap:
                    raise FeasibilityError(f"{kind} log cap or regular-file invariant failed")
                digest.update(block)
            after = os.fstat(descriptor)
            named = os.stat(relative.parts[-1], dir_fd=directory, follow_symlinks=False)
            root_named = os.lstat(self._node_root)
            if (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns) != (
                after.st_dev,
                after.st_ino,
                after.st_size,
                after.st_mtime_ns,
            ) or (named.st_dev, named.st_ino) != (after.st_dev, after.st_ino):
                raise FeasibilityError("log changed during stable collection")
            if (root_named.st_dev, root_named.st_ino) != (
                root_before.st_dev,
                root_before.st_ino,
            ):
                raise FeasibilityError("bounded log root changed during stable collection")
            return BoundedArtifact(path, after.st_size, digest.hexdigest(), True, True, True)
        finally:
            os.close(descriptor)
            os.close(directory)
            os.close(root_descriptor)


@dataclass(frozen=True)
class RuntimePlatformFeasibilityRunReceipt:
    """Canonical receipt for one explicit 1-, 2-, or 16-node F job."""

    schema_version: Literal["q30t-runtime-platform-feasibility-run-v1"]
    cluster: Literal["ptyche"]
    node_count: Literal[1, 2, 16]
    producer_commit: str
    feasibility_bound_blobs_sha256: str
    job_id: str
    ordered_nodes: tuple[str, ...]
    prerequisite_node_count: Literal[1, 2] | None
    prerequisite_receipt_path: str | None
    prerequisite_receipt_file_sha256: str | None
    producer_uid: int
    yama_ptrace_scope: int
    raid_filesystem_type: str
    lustre_filesystem_type: Literal["lustre"]
    lease_break_time_seconds: int
    scm_rights_same_ofd_proven: Literal[True]
    sibling_pidfd_getfd_denied: Literal[True]
    protected_link_same_inode_proven: Literal[True]
    same_uid_write_break_proven: Literal[True]
    sigio_setup_order_proven: Literal[True]
    unnamed_direct_squashfuse_proven: Literal[True]
    live_backing_fd_same_inode_proven: Literal[True]
    persistent_rootfs_absent: Literal[True]
    global_multinode_path_routing_proven: bool
    global_go_abort_barrier_proven: Literal[True]
    one_global_execution_step_proven: Literal[True]
    global_srun_argv_sha256: str
    global_rank_count: int
    segment_count: Literal[16] | None
    collective_artifact_file_sha256: str
    remote_exit_proof_proven: Literal[True]
    ordered_gateway_endpoint_identities: tuple[RuntimeServiceEndpointIdentity, ...]
    protected_runtime_root: str
    build_service_endpoint_identity: RuntimeServiceEndpointIdentity
    hermetic_builder_image_path: str
    hermetic_builder_image_sha256: str
    hermetic_builder_closure_manifest_sha256: str
    pyxis_plugin_path: str
    pyxis_plugin_sha256: str
    pyxis_version: str
    enroot_path: str
    enroot_sha256: str
    enroot_version: str
    squashfuse_path: str
    squashfuse_sha256: str
    slurm_spank_manifest_sha256: str
    enroot_system_config_manifest_sha256: str
    empty_user_config_root_device: int
    empty_user_config_root_inode: int
    ordered_probe_artifact_file_sha256s: tuple[str, ...]
    ordered_bounded_log_file_sha256s: tuple[str, ...]
    receipt_sha256: str

    def body_dict(self) -> dict[str, object]:
        """Return the complete self-hashed run body."""
        return {
            field.name: _jsonable(getattr(self, field.name))
            for field in dataclasses.fields(self)
            if field.name != "receipt_sha256"
        }

    def to_dict(self) -> dict[str, object]:
        """Return one JSON-compatible run receipt."""
        return self.body_dict() | {"receipt_sha256": self.receipt_sha256}

    def canonical_bytes(self) -> bytes:
        """Return exact canonical receipt bytes without publication newline."""
        return _canonical(self.to_dict())

    @classmethod
    def create(cls, **fields: Any) -> RuntimePlatformFeasibilityRunReceipt:
        """Create and validate one self-hashed run receipt."""
        receipt = cls(receipt_sha256=canonical_sha256(fields), **fields)
        receipt.validate(expected_node_count=cast("int", fields["node_count"]))
        return receipt

    @classmethod
    def from_dict(cls, raw: object) -> RuntimePlatformFeasibilityRunReceipt:
        """Decode one exact run receipt and nested endpoint identities."""
        names = {field.name for field in dataclasses.fields(cls)}
        if not isinstance(raw, dict) or set(raw) != names:
            raise FeasibilityError("run receipt has an unexpected schema")
        converted = dict(raw)
        nodes = converted.get("ordered_nodes")
        endpoints = converted.get("ordered_gateway_endpoint_identities")
        probe_hashes = converted.get("ordered_probe_artifact_file_sha256s")
        log_hashes = converted.get("ordered_bounded_log_file_sha256s")
        if not all(
            isinstance(value, list) for value in (nodes, endpoints, probe_hashes, log_hashes)
        ):
            raise FeasibilityError("run receipt arrays are invalid")
        node_items = cast("list[object]", nodes)
        endpoint_items = cast("list[object]", endpoints)
        probe_items = cast("list[object]", probe_hashes)
        log_items = cast("list[object]", log_hashes)
        converted["ordered_nodes"] = tuple(node_items)
        converted["ordered_gateway_endpoint_identities"] = tuple(
            RuntimeServiceEndpointIdentity.from_dict(item) for item in endpoint_items
        )
        converted["build_service_endpoint_identity"] = RuntimeServiceEndpointIdentity.from_dict(
            converted["build_service_endpoint_identity"]
        )
        converted["ordered_probe_artifact_file_sha256s"] = tuple(probe_items)
        converted["ordered_bounded_log_file_sha256s"] = tuple(log_items)
        try:
            return cls(**converted)
        except (TypeError, ValueError) as error:
            raise FeasibilityError("run receipt contains invalid values") from error

    def validate(self, *, expected_node_count: int | None = None) -> None:
        """Validate exact F topology, proofs, endpoints, and self hash."""
        if self.schema_version != RUN_SCHEMA or self.cluster != "ptyche":
            raise FeasibilityError("run receipt schema or cluster is invalid")
        if self.node_count not in (1, 2, 16) or (
            expected_node_count is not None and self.node_count != expected_node_count
        ):
            raise FeasibilityError("run receipt node count mismatch")
        if (
            len(self.ordered_nodes) != self.node_count
            or len(set(self.ordered_nodes)) != self.node_count
        ):
            raise FeasibilityError("run receipt node tuple is invalid")
        if len(self.ordered_gateway_endpoint_identities) != self.node_count:
            raise FeasibilityError("run receipt gateway endpoint cardinality mismatch")
        for node, endpoint in zip(
            self.ordered_nodes,
            self.ordered_gateway_endpoint_identities,
            strict=True,
        ):
            endpoint.validate()
            if endpoint.service_kind != "gateway" or endpoint.endpoint_node_name != node:
                raise FeasibilityError("run receipt gateway endpoint order is invalid")
        self.build_service_endpoint_identity.validate()
        if self.build_service_endpoint_identity.service_kind != "build-service":
            raise FeasibilityError("run receipt build endpoint is invalid")
        if self.producer_uid in (
            self.build_service_endpoint_identity.expected_service_uid,
            *(
                endpoint.expected_service_uid
                for endpoint in self.ordered_gateway_endpoint_identities
            ),
        ):
            raise FeasibilityError("run receipt producer shares a service protection identity")
        if not _COMMIT_RE.fullmatch(self.producer_commit) or not _JOB_RE.fullmatch(self.job_id):
            raise FeasibilityError("run receipt commit or job ID is invalid")
        hashes = (
            self.feasibility_bound_blobs_sha256,
            self.global_srun_argv_sha256,
            self.collective_artifact_file_sha256,
            self.hermetic_builder_image_sha256,
            self.hermetic_builder_closure_manifest_sha256,
            self.pyxis_plugin_sha256,
            self.enroot_sha256,
            self.squashfuse_sha256,
            self.slurm_spank_manifest_sha256,
            self.enroot_system_config_manifest_sha256,
            *self.ordered_probe_artifact_file_sha256s,
            *self.ordered_bounded_log_file_sha256s,
        )
        if not all(_is_hash(value) for value in hashes):
            label = (
                "global argv"
                if not _is_hash(self.global_srun_argv_sha256)
                else "collective"
                if not _is_hash(self.collective_artifact_file_sha256)
                else "evidence"
            )
            raise FeasibilityError(f"run receipt {label} hash is invalid")
        proofs = (
            self.scm_rights_same_ofd_proven,
            self.sibling_pidfd_getfd_denied,
            self.protected_link_same_inode_proven,
            self.same_uid_write_break_proven,
            self.sigio_setup_order_proven,
            self.unnamed_direct_squashfuse_proven,
            self.live_backing_fd_same_inode_proven,
            self.persistent_rootfs_absent,
            self.global_go_abort_barrier_proven,
            self.remote_exit_proof_proven,
        )
        if not all(value is True for value in proofs):
            raise FeasibilityError("run receipt mandatory proof literal is false")
        if not self.one_global_execution_step_proven:
            raise FeasibilityError("run receipt must prove one global step")
        if self.global_rank_count != self.node_count:
            raise FeasibilityError("run receipt global rank count mismatch")
        if self.node_count == 16:
            if self.segment_count != 16 or not self.global_multinode_path_routing_proven:
                raise FeasibilityError("run receipt segment-16 proof is invalid")
        elif self.segment_count is not None:
            raise FeasibilityError("run receipt segment must be null for node count 1/2")
        if self.node_count in (2, 16) and not self.global_multinode_path_routing_proven:
            raise FeasibilityError("run receipt multinode path routing proof is invalid")
        if self.node_count == 1:
            if any(
                value is not None
                for value in (
                    self.prerequisite_node_count,
                    self.prerequisite_receipt_path,
                    self.prerequisite_receipt_file_sha256,
                )
            ):
                raise FeasibilityError("one-node run receipt must not have a prerequisite")
        else:
            expected_prerequisite = 1 if self.node_count == 2 else 2
            if (
                self.prerequisite_node_count != expected_prerequisite
                or self.prerequisite_receipt_path is None
                or not _is_absolute(self.prerequisite_receipt_path)
                or not _is_hash(self.prerequisite_receipt_file_sha256)
            ):
                raise FeasibilityError("run receipt prerequisite is invalid")
        if (
            self.yama_ptrace_scope < 1
            or self.lease_break_time_seconds < 15
            or not self.raid_filesystem_type
            or self.lustre_filesystem_type != "lustre"
            or not _is_absolute(self.protected_runtime_root)
            or self.empty_user_config_root_device < 0
            or self.empty_user_config_root_inode <= 0
        ):
            raise FeasibilityError("run receipt platform identity is invalid")
        if self.receipt_sha256 != canonical_sha256(self.body_dict()):
            raise FeasibilityError("run receipt self hash mismatch")


@dataclass(frozen=True)
class RuntimePlatformFeasibilityReceipt:
    """Canonical v3 local join over successful 1 -> 2 -> 16 run receipts."""

    schema_version: Literal["q30t-runtime-platform-feasibility-v3"]
    cluster: Literal["ptyche"]
    producer_commit: str
    feasibility_bound_blobs_sha256: str
    one_node_receipt_path: str
    one_node_receipt_file_sha256: str
    one_node_job_id: str
    two_node_receipt_path: str
    two_node_receipt_file_sha256: str
    two_node_job_id: str
    sixteen_node_receipt_path: str
    sixteen_node_receipt_file_sha256: str
    sixteen_node_job_id: str
    ordered_nodes: tuple[str, ...]
    ordered_gateway_endpoint_identities: tuple[RuntimeServiceEndpointIdentity, ...]
    protected_runtime_root: str
    build_service_endpoint_identity: RuntimeServiceEndpointIdentity
    hermetic_builder_image_path: str
    hermetic_builder_image_sha256: str
    hermetic_builder_closure_manifest_sha256: str
    ordered_run_receipt_sha256s: tuple[str, str, str]
    receipt_sha256: str

    def body_dict(self) -> dict[str, object]:
        """Return the aggregate self-hashed body."""
        return {
            field.name: _jsonable(getattr(self, field.name))
            for field in dataclasses.fields(self)
            if field.name != "receipt_sha256"
        }

    def to_dict(self) -> dict[str, object]:
        """Return one JSON-compatible aggregate receipt."""
        return self.body_dict() | {"receipt_sha256": self.receipt_sha256}

    def canonical_bytes(self) -> bytes:
        """Return exact canonical aggregate bytes."""
        return _canonical(self.to_dict())

    @classmethod
    def from_dict(cls, raw: object) -> RuntimePlatformFeasibilityReceipt:
        """Decode one exact v3 aggregate."""
        names = {field.name for field in dataclasses.fields(cls)}
        if not isinstance(raw, dict) or set(raw) != names:
            raise FeasibilityError("aggregate receipt has an unexpected schema")
        converted = dict(raw)
        for key in ("ordered_nodes", "ordered_run_receipt_sha256s"):
            if not isinstance(converted[key], list):
                raise FeasibilityError("aggregate receipt array is invalid")
            converted[key] = tuple(converted[key])
        endpoints = converted["ordered_gateway_endpoint_identities"]
        if not isinstance(endpoints, list):
            raise FeasibilityError("aggregate endpoint array is invalid")
        converted["ordered_gateway_endpoint_identities"] = tuple(
            RuntimeServiceEndpointIdentity.from_dict(item) for item in endpoints
        )
        converted["build_service_endpoint_identity"] = RuntimeServiceEndpointIdentity.from_dict(
            converted["build_service_endpoint_identity"]
        )
        try:
            return cls(**converted)
        except (TypeError, ValueError) as error:
            raise FeasibilityError("aggregate receipt contains invalid values") from error


_VERIFIED_PLATFORM_FEASIBILITY: dict[
    int,
    tuple[
        weakref.ReferenceType[RuntimePlatformFeasibilityReceipt],
        Path,
        str,
    ],
] = {}


def require_verified_platform_feasibility(feasibility: object) -> None:
    """Revalidate the exact live object returned by the physical aggregate loader."""
    registered = _VERIFIED_PLATFORM_FEASIBILITY.get(id(feasibility))
    if registered is None or registered[0]() is not feasibility:
        raise FeasibilityError("aggregate is not the exact stable-loaded F receipt")
    receipt_ref, path, file_sha256 = registered
    current = RuntimePlatformFeasibilityReceipt.from_dict(_stable_load_json(path, file_sha256))
    replayed = join_platform_feasibility_runs(
        one_node_receipt_path=Path(current.one_node_receipt_path),
        one_node_receipt_file_sha256=current.one_node_receipt_file_sha256,
        two_node_receipt_path=Path(current.two_node_receipt_path),
        two_node_receipt_file_sha256=current.two_node_receipt_file_sha256,
        sixteen_node_receipt_path=Path(current.sixteen_node_receipt_path),
        sixteen_node_receipt_file_sha256=current.sixteen_node_receipt_file_sha256,
    )
    if receipt_ref() is not feasibility or current != feasibility or replayed != current:
        raise FeasibilityError("stable-loaded F receipt provenance is no longer valid")


def _stable_load_json(path: Path, expected_file_sha256: str) -> object:
    if not path.is_absolute() or not _is_hash(expected_file_sha256):
        raise FeasibilityError("receipt path/hash is invalid")
    descriptor = os.open(path, os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW)
    try:
        before = os.fstat(descriptor)
        if (
            not stat.S_ISREG(before.st_mode)
            or before.st_nlink != 1
            or before.st_size > _MAX_RECEIPT_BYTES
        ):
            raise FeasibilityError("receipt must be a bounded single-link regular file")
        chunks: list[bytes] = []
        digest = hashlib.sha256()
        total = 0
        while block := os.read(descriptor, 1024 * 1024):
            total += len(block)
            if total > _MAX_RECEIPT_BYTES:
                raise FeasibilityError("receipt exceeds the bounded size")
            digest.update(block)
            chunks.append(block)
        after = os.fstat(descriptor)
        named = os.lstat(path)
        if (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns) != (
            after.st_dev,
            after.st_ino,
            after.st_size,
            after.st_mtime_ns,
        ) or (named.st_dev, named.st_ino) != (after.st_dev, after.st_ino):
            raise FeasibilityError("receipt changed during stable read")
        if digest.hexdigest() != expected_file_sha256:
            raise FeasibilityError("receipt physical SHA-256 mismatch")
        raw = b"".join(chunks)
        try:
            parsed = json.loads(raw)
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise FeasibilityError("receipt JSON is malformed") from error
        canonical_with_newline = _canonical(parsed) + b"\n"
        if raw not in (_canonical(parsed), canonical_with_newline):
            raise FeasibilityError("receipt is not canonical JSON")
        return parsed
    finally:
        os.close(descriptor)


def load_platform_feasibility_run_receipt(
    path: Path,
    expected_file_sha256: str,
    *,
    expected_node_count: Literal[1, 2, 16],
) -> RuntimePlatformFeasibilityRunReceipt:
    """Stable-load and validate one exact physical run receipt."""
    receipt = RuntimePlatformFeasibilityRunReceipt.from_dict(
        _stable_load_json(path, expected_file_sha256)
    )
    receipt.validate(expected_node_count=expected_node_count)
    return receipt


def _validate_shared_endpoint_identity(
    runs: tuple[RuntimePlatformFeasibilityRunReceipt, ...],
) -> None:
    by_node: dict[str, RuntimeServiceEndpointIdentity] = {}
    for run in runs:
        for node, endpoint in zip(
            run.ordered_nodes, run.ordered_gateway_endpoint_identities, strict=True
        ):
            prior = by_node.setdefault(node, endpoint)
            if prior != endpoint:
                raise FeasibilityError("shared gateway endpoint identity changed across F runs")


def join_platform_feasibility_runs(
    *,
    one_node_receipt_path: Path,
    one_node_receipt_file_sha256: str,
    two_node_receipt_path: Path,
    two_node_receipt_file_sha256: str,
    sixteen_node_receipt_path: Path,
    sixteen_node_receipt_file_sha256: str,
) -> RuntimePlatformFeasibilityReceipt:
    """Join the three successful prerequisite-ordered F jobs locally."""
    one = load_platform_feasibility_run_receipt(
        one_node_receipt_path,
        one_node_receipt_file_sha256,
        expected_node_count=1,
    )
    two = load_platform_feasibility_run_receipt(
        two_node_receipt_path,
        two_node_receipt_file_sha256,
        expected_node_count=2,
    )
    sixteen = load_platform_feasibility_run_receipt(
        sixteen_node_receipt_path,
        sixteen_node_receipt_file_sha256,
        expected_node_count=16,
    )
    runs = (one, two, sixteen)
    if len({run.job_id for run in runs}) != 3:
        raise FeasibilityError("F aggregate requires three distinct real job IDs")
    if (
        two.prerequisite_receipt_path != str(one_node_receipt_path)
        or two.prerequisite_receipt_file_sha256 != one_node_receipt_file_sha256
        or sixteen.prerequisite_receipt_path != str(two_node_receipt_path)
        or sixteen.prerequisite_receipt_file_sha256 != two_node_receipt_file_sha256
    ):
        raise FeasibilityError("F aggregate prerequisite chain is invalid")
    if (
        len({run.producer_commit for run in runs}) != 1
        or len({run.feasibility_bound_blobs_sha256 for run in runs}) != 1
    ):
        raise FeasibilityError("F aggregate producer lineage changed across runs")
    if len({run.protected_runtime_root for run in runs}) != 1:
        raise FeasibilityError("F aggregate protected runtime root changed across runs")
    if any(
        run.build_service_endpoint_identity != one.build_service_endpoint_identity for run in runs
    ):
        raise FeasibilityError("F aggregate build coordinator identity changed across runs")
    if any(
        (
            run.hermetic_builder_image_path,
            run.hermetic_builder_image_sha256,
            run.hermetic_builder_closure_manifest_sha256,
        )
        != (
            one.hermetic_builder_image_path,
            one.hermetic_builder_image_sha256,
            one.hermetic_builder_closure_manifest_sha256,
        )
        for run in runs
    ):
        raise FeasibilityError("F aggregate hermetic builder identity changed across runs")
    _validate_shared_endpoint_identity(runs)
    fields = {
        "build_service_endpoint_identity": one.build_service_endpoint_identity,
        "cluster": "ptyche",
        "feasibility_bound_blobs_sha256": one.feasibility_bound_blobs_sha256,
        "hermetic_builder_closure_manifest_sha256": one.hermetic_builder_closure_manifest_sha256,
        "hermetic_builder_image_path": one.hermetic_builder_image_path,
        "hermetic_builder_image_sha256": one.hermetic_builder_image_sha256,
        "one_node_job_id": one.job_id,
        "one_node_receipt_file_sha256": one_node_receipt_file_sha256,
        "one_node_receipt_path": str(one_node_receipt_path),
        "ordered_gateway_endpoint_identities": sixteen.ordered_gateway_endpoint_identities,
        "ordered_nodes": sixteen.ordered_nodes,
        "ordered_run_receipt_sha256s": (
            one.receipt_sha256,
            two.receipt_sha256,
            sixteen.receipt_sha256,
        ),
        "producer_commit": one.producer_commit,
        "protected_runtime_root": one.protected_runtime_root,
        "schema_version": AGGREGATE_SCHEMA,
        "sixteen_node_job_id": sixteen.job_id,
        "sixteen_node_receipt_file_sha256": sixteen_node_receipt_file_sha256,
        "sixteen_node_receipt_path": str(sixteen_node_receipt_path),
        "two_node_job_id": two.job_id,
        "two_node_receipt_file_sha256": two_node_receipt_file_sha256,
        "two_node_receipt_path": str(two_node_receipt_path),
    }
    return RuntimePlatformFeasibilityReceipt(
        receipt_sha256=canonical_sha256(fields),
        **fields,
    )


def load_platform_feasibility_receipt(
    path: Path,
    expected_file_sha256: str,
) -> RuntimePlatformFeasibilityReceipt:
    """Stable-load the v3 aggregate and replay all three physical runs."""
    receipt = RuntimePlatformFeasibilityReceipt.from_dict(
        _stable_load_json(path, expected_file_sha256)
    )
    if receipt.schema_version != AGGREGATE_SCHEMA or receipt.cluster != "ptyche":
        raise FeasibilityError("aggregate receipt schema or cluster is invalid")
    if receipt.receipt_sha256 != canonical_sha256(receipt.body_dict()):
        raise FeasibilityError("aggregate receipt self hash mismatch")
    replayed = join_platform_feasibility_runs(
        one_node_receipt_path=Path(receipt.one_node_receipt_path),
        one_node_receipt_file_sha256=receipt.one_node_receipt_file_sha256,
        two_node_receipt_path=Path(receipt.two_node_receipt_path),
        two_node_receipt_file_sha256=receipt.two_node_receipt_file_sha256,
        sixteen_node_receipt_path=Path(receipt.sixteen_node_receipt_path),
        sixteen_node_receipt_file_sha256=receipt.sixteen_node_receipt_file_sha256,
    )
    if replayed != receipt:
        raise FeasibilityError("aggregate receipt does not replay the physical F chain")
    object_id = id(receipt)

    def discard(stale: weakref.ReferenceType[RuntimePlatformFeasibilityReceipt]) -> None:
        registered = _VERIFIED_PLATFORM_FEASIBILITY.get(object_id)
        if registered is not None and registered[0] is stale:
            _VERIFIED_PLATFORM_FEASIBILITY.pop(object_id, None)

    _VERIFIED_PLATFORM_FEASIBILITY[object_id] = (
        weakref.ref(receipt, discard),
        path,
        expected_file_sha256,
    )
    return receipt


def build_global_srun_argv(
    *,
    node_count: int,
    protected_runtime_path: Path,
    control_manifest_sha256: str,
    operation: str,
    canonical_ro_rprivate_mounts: tuple[str, ...],
) -> tuple[str, ...]:
    """Construct the exact single global Pyxis argv for one F operation."""
    if node_count not in (1, 2, 16) or not protected_runtime_path.is_absolute():
        raise FeasibilityError("global srun topology or image path is invalid")
    if not _is_hash(control_manifest_sha256) or operation != "platform-feasibility":
        raise FeasibilityError("global srun manifest or operation is invalid")
    if not canonical_ro_rprivate_mounts or any(
        not mount.endswith(":ro+rprivate") for mount in canonical_ro_rprivate_mounts
    ):
        raise FeasibilityError("global srun mount tuple is invalid")
    return (
        "/usr/bin/srun",
        "--overlap",
        f"--nodes={node_count}",
        f"--ntasks={node_count}",
        "--ntasks-per-node=1",
        "--gpus-per-node=4",
        f"--container-image={protected_runtime_path}",
        "--container-readonly",
        "--no-container-mount-home",
        "--no-container-remap-root",
        "--no-container-entrypoint",
        f"--container-mounts={','.join(canonical_ro_rprivate_mounts)}",
        "--container-workdir=/run/q30t/work",
        "/opt/q30t-runtime/bin/python",
        "-I",
        "-S",
        "/run/q30t/control/q30t_runtime_bootstrap.py",
        "--manifest",
        "/run/q30t/control/held-modules.json",
        "--manifest-sha256",
        control_manifest_sha256,
        "--entry-module",
        "common.specdec.ptv23_runtime_attestation",
        "--operation",
        operation,
    )


def build_platform_feasibility_submission_argv(
    *,
    node_count: int,
    prerequisite_receipt_path: Path | None,
    prerequisite_receipt_file_sha256: str | None,
    submit_mode: Literal["test-only", "parsable"],
) -> tuple[str, ...]:
    """Construct one sterile exact F sbatch tuple."""
    if node_count not in (1, 2, 16):
        raise FeasibilityError("explicit node count must be 1, 2, or 16")
    if node_count == 1:
        if prerequisite_receipt_path is not None or prerequisite_receipt_file_sha256 is not None:
            raise FeasibilityError("one-node submission cannot have prerequisite arguments")
    elif (
        prerequisite_receipt_path is None
        or not prerequisite_receipt_path.is_absolute()
        or not _is_hash(prerequisite_receipt_file_sha256)
    ):
        raise FeasibilityError(
            "dependent submission requires an absolute receipt and physical hash"
        )
    if submit_mode not in ("test-only", "parsable"):
        raise FeasibilityError("submission mode is invalid")
    runner = Path(__file__).with_name("run_q30t_runtime_platform_feasibility.sbatch")
    argv = [
        "/usr/bin/sbatch",
        f"--{submit_mode}",
        "--export=NONE",
        f"--nodes={node_count}",
        f"--ntasks={node_count}",
        "--ntasks-per-node=1",
        "--gpus-per-node=4",
        "--comment=q30t-runtime-platform-feasibility",
        "--output=/raid/scratch/%u/q30t-runtime-logs/q30t-runtime-%j.batch.log",
        str(runner),
        "--platform-feasibility",
        "--node-count",
        str(node_count),
    ]
    if prerequisite_receipt_path is not None:
        prefix = "one" if node_count == 2 else "two"
        argv.extend(
            (
                f"--{prefix}-node-receipt",
                str(prerequisite_receipt_path),
                f"--{prefix}-node-receipt-sha256",
                cast("str", prerequisite_receipt_file_sha256),
            )
        )
    return tuple(argv)


def execute_test_only_first(
    test_only_argv: tuple[str, ...],
    *,
    invoke: Callable[[tuple[str, ...]], str],
) -> str:
    """Run test-only, then the byte-identical parsable tuple exactly once."""
    if len(test_only_argv) < 2 or test_only_argv[1] != "--test-only":
        raise FeasibilityError("test-only argv is invalid")
    try:
        invoke(test_only_argv)
    except (OSError, subprocess.CalledProcessError) as error:
        raise FeasibilityError("sbatch test-only failed; real submission suppressed") from error
    parsable_argv = (test_only_argv[0], "--parsable", *test_only_argv[2:])
    try:
        output = invoke(parsable_argv).strip()
    except (OSError, subprocess.CalledProcessError) as error:
        raise FeasibilityError("real parsable sbatch submission failed") from error
    job_id = output.split(";", 1)[0]
    if not _JOB_RE.fullmatch(job_id):
        raise FeasibilityError("parsable sbatch did not return a canonical real job ID")
    return job_id


def _submit(arguments: argparse.Namespace) -> int:
    prerequisite_path: Path | None = None
    prerequisite_hash: str | None = None
    if arguments.node_count == 2:
        prerequisite_path = arguments.one_node_receipt
        prerequisite_hash = arguments.one_node_receipt_sha256
        if prerequisite_path is None or prerequisite_hash is None:
            raise FeasibilityError("one-node prerequisite path/hash is required")
        load_platform_feasibility_run_receipt(
            prerequisite_path,
            prerequisite_hash,
            expected_node_count=1,
        )
    elif arguments.node_count == 16:
        prerequisite_path = arguments.two_node_receipt
        prerequisite_hash = arguments.two_node_receipt_sha256
        if prerequisite_path is None or prerequisite_hash is None:
            raise FeasibilityError("two-node prerequisite path/hash is required")
        load_platform_feasibility_run_receipt(
            prerequisite_path,
            prerequisite_hash,
            expected_node_count=2,
        )
    test_argv = build_platform_feasibility_submission_argv(
        node_count=arguments.node_count,
        prerequisite_receipt_path=prerequisite_path,
        prerequisite_receipt_file_sha256=prerequisite_hash,
        submit_mode="test-only",
    )
    if arguments.dry_run:
        print(json.dumps(list(test_argv)))
        print(json.dumps([test_argv[0], "--parsable", *test_argv[2:]]))
        return 0

    def invoke(argv: tuple[str, ...]) -> str:
        result = subprocess.run(
            argv,
            check=True,
            capture_output=True,
            text=True,
            env={"LC_ALL": "C", "PATH": "/usr/bin:/bin"},
        )
        return result.stdout

    print(execute_test_only_first(test_argv, invoke=invoke))
    return 0


def _run(arguments: argparse.Namespace) -> int:
    # A local host can only fail closed; the live installed gateway/build service
    # produces receipts through Task 1 on Ptyche.
    if arguments.node_count not in (1, 2, 16):
        raise FeasibilityError("runner node count is invalid")
    print("BLOCKED: live Ptyche administrator gateway/build service probe is required")
    return 2


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command in ("submit", "run"):
        child = subparsers.add_parser(command)
        child.add_argument("--platform-feasibility", action="store_true", required=True)
        child.add_argument("--node-count", type=int, choices=(1, 2, 16), required=True)
        child.add_argument("--one-node-receipt", type=Path)
        child.add_argument("--one-node-receipt-sha256")
        child.add_argument("--two-node-receipt", type=Path)
        child.add_argument("--two-node-receipt-sha256")
        child.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the F-only submit or live-probe CLI."""
    parser = _parser()
    arguments = parser.parse_args(argv)
    if arguments.node_count == 1 and any(
        value is not None
        for value in (
            arguments.one_node_receipt,
            arguments.one_node_receipt_sha256,
            arguments.two_node_receipt,
            arguments.two_node_receipt_sha256,
        )
    ):
        parser.error("node count 1 accepts no prerequisite arguments")
    if arguments.node_count == 2 and (
        arguments.one_node_receipt is None
        or arguments.one_node_receipt_sha256 is None
        or arguments.two_node_receipt is not None
        or arguments.two_node_receipt_sha256 is not None
    ):
        parser.error("node count 2 requires exactly the one-node receipt path/hash")
    if arguments.node_count == 16 and (
        arguments.two_node_receipt is None
        or arguments.two_node_receipt_sha256 is None
        or arguments.one_node_receipt is not None
        or arguments.one_node_receipt_sha256 is not None
    ):
        parser.error("node count 16 requires exactly the two-node receipt path/hash")
    return _submit(arguments) if arguments.command == "submit" else _run(arguments)


if __name__ == "__main__":
    raise SystemExit(main())
