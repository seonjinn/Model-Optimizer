# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# ruff: noqa: D102, D103, TC003

"""Hostile tests for the Q30T platform-feasibility producer."""

from __future__ import annotations

import dataclasses
import fcntl
import os
import socket
import sys
from contextlib import ExitStack, closing
from dataclasses import dataclass
from pathlib import Path

import pytest
from common.specdec.q30t_runtime_gateway_client import ProtectedImagePublication
from common.specdec.q30t_runtime_platform_feasibility import (
    ConfigManifestEntry,
    DirectPyxisTrace,
    FeasibilityError,
    KeeperSameOFDReady,
    MultinodeFeasibilityTrace,
    PyxisDirectMountAttestation,
    RecursiveConfigManifest,
    RemoteStepExitProof,
    StagingAgentTrace,
    assert_sibling_pidfd_getfd_denied,
    canonical_sha256,
    prove_keeper_same_ofd,
    read_process_start_ticks,
    receive_keeper_same_ofd,
    require_pidfd_identity,
    validate_direct_mount_attestations,
    validate_direct_pyxis_trace,
    validate_multinode_feasibility_trace,
    validate_recursive_config_manifest,
    validate_remote_step_exit_proof,
)

_LINUX = pytest.mark.skipif(sys.platform != "linux", reason="requires Linux descriptor syscalls")


@dataclass
class LeasedTmpfileFixture:
    """One retained unlinked inode and its descriptor-transfer channel."""

    fd: int
    pid: int
    ready: KeeperSameOFDReady
    receiver: socket.socket
    sender: socket.socket

    def close(self) -> None:
        self.receiver.close()
        self.sender.close()
        os.close(self.fd)


def _ready_for_fd(fd: int) -> KeeperSameOFDReady:
    metadata = os.fstat(fd)
    body = {
        "all_threads_sigio_blocked": True,
        "all_writer_descriptors_closed": True,
        "job_id": "12345",
        "keeper_gid": os.getgid(),
        "keeper_pid": os.getpid(),
        "keeper_start_ticks": read_process_start_ticks(os.getpid()),
        "keeper_uid": os.getuid(),
        "lease_break_time_seconds": 15,
        "lease_signal": "SIGIO",
        "lease_type": "F_RDLCK",
        "node_name": "ptyche-n001",
        "operation_id": "a" * 64,
        "schema_version": "ptv23-keeper-same-ofd-ready-v1",
        "signal_owner_pid": os.getpid(),
        "signal_owner_type": "F_OWNER_PID",
        "sigwait_thread_armed_before_lease": True,
        "staged_device": metadata.st_dev,
        "staged_inode": metadata.st_ino,
        "staged_mode": 0o400,
        "staged_nlink_before_gateway": 0,
        "staged_sha256": "0" * 64,
        "staged_size": metadata.st_size,
    }
    return KeeperSameOFDReady(packet_sha256=canonical_sha256(body), **body)


def make_leased_tmpfile_fixture(tmp_path: Path) -> LeasedTmpfileFixture:
    """Create one unlinked inode whose retained read OFD owns a read lease."""
    named = tmp_path / "runtime.sqsh"
    named.write_bytes(b"q30t-runtime")
    named.chmod(0o400)
    fd = os.open(named, os.O_RDONLY)
    named.unlink()
    try:
        fcntl.fcntl(fd, fcntl.F_SETLEASE, fcntl.F_RDLCK)
    except OSError as error:
        os.close(fd)
        pytest.skip(f"host filesystem does not support file leases: {error}")
    receiver, sender = socket.socketpair(
        socket.AF_UNIX,
        socket.SOCK_SEQPACKET | socket.SOCK_CLOEXEC,
    )
    sender.sendmsg([b"fd"], [(socket.SOL_SOCKET, socket.SCM_RIGHTS, fd.to_bytes(4, sys.byteorder))])
    return LeasedTmpfileFixture(fd, os.getpid(), _ready_for_fd(fd), receiver, sender)


def receive_one_fd(receiver: socket.socket) -> int:
    """Receive exactly one descriptor from a test seqpacket."""
    _, ancillary, _, _ = receiver.recvmsg(16, socket.CMSG_SPACE(4))
    assert len(ancillary) == 1
    level, kind, data = ancillary[0]
    assert (level, kind) == (socket.SOL_SOCKET, socket.SCM_RIGHTS)
    return int.from_bytes(data[:4], sys.byteorder)


@_LINUX
def test_scm_rights_duplicate_observes_keeper_lease(tmp_path: Path) -> None:
    fixture = make_leased_tmpfile_fixture(tmp_path)
    with ExitStack() as stack:
        stack.callback(fixture.close)
        received_fd = receive_one_fd(fixture.receiver)
        stack.callback(os.close, received_fd)
        assert fcntl.fcntl(received_fd, fcntl.F_GETLEASE) == fcntl.F_RDLCK
        assert prove_keeper_same_ofd(received_fd, fixture.ready) == fixture.ready


@_LINUX
def test_procfd_reopen_is_rejected_as_same_ofd_proof(tmp_path: Path) -> None:
    fixture = make_leased_tmpfile_fixture(tmp_path)
    with ExitStack() as stack:
        stack.callback(fixture.close)
        reopened_fd = os.open(f"/proc/{fixture.pid}/fd/{fixture.fd}", os.O_RDONLY)
        stack.callback(os.close, reopened_fd)
        with pytest.raises(FeasibilityError, match="same open file description"):
            prove_keeper_same_ofd(reopened_fd, fixture.ready)


@_LINUX
def test_pidfd_rejects_pid_mismatch_and_reused_start_ticks() -> None:
    with closing(os.pidfd_open(os.getpid())) as pidfd:
        start_ticks = read_process_start_ticks(os.getpid())
        with pytest.raises(FeasibilityError, match="PID does not match"):
            require_pidfd_identity(pidfd.fileno(), os.getpid() + 1, start_ticks)
        with pytest.raises(FeasibilityError, match="start ticks"):
            require_pidfd_identity(pidfd.fileno(), os.getpid(), start_ticks + 1)
        require_pidfd_identity(pidfd.fileno(), os.getpid(), start_ticks)


@_LINUX
def test_sibling_pidfd_getfd_attack_is_denied_or_syscall_is_unsupported() -> None:
    try:
        assert_sibling_pidfd_getfd_denied()
    except NotImplementedError as error:
        pytest.skip(str(error))


@_LINUX
def test_scm_credentials_come_from_post_spawn_child_not_cached_peercred() -> None:
    receiver, sender = socket.socketpair(socket.AF_UNIX, socket.SOCK_SEQPACKET)
    with receiver, sender:
        receiver.setsockopt(socket.SOL_SOCKET, socket.SO_PASSCRED, 1)
        creator_pid = os.getpid()
        cached_pid = receiver.getsockopt(
            socket.SOL_SOCKET,
            socket.SO_PEERCRED,
            12,
        )[:4]
        cached_pid = int.from_bytes(cached_pid, sys.byteorder)
        assert cached_pid == creator_pid

        child_pid = os.fork()
        if child_pid == 0:
            try:
                receiver.close()
                child_fd = os.open("/dev/null", os.O_RDONLY)
                ready = _ready_for_fd(child_fd)
                payload = ready.canonical_bytes()
                sender.sendmsg(
                    [payload],
                    [(socket.SOL_SOCKET, socket.SCM_RIGHTS, child_fd.to_bytes(4, sys.byteorder))],
                )
                os.close(child_fd)
            finally:
                os._exit(0)

        sender.close()
        with closing(os.pidfd_open(child_pid)) as pidfd:
            start_ticks = read_process_start_ticks(child_pid)
            ready, received_fd = receive_keeper_same_ofd(
                receiver,
                expected_keeper_pidfd=pidfd.fileno(),
                expected_keeper_start_ticks=start_ticks,
                expected_keeper_uid=os.getuid(),
                expected_keeper_gid=os.getgid(),
                require_lease=False,
            )
            os.close(received_fd)
        _, status = os.waitpid(child_pid, 0)
        assert status == 0
        assert ready.keeper_pid == child_pid
        assert (ready.keeper_uid, ready.keeper_gid) == (os.getuid(), os.getgid())


@_LINUX
@pytest.mark.parametrize("mutation", ["missing", "duplicate", "mismatch"])
def test_scm_credentials_reject_missing_duplicate_or_mismatched_records(
    mutation: str,
) -> None:
    with pytest.raises(FeasibilityError, match="SCM_CREDENTIALS"):
        receive_keeper_same_ofd.decode_ancillary_for_test(
            credential_records=()
            if mutation == "missing"
            else ((os.getpid(), os.getuid(), os.getgid()),) * 2
            if mutation == "duplicate"
            else ((os.getpid() + 1, os.getuid(), os.getgid()),),
            rights_records=((0,),),
            expected_pid=os.getpid(),
            expected_uid=os.getuid(),
            expected_gid=os.getgid(),
        )


def _direct_trace() -> DirectPyxisTrace:
    return DirectPyxisTrace(
        ordered_log_lines=("skipping container creation (squashfuse enabled)",),
        container_image="/run/q30t-protected/12345/" + "a" * 64 + "/runtime.sqsh",
        container_name=None,
        container_save=None,
        enroot_data_path=None,
        persistent_rootfs_paths=(),
        nested_execution_count=0,
    )


def test_direct_pyxis_requires_exact_positive_line_and_unnamed_temporary_root() -> None:
    validate_direct_pyxis_trace(_direct_trace())
    with pytest.raises(FeasibilityError, match="direct SquashFUSE"):
        validate_direct_pyxis_trace(
            dataclasses.replace(_direct_trace(), ordered_log_lines=("some squashfuse log",))
        )


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        (
            "ordered_log_lines",
            (
                "skipping container creation (squashfuse enabled)",
                "creating container filesystem",
            ),
            "container creation",
        ),
        ("container_name", "reusable", "named root"),
        ("container_save", "/tmp/rootfs", "named root"),
        ("enroot_data_path", "/tmp/enroot", "persistent ENROOT_DATA_PATH"),
        ("container_image", "/proc/123/fd/4", "procfd"),
        ("persistent_rootfs_paths", ("/tmp/rootfs",), "persistent rootfs"),
        ("nested_execution_count", 1, "nested execution"),
    ],
)
def test_direct_pyxis_rejects_container_creation_named_procfd_or_persistence(
    field: str,
    value: object,
    message: str,
) -> None:
    with pytest.raises(FeasibilityError, match=message):
        validate_direct_pyxis_trace(dataclasses.replace(_direct_trace(), **{field: value}))


def _publication(node: str, *, device: int, inode: int) -> ProtectedImagePublication:
    return ProtectedImagePublication(
        schema_version="q30t-protected-image-publication-v1",
        job_id="12345",
        node_name=node,
        operation_id="a" * 64,
        producer_uid=1000,
        protected_path="/run/q30t-protected/12345/" + "a" * 64 + "/runtime.sqsh",
        device=device,
        inode=inode,
        nlink=1,
        mode=0o400,
        size=10,
        sha256="b" * 64,
        same_ofd_lease_validated=True,
        gateway_signature="signed",
    )


def _mount_attestation(
    node: str,
    *,
    device: int,
    inode: int,
    pid: int,
    mount_id: int,
) -> PyxisDirectMountAttestation:
    return PyxisDirectMountAttestation(
        schema_version="q30t-pyxis-direct-mount-attestation-v1",
        job_id="12345",
        step_id="0",
        node_name=node,
        operation_id="a" * 64,
        protected_path="/run/q30t-protected/12345/" + "a" * 64 + "/runtime.sqsh",
        image_device=device,
        image_inode=inode,
        squashfuse_pid=pid,
        squashfuse_start_ticks=pid * 10,
        squashfuse_cgroup_relative_path=f"/slurm/12345/step_0/{pid}",
        backing_fd_number=7,
        backing_fd_access_mode="O_RDONLY",
        backing_fd_device=device,
        backing_fd_inode=inode,
        root_mount_id=mount_id,
        root_mount_readonly=True,
        temporary_rootfs=True,
        container_creation_absent=True,
        persistent_rootfs_absent=True,
        gateway_signature="signed",
    )


def test_direct_mount_requires_gateway_signed_live_backing_fd_same_inode() -> None:
    publications = (
        _publication("ptyche-n001", device=10, inode=20),
        _publication("ptyche-n002", device=11, inode=21),
    )
    attestations = (
        _mount_attestation("ptyche-n001", device=10, inode=20, pid=101, mount_id=301),
        _mount_attestation("ptyche-n002", device=11, inode=21, pid=102, mount_id=302),
    )
    validate_direct_mount_attestations(attestations, publications)
    with pytest.raises(FeasibilityError, match="backing fd"):
        validate_direct_mount_attestations(
            (dataclasses.replace(attestations[0], backing_fd_inode=99), attestations[1]),
            publications,
        )
    with pytest.raises(FeasibilityError, match="fresh SquashFUSE"):
        validate_direct_mount_attestations(
            (attestations[0], dataclasses.replace(attestations[1], squashfuse_pid=101)),
            publications,
        )


def _multinode_trace(node_count: int) -> MultinodeFeasibilityTrace:
    nodes = tuple(f"ptyche-n{index:03d}" for index in range(1, node_count + 1))
    path = "/run/q30t-protected/12345/" + "a" * 64 + "/runtime.sqsh"
    agents = tuple(
        StagingAgentTrace(
            node_name=node,
            agent_pid=1000 + index,
            keeper_pid=2000 + index,
            leased_fd=10 + index,
            device=20 + index,
            inode=30 + index,
            endpoint_identity_sha256=f"{index + 1:064x}",
            protected_path=path,
        )
        for index, node in enumerate(nodes)
    )
    return MultinodeFeasibilityTrace(
        node_count=node_count,
        ordered_nodes=nodes,
        ordered_staging_agents=agents,
        global_srun_argv=(
            "/usr/bin/srun",
            "--overlap",
            f"--nodes={node_count}",
            f"--ntasks={node_count}",
            "--ntasks-per-node=1",
            "--gpus-per-node=4",
            f"--container-image={path}",
        ),
        global_execution_step_count=1,
        global_rank_count=node_count,
        ordered_global_ranks=tuple(range(node_count)),
        collective_token_sha256="f" * 64,
        segment_count=16 if node_count == 16 else None,
        canary_full_feasibility=node_count == 16,
    )


@pytest.mark.parametrize("node_count", [2, 16])
def test_multinode_global_step_has_distinct_local_agents_and_one_shared_path(
    node_count: int,
) -> None:
    validate_multinode_feasibility_trace(_multinode_trace(node_count))


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda trace: dataclasses.replace(trace, global_execution_step_count=2), "one global step"),
        (lambda trace: dataclasses.replace(trace, global_rank_count=15), "rank count"),
        (lambda trace: dataclasses.replace(trace, node_count=15), "node count"),
        (
            lambda trace: dataclasses.replace(
                trace,
                ordered_staging_agents=(
                    trace.ordered_staging_agents[0],
                    dataclasses.replace(
                        trace.ordered_staging_agents[1],
                        endpoint_identity_sha256=trace.ordered_staging_agents[0].endpoint_identity_sha256,
                    ),
                    *trace.ordered_staging_agents[2:],
                ),
            ),
            "endpoint reuse",
        ),
    ],
)
def test_multinode_rejects_nested_rank_node_or_endpoint_reuse(
    mutation: object,
    message: str,
) -> None:
    with pytest.raises(FeasibilityError, match=message):
        validate_multinode_feasibility_trace(mutation(_multinode_trace(16)))


def test_multinode_two_node_trace_cannot_be_relabelled_as_segment_sixteen() -> None:
    with pytest.raises(FeasibilityError, match="segment-16"):
        validate_multinode_feasibility_trace(
            dataclasses.replace(
                _multinode_trace(2),
                segment_count=16,
                canary_full_feasibility=True,
            )
        )


def _config_manifest() -> RecursiveConfigManifest:
    entries = (
        ConfigManifestEntry("etc/enroot/enroot.conf", "regular", 0o644, 0, 0, 6, "1" * 64),
        ConfigManifestEntry("etc/enroot/hooks.d/50-slurm.sh", "regular", 0o755, 0, 0, 7, "2" * 64),
        ConfigManifestEntry("etc/enroot/mounts.d/10-site.fstab", "regular", 0o644, 0, 0, 8, "3" * 64),
        ConfigManifestEntry("etc/enroot/environ.d/10-site.env", "regular", 0o644, 0, 0, 9, "4" * 64),
        ConfigManifestEntry("usr/lib/slurm/spank_pyxis.so", "regular", 0o755, 0, 0, 10, "5" * 64),
        ConfigManifestEntry("usr/bin/enroot", "regular", 0o755, 0, 0, 11, "6" * 64),
        ConfigManifestEntry("usr/bin/squashfuse", "regular", 0o755, 0, 0, 12, "7" * 64),
    )
    return RecursiveConfigManifest(
        ordered_entries=entries,
        manifest_sha256=canonical_sha256([dataclasses.asdict(entry) for entry in entries]),
        empty_user_config_root_device=91,
        empty_user_config_root_inode=92,
        home="/run/q30t-empty-config",
        xdg_config_home="/run/q30t-empty-config",
    )


def test_config_manifest_rejects_extra_changed_system_or_user_config() -> None:
    reviewed = _config_manifest()
    validate_recursive_config_manifest(reviewed, reviewed)
    extra = ConfigManifestEntry("etc/enroot/hooks.d/99-extra.sh", "regular", 0o755, 0, 0, 1, "8" * 64)
    with pytest.raises(FeasibilityError, match="config manifest"):
        validate_recursive_config_manifest(
            dataclasses.replace(reviewed, ordered_entries=(*reviewed.ordered_entries, extra)),
            reviewed,
        )
    with pytest.raises(FeasibilityError, match="gateway-owned empty"):
        validate_recursive_config_manifest(
            dataclasses.replace(reviewed, home="/home/producer", xdg_config_home="/home/producer/.config"),
            reviewed,
        )


def _remote_exit_proof() -> RemoteStepExitProof:
    values = {
        "allocation_state": "RUNNING",
        "authenticated_exit_nodes": ("ptyche-n001", "ptyche-n002"),
        "cancel_elapsed_ms": 4000,
        "every_cgroup_populated_zero": True,
        "exact_step_terminal_state": "CANCELLED",
        "job_id": "12345",
        "local_srun_returncode": -9,
        "ordered_cgroup_identities": (
            ("ptyche-n001", 10, 20),
            ("ptyche-n002", 11, 21),
        ),
        "ordered_nodes": ("ptyche-n001", "ptyche-n002"),
        "ordered_wrapper_pid_start": (
            ("ptyche-n001", 100, 200),
            ("ptyche-n002", 101, 201),
        ),
        "proof_completed_monotonic_ns": 14_000_000_000,
        "schema_version": "q30t-remote-step-exit-proof-v1",
        "step_id": "0",
    }
    return RemoteStepExitProof(proof_sha256=canonical_sha256(values), **values)


def test_remote_exit_requires_exact_step_cgroup_empty_reap_and_allocation() -> None:
    validate_remote_step_exit_proof(
        _remote_exit_proof(),
        expected_job_id="12345",
        expected_step_id="0",
        expected_nodes=("ptyche-n001", "ptyche-n002"),
        lease_acquired_monotonic_ns=10_000_000_000,
        lease_break_time_seconds=15,
    )


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("step_id", "1", "exact step"),
        ("every_cgroup_populated_zero", False, "cgroup"),
        ("local_srun_returncode", 0, "srun reaped"),
        ("exact_step_terminal_state", "RUNNING", "terminal"),
        ("allocation_state", "", "allocation"),
        ("cancel_elapsed_ms", 5000, "less than five seconds"),
    ],
)
def test_remote_exit_rejects_incomplete_cgroup_scheduler_or_allocation_proof(
    field: str,
    value: object,
    message: str,
) -> None:
    with pytest.raises(FeasibilityError, match=message):
        validate_remote_step_exit_proof(
            dataclasses.replace(_remote_exit_proof(), **{field: value}),
            expected_job_id="12345",
            expected_step_id="0",
            expected_nodes=("ptyche-n001", "ptyche-n002"),
            lease_acquired_monotonic_ns=10_000_000_000,
            lease_break_time_seconds=15,
        )


def test_remote_exit_deadline_equality_is_rejected() -> None:
    with pytest.raises(FeasibilityError, match="before lease deadline"):
        validate_remote_step_exit_proof(
            dataclasses.replace(
                _remote_exit_proof(),
                proof_completed_monotonic_ns=20_000_000_000,
            ),
            expected_job_id="12345",
            expected_step_id="0",
            expected_nodes=("ptyche-n001", "ptyche-n002"),
            lease_acquired_monotonic_ns=10_000_000_000,
            lease_break_time_seconds=15,
        )
