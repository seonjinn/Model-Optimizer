# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# ruff: noqa: D102, D103

"""Hostile tests for the Q30T platform-feasibility producer."""

from __future__ import annotations

import base64
import dataclasses
import fcntl
import gc
import hashlib
import os
import shutil
import socket
import subprocess
import sys
import tempfile
from contextlib import ExitStack, closing
from dataclasses import dataclass
from pathlib import Path

import pytest
from common.specdec import q30t_runtime_gateway_client as gateway_module
from common.specdec.q30t_runtime_gateway_client import (
    ProtectedImagePublication,
    RuntimeServiceEndpointIdentity,
    reconstruct_runtime_gateway_endpoint,
)
from common.specdec.q30t_runtime_platform_feasibility import (
    CONTROL_LOG_CAP_BYTES,
    BoundedArtifactCollector,
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


def _endpoint_identity(
    node: str, index: int, *, build: bool = False
) -> RuntimeServiceEndpointIdentity:
    values = {
        "control_socket_device": 100 + index,
        "control_socket_gid": 1974,
        "control_socket_inode": 200 + index,
        "control_socket_mode": 0o660,
        "control_socket_path": f"/run/q30t/{'build' if build else node}.sock",
        "control_socket_uid": 0,
        "endpoint_node_name": "ptyche-build" if build else node,
        "executable_path": "/usr/libexec/q30t-runtime-build"
        if build
        else "/usr/libexec/q30t-runtime-gateway",
        "executable_sha256": ("a" if build else "b") * 64,
        "expected_service_gid": 1974,
        "expected_service_uid": 0,
        "public_key_path": "/etc/q30t/build.pub" if build else "/etc/q30t/gateway.pub",
        "public_key_sha256": ("c" if build else "d") * 64,
        "service_identity": (
            "systemd:q30t-runtime-build.service"
            if build
            else "systemd:q30t-runtime-gateway.service"
        ),
        "service_kind": "build-service" if build else "gateway",
    }
    return RuntimeServiceEndpointIdentity(identity_sha256=canonical_sha256(values), **values)


def _write_run_receipt(
    tmp_path: Path,
    node_count: int,
    job_id: str,
    *,
    prerequisite_path: Path | None,
    prerequisite_sha256: str | None,
    global_routing_proven: bool | None = None,
) -> tuple[Path, str]:
    from common.specdec.q30t_runtime_platform_feasibility import (
        RuntimePlatformFeasibilityRunReceipt,
        build_global_srun_argv,
    )

    nodes = tuple(f"ptyche-n{index:03d}" for index in range(1, node_count + 1))
    protected = Path("/run/q30t-protected/12345/" + "e" * 64 + "/runtime.sqsh")
    argv = build_global_srun_argv(
        node_count=node_count,
        protected_runtime_path=protected,
        control_manifest_sha256="f" * 64,
        operation="platform-feasibility",
        canonical_ro_rprivate_mounts=("/run/control:/run/q30t/control:ro+rprivate",),
    )
    fields = {
        "build_service_endpoint_identity": _endpoint_identity("ptyche-build", 99, build=True),
        "cluster": "ptyche",
        "collective_artifact_file_sha256": "1" * 64,
        "empty_user_config_root_device": 501,
        "empty_user_config_root_inode": 502,
        "enroot_path": "/usr/bin/enroot",
        "enroot_sha256": "2" * 64,
        "enroot_system_config_manifest_sha256": "3" * 64,
        "enroot_version": "3.5.0",
        "feasibility_bound_blobs_sha256": "4" * 64,
        "global_go_abort_barrier_proven": True,
        "global_multinode_path_routing_proven": (
            node_count > 1 if global_routing_proven is None else global_routing_proven
        ),
        "global_rank_count": node_count,
        "global_srun_argv_sha256": canonical_sha256(argv),
        "hermetic_builder_closure_manifest_sha256": "5" * 64,
        "hermetic_builder_image_path": "/lustre/q30t-builder.sqsh",
        "hermetic_builder_image_sha256": "6" * 64,
        "job_id": job_id,
        "lease_break_time_seconds": 15,
        "live_backing_fd_same_inode_proven": True,
        "lustre_filesystem_type": "lustre",
        "node_count": node_count,
        "one_global_execution_step_proven": True,
        "ordered_bounded_log_file_sha256s": ("7" * 64,),
        "ordered_gateway_endpoint_identities": tuple(
            _endpoint_identity(node, index) for index, node in enumerate(nodes, start=1)
        ),
        "ordered_nodes": nodes,
        "ordered_probe_artifact_file_sha256s": ("8" * 64,),
        "persistent_rootfs_absent": True,
        "prerequisite_node_count": None if node_count == 1 else 1 if node_count == 2 else 2,
        "prerequisite_receipt_file_sha256": prerequisite_sha256,
        "prerequisite_receipt_path": str(prerequisite_path) if prerequisite_path else None,
        "producer_commit": "9" * 40,
        "producer_uid": 1000,
        "protected_link_same_inode_proven": True,
        "protected_runtime_root": "/run/q30t-protected",
        "pyxis_plugin_path": "/usr/lib/slurm/spank_pyxis.so",
        "pyxis_plugin_sha256": "a" * 64,
        "pyxis_version": "0.20.0",
        "raid_filesystem_type": "xfs",
        "remote_exit_proof_proven": True,
        "same_uid_write_break_proven": True,
        "schema_version": "q30t-runtime-platform-feasibility-run-v1",
        "scm_rights_same_ofd_proven": True,
        "segment_count": 16 if node_count == 16 else None,
        "sibling_pidfd_getfd_denied": True,
        "sigio_setup_order_proven": True,
        "slurm_spank_manifest_sha256": "b" * 64,
        "squashfuse_path": "/usr/bin/squashfuse",
        "squashfuse_sha256": "c" * 64,
        "unnamed_direct_squashfuse_proven": True,
        "yama_ptrace_scope": 1,
    }
    receipt = RuntimePlatformFeasibilityRunReceipt.create(**fields)
    path = (tmp_path / f"run-{node_count}.json").resolve()
    path.write_bytes(receipt.canonical_bytes() + b"\n")
    return path, hashlib.sha256(path.read_bytes()).hexdigest()


def _loaded_aggregate(tmp_path: Path) -> object:
    from common.specdec.q30t_runtime_platform_feasibility import (
        join_platform_feasibility_runs,
        load_platform_feasibility_receipt,
    )

    one = _write_run_receipt(tmp_path, 1, "1001", prerequisite_path=None, prerequisite_sha256=None)
    two = _write_run_receipt(
        tmp_path, 2, "1002", prerequisite_path=one[0], prerequisite_sha256=one[1]
    )
    sixteen = _write_run_receipt(
        tmp_path, 16, "1003", prerequisite_path=two[0], prerequisite_sha256=two[1]
    )
    aggregate = join_platform_feasibility_runs(
        one_node_receipt_path=one[0],
        one_node_receipt_file_sha256=one[1],
        two_node_receipt_path=two[0],
        two_node_receipt_file_sha256=two[1],
        sixteen_node_receipt_path=sixteen[0],
        sixteen_node_receipt_file_sha256=sixteen[1],
    )
    path = (tmp_path / "aggregate.json").resolve()
    path.write_bytes(aggregate.canonical_bytes() + b"\n")
    return load_platform_feasibility_receipt(
        path,
        hashlib.sha256(path.read_bytes()).hexdigest(),
    )


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


def _publication(node: str, path: Path, descriptor: int) -> ProtectedImagePublication:
    metadata = os.fstat(descriptor)
    return ProtectedImagePublication(
        schema_version="q30t-protected-image-publication-v1",
        job_id="12345",
        node_name=node,
        operation_id="a" * 64,
        producer_uid=1000,
        protected_path=str(path),
        device=metadata.st_dev,
        inode=metadata.st_ino,
        nlink=1,
        mode=0o400,
        size=metadata.st_size,
        sha256="b" * 64,
        same_ofd_lease_validated=True,
        gateway_signature="signed",
    )


def _live_gateway_identity(tmp_path: Path, node: str) -> RuntimeServiceEndpointIdentity:
    openssl = shutil.which("openssl")
    assert openssl is not None
    private_key = tmp_path / "gateway.key"
    public_key = tmp_path / "gateway.pub"
    subprocess.run(
        (openssl, "genpkey", "-algorithm", "ED25519", "-out", str(private_key)),
        check=True,
        capture_output=True,
    )
    subprocess.run(
        (openssl, "pkey", "-in", str(private_key), "-pubout", "-out", str(public_key)),
        check=True,
        capture_output=True,
    )
    values = {
        "control_socket_device": 1,
        "control_socket_gid": 1974,
        "control_socket_inode": 2,
        "control_socket_mode": 0o660,
        "control_socket_path": str(tmp_path / "gateway.sock"),
        "control_socket_uid": 0,
        "endpoint_node_name": node,
        "executable_path": "/usr/libexec/q30t-runtime-gateway",
        "executable_sha256": "a" * 64,
        "expected_service_gid": 1974,
        "expected_service_uid": 0,
        "public_key_path": str(public_key),
        "public_key_sha256": hashlib.sha256(public_key.read_bytes()).hexdigest(),
        "service_identity": "systemd:q30t-runtime-gateway.service",
        "service_kind": "gateway",
    }
    return RuntimeServiceEndpointIdentity(identity_sha256=canonical_sha256(values), **values)


def _sign_mount_attestation(
    attestation: PyxisDirectMountAttestation,
    identity: RuntimeServiceEndpointIdentity,
) -> PyxisDirectMountAttestation:
    openssl = shutil.which("openssl")
    assert openssl is not None
    with tempfile.NamedTemporaryFile() as payload:
        payload.write(gateway_module._canonical(attestation.body_dict()))
        payload.flush()
        signed = subprocess.run(
            (
                openssl,
                "pkeyutl",
                "-sign",
                "-inkey",
                str(Path(identity.public_key_path).with_suffix(".key")),
                "-rawin",
                "-in",
                payload.name,
            ),
            check=True,
            capture_output=True,
        ).stdout
    return dataclasses.replace(
        attestation,
        gateway_signature=base64.b64encode(signed).decode(),
    )


def _mount_attestation(
    node: str,
    path: Path,
    descriptor: int,
    identity: RuntimeServiceEndpointIdentity,
) -> PyxisDirectMountAttestation:
    metadata = os.fstat(descriptor)
    cgroup = Path(f"/proc/{os.getpid()}/cgroup").read_text().splitlines()[0].split(":", 2)[2]
    readonly_mount = next(
        fields
        for line in Path(f"/proc/{os.getpid()}/mountinfo").read_text().splitlines()
        if "ro" in (fields := line.split())[5].split(",")
    )
    attestation = PyxisDirectMountAttestation(
        schema_version="q30t-pyxis-direct-mount-attestation-v1",
        job_id="12345",
        step_id="0",
        node_name=node,
        operation_id="a" * 64,
        protected_path=str(path),
        image_device=metadata.st_dev,
        image_inode=metadata.st_ino,
        squashfuse_pid=os.getpid(),
        squashfuse_start_ticks=read_process_start_ticks(os.getpid()),
        squashfuse_cgroup_relative_path=cgroup,
        backing_fd_number=descriptor,
        backing_fd_access_mode="O_RDONLY",
        backing_fd_device=metadata.st_dev,
        backing_fd_inode=metadata.st_ino,
        root_mount_id=int(readonly_mount[0]),
        root_mount_readonly=True,
        temporary_rootfs=True,
        container_creation_absent=True,
        persistent_rootfs_absent=True,
        gateway_signature="",
    )
    return _sign_mount_attestation(attestation, identity)


@pytest.mark.skipif(sys.platform != "linux", reason="live mount identity requires Linux /proc")
def test_direct_mount_requires_gateway_signed_live_backing_fd_same_inode(tmp_path: Path) -> None:
    path = tmp_path / "runtime.sqsh"
    path.write_bytes(b"live image")
    descriptor = os.open(path, os.O_RDONLY | os.O_CLOEXEC)
    try:
        identity = _live_gateway_identity(tmp_path, "ptyche-n001")
        publication = _publication("ptyche-n001", path, descriptor)
        attestation = _mount_attestation("ptyche-n001", path, descriptor, identity)
        validate_direct_mount_attestations((attestation,), (publication,), (identity,))
        with pytest.raises(FeasibilityError, match="backing fd"):
            validate_direct_mount_attestations(
                (dataclasses.replace(attestation, backing_fd_inode=99),),
                (publication,),
                (identity,),
            )
        with pytest.raises(FeasibilityError, match="fresh SquashFUSE"):
            validate_direct_mount_attestations(
                (attestation, attestation),
                (publication, publication),
                (identity, identity),
            )
    finally:
        os.close(descriptor)


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
        (
            lambda trace: dataclasses.replace(trace, global_execution_step_count=2),
            "one global step",
        ),
        (lambda trace: dataclasses.replace(trace, global_rank_count=15), "rank count"),
        (lambda trace: dataclasses.replace(trace, node_count=15), "node count"),
        (
            lambda trace: dataclasses.replace(
                trace,
                ordered_staging_agents=(
                    trace.ordered_staging_agents[0],
                    dataclasses.replace(
                        trace.ordered_staging_agents[1],
                        endpoint_identity_sha256=trace.ordered_staging_agents[
                            0
                        ].endpoint_identity_sha256,
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


@pytest.mark.parametrize("field", ["agent_pid", "keeper_pid", "leased_fd"])
def test_multinode_rejects_reuse_in_each_local_identity_column(field: str) -> None:
    trace = _multinode_trace(2)
    replacement = dataclasses.replace(
        trace.ordered_staging_agents[1],
        **{field: getattr(trace.ordered_staging_agents[0], field)},
    )
    with pytest.raises(FeasibilityError, match="reuse"):
        validate_multinode_feasibility_trace(
            dataclasses.replace(
                trace,
                ordered_staging_agents=(trace.ordered_staging_agents[0], replacement),
            )
        )


def test_multinode_rejects_duplicate_or_conflicting_topology_flag() -> None:
    trace = _multinode_trace(2)
    with pytest.raises(FeasibilityError, match="flag cardinality"):
        validate_multinode_feasibility_trace(
            dataclasses.replace(trace, global_srun_argv=(*trace.global_srun_argv, "--nodes=16"))
        )


def _config_manifest() -> RecursiveConfigManifest:
    entries = (
        ConfigManifestEntry("etc/enroot/enroot.conf", "regular", 0o644, 0, 0, 6, "1" * 64),
        ConfigManifestEntry(
            "etc/enroot/environ.d/10-site.env", "regular", 0o644, 0, 0, 9, "4" * 64
        ),
        ConfigManifestEntry("etc/enroot/hooks.d/50-slurm.sh", "regular", 0o755, 0, 0, 7, "2" * 64),
        ConfigManifestEntry(
            "etc/enroot/mounts.d/10-site.fstab", "regular", 0o644, 0, 0, 8, "3" * 64
        ),
        ConfigManifestEntry("usr/bin/enroot", "regular", 0o755, 0, 0, 11, "6" * 64),
        ConfigManifestEntry("usr/bin/squashfuse", "regular", 0o755, 0, 0, 12, "7" * 64),
        ConfigManifestEntry("usr/lib/slurm/spank_pyxis.so", "regular", 0o755, 0, 0, 10, "5" * 64),
    )
    manifest = RecursiveConfigManifest(
        ordered_entries=entries,
        manifest_sha256="",
        empty_user_config_root_device=91,
        empty_user_config_root_inode=92,
        empty_user_config_root_uid=0,
        empty_user_config_root_gid=0,
        empty_user_config_root_mode=0o555,
        home="/run/q30t-empty-config",
        xdg_config_home="/run/q30t-empty-config",
    )
    return dataclasses.replace(manifest, manifest_sha256=canonical_sha256(manifest.body_dict()))


def test_config_manifest_rejects_extra_changed_system_or_user_config() -> None:
    reviewed = _config_manifest()
    validate_recursive_config_manifest(reviewed, reviewed)
    extra = ConfigManifestEntry(
        "etc/enroot/hooks.d/99-extra.sh", "regular", 0o755, 0, 0, 1, "8" * 64
    )
    with pytest.raises(FeasibilityError, match="config manifest"):
        validate_recursive_config_manifest(
            dataclasses.replace(reviewed, ordered_entries=(*reviewed.ordered_entries, extra)),
            reviewed,
        )


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda value: dataclasses.replace(value, empty_user_config_root_uid=1000), "gateway-owned"),
        (lambda value: dataclasses.replace(value, empty_user_config_root_mode=0o755), "gateway-owned"),
        (
            lambda value: dataclasses.replace(
                value,
                ordered_entries=value.ordered_entries[::-1],
                manifest_sha256=canonical_sha256(
                    dataclasses.replace(
                        value,
                        ordered_entries=value.ordered_entries[::-1],
                        manifest_sha256="",
                    ).body_dict()
                ),
            ),
            "sorted",
        ),
        (
            lambda value: dataclasses.replace(
                value,
                ordered_entries=(
                    dataclasses.replace(value.ordered_entries[0], entry_type="symlink"),
                    *value.ordered_entries[1:],
                ),
                manifest_sha256=canonical_sha256(
                    dataclasses.replace(
                        value,
                        ordered_entries=(
                            dataclasses.replace(value.ordered_entries[0], entry_type="symlink"),
                            *value.ordered_entries[1:],
                        ),
                        manifest_sha256="",
                    ).body_dict()
                ),
            ),
            "entry identity",
        ),
    ],
)
def test_config_manifest_requires_recursive_path_type_and_root_identity(
    mutation: object,
    message: str,
) -> None:
    reviewed = _config_manifest()
    observed = mutation(reviewed)
    with pytest.raises(FeasibilityError, match=message):
        validate_recursive_config_manifest(observed, observed)
    with pytest.raises(FeasibilityError, match="gateway-owned empty"):
        validate_recursive_config_manifest(
            dataclasses.replace(
                reviewed, home="/home/producer", xdg_config_home="/home/producer/.config"
            ),
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


@pytest.mark.parametrize(
    "changes",
    [
        {"authenticated_exit_nodes": ("ptyche-n002", "ptyche-n001")},
        {"ordered_wrapper_pid_start": (("ptyche-n001", 100, 200),)},
        {
            "ordered_cgroup_identities": (
                ("ptyche-n001", 10, 20),
                ("ptyche-n002", 10, 20),
            )
        },
    ],
)
def test_remote_exit_requires_exact_ordered_cardinality_and_identity_joins(
    changes: dict[str, object],
) -> None:
    proof = dataclasses.replace(_remote_exit_proof(), **changes)
    with pytest.raises(FeasibilityError, match=r"identity joins|identity is invalid"):
        validate_remote_step_exit_proof(
            proof,
            expected_job_id="12345",
            expected_step_id="0",
            expected_nodes=("ptyche-n001", "ptyche-n002"),
            lease_acquired_monotonic_ns=10_000_000_000,
            lease_break_time_seconds=15,
        )


def test_two_and_sixteen_node_receipts_require_global_routing_proof(tmp_path: Path) -> None:
    for node_count in (2, 16):
        prerequisite = (tmp_path / "prior.json").resolve()
        with pytest.raises(FeasibilityError, match=r"routing|segment-16"):
            _write_run_receipt(
                tmp_path,
                node_count,
                str(2000 + node_count),
                prerequisite_path=prerequisite,
                prerequisite_sha256="1" * 64,
                global_routing_proven=False,
            )


def test_bounded_collector_rejects_symlinks_and_streams_only_cap_plus_one(tmp_path: Path) -> None:
    root = (tmp_path / "logs").resolve()
    root.mkdir()
    target = root / "control.log"
    target.write_bytes(b"x" * (CONTROL_LOG_CAP_BYTES + 1))
    collector = BoundedArtifactCollector(node_root=root)
    with pytest.raises(FeasibilityError, match="cap"):
        collector.collect(target, kind="control")
    target.write_bytes(b"ok")
    symlink = root / "link.log"
    symlink.symlink_to(target)
    with pytest.raises(FeasibilityError, match="nofollow"):
        collector.collect(symlink, kind="control")


def test_endpoint_reconstruction_accepts_only_exact_stable_loader_result(tmp_path: Path) -> None:
    loaded = _loaded_aggregate(tmp_path)
    endpoint = reconstruct_runtime_gateway_endpoint(loaded, node_name="ptyche-n001")
    assert endpoint.identity.endpoint_node_name == "ptyche-n001"
    forged_equal = dataclasses.replace(loaded)
    with pytest.raises(Exception, match="stable-loaded"):
        reconstruct_runtime_gateway_endpoint(forged_equal, node_name="ptyche-n001")
    del loaded
    gc.collect()
    with pytest.raises(Exception, match="stable-loaded"):
        reconstruct_runtime_gateway_endpoint(forged_equal, node_name="ptyche-n001")


def test_endpoint_reconstruction_replays_physical_receipt_chain(tmp_path: Path) -> None:
    loaded = _loaded_aggregate(tmp_path)
    one_node_path = Path(loaded.one_node_receipt_path)
    one_node_path.write_bytes(one_node_path.read_bytes() + b"\n")
    with pytest.raises(Exception, match="stable-loaded"):
        reconstruct_runtime_gateway_endpoint(loaded, node_name="ptyche-n001")
