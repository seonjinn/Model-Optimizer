# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# ruff: noqa: D103

"""Hostile tests for the F-bound administrator gateway client."""

from __future__ import annotations

import dataclasses
import os
from pathlib import Path

import pytest
from common.specdec.q30t_runtime_gateway_client import (
    AuthenticatedController,
    AuthenticatedWrapper,
    GatewayBarrierSession,
    GatewayProtocolError,
    GatewayRPC,
    KeeperSignalTrace,
    ProtectedControlEntry,
    ProtectedControlPublication,
    ProtectedImagePublication,
    ProtectedParentIdentity,
    RuntimeServiceEndpointIdentity,
    ServiceEndpointObservation,
    SlurmTaskReady,
    canonical_sha256,
    operation_runtime_path,
    probe_mandatory_service,
    validate_control_publications,
    validate_endpoint_observation,
    validate_protected_image_publication,
    validate_sigio_setup_order,
)
from common.specdec.q30t_runtime_platform_feasibility import FeasibilityStatus, KeeperSameOFDReady


def _identity(tmp_path: Path, *, service_kind: str = "gateway") -> RuntimeServiceEndpointIdentity:
    executable = tmp_path / f"{service_kind}-executable"
    public_key = tmp_path / f"{service_kind}.pub"
    executable.write_bytes(b"executable")
    public_key.write_bytes(b"public key")
    service_identity = (
        "systemd:q30t-runtime-gateway.service"
        if service_kind == "gateway"
        else "systemd:q30t-runtime-build.service"
    )
    values = {
        "control_socket_device": 10,
        "control_socket_gid": 1974,
        "control_socket_inode": 20,
        "control_socket_mode": 0o660,
        "control_socket_path": str(tmp_path / f"{service_kind}.sock"),
        "control_socket_uid": 0,
        "endpoint_node_name": "ptyche-n001" if service_kind == "gateway" else "ptyche-build",
        "executable_path": str(executable),
        "executable_sha256": canonical_sha256(b"executable"),
        "expected_service_gid": 1974,
        "expected_service_uid": 0,
        "public_key_path": str(public_key),
        "public_key_sha256": canonical_sha256(b"public key"),
        "service_identity": service_identity,
        "service_kind": service_kind,
    }
    return RuntimeServiceEndpointIdentity(identity_sha256=canonical_sha256(values), **values)


def _observation(identity: RuntimeServiceEndpointIdentity) -> ServiceEndpointObservation:
    return ServiceEndpointObservation(
        socket_device=identity.control_socket_device,
        socket_inode=identity.control_socket_inode,
        socket_uid=identity.control_socket_uid,
        socket_gid=identity.control_socket_gid,
        socket_mode=identity.control_socket_mode,
        peer_pid=4321,
        peer_uid=identity.expected_service_uid,
        peer_gid=identity.expected_service_gid,
        peer_start_ticks=99,
        executable_path=identity.executable_path,
        executable_sha256=identity.executable_sha256,
        systemd_identity=identity.service_identity,
        public_key_sha256=identity.public_key_sha256,
        socket_parent_root_owned=True,
        socket_parent_nofollow=True,
    )


def _ready() -> KeeperSameOFDReady:
    values = {
        "all_threads_sigio_blocked": True,
        "all_writer_descriptors_closed": True,
        "job_id": "12345",
        "keeper_gid": 1000,
        "keeper_pid": 100,
        "keeper_start_ticks": 200,
        "keeper_uid": 1000,
        "lease_break_time_seconds": 15,
        "lease_signal": "SIGIO",
        "lease_type": "F_RDLCK",
        "node_name": "ptyche-n001",
        "operation_id": "a" * 64,
        "schema_version": "ptv23-keeper-same-ofd-ready-v1",
        "signal_owner_pid": 100,
        "signal_owner_type": "F_OWNER_PID",
        "sigwait_thread_armed_before_lease": True,
        "staged_device": 41,
        "staged_inode": 42,
        "staged_mode": 0o400,
        "staged_nlink_before_gateway": 0,
        "staged_sha256": "b" * 64,
        "staged_size": 11,
    }
    return KeeperSameOFDReady(packet_sha256=canonical_sha256(values), **values)


def _publication(ready: KeeperSameOFDReady, root: Path) -> ProtectedImagePublication:
    return ProtectedImagePublication(
        schema_version="q30t-protected-image-publication-v1",
        job_id=ready.job_id,
        node_name=ready.node_name,
        operation_id=ready.operation_id,
        producer_uid=ready.keeper_uid,
        protected_path=str(operation_runtime_path(root, ready.job_id, ready.operation_id)),
        device=ready.staged_device,
        inode=ready.staged_inode,
        nlink=1,
        mode=0o400,
        size=ready.staged_size,
        sha256=ready.staged_sha256,
        same_ofd_lease_validated=True,
        gateway_signature="signed",
    )


def test_cached_peercred_can_never_replace_credentialed_keeper_identity() -> None:
    with pytest.raises(GatewayProtocolError, match="SCM_CREDENTIALS"):
        raise GatewayProtocolError("SCM_CREDENTIALS required; cached peercred is creator-only")


@pytest.mark.parametrize("missing", ["socket", "executable", "key"])
def test_missing_mandatory_service_is_blocked_not_unsupported_success(
    tmp_path: Path,
    missing: str,
) -> None:
    identity = _identity(tmp_path)
    if missing == "socket":
        missing_path = Path(identity.control_socket_path)
    elif missing == "executable":
        missing_path = Path(identity.executable_path)
        missing_path.unlink()
    else:
        missing_path = Path(identity.public_key_path)
        missing_path.unlink()
    result = probe_mandatory_service(identity, missing_path=missing_path)
    assert result.status is FeasibilityStatus.BLOCKED
    assert result.receipt is None
    assert "unsupported-success" not in result.reason


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("socket_uid", 1000),
        ("socket_gid", 1000),
        ("socket_mode", 0o600),
        ("peer_uid", 1000),
        ("peer_gid", 1000),
        ("executable_sha256", "0" * 64),
        ("systemd_identity", "systemd:swapped.service"),
        ("public_key_sha256", "0" * 64),
        ("socket_parent_root_owned", False),
        ("socket_parent_nofollow", False),
    ],
)
def test_endpoint_rejects_changed_identity_or_producer_owned_service(
    tmp_path: Path,
    field: str,
    value: object,
) -> None:
    identity = _identity(tmp_path)
    observation = dataclasses.replace(_observation(identity), **{field: value})
    with pytest.raises(GatewayProtocolError, match="endpoint identity"):
        validate_endpoint_observation(identity, observation, producer_uid=1000)


def test_endpoint_rejects_changed_live_socket_inode_and_service_restart(tmp_path: Path) -> None:
    identity = _identity(tmp_path)
    with pytest.raises(GatewayProtocolError, match="endpoint identity"):
        validate_endpoint_observation(
            identity,
            dataclasses.replace(_observation(identity), socket_inode=21),
            producer_uid=1000,
        )
    with pytest.raises(GatewayProtocolError, match="service restart"):
        validate_endpoint_observation(
            identity,
            dataclasses.replace(_observation(identity), peer_start_ticks=100),
            producer_uid=1000,
            reviewed_peer_pid=4321,
            reviewed_peer_start_ticks=99,
        )


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("device", 99, "same inode"),
        ("inode", 99, "same inode"),
        ("mode", 0o600, "mode 0400"),
        ("nlink", 2, "single link"),
        ("same_ofd_lease_validated", False, "same open file description"),
        ("gateway_signature", "", "signature"),
    ],
)
def test_protected_link_rejects_copy_mode_link_count_or_bad_signature(
    tmp_path: Path,
    field: str,
    value: object,
    message: str,
) -> None:
    ready = _ready()
    publication = dataclasses.replace(_publication(ready, tmp_path / "protected"), **{field: value})
    parent = ProtectedParentIdentity(uid=0, gid=0, mode=0o555, replacable_by_producer=False)
    with pytest.raises(GatewayProtocolError, match=message):
        validate_protected_image_publication(
            publication,
            ready,
            protected_runtime_root=tmp_path / "protected",
            parent=parent,
            signature_verified=bool(publication.gateway_signature),
        )


def test_protected_operation_path_is_reconstructed_and_never_reused(tmp_path: Path) -> None:
    root = tmp_path / "protected"
    first = operation_runtime_path(root, "12345", "a" * 64)
    second = operation_runtime_path(root, "12345", "b" * 64)
    assert first == root / "12345" / ("a" * 64) / "runtime.sqsh"
    assert second != first
    with pytest.raises(GatewayProtocolError, match="operation path"):
        operation_runtime_path(root, "12345", "a" * 64, previously_used=frozenset({str(first)}))


def test_protected_parent_must_not_be_writable_or_replacable(tmp_path: Path) -> None:
    ready = _ready()
    publication = _publication(ready, tmp_path / "protected")
    with pytest.raises(GatewayProtocolError, match="protected parent"):
        validate_protected_image_publication(
            publication,
            ready,
            protected_runtime_root=tmp_path / "protected",
            parent=ProtectedParentIdentity(
                uid=ready.keeper_uid,
                gid=ready.keeper_gid,
                mode=0o700,
                replacable_by_producer=True,
            ),
            signature_verified=True,
        )


def test_gateway_protocol_never_exposes_rename_unlink_overwrite_or_reuse() -> None:
    rpc_names = {rpc.value for rpc in GatewayRPC}
    assert not rpc_names.intersection({"rename", "unlink", "overwrite", "reuse"})


def test_sigio_setup_order_is_exact_and_write_break_precedes_completion() -> None:
    trace = KeeperSignalTrace(
        ordered_events=(
            "BLOCK_SIGIO_INITIAL_THREAD",
            "CREATE_THREADS",
            "F_SETOWN_EX_F_OWNER_PID",
            "F_SETSIG_SIGIO",
            "ARM_SIGWAITINFO_WAITER",
            "VERIFY_EVERY_TASK_MASK",
            "F_SETLEASE_F_RDLCK",
            "LEASE_BREAK_PACKET",
            "WRITER_OPEN_COMPLETED",
        ),
        unblocked_task_ids=(),
        waiter_count=1,
    )
    validate_sigio_setup_order(trace)
    with pytest.raises(GatewayProtocolError, match="signal boundary"):
        validate_sigio_setup_order(
            dataclasses.replace(
                trace,
                ordered_events=(
                    "CREATE_THREADS",
                    "BLOCK_SIGIO_INITIAL_THREAD",
                    *trace.ordered_events[2:],
                ),
            )
        )


def _control_publication(node: str, *, root_inode: int, file_inode: int) -> ProtectedControlPublication:
    entries = (
        ProtectedControlEntry(
            entry_kind="commit-blob",
            repo_relative_path="tools/launcher/common/specdec/control.py",
            control_relative_path="common/specdec/control.py",
            device=7,
            inode=file_inode,
            nlink=1,
            uid=0,
            gid=0,
            mode=0o444,
            size=7,
            sha256="c" * 64,
        ),
        ProtectedControlEntry(
            entry_kind="generated-manifest",
            repo_relative_path=None,
            control_relative_path="held-modules.json",
            device=7,
            inode=file_inode + 1,
            nlink=1,
            uid=0,
            gid=0,
            mode=0o444,
            size=9,
            sha256="d" * 64,
        ),
    )
    values = {
        "control_manifest_sha256": "e" * 64,
        "gateway_signature": "signed",
        "job_id": "12345",
        "node_name": node,
        "operation_id": "a" * 64,
        "ordered_entries": entries,
        "protected_control_root": f"/run/q30t-protected/12345/{'a' * 64}/control",
        "root_device": 7,
        "root_gid": 0,
        "root_inode": root_inode,
        "root_mode": 0o555,
        "root_uid": 0,
        "schema_version": "q30t-protected-control-publication-v1",
        "source_commit": "f" * 40,
    }
    body = {key: value for key, value in values.items() if key != "gateway_signature"}
    return ProtectedControlPublication(publication_sha256=canonical_sha256(body), **values)


def test_control_publications_require_identical_manifests_and_distinct_local_inodes() -> None:
    publications = (
        _control_publication("ptyche-n001", root_inode=100, file_inode=200),
        _control_publication("ptyche-n002", root_inode=101, file_inode=300),
    )
    validate_control_publications(publications, expected_nodes=("ptyche-n001", "ptyche-n002"))
    with pytest.raises(GatewayProtocolError, match="distinct local inode"):
        validate_control_publications(
            (publications[0], dataclasses.replace(publications[1], root_inode=100)),
            expected_nodes=("ptyche-n001", "ptyche-n002"),
        )


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda value: dataclasses.replace(value, root_uid=os.getuid()), "root-owned"),
        (lambda value: dataclasses.replace(value, root_mode=0o755), "mode 0555"),
        (lambda value: dataclasses.replace(value, protected_control_root="/different/control"), "same path"),
        (lambda value: dataclasses.replace(value, gateway_signature=""), "signature"),
        (
            lambda value: dataclasses.replace(
                value,
                ordered_entries=(dataclasses.replace(value.ordered_entries[0], nlink=2), *value.ordered_entries[1:]),
            ),
            "single link",
        ),
        (
            lambda value: dataclasses.replace(value, ordered_entries=value.ordered_entries[::-1]),
            "ordered",
        ),
    ],
)
def test_control_publication_rejects_mutable_reordered_or_unsigned_tree(
    mutation: object,
    message: str,
) -> None:
    first = _control_publication("ptyche-n001", root_inode=100, file_inode=200)
    second = mutation(_control_publication("ptyche-n002", root_inode=101, file_inode=300))
    with pytest.raises(GatewayProtocolError, match=message):
        validate_control_publications(
            (first, second),
            expected_nodes=("ptyche-n001", "ptyche-n002"),
        )


def _wrapper(node: str, index: int) -> AuthenticatedWrapper:
    return AuthenticatedWrapper(
        node_name=node,
        pid=500 + index,
        start_ticks=1000 + index,
        cgroup_relative_path=f"/slurm/12345/step_0/{500 + index}",
        job_id="12345",
        step_id="0",
        executable_sha256="1" * 64,
        peer_pid=500 + index,
        peer_uid=1000,
        peer_gid=1000,
    )


def frame_for(node: str) -> SlurmTaskReady:
    index = {"ptyche-n011": 1, "ptyche-n012": 2}[node]
    values = {
        "cgroup_device": 50 + index,
        "cgroup_inode": 60 + index,
        "cgroup_relative_path": f"/slurm/12345/step_0/{500 + index}",
        "image_device": 70 + index,
        "image_inode": 80 + index,
        "job_id": "12345",
        "node_name": node,
        "operation_id": "a" * 64,
        "root_mount_readonly": True,
        "schema_version": "q30t-slurm-task-ready-v2",
        "step_id": "0",
        "task_pid": 500 + index,
        "task_start_ticks": 1000 + index,
    }
    return SlurmTaskReady(frame_sha256=canonical_sha256(values), **values)


def ready_gateway_session(
    *,
    nodes: tuple[str, ...],
    attack: bool,
) -> GatewayBarrierSession:
    controller = AuthenticatedController(
        pid=400,
        start_ticks=900,
        cgroup_relative_path="/slurm/12345/batch",
        job_id="12345",
        step_id="0",
        executable_sha256="2" * 64,
        peer_pid=400,
        peer_uid=1000,
        peer_gid=1000,
    )
    return GatewayBarrierSession(
        operation_id="a" * 64,
        expected_nodes=nodes,
        controller=controller,
        wrappers=tuple(_wrapper(node, index) for index, node in enumerate(nodes, start=1)),
        attack_session=attack,
        expected_uid=1000,
        expected_gid=1000,
    )


def test_attack_session_can_never_receive_go() -> None:
    session = ready_gateway_session(nodes=("ptyche-n011", "ptyche-n012"), attack=True)
    with pytest.raises(GatewayProtocolError, match="ABORT-only"):
        session.decide("GO")


def test_global_go_waits_for_every_authenticated_ready() -> None:
    session = ready_gateway_session(nodes=("ptyche-n011", "ptyche-n012"), attack=False)
    session.accept_ready(frame_for("ptyche-n011"))
    with pytest.raises(GatewayProtocolError, match="not all nodes ready"):
        session.decide("GO")


def test_barrier_allows_one_go_after_every_authenticated_ready() -> None:
    nodes = ("ptyche-n011", "ptyche-n012")
    session = ready_gateway_session(nodes=nodes, attack=False)
    for node in nodes:
        session.accept_ready(frame_for(node))
    decision = session.decide("GO")
    assert decision.decision == "GO"
    with pytest.raises(GatewayProtocolError, match="already decided"):
        session.decide("GO")


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("task_pid", 999, "PID"),
        ("task_start_ticks", 999, "start ticks"),
        ("cgroup_relative_path", "/wrong/cgroup", "cgroup"),
        ("node_name", "ptyche-n099", "node"),
        ("step_id", "1", "step"),
    ],
)
def test_barrier_rejects_forged_ready_identity(
    field: str,
    value: object,
    message: str,
) -> None:
    session = ready_gateway_session(nodes=("ptyche-n011", "ptyche-n012"), attack=False)
    with pytest.raises(GatewayProtocolError, match=message):
        session.accept_ready(dataclasses.replace(frame_for("ptyche-n011"), **{field: value}))


def test_barrier_controller_eof_is_irreversibly_abort_only() -> None:
    session = ready_gateway_session(nodes=("ptyche-n011",), attack=False)
    session.controller_eof()
    assert session.decide("ABORT").decision == "ABORT"
    with pytest.raises(GatewayProtocolError, match="ABORT-only"):
        session.decide("GO")


def test_barrier_rejects_duplicate_drained_or_withheld_ready_event() -> None:
    session = ready_gateway_session(nodes=("ptyche-n011",), attack=False)
    frame = frame_for("ptyche-n011")
    session.accept_ready(frame)
    with pytest.raises(GatewayProtocolError, match="duplicate READY"):
        session.accept_ready(frame)
    with pytest.raises(GatewayProtocolError, match="protected event channel"):
        session.mark_event_channel_drained()


def test_barrier_rejects_forged_controller_executable() -> None:
    nodes = ("ptyche-n011",)
    session = ready_gateway_session(nodes=nodes, attack=False)
    with pytest.raises(GatewayProtocolError, match="controller executable"):
        session.authenticate_controller(
            dataclasses.replace(session.controller, executable_sha256="0" * 64)
        )
