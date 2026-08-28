# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Hostile behavioral contracts for Q30 Ptyche runtime attestation."""

# Test names describe the protected behavior; fixture helpers are intentionally local.
# ruff: noqa: D101, D103

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import pytest
from common.specdec import ptv23_runtime_attestation as module
from common.specdec.ptv23_node_keeper import KeeperItem, KeeperReceipt
from common.specdec.ptv23_runtime_attestation import (
    AttestationError,
    RuntimeKeeperLossEvidence,
    RuntimeNodeAttestationInput,
    RuntimeNodeObservation,
    RuntimeNodeReuseEvidence,
    RuntimeQualificationContext,
    attest_runtime_node,
    load_runtime_node_receipt,
    load_runtime_qualification_receipt,
    reconcile_runtime_receipts,
)
from common.specdec.q30t_runtime_archive_receipt import RuntimeArchiveTreeReceipt
from common.specdec.q30t_runtime_identity import RuntimeTreeIdentity, runtime_tree_identity

if TYPE_CHECKING:
    from collections.abc import Callable


def canonical(value: object) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def publish_model(path: Path, value: Any) -> str:
    payload = value.to_dict() if hasattr(value, "to_dict") else asdict(value)
    path.write_bytes(canonical(payload) + b"\n")
    return file_sha256(path)


def self_hash(body: object) -> str:
    return hashlib.sha256(canonical(body)).hexdigest()


def make_archive_receipt(
    path: Path,
    *,
    runtime_identity: RuntimeTreeIdentity,
    source_commit: str,
) -> tuple[Path, str]:
    archive = path.parent / "runtime.tar.zst"
    producer = path.parent / "archive-producer.py"
    archive.write_bytes(b"runtime-archive")
    producer.write_bytes(b"archive-producer")
    body: dict[str, object] = {
        "archive_path": str(archive),
        "archive_sha256": file_sha256(archive),
        "archive_size": archive.stat().st_size,
        "producer_path": str(producer),
        "producer_sha256": file_sha256(producer),
        "runtime_file_count": runtime_identity.file_count,
        "runtime_symlink_count": runtime_identity.symlink_count,
        "runtime_total_regular_bytes": runtime_identity.total_regular_bytes,
        "runtime_tree_sha256": runtime_identity.sha256,
        "schema_version": "q30t-runtime-archive-tree-v1",
        "sentinel_inventory_sha256": "c" * 64,
        "sentinel_paths": ["bin/python", "pyvenv.cfg"],
        "source_commit": source_commit,
    }
    receipt = RuntimeArchiveTreeReceipt.from_dict(body | {"receipt_sha256": self_hash(body)})
    publish_model(path, receipt)
    return path, file_sha256(path)


def make_keeper_receipt(
    path: Path,
    *,
    job_id: str,
    node_name: str,
    keeper_pid: int,
    keeper_start_ticks: int,
    image_path: Path,
    image_sha256: str,
    anchor_path: Path,
    descriptor: int = 17,
) -> tuple[KeeperReceipt, str]:
    item = KeeperItem(
        name="runtime-image",
        source_path=image_path,
        expected_sha256=image_sha256,
        staged_size=image_path.stat().st_size,
        staged_sha256=image_sha256,
        anchor_path=anchor_path,
        descriptor=descriptor,
    )
    body: dict[str, object] = {
        "items": [
            {
                "anchor_path": str(item.anchor_path),
                "descriptor": item.descriptor,
                "expected_sha256": item.expected_sha256,
                "name": item.name,
                "source_path": str(item.source_path),
                "staged_sha256": item.staged_sha256,
                "staged_size": item.staged_size,
            }
        ],
        "job_id": job_id,
        "keeper_pid": keeper_pid,
        "keeper_start_ticks": keeper_start_ticks,
        "node_name": node_name,
        "schema_version": "ptv23-node-keeper-v1",
    }
    receipt = KeeperReceipt(
        schema_version="ptv23-node-keeper-v1",
        job_id=job_id,
        node_name=node_name,
        keeper_pid=keeper_pid,
        keeper_start_ticks=keeper_start_ticks,
        items=(item,),
        receipt_sha256=self_hash(body),
    )
    publish_model(path, receipt)
    return receipt, file_sha256(path)


def make_observation(**overrides: object) -> RuntimeNodeObservation:
    body: dict[str, object] = {
        "anchor_device": 19,
        "anchor_inode": 101,
        "anchor_path": "/raid/scratch/user/probe/anchors/runtime-image",
        "anchor_target": "/proc/41/fd/17",
        "archive_sha256": hashlib.sha256(b"runtime-archive").hexdigest(),
        "archive_tree_receipt_file_sha256": "d" * 64,
        "attestation_tool_sha256": "e" * 64,
        "cluster": "ptyche",
        "container_name": "q30t-700k-unit-1",
        "container_root_device": 51,
        "container_root_inode": 61,
        "container_sentinel_sha256": "f" * 64,
        "contract_file_sha256": "1" * 64,
        "cuda_visible_devices": "0,1,2,3",
        "expected_image_sha256": hashlib.sha256(b"runtime-image").hexdigest(),
        "job_id": "unit-1",
        "keeper_pid": 41,
        "keeper_receipt_sha256": "2" * 64,
        "keeper_start_ticks": 700,
        "keeper_tool_sha256": "3" * 64,
        "machine": "aarch64",
        "node_name": "ptyche-n001",
        "observed_image_sha256": hashlib.sha256(b"runtime-image").hexdigest(),
        "phase": "one-node",
        "profile_file_sha256": "4" * 64,
        "python_executable": "/runtime/bin/python",
        "python_import_origins": [
            ["accelerate", "/runtime/lib/python3.12/site-packages/accelerate/__init__.py"],
            ["datasets", "/runtime/lib/python3.12/site-packages/datasets/__init__.py"],
            ["modelopt", "/source/modelopt/__init__.py"],
            ["wandb", "/runtime/lib/python3.12/site-packages/wandb/__init__.py"],
        ],
        "python_version": "3.12.11",
        "pyxis_version": "pyxis 0.21.0",
        "runner_sha256": "5" * 64,
        "runtime_tree_sha256": "b" * 64,
        "schema_version": "q30t-runtime-node-observation-v1",
        "sentinel_inventory_sha256": "c" * 64,
        "source_commit": "a" * 40,
        "visible_gpu_count": 4,
        "visible_gpu_identities": [
            "GPU-0001|NVIDIA GB200",
            "GPU-0002|NVIDIA GB200",
            "GPU-0003|NVIDIA GB200",
            "GPU-0004|NVIDIA GB200",
        ],
        "enroot_version": "enroot 3.5.0",
    }
    body.update(overrides)
    return RuntimeNodeObservation.from_dict(body | {"observation_sha256": self_hash(body)})


def rehash_observation(
    observation: RuntimeNodeObservation, **overrides: object
) -> RuntimeNodeObservation:
    body = observation.body_dict()
    body.update(overrides)
    return RuntimeNodeObservation.from_dict(body | {"observation_sha256": self_hash(body)})


def make_reuse(
    observation: RuntimeNodeObservation, **overrides: object
) -> RuntimeNodeReuseEvidence:
    body: dict[str, object] = {
        "container_name": observation.container_name,
        "container_root_device": observation.container_root_device,
        "container_root_inode": observation.container_root_inode,
        "container_sentinel_sha256": observation.container_sentinel_sha256,
        "job_id": observation.job_id,
        "named_container_reused": True,
        "node_name": observation.node_name,
        "observation_sha256": observation.observation_sha256,
        "phase": observation.phase,
        "schema_version": "q30t-runtime-node-reuse-v1",
    }
    body.update(overrides)
    return RuntimeNodeReuseEvidence.from_dict(body | {"reuse_sha256": self_hash(body)})


def make_keeper_loss(
    observation: RuntimeNodeObservation, **overrides: object
) -> RuntimeKeeperLossEvidence:
    body: dict[str, object] = {
        "anchor_device": observation.anchor_device,
        "anchor_inode": observation.anchor_inode,
        "anchor_path": observation.anchor_path,
        "error_classification": "anchor-unreadable",
        "job_id": observation.job_id,
        "keeper_loss_fresh_use_failed": True,
        "keeper_pid": observation.keeper_pid,
        "keeper_start_ticks": observation.keeper_start_ticks,
        "node_name": observation.node_name,
        "phase": observation.phase,
        "schema_version": "q30t-runtime-node-keeper-loss-v1",
    }
    body.update(overrides)
    return RuntimeKeeperLossEvidence.from_dict(body | {"keeper_loss_sha256": self_hash(body)})


@dataclass(frozen=True)
class RuntimeFixture:
    inputs: RuntimeNodeAttestationInput
    archive_sha256: str
    image_sha256: str


def runtime_fixture(
    tmp_path: Path,
    *,
    node_name: str = "ptyche-n001",
    phase: str = "one-node",
    job_id: str = "unit-1",
    keeper_pid: int = 41,
    anchor_inode: int = 101,
    gpu_offset: int = 0,
    common_root: Path | None = None,
) -> RuntimeFixture:
    root = tmp_path / node_name
    root.mkdir(parents=True)
    asset_root = common_root or root
    asset_root.mkdir(parents=True, exist_ok=True)
    profile = asset_root / "profile.json"
    profile.write_bytes(b'{"profile":"reviewed"}\n')
    image = asset_root / "runtime.sqsh"
    image.write_bytes(b"runtime-image")
    runtime = asset_root / "runtime"
    source = asset_root / "source"
    (runtime / "lib/python3.12/site-packages").mkdir(parents=True, exist_ok=True)
    (source / "modelopt").mkdir(parents=True, exist_ok=True)
    python_executable = runtime / "bin/python"
    python_executable.parent.mkdir(parents=True, exist_ok=True)
    python_executable.write_bytes(b"#!/usr/bin/env python3\n")
    for package in ("accelerate", "datasets", "wandb"):
        origin = runtime / f"lib/python3.12/site-packages/{package}/__init__.py"
        origin.parent.mkdir(parents=True, exist_ok=True)
        origin.write_bytes(f"{package}\n".encode())
    (source / "modelopt/__init__.py").write_bytes(b"modelopt\n")
    tools = {}
    for name in ("contract", "runner", "keeper", "attestation"):
        tool = source / f"{name}.py"
        tool.write_bytes(name.encode())
        tools[name] = tool
    if not (source / ".git").exists():
        subprocess.run(("/usr/bin/git", "-C", str(source), "init", "-q"), check=True)
        subprocess.run(("/usr/bin/git", "-C", str(source), "add", "--all"), check=True)
        subprocess.run(
            (
                "/usr/bin/git",
                "-C",
                str(source),
                "-c",
                "user.name=Runtime Test",
                "-c",
                "user.email=runtime@example.invalid",
                "commit",
                "-qm",
                "fixture",
            ),
            check=True,
        )
    source_commit = subprocess.run(
        ("/usr/bin/git", "-C", str(source), "rev-parse", "HEAD"),
        check=True,
        stdout=subprocess.PIPE,
        text=True,
    ).stdout.strip()
    runtime_identity = runtime_tree_identity(runtime)
    archive_receipt_path = (common_root or root) / "archive-receipt.json"
    archive_receipt_path.parent.mkdir(parents=True, exist_ok=True)
    if archive_receipt_path.exists():
        archive_receipt_file_sha256 = file_sha256(archive_receipt_path)
    else:
        archive_receipt_path, archive_receipt_file_sha256 = make_archive_receipt(
            archive_receipt_path,
            runtime_identity=runtime_identity,
            source_commit=source_commit,
        )
    archive_receipt = module.load_runtime_archive_tree_receipt(
        archive_receipt_path, archive_receipt_file_sha256
    )
    anchor = root / "anchor"
    image_sha256 = file_sha256(image)
    keeper_receipt, keeper_receipt_file_sha256 = make_keeper_receipt(
        root / "keeper.json",
        job_id=job_id,
        node_name=node_name,
        keeper_pid=keeper_pid,
        keeper_start_ticks=700 + keeper_pid,
        image_path=image,
        image_sha256=image_sha256,
        anchor_path=anchor,
    )
    observation = make_observation(
        anchor_inode=anchor_inode,
        anchor_path=str(anchor),
        anchor_target=f"/proc/{keeper_pid}/fd/17",
        archive_sha256=archive_receipt.archive_sha256,
        archive_tree_receipt_file_sha256=archive_receipt_file_sha256,
        attestation_tool_sha256=file_sha256(tools["attestation"]),
        container_name=f"q30t-700k-{job_id}",
        contract_file_sha256=file_sha256(tools["contract"]),
        expected_image_sha256=image_sha256,
        job_id=job_id,
        keeper_pid=keeper_pid,
        keeper_receipt_sha256=keeper_receipt.receipt_sha256,
        keeper_start_ticks=keeper_receipt.keeper_start_ticks,
        keeper_tool_sha256=file_sha256(tools["keeper"]),
        node_name=node_name,
        observed_image_sha256=image_sha256,
        phase=phase,
        profile_file_sha256=file_sha256(profile),
        python_executable=str(python_executable),
        python_import_origins=[
            [
                "accelerate",
                str(runtime / "lib/python3.12/site-packages/accelerate/__init__.py"),
            ],
            ["datasets", str(runtime / "lib/python3.12/site-packages/datasets/__init__.py")],
            ["modelopt", str(source / "modelopt/__init__.py")],
            ["wandb", str(runtime / "lib/python3.12/site-packages/wandb/__init__.py")],
        ],
        runner_sha256=file_sha256(tools["runner"]),
        runtime_tree_sha256=archive_receipt.runtime_tree_sha256,
        sentinel_inventory_sha256=archive_receipt.sentinel_inventory_sha256,
        source_commit=source_commit,
        visible_gpu_identities=[
            f"GPU-{gpu_offset + index:04d}|NVIDIA GB200" for index in range(1, 5)
        ],
    )
    inputs = RuntimeNodeAttestationInput(
        profile_path=profile,
        profile_file_sha256=file_sha256(profile),
        archive_tree_receipt_path=archive_receipt_path,
        archive_tree_receipt_file_sha256=archive_receipt_file_sha256,
        archive_tree_receipt=archive_receipt,
        keeper_receipt_path=root / "keeper.json",
        keeper_receipt_file_sha256=keeper_receipt_file_sha256,
        keeper_receipt=keeper_receipt,
        mounted_sqsh_path=image,
        extracted_runtime_path=runtime,
        source_checkout=source,
        contract_path=tools["contract"],
        runner_path=tools["runner"],
        keeper_tool_path=tools["keeper"],
        attestation_tool_path=tools["attestation"],
        observation=observation,
        reuse_evidence=make_reuse(observation),
        keeper_loss_evidence=make_keeper_loss(observation),
    )
    return RuntimeFixture(
        inputs=inputs,
        archive_sha256=archive_receipt.archive_sha256,
        image_sha256=image_sha256,
    )


def test_node_attestation_binds_observed_runtime_and_keeper(tmp_path: Path) -> None:
    fixture = runtime_fixture(tmp_path, node_name="ptyche-n001")
    receipt = attest_runtime_node(
        fixture.inputs,
        output_path=tmp_path / "node.json",
        publication_job_id="unit-1",
    )
    assert receipt.schema_version == "q30t-runtime-node-attestation-v1"
    assert receipt.node_name == "ptyche-n001"
    assert receipt.observed_image_sha256 == fixture.image_sha256
    assert receipt.archive_sha256 == fixture.archive_sha256
    assert receipt.anchor_inode > 0
    assert receipt.visible_gpu_count == 4
    assert receipt.machine == "aarch64"
    assert receipt.container_name == "q30t-700k-unit-1"
    assert receipt.named_container_reused is True
    assert receipt.keeper_loss_fresh_use_failed is True
    assert (
        load_runtime_node_receipt(tmp_path / "node.json", file_sha256(tmp_path / "node.json"))
        == receipt
    )


@pytest.mark.parametrize("missing", ["python", "accelerate", "datasets", "modelopt", "wandb"])
def test_node_attestation_requires_existing_python_runtime_files(
    tmp_path: Path, missing: str
) -> None:
    fixture = runtime_fixture(tmp_path)
    if missing == "python":
        observation = rehash_observation(
            fixture.inputs.observation,
            python_executable=str(fixture.inputs.extracted_runtime_path / "bin/missing-python"),
        )
        message = "Python executable"
    else:
        origins = [list(item) for item in fixture.inputs.observation.python_import_origins]
        for origin in origins:
            if origin[0] == missing:
                origin[1] = str(Path(origin[1]).with_name("missing.py"))
        observation = rehash_observation(fixture.inputs.observation, python_import_origins=origins)
        message = f"{missing} import origin"
    inputs = replace(
        fixture.inputs,
        observation=observation,
        reuse_evidence=make_reuse(observation),
        keeper_loss_evidence=make_keeper_loss(observation),
    )

    with pytest.raises(AttestationError, match=message):
        attest_runtime_node(inputs, output_path=tmp_path / "node.json", publication_job_id="unit")


def test_node_attestation_authenticates_runtime_and_source_trees(tmp_path: Path) -> None:
    changed_runtime = runtime_fixture(tmp_path / "runtime-changed")
    changed_runtime.inputs.extracted_runtime_path.joinpath("bin/python").write_bytes(b"changed\n")
    with pytest.raises(AttestationError, match="runtime tree"):
        attest_runtime_node(
            changed_runtime.inputs,
            output_path=tmp_path / "runtime-node.json",
            publication_job_id="runtime",
        )

    changed_source = runtime_fixture(tmp_path / "source-changed")
    (changed_source.inputs.source_checkout / "untracked.txt").write_bytes(b"untracked\n")
    with pytest.raises(AttestationError, match="source checkout"):
        attest_runtime_node(
            changed_source.inputs,
            output_path=tmp_path / "source-node.json",
            publication_job_id="source",
        )

    changed_commit = runtime_fixture(tmp_path / "source-commit-changed")
    source_root = changed_commit.inputs.source_checkout
    (source_root / "modelopt/__init__.py").write_bytes(b"new commit\n")
    subprocess.run(("/usr/bin/git", "-C", str(source_root), "add", "--all"), check=True)
    subprocess.run(
        (
            "/usr/bin/git",
            "-C",
            str(source_root),
            "-c",
            "user.name=Runtime Test",
            "-c",
            "user.email=runtime@example.invalid",
            "commit",
            "-qm",
            "changed",
        ),
        check=True,
    )
    with pytest.raises(AttestationError, match="source checkout"):
        attest_runtime_node(
            changed_commit.inputs,
            output_path=tmp_path / "source-commit-node.json",
            publication_job_id="source-commit",
        )


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda value: replace(value, profile_file_sha256="9" * 64), "profile"),
        (lambda value: replace(value, expected_image_sha256="9" * 64), "image"),
        (lambda value: replace(value, keeper_pid=99), "keeper"),
        (lambda value: replace(value, anchor_inode=999), "anchor"),
        (lambda value: replace(value, container_name="wrong"), "container"),
    ],
)
def test_node_attestation_rejects_cross_replay_mismatch(
    tmp_path: Path,
    mutate: Callable[[RuntimeNodeObservation], RuntimeNodeObservation],
    message: str,
) -> None:
    fixture = runtime_fixture(tmp_path)
    changed = mutate(fixture.inputs.observation)
    inputs = replace(fixture.inputs, observation=changed)
    with pytest.raises(AttestationError, match=message):
        attest_runtime_node(inputs, output_path=tmp_path / "node.json", publication_job_id="x")


@pytest.mark.parametrize(
    "path_field",
    [
        "profile_path",
        "archive_tree_receipt_path",
        "keeper_receipt_path",
        "mounted_sqsh_path",
        "runner_path",
    ],
)
def test_node_attestation_rejects_changed_replayed_input(tmp_path: Path, path_field: str) -> None:
    fixture = runtime_fixture(tmp_path)
    path = getattr(fixture.inputs, path_field)
    path.write_bytes(path.read_bytes() + b"forged")
    with pytest.raises((AttestationError, ValueError), match=r"SHA-256|canonical|mismatch"):
        attest_runtime_node(
            fixture.inputs,
            output_path=tmp_path / "node.json",
            publication_job_id="changed-input",
        )


def test_node_attestation_rejects_source_import_origin_substitution(tmp_path: Path) -> None:
    fixture = runtime_fixture(tmp_path)
    observation = fixture.inputs.observation
    origins = tuple(
        (name, "/runtime/forged/modelopt.py" if name == "modelopt" else origin)
        for name, origin in observation.python_import_origins
    )
    changed = replace(observation, python_import_origins=origins)
    inputs = replace(fixture.inputs, observation=changed)
    with pytest.raises(AttestationError, match="modelopt import origin"):
        attest_runtime_node(
            inputs,
            output_path=tmp_path / "node.json",
            publication_job_id="source-substitution",
        )


@pytest.mark.parametrize(
    ("factory", "field"),
    [
        (make_reuse, "named_container_reused"),
        (make_keeper_loss, "keeper_loss_fresh_use_failed"),
    ],
)
def test_proof_evidence_rejects_false_boolean(factory: Callable[..., object], field: str) -> None:
    observation = make_observation()
    with pytest.raises(AttestationError, match=field):
        factory(observation, **{field: False})


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"visible_gpu_identities": ["GPU-x|GB200"] * 4}, "GPU"),
        ({"visible_gpu_count": 3}, "GPU"),
        ({"cuda_visible_devices": "0,0,1,2"}, "CUDA"),
        ({"cuda_visible_devices": "0, 0,1,2"}, "CUDA"),
        ({"observed_image_sha256": "9" * 64}, "image"),
        ({"anchor_target": "/proc/999/fd/17"}, "anchor target"),
        ({"python_version": "3.12\nforged"}, "control"),
        ({"pyxis_version": "x" * 4097}, "bounded"),
        (
            {
                "python_import_origins": [
                    ["accelerate", "/runtime/a"],
                    ["datasets", "/runtime/d"],
                    ["datasets", "/runtime/forged/modelopt.py"],
                    ["wandb", "/runtime/w"],
                ]
            },
            "required imports",
        ),
    ],
)
def test_observation_rejects_invalid_runtime_metadata(
    change: dict[str, object], message: str
) -> None:
    with pytest.raises(AttestationError, match=message):
        make_observation(**change)


@dataclass(frozen=True)
class MaterializedNode:
    receipt: module.RuntimeNodeAttestationReceipt
    path: Path
    file_sha256: str
    inputs: RuntimeNodeAttestationInput


def make_node(
    tmp_path: Path,
    *,
    node: str,
    keeper_pid: int,
    anchor_inode: int,
    gpu_offset: int,
    phase: str,
    job_id: str,
) -> MaterializedNode:
    fixture = runtime_fixture(
        tmp_path / f"fixture-{node}",
        node_name=node,
        phase=phase,
        job_id=job_id,
        keeper_pid=keeper_pid,
        anchor_inode=anchor_inode,
        gpu_offset=gpu_offset,
        common_root=tmp_path / "common",
    )
    path = tmp_path / f"{node}.json"
    receipt = attest_runtime_node(fixture.inputs, output_path=path, publication_job_id=job_id)
    return MaterializedNode(receipt, path, file_sha256(path), fixture.inputs)


def one_node_context(node: MaterializedNode, tmp_path: Path) -> RuntimeQualificationContext:
    del tmp_path
    receipt = node.receipt
    return RuntimeQualificationContext(
        job_id=receipt.job_id,
        phase="one-node",
        cluster="ptyche",
        profile_path=node.inputs.profile_path,
        profile_file_sha256=receipt.profile_file_sha256,
        archive_tree_receipt_path=node.inputs.archive_tree_receipt_path,
        archive_tree_receipt_file_sha256=receipt.archive_tree_receipt_file_sha256,
        archive_tree_receipt=node.inputs.archive_tree_receipt,
        archive_sha256=receipt.archive_sha256,
        image_sha256=receipt.expected_image_sha256,
        runtime_tree_sha256=receipt.runtime_tree_sha256,
        source_commit=receipt.source_commit,
        contract_file_sha256=receipt.contract_file_sha256,
        runner_sha256=receipt.runner_sha256,
        keeper_tool_sha256=receipt.keeper_tool_sha256,
        attestation_tool_sha256=receipt.attestation_tool_sha256,
        prerequisite_one_node_receipt_path=None,
        prerequisite_one_node_receipt_file_sha256=None,
    )


def reconcile_one(
    tmp_path: Path, node: MaterializedNode
) -> tuple[module.RuntimeQualificationReceipt, Path, str]:
    path = tmp_path / "one-node.json"
    receipt = reconcile_runtime_receipts(
        receipts=(node.receipt,),
        receipt_file_sha256s=(node.file_sha256,),
        expected_nodes=(node.receipt.node_name,),
        context=one_node_context(node, tmp_path),
        output_path=path,
        publication_job_id=node.receipt.job_id,
    )
    return receipt, path, file_sha256(path)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("job_id", "unit\nforged"),
        ("ordered_nodes", ["ptyche-n001\nforged"]),
        ("ordered_nodes", ["n" * 4097]),
    ],
)
def test_qualification_rejects_unbounded_job_and_node_identities(
    tmp_path: Path, field: str, value: object
) -> None:
    node = make_node(
        tmp_path,
        node="ptyche-n001",
        keeper_pid=91,
        anchor_inode=191,
        gpu_offset=0,
        phase="one-node",
        job_id="unit-one",
    )
    qualification, _, _ = reconcile_one(tmp_path, node)
    body = qualification.body_dict()
    body[field] = value
    if field == "ordered_nodes":
        nodes = cast("list[str]", value)
        receipt_hashes = cast("list[str]", body["ordered_node_receipt_file_sha256s"])
        body["ordered_node_receipts_sha256"] = self_hash(
            [
                {
                    "node_name": nodes[0],
                    "receipt_file_sha256": receipt_hashes[0],
                }
            ]
        )

    with pytest.raises(AttestationError, match=r"control|bounded"):
        module.RuntimeQualificationReceipt.from_dict(body | {"receipt_sha256": self_hash(body)})


def test_reconciliation_rejects_control_node_before_set_comparison(tmp_path: Path) -> None:
    node = make_node(
        tmp_path,
        node="ptyche-n001",
        keeper_pid=91,
        anchor_inode=191,
        gpu_offset=0,
        phase="one-node",
        job_id="unit-one",
    )
    with pytest.raises(AttestationError, match="control"):
        reconcile_runtime_receipts(
            receipts=(node.receipt,),
            receipt_file_sha256s=(node.file_sha256,),
            expected_nodes=("ptyche-n001\nforged",),
            context=one_node_context(node, tmp_path),
            output_path=tmp_path / "aggregate.json",
            publication_job_id="unit-one",
        )


def test_node_reference_list_rejects_cardinality_before_loading_receipts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    references = [
        {
            "node_name": f"ptyche-n00{index}",
            "path": str(tmp_path / f"node-{index}.json"),
            "sha256": f"{index}" * 64,
        }
        for index in (1, 2, 3)
    ]
    reference_path = tmp_path / "references.json"
    reference_path.write_bytes(canonical(references) + b"\n")
    loaded: list[Path] = []

    def forbidden_load(path: Path, expected_file_sha256: str) -> None:
        del expected_file_sha256
        loaded.append(path)
        raise AssertionError("referenced receipt loaded before cardinality validation")

    monkeypatch.setattr(module, "load_runtime_node_receipt", forbidden_load)
    with pytest.raises(AttestationError, match="one or two"):
        module._load_node_receipt_list(reference_path, None)
    assert loaded == []


def two_node_context(
    nodes: tuple[MaterializedNode, MaterializedNode],
    one_path: Path,
    one_sha256: str,
) -> RuntimeQualificationContext:
    receipt = nodes[0].receipt
    base = one_node_context(nodes[0], one_path.parent)
    return replace(
        base,
        job_id=receipt.job_id,
        phase="two-node",
        prerequisite_one_node_receipt_path=one_path,
        prerequisite_one_node_receipt_file_sha256=one_sha256,
    )


def valid_two_node_context(
    tmp_path: Path, nodes: tuple[MaterializedNode, MaterializedNode]
) -> RuntimeQualificationContext:
    one = make_node(
        tmp_path,
        node="ptyche-n001",
        keeper_pid=91,
        anchor_inode=191,
        gpu_offset=0,
        phase="one-node",
        job_id="unit-one",
    )
    _, one_path, one_sha256 = reconcile_one(tmp_path, one)
    return two_node_context(nodes, one_path, one_sha256)


def test_two_node_reconciliation_requires_reviewed_one_node_receipt(tmp_path: Path) -> None:
    one_node = make_node(
        tmp_path,
        node="ptyche-n001",
        keeper_pid=91,
        anchor_inode=191,
        gpu_offset=0,
        phase="one-node",
        job_id="unit-one",
    )
    _, one_path, one_sha256 = reconcile_one(tmp_path, one_node)
    nodes = (
        make_node(
            tmp_path,
            node="ptyche-n011",
            keeper_pid=101,
            anchor_inode=201,
            gpu_offset=10,
            phase="two-node",
            job_id="unit-two",
        ),
        make_node(
            tmp_path,
            node="ptyche-n012",
            keeper_pid=102,
            anchor_inode=202,
            gpu_offset=20,
            phase="two-node",
            job_id="unit-two",
        ),
    )
    result = reconcile_runtime_receipts(
        receipts=tuple(item.receipt for item in reversed(nodes)),
        receipt_file_sha256s=tuple(item.file_sha256 for item in reversed(nodes)),
        expected_nodes=("ptyche-n011", "ptyche-n012"),
        context=two_node_context(nodes, one_path, one_sha256),
        output_path=tmp_path / "two-node.json",
        publication_job_id="unit-two",
    )
    assert result.prerequisite_one_node_receipt_file_sha256 == one_sha256
    assert result.ordered_nodes == ("ptyche-n011", "ptyche-n012")
    assert (
        load_runtime_qualification_receipt(
            tmp_path / "two-node.json", file_sha256(tmp_path / "two-node.json")
        )
        == result
    )


def test_two_node_reconciliation_rejects_prerequisite_sha_substitution(tmp_path: Path) -> None:
    nodes = (
        make_node(
            tmp_path,
            node="ptyche-n011",
            keeper_pid=101,
            anchor_inode=201,
            gpu_offset=10,
            phase="two-node",
            job_id="unit-two",
        ),
        make_node(
            tmp_path,
            node="ptyche-n012",
            keeper_pid=102,
            anchor_inode=202,
            gpu_offset=20,
            phase="two-node",
            job_id="unit-two",
        ),
    )
    context = replace(
        valid_two_node_context(tmp_path, nodes),
        prerequisite_one_node_receipt_file_sha256="9" * 64,
    )
    with pytest.raises(AttestationError, match="whole-file SHA-256 mismatch"):
        reconcile_runtime_receipts(
            receipts=tuple(node.receipt for node in nodes),
            receipt_file_sha256s=tuple(node.file_sha256 for node in nodes),
            expected_nodes=tuple(node.receipt.node_name for node in nodes),
            context=context,
            output_path=tmp_path / "aggregate.json",
            publication_job_id="unit-two",
        )


def test_reconciliation_rejects_node_set_mismatch(tmp_path: Path) -> None:
    node = make_node(
        tmp_path,
        node="ptyche-n001",
        keeper_pid=41,
        anchor_inode=101,
        gpu_offset=0,
        phase="one-node",
        job_id="one",
    )
    context = one_node_context(node, tmp_path)
    with pytest.raises(AttestationError, match="node set"):
        reconcile_runtime_receipts(
            receipts=(node.receipt,),
            receipt_file_sha256s=(node.file_sha256,),
            expected_nodes=("ptyche-extra",),
            context=context,
            output_path=tmp_path / "aggregate.json",
            publication_job_id="one",
        )


def test_reconciliation_binds_each_receipt_to_its_file_sha(tmp_path: Path) -> None:
    node = make_node(
        tmp_path,
        node="ptyche-n001",
        keeper_pid=41,
        anchor_inode=101,
        gpu_offset=0,
        phase="one-node",
        job_id="one",
    )
    with pytest.raises(AttestationError, match="canonical bytes"):
        reconcile_runtime_receipts(
            receipts=(node.receipt,),
            receipt_file_sha256s=("9" * 64,),
            expected_nodes=(node.receipt.node_name,),
            context=one_node_context(node, tmp_path),
            output_path=tmp_path / "aggregate.json",
            publication_job_id="one",
        )


@pytest.mark.parametrize(
    ("identity", "message"), [("keeper", "PID"), ("anchor", "anchor"), ("gpu", "GPU")]
)
def test_two_node_reconciliation_rejects_duplicate_identity(
    tmp_path: Path, identity: str, message: str
) -> None:
    first = make_node(
        tmp_path,
        node="ptyche-n011",
        keeper_pid=101,
        anchor_inode=201,
        gpu_offset=10,
        phase="two-node",
        job_id="unit-two",
    )
    second = make_node(
        tmp_path,
        node="ptyche-n012",
        keeper_pid=102,
        anchor_inode=202,
        gpu_offset=20,
        phase="two-node",
        job_id="unit-two",
    )
    changes: dict[str, object]
    if identity == "keeper":
        changes = {"keeper_pid": first.receipt.keeper_pid}
    elif identity == "anchor":
        changes = {
            "anchor_device": first.receipt.anchor_device,
            "anchor_inode": first.receipt.anchor_inode,
        }
    else:
        changes = {"visible_gpu_identities": first.receipt.visible_gpu_identities}
    second = replace(second, receipt=replace(second.receipt, **changes))
    context = valid_two_node_context(tmp_path, (first, second))
    with pytest.raises(AttestationError, match=message):
        reconcile_runtime_receipts(
            receipts=(first.receipt, second.receipt),
            receipt_file_sha256s=(first.file_sha256, second.file_sha256),
            expected_nodes=(first.receipt.node_name, second.receipt.node_name),
            context=context,
            output_path=tmp_path / "aggregate.json",
            publication_job_id="unit-two",
        )


@pytest.mark.parametrize(
    "field",
    [
        "archive_sha256",
        "observed_image_sha256",
        "runtime_tree_sha256",
        "profile_file_sha256",
        "source_commit",
        "runner_sha256",
        "keeper_tool_sha256",
        "attestation_tool_sha256",
        "python_version",
        "python_executable",
        "python_import_origins",
        "pyxis_version",
        "enroot_version",
        "container_name",
    ],
)
def test_two_node_reconciliation_rejects_common_identity_mismatch(
    tmp_path: Path, field: str
) -> None:
    nodes = [
        make_node(
            tmp_path,
            node=f"ptyche-n01{index}",
            keeper_pid=100 + index,
            anchor_inode=200 + index,
            gpu_offset=index * 10,
            phase="two-node",
            job_id="unit-two",
        )
        for index in (1, 2)
    ]
    original = getattr(nodes[1].receipt, field)
    changed: object = ("changed",) if isinstance(original, tuple) else "9" * 64
    if field in {
        "python_version",
        "python_executable",
        "pyxis_version",
        "enroot_version",
        "container_name",
    }:
        changed = "changed"
    if field == "source_commit":
        changed = "9" * 40
    nodes[1] = replace(nodes[1], receipt=replace(nodes[1].receipt, **{field: changed}))
    context = valid_two_node_context(tmp_path, (nodes[0], nodes[1]))
    with pytest.raises(AttestationError, match="common"):
        reconcile_runtime_receipts(
            receipts=tuple(node.receipt for node in nodes),
            receipt_file_sha256s=tuple(node.file_sha256 for node in nodes),
            expected_nodes=tuple(node.receipt.node_name for node in nodes),
            context=context,
            output_path=tmp_path / "aggregate.json",
            publication_job_id="unit-two",
        )


@pytest.mark.parametrize("loader_name", ["node", "qualification"])
def test_receipt_loader_rejects_unknown_reordered_duplicate_and_mutated_bytes(
    tmp_path: Path, loader_name: str
) -> None:
    node = make_node(
        tmp_path,
        node="ptyche-n001",
        keeper_pid=41,
        anchor_inode=101,
        gpu_offset=0,
        phase="one-node",
        job_id="one",
    )
    if loader_name == "node":
        receipt_path = node.path
        loader = load_runtime_node_receipt
    else:
        _, receipt_path, _ = reconcile_one(tmp_path, node)
        loader = load_runtime_qualification_receipt
    pristine = json.loads(receipt_path.read_bytes())
    mutations = (
        canonical({key: value for key, value in pristine.items() if key != "job_id"}) + b"\n",
        canonical(pristine | {"unknown": 1}) + b"\n",
        json.dumps(pristine, sort_keys=False, indent=2).encode() + b"\n",
        receipt_path.read_bytes().replace(b'"job_id"', b'"job_id":"forged","job_id"', 1),
        receipt_path.read_bytes().replace(b'"receipt_sha256":"', b'"receipt_sha256":"0', 1),
    )
    for index, raw in enumerate(mutations):
        candidate = tmp_path / f"mutated-{loader_name}-{index}.json"
        candidate.write_bytes(raw)
        with pytest.raises((AttestationError, ValueError)):
            loader(candidate, file_sha256(candidate))


def test_qualification_loader_recomputes_ordered_node_digest(tmp_path: Path) -> None:
    node = make_node(
        tmp_path,
        node="ptyche-n001",
        keeper_pid=41,
        anchor_inode=101,
        gpu_offset=0,
        phase="one-node",
        job_id="one",
    )
    _, receipt_path, _ = reconcile_one(tmp_path, node)
    raw = json.loads(receipt_path.read_bytes())
    raw["ordered_node_receipts_sha256"] = "9" * 64
    body = {key: value for key, value in raw.items() if key != "receipt_sha256"}
    raw["receipt_sha256"] = self_hash(body)
    receipt_path.write_bytes(canonical(raw) + b"\n")
    with pytest.raises(AttestationError, match="ordered node receipt digest"):
        load_runtime_qualification_receipt(receipt_path, file_sha256(receipt_path))


def test_loader_rejects_hard_link_and_growth(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    node = make_node(
        tmp_path,
        node="ptyche-n001",
        keeper_pid=41,
        anchor_inode=101,
        gpu_offset=0,
        phase="one-node",
        job_id="one",
    )
    hard_link = tmp_path / "hard-link.json"
    os.link(node.path, hard_link)
    with pytest.raises(AttestationError, match="single-link"):
        load_runtime_node_receipt(node.path, node.file_sha256)
    hard_link.unlink()
    monkeypatch.setattr(
        module, "_READ_HOOK", lambda: node.path.write_bytes(node.path.read_bytes() + b"x")
    )
    with pytest.raises(AttestationError, match="changed"):
        load_runtime_node_receipt(node.path, node.file_sha256)


def test_loader_rejects_path_rebind(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    node = make_node(
        tmp_path,
        node="ptyche-n001",
        keeper_pid=41,
        anchor_inode=101,
        gpu_offset=0,
        phase="one-node",
        job_id="one",
    )

    def rebind() -> None:
        detached = node.path.with_suffix(".detached")
        node.path.rename(detached)
        node.path.write_bytes(detached.read_bytes())

    monkeypatch.setattr(module, "_READ_HOOK", rebind)
    with pytest.raises(AttestationError, match="changed"):
        load_runtime_node_receipt(node.path, node.file_sha256)


def test_publication_adopts_only_exact_bytes(tmp_path: Path) -> None:
    fixture = runtime_fixture(tmp_path)
    output = tmp_path / "node.json"
    first = attest_runtime_node(fixture.inputs, output_path=output, publication_job_id="first")
    assert (
        attest_runtime_node(fixture.inputs, output_path=output, publication_job_id="second")
        == first
    )
    output.unlink()
    output.write_bytes(b"foreign\n")
    with pytest.raises(FileExistsError, match="differ"):
        attest_runtime_node(fixture.inputs, output_path=output, publication_job_id="third")


def test_publication_rejects_intermediate_symlink_without_foreign_mutation(
    tmp_path: Path,
) -> None:
    fixture = runtime_fixture(tmp_path / "fixture")
    lexical = tmp_path / "lexical"
    foreign = tmp_path / "foreign"
    lexical.mkdir()
    foreign.mkdir()
    (lexical / "redirect").symlink_to(foreign, target_is_directory=True)

    with pytest.raises((AttestationError, OSError)):
        attest_runtime_node(
            fixture.inputs,
            output_path=lexical / "redirect/nested/node.json",
            publication_job_id="symlink",
        )

    assert list(foreign.iterdir()) == []


def test_exact_adoption_does_not_bypass_publication_job_validation(tmp_path: Path) -> None:
    fixture = runtime_fixture(tmp_path)
    output = tmp_path / "node.json"
    attest_runtime_node(fixture.inputs, output_path=output, publication_job_id="valid")
    with pytest.raises(ValueError, match="publication job"):
        attest_runtime_node(
            fixture.inputs,
            output_path=output,
            publication_job_id="unsafe job",
        )


def test_publication_does_not_adopt_after_unrelated_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = runtime_fixture(tmp_path)

    def fail(*args: object, **kwargs: object) -> None:
        raise PermissionError("publication parent denied")

    monkeypatch.setattr(module, "_publish_fresh_at", fail)
    with pytest.raises(PermissionError, match="denied"):
        attest_runtime_node(
            fixture.inputs, output_path=tmp_path / "node.json", publication_job_id="fail"
        )


def test_verify_cli_replays_without_writing(tmp_path: Path) -> None:
    node = make_node(
        tmp_path,
        node="ptyche-n001",
        keeper_pid=41,
        anchor_inode=101,
        gpu_offset=0,
        phase="one-node",
        job_id="one",
    )
    result = subprocess.run(
        (
            sys.executable,
            "-m",
            "common.specdec.ptv23_runtime_attestation",
            "verify",
            "--receipt",
            str(node.path),
            "--sha256",
            node.file_sha256,
        ),
        check=False,
        capture_output=True,
        env={"PATH": os.environ["PATH"], "PYTHONPATH": "tools/launcher"},
    )
    assert result.returncode == 0, result.stderr.decode()
    assert json.loads(result.stdout)["schema_version"] == "q30t-runtime-node-attestation-v1"


def test_attest_node_cli_replays_explicit_evidence_files(tmp_path: Path) -> None:
    fixture = runtime_fixture(tmp_path)
    observation_path = tmp_path / "observation.json"
    observation_path.write_bytes(fixture.inputs.observation.canonical_bytes() + b"\n")
    reuse_path = tmp_path / "reuse.json"
    reuse_path.write_bytes(fixture.inputs.reuse_evidence.canonical_bytes() + b"\n")
    keeper_loss_path = tmp_path / "keeper-loss.json"
    keeper_loss_path.write_bytes(fixture.inputs.keeper_loss_evidence.canonical_bytes() + b"\n")
    input_path = tmp_path / "input.json"
    input_path.write_bytes(
        canonical(
            {
                "archive_tree_receipt_file_sha256": (
                    fixture.inputs.archive_tree_receipt_file_sha256
                ),
                "archive_tree_receipt_path": str(fixture.inputs.archive_tree_receipt_path),
                "attestation_tool_path": str(fixture.inputs.attestation_tool_path),
                "contract_path": str(fixture.inputs.contract_path),
                "extracted_runtime_path": str(fixture.inputs.extracted_runtime_path),
                "keeper_loss_evidence_file_sha256": file_sha256(keeper_loss_path),
                "keeper_loss_evidence_path": str(keeper_loss_path),
                "keeper_receipt_file_sha256": fixture.inputs.keeper_receipt_file_sha256,
                "keeper_receipt_path": str(fixture.inputs.keeper_receipt_path),
                "keeper_tool_path": str(fixture.inputs.keeper_tool_path),
                "mounted_sqsh_path": str(fixture.inputs.mounted_sqsh_path),
                "observation_file_sha256": file_sha256(observation_path),
                "observation_path": str(observation_path),
                "profile_file_sha256": fixture.inputs.profile_file_sha256,
                "profile_path": str(fixture.inputs.profile_path),
                "reuse_evidence_file_sha256": file_sha256(reuse_path),
                "reuse_evidence_path": str(reuse_path),
                "runner_path": str(fixture.inputs.runner_path),
                "source_checkout": str(fixture.inputs.source_checkout),
            }
        )
        + b"\n"
    )
    output = tmp_path / "node-cli.json"
    result = subprocess.run(
        (
            sys.executable,
            "-m",
            "common.specdec.ptv23_runtime_attestation",
            "attest-node",
            "--input",
            str(input_path),
            "--output",
            str(output),
            "--job-id",
            "unit-1",
        ),
        check=False,
        capture_output=True,
        env={"PATH": os.environ["PATH"], "PYTHONPATH": "tools/launcher"},
    )
    assert result.returncode == 0, result.stderr.decode()
    assert load_runtime_node_receipt(output, file_sha256(output)).node_name == "ptyche-n001"


def test_reconcile_cli_replays_canonical_control_files(tmp_path: Path) -> None:
    node = make_node(
        tmp_path,
        node="ptyche-n001",
        keeper_pid=41,
        anchor_inode=101,
        gpu_offset=0,
        phase="one-node",
        job_id="one",
    )
    context = one_node_context(node, tmp_path)
    context_path = tmp_path / "context.json"
    context_path.write_bytes(
        canonical(
            {
                "archive_sha256": context.archive_sha256,
                "archive_tree_receipt_file_sha256": context.archive_tree_receipt_file_sha256,
                "archive_tree_receipt_path": str(context.archive_tree_receipt_path),
                "attestation_tool_sha256": context.attestation_tool_sha256,
                "cluster": context.cluster,
                "contract_file_sha256": context.contract_file_sha256,
                "image_sha256": context.image_sha256,
                "job_id": context.job_id,
                "keeper_tool_sha256": context.keeper_tool_sha256,
                "phase": context.phase,
                "prerequisite_one_node_receipt_file_sha256": None,
                "prerequisite_one_node_receipt_path": None,
                "profile_file_sha256": context.profile_file_sha256,
                "profile_path": str(context.profile_path),
                "runner_sha256": context.runner_sha256,
                "runtime_tree_sha256": context.runtime_tree_sha256,
                "source_commit": context.source_commit,
            }
        )
        + b"\n"
    )
    node_list = tmp_path / "nodes.json"
    node_list.write_bytes(
        canonical(
            [
                {
                    "node_name": node.receipt.node_name,
                    "path": str(node.path),
                    "sha256": node.file_sha256,
                }
            ]
        )
        + b"\n"
    )
    output = tmp_path / "qualification.json"
    result = subprocess.run(
        (
            sys.executable,
            "-m",
            "common.specdec.ptv23_runtime_attestation",
            "reconcile",
            "--context",
            str(context_path),
            "--node-receipt-list",
            str(node_list),
            "--output",
            str(output),
            "--job-id",
            "one",
        ),
        check=False,
        capture_output=True,
        env={"PATH": os.environ["PATH"], "PYTHONPATH": "tools/launcher"},
    )
    assert result.returncode == 0, result.stderr.decode()
    assert load_runtime_qualification_receipt(output, file_sha256(output)).phase == "one-node"
