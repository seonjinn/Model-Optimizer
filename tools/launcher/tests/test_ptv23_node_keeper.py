# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Behavioral contracts for the node-local descriptor keeper."""

from __future__ import annotations

import hashlib
import json
import os
import signal
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, replace
from pathlib import Path

import pytest
from common.specdec import ptv23_node_keeper as keeper_module
from common.specdec.ptv23_node_keeper import (
    KeeperError,
    KeeperItem,
    KeeperPlan,
    KeeperReceipt,
    KeeperSource,
    load_receipt,
    stage_regular_to_tmpfile,
    validate_keeper_receipt,
)


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _scratch_root(job_id: str = "12345") -> Path:
    return keeper_module._expected_scratch_root(job_id)


def _test_job_id(root: Path) -> str:
    return f"test-{hashlib.sha256(str(root).encode()).hexdigest()[:16]}"


def _plan(root: Path, inputs: dict[str, bytes]) -> KeeperPlan:
    source_root = root / "sources"
    source_root.mkdir(parents=True)
    sources = []
    for name, payload in sorted(inputs.items()):
        source = source_root / name
        source.write_bytes(payload)
        sources.append(
            KeeperSource(name=name, source_path=source, expected_sha256=_sha256(payload))
        )
    fifo_path = root / "controller.fifo"
    os.mkfifo(fifo_path, 0o600)
    return KeeperPlan(
        schema_version="ptv23-node-keeper-v1",
        job_id=_test_job_id(root),
        node_name="node-a",
        scratch_root=_scratch_root(_test_job_id(root)),
        directory_roots=(source_root,),
        anchor_root=_scratch_root(_test_job_id(root)) / "anchors",
        fifo_path=fifo_path,
        sources=tuple(sources),
    )


def _require_tmpfile(root: Path) -> None:
    tmpfile = getattr(os, "O_TMPFILE", None)
    if tmpfile is None:
        pytest.skip("O_TMPFILE is unavailable")
    try:
        descriptor = os.open(root, os.O_RDWR | tmpfile, 0o600)
    except OSError:
        pytest.skip("test filesystem does not support O_TMPFILE")
    else:
        os.close(descriptor)


def _require_keeper_scratch(root: Path) -> None:
    scratch = _scratch_root(_test_job_id(root))
    try:
        scratch.mkdir(parents=True, exist_ok=True)
    except OSError:
        pytest.skip("node-local /raid/scratch is unavailable")
    _require_tmpfile(scratch)


@dataclass
class _StartedKeeper:
    process: subprocess.Popen[bytes]
    receipt: Path

    def kill(self) -> None:
        self.process.kill()
        self.process.wait(timeout=10)


def _wait_ready(receipt_path: Path) -> None:
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        if receipt_path.is_file():
            return
        time.sleep(0.02)
    raise AssertionError("keeper did not write its readiness receipt")


def _start_keeper(root: Path, inputs: dict[str, bytes]) -> _StartedKeeper:
    plan = _plan(root, inputs)
    plan_path = root / "plan.json"
    plan_path.write_bytes(plan.canonical_bytes() + b"\n")
    receipt = root / "receipt.json"
    process = subprocess.Popen(
        [
            sys.executable,
            "-I",
            "tools/launcher/common/specdec/ptv23_node_keeper.py",
            "serve",
            "--plan",
            str(plan_path),
            "--receipt",
            str(receipt),
        ],
        cwd=Path(__file__).resolve().parents[3],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    _wait_ready(receipt)
    return _StartedKeeper(process=process, receipt=receipt)


def _run_keeper_once(root: Path, inputs: dict[str, bytes]):
    keeper = _start_keeper(root, inputs)
    try:
        receipt = load_receipt(keeper.receipt)
        validate_keeper_receipt(keeper.receipt)
        return receipt
    finally:
        keeper.kill()


def test_keeper_stages_every_file_with_postcopy_digest(tmp_path: Path) -> None:
    """Every named source is independently rehashed after descriptor staging."""
    _require_tmpfile(tmp_path)
    _require_keeper_scratch(tmp_path)

    receipt = _run_keeper_once(tmp_path, inputs={"image": b"image", "contract": b"contract"})

    assert receipt.names == ("contract", "image")
    assert all(item.source_sha256 == item.staged_sha256 for item in receipt.items)


def test_keeper_short_write_and_dead_process_fail_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Short writes stay complete and a dead keeper invalidates its receipt."""
    _require_tmpfile(tmp_path)
    source = tmp_path / "source"
    payload = b"short-write-safe" * 100
    source.write_bytes(payload)
    real_write = os.write

    def short_write(descriptor: int, data: bytes | memoryview) -> int:
        return real_write(descriptor, data[: max(1, len(data) // 3)])

    monkeypatch.setattr(os, "write", short_write)
    descriptor, staged_size, staged_sha256 = stage_regular_to_tmpfile(
        source, _sha256(payload), tmp_path
    )
    try:
        assert staged_size == len(payload)
        assert staged_sha256 == _sha256(payload)
        assert os.read(descriptor, len(payload) + 1) == payload
    finally:
        os.close(descriptor)
    monkeypatch.setattr(os, "write", real_write)

    _require_keeper_scratch(tmp_path / "keeper")
    keeper = _start_keeper(tmp_path / "keeper", {"image": b"image"})
    keeper.kill()
    with pytest.raises(KeeperError, match="keeper is not alive"):
        validate_keeper_receipt(keeper.receipt)


def test_plan_and_receipt_are_canonical_and_reject_foreign_anchor_replacement(
    tmp_path: Path,
) -> None:
    """Receipt bytes stay canonical and cleanup preserves foreign anchors."""
    _require_tmpfile(tmp_path)
    _require_keeper_scratch(tmp_path)
    keeper = _start_keeper(tmp_path, {"image": b"image"})
    try:
        receipt = load_receipt(keeper.receipt)
        parsed = json.loads(keeper.receipt.read_text())
        assert (
            keeper.receipt.read_bytes()
            == json.dumps(parsed, sort_keys=True, separators=(",", ":")).encode() + b"\n"
        )
        anchor = receipt.items[0].anchor_path
        anchor.unlink()
        anchor.write_text("foreign")
    finally:
        keeper.kill()
    assert anchor.read_text() == "foreign"


def test_canonical_plan_and_receipt_reject_noncanonical_wire_bytes(tmp_path: Path) -> None:
    """Controller evidence must have exactly one JSON encoding before it is trusted."""
    plan = _plan(tmp_path, {"image": b"image"})
    plan_path = tmp_path / "plan.json"
    plan_path.write_bytes(plan.canonical_bytes() + b"\n")
    assert keeper_module.load_plan(plan_path) == plan
    plan_path.write_text(json.dumps(plan.to_dict(), indent=2) + "\n")
    with pytest.raises(KeeperError, match="not canonical"):
        keeper_module.load_plan(plan_path)

    item = KeeperItem(
        name="image",
        source_path=tmp_path / "sources" / "image",
        expected_sha256=_sha256(b"image"),
        staged_size=5,
        staged_sha256=_sha256(b"image"),
        anchor_path=tmp_path / "anchors" / "image",
        descriptor=0,
    )
    body = {
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
        "job_id": "12345",
        "keeper_pid": 1,
        "keeper_start_ticks": 1,
        "node_name": "node-a",
        "schema_version": "ptv23-node-keeper-v1",
    }
    receipt = KeeperReceipt(
        schema_version="ptv23-node-keeper-v1",
        job_id="12345",
        node_name="node-a",
        keeper_pid=1,
        keeper_start_ticks=1,
        items=(item,),
        receipt_sha256=hashlib.sha256(
            json.dumps(body, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest(),
    )
    receipt_path = tmp_path / "receipt.json"
    receipt_path.write_bytes(
        json.dumps(receipt.to_dict(), sort_keys=True, separators=(",", ":")).encode() + b"\n"
    )
    assert keeper_module.load_receipt(receipt_path) == receipt
    receipt_path.write_text(json.dumps(receipt.to_dict(), indent=2) + "\n")
    with pytest.raises(KeeperError, match="not canonical"):
        keeper_module.load_receipt(receipt_path)


def test_unsupported_tmpfile_and_foreign_anchor_cleanup_fail_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Absent anonymous staging and foreign anchor replacement never degrade safety."""
    source = tmp_path / "source"
    source.write_bytes(b"image")
    monkeypatch.delattr(os, "O_TMPFILE", raising=False)
    with pytest.raises(KeeperError, match="O_TMPFILE"):
        stage_regular_to_tmpfile(source, _sha256(b"image"), tmp_path)

    anchor_root = tmp_path / "anchors"
    anchor_root.mkdir(mode=0o700)
    anchor = anchor_root / "image"
    anchor.symlink_to("/proc/1/fd/1")
    metadata = anchor.lstat()
    owned = keeper_module._OwnedAnchor("image", metadata.st_dev, metadata.st_ino, "/proc/1/fd/1")
    anchor.unlink()
    anchor.write_text("foreign")
    descriptor = os.open(anchor_root, os.O_RDONLY | os.O_DIRECTORY)
    try:
        keeper_module._cleanup_owned_anchors(descriptor, (owned,))
    finally:
        os.close(descriptor)
    assert anchor.read_text() == "foreign"


def test_plan_rejects_dotdot_directory_root_before_any_open(tmp_path: Path) -> None:
    """A lexical allowlist escape cannot authorize staging from a foreign tree."""
    foreign_source = tmp_path / "allowed" / ".." / "foreign" / "image"
    with pytest.raises(KeeperError, match="canonical"):
        KeeperPlan(
            schema_version="ptv23-node-keeper-v1",
            job_id="12345",
            node_name="node-a",
            scratch_root=_scratch_root(),
            directory_roots=(tmp_path / "allowed" / "..",),
            anchor_root=_scratch_root() / "anchors",
            fifo_path=tmp_path / "controller.fifo",
            sources=(
                KeeperSource(
                    name="image",
                    source_path=foreign_source,
                    expected_sha256=_sha256(b"image"),
                ),
            ),
        )


def test_plan_rejects_non_node_local_or_other_user_scratch(tmp_path: Path) -> None:
    """A keeper plan cannot place locks or anonymous files on shared storage."""
    source = tmp_path / "image"
    source.write_bytes(b"image")
    with pytest.raises(KeeperError, match="node-local job namespace"):
        KeeperPlan(
            schema_version="ptv23-node-keeper-v1",
            job_id="12345",
            node_name="node-a",
            scratch_root=tmp_path / "scratch",
            directory_roots=(tmp_path,),
            anchor_root=tmp_path / "scratch" / "anchors",
            fifo_path=tmp_path / "controller.fifo",
            sources=(KeeperSource("image", source, _sha256(b"image")),),
        )


def test_absent_nofollow_fails_before_source_or_tmpfile_open(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The keeper refuses platforms without a meaningful no-follow primitive."""
    source = tmp_path / "source"
    source.write_bytes(b"image")
    monkeypatch.setattr(os, "O_NOFOLLOW", 0, raising=False)
    with pytest.raises(KeeperError, match="O_NOFOLLOW"):
        stage_regular_to_tmpfile(source, _sha256(b"image"), tmp_path)


def test_controller_path_must_be_a_nofollow_fifo(tmp_path: Path) -> None:
    """The controller channel rejects a regular file before it can control lifetime."""
    regular = tmp_path / "controller"
    regular.write_text("stop")
    with pytest.raises(KeeperError, match="FIFO"):
        keeper_module._open_controller_fifo(regular)

    real_parent = tmp_path / "real"
    real_parent.mkdir()
    os.mkfifo(real_parent / "controller.fifo", 0o600)
    (tmp_path / "link").symlink_to(real_parent, target_is_directory=True)
    with pytest.raises(OSError):
        keeper_module._open_controller_fifo(tmp_path / "link" / "controller.fifo")


def test_controller_fifo_stop_and_eof_are_descriptor_lifetime_events(tmp_path: Path) -> None:
    """A verified FIFO delivers both explicit stop bytes and controller EOF."""
    fifo = tmp_path / "controller.fifo"
    os.mkfifo(fifo, 0o600)
    opened: list[int] = []
    thread = threading.Thread(
        target=lambda: opened.append(keeper_module._open_controller_fifo(fifo))
    )
    thread.start()
    writer = os.open(fifo, os.O_WRONLY)
    thread.join(timeout=5)
    assert not thread.is_alive()
    descriptor = opened.pop()
    try:
        os.write(writer, b"stop\n")
        assert os.read(descriptor, 4096) == b"stop\n"
    finally:
        os.close(writer)
        os.close(descriptor)

    opened = []
    thread = threading.Thread(
        target=lambda: opened.append(keeper_module._open_controller_fifo(fifo))
    )
    thread.start()
    writer = os.open(fifo, os.O_WRONLY)
    thread.join(timeout=5)
    descriptor = opened.pop()
    os.close(writer)
    try:
        assert os.read(descriptor, 4096) == b""
    finally:
        os.close(descriptor)


def test_keeper_lock_prevents_distinct_anchor_roots_for_one_identity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """One job/node identity cannot acquire a second keeper lifetime lock."""
    plan = _plan(tmp_path, {"image": b"image"})
    monkeypatch.setattr(
        keeper_module,
        "open_tree_root",
        lambda _path: os.open(tmp_path, os.O_RDONLY | os.O_DIRECTORY),
    )
    first = keeper_module._acquire_keeper_lock(plan)
    try:
        plan_with_other_anchor_root = replace(
            plan, anchor_root=_scratch_root(plan.job_id) / "other-anchors"
        )
        with pytest.raises(KeeperError, match="already exists"):
            keeper_module._acquire_keeper_lock(plan_with_other_anchor_root)
    finally:
        os.close(first)


def test_signal_during_staging_publishes_a_canonical_failed_receipt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """TERM during pre-publication staging leaves controller-readable failure evidence."""
    plan = _plan(tmp_path, {"image": b"image"})
    anchor_root = tmp_path / "private-anchors"
    anchor_root.mkdir(mode=0o700)
    lock = tmp_path / "keeper.lock"
    lock.write_text("")
    monkeypatch.setattr(keeper_module, "_process_start_ticks", lambda _pid: 1)
    monkeypatch.setattr(
        keeper_module, "_acquire_keeper_lock", lambda _plan: os.open(lock, os.O_RDWR)
    )
    monkeypatch.setattr(
        keeper_module,
        "_create_private_anchor_root",
        lambda _plan: os.open(anchor_root, os.O_RDONLY | os.O_DIRECTORY),
    )

    def interrupt_staging(*_args: object) -> tuple[int, int, str]:
        os.kill(os.getpid(), signal.SIGTERM)
        raise AssertionError("signal handler must interrupt staging")

    monkeypatch.setattr(keeper_module, "stage_regular_to_tmpfile", interrupt_staging)
    receipt_path = tmp_path / "failed.json"
    keeper_module.serve(plan, receipt_path)
    assert keeper_module.load_receipt(receipt_path.with_name("failed.json.failed")).items == ()


def test_cleanup_race_hook_cannot_delete_a_foreign_replacement(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A replacement at the former final-cleanup point remains untouched."""
    anchor_root = tmp_path / "anchors"
    anchor_root.mkdir(mode=0o700)
    anchor = anchor_root / "image"
    anchor.symlink_to("/proc/1/fd/1")
    metadata = anchor.lstat()
    owned = keeper_module._OwnedAnchor("image", metadata.st_dev, metadata.st_ino, "/proc/1/fd/1")

    def replace_anchor() -> None:
        anchor.unlink()
        anchor.write_text("foreign")

    monkeypatch.setattr(keeper_module, "_CLEANUP_RACE_HOOK", replace_anchor)
    descriptor = os.open(anchor_root, os.O_RDONLY | os.O_DIRECTORY)
    try:
        keeper_module._cleanup_owned_anchors(descriptor, (owned,))
    finally:
        os.close(descriptor)
    assert anchor.read_text() == "foreign"
