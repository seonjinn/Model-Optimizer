# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Behavioral contracts for the node-local descriptor keeper."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass
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
        job_id="12345",
        node_name="node-a",
        scratch_root=root,
        directory_roots=(source_root,),
        anchor_root=root / "anchors",
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

    keeper = _start_keeper(tmp_path / "keeper", {"image": b"image"})
    keeper.kill()
    with pytest.raises(KeeperError, match="keeper is not alive"):
        validate_keeper_receipt(keeper.receipt)


def test_plan_and_receipt_are_canonical_and_reject_foreign_anchor_replacement(
    tmp_path: Path,
) -> None:
    """Receipt bytes stay canonical and cleanup preserves foreign anchors."""
    _require_tmpfile(tmp_path)
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
    )
    body = {
        "items": [
            {
                "anchor_path": str(item.anchor_path),
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
