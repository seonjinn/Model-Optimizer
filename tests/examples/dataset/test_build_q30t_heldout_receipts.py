# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Canonical Q30 held-out receipt producer contracts."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
MODULE_PATH = ROOT / "examples/dataset/build_q30t_heldout_receipts.py"


def _load_module():
    assert MODULE_PATH.is_file(), "the Q30 held-out receipt builder must exist"
    spec = importlib.util.spec_from_file_location("build_q30t_heldout_receipts", MODULE_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    sys.path.insert(0, str(MODULE_PATH.parent))
    try:
        spec.loader.exec_module(module)
    finally:
        sys.path.pop(0)
    return module


def _canonical(value: object) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()


def _write_source(path: Path, uuids: list[str]) -> tuple[Path, str]:
    path.write_bytes(_canonical({"prompt_uuids": uuids}) + b"\n")
    return path, hashlib.sha256(path.read_bytes()).hexdigest()


def _write_manifest_file(tmp_path: Path, names: list[str]) -> tuple[Path, str]:
    module = _load_module()
    sources = []
    for index, name in enumerate(names, start=1):
        path, digest = _write_source(tmp_path / f"{name}-source.json", [f"{index:064x}"])
        sources.append(
            {
                "name": name,
                "path": str(path),
                "sha256": digest,
                "repository": f"fixture/{name}",
                "revision": f"{index:040x}",
                "variant": "held-out",
            }
        )
    manifest = tmp_path / "sources.json"
    manifest.write_bytes(
        _canonical(
            {
                "schema_version": module.HELD_OUT_SOURCES_SCHEMA,
                "sources": sources,
            }
        )
        + b"\n"
    )
    return manifest, hashlib.sha256(manifest.read_bytes()).hexdigest()


def _write_manifest(tmp_path: Path, names: list[str]):
    module = _load_module()
    manifest, digest = _write_manifest_file(tmp_path, names)
    return module.load_sources_manifest(manifest, digest)


@pytest.mark.parametrize(
    ("uuids", "error"),
    [
        (["A" * 64], "lowercase"),
        (["a" * 64, "a" * 64], "duplicate"),
        (["b" * 64, "a" * 64], "sorted"),
        (["a" * 63], "64"),
        ([], "empty"),
    ],
)
def test_held_out_source_rejects_noncanonical_uuid_sets(
    tmp_path: Path, uuids: list[str], error: str
) -> None:
    module = _load_module()
    source, digest = _write_source(tmp_path / "source.json", uuids)

    with pytest.raises(module.HeldOutError, match=error):
        module.load_heldout_source(source, digest)


def test_held_out_receipts_are_emitted_in_fixed_name_order_independent_of_manifest_order(
    tmp_path: Path,
) -> None:
    module = _load_module()
    sources = _write_manifest(tmp_path, ["tool", "speed", "swe", "code", "math"])

    outputs = module.build_all(sources, tmp_path / "receipts", job_id="fixed-order")

    assert tuple(outputs) == ("speed", "math", "code", "swe", "tool")


def test_held_out_manifest_rejects_duplicate_names_before_source_set(tmp_path: Path) -> None:
    module = _load_module()
    manifest, _ = _write_manifest_file(tmp_path, ["speed", "speed", "math", "code", "swe", "tool"])

    with pytest.raises(module.HeldOutError, match="duplicate name"):
        module.load_sources_manifest(manifest, hashlib.sha256(manifest.read_bytes()).hexdigest())


def test_held_out_source_reader_rejects_fifo_and_hardlink_without_blocking(
    tmp_path: Path,
) -> None:
    module = _load_module()
    fifo = tmp_path / "fifo.json"
    os.mkfifo(fifo)
    with pytest.raises(module.HeldOutError, match="single-link regular"):
        module.load_heldout_source(fifo, "0" * 64)

    source, digest = _write_source(tmp_path / "source.json", ["a" * 64])
    os.link(source, tmp_path / "second-link.json")
    with pytest.raises(module.HeldOutError, match="single-link regular"):
        module.load_heldout_source(source, digest)


def test_held_out_source_reader_rejects_path_replacement(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _load_module()
    source, digest = _write_source(tmp_path / "source.json", ["a" * 64])
    raw = source.read_bytes()
    original_read = module.os.read
    attacked = False

    def replace_after_read(descriptor: int, size: int) -> bytes:
        nonlocal attacked
        block = original_read(descriptor, size)
        if block and not attacked:
            attacked = True
            source.rename(tmp_path / "original.json")
            source.write_bytes(raw)
        return block

    monkeypatch.setattr(module.os, "read", replace_after_read)

    with pytest.raises(module.HeldOutError, match="changed while reading"):
        module.load_heldout_source(source, digest)
    assert attacked


def test_held_out_source_reader_rejects_parent_symlink_rebind(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _load_module()
    parent = tmp_path / "sources"
    parent.mkdir()
    source, digest = _write_source(parent / "source.json", ["a" * 64])
    moved = tmp_path / "moved-sources"
    original_read = module.os.read
    attacked = False

    def rebind_parent(descriptor: int, size: int) -> bytes:
        nonlocal attacked
        block = original_read(descriptor, size)
        if block and not attacked:
            attacked = True
            parent.rename(moved)
            parent.symlink_to(moved, target_is_directory=True)
        return block

    monkeypatch.setattr(module.os, "read", rebind_parent)

    with pytest.raises(module.HeldOutError, match="changed while reading"):
        module.load_heldout_source(source, digest)
    assert attacked
    assert (moved / source.name).read_bytes().endswith(b"\n")


def test_held_out_source_reader_rejects_growth_during_bounded_read(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _load_module()
    source, digest = _write_source(tmp_path / "source.json", ["a" * 64])
    original_read = module.os.read
    attacked = False

    def grow_after_read(descriptor: int, size: int) -> bytes:
        nonlocal attacked
        block = original_read(descriptor, size)
        if block and not attacked:
            attacked = True
            with source.open("ab") as stream:
                stream.write(b"foreign-growth")
        return block

    monkeypatch.setattr(module.os, "read", grow_after_read)

    with pytest.raises(module.HeldOutError, match="changed while reading"):
        module.load_heldout_source(source, digest)
    assert attacked


def test_held_out_consumer_and_provenance_receipts_are_separate_and_cross_bound(
    tmp_path: Path,
) -> None:
    module = _load_module()
    sources = _write_manifest(tmp_path, ["speed", "math", "code", "swe", "tool"])
    speed, math = sources[:2]
    speed_receipt = module.build_heldout_receipt(speed)
    math_receipt = module.build_heldout_receipt(math)
    speed_provenance = module.build_heldout_provenance(speed, speed_receipt)

    consumer = json.loads(speed_receipt)
    provenance = module.verify_heldout_provenance(
        speed_provenance, receipt=speed_receipt, source=speed
    )
    assert set(consumer) == {
        "schema_version",
        "prompt_uuids",
        "prompt_uuids_sha256",
        "receipt_sha256",
    }
    assert provenance["name"] == "speed"
    with pytest.raises(module.HeldOutError, match="does not match its source"):
        module.verify_heldout_provenance(speed_provenance, receipt=math_receipt, source=speed)
    with pytest.raises(module.HeldOutError, match="does not match its source"):
        module.verify_heldout_provenance(speed_provenance, receipt=speed_receipt, source=math)


def test_held_out_empty_source_set_is_never_synthesized(tmp_path: Path) -> None:
    module = _load_module()
    with pytest.raises(module.HeldOutError, match="exact five names"):
        module.build_all((), tmp_path / "receipts", job_id="empty")
    assert not (tmp_path / "receipts").exists()


def test_held_out_publication_adopts_exact_bytes_and_never_clobbers_foreign_destination(
    tmp_path: Path,
) -> None:
    module = _load_module()
    destination = tmp_path / "speed.json"
    destination.write_bytes(b"canonical\n")
    module.publish_receipt(destination, b"canonical\n", job_id="adopt")
    destination.write_bytes(b"foreign\n")

    with pytest.raises(FileExistsError):
        module.publish_receipt(destination, b"canonical\n", job_id="foreign")
    assert destination.read_bytes() == b"foreign\n"


@pytest.mark.parametrize("kind", ["symlink", "hardlink", "fifo"])
def test_held_out_publication_rejects_nonregular_or_multiply_linked_destinations(
    tmp_path: Path, kind: str
) -> None:
    module = _load_module()
    destination = tmp_path / "speed.json"
    foreign = tmp_path / "foreign.json"
    foreign.write_bytes(b"foreign\n")
    if kind == "symlink":
        destination.symlink_to(foreign)
    elif kind == "hardlink":
        os.link(foreign, destination)
    else:
        os.mkfifo(destination)

    with pytest.raises(FileExistsError):
        module.publish_receipt(destination, b"canonical\n", job_id=kind)
    assert foreign.read_bytes() == b"foreign\n"


def test_held_out_adoption_rejects_parent_rebind_when_same_file_is_moved_back(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _load_module()
    parent = tmp_path / "receipts"
    parent.mkdir()
    destination = parent / "speed.json"
    destination.write_bytes(b"canonical\n")
    moved = tmp_path / "moved-receipts"
    original_read = module.os.read
    attacked = False

    def rebind_parent(descriptor: int, size: int) -> bytes:
        nonlocal attacked
        block = original_read(descriptor, size)
        if block and not attacked:
            attacked = True
            parent.rename(moved)
            parent.mkdir()
            (moved / destination.name).rename(destination)
        return block

    monkeypatch.setattr(module.os, "read", rebind_parent)

    with pytest.raises(FileExistsError, match="cannot be adopted"):
        module.publish_receipt(destination, b"canonical\n", job_id="parent-rebind")
    assert attacked
    assert destination.read_bytes() == b"canonical\n"
    assert moved.is_dir()


def test_held_out_cli_build_and_verify_publish_exact_ordered_review(tmp_path: Path) -> None:
    module = _load_module()
    manifest, manifest_sha256 = _write_manifest_file(
        tmp_path, ["tool", "speed", "swe", "code", "math"]
    )
    receipt_root = tmp_path / "receipts"
    review_output = tmp_path / "review.json"

    subprocess.run(
        [
            sys.executable,
            str(MODULE_PATH),
            "build",
            "--sources-manifest",
            str(manifest),
            "--sources-manifest-sha256",
            manifest_sha256,
            "--output-root",
            str(receipt_root),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    subprocess.run(
        [
            sys.executable,
            str(MODULE_PATH),
            "verify",
            "--sources-manifest",
            str(manifest),
            "--sources-manifest-sha256",
            manifest_sha256,
            "--receipt-root",
            str(receipt_root),
            "--review-output",
            str(review_output),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    review = json.loads(review_output.read_bytes())
    assert [entry["name"] for entry in review["receipts"]] == list(module.HELD_OUT_ORDER)
    assert review["schema_version"] == "q30t-held-out-review-v1"
