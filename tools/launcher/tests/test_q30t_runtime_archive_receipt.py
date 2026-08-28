# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Hostile tests for canonical Q30 runtime archive-to-tree receipts."""

from __future__ import annotations

import hashlib
import io
import json
import os
import shutil
import subprocess
import tarfile
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pytest
from common.specdec import q30t_runtime_archive_receipt as module
from common.specdec.q30t_runtime_archive_receipt import (
    load_runtime_archive_tree_receipt,
    produce_runtime_archive_tree_receipt,
)
from common.specdec.q30t_runtime_identity import runtime_tree_identity

if TYPE_CHECKING:
    from collections.abc import Mapping


def file_sha256(path: Path) -> str:
    """Return the small-fixture SHA-256 used at caller boundaries."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _zstd() -> str:
    executable = shutil.which("zstd")
    if executable is None:
        pytest.skip("zstd is required for runtime archive tests")
    return executable


def make_runtime_archive(tmp_path: Path, entries: Mapping[str, bytes]) -> Path:
    """Build one zstd-compressed runtime archive fixture."""
    tar_path = tmp_path / f"runtime-{len(tuple(tmp_path.glob('*.tar')))}.tar"
    archive = tar_path.with_suffix(".tar.zst")
    with tarfile.open(tar_path, "w") as stream:
        for path, content in entries.items():
            member = tarfile.TarInfo(f"./{path}")
            member.mode = 0o755 if path.startswith("bin/") else 0o644
            member.size = len(content)
            stream.addfile(member, io.BytesIO(content))
    subprocess.run((_zstd(), "-q", "-f", str(tar_path), "-o", str(archive)), check=True)
    return archive


def make_custom_archive(tmp_path: Path, members: tuple[tarfile.TarInfo, ...]) -> Path:
    """Build one archive containing caller-supplied hostile members."""
    tar_path = tmp_path / f"hostile-{len(tuple(tmp_path.glob('hostile-*.tar')))}.tar"
    archive = tar_path.with_suffix(".tar.zst")
    with tarfile.open(tar_path, "w") as stream:
        for member in members:
            content = b"payload" if member.isreg() else None
            if content is not None:
                member.size = len(content)
            stream.addfile(member, io.BytesIO(content) if content else None)
    subprocess.run((_zstd(), "-q", "-f", str(tar_path), "-o", str(archive)), check=True)
    return archive


def exact_arguments(
    tmp_path: Path,
    archive: Path,
    *,
    extraction_name: str = "runtime",
    output_name: str = "receipt.json",
    producer: Path | None = None,
) -> dict[str, object]:
    """Return the exact production arguments for one fixture archive."""
    producer_path = producer or Path(__file__).resolve()
    return {
        "archive_path": archive,
        "expected_archive_sha256": file_sha256(archive),
        "extraction_root": tmp_path / extraction_name,
        "output_path": tmp_path / output_name,
        "source_commit": "2" * 40,
        "producer_path": producer_path,
        "producer_sha256": file_sha256(producer_path),
        "job_id": "unit-1",
    }


def test_archive_receipt_binds_archive_tree_and_producer(tmp_path: Path) -> None:
    """The receipt binds the archive, exact extraction root, tree, and producer."""
    archive = make_runtime_archive(
        tmp_path,
        {"bin/python": b"python", "pyvenv.cfg": b"home=x\n"},
    )
    receipt = produce_runtime_archive_tree_receipt(**exact_arguments(tmp_path, archive))

    assert receipt.schema_version == "q30t-runtime-archive-tree-v1"
    assert receipt.archive_sha256 == file_sha256(archive)
    assert receipt.runtime_tree_sha256 == runtime_tree_identity(tmp_path / "runtime").sha256
    assert receipt.sentinel_paths == ("bin/python", "pyvenv.cfg")
    assert (
        load_runtime_archive_tree_receipt(
            tmp_path / "receipt.json", file_sha256(tmp_path / "receipt.json")
        )
        == receipt
    )


def test_archive_receipt_inventory_includes_optional_activate(tmp_path: Path) -> None:
    """Only the three named sentinels contribute to the sentinel digest."""
    archive = make_runtime_archive(
        tmp_path,
        {
            "bin/activate": b"activate",
            "bin/python": b"python",
            "pyvenv.cfg": b"home=x\n",
            "lib/ignored.py": b"ignored",
        },
    )
    receipt = produce_runtime_archive_tree_receipt(**exact_arguments(tmp_path, archive))
    entries = {entry.path: entry for entry in runtime_tree_identity(tmp_path / "runtime").entries}
    expected_inventory = [
        {
            "path": entry.path,
            "sha256": entry.sha256,
            "size": entry.size,
            "target": entry.target,
            "type": entry.type,
        }
        for entry in (entries["bin/activate"], entries["bin/python"], entries["pyvenv.cfg"])
    ]
    expected = json.dumps(
        expected_inventory,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode()

    assert receipt.sentinel_paths == ("bin/activate", "bin/python", "pyvenv.cfg")
    assert receipt.sentinel_inventory_sha256 == hashlib.sha256(expected).hexdigest()


def test_archive_receipt_requires_exact_sentinels(tmp_path: Path) -> None:
    """A runtime without pyvenv.cfg cannot receive qualification evidence."""
    archive = make_runtime_archive(tmp_path, {"bin/python": b"python"})

    with pytest.raises(ValueError, match="required runtime sentinels"):
        produce_runtime_archive_tree_receipt(**exact_arguments(tmp_path, archive))


def test_archive_receipt_rejects_rebound_archive(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An archive mutated through its held descriptor fails closed."""
    archive = make_runtime_archive(
        tmp_path,
        {"bin/python": b"python", "pyvenv.cfg": b"home=x\n"},
    )
    monkeypatch.setattr(module, "_POST_HASH_HOOK", lambda: archive.write_bytes(b"rebound"))

    with pytest.raises(ValueError, match="changed while hashing"):
        produce_runtime_archive_tree_receipt(**exact_arguments(tmp_path, archive))


def test_archive_receipt_rejects_changed_producer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A producer mutated while hashing cannot attest the archive."""
    archive = make_runtime_archive(
        tmp_path,
        {"bin/python": b"python", "pyvenv.cfg": b"home=x\n"},
    )
    producer = tmp_path / "producer.py"
    producer.write_bytes(b"producer")
    calls = 0

    def mutate_second_hash() -> None:
        nonlocal calls
        calls += 1
        if calls == 2:
            producer.write_bytes(b"changed")

    monkeypatch.setattr(module, "_POST_HASH_HOOK", mutate_second_hash)

    with pytest.raises(ValueError, match="changed while hashing"):
        produce_runtime_archive_tree_receipt(
            **exact_arguments(tmp_path, archive, producer=producer)
        )


@pytest.mark.parametrize("input_name", ["archive", "producer"])
def test_archive_receipt_rejects_non_single_link_input(tmp_path: Path, input_name: str) -> None:
    """Archive and producer identities reject alternate hardlink names."""
    archive = make_runtime_archive(
        tmp_path,
        {"bin/python": b"python", "pyvenv.cfg": b"home=x\n"},
    )
    producer = tmp_path / "producer.py"
    producer.write_bytes(b"producer")
    target = archive if input_name == "archive" else producer
    os.link(target, tmp_path / f"{input_name}-hardlink")

    with pytest.raises(ValueError, match="single-link regular file"):
        produce_runtime_archive_tree_receipt(
            **exact_arguments(tmp_path, archive, producer=producer)
        )


def test_archive_receipt_rejects_non_fresh_extraction_without_deleting_it(
    tmp_path: Path,
) -> None:
    """A foreign extraction root remains byte-for-byte untouched."""
    archive = make_runtime_archive(
        tmp_path,
        {"bin/python": b"python", "pyvenv.cfg": b"home=x\n"},
    )
    extraction_root = tmp_path / "runtime"
    extraction_root.mkdir()
    foreign = extraction_root / "foreign"
    foreign.write_bytes(b"preserve")

    with pytest.raises(FileExistsError):
        produce_runtime_archive_tree_receipt(**exact_arguments(tmp_path, archive))
    assert foreign.read_bytes() == b"preserve"


def test_archive_receipt_rejects_foreign_output_without_replacing_it(tmp_path: Path) -> None:
    """Foreign output bytes are preserved and rejected."""
    archive = make_runtime_archive(
        tmp_path,
        {"bin/python": b"python", "pyvenv.cfg": b"home=x\n"},
    )
    output = tmp_path / "receipt.json"
    output.write_bytes(b"foreign")

    with pytest.raises(FileExistsError, match="differs"):
        produce_runtime_archive_tree_receipt(**exact_arguments(tmp_path, archive))
    assert output.read_bytes() == b"foreign"


def test_archive_receipt_adopts_only_exact_existing_bytes(tmp_path: Path) -> None:
    """An idempotent retry adopts only the identical canonical receipt."""
    archive = make_runtime_archive(
        tmp_path,
        {"bin/python": b"python", "pyvenv.cfg": b"home=x\n"},
    )
    first = produce_runtime_archive_tree_receipt(**exact_arguments(tmp_path, archive))
    original = (tmp_path / "receipt.json").read_bytes()

    second = produce_runtime_archive_tree_receipt(
        **exact_arguments(tmp_path, archive, extraction_name="retry-runtime")
    )

    assert second == first
    assert (tmp_path / "receipt.json").read_bytes() == original


def _materialized_receipt(tmp_path: Path) -> tuple[Path, dict[str, Any]]:
    archive = make_runtime_archive(
        tmp_path,
        {"bin/python": b"python", "pyvenv.cfg": b"home=x\n"},
    )
    produce_runtime_archive_tree_receipt(**exact_arguments(tmp_path, archive))
    receipt_path = tmp_path / "receipt.json"
    return receipt_path, json.loads(receipt_path.read_bytes())


def test_loader_rejects_extra_key(tmp_path: Path) -> None:
    """An approval-like extra field cannot broaden the strict schema."""
    receipt_path, payload = _materialized_receipt(tmp_path)
    payload["approval"] = True
    receipt_path.write_bytes(module.canonical_json_bytes(payload) + b"\n")

    with pytest.raises(ValueError, match="unexpected schema"):
        load_runtime_archive_tree_receipt(receipt_path, file_sha256(receipt_path))


def test_loader_rejects_reordered_noncanonical_json(tmp_path: Path) -> None:
    """Semantically equal but reordered JSON is not canonical evidence."""
    receipt_path, payload = _materialized_receipt(tmp_path)
    reordered = dict(reversed(tuple(payload.items())))
    receipt_path.write_bytes(json.dumps(reordered, separators=(",", ":")).encode() + b"\n")

    with pytest.raises(ValueError, match="not canonical"):
        load_runtime_archive_tree_receipt(receipt_path, file_sha256(receipt_path))


def test_loader_rejects_duplicate_json_key(tmp_path: Path) -> None:
    """Duplicate JSON names cannot exploit parser last-value behavior."""
    receipt_path, payload = _materialized_receipt(tmp_path)
    canonical = module.canonical_json_bytes(payload)
    forged = canonical.replace(
        b'{"archive_path":', b'{"archive_path":"duplicate","archive_path":', 1
    )
    receipt_path.write_bytes(forged + b"\n")

    with pytest.raises(ValueError, match="duplicate JSON key"):
        load_runtime_archive_tree_receipt(receipt_path, file_sha256(receipt_path))


def test_loader_rejects_wrong_self_hash(tmp_path: Path) -> None:
    """The loader independently reproduces the receipt body self-hash."""
    receipt_path, payload = _materialized_receipt(tmp_path)
    payload["receipt_sha256"] = "0" * 64
    receipt_path.write_bytes(module.canonical_json_bytes(payload) + b"\n")

    with pytest.raises(ValueError, match="self-hash"):
        load_runtime_archive_tree_receipt(receipt_path, file_sha256(receipt_path))


def test_loader_rejects_wrong_whole_file_sha(tmp_path: Path) -> None:
    """Caller-bound whole-file identity is checked before receipt replay."""
    receipt_path, _ = _materialized_receipt(tmp_path)

    with pytest.raises(ValueError, match="whole-file SHA-256 mismatch"):
        load_runtime_archive_tree_receipt(receipt_path, "0" * 64)


def test_loader_rejects_non_single_link_receipt(tmp_path: Path) -> None:
    """A receipt with another hardlink name is not immutable evidence."""
    receipt_path, _ = _materialized_receipt(tmp_path)
    os.link(receipt_path, tmp_path / "receipt-hardlink.json")

    with pytest.raises(ValueError, match="single-link regular file"):
        load_runtime_archive_tree_receipt(receipt_path, file_sha256(receipt_path))


def test_loader_rejects_unknown_sentinel_key_even_with_recomputed_self_hash(
    tmp_path: Path,
) -> None:
    """A forged extra sentinel stays invalid after self-hash recomputation."""
    receipt_path, payload = _materialized_receipt(tmp_path)
    payload["sentinel_paths"] = ["bin/python", "bin/other", "pyvenv.cfg"]
    body = {key: value for key, value in payload.items() if key != "receipt_sha256"}
    payload["receipt_sha256"] = hashlib.sha256(module.canonical_json_bytes(body)).hexdigest()
    receipt_path.write_bytes(module.canonical_json_bytes(payload) + b"\n")

    with pytest.raises(ValueError, match="sentinel inventory"):
        load_runtime_archive_tree_receipt(receipt_path, file_sha256(receipt_path))


def test_producer_rejects_output_as_its_own_producer(tmp_path: Path) -> None:
    """The output artifact cannot serve as its own producer evidence."""
    archive = make_runtime_archive(
        tmp_path,
        {"bin/python": b"python", "pyvenv.cfg": b"home=x\n"},
    )
    output = tmp_path / "receipt.json"
    output.write_bytes(b"self producer")

    with pytest.raises(ValueError, match="cannot approve itself"):
        produce_runtime_archive_tree_receipt(**exact_arguments(tmp_path, archive, producer=output))


@pytest.mark.parametrize(
    ("member", "message"),
    [
        (tarfile.TarInfo("../escaped"), "unsafe archive member path"),
        (tarfile.TarInfo("/absolute"), "unsafe archive member path"),
        (tarfile.TarInfo("./bin/../escaped"), "unsafe archive member path"),
    ],
)
def test_archive_preflight_rejects_path_traversal_before_extraction(
    tmp_path: Path, member: tarfile.TarInfo, message: str
) -> None:
    """Traversal and absolute member paths fail before root creation."""
    archive = make_custom_archive(tmp_path, (member,))

    with pytest.raises(ValueError, match=message):
        produce_runtime_archive_tree_receipt(**exact_arguments(tmp_path, archive))
    assert not (tmp_path / "runtime").exists()
    assert not (tmp_path.parent / "escaped").exists()


def test_archive_preflight_rejects_escaping_symlink(tmp_path: Path) -> None:
    """A symlink target cannot escape the fresh extraction tree."""
    link = tarfile.TarInfo("./bin/python")
    link.type = tarfile.SYMTYPE
    link.linkname = "../../outside"
    config = tarfile.TarInfo("./pyvenv.cfg")
    archive = make_custom_archive(tmp_path, (link, config))

    with pytest.raises(ValueError, match="unsafe archive link target"):
        produce_runtime_archive_tree_receipt(**exact_arguments(tmp_path, archive))
    assert not (tmp_path / "runtime").exists()


@pytest.mark.parametrize("member_type", [tarfile.CHRTYPE, tarfile.FIFOTYPE, tarfile.LNKTYPE])
def test_archive_preflight_rejects_unsafe_member_types(tmp_path: Path, member_type: bytes) -> None:
    """Devices, FIFOs, and hardlinks never reach GNU tar extraction."""
    member = tarfile.TarInfo("./unsafe")
    member.type = member_type
    if member_type == tarfile.LNKTYPE:
        member.linkname = "./pyvenv.cfg"
    archive = make_custom_archive(tmp_path, (member,))

    with pytest.raises(ValueError, match="unsupported archive member"):
        produce_runtime_archive_tree_receipt(**exact_arguments(tmp_path, archive))
    assert not (tmp_path / "runtime").exists()


def test_sbatch_is_one_node_cpu_only_and_uses_exact_runtime_root() -> None:
    """The Slurm producer preserves the sterile CPU-only runtime contract."""
    script = (
        Path(__file__).parents[1] / "common/specdec/run_q30t_runtime_archive_receipt.sbatch"
    ).read_text()

    assert "#SBATCH --nodes=1\n" in script
    assert "#SBATCH --export=NONE\n" in script
    assert "#SBATCH --gpus" not in script
    assert 'scratch_root="/raid/scratch/$USER/q30t-runtime-archive-$SLURM_JOB_ID"' in script
    assert '--extraction-root "$scratch_root/runtime"' in script
    assert "env -i PATH=/usr/bin:/bin" in script
    assert "/usr/bin/python3.12 -m common.specdec.q30t_runtime_archive_receipt produce" in script
