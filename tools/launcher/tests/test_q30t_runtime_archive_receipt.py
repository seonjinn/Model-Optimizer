# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Hostile tests for canonical Q30 runtime archive-to-tree receipts."""

from __future__ import annotations

import hashlib
import io
import json
import os
import shlex
import shutil
import subprocess
import sys
import tarfile
import tempfile
from dataclasses import dataclass
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


_APPROVED_ARCHIVE = (
    "/lustre/fsw/coreai_dlalgo_llm/users/sna/modelopt-qwen3-drafter-training/assets/"
    "q235-training-prereqs-vllm0271-v1/runtime/"
    "modelopt-vllm-0.27.1-py312-aarch64-symlinks.tar.zst"
)
_APPROVED_ARCHIVE_SHA256 = "4a20aee61f290c48bed22a84b4a0ae0cbdc54e3e3910854d253188c8854f5dc9"
_RUNNER = Path(__file__).parents[1] / "common/specdec/run_q30t_runtime_archive_receipt.sbatch"
_NON_LINUX_FAKE_TOOLS = pytest.mark.skipif(
    sys.platform == "linux",
    reason="Linux production rejects the non-Linux fake-tool boundary",
)
_LINUX_PRODUCTION_TOOL_PATHS = (
    Path("/bin/bash"),
    Path("/usr/bin/env"),
    Path("/usr/bin/python3"),
    Path("/usr/bin/python3.12"),
    Path("/usr/bin/tar"),
    Path("/usr/bin/zstd"),
    Path("/usr/bin/dirname"),
    Path("/bin/mkdir"),
    Path("/usr/bin/uname"),
)
_REAL_OR_FAKE_RUNNER_TOOLS = pytest.mark.skipif(
    sys.platform == "linux"
    and not all(
        path.is_file() and os.access(path, os.X_OK) for path in _LINUX_PRODUCTION_TOOL_PATHS
    ),
    reason="Linux host lacks the fixed production runner toolset",
)


@pytest.fixture(autouse=True)
def _explicit_non_linux_functional_snapshot(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Enable an explicitly unsealed fallback only for non-security local tests."""
    if sys.platform == "linux":
        return

    def open_unsealed_snapshot(parent: Path) -> int:
        with tempfile.TemporaryFile(prefix="q30t-test-snapshot-", dir=parent) as temporary:
            return os.dup(temporary.fileno())

    monkeypatch.setattr(module, "_UNSEALED_TEST_SNAPSHOT_FACTORY", open_unsealed_snapshot)


@dataclass(frozen=True)
class RunnerHarness:
    """Controlled non-Linux tool boundary for behavioral sbatch tests."""

    environment: dict[str, str]
    arguments: tuple[str, ...]
    capture_path: Path
    tools: dict[str, Path]


def _write_executable(path: Path, body: str) -> None:
    """Create one explicit test executable without PATH lookup."""
    path.write_text(f"#!/bin/bash\nset -euo pipefail\n{body}\n")
    path.chmod(0o700)


def _runner_harness(tmp_path: Path) -> RunnerHarness:
    """Build explicit fake tools and exact valid runner arguments."""
    tool_root = tmp_path / "tools"
    tool_root.mkdir()
    capture_path = tmp_path / "env-arguments.txt"
    output_parent = tmp_path / "receipts"
    output_parent.mkdir()
    source_checkout = tmp_path / "source"
    source_checkout.mkdir()
    tools = {
        name: tool_root / name
        for name in (
            "bash",
            "env",
            "python3.12",
            "tar",
            "zstd",
            "dirname",
            "mkdir",
        )
    }
    for name in ("bash", "python3.12", "tar", "zstd"):
        _write_executable(tools[name], "exit 0")
    _write_executable(tools["dirname"], f"echo {shlex.quote(str(output_parent))}")
    _write_executable(tools["mkdir"], "exit 0")
    _write_executable(
        tools["env"],
        "capture="
        + shlex.quote(str(capture_path))
        + '\n: > "$capture"\nfor argument in "$@"; do printf \'%s\\n\' "$argument" >> "$capture"; done',
    )
    environment = {
        "SLURM_EXPORT_ENV": "NONE",
        "SLURM_JOB_ID": "12345",
        "SLURM_NNODES": "1",
        "USER": "runner-test",
    }
    if sys.platform != "linux":
        environment.update(
            {
                "Q30T_TEST_ALLOW_SYSTEM_EXECUTABLES": "non-linux-test",
                "Q30T_TEST_BASH": str(tools["bash"]),
                "Q30T_TEST_ENV": str(tools["env"]),
                "Q30T_TEST_PYTHON": str(tools["python3.12"]),
                "Q30T_TEST_TAR": str(tools["tar"]),
                "Q30T_TEST_ZSTD": str(tools["zstd"]),
                "Q30T_TEST_DIRNAME": str(tools["dirname"]),
                "Q30T_TEST_MKDIR": str(tools["mkdir"]),
            }
        )
    arguments = (
        _APPROVED_ARCHIVE,
        _APPROVED_ARCHIVE_SHA256,
        str(source_checkout),
        "2" * 40,
        "3" * 64,
        str(output_parent / "receipt.json"),
        str(output_parent),
    )
    return RunnerHarness(environment, arguments, capture_path, tools)


def _run_runner(
    harness: RunnerHarness,
    *,
    environment_updates: Mapping[str, str] | None = None,
    arguments: tuple[str, ...] | None = None,
) -> subprocess.CompletedProcess[str]:
    """Execute the real sbatch artifact through privileged Bash."""
    environment = harness.environment | dict(environment_updates or {})
    return subprocess.run(
        ("/bin/bash", "-p", str(_RUNNER), *(arguments or harness.arguments)),
        check=False,
        capture_output=True,
        text=True,
        env=environment,
    )


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


@pytest.mark.skipif(
    sys.platform != "linux", reason="immutable snapshot trust requires sealed memfd"
)
def test_archive_extraction_uses_private_snapshot_after_source_mutation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Mutation after preflight cannot make tar consume mutable source bytes."""
    archive = make_runtime_archive(
        tmp_path,
        {"bin/python": b"python", "pyvenv.cfg": b"home=x\n"},
    )
    replacement = make_runtime_archive(
        tmp_path,
        {"bin/python": b"attacker", "pyvenv.cfg": b"home=attacker\n"},
    )
    arguments = exact_arguments(tmp_path, archive)

    monkeypatch.setattr(
        module,
        "_PRE_EXTRACT_HOOK",
        lambda: archive.write_bytes(replacement.read_bytes()),
    )

    with pytest.raises(ValueError, match="changed while extracting"):
        produce_runtime_archive_tree_receipt(**arguments)
    assert (tmp_path / "runtime/bin/python").read_bytes() == b"python"


@pytest.mark.skipif(sys.platform == "linux", reason="Linux production uses sealed memfd")
def test_non_linux_production_rejects_unsealed_snapshot(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Production cannot qualify an archive without kernel-enforced sealing."""
    archive = make_runtime_archive(
        tmp_path,
        {"bin/python": b"python", "pyvenv.cfg": b"home=x\n"},
    )
    monkeypatch.setattr(module, "_UNSEALED_TEST_SNAPSHOT_FACTORY", None)

    with pytest.raises(RuntimeError, match="sealed snapshots require Linux"):
        produce_runtime_archive_tree_receipt(**exact_arguments(tmp_path, archive))
    assert not (tmp_path / "runtime").exists()


def test_archive_extraction_rejects_root_rebind_before_tar_side_effects(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Tar never follows a replacement extraction-root pathname."""
    archive = make_runtime_archive(
        tmp_path,
        {"bin/python": b"python", "pyvenv.cfg": b"home=x\n"},
    )
    extraction_root = tmp_path / "runtime"
    detached_root = tmp_path / "detached-runtime"
    foreign_root = tmp_path / "foreign-runtime"
    foreign_root.mkdir()

    def rebind_root() -> None:
        extraction_root.rename(detached_root)
        extraction_root.symlink_to(foreign_root, target_is_directory=True)

    monkeypatch.setattr(module, "_PRE_EXTRACT_HOOK", rebind_root)

    with pytest.raises(ValueError, match="extraction root changed before extracting"):
        produce_runtime_archive_tree_receipt(**exact_arguments(tmp_path, archive))
    assert tuple(foreign_root.iterdir()) == ()
    assert tuple(detached_root.iterdir()) == ()


def test_archive_tar_remains_bound_when_root_rebinds_after_precheck(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A post-check pathname replacement cannot redirect tar side effects."""
    archive = make_runtime_archive(
        tmp_path,
        {"bin/python": b"python", "pyvenv.cfg": b"home=x\n"},
    )
    extraction_root = tmp_path / "runtime"
    detached_root = tmp_path / "detached-runtime"
    foreign_root = tmp_path / "foreign-runtime"
    foreign_root.mkdir()
    real_run = subprocess.run

    def rebind_at_tar(*args: object, **kwargs: object) -> subprocess.CompletedProcess[bytes]:
        command = args[0]
        if isinstance(command, tuple) and command[0] == "/usr/bin/tar":
            extraction_root.rename(detached_root)
            extraction_root.symlink_to(foreign_root, target_is_directory=True)
        return real_run(*args, **kwargs)  # type: ignore[arg-type, return-value]

    monkeypatch.setattr(module.subprocess, "run", rebind_at_tar)

    with pytest.raises(ValueError, match="extraction root changed after extracting"):
        produce_runtime_archive_tree_receipt(**exact_arguments(tmp_path, archive))
    assert tuple(foreign_root.iterdir()) == ()
    assert (detached_root / "bin/python").read_bytes() == b"python"


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


def test_stable_hash_reads_exact_initial_size_then_probes_growth(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Stable hashing never reads beyond the initial size except one probe byte."""
    source = tmp_path / "source.bin"
    source.write_bytes(b"abc")
    requested_sizes: list[int] = []
    real_read = os.read

    def tracked_read(descriptor: int, size: int) -> bytes:
        requested_sizes.append(size)
        return real_read(descriptor, size)

    monkeypatch.setattr(module.os, "read", tracked_read)

    assert module.stable_regular_file_sha256(source) == (3, hashlib.sha256(b"abc").hexdigest())
    assert requested_sizes == [3, 1]


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


def test_archive_receipt_propagates_atomic_durability_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An exact destination cannot conceal a publisher durability failure."""
    archive = make_runtime_archive(
        tmp_path,
        {"bin/python": b"python", "pyvenv.cfg": b"home=x\n"},
    )

    def fail_after_install(destination: Path, payload: bytes, *, job_id: str) -> None:
        del job_id
        destination.write_bytes(payload)
        raise RuntimeError("simulated fsync failure")

    monkeypatch.setattr(module, "atomic_publish_bytes", fail_after_install)

    with pytest.raises(RuntimeError, match="simulated fsync failure"):
        produce_runtime_archive_tree_receipt(**exact_arguments(tmp_path, archive))


def test_archive_parent_content_churn_does_not_look_like_path_rebind(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Stable directory identity ignores ordinary concurrent content changes."""
    archive = make_runtime_archive(
        tmp_path,
        {"bin/python": b"python", "pyvenv.cfg": b"home=x\n"},
    )
    real_open = os.open
    churned = False

    def open_with_content_churn(
        path: str | bytes | os.PathLike[str] | os.PathLike[bytes],
        flags: int,
        mode: int = 0o777,
        *,
        dir_fd: int | None = None,
    ) -> int:
        nonlocal churned
        descriptor = real_open(path, flags, mode, dir_fd=dir_fd)
        if not churned and path == tmp_path.name and dir_fd is not None:
            marker = real_open(
                "concurrent-entry",
                os.O_WRONLY | os.O_CREAT | os.O_EXCL,
                0o600,
                dir_fd=descriptor,
            )
            os.close(marker)
            churned = True
        return descriptor

    monkeypatch.setattr(module.os, "open", open_with_content_churn)

    produce_runtime_archive_tree_receipt(**exact_arguments(tmp_path, archive))

    assert churned


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


def test_archive_preflight_rejects_c1_control_in_member_path(tmp_path: Path) -> None:
    """C1 control characters cannot enter extracted member names."""
    member = tarfile.TarInfo("./bin/\x85python")
    archive = make_custom_archive(tmp_path, (member,))

    with pytest.raises(ValueError, match="unsafe archive member path"):
        produce_runtime_archive_tree_receipt(**exact_arguments(tmp_path, archive))
    assert not (tmp_path / "runtime").exists()


def test_archive_preflight_rejects_c1_control_in_link_target(tmp_path: Path) -> None:
    """C1 control characters cannot enter symlink targets."""
    link = tarfile.TarInfo("./bin/python")
    link.type = tarfile.SYMTYPE
    link.linkname = "python\x85target"
    config = tarfile.TarInfo("./pyvenv.cfg")
    archive = make_custom_archive(tmp_path, (link, config))

    with pytest.raises(ValueError, match="unsafe archive link target"):
        produce_runtime_archive_tree_receipt(**exact_arguments(tmp_path, archive))
    assert not (tmp_path / "runtime").exists()


@pytest.mark.parametrize(
    "member_type",
    [
        tarfile.CHRTYPE,
        tarfile.FIFOTYPE,
        tarfile.LNKTYPE,
        tarfile.CONTTYPE,
        tarfile.GNUTYPE_SPARSE,
    ],
)
def test_archive_preflight_rejects_unsafe_member_types(tmp_path: Path, member_type: bytes) -> None:
    """Non-ordinary files never reach GNU tar extraction."""
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
    script = _RUNNER.read_text()

    assert "#SBATCH --nodes=1\n" in script
    assert "#SBATCH --export=NONE\n" in script
    assert "#SBATCH --gpus" not in script
    assert 'scratch_root="/raid/scratch/$USER/q30t-runtime-archive-$SLURM_JOB_ID"' in script
    assert '--extraction-root "$scratch_root/runtime"' in script


@_NON_LINUX_FAKE_TOOLS
def test_sbatch_executes_with_authenticated_test_tools_and_sterile_env(tmp_path: Path) -> None:
    """The runner reaches Python only through explicit validated tools and env -i."""
    harness = _runner_harness(tmp_path)

    result = _run_runner(harness, environment_updates={"PATH": str(tmp_path / "attacker")})

    assert result.returncode == 0, result.stderr
    captured = harness.capture_path.read_text().splitlines()
    assert captured[:4] == [
        "-i",
        "PATH=/usr/bin:/bin",
        f"PYTHONPATH={harness.arguments[2]}/tools/launcher",
        str(harness.tools["python3.12"]),
    ]
    assert "common.specdec.q30t_runtime_archive_receipt" in captured


def test_sbatch_behaviorally_rejects_wrong_arity(tmp_path: Path) -> None:
    """The real runner rejects any positional shape other than seven."""
    harness = _runner_harness(tmp_path)

    result = _run_runner(harness, arguments=harness.arguments[:-1])

    assert result.returncode == 2
    assert "usage:" in result.stderr


@pytest.mark.parametrize(
    ("argument_index", "replacement"),
    [(0, "/lustre/foreign.tar.zst"), (1, "0" * 64)],
)
@_REAL_OR_FAKE_RUNNER_TOOLS
def test_sbatch_behaviorally_rejects_wrong_fixed_archive_identity(
    tmp_path: Path, argument_index: int, replacement: str
) -> None:
    """The production archive path and SHA cannot be substituted."""
    harness = _runner_harness(tmp_path)
    arguments = list(harness.arguments)
    arguments[argument_index] = replacement

    result = _run_runner(harness, arguments=tuple(arguments))

    assert result.returncode == 2
    assert "runtime archive identity is not approved" in result.stderr


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("SLURM_JOB_GPUS", "0"),
        ("SLURM_STEP_GPUS", "0"),
        ("SLURM_GPUS_ON_NODE", "0"),
        ("SLURM_GPUS", "0"),
        ("SLURM_GPUS_PER_NODE", "0"),
        ("SLURM_GPUS_PER_TASK", "0"),
        ("SLURM_TRES_PER_NODE", "gpu:1"),
    ],
)
@_REAL_OR_FAKE_RUNNER_TOOLS
def test_sbatch_behaviorally_rejects_gpu_environment(tmp_path: Path, name: str, value: str) -> None:
    """GPU variables and GPU TRES fail the CPU-only allocation boundary."""
    harness = _runner_harness(tmp_path)

    result = _run_runner(harness, environment_updates={name: value})

    assert result.returncode == 2
    assert "requires a CPU-only allocation" in result.stderr


@_NON_LINUX_FAKE_TOOLS
def test_sbatch_behaviorally_rejects_output_parent_mismatch(tmp_path: Path) -> None:
    """The seventh argument must be the exact output pathname parent."""
    harness = _runner_harness(tmp_path)
    arguments = (*harness.arguments[:-1], str(tmp_path / "foreign-parent"))

    result = _run_runner(harness, arguments=arguments)

    assert result.returncode == 2
    assert "receipt destination is invalid" in result.stderr


def test_sbatch_behaviorally_rejects_prohibited_python_environment(tmp_path: Path) -> None:
    """Python environment injection is rejected before any tool executes."""
    harness = _runner_harness(tmp_path)

    result = _run_runner(harness, environment_updates={"PYTHONPATH": "/attacker"})

    assert result.returncode == 2
    assert "prohibited exported environment variable: PYTHONPATH" in result.stderr
    assert not harness.capture_path.exists()


def test_sbatch_behaviorally_rejects_exported_spoofed_ostype(tmp_path: Path) -> None:
    """An exported platform claim cannot authorize user-controlled tools."""
    harness = _runner_harness(tmp_path)

    result = _run_runner(harness, environment_updates={"OSTYPE": "darwin-test"})

    assert result.returncode == 2
    assert "prohibited exported environment variable: OSTYPE" in result.stderr
    assert not harness.capture_path.exists()


@pytest.mark.skipif(sys.platform != "linux", reason="Linux-specific production boundary")
@pytest.mark.parametrize(
    "override_name",
    [
        "Q30T_TEST_ALLOW_SYSTEM_EXECUTABLES",
        "Q30T_TEST_BASH",
        "Q30T_TEST_ENV",
        "Q30T_TEST_PYTHON",
        "Q30T_TEST_TAR",
        "Q30T_TEST_ZSTD",
        "Q30T_TEST_DIRNAME",
        "Q30T_TEST_MKDIR",
    ],
)
def test_sbatch_linux_rejects_every_fake_tool_override_before_execution(
    tmp_path: Path, override_name: str
) -> None:
    """Authenticated Linux identity makes each test override fail before its marker."""
    harness = _runner_harness(tmp_path)
    override_values = {
        "Q30T_TEST_ALLOW_SYSTEM_EXECUTABLES": "non-linux-test",
        "Q30T_TEST_BASH": str(harness.tools["bash"]),
        "Q30T_TEST_ENV": str(harness.tools["env"]),
        "Q30T_TEST_PYTHON": str(harness.tools["python3.12"]),
        "Q30T_TEST_TAR": str(harness.tools["tar"]),
        "Q30T_TEST_ZSTD": str(harness.tools["zstd"]),
        "Q30T_TEST_DIRNAME": str(harness.tools["dirname"]),
        "Q30T_TEST_MKDIR": str(harness.tools["mkdir"]),
    }

    result = _run_runner(
        harness,
        environment_updates={override_name: override_values[override_name]},
    )

    assert result.returncode == 2
    assert "test executable configuration is invalid" in result.stderr
    assert not harness.capture_path.exists()


@pytest.mark.parametrize(
    "tool_name",
    ["bash", "env", "python3.12", "tar", "zstd", "dirname", "mkdir"],
)
@_NON_LINUX_FAKE_TOOLS
def test_sbatch_behaviorally_rejects_symlink_tool_substitution(
    tmp_path: Path, tool_name: str
) -> None:
    """Even test-mode explicit tools must be regular, never symlink substitutions."""
    harness = _runner_harness(tmp_path)
    tool = harness.tools[tool_name]
    tool.unlink()
    tool.symlink_to("/bin/true")

    result = _run_runner(harness)

    assert result.returncode == 2
    assert "approved test executable is unavailable" in result.stderr
    assert not harness.capture_path.exists()


@_NON_LINUX_FAKE_TOOLS
def test_sbatch_behaviorally_rejects_relative_tool_substitution(tmp_path: Path) -> None:
    """A relative executable override is rejected before invocation."""
    harness = _runner_harness(tmp_path)

    result = _run_runner(harness, environment_updates={"Q30T_TEST_TAR": "attacker-tar"})

    assert result.returncode == 2
    assert "test executable configuration is invalid" in result.stderr
    assert not harness.capture_path.exists()
