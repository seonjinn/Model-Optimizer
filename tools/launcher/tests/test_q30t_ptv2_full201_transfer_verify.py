# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Compute-side verification contracts for the Q30 Thinking PTV2 transfer."""

from __future__ import annotations

import fcntl
import hashlib
import importlib.util
import json
import os
import shutil
import stat
import subprocess
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, TimeoutError
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from collections.abc import Mapping
    from types import ModuleType

ROOT = Path(__file__).resolve().parents[3]
MODULE_PATH = ROOT / "tools/launcher/common/specdec/verify_q30t_ptv2_full201_transfer.py"
RUNNER = ROOT / "tools/launcher/common/specdec/run_q30t_ptv2_full201_transfer_verify.sbatch"
SUBMITTER = ROOT / "tools/launcher/common/specdec/submit_q30t_ptv2_full201_transfer_verify.sh"
_BASH = shutil.which("bash")
if _BASH is None:
    raise RuntimeError("bash is required for transfer verifier runner tests")
BASH: str = _BASH

REPOSITORY = "nvidia/Nemotron-Post-Training-Dataset-v2"
REVISION = "5c89e01dd720ae0f4058445ed49c5fb68a03c76e"
SPLIT_COUNTS = {"chat": 2, "stem": 1}
CANONICAL_BRANCH = "sj/q30t-ptv23-complement-700k"
CANONICAL_REMOTE_REF = f"gitlab/{CANONICAL_BRANCH}"
APPROVED_GITLAB_FETCH_URL = "ssh://git@gitlab-master.nvidia.com:12051/sna/modelopt.git"


def _load_module() -> ModuleType:
    assert MODULE_PATH.exists(), "transfer verifier is not implemented"
    spec = importlib.util.spec_from_file_location("verify_q30t_ptv2_full201_transfer", MODULE_PATH)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(spec.name, None)
        raise
    return module


def _canonical_json(value: object) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode()


def _fixture(
    module: ModuleType, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> tuple[Path, Path, dict[str, bytes]]:
    source_root = tmp_path / "q30t-ptv2-full201-source-v1"
    data_root = source_root / "data"
    data_root.mkdir(parents=True)
    contents = {
        "chat-00000-of-00002.parquet": b"chat-zero\n",
        "chat-00001-of-00002.parquet": b"chat-one!\n",
        "stem-00000-of-00001.parquet": b"stem-only\n",
    }
    grouped: list[dict[str, object]] = []
    for split, count in SPLIT_COUNTS.items():
        files = []
        for index in range(count):
            name = f"{split}-{index:05d}-of-{count:05d}.parquet"
            payload = contents[name]
            (data_root / name).write_bytes(payload)
            files.append(
                {
                    "path": f"data/{name}",
                    "bytes": len(payload),
                    "sha256": hashlib.sha256(payload).hexdigest(),
                }
            )
        grouped.append(
            {
                "repository_id": REPOSITORY,
                "configuration": "default",
                "split": split,
                "revision": REVISION,
                "license_expression": "CC-BY-4.0",
                "approved_use": True,
                "cell": split,
                "lane": "target-synth",
                "files": files,
            }
        )
    plan_bytes = _canonical_json(
        {
            "schema_version": 1,
            "name": "qwen3-4b-ptv2-full-source-v1",
            "sources": grouped,
        }
    )
    plan_path = source_root / "SOURCE_PLAN.json"
    plan_path.write_bytes(plan_bytes)
    plan_sha256 = hashlib.sha256(plan_bytes).hexdigest()
    split_bytes = {
        split: sum(
            len(payload) for name, payload in contents.items() if name.startswith(f"{split}-")
        )
        for split in SPLIT_COUNTS
    }
    completion_bytes = _canonical_json(
        {
            "schema_version": 1,
            "complete": True,
            "repository_id": REPOSITORY,
            "configuration": "default",
            "revision": REVISION,
            "license_expression": "CC-BY-4.0",
            "approved_use": True,
            "source_commit": "a" * 40,
            "source_root_realpath": "/lustre/source-symlinks",
            "approved_hao_root_realpath": "/lustre/approved-hao",
            "manifest": {
                "path": "SOURCE_PLAN.json",
                "bytes": len(plan_bytes),
                "sha256": plan_sha256,
            },
            "split_counts": SPLIT_COUNTS,
            "split_bytes": split_bytes,
            "target_inventory": {"count": 3, "sha256": "b" * 64},
            "readme": {
                "path": "/lustre/approved-hao/README.md",
                "bytes": 123,
                "sha256": "c" * 64,
                "target_device": 1,
                "target_inode": 2,
                "target_mtime_ns": 3,
            },
            "workers": {"requested": 96, "effective": 3},
            "timing": {
                "started_at_utc": "2026-08-26T00:00:00+00:00",
                "completed_at_utc": "2026-08-26T00:01:00+00:00",
                "duration_seconds": 60.0,
            },
        }
    )
    completion_path = source_root / "SOURCE_MANIFEST_COMPLETION.json"
    completion_path.write_bytes(completion_bytes)

    receipt_ancestor = tmp_path / "receipts/q30t-ptv23-complement-700k-v1"
    receipt_ancestor.mkdir(parents=True)
    output = receipt_ancestor / "ptv2-full201-transfer/VERIFY.json"
    monkeypatch.setattr(module, "APPROVED_SOURCE_ROOT", source_root)
    monkeypatch.setattr(module, "APPROVED_RECEIPT_ANCESTOR", receipt_ancestor, raising=False)
    monkeypatch.setattr(module, "APPROVED_OUTPUT_PATH", output, raising=False)
    monkeypatch.setattr(module, "APPROVED_SPLIT_COUNTS", SPLIT_COUNTS)
    monkeypatch.setattr(module, "APPROVED_FILE_COUNT", 3)
    monkeypatch.setattr(module, "APPROVED_TOTAL_BYTES", sum(map(len, contents.values())))
    monkeypatch.setattr(module, "APPROVED_SOURCE_PLAN_SHA256", plan_sha256)
    monkeypatch.setattr(
        module,
        "APPROVED_COMPLETION_FILE_SHA256",
        hashlib.sha256(completion_bytes).hexdigest(),
    )
    if sys.platform != "linux":
        monkeypatch.setattr(module, "_ALLOW_NON_LINUX_OPENAT2_TEST_FALLBACK", True)
        monkeypatch.setattr(module, "_acquire_read_lease", lambda *_: None)
        monkeypatch.setattr(module, "_require_read_lease", lambda *_: None)
        monkeypatch.setattr(module, "_release_read_lease", lambda *_: None)
    return source_root, output, contents


def test_verifier_rehashes_the_exact_tree_and_publishes_a_canonical_receipt(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Changing a shard, plan binding, or receipt field changes the published evidence."""
    module = _load_module()
    source_root, output, contents = _fixture(module, monkeypatch, tmp_path)

    receipt_bytes = module.verify_and_publish_q30t_ptv2_full201_transfer(
        source_root=source_root, output=output, workers=2
    )

    assert output.read_bytes() == receipt_bytes
    assert stat.S_IMODE(output.parent.stat().st_mode) == 0o700
    assert receipt_bytes.endswith(b"\n")
    receipt = json.loads(receipt_bytes)
    assert receipt == {
        "authorization_scope": "point-in-time-inventory-only",
        "observed_complete_at_verification": True,
        "repository_id": REPOSITORY,
        "requires_consumer_live_revalidation": True,
        "revision": REVISION,
        "schema_version": "q30t-ptv2-full201-transfer-inventory-audit-v1",
        "source_bytes": sum(map(len, contents.values())),
        "source_file_count": 3,
        "source_manifest_completion": {
            "path": "SOURCE_MANIFEST_COMPLETION.json",
            "sha256": module.APPROVED_COMPLETION_FILE_SHA256,
        },
        "source_plan": {
            "path": "SOURCE_PLAN.json",
            "sha256": module.APPROVED_SOURCE_PLAN_SHA256,
        },
        "source_root": str(source_root),
        "split_bytes": {"chat": 20, "stem": 10},
        "split_counts": SPLIT_COUNTS,
    }


def test_verifier_is_no_clobber(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """A pre-existing receipt cannot be overwritten by a later verification attempt."""
    module = _load_module()
    source_root, output, _ = _fixture(module, monkeypatch, tmp_path)
    output.parent.mkdir(mode=0o700)
    output.write_bytes(b"preserve\n")

    with pytest.raises(module.TransferVerificationError, match="already exists"):
        module.verify_and_publish_q30t_ptv2_full201_transfer(
            source_root=source_root, output=output, workers=1
        )

    assert output.read_bytes() == b"preserve\n"


def test_publication_rechecks_source_leases_immediately_before_atomic_install(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A lease break detected after receipt staging must prevent the atomic install."""
    module = _load_module()
    source_root, output, _ = _fixture(module, monkeypatch, tmp_path)
    real_require = module._ReadLeaseGuard.require_no_break
    real_rename = module._rename_no_replace_at
    checks = 0

    def count_lease_check(guard: object) -> None:
        nonlocal checks
        checks += 1
        real_require(guard)

    def require_last_check_before_install(parent_fd: int, source: str, destination: str) -> None:
        assert checks == 3
        real_rename(parent_fd, source, destination)

    monkeypatch.setattr(module._ReadLeaseGuard, "require_no_break", count_lease_check)
    monkeypatch.setattr(module, "_rename_no_replace_at", require_last_check_before_install)

    module.verify_and_publish_q30t_ptv2_full201_transfer(
        source_root=source_root, output=output, workers=2
    )

    assert checks == 3


def test_publication_preserves_a_foreign_replacement_of_its_temporary_name(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Cleanup cannot unlink a same-UID file that replaced the verifier's temporary inode."""
    module = _load_module()
    source_root, output, _ = _fixture(module, monkeypatch, tmp_path)
    real_fsync = module.os.fsync
    replacements: list[Path] = []

    def replace_after_durable_write(descriptor: int) -> None:
        real_fsync(descriptor)
        if replacements or not output.parent.exists():
            return
        temporary_files = list(output.parent.glob(f".{output.name}.partial-*"))
        if not temporary_files:
            return
        temporary = temporary_files[0]
        temporary.rename(output.parent / "owned-temporary-moved")
        temporary.write_bytes(b"foreign replacement\n")
        replacements.append(temporary)

    monkeypatch.setattr(module.os, "fsync", replace_after_durable_write)

    with pytest.raises(module.TransferVerificationError, match=r"temporary|reread"):
        module.verify_and_publish_q30t_ptv2_full201_transfer(
            source_root=source_root, output=output, workers=2
        )

    assert len(replacements) == 1
    assert replacements[0].read_bytes() == b"foreign replacement\n"
    assert not output.exists()


def test_publication_consumes_the_temporary_inode_without_a_second_link(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The installed receipt is never exposed as a second link needing pathname cleanup."""
    module = _load_module()
    source_root, output, _ = _fixture(module, monkeypatch, tmp_path)
    real_read = module._read_regular_at
    installed_link_counts: list[int] = []

    def observe_installed_link_count(
        directory_fd: int,
        name: str,
        display_path: Path,
        *,
        retain: bool,
        max_retained_bytes: int = 64 * 1024 * 1024,
        lease_guard: object | None = None,
    ) -> tuple[int, str, bytes | None, os.stat_result]:
        result = real_read(
            directory_fd,
            name,
            display_path,
            retain=retain,
            max_retained_bytes=max_retained_bytes,
            lease_guard=lease_guard,
        )
        if display_path == output:
            installed_link_counts.append(result[3].st_nlink)
        return result

    monkeypatch.setattr(module, "_read_regular_at", observe_installed_link_count)

    module.verify_and_publish_q30t_ptv2_full201_transfer(
        source_root=source_root, output=output, workers=2
    )

    assert installed_link_counts == [1, 1]
    assert not list(output.parent.glob(f".{output.name}.partial-*"))


def test_publication_rebinds_the_output_parent_after_installation(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A receipt installed through an old directory fd cannot authenticate a replacement path."""
    module = _load_module()
    source_root, output, _ = _fixture(module, monkeypatch, tmp_path)
    moved_parent = output.parent.with_name("installed-output-parent-moved")
    replacement = output.parent.with_name("foreign-output-parent")
    replacement.mkdir(mode=0o700)
    real_rename = module._rename_no_replace_at
    swapped = False

    def swap_after_install(parent_fd: int, source: str, destination: str) -> None:
        nonlocal swapped
        real_rename(parent_fd, source, destination)
        if not swapped:
            output.parent.rename(moved_parent)
            replacement.rename(output.parent)
            swapped = True

    monkeypatch.setattr(module, "_rename_no_replace_at", swap_after_install)

    with pytest.raises(module.TransferVerificationError, match="receipt parent changed"):
        module.verify_and_publish_q30t_ptv2_full201_transfer(
            source_root=source_root, output=output, workers=2
        )

    assert not output.exists()
    assert (moved_parent / output.name).is_file()


def test_publication_rebinds_the_output_parent_after_receipt_reread(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A parent swapped during final receipt reread cannot authenticate the old directory fd."""
    module = _load_module()
    source_root, output, _ = _fixture(module, monkeypatch, tmp_path)
    moved_parent = output.parent.with_name("reread-output-parent-moved")
    replacement = output.parent.with_name("reread-output-parent-foreign")
    replacement.mkdir(mode=0o700)
    real_read = module._read_regular_at
    swapped = False
    output_reads = 0

    def swap_after_receipt_reread(
        directory_fd: int,
        name: str,
        display_path: Path,
        *,
        retain: bool,
        max_retained_bytes: int = module._MAX_JSON_BYTES,
        lease_guard: object | None = None,
    ) -> object:
        nonlocal output_reads, swapped
        result = real_read(
            directory_fd,
            name,
            display_path,
            retain=retain,
            max_retained_bytes=max_retained_bytes,
            lease_guard=lease_guard,
        )
        if display_path == output:
            output_reads += 1
            if output_reads == 2 and not swapped:
                output.parent.rename(moved_parent)
                replacement.rename(output.parent)
                swapped = True
        return result

    monkeypatch.setattr(module, "_read_regular_at", swap_after_receipt_reread)

    with pytest.raises(module.TransferVerificationError, match="receipt parent changed"):
        module.verify_and_publish_q30t_ptv2_full201_transfer(
            source_root=source_root, output=output, workers=2
        )

    assert not output.exists()
    assert (moved_parent / output.name).is_file()


def test_publication_never_deletes_a_replacement_created_at_unlink_time(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Publication cannot use stat-then-unlink cleanup on a rebindable temporary name."""
    module = _load_module()
    source_root, output, _ = _fixture(module, monkeypatch, tmp_path)
    real_unlink = module.os.unlink
    replacements: list[Path] = []

    def replace_at_unlink(
        name: str | bytes,
        *,
        dir_fd: int | None = None,
    ) -> None:
        if (
            dir_fd is not None
            and isinstance(name, str)
            and name.startswith(f".{output.name}.partial-")
            and not replacements
        ):
            moved_name = f"owned-{name}"
            module.os.rename(name, moved_name, src_dir_fd=dir_fd, dst_dir_fd=dir_fd)
            descriptor = module.os.open(
                name,
                module.os.O_WRONLY | module.os.O_CREAT | module.os.O_EXCL,
                0o600,
                dir_fd=dir_fd,
            )
            try:
                module.os.write(descriptor, b"foreign replacement\n")
            finally:
                module.os.close(descriptor)
            replacements.append(output.parent / name)
        real_unlink(name, dir_fd=dir_fd)

    monkeypatch.setattr(module.os, "unlink", replace_at_unlink)

    module.verify_and_publish_q30t_ptv2_full201_transfer(
        source_root=source_root, output=output, workers=2
    )

    assert not replacements or replacements[0].read_bytes() == b"foreign replacement\n"


def test_publication_rebinds_the_receipt_file_after_atomic_install(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The final receipt path must still name the inode and bytes installed by the verifier."""
    module = _load_module()
    source_root, output, _ = _fixture(module, monkeypatch, tmp_path)
    moved_receipt = output.with_name("installed-receipt-moved.json")
    real_rename = module._rename_no_replace_at
    swapped = False

    def swap_after_install(parent_fd: int, source: str, destination: str) -> None:
        nonlocal swapped
        real_rename(parent_fd, source, destination)
        if not swapped:
            output.rename(moved_receipt)
            output.write_bytes(b"foreign receipt\n")
            swapped = True

    monkeypatch.setattr(module, "_rename_no_replace_at", swap_after_install)

    with pytest.raises(
        module.TransferVerificationError, match=r"published (verification )?receipt"
    ):
        module.verify_and_publish_q30t_ptv2_full201_transfer(
            source_root=source_root, output=output, workers=2
        )

    assert output.read_bytes() == b"foreign receipt\n"
    assert moved_receipt.is_file()


@pytest.mark.parametrize("fault", ["tamper", "extra", "symlink", "directory"])
def test_verifier_rejects_noncanonical_file_trees(
    fault: str, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Unexpected entries and anything other than the hashed regular shards fail closed."""
    module = _load_module()
    source_root, output, _ = _fixture(module, monkeypatch, tmp_path)
    shard = source_root / "data/chat-00000-of-00002.parquet"
    if fault == "tamper":
        shard.write_bytes(b"chat-ZERO\n")
    elif fault == "extra":
        (source_root / "data/orphan.parquet").write_bytes(b"orphan")
    elif fault == "symlink":
        payload = shard.read_bytes()
        target = tmp_path / "external.parquet"
        target.write_bytes(payload)
        shard.unlink()
        shard.symlink_to(target)
    else:
        shard.unlink()
        shard.mkdir()

    with pytest.raises(module.TransferVerificationError):
        module.verify_and_publish_q30t_ptv2_full201_transfer(
            source_root=source_root, output=output, workers=2
        )

    assert not output.exists()


@pytest.mark.parametrize(
    "relative_path",
    [
        "SOURCE_PLAN.json",
        "SOURCE_MANIFEST_COMPLETION.json",
        "data/chat-00000-of-00002.parquet",
    ],
)
def test_verifier_rejects_multiply_linked_inputs(
    relative_path: str, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Metadata and shard inputs must each have exactly one filesystem link."""
    module = _load_module()
    source_root, output, _ = _fixture(module, monkeypatch, tmp_path)
    os.link(source_root / relative_path, tmp_path / "foreign-hardlink")

    with pytest.raises(module.TransferVerificationError, match="link"):
        module.verify_and_publish_q30t_ptv2_full201_transfer(
            source_root=source_root, output=output, workers=2
        )

    assert not output.exists()


def test_read_lease_backend_acquires_checks_and_releases_the_linux_contract(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The lease helper requests, verifies, and releases a write-excluding read lease."""
    module = _load_module()
    monkeypatch.setattr(module, "_F_SETLEASE", 1024, raising=False)
    monkeypatch.setattr(module, "_F_GETLEASE", 1025, raising=False)
    monkeypatch.setattr(module, "_F_RDLCK", 0, raising=False)
    monkeypatch.setattr(module, "_F_UNLCK", 2, raising=False)
    active = False
    calls: list[tuple[int, int, int | None]] = []

    def fake_fcntl(descriptor: int, command: int, argument: int | None = None) -> int:
        nonlocal active
        calls.append((descriptor, command, argument))
        if command == 1024 and argument == 0:
            active = True
            return 0
        if command == 1025:
            return 0 if active else 2
        if command == 1024 and argument == 2:
            active = False
            return 0
        raise AssertionError((descriptor, command, argument))

    monkeypatch.setattr(module.fcntl, "fcntl", fake_fcntl)

    module._acquire_read_lease(17, Path("/lease/input"))
    module._require_read_lease(17, Path("/lease/input"))
    module._release_read_lease(17)

    assert calls == [(17, 1024, 0), (17, 1025, None), (17, 1024, 2)]


def test_stable_reader_does_not_block_if_a_regular_path_becomes_a_fifo(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A stat-to-open FIFO replacement fails closed without waiting for a writer."""
    module = _load_module()
    directory = tmp_path / "fifo-race"
    directory.mkdir()
    target = directory / "shard.parquet"
    target.write_bytes(b"regular before stat\n")
    directory_fd = module.os.open(
        directory,
        module.os.O_RDONLY | getattr(module.os, "O_DIRECTORY", 0),
    )
    real_stat = module.os.stat
    replaced = False

    def replace_after_stat(
        path: str | bytes | int | os.PathLike[str] | os.PathLike[bytes],
        *args: object,
        **kwargs: object,
    ) -> os.stat_result:
        nonlocal replaced
        result = real_stat(path, *args, **kwargs)
        if path == target.name and kwargs.get("dir_fd") == directory_fd and not replaced:
            module.os.unlink(target.name, dir_fd=directory_fd)
            module.os.mkfifo(target.name, mode=0o600, dir_fd=directory_fd)
            replaced = True
        return result

    monkeypatch.setattr(module.os, "stat", replace_after_stat)
    executor = ThreadPoolExecutor(max_workers=1)
    future = executor.submit(
        module._read_regular_at,
        directory_fd,
        target.name,
        target,
        retain=False,
    )
    blocked = False
    try:
        try:
            with pytest.raises(module.TransferVerificationError):
                future.result(timeout=0.25)
        except TimeoutError:
            blocked = True
            writer = module.os.open(target, module.os.O_WRONLY | module.os.O_NONBLOCK)
            module.os.close(writer)
            with pytest.raises(module.TransferVerificationError):
                future.result(timeout=1)
    finally:
        executor.shutdown(wait=True)
        module.os.close(directory_fd)

    assert not blocked, "reader blocked after the regular path was replaced by a FIFO"


def test_verifier_rejects_a_different_source_root(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A caller cannot redirect the production verifier to an unapproved tree."""
    module = _load_module()
    source_root, output, _ = _fixture(module, monkeypatch, tmp_path)

    with pytest.raises(module.TransferVerificationError, match="approved source root"):
        module.verify_and_publish_q30t_ptv2_full201_transfer(
            source_root=source_root.with_name("other"), output=output, workers=1
        )


def test_verifier_detects_a_shard_changed_after_its_stream_hash(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A shard cannot change after its descriptor hash but before the tree receipt."""
    module = _load_module()
    source_root, output, _ = _fixture(module, monkeypatch, tmp_path)
    real_hash = module._hash_and_check_shard

    def mutate_after_hash(
        data_fd: int, record: Mapping[str, object], lease_guard: object
    ) -> object:
        result = real_hash(data_fd, record, lease_guard)
        if record["path"] == "data/chat-00000-of-00002.parquet":
            (source_root / str(record["path"])).write_bytes(b"CHAT-ZERO\n")
        return result

    monkeypatch.setattr(module, "_hash_and_check_shard", mutate_after_hash)

    with pytest.raises(module.TransferVerificationError, match="changed"):
        module.verify_and_publish_q30t_ptv2_full201_transfer(
            source_root=source_root, output=output, workers=2
        )

    assert not output.exists()


@pytest.mark.skipif(sys.platform != "linux", reason="Linux F_SETLEASE is required")
def test_verifier_blocks_a_writer_after_the_first_final_shard_check(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A conflicting writer cannot mutate an early shard before receipt publication."""
    module = _load_module()
    source_root, output, _ = _fixture(module, monkeypatch, tmp_path)
    shard = source_root / "data/chat-00000-of-00002.parquet"
    real_require = module._require_unchanged_at
    writer_attempted = threading.Event()
    writer: threading.Thread | None = None

    def write_shard() -> None:
        writer_attempted.set()
        shard.write_bytes(b"writer-after-final-check\n")

    def start_writer_after_first_check(
        directory_fd: int,
        name: str,
        expected: os.stat_result,
        display_path: Path,
    ) -> None:
        nonlocal writer
        real_require(directory_fd, name, expected, display_path)
        if writer is None:
            writer = threading.Thread(target=write_shard, daemon=True)
            writer.start()
            assert writer_attempted.wait(timeout=1)
            writer.join(timeout=0.2)
            assert writer.is_alive(), "conflicting writer was not excluded by the read lease"

    monkeypatch.setattr(module, "_require_unchanged_at", start_writer_after_first_check)

    with pytest.raises(module.TransferVerificationError, match="lease break"):
        module.verify_and_publish_q30t_ptv2_full201_transfer(
            source_root=source_root, output=output, workers=2
        )

    assert writer is not None
    writer.join(timeout=2)
    assert not writer.is_alive()
    assert not output.exists()


def test_verifier_rebinds_the_approved_root_path_after_hashing(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Renaming the verified root cannot bind its receipt to a foreign replacement path."""
    module = _load_module()
    source_root, output, _ = _fixture(module, monkeypatch, tmp_path)
    moved_root = source_root.with_name("verified-root-moved")
    replacement = source_root.with_name("foreign-replacement")
    replacement.mkdir()
    root_inode = source_root.stat().st_ino
    real_fstat = module.os.fstat
    root_observations = 0
    swapped = False

    def swap_after_final_descriptor_check(descriptor: int) -> os.stat_result:
        nonlocal root_observations, swapped
        result = real_fstat(descriptor)
        if result.st_ino == root_inode:
            root_observations += 1
        if root_observations == 2 and not swapped:
            source_root.rename(moved_root)
            replacement.rename(source_root)
            swapped = True
        return result

    monkeypatch.setattr(module.os, "fstat", swap_after_final_descriptor_check)

    with pytest.raises(module.TransferVerificationError, match="approved source root changed"):
        module.verify_and_publish_q30t_ptv2_full201_transfer(
            source_root=source_root, output=output, workers=2
        )

    assert not output.exists()


def test_verifier_rejects_source_namespace_replacement_by_a_separate_process(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A separate process cannot replace the audited pathname before publication."""
    module = _load_module()
    source_root, output, _ = _fixture(module, monkeypatch, tmp_path)
    moved_root = source_root.with_name("audited-root-moved")
    replacement = source_root.with_name("foreign-root")
    replacement.mkdir()
    swap_request = tmp_path / "request-namespace-swap"
    child = subprocess.Popen(
        [
            sys.executable,
            "-c",
            (
                "import pathlib,sys,time;"
                "source,moved,replacement,request=map(pathlib.Path,sys.argv[1:]);"
                "deadline=time.monotonic()+5;"
                "\nwhile not request.exists():\n"
                " assert time.monotonic()<deadline\n"
                " time.sleep(0.01)\n"
                "source.rename(moved);replacement.rename(source)"
            ),
            str(source_root),
            str(moved_root),
            str(replacement),
            str(swap_request),
        ]
    )
    real_require = module._require_absolute_directory_binding

    def replace_before_rebinding(path: Path, expected: os.stat_result, *, message: str) -> None:
        swap_request.touch()
        assert child.wait(timeout=5) == 0
        real_require(path, expected, message=message)

    monkeypatch.setattr(module, "_require_absolute_directory_binding", replace_before_rebinding)
    try:
        with pytest.raises(module.TransferVerificationError, match="approved source root changed"):
            module.verify_and_publish_q30t_ptv2_full201_transfer(
                source_root=source_root, output=output, workers=2
            )
    finally:
        if child.poll() is None:
            child.kill()
            child.wait()

    assert not output.exists()


def test_absolute_directory_open_is_atomic_across_separate_process_ancestor_swap(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Resolution cannot continue through an ancestor detached by another process."""
    module = _load_module()
    if sys.platform != "linux":
        monkeypatch.setattr(module, "_ALLOW_NON_LINUX_OPENAT2_TEST_FALLBACK", True)
    ancestor = tmp_path / "approved-ancestor"
    leaf = ancestor / "nested/source"
    leaf.mkdir(parents=True)
    moved_ancestor = tmp_path / "detached-ancestor"
    replacement = tmp_path / "foreign-ancestor"
    replacement_leaf = replacement / "nested/source"
    replacement_leaf.mkdir(parents=True)
    request = tmp_path / "request-ancestor-swap"
    child = subprocess.Popen(
        [
            sys.executable,
            "-c",
            (
                "import pathlib,sys,time;"
                "ancestor,moved,replacement,request=map(pathlib.Path,sys.argv[1:]);"
                "deadline=time.monotonic()+5;"
                "\nwhile not request.exists():\n"
                " assert time.monotonic()<deadline\n"
                " time.sleep(0.01)\n"
                "ancestor.rename(moved);replacement.rename(ancestor)"
            ),
            str(ancestor),
            str(moved_ancestor),
            str(replacement),
            str(request),
        ]
    )
    swapped = False

    def swap_ancestor() -> None:
        nonlocal swapped
        if swapped:
            return
        request.touch()
        assert child.wait(timeout=5) == 0
        swapped = True

    real_component_open = module._open_directory_at

    def swap_after_component_open(
        parent_fd: int, name: str, display_path: Path
    ) -> tuple[int, os.stat_result]:
        result = real_component_open(parent_fd, name, display_path)
        if display_path == ancestor:
            swap_ancestor()
        return result

    monkeypatch.setattr(module, "_open_directory_at", swap_after_component_open)
    if hasattr(module, "_openat2_directory_from_root"):
        real_atomic_open = module._openat2_directory_from_root

        def swap_before_atomic_open(root_fd: int, relative_path: str) -> int:
            swap_ancestor()
            return real_atomic_open(root_fd, relative_path)

        monkeypatch.setattr(module, "_openat2_directory_from_root", swap_before_atomic_open)
    try:
        descriptor, resolved = module._open_absolute_directory(leaf)
        try:
            current = os.stat(leaf, follow_symlinks=False)
            assert module._identity(os.fstat(descriptor)) == module._identity(resolved)
            assert module._identity(resolved) == module._identity(current)
        finally:
            os.close(descriptor)
    finally:
        if child.poll() is None:
            child.kill()
            child.wait()


def _git(command: list[str], *, cwd: Path) -> str:
    result = subprocess.run(["git", *command], cwd=cwd, check=True, capture_output=True, text=True)
    return result.stdout.strip()


def _pushed_checkout(tmp_path: Path) -> tuple[Path, str]:
    remote = tmp_path / "remote.git"
    subprocess.run(["git", "init", "--bare", str(remote)], check=True, capture_output=True)
    repository = tmp_path / "repo"
    repository.mkdir()
    _git(["init", "-b", CANONICAL_BRANCH], cwd=repository)
    (repository / "tracked").write_text("source\n")
    verifier = repository / "tools/launcher/common/specdec/verify_q30t_ptv2_full201_transfer.py"
    verifier.parent.mkdir(parents=True)
    verifier.write_text(
        "# $Id$\n"
        "import os\n"
        "import pathlib\n"
        "import sys\n"
        "def _is_closed(fd):\n"
        "    try:\n"
        "        os.fstat(fd)\n"
        "    except OSError:\n"
        "        return True\n"
        "    return False\n"
        "calls = pathlib.Path(os.environ['PYTHON_CALLS'])\n"
        "ident = '$Id$'\n"
        "calls.write_text(\n"
        "    'EXECUTED=AUTHENTICATED\\n'\n"
        "    + f\"PYTHONPATH={os.environ.get('PYTHONPATH', '<unset>')}\\n\"\n"
        "    + f'IDENT_RAW={int(ident == chr(36) + \"Id\" + chr(36))}\\n'\n"
        "    + (\n"
        "        f\"FORBIDDEN_FD_CLOSED={int(_is_closed(int(os.environ['FORBIDDEN_FD'])))}\\n\"\n"
        "        if os.environ.get('FORBIDDEN_FD')\n"
        "        else ''\n"
        "    )\n"
        "    + '\\n'.join(sys.argv[1:])\n"
        "    + '\\n'\n"
        ")\n"
    )
    attributes = repository / ".gitattributes"
    attributes.write_text(f"{verifier.relative_to(repository)} ident\n")
    runner = (
        repository / "tools/launcher/common/specdec/run_q30t_ptv2_full201_transfer_verify.sbatch"
    )
    runner.write_text("#!/bin/bash\nprintf 'AUTHENTICATED RUNNER\\n'\n")
    _git(
        [
            "add",
            "tracked",
            ".gitattributes",
            str(verifier.relative_to(repository)),
            str(runner.relative_to(repository)),
        ],
        cwd=repository,
    )
    _git(
        [
            "-c",
            "user.name=Fixture",
            "-c",
            "user.email=fixture@example.com",
            "-c",
            "commit.gpgsign=false",
            "commit",
            "-m",
            "fixture",
        ],
        cwd=repository,
    )
    _git(["remote", "add", "gitlab", str(remote)], cwd=repository)
    _git(["push", "-u", "gitlab", CANONICAL_BRANCH], cwd=repository)
    _git(["config", "test.fetchUrl", str(remote)], cwd=repository)
    _git(["remote", "set-url", "gitlab", APPROVED_GITLAB_FETCH_URL], cwd=repository)
    return repository, _git(["rev-parse", "HEAD"], cwd=repository)


def _runner_environment(
    tmp_path: Path, repository: Path, source_commit: str
) -> tuple[dict[str, str], Path]:
    bin_root = tmp_path / "bin"
    bin_root.mkdir()
    calls = tmp_path / "python-calls"
    python = bin_root / "python3"
    python.write_text(
        "#!/bin/bash\n"
        "set -euo pipefail\n"
        'if [[ "${1:-}" == -I || ( "${1:-}" == -S && "${2:-}" == -I ) ]]; then\n'
        '  if [[ "${1:-}" == -I ]]; then stage="$4"; manifest="${5:-}"; '
        'else stage="$5"; manifest=""; fi\n'
        '  if [[ "${INJECT_MANIFEST_PATH_SWAP:-0}" == 1 && -f "$manifest" ]]; then\n'
        '    cp "$manifest" "$manifest.foreign"\n'
        '    rm "$manifest"\n'
        '    ln -s "$manifest.foreign" "$manifest"\n'
        '    touch "$MANIFEST_SWAP_MARKER"\n'
        "  fi\n"
        '  if [[ "${INJECT_STAGE_ENTRY:-}" == fifo ]]; then mkfifo "$stage/foreign-fifo"; fi\n'
        '  if [[ "${INJECT_STAGE_ENTRY:-}" == empty-dir ]]; then mkdir "$stage/foreign-empty"; fi\n'
        '  if [[ -e "$stage/../manifest-escaped" ]]; then touch "$ESCAPED_WRITE_MARKER"; fi\n'
        '  "$REAL_PYTHON" "$@"\n'
        "  status=$?\n"
        '  if [[ -e "$stage/../manifest-escaped" ]]; then touch "$ESCAPED_WRITE_MARKER"; fi\n'
        '  exit "$status"\n'
        "fi\n"
        "exit 99\n"
    )
    python.chmod(0o755)
    git = bin_root / "git"
    git.write_text(
        "#!/bin/bash\n"
        "set -euo pipefail\n"
        'printf "%s\\n" "$*" >> "$GIT_CALLS"\n'
        'args=("$@")\n'
        "is_fetch=0\n"
        "is_ls_tree=0\n"
        'for arg in "${args[@]}"; do [[ "$arg" == fetch ]] && is_fetch=1; done\n'
        'for arg in "${args[@]}"; do [[ "$arg" == ls-tree ]] && is_ls_tree=1; done\n'
        "if [[ $is_fetch == 1 ]]; then\n"
        '  for index in "${!args[@]}"; do\n'
        '    if [[ "${args[$index]}" == gitlab '
        '|| ( "${args[$index]}" == "$APPROVED_TEST_GIT_URL" '
        '&& "${DISABLE_APPROVED_URL_REWRITE:-0}" != 1 ) ]]; then\n'
        '      args[$index]="$TEST_GIT_FETCH_URL"\n'
        "    fi\n"
        "  done\n"
        '  if [[ "${INJECT_SCRATCH_REPLACEMENT:-0}" == 1 ]]; then\n'
        '    "$REAL_GIT" "${args[@]}"\n'
        '    for argument in "${args[@]}"; do\n'
        '      if [[ "$argument" == --git-dir=* ]]; then\n'
        '        repository_path="${argument#--git-dir=}"\n'
        '        scratch_path="${repository_path%/repository.git}"\n'
        '        mv -- "$scratch_path" "$scratch_path.owned"\n'
        '        mkdir -m 0700 -- "$scratch_path"\n'
        '        marker="$scratch_path/foreign-preserve"\n'
        '        printf "foreign\\n" > "$marker"\n'
        '        printf "%s\\n" "$marker" > "$SCRATCH_REPLACEMENT_POINTER"\n'
        "        exit 0\n"
        "      fi\n"
        "    done\n"
        "    exit 89\n"
        "  fi\n"
        "fi\n"
        "if [[ $is_ls_tree == 1 ]]; then\n"
        '  "$REAL_GIT" "${args[@]}"\n'
        '  if [[ "${INJECT_AUTH_TREE_RECORD:-}" == traversal ]]; then\n'
        '    printf "100644 blob %s\\t../manifest-escaped\\0" "$TEST_TREE_OBJECT_ID"\n'
        '  elif [[ "${INJECT_AUTH_TREE_RECORD:-}" == duplicate ]]; then\n'
        '    printf "100644 blob %s\\t%s\\0" "$TEST_TREE_OBJECT_ID" "$TEST_TREE_DUPLICATE_PATH"\n'
        "  fi\n"
        "  exit 0\n"
        "fi\n"
        'exec "$REAL_GIT" "${args[@]}"\n'
    )
    git.chmod(0o755)
    ssh = bin_root / "ssh"
    ssh.write_text("#!/bin/sh\nexit 97\n")
    ssh.chmod(0o755)
    real_git = shutil.which("git")
    if real_git is None:
        raise RuntimeError("git is required for transfer verifier runner tests")
    real_python = shutil.which("python3")
    if real_python is None:
        raise RuntimeError("python3 is required for transfer verifier runner tests")
    environment = os.environ.copy()
    approved_output = (
        "/lustre/fsw/coreai_dlalgo_llm/users/sna/modelopt-qwen3-drafter-training/receipts/"
        "q30t-ptv23-complement-700k-v1/ptv2-full201-transfer/VERIFY.json"
    )
    environment.update(
        {
            "PATH": f"{bin_root}{os.pathsep}{environment['PATH']}",
            "PYTHON_CALLS": str(calls),
            "GIT_CALLS": str(tmp_path / "git-calls"),
            "REAL_GIT": real_git,
            "REAL_PYTHON": real_python,
            "TEST_GIT_FETCH_URL": _git(["config", "--get", "test.fetchUrl"], cwd=repository),
            "APPROVED_TEST_GIT_URL": APPROVED_GITLAB_FETCH_URL,
            "TEST_TREE_OBJECT_ID": _git(["rev-parse", "HEAD:tracked"], cwd=repository),
            "TEST_TREE_DUPLICATE_PATH": (
                "tools/launcher/common/specdec/verify_q30t_ptv2_full201_transfer.py"
            ),
            "ESCAPED_WRITE_MARKER": str(tmp_path / "escaped-write-observed"),
            "MANIFEST_SWAP_MARKER": str(tmp_path / "manifest-path-swapped"),
            "PYTHONPATH": "/poisoned/inherited/path",
            "SLURM_JOB_ID": "4242",
            "SLURM_NNODES": "1",
            "SLURM_CPUS_PER_TASK": "32",
            "SOURCE_PATH": str(repository),
            "SOURCE_SHA": source_commit,
            "SOURCE_REMOTE_REF": CANONICAL_REMOTE_REF,
            "OUTPUT_PATH": approved_output,
        }
    )
    slurm_tmp = tmp_path / "slurm-tmp"
    slurm_tmp.mkdir()
    environment["SLURM_TMPDIR"] = str(slurm_tmp)
    environment.pop("SLURM_JOB_GPUS", None)
    environment.pop("SLURM_GPUS_ON_NODE", None)
    if sys.platform != "linux":
        environment["Q30T_TEST_ALLOW_UNSEALED_PLATFORM"] = sys.platform
    return environment, calls


def _runner_command(environment: Mapping[str, str]) -> list[str]:
    return [BASH, str(RUNNER), environment["SOURCE_SHA"], environment["OUTPUT_PATH"]]


def _submitter_environment(tmp_path: Path, repository: Path) -> tuple[dict[str, str], Path, Path]:
    bin_root = tmp_path / "submit-bin"
    bin_root.mkdir()
    sbatch_calls = tmp_path / "sbatch-calls"
    spooled_runner = tmp_path / "spooled-runner"
    runner_evidence = tmp_path / "sbatch-runner-evidence.jsonl"
    git = bin_root / "git"
    git.write_text(
        "#!/bin/bash\n"
        "set -euo pipefail\n"
        'args=("$@")\n'
        'for index in "${!args[@]}"; do\n'
        '  if [[ "${args[$index]}" == "$APPROVED_TEST_GIT_URL" ]]; then\n'
        '    args[$index]="$TEST_GIT_FETCH_URL"\n'
        "  fi\n"
        "done\n"
        "is_fetch=0\n"
        'for argument in "${args[@]}"; do [[ "$argument" == fetch ]] && is_fetch=1; done\n'
        'if [[ $is_fetch == 1 && "${INJECT_SCRATCH_REPLACEMENT:-0}" == 1 ]]; then\n'
        '  "$REAL_GIT" "${args[@]}"\n'
        '  for argument in "${args[@]}"; do\n'
        '    if [[ "$argument" == --git-dir=* ]]; then\n'
        '      repository_path="${argument#--git-dir=}"\n'
        '      scratch_path="${repository_path%/repository.git}"\n'
        '      mv -- "$scratch_path" "$scratch_path.owned"\n'
        '      mkdir -m 0700 -- "$scratch_path"\n'
        '      marker="$scratch_path/foreign-preserve"\n'
        '      printf "foreign\\n" > "$marker"\n'
        '      printf "%s\\n" "$marker" > "$SCRATCH_REPLACEMENT_POINTER"\n'
        "      exit 0\n"
        "    fi\n"
        "  done\n"
        "  exit 89\n"
        "fi\n"
        'exec "$REAL_GIT" "${args[@]}"\n'
    )
    git.chmod(0o755)
    sbatch = bin_root / "sbatch"
    sbatch.write_text(
        "#!/bin/bash\n"
        "set -euo pipefail\n"
        'printf "%s\\0" "$@" >> "$SBATCH_CALLS"\n'
        'printf "\\0" >> "$SBATCH_CALLS"\n'
        'for argument in "$@"; do\n'
        '  if [[ ( "$argument" == *.sbatch || "$argument" == /proc/self/fd/* '
        '|| "$argument" == /dev/fd/* ) && -f "$argument" ]]; then\n'
        '    cp -f -- "$argument" "$SPOOLED_RUNNER"\n'
        '    "$REAL_PYTHON" - "$argument" "$SBATCH_RUNNER_EVIDENCE" <<\'PY\'\n'
        "import fcntl, hashlib, json, os, pathlib, sys\n"
        "path = pathlib.Path(sys.argv[1])\n"
        "descriptor = os.open(path, os.O_RDONLY)\n"
        "try:\n"
        "    payload = os.pread(descriptor, 16 * 1024 * 1024, 0)\n"
        "    inode = os.fstat(descriptor).st_ino\n"
        "    seals = fcntl.fcntl(descriptor, fcntl.F_GET_SEALS) if sys.platform == 'linux' else None\n"
        "finally:\n"
        "    os.close(descriptor)\n"
        "record = {'inode': inode, 'path': str(path), 'seals': seals, "
        "'sha256': hashlib.sha256(payload).hexdigest()}\n"
        "with pathlib.Path(sys.argv[2]).open('a') as stream:\n"
        "    stream.write(json.dumps(record, sort_keys=True) + '\\n')\n"
        "PY\n"
        "  fi\n"
        "done\n"
        'if [[ " $* " == *" --parsable "* ]]; then printf "778899\\n"; fi\n'
    )
    sbatch.chmod(0o755)
    real_git = shutil.which("git")
    if real_git is None:
        raise RuntimeError("git is required for submitter tests")
    environment = os.environ | {
        "PATH": f"{bin_root}{os.pathsep}{os.environ['PATH']}",
        "REAL_GIT": real_git,
        "TEST_GIT_FETCH_URL": _git(["config", "--get", "test.fetchUrl"], cwd=repository),
        "APPROVED_TEST_GIT_URL": APPROVED_GITLAB_FETCH_URL,
        "SBATCH_CALLS": str(sbatch_calls),
        "SPOOLED_RUNNER": str(spooled_runner),
        "SBATCH_RUNNER_EVIDENCE": str(runner_evidence),
        "REAL_PYTHON": sys.executable,
        "FORBIDDEN_SUBMISSION_VALUE": "must-not-leak",
    }
    if sys.platform != "linux":
        environment["Q30T_TEST_ALLOW_UNSEALED_PLATFORM"] = sys.platform
    return environment, sbatch_calls, spooled_runner


def test_trusted_submitter_spools_the_exact_fetched_runner_with_no_exported_environment(
    tmp_path: Path,
) -> None:
    """Only canonical fetched runner bytes can cross the sbatch trust boundary."""
    assert SUBMITTER.exists(), "trusted transfer verifier submitter is not implemented"
    repository, source_commit = _pushed_checkout(tmp_path)
    environment, sbatch_calls, spooled_runner = _submitter_environment(tmp_path, repository)
    runner_path = "tools/launcher/common/specdec/run_q30t_ptv2_full201_transfer_verify.sbatch"
    _git(["update-index", "--skip-worktree", runner_path], cwd=repository)
    (repository / runner_path).write_text("#!/bin/bash\nprintf 'FOREIGN RUNNER\\n'\n")
    assert _git(["status", "--porcelain"], cwd=repository) == ""
    approved_output = environment.get(
        "OUTPUT_PATH",
        "/lustre/fsw/coreai_dlalgo_llm/users/sna/modelopt-qwen3-drafter-training/receipts/"
        "q30t-ptv23-complement-700k-v1/ptv2-full201-transfer/VERIFY.json",
    )

    result = subprocess.run(
        [
            BASH,
            str(SUBMITTER),
            "--test-only",
            "--source-path",
            str(repository),
            "--output",
            approved_output,
            "--slurm-output",
            str(tmp_path / "slurm-%j.out"),
        ],
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    expected_runner = subprocess.run(
        [environment["REAL_GIT"], "show", f"{source_commit}:{runner_path}"],
        cwd=repository,
        check=True,
        capture_output=True,
    ).stdout
    assert spooled_runner.read_bytes() == expected_runner
    calls = sbatch_calls.read_bytes().split(b"\0\0")
    assert len(calls) == 2 and calls[1] == b""
    arguments = calls[0].split(b"\0")
    assert b"--test-only" in arguments
    assert b"--export=NONE" in arguments
    assert source_commit.encode() in arguments
    assert b"must-not-leak" not in calls[0]


def test_trusted_submitter_test_only_precedes_the_real_submission(tmp_path: Path) -> None:
    """The same authenticated runner and exact arguments pass test-only before submission."""
    repository, source_commit = _pushed_checkout(tmp_path)
    environment, sbatch_calls, _ = _submitter_environment(tmp_path, repository)

    result = subprocess.run(
        [
            BASH,
            str(SUBMITTER),
            "--submit",
            "--source-path",
            str(repository),
            "--output",
            "/lustre/fsw/coreai_dlalgo_llm/users/sna/modelopt-qwen3-drafter-training/receipts/"
            "q30t-ptv23-complement-700k-v1/ptv2-full201-transfer/VERIFY.json",
            "--slurm-output",
            str(tmp_path / "slurm-%j.out"),
        ],
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "778899"
    records = sbatch_calls.read_bytes().split(b"\0\0")
    assert len(records) == 3 and records[2] == b""
    test_only = records[0].split(b"\0")
    submitted = records[1].split(b"\0")
    assert b"--test-only" in test_only and b"--parsable" not in test_only
    assert b"--parsable" in submitted and b"--test-only" not in submitted
    for arguments in (test_only, submitted):
        assert b"--export=NONE" in arguments
        assert b"--job-name=q30t-ptv2-full201-transfer-verify" in arguments
        assert source_commit.encode() in arguments
    evidence = [
        json.loads(line)
        for line in Path(environment["SBATCH_RUNNER_EVIDENCE"]).read_text().splitlines()
    ]
    assert len(evidence) == 2
    expected_prefix = "/proc/self/fd/" if sys.platform == "linux" else "/dev/fd/"
    assert all(record["path"].startswith(expected_prefix) for record in evidence)
    assert evidence[0]["inode"] != evidence[1]["inode"]
    expected_runner = subprocess.run(
        [
            environment["REAL_GIT"],
            "show",
            f"{source_commit}:tools/launcher/common/specdec/"
            "run_q30t_ptv2_full201_transfer_verify.sbatch",
        ],
        cwd=repository,
        check=True,
        capture_output=True,
    ).stdout
    assert {record["sha256"] for record in evidence} == {
        hashlib.sha256(expected_runner).hexdigest()
    }
    if sys.platform == "linux":
        required_seals = (
            fcntl.F_SEAL_WRITE | fcntl.F_SEAL_GROW | fcntl.F_SEAL_SHRINK | fcntl.F_SEAL_SEAL
        )
        assert all(record["seals"] & required_seals == required_seals for record in evidence)


def test_submitter_exit_preserves_a_foreign_scratch_namespace_replacement(
    tmp_path: Path,
) -> None:
    """Exit handling cannot recursively delete a same-UID replacement scratch tree."""
    repository, _ = _pushed_checkout(tmp_path)
    environment, sbatch_calls, _ = _submitter_environment(tmp_path, repository)
    pointer = tmp_path / "submitter-scratch-replacement-path"
    environment.update(
        {
            "INJECT_SCRATCH_REPLACEMENT": "1",
            "SCRATCH_REPLACEMENT_POINTER": str(pointer),
        }
    )

    result = subprocess.run(
        [
            BASH,
            str(SUBMITTER),
            "--test-only",
            "--source-path",
            str(repository),
            "--output",
            "/lustre/fsw/coreai_dlalgo_llm/users/sna/modelopt-qwen3-drafter-training/receipts/"
            "q30t-ptv23-complement-700k-v1/ptv2-full201-transfer/VERIFY.json",
            "--slurm-output",
            str(tmp_path / "slurm-%j.out"),
        ],
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode != 0
    marker = Path(pointer.read_text().strip())
    assert marker.read_text() == "foreign\n"
    assert not sbatch_calls.exists()


def test_runner_uses_one_cpu_node_and_the_exact_production_source(
    tmp_path: Path,
) -> None:
    """A clean pushed checkout invokes the verifier without inheriting a GPU or Python path."""
    repository, source_commit = _pushed_checkout(tmp_path)
    environment, calls = _runner_environment(tmp_path, repository, source_commit)

    result = subprocess.run(
        _runner_command(environment), env=environment, check=False, capture_output=True, text=True
    )

    assert result.returncode == 0, result.stderr
    approved_root = (
        "/lustre/fsw/coreai_dlalgo_llm/users/sna/modelopt-qwen3-drafter-training/"
        "assets/q30t-ptv2-full201-source-v1"
    )
    call_lines = calls.read_text().splitlines()
    assert call_lines == [
        "EXECUTED=AUTHENTICATED",
        "PYTHONPATH=<unset>",
        "IDENT_RAW=1",
        "--source-root",
        approved_root,
        "--output",
        environment["OUTPUT_PATH"],
        "--workers",
        "16",
    ]


@pytest.mark.skipif(
    sys.platform == "linux",
    reason="Linux production fallback is fixed at /raid/scratch and requires a target-node probe",
)
def test_runner_uses_a_job_bound_node_local_fallback_without_slurm_tmpdir(
    tmp_path: Path,
) -> None:
    """Ptyche jobs without SLURM_TMPDIR use an exclusive job-bound node-local directory."""
    repository, source_commit = _pushed_checkout(tmp_path)
    environment, calls = _runner_environment(tmp_path, repository, source_commit)
    environment.pop("SLURM_TMPDIR")
    node_local_root = tmp_path / "raid-scratch"
    user_name = subprocess.run(
        ["id", "-un"], check=True, capture_output=True, text=True
    ).stdout.strip()
    user_root = node_local_root / user_name
    user_root.mkdir(parents=True, mode=0o700)
    environment.update(
        {
            "Q30T_TEST_ALLOW_NODE_LOCAL_SCRATCH_ROOT": "1",
            "Q30T_TEST_NODE_LOCAL_SCRATCH_ROOT": str(node_local_root),
        }
    )

    result = subprocess.run(
        _runner_command(environment), env=environment, check=False, capture_output=True, text=True
    )

    assert result.returncode == 0, result.stderr
    assert calls.exists()
    scratch_directories = list(user_root.glob("q30t-transfer-verifier.4242.*"))
    assert len(scratch_directories) == 1
    scratch = scratch_directories[0]
    assert scratch.is_dir() and not scratch.is_symlink()
    assert stat.S_IMODE(scratch.stat().st_mode) == 0o700
    git_calls = Path(environment["GIT_CALLS"]).read_text()
    assert f"--git-dir={scratch}/repository.git" in git_calls


def test_runner_exit_preserves_a_foreign_scratch_namespace_replacement(
    tmp_path: Path,
) -> None:
    """Job exit cannot recursively delete a replacement of scheduler-owned scratch."""
    repository, source_commit = _pushed_checkout(tmp_path)
    environment, calls = _runner_environment(tmp_path, repository, source_commit)
    pointer = tmp_path / "runner-scratch-replacement-path"
    environment.update(
        {
            "INJECT_SCRATCH_REPLACEMENT": "1",
            "SCRATCH_REPLACEMENT_POINTER": str(pointer),
        }
    )

    result = subprocess.run(
        _runner_command(environment), env=environment, check=False, capture_output=True, text=True
    )

    assert result.returncode != 0
    marker = Path(pointer.read_text().strip())
    assert marker.read_text() == "foreign\n"
    assert not calls.exists()


def test_runner_fetches_and_executes_only_the_standalone_verifier_blob(
    tmp_path: Path,
) -> None:
    """The job never inventories or stages unrelated paths from the fetched commit."""
    repository, source_commit = _pushed_checkout(tmp_path)
    environment, calls = _runner_environment(tmp_path, repository, source_commit)

    result = subprocess.run(
        [BASH, str(RUNNER), source_commit, environment["OUTPUT_PATH"]],
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert calls.read_text().splitlines()[0] == "EXECUTED=AUTHENTICATED"
    git_calls = Path(environment["GIT_CALLS"]).read_text()
    verifier_path = "tools/launcher/common/specdec/verify_q30t_ptv2_full201_transfer.py"
    assert "ls-tree -rz -r --full-tree" not in git_calls
    assert "fetch --filter=blob:none" in git_calls
    assert verifier_path in git_calls
    assert git_calls.count("cat-file blob ") == 1


def test_runner_executes_fetched_bytes_when_skip_worktree_hides_a_modification(
    tmp_path: Path,
) -> None:
    """Mutable skip-worktree bytes cannot become the verifier's executed source tree."""
    repository, source_commit = _pushed_checkout(tmp_path)
    environment, calls = _runner_environment(tmp_path, repository, source_commit)
    verifier_path = "tools/launcher/common/specdec/verify_q30t_ptv2_full201_transfer.py"
    _git(["update-index", "--skip-worktree", "tracked", verifier_path], cwd=repository)
    (repository / "tracked").write_text("foreign mutable bytes\n")
    (repository / verifier_path).write_text(
        "import os, pathlib\n"
        "pathlib.Path(os.environ['PYTHON_CALLS']).write_text('EXECUTED=FOREIGN\\n')\n"
    )
    assert _git(["status", "--porcelain"], cwd=repository) == ""

    result = subprocess.run(
        _runner_command(environment), env=environment, check=False, capture_output=True, text=True
    )

    assert result.returncode == 0, result.stderr
    call_lines = calls.read_text().splitlines()
    assert call_lines[0] == "EXECUTED=AUTHENTICATED"
    assert call_lines[1:3] == ["PYTHONPATH=<unset>", "IDENT_RAW=1"]


def test_runner_seals_the_authenticated_verifier_before_execution(tmp_path: Path) -> None:
    """The final verifier FD rejects in-place and reopened writes before it executes."""
    repository, source_commit = _pushed_checkout(tmp_path)
    environment, calls = _runner_environment(tmp_path, repository, source_commit)

    result = subprocess.run(
        _runner_command(environment), env=environment, check=False, capture_output=True, text=True
    )

    assert result.returncode == 0, result.stderr
    assert calls.read_text().splitlines()[0] == "EXECUTED=AUTHENTICATED"
    expected_audit = (
        "sealed-verifier-write-check: inplace=EPERM reopened=EPERM"
        if sys.platform == "linux"
        else f"sealed-verifier-write-check: test-fallback={sys.platform}"
    )
    assert expected_audit in result.stderr


def test_runner_closes_every_inherited_descriptor_except_the_sealed_verifier(
    tmp_path: Path,
) -> None:
    """An unrelated inherited descriptor cannot survive final verifier exec."""
    repository, source_commit = _pushed_checkout(tmp_path)
    environment, calls = _runner_environment(tmp_path, repository, source_commit)
    pipe_reader, write_descriptor = os.pipe()
    read_descriptor = fcntl.fcntl(pipe_reader, fcntl.F_DUPFD, 10)
    os.close(pipe_reader)
    environment["FORBIDDEN_FD"] = str(read_descriptor)
    try:
        result = subprocess.run(
            _runner_command(environment),
            env=environment,
            check=False,
            capture_output=True,
            text=True,
            pass_fds=(read_descriptor,),
        )
    finally:
        os.close(read_descriptor)
        os.close(write_descriptor)

    assert result.returncode == 0, result.stderr
    assert "FORBIDDEN_FD_CLOSED=1" in calls.read_text().splitlines()


def test_runner_rejects_unsafe_or_duplicate_manifest_paths_before_materialization(
    tmp_path: Path,
) -> None:
    """Traversal and duplicate records cannot write or replace staged paths."""
    repository, source_commit = _pushed_checkout(tmp_path)
    for fault in ("traversal", "duplicate"):
        run_root = tmp_path / fault
        run_root.mkdir()
        environment, calls = _runner_environment(run_root, repository, source_commit)
        environment["INJECT_AUTH_TREE_RECORD"] = fault

        result = subprocess.run(
            _runner_command(environment),
            env=environment,
            check=False,
            capture_output=True,
            text=True,
        )

        assert result.returncode == 2
        assert not calls.exists()
        assert not Path(environment["ESCAPED_WRITE_MARKER"]).exists()


def test_runner_never_invokes_a_worktree_fsmonitor_before_remote_authentication(
    tmp_path: Path,
) -> None:
    """Mutable core.fsmonitor configuration cannot execute during source authentication."""
    repository, source_commit = _pushed_checkout(tmp_path)
    environment, _ = _runner_environment(tmp_path, repository, source_commit)
    marker = tmp_path / "fsmonitor-executed"
    monitor = tmp_path / "fsmonitor.sh"
    monitor.write_text(f"#!/bin/sh\ntouch {marker!s}\nprintf '\\n'\n")
    monitor.chmod(0o755)
    _git(["config", "core.fsmonitor", str(monitor)], cwd=repository)

    result = subprocess.run(
        _runner_command(environment), env=environment, check=False, capture_output=True, text=True
    )

    assert result.returncode == 0, result.stderr
    assert not marker.exists()


def test_runner_never_reopens_the_authenticated_tree_manifest(tmp_path: Path) -> None:
    """Staging consumes immutable parsed records rather than a replaceable manifest pathname."""
    repository, source_commit = _pushed_checkout(tmp_path)
    environment, calls = _runner_environment(tmp_path, repository, source_commit)
    environment["INJECT_MANIFEST_PATH_SWAP"] = "1"

    result = subprocess.run(
        _runner_command(environment), env=environment, check=False, capture_output=True, text=True
    )

    assert result.returncode == 0, result.stderr
    assert calls.read_text().splitlines()[0] == "EXECUTED=AUTHENTICATED"
    assert not Path(environment["MANIFEST_SWAP_MARKER"]).exists()


def test_runner_disables_global_sitecustomize_for_inventory_and_execution(tmp_path: Path) -> None:
    """Neither trusted inventory nor verifier execution imports global site packages."""
    repository, source_commit = _pushed_checkout(tmp_path)
    environment, calls = _runner_environment(tmp_path, repository, source_commit)
    poisoned_python = tmp_path / "poisoned-python"
    subprocess.run(
        [sys.executable, "-m", "venv", "--without-pip", str(poisoned_python)], check=True
    )
    python_version = f"python{sys.version_info.major}.{sys.version_info.minor}"
    site_packages = poisoned_python / "lib" / python_version / "site-packages"
    attack_marker = tmp_path / "global-sitecustomize-executed"
    (site_packages / "sitecustomize.py").write_text(
        "import os, pathlib\npathlib.Path(os.environ['GLOBAL_SITECUSTOMIZE_MARKER']).touch()\n"
    )
    environment.update(
        {
            "GLOBAL_SITECUSTOMIZE_MARKER": str(attack_marker),
            "REAL_PYTHON": str(poisoned_python / "bin/python"),
        }
    )

    result = subprocess.run(
        _runner_command(environment), env=environment, check=False, capture_output=True, text=True
    )

    assert result.returncode == 0, result.stderr
    assert calls.read_text().splitlines()[0] == "EXECUTED=AUTHENTICATED"
    assert not attack_marker.exists()


def test_runner_does_not_inventory_or_stage_unrelated_authenticated_tree_paths(
    tmp_path: Path,
) -> None:
    """Even unusual unrelated Git paths are outside the standalone verifier stage."""
    repository, _ = _pushed_checkout(tmp_path)
    unsafe_path = repository / "unsafe\t\x1b\x7f"
    unsafe_path.write_text("unsafe\n")
    _git(["add", "--", unsafe_path.name], cwd=repository)
    _git(
        [
            "-c",
            "user.name=Fixture",
            "-c",
            "user.email=fixture@example.com",
            "-c",
            "commit.gpgsign=false",
            "commit",
            "-m",
            "unsafe-control-path",
        ],
        cwd=repository,
    )
    source_commit = _git(["rev-parse", "HEAD"], cwd=repository)
    fixture_remote = _git(["config", "--get", "test.fetchUrl"], cwd=repository)
    _git(["push", fixture_remote, CANONICAL_BRANCH], cwd=repository)
    environment, calls = _runner_environment(tmp_path, repository, source_commit)

    result = subprocess.run(
        _runner_command(environment), env=environment, check=False, capture_output=True, text=True
    )

    assert result.returncode == 0, result.stderr
    assert calls.read_text().splitlines()[0] == "EXECUTED=AUTHENTICATED"


def test_runner_rejects_gpu_or_unpushed_execution(tmp_path: Path) -> None:
    """Durable evidence cannot come from a GPU allocation or a local-only source commit."""
    repository, source_commit = _pushed_checkout(tmp_path)
    environment, calls = _runner_environment(tmp_path, repository, source_commit)
    environment["SLURM_JOB_GPUS"] = "0"

    gpu_result = subprocess.run(
        _runner_command(environment), env=environment, check=False, capture_output=True, text=True
    )

    assert gpu_result.returncode == 2
    assert "CPU-only" in gpu_result.stderr
    assert not calls.exists()

    environment.pop("SLURM_JOB_GPUS")
    (repository / "tracked").write_text("local-only\n")
    _git(["add", "tracked"], cwd=repository)
    _git(
        [
            "-c",
            "user.name=Fixture",
            "-c",
            "user.email=fixture@example.com",
            "-c",
            "commit.gpgsign=false",
            "commit",
            "-m",
            "local-only",
        ],
        cwd=repository,
    )
    environment["SOURCE_SHA"] = _git(["rev-parse", "HEAD"], cwd=repository)

    local_result = subprocess.run(
        _runner_command(environment), env=environment, check=False, capture_output=True, text=True
    )

    assert local_result.returncode == 2
    assert "freshly fetched" in local_result.stderr
    assert not calls.exists()


def test_runner_rejects_an_unpushed_head_even_if_the_local_tracking_ref_is_forged(
    tmp_path: Path,
) -> None:
    """Only a fresh fetch, not a mutable refs/remotes pathname, authenticates source HEAD."""
    repository, source_commit = _pushed_checkout(tmp_path)
    environment, calls = _runner_environment(tmp_path, repository, source_commit)
    (repository / "tracked").write_text("unpushed but forged\n")
    _git(["add", "tracked"], cwd=repository)
    _git(
        [
            "-c",
            "user.name=Fixture",
            "-c",
            "user.email=fixture@example.com",
            "-c",
            "commit.gpgsign=false",
            "commit",
            "-m",
            "unpushed-forged",
        ],
        cwd=repository,
    )
    forged_head = _git(["rev-parse", "HEAD"], cwd=repository)
    _git(
        ["update-ref", f"refs/remotes/{CANONICAL_REMOTE_REF}", forged_head],
        cwd=repository,
    )
    environment["SOURCE_SHA"] = forged_head

    result = subprocess.run(
        _runner_command(environment), env=environment, check=False, capture_output=True, text=True
    )

    assert result.returncode == 2
    assert "freshly fetched" in result.stderr
    assert not calls.exists()


def test_trusted_submitter_rejects_a_forged_gitlab_remote_url_even_when_it_contains_head(
    tmp_path: Path,
) -> None:
    """The submit trust boundary rejects an attacker-controlled canonical remote name."""
    repository, _ = _pushed_checkout(tmp_path)
    environment, sbatch_calls, _ = _submitter_environment(tmp_path, repository)
    forged_url = environment["TEST_GIT_FETCH_URL"]
    _git(["remote", "set-url", "gitlab", forged_url], cwd=repository)

    result = subprocess.run(
        [
            BASH,
            str(SUBMITTER),
            "--test-only",
            "--source-path",
            str(repository),
            "--output",
            "/lustre/fsw/coreai_dlalgo_llm/users/sna/modelopt-qwen3-drafter-training/receipts/"
            "q30t-ptv23-complement-700k-v1/ptv2-full201-transfer/VERIFY.json",
            "--slurm-output",
            str(tmp_path / "slurm-%j.out"),
        ],
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 2
    assert "approved GitLab fetch URL" in result.stderr
    assert not sbatch_calls.exists()


def test_trusted_submitter_rejects_a_worktree_transport_override(tmp_path: Path) -> None:
    """A worktree URL rewrite cannot redirect the trusted canonical fetch."""
    repository, _ = _pushed_checkout(tmp_path)
    environment, sbatch_calls, _ = _submitter_environment(tmp_path, repository)
    forged_url = Path(environment["TEST_GIT_FETCH_URL"]).as_uri()
    _git(["config", "extensions.worktreeConfig", "true"], cwd=repository)
    _git(
        [
            "config",
            "--worktree",
            f"url.{forged_url}.insteadOf",
            APPROVED_GITLAB_FETCH_URL,
        ],
        cwd=repository,
    )
    result = subprocess.run(
        [
            BASH,
            str(SUBMITTER),
            "--test-only",
            "--source-path",
            str(repository),
            "--output",
            "/lustre/fsw/coreai_dlalgo_llm/users/sna/modelopt-qwen3-drafter-training/receipts/"
            "q30t-ptv23-complement-700k-v1/ptv2-full201-transfer/VERIFY.json",
            "--slurm-output",
            str(tmp_path / "slurm-%j.out"),
        ],
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 2
    assert "disallowed Git transport override" in result.stderr
    assert not sbatch_calls.exists()


def test_runner_ignores_a_repository_selected_by_inherited_git_environment(
    tmp_path: Path,
) -> None:
    """Inherited repository selectors cannot replace the sterile fetched verifier blob."""
    repository, source_commit = _pushed_checkout(tmp_path)
    environment, calls = _runner_environment(tmp_path, repository, source_commit)
    foreign_source = tmp_path / "foreign-source"
    foreign_source.mkdir()
    environment.update(
        {
            "SOURCE_PATH": str(foreign_source),
            "GIT_DIR": str(repository / ".git"),
            "GIT_WORK_TREE": str(repository),
        }
    )

    result = subprocess.run(
        _runner_command(environment), env=environment, check=False, capture_output=True, text=True
    )

    assert result.returncode == 0, result.stderr
    assert calls.read_text().splitlines()[0] == "EXECUTED=AUTHENTICATED"


def test_runner_rejects_a_noncanonical_output_after_the_pushed_source_gate(
    tmp_path: Path,
) -> None:
    """The runner cannot create or publish outside the reviewed receipt component."""
    repository, source_commit = _pushed_checkout(tmp_path)
    environment, calls = _runner_environment(tmp_path, repository, source_commit)
    environment["OUTPUT_PATH"] = str(tmp_path / "foreign/VERIFY.json")

    result = subprocess.run(
        _runner_command(environment), env=environment, check=False, capture_output=True, text=True
    )

    assert result.returncode == 2
    assert "approved verification receipt" in result.stderr
    assert not calls.exists()
