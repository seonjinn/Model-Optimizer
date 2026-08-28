# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Validate the immutable CPU-only Q30 row-schema launcher boundary."""

from __future__ import annotations

import hashlib
import importlib.util
import inspect
import os
import re
import shutil
import stat
import subprocess
import sys
from pathlib import Path
from typing import TYPE_CHECKING

import pyarrow  # pyright: ignore[reportMissingImports]
import pytest

if TYPE_CHECKING:
    from types import ModuleType

ROOT = Path(__file__).resolve().parents[2]
RUNNER = ROOT / "launcher/common/specdec/run_q30t_row_schema_observation.sbatch"
SUBMITTER = ROOT / "launcher/common/specdec/submit_q30t_row_schema_observation.sh"
OBSERVER = ROOT.parent / "examples/dataset/observe_q30t_ptv23_row_schemas.py"
ATOMIC = ROOT / "launcher/common/specdec/qwen4b_b_atomic.py"


def _runner_embedded_python(marker: str) -> str:
    runner = RUNNER.read_text()
    return runner.split(f"# {marker}\n", 1)[1].split("\nPY\n", 1)[0]


def _scratch_allocator_test_source(scratch_root: Path) -> str:
    runner = RUNNER.read_text()
    marker = "# Q30T_SCRATCH_ALLOCATOR\n"
    assert marker in runner, "runner lacks the bounded scratch allocator"
    source = runner.split(marker, 1)[1].split("\nPY\n", 1)[0]
    source = source.replace(
        'scratch_root_path = "/raid/scratch"',
        f"scratch_root_path = {str(scratch_root.absolute())!r}",
    )
    source = source.replace("required_root_uids = {0}", "required_root_uids = {0, os.getuid()}")
    source = source.replace("metadata.st_gid != 0", "metadata.st_gid != os.getgid()")
    source = source.replace("or mode != 0o1777:", "or mode not in {0o700, 0o1777}:")
    return source


def _run_root_component_policy(
    *,
    mode: int,
    owner: int,
    group: int,
    effective_groups: set[int],
    shared_root: bool = False,
) -> subprocess.CompletedProcess[str]:
    allocator = _runner_embedded_python("Q30T_SCRATCH_ALLOCATOR")
    primitives = allocator.rsplit("\nmain()", 1)[0]
    exercise = (
        primitives
        + f"\neffective_group_ids = {effective_groups!r}\n"
        + f"metadata = os.stat_result(({stat.S_IFDIR | mode}, 1, 1, 2, {owner}, {group}, 0, 0, 0, 0))\n"
        + f"require_root_component(metadata, 'scratch root ancestor', shared_root={shared_root!r})\n"
    )
    return subprocess.run(
        [sys.executable, "-I", "-S", "-"],
        input=exercise,
        text=True,
        capture_output=True,
        check=False,
    )


def _run_current_materialization_segment(
    tmp_path: Path, scratch_root: Path
) -> subprocess.CompletedProcess[str]:
    materializer = _runner_embedded_python("Q30T_BLOB_MATERIALIZER")
    fake_git = tmp_path / "git"
    fake_git.write_text(
        "#!/bin/sh\n"
        "for last do :; done\n"
        'case "$last" in\n'
        "  *\\^\\{commit\\}) exit 0 ;;\n"
        "  *observe_q30t*) printf observer-bytes ;;\n"
        "  *stage_subset*) printf plan-bytes ;;\n"
        "  *qwen4b_b_atomic*) printf atomic-bytes ;;\n"
        "esac\n"
    )
    fake_git.chmod(0o700)
    source = tmp_path / "source"
    source.mkdir()
    descriptor = os.open(scratch_root, os.O_RDONLY)
    try:
        return subprocess.run(
            [
                sys.executable,
                "-I",
                "-S",
                "-",
                str(descriptor),
                str(fake_git),
                str(source),
                "a" * 40,
                "examples/dataset/observe_q30t_ptv23_row_schemas.py",
                "examples/dataset/qwen3_30ba3b_thinking_ptv3_stage_subset_v1.json",
                "tools/launcher/common/specdec/qwen4b_b_atomic.py",
            ],
            input=materializer,
            text=True,
            capture_output=True,
            pass_fds=(descriptor,),
            check=False,
        )
    finally:
        os.close(descriptor)


def _run_current_observer_adoption(
    scratch_root: Path, materialized_record: str
) -> tuple[subprocess.CompletedProcess[str], bool]:
    bootstrap = _runner_embedded_python("Q30T_IMMUTABLE_BOOTSTRAP")
    primitives = bootstrap.split("# Q30T_BOOTSTRAP_ENTRY", 1)[0]
    observer_device, observer_inode, observer_size, observer_sha256 = (
        materialized_record.strip().split("\t")[:4]
    )
    exercise = (
        primitives
        + "\nroot_fd = int(sys.argv[1])\n"
        + "descriptor, _ = adopt_materialized_blob(root_fd, *sys.argv[2:7], 'observer')\n"
        + "os.close(descriptor)\n"
    )
    root_descriptor = os.open(scratch_root, os.O_RDONLY)
    process = subprocess.Popen(
        [
            sys.executable,
            "-I",
            "-S",
            "-",
            str(root_descriptor),
            "examples/dataset/observe_q30t_ptv23_row_schemas.py",
            observer_device,
            observer_inode,
            observer_size,
            observer_sha256,
        ],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        pass_fds=(root_descriptor,),
    )
    try:
        stdout, stderr = process.communicate(exercise, timeout=0.5)
        timed_out = False
    except subprocess.TimeoutExpired:
        timed_out = True
        process.kill()
        stdout, stderr = process.communicate()
    finally:
        os.close(root_descriptor)
    return subprocess.CompletedProcess(process.args, process.returncode, stdout, stderr), timed_out


def _load_observer_test_support() -> ModuleType:
    path = ROOT.parent / "tests/examples/dataset/test_observe_q30t_ptv23_row_schemas.py"
    spec = importlib.util.spec_from_file_location("q30t_observer_test_support", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _generated_submitter(
    tmp_path: Path,
    *,
    python: str,
    git: str,
    sbatch: Path,
    ssh: Path,
    receipt_root: Path,
    fetch_url: str,
) -> Path:
    """Create a non-production harness by substituting reviewed constants only."""
    source = SUBMITTER.read_text()
    receipt_constant = (
        'readonly receipt_root="/lustre/fsw/coreai_dlalgo_llm/users/sna/'
        'modelopt-qwen3-drafter-training/receipts/q30t-ptv23-complement-700k-v1"'
    )
    fetch_constant = (
        'readonly approved_fetch_url="ssh://git@gitlab-master.nvidia.com:12051/sna/modelopt.git"'
    )
    replacements = {
        receipt_constant: f'readonly receipt_root="{receipt_root}"',
        'readonly approved_git="/usr/bin/git"': f'readonly approved_git="{git}"',
        'readonly approved_sbatch="/usr/bin/sbatch"': f'readonly approved_sbatch="{sbatch}"',
        'readonly approved_ssh="/usr/bin/ssh"': f'readonly approved_ssh="{ssh}"',
        'readonly approved_python="/usr/bin/python3.12"': f'readonly approved_python="{python}"',
        fetch_constant: f'readonly approved_fetch_url="{fetch_url}"',
        "approved_uids = {0}": "approved_uids = {0, os.getuid()}",
        "if sys.version_info < (3, 12):": "if False and sys.version_info < (3, 12):",
    }
    for original, replacement in replacements.items():
        assert source.count(original) == 1
        source = source.replace(original, replacement)
    harness = tmp_path / "submitter-harness.sh"
    harness.write_text(source)
    harness.chmod(0o700)
    return harness


def test_static_runner_requests_one_cpu_only_node() -> None:
    """The static scheduler request must remain CPU-only."""
    runner = RUNNER.read_text()

    assert "#SBATCH --nodes=1" in runner
    assert "#SBATCH --ntasks=1" in runner
    assert "#SBATCH --cpus-per-task=32" in runner
    assert "--gpus" not in runner


def test_scratch_allocator_falls_back_when_slurm_tmpdir_is_absent(tmp_path: Path) -> None:
    """Ptyche without SLURM_TMPDIR receives a private bounded node-scratch directory."""
    scratch_root = tmp_path / "raid" / "scratch"
    scratch_root.mkdir(parents=True, mode=0o700)
    scratch_root.chmod(0o1777)
    allocator = _scratch_allocator_test_source(scratch_root)

    result = subprocess.run(
        [sys.executable, "-I", "-S", "-", "test_user", "75209087", ""],
        input=allocator,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    path_value, device, inode = result.stdout.strip().split("\t")
    path = Path(path_value)
    assert path.parent == scratch_root / "test_user"
    assert re.fullmatch(r"q30t-row-schema-75209087-[0-9a-f]{32}", path.name)
    assert path.stat().st_mode & 0o777 == 0o700
    assert (path.stat().st_dev, path.stat().st_ino) == (int(device), int(inode))


def test_root_group_writable_raid_ancestor_is_safe_for_a_non_root_group_user() -> None:
    """Ptyche's root:root mode-0775 /raid is an approved fixed ancestor."""
    result = _run_root_component_policy(
        mode=0o775,
        owner=0,
        group=0,
        effective_groups={20, 100},
    )

    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize(
    ("owner", "group", "mode", "effective_groups"),
    [
        (1000, 0, 0o775, {20, 100}),
        (0, 20, 0o775, {20, 100}),
        (0, 0, 0o775, {0, 20}),
        (0, 0, 0o777, {20, 100}),
    ],
)
def test_writable_scratch_ancestors_reject_untrusted_owner_group_or_permissions(
    owner: int, group: int, mode: int, effective_groups: set[int]
) -> None:
    """Only root:root group-write outside the caller's groups is safe."""
    result = _run_root_component_policy(
        mode=mode,
        owner=owner,
        group=group,
        effective_groups=effective_groups,
    )

    assert result.returncode != 0
    assert "unsafe" in result.stderr or "root-owned" in result.stderr


def test_final_shared_scratch_root_accepts_exact_root_sticky_mode_1777() -> None:
    """The final shared boundary retains Ptyche's root:root sticky-1777 contract."""
    result = _run_root_component_policy(
        mode=0o1777,
        owner=0,
        group=0,
        effective_groups={20, 100},
        shared_root=True,
    )

    assert result.returncode == 0, result.stderr


def test_final_shared_scratch_root_rejects_non_exact_sticky_mode() -> None:
    """Sticky semantics do not broaden the fixed final mode beyond 1777."""
    result = _run_root_component_policy(
        mode=0o1770,
        owner=0,
        group=0,
        effective_groups={20, 100},
        shared_root=True,
    )

    assert result.returncode != 0
    assert "unsafe write permissions" in result.stderr


def test_final_shared_scratch_root_rejects_non_root_group() -> None:
    """The fixed final boundary must remain root:root, not only root-owned."""
    result = _run_root_component_policy(
        mode=0o1777,
        owner=0,
        group=20,
        effective_groups={100},
        shared_root=True,
    )

    assert result.returncode != 0
    assert "unsafe write permissions" in result.stderr


def test_materialization_never_truncates_through_a_descendant_symlink(tmp_path: Path) -> None:
    """An attacker-created examples symlink cannot redirect blob writes to foreign data."""
    scratch_root = tmp_path / "scratch"
    scratch_root.mkdir(mode=0o700)
    foreign = tmp_path / "foreign-dir"
    (foreign / "dataset").mkdir(parents=True)
    foreign_observer = foreign / "dataset" / "observe_q30t_ptv23_row_schemas.py"
    foreign_observer.write_bytes(b"foreign-data-must-survive")
    foreign_identity = (foreign_observer.stat().st_dev, foreign_observer.stat().st_ino)
    (scratch_root / "examples").symlink_to(foreign, target_is_directory=True)

    result = _run_current_materialization_segment(tmp_path, scratch_root)

    assert result.returncode != 0
    assert foreign_observer.read_bytes() == b"foreign-data-must-survive"
    assert (foreign_observer.stat().st_dev, foreign_observer.stat().st_ino) == foreign_identity


def test_materialization_never_truncates_an_existing_hardlink(tmp_path: Path) -> None:
    """Exclusive final creation rejects a hostile hardlink without changing its target."""
    scratch_root = tmp_path / "scratch"
    target_parent = scratch_root / "examples" / "dataset"
    target_parent.mkdir(parents=True, mode=0o700)
    foreign = tmp_path / "foreign.keep"
    foreign.write_bytes(b"foreign-hardlink-data")
    target = target_parent / "observe_q30t_ptv23_row_schemas.py"
    os.link(foreign, target)
    identity = (foreign.stat().st_dev, foreign.stat().st_ino, foreign.stat().st_nlink)

    result = _run_current_materialization_segment(tmp_path, scratch_root)

    assert result.returncode != 0
    assert foreign.read_bytes() == b"foreign-hardlink-data"
    assert target.read_bytes() == b"foreign-hardlink-data"
    assert (foreign.stat().st_dev, foreign.stat().st_ino, foreign.stat().st_nlink) == identity


def test_materialization_creates_exact_read_only_single_link_blobs(tmp_path: Path) -> None:
    """The held-root materializer publishes complete authenticated blobs exclusively."""
    scratch_root = tmp_path / "scratch"
    scratch_root.mkdir(mode=0o700)

    result = _run_current_materialization_segment(tmp_path, scratch_root)

    assert result.returncode == 0, result.stderr
    values = result.stdout.strip().split("\t")
    assert len(values) == 12
    expected = (b"observer-bytes", b"plan-bytes", b"atomic-bytes")
    paths = (
        scratch_root / "examples/dataset/observe_q30t_ptv23_row_schemas.py",
        scratch_root / "examples/dataset/qwen3_30ba3b_thinking_ptv3_stage_subset_v1.json",
        scratch_root / "tools/launcher/common/specdec/qwen4b_b_atomic.py",
    )
    for ordinal, (path, content) in enumerate(zip(paths, expected, strict=True)):
        metadata = path.stat()
        offset = ordinal * 4
        assert path.read_bytes() == content
        assert metadata.st_mode & 0o777 == 0o400
        assert metadata.st_nlink == 1
        assert values[offset : offset + 4] == [
            str(metadata.st_dev),
            str(metadata.st_ino),
            str(len(content)),
            hashlib.sha256(content).hexdigest(),
        ]


def test_post_materialization_fifo_rebind_never_reaches_a_bash_pathname_open(
    tmp_path: Path,
) -> None:
    """A descendant rebind to a foreign FIFO must reject without blocking."""
    scratch_root = tmp_path / "scratch"
    scratch_root.mkdir(mode=0o700)
    materialized = _run_current_materialization_segment(tmp_path, scratch_root)
    assert materialized.returncode == 0, materialized.stderr
    (scratch_root / "examples").rename(scratch_root / "examples-held")
    foreign = tmp_path / "foreign"
    (foreign / "dataset").mkdir(parents=True)
    sentinel = foreign / "preserve.keep"
    sentinel.write_bytes(b"foreign-preserved")
    sentinel_identity = (sentinel.stat().st_dev, sentinel.stat().st_ino)
    os.mkfifo(foreign / "dataset/observe_q30t_ptv23_row_schemas.py", 0o600)
    (scratch_root / "examples").symlink_to(foreign, target_is_directory=True)

    result, timed_out = _run_current_observer_adoption(scratch_root, materialized.stdout)

    assert 'exec 10<"$observer"' not in RUNNER.read_text()
    assert not timed_out, "Bash pathname adoption blocked on the rebound foreign FIFO"
    assert result.returncode != 0
    assert sentinel.read_bytes() == b"foreign-preserved"
    assert (sentinel.stat().st_dev, sentinel.stat().st_ino) == sentinel_identity


def test_post_materialization_hardlink_rebind_preserves_foreign_bytes(tmp_path: Path) -> None:
    """A substituted hardlink rejects without reading it as the reviewed blob."""
    scratch_root = tmp_path / "scratch"
    scratch_root.mkdir(mode=0o700)
    materialized = _run_current_materialization_segment(tmp_path, scratch_root)
    assert materialized.returncode == 0, materialized.stderr
    (scratch_root / "examples").rename(scratch_root / "examples-held")
    foreign = tmp_path / "foreign"
    (foreign / "dataset").mkdir(parents=True)
    sentinel = foreign / "preserve.keep"
    sentinel.write_bytes(b"foreign-hardlink-preserved")
    target = foreign / "dataset/observe_q30t_ptv23_row_schemas.py"
    os.link(sentinel, target)
    sentinel_identity = (
        sentinel.stat().st_dev,
        sentinel.stat().st_ino,
        sentinel.stat().st_nlink,
    )
    (scratch_root / "examples").symlink_to(foreign, target_is_directory=True)

    result, timed_out = _run_current_observer_adoption(scratch_root, materialized.stdout)

    assert not timed_out
    assert result.returncode != 0
    assert sentinel.read_bytes() == b"foreign-hardlink-preserved"
    assert target.read_bytes() == b"foreign-hardlink-preserved"
    assert (
        sentinel.stat().st_dev,
        sentinel.stat().st_ino,
        sentinel.stat().st_nlink,
    ) == sentinel_identity


def test_descriptor_relative_handoff_adopts_the_exact_materialized_blob(
    tmp_path: Path,
) -> None:
    """The bootstrap accepts the exact reviewed identity without a Bash reopen."""
    scratch_root = tmp_path / "scratch"
    scratch_root.mkdir(mode=0o700)
    materialized = _run_current_materialization_segment(tmp_path, scratch_root)
    assert materialized.returncode == 0, materialized.stderr

    result, timed_out = _run_current_observer_adoption(scratch_root, materialized.stdout)

    assert not timed_out
    assert result.returncode == 0, result.stderr


def test_scratch_allocator_preserves_safe_slurm_tmpdir_path(tmp_path: Path) -> None:
    """A safe scheduler-provided directory remains the scratch parent."""
    scratch_root = tmp_path / "raid" / "scratch"
    slurm_tmpdir = scratch_root / "slurm-job-75209087"
    slurm_tmpdir.mkdir(parents=True, mode=0o700)
    scratch_root.chmod(0o700)
    foreign = slurm_tmpdir / "foreign.keep"
    foreign.write_bytes(b"preserve-exactly")
    foreign_identity = (foreign.stat().st_dev, foreign.stat().st_ino)
    allocator = _scratch_allocator_test_source(scratch_root)

    result = subprocess.run(
        [sys.executable, "-I", "-S", "-", "test_user", "75209087", str(slurm_tmpdir)],
        input=allocator,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    path = Path(result.stdout.split("\t", 1)[0])
    assert path.parent == slurm_tmpdir
    assert not (scratch_root / "test_user").exists()
    assert foreign.read_bytes() == b"preserve-exactly"
    assert (foreign.stat().st_dev, foreign.stat().st_ino) == foreign_identity


def test_scratch_adopter_rejects_path_rebind_while_preserving_both_namespaces(
    tmp_path: Path,
) -> None:
    """Held-FD adoption fails if the returned scratch pathname is rebound."""
    adopter = _runner_embedded_python("Q30T_SCRATCH_ADOPTER")
    scratch = tmp_path / "scratch"
    scratch.mkdir(mode=0o700)
    original_sentinel = scratch / "original.keep"
    original_sentinel.write_bytes(b"original")
    descriptor = os.open(scratch, os.O_RDONLY)
    metadata = os.fstat(descriptor)
    moved = tmp_path / "scratch-moved"
    scratch.rename(moved)
    scratch.mkdir(mode=0o700)
    replacement_sentinel = scratch / "replacement.keep"
    replacement_sentinel.write_bytes(b"replacement")
    try:
        result = subprocess.run(
            [
                sys.executable,
                "-I",
                "-S",
                "-",
                str(descriptor),
                str(scratch),
                str(metadata.st_dev),
                str(metadata.st_ino),
            ],
            input=adopter,
            text=True,
            capture_output=True,
            pass_fds=(descriptor,),
            check=False,
        )
    finally:
        os.close(descriptor)

    assert result.returncode != 0
    assert "namespace changed" in result.stderr
    assert (moved / "original.keep").read_bytes() == b"original"
    assert replacement_sentinel.read_bytes() == b"replacement"


def test_scratch_adopter_accepts_the_exact_held_descriptor(tmp_path: Path) -> None:
    """The shell-to-Python handoff retains the authenticated directory descriptor."""
    adopter = _runner_embedded_python("Q30T_SCRATCH_ADOPTER")
    scratch = tmp_path / "scratch"
    scratch.mkdir(mode=0o700)
    descriptor = os.open(scratch, os.O_RDONLY)
    metadata = os.fstat(descriptor)
    try:
        result = subprocess.run(
            [
                sys.executable,
                "-I",
                "-S",
                "-",
                str(descriptor),
                str(scratch),
                str(metadata.st_dev),
                str(metadata.st_ino),
            ],
            input=adopter,
            text=True,
            capture_output=True,
            pass_fds=(descriptor,),
            check=False,
        )
    finally:
        os.close(descriptor)

    assert result.returncode == 0, result.stderr


def test_scratch_allocator_rejects_unsafe_existing_user_dir_without_deletion(
    tmp_path: Path,
) -> None:
    """An unsafe fallback namespace fails without touching foreign contents."""
    scratch_root = tmp_path / "raid" / "scratch"
    user_dir = scratch_root / "test_user"
    user_dir.mkdir(parents=True, mode=0o700)
    scratch_root.chmod(0o700)
    user_dir.chmod(0o755)
    foreign = user_dir / "foreign.keep"
    foreign.write_bytes(b"do-not-delete")
    foreign_identity = (foreign.stat().st_dev, foreign.stat().st_ino)
    allocator = _scratch_allocator_test_source(scratch_root)

    result = subprocess.run(
        [sys.executable, "-I", "-S", "-", "test_user", "75209087", ""],
        input=allocator,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode != 0
    assert "mode-0700" in result.stderr
    assert foreign.read_bytes() == b"do-not-delete"
    assert (foreign.stat().st_dev, foreign.stat().st_ino) == foreign_identity
    assert {path.name for path in user_dir.iterdir()} == {"foreign.keep"}


def test_scratch_allocator_rejects_a_symlinked_user_dir_and_preserves_target(
    tmp_path: Path,
) -> None:
    """Fallback adoption never follows a user-directory symlink."""
    scratch_root = tmp_path / "raid" / "scratch"
    scratch_root.mkdir(parents=True, mode=0o700)
    foreign_target = tmp_path / "foreign-target"
    foreign_target.mkdir(mode=0o700)
    sentinel = foreign_target / "sentinel"
    sentinel.write_bytes(b"unchanged")
    (scratch_root / "test_user").symlink_to(foreign_target, target_is_directory=True)
    allocator = _scratch_allocator_test_source(scratch_root)

    result = subprocess.run(
        [sys.executable, "-I", "-S", "-", "test_user", "75209087", ""],
        input=allocator,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode != 0
    assert "component is unavailable" in result.stderr
    assert sentinel.read_bytes() == b"unchanged"
    assert {path.name for path in foreign_target.iterdir()} == {"sentinel"}


def test_scratch_allocator_rejects_a_symlinked_root_ancestor_and_preserves_target(
    tmp_path: Path,
) -> None:
    """The fixed-root component walk never follows a rebound ancestor symlink."""
    foreign_raid = tmp_path / "foreign-raid"
    scratch_root = foreign_raid / "scratch"
    scratch_root.mkdir(parents=True, mode=0o700)
    sentinel = foreign_raid / "foreign.keep"
    sentinel.write_bytes(b"unchanged-root-target")
    (tmp_path / "raid").symlink_to(foreign_raid, target_is_directory=True)
    allocator = _scratch_allocator_test_source(tmp_path / "raid" / "scratch")

    result = subprocess.run(
        [sys.executable, "-I", "-S", "-", "test_user", "75209087", ""],
        input=allocator,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode != 0
    assert "component is unavailable" in result.stderr
    assert sentinel.read_bytes() == b"unchanged-root-target"
    assert not (scratch_root / "test_user").exists()


def test_scratch_allocator_rejects_unsafe_shared_root_mode(tmp_path: Path) -> None:
    """A writable shared root without sticky semantics cannot host fallback state."""
    scratch_root = tmp_path / "raid" / "scratch"
    scratch_root.mkdir(parents=True, mode=0o700)
    scratch_root.chmod(0o777)
    allocator = _scratch_allocator_test_source(scratch_root)

    result = subprocess.run(
        [sys.executable, "-I", "-S", "-", "test_user", "75209087", ""],
        input=allocator,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode != 0
    assert "unsafe write permissions" in result.stderr
    assert not (scratch_root / "test_user").exists()


def test_scratch_allocator_rejects_root_namespace_swap_without_deletion(tmp_path: Path) -> None:
    """A fixed-root rebind is detected through the held ancestor descriptor."""
    scratch_root = tmp_path / "raid" / "scratch"
    scratch_root.mkdir(parents=True, mode=0o700)
    foreign = scratch_root / "foreign.keep"
    foreign.write_bytes(b"original-root")
    allocator = _scratch_allocator_test_source(scratch_root)
    primitives = allocator.rsplit("\nmain()", 1)[0]
    exercise = (
        primitives
        + "\nroot_parent_fd, root_fd, root_name = open_fixed_root(scratch_root_path)\n"
        + "replacement = scratch_root_path + '-replacement'\n"
        + "os.rename(scratch_root_path, replacement)\n"
        + "os.mkdir(scratch_root_path, 0o700)\n"
        + "try:\n    require_named_binding(root_parent_fd, root_name, root_fd, 'scratch root')\n"
        + "finally:\n    os.close(root_fd); os.close(root_parent_fd)\n"
    )

    result = subprocess.run(
        [sys.executable, "-I", "-S", "-"],
        input=exercise,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode != 0
    assert "scratch root namespace changed" in result.stderr
    assert (Path(f"{scratch_root}-replacement") / "foreign.keep").read_bytes() == b"original-root"
    assert not any(scratch_root.iterdir())


@pytest.mark.parametrize(
    ("username", "job_id", "message"),
    [("../user", "75209087", "scratch user name"), ("test_user", "job-7", "SLURM job id")],
)
def test_scratch_allocator_rejects_invalid_user_and_job_names(
    tmp_path: Path, username: str, job_id: str, message: str
) -> None:
    """Untrusted identity strings cannot become scratch path components."""
    scratch_root = tmp_path / "raid" / "scratch"
    scratch_root.mkdir(parents=True, mode=0o700)
    allocator = _scratch_allocator_test_source(scratch_root)

    result = subprocess.run(
        [sys.executable, "-I", "-S", "-", username, job_id, ""],
        input=allocator,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode != 0
    assert message in result.stderr
    assert not any(scratch_root.iterdir())


def test_completion_binder_rejects_a_file_beyond_the_declared_bound(tmp_path: Path) -> None:
    """The executable completion binder must fail before hashing an oversized file."""
    runner = RUNNER.read_text()
    marker = "# Q30T_COMPLETION_BINDER\n"
    binder = runner.split(marker, 1)[1].split("\nPY\n", 1)[0]
    completion = tmp_path / "completion.json"
    completion.write_bytes(b"x" * (1024 * 1024 + 1))

    result = subprocess.run(
        [sys.executable, "-I", "-S", "-", str(completion)],
        input=binder,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode != 0
    assert "exceeds" in result.stderr
    assert not result.stdout.strip()


def test_tool_authenticator_rejects_group_writable_executables(tmp_path: Path) -> None:
    """A writable external tool must fail the executable trust boundary."""
    runner = RUNNER.read_text()
    marker = "# Q30T_TOOL_AUTHENTICATOR\n"
    authenticator = runner.split(marker, 1)[1].split("\nPY\n", 1)[0]
    executable = tmp_path / "git"
    executable.write_text("#!/bin/sh\nexit 0\n")
    executable.chmod(0o775)

    result = subprocess.run(
        [sys.executable, "-I", "-S", "-", "--test-owner", str(executable)],
        input=authenticator,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode != 0
    assert "writable" in result.stderr
    assert os.access(executable, os.X_OK)


def test_immutable_bootstrap_rejects_replaced_materialized_observer(tmp_path: Path) -> None:
    """A same-UID pathname replacement cannot substitute the authenticated code bytes."""
    bootstrap = _runner_embedded_python("Q30T_IMMUTABLE_BOOTSTRAP")
    observer = tmp_path / "observer.py"
    trusted = b"def main():\n    return 0\n"
    observer.write_bytes(trusted)
    expected = hashlib.sha256(trusted).hexdigest()
    marker = tmp_path / "substituted-code-ran"
    observer.write_text(f"from pathlib import Path\nPath({str(marker)!r}).touch()\n")
    atomic = tmp_path / "atomic.py"
    atomic.write_bytes(ATOMIC.read_bytes())
    atomic_sha256 = hashlib.sha256(atomic.read_bytes()).hexdigest()

    result = subprocess.run(
        [
            sys.executable,
            "-I",
            "-S",
            "-",
            str(observer),
            expected,
            str(atomic),
            atomic_sha256,
        ],
        input=bootstrap,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode != 0
    assert "observer blob SHA-256 mismatch" in result.stderr
    assert not marker.exists()


def test_immutable_bootstrap_executes_held_bytes_after_path_rebind(tmp_path: Path) -> None:
    """A pathname replacement after authentication cannot change executed bytes."""
    bootstrap = _runner_embedded_python("Q30T_IMMUTABLE_BOOTSTRAP")
    primitives = bootstrap.split("# Q30T_BOOTSTRAP_ENTRY", 1)[0]
    trusted_marker = tmp_path / "trusted-code-ran"
    substituted_marker = tmp_path / "substituted-code-ran"
    observer = tmp_path / "observer.py"
    trusted = f"from pathlib import Path\nPath({str(trusted_marker)!r}).touch()\n".encode()
    observer.write_bytes(trusted)
    replacement = tmp_path / "replacement.py"
    replacement.write_text(f"from pathlib import Path\nPath({str(substituted_marker)!r}).touch()\n")
    exercise = (
        primitives
        + "\npath, expected, replacement = sys.argv[1:4]\n"
        + "descriptor, authenticated = open_blob(path, expected, 'observer')\n"
        + "os.replace(replacement, path)\n"
        + "try:\n    exec(compile(authenticated, f'/proc/self/fd/{descriptor}', 'exec'), {})\n"
        + "finally:\n    os.close(descriptor)\n"
    )

    result = subprocess.run(
        [
            sys.executable,
            "-I",
            "-S",
            "-",
            str(observer),
            hashlib.sha256(trusted).hexdigest(),
            str(replacement),
        ],
        input=exercise,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert trusted_marker.exists()
    assert not substituted_marker.exists()


def test_immutable_bootstrap_executes_adopted_descriptor_after_name_rebind(
    tmp_path: Path,
) -> None:
    """Production's fd:N interface never reopens a rebound scratch pathname."""
    bootstrap = _runner_embedded_python("Q30T_IMMUTABLE_BOOTSTRAP")
    primitives = bootstrap.split("# Q30T_BOOTSTRAP_ENTRY", 1)[0]
    trusted_marker = tmp_path / "trusted-code-ran"
    substituted_marker = tmp_path / "substituted-code-ran"
    observer = tmp_path / "observer.py"
    trusted = f"from pathlib import Path\nPath({str(trusted_marker)!r}).touch()\n".encode()
    observer.write_bytes(trusted)
    observer.chmod(0o400)
    descriptor = os.open(observer, os.O_RDONLY)
    moved = tmp_path / "observer-held.py"
    observer.rename(moved)
    observer.write_text(f"from pathlib import Path\nPath({str(substituted_marker)!r}).touch()\n")
    exercise = (
        primitives
        + "\nreference, expected = sys.argv[1:3]\n"
        + "descriptor, authenticated = open_blob(reference, expected, 'observer')\n"
        + "try:\n    exec(compile(authenticated, f'/proc/self/fd/{descriptor}', 'exec'), {})\n"
        + "finally:\n    os.close(descriptor)\n"
    )
    try:
        result = subprocess.run(
            [
                sys.executable,
                "-I",
                "-S",
                "-",
                f"fd:{descriptor}",
                hashlib.sha256(trusted).hexdigest(),
            ],
            input=exercise,
            text=True,
            capture_output=True,
            pass_fds=(descriptor,),
            check=False,
        )
    finally:
        os.close(descriptor)

    assert result.returncode == 0, result.stderr
    assert trusted_marker.exists()
    assert not substituted_marker.exists()


def test_bootstrap_authenticates_runtime_before_observer_top_level(tmp_path: Path) -> None:
    """An unapproved runtime must fail before any authenticated observer byte executes."""
    bootstrap = _runner_embedded_python("Q30T_IMMUTABLE_BOOTSTRAP")
    marker = tmp_path / "observer-top-level-ran"
    observer = tmp_path / "observer.py"
    observer.write_text(
        f"from pathlib import Path\nPath({str(marker)!r}).touch()\ndef main():\n    return 0\n"
    )
    observer_sha256 = hashlib.sha256(observer.read_bytes()).hexdigest()
    atomic = tmp_path / "atomic.py"
    atomic.write_bytes(ATOMIC.read_bytes())
    atomic_sha256 = hashlib.sha256(atomic.read_bytes()).hexdigest()

    result = subprocess.run(
        [
            sys.executable,
            "-I",
            "-S",
            "-",
            str(observer),
            observer_sha256,
            str(atomic),
            atomic_sha256,
        ],
        input=bootstrap,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode != 0
    assert "runtime" in result.stderr.lower()
    assert not marker.exists()


def test_submitter_behavior_rejects_wrong_arity_and_unapproved_roots(tmp_path: Path) -> None:
    """Only the fixed submitter interface and receipt roots are accepted."""
    bash = shutil.which("bash")
    assert bash
    no_arguments = subprocess.run(
        [bash, "-p", str(SUBMITTER)], text=True, capture_output=True, check=False
    )
    assert no_arguments.returncode == 2
    assert "usage:" in no_arguments.stderr

    source = tmp_path / "source"
    source.mkdir()
    wrong_root = subprocess.run(
        [
            bash,
            "-p",
            str(SUBMITTER),
            "--test-only",
            "--source-path",
            str(source),
            "--output",
            str(tmp_path / "observation-a.json"),
            "--slurm-output",
            str(tmp_path / "row-schema-a-%j.out"),
        ],
        text=True,
        capture_output=True,
        check=False,
    )
    assert wrong_root.returncode == 2
    assert "approved row-schema observation path" in wrong_root.stderr


def test_exact_submitter_has_no_environment_test_override_surface(tmp_path: Path) -> None:
    """Linux-style fake tool overrides cannot execute through the production artifact."""
    submitter = SUBMITTER.read_text()
    assert "Q30T_TEST_" not in submitter
    assert "OSTYPE" not in submitter
    bash = shutil.which("bash")
    assert bash
    source = tmp_path / "source"
    source.mkdir()
    marker = tmp_path / "fake-tool-executed"
    fake_directory = tmp_path / "fake-bin"
    fake_directory.mkdir()
    fake = fake_directory / "git"
    fake.write_text(f"#!/bin/sh\ntouch '{marker}'\nexit 99\n")
    fake.chmod(0o700)
    receipt_root = Path(
        "/lustre/fsw/coreai_dlalgo_llm/users/sna/modelopt-qwen3-drafter-training/receipts/"
        "q30t-ptv23-complement-700k-v1"
    )
    environment = {
        "PATH": str(fake_directory),
        "OSTYPE": "linux-gnu",
        "Q30T_TEST_ALLOW_SYSTEM_EXECUTABLES": "non-linux-test",
        "Q30T_TEST_PYTHON": str(fake),
        "Q30T_TEST_GIT": str(fake),
        "Q30T_TEST_SBATCH": str(fake),
        "Q30T_TEST_SSH": str(fake),
        "Q30T_TEST_RECEIPT_ROOT": str(tmp_path / "receipts"),
        "Q30T_TEST_FETCH_URL": str(tmp_path / "remote.git"),
    }
    result = subprocess.run(
        [
            bash,
            "-p",
            str(SUBMITTER),
            "--test-only",
            "--source-path",
            str(source),
            "--output",
            str(receipt_root / "row-schema/observation-a.json"),
            "--slurm-output",
            str(receipt_root / "logs/row-schema-a-%j.out"),
        ],
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode != 0
    assert not marker.exists()


def test_submitter_behavior_requires_clean_live_pushed_source_and_preflights(
    tmp_path: Path,
) -> None:
    """Only a clean, live-pushed commit can reach ordered scheduler preflights."""
    git = shutil.which("git")
    bash = shutil.which("bash")
    assert git and bash
    bare = tmp_path / "remote.git"
    source = tmp_path / "source"
    subprocess.run([git, "init", "--bare", str(bare)], check=True, capture_output=True)
    subprocess.run([git, "init", "-b", "main", str(source)], check=True, capture_output=True)
    subprocess.run([git, "-C", str(source), "config", "user.email", "test@example.com"], check=True)
    subprocess.run([git, "-C", str(source), "config", "user.name", "Test"], check=True)
    tracked = source / "tools/launcher/common/specdec"
    tracked.mkdir(parents=True)
    (tracked / RUNNER.name).write_bytes(RUNNER.read_bytes())
    subprocess.run([git, "-C", str(source), "add", "."], check=True)
    subprocess.run(
        [git, "-C", str(source), "commit", "-m", "runner"], check=True, capture_output=True
    )
    subprocess.run([git, "-C", str(source), "remote", "add", "gitlab", str(bare)], check=True)
    subprocess.run(
        [git, "-C", str(source), "push", "-u", "gitlab", "main"], check=True, capture_output=True
    )
    calls = tmp_path / "sbatch.calls"
    spooled = tmp_path / "spooled-runner"
    sbatch = tmp_path / "sbatch"
    sbatch.write_text(f"#!/bin/sh\nprintf '%s\\n' \"$*\" >>'{calls}'\ncat >'{spooled}'\n")
    sbatch.chmod(0o700)
    ssh_calls = tmp_path / "ssh.calls"
    ssh = tmp_path / "ssh"
    ssh.write_text(
        f"#!/bin/sh\nprintf '%s\\n' \"$*\" >>'{ssh_calls}'\nexec '{git}' upload-pack '{bare}'\n"
    )
    ssh.chmod(0o700)
    fetch_url = "ssh://git@example.invalid/approved/modelopt.git"
    subprocess.run([git, "-C", str(source), "remote", "set-url", "gitlab", fetch_url], check=True)
    receipt_root = tmp_path / "receipts"
    output = receipt_root / "row-schema/observation-a.json"
    slurm_output = receipt_root / "logs/row-schema-a-%j.out"
    harness = _generated_submitter(
        tmp_path,
        python=sys.executable,
        git=git,
        sbatch=sbatch,
        ssh=ssh,
        receipt_root=receipt_root,
        fetch_url=fetch_url,
    )
    environment = {
        "HOME": str(tmp_path),
        "USER": "test",
        "LOGNAME": "test",
    }
    command = [
        bash,
        "-p",
        str(harness),
        "--submit",
        "--source-path",
        str(source),
        "--output",
        str(output),
        "--slurm-output",
        str(slurm_output),
    ]

    result = subprocess.run(command, env=environment, text=True, capture_output=True, check=False)
    assert result.returncode == 0, result.stderr
    recorded = calls.read_text().splitlines()
    assert "--test-only" in recorded[0]
    assert "--parsable" in recorded[1]
    assert spooled.read_bytes() == RUNNER.read_bytes()
    ssh_arguments = ssh_calls.read_text().splitlines()
    assert ssh_arguments
    assert all("-F /dev/null" in arguments for arguments in ssh_arguments)

    (tracked / RUNNER.name).write_text("#!/bin/sh\nexit 99\n")
    mutable = subprocess.run(command, env=environment, text=True, capture_output=True, check=False)
    assert mutable.returncode != 0
    assert "not clean" in mutable.stderr
    assert spooled.read_bytes() == RUNNER.read_bytes()
    (tracked / RUNNER.name).write_bytes(RUNNER.read_bytes())

    (source / "dirty").write_text("untracked")
    rejected = subprocess.run(command, env=environment, text=True, capture_output=True, check=False)
    assert rejected.returncode != 0
    assert "not clean" in rejected.stderr
    assert calls.read_text().splitlines() == recorded

    (source / "dirty").unlink()
    (source / "unpushed").write_text("new commit")
    subprocess.run([git, "-C", str(source), "add", "unpushed"], check=True)
    subprocess.run(
        [git, "-C", str(source), "commit", "-m", "not pushed"],
        check=True,
        capture_output=True,
    )
    unpushed = subprocess.run(command, env=environment, text=True, capture_output=True, check=False)
    assert unpushed.returncode != 0
    assert "not pushed" in unpushed.stderr
    assert calls.read_text().splitlines() == recorded


def test_submitter_rejects_imported_bash_and_loader_environment(tmp_path: Path) -> None:
    """Imported functions and loader settings cannot enter launcher children."""
    bash = shutil.which("bash")
    assert bash
    source = tmp_path / "source"
    source.mkdir()
    marker = tmp_path / "sourced"
    injection = tmp_path / "inject.sh"
    injection.write_text(f"touch '{marker}'\n")
    receipt_root = Path(
        "/lustre/fsw/coreai_dlalgo_llm/users/sna/modelopt-qwen3-drafter-training/receipts/"
        "q30t-ptv23-complement-700k-v1"
    )
    command = [
        bash,
        "-p",
        str(SUBMITTER),
        "--test-only",
        "--source-path",
        str(source),
        "--output",
        str(receipt_root / "row-schema/observation-a.json"),
        "--slurm-output",
        str(receipt_root / "logs/row-schema-a-%j.out"),
    ]
    environment = {
        "BASH_ENV": str(injection),
        "LD_LIBRARY_PATH": str(tmp_path),
        "BASH_FUNC_git%%": "() { touch /tmp/forbidden; }",
    }

    result = subprocess.run(command, env=environment, text=True, capture_output=True, check=False)

    assert result.returncode != 0
    assert "unsafe inherited environment" in result.stderr
    assert not marker.exists()


def test_materialized_observer_and_atomic_dependency_execute_under_isolated_no_site_python(
    tmp_path: Path,
) -> None:
    """Exact commit blobs must execute the controlled observer CLI through publication."""
    support = _load_observer_test_support()
    fixture = support.stage_inputs(tmp_path / "fixture")
    git = shutil.which("git")
    assert git
    repository = tmp_path / "repository"
    subprocess.run([git, "init", str(repository)], check=True, capture_output=True)
    subprocess.run(
        [git, "-C", str(repository), "config", "user.email", "test@example.com"], check=True
    )
    subprocess.run([git, "-C", str(repository), "config", "user.name", "Test"], check=True)
    tracked_paths = (
        "examples/dataset/observe_q30t_ptv23_row_schemas.py",
        "tools/launcher/common/specdec/qwen4b_b_atomic.py",
    )
    for relative, source in zip(tracked_paths, (OBSERVER, ATOMIC), strict=True):
        destination = repository / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(source.read_bytes())
    subprocess.run([git, "-C", str(repository), "add", "."], check=True)
    subprocess.run(
        [git, "-C", str(repository), "commit", "-m", "materialized inputs"],
        check=True,
        capture_output=True,
    )
    commit = subprocess.run(
        [git, "-C", str(repository), "rev-parse", "HEAD^{commit}"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    materialized = tmp_path / "materialized"
    for relative in tracked_paths:
        destination = materialized / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        blob = subprocess.run(
            [git, "-C", str(repository), "cat-file", "blob", f"{commit}:{relative}"],
            check=True,
            capture_output=True,
        ).stdout
        destination.write_bytes(blob)
        destination.chmod(0o400)
    observer = materialized / tracked_paths[0]
    output_parent = tmp_path / "output"
    output_parent.mkdir()
    output = output_parent / "observation.json"
    runtime_json = support.json.dumps(support._TEST_RUNTIME, sort_keys=True)
    assert pyarrow.__file__ is not None
    site_parent = Path(pyarrow.__file__).resolve().parent.parent
    bootstrap = """
import json, runpy, sys
root, script, package_parent, runtime_json = sys.argv[1:5]
arguments = sys.argv[5:]
sys.path.insert(0, root)
sys.path.insert(0, package_parent)
namespace = runpy.run_path(script, run_name="materialized_observer")
runtime = json.loads(runtime_json)
module_globals = namespace["main"].__globals__
module_globals["_authenticate_runtime"] = lambda: runtime
module_globals["_require_imported_runtime"] = lambda value: None
sys.argv = [script, *arguments]
raise SystemExit(namespace["main"]())
"""
    inputs = fixture.inputs
    result = subprocess.run(
        [
            sys.executable,
            "-I",
            "-S",
            "-c",
            bootstrap,
            str(materialized),
            str(observer),
            str(site_parent),
            runtime_json,
            "--ptv2-plan",
            str(inputs.ptv2_plan_path),
            "--ptv2-plan-sha256",
            inputs.ptv2_plan_sha256,
            "--ptv2-completion",
            str(inputs.ptv2_completion_path),
            "--ptv2-completion-sha256",
            inputs.ptv2_completion_sha256,
            "--ptv2-root",
            str(inputs.ptv2_root),
            "--ptv3-plan",
            str(inputs.ptv3_plan_path),
            "--ptv3-plan-sha256",
            inputs.ptv3_plan_sha256,
            "--ptv3-completion",
            str(inputs.ptv3_completion_path),
            "--ptv3-completion-sha256",
            inputs.ptv3_completion_sha256,
            "--ptv3-root",
            str(inputs.ptv3_root),
            "--output",
            str(output),
        ],
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert output.exists()
    assert support.json.loads(output.read_bytes())["runtime"] == support._TEST_RUNTIME


def test_linux_production_gate_cannot_skip_missing_runtime() -> None:
    """A designated Linux gate must fail if its approved runtime closure is absent."""
    source = inspect.getsource(
        test_linux_production_bootstrap_executes_exact_artifacts_without_guard_bypass
    )

    assert "pytest.skip" not in source


@pytest.mark.skipif(
    sys.platform != "linux", reason="production runtime identity uses /proc/self/exe"
)
def test_linux_production_bootstrap_executes_exact_artifacts_without_guard_bypass(
    tmp_path: Path,
) -> None:
    """Linux executes the held exact artifacts through all production runtime guards."""
    approved_python = Path("/usr/bin/python3.12")
    assert approved_python.is_file(), "approved production Python is unavailable"
    probe = subprocess.run(
        [
            str(approved_python),
            "-I",
            "-S",
            "-c",
            (
                "import os,sysconfig; p=sysconfig.get_paths(); "
                "raise SystemExit(0 if any(os.path.isdir(os.path.join(p[k],'pyarrow')) "
                "for k in ('purelib','platlib')) else 1)"
            ),
        ],
        text=True,
        capture_output=True,
        check=False,
    )
    assert probe.returncode == 0, (
        "approved production PyArrow runtime is unavailable: "
        f"stdout={probe.stdout!r} stderr={probe.stderr!r}"
    )
    support = _load_observer_test_support()
    fixture = support.stage_inputs(tmp_path / "fixture")
    observer = tmp_path / "observe.py"
    atomic = tmp_path / "atomic.py"
    observer.write_bytes(OBSERVER.read_bytes())
    atomic.write_bytes(ATOMIC.read_bytes())
    observer.chmod(0o400)
    atomic.chmod(0o400)
    output_parent = tmp_path / "output"
    output_parent.mkdir()
    output = output_parent / "observation.json"
    inputs = fixture.inputs
    bootstrap = _runner_embedded_python("Q30T_IMMUTABLE_BOOTSTRAP")

    result = subprocess.run(
        [
            str(approved_python),
            "-I",
            "-S",
            "-",
            str(observer),
            hashlib.sha256(observer.read_bytes()).hexdigest(),
            str(atomic),
            hashlib.sha256(atomic.read_bytes()).hexdigest(),
            "--ptv2-plan",
            str(inputs.ptv2_plan_path),
            "--ptv2-plan-sha256",
            inputs.ptv2_plan_sha256,
            "--ptv2-completion",
            str(inputs.ptv2_completion_path),
            "--ptv2-completion-sha256",
            inputs.ptv2_completion_sha256,
            "--ptv2-root",
            str(inputs.ptv2_root),
            "--ptv3-plan",
            str(inputs.ptv3_plan_path),
            "--ptv3-plan-sha256",
            inputs.ptv3_plan_sha256,
            "--ptv3-completion",
            str(inputs.ptv3_completion_path),
            "--ptv3-completion-sha256",
            inputs.ptv3_completion_sha256,
            "--ptv3-root",
            str(inputs.ptv3_root),
            "--output",
            str(output),
        ],
        input=bootstrap,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    runtime = support.json.loads(output.read_bytes())["runtime"]
    assert runtime["python_stdlib_tree_file_count"] > 0
    assert runtime["site_packages_tree_file_count"] > 0
