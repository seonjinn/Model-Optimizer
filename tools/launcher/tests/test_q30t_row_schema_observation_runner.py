# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Validate the immutable CPU-only Q30 row-schema launcher boundary."""

from __future__ import annotations

import importlib.util
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import TYPE_CHECKING

import pyarrow  # pyright: ignore[reportMissingImports]

if TYPE_CHECKING:
    from types import ModuleType

ROOT = Path(__file__).resolve().parents[2]
RUNNER = ROOT / "launcher/common/specdec/run_q30t_row_schema_observation.sbatch"
SUBMITTER = ROOT / "launcher/common/specdec/submit_q30t_row_schema_observation.sh"
OBSERVER = ROOT.parent / "examples/dataset/observe_q30t_ptv23_row_schemas.py"
ATOMIC = ROOT / "launcher/common/specdec/qwen4b_b_atomic.py"


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


def test_completion_binder_rejects_a_file_beyond_the_declared_bound(tmp_path: Path) -> None:
    """The executable completion binder must fail before hashing an oversized file."""
    runner = RUNNER.read_text()
    marker = "# Q30T_COMPLETION_BINDER\n"
    binder = runner.split(marker, 1)[1].split("\nPY\n", 1)[0]
    completion = tmp_path / "completion.json"
    completion.write_bytes(b"x" * (1024 * 1024 + 1))

    result = subprocess.run(
        [sys.executable, "-I", "-", str(completion)],
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
        [sys.executable, "-I", "-", "--test-owner", str(executable)],
        input=authenticator,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode != 0
    assert "writable" in result.stderr
    assert os.access(executable, os.X_OK)


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
