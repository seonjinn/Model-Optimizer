# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Validate the immutable CPU-only Q30 row-schema launcher boundary."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RUNNER = ROOT / "launcher/common/specdec/run_q30t_row_schema_observation.sbatch"
SUBMITTER = ROOT / "launcher/common/specdec/submit_q30t_row_schema_observation.sh"


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
    ssh = tmp_path / "ssh"
    ssh.write_text("#!/bin/sh\nexit 97\n")
    ssh.chmod(0o700)
    receipt_root = tmp_path / "receipts"
    output = receipt_root / "row-schema/observation-a.json"
    slurm_output = receipt_root / "logs/row-schema-a-%j.out"
    environment = {
        "HOME": str(tmp_path),
        "USER": "test",
        "LOGNAME": "test",
        "OSTYPE": sys.platform,
        "Q30T_TEST_ALLOW_SYSTEM_EXECUTABLES": "non-linux-test",
        "Q30T_TEST_PYTHON": sys.executable,
        "Q30T_TEST_GIT": git,
        "Q30T_TEST_SBATCH": str(sbatch),
        "Q30T_TEST_SSH": str(ssh),
        "Q30T_TEST_RECEIPT_ROOT": str(receipt_root),
        "Q30T_TEST_FETCH_URL": str(bare),
    }
    command = [
        bash,
        "-p",
        str(SUBMITTER),
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
