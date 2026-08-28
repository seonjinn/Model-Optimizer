# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Compute-node execution contract for Q30 Thinking parent receipts."""

from __future__ import annotations

import os
import shlex
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
RUNNER = ROOT / "tools/launcher/common/specdec/run_q30t_parent_receipts.sbatch"
_BASH = shutil.which("bash")
if _BASH is None:
    raise RuntimeError("bash is required for Q30 parent runner tests")
BASH: str = _BASH


def _git(command: list[str], *, cwd: Path) -> str:
    result = subprocess.run(["git", *command], cwd=cwd, check=True, capture_output=True, text=True)
    return result.stdout.strip()


def _pushed_checkout(tmp_path: Path) -> tuple[Path, str]:
    remote = tmp_path / "remote.git"
    subprocess.run(["git", "init", "--bare", str(remote)], check=True, capture_output=True)
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(["init", "-b", "main"], cwd=repo)
    (repo / "tracked").write_text("source\n")
    module_root = repo / "tools/launcher/common/specdec"
    module_root.mkdir(parents=True)
    (module_root.parent / "__init__.py").write_text("")
    (module_root / "__init__.py").write_text("")
    (module_root / "q30t_parent_receipt.py").write_text(
        """import os
import sys
from pathlib import Path

args = sys.argv[1:]
output_root = Path(args[args.index("--output-root") + 1])
evidence = output_root.with_name("python-calls")
lines = [
    "NO_SITE=" + str(sys.flags.no_site),
    "ISOLATED=" + str(sys.flags.isolated),
    "SITE_IMPORTED=" + str("site" in sys.modules),
    "PATH=" + os.environ.get("PATH", "<unset>"),
    "PYTHONPATH=" + os.environ.get("PYTHONPATH", "<unset>"),
    "LD_PRELOAD=" + os.environ.get("LD_PRELOAD", "<unset>"),
    "SSH_ASKPASS=" + os.environ.get("SSH_ASKPASS", "<unset>"),
] + args
evidence.write_text("\\n".join(lines) + "\\n")
"""
    )
    _git(["add", "."], cwd=repo)
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
        cwd=repo,
    )
    _git(["remote", "add", "origin", str(remote)], cwd=repo)
    _git(["push", "-u", "origin", "main"], cwd=repo)
    return repo, _git(["rev-parse", "HEAD"], cwd=repo)


def _runner_environment(tmp_path: Path, repo: Path, commit: str) -> tuple[dict[str, str], Path]:
    bin_root = tmp_path / "bin"
    bin_root.mkdir()
    calls = tmp_path / "python-calls"
    git_evidence = tmp_path / "git-environments"
    git = bin_root / "git"
    git.write_text(
        '#!/bin/sh\n[ "$1" = --q30t-test-exec ] || exit 98\n'
        'evidence="$2"\nreal_git="$3"\nshift 3\n'
        'printf "PATH=%s LD_PRELOAD=%s SSH_ASKPASS=%s GIT_CONFIG_NOSYSTEM=%s '
        'GIT_CONFIG_GLOBAL=%s GIT_SSH_COMMAND=%s\\n" '
        '"${PATH-<unset>}" "${LD_PRELOAD-<unset>}" "${SSH_ASKPASS-<unset>}" '
        '"${GIT_CONFIG_NOSYSTEM-<unset>}" "${GIT_CONFIG_GLOBAL-<unset>}" '
        '"${GIT_SSH_COMMAND-<unset>}" '
        '>> "$evidence"\nexec "$real_git" "$@"\n'
    )
    git.chmod(0o755)
    audit = tmp_path / "AUDIT.json"
    audit.write_text("audit\n")
    target = tmp_path / "target"
    target.mkdir()
    target_identity = tmp_path / "identity.json"
    target_identity.write_text("identity\n")
    target_model_sha256 = tmp_path / "model.sha256"
    target_model_sha256.write_text("manifest\n")
    dflash = tmp_path / "dflash"
    dflash.mkdir()
    dspark = tmp_path / "dspark"
    dspark.mkdir()
    runtime = tmp_path / "runtime.tar.zst"
    runtime.write_bytes(b"runtime")
    environment = os.environ.copy()
    for name in list(environment):
        if name in {
            "BASH_ENV",
            "ENV",
            "CDPATH",
            "GLOBIGNORE",
            "SHELLOPTS",
            "BASHOPTS",
            "PERL5LIB",
            "RUBYLIB",
            "SSH_ASKPASS",
            "SSH_ASKPASS_REQUIRE",
        } or name.startswith(("LD_", "DYLD_", "PYTHON", "GIT_")):
            environment.pop(name)
    environment.update(
        {
            "PATH": f"{bin_root}{os.pathsep}{environment['PATH']}",
            "Q30T_TEST_ALLOW_SYSTEM_EXECUTABLES": "non-linux-test",
            "Q30T_TEST_PYTHON": str(Path(sys.executable).resolve()),
            "Q30T_TEST_GIT": str(git),
            "Q30T_TEST_GIT_EVIDENCE": str(git_evidence),
            "Q30T_TEST_REAL_GIT": "/usr/bin/git",
            "SLURM_EXPORT_ENV": "NONE",
            "SLURM_JOB_ID": "4242",
            "SLURM_NNODES": "1",
            "SLURM_CPUS_PER_TASK": "64",
            "REPO_ROOT": str(repo),
            "SOURCE_COMMIT": commit,
            "SOURCE_REMOTE_REF": "origin/main",
            "HISTORICAL_AUDIT": str(audit),
            "TARGET_MODEL_ROOT": str(target),
            "TARGET_REVISION": "a" * 40,
            "TARGET_IDENTITY_PATH": str(target_identity),
            "TARGET_IDENTITY_SHA256": "c" * 64,
            "TARGET_MODEL_SHA256_PATH": str(target_model_sha256),
            "TARGET_MODEL_SHA256_SHA256": "d" * 64,
            "DFLASH_PARENT_ROOT": str(dflash),
            "DSPARK_PARENT_ROOT": str(dspark),
            "RUNTIME_ARCHIVE": str(runtime),
            "OUTPUT_ROOT": str(tmp_path / "output"),
        }
    )
    environment.pop("SLURM_JOB_GPUS", None)
    environment.pop("SLURM_GPUS_ON_NODE", None)
    return environment, calls


def _runner_command(environment: dict[str, str]) -> list[str]:
    return [
        BASH,
        str(RUNNER),
        environment["REPO_ROOT"],
        environment["SOURCE_COMMIT"],
        environment["SOURCE_REMOTE_REF"],
        environment["HISTORICAL_AUDIT"],
        environment["TARGET_MODEL_ROOT"],
        environment["TARGET_REVISION"],
        environment["TARGET_IDENTITY_PATH"],
        environment["TARGET_IDENTITY_SHA256"],
        environment["TARGET_MODEL_SHA256_PATH"],
        environment["TARGET_MODEL_SHA256_SHA256"],
        environment["DFLASH_PARENT_ROOT"],
        environment["DSPARK_PARENT_ROOT"],
        environment["RUNTIME_ARCHIVE"],
        environment["OUTPUT_ROOT"],
    ]


def test_parent_runner_uses_sterile_absolute_executable_boundaries(tmp_path: Path) -> None:
    """PATH and all non-reserved environment state are absent from receipt children."""
    repo, commit = _pushed_checkout(tmp_path)
    environment, calls = _runner_environment(tmp_path, repo, commit)

    result = subprocess.run(
        _runner_command(environment),
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    python_evidence = calls.read_text().splitlines()[:7]
    assert python_evidence == [
        "NO_SITE=1",
        "ISOLATED=1",
        "SITE_IMPORTED=False",
        "PATH=<unset>",
        "PYTHONPATH=<unset>",
        "LD_PRELOAD=<unset>",
        "SSH_ASKPASS=<unset>",
    ]
    git_evidence = Path(environment["Q30T_TEST_GIT_EVIDENCE"]).read_text().splitlines()
    assert git_evidence
    assert all(str(tmp_path / "bin") not in line for line in git_evidence)
    assert all("LD_PRELOAD=<unset> SSH_ASKPASS=<unset>" in line for line in git_evidence)
    assert any(
        "GIT_CONFIG_NOSYSTEM=1 GIT_CONFIG_GLOBAL=/dev/null "
        "GIT_SSH_COMMAND=/usr/bin/ssh -F /dev/null -oBatchMode=yes" in line
        for line in git_evidence
    )


def test_parent_runner_executes_committed_module_bytes_after_checkout_mutation(
    tmp_path: Path,
) -> None:
    """A checkout change after Git authentication cannot change executed Python bytes."""
    repo, commit = _pushed_checkout(tmp_path)
    environment, calls = _runner_environment(tmp_path, repo, commit)
    module = repo / "tools/launcher/common/specdec/q30t_parent_receipt.py"
    git = Path(environment["Q30T_TEST_GIT"])
    malicious = f"from pathlib import Path\nPath({str(calls)!r}).write_text('MUTATED\\n')\n"
    git.write_text(
        "#!/bin/sh\n"
        '[ "$1" = --q30t-test-exec ] || exit 98\n'
        "real_git=$3\n"
        "shift 3\n"
        'case " $* " in\n'
        '  *" merge-base --is-ancestor "*)\n'
        '    "$real_git" "$@" || exit $?\n'
        f"    printf %s {shlex.quote(malicious)} > {shlex.quote(str(module))}\n"
        "    exit 0\n"
        "    ;;\n"
        "esac\n"
        'exec "$real_git" "$@"\n'
    )
    git.chmod(0o755)

    result = subprocess.run(
        _runner_command(environment),
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert calls.read_text().startswith("NO_SITE=1\n")


def test_parent_runner_rejects_git_replace_object_substitution(tmp_path: Path) -> None:
    """A replace ref cannot substitute parent-receipt code under a trusted commit hash."""
    repo, trusted_commit = _pushed_checkout(tmp_path)
    module = repo / "tools/launcher/common/specdec/q30t_parent_receipt.py"
    module.write_text("raise SystemExit(73)\n")
    _git(["add", str(module.relative_to(repo))], cwd=repo)
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
            "forged replacement",
        ],
        cwd=repo,
    )
    forged_commit = _git(["rev-parse", "HEAD"], cwd=repo)
    _git(["replace", trusted_commit, forged_commit], cwd=repo)
    _git(["reset", "--hard", trusted_commit], cwd=repo)
    environment, calls = _runner_environment(tmp_path, repo, trusted_commit)

    result = subprocess.run(
        _runner_command(environment),
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 2
    assert "clean" in result.stderr
    assert not calls.exists()


def test_parent_runner_requires_slurm_export_none_boundary(tmp_path: Path) -> None:
    """Receipt execution refuses a job environment exported outside Slurm reserved state."""
    runner = RUNNER.read_text(encoding="utf-8")
    assert "#SBATCH --export=NONE" in runner
    repo, commit = _pushed_checkout(tmp_path)
    environment, calls = _runner_environment(tmp_path, repo, commit)
    environment["SLURM_EXPORT_ENV"] = "ALL"

    result = subprocess.run(
        _runner_command(environment),
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 2
    assert "export=NONE" in result.stderr
    assert not calls.exists()


def test_parent_runner_strips_ptyche_debuginfod_environment() -> None:
    """Ptyche's site-injected debuginfod setting does not reject an export-none job."""
    result = subprocess.run(
        [BASH, str(RUNNER)],
        env={
            "DEBUGINFOD_URLS": "https://debuginfod.ubuntu.com",
            "SLURM_EXPORT_ENV": "NONE",
        },
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 2
    assert "usage:" in result.stderr
    assert "unexpected exported environment variable" not in result.stderr


def test_parent_runner_still_rejects_unknown_site_environment() -> None:
    """Only explicitly reviewed site state may cross the export-none boundary."""
    result = subprocess.run(
        [BASH, str(RUNNER)],
        env={"Q30T_UNKNOWN_SITE_STATE": "injected", "SLURM_EXPORT_ENV": "NONE"},
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 2
    assert "unexpected exported environment variable" in result.stderr


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("BASH_ENV", "/dev/null"),
        ("LD_PRELOAD", "/nonexistent/q30t-parent-loader.so"),
        ("PYTHONPATH", "/poisoned/inherited/path"),
        ("SSH_ASKPASS", "/nonexistent/injected-askpass"),
    ],
)
def test_parent_runner_rejects_prohibited_exported_state(
    tmp_path: Path, name: str, value: str
) -> None:
    """The runtime boundary rejects loader, shell, Python, and Git helper injection."""
    repo, commit = _pushed_checkout(tmp_path)
    environment, calls = _runner_environment(tmp_path, repo, commit)
    environment[name] = value

    result = subprocess.run(
        _runner_command(environment),
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 2
    assert "prohibited exported environment variable" in result.stderr
    assert not calls.exists()


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("SLURM_JOB_GPUS", "0"),
        ("SLURM_STEP_GPUS", "0"),
        ("SLURM_GPUS_ON_NODE", "1"),
        ("SLURM_GPUS", "1"),
        ("SLURM_GPUS_PER_NODE", "1"),
        ("SLURM_GPUS_PER_TASK", "1"),
        ("SLURM_TRES_PER_NODE", "gres/gpu:1"),
    ],
)
def test_parent_runner_rejects_any_gpu_allocation(tmp_path: Path, name: str, value: str) -> None:
    """A CPU receipt cannot run when any Slurm GPU allocation field is populated."""
    repo, commit = _pushed_checkout(tmp_path)
    environment, calls = _runner_environment(tmp_path, repo, commit)
    environment[name] = value

    result = subprocess.run(
        _runner_command(environment),
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 2
    assert "CPU-only" in result.stderr
    assert not calls.exists()


def test_parent_runner_uses_one_cpu_node_and_defers_hashes_to_the_cli(tmp_path: Path) -> None:
    """A clean pushed checkout invokes the real builder interface without a GPU request."""
    repo, commit = _pushed_checkout(tmp_path)
    environment, calls = _runner_environment(tmp_path, repo, commit)
    runner = RUNNER.read_text(encoding="utf-8")
    assert runner.startswith("#!/bin/bash\n")
    assert " -S -I -c " in runner
    assert 'readonly approved_env="/usr/bin/env"' in runner
    assert "stat.S_ISLNK(entry.st_mode)" in runner
    assert "entry_permissions=stat.S_ISLNK(entry.st_mode) or not entry.st_mode & 0o022" in runner

    result = subprocess.run(
        _runner_command(environment),
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    invocation = calls.read_text().splitlines()
    assert invocation[7:] == [
        "--historical-audit",
        environment["HISTORICAL_AUDIT"],
        "--target-model-root",
        environment["TARGET_MODEL_ROOT"],
        "--target-revision",
        "a" * 40,
        "--target-identity-path",
        environment["TARGET_IDENTITY_PATH"],
        "--target-identity-sha256",
        "c" * 64,
        "--target-model-sha256-path",
        environment["TARGET_MODEL_SHA256_PATH"],
        "--target-model-sha256-sha256",
        "d" * 64,
        "--dflash-parent-root",
        environment["DFLASH_PARENT_ROOT"],
        "--dspark-parent-root",
        environment["DSPARK_PARENT_ROOT"],
        "--runtime-archive",
        environment["RUNTIME_ARCHIVE"],
        "--source-commit",
        commit,
        "--output-root",
        environment["OUTPUT_ROOT"],
    ]


def test_parent_runner_rejects_multinode_execution(tmp_path: Path) -> None:
    """The receipt hash job cannot silently become a multinode allocation."""
    repo, commit = _pushed_checkout(tmp_path)
    environment, calls = _runner_environment(tmp_path, repo, commit)
    environment["SLURM_NNODES"] = "2"

    result = subprocess.run(
        _runner_command(environment),
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 2
    assert "one node" in result.stderr
    assert not calls.exists()


def test_parent_runner_rejects_a_clean_but_unpushed_source_commit(tmp_path: Path) -> None:
    """A local-only commit is not reproducible source for durable parent receipts."""
    repo, _ = _pushed_checkout(tmp_path)
    (repo / "tracked").write_text("local-only\n")
    _git(["add", "tracked"], cwd=repo)
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
        cwd=repo,
    )
    commit = _git(["rev-parse", "HEAD"], cwd=repo)
    environment, calls = _runner_environment(tmp_path, repo, commit)

    result = subprocess.run(
        _runner_command(environment),
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 2
    assert "pushed branch" in result.stderr
    assert not calls.exists()


def test_parent_runner_rejects_a_local_ref_disguised_as_the_pushed_ref(tmp_path: Path) -> None:
    """The trusted ref must resolve through refs/remotes, not a caller-chosen local branch."""
    repo, commit = _pushed_checkout(tmp_path)
    environment, calls = _runner_environment(tmp_path, repo, commit)
    environment["SOURCE_REMOTE_REF"] = "main"

    result = subprocess.run(
        _runner_command(environment),
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 2
    assert "remote-tracking" in result.stderr
    assert not calls.exists()


def test_parent_runner_rejects_a_forged_remote_tracking_ref(tmp_path: Path) -> None:
    """A locally forged refs/remotes entry cannot authorize an unpushed commit."""
    repo, _ = _pushed_checkout(tmp_path)
    (repo / "tracked").write_text("local-only\n")
    _git(["add", "tracked"], cwd=repo)
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
        cwd=repo,
    )
    commit = _git(["rev-parse", "HEAD"], cwd=repo)
    _git(["update-ref", "refs/remotes/origin/main", commit], cwd=repo)
    environment, calls = _runner_environment(tmp_path, repo, commit)

    result = subprocess.run(
        _runner_command(environment),
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 2
    assert "pushed branch" in result.stderr
    assert not calls.exists()


def test_parent_runner_defers_completed_retry_authentication_to_cli(tmp_path: Path) -> None:
    """An existing directory reaches the CLI for exact pair authentication and adoption."""
    repo, commit = _pushed_checkout(tmp_path)
    environment, calls = _runner_environment(tmp_path, repo, commit)
    Path(environment["OUTPUT_ROOT"]).mkdir()

    result = subprocess.run(
        _runner_command(environment),
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert calls.exists()
