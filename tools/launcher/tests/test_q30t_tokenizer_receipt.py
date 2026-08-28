# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Q30 Thinking tokenizer receipt contracts."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import shlex
import subprocess
import sys
import tracemalloc
from pathlib import Path

import pytest
from common.specdec import q30t_tokenizer_receipt as receipt_module
from common.specdec import q30t_tree_digest as tree_digest_module
from common.specdec.q30t_parent_receipt import _tree_sha256 as parent_tree_sha256
from common.specdec.q30t_tokenizer_receipt import (
    build_q30t_tokenizer_receipt,
    snapshot_tree_sha256,
    verify_q30t_tokenizer_receipt,
)

ROOT = Path(__file__).resolve().parents[3]
BUILDER_PATH = ROOT / "examples/dataset/build_qwen4b_ptv23_complement.py"
Q30_REPOSITORY = "Qwen/Qwen3-30B-A3B-Thinking-2507"
FIXTURE_Q30_REVISION = "a" * 40
RUNNER_PATH = ROOT / "tools/launcher/common/specdec/run_q30t_tokenizer_receipt.sbatch"
OFFICIAL_TEMPLATE = """{% for message in messages %}
    {%- elif message.role == "assistant" %}
        {{- '<|im_start|>' + message.role + '\\n' + message.content }}
        {{- '<|im_end|>\\n' }}
    {%- elif message.role == "tool" %}
{% endfor %}"""
TRAINING_TEMPLATE = """{% for message in messages %}
    {%- elif message.role == "assistant" %}
        {{- '<|im_start|>' + message.role + '\\n' }}
        {%- generation %}
        {{- message.content }}
        {{- '<|im_end|>\\n' }}
        {%- endgeneration %}
    {%- elif message.role == "tool" %}
{% endfor %}"""


def _canonical(value: object) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()


def _make_q30_snapshot(tmp_path: Path, im_start_token_id: int = 17) -> Path:
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    (snapshot / "tokenizer_config.json").write_bytes(
        _canonical({"chat_template": OFFICIAL_TEMPLATE})
    )
    (snapshot / "tokenizer.json").write_bytes(
        _canonical(
            {
                "added_tokens": [
                    {"content": "<|im_start|>", "id": im_start_token_id},
                    {"content": "<|im_end|>", "id": im_start_token_id + 1},
                ]
            }
        )
    )
    nested = snapshot / "assets"
    nested.mkdir()
    (nested / "tokenizer.model").write_bytes(b"fixture tokenizer asset")
    return snapshot


def _write_model_asset_manifests(snapshot: Path, root: Path) -> tuple[Path, str, Path, str]:
    root.mkdir()
    records = [
        (
            path.relative_to(snapshot).as_posix(),
            path.stat().st_size,
            hashlib.sha256(path.read_bytes()).hexdigest(),
        )
        for path in sorted(snapshot.rglob("*"))
        if path.is_file()
    ]
    model_sha256 = root / "model.sha256"
    model_sha256.write_text(
        "".join(f"{digest}  ./{path}\n" for path, _, digest in records), encoding="utf-8"
    )
    model_manifest_sha256 = hashlib.sha256(model_sha256.read_bytes()).hexdigest()
    identity = root / "identity.json"
    identity.write_bytes(
        _canonical(
            {
                "artifact": Q30_REPOSITORY,
                "bytes": sum(size for _, size, _ in records),
                "files": len(records),
                "model_sha256_manifest": model_manifest_sha256,
                "source_revision": FIXTURE_Q30_REVISION,
                "symlinks": 0,
            }
        )
        + b"\n"
    )
    return (
        identity,
        hashlib.sha256(identity.read_bytes()).hexdigest(),
        model_sha256,
        model_manifest_sha256,
    )


def _load_builder():
    spec = importlib.util.spec_from_file_location("build_qwen4b_ptv23_complement", BUILDER_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    sys.path.insert(0, str(BUILDER_PATH.parent))
    try:
        spec.loader.exec_module(module)
    finally:
        sys.path.pop(0)
    return module


def _rehashed(payload: dict[str, object]) -> bytes:
    body = {name: value for name, value in payload.items() if name != "receipt_sha256"}
    payload["receipt_sha256"] = hashlib.sha256(_canonical(body)).hexdigest()
    return _canonical(payload) + b"\n"


def _git(command: list[str], *, cwd: Path) -> str:
    result = subprocess.run(["git", *command], cwd=cwd, check=True, capture_output=True, text=True)
    return result.stdout.strip()


def _pushed_checkout(tmp_path: Path) -> tuple[Path, str]:
    remote = tmp_path / "remote.git"
    subprocess.run(["git", "init", "--bare", str(remote)], check=True, capture_output=True)
    repository = tmp_path / "repo"
    repository.mkdir()
    _git(["init", "-b", "main"], cwd=repository)
    (repository / "tracked").write_text("source\n")
    module_root = repository / "tools/launcher/common/specdec"
    module_root.mkdir(parents=True)
    (module_root.parent / "__init__.py").write_text("")
    (module_root / "__init__.py").write_text("")
    (module_root / "q30t_tokenizer_receipt.py").write_text(
        """import os
import sys
from pathlib import Path

args = sys.argv[1:]
output = Path(args[args.index("--output") + 1])
evidence = output.with_name("python-calls")
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
    _git(["add", "."], cwd=repository)
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
    _git(["remote", "add", "origin", str(remote)], cwd=repository)
    _git(["push", "-u", "origin", "main"], cwd=repository)
    return repository, _git(["rev-parse", "HEAD"], cwd=repository)


def _tokenizer_runner_environment(
    tmp_path: Path, repository: Path, source_commit: str
) -> tuple[dict[str, str], Path]:
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
    snapshot = _make_q30_snapshot(tmp_path)
    identity, identity_sha256, model_sha256, model_sha256_sha256 = _write_model_asset_manifests(
        snapshot, tmp_path / "manifest"
    )
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
            "SOURCE_PATH": str(repository),
            "SOURCE_SHA": source_commit,
            "SOURCE_REMOTE_REF": "origin/main",
            "TOKENIZER_REPOSITORY": Q30_REPOSITORY,
            "TOKENIZER_REVISION": FIXTURE_Q30_REVISION,
            "TOKENIZER_SNAPSHOT": str(snapshot),
            "IDENTITY_PATH": str(identity),
            "IDENTITY_SHA256": identity_sha256,
            "MODEL_SHA256_PATH": str(model_sha256),
            "MODEL_SHA256_SHA256": model_sha256_sha256,
            "OUTPUT_PATH": str(tmp_path / "TOKENIZER.json"),
        }
    )
    return environment, calls


def _tokenizer_runner_command(environment: dict[str, str]) -> list[str]:
    return [
        "/bin/bash",
        "-p",
        str(RUNNER_PATH),
        environment["SOURCE_PATH"],
        environment["SOURCE_SHA"],
        environment["SOURCE_REMOTE_REF"],
        environment["TOKENIZER_REPOSITORY"],
        environment["TOKENIZER_REVISION"],
        environment["TOKENIZER_SNAPSHOT"],
        environment["IDENTITY_PATH"],
        environment["IDENTITY_SHA256"],
        environment["MODEL_SHA256_PATH"],
        environment["MODEL_SHA256_SHA256"],
        environment["OUTPUT_PATH"],
    ]


def test_q30t_tokenizer_runner_uses_sterile_absolute_executable_boundaries(
    tmp_path: Path,
) -> None:
    """PATH and all non-reserved environment state are absent from receipt children."""
    repository, source_commit = _pushed_checkout(tmp_path)
    environment, calls = _tokenizer_runner_environment(tmp_path, repository, source_commit)

    result = subprocess.run(
        _tokenizer_runner_command(environment),
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


def test_tokenizer_runner_executes_committed_module_bytes_after_checkout_mutation(
    tmp_path: Path,
) -> None:
    """A checkout change after Git authentication cannot change executed Python bytes."""
    repository, source_commit = _pushed_checkout(tmp_path)
    environment, calls = _tokenizer_runner_environment(tmp_path, repository, source_commit)
    module = repository / "tools/launcher/common/specdec/q30t_tokenizer_receipt.py"
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
        _tokenizer_runner_command(environment),
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert calls.read_text().startswith("NO_SITE=1\n")


def test_tokenizer_runner_rejects_git_replace_object_substitution(tmp_path: Path) -> None:
    """Git replace refs cannot redefine the remotely authenticated source commit."""
    repository, trusted_commit = _pushed_checkout(tmp_path)
    module = repository / "tools/launcher/common/specdec/q30t_tokenizer_receipt.py"
    module.write_text("raise SystemExit(73)\n")
    _git(["add", str(module.relative_to(repository))], cwd=repository)
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
        cwd=repository,
    )
    forged_commit = _git(["rev-parse", "HEAD"], cwd=repository)
    _git(["replace", trusted_commit, forged_commit], cwd=repository)
    _git(["reset", "--hard", trusted_commit], cwd=repository)
    environment, calls = _tokenizer_runner_environment(tmp_path, repository, trusted_commit)

    result = subprocess.run(
        _tokenizer_runner_command(environment),
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 2
    assert "dirty" in result.stderr
    assert not calls.exists()


def test_q30t_tokenizer_runner_requires_slurm_export_none_boundary(tmp_path: Path) -> None:
    """Receipt execution refuses a job environment exported outside Slurm reserved state."""
    runner = RUNNER_PATH.read_text(encoding="utf-8")
    assert "#SBATCH --export=NONE" in runner
    repository, source_commit = _pushed_checkout(tmp_path)
    environment, calls = _tokenizer_runner_environment(tmp_path, repository, source_commit)
    environment["SLURM_EXPORT_ENV"] = "ALL"

    result = subprocess.run(
        _tokenizer_runner_command(environment),
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 2
    assert "export=NONE" in result.stderr
    assert not calls.exists()


def test_q30t_tokenizer_runner_requires_bash_privileged_mode() -> None:
    """Direct non-privileged Bash invocation cannot enter the receipt boundary."""
    result = subprocess.run(
        ["/bin/bash", str(RUNNER_PATH)],
        env={},
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 2
    assert "requires Bash privileged mode" in result.stderr


def test_q30t_tokenizer_runner_strips_standard_ptyche_batch_environment() -> None:
    """Standard site, login, and unknown benign state is cleared at the job boundary."""
    result = subprocess.run(
        ["/bin/bash", "-p", str(RUNNER_PATH)],
        env={
            "DEBUGINFOD_URLS": "https://debuginfod.ubuntu.com",
            "ENVIRONMENT": "BATCH",
            "HOME": "/home/sna",
            "LOGNAME": "sna",
            "PATH": "/usr/bin:/bin",
            "Q30T_UNKNOWN_SITE_STATE": "injected",
            "SLURM_EXPORT_ENV": "NONE",
            "USER": "sna",
        },
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 2
    assert "usage:" in result.stderr
    assert "unexpected exported environment variable" not in result.stderr


def test_q30t_tokenizer_runner_fails_closed_when_exported_state_is_readonly() -> None:
    """An ambient variable that Bash cannot clear aborts before external code."""
    result = subprocess.run(
        [
            "/bin/bash",
            "-p",
            "-c",
            'export HOME=/home/sna SLURM_EXPORT_ENV=NONE; readonly HOME; source "$1"',
            "q30t-tokenizer-test",
            str(RUNNER_PATH),
        ],
        env={},
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 2
    assert "usage:" not in result.stderr
    assert "cannot clear exported environment variable: HOME" in result.stderr


@pytest.mark.parametrize(
    ("name", "marker"),
    [("compgen", "COMPGEN_MARKER"), ("unset", "UNSET_MARKER")],
)
def test_q30t_tokenizer_runner_rejects_imported_boundary_functions(name: str, marker: str) -> None:
    """Imported functions cannot execute or shadow the pre-exec environment boundary."""
    result = subprocess.run(
        ["/bin/bash", "-p", str(RUNNER_PATH)],
        env={
            f"BASH_FUNC_{name}%%": f"() {{ printf {marker} >&2; }}",
            "SLURM_EXPORT_ENV": "NONE",
        },
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 2
    assert marker not in result.stderr
    assert "prohibited exported environment variable" in result.stderr


def test_q30t_tokenizer_runner_rejects_an_existing_function_namespace() -> None:
    """Sourcing from a privileged shell with existing functions fails before execution."""
    result = subprocess.run(
        [
            "/bin/bash",
            "-p",
            "-c",
            'compgen() { printf FUNCTION_MARKER >&2; }; source "$1"',
            "q30t-tokenizer-test",
            str(RUNNER_PATH),
        ],
        env={},
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 2
    assert "FUNCTION_MARKER" not in result.stderr
    assert "imported shell functions are prohibited" in result.stderr


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("BASH_ENV", "/dev/null"),
        ("LD_PRELOAD", "/nonexistent/q30t-tokenizer-loader.so"),
        ("PYTHONPATH", "/poisoned/inherited/path"),
        ("SSH_ASKPASS", "/nonexistent/injected-askpass"),
    ],
)
def test_q30t_tokenizer_runner_rejects_prohibited_exported_state(
    tmp_path: Path, name: str, value: str
) -> None:
    """The runtime boundary rejects loader, shell, Python, and Git helper injection."""
    repository, source_commit = _pushed_checkout(tmp_path)
    environment, calls = _tokenizer_runner_environment(tmp_path, repository, source_commit)
    environment[name] = value

    result = subprocess.run(
        _tokenizer_runner_command(environment),
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
def test_q30t_tokenizer_runner_rejects_any_gpu_allocation(
    tmp_path: Path, name: str, value: str
) -> None:
    """A CPU receipt cannot run when any Slurm GPU allocation field is populated."""
    repository, source_commit = _pushed_checkout(tmp_path)
    environment, calls = _tokenizer_runner_environment(tmp_path, repository, source_commit)
    environment[name] = value

    result = subprocess.run(
        _tokenizer_runner_command(environment),
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 2
    assert "CPU-only" in result.stderr
    assert not calls.exists()


def test_q30t_receipt_binds_snapshot_template_and_special_tokens(tmp_path: Path) -> None:
    """A receipt binds all locally recomputed tokenizer evidence."""
    snapshot = _make_q30_snapshot(tmp_path)

    receipt = build_q30t_tokenizer_receipt(
        snapshot=snapshot,
        repository=Q30_REPOSITORY,
        revision=FIXTURE_Q30_REVISION,
    )

    parsed = json.loads(receipt)
    assert parsed["schema_version"] == "qwen3-30ba3b-thinking-tokenizer-trust-v1"
    assert parsed["repository"] == Q30_REPOSITORY
    assert parsed["revision"] == FIXTURE_Q30_REVISION
    assert parsed["snapshot_path"] == str(snapshot)
    assert parsed["snapshot_tree_sha256"] == snapshot_tree_sha256(snapshot)
    assert parsed["chat_template_sha256"] == hashlib.sha256(OFFICIAL_TEMPLATE.encode()).hexdigest()
    assert (
        parsed["training_chat_template_sha256"]
        == hashlib.sha256(TRAINING_TEMPLATE.encode()).hexdigest()
    )
    assert parsed["im_start_token_id"] == 17
    assert parsed["im_end_token_id"] == 18
    assert verify_q30t_tokenizer_receipt(receipt) == parsed


def test_q30t_tokenizer_and_parent_bind_the_identical_target_tree_digest(tmp_path: Path) -> None:
    """One snapshot has one tree identity across tokenizer and parent receipts."""
    snapshot = _make_q30_snapshot(tmp_path)

    tokenizer_digest = snapshot_tree_sha256(snapshot)
    parent_digest = parent_tree_sha256(snapshot, "target model")

    assert tokenizer_digest == parent_digest


def test_q30t_absolute_walk_allows_intermediate_directory_metadata_changes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An unchanged ancestor inode remains valid when an unrelated child changes its ctime."""
    ancestor = tmp_path / "ancestor"
    ancestor.mkdir()
    snapshot = _make_q30_snapshot(ancestor)
    expected = tree_digest_module.os.stat(snapshot, follow_symlinks=False)
    original_open = tree_digest_module.os.open
    mutated = False

    def mutate_ancestor_before_open(path: object, *args: object, **kwargs: object) -> int:
        nonlocal mutated
        if path == ancestor.name and kwargs.get("dir_fd") is not None and not mutated:
            mutated = True
            (ancestor / "unrelated").write_bytes(b"unrelated\n")
        return original_open(path, *args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(tree_digest_module.os, "open", mutate_ancestor_before_open)

    tree_digest_module.require_stable_absolute_tree_root(snapshot, expected)
    assert mutated is True


def test_q30t_tree_digest_rebinds_the_absolute_root_after_walking(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A replacement installed at the absolute root during hashing is never authenticated."""
    snapshot = _make_q30_snapshot(tmp_path)
    displaced = tmp_path / "displaced-snapshot"
    replacement = tmp_path / "replacement-snapshot"
    replacement.mkdir()
    (replacement / "foreign").write_bytes(b"foreign\n")
    original_open = tree_digest_module.os.open
    original_close = tree_digest_module.os.close
    root_descriptors: set[int] = set()
    swapped = False

    def capture_root_open(path: object, *args: object, **kwargs: object) -> int:
        descriptor = original_open(path, *args, **kwargs)  # type: ignore[arg-type]
        if path == snapshot and kwargs.get("dir_fd") is None:
            root_descriptors.add(descriptor)
        return descriptor

    def swap_before_root_close(descriptor: int) -> None:
        nonlocal swapped
        if descriptor in root_descriptors and not swapped:
            swapped = True
            snapshot.rename(displaced)
            replacement.rename(snapshot)
        original_close(descriptor)

    monkeypatch.setattr(tree_digest_module.os, "open", capture_root_open)
    monkeypatch.setattr(tree_digest_module.os, "close", swap_before_root_close)

    with pytest.raises(ValueError, match=r"root changed|absolute root"):
        tree_digest_module.descriptor_stable_tree_entries(snapshot)


def test_q30t_tokenizer_walk_rebinds_the_absolute_root_after_traversal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The tokenizer-specific walker rejects a root replaced at its final close window."""
    snapshot = _make_q30_snapshot(tmp_path)
    displaced = tmp_path / "displaced-snapshot"
    replacement_root = tmp_path / "replacement"
    replacement_root.mkdir()
    replacement = _make_q30_snapshot(replacement_root)
    original_open = receipt_module.os.open
    original_close = receipt_module.os.close
    root_descriptors: set[int] = set()
    swapped = False

    def capture_root_open(path: object, *args: object, **kwargs: object) -> int:
        descriptor = original_open(path, *args, **kwargs)  # type: ignore[arg-type]
        if path == snapshot and kwargs.get("dir_fd") is None:
            root_descriptors.add(descriptor)
        return descriptor

    def swap_before_root_close(descriptor: int) -> None:
        nonlocal swapped
        if descriptor in root_descriptors and not swapped:
            swapped = True
            snapshot.rename(displaced)
            replacement.rename(snapshot)
        original_close(descriptor)

    monkeypatch.setattr(receipt_module.os, "open", capture_root_open)
    monkeypatch.setattr(receipt_module.os, "close", swap_before_root_close)

    with pytest.raises(ValueError, match=r"root changed|absolute root"):
        receipt_module._walk_snapshot(snapshot)


def test_q30t_tokenizer_cli_reconciles_exact_asset_identity_and_file_inventory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Receipt publication requires the reviewed model identity and every listed file."""
    snapshot = _make_q30_snapshot(tmp_path)
    identity, identity_sha256, model_sha256, model_manifest_sha256 = _write_model_asset_manifests(
        snapshot, tmp_path / "manifest"
    )
    monkeypatch.setattr(tree_digest_module, "APPROVED_Q30T_MODEL_ROOT", snapshot, raising=False)
    monkeypatch.setattr(
        tree_digest_module, "Q30T_MODEL_REVISION", FIXTURE_Q30_REVISION, raising=False
    )
    monkeypatch.setattr(
        tree_digest_module, "APPROVED_Q30T_MODEL_IDENTITY_SHA256", identity_sha256, raising=False
    )
    monkeypatch.setattr(
        tree_digest_module,
        "APPROVED_Q30T_MODEL_SHA256_MANIFEST_SHA256",
        model_manifest_sha256,
        raising=False,
    )
    output = tmp_path / "TOKENIZER.json"

    arguments = [
        "--repository",
        Q30_REPOSITORY,
        "--revision",
        FIXTURE_Q30_REVISION,
        "--snapshot",
        str(snapshot),
        "--identity-path",
        str(identity),
        "--identity-sha256",
        identity_sha256,
        "--model-sha256-path",
        str(model_sha256),
        "--model-sha256-sha256",
        model_manifest_sha256,
        "--output",
        str(output),
    ]
    try:
        result = receipt_module.main(arguments)
    except SystemExit:
        pytest.fail("Q30 tokenizer CLI does not accept the reviewed asset manifests")

    assert result == 0
    assert output.is_file()


@pytest.mark.parametrize("invalid_name", ["../escape", "assets/../escape"])
def test_q30t_tokenizer_cli_rejects_model_manifest_path_traversal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, invalid_name: str
) -> None:
    """The reviewed sha256sum grammar cannot name files outside the model root."""
    snapshot = _make_q30_snapshot(tmp_path)
    identity, _, model_sha256, _ = _write_model_asset_manifests(snapshot, tmp_path / "manifest")
    lines = model_sha256.read_text(encoding="utf-8").splitlines()
    digest = lines[0].split("  ", 1)[0]
    lines[0] = f"{digest}  ./{invalid_name}"
    model_sha256.write_text("\n".join(lines) + "\n", encoding="utf-8")
    model_manifest_sha256 = hashlib.sha256(model_sha256.read_bytes()).hexdigest()
    identity_payload = json.loads(identity.read_bytes())
    identity_payload["model_sha256_manifest"] = model_manifest_sha256
    identity.write_bytes(_canonical(identity_payload) + b"\n")
    identity_sha256 = hashlib.sha256(identity.read_bytes()).hexdigest()
    monkeypatch.setattr(tree_digest_module, "APPROVED_Q30T_MODEL_ROOT", snapshot)
    monkeypatch.setattr(tree_digest_module, "Q30T_MODEL_REVISION", FIXTURE_Q30_REVISION)
    monkeypatch.setattr(tree_digest_module, "APPROVED_Q30T_MODEL_IDENTITY_SHA256", identity_sha256)
    monkeypatch.setattr(
        tree_digest_module,
        "APPROVED_Q30T_MODEL_SHA256_MANIFEST_SHA256",
        model_manifest_sha256,
    )

    with pytest.raises(ValueError, match="sha256sum manifest path"):
        receipt_module.main(
            [
                "--repository",
                Q30_REPOSITORY,
                "--revision",
                FIXTURE_Q30_REVISION,
                "--snapshot",
                str(snapshot),
                "--identity-path",
                str(identity),
                "--identity-sha256",
                identity_sha256,
                "--model-sha256-path",
                str(model_sha256),
                "--model-sha256-sha256",
                model_manifest_sha256,
                "--output",
                str(tmp_path / "TOKENIZER.json"),
            ]
        )


def test_q30t_tokenizer_cli_rejects_duplicate_model_manifest_paths(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Every regular model file must occur exactly once in the sha256sum inventory."""
    snapshot = _make_q30_snapshot(tmp_path)
    identity, _, model_sha256, _ = _write_model_asset_manifests(snapshot, tmp_path / "manifest")
    lines = model_sha256.read_text(encoding="utf-8").splitlines()
    model_sha256.write_text("\n".join([*lines, lines[-1]]) + "\n", encoding="utf-8")
    model_manifest_sha256 = hashlib.sha256(model_sha256.read_bytes()).hexdigest()
    identity_payload = json.loads(identity.read_bytes())
    identity_payload["model_sha256_manifest"] = model_manifest_sha256
    identity.write_bytes(_canonical(identity_payload) + b"\n")
    identity_sha256 = hashlib.sha256(identity.read_bytes()).hexdigest()
    monkeypatch.setattr(tree_digest_module, "APPROVED_Q30T_MODEL_ROOT", snapshot)
    monkeypatch.setattr(tree_digest_module, "Q30T_MODEL_REVISION", FIXTURE_Q30_REVISION)
    monkeypatch.setattr(tree_digest_module, "APPROVED_Q30T_MODEL_IDENTITY_SHA256", identity_sha256)
    monkeypatch.setattr(
        tree_digest_module,
        "APPROVED_Q30T_MODEL_SHA256_MANIFEST_SHA256",
        model_manifest_sha256,
    )

    with pytest.raises(ValueError, match="sha256sum manifest path"):
        receipt_module.main(
            [
                "--repository",
                Q30_REPOSITORY,
                "--revision",
                FIXTURE_Q30_REVISION,
                "--snapshot",
                str(snapshot),
                "--identity-path",
                str(identity),
                "--identity-sha256",
                identity_sha256,
                "--model-sha256-path",
                str(model_sha256),
                "--model-sha256-sha256",
                model_manifest_sha256,
                "--output",
                str(tmp_path / "TOKENIZER.json"),
            ]
        )


def test_q30t_receipt_cli_builds_and_verifies_twice_before_no_clobber_publish(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The CLI reconciles two independent reads and never replaces an immutable receipt."""
    snapshot = _make_q30_snapshot(tmp_path)
    identity, identity_sha256, model_sha256, model_manifest_sha256 = _write_model_asset_manifests(
        snapshot, tmp_path / "manifest"
    )
    monkeypatch.setattr(tree_digest_module, "APPROVED_Q30T_MODEL_ROOT", snapshot)
    monkeypatch.setattr(tree_digest_module, "Q30T_MODEL_REVISION", FIXTURE_Q30_REVISION)
    monkeypatch.setattr(tree_digest_module, "APPROVED_Q30T_MODEL_IDENTITY_SHA256", identity_sha256)
    monkeypatch.setattr(
        tree_digest_module,
        "APPROVED_Q30T_MODEL_SHA256_MANIFEST_SHA256",
        model_manifest_sha256,
    )
    output = tmp_path / "receipts" / "TOKENIZER.json"
    original_build = receipt_module.build_q30t_tokenizer_receipt
    original_verify = receipt_module.verify_q30t_tokenizer_receipt
    build_calls = 0
    verify_calls = 0

    def counted_build(snapshot: Path, repository: str, revision: str) -> bytes:
        nonlocal build_calls
        build_calls += 1
        return original_build(snapshot, repository, revision)

    def counted_verify(receipt: bytes) -> dict[str, object]:
        nonlocal verify_calls
        verify_calls += 1
        return original_verify(receipt)

    monkeypatch.setattr(receipt_module, "build_q30t_tokenizer_receipt", counted_build)
    monkeypatch.setattr(receipt_module, "verify_q30t_tokenizer_receipt", counted_verify)

    assert (
        receipt_module.main(
            [
                "--repository",
                Q30_REPOSITORY,
                "--revision",
                FIXTURE_Q30_REVISION,
                "--snapshot",
                str(snapshot),
                "--identity-path",
                str(identity),
                "--identity-sha256",
                identity_sha256,
                "--model-sha256-path",
                str(model_sha256),
                "--model-sha256-sha256",
                model_manifest_sha256,
                "--output",
                str(output),
            ]
        )
        == 0
    )
    published = output.read_bytes()
    assert build_calls == 2
    assert verify_calls == 2
    assert published == original_build(snapshot, Q30_REPOSITORY, FIXTURE_Q30_REVISION)

    assert (
        receipt_module.main(
            [
                "--repository",
                Q30_REPOSITORY,
                "--revision",
                FIXTURE_Q30_REVISION,
                "--snapshot",
                str(snapshot),
                "--identity-path",
                str(identity),
                "--identity-sha256",
                identity_sha256,
                "--model-sha256-path",
                str(model_sha256),
                "--model-sha256-sha256",
                model_manifest_sha256,
                "--output",
                str(output),
            ]
        )
        == 0
    )
    assert output.read_bytes() == published
    assert build_calls == 4
    assert verify_calls == 4


def test_q30t_tokenizer_cli_rejects_a_stable_snapshot_swapped_after_manifest_review(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Both generated receipts must remain bound to the manifest-reviewed tree."""
    snapshot = _make_q30_snapshot(tmp_path)
    identity, identity_sha256, model_sha256, model_manifest_sha256 = _write_model_asset_manifests(
        snapshot, tmp_path / "manifest"
    )
    monkeypatch.setattr(tree_digest_module, "APPROVED_Q30T_MODEL_ROOT", snapshot)
    monkeypatch.setattr(tree_digest_module, "Q30T_MODEL_REVISION", FIXTURE_Q30_REVISION)
    monkeypatch.setattr(tree_digest_module, "APPROVED_Q30T_MODEL_IDENTITY_SHA256", identity_sha256)
    monkeypatch.setattr(
        tree_digest_module,
        "APPROVED_Q30T_MODEL_SHA256_MANIFEST_SHA256",
        model_manifest_sha256,
    )
    original_reconcile = receipt_module.reconcile_q30t_model_asset
    reviewed_tree_sha256: list[str] = []

    def reconcile_then_swap(**kwargs) -> str:
        digest = original_reconcile(**kwargs)
        reviewed_tree_sha256.append(digest)
        (snapshot / "assets/tokenizer.model").write_bytes(b"stable but unreviewed replacement")
        return digest

    monkeypatch.setattr(receipt_module, "reconcile_q30t_model_asset", reconcile_then_swap)
    output = tmp_path / "TOKENIZER.json"

    with pytest.raises(ValueError, match="reviewed model tree"):
        receipt_module.main(
            [
                "--repository",
                Q30_REPOSITORY,
                "--revision",
                FIXTURE_Q30_REVISION,
                "--snapshot",
                str(snapshot),
                "--identity-path",
                str(identity),
                "--identity-sha256",
                identity_sha256,
                "--model-sha256-path",
                str(model_sha256),
                "--model-sha256-sha256",
                model_manifest_sha256,
                "--output",
                str(output),
            ]
        )

    assert reviewed_tree_sha256
    assert snapshot_tree_sha256(snapshot) != reviewed_tree_sha256[0]
    assert not output.exists()


def test_q30t_tokenizer_cli_adopts_an_exact_ambiguous_publish_result(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A rename ambiguity is successful only when durable bytes authenticate exactly."""
    snapshot = _make_q30_snapshot(tmp_path)
    identity, identity_sha256, model_sha256, model_manifest_sha256 = _write_model_asset_manifests(
        snapshot, tmp_path / "manifest"
    )
    monkeypatch.setattr(tree_digest_module, "APPROVED_Q30T_MODEL_ROOT", snapshot)
    monkeypatch.setattr(tree_digest_module, "Q30T_MODEL_REVISION", FIXTURE_Q30_REVISION)
    monkeypatch.setattr(tree_digest_module, "APPROVED_Q30T_MODEL_IDENTITY_SHA256", identity_sha256)
    monkeypatch.setattr(
        tree_digest_module,
        "APPROVED_Q30T_MODEL_SHA256_MANIFEST_SHA256",
        model_manifest_sha256,
    )
    output = tmp_path / "TOKENIZER.json"

    def ambiguous_publish(destination: Path, payload: bytes, *, job_id: str) -> None:
        del job_id
        destination.write_bytes(payload)
        raise FileExistsError("ambiguous rename")

    monkeypatch.setattr(receipt_module, "atomic_publish_bytes", ambiguous_publish)

    assert (
        receipt_module.main(
            [
                "--repository",
                Q30_REPOSITORY,
                "--revision",
                FIXTURE_Q30_REVISION,
                "--snapshot",
                str(snapshot),
                "--identity-path",
                str(identity),
                "--identity-sha256",
                identity_sha256,
                "--model-sha256-path",
                str(model_sha256),
                "--model-sha256-sha256",
                model_manifest_sha256,
                "--output",
                str(output),
            ]
        )
        == 0
    )


def test_q30t_tokenizer_successful_publish_is_reauthenticated(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A nominal publisher success cannot bypass durable exact-byte adoption."""
    output = tmp_path / "TOKENIZER.json"
    expected = b'{"receipt":"expected"}\n'

    def successful_foreign_publish(destination: Path, payload: bytes, *, job_id: str) -> None:
        del payload, job_id
        destination.write_bytes(b"foreign\n")

    monkeypatch.setattr(receipt_module, "atomic_publish_bytes", successful_foreign_publish)

    with pytest.raises(FileExistsError, match="differs"):
        receipt_module._publish_or_adopt_receipt(output, expected, job_id="4242")


def test_q30t_tokenizer_cli_never_suppresses_failed_adoption_fsync(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Matching bytes cannot turn an unrepaired publisher fsync error into success."""
    snapshot = _make_q30_snapshot(tmp_path)
    identity, identity_sha256, model_sha256, model_manifest_sha256 = _write_model_asset_manifests(
        snapshot, tmp_path / "manifest"
    )
    monkeypatch.setattr(tree_digest_module, "APPROVED_Q30T_MODEL_ROOT", snapshot)
    monkeypatch.setattr(tree_digest_module, "Q30T_MODEL_REVISION", FIXTURE_Q30_REVISION)
    monkeypatch.setattr(tree_digest_module, "APPROVED_Q30T_MODEL_IDENTITY_SHA256", identity_sha256)
    monkeypatch.setattr(
        tree_digest_module,
        "APPROVED_Q30T_MODEL_SHA256_MANIFEST_SHA256",
        model_manifest_sha256,
    )
    output = tmp_path / "TOKENIZER.json"
    fail_adoption_fsync = False
    original_fsync = os.fsync

    def ambiguous_publish(destination: Path, payload: bytes, *, job_id: str) -> None:
        nonlocal fail_adoption_fsync
        del job_id
        destination.write_bytes(payload)
        fail_adoption_fsync = True
        raise OSError("injected publisher fsync failure")

    def injected_fsync(descriptor: int) -> None:
        if fail_adoption_fsync:
            raise OSError("injected adoption fsync failure")
        original_fsync(descriptor)

    monkeypatch.setattr(receipt_module, "atomic_publish_bytes", ambiguous_publish)
    monkeypatch.setattr(receipt_module.os, "fsync", injected_fsync)

    with pytest.raises(OSError, match="adoption fsync"):
        receipt_module.main(
            [
                "--repository",
                Q30_REPOSITORY,
                "--revision",
                FIXTURE_Q30_REVISION,
                "--snapshot",
                str(snapshot),
                "--identity-path",
                str(identity),
                "--identity-sha256",
                identity_sha256,
                "--model-sha256-path",
                str(model_sha256),
                "--model-sha256-sha256",
                model_manifest_sha256,
                "--output",
                str(output),
            ]
        )


def test_q30t_tokenizer_retry_reauthenticates_after_parent_fsync(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A same-byte receipt replacement after parent fsync cannot be adopted."""
    output = tmp_path / "TOKENIZER.json"
    expected = b'{"receipt":"exact"}\n'
    output.write_bytes(expected)
    replacement = tmp_path / "replacement.json"
    replacement.write_bytes(expected)
    displaced = tmp_path / "displaced.json"
    original_open = receipt_module.os.open
    parent_fd = original_open(tmp_path, os.O_RDONLY)
    original_fsync = receipt_module.os.fsync
    swapped = False

    def fsync_then_swap(descriptor: int) -> None:
        nonlocal swapped
        original_fsync(descriptor)
        if descriptor == parent_fd and not swapped:
            swapped = True
            output.rename(displaced)
            replacement.rename(output)

    def inject_parent_descriptor(*args: object, **kwargs: object) -> int:
        if args[0] == tmp_path:
            return parent_fd
        return original_open(*args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(receipt_module.os, "open", inject_parent_descriptor)
    monkeypatch.setattr(receipt_module.os, "fsync", fsync_then_swap)

    with pytest.raises(FileExistsError, match=r"changed|rebound"):
        receipt_module._adopt_exact_receipt(output, expected)


def test_q30t_tokenizer_cli_never_adopts_mismatched_existing_bytes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Completed retry adoption cannot bless a conflicting receipt pathname."""
    snapshot = _make_q30_snapshot(tmp_path)
    identity, identity_sha256, model_sha256, model_manifest_sha256 = _write_model_asset_manifests(
        snapshot, tmp_path / "manifest"
    )
    monkeypatch.setattr(tree_digest_module, "APPROVED_Q30T_MODEL_ROOT", snapshot)
    monkeypatch.setattr(tree_digest_module, "Q30T_MODEL_REVISION", FIXTURE_Q30_REVISION)
    monkeypatch.setattr(tree_digest_module, "APPROVED_Q30T_MODEL_IDENTITY_SHA256", identity_sha256)
    monkeypatch.setattr(
        tree_digest_module,
        "APPROVED_Q30T_MODEL_SHA256_MANIFEST_SHA256",
        model_manifest_sha256,
    )
    output = tmp_path / "TOKENIZER.json"
    output.write_bytes(b"foreign\n")

    with pytest.raises(FileExistsError, match="differs"):
        receipt_module.main(
            [
                "--repository",
                Q30_REPOSITORY,
                "--revision",
                FIXTURE_Q30_REVISION,
                "--snapshot",
                str(snapshot),
                "--identity-path",
                str(identity),
                "--identity-sha256",
                identity_sha256,
                "--model-sha256-path",
                str(model_sha256),
                "--model-sha256-sha256",
                model_manifest_sha256,
                "--output",
                str(output),
            ]
        )
    assert output.read_bytes() == b"foreign\n"


def test_q30t_tokenizer_runner_is_one_node_ptyche_cpu_only_and_checks_identities(
    tmp_path: Path,
) -> None:
    """A clean pushed checkout passes manifests to the isolated compute-side CLI."""
    subprocess.run(["bash", "-n", RUNNER_PATH], check=True)
    runner = RUNNER_PATH.read_text(encoding="utf-8")
    assert "#SBATCH --partition=36x2-a01r" in runner
    assert "#SBATCH --nodes=1" in runner
    assert "#SBATCH --gpus" not in runner
    assert "#SBATCH --gres" not in runner
    assert runner.startswith("#!/bin/bash -p\n")
    assert " -S -I -c " in runner
    assert 'readonly approved_env="/usr/bin/env"' in runner
    assert "stat.S_ISLNK(entry.st_mode)" in runner
    assert "entry_permissions=stat.S_ISLNK(entry.st_mode) or not entry.st_mode & 0o022" in runner
    repository, source_commit = _pushed_checkout(tmp_path)
    environment, calls = _tokenizer_runner_environment(tmp_path, repository, source_commit)

    result = subprocess.run(
        _tokenizer_runner_command(environment),
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    invocation = calls.read_text().splitlines()
    assert invocation[7:] == [
        "--repository",
        Q30_REPOSITORY,
        "--revision",
        FIXTURE_Q30_REVISION,
        "--snapshot",
        environment["TOKENIZER_SNAPSHOT"],
        "--identity-path",
        environment["IDENTITY_PATH"],
        "--identity-sha256",
        environment["IDENTITY_SHA256"],
        "--model-sha256-path",
        environment["MODEL_SHA256_PATH"],
        "--model-sha256-sha256",
        environment["MODEL_SHA256_SHA256"],
        "--output",
        environment["OUTPUT_PATH"],
    ]


def test_q30t_tokenizer_runner_rejects_a_local_source_ref(tmp_path: Path) -> None:
    """A local branch cannot impersonate the required pushed remote-tracking ref."""
    repository, source_commit = _pushed_checkout(tmp_path)
    environment, calls = _tokenizer_runner_environment(tmp_path, repository, source_commit)
    environment["SOURCE_REMOTE_REF"] = "main"

    result = subprocess.run(
        _tokenizer_runner_command(environment),
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 2
    assert "remote-tracking" in result.stderr
    assert not calls.exists()


def test_q30t_tokenizer_runner_rejects_a_forged_remote_tracking_ref(tmp_path: Path) -> None:
    """A locally written refs/remotes entry is not evidence that the commit was pushed."""
    repository, _ = _pushed_checkout(tmp_path)
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
    source_commit = _git(["rev-parse", "HEAD"], cwd=repository)
    _git(["update-ref", "refs/remotes/origin/main", source_commit], cwd=repository)
    environment, calls = _tokenizer_runner_environment(tmp_path, repository, source_commit)

    result = subprocess.run(
        _tokenizer_runner_command(environment),
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 2
    assert "pushed branch" in result.stderr
    assert not calls.exists()


def test_q30t_tokenizer_runner_defers_completed_retry_authentication_to_cli(
    tmp_path: Path,
) -> None:
    """The runner must not preempt the CLI's exact-byte adoption decision."""
    repository, source_commit = _pushed_checkout(tmp_path)
    environment, calls = _tokenizer_runner_environment(tmp_path, repository, source_commit)
    Path(environment["OUTPUT_PATH"]).write_bytes(b"completed receipt\n")

    result = subprocess.run(
        _tokenizer_runner_command(environment),
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert calls.exists()


def test_q30t_receipt_builder_rejects_relative_snapshot_identity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A builder receipt must never emit a path its verifier cannot authenticate."""
    snapshot = _make_q30_snapshot(tmp_path)
    monkeypatch.chdir(tmp_path)

    with pytest.raises(ValueError, match="absolute"):
        build_q30t_tokenizer_receipt(
            snapshot=snapshot.relative_to(tmp_path),
            repository=Q30_REPOSITORY,
            revision=FIXTURE_Q30_REVISION,
        )


def test_q30t_snapshot_reader_closes_descriptor_when_fdopen_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Ownership-transfer failure must not leak the opened snapshot descriptor."""
    snapshot = _make_q30_snapshot(tmp_path)
    directory_fd = os.open(snapshot, os.O_RDONLY)
    expected = os.stat("tokenizer.json", dir_fd=directory_fd, follow_symlinks=False)
    real_open = os.open
    captured: list[int] = []

    def capture_open(*args: object, **kwargs: object) -> int:
        descriptor = real_open(*args, **kwargs)  # type: ignore[arg-type]
        captured.append(descriptor)
        return descriptor

    def fail_fdopen(*_args: object, **_kwargs: object) -> None:
        raise ValueError("injected wrapper failure")

    monkeypatch.setattr(receipt_module.os, "open", capture_open)
    monkeypatch.setattr(receipt_module.os, "fdopen", fail_fdopen)
    try:
        with pytest.raises(ValueError, match="injected wrapper failure"):
            receipt_module._read_regular_at(
                directory_fd,
                "tokenizer.json",
                str(snapshot / "tokenizer.json"),
                expected,
                True,
            )
        assert captured
        with pytest.raises(OSError):
            os.fstat(captured[-1])
    finally:
        os.close(directory_fd)


@pytest.mark.parametrize("field", ["snapshot_tree_sha256", "training_chat_template_sha256"])
def test_q30t_receipt_rejects_tampering(tmp_path: Path, field: str) -> None:
    """Rehashed derived receipt fields still fail snapshot reconciliation."""
    payload = json.loads(
        build_q30t_tokenizer_receipt(
            snapshot=_make_q30_snapshot(tmp_path),
            repository=Q30_REPOSITORY,
            revision=FIXTURE_Q30_REVISION,
        )
    )
    payload[field] = "b" * 64

    with pytest.raises(ValueError, match="evidence"):
        verify_q30t_tokenizer_receipt(_rehashed(payload))


def test_q30t_receipt_rejects_missing_canonical_newline(tmp_path: Path) -> None:
    """Canonical receipt bytes include the required terminating newline."""
    receipt = build_q30t_tokenizer_receipt(
        snapshot=_make_q30_snapshot(tmp_path),
        repository=Q30_REPOSITORY,
        revision=FIXTURE_Q30_REVISION,
    )

    with pytest.raises(ValueError, match="identity"):
        verify_q30t_tokenizer_receipt(receipt.rstrip(b"\n"))


def test_q30t_approved_loader_hashes_membership_and_replays(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The approved loader binds caller bytes and replays snapshot evidence."""
    snapshot = _make_q30_snapshot(tmp_path)
    raw = build_q30t_tokenizer_receipt(snapshot, Q30_REPOSITORY, FIXTURE_Q30_REVISION)
    receipt = tmp_path / "TOKENIZER.json"
    receipt.write_bytes(raw)
    receipt_sha256 = hashlib.sha256(raw).hexdigest()
    monkeypatch.setattr(
        receipt_module,
        "APPROVED_Q30T_TOKENIZER_RECEIPT_FILE_SHA256S",
        frozenset({receipt_sha256}),
    )

    loaded = receipt_module.load_q30t_tokenizer_receipt(receipt, expected_sha256=receipt_sha256)

    assert loaded["revision"] == FIXTURE_Q30_REVISION
    with pytest.raises(ValueError, match="caller SHA-256 mismatch"):
        receipt_module.load_q30t_tokenizer_receipt(receipt, expected_sha256="0" * 64)


def test_q30t_approved_loader_rejects_unreviewed_valid_receipt(tmp_path: Path) -> None:
    """A valid receipt cannot self-authorize through its caller-provided hash."""
    raw = build_q30t_tokenizer_receipt(
        _make_q30_snapshot(tmp_path), Q30_REPOSITORY, FIXTURE_Q30_REVISION
    )
    receipt = tmp_path / "TOKENIZER.json"
    receipt.write_bytes(raw)

    with pytest.raises(ValueError, match="not independently reviewed"):
        receipt_module.load_q30t_tokenizer_receipt(
            receipt, expected_sha256=hashlib.sha256(raw).hexdigest()
        )


def test_q30t_approved_loader_rejects_hardlinked_receipt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An extra hardlink invalidates the approved receipt's path identity."""
    raw = build_q30t_tokenizer_receipt(
        _make_q30_snapshot(tmp_path), Q30_REPOSITORY, FIXTURE_Q30_REVISION
    )
    receipt = tmp_path / "TOKENIZER.json"
    receipt.write_bytes(raw)
    hardlink = tmp_path / "TOKENIZER-hardlink.json"
    os.link(receipt, hardlink)
    receipt_sha256 = hashlib.sha256(raw).hexdigest()
    monkeypatch.setattr(
        receipt_module,
        "APPROVED_Q30T_TOKENIZER_RECEIPT_FILE_SHA256S",
        frozenset({receipt_sha256}),
    )

    with pytest.raises(ValueError, match="single-link regular file"):
        receipt_module.load_q30t_tokenizer_receipt(hardlink, expected_sha256=receipt_sha256)


def test_q30t_approved_loader_rebinds_name_after_read(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A pathname replacement after descriptor reading cannot be accepted."""
    raw = build_q30t_tokenizer_receipt(
        _make_q30_snapshot(tmp_path), Q30_REPOSITORY, FIXTURE_Q30_REVISION
    )
    receipt = tmp_path / "TOKENIZER.json"
    receipt.write_bytes(raw)
    replacement = tmp_path / "TOKENIZER-replacement.json"
    replacement.write_bytes(raw)
    receipt_sha256 = hashlib.sha256(raw).hexdigest()
    monkeypatch.setattr(
        receipt_module,
        "APPROVED_Q30T_TOKENIZER_RECEIPT_FILE_SHA256S",
        frozenset({receipt_sha256}),
    )
    real_read = os.read
    rebound = False

    def replace_after_read(descriptor: int, size: int) -> bytes:
        nonlocal rebound
        block = real_read(descriptor, size)
        if not block and not rebound:
            os.replace(replacement, receipt)
            rebound = True
        return block

    monkeypatch.setattr(receipt_module.os, "read", replace_after_read)

    with pytest.raises(ValueError, match="Q30 tokenizer receipt changed while reading"):
        receipt_module.load_q30t_tokenizer_receipt(receipt, expected_sha256=receipt_sha256)


def test_q30t_builder_recomputes_tokenizer_receipt_evidence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The shared builder accepts Q30 special-token IDs only after recomputation."""
    builder = _load_builder()
    snapshot = _make_q30_snapshot(tmp_path)
    receipt = build_q30t_tokenizer_receipt(
        snapshot=snapshot,
        repository=Q30_REPOSITORY,
        revision=FIXTURE_Q30_REVISION,
    )
    receipt_path = tmp_path / "receipt.json"
    receipt_path.write_bytes(receipt)
    monkeypatch.setattr(
        builder,
        "APPROVED_Q30T_TOKENIZER_RECEIPT_FILE_SHA256S",
        frozenset({hashlib.sha256(receipt).hexdigest()}),
    )
    policy = builder.TargetTokenizerPolicy(
        schema_version="ptv23-complement-target-policy-v1",
        scientific_identity="fixture-q30",
        target_repository=Q30_REPOSITORY,
        target_revision=FIXTURE_Q30_REVISION,
        tokenizer_repository=Q30_REPOSITORY,
        tokenizer_revision=FIXTURE_Q30_REVISION,
        tokenizer_trust_schema="qwen3-30ba3b-thinking-tokenizer-trust-v1",
        training_sequence_length=4096,
        quota_config_path="quota.json",
        quota_config_sha256="1" * 64,
        source_requirements_path="sources.json",
        source_requirements_sha256="2" * 64,
        file_sha256="3" * 64,
    )

    trust = builder.load_tokenizer_trust(
        receipt_path,
        expected_sha256=hashlib.sha256(receipt).hexdigest(),
        policy=policy,
    )

    assert (trust.im_start_token_id, trust.im_end_token_id) == (17, 18)


def test_q30t_builder_fails_closed_without_a_reviewed_receipt_root(tmp_path: Path) -> None:
    """A caller-provided receipt SHA cannot authorize an otherwise valid Q30 receipt."""
    builder = _load_builder()
    snapshot = _make_q30_snapshot(tmp_path)
    receipt = build_q30t_tokenizer_receipt(
        snapshot=snapshot,
        repository=Q30_REPOSITORY,
        revision=FIXTURE_Q30_REVISION,
    )
    receipt_path = tmp_path / "receipt.json"
    receipt_path.write_bytes(receipt)
    policy = builder.TargetTokenizerPolicy(
        schema_version="ptv23-complement-target-policy-v1",
        scientific_identity="fixture-q30",
        target_repository=Q30_REPOSITORY,
        target_revision=FIXTURE_Q30_REVISION,
        tokenizer_repository=Q30_REPOSITORY,
        tokenizer_revision=FIXTURE_Q30_REVISION,
        tokenizer_trust_schema="qwen3-30ba3b-thinking-tokenizer-trust-v1",
        training_sequence_length=4096,
        quota_config_path="quota.json",
        quota_config_sha256="1" * 64,
        source_requirements_path="sources.json",
        source_requirements_sha256="2" * 64,
        file_sha256="3" * 64,
    )

    with pytest.raises(builder.ComplementError, match="reviewed Q30 tokenizer receipt"):
        builder.load_tokenizer_trust(
            receipt_path,
            expected_sha256=hashlib.sha256(receipt).hexdigest(),
            policy=policy,
        )


def test_q30t_receipt_rejects_hardlinked_snapshot_files(tmp_path: Path) -> None:
    """A closed tokenizer snapshot cannot admit a multiply-linked regular file."""
    snapshot = _make_q30_snapshot(tmp_path)
    os.link(snapshot / "tokenizer.json", snapshot / "duplicate-tokenizer.json")

    with pytest.raises(ValueError, match="hardlinks"):
        build_q30t_tokenizer_receipt(
            snapshot=snapshot,
            repository=Q30_REPOSITORY,
            revision=FIXTURE_Q30_REVISION,
        )


def test_q30t_builder_uses_an_imported_receipt_verifier() -> None:
    """Q30 verification cannot dynamically execute a mutable source pathname."""
    source = BUILDER_PATH.read_text(encoding="utf-8")

    assert "exec_module" not in source


def test_q30t_receipt_rejects_boolean_special_token_ids(tmp_path: Path) -> None:
    """Boolean JSON values cannot alias valid integer tokenizer IDs."""
    payload = json.loads(
        build_q30t_tokenizer_receipt(
            snapshot=_make_q30_snapshot(tmp_path, im_start_token_id=1),
            repository=Q30_REPOSITORY,
            revision=FIXTURE_Q30_REVISION,
        )
    )
    payload["im_start_token_id"] = True

    with pytest.raises(ValueError, match="identity"):
        verify_q30t_tokenizer_receipt(_rehashed(payload))


def test_q30t_snapshot_uses_nonblocking_nofollow_regular_opens(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A regular-file path replacement cannot turn snapshot hashing into a FIFO wait."""
    snapshot = _make_q30_snapshot(tmp_path)
    observed_flags: list[int] = []
    original_open = receipt_module.os.open

    def checked_open(path, flags, *args, **kwargs):
        if path == "tokenizer.json":
            observed_flags.append(flags)
        return original_open(path, flags, *args, **kwargs)

    monkeypatch.setattr(receipt_module.os, "open", checked_open)
    snapshot_tree_sha256(snapshot)

    assert observed_flags
    assert all(flags & os.O_NONBLOCK for flags in observed_flags)
    assert all(flags & os.O_NOFOLLOW for flags in observed_flags)


def test_q30t_snapshot_reconciles_pre_stat_with_opened_fd(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Changing a name after pre-stat cannot substitute a different opened file."""
    snapshot = _make_q30_snapshot(tmp_path)
    directory_fd = os.open(snapshot, os.O_RDONLY)
    expected = os.stat("tokenizer.json", dir_fd=directory_fd, follow_symlinks=False)
    original_open = receipt_module.os.open

    def substituted_open(path, flags, *args, **kwargs):
        if path == "tokenizer.json":
            return original_open("assets/tokenizer.model", flags, *args, **kwargs)
        return original_open(path, flags, *args, **kwargs)

    monkeypatch.setattr(receipt_module.os, "open", substituted_open)
    try:
        with pytest.raises(ValueError, match="not regular"):
            receipt_module._read_regular_at(
                directory_fd, "tokenizer.json", "tokenizer.json", expected, False
            )
    finally:
        os.close(directory_fd)


def test_q30t_snapshot_closes_fd_when_fdopen_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A stream-wrapper failure must not leak the already opened snapshot descriptor."""
    snapshot = _make_q30_snapshot(tmp_path)
    directory_fd = os.open(snapshot, os.O_RDONLY)
    expected = os.stat("tokenizer.json", dir_fd=directory_fd, follow_symlinks=False)
    opened: list[int] = []
    closed: list[int] = []
    original_open = receipt_module.os.open
    original_close = receipt_module.os.close

    def recorded_open(path, flags, *args, **kwargs):
        descriptor = original_open(path, flags, *args, **kwargs)
        if path == "tokenizer.json":
            opened.append(descriptor)
        return descriptor

    def recorded_close(descriptor: int) -> None:
        closed.append(descriptor)
        original_close(descriptor)

    monkeypatch.setattr(receipt_module.os, "open", recorded_open)
    monkeypatch.setattr(receipt_module.os, "close", recorded_close)
    monkeypatch.setattr(
        receipt_module.os,
        "fdopen",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(OSError("boom")),
    )
    try:
        with pytest.raises(ValueError, match="cannot be read"):
            receipt_module._read_regular_at(
                directory_fd, "tokenizer.json", "tokenizer.json", expected, False
            )
    finally:
        original_close(directory_fd)

    assert opened == closed


def test_q30t_snapshot_streams_unneeded_file_bytes(tmp_path: Path) -> None:
    """Large unrelated snapshot files do not remain resident while computing the tree."""
    snapshot = _make_q30_snapshot(tmp_path)
    (snapshot / "assets" / "large.bin").write_bytes(b"x" * (8 * 1024 * 1024))

    tracemalloc.start()
    try:
        snapshot_tree_sha256(snapshot)
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()

    assert peak < 4 * 1024 * 1024


def test_q30t_builder_streams_the_full_model_tree_with_bounded_memory(tmp_path: Path) -> None:
    """Q30 trust verification never retains a model shard in process memory."""
    builder = _load_builder()
    snapshot = _make_q30_snapshot(tmp_path)
    (snapshot / "model.safetensors").write_bytes(b"x" * (8 * 1024 * 1024))

    tracemalloc.start()
    try:
        builder._q30t_snapshot_tree_sha256(snapshot)
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()

    assert peak < 4 * 1024 * 1024


def test_q30t_builder_rejects_fifo_entries_without_blocking(tmp_path: Path) -> None:
    """An untrusted special file cannot be ignored or opened as a blocking input."""
    builder = _load_builder()
    snapshot = _make_q30_snapshot(tmp_path)
    os.mkfifo(snapshot / "model.fifo")

    with pytest.raises(builder.ComplementError, match="unsupported"):
        builder._q30t_snapshot_tree_sha256(snapshot)


def test_q30t_builder_reconciles_named_entry_with_opened_descriptor(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A pathname swap between stat and open cannot substitute different bytes."""
    builder = _load_builder()
    snapshot = _make_q30_snapshot(tmp_path)
    (snapshot / "replacement.json").write_bytes(b"replacement")
    original_open = os.open

    def substituted_open(path, flags, *args, **kwargs):
        if os.fspath(path).endswith("tokenizer.json"):
            if kwargs.get("dir_fd") is not None:
                return original_open("replacement.json", flags, *args, **kwargs)
            return original_open(snapshot / "replacement.json", flags, *args, **kwargs)
        return original_open(path, flags, *args, **kwargs)

    monkeypatch.setattr(tree_digest_module.os, "open", substituted_open)

    with pytest.raises(builder.ComplementError, match=r"changed|identity"):
        builder._q30t_snapshot_tree_sha256(snapshot)


def test_q30t_builder_stages_only_tokenizer_assets_without_retaining_weights(
    tmp_path: Path,
) -> None:
    """Staging streams the authenticated tree but copies no model-weight payloads."""
    builder = _load_builder()
    snapshot = _make_q30_snapshot(tmp_path)
    (snapshot / "config.json").write_bytes(b"{}\n")
    (snapshot / "model.safetensors").write_bytes(b"x" * (8 * 1024 * 1024))
    trust = builder.TokenizerTrust(
        file_sha256="a" * 64,
        receipt_sha256="b" * 64,
        repository=Q30_REPOSITORY,
        revision=FIXTURE_Q30_REVISION,
        snapshot_path=snapshot,
        snapshot_tree_sha256=snapshot_tree_sha256(snapshot),
        chat_template_sha256="c" * 64,
        training_chat_template_sha256="d" * 64,
        im_start_token_id=17,
        im_end_token_id=18,
    )

    tracemalloc.start()
    try:
        staged = builder._stage_tokenizer_snapshot(trust, tmp_path / "scratch")
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()

    assert (staged / "tokenizer.json").is_file()
    assert (staged / "tokenizer_config.json").is_file()
    assert (staged / "assets/tokenizer.model").is_file()
    assert not (staged / "model.safetensors").exists()
    assert peak < 4 * 1024 * 1024
