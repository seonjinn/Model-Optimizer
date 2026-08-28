# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# ruff: noqa: D101, D102, D103

"""Q30 Ptyche runtime-profile trust-boundary tests."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
from dataclasses import dataclass
from typing import TYPE_CHECKING

import pytest
from common.specdec import q30t_ptv23_cluster_profile as profile_module
from common.specdec.q30t_ptv23_cluster_profile import (
    finalize_ptyche_runtime_profile,
    load_ptyche_runtime_profile,
)

if TYPE_CHECKING:
    from pathlib import Path

_TOOL_CONTENTS = {
    "tools/launcher/common/specdec/q30t_runtime_archive_receipt.py": b"archive producer\n",
    "tools/launcher/common/specdec/ptv23_node_keeper.py": b"keeper tool\n",
    "tools/launcher/common/specdec/ptv23_runtime_attestation.py": b"attestation tool\n",
    "tools/launcher/common/specdec/probe_ptv23_pyxis_keeper.sbatch": b"probe runner\n",
}


@dataclass(frozen=True)
class ProfileFixture:
    source_checkout: Path
    source_commit: str
    durable_receipt_root: Path
    output_path: Path

    def arguments(self) -> dict[str, object]:
        return {
            "source_checkout": self.source_checkout,
            "source_commit": self.source_commit,
            "durable_receipt_root": self.durable_receipt_root,
            "output_path": self.output_path,
            "job_id": "profile-unit-1",
        }


def _git(repository: Path, *arguments: str) -> str:
    result = subprocess.run(
        ("/usr/bin/git", "-c", "commit.gpgsign=false", "-C", str(repository), *arguments),
        check=True,
        capture_output=True,
        text=True,
        env={
            "GIT_CONFIG_GLOBAL": "/dev/null",
            "GIT_CONFIG_NOSYSTEM": "1",
            "GIT_NO_REPLACE_OBJECTS": "1",
            "LC_ALL": "C",
            "PATH": "/usr/bin:/bin",
        },
    )
    return result.stdout.strip()


def _commit_and_push(fixture: ProfileFixture, message: str, *, allow_empty: bool = False) -> str:
    if not allow_empty:
        _git(fixture.source_checkout, "add", "--all")
    arguments = ["commit", "-m", message]
    if allow_empty:
        arguments.append("--allow-empty")
    _git(fixture.source_checkout, *arguments)
    _git(fixture.source_checkout, "push", "origin", "HEAD")
    return _git(fixture.source_checkout, "rev-parse", "HEAD")


@pytest.fixture
def profile_fixture(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> ProfileFixture:
    home_root = tmp_path / "home"
    source_checkout = home_root / "sna/ModelOpt_SpecDec"
    source_checkout.mkdir(parents=True)
    _git(source_checkout, "init", "-b", "main")
    _git(source_checkout, "config", "user.email", "profile-tests@example.invalid")
    _git(source_checkout, "config", "user.name", "Profile Tests")
    for relative, content in _TOOL_CONTENTS.items():
        path = source_checkout / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
    _git(source_checkout, "add", "--all")
    _git(source_checkout, "commit", "-m", "fixture")

    remote = tmp_path / "remote.git"
    subprocess.run(
        ("/usr/bin/git", "init", "--bare", str(remote)),
        check=True,
        capture_output=True,
        env={"LC_ALL": "C", "PATH": "/usr/bin:/bin"},
    )
    _git(source_checkout, "remote", "add", "origin", str(remote))
    _git(source_checkout, "push", "-u", "origin", "main")

    durable_receipt_root = tmp_path / "lustre/runtime-qualification"
    durable_receipt_root.mkdir(parents=True)
    monkeypatch.setattr(profile_module, "_SOURCE_ROOT", home_root)
    monkeypatch.setattr(profile_module, "_EXPECTED_DURABLE_RECEIPT_ROOT", durable_receipt_root)
    return ProfileFixture(
        source_checkout=source_checkout,
        source_commit=_git(source_checkout, "rev-parse", "HEAD"),
        durable_receipt_root=durable_receipt_root,
        output_path=durable_receipt_root / "ptyche-profile.json",
    )


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _rewrite_profile(path: Path, field: str, value: object) -> str:
    payload = json.loads(path.read_bytes())
    payload[field] = value
    body = {key: item for key, item in payload.items() if key != "receipt_sha256"}
    body_bytes = json.dumps(body, sort_keys=True, separators=(",", ":")).encode()
    payload["receipt_sha256"] = hashlib.sha256(body_bytes).hexdigest()
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode() + b"\n"
    path.write_bytes(raw)
    return hashlib.sha256(raw).hexdigest()


def test_ptyche_profile_binds_exact_runtime_and_topologies(
    profile_fixture: ProfileFixture,
) -> None:
    profile = finalize_ptyche_runtime_profile(**profile_fixture.arguments())
    assert profile.cluster == "ptyche"
    assert profile.account == "coreai_dlalgo_llm"
    assert profile.partition == "36x2-a01r"
    assert profile.probe_node_counts == (1, 2)
    assert profile.training_nodes == 16
    assert profile.training_segment == 16
    assert profile.gpus_per_node == 4
    assert profile.scratch_root == "/raid/scratch"
    assert profile.archive_path == (
        "/lustre/fsw/coreai_dlalgo_llm/users/sna/modelopt-qwen3-drafter-training/"
        "assets/q235-training-prereqs-vllm0271-v1/runtime/"
        "modelopt-vllm-0.27.1-py312-aarch64-symlinks.tar.zst"
    )
    assert profile.archive_sha256 == (
        "4a20aee61f290c48bed22a84b4a0ae0cbdc54e3e3910854d253188c8854f5dc9"
    )
    assert profile.image_path == (
        "/lustre/fsw/coreai_dlalgo_llm/users/sna/modelopt-qwen3-drafter-training/"
        "assets/q235-training-prereqs-vllm0271-v1/image/"
        "vllm_openai_v0271_aarch64_20260813_2688476.sqsh"
    )
    assert profile.image_sha256 == (
        "e7be53f2754097c88f7c801da92f6d94794ec4d78d9df937fcd315a6994297f0"
    )
    assert profile.one_node_walltime == "00:20:00"
    assert profile.two_node_walltime == "00:30:00"
    assert profile.source_commit == profile_fixture.source_commit
    assert profile.source_checkout == str(profile_fixture.source_checkout)
    assert profile.durable_receipt_root == str(profile_fixture.durable_receipt_root)
    assert profile.archive_producer_sha256 == _file_sha256(
        profile_fixture.source_checkout
        / "tools/launcher/common/specdec/q30t_runtime_archive_receipt.py"
    )
    assert profile.keeper_tool_sha256 == _file_sha256(
        profile_fixture.source_checkout / "tools/launcher/common/specdec/ptv23_node_keeper.py"
    )
    assert profile.attestation_tool_sha256 == _file_sha256(
        profile_fixture.source_checkout
        / "tools/launcher/common/specdec/ptv23_runtime_attestation.py"
    )
    assert profile.probe_runner_sha256 == _file_sha256(
        profile_fixture.source_checkout
        / "tools/launcher/common/specdec/probe_ptv23_pyxis_keeper.sbatch"
    )


def test_profile_requires_all_tools_in_one_frozen_checkout(
    profile_fixture: ProfileFixture,
) -> None:
    missing = (
        profile_fixture.source_checkout
        / "tools/launcher/common/specdec/ptv23_runtime_attestation.py"
    )
    missing.unlink()
    source_commit = _commit_and_push(profile_fixture, "remove attester")
    arguments = profile_fixture.arguments() | {"source_commit": source_commit}
    with pytest.raises(ValueError, match="required profile tool"):
        finalize_ptyche_runtime_profile(**arguments)


def test_profile_rejects_dirty_source_checkout(profile_fixture: ProfileFixture) -> None:
    (profile_fixture.source_checkout / "untracked.txt").write_text("dirty\n")
    with pytest.raises(ValueError, match="clean"):
        finalize_ptyche_runtime_profile(**profile_fixture.arguments())


def test_profile_rejects_mismatched_source_commit(profile_fixture: ProfileFixture) -> None:
    arguments = profile_fixture.arguments() | {"source_commit": "f" * 40}
    with pytest.raises(ValueError, match="HEAD"):
        finalize_ptyche_runtime_profile(**arguments)


def test_profile_rejects_unpushed_head(profile_fixture: ProfileFixture) -> None:
    source_commit = _commit_and_push(profile_fixture, "pushed baseline", allow_empty=True)
    _git(profile_fixture.source_checkout, "commit", "--allow-empty", "-m", "local only")
    arguments = profile_fixture.arguments() | {
        "source_commit": _git(profile_fixture.source_checkout, "rev-parse", "HEAD")
    }
    assert source_commit != arguments["source_commit"]
    with pytest.raises(ValueError, match="upstream"):
        finalize_ptyche_runtime_profile(**arguments)


def test_profile_rejects_git_replace_refs(profile_fixture: ProfileFixture) -> None:
    source_commit = _commit_and_push(profile_fixture, "replacement target", allow_empty=True)
    _git(profile_fixture.source_checkout, "replace", source_commit, f"{source_commit}^")
    arguments = profile_fixture.arguments() | {"source_commit": source_commit}
    with pytest.raises(ValueError, match="replace"):
        finalize_ptyche_runtime_profile(**arguments)


def test_profile_rejects_source_outside_home(
    profile_fixture: ProfileFixture, tmp_path: Path
) -> None:
    outside = tmp_path / "outside-source"
    shutil.copytree(profile_fixture.source_checkout, outside, symlinks=True)
    arguments = profile_fixture.arguments() | {"source_checkout": outside}
    with pytest.raises(ValueError, match="/home"):
        finalize_ptyche_runtime_profile(**arguments)


def test_profile_rejects_receipt_root_outside_lustre(
    profile_fixture: ProfileFixture, tmp_path: Path
) -> None:
    outside = tmp_path / "outside-receipts"
    outside.mkdir()
    arguments = profile_fixture.arguments() | {
        "durable_receipt_root": outside,
        "output_path": outside / "profile.json",
    }
    with pytest.raises(ValueError, match="durable receipt root"):
        finalize_ptyche_runtime_profile(**arguments)


def test_profile_rejects_output_outside_receipt_root(
    profile_fixture: ProfileFixture, tmp_path: Path
) -> None:
    arguments = profile_fixture.arguments() | {"output_path": tmp_path / "profile.json"}
    with pytest.raises(ValueError, match="output"):
        finalize_ptyche_runtime_profile(**arguments)


def test_profile_rejects_source_checkout_path_rebind(
    profile_fixture: ProfileFixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    original = profile_fixture.source_checkout.with_name("original-checkout")

    def rebind() -> None:
        profile_fixture.source_checkout.rename(original)
        profile_fixture.source_checkout.mkdir()

    monkeypatch.setattr(profile_module, "_POST_SOURCE_AUTHENTICATION_HOOK", rebind)
    with pytest.raises(ValueError, match="rebound"):
        finalize_ptyche_runtime_profile(**profile_fixture.arguments())


def test_profile_rejects_tool_mutation_after_source_authentication(
    profile_fixture: ProfileFixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    tool = (
        profile_fixture.source_checkout
        / "tools/launcher/common/specdec/q30t_runtime_archive_receipt.py"
    )
    monkeypatch.setattr(
        profile_module,
        "_POST_SOURCE_AUTHENTICATION_HOOK",
        lambda: tool.write_bytes(b"mutated after clean check\n"),
    )
    with pytest.raises(ValueError, match="authenticated commit"):
        finalize_ptyche_runtime_profile(**profile_fixture.arguments())


def test_profile_rejects_tool_path_rebind_during_hash(
    profile_fixture: ProfileFixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    tool = (
        profile_fixture.source_checkout
        / "tools/launcher/common/specdec/q30t_runtime_archive_receipt.py"
    )
    displaced = tool.with_suffix(".original")

    def rebind(opened_path: Path) -> None:
        if opened_path == tool:
            tool.rename(displaced)
            tool.write_bytes(
                _TOOL_CONTENTS[tool.relative_to(profile_fixture.source_checkout).as_posix()]
            )

    monkeypatch.setattr(profile_module, "_POST_TOOL_OPEN_HOOK", rebind)
    with pytest.raises(ValueError, match="changed while hashing"):
        finalize_ptyche_runtime_profile(**profile_fixture.arguments())


def test_profile_adopts_only_exact_existing_output(profile_fixture: ProfileFixture) -> None:
    first = finalize_ptyche_runtime_profile(**profile_fixture.arguments())
    identity = os.stat(profile_fixture.output_path, follow_symlinks=False).st_ino
    second = finalize_ptyche_runtime_profile(**profile_fixture.arguments())
    assert second == first
    assert os.stat(profile_fixture.output_path, follow_symlinks=False).st_ino == identity


def test_profile_preserves_and_rejects_foreign_output(profile_fixture: ProfileFixture) -> None:
    foreign = b"foreign-profile\n"
    profile_fixture.output_path.write_bytes(foreign)
    with pytest.raises(ValueError, match="foreign"):
        finalize_ptyche_runtime_profile(**profile_fixture.arguments())
    assert profile_fixture.output_path.read_bytes() == foreign


def test_profile_loader_rejects_unknown_key(profile_fixture: ProfileFixture) -> None:
    finalize_ptyche_runtime_profile(**profile_fixture.arguments())
    payload = json.loads(profile_fixture.output_path.read_bytes())
    payload["approved"] = True
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode() + b"\n"
    profile_fixture.output_path.write_bytes(raw)
    with pytest.raises(ValueError, match="keys"):
        load_ptyche_runtime_profile(profile_fixture.output_path, hashlib.sha256(raw).hexdigest())


def test_profile_loader_rejects_noncanonical_bytes(profile_fixture: ProfileFixture) -> None:
    finalize_ptyche_runtime_profile(**profile_fixture.arguments())
    payload = json.loads(profile_fixture.output_path.read_bytes())
    raw = json.dumps(payload, indent=2, sort_keys=True).encode() + b"\n"
    profile_fixture.output_path.write_bytes(raw)
    with pytest.raises(ValueError, match="canonical"):
        load_ptyche_runtime_profile(profile_fixture.output_path, hashlib.sha256(raw).hexdigest())


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("schema_version", "q30t-runtime-cluster-profile-v2"),
        ("cluster", "lyris"),
        ("account", "wrong-account"),
        ("partition", "batch"),
        ("partition", "wrong-partition"),
        ("source_checkout", "/tmp/source"),
        ("durable_receipt_root", "/tmp/receipts"),
        ("scratch_root", "/tmp/scratch"),
        ("archive_path", "/lustre/foreign.tar.zst"),
        ("archive_sha256", "1" * 64),
        ("image_path", "/lustre/foreign.sqsh"),
        ("image_sha256", "2" * 64),
        ("probe_node_counts", [1, 3]),
        ("training_nodes", 8),
        ("training_segment", 8),
        ("gpus_per_node", 8),
        ("one_node_walltime", "01:00:00"),
        ("two_node_walltime", "01:00:00"),
    ],
)
def test_profile_loader_rejects_wrong_canonical_identity(
    profile_fixture: ProfileFixture, field: str, value: object
) -> None:
    finalize_ptyche_runtime_profile(**profile_fixture.arguments())
    whole_file_sha256 = _rewrite_profile(profile_fixture.output_path, field, value)
    with pytest.raises(ValueError, match=field.replace("_", " ")):
        load_ptyche_runtime_profile(profile_fixture.output_path, whole_file_sha256)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("source_commit", "F" * 40),
        ("archive_producer_sha256", "A" * 64),
        ("keeper_tool_sha256", "not-a-hash"),
        ("training_nodes", True),
        ("probe_node_counts", [True, 2]),
    ],
)
def test_profile_loader_rejects_wrong_types_and_hashes(
    profile_fixture: ProfileFixture, field: str, value: object
) -> None:
    finalize_ptyche_runtime_profile(**profile_fixture.arguments())
    whole_file_sha256 = _rewrite_profile(profile_fixture.output_path, field, value)
    with pytest.raises(ValueError, match=field.replace("_", " ")):
        load_ptyche_runtime_profile(profile_fixture.output_path, whole_file_sha256)


def test_profile_loader_rejects_wrong_self_hash(profile_fixture: ProfileFixture) -> None:
    finalize_ptyche_runtime_profile(**profile_fixture.arguments())
    payload = json.loads(profile_fixture.output_path.read_bytes())
    payload["receipt_sha256"] = "0" * 64
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode() + b"\n"
    profile_fixture.output_path.write_bytes(raw)
    with pytest.raises(ValueError, match="self-hash"):
        load_ptyche_runtime_profile(profile_fixture.output_path, hashlib.sha256(raw).hexdigest())


def test_profile_loader_rejects_wrong_whole_file_hash(profile_fixture: ProfileFixture) -> None:
    finalize_ptyche_runtime_profile(**profile_fixture.arguments())
    with pytest.raises(ValueError, match="whole-file"):
        load_ptyche_runtime_profile(profile_fixture.output_path, "0" * 64)


def test_profile_loader_rejects_hardlinked_receipt(profile_fixture: ProfileFixture) -> None:
    finalize_ptyche_runtime_profile(**profile_fixture.arguments())
    hardlink = profile_fixture.durable_receipt_root / "hardlink.json"
    os.link(profile_fixture.output_path, hardlink)
    with pytest.raises(ValueError, match="single-link"):
        load_ptyche_runtime_profile(
            profile_fixture.output_path, _file_sha256(profile_fixture.output_path)
        )


def test_profile_cli_finalizes_and_verifies(
    profile_fixture: ProfileFixture, capsys: pytest.CaptureFixture[str]
) -> None:
    result = profile_module.main(
        [
            "finalize",
            "--source-checkout",
            str(profile_fixture.source_checkout),
            "--source-commit",
            profile_fixture.source_commit,
            "--durable-receipt-root",
            str(profile_fixture.durable_receipt_root),
            "--output",
            str(profile_fixture.output_path),
            "--job-id",
            "profile-cli",
        ]
    )
    assert result == 0
    whole_file_sha256 = _file_sha256(profile_fixture.output_path)
    assert capsys.readouterr().out == f"{whole_file_sha256}\n"
    assert (
        profile_module.main(
            [
                "verify",
                "--profile",
                str(profile_fixture.output_path),
                "--profile-sha256",
                whole_file_sha256,
            ]
        )
        == 0
    )
    assert capsys.readouterr().out == f"{whole_file_sha256}\n"
