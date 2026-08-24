# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Contracts for the immutable full-201 PTV2 source-manifest bootstrap."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from collections.abc import Mapping
    from types import ModuleType

ROOT = Path(__file__).resolve().parents[3]
MODULE_PATH = ROOT / "examples/dataset/bootstrap_ptv2_source_manifest.py"
RUNNER = ROOT / "tools/launcher/common/specdec/run_ptv2_source_manifest_bootstrap.sbatch"
SUBMITTER = ROOT / "tools/launcher/common/specdec/submit_ptv2_source_manifest_bootstrap.sh"
REVISION = "5c89e01dd720ae0f4058445ed49c5fb68a03c76e"
PRODUCTION_COUNTS = {
    "chat": 12,
    "math": 2,
    "code": 2,
    "stem": 2,
    "multilingual_de": 38,
    "multilingual_ja": 37,
    "multilingual_es": 33,
    "multilingual_fr": 37,
    "multilingual_it": 38,
}


def _load_module() -> ModuleType:
    spec = importlib.util.spec_from_file_location("bootstrap_ptv2_source_manifest", MODULE_PATH)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(spec.name, None)
        raise
    return module


def _fixture(
    tmp_path: Path, counts: Mapping[str, int]
) -> tuple[Path, Path, Path, dict[str, bytes]]:
    approved = tmp_path / "hao-immutable"
    blobs = approved / "blobs"
    blobs.mkdir(parents=True)
    symlink_root = tmp_path / "exact-symlinks"
    symlink_root.mkdir()
    contents: dict[str, bytes] = {}
    for split, count in counts.items():
        for index in range(count):
            name = f"{split}-{index:05d}-of-{count:05d}.parquet"
            payload = f"{split}:{index}\n".encode()
            target = blobs / f"{name}.blob"
            target.write_bytes(payload)
            (symlink_root / name).symlink_to(target)
            contents[name] = payload
    readme = approved / "README.md"
    readme.write_text("---\nlicense: cc-by-4.0\n---\n", encoding="utf-8")
    return symlink_root, approved, readme, contents


def _bootstrap(
    module: ModuleType,
    *,
    symlink_root: Path,
    approved: Path,
    readme: Path,
    output: Path,
    workers: int,
    counts: Mapping[str, int],
):
    return module._bootstrap_source_manifest(
        source_root=symlink_root,
        approved_hao_root=approved,
        readme_path=readme,
        output_root=output,
        source_commit="a" * 40,
        workers=workers,
        split_counts=counts,
    )


def test_scaled_one_worker_and_96_workers_publish_identical_manifests(tmp_path: Path) -> None:
    module = _load_module()
    counts = {"chat": 2, "math": 1, "multilingual_de": 2}
    symlink_root, approved, readme, _ = _fixture(tmp_path, counts)

    one = _bootstrap(
        module,
        symlink_root=symlink_root,
        approved=approved,
        readme=readme,
        output=tmp_path / "one",
        workers=1,
        counts=counts,
    )
    many = _bootstrap(
        module,
        symlink_root=symlink_root,
        approved=approved,
        readme=readme,
        output=tmp_path / "many",
        workers=96,
        counts=counts,
    )

    assert one.manifest_path.read_bytes() == many.manifest_path.read_bytes()
    assert one.manifest_sha256 == many.manifest_sha256
    one_receipt = json.loads(one.completion_path.read_bytes())
    many_receipt = json.loads(many.completion_path.read_bytes())
    assert one_receipt["workers"] == {"effective": 1, "requested": 1}
    assert many_receipt["workers"] == {"effective": 5, "requested": 96}


def test_exact_201_manifest_is_directly_loadable_and_has_exact_metadata(tmp_path: Path) -> None:
    module = _load_module()
    symlink_root, approved, readme, contents = _fixture(tmp_path, PRODUCTION_COUNTS)
    result = module.bootstrap_ptv2_source_manifest(
        source_root=symlink_root,
        approved_hao_root=approved,
        readme_path=readme,
        output_root=tmp_path / "published",
        source_commit="b" * 40,
        workers=96,
    )

    stage_path = ROOT / "examples/dataset/stage_ptv23_sources.py"
    stage_spec = importlib.util.spec_from_file_location("stage_ptv23_sources", stage_path)
    assert stage_spec and stage_spec.loader
    stage_module = importlib.util.module_from_spec(stage_spec)
    sys.modules[stage_spec.name] = stage_module
    try:
        stage_spec.loader.exec_module(stage_module)
        inventory = getattr(stage_module, "load_source_inventory")(result.manifest_path)
    finally:
        sys.modules.pop(stage_spec.name, None)

    assert len(inventory.sources) == 9
    assert sum(len(source.files) for source in inventory.sources) == 201
    assert [source.split for source in inventory.sources] == list(PRODUCTION_COUNTS)
    for source in inventory.sources:
        assert source.repository_id == "nvidia/Nemotron-Post-Training-Dataset-v2"
        assert source.configuration == "default"
        assert source.revision == REVISION
        assert source.license_expression == "CC-BY-4.0"
        assert source.approved_use is True
        assert source.cell == source.split
        assert source.lane == "source-native"
        assert len(source.files) == PRODUCTION_COUNTS[source.split]
        for descriptor in source.files:
            basename = Path(descriptor.path).name
            assert descriptor.path == f"data/{basename}"
            assert descriptor.bytes == len(contents[basename])
            assert descriptor.sha256 == hashlib.sha256(contents[basename]).hexdigest()

    receipt = json.loads(result.completion_path.read_bytes())
    assert receipt["complete"] is True
    assert receipt["source_commit"] == "b" * 40
    assert receipt["source_root_realpath"] == str(symlink_root.resolve())
    assert receipt["approved_hao_root_realpath"] == str(approved.resolve())
    assert receipt["manifest"] == {
        "bytes": result.manifest_path.stat().st_size,
        "path": "SOURCE_PLAN.json",
        "sha256": result.manifest_sha256,
    }
    assert receipt["readme"]["sha256"] == hashlib.sha256(readme.read_bytes()).hexdigest()
    assert receipt["split_counts"] == PRODUCTION_COUNTS
    assert receipt["split_bytes"] == {
        split: sum(len(value) for name, value in contents.items() if name.startswith(f"{split}-"))
        for split in PRODUCTION_COUNTS
    }
    assert receipt["workers"] == {"effective": 96, "requested": 96}
    assert receipt["timing"]["duration_seconds"] >= 0


@pytest.mark.parametrize("fault", ["missing", "orphan", "regular", "escape"])
def test_source_contract_fails_closed(tmp_path: Path, fault: str) -> None:
    module = _load_module()
    counts = {"chat": 2, "math": 1}
    symlink_root, approved, readme, _ = _fixture(tmp_path, counts)
    first = symlink_root / "chat-00000-of-00002.parquet"
    if fault == "missing":
        first.unlink()
    elif fault == "orphan":
        target = approved / "blobs/orphan.blob"
        target.write_bytes(b"orphan")
        (symlink_root / "multilingual-00000-of-00001.parquet").symlink_to(target)
    elif fault == "regular":
        payload = first.read_bytes()
        first.unlink()
        first.write_bytes(payload)
    else:
        outside = tmp_path / "outside.parquet"
        outside.write_bytes(b"outside")
        first.unlink()
        first.symlink_to(outside)

    with pytest.raises(module.PTV2SourceManifestError):
        _bootstrap(
            module,
            symlink_root=symlink_root,
            approved=approved,
            readme=readme,
            output=tmp_path / "out",
            workers=2,
            counts=counts,
        )


def test_worker_failure_propagates_without_publication(tmp_path: Path) -> None:
    module = _load_module()
    counts = {"chat": 2}
    symlink_root, approved, readme, _ = _fixture(tmp_path, counts)
    broken = symlink_root / "chat-00001-of-00002.parquet"
    broken.unlink()
    broken.symlink_to(approved / "blobs/absent")
    output = tmp_path / "out"

    with pytest.raises(module.PTV2SourceManifestError):
        _bootstrap(
            module,
            symlink_root=symlink_root,
            approved=approved,
            readme=readme,
            output=output,
            workers=2,
            counts=counts,
        )
    assert not output.exists()


def test_fd_mutation_is_detected(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    module = _load_module()
    counts = {"chat": 1}
    symlink_root, approved, readme, _ = _fixture(tmp_path, counts)
    real_fstat = module.os.fstat
    observations = 0

    def unstable_fstat(descriptor: int):
        nonlocal observations
        result = real_fstat(descriptor)
        observations += 1
        if observations == 2:
            values = list(result)
            values[8] += 1
            return os.stat_result(values)
        return result

    monkeypatch.setattr(module.os, "fstat", unstable_fstat)
    with pytest.raises(module.PTV2SourceManifestError, match="changed while hashing"):
        _bootstrap(
            module,
            symlink_root=symlink_root,
            approved=approved,
            readme=readme,
            output=tmp_path / "out",
            workers=1,
            counts=counts,
        )


def test_publication_is_byte_idempotent_and_refuses_changed_input(tmp_path: Path) -> None:
    module = _load_module()
    counts = {"chat": 1}
    symlink_root, approved, readme, _ = _fixture(tmp_path, counts)
    output = tmp_path / "out"
    first = _bootstrap(
        module,
        symlink_root=symlink_root,
        approved=approved,
        readme=readme,
        output=output,
        workers=1,
        counts=counts,
    )
    before = (first.manifest_path.read_bytes(), first.completion_path.read_bytes())
    second = _bootstrap(
        module,
        symlink_root=symlink_root,
        approved=approved,
        readme=readme,
        output=output,
        workers=1,
        counts=counts,
    )
    assert (second.manifest_path.read_bytes(), second.completion_path.read_bytes()) == before

    target = next((approved / "blobs").iterdir())
    target.write_bytes(b"changed")
    with pytest.raises(module.PTV2SourceManifestError, match="immutable publication mismatch"):
        _bootstrap(
            module,
            symlink_root=symlink_root,
            approved=approved,
            readme=readme,
            output=output,
            workers=1,
            counts=counts,
        )
    assert (first.manifest_path.read_bytes(), first.completion_path.read_bytes()) == before


def test_cpu_datamover_runner_and_submitter_contracts() -> None:
    runner = RUNNER.read_text(encoding="utf-8")
    submitter = SUBMITTER.read_text(encoding="utf-8")

    assert "#SBATCH --partition=cpu_datamover" in runner
    assert "#SBATCH --nodes=1" in runner
    assert "#SBATCH --cpus-per-task=96" in runner
    assert '[[ "${SLURM_CPUS_PER_TASK:-}" == "96" ]]' in runner
    assert '--workers "$SLURM_CPUS_PER_TASK"' in runner
    assert 'git -C "$REPO_ROOT" pull --ff-only' in submitter
    assert 'git -C "$REPO_ROOT" status --porcelain' in submitter
    assert '[[ "$(git -C "$REPO_ROOT" rev-parse HEAD)" == "$SOURCE_COMMIT" ]]' in submitter
    assert "nemotron_n4_post|nemotron_sw_post" in submitter
    assert "sbatch --test-only" in submitter
    assert "--partition=cpu_datamover" in submitter
    assert "--cpus-per-task=96" in submitter


def test_submitter_dry_run_pulls_exact_clean_commit_and_test_only(
    tmp_path: Path,
) -> None:
    source_commit = "d" * 40
    repo = tmp_path / "repo"
    runner = repo / "tools/launcher/common/specdec/run_ptv2_source_manifest_bootstrap.sbatch"
    runner.parent.mkdir(parents=True)
    runner.write_text("#!/usr/bin/env bash\n", encoding="utf-8")
    source_root, approved, readme, _ = _fixture(tmp_path, {"chat": 1})
    binary_root = tmp_path / "bin"
    binary_root.mkdir()
    trace = tmp_path / "trace"
    git = binary_root / "git"
    git.write_text(
        "#!/usr/bin/env bash\n"
        "set -eu\n"
        f"printf 'git %s\\n' \"$*\" >> {trace!s}\n"
        f"[[ \"$*\" == *'rev-parse HEAD'* ]] && printf '%s\\n' {source_commit} || true\n",
        encoding="utf-8",
    )
    sbatch = binary_root / "sbatch"
    sbatch.write_text(
        f"#!/usr/bin/env bash\nset -eu\nprintf 'sbatch %s\\n' \"$*\" >> {trace!s}\n",
        encoding="utf-8",
    )
    git.chmod(0o755)
    sbatch.chmod(0o755)

    result = subprocess.run(
        [
            str(SUBMITTER),
            "--account",
            "nemotron_n4_post",
            "--repo-root",
            str(repo),
            "--source-commit",
            source_commit,
            "--source-root",
            str(source_root),
            "--approved-hao-root",
            str(approved),
            "--readme",
            str(readme),
            "--output-root",
            str(tmp_path / "output"),
            "--time",
            "01:00:00",
            "--dry-run",
        ],
        check=False,
        capture_output=True,
        text=True,
        env={**os.environ, "PATH": f"{binary_root}:{os.environ['PATH']}"},
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout == "test-only passed\n"
    calls = trace.read_text(encoding="utf-8")
    assert "git -C " in calls and " pull --ff-only" in calls
    assert "status --porcelain" in calls
    assert "rev-parse HEAD" in calls
    assert "sbatch --test-only" in calls
    assert "--partition=cpu_datamover" in calls
    assert "--cpus-per-task=96" in calls


def test_cli_rejects_nonproduction_scaled_root(tmp_path: Path) -> None:
    counts = {"chat": 1}
    symlink_root, approved, readme, _ = _fixture(tmp_path, counts)
    result = subprocess.run(
        [
            sys.executable,
            str(MODULE_PATH),
            "--source-root",
            str(symlink_root),
            "--approved-hao-root",
            str(approved),
            "--readme",
            str(readme),
            "--output-root",
            str(tmp_path / "out"),
            "--source-commit",
            "c" * 40,
            "--workers",
            "1",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert "expected exactly 201" in result.stderr
