# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Contracts for fail-closed PTV2/PTV3 source staging."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, cast

import pytest

ROOT = Path(__file__).resolve().parents[3]
MODULE_DIR = ROOT / "examples/dataset"
SOURCE_MANIFEST = MODULE_DIR / "bprime_cd_sources.json"
UNPINNED_PTV3_CONFIG = MODULE_DIR / "nemotron_ptv3_datasets.yaml"
SUBMITTER = ROOT / "tools/launcher/common/specdec/submit_ptv23_source_stage.sh"

sys.path.insert(0, str(MODULE_DIR))
try:
    from stage_ptv23_sources import (  # pyright: ignore[reportMissingImports]
        SourceManifestError,
        load_source_inventory,
        stage_source_inventory,
        validate_ptv3_config_pins,
    )
finally:
    sys.path.pop(0)


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _write_manifest(
    root: Path,
    *,
    approved_use: object = True,
    revision: object = "a" * 40,
    license_expression: object = "CC-BY-4.0",
    file_bytes: object | None = None,
    file_sha256: object | None = None,
    sources: list[dict[str, object]] | None = None,
) -> tuple[Path, Path]:
    source_root = root / "source-cache"
    source = source_root / "nvidia/Test" / ("a" * 40) / "data/train.jsonl"
    source.parent.mkdir(parents=True)
    data = b'{"id": 1}\n{"id": 2}\n'
    source.write_bytes(data)
    default_source: dict[str, object] = {
        "repository_id": "nvidia/Test",
        "configuration": "default",
        "split": "train",
        "revision": revision,
        "license_expression": license_expression,
        "approved_use": approved_use,
        "cell": "math",
        "lane": "target-synth",
        "files": [
            {
                "path": "data/train.jsonl",
                "bytes": len(data) if file_bytes is None else file_bytes,
                "sha256": _sha256(data) if file_sha256 is None else file_sha256,
            }
        ],
    }
    manifest = root / "sources.json"
    manifest.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "name": "test-sources",
                "sources": sources if sources is not None else [default_source],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    return manifest, source_root


@pytest.mark.parametrize(
    ("override", "message"),
    [
        ({"approved_use": False}, "approved use"),
        ({"approved_use": "yes"}, "approved_use.*boolean"),
        ({"revision": "a" * 39}, "40-character"),
        ({"revision": "g" * 40}, "40-character"),
        ({"license_expression": ""}, "license expression"),
        ({"file_bytes": 0}, "positive byte"),
        ({"file_sha256": "f" * 63}, "SHA-256"),
    ],
)
def test_source_inventory_requires_pin_license_digest_and_approval(
    tmp_path: Path, override: dict[str, object], message: str
) -> None:
    """Removing any provenance or approval gate must make a source ineligible."""
    manifest, _ = _write_manifest(tmp_path, **cast("dict[str, Any]", override))

    with pytest.raises(SourceManifestError, match=message):
        load_source_inventory(manifest)


def test_source_inventory_rejects_duplicate_logical_files(tmp_path: Path) -> None:
    """The same repository/revision/path must not be counted through two identities."""
    manifest, _ = _write_manifest(tmp_path)
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    duplicate = dict(payload["sources"][0])
    duplicate["split"] = "alias"
    payload["sources"].append(duplicate)
    manifest.write_text(json.dumps(payload) + "\n", encoding="utf-8")

    with pytest.raises(SourceManifestError, match="duplicate logical file"):
        load_source_inventory(manifest)


def test_staging_verifies_bytes_sha_counts_and_detects_stale_files(tmp_path: Path) -> None:
    """A byte mutation must fail before or after publication, never be silently reused."""
    manifest, source_root = _write_manifest(tmp_path)
    inventory = load_source_inventory(manifest)
    durable_root = tmp_path / "durable"
    scratch_root = tmp_path / "scratch"

    staged = stage_source_inventory(
        inventory,
        durable_root=durable_root,
        scratch_root=scratch_root,
        local_source_root=source_root,
    )

    assert staged.raw_counts == {"math": 2}
    assert staged.staged_root == durable_root / inventory.manifest_sha256
    receipt = staged.staged_root / "SOURCE_INVENTORY.json"
    assert receipt.is_file()
    reloaded = load_source_inventory(receipt)
    assert reloaded.sources == inventory.sources
    assert reloaded.raw_counts == {"math": 2}
    assert reloaded.staged_root == staged.staged_root
    staged_file = next(path for path in staged.staged_root.rglob("train.jsonl"))
    assert staged_file.stat().st_size == inventory.sources[0].files[0].bytes
    assert _sha256(staged_file.read_bytes()) == inventory.sources[0].files[0].sha256
    assert not list(scratch_root.glob(".ptv23-source-stage.*"))

    staged_file.write_bytes(b"stale\n")
    with pytest.raises(SourceManifestError, match="stale staged file"):
        stage_source_inventory(
            inventory,
            durable_root=durable_root,
            scratch_root=scratch_root,
            local_source_root=source_root,
        )


def test_staging_rejects_stale_local_source_before_publication(tmp_path: Path) -> None:
    """A cache entry that no longer matches its descriptor must not reach durable storage."""
    manifest, source_root = _write_manifest(tmp_path)
    inventory = load_source_inventory(manifest)
    local = source_root / "nvidia/Test" / ("a" * 40) / "data/train.jsonl"
    local.write_bytes(b"tampered\n")
    durable_root = tmp_path / "durable"

    with pytest.raises(SourceManifestError, match="source file identity mismatch"):
        stage_source_inventory(
            inventory,
            durable_root=durable_root,
            scratch_root=tmp_path / "scratch",
            local_source_root=source_root,
        )
    assert not durable_root.exists() or not any(durable_root.iterdir())


def test_staging_rejects_tampered_raw_count_receipt(tmp_path: Path) -> None:
    """Raw cell capacity must stay bound to the verified physical file set."""
    manifest, source_root = _write_manifest(tmp_path)
    inventory = load_source_inventory(manifest)
    durable_root = tmp_path / "durable"
    staged = stage_source_inventory(
        inventory,
        durable_root=durable_root,
        scratch_root=tmp_path / "scratch",
        local_source_root=source_root,
    )
    assert staged.staged_root is not None
    receipt = staged.staged_root / "SOURCE_INVENTORY.json"
    payload = json.loads(receipt.read_text(encoding="utf-8"))
    payload["raw_counts"]["math"] += 1
    receipt.write_text(json.dumps(payload) + "\n", encoding="utf-8")

    with pytest.raises(SourceManifestError, match="raw counts"):
        stage_source_inventory(
            inventory,
            durable_root=durable_root,
            scratch_root=tmp_path / "scratch",
            local_source_root=source_root,
        )


def test_unpinned_ptv3_yaml_is_refused() -> None:
    """The broad convenience YAML must never become an implicit production source plan."""
    with pytest.raises(SourceManifestError, match=r"unpinned.*revision"):
        validate_ptv3_config_pins(UNPINNED_PTV3_CONFIG)


def test_checked_in_inventory_has_complete_requested_source_lanes() -> None:
    """Dropping an approved D-lane source or its exact file pin breaks the inventory."""
    inventory = load_source_inventory(SOURCE_MANIFEST)
    identities = {
        (source.repository_id, source.split, source.lane): source for source in inventory.sources
    }

    assert ("nvidia/Nemotron-SFT-SWE-v2", "agentless", "agentless-swe") in identities
    assert (
        "nvidia/Nemotron-SFT-SWE-v2",
        "openhands_swe",
        "interactive-swe-replay",
    ) in identities
    assert (
        "nvidia/Nemotron-SWE-v1",
        "r2e_gym",
        "interactive-swe-replay",
    ) in identities
    assert (
        "nvidia/Nemotron-Agentic-v1",
        "interactive_agent",
        "generic-tool-replay",
    ) in identities
    assert (
        "nvidia/Nemotron-Agentic-v1",
        "tool_calling",
        "generic-tool-replay",
    ) in identities
    assert all(source.approved_use and source.files for source in inventory.sources)


def test_submitter_pulls_then_test_only_before_real_submission(tmp_path: Path) -> None:
    """Source staging must update safely and pass scheduler validation before submission."""
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    calls = tmp_path / "calls"
    git = fake_bin / "git"
    git.write_text(
        "#!/usr/bin/env bash\n"
        "set -euo pipefail\n"
        'printf \'git %s\\n\' "$*" >>"$CALLS"\n'
        'case "$*" in\n'
        "  *'rev-parse --show-toplevel') printf '%s\\n' \"$FAKE_REPO_ROOT\" ;;\n"
        "  *'rev-parse HEAD') printf '%040d\\n' 0 ;;\n"
        "  *'status --porcelain') : ;;\n"
        "  *'pull --ff-only') : ;;\n"
        "  *) exit 3 ;;\n"
        "esac\n",
        encoding="utf-8",
    )
    sbatch = fake_bin / "sbatch"
    sbatch.write_text(
        "#!/usr/bin/env bash\n"
        "set -euo pipefail\n"
        'printf \'sbatch %s\\n\' "$*" >>"$CALLS"\n'
        "[[ \" $* \" == *' --test-only '* ]] || printf '12345\\n'\n",
        encoding="utf-8",
    )
    git.chmod(0o755)
    sbatch.chmod(0o755)
    manifest, source_root = _write_manifest(tmp_path)
    env = os.environ | {
        "CALLS": str(calls),
        "FAKE_REPO_ROOT": str(ROOT),
        "PATH": f"{fake_bin}:{os.environ['PATH']}",
    }

    result = subprocess.run(
        [
            "bash",
            str(SUBMITTER),
            "--manifest",
            str(manifest),
            "--durable-root",
            str(tmp_path / "durable"),
            "--local-source-root",
            str(source_root),
            "--account",
            "test-account",
            "--partition",
            "gb200",
            "--time",
            "01:00:00",
        ],
        text=True,
        capture_output=True,
        env=env,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "12345"
    lines = calls.read_text(encoding="utf-8").splitlines()
    pull_index = next(index for index, line in enumerate(lines) if "pull --ff-only" in line)
    test_index = next(index for index, line in enumerate(lines) if "--test-only" in line)
    real_index = next(
        index
        for index, line in enumerate(lines)
        if line.startswith("sbatch ") and "--test-only" not in line
    )
    assert pull_index < test_index < real_index
    submitted = lines[real_index]
    assert "--cpus-per-task=144" in submitted
    assert "--gpus" not in submitted and "--gres" not in submitted
    assert hashlib.sha256(manifest.read_bytes()).hexdigest() in submitted
