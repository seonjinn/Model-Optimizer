# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
PREPARER = ROOT / "examples/dataset/prepare_ptv2_baseline_audit.py"
SOURCE_REVISION = "5c89e01dd720ae0f4058445ed49c5fb68a03c76e"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_preparer_authenticates_asset_and_invokes_audit_with_canonical_view(
    tmp_path: Path,
) -> None:
    asset = tmp_path / "asset"
    source = asset / "data"
    source.mkdir(parents=True)
    receipt_root = asset / "manifest"
    receipt_root.mkdir()
    source_files = []
    for index in range(26):
        path = source / f"chat-{index:05d}-of-00026.parquet"
        path.write_bytes(f"fixture-{index}\n".encode())
        source_files.append(path)
    identity = receipt_root / "identity.json"
    identity.write_text('{"identity":"fixture"}\n')
    completion = asset / "completion.json"
    completion.write_text('{"status":"complete"}\n')
    sha_list = receipt_root / "nemotron-1p3m.sha256"
    sha_list.write_text("".join(f"{_sha256(path)}  ./{path.name}\n" for path in source_files))
    audit = tmp_path / "record_audit.py"
    audit.write_text(
        """from __future__ import annotations
import json
import sys
from pathlib import Path

arguments = dict(zip(sys.argv[1::2], sys.argv[2::2]))
root = Path(arguments[\"--root\"])
manifest = Path(arguments[\"--source-manifest\"])
record = {
    \"arguments\": arguments,
    \"links\": [
        {\"name\": link.name, \"target\": str(link.resolve(strict=True))}
        for link in sorted(root.iterdir())
    ],
    \"manifest\": json.loads(manifest.read_bytes()),
}
Path(arguments[\"--output\"]).write_text(json.dumps(record, sort_keys=True))
"""
    )
    work = tmp_path / "work"
    output = tmp_path / "audit-output.json"

    subprocess.run(
        [
            sys.executable,
            str(PREPARER),
            "--asset-root",
            str(asset),
            "--source-root",
            str(source),
            "--identity-path",
            str(identity),
            "--identity-sha256",
            _sha256(identity),
            "--completion-path",
            str(completion),
            "--completion-sha256",
            _sha256(completion),
            "--sha256-list-path",
            str(sha_list),
            "--sha256-list-sha256",
            _sha256(sha_list),
            "--work-root",
            str(work),
            "--audit-script",
            str(audit),
            "--output",
            str(output),
        ],
        check=True,
    )

    record = json.loads(output.read_bytes())
    manifest = record["manifest"]
    assert record["arguments"] == {
        "--output": str(output),
        "--root": str(work / "data"),
        "--source-manifest": str(work / "SOURCE_MANIFEST.json"),
        "--source-manifest-sha256": _sha256(work / "SOURCE_MANIFEST.json"),
    }
    assert record["links"] == [{"name": path.name, "target": str(path)} for path in source_files]
    assert manifest["source_revision"] == SOURCE_REVISION
    assert manifest["files"] == [
        {"bytes": path.stat().st_size, "name": path.name, "sha256": _sha256(path)}
        for path in source_files
    ]
    assert manifest["asset_evidence"] == {
        "asset_root": str(asset),
        "completion": {"path": "completion.json", "sha256": _sha256(completion)},
        "identity": {"path": "manifest/identity.json", "sha256": _sha256(identity)},
        "sha256_list": {
            "path": "manifest/nemotron-1p3m.sha256",
            "sha256": _sha256(sha_list),
        },
    }
