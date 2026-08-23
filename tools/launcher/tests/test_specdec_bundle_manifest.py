# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Corpus-specific transfer manifest identity contracts."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

_PATH = Path(__file__).parents[1] / "common/specdec/transfers/bundle_manifest.py"
_SPEC = importlib.util.spec_from_file_location("bundle_manifest_task8", _PATH)
assert _SPEC is not None and _SPEC.loader is not None
bundle_manifest = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = bundle_manifest
_SPEC.loader.exec_module(bundle_manifest)


def test_completion_marker_binds_artifact_source_commit(tmp_path: Path) -> None:
    """A completion marker cannot be reused by a different artifact-producing commit."""
    completion = tmp_path / "completion.json"
    bundle_manifest.write_completion(completion, "artifact", "b" * 64, "a" * 40)
    bundle_manifest.verify_completion(completion, "artifact", "b" * 64, "a" * 40)
    with pytest.raises(bundle_manifest.BundleError, match="artifact source commit"):
        bundle_manifest.verify_completion(completion, "artifact", "b" * 64, "c" * 40)

    original = completion.read_bytes()
    bundle_manifest.write_completion(completion, "artifact", "b" * 64, "a" * 40)
    assert completion.read_bytes() == original
    with pytest.raises(bundle_manifest.BundleError, match="immutable output"):
        bundle_manifest.write_completion(completion, "other", "b" * 64, "a" * 40)

    with pytest.raises(SystemExit):
        bundle_manifest.main(
            [
                "write-completion",
                "--output",
                str(tmp_path / "missing-commit.json"),
                "--artifact-id",
                "artifact",
                "--manifest-sha256",
                "b" * 64,
            ]
        )


def test_manifest_rejects_changed_or_extra_publication_files(tmp_path: Path) -> None:
    """Transfer verification authenticates the exact published file set and bytes."""
    durable = tmp_path / "durable"
    source = durable / "artifact"
    source.mkdir(parents=True)
    (source / "payload.bin").write_bytes(b"payload")
    manifest = tmp_path / "manifest.json"
    digest = bundle_manifest.create_manifest(source, durable, manifest, "a" * 40)
    bundle_manifest.load_manifest(manifest, digest, "a" * 40)
    bundle_manifest.verify_tree(manifest, source)

    (source / "extra").write_bytes(b"extra")
    with pytest.raises(bundle_manifest.BundleError, match="file set"):
        bundle_manifest.verify_tree(manifest, source)


def test_manifest_rejects_publication_from_another_artifact_commit(tmp_path: Path) -> None:
    """A transfer manifest cannot relabel a corpus produced by another source commit."""
    durable = tmp_path / "durable"
    source = durable / "artifact"
    source.mkdir(parents=True)
    receipt = {
        "artifact_id": "d" * 64,
        "artifact_source_commit": "c" * 40,
        "corpus_manifest_sha256": "b" * 64,
        "file_count": 1,
        "published_path": ".",
        "schema": "modelopt-specdec-publication-receipt-v1",
        "selection_manifest_sha256": "e" * 64,
        "total_bytes": 1,
    }
    (source / "PUBLICATION.json").write_text(
        json.dumps(receipt, sort_keys=True, separators=(",", ":")) + "\n"
    )
    with pytest.raises(bundle_manifest.BundleError, match="artifact source commit"):
        bundle_manifest.create_manifest(source, durable, tmp_path / "manifest.json", "a" * 40)
