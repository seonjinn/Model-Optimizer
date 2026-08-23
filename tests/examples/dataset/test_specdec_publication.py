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

"""Immutable SpecDec corpus publication contracts."""

from __future__ import annotations

import hashlib
import importlib
import json
import sys
import tracemalloc
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq
import pytest

sys.path.insert(0, str(Path(__file__).parents[3] / "examples/dataset"))
publication = importlib.import_module("specdec_publication")


def _canonical(value: object) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode()


def _receipt(root: Path, role: str, records: list[dict[str, Any]]) -> publication.InputArtifact:
    root.mkdir(parents=True, exist_ok=True)
    payload_path = root / f"{role}.jsonl"
    payload_path.write_bytes(b"".join(_canonical(record) for record in records))
    body = {
        "role": role,
        "files": [
            {
                "path": payload_path.name,
                "bytes": payload_path.stat().st_size,
                "sha256": hashlib.sha256(payload_path.read_bytes()).hexdigest(),
            }
        ],
    }
    body["receipt_sha256"] = hashlib.sha256(_canonical(body)).hexdigest()
    receipt_path = root / f"{role}.json"
    receipt_path.write_bytes(_canonical(body))
    return publication.InputArtifact(
        role=role,
        receipt_path=receipt_path,
        receipt_sha256=hashlib.sha256(receipt_path.read_bytes()).hexdigest(),
    )


def _rows(count: int = 5) -> list[dict[str, Any]]:
    return [
        {
            "prompt_uuid": f"{index:064x}",
            "domain": "math",
            "lane": "target-synthesis",
            "context_bucket": "le4k",
            "input_ids": [index, index + 1],
            "loss_mask": [False, True],
            "assistant_tokens": 1,
            "rejection_reason": None,
        }
        for index in range(count)
    ]


def _bundle(root: Path, *, count: int = 5) -> publication.CorpusBundle:
    rows = _rows(count)
    artifacts = tuple(_receipt(root, role, rows[:1]) for role in publication.REQUIRED_ROLES)
    return publication.CorpusBundle(
        artifacts=artifacts,
        rows=lambda: iter(rows),
        prompt_count=count,
        assistant_token_count=count,
        quarantine_count=0,
        selection_manifest_sha256=artifacts[1].receipt_sha256,
        artifact_source_commit="a" * 40,
    )


@pytest.mark.parametrize(
    "phase",
    ["shard_open", "shard_closed", "manifest", "directory_fsync", "rename"],
)
def test_interruption_preserves_partial_and_retry_resumes_verified_shards(
    tmp_path: Path, phase: str
) -> None:
    bundle = _bundle(tmp_path / "inputs")
    destination = tmp_path / "published"
    seen: list[str] = []

    def interrupt(current: str) -> None:
        seen.append(current)
        if current == phase:
            raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt) as captured:
        publication.publish_bundle(
            bundle, destination, "job-42", rows_per_shard=2, _phase_hook=interrupt
        )
    assert not destination.exists()
    state = publication.publication_recovery_state(captured.value)
    assert state.partial_path.exists()
    assert state.destination_observation.status == "absent"

    receipt = publication.publish_bundle(bundle, destination, "job-42", rows_per_shard=2)
    assert receipt.file_count >= 9
    assert destination.is_dir()
    assert (
        sum(pq.read_metadata(path).num_rows for path in destination.glob("shards/*.parquet")) == 5
    )
    assert "rename" in seen or phase != "rename"


def test_stale_partial_is_quarantined_and_collision_is_fail_closed(tmp_path: Path) -> None:
    bundle = _bundle(tmp_path / "first")
    destination = tmp_path / "published"

    def interrupt(phase: str) -> None:
        if phase == "shard_closed":
            raise RuntimeError("stop")

    with pytest.raises(RuntimeError):
        publication.publish_bundle(
            bundle, destination, "same-job", rows_per_shard=2, _phase_hook=interrupt
        )
    stale_partial = next(tmp_path.glob(".published.partial-same-job"))
    changed = _bundle(tmp_path / "second", count=3)
    receipt = publication.publish_bundle(changed, destination, "same-job", rows_per_shard=2)
    quarantines = list(tmp_path.glob(".published.quarantine-same-job-*"))
    assert receipt.artifact_id
    assert quarantines and not stale_partial.exists()

    with pytest.raises(FileExistsError, match="immutable publication destination"):
        publication.publish_bundle(changed, destination, "other-job", rows_per_shard=2)


@pytest.mark.parametrize("mutation", ["missing", "extra", "changed", "symlink"])
def test_input_receipts_reject_corrupt_or_unlisted_files(tmp_path: Path, mutation: str) -> None:
    bundle = _bundle(tmp_path / "inputs")
    artifact = bundle.artifacts[0]
    declared = artifact.receipt_path.parent / "source.jsonl"
    if mutation == "missing":
        declared.unlink()
    elif mutation == "extra":
        (artifact.receipt_path.parent / "unlisted.bin").write_bytes(b"extra")
    elif mutation == "changed":
        declared.write_bytes(b"changed")
    else:
        target = tmp_path / "target"
        target.write_bytes(declared.read_bytes())
        declared.unlink()
        declared.symlink_to(target)

    with pytest.raises(publication.PublicationError):
        publication.publish_bundle(bundle, tmp_path / "published", "job")


def test_complete_role_set_and_count_reconciliation_are_required(tmp_path: Path) -> None:
    bundle = _bundle(tmp_path / "inputs")
    incomplete = publication.CorpusBundle(
        artifacts=tuple(artifact for artifact in bundle.artifacts if artifact.role != "selection"),
        rows=bundle.rows,
        prompt_count=bundle.prompt_count,
        assistant_token_count=bundle.assistant_token_count,
        quarantine_count=bundle.quarantine_count,
        selection_manifest_sha256=bundle.selection_manifest_sha256,
        artifact_source_commit=bundle.artifact_source_commit,
    )
    with pytest.raises(publication.PublicationError, match="artifact roles"):
        publication.publish_bundle(incomplete, tmp_path / "published", "job")

    wrong_count = publication.CorpusBundle(
        artifacts=bundle.artifacts,
        rows=bundle.rows,
        prompt_count=99,
        assistant_token_count=bundle.assistant_token_count,
        quarantine_count=bundle.quarantine_count,
        selection_manifest_sha256=bundle.selection_manifest_sha256,
        artifact_source_commit=bundle.artifact_source_commit,
    )
    with pytest.raises(publication.PublicationError, match="prompt count"):
        publication.publish_bundle(wrong_count, tmp_path / "published", "job-2")


def test_rename_races_preserve_every_observed_inode(tmp_path: Path, monkeypatch: Any) -> None:
    """An ambiguous rename never causes the publisher to delete either pathname."""
    bundle = _bundle(tmp_path / "inputs")
    destination = tmp_path / "published"
    preserved = tmp_path / "preserved-partial"

    def racing_rename(source: Path, target: Path) -> None:
        source.rename(preserved)
        target.mkdir()
        (target / "sentinel").write_bytes(b"winner")
        raise FileExistsError(target)

    monkeypatch.setattr(publication, "_rename_no_replace", racing_rename)
    with pytest.raises(FileExistsError) as captured:
        publication.publish_bundle(bundle, destination, "race", rows_per_shard=2)
    state = publication.publication_recovery_state(captured.value)
    assert (destination / "sentinel").read_bytes() == b"winner"
    assert preserved.is_dir()
    assert state.partial_observation.status == "absent"
    assert state.destination_observation.status == "present"


def test_successful_but_raised_rename_and_parent_fsync_are_typed(
    tmp_path: Path, monkeypatch: Any
) -> None:
    """Post-rename ambiguity carries the installed inode and leaves it untouched."""
    bundle = _bundle(tmp_path / "inputs")
    destination = tmp_path / "published"
    rename = publication._rename_no_replace

    def ambiguous_rename(source: Path, target: Path) -> None:
        rename(source, target)
        raise RuntimeError("helper lost acknowledgement")

    monkeypatch.setattr(publication, "_rename_no_replace", ambiguous_rename)
    with pytest.raises(RuntimeError) as captured:
        publication.publish_bundle(bundle, destination, "ambiguous", rows_per_shard=2)
    state = publication.publication_recovery_state(captured.value)
    assert destination.is_dir()
    assert state.destination_observation.identity == state.expected_partial_identity

    second = tmp_path / "second"
    monkeypatch.setattr(publication, "_rename_no_replace", rename)
    fsync = publication._fsync_directory

    def failing_parent_fsync(path: Path) -> None:
        if path == tmp_path:
            raise OSError("durability unknown")
        fsync(path)

    monkeypatch.setattr(publication, "_fsync_directory", failing_parent_fsync)
    with pytest.raises(OSError) as fsync_captured:
        publication.publish_bundle(bundle, second, "fsync", rows_per_shard=2)
    fsync_state = publication.publication_recovery_state(fsync_captured.value)
    assert second.is_dir()
    assert fsync_state.phase is publication.PublicationPhase.PARENT_FSYNC
    assert fsync_state.destination_observation.identity == fsync_state.expected_partial_identity


def test_streaming_publication_has_bounded_memory_and_byte_idempotent_receipt(
    tmp_path: Path,
) -> None:
    count = 1_000
    base = _bundle(tmp_path / "inputs", count=1)

    def stream() -> Any:
        for index in range(count):
            yield _rows(1)[0] | {"prompt_uuid": f"{index:064x}"}

    bundle = publication.CorpusBundle(
        artifacts=base.artifacts,
        rows=stream,
        prompt_count=count,
        assistant_token_count=count,
        quarantine_count=0,
        selection_manifest_sha256=base.selection_manifest_sha256,
        artifact_source_commit=base.artifact_source_commit,
    )
    tracemalloc.start()
    first = publication.publish_bundle(bundle, tmp_path / "one", "one", rows_per_shard=127)
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    second = publication.publish_bundle(bundle, tmp_path / "two", "two", rows_per_shard=127)
    assert peak < 32 * 1024 * 1024
    assert first.artifact_id == second.artifact_id
    assert first.corpus_manifest_sha256 == second.corpus_manifest_sha256
    assert (tmp_path / "one/PUBLICATION.json").read_bytes() == (
        tmp_path / "two/PUBLICATION.json"
    ).read_bytes()
    assert json.loads((tmp_path / "one/PUBLICATION.json").read_bytes())[
        "artifact_source_commit"
    ] == ("a" * 40)
