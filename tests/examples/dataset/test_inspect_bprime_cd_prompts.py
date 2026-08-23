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

from __future__ import annotations

import hashlib
import json
import sqlite3
import sys
import tracemalloc
from dataclasses import replace
from pathlib import Path
from types import MappingProxyType

import pytest

ROOT = Path(__file__).resolve().parents[3]
MODULE_DIR = ROOT / "examples/dataset"
TEST_DIR = Path(__file__).parent

sys.path[:0] = [str(MODULE_DIR), str(TEST_DIR)]
try:
    import select_bprime_cd_prompts as selection_module
    from bprime_cd_policy import PromptCell
    from inspect_bprime_cd_prompts import inspect_prompt_manifest, main
    from select_bprime_cd_prompts import (
        PromptPublicationDurabilityError,
        publish_prompt_view_bundle,
        select_prompt_views,
    )
    from specdec_corpus_contracts import canonical_json, sha256_bytes
    from test_select_bprime_cd_prompts import (
        BASELINE_RECEIPT_SHA256,
        HELD_OUT_RECEIPT_SHA256,
        _candidate,
        _inventory,
        _policy,
    )
finally:
    del sys.path[:2]


def _publish(path: Path, *, reverse: bool = False, rows_per_shard: int = 37):
    bundle = select_prompt_views(
        _inventory(reverse=reverse),
        _policy(),
        baseline_receipt_sha256=BASELINE_RECEIPT_SHA256,
        held_out_receipt_sha256=HELD_OUT_RECEIPT_SHA256,
    )
    return publish_prompt_view_bundle(bundle, path, rows_per_shard=rows_per_shard)


def test_sharded_publication_and_indexes_are_stable_across_inventory_order(tmp_path: Path) -> None:
    forward = _publish(tmp_path / "forward")
    reverse = _publish(tmp_path / "reverse", reverse=True)

    first = inspect_prompt_manifest(
        forward.manifest_path, expected_root_sha256=forward.root_sha256, arm="C", limit=1_000
    )
    second = inspect_prompt_manifest(
        reverse.manifest_path, expected_root_sha256=reverse.root_sha256, arm="C", limit=1_000
    )

    assert forward.root_sha256 == reverse.root_sha256
    assert first == second
    manifest = json.loads(forward.manifest_path.read_bytes())
    assert "rows" not in json.dumps(manifest["arms"])
    assert len(manifest["shards"]) > 1
    assert [row["selection_index"] for row in first["rows"] if row["status"] == "primary"] == list(
        range(400)
    )
    assert [row["selection_index"] for row in first["rows"] if row["status"] == "reserve"] == list(
        range(80)
    )


def test_filters_are_exact_and_inspection_reads_only_the_bounded_rows(tmp_path: Path) -> None:
    published = _publish(tmp_path / "selection", rows_per_shard=5)
    filtered = inspect_prompt_manifest(
        published.manifest_path,
        expected_root_sha256=published.root_sha256,
        arm="B-prime",
        domain="multilingual",
        lane="target-synth",
        language="ja",
        offset=3,
        limit=2,
    )
    assert filtered["total_matches"] == 30
    assert filtered["rows_examined"] == len(filtered["rows"]) == 2
    assert all(row["language"] == "ja" for row in filtered["rows"])
    prompt_uuid = filtered["rows"][0]["prompt_uuid"]
    exact = inspect_prompt_manifest(
        published.manifest_path,
        expected_root_sha256=published.root_sha256,
        prompt_uuid=prompt_uuid,
        limit=10,
    )
    assert {row["prompt_uuid"] for row in exact["rows"]} == {prompt_uuid}


def test_large_synthetic_selection_publication_and_inspection_are_bounded(tmp_path: Path) -> None:
    inventory = _inventory()
    noise = (
        _candidate(
            1_000_000 + index,
            domain="unused-domain",
            lane="target-synth",
            language="en",
            bucket="le4k",
        )
        for index in range(10_000)
    )
    streaming_inventory = replace(inventory, rows=(*inventory.rows, *noise))
    streaming_inventory = replace(
        streaming_inventory,
        inventory_sha256=selection_module.candidate_inventory_sha256(streaming_inventory),
    )

    tracemalloc.start()
    bundle = select_prompt_views(
        streaming_inventory,
        _policy(),
        baseline_receipt_sha256=BASELINE_RECEIPT_SHA256,
        held_out_receipt_sha256=HELD_OUT_RECEIPT_SHA256,
    )
    published = publish_prompt_view_bundle(bundle, tmp_path / "large", rows_per_shard=11)
    result = inspect_prompt_manifest(
        published.manifest_path,
        expected_root_sha256=published.root_sha256,
        arm="D",
        offset=397,
        limit=3,
    )
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    assert result["rows_examined"] == 3
    assert peak < 16 * 1024 * 1024


def test_repaired_outer_digest_cannot_replace_the_externally_trusted_root(tmp_path: Path) -> None:
    published = _publish(tmp_path / "selection")
    manifest = json.loads(published.manifest_path.read_bytes())
    manifest["identity"]["seed"] += 1
    unsigned = dict(manifest)
    del unsigned["root_sha256"]
    manifest["root_sha256"] = sha256_bytes(canonical_json(unsigned))
    published.manifest_path.write_bytes(canonical_json(manifest) + b"\n")

    with pytest.raises(ValueError, match="externally trusted root"):
        inspect_prompt_manifest(
            published.manifest_path,
            expected_root_sha256=published.root_sha256,
        )
    with pytest.raises(ValueError, match="nonzero"):
        inspect_prompt_manifest(
            published.manifest_path,
            expected_root_sha256="0" * 64,
        )
    published.manifest_path.write_text("[]\n", encoding="utf-8")
    with pytest.raises(ValueError, match="manifest"):
        inspect_prompt_manifest(
            published.manifest_path,
            expected_root_sha256=published.root_sha256,
        )


def test_repaired_semantic_proof_and_corrupt_unqueried_shard_are_rejected(tmp_path: Path) -> None:
    published = _publish(tmp_path / "proof", rows_per_shard=10)
    manifest = json.loads(published.manifest_path.read_bytes())
    manifest["arms"]["B-prime"]["count_proof_sha256"] = "9" * 64
    unsigned = dict(manifest)
    del unsigned["root_sha256"]
    repaired_root = sha256_bytes(canonical_json(unsigned))
    manifest["root_sha256"] = repaired_root
    published.manifest_path.write_bytes(canonical_json(manifest) + b"\n")

    with pytest.raises(ValueError, match=r"row-derived|count proof"):
        inspect_prompt_manifest(
            published.manifest_path,
            expected_root_sha256=repaired_root,
            arm="B-prime",
            limit=1,
        )


def test_repaired_index_semantics_and_bogus_nonagentic_floors_are_rejected(
    tmp_path: Path,
) -> None:
    indexed = _publish(tmp_path / "indexed")
    connection = sqlite3.connect(indexed.index_path)
    connection.execute("UPDATE rows SET domain='tampered-domain' WHERE global_index=0")
    connection.commit()
    connection.execute("VACUUM")
    connection.close()
    manifest = json.loads(indexed.manifest_path.read_bytes())
    manifest["index"]["sha256"] = hashlib.sha256(indexed.index_path.read_bytes()).hexdigest()
    unsigned = dict(manifest)
    del unsigned["root_sha256"]
    repaired_root = sha256_bytes(canonical_json(unsigned))
    manifest["root_sha256"] = repaired_root
    indexed.manifest_path.write_bytes(canonical_json(manifest) + b"\n")
    with pytest.raises(ValueError, match="index row metadata"):
        inspect_prompt_manifest(indexed.manifest_path, expected_root_sha256=repaired_root)

    floors = _publish(tmp_path / "floors")
    manifest = json.loads(floors.manifest_path.read_bytes())
    for arm in ("C", "D"):
        manifest["arms"][arm]["non_agentic_bucket_floors"]["math"] = {
            "le4k": 45,
            "4k_16k": 35,
        }
    unsigned = dict(manifest)
    del unsigned["root_sha256"]
    repaired_root = sha256_bytes(canonical_json(unsigned))
    manifest["root_sha256"] = repaired_root
    floors.manifest_path.write_bytes(canonical_json(manifest) + b"\n")
    with pytest.raises(ValueError, match="row-derived non-agentic"):
        inspect_prompt_manifest(floors.manifest_path, expected_root_sha256=repaired_root)

    clean = _publish(tmp_path / "shard", rows_per_shard=10)
    clean_manifest = json.loads(clean.manifest_path.read_bytes())
    unqueried = clean.manifest_path.parent / clean_manifest["shards"][-1]["path"]
    data = bytearray(unqueried.read_bytes())
    data[-2] ^= 1
    unqueried.write_bytes(data)
    with pytest.raises(ValueError, match="shard digest"):
        inspect_prompt_manifest(
            clean.manifest_path,
            expected_root_sha256=clean.root_sha256,
            arm="B-prime",
            limit=1,
        )


def test_prerename_path_replacement_preserves_every_namespace_for_recovery(
    tmp_path: Path, monkeypatch
) -> None:
    bundle = select_prompt_views(
        _inventory(),
        _policy(),
        baseline_receipt_sha256=BASELINE_RECEIPT_SHA256,
        held_out_receipt_sha256=HELD_OUT_RECEIPT_SHA256,
    )
    output = tmp_path / "race"
    displaced = tmp_path / "publisher-partial-before-replacement"
    partial: Path | None = None

    def race(source: Path, destination: Path) -> None:
        nonlocal partial
        partial = source
        source.rename(displaced)
        source.mkdir()
        (source / "sentinel").write_text("concurrent partial owner", encoding="utf-8")
        destination.mkdir()
        (destination / "sentinel").write_text("winner", encoding="utf-8")
        raise FileExistsError(destination)

    monkeypatch.setattr(selection_module, "_rename_no_replace", race, raising=False)
    with pytest.raises(FileExistsError) as caught:
        publish_prompt_view_bundle(bundle, output)

    assert partial is not None
    recovery = selection_module.prompt_publication_recovery_state(caught.value)
    displaced_identity = displaced.stat(follow_symlinks=False)
    partial_identity = partial.stat(follow_symlinks=False)
    destination_identity = output.stat(follow_symlinks=False)
    assert recovery.phase.value == "rename"
    assert recovery.partial_path == partial
    assert recovery.destination_path == output
    assert recovery.expected_artifact_identity is not None
    assert recovery.expected_artifact_identity.device == displaced_identity.st_dev
    assert recovery.expected_artifact_identity.inode == displaced_identity.st_ino
    assert recovery.expected_artifact_identity.root_sha256 is not None
    assert recovery.partial_observation.status == "present"
    assert recovery.partial_observation.device == partial_identity.st_dev
    assert recovery.partial_observation.inode == partial_identity.st_ino
    assert recovery.destination_observation.status == "present"
    assert recovery.destination_observation.device == destination_identity.st_dev
    assert recovery.destination_observation.inode == destination_identity.st_ino
    assert (partial / "sentinel").read_text(encoding="utf-8") == "concurrent partial owner"
    assert (output / "sentinel").read_text(encoding="utf-8") == "winner"
    assert list(displaced.iterdir())


def test_ordinary_setup_failure_preserves_replaced_partial_with_recovery_state(
    tmp_path: Path, monkeypatch
) -> None:
    bundle = select_prompt_views(
        _inventory(),
        _policy(),
        baseline_receipt_sha256=BASELINE_RECEIPT_SHA256,
        held_out_receipt_sha256=HELD_OUT_RECEIPT_SHA256,
    )
    output = tmp_path / "setup"
    displaced = tmp_path / "publisher-setup-partial"
    partial: Path | None = None

    def replace_then_fail(index_path: Path, *args, **kwargs):
        del args, kwargs
        nonlocal partial
        partial = Path(index_path).parent
        partial.rename(displaced)
        partial.mkdir()
        (partial / "sentinel").write_text("concurrent setup owner", encoding="utf-8")
        raise sqlite3.OperationalError("setup")

    monkeypatch.setattr(selection_module.sqlite3, "connect", replace_then_fail)
    with pytest.raises(sqlite3.OperationalError, match="setup") as caught:
        publish_prompt_view_bundle(bundle, output)

    assert partial is not None
    recovery = selection_module.prompt_publication_recovery_state(caught.value)
    displaced_identity = displaced.stat(follow_symlinks=False)
    partial_identity = partial.stat(follow_symlinks=False)
    assert recovery.phase.value == "partial_setup"
    assert recovery.partial_path == partial
    assert recovery.destination_path == output
    assert recovery.expected_artifact_identity is not None
    assert recovery.expected_artifact_identity.device == displaced_identity.st_dev
    assert recovery.expected_artifact_identity.inode == displaced_identity.st_ino
    assert recovery.expected_artifact_identity.root_sha256 is None
    assert recovery.partial_observation.status == "present"
    assert recovery.partial_observation.device == partial_identity.st_dev
    assert recovery.partial_observation.inode == partial_identity.st_ino
    assert recovery.destination_observation.status == "absent"
    assert (partial / "sentinel").read_text(encoding="utf-8") == "concurrent setup owner"
    assert list(displaced.iterdir())


def test_successful_but_raising_rename_preserves_ambiguous_paths_for_recovery(
    tmp_path: Path, monkeypatch
) -> None:
    bundle = select_prompt_views(
        _inventory(),
        _policy(),
        baseline_receipt_sha256=BASELINE_RECEIPT_SHA256,
        held_out_receipt_sha256=HELD_OUT_RECEIPT_SHA256,
    )
    output = tmp_path / "ambiguous"
    original_rename = selection_module._rename_no_replace
    partial: Path | None = None

    def rename_then_raise(source: Path, destination: Path) -> None:
        nonlocal partial
        original_rename(source, destination)
        partial = source
        partial.mkdir()
        (partial / "sentinel").write_text("concurrent post-rename owner", encoding="utf-8")
        raise OSError("rename helper raised after success")

    monkeypatch.setattr(selection_module, "_rename_no_replace", rename_then_raise)
    with pytest.raises(OSError, match="rename helper raised after success") as caught:
        publish_prompt_view_bundle(bundle, output)

    assert partial is not None
    recovery = selection_module.prompt_publication_recovery_state(caught.value)
    published_identity = output.stat(follow_symlinks=False)
    partial_identity = partial.stat(follow_symlinks=False)
    assert recovery.phase.value == "rename"
    assert recovery.expected_artifact_identity is not None
    assert recovery.expected_artifact_identity.device == published_identity.st_dev
    assert recovery.expected_artifact_identity.inode == published_identity.st_ino
    assert recovery.destination_observation.status == "present"
    assert recovery.destination_observation.inode == published_identity.st_ino
    assert recovery.partial_observation.status == "present"
    assert recovery.partial_observation.inode == partial_identity.st_ino
    assert (partial / "sentinel").read_text(encoding="utf-8") == ("concurrent post-rename owner")
    assert (output / "SELECTION_MANIFEST.json").is_file()


def test_async_keyboard_interrupt_preserves_ambiguous_rename_state(
    tmp_path: Path, monkeypatch
) -> None:
    bundle = select_prompt_views(
        _inventory(),
        _policy(),
        baseline_receipt_sha256=BASELINE_RECEIPT_SHA256,
        held_out_receipt_sha256=HELD_OUT_RECEIPT_SHA256,
    )
    output = tmp_path / "interrupt"
    original_rename = selection_module._rename_no_replace
    partial: Path | None = None

    def rename_then_interrupt(source: Path, destination: Path) -> None:
        nonlocal partial
        original_rename(source, destination)
        partial = source
        partial.mkdir()
        (partial / "sentinel").write_text("concurrent interrupt owner", encoding="utf-8")
        raise KeyboardInterrupt("injected after rename")

    monkeypatch.setattr(selection_module, "_rename_no_replace", rename_then_interrupt)
    with pytest.raises(KeyboardInterrupt, match="injected after rename") as caught:
        publish_prompt_view_bundle(bundle, output)

    assert partial is not None
    recovery = selection_module.prompt_publication_recovery_state(caught.value)
    published_identity = output.stat(follow_symlinks=False)
    partial_identity = partial.stat(follow_symlinks=False)
    assert type(caught.value) is KeyboardInterrupt
    assert recovery.phase.value == "rename"
    assert recovery.expected_artifact_identity is not None
    assert recovery.expected_artifact_identity.inode == published_identity.st_ino
    assert recovery.destination_observation.status == "present"
    assert recovery.destination_observation.inode == published_identity.st_ino
    assert recovery.partial_observation.status == "present"
    assert recovery.partial_observation.inode == partial_identity.st_ino
    assert (partial / "sentinel").read_text(encoding="utf-8") == "concurrent interrupt owner"
    assert (output / "SELECTION_MANIFEST.json").is_file()


def test_postrename_fsync_failure_preserves_a_concurrent_winner_for_recovery(
    tmp_path: Path, monkeypatch
) -> None:
    bundle = select_prompt_views(
        _inventory(),
        _policy(),
        baseline_receipt_sha256=BASELINE_RECEIPT_SHA256,
        held_out_receipt_sha256=HELD_OUT_RECEIPT_SHA256,
    )

    output = tmp_path / "fsync"
    displaced = tmp_path / "published-before-swap"
    original_fsync = selection_module._fsync_directory
    original_rename = selection_module._rename_no_replace
    former_partial: Path | None = None

    def rename_then_recreate_partial(source: Path, destination: Path) -> None:
        nonlocal former_partial
        original_rename(source, destination)
        former_partial = source
        former_partial.mkdir()
        (former_partial / "sentinel").write_text(
            "concurrent former-partial owner", encoding="utf-8"
        )

    def fail_parent(path: Path) -> None:
        if path == output.parent:
            output.rename(displaced)
            output.mkdir()
            (output / "sentinel").write_text("concurrent winner", encoding="utf-8")
            raise OSError("parent fsync failed")
        original_fsync(path)

    monkeypatch.setattr(selection_module, "_fsync_directory", fail_parent)
    monkeypatch.setattr(selection_module, "_rename_no_replace", rename_then_recreate_partial)
    with pytest.raises(PromptPublicationDurabilityError) as caught:
        publish_prompt_view_bundle(bundle, output)

    recovery = caught.value
    recovery_state = selection_module.prompt_publication_recovery_state(recovery)
    assert former_partial is not None
    displaced_identity = displaced.stat(follow_symlinks=False)
    partial_identity = former_partial.stat(follow_symlinks=False)
    destination_identity = output.stat(follow_symlinks=False)
    displaced_manifest = json.loads((displaced / "SELECTION_MANIFEST.json").read_bytes())
    assert (output / "sentinel").read_text(encoding="utf-8") == "concurrent winner"
    assert (former_partial / "sentinel").read_text(encoding="utf-8") == (
        "concurrent former-partial owner"
    )
    assert recovery.__cause__ is not None
    assert str(recovery.__cause__) == "parent fsync failed"
    assert recovery.recovery_required is True
    assert recovery.destination == output
    assert recovery.expected_identity == (
        displaced_identity.st_dev,
        displaced_identity.st_ino,
    )
    assert recovery.expected_device == displaced_identity.st_dev
    assert recovery.expected_inode == displaced_identity.st_ino
    assert recovery.expected_root_sha256 == displaced_manifest["root_sha256"]
    assert recovery_state.phase.value == "parent_fsync"
    assert recovery_state.partial_path == former_partial
    assert recovery_state.destination_path == output
    assert recovery_state.expected_artifact_identity is not None
    assert recovery_state.expected_artifact_identity.device == displaced_identity.st_dev
    assert recovery_state.expected_artifact_identity.inode == displaced_identity.st_ino
    assert (
        recovery_state.expected_artifact_identity.root_sha256 == displaced_manifest["root_sha256"]
    )
    assert recovery_state.partial_observation.status == "present"
    assert recovery_state.partial_observation.inode == partial_identity.st_ino
    assert recovery_state.destination_observation.status == "present"
    assert recovery_state.destination_observation.inode == destination_identity.st_ino
    assert recovery.receipt == recovery_state.receipt
    assert recovery.receipt["phase"] == "parent_fsync"
    assert recovery.receipt["partial_path"] == str(former_partial)
    assert recovery.receipt["destination_path"] == str(output)
    assert "independently" in recovery.verification_instructions.lower()
    assert "without following symlinks" in recovery.verification_instructions
    assert set(tmp_path.glob(".fsync.partial-*")) == {former_partial}

    monkeypatch.setattr(selection_module, "_fsync_directory", original_fsync)
    before_retry = set(tmp_path.glob(".fsync.partial-*"))
    with pytest.raises(FileExistsError) as retry_caught:
        publish_prompt_view_bundle(bundle, output)
    retry_recovery = selection_module.prompt_publication_recovery_state(retry_caught.value)
    after_retry = set(tmp_path.glob(".fsync.partial-*"))
    assert retry_recovery.phase.value == "rename"
    assert retry_recovery.destination_observation.status == "present"
    assert retry_recovery.destination_observation.inode == destination_identity.st_ino
    assert before_retry < after_retry
    assert (output / "sentinel").read_text(encoding="utf-8") == "concurrent winner"
    assert (former_partial / "sentinel").read_text(encoding="utf-8") == (
        "concurrent former-partial owner"
    )
    assert former_partial in after_retry


def test_inspector_accepts_exact_odd_d_lane_bucket_floor_proofs(tmp_path: Path) -> None:
    policy = _policy()
    d = policy.arms["D"]
    cells = dict(d.cells)
    cells["swe-agentic-tool"] = PromptCell(9)
    odd_d = replace(
        d,
        prompt_count=289,
        cells=MappingProxyType(cells),
        lanes=MappingProxyType(dict.fromkeys(d.lanes, 3)),
    )
    bundle = select_prompt_views(
        _inventory(),
        replace(policy, arms=MappingProxyType({**policy.arms, "D": odd_d})),
        baseline_receipt_sha256=BASELINE_RECEIPT_SHA256,
        held_out_receipt_sha256=HELD_OUT_RECEIPT_SHA256,
    )
    published = publish_prompt_view_bundle(bundle, tmp_path / "odd")

    result = inspect_prompt_manifest(
        published.manifest_path,
        expected_root_sha256=published.root_sha256,
        arm="D",
        limit=1,
    )
    assert result["total_matches"] == 351


def test_canonical_rows_preserve_content_and_cli_is_deterministic(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    published = _publish(tmp_path / "selection")
    result = inspect_prompt_manifest(
        published.manifest_path,
        expected_root_sha256=published.root_sha256,
        arm="D",
        offset=7,
        limit=3,
    )
    for row in result["rows"]:
        assert (
            hashlib.sha256(canonical_json(row["canonical_prompt"])).hexdigest()
            == row["prompt_uuid"]
        )
        assert {
            "source_row_index",
            "candidate_rank",
            "selection_index",
            "canonical_prompt",
        } <= row.keys()
    arguments = [
        str(published.manifest_path),
        "--expected-root-sha256",
        published.root_sha256,
        "--arm",
        "D",
        "--offset",
        "7",
        "--limit",
        "3",
    ]
    assert main(arguments) == 0
    first = capsys.readouterr().out
    assert main(arguments) == 0
    second = capsys.readouterr().out
    assert first == second
    assert json.loads(first) == result
