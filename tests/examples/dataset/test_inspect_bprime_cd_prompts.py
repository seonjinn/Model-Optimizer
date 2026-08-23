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
    from select_bprime_cd_prompts import publish_prompt_view_bundle, select_prompt_views
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


def test_publication_refuses_rename_race_without_replacing_destination(
    tmp_path: Path, monkeypatch
) -> None:
    bundle = select_prompt_views(
        _inventory(),
        _policy(),
        baseline_receipt_sha256=BASELINE_RECEIPT_SHA256,
        held_out_receipt_sha256=HELD_OUT_RECEIPT_SHA256,
    )
    output = tmp_path / "race"

    def race(source: Path, destination: Path) -> None:
        del source
        destination.mkdir()
        (destination / "sentinel").write_text("winner", encoding="utf-8")
        raise FileExistsError(destination)

    monkeypatch.setattr(selection_module, "_rename_no_replace", race, raising=False)
    with pytest.raises(FileExistsError):
        publish_prompt_view_bundle(bundle, output)

    assert (output / "sentinel").read_text(encoding="utf-8") == "winner"
    assert not list(tmp_path.glob(".race.partial-*"))


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
