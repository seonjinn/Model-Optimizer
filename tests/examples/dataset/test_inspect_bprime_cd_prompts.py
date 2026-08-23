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
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
MODULE_DIR = ROOT / "examples/dataset"
TEST_DIR = Path(__file__).parent

sys.path[:0] = [str(MODULE_DIR), str(TEST_DIR)]
try:
    from inspect_bprime_cd_prompts import inspect_prompt_manifest, main
    from select_bprime_cd_prompts import prompt_view_manifest, select_prompt_views
    from test_select_bprime_cd_prompts import _inventory, _policy
finally:
    del sys.path[:2]


def _write_manifest(path: Path, *, reverse: bool = False) -> dict:
    manifest = prompt_view_manifest(select_prompt_views(_inventory(reverse=reverse), _policy()))
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


def test_inspection_order_and_indexes_are_stable_across_inventory_order(tmp_path: Path) -> None:
    forward = tmp_path / "forward.json"
    reverse = tmp_path / "reverse.json"
    _write_manifest(forward)
    _write_manifest(reverse, reverse=True)

    first = inspect_prompt_manifest(forward, arm="C", offset=0, limit=1_000)
    second = inspect_prompt_manifest(reverse, arm="C", offset=0, limit=1_000)

    assert first == second
    assert [row["selection_index"] for row in first["rows"] if row["status"] == "primary"] == list(
        range(400)
    )
    assert [row["selection_index"] for row in first["rows"] if row["status"] == "reserve"] == list(
        range(80)
    )
    assert all(isinstance(row["candidate_rank"], int) for row in first["rows"])


def test_filters_are_exact_and_offset_limit_are_bounded(tmp_path: Path) -> None:
    path = tmp_path / "selection.json"
    _write_manifest(path)

    filtered = inspect_prompt_manifest(
        path,
        arm="B-prime",
        domain="multilingual",
        lane="target-synth",
        language="ja",
        offset=3,
        limit=2,
    )
    assert filtered["total_matches"] == 30
    assert len(filtered["rows"]) == 2
    assert all(
        row["arm"] == "B-prime"
        and row["domain"] == "multilingual"
        and row["lane"] == "target-synth"
        and row["language"] == "ja"
        for row in filtered["rows"]
    )
    prompt_uuid = filtered["rows"][0]["prompt_uuid"]
    exact = inspect_prompt_manifest(path, prompt_uuid=prompt_uuid, limit=10)
    assert {row["prompt_uuid"] for row in exact["rows"]} == {prompt_uuid}

    with pytest.raises(ValueError, match="offset"):
        inspect_prompt_manifest(path, offset=-1)
    with pytest.raises(ValueError, match="limit"):
        inspect_prompt_manifest(path, limit=0)
    with pytest.raises(ValueError, match="limit"):
        inspect_prompt_manifest(path, limit=1_001)


def test_malformed_and_tampered_manifests_are_rejected(tmp_path: Path) -> None:
    path = tmp_path / "selection.json"
    manifest = _write_manifest(path)
    manifest["arms"]["B-prime"]["rows"][0]["canonical_prompt"]["messages"][0]["content"] = (
        "tampered"
    )
    path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="artifact digest"):
        inspect_prompt_manifest(path)

    path.write_text("[]", encoding="utf-8")
    with pytest.raises(ValueError, match="manifest"):
        inspect_prompt_manifest(path)


def test_canonical_selected_rows_preserve_prompt_content_and_cli_is_deterministic(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    path = tmp_path / "selection.json"
    _write_manifest(path)
    result = inspect_prompt_manifest(path, arm="D", offset=7, limit=3)

    for row in result["rows"]:
        canonical = json.dumps(
            row["canonical_prompt"], ensure_ascii=False, sort_keys=True, separators=(",", ":")
        ).encode()
        assert hashlib.sha256(canonical).hexdigest() == row["prompt_uuid"]
        assert {
            "prompt_uuid",
            "arm",
            "domain",
            "lane",
            "language",
            "context_bucket",
            "source_id",
            "source_revision",
            "source_file_sha256",
            "source_manifest_sha256",
            "source_file_path",
            "source_row_index",
            "candidate_rank",
            "selection_index",
            "status",
            "canonical_prompt",
        } <= row.keys()

    arguments = [str(path), "--arm", "D", "--offset", "7", "--limit", "3"]
    assert main(arguments) == 0
    first = capsys.readouterr().out
    assert main(arguments) == 0
    second = capsys.readouterr().out
    assert first == second
    assert json.loads(first) == result
