# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Contracts for reproducible Q30T PTV2/PTV3 source-stage plans."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from types import ModuleType

ROOT = Path(__file__).resolve().parents[3]
MODULE_PATH = ROOT / "examples/dataset/generate_q30t_ptv23_stage_plans.py"
SOURCES_PATH = ROOT / "examples/dataset/qwen3_30ba3b_thinking_ptv23_complement_sources_v1.json"
PTV3_PLAN_PATH = ROOT / "examples/dataset/qwen3_30ba3b_thinking_ptv3_stage_subset_v1.json"
PTV2_INVENTORY_PATH = (
    ROOT / "examples/dataset/qwen3_30ba3b_thinking_ptv2_full201_stage_inventory_v1.json"
)
PTV2_REVISION = "5c89e01dd720ae0f4058445ed49c5fb68a03c76e"
SWE_V3_REVISION = "3f73de64c1fe928a8f538fe45ccc10c228cc4c6a"
PTYCHE_CONTAINER = (
    "/lustre/fsw/coreai_dlalgo_llm/users/sna/containers/nemo2606/"
    "nemo_rl_nightly_nemo2606_20260812_2574659.sqsh"
)
PTYCHE_CONTAINER_SHA256 = "ab3380e548e5c62aa0bbaeaba3d1b47896151868f74e5859ac4eb311f1a069ab"
PTV2_COUNTS = {
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
    spec = importlib.util.spec_from_file_location("q30t_stage_plans", MODULE_PATH)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(spec.name, None)
        raise
    return module


def _tree_record(path: str, size: int, sha256: str) -> dict[str, object]:
    return {
        "type": "file",
        "oid": "a" * 40,
        "size": size,
        "lfs": {"oid": sha256, "size": size, "pointerSize": 130},
        "path": path,
    }


def _swe_tree() -> list[dict[str, object]]:
    return [
        _tree_record(
            f"data/train-{index:05d}-of-00096.parquet",
            index + 1,
            f"{index + 1:064x}",
        )
        for index in range(96)
    ]


def _ptv2_tree(*, redacted: bool) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    ordinal = 0
    for split, count in PTV2_COUNTS.items():
        for index in range(count):
            ordinal += 1
            sha256 = "*" * 64 if redacted else f"{ordinal:064x}"
            records.append(
                _tree_record(
                    f"data/{split}-{index:05d}-of-{count:05d}.parquet",
                    ordinal,
                    sha256,
                )
            )
    records.append(
        _tree_record(
            "data/multilingual-00000-of-00001.parquet",
            999,
            "*" * 64 if redacted else "f" * 64,
        )
    )
    return records


def test_ptv3_plan_selects_all_swe_v3_shards_and_four_exact_trace_files() -> None:
    module = _load_module()
    sources = json.loads(SOURCES_PATH.read_text(encoding="utf-8"))

    plan = module.build_ptv3_stage_plan(
        _swe_tree(),
        sources,
        container={"path": "/lustre/container.sqsh", "sha256": "e" * 64},
    )

    assert plan["schema_version"] == 1
    assert plan["name"] == "qwen3-30ba3b-thinking-ptv3-stage-subset-v1"
    assert len(plan["files"]) == 100
    assert sum(record["bytes"] for record in plan["files"][:96]) == sum(range(1, 97))
    assert {
        (record["source_id"], record["revision"], record["path"]) for record in plan["files"][96:]
    } == {
        (
            "nvidia/Nemotron-SFT-SWE-v2",
            "bd151f3f2d89c4804dda0083d912bd9f6a0a9fb7",
            "data/swe.jsonl",
        ),
        (
            "nvidia/Nemotron-SWE-v1",
            "0fe17a965b297a9c943a59050a14c42d5f0083ce",
            "data/r2e_gym.jsonl",
        ),
        (
            "nvidia/Nemotron-Agentic-v1",
            "650d590978ca35c8f1ecea2faf136e5fac421b62",
            "data/interactive_agent.jsonl",
        ),
        (
            "nvidia/Nemotron-Agentic-v1",
            "650d590978ca35c8f1ecea2faf136e5fac421b62",
            "data/tool_calling.jsonl",
        ),
    }
    assert {(record["path"], record["category"]) for record in plan["files"][96:]} == {
        ("data/swe.jsonl", "ptv3_interactive_agentic_swe"),
        ("data/r2e_gym.jsonl", "ptv3_interactive_agentic_swe"),
        ("data/interactive_agent.jsonl", "ptv3_general_tool_trajectories"),
        ("data/tool_calling.jsonl", "ptv3_general_tool_trajectories"),
    }
    assert all(record["revision"] == SWE_V3_REVISION for record in plan["files"][:96])
    assert all(record["license"] == "CC-BY-4.0" for record in plan["files"][:96])
    assert all(record["response_source"] == "trace-replay" for record in plan["files"])
    assert all(record["tool_lane"] == "recorded-trace" for record in plan["files"])


def test_ptv3_plan_rejects_missing_or_unpinned_swe_v3_files() -> None:
    module = _load_module()
    sources = json.loads(SOURCES_PATH.read_text(encoding="utf-8"))
    container = {"path": "/lustre/container.sqsh", "sha256": "e" * 64}

    with pytest.raises(ValueError, match="exactly 96"):
        module.build_ptv3_stage_plan(_swe_tree()[:-1], sources, container=container)

    tree = _swe_tree()
    tree[0]["lfs"] = {"oid": "*" * 64, "size": 1, "pointerSize": 130}
    with pytest.raises(ValueError, match="LFS SHA256"):
        module.build_ptv3_stage_plan(tree, sources, container=container)


def test_ptv2_inventory_declares_all_201_paths_and_sha_publication_blocker() -> None:
    module = _load_module()

    inventory = module.build_ptv2_stage_inventory(_ptv2_tree(redacted=True))

    assert inventory["repository"] == "nvidia/Nemotron-Post-Training-Dataset-v2"
    assert inventory["revision"] == PTV2_REVISION
    assert inventory["configuration"] == "default"
    assert inventory["license"] == "CC-BY-4.0"
    assert inventory["file_count"] == 201
    assert inventory["total_bytes"] == sum(range(1, 202))
    assert inventory["source_sha256_status"] == "gated-redacted-require-bootstrap"
    assert inventory["bootstrap_entrypoint"] == "examples/dataset/bootstrap_ptv2_source_manifest.py"
    assert [source["split"] for source in inventory["sources"]] == list(PTV2_COUNTS)
    assert all(
        record["sha256"] is None for source in inventory["sources"] for record in source["files"]
    )


def test_ptv2_inventory_rejects_incomplete_or_mixed_sha_metadata() -> None:
    module = _load_module()

    incomplete = _ptv2_tree(redacted=True)
    del incomplete[-2]
    with pytest.raises(ValueError, match="exactly 201"):
        module.build_ptv2_stage_inventory(incomplete)

    mixed = _ptv2_tree(redacted=True)
    mixed[0]["lfs"] = {"oid": "1" * 64, "size": 1, "pointerSize": 130}
    with pytest.raises(ValueError, match="uniformly redacted"):
        module.build_ptv2_stage_inventory(mixed)


def test_checked_in_stage_plans_match_official_aggregate_identity() -> None:
    module = _load_module()
    ptv3 = json.loads(PTV3_PLAN_PATH.read_text(encoding="utf-8"))
    ptv2 = json.loads(PTV2_INVENTORY_PATH.read_text(encoding="utf-8"))

    module.validate_ptv3_stage_plan(ptv3)
    module.validate_ptv2_stage_inventory(ptv2)
    assert ptv3["container"] == {
        "path": PTYCHE_CONTAINER,
        "sha256": PTYCHE_CONTAINER_SHA256,
    }
    assert len(ptv3["files"]) == 100
    assert sum(record["bytes"] for record in ptv3["files"][:96]) == 11_677_930_655
    assert sum(record["bytes"] for record in ptv3["files"]) == 39_956_713_421
    assert ptv2["file_count"] == 201
    assert ptv2["total_bytes"] == 44_423_886_661
