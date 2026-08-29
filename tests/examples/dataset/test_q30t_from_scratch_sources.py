# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Contracts for the Q30 from-scratch physical source registry."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
MODULE_DIR = ROOT / "examples/dataset"

sys.path.insert(0, str(MODULE_DIR))
try:
    from q30t_from_scratch_sources import (  # pyright: ignore[reportMissingImports]
        SourceRegistryError,
        load_authenticated_source_registry,
        resolve_source_path,
    )
    from stage_ptv23_sources import (  # pyright: ignore[reportMissingImports]
        load_source_inventory,
        stage_source_inventory,
    )
finally:
    sys.path.pop(0)


def _canonical(value: object) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode()


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _requirements(source_ids: tuple[str, ...]) -> dict[str, object]:
    lanes = (
        "agentless_swe",
        "interactive_swe",
        "general_tool",
        "math",
        "code",
        "stem",
        "instruction",
        "multilingual",
    )
    return {
        "schema_version": "q30t-ptv23-from-scratch-source-requirements-v1",
        "scientific_identity": "q30t-ptv23-from-scratch-drafter-study-v1",
        "required_source_fields": [
            "repository",
            "revision",
            "configuration",
            "split",
            "relative_path",
            "bytes",
            "sha256",
            "row_count",
            "row_schema_sha256",
            "license_expression",
            "approved_use",
            "lane",
            "adapter",
        ],
        "source_families": {
            lane: list(source_ids) if lane == "math" else [] for lane in lanes
        },
        "adapters": {
            "ptv2_messages_tools_v1": {
                "messages_field": "messages",
                "tools_field": "tools",
            },
            "ptv3_messages_tools_v1": {
                "messages_field": "messages",
                "tools_field": "tools",
            },
        },
    }


def source_entry(source_id: str = "Nemotron-Math") -> dict[str, object]:
    """Return a hand-checked source fixture for a declared Math lane."""
    return {
        "source_id": source_id,
        "repository": f"nvidia/{source_id}",
        "configuration": "default",
        "split": "train",
        "revision": "a" * 40,
        "license_expression": "CC-BY-4.0",
        "adapter": "ptv3_messages_tools_v1",
    }


def _write_stage_receipt(root: Path, entry: dict[str, object], *, workers: int = 1) -> Path:
    source_root = root / "source-cache"
    source_path = source_root / str(entry["repository"]) / str(entry["revision"]) / "data/train.jsonl"
    source_path.parent.mkdir(parents=True)
    source_path.write_bytes(b'{"messages":[{"role":"user","content":"solve"}],"tools":[]}\n')
    source_manifest = {
        "schema_version": 1,
        "name": "fixture-source-stage",
        "sources": [
            {
                "repository_id": entry["repository"],
                "configuration": entry["configuration"],
                "split": entry["split"],
                "revision": entry["revision"],
                "license_expression": entry["license_expression"],
                "approved_use": True,
                "cell": entry["source_id"],
                "lane": "staging-only",
                "files": [
                    {
                        "path": "data/train.jsonl",
                        "bytes": source_path.stat().st_size,
                        "sha256": _sha256(source_path),
                    }
                ],
            }
        ],
    }
    manifest_path = root / "source-plan.json"
    manifest_path.write_bytes(_canonical(source_manifest))
    inventory = stage_source_inventory(
        load_source_inventory(manifest_path),
        durable_root=root / "durable",
        scratch_root=root / "scratch",
        local_source_root=source_root,
    )
    assert inventory.staged_root is not None
    inventory_receipt = inventory.staged_root / "SOURCE_INVENTORY.json"
    inventory_file = json.loads(inventory_receipt.read_text(encoding="utf-8"))["files"][0]
    row_schema = {
        "messages_field": "messages",
        "physical_format": "jsonl",
        "tools_field": "tools",
    }
    schema_receipt = {
        "schema_version": "q30t-ptv23-from-scratch-stage-receipt-v1",
        "source_inventory": {
            "path": "SOURCE_INVENTORY.json",
            "sha256": _sha256(inventory_receipt),
        },
        "workers": {"requested": workers, "effective": workers},
        "files": [
            {
                "source_id": entry["source_id"],
                "repository": entry["repository"],
                "configuration": entry["configuration"],
                "split": entry["split"],
                "revision": entry["revision"],
                "relative_path": "data/train.jsonl",
                "staged_path": inventory_file["staged_path"],
                "bytes": inventory_file["bytes"],
                "sha256": inventory_file["sha256"],
                "row_count": 1,
                "row_schema": row_schema,
                "row_schema_sha256": hashlib.sha256(
                    json.dumps(row_schema, sort_keys=True, separators=(",", ":")).encode()
                ).hexdigest(),
                "adapter": entry["adapter"],
                "lane": "math",
            }
        ],
    }
    receipt_path = inventory.staged_root / "ROW_SCHEMAS.json"
    receipt_path.write_bytes(_canonical(schema_receipt))
    return receipt_path


def load_test_registry(
    tmp_path: Path,
    *,
    entries: list[dict[str, object]],
    workers: int = 1,
    reverse_receipts: bool = False,
):
    """Build an independently hashed requirements fixture and staged receipts."""
    tmp_path.mkdir(parents=True, exist_ok=True)
    requirements_path = tmp_path / "requirements.json"
    requirements_path.write_bytes(
        _canonical(_requirements(tuple(str(item["source_id"]) for item in entries)))
    )
    receipts = [
        _write_stage_receipt(tmp_path / f"stage-{index}", entry, workers=workers)
        for index, entry in enumerate(entries)
    ]
    if reverse_receipts:
        receipts.reverse()
    return load_authenticated_source_registry(
        requirements_path,
        expected_sha256=_sha256(requirements_path),
        stage_receipts={_sha256(receipt): receipt for receipt in receipts},
    )


def load_mutated_registry(tmp_path: Path, mutation: str):
    """Build one staged source and apply one independent contract violation."""
    entry = source_entry()
    requirements_path = tmp_path / "requirements.json"
    requirements_path.write_bytes(_canonical(_requirements((str(entry["source_id"]),))))
    receipt_path = _write_stage_receipt(tmp_path / "stage", entry)
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    if mutation == "floating_revision":
        receipt["files"][0]["revision"] = "main"
    elif mutation == "unmapped_file":
        extra = dict(receipt["files"][0])
        extra["relative_path"] = "data/unmapped.jsonl"
        receipt["files"].append(extra)
    elif mutation == "row_schema_mismatch":
        receipt["files"][0]["row_schema_sha256"] = "0" * 64
    elif mutation == "missing_license":
        inventory_path = receipt_path.parent / "SOURCE_INVENTORY.json"
        inventory = json.loads(inventory_path.read_text(encoding="utf-8"))
        inventory["files"][0]["license_expression"] = ""
        inventory_path.write_bytes(_canonical(inventory))
        receipt["source_inventory"]["sha256"] = _sha256(inventory_path)
    else:
        raise AssertionError(f"unknown mutation: {mutation}")
    receipt_path.write_bytes(_canonical(receipt))
    return load_authenticated_source_registry(
        requirements_path,
        expected_sha256=_sha256(requirements_path),
        stage_receipts={_sha256(receipt_path): receipt_path},
    )


def test_registry_binds_every_physical_file_and_license(tmp_path: Path) -> None:
    """Changing a staged file or license binding must invalidate the registry."""
    registry = load_test_registry(tmp_path, entries=[source_entry()])
    entry = registry.entries[0]

    assert entry.file_sha256 == _sha256(resolve_source_path(registry, entry))
    assert entry.license_expression == "CC-BY-4.0"
    assert entry.approved_use == "specdec-drafter-training"
    assert entry.lane == "math"


@pytest.mark.parametrize(
    "mutation",
    ["floating_revision", "unmapped_file", "row_schema_mismatch", "missing_license"],
)
def test_registry_fails_closed_on_unbound_source_facts(tmp_path: Path, mutation: str) -> None:
    """Removing a pin, mapping, schema binding, or license must reject the source."""
    with pytest.raises(SourceRegistryError):
        load_mutated_registry(tmp_path, mutation)


def test_registry_digest_ignores_receipt_order_and_worker_count(tmp_path: Path) -> None:
    """Execution scheduling cannot alter the scientific source identity."""
    first = load_test_registry(
        tmp_path / "first",
        entries=[source_entry("Nemotron-Math"), source_entry("RL-Math")],
        workers=1,
    )
    second = load_test_registry(
        tmp_path / "second",
        entries=[source_entry("Nemotron-Math"), source_entry("RL-Math")],
        workers=8,
        reverse_receipts=True,
    )

    assert first.registry_sha256 == second.registry_sha256
    assert tuple(entry.source_id for entry in first.entries) == (
        "Nemotron-Math",
        "RL-Math",
    )
