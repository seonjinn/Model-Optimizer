# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Authenticated PTV2/PTV3 700K continuation corpus contracts."""

from __future__ import annotations

import ast
import ctypes
import errno
import hashlib
import importlib.util
import inspect
import json
import os
import subprocess
import sys
from contextlib import suppress
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
CONFIG_PATH = ROOT / "examples/dataset/qwen3_4b_ptv23_complement_700k_v1.json"
MODULE_PATH = ROOT / "examples/dataset/build_qwen4b_ptv23_complement.py"
SOURCE_REQUIREMENTS_PATH = ROOT / "examples/dataset/qwen3_4b_ptv23_complement_sources_v1.json"


def _load_module():
    assert MODULE_PATH.is_file(), "the continuation builder module must exist"
    spec = importlib.util.spec_from_file_location("build_qwen4b_ptv23_complement", MODULE_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    sys.path.insert(0, str(MODULE_PATH.parent))
    try:
        spec.loader.exec_module(module)
    finally:
        sys.path.pop(0)
    baseline_type = type(getattr(module, "EXPECTED_BASELINE"))
    setattr(
        module,
        "EXPECTED_BASELINE",
        baseline_type(
            "5c89e01dd720ae0f4058445ed49c5fb68a03c76e",
            1,
            {"chat": 3},
            2,
            4,
            1,
            "chat",
            True,
            "b" * 64,
        ),
    )
    return module


class _Tokenizer:
    def apply_chat_template(self, messages, **_kwargs):
        content = str(messages[-1]["content"])
        length = 9 if content == "long" else 4
        return {
            "input_ids": list(range(length)),
            "assistant_masks": [0] * (length - 2) + [1, 1],
        }


def _row(category: str, label: str, index: int) -> dict[str, object]:
    return {
        "category": category,
        "source_id": "fixture/source",
        "source_file_sha256": "a" * 64,
        "source_row_index": index,
        "messages": [
            {"role": "user", "content": f"prompt-{label}"},
            {"role": "assistant", "content": label},
        ],
        "tools": [],
    }


def _canonical(value: object) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()


def _write_inventory(path: Path, source_file: Path) -> tuple[Path, str]:
    row_schema = {
        "format": "jsonl",
        "messages_field": "messages",
        "tools_field": "tools",
    }
    raw = source_file.read_bytes()
    payload = {
        "schema_version": "ptv2-ptv3-complement-source-inventory-v1",
        "scientific_identity": "ptv2-ptv3-complement-700k-v1",
        "sources": [
            {
                "category": "stem",
                "source_id": "nvidia/Fixture",
                "revision": "a" * 40,
                "split": "train",
                "license_expression": "CC-BY-4.0",
                "approved_use": True,
                "replay_lane": "none",
                "row_schema": row_schema,
                "row_schema_sha256": hashlib.sha256(_canonical(row_schema)).hexdigest(),
                "files": [
                    {
                        "logical_path": "data/train.jsonl",
                        "path": str(source_file),
                        "bytes": len(raw),
                        "sha256": hashlib.sha256(raw).hexdigest(),
                        "row_count": 1,
                    }
                ],
            }
        ],
    }
    path.write_bytes(_canonical(payload) + b"\n")
    return path, hashlib.sha256(path.read_bytes()).hexdigest()


def _write_historical_receipt(path: Path) -> tuple[Path, str]:
    occurrences = ["1" * 64, "2" * 64, "1" * 64]
    unique = sorted(set(occurrences))
    body = {
        "source_revision": "5c89e01dd720ae0f4058445ed49c5fb68a03c76e",
        "source_manifest_sha256": "b" * 64,
        "row_count": 3,
        "split_rows": {"chat": 3},
        "files": [{"path": "/immutable/raw-chat.parquet", "bytes": 123, "sha256": "a" * 64}],
        "occurrence_count": len(occurrences),
        "unique_prompt_count": len(unique),
        "occurrence_prompt_ids": occurrences,
        "occurrence_prompt_ids_sha256": hashlib.sha256(_canonical(occurrences)).hexdigest(),
        "exclusion_prompt_ids": unique,
        "exclusion_prompt_ids_sha256": hashlib.sha256(_canonical(unique)).hexdigest(),
        "duplicate_uuid_multiplicity": {"1" * 64: 2},
        "physical_row_count": 4,
        "selection_policy": "hf-streaming-sorted-parquet-take",
        "selection_boundary": {
            "file": "chat-0.parquet",
            "rows_selected": 3,
            "rows_available": 4,
            "excluded_tail_rows": 1,
        },
    }
    payload = body | {"receipt_sha256": hashlib.sha256(_canonical(body)).hexdigest()}
    path.write_bytes(_canonical(payload) + b"\n")
    return path, hashlib.sha256(path.read_bytes()).hexdigest()


def _write_heldout_receipt(path: Path, values: list[str]) -> tuple[Path, str]:
    prompt_uuids = sorted(set(values))
    body = {
        "schema_version": "specdec-held-out-uuid-receipt-v1",
        "prompt_uuids": prompt_uuids,
        "prompt_uuids_sha256": hashlib.sha256(_canonical(prompt_uuids)).hexdigest(),
    }
    payload = body | {"receipt_sha256": hashlib.sha256(_canonical(body)).hexdigest()}
    path.write_bytes(_canonical(payload) + b"\n")
    return path, hashlib.sha256(path.read_bytes()).hexdigest()


def _write_tokenizer_trust(path: Path, snapshot: Path) -> tuple[Path, str]:
    entries = [
        [item.relative_to(snapshot).as_posix(), hashlib.sha256(item.read_bytes()).hexdigest()]
        for item in sorted(snapshot.rglob("*"))
        if item.is_file()
    ]
    body = {
        "schema_version": "qwen3-4b-tokenizer-trust-v1",
        "repository": "Qwen/Qwen3-4B",
        "revision": "1cfa9a7208912126459214e8b04321603b3df60c",
        "snapshot_path": str(snapshot),
        "snapshot_tree_sha256": hashlib.sha256(_canonical(entries)).hexdigest(),
        "chat_template_sha256": "6" * 64,
        "training_chat_template_sha256": "7" * 64,
        "im_start_token_id": 151644,
        "im_end_token_id": 151645,
    }
    payload = body | {"receipt_sha256": hashlib.sha256(_canonical(body)).hexdigest()}
    path.write_bytes(_canonical(payload) + b"\n")
    return path, hashlib.sha256(path.read_bytes()).hexdigest()


def _approved_q30_policy(module):
    policy = module.Q4_TARGET_POLICY
    return module.TargetTokenizerPolicy(
        schema_version=policy.schema_version,
        scientific_identity="ptv2-ptv3-complement-700k-qwen3-30ba3b-thinking-v1",
        target_repository="Qwen/Qwen3-30B-A3B-Thinking-2507",
        target_revision="a" * 40,
        tokenizer_repository="Qwen/Qwen3-30B-A3B-Thinking-2507",
        tokenizer_revision="a" * 40,
        tokenizer_trust_schema="qwen3-30ba3b-thinking-tokenizer-trust-v1",
        training_sequence_length=policy.training_sequence_length,
        quota_config_path=policy.quota_config_path,
        quota_config_sha256=policy.quota_config_sha256,
        source_requirements_path=policy.source_requirements_path,
        source_requirements_sha256=policy.source_requirements_sha256,
        file_sha256="b" * 64,
    )


def _q30_tokenizer_payload(tmp_path: Path) -> dict[str, object]:
    return {
        "schema_version": "qwen3-30ba3b-thinking-tokenizer-trust-v1",
        "repository": "Qwen/Qwen3-30B-A3B-Thinking-2507",
        "revision": "a" * 40,
        "snapshot_path": str(tmp_path / "snapshot"),
        "snapshot_tree_sha256": "1" * 64,
        "chat_template_sha256": "2" * 64,
        "training_chat_template_sha256": "3" * 64,
        "im_start_token_id": 151644,
        "im_end_token_id": 151645,
        "receipt_sha256": "4" * 64,
    }


def _toy_verifier_roots(
    module,
    monkeypatch,
    tmp_path: Path,
    rows_by_category: dict[str, list[dict[str, object]]],
    historical,
    held_out,
) -> dict[str, object]:
    monkeypatch.setattr(module, "_require_external_approval_roots", lambda: None)
    monkeypatch.setattr(module, "_authenticate_provenance_roots", lambda *_args: None)
    monkeypatch.setattr(
        module,
        "load_complement_config",
        lambda _path, **_kwargs: {"quotas": module.APPROVED_QUOTAS},
    )
    monkeypatch.setattr(module, "reconcile_source_requirements", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(module, "_category_rows", lambda _inventory: rows_by_category)
    inventory = module.SourceInventory(
        module.SCIENTIFIC_IDENTITY,
        "6" * 64,
        (),
    )
    return {
        "config_path": CONFIG_PATH,
        "inventory": inventory,
        "historical": historical,
        "held_out": held_out,
        "tokenizer_trust_file_sha256": "7" * 64,
        "scratch_root": tmp_path / "verify-scratch",
    }


def test_policy_pins_the_approved_scientific_identity_and_exact_quotas() -> None:
    assert CONFIG_PATH.is_file(), "the approved continuation policy must be versioned"
    payload = json.loads(CONFIG_PATH.read_bytes())

    assert payload["scientific_identity"] == "ptv2-ptv3-complement-700k-v1"
    assert payload["tokenizer"] == {
        "repository": "Qwen/Qwen3-4B",
        "revision": "1cfa9a7208912126459214e8b04321603b3df60c",
        "training_sequence_length": 4096,
        "trust_schema": "qwen3-4b-tokenizer-trust-v1",
    }
    assert payload["source_requirements"] == {
        "path": "qwen3_4b_ptv23_complement_sources_v1.json",
        "sha256": "e61ec87c2aba19c67c4a4549dba11f33abe8a3126f76232df0d72da43e8c81e1",
    }
    assert payload["quotas"] == {
        "ptv2_stem": 300_000,
        "ptv2_multilingual_ja": 50_000,
        "ptv2_multilingual_es": 50_000,
        "ptv2_multilingual_fr": 50_000,
        "ptv2_multilingual_it": 50_000,
        "ptv3_swe_v3": 100_000,
        "ptv3_interactive_agentic_swe": 19_000,
        "ptv3_general_tool_trajectories": 81_000,
    }
    assert sum(payload["quotas"].values()) == 700_000


def test_config_loader_rejects_any_noncanonical_policy_change(tmp_path: Path) -> None:
    module = _load_module()
    payload = json.loads(CONFIG_PATH.read_bytes())
    payload["quotas"]["ptv3_general_tool_trajectories"] -= 1
    changed = tmp_path / "changed.json"
    changed.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(module.ComplementError, match="approved continuation policy"):
        module.load_complement_config(changed)


def test_selection_uses_source_order_and_refills_only_from_the_same_category() -> None:
    module = _load_module()
    heldout_row = _row("stem", "heldout", 0)
    heldout_uuid = module.prompt_uuid_from_row(heldout_row)
    good_one = _row("stem", "one", 1)
    invalid = _row("stem", "invalid", 2)
    invalid["messages"] = [{"role": "user", "content": "missing assistant"}]
    good_two = _row("stem", "two", 4)
    duplicate = _row("tools", "one", 0)
    duplicate["messages"] = good_one["messages"]

    selection = module.select_continuation_rows(
        {
            "stem": [heldout_row, good_one, invalid, _row("stem", "long", 3), good_two],
            "tools": [duplicate, _row("tools", "three", 1)],
        },
        quotas={"stem": 2, "tools": 1},
        prior_prompt_uuids=set(),
        held_out_prompt_uuids={heldout_uuid},
        tokenizer=_Tokenizer(),
        training_sequence_length=8,
        replay_categories=frozenset(),
    )

    assert [row.messages[-1]["content"] for row in selection.rows] == ["one", "two", "three"]
    assert [row.source_row_index for row in selection.rows] == [1, 4, 1]
    assert selection.exclusions == {
        "duplicate": 1,
        "held_out": 1,
        "invalid": 1,
        "overlength": 1,
    }


def test_replay_validation_rejects_unresolved_calls_and_preserves_native_trace() -> None:
    module = _load_module()
    tools = [
        {
            "type": "function",
            "function": {
                "name": "shell",
                "description": "run a command",
                "parameters": {"type": "object", "properties": {}},
            },
        }
    ]
    call = {
        "id": "call-7",
        "type": "function",
        "function": {"name": "shell", "arguments": "{}"},
    }
    unresolved = _row("agentic", "ignored", 0)
    unresolved["tools"] = tools
    unresolved["messages"] = [
        {"role": "user", "content": "inspect"},
        {
            "role": "assistant",
            "content": "",
            "reasoning_content": "I should inspect first.",
            "tool_calls": [call],
        },
    ]
    valid = _row("agentic", "ignored", 1)
    valid["tools"] = tools
    valid["messages"] = [
        {"role": "user", "content": "inspect"},
        {
            "role": "assistant",
            "content": "",
            "reasoning_content": "I should inspect first.",
            "tool_calls": [call],
        },
        {"role": "tool", "tool_call_id": "call-7", "name": "shell", "content": "ok"},
        {"role": "assistant", "content": "done", "reasoning_content": "The result is clear."},
    ]

    selection = module.select_continuation_rows(
        {"agentic": [unresolved, valid]},
        quotas={"agentic": 1},
        prior_prompt_uuids=set(),
        held_out_prompt_uuids=set(),
        tokenizer=_Tokenizer(),
        training_sequence_length=8,
        replay_categories=frozenset({"agentic"}),
    )

    assert selection.rows[0].messages == valid["messages"]
    assert selection.rows[0].tools == tools
    assert selection.exclusions == {"unresolved_tool_call": 1}


def test_source_inventory_authenticates_repo_revision_file_schema_and_order(
    tmp_path: Path,
) -> None:
    module = _load_module()
    source_file = tmp_path / "train.jsonl"
    source_file.write_bytes(_canonical(_row("stem", "one", 0)) + b"\n")
    inventory_path, inventory_file_sha256 = _write_inventory(
        tmp_path / "inventory.json", source_file
    )

    inventory = module.load_source_inventory(inventory_path, expected_sha256=inventory_file_sha256)

    assert inventory.scientific_identity == "ptv2-ptv3-complement-700k-v1"
    assert inventory.sources[0].source_id == "nvidia/Fixture"
    assert inventory.sources[0].revision == "a" * 40
    assert (
        inventory.sources[0].files[0].sha256 == hashlib.sha256(source_file.read_bytes()).hexdigest()
    )


def test_source_inventory_rejects_forged_physical_row_count(tmp_path: Path) -> None:
    module = _load_module()
    source_file = tmp_path / "train.jsonl"
    source_file.write_bytes(_canonical(_row("stem", "one", 0)) + b"\n")
    inventory_path, _ = _write_inventory(tmp_path / "inventory.json", source_file)
    payload = json.loads(inventory_path.read_bytes())
    payload["sources"][0]["files"][0]["row_count"] = 2
    inventory_path.write_bytes(_canonical(payload) + b"\n")

    with pytest.raises(module.ComplementError, match="physical row count"):
        module.load_source_inventory(
            inventory_path,
            expected_sha256=hashlib.sha256(inventory_path.read_bytes()).hexdigest(),
        )


def test_historical_exclusion_requires_caller_pin_and_binds_order_and_multiplicity(
    tmp_path: Path,
) -> None:
    module = _load_module()
    receipt_path, receipt_file_sha256 = _write_historical_receipt(tmp_path / "historical.json")

    receipt = module.load_historical_exclusion(
        receipt_path,
        expected_sha256=receipt_file_sha256,
        expected_occurrence_count=3,
    )

    assert receipt.prompt_uuids == frozenset({"1" * 64, "2" * 64})
    assert (
        receipt.ordered_prompt_uuids_sha256
        == hashlib.sha256(_canonical(["1" * 64, "2" * 64, "1" * 64])).hexdigest()
    )
    assert receipt.duplicate_uuid_multiplicity == {"1" * 64: 2}


def test_builder_authenticates_original_baseline_audit_without_derivative_schema(
    tmp_path: Path,
) -> None:
    module = _load_module()
    audit_path, file_sha256 = _write_historical_receipt(tmp_path / "AUDIT.json")
    payload = json.loads(audit_path.read_bytes())
    directory_entries = set(tmp_path.iterdir())

    loaded = module.load_historical_exclusion(
        audit_path,
        expected_sha256=file_sha256,
        expected_occurrence_count=3,
    )

    assert loaded.file_sha256 == file_sha256
    assert loaded.receipt_sha256 == payload["receipt_sha256"]
    assert loaded.ordered_prompt_uuids_sha256 == payload["occurrence_prompt_ids_sha256"]
    assert loaded.prompt_uuids == frozenset(payload["exclusion_prompt_ids"])
    assert set(tmp_path.iterdir()) == directory_entries


def test_held_out_receipts_are_individually_authenticated_then_unioned(tmp_path: Path) -> None:
    module = _load_module()
    first = _write_heldout_receipt(tmp_path / "first.json", ["3" * 64, "4" * 64])
    second = _write_heldout_receipt(tmp_path / "second.json", ["4" * 64, "5" * 64])

    union = module.load_held_out_union([first, second], required_names=frozenset())

    assert union.prompt_uuids == frozenset({"3" * 64, "4" * 64, "5" * 64})
    assert union.receipt_file_sha256s == (first[1], second[1])
    assert (
        union.prompt_uuids_sha256
        == hashlib.sha256(_canonical(sorted(union.prompt_uuids))).hexdigest()
    )


def test_held_out_union_uses_fixed_name_order_independent_of_argument_order(
    tmp_path: Path,
) -> None:
    module = _load_module()
    by_name = {
        name: _write_heldout_receipt(tmp_path / f"{name}.json", [f"{index:064x}"])
        for index, name in enumerate(("speed", "math", "code", "swe", "tool"), start=1)
    }
    argument_order = ("tool", "speed", "swe", "code", "math")

    union = module.load_held_out_union([(name, *by_name[name]) for name in argument_order])

    assert union.receipt_names == ("speed", "math", "code", "swe", "tool")
    assert union.receipt_file_sha256s == tuple(by_name[name][1] for name in union.receipt_names)


def test_held_out_union_rejects_duplicate_names_before_set_reconciliation(
    tmp_path: Path,
) -> None:
    module = _load_module()
    receipts = {
        name: _write_heldout_receipt(tmp_path / f"{name}.json", [f"{index:064x}"])
        for index, name in enumerate(("speed", "math", "code", "swe", "tool"), start=1)
    }
    arguments = [(name, *receipt) for name, receipt in receipts.items()]
    arguments.append(("speed", *receipts["speed"]))

    with pytest.raises(module.ComplementError, match="duplicate"):
        module.load_held_out_union(arguments)


def test_held_out_union_rejects_canonical_empty_named_receipt(tmp_path: Path) -> None:
    module = _load_module()
    receipts = {
        name: _write_heldout_receipt(
            tmp_path / f"{name}.json",
            [] if name == "speed" else [f"{index:064x}"],
        )
        for index, name in enumerate(("speed", "math", "code", "swe", "tool"), start=1)
    }

    with pytest.raises(module.ComplementError, match="cannot be empty"):
        module.load_held_out_union([(name, *receipt) for name, receipt in receipts.items()])


def test_held_out_union_rejects_empty_named_receipt_when_required_names_is_empty(
    tmp_path: Path,
) -> None:
    module = _load_module()
    empty = _write_heldout_receipt(tmp_path / "speed.json", [])

    with pytest.raises(module.ComplementError, match="cannot be empty"):
        module.load_held_out_union(
            [("speed", *empty)],
            required_names=frozenset(),
        )


def test_held_out_union_keeps_empty_unnamed_fixture_compatibility(tmp_path: Path) -> None:
    module = _load_module()
    empty = _write_heldout_receipt(tmp_path / "empty.json", [])

    union = module.load_held_out_union([empty], required_names=frozenset())

    assert union.prompt_uuids == frozenset()
    assert union.receipt_file_sha256s == (empty[1],)
    assert union.receipt_names == ("",)


def test_production_held_out_union_requires_exact_named_receipt_set() -> None:
    module = _load_module()

    with pytest.raises(module.ComplementError, match="required held-out receipt set"):
        module.load_held_out_union([])


def test_source_reconciliation_fails_until_external_approval_allowlist_is_pinned(
    tmp_path: Path,
) -> None:
    module = _load_module()
    source_file = tmp_path / "train.jsonl"
    source_file.write_bytes(_canonical(_row("stem", "one", 0)) + b"\n")
    inventory_path, inventory_sha256 = _write_inventory(tmp_path / "inventory.json", source_file)
    inventory = module.load_source_inventory(inventory_path, expected_sha256=inventory_sha256)

    with pytest.raises(module.ComplementError, match="external approval root"):
        module.reconcile_source_requirements(
            SOURCE_REQUIREMENTS_PATH,
            inventory=inventory,
            capacity_receipt_path=tmp_path / "capacity.json",
        )


def test_publication_hashes_large_evidence_without_materialized_lists() -> None:
    source = MODULE_PATH.read_text()

    assert "ordered_prompt_uuids = [" not in source
    assert "source_occurrences = [" not in source
    assert "token_evidence = [" not in source


def test_bundle_binds_canonical_bytes_order_quotas_occurrences_exclusions_and_trust(
    tmp_path: Path,
) -> None:
    module = _load_module()
    selection = module.select_continuation_rows(
        {"stem": [_row("stem", "one", 0), _row("stem", "two", 1)]},
        quotas={"stem": 2},
        prior_prompt_uuids=set(),
        held_out_prompt_uuids=set(),
        tokenizer=_Tokenizer(),
        training_sequence_length=8,
        replay_categories=frozenset(),
    )
    historical_path, historical_file_sha256 = _write_historical_receipt(
        tmp_path / "historical.json"
    )
    historical = module.load_historical_exclusion(
        historical_path,
        expected_sha256=historical_file_sha256,
        expected_occurrence_count=3,
    )
    held_out = module.load_held_out_union([], required_names=frozenset())
    completion = module.publish_selection_bundle(
        selection,
        output_root=tmp_path / "lustre/bundle",
        scratch_root=tmp_path / "raid",
        quotas={"stem": 2},
        config_file_sha256=hashlib.sha256(CONFIG_PATH.read_bytes()).hexdigest(),
        source_inventory_file_sha256="6" * 64,
        historical=historical,
        held_out=held_out,
        tokenizer_trust_file_sha256="7" * 64,
        runtime_sha256="8" * 64,
        source_commit="9" * 40,
        enforce_production_paths=False,
    )
    manifest_raw = (completion.output_root / "MANIFEST.json").read_bytes()
    manifest = json.loads(manifest_raw)

    assert manifest_raw == _canonical(manifest) + b"\n"
    assert manifest["row_count"] == 2
    assert manifest["quotas"] == {"stem": 2}
    assert (
        manifest["ordered_prompt_uuids_sha256"]
        == hashlib.sha256(_canonical([row.prompt_uuid for row in selection.rows])).hexdigest()
    )
    assert manifest["duplicate_uuid_multiplicity"] == {}
    assert manifest["historical"]["ordered_prompt_uuids_sha256"] == (
        historical.ordered_prompt_uuids_sha256
    )
    assert (
        manifest["source_occurrences_sha256"]
        == hashlib.sha256(
            _canonical(
                [
                    [row.source_id, row.source_file_sha256, row.source_row_index]
                    for row in selection.rows
                ]
            )
        ).hexdigest()
    )
    assert manifest["trust"]["runtime_sha256"] == "8" * 64
    assert manifest["trust"]["source_commit"] == "9" * 40


def test_verifier_replays_selected_rows_and_rejects_data_tampering(
    tmp_path: Path, monkeypatch
) -> None:
    module = _load_module()
    monkeypatch.setattr(module, "APPROVED_QUOTAS", {"stem": 1})
    monkeypatch.setattr(module, "REQUIRED_HELD_OUT_RECEIPT_NAMES", frozenset())
    selection = module.select_continuation_rows(
        {"stem": [_row("stem", "one", 0)]},
        quotas={"stem": 1},
        prior_prompt_uuids=set(),
        held_out_prompt_uuids=set(),
        tokenizer=_Tokenizer(),
        training_sequence_length=8,
        replay_categories=frozenset(),
        capacity_receipt_path=tmp_path / "capacity.json",
    )
    historical_path, historical_file_sha256 = _write_historical_receipt(
        tmp_path / "historical.json"
    )
    historical = module.load_historical_exclusion(
        historical_path,
        expected_sha256=historical_file_sha256,
        expected_occurrence_count=3,
    )
    held_out = module.load_held_out_union([], required_names=frozenset())
    completion = module.publish_selection_bundle(
        selection,
        output_root=tmp_path / "lustre/bundle",
        scratch_root=tmp_path / "raid",
        quotas={"stem": 1},
        config_file_sha256=hashlib.sha256(CONFIG_PATH.read_bytes()).hexdigest(),
        source_inventory_file_sha256="6" * 64,
        historical=historical,
        held_out=held_out,
        tokenizer_trust_file_sha256="7" * 64,
        runtime_sha256="8" * 64,
        source_commit="9" * 40,
        enforce_production_paths=False,
    )

    verifier_roots = _toy_verifier_roots(
        module,
        monkeypatch,
        tmp_path,
        {"stem": [_row("stem", "one", 0)]},
        historical,
        held_out,
    )
    module.verify_selection_bundle(
        completion.output_root,
        expected_manifest_file_sha256=completion.manifest_file_sha256,
        tokenizer=_Tokenizer(),
        **verifier_roots,
        expected_runtime_sha256="8" * 64,
        expected_source_commit="9" * 40,
    )
    with (completion.output_root / "DATA.jsonl").open("ab") as stream:
        stream.write(b"{}\n")

    with pytest.raises(module.ComplementError, match="data identity"):
        module.verify_selection_bundle(
            completion.output_root,
            expected_manifest_file_sha256=completion.manifest_file_sha256,
            tokenizer=_Tokenizer(),
            **verifier_roots,
            expected_runtime_sha256="8" * 64,
            expected_source_commit="9" * 40,
        )


def test_cli_exposes_authenticated_build_and_verify_modes() -> None:
    result = subprocess.run(
        [sys.executable, str(MODULE_PATH), "--help"],
        check=True,
        capture_output=True,
        text=True,
    )

    assert "{build,verify}" in result.stdout
    assert "--historical-receipt-sha256" in result.stdout
    assert "--source-inventory-sha256" in result.stdout
    assert "--tokenizer-trust-sha256" in result.stdout
    assert "--runtime-sha256" in result.stdout


def test_executable_defines_config_loader_before_main_guard() -> None:
    """The direct build path must not call a function defined after its guard."""
    tree = ast.parse(MODULE_PATH.read_text())
    loader_position = next(
        index
        for index, node in enumerate(tree.body)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == "load_complement_config"
    )
    guard_position = next(
        index
        for index, node in enumerate(tree.body)
        if isinstance(node, ast.If)
        and isinstance(node.test, ast.Compare)
        and isinstance(node.test.left, ast.Name)
        and node.test.left.id == "__name__"
    )

    assert loader_position < guard_position


def test_verifier_rejects_manifest_defined_toy_quota(tmp_path: Path) -> None:
    """The approved identity may never be reduced to attacker-selected toy quotas."""
    module = _load_module()
    selection = module.select_continuation_rows(
        {"stem": [_row("stem", "one", 0)]},
        quotas={"stem": 1},
        prior_prompt_uuids=set(),
        held_out_prompt_uuids=set(),
        tokenizer=_Tokenizer(),
        training_sequence_length=8,
        replay_categories=frozenset(),
    )
    historical_path, historical_file_sha256 = _write_historical_receipt(
        tmp_path / "historical.json"
    )
    completion = module.publish_selection_bundle(
        selection,
        output_root=tmp_path / "bundle",
        scratch_root=tmp_path / "scratch",
        quotas={"stem": 1},
        config_file_sha256="5" * 64,
        source_inventory_file_sha256="6" * 64,
        historical=module.load_historical_exclusion(
            historical_path,
            expected_sha256=historical_file_sha256,
            expected_occurrence_count=3,
        ),
        held_out=module.load_held_out_union([], required_names=frozenset()),
        tokenizer_trust_file_sha256="7" * 64,
        runtime_sha256="8" * 64,
        source_commit="9" * 40,
        enforce_production_paths=False,
    )

    with pytest.raises(module.ComplementError, match="approved quotas"):
        module.verify_selection_bundle(
            completion.output_root,
            expected_manifest_file_sha256=completion.manifest_file_sha256,
            tokenizer=_Tokenizer(),
            config_path=CONFIG_PATH,
            inventory=module.SourceInventory(module.SCIENTIFIC_IDENTITY, "6" * 64, ()),
            historical=module.load_historical_exclusion(
                historical_path,
                expected_sha256=historical_file_sha256,
                expected_occurrence_count=3,
            ),
            held_out=module.load_held_out_union([], required_names=frozenset()),
            tokenizer_trust_file_sha256="7" * 64,
            scratch_root=tmp_path / "verify-scratch",
            expected_runtime_sha256="8" * 64,
            expected_source_commit="9" * 40,
        )


def test_verifier_revalidates_agentic_trajectory_integrity(tmp_path: Path, monkeypatch) -> None:
    """Replay verification must reject an unresolved tool call in an installed bundle."""
    module = _load_module()
    category = "ptv3_interactive_agentic_swe"
    monkeypatch.setattr(module, "APPROVED_QUOTAS", {category: 1})
    monkeypatch.setattr(module, "REQUIRED_HELD_OUT_RECEIPT_NAMES", frozenset())
    tools = [
        {
            "type": "function",
            "function": {
                "name": "shell",
                "description": "run a command",
                "parameters": {"type": "object", "properties": {}},
            },
        }
    ]
    unresolved = _row(category, "ignored", 0)
    unresolved["tools"] = tools
    unresolved["messages"] = [
        {"role": "user", "content": "inspect"},
        {
            "role": "assistant",
            "content": "",
            "reasoning_content": "inspect first",
            "tool_calls": [
                {
                    "id": "call-7",
                    "type": "function",
                    "function": {"name": "shell", "arguments": "{}"},
                }
            ],
        },
    ]
    selection = module.select_continuation_rows(
        {category: [unresolved]},
        quotas={category: 1},
        prior_prompt_uuids=set(),
        held_out_prompt_uuids=set(),
        tokenizer=_Tokenizer(),
        training_sequence_length=8,
        replay_categories=frozenset(),
        capacity_receipt_path=tmp_path / "capacity.json",
    )
    historical_path, historical_sha256 = _write_historical_receipt(tmp_path / "historical.json")
    historical = module.load_historical_exclusion(
        historical_path,
        expected_sha256=historical_sha256,
        expected_occurrence_count=3,
    )
    held_out = module.load_held_out_union([], required_names=frozenset())
    completion = module.publish_selection_bundle(
        selection,
        output_root=tmp_path / "bundle",
        scratch_root=tmp_path / "scratch",
        quotas={category: 1},
        config_file_sha256=hashlib.sha256(CONFIG_PATH.read_bytes()).hexdigest(),
        source_inventory_file_sha256="6" * 64,
        historical=historical,
        held_out=held_out,
        tokenizer_trust_file_sha256="7" * 64,
        runtime_sha256="8" * 64,
        source_commit="9" * 40,
        enforce_production_paths=False,
    )

    verifier_roots = _toy_verifier_roots(
        module, monkeypatch, tmp_path, {category: [unresolved]}, historical, held_out
    )
    with pytest.raises(module.ComplementError, match=r"trajectory|insufficient eligible capacity"):
        module.verify_selection_bundle(
            completion.output_root,
            expected_manifest_file_sha256=completion.manifest_file_sha256,
            tokenizer=_Tokenizer(),
            **verifier_roots,
            expected_runtime_sha256="8" * 64,
            expected_source_commit="9" * 40,
        )


def test_source_requirements_must_be_reconciled_before_build() -> None:
    """The executable boundary must expose strict source-requirement reconciliation."""
    module = _load_module()

    assert hasattr(module, "reconcile_source_requirements")


def test_source_reconciliation_rejects_inventory_missing_known_pins(
    tmp_path: Path, monkeypatch
) -> None:
    module = _load_module()
    monkeypatch.setattr(module, "APPROVED_PTV3_SWE_SOURCE_PINS", (("x", "1" * 40, "2" * 64),))
    monkeypatch.setattr(module, "APPROVED_ROW_SCHEMA_SHA256S", frozenset({"2" * 64}))
    source_file = tmp_path / "train.jsonl"
    source_file.write_bytes(_canonical(_row("stem", "one", 0)) + b"\n")
    inventory_path, inventory_sha256 = _write_inventory(tmp_path / "inventory.json", source_file)
    inventory = module.load_source_inventory(inventory_path, expected_sha256=inventory_sha256)
    monkeypatch.setattr(module, "APPROVED_SOURCE_INVENTORY_FILE_SHA256", inventory_sha256)
    monkeypatch.setattr(module, "APPROVED_HISTORICAL_RECEIPT_FILE_SHA256", "3" * 64)
    monkeypatch.setattr(
        module,
        "APPROVED_HELD_OUT_RECEIPT_FILE_SHA256S",
        dict.fromkeys(module.REQUIRED_HELD_OUT_RECEIPT_NAMES, "4" * 64),
    )

    with pytest.raises(module.ComplementError, match="known source pin"):
        module.reconcile_source_requirements(
            SOURCE_REQUIREMENTS_PATH,
            inventory=inventory,
            capacity_receipt_path=tmp_path / "missing-capacity.json",
        )


def test_source_iteration_rejects_mutation_after_inventory_authentication(tmp_path: Path) -> None:
    """Rows may not be emitted under a stale authenticated source-file digest."""
    module = _load_module()
    source_file = tmp_path / "train.jsonl"
    source_file.write_bytes(_canonical(_row("stem", "one", 0)) + b"\n")
    inventory_path, inventory_sha256 = _write_inventory(tmp_path / "inventory.json", source_file)
    inventory = module.load_source_inventory(inventory_path, expected_sha256=inventory_sha256)
    source_file.write_bytes(_canonical(_row("stem", "two", 0)) + b"\n")

    with pytest.raises(module.ComplementError, match=r"changed|identity"):
        list(module._iter_source_file(inventory.sources[0], inventory.sources[0].files[0]))


def test_verifier_streams_data_instead_of_materializing_whole_file(
    tmp_path: Path, monkeypatch
) -> None:
    """DATA.jsonl verification must not use the whole-file byte loader."""
    module = _load_module()
    monkeypatch.setattr(module, "APPROVED_QUOTAS", {"stem": 1})
    monkeypatch.setattr(module, "REQUIRED_HELD_OUT_RECEIPT_NAMES", frozenset())
    selection = module.select_continuation_rows(
        {"stem": [_row("stem", "one", 0)]},
        quotas={"stem": 1},
        prior_prompt_uuids=set(),
        held_out_prompt_uuids=set(),
        tokenizer=_Tokenizer(),
        training_sequence_length=8,
        replay_categories=frozenset(),
        capacity_receipt_path=tmp_path / "capacity.json",
    )
    historical_path, historical_sha256 = _write_historical_receipt(tmp_path / "historical.json")
    historical = module.load_historical_exclusion(
        historical_path,
        expected_sha256=historical_sha256,
        expected_occurrence_count=3,
    )
    held_out = module.load_held_out_union([], required_names=frozenset())
    completion = module.publish_selection_bundle(
        selection,
        output_root=tmp_path / "bundle",
        scratch_root=tmp_path / "scratch",
        quotas={"stem": 1},
        config_file_sha256=hashlib.sha256(CONFIG_PATH.read_bytes()).hexdigest(),
        source_inventory_file_sha256="6" * 64,
        historical=historical,
        held_out=held_out,
        tokenizer_trust_file_sha256="7" * 64,
        runtime_sha256="8" * 64,
        source_commit="9" * 40,
        enforce_production_paths=False,
    )
    original = module._stable_regular_bytes

    def reject_data_materialization(path: Path, *, max_bytes: int = 512 * 1024 * 1024) -> bytes:
        if path.name == "DATA.jsonl":
            raise AssertionError("DATA.jsonl was materialized")
        return original(path, max_bytes=max_bytes)

    monkeypatch.setattr(module, "_stable_regular_bytes", reject_data_materialization)

    verifier_roots = _toy_verifier_roots(
        module,
        monkeypatch,
        tmp_path,
        {"stem": [_row("stem", "one", 0)]},
        historical,
        held_out,
    )
    module.verify_selection_bundle(
        completion.output_root,
        expected_manifest_file_sha256=completion.manifest_file_sha256,
        tokenizer=_Tokenizer(),
        **verifier_roots,
        expected_runtime_sha256="8" * 64,
        expected_source_commit="9" * 40,
    )


def test_verifier_rejects_bundle_root_swap_during_replay(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Verification remains bound to one held bundle root for manifest, capacity, and data."""
    module = _load_module()
    monkeypatch.setattr(module, "APPROVED_QUOTAS", {"stem": 1})
    monkeypatch.setattr(module, "REQUIRED_HELD_OUT_RECEIPT_NAMES", frozenset())
    rows = {"stem": [_row("stem", "one", 0)]}
    selection = module.select_continuation_rows(
        rows,
        quotas={"stem": 1},
        prior_prompt_uuids=set(),
        held_out_prompt_uuids=set(),
        tokenizer=_Tokenizer(),
        training_sequence_length=8,
        replay_categories=frozenset(),
        capacity_receipt_path=tmp_path / "capacity.json",
    )
    historical_path, historical_sha256 = _write_historical_receipt(tmp_path / "historical.json")
    historical = module.load_historical_exclusion(
        historical_path, expected_sha256=historical_sha256, expected_occurrence_count=3
    )
    held_out = module.load_held_out_union([], required_names=frozenset())
    completion = module.publish_selection_bundle(
        selection,
        output_root=tmp_path / "bundle",
        scratch_root=tmp_path / "scratch",
        quotas={"stem": 1},
        config_file_sha256=hashlib.sha256(CONFIG_PATH.read_bytes()).hexdigest(),
        source_inventory_file_sha256="6" * 64,
        historical=historical,
        held_out=held_out,
        tokenizer_trust_file_sha256="7" * 64,
        runtime_sha256="8" * 64,
        source_commit="9" * 40,
        enforce_production_paths=False,
    )
    replacement = tmp_path / "replacement"
    replacement.mkdir()
    for child in completion.output_root.iterdir():
        (replacement / child.name).write_bytes(child.read_bytes())
    displaced = tmp_path / "displaced"
    verifier_roots = _toy_verifier_roots(module, monkeypatch, tmp_path, rows, historical, held_out)
    original_select = module.select_continuation_rows
    swapped = False

    def select_then_swap(*args, **kwargs):
        nonlocal swapped
        replay = original_select(*args, **kwargs)
        if not swapped:
            swapped = True
            completion.output_root.rename(displaced)
            replacement.rename(completion.output_root)
        return replay

    monkeypatch.setattr(module, "select_continuation_rows", select_then_swap)

    with pytest.raises(module.ComplementError, match=r"bundle|root|identity|changed"):
        module.verify_selection_bundle(
            completion.output_root,
            expected_manifest_file_sha256=completion.manifest_file_sha256,
            tokenizer=_Tokenizer(),
            **verifier_roots,
            expected_runtime_sha256="8" * 64,
            expected_source_commit="9" * 40,
        )


def test_verifier_bundle_reader_requires_exact_nonblocking_single_link_tree() -> None:
    """FIFO, hardlink, and extra-name inputs fail before any bundle payload is consumed."""
    source = MODULE_PATH.read_text()
    reader = source[
        source.index("def _held_selection_bundle") : source.index("def verify_selection_bundle")
    ]
    verifier = source[
        source.index("def _verify_selection_bundle_with_replay_root") : source.index(
            "def _parse_args"
        )
    ]

    assert "O_NONBLOCK" in reader
    assert "st_nlink != 1" in reader
    assert "frozenset(os.listdir(" in reader
    assert "_held_selection_bundle(output_root)" in verifier
    assert "data_path.is_file()" not in verifier


def test_selection_does_not_advance_source_after_quota_is_full() -> None:
    """Capacity evidence must count exactly the rows needed to fill a quota."""
    module = _load_module()

    class CountingRows:
        def __init__(self) -> None:
            self.advances = 0

        def __iter__(self):
            for row in (_row("stem", "one", 0), _row("stem", "two", 1)):
                self.advances += 1
                yield row

    rows = CountingRows()
    selection = module.select_continuation_rows(
        {"stem": rows},
        quotas={"stem": 1},
        prior_prompt_uuids=set(),
        held_out_prompt_uuids=set(),
        tokenizer=_Tokenizer(),
        training_sequence_length=8,
        replay_categories=frozenset(),
    )

    assert len(selection.rows) == 1
    assert rows.advances == 1


def test_checked_source_requirements_pin_known_agentic_files_and_name_real_blockers() -> None:
    assert SOURCE_REQUIREMENTS_PATH.is_file()
    payload = json.loads(SOURCE_REQUIREMENTS_PATH.read_bytes())

    assert payload["scientific_identity"] == "ptv2-ptv3-complement-700k-v1"
    assert payload["known_pinned_sources"] == [
        {
            "category": "ptv3_interactive_agentic_swe",
            "repository": "nvidia/Nemotron-SFT-SWE-v2",
            "revision": "bd151f3f2d89c4804dda0083d912bd9f6a0a9fb7",
            "split": "openhands_swe",
            "path": "data/swe.jsonl",
            "bytes": 11350621642,
            "sha256": "e887bd7ff4bd11a187d45af2e46db4847bd982771a6493dbe07d83d915c35bde",
        },
        {
            "category": "ptv3_interactive_agentic_swe",
            "repository": "nvidia/Nemotron-SWE-v1",
            "revision": "0fe17a965b297a9c943a59050a14c42d5f0083ce",
            "split": "r2e_gym",
            "path": "data/r2e_gym.jsonl",
            "bytes": 11141242062,
            "sha256": "1e0fb6d9a8d955fb0f2160e44a4946e5f2c4eb3931e80dadb724ff823cdbc14c",
        },
        {
            "category": "ptv3_general_tool_trajectories",
            "repository": "nvidia/Nemotron-Agentic-v1",
            "revision": "650d590978ca35c8f1ecea2faf136e5fac421b62",
            "split": "interactive_agent",
            "path": "data/interactive_agent.jsonl",
            "bytes": 448570455,
            "sha256": "dcfeda22372fa707c979cab29ddfe896b89a933f15ed4acbb4f16e7e3787d9dd",
        },
        {
            "category": "ptv3_general_tool_trajectories",
            "repository": "nvidia/Nemotron-Agentic-v1",
            "revision": "650d590978ca35c8f1ecea2faf136e5fac421b62",
            "split": "tool_calling",
            "path": "data/tool_calling.jsonl",
            "bytes": 5338348607,
            "sha256": "f537a901d38a999627b8fe59e77a1007af0d79d71a892ad9a4a3d80456e5601b",
        },
    ]
    assert payload["blocking_external_pins"] == [
        "ptv2_candidate_file_inventory_and_sha256",
        "ptv3_swe_v3_authoritative_revision_file_inventory_and_sha256",
        "source_row_schema_sha256_for_every_file",
        "post_exclusion_post_tokenization_capacity_receipt_for_every_category",
    ]


def test_tokenizer_trust_reuses_the_caller_pinned_official_qwen_snapshot(tmp_path: Path) -> None:
    module = _load_module()
    snapshot = tmp_path / "tokenizer"
    snapshot.mkdir()
    (snapshot / "tokenizer.json").write_bytes(b"fixture")
    trust_path, trust_file_sha256 = _write_tokenizer_trust(
        tmp_path / "tokenizer-trust.json", snapshot
    )

    trust = module.load_tokenizer_trust(trust_path, expected_sha256=trust_file_sha256)

    assert trust.repository == "Qwen/Qwen3-4B"
    assert trust.revision == "1cfa9a7208912126459214e8b04321603b3df60c"
    assert trust.snapshot_path == snapshot
    assert trust.im_start_token_id == 151644
    assert trust.im_end_token_id == 151645


def test_q30_builder_delegates_to_controller_owned_tokenizer_loader(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _load_module()
    receipt_path = tmp_path / "tokenizer.json"
    receipt_path.write_bytes(b"controller-owned\n")
    expected_sha256 = "5ba642c455e60b67eca295dce92dd7da47292fdba66c5f9d269669c14cafc509"
    calls: list[tuple[Path, str]] = []

    def approved_loader(path: Path, *, expected_sha256: str) -> dict[str, object]:
        calls.append((path, expected_sha256))
        return _q30_tokenizer_payload(tmp_path)

    monkeypatch.setattr(module, "load_q30t_tokenizer_receipt", approved_loader)
    trust = module.load_tokenizer_trust(
        receipt_path,
        expected_sha256=expected_sha256,
        policy=_approved_q30_policy(module),
    )

    assert calls == [(receipt_path, expected_sha256)]
    assert trust.file_sha256 == expected_sha256


def test_q30_builder_translates_loader_rejection_without_fallback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _load_module()
    calls = 0

    def rejected_loader(path: Path, *, expected_sha256: str) -> dict[str, object]:
        nonlocal calls
        calls += 1
        raise ValueError("Q30 tokenizer receipt is not approved")

    monkeypatch.setattr(module, "load_q30t_tokenizer_receipt", rejected_loader)
    with pytest.raises(module.ComplementError, match="not approved"):
        module.load_tokenizer_trust(
            tmp_path / "receipt.json",
            expected_sha256="5ba642c455e60b67eca295dce92dd7da47292fdba66c5f9d269669c14cafc509",
            policy=_approved_q30_policy(module),
        )

    assert calls == 1


def test_q30_builder_rejects_an_unavailable_controller_loader(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _load_module()
    monkeypatch.setattr(module, "load_q30t_tokenizer_receipt", None)

    with pytest.raises(module.ComplementError, match="Q30 tokenizer loader is unavailable"):
        module.load_tokenizer_trust(
            tmp_path / "receipt.json",
            expected_sha256="5ba642c455e60b67eca295dce92dd7da47292fdba66c5f9d269669c14cafc509",
            policy=_approved_q30_policy(module),
        )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("receipt_sha256", "A" * 64),
        ("repository", ""),
        ("revision", 17),
        ("snapshot_path", Path("snapshot")),
        ("snapshot_tree_sha256", "1" * 63),
        ("chat_template_sha256", None),
        ("training_chat_template_sha256", b"3" * 64),
        ("im_start_token_id", True),
        ("im_end_token_id", -1),
    ],
)
def test_q30_builder_rejects_malformed_controller_payload(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    field: str,
    value: object,
) -> None:
    module = _load_module()
    payload = _q30_tokenizer_payload(tmp_path)
    payload[field] = value
    monkeypatch.setattr(
        module,
        "load_q30t_tokenizer_receipt",
        lambda _path, *, expected_sha256: payload,
    )

    with pytest.raises(module.ComplementError, match=rf"Q30 tokenizer {field} is invalid"):
        module.load_tokenizer_trust(
            tmp_path / "receipt.json",
            expected_sha256="5ba642c455e60b67eca295dce92dd7da47292fdba66c5f9d269669c14cafc509",
            policy=_approved_q30_policy(module),
        )


def test_production_selection_spools_selected_conversations_to_raid_backed_sqlite(
    tmp_path: Path,
) -> None:
    module = _load_module()
    spool_path = tmp_path / "raid/selection.sqlite"

    selection = module.select_continuation_rows(
        {"stem": [_row("stem", "one", 0), _row("stem", "two", 1)]},
        quotas={"stem": 2},
        prior_prompt_uuids=set(),
        held_out_prompt_uuids=set(),
        tokenizer=_Tokenizer(),
        training_sequence_length=8,
        replay_categories=frozenset(),
        spool_path=spool_path,
    )

    assert spool_path.is_file()
    assert not isinstance(selection.rows, tuple)
    assert [row.messages[-1]["content"] for row in selection.rows] == ["one", "two"]


def test_capacity_failure_persists_exact_same_category_blocker_without_redistribution(
    tmp_path: Path,
) -> None:
    module = _load_module()
    receipt_path = tmp_path / "capacity.json"

    with pytest.raises(module.ComplementError, match="insufficient eligible capacity for stem"):
        module.select_continuation_rows(
            {"stem": [_row("stem", "one", 0)], "tools": [_row("tools", "two", 0)]},
            quotas={"stem": 2, "tools": 1},
            prior_prompt_uuids=set(),
            held_out_prompt_uuids=set(),
            tokenizer=_Tokenizer(),
            training_sequence_length=8,
            replay_categories=frozenset(),
            capacity_receipt_path=receipt_path,
        )

    payload = json.loads(receipt_path.read_bytes())
    assert receipt_path.read_bytes() == _canonical(payload) + b"\n"
    assert payload["status"] == "insufficient-capacity"
    assert payload["blocking_category"] == "stem"
    assert payload["required"] == 2
    assert payload["selected"] == 1
    assert payload["redistribution"] == "forbidden"


def test_verifier_requires_live_external_provenance_and_exclusion_roots() -> None:
    """Self-claimed manifest hashes cannot authenticate source or exclusion semantics."""
    module = _load_module()
    parameters = inspect.signature(module.verify_selection_bundle).parameters

    assert {
        "config_path",
        "inventory",
        "historical",
        "held_out",
        "tokenizer_trust_file_sha256",
        "scratch_root",
    } <= set(parameters)


def test_verifier_enforces_nonzero_assistant_supervision() -> None:
    """Replay must enforce the same nonzero assistant mask gate as selection."""
    source = MODULE_PATH.read_text()
    verifier = source[source.index("def verify_selection_bundle") : source.index("def _parse_args")]

    assert "assistant_tokens < 1" in verifier


def test_authenticated_builder_readers_use_one_nofollow_descriptor() -> None:
    """Trusted input bytes cannot be reopened through a swapped path."""
    source = MODULE_PATH.read_text()
    stable_bytes = source[
        source.index("def _stable_regular_bytes") : source.index("def _stable_regular_evidence")
    ]
    stable_evidence = source[
        source.index("def _stable_regular_evidence") : source.index("def _physical_row_count")
    ]

    for helper in (stable_bytes, stable_evidence):
        assert "os.open" in helper
        assert "O_NOFOLLOW" in helper
        assert "O_NONBLOCK" in helper
        assert ".read_bytes()" not in helper
        assert "path.open(" not in helper


def test_stable_regular_bytes_rejects_oversized_retained_input(tmp_path: Path) -> None:
    """Metadata readers bound retained bytes before allocating an artifact in memory."""
    module = _load_module()
    oversized = tmp_path / "oversized.json"
    oversized.write_bytes(b"123456789")

    with pytest.raises(module.ComplementError, match="too large"):
        module._stable_regular_bytes(oversized, max_bytes=8)


def test_build_fails_on_unresolved_approval_roots_before_selection() -> None:
    """A known policy blocker must fail before a 700K tokenize/select pass."""
    source = MODULE_PATH.read_text()
    build_branch = source[
        source.index('if args.command == "build"') : source.index(
            "return 0", source.index('if args.command == "build"')
        )
    ]

    assert build_branch.index("_require_external_approval_roots") < build_branch.index(
        "select_continuation_rows"
    )


def test_publication_rehashes_copied_partial_before_atomic_rename() -> None:
    """The bytes installed on shared storage must be authenticated after copying."""
    source = MODULE_PATH.read_text()
    publication = source[
        source.index("def publish_selection_bundle") : source.index("def verify_selection_bundle")
    ]

    assert "publication partial changed while copying" in publication


def test_bundle_publication_survives_lustre_renameat2_einval(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The production dataset bundle uses the shared Lustre-safe publication primitive."""

    class UnsupportedRename:
        argtypes: object = None
        restype: object = None

        def __call__(self, *_args: object) -> int:
            ctypes.set_errno(errno.EINVAL)
            return -1

    class UnsupportedLibrary:
        renameat2 = UnsupportedRename()
        renamex_np = UnsupportedRename()

    module = _load_module()
    partial = tmp_path / ".bundle.partial"
    partial.mkdir()
    (partial / "DATA.jsonl").write_bytes(b"data\n")
    destination = tmp_path / "bundle"
    monkeypatch.setattr(ctypes, "CDLL", lambda *_args, **_kwargs: UnsupportedLibrary())

    module._rename_noreplace(partial, destination)

    assert (destination / "DATA.jsonl").read_bytes() == b"data\n"


def test_exact_dataset_retry_authenticates_the_durable_completion_marker(
    tmp_path: Path,
) -> None:
    """A final bundle is adoptable only with the stable exact completed marker state."""
    module = _load_module()
    partial = tmp_path / ".bundle.partial"
    partial.mkdir()
    data_raw = b'{"row":1}\n'
    data_sha256 = hashlib.sha256(data_raw).hexdigest()
    manifest = {"data": {"path": "DATA.jsonl", "bytes": len(data_raw), "sha256": data_sha256}}
    manifest_raw = _canonical(manifest) + b"\n"
    (partial / "DATA.jsonl").write_bytes(data_raw)
    (partial / "MANIFEST.json").write_bytes(manifest_raw)
    root = tmp_path / "bundle"

    module._rename_noreplace(partial, root)

    assert module._published_bundle_matches(
        root,
        manifest_raw,
        data_sha256,
        "",
        require_completion_marker=True,
    )
    (root / module.Q30_DIRECTORY_COMPLETION_MARKER).write_bytes(b"foreign\n")
    assert not module._published_bundle_matches(
        root,
        manifest_raw,
        data_sha256,
        "",
        require_completion_marker=True,
    )


@pytest.mark.parametrize("name", ["DATA.jsonl", "MANIFEST.json", "CAPACITY.json"])
def test_exact_dataset_retry_rejects_multiply_linked_bundle_files(
    tmp_path: Path, name: str
) -> None:
    """Exact-byte retry adoption cannot authenticate a multiply-linked durable file."""
    module = _load_module()
    root = tmp_path / "bundle"
    root.mkdir()
    data_raw = b'{"row":1}\n'
    manifest_raw = b'{"manifest":1}\n'
    capacity_raw = b'{"capacity":1}\n'
    (root / "DATA.jsonl").write_bytes(data_raw)
    (root / "MANIFEST.json").write_bytes(manifest_raw)
    (root / "CAPACITY.json").write_bytes(capacity_raw)
    os.link(root / name, tmp_path / f"linked-{name}")

    assert not module._published_bundle_matches(
        root,
        manifest_raw,
        hashlib.sha256(data_raw).hexdigest(),
        hashlib.sha256(capacity_raw).hexdigest(),
    )


def test_exact_dataset_retry_reauthenticates_after_parent_fsync(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A same-byte bundle root replacement after parent fsync cannot be adopted."""
    module = _load_module()
    root = tmp_path / "bundle"
    root.mkdir()
    data_raw = b'{"row":1}\n'
    manifest_raw = b'{"manifest":1}\n'
    (root / "DATA.jsonl").write_bytes(data_raw)
    (root / "MANIFEST.json").write_bytes(manifest_raw)
    replacement = tmp_path / "replacement"
    replacement.mkdir()
    (replacement / "DATA.jsonl").write_bytes(data_raw)
    (replacement / "MANIFEST.json").write_bytes(manifest_raw)
    displaced = tmp_path / "displaced"
    parent_identity = (root.parent.stat().st_dev, root.parent.stat().st_ino)
    original_fsync = module.os.fsync
    swapped = False

    def fsync_then_swap(descriptor: int) -> None:
        nonlocal swapped
        original_fsync(descriptor)
        status = os.fstat(descriptor)
        if (status.st_dev, status.st_ino) == parent_identity and not swapped:
            swapped = True
            root.rename(displaced)
            replacement.rename(root)

    monkeypatch.setattr(module.os, "fsync", fsync_then_swap)

    assert not module._published_bundle_matches(
        root,
        manifest_raw,
        hashlib.sha256(data_raw).hexdigest(),
        "",
    )


def test_exact_dataset_retry_rejects_declared_size_before_hashing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A foreign DATA size is rejected from fstat before its payload can be streamed."""
    module = _load_module()
    root = tmp_path / "bundle"
    root.mkdir()
    data_raw = b'{"row":1}\n'
    data = root / "DATA.jsonl"
    data.write_bytes(data_raw)
    manifest = {
        "data": {
            "path": "DATA.jsonl",
            "bytes": len(data_raw) + 1,
            "sha256": hashlib.sha256(data_raw).hexdigest(),
        },
        "capacity": None,
    }
    manifest_raw = _canonical(manifest) + b"\n"
    (root / "MANIFEST.json").write_bytes(manifest_raw)
    data_identity = (data.stat().st_dev, data.stat().st_ino)
    original_fdopen = module.os.fdopen

    def reject_data_hash(descriptor: int, *args, **kwargs):
        status = os.fstat(descriptor)
        if (status.st_dev, status.st_ino) == data_identity:
            raise AssertionError("DATA was streamed before declared-size rejection")
        return original_fdopen(descriptor, *args, **kwargs)

    monkeypatch.setattr(module.os, "fdopen", reject_data_hash)

    assert not module._published_bundle_matches(
        root,
        manifest_raw,
        hashlib.sha256(data_raw).hexdigest(),
        "",
    )


def test_completion_receipt_adopts_an_exact_deterministic_retry(tmp_path: Path) -> None:
    """A completed receipt is reusable only as the same durable inode and exact bytes."""
    module = _load_module()
    completion = module.ComplementCompletion(
        output_root=tmp_path / "bundle",
        row_count=700_000,
        manifest_file_sha256="a" * 64,
        manifest_sha256="b" * 64,
        scratch_root=tmp_path / "scratch",
    )
    path = tmp_path / "receipts/COMPLETION.json"

    first = module._write_completion_receipt(
        path, completion, runtime_sha256="c" * 64, source_commit="d" * 40
    )
    before = path.stat()
    expected = path.read_bytes()
    second = module._write_completion_receipt(
        path, completion, runtime_sha256="c" * 64, source_commit="d" * 40
    )

    assert second == first
    assert path.read_bytes() == expected
    assert (path.stat().st_dev, path.stat().st_ino) == (before.st_dev, before.st_ino)


def test_completion_receipt_retry_reauthenticates_after_parent_fsync(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A same-byte pathname replacement during retry durability cannot be adopted."""
    module = _load_module()
    completion = module.ComplementCompletion(
        output_root=tmp_path / "bundle",
        row_count=700_000,
        manifest_file_sha256="a" * 64,
        manifest_sha256="b" * 64,
        scratch_root=tmp_path / "scratch",
    )
    path = tmp_path / "receipts/COMPLETION.json"
    module._write_completion_receipt(
        path, completion, runtime_sha256="c" * 64, source_commit="d" * 40
    )
    replacement = tmp_path / "replacement.json"
    replacement.write_bytes(path.read_bytes())
    displaced = tmp_path / "displaced.json"
    original_fsync = module.os.fsync
    parent_identity = (path.parent.stat().st_dev, path.parent.stat().st_ino)
    swap_enabled = True

    def fsync_then_swap(descriptor: int) -> None:
        nonlocal swap_enabled
        original_fsync(descriptor)
        status = os.fstat(descriptor)
        if swap_enabled and (status.st_dev, status.st_ino) == parent_identity:
            swap_enabled = False
            path.rename(displaced)
            replacement.rename(path)

    monkeypatch.setattr(module.os, "fsync", fsync_then_swap)

    with pytest.raises(module.ComplementError, match=r"changed|rebound"):
        module._write_completion_receipt(
            path, completion, runtime_sha256="c" * 64, source_commit="d" * 40
        )


def test_completion_receipt_retry_rejects_replacement_parent_with_same_inode(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Durability and final rebind use one held parent even if the receipt inode is moved."""
    module = _load_module()
    completion = module.ComplementCompletion(
        output_root=tmp_path / "bundle",
        row_count=700_000,
        manifest_file_sha256="a" * 64,
        manifest_sha256="b" * 64,
        scratch_root=tmp_path / "scratch",
    )
    path = tmp_path / "receipts/COMPLETION.json"
    module._write_completion_receipt(
        path, completion, runtime_sha256="c" * 64, source_commit="d" * 40
    )
    displaced_parent = tmp_path / "displaced-receipts"
    original_fsync = module.os.fsync
    parent_identity = (path.parent.stat().st_dev, path.parent.stat().st_ino)
    swapped = False

    def fsync_then_replace_parent(descriptor: int) -> None:
        nonlocal swapped
        original_fsync(descriptor)
        status = os.fstat(descriptor)
        if not swapped and (status.st_dev, status.st_ino) == parent_identity:
            swapped = True
            path.parent.rename(displaced_parent)
            path.parent.mkdir()
            (displaced_parent / path.name).rename(path)

    monkeypatch.setattr(module.os, "fsync", fsync_then_replace_parent)

    with pytest.raises(module.ComplementError, match=r"parent|changed|rebound|durability"):
        module._write_completion_receipt(
            path, completion, runtime_sha256="c" * 64, source_commit="d" * 40
        )


def test_completion_receipt_durability_uses_one_held_parent_descriptor() -> None:
    """One adoption owns file fsync, parent fsync, and the final absolute parent rebind."""
    source = MODULE_PATH.read_text()
    adoption = source[
        source.index("def _adopt_exact_completion_receipt") : source.index(
            "def _write_completion_receipt"
        )
    ]
    writer = source[
        source.index("def _write_completion_receipt") : source.index("def _load_completion_receipt")
    ]

    assert "os.fsync(parent_fd)" in adoption
    assert "os.stat(path.parent, follow_symlinks=False)" in adoption
    assert writer.count("_adopt_exact_completion_receipt") == 1
    assert "_fsync_directory(path.parent)" not in writer


def test_completion_receipt_retry_rejects_hardlinks_and_never_overwrites(
    tmp_path: Path,
) -> None:
    """Retry adoption fails closed on linked or conflicting names without replacing them."""
    module = _load_module()
    completion = module.ComplementCompletion(
        output_root=tmp_path / "bundle",
        row_count=700_000,
        manifest_file_sha256="a" * 64,
        manifest_sha256="b" * 64,
        scratch_root=tmp_path / "scratch",
    )
    linked = tmp_path / "linked/COMPLETION.json"
    module._write_completion_receipt(
        linked, completion, runtime_sha256="c" * 64, source_commit="d" * 40
    )
    linked_bytes = linked.read_bytes()
    os.link(linked, tmp_path / "receipt-hardlink.json")

    with pytest.raises(module.ComplementError, match="single-link"):
        module._write_completion_receipt(
            linked, completion, runtime_sha256="c" * 64, source_commit="d" * 40
        )
    assert linked.read_bytes() == linked_bytes

    conflicting = tmp_path / "conflicting/COMPLETION.json"
    conflicting.parent.mkdir()
    conflicting.write_bytes(b"foreign\n")
    with pytest.raises(module.ComplementError, match=r"differs|changed"):
        module._write_completion_receipt(
            conflicting, completion, runtime_sha256="c" * 64, source_commit="d" * 40
        )
    assert conflicting.read_bytes() == b"foreign\n"


def test_publication_reauthenticates_the_installed_destination(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A pathname replacement at rename time cannot become the completion identity."""
    module = _load_module()
    selection = module.select_continuation_rows(
        {"stem": [_row("stem", "one", 0)]},
        quotas={"stem": 1},
        prior_prompt_uuids=set(),
        held_out_prompt_uuids=set(),
        tokenizer=_Tokenizer(),
        training_sequence_length=8,
        replay_categories=frozenset(),
    )
    historical_path, historical_file_sha256 = _write_historical_receipt(
        tmp_path / "historical.json"
    )
    historical = module.load_historical_exclusion(
        historical_path,
        expected_sha256=historical_file_sha256,
        expected_occurrence_count=3,
    )
    held_out = module.load_held_out_union([], required_names=frozenset())

    def install_foreign_bundle(
        _source: Path,
        destination: Path,
        **_kwargs: object,
    ) -> None:
        destination.mkdir()
        (destination / "DATA.jsonl").write_bytes(b"foreign\n")
        (destination / "MANIFEST.json").write_bytes(b"{}\n")

    monkeypatch.setattr(module, "_rename_noreplace", install_foreign_bundle)
    with pytest.raises(module.ComplementError, match="destination changed after install"):
        module.publish_selection_bundle(
            selection,
            output_root=tmp_path / "bundle",
            scratch_root=tmp_path / "scratch",
            quotas={"stem": 1},
            config_file_sha256=hashlib.sha256(CONFIG_PATH.read_bytes()).hexdigest(),
            source_inventory_file_sha256="6" * 64,
            historical=historical,
            held_out=held_out,
            tokenizer_trust_file_sha256="7" * 64,
            runtime_sha256="8" * 64,
            source_commit="9" * 40,
            enforce_production_paths=False,
        )


def test_tokenizer_is_loaded_only_from_a_fresh_verified_snapshot_stage() -> None:
    source = MODULE_PATH.read_text()
    loader = source[
        source.index("def _load_qwen_tokenizer") : source.index("def _authenticated_source_stream")
    ]

    assert "_stage_tokenizer_snapshot" in loader
    assert "staged_snapshot" in loader


def test_tokenizer_stage_failure_never_deletes_a_reused_foreign_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A same-UID replacement of private scratch is preserved on staging failure."""
    module = _load_module()
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    trust = module.TokenizerTrust(
        file_sha256="a" * 64,
        receipt_sha256="b" * 64,
        repository=module.Q30T_TOKENIZER_REPOSITORY,
        revision="c" * 40,
        snapshot_path=snapshot,
        snapshot_tree_sha256="d" * 64,
        chat_template_sha256="e" * 64,
        training_chat_template_sha256="f" * 64,
        im_start_token_id=1,
        im_end_token_id=2,
    )
    foreign_marker = tmp_path / "foreign-marker"

    def replace_stage(*_args: object, copy_root: Path, **_kwargs: object) -> None:
        stage_root = copy_root.parent
        displaced = tmp_path / "displaced-stage"
        stage_root.rename(displaced)
        stage_root.mkdir()
        marker = stage_root / "foreign"
        marker.write_bytes(b"preserve-me\n")
        foreign_marker.write_text(str(marker))
        raise ValueError("injected staging failure")

    monkeypatch.setattr(module, "descriptor_stable_tree_entries", replace_stage)
    monkeypatch.setattr(module, "canonical_tree_sha256", lambda _entries: "d" * 64)

    with pytest.raises(module.ComplementError, match="staging failed"):
        module._stage_tokenizer_snapshot(trust, tmp_path / "scratch")

    marker = Path(foreign_marker.read_text())
    assert marker.read_bytes() == b"preserve-me\n"


def test_tokenizer_stage_failure_never_unlinks_a_replaced_child_after_stat(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Private scratch retention avoids the irreducible stat-to-unlink child race."""
    module = _load_module()
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    trust = module.TokenizerTrust(
        file_sha256="a" * 64,
        receipt_sha256="b" * 64,
        repository=module.Q30T_TOKENIZER_REPOSITORY,
        revision="c" * 40,
        snapshot_path=snapshot,
        snapshot_tree_sha256="d" * 64,
        chat_template_sha256="e" * 64,
        training_chat_template_sha256="f" * 64,
        im_start_token_id=1,
        im_end_token_id=2,
    )
    staged_snapshot: Path | None = None
    real_stat = module.os.stat
    real_open = module.os.open
    real_rename = module.os.rename
    victim_stats = 0
    attacked = False

    def fail_after_child(*_args: object, copy_root: Path, **_kwargs: object) -> None:
        nonlocal staged_snapshot
        staged_snapshot = copy_root
        (copy_root / "victim").write_bytes(b"owned\n")
        raise ValueError("injected staging failure")

    def replace_child_after_stat(
        path: str | bytes | Path,
        *,
        dir_fd: int | None = None,
        follow_symlinks: bool = True,
    ) -> os.stat_result:
        nonlocal victim_stats, attacked
        status = real_stat(path, dir_fd=dir_fd, follow_symlinks=follow_symlinks)
        if path == "victim" and dir_fd is not None:
            victim_stats += 1
            if victim_stats == 2:
                attacked = True
                real_rename("victim", "victim-displaced", src_dir_fd=dir_fd, dst_dir_fd=dir_fd)
                descriptor = real_open(
                    "victim",
                    os.O_WRONLY | os.O_CREAT | os.O_EXCL,
                    0o600,
                    dir_fd=dir_fd,
                )
                try:
                    os.write(descriptor, b"foreign\n")
                finally:
                    os.close(descriptor)
        return status

    monkeypatch.setattr(module, "descriptor_stable_tree_entries", fail_after_child)
    monkeypatch.setattr(module, "canonical_tree_sha256", lambda _entries: "d" * 64)
    monkeypatch.setattr(module.os, "stat", replace_child_after_stat)

    with pytest.raises(module.ComplementError, match="staging failed"):
        module._stage_tokenizer_snapshot(trust, tmp_path / "scratch")

    assert not attacked
    assert staged_snapshot is not None
    assert (staged_snapshot / "victim").read_bytes() == b"owned\n"


def test_tokenizer_stage_failure_never_rmdirs_a_replaced_root_after_stat(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Private scratch retention avoids the irreducible terminal stat-to-rmdir race."""
    module = _load_module()
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    trust = module.TokenizerTrust(
        file_sha256="a" * 64,
        receipt_sha256="b" * 64,
        repository=module.Q30T_TOKENIZER_REPOSITORY,
        revision="c" * 40,
        snapshot_path=snapshot,
        snapshot_tree_sha256="d" * 64,
        chat_template_sha256="e" * 64,
        training_chat_template_sha256="f" * 64,
        im_start_token_id=1,
        im_end_token_id=2,
    )
    stage_root: Path | None = None
    real_stat = module.os.stat
    real_rename = module.os.rename
    real_mkdir = module.os.mkdir
    attacked = False

    def fail_with_empty_stage(*_args: object, copy_root: Path, **_kwargs: object) -> None:
        nonlocal stage_root
        stage_root = copy_root.parent
        raise ValueError("injected staging failure")

    def replace_root_after_stat(
        path: str | bytes | Path,
        *,
        dir_fd: int | None = None,
        follow_symlinks: bool = True,
    ) -> os.stat_result:
        nonlocal attacked
        status = real_stat(path, dir_fd=dir_fd, follow_symlinks=follow_symlinks)
        if (
            not attacked
            and stage_root is not None
            and path == stage_root.name
            and dir_fd is not None
        ):
            attacked = True
            real_rename(
                stage_root.name,
                "stage-root-displaced",
                src_dir_fd=dir_fd,
                dst_dir_fd=dir_fd,
            )
            real_mkdir(stage_root.name, 0o700, dir_fd=dir_fd)
        return status

    monkeypatch.setattr(module, "descriptor_stable_tree_entries", fail_with_empty_stage)
    monkeypatch.setattr(module, "canonical_tree_sha256", lambda _entries: "d" * 64)
    monkeypatch.setattr(module.os, "stat", replace_root_after_stat)

    with pytest.raises(module.ComplementError, match="staging failed"):
        module._stage_tokenizer_snapshot(trust, tmp_path / "scratch")

    assert not attacked
    assert stage_root is not None
    assert stage_root.is_dir()


def test_tokenizer_stage_retains_its_bound_root_when_snapshot_creation_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Scheduler scratch lifecycle owns a stage retained after any early failure."""
    module = _load_module()
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    trust = module.TokenizerTrust(
        file_sha256="a" * 64,
        receipt_sha256="b" * 64,
        repository=module.Q30T_TOKENIZER_REPOSITORY,
        revision="c" * 40,
        snapshot_path=snapshot,
        snapshot_tree_sha256="d" * 64,
        chat_template_sha256="e" * 64,
        training_chat_template_sha256="f" * 64,
        im_start_token_id=1,
        im_end_token_id=2,
    )
    real_mkdir = module.Path.mkdir

    def fail_snapshot(path: Path, *args: object, **kwargs: object) -> None:
        if path.name == "snapshot" and path.parent.name.startswith("qwen-tokenizer-"):
            raise OSError("injected snapshot mkdir failure")
        real_mkdir(path, *args, **kwargs)

    monkeypatch.setattr(module.Path, "mkdir", fail_snapshot)
    scratch = tmp_path / "scratch"

    with pytest.raises(OSError, match="injected snapshot mkdir failure"):
        module._stage_tokenizer_snapshot(trust, scratch)

    retained = list(scratch.glob("qwen-tokenizer-*"))
    assert len(retained) == 1
    assert retained[0].is_dir()


def test_bundle_publication_never_follows_a_replaced_partial_child(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Publication cannot truncate a foreign target through a substituted symlink."""
    module = _load_module()
    selection = module.select_continuation_rows(
        {"stem": [_row("stem", "one", 0)]},
        quotas={"stem": 1},
        prior_prompt_uuids=set(),
        held_out_prompt_uuids=set(),
        tokenizer=_Tokenizer(),
        training_sequence_length=8,
        replay_categories=frozenset(),
    )
    historical_path, historical_file_sha256 = _write_historical_receipt(
        tmp_path / "historical.json"
    )
    historical = module.load_historical_exclusion(
        historical_path,
        expected_sha256=historical_file_sha256,
        expected_occurrence_count=3,
    )
    held_out = module.load_held_out_union([], required_names=frozenset())
    foreign = tmp_path / "foreign-target"
    foreign.write_bytes(b"preserve-me\n")
    original_open = module.os.open
    attacked = False

    def replace_partial(
        path: str | bytes | Path,
        flags: int,
        mode: int = 0o777,
        *,
        dir_fd: int | None = None,
    ) -> int:
        nonlocal attacked
        if not attacked and path == "DATA.jsonl" and dir_fd is not None:
            attacked = True
            partial = next(tmp_path.glob(".bundle.partial-*"))
            partial.rename(tmp_path / "displaced-partial")
            partial.mkdir()
            (partial / "DATA.jsonl").symlink_to(foreign)
        return original_open(path, flags, mode, dir_fd=dir_fd)

    monkeypatch.setattr(module.os, "open", replace_partial)

    with suppress(module.ComplementError):
        module.publish_selection_bundle(
            selection,
            output_root=tmp_path / "bundle",
            scratch_root=tmp_path / "scratch",
            quotas={"stem": 1},
            config_file_sha256=hashlib.sha256(CONFIG_PATH.read_bytes()).hexdigest(),
            source_inventory_file_sha256="6" * 64,
            historical=historical,
            held_out=held_out,
            tokenizer_trust_file_sha256="7" * 64,
            runtime_sha256="8" * 64,
            source_commit="9" * 40,
            enforce_production_paths=False,
        )

    assert attacked
    assert foreign.read_bytes() == b"preserve-me\n"


def test_builder_stable_readers_require_regular_descriptors() -> None:
    source = MODULE_PATH.read_text()
    stable = source[
        source.index("def _stable_regular_bytes") : source.index("def _physical_row_count")
    ]

    assert "stat.S_ISREG" in stable


def test_replay_scratch_is_cleaned_on_every_exit() -> None:
    source = MODULE_PATH.read_text()
    verifier = source[source.index("def verify_selection_bundle") : source.index("def _parse_args")]

    assert "with tempfile.TemporaryDirectory" in verifier


def test_heldout_help_documents_named_receipt_syntax() -> None:
    help_text = subprocess.run(
        [sys.executable, str(MODULE_PATH), "build", "--help"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout

    assert "NAME=PATH:SHA256" in help_text


def test_physical_row_count_uses_a_nofollow_regular_descriptor(tmp_path: Path) -> None:
    """Capacity counting cannot reopen a trusted path as a FIFO or symlink."""
    module = _load_module()
    nonregular = tmp_path / "directory"
    nonregular.mkdir()
    source = MODULE_PATH.read_text()
    counter = source[
        source.index("def _physical_row_count") : source.index("def load_source_inventory")
    ]

    assert "os.open" in counter
    assert "O_NONBLOCK" in counter
    assert "stat.S_ISREG" in counter
    assert "path.open" not in counter
    with pytest.raises(module.ComplementError, match="regular file"):
        module._physical_row_count(nonregular, "jsonl")


def test_authenticated_source_stream_uses_nonblocking_single_link_descriptor() -> None:
    """Source iteration rejects special and multiply-linked inputs before any stream read."""
    source = MODULE_PATH.read_text()
    reader = source[
        source.index("def _authenticated_source_stream") : source.index("def _iter_source_file")
    ]

    assert "O_NONBLOCK" in reader
    assert "stat.S_ISREG" in reader
    assert "st_nlink != 1" in reader


def test_replay_workspace_uses_context_managed_cleanup() -> None:
    """Replay scratch is removed immediately on success and every exception."""
    source = MODULE_PATH.read_text()
    verifier = source[source.index("def verify_selection_bundle") : source.index("def _parse_args")]

    assert "with tempfile.TemporaryDirectory" in verifier
