# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Target-policy isolation for the PTV2/PTV3 complement builder."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import subprocess
import sys
from dataclasses import replace
from pathlib import Path
from types import MappingProxyType

import pytest

ROOT = Path(__file__).resolve().parents[3]
BUILDER_PATH = ROOT / "examples/dataset/build_qwen4b_ptv23_complement.py"
POLICY_PATH = ROOT / "examples/dataset/ptv23_complement_target_policy.py"
Q30_POLICY_PATH = ROOT / "examples/dataset/qwen3_30ba3b_thinking_ptv23_target_v1.json"
Q30_SWE_HEAVY_POLICY_PATH = (
    ROOT / "examples/dataset/qwen3_30ba3b_thinking_ptv23_swe_heavy_target_v1.json"
)
Q30_SWE_HEAVY_CONFIG_PATH = (
    ROOT / "examples/dataset/qwen3_30ba3b_thinking_ptv23_swe_heavy_700k_v1.json"
)
Q30_SOURCE_REQUIREMENTS_PATH = (
    ROOT / "examples/dataset/qwen3_30ba3b_thinking_ptv23_complement_sources_v1.json"
)
Q30_SWE_HEAVY_POLICY_SHA256 = "940a3cfdd04a8cff94fff8a6980110dcb2e877ebd6f1897da75acff7d2023925"
Q30_SWE_HEAVY_CONFIG_SHA256 = "9f13a84a41db3ad127926dc3d5ee02020fad3223d9a36fc46dd80edd7ee65a63"
Q30_SOURCE_REQUIREMENTS_SHA256 = "8cb600a2a839148814b0d26a6a80a7f061db5456291cd5d708be4520521d8e6b"
Q30_BALANCED_IDENTITY = "ptv2-ptv3-complement-700k-qwen3-30ba3b-thinking-v1"
Q30_SWE_HEAVY_IDENTITY = "ptv2-ptv3-complement-700k-qwen3-30ba3b-thinking-swe-heavy-v1"
Q30_SWE_HEAVY_QUOTAS = {
    "ptv2_stem": 200_000,
    "ptv2_multilingual_ja": 25_000,
    "ptv2_multilingual_es": 25_000,
    "ptv2_multilingual_fr": 25_000,
    "ptv2_multilingual_it": 25_000,
    "ptv3_swe_v3": 180_000,
    "ptv3_interactive_agentic_swe": 60_000,
    "ptv3_general_tool_trajectories": 160_000,
}


def test_target_policy_reader_rejects_a_multiply_linked_file(tmp_path: Path) -> None:
    policy_module = _load_module("ptv23_complement_target_policy_hardlink", POLICY_PATH)
    policy = tmp_path / "policy.json"
    alias = tmp_path / "policy-alias.json"
    policy.write_bytes(b"{}\n")
    os.link(policy, alias)

    with pytest.raises(policy_module.ComplementError, match="single-link regular file"):
        policy_module._stable_regular_bytes(policy)


def test_target_policy_reader_rejects_an_oversized_file(tmp_path: Path) -> None:
    policy_module = _load_module("ptv23_complement_target_policy_oversize", POLICY_PATH)
    policy = tmp_path / "policy.json"
    policy.write_bytes(b"x" * (1 * 1024 * 1024 + 1))

    with pytest.raises(policy_module.ComplementError, match="too large"):
        policy_module._stable_regular_bytes(policy)


def test_target_policy_reader_rejects_a_path_rebound_during_read(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    policy_module = _load_module("ptv23_complement_target_policy_rebind", POLICY_PATH)
    policy = tmp_path / "policy.json"
    displaced = tmp_path / "policy-displaced.json"
    policy.write_bytes(b"{}\n")
    real_fdopen = policy_module.os.fdopen

    class RebindingStream:
        def __init__(self, descriptor: int) -> None:
            self._stream = real_fdopen(descriptor, "rb")

        def __enter__(self):
            self._stream.__enter__()
            return self

        def __exit__(self, *args: object) -> object:
            return self._stream.__exit__(*args)

        def fileno(self) -> int:
            return self._stream.fileno()

        def read(self, size: int = -1) -> bytes:
            raw = self._stream.read(size)
            policy.rename(displaced)
            policy.write_bytes(b"foreign\n")
            return raw

    monkeypatch.setattr(
        policy_module.os,
        "fdopen",
        lambda descriptor, _mode: RebindingStream(descriptor),
    )

    with pytest.raises(policy_module.ComplementError, match="changed while reading"):
        policy_module._stable_regular_bytes(policy)


def test_target_policy_reader_rejects_a_fifo_without_waiting_for_a_writer(tmp_path: Path) -> None:
    fifo = tmp_path / "policy.fifo"
    os.mkfifo(fifo)
    program = f"""
import importlib.util
from pathlib import Path
import sys
spec = importlib.util.spec_from_file_location('policy_fifo', {str(POLICY_PATH)!r})
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)
try:
    module._stable_regular_bytes(Path({str(fifo)!r}))
except module.ComplementError:
    raise SystemExit(0)
raise SystemExit(1)
"""

    completed = subprocess.run(
        [sys.executable, "-c", program],
        check=False,
        timeout=1,
    )

    assert completed.returncode == 0


def _canonical(value: object) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    sys.path.insert(0, str(path.parent))
    try:
        spec.loader.exec_module(module)
    finally:
        sys.path.pop(0)
    return module


def _write_receipt(tmp_path: Path, repository: str) -> tuple[Path, str]:
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    (snapshot / "tokenizer.json").write_bytes(b"fixture")
    tree = [["tokenizer.json", hashlib.sha256(b"fixture").hexdigest()]]
    body = {
        "schema_version": "qwen3-4b-tokenizer-trust-v1",
        "repository": repository,
        "revision": "1cfa9a7208912126459214e8b04321603b3df60c",
        "snapshot_path": str(snapshot),
        "snapshot_tree_sha256": hashlib.sha256(_canonical(tree)).hexdigest(),
        "chat_template_sha256": "6" * 64,
        "training_chat_template_sha256": "7" * 64,
        "im_start_token_id": 151644,
        "im_end_token_id": 151645,
    }
    payload = body | {"receipt_sha256": hashlib.sha256(_canonical(body)).hexdigest()}
    path = tmp_path / "receipt.json"
    path.write_bytes(_canonical(payload) + b"\n")
    return path, hashlib.sha256(path.read_bytes()).hexdigest()


def _approved_q30_policy(monkeypatch: pytest.MonkeyPatch):
    policy_module = _load_module("ptv23_complement_target_policy", POLICY_PATH)
    policy_sha256 = hashlib.sha256(Q30_POLICY_PATH.read_bytes()).hexdigest()
    approved_policy = policy_module.TargetTokenizerPolicy(
        schema_version="ptv23-complement-target-policy-v1",
        scientific_identity=Q30_BALANCED_IDENTITY,
        target_repository="Qwen/Qwen3-30B-A3B-Thinking-2507",
        target_revision="",
        tokenizer_repository="Qwen/Qwen3-30B-A3B-Thinking-2507",
        tokenizer_revision="",
        tokenizer_trust_schema="qwen3-30ba3b-thinking-tokenizer-trust-v1",
        training_sequence_length=4_096,
        quota_config_path="qwen3_30ba3b_thinking_ptv23_complement_700k_v1.json",
        quota_config_sha256="b7436725b75f09f3557ff589f99da67b68472abfd7e328a41d6ce5ff639bb88d",
        source_requirements_path=Q30_SOURCE_REQUIREMENTS_PATH.name,
        source_requirements_sha256=Q30_SOURCE_REQUIREMENTS_SHA256,
        file_sha256=policy_sha256,
    )
    monkeypatch.setattr(
        policy_module,
        "APPROVED_TARGET_POLICY_SHA256S",
        policy_module.APPROVED_TARGET_POLICY_SHA256S | {policy_sha256},
    )
    monkeypatch.setattr(
        policy_module,
        "APPROVED_TARGET_POLICIES",
        dict(policy_module.APPROVED_TARGET_POLICIES) | {policy_sha256: approved_policy},
    )
    return policy_module.load_target_policy(Q30_POLICY_PATH, expected_sha256=policy_sha256)


def _approved_swe_heavy_policy(monkeypatch: pytest.MonkeyPatch):
    policy_module = _load_module("ptv23_complement_target_policy", POLICY_PATH)
    approved_policy = policy_module.TargetTokenizerPolicy(
        schema_version="ptv23-complement-target-policy-v1",
        scientific_identity=Q30_SWE_HEAVY_IDENTITY,
        target_repository="Qwen/Qwen3-30B-A3B-Thinking-2507",
        target_revision="",
        tokenizer_repository="Qwen/Qwen3-30B-A3B-Thinking-2507",
        tokenizer_revision="",
        tokenizer_trust_schema="qwen3-30ba3b-thinking-tokenizer-trust-v1",
        training_sequence_length=4_096,
        quota_config_path=Q30_SWE_HEAVY_CONFIG_PATH.name,
        quota_config_sha256=Q30_SWE_HEAVY_CONFIG_SHA256,
        source_requirements_path=Q30_SOURCE_REQUIREMENTS_PATH.name,
        source_requirements_sha256=Q30_SOURCE_REQUIREMENTS_SHA256,
        file_sha256=Q30_SWE_HEAVY_POLICY_SHA256,
    )
    monkeypatch.setattr(
        policy_module,
        "APPROVED_TARGET_POLICY_SHA256S",
        policy_module.APPROVED_TARGET_POLICY_SHA256S | {Q30_SWE_HEAVY_POLICY_SHA256},
    )
    monkeypatch.setattr(
        policy_module,
        "APPROVED_TARGET_POLICIES",
        MappingProxyType(
            dict(policy_module.APPROVED_TARGET_POLICIES)
            | {Q30_SWE_HEAVY_POLICY_SHA256: approved_policy}
        ),
    )
    return policy_module.load_target_policy(
        Q30_SWE_HEAVY_POLICY_PATH,
        expected_sha256=Q30_SWE_HEAVY_POLICY_SHA256,
    )


def _q30_inventory(builder, policy):
    requirements = json.loads(Q30_SOURCE_REQUIREMENTS_PATH.read_bytes())
    schema_sha256 = "a" * 64
    sources = [
        builder.InventorySource(
            category=category,
            source_id=f"fixture/{category}",
            revision=builder.PTV2_REVISION,
            split="train",
            license_expression="CC-BY-4.0",
            replay_lane="none",
            row_schema={},
            row_schema_sha256=schema_sha256,
            files=(
                builder.InventoryFile(
                    "data/train.jsonl", Path("/fixture/ptv2.jsonl"), 1, "b" * 64, 1
                ),
            ),
        )
        for category in (
            "ptv2_stem",
            "ptv2_multilingual_ja",
            "ptv2_multilingual_es",
            "ptv2_multilingual_fr",
            "ptv2_multilingual_it",
        )
    ]
    swe_source = builder.InventorySource(
        category="ptv3_swe_v3",
        source_id="fixture/swe",
        revision="c" * 40,
        split="train",
        license_expression="CC-BY-4.0",
        replay_lane="none",
        row_schema={},
        row_schema_sha256=schema_sha256,
        files=(
            builder.InventoryFile("data/swe.jsonl", Path("/fixture/swe.jsonl"), 1, "d" * 64, 1),
        ),
    )
    sources.append(swe_source)
    sources.extend(
        builder.InventorySource(
            category=pin["category"],
            source_id=pin["repository"],
            revision=pin["revision"],
            split=pin["split"],
            license_expression="CC-BY-4.0",
            replay_lane="none",
            row_schema={},
            row_schema_sha256=schema_sha256,
            files=(
                builder.InventoryFile(
                    pin["path"], Path(f"/fixture/{pin['path']}"), pin["bytes"], pin["sha256"], 1
                ),
            ),
        )
        for pin in requirements["known_pinned_sources"]
    )
    return (
        builder.SourceInventory(policy.scientific_identity, "e" * 64, tuple(sources)),
        swe_source,
        schema_sha256,
    )


def _write_q30_capacity_receipt(
    tmp_path: Path,
    builder,
    policy,
    *,
    quotas: dict[str, int] | None = None,
) -> Path:
    quotas = builder.APPROVED_QUOTAS if quotas is None else quotas
    payload = {
        "schema_version": "ptv2-ptv3-complement-capacity-v1",
        "scientific_identity": policy.scientific_identity,
        "status": "sufficient",
        "redistribution": "forbidden",
        "categories": [
            {"category": category, "required": quota, "selected": quota}
            for category, quota in quotas.items()
        ],
    }
    path = tmp_path / "capacity.json"
    path.write_bytes(_canonical(payload) + b"\n")
    return path


def _reconcile_q30_source_requirements(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    policy = _approved_q30_policy(monkeypatch)
    builder = _load_module("build_qwen4b_ptv23_complement", BUILDER_PATH)
    inventory, swe_source, schema_sha256 = _q30_inventory(builder, policy)
    monkeypatch.setattr(builder, "_require_external_approval_roots", lambda: None)
    monkeypatch.setattr(
        builder,
        "APPROVED_PTV3_SWE_SOURCE_PINS",
        ((swe_source.source_id, swe_source.revision, schema_sha256),),
    )
    monkeypatch.setattr(builder, "APPROVED_ROW_SCHEMA_SHA256S", frozenset({schema_sha256}))
    builder.reconcile_source_requirements(
        Q30_SOURCE_REQUIREMENTS_PATH,
        inventory=inventory,
        capacity_receipt_path=_write_q30_capacity_receipt(tmp_path, builder, policy),
        policy=policy,
    )


def test_q30_policy_rejects_q4_tokenizer_receipt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    builder = _load_module("build_qwen4b_ptv23_complement", BUILDER_PATH)
    policy = _approved_q30_policy(monkeypatch)
    receipt, receipt_sha256 = _write_receipt(tmp_path, "Qwen/Qwen3-4B")

    with pytest.raises(builder.ComplementError, match="target tokenizer identity"):
        builder.load_tokenizer_trust(receipt, expected_sha256=receipt_sha256, policy=policy)


def test_target_policy_cannot_self_authorize_an_unknown_file(tmp_path: Path) -> None:
    policy_module = _load_module("ptv23_complement_target_policy", POLICY_PATH)
    forged = tmp_path / "forged.json"
    forged.write_bytes(Q30_POLICY_PATH.read_bytes().replace(b"Qwen/Qwen3", b"Qwen/forged"))

    with pytest.raises(policy_module.ComplementError, match="target policy is not approved"):
        policy_module.load_target_policy(
            forged, expected_sha256=hashlib.sha256(forged.read_bytes()).hexdigest()
        )


def test_q30_policy_is_fail_closed_until_a_reviewed_receipt_approves_it() -> None:
    policy_module = _load_module("ptv23_complement_target_policy", POLICY_PATH)
    policy_sha256 = hashlib.sha256(Q30_POLICY_PATH.read_bytes()).hexdigest()

    with pytest.raises(policy_module.ComplementError, match="target policy is not approved"):
        policy_module.load_target_policy(Q30_POLICY_PATH, expected_sha256=policy_sha256)


def test_q30_approved_policy_accepts_its_source_blocker_vocabulary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _reconcile_q30_source_requirements(tmp_path, monkeypatch)


def test_q30_approved_policy_reconciles_its_capacity_identity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _reconcile_q30_source_requirements(tmp_path, monkeypatch)


def test_reviewed_swe_heavy_identity_reuses_only_the_exact_q30_source_contract(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    policy = _approved_swe_heavy_policy(monkeypatch)
    builder = _load_module("build_qwen4b_ptv23_complement", BUILDER_PATH)
    inventory, swe_source, schema_sha256 = _q30_inventory(builder, policy)
    monkeypatch.setattr(builder, "_require_external_approval_roots", lambda: None)
    monkeypatch.setattr(
        builder,
        "APPROVED_PTV3_SWE_SOURCE_PINS",
        ((swe_source.source_id, swe_source.revision, schema_sha256),),
    )
    monkeypatch.setattr(builder, "APPROVED_ROW_SCHEMA_SHA256S", frozenset({schema_sha256}))

    builder.reconcile_source_requirements(
        Q30_SOURCE_REQUIREMENTS_PATH,
        inventory=inventory,
        capacity_receipt_path=_write_q30_capacity_receipt(
            tmp_path,
            builder,
            policy,
            quotas=Q30_SWE_HEAVY_QUOTAS,
        ),
        policy=policy,
    )


def test_shared_q30_source_contract_rejects_an_unrelated_policy_identity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    policy = replace(_approved_swe_heavy_policy(monkeypatch), scientific_identity="unreviewed-arm")
    builder = _load_module("build_qwen4b_ptv23_complement", BUILDER_PATH)
    inventory, _, _ = _q30_inventory(builder, policy)

    with pytest.raises(builder.ComplementError, match="target policy object is not approved"):
        builder.reconcile_source_requirements(
            Q30_SOURCE_REQUIREMENTS_PATH,
            inventory=inventory,
            capacity_receipt_path=tmp_path / "unused-capacity.json",
            policy=policy,
        )


def test_shared_q30_source_contract_rejects_a_different_physical_hash(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    policy = replace(_approved_swe_heavy_policy(monkeypatch), source_requirements_sha256="0" * 64)
    builder = _load_module("build_qwen4b_ptv23_complement", BUILDER_PATH)
    inventory, _, _ = _q30_inventory(builder, policy)

    with pytest.raises(builder.ComplementError, match="target policy object is not approved"):
        builder.reconcile_source_requirements(
            Q30_SOURCE_REQUIREMENTS_PATH,
            inventory=inventory,
            capacity_receipt_path=tmp_path / "unused-capacity.json",
            policy=policy,
        )


def test_shared_q30_source_contract_rejects_an_unapproved_target_policy_hash(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    policy = replace(_approved_swe_heavy_policy(monkeypatch), file_sha256="f" * 64)
    builder = _load_module("build_qwen4b_ptv23_complement", BUILDER_PATH)
    inventory, _, _ = _q30_inventory(builder, policy)

    with pytest.raises(builder.ComplementError, match="target policy object is not approved"):
        builder.reconcile_source_requirements(
            Q30_SOURCE_REQUIREMENTS_PATH,
            inventory=inventory,
            capacity_receipt_path=tmp_path / "unused-capacity.json",
            policy=policy,
        )


def test_reconciliation_rejects_caller_supplied_toy_quotas_for_approved_swe_policy(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    policy = _approved_swe_heavy_policy(monkeypatch)
    builder = _load_module("build_qwen4b_ptv23_complement", BUILDER_PATH)
    inventory, swe_source, schema_sha256 = _q30_inventory(builder, policy)
    toy_quotas = dict.fromkeys(Q30_SWE_HEAVY_QUOTAS, 1)
    monkeypatch.setattr(builder, "_require_external_approval_roots", lambda: None)
    monkeypatch.setattr(
        builder,
        "APPROVED_PTV3_SWE_SOURCE_PINS",
        ((swe_source.source_id, swe_source.revision, schema_sha256),),
    )
    monkeypatch.setattr(builder, "APPROVED_ROW_SCHEMA_SHA256S", frozenset({schema_sha256}))

    with pytest.raises(TypeError, match="unexpected keyword argument 'quotas'"):
        builder.reconcile_source_requirements(
            Q30_SOURCE_REQUIREMENTS_PATH,
            inventory=inventory,
            capacity_receipt_path=_write_q30_capacity_receipt(
                tmp_path,
                builder,
                policy,
                quotas=toy_quotas,
            ),
            policy=policy,
            quotas=toy_quotas,
        )


def test_reconciliation_rejects_forged_policy_fields_with_an_approved_file_hash(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    approved_policy = _approved_swe_heavy_policy(monkeypatch)
    builder = _load_module("build_qwen4b_ptv23_complement", BUILDER_PATH)
    forged_payload = json.loads(Q30_SWE_HEAVY_CONFIG_PATH.read_bytes())
    forged_payload["quotas"]["ptv2_stem"] = 200_001
    forged_payload["quotas"]["ptv3_swe_v3"] = 179_999
    forged_config = tmp_path / "forged-swe-heavy-config.json"
    forged_config.write_bytes(_canonical(forged_payload) + b"\n")
    forged_policy = replace(
        approved_policy,
        quota_config_path=forged_config.name,
        quota_config_sha256=hashlib.sha256(forged_config.read_bytes()).hexdigest(),
    )
    requirements_path = tmp_path / Q30_SOURCE_REQUIREMENTS_PATH.name
    requirements_path.write_bytes(Q30_SOURCE_REQUIREMENTS_PATH.read_bytes())
    inventory, swe_source, schema_sha256 = _q30_inventory(builder, forged_policy)
    forged_quotas = dict(Q30_SWE_HEAVY_QUOTAS)
    forged_quotas["ptv2_stem"] = 200_001
    forged_quotas["ptv3_swe_v3"] = 179_999
    monkeypatch.setattr(builder, "_require_external_approval_roots", lambda: None)
    monkeypatch.setattr(
        builder,
        "APPROVED_PTV3_SWE_SOURCE_PINS",
        ((swe_source.source_id, swe_source.revision, schema_sha256),),
    )
    monkeypatch.setattr(builder, "APPROVED_ROW_SCHEMA_SHA256S", frozenset({schema_sha256}))

    with pytest.raises(builder.ComplementError, match="target policy object is not approved"):
        builder.reconcile_source_requirements(
            requirements_path,
            inventory=inventory,
            capacity_receipt_path=_write_q30_capacity_receipt(
                tmp_path,
                builder,
                forged_policy,
                quotas=forged_quotas,
            ),
            policy=forged_policy,
        )


def test_swe_heavy_policy_loads_its_sha_bound_exact_700k_quotas(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    policy = _approved_swe_heavy_policy(monkeypatch)
    builder = _load_module("build_qwen4b_ptv23_complement", BUILDER_PATH)

    config = builder.load_complement_config(Q30_SWE_HEAVY_CONFIG_PATH, policy=policy)

    assert config["scientific_identity"] == Q30_SWE_HEAVY_IDENTITY
    assert config["quotas"] == Q30_SWE_HEAVY_QUOTAS
    assert sum(config["quotas"].values()) == 700_000


def test_swe_heavy_target_policy_is_canonical_but_fail_closed_pending_receipt() -> None:
    policy_module = _load_module("ptv23_complement_target_policy", POLICY_PATH)
    raw = Q30_SWE_HEAVY_POLICY_PATH.read_bytes()
    payload = json.loads(raw)

    assert raw == _canonical(payload) + b"\n"
    assert hashlib.sha256(raw).hexdigest() == Q30_SWE_HEAVY_POLICY_SHA256
    with pytest.raises(policy_module.ComplementError, match="target policy is not approved"):
        policy_module.load_target_policy(
            Q30_SWE_HEAVY_POLICY_PATH,
            expected_sha256=Q30_SWE_HEAVY_POLICY_SHA256,
        )

    assert payload["scientific_identity"] == Q30_SWE_HEAVY_IDENTITY
    assert payload["quota_config"]["path"] == Q30_SWE_HEAVY_CONFIG_PATH.name
    assert payload["quota_config"]["sha256"] == Q30_SWE_HEAVY_CONFIG_SHA256
    assert payload["source_requirements"]["sha256"] == Q30_SOURCE_REQUIREMENTS_SHA256


def test_reviewed_target_policy_registry_is_immutable(monkeypatch: pytest.MonkeyPatch) -> None:
    policy_module = _load_module("ptv23_complement_target_policy", POLICY_PATH)
    policy = _approved_swe_heavy_policy(monkeypatch)
    registry = policy_module.APPROVED_TARGET_POLICIES

    with pytest.raises(TypeError):
        registry["f" * 64] = policy


def test_sha_bound_swe_heavy_policy_rejects_forged_quota_bytes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    policy = _approved_swe_heavy_policy(monkeypatch)
    builder = _load_module("build_qwen4b_ptv23_complement", BUILDER_PATH)
    forged = tmp_path / Q30_SWE_HEAVY_CONFIG_PATH.name
    payload = json.loads(Q30_SWE_HEAVY_CONFIG_PATH.read_bytes())
    payload["quotas"]["ptv2_stem"] -= 1
    payload["quotas"]["ptv3_swe_v3"] += 1
    forged.write_bytes(_canonical(payload) + b"\n")

    with pytest.raises(builder.ComplementError, match="approved continuation policy"):
        builder.load_complement_config(forged, policy=policy)
