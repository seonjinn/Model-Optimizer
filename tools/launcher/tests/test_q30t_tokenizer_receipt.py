# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Q30 Thinking tokenizer receipt contracts."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import pytest
from common.specdec.q30t_tokenizer_receipt import (
    build_q30t_tokenizer_receipt,
    snapshot_tree_sha256,
    verify_q30t_tokenizer_receipt,
)

ROOT = Path(__file__).resolve().parents[3]
BUILDER_PATH = ROOT / "examples/dataset/build_qwen4b_ptv23_complement.py"
Q30_REPOSITORY = "Qwen/Qwen3-30B-A3B-Thinking-2507"
FIXTURE_Q30_REVISION = "a" * 40
OFFICIAL_TEMPLATE = """{% for message in messages %}
    {%- elif message.role == "assistant" %}
        {{- '<|im_start|>' + message.role + '\\n' + message.content }}
        {{- '<|im_end|>\\n' }}
    {%- elif message.role == "tool" %}
{% endfor %}"""
TRAINING_TEMPLATE = """{% for message in messages %}
    {%- elif message.role == "assistant" %}
        {{- '<|im_start|>' + message.role + '\\n' }}
        {%- generation %}
        {{- message.content }}
        {{- '<|im_end|>\\n' }}
        {%- endgeneration %}
    {%- elif message.role == "tool" %}
{% endfor %}"""


def _canonical(value: object) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()


def _make_q30_snapshot(tmp_path: Path) -> Path:
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    (snapshot / "tokenizer_config.json").write_bytes(
        _canonical({"chat_template": OFFICIAL_TEMPLATE})
    )
    (snapshot / "tokenizer.json").write_bytes(
        _canonical(
            {
                "added_tokens": [
                    {"content": "<|im_start|>", "id": 17},
                    {"content": "<|im_end|>", "id": 18},
                ]
            }
        )
    )
    nested = snapshot / "assets"
    nested.mkdir()
    (nested / "tokenizer.model").write_bytes(b"fixture tokenizer asset")
    return snapshot


def _load_builder():
    spec = importlib.util.spec_from_file_location("build_qwen4b_ptv23_complement", BUILDER_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    sys.path.insert(0, str(BUILDER_PATH.parent))
    try:
        spec.loader.exec_module(module)
    finally:
        sys.path.pop(0)
    return module


def test_q30t_receipt_binds_snapshot_template_and_special_tokens(tmp_path: Path) -> None:
    """A receipt binds all locally recomputed tokenizer evidence."""
    snapshot = _make_q30_snapshot(tmp_path)

    receipt = build_q30t_tokenizer_receipt(
        snapshot=snapshot,
        repository=Q30_REPOSITORY,
        revision=FIXTURE_Q30_REVISION,
    )

    parsed = json.loads(receipt)
    assert parsed["schema_version"] == "qwen3-30ba3b-thinking-tokenizer-trust-v1"
    assert parsed["repository"] == Q30_REPOSITORY
    assert parsed["revision"] == FIXTURE_Q30_REVISION
    assert parsed["snapshot_path"] == str(snapshot)
    assert parsed["snapshot_tree_sha256"] == snapshot_tree_sha256(snapshot)
    assert parsed["chat_template_sha256"] == hashlib.sha256(OFFICIAL_TEMPLATE.encode()).hexdigest()
    assert parsed["training_chat_template_sha256"] == hashlib.sha256(TRAINING_TEMPLATE.encode()).hexdigest()
    assert parsed["im_start_token_id"] == 17
    assert parsed["im_end_token_id"] == 18
    assert verify_q30t_tokenizer_receipt(receipt) == parsed


@pytest.mark.parametrize("field", ["revision", "snapshot_tree_sha256", "training_chat_template_sha256"])
def test_q30t_receipt_rejects_tampering(tmp_path: Path, field: str) -> None:
    """Each signed receipt field rejects a post-build alteration."""
    payload = json.loads(
        build_q30t_tokenizer_receipt(
            snapshot=_make_q30_snapshot(tmp_path),
            repository=Q30_REPOSITORY,
            revision=FIXTURE_Q30_REVISION,
        )
    )
    payload[field] = "b" * (40 if field == "revision" else 64)

    with pytest.raises(ValueError, match="receipt"):
        verify_q30t_tokenizer_receipt(_canonical(payload))


def test_q30t_builder_recomputes_tokenizer_receipt_evidence(tmp_path: Path) -> None:
    """The shared builder accepts Q30 special-token IDs only after recomputation."""
    builder = _load_builder()
    snapshot = _make_q30_snapshot(tmp_path)
    receipt = build_q30t_tokenizer_receipt(
        snapshot=snapshot,
        repository=Q30_REPOSITORY,
        revision=FIXTURE_Q30_REVISION,
    )
    receipt_path = tmp_path / "receipt.json"
    receipt_path.write_bytes(receipt)
    policy = builder.TargetTokenizerPolicy(
        schema_version="ptv23-complement-target-policy-v1",
        scientific_identity="fixture-q30",
        target_repository=Q30_REPOSITORY,
        target_revision=FIXTURE_Q30_REVISION,
        tokenizer_repository=Q30_REPOSITORY,
        tokenizer_revision=FIXTURE_Q30_REVISION,
        tokenizer_trust_schema="qwen3-30ba3b-thinking-tokenizer-trust-v1",
        training_sequence_length=4096,
        quota_config_path="quota.json",
        quota_config_sha256="1" * 64,
        source_requirements_path="sources.json",
        source_requirements_sha256="2" * 64,
        file_sha256="3" * 64,
    )

    trust = builder.load_tokenizer_trust(
        receipt_path,
        expected_sha256=hashlib.sha256(receipt).hexdigest(),
        policy=policy,
    )

    assert (trust.im_start_token_id, trust.im_end_token_id) == (17, 18)
