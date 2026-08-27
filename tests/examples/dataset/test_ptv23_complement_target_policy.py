# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Target-policy isolation for the PTV2/PTV3 complement builder."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
BUILDER_PATH = ROOT / "examples/dataset/build_qwen4b_ptv23_complement.py"
POLICY_PATH = ROOT / "examples/dataset/ptv23_complement_target_policy.py"
Q30_POLICY_PATH = ROOT / "examples/dataset/qwen3_30ba3b_thinking_ptv23_target_v1.json"


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


def test_q30_policy_rejects_q4_tokenizer_receipt(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    policy_module = _load_module("ptv23_complement_target_policy", POLICY_PATH)
    builder = _load_module("build_qwen4b_ptv23_complement", BUILDER_PATH)
    policy_sha256 = hashlib.sha256(Q30_POLICY_PATH.read_bytes()).hexdigest()
    monkeypatch.setattr(policy_module, "APPROVED_TARGET_POLICY_SHA256S", frozenset({policy_sha256}))
    policy = policy_module.load_target_policy(Q30_POLICY_PATH, expected_sha256=policy_sha256)
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
