# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Q30 Thinking tokenizer receipt contracts."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import sys
import tracemalloc
from pathlib import Path

import pytest
from common.specdec import q30t_tokenizer_receipt as receipt_module
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


def _make_q30_snapshot(tmp_path: Path, im_start_token_id: int = 17) -> Path:
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    (snapshot / "tokenizer_config.json").write_bytes(
        _canonical({"chat_template": OFFICIAL_TEMPLATE})
    )
    (snapshot / "tokenizer.json").write_bytes(
        _canonical(
            {
                "added_tokens": [
                    {"content": "<|im_start|>", "id": im_start_token_id},
                    {"content": "<|im_end|>", "id": im_start_token_id + 1},
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


def _rehashed(payload: dict[str, object]) -> bytes:
    body = {name: value for name, value in payload.items() if name != "receipt_sha256"}
    payload["receipt_sha256"] = hashlib.sha256(_canonical(body)).hexdigest()
    return _canonical(payload) + b"\n"


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
    assert (
        parsed["training_chat_template_sha256"]
        == hashlib.sha256(TRAINING_TEMPLATE.encode()).hexdigest()
    )
    assert parsed["im_start_token_id"] == 17
    assert parsed["im_end_token_id"] == 18
    assert verify_q30t_tokenizer_receipt(receipt) == parsed


@pytest.mark.parametrize("field", ["snapshot_tree_sha256", "training_chat_template_sha256"])
def test_q30t_receipt_rejects_tampering(tmp_path: Path, field: str) -> None:
    """Rehashed derived receipt fields still fail snapshot reconciliation."""
    payload = json.loads(
        build_q30t_tokenizer_receipt(
            snapshot=_make_q30_snapshot(tmp_path),
            repository=Q30_REPOSITORY,
            revision=FIXTURE_Q30_REVISION,
        )
    )
    payload[field] = "b" * 64

    with pytest.raises(ValueError, match="evidence"):
        verify_q30t_tokenizer_receipt(_rehashed(payload))


def test_q30t_receipt_rejects_missing_canonical_newline(tmp_path: Path) -> None:
    """Canonical receipt bytes include the required terminating newline."""
    receipt = build_q30t_tokenizer_receipt(
        snapshot=_make_q30_snapshot(tmp_path),
        repository=Q30_REPOSITORY,
        revision=FIXTURE_Q30_REVISION,
    )

    with pytest.raises(ValueError, match="identity"):
        verify_q30t_tokenizer_receipt(receipt.rstrip(b"\n"))


def test_q30t_builder_recomputes_tokenizer_receipt_evidence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
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
    monkeypatch.setattr(
        builder,
        "APPROVED_Q30T_TOKENIZER_RECEIPT_FILE_SHA256S",
        frozenset({hashlib.sha256(receipt).hexdigest()}),
    )
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


def test_q30t_builder_fails_closed_without_a_reviewed_receipt_root(tmp_path: Path) -> None:
    """A caller-provided receipt SHA cannot authorize an otherwise valid Q30 receipt."""
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

    with pytest.raises(builder.ComplementError, match="reviewed Q30 tokenizer receipt"):
        builder.load_tokenizer_trust(
            receipt_path,
            expected_sha256=hashlib.sha256(receipt).hexdigest(),
            policy=policy,
        )


def test_q30t_receipt_rejects_hardlinked_snapshot_files(tmp_path: Path) -> None:
    """A closed tokenizer snapshot cannot admit a multiply-linked regular file."""
    snapshot = _make_q30_snapshot(tmp_path)
    os.link(snapshot / "tokenizer.json", snapshot / "duplicate-tokenizer.json")

    with pytest.raises(ValueError, match="hardlinks"):
        build_q30t_tokenizer_receipt(
            snapshot=snapshot,
            repository=Q30_REPOSITORY,
            revision=FIXTURE_Q30_REVISION,
        )


def test_q30t_builder_uses_an_imported_receipt_verifier() -> None:
    """Q30 verification cannot dynamically execute a mutable source pathname."""
    source = BUILDER_PATH.read_text(encoding="utf-8")

    assert "exec_module" not in source


def test_q30t_receipt_rejects_boolean_special_token_ids(tmp_path: Path) -> None:
    """Boolean JSON values cannot alias valid integer tokenizer IDs."""
    payload = json.loads(
        build_q30t_tokenizer_receipt(
            snapshot=_make_q30_snapshot(tmp_path, im_start_token_id=1),
            repository=Q30_REPOSITORY,
            revision=FIXTURE_Q30_REVISION,
        )
    )
    payload["im_start_token_id"] = True

    with pytest.raises(ValueError, match="identity"):
        verify_q30t_tokenizer_receipt(_rehashed(payload))


def test_q30t_snapshot_uses_nonblocking_nofollow_regular_opens(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A regular-file path replacement cannot turn snapshot hashing into a FIFO wait."""
    snapshot = _make_q30_snapshot(tmp_path)
    observed_flags: list[int] = []
    original_open = receipt_module.os.open

    def checked_open(path, flags, *args, **kwargs):
        if path == "tokenizer.json":
            observed_flags.append(flags)
        return original_open(path, flags, *args, **kwargs)

    monkeypatch.setattr(receipt_module.os, "open", checked_open)
    snapshot_tree_sha256(snapshot)

    assert observed_flags
    assert all(flags & os.O_NONBLOCK for flags in observed_flags)
    assert all(flags & os.O_NOFOLLOW for flags in observed_flags)


def test_q30t_snapshot_reconciles_pre_stat_with_opened_fd(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Changing a name after pre-stat cannot substitute a different opened file."""
    snapshot = _make_q30_snapshot(tmp_path)
    directory_fd = os.open(snapshot, os.O_RDONLY)
    expected = os.stat("tokenizer.json", dir_fd=directory_fd, follow_symlinks=False)
    original_open = receipt_module.os.open

    def substituted_open(path, flags, *args, **kwargs):
        if path == "tokenizer.json":
            return original_open("assets/tokenizer.model", flags, *args, **kwargs)
        return original_open(path, flags, *args, **kwargs)

    monkeypatch.setattr(receipt_module.os, "open", substituted_open)
    try:
        with pytest.raises(ValueError, match="not regular"):
            receipt_module._read_regular_at(
                directory_fd, "tokenizer.json", "tokenizer.json", expected, False
            )
    finally:
        os.close(directory_fd)


def test_q30t_snapshot_closes_fd_when_fdopen_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A stream-wrapper failure must not leak the already opened snapshot descriptor."""
    snapshot = _make_q30_snapshot(tmp_path)
    directory_fd = os.open(snapshot, os.O_RDONLY)
    expected = os.stat("tokenizer.json", dir_fd=directory_fd, follow_symlinks=False)
    opened: list[int] = []
    closed: list[int] = []
    original_open = receipt_module.os.open
    original_close = receipt_module.os.close

    def recorded_open(path, flags, *args, **kwargs):
        descriptor = original_open(path, flags, *args, **kwargs)
        if path == "tokenizer.json":
            opened.append(descriptor)
        return descriptor

    def recorded_close(descriptor: int) -> None:
        closed.append(descriptor)
        original_close(descriptor)

    monkeypatch.setattr(receipt_module.os, "open", recorded_open)
    monkeypatch.setattr(receipt_module.os, "close", recorded_close)
    monkeypatch.setattr(
        receipt_module.os,
        "fdopen",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(OSError("boom")),
    )
    try:
        with pytest.raises(ValueError, match="cannot be read"):
            receipt_module._read_regular_at(
                directory_fd, "tokenizer.json", "tokenizer.json", expected, False
            )
    finally:
        original_close(directory_fd)

    assert opened == closed


def test_q30t_snapshot_streams_unneeded_file_bytes(tmp_path: Path) -> None:
    """Large unrelated snapshot files do not remain resident while computing the tree."""
    snapshot = _make_q30_snapshot(tmp_path)
    (snapshot / "assets" / "large.bin").write_bytes(b"x" * (8 * 1024 * 1024))

    tracemalloc.start()
    try:
        snapshot_tree_sha256(snapshot)
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()

    assert peak < 4 * 1024 * 1024
