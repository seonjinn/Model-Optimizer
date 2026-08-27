# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Canonical receipt contracts for the two approved Q30 Thinking parents."""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import replace
from pathlib import Path
from typing import Literal

import pytest
from common.specdec import q30t_parent_receipt as receipt_module
from common.specdec.q30t_parent_receipt import (
    ParentIdentity,
    build_q30t_parent_receipt,
    load_q30t_parent_receipt,
    require_matching_historical_lineage,
)


def _canonical(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def _sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _regular_hashes(root: Path) -> dict[str, str]:
    return {
        path.relative_to(root).as_posix(): _sha(path.read_bytes())
        for path in sorted(root.rglob("*"))
        if path.is_file() and not path.is_symlink()
    }


def _fixture_identity(
    tmp_path: Path, method: Literal["DFlash", "DSpark"] = "DFlash"
) -> ParentIdentity:
    run_name = f"q30t-nemo-{method.lower()}-b8-16n-922609729"
    checkpoint = tmp_path / run_name / "milestones" / "step-025391" / "resume-checkpoint-025391"
    return ParentIdentity(
        method=method,
        completion_job_id=2638009 if method == "DFlash" else 2638015,
        node_count=16,
        parent_run_identity=run_name,
        checkpoint_path=checkpoint,
    )


def _write_parent_tree(identity: ParentIdentity) -> None:
    checkpoint = identity.checkpoint_path
    checkpoint.mkdir(parents=True)
    (checkpoint / "model.safetensors").write_bytes(b"drafter weights")
    (checkpoint / "modelopt_state.pth").write_bytes(b"modelopt lineage")
    (checkpoint / "optimizer.pt").write_bytes(b"optimizer is bound but not restored")
    (checkpoint / "trainer_state.json").write_bytes(_canonical({"global_step": 25391}) + b"\n")
    (checkpoint / "a").mkdir()
    (checkpoint / "a/nested.json").write_bytes(b"nested")
    (checkpoint / "a.txt").write_bytes(b"adjacent")
    hashes = _regular_hashes(checkpoint)
    manifest = {
        "exact_model_path": "../../exported-checkpoint-25391",
        "exact_model_step": 25391,
        "exact_model_sha256": {"model.safetensors": "7" * 64},
        "resume_checkpoint_path": checkpoint.name,
        "resume_checkpoint_step": 25391,
        "resume_checkpoint_sha256": hashes,
        "resume_checkpoint_storage": dict.fromkeys(hashes, "copy"),
    }
    (checkpoint.parent / "manifest.json").write_bytes(_canonical(manifest) + b"\n")
    completion = {
        "checkpoint": str(checkpoint.parents[2]),
        "global_step": 25391,
        "job_id": str(identity.completion_job_id),
        "restart_count": 0,
        "status": "completed",
        "target_step": 25391,
    }
    control = checkpoint.parents[2] / "control"
    control.mkdir()
    (control / "training-complete-s25391.json").write_bytes(_canonical(completion) + b"\n")


def _build_fixture_receipt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    method: Literal["DFlash", "DSpark"] = "DFlash",
) -> tuple[bytes, Path]:
    identity = _fixture_identity(tmp_path, method)
    _write_parent_tree(identity)
    identities = dict(receipt_module.Q30T_PARENT_IDENTITIES)
    identities[method] = identity
    monkeypatch.setattr(receipt_module, "Q30T_PARENT_IDENTITIES", identities)
    raw = build_q30t_parent_receipt(
        checkpoint=identity.checkpoint_path,
        method=method,
        target_revision="a" * 40,
        target_tree_sha256="b" * 64,
        historical_receipt_file_sha256="c" * 64,
        historical_ordered_prompt_uuids_sha256="d" * 64,
        source_commit="e" * 40,
        runtime_sha256="f" * 64,
    )
    path = tmp_path / f"{method.lower()}-parent.json"
    path.write_bytes(raw)
    return raw, path


def _rehashed(payload: dict[str, object]) -> bytes:
    body = {name: value for name, value in payload.items() if name != "receipt_sha256"}
    payload["receipt_sha256"] = _sha(_canonical(body))
    return _canonical(payload) + b"\n"


def test_approved_parent_identities_are_exact_ptyche_completions() -> None:
    """Only the two design-approved completed Ptyche identities are selectable."""
    identities = receipt_module.Q30T_PARENT_IDENTITIES

    assert identities["DFlash"].completion_job_id == 2638009
    assert identities["DSpark"].completion_job_id == 2638015
    assert identities["DFlash"].node_count == identities["DSpark"].node_count == 16
    assert str(identities["DFlash"].checkpoint_path) == (
        "/lustre/fsw/coreai_dlalgo_llm/users/sna/modelopt-qwen3-drafter-training/training/"
        "q30t-nemo-dflash-b8-16n-922609729/milestones/step-025391/"
        "resume-checkpoint-025391"
    )
    assert str(identities["DSpark"].checkpoint_path) == (
        "/lustre/fsw/coreai_dlalgo_llm/users/sna/modelopt-qwen3-drafter-training/training/"
        "q30t-nemo-dspark-b8-16n-922609729/milestones/step-025391/"
        "resume-checkpoint-025391"
    )
    assert frozenset() == receipt_module.APPROVED_Q30T_PARENT_RECEIPT_FILE_SHA256S


def test_q30t_parent_receipt_binds_manifest_files_weights_and_modelopt_state(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The canonical receipt binds the complete checkpoint and ModelOpt lineage."""
    raw, path = _build_fixture_receipt(tmp_path, monkeypatch)
    payload = json.loads(raw)
    monkeypatch.setattr(
        receipt_module, "APPROVED_Q30T_PARENT_RECEIPT_FILE_SHA256S", frozenset({_sha(raw)})
    )

    parent = load_q30t_parent_receipt(path, expected_sha256=_sha(raw))

    assert parent.method == "DFlash"
    assert parent.target_repository == "Qwen/Qwen3-30B-A3B-Thinking-2507"
    assert parent.variant == "Thinking-2507"
    assert parent.block_size == 8
    assert parent.global_step == 25391
    assert parent.modelopt_state_sha256 == _sha(b"modelopt lineage")
    assert parent.historical_receipt_file_sha256 == "c" * 64
    assert parent.historical_ordered_prompt_uuids_sha256 == "d" * 64
    assert parent.receipt_file_sha256 == _sha(raw)
    assert payload["historical_occurrence_count"] == 1_300_000
    assert payload["completion_receipt_path"].endswith("control/training-complete-s25391.json")
    assert len(payload["completion_receipt_file_sha256"]) == 64
    assert {entry["path"] for entry in payload["checkpoint_files"]} == {
        "a.txt",
        "a/nested.json",
        "model.safetensors",
        "modelopt_state.pth",
        "optimizer.pt",
        "trainer_state.json",
    }
    assert {entry["type"] for entry in payload["checkpoint_files"]} == {"regular"}


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("method", "DFlash2"),
        ("target_repository", "Qwen/Qwen3-30B-A3B-Base"),
        ("variant", "Base"),
        ("block_size", 16),
        ("global_step", 14500),
        ("completion_job_id", 2638015),
        ("node_count", 8),
        ("parent_run_identity", "q30t-nemo-dflash-b16-16n-922609729"),
        ("checkpoint_path", "/lustre/oci/partial/checkpoint-25391"),
    ],
)
def test_q30t_parent_rejects_wrong_identity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    field: str,
    value: object,
) -> None:
    """Changing any method-specific production identity fails before approval."""
    raw, path = _build_fixture_receipt(tmp_path, monkeypatch)
    payload = json.loads(raw)
    payload[field] = value
    path.write_bytes(_rehashed(payload))

    with pytest.raises(ValueError, match="Q30 Thinking parent identity"):
        load_q30t_parent_receipt(path, expected_sha256=_sha(path.read_bytes()))


def test_q30t_parent_cannot_self_authorize_a_valid_receipt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A syntactically valid caller digest cannot replace independent review."""
    raw, path = _build_fixture_receipt(tmp_path, monkeypatch)

    with pytest.raises(ValueError, match="reviewed Q30 Thinking parent receipt"):
        load_q30t_parent_receipt(path, expected_sha256=_sha(raw))


def test_q30t_parent_reconciles_checkpoint_and_milestone_bytes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Post-receipt checkpoint replacement fails independent local reconciliation."""
    raw, path = _build_fixture_receipt(tmp_path, monkeypatch)
    monkeypatch.setattr(
        receipt_module, "APPROVED_Q30T_PARENT_RECEIPT_FILE_SHA256S", frozenset({_sha(raw)})
    )
    checkpoint = Path(json.loads(raw)["checkpoint_path"])
    (checkpoint / "model.safetensors").write_bytes(b"swapped weights")

    with pytest.raises(ValueError, match="checkpoint evidence"):
        load_q30t_parent_receipt(path, expected_sha256=_sha(raw))


def test_q30t_parent_rejects_missing_weights_or_modelopt_state(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A parent without both usable weights and ModelOpt state is incomplete."""
    identity = _fixture_identity(tmp_path)
    _write_parent_tree(identity)
    identities = dict(receipt_module.Q30T_PARENT_IDENTITIES)
    identities["DFlash"] = identity
    monkeypatch.setattr(receipt_module, "Q30T_PARENT_IDENTITIES", identities)

    for missing in ("model.safetensors", "modelopt_state.pth"):
        removed = identity.checkpoint_path / missing
        contents = removed.read_bytes()
        removed.unlink()
        with pytest.raises(ValueError, match="weights and ModelOpt state"):
            build_q30t_parent_receipt(
                checkpoint=identity.checkpoint_path,
                method="DFlash",
                target_revision="a" * 40,
                target_tree_sha256="b" * 64,
                historical_receipt_file_sha256="c" * 64,
                historical_ordered_prompt_uuids_sha256="d" * 64,
                source_commit="e" * 40,
                runtime_sha256="f" * 64,
            )
        removed.write_bytes(contents)


def _replace_monolithic_weight_with_index(identity: ParentIdentity, raw: bytes) -> None:
    checkpoint = identity.checkpoint_path
    (checkpoint / "model.safetensors").unlink()
    (checkpoint / "model.safetensors.index.json").write_bytes(raw)


def _refresh_checkpoint_manifest(identity: ParentIdentity) -> None:
    checkpoint = identity.checkpoint_path
    manifest_path = checkpoint.parent / "manifest.json"
    manifest = json.loads(manifest_path.read_bytes())
    hashes = _regular_hashes(checkpoint)
    manifest["resume_checkpoint_sha256"] = hashes
    manifest["resume_checkpoint_storage"] = dict.fromkeys(hashes, "copy")
    manifest_path.write_bytes(_canonical(manifest) + b"\n")


def _build_parent_for_weight_failure(
    identity: ParentIdentity,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _refresh_checkpoint_manifest(identity)
    identities = dict(receipt_module.Q30T_PARENT_IDENTITIES)
    identities["DFlash"] = identity
    monkeypatch.setattr(receipt_module, "Q30T_PARENT_IDENTITIES", identities)
    build_q30t_parent_receipt(
        checkpoint=identity.checkpoint_path,
        method="DFlash",
        target_revision="a" * 40,
        target_tree_sha256="b" * 64,
        historical_receipt_file_sha256="c" * 64,
        historical_ordered_prompt_uuids_sha256="d" * 64,
        source_commit="e" * 40,
        runtime_sha256="f" * 64,
    )


def test_q30t_parent_rejects_index_without_referenced_weight_shards(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An index JSON is metadata, not usable weights by itself."""
    identity = _fixture_identity(tmp_path)
    _write_parent_tree(identity)
    index = {"metadata": {}, "weight_map": {}}
    _replace_monolithic_weight_with_index(identity, _canonical(index) + b"\n")

    with pytest.raises(ValueError, match="weights and ModelOpt state"):
        _build_parent_for_weight_failure(identity, monkeypatch)


def test_q30t_parent_rejects_index_with_missing_or_empty_weight_shards(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Every shard referenced by the HF weight map must exist and be nonempty."""
    identity = _fixture_identity(tmp_path)
    _write_parent_tree(identity)
    index = {"metadata": {}, "weight_map": {"model.layer.weight": "model-00001.safetensors"}}
    _replace_monolithic_weight_with_index(identity, _canonical(index) + b"\n")

    with pytest.raises(ValueError, match="weights and ModelOpt state"):
        _build_parent_for_weight_failure(identity, monkeypatch)

    (identity.checkpoint_path / "model-00001.safetensors").write_bytes(b"")
    with pytest.raises(ValueError, match="weights and ModelOpt state"):
        _build_parent_for_weight_failure(identity, monkeypatch)


@pytest.mark.parametrize("reference", ["../foreign.safetensors", "/foreign.safetensors", ""])
def test_q30t_parent_rejects_invalid_index_weight_shard_paths(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    reference: str,
) -> None:
    """HF index shard references cannot escape or ambiguously name the checkpoint."""
    identity = _fixture_identity(tmp_path)
    _write_parent_tree(identity)
    index = {"metadata": {}, "weight_map": {"model.layer.weight": reference}}
    _replace_monolithic_weight_with_index(identity, _canonical(index) + b"\n")

    with pytest.raises(ValueError, match="weights and ModelOpt state"):
        _build_parent_for_weight_failure(identity, monkeypatch)


def test_q30t_parent_rejects_duplicate_or_invalid_index_weight_map_entries(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Duplicate parameter keys and non-string shard references are not exact HF indexes."""
    identity = _fixture_identity(tmp_path)
    _write_parent_tree(identity)
    duplicate_key = (
        b'{"metadata":{},"weight_map":{"model.layer.weight":"model-00001.safetensors",'
        b'"model.layer.weight":"model-00002.safetensors"}}\n'
    )
    _replace_monolithic_weight_with_index(identity, duplicate_key)

    with pytest.raises(ValueError, match="weights and ModelOpt state"):
        _build_parent_for_weight_failure(identity, monkeypatch)

    invalid_reference = {"metadata": {}, "weight_map": {"model.layer.weight": 7}}
    (identity.checkpoint_path / "model.safetensors.index.json").write_bytes(
        _canonical(invalid_reference) + b"\n"
    )
    with pytest.raises(ValueError, match="weights and ModelOpt state"):
        _build_parent_for_weight_failure(identity, monkeypatch)


def test_q30t_parent_accepts_complete_hf_weight_index(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A canonical HF index is usable when all referenced shards are bound and nonempty."""
    identity = _fixture_identity(tmp_path)
    _write_parent_tree(identity)
    checkpoint = identity.checkpoint_path
    (checkpoint / "model-00001-of-00002.safetensors").write_bytes(b"first shard")
    (checkpoint / "model-00002-of-00002.safetensors").write_bytes(b"second shard")
    index = {
        "metadata": {"total_size": 22},
        "weight_map": {
            "model.layer.0.weight": "model-00001-of-00002.safetensors",
            "model.layer.1.weight": "model-00002-of-00002.safetensors",
        },
    }
    _replace_monolithic_weight_with_index(identity, _canonical(index) + b"\n")
    _refresh_checkpoint_manifest(identity)
    identities = dict(receipt_module.Q30T_PARENT_IDENTITIES)
    identities["DFlash"] = identity
    monkeypatch.setattr(receipt_module, "Q30T_PARENT_IDENTITIES", identities)

    raw = build_q30t_parent_receipt(
        checkpoint=checkpoint,
        method="DFlash",
        target_revision="a" * 40,
        target_tree_sha256="b" * 64,
        historical_receipt_file_sha256="c" * 64,
        historical_ordered_prompt_uuids_sha256="d" * 64,
        source_commit="e" * 40,
        runtime_sha256="f" * 64,
    )

    assert json.loads(raw)["checkpoint_tree_sha256"]


def test_q30t_parent_closes_descriptor_when_fdopen_raises_value_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A non-OSError fdopen failure cannot leak the already-open descriptor."""
    candidate = tmp_path / "candidate.bin"
    candidate.write_bytes(b"payload")
    directory_fd = os.open(tmp_path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    expected = os.stat(candidate.name, dir_fd=directory_fd, follow_symlinks=False)
    real_open = os.open
    real_close = os.close
    opened: list[int] = []
    closed: list[int] = []

    def tracked_open(*args: object, **kwargs: object) -> int:
        descriptor = real_open(*args, **kwargs)  # type: ignore[arg-type]
        opened.append(descriptor)
        return descriptor

    def tracked_close(descriptor: int) -> None:
        closed.append(descriptor)
        real_close(descriptor)

    def fail_fdopen(*_args: object, **_kwargs: object) -> None:
        raise ValueError("synthetic fdopen failure")

    monkeypatch.setattr(receipt_module.os, "open", tracked_open)
    monkeypatch.setattr(receipt_module.os, "close", tracked_close)
    monkeypatch.setattr(receipt_module.os, "fdopen", fail_fdopen)
    try:
        with pytest.raises(ValueError, match="synthetic fdopen failure"):
            receipt_module._read_regular_at(
                directory_fd,
                candidate.name,
                str(candidate),
                expected,
                retain=False,
                require_single_link=False,
            )
        assert opened[-1] in closed
    finally:
        if opened and opened[-1] not in closed:
            real_close(opened[-1])
        real_close(directory_fd)


def test_q30t_parent_requires_completed_exact_job_evidence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An exact-looking milestone without the completed-job receipt is not a parent."""
    identity = _fixture_identity(tmp_path)
    _write_parent_tree(identity)
    identities = dict(receipt_module.Q30T_PARENT_IDENTITIES)
    identities["DFlash"] = identity
    monkeypatch.setattr(receipt_module, "Q30T_PARENT_IDENTITIES", identities)
    (identity.checkpoint_path.parents[2] / "control/training-complete-s25391.json").unlink()

    with pytest.raises(ValueError, match="completion evidence"):
        build_q30t_parent_receipt(
            checkpoint=identity.checkpoint_path,
            method="DFlash",
            target_revision="a" * 40,
            target_tree_sha256="b" * 64,
            historical_receipt_file_sha256="c" * 64,
            historical_ordered_prompt_uuids_sha256="d" * 64,
            source_commit="e" * 40,
            runtime_sha256="f" * 64,
        )


def test_q30t_parent_rejects_extra_completion_evidence_fields(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A self-authored completion extension cannot enter the parent trust root."""
    identity = _fixture_identity(tmp_path)
    _write_parent_tree(identity)
    identities = dict(receipt_module.Q30T_PARENT_IDENTITIES)
    identities["DFlash"] = identity
    monkeypatch.setattr(receipt_module, "Q30T_PARENT_IDENTITIES", identities)
    completion_path = identity.checkpoint_path.parents[2] / "control/training-complete-s25391.json"
    completion = json.loads(completion_path.read_bytes())
    completion["caller_claim"] = "completed"
    completion_path.write_bytes(_canonical(completion) + b"\n")

    with pytest.raises(ValueError, match="completion evidence does not reconcile"):
        build_q30t_parent_receipt(
            checkpoint=identity.checkpoint_path,
            method="DFlash",
            target_revision="a" * 40,
            target_tree_sha256="b" * 64,
            historical_receipt_file_sha256="c" * 64,
            historical_ordered_prompt_uuids_sha256="d" * 64,
            source_commit="e" * 40,
            runtime_sha256="f" * 64,
        )


def test_q30t_parent_rejects_symlinked_checkpoint_entries(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Checkpoint traversal never follows a symbolic link."""
    identity = _fixture_identity(tmp_path)
    _write_parent_tree(identity)
    identities = dict(receipt_module.Q30T_PARENT_IDENTITIES)
    identities["DFlash"] = identity
    monkeypatch.setattr(receipt_module, "Q30T_PARENT_IDENTITIES", identities)
    target = tmp_path / "foreign"
    target.write_bytes(b"foreign")
    os.symlink(target, identity.checkpoint_path / "foreign-link")

    with pytest.raises(ValueError, match="unsupported checkpoint entry"):
        build_q30t_parent_receipt(
            checkpoint=identity.checkpoint_path,
            method="DFlash",
            target_revision="a" * 40,
            target_tree_sha256="b" * 64,
            historical_receipt_file_sha256="c" * 64,
            historical_ordered_prompt_uuids_sha256="d" * 64,
            source_commit="e" * 40,
            runtime_sha256="f" * 64,
        )


@pytest.mark.parametrize(
    "field", ["historical_receipt_file_sha256", "historical_ordered_prompt_uuids_sha256"]
)
def test_dflash_and_dspark_must_share_historical_lineage(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    field: str,
) -> None:
    """Both the 1.3M receipt and its ordered UUID digest must match."""
    dflash_raw, dflash_path = _build_fixture_receipt(tmp_path / "dflash", monkeypatch, "DFlash")
    dspark_raw, dspark_path = _build_fixture_receipt(tmp_path / "dspark", monkeypatch, "DSpark")
    approved = {_sha(dflash_raw), _sha(dspark_raw)}
    monkeypatch.setattr(
        receipt_module, "APPROVED_Q30T_PARENT_RECEIPT_FILE_SHA256S", frozenset(approved)
    )
    dflash = load_q30t_parent_receipt(dflash_path, expected_sha256=_sha(dflash_raw))
    dspark = load_q30t_parent_receipt(dspark_path, expected_sha256=_sha(dspark_raw))
    dspark = replace(dspark, **{field: "9" * 64})

    with pytest.raises(ValueError, match="historical lineage"):
        require_matching_historical_lineage(dflash, dspark)


def test_parent_pair_rejects_duplicate_methods(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Lineage comparison requires one parent from each approved method."""
    raw, path = _build_fixture_receipt(tmp_path, monkeypatch)
    monkeypatch.setattr(
        receipt_module, "APPROVED_Q30T_PARENT_RECEIPT_FILE_SHA256S", frozenset({_sha(raw)})
    )
    dflash = load_q30t_parent_receipt(path, expected_sha256=_sha(raw))

    with pytest.raises(ValueError, match="DFlash and DSpark"):
        require_matching_historical_lineage(dflash, dflash)
