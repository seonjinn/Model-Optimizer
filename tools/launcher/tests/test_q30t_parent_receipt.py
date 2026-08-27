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
