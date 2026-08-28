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
from common.specdec import q30t_tree_digest as tree_digest_module
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


def _write_directory_completion_marker(root: Path) -> None:
    (root / receipt_module.Q30_DIRECTORY_COMPLETION_MARKER).write_bytes(
        b"q30-directory-publication-complete-v1\n"
    )


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


def test_q30t_parent_approval_roots_are_the_independently_reviewed_pair() -> None:
    """Only the independently reviewed DFlash and DSpark roots are approved."""
    assert receipt_module.APPROVED_Q30T_PARENT_RECEIPT_FILE_SHA256S == frozenset(  # noqa: SIM300
        {
            "393a2b7c5cbe2037914cbfa531d1d6fdce2bcac2f4204aa001db89d345c3a500",
            "d5416b8fd9644802b42f04511791418bf00dd023f8c5df96593010be95ff2571",
        }
    )


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


def test_q30t_parent_walk_rebinds_the_absolute_root_after_traversal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The checkpoint walker rejects a root replaced at its final close window."""
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    (checkpoint / "model.safetensors").write_bytes(b"weights")
    replacement = tmp_path / "replacement"
    replacement.mkdir()
    (replacement / "model.safetensors").write_bytes(b"weights")
    displaced = tmp_path / "displaced"
    original_open = receipt_module.os.open
    original_close = receipt_module.os.close
    root_descriptors: set[int] = set()
    swapped = False

    def capture_root_open(path: object, *args: object, **kwargs: object) -> int:
        descriptor = original_open(path, *args, **kwargs)  # type: ignore[arg-type]
        if path == checkpoint and kwargs.get("dir_fd") is None:
            root_descriptors.add(descriptor)
        return descriptor

    def swap_before_root_close(descriptor: int) -> None:
        nonlocal swapped
        if descriptor in root_descriptors and not swapped:
            swapped = True
            checkpoint.rename(displaced)
            replacement.rename(checkpoint)
        original_close(descriptor)

    monkeypatch.setattr(receipt_module.os, "open", capture_root_open)
    monkeypatch.setattr(receipt_module.os, "close", swap_before_root_close)

    with pytest.raises(ValueError, match=r"root changed|absolute root"):
        receipt_module._walk_checkpoint(checkpoint)


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


def _write_historical_audit(path: Path) -> tuple[str, str]:
    occurrences = ["1" * 64, "2" * 64, "1" * 64]
    exclusions = sorted(set(occurrences))
    occurrence_sha256 = _sha(_canonical(occurrences))
    body = {
        "source_revision": "5c89e01dd720ae0f4058445ed49c5fb68a03c76e",
        "source_manifest_sha256": "3" * 64,
        "row_count": 3,
        "split_rows": {"chat": 3},
        "files": [
            {
                "path": "/lustre/fixture/chat-00000-of-00001.parquet",
                "bytes": 17,
                "sha256": "4" * 64,
            }
        ],
        "occurrence_count": 3,
        "unique_prompt_count": 2,
        "occurrence_prompt_ids": occurrences,
        "occurrence_prompt_ids_sha256": occurrence_sha256,
        "exclusion_prompt_ids": exclusions,
        "exclusion_prompt_ids_sha256": _sha(_canonical(exclusions)),
        "duplicate_uuid_multiplicity": {"1" * 64: 2},
        "physical_row_count": 4,
        "selection_policy": "hf-streaming-sorted-parquet-take",
        "selection_boundary": {
            "file": "chat-00000-of-00001.parquet",
            "rows_selected": 3,
            "rows_available": 4,
            "excluded_tail_rows": 1,
        },
    }
    payload = body | {"receipt_sha256": _sha(_canonical(body))}
    raw = _canonical(payload) + b"\n"
    path.write_bytes(raw)
    return _sha(raw), occurrence_sha256


def _target_asset_arguments(target: Path, monkeypatch: pytest.MonkeyPatch) -> list[str]:
    manifest = target.parent / "manifest"
    manifest.mkdir()
    records = [
        (
            path.relative_to(target).as_posix(),
            path.stat().st_size,
            _sha(path.read_bytes()),
        )
        for path in sorted(target.rglob("*"))
        if path.is_file()
    ]
    model_sha256 = manifest / "model.sha256"
    model_sha256.write_text(
        "".join(f"{digest}  ./{path}\n" for path, _, digest in records), encoding="utf-8"
    )
    model_manifest_sha256 = _sha(model_sha256.read_bytes())
    identity = manifest / "identity.json"
    identity.write_bytes(
        _canonical(
            {
                "artifact": "Qwen/Qwen3-30B-A3B-Thinking-2507",
                "bytes": sum(size for _, size, _ in records),
                "files": len(records),
                "model_sha256_manifest": model_manifest_sha256,
                "source_revision": "a" * 40,
                "symlinks": 0,
            }
        )
        + b"\n"
    )
    identity_sha256 = _sha(identity.read_bytes())
    monkeypatch.setattr(tree_digest_module, "APPROVED_Q30T_MODEL_ROOT", target)
    monkeypatch.setattr(tree_digest_module, "Q30T_MODEL_REVISION", "a" * 40)
    monkeypatch.setattr(tree_digest_module, "APPROVED_Q30T_MODEL_IDENTITY_SHA256", identity_sha256)
    monkeypatch.setattr(
        tree_digest_module,
        "APPROVED_Q30T_MODEL_SHA256_MANIFEST_SHA256",
        model_manifest_sha256,
    )
    return [
        "--target-identity-path",
        str(identity),
        "--target-identity-sha256",
        identity_sha256,
        "--target-model-sha256-path",
        str(model_sha256),
        "--target-model-sha256-sha256",
        model_manifest_sha256,
    ]


def _install_parent_pair(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Path, Path]:
    identities = {
        method: _fixture_identity(tmp_path / method.lower(), method)
        for method in ("DFlash", "DSpark")
    }
    for identity in identities.values():
        _write_parent_tree(identity)
    monkeypatch.setattr(receipt_module, "Q30T_PARENT_IDENTITIES", identities)
    monkeypatch.setattr(receipt_module, "Q30T_HISTORICAL_OCCURRENCES", 3)
    monkeypatch.setattr(receipt_module, "Q30T_HISTORICAL_SPLIT_ROWS", {"chat": 3})
    return identities["DFlash"].checkpoint_path, identities["DSpark"].checkpoint_path


def test_parent_cli_derives_and_cross_binds_history_target_runtime_and_parents(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """The compute-side CLI derives every heavy digest and emits a byte-stable pair."""
    dflash, dspark = _install_parent_pair(tmp_path, monkeypatch)
    audit = tmp_path / "AUDIT.json"
    audit_file_sha256, ordered_sha256 = _write_historical_audit(audit)
    monkeypatch.setattr(
        receipt_module, "APPROVED_Q30T_HISTORICAL_AUDIT_FILE_SHA256", audit_file_sha256
    )
    monkeypatch.setattr(
        receipt_module,
        "APPROVED_Q30T_HISTORICAL_ORDERED_OCCURRENCES_SHA256",
        ordered_sha256,
    )
    target = tmp_path / "target"
    target.mkdir()
    (target / "config.json").write_bytes(b"{}\n")
    (target / "model.safetensors").write_bytes(b"target weights")
    target_asset_arguments = _target_asset_arguments(target, monkeypatch)
    runtime = tmp_path / "runtime.tar.zst"
    runtime.write_bytes(b"runtime archive")
    first = tmp_path / "receipts-first"
    second = tmp_path / "receipts-second"

    common = [
        "--historical-audit",
        str(audit),
        "--target-model-root",
        str(target),
        "--target-revision",
        "a" * 40,
        *target_asset_arguments,
        "--dflash-parent-root",
        str(dflash),
        "--dspark-parent-root",
        str(dspark),
        "--runtime-archive",
        str(runtime),
        "--source-commit",
        "b" * 40,
    ]
    assert receipt_module.main([*common, "--output-root", str(first)]) == 0
    first_summary = json.loads(capsys.readouterr().out)
    assert receipt_module.main([*common, "--output-root", str(second)]) == 0
    capsys.readouterr()

    dflash_raw = (first / "q30t-dflash-parent.json").read_bytes()
    dspark_raw = (first / "q30t-dspark-parent.json").read_bytes()
    dflash_payload = json.loads(dflash_raw)
    dspark_payload = json.loads(dspark_raw)
    expected_target_entries = [
        {"path": "config.json", "type": "regular", "size": 3, "sha256": _sha(b"{}\n")},
        {
            "path": "model.safetensors",
            "type": "regular",
            "size": 14,
            "sha256": _sha(b"target weights"),
        },
    ]
    expected_target_tree_sha256 = _sha(_canonical(expected_target_entries))

    assert first_summary == {
        "DFlash": {
            "path": str(first / "q30t-dflash-parent.json"),
            "sha256": _sha(dflash_raw),
        },
        "DSpark": {
            "path": str(first / "q30t-dspark-parent.json"),
            "sha256": _sha(dspark_raw),
        },
    }
    for payload in (dflash_payload, dspark_payload):
        assert payload["historical_occurrence_count"] == 3
        assert payload["historical_receipt_file_sha256"] == audit_file_sha256
        assert payload["historical_ordered_prompt_uuids_sha256"] == ordered_sha256
        assert payload["target_tree_sha256"] == expected_target_tree_sha256
        assert payload["runtime_sha256"] == _sha(b"runtime archive")
    assert dflash_payload["method"] == "DFlash"
    assert dspark_payload["method"] == "DSpark"
    assert (second / "q30t-dflash-parent.json").read_bytes() == dflash_raw
    assert (second / "q30t-dspark-parent.json").read_bytes() == dspark_raw


def test_parent_cli_rejects_tampered_historical_internal_receipt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An AUDIT file hash alone cannot replace its canonical internal receipt proof."""
    dflash, dspark = _install_parent_pair(tmp_path, monkeypatch)
    audit = tmp_path / "AUDIT.json"
    audit_file_sha256, ordered_sha256 = _write_historical_audit(audit)
    monkeypatch.setattr(
        receipt_module, "APPROVED_Q30T_HISTORICAL_AUDIT_FILE_SHA256", audit_file_sha256
    )
    monkeypatch.setattr(
        receipt_module,
        "APPROVED_Q30T_HISTORICAL_ORDERED_OCCURRENCES_SHA256",
        ordered_sha256,
    )
    payload = json.loads(audit.read_bytes())
    payload["occurrence_prompt_ids_sha256"] = "9" * 64
    audit.write_bytes(_canonical(payload) + b"\n")
    target = tmp_path / "target"
    target.mkdir()
    (target / "config.json").write_bytes(b"{}")
    target_asset_arguments = _target_asset_arguments(target, monkeypatch)
    runtime = tmp_path / "runtime.tar.zst"
    runtime.write_bytes(b"runtime")

    with pytest.raises(ValueError, match="historical AUDIT"):
        receipt_module.main(
            [
                "--historical-audit",
                str(audit),
                "--target-model-root",
                str(target),
                "--target-revision",
                "a" * 40,
                *target_asset_arguments,
                "--dflash-parent-root",
                str(dflash),
                "--dspark-parent-root",
                str(dspark),
                "--runtime-archive",
                str(runtime),
                "--source-commit",
                "b" * 40,
                "--output-root",
                str(tmp_path / "receipts"),
            ]
        )


def test_parent_cli_rejects_a_self_consistent_noncanonical_historical_split(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A rehashed alternate 1.3M selection cannot impersonate the known PTV2 prefix."""
    monkeypatch.setattr(receipt_module, "Q30T_HISTORICAL_OCCURRENCES", 3)
    monkeypatch.setattr(receipt_module, "Q30T_HISTORICAL_SPLIT_ROWS", {"chat": 3}, raising=False)
    audit = tmp_path / "AUDIT.json"
    _write_historical_audit(audit)
    payload = json.loads(audit.read_bytes())
    payload["split_rows"] = {"code": 3}
    body = {name: value for name, value in payload.items() if name != "receipt_sha256"}
    payload["receipt_sha256"] = _sha(_canonical(body))
    audit.write_bytes(_canonical(payload) + b"\n")

    with pytest.raises(ValueError, match="historical AUDIT identity"):
        receipt_module._historical_audit_identity(audit)


def test_parent_cli_rejects_rehashed_history_that_differs_from_reviewed_ordering(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Internal consistency cannot replace the independently reviewed 1.3M roots."""
    monkeypatch.setattr(receipt_module, "Q30T_HISTORICAL_OCCURRENCES", 3)
    monkeypatch.setattr(receipt_module, "Q30T_HISTORICAL_SPLIT_ROWS", {"chat": 3})
    audit = tmp_path / "AUDIT.json"
    reviewed_file_sha256, reviewed_ordered_sha256 = _write_historical_audit(audit)
    monkeypatch.setattr(
        receipt_module,
        "APPROVED_Q30T_HISTORICAL_AUDIT_FILE_SHA256",
        reviewed_file_sha256,
        raising=False,
    )
    monkeypatch.setattr(
        receipt_module,
        "APPROVED_Q30T_HISTORICAL_ORDERED_OCCURRENCES_SHA256",
        reviewed_ordered_sha256,
        raising=False,
    )
    payload = json.loads(audit.read_bytes())
    occurrences = payload["occurrence_prompt_ids"][1:] + payload["occurrence_prompt_ids"][:1]
    payload["occurrence_prompt_ids"] = occurrences
    payload["occurrence_prompt_ids_sha256"] = _sha(_canonical(occurrences))
    body = {name: value for name, value in payload.items() if name != "receipt_sha256"}
    payload["receipt_sha256"] = _sha(_canonical(body))
    audit.write_bytes(_canonical(payload) + b"\n")

    with pytest.raises(ValueError, match="reviewed historical AUDIT"):
        receipt_module._historical_audit_identity(audit)


def test_parent_pair_publication_creates_missing_parents_and_adopts_exact_retry(
    tmp_path: Path,
) -> None:
    """A completed pair is atomically reusable after retry without replacement."""
    output = tmp_path / "missing" / "durable" / "parent-pair"
    receipts = {"DFlash": b"dflash\n", "DSpark": b"dspark\n"}

    first = receipt_module._publish_receipt_pair(output, receipts)
    first_bytes = {path.name: path.read_bytes() for path in output.iterdir()}
    second = receipt_module._publish_receipt_pair(output, receipts)

    assert second == first
    assert {path.name: path.read_bytes() for path in output.iterdir()} == first_bytes


def test_parent_pair_adoption_rejects_a_corrupt_completion_marker(tmp_path: Path) -> None:
    """Exact retry adoption requires the durable marker inode and complete state."""
    output = tmp_path / "parent-pair"
    receipts = {"DFlash": b"dflash\n", "DSpark": b"dspark\n"}
    receipt_module._publish_receipt_pair(output, receipts)
    (output / receipt_module.Q30_DIRECTORY_COMPLETION_MARKER).write_bytes(b"foreign\n")

    with pytest.raises(FileExistsError, match="completion marker"):
        receipt_module._publish_receipt_pair(output, receipts)


def test_parent_pair_publication_preserves_a_foreign_preexisting_partial(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A failed partial mkdir never grants ownership of another attempt's directory."""
    monkeypatch.setenv("SLURM_JOB_ID", "4242")
    monkeypatch.setattr(receipt_module.os, "getpid", lambda: 31337)
    output = tmp_path / "parent-pair"
    partial = tmp_path / ".parent-pair.partial-4242-31337"
    partial.mkdir()
    sentinel = partial / "foreign"
    sentinel.write_bytes(b"preserve\n")

    with pytest.raises(FileExistsError):
        receipt_module._publish_receipt_pair(output, {"DFlash": b"dflash\n", "DSpark": b"dspark\n"})

    assert sentinel.read_bytes() == b"preserve\n"


def test_parent_pair_publication_preserves_a_replacement_partial_inode(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Cleanup stops if the pathname no longer names the partial inode we created."""
    output = tmp_path / "parent-pair"
    replacement = tmp_path / "replacement"

    def replace_partial(partial: Path, destination: Path) -> None:
        del destination
        for child in partial.iterdir():
            child.unlink()
        partial.rmdir()
        replacement.mkdir()
        replacement.rename(partial)
        (partial / "foreign").write_bytes(b"preserve\n")
        raise RuntimeError("injected publication failure")

    monkeypatch.setattr(receipt_module, "atomic_publish_directory", replace_partial)

    with pytest.raises(RuntimeError, match="injected"):
        receipt_module._publish_receipt_pair(output, {"DFlash": b"dflash\n", "DSpark": b"dspark\n"})

    partials = list(tmp_path.glob(".parent-pair.partial-*"))
    assert len(partials) == 1
    assert (partials[0] / "foreign").read_bytes() == b"preserve\n"


def test_parent_pair_failure_never_cleans_a_replaced_partial_child(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Failed publication retains recovery files without a stat-to-unlink race."""
    monkeypatch.setenv("SLURM_JOB_ID", "4242")
    monkeypatch.setattr(receipt_module.os, "getpid", lambda: 31337)
    output = tmp_path / "parent-pair"
    partial = tmp_path / ".parent-pair.partial-4242-31337"
    receipt = partial / "q30t-dflash-parent.json"
    displaced = partial / "displaced-dflash-parent.json"
    real_stat = receipt_module.os.stat
    attacked = False

    def fail_publication(_partial: Path, _destination: Path) -> None:
        raise RuntimeError("injected publication failure")

    def replace_before_cleanup(
        path: str | bytes | Path,
        *,
        dir_fd: int | None = None,
        follow_symlinks: bool = True,
    ) -> os.stat_result:
        nonlocal attacked
        status = real_stat(path, dir_fd=dir_fd, follow_symlinks=follow_symlinks)
        if not attacked and dir_fd is None and Path(os.fsdecode(path)) == receipt:
            attacked = True
            receipt.rename(displaced)
            receipt.write_bytes(b"foreign\n")
        return status

    monkeypatch.setattr(receipt_module, "atomic_publish_directory", fail_publication)
    monkeypatch.setattr(receipt_module.os, "stat", replace_before_cleanup)

    with pytest.raises(RuntimeError, match="injected publication failure"):
        receipt_module._publish_receipt_pair(output, {"DFlash": b"dflash\n", "DSpark": b"dspark\n"})

    assert not attacked
    assert receipt.read_bytes() == b"dflash\n"
    assert (partial / "q30t-dspark-parent.json").read_bytes() == b"dspark\n"


def test_parent_pair_publication_adopts_an_exact_ambiguous_directory_result(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A post-rename ambiguity succeeds only after exact pair authentication."""
    output = tmp_path / "parent-pair"
    receipts = {"DFlash": b"dflash\n", "DSpark": b"dspark\n"}

    def ambiguous_publish(partial: Path, destination: Path) -> None:
        partial.rename(destination)
        _write_directory_completion_marker(destination)
        raise RuntimeError("injected post-rename ambiguity")

    monkeypatch.setattr(receipt_module, "atomic_publish_directory", ambiguous_publish)

    result = receipt_module._publish_receipt_pair(output, receipts)

    assert set(result) == {"DFlash", "DSpark"}
    assert (output / "q30t-dflash-parent.json").read_bytes() == b"dflash\n"
    assert (output / "q30t-dspark-parent.json").read_bytes() == b"dspark\n"


def test_parent_pair_successful_publish_is_reauthenticated(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A nominal directory-publisher success cannot bypass exact pair adoption."""
    output = tmp_path / "parent-pair"
    receipts = {"DFlash": b"dflash\n", "DSpark": b"dspark\n"}

    def successful_foreign_publish(partial: Path, destination: Path) -> None:
        partial.rename(destination)
        _write_directory_completion_marker(destination)
        (destination / "q30t-dflash-parent.json").write_bytes(b"foreign\n")

    monkeypatch.setattr(receipt_module, "atomic_publish_directory", successful_foreign_publish)

    with pytest.raises(FileExistsError, match="differs"):
        receipt_module._publish_receipt_pair(output, receipts)


def test_parent_pair_publication_never_suppresses_failed_adoption_fsync(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A post-rename fsync failure remains fatal when durability cannot be restored."""
    output = tmp_path / "parent-pair"
    receipts = {"DFlash": b"dflash\n", "DSpark": b"dspark\n"}
    fail_adoption_fsync = False
    original_fsync = os.fsync

    def ambiguous_publish(partial: Path, destination: Path) -> None:
        nonlocal fail_adoption_fsync
        partial.rename(destination)
        _write_directory_completion_marker(destination)
        fail_adoption_fsync = True
        raise OSError("injected publisher fsync failure")

    def injected_fsync(descriptor: int) -> None:
        if fail_adoption_fsync:
            raise OSError("injected adoption fsync failure")
        original_fsync(descriptor)

    monkeypatch.setattr(receipt_module, "atomic_publish_directory", ambiguous_publish)
    monkeypatch.setattr(receipt_module.os, "fsync", injected_fsync)

    with pytest.raises(OSError, match="adoption fsync"):
        receipt_module._publish_receipt_pair(output, receipts)


def test_parent_pair_adoption_rejects_a_root_swap_after_open(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The held root descriptor must remain bound to its name in the held parent."""
    output = tmp_path / "parent-pair"
    receipts = {"DFlash": b"dflash\n", "DSpark": b"dspark\n"}
    receipt_module._publish_receipt_pair(output, receipts)
    replacement = tmp_path / "replacement-root"
    replacement.mkdir()
    for child in output.iterdir():
        (replacement / child.name).write_bytes(child.read_bytes())
    displaced = tmp_path / "displaced-root"
    original_listdir = os.listdir
    swapped = False

    def swapped_listdir(path):
        nonlocal swapped
        if not swapped:
            swapped = True
            output.rename(displaced)
            replacement.rename(output)
        return original_listdir(path)

    monkeypatch.setattr(receipt_module.os, "listdir", swapped_listdir)

    with pytest.raises(FileExistsError, match=r"changed|differs|rebound"):
        receipt_module._publish_receipt_pair(output, receipts)


def test_parent_pair_adoption_rejects_a_concurrent_extra_name(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The exact two-file view is re-enumerated before an adopted pair succeeds."""
    output = tmp_path / "parent-pair"
    receipts = {"DFlash": b"dflash\n", "DSpark": b"dspark\n"}
    receipt_module._publish_receipt_pair(output, receipts)
    original_listdir = os.listdir
    calls = 0

    def injected_listdir(path):
        nonlocal calls
        calls += 1
        if calls == 2:
            (output / "foreign").write_bytes(b"extra\n")
        return original_listdir(path)

    monkeypatch.setattr(receipt_module.os, "listdir", injected_listdir)

    with pytest.raises(FileExistsError, match=r"incomplete|changed"):
        receipt_module._publish_receipt_pair(output, receipts)


def test_parent_pair_adoption_reauthenticates_after_final_parent_fsync(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A same-byte root replacement after final fsync cannot be adopted."""
    output = tmp_path / "parent-pair"
    receipts = {"DFlash": b"dflash\n", "DSpark": b"dspark\n"}
    receipt_module._publish_receipt_pair(output, receipts)
    replacement = tmp_path / "replacement-root"
    replacement.mkdir()
    for child in output.iterdir():
        (replacement / child.name).write_bytes(child.read_bytes())
    displaced = tmp_path / "displaced-root"
    original_open = receipt_module.os.open
    original_fsync = receipt_module.os.fsync
    parent_descriptors: set[int] = set()
    swapped = False

    def capture_parent_open(path: object, *args: object, **kwargs: object) -> int:
        descriptor = original_open(path, *args, **kwargs)  # type: ignore[arg-type]
        if path == output.parent and kwargs.get("dir_fd") is None:
            parent_descriptors.add(descriptor)
        return descriptor

    def fsync_then_swap(descriptor: int) -> None:
        nonlocal swapped
        original_fsync(descriptor)
        if descriptor in parent_descriptors and not swapped:
            swapped = True
            output.rename(displaced)
            replacement.rename(output)

    monkeypatch.setattr(receipt_module.os, "open", capture_parent_open)
    monkeypatch.setattr(receipt_module.os, "fsync", fsync_then_swap)

    with pytest.raises(FileExistsError, match=r"changed|rebound"):
        receipt_module._publish_receipt_pair(output, receipts)


def test_parent_pair_adoption_rebinds_root_after_final_enumeration(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A same-byte root replacement after the final listing cannot be adopted."""
    output = tmp_path / "parent-pair"
    receipts = {"DFlash": b"dflash\n", "DSpark": b"dspark\n"}
    receipt_module._publish_receipt_pair(output, receipts)
    replacement = tmp_path / "replacement-root"
    replacement.mkdir()
    for child in output.iterdir():
        (replacement / child.name).write_bytes(child.read_bytes())
    displaced = tmp_path / "displaced-root"
    original_listdir = receipt_module.os.listdir
    calls = 0

    def list_then_swap(path: object) -> list[str]:
        nonlocal calls
        calls += 1
        names = original_listdir(path)  # type: ignore[arg-type]
        if calls == 4:
            output.rename(displaced)
            replacement.rename(output)
        return names

    monkeypatch.setattr(receipt_module.os, "listdir", list_then_swap)

    with pytest.raises(FileExistsError, match=r"changed|rebound"):
        receipt_module._publish_receipt_pair(output, receipts)


def test_parent_cli_rejects_hardlinked_target_model_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The reviewed target asset cannot contain multiply-linked regular files."""
    dflash, dspark = _install_parent_pair(tmp_path, monkeypatch)
    audit = tmp_path / "AUDIT.json"
    audit_file_sha256, ordered_sha256 = _write_historical_audit(audit)
    monkeypatch.setattr(
        receipt_module, "APPROVED_Q30T_HISTORICAL_AUDIT_FILE_SHA256", audit_file_sha256
    )
    monkeypatch.setattr(
        receipt_module,
        "APPROVED_Q30T_HISTORICAL_ORDERED_OCCURRENCES_SHA256",
        ordered_sha256,
    )
    target = tmp_path / "target"
    target.mkdir()
    (target / "config.json").write_bytes(b"{}\n")
    os.link(target / "config.json", target / "duplicate.json")
    target_asset_arguments = _target_asset_arguments(target, monkeypatch)
    runtime = tmp_path / "runtime.tar.zst"
    runtime.write_bytes(b"runtime")

    with pytest.raises(ValueError, match=r"target model tree|hardlink"):
        receipt_module.main(
            [
                "--historical-audit",
                str(audit),
                "--target-model-root",
                str(target),
                "--target-revision",
                "a" * 40,
                *target_asset_arguments,
                "--dflash-parent-root",
                str(dflash),
                "--dspark-parent-root",
                str(dspark),
                "--runtime-archive",
                str(runtime),
                "--source-commit",
                "b" * 40,
                "--output-root",
                str(tmp_path / "receipts"),
            ]
        )


def test_parent_cli_publishes_without_replacing_an_existing_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A conflicting durable root is left byte-for-byte untouched."""
    dflash, dspark = _install_parent_pair(tmp_path, monkeypatch)
    audit = tmp_path / "AUDIT.json"
    audit_file_sha256, ordered_sha256 = _write_historical_audit(audit)
    monkeypatch.setattr(
        receipt_module, "APPROVED_Q30T_HISTORICAL_AUDIT_FILE_SHA256", audit_file_sha256
    )
    monkeypatch.setattr(
        receipt_module,
        "APPROVED_Q30T_HISTORICAL_ORDERED_OCCURRENCES_SHA256",
        ordered_sha256,
    )
    target = tmp_path / "target"
    target.mkdir()
    (target / "config.json").write_bytes(b"{}")
    target_asset_arguments = _target_asset_arguments(target, monkeypatch)
    runtime = tmp_path / "runtime.tar.zst"
    runtime.write_bytes(b"runtime")
    output = tmp_path / "receipts"
    output.mkdir()
    sentinel = output / "sentinel"
    sentinel.write_bytes(b"do not replace")

    with pytest.raises(FileExistsError, match="output root is incomplete"):
        receipt_module.main(
            [
                "--historical-audit",
                str(audit),
                "--target-model-root",
                str(target),
                "--target-revision",
                "a" * 40,
                *target_asset_arguments,
                "--dflash-parent-root",
                str(dflash),
                "--dspark-parent-root",
                str(dspark),
                "--runtime-archive",
                str(runtime),
                "--source-commit",
                "b" * 40,
                "--output-root",
                str(output),
            ]
        )

    assert sentinel.read_bytes() == b"do not replace"
    assert sorted(path.name for path in output.iterdir()) == ["sentinel"]
