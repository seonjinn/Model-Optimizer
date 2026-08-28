# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Hostile tests for deterministic Q30T row-observation runtime evidence."""

from __future__ import annotations

import hashlib
import json
import os
from typing import TYPE_CHECKING

import pytest
from common.specdec.q30t_row_observation_runtime import (
    RowObservationRuntimeEvidence,
    canonical_json_bytes,
    load_row_observation_runtime_evidence,
)

if TYPE_CHECKING:
    from pathlib import Path


def _record() -> dict[str, object]:
    return {
        "approved_tool_sha256s": {
            "ptv23_node_keeper.py": "1" * 64,
            "q30t_runtime_attestation.py": "2" * 64,
        },
        "archive_sha256": "3" * 64,
        "archive_tree_receipt_file_sha256": "4" * 64,
        "base_image_sha256": "5" * 64,
        "derived_image_receipt_file_sha256": "6" * 64,
        "derived_image_sha256": "7" * 64,
        "profile_file_sha256": "8" * 64,
        "pyarrow_relative_path": "lib/python3.12/site-packages/pyarrow/__init__.py",
        "pyarrow_tree_sha256": "9" * 64,
        "pyarrow_version": "19.0.1",
        "python_relative_path": "bin/python",
        "python_version": "3.12.10",
        "qualification_receipt_file_sha256": "a" * 64,
        "qualification_receipt_sha256": "b" * 64,
        "qualification_source_commit": "c" * 40,
        "runtime_tree_sha256": "d" * 64,
        "schema_version": "q30t-row-observation-runtime-evidence-v1",
    }


def _materialize(tmp_path: Path) -> tuple[Path, dict[str, object]]:
    body = _record()
    record = body | {
        "runtime_evidence_sha256": hashlib.sha256(canonical_json_bytes(body)).hexdigest()
    }
    path = tmp_path / "runtime-evidence.json"
    path.write_bytes(canonical_json_bytes(record) + b"\n")
    return path, record


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_runtime_evidence_loads_a_canonical_self_hashed_runtime_record(tmp_path: Path) -> None:
    """Removing a required lineage field or changing its body value must fail replay."""
    path, record = _materialize(tmp_path)

    evidence = load_row_observation_runtime_evidence(path, expected_sha256=_file_sha256(path))

    assert isinstance(evidence, RowObservationRuntimeEvidence)
    assert evidence.to_dict() == record
    assert evidence.canonical_bytes() == canonical_json_bytes(record)
    assert evidence.python_relative_path == "bin/python"
    assert evidence.pyarrow_relative_path == "lib/python3.12/site-packages/pyarrow/__init__.py"


@pytest.mark.parametrize(
    "forbidden_key",
    [
        "job_id",
        "node_name",
        "row_source_commit",
        "scratch_root",
        "container_path",
        "keeper_receipt_sha256",
        "container_evidence_sha256",
    ],
)
def test_runtime_evidence_rejects_job_specific_or_container_fields(
    tmp_path: Path, forbidden_key: str
) -> None:
    """A scientific runtime record must not acquire per-operation identity."""
    path, record = _materialize(tmp_path)
    record[forbidden_key] = "forbidden"
    body = {key: value for key, value in record.items() if key != "runtime_evidence_sha256"}
    record["runtime_evidence_sha256"] = hashlib.sha256(canonical_json_bytes(body)).hexdigest()
    path.write_bytes(canonical_json_bytes(record) + b"\n")

    with pytest.raises(ValueError, match="unexpected schema"):
        load_row_observation_runtime_evidence(path, expected_sha256=_file_sha256(path))


def test_runtime_evidence_rejects_noncanonical_duplicate_and_wrong_file_identity(
    tmp_path: Path,
) -> None:
    """Parser ambiguity and caller SHA drift must fail before evidence is consumed."""
    path, record = _materialize(tmp_path)
    canonical = canonical_json_bytes(record)
    forged = canonical.replace(
        b'{"approved_tool_sha256s":',
        b'{"approved_tool_sha256s":{},"approved_tool_sha256s":',
        1,
    )
    path.write_bytes(forged + b"\n")

    with pytest.raises(ValueError, match="duplicate JSON key"):
        load_row_observation_runtime_evidence(path, expected_sha256=_file_sha256(path))

    path, record = _materialize(tmp_path)
    path.write_bytes(json.dumps(record, separators=(",", ":")).encode() + b"\n")
    with pytest.raises(ValueError, match="not canonical"):
        load_row_observation_runtime_evidence(path, expected_sha256=_file_sha256(path))
    with pytest.raises(ValueError, match="whole-file SHA-256 mismatch"):
        load_row_observation_runtime_evidence(path, expected_sha256="0" * 64)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("python_relative_path", "/opt/q30t-runtime/bin/python"),
        ("python_relative_path", "bin/../python"),
        ("python_relative_path", "bin/py\nthon"),
        (
            "pyarrow_relative_path",
            "/opt/q30t-runtime/lib/python3.12/site-packages/pyarrow/__init__.py",
        ),
        ("pyarrow_relative_path", "lib/python3.12/site-packages/pyarrow/\x00init__.py"),
    ],
)
def test_runtime_evidence_rejects_absolute_or_controlled_runtime_paths(
    tmp_path: Path, field: str, value: str
) -> None:
    """Runtime paths must be portable relative paths below the fixed image root."""
    path, record = _materialize(tmp_path)
    record[field] = value
    body = {key: value for key, value in record.items() if key != "runtime_evidence_sha256"}
    record["runtime_evidence_sha256"] = hashlib.sha256(canonical_json_bytes(body)).hexdigest()
    path.write_bytes(canonical_json_bytes(record) + b"\n")

    with pytest.raises(ValueError, match="relative path"):
        load_row_observation_runtime_evidence(path, expected_sha256=_file_sha256(path))


def test_runtime_evidence_rejects_linked_and_oversized_evidence(tmp_path: Path) -> None:
    """A substituted or oversized descriptor cannot become runtime evidence."""
    path, _ = _materialize(tmp_path)
    os.link(path, tmp_path / "runtime-evidence-link.json")

    with pytest.raises(ValueError, match="single-link regular file"):
        load_row_observation_runtime_evidence(path, expected_sha256=_file_sha256(path))

    path.unlink()
    path.write_bytes(b"x" * (1024 * 1024 + 1))
    with pytest.raises(ValueError, match="size limit"):
        load_row_observation_runtime_evidence(path, expected_sha256=_file_sha256(path))


def test_runtime_evidence_rejects_absolute_parent_rebinding_after_open(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A held parent descriptor must remain bound to the caller's absolute path."""
    import common.specdec.q30t_row_observation_runtime as runtime_module

    parent = tmp_path / "runtime-evidence-parent"
    parent.mkdir()
    path, _ = _materialize(parent)
    expected_sha256 = _file_sha256(path)
    displaced = tmp_path / "displaced-runtime-evidence-parent"
    replacement = tmp_path / "replacement-runtime-evidence-parent"
    replacement.mkdir()
    original_read = runtime_module.os.read
    rebound = False

    def rebind_after_read(descriptor: int, count: int) -> bytes:
        nonlocal rebound
        block = original_read(descriptor, count)
        if block and not rebound:
            rebound = True
            parent.rename(displaced)
            replacement.rename(parent)
        return block

    monkeypatch.setattr(runtime_module.os, "read", rebind_after_read)

    with pytest.raises(ValueError, match=r"absolute path.*changed"):
        load_row_observation_runtime_evidence(path, expected_sha256=expected_sha256)
