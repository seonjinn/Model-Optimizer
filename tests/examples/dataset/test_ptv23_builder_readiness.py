# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Fail-closed readiness tests for the PTV2/PTV3 corpus builder."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
MODULE_PATH = ROOT / "examples/dataset/ptv23_builder_readiness.py"
sys.path.insert(0, str(MODULE_PATH.parent))
try:
    from specdec_corpus_contracts import canonical_json
finally:
    sys.path.pop(0)


def _load_module():
    spec = importlib.util.spec_from_file_location("ptv23_builder_readiness", MODULE_PATH)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(MODULE_PATH.parent))
    try:
        spec.loader.exec_module(module)
    finally:
        sys.path.pop(0)
    return module


REVISION = "5c89e01dd720ae0f4058445ed49c5fb68a03c76e"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _fixture(tmp_path: Path) -> tuple[Path, str, Path, Path, Path]:
    ptv2 = tmp_path / "ptv2"
    ptv2.mkdir()
    source_files = []
    for name in (
        "chat-00000-of-00001.parquet",
        "code-00000-of-00001.parquet",
        "math-00000-of-00001.parquet",
        "stem-00000-of-00001.parquet",
        "multilingual_ja-00000-of-00001.parquet",
    ):
        path = ptv2 / name
        path.write_bytes(name.encode())
        if name.startswith(("chat-", "code-", "math-")):
            source_files.append(
                {"path": str(path.resolve()), "bytes": path.stat().st_size, "sha256": _sha256(path)}
            )

    audit_payload = {
        "source_revision": REVISION,
        "row_count": 30,
        "split_rows": {"chat": 10, "code": 10, "math": 10},
        "files": source_files,
        "prompt_uuids": ["a", "b"],
        "duplicate_uuid_multiplicity": {"a": 29},
        "physical_row_count": 30,
        "selection_policy": "hf-streaming-sorted-parquet-take",
        "selection_boundary": {
            "file": "math-00000-of-00001.parquet",
            "rows_selected": 10,
            "rows_available": 10,
            "excluded_tail_rows": 0,
        },
    }
    audit_sha = hashlib.sha256(canonical_json(audit_payload)).hexdigest()
    audit_payload["receipt_sha256"] = audit_sha
    audit = tmp_path / "audit.json"
    audit.write_text(json.dumps(audit_payload), encoding="utf-8")

    prompt_root = tmp_path / "prompts"
    (prompt_root / "target-synth").mkdir(parents=True)
    (prompt_root / "trace-replay").mkdir()
    source_manifest = tmp_path / "ptv3-source.json"
    source_manifest.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "files": [
                    {"category": "swe", "source_id": "agentless", "tool_lane": "recorded-trace"},
                    {
                        "category": "agentic_tool",
                        "source_id": "generic-tools",
                        "tool_lane": "recorded-trace",
                    },
                ],
            }
        ),
        encoding="utf-8",
    )
    source_sha = _sha256(source_manifest)
    target = {
        "schema_version": 1,
        "source_manifest_sha256": source_sha,
        "selected_assistant_tokens": 0,
        "category_counts": {
            "swe": 1,
            "math": 1,
            "code": 1,
            "science": 1,
            "chat": 1,
            "multilingual": 1,
        },
    }
    trace = {
        "schema_version": 1,
        "source_manifest_sha256": source_sha,
        "selected_assistant_tokens": 100,
        "category_counts": {"agentic_tool": 1},
    }
    target_path = prompt_root / "target-synth" / "MANIFEST.json"
    trace_path = prompt_root / "trace-replay" / "MANIFEST.json"
    target_path.write_text(json.dumps(target), encoding="utf-8")
    trace_path.write_text(json.dumps(trace), encoding="utf-8")
    (prompt_root / "SELECTION_RECEIPT.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "source_manifest_sha256": source_sha,
                "lane_manifest_sha256": {
                    "target-synth": _sha256(target_path),
                    "trace-replay": _sha256(trace_path),
                },
            }
        ),
        encoding="utf-8",
    )
    return audit, audit_sha, ptv2, prompt_root, source_manifest


def test_readiness_fails_closed_for_exhausted_ptv2_and_missing_ptv3_lanes(
    tmp_path: Path,
) -> None:
    module = _load_module()
    audit, audit_sha, ptv2, prompt_root, source_manifest = _fixture(tmp_path)

    receipt = module.assess_builder_readiness(
        audit_receipt=audit,
        expected_audit_receipt_sha256=audit_sha,
        ptv2_root=ptv2,
        ptv2_revision=REVISION,
        ptv3_prompt_root=prompt_root,
        ptv3_source_manifest=source_manifest,
    )

    assert receipt["ready"] == {"B": False, "C": False, "D": False}
    assert receipt["ptv2"]["unseen_candidate_upper_bound"] == {
        "chat": 0,
        "code": 0,
        "math": 0,
    }
    assert receipt["ptv3"]["target_synth_assistant_tokens"] == 0
    assert receipt["ptv3"]["interactive_swe_replay_source_count"] == 0
    assert receipt["ptv3"]["generic_tool_replay_assistant_tokens"] == 100
    assert any("math" in reason for reason in receipt["blockers"]["B"])
    assert any("target-synth" in reason for reason in receipt["blockers"]["C"])
    assert any("interactive-SWE" in reason for reason in receipt["blockers"]["D"])


def test_readiness_rejects_wrong_audit_identity(tmp_path: Path) -> None:
    module = _load_module()
    audit, _, ptv2, prompt_root, source_manifest = _fixture(tmp_path)

    with pytest.raises(ValueError, match="audit receipt SHA-256 mismatch"):
        module.assess_builder_readiness(
            audit_receipt=audit,
            expected_audit_receipt_sha256="0" * 64,
            ptv2_root=ptv2,
            ptv2_revision=REVISION,
            ptv3_prompt_root=prompt_root,
            ptv3_source_manifest=source_manifest,
        )


def test_readiness_receipt_is_atomic_and_immutable(tmp_path: Path) -> None:
    module = _load_module()
    output = tmp_path / "receipt.json"
    payload = {"schema_version": 1, "ready": {"B": False, "C": False, "D": False}}

    module.write_readiness_receipt(payload, output)

    written = json.loads(output.read_text())
    receipt_sha = written.pop("receipt_sha256")
    assert receipt_sha == hashlib.sha256(canonical_json(written)).hexdigest()
    with pytest.raises(FileExistsError, match="already exists"):
        module.write_readiness_receipt(payload, output)


def test_readiness_runner_binds_exact_inputs_and_uses_slurm() -> None:
    runner = (
        Path(__file__).resolve().parents[3]
        / "tools/launcher/common/specdec/run_ptv23_builder_readiness.sbatch"
    ).read_text()

    assert "#SBATCH --nodes=1" in runner
    assert "AUDIT_RECEIPT_SHA256" in runner
    assert "PTV2_REVISION" in runner
    assert "PTV3_SOURCE_MANIFEST" in runner
    assert 'git -C "$SOURCE_PATH" rev-parse HEAD' in runner
    assert "--output-receipt" in runner
