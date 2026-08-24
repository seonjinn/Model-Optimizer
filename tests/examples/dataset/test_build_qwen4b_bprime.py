# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Production Task5 B-prime builder contracts."""

from __future__ import annotations

import json
import sys
from dataclasses import replace
from pathlib import Path

import pytest

MODULE_DIR = Path(__file__).resolve().parents[3] / "examples/dataset"
sys.path.insert(0, str(MODULE_DIR))
try:
    from audit_ptv2_baseline import (
        EXPECTED_BASELINE,
        BaselineAudit,
        BaselineExpectation,
        SelectionBoundary,
        write_audit_receipt,
    )
    from build_qwen4b_bprime import (
        authenticate_baseline_audit,
        baseline_exclusion_from_audit,
        publish_candidate_diagnostic_receipt,
    )
    from build_specdec_inventory import make_exclusion_receipt
    from specdec_corpus_contracts import (
        SourceFile,
        canonical_json,
        sha256_bytes,
        sha256_canonical_json,
    )
finally:
    sys.path.pop(0)


def _genuine_audit(path: Path) -> tuple[BaselineAudit, BaselineExpectation]:
    prompt_ids = ("1" * 64, "2" * 64, "1" * 64)
    exclusion_ids = tuple(sorted(set(prompt_ids)))
    audit = BaselineAudit(
        source_revision=EXPECTED_BASELINE.source_revision,
        source_manifest_sha256="3" * 64,
        row_count=3,
        split_rows={"chat": 3},
        files=(SourceFile("chat.parquet", 7, "4" * 64),),
        occurrence_count=3,
        unique_prompt_count=2,
        occurrence_prompt_ids=prompt_ids,
        occurrence_prompt_ids_sha256=sha256_canonical_json(prompt_ids),
        exclusion_prompt_ids=exclusion_ids,
        exclusion_prompt_ids_sha256=sha256_canonical_json(exclusion_ids),
        duplicate_uuid_multiplicity={"1" * 64: 2},
        physical_row_count=3,
        selection_policy="hf-streaming-sorted-parquet-take",
        selection_boundary=SelectionBoundary("chat.parquet", 3, 3, 0),
    )
    write_audit_receipt(audit, path)
    expected = replace(
        EXPECTED_BASELINE,
        shard_count=1,
        split_rows={"chat": 3},
        unique_prompt_count=2,
        physical_row_count=3,
        excluded_tail_rows=0,
        excluded_tail_split="chat.parquet",
    )
    return audit, expected


def test_baseline_exclusion_consumes_the_genuine_audit_producer(tmp_path: Path) -> None:
    path = tmp_path / "audit.json"
    audit, expected = _genuine_audit(path)

    assert baseline_exclusion_from_audit(path, expected=expected) == make_exclusion_receipt(
        "baseline", audit.exclusion_prompt_ids
    )
    authenticated = authenticate_baseline_audit(path, expected=expected)
    assert authenticated.exclusion == make_exclusion_receipt(
        "baseline", audit.exclusion_prompt_ids
    )
    assert authenticated.payload["occurrence_prompt_ids"] == list(
        audit.occurrence_prompt_ids
    )
    assert authenticated.receipt_sha256 == json.loads(path.read_bytes())["receipt_sha256"]


def test_candidate_diagnostic_receipt_is_exclusive_canonical_and_self_authenticated(
    tmp_path: Path,
) -> None:
    destination = tmp_path / "diagnostics/candidate.json"
    body = {"schema_version": 1, "source_commit": "1" * 40, "shards": []}
    payload = body | {"receipt_sha256": sha256_bytes(canonical_json(body))}

    publish_candidate_diagnostic_receipt(payload, destination)

    assert destination.read_bytes() == canonical_json(payload) + b"\n"
    with pytest.raises(FileExistsError):
        publish_candidate_diagnostic_receipt(payload, destination)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("exclusion_prompt_ids_sha256", "0" * 64),
        ("unique_prompt_count", 3),
        ("source_revision", "0" * 40),
    ],
)
def test_baseline_exclusion_rejects_forged_genuine_fields(
    tmp_path: Path, field: str, value: object
) -> None:
    path = tmp_path / "audit.json"
    _audit, expected = _genuine_audit(path)
    payload = json.loads(path.read_bytes())
    payload[field] = value
    body = {key: item for key, item in payload.items() if key != "receipt_sha256"}
    payload["receipt_sha256"] = sha256_bytes(canonical_json(body))
    path.write_bytes(canonical_json(payload) + b"\n")

    with pytest.raises(ValueError, match="baseline audit"):
        baseline_exclusion_from_audit(path, expected=expected)


def test_task5_runner_uses_an_authenticated_container_runtime() -> None:
    root = Path(__file__).resolve().parents[3]
    runner = (
        root / "tools/launcher/common/specdec/run_qwen4b_task5_bprime.sbatch"
    ).read_text()
    submitter = (
        root / "tools/launcher/common/specdec/submit_qwen4b_task5_bprime.sh"
    ).read_text()

    assert "IMAGE_PATH IMAGE_SHA256" in runner
    assert 'sha256sum "$IMAGE_PATH"' in runner
    assert '--no-container-mount-home --container-image="$IMAGE_PATH"' in runner
    assert "--container-mounts=" in runner
    assert "--image PATH --image-sha256 SHA256" in submitter
    assert "IMAGE_PATH=$IMAGE_PATH,IMAGE_SHA256=$IMAGE_SHA256" in submitter
    assert '"$OUTPUT_DIR/EXECUTION_RECEIPT.json"' in runner
    assert '"$OUTPUT_DIR.EXECUTION_RECEIPT.json"' not in runner
    for name in (
        "ARROW_NUM_THREADS",
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        assert f"{name}=1" in runner
