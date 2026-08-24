# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import sys
from pathlib import Path

import pytest

MODULE_DIR = Path(__file__).resolve().parents[3] / "examples/dataset"
sys.path.insert(0, str(MODULE_DIR))
try:
    from audit_ptv2_baseline import BaselineExpectation
    from build_qwen4b_a_repair import reconstruct_baseline_sequence
    from build_qwen4b_bprime import authenticate_baseline_audit
    from qwen3_4b_ptv2_study import PTV2StudySourceRow
finally:
    sys.path.pop(0)

from test_build_qwen4b_bprime import _genuine_audit


def _row(prompt_uuid: str, source_row: int) -> PTV2StudySourceRow:
    return PTV2StudySourceRow(
        prompt_uuid=prompt_uuid,
        source_identity_sha256="a" * 64,
        source_row=source_row,
        cell="chat",
        canonical_conversation="{}",
        assistant_response="{}",
        language="",
    )


def test_reconstruct_baseline_sequence_binds_legacy_set_to_source_order(tmp_path: Path) -> None:
    path = tmp_path / "audit.json"
    audit, expected = _genuine_audit(path)
    authenticated = authenticate_baseline_audit(path, expected=expected)

    rebuilt = reconstruct_baseline_sequence(
        (_row(value, index) for index, value in enumerate(audit.occurrence_prompt_ids)),
        authenticated,
        expected=expected,
    )

    assert rebuilt.occurrence_prompt_ids == audit.occurrence_prompt_ids
    assert rebuilt.exclusion_prompt_ids == audit.exclusion_prompt_ids
    assert rebuilt.duplicate_uuid_multiplicity == audit.duplicate_uuid_multiplicity


def test_reconstruct_baseline_sequence_rejects_source_multiplicity_drift(tmp_path: Path) -> None:
    path = tmp_path / "audit.json"
    _audit, expected = _genuine_audit(path)
    authenticated = authenticate_baseline_audit(path, expected=expected)
    drifted = (_row(value, index) for index, value in enumerate(("1" * 64,) * 3))

    with pytest.raises(ValueError, match="exclusion set"):
        reconstruct_baseline_sequence(drifted, authenticated, expected=expected)


def test_reconstruct_baseline_sequence_requires_exact_split_counts(tmp_path: Path) -> None:
    path = tmp_path / "audit.json"
    audit, expected = _genuine_audit(path)
    authenticated = authenticate_baseline_audit(path, expected=expected)
    wrong = BaselineExpectation(
        expected.source_revision,
        expected.shard_count,
        {"code": 3},
        expected.unique_prompt_count,
        expected.physical_row_count,
        expected.excluded_tail_rows,
        expected.excluded_tail_split,
    )

    with pytest.raises(ValueError, match="split counts"):
        reconstruct_baseline_sequence(
            (_row(value, index) for index, value in enumerate(audit.occurrence_prompt_ids)),
            authenticated,
            expected=wrong,
        )


def test_task9_a_runner_is_a_96_cpu_authenticated_container_job() -> None:
    root = Path(__file__).resolve().parents[3]
    runner = (root / "tools/launcher/common/specdec/run_qwen4b_task9_a.sbatch").read_text()
    submitter = (root / "tools/launcher/common/specdec/submit_qwen4b_task9_a.sh").read_text()

    assert '"${SLURM_CPUS_PER_TASK:-}" == 96' in runner
    assert "IMAGE_PATH IMAGE_SHA256" in runner
    assert '--no-container-mount-home --container-image="$IMAGE_PATH"' in runner
    assert "--task5-manifest PATH --task5-manifest-sha256 SHA256" in submitter
    assert 'sbatch --test-only "${args[@]}"' in submitter


def test_task9_a_controller_persists_the_deferred_job_identity() -> None:
    root = Path(__file__).resolve().parents[3]
    controller = (
        root / "tools/launcher/common/specdec/run_qwen4b_task9_a_controller.sbatch"
    ).read_text()

    assert "TASK5_PARENT_JOB_ID" in controller
    assert 'sha256sum "$TASK5_MANIFEST"' in controller
    assert "submit_qwen4b_task9_a.sh" in controller
    assert "TASK9_A_CONTROLLER_RECEIPT" in controller
    assert 'open(receipt, "xb")' in controller
