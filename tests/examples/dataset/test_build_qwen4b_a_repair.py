# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import sys
from hashlib import sha256
from pathlib import Path
from types import SimpleNamespace

import pytest

MODULE_DIR = Path(__file__).resolve().parents[3] / "examples/dataset"
sys.path.insert(0, str(MODULE_DIR))
try:
    import build_qwen4b_a_repair as a_repair_module
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
    assert runner.index("export ARROW_NUM_THREADS=1") < runner.index("python3 -P")
    assert "--workers 96" in runner
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


def test_task9_a_cli_publishes_the_finalized_p96_execution_receipt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The production CLI must not silently fall back to the legacy serial receipt."""
    source = tmp_path / "SOURCE_INVENTORY.json"
    baseline = tmp_path / "baseline.json"
    held_out = tmp_path / "held-out.json"
    policy = tmp_path / "policy.yaml"
    task5 = tmp_path / "task5.json"
    for path in (source, baseline, held_out, policy, task5):
        path.write_text("{}", encoding="utf-8")
    task5_sha256 = sha256(task5.read_bytes()).hexdigest()
    execution = {"schema_version": 1, "effective_workers": 96}
    view = SimpleNamespace(
        strategy="A-repair",
        selection_sha256="a" * 64,
        execution_receipt=execution,
    )
    authenticated = SimpleNamespace(exclusion=SimpleNamespace(receipt_sha256="b" * 64))
    held_out_receipt = SimpleNamespace(prompt_ids=(), receipt_sha256="c" * 64)
    seen: dict[str, object] = {}
    monkeypatch.setattr(a_repair_module, "load_ptv2_study_policy", lambda _: object())
    monkeypatch.setattr(
        a_repair_module,
        "load_source_inventory",
        lambda _: SimpleNamespace(manifest_sha256="d" * 64),
    )
    monkeypatch.setattr(a_repair_module, "authenticate_baseline_audit", lambda _: authenticated)
    monkeypatch.setattr(a_repair_module, "held_out_exclusion_from_json", lambda _: held_out_receipt)
    monkeypatch.setattr(a_repair_module, "load_prompt_view", lambda *args, **kwargs: object())
    monkeypatch.setattr(
        a_repair_module, "_authenticate_task5_bprime", lambda *args, **kwargs: "e" * 64
    )

    def select(*args, **kwargs):
        seen["workers"] = kwargs["workers"]
        seen["source_commit"] = kwargs["source_commit"]
        seen["baseline_builder"] = kwargs["baseline_builder"]
        return view

    receipt = tmp_path / "selection" / "SELECTION_RECEIPT.json"
    receipt.parent.mkdir()
    receipt.write_text("{}\n", encoding="utf-8")
    monkeypatch.setattr(a_repair_module, "select_authenticated_a_repair_view", select)

    def publish(*args, **kwargs):
        seen["published_execution"] = kwargs["execution_receipt"]
        return receipt

    monkeypatch.setattr(a_repair_module, "write_ptv2_selection_receipt", publish)
    monkeypatch.setenv("SOURCE_COMMIT", "f" * 40)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "build_qwen4b_a_repair.py",
            "--source-inventory",
            str(source),
            "--baseline-audit",
            str(baseline),
            "--held-out-uuids",
            str(held_out),
            "--policy",
            str(policy),
            "--task5-manifest",
            str(task5),
            "--task5-manifest-sha256",
            task5_sha256,
            "--work-root",
            str(tmp_path / "work"),
            "--selection-receipt-root",
            str(tmp_path / "selection"),
            "--workers",
            "96",
        ],
    )

    assert a_repair_module.main() == 0
    assert seen["workers"] == 96
    assert seen["source_commit"] == "f" * 40
    assert callable(seen["baseline_builder"])
    assert seen["published_execution"] is execution
