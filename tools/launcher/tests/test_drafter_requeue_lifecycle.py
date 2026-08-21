# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Behavioral tests for the opt-in drafter self-requeue lifecycle."""

from __future__ import annotations

import hashlib
import json
import os
import signal
import subprocess
import time
from pathlib import Path

_LAUNCHER_DIR = Path(__file__).resolve().parents[1]
_LIFECYCLE = _LAUNCHER_DIR / "common/specdec/drafter_requeue_lifecycle.sh"


def _write_checkpoint(output_root: Path, step: int) -> Path:
    checkpoint = output_root / f"checkpoint-{step}"
    checkpoint.mkdir(parents=True)
    (checkpoint / "model.safetensors").write_bytes(b"model")
    (checkpoint / "optimizer.pt").write_bytes(b"optimizer")
    (checkpoint / "scheduler.pt").write_bytes(b"scheduler")
    for rank in range(8):
        (checkpoint / f"rng_state_{rank}.pth").write_bytes(b"rng")
    (checkpoint / "trainer_state.json").write_text(json.dumps({"global_step": step}))
    return checkpoint


def _write_fake_scontrol(tmp_path: Path) -> tuple[Path, Path]:
    calls = tmp_path / "scontrol.calls"
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    scontrol = fake_bin / "scontrol"
    scontrol.write_text(
        "#!/usr/bin/env bash\n"
        "set -euo pipefail\n"
        'printf "%s\\n" "$*" >>"$SCONTROL_CALLS"\n'
        'if [[ "${SCONTROL_KILLS_CALLER:-0}" == 1 ]]; then kill -KILL "$PPID"; fi\n'
    )
    scontrol.chmod(0o755)
    return fake_bin, calls


def _start_lifecycle(
    tmp_path: Path,
    output_root: Path,
    *,
    target_step: int,
    restart_count: int = 0,
    max_requeues: int = 3,
    child_command: str | None = None,
    terminate_grace_seconds: int = 1,
    scontrol_kills_caller: bool = False,
    milestone_steps: str = "4166,25391",
    milestone_force_copy: bool = False,
) -> tuple[subprocess.Popen[str], Path, Path]:
    fake_bin, calls = _write_fake_scontrol(tmp_path)
    ready = tmp_path / "ready"
    child_ready = tmp_path / "child-ready"
    wait_for_child = child_command is None
    if child_command is None:
        child_command = 'touch "$CHILD_READY"; exec tail -f /dev/null'
    driver = tmp_path / "driver.sh"
    driver.write_text(
        "#!/usr/bin/env bash\n"
        "set -euo pipefail\n"
        f'source "{_LIFECYCLE}"\n'
        "drafter_requeue_init\n"
        f'touch "{ready}"\n'
        f"drafter_run_requeueable_step bash -c {json.dumps(child_command)}\n"
        "drafter_mark_training_complete\n"
    )
    driver.chmod(0o755)
    env = {
        **os.environ,
        "PATH": f"{fake_bin}:{os.environ['PATH']}",
        "OUTPUT_ROOT": str(output_root),
        "EXPORT_PATH": str(output_root / f"exported-checkpoint-{target_step}"),
        "MAX_STEPS": str(target_step),
        "MAX_REQUEUES": str(max_requeues),
        "SELF_REQUEUE": "1",
        "SLURM_JOB_ID": "4242",
        "SLURM_RESTART_COUNT": str(restart_count),
        "SCONTROL_CALLS": str(calls),
        "SCONTROL_KILLS_CALLER": "1" if scontrol_kills_caller else "0",
        "CHILD_READY": str(child_ready),
        "REQUEUE_TERMINATE_GRACE_SECONDS": str(terminate_grace_seconds),
        "MILESTONE_STEPS": milestone_steps,
        "MILESTONE_FORCE_COPY": "1" if milestone_force_copy else "0",
    }
    process = subprocess.Popen(
        [str(driver)],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    deadline = time.monotonic() + 5
    while not ready.exists() and process.poll() is None and time.monotonic() < deadline:
        time.sleep(0.01)
    assert ready.exists(), process.communicate(timeout=1)
    while (
        wait_for_child
        and not child_ready.exists()
        and process.poll() is None
        and time.monotonic() < deadline
    ):
        time.sleep(0.01)
    if wait_for_child:
        assert child_ready.exists(), process.communicate(timeout=1)
    return process, calls, output_root / "control/requeue/job-4242/attempt-0.json"


def test_usr1_requeues_same_job_only_after_complete_checkpoint(tmp_path: Path) -> None:
    output_root = tmp_path / "output"
    checkpoint = _write_checkpoint(output_root, 20)
    process, calls, receipt = _start_lifecycle(tmp_path, output_root, target_step=70)

    process.send_signal(signal.SIGUSR1)
    stdout, stderr = process.communicate(timeout=10)

    assert process.returncode == 0, (stdout, stderr)
    assert calls.read_text().splitlines() == ["requeue 4242"]
    assert json.loads(receipt.read_text()) == {
        "checkpoint": str(checkpoint),
        "global_step": 20,
        "job_id": "4242",
        "restart_count": 0,
        "status": "requeued",
        "target_step": 70,
    }
    assert not (output_root / "control/training-complete-s70.json").exists()


def test_requeue_receipt_survives_scheduler_terminating_batch_shell(tmp_path: Path) -> None:
    output_root = tmp_path / "output"
    checkpoint = _write_checkpoint(output_root, 20)
    process, calls, receipt = _start_lifecycle(
        tmp_path,
        output_root,
        target_step=70,
        scontrol_kills_caller=True,
    )
    assert process.stdout is not None
    assert process.stderr is not None

    process.send_signal(signal.SIGUSR1)
    process.wait(timeout=10)

    assert process.returncode != 0
    assert calls.read_text().splitlines() == ["requeue 4242"]
    record = json.loads(receipt.read_text())
    assert record["checkpoint"] == str(checkpoint)
    assert record["status"] == "requeue-requested"


def test_usr1_reaps_a_terminated_step_without_waiting_full_grace(tmp_path: Path) -> None:
    output_root = tmp_path / "output"
    _write_checkpoint(output_root, 20)
    process, _, _ = _start_lifecycle(
        tmp_path,
        output_root,
        target_step=70,
        terminate_grace_seconds=5,
    )

    started = time.monotonic()
    process.send_signal(signal.SIGUSR1)
    stdout, stderr = process.communicate(timeout=3)

    assert process.returncode == 0, (stdout, stderr)
    assert time.monotonic() - started < 3


def test_usr1_fails_closed_without_a_complete_checkpoint(tmp_path: Path) -> None:
    output_root = tmp_path / "output"
    process, calls, receipt = _start_lifecycle(tmp_path, output_root, target_step=70)

    process.send_signal(signal.SIGUSR1)
    stdout, stderr = process.communicate(timeout=10)

    assert process.returncode != 0, (stdout, stderr)
    assert not calls.exists()
    record = json.loads(receipt.read_text())
    assert record["status"] == "failed-no-complete-checkpoint"


def test_usr1_quarantines_incomplete_newer_checkpoint_before_requeue(tmp_path: Path) -> None:
    output_root = tmp_path / "output"
    expected = _write_checkpoint(output_root, 20)
    incomplete = output_root / "checkpoint-40"
    incomplete.mkdir()
    (incomplete / "trainer_state.json").write_text(json.dumps({"global_step": 40}))
    process, calls, receipt = _start_lifecycle(tmp_path, output_root, target_step=70)

    process.send_signal(signal.SIGUSR1)
    stdout, stderr = process.communicate(timeout=10)

    assert process.returncode == 0, (stdout, stderr)
    assert calls.read_text().splitlines() == ["requeue 4242"]
    assert json.loads(receipt.read_text())["checkpoint"] == str(expected)
    assert not incomplete.exists()
    quarantined = output_root / "control/requeue/job-4242/quarantine-0/checkpoint-40"
    assert (quarantined / "trainer_state.json").is_file()


def test_usr1_fails_closed_at_restart_limit(tmp_path: Path) -> None:
    output_root = tmp_path / "output"
    _write_checkpoint(output_root, 40)
    process, calls, _ = _start_lifecycle(
        tmp_path,
        output_root,
        target_step=70,
        restart_count=3,
        max_requeues=3,
    )

    process.send_signal(signal.SIGUSR1)
    stdout, stderr = process.communicate(timeout=10)

    assert process.returncode != 0, (stdout, stderr)
    assert not calls.exists()
    receipt = output_root / "control/requeue/job-4242/attempt-3.json"
    assert json.loads(receipt.read_text())["status"] == "failed-restart-limit"


def test_restart_count_49_is_the_last_permitted_requeue(tmp_path: Path) -> None:
    output_root = tmp_path / "output"
    _write_checkpoint(output_root, 40)
    process, calls, _ = _start_lifecycle(
        tmp_path,
        output_root,
        target_step=70,
        restart_count=49,
        max_requeues=50,
    )

    process.send_signal(signal.SIGUSR1)
    stdout, stderr = process.communicate(timeout=10)

    assert process.returncode == 0, (stdout, stderr)
    assert calls.read_text().splitlines() == ["requeue 4242"]


def test_restart_count_50_refuses_another_requeue(tmp_path: Path) -> None:
    output_root = tmp_path / "output"
    _write_checkpoint(output_root, 40)
    process, calls, _ = _start_lifecycle(
        tmp_path,
        output_root,
        target_step=70,
        restart_count=50,
        max_requeues=50,
    )

    process.send_signal(signal.SIGUSR1)
    stdout, stderr = process.communicate(timeout=10)

    assert process.returncode != 0, (stdout, stderr)
    assert not calls.exists()
    receipt = output_root / "control/requeue/job-4242/attempt-50.json"
    assert json.loads(receipt.read_text())["status"] == "failed-restart-limit"


def test_normal_target_completion_marks_done_without_requeue(tmp_path: Path) -> None:
    output_root = tmp_path / "output"
    _write_checkpoint(output_root, 70)
    (output_root / "model.safetensors").write_bytes(b"final-model")
    (output_root / "trainer_state.json").write_text(json.dumps({"global_step": 70}))
    export_path = output_root / "exported-checkpoint-70"
    export_path.mkdir()
    (export_path / "model.safetensors").write_bytes(b"export")
    process, calls, _ = _start_lifecycle(
        tmp_path,
        output_root,
        target_step=70,
        child_command="exit 0",
    )

    stdout, stderr = process.communicate(timeout=10)

    assert process.returncode == 0, (stdout, stderr)
    assert not calls.exists()
    marker = output_root / "control/training-complete-s70.json"
    assert json.loads(marker.read_text()) == {
        "checkpoint": str(output_root),
        "global_step": 70,
        "job_id": "4242",
        "restart_count": 0,
        "status": "completed",
        "target_step": 70,
    }


def test_normal_non_divisible_target_accepts_valid_final_root(tmp_path: Path) -> None:
    output_root = tmp_path / "output"
    _write_checkpoint(output_root, 60)
    (output_root / "model.safetensors").write_bytes(b"final-model")
    (output_root / "trainer_state.json").write_text(json.dumps({"global_step": 70}))
    export_path = output_root / "exported-checkpoint-70"
    export_path.mkdir()
    (export_path / "model.safetensors").write_bytes(b"export")
    process, calls, _ = _start_lifecycle(
        tmp_path,
        output_root,
        target_step=70,
        child_command="exit 0",
    )

    stdout, stderr = process.communicate(timeout=10)

    assert process.returncode == 0, (stdout, stderr)
    assert not calls.exists()
    marker = output_root / "control/training-complete-s70.json"
    assert json.loads(marker.read_text())["checkpoint"] == str(output_root)
    assert json.loads(marker.read_text())["global_step"] == 70


def test_milestone_preserves_exact_export_and_nearest_resumable_checkpoint(
    tmp_path: Path,
) -> None:
    output_root = tmp_path / "output"
    checkpoint = _write_checkpoint(output_root, 4150)
    future_checkpoint = _write_checkpoint(output_root, 5000)
    (output_root / "model.safetensors").write_bytes(b"exact-final-model")
    (output_root / "trainer_state.json").write_text(json.dumps({"global_step": 4166}))
    export_path = output_root / "exported-checkpoint-4166"
    export_path.mkdir()
    (export_path / "model.safetensors").write_bytes(b"exact-export")
    process, calls, _ = _start_lifecycle(
        tmp_path,
        output_root,
        target_step=4166,
        child_command="exit 0",
    )

    stdout, stderr = process.communicate(timeout=10)

    assert process.returncode == 0, (stdout, stderr)
    assert not calls.exists()
    milestone = output_root / "milestones/step-004166"
    exact_model = milestone / "exact-model"
    resume = milestone / "resume-checkpoint-004150"
    manifest = json.loads((milestone / "manifest.json").read_text())
    assert exact_model.is_symlink()
    assert exact_model.resolve() == export_path.resolve()
    assert {key: manifest[key] for key in (
        "exact_model_path",
        "exact_model_step",
        "resume_checkpoint_path",
        "resume_checkpoint_step",
    )} == {
        "exact_model_path": "../../exported-checkpoint-4166",
        "exact_model_step": 4166,
        "resume_checkpoint_path": "resume-checkpoint-004150",
        "resume_checkpoint_step": 4150,
    }
    assert manifest["exact_model_sha256"] == {
        "model.safetensors": hashlib.sha256(b"exact-export").hexdigest()
    }
    assert manifest["resume_checkpoint_sha256"]["optimizer.pt"] == hashlib.sha256(
        b"optimizer"
    ).hexdigest()
    assert set(manifest["resume_checkpoint_storage"].values()) == {"hardlink"}
    assert (resume / "optimizer.pt").stat().st_ino == (checkpoint / "optimizer.pt").stat().st_ino
    assert (resume / "optimizer.pt").stat().st_nlink >= 2

    for path in sorted(checkpoint.rglob("*"), reverse=True):
        if path.is_file() or path.is_symlink():
            path.unlink()
        elif path.is_dir():
            path.rmdir()
    checkpoint.rmdir()
    assert (resume / "optimizer.pt").read_bytes() == b"optimizer"
    assert (exact_model / "model.safetensors").read_bytes() == b"exact-export"
    assert future_checkpoint.is_dir()


def test_non_milestone_completion_does_not_create_a_preservation_tree(tmp_path: Path) -> None:
    output_root = tmp_path / "output"
    _write_checkpoint(output_root, 14500)
    (output_root / "model.safetensors").write_bytes(b"final-model")
    (output_root / "trainer_state.json").write_text(json.dumps({"global_step": 14500}))
    export_path = output_root / "exported-checkpoint-14500"
    export_path.mkdir()
    (export_path / "model.safetensors").write_bytes(b"export")
    process, calls, _ = _start_lifecycle(
        tmp_path,
        output_root,
        target_step=14500,
        child_command="exit 0",
    )

    stdout, stderr = process.communicate(timeout=10)

    assert process.returncode == 0, (stdout, stderr)
    assert not calls.exists()
    assert not (output_root / "milestones").exists()


def test_final_target_is_also_a_permanent_milestone(tmp_path: Path) -> None:
    output_root = tmp_path / "output"
    _write_checkpoint(output_root, 25350)
    (output_root / "model.safetensors").write_bytes(b"final-model")
    (output_root / "trainer_state.json").write_text(json.dumps({"global_step": 25391}))
    export_path = output_root / "exported-checkpoint-25391"
    export_path.mkdir()
    (export_path / "model.safetensors").write_bytes(b"final-export")
    process, calls, _ = _start_lifecycle(
        tmp_path,
        output_root,
        target_step=25391,
        child_command="exit 0",
    )

    stdout, stderr = process.communicate(timeout=10)

    assert process.returncode == 0, (stdout, stderr)
    assert not calls.exists()
    milestone = output_root / "milestones/step-025391"
    manifest = json.loads((milestone / "manifest.json").read_text())
    assert manifest["exact_model_step"] == 25391
    assert manifest["resume_checkpoint_step"] == 25350
    assert (milestone / "exact-model").resolve() == export_path.resolve()


def test_milestone_copy_fallback_is_checksum_verified_and_recorded(tmp_path: Path) -> None:
    output_root = tmp_path / "output"
    checkpoint = _write_checkpoint(output_root, 4150)
    (output_root / "model.safetensors").write_bytes(b"final-model")
    (output_root / "trainer_state.json").write_text(json.dumps({"global_step": 4166}))
    export_path = output_root / "exported-checkpoint-4166"
    export_path.mkdir()
    (export_path / "model.safetensors").write_bytes(b"export")
    process, _, _ = _start_lifecycle(
        tmp_path,
        output_root,
        target_step=4166,
        child_command="exit 0",
        milestone_force_copy=True,
    )

    stdout, stderr = process.communicate(timeout=10)

    assert process.returncode == 0, (stdout, stderr)
    milestone = output_root / "milestones/step-004166"
    resume = milestone / "resume-checkpoint-004150"
    manifest = json.loads((milestone / "manifest.json").read_text())
    assert set(manifest["resume_checkpoint_storage"].values()) == {"copy"}
    assert (resume / "optimizer.pt").stat().st_ino != (checkpoint / "optimizer.pt").stat().st_ino
    assert (resume / "optimizer.pt").read_bytes() == (checkpoint / "optimizer.pt").read_bytes()
    assert manifest["resume_checkpoint_sha256"]["optimizer.pt"] == hashlib.sha256(
        b"optimizer"
    ).hexdigest()


def test_incomplete_newer_checkpoint_does_not_hide_latest_complete_one(tmp_path: Path) -> None:
    output_root = tmp_path / "output"
    expected = _write_checkpoint(output_root, 20)
    incomplete = output_root / "checkpoint-40"
    incomplete.mkdir()
    (incomplete / "trainer_state.json").write_text(json.dumps({"global_step": 40}))

    completed = subprocess.run(
        ["bash", str(_LIFECYCLE), "latest-complete", str(output_root)],
        capture_output=True,
        check=False,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.strip() == f"{expected}\t20"


def test_zero_byte_checkpoint_state_is_not_resumable(tmp_path: Path) -> None:
    output_root = tmp_path / "output"
    checkpoint = output_root / "checkpoint-20"
    checkpoint.mkdir(parents=True)
    (checkpoint / "model.safetensors").write_bytes(b"")
    (checkpoint / "optimizer.pt").write_bytes(b"")
    (checkpoint / "scheduler.pt").write_bytes(b"")
    (checkpoint / "trainer_state.json").write_text(json.dumps({"global_step": 20}))

    completed = subprocess.run(
        ["bash", str(_LIFECYCLE), "latest-complete", str(output_root)],
        capture_output=True,
        check=False,
        text=True,
    )

    assert completed.returncode != 0
    assert completed.stdout == ""


def test_checkpoint_missing_a_trainer_rank_rng_state_is_not_resumable(tmp_path: Path) -> None:
    output_root = tmp_path / "output"
    checkpoint = _write_checkpoint(output_root, 20)
    (checkpoint / "rng_state_7.pth").unlink()

    completed = subprocess.run(
        ["bash", str(_LIFECYCLE), "latest-complete", str(output_root)],
        capture_output=True,
        check=False,
        text=True,
    )

    assert completed.returncode != 0
    assert completed.stdout == ""


def test_resume_preflight_quarantines_incomplete_newer_checkpoint(tmp_path: Path) -> None:
    """Transformers cannot select a newer partial checkpoint after preflight."""
    output_root = tmp_path / "output"
    complete = _write_checkpoint(output_root, 20)
    incomplete = output_root / "checkpoint-40"
    incomplete.mkdir()
    (incomplete / "trainer_state.json").write_text(json.dumps({"global_step": 40}))
    command = f'source "{_LIFECYCLE}"; drafter_prepare_training_output'

    completed = subprocess.run(
        ["bash", "-c", command],
        capture_output=True,
        check=False,
        env={
            **os.environ,
            "OUTPUT_ROOT": str(output_root),
            "TRAINING_IDENTITY": "experiment-1",
            "TRAINING_FINGERPRINT": "fingerprint-1",
            "SOURCE_SHA": "a" * 40,
            "LEGACY_ADOPTION_SOURCE_SHA": "b" * 40,
            "LEGACY_ADOPTION_CHECKPOINT_STEP": "20",
            "SLURM_JOB_ID": "4242",
            "SLURM_RESTART_COUNT": "0",
        },
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    assert complete.is_dir()
    assert not incomplete.exists()
    assert (
        output_root
        / "control/preflight/job-4242-attempt-0/quarantine/checkpoint-40/trainer_state.json"
    ).is_file()


def test_resume_preflight_rejects_output_identity_mismatch(tmp_path: Path) -> None:
    """A different experiment cannot adopt an existing output namespace."""
    output_root = tmp_path / "output"
    output_root.mkdir()
    command = f'source "{_LIFECYCLE}"; drafter_prepare_training_output'
    base_env = {
        **os.environ,
        "OUTPUT_ROOT": str(output_root),
        "SLURM_JOB_ID": "4242",
        "SLURM_RESTART_COUNT": "0",
        "TRAINING_FINGERPRINT": "fingerprint-1",
        "SOURCE_SHA": "a" * 40,
    }
    first = subprocess.run(
        ["bash", "-c", command],
        capture_output=True,
        check=False,
        env={**base_env, "TRAINING_IDENTITY": "experiment-1"},
        text=True,
    )
    second = subprocess.run(
        ["bash", "-c", command],
        capture_output=True,
        check=False,
        env={**base_env, "TRAINING_IDENTITY": "experiment-2"},
        text=True,
    )

    assert first.returncode == 0, first.stderr
    assert second.returncode != 0
    assert "identity mismatch" in second.stderr


def test_resume_preflight_rejects_unapproved_legacy_checkpoint(tmp_path: Path) -> None:
    """A complete legacy checkpoint still requires explicit source provenance."""
    output_root = tmp_path / "output"
    _write_checkpoint(output_root, 20)
    command = f'source "{_LIFECYCLE}"; drafter_prepare_training_output'
    completed = subprocess.run(
        ["bash", "-c", command],
        capture_output=True,
        check=False,
        env={
            **os.environ,
            "OUTPUT_ROOT": str(output_root),
            "TRAINING_IDENTITY": "experiment-1",
            "TRAINING_FINGERPRINT": "fingerprint-1",
            "SOURCE_SHA": "a" * 40,
            "SLURM_JOB_ID": "4242",
            "SLURM_RESTART_COUNT": "0",
        },
        text=True,
    )

    assert completed.returncode != 0
    assert "legacy checkpoint adoption requires" in completed.stderr
    assert not (output_root / "control/training-identity.json").exists()
