# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Production contracts for Qwen3-4B Task8 CPU publication."""

from __future__ import annotations

import hashlib
import importlib
import json
import os
import sqlite3
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[3]
_DATASET_ROOT = _REPO_ROOT / "examples/dataset"
if str(_DATASET_ROOT) not in sys.path:
    sys.path.insert(0, str(_DATASET_ROOT))

task8 = importlib.import_module("common.specdec.build_qwen4b_task8")


def _canonical(value: object) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class _Tokenizer:
    tokenizer_sha256 = "1" * 64
    chat_template = "template"
    assistant_loss_target_sha256 = "2" * 64

    def apply_chat_template(self, messages: object, **_: object) -> dict[str, list[int]]:
        assert isinstance(messages, list)
        return {"input_ids": [10, 11, 12], "assistant_masks": [0, 1, 1]}


def test_exact_two_million_publication_partitions_into_201_shards() -> None:
    """A 200-shard default would be rejected by both Task10 canary consumers."""
    assert task8.publication_rows_per_shard(2_000_000) == 9_951
    assert (2_000_000 + 9_951 - 1) // 9_951 == 201


def _selection(tmp_path: Path) -> tuple[SimpleNamespace, Path]:
    root = tmp_path / "selection"
    root.mkdir()
    index = root / "selection.sqlite3"
    connection = sqlite3.connect(index)
    connection.executescript(
        "CREATE TABLE source_rows(source_identity_sha256 TEXT,source_row INTEGER,cell TEXT,"
        "language TEXT,canonical_conversation TEXT,assistant_response TEXT);"
        "CREATE TABLE occurrences(strategy TEXT,ordinal INTEGER,prompt_uuid TEXT,"
        "source_identity_sha256 TEXT,source_row INTEGER,cell TEXT,reuse_index INTEGER,"
        "conversation_sha256 TEXT,assistant_response_sha256 TEXT);"
    )
    occurrence_digest = hashlib.sha256()
    response_digest = hashlib.sha256()
    for ordinal, (cell, language) in enumerate((("math", ""), ("multilingual", "ja"))):
        assistant = {"role": "assistant", "content": f"answer-{ordinal}"}
        conversation = _canonical({"messages": [{"role": "user", "content": "q"}, assistant]}).decode()
        response = _canonical(assistant).decode()
        source_identity = str(ordinal + 3) * 64
        prompt_uuid = str(ordinal + 5) * 64
        conversation_sha = hashlib.sha256(conversation.encode()).hexdigest()
        response_sha = hashlib.sha256(response.encode()).hexdigest()
        connection.execute(
            "INSERT INTO source_rows VALUES(?,?,?,?,?,?)",
            (source_identity, ordinal, cell, language, conversation, response),
        )
        occurrence = (
            ordinal,
            prompt_uuid,
            source_identity,
            ordinal,
            cell,
            0,
            conversation_sha,
            response_sha,
        )
        connection.execute(
            "INSERT INTO occurrences VALUES(?,?,?,?,?,?,?,?,?)",
            ("B-balanced", *occurrence),
        )
        occurrence_digest.update(_canonical(list(occurrence)) + b"\n")
        response_digest.update(
            _canonical([source_identity, ordinal, conversation_sha, response_sha]) + b"\n"
        )
    connection.commit()
    connection.close()
    return (
        SimpleNamespace(
            strategy="B-balanced",
            occurrence_count=2,
            trainer_epochs=1,
            unique_prompt_count=2,
            natural_duplicate_count=0,
            constructed_repeat_count=0,
            index_path=index,
            ordered_occurrences_sha256=occurrence_digest.hexdigest(),
            source_response_root_sha256=response_digest.hexdigest(),
            selection_sha256="9" * 64,
        ),
        index,
    )


def _artifact(root: Path, role: str, payload: dict[str, object]) -> tuple[Path, str]:
    root.mkdir()
    data = root / f"{role}.jsonl"
    data.write_bytes(b"{}\n")
    receipt = root / f"{role}.json"
    body = payload | {
        "role": role,
        "files": [
            {
                "path": data.name,
                "bytes": data.stat().st_size,
                "sha256": _digest(data),
            }
        ],
    }
    body["receipt_sha256"] = hashlib.sha256(_canonical(body)).hexdigest()
    receipt.write_bytes(_canonical(body) + b"\n")
    return receipt, _digest(receipt)


def test_small_genuine_selection_publishes_complete_six_role_bundle(tmp_path: Path) -> None:
    """Dropping any role or the real selection/token join would make publication fail."""
    view, _ = _selection(tmp_path)
    source, source_sha = _artifact(
        tmp_path / "source",
        "source",
        {"source_manifest_sha256": "8" * 64},
    )
    selection, selection_sha = _artifact(
        tmp_path / "selection-artifact",
        "selection",
        {"selection_sha256": view.selection_sha256},
    )
    response, response_sha = _artifact(
        tmp_path / "response",
        "response",
        {
            "selection_sha256": view.selection_sha256,
            "source_response_root_sha256": view.source_response_root_sha256,
            "schema_version": 1,
            "source_commit": "a" * 40,
            "occurrence_count": view.occurrence_count,
        },
    )
    rejection, rejection_sha = _artifact(
        tmp_path / "rejection",
        "rejection",
        {
            "schema_version": 1,
            "source_commit": "a" * 40,
            "selection_sha256": view.selection_sha256,
            "occurrence_count": view.occurrence_count,
            "rejection_count": 0,
            "reason_counts": {},
        },
    )

    receipt = task8.materialize_task8_publication(
        view=view,
        tokenizer=_Tokenizer(),
        tokenizer_sha256=_Tokenizer.tokenizer_sha256,
        chat_template_sha256=hashlib.sha256(_Tokenizer.chat_template.encode()).hexdigest(),
        assistant_loss_target_sha256=_Tokenizer.assistant_loss_target_sha256,
        training_config_sha256="3" * 64,
        source_receipt=source,
        source_receipt_sha256=source_sha,
        selection_receipt=selection,
        selection_receipt_sha256=selection_sha,
        response_receipt=response,
        response_receipt_sha256=response_sha,
        rejection_receipt=rejection,
        rejection_receipt_sha256=rejection_sha,
        work_root=tmp_path / "work",
        materialization_root=tmp_path / "materialized",
        publication_root=tmp_path / "published",
        source_commit="a" * 40,
        job_id="test-job",
        workers=1,
    )

    published = Path(receipt.published_path)
    assert published.is_dir()
    assert {path.name for path in (published / "inputs").iterdir()} == {
        "source",
        "selection",
        "response",
        "tokenized",
        "exposure",
        "rejection",
    }
    assert receipt.selection_manifest_sha256 == selection_sha


def test_missing_response_or_rejection_blocks_before_tokenization(tmp_path: Path) -> None:
    """Unavailable provenance must block instead of manufacturing a role receipt."""
    view, _ = _selection(tmp_path)
    missing = tmp_path / "missing.json"
    with pytest.raises(task8.Task8BuildError, match="response receipt"):
        task8.validate_required_role_receipts(
            view=view,
            source_commit="a" * 40,
            response_receipt=missing,
            response_receipt_sha256="4" * 64,
            rejection_receipt=missing,
            rejection_receipt_sha256="5" * 64,
        )


def test_response_declared_file_is_authenticated_before_tokenization(tmp_path: Path) -> None:
    """Changing selected-response evidence must fail before the expensive tokenizer phase."""
    view, _ = _selection(tmp_path)
    response, response_sha = _artifact(
        tmp_path / "response",
        "response",
        {
            "selection_sha256": view.selection_sha256,
            "source_response_root_sha256": view.source_response_root_sha256,
            "schema_version": 1,
            "source_commit": "a" * 40,
            "occurrence_count": view.occurrence_count,
        },
    )
    rejection, rejection_sha = _artifact(
        tmp_path / "rejection",
        "rejection",
        {
            "schema_version": 1,
            "source_commit": "a" * 40,
            "selection_sha256": view.selection_sha256,
            "occurrence_count": view.occurrence_count,
            "rejection_count": 0,
            "reason_counts": {},
        },
    )
    (response.parent / "response.jsonl").write_bytes(b"changed\n")
    with pytest.raises(task8.Task8BuildError, match="response declared file"):
        task8.validate_required_role_receipts(
            view=view,
            source_commit="a" * 40,
            response_receipt=response,
            response_receipt_sha256=response_sha,
            rejection_receipt=rejection,
            rejection_receipt_sha256=rejection_sha,
        )


def test_response_occurrence_count_cannot_misrepresent_selection(tmp_path: Path) -> None:
    """A pinned file set is insufficient when its typed response counts are false."""
    view, _ = _selection(tmp_path)
    response, response_sha = _artifact(
        tmp_path / "response",
        "response",
        {
            "schema_version": 1,
            "source_commit": "a" * 40,
            "selection_sha256": view.selection_sha256,
            "source_response_root_sha256": view.source_response_root_sha256,
            "occurrence_count": 1,
        },
    )
    rejection, rejection_sha = _artifact(
        tmp_path / "rejection",
        "rejection",
        {
            "schema_version": 1,
            "source_commit": "a" * 40,
            "selection_sha256": view.selection_sha256,
            "occurrence_count": view.occurrence_count,
            "rejection_count": 0,
            "reason_counts": {},
        },
    )
    with pytest.raises(task8.Task8BuildError, match="response receipt schema"):
        task8.validate_required_role_receipts(
            view=view,
            source_commit="a" * 40,
            response_receipt=response,
            response_receipt_sha256=response_sha,
            rejection_receipt=rejection,
            rejection_receipt_sha256=rejection_sha,
        )


def test_intermediate_symlink_in_declared_response_path_is_rejected(tmp_path: Path) -> None:
    """A symlinked directory inside the receipt root must not be traversed."""
    view, _ = _selection(tmp_path)
    root = tmp_path / "response"
    root.mkdir()
    actual = root / "actual"
    actual.mkdir()
    data = actual / "response.jsonl"
    data.write_bytes(b"{}\n")
    (root / "nested").symlink_to(actual, target_is_directory=True)
    response = root / "response.json"
    body = {
        "schema_version": 1,
        "role": "response",
        "source_commit": "a" * 40,
        "selection_sha256": view.selection_sha256,
        "source_response_root_sha256": view.source_response_root_sha256,
        "occurrence_count": view.occurrence_count,
        "files": [
            {
                "path": "nested/response.jsonl",
                "bytes": data.stat().st_size,
                "sha256": _digest(data),
            }
        ],
    }
    body["receipt_sha256"] = hashlib.sha256(_canonical(body)).hexdigest()
    response.write_bytes(_canonical(body) + b"\n")
    rejection, rejection_sha = _artifact(
        tmp_path / "rejection",
        "rejection",
        {
            "schema_version": 1,
            "source_commit": "a" * 40,
            "selection_sha256": view.selection_sha256,
            "occurrence_count": view.occurrence_count,
            "rejection_count": 0,
            "reason_counts": {},
        },
    )
    with pytest.raises(task8.Task8BuildError, match="symlink"):
        task8.validate_required_role_receipts(
            view=view,
            source_commit="a" * 40,
            response_receipt=response,
            response_receipt_sha256=_digest(response),
            rejection_receipt=rejection,
            rejection_receipt_sha256=rejection_sha,
        )


def test_task9_loader_authenticates_policy_bytes_before_tokenization(tmp_path: Path) -> None:
    """A policy mutation retaining trainer_epochs=1 must still fail its pinned descriptor."""
    root = tmp_path / "task9"
    root.mkdir()
    index = root / "selection.sqlite3"
    connection = sqlite3.connect(index)
    connection.execute(
        "CREATE TABLE occurrences(strategy TEXT,ordinal INTEGER,prompt_uuid TEXT,"
        "source_identity_sha256 TEXT,source_row INTEGER,cell TEXT,reuse_index INTEGER,"
        "conversation_sha256 TEXT,assistant_response_sha256 TEXT)"
    )
    connection.commit()
    connection.close()
    policy = root / "policy.yaml"
    policy.write_text("trainer_epochs: 1\nsegment_occurrences: [1300000, 700000]\n")

    def descriptor(path: Path) -> dict[str, object]:
        return {"path": path.name, "bytes": path.stat().st_size, "sha256": _digest(path)}

    receipt = root / "SELECTION_RECEIPT.json"
    payload = {
        "schema_version": 3,
        "strategy": "B-balanced",
        "occurrence_count": 2_000_000,
        "execution_receipt": {"path": "EXECUTION_RECEIPT.json"},
        "selection_sha256": "9" * 64,
        "selection_identity": {
            "unique_prompt_count": 2_000_000,
            "ordered_occurrences_sha256": "7" * 64,
            "source_response_root_sha256": "8" * 64,
        },
        "index": descriptor(index),
        "policy": descriptor(policy),
    }
    receipt.write_bytes(_canonical(payload) + b"\n")
    receipt_sha = _digest(receipt)
    policy.write_text(
        "trainer_epochs: 1\nsegment_occurrences: [1300000, 700000]\nunpinned: true\n"
    )
    with pytest.raises(task8.Task8BuildError, match="policy authentication"):
        task8.load_task9_selection(receipt, receipt_sha, "a" * 40)


def test_runner_rejects_non_96_cpu_allocation_before_python(tmp_path: Path) -> None:
    """A wrong CPU allocation must never start the production Python producer."""
    runner = _REPO_ROOT / "tools/launcher/common/specdec/run_qwen4b_task8.sbatch"
    result = subprocess.run(
        ["bash", str(runner)],
        check=False,
        capture_output=True,
        text=True,
        env={"PATH": "/usr/bin:/bin", "SLURM_JOB_ID": "7", "SLURM_CPUS_PER_TASK": "95"},
    )
    assert result.returncode == 2
    assert "96 CPUs" in result.stderr


def test_runner_rejects_publication_outside_durable_study_root(tmp_path: Path) -> None:
    """A Task8 publication must never land in home, tmp, or an arbitrary absolute path."""
    runner = _REPO_ROOT / "tools/launcher/common/specdec/run_qwen4b_task8.sbatch"
    environment = {
        "PATH": "/usr/bin:/bin",
        "SLURM_JOB_ID": "7",
        "SLURM_CPUS_PER_TASK": "96",
        "REPO_ROOT": str(_REPO_ROOT),
        "SOURCE_COMMIT": "a" * 40,
        "SOURCE_RECEIPT": str(tmp_path / "source"),
        "SOURCE_RECEIPT_SHA256": "1" * 64,
        "SELECTION_RECEIPT": str(tmp_path / "selection"),
        "SELECTION_RECEIPT_SHA256": "2" * 64,
        "RESPONSE_RECEIPT": str(tmp_path / "response"),
        "RESPONSE_RECEIPT_SHA256": "3" * 64,
        "REJECTION_RECEIPT": str(tmp_path / "rejection"),
        "REJECTION_RECEIPT_SHA256": "4" * 64,
        "TOKENIZER_RECEIPT": str(tmp_path / "tokenizer"),
        "TOKENIZER_RECEIPT_SHA256": "5" * 64,
        "ASSISTANT_LOSS_TARGET_SHA256": "6" * 64,
        "TRAINING_CONFIG_SHA256": "7" * 64,
        "IMAGE_PATH": str(tmp_path / "image.sqsh"),
        "IMAGE_SHA256": "8" * 64,
        "PUBLICATION_ROOT": "/tmp/qwen4b-task8",
    }
    result = subprocess.run(
        ["bash", str(runner)],
        check=False,
        capture_output=True,
        text=True,
        env=environment,
    )
    assert result.returncode == 2
    assert "durable dataset-study root" in result.stderr


def test_submitter_dry_run_requires_real_response_and_rejection_receipts(tmp_path: Path) -> None:
    """Dry-run scheduling cannot bypass the two externally produced trust roots."""
    submitter = _REPO_ROOT / "tools/launcher/common/specdec/submit_qwen4b_task8.sh"
    result = subprocess.run(
        ["bash", str(submitter), "--dry-run"],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 2
    assert "response-receipt" in result.stderr
    assert "rejection-receipt" in result.stderr


def test_submitter_dry_run_executes_only_scheduler_test_only(tmp_path: Path) -> None:
    """Removing the preflight or issuing a real sbatch would make this test fail."""
    submitter = _REPO_ROOT / "tools/launcher/common/specdec/submit_qwen4b_task8.sh"
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    calls = tmp_path / "sbatch.calls"
    fake_git = fake_bin / "git"
    fake_git.write_text(
        "#!/usr/bin/env bash\n"
        "case \"$*\" in\n"
        "  *'rev-parse --show-toplevel'*) printf '%s\\n' \"$FAKE_REPO_ROOT\" ;;\n"
        "  *'status --porcelain'*) : ;;\n"
        "  *'pull --ff-only'*) : ;;\n"
        "  *'rev-parse HEAD'*) printf '%040d\\n' 0 | tr 0 a ;;\n"
        "  *) exit 3 ;;\n"
        "esac\n"
    )
    fake_sbatch = fake_bin / "sbatch"
    fake_sbatch.write_text("#!/usr/bin/env bash\nprintf '%s\\n' \"$*\" >>\"$SBATCH_CALLS\"\n")
    fake_mkdir = fake_bin / "mkdir"
    fake_mkdir.write_text("#!/usr/bin/env bash\nexit 0\n")
    fake_git.chmod(0o755)
    fake_sbatch.chmod(0o755)
    fake_mkdir.chmod(0o755)
    inputs: list[str] = []
    for name in ("source", "selection", "response", "rejection", "tokenizer"):
        path = tmp_path / f"{name}.json"
        path.write_bytes(b"{}\n")
        inputs.extend((f"--{name}-receipt", str(path), f"--{name}-receipt-sha256", _digest(path)))
    image = tmp_path / "runtime.sqsh"
    image.write_bytes(b"authenticated-container")
    result = subprocess.run(
        [
            "bash",
            str(submitter),
            *inputs,
            "--assistant-loss-target-sha256",
            "1" * 64,
            "--training-config-sha256",
            "2" * 64,
            "--image-path",
            str(image),
            "--image-sha256",
            _digest(image),
            "--publication-root",
            "/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sna/"
            "modelopt-specdec/dataset-studies/task8/test-publication",
            "--account",
            "nemotron_n4_post",
            "--partition",
            "cpu_datamover",
            "--time",
            "03:00:00",
            "--dry-run",
        ],
        check=False,
        capture_output=True,
        text=True,
        env={
            **os.environ,
            "PATH": f"{fake_bin}:{os.environ['PATH']}",
            "FAKE_REPO_ROOT": str(_REPO_ROOT),
            "SBATCH_CALLS": str(calls),
        },
    )
    assert result.returncode == 0, result.stderr
    submitted = calls.read_text().splitlines()
    assert len(submitted) == 1
    assert "--test-only" in submitted[0]
    assert "--parsable" not in submitted[0]
    assert f"IMAGE_PATH={image}" in submitted[0]
    assert f"IMAGE_SHA256={_digest(image)}" in submitted[0]
