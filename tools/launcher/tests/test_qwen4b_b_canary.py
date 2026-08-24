# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Contracts for the B-balanced OCI-HSG canary path."""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import pytest
from common.specdec import build_qwen4b_b_canary as b_builder
from common.specdec.build_qwen4b_b_canary import (
    CANARY_CELL_QUOTAS,
    CANARY_LANGUAGE_QUOTAS,
    build_canary_occurrences,
)
from common.specdec.qwen4b_b_canary_manifest import (
    BCanaryManifest,
    BCanaryTopology,
    load_b_canary_manifest,
    write_b_canary_manifest,
)
from common.specdec.qwen4b_b_readiness import (
    BReadinessInputs,
    Task9BalancedView,
    assess_b_readiness,
)


def _task9_view(shard_root: Path) -> Task9BalancedView:
    shards = tuple(f"shard-{index:03d}.jsonl" for index in range(201))
    for shard in shards:
        (shard_root / shard).write_text("{}\n")
    return Task9BalancedView(
        strategy="B-balanced",
        occurrence_count=2_000_000,
        cell_occurrence_counts={
            "math": 500_000,
            "code": 400_000,
            "stem": 500_000,
            "chat": 400_000,
            "multilingual": 200_000,
        },
        multilingual_occurrence_counts=dict.fromkeys(("de", "ja", "es", "fr", "it"), 40000),
        declared_shards=shards,
        shard_root=shard_root,
        trainer_epochs=1,
        global_batch_size=512,
        segment_occurrences=(1_300_000, 700_000),
        segment_steps=(2_540, 1_368),
        cumulative_steps=(2_540, 3_908),
        segment_final_valid_occurrences=(32, 96),
        source_native_responses=True,
        publication_sha256="a" * 64,
        destination_sha256="b" * 64,
        tokenizer_sha256="c" * 64,
        chat_template_sha256="d" * 64,
        assistant_loss_mask_sha256="e" * 64,
    )


def _write_exact_canary_shards(root: Path, *, shard_count: int = 201) -> tuple[Path, ...]:
    root.mkdir()
    shard_rows: list[list[str]] = [[] for _ in range(shard_count)]
    ordinal = 0
    for cell, quota in b_builder.CANARY_CELL_QUOTAS.items():
        languages = b_builder.CANARY_LANGUAGE_QUOTAS if cell == "multilingual" else {"": quota}
        for language, language_quota in languages.items():
            for index in range(language_quota):
                row = {
                    "cell": cell,
                    "language": language,
                    "source_native": True,
                    "uuid": f"{cell}-{language}-{index}",
                }
                shard_rows[ordinal % shard_count].append(
                    json.dumps(row, sort_keys=True, separators=(",", ":"))
                )
                ordinal += 1
        for language in languages:
            extra = {
                "cell": cell,
                "language": language,
                "source_native": True,
                "uuid": f"{cell}-{language}-extra",
            }
            shard_rows[ordinal % shard_count].append(
                json.dumps(extra, sort_keys=True, separators=(",", ":"))
            )
            ordinal += 1
    shards = tuple(root / f"part-{index:03d}.jsonl" for index in range(shard_count))
    for path, rows in zip(shards, shard_rows, strict=True):
        path.write_text("\n".join(rows) + "\n")
    return shards


def test_b_readiness_is_independent_and_rejects_orphan_shards(tmp_path: Path) -> None:
    """B accepts its complete Task9 projection without any A-repair receipt."""
    shard_root = tmp_path / "b-shards"
    shard_root.mkdir()
    inputs = BReadinessInputs(task9_b_view=_task9_view(shard_root), source_commit="f" * 40)

    receipt = assess_b_readiness(inputs)

    assert receipt.ready
    assert receipt.blocker_codes == ()
    assert receipt.declared_shard_count == 201
    (shard_root / "orphan.jsonl").write_text("{}\n")
    rejected = assess_b_readiness(inputs)
    assert not rejected.ready
    assert rejected.blocker_codes == ("B_ORPHAN_SHARD",)


def test_b_readiness_rejects_renormalized_quota_and_wrong_schedule(tmp_path: Path) -> None:
    """B fails closed when a Task9 projection diverges from the scientific policy."""
    shard_root = tmp_path / "b-shards"
    shard_root.mkdir()
    view = _task9_view(shard_root)
    inputs = BReadinessInputs(
        task9_b_view=Task9BalancedView(
            **{
                **view.__dict__,
                "cell_occurrence_counts": {**view.cell_occurrence_counts, "math": 499_999},
                "cumulative_steps": (2_540, 25_391),
            }
        ),
        source_commit="f" * 40,
    )

    receipt = assess_b_readiness(inputs)

    assert not receipt.ready
    assert receipt.blocker_codes == ("B_CELL_QUOTA", "B_SCHEDULE")


def test_b_canary_builder_preserves_exact_scaled_cell_and_language_quotas() -> None:
    """The B canary uses exactly 102,400 source-native occurrences."""
    rows: list[dict[str, object]] = []
    for cell, quota in CANARY_CELL_QUOTAS.items():
        if cell == "multilingual":
            for language, language_quota in CANARY_LANGUAGE_QUOTAS.items():
                rows.extend(
                    {
                        "uuid": f"{language}-{index}",
                        "cell": cell,
                        "language": language,
                        "source_native": True,
                    }
                    for index in range(language_quota)
                )
        else:
            rows.extend(
                {"uuid": f"{cell}-{index}", "cell": cell, "language": "", "source_native": True}
                for index in range(quota)
            )

    selected = build_canary_occurrences(rows, seed=17)

    assert len(selected) == 102_400
    assert {
        cell: sum(row["cell"] == cell for row in selected) for cell in CANARY_CELL_QUOTAS
    } == CANARY_CELL_QUOTAS
    assert {
        language: sum(row["language"] == language for row in selected)
        for language in CANARY_LANGUAGE_QUOTAS
    } == CANARY_LANGUAGE_QUOTAS


def test_b_builder_defaults_to_slurm_cpus_and_caps_workers_by_declared_shards() -> None:
    """Worker resolution cannot oversubscribe the allocation or the 201-shard inventory."""
    assert b_builder.resolve_worker_count(
        requested_workers=None,
        declared_shard_count=201,
        environ={"SLURM_CPUS_PER_TASK": "96"},
    ) == (96, 96)
    assert b_builder.resolve_worker_count(
        requested_workers=200,
        declared_shard_count=7,
        environ={"SLURM_CPUS_PER_TASK": "96"},
    ) == (7, 96)


def test_parallel_b_builder_is_byte_identical_to_one_worker_and_records_provenance(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Changing worker count cannot change selected occurrence bytes or their identity."""
    monkeypatch.setattr(
        b_builder,
        "CANARY_CELL_QUOTAS",
        {"math": 4, "code": 3, "stem": 4, "chat": 3, "multilingual": 5},
    )
    monkeypatch.setattr(
        b_builder,
        "CANARY_LANGUAGE_QUOTAS",
        dict.fromkeys(("de", "ja", "es", "fr", "it"), 1),
    )
    shards = _write_exact_canary_shards(tmp_path / "shards")
    single_output = tmp_path / "single.jsonl"
    parallel_output = tmp_path / "parallel.jsonl"
    single_receipt = tmp_path / "single-receipt.json"
    parallel_receipt = tmp_path / "parallel-receipt.json"

    b_builder.materialize_canary_from_shards(
        shards,
        output_path=single_output,
        receipt_path=single_receipt,
        seed=17,
        requested_workers=1,
        environ={"SLURM_CPUS_PER_TASK": "96"},
        scratch_root=tmp_path / "single-scratch",
    )
    b_builder.materialize_canary_from_shards(
        shards,
        output_path=parallel_output,
        receipt_path=parallel_receipt,
        seed=17,
        requested_workers=8,
        environ={"SLURM_CPUS_PER_TASK": "96"},
        scratch_root=tmp_path / "parallel-scratch",
    )

    assert single_output.read_bytes() == parallel_output.read_bytes()
    single = json.loads(single_receipt.read_text())
    parallel = json.loads(parallel_receipt.read_text())
    assert single["output_sha256"] == parallel["output_sha256"]
    assert single["occurrence_count"] == parallel["occurrence_count"] == 19
    assert single["source_row_count"] == parallel["source_row_count"] == 28
    assert single["execution"]["effective_workers"] == 1
    assert parallel["execution"]["allocated_cpus"] == 96
    assert parallel["execution"]["effective_workers"] == 8
    assert parallel["execution"]["omp_threads_per_worker"] == 1
    assert parallel["execution"]["arrow_threads_per_worker"] == 1
    assert len(single["shard_timings"]) == len(parallel["shard_timings"]) == 201
    assert all("spool_path" not in timing for timing in parallel["shard_timings"])


def test_parallel_b_builder_propagates_worker_failure_without_publication(tmp_path: Path) -> None:
    """A malformed worker shard cannot leave output, receipt, or scratch state behind."""
    valid = tmp_path / "valid.jsonl"
    invalid = tmp_path / "invalid.jsonl"
    valid.write_text(
        json.dumps({"cell": "math", "language": "", "source_native": True, "uuid": "valid"}) + "\n"
    )
    invalid.write_text(
        json.dumps({"cell": "code", "language": "", "source_native": False, "uuid": "invalid"})
        + "\n"
    )
    output = tmp_path / "canary.jsonl"
    receipt = tmp_path / "receipt.json"
    scratch = tmp_path / "scratch"

    with pytest.raises(ValueError, match="source-native"):
        b_builder.materialize_canary_from_shards(
            (valid, invalid),
            output_path=output,
            receipt_path=receipt,
            seed=17,
            requested_workers=2,
            environ={"SLURM_CPUS_PER_TASK": "2"},
            scratch_root=scratch,
        )

    assert not output.exists()
    assert not receipt.exists()
    assert not scratch.exists()


def test_cpu_datamover_runner_passes_all_96_cpus_as_builder_workers(tmp_path: Path) -> None:
    """The CPU runner validates its allocation and passes every assigned core to the builder."""
    root = Path(__file__).resolve().parents[1] / "common/specdec"
    runner = root / "run_qwen4b_b_builder.sbatch"
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    capture = tmp_path / "python-args.txt"
    fake_python = fake_bin / "python3"
    fake_python.write_text('#!/usr/bin/env bash\nprintf "%s\\n" "$@" > "$CAPTURE"\n')
    fake_python.chmod(0o755)
    fake_git = fake_bin / "git"
    fake_git.write_text(
        "#!/usr/bin/env bash\n"
        'if [[ "$*" == *"rev-parse HEAD"* ]]; then printf "%s\\n" "$SOURCE_COMMIT"; fi\n'
    )
    fake_git.chmod(0o755)
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    task9_view = tmp_path / "task9.json"
    task9_view.write_text("{}\n")
    environment = {
        **os.environ,
        "PATH": f"{fake_bin}:{os.environ['PATH']}",
        "CAPTURE": str(capture),
        "REPO_ROOT": str(repo_root),
        "SOURCE_COMMIT": "a" * 40,
        "TASK9_B_VIEW": str(task9_view),
        "B_CANARY_OUTPUT": str(tmp_path / "canary.jsonl"),
        "B_CANARY_BUILD_RECEIPT": str(tmp_path / "receipt.json"),
        "B_CANARY_SEED": "17",
        "SLURM_JOB_ID": "123",
        "SLURM_JOB_PARTITION": "cpu_datamover",
        "SLURM_CPUS_PER_TASK": "96",
        "SLURM_TMPDIR": str(tmp_path),
        "SLURM_JOB_GPUS": "",
        "SLURM_GPUS_ON_NODE": "",
    }

    result = subprocess.run(
        ["bash", str(runner)],
        check=False,
        capture_output=True,
        text=True,
        env=environment,
    )

    assert result.returncode == 0, result.stderr
    arguments = capture.read_text().splitlines()
    assert arguments[0].endswith("build_qwen4b_b_canary.py")
    assert arguments[arguments.index("--workers") + 1] == "96"
    assert arguments[arguments.index("--source-commit") + 1] == "a" * 40


def test_manifest_binds_oci_16_node_200_step_runtime_contract(tmp_path: Path) -> None:
    """The immutable canary receipt describes all 64 working GPUs and GBS512."""
    manifest = BCanaryManifest(
        readiness_receipt_sha256="a" * 64,
        canary_occurrence_count=102_400,
        topology=BCanaryTopology(
            cluster="oci-hsg",
            account="nemotron_sw_post",
            partition="batch",
            nodes=16,
            segment=16,
            serve_nodes=8,
            train_nodes=8,
            gpus_per_node=4,
            cpu_datamover=96,
            per_device_batch_size=4,
            gradient_accumulation_steps=4,
            max_steps=200,
            wandb_project="sna-qwen3-4b-dataset-study",
        ),
        source_commit="f" * 40,
    )
    path = tmp_path / "b-canary.json"

    write_b_canary_manifest(path, manifest)

    assert load_b_canary_manifest(path) == manifest
    assert manifest.global_batch_size == 512
    assert manifest.active_gpu_ranks == 64
    assert manifest.scientific_milestone is False
    assert json.loads(path.read_text())["topology"]["cpu_datamover"] == 96


def test_canary_runner_and_submitter_enforce_bounded_evidence_contract() -> None:
    """The launch surface runs only B's bounded test-only canary."""
    root = Path(__file__).resolve().parents[1] / "common/specdec"
    runner = (root / "run_qwen4b_b_canary.sbatch").read_text()
    submitter = (root / "submit_qwen4b_b_canary.sh").read_text()

    assert "#SBATCH --nodes=16" in runner
    assert "#SBATCH --gpus-per-node=4" in runner
    assert "--cpus-per-task=96" in runner
    assert "--nodes=8" in runner
    assert "finite_loss" in runner
    assert "checkpoint_reloaded" in runner
    assert "drafter_exported" in runner
    assert "evaluator_completed" in runner
    assert "all_64_gpus_active" in runner
    assert 'payload["scientific_milestone"] = False' in runner
    assert "sbatch --test-only" in submitter
    assert "--test-only" in submitter
    assert "nemotron_sw_post" in submitter
    assert "nemotron_n4_post" in submitter
    assert "production submission is not supported" in submitter
