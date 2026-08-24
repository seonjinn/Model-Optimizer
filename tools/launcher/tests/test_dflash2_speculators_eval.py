# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Contracts for the matched Q30 Base DFlash2 step-4166 evaluation."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import subprocess
import tarfile
from pathlib import Path

import pytest
from common.specdec.dflash2_speculators_eval import (
    DATASET_REVISION,
    STANDARD_SUBSETS,
    compute_prompt_set,
    materialize_prompt_set,
    summarize_pair,
    validate_milestone_export,
    validate_output_equivalence,
)

_LAUNCHER = Path(__file__).resolve().parents[1]
_WRAPPER = _LAUNCHER / "common/specdec/run_speculators_eval.sh"
_PAIR = _LAUNCHER / "common/specdec/run_speculators_eval_pair.sbatch"
_SERVER_STAGER = _LAUNCHER / "common/specdec/stage_speculators_eval_server_runtime.sh"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_prompt_snapshot(tmp_path: Path, *, rows: int = 200) -> tuple[Path, Path]:
    hf_home = tmp_path / "hf"
    snapshot = (
        hf_home / "hub/datasets--RedHatAI--speculator_benchmarks/snapshots" / DATASET_REVISION
    )
    snapshot.mkdir(parents=True)
    files: dict[str, dict[str, str]] = {}
    for subset in STANDARD_SUBSETS:
        path = snapshot / f"{subset}.jsonl"
        path.write_text(
            "".join(
                json.dumps({"prompt": f"{subset} prompt {index}"}, sort_keys=True) + "\n"
                for index in range(rows)
            )
        )
        files[subset] = {"path": str(path), "sha256": _sha256(path)}
    manifest = tmp_path / "dataset-manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "dataset_id": "RedHatAI/speculator_benchmarks",
                "revision": DATASET_REVISION,
                "hf_home": str(hf_home),
                "files": files,
            },
            sort_keys=True,
        )
        + "\n"
    )
    return manifest, hf_home


def _write_ledger(path: Path, *, mutate: tuple[str, int] | None = None) -> None:
    with path.open("w") as stream:
        for subset in STANDARD_SUBSETS:
            for index in range(200):
                output = f"answer:{subset}:{index}"
                if mutate == (subset, index):
                    output += ":wrong"
                record = {
                    "subset": subset,
                    "index": index,
                    "source_row": index,
                    "prompt_sha256": hashlib.sha256(
                        f"{subset} prompt {index}".encode()
                    ).hexdigest(),
                    "output_sha256": hashlib.sha256(output.encode()).hexdigest(),
                    "finish_reason": "stop",
                }
                stream.write(json.dumps(record, sort_keys=True) + "\n")


def _write_perf(path: Path, output_tps: float, latency: float) -> None:
    fields = (
        "subset",
        "strategy",
        "max_concurrency",
        "request_budget",
        "completed_requests",
        "rps_median",
        "latency_median_s",
        "itl_median_ms",
        "ttft_median_ms",
        "output_tps_median",
        "total_output_tokens",
    )
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for subset in STANDARD_SUBSETS:
            writer.writerow(
                {
                    "subset": subset,
                    "strategy": "throughput",
                    "max_concurrency": 32,
                    "request_budget": 200,
                    "completed_requests": 200,
                    "rps_median": 1,
                    "latency_median_s": latency,
                    "itl_median_ms": 2,
                    "ttft_median_ms": 10,
                    "output_tps_median": output_tps,
                    "total_output_tokens": 2000,
                }
            )


def _write_acceptance(path: Path) -> None:
    positions = tuple(f"acceptance_at_pos_{index}" for index in range(7))
    fields = (
        "subset",
        "num_drafts",
        "num_draft_tokens",
        "num_accepted_tokens",
        "acceptance_length",
        *positions,
    )
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for subset in STANDARD_SUBSETS:
            writer.writerow(
                {
                    "subset": subset,
                    "num_drafts": 100,
                    "num_draft_tokens": 700,
                    "num_accepted_tokens": 350,
                    "acceptance_length": 4.5,
                    **dict.fromkeys(positions, 0.5),
                }
            )


def test_prompt_digest_binds_exact_revision_nine_subsets_and_200_rows(tmp_path: Path) -> None:
    """The benchmark prompt set is an exact ordered 9x200 prefix."""
    manifest, hf_home = _write_prompt_snapshot(tmp_path)

    provenance = compute_prompt_set(manifest, hf_home, requests_per_subset=200)

    assert provenance["revision"] == DATASET_REVISION
    assert provenance["subsets"] == list(STANDARD_SUBSETS)
    assert provenance["requests_per_subset"] == 200
    assert provenance["total_requests"] == 1800
    assert len(str(provenance["prompt_sha256"])) == 64


def test_prompt_digest_cycles_short_subsets_to_exact_request_budget(tmp_path: Path) -> None:
    """A finite benchmark subset is replayed deterministically to the 200-request budget."""
    manifest, hf_home = _write_prompt_snapshot(tmp_path, rows=3)

    provenance = compute_prompt_set(manifest, hf_home, requests_per_subset=200)

    assert provenance["total_requests"] == 1800
    assert all(entry["available_rows"] == 3 for entry in provenance["files"].values())


def test_materialized_prompt_set_is_exact_200_row_authenticated_cycle(tmp_path: Path) -> None:
    """GuideLLM and correctness consume the same source-bound 200-row files."""
    manifest, hf_home = _write_prompt_snapshot(tmp_path, rows=3)
    output_home = tmp_path / "matched-hf"
    output_manifest = tmp_path / "matched-manifest.json"

    materialize_prompt_set(manifest, hf_home, output_home, output_manifest)
    provenance = compute_prompt_set(output_manifest, output_home)

    assert provenance["total_requests"] == 1800
    assert provenance["ordered_prompts"][3]["source_row"] == 0
    for entry in provenance["files"].values():
        assert sum(1 for _ in Path(entry["path"]).open()) == 200

    payload = json.loads(output_manifest.read_text())
    source = Path(payload["derivation"]["source_files"][STANDARD_SUBSETS[0]]["path"])
    source.write_text(source.read_text().replace("prompt 0", "tampered", 1))
    with pytest.raises(ValueError):
        compute_prompt_set(output_manifest, output_home)


@pytest.mark.parametrize("mutation", ["revision", "empty-subset", "prompt"])
def test_prompt_digest_rejects_unmatched_inputs(tmp_path: Path, mutation: str) -> None:
    """Wrong revisions, empty subsets, and mutated bytes fail closed."""
    manifest, hf_home = _write_prompt_snapshot(
        tmp_path, rows=0 if mutation == "empty-subset" else 200
    )
    if mutation == "revision":
        payload = json.loads(manifest.read_text())
        payload["revision"] = "0" * 40
        manifest.write_text(json.dumps(payload))
    elif mutation == "prompt":
        payload = json.loads(manifest.read_text())
        path = Path(payload["files"][STANDARD_SUBSETS[0]]["path"])
        path.write_text(path.read_text().replace("prompt 0", "tampered", 1))

    with pytest.raises(ValueError):
        compute_prompt_set(manifest, hf_home, requests_per_subset=200)


def test_step4166_milestone_binds_exact_export_bytes(tmp_path: Path) -> None:
    """The evaluator consumes the lifecycle-preserved step-4166 export only."""
    output_root = tmp_path / "train"
    export = output_root / "export-4166"
    export.mkdir(parents=True)
    (export / "config.json").write_text(
        json.dumps(
            {
                "architectures": ["DFlash2DraftModel"],
                "block_size": 8,
                "num_attention_heads": 32,
                "num_key_value_heads": 4,
                "head_dim": 128,
                "intermediate_size": 6144,
                "dflash_config": {"projector_type": "dflash2"},
            }
        )
        + "\n"
    )
    (export / "model.safetensors").write_bytes(b"draft-weights")
    milestone = output_root / "milestones/step-004166"
    milestone.mkdir(parents=True)
    (milestone / "exact-model").symlink_to(export)
    hashes = {
        str(path.relative_to(export)): _sha256(path)
        for path in sorted(export.rglob("*"))
        if path.is_file()
    }
    manifest = milestone / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "exact_model_path": "../../export-4166",
                "exact_model_step": 4166,
                "exact_model_sha256": hashes,
                "resume_checkpoint_path": "resume-checkpoint-004166",
                "resume_checkpoint_step": 4166,
                "resume_checkpoint_sha256": {"trainer_state.json": "f" * 64},
            }
        )
    )

    result = validate_milestone_export(manifest, export, expected_step=4166)

    assert result["step"] == 4166
    assert result["export_path"] == str(export.resolve())
    assert len(str(result["export_tree_sha256"])) == 64
    (export / "model.safetensors").write_bytes(b"mutated-weights")
    with pytest.raises(ValueError, match="export file hashes"):
        validate_milestone_export(manifest, export, expected_step=4166)


def test_output_equivalence_is_a_fail_closed_gate(tmp_path: Path) -> None:
    """Any deterministic target-output mismatch blocks a speedup claim."""
    baseline = tmp_path / "baseline.jsonl"
    draft = tmp_path / "draft.jsonl"
    _write_ledger(baseline)
    _write_ledger(draft)

    receipt = validate_output_equivalence(baseline, draft, requests_per_subset=200)

    assert receipt["status"] == "passed"
    assert receipt["matched_requests"] == 1800
    expected_schedule = [
        {
            "subset": subset,
            "index": index,
            "source_row": index,
            "prompt_sha256": hashlib.sha256(f"{subset} prompt {index}".encode()).hexdigest(),
        }
        for subset in STANDARD_SUBSETS
        for index in range(200)
    ]
    validate_output_equivalence(
        baseline,
        draft,
        requests_per_subset=200,
        expected_prompt_schedule=expected_schedule,
    )
    expected_schedule[0]["source_row"] = 1
    with pytest.raises(ValueError, match="does not match artifact identity"):
        validate_output_equivalence(
            baseline,
            draft,
            requests_per_subset=200,
            expected_prompt_schedule=expected_schedule,
        )
    _write_ledger(draft, mutate=(STANDARD_SUBSETS[-1], 199))
    with pytest.raises(ValueError, match="target output mismatch"):
        validate_output_equivalence(baseline, draft, requests_per_subset=200)
    rows = baseline.read_text().splitlines()
    first = json.loads(rows[0])
    first["source_row"] = True
    rows[0] = json.dumps(first, sort_keys=True)
    baseline.write_text("\n".join(rows) + "\n")
    with pytest.raises(ValueError, match="ledger row schema"):
        validate_output_equivalence(baseline, draft, requests_per_subset=200)


def test_pair_summary_reports_per_gpu_speed_latency_and_acceptance(tmp_path: Path) -> None:
    """The report contains matched performance and speculative-quality metrics."""
    baseline = tmp_path / "baseline"
    draft = tmp_path / "draft"
    baseline.mkdir()
    draft.mkdir()
    _write_perf(baseline / "perf_results.csv", output_tps=100, latency=0.8)
    _write_perf(draft / "perf_results.csv", output_tps=160, latency=0.5)
    _write_acceptance(draft / "acceptance.csv")

    report = summarize_pair(baseline, draft, tensor_parallel_size=2)

    assert report["aggregate"]["output_tps_per_gpu_baseline"] == 50
    assert report["aggregate"]["output_tps_per_gpu_dflash2"] == 80
    assert report["aggregate"]["output_tps_speedup"] == 1.6
    assert report["aggregate"]["latency_speedup"] == 1.6
    assert report["aggregate"]["acceptance_rate"] == 0.5
    assert report["aggregate"]["mean_accepted_length"] == 4.5
    assert len(report["subsets"]) == 9

    acceptance_path = draft / "acceptance.csv"
    acceptance_path.write_text(acceptance_path.read_text().replace(",350,4.5,", ",0,1.0,"))
    zero_report = summarize_pair(baseline, draft, tensor_parallel_size=2)
    assert zero_report["aggregate"]["acceptance_rate"] == 0


def test_split_runtime_pair_wiring_is_mandatory_for_matched_dflash2() -> None:
    """Server and GuideLLM client runtimes are independently pinned."""
    pair = _PAIR.read_text()
    wrapper = _WRAPPER.read_text()

    for required in (
        "CLIENT_RUNTIME_ARCHIVE",
        "CLIENT_RUNTIME_ARCHIVE_SHA256",
        "SERVER_RUNTIME_ARCHIVE",
        "SERVER_RUNTIME_ARCHIVE_SHA256",
        "SERVER_RUNTIME_RECEIPT_SHA256",
        "dflash2:8:7",
        "correctness:1:200:2|performance:32:200:2",
    ):
        assert required in pair
    assert 'SPECULATORS_CLIENT_RUNTIME="${JOB_CLIENT_RUNTIME}"' in pair
    assert 'VLLM_SERVER_RUNTIME="${JOB_SERVER_RUNTIME}"' in pair
    assert '"${SERVER_PYTHON}" "${SERVER_ARGS[@]}"' in wrapper
    assert '"${SPECULATORS_CLIENT_RUNTIME}/bin/python3" "${subset_args[@]}"' in wrapper
    assert '[[ "${SPEC_METHOD}" == dflash2 ]] && printf dflash' in wrapper
    assert "export VLLM_USE_V2_MODEL_RUNNER=1" in wrapper


def test_exact_matched_matrix_requires_c1_c32_200_and_tp2() -> None:
    """Only the approved C1/C32 TP2+TP2 matrix is accepted."""
    pair = _PAIR.read_text()

    assert 'case "${PAIR_PHASE}:${concurrency}:${max_requests}:${tp_size}"' in pair
    assert "correctness:1:200:2|performance:32:200:2" in pair
    assert "baseline:0:0" in pair
    assert "dflash2:8:7" in pair
    assert "c1-performance-report.json" in pair
    assert "c32-performance-report.json" in pair
    assert '"${STUDY_HELPER}" materialize-prompts' in pair
    assert 'DATASET_MANIFEST_PATH="${MATCHED_DATASET_MANIFEST_PATH}"' in pair
    assert 'HF_HOME_DURABLE="${MATCHED_HF_HOME}"' in pair
    equivalence_run = pair.index("run_pair_cells 1")
    correctness_gate = pair.index("compare-outputs", equivalence_run)
    performance_run = pair.index("run_pair_cells 0", correctness_gate)
    assert equivalence_run < correctness_gate < performance_run
    assert "--concurrency 1" in pair
    assert "--concurrency 32" in pair


def test_server_runtime_stager_binds_archive_and_embedded_receipt(tmp_path: Path) -> None:
    """The server venv is extracted once with both immutable markers."""
    source = tmp_path / "source"
    (source / "bin").mkdir(parents=True)
    python = source / "bin/python"
    python.write_text("#!/bin/bash\nexit 0\n")
    python.chmod(0o755)
    old_venv = "/durable/source-runtime"
    (source / "bin/activate").write_text(f'export VIRTUAL_ENV="{old_venv}"\n')
    (source / "pyvenv.cfg").write_text(f"home = {old_venv}\n")
    receipt = source / "dflash2-vllm-runtime-receipt.json"
    receipt.write_text('{"fixture":true}\n')
    archive = tmp_path / "server.tar"
    with tarfile.open(archive, "w") as stream:
        for path in source.rglob("*"):
            stream.add(path, arcname=path.relative_to(source))
    scratch = tmp_path / "scratch"
    destination = scratch / "123/server"
    env = {
        **os.environ,
        "SERVER_RUNTIME_ARCHIVE": str(archive),
        "SERVER_RUNTIME_ARCHIVE_SHA256": _sha256(archive),
        "SERVER_RUNTIME_RECEIPT_SHA256": _sha256(receipt),
        "VLLM_SERVER_RUNTIME": str(destination),
        "MARS_SCRATCH_ROOT": str(scratch),
        "SLURM_JOB_ID": "123",
    }

    first = subprocess.run(
        ["bash", str(_SERVER_STAGER)], env=env, capture_output=True, text=True, check=False
    )
    second = subprocess.run(
        ["bash", str(_SERVER_STAGER)], env=env, capture_output=True, text=True, check=False
    )

    assert first.returncode == 0, first.stderr
    assert second.returncode == 0, second.stderr
    assert (destination / ".archive.sha256").read_text().strip() == _sha256(archive)
    assert _sha256(destination / "dflash2-vllm-runtime-receipt.json") == _sha256(receipt)
    assert str(destination) in (destination / "bin/activate").read_text()


def test_server_runtime_stager_rejects_receipt_tamper(tmp_path: Path) -> None:
    """An archive cannot be relabelled with an unrelated runtime receipt."""
    source = tmp_path / "source"
    (source / "bin").mkdir(parents=True)
    python = source / "bin/python"
    python.write_text("#!/bin/bash\nexit 0\n")
    python.chmod(0o755)
    (source / "bin/activate").write_text('export VIRTUAL_ENV="/old/runtime"\n')
    (source / "pyvenv.cfg").write_text("home = /old/runtime\n")
    (source / "dflash2-vllm-runtime-receipt.json").write_text("{}\n")
    archive = tmp_path / "server.tar"
    with tarfile.open(archive, "w") as stream:
        for path in source.rglob("*"):
            stream.add(path, arcname=path.relative_to(source))
    scratch = tmp_path / "scratch"
    result = subprocess.run(
        ["bash", str(_SERVER_STAGER)],
        env={
            **os.environ,
            "SERVER_RUNTIME_ARCHIVE": str(archive),
            "SERVER_RUNTIME_ARCHIVE_SHA256": _sha256(archive),
            "SERVER_RUNTIME_RECEIPT_SHA256": "0" * 64,
            "VLLM_SERVER_RUNTIME": str(scratch / "123/server"),
            "MARS_SCRATCH_ROOT": str(scratch),
            "SLURM_JOB_ID": "123",
        },
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode != 0
