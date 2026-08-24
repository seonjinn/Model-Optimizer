# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Contracts for the genuine A-repair Qwen3-4B canary producer."""

from __future__ import annotations

import json
from dataclasses import replace
from hashlib import sha256
from pathlib import Path

import pytest
from common.specdec.build_qwen4b_a_canary import (
    A_COMPLEMENT_QUOTAS,
    build_a_canary_occurrences,
    resolve_worker_count,
)
from common.specdec.qwen4b_a_canary_manifest import (
    ACanaryManifest,
    ACanaryRuntimeIdentity,
    ACanaryTopology,
    last_finite_training_loss,
    load_a_canary_manifest,
    publish_a_authorization,
    validate_evaluator_receipt,
    write_a_canary_manifest,
)

_LAUNCHER_ROOT = Path(__file__).resolve().parents[1]


def _row(domain: str, name: str, *, language: str = "") -> dict[str, object]:
    producer = {
        "domain": domain,
        "language": language,
        "prompt_uuid": name,
        "response": f"native-{name}",
    }
    return {
        "prompt_uuid": name,
        "domain": domain,
        "lane": "ptv2",
        "context_bucket": "short",
        "input_ids": [1, 2, 3],
        "loss_mask": [0, 1, 1],
        "assistant_tokens": 2,
        "rejection_reason": None,
        "record_json": json.dumps(producer, sort_keys=True, separators=(",", ":")),
    }


def _scaled_rows() -> list[dict[str, object]]:
    rows = [_row("chat", f"h-{index}") for index in range(13)]
    rows.extend(_row("stem", f"stem-{index}") for index in range(3))
    for language in ("ja", "es", "fr", "it"):
        rows.extend(
            _row("multilingual", f"{language}-{index}", language=language) for index in range(2)
        )
    return rows


def test_a_selection_preserves_source_order_and_exact_scaled_quota() -> None:
    """A is prefix-order plus exact complement-order, not random sampling."""
    selected = build_a_canary_occurrences(
        _scaled_rows(),
        historical_boundary=13,
        historical_quota=13,
        complement_quotas={"stem": 2, "ja": 1, "es": 1, "fr": 1, "it": 1, "de": 0},
    )

    assert [row["uuid"] for row in selected] == [
        *(f"h-{index}" for index in range(13)),
        "stem-0",
        "stem-1",
        "ja-0",
        "es-0",
        "fr-0",
        "it-0",
    ]
    assert all(set(row) == {"uuid", "input_ids", "loss_mask"} for row in selected)
    assert all(row["input_ids"] == [1, 2, 3] for row in selected)


def test_a_selection_rejects_c_d_de_and_non_native_rows() -> None:
    """The repair segment fails closed on contamination or invalid supervision."""
    for domain in ("math", "code", "chat", "swe", "agentic", "tool_call"):
        rows = _scaled_rows()
        rows[13] = _row(domain, f"{domain}-contamination")
        with pytest.raises(ValueError, match="C/D contamination"):
            build_a_canary_occurrences(
                rows,
                historical_boundary=13,
                historical_quota=13,
                complement_quotas={
                    "stem": 2,
                    "ja": 1,
                    "es": 1,
                    "fr": 1,
                    "it": 1,
                    "de": 0,
                },
            )

    rows = _scaled_rows()
    rows[13] = _row("multilingual", "de-row", language="de")
    with pytest.raises(ValueError, match="German complement"):
        build_a_canary_occurrences(
            rows,
            historical_boundary=13,
            historical_quota=13,
            complement_quotas={"stem": 2, "ja": 1, "es": 1, "fr": 1, "it": 1, "de": 0},
        )

    rows = _scaled_rows()
    rows[0]["assistant_tokens"] = 0
    with pytest.raises(ValueError, match="source-native assistant"):
        build_a_canary_occurrences(
            rows,
            historical_boundary=13,
            historical_quota=13,
            complement_quotas={"stem": 2, "ja": 1, "es": 1, "fr": 1, "it": 1, "de": 0},
        )


def test_production_a_quotas_and_cpu_parallelism_are_pinned() -> None:
    """Production quotas scale exactly to 200 steps and use 96 CPUs."""
    assert A_COMPLEMENT_QUOTAS == {
        "stem": 10_240,
        "ja": 6_400,
        "es": 6_400,
        "fr": 6_400,
        "it": 6_400,
        "de": 0,
    }
    assert sum(A_COMPLEMENT_QUOTAS.values()) == 35_840
    assert resolve_worker_count(
        requested_workers=96,
        declared_shard_count=201,
        environ={"SLURM_CPUS_PER_TASK": "96"},
    ) == (96, 96)


def _manifest(tmp_path: Path) -> ACanaryManifest:
    runtime = ACanaryRuntimeIdentity(
        source_commit="a" * 40,
        target_revision="b" * 40,
        tokenizer_sha256="c" * 64,
        chat_template_sha256="d" * 64,
        container_sha256="e" * 64,
        producer_module_path="tools/launcher/common/specdec/qwen4b_a_canary_manifest.py",
        producer_module_sha256="f" * 64,
        runner_path="tools/launcher/common/specdec/run_qwen4b_a_canary.sbatch",
        runner_sha256="1" * 64,
        supervisor_path="tools/launcher/common/eagle3/train_eagle_streaming.sh",
        supervisor_sha256="2" * 64,
        recipe_path="tools/launcher/modules/Model-Optimizer/modelopt_recipes/general/speculative_decoding/dflash.yaml",
        recipe_sha256="3" * 64,
        evaluator_path="tools/launcher/common/specdec/run_qwen4b_a_canary_eval.sh",
        evaluator_sha256="8" * 64,
    )
    return ACanaryManifest(
        runtime=runtime,
        topology=ACanaryTopology(),
        task9_a_selection_path=str(tmp_path / "selection.json"),
        task9_a_selection_sha256="4" * 64,
        task8_publication_path=str(tmp_path / "publication.json"),
        task8_publication_sha256="5" * 64,
        builder_receipt_path=str(tmp_path / "build.json"),
        builder_receipt_sha256="6" * 64,
        builder_output_path=str(tmp_path / "canary.jsonl"),
        builder_output_sha256="7" * 64,
    )


def test_a_manifest_pins_actual_training_schema_and_64_gpu_topology(tmp_path: Path) -> None:
    """The canary remains a non-scientific, identical-parent 64-GPU run."""
    manifest = _manifest(tmp_path)

    assert manifest.topology.nodes == 16
    assert manifest.topology.serve_nodes == 8
    assert manifest.topology.trainer_nodes == 8
    assert manifest.topology.active_gpu_ranks == 64
    assert manifest.topology.global_batch_size == 512
    assert manifest.topology.max_steps == 200
    assert manifest.topology.training_seq_len == 4096
    assert manifest.topology.dflash_dims == (32, 8, 128, 9728)
    assert manifest.scientific_training_authorized is False
    with pytest.raises(ValueError, match="scientific"):
        replace(manifest, scientific_training_authorized=True)


def test_last_finite_loss_reads_hf_log_history() -> None:
    """Evidence comes from HF Trainer log_history, not a synthetic field."""
    state = {
        "global_step": 200,
        "log_history": [
            {"loss": 2.5, "step": 10},
            {"eval_loss": 2.0, "step": 100},
            {"loss": 1.25, "step": 200},
        ],
    }
    assert last_finite_training_loss(state, expected_step=200) == 1.25
    with pytest.raises(ValueError, match="finite"):
        last_finite_training_loss(
            {"global_step": 200, "log_history": [{"loss": float("nan"), "step": 200}]},
            expected_step=200,
        )


def test_a_manifest_digest_rejects_forgery(tmp_path: Path) -> None:
    """Any post-publication mutation invalidates the manifest identity."""
    path = tmp_path / "A_MANIFEST.json"
    write_a_canary_manifest(path, _manifest(tmp_path))
    assert load_a_canary_manifest(path) == _manifest(tmp_path)
    payload = json.loads(path.read_bytes())
    payload["occurrence_count"] = 102_399
    path.write_text(json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n")
    with pytest.raises(ValueError, match="identity"):
        load_a_canary_manifest(path)


def test_evaluator_receipt_requires_real_completed_requests_and_metrics(tmp_path: Path) -> None:
    """A file-only export check cannot masquerade as an evaluator run."""
    body = {
        "schema_version": 1,
        "slurm_job_id": "123",
        "status": "passed",
        "evaluator": "specdec-bench-v1",
        "export_sha256": "a" * 64,
        "completed_requests": 4,
        "metrics": {"acceptance_rate": 0.3, "mean_accepted_length": 2.1},
    }
    body["receipt_sha256"] = sha256(
        json.dumps(body, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    path = tmp_path / "eval.json"
    path.write_text(json.dumps(body, sort_keys=True, separators=(",", ":")) + "\n")

    validate_evaluator_receipt(path, job_id="123", export_sha256="a" * 64)
    forged = dict(body)
    forged["completed_requests"] = 0
    forged["receipt_sha256"] = sha256(
        json.dumps(
            {key: value for key, value in forged.items() if key != "receipt_sha256"},
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()
    path.write_text(json.dumps(forged, sort_keys=True, separators=(",", ":")) + "\n")
    with pytest.raises(ValueError, match="completed requests"):
        validate_evaluator_receipt(path, job_id="123", export_sha256="a" * 64)


def test_runner_uses_canonical_omegacon_keys_and_supervisor() -> None:
    """Runner overrides conform to the real recipe and inherit PID cleanup."""
    runner = (_LAUNCHER_ROOT / "common/specdec/run_qwen4b_a_canary.sbatch").read_text()
    required = (
        "common/eagle3/train_eagle_streaming.sh",
        'data.data_path="$node_root/input/corpus"',
        "training.training_seq_len=4096",
        "dflash.dflash_block_size=8",
        "dflash.dflash_architecture_config.num_attention_heads=32",
        "dflash.dflash_architecture_config.num_key_value_heads=8",
        "dflash.dflash_architecture_config.head_dim=128",
        "dflash.dflash_architecture_config.intermediate_size=9728",
    )
    assert all(fragment in runner for fragment in required)
    forbidden = ("model.draft_", "data.dataset", "data.max_length", "streaming_server_url=")
    assert all(fragment not in runner for fragment in forbidden)
    supervisor = (_LAUNCHER_ROOT / "common/eagle3/train_eagle_streaming.sh").read_text()
    assert "trap cleanup INT TERM EXIT" in supervisor
    assert 'kill "$pid"' in supervisor
    assert 'wait "$pid"' in supervisor
    assert "vllm serve" not in runner


def test_submitter_tests_both_jobs_before_cpu_then_afterok_gpu() -> None:
    """Both scheduler requests are tested before an ordered submission."""
    submitter = (_LAUNCHER_ROOT / "common/specdec/submit_qwen4b_a_canary.sh").read_text()
    assert submitter.index('--test-only --parsable "$CPU_SCRIPT"') < submitter.index(
        '--test-only --parsable "$GPU_SCRIPT"'
    )
    assert submitter.index('builder_job_id="$(sbatch --parsable') < submitter.index(
        '--dependency="afterok:${dependency_id}"'
    )
    assert "nemotron_sw_post|nemotron_n4_post" in submitter
    assert "scientific_training_authorized=false" in submitter


def test_publish_authorization_refuses_missing_real_evaluator(tmp_path: Path) -> None:
    """No authorization can be produced without genuine runtime artifacts."""
    with pytest.raises((FileNotFoundError, ValueError)):
        publish_a_authorization(
            tmp_path / "AUTHORIZATION.json",
            _manifest(tmp_path),
            job_id="123",
            checkpoint_path=tmp_path / "checkpoint-200",
            export_path=tmp_path / "export",
            gpu_evidence_path=tmp_path / "gpu.json",
            evaluation_receipt_path=tmp_path / "eval.json",
        )
