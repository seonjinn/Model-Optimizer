# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Contracts for the genuine A-repair Qwen3-4B canary producer."""

from __future__ import annotations

import json
import sqlite3
from dataclasses import replace
from hashlib import sha256
from pathlib import Path

import pytest
from common.specdec.build_qwen4b_a_canary import (
    A_COMPLEMENT_QUOTAS,
    build_a_canary_occurrences,
    load_task8_a_publication,
    resolve_worker_count,
)
from common.specdec.qwen4b_a_canary_manifest import (
    ACanaryManifest,
    ACanaryRuntimeIdentity,
    ACanaryTopology,
    last_finite_training_loss,
    load_a_canary_manifest,
    publish_a_authorization,
    publish_a_supervisor_completion,
    snapshot_target_identity,
    validate_evaluator_receipt,
    validate_supervisor_completion,
    write_a_canary_manifest,
)

_LAUNCHER_ROOT = Path(__file__).resolve().parents[1]


def _row(domain: str, name: str, *, language: str = "") -> dict[str, object]:
    producer = {
        "domain": domain,
        "language": language,
        "prompt_uuid": name,
        "response": f"native-{name}",
        "messages": [
            {"role": "user", "content": name},
            {"role": "assistant", "content": f"native-{name}"},
        ],
        "tools": [],
        "input_ids": [1, 2, 3],
        "loss_mask": [False, True, True],
        "assistant_tokens": 2,
    }
    return {
        "prompt_uuid": name,
        "domain": domain,
        "lane": "ptv2",
        "context_bucket": "short",
        "input_ids": [1, 2, 3],
        "loss_mask": [False, True, True],
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

    assert [row["conversation_id"] for row in selected] == [
        *(f"h-{index}" for index in range(13)),
        "stem-0",
        "stem-1",
        "ja-0",
        "es-0",
        "fr-0",
        "it-0",
    ]
    assert all(
        set(row) == {"conversation_id", "messages", "tools", "input_ids", "loss_mask"}
        for row in selected
    )
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
    with pytest.raises(ValueError, match="source-native"):
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


def _write_task9_task8_replay_fixture(root: Path, *, substitute: bool = False) -> tuple[str, str]:
    """Write a tiny schema-v3 Task9 selection/token view and Task8 publication."""
    def canonical(value: object) -> str:
        return json.dumps(value, sort_keys=True, separators=(",", ":"))

    def digest(raw: bytes) -> str:
        return sha256(raw).hexdigest()
    selection_files = root / "inputs/selection/files"
    token_files = root / "inputs/tokenized/files"
    shards = root / "shards"
    selection_files.mkdir(parents=True)
    token_files.mkdir(parents=True)
    shards.mkdir()
    messages = [
        {"role": "user", "content": "question"},
        {"role": "assistant", "content": "answer"},
    ]
    conversation = canonical({"messages": messages, "tools": []})
    response = canonical(messages[-1])
    conversation_sha = digest(conversation.encode())
    response_sha = digest(response.encode())
    source_identity = "1" * 64
    prompt_uuid = "2" * 64
    occurrence = [
        0,
        prompt_uuid,
        source_identity,
        7,
        "stem",
        0,
        conversation_sha,
        response_sha,
    ]
    occurrence_root = digest((canonical(occurrence) + "\n").encode())
    response_root = digest(
        (canonical([source_identity, 7, conversation_sha, response_sha]) + "\n").encode()
    )
    index = selection_files / "selection.sqlite3"
    connection = sqlite3.connect(index)
    connection.execute(
        "CREATE TABLE source_rows(source_identity_sha256 TEXT,source_row INTEGER,cell TEXT,"
        "language TEXT,canonical_conversation TEXT,assistant_response TEXT)"
    )
    connection.execute(
        "CREATE TABLE occurrences(strategy TEXT,ordinal INTEGER,prompt_uuid TEXT,"
        "source_identity_sha256 TEXT,source_row INTEGER,cell TEXT,reuse_index INTEGER,"
        "conversation_sha256 TEXT,assistant_response_sha256 TEXT)"
    )
    connection.execute(
        "INSERT INTO source_rows VALUES(?,?,?,?,?,?)",
        (source_identity, 7, "stem", "", conversation, response),
    )
    connection.execute("INSERT INTO occurrences VALUES(?,?,?,?,?,?,?,?,?)", ("A-repair", *occurrence))
    connection.commit()
    connection.close()
    occurrence_shard = selection_files / "occurrences-000000.jsonl"
    occurrence_shard.write_text(canonical(occurrence) + "\n")
    def descriptor(path: Path) -> dict[str, object]:
        return {
            "path": path.name,
            "bytes": path.stat().st_size,
            "sha256": digest(path.read_bytes()),
        }
    identity = {
        "strategy": "A-repair",
        "occurrence_count": 1,
        "repair_complement_counts": {"stem": 1, "ja": 0, "es": 0, "fr": 0, "it": 0, "de": 0},
        "ordered_occurrences_sha256": occurrence_root,
        "source_response_root_sha256": response_root,
    }
    selection = {
        "schema_version": 3,
        "strategy": "A-repair",
        "occurrence_count": 1,
        "selection_sha256": digest(canonical(identity).encode()),
        "selection_identity": identity,
        "index": descriptor(index),
        "shards": [descriptor(occurrence_shard)],
    }
    selection["root_sha256"] = digest(canonical(selection).encode())
    selection_receipt = root / "inputs/selection/receipt.json"
    selection_receipt.write_text(canonical(selection) + "\n")
    selection_receipt_sha = digest(selection_receipt.read_bytes())

    token_db = token_files / "records.sqlite3"
    connection = sqlite3.connect(token_db)
    connection.execute(
        "CREATE TABLE records(ordinal INTEGER,prompt_uuid TEXT,input_ids_json TEXT,"
        "loss_mask_json TEXT,assistant_tokens INTEGER)"
    )
    connection.execute(
        "INSERT INTO records VALUES(?,?,?,?,?)",
        (0, prompt_uuid, "[1,2,3]", "[false,true,true]", 2),
    )
    connection.commit()
    connection.close()
    token = {
        "schema_version": 1,
        "strategy": "A-repair",
        "occurrence_count": 1,
        "selection_sha256": selection["selection_sha256"],
        "ordered_occurrences_sha256": occurrence_root,
        "source_response_root_sha256": response_root,
        "tokenizer_sha256": "3" * 64,
        "chat_template_sha256": "4" * 64,
        "database_path": token_db.name,
        "database_bytes": token_db.stat().st_size,
        "database_sha256": digest(token_db.read_bytes()),
    }
    token["receipt_sha256"] = digest(canonical(token).encode())
    (root / "inputs/tokenized/receipt.json").write_text(canonical(token) + "\n")

    producer_messages = (
        [messages[0], {"role": "assistant", "content": "forged"}]
        if substitute
        else messages
    )
    producer = {
        "prompt_uuid": prompt_uuid,
        "domain": "stem",
        "language": "",
        "messages": producer_messages,
        "tools": [],
        "response": "forged" if substitute else response,
        "input_ids": [1, 2, 3],
        "loss_mask": [False, True, True],
        "assistant_tokens": 2,
    }
    task8_row = {
        "prompt_uuid": prompt_uuid,
        "domain": "stem",
        "lane": "ptv2",
        "context_bucket": "short",
        "input_ids": [1, 2, 3],
        "loss_mask": [False, True, True],
        "assistant_tokens": 2,
        "rejection_reason": None,
        "record_json": canonical(producer),
    }
    shard = shards / "part-000000.jsonl"
    shard.write_text(canonical(task8_row) + "\n")
    shard_descriptor = {
        "path": "shards/part-000000.jsonl",
        "bytes": shard.stat().st_size,
        "row_count": 1,
        "sha256": digest(shard.read_bytes()),
    }
    manifest = {
        "schema": "modelopt-specdec-corpus-manifest-v1",
        "artifact_source_commit": "a" * 40,
        "selection_manifest_sha256": selection_receipt_sha,
        "shards": [shard_descriptor],
        "files": [
            {key: value for key, value in shard_descriptor.items() if key != "row_count"}
        ],
    }
    manifest_path = root / "CORPUS_MANIFEST.json"
    manifest_path.write_text(canonical(manifest) + "\n")
    publication = {
        "schema": "modelopt-specdec-publication-receipt-v1",
        "artifact_source_commit": "a" * 40,
        "selection_manifest_sha256": selection_receipt_sha,
        "corpus_manifest_sha256": digest(manifest_path.read_bytes()),
    }
    publication_path = root / "PUBLICATION.json"
    publication_path.write_text(canonical(publication) + "\n")
    return digest(publication_path.read_bytes()), selection_receipt_sha


def test_task9_to_task8_replay_rejects_authenticated_corpus_substitution(tmp_path: Path) -> None:
    """A Task8 bundle cannot replace/reorder Task9's exact source-native occurrence stream."""
    good = tmp_path / "good"
    publication_sha, selection_sha = _write_task9_task8_replay_fixture(good)
    loaded = load_task8_a_publication(
        good,
        expected_publication_sha256=publication_sha,
        expected_selection_receipt_sha256=selection_sha,
        source_commit="a" * 40,
        _expected_occurrences=1,
        _expected_shards=1,
        _expected_repair={"stem": 1, "ja": 0, "es": 0, "fr": 0, "it": 0, "de": 0},
    )
    assert loaded.tokenizer_sha256 == "3" * 64

    forged = tmp_path / "forged"
    forged_publication_sha, forged_selection_sha = _write_task9_task8_replay_fixture(
        forged, substitute=True
    )
    with pytest.raises(ValueError, match="exact Task9 occurrence"):
        load_task8_a_publication(
            forged,
            expected_publication_sha256=forged_publication_sha,
            expected_selection_receipt_sha256=forged_selection_sha,
            source_commit="a" * 40,
            _expected_occurrences=1,
            _expected_shards=1,
            _expected_repair={"stem": 1, "ja": 0, "es": 0, "fr": 0, "it": 0, "de": 0},
        )


def _manifest(tmp_path: Path) -> ACanaryManifest:
    target = tmp_path / "cache/snapshots" / ("b" * 40)
    target.mkdir(parents=True, exist_ok=True)
    (target / "config.json").write_text(
        json.dumps(
            {
                "hidden_size": 2560,
                "num_hidden_layers": 36,
                "num_attention_heads": 32,
                "num_key_value_heads": 8,
                "head_dim": 128,
                "intermediate_size": 9728,
                "vocab_size": 151936,
                "_commit_hash": "b" * 40,
                "model_type": "qwen3",
            },
            sort_keys=True,
            separators=(",", ":"),
        )
    )
    (target / "tokenizer.json").write_text("tokenizer")
    (target / "tokenizer_config.json").write_text(
        json.dumps({"chat_template": "{{ messages }}"}, separators=(",", ":"))
    )
    (target / "chat_template.jinja").write_text("{{ messages }}")
    container = tmp_path / "image.sqsh"
    container.write_bytes(b"container")
    identity = snapshot_target_identity(target, container_path=container)
    runtime = ACanaryRuntimeIdentity(
        source_commit="a" * 40,
        target_revision="b" * 40,
        target_path=str(target.resolve()),
        target_snapshot_sha256=identity["target_snapshot_sha256"],
        target_config_sha256=identity["target_config_sha256"],
        tokenizer_sha256=identity["tokenizer_sha256"],
        chat_template_sha256=identity["chat_template_sha256"],
        container_path=str(container.resolve()),
        container_sha256=identity["container_sha256"],
        wandb_netrc_sha256="e" * 64,
        wandb_dir=str((tmp_path / "wandb/run").resolve()),
        wandb_cache_dir=str((tmp_path / "wandb/cache").resolve()),
        wandb_config_dir=str((tmp_path / "wandb/config").resolve()),
        wandb_artifact_dir=str((tmp_path / "wandb/artifacts").resolve()),
        producer_module_path="tools/launcher/common/specdec/qwen4b_a_canary_manifest.py",
        producer_module_sha256="f" * 64,
        runner_path="tools/launcher/common/specdec/run_qwen4b_a_canary.sbatch",
        runner_sha256="1" * 64,
        supervisor_path="tools/launcher/common/eagle3/train_eagle_streaming.sh",
        supervisor_sha256="2" * 64,
        recipe_path="modelopt_recipes/general/speculative_decoding/dflash.yaml",
        recipe_sha256="3" * 64,
        evaluator_path="tools/launcher/common/specdec/run_qwen4b_a_canary_eval.sh",
        evaluator_sha256="8" * 64,
        exporter_path="examples/speculative_decoding/scripts/export_hf_checkpoint.py",
        exporter_sha256="9" * 64,
        train_script_path="examples/speculative_decoding/launch_train.sh",
        train_script_sha256="0" * 64,
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
    with pytest.raises(ValueError, match="step 200"):
        last_finite_training_loss(
            {"global_step": 200, "log_history": [{"loss": 1.25, "step": 199}]},
            expected_step=200,
        )
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
    forbidden = (
        "model.draft_",
        "data.dataset",
        "data.max_length",
        "data.mode",
        "data.sample_size",
        "streaming_server_url=",
    )
    assert all(fragment not in runner for fragment in forbidden)
    assert (
        'DFLASH_CONFIG="$REPO_ROOT/modelopt_recipes/general/speculative_decoding/dflash.yaml"'
        in runner
    )
    assert "LAUNCHER_ROOT/modules/Model-Optimizer/modelopt_recipes" not in runner
    supervisor = (_LAUNCHER_ROOT / "common/eagle3/train_eagle_streaming.sh").read_text()
    assert "trap cleanup INT TERM EXIT" in supervisor
    assert 'kill "$pid"' in supervisor
    assert 'wait "$pid"' in supervisor
    assert "vllm serve" not in runner
    assert 'WANDB_NETRC_PATH="/run/secrets/wandb.netrc"' in runner
    assert "validate_runtime_mounts" in runner
    assert "TARGET_PATH" not in runner


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
    assert "/run/secrets/wandb.netrc:ro" in submitter
    assert "--wandb-dir" in submitter
    assert "--wandb-cache-dir" in submitter
    assert "--wandb-config-dir" in submitter
    assert "--wandb-artifact-dir" in submitter
    assert '--container-mounts="$container_mounts"' in submitter


def test_target_snapshot_binds_exact_q4_bytes_and_rejects_mutation(tmp_path: Path) -> None:
    """The GPU job must rehash the exact target/config/tokenizer/template/container bytes."""
    manifest = _manifest(tmp_path)
    target = Path(manifest.runtime.target_path)
    assert manifest.runtime.target_snapshot_sha256
    (target / "config.json").write_text("{}")
    with pytest.raises(ValueError, match="target snapshot"):
        snapshot_target_identity(
            target,
            container_path=Path(manifest.runtime.container_path),
            expected=manifest.runtime,
        )


def test_supervisor_completion_is_job_issued_and_step200_exact(tmp_path: Path) -> None:
    """Authorization cannot invent reload/export success without a bound completion receipt."""
    manifest = _manifest(tmp_path)
    checkpoint = tmp_path / "checkpoint-200"
    export = tmp_path / "export"
    checkpoint.mkdir()
    export.mkdir()
    (checkpoint / "trainer_state.json").write_text(
        json.dumps({"global_step": 200, "log_history": [{"step": 200, "loss": 1.0}]})
    )
    (checkpoint / "weights.bin").write_bytes(b"checkpoint")
    (export / "config.json").write_text("{}")
    (export / "weights.bin").write_bytes(b"export")
    evaluation = tmp_path / "eval.json"
    export_sha = __import__(
        "common.specdec.qwen4b_a_canary_manifest", fromlist=["_directory_sha256"]
    )._directory_sha256(export)
    eval_body = {
        "schema_version": 1,
        "slurm_job_id": "123",
        "status": "passed",
        "evaluator": "specdec-bench-v1",
        "export_sha256": export_sha,
        "completed_requests": 1,
        "metrics": {"acceptance_rate": 0.5},
        "checkpoint_path": str(checkpoint),
        "checkpoint_sha256": __import__(
            "common.specdec.qwen4b_a_canary_manifest", fromlist=["_directory_sha256"]
        )._directory_sha256(checkpoint),
        "exporter_path": manifest.runtime.exporter_path,
        "exporter_sha256": manifest.runtime.exporter_sha256,
        "checkpoint_reloaded_by_exporter": True,
    }
    eval_body["receipt_sha256"] = sha256(
        json.dumps(eval_body, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    evaluation.write_text(json.dumps(eval_body, sort_keys=True, separators=(",", ":")) + "\n")
    completion = tmp_path / "completion.json"
    publish_a_supervisor_completion(
        completion,
        manifest,
        job_id="123",
        checkpoint_path=checkpoint,
        export_path=export,
        evaluation_receipt_path=evaluation,
        started_at="2026-08-23T12:00:00Z",
        finished_at="2026-08-23T12:05:00Z",
    )
    validate_supervisor_completion(
        completion,
        manifest=manifest,
        job_id="123",
        checkpoint_path=checkpoint,
        export_path=export,
        evaluation_receipt_path=evaluation,
    )
    forged = json.loads(completion.read_text())
    forged["checkpoint_reloaded_by_exporter"] = False
    unsigned = {key: value for key, value in forged.items() if key != "receipt_sha256"}
    forged["receipt_sha256"] = sha256(
        json.dumps(unsigned, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    completion.write_text(json.dumps(forged, sort_keys=True, separators=(",", ":")) + "\n")
    with pytest.raises(ValueError, match="completion"):
        validate_supervisor_completion(
            completion,
            manifest=manifest,
            job_id="123",
            checkpoint_path=checkpoint,
            export_path=export,
            evaluation_receipt_path=evaluation,
        )


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
            supervisor_completion_path=tmp_path / "completion.json",
            manifest_path=tmp_path / "manifest.json",
            repository_root=tmp_path,
        )
