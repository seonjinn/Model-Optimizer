# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Contracts for the B-balanced OCI-HSG canary path."""

from __future__ import annotations

import ast
import hashlib
import json
import os
import subprocess
import sys
from dataclasses import replace
from hashlib import sha256
from pathlib import Path
from typing import Any, cast

import pytest
from common.specdec import build_qwen4b_b_canary as b_builder
from common.specdec import qwen4b_b_atomic as b_atomic
from common.specdec import qwen4b_b_canary_manifest as b_manifest_module
from common.specdec import qwen4b_b_readiness as b_readiness_module
from common.specdec.build_qwen4b_b_canary import (
    CANARY_CELL_QUOTAS,
    CANARY_LANGUAGE_QUOTAS,
    build_canary_occurrences,
)
from common.specdec.qwen4b_b_canary_manifest import (
    BCanaryManifest,
    BCanaryRuntimeIdentity,
    BCanaryTopology,
    load_b_canary_manifest,
    publish_a_scheduler_observation,
    publish_b_canary_evidence,
    publish_b_exporter_invocation,
    publish_b_supervisor_completion,
    validate_a_authorization_receipt,
    validate_a_scheduler_observation,
    validate_bound_artifacts,
    validate_runtime_artifacts,
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
        projection_sha256="9" * 64,
    )


def _builder_artifacts(root: Path) -> tuple[Path, str, Path, str]:
    root.mkdir(exist_ok=True)
    output = root / "canary.jsonl"
    output.write_bytes(b"task8-row\n")
    output_sha = sha256(output.read_bytes()).hexdigest()
    body = {
        "schema_version": 1,
        "source_commit": "f" * 40,
        "source_projection_sha256": "9" * 64,
        "source_task8_publication_sha256": "a" * 64,
        "source_corpus_manifest_sha256": "2" * 64,
        "source_selection_receipt_sha256": "b" * 64,
        "source_shard_inventory_sha256": "3" * 64,
        "declared_shard_count": 201,
        "source_row_count": 2_000_000,
        "occurrence_count": 102_400,
        "output_path": "canary.jsonl",
        "output_bytes": output.stat().st_size,
        "output_sha256": output_sha,
        "execution": {},
        "timing": {},
        "shard_timings": [],
    }
    body["receipt_sha256"] = sha256(
        json.dumps(body, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    receipt = root / "BUILD_RECEIPT.json"
    receipt.write_text(json.dumps(body, sort_keys=True, separators=(",", ":")) + "\n")
    return receipt, sha256(receipt.read_bytes()).hexdigest(), output, output_sha


def _readiness_inputs(tmp_path: Path, view: Task9BalancedView) -> BReadinessInputs:
    receipt, receipt_sha, output, output_sha = _builder_artifacts(tmp_path / "builder")
    return BReadinessInputs(
        task9_b_view=view,
        source_commit="f" * 40,
        builder_receipt_path=receipt,
        builder_receipt_sha256=receipt_sha,
        builder_output_path=output,
        builder_output_sha256=output_sha,
        task8_publication_sha256="a" * 64,
        corpus_manifest_sha256="2" * 64,
        selection_receipt_sha256="b" * 64,
        shard_inventory_sha256="3" * 64,
    )


def _write_exact_canary_shards(root: Path, *, shard_count: int = 201) -> tuple[Path, ...]:
    root.mkdir()
    shard_rows: list[list[str]] = [[] for _ in range(shard_count)]
    ordinal = 0
    for cell, quota in b_builder.CANARY_CELL_QUOTAS.items():
        languages = b_builder.CANARY_LANGUAGE_QUOTAS if cell == "multilingual" else {"": quota}
        for language, language_quota in languages.items():
            for index in range(language_quota):
                row = _task8_row(cell, f"{cell}-{language}-{index}", language=language)
                shard_rows[ordinal % shard_count].append(
                    json.dumps(row, sort_keys=True, separators=(",", ":"))
                )
                ordinal += 1
        for language in languages:
            extra = _task8_row(cell, f"{cell}-{language}-extra", language=language)
            shard_rows[ordinal % shard_count].append(
                json.dumps(extra, sort_keys=True, separators=(",", ":"))
            )
            ordinal += 1
    shards = tuple(root / f"part-{index:03d}.jsonl" for index in range(shard_count))
    for path, rows in zip(shards, shard_rows, strict=True):
        path.write_text("\n".join(rows) + "\n")
    return shards


def _shard_descriptors(
    shards: tuple[Path, ...],
) -> tuple[b_builder.Task8Shard, ...]:
    return tuple(
        b_builder.Task8Shard(
            path=path,
            relative_path=f"shards/{path.name}",
            rows=sum(bool(line) for line in path.read_text().splitlines()),
            bytes=path.stat().st_size,
            sha256=sha256(path.read_bytes()).hexdigest(),
        )
        for path in shards
    )


def test_b_readiness_is_independent_and_rejects_orphan_shards(tmp_path: Path) -> None:
    """B accepts its complete Task9 projection without any A-repair receipt."""
    shard_root = tmp_path / "b-shards"
    shard_root.mkdir()
    inputs = _readiness_inputs(tmp_path, _task9_view(shard_root))

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
    artifacts = _builder_artifacts(tmp_path / "builder")
    inputs = BReadinessInputs(
        task9_b_view=Task9BalancedView(
            **{
                **view.__dict__,
                "cell_occurrence_counts": {**view.cell_occurrence_counts, "math": 499_999},
                "cumulative_steps": (2_540, 25_391),
            }
        ),
        source_commit="f" * 40,
        builder_receipt_path=artifacts[0],
        builder_receipt_sha256=artifacts[1],
        builder_output_path=artifacts[2],
        builder_output_sha256=artifacts[3],
        task8_publication_sha256="a" * 64,
        corpus_manifest_sha256="2" * 64,
        selection_receipt_sha256="b" * 64,
        shard_inventory_sha256="3" * 64,
    )

    receipt = assess_b_readiness(inputs)

    assert not receipt.ready
    assert receipt.blocker_codes == ("B_CELL_QUOTA", "B_SCHEDULE")


def test_b_readiness_rejects_tampered_builder_output(tmp_path: Path) -> None:
    """Readiness transitively authenticates builder receipt and output bytes."""
    shard_root = tmp_path / "b-shards"
    shard_root.mkdir()
    inputs = _readiness_inputs(tmp_path, _task9_view(shard_root))
    inputs.builder_output_path.write_bytes(b"tampered\n")

    receipt = assess_b_readiness(inputs)

    assert not receipt.ready
    assert "B_BUILDER_OUTPUT_IDENTITY" in receipt.blocker_codes


def test_b_canary_builder_preserves_exact_scaled_cell_and_language_quotas() -> None:
    """The B canary consumes Task8's normalized producer row contract."""
    rows: list[dict[str, object]] = []
    for cell, quota in CANARY_CELL_QUOTAS.items():
        if cell == "multilingual":
            for language, language_quota in CANARY_LANGUAGE_QUOTAS.items():
                rows.extend(
                    _task8_row(cell, f"{language}-{index}", language=language)
                    for index in range(language_quota)
                )
        else:
            rows.extend(_task8_row(cell, f"{cell}-{index}") for index in range(quota))

    selected = build_canary_occurrences(rows, seed=17)

    assert len(selected) == 102_400
    assert all(
        set(row) == {"conversation_id", "messages", "tools", "input_ids", "loss_mask"}
        for row in selected
    )
    assert len({str(row["conversation_id"]) for row in selected}) == len(selected)


def _task8_row(cell: str, identifier: str, *, language: str = "") -> dict[str, object]:
    messages = [
        {"role": "user", "content": f"question-{identifier}"},
        {"role": "assistant", "content": f"answer-{identifier}"},
    ]
    raw: dict[str, object] = {
        "prompt_uuid": sha256(identifier.encode()).hexdigest(),
        "domain": cell,
        "lane": "target-synthesis",
        "context_bucket": "short",
        "input_ids": [1, 2],
        "loss_mask": [False, True],
        "assistant_tokens": 1,
        "rejection_reason": None,
        "language": language,
        "messages": messages,
        "tools": [],
    }
    return {key: value for key, value in raw.items() if key != "language"} | {
        "record_json": json.dumps(raw, sort_keys=True, separators=(",", ":"))
    }


def test_builder_rejects_invented_pre_task8_row_schema() -> None:
    """An invented launcher-only row cannot stand in for producer-issued Task8 data."""
    with pytest.raises(ValueError, match="Task8"):
        build_canary_occurrences(
            [{"uuid": "x", "cell": "math", "language": "", "source_native": True}],
            seed=17,
        )


def test_builder_emits_canonical_streaming_dataset_rows(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Task8 wrappers become directly loadable StreamingDataset JSONL rows."""
    monkeypatch.setattr(
        b_builder,
        "CANARY_CELL_QUOTAS",
        {"math": 1, "code": 0, "stem": 0, "chat": 0, "multilingual": 0},
    )
    monkeypatch.setattr(
        b_builder,
        "CANARY_LANGUAGE_QUOTAS",
        dict.fromkeys(("de", "ja", "es", "fr", "it"), 0),
    )
    selected = build_canary_occurrences([_task8_row("math", "streaming")], seed=17)

    assert selected == [
        {
            "conversation_id": sha256(b"streaming").hexdigest(),
            "messages": [
                {"role": "user", "content": "question-streaming"},
                {"role": "assistant", "content": "answer-streaming"},
            ],
            "tools": [],
            "input_ids": [1, 2],
            "loss_mask": [0, 1],
        }
    ]
    loader_path = (
        Path(__file__).resolve().parents[3]
        / "modelopt/torch/speculative/plugins/hf_streaming_dataset.py"
    )
    module = ast.parse(loader_path.read_text())
    function = next(
        node
        for node in module.body
        if isinstance(node, ast.FunctionDef) and node.name == "normalize_streaming_entry"
    )
    namespace: dict[str, object] = {"json": json, "hashlib": hashlib}
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(loader_path), "exec"), namespace)
    normalize = namespace["normalize_streaming_entry"]
    assert callable(normalize)
    assert normalize(selected[0]) == (
        sha256(b"streaming").hexdigest(),
        selected[0]["messages"],
    )


def test_builder_accepts_developer_messages_and_validates_hf_tool_schema(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Source-native developer/tool conversations remain valid HF JSON rows."""
    monkeypatch.setattr(
        b_builder,
        "CANARY_CELL_QUOTAS",
        {"math": 1, "code": 0, "stem": 0, "chat": 0, "multilingual": 0},
    )
    monkeypatch.setattr(
        b_builder,
        "CANARY_LANGUAGE_QUOTAS",
        dict.fromkeys(("de", "ja", "es", "fr", "it"), 0),
    )
    row = _task8_row("math", "developer")
    producer = json.loads(str(row["record_json"]))
    producer["messages"].insert(0, {"role": "developer", "content": "Be exact."})
    producer_tools: list[dict[str, Any]] = [
        {
            "type": "function",
            "function": {
                "name": "lookup",
                "description": "Lookup a value.",
                "parameters": {"type": "object", "properties": {}},
            },
        }
    ]
    producer["tools"] = producer_tools
    row["record_json"] = json.dumps(producer, sort_keys=True, separators=(",", ":"))

    selected = build_canary_occurrences([row], seed=17)

    messages = cast("list[dict[str, object]]", selected[0]["messages"])
    assert messages[0]["role"] == "developer"
    assert selected[0]["tools"] == producer["tools"]
    producer_tools[0]["function"]["parameters"] = "not-json-schema"
    row["record_json"] = json.dumps(producer, sort_keys=True, separators=(",", ":"))
    with pytest.raises(ValueError, match="tool schema"):
        build_canary_occurrences([row], seed=17)


def test_builder_output_loads_with_huggingface_json_dataset(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Canonical builder JSONL is accepted by the production HF JSON loader."""
    from datasets import load_dataset  # pyright: ignore[reportMissingImports]

    monkeypatch.setattr(
        b_builder,
        "CANARY_CELL_QUOTAS",
        {"math": 1, "code": 0, "stem": 0, "chat": 0, "multilingual": 0},
    )
    monkeypatch.setattr(
        b_builder,
        "CANARY_LANGUAGE_QUOTAS",
        dict.fromkeys(("de", "ja", "es", "fr", "it"), 0),
    )
    selected = build_canary_occurrences([_task8_row("math", "hf-json")], seed=17)
    output = tmp_path / "canary.jsonl"
    output.write_text(
        "".join(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n" for row in selected)
    )

    dataset = load_dataset("json", data_files=str(output), split="train")

    assert len(dataset) == 1
    assert dataset[0]["conversation_id"] == sha256(b"hf-json").hexdigest()
    assert dataset[0]["messages"][-1] == {
        "role": "assistant",
        "content": "answer-hf-json",
    }
    assert dataset[0]["loss_mask"] == [0, 1]


def test_manifest_requires_transitive_builder_identities() -> None:
    """The GPU manifest cannot omit either immutable builder artifact identity."""
    with pytest.raises(TypeError):
        BCanaryManifest(  # pyright: ignore[reportCallIssue]
            readiness_receipt_sha256="a" * 64,
            canary_occurrence_count=102_400,
            topology=_topology(),
            source_commit="f" * 40,
        )


def test_runtime_artifacts_rehash_exact_qwen4b_snapshot_and_container(tmp_path: Path) -> None:
    """The GPU runtime cannot drift from the pinned target/tokenizer/template/container."""
    repo_root = Path(__file__).resolve().parents[3]
    revision = "1" * 40
    target = tmp_path / revision
    target.mkdir()
    (target / "config.json").write_text(
        json.dumps(
            {
                "model_type": "qwen3",
                "vocab_size": 151936,
                "hidden_size": 2560,
                "intermediate_size": 9728,
                "num_hidden_layers": 36,
                "num_attention_heads": 32,
                "num_key_value_heads": 8,
                "head_dim": 128,
            }
        )
    )
    template = target / "chat_template.jinja"
    template.write_text("{% generation %}{{ content }}{% endgeneration %}\n")
    (target / "tokenizer_config.json").write_text("{}\n")
    (target / "tokenizer.json").write_text("{}\n")
    weights = target / "model.safetensors"
    weights.write_bytes(b"weights")
    tokenizer_digest = sha256()
    for path in sorted((template, target / "tokenizer.json", target / "tokenizer_config.json")):
        tokenizer_digest.update(path.name.encode())
        tokenizer_digest.update(b"\0")
        tokenizer_digest.update(bytes.fromhex(sha256(path.read_bytes()).hexdigest()))
    container = tmp_path / "runtime.sqsh"
    container.write_bytes(b"runtime")
    corpus = tmp_path / "canary.jsonl"
    corpus.write_text("{}\n")
    base = _manifest(tmp_path, corpus)
    runtime = replace(
        base.runtime,
        target_path=str(target),
        target_revision=revision,
        target_snapshot_sha256=_directory_digest(target),
        tokenizer_sha256=tokenizer_digest.hexdigest(),
        chat_template_sha256=sha256(template.read_bytes()).hexdigest(),
        container_path=str(container),
        container_sha256=sha256(container.read_bytes()).hexdigest(),
        runner_sha256=sha256(
            (repo_root / "tools/launcher/common/specdec/run_qwen4b_b_canary.sbatch").read_bytes()
        ).hexdigest(),
        evaluator_sha256=sha256(
            (repo_root / "tools/launcher/common/specdec/run_qwen4b_b_canary_eval.sh").read_bytes()
        ).hexdigest(),
        train_script_sha256=sha256(
            (repo_root / "examples/speculative_decoding/launch_train.sh").read_bytes()
        ).hexdigest(),
        exporter_sha256=sha256(
            (
                repo_root / "examples/speculative_decoding/scripts/export_hf_checkpoint.py"
            ).read_bytes()
        ).hexdigest(),
    )
    manifest = replace(base, runtime=runtime)

    validate_runtime_artifacts(
        manifest,
        supervisor_path=Path(runtime.supervisor_path),
        config_path=Path(runtime.config_path),
        repo_root=repo_root,
        container_path=container,
    )
    weights.write_bytes(b"tampered")
    with pytest.raises(ValueError, match="full target snapshot"):
        validate_runtime_artifacts(
            manifest,
            supervisor_path=Path(runtime.supervisor_path),
            config_path=Path(runtime.config_path),
            repo_root=repo_root,
            container_path=container,
        )
    weights.write_bytes(b"weights")
    container.write_bytes(b"drift")
    with pytest.raises(ValueError, match="container identity"):
        validate_runtime_artifacts(
            manifest,
            supervisor_path=Path(runtime.supervisor_path),
            config_path=Path(runtime.config_path),
            repo_root=repo_root,
            container_path=container,
        )


def test_builder_public_api_has_no_unauthenticated_input_flag() -> None:
    """The public builder CLI accepts only the authenticated producer publication."""
    source = Path(b_builder.__file__).read_text(encoding="utf-8")
    assert 'add_argument("--input"' not in source
    assert 'add_argument("--task8-publication"' in source


def test_genuine_task9_task8_builder_readiness_runner_chain(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Real Task9/Task8 producers feed the CPU builder and GPU preflight chain."""
    module_dir = Path(__file__).resolve().parents[3] / "examples/dataset"
    sys.path.insert(0, str(module_dir))
    try:
        import qwen3_4b_ptv2_study as study  # pyright: ignore[reportMissingImports]
        import specdec_publication as publication  # pyright: ignore[reportMissingImports]
    finally:
        sys.path.pop(0)

    cell_counts = {"math": 4, "code": 3, "stem": 4, "chat": 3, "multilingual": 5}
    language_counts: dict[str, int] = dict.fromkeys(("de", "ja", "es", "fr", "it"), 1)
    monkeypatch.setattr(b_builder, "CANARY_CELL_QUOTAS", cell_counts)
    monkeypatch.setattr(b_builder, "CANARY_LANGUAGE_QUOTAS", language_counts)
    monkeypatch.setattr(b_readiness_module, "_CANARY_OCCURRENCE_COUNT", 19)
    monkeypatch.setattr(b_manifest_module, "_CANARY_OCCURRENCE_COUNT", 19)
    policy_path = module_dir / "qwen3_4b_ptv2_study.yaml"
    policy = replace(
        study.load_ptv2_study_policy(policy_path),
        balanced_occurrences=cell_counts,
        multilingual_occurrences=language_counts,
        segment_occurrences=(10, 9),
        segment_steps=(1, 1),
        cumulative_steps=(1, 2),
        segment_final_valid_occurrences=(10, 9),
    )
    source_rows = []
    ordinal = 0
    for cell, count in cell_counts.items():
        languages = language_counts if cell == "multilingual" else {"": count}
        for language, language_count in languages.items():
            for _ in range(language_count):
                prompt = f"{cell}-{language}-{ordinal}"
                conversation = json.dumps(
                    {"messages": [{"role": "user", "content": prompt}]},
                    sort_keys=True,
                    separators=(",", ":"),
                )
                response = json.dumps(
                    {"role": "assistant", "content": f"answer-{prompt}"},
                    sort_keys=True,
                    separators=(",", ":"),
                )
                source_rows.append(
                    study.PTV2StudySourceRow(
                        prompt_uuid=sha256(f"prompt:{prompt}".encode()).hexdigest(),
                        source_identity_sha256=sha256(f"source:{cell}".encode()).hexdigest(),
                        source_row=ordinal,
                        cell=cell,
                        canonical_conversation=conversation,
                        assistant_response=response,
                        language=language,
                    )
                )
                ordinal += 1
    view = study.select_ptv2_b_balanced_view(
        source_rows,
        policy=policy,
        output_root=tmp_path / "selection-index",
        trust_roots={
            "source_inventory_sha256": "1" * 64,
            "held_out_receipt_sha256": "3" * 64,
        },
    )
    selection_receipt = study.write_ptv2_selection_receipt(
        tmp_path / "selection-receipt",
        view,
        policy=policy,
        policy_path=policy_path,
        source_inventory_sha256="1" * 64,
        held_out_receipt_sha256="3" * 64,
    )

    def artifact(role: str, lineage: dict[str, object]) -> object:
        root = tmp_path / "inputs" / role
        root.mkdir(parents=True)
        declared = root / f"{role}.jsonl"
        declared.write_bytes(b"{}\n")
        payload = {
            "role": role,
            "files": [
                {
                    "path": declared.name,
                    "bytes": declared.stat().st_size,
                    "sha256": sha256(declared.read_bytes()).hexdigest(),
                }
            ],
            **lineage,
        }
        path = root / f"{role}.json"
        path.write_text(json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n")
        return publication.InputArtifact(
            role,
            path,
            sha256(path.read_bytes()).hexdigest(),
        )

    selection_sha = sha256(selection_receipt.read_bytes()).hexdigest()
    selection_identity = json.loads(selection_receipt.read_bytes())["selection_sha256"]
    response_root = "4" * 64
    tokenizer = "5" * 64
    template = "6" * 64
    loss_target = "7" * 64
    database = "8" * 64
    artifacts = (
        artifact("source", {"source_manifest_sha256": "1" * 64}),
        publication.InputArtifact("selection", selection_receipt, selection_sha),
        artifact(
            "response",
            {
                "selection_sha256": selection_identity,
                "source_response_root_sha256": response_root,
            },
        ),
        artifact(
            "tokenized",
            {
                "selection_sha256": selection_identity,
                "source_response_root_sha256": response_root,
                "tokenizer_sha256": tokenizer,
                "chat_template_sha256": template,
                "assistant_loss_target_sha256": loss_target,
                "database_sha256": database,
            },
        ),
        artifact(
            "exposure",
            {
                "selection_sha256": selection_identity,
                "source_response_root_sha256": response_root,
                "tokenizer_sha256": tokenizer,
                "chat_template_sha256": template,
                "assistant_loss_target_sha256": loss_target,
                "tokenized_sha256": database,
            },
        ),
        artifact("rejection", {"selection_sha256": selection_identity}),
    )
    raw_rows = [
        json.loads(str(_task8_row(row.cell, str(index), language=row.language)["record_json"]))
        for index, row in enumerate(source_rows)
    ]
    corpus_rows = [
        raw_rows[index % len(raw_rows)]
        | {"prompt_uuid": sha256(f"corpus:{index}".encode()).hexdigest()}
        for index in range(201)
    ]
    published_root = tmp_path / "task8-publication"
    publication.publish_bundle(
        publication.CorpusBundle(
            artifacts=artifacts,
            rows=lambda: iter(corpus_rows),
            prompt_count=201,
            assistant_token_count=201,
            quarantine_count=0,
            selection_manifest_sha256=selection_sha,
            artifact_source_commit="a" * 40,
        ),
        published_root,
        "producer-e2e",
        rows_per_shard=1,
    )
    publication_sha = sha256((published_root / "PUBLICATION.json").read_bytes()).hexdigest()
    authenticated = b_builder.load_task8_publication(
        published_root,
        expected_publication_sha256=publication_sha,
        expected_selection_receipt_sha256=selection_sha,
        source_commit="a" * 40,
        expected_occurrence_count=19,
        expected_cell_counts=cell_counts,
        expected_language_counts=language_counts,
    )
    production_policy = study.load_ptv2_study_policy(policy_path)
    projection_view = replace(
        view,
        occurrence_count=2_000_000,
        cell_occurrence_counts=production_policy.balanced_occurrences,
        multilingual_occurrence_counts=production_policy.multilingual_occurrences,
    )
    projection_path = tmp_path / "TASK9_B_BALANCED.json"
    study.write_task9_balanced_view_json(
        projection_path,
        projection_view,
        policy=production_policy,
        shard_root=published_root / "shards",
        declared_shards=(item.path.name for item in authenticated.shards),
        publication_sha256=authenticated.publication_sha256,
        destination_sha256="b" * 64,
        tokenizer_sha256=tokenizer,
        chat_template_sha256=template,
        assistant_loss_mask_sha256=loss_target,
    )
    projection_sha = sha256(projection_path.read_bytes()).hexdigest()
    task9_view = b_readiness_module.load_task9_balanced_view(projection_path, projection_sha)
    build = b_builder.materialize_canary_from_shards(
        (item.path for item in authenticated.shards),
        output_root=tmp_path / "build",
        job_id="producer-e2e",
        seed=17,
        requested_workers=4,
        environ={"SLURM_CPUS_PER_TASK": "96"},
        scratch_root=tmp_path / "scratch",
        source_projection_sha256=projection_sha,
        source_task8_publication_sha256=authenticated.publication_sha256,
        source_corpus_manifest_sha256=authenticated.corpus_manifest_sha256,
        source_selection_receipt_sha256=selection_sha,
        source_shard_inventory_sha256=authenticated.shard_inventory_sha256,
        source_commit="a" * 40,
        expected_shards=authenticated.shards,
    )
    readiness = assess_b_readiness(
        BReadinessInputs(
            task9_b_view=task9_view,
            source_commit="a" * 40,
            builder_receipt_path=build.receipt_path,
            builder_receipt_sha256=build.receipt_sha256,
            builder_output_path=build.output_path,
            builder_output_sha256=build.output_sha256,
            task8_publication_sha256=authenticated.publication_sha256,
            corpus_manifest_sha256=authenticated.corpus_manifest_sha256,
            selection_receipt_sha256=authenticated.selection_receipt_sha256,
            shard_inventory_sha256=authenticated.shard_inventory_sha256,
        )
    )
    assert readiness.ready
    readiness_path = tmp_path / "READINESS.json"
    b_readiness_module.write_b_readiness_receipt(readiness_path, readiness)
    readiness_identity = json.loads(readiness_path.read_bytes())["receipt_sha256"]
    manifest = BCanaryManifest(
        readiness_receipt_sha256=readiness_identity,
        builder_receipt_sha256=build.receipt_sha256,
        builder_output_sha256=build.output_sha256,
        task9_projection_sha256=task9_view.projection_sha256,
        task8_publication_sha256=authenticated.publication_sha256,
        corpus_manifest_sha256=authenticated.corpus_manifest_sha256,
        selection_receipt_sha256=authenticated.selection_receipt_sha256,
        shard_inventory_sha256=authenticated.shard_inventory_sha256,
        canary_occurrence_count=19,
        topology=_topology(),
        runtime=_runtime(tmp_path, build.output_path),
        source_commit="a" * 40,
    )
    manifest_path = tmp_path / "MANIFEST.json"
    write_b_canary_manifest(manifest_path, manifest)
    validate_bound_artifacts(
        load_b_canary_manifest(manifest_path),
        readiness_path=readiness_path,
        builder_receipt_path=build.receipt_path,
        builder_output_path=build.output_path,
    )


def _topology() -> BCanaryTopology:
    return BCanaryTopology(
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
    )


def _runtime(tmp_path: Path, corpus_path: Path) -> BCanaryRuntimeIdentity:
    supervisor = tmp_path / "repo/tools/launcher/common/eagle3/train_eagle_streaming.sh"
    config = tmp_path / "repo/modelopt_recipes/general/speculative_decoding/dflash.yaml"
    supervisor.parent.mkdir(parents=True, exist_ok=True)
    config.parent.mkdir(parents=True, exist_ok=True)
    supervisor.write_text("supervisor\n")
    config.write_text("config\n")
    return BCanaryRuntimeIdentity(
        target_model_id="Qwen/Qwen3-4B",
        target_path=str(tmp_path / "target"),
        target_revision="1" * 40,
        target_snapshot_sha256="1" * 64,
        tokenizer_sha256="2" * 64,
        chat_template_sha256="3" * 64,
        container_path=str(tmp_path / "runtime.sqsh"),
        container_sha256="4" * 64,
        thinking_mode="off",
        method="dflash",
        block_size=8,
        training_seq_len=4096,
        seed=42,
        supervisor_path=str(supervisor),
        supervisor_sha256=sha256(supervisor.read_bytes()).hexdigest(),
        config_path=str(config),
        config_sha256=sha256(config.read_bytes()).hexdigest(),
        runner_sha256="5" * 64,
        evaluator_sha256="6" * 64,
        train_script_sha256="7" * 64,
        exporter_sha256="8" * 64,
        wandb_netrc_sha256="9" * 64,
        wandb_durable_root="/lustre/q4b-tests/wandb",
        wandb_scratch_namespace="qwen4b-b",
        corpus_arm="B-balanced",
        corpus_path=str(corpus_path),
        output_root=str(tmp_path / "output"),
        wandb_project="sna-qwen3-4b-dataset-study",
        wandb_run_id="q4b-b-test",
    )


def test_wandb_runtime_requires_read_only_secret_and_job_scoped_storage(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """B W&B state is durable, while secrets are RO and caches stay job-local."""
    corpus = tmp_path / "canary.jsonl"
    corpus.write_text("{}\n")
    manifest = _manifest(tmp_path, corpus)
    netrc = tmp_path / "wandb.netrc"
    netrc.write_text("machine api.wandb.ai\n")
    manifest = replace(
        manifest,
        runtime=replace(
            manifest.runtime,
            wandb_netrc_sha256=sha256(netrc.read_bytes()).hexdigest(),
        ),
    )
    monkeypatch.setattr(b_manifest_module, "_WANDB_NETRC_PATH", netrc)
    monkeypatch.setattr(b_manifest_module, "_path_on_read_only_mount", lambda _path: True)
    slurm_tmpdir = tmp_path / "slurm-job"
    slurm_tmpdir.mkdir()

    paths = b_manifest_module.validate_wandb_runtime(
        manifest,
        netrc_path=netrc,
        slurm_tmpdir=slurm_tmpdir,
        job_id="42",
        repository_root=tmp_path / "repo",
    )

    assert paths == {
        "WANDB_DIR": manifest.runtime.wandb_durable_root + "/jobs/42/run",
        "WANDB_CONFIG_DIR": manifest.runtime.wandb_durable_root + "/jobs/42/config",
        "WANDB_ARTIFACT_DIR": manifest.runtime.wandb_durable_root + "/jobs/42/artifacts",
        "WANDB_CACHE_DIR": str(slurm_tmpdir / "qwen4b-b-42" / "wandb-cache"),
    }
    netrc.write_text("mutated\n")
    with pytest.raises(ValueError, match="netrc identity"):
        b_manifest_module.validate_wandb_runtime(
            manifest,
            netrc_path=netrc,
            slurm_tmpdir=slurm_tmpdir,
            job_id="42",
            repository_root=tmp_path / "repo",
        )


def _directory_digest(root: Path) -> str:
    digest = sha256()
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        digest.update(path.relative_to(root).as_posix().encode())
        digest.update(b"\0")
        digest.update(bytes.fromhex(sha256(path.read_bytes()).hexdigest()))
    return digest.hexdigest()


def _manifest(tmp_path: Path, corpus_path: Path) -> BCanaryManifest:
    return BCanaryManifest(
        readiness_receipt_sha256="a" * 64,
        builder_receipt_sha256="b" * 64,
        builder_output_sha256="c" * 64,
        task9_projection_sha256="d" * 64,
        task8_publication_sha256="e" * 64,
        corpus_manifest_sha256="f" * 64,
        selection_receipt_sha256="1" * 64,
        shard_inventory_sha256="2" * 64,
        canary_occurrence_count=102_400,
        topology=_topology(),
        runtime=_runtime(tmp_path, corpus_path),
        source_commit="f" * 40,
    )


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
    descriptors = _shard_descriptors(shards)
    single_root = tmp_path / "single"
    parallel_root = tmp_path / "parallel"

    b_builder.materialize_canary_from_shards(
        shards,
        output_root=single_root,
        job_id="single",
        seed=17,
        requested_workers=1,
        environ={"SLURM_CPUS_PER_TASK": "96"},
        scratch_root=tmp_path / "single-scratch",
        source_projection_sha256="a" * 64,
        source_task8_publication_sha256="d" * 64,
        source_corpus_manifest_sha256="e" * 64,
        source_selection_receipt_sha256="b" * 64,
        source_shard_inventory_sha256="f" * 64,
        source_commit="c" * 40,
        expected_shards=descriptors,
    )
    b_builder.materialize_canary_from_shards(
        shards,
        output_root=parallel_root,
        job_id="parallel",
        seed=17,
        requested_workers=8,
        environ={"SLURM_CPUS_PER_TASK": "96"},
        scratch_root=tmp_path / "parallel-scratch",
        source_projection_sha256="a" * 64,
        source_task8_publication_sha256="d" * 64,
        source_corpus_manifest_sha256="e" * 64,
        source_selection_receipt_sha256="b" * 64,
        source_shard_inventory_sha256="f" * 64,
        source_commit="c" * 40,
        expected_shards=descriptors,
    )

    assert (single_root / "canary.jsonl").read_bytes() == (
        parallel_root / "canary.jsonl"
    ).read_bytes()
    single = json.loads((single_root / "BUILD_RECEIPT.json").read_text())
    parallel = json.loads((parallel_root / "BUILD_RECEIPT.json").read_text())
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
    valid.write_text(json.dumps(_task8_row("math", "valid")) + "\n")
    invalid.write_text(json.dumps({"domain": "code", "prompt_uuid": "invalid"}) + "\n")
    output = tmp_path / "canary"
    scratch = tmp_path / "scratch"
    shards = (valid, invalid)
    descriptors = _shard_descriptors(shards)

    with pytest.raises(ValueError, match="Task8"):
        b_builder.materialize_canary_from_shards(
            shards,
            output_root=output,
            job_id="failure",
            seed=17,
            requested_workers=2,
            environ={"SLURM_CPUS_PER_TASK": "2"},
            scratch_root=scratch,
            source_projection_sha256="a" * 64,
            source_task8_publication_sha256="d" * 64,
            source_corpus_manifest_sha256="e" * 64,
            source_selection_receipt_sha256="b" * 64,
            source_shard_inventory_sha256="f" * 64,
            source_commit="c" * 40,
            expected_shards=descriptors,
        )

    assert not output.exists()
    assert scratch.exists()
    assert (tmp_path / ".canary.partial-failure").exists()


def test_shard_namespace_swap_is_rejected_after_single_fd_parse(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A pathname replacement cannot redirect or bless the already-open Task8 shard."""
    shard = tmp_path / "part.jsonl"
    shard.write_text(json.dumps(_task8_row("math", "stable")) + "\n")
    replacement = tmp_path / "replacement.jsonl"
    replacement.write_text(json.dumps(_task8_row("math", "replacement")) + "\n")
    original = b_builder._iter_authenticated_rows

    def replace_after_hash(source: object, path: Path) -> object:
        result = original(source, path)
        os.replace(replacement, shard)
        return result

    monkeypatch.setattr(b_builder, "_iter_authenticated_rows", replace_after_hash)
    task = b_builder._ShardTask(
        index=0,
        path=shard,
        spool_path=tmp_path / "spool.jsonl",
        expected_rows=1,
        expected_bytes=shard.stat().st_size,
        expected_sha256=sha256(shard.read_bytes()).hexdigest(),
        seed=17,
    )

    with pytest.raises(ValueError, match="mutated"):
        b_builder._process_shard(task)


def test_parent_fsync_ambiguity_preserves_typed_recovery_evidence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A post-rename durability failure preserves the destination and typed evidence."""
    destination = tmp_path / "artifact.json"

    def fail_parent_fsync(_path: Path) -> None:
        raise OSError("injected parent fsync failure")

    monkeypatch.setattr(b_atomic, "_fsync_directory", fail_parent_fsync)
    with pytest.raises(OSError, match="injected") as captured:
        b_atomic.atomic_publish_bytes(destination, b"payload\n", job_id="crash")

    state = b_atomic.recovery_state(captured.value)
    assert state.phase == "parent_fsync"
    assert state.destination_observation.status == "present"
    assert state.partial_observation.status == "absent"
    assert destination.read_bytes() == b"payload\n"


def test_cpu_datamover_runner_passes_all_96_cpus_as_builder_workers(tmp_path: Path) -> None:
    """The CPU runner validates its allocation and passes every assigned core to the builder."""
    root = Path(__file__).resolve().parents[1] / "common/specdec"
    runner = root / "run_qwen4b_b_builder.sbatch"
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    capture = tmp_path / "python-args.txt"
    fake_python = fake_bin / "python3"
    fake_python.write_text(
        '#!/usr/bin/env bash\nif [[ "${1:-}" == *"build_qwen4b_b_canary.py" ]]; then '
        'printf "%s\\n" "$@" > "$CAPTURE"; fi\n'
    )
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
    task8_publication = tmp_path / "task8"
    task8_publication.mkdir()
    environment = {
        **os.environ,
        "PATH": f"{fake_bin}:{os.environ['PATH']}",
        "CAPTURE": str(capture),
        "REPO_ROOT": str(repo_root),
        "SOURCE_COMMIT": "a" * 40,
        "TASK9_B_VIEW": str(task9_view),
        "TASK9_B_VIEW_SHA256": "d" * 64,
        "TASK8_PUBLICATION": str(task8_publication),
        "TASK8_PUBLICATION_SHA256": "b" * 64,
        "TASK9_SELECTION_RECEIPT_SHA256": "c" * 64,
        "B_CANARY_BUILD_ROOT": str(tmp_path / "canary"),
        "B_CANARY_READINESS": str(tmp_path / "readiness.json"),
        "B_CANARY_MANIFEST": str(tmp_path / "manifest.json"),
        "B_CANARY_SEED": "17",
        "B_CANARY_ACCOUNT": "nemotron_sw_post",
        "TARGET_PATH": str(tmp_path / "target"),
        "TARGET_REVISION": "e" * 40,
        "CONTAINER_IMAGE": str(tmp_path / "runtime.sqsh"),
        "CONTAINER_SHA256": "f" * 64,
        "B_CANARY_OUTPUT_ROOT": str(tmp_path / "output"),
        "B_CANARY_WANDB_RUN_ID": "q4b-b-test",
        "B_WANDB_NETRC_SHA256": "9" * 64,
        "B_WANDB_DURABLE_ROOT": "/lustre/q4b-tests/wandb",
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
    assert arguments[arguments.index("--job-id") + 1] == "123"
    assert arguments[arguments.index("--task8-publication") + 1] == str(task8_publication)


def test_manifest_binds_oci_16_node_200_step_runtime_contract(tmp_path: Path) -> None:
    """The immutable canary receipt describes all 64 working GPUs and GBS512."""
    manifest = BCanaryManifest(
        readiness_receipt_sha256="a" * 64,
        builder_receipt_sha256="b" * 64,
        builder_output_sha256="c" * 64,
        task9_projection_sha256="d" * 64,
        task8_publication_sha256="e" * 64,
        corpus_manifest_sha256="f" * 64,
        selection_receipt_sha256="1" * 64,
        shard_inventory_sha256="2" * 64,
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
        runtime=_runtime(tmp_path, tmp_path / "canary.jsonl"),
        source_commit="f" * 40,
    )
    path = tmp_path / "b-canary.json"

    write_b_canary_manifest(path, manifest)

    assert load_b_canary_manifest(path) == manifest
    assert manifest.global_batch_size == 512
    assert manifest.active_gpu_ranks == 64
    assert manifest.scientific_milestone is False
    assert json.loads(path.read_text())["topology"]["cpu_datamover"] == 96


def test_a_authorization_rejects_non_producer_fixture(tmp_path: Path) -> None:
    """Handwritten JSON cannot impersonate the genuine repository-owned A producer."""
    checkpoint = tmp_path / "a-checkpoint"
    export = tmp_path / "a-export"
    checkpoint.mkdir()
    export.mkdir()
    lineage_paths = {
        "task9_a_selection": tmp_path / "A_SELECTION.json",
        "task8_publication": tmp_path / "A_PUBLICATION.json",
        "builder_receipt": tmp_path / "A_BUILD_RECEIPT.json",
        "builder_output": tmp_path / "A_CANARY.jsonl",
    }
    selection_body = {
        "schema_version": 3,
        "strategy": "A-repair",
        "selection_identity": {"strategy": "A-repair"},
    }
    lineage_paths["task9_a_selection"].write_text(
        json.dumps(selection_body, sort_keys=True, separators=(",", ":")) + "\n"
    )
    selection_sha = sha256(lineage_paths["task9_a_selection"].read_bytes()).hexdigest()
    publication_body = {
        "selection_manifest_sha256": selection_sha,
        "artifact_source_commit": "f" * 40,
    }
    lineage_paths["task8_publication"].write_text(
        json.dumps(publication_body, sort_keys=True, separators=(",", ":")) + "\n"
    )
    publication_sha = sha256(lineage_paths["task8_publication"].read_bytes()).hexdigest()
    lineage_paths["builder_output"].write_text("A-repair-canary\n")
    builder_output_sha = sha256(lineage_paths["builder_output"].read_bytes()).hexdigest()
    builder_body = {
        "source_commit": "f" * 40,
        "source_projection_sha256": selection_sha,
        "source_task8_publication_sha256": publication_sha,
        "output_sha256": builder_output_sha,
    }
    builder_body["receipt_sha256"] = sha256(
        json.dumps(builder_body, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    lineage_paths["builder_receipt"].write_text(
        json.dumps(builder_body, sort_keys=True, separators=(",", ":")) + "\n"
    )
    (checkpoint / "trainer_state.json").write_text(
        json.dumps({"global_step": 200, "loss_history": [1.5, 1.25]}) + "\n"
    )
    (export / "config.json").write_text("{}\n")
    gpu = tmp_path / "a-gpu.json"
    gpu.write_text(json.dumps({"slurm_job_id": "42", "active_gpu_ranks": list(range(64))}))
    body = {
        "schema_version": 1,
        "producer": "qwen4b-a-repair-canary-job-v1",
        "arm": "A-repair",
        "authorization": "B-balanced-canary",
        "complete": True,
        "canary_completed": True,
        "source_commit": "f" * 40,
        "task9_a_selection_path": str(lineage_paths["task9_a_selection"]),
        "task9_a_selection_sha256": selection_sha,
        "task8_publication_path": str(lineage_paths["task8_publication"]),
        "task8_publication_sha256": publication_sha,
        "builder_receipt_path": str(lineage_paths["builder_receipt"]),
        "builder_receipt_sha256": sha256(lineage_paths["builder_receipt"].read_bytes()).hexdigest(),
        "builder_output_path": str(lineage_paths["builder_output"]),
        "builder_output_sha256": builder_output_sha,
        "target_model_id": "Qwen/Qwen3-4B",
        "target_revision": "5" * 40,
        "tokenizer_sha256": "6" * 64,
        "chat_template_sha256": "7" * 64,
        "container_sha256": "8" * 64,
        "thinking_mode": "off",
        "method": "dflash",
        "block_size": 8,
        "seed": 42,
        "global_batch_size": 512,
        "max_steps": 200,
        "finite_loss": 1.25,
        "checkpoint_reloaded": True,
        "drafter_exported": True,
        "all_gpus_active": True,
        "scientific_training_authorized": False,
        "slurm_job_id": "42",
        "checkpoint_path": str(checkpoint),
        "checkpoint_sha256": _directory_digest(checkpoint),
        "export_path": str(export),
        "export_sha256": _directory_digest(export),
        "gpu_evidence_path": str(gpu),
        "gpu_evidence_sha256": sha256(gpu.read_bytes()).hexdigest(),
    }
    body["receipt_sha256"] = sha256(
        json.dumps(body, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    receipt = tmp_path / "A_AUTHORIZATION.json"
    receipt.write_text(json.dumps(body, sort_keys=True, separators=(",", ":")) + "\n")
    pinned = sha256(receipt.read_bytes()).hexdigest()

    with pytest.raises((FileNotFoundError, ValueError), match=r"semantics|manifest|No such file"):
        validate_a_authorization_receipt(
            receipt,
            pinned,
            source_commit="f" * 40,
            target_revision="5" * 40,
            tokenizer_sha256="6" * 64,
            chat_template_sha256="7" * 64,
            container_sha256="8" * 64,
            repo_root=tmp_path,
        )


def test_b_evidence_is_derived_from_current_job_outputs_and_no_replace(tmp_path: Path) -> None:
    """Caller booleans cannot replace the canonical job-output evidence chain."""
    corpus = tmp_path / "canary.jsonl"
    corpus.write_text("{}\n")
    manifest = _manifest(tmp_path, corpus)
    checkpoint = Path(manifest.runtime.output_root) / "checkpoint" / "checkpoint-200"
    export = Path(manifest.runtime.output_root) / "export"
    checkpoint.mkdir(parents=True)
    export.mkdir(parents=True)
    exporter = tmp_path / "repo/examples/speculative_decoding/scripts/export_hf_checkpoint.py"
    exporter.parent.mkdir(parents=True)
    exporter.write_text("exporter\n")
    manifest = replace(
        manifest,
        runtime=replace(
            manifest.runtime,
            exporter_sha256=sha256(exporter.read_bytes()).hexdigest(),
        ),
    )
    (checkpoint / "trainer_state.json").write_text(
        json.dumps({"global_step": 200, "loss_history": [1.5, 1.25]}) + "\n"
    )
    (export / "config.json").write_text(
        json.dumps({"draft_model_type": "dflash", "draft_block_size": 8}) + "\n"
    )
    (export / "model.safetensors").write_bytes(b"weights")
    evaluation = tmp_path / "evaluation.json"
    evaluation_body = {
        "schema_version": 1,
        "slurm_job_id": "42",
        "status": "passed",
        "evaluator": "specdec-bench-v1",
        "export_sha256": _directory_digest(export),
        "checkpoint_path": str(checkpoint),
        "checkpoint_sha256": _directory_digest(checkpoint),
        "exporter_path": "examples/speculative_decoding/scripts/export_hf_checkpoint.py",
        "exporter_sha256": manifest.runtime.exporter_sha256,
        "exporter_invocation_sha256": "0" * 64,
        "checkpoint_reloaded_by_exporter": True,
        "completed_requests": 50,
        "metrics": {"acceptance_rate": 0.5, "generation_throughput": 123.0},
    }
    evaluation_body["receipt_sha256"] = sha256(
        json.dumps(evaluation_body, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    evaluation.write_text(json.dumps(evaluation_body, sort_keys=True, separators=(",", ":")) + "\n")
    gpu = tmp_path / "gpu.json"
    gpu.write_text(json.dumps({"slurm_job_id": "42", "active_gpu_ranks": list(range(64))}) + "\n")
    supervisor_receipt = tmp_path / "supervisor.json"
    with pytest.raises(ValueError, match="log_history"):
        publish_b_supervisor_completion(
            supervisor_receipt,
            manifest,
            job_id="42",
            checkpoint_path=checkpoint,
            export_path=export,
            exporter_receipt_path=tmp_path / "missing-exporter.json",
            evaluation_receipt_path=evaluation,
            started_at="2026-08-23T20:00:00+00:00",
            finished_at="2026-08-23T20:10:00+00:00",
        )
    (checkpoint / "trainer_state.json").write_text(
        json.dumps(
            {
                "global_step": 200,
                "log_history": [{"step": 199, "loss": 1.5}, {"step": 200, "loss": 1.25}],
            }
        )
        + "\n"
    )
    exporter_receipt = tmp_path / "EXPORTER_INVOCATION.json"
    publish_b_exporter_invocation(
        exporter_receipt,
        manifest,
        job_id="42",
        exporter_path=exporter,
        checkpoint_path=checkpoint,
        export_path=export,
        argv=[
            str(exporter),
            "--model_path",
            str(checkpoint),
            "--export_path",
            str(export),
        ],
        started_at="2026-08-23T20:08:00+00:00",
        finished_at="2026-08-23T20:09:00+00:00",
    )
    evaluation_body["exporter_invocation_path"] = str(exporter_receipt)
    evaluation_body["exporter_invocation_sha256"] = sha256(
        exporter_receipt.read_bytes()
    ).hexdigest()
    evaluation_body["checkpoint_sha256"] = _directory_digest(checkpoint)
    evaluation_body["receipt_sha256"] = sha256(
        json.dumps(
            {key: value for key, value in evaluation_body.items() if key != "receipt_sha256"},
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()
    evaluation.write_text(json.dumps(evaluation_body, sort_keys=True, separators=(",", ":")) + "\n")
    publish_b_supervisor_completion(
        supervisor_receipt,
        manifest,
        job_id="42",
        checkpoint_path=checkpoint,
        export_path=export,
        exporter_receipt_path=exporter_receipt,
        evaluation_receipt_path=evaluation,
        started_at="2026-08-23T20:00:00+00:00",
        finished_at="2026-08-23T20:10:00+00:00",
    )
    evidence = tmp_path / "B_EVIDENCE.json"

    publish_b_canary_evidence(
        evidence,
        manifest,
        job_id="42",
        checkpoint_path=checkpoint,
        export_path=export,
        exporter_receipt_path=exporter_receipt,
        evaluation_receipt_path=evaluation,
        gpu_evidence_path=gpu,
        supervisor_receipt_path=supervisor_receipt,
    )
    payload = json.loads(evidence.read_bytes())
    assert payload["finite_loss"] is True
    assert payload["all_64_gpus_active"] is True
    with pytest.raises(FileExistsError):
        publish_b_canary_evidence(
            evidence,
            manifest,
            job_id="42",
            checkpoint_path=checkpoint,
            export_path=export,
            exporter_receipt_path=exporter_receipt,
            evaluation_receipt_path=evaluation,
            gpu_evidence_path=gpu,
            supervisor_receipt_path=supervisor_receipt,
        )


def test_exporter_invocation_requires_exact_checkpoint_200_child(tmp_path: Path) -> None:
    """The final HF export must reload checkpoint-200, never its training parent."""
    corpus = tmp_path / "canary.jsonl"
    corpus.write_text("{}\n")
    manifest = _manifest(tmp_path, corpus)
    checkpoint_parent = Path(manifest.runtime.output_root) / "checkpoint"
    checkpoint = checkpoint_parent / "checkpoint-200"
    export = Path(manifest.runtime.output_root) / "export"
    exporter = tmp_path / "repo/examples/speculative_decoding/scripts/export_hf_checkpoint.py"
    exporter.parent.mkdir(parents=True)
    checkpoint.mkdir(parents=True)
    export.mkdir(parents=True)
    exporter.write_text("exporter\n")
    manifest = replace(
        manifest,
        runtime=replace(
            manifest.runtime,
            exporter_sha256=sha256(exporter.read_bytes()).hexdigest(),
        ),
    )
    (checkpoint / "model.safetensors").write_bytes(b"checkpoint")
    (export / "config.json").write_text("{}\n")
    (export / "model.safetensors").write_bytes(b"export")
    receipt = tmp_path / "EXPORTER_INVOCATION.json"

    with pytest.raises(ValueError, match="checkpoint-200"):
        publish_b_exporter_invocation(
            receipt,
            manifest,
            job_id="42",
            exporter_path=exporter,
            checkpoint_path=checkpoint_parent,
            export_path=export,
            argv=[
                str(exporter),
                "--model_path",
                str(checkpoint_parent),
                "--export_path",
                str(export),
            ],
            started_at="2026-08-23T20:00:00+00:00",
            finished_at="2026-08-23T20:01:00+00:00",
        )

    publish_b_exporter_invocation(
        receipt,
        manifest,
        job_id="42",
        exporter_path=exporter,
        checkpoint_path=checkpoint,
        export_path=export,
        argv=[
            str(exporter),
            "--model_path",
            str(checkpoint),
            "--export_path",
            str(export),
        ],
        started_at="2026-08-23T20:00:00+00:00",
        finished_at="2026-08-23T20:01:00+00:00",
    )
    payload = json.loads(receipt.read_bytes())
    assert payload["model_path"] == str(checkpoint)
    assert payload["argv"][2] == str(checkpoint)
    with pytest.raises(FileExistsError):
        publish_b_exporter_invocation(
            receipt,
            manifest,
            job_id="42",
            exporter_path=exporter,
            checkpoint_path=checkpoint,
            export_path=export,
            argv=[
                str(exporter),
                "--model_path",
                str(checkpoint),
                "--export_path",
                str(export),
            ],
            started_at="2026-08-23T20:00:00+00:00",
            finished_at="2026-08-23T20:01:00+00:00",
        )


def test_last_finite_loss_requires_the_exact_final_step() -> None:
    """A later unstepped loss cannot impersonate the step-200 training result."""
    state = {
        "global_step": 200,
        "log_history": [
            {"step": 200, "loss": 1.25},
            {"step": 199, "loss": 0.01},
            {"loss": 0.001},
        ],
    }

    assert b_manifest_module._last_finite_training_loss(state, expected_step=200) == 1.25
    with pytest.raises(ValueError, match="step 200"):
        b_manifest_module._last_finite_training_loss(
            {"global_step": 200, "log_history": [{"step": 199, "loss": 0.01}]},
            expected_step=200,
        )


def test_canary_runner_and_submitter_enforce_bounded_evidence_contract() -> None:
    """The launch surface chains CPU build to a bounded B-preparation canary."""
    root = Path(__file__).resolve().parents[1] / "common/specdec"
    runner = (root / "run_qwen4b_b_canary.sbatch").read_text()
    submitter = (root / "submit_qwen4b_b_canary.sh").read_text()

    assert "#SBATCH --nodes=16" in runner
    assert "#SBATCH --gpus-per-node=4" in runner
    assert "--cpus-per-task=96" in runner
    assert "train_eagle_streaming.sh" in runner
    assert "dflash.yaml" in runner
    assert "modelopt_recipes/general/speculative_decoding/dflash.yaml" in runner
    assert 'data.data_path="$node_root/input/corpus"' in runner
    assert "training.training_seq_len=4096" in runner
    assert "dflash.dflash_block_size=8" in runner
    assert "dflash.dflash_architecture_config.num_hidden_layers=5" in runner
    assert "dflash.dflash_architecture_config.num_attention_heads=32" in runner
    assert "dflash.dflash_architecture_config.num_key_value_heads=8" in runner
    assert "dflash.dflash_architecture_config.head_dim=128" in runner
    assert "dflash.dflash_architecture_config.intermediate_size=9728" in runner
    assert "dflash.dflash_mask_token_id=151669" in runner
    assert "model.use_fake_base_for_offline=true" in runner
    assert "training.save_strategy=steps" in runner
    assert "training.save_steps=200" in runner
    assert "training.save_total_limit=1" in runner
    assert 'cd "$LAUNCHER_ROOT"' in runner
    assert 'node_root="${SLURM_TMPDIR}/qwen4b-b-${SLURM_JOB_ID}"' in runner
    assert "sha256sum" in runner
    assert "B canary output already exists" in runner
    assert 'printf \'%s\\n\' "$SLURM_JOB_ID" >"$job_owner"' in runner
    assert "model.draft_model_type" not in runner
    assert "model.draft_block_size" not in runner
    assert "data.dataset=" not in runner
    assert "data.max_length=" not in runner
    assert "B_CANARY_SERVE_COMMAND" not in runner
    assert "B_CANARY_TRAIN_COMMAND" not in runner
    assert "publish_b_canary_evidence" in runner
    assert "run_qwen4b_b_canary_eval.sh" in runner
    evaluator = (root / "run_qwen4b_b_canary_eval.sh").read_text()
    assert '"checkpoint_reloaded_by_exporter": True' in evaluator
    assert 'export EXPORT_PATH="$B_OUTPUT_ROOT/control/parent-export"' in runner
    assert '--model_path "$4" --export_path "$5"' in runner
    assert '"$B_CANARY_CHECKPOINT/checkpoint-200" "$B_CANARY_EXPORT"' in runner
    assert "publish_b_exporter_invocation" in runner
    assert "B_CANARY_EXPORTER_RECEIPT" in evaluator
    assert '"exporter_invocation_sha256"' in evaluator
    assert '"checkpoint_sha256": _directory_sha256(checkpoint_path)' in evaluator
    assert "validate_wandb_runtime" in runner
    assert "validate_wandb_runtime" in b_manifest_module.__dict__
    assert "qwen4b-b-$SLURM_JOB_ID" not in runner
    assert "--wandb-netrc" in submitter
    assert "--wandb-durable-root" in submitter
    assert "WANDB_NETRC_PATH=/run/secrets/wandb.netrc" in submitter
    assert ":/run/secrets/wandb.netrc:ro" in submitter
    assert "publish_b_export_evaluation" not in runner
    assert "sbatch --test-only" in submitter
    assert "sbatch --parsable" in submitter
    assert "run_qwen4b_b_builder.sbatch" in submitter
    assert "--submit-prep" in submitter
    assert "--submit-canary-only" in submitter
    assert "--a-authorization-receipt" in submitter
    assert "--a-authorization-receipt-sha256" in submitter
    assert "B_CANARY_BUILD_RECEIPT" in runner
    assert "validate_bound_artifacts" in runner
    assert "container_path=Path(sys.argv[12])" in runner
    assert "--test-only" in submitter
    assert "nemotron_sw_post" in submitter
    assert "nemotron_n4_post" in submitter
    assert '"required_training_order": "A-repair-first"' in submitter
    assert '"b_preparation_only": sys.argv[3] not in {' in submitter
    supervisor = (
        Path(__file__).resolve().parents[1] / "common/eagle3/train_eagle_streaming.sh"
    ).read_text()
    assert "died early" in supervisor
    assert 'kill "$pid"' in supervisor
    assert 'wait "$pid"' in supervisor


def test_b_runner_dotlist_parses_against_actual_dflash_recipe() -> None:
    """The production B overrides parse with the checked-in DFlash YAML types."""
    from omegaconf import OmegaConf  # pyright: ignore[reportMissingImports]

    recipe = Path(__file__).resolve().parents[3] / (
        "modelopt_recipes/general/speculative_decoding/dflash.yaml"
    )
    config = OmegaConf.load(recipe)
    overrides = OmegaConf.from_dotlist(
        [
            "model.model_name_or_path=/immutable/model",
            "model.use_fake_base_for_offline=true",
            "data.data_path=/job/input/corpus",
            "training.output_dir=/job/checkpoint",
            "training.max_steps=200",
            "training.save_strategy=steps",
            "training.save_steps=200",
            "training.save_total_limit=1",
            "training.training_seq_len=4096",
            "training.answer_only_loss=true",
            "training.seed=42",
            "training.per_device_train_batch_size=4",
            "training.gradient_accumulation_steps=4",
            "training.report_to=wandb",
            "training.run_name=q4b-b-test",
            "dflash.dflash_block_size=8",
            "dflash.dflash_num_anchors=512",
            "dflash.dflash_mask_token_id=151669",
            "dflash.dflash_architecture_config.num_hidden_layers=5",
            "dflash.dflash_architecture_config.num_attention_heads=32",
            "dflash.dflash_architecture_config.num_key_value_heads=8",
            "dflash.dflash_architecture_config.head_dim=128",
            "dflash.dflash_architecture_config.intermediate_size=9728",
        ]
    )
    parsed = OmegaConf.merge(config, overrides)

    assert parsed.model.use_fake_base_for_offline is True
    assert parsed.training.training_seq_len == 4096
    assert parsed.training.max_steps == parsed.training.save_steps == 200
    assert parsed.dflash.dflash_mask_token_id == 151669
    assert parsed.dflash.dflash_architecture_config.intermediate_size == 9728


def test_submit_prep_schedules_only_cpu_builder(tmp_path: Path) -> None:
    """B preparation is allowed before A, but cannot schedule a GPU canary."""
    repo_root = Path(__file__).resolve().parents[3]
    submitter = repo_root / "tools/launcher/common/specdec/submit_qwen4b_b_canary.sh"
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    calls = tmp_path / "sbatch.calls"
    counter = tmp_path / "counter"
    fake_sbatch = fake_bin / "sbatch"
    fake_sbatch.write_text(
        "#!/usr/bin/env bash\n"
        'printf "%s\\n" "$*" >> "$SBATCH_CALLS"\n'
        'if [[ "$*" == *"--parsable"* ]]; then '
        'n=$(($(cat "$SBATCH_COUNTER" 2>/dev/null || echo 700)+1)); '
        'printf "%s" "$n" > "$SBATCH_COUNTER"; printf "%s\\n" "$n"; fi\n'
    )
    fake_sbatch.chmod(0o755)
    task9_view = tmp_path / "TASK9.json"
    task9_view.write_text("{}\n")
    task9_view_sha = sha256(task9_view.read_bytes()).hexdigest()
    task8 = tmp_path / "task8"
    task8.mkdir()
    wandb_netrc = tmp_path / "wandb.netrc"
    wandb_netrc.write_text("machine api.wandb.ai\n")
    receipt = tmp_path / "SUBMISSION.json"
    environment = {
        **os.environ,
        "PATH": f"{fake_bin}:{os.environ['PATH']}",
        "SBATCH_CALLS": str(calls),
        "SBATCH_COUNTER": str(counter),
    }
    result = subprocess.run(
        [
            "bash",
            str(submitter),
            "--submit-prep",
            "--account",
            "nemotron_sw_post",
            "--repo-root",
            str(repo_root),
            "--source-commit",
            "a" * 40,
            "--task9-view",
            str(task9_view),
            "--task9-view-sha256",
            task9_view_sha,
            "--task8-publication",
            str(task8),
            "--task8-publication-sha256",
            "b" * 64,
            "--task9-selection-receipt-sha256",
            "c" * 64,
            "--build-root",
            str(tmp_path / "build"),
            "--readiness",
            str(tmp_path / "readiness.json"),
            "--manifest",
            str(tmp_path / "manifest.json"),
            "--evidence",
            str(tmp_path / "evidence.json"),
            "--receipt",
            str(receipt),
            "--target-path",
            str(tmp_path / "target"),
            "--target-revision",
            "d" * 40,
            "--container-image",
            str(tmp_path / "container.sqsh"),
            "--container-sha256",
            "e" * 64,
            "--modelopt-runtime",
            str(tmp_path / "runtime"),
            "--output-root",
            str(tmp_path / "output"),
            "--wandb-netrc",
            str(wandb_netrc),
            "--wandb-durable-root",
            "/lustre/q4b-tests/wandb",
        ],
        check=False,
        capture_output=True,
        text=True,
        env=environment,
    )

    assert result.returncode == 0, result.stderr
    submitted = calls.read_text().splitlines()
    assert len(submitted) == 2
    assert "--partition=cpu_datamover" in submitted[0]
    assert "run_qwen4b_b_builder.sbatch" in submitted[0]
    assert "--test-only" in submitted[0]
    assert "--parsable" in submitted[1]
    assert "--partition=cpu_datamover" in submitted[1]
    assert "run_qwen4b_b_builder.sbatch" in submitted[1]
    payload = json.loads(receipt.read_bytes())
    assert payload["builder_job_id"] == "701"
    assert payload["canary_job_id"] is None
    assert payload["required_training_order"] == "A-repair-first"
    assert payload["b_preparation_only"] is True


@pytest.mark.parametrize("mode", ["--submit-canary", "--submit-canary-only"])
def test_submit_canary_rejects_missing_a_authorization_before_sbatch(
    tmp_path: Path, mode: str
) -> None:
    """The A-repair trust root must be authenticated before any scheduler call."""
    repo_root = Path(__file__).resolve().parents[3]
    submitter = repo_root / "tools/launcher/common/specdec/submit_qwen4b_b_canary.sh"
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    calls = tmp_path / "sbatch.calls"
    fake_sbatch = fake_bin / "sbatch"
    fake_sbatch.write_text('#!/usr/bin/env bash\nprintf "%s\\n" "$*" >> "$SBATCH_CALLS"\n')
    fake_sbatch.chmod(0o755)
    task9_view = tmp_path / "TASK9.json"
    task9_view.write_text("{}\n")
    task9_view_sha = sha256(task9_view.read_bytes()).hexdigest()
    task8 = tmp_path / "task8"
    task8.mkdir()
    wandb_netrc = tmp_path / "wandb.netrc"
    wandb_netrc.write_text("machine api.wandb.ai\n")
    result = subprocess.run(
        [
            "bash",
            str(submitter),
            mode,
            "--account",
            "nemotron_sw_post",
            "--repo-root",
            str(repo_root),
            "--source-commit",
            "a" * 40,
            "--task9-view",
            str(task9_view),
            "--task9-view-sha256",
            task9_view_sha,
            "--task8-publication",
            str(task8),
            "--task8-publication-sha256",
            "b" * 64,
            "--task9-selection-receipt-sha256",
            "c" * 64,
            "--build-root",
            str(tmp_path / "build"),
            "--readiness",
            str(tmp_path / "readiness.json"),
            "--manifest",
            str(tmp_path / "manifest.json"),
            "--evidence",
            str(tmp_path / "evidence.json"),
            "--receipt",
            str(tmp_path / "submission.json"),
            "--target-path",
            str(tmp_path / "target"),
            "--target-revision",
            "d" * 40,
            "--container-image",
            str(tmp_path / "container.sqsh"),
            "--container-sha256",
            "e" * 64,
            "--modelopt-runtime",
            str(tmp_path / "runtime"),
            "--output-root",
            str(tmp_path / "output"),
            "--wandb-netrc",
            str(wandb_netrc),
            "--wandb-durable-root",
            "/lustre/q4b-tests/wandb",
        ],
        check=False,
        capture_output=True,
        text=True,
        env={**os.environ, "PATH": f"{fake_bin}:{os.environ['PATH']}", "SBATCH_CALLS": str(calls)},
    )

    assert result.returncode != 0
    assert not calls.exists()


def test_sacct_observation_requires_completed_canonical_a_parent(tmp_path: Path) -> None:
    """B re-observes the exact A submit/completion/controller chain via live sacct."""
    from common.specdec.qwen4b_a_canary_manifest import (
        finalize_a_authorization,
        publish_a_submission_receipt,
    )

    def write_self_hashed(path: Path, body: dict[str, object]) -> None:
        body["receipt_sha256"] = sha256(
            json.dumps(body, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        path.write_text(json.dumps(body, sort_keys=True, separators=(",", ":")) + "\n")

    submission = tmp_path / "A_SUBMISSION.json"
    publish_a_submission_receipt(
        submission,
        source_commit="a" * 40,
        builder_job_id="4241",
        gpu_job_id="4242",
        account="nemotron_n4_post",
        job_comment="q4-a-repair",
        stdout_path="/logs/q4-a-4242.out",
        manifest_path=(tmp_path / "A_MANIFEST.json").resolve(),
    )
    completion = tmp_path / "A_JOB_COMPLETION.json"
    write_self_hashed(
        completion,
        {
            "schema_version": 1,
            "producer": "qwen4b-a-repair-job-completion-v1",
            "slurm_job_id": "4242",
            "slurm_account": "nemotron_n4_post",
            "slurm_job_name": "q4b-a-repair-canary",
            "slurm_job_comment": "q4-a-repair",
            "slurm_output_path": "/logs/q4-a-4242.out",
            "submission_receipt_path": str(submission),
            "submission_receipt_sha256": sha256(submission.read_bytes()).hexdigest(),
            "started_at": "2026-08-23T20:01:00+00:00",
            "finished_at": "2026-08-23T20:19:00+00:00",
        },
    )
    authorization = tmp_path / "A_AUTHORIZATION.json"
    controller_observation = tmp_path / "A_CONTROLLER_OBSERVATION.json"
    row = (
        "4242|COMPLETED|0:0|nemotron_n4_post|q4b-a-repair-canary|"
        "2026-08-23T20:00:00+00:00|2026-08-23T20:20:00+00:00|"
        "q4-a-repair|/logs/q4-a-4242.out\n"
    )
    finalize_a_authorization(
        authorization,
        controller_observation,
        submission_receipt_path=submission,
        job_completion_path=completion,
        sacct_output=row,
    )
    authorization_sha = sha256(authorization.read_bytes()).hexdigest()
    observation = tmp_path / "A_SCHEDULER.json"

    observation_sha = publish_a_scheduler_observation(
        observation,
        authorization_path=authorization,
        authorization_sha256=authorization_sha,
        sacct_output=row,
    )

    validate_a_scheduler_observation(
        observation,
        observation_sha,
        authorization_path=authorization,
        authorization_sha256=authorization_sha,
    )
    with pytest.raises(FileExistsError):
        publish_a_scheduler_observation(
            observation,
            authorization_path=authorization,
            authorization_sha256=authorization_sha,
            sacct_output=row,
        )
    failed = row.replace("COMPLETED|0:0", "FAILED|1:0")
    with pytest.raises(ValueError, match="successful canonical job"):
        publish_a_scheduler_observation(
            tmp_path / "FAILED.json",
            authorization_path=authorization,
            authorization_sha256=authorization_sha,
            sacct_output=failed,
        )
    forged = json.loads(observation.read_bytes())
    forged["comment"] = "forged"
    forged["receipt_sha256"] = sha256(
        json.dumps(
            {key: value for key, value in forged.items() if key != "receipt_sha256"},
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()
    forged_path = tmp_path / "FORGED_SCHEDULER.json"
    forged_path.write_text(json.dumps(forged, sort_keys=True, separators=(",", ":")) + "\n")
    with pytest.raises(ValueError, match="semantics"):
        validate_a_scheduler_observation(
            forged_path,
            sha256(forged_path.read_bytes()).hexdigest(),
            authorization_path=authorization,
            authorization_sha256=authorization_sha,
        )
