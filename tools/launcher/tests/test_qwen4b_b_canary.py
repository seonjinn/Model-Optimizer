# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Contracts for the B-balanced OCI-HSG canary path."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from dataclasses import replace
from hashlib import sha256
from pathlib import Path

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
    publish_b_canary_evidence,
    publish_b_export_evaluation,
    validate_a_authorization_receipt,
    validate_bound_artifacts,
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
    assert {
        cell: sum(row["domain"] == cell for row in selected) for cell in CANARY_CELL_QUOTAS
    } == CANARY_CELL_QUOTAS
    assert {
        language: sum(
            json.loads(str(row["record_json"]))["language"] == language for row in selected
        )
        for language in CANARY_LANGUAGE_QUOTAS
    } == CANARY_LANGUAGE_QUOTAS


def _task8_row(cell: str, identifier: str, *, language: str = "") -> dict[str, object]:
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


def test_manifest_requires_transitive_builder_identities() -> None:
    """The GPU manifest cannot omit either immutable builder artifact identity."""
    with pytest.raises(TypeError):
        BCanaryManifest(  # pyright: ignore[reportCallIssue]
            readiness_receipt_sha256="a" * 64,
            canary_occurrence_count=102_400,
            topology=_topology(),
            source_commit="f" * 40,
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
    supervisor = tmp_path / "train_eagle_streaming.sh"
    config = tmp_path / "dflash.yaml"
    supervisor.write_text("supervisor\n")
    config.write_text("config\n")
    return BCanaryRuntimeIdentity(
        target_model_id="Qwen/Qwen3-4B",
        target_path=str(tmp_path / "target"),
        target_revision="1" * 40,
        tokenizer_sha256="2" * 64,
        chat_template_sha256="3" * 64,
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
        corpus_arm="B-balanced",
        corpus_path=str(corpus_path),
        output_root=str(tmp_path / "output"),
        wandb_project="sna-qwen3-4b-dataset-study",
        wandb_run_id="q4b-b-test",
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
        "CONTAINER_SHA256": "f" * 64,
        "B_CANARY_OUTPUT_ROOT": str(tmp_path / "output"),
        "B_CANARY_WANDB_RUN_ID": "q4b-b-test",
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


def test_a_authorization_is_caller_pinned_job_evidence(tmp_path: Path) -> None:
    """A authorization recomputes self, file, checkpoint, export, and GPU identities."""
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

    authorization = validate_a_authorization_receipt(
        receipt,
        pinned,
        source_commit="f" * 40,
        target_revision="5" * 40,
        tokenizer_sha256="6" * 64,
        chat_template_sha256="7" * 64,
        container_sha256="8" * 64,
    )
    assert authorization.receipt_sha256 == pinned
    (checkpoint / "trainer_state.json").write_text('{"tampered":true}\n')
    with pytest.raises(ValueError, match="artifact identity"):
        validate_a_authorization_receipt(
            receipt,
            pinned,
            source_commit="f" * 40,
            target_revision="5" * 40,
            tokenizer_sha256="6" * 64,
            chat_template_sha256="7" * 64,
            container_sha256="8" * 64,
        )


def test_b_evidence_is_derived_from_current_job_outputs_and_no_replace(tmp_path: Path) -> None:
    """Caller booleans cannot replace the canonical job-output evidence chain."""
    corpus = tmp_path / "canary.jsonl"
    corpus.write_text("{}\n")
    manifest = _manifest(tmp_path, corpus)
    checkpoint = tmp_path / "checkpoint"
    export = tmp_path / "export"
    checkpoint.mkdir()
    export.mkdir()
    (checkpoint / "trainer_state.json").write_text(
        json.dumps({"global_step": 200, "loss_history": [1.5, 1.25]}) + "\n"
    )
    (export / "config.json").write_text(
        json.dumps({"draft_model_type": "dflash", "draft_block_size": 8}) + "\n"
    )
    (export / "model.safetensors").write_bytes(b"weights")
    evaluation = tmp_path / "evaluation.json"
    publish_b_export_evaluation(evaluation, manifest, job_id="42", export_path=export)
    gpu = tmp_path / "gpu.json"
    gpu.write_text(json.dumps({"slurm_job_id": "42", "active_gpu_ranks": list(range(64))}) + "\n")
    evidence = tmp_path / "B_EVIDENCE.json"

    publish_b_canary_evidence(
        evidence,
        manifest,
        job_id="42",
        checkpoint_path=checkpoint,
        export_path=export,
        evaluation_receipt_path=evaluation,
        gpu_evidence_path=gpu,
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
            evaluation_receipt_path=evaluation,
            gpu_evidence_path=gpu,
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
    assert "B_CANARY_SERVE_COMMAND" not in runner
    assert "B_CANARY_TRAIN_COMMAND" not in runner
    assert "publish_b_canary_evidence" in runner
    assert "sbatch --test-only" in submitter
    assert "sbatch --parsable" in submitter
    assert "run_qwen4b_b_builder.sbatch" in submitter
    assert "--submit-prep" in submitter
    assert "--a-authorization-receipt" in submitter
    assert "--a-authorization-receipt-sha256" in submitter
    assert "B_CANARY_BUILD_RECEIPT" in runner
    assert "validate_bound_artifacts" in runner
    assert "--test-only" in submitter
    assert "nemotron_sw_post" in submitter
    assert "nemotron_n4_post" in submitter
    assert '"required_training_order": "A-repair-first"' in submitter
    assert '"b_preparation_only": sys.argv[3] != "--submit-canary"' in submitter


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


def test_submit_canary_rejects_missing_a_authorization_before_sbatch(tmp_path: Path) -> None:
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
    result = subprocess.run(
        [
            "bash",
            str(submitter),
            "--submit-canary",
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
        ],
        check=False,
        capture_output=True,
        text=True,
        env={**os.environ, "PATH": f"{fake_bin}:{os.environ['PATH']}", "SBATCH_CALLS": str(calls)},
    )

    assert result.returncode != 0
    assert not calls.exists()
