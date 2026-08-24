# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Contracts for the Qwen3-4B one-pass PTV2 comparison."""

from __future__ import annotations

import json
import sys
from collections import Counter
from dataclasses import replace
from hashlib import sha256
from pathlib import Path
from types import MappingProxyType, SimpleNamespace

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[3]
MODULE_DIR = ROOT / "examples/dataset"
POLICY = MODULE_DIR / "qwen3_4b_ptv2_study.yaml"

sys.path.insert(0, str(MODULE_DIR))
try:
    import qwen3_4b_ptv2_study as study_module
    from bprime_cd_policy import ArmPolicy, PromptCell, PromptPolicy
    from build_specdec_inventory import (
        APPROVED_PTV2_ALLOWLIST_SHA256,
        CandidateCell,
        CandidateInventory,
        CandidatePrompt,
        ExclusionProof,
        candidate_inventory_sha256,
        make_exclusion_receipt,
    )
    from qwen3_4b_ptv2_study import (
        PTV2StudyError,
        PTV2StudyRecoveryError,
        PTV2StudySourceRow,
        iter_ptv2_staged_source_rows,
        iter_ptv2_study_occurrences,
        load_ptv2_study_policy,
        main,
        select_a_repair_view,
        select_authenticated_b_balanced_view,
        select_ptv2_b_balanced_view,
        select_ptv2_study_views,
        write_ptv2_selection_receipt,
        write_task9_balanced_view_json,
    )
    from select_bprime_cd_prompts import (
        BPrimePromptViewBundle,
        PromptView,
        SelectedPrompt,
        publish_bprime_prompt_view_bundle,
        publish_prompt_view_bundle,
        select_bprime_prompt_view,
    )
    from specdec_corpus_contracts import canonical_json
    from specdec_identity import prompt_uuid
    from stage_ptv23_sources import (
        SourceFile,
        SourceIdentity,
        SourceInventory,
        SourceManifestError,
        load_source_inventory,
        stage_source_inventory,
    )
finally:
    sys.path.pop(0)


def _digest(value: str) -> str:
    return sha256(value.encode("utf-8")).hexdigest()


def _row(cell: str, source_row: int, prompt: str, *, language: str = "") -> PTV2StudySourceRow:
    conversation = f'{{"messages":[{{"content":"{prompt}","role":"user"}}]}}'
    completion = f'{{"content":"answer-{prompt}","role":"assistant"}}'
    return PTV2StudySourceRow(
        prompt_uuid=_digest(f"prompt:{prompt}"),
        source_identity_sha256=_digest(f"source:{cell}"),
        source_row=source_row,
        cell=cell,
        canonical_conversation=conversation,
        assistant_response=completion,
        language=language,
    )


def _scaled_policy():
    policy = load_ptv2_study_policy(POLICY)
    return replace(
        policy,
        historical_occurrences=3,
        repair_complement_occurrences={"stem": 1, "de": 0, "ja": 1, "es": 0, "fr": 0, "it": 0},
        balanced_occurrences={"math": 2, "code": 3, "stem": 2, "chat": 2, "multilingual": 2},
        segment_occurrences=(3, 2),
        segment_steps=(1, 1),
        cumulative_steps=(1, 2),
        segment_final_valid_occurrences=(3, 2),
    )


def _fixture_policy():
    return replace(load_ptv2_study_policy(POLICY), ptv2_revision="a" * 40)


def _write_authenticated_staged_parquet(tmp_path: Path) -> tuple[Path, Path]:
    pa = pytest.importorskip("pyarrow")
    pq = pytest.importorskip("pyarrow.parquet")
    revision = "a" * 40
    source_path = (
        tmp_path / "source-cache" / "nvidia/PTV2Fixture" / revision / "data/declared.parquet"
    )
    source_path.parent.mkdir(parents=True)
    messages = [
        [
            {"role": "user", "content": "first"},
            {"role": "assistant", "content": "answer-first"},
        ],
        [
            {"role": "user", "content": "second"},
            {"role": "assistant", "content": "answer-second"},
        ],
    ]
    tools = [[{"type": "function", "function": {"name": "tool"}}] for _ in messages]
    pq.write_table(
        pa.table(
            {
                "messages": [json.dumps(value) for value in messages],
                "tools": [json.dumps(value) for value in tools],
            }
        ),
        source_path,
    )
    data = source_path.read_bytes()
    manifest = tmp_path / "SOURCE_PLAN.json"
    manifest.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "name": "ptv2-fixture",
                "sources": [
                    {
                        "repository_id": "nvidia/PTV2Fixture",
                        "configuration": "default",
                        "split": "train",
                        "revision": revision,
                        "license_expression": "CC-BY-4.0",
                        "approved_use": True,
                        "cell": "math",
                        "lane": "target-synth",
                        "files": [
                            {
                                "path": "data/declared.parquet",
                                "bytes": len(data),
                                "sha256": sha256(data).hexdigest(),
                            }
                        ],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    inventory = load_source_inventory(manifest)
    staged = stage_source_inventory(
        inventory,
        durable_root=tmp_path / "durable",
        scratch_root=tmp_path / "scratch",
        local_source_root=tmp_path / "source-cache",
    )
    assert staged.staged_root is not None
    receipt = staged.staged_root / "SOURCE_INVENTORY.json"
    staged_file = (
        staged.staged_root / "sources/nvidia/PTV2Fixture" / revision / "data/declared.parquet"
    )
    return receipt, staged_file


def _genuine_scaled_task5_bundle() -> BPrimePromptViewBundle:
    approved_ptv2_revision = "5c89e01dd720ae0f4058445ed49c5fb68a03c76e"
    b_cells = {name: PromptCell(1) for name in ("stem", "japanese", "spanish", "french", "italian")}
    non_agentic = {
        name: PromptCell(1)
        for name in ("math", "code", "stem-science", "multilingual", "instruction-chat")
    }
    task5_policy = PromptPolicy(
        1,
        20260822,
        1,
        1,
        MappingProxyType(
            {
                "B-prime": ArmPolicy(5, MappingProxyType(b_cells), MappingProxyType({})),
                "C": ArmPolicy(
                    6,
                    MappingProxyType({**non_agentic, "swe-agentic-tool": PromptCell(1)}),
                    MappingProxyType({}),
                ),
                "D": ArmPolicy(
                    8,
                    MappingProxyType({**non_agentic, "swe-agentic-tool": PromptCell(3)}),
                    MappingProxyType(
                        {
                            "agentless-swe": 1,
                            "interactive-swe-replay": 1,
                            "generic-tool-replay": 1,
                        }
                    ),
                ),
            }
        ),
        (256_000_000, 1_000_000_000),
        4096,
        32768,
        frozenset({"en", "ja", "es", "fr", "it"}),
        frozenset({"de"}),
        "a" * 64,
    )
    specifications = (
        ("stem-science", "target-synth", "en", "ptv2", "stem"),
        ("multilingual", "target-synth", "ja", "ptv2", "multilingual_ja"),
        ("multilingual", "target-synth", "es", "ptv2", "multilingual_es"),
        ("multilingual", "target-synth", "fr", "ptv2", "multilingual_fr"),
        ("multilingual", "target-synth", "it", "ptv2", "multilingual_it"),
        ("math", "target-synth", "en", "ptv3", "math"),
        ("code", "target-synth", "en", "ptv3", "code"),
        ("stem-science", "target-synth", "en", "ptv3", "stem"),
        ("multilingual", "target-synth", "ja", "ptv3", "multilingual"),
        ("instruction-chat", "target-synth", "en", "ptv3", "chat"),
        ("swe-agentic-tool", "agentless-swe", "en", "ptv3", "agentless"),
        ("swe-agentic-tool", "interactive-swe-replay", "en", "ptv3", "interactive"),
        ("swe-agentic-tool", "generic-tool-replay", "en", "ptv3", "generic"),
    )
    candidates = []
    for ordinal, (domain, lane, language, family, split) in enumerate(specifications):
        canonical = canonical_json(
            {"messages": [{"role": "user", "content": f"task5-{ordinal}"}], "tools": []}
        )
        candidates.append(
            CandidatePrompt(
                sha256(canonical).hexdigest(),
                canonical,
                f"fixture/{ordinal}",
                (approved_ptv2_revision if family == "ptv2" else "9" * 40),
                sha256(f"file-{ordinal}".encode()).hexdigest(),
                ordinal,
                domain,
                language,
                lane,
                "le4k",
                2,
                "c" * 64,
                f"data/{ordinal}.jsonl",
                (1, 2),
                "d" * 64,
                lane in {"interactive-swe-replay", "generic-tool-replay"},
                family,
                (
                    "nvidia/Nemotron-Post-Training-Dataset-v2"
                    if family == "ptv2"
                    else "fixture/source"
                ),
                "default",
                split,
            )
        )
    candidates = [row for row in candidates if row.source_family == "ptv2"]
    task3_manifest = b"authenticated-task3-ptv2-fixture"
    task3_sha256 = sha256(task3_manifest).hexdigest()
    candidates = [replace(row, source_manifest_sha256=task3_sha256) for row in candidates]
    capacity = Counter(
        CandidateCell(row.domain, row.lane, row.language, row.context_bucket) for row in candidates
    )
    baseline_sha256 = "1" * 64
    held_out_sha256 = "2" * 64
    inventory = CandidateInventory(
        tuple(candidates),
        MappingProxyType(dict(capacity)),
        MappingProxyType({}),
        "0" * 64,
        ExclusionProof(baseline_sha256, "3" * 64, 0, 0),
        ExclusionProof(held_out_sha256, "4" * 64, 0, 0),
        approved_ptv2_revision,
        APPROVED_PTV2_ALLOWLIST_SHA256,
    )
    inventory = replace(inventory, inventory_sha256=candidate_inventory_sha256(inventory))
    matching = {row.source_split: row for row in candidates}
    source_splits = (
        "chat",
        "code",
        "math",
        "stem",
        "multilingual_ja",
        "multilingual_it",
        "multilingual_de",
        "multilingual_es",
        "multilingual_fr",
    )
    remaining = 201
    sources = []
    for index, split in enumerate(source_splits):
        count = remaining // (len(source_splits) - index)
        remaining -= count
        selected = matching.get(split)
        files = tuple(
            SourceFile(
                selected.source_file_path
                if file_index == 0 and selected
                else f"{split}/{file_index}.parquet",
                1,
                selected.source_file_sha256
                if file_index == 0 and selected
                else sha256(f"{split}/{file_index}".encode()).hexdigest(),
            )
            for file_index in range(count)
        )
        sources.append(
            SourceIdentity(
                "nvidia/Nemotron-Post-Training-Dataset-v2",
                "default",
                split,
                approved_ptv2_revision,
                "NVIDIA Open Model License",
                True,
                split,
                "target-synth",
                files,
            )
        )
    source_inventory = SourceInventory(
        1,
        "ptv2-fixture",
        tuple(sources),
        task3_sha256,
        task3_manifest,
        MappingProxyType({}),
    )
    return select_bprime_prompt_view(
        inventory,
        task5_policy,
        source_inventory=source_inventory,
        baseline_receipt_sha256=baseline_sha256,
        held_out_receipt_sha256=held_out_sha256,
    )


def test_approved_ptv2_study_policy_is_exact() -> None:
    """Changing one-pass arithmetic or source-native response policy must fail."""
    policy = load_ptv2_study_policy(POLICY)

    assert policy.historical_occurrences == 1_300_000
    assert policy.repair_complement_occurrences == {
        "stem": 200_000,
        "ja": 125_000,
        "es": 125_000,
        "fr": 125_000,
        "it": 125_000,
        "de": 0,
    }
    assert policy.balanced_occurrences == {
        "math": 500_000,
        "code": 400_000,
        "stem": 500_000,
        "chat": 400_000,
        "multilingual": 200_000,
    }
    assert policy.segment_occurrences == (1_300_000, 700_000)
    assert policy.segment_steps == (2_540, 1_368)
    assert policy.cumulative_steps == (2_540, 3_908)
    assert policy.segment_final_valid_occurrences == (32, 96)
    assert policy.runtime_screen_tokens == 64_000_000
    assert policy.conditional_scientific_tokens == 256_000_000
    assert policy.assistant_responses == "source-native"
    assert policy.trainer_epochs == 1
    assert policy.global_batch_size == 512
    assert policy.multilingual_occurrences == {
        "de": 40_000,
        "ja": 40_000,
        "es": 40_000,
        "fr": 40_000,
        "it": 40_000,
    }


@pytest.mark.parametrize(
    "replacement",
    [
        ("math: 500000", "math: 0.25"),
        ("scientific_exposures: [one-pass]", "scientific_exposures: [64000000]"),
        ("trainer_epochs: 1", "trainer_epochs: 2"),
        ("occurrences: 1300000", "occurrences: 25391"),
        ("schema_version: 1", "schema_version: 1.0"),
    ],
)
def test_policy_rejects_nonapproved_study_representations(
    tmp_path: Path, replacement: tuple[str, str]
) -> None:
    """Floats, old epoch schedules, and scientific milestones cannot enter the study."""
    source, target = replacement
    invalid = tmp_path / "invalid.yaml"
    invalid.write_text(
        POLICY.read_text(encoding="utf-8").replace(source, target, 1), encoding="utf-8"
    )

    with pytest.raises(PTV2StudyError):
        load_ptv2_study_policy(invalid)


def test_prefix_preserves_source_occurrences_and_natural_duplicates(tmp_path: Path) -> None:
    """UUID deduplication, held-out filtering, or synthesized repeats breaks A-prefix."""
    rows = (
        _row("math", 0, "a"),
        _row("math", 1, "a"),
        _row("code", 2, "b"),
        _row("stem", 3, "c"),
        _row("chat", 4, "d"),
        _row("multilingual", 5, "e"),
    )
    policy = _scaled_policy()

    view = select_ptv2_study_views(
        rows,
        policy=policy,
        output_root=tmp_path / "selection",
        held_out_prompt_uuids={rows[2].prompt_uuid},
    ).a_prefix

    selected = tuple(iter_ptv2_study_occurrences(view))
    assert [(row.source_row, row.reuse_index) for row in selected] == [
        (0, 0),
        (1, 0),
        (2, 0),
        (3, 0),
        (4, 0),
    ]
    assert view.occurrence_count == 5
    assert view.unique_prompt_count == 4
    assert view.natural_duplicate_count == 1
    assert view.constructed_repeat_count == 0
    assert view.held_out_overlap_count == 1
    assert view.index_path.is_file()


def test_a_repair_preserves_history_then_appends_only_the_repair_complement(tmp_path: Path) -> None:
    """A's second segment is not the next source-order tail and excludes German."""
    policy = _scaled_policy()
    history = (
        _row("chat", 0, "history-a"),
        _row("chat", 1, "history-a"),
        _row("code", 2, "history-b"),
    )
    baseline_ids = tuple(row.prompt_uuid for row in history)
    baseline = SimpleNamespace(
        occurrence_count=3,
        occurrence_prompt_ids=baseline_ids,
        occurrence_prompt_ids_sha256=sha256(canonical_json(list(baseline_ids))).hexdigest(),
        unique_prompt_count=2,
    )
    complement = (
        _row("multilingual", 3, "wrong-de", language="de"),
        _row("stem", 4, "stem"),
        _row("multilingual", 5, "ja", language="ja"),
        _row("chat", 6, "wrong-tail"),
    )

    view = select_a_repair_view(
        history, complement, policy=policy, baseline=baseline, output_root=tmp_path
    )

    selected = tuple(iter_ptv2_study_occurrences(view))
    assert [row.source_row for row in selected] == [0, 1, 2, 4, 5]
    assert view.strategy == "A-repair"
    assert view.repair_complement_counts["de"] == 0


def test_balanced_view_repeats_within_cell_without_renormalizing(tmp_path: Path) -> None:
    """A deficient cell must cycle only itself, keeping source-row multiplicity explicit."""
    rows = (
        _row("math", 0, "math"),
        _row("code", 1, "code"),
        _row("stem", 2, "stem"),
        _row("chat", 3, "chat"),
        _row("multilingual", 4, "multi"),
    )
    policy = _scaled_policy()

    bundle = select_ptv2_study_views(rows, policy=policy, output_root=tmp_path / "selection")
    view = bundle.b_balanced
    selected = tuple(iter_ptv2_study_occurrences(view))

    assert view.cell_occurrence_counts == {
        "math": 2,
        "code": 3,
        "stem": 2,
        "chat": 2,
        "multilingual": 2,
    }
    assert view.occurrence_count == 11
    assert view.constructed_repeat_count == 6
    assert view.trainer_epochs == 1
    assert [
        (row.cell, row.source_row, row.reuse_index) for row in selected if row.cell == "code"
    ] == [
        ("code", 1, 0),
        ("code", 1, 1),
        ("code", 1, 2),
    ]


def test_b_balanced_readiness_does_not_require_an_a_prefix(tmp_path: Path) -> None:
    """B must remain buildable when A's historical source-order stream is unavailable."""
    rows = (
        _row("math", 0, "math"),
        _row("code", 1, "code"),
        _row("stem", 2, "stem"),
        _row("chat", 3, "chat"),
        _row("multilingual", 4, "multi"),
    )

    view = select_ptv2_b_balanced_view(
        rows, policy=_scaled_policy(), output_root=tmp_path / "b-only"
    )

    assert view.strategy == "B-balanced"
    assert view.occurrence_count == 11


def test_authenticated_b_selection_rejects_untyped_exclusion_root(tmp_path: Path) -> None:
    """Production B cannot substitute a caller-provided digest/list for ExclusionIndex."""
    with pytest.raises(PTV2StudyError, match="requires an ExclusionIndex"):
        select_authenticated_b_balanced_view(
            tmp_path / "SOURCE_INVENTORY.json",
            policy=load_ptv2_study_policy(POLICY),
            exclusions=object(),  # type: ignore[arg-type]
        )


def test_task10_balanced_projection_is_a_stable_b_only_json_contract(tmp_path: Path) -> None:
    """Task9 publishes the documented projection consumed by Task10 without A fields."""
    policy = load_ptv2_study_policy(POLICY)
    shards = tuple(f"shard-{index:03d}.jsonl" for index in range(201))
    view = SimpleNamespace(
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
        trainer_epochs=1,
    )
    projection = tmp_path / "TASK9_B_BALANCED.json"

    write_task9_balanced_view_json(
        projection,
        view,
        policy=policy,
        shard_root=tmp_path / "shards",
        declared_shards=shards,
        publication_sha256="1" * 64,
        destination_sha256="2" * 64,
        tokenizer_sha256="3" * 64,
        chat_template_sha256="4" * 64,
        assistant_loss_mask_sha256="5" * 64,
    )

    payload = json.loads(projection.read_bytes())
    assert payload["strategy"] == "B-balanced"
    assert payload["declared_shards"] == list(shards)
    assert payload["segment_steps"] == [2540, 1368]
    assert "a_repair" not in payload


def test_schema_v3_selection_writer_recomputes_identity_and_streams_source_rows(
    tmp_path: Path,
) -> None:
    """The production Task9 receipt is a self-contained, semantically replayable input."""
    rows = (
        _row("math", 0, "math"),
        _row("code", 1, "code-a"),
        _row("code", 2, "code-b"),
        _row("stem", 3, "stem"),
        _row("chat", 4, "chat"),
        _row("multilingual", 5, "multi"),
    )
    policy = _scaled_policy()
    trust_roots = {
        "source_inventory_sha256": "1" * 64,
        "held_out_receipt_sha256": "3" * 64,
    }
    view = select_ptv2_b_balanced_view(
        rows,
        policy=policy,
        output_root=tmp_path / "index",
        trust_roots=trust_roots,
    )
    policy_path = tmp_path / "policy.yaml"
    policy_path.write_bytes(POLICY.read_bytes())

    with pytest.raises(PTV2StudyError, match="trust-root preimage"):
        write_ptv2_selection_receipt(
            tmp_path / "forged-receipt",
            view,
            policy=policy,
            policy_path=policy_path,
            source_inventory_sha256="9" * 64,
            held_out_receipt_sha256="3" * 64,
        )

    with pytest.raises(PTV2StudyError, match="must not carry baseline"):
        write_ptv2_selection_receipt(
            tmp_path / "baseline-forged-receipt",
            view,
            policy=policy,
            policy_path=policy_path,
            source_inventory_sha256="1" * 64,
            baseline_receipt_sha256="2" * 64,
            held_out_receipt_sha256="3" * 64,
        )

    receipt = write_ptv2_selection_receipt(
        tmp_path / "receipt",
        view,
        policy=policy,
        policy_path=policy_path,
        source_inventory_sha256="1" * 64,
        held_out_receipt_sha256="3" * 64,
    )

    payload = json.loads(receipt.read_bytes())
    assert payload["schema_version"] == 3
    assert payload["trust_roots"] == trust_roots
    assert "baseline_receipt_sha256" not in payload
    assert "complement_selection_sha256" not in payload
    assert (
        payload["selection_sha256"]
        == sha256(canonical_json(payload["selection_identity"])).hexdigest()
    )
    assert payload["occurrence_count"] == view.occurrence_count
    assert (receipt.parent / payload["index"]["path"]).is_file()
    shard = receipt.parent / payload["shards"][0]["path"]
    assert len(shard.read_text(encoding="utf-8").splitlines()) == view.occurrence_count
    sys.path.insert(0, str(MODULE_DIR))
    try:
        import specdec_publication as publication
    finally:
        sys.path.pop(0)
    descriptors = publication._role_file_descriptors("selection", payload)
    files = [
        (
            item["path"],
            receipt.parent / item["path"],
            item["bytes"],
            item["sha256"],
        )
        for item in descriptors
    ]
    publication._validate_ptv2_selection_policy(payload, files)


def test_a_repair_schema_v3_receipt_replays_strategy_specific_roots(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A-repair publishes baseline/complement roots and passes Task8 semantic replay."""
    inventory_receipt, _ = _write_authenticated_staged_parquet(tmp_path)
    monkeypatch.setattr(study_module, "_DECLARED_PTV2_PARQUET_SHARDS", 1)
    inventory = load_source_inventory(inventory_receipt)
    history = (
        _row("chat", 0, "history-a"),
        _row("chat", 1, "history-a"),
        _row("code", 2, "history-b"),
    )
    baseline_ids = tuple(row.prompt_uuid for row in history)
    baseline = SimpleNamespace(
        occurrence_count=3,
        occurrence_prompt_ids=baseline_ids,
        occurrence_prompt_ids_sha256=sha256(canonical_json(list(baseline_ids))).hexdigest(),
        unique_prompt_count=2,
    )
    complement = (
        _row("stem", 3, "stem"),
        _row("multilingual", 4, "ja", language="ja"),
    )
    policy_document = {
        "repair": {
            "historical": {"occurrences": 3},
            "complement": {"stem": 1, "ja": 1, "es": 0, "fr": 0, "it": 0, "de": 0},
        },
        "balanced": {"multilingual_occurrences": {"de": 0, "ja": 1, "es": 0, "fr": 0, "it": 0}},
    }
    policy_path = tmp_path / "scaled-policy.yaml"
    policy_path.write_text(yaml.safe_dump(policy_document), encoding="utf-8")
    policy = replace(
        _scaled_policy(),
        segment_occurrences=(1_300_000, 700_000),
        policy_sha256=sha256(canonical_json(policy_document)).hexdigest(),
    )
    baseline_receipt = make_exclusion_receipt("baseline", tuple(set(baseline_ids)))
    held_out_receipt = make_exclusion_receipt("held-out", ())
    complement_identity = "4" * 64
    view = select_a_repair_view(
        history,
        complement,
        policy=policy,
        baseline=baseline,
        output_root=tmp_path / "a-index",
        source_inventory=inventory,
        baseline_receipt=baseline_receipt,
        held_out_receipt=held_out_receipt,
        complement_selection_sha256=complement_identity,
    )
    receipt = write_ptv2_selection_receipt(
        tmp_path / "a-receipt",
        view,
        policy=policy,
        policy_path=policy_path,
        source_inventory_sha256=inventory.manifest_sha256,
        held_out_receipt_sha256=held_out_receipt.receipt_sha256,
        baseline_receipt_sha256=baseline_receipt.receipt_sha256,
        complement_selection_sha256=complement_identity,
    )
    payload = json.loads(receipt.read_bytes())

    assert payload["trust_roots"] == {
        "source_inventory_sha256": inventory.manifest_sha256,
        "baseline_receipt_sha256": baseline_receipt.receipt_sha256,
        "held_out_receipt_sha256": held_out_receipt.receipt_sha256,
        "complement_selection_sha256": complement_identity,
    }
    sys.path.insert(0, str(MODULE_DIR))
    try:
        import specdec_publication as publication
    finally:
        sys.path.pop(0)
    descriptors = publication._role_file_descriptors("selection", payload)
    publication._validate_ptv2_selection_policy(
        payload,
        [
            (
                item["path"],
                receipt.parent / item["path"],
                item["bytes"],
                item["sha256"],
            )
            for item in descriptors
        ],
    )


def test_b_index_publication_is_no_replace_and_preserves_prior_receipt(tmp_path: Path) -> None:
    """A retry must not overwrite an immutable B index with a fresh SQLite file."""
    rows = (
        _row("math", 0, "math"),
        _row("code", 1, "code"),
        _row("stem", 2, "stem"),
        _row("chat", 3, "chat"),
        _row("multilingual", 4, "multi"),
    )
    root = tmp_path / "b-only"
    first = select_ptv2_b_balanced_view(rows, policy=_scaled_policy(), output_root=root)
    before = first.index_path.read_bytes()

    with pytest.raises(FileExistsError, match="immutable B index"):
        select_ptv2_b_balanced_view(rows, policy=_scaled_policy(), output_root=root)

    assert first.index_path.read_bytes() == before


def test_selection_receipt_rename_and_parent_fsync_failures_carry_typed_recovery(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Ambiguous install and durability failures expose inode-bound recovery evidence."""
    policy = _scaled_policy()
    trust_roots = {
        "source_inventory_sha256": "1" * 64,
        "held_out_receipt_sha256": "3" * 64,
    }
    rows = tuple(
        _row(cell, index, cell)
        for index, cell in enumerate(("math", "code", "stem", "chat", "multilingual"))
    )
    view = select_ptv2_b_balanced_view(
        rows, policy=policy, output_root=tmp_path / "index", trust_roots=trust_roots
    )
    policy_path = tmp_path / "policy.yaml"
    policy_path.write_bytes(POLICY.read_bytes())
    rename = study_module._rename_no_replace

    def ambiguous_rename(source: Path, destination: Path) -> None:
        rename(source, destination)
        raise OSError("lost rename acknowledgement")

    monkeypatch.setattr(study_module, "_rename_no_replace", ambiguous_rename)
    installed = tmp_path / "installed-after-error"
    with pytest.raises(PTV2StudyRecoveryError) as caught:
        write_ptv2_selection_receipt(
            installed,
            view,
            policy=policy,
            policy_path=policy_path,
            source_inventory_sha256="1" * 64,
            held_out_receipt_sha256="3" * 64,
        )
    state = study_module.ptv2_selection_recovery_state(caught.value)
    assert state.phase is study_module.PTV2SelectionPublicationPhase.RENAME
    assert state.destination_observation.identity == state.expected_partial_identity
    assert state.partial_observation.status == "absent"

    monkeypatch.setattr(study_module, "_rename_no_replace", rename)
    fsync_directory = study_module._fsync_directory
    installed = tmp_path / "installed-before-fsync-error"

    def fail_parent_fsync(path: Path) -> None:
        if path == installed.parent and installed.exists():
            raise OSError("parent fsync acknowledgement lost")
        fsync_directory(path)

    monkeypatch.setattr(study_module, "_fsync_directory", fail_parent_fsync)
    with pytest.raises(PTV2StudyRecoveryError) as caught:
        write_ptv2_selection_receipt(
            installed,
            view,
            policy=policy,
            policy_path=policy_path,
            source_inventory_sha256="1" * 64,
            held_out_receipt_sha256="3" * 64,
        )
    state = study_module.ptv2_selection_recovery_state(caught.value)
    assert state.phase is study_module.PTV2SelectionPublicationPhase.PARENT_FSYNC
    assert state.destination_observation.identity == state.expected_partial_identity


def test_b_index_publication_requires_typed_recovery_for_interrupted_partial(
    tmp_path: Path,
) -> None:
    """An interrupted private SQLite build is never silently reused or removed."""
    root = tmp_path / "b-only"
    root.mkdir()
    partial = root / ".ptv2-b-balanced-index.sqlite3.partial-interrupted"
    partial.write_bytes(b"incomplete")
    rows = tuple(
        _row(cell, index, cell)
        for index, cell in enumerate(("math", "code", "stem", "chat", "multilingual"))
    )

    with pytest.raises(PTV2StudyRecoveryError, match="requires recovery"):
        select_ptv2_b_balanced_view(rows, policy=_scaled_policy(), output_root=root)

    assert partial.read_bytes() == b"incomplete"


def test_balanced_cycle_is_round_robin_and_input_permutation_stable(tmp_path: Path) -> None:
    """Grouping repeated rows or depending on ingestion order breaks B-balanced replay."""
    rows = (
        _row("math", 0, "math"),
        _row("code", 1, "code-a"),
        _row("code", 2, "code-b"),
        _row("stem", 3, "stem"),
        _row("chat", 4, "chat"),
        _row("multilingual", 5, "multi"),
    )
    policy = _scaled_policy()
    first = select_ptv2_study_views(rows, policy=policy, output_root=tmp_path / "first").b_balanced
    second = select_ptv2_study_views(
        tuple(reversed(rows)), policy=policy, output_root=tmp_path / "second"
    ).b_balanced
    code_rows = [row.source_row for row in iter_ptv2_study_occurrences(first) if row.cell == "code"]

    assert code_rows[0] != code_rows[1]
    assert code_rows[0] == code_rows[2]
    assert first.ordered_occurrences_sha256 == second.ordered_occurrences_sha256


def test_selection_rejects_unknown_or_missing_source_native_response(tmp_path: Path) -> None:
    """An invalid cell or response blocks selection rather than substituting a reserve row."""
    policy = _scaled_policy()
    unknown = replace(_row("math", 0, "a"), cell="unknown")
    missing_response = replace(_row("math", 0, "b"), assistant_response="")

    with pytest.raises(PTV2StudyError, match="unknown cell"):
        select_ptv2_study_views((unknown,), policy=policy, output_root=tmp_path / "unknown")
    with pytest.raises(PTV2StudyError, match="source-native"):
        select_ptv2_study_views(
            (missing_response,), policy=policy, output_root=tmp_path / "response"
        )


def test_full_policy_rejects_multilingual_rows_without_an_approved_language(tmp_path: Path) -> None:
    """The 200K multilingual quota cannot borrow an unlabelled source occurrence."""
    policy = load_ptv2_study_policy(POLICY)

    with pytest.raises(PTV2StudyError, match="approved language"):
        select_ptv2_b_balanced_view(
            (_row("multilingual", 0, "unlabelled"),), policy=policy, output_root=tmp_path
        )


def test_b_balanced_multilingual_quota_is_per_language_without_borrowing(tmp_path: Path) -> None:
    """Each approved language must contribute its own ranked/cycled B subcell quota."""
    policy = replace(
        load_ptv2_study_policy(POLICY),
        balanced_occurrences={"math": 1, "code": 1, "stem": 1, "chat": 1, "multilingual": 5},
        multilingual_occurrences={"de": 1, "ja": 1, "es": 1, "fr": 1, "it": 1},
    )
    rows = (
        _row("math", 0, "math"),
        _row("code", 1, "code"),
        _row("stem", 2, "stem"),
        _row("chat", 3, "chat"),
        *(
            _row("multilingual", 10 + index, language, language=language)
            for index, language in enumerate(("de", "ja", "es", "fr", "it"))
        ),
    )

    view = select_ptv2_b_balanced_view(rows, policy=policy, output_root=tmp_path)
    multilingual_rows = [
        occurrence.source_row
        for occurrence in iter_ptv2_study_occurrences(view)
        if occurrence.cell == "multilingual"
    ]

    assert sorted(multilingual_rows) == [10, 11, 12, 13, 14]


def test_staged_inventory_authenticates_declared_bytes_and_physical_row_order(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Task3's published receipt binds the source file, native response, and row order."""
    receipt, staged_file = _write_authenticated_staged_parquet(tmp_path)

    monkeypatch.setattr(study_module, "_DECLARED_PTV2_PARQUET_SHARDS", 1)
    rows = tuple(iter_ptv2_staged_source_rows(receipt, policy=_fixture_policy()))

    assert [(row.source_row, row.cell) for row in rows] == [(0, "math"), (1, "math")]
    assert [json.loads(row.assistant_response)["content"] for row in rows] == [
        "answer-first",
        "answer-second",
    ]
    assert rows[0].prompt_uuid == prompt_uuid(
        [{"role": "user", "content": "first"}], [{"type": "function", "function": {"name": "tool"}}]
    )
    assert json.loads(rows[0].canonical_conversation)["tools"] == [
        {"type": "function", "function": {"name": "tool"}}
    ]
    staged_file.write_bytes(b"mutated payload")
    with pytest.raises(SourceManifestError, match="stale staged file"):
        tuple(iter_ptv2_staged_source_rows(receipt, policy=_fixture_policy()))


def test_staged_inventory_rejects_undeclared_orphan_parquet(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The known multilingual orphan cannot enter B through a physical-directory scan."""
    receipt, staged_file = _write_authenticated_staged_parquet(tmp_path)
    orphan = staged_file.parent / "multilingual-00000-of-00001.parquet"
    orphan.write_bytes(staged_file.read_bytes())

    monkeypatch.setattr(study_module, "_DECLARED_PTV2_PARQUET_SHARDS", 1)
    with pytest.raises(PTV2StudyError, match="undeclared or missing Parquet"):
        tuple(iter_ptv2_staged_source_rows(receipt, policy=_fixture_policy()))


def test_staged_inventory_requires_a_published_task3_receipt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A plan object without Task3's staged trust root is not a B source of truth."""
    source = SourceIdentity(
        repository_id="nvidia/PTV2Fixture",
        configuration="default",
        split="train",
        revision="a" * 40,
        license_expression="CC-BY-4.0",
        approved_use=True,
        cell="math",
        lane="target-synth",
        files=(SourceFile("data/declared.parquet", 1, sha256(b"x").hexdigest()),),
    )
    inventory = SourceInventory(1, "fixture", (source,), "0" * 64, b"{}", {}, None)
    monkeypatch.setattr(study_module, "load_source_inventory", lambda _: inventory)

    with pytest.raises(PTV2StudyError, match="authenticated staged SourceInventory"):
        tuple(iter_ptv2_staged_source_rows(tmp_path / "SOURCE_PLAN.json", policy=_fixture_policy()))


def test_task5_published_complement_joins_the_task3_physical_row_stream(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """C2 accepts the real Task5 schema-v2 artifact only when its row joins Task3."""
    inventory_receipt, _ = _write_authenticated_staged_parquet(tmp_path)
    monkeypatch.setattr(study_module, "_DECLARED_PTV2_PARQUET_SHARDS", 1)
    policy = _fixture_policy()
    physical = next(iter_ptv2_staged_source_rows(inventory_receipt, policy=policy))
    inventory = load_source_inventory(inventory_receipt)
    source = inventory.sources[0]
    policy_sha256 = "1" * 64
    seed = 20260822
    rank_fields = (
        policy_sha256,
        str(seed),
        "math",
        "target-synth",
        "",
        "le4k",
        "fixture",
        source.revision,
        source.files[0].sha256,
        inventory.manifest_sha256,
        source.files[0].path,
        str(physical.source_row),
        physical.prompt_uuid,
    )
    selected = SelectedPrompt(
        physical.prompt_uuid,
        "C",
        "math",
        "target-synth",
        "",
        "le4k",
        "fixture",
        "ptv2",
        source.repository_id,
        source.configuration,
        source.split,
        source.revision,
        source.files[0].sha256,
        inventory.manifest_sha256,
        source.files[0].path,
        physical.source_row,
        0,
        sha256("\0".join(rank_fields).encode()).hexdigest(),
        0,
        "primary",
        canonical_json({"messages": [{"role": "user", "content": "first"}]}).decode(),
    )
    b_prime = PromptView("B-prime", (), (), {}, {}, {}, {}, {}, "d" * 64)
    d_view = PromptView("D", (), (), {}, {}, {}, {}, {}, "d" * 64)
    bundle = SimpleNamespace(
        B_prime=b_prime,
        C=PromptView("C", (selected,), (), {"math": 1}, {"target-synth": 1}, {}, {}, {}, "d" * 64),
        D=d_view,
        policy_sha256=policy_sha256,
        seed=seed,
        source_inventory_sha256=inventory.manifest_sha256,
        baseline_receipt_sha256="2" * 64,
        held_out_receipt_sha256="3" * 64,
        ptv2_revision=source.revision,
        ptv2_allowlist_sha256="4" * 64,
        reserve_numerator=1,
        reserve_denominator=10,
        paired_cd_sha256="5" * 64,
        selection_sha256="6" * 64,
    )
    published = publish_prompt_view_bundle(bundle, tmp_path / "task5", rows_per_shard=1)
    manifest_path = published.manifest_path
    view = study_module.load_prompt_view(
        manifest_path,
        expected_manifest_sha256=sha256(manifest_path.read_bytes()).hexdigest(),
        arm="C",
    )

    with pytest.raises(PTV2StudyError, match="B-prime"):
        tuple(study_module._iter_task5_selected_rows(view, inventory_receipt, policy))


def test_a_repair_authenticates_genuine_task5_bprime_selection_and_arm_proof(
    tmp_path: Path,
) -> None:
    """A genuine Task5 selector/publication is replayed; a forged selection root is rejected."""
    task5_bundle = _genuine_scaled_task5_bundle()
    try:
        published = publish_bprime_prompt_view_bundle(
            task5_bundle, tmp_path / "task5", rows_per_shard=4
        )
    finally:
        task5_bundle.close()
    manifest_path = published.manifest_path
    manifest_sha256 = sha256(manifest_path.read_bytes()).hexdigest()
    view = study_module.load_prompt_view(
        manifest_path,
        expected_manifest_sha256=manifest_sha256,
        arm="B-prime",
    )
    study_policy = replace(
        _fixture_policy(),
        repair_complement_occurrences={
            "stem": 1,
            "ja": 1,
            "es": 1,
            "fr": 1,
            "it": 1,
            "de": 0,
        },
    )

    arm_identity = study_module._authenticate_task5_bprime(
        manifest_path,
        expected_manifest_sha256=manifest_sha256,
        view=view,
        policy=study_policy,
    )

    assert len(arm_identity) == 64
    manifest = json.loads(manifest_path.read_bytes())

    contaminated = json.loads(manifest_path.read_bytes())
    contaminated["arms"]["C"] = contaminated["arms"]["B-prime"]
    root_record = {key: value for key, value in contaminated.items() if key != "root_sha256"}
    contaminated["root_sha256"] = sha256(canonical_json(root_record)).hexdigest()
    manifest_path.write_bytes(canonical_json(contaminated) + b"\n")
    contaminated_sha256 = sha256(manifest_path.read_bytes()).hexdigest()
    contaminated_view = study_module.load_prompt_view(
        manifest_path,
        expected_manifest_sha256=contaminated_sha256,
        arm="B-prime",
    )
    with pytest.raises(PTV2StudyError, match="B-prime-only"):
        study_module._authenticate_task5_bprime(
            manifest_path,
            expected_manifest_sha256=contaminated_sha256,
            view=contaminated_view,
            policy=study_policy,
        )

    manifest_path.write_bytes(canonical_json(manifest) + b"\n")
    forged_source = json.loads(manifest_path.read_bytes())
    forged_source["identity"]["source_inventory_sha256"] = "e" * 64
    root_record = {key: value for key, value in forged_source.items() if key != "root_sha256"}
    forged_source["root_sha256"] = sha256(canonical_json(root_record)).hexdigest()
    manifest_path.write_bytes(canonical_json(forged_source) + b"\n")
    forged_source_sha256 = sha256(manifest_path.read_bytes()).hexdigest()
    forged_source_view = study_module.load_prompt_view(
        manifest_path,
        expected_manifest_sha256=forged_source_sha256,
        arm="B-prime",
    )
    with pytest.raises(PTV2StudyError, match="selection digest"):
        study_module._authenticate_task5_bprime(
            manifest_path,
            expected_manifest_sha256=forged_source_sha256,
            view=forged_source_view,
            policy=study_policy,
        )

    manifest_path.write_bytes(canonical_json(manifest) + b"\n")
    manifest["selection_sha256"] = "f" * 64
    root_record = {key: value for key, value in manifest.items() if key != "root_sha256"}
    manifest["root_sha256"] = sha256(canonical_json(root_record)).hexdigest()
    manifest_path.write_bytes(canonical_json(manifest) + b"\n")
    forged_manifest_sha256 = sha256(manifest_path.read_bytes()).hexdigest()
    forged_view = study_module.load_prompt_view(
        manifest_path,
        expected_manifest_sha256=forged_manifest_sha256,
        arm="B-prime",
    )
    with pytest.raises(PTV2StudyError, match="selection digest"):
        study_module._authenticate_task5_bprime(
            manifest_path,
            expected_manifest_sha256=forged_manifest_sha256,
            view=forged_view,
            policy=study_policy,
        )


def test_b_cli_uses_the_immutable_declared_shard_contract(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The executable B path defaults to the audited 201 declared PTV2 shards."""
    rows = tuple(
        _row(cell, index, cell)
        for index, cell in enumerate(("math", "code", "stem", "chat", "multilingual"))
    )
    seen: dict[str, object] = {}

    def _authenticated(_: Path, *, policy, exclusions, output_root):
        seen["policy"] = policy
        seen["held_out"] = exclusions.held_out
        return select_ptv2_b_balanced_view(rows, policy=policy, output_root=output_root)

    monkeypatch.setattr(study_module, "load_ptv2_study_policy", lambda _: _scaled_policy())
    monkeypatch.setattr(study_module, "select_authenticated_b_balanced_view", _authenticated)
    held_out = tmp_path / "held-out.json"
    held_out.write_text("[]", encoding="utf-8")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "qwen3_4b_ptv2_study.py",
            "--policy",
            str(POLICY),
            "--source-inventory",
            str(tmp_path / "SOURCE_INVENTORY.json"),
            "--output-root",
            str(tmp_path / "out"),
            "--held-out-uuids",
            str(held_out),
        ],
    )

    assert main() == 0
    assert seen == {"policy": _scaled_policy(), "held_out": set()}
