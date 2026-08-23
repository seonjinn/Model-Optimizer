# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Contracts for the Qwen3-4B one-pass PTV2 comparison."""

from __future__ import annotations

import json
import sys
from dataclasses import replace
from hashlib import sha256
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
MODULE_DIR = ROOT / "examples/dataset"
POLICY = MODULE_DIR / "qwen3_4b_ptv2_study.yaml"

sys.path.insert(0, str(MODULE_DIR))
try:
    import qwen3_4b_ptv2_study as study_module
    from qwen3_4b_ptv2_study import (
        PTV2StudyError,
        PTV2StudyRecoveryError,
        PTV2StudySourceRow,
        iter_ptv2_staged_source_rows,
        iter_ptv2_study_occurrences,
        load_ptv2_study_policy,
        main,
        select_ptv2_b_balanced_view,
        select_ptv2_study_views,
    )
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
        prefix_occurrences=5,
        balanced_occurrences={"math": 2, "code": 3, "stem": 2, "chat": 2, "multilingual": 2},
        occurrence_milestones=(2, 3, 4, 5),
        milestone_steps=(1, 2, 3, 4),
    )


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
    pq.write_table(pa.table({"messages": [json.dumps(value) for value in messages]}), source_path)
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


def test_approved_ptv2_study_policy_is_exact() -> None:
    """Changing one-pass arithmetic or source-native response policy must fail."""
    policy = load_ptv2_study_policy(POLICY)

    assert policy.prefix_occurrences == 2_000_000
    assert policy.balanced_occurrences == {
        "math": 500_000,
        "code": 400_000,
        "stem": 500_000,
        "chat": 400_000,
        "multilingual": 200_000,
    }
    assert policy.occurrence_milestones == (500_000, 1_000_000, 1_300_000, 2_000_000)
    assert policy.milestone_steps == (977, 1_954, 2_540, 3_907)
    assert policy.milestone_cursors == (500_224, 1_000_448, 1_300_480, 2_000_000)
    assert policy.runtime_screen_tokens == 64_000_000
    assert policy.conditional_scientific_tokens == 256_000_000
    assert policy.assistant_responses == "source-native"
    assert policy.trainer_epochs == 1
    assert policy.global_batch_size == 512
    assert policy.final_partial_batch_occurrences == 128
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
        ("occurrences: 2000000", "occurrences: 25391"),
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
        select_ptv2_study_views(
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
    tmp_path: Path,
) -> None:
    """Task3's published receipt binds the source file, native response, and row order."""
    receipt, staged_file = _write_authenticated_staged_parquet(tmp_path)

    rows = tuple(iter_ptv2_staged_source_rows(receipt))

    assert [(row.source_row, row.cell) for row in rows] == [(0, "math"), (1, "math")]
    assert [json.loads(row.assistant_response)["content"] for row in rows] == [
        "answer-first",
        "answer-second",
    ]
    staged_file.write_bytes(b"mutated payload")
    with pytest.raises(SourceManifestError, match="stale staged file"):
        tuple(iter_ptv2_staged_source_rows(receipt))


def test_staged_inventory_rejects_undeclared_orphan_parquet(tmp_path: Path) -> None:
    """The known multilingual orphan cannot enter B through a physical-directory scan."""
    receipt, staged_file = _write_authenticated_staged_parquet(tmp_path)
    orphan = staged_file.parent / "multilingual-00000-of-00001.parquet"
    orphan.write_bytes(staged_file.read_bytes())

    with pytest.raises(PTV2StudyError, match="undeclared or missing Parquet"):
        tuple(iter_ptv2_staged_source_rows(receipt))


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
        tuple(iter_ptv2_staged_source_rows(tmp_path / "SOURCE_PLAN.json"))


def test_b_cli_requires_the_full_declared_shard_count(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The executable B path defaults to the audited 201 declared PTV2 shards."""
    rows = tuple(
        _row(cell, index, cell)
        for index, cell in enumerate(("math", "code", "stem", "chat", "multilingual"))
    )
    seen: dict[str, int] = {}

    def _rows(_: Path, *, expected_declared_shards: int | None = None):
        seen["expected_declared_shards"] = expected_declared_shards or 0
        return iter(rows)

    monkeypatch.setattr(study_module, "load_ptv2_study_policy", lambda _: _scaled_policy())
    monkeypatch.setattr(study_module, "iter_ptv2_staged_source_rows", _rows)
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
        ],
    )

    assert main() == 0
    assert seen == {"expected_declared_shards": 201}
