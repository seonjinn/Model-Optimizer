# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

import hashlib
import json
import sqlite3
import sys
from collections import Counter
from dataclasses import replace
from pathlib import Path
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, cast

import pytest

if TYPE_CHECKING:
    from collections.abc import Sequence

ROOT = Path(__file__).resolve().parents[3]
MODULE_DIR = ROOT / "examples/dataset"

sys.path.insert(0, str(MODULE_DIR))
try:
    import build_specdec_inventory as inventory_module
    import select_bprime_cd_prompts as selection_module
    from bprime_cd_policy import ArmPolicy, PromptCell, PromptPolicy, load_prompt_policy
    from build_specdec_inventory import (
        APPROVED_PTV2_ALLOWLIST_SHA256,
        CandidateCell,
        CandidateInventory,
        CandidatePrompt,
        ExclusionProof,
        build_candidate_inventory,
        candidate_inventory_sha256,
        make_exclusion_receipt,
        write_tokenizer_snapshot_receipt,
    )
    from select_bprime_cd_prompts import (
        DiskBackedSelectedRows,
        PromptSelectionBlocked,
        build_policy_count_proofs,
        publish_bprime_prompt_view_bundle,
        select_bprime_prompt_view,
        select_prompt_views,
    )
    from specdec_corpus_contracts import canonical_json
    from stage_ptv23_sources import (
        SourceFile,
        SourceIdentity,
        SourceInventory,
        load_source_inventory,
        stage_source_inventory,
    )
finally:
    sys.path.pop(0)

BASELINE_RECEIPT_SHA256 = "1" * 64
HELD_OUT_RECEIPT_SHA256 = "2" * 64
PTV2_REVISION = "5c89e01dd720ae0f4058445ed49c5fb68a03c76e"
PTV2_REPOSITORY = "nvidia/Nemotron-Post-Training-Dataset-v2"
PTV2_SPLITS = (
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


def _parallel_shards() -> list[dict[str, object]]:
    return [
        {
            "index": index,
            "spool_path": f"shard-{index:03d}.sqlite3",
            "row_count": index + 1,
            "spool_bytes": index + 1,
            "spool_sha256": hashlib.sha256(str(index).encode()).hexdigest(),
            "elapsed_seconds": 0.1,
            "worker_pid": 1_000 + index,
        }
        for index in range(201)
    ]


def _tokenization_shards() -> list[dict[str, object]]:
    return [
        {
            "index": index,
            "row_count": index + 1,
            "elapsed_seconds": 0.1,
            "worker_pid": 1_000 + index,
        }
        for index in range(201)
    ]


PTV2_SHARD_COUNTS = {
    "chat": 12,
    "math": 2,
    "code": 2,
    "stem": 2,
    "multilingual_de": 38,
    "multilingual_ja": 37,
    "multilingual_es": 33,
    "multilingual_fr": 37,
    "multilingual_it": 38,
}


class _CandidateTokenizer:
    def apply_chat_template(self, messages, **kwargs):
        assert kwargs["add_generation_prompt"] is True
        return {"input_ids": list(range(1, len(messages) + 1))}


def _authenticated_tokenizer_snapshot(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    root = tmp_path / "tokenizer"
    root.mkdir(exist_ok=True)
    (root / "tokenizer.json").write_text('{"version":"fixture"}', encoding="utf-8")
    (root / "tokenizer_config.json").write_text(
        json.dumps({"chat_template": "{{ messages }}"}), encoding="utf-8"
    )
    receipt = tmp_path / "TOKENIZER_SNAPSHOT.json"
    snapshot = write_tokenizer_snapshot_receipt(root, receipt)
    monkeypatch.setattr(
        inventory_module, "_load_tokenizer_from_snapshot", lambda _snapshot: _CandidateTokenizer()
    )
    return snapshot, {
        "tokenizer_snapshot_receipt": receipt,
        "tokenizer_snapshot_receipt_sha256": snapshot.receipt_sha256,
    }


def _genuine_task3_ptv2(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    splits: tuple[str, ...] = PTV2_SPLITS,
    omit_first_assistant: bool = False,
) -> tuple[SourceInventory, CandidateInventory, dict[str, object]]:
    pa = pytest.importorskip("pyarrow")
    pq = pytest.importorskip("pyarrow.parquet")
    snapshot, tokenizer_arguments = _authenticated_tokenizer_snapshot(tmp_path, monkeypatch)
    local = tmp_path / "local"
    sources = []
    for split_index, split in enumerate(splits):
        count = PTV2_SHARD_COUNTS[split]
        files = []
        for file_index in range(count):
            relative = f"data/{split}/{file_index:03d}.parquet"
            source = local / PTV2_REPOSITORY / PTV2_REVISION / relative
            source.parent.mkdir(parents=True, exist_ok=True)
            records = []
            for row_index in range(30):
                messages = [
                    {
                        "role": "user",
                        "content": f"question-{split}-{file_index}-{row_index}",
                    },
                    {
                        "role": "assistant",
                        "content": f"answer-{split}-{file_index}-{row_index}",
                    },
                ]
                if omit_first_assistant and split_index == file_index == row_index == 0:
                    messages.pop()
                records.append({"messages": messages})
            pq.write_table(pa.Table.from_pylist(records), source)
            files.append(
                {
                    "path": relative,
                    "bytes": source.stat().st_size,
                    "sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                }
            )
        sources.append(
            {
                "repository_id": PTV2_REPOSITORY,
                "configuration": "default",
                "split": split,
                "revision": PTV2_REVISION,
                "license_expression": "NVIDIA Open Model License",
                "approved_use": True,
                "cell": split,
                "lane": "target-synth",
                "files": files,
            }
        )
    plan_path = tmp_path / "SOURCE_PLAN_INPUT.json"
    plan_path.write_text(
        json.dumps({"schema_version": 1, "name": "ptv2-fixture", "sources": sources}),
        encoding="utf-8",
    )
    staged = stage_source_inventory(
        load_source_inventory(plan_path),
        durable_root=tmp_path / "durable",
        scratch_root=tmp_path / "scratch",
        local_source_root=local,
    )
    candidates = build_candidate_inventory(
        staged,
        tokenizer=_CandidateTokenizer(),
        tokenizer_sha256=snapshot.tokenizer_sha256,
        chat_template_sha256=snapshot.chat_template_sha256,
        baseline_exclusion=make_exclusion_receipt("baseline", ()),
        held_out_exclusion=make_exclusion_receipt("held-out", ()),
        storage_dir=tmp_path / "candidate-storage",
    )
    return staged, candidates, tokenizer_arguments


def _policy() -> PromptPolicy:
    b_counts = {"stem": 40, "japanese": 25, "spanish": 25, "french": 25, "italian": 25}
    cd_counts = {
        "swe-agentic-tool": 120,
        "math": 80,
        "code": 40,
        "stem-science": 80,
        "multilingual": 60,
        "instruction-chat": 20,
    }
    arms = {
        "B-prime": ArmPolicy(
            140,
            MappingProxyType({key: PromptCell(value) for key, value in b_counts.items()}),
            MappingProxyType({}),
        ),
        "C": ArmPolicy(
            400,
            MappingProxyType({key: PromptCell(value) for key, value in cd_counts.items()}),
            MappingProxyType({}),
        ),
        "D": ArmPolicy(
            400,
            MappingProxyType({key: PromptCell(value) for key, value in cd_counts.items()}),
            MappingProxyType(
                {
                    "agentless-swe": 40,
                    "interactive-swe-replay": 40,
                    "generic-tool-replay": 40,
                }
            ),
        ),
    }
    return PromptPolicy(
        schema_version=1,
        seed=20260822,
        reserve_numerator=6,
        reserve_denominator=5,
        arms=MappingProxyType(arms),
        exposure_tokens=(256_000_000, 1_000_000_000),
        sequence_length=4_096,
        full_context_maximum=32_768,
        allowed_languages=frozenset({"en", "ja", "es", "fr", "it"}),
        denied_languages=frozenset({"de"}),
        policy_sha256="a" * 64,
    )


def _candidate(
    ordinal: int,
    *,
    domain: str,
    lane: str,
    language: str,
    bucket: str,
    source_family: str | None = None,
) -> CandidatePrompt:
    prompt = {"messages": [{"role": "user", "content": f"prompt-{ordinal}"}], "tools": []}
    canonical = json.dumps(prompt, sort_keys=True, separators=(",", ":")).encode()
    family = source_family or ("ptv2" if domain in {"stem-science", "multilingual"} else "ptv3")
    ptv2_split = "stem" if domain == "stem-science" else f"multilingual_{language}"
    return CandidatePrompt(
        prompt_uuid=hashlib.sha256(canonical).hexdigest(),
        canonical_bytes=canonical,
        source_id=f"fixture/{domain}/{lane}",
        source_revision=PTV2_REVISION if family == "ptv2" else "9" * 40,
        source_file_sha256=hashlib.sha256(f"{domain}/{lane}".encode()).hexdigest(),
        source_row_index=ordinal,
        domain=domain,
        language=language,
        lane=lane,
        context_bucket=bucket,
        full_token_count=2 if bucket == "le4k" else 5_000,
        source_manifest_sha256="c" * 64,
        source_file_path=f"{domain}/{lane}.jsonl",
        input_ids=(1, 2),
        tokenizer_sha256="d" * 64,
        replay_valid=lane in {"interactive-swe-replay", "generic-tool-replay"},
        source_family=family,
        source_repository_id=(
            "nvidia/Nemotron-Post-Training-Dataset-v2" if family == "ptv2" else "fixture/source"
        ),
        source_configuration="default",
        source_split=ptv2_split if family == "ptv2" else lane,
    )


def _inventory(*, reverse: bool = False, drop_last: bool = False) -> CandidateInventory:
    specifications = [
        ("stem-science", "target-synth", "en", 120),
        ("multilingual", "target-synth", "ja", 30),
        ("multilingual", "target-synth", "es", 30),
        ("multilingual", "target-synth", "fr", 30),
        ("multilingual", "target-synth", "it", 30),
        ("math", "target-synth", "en", 96),
        ("code", "target-synth", "en", 48),
        ("instruction-chat", "target-synth", "en", 24),
        ("swe-agentic-tool", "agentless-swe", "en", 144),
        ("swe-agentic-tool", "interactive-swe-replay", "en", 48),
        ("swe-agentic-tool", "generic-tool-replay", "en", 48),
    ]
    rows: list[CandidatePrompt] = []
    ordinal = 0
    for domain, lane, language, count in specifications:
        for index in range(count):
            rows.append(
                _candidate(
                    ordinal,
                    domain=domain,
                    lane=lane,
                    language=language,
                    bucket=("le4k" if domain == "multilingual" or index % 2 == 0 else "4k_16k"),
                )
            )
            ordinal += 1
    if drop_last:
        rows.pop()
    capacity = Counter(
        CandidateCell(row.domain, row.lane, row.language, row.context_bucket) for row in rows
    )
    inventory = CandidateInventory(
        rows=tuple(reversed(rows)) if reverse else tuple(rows),
        capacity=MappingProxyType(dict(capacity)),
        quarantine_counts=MappingProxyType({}),
        inventory_sha256="0" * 64,
        baseline_exclusion=ExclusionProof(BASELINE_RECEIPT_SHA256, "6" * 64, 1, 0),
        held_out_exclusion=ExclusionProof(HELD_OUT_RECEIPT_SHA256, "7" * 64, 0, 0),
        ptv2_revision=PTV2_REVISION,
        ptv2_allowlist_sha256=APPROVED_PTV2_ALLOWLIST_SHA256,
    )
    return replace(inventory, inventory_sha256=candidate_inventory_sha256(inventory))


def _all_rows(view):
    return (*view.primary_rows, *view.reserve_rows)


def _select(inventory: CandidateInventory, policy: PromptPolicy):
    return select_prompt_views(
        inventory,
        policy,
        baseline_receipt_sha256=BASELINE_RECEIPT_SHA256,
        held_out_receipt_sha256=HELD_OUT_RECEIPT_SHA256,
    )


def _rehash(inventory: CandidateInventory) -> CandidateInventory:
    return replace(inventory, inventory_sha256=candidate_inventory_sha256(inventory))


def _candidate_rows(inventory: CandidateInventory) -> Sequence[CandidatePrompt]:
    return cast("Sequence[CandidatePrompt]", inventory.rows)


def _ptv2_only_inventory() -> CandidateInventory:
    inventory = _inventory()
    rows = tuple(row for row in _candidate_rows(inventory) if row.source_family == "ptv2")
    capacity = Counter(
        CandidateCell(row.domain, row.lane, row.language, row.context_bucket) for row in rows
    )
    return _rehash(replace(inventory, rows=rows, capacity=MappingProxyType(dict(capacity))))


def _task3_ptv2_inventory(
    candidates: CandidateInventory,
) -> tuple[SourceInventory, CandidateInventory]:
    canonical = b"authenticated-task3-ptv2-fixture"
    manifest_sha256 = hashlib.sha256(canonical).hexdigest()
    splits = (
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
    matching = {row.source_split: row for row in _candidate_rows(candidates)}
    sources = []
    remaining = 201
    for index, split in enumerate(splits):
        count = remaining // (len(splits) - index)
        remaining -= count
        row = matching.get(split)
        files = []
        for file_index in range(count):
            if file_index == 0 and row is not None:
                path = row.source_file_path
                digest = row.source_file_sha256
            else:
                path = f"{split}/extra-{file_index:03d}.parquet"
                digest = hashlib.sha256(path.encode()).hexdigest()
            files.append(SourceFile(path, 1, digest))
        sources.append(
            SourceIdentity(
                "nvidia/Nemotron-Post-Training-Dataset-v2",
                "default",
                split,
                PTV2_REVISION,
                "NVIDIA Open Model License",
                True,
                split,
                "target-synth",
                tuple(files),
            )
        )
    source_inventory = SourceInventory(
        1,
        "ptv2-fixture",
        tuple(sources),
        manifest_sha256,
        canonical,
        MappingProxyType({}),
    )
    rows = tuple(
        replace(row, source_manifest_sha256=manifest_sha256) for row in _candidate_rows(candidates)
    )
    return source_inventory, _rehash(replace(candidates, rows=rows))


def test_bprime_only_producer_rejects_self_hashed_unstaged_task3_inventory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An in-memory descriptor graph is not an authenticated Task3 receipt."""
    source_inventory, candidates = _task3_ptv2_inventory(_ptv2_only_inventory())
    _snapshot, tokenizer_arguments = _authenticated_tokenizer_snapshot(tmp_path, monkeypatch)
    with pytest.raises(ValueError, match="staged Task 3"):
        select_bprime_prompt_view(
            candidates,
            _policy(),
            source_inventory=source_inventory,
            **tokenizer_arguments,
            baseline_receipt_sha256=BASELINE_RECEIPT_SHA256,
            held_out_receipt_sha256=HELD_OUT_RECEIPT_SHA256,
        )


def test_bprime_only_producer_authenticates_physical_task3_and_publishes_only_bprime(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The genuine 201-shard stage/build/select/publish path is executable."""
    source_inventory, candidates, tokenizer_arguments = _genuine_task3_ptv2(tmp_path, monkeypatch)
    bundle = select_bprime_prompt_view(
        candidates,
        _policy(),
        source_inventory=source_inventory,
        **tokenizer_arguments,
        baseline_receipt_sha256=candidates.baseline_exclusion.receipt_sha256,
        held_out_receipt_sha256=candidates.held_out_exclusion.receipt_sha256,
    )
    execution: dict[str, Any] = {
        "schema_version": 1,
        "source_commit": "a" * 40,
        "source_manifest_sha256": source_inventory.manifest_sha256,
        "declared_shard_count": 201,
        "allocated_cpus": 96,
        "requested_workers": 96,
        "effective_workers": 96,
        "threads_per_worker": 1,
        "thread_environment": dict.fromkeys(
            (
                "ARROW_NUM_THREADS",
                "OMP_NUM_THREADS",
                "MKL_NUM_THREADS",
                "OPENBLAS_NUM_THREADS",
                "NUMEXPR_NUM_THREADS",
            ),
            "1",
        ),
        "started_at_ns": 1,
        "finished_at_ns": 2,
        "elapsed_seconds": 1.0,
        "accepted_count": len(candidates.rows),
        "quarantine_counts": {},
        "shards": _parallel_shards(),
        "tokenization_shards": _tokenization_shards(),
    }
    execution["receipt_sha256"] = hashlib.sha256(canonical_json(execution)).hexdigest()
    destination = tmp_path / "bprime-only"
    try:
        forged_execution = dict(execution)
        forged_execution["source_manifest_sha256"] = "f" * 64
        forged_execution.pop("receipt_sha256")
        forged_execution["receipt_sha256"] = hashlib.sha256(
            canonical_json(forged_execution)
        ).hexdigest()
        forged_destination = tmp_path / "forged-source-root"
        with pytest.raises(ValueError, match="source manifest"):
            publish_bprime_prompt_view_bundle(
                bundle,
                forged_destination,
                rows_per_shard=37,
                execution_receipt=forged_execution,
            )
        assert not forged_destination.exists()
        published = publish_bprime_prompt_view_bundle(
            bundle,
            destination,
            rows_per_shard=37,
            execution_receipt=execution,
        )
        before = published.manifest_path.read_bytes()
        with pytest.raises(FileExistsError) as caught:
            publish_bprime_prompt_view_bundle(
                bundle,
                destination,
                rows_per_shard=37,
                execution_receipt=execution,
            )
    finally:
        bundle.close()
        candidates.rows.close()
    manifest = json.loads(published.manifest_path.read_bytes())

    assert manifest["selection_mode"] == "B-prime-only"
    assert set(manifest["arms"]) == {"B-prime"}
    assert manifest["identity"]["candidate_inventory_sha256"] == candidates.inventory_sha256
    assert manifest["identity"]["source_inventory_sha256"] == candidates.inventory_sha256
    assert manifest["identity"]["source_manifest_sha256"] == source_inventory.manifest_sha256
    assert published.manifest_path.read_bytes() == before
    assert (destination / "EXECUTION_RECEIPT.json").is_file()
    assert getattr(caught.value, "recovery_state").destination_path == destination
    pytest.importorskip("pyarrow")
    sys.path.insert(0, str(MODULE_DIR))
    try:
        import specdec_publication as publication
    finally:
        sys.path.pop(0)
    descriptors = publication._role_file_descriptors("selection", manifest)
    assert manifest["execution_receipt"] in descriptors
    authenticated_files = tuple(
        (
            descriptor["path"],
            destination / descriptor["path"],
            descriptor.get("bytes", descriptor.get("byte_count", (destination / descriptor["path"]).stat().st_size)),
            descriptor["sha256"],
        )
        for descriptor in descriptors
    )
    publication._validate_task5_execution_receipt(manifest, list(authenticated_files))
    copied = tmp_path / "copied-publication"
    copied.mkdir()
    publication._copy_authenticated_inputs(
        copied,
        (
            publication._AuthenticatedArtifact(
                "selection",
                published.manifest_path,
                hashlib.sha256(published.manifest_path.read_bytes()).hexdigest(),
                authenticated_files,
            ),
        ),
    )
    assert (copied / "inputs/selection/files/EXECUTION_RECEIPT.json").is_file()
    with sqlite3.connect(published.index_path) as connection:
        assert {row[0] for row in connection.execute("SELECT DISTINCT arm FROM rows")} == {
            "B-prime"
        }


def test_bprime_selection_preserves_source_response_identities(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Task5 rows and their published reload bind the physical PTV2 response."""
    source_inventory, candidates, tokenizer_arguments = _genuine_task3_ptv2(tmp_path, monkeypatch)
    bundle = select_bprime_prompt_view(
        candidates,
        _policy(),
        source_inventory=source_inventory,
        **tokenizer_arguments,
        baseline_receipt_sha256=candidates.baseline_exclusion.receipt_sha256,
        held_out_receipt_sha256=candidates.held_out_exclusion.receipt_sha256,
    )
    try:
        selected = bundle.B_prime.primary_rows[0]
        assert bundle.B_prime.tokenizer_sha256 == bundle.tokenizer_sha256
        assert bundle.B_prime.chat_template_sha256 == bundle.chat_template_sha256
        assert selected.source_conversation_sha256 is not None
        assert selected.source_response_sha256 is not None
        assert selected.tokenizer_sha256 == bundle.tokenizer_sha256
        assert selected.chat_template_sha256 == bundle.chat_template_sha256
        published = publish_bprime_prompt_view_bundle(
            bundle, tmp_path / "response-bound", rows_per_shard=17
        )
    finally:
        bundle.close()
        candidates.rows.close()
    sys.path.insert(0, str(MODULE_DIR))
    try:
        from promote_synthesis_reserve import load_prompt_view
    finally:
        sys.path.pop(0)
    loaded = load_prompt_view(
        published.manifest_path,
        expected_manifest_sha256=hashlib.sha256(published.manifest_path.read_bytes()).hexdigest(),
        arm="B-prime",
    )
    assert loaded.primary_rows[0].source_conversation_sha256 is not None
    assert loaded.primary_rows[0].source_response_sha256 is not None
    assert loaded.primary_rows[0].tokenizer_sha256 == loaded.tokenizer_sha256
    assert loaded.primary_rows[0].chat_template_sha256 == loaded.chat_template_sha256


def test_bprime_only_producer_rejects_ptv3_candidate_contamination(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """No PTV3 candidate can enter the physical PTV2-only producer."""
    source_inventory, candidates, tokenizer_arguments = _genuine_task3_ptv2(tmp_path, monkeypatch)
    ptv3_row = next(row for row in _candidate_rows(_inventory()) if row.source_family == "ptv3")
    rows = (*_candidate_rows(candidates), ptv3_row)
    capacity = Counter(
        CandidateCell(row.domain, row.lane, row.language, row.context_bucket) for row in rows
    )
    mixed = _rehash(replace(candidates, rows=rows, capacity=MappingProxyType(dict(capacity))))
    try:
        with pytest.raises(ValueError, match="PTV2-only"):
            select_bprime_prompt_view(
                mixed,
                _policy(),
                source_inventory=source_inventory,
                **tokenizer_arguments,
                baseline_receipt_sha256=candidates.baseline_exclusion.receipt_sha256,
                held_out_receipt_sha256=candidates.held_out_exclusion.receipt_sha256,
            )
    finally:
        candidates.rows.close()


def test_bprime_only_producer_rejects_wrong_task3_split_topology(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A staged 201-shard receipt must contain every approved split exactly once."""
    source_inventory, candidates, tokenizer_arguments = _genuine_task3_ptv2(
        tmp_path, monkeypatch, splits=PTV2_SPLITS[:-1]
    )
    try:
        with pytest.raises(ValueError, match="approved PTV2 topology"):
            select_bprime_prompt_view(
                candidates,
                _policy(),
                source_inventory=source_inventory,
                **tokenizer_arguments,
                baseline_receipt_sha256=candidates.baseline_exclusion.receipt_sha256,
                held_out_receipt_sha256=candidates.held_out_exclusion.receipt_sha256,
            )
    finally:
        candidates.rows.close()


@pytest.mark.parametrize("mutation", ["missing", "extra", "modified"])
def test_bprime_only_producer_rejects_changed_physical_task3_shards(
    tmp_path: Path, mutation: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Physical staged shards remain an authenticated producer input."""
    source_inventory, candidates, tokenizer_arguments = _genuine_task3_ptv2(tmp_path, monkeypatch)
    assert source_inventory.staged_root is not None
    first = next(
        path
        for path in (source_inventory.staged_root / "sources").rglob("*.parquet")
        if path.is_file()
    )
    if mutation == "missing":
        first.unlink()
    elif mutation == "extra":
        extra = first.parent / "undeclared.jsonl"
        extra.write_bytes(first.read_bytes())
    else:
        first.write_bytes(first.read_bytes() + b"\n")
    try:
        with pytest.raises(ValueError, match="staged Task 3"):
            select_bprime_prompt_view(
                candidates,
                _policy(),
                source_inventory=source_inventory,
                **tokenizer_arguments,
                baseline_receipt_sha256=candidates.baseline_exclusion.receipt_sha256,
                held_out_receipt_sha256=candidates.held_out_exclusion.receipt_sha256,
            )
    finally:
        candidates.rows.close()


def test_bprime_only_producer_rejects_undeclared_symlink_alias(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Resolved-path set equality must not hide an extra staged directory entry."""
    source_inventory, candidates, tokenizer_arguments = _genuine_task3_ptv2(tmp_path, monkeypatch)
    assert source_inventory.staged_root is not None
    first = next(
        path
        for path in (source_inventory.staged_root / "sources").rglob("*.parquet")
        if path.is_file()
    )
    (first.parent / "undeclared-link.jsonl").symlink_to(first.name)
    try:
        with pytest.raises(ValueError, match="physical shard set"):
            select_bprime_prompt_view(
                candidates,
                _policy(),
                source_inventory=source_inventory,
                **tokenizer_arguments,
                baseline_receipt_sha256=candidates.baseline_exclusion.receipt_sha256,
                held_out_receipt_sha256=candidates.held_out_exclusion.receipt_sha256,
            )
    finally:
        candidates.rows.close()


def test_bprime_only_producer_rejects_source_mutation_during_stable_fd_scan(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A shard changed after open cannot pass using its earlier authenticated digest."""
    source_inventory, candidates, tokenizer_arguments = _genuine_task3_ptv2(tmp_path, monkeypatch)
    assert source_inventory.staged_root is not None
    source = source_inventory.sources[0]
    source_file = source.files[0]
    path = (
        source_inventory.staged_root
        / "sources"
        / source.repository_id
        / source.revision
        / source_file.path
    )
    original = inventory_module._iter_candidate_rows_fd
    mutated = False

    def mutating_rows(descriptor: int, suffix: str):
        nonlocal mutated
        for row in original(descriptor, suffix):
            yield row
            if not mutated:
                path.write_bytes(path.read_bytes() + b" ")
                mutated = True

    monkeypatch.setattr(inventory_module, "_iter_candidate_rows_fd", mutating_rows)
    try:
        with pytest.raises(ValueError, match="changed during authentication"):
            select_bprime_prompt_view(
                candidates,
                _policy(),
                source_inventory=source_inventory,
                **tokenizer_arguments,
                baseline_receipt_sha256=candidates.baseline_exclusion.receipt_sha256,
                held_out_receipt_sha256=candidates.held_out_exclusion.receipt_sha256,
            )
    finally:
        candidates.rows.close()


def test_bprime_only_producer_rejects_rehashed_tokenization_and_capacity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Bucket capacity is derived from the pinned tokenizer, not candidate metadata."""
    source_inventory, candidates, tokenizer_arguments = _genuine_task3_ptv2(tmp_path, monkeypatch)
    rows = list(_candidate_rows(candidates))
    rows[0] = replace(rows[0], context_bucket="16k_32k", full_token_count=20_000, input_ids=(999,))
    capacity = Counter(
        CandidateCell(row.domain, row.lane, row.language, row.context_bucket) for row in rows
    )
    forged = _rehash(
        replace(candidates, rows=tuple(rows), capacity=MappingProxyType(dict(capacity)))
    )
    try:
        with pytest.raises(ValueError, match="tokenization"):
            select_bprime_prompt_view(
                forged,
                _policy(),
                source_inventory=source_inventory,
                **tokenizer_arguments,
                baseline_receipt_sha256=candidates.baseline_exclusion.receipt_sha256,
                held_out_receipt_sha256=candidates.held_out_exclusion.receipt_sha256,
            )
    finally:
        candidates.rows.close()


def test_bprime_only_producer_rejects_wrong_tokenizer_snapshot_receipt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The caller cannot relabel a different tokenizer/template snapshot."""
    source_inventory, candidates, tokenizer_arguments = _genuine_task3_ptv2(tmp_path, monkeypatch)
    tokenizer_arguments["tokenizer_snapshot_receipt_sha256"] = "e" * 64
    try:
        with pytest.raises(ValueError, match="receipt SHA-256 mismatch"):
            select_bprime_prompt_view(
                candidates,
                _policy(),
                source_inventory=source_inventory,
                **tokenizer_arguments,
                baseline_receipt_sha256=candidates.baseline_exclusion.receipt_sha256,
                held_out_receipt_sha256=candidates.held_out_exclusion.receipt_sha256,
            )
    finally:
        candidates.rows.close()


def test_bprime_only_producer_has_no_caller_adapter_surface() -> None:
    """A lying caller adapter cannot be supplied to production verification."""
    with pytest.raises(TypeError, match="unexpected keyword argument 'tokenizer'"):
        select_bprime_prompt_view(  # type: ignore[call-arg]
            _ptv2_only_inventory(),
            _policy(),
            source_inventory=cast("SourceInventory", object()),
            tokenizer=_CandidateTokenizer(),
            tokenizer_sha256="d" * 64,
            baseline_receipt_sha256=BASELINE_RECEIPT_SHA256,
            held_out_receipt_sha256=HELD_OUT_RECEIPT_SHA256,
        )


def test_bprime_only_producer_rejects_missing_source_native_response(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Every PTV2 target-synthesis candidate must bind a terminal assistant response."""
    with pytest.raises(ValueError, match="source-native assistant response"):
        _genuine_task3_ptv2(tmp_path, monkeypatch, omit_first_assistant=True)


@pytest.mark.parametrize("mutation", ["row", "uuid", "canonical"])
def test_bprime_only_producer_rejects_candidate_not_on_its_physical_row(
    tmp_path: Path, mutation: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Rehashing forged candidate metadata cannot create physical membership."""
    source_inventory, candidates, tokenizer_arguments = _genuine_task3_ptv2(tmp_path, monkeypatch)
    rows = list(_candidate_rows(candidates))
    if mutation == "row":
        rows[0] = replace(rows[0], source_row_index=100_000)
    elif mutation == "uuid":
        rows[0] = replace(rows[0], prompt_uuid="f" * 64)
    else:
        rows[0] = replace(rows[0], canonical_bytes=b'{"messages":[],"tools":[]}')
    forged = _rehash(replace(candidates, rows=tuple(rows)))
    try:
        with pytest.raises(ValueError, match="physical Task 3 row"):
            select_bprime_prompt_view(
                forged,
                _policy(),
                source_inventory=source_inventory,
                **tokenizer_arguments,
                baseline_receipt_sha256=candidates.baseline_exclusion.receipt_sha256,
                held_out_receipt_sha256=candidates.held_out_exclusion.receipt_sha256,
            )
    finally:
        candidates.rows.close()


@pytest.mark.parametrize(
    ("replacement", "message"),
    [
        ({"domain": "stem"}, "canonical cell"),
        ({"language": "de"}, "language"),
    ],
)
def test_bprime_only_producer_rejects_wrong_ptv2_cell_or_language(
    tmp_path: Path,
    replacement: dict[str, str],
    message: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Noncanonical Task3-to-Task5 cell mappings cannot enter the complement."""
    source_inventory, inventory, tokenizer_arguments = _genuine_task3_ptv2(tmp_path, monkeypatch)
    rows = list(_candidate_rows(inventory))
    rows[0] = replace(rows[0], **replacement)
    capacity = Counter(
        CandidateCell(row.domain, row.lane, row.language, row.context_bucket) for row in rows
    )
    forged = _rehash(
        replace(inventory, rows=tuple(rows), capacity=MappingProxyType(dict(capacity)))
    )
    try:
        with pytest.raises(ValueError, match=message):
            select_bprime_prompt_view(
                forged,
                _policy(),
                source_inventory=source_inventory,
                **tokenizer_arguments,
                baseline_receipt_sha256=inventory.baseline_exclusion.receipt_sha256,
                held_out_receipt_sha256=inventory.held_out_exclusion.receipt_sha256,
            )
    finally:
        inventory.rows.close()


def test_bprime_only_producer_rejects_wrong_source_digest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The dedicated producer authenticates the full Task3 shard and row identities."""
    source_inventory, inventory, tokenizer_arguments = _genuine_task3_ptv2(tmp_path, monkeypatch)
    forged_rows = list(_candidate_rows(inventory))
    forged_rows[0] = replace(forged_rows[0], source_file_sha256="f" * 64)
    forged = _rehash(replace(inventory, rows=tuple(forged_rows)))
    try:
        with pytest.raises(ValueError, match="source digest"):
            select_bprime_prompt_view(
                forged,
                _policy(),
                source_inventory=source_inventory,
                **tokenizer_arguments,
                baseline_receipt_sha256=inventory.baseline_exclusion.receipt_sha256,
                held_out_receipt_sha256=inventory.held_out_exclusion.receipt_sha256,
            )
    finally:
        inventory.rows.close()


def test_selection_requires_nonzero_inventory_bound_exclusion_receipts() -> None:
    inventory = _inventory()
    policy = _policy()

    with pytest.raises(TypeError):
        select_prompt_views(inventory, policy)
    with pytest.raises(ValueError, match="nonzero"):
        select_prompt_views(
            inventory,
            policy,
            baseline_receipt_sha256="0" * 64,
            held_out_receipt_sha256=HELD_OUT_RECEIPT_SHA256,
        )
    with pytest.raises(ValueError, match="does not match candidate inventory"):
        select_prompt_views(
            inventory,
            policy,
            baseline_receipt_sha256="8" * 64,
            held_out_receipt_sha256=HELD_OUT_RECEIPT_SHA256,
        )
    inconsistent = _rehash(
        replace(
            inventory,
            baseline_exclusion=replace(inventory.baseline_exclusion, excluded_candidate_count=1),
        )
    )
    with pytest.raises(ValueError, match="exclusion reconciliation"):
        _select(inconsistent, policy)


def test_bprime_rejects_non_ptv2_or_unpinned_source_rows() -> None:
    inventory = _inventory()
    rows = tuple(
        replace(row, source_family="ptv3", source_revision="9" * 40)
        if row.domain in {"stem-science", "multilingual"}
        else row
        for row in inventory.rows
    )

    with pytest.raises(PromptSelectionBlocked) as caught:
        _select(_rehash(replace(inventory, rows=rows)), _policy())

    assert caught.value.receipt["arm"] == "B-prime"
    assert caught.value.receipt["reason"] == "insufficient_candidates"


def test_selection_independently_recomputes_streamed_inventory_identity() -> None:
    inventory = _inventory()
    tampered_rows = (
        replace(inventory.rows[0], source_file_path="tampered.jsonl"),
        *inventory.rows[1:],
    )

    with pytest.raises(ValueError, match="streamed identity mismatch"):
        _select(replace(inventory, rows=tampered_rows), _policy())


def test_rehashed_unapproved_structured_bprime_source_is_rejected() -> None:
    inventory = _inventory()
    tampered_rows = tuple(
        replace(row, source_repository_id="attacker/revision-spoof")
        if row.source_family == "ptv2"
        else row
        for row in inventory.rows
    )

    with pytest.raises(PromptSelectionBlocked) as caught:
        _select(_rehash(replace(inventory, rows=tampered_rows)), _policy())

    assert caught.value.receipt["arm"] == "B-prime"


def test_bprime_has_exact_language_counts_reserves_and_stable_unique_order() -> None:
    policy = _policy()
    forward = _select(_inventory(), policy)
    reverse = _select(_inventory(reverse=True), policy)

    assert forward.B_prime.cell_counts == {
        "stem": 40,
        "japanese": 25,
        "spanish": 25,
        "french": 25,
        "italian": 25,
    }
    assert len(forward.B_prime.primary_prompt_ids) == 140
    assert len(forward.B_prime.reserve_prompt_ids) == 28
    assert Counter((row.domain, row.language) for row in forward.B_prime.primary_rows) == {
        ("stem-science", "en"): 40,
        ("multilingual", "ja"): 25,
        ("multilingual", "es"): 25,
        ("multilingual", "fr"): 25,
        ("multilingual", "it"): 25,
    }
    assert len({row.prompt_uuid for row in _all_rows(forward.B_prime)}) == 168
    assert all(row.language != "de" for row in _all_rows(forward.B_prime))
    assert forward.B_prime.primary_prompt_ids == reverse.B_prime.primary_prompt_ids
    assert forward.B_prime.reserve_prompt_ids == reverse.B_prime.reserve_prompt_ids
    assert forward.selection_sha256 == reverse.selection_sha256
    assert isinstance(forward.B_prime.primary_rows, DiskBackedSelectedRows)
    assert forward.B_prime.primary_rows.resident_row_count == 0


def test_cd_views_are_paired_outside_agentic_lanes() -> None:
    bundle = _select(_inventory(), _policy())

    assert len(bundle.C.primary_prompt_ids) == 400
    assert len(bundle.D.primary_prompt_ids) == 400
    assert (
        bundle.C.cell_counts
        == bundle.D.cell_counts
        == {
            "swe-agentic-tool": 120,
            "math": 80,
            "code": 40,
            "stem-science": 80,
            "multilingual": 60,
            "instruction-chat": 20,
        }
    )
    assert bundle.C.non_agentic_prompt_ids == bundle.D.non_agentic_prompt_ids
    assert bundle.C.non_agentic_bucket_floors == bundle.D.non_agentic_bucket_floors
    assert {
        row.prompt_uuid for row in bundle.C.reserve_rows if row.domain != "swe-agentic-tool"
    } == {row.prompt_uuid for row in bundle.D.reserve_rows if row.domain != "swe-agentic-tool"}
    assert bundle.C.agentic_prompt_ids != bundle.D.agentic_prompt_ids
    assert bundle.C.lane_counts == {"agentless-swe": 120}
    assert bundle.D.lane_counts == {
        "agentless-swe": 40,
        "interactive-swe-replay": 40,
        "generic-tool-replay": 40,
    }
    assert Counter(
        row.lane for row in bundle.D.primary_rows if row.domain == "swe-agentic-tool"
    ) == {
        "agentless-swe": 40,
        "interactive-swe-replay": 40,
        "generic-tool-replay": 40,
    }
    assert bundle.C.non_agentic_bucket_floors["math"] == {
        "le4k": 40,
        "4k_16k": 40,
    }
    assert len(bundle.C.reserve_prompt_ids) == len(bundle.D.reserve_prompt_ids) == 80
    assert bundle.paired_cd_sha256


def test_selection_identity_binds_seed_receipts_source_and_quota() -> None:
    inventory = _inventory()
    policy = _policy()
    original = _select(inventory, policy)
    changed_seed = _select(inventory, replace(policy, seed=policy.seed + 1))
    changed_heldout_inventory = _rehash(
        replace(
            inventory,
            held_out_exclusion=replace(inventory.held_out_exclusion, receipt_sha256="4" * 64),
        )
    )
    changed_heldout = select_prompt_views(
        changed_heldout_inventory,
        policy,
        baseline_receipt_sha256=BASELINE_RECEIPT_SHA256,
        held_out_receipt_sha256="4" * 64,
    )
    changed_baseline_inventory = _rehash(
        replace(
            inventory,
            baseline_exclusion=replace(inventory.baseline_exclusion, receipt_sha256="6" * 64),
        )
    )
    changed_baseline = select_prompt_views(
        changed_baseline_inventory,
        policy,
        baseline_receipt_sha256="6" * 64,
        held_out_receipt_sha256=HELD_OUT_RECEIPT_SHA256,
    )
    changed_rows = tuple(replace(row, source_manifest_sha256="8" * 64) for row in inventory.rows)
    changed_source = _select(_rehash(replace(inventory, rows=changed_rows)), policy)
    b_arm = policy.arms["B-prime"]
    cells = dict(b_arm.cells)
    cells["stem"] = PromptCell(39)
    changed_b = replace(b_arm, prompt_count=139, cells=MappingProxyType(cells))
    changed_quota = select_prompt_views(
        inventory,
        replace(policy, arms=MappingProxyType({**policy.arms, "B-prime": changed_b})),
        baseline_receipt_sha256=BASELINE_RECEIPT_SHA256,
        held_out_receipt_sha256=HELD_OUT_RECEIPT_SHA256,
    )

    digests = {
        original.selection_sha256,
        changed_seed.selection_sha256,
        changed_heldout.selection_sha256,
        changed_baseline.selection_sha256,
        changed_source.selection_sha256,
        changed_quota.selection_sha256,
    }
    assert len(digests) == 6


def test_one_row_shortfall_is_fatal_and_emits_a_blocker_receipt() -> None:
    with pytest.raises(PromptSelectionBlocked) as caught:
        _select(_inventory(drop_last=True), _policy())

    assert caught.value.receipt["status"] == "blocked"
    assert caught.value.receipt["reason"] == "insufficient_candidates"
    assert caught.value.receipt["arm"] == "D"
    assert caught.value.receipt["lane"] == "generic-tool-replay"
    assert caught.value.receipt["deficit"] == 1
    assert "partial_output" not in caught.value.receipt


def test_selection_storage_closes_on_shortfall_and_proof_failure(monkeypatch) -> None:
    original = selection_module._SelectionStorage
    closed: list[Path] = []

    class SpyStorage(original):
        def close(self) -> None:
            closed.append(self.path)
            super().close()

    monkeypatch.setattr(selection_module, "_SelectionStorage", SpyStorage)
    with pytest.raises(PromptSelectionBlocked):
        _select(_inventory(drop_last=True), _policy())
    assert len(closed) == 1 and not closed[-1].parent.exists()

    monkeypatch.setattr(
        selection_module,
        "_build_disk_view",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("proof failure")),
    )
    with pytest.raises(AssertionError, match="proof failure"):
        _select(_inventory(), _policy())
    assert len(closed) == 2 and not closed[-1].parent.exists()


def test_canonical_large_count_proof_uses_exact_policy_arithmetic_only() -> None:
    policy = load_prompt_policy(MODULE_DIR / "bprime_cd_prompt_policy.yaml")
    proof = build_policy_count_proofs(policy)

    assert proof["B-prime"]["primary_count"] == 700_000
    assert proof["C"]["primary_count"] == proof["D"]["primary_count"] == 2_000_000
    assert proof["B-prime"]["acquisition_lower_bound_count"] == 840_000
    assert (
        proof["C"]["acquisition_lower_bound_count"]
        == proof["D"]["acquisition_lower_bound_count"]
        == 2_400_000
    )
    assert sum(proof["B-prime"]["cell_counts"].values()) == 700_000
    assert sum(proof["D"]["lane_counts"].values()) == 600_000
    assert proof["C"]["non_agentic_count"] == proof["D"]["non_agentic_count"]
    assert proof["paired_cd_sha256"]


def test_count_proof_labels_cell_ceils_as_lower_bounds_for_odd_bucket_splits() -> None:
    policy = _policy()
    b = policy.arms["B-prime"]
    cells = dict(b.cells)
    cells["stem"] = PromptCell(2)
    odd = replace(
        policy,
        arms=MappingProxyType(
            {**policy.arms, "B-prime": replace(b, prompt_count=102, cells=MappingProxyType(cells))}
        ),
    )

    proof = build_policy_count_proofs(odd)["B-prime"]

    assert "acquisition_count" not in proof
    assert proof["cell_acquisition_lower_bound_counts"]["stem"] == 3

    cells["stem"] = PromptCell(3)
    odd = replace(
        policy,
        arms=MappingProxyType(
            {**policy.arms, "B-prime": replace(b, prompt_count=103, cells=MappingProxyType(cells))}
        ),
    )
    selected = _select(_inventory(), odd)

    assert selected.B_prime.bucket_floors["stem"] == {"le4k": 2, "4k_16k": 1}
    assert len([row for row in selected.B_prime.reserve_rows if row.domain == "stem-science"]) == 2


def test_d_preserves_odd_agentic_floors_per_lane_and_bucket() -> None:
    policy = _policy()
    d = policy.arms["D"]
    cells = dict(d.cells)
    cells["swe-agentic-tool"] = PromptCell(9)
    odd_d = replace(
        d,
        prompt_count=289,
        cells=MappingProxyType(cells),
        lanes=MappingProxyType(dict.fromkeys(d.lanes, 3)),
    )

    selected = _select(
        _inventory(), replace(policy, arms=MappingProxyType({**policy.arms, "D": odd_d}))
    )

    assert selected.D.lane_bucket_floors == {
        "agentless-swe": {"le4k": 2, "4k_16k": 1},
        "generic-tool-replay": {"le4k": 2, "4k_16k": 1},
        "interactive-swe-replay": {"le4k": 2, "4k_16k": 1},
    }
    assert len([row for row in selected.D.reserve_rows if row.domain == "swe-agentic-tool"]) == 6
