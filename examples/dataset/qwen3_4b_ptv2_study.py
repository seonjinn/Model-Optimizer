# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Build exact one-pass A-prefix and B-balanced Qwen3-4B PTV2 study views."""

from __future__ import annotations

import argparse
import json
import os
import re
import sqlite3
import tempfile
import uuid
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Literal

import yaml
from specdec_corpus_contracts import canonical_json
from specdec_identity import ExclusionIndex, prompt_uuid
from stage_ptv23_sources import load_source_inventory, stage_source_inventory

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator, Mapping

__all__ = [
    "PTV2StudyBundle",
    "PTV2StudyError",
    "PTV2StudyPolicy",
    "PTV2StudyRecoveryError",
    "PTV2StudySourceRow",
    "PTV2StudyView",
    "StudyOccurrence",
    "iter_ptv2_staged_source_rows",
    "iter_ptv2_study_occurrences",
    "load_ptv2_study_policy",
    "select_a_repair_view",
    "select_authenticated_b_balanced_view",
    "select_ptv2_b_balanced_view",
    "select_ptv2_study_views",
]

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_REVISION = re.compile(r"^[0-9a-f]{40}$")
_CELLS = ("math", "code", "stem", "chat", "multilingual")
_BALANCED_COUNTS = {
    "math": 500_000,
    "code": 400_000,
    "stem": 500_000,
    "chat": 400_000,
    "multilingual": 200_000,
}
_REPAIR_COMPLEMENT_COUNTS = {
    "stem": 200_000,
    "ja": 125_000,
    "es": 125_000,
    "fr": 125_000,
    "it": 125_000,
    "de": 0,
}
_SEGMENT_OCCURRENCES = (1_300_000, 700_000)
_SEGMENT_STEPS = (2_540, 1_368)
_CUMULATIVE_STEPS = (2_540, 3_908)
_SEGMENT_FINAL_VALID = (32, 96)
_DECLARED_PTV2_PARQUET_SHARDS = 201
_ROOT_KEYS = frozenset(
    {
        "schema_version",
        "seed",
        "ptv2_revision",
        "repair",
        "balanced",
        "assistant_responses",
        "trainer_epochs",
        "global_batch_size",
        "segment_occurrences",
        "segment_steps",
        "cumulative_steps",
        "segment_final_valid_occurrences",
        "runtime_screen_tokens",
        "scientific_exposures",
        "conditional_scientific_tokens",
        "sequence_length",
    }
)


class PTV2StudyError(ValueError):
    """The exact one-pass PTV2 study identity or input is invalid."""


class PTV2StudyRecoveryError(PTV2StudyError):
    """An interrupted immutable index publication requires explicit recovery."""


@dataclass(frozen=True)
class PTV2StudyPolicy:
    """The approved occurrence-count and one-pass exposure policy."""

    schema_version: int
    seed: int
    ptv2_revision: str
    historical_occurrences: int
    repair_complement_occurrences: Mapping[str, int]
    balanced_occurrences: Mapping[str, int]
    multilingual_occurrences: Mapping[str, int]
    segment_occurrences: tuple[int, int]
    segment_steps: tuple[int, int]
    cumulative_steps: tuple[int, int]
    segment_final_valid_occurrences: tuple[int, int]
    runtime_screen_tokens: int
    conditional_scientific_tokens: int
    assistant_responses: Literal["source-native"]
    trainer_epochs: int
    global_batch_size: int
    sequence_length: int
    policy_sha256: str

    @property
    def total_occurrences(self) -> int:
        """The immutable 1.3M plus 700K one-pass occurrence total."""
        return sum(self.segment_occurrences)


@dataclass(frozen=True)
class PTV2StudySourceRow:
    """One authenticated source occurrence before A/B selection."""

    prompt_uuid: str
    source_identity_sha256: str
    source_row: int
    cell: str
    canonical_conversation: str
    assistant_response: str
    language: str = ""


@dataclass(frozen=True)
class StudyOccurrence:
    """One selected source occurrence, including every response identity needed for replay."""

    ordinal: int
    prompt_uuid: str
    source_identity_sha256: str
    source_row: int
    cell: str
    reuse_index: int
    conversation_sha256: str
    assistant_response_sha256: str


@dataclass(frozen=True)
class PTV2StudyView:
    """A disk-backed exact occurrence view for one PTV2 strategy."""

    strategy: Literal["A-repair", "B-balanced"]
    occurrence_count: int
    trainer_epochs: int
    unique_prompt_count: int
    cell_occurrence_counts: Mapping[str, int]
    multilingual_occurrence_counts: Mapping[str, int]
    repair_complement_counts: Mapping[str, int]
    segment_occurrence_counts: tuple[int, int]
    cell_unique_prompt_counts: Mapping[str, int]
    language_unique_prompt_counts: Mapping[str, int]
    maximum_multiplicity: int
    natural_duplicate_count: int
    constructed_repeat_count: int
    index_path: Path
    occurrence_multiplicity_sha256: str
    ordered_occurrences_sha256: str
    ordered_prompt_uuids_sha256: str
    source_response_root_sha256: str
    selection_sha256: str
    held_out_overlap_count: int
    source_capacity_counts: Mapping[str, int]


@dataclass(frozen=True)
class PTV2StudyBundle:
    """Paired occurrence views and their input identities."""

    policy_sha256: str
    source_inventory_sha256: str
    baseline_receipt_sha256: str
    held_out_receipt_sha256: str
    a_repair: PTV2StudyView
    b_balanced: PTV2StudyView

    @property
    def a_prefix(self) -> PTV2StudyView:
        """Compatibility alias for pre-repair callers; production receipts use A-repair."""
        return self.a_repair


def load_ptv2_study_policy(path: Path) -> PTV2StudyPolicy:
    """Load only the approved full-PTV2 A-prefix/B-balanced policy."""
    try:
        raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as error:
        raise PTV2StudyError(f"unable to read PTV2 study policy: {path}") from error
    root = _mapping(raw, "policy")
    _exact_keys(root, _ROOT_KEYS, "policy")
    repair = _mapping(root["repair"], "repair")
    _exact_keys(repair, frozenset({"historical", "complement"}), "repair")
    historical = _mapping(repair["historical"], "repair.historical")
    _exact_keys(
        historical,
        frozenset({"selection", "occurrences", "preserve_duplicates"}),
        "repair.historical",
    )
    balanced = _mapping(root["balanced"], "balanced")
    _exact_keys(
        balanced,
        frozenset({"selection", "occurrences", "multilingual_occurrences", "capacity_shortfall"}),
        "balanced",
    )
    counts = _positive_count_mapping(balanced["occurrences"], "balanced.occurrences")
    multilingual = _language_counts(balanced["multilingual_occurrences"])
    complement = _repair_counts(repair["complement"])
    segment_occurrences = _strict_int_tuple(root["segment_occurrences"], "segment_occurrences")
    segment_steps = _strict_int_tuple(root["segment_steps"], "segment_steps")
    cumulative_steps = _strict_int_tuple(root["cumulative_steps"], "cumulative_steps")
    segment_final_valid = _strict_int_tuple(
        root["segment_final_valid_occurrences"], "segment_final_valid_occurrences"
    )
    _positive_int(root["global_batch_size"], "global_batch_size")
    _positive_int(root["schema_version"], "schema_version")
    _positive_int(root["seed"], "seed")
    _positive_int(historical["occurrences"], "repair.historical.occurrences")
    if not isinstance(historical["preserve_duplicates"], bool):
        raise PTV2StudyError("repair.historical.preserve_duplicates must be boolean")
    _positive_int(root["trainer_epochs"], "trainer_epochs")
    _positive_int(root["runtime_screen_tokens"], "runtime_screen_tokens")
    _positive_int(root["conditional_scientific_tokens"], "conditional_scientific_tokens")
    _positive_int(root["sequence_length"], "sequence_length")
    _require_approved_policy(
        root,
        historical,
        balanced,
        counts,
        complement,
        segment_occurrences,
        segment_steps,
        cumulative_steps,
        segment_final_valid,
    )
    policy_sha256 = sha256(canonical_json(root)).hexdigest()
    return PTV2StudyPolicy(
        schema_version=1,
        seed=int(root["seed"]),
        ptv2_revision=str(root["ptv2_revision"]),
        historical_occurrences=_positive_int(
            historical["occurrences"], "repair.historical.occurrences"
        ),
        repair_complement_occurrences=MappingProxyType(complement),
        balanced_occurrences=MappingProxyType(counts),
        multilingual_occurrences=MappingProxyType(multilingual),
        segment_occurrences=(segment_occurrences[0], segment_occurrences[1]),
        segment_steps=(segment_steps[0], segment_steps[1]),
        cumulative_steps=(cumulative_steps[0], cumulative_steps[1]),
        segment_final_valid_occurrences=(segment_final_valid[0], segment_final_valid[1]),
        runtime_screen_tokens=int(root["runtime_screen_tokens"]),
        conditional_scientific_tokens=int(root["conditional_scientific_tokens"]),
        assistant_responses="source-native",
        trainer_epochs=1,
        global_batch_size=int(root["global_batch_size"]),
        sequence_length=int(root["sequence_length"]),
        policy_sha256=policy_sha256,
    )


def select_ptv2_study_views(
    source_rows: Iterable[PTV2StudySourceRow],
    *,
    policy: PTV2StudyPolicy,
    output_root: Path | None = None,
    held_out_prompt_uuids: Iterable[str] = (),
    baseline: Any | None = None,
    source_inventory: object | None = None,
    exclusions: ExclusionIndex | None = None,
) -> PTV2StudyBundle:
    """Select a paired study only when the full policy has its typed trust roots."""
    if policy.total_occurrences == 2_000_000 and (
        source_inventory is None or baseline is None or not isinstance(exclusions, ExclusionIndex)
    ):
        raise PTV2StudyError(
            "full PTV2 paired selection requires SourceInventory, BaselineAudit, and ExclusionIndex"
        )
    root = _selection_root(output_root)
    index_path = root / "ptv2-study-index.sqlite3"
    temporary_index = _prepare_unpublished_index(root, index_path)
    held_out = _held_out_set(
        exclusions.held_out if exclusions is not None else held_out_prompt_uuids
    )
    connection = sqlite3.connect(temporary_index)
    try:
        _create_schema(connection)
        _spool_source_rows(connection, source_rows, policy)
        capacities = _capacity_counts(connection)
        _insert_a_prefix(connection, policy.total_occurrences)
        _verify_baseline_prefix(connection, policy, baseline)
        _insert_b_balanced(connection, policy)
        connection.commit()
        inventory_digest = (
            getattr(source_inventory, "manifest_sha256", None)
            if source_inventory is not None
            else _source_inventory_digest(connection)
        )
        _require_digest(inventory_digest, "source inventory")
        baseline_digest = _baseline_digest(baseline)
        held_out_digest = sha256(canonical_json(sorted(held_out))).hexdigest()
        _require_digest(baseline_digest, "baseline receipt")
        _require_digest(held_out_digest, "held-out receipt")
        a_prefix = _build_view(
            connection,
            "A-repair",
            policy,
            index_path,
            held_out,
            capacities,
        )
        b_balanced = _build_view(
            connection,
            "B-balanced",
            policy,
            index_path,
            held_out,
            capacities,
        )
    finally:
        connection.close()
    _publish_unpublished_index(temporary_index, index_path)
    return PTV2StudyBundle(
        policy.policy_sha256,
        inventory_digest,
        baseline_digest,
        held_out_digest,
        a_prefix,
        b_balanced,
    )


def select_ptv2_b_balanced_view(
    source_rows: Iterable[PTV2StudySourceRow],
    *,
    policy: PTV2StudyPolicy,
    output_root: Path | None = None,
    held_out_prompt_uuids: Iterable[str] = (),
) -> PTV2StudyView:
    """Build B-balanced independently of the evolving A-repair source-order plan."""
    root = _selection_root(output_root)
    index_path = root / "ptv2-b-balanced-index.sqlite3"
    temporary_index = _prepare_unpublished_index(root, index_path)
    held_out = _held_out_set(held_out_prompt_uuids)
    connection = sqlite3.connect(temporary_index)
    try:
        _create_schema(connection)
        _spool_source_rows(connection, source_rows, policy)
        capacities = _capacity_counts(connection)
        _insert_b_balanced(connection, policy)
        connection.commit()
        view = _build_view(connection, "B-balanced", policy, index_path, held_out, capacities)
    finally:
        connection.close()
    _publish_unpublished_index(temporary_index, index_path)
    return view


def select_authenticated_b_balanced_view(
    inventory_receipt: Path,
    *,
    policy: PTV2StudyPolicy,
    exclusions: ExclusionIndex,
    output_root: Path | None = None,
) -> PTV2StudyView:
    """Production B entrypoint: derive rows and overlap state from authenticated roots only."""
    if not isinstance(exclusions, ExclusionIndex):
        raise PTV2StudyError("B-balanced production selection requires an ExclusionIndex")
    inventory = load_source_inventory(inventory_receipt)
    if inventory.staged_root is None:
        raise PTV2StudyError("B-balanced production selection requires staged SourceInventory")
    if any(source.revision != policy.ptv2_revision for source in inventory.sources):
        raise PTV2StudyError("SourceInventory revision does not match the study policy")
    return select_ptv2_b_balanced_view(
        iter_ptv2_staged_source_rows(inventory_receipt, policy=policy),
        policy=policy,
        output_root=output_root,
        held_out_prompt_uuids=exclusions.held_out,
    )


def select_a_repair_view(
    historical_rows: Iterable[PTV2StudySourceRow],
    repair_complement_rows: Iterable[PTV2StudySourceRow],
    *,
    policy: PTV2StudyPolicy,
    baseline: object,
    held_out_prompt_uuids: Iterable[str] = (),
    output_root: Path | None = None,
) -> PTV2StudyView:
    """Materialize A-repair as verified 1.3M history followed by its fixed 700K complement."""
    root = _selection_root(output_root)
    index_path = root / "ptv2-a-repair-index.sqlite3"
    temporary_index = _prepare_unpublished_index(root, index_path)
    held_out = _held_out_set(held_out_prompt_uuids)
    connection = sqlite3.connect(temporary_index)
    try:
        _create_schema(connection)
        history_count = _spool_source_rows(connection, historical_rows, policy)
        if history_count != policy.historical_occurrences:
            raise PTV2StudyError("A-repair historical stream is not exactly 1.3M occurrences")
        _spool_source_rows(connection, repair_complement_rows, policy, start_ordinal=history_count)
        capacities = _capacity_counts(connection)
        _insert_a_historical(connection, history_count)
        _verify_baseline_prefix(connection, policy, baseline)
        _insert_a_repair_complement(connection, policy, history_count, held_out)
        connection.commit()
        view = _build_view(connection, "A-repair", policy, index_path, held_out, capacities)
    finally:
        connection.close()
    _publish_unpublished_index(temporary_index, index_path)
    return view


def iter_ptv2_staged_source_rows(
    inventory_path: Path, *, policy: PTV2StudyPolicy
) -> Iterator[PTV2StudySourceRow]:
    """Stream only receipt-declared staged PTV2 Parquet rows and native assistants."""
    inventory = load_source_inventory(inventory_path)
    if inventory.staged_root is None:
        raise PTV2StudyError("B study requires an authenticated staged SourceInventory receipt")
    expected: dict[Path, Any] = {}
    ordered_paths: list[Path] = []
    for source in inventory.sources:
        for source_file in source.files:
            path = (
                inventory.staged_root
                / "sources"
                / source.repository_id
                / source.revision
                / source_file.path
            )
            expected[path] = (source, source_file)
            ordered_paths.append(path)
    if len(expected) != _DECLARED_PTV2_PARQUET_SHARDS:
        raise PTV2StudyError(
            f"PTV2 staged inventory declares {len(expected)} Parquet shards, "
            f"expected {_DECLARED_PTV2_PARQUET_SHARDS}"
        )
    if any(source.revision != policy.ptv2_revision for source in inventory.sources):
        raise PTV2StudyError("staged PTV2 SourceInventory revision does not match the study policy")
    actual = set((inventory.staged_root / "sources").rglob("*.parquet"))
    if actual != set(expected):
        raise PTV2StudyError("staged PTV2 contains undeclared or missing Parquet shards")
    try:
        import pyarrow.parquet as pq
    except ImportError as error:
        raise PTV2StudyError("pyarrow is required to read staged PTV2 rows") from error
    for path in ordered_paths:
        source, source_file = expected[path]
        _verify_file(path, source_file.bytes, source_file.sha256)
        cell, language = _normalize_source_cell(source.cell)
        identity = sha256(
            canonical_json(
                [
                    source.repository_id,
                    source.configuration,
                    source.split,
                    source.revision,
                    source_file.path,
                ]
            )
        ).hexdigest()
        parquet = pq.ParquetFile(path)
        names = set(parquet.schema_arrow.names)
        if "messages" not in names:
            raise PTV2StudyError(f"PTV2 shard has no messages column: {path}")
        source_row = 0
        columns = ["messages"] + (["tools"] if "tools" in names else [])
        for batch in parquet.iter_batches(columns=columns, batch_size=8192):
            for offset, record in enumerate(batch.to_pylist()):
                messages = record["messages"]
                if isinstance(messages, str):
                    messages = json.loads(messages)
                if not isinstance(messages, list) or not all(
                    isinstance(item, dict) for item in messages
                ):
                    raise PTV2StudyError("PTV2 messages must be a list of mappings")
                assistants = [item for item in messages if item.get("role") == "assistant"]
                if not assistants:
                    raise PTV2StudyError("PTV2 row has no source-native assistant response")
                response = canonical_json(assistants[-1]).decode("utf-8")
                conversation = canonical_json({"messages": messages}).decode("utf-8")
                tools = record.get("tools")
                if isinstance(tools, str):
                    tools = json.loads(tools)
                if tools is not None and (
                    not isinstance(tools, list) or not all(isinstance(item, dict) for item in tools)
                ):
                    raise PTV2StudyError("PTV2 tools must be a list of mappings")
                prompt_messages = [
                    item for item in messages if item.get("role") in {"system", "developer", "user"}
                ]
                if not prompt_messages:
                    raise PTV2StudyError("PTV2 row has no prompt-bearing message")
                yield PTV2StudySourceRow(
                    prompt_uuid=prompt_uuid(prompt_messages, tools),
                    source_identity_sha256=identity,
                    source_row=source_row + offset,
                    cell=cell,
                    canonical_conversation=conversation,
                    assistant_response=response,
                    language=language,
                )
            source_row += batch.num_rows


def _verify_file(path: Path, expected_bytes: int, expected_sha256: str) -> None:
    if path.is_symlink() or not path.is_file() or path.stat().st_size != expected_bytes:
        raise PTV2StudyError(f"staged PTV2 shard does not match inventory: {path}")
    digest = sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    if digest.hexdigest() != expected_sha256:
        raise PTV2StudyError(f"staged PTV2 shard digest mismatch: {path}")


def _normalize_source_cell(cell: str) -> tuple[str, str]:
    if cell.startswith("multilingual_"):
        return "multilingual", cell.removeprefix("multilingual_")
    if cell in _CELLS:
        return cell, ""
    raise PTV2StudyError(f"unapproved PTV2 source cell: {cell}")


def main() -> int:
    """Build the authenticated B-balanced view locally; cluster launch is deliberately separate."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", type=Path, required=True)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--source-inventory", type=Path)
    source.add_argument("--source-plan", type=Path)
    parser.add_argument("--source-cache", type=Path)
    parser.add_argument("--durable-root", type=Path)
    parser.add_argument("--scratch-root", type=Path)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    if args.source_plan is not None:
        if args.source_cache is None or args.durable_root is None or args.scratch_root is None:
            parser.error(
                "--source-plan requires --source-cache, --durable-root, and --scratch-root"
            )
        staged = stage_source_inventory(
            load_source_inventory(args.source_plan),
            durable_root=args.durable_root,
            scratch_root=args.scratch_root,
            local_source_root=args.source_cache,
        )
        if staged.staged_root is None:
            raise PTV2StudyError("Task3 source staging did not publish a staged inventory receipt")
        inventory_path = staged.staged_root / "SOURCE_INVENTORY.json"
    else:
        inventory_path = args.source_inventory
        if inventory_path is None:
            raise AssertionError("argparse requires one source input")
    policy = load_ptv2_study_policy(args.policy)
    view = select_ptv2_b_balanced_view(
        iter_ptv2_staged_source_rows(inventory_path, policy=policy),
        policy=policy,
        output_root=args.output_root,
    )
    print(
        json.dumps(
            {
                "strategy": view.strategy,
                "index_path": str(view.index_path),
                "selection_sha256": view.selection_sha256,
            },
            sort_keys=True,
        )
    )
    return 0


def iter_ptv2_study_occurrences(view: PTV2StudyView) -> Iterator[StudyOccurrence]:
    """Iterate the persisted occurrence order without materializing production rows."""
    connection = sqlite3.connect(view.index_path)
    try:
        cursor = connection.execute(
            "SELECT ordinal,prompt_uuid,source_identity_sha256,source_row,cell,reuse_index,"
            "conversation_sha256,assistant_response_sha256 FROM occurrences "
            "WHERE strategy=? ORDER BY ordinal",
            (view.strategy,),
        )
        for row in cursor:
            yield StudyOccurrence(*row)
    finally:
        connection.close()


def _require_approved_policy(
    root: Mapping[str, Any],
    historical: Mapping[str, Any],
    balanced: Mapping[str, Any],
    counts: Mapping[str, int],
    complement: Mapping[str, int],
    segment_occurrences: tuple[int, ...],
    segment_steps: tuple[int, ...],
    cumulative_steps: tuple[int, ...],
    segment_final_valid: tuple[int, ...],
) -> None:
    if (
        _positive_int(root["schema_version"], "schema_version") != 1
        or _positive_int(root["seed"], "seed") != 20_260_822
    ):
        raise PTV2StudyError("policy schema_version and seed must be approved values")
    revision = root["ptv2_revision"]
    if not isinstance(revision, str) or _REVISION.fullmatch(revision) is None:
        raise PTV2StudyError("ptv2_revision must be an exact commit")
    if revision != "5c89e01dd720ae0f4058445ed49c5fb68a03c76e":
        raise PTV2StudyError("ptv2_revision is not the approved PTV2 revision")
    if historical != {
        "selection": "source-order",
        "occurrences": 1_300_000,
        "preserve_duplicates": True,
    }:
        raise PTV2StudyError(
            "repair historical must be source-order take(1300000) with duplicates preserved"
        )
    if complement != _REPAIR_COMPLEMENT_COUNTS:
        raise PTV2StudyError("repair complement must be STEM200K plus JA/ES/FR/IT125K and DE0")
    if balanced.get("selection") != "seeded-within-cell" or counts != _BALANCED_COUNTS:
        raise PTV2StudyError("balanced occurrence counts do not match approved semantic quotas")
    if balanced.get("capacity_shortfall") != "deterministic-within-cell-cycle":
        raise PTV2StudyError("balanced capacity shortfall must cycle within its cell")
    if root["assistant_responses"] != "source-native":
        raise PTV2StudyError("assistant responses must be source-native")
    if _positive_int(root["trainer_epochs"], "trainer_epochs") != 1:
        raise PTV2StudyError("trainer_epochs must be exactly one")
    if _positive_int(root["global_batch_size"], "global_batch_size") != 512:
        raise PTV2StudyError("global_batch_size must be exactly 512")
    if segment_occurrences != _SEGMENT_OCCURRENCES:
        raise PTV2StudyError("PTV2 must have exact 1.3M plus 700K segments")
    if segment_steps != _SEGMENT_STEPS or cumulative_steps != _CUMULATIVE_STEPS:
        raise PTV2StudyError("PTV2 must use the exact 2540 plus 1368 segment schedule")
    if segment_final_valid != _SEGMENT_FINAL_VALID:
        raise PTV2StudyError("PTV2 segment final valid occurrence counts must be 32 and 96")
    if _positive_int(root["runtime_screen_tokens"], "runtime_screen_tokens") != 64_000_000:
        raise PTV2StudyError("runtime_screen_tokens must be exactly 64000000")
    if root["scientific_exposures"] != ["one-pass"]:
        raise PTV2StudyError("scientific exposures must be one-pass only")
    if (
        _positive_int(root["conditional_scientific_tokens"], "conditional_scientific_tokens")
        != 256_000_000
    ):
        raise PTV2StudyError("conditional scientific boundary must be 256000000")
    if _positive_int(root["sequence_length"], "sequence_length") != 4_096:
        raise PTV2StudyError("sequence_length must be exactly 4096")


def _selection_root(output_root: Path | None) -> Path:
    root = (
        Path(output_root)
        if output_root is not None
        else Path(tempfile.mkdtemp(prefix="ptv2-study-"))
    )
    root.mkdir(parents=True, exist_ok=True)
    if not root.is_dir() or root.is_symlink():
        raise PTV2StudyError("selection output root must be a regular directory")
    return root


def _prepare_unpublished_index(root: Path, destination: Path) -> Path:
    """Create a private no-follow SQLite target without reserving the final name."""
    if os.path.lexists(destination):
        raise FileExistsError(f"immutable B index already exists: {destination}")
    partials = tuple(root.glob(f".{destination.name}.partial-*"))
    if partials:
        raise PTV2StudyRecoveryError(
            f"unpublished PTV2 index requires recovery before retry: {partials[0]}"
        )
    temporary = root / f".{destination.name}.partial-{uuid.uuid4().hex}"
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(temporary, flags, 0o600)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    return temporary


def _publish_unpublished_index(temporary: Path, destination: Path) -> None:
    """Durably link an fsynced private index into place without overwriting receipts."""
    temporary_stat = os.lstat(temporary)
    if not os.path.isfile(temporary) or os.path.islink(temporary) or temporary_stat.st_nlink != 1:
        raise PTV2StudyRecoveryError(
            f"unpublished PTV2 index is not a private regular file: {temporary}"
        )
    _fsync_file(temporary)
    try:
        os.link(temporary, destination, follow_symlinks=False)
    except FileExistsError as error:
        raise FileExistsError(f"immutable B index already exists: {destination}") from error
    _fsync_directory(destination.parent)
    temporary.unlink()
    _fsync_directory(destination.parent)


def _create_schema(connection: sqlite3.Connection) -> None:
    connection.executescript(
        """
        PRAGMA journal_mode=DELETE;
        PRAGMA synchronous=FULL;
        CREATE TABLE source_rows(
          source_ordinal INTEGER PRIMARY KEY, prompt_uuid TEXT NOT NULL,
          source_identity_sha256 TEXT NOT NULL, source_row INTEGER NOT NULL,
          cell TEXT NOT NULL, language TEXT NOT NULL, canonical_conversation TEXT NOT NULL,
          assistant_response TEXT NOT NULL, conversation_sha256 TEXT NOT NULL,
          assistant_response_sha256 TEXT NOT NULL, rank TEXT NOT NULL
        );
        CREATE INDEX source_rows_cell_rank ON source_rows(
          cell, language, rank, source_identity_sha256, source_row, prompt_uuid
        );
        CREATE TABLE occurrences(
          strategy TEXT NOT NULL, ordinal INTEGER NOT NULL, prompt_uuid TEXT NOT NULL,
          source_identity_sha256 TEXT NOT NULL, source_row INTEGER NOT NULL,
          cell TEXT NOT NULL, reuse_index INTEGER NOT NULL, conversation_sha256 TEXT NOT NULL,
          assistant_response_sha256 TEXT NOT NULL, PRIMARY KEY(strategy, ordinal)
        );
        CREATE INDEX occurrences_strategy_uuid ON occurrences(strategy, prompt_uuid);
        CREATE TABLE historical_uuids(prompt_uuid TEXT PRIMARY KEY);
        """
    )


def _spool_source_rows(
    connection: sqlite3.Connection,
    source_rows: Iterable[PTV2StudySourceRow],
    policy: PTV2StudyPolicy,
    *,
    start_ordinal: int = 0,
) -> int:
    next_ordinal = start_ordinal
    for ordinal, row in enumerate(source_rows, start=start_ordinal):
        if not isinstance(row, PTV2StudySourceRow):
            raise PTV2StudyError("source rows must be PTV2StudySourceRow values")
        _validate_source_row(row)
        if (
            policy.total_occurrences == 2_000_000
            and row.cell == "multilingual"
            and row.language not in policy.multilingual_occurrences
        ):
            raise PTV2StudyError("multilingual source row has no approved language")
        conversation_sha256 = sha256(row.canonical_conversation.encode("utf-8")).hexdigest()
        response_sha256 = sha256(row.assistant_response.encode("utf-8")).hexdigest()
        rank = sha256(
            (
                policy.policy_sha256
                + row.cell
                + row.language
                + row.source_identity_sha256
                + str(row.source_row)
                + row.prompt_uuid
            ).encode("utf-8")
        ).hexdigest()
        connection.execute(
            "INSERT INTO source_rows VALUES(?,?,?,?,?,?,?,?,?,?,?)",
            (
                ordinal,
                row.prompt_uuid,
                row.source_identity_sha256,
                row.source_row,
                row.cell,
                row.language,
                row.canonical_conversation,
                row.assistant_response,
                conversation_sha256,
                response_sha256,
                rank,
            ),
        )
        next_ordinal = ordinal + 1
    if connection.execute("SELECT count(*) FROM source_rows").fetchone()[0] == 0:
        raise PTV2StudyError("authenticated source inventory is empty")
    return next_ordinal


def _validate_source_row(row: PTV2StudySourceRow) -> None:
    _require_digest(row.prompt_uuid, "prompt UUID")
    _require_digest(row.source_identity_sha256, "source identity")
    if (
        isinstance(row.source_row, bool)
        or not isinstance(row.source_row, int)
        or row.source_row < 0
    ):
        raise PTV2StudyError("source row must be a nonnegative integer")
    if row.cell not in _CELLS:
        raise PTV2StudyError(f"unknown cell: {row.cell}")
    if not isinstance(row.canonical_conversation, str) or not row.canonical_conversation:
        raise PTV2StudyError("canonical source conversation is required")
    try:
        conversation = json.loads(row.canonical_conversation)
    except json.JSONDecodeError as error:
        raise PTV2StudyError("canonical source conversation is invalid JSON") from error
    if canonical_json(conversation).decode("utf-8") != row.canonical_conversation:
        raise PTV2StudyError("canonical source conversation is not canonically encoded")
    if not isinstance(row.assistant_response, str) or not row.assistant_response:
        raise PTV2StudyError("source-native assistant response is required")


def _capacity_counts(connection: sqlite3.Connection) -> Mapping[str, int]:
    counts = {
        str(cell): int(count)
        for cell, count in connection.execute("SELECT cell,count(*) FROM source_rows GROUP BY cell")
    }
    return MappingProxyType({cell: counts.get(cell, 0) for cell in _CELLS})


def _insert_a_prefix(connection: sqlite3.Connection, count: int) -> None:
    available = int(connection.execute("SELECT count(*) FROM source_rows").fetchone()[0])
    if available < count:
        raise PTV2StudyError(f"A-repair source stream is shorter than {count} occurrences")
    connection.execute(
        "INSERT INTO occurrences "
        "SELECT 'A-repair',source_ordinal,prompt_uuid,source_identity_sha256,source_row,cell,0,"
        "conversation_sha256,assistant_response_sha256 FROM source_rows "
        "ORDER BY source_ordinal LIMIT ?",
        (count,),
    )


def _insert_a_historical(connection: sqlite3.Connection, count: int) -> None:
    connection.execute(
        "INSERT INTO occurrences "
        "SELECT 'A-repair',source_ordinal,prompt_uuid,source_identity_sha256,source_row,cell,0,"
        "conversation_sha256,assistant_response_sha256 FROM source_rows "
        "WHERE source_ordinal<? ORDER BY source_ordinal",
        (count,),
    )


def _insert_a_repair_complement(
    connection: sqlite3.Connection,
    policy: PTV2StudyPolicy,
    history_count: int,
    held_out: set[str],
) -> None:
    """Append only frozen source-order complement rows meeting the repair quota contract."""
    quotas = dict(policy.repair_complement_occurrences)
    selected = dict.fromkeys(quotas, 0)
    connection.execute(
        "INSERT OR IGNORE INTO historical_uuids SELECT prompt_uuid FROM source_rows WHERE source_ordinal<?",
        (history_count,),
    )
    ordinal = history_count
    rows = connection.execute(
        "SELECT source_ordinal,prompt_uuid,source_identity_sha256,source_row,cell,language,"
        "conversation_sha256,assistant_response_sha256 FROM source_rows "
        "WHERE source_ordinal>=? ORDER BY source_ordinal",
        (history_count,),
    )
    for (
        _source_ordinal,
        uuid_value,
        identity,
        source_row,
        cell,
        language,
        conversation,
        response,
    ) in rows:
        key = "stem" if cell == "stem" else language if cell == "multilingual" else ""
        if key not in quotas or quotas[key] == 0 or selected[key] >= quotas[key]:
            continue
        if (
            connection.execute(
                "SELECT 1 FROM historical_uuids WHERE prompt_uuid=?", (uuid_value,)
            ).fetchone()
            is not None
        ):
            raise PTV2StudyError("A-repair complement overlaps the historical UUID set")
        if uuid_value in held_out:
            raise PTV2StudyError("A-repair complement overlaps evaluator-held-out UUIDs")
        connection.execute(
            "INSERT INTO occurrences VALUES(?,?,?,?,?,?,?,?,?)",
            (
                "A-repair",
                ordinal,
                uuid_value,
                identity,
                source_row,
                cell,
                0,
                conversation,
                response,
            ),
        )
        selected[key] += 1
        ordinal += 1
    if selected != quotas:
        raise PTV2StudyError(
            f"A-repair complement counts do not match immutable policy: {selected}"
        )


def _insert_b_balanced(connection: sqlite3.Connection, policy: PTV2StudyPolicy) -> None:
    ordinal = 0
    for cell in _CELLS:
        quota = policy.balanced_occurrences[cell]
        if cell != "multilingual" or policy.total_occurrences != 2_000_000:
            ordinal = _insert_b_cell(connection, cell, quota, ordinal)
            continue
        if sum(policy.multilingual_occurrences.values()) != quota:
            raise PTV2StudyError(
                "multilingual language quotas must equal the B-balanced 200K allocation"
            )
        for language, language_quota in policy.multilingual_occurrences.items():
            ordinal = _insert_b_cell(connection, cell, language_quota, ordinal, language=language)
    if ordinal != sum(policy.balanced_occurrences.values()):
        raise AssertionError("B-balanced occurrence arithmetic does not reconcile")


def _insert_b_cell(
    connection: sqlite3.Connection,
    cell: str,
    quota: int,
    ordinal: int,
    *,
    language: str | None = None,
) -> int:
    """Append one independently ranked source cell using deterministic round-robin cycling."""
    predicate = "cell=?" if language is None else "cell=? AND language=?"
    parameters: tuple[str, ...] = (cell,) if language is None else (cell, language)
    capacity = int(
        connection.execute(
            f"SELECT count(*) FROM source_rows WHERE {predicate}", parameters
        ).fetchone()[0]
    )
    label = cell if language is None else f"{cell}:{language}"
    if capacity == 0:
        raise PTV2StudyError(f"B-balanced cell has no approved source occurrences: {label}")
    base, remainder = divmod(quota, capacity)
    for reuse_index in range(base + int(remainder > 0)):
        limit = capacity if reuse_index < base else remainder
        cursor = connection.execute(
            "SELECT prompt_uuid,source_identity_sha256,source_row,cell,conversation_sha256,"
            f"assistant_response_sha256 FROM source_rows WHERE {predicate} "
            "ORDER BY rank,source_identity_sha256,source_row,prompt_uuid LIMIT ?",
            (*parameters, limit),
        )
        for row in cursor:
            connection.execute(
                "INSERT INTO occurrences VALUES(?,?,?,?,?,?,?,?,?)",
                ("B-balanced", ordinal, *row[:4], reuse_index, *row[4:]),
            )
            ordinal += 1
    return ordinal


def _verify_baseline_prefix(
    connection: sqlite3.Connection, policy: PTV2StudyPolicy, baseline: Any | None
) -> None:
    if baseline is None:
        return
    expected_ids = getattr(baseline, "occurrence_prompt_ids", None)
    expected_digest = getattr(baseline, "occurrence_prompt_ids_sha256", None)
    expected_count = getattr(baseline, "occurrence_count", None)
    if expected_count != policy.historical_occurrences or not isinstance(expected_ids, tuple):
        raise PTV2StudyError("BaselineAudit must prove the exact historical segment")
    actual_digest = sha256()
    actual_digest.update(b"[")
    matches = True
    count = 0
    for count, (actual_id,) in enumerate(
        connection.execute(
            "SELECT prompt_uuid FROM occurrences WHERE strategy='A-repair' ORDER BY ordinal LIMIT ?",
            (policy.historical_occurrences,),
        ),
        start=1,
    ):
        if count > 1:
            actual_digest.update(b",")
        actual_digest.update(canonical_json(actual_id))
        matches = matches and actual_id == expected_ids[count - 1]
    actual_digest.update(b"]")
    if count != len(expected_ids) or not matches or actual_digest.hexdigest() != expected_digest:
        raise PTV2StudyError("A-repair first 1.3M does not match BaselineAudit")
    if (
        policy.historical_occurrences == 1_300_000
        and getattr(baseline, "unique_prompt_count", None) != 931_363
    ):
        raise PTV2StudyError("BaselineAudit unique UUID count is not the historical 931363")


def _build_view(
    connection: sqlite3.Connection,
    strategy: Literal["A-repair", "B-balanced"],
    policy: PTV2StudyPolicy,
    index_path: Path,
    held_out: set[str],
    capacities: Mapping[str, int],
) -> PTV2StudyView:
    rows = connection.execute(
        "SELECT ordinal,prompt_uuid,source_identity_sha256,source_row,cell,reuse_index,"
        "conversation_sha256,assistant_response_sha256 FROM occurrences WHERE strategy=? ORDER BY ordinal",
        (strategy,),
    )
    occurrence_digest = sha256()
    prompt_digest = sha256()
    response_digest = sha256()
    count = constructed = overlap = 0
    for row in rows:
        occurrence = StudyOccurrence(*row)
        occurrence_digest.update(canonical_json(list(row)))
        occurrence_digest.update(b"\n")
        prompt_digest.update(canonical_json(occurrence.prompt_uuid))
        prompt_digest.update(b"\n")
        response_digest.update(
            canonical_json(
                [
                    occurrence.source_identity_sha256,
                    occurrence.source_row,
                    occurrence.conversation_sha256,
                    occurrence.assistant_response_sha256,
                ]
            )
        )
        response_digest.update(b"\n")
        count += 1
        constructed += int(occurrence.reuse_index > 0)
        overlap += int(occurrence.prompt_uuid in held_out)
    unique = int(
        connection.execute(
            "SELECT count(DISTINCT prompt_uuid) FROM occurrences WHERE strategy=?", (strategy,)
        ).fetchone()[0]
    )
    natural = count - unique - constructed
    multiplicity_digest = sha256()
    for row in connection.execute(
        "SELECT prompt_uuid,source_identity_sha256,source_row,count(*) FROM occurrences "
        "WHERE strategy=? GROUP BY prompt_uuid,source_identity_sha256,source_row "
        "ORDER BY prompt_uuid,source_identity_sha256,source_row",
        (strategy,),
    ):
        multiplicity_digest.update(canonical_json(list(row)))
        multiplicity_digest.update(b"\n")
    cell_counts = {
        str(cell): int(value)
        for cell, value in connection.execute(
            "SELECT cell,count(*) FROM occurrences WHERE strategy=? GROUP BY cell", (strategy,)
        )
    }
    complete_cells = MappingProxyType({cell: cell_counts.get(cell, 0) for cell in _CELLS})
    language_counts = {
        str(language): int(value)
        for language, value in connection.execute(
            "SELECT source_rows.language,count(*) FROM occurrences "
            "JOIN source_rows ON occurrences.source_identity_sha256=source_rows.source_identity_sha256 "
            "AND occurrences.source_row=source_rows.source_row "
            "WHERE occurrences.strategy=? AND occurrences.cell='multilingual' "
            "GROUP BY source_rows.language",
            (strategy,),
        )
    }
    complete_languages = MappingProxyType(
        {language: language_counts.get(language, 0) for language in policy.multilingual_occurrences}
    )
    cell_unique_counts = {
        str(cell): int(value)
        for cell, value in connection.execute(
            "SELECT cell,count(DISTINCT prompt_uuid) FROM occurrences WHERE strategy=? GROUP BY cell",
            (strategy,),
        )
    }
    language_unique_counts = {
        str(language): int(value)
        for language, value in connection.execute(
            "SELECT source_rows.language,count(DISTINCT occurrences.prompt_uuid) FROM occurrences "
            "JOIN source_rows ON occurrences.source_identity_sha256=source_rows.source_identity_sha256 "
            "AND occurrences.source_row=source_rows.source_row "
            "WHERE occurrences.strategy=? AND occurrences.cell='multilingual' "
            "GROUP BY source_rows.language",
            (strategy,),
        )
    }
    maximum_multiplicity = int(
        connection.execute(
            "SELECT coalesce(max(multiplicity),0) FROM ("
            "SELECT count(*) AS multiplicity FROM occurrences WHERE strategy=? "
            "GROUP BY prompt_uuid,source_identity_sha256,source_row)",
            (strategy,),
        ).fetchone()[0]
    )
    selection = {
        "strategy": strategy,
        "policy_sha256": policy.policy_sha256,
        "occurrence_count": count,
        "unique_prompt_count": unique,
        "cell_occurrence_counts": dict(complete_cells),
        "multilingual_occurrence_counts": dict(complete_languages),
        "ordered_occurrences_sha256": occurrence_digest.hexdigest(),
        "ordered_prompt_uuids_sha256": prompt_digest.hexdigest(),
        "source_response_root_sha256": response_digest.hexdigest(),
        "occurrence_multiplicity_sha256": multiplicity_digest.hexdigest(),
    }
    return PTV2StudyView(
        strategy,
        count,
        policy.trainer_epochs,
        unique,
        complete_cells,
        complete_languages,
        MappingProxyType(
            dict(policy.repair_complement_occurrences) if strategy == "A-repair" else {}
        ),
        policy.segment_occurrences,
        MappingProxyType({cell: cell_unique_counts.get(cell, 0) for cell in _CELLS}),
        MappingProxyType(
            {
                language: language_unique_counts.get(language, 0)
                for language in policy.multilingual_occurrences
            }
        ),
        maximum_multiplicity,
        natural,
        constructed,
        index_path,
        multiplicity_digest.hexdigest(),
        occurrence_digest.hexdigest(),
        prompt_digest.hexdigest(),
        response_digest.hexdigest(),
        sha256(canonical_json(selection)).hexdigest(),
        overlap,
        capacities,
    )


def _source_inventory_digest(connection: sqlite3.Connection) -> str:
    digest = sha256()
    for row in connection.execute(
        "SELECT source_ordinal,prompt_uuid,source_identity_sha256,source_row,cell,conversation_sha256,"
        "assistant_response_sha256 FROM source_rows ORDER BY source_ordinal"
    ):
        digest.update(canonical_json(list(row)))
        digest.update(b"\n")
    return digest.hexdigest()


def _baseline_digest(baseline: Any | None) -> str:
    if baseline is None:
        return sha256(canonical_json({"fixture_baseline": None})).hexdigest()
    return sha256(canonical_json(getattr(baseline, "occurrence_prompt_ids_sha256", ""))).hexdigest()


def _held_out_set(value: Iterable[str]) -> set[str]:
    if hasattr(value, "held_out"):
        value = getattr(value, "held_out")
    result = set(value)
    for held_out_uuid in result:
        _require_digest(held_out_uuid, "held-out prompt UUID")
    return result


def _mapping(value: object, name: str) -> Mapping[str, Any]:
    if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
        raise PTV2StudyError(f"{name} must be a mapping with string keys")
    return value


def _exact_keys(value: Mapping[str, Any], expected: frozenset[str], name: str) -> None:
    if set(value) != expected:
        raise PTV2StudyError(f"{name} has unknown or missing keys: {sorted(set(value) ^ expected)}")


def _positive_int(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise PTV2StudyError(f"{name} must be a positive integer")
    return value


def _positive_count_mapping(value: object, name: str) -> dict[str, int]:
    mapping = _mapping(value, name)
    if set(mapping) != set(_CELLS):
        raise PTV2StudyError(f"{name} must contain exactly the approved cells")
    return {cell: _positive_int(mapping[cell], f"{name}.{cell}") for cell in _CELLS}


def _language_counts(value: object) -> dict[str, int]:
    mapping = _mapping(value, "multilingual")
    languages = ("de", "ja", "es", "fr", "it")
    if set(mapping) != set(languages):
        raise PTV2StudyError("multilingual must contain exactly de, ja, es, fr, and it")
    return {
        language: _positive_int(mapping[language], f"multilingual.{language}")
        for language in languages
    }


def _repair_counts(value: object) -> dict[str, int]:
    mapping = _mapping(value, "repair.complement")
    if set(mapping) != set(_REPAIR_COMPLEMENT_COUNTS):
        raise PTV2StudyError("repair complement must contain STEM, DE, JA, ES, FR, and IT")
    result: dict[str, int] = {}
    for language in _REPAIR_COMPLEMENT_COUNTS:
        item = mapping[language]
        if isinstance(item, bool) or not isinstance(item, int) or item < 0:
            raise PTV2StudyError(f"repair.complement.{language} must be a nonnegative integer")
        result[language] = item
    return result


def _strict_int_tuple(value: object, name: str) -> tuple[int, ...]:
    if not isinstance(value, list):
        raise PTV2StudyError(f"{name} must be a list of positive integers")
    return tuple(_positive_int(item, f"{name}[{index}]") for index, item in enumerate(value))


def _ceil_div(value: int, divisor: int) -> int:
    return (value + divisor - 1) // divisor


def _require_digest(value: object, name: str) -> None:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise PTV2StudyError(f"{name} must be an exact lowercase SHA-256")


def _fsync_file(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


if __name__ == "__main__":
    raise SystemExit(main())
