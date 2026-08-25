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

"""Stage exact assistant-loss-token exposure views from promoted responses."""

from __future__ import annotations

import heapq
import json
import os
import re
import shutil
import sqlite3
import stat
import tempfile
import time
import uuid
from collections import Counter
from collections.abc import Iterator, Mapping
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import suppress
from dataclasses import asdict, dataclass
from fractions import Fraction
from hashlib import sha256
from itertools import islice, pairwise
from multiprocessing import get_context
from pathlib import Path
from types import MappingProxyType
from typing import Any, Literal, cast

from specdec_corpus_contracts import canonical_json
from stage_ptv23_sources import _rename_no_replace
from trajectory_schema import TrajectoryValidationError, validate_trajectory

__all__ = [
    "ExposureView",
    "ExposureViewError",
    "PTV2OnePassCorpus",
    "PTV2StudyExposureViews",
    "PTV2TokenizedRecoveryError",
    "PairedExposureViews",
    "build_exposure_views",
    "build_paired_exposure_views",
    "build_ptv2_study_exposures",
    "derive_ptv2_historical_corpus",
    "derive_ptv2_one_pass_corpus",
]

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_SAFE_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
_PRODUCTION_BOUNDARIES = (256_000_000, 1_000_000_000)
_RUNTIME_SCREEN_TOKENS = 64_000_000
_EXPOSURE_POLICY_SHA256 = sha256(
    canonical_json(
        {
            "production_assistant_token_boundaries": list(_PRODUCTION_BOUNDARIES),
            "runtime_screen_assistant_tokens": _RUNTIME_SCREEN_TOKENS,
        }
    )
).hexdigest()


class ExposureViewError(ValueError):
    """An exact-token staging identity or record contract is invalid."""


class PTV2TokenizedRecoveryError(ExposureViewError):
    """A preserved Task 7 partial requires an explicit operator recovery action."""


@dataclass(frozen=True)
class ExposureView:
    """Authenticated, consumable row order for one assistant-token boundary."""

    schema_version: int
    name: str
    arm: str
    response_corpus_sha256: str
    tokenizer_sha256: str
    chat_template_sha256: str
    selection_sha256: str
    training_sequence_length: int
    purpose: str
    policy_sha256: str
    source_policy_sha256: str
    seed: int
    assistant_tokens: int
    full_prompt_assistant_tokens: int
    row_count: int
    unique_prompt_count: int
    repeated_prompt_count: int
    completed_epochs: int
    records_path: str
    records_bytes: int
    records_sha256: str
    row_order_sha256: str
    input_ids_sha256: str
    loss_masks_sha256: str
    cumulative_tokens_sha256: str
    paired_selection_bucket_sha256: str
    task5_paired_selection_bucket_sha256: str
    task6_completion_sha256: str
    final_paired_response_sha256: str
    resume_fingerprint: str
    receipt_path: str
    receipt_sha256: str


@dataclass(frozen=True)
class PairedExposureViews:
    """C/D exact-token views bound to their common selection and bucket proof."""

    C: Mapping[str, ExposureView]
    D: Mapping[str, ExposureView]
    common_boundary: int
    C_full_prompt_assistant_tokens: int
    D_full_prompt_assistant_tokens: int
    paired_selection_bucket_sha256: str
    final_paired_response_sha256: str
    C_task6_completion_sha256: str
    D_task6_completion_sha256: str
    source_policy_sha256: str
    exposure_policy_sha256: str
    receipt_path: str
    receipt_sha256: str


@dataclass(frozen=True)
class PTV2OnePassCorpus:
    """Authenticated token totals for one immutable PTV2 base arm."""

    strategy: Literal["A-repair", "B-balanced", "H-historical-cyclic"]
    occurrence_count: int
    trainer_epochs: int
    assistant_tokens: int
    tokenizer_sha256: str
    chat_template_sha256: str
    assistant_loss_target_sha256: str
    training_config_sha256: str
    source_response_root_sha256: str
    ordered_occurrences_sha256: str
    selection_sha256: str = "0" * 64
    unique_prompt_count: int = 2_000_000
    natural_duplicate_count: int = 0
    constructed_repeat_count: int = 0
    serialized_tokens: int = 0
    # This is a lower bound until a concrete packer receipt is attached.  It is
    # deliberately not described as a measured packed-sequence count.
    packed_sequence_lower_bound: int = 0
    milestone_occurrences: tuple[int, ...] = (500_224, 1_000_448, 1_300_000, 2_000_000)
    milestone_steps: tuple[int, ...] = (977, 1_954, 2_540, 3_908)
    segment_occurrences: tuple[int, int] = (1_300_000, 700_000)
    segment_steps: tuple[int, int] = (2_540, 1_368)
    cumulative_segment_steps: tuple[int, int] = (2_540, 3_908)
    segment_final_valid_occurrences: tuple[int, int] = (32, 96)
    tokenized_path: str = ""
    tokenized_sha256: str = "0" * 64
    receipt_path: str = ""
    receipt_sha256: str = "0" * 64
    historical_parent_receipt_sha256: str | None = None
    historical_parent_tokenized_sha256: str | None = None
    historical_prefix_record_stream_sha256: str | None = None


@dataclass(frozen=True)
class PTV2StudyExposureViews:
    """One-pass A/B and authenticated cyclic-H scientific receipt boundaries."""

    runtime_screen_tokens: int
    scientific_tokens: int
    prefix_u2m: int
    balanced_u2m: int
    paired_scientific_reached: bool
    runtime_artifacts: Mapping[str, str] = MappingProxyType({})
    scientific_artifacts: Mapping[str, str] = MappingProxyType({})
    historical_h1m: int | None = None
    completion_receipt_path: str | None = None
    completion_receipt_sha256: str | None = None


_TASK8_TOKENIZER: Any | None = None


@dataclass(frozen=True)
class _Task8TokenRange:
    index: int
    start: int
    stop: int
    selection_index: Path
    strategy: str
    sequence_length: int
    spool_path: Path


@dataclass(frozen=True)
class _PTV2ExposureSourceRow:
    ordinal: int
    prompt_uuid: str
    source_identity_sha256: str
    source_row: int
    cell: str
    language: str
    reuse_index: int
    assistant_tokens: int


@dataclass
class _VerifiedPTV2ExposureSource:
    rows: list[_PTV2ExposureSourceRow]
    weights: dict[tuple[str, str], int]
    base_root: str
    effective_workers: int
    database_path: Path
    receipt: Mapping[str, Any]
    temporary: tempfile.TemporaryDirectory[str]

    def close(self) -> None:
        self.temporary.cleanup()


@dataclass(frozen=True)
class _Task8TokenResult:
    index: int
    start: int
    stop: int
    row_count: int
    spool_path: Path
    spool_bytes: int
    spool_sha256: str
    elapsed_seconds: float
    worker_pid: int


@dataclass(frozen=True)
class _Task8StagedView:
    source: Any
    index_path: Path

    def __getattr__(self, name: str) -> Any:
        return getattr(self.source, name)


def resolve_task8_worker_count(
    requested_workers: int,
    *,
    occurrence_count: int,
    declared_ranges: int = 201,
    environ: Mapping[str, str] | None = None,
) -> tuple[int, int]:
    """Bound Task8 tokenization by allocation and deterministic ordinal ranges."""
    if isinstance(requested_workers, bool) or requested_workers < 1:
        raise ExposureViewError("Task8 workers must be a positive integer")
    if occurrence_count < 1 or declared_ranges < 1 or declared_ranges > 201:
        raise ExposureViewError("Task8 ordinal range contract is invalid")
    environment = os.environ if environ is None else environ
    try:
        allocated = int(environment.get("SLURM_CPUS_PER_TASK", os.cpu_count() or 1))
    except ValueError as error:
        raise ExposureViewError("SLURM_CPUS_PER_TASK must be a positive integer") from error
    if allocated < 1:
        raise ExposureViewError("Task8 requires a positive CPU allocation")
    return min(requested_workers, allocated, declared_ranges, occurrence_count, 96), allocated


def _task8_ranges(count: int, declared_ranges: int) -> tuple[tuple[int, int], ...]:
    ranges = min(count, declared_ranges)
    quotient, remainder = divmod(count, ranges)
    result = []
    start = 0
    for index in range(ranges):
        size = quotient + int(index < remainder)
        result.append((start, start + size))
        start += size
    if start != count or any(stop <= start for start, stop in result):
        raise AssertionError("Task8 ordinal partition did not cover the exact selection")
    return tuple(result)


def _initialize_task8_worker() -> None:
    for name in (
        "ARROW_NUM_THREADS",
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        os.environ[name] = "1"


def _ptv2_token_payload_is_exact(
    input_ids: object, loss_mask: object, assistant_tokens: object | None = None
) -> bool:
    if (
        not isinstance(input_ids, list)
        or not input_ids
        or not isinstance(loss_mask, list)
        or len(input_ids) != len(loss_mask)
        or any(type(value) is not int or value < 0 for value in input_ids)
        or any(type(value) is not int or value not in (0, 1) for value in loss_mask)
    ):
        return False
    return assistant_tokens is None or (
        type(assistant_tokens) is int
        and assistant_tokens > 0
        and sum(loss_mask) == assistant_tokens
    )


def _stage_task8_selection_index(source: Path, destination: Path) -> tuple[int, str]:
    """Authenticate and copy the shared Task9 SQLite index exactly once node-locally."""
    source_flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    source_descriptor = os.open(source, source_flags)
    initial = os.fstat(source_descriptor)
    if not stat.S_ISREG(initial.st_mode):
        os.close(source_descriptor)
        raise ExposureViewError("Task8 selection index is not a no-follow regular file")
    destination_flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
    destination_descriptor = os.open(destination, destination_flags, 0o600)
    digest = sha256()
    try:
        with (
            os.fdopen(os.dup(source_descriptor), "rb") as input_stream,
            os.fdopen(destination_descriptor, "wb") as output_stream,
        ):
            for chunk in iter(lambda: input_stream.read(8 * 1024 * 1024), b""):
                digest.update(chunk)
                output_stream.write(chunk)
            output_stream.flush()
            os.fsync(output_stream.fileno())
        final = os.fstat(source_descriptor)
        pathname = os.lstat(source)
        if (
            stat.S_ISLNK(pathname.st_mode)
            or (pathname.st_dev, pathname.st_ino) != (final.st_dev, final.st_ino)
            or (initial.st_dev, initial.st_ino, initial.st_size, initial.st_mtime_ns)
            != (final.st_dev, final.st_ino, final.st_size, final.st_mtime_ns)
        ):
            raise ExposureViewError("Task8 selection index changed during node-local staging")
    except BaseException:
        os.close(source_descriptor)
        destination.unlink(missing_ok=True)
        raise
    os.close(source_descriptor)
    return initial.st_size, digest.hexdigest()


def _tokenize_ptv2_conversation(
    conversation: str, response: str, sequence_length: int
) -> tuple[list[int], list[int]]:
    assert _TASK8_TOKENIZER is not None
    try:
        canonical = json.loads(conversation)
    except json.JSONDecodeError as error:
        raise ExposureViewError("PTV2 selected conversation is invalid JSON") from error
    if not isinstance(canonical, dict) or not isinstance(canonical.get("messages"), list):
        raise ExposureViewError("PTV2 selected conversation has no messages")
    assistants = [
        message
        for message in canonical["messages"]
        if isinstance(message, dict) and message.get("role") == "assistant"
    ]
    if not assistants or canonical_json(assistants[-1]).decode("utf-8") != response:
        raise ExposureViewError(
            "PTV2 selected response is not the final assistant in its conversation"
        )
    encoded = _TASK8_TOKENIZER.apply_chat_template(
        canonical["messages"],
        tools=canonical.get("tools") or None,
        tokenize=True,
        add_generation_prompt=False,
        return_dict=True,
        return_assistant_tokens_mask=True,
    )
    input_ids = encoded.get("input_ids") if isinstance(encoded, Mapping) else None
    loss_mask = encoded.get("assistant_masks") if isinstance(encoded, Mapping) else None
    if loss_mask is None and isinstance(encoded, Mapping):
        loss_mask = encoded.get("assistant_tokens_mask")
    if not _ptv2_token_payload_is_exact(input_ids, loss_mask):
        raise ExposureViewError("PTV2 tokenizer did not return aligned IDs and assistant mask")
    input_ids = cast("list[int]", input_ids)
    loss_mask = cast("list[int]", loss_mask)
    input_ids = input_ids[:sequence_length]
    loss_mask = loss_mask[:sequence_length]
    if sum(loss_mask) < 1:
        raise ExposureViewError("PTV2 final training boundary has no assistant tokens")
    return input_ids, loss_mask


def _process_task8_range(task: _Task8TokenRange) -> _Task8TokenResult:
    started = time.monotonic_ns()
    observed = os.lstat(task.selection_index)
    if stat.S_ISLNK(observed.st_mode) or not stat.S_ISREG(observed.st_mode):
        raise ExposureViewError("Task8 selection index is not a no-follow regular file")
    source = sqlite3.connect(f"file:{task.selection_index}?mode=ro", uri=True)
    output = sqlite3.connect(task.spool_path)
    output.execute("PRAGMA synchronous=NORMAL")
    output.execute(
        "CREATE TABLE records(ordinal INTEGER PRIMARY KEY,prompt_uuid TEXT NOT NULL,"
        "source_identity_sha256 TEXT NOT NULL,source_row INTEGER NOT NULL,cell TEXT NOT NULL,"
        "reuse_index INTEGER NOT NULL,conversation_sha256 TEXT NOT NULL,"
        "assistant_response_sha256 TEXT NOT NULL,input_ids_json TEXT NOT NULL,"
        "loss_mask_json TEXT NOT NULL,assistant_tokens INTEGER NOT NULL,serialized_tokens INTEGER NOT NULL)"
    )
    count = 0
    try:
        cursor = source.execute(
            "SELECT occurrences.ordinal,occurrences.prompt_uuid,occurrences.source_identity_sha256,"
            "occurrences.source_row,occurrences.cell,occurrences.reuse_index,"
            "occurrences.conversation_sha256,occurrences.assistant_response_sha256,"
            "source_rows.canonical_conversation,source_rows.assistant_response "
            "FROM occurrences JOIN source_rows "
            "ON occurrences.source_identity_sha256=source_rows.source_identity_sha256 "
            "AND occurrences.source_row=source_rows.source_row "
            "WHERE occurrences.strategy=? AND occurrences.ordinal>=? AND occurrences.ordinal<? "
            "ORDER BY occurrences.ordinal",
            (task.strategy, task.start, task.stop),
        )
        for row in cursor:
            occurrence, conversation, response = row[:8], row[8], row[9]
            if sha256(conversation.encode("utf-8")).hexdigest() != occurrence[6]:
                raise ExposureViewError("PTV2 selected conversation hash mismatch")
            if sha256(response.encode("utf-8")).hexdigest() != occurrence[7]:
                raise ExposureViewError("PTV2 selected assistant response hash mismatch")
            input_ids, loss_mask = _tokenize_ptv2_conversation(
                conversation, response, task.sequence_length
            )
            output.execute(
                "INSERT INTO records VALUES(?,?,?,?,?,?,?,?,?,?,?,?)",
                (
                    *occurrence,
                    canonical_json(input_ids).decode("utf-8"),
                    canonical_json(loss_mask).decode("utf-8"),
                    sum(loss_mask),
                    len(input_ids),
                ),
            )
            count += 1
            if count % 10_000 == 0:
                output.commit()
        output.commit()
    except BaseException:
        output.close()
        source.close()
        task.spool_path.unlink(missing_ok=True)
        raise
    output.close()
    source.close()
    after = os.lstat(task.selection_index)
    if (
        stat.S_ISLNK(after.st_mode)
        or not stat.S_ISREG(after.st_mode)
        or (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
        != (observed.st_dev, observed.st_ino, observed.st_size, observed.st_mtime_ns)
    ):
        task.spool_path.unlink(missing_ok=True)
        raise ExposureViewError("Task8 selection index changed during tokenization")
    if count != task.stop - task.start:
        task.spool_path.unlink(missing_ok=True)
        raise ExposureViewError("Task8 ordinal range is incomplete")
    return _Task8TokenResult(
        task.index,
        task.start,
        task.stop,
        count,
        task.spool_path,
        task.spool_path.stat().st_size,
        _sha256_file(task.spool_path),
        round((time.monotonic_ns() - started) / 1_000_000_000, 6),
        os.getpid(),
    )


class _Task8LookupTokenizer:
    def __init__(self, tokenizer: Any, database: Path) -> None:
        self.tokenizer_sha256 = tokenizer.tokenizer_sha256
        self.chat_template = tokenizer.chat_template
        self.assistant_loss_target_sha256 = tokenizer.assistant_loss_target_sha256
        self._database = database
        self._connection = sqlite3.connect(f"file:{database}?mode=ro", uri=True)

    def apply_chat_template(self, messages: Any, **kwargs: Any) -> dict[str, list[int]]:
        conversation = canonical_json(
            {"messages": messages, "tools": kwargs.get("tools") or []}
        ).decode("utf-8")
        digest = sha256(conversation.encode("utf-8")).hexdigest()
        row = self._connection.execute(
            "SELECT input_ids_json,loss_mask_json FROM tokenized WHERE conversation_sha256=?",
            (digest,),
        ).fetchone()
        if row is None:
            raise ExposureViewError("Task8 parallel token lookup is incomplete")
        return {"input_ids": json.loads(row[0]), "assistant_masks": json.loads(row[1])}

    def close(self) -> None:
        self._connection.close()


def _pretokenize_task8(
    *,
    index_path: Path,
    strategy: str,
    occurrence_count: int,
    sequence_length: int,
    tokenizer: Any,
    work_root: Path,
    workers: int,
    source_commit: str,
) -> tuple[
    tempfile.TemporaryDirectory[str],
    Path,
    _Task8LookupTokenizer,
    Mapping[str, Any],
    int,
]:
    effective, allocated = resolve_task8_worker_count(workers, occurrence_count=occurrence_count)
    work_root.mkdir(parents=True, exist_ok=True)
    resolved_work_root = work_root.resolve(strict=True)
    if (
        work_root.is_symlink()
        or not work_root.is_dir()
        or resolved_work_root.parts[:2] in {("/", "lustre"), ("/", "home")}
    ):
        raise ExposureViewError("Task8 work root must be a safe node-local directory")
    started_wall_ns = time.time_ns()
    started = time.monotonic_ns()
    temporary = tempfile.TemporaryDirectory(prefix="task8-token-ranges-", dir=resolved_work_root)
    root = Path(temporary.name)
    staged_index = root / "selection-index.sqlite3"
    stage_started = time.monotonic_ns()
    try:
        source_index_bytes, source_index_sha256 = _stage_task8_selection_index(
            index_path, staged_index
        )
    except BaseException:
        temporary.cleanup()
        raise
    source_stage_elapsed_seconds = round((time.monotonic_ns() - stage_started) / 1_000_000_000, 6)
    ranges = _task8_ranges(occurrence_count, 201)
    tasks = tuple(
        _Task8TokenRange(
            index,
            start,
            stop,
            staged_index,
            strategy,
            sequence_length,
            root / f"range-{index:03d}.sqlite3",
        )
        for index, (start, stop) in enumerate(ranges)
    )
    global _TASK8_TOKENIZER
    _TASK8_TOKENIZER = tokenizer
    _initialize_task8_worker()
    thread_environment = {
        name: os.environ.get(name)
        for name in (
            "ARROW_NUM_THREADS",
            "OMP_NUM_THREADS",
            "MKL_NUM_THREADS",
            "OPENBLAS_NUM_THREADS",
            "NUMEXPR_NUM_THREADS",
        )
    }
    if set(thread_environment.values()) != {"1"}:
        raise ExposureViewError("Task8 thread caps did not reconcile")
    try:
        results: dict[int, _Task8TokenResult] = {}
        with ProcessPoolExecutor(
            max_workers=effective,
            mp_context=get_context("fork"),
            initializer=_initialize_task8_worker,
        ) as executor:
            futures = {executor.submit(_process_task8_range, task): task.index for task in tasks}
            try:
                for future in as_completed(futures):
                    result = future.result()
                    results[result.index] = result
            except BaseException:
                for future in futures:
                    future.cancel()
                raise
        ordered = tuple(results[index] for index in range(len(tasks)))
        database = root / "lookup.sqlite3"
        connection = sqlite3.connect(database)
        connection.execute(
            "CREATE TABLE tokenized(conversation_sha256 TEXT PRIMARY KEY,input_ids_json TEXT NOT NULL,"
            "loss_mask_json TEXT NOT NULL) WITHOUT ROWID"
        )
        for result in ordered:
            observed = os.lstat(result.spool_path)
            if (
                not stat.S_ISREG(observed.st_mode)
                or observed.st_size != result.spool_bytes
                or _sha256_file(result.spool_path) != result.spool_sha256
            ):
                raise ExposureViewError("Task8 token spool changed before deterministic merge")
            with sqlite3.connect(result.spool_path) as shard:
                for conversation_sha, input_ids, loss_mask in shard.execute(
                    "SELECT conversation_sha256,input_ids_json,loss_mask_json "
                    "FROM records ORDER BY ordinal"
                ):
                    previous = connection.execute(
                        "SELECT input_ids_json,loss_mask_json FROM tokenized "
                        "WHERE conversation_sha256=?",
                        (conversation_sha,),
                    ).fetchone()
                    if previous is not None and previous != (input_ids, loss_mask):
                        raise ExposureViewError(
                            "Task8 duplicate conversation tokenization diverged"
                        )
                    connection.execute(
                        "INSERT OR IGNORE INTO tokenized VALUES(?,?,?)",
                        (conversation_sha, input_ids, loss_mask),
                    )
            connection.commit()
        connection.close()
        parallel_finished_wall_ns = time.time_ns()
        parallel_elapsed_seconds = round((time.monotonic_ns() - started) / 1_000_000_000, 6)
        execution: dict[str, Any] = {
            "schema_version": 1,
            "source_commit": source_commit,
            "declared_range_count": len(ranges),
            "occurrence_count": occurrence_count,
            "allocated_cpus": allocated,
            "requested_workers": workers,
            "effective_workers": effective,
            "threads_per_worker": 1,
            "thread_environment": thread_environment,
            "source_index_bytes": source_index_bytes,
            "source_index_sha256": source_index_sha256,
            "source_stage_elapsed_seconds": source_stage_elapsed_seconds,
            "started_at_ns": started_wall_ns,
            "parallel_phase_finished_at_ns": parallel_finished_wall_ns,
            "parallel_phase_elapsed_seconds": parallel_elapsed_seconds,
            "ranges": [
                {**asdict(result), "spool_path": result.spool_path.name} for result in ordered
            ],
        }
        return (
            temporary,
            staged_index,
            _Task8LookupTokenizer(tokenizer, database),
            MappingProxyType(execution),
            started,
        )
    except BaseException:
        temporary.cleanup()
        raise
    finally:
        _TASK8_TOKENIZER = None


def derive_ptv2_one_pass_corpus(
    view: Any,
    *,
    tokenizer_sha256: str,
    chat_template_sha256: str,
    assistant_loss_target_sha256: str,
    training_config_sha256: str,
    tokenizer: Any,
    output_root: Path,
    work_root: Path | None = None,
    sequence_length: int = 4_096,
    workers: int = 1,
    source_commit: str | None = None,
    _parallel_execution: Mapping[str, Any] | None = None,
    _parallel_started_monotonic_ns: int | None = None,
    milestone_occurrences: tuple[int, ...] = (500_224, 1_000_448, 1_300_000, 2_000_000),
    milestone_steps: tuple[int, ...] = (977, 1_954, 2_540, 3_908),
) -> PTV2OnePassCorpus:
    """Materialize authenticated assistant masks from the selected SQLite responses."""
    for digest_name, digest in (
        ("tokenizer", tokenizer_sha256),
        ("chat template", chat_template_sha256),
        ("assistant loss target", assistant_loss_target_sha256),
        ("training config", training_config_sha256),
    ):
        _require_digest(digest_name, digest)
    actual_tokenizer_sha256 = getattr(tokenizer, "tokenizer_sha256", None)
    actual_template = getattr(tokenizer, "chat_template", None)
    actual_mask_sha256 = getattr(tokenizer, "assistant_loss_target_sha256", None)
    if (
        actual_tokenizer_sha256 != tokenizer_sha256
        or not isinstance(actual_template, str)
        or sha256(actual_template.encode("utf-8")).hexdigest() != chat_template_sha256
        or actual_mask_sha256 != assistant_loss_target_sha256
    ):
        raise ExposureViewError("PTV2 tokenizer, template, or assistant-mask identity mismatch")
    strategy = getattr(view, "strategy", None)
    if strategy not in {"A-repair", "B-balanced"}:
        raise ExposureViewError("PTV2 materialized view strategy is invalid")
    index_path = Path(getattr(view, "index_path", ""))
    raw_selection_sha256 = getattr(view, "selection_sha256", None)
    if not isinstance(raw_selection_sha256, str):
        raise ExposureViewError("PTV2 selection digest is missing")
    selection_sha256 = raw_selection_sha256
    _require_digest("PTV2 selection", selection_sha256)
    if not index_path.is_file() or index_path.is_symlink():
        raise ExposureViewError("PTV2 selection SQLite index is missing or unsafe")
    if (
        not isinstance(sequence_length, int)
        or isinstance(sequence_length, bool)
        or sequence_length < 1
        or len(milestone_occurrences) != len(milestone_steps)
        or any(
            not isinstance(value, int) or isinstance(value, bool) or value < 1
            for value in (*milestone_occurrences, *milestone_steps)
        )
    ):
        raise ExposureViewError("PTV2 sequence and milestone policy is invalid")
    root = Path(output_root)
    root.mkdir(parents=True, exist_ok=True)
    if root.is_symlink() or not root.is_dir():
        raise ExposureViewError("PTV2 tokenized output root is unsafe")
    if workers > 1:
        resolved_commit = source_commit or os.environ.get("SOURCE_COMMIT", "")
        if re.fullmatch(r"[0-9a-f]{40}", resolved_commit) is None:
            raise ExposureViewError("parallel Task8 processing requires an exact source commit")
        if work_root is None:
            raise ExposureViewError("parallel Task8 processing requires a node-local work root")
        temporary, staged_index, lookup, execution, started_monotonic_ns = _pretokenize_task8(
            index_path=index_path,
            strategy=strategy,
            occurrence_count=int(getattr(view, "occurrence_count", 0)),
            sequence_length=sequence_length,
            tokenizer=tokenizer,
            work_root=Path(work_root),
            workers=workers,
            source_commit=resolved_commit,
        )
        try:
            corpus = derive_ptv2_one_pass_corpus(
                _Task8StagedView(view, staged_index),
                tokenizer_sha256=tokenizer_sha256,
                chat_template_sha256=chat_template_sha256,
                assistant_loss_target_sha256=assistant_loss_target_sha256,
                training_config_sha256=training_config_sha256,
                tokenizer=lookup,
                output_root=output_root,
                sequence_length=sequence_length,
                workers=1,
                _parallel_execution=execution,
                _parallel_started_monotonic_ns=started_monotonic_ns,
                milestone_occurrences=milestone_occurrences,
                milestone_steps=milestone_steps,
            )
            return corpus
        finally:
            lookup.close()
            temporary.cleanup()
    bundle_path = root / f"{strategy.lower()}-tokenized"
    temporary_bundle = _prepare_ptv2_tokenized_bundle(root, bundle_path)
    temporary_database = temporary_bundle / "records.sqlite3"
    temporary_receipt = temporary_bundle / "TOKENIZED.json"
    database_path = bundle_path / "records.sqlite3"
    receipt_path = bundle_path / "TOKENIZED.json"
    occurrence_digest = sha256()
    response_digest = sha256()
    multiplicity_digest = sha256()
    bucket_assistant_tokens: dict[str, Counter[str]] = {}
    count = assistant_tokens = serialized_tokens = 0
    connection_out = sqlite3.connect(temporary_database)
    connection_out.execute("PRAGMA synchronous=FULL")
    connection_out.execute(
        "CREATE TABLE records(ordinal INTEGER PRIMARY KEY,prompt_uuid TEXT NOT NULL,"
        "source_identity_sha256 TEXT NOT NULL,source_row INTEGER NOT NULL,"
        "cell TEXT NOT NULL,language TEXT NOT NULL,reuse_index INTEGER NOT NULL,"
        "input_ids_json TEXT NOT NULL,loss_mask_json TEXT NOT NULL,assistant_tokens INTEGER NOT NULL)"
    )
    connection = sqlite3.connect(f"file:{index_path}?mode=ro", uri=True)
    try:
        cursor = connection.execute(
            "SELECT occurrences.ordinal,occurrences.prompt_uuid,occurrences.source_identity_sha256,"
            "occurrences.source_row,occurrences.cell,source_rows.language,occurrences.reuse_index,"
            "occurrences.conversation_sha256,occurrences.assistant_response_sha256,"
            "source_rows.canonical_conversation,source_rows.assistant_response "
            "FROM occurrences JOIN source_rows "
            "ON occurrences.source_identity_sha256=source_rows.source_identity_sha256 "
            "AND occurrences.source_row=source_rows.source_row "
            "WHERE occurrences.strategy=? ORDER BY occurrences.ordinal",
            (strategy,),
        )
        for row in cursor:
            occurrence = (row[0], row[1], row[2], row[3], row[4], row[6], row[7], row[8])
            language = row[5]
            conversation = row[9]
            response = row[10]
            if sha256(conversation.encode("utf-8")).hexdigest() != occurrence[6]:
                raise ExposureViewError("PTV2 selected conversation hash mismatch")
            if sha256(response.encode("utf-8")).hexdigest() != occurrence[7]:
                raise ExposureViewError("PTV2 selected assistant response hash mismatch")
            try:
                canonical = json.loads(conversation)
            except json.JSONDecodeError as error:
                raise ExposureViewError("PTV2 selected conversation is invalid JSON") from error
            if not isinstance(canonical, dict) or not isinstance(canonical.get("messages"), list):
                raise ExposureViewError("PTV2 selected conversation has no messages")
            assistants = [
                message
                for message in canonical["messages"]
                if isinstance(message, dict) and message.get("role") == "assistant"
            ]
            if not assistants or canonical_json(assistants[-1]).decode("utf-8") != response:
                raise ExposureViewError(
                    "PTV2 selected response is not the final assistant in its conversation"
                )
            encoded = tokenizer.apply_chat_template(
                canonical["messages"],
                tools=canonical.get("tools") or None,
                tokenize=True,
                add_generation_prompt=False,
                return_dict=True,
                return_assistant_tokens_mask=True,
            )
            input_ids = encoded.get("input_ids") if isinstance(encoded, Mapping) else None
            loss_mask = encoded.get("assistant_masks") if isinstance(encoded, Mapping) else None
            if loss_mask is None and isinstance(encoded, Mapping):
                loss_mask = encoded.get("assistant_tokens_mask")
            if not _ptv2_token_payload_is_exact(input_ids, loss_mask):
                raise ExposureViewError(
                    "PTV2 tokenizer did not return aligned IDs and assistant mask"
                )
            input_ids = cast("list[int]", input_ids)
            loss_mask = cast("list[int]", loss_mask)
            input_ids = input_ids[:sequence_length]
            loss_mask = loss_mask[:sequence_length]
            token_count = sum(loss_mask)
            if token_count < 1:
                raise ExposureViewError("PTV2 final training boundary has no assistant tokens")
            occurrence_digest.update(canonical_json(list(occurrence)))
            occurrence_digest.update(b"\n")
            multiplicity_digest.update(
                canonical_json(
                    [
                        occurrence[1],
                        occurrence[2],
                        occurrence[3],
                        occurrence[4],
                        language,
                        occurrence[5],
                    ]
                )
            )
            multiplicity_digest.update(b"\n")
            response_digest.update(
                canonical_json([occurrence[2], occurrence[3], occurrence[6], occurrence[7]])
            )
            response_digest.update(b"\n")
            count += 1
            assistant_tokens += token_count
            serialized_tokens += len(input_ids)
            bucket_assistant_tokens.setdefault(occurrence[4], Counter())[str(language)] += (
                token_count
            )
            connection_out.execute(
                "INSERT INTO records VALUES(?,?,?,?,?,?,?,?,?,?)",
                (
                    occurrence[0],
                    occurrence[1],
                    occurrence[2],
                    occurrence[3],
                    occurrence[4],
                    language,
                    occurrence[5],
                    canonical_json(input_ids).decode("utf-8"),
                    canonical_json(loss_mask).decode("utf-8"),
                    token_count,
                ),
            )
        connection_out.commit()
    except Exception:
        connection_out.rollback()
        raise
    finally:
        connection.close()
        connection_out.close()
    if count != getattr(view, "occurrence_count", None):
        raise ExposureViewError(
            "PTV2 materialized occurrence count does not match selection receipt"
        )
    if occurrence_digest.hexdigest() != getattr(view, "ordered_occurrences_sha256", None):
        raise ExposureViewError(
            "PTV2 materialized occurrence order does not match selection receipt"
        )
    if response_digest.hexdigest() != getattr(view, "source_response_root_sha256", None):
        raise ExposureViewError("PTV2 materialized response root does not match selection receipt")
    unique = int(getattr(view, "unique_prompt_count", 0))
    natural = int(getattr(view, "natural_duplicate_count", 0))
    constructed = int(getattr(view, "constructed_repeat_count", 0))
    if unique < 1 or natural < 0 or constructed < 0 or unique + natural + constructed != count:
        raise ExposureViewError("PTV2 selection multiplicity summary does not reconcile")
    _fsync_file(temporary_database)
    database_sha256 = _sha256_file(temporary_database)
    execution_descriptor: dict[str, Any] | None = None
    if _parallel_execution is not None:
        execution_payload = dict(_parallel_execution)
        if _parallel_started_monotonic_ns is None:
            raise ExposureViewError("Task8 total execution start is missing")
        execution_payload.update(
            {
                "finished_at_ns": time.time_ns(),
                "elapsed_seconds": round(
                    (time.monotonic_ns() - _parallel_started_monotonic_ns) / 1_000_000_000,
                    6,
                ),
                "selection_sha256": selection_sha256,
                "ordered_occurrences_sha256": occurrence_digest.hexdigest(),
                "source_response_root_sha256": response_digest.hexdigest(),
                "tokenizer_sha256": tokenizer_sha256,
                "chat_template_sha256": chat_template_sha256,
                "assistant_loss_target_sha256": assistant_loss_target_sha256,
                "tokenized_sha256": database_sha256,
            }
        )
        execution_payload["receipt_sha256"] = sha256(canonical_json(execution_payload)).hexdigest()
        execution_path = temporary_bundle / "EXECUTION_RECEIPT.json"
        execution_bytes = canonical_json(execution_payload) + b"\n"
        _write_exclusive(execution_path, execution_bytes)
        _fsync_file(execution_path)
        execution_descriptor = {
            "path": execution_path.name,
            "bytes": len(execution_bytes),
            "sha256": sha256(execution_bytes).hexdigest(),
        }
    receipt = {
        "schema_version": 2,
        "strategy": strategy,
        "occurrence_count": count,
        "trainer_epochs": getattr(view, "trainer_epochs", 0),
        "assistant_tokens": assistant_tokens,
        "serialized_tokens": serialized_tokens,
        "packed_sequence_lower_bound": (serialized_tokens + sequence_length - 1) // sequence_length,
        "unique_prompt_count": unique,
        "natural_duplicate_count": natural,
        "constructed_repeat_count": constructed,
        "milestone_occurrences": list(milestone_occurrences),
        "milestone_steps": list(milestone_steps),
        "segment_occurrences": [1_300_000, 700_000],
        "segment_steps": [2_540, 1_368],
        "cumulative_segment_steps": [2_540, 3_908],
        "segment_final_valid_occurrences": [32, 96],
        "tokenizer_sha256": tokenizer_sha256,
        "chat_template_sha256": chat_template_sha256,
        "assistant_loss_target_sha256": assistant_loss_target_sha256,
        "training_config_sha256": training_config_sha256,
        "source_response_root_sha256": response_digest.hexdigest(),
        "ordered_occurrences_sha256": occurrence_digest.hexdigest(),
        "selection_sha256": selection_sha256,
        "base_occurrence_multiplicity_sha256": multiplicity_digest.hexdigest(),
        "bucket_assistant_token_histogram": {
            cell: dict(sorted(languages.items()))
            for cell, languages in sorted(bucket_assistant_tokens.items())
        },
        "database_path": "records.sqlite3",
        "database_sha256": database_sha256,
        "database_bytes": temporary_database.stat().st_size,
    }
    if execution_descriptor is not None:
        receipt["execution_receipt"] = execution_descriptor
        assert _parallel_execution is not None
        receipt["source_index_bytes"] = _parallel_execution["source_index_bytes"]
        receipt["source_index_sha256"] = _parallel_execution["source_index_sha256"]
    receipt_sha256 = sha256(canonical_json(receipt)).hexdigest()
    _write_exclusive(
        temporary_receipt, canonical_json(receipt | {"receipt_sha256": receipt_sha256}) + b"\n"
    )
    _fsync_file(temporary_receipt)
    _fsync_directory(temporary_bundle)
    _publish_ptv2_tokenized_bundle(temporary_bundle, bundle_path)
    return PTV2OnePassCorpus(
        strategy=strategy,
        occurrence_count=count,
        trainer_epochs=getattr(view, "trainer_epochs", 0),
        assistant_tokens=assistant_tokens,
        tokenizer_sha256=tokenizer_sha256,
        chat_template_sha256=chat_template_sha256,
        assistant_loss_target_sha256=assistant_loss_target_sha256,
        training_config_sha256=training_config_sha256,
        source_response_root_sha256=response_digest.hexdigest(),
        ordered_occurrences_sha256=occurrence_digest.hexdigest(),
        selection_sha256=selection_sha256,
        unique_prompt_count=unique,
        natural_duplicate_count=natural,
        constructed_repeat_count=constructed,
        serialized_tokens=serialized_tokens,
        packed_sequence_lower_bound=receipt["packed_sequence_lower_bound"],
        milestone_occurrences=milestone_occurrences,
        milestone_steps=milestone_steps,
        tokenized_path=str(database_path),
        tokenized_sha256=database_sha256,
        receipt_path=str(receipt_path),
        receipt_sha256=receipt_sha256,
    )


def derive_ptv2_historical_corpus(
    prefix: PTV2OnePassCorpus, output_root: os.PathLike[str] | str
) -> PTV2OnePassCorpus:
    """Derive the immutable historical arm from the authenticated first A segment."""
    if not isinstance(prefix, PTV2OnePassCorpus) or prefix.strategy != "A-repair":
        raise ExposureViewError("PTV2 historical derivation requires an A-repair corpus")
    source = _load_ptv2_exposure_source(prefix, 1)
    root = Path(output_root)
    root.mkdir(parents=True, exist_ok=True)
    if root.is_symlink() or not root.is_dir():
        source.close()
        raise ExposureViewError("PTV2 historical output root is unsafe")
    destination = root / "h-historical-cyclic-tokenized"
    temporary = _prepare_ptv2_tokenized_bundle(root, destination)
    database = temporary / "records.sqlite3"
    receipt_path = temporary / "TOKENIZED.json"
    try:
        prefix_count = int(source.receipt["segment_occurrences"][0])
        if prefix_count < 1 or prefix_count > len(source.rows):
            raise ExposureViewError("PTV2 historical prefix boundary is invalid")
        output = sqlite3.connect(database)
        output.execute("PRAGMA synchronous=FULL")
        output.execute(
            "CREATE TABLE records(ordinal INTEGER PRIMARY KEY,prompt_uuid TEXT NOT NULL,"
            "source_identity_sha256 TEXT NOT NULL,source_row INTEGER NOT NULL,"
            "cell TEXT NOT NULL,language TEXT NOT NULL,reuse_index INTEGER NOT NULL,"
            "input_ids_json TEXT NOT NULL,loss_mask_json TEXT NOT NULL,"
            "assistant_tokens INTEGER NOT NULL)"
        )
        parent = sqlite3.connect(f"file:{source.database_path}?mode=ro", uri=True)
        record_stream = sha256()
        multiplicity = sha256()
        histogram: Counter[tuple[str, str]] = Counter()
        assistant_tokens = serialized_tokens = count = 0
        try:
            query = (
                "SELECT ordinal,prompt_uuid,source_identity_sha256,source_row,cell,language,"
                "reuse_index,input_ids_json,loss_mask_json,assistant_tokens FROM records "
                "WHERE ordinal<? ORDER BY ordinal"
            )
            for row in parent.execute(query, (prefix_count,)):
                if row[0] != count:
                    raise ExposureViewError("PTV2 historical prefix order is not contiguous")
                ids = json.loads(row[7])
                mask = json.loads(row[8])
                if not _ptv2_token_payload_is_exact(ids, mask, row[9]):
                    raise ExposureViewError("PTV2 historical prefix payload is invalid")
                canonical_row = canonical_json(list(row))
                record_stream.update(canonical_row + b"\n")
                multiplicity.update(canonical_json(list(row[1:7])) + b"\n")
                histogram[(row[4], row[5])] += row[9]
                assistant_tokens += row[9]
                serialized_tokens += len(ids)
                count += 1
                output.execute("INSERT INTO records VALUES(?,?,?,?,?,?,?,?,?,?)", row)
            output.commit()
        except BaseException:
            output.rollback()
            raise
        finally:
            parent.close()
            output.close()
        if count != prefix_count:
            raise ExposureViewError("PTV2 historical prefix is incomplete")
        with sqlite3.connect(f"file:{database}?mode=ro", uri=True) as connection:
            unique = int(
                connection.execute("SELECT COUNT(DISTINCT prompt_uuid) FROM records").fetchone()[0]
            )
        _fsync_file(database)
        database_sha256 = _sha256_file(database)
        record_stream_sha256 = record_stream.hexdigest()
        ordered_sha256 = sha256(
            canonical_json(
                [
                    "ptv2-historical-prefix-order-v1",
                    source.receipt["ordered_occurrences_sha256"],
                    prefix_count,
                    record_stream_sha256,
                ]
            )
        ).hexdigest()
        response_sha256 = sha256(
            canonical_json(
                [
                    "ptv2-historical-prefix-response-v1",
                    source.receipt["source_response_root_sha256"],
                    prefix_count,
                    record_stream_sha256,
                ]
            )
        ).hexdigest()
        selection_sha256 = sha256(
            canonical_json(
                [
                    "ptv2-historical-prefix-selection-v1",
                    source.receipt["selection_sha256"],
                    prefix_count,
                    record_stream_sha256,
                ]
            )
        ).hexdigest()
        steps = (prefix_count + 511) // 512
        final_valid = prefix_count % 512
        receipt = {
            "schema_version": 2,
            "strategy": "H-historical-cyclic",
            "occurrence_count": prefix_count,
            "trainer_epochs": 1,
            "assistant_tokens": assistant_tokens,
            "serialized_tokens": serialized_tokens,
            "packed_sequence_lower_bound": (serialized_tokens + 4_095) // 4_096,
            "unique_prompt_count": unique,
            "natural_duplicate_count": prefix_count - unique,
            "constructed_repeat_count": 0,
            "milestone_occurrences": [prefix_count],
            "milestone_steps": [steps],
            "segment_occurrences": [prefix_count, 0],
            "segment_steps": [steps, 0],
            "cumulative_segment_steps": [steps, steps],
            "segment_final_valid_occurrences": [final_valid, 0],
            "tokenizer_sha256": source.receipt["tokenizer_sha256"],
            "chat_template_sha256": source.receipt["chat_template_sha256"],
            "assistant_loss_target_sha256": source.receipt["assistant_loss_target_sha256"],
            "training_config_sha256": source.receipt["training_config_sha256"],
            "source_response_root_sha256": response_sha256,
            "ordered_occurrences_sha256": ordered_sha256,
            "selection_sha256": selection_sha256,
            "base_occurrence_multiplicity_sha256": multiplicity.hexdigest(),
            "bucket_assistant_token_histogram": _nested_ptv2_histogram(histogram),
            "historical_parent_receipt_sha256": prefix.receipt_sha256,
            "historical_parent_tokenized_sha256": prefix.tokenized_sha256,
            "historical_prefix_occurrence_count": prefix_count,
            "historical_prefix_record_stream_sha256": record_stream_sha256,
            "database_path": "records.sqlite3",
            "database_sha256": database_sha256,
            "database_bytes": database.stat().st_size,
        }
        receipt_sha256 = sha256(canonical_json(receipt)).hexdigest()
        _write_exclusive(
            receipt_path, canonical_json(receipt | {"receipt_sha256": receipt_sha256}) + b"\n"
        )
        _fsync_file(receipt_path)
        _fsync_directory(temporary)
        _publish_ptv2_tokenized_bundle(temporary, destination)
        return PTV2OnePassCorpus(
            strategy="H-historical-cyclic",
            occurrence_count=prefix_count,
            trainer_epochs=1,
            assistant_tokens=assistant_tokens,
            tokenizer_sha256=str(receipt["tokenizer_sha256"]),
            chat_template_sha256=str(receipt["chat_template_sha256"]),
            assistant_loss_target_sha256=str(receipt["assistant_loss_target_sha256"]),
            training_config_sha256=str(receipt["training_config_sha256"]),
            source_response_root_sha256=response_sha256,
            ordered_occurrences_sha256=ordered_sha256,
            selection_sha256=selection_sha256,
            unique_prompt_count=unique,
            natural_duplicate_count=prefix_count - unique,
            constructed_repeat_count=0,
            serialized_tokens=serialized_tokens,
            packed_sequence_lower_bound=int(receipt["packed_sequence_lower_bound"]),
            milestone_occurrences=(prefix_count,),
            milestone_steps=(steps,),
            segment_occurrences=(prefix_count, 0),
            segment_steps=(steps, 0),
            cumulative_segment_steps=(steps, steps),
            segment_final_valid_occurrences=(final_valid, 0),
            tokenized_path=str(destination / "records.sqlite3"),
            tokenized_sha256=database_sha256,
            receipt_path=str(destination / "TOKENIZED.json"),
            receipt_sha256=receipt_sha256,
            historical_parent_receipt_sha256=prefix.receipt_sha256,
            historical_parent_tokenized_sha256=prefix.tokenized_sha256,
            historical_prefix_record_stream_sha256=record_stream_sha256,
        )
    finally:
        source.close()
        if temporary.exists():
            shutil.rmtree(temporary)


def build_ptv2_study_exposures(
    prefix: PTV2OnePassCorpus,
    balanced: PTV2OnePassCorpus,
    historical: PTV2OnePassCorpus | None = None,
    *,
    scientific_tokens: int = 256_000_000,
    runtime_screen_tokens: int = 64_000_000,
    workers: int = 1,
    expected_completion_receipt_sha256: str | None = None,
) -> PTV2StudyExposureViews:
    """Authorize common-order A/B/H PTV2 exposures at exact assistant-token totals."""
    if scientific_tokens != 256_000_000:
        raise ExposureViewError("PTV2 scientific tokens must be exactly 256M")
    if runtime_screen_tokens != 64_000_000:
        raise ExposureViewError("PTV2 runtime screen must be exactly 64M")
    if any(corpus.assistant_tokens < scientific_tokens for corpus in (prefix, balanced)):
        raise ExposureViewError(
            "PTV2 A/B one-pass receipts must each reach the exact 256M scientific boundary"
        )
    _validate_ptv2_one_pass(prefix, "A-repair")
    _validate_ptv2_one_pass(balanced, "B-balanced")
    if historical is not None:
        _validate_ptv2_historical(historical)
        _require_ptv2_historical_parent(historical, prefix)
    identity_fields = (
        "tokenizer_sha256",
        "chat_template_sha256",
        "assistant_loss_target_sha256",
        "training_config_sha256",
    )
    corpora = (prefix, balanced) + ((historical,) if historical is not None else ())
    if any(
        getattr(prefix, field) != getattr(corpus, field)
        for corpus in corpora
        for field in identity_fields
    ):
        raise ExposureViewError(
            "A/B/H tokenizer, template, target mask, and training config must match"
        )
    runtime, scientific, completion_sha256 = _materialize_ptv2_exposure_set(
        corpora,
        runtime_screen_tokens=runtime_screen_tokens,
        scientific_tokens=scientific_tokens,
        workers=workers,
        expected_completion_receipt_sha256=expected_completion_receipt_sha256,
    )
    completion_path = (
        Path(prefix.tokenized_path).parent.parent / "ptv2-study-exposures" / "COMPLETE.json"
    )
    return PTV2StudyExposureViews(
        runtime_screen_tokens,
        scientific_tokens,
        prefix.assistant_tokens,
        balanced.assistant_tokens,
        True,
        MappingProxyType(runtime),
        MappingProxyType(scientific),
        historical.assistant_tokens if historical is not None else None,
        str(completion_path),
        completion_sha256,
    )


def _nested_ptv2_histogram(histogram: Mapping[tuple[str, str], int]) -> dict[str, dict[str, int]]:
    nested: dict[str, dict[str, int]] = {}
    for (cell, language), value in sorted(histogram.items()):
        nested.setdefault(cell, {})[language] = value
    return nested


def _flatten_ptv2_histogram(value: Any) -> dict[tuple[str, str], int]:
    if not isinstance(value, dict) or not value:
        raise ExposureViewError("PTV2 bucket assistant-token histogram is malformed")
    flattened: dict[tuple[str, str], int] = {}
    for cell, languages in value.items():
        if not isinstance(cell, str) or not cell or not isinstance(languages, dict):
            raise ExposureViewError("PTV2 bucket assistant-token histogram is malformed")
        for language, tokens in languages.items():
            if (
                not isinstance(language, str)
                or not language
                or isinstance(tokens, bool)
                or not isinstance(tokens, int)
                or tokens < 1
            ):
                raise ExposureViewError("PTV2 bucket assistant-token histogram is malformed")
            flattened[(cell, language)] = tokens
    return flattened


def _ptv2_target_histogram(
    weights: Mapping[tuple[str, str], int], target_tokens: int
) -> dict[tuple[str, str], int]:
    total = sum(weights.values())
    targets = {bucket: target_tokens * weight // total for bucket, weight in weights.items()}
    remaining = target_tokens - sum(targets.values())
    remainders = sorted(
        weights,
        key=lambda bucket: (-(target_tokens * weights[bucket] % total), bucket),
    )
    for bucket in remainders[:remaining]:
        targets[bucket] += 1
    return targets


def _ptv2_file_snapshot(metadata: os.stat_result) -> tuple[int, int, int, int, int]:
    return (
        metadata.st_dev,
        metadata.st_ino,
        metadata.st_size,
        metadata.st_mtime_ns,
        metadata.st_ctime_ns,
    )


def _open_ptv2_regular_componentwise(path: Path, label: str) -> tuple[int, int, str]:
    raw = os.fspath(path)
    if not raw.startswith("/") or raw == "/" or "//" in raw:
        raise ExposureViewError(f"{label} must be a canonical absolute path")
    components = raw.split("/")[1:]
    if not components or any(component in {"", ".", ".."} for component in components):
        raise ExposureViewError(f"{label} must be a canonical absolute path")
    directory_flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
    parent_descriptor = os.open("/", directory_flags)
    try:
        for component in components[:-1]:
            child = os.open(component, directory_flags, dir_fd=parent_descriptor)
            os.close(parent_descriptor)
            parent_descriptor = child
        descriptor = os.open(
            components[-1],
            os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0),
            dir_fd=parent_descriptor,
        )
    except OSError as error:
        os.close(parent_descriptor)
        raise ExposureViewError(f"{label} contains a symlink or unsafe path component") from error
    if not stat.S_ISREG(os.fstat(descriptor).st_mode):
        os.close(descriptor)
        os.close(parent_descriptor)
        raise ExposureViewError(f"{label} must be a no-follow regular file")
    return descriptor, parent_descriptor, components[-1]


def _copy_ptv2_source_file(source: Path, destination: Path, label: str) -> tuple[int, str]:
    source_descriptor, parent_descriptor, entry_name = _open_ptv2_regular_componentwise(
        source, label
    )
    destination_flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
    destination_descriptor = os.open(destination, destination_flags, 0o600)
    digest = sha256()
    before = os.fstat(source_descriptor)
    try:
        with (
            os.fdopen(os.dup(source_descriptor), "rb") as input_stream,
            os.fdopen(destination_descriptor, "wb") as output_stream,
        ):
            for chunk in iter(lambda: input_stream.read(8 * 1024 * 1024), b""):
                digest.update(chunk)
                output_stream.write(chunk)
            output_stream.flush()
            os.fsync(output_stream.fileno())
        after = os.fstat(source_descriptor)
        namespace = os.stat(entry_name, dir_fd=parent_descriptor, follow_symlinks=False)
        if _ptv2_file_snapshot(before) != _ptv2_file_snapshot(after) or (
            namespace.st_dev,
            namespace.st_ino,
        ) != (after.st_dev, after.st_ino):
            raise ExposureViewError(f"{label} changed during authenticated staging")
    except BaseException:
        destination.unlink(missing_ok=True)
        raise
    finally:
        os.close(source_descriptor)
        os.close(parent_descriptor)
    return before.st_size, digest.hexdigest()


def _read_ptv2_regular_stable(path: Path, label: str, *, max_bytes: int) -> bytes:
    try:
        descriptor, parent_descriptor, entry_name = _open_ptv2_regular_componentwise(path, label)
    except ExposureViewError as error:
        raise ExposureViewError(f"{label} is unsafe or missing") from error
    before = os.fstat(descriptor)
    try:
        if before.st_size > max_bytes:
            raise ExposureViewError(f"{label} exceeds its bounded size")
        with os.fdopen(os.dup(descriptor), "rb") as stream:
            raw = stream.read()
        after = os.fstat(descriptor)
        namespace = os.stat(entry_name, dir_fd=parent_descriptor, follow_symlinks=False)
        if _ptv2_file_snapshot(before) != _ptv2_file_snapshot(after) or (
            namespace.st_dev,
            namespace.st_ino,
        ) != (after.st_dev, after.st_ino):
            raise ExposureViewError(f"{label} changed while it was read")
        return raw
    except OSError as error:
        raise ExposureViewError(f"{label} is unsafe or unreadable") from error
    finally:
        os.close(descriptor)
        os.close(parent_descriptor)


def _stage_ptv2_exposure_source(
    corpus: PTV2OnePassCorpus,
) -> tuple[tempfile.TemporaryDirectory[str], Path, Path]:
    if not corpus.tokenized_path or not corpus.receipt_path:
        raise ExposureViewError("PTV2 v2 tokenized artifacts are missing or unauthenticated")
    temporary = tempfile.TemporaryDirectory(prefix="ptv2-exposure-source-")
    root = Path(temporary.name)
    database = root / "records.sqlite3"
    receipt = root / "TOKENIZED.json"
    try:
        _, database_sha256 = _copy_ptv2_source_file(
            Path(corpus.tokenized_path), database, "PTV2 tokenized database"
        )
        _copy_ptv2_source_file(Path(corpus.receipt_path), receipt, "PTV2 tokenized receipt")
        if database_sha256 != corpus.tokenized_sha256:
            raise ExposureViewError("PTV2 v2 tokenized database digest mismatch")
    except BaseException:
        temporary.cleanup()
        raise
    return temporary, database, receipt


def _read_ptv2_source_range(
    database_path: Path, start: int, stop: int
) -> list[_PTV2ExposureSourceRow]:
    connection = sqlite3.connect(f"file:{database_path}?mode=ro", uri=True)
    try:
        rows = []
        for value in connection.execute(
            "SELECT ordinal,prompt_uuid,source_identity_sha256,source_row,cell,language,"
            "reuse_index,assistant_tokens,input_ids_json,loss_mask_json FROM records "
            "WHERE ordinal>=? AND ordinal<? "
            "ORDER BY ordinal",
            (start, stop),
        ):
            row = _PTV2ExposureSourceRow(*value[:8])
            input_ids = json.loads(value[8])
            loss_mask = json.loads(value[9])
            if (
                row.ordinal < 0
                or not row.prompt_uuid
                or _SHA256.fullmatch(row.source_identity_sha256) is None
                or row.source_row < 0
                or not row.cell
                or not row.language
                or row.reuse_index < 0
                or row.assistant_tokens < 1
                or not _ptv2_token_payload_is_exact(input_ids, loss_mask, row.assistant_tokens)
            ):
                raise ExposureViewError("PTV2 v2 source-row identity is malformed")
            rows.append(row)
        return rows
    except (json.JSONDecodeError, sqlite3.Error) as error:
        raise ExposureViewError("PTV2 v2 tokenized database schema is invalid") from error
    finally:
        connection.close()


def _load_ptv2_exposure_source(
    corpus: PTV2OnePassCorpus, workers: int
) -> _VerifiedPTV2ExposureSource:
    if isinstance(workers, bool) or not isinstance(workers, int) or workers not in {1, 96}:
        raise ExposureViewError("PTV2 exposure workers must be exactly serial-1 or parallel-96")
    temporary, database_path, receipt_path = _stage_ptv2_exposure_source(corpus)
    try:
        raw_receipt = receipt_path.read_bytes()
        receipt = json.loads(raw_receipt)
        if not isinstance(receipt, dict):
            raise ExposureViewError("PTV2 v2 tokenized receipt is malformed")
        claimed = receipt.pop("receipt_sha256", None)
        if (
            receipt.get("schema_version") != 2
            or claimed != corpus.receipt_sha256
            or claimed != sha256(canonical_json(receipt)).hexdigest()
            or raw_receipt != canonical_json(receipt | {"receipt_sha256": claimed}) + b"\n"
        ):
            raise ExposureViewError("PTV2 v2 tokenized receipt digest mismatch")
        expected = {
            "strategy": corpus.strategy,
            "occurrence_count": corpus.occurrence_count,
            "trainer_epochs": corpus.trainer_epochs,
            "assistant_tokens": corpus.assistant_tokens,
            "serialized_tokens": corpus.serialized_tokens,
            "packed_sequence_lower_bound": corpus.packed_sequence_lower_bound,
            "unique_prompt_count": corpus.unique_prompt_count,
            "natural_duplicate_count": corpus.natural_duplicate_count,
            "constructed_repeat_count": corpus.constructed_repeat_count,
            "milestone_occurrences": list(corpus.milestone_occurrences),
            "milestone_steps": list(corpus.milestone_steps),
            "segment_occurrences": list(corpus.segment_occurrences),
            "segment_steps": list(corpus.segment_steps),
            "cumulative_segment_steps": list(corpus.cumulative_segment_steps),
            "segment_final_valid_occurrences": list(corpus.segment_final_valid_occurrences),
            "tokenizer_sha256": corpus.tokenizer_sha256,
            "chat_template_sha256": corpus.chat_template_sha256,
            "assistant_loss_target_sha256": corpus.assistant_loss_target_sha256,
            "training_config_sha256": corpus.training_config_sha256,
            "source_response_root_sha256": corpus.source_response_root_sha256,
            "ordered_occurrences_sha256": corpus.ordered_occurrences_sha256,
            "selection_sha256": corpus.selection_sha256,
            "database_path": "records.sqlite3",
            "database_sha256": corpus.tokenized_sha256,
            "database_bytes": database_path.stat().st_size,
        }
        historical_expected = {
            "historical_parent_receipt_sha256": corpus.historical_parent_receipt_sha256,
            "historical_parent_tokenized_sha256": corpus.historical_parent_tokenized_sha256,
            "historical_prefix_record_stream_sha256": corpus.historical_prefix_record_stream_sha256,
        }
        if corpus.strategy == "H-historical-cyclic":
            expected |= historical_expected | {
                "historical_prefix_occurrence_count": corpus.occurrence_count
            }
        elif any(value is not None for value in historical_expected.values()):
            raise ExposureViewError("PTV2 non-historical corpus carries historical lineage")
        if any(receipt.get(key) != value for key, value in expected.items()):
            raise ExposureViewError("PTV2 v2 tokenized receipt does not match the claimed corpus")
        base_root = receipt.get("base_occurrence_multiplicity_sha256")
        if not isinstance(base_root, str):
            raise ExposureViewError("PTV2 base occurrence multiplicity root is missing")
        _require_digest("PTV2 base occurrence multiplicity", base_root)
        weights = _flatten_ptv2_histogram(receipt.get("bucket_assistant_token_histogram"))
        if sum(weights.values()) != receipt["assistant_tokens"]:
            raise ExposureViewError("PTV2 v2 bucket histogram does not match token total")
        count = int(receipt["occurrence_count"])
        try:
            allocation = int(os.environ.get("SLURM_CPUS_PER_TASK", os.cpu_count() or 1))
        except ValueError as error:
            raise ExposureViewError("PTV2 exposure CPU allocation is invalid") from error
        if allocation < 1:
            raise ExposureViewError("PTV2 exposure CPU allocation is invalid")
        effective_workers = min(workers, allocation, count)
        boundaries = [count * index // effective_workers for index in range(effective_workers + 1)]
        if effective_workers == 1:
            partitions = [_read_ptv2_source_range(database_path, 0, count)]
        else:
            results: dict[int, list[_PTV2ExposureSourceRow]] = {}
            with ProcessPoolExecutor(
                max_workers=effective_workers,
                mp_context=get_context("fork"),
                initializer=_initialize_task8_worker,
            ) as executor:
                futures = {
                    executor.submit(_read_ptv2_source_range, database_path, start, stop): index
                    for index, (start, stop) in enumerate(pairwise(boundaries))
                    if start < stop
                }
                try:
                    for future in as_completed(futures):
                        results[futures[future]] = future.result()
                except BaseException:
                    for future in futures:
                        future.cancel()
                    raise
            partitions = [results[index] for index in range(len(results))]
        rows = [row for partition in partitions for row in partition]
        if len(rows) != count or any(row.ordinal != index for index, row in enumerate(rows)):
            raise ExposureViewError("PTV2 v2 source ordinals are not exact and contiguous")
        observed_weights: Counter[tuple[str, str]] = Counter()
        multiplicity = sha256()
        for row in rows:
            observed_weights[(row.cell, row.language)] += row.assistant_tokens
            multiplicity.update(
                canonical_json(
                    [
                        row.prompt_uuid,
                        row.source_identity_sha256,
                        row.source_row,
                        row.cell,
                        row.language,
                        row.reuse_index,
                    ]
                )
                + b"\n"
            )
        if dict(observed_weights) != weights or multiplicity.hexdigest() != base_root:
            raise ExposureViewError("PTV2 v2 source identity/multiplicity reconciliation failed")
        return _VerifiedPTV2ExposureSource(
            rows,
            weights,
            base_root,
            effective_workers,
            database_path,
            MappingProxyType(dict(receipt)),
            temporary,
        )
    except BaseException:
        temporary.cleanup()
        raise


def _ptv2_interleaved_rows(
    rows: list[_PTV2ExposureSourceRow],
    weights: Mapping[tuple[str, str], int],
    *,
    cyclic: bool,
) -> Iterator[tuple[_PTV2ExposureSourceRow, int]]:
    buckets: dict[tuple[str, str], list[_PTV2ExposureSourceRow]] = {}
    for row in rows:
        buckets.setdefault((row.cell, row.language), []).append(row)
    for values in buckets.values():
        values.sort(
            key=lambda row: (
                row.ordinal,
                row.source_identity_sha256,
                row.source_row,
                row.prompt_uuid,
                row.reuse_index,
            )
        )
    positions = dict.fromkeys(buckets, 0)
    cycles = dict.fromkeys(buckets, 0)
    emitted = dict.fromkeys(buckets, 0)
    queue: list[tuple[Any, ...]] = []

    def push(bucket: tuple[str, str]) -> None:
        row = buckets[bucket][positions[bucket]]
        heapq.heappush(
            queue,
            (
                Fraction(emitted[bucket] + row.assistant_tokens, weights[bucket]),
                bucket[0],
                bucket[1],
                row.ordinal,
                cycles[bucket],
                row.source_identity_sha256,
                row.source_row,
                row.prompt_uuid,
                row.reuse_index,
                bucket,
            ),
        )

    for bucket in sorted(buckets):
        push(bucket)
    while queue:
        *_, bucket = heapq.heappop(queue)
        row = buckets[bucket][positions[bucket]]
        cycle = cycles[bucket]
        yield row, cycle
        emitted[bucket] += row.assistant_tokens
        positions[bucket] += 1
        if positions[bucket] == len(buckets[bucket]):
            if not cyclic:
                continue
            positions[bucket] = 0
            cycles[bucket] += 1
        push(bucket)


def _read_ptv2_payload_batch(
    connection: sqlite3.Connection, source_rows: list[_PTV2ExposureSourceRow]
) -> list[tuple[list[int], list[int], int]]:
    ordinals = sorted({row.ordinal for row in source_rows})
    placeholders = ",".join("?" for _ in ordinals)
    try:
        payloads = {
            ordinal: (json.loads(ids_json), json.loads(mask_json), assistant_tokens)
            for ordinal, ids_json, mask_json, assistant_tokens in connection.execute(
                "SELECT ordinal,input_ids_json,loss_mask_json,assistant_tokens FROM records "
                f"WHERE ordinal IN ({placeholders})",
                ordinals,
            )
        }
    except (json.JSONDecodeError, sqlite3.Error) as error:
        raise ExposureViewError("PTV2 exposure source payload batch is invalid") from error
    if set(payloads) != set(ordinals):
        raise ExposureViewError("PTV2 exposure source payload batch is incomplete")
    ordered = []
    for row in source_rows:
        input_ids, loss_mask, assistant_tokens = payloads[row.ordinal]
        if (
            not _ptv2_token_payload_is_exact(input_ids, loss_mask, assistant_tokens)
            or assistant_tokens != row.assistant_tokens
        ):
            raise ExposureViewError("PTV2 exposure source payload is invalid")
        ordered.append((input_ids, loss_mask, assistant_tokens))
    return ordered


def _ptv2_exposure_metrics(
    records_path: Path, weights: Mapping[tuple[str, str], int]
) -> dict[str, Any]:
    total_weight = sum(weights.values())
    realized: Counter[tuple[str, str]] = Counter()
    occurrence_digest = sha256()
    boundary_digest = sha256()
    cumulative = row_count = boundary_count = maximum = 0
    required = {
        "ordinal",
        "base_ordinal",
        "cycle_index",
        "prompt_uuid",
        "source_identity_sha256",
        "source_row",
        "cell",
        "language",
        "reuse_index",
        "input_ids",
        "loss_mask",
        "assistant_tokens",
        "cumulative_assistant_tokens",
    }
    with records_path.open("rb") as source:
        for raw in source:
            try:
                row = json.loads(raw)
            except json.JSONDecodeError as error:
                raise ExposureViewError(
                    "PTV2 exposure reconciliation found invalid JSON"
                ) from error
            if set(row) != required or canonical_json(row) + b"\n" != raw:
                raise ExposureViewError("PTV2 exposure reconciliation found a row schema mismatch")
            mask = row["loss_mask"]
            ids = row["input_ids"]
            retained = row["assistant_tokens"]
            if (
                row["ordinal"] != row_count
                or not _ptv2_token_payload_is_exact(ids, mask, retained)
                or isinstance(retained, bool)
                or retained < 1
                or row["cumulative_assistant_tokens"] != cumulative + retained
            ):
                raise ExposureViewError("PTV2 exposure reconciliation found invalid mask semantics")
            bucket = (row["cell"], row["language"])
            if bucket not in weights:
                raise ExposureViewError("PTV2 exposure reconciliation found an unknown bucket")
            realized[bucket] += retained
            cumulative += retained
            row_count += 1
            occurrence_digest.update(
                canonical_json(
                    [
                        row["base_ordinal"],
                        row["cycle_index"],
                        row["prompt_uuid"],
                        row["source_identity_sha256"],
                        row["source_row"],
                        row["cell"],
                        row["language"],
                        row["reuse_index"],
                    ]
                )
                + b"\n"
            )
            for candidate, weight in weights.items():
                maximum = max(
                    maximum,
                    abs(realized[candidate] * total_weight - cumulative * weight),
                )
            if row_count % 512 == 0:
                target = _ptv2_target_histogram(weights, cumulative)
                boundary = {
                    "row_count": row_count,
                    "cumulative_assistant_tokens": cumulative,
                    "target_bucket_assistant_tokens": _nested_ptv2_histogram(target),
                    "realized_bucket_assistant_tokens": _nested_ptv2_histogram(realized),
                }
                boundary_digest.update(canonical_json(boundary) + b"\n")
                boundary_count += 1
    if row_count == 0 or row_count % 512 != 0:
        raise ExposureViewError("PTV2 exposure reconciliation requires complete 512-row batches")
    return {
        "assistant_tokens": cumulative,
        "row_count": row_count,
        "realized": dict(realized),
        "occurrence_sha256": occurrence_digest.hexdigest(),
        "batch_boundary_count": boundary_count,
        "batch_boundary_sha256": boundary_digest.hexdigest(),
        "maximum": maximum,
        "denominator": total_weight,
    }


def _reconcile_ptv2_exposure_order_and_trim(
    records_path: Path,
    database_path: Path,
    source_rows: list[_PTV2ExposureSourceRow],
    weights: Mapping[tuple[str, str], int],
    row_count: int,
    target_tokens: int,
    *,
    cyclic: bool,
) -> None:
    expected = _ptv2_interleaved_rows(source_rows, weights, cyclic=cyclic)
    final_start = row_count - 512
    final_rows: list[tuple[dict[str, Any], list[int]]] = []
    connection = sqlite3.connect(f"file:{database_path}?mode=ro", uri=True)
    try:
        with records_path.open("rb") as source:
            ordinal = 0
            while raw_batch := list(islice(source, 512)):
                rows = [json.loads(raw) for raw in raw_batch]
                expected_batch = [next(expected) for _ in rows]
                base_payloads = _read_ptv2_payload_batch(
                    connection, [value[0] for value in expected_batch]
                )
                for row, (source_row, cycle_index), (base_ids, base_mask, _) in zip(
                    rows, expected_batch, base_payloads, strict=True
                ):
                    identity = (
                        row.get("base_ordinal"),
                        row.get("cycle_index"),
                        row.get("prompt_uuid"),
                        row.get("source_identity_sha256"),
                        row.get("source_row"),
                        row.get("cell"),
                        row.get("language"),
                        row.get("reuse_index"),
                    )
                    expected_identity = (
                        source_row.ordinal,
                        cycle_index,
                        source_row.prompt_uuid,
                        source_row.source_identity_sha256,
                        source_row.source_row,
                        source_row.cell,
                        source_row.language,
                        source_row.reuse_index,
                    )
                    if identity != expected_identity or row.get("input_ids") != base_ids:
                        raise ExposureViewError(
                            "PTV2 exposure reconciliation found an order or identity mismatch"
                        )
                    if ordinal < final_start:
                        if row.get("loss_mask") != base_mask:
                            raise ExposureViewError(
                                "PTV2 exposure reconciliation found a trim before the final batch"
                            )
                    else:
                        final_rows.append((row, base_mask))
                    ordinal += 1
        if len(final_rows) != 512:
            raise ExposureViewError("PTV2 exposure reconciliation found an incomplete final batch")
        base_total = sum(sum(mask) for _, mask in final_rows)
        before_final = target_tokens - sum(row["assistant_tokens"] for row, _ in final_rows)
        surplus = before_final + base_total - target_tokens
        if surplus < 0:
            raise ExposureViewError("PTV2 exposure reconciliation found a short final batch")
        expected_masks = [list(mask) for _, mask in final_rows]
        for index in range(511, -1, -1):
            removable = sum(expected_masks[index]) - 1
            removed = min(surplus, removable)
            if removed:
                expected_masks[index] = _trim_mask(
                    expected_masks[index], sum(expected_masks[index]) - removed
                )
                surplus -= removed
        if surplus or any(
            row["loss_mask"] != expected_mask
            for (row, _), expected_mask in zip(final_rows, expected_masks, strict=True)
        ):
            raise ExposureViewError(
                "PTV2 exposure reconciliation found a non-reverse final-batch trim"
            )
    except (json.JSONDecodeError, sqlite3.Error) as error:
        raise ExposureViewError("PTV2 exposure order reconciliation failed") from error
    finally:
        connection.close()


def _read_ptv2_self_hashed_receipt(
    path: Path, label: str, *, max_bytes: int = 1_048_576
) -> tuple[dict[str, Any], str, bytes]:
    raw = _read_ptv2_regular_stable(path, label, max_bytes=max_bytes)
    try:
        payload = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ExposureViewError(f"{label} is not JSON") from error
    if not isinstance(payload, dict):
        raise ExposureViewError(f"{label} is malformed")
    body = dict(payload)
    claimed = body.pop("receipt_sha256", None)
    if (
        not isinstance(claimed, str)
        or _SHA256.fullmatch(claimed) is None
        or claimed != sha256(canonical_json(body)).hexdigest()
        or raw != canonical_json(body | {"receipt_sha256": claimed}) + b"\n"
    ):
        raise ExposureViewError(f"{label} digest mismatch")
    return body, claimed, raw


def _ptv2_execution_evidence(
    bundle: Path, scientific_receipt_sha256: str, records_sha256: str
) -> dict[str, Any]:
    execution, claimed, raw = _read_ptv2_self_hashed_receipt(
        bundle / "EXECUTION.json", "PTV2 exposure execution receipt"
    )
    required = {
        "schema_version",
        "requested_workers",
        "effective_workers",
        "started_at_ns",
        "finished_at_ns",
        "elapsed_seconds",
        "scientific_receipt_sha256",
        "records_sha256",
    }
    requested = execution.get("requested_workers")
    effective = execution.get("effective_workers")
    started = execution.get("started_at_ns")
    finished = execution.get("finished_at_ns")
    elapsed = execution.get("elapsed_seconds")
    if (
        set(execution) != required
        or execution.get("schema_version") != 1
        or type(requested) is not int
        or requested not in {1, 96}
        or type(effective) is not int
        or not 1 <= effective <= requested
        or type(started) is not int
        or type(finished) is not int
        or started < 1
        or finished < started
        or isinstance(elapsed, bool)
        or not isinstance(elapsed, (int, float))
        or elapsed < 0
        or execution.get("scientific_receipt_sha256") != scientific_receipt_sha256
        or execution.get("records_sha256") != records_sha256
    ):
        raise ExposureViewError("PTV2 exposure execution receipt reconciliation failed")
    return {
        "path": "EXECUTION.json",
        "bytes": len(raw),
        "sha256": sha256(raw).hexdigest(),
        "receipt_sha256": claimed,
        "requested_workers": requested,
        "effective_workers": effective,
        "started_at_ns": started,
        "finished_at_ns": finished,
        "elapsed_seconds": elapsed,
    }


def _ptv2_bundle_execution_evidence(bundle: Path, scientific_receipt_sha256: str) -> dict[str, Any]:
    scientific, claimed, _ = _read_ptv2_self_hashed_receipt(
        bundle / "SCIENTIFIC.json", "PTV2 exposure scientific receipt"
    )
    if claimed != scientific_receipt_sha256:
        raise ExposureViewError("PTV2 exposure scientific receipt changed after validation")
    records_sha256 = scientific.get("records_sha256")
    if not isinstance(records_sha256, str) or _SHA256.fullmatch(records_sha256) is None:
        raise ExposureViewError("PTV2 exposure scientific records digest is invalid")
    return _ptv2_execution_evidence(bundle, claimed, records_sha256)


def _validate_ptv2_scientific_receipt_schema(receipt: Mapping[str, Any]) -> None:
    required = {
        "schema_version",
        "strategy",
        "ordering_algorithm",
        "bucket_tie_break",
        "repeat_policy",
        "target_assistant_tokens",
        "records_path",
        "records_bytes",
        "records_sha256",
        "row_count",
        "batch_size_rows",
        "final_batch_reverse_trim_only",
        "tokenized_sha256",
        "selection_sha256",
        "source_response_root_sha256",
        "tokenizer_sha256",
        "chat_template_sha256",
        "assistant_loss_target_sha256",
        "training_config_sha256",
        "base_occurrence_multiplicity_sha256",
        "target_bucket_assistant_tokens",
        "realized_bucket_assistant_tokens",
        "exposure_occurrence_stream_sha256",
        "batch_boundary_count",
        "batch_boundary_histogram_sha256",
        "max_prefix_discrepancy",
    }
    exact_ints = (
        "schema_version",
        "target_assistant_tokens",
        "records_bytes",
        "row_count",
        "batch_size_rows",
        "batch_boundary_count",
    )
    digest_fields = (
        "records_sha256",
        "tokenized_sha256",
        "selection_sha256",
        "source_response_root_sha256",
        "tokenizer_sha256",
        "chat_template_sha256",
        "assistant_loss_target_sha256",
        "training_config_sha256",
        "base_occurrence_multiplicity_sha256",
        "exposure_occurrence_stream_sha256",
        "batch_boundary_histogram_sha256",
    )
    discrepancy = receipt.get("max_prefix_discrepancy")
    try:
        _flatten_ptv2_histogram(receipt.get("target_bucket_assistant_tokens"))
        _flatten_ptv2_histogram(receipt.get("realized_bucket_assistant_tokens"))
    except ExposureViewError as error:
        raise ExposureViewError("PTV2 scientific receipt schema is malformed") from error
    if (
        set(receipt) != required
        or any(type(receipt.get(field)) is not int for field in exact_ints)
        or receipt.get("schema_version") != 2
        or any(
            not isinstance(receipt.get(field), str) or _SHA256.fullmatch(receipt[field]) is None
            for field in digest_fields
        )
        or any(
            not isinstance(receipt.get(field), str)
            for field in ("strategy", "ordering_algorithm", "bucket_tie_break", "repeat_policy")
        )
        or not isinstance(receipt.get("records_path"), str)
        or type(receipt.get("final_batch_reverse_trim_only")) is not bool
        or not isinstance(discrepancy, dict)
        or set(discrepancy) != {"numerator", "denominator", "bound_numerator"}
        or any(type(discrepancy.get(field)) is not int for field in discrepancy)
    ):
        raise ExposureViewError("PTV2 scientific receipt schema is malformed")


def _validate_ptv2_completion_receipt_schema(
    receipt: Mapping[str, Any], arm_order: tuple[str, ...]
) -> None:
    required = {
        "schema_version",
        "arms",
        "runtime_screen_tokens",
        "scientific_tokens",
        "corpus_receipts",
        "runtime_artifacts",
        "scientific_artifacts",
        "execution_artifacts",
    }
    evidence_keys = {
        "path",
        "bytes",
        "sha256",
        "receipt_sha256",
        "requested_workers",
        "effective_workers",
        "started_at_ns",
        "finished_at_ns",
        "elapsed_seconds",
    }
    arms = set(arm_order)
    digests = (
        receipt.get("corpus_receipts"),
        receipt.get("runtime_artifacts"),
        receipt.get("scientific_artifacts"),
    )
    execution = receipt.get("execution_artifacts")
    malformed = (
        set(receipt) != required
        or type(receipt.get("schema_version")) is not int
        or receipt.get("schema_version") != 2
        or receipt.get("arms") != list(arm_order)
        or type(receipt.get("runtime_screen_tokens")) is not int
        or type(receipt.get("scientific_tokens")) is not int
        or any(not isinstance(values, dict) or set(values) != arms for values in digests)
        or any(
            not isinstance(value, str) or _SHA256.fullmatch(value) is None
            for values in digests
            if isinstance(values, dict)
            for value in values.values()
        )
        or not isinstance(execution, dict)
        or set(execution) != {"runtime", "scientific"}
    )
    if malformed:
        raise ExposureViewError("PTV2 completion receipt schema is malformed")
    assert isinstance(execution, dict)
    for phase in ("runtime", "scientific"):
        phase_evidence = execution[phase]
        if not isinstance(phase_evidence, dict) or set(phase_evidence) != arms:
            raise ExposureViewError("PTV2 completion receipt schema is malformed")
        for evidence in phase_evidence.values():
            if (
                not isinstance(evidence, dict)
                or set(evidence) != evidence_keys
                or evidence.get("path") != "EXECUTION.json"
                or type(evidence.get("bytes")) is not int
                or type(evidence.get("requested_workers")) is not int
                or type(evidence.get("effective_workers")) is not int
                or type(evidence.get("started_at_ns")) is not int
                or type(evidence.get("finished_at_ns")) is not int
                or isinstance(evidence.get("elapsed_seconds"), bool)
                or not isinstance(evidence.get("elapsed_seconds"), (int, float))
                or any(
                    not isinstance(evidence.get(field), str)
                    or _SHA256.fullmatch(evidence[field]) is None
                    for field in ("sha256", "receipt_sha256")
                )
            ):
                raise ExposureViewError("PTV2 completion receipt schema is malformed")


def _validate_ptv2_exposure_bundle(
    bundle: Path,
    corpus: PTV2OnePassCorpus,
    *,
    expected_target_tokens: int | None = None,
    verified_source: _VerifiedPTV2ExposureSource | None = None,
) -> str:
    records_path = bundle / "records.jsonl"
    receipt_path = bundle / "SCIENTIFIC.json"
    if (
        not bundle.is_dir()
        or bundle.is_symlink()
        or not records_path.is_file()
        or records_path.is_symlink()
        or not receipt_path.is_file()
        or receipt_path.is_symlink()
    ):
        raise ExposureViewError("PTV2 exposure reconciliation found an unsafe bundle")
    receipt, claimed, _ = _read_ptv2_self_hashed_receipt(
        receipt_path, "PTV2 exposure scientific receipt"
    )
    _validate_ptv2_scientific_receipt_schema(receipt)
    records_sha256 = receipt.get("records_sha256")
    if not isinstance(records_sha256, str) or _SHA256.fullmatch(records_sha256) is None:
        raise ExposureViewError("PTV2 exposure reconciliation found a records digest mismatch")
    _ptv2_execution_evidence(bundle, claimed, records_sha256)
    source = verified_source or _load_ptv2_exposure_source(corpus, 1)
    owns_source = verified_source is None
    try:
        metrics = _ptv2_exposure_metrics(records_path, source.weights)
        if (
            expected_target_tokens is not None
            and metrics["assistant_tokens"] != expected_target_tokens
        ):
            raise ExposureViewError("PTV2 exposure reconciliation found the wrong target")
        discrepancy_bound = max(row.assistant_tokens for row in source.rows) * sum(
            source.weights.values()
        )
        if metrics["maximum"] > discrepancy_bound:
            raise ExposureViewError("PTV2 exposure prefix discrepancy exceeds the packet bound")
        _reconcile_ptv2_exposure_order_and_trim(
            records_path,
            source.database_path,
            source.rows,
            source.weights,
            metrics["row_count"],
            metrics["assistant_tokens"],
            cyclic=source.receipt["strategy"] == "H-historical-cyclic",
        )
        lineage = source.receipt
        expected = {
            "strategy": lineage["strategy"],
            "repeat_policy": (
                "authenticated-historical-cyclic-v1"
                if lineage["strategy"] == "H-historical-cyclic"
                else "authenticated-one-pass-v1"
            ),
            "target_assistant_tokens": metrics["assistant_tokens"],
            "records_path": records_path.name,
            "records_bytes": records_path.stat().st_size,
            "records_sha256": _sha256_file(records_path),
            "row_count": metrics["row_count"],
            "tokenized_sha256": lineage["database_sha256"],
            "selection_sha256": lineage["selection_sha256"],
            "source_response_root_sha256": lineage["source_response_root_sha256"],
            "tokenizer_sha256": lineage["tokenizer_sha256"],
            "chat_template_sha256": lineage["chat_template_sha256"],
            "assistant_loss_target_sha256": lineage["assistant_loss_target_sha256"],
            "training_config_sha256": lineage["training_config_sha256"],
            "base_occurrence_multiplicity_sha256": source.base_root,
            "target_bucket_assistant_tokens": _nested_ptv2_histogram(
                _ptv2_target_histogram(source.weights, metrics["assistant_tokens"])
            ),
            "realized_bucket_assistant_tokens": _nested_ptv2_histogram(metrics["realized"]),
            "exposure_occurrence_stream_sha256": metrics["occurrence_sha256"],
            "batch_boundary_count": metrics["batch_boundary_count"],
            "batch_boundary_histogram_sha256": metrics["batch_boundary_sha256"],
            "max_prefix_discrepancy": {
                "numerator": metrics["maximum"],
                "denominator": metrics["denominator"],
                "bound_numerator": discrepancy_bound,
            },
        }
        if any(receipt.get(key) != value for key, value in expected.items()):
            raise ExposureViewError("PTV2 exposure reconciliation failed")
        if (
            receipt.get("ordering_algorithm") != "assistant-token-weighted-fair-v1"
            or receipt.get("bucket_tie_break") != "cell-language-base_ordinal-cycle-identity"
            or receipt.get("batch_size_rows") != 512
            or receipt.get("final_batch_reverse_trim_only") is not True
        ):
            raise ExposureViewError("PTV2 exposure reconciliation found a policy mismatch")
        return claimed
    finally:
        if owns_source:
            source.close()


def _materialize_ptv2_exposure(
    corpus: PTV2OnePassCorpus, target_tokens: int, *, workers: int = 1
) -> str:
    """Publish a deterministic assistant-token weighted, batch-aligned v2 exposure."""
    if isinstance(target_tokens, bool) or not isinstance(target_tokens, int) or target_tokens < 512:
        raise ExposureViewError("PTV2 exposure target must support at least one 512-row batch")
    if corpus.strategy in {"A-repair", "B-balanced"} and target_tokens > corpus.assistant_tokens:
        raise ExposureViewError("PTV2 A/B exposure exceeds authenticated one-pass reachability")
    started = time.monotonic_ns()
    root = Path(corpus.tokenized_path).parent.parent / f"{corpus.strategy.lower()}-exposures"
    if os.path.lexists(root) and (root.is_symlink() or not root.is_dir()):
        raise ExposureViewError("PTV2 exposure output root is unsafe")
    stem = f"{corpus.strategy.lower()}-{target_tokens}-assistant-tokens-v2"
    destination = root / stem
    if os.path.lexists(destination):
        return _validate_ptv2_exposure_bundle(
            destination, corpus, expected_target_tokens=target_tokens
        )
    source = _load_ptv2_exposure_source(corpus, workers)
    try:
        _ensure_ptv2_durable_directory(root)
    except BaseException:
        source.close()
        raise
    temporary = Path(tempfile.mkdtemp(prefix=f".{stem}.partial-", dir=root))
    records_path = temporary / "records.jsonl"
    receipt_path = temporary / "SCIENTIFIC.json"
    execution_path = temporary / "EXECUTION.json"
    connection = sqlite3.connect(f"file:{source.database_path}?mode=ro", uri=True)
    cumulative = output_ordinal = 0
    try:
        cyclic = corpus.strategy == "H-historical-cyclic"
        stream = _ptv2_interleaved_rows(source.rows, source.weights, cyclic=cyclic)
        with _open_nofollow_exclusive(records_path) as output:
            while cumulative < target_tokens:
                batch = list(islice(stream, 512))
                if len(batch) != 512:
                    raise ExposureViewError(
                        "PTV2 A/B exposure cannot form another batch within one-pass reachability"
                    )
                payloads: list[tuple[_PTV2ExposureSourceRow, int, list[int], list[int]]] = []
                batch_rows = [value[0] for value in batch]
                batch_payloads = _read_ptv2_payload_batch(connection, batch_rows)
                for (source_row, cycle_index), (input_ids, loss_mask, _) in zip(
                    batch, batch_payloads, strict=True
                ):
                    payloads.append((source_row, cycle_index, input_ids, loss_mask))
                batch_tokens = sum(value[0].assistant_tokens for value in payloads)
                final = cumulative + batch_tokens >= target_tokens
                if final:
                    surplus = cumulative + batch_tokens - target_tokens
                    removable = sum(value[0].assistant_tokens - 1 for value in payloads)
                    if surplus > removable:
                        raise ExposureViewError(
                            "PTV2 exact target cannot preserve one token in every final-batch row"
                        )
                    for index in range(len(payloads) - 1, -1, -1):
                        if surplus == 0:
                            break
                        source_row, cycle_index, input_ids, loss_mask = payloads[index]
                        removed = min(surplus, source_row.assistant_tokens - 1)
                        if removed:
                            payloads[index] = (
                                source_row,
                                cycle_index,
                                input_ids,
                                _trim_mask(loss_mask, source_row.assistant_tokens - removed),
                            )
                            surplus -= removed
                    if surplus:
                        raise ExposureViewError("PTV2 final-batch reverse trim is incomplete")
                for source_row, cycle_index, input_ids, loss_mask in payloads:
                    retained = sum(loss_mask)
                    cumulative += retained
                    record = {
                        "ordinal": output_ordinal,
                        "base_ordinal": source_row.ordinal,
                        "cycle_index": cycle_index,
                        "prompt_uuid": source_row.prompt_uuid,
                        "source_identity_sha256": source_row.source_identity_sha256,
                        "source_row": source_row.source_row,
                        "cell": source_row.cell,
                        "language": source_row.language,
                        "reuse_index": source_row.reuse_index,
                        "input_ids": input_ids,
                        "loss_mask": loss_mask,
                        "assistant_tokens": retained,
                        "cumulative_assistant_tokens": cumulative,
                    }
                    output.write(canonical_json(record) + b"\n")
                    output_ordinal += 1
                if final:
                    break
            output.flush()
            os.fsync(output.fileno())
        if cumulative != target_tokens:
            raise ExposureViewError("PTV2 tokenized corpus cannot materialize the exact exposure")
        metrics = _ptv2_exposure_metrics(records_path, source.weights)
        discrepancy_bound = max(row.assistant_tokens for row in source.rows) * sum(
            source.weights.values()
        )
        if metrics["maximum"] > discrepancy_bound:
            raise ExposureViewError("PTV2 exposure prefix discrepancy exceeds the packet bound")
        lineage = source.receipt
        receipt = {
            "schema_version": 2,
            "strategy": lineage["strategy"],
            "ordering_algorithm": "assistant-token-weighted-fair-v1",
            "bucket_tie_break": "cell-language-base_ordinal-cycle-identity",
            "repeat_policy": (
                "authenticated-historical-cyclic-v1" if cyclic else "authenticated-one-pass-v1"
            ),
            "target_assistant_tokens": target_tokens,
            "records_path": records_path.name,
            "records_bytes": records_path.stat().st_size,
            "records_sha256": _sha256_file(records_path),
            "row_count": metrics["row_count"],
            "batch_size_rows": 512,
            "final_batch_reverse_trim_only": True,
            "tokenized_sha256": lineage["database_sha256"],
            "selection_sha256": lineage["selection_sha256"],
            "source_response_root_sha256": lineage["source_response_root_sha256"],
            "tokenizer_sha256": lineage["tokenizer_sha256"],
            "chat_template_sha256": lineage["chat_template_sha256"],
            "assistant_loss_target_sha256": lineage["assistant_loss_target_sha256"],
            "training_config_sha256": lineage["training_config_sha256"],
            "base_occurrence_multiplicity_sha256": source.base_root,
            "target_bucket_assistant_tokens": _nested_ptv2_histogram(
                _ptv2_target_histogram(source.weights, target_tokens)
            ),
            "realized_bucket_assistant_tokens": _nested_ptv2_histogram(metrics["realized"]),
            "exposure_occurrence_stream_sha256": metrics["occurrence_sha256"],
            "batch_boundary_count": metrics["batch_boundary_count"],
            "batch_boundary_histogram_sha256": metrics["batch_boundary_sha256"],
            "max_prefix_discrepancy": {
                "numerator": metrics["maximum"],
                "denominator": metrics["denominator"],
                "bound_numerator": discrepancy_bound,
            },
        }
        receipt_sha256 = sha256(canonical_json(receipt)).hexdigest()
        _write_exclusive(
            receipt_path, canonical_json(receipt | {"receipt_sha256": receipt_sha256}) + b"\n"
        )
        execution = {
            "schema_version": 1,
            "requested_workers": workers,
            "effective_workers": source.effective_workers,
            "started_at_ns": time.time_ns() - (time.monotonic_ns() - started),
            "finished_at_ns": time.time_ns(),
            "elapsed_seconds": round((time.monotonic_ns() - started) / 1_000_000_000, 6),
            "scientific_receipt_sha256": receipt_sha256,
            "records_sha256": receipt["records_sha256"],
        }
        execution_sha256 = sha256(canonical_json(execution)).hexdigest()
        _write_exclusive(
            execution_path,
            canonical_json(execution | {"receipt_sha256": execution_sha256}) + b"\n",
        )
        for path in (records_path, receipt_path, execution_path):
            _fsync_file(path)
        _validate_ptv2_exposure_bundle(
            temporary,
            corpus,
            expected_target_tokens=target_tokens,
            verified_source=source,
        )
        _fsync_directory(temporary)
        installed = os.lstat(temporary)
        try:
            _rename_no_replace(temporary, destination)
        except Exception as error:
            raise ExposureViewError(
                "PTV2 exposure artifact is immutable and already exists"
            ) from error
        try:
            _fsync_directory(root)
        except BaseException:
            _rollback_ptv2_publication(destination, installed)
            raise
        return receipt_sha256
    finally:
        connection.close()
        source.close()
        if temporary.exists():
            shutil.rmtree(temporary)


def _materialize_ptv2_exposure_set(
    corpora: tuple[PTV2OnePassCorpus, ...],
    *,
    runtime_screen_tokens: int,
    scientific_tokens: int,
    workers: int,
    expected_completion_receipt_sha256: str | None = None,
) -> tuple[dict[str, str], dict[str, str], str]:
    arm_order = ("A-repair", "B-balanced", "H-historical-cyclic")
    by_arm = {corpus.strategy: corpus for corpus in corpora}
    if len(corpora) != 3 or set(by_arm) != set(arm_order):
        raise ExposureViewError("PTV2 completion requires the exact A/B/H arm set")
    corpora = tuple(by_arm[arm] for arm in arm_order)
    roots = {Path(corpus.tokenized_path).parent.parent for corpus in corpora}
    if len(roots) != 1:
        raise ExposureViewError("PTV2 exposure arms must share one publication root")
    study_root = roots.pop()
    root = study_root / "ptv2-study-exposures"
    destination = root / "COMPLETE.json"
    scope = {
        "schema_version": 2,
        "arms": list(arm_order),
        "runtime_screen_tokens": runtime_screen_tokens,
        "scientific_tokens": scientific_tokens,
        "corpus_receipts": {corpus.strategy: corpus.receipt_sha256 for corpus in corpora},
    }
    existing: tuple[dict[str, Any], str, bytes] | None = None
    if os.path.lexists(root):
        if root.is_symlink() or not root.is_dir():
            raise ExposureViewError("PTV2 study completion root is unsafe")
        if os.path.lexists(destination):
            existing = _read_ptv2_self_hashed_receipt(destination, "PTV2 study completion receipt")
            body = existing[0]
            if body.get("arms") != list(arm_order):
                raise ExposureViewError("PTV2 study completion scope is incompatible")
            _validate_ptv2_completion_receipt_schema(body, arm_order)
            if expected_completion_receipt_sha256 is None:
                raise ExposureViewError(
                    "PTV2 completion replay requires a caller-pinned completion identity"
                )
            _require_digest("expected PTV2 completion", expected_completion_receipt_sha256)
            if existing[1] != expected_completion_receipt_sha256:
                raise ExposureViewError("PTV2 study completion identity mismatch")
            if any(body.get(key) != value for key, value in scope.items()):
                raise ExposureViewError("PTV2 study completion scope is incompatible")
            for phase, target in (
                ("runtime", runtime_screen_tokens),
                ("scientific", scientific_tokens),
            ):
                artifact_key = f"{phase}_artifacts"
                artifacts = body.get(artifact_key)
                execution_artifacts = body.get("execution_artifacts")
                phase_execution = (
                    execution_artifacts.get(phase)
                    if isinstance(execution_artifacts, dict)
                    else None
                )
                if not isinstance(artifacts, dict) or not isinstance(phase_execution, dict):
                    raise ExposureViewError("PTV2 study completion execution evidence is missing")
                for corpus in corpora:
                    scientific_sha256 = artifacts.get(corpus.strategy)
                    if not isinstance(scientific_sha256, str):
                        raise ExposureViewError(
                            "PTV2 study completion execution evidence is malformed"
                        )
                    bundle = (
                        study_root
                        / f"{corpus.strategy.lower()}-exposures"
                        / f"{corpus.strategy.lower()}-{target}-assistant-tokens-v2"
                    )
                    observed = _ptv2_bundle_execution_evidence(bundle, scientific_sha256)
                    if observed != phase_execution.get(corpus.strategy):
                        raise ExposureViewError("PTV2 study completion execution evidence changed")
    if existing is None and expected_completion_receipt_sha256 is not None:
        raise ExposureViewError("caller-pinned PTV2 completion receipt is missing")
    runtime = {
        corpus.strategy: _materialize_ptv2_exposure(corpus, runtime_screen_tokens, workers=workers)
        for corpus in corpora
    }
    scientific = {
        corpus.strategy: _materialize_ptv2_exposure(corpus, scientific_tokens, workers=workers)
        for corpus in corpora
    }
    execution_artifacts = {
        phase: {
            corpus.strategy: _ptv2_bundle_execution_evidence(
                study_root
                / f"{corpus.strategy.lower()}-exposures"
                / f"{corpus.strategy.lower()}-{target}-assistant-tokens-v2",
                artifacts[corpus.strategy],
            )
            for corpus in corpora
        }
        for phase, target, artifacts in (
            ("runtime", runtime_screen_tokens, runtime),
            ("scientific", scientific_tokens, scientific),
        )
    }
    _ensure_ptv2_durable_directory(root)
    payload = scope | {
        "runtime_artifacts": runtime,
        "scientific_artifacts": scientific,
        "execution_artifacts": execution_artifacts,
    }
    _validate_ptv2_completion_receipt_schema(payload, arm_order)
    receipt_sha256 = sha256(canonical_json(payload)).hexdigest()
    expected = canonical_json(payload | {"receipt_sha256": receipt_sha256}) + b"\n"
    if existing is not None:
        if existing[2] != expected:
            raise ExposureViewError("PTV2 study completion receipt conflicts with its arms")
        return runtime, scientific, receipt_sha256
    temporary = root / f".COMPLETE.json.partial-{uuid.uuid4().hex}"
    try:
        _write_exclusive(temporary, expected)
        _fsync_file(temporary)
        installed = os.lstat(temporary)
        _rename_no_replace(temporary, destination)
        try:
            _fsync_directory(root)
        except BaseException:
            _rollback_ptv2_publication(destination, installed)
            raise
    finally:
        temporary.unlink(missing_ok=True)
    return runtime, scientific, receipt_sha256


def _validate_ptv2_historical(corpus: PTV2OnePassCorpus) -> None:
    if (
        not isinstance(corpus, PTV2OnePassCorpus)
        or corpus.strategy != "H-historical-cyclic"
        or corpus.occurrence_count != 1_300_000
        or corpus.trainer_epochs != 1
        or corpus.constructed_repeat_count != 0
        or corpus.historical_parent_receipt_sha256 is None
        or corpus.historical_parent_tokenized_sha256 is None
        or corpus.historical_prefix_record_stream_sha256 is None
    ):
        raise ExposureViewError(
            "PTV2 H baseline must bind the immutable 1.3M historical base exactly once"
        )
    for label, digest in (
        ("historical parent receipt", corpus.historical_parent_receipt_sha256),
        ("historical parent tokenized", corpus.historical_parent_tokenized_sha256),
        ("historical prefix record stream", corpus.historical_prefix_record_stream_sha256),
    ):
        _require_digest(label, digest)
    source = _load_ptv2_exposure_source(corpus, 1)
    source.close()


def _require_ptv2_historical_parent(
    historical: PTV2OnePassCorpus, prefix: PTV2OnePassCorpus
) -> None:
    if (
        historical.strategy != "H-historical-cyclic"
        or prefix.strategy != "A-repair"
        or historical.historical_parent_receipt_sha256 != prefix.receipt_sha256
        or historical.historical_parent_tokenized_sha256 != prefix.tokenized_sha256
        or historical.occurrence_count != prefix.segment_occurrences[0]
    ):
        raise ExposureViewError("PTV2 historical parent does not match the A baseline")


def _validate_ptv2_one_pass(corpus: PTV2OnePassCorpus, strategy: str) -> None:
    if not isinstance(corpus, PTV2OnePassCorpus) or corpus.strategy != strategy:
        raise ExposureViewError(f"PTV2 one-pass receipt must be {strategy}")
    if corpus.occurrence_count != 2_000_000:
        raise ExposureViewError("PTV2 one-pass receipt must bind exactly 2M occurrences")
    if corpus.trainer_epochs != 1:
        raise ExposureViewError("PTV2 one-pass receipt must bind exactly one trainer epoch")
    if isinstance(corpus.assistant_tokens, bool) or corpus.assistant_tokens < 1:
        raise ExposureViewError("PTV2 one-pass assistant tokens must be positive")
    if (
        corpus.unique_prompt_count < 1
        or corpus.natural_duplicate_count < 0
        or corpus.constructed_repeat_count < 0
        or corpus.unique_prompt_count
        + corpus.natural_duplicate_count
        + corpus.constructed_repeat_count
        != corpus.occurrence_count
    ):
        raise ExposureViewError("PTV2 one-pass multiplicity summary does not reconcile")
    if corpus.milestone_occurrences != (
        500_224,
        1_000_448,
        1_300_000,
        2_000_000,
    ) or corpus.milestone_steps != (
        977,
        1_954,
        2_540,
        3_908,
    ):
        raise ExposureViewError("PTV2 one-pass receipt has the wrong optimizer milestones")
    if (
        corpus.segment_occurrences != (1_300_000, 700_000)
        or corpus.segment_steps != (2_540, 1_368)
        or corpus.cumulative_segment_steps != (2_540, 3_908)
        or corpus.segment_final_valid_occurrences != (32, 96)
    ):
        raise ExposureViewError("PTV2 one-pass receipt has the wrong two-segment schedule")
    for field in (
        "tokenizer_sha256",
        "chat_template_sha256",
        "assistant_loss_target_sha256",
        "training_config_sha256",
        "source_response_root_sha256",
        "ordered_occurrences_sha256",
        "tokenized_sha256",
        "receipt_sha256",
    ):
        _require_digest(field, getattr(corpus, field))
    source = _load_ptv2_exposure_source(corpus, 1)
    source.close()


@dataclass(frozen=True)
class _TokenizedCorpus:
    schema_version: int
    arm: str
    corpus_sha256: str
    tokenizer_sha256: str
    chat_template_sha256: str
    selection_sha256: str
    training_sequence_length: int
    purpose: str
    policy_sha256: str
    source_policy_sha256: str
    seed: int
    record_count: int
    total_assistant_tokens: int
    paired_selection_bucket_sha256: str
    task5_paired_selection_bucket_sha256: str
    task6_completion_sha256: str
    final_paired_response_sha256: str
    database_path: str
    database_bytes: int
    database_sha256: str
    resume_fingerprint: str


@dataclass(frozen=True)
class _ProductionProof:
    selection_sha256: str
    task5_paired_sha256: str
    record_paired_sha256: str
    source_policy_sha256: str
    task6_completion_sha256: str
    final_paired_response_sha256: str


@dataclass(frozen=True)
class _AuthenticatedResponse:
    selection_sha256: str
    paired_sha256: str
    index_path: Path
    completion_sha256: str


def _require_digest(name: str, value: str) -> None:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ExposureViewError(f"{name} must be an exact lowercase SHA-256")


def _sha256_file(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_exclusive(path: Path, payload: bytes) -> None:
    with path.open("xb") as output:
        output.write(payload)
        output.flush()
        os.fsync(output.fileno())


def _open_nofollow_exclusive(path: Path):
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    return os.fdopen(os.open(path, flags, 0o600), "wb")


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _fsync_file(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _same_ptv2_inode(left: os.stat_result, right: os.stat_result) -> bool:
    return (
        left.st_dev,
        left.st_ino,
        stat.S_IFMT(left.st_mode),
    ) == (
        right.st_dev,
        right.st_ino,
        stat.S_IFMT(right.st_mode),
    )


def _ensure_ptv2_durable_directory(path: Path) -> None:
    """Create one output directory and durably persist its parent entry."""
    if os.path.lexists(path):
        metadata = os.lstat(path)
        if path.is_symlink() or not stat.S_ISDIR(metadata.st_mode):
            raise ExposureViewError("PTV2 output root is unsafe")
        _fsync_directory(path.parent)
        return
    try:
        path.mkdir(mode=0o700)
    except FileExistsError:
        metadata = os.lstat(path)
        if path.is_symlink() or not stat.S_ISDIR(metadata.st_mode):
            raise ExposureViewError("PTV2 output root is unsafe")
        _fsync_directory(path.parent)
        return
    installed = os.lstat(path)
    try:
        _fsync_directory(path.parent)
    except BaseException:
        with suppress(FileNotFoundError, OSError):
            observed = os.lstat(path)
            if _same_ptv2_inode(installed, observed):
                path.rmdir()
        raise


def _rollback_ptv2_publication(destination: Path, installed: os.stat_result) -> None:
    """Best-effort rollback that never deletes an inode other than the one just installed."""
    if not os.path.lexists(destination):
        return
    quarantine = destination.with_name(f".{destination.name}.rollback-{uuid.uuid4().hex}")
    try:
        _rename_no_replace(destination, quarantine)
    except Exception:
        return
    try:
        moved = os.lstat(quarantine)
        if not _same_ptv2_inode(moved, installed):
            with suppress(Exception):
                _rename_no_replace(quarantine, destination)
            return
        if stat.S_ISDIR(moved.st_mode):
            shutil.rmtree(quarantine)
        else:
            quarantine.unlink()
    finally:
        with suppress(OSError):
            _fsync_directory(destination.parent)


def _prepare_ptv2_tokenized_bundle(root: Path, destination: Path) -> Path:
    """Create a private bundle so SQLite and its receipt publish as one unit."""
    if os.path.lexists(destination):
        raise ExposureViewError("PTV2 tokenized output is immutable and already exists")
    partials = tuple(root.glob(f".{destination.name}.partial-*"))
    if partials:
        raise PTV2TokenizedRecoveryError("PTV2 tokenized partial requires typed recovery")
    temporary = root / f".{destination.name}.partial-{uuid.uuid4().hex}"
    temporary.mkdir(mode=0o700)
    metadata = os.lstat(temporary)
    if temporary.is_symlink() or not stat.S_ISDIR(metadata.st_mode):
        raise ExposureViewError("PTV2 tokenized partial is not a private directory")
    descriptor = os.open(temporary, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    return temporary


def _publish_ptv2_tokenized_bundle(temporary: Path, destination: Path) -> None:
    """Durably install the fully authenticated token bundle without replacement."""
    metadata = os.lstat(temporary)
    if (
        not temporary.is_dir()
        or temporary.is_symlink()
        or not stat.S_ISDIR(metadata.st_mode)
        or any(
            (temporary / name).is_symlink() or not (temporary / name).is_file()
            for name in ("records.sqlite3", "TOKENIZED.json")
        )
    ):
        raise ExposureViewError("PTV2 tokenized partial is not a private bundle")
    installed = os.lstat(temporary)
    try:
        _rename_no_replace(temporary, destination)
    except Exception as error:
        raise ExposureViewError("PTV2 tokenized output is immutable and already exists") from error
    if not destination.is_dir() or destination.is_symlink():
        raise ExposureViewError("PTV2 published tokenized bundle is unsafe")
    try:
        _fsync_directory(destination.parent)
    except BaseException:
        _rollback_ptv2_publication(destination, installed)
        raise


def _identity_payload(
    corpus: Any,
    *,
    tokenizer_sha256: str,
    chat_template_sha256: str,
    seed: int,
    training_sequence_length: int,
    purpose: str,
    policy_sha256: str,
    source_policy_sha256: str,
    task5_paired_sha256: str,
    task6_completion_sha256: str,
    final_paired_response_sha256: str,
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "arm": corpus.arm,
        "response_corpus_sha256": corpus.corpus_sha256,
        "source_selection_sha256": corpus.source_selection_sha256,
        "generation_identity_sha256": corpus.generation_identity_sha256,
        "tokenizer_sha256": tokenizer_sha256,
        "chat_template_sha256": chat_template_sha256,
        "seed": seed,
        "training_sequence_length": training_sequence_length,
        "purpose": purpose,
        "policy_sha256": policy_sha256,
        "source_policy_sha256": source_policy_sha256,
        "task5_paired_selection_bucket_sha256": task5_paired_sha256,
        "task6_completion_sha256": task6_completion_sha256,
        "final_paired_response_sha256": final_paired_response_sha256,
    }


def _prepare_root(root: Path, identity: dict[str, Any]) -> str:
    fingerprint = sha256(canonical_json(identity)).hexdigest()
    identity_path = root / "STAGING_IDENTITY.json"
    if root.exists():
        if not root.is_dir() or not identity_path.is_file():
            raise ExposureViewError("existing staging path has no authenticated identity")
        try:
            prior = json.loads(identity_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise ExposureViewError("existing staging identity is malformed") from error
        if prior != identity | {"resume_fingerprint": fingerprint}:
            raise ExposureViewError("resume fingerprint mismatch")
        return fingerprint
    root.mkdir(parents=True)
    _write_exclusive(
        identity_path,
        canonical_json(identity | {"resume_fingerprint": fingerprint}) + b"\n",
    )
    _fsync_directory(root)
    return fingerprint


def _validated_record_sha256(record: Any) -> str:
    try:
        payload = asdict(record)
    except TypeError as error:
        raise ExposureViewError("response corpus contains a non-record value") from error
    claimed = payload.pop("record_sha256", None)
    if not isinstance(claimed, str) or claimed != sha256(canonical_json(payload)).hexdigest():
        raise ExposureViewError(f"promoted record digest mismatch: {record.prompt_uuid}")
    return claimed


def _message_prefix_tokens(
    tokenizer: Any,
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]],
    end: int,
) -> int:
    encoded = tokenizer.apply_chat_template(
        messages[:end],
        tools=tools or None,
        tokenize=True,
        add_generation_prompt=False,
        return_dict=True,
    )
    input_ids = encoded.get("input_ids") if isinstance(encoded, Mapping) else None
    if not isinstance(input_ids, list):
        raise ExposureViewError("tokenizer did not return IDs for a replay boundary")
    return len(input_ids)


def _validate_replay_training_cut(
    tokenizer: Any,
    canonical: dict[str, Any],
    record: Any,
    *,
    training_sequence_length: int,
    full_token_count: int,
) -> None:
    try:
        validate_trajectory(
            canonical,
            source_id=record.prompt_uuid,
            lane=record.lane,
        )
    except TrajectoryValidationError as error:
        raise ExposureViewError(
            f"trainer view rejects replay {record.prompt_uuid}: {error.reason}"
        ) from error
    if full_token_count <= training_sequence_length:
        return
    messages = canonical["messages"]
    tools = canonical.get("tools") or []
    lower = 1
    upper = len(messages)
    while lower < upper:
        middle = (lower + upper) // 2
        if _message_prefix_tokens(tokenizer, messages, tools, middle) > training_sequence_length:
            upper = middle
        else:
            lower = middle + 1
    cut_message = lower - 1
    pending: set[str] = set()
    transaction_start = -1
    for index, message in enumerate(messages):
        calls = message.get("tool_calls") if message.get("role") == "assistant" else None
        if isinstance(calls, list) and calls:
            transaction_start = index
            pending = {
                str(call.get("id") or call.get("tool_call_id"))
                for call in calls
                if isinstance(call, dict)
            }
        if message.get("role") != "tool":
            continue
        pending.remove(str(message.get("tool_call_id") or message.get("id")))
        if pending or not transaction_start <= cut_message <= index:
            continue
        before = _message_prefix_tokens(tokenizer, messages, tools, transaction_start)
        after = _message_prefix_tokens(tokenizer, messages, tools, index + 1)
        if before < training_sequence_length < after:
            raise ExposureViewError(
                f"trainer view rejects replay {record.prompt_uuid}: split_tool_transaction"
            )


def _tokenize_record(
    tokenizer: Any,
    record: Any,
    *,
    training_sequence_length: int,
) -> tuple[list[int], list[int]]:
    try:
        canonical = json.loads(record.canonical_record_json)
    except json.JSONDecodeError as error:
        raise ExposureViewError(f"record is not canonical JSON: {record.prompt_uuid}") from error
    if canonical_json(canonical).decode("utf-8") != record.canonical_record_json:
        raise ExposureViewError(f"record is not canonically encoded: {record.prompt_uuid}")
    if not isinstance(canonical, dict) or not isinstance(canonical.get("messages"), list):
        raise ExposureViewError(f"record has no canonical messages: {record.prompt_uuid}")
    encoded = tokenizer.apply_chat_template(
        canonical["messages"],
        tools=canonical.get("tools") or None,
        tokenize=True,
        add_generation_prompt=False,
        return_dict=True,
        return_assistant_tokens_mask=True,
    )
    if not isinstance(encoded, Mapping):
        raise ExposureViewError("tokenizer did not return a mapping")
    input_ids = encoded.get("input_ids")
    loss_mask = encoded.get("assistant_masks")
    if loss_mask is None:
        loss_mask = encoded.get("assistant_tokens_mask")
    if (
        not isinstance(input_ids, list)
        or not input_ids
        or not all(isinstance(token, int) and not isinstance(token, bool) for token in input_ids)
        or not isinstance(loss_mask, list)
        or len(input_ids) != len(loss_mask)
        or any(value not in (0, 1) for value in loss_mask)
    ):
        raise ExposureViewError("tokenizer did not return aligned IDs and a binary assistant mask")
    if record.lane in {"interactive-swe-replay", "generic-tool-replay"}:
        _validate_replay_training_cut(
            tokenizer,
            canonical,
            record,
            training_sequence_length=training_sequence_length,
            full_token_count=len(input_ids),
        )
    input_ids = input_ids[:training_sequence_length]
    loss_mask = loss_mask[:training_sequence_length]
    if sum(loss_mask) < 1:
        raise ExposureViewError(
            f"record has no assistant-owned tokens after exact retokenization: {record.prompt_uuid}"
        )
    return input_ids, loss_mask


def _load_tokenized_receipt(root: Path, resume_fingerprint: str) -> _TokenizedCorpus | None:
    receipt_path = root / "TOKENIZED.json"
    database_path = root / "tokenized.sqlite3"
    if not receipt_path.exists():
        if database_path.exists():
            raise ExposureViewError("incomplete tokenized staging state requires recovery")
        return None
    try:
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        receipt_sha256 = receipt.pop("receipt_sha256")
    except (OSError, json.JSONDecodeError, KeyError) as error:
        raise ExposureViewError("tokenized staging receipt is malformed") from error
    if (
        receipt_sha256 != sha256(canonical_json(receipt)).hexdigest()
        or receipt.get("resume_fingerprint") != resume_fingerprint
        or receipt.get("database_path") != str(database_path)
        or not database_path.is_file()
        or receipt.get("database_bytes") != database_path.stat().st_size
        or receipt.get("database_sha256") != _sha256_file(database_path)
    ):
        raise ExposureViewError("tokenized staging authentication failed")
    try:
        return _TokenizedCorpus(**receipt)
    except TypeError as error:
        raise ExposureViewError("tokenized staging receipt has an invalid schema") from error


def _stage_tokenized_corpus(
    corpus: Any,
    tokenizer: Any,
    root: Path,
    *,
    tokenizer_sha256: str,
    chat_template_sha256: str,
    seed: int,
    training_sequence_length: int,
    purpose: str,
    policy_sha256: str,
    source_policy_sha256: str,
    task5_paired_sha256: str,
    task6_completion_sha256: str,
    final_paired_response_sha256: str,
    resume_fingerprint: str,
    reconcile_resume: bool,
) -> _TokenizedCorpus:
    resumed = _load_tokenized_receipt(root, resume_fingerprint)
    if resumed is not None:
        expected_identity = (
            corpus.arm,
            corpus.corpus_sha256,
            tokenizer_sha256,
            chat_template_sha256,
            seed,
            training_sequence_length,
            purpose,
            policy_sha256,
            source_policy_sha256,
            task5_paired_sha256,
            task6_completion_sha256,
            final_paired_response_sha256,
        )
        actual_identity = (
            resumed.arm,
            resumed.corpus_sha256,
            resumed.tokenizer_sha256,
            resumed.chat_template_sha256,
            resumed.seed,
            resumed.training_sequence_length,
            resumed.purpose,
            resumed.policy_sha256,
            resumed.source_policy_sha256,
            resumed.task5_paired_selection_bucket_sha256,
            resumed.task6_completion_sha256,
            resumed.final_paired_response_sha256,
        )
        if actual_identity != expected_identity:
            raise ExposureViewError("tokenized staging identity reconciliation failed")
        if reconcile_resume:
            _reconcile_tokenized_corpus(
                resumed,
                corpus,
                tokenizer,
                training_sequence_length=training_sequence_length,
            )
        return resumed
    database_path = root / "tokenized.sqlite3"
    connection = sqlite3.connect(database_path)
    total_tokens = 0
    record_count = 0
    paired_digest: str | None = None
    selection_digest: str | None = None
    try:
        connection.execute("PRAGMA synchronous=FULL")
        connection.execute(
            """
            CREATE TABLE records(
                source_ordinal INTEGER PRIMARY KEY,
                prompt_uuid TEXT NOT NULL UNIQUE,
                domain TEXT NOT NULL,
                lane TEXT NOT NULL,
                context_bucket TEXT NOT NULL,
                input_ids_json TEXT NOT NULL,
                loss_mask_json TEXT NOT NULL,
                assistant_tokens INTEGER NOT NULL,
                promoted_record_sha256 TEXT NOT NULL
            )
            """
        )
        for ordinal, record in enumerate(corpus.records):
            record_sha256 = _validated_record_sha256(record)
            if (
                record.arm != corpus.arm
                or record.generation_identity_sha256 != corpus.generation_identity_sha256
            ):
                raise ExposureViewError(f"record provenance mismatch: {record.prompt_uuid}")
            _require_digest("selection digest", record.selection_sha256)
            if selection_digest is None:
                selection_digest = record.selection_sha256
            elif selection_digest != record.selection_sha256:
                raise ExposureViewError("record selection digest is not uniform")
            _require_digest("paired C/D digest", record.paired_cd_sha256)
            if corpus.arm in {"C", "D"}:
                if record.paired_cd_sha256 == "0" * 64:
                    raise ExposureViewError("C/D record has no paired selection/bucket proof")
                if paired_digest is None:
                    paired_digest = record.paired_cd_sha256
                elif paired_digest != record.paired_cd_sha256:
                    raise ExposureViewError("C/D paired selection/bucket proof is not uniform")
            input_ids, loss_mask = _tokenize_record(
                tokenizer,
                record,
                training_sequence_length=training_sequence_length,
            )
            assistant_tokens = sum(loss_mask)
            connection.execute(
                "INSERT INTO records VALUES(?,?,?,?,?,?,?,?,?)",
                (
                    ordinal,
                    record.prompt_uuid,
                    record.domain,
                    record.lane,
                    record.context_bucket,
                    canonical_json(input_ids).decode("utf-8"),
                    canonical_json(loss_mask).decode("utf-8"),
                    assistant_tokens,
                    record_sha256,
                ),
            )
            total_tokens += assistant_tokens
            record_count += 1
        if record_count == 0:
            raise ExposureViewError("response corpus contains no promoted records")
        connection.commit()
    except Exception:
        connection.close()
        raise
    finally:
        if connection:
            connection.close()
    database_bytes = database_path.stat().st_size
    database_sha256 = _sha256_file(database_path)
    tokenized = _TokenizedCorpus(
        schema_version=1,
        arm=corpus.arm,
        corpus_sha256=corpus.corpus_sha256,
        tokenizer_sha256=tokenizer_sha256,
        chat_template_sha256=chat_template_sha256,
        selection_sha256=selection_digest or "0" * 64,
        training_sequence_length=training_sequence_length,
        purpose=purpose,
        policy_sha256=policy_sha256,
        source_policy_sha256=source_policy_sha256,
        seed=seed,
        record_count=record_count,
        total_assistant_tokens=total_tokens,
        paired_selection_bucket_sha256=paired_digest or "0" * 64,
        task5_paired_selection_bucket_sha256=task5_paired_sha256,
        task6_completion_sha256=task6_completion_sha256,
        final_paired_response_sha256=final_paired_response_sha256,
        database_path=str(database_path),
        database_bytes=database_bytes,
        database_sha256=database_sha256,
        resume_fingerprint=resume_fingerprint,
    )
    receipt = asdict(tokenized)
    receipt_sha256 = sha256(canonical_json(receipt)).hexdigest()
    _write_exclusive(
        root / "TOKENIZED.json",
        canonical_json(receipt | {"receipt_sha256": receipt_sha256}) + b"\n",
    )
    _fsync_directory(root)
    return tokenized


def _reconcile_tokenized_corpus(
    tokenized: _TokenizedCorpus,
    corpus: Any,
    tokenizer: Any,
    *,
    training_sequence_length: int,
) -> None:
    total_tokens = 0
    record_count = 0
    selection_digest: str | None = None
    paired_digest: str | None = None
    records = iter(corpus.records)
    connection = sqlite3.connect(f"file:{tokenized.database_path}?mode=ro", uri=True)
    try:
        query = """
            SELECT source_ordinal,prompt_uuid,domain,lane,context_bucket,input_ids_json,
                   loss_mask_json,assistant_tokens,promoted_record_sha256
            FROM records ORDER BY source_ordinal
        """
        for database_row in connection.execute(query):
            try:
                record = next(records)
            except StopIteration as error:
                raise ExposureViewError(
                    "tokenized corpus reconciliation found an extra row"
                ) from error
            record_sha256 = _validated_record_sha256(record)
            input_ids, loss_mask = _tokenize_record(
                tokenizer,
                record,
                training_sequence_length=training_sequence_length,
            )
            if (
                int(database_row[0]) != record_count
                or database_row[1] != record.prompt_uuid
                or database_row[2] != record.domain
                or database_row[3] != record.lane
                or database_row[4] != record.context_bucket
                or json.loads(database_row[5]) != input_ids
                or json.loads(database_row[6]) != loss_mask
                or int(database_row[7]) != sum(loss_mask)
                or database_row[8] != record_sha256
            ):
                raise ExposureViewError("tokenized corpus reconciliation failed")
            selection_digest = selection_digest or record.selection_sha256
            if selection_digest != record.selection_sha256:
                raise ExposureViewError("tokenized corpus selection reconciliation failed")
            if corpus.arm in {"C", "D"}:
                paired_digest = paired_digest or record.paired_cd_sha256
                if paired_digest != record.paired_cd_sha256:
                    raise ExposureViewError("tokenized corpus pair reconciliation failed")
            total_tokens += sum(loss_mask)
            record_count += 1
        try:
            next(records)
        except StopIteration:
            pass
        else:
            raise ExposureViewError("tokenized corpus reconciliation found a missing row")
    finally:
        connection.close()
    expected = (
        record_count,
        total_tokens,
        selection_digest or "0" * 64,
        paired_digest or "0" * 64,
    )
    actual = (
        tokenized.record_count,
        tokenized.total_assistant_tokens,
        tokenized.selection_sha256,
        tokenized.paired_selection_bucket_sha256,
    )
    if actual != expected:
        raise ExposureViewError("tokenized corpus receipt reconciliation failed")


def _trim_mask(loss_mask: list[int], keep: int) -> list[int]:
    remaining = keep
    trimmed: list[int] = []
    for value in loss_mask:
        retained = int(value == 1 and remaining > 0)
        trimmed.append(retained)
        remaining -= retained
    if remaining != 0:
        raise ExposureViewError("final assistant mask cannot satisfy the exact boundary")
    return trimmed


def _epoch_rank(
    corpus_sha256: str,
    epoch: int,
    domain: str,
    lane: str,
    prompt_uuid: str,
    seed: int,
) -> str:
    identity = "\0".join((corpus_sha256, str(epoch), domain, lane, prompt_uuid, str(seed)))
    return sha256(identity.encode()).hexdigest()


def _digest_item(digest: Any, value: Any) -> None:
    digest.update(canonical_json(value))
    digest.update(b"\n")


def _reconcile_view(view: ExposureView, tokenized: _TokenizedCorpus) -> None:
    if (
        view.schema_version,
        view.arm,
        view.response_corpus_sha256,
        view.tokenizer_sha256,
        view.chat_template_sha256,
        view.selection_sha256,
        view.training_sequence_length,
        view.purpose,
        view.policy_sha256,
        view.source_policy_sha256,
        view.seed,
        view.paired_selection_bucket_sha256,
        view.task5_paired_selection_bucket_sha256,
        view.task6_completion_sha256,
        view.final_paired_response_sha256,
        view.resume_fingerprint,
    ) != (
        1,
        tokenized.arm,
        tokenized.corpus_sha256,
        tokenized.tokenizer_sha256,
        tokenized.chat_template_sha256,
        tokenized.selection_sha256,
        tokenized.training_sequence_length,
        tokenized.purpose,
        tokenized.policy_sha256,
        tokenized.source_policy_sha256,
        tokenized.seed,
        tokenized.paired_selection_bucket_sha256,
        tokenized.task5_paired_selection_bucket_sha256,
        tokenized.task6_completion_sha256,
        tokenized.final_paired_response_sha256,
        tokenized.resume_fingerprint,
    ):
        raise ExposureViewError("exposure receipt identity reconciliation failed")
    records_digest = sha256()
    row_order_digest = sha256()
    input_ids_digest = sha256()
    loss_masks_digest = sha256()
    cumulative_digest = sha256()
    cumulative = 0
    row_count = 0
    previous_epoch = -1
    rows_in_epoch = 0
    previous_rank = ""
    partial_seen = False
    required = {
        "view_ordinal",
        "epoch",
        "source_ordinal",
        "prompt_uuid",
        "domain",
        "lane",
        "context_bucket",
        "input_ids",
        "loss_mask",
        "assistant_tokens",
        "cumulative_assistant_tokens",
        "promoted_record_sha256",
    }
    connection = sqlite3.connect(f"file:{tokenized.database_path}?mode=ro", uri=True)
    try:
        with Path(view.records_path).open("rb") as source:
            for raw_line in source:
                records_digest.update(raw_line)
                try:
                    row = json.loads(raw_line)
                except json.JSONDecodeError as error:
                    raise ExposureViewError(
                        "exposure row reconciliation found invalid JSON"
                    ) from error
                if set(row) != required or canonical_json(row) + b"\n" != raw_line:
                    raise ExposureViewError("exposure row reconciliation found a schema mismatch")
                if partial_seen:
                    raise ExposureViewError("exposure row reconciliation found data after a trim")
                epoch = row["epoch"]
                source_ordinal = row["source_ordinal"]
                if (
                    not isinstance(epoch, int)
                    or isinstance(epoch, bool)
                    or not isinstance(source_ordinal, int)
                    or isinstance(source_ordinal, bool)
                    or row["view_ordinal"] != row_count
                ):
                    raise ExposureViewError("exposure row reconciliation found invalid ordering")
                if epoch != previous_epoch:
                    if previous_epoch >= 0 and rows_in_epoch != tokenized.record_count:
                        raise ExposureViewError("exposure row reconciliation skipped an epoch row")
                    if epoch != previous_epoch + 1:
                        raise ExposureViewError("exposure row reconciliation skipped an epoch")
                    previous_epoch = epoch
                    rows_in_epoch = 0
                    previous_rank = ""
                expected = connection.execute(
                    "SELECT prompt_uuid,domain,lane,context_bucket,input_ids_json,loss_mask_json,"
                    "promoted_record_sha256 FROM records WHERE source_ordinal=?",
                    (source_ordinal,),
                ).fetchone()
                if expected is None:
                    raise ExposureViewError("exposure row reconciliation found an unknown source")
                input_ids = json.loads(expected[4])
                full_mask = json.loads(expected[5])
                loss_mask = row["loss_mask"]
                rank = _epoch_rank(
                    tokenized.corpus_sha256,
                    epoch,
                    str(expected[1]),
                    str(expected[2]),
                    str(expected[0]),
                    tokenized.seed,
                )
                if rank <= previous_rank:
                    raise ExposureViewError(
                        "exposure row reconciliation found a shuffled-order error"
                    )
                previous_rank = rank
                retained = sum(loss_mask) if isinstance(loss_mask, list) else -1
                if (
                    row["prompt_uuid"] != expected[0]
                    or row["domain"] != expected[1]
                    or row["lane"] != expected[2]
                    or row["context_bucket"] != expected[3]
                    or row["input_ids"] != input_ids
                    or row["promoted_record_sha256"] != expected[6]
                    or not isinstance(loss_mask, list)
                    or len(loss_mask) != len(input_ids)
                    or any(value not in (0, 1) for value in loss_mask)
                    or row["assistant_tokens"] != retained
                ):
                    raise ExposureViewError("exposure row reconciliation found a content mismatch")
                if loss_mask != full_mask:
                    if retained < 1 or loss_mask != _trim_mask(full_mask, retained):
                        raise ExposureViewError("exposure row reconciliation found an invalid trim")
                    partial_seen = True
                cumulative += retained
                if row["cumulative_assistant_tokens"] != cumulative:
                    raise ExposureViewError(
                        "exposure row reconciliation found a cumulative mismatch"
                    )
                _digest_item(
                    row_order_digest, [row_count, epoch, row["prompt_uuid"], source_ordinal]
                )
                _digest_item(input_ids_digest, [row_count, input_ids])
                _digest_item(loss_masks_digest, [row_count, loss_mask])
                _digest_item(cumulative_digest, [row_count, cumulative])
                row_count += 1
                rows_in_epoch += 1
    finally:
        connection.close()
    actual = (
        records_digest.hexdigest(),
        row_order_digest.hexdigest(),
        input_ids_digest.hexdigest(),
        loss_masks_digest.hexdigest(),
        cumulative_digest.hexdigest(),
        cumulative,
        row_count,
        min(row_count, tokenized.record_count),
        row_count - min(row_count, tokenized.record_count),
        cumulative // tokenized.total_assistant_tokens,
    )
    declared = (
        view.records_sha256,
        view.row_order_sha256,
        view.input_ids_sha256,
        view.loss_masks_sha256,
        view.cumulative_tokens_sha256,
        view.assistant_tokens,
        view.row_count,
        view.unique_prompt_count,
        view.repeated_prompt_count,
        view.completed_epochs,
    )
    if actual != declared:
        raise ExposureViewError("exposure receipt reconciliation failed")


def _load_view(root: Path, name: str, tokenized: _TokenizedCorpus) -> ExposureView | None:
    receipt_path = root / f"{name}.json"
    records_path = root / f"{name}.jsonl"
    if not receipt_path.exists():
        if records_path.exists():
            raise ExposureViewError(f"incomplete {name!r} exposure staging requires recovery")
        return None
    try:
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        receipt_sha256 = receipt.pop("receipt_sha256")
    except (OSError, json.JSONDecodeError, KeyError) as error:
        raise ExposureViewError(f"{name!r} exposure receipt is malformed") from error
    if (
        receipt_sha256 != sha256(canonical_json(receipt)).hexdigest()
        or receipt.get("schema_version") != 1
        or receipt.get("name") != name
        or receipt.get("resume_fingerprint") != tokenized.resume_fingerprint
        or receipt.get("receipt_path") != str(receipt_path)
        or receipt.get("records_path") != str(records_path)
        or not records_path.is_file()
        or receipt.get("records_bytes") != records_path.stat().st_size
        or receipt.get("records_sha256") != _sha256_file(records_path)
    ):
        raise ExposureViewError(f"{name!r} exposure staging authentication failed")
    try:
        view = ExposureView(**receipt, receipt_sha256=receipt_sha256)
    except TypeError as error:
        raise ExposureViewError(f"{name!r} exposure receipt has an invalid schema") from error
    _reconcile_view(view, tokenized)
    return view


def _build_view(
    tokenized: _TokenizedCorpus,
    root: Path,
    *,
    name: str,
    target_tokens: int,
) -> ExposureView:
    if _SAFE_NAME.fullmatch(name) is None:
        raise ExposureViewError(f"unsafe exposure view name: {name!r}")
    if not isinstance(target_tokens, int) or isinstance(target_tokens, bool) or target_tokens < 1:
        raise ExposureViewError("assistant-token boundary must be a positive integer")
    resumed = _load_view(root, name, tokenized)
    if resumed is not None:
        if resumed.assistant_tokens != target_tokens:
            raise ExposureViewError(f"{name!r} exposure boundary mismatch")
        return resumed

    records_path = root / f"{name}.jsonl"
    receipt_path = root / f"{name}.json"
    records_digest = sha256()
    row_order_digest = sha256()
    input_ids_digest = sha256()
    loss_masks_digest = sha256()
    cumulative_digest = sha256()
    cumulative = 0
    row_count = 0
    epoch = 0
    connection = sqlite3.connect(f"file:{tokenized.database_path}?mode=ro", uri=True)
    connection.execute("PRAGMA temp_store=FILE")
    try:
        connection.create_function(
            "epoch_rank",
            3,
            lambda domain, lane, prompt_uuid: _epoch_rank(
                tokenized.corpus_sha256,
                epoch,
                str(domain),
                str(lane),
                str(prompt_uuid),
                tokenized.seed,
            ),
            deterministic=True,
        )
        with records_path.open("xb") as output:
            while cumulative < target_tokens:
                rows_this_epoch = 0
                query = """
                    SELECT source_ordinal,prompt_uuid,domain,lane,context_bucket,
                           input_ids_json,loss_mask_json,assistant_tokens,promoted_record_sha256
                    FROM records
                    ORDER BY epoch_rank(domain,lane,prompt_uuid),prompt_uuid
                """
                for row in connection.execute(query):
                    remaining = target_tokens - cumulative
                    input_ids = json.loads(row[5])
                    full_mask = json.loads(row[6])
                    retained = min(int(row[7]), remaining)
                    loss_mask = (
                        full_mask if retained == int(row[7]) else _trim_mask(full_mask, retained)
                    )
                    cumulative += retained
                    record = {
                        "view_ordinal": row_count,
                        "epoch": epoch,
                        "source_ordinal": int(row[0]),
                        "prompt_uuid": row[1],
                        "domain": row[2],
                        "lane": row[3],
                        "context_bucket": row[4],
                        "input_ids": input_ids,
                        "loss_mask": loss_mask,
                        "assistant_tokens": retained,
                        "cumulative_assistant_tokens": cumulative,
                        "promoted_record_sha256": row[8],
                    }
                    encoded = canonical_json(record) + b"\n"
                    output.write(encoded)
                    records_digest.update(encoded)
                    _digest_item(row_order_digest, [row_count, epoch, row[1], int(row[0])])
                    _digest_item(input_ids_digest, [row_count, input_ids])
                    _digest_item(loss_masks_digest, [row_count, loss_mask])
                    _digest_item(cumulative_digest, [row_count, cumulative])
                    row_count += 1
                    rows_this_epoch += 1
                    if cumulative == target_tokens:
                        break
                if rows_this_epoch == 0:
                    raise ExposureViewError("tokenized corpus cannot advance exposure")
                if cumulative < target_tokens:
                    epoch += 1
            output.flush()
            os.fsync(output.fileno())
    finally:
        connection.close()

    unique_prompt_count = min(row_count, tokenized.record_count)
    view_without_receipt: dict[str, Any] = {
        "schema_version": 1,
        "name": name,
        "arm": tokenized.arm,
        "response_corpus_sha256": tokenized.corpus_sha256,
        "tokenizer_sha256": tokenized.tokenizer_sha256,
        "chat_template_sha256": tokenized.chat_template_sha256,
        "selection_sha256": tokenized.selection_sha256,
        "training_sequence_length": tokenized.training_sequence_length,
        "purpose": tokenized.purpose,
        "policy_sha256": tokenized.policy_sha256,
        "source_policy_sha256": tokenized.source_policy_sha256,
        "seed": tokenized.seed,
        "assistant_tokens": cumulative,
        "full_prompt_assistant_tokens": tokenized.total_assistant_tokens,
        "row_count": row_count,
        "unique_prompt_count": unique_prompt_count,
        "repeated_prompt_count": row_count - unique_prompt_count,
        "completed_epochs": target_tokens // tokenized.total_assistant_tokens,
        "records_path": str(records_path),
        "records_bytes": records_path.stat().st_size,
        "records_sha256": records_digest.hexdigest(),
        "row_order_sha256": row_order_digest.hexdigest(),
        "input_ids_sha256": input_ids_digest.hexdigest(),
        "loss_masks_sha256": loss_masks_digest.hexdigest(),
        "cumulative_tokens_sha256": cumulative_digest.hexdigest(),
        "paired_selection_bucket_sha256": tokenized.paired_selection_bucket_sha256,
        "task5_paired_selection_bucket_sha256": tokenized.task5_paired_selection_bucket_sha256,
        "task6_completion_sha256": tokenized.task6_completion_sha256,
        "final_paired_response_sha256": tokenized.final_paired_response_sha256,
        "resume_fingerprint": tokenized.resume_fingerprint,
        "receipt_path": str(receipt_path),
    }
    receipt_sha256 = sha256(canonical_json(view_without_receipt)).hexdigest()
    _write_exclusive(
        receipt_path,
        canonical_json(view_without_receipt | {"receipt_sha256": receipt_sha256}) + b"\n",
    )
    _fsync_directory(root)
    return ExposureView(**view_without_receipt, receipt_sha256=receipt_sha256)


def _canonical_document(path: Path, description: str) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise ExposureViewError(f"{description} is not a regular file")
    raw = path.read_bytes()
    try:
        value = json.loads(raw)
    except json.JSONDecodeError as error:
        raise ExposureViewError(f"{description} is not JSON") from error
    if not isinstance(value, dict) or canonical_json(value) + b"\n" != raw:
        raise ExposureViewError(f"{description} is not canonical JSON")
    return value


def _declared_file(root: Path, descriptor: Any, description: str) -> Path:
    if not isinstance(descriptor, dict) or not isinstance(descriptor.get("path"), str):
        raise ExposureViewError(f"{description} descriptor is malformed")
    root_resolved = root.resolve(strict=True)
    unresolved = root / descriptor["path"]
    if unresolved.is_symlink():
        raise ExposureViewError(f"{description} authentication failed")
    path = unresolved.resolve(strict=True)
    expected_size = descriptor.get("bytes", descriptor.get("byte_count"))
    if (
        not path.is_relative_to(root_resolved)
        or not path.is_file()
        or (expected_size is not None and path.stat().st_size != expected_size)
        or _sha256_file(path) != descriptor.get("sha256")
    ):
        raise ExposureViewError(f"{description} authentication failed")
    return path


def _authenticate_selection_manifest(
    manifest_path: Path,
    *,
    expected_manifest_sha256: str,
) -> tuple[str, str, str]:
    if _sha256_file(manifest_path) != expected_manifest_sha256:
        raise ExposureViewError("Task 5 selection manifest identity mismatch")
    manifest = _canonical_document(manifest_path, "Task 5 selection manifest")
    root_record = {key: value for key, value in manifest.items() if key != "root_sha256"}
    if (
        manifest.get("schema_version") != 2
        or manifest.get("root_sha256") != sha256(canonical_json(root_record)).hexdigest()
    ):
        raise ExposureViewError("Task 5 selection root authentication failed")
    root = manifest_path.parent
    shard_rows = 0
    for descriptor in manifest.get("shards", []):
        _declared_file(root, descriptor, "Task 5 selection shard")
        if not isinstance(descriptor.get("row_count"), int):
            raise ExposureViewError("Task 5 selection shard row count is malformed")
        shard_rows += descriptor["row_count"]
    index_path = _declared_file(root, manifest.get("index"), "Task 5 selection index")
    arms = manifest.get("arms")
    identity = manifest.get("identity")
    if not isinstance(arms, dict) or not isinstance(identity, dict):
        raise ExposureViewError("Task 5 selection identity is incomplete")
    connection = sqlite3.connect(f"file:{index_path}?mode=ro", uri=True)
    try:
        row_count = int(connection.execute("SELECT count(*) FROM rows").fetchone()[0])
        if row_count != manifest.get("row_count") or row_count != shard_rows:
            raise ExposureViewError("Task 5 selection row count reconciliation failed")
        for left, right in (("C", "D"), ("D", "C")):
            mismatch = int(
                connection.execute(
                    "SELECT count(*) FROM ("
                    "SELECT prompt_uuid,status FROM rows WHERE arm=? AND domain!='swe-agentic-tool' "
                    "EXCEPT SELECT prompt_uuid,status FROM rows "
                    "WHERE arm=? AND domain!='swe-agentic-tool')",
                    (left, right),
                ).fetchone()[0]
            )
            if mismatch:
                raise ExposureViewError("Task 5 C/D paired UUID proof mismatch")
        try:
            paired_seed = {
                "bucket_floors": arms["C"]["non_agentic_bucket_floors"],
                "C_count_proof_sha256": arms["C"]["count_proof_sha256"],
                "D_count_proof_sha256": arms["D"]["count_proof_sha256"],
            }
        except (KeyError, TypeError) as error:
            raise ExposureViewError("Task 5 paired bucket proof is incomplete") from error
        paired = sha256(canonical_json(paired_seed))
        for row in connection.execute(
            "SELECT status,prompt_uuid FROM rows "
            "WHERE arm='C' AND domain!='swe-agentic-tool' ORDER BY status,prompt_uuid"
        ):
            _digest_item(paired, list(row))
        paired_sha256 = paired.hexdigest()
        if paired_sha256 != manifest.get("paired_cd_sha256"):
            raise ExposureViewError("Task 5 paired selection/bucket proof mismatch")
        metadata = {
            "schema_version": 2,
            "policy_sha256": identity.get("policy_sha256"),
            "seed": identity.get("seed"),
            "source_inventory_sha256": identity.get("source_inventory_sha256"),
            "baseline_receipt_sha256": identity.get("baseline_receipt_sha256"),
            "held_out_receipt_sha256": identity.get("held_out_receipt_sha256"),
            "ptv2_revision": identity.get("ptv2_revision"),
            "ptv2_allowlist_sha256": identity.get("ptv2_allowlist_sha256"),
            "paired_cd_sha256": paired_sha256,
            "arms": arms,
        }
        selection = sha256(canonical_json(metadata))
        for row in connection.execute(
            "SELECT arm,status,selection_index,prompt_uuid FROM rows "
            "ORDER BY arm,status,selection_index"
        ):
            _digest_item(selection, list(row))
        selection_sha256 = selection.hexdigest()
        if selection_sha256 != manifest.get("selection_sha256"):
            raise ExposureViewError("Task 5 selection stream proof mismatch")
    finally:
        connection.close()
    source_policy_sha256 = str(identity.get("policy_sha256"))
    _require_digest("Task 5 source policy", source_policy_sha256)
    return selection_sha256, paired_sha256, source_policy_sha256


def _authenticate_generation_identity(corpus: Any) -> dict[str, Any]:
    try:
        payload = asdict(corpus.generation_identity)
    except TypeError as error:
        raise ExposureViewError("generation identity must be a frozen Task 6 record") from error
    required = {
        "target_revision",
        "tokenizer_sha256",
        "chat_template_sha256",
        "runtime_sha256",
        "container_sha256",
        "source_selection_sha256",
        "thinking_mode",
        "temperature",
        "max_tokens",
        "max_total_length",
    }
    if set(payload) != required:
        raise ExposureViewError("generation identity schema mismatch")
    actual = sha256(canonical_json(payload)).hexdigest()
    if actual != corpus.generation_identity_sha256:
        raise ExposureViewError("generation identity digest mismatch")
    if payload["source_selection_sha256"] != corpus.source_selection_sha256:
        raise ExposureViewError("generation identity source selection mismatch")
    return payload


def _authenticate_task6_completion(root: Path, *, expected_sha256: str) -> dict[str, Any]:
    _require_digest("expected Task 6 completion", expected_sha256)
    completion_path = root.parent / "completion.json"
    if (
        completion_path.is_symlink()
        or not completion_path.is_file()
        or _sha256_file(completion_path) != expected_sha256
    ):
        raise ExposureViewError("Task 6 completion identity mismatch")
    try:
        completion = json.loads(completion_path.read_bytes())
    except json.JSONDecodeError as error:
        raise ExposureViewError("Task 6 completion is not JSON") from error
    if not isinstance(completion, dict):
        raise ExposureViewError("Task 6 completion is malformed")
    return completion


def _authenticate_response_corpus(
    corpus: Any,
    root: Path,
    *,
    expected_completion_sha256: str,
) -> _AuthenticatedResponse:
    completion = _authenticate_task6_completion(root, expected_sha256=expected_completion_sha256)
    receipt = _canonical_document(root / "PROMOTION.json", "Task 6 promotion receipt")
    promotion_sha256 = _sha256_file(root / "PROMOTION.json")
    identity = _authenticate_generation_identity(corpus)
    completion_expected = {
        "generation_identity_sha256": corpus.generation_identity_sha256,
        "target_revision": identity["target_revision"],
        "tokenizer_sha256": identity["tokenizer_sha256"],
        "chat_template_sha256": identity["chat_template_sha256"],
        "runtime_sha256": identity["runtime_sha256"],
        "container_sha256": identity["container_sha256"],
        "source_selection_sha256": identity["source_selection_sha256"],
        "thinking_mode": identity["thinking_mode"],
        "promotion_status": "passed",
        "corpus_sha256": corpus.corpus_sha256,
        "promotion_receipt_sha256": promotion_sha256,
    }
    if any(completion.get(key) != value for key, value in completion_expected.items()):
        raise ExposureViewError("Task 6 completion reconciliation failed")
    completion_files = completion.get("files")
    if not isinstance(completion_files, list) or completion.get("file_count") != len(
        completion_files
    ):
        raise ExposureViewError("Task 6 completion file manifest is malformed")
    for descriptor in completion_files:
        _declared_file(root.parent, descriptor, "Task 6 published file")
    for name, expected in (
        ("corpus_sha256", corpus.corpus_sha256),
        ("generation_identity_sha256", corpus.generation_identity_sha256),
        ("source_selection_sha256", corpus.source_selection_sha256),
    ):
        if receipt.get(name) != expected:
            raise ExposureViewError(f"Task 6 {name} mismatch")
    descriptors = receipt.get("files")
    if not isinstance(descriptors, list):
        raise ExposureViewError("Task 6 promotion file manifest is missing")
    files = {
        descriptor.get("path"): descriptor
        for descriptor in descriptors
        if isinstance(descriptor, dict)
    }
    response_path = _declared_file(root, files.get("responses.jsonl"), "Task 6 response stream")
    index_path = _declared_file(root, files.get("response-index.sqlite3"), "Task 6 response index")
    for descriptor in descriptors:
        _declared_file(root, descriptor, "Task 6 promoted file")

    promoted_digest = sha256()
    cell_counts: Counter[str] = Counter()
    histograms: dict[str, Counter[int]] = {}
    records = iter(corpus.records)
    promoted_count = 0
    connection = sqlite3.connect(f"file:{index_path}?mode=ro", uri=True)
    try:
        with response_path.open("rb") as response_stream:
            query = "SELECT ordinal,prompt_uuid,cell,assistant_tokens,payload FROM promoted ORDER BY ordinal"
            for ordinal, prompt_uuid, cell, assistant_tokens, payload in connection.execute(query):
                try:
                    record = next(records)
                except StopIteration as error:
                    raise ExposureViewError("Task 6 corpus has an extra indexed record") from error
                line = response_stream.readline()
                expected_payload = canonical_json(asdict(record)).decode("utf-8")
                if (
                    ordinal != promoted_count
                    or prompt_uuid != record.prompt_uuid
                    or payload != expected_payload
                    or line != payload.encode() + b"\n"
                ):
                    raise ExposureViewError("Task 6 promoted record reconciliation failed")
                promoted_digest.update(line)
                cell_counts[str(cell)] += 1
                histograms.setdefault(str(cell), Counter())[int(assistant_tokens)] += 1
                promoted_count += 1
            if response_stream.read(1):
                raise ExposureViewError("Task 6 response stream has unindexed data")
        try:
            next(records)
        except StopIteration:
            pass
        else:
            raise ExposureViewError("Task 6 corpus is missing an indexed record")
        attempt_history = sha256()
        for (payload,) in connection.execute("SELECT payload FROM attempts ORDER BY ordinal"):
            attempt_history.update(payload.encode() + b"\n")
        unused_count = int(connection.execute("SELECT count(*) FROM unused_reserve").fetchone()[0])
    finally:
        connection.close()
    normalized_histograms = {
        cell: dict(sorted(values.items())) for cell, values in sorted(histograms.items())
    }
    if (
        promoted_count != receipt.get("promoted_count")
        or dict(cell_counts) != receipt.get("cell_counts")
        or promoted_digest.hexdigest() != receipt.get("promoted_record_stream_sha256")
        or attempt_history.hexdigest() != receipt.get("attempt_history_stream_sha256")
        or unused_count != receipt.get("unused_reserve_count")
        or canonical_json(normalized_histograms)
        != canonical_json(receipt.get("response_token_histograms"))
    ):
        raise ExposureViewError("Task 6 promotion receipt reconciliation failed")
    corpus_record = {
        "arm": corpus.arm,
        "generation_identity_sha256": corpus.generation_identity_sha256,
        "selection_sha256": receipt.get("selection_sha256"),
        "paired_cd_sha256": receipt.get("paired_cd_sha256"),
        "cell_counts": receipt.get("cell_counts"),
        "promoted_record_stream_sha256": promoted_digest.hexdigest(),
        "attempt_history_stream_sha256": attempt_history.hexdigest(),
        "unused_reserve_count": unused_count,
        "response_token_histograms": normalized_histograms,
    }
    if sha256(canonical_json(corpus_record)).hexdigest() != corpus.corpus_sha256:
        raise ExposureViewError("Task 6 response corpus root mismatch")
    selection_sha256 = str(receipt.get("selection_sha256"))
    paired_sha256 = str(receipt.get("paired_cd_sha256"))
    _require_digest("Task 6 selection", selection_sha256)
    _require_digest("Task 6 paired C/D", paired_sha256)
    return _AuthenticatedResponse(
        selection_sha256=selection_sha256,
        paired_sha256=paired_sha256,
        index_path=index_path,
        completion_sha256=expected_completion_sha256,
    )


def _authenticate_final_paired_rows(c_index: Path, d_index: Path) -> str:
    def rows(connection: sqlite3.Connection) -> Iterator[tuple[str, str]]:
        for (payload,) in connection.execute("SELECT payload FROM promoted ORDER BY prompt_uuid"):
            record = json.loads(payload)
            if record.get("domain") != "swe-agentic-tool":
                yield str(record.get("prompt_uuid")), str(record.get("context_bucket"))

    c_connection = sqlite3.connect(f"file:{c_index}?mode=ro", uri=True)
    d_connection = sqlite3.connect(f"file:{d_index}?mode=ro", uri=True)
    try:
        c_rows = iter(rows(c_connection))
        d_rows = iter(rows(d_connection))
        digest = sha256(
            canonical_json({"schema_version": 1, "scope": "final-promoted-non-agentic-cd"})
        )
        while True:
            c_row = next(c_rows, None)
            d_row = next(d_rows, None)
            if c_row is None and d_row is None:
                break
            if c_row is None or d_row is None or c_row != d_row:
                raise ExposureViewError("final C/D promoted rows differ by UUID or context bucket")
            _digest_item(digest, list(c_row))
        return digest.hexdigest()
    finally:
        c_connection.close()
        d_connection.close()


def _authenticate_tokenizer_root(
    root: Path,
    *,
    tokenizer_sha256: str,
    chat_template_sha256: str,
) -> Any:
    names = {
        "added_tokens.json",
        "chat_template.jinja",
        "merges.txt",
        "special_tokens_map.json",
        "tokenizer.json",
        "tokenizer.model",
        "tokenizer_config.json",
        "vocab.json",
        "vocab.txt",
    }
    files = sorted(path for path in root.iterdir() if path.is_file() and path.name in names)
    if not files or any(path.is_symlink() for path in files):
        raise ExposureViewError("trainer tokenizer snapshot is missing or contains symlinks")
    digest = sha256()
    for path in files:
        digest.update(path.name.encode())
        digest.update(b"\0")
        digest.update(bytes.fromhex(_sha256_file(path)))
    if digest.hexdigest() != tokenizer_sha256:
        raise ExposureViewError("trainer tokenizer snapshot digest mismatch")
    template_path = root / "chat_template.jinja"
    if not template_path.is_file() or _sha256_file(template_path) != chat_template_sha256:
        raise ExposureViewError("trainer chat-template file digest mismatch")
    # Production must use the exact local snapshot just authenticated above.
    from transformers import AutoTokenizer  # pyright: ignore[reportMissingImports]

    try:
        tokenizer = AutoTokenizer.from_pretrained(
            root,
            local_files_only=True,
            trust_remote_code=False,
        )
    except (OSError, ValueError) as error:
        raise ExposureViewError(
            "authenticated trainer tokenizer snapshot cannot be loaded"
        ) from error
    template = getattr(tokenizer, "chat_template", None)
    if (
        not isinstance(template, str)
        or sha256(template.encode()).hexdigest() != chat_template_sha256
    ):
        raise ExposureViewError("loaded trainer chat template digest mismatch")
    return tokenizer


def _build_exposure_views(
    corpus: Any,
    tokenizer: Any,
    *,
    output_root: Path,
    boundaries: Mapping[str, int | None],
    seed: int,
    tokenizer_sha256: str,
    chat_template_sha256: str,
    training_sequence_length: int = 4_096,
    production: bool = False,
    policy_sha256: str | None = None,
    response_artifact_root: Path | None = None,
    expected_task6_completion_sha256: str | None = None,
    selection_manifest_path: Path | None = None,
    tokenizer_root: Path | None = None,
    _production_proof: _ProductionProof | None = None,
    _allow_partial_production_suite: bool = False,
    _allow_common_production: bool = False,
    _reconcile_resume: bool = True,
) -> Mapping[str, ExposureView]:
    """Retokenize a response corpus and stage exact consumable exposure views."""
    _require_digest("response corpus", corpus.corpus_sha256)
    _require_digest("source selection", corpus.source_selection_sha256)
    _require_digest("generation identity", corpus.generation_identity_sha256)
    _require_digest("tokenizer", tokenizer_sha256)
    _require_digest("chat template", chat_template_sha256)
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ExposureViewError("seed must be an integer")
    if (
        isinstance(training_sequence_length, bool)
        or not isinstance(training_sequence_length, int)
        or training_sequence_length < 1
    ):
        raise ExposureViewError("training sequence length must be a positive integer")
    if not boundaries:
        raise ExposureViewError("at least one exposure boundary is required")
    purpose = "production-comparison" if production else "content-view"
    if production:
        if training_sequence_length != 4_096:
            raise ExposureViewError("production training sequence length must be exactly 4096")
        if boundaries.get("one-pass") is not None:
            raise ExposureViewError("production one-pass view must use the full corpus")
        if "common" in boundaries and not _allow_common_production:
            raise ExposureViewError("common exposure is available only from the paired builder")
        required_names = {str(value) for value in _PRODUCTION_BOUNDARIES} | {"one-pass"}
        valid_suites = (
            (required_names, required_names | {"common"})
            if _allow_common_production
            else (required_names,)
        )
        if not _allow_partial_production_suite and set(boundaries) not in valid_suites:
            raise ExposureViewError(
                "production boundaries must include exact 256M, 1B, and one-pass views; "
                "64M is runtime-screen only"
            )
        if _allow_partial_production_suite and set(boundaries) - (required_names | {"common"}):
            raise ExposureViewError("production boundaries cannot include runtime screens")
        for value in _PRODUCTION_BOUNDARIES:
            name = str(value)
            if name in boundaries and boundaries[name] != value:
                raise ExposureViewError("production views require exact boundary values")
        if policy_sha256 is not None and policy_sha256 != _EXPOSURE_POLICY_SHA256:
            raise ExposureViewError("production exposure policy digest mismatch")
    resolved_policy_sha256 = policy_sha256 or (_EXPOSURE_POLICY_SHA256 if production else "0" * 64)
    resolved_source_policy_sha256 = "0" * 64
    task5_paired_sha256 = "0" * 64
    task6_completion_sha256 = "0" * 64
    final_paired_response_sha256 = "0" * 64
    proof = _production_proof
    if production:
        _authenticate_generation_identity(corpus)
        if proof is None:
            if (
                response_artifact_root is None
                or expected_task6_completion_sha256 is None
                or selection_manifest_path is None
                or tokenizer_root is None
            ):
                raise ExposureViewError(
                    "production views require Task 5, Task 6, and tokenizer artifact roots"
                )
            selection_sha256, paired_sha256, source_policy = _authenticate_selection_manifest(
                selection_manifest_path,
                expected_manifest_sha256=corpus.source_selection_sha256,
            )
            response = _authenticate_response_corpus(
                corpus,
                response_artifact_root,
                expected_completion_sha256=expected_task6_completion_sha256,
            )
            if (response.selection_sha256, response.paired_sha256) != (
                selection_sha256,
                paired_sha256,
            ):
                raise ExposureViewError("Task 5/Task 6 selection or paired proof mismatch")
            tokenizer = _authenticate_tokenizer_root(
                tokenizer_root,
                tokenizer_sha256=tokenizer_sha256,
                chat_template_sha256=chat_template_sha256,
            )
            record_paired_sha256 = paired_sha256 if corpus.arm in {"C", "D"} else "0" * 64
            proof = _ProductionProof(
                selection_sha256,
                paired_sha256,
                record_paired_sha256,
                source_policy,
                response.completion_sha256,
                "0" * 64,
            )
        resolved_source_policy_sha256 = proof.source_policy_sha256
        task5_paired_sha256 = proof.task5_paired_sha256
        task6_completion_sha256 = proof.task6_completion_sha256
        final_paired_response_sha256 = proof.final_paired_response_sha256
    _require_digest("policy", resolved_policy_sha256)
    _require_digest("source policy", resolved_source_policy_sha256)
    identity = corpus.generation_identity
    if identity.tokenizer_sha256 != tokenizer_sha256:
        raise ExposureViewError("trainer tokenizer does not match the response identity")
    if identity.chat_template_sha256 != chat_template_sha256:
        raise ExposureViewError("trainer chat template does not match the response identity")
    template = getattr(tokenizer, "chat_template", None)
    if (
        not isinstance(template, str)
        or sha256(template.encode()).hexdigest() != chat_template_sha256
    ):
        raise ExposureViewError("loaded trainer chat template digest mismatch")

    resume_fingerprint = _prepare_root(
        output_root,
        _identity_payload(
            corpus,
            tokenizer_sha256=tokenizer_sha256,
            chat_template_sha256=chat_template_sha256,
            seed=seed,
            training_sequence_length=training_sequence_length,
            purpose=purpose,
            policy_sha256=resolved_policy_sha256,
            source_policy_sha256=resolved_source_policy_sha256,
            task5_paired_sha256=task5_paired_sha256,
            task6_completion_sha256=task6_completion_sha256,
            final_paired_response_sha256=final_paired_response_sha256,
        ),
    )
    tokenized = _stage_tokenized_corpus(
        corpus,
        tokenizer,
        output_root,
        tokenizer_sha256=tokenizer_sha256,
        chat_template_sha256=chat_template_sha256,
        seed=seed,
        training_sequence_length=training_sequence_length,
        purpose=purpose,
        policy_sha256=resolved_policy_sha256,
        source_policy_sha256=resolved_source_policy_sha256,
        task5_paired_sha256=task5_paired_sha256,
        task6_completion_sha256=task6_completion_sha256,
        final_paired_response_sha256=final_paired_response_sha256,
        resume_fingerprint=resume_fingerprint,
        reconcile_resume=_reconcile_resume,
    )
    if production:
        if proof is None:
            raise ExposureViewError("production proof construction failed")
        if (
            tokenized.selection_sha256 != proof.selection_sha256
            or tokenized.paired_selection_bucket_sha256 != proof.record_paired_sha256
            or tokenized.task5_paired_selection_bucket_sha256 != proof.task5_paired_sha256
            or tokenized.task6_completion_sha256 != proof.task6_completion_sha256
            or tokenized.final_paired_response_sha256 != proof.final_paired_response_sha256
        ):
            raise ExposureViewError(
                "authenticated production proof does not match tokenized records"
            )
    views: dict[str, ExposureView] = {}
    for name, requested_tokens in boundaries.items():
        target_tokens = (
            tokenized.total_assistant_tokens if requested_tokens is None else requested_tokens
        )
        views[name] = _build_view(tokenized, output_root, name=name, target_tokens=target_tokens)
    return MappingProxyType(views)


def build_exposure_views(
    corpus: Any,
    tokenizer: Any,
    *,
    output_root: Path,
    boundaries: Mapping[str, int | None],
    seed: int,
    tokenizer_sha256: str,
    chat_template_sha256: str,
    training_sequence_length: int = 4_096,
    production: bool = False,
    response_artifact_root: Path | None = None,
    expected_task6_completion_sha256: str | None = None,
    selection_manifest_path: Path | None = None,
    tokenizer_root: Path | None = None,
) -> Mapping[str, ExposureView]:
    """Retokenize a response corpus into a complete content or production view suite."""
    return _build_exposure_views(
        corpus,
        tokenizer,
        output_root=output_root,
        boundaries=boundaries,
        seed=seed,
        tokenizer_sha256=tokenizer_sha256,
        chat_template_sha256=chat_template_sha256,
        training_sequence_length=training_sequence_length,
        production=production,
        response_artifact_root=response_artifact_root,
        expected_task6_completion_sha256=expected_task6_completion_sha256,
        selection_manifest_path=selection_manifest_path,
        tokenizer_root=tokenizer_root,
    )


def _write_paired_receipt(
    root: Path,
    *,
    views_c: Mapping[str, ExposureView],
    views_d: Mapping[str, ExposureView],
    common_boundary: int,
    paired_digest: str,
    source_policy_sha256: str,
    exposure_policy_sha256: str,
    final_paired_response_sha256: str,
    c_task6_completion_sha256: str,
    d_task6_completion_sha256: str,
) -> tuple[str, str]:
    receipt_path = root / "PAIRED_EXPOSURE.json"
    payload = {
        "schema_version": 1,
        "common_boundary": common_boundary,
        "C_full_prompt_assistant_tokens": views_c["one-pass"].full_prompt_assistant_tokens,
        "D_full_prompt_assistant_tokens": views_d["one-pass"].full_prompt_assistant_tokens,
        "paired_selection_bucket_sha256": paired_digest,
        "source_policy_sha256": source_policy_sha256,
        "exposure_policy_sha256": exposure_policy_sha256,
        "final_paired_response_sha256": final_paired_response_sha256,
        "C_task6_completion_sha256": c_task6_completion_sha256,
        "D_task6_completion_sha256": d_task6_completion_sha256,
        "C_view_receipts": {name: view.receipt_sha256 for name, view in views_c.items()},
        "D_view_receipts": {name: view.receipt_sha256 for name, view in views_d.items()},
    }
    receipt_sha256 = sha256(canonical_json(payload)).hexdigest()
    expected = canonical_json(payload | {"receipt_sha256": receipt_sha256}) + b"\n"
    if receipt_path.exists():
        if receipt_path.read_bytes() != expected:
            raise ExposureViewError("paired exposure receipt conflicts with staged views")
    else:
        _write_exclusive(receipt_path, expected)
        _fsync_directory(root)
    return str(receipt_path), receipt_sha256


def build_paired_exposure_views(
    corpus_c: Any,
    corpus_d: Any,
    tokenizer: Any,
    *,
    output_root: Path,
    fixed_boundaries: tuple[int, ...] = (256_000_000, 1_000_000_000),
    seed: int,
    tokenizer_sha256: str,
    chat_template_sha256: str,
    training_sequence_length: int = 4_096,
    production: bool = True,
    response_artifact_roots: Mapping[str, Path] | None = None,
    expected_task6_completion_sha256s: Mapping[str, str] | None = None,
    selection_manifest_path: Path | None = None,
    tokenizer_root: Path | None = None,
) -> PairedExposureViews:
    """Stage C/D fixed, largest-common, and separate one-pass exposure views."""
    if corpus_c.arm != "C" or corpus_d.arm != "D":
        raise ExposureViewError("paired exposure views require C and D response corpora")
    if (
        not fixed_boundaries
        or len(set(fixed_boundaries)) != len(fixed_boundaries)
        or any(
            not isinstance(value, int) or isinstance(value, bool) or value < 1
            for value in fixed_boundaries
        )
    ):
        raise ExposureViewError("fixed exposure boundaries must be nonempty and unique")
    if production and fixed_boundaries != _PRODUCTION_BOUNDARIES:
        raise ExposureViewError(
            "production boundaries must be exactly 256M and 1B; 64M is runtime-screen only"
        )
    policy_sha256 = _EXPOSURE_POLICY_SHA256
    source_policy_sha256 = "0" * 64
    production_proof: _ProductionProof | None = None
    if production:
        if (
            response_artifact_roots is None
            or set(response_artifact_roots) != {"C", "D"}
            or expected_task6_completion_sha256s is None
            or set(expected_task6_completion_sha256s) != {"C", "D"}
            or selection_manifest_path is None
            or tokenizer_root is None
        ):
            raise ExposureViewError(
                "production views require Task 5, Task 6, and tokenizer artifact roots"
            )
        if corpus_c.source_selection_sha256 != corpus_d.source_selection_sha256:
            raise ExposureViewError("C/D source selection manifest identities differ")
        selection_sha256, paired_sha256, source_policy_sha256 = _authenticate_selection_manifest(
            selection_manifest_path,
            expected_manifest_sha256=corpus_c.source_selection_sha256,
        )
        c_response = _authenticate_response_corpus(
            corpus_c,
            response_artifact_roots["C"],
            expected_completion_sha256=expected_task6_completion_sha256s["C"],
        )
        d_response = _authenticate_response_corpus(
            corpus_d,
            response_artifact_roots["D"],
            expected_completion_sha256=expected_task6_completion_sha256s["D"],
        )
        if (c_response.selection_sha256, d_response.selection_sha256) != (
            selection_sha256,
            selection_sha256,
        ) or (
            c_response.paired_sha256,
            d_response.paired_sha256,
        ) != (paired_sha256, paired_sha256):
            raise ExposureViewError("Task 5/Task 6 selection or paired proof mismatch")
        final_paired_response_sha256 = _authenticate_final_paired_rows(
            c_response.index_path,
            d_response.index_path,
        )
        tokenizer = _authenticate_tokenizer_root(
            tokenizer_root,
            tokenizer_sha256=tokenizer_sha256,
            chat_template_sha256=chat_template_sha256,
        )
        c_production_proof = _ProductionProof(
            selection_sha256,
            paired_sha256,
            paired_sha256,
            source_policy_sha256,
            c_response.completion_sha256,
            final_paired_response_sha256,
        )
        d_production_proof = _ProductionProof(
            selection_sha256,
            paired_sha256,
            paired_sha256,
            source_policy_sha256,
            d_response.completion_sha256,
            final_paired_response_sha256,
        )
    else:
        c_production_proof = production_proof
        d_production_proof = production_proof
        final_paired_response_sha256 = "0" * 64
    output_root.mkdir(parents=True, exist_ok=True)
    one_pass = {"one-pass": None}
    initial_c = _build_exposure_views(
        corpus_c,
        tokenizer,
        output_root=output_root / "C",
        boundaries=one_pass,
        seed=seed,
        tokenizer_sha256=tokenizer_sha256,
        chat_template_sha256=chat_template_sha256,
        training_sequence_length=training_sequence_length,
        production=production,
        policy_sha256=policy_sha256,
        _production_proof=c_production_proof,
        _allow_partial_production_suite=True,
    )
    initial_d = _build_exposure_views(
        corpus_d,
        tokenizer,
        output_root=output_root / "D",
        boundaries=one_pass,
        seed=seed,
        tokenizer_sha256=tokenizer_sha256,
        chat_template_sha256=chat_template_sha256,
        training_sequence_length=training_sequence_length,
        production=production,
        policy_sha256=policy_sha256,
        _production_proof=d_production_proof,
        _allow_partial_production_suite=True,
    )
    paired_digest = initial_c["one-pass"].paired_selection_bucket_sha256
    if (
        paired_digest == "0" * 64
        or initial_d["one-pass"].paired_selection_bucket_sha256 != paired_digest
        or initial_c["one-pass"].selection_sha256 != initial_d["one-pass"].selection_sha256
    ):
        raise ExposureViewError("C/D paired selection/bucket proof mismatch")
    common_boundary = min(
        initial_c["one-pass"].full_prompt_assistant_tokens,
        initial_d["one-pass"].full_prompt_assistant_tokens,
    )
    boundaries: dict[str, int | None] = {str(value): value for value in fixed_boundaries}
    boundaries.update({"common": common_boundary, "one-pass": None})
    views_c = _build_exposure_views(
        corpus_c,
        tokenizer,
        output_root=output_root / "C",
        boundaries=boundaries,
        seed=seed,
        tokenizer_sha256=tokenizer_sha256,
        chat_template_sha256=chat_template_sha256,
        training_sequence_length=training_sequence_length,
        production=production,
        policy_sha256=policy_sha256,
        _production_proof=c_production_proof,
        _reconcile_resume=False,
        _allow_common_production=True,
    )
    views_d = _build_exposure_views(
        corpus_d,
        tokenizer,
        output_root=output_root / "D",
        boundaries=boundaries,
        seed=seed,
        tokenizer_sha256=tokenizer_sha256,
        chat_template_sha256=chat_template_sha256,
        training_sequence_length=training_sequence_length,
        production=production,
        policy_sha256=policy_sha256,
        _production_proof=d_production_proof,
        _reconcile_resume=False,
        _allow_common_production=True,
    )
    receipt_path, receipt_sha256 = _write_paired_receipt(
        output_root,
        views_c=views_c,
        views_d=views_d,
        common_boundary=common_boundary,
        paired_digest=paired_digest,
        source_policy_sha256=source_policy_sha256,
        exposure_policy_sha256=policy_sha256,
        final_paired_response_sha256=final_paired_response_sha256,
        c_task6_completion_sha256=views_c["one-pass"].task6_completion_sha256,
        d_task6_completion_sha256=views_d["one-pass"].task6_completion_sha256,
    )
    return PairedExposureViews(
        C=views_c,
        D=views_d,
        common_boundary=common_boundary,
        C_full_prompt_assistant_tokens=views_c["one-pass"].full_prompt_assistant_tokens,
        D_full_prompt_assistant_tokens=views_d["one-pass"].full_prompt_assistant_tokens,
        paired_selection_bucket_sha256=paired_digest,
        final_paired_response_sha256=final_paired_response_sha256,
        C_task6_completion_sha256=views_c["one-pass"].task6_completion_sha256,
        D_task6_completion_sha256=views_d["one-pass"].task6_completion_sha256,
        source_policy_sha256=source_policy_sha256,
        exposure_policy_sha256=policy_sha256,
        receipt_path=receipt_path,
        receipt_sha256=receipt_sha256,
    )
