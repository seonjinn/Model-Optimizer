# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Materialize the exact scaled B-balanced source-native canary view."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sqlite3
import stat
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from common.specdec.qwen4b_b_atomic import atomic_publish_bytes, atomic_publish_directory

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator, Mapping

__all__ = [
    "CANARY_CELL_QUOTAS",
    "CANARY_LANGUAGE_QUOTAS",
    "build_canary_occurrences",
    "load_task8_publication",
    "materialize_canary_from_shards",
    "resolve_worker_count",
    "write_canary_occurrences",
]

_SHA256 = re.compile(r"^[0-9a-f]{64}$")

CANARY_CELL_QUOTAS = {
    "math": 25_600,
    "code": 20_480,
    "stem": 25_600,
    "chat": 20_480,
    "multilingual": 10_240,
}
CANARY_LANGUAGE_QUOTAS = dict.fromkeys(("de", "ja", "es", "fr", "it"), 2048)

_ARROW_BATCH_SIZE = 256
_MAX_DECLARED_SHARDS = 201
_THREAD_ENVIRONMENT = (
    "ARROW_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
)


@dataclass(frozen=True)
class _ShardTask:
    index: int
    path: Path
    spool_path: Path
    expected_rows: int
    expected_bytes: int
    expected_sha256: str
    seed: int


@dataclass(frozen=True)
class Task8Shard:
    """One manifest-declared Task8 shard."""

    path: Path
    relative_path: str
    rows: int
    bytes: int
    sha256: str


@dataclass(frozen=True)
class Task8Publication:
    """Authenticated Task8 publication facts consumed by the B builder."""

    root: Path
    publication_sha256: str
    corpus_manifest_sha256: str
    selection_receipt_sha256: str
    shard_inventory_sha256: str
    source_commit: str
    shards: tuple[Task8Shard, ...]


@dataclass(frozen=True)
class BCanaryBuildArtifact:
    """Paths and identities of one immutable builder bundle."""

    root: Path
    output_path: Path
    receipt_path: Path
    output_sha256: str
    receipt_sha256: str


def load_task8_publication(
    root: Path,
    *,
    expected_publication_sha256: str,
    expected_selection_receipt_sha256: str,
    source_commit: str,
    expected_occurrence_count: int = 2_000_000,
    expected_cell_counts: Mapping[str, int] | None = None,
    expected_language_counts: Mapping[str, int] | None = None,
    expected_shard_count: int = _MAX_DECLARED_SHARDS,
) -> Task8Publication:
    """Authenticate Task8's publication, schema-v3 selection, and shard inventory."""
    for label, digest in (
        ("publication", expected_publication_sha256),
        ("selection receipt", expected_selection_receipt_sha256),
    ):
        if _SHA256.fullmatch(digest) is None:
            raise ValueError(f"Task8 {label} SHA-256 must be exact")
    if re.fullmatch(r"[0-9a-f]{40}", source_commit) is None:
        raise ValueError("Task8 source commit must be exact")
    publication_path = root / "PUBLICATION.json"
    publication_raw = _read_nofollow_stable(publication_path)
    if hashlib.sha256(publication_raw).hexdigest() != expected_publication_sha256:
        raise ValueError("Task8 publication receipt SHA-256 mismatch")
    publication = _canonical_document(publication_raw, "Task8 publication receipt")
    if publication.get("schema") != "modelopt-specdec-publication-receipt-v1":
        raise ValueError("Task8 publication receipt schema is invalid")
    if publication.get("artifact_source_commit") != source_commit:
        raise ValueError("Task8 publication source commit mismatch")
    if publication.get("selection_manifest_sha256") != expected_selection_receipt_sha256:
        raise ValueError("Task8 publication selection identity mismatch")
    manifest_path = root / "CORPUS_MANIFEST.json"
    manifest_raw = _read_nofollow_stable(manifest_path)
    if hashlib.sha256(manifest_raw).hexdigest() != publication.get("corpus_manifest_sha256"):
        raise ValueError("Task8 corpus manifest SHA-256 mismatch")
    manifest = _canonical_document(manifest_raw, "Task8 corpus manifest")
    if (
        manifest.get("schema") != "modelopt-specdec-corpus-manifest-v1"
        or manifest.get("artifact_source_commit") != source_commit
        or manifest.get("selection_manifest_sha256") != expected_selection_receipt_sha256
    ):
        raise ValueError("Task8 corpus manifest lineage mismatch")
    selection_path = root / "inputs/selection/receipt.json"
    selection_raw = _read_nofollow_stable(selection_path)
    if hashlib.sha256(selection_raw).hexdigest() != expected_selection_receipt_sha256:
        raise ValueError("Task9 schema-v3 selection receipt SHA-256 mismatch")
    selection = _json_object(selection_raw, "Task9 schema-v3 selection receipt")
    identity = selection.get("selection_identity")
    cell_counts = (
        {
            "math": 500_000,
            "code": 400_000,
            "stem": 500_000,
            "chat": 400_000,
            "multilingual": 200_000,
        }
        if expected_cell_counts is None
        else dict(expected_cell_counts)
    )
    language_counts = (
        dict.fromkeys(("de", "ja", "es", "fr", "it"), 40_000)
        if expected_language_counts is None
        else dict(expected_language_counts)
    )
    if (
        selection.get("schema_version") != 3
        or selection.get("strategy") != "B-balanced"
        or not isinstance(identity, dict)
        or identity.get("strategy") != "B-balanced"
        or identity.get("occurrence_count") != expected_occurrence_count
        or identity.get("cell_occurrence_counts") != cell_counts
        or identity.get("multilingual_occurrence_counts") != language_counts
    ):
        raise ValueError("Task9 schema-v3 B selection semantics are invalid")
    shards_raw = manifest.get("shards")
    if not isinstance(shards_raw, list) or len(shards_raw) != expected_shard_count:
        raise ValueError(f"Task8 publication must declare exactly {expected_shard_count} shards")
    file_records = manifest.get("files")
    if not isinstance(file_records, list):
        raise ValueError("Task8 corpus manifest files must be a list")
    declared_files = {
        record.get("path"): (record.get("bytes"), record.get("sha256"))
        for record in file_records
        if isinstance(record, dict)
    }
    shards: list[Task8Shard] = []
    for descriptor in shards_raw:
        if not isinstance(descriptor, dict):
            raise ValueError("Task8 shard descriptor must be an object")
        relative = _safe_relative(descriptor.get("path"), "Task8 shard")
        size = descriptor.get("bytes")
        rows = descriptor.get("row_count")
        digest = descriptor.get("sha256")
        if (
            not relative.startswith("shards/")
            or Path(relative).suffix not in {".jsonl", ".parquet"}
            or isinstance(size, bool)
            or not isinstance(size, int)
            or size < 0
            or isinstance(rows, bool)
            or not isinstance(rows, int)
            or rows < 0
            or not isinstance(digest, str)
            or _SHA256.fullmatch(digest) is None
            or declared_files.get(relative) != (size, digest)
        ):
            raise ValueError("Task8 shard descriptor is invalid")
        shards.append(Task8Shard(root / relative, relative, rows, size, digest))
    if len({shard.path for shard in shards}) != len(shards):
        raise ValueError("Task8 shard descriptors are not unique")
    actual = set((root / "shards").glob("*"))
    if actual != {shard.path for shard in shards}:
        raise ValueError("Task8 publication has missing or orphan shards")
    return Task8Publication(
        root,
        expected_publication_sha256,
        hashlib.sha256(manifest_raw).hexdigest(),
        expected_selection_receipt_sha256,
        _sha256_json(
            {
                "shards": [
                    {
                        "path": shard.relative_path,
                        "rows": shard.rows,
                        "bytes": shard.bytes,
                        "sha256": shard.sha256,
                    }
                    for shard in shards
                ]
            }
        ),
        source_commit,
        tuple(shards),
    )


@dataclass(frozen=True)
class _ShardResult:
    index: int
    path: str
    spool_path: str
    row_count: int
    bytes: int
    sha256: str
    elapsed_seconds: float
    worker_pid: int


def resolve_worker_count(
    *,
    requested_workers: int | None,
    declared_shard_count: int,
    environ: Mapping[str, str] | None = None,
) -> tuple[int, int]:
    """Resolve the worker count without oversubscribing CPUs or declared shards."""
    if not 1 <= declared_shard_count <= _MAX_DECLARED_SHARDS:
        raise ValueError("B builder requires between 1 and 201 declared shards")
    environment = os.environ if environ is None else environ
    raw_cpus = environment.get("SLURM_CPUS_PER_TASK")
    try:
        allocated_cpus = int(raw_cpus) if raw_cpus is not None else (os.cpu_count() or 1)
    except ValueError as error:
        raise ValueError("SLURM_CPUS_PER_TASK must be a positive integer") from error
    if allocated_cpus < 1:
        raise ValueError("B builder requires a positive CPU allocation")
    requested = allocated_cpus if requested_workers is None else requested_workers
    if requested < 1:
        raise ValueError("B builder requires at least one worker")
    return min(
        requested, allocated_cpus, declared_shard_count, _MAX_DECLARED_SHARDS
    ), allocated_cpus


def build_canary_occurrences(
    rows: Iterable[Mapping[str, object]], *, seed: int
) -> list[dict[str, object]]:
    """Deterministically select the exact B-scaled cell and language quota view."""
    candidates: dict[str, list[dict[str, object]]] = {}
    for ordinal, raw in enumerate(rows):
        row, category = _prepare_candidate(raw, seed=seed, ordinal=ordinal)
        candidates.setdefault(category, []).append(row)

    selected: list[dict[str, object]] = []
    for category, quota in _category_quotas().items():
        selected.extend(_take_exact(candidates.get(category, []), quota, category))
    selected.sort(key=lambda row: str(row["_rank"]))
    for row in selected:
        row.pop("_rank")
    expected_occurrences = sum(_category_quotas().values())
    if len(selected) != expected_occurrences:
        raise AssertionError(f"B canary selection must contain {expected_occurrences} occurrences")
    return selected


def materialize_canary_from_shards(
    shard_paths: Iterable[Path],
    *,
    output_root: Path,
    job_id: str,
    seed: int,
    requested_workers: int | None = None,
    environ: Mapping[str, str] | None = None,
    scratch_root: Path,
    source_projection_sha256: str,
    source_task8_publication_sha256: str,
    source_corpus_manifest_sha256: str,
    source_selection_receipt_sha256: str,
    source_shard_inventory_sha256: str,
    source_commit: str,
    expected_shards: tuple[Task8Shard, ...],
) -> BCanaryBuildArtifact:
    """Build a byte-stable canary through bounded shard-parallel spooling."""
    shards = tuple(Path(path) for path in shard_paths)
    _validate_build_paths(shards, output_root, scratch_root)
    if (
        any(
            _SHA256.fullmatch(value) is None
            for value in (
                source_projection_sha256,
                source_task8_publication_sha256,
                source_corpus_manifest_sha256,
                source_selection_receipt_sha256,
                source_shard_inventory_sha256,
            )
        )
        or re.fullmatch(r"[0-9a-f]{40}", source_commit) is None
    ):
        raise ValueError("B builder source identities must be exact")
    effective_workers, allocated_cpus = resolve_worker_count(
        requested_workers=requested_workers,
        declared_shard_count=len(shards),
        environ=environ,
    )
    started_wall_ns = time.time_ns()
    started_monotonic_ns = time.monotonic_ns()
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", job_id):
        raise ValueError("B builder job_id must be a safe non-empty identifier")
    output_root.parent.mkdir(parents=True, exist_ok=True)
    partial = output_root.with_name(f".{output_root.name}.partial-{job_id}")
    if os.path.lexists(partial):
        raise FileExistsError(f"B builder partial requires recovery: {partial}")
    partial.mkdir(mode=0o700)
    output_path = partial / "canary.jsonl"
    receipt_path = partial / "BUILD_RECEIPT.json"
    scratch_root.mkdir(mode=0o700, parents=True)
    try:
        if tuple(item.path for item in expected_shards) != shards:
            raise ValueError("Task8 shard descriptors do not match the declared order")
        tasks = tuple(
            _ShardTask(
                index=index,
                path=descriptor.path,
                spool_path=scratch_root / f"shard-{index:03d}.jsonl",
                expected_rows=descriptor.rows,
                expected_bytes=descriptor.bytes,
                expected_sha256=descriptor.sha256,
                seed=seed,
            )
            for index, descriptor in enumerate(expected_shards)
        )
        results = _run_shard_workers(tasks, effective_workers)
        occurrence_count, output_sha256, output_bytes = _merge_spools(
            results,
            database_path=scratch_root / "candidates.sqlite3",
            output_path=output_path,
        )
        finished_wall_ns = time.time_ns()
        payload: dict[str, Any] = {
            "schema_version": 1,
            "seed": seed,
            "source_commit": source_commit,
            "source_projection_sha256": source_projection_sha256,
            "source_task8_publication_sha256": source_task8_publication_sha256,
            "source_corpus_manifest_sha256": source_corpus_manifest_sha256,
            "source_selection_receipt_sha256": source_selection_receipt_sha256,
            "source_shard_inventory_sha256": source_shard_inventory_sha256,
            "declared_shard_count": len(shards),
            "source_row_count": sum(result.row_count for result in results),
            "occurrence_count": occurrence_count,
            "output_path": "canary.jsonl",
            "output_bytes": output_bytes,
            "output_sha256": output_sha256,
            "execution": {
                "allocated_cpus": allocated_cpus,
                "requested_workers": requested_workers,
                "effective_workers": effective_workers,
                "omp_threads_per_worker": 1,
                "arrow_threads_per_worker": 1,
                "arrow_batch_size": _ARROW_BATCH_SIZE,
                "max_concurrent_open_files": effective_workers * 2 + 3,
            },
            "timing": {
                "started_at": _utc_timestamp(started_wall_ns),
                "finished_at": _utc_timestamp(finished_wall_ns),
                "elapsed_seconds": round(
                    (time.monotonic_ns() - started_monotonic_ns) / 1_000_000_000,
                    6,
                ),
            },
            "shard_timings": [_shard_provenance(result) for result in results],
        }
        payload["receipt_sha256"] = _sha256_json(payload)
        _write_canonical_json(receipt_path, payload)
        if _sha256_file(output_path) != output_sha256:
            raise ValueError("B builder output changed before publication")
        receipt_sha256 = _sha256_file(receipt_path)
        atomic_publish_directory(partial, output_root)
        return BCanaryBuildArtifact(
            output_root,
            output_root / "canary.jsonl",
            output_root / "BUILD_RECEIPT.json",
            output_sha256,
            receipt_sha256,
        )
    except BaseException:
        raise


def write_canary_occurrences(path: Path, rows: Iterable[Mapping[str, object]]) -> None:
    """Atomically write canonical JSONL B canary occurrences."""
    if path.exists():
        raise FileExistsError(f"B canary output already exists: {path}")
    payload = b"".join((_canonical_json(dict(row)) + "\n").encode() for row in rows)
    atomic_publish_bytes(path, payload, job_id=str(os.getpid()))


def _run_shard_workers(tasks: tuple[_ShardTask, ...], workers: int) -> tuple[_ShardResult, ...]:
    results: dict[int, _ShardResult] = {}
    with ProcessPoolExecutor(max_workers=workers, initializer=_initialize_worker) as executor:
        futures = {executor.submit(_process_shard, task): task.index for task in tasks}
        try:
            for future in as_completed(futures):
                result = future.result()
                results[result.index] = result
        except BaseException:
            for future in futures:
                future.cancel()
            raise
    if len(results) != len(tasks):
        raise RuntimeError("B builder did not receive every declared shard result")
    return tuple(results[index] for index in range(len(tasks)))


def _initialize_worker() -> None:
    for name in _THREAD_ENVIRONMENT:
        os.environ[name] = "1"


def _process_shard(task: _ShardTask) -> _ShardResult:
    started = time.monotonic_ns()
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(task.path, flags)
    row_count = 0
    digest = hashlib.sha256()
    byte_count = 0
    with os.fdopen(descriptor, "rb") as source:
        opened = os.fstat(source.fileno())
        namespace_before = os.lstat(task.path)
        if not stat.S_ISREG(opened.st_mode) or (opened.st_dev, opened.st_ino) != (
            namespace_before.st_dev,
            namespace_before.st_ino,
        ):
            raise ValueError(f"Task8 shard is not an inode-stable regular file: {task.path}")
        spool_flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
        spool_descriptor = os.open(task.spool_path, spool_flags, 0o600)
        with os.fdopen(spool_descriptor, "w", encoding="utf-8") as spool:
            rows, byte_count, digest_value = _iter_authenticated_rows(source, task.path)
            digest = digest_value
            for row_count, raw in enumerate(rows, start=1):
                row, category = _prepare_candidate(
                    raw,
                    seed=task.seed,
                    ordinal=(task.index << 48) + row_count - 1,
                )
                rank = str(row.pop("_rank"))
                record = {
                    "category": category,
                    "rank": rank,
                    "row": row,
                    "row_index": row_count - 1,
                    "shard_index": task.index,
                }
                spool.write(_canonical_json(record) + "\n")
            spool.flush()
            os.fsync(spool.fileno())
            closed = os.fstat(source.fileno())
            namespace_after = os.lstat(task.path)
            if _file_snapshot(opened) != _file_snapshot(closed) or (
                namespace_after.st_dev,
                namespace_after.st_ino,
            ) != (opened.st_dev, opened.st_ino):
                raise ValueError(f"Task8 shard mutated while it was read: {task.path}")
    if task.expected_rows >= 0 and row_count != task.expected_rows:
        raise ValueError(
            f"B builder shard row count changed for {task.path}: "
            f"expected {task.expected_rows}, found {row_count}"
        )
    if byte_count != task.expected_bytes or digest.hexdigest() != task.expected_sha256:
        raise ValueError(f"Task8 shard content identity mismatch: {task.path}")
    return _ShardResult(
        index=task.index,
        path=str(task.path),
        spool_path=str(task.spool_path),
        row_count=row_count,
        bytes=byte_count,
        sha256=digest.hexdigest(),
        elapsed_seconds=round((time.monotonic_ns() - started) / 1_000_000_000, 6),
        worker_pid=os.getpid(),
    )


def _merge_spools(
    results: tuple[_ShardResult, ...],
    *,
    database_path: Path,
    output_path: Path,
) -> tuple[int, str, int]:
    connection = sqlite3.connect(database_path)
    try:
        connection.executescript(
            """
            PRAGMA journal_mode=OFF;
            PRAGMA synchronous=OFF;
            PRAGMA temp_store=FILE;
            CREATE TABLE candidates (
                category TEXT NOT NULL,
                rank TEXT NOT NULL,
                shard_index INTEGER NOT NULL,
                row_index INTEGER NOT NULL,
                payload TEXT NOT NULL,
                PRIMARY KEY (category, rank, shard_index, row_index)
            ) WITHOUT ROWID;
            CREATE TABLE selected (
                rank TEXT NOT NULL,
                shard_index INTEGER NOT NULL,
                row_index INTEGER NOT NULL,
                payload TEXT NOT NULL,
                PRIMARY KEY (rank, shard_index, row_index)
            ) WITHOUT ROWID;
            """
        )
        for result in results:
            with Path(result.spool_path).open(encoding="utf-8") as spool:
                records = (_decode_spool_record(line) for line in spool if line.strip())
                connection.executemany(
                    "INSERT INTO candidates(category,rank,shard_index,row_index,payload) "
                    "VALUES(?,?,?,?,?)",
                    records,
                )
                connection.commit()
        for category, quota in _category_quotas().items():
            available = connection.execute(
                "SELECT COUNT(*) FROM candidates WHERE category=?",
                (category,),
            ).fetchone()[0]
            if available < quota:
                raise ValueError(
                    f"B canary {category} has {available} candidates, requires {quota}"
                )
            connection.execute(
                "INSERT INTO selected(rank,shard_index,row_index,payload) "
                "SELECT rank,shard_index,row_index,payload FROM candidates "
                "WHERE category=? ORDER BY rank,shard_index,row_index LIMIT ?",
                (category, quota),
            )
        connection.commit()
        rows = connection.execute(
            "SELECT payload FROM selected ORDER BY rank,shard_index,row_index"
        )
        digest = hashlib.sha256()
        byte_count = 0
        occurrence_count = 0
        with output_path.open("xb") as destination:
            for (payload,) in rows:
                encoded = payload.encode() + b"\n"
                destination.write(encoded)
                digest.update(encoded)
                byte_count += len(encoded)
                occurrence_count += 1
            destination.flush()
            os.fsync(destination.fileno())
        expected_occurrences = sum(_category_quotas().values())
        if occurrence_count != expected_occurrences:
            raise AssertionError(
                f"B canary selection must contain {expected_occurrences} occurrences"
            )
        return occurrence_count, digest.hexdigest(), byte_count
    finally:
        connection.close()


def _decode_spool_record(line: str) -> tuple[str, str, int, int, str]:
    record = json.loads(line)
    return (
        record["category"],
        record["rank"],
        record["shard_index"],
        record["row_index"],
        _canonical_json(record["row"]),
    )


def _shard_provenance(result: _ShardResult) -> dict[str, object]:
    payload = asdict(result)
    payload.pop("spool_path")
    return payload


def _iter_authenticated_rows(
    source: Any, path: Path
) -> tuple[Iterator[Mapping[str, object]], int, hashlib._Hash]:
    """Hash then parse one already-open no-follow descriptor without reopening it."""
    digest = hashlib.sha256()
    byte_count = 0
    for block in iter(lambda: source.read(1024 * 1024), b""):
        digest.update(block)
        byte_count += len(block)
    source.seek(0)
    if path.suffix == ".jsonl":

        def rows() -> Iterator[Mapping[str, object]]:
            for raw_line in source:
                if raw_line.strip():
                    row = json.loads(raw_line)
                    if not isinstance(row, dict):
                        raise ValueError(f"Task8 shard row must be an object: {path}")
                    yield row

        return rows(), byte_count, digest
    if path.suffix == ".parquet":
        try:
            import pyarrow as pa  # pyright: ignore[reportMissingImports]
            import pyarrow.parquet as pq  # pyright: ignore[reportMissingImports]
        except ImportError as error:
            raise ValueError("pyarrow is required to build from Parquet shards") from error
        pa.set_cpu_count(1)
        pa.set_io_thread_count(1)
        parquet = pq.ParquetFile(source)

        def rows() -> Iterator[Mapping[str, object]]:
            for batch in parquet.iter_batches(batch_size=_ARROW_BATCH_SIZE, use_threads=False):
                yield from batch.to_pylist()

        return rows(), byte_count, digest
    raise ValueError(f"B builder does not support shard type: {path}")


def _prepare_candidate(
    raw: Mapping[str, object],
    *,
    seed: int,
    ordinal: int,
) -> tuple[dict[str, object], str]:
    row = dict(raw)
    cell = _required_task8_string(row, "domain").lower()
    record_json = _required_task8_string(row, "record_json")
    try:
        producer_row = json.loads(record_json)
    except json.JSONDecodeError as error:
        raise ValueError("Task8 record_json must be valid JSON") from error
    if not isinstance(producer_row, dict) or _canonical_json(producer_row) != record_json:
        raise ValueError("Task8 record_json must be a canonical object")
    if producer_row.get("prompt_uuid") != row.get("prompt_uuid"):
        raise ValueError("Task8 prompt identity does not match record_json")
    if producer_row.get("domain") != row.get("domain"):
        raise ValueError("Task8 domain does not match record_json")
    language_value = producer_row.get("language", "")
    if not isinstance(language_value, str):
        raise ValueError("Task8 record language must be a string")
    language = language_value.lower()
    if cell not in CANARY_CELL_QUOTAS:
        raise ValueError(f"unsupported B canary cell: {cell}")
    if cell == "multilingual":
        if language not in CANARY_LANGUAGE_QUOTAS:
            raise ValueError(f"unsupported B canary language: {language}")
        category = f"{cell}/{language}"
    else:
        if language:
            raise ValueError("non-multilingual B canary rows must have an empty language")
        category = cell
    input_ids = row.get("input_ids")
    producer_input_ids = producer_row.get("input_ids")
    loss_mask = row.get("loss_mask")
    producer_loss_mask = producer_row.get("loss_mask")
    if (
        not isinstance(input_ids, list)
        or not input_ids
        or any(isinstance(token, bool) or not isinstance(token, int) for token in input_ids)
        or input_ids != producer_input_ids
    ):
        raise ValueError("Task8 token IDs do not match record_json")
    if (
        not isinstance(loss_mask, list)
        or len(loss_mask) != len(input_ids)
        or any(not isinstance(masked, bool) for masked in loss_mask)
        or loss_mask != producer_loss_mask
        or row.get("assistant_tokens") != sum(loss_mask)
        or producer_row.get("assistant_tokens") != sum(loss_mask)
    ):
        raise ValueError("Task8 loss mask does not match record_json")
    messages = producer_row.get("messages")
    if (
        not isinstance(messages, list)
        or not messages
        or any(
            not isinstance(message, dict)
            or message.get("role") not in {"system", "user", "assistant", "tool"}
            or not isinstance(message.get("content"), str)
            for message in messages
        )
    ):
        raise ValueError("Task8 record_json lacks canonical messages")
    tools = producer_row.get("tools", [])
    if not isinstance(tools, list):
        raise ValueError("Task8 record_json tools must be a list")
    identifier = _required_task8_string(row, "prompt_uuid")
    streaming_row: dict[str, object] = {
        "conversation_id": identifier,
        "messages": messages,
        "tools": tools,
        "input_ids": input_ids,
        "loss_mask": [int(masked) for masked in loss_mask],
        "_rank": _rank(seed, ordinal, identifier),
    }
    return streaming_row, category


def _category_quotas() -> dict[str, int]:
    quotas = {cell: quota for cell, quota in CANARY_CELL_QUOTAS.items() if cell != "multilingual"}
    quotas.update(
        {f"multilingual/{language}": quota for language, quota in CANARY_LANGUAGE_QUOTAS.items()}
    )
    return quotas


def _take_exact(rows: list[dict[str, object]], quota: int, label: str) -> list[dict[str, object]]:
    if len(rows) < quota:
        raise ValueError(f"B canary {label} has {len(rows)} candidates, requires {quota}")
    return sorted(rows, key=lambda row: str(row["_rank"]))[:quota]


def _rank(seed: int, ordinal: int, identifier: str) -> str:
    return hashlib.sha256(f"{seed}:{ordinal}:{identifier}".encode()).hexdigest()


def _required_task8_string(row: Mapping[str, object], key: str) -> str:
    value = row.get(key)
    if not isinstance(value, str):
        raise ValueError(f"Task8 B canary row {key} must be a string")
    return value


def _file_snapshot(metadata: os.stat_result) -> tuple[int, int, int, int, int]:
    return (
        metadata.st_dev,
        metadata.st_ino,
        metadata.st_size,
        metadata.st_mtime_ns,
        metadata.st_ctime_ns,
    )


def _read_nofollow_stable(path: Path) -> bytes:
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags)
    try:
        before = os.fstat(descriptor)
        namespace = os.lstat(path)
        if not stat.S_ISREG(before.st_mode) or (before.st_dev, before.st_ino) != (
            namespace.st_dev,
            namespace.st_ino,
        ):
            raise ValueError(f"authenticated file is not a stable regular file: {path}")
        with os.fdopen(descriptor, "rb") as source:
            raw = source.read()
            after = os.fstat(source.fileno())
        namespace_after = os.lstat(path)
    except BaseException:
        raise
    if _file_snapshot(before) != _file_snapshot(after) or (
        namespace_after.st_dev,
        namespace_after.st_ino,
    ) != (before.st_dev, before.st_ino):
        raise ValueError(f"authenticated file changed while it was read: {path}")
    return raw


def _canonical_document(raw: bytes, label: str) -> dict[str, Any]:
    payload = _json_object(raw, label)
    canonical = _canonical_json(payload).encode()
    if raw not in {canonical, canonical + b"\n"}:
        raise ValueError(f"{label} is not canonical JSON")
    return payload


def _json_object(raw: bytes, label: str) -> dict[str, Any]:
    try:
        payload = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{label} is not JSON") from error
    if not isinstance(payload, dict):
        raise ValueError(f"{label} must be a JSON object")
    return payload


def _safe_relative(value: object, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{label} path must be a non-empty string")
    relative = Path(value)
    if relative.is_absolute() or ".." in relative.parts or relative.as_posix() != value:
        raise ValueError(f"{label} path is unsafe")
    return value


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags)
    try:
        with os.fdopen(descriptor, "rb") as source:
            for block in iter(lambda: source.read(1024 * 1024), b""):
                digest.update(block)
    except BaseException:
        raise
    return digest.hexdigest()


def _validate_build_paths(
    shards: tuple[Path, ...],
    output_root: Path,
    scratch_root: Path,
) -> None:
    if not 1 <= len(shards) <= _MAX_DECLARED_SHARDS:
        raise ValueError("B builder requires between 1 and 201 declared shards")
    if len(set(shards)) != len(shards):
        raise ValueError("B builder declared shards must be unique")
    if any(not path.is_file() or path.is_symlink() for path in shards):
        raise ValueError("B builder requires regular non-symlink declared shards")
    if os.path.lexists(output_root):
        raise FileExistsError("B builder output bundle must be immutable")
    if scratch_root.exists() or scratch_root.is_symlink():
        raise FileExistsError(f"B builder scratch root already exists: {scratch_root}")


def _write_canonical_json(path: Path, payload: Mapping[str, object]) -> None:
    with path.open("xb") as destination:
        destination.write((_canonical_json(payload) + "\n").encode())
        destination.flush()
        os.fsync(destination.fileno())


def _canonical_json(payload: Mapping[str, object]) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


def _sha256_json(payload: Mapping[str, object]) -> str:
    return hashlib.sha256(_canonical_json(payload).encode()).hexdigest()


def _utc_timestamp(wall_ns: int) -> str:
    return datetime.fromtimestamp(wall_ns / 1_000_000_000, tz=UTC).isoformat(
        timespec="microseconds"
    )


def main() -> int:
    """Build a B canary bundle from an authenticated Task8 publication."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--task8-publication", type=Path, required=True)
    parser.add_argument("--task9-view", type=Path, required=True)
    parser.add_argument("--task9-view-sha256", required=True)
    parser.add_argument("--task8-publication-sha256", required=True)
    parser.add_argument("--task9-selection-receipt-sha256", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--scratch-root", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--workers", type=int)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--job-id", required=True)
    args = parser.parse_args()
    from common.specdec.qwen4b_b_readiness import load_task9_balanced_view

    view = load_task9_balanced_view(args.task9_view, args.task9_view_sha256)
    if view.strategy != "B-balanced" or view.occurrence_count != 2_000_000:
        raise ValueError("Task9 B projection does not describe the approved B-balanced arm")
    source_projection_sha256 = view.projection_sha256
    publication = load_task8_publication(
        args.task8_publication,
        expected_publication_sha256=args.task8_publication_sha256,
        expected_selection_receipt_sha256=args.task9_selection_receipt_sha256,
        source_commit=args.source_commit,
    )
    shards = tuple(item.path for item in publication.shards)
    if view.publication_sha256 != publication.publication_sha256:
        raise ValueError("Task9 projection and Task8 publication identities differ")
    if tuple((view.shard_root / name).resolve() for name in view.declared_shards) != tuple(
        item.path.resolve() for item in publication.shards
    ):
        raise ValueError("Task9 projection and Task8 shard inventories differ")
    materialize_canary_from_shards(
        shards,
        output_root=args.output_root,
        job_id=args.job_id,
        scratch_root=args.scratch_root,
        seed=args.seed,
        requested_workers=args.workers,
        source_projection_sha256=source_projection_sha256,
        source_task8_publication_sha256=publication.publication_sha256,
        source_corpus_manifest_sha256=publication.corpus_manifest_sha256,
        source_selection_receipt_sha256=publication.selection_receipt_sha256,
        source_shard_inventory_sha256=publication.shard_inventory_sha256,
        source_commit=args.source_commit,
        expected_shards=publication.shards,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
