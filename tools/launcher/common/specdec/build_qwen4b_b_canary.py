# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Materialize the exact scaled B-balanced source-native canary view."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sqlite3
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator, Mapping

__all__ = [
    "CANARY_CELL_QUOTAS",
    "CANARY_LANGUAGE_QUOTAS",
    "build_canary_occurrences",
    "materialize_canary_from_shards",
    "resolve_worker_count",
    "write_canary_occurrences",
]

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
    ordinal_base: int
    expected_rows: int
    seed: int


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
    if len(selected) != 102_400:
        raise AssertionError("B canary selection must contain 102400 occurrences")
    return selected


def materialize_canary_from_shards(
    shard_paths: Iterable[Path],
    *,
    output_path: Path,
    receipt_path: Path,
    seed: int,
    requested_workers: int | None = None,
    environ: Mapping[str, str] | None = None,
    scratch_root: Path,
    source_projection_sha256: str | None = None,
    source_commit: str | None = None,
) -> None:
    """Build a byte-stable canary through bounded shard-parallel spooling."""
    shards = tuple(Path(path) for path in shard_paths)
    _validate_build_paths(shards, output_path, receipt_path, scratch_root)
    effective_workers, allocated_cpus = resolve_worker_count(
        requested_workers=requested_workers,
        declared_shard_count=len(shards),
        environ=environ,
    )
    started_wall_ns = time.time_ns()
    started_monotonic_ns = time.monotonic_ns()
    scratch_root.mkdir(mode=0o700, parents=True)
    try:
        row_counts = tuple(_count_shard_rows(path) for path in shards)
        ordinal_bases = _exclusive_prefix_sum(row_counts)
        tasks = tuple(
            _ShardTask(
                index=index,
                path=path,
                spool_path=scratch_root / f"shard-{index:03d}.jsonl",
                ordinal_base=ordinal_bases[index],
                expected_rows=row_counts[index],
                seed=seed,
            )
            for index, path in enumerate(shards)
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
            "declared_shard_count": len(shards),
            "source_row_count": sum(result.row_count for result in results),
            "occurrence_count": occurrence_count,
            "output_path": str(output_path),
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
    except BaseException:
        output_path.unlink(missing_ok=True)
        receipt_path.unlink(missing_ok=True)
        raise
    finally:
        shutil.rmtree(scratch_root, ignore_errors=True)


def write_canary_occurrences(path: Path, rows: Iterable[Mapping[str, object]]) -> None:
    """Atomically write canonical JSONL B canary occurrences."""
    if path.exists():
        raise FileExistsError(f"B canary output already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.partial-{os.getpid()}")
    try:
        with temporary.open("w", encoding="utf-8") as destination:
            for row in rows:
                destination.write(_canonical_json(dict(row)) + "\n")
            destination.flush()
            os.fsync(destination.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


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
    digest = hashlib.sha256()
    byte_count = 0
    with task.path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
            byte_count += len(block)
    row_count = 0
    with task.spool_path.open("w", encoding="utf-8") as spool:
        for row_count, raw in enumerate(_iter_shard_rows(task.path), start=1):
            row, category = _prepare_candidate(
                raw,
                seed=task.seed,
                ordinal=task.ordinal_base + row_count - 1,
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
    if row_count != task.expected_rows:
        raise ValueError(
            f"B builder shard row count changed for {task.path}: "
            f"expected {task.expected_rows}, found {row_count}"
        )
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
        temporary = output_path.with_name(f".{output_path.name}.partial-{os.getpid()}")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        digest = hashlib.sha256()
        byte_count = 0
        occurrence_count = 0
        try:
            with temporary.open("wb") as destination:
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
            os.replace(temporary, output_path)
        finally:
            temporary.unlink(missing_ok=True)
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


def _count_shard_rows(path: Path) -> int:
    if path.suffix == ".jsonl":
        with path.open(encoding="utf-8") as source:
            return sum(bool(line.strip()) for line in source)
    if path.suffix == ".parquet":
        try:
            import pyarrow.parquet as pq  # pyright: ignore[reportMissingImports]
        except ImportError as error:
            raise ValueError("pyarrow is required to build from Parquet shards") from error
        return pq.ParquetFile(path).metadata.num_rows
    raise ValueError(f"B builder does not support shard type: {path}")


def _iter_shard_rows(path: Path) -> Iterator[Mapping[str, object]]:
    if path.suffix == ".jsonl":
        with path.open(encoding="utf-8") as source:
            for line in source:
                if line.strip():
                    row = json.loads(line)
                    if not isinstance(row, dict):
                        raise ValueError(f"B builder shard row must be an object: {path}")
                    yield row
        return
    if path.suffix == ".parquet":
        try:
            import pyarrow as pa  # pyright: ignore[reportMissingImports]
            import pyarrow.parquet as pq  # pyright: ignore[reportMissingImports]
        except ImportError as error:
            raise ValueError("pyarrow is required to build from Parquet shards") from error
        pa.set_cpu_count(1)
        pa.set_io_thread_count(1)
        parquet = pq.ParquetFile(path)
        for batch in parquet.iter_batches(batch_size=_ARROW_BATCH_SIZE, use_threads=False):
            yield from batch.to_pylist()
        return
    raise ValueError(f"B builder does not support shard type: {path}")


def _prepare_candidate(
    raw: Mapping[str, object],
    *,
    seed: int,
    ordinal: int,
) -> tuple[dict[str, object], str]:
    row = dict(raw)
    cell = _required_string(row, "cell").lower()
    language = _required_string(row, "language").lower()
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
    if row.get("source_native") is not True:
        raise ValueError("B canary requires source-native completions")
    identifier = _required_string(row, "uuid")
    row["cell"] = cell
    row["language"] = language
    row["_rank"] = _rank(seed, ordinal, identifier)
    return row, category


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


def _required_string(row: Mapping[str, object], key: str) -> str:
    value = row.get(key)
    if not isinstance(value, str):
        raise ValueError(f"B canary row {key} must be a string")
    return value


def _exclusive_prefix_sum(values: tuple[int, ...]) -> tuple[int, ...]:
    total = 0
    result = []
    for value in values:
        result.append(total)
        total += value
    return tuple(result)


def _validate_build_paths(
    shards: tuple[Path, ...],
    output_path: Path,
    receipt_path: Path,
    scratch_root: Path,
) -> None:
    if not 1 <= len(shards) <= _MAX_DECLARED_SHARDS:
        raise ValueError("B builder requires between 1 and 201 declared shards")
    if len(set(shards)) != len(shards):
        raise ValueError("B builder declared shards must be unique")
    if any(not path.is_file() or path.is_symlink() for path in shards):
        raise ValueError("B builder requires regular non-symlink declared shards")
    if output_path == receipt_path:
        raise ValueError("B builder output and receipt paths must be distinct")
    if output_path.exists() or receipt_path.exists():
        raise FileExistsError("B builder output and receipt must be immutable")
    if scratch_root.exists() or scratch_root.is_symlink():
        raise FileExistsError(f"B builder scratch root already exists: {scratch_root}")


def _write_canonical_json(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.partial-{os.getpid()}")
    try:
        with temporary.open("w", encoding="utf-8") as destination:
            destination.write(_canonical_json(payload) + "\n")
            destination.flush()
            os.fsync(destination.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _canonical_json(payload: Mapping[str, object]) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


def _sha256_json(payload: Mapping[str, object]) -> str:
    return hashlib.sha256(_canonical_json(payload).encode()).hexdigest()


def _utc_timestamp(wall_ns: int) -> str:
    return datetime.fromtimestamp(wall_ns / 1_000_000_000, tz=UTC).isoformat(
        timespec="microseconds"
    )


def main() -> int:
    """Build a B canary JSONL file from authenticated Task9 occurrence shards."""
    parser = argparse.ArgumentParser()
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--input", type=Path)
    source.add_argument("--task9-view", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--scratch-root", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--workers", type=int)
    parser.add_argument("--source-commit")
    args = parser.parse_args()
    source_projection_sha256 = None
    if args.task9_view is not None:
        from common.specdec.qwen4b_b_readiness import (
            BReadinessInputs,
            assess_b_readiness,
            load_task9_balanced_view,
        )

        if args.source_commit is None:
            parser.error("--source-commit is required with --task9-view")
        view = load_task9_balanced_view(args.task9_view)
        readiness = assess_b_readiness(
            BReadinessInputs(task9_b_view=view, source_commit=args.source_commit)
        )
        if not readiness.ready:
            raise ValueError(
                "Task9 B projection is not ready: " + ",".join(readiness.blocker_codes)
            )
        shards = tuple(view.shard_root / name for name in view.declared_shards)
        source_projection_sha256 = hashlib.sha256(args.task9_view.read_bytes()).hexdigest()
    else:
        assert args.input is not None
        shards = (args.input,)
    materialize_canary_from_shards(
        shards,
        output_path=args.output,
        receipt_path=args.receipt,
        scratch_root=args.scratch_root,
        seed=args.seed,
        requested_workers=args.workers,
        source_projection_sha256=source_projection_sha256,
        source_commit=args.source_commit,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
