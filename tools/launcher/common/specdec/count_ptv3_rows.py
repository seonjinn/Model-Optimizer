# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Count rows in the staged PTv3 shards.

The Hugging Face dataset index reports row counts only for repos its parquet
conversion covers. Four of the ten PTv3 candidates ship raw JSONL and are not
converted, so their counts have to be measured from the staged bytes. Those
four are exactly the repos whose share of the blend is still unpinned.

The split runs the other way too: the converted repos are stored *as* parquet,
so a counter that only understands JSONL reports them as zero rows from zero
shards. Both formats are handled here, and a repo that yields no shards at all
is an error rather than a zero - an unrecognised layout is not an empty
dataset, and this is the one number the blend is computed from.

Counting lines is easy to get quietly wrong - a trailing newline, a shard
missed by a glob, a gzip member boundary - and a wrong count here silently
reweights the corpus rather than failing. So the counter validates itself
first: it counts the repos that *do* have an indexed count and refuses to
report anything if its own numbers disagree with the index.
"""

from __future__ import annotations

import argparse
import gzip
import json
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import TYPE_CHECKING, BinaryIO

if TYPE_CHECKING:
    from collections.abc import Iterator

__all__ = ["count_repo", "count_shard", "main"]

_READ_BLOCK_BYTES = 8 * 1024 * 1024
# Six staged shards are 5-15 GB each. Whole-file tasks would pin one worker per
# shard and leave the rest of a 96-core node idle for as long as the largest
# file takes, so plain JSONL above this size is split into byte ranges that the
# pool schedules independently.
_CHUNK_TARGET_BYTES = 256 * 1024 * 1024
# A bare ".gz" is deliberately absent: it would match any gzip file under the
# repo -- a compressed README, an archive -- and fold its newline count into the
# row total. Every staged shard names its format before the compression suffix.
_LINE_SUFFIXES = (".jsonl", ".jsonl.gz", ".json.gz")
_PARQUET_SUFFIX = ".parquet"
_SHARD_SUFFIXES = (*_LINE_SUFFIXES, _PARQUET_SUFFIX)

# (path, start, end); start is None for a whole-file task.
_Task = tuple[Path, "int | None", "int | None"]


def _is_shard(path: Path) -> bool:
    name = path.name
    return path.is_file() and any(name.endswith(suffix) for suffix in _SHARD_SUFFIXES)


def count_shard(path: Path) -> int:
    """Count records in one shard, reading parquet footers and JSONL newlines."""
    if path.name.endswith(_PARQUET_SUFFIX):
        return _count_parquet(path)
    return _count_lines(path)


def _count_parquet(path: Path) -> int:
    """Read the row count out of the parquet footer without scanning the data."""
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise SystemExit(
            f"{path} is parquet but pyarrow is unavailable; run this inside the "
            "pinned runtime image rather than reporting a zero"
        ) from exc
    return pq.ParquetFile(path).metadata.num_rows


def _iter_segments(handle: BinaryIO, position: int) -> Iterator[tuple[int, bytes]]:
    """Yield (absolute offset, bytes) for each newline-delimited segment.

    ``position`` is the file offset the handle is already sitting at, so the
    offsets are absolute even when a chunk starts mid-file. The final segment is
    yielded whether or not the file ends in a newline.
    """
    carry = b""
    carry_offset = position
    while True:
        block = handle.read(_READ_BLOCK_BYTES)
        if not block:
            break
        pieces = (carry + block).split(b"\n")
        carry = pieces.pop()
        offset = carry_offset
        for piece in pieces:
            yield offset, piece
            offset += len(piece) + 1
        carry_offset = offset
    if carry:
        yield carry_offset, carry


def _validate_record(path: Path, record: bytes) -> None:
    """Fail on a shard whose first record is not JSON.

    A file that reached this reader without being JSON lines -- wrong suffix, or
    a whole file on one line because it uses bare carriage returns -- would
    otherwise report a plausible count instead of failing.
    """
    try:
        json.loads(record)
    except ValueError as exc:
        raise SystemExit(
            f"{path}: first record is not JSON ({exc}); it is not a JSONL "
            "shard, so its line count would not be a row count"
        ) from exc


def _count_lines(path: Path) -> int:
    """Count non-empty newline-delimited records in one shard, gzipped or plain.

    Counting newline bytes instead would inflate the total by one for any shard
    that ends in a blank line, which is the ordinary result of concatenating
    shards. The row counts here set the corpus blend ratios, so an off-by-one
    per shard is a silent reweighting rather than a visible failure.
    """
    opener = gzip.open if path.name.endswith(".gz") else open
    records = 0
    first = None
    with opener(path, "rb") as handle:
        for _, segment in _iter_segments(handle, 0):
            if not segment.strip():
                continue
            records += 1
            if first is None:
                first = segment
    if first is not None:
        _validate_record(path, first)
    return records


def _record_start_at_or_after(handle: BinaryIO, start: int) -> int | None:
    """Offset of the first record a chunk beginning at ``start`` owns.

    A record is owned by the chunk holding its *first* byte. The record
    straddling ``start`` therefore belongs to the previous chunk, so this skips
    forward past it. Returns None when the rest of the file holds no record
    start.
    """
    if start == 0:
        return 0
    handle.seek(start - 1)
    previous = handle.read(1)
    if not previous:
        return None
    if previous == b"\n":
        return start
    while True:
        block = handle.read(_READ_BLOCK_BYTES)
        if not block:
            return None
        index = block.find(b"\n")
        if index != -1:
            return handle.tell() - len(block) + index + 1


def _count_range(path: Path, start: int, end: int) -> int:
    """Count records whose first byte lies in ``[start, end)``.

    Chunks tile the file, and every record is owned by exactly one of them, so
    the summed result equals the serial count no matter how the pool schedules
    the workers. That is why a chunk reads past ``end`` to finish the last
    record it owns rather than stopping on a byte boundary.
    """
    records = 0
    with open(path, "rb") as handle:
        first = _record_start_at_or_after(handle, start)
        if first is None or first >= end:
            return 0
        handle.seek(first)
        for offset, segment in _iter_segments(handle, first):
            if offset >= end:
                break
            if segment.strip():
                records += 1
    return records


def _count_task(task: _Task) -> int:
    path, start, end = task
    if start is None or end is None:
        return count_shard(path)
    return _count_range(path, start, end)


def _plan_shard(path: Path) -> list[_Task]:
    """Split one shard into pool tasks.

    Parquet is a footer read, and gzip members cannot be entered at an arbitrary
    offset, so both stay whole-file. Only plain JSONL large enough to matter is
    chunked.
    """
    name = path.name
    if name.endswith((_PARQUET_SUFFIX, ".gz")):
        return [(path, None, None)]
    size = path.stat().st_size
    if size <= _CHUNK_TARGET_BYTES:
        return [(path, None, None)]
    # The chunked path never sees the file's first record as a whole, so the
    # JSON check that _count_lines does per shard happens here instead.
    with open(path, "rb") as handle:
        for _, segment in _iter_segments(handle, 0):
            if segment.strip():
                _validate_record(path, segment)
                break
    return [
        (path, start, min(start + _CHUNK_TARGET_BYTES, size))
        for start in range(0, size, _CHUNK_TARGET_BYTES)
    ]


def count_repo(repo_dir: Path, workers: int) -> tuple[int, list[str]]:
    """Return the repo's total row count and the shards that produced it."""
    shards = sorted(path for path in repo_dir.rglob("*") if _is_shard(path))
    if not shards:
        return 0, []
    tasks = [task for shard in shards for task in _plan_shard(shard)]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        counts = list(pool.map(_count_task, tasks))
    return sum(counts), [str(shard.relative_to(repo_dir)) for shard in shards]


def _default_workers() -> int:
    """Size the pool from the allocation, not from the machine.

    os.cpu_count() reports the node, not the cgroup, so it would oversubscribe a
    partial allocation. Slurm sets CPUS_PER_TASK only when the job asked for it
    and CPUS_ON_NODE whenever a node is held, which is the exclusive case.
    """
    for variable in ("SLURM_CPUS_PER_TASK", "SLURM_CPUS_ON_NODE"):
        allocated = os.environ.get(variable, "")
        if allocated.isdigit() and int(allocated) > 0:
            return int(allocated)
    return os.cpu_count() or 16


def main() -> None:
    """Count staged repos, self-check against the index, and write a receipt."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=_default_workers())
    parser.add_argument(
        "--expect",
        action="append",
        default=[],
        metavar="REPO=ROWS",
        help="indexed row count to validate the counter against",
    )
    parser.add_argument("--repo", action="append", default=[], help="repo to count")
    args = parser.parse_args()

    expected = {}
    for item in args.expect:
        repo, _, rows = item.partition("=")
        expected[repo] = int(rows)

    results: dict[str, dict[str, object]] = {}
    for repo in sorted(set(args.repo) | set(expected)):
        repo_dir = args.data_root / repo
        if not repo_dir.is_dir():
            results[repo] = {"status": "absent"}
            continue
        rows, shards = count_repo(repo_dir, args.workers)
        if not shards:
            results[repo] = {"status": "no_shards"}
            continue
        entry: dict[str, object] = {
            "status": "counted",
            "rows": rows,
            "shard_count": len(shards),
            "shards": shards,
        }
        if repo in expected:
            entry["indexed_rows"] = expected[repo]
            entry["agrees_with_index"] = rows == expected[repo]
        results[repo] = entry

    checked = [
        repo
        for repo, entry in results.items()
        if entry.get("status") == "counted" and "agrees_with_index" in entry
    ]
    disagreed = [repo for repo in checked if not results[repo]["agrees_with_index"]]
    if not checked:
        raise SystemExit("no indexed repo was available to validate the counter")
    # A staged repo that yields no recognised shard is a format this counter
    # does not read, not an empty dataset. Reporting it as zero would drop it
    # from the blend with nothing failing.
    empty = [repo for repo, entry in results.items() if entry["status"] == "no_shards"]
    if empty:
        for repo in empty:
            print(f"{repo}: staged but no recognised shard found")
        raise SystemExit("unrecognised shard layout; counts withheld")
    if disagreed:
        for repo in disagreed:
            entry = results[repo]
            print(f"{repo}: counted {entry['rows']} but index says {entry['indexed_rows']}")
        raise SystemExit("counter disagrees with the dataset index; counts withheld")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w") as handle:
        json.dump(
            {
                "schema": "ptv3-row-count-v1",
                "data_root": str(args.data_root),
                "validated_against": sorted(checked),
                "repos": results,
            },
            handle,
            indent=2,
            sort_keys=True,
        )
        handle.write("\n")

    for repo, entry in sorted(results.items()):
        if entry.get("status") != "counted":
            print(f"{repo:<38} {entry['status']}")
            continue
        note = " (matches index)" if entry.get("agrees_with_index") else ""
        print(f"{repo:<38} rows={entry['rows']:>10,} shards={entry['shard_count']}{note}")


if __name__ == "__main__":
    main()
