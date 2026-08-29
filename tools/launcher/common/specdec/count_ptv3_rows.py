# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Count rows in the staged PTv3 line-delimited shards.

The Hugging Face dataset index reports row counts only for repos its parquet
conversion covers. Four of the ten PTv3 candidates ship raw JSONL and are not
converted, so their counts have to be measured from the staged bytes. Those
four are exactly the repos whose share of the blend is still unpinned.

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
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

__all__ = ["count_repo", "count_shard", "main"]

_READ_BLOCK_BYTES = 8 * 1024 * 1024
_SHARD_SUFFIXES = (".jsonl", ".jsonl.gz", ".json.gz", ".gz")


def _is_shard(path: Path) -> bool:
    name = path.name
    return path.is_file() and any(name.endswith(suffix) for suffix in _SHARD_SUFFIXES)


def count_shard(path: Path) -> int:
    """Count newline-delimited records in one shard, gzipped or plain."""
    opener = gzip.open if path.name.endswith(".gz") else open
    lines = 0
    trailing_newline = True
    with opener(path, "rb") as handle:
        while True:
            block = handle.read(_READ_BLOCK_BYTES)
            if not block:
                break
            lines += block.count(b"\n")
            trailing_newline = block.endswith(b"\n")
    # A final record without a trailing newline is still a record.
    if not trailing_newline:
        lines += 1
    return lines


def count_repo(repo_dir: Path, workers: int) -> tuple[int, list[str]]:
    """Return the repo's total row count and the shards that produced it."""
    shards = sorted(path for path in repo_dir.rglob("*") if _is_shard(path))
    if not shards:
        return 0, []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        counts = list(pool.map(count_shard, shards))
    return sum(counts), [str(shard.relative_to(repo_dir)) for shard in shards]


def main() -> None:
    """Count staged repos, self-check against the index, and write a receipt."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=16)
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
