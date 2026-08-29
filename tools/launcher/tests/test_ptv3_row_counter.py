# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Byte-range row counting must reproduce the serial count exactly.

The counter splits large JSONL shards across workers by byte offset. A record
is owned by the chunk holding its first byte, so a chunk skips the record
straddling its start and reads past its end to finish the record it owns. These
tests pin that rule at every possible boundary alignment: an off-by-one here
would reweight the corpus blend silently rather than fail.
"""

from __future__ import annotations

import gzip
import json
from itertools import pairwise
from typing import TYPE_CHECKING

import pytest
from common.specdec import count_ptv3_rows as counter

if TYPE_CHECKING:
    from pathlib import Path

# Each case pairs raw shard bytes with the number of records the serial reader
# finds. Blank lines and a missing trailing newline are both ordinary results of
# concatenating shards.
_CORPUS_CASES = {
    "trailing_newline": (b'{"a": 1}\n{"a": 2}\n{"a": 3}\n', 3),
    "no_trailing_newline": (b'{"a": 1}\n{"a": 2}\n{"a": 3}', 3),
    "blank_lines_between": (b'{"a": 1}\n\n{"a": 2}\n\n\n{"a": 3}\n', 3),
    "leading_blank_line": (b'\n{"a": 1}\n{"a": 2}\n', 2),
    "whitespace_only_line": (b'{"a": 1}\n   \n{"a": 2}\n', 2),
    "single_record": (b'{"a": 1}\n', 1),
    "uneven_record_widths": (
        b'{"a": 1}\n{"text": "a much longer record than the others"}\n{"a": 3}\n',
        3,
    ),
    "empty": (b"", 0),
    "blank_only": (b"\n\n\n", 0),
}


def _write(tmp_path: Path, name: str, payload: bytes) -> Path:
    path = tmp_path / name
    path.write_bytes(payload)
    return path


def _chunked_total(path: Path, chunk: int) -> int:
    """Sum _count_range over chunks that tile the file, as the pool would."""
    size = path.stat().st_size
    if size == 0:
        return 0
    return sum(
        counter._count_range(path, start, min(start + chunk, size))
        for start in range(0, size, chunk)
    )


@pytest.mark.parametrize("case", sorted(_CORPUS_CASES))
def test_every_chunk_boundary_reproduces_the_serial_count(tmp_path, case):
    """Sweep every alignment: mid-record, exactly on a newline, past the end."""
    payload, expected = _CORPUS_CASES[case]
    path = _write(tmp_path, "shard.jsonl", payload)
    assert counter._count_lines(path) == expected
    for chunk in range(1, len(payload) + 3):
        assert _chunked_total(path, chunk) == expected, f"{case} at chunk={chunk}"


def test_a_chunk_smaller_than_one_record_still_counts_it_once(tmp_path):
    """A record spanning many chunks belongs only to the chunk holding byte 0."""
    payload = b'{"text": "' + b"x" * 4096 + b'"}\n{"a": 2}\n'
    path = _write(tmp_path, "shard.jsonl", payload)
    assert _chunked_total(path, 64) == 2


def test_a_chunk_landing_on_the_final_newline_adds_nothing(tmp_path):
    """The last chunk owns no record when the file ends in a newline."""
    payload = b'{"a": 1}\n{"a": 2}\n'
    path = _write(tmp_path, "shard.jsonl", payload)
    size = len(payload)
    assert counter._count_range(path, 0, 9) == 1
    assert counter._count_range(path, 9, size) == 1
    assert counter._count_range(path, size - 1, size) == 0


def test_chunked_and_whole_file_planning_agree_on_a_repo(tmp_path, monkeypatch):
    """count_repo with chunking on returns what the whole-file reader returns."""
    repo = tmp_path / "repo" / "data"
    repo.mkdir(parents=True)
    big = b"".join(b'{"i": %d}\n' % index for index in range(5000))
    small = b'{"i": 0}\n{"i": 1}\n'
    (repo / "big.jsonl").write_bytes(big)
    (repo / "small.jsonl").write_bytes(small)
    (repo / "packed.jsonl.gz").write_bytes(gzip.compress(small))

    serial, shards = counter.count_repo(tmp_path / "repo", workers=2)
    assert serial == 5004
    assert len(shards) == 3

    monkeypatch.setattr(counter, "_CHUNK_TARGET_BYTES", 512)
    chunked, chunked_shards = counter.count_repo(tmp_path / "repo", workers=4)
    assert chunked == serial
    assert chunked_shards == shards


def test_gzip_and_parquet_shards_are_never_split(tmp_path, monkeypatch):
    """Gzip members cannot be entered at an offset; parquet is a footer read."""
    monkeypatch.setattr(counter, "_CHUNK_TARGET_BYTES", 8)
    packed = _write(tmp_path, "shard.jsonl.gz", gzip.compress(b'{"a": 1}\n' * 100))
    table = _write(tmp_path, "shard.parquet", b"PAR1notreallyparquet")
    assert counter._plan_shard(packed) == [(packed, None, None)]
    assert counter._plan_shard(table) == [(table, None, None)]


def test_a_large_plain_shard_is_split_into_tiling_chunks(tmp_path, monkeypatch):
    """Chunks must tile [0, size) with no gap and no overlap."""
    monkeypatch.setattr(counter, "_CHUNK_TARGET_BYTES", 16)
    payload = b'{"a": 1}\n' * 20
    path = _write(tmp_path, "shard.jsonl", payload)
    tasks = counter._plan_shard(path)
    bounds = [(start, end) for _, start, end in tasks]
    assert bounds[0][0] == 0
    assert bounds[-1][1] == len(payload)
    assert all(left[1] == right[0] for left, right in pairwise(bounds))


def test_a_chunked_shard_still_fails_when_its_first_record_is_not_json(tmp_path, monkeypatch):
    """Planning validates the first record the chunked readers never see whole."""
    monkeypatch.setattr(counter, "_CHUNK_TARGET_BYTES", 16)
    path = _write(tmp_path, "shard.jsonl", b"id,text\n1,hello\n2,world\n")
    with pytest.raises(SystemExit, match="not a JSONL"):
        counter._plan_shard(path)


def test_the_counted_records_are_the_json_records(tmp_path):
    """Guard the record definition itself, not just the arithmetic."""
    records = [{"i": index} for index in range(64)]
    payload = b"".join(json.dumps(record).encode() + b"\n" for record in records)
    path = _write(tmp_path, "shard.jsonl", payload)
    assert counter._count_lines(path) == len(records)
    assert _chunked_total(path, 7) == len(records)


def test_workers_default_to_the_slurm_allocation(monkeypatch):
    """--workers 48 on a 96-core node left half the node idle."""
    monkeypatch.delenv("SLURM_CPUS_ON_NODE", raising=False)
    monkeypatch.setenv("SLURM_CPUS_PER_TASK", "96")
    assert counter._default_workers() == 96


def test_an_exclusive_allocation_falls_back_to_cpus_on_node(monkeypatch):
    """--exclusive sets no CPUS_PER_TASK, and data-mover nodes are 90 or 96."""
    monkeypatch.delenv("SLURM_CPUS_PER_TASK", raising=False)
    monkeypatch.setenv("SLURM_CPUS_ON_NODE", "90")
    assert counter._default_workers() == 90


def test_an_unset_or_empty_allocation_falls_back_to_the_machine(monkeypatch):
    """Outside Slurm the counter still runs, just at whatever the host has."""
    monkeypatch.setenv("SLURM_CPUS_PER_TASK", "")
    monkeypatch.setenv("SLURM_CPUS_ON_NODE", "")
    assert counter._default_workers() >= 1
    monkeypatch.delenv("SLURM_CPUS_PER_TASK", raising=False)
    monkeypatch.delenv("SLURM_CPUS_ON_NODE", raising=False)
    assert counter._default_workers() >= 1
