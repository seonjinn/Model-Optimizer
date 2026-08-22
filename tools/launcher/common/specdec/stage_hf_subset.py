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

"""Selectively stage pinned Hugging Face files as checksummed Parquet/Zstd shards."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import re
import shutil
import tempfile
import urllib.parse
import urllib.request
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Callable

_SHA = re.compile(r"^[0-9a-f]{40}$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_SOURCE_ID = re.compile(r"^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+$")


def sha256_file(path: Path) -> str:
    """Return the streaming SHA-256 digest for one file."""
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_subset_plan(path: Path) -> dict[str, Any]:
    """Load a complete immutable subset plan and attach its content digest."""
    raw = path.read_bytes()
    payload = json.loads(raw)
    if not isinstance(payload, dict) or set(payload) != {
        "schema_version",
        "name",
        "container",
        "files",
    }:
        raise ValueError("subset plan has invalid fields")
    if payload["schema_version"] != 1 or not isinstance(payload["name"], str):
        raise ValueError("subset plan has invalid identity")
    container = payload["container"]
    if (
        not isinstance(container, dict)
        or set(container) != {"path", "sha256"}
        or not Path(str(container["path"])).is_absolute()
        or _SHA256.fullmatch(str(container["sha256"])) is None
    ):
        raise ValueError("subset plan has invalid container identity")
    files = payload["files"]
    if not isinstance(files, list) or not files:
        raise ValueError("subset plan must select at least one file")
    required = {
        "source_id",
        "revision",
        "license",
        "pool",
        "category",
        "response_source",
        "tool_lane",
        "path",
        "bytes",
        "sha256",
    }
    identities: set[tuple[str, str, str]] = set()
    for record in files:
        if not isinstance(record, dict) or set(record) != required:
            raise ValueError("subset plan file has invalid fields")
        if (
            _SHA.fullmatch(str(record["revision"])) is None
            or _SHA256.fullmatch(str(record["sha256"])) is None
        ):
            raise ValueError("subset plan file is not revision and LFS-hash pinned")
        if (
            not isinstance(record["bytes"], int)
            or isinstance(record["bytes"], bool)
            or record["bytes"] < 1
            or record["pool"] != "ptv3"
            or record["response_source"] not in {"target-synth", "trace-replay"}
            or not all(
                isinstance(record[field], str) and record[field]
                for field in ("source_id", "license", "category", "path")
            )
            or _SOURCE_ID.fullmatch(record["source_id"]) is None
            or Path(record["path"]).is_absolute()
            or ".." in Path(record["path"]).parts
            or Path(record["path"]).suffix not in {".jsonl", ".parquet"}
        ):
            raise ValueError("subset plan file has invalid size or lane")
        expected_lane = "recorded-trace" if record["response_source"] == "trace-replay" else "none"
        if record["tool_lane"] != expected_lane:
            raise ValueError("subset plan file has mismatched tool lane")
        identity = (str(record["source_id"]), str(record["revision"]), str(record["path"]))
        if identity in identities:
            raise ValueError("subset plan contains duplicate source files")
        identities.add(identity)
    payload["plan_sha256"] = hashlib.sha256(raw).hexdigest()
    return payload


def _output_name(record: dict[str, Any]) -> str:
    identity = f"{record['source_id']}\0{record['revision']}\0{record['path']}"
    prefix = hashlib.sha256(identity.encode()).hexdigest()[:16]
    stem = Path(str(record["path"])).name.removesuffix(".jsonl").removesuffix(".parquet")
    safe_stem = re.sub(r"[^A-Za-z0-9_.-]+", "-", stem)
    return f"{prefix}-{safe_stem}.parquet"


def _convert_jsonl(source: Path, destination: Path) -> int:
    import pyarrow as pa
    import pyarrow.parquet as pq

    schema = pa.schema([pa.field("raw_json", pa.large_string(), nullable=False)])
    writer = pq.ParquetWriter(destination, schema, compression="zstd")
    row_count = 0
    rows: list[str] = []
    try:
        with source.open(encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, start=1):
                value = line.strip()
                if not value:
                    continue
                try:
                    json.loads(value)
                except json.JSONDecodeError as error:
                    raise ValueError(f"invalid JSONL source row {line_number}") from error
                rows.append(value)
                if len(rows) == 8192:
                    writer.write_table(pa.table({"raw_json": rows}, schema=schema))
                    row_count += len(rows)
                    rows = []
        if rows:
            writer.write_table(pa.table({"raw_json": rows}, schema=schema))
            row_count += len(rows)
    finally:
        writer.close()
    if row_count < 1:
        raise ValueError("source file contains no rows")
    return row_count


def _convert_parquet(source: Path, destination: Path) -> int:
    import pyarrow.parquet as pq

    reader = pq.ParquetFile(source)
    writer = pq.ParquetWriter(destination, reader.schema_arrow, compression="zstd")
    row_count = 0
    try:
        for batch in reader.iter_batches(batch_size=8192):
            writer.write_batch(batch)
            row_count += batch.num_rows
    finally:
        writer.close()
    if row_count < 1:
        raise ValueError("source file contains no rows")
    return row_count


def _fetch_huggingface(record: dict[str, Any], destination: Path) -> None:
    quoted_path = urllib.parse.quote(str(record["path"]), safe="/")
    url = (
        f"https://huggingface.co/datasets/{record['source_id']}/resolve/"
        f"{record['revision']}/{quoted_path}?download=true"
    )
    with (
        urllib.request.urlopen(  # nosec B310 -- URL has a fixed HTTPS origin.
            url, timeout=120
        ) as response,
        destination.open("wb") as output,
    ):
        shutil.copyfileobj(response, output, length=8 * 1024 * 1024)


def _manifest_file(root: Path, path: Path, source: dict[str, Any], row_count: int) -> dict:
    return {
        "path": str(path.relative_to(root)),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
        "row_count": row_count,
        "source_id": source["source_id"],
        "source_revision": source["revision"],
        "source_path": source["path"],
        "source_bytes": source["bytes"],
        "source_sha256": source["sha256"],
        "license": source["license"],
        "pool": source["pool"],
        "category": source["category"],
        "response_source": source["response_source"],
        "tool_lane": source["tool_lane"],
    }


def verify_completion(root: Path, *, expected_plan_sha256: str) -> dict[str, Any]:
    """Rehash every published shard and its manifest before reuse."""
    completion_path = root / "completion.json"
    manifest_path = root / "MANIFEST.json"
    try:
        completion = json.loads(completion_path.read_text(encoding="utf-8"))
        manifest_bytes = manifest_path.read_bytes()
        manifest = json.loads(manifest_bytes)
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError("published subset completion is invalid") from error
    if completion != {
        "complete": True,
        "manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
        "plan_sha256": expected_plan_sha256,
        "schema_version": 1,
    }:
        raise ValueError("published subset completion identity mismatch")
    files = manifest.get("files") if isinstance(manifest, dict) else None
    if not isinstance(manifest, dict) or (
        manifest.get("schema_version") != 1
        or manifest.get("plan_sha256") != expected_plan_sha256
        or not isinstance(files, list)
        or manifest.get("file_count") != len(files)
    ):
        raise ValueError("published subset manifest identity mismatch")
    for record in files:
        path = root / record["path"]
        if (
            not path.is_file()
            or path.stat().st_size != record["bytes"]
            or sha256_file(path) != record["sha256"]
        ):
            raise ValueError(f"published subset file identity mismatch: {record['path']}")
    return manifest


def _write_json_durable(path: Path, payload: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _fsync_file(path: Path) -> None:
    with path.open("rb") as stream:
        os.fsync(stream.fileno())


def materialize_subset(
    plan: dict[str, Any],
    output_root: Path,
    *,
    fetch: Callable[[dict[str, Any], Path], Any] = _fetch_huggingface,
    scratch_root: Path | None = None,
) -> dict[str, Any]:
    """Download, verify, convert, and atomically publish one selective subset."""
    if output_root.exists():
        return verify_completion(output_root, expected_plan_sha256=plan["plan_sha256"])
    output_root.parent.mkdir(parents=True, exist_ok=True)
    lock_path = output_root.with_name(f".{output_root.name}.lock")
    lock = lock_path.open("a+")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as error:
        lock.close()
        raise RuntimeError(f"subset publication is already locked: {output_root}") from error
    partial = output_root.with_name(f".{output_root.name}.partial")
    try:
        if output_root.exists():
            return verify_completion(output_root, expected_plan_sha256=plan["plan_sha256"])
        if partial.exists():
            shutil.rmtree(partial)
        partial.mkdir()
        scratch_parent = scratch_root or output_root.parent
        scratch_parent.mkdir(parents=True, exist_ok=True)
        records = []
        with tempfile.TemporaryDirectory(dir=scratch_parent) as temporary:
            temporary_root = Path(temporary)
            for index, source_record in enumerate(plan["files"]):
                downloaded = temporary_root / f"source-{index:03d}"
                fetch(source_record, downloaded)
                if (
                    downloaded.stat().st_size != source_record["bytes"]
                    or sha256_file(downloaded) != source_record["sha256"]
                ):
                    raise ValueError(f"source file identity mismatch: {source_record['path']}")
                destination = partial / _output_name(source_record)
                row_count = (
                    _convert_parquet(downloaded, destination)
                    if str(source_record["path"]).endswith(".parquet")
                    else _convert_jsonl(downloaded, destination)
                )
                _fsync_file(destination)
                records.append(_manifest_file(partial, destination, source_record, row_count))
                downloaded.unlink()
        manifest = {
            "schema_version": 1,
            "name": plan["name"],
            "plan_sha256": plan["plan_sha256"],
            "container": plan["container"],
            "format": "parquet",
            "compression": "zstd",
            "file_count": len(records),
            "row_count": sum(record["row_count"] for record in records),
            "files": records,
        }
        manifest_path = partial / "MANIFEST.json"
        _write_json_durable(manifest_path, manifest)
        completion = {
            "schema_version": 1,
            "complete": True,
            "plan_sha256": plan["plan_sha256"],
            "manifest_sha256": sha256_file(manifest_path),
        }
        _write_json_durable(partial / "completion.json", completion)
        _fsync_directory(partial)
        os.rename(partial, output_root)
        _fsync_directory(output_root.parent)
    except BaseException:
        shutil.rmtree(partial, ignore_errors=True)
        raise
    finally:
        fcntl.flock(lock, fcntl.LOCK_UN)
        lock.close()
    return verify_completion(output_root, expected_plan_sha256=plan["plan_sha256"])


def main() -> int:
    """Run the selective subset stager CLI."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--scratch-root", type=Path, required=True)
    args = parser.parse_args()
    materialize_subset(
        load_subset_plan(args.plan),
        args.output_root,
        scratch_root=args.scratch_root,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
