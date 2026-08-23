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

"""Inspect bounded rows from an externally authenticated sharded prompt selection."""

from __future__ import annotations

import argparse
import json
import re
import sqlite3
from hashlib import sha256
from pathlib import Path
from typing import TYPE_CHECKING, Any

from specdec_corpus_contracts import canonical_json, sha256_bytes

if TYPE_CHECKING:
    from collections.abc import Sequence

__all__ = ["inspect_prompt_manifest", "main"]

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_REVISION = re.compile(r"^[0-9a-f]{40}$")
_ARMS = ("B-prime", "C", "D")
_MAX_LIMIT = 1_000


def inspect_prompt_manifest(
    path: Path,
    *,
    expected_root_sha256: str,
    offset: int = 0,
    limit: int = 100,
    arm: str | None = None,
    domain: str | None = None,
    lane: str | None = None,
    language: str | None = None,
    prompt_uuid: str | None = None,
) -> dict[str, Any]:
    """Authenticate compact proofs, then read only the indexed requested row spans."""
    if isinstance(offset, bool) or not isinstance(offset, int) or offset < 0:
        raise ValueError("offset must be a non-negative integer")
    if isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= _MAX_LIMIT:
        raise ValueError(f"limit must be between 1 and {_MAX_LIMIT}")
    if arm is not None and arm not in _ARMS:
        raise ValueError(f"arm must be one of {_ARMS}")
    _validate_digest(expected_root_sha256, "externally trusted root")
    if expected_root_sha256 == "0" * 64:
        raise ValueError("externally trusted root digest must be nonzero")
    if prompt_uuid is not None:
        _validate_digest(prompt_uuid, "prompt UUID filter")
    manifest_path = path / "SELECTION_MANIFEST.json" if path.is_dir() else path
    manifest = _read_manifest(manifest_path, expected_root_sha256)
    root = manifest_path.parent
    index_path = _confined(root, manifest["index"]["path"])
    if _file_sha256(index_path) != manifest["index"]["sha256"]:
        raise ValueError("selection index digest mismatch")
    filters = {
        "arm": arm,
        "domain": domain,
        "lane": lane,
        "language": language,
        "prompt_uuid": prompt_uuid,
    }
    clauses: list[str] = []
    values: list[Any] = []
    for key, value in filters.items():
        if value is not None:
            clauses.append(f"{key}=?")
            values.append(value)
    where = f" WHERE {' AND '.join(clauses)}" if clauses else ""
    uri = f"file:{index_path}?mode=ro&immutable=1"
    try:
        connection = sqlite3.connect(uri, uri=True)
        total = int(connection.execute(f"SELECT count(*) FROM rows{where}", values).fetchone()[0])
        records = connection.execute(
            "SELECT arm,status,selection_index,shard_path,byte_offset,byte_length,row_sha256 "
            f"FROM rows{where} ORDER BY global_index LIMIT ? OFFSET ?",
            (*values, limit, offset),
        )
        rows = [_read_indexed_row(root, record, manifest) for record in records]
    except sqlite3.DatabaseError as error:
        raise ValueError(f"selection index is malformed: {error}") from error
    finally:
        if "connection" in locals():
            connection.close()
    return {
        "schema_version": 2,
        "selection_sha256": manifest["selection_sha256"],
        "root_sha256": expected_root_sha256,
        "filters": filters,
        "total_matches": total,
        "offset": offset,
        "limit": limit,
        "rows_examined": len(rows),
        "rows": rows,
    }


def _read_manifest(path: Path, expected_root: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_bytes())
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"selection manifest is unreadable: {error}") from error
    if not isinstance(payload, dict) or payload.get("schema_version") != 2:
        raise ValueError("selection manifest has an invalid version or shape")
    actual = payload.get("root_sha256")
    _validate_digest(actual, "selection manifest root")
    unsigned = dict(payload)
    del unsigned["root_sha256"]
    computed = sha256_bytes(canonical_json(unsigned))
    if computed != actual or actual != expected_root:
        raise ValueError("selection manifest does not match externally trusted root digest")
    required = {
        "selection_sha256",
        "paired_cd_sha256",
        "identity",
        "arms",
        "row_count",
        "shards",
        "index",
    }
    if not required <= payload.keys():
        raise ValueError("selection manifest is missing semantic proofs")
    for name in ("selection_sha256", "paired_cd_sha256"):
        _validate_digest(payload[name], name)
    identity = payload["identity"]
    if not isinstance(identity, dict):
        raise ValueError("selection manifest identity is malformed")
    for name in (
        "policy_sha256",
        "source_inventory_sha256",
        "baseline_receipt_sha256",
        "held_out_receipt_sha256",
    ):
        _validate_digest(identity.get(name), name)
        if name.endswith("receipt_sha256") and identity[name] == "0" * 64:
            raise ValueError("selection manifest contains a zero exclusion receipt")
    if (
        not isinstance(identity.get("ptv2_revision"), str)
        or _REVISION.fullmatch(identity["ptv2_revision"]) is None
    ):
        raise ValueError("selection PTV2 revision is malformed")
    if isinstance(identity.get("seed"), bool) or not isinstance(identity.get("seed"), int):
        raise ValueError("selection seed is malformed")
    if not isinstance(payload["index"], dict):
        raise ValueError("selection index descriptor is malformed")
    _validate_digest(payload["index"].get("sha256"), "selection index")
    try:
        _validate_semantic_counts(payload)
    except (KeyError, TypeError, AttributeError) as error:
        raise ValueError("selection manifest semantic proofs are malformed") from error
    return payload


def _validate_semantic_counts(manifest: dict[str, Any]) -> None:
    arms = manifest["arms"]
    if not isinstance(arms, dict) or set(arms) != set(_ARMS):
        raise ValueError("selection manifest arms are malformed")
    total = 0
    if not isinstance(manifest["shards"], list):
        raise ValueError("selection manifest shard proofs are malformed")
    numerator = manifest["identity"].get("reserve_numerator")
    denominator = manifest["identity"].get("reserve_denominator")
    if not isinstance(numerator, int) or not isinstance(denominator, int) or denominator <= 0:
        raise ValueError("selection reserve ratio is malformed")
    for arm, proof in arms.items():
        if not isinstance(proof, dict):
            raise ValueError(f"selection manifest arm {arm} proof is malformed")
        _validate_digest(proof.get("count_proof_sha256"), f"selection arm {arm} count proof")
        if proof["primary_count"] != sum(proof["cell_counts"].values()):
            raise ValueError(f"selection manifest arm {arm} cell count proof mismatch")
        expected_reserve = sum(
            (floor * numerator + denominator - 1) // denominator - floor
            for buckets in proof["bucket_floors"].values()
            for floor in buckets.values()
        )
        if proof["reserve_count"] != expected_reserve:
            raise ValueError(f"selection manifest arm {arm} reserve arithmetic mismatch")
        total += proof["primary_count"] + proof["reserve_count"]
    if total != manifest["row_count"]:
        raise ValueError("selection manifest row count proof mismatch")
    if arms["C"]["non_agentic_bucket_floors"] != arms["D"]["non_agentic_bucket_floors"]:
        raise ValueError("selection manifest C/D pairing floor proof mismatch")
    if sum(shard["row_count"] for shard in manifest["shards"]) != total:
        raise ValueError("selection manifest shard count proof mismatch")
    for shard in manifest["shards"]:
        _validate_digest(shard.get("sha256"), "selection shard")


def _read_indexed_row(
    root: Path, index_row: Sequence[Any], manifest: dict[str, Any]
) -> dict[str, Any]:
    arm, status, selection_index, relative, offset, length, row_sha = index_row
    descriptors = {shard["path"]: shard for shard in manifest["shards"]}
    if relative not in descriptors:
        raise ValueError("selection index references an unauthenticated shard")
    shard = _confined(root, relative)
    with shard.open("rb") as handle:
        handle.seek(offset)
        line = handle.read(length)
    if sha256(line).hexdigest() != row_sha:
        raise ValueError("selected row digest mismatch")
    try:
        row = json.loads(line)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("selected row is malformed") from error
    if (
        row.get("arm") != arm
        or row.get("status") != status
        or row.get("selection_index") != selection_index
    ):
        raise ValueError("selection index row metadata mismatch")
    canonical = row.get("canonical_prompt")
    if not isinstance(canonical, dict) or sha256(canonical_json(canonical)).hexdigest() != row.get(
        "prompt_uuid"
    ):
        raise ValueError("selected row canonical prompt identity mismatch")
    fields = (
        manifest["identity"]["policy_sha256"],
        str(manifest["identity"]["seed"]),
        row["domain"],
        row["lane"],
        row["language"],
        row["context_bucket"],
        row["source_id"],
        row["source_revision"],
        row["source_file_sha256"],
        row["source_manifest_sha256"],
        row["source_file_path"],
        str(row["source_row_index"]),
        row["prompt_uuid"],
    )
    if sha256("\0".join(fields).encode()).hexdigest() != row.get("candidate_rank_sha256"):
        raise ValueError("selected row candidate ranking proof mismatch")
    if row["arm"] == "B-prime" and (
        row.get("source_family") != "ptv2"
        or row["source_revision"] != manifest["identity"]["ptv2_revision"]
    ):
        raise ValueError("selected B-prime row is not from the pinned PTV2 source")
    return row


def _validate_digest(value: Any, label: str) -> None:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ValueError(f"{label} must be an exact lowercase SHA-256")


def _file_sha256(path: Path) -> str:
    digest = sha256()
    try:
        with path.open("rb") as handle:
            while chunk := handle.read(1024 * 1024):
                digest.update(chunk)
    except OSError as error:
        raise ValueError(f"selection index is unreadable: {error}") from error
    return digest.hexdigest()


def _confined(root: Path, relative: str) -> Path:
    if not isinstance(relative, str) or not relative:
        raise ValueError("selection artifact path is malformed")
    resolved = (root / relative).resolve()
    if root.resolve() not in resolved.parents:
        raise ValueError("selection artifact path escapes publication root")
    return resolved


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("selection_manifest", type=Path)
    parser.add_argument("--expected-root-sha256", required=True)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--arm", choices=_ARMS)
    parser.add_argument("--domain")
    parser.add_argument("--lane")
    parser.add_argument("--language")
    parser.add_argument("--prompt-uuid")
    args = parser.parse_args(argv)
    result = inspect_prompt_manifest(
        args.selection_manifest,
        expected_root_sha256=args.expected_root_sha256,
        offset=args.offset,
        limit=args.limit,
        arm=args.arm,
        domain=args.domain,
        lane=args.lane,
        language=args.language,
        prompt_uuid=args.prompt_uuid,
    )
    print(json.dumps(result, ensure_ascii=False, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
