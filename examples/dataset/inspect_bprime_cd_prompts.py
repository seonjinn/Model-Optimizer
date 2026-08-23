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
from collections import defaultdict
from hashlib import sha256
from pathlib import Path
from typing import TYPE_CHECKING, Any

from build_specdec_inventory import APPROVED_PTV2_ALLOWLIST_SHA256, APPROVED_PTV2_REVISION
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
    _verify_complete_artifact(root, index_path, manifest)
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
        "ptv2_allowlist_sha256",
    ):
        _validate_digest(identity.get(name), name)
        if name.endswith("receipt_sha256") and identity[name] == "0" * 64:
            raise ValueError("selection manifest contains a zero exclusion receipt")
    if (
        not isinstance(identity.get("ptv2_revision"), str)
        or _REVISION.fullmatch(identity["ptv2_revision"]) is None
    ):
        raise ValueError("selection PTV2 revision is malformed")
    if (
        identity["ptv2_revision"] != APPROVED_PTV2_REVISION
        or identity["ptv2_allowlist_sha256"] != APPROVED_PTV2_ALLOWLIST_SHA256
    ):
        raise ValueError("selection PTV2 allowlist identity is not approved")
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
        floor_groups = (
            list(proof["bucket_floors"].values())
            if arm == "B-prime"
            else [
                *proof["non_agentic_bucket_floors"].values(),
                *proof["lane_bucket_floors"].values(),
            ]
        )
        expected_reserve = sum(
            (floor * numerator + denominator - 1) // denominator - floor
            for buckets in floor_groups
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


def _verify_complete_artifact(root: Path, index_path: Path, manifest: dict[str, Any]) -> None:
    descriptors = {shard["path"]: shard for shard in manifest["shards"]}
    for relative, descriptor in descriptors.items():
        path = _confined(root, relative)
        digest = sha256()
        byte_count = 0
        row_count = 0
        try:
            with path.open("rb") as handle:
                for line in handle:
                    digest.update(line)
                    byte_count += len(line)
                    row_count += 1
        except OSError as error:
            raise ValueError(f"selection shard is unreadable: {error}") from error
        if (
            digest.hexdigest() != descriptor["sha256"]
            or byte_count != descriptor["byte_count"]
            or row_count != descriptor["row_count"]
        ):
            raise ValueError(f"selection shard digest/count mismatch: {relative}")

    uri = f"file:{index_path}?mode=ro&immutable=1"
    connection = sqlite3.connect(uri, uri=True)
    primary_counts: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    lane_counts: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    bucket_floors: dict[str, dict[str, dict[str, int]]] = defaultdict(
        lambda: defaultdict(lambda: defaultdict(int))
    )
    lane_floors: dict[str, dict[str, dict[str, int]]] = defaultdict(
        lambda: defaultdict(lambda: defaultdict(int))
    )
    status_counts: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    rank_moments: dict[tuple[str, str, str, str], list[int]] = defaultdict(lambda: [0, 0, 0, -1])
    global_expected = 0
    current_relative: str | None = None
    shard_handle: Any = None
    try:
        duplicate = connection.execute(
            "SELECT 1 FROM rows GROUP BY arm,prompt_uuid HAVING count(*) != 1 LIMIT 1"
        ).fetchone()
        if duplicate is not None:
            raise ValueError("selection row-derived UUID uniqueness proof mismatch")
        records = connection.execute(
            "SELECT global_index,arm,status,selection_index,shard_path,byte_offset,byte_length,"
            "row_sha256 FROM rows ORDER BY global_index"
        )
        for record in records:
            global_index, arm, status, selection_index, *location = record
            if global_index != global_expected:
                raise ValueError("selection index global ordering proof mismatch")
            global_expected += 1
            if selection_index != status_counts[arm][status]:
                raise ValueError("selection index per-arm ordering proof mismatch")
            status_counts[arm][status] += 1
            relative = location[0]
            if relative != current_relative:
                if shard_handle is not None:
                    shard_handle.close()
                shard_handle = _confined(root, relative).open("rb")
                current_relative = relative
            row = _read_indexed_row(
                root,
                (arm, status, selection_index, *location),
                manifest,
                descriptors,
                shard_handle,
            )
            cell = _semantic_cell(row)
            group = (arm, cell, row["lane"], row["context_bucket"])
            moment = rank_moments[group]
            rank = row["candidate_rank"]
            moment[0] += 1
            moment[1] += rank
            moment[2] += rank * rank
            moment[3] = max(moment[3], rank)
            if status == "primary":
                primary_counts[arm][cell] += 1
                bucket_floors[arm][cell][row["context_bucket"]] += 1
                if row["domain"] == "swe-agentic-tool":
                    lane_counts[arm][row["lane"]] += 1
                    lane_floors[arm][row["lane"]][row["context_bucket"]] += 1
    finally:
        if shard_handle is not None:
            shard_handle.close()
        connection.close()
    if global_expected != manifest["row_count"]:
        raise ValueError("selection row-derived total count proof mismatch")
    for moment in rank_moments.values():
        count, total, squares, maximum = moment
        if (
            maximum != count - 1
            or total != count * (count - 1) // 2
            or squares != count * (count - 1) * (2 * count - 1) // 6
        ):
            raise ValueError("selection candidate-rank ordering proof mismatch")
    for arm, proof in manifest["arms"].items():
        derived_cells = dict(primary_counts[arm])
        if derived_cells != proof["cell_counts"]:
            raise ValueError(f"selection arm {arm} row-derived cell count proof mismatch")
        derived_lanes = dict(lane_counts[arm])
        if derived_lanes != proof["lane_counts"]:
            raise ValueError(f"selection arm {arm} row-derived lane count proof mismatch")
        derived_buckets = _plain_nested(bucket_floors[arm])
        if derived_buckets != proof["bucket_floors"]:
            raise ValueError(f"selection arm {arm} row-derived bucket floor proof mismatch")
        derived_lane_floors = _plain_nested(lane_floors[arm])
        if derived_lane_floors != proof["lane_bucket_floors"]:
            raise ValueError(f"selection arm {arm} row-derived lane/bucket floor proof mismatch")
        count_payload = {
            "arm": arm,
            "primary_count": status_counts[arm]["primary"],
            "reserve_count": status_counts[arm]["reserve"],
            "cell_counts": proof["cell_counts"],
            "lane_counts": proof["lane_counts"],
            "bucket_floors": proof["bucket_floors"],
            "lane_bucket_floors": proof["lane_bucket_floors"],
            "acquisition_count": status_counts[arm]["primary"] + status_counts[arm]["reserve"],
        }
        if sha256_bytes(canonical_json(count_payload)) != proof["count_proof_sha256"]:
            raise ValueError(f"selection arm {arm} count proof SHA mismatch")
    _verify_derived_digests(index_path, manifest)


def _verify_derived_digests(index_path: Path, manifest: dict[str, Any]) -> None:
    c = manifest["arms"]["C"]
    d = manifest["arms"]["D"]
    paired = sha256(
        canonical_json(
            {
                "bucket_floors": c["non_agentic_bucket_floors"],
                "C_count_proof_sha256": c["count_proof_sha256"],
                "D_count_proof_sha256": d["count_proof_sha256"],
            }
        )
    )
    connection = sqlite3.connect(f"file:{index_path}?mode=ro&immutable=1", uri=True)
    try:
        mismatch = connection.execute(
            "SELECT count(*) FROM ("
            "SELECT status,prompt_uuid FROM rows WHERE arm='C' AND domain!='swe-agentic-tool' "
            "EXCEPT SELECT status,prompt_uuid FROM rows WHERE arm='D' AND domain!='swe-agentic-tool'"
            ")"
        ).fetchone()[0]
        if mismatch:
            raise ValueError("selection paired C/D UUID set proof mismatch")
        for row in connection.execute(
            "SELECT status,prompt_uuid FROM rows WHERE arm='C' AND domain!='swe-agentic-tool' "
            "ORDER BY status,prompt_uuid"
        ):
            paired.update(canonical_json(list(row)))
            paired.update(b"\n")
        if paired.hexdigest() != manifest["paired_cd_sha256"]:
            raise ValueError("selection paired C/D UUID proof mismatch")
        identity = manifest["identity"]
        selection_metadata = {
            "schema_version": 2,
            "policy_sha256": identity["policy_sha256"],
            "seed": identity["seed"],
            "source_inventory_sha256": identity["source_inventory_sha256"],
            "baseline_receipt_sha256": identity["baseline_receipt_sha256"],
            "held_out_receipt_sha256": identity["held_out_receipt_sha256"],
            "ptv2_revision": identity["ptv2_revision"],
            "ptv2_allowlist_sha256": identity["ptv2_allowlist_sha256"],
            "paired_cd_sha256": manifest["paired_cd_sha256"],
            "arms": manifest["arms"],
        }
        selection = sha256(canonical_json(selection_metadata))
        for row in connection.execute(
            "SELECT arm,status,selection_index,prompt_uuid FROM rows "
            "ORDER BY arm,status,selection_index"
        ):
            selection.update(canonical_json(list(row)))
            selection.update(b"\n")
        if selection.hexdigest() != manifest["selection_sha256"]:
            raise ValueError("selection SHA row-derived identity mismatch")
    finally:
        connection.close()


def _semantic_cell(row: dict[str, Any]) -> str:
    if row["arm"] != "B-prime":
        return str(row["domain"])
    if row["domain"] == "stem-science":
        return "stem"
    return {
        "ja": "japanese",
        "es": "spanish",
        "fr": "french",
        "it": "italian",
    }.get(row["language"], f"invalid:{row['language']}")


def _plain_nested(value: dict[str, dict[str, int]]) -> dict[str, dict[str, int]]:
    return {
        outer: {inner: counts[inner] for inner in sorted(counts, key=_bucket_sort_key)}
        for outer, counts in sorted(value.items())
    }


def _bucket_sort_key(bucket: str) -> tuple[int, str]:
    order = ("le4k", "4k_16k", "16k_32k")
    return (order.index(bucket), bucket) if bucket in order else (len(order), bucket)


def _read_indexed_row(
    root: Path,
    index_row: Sequence[Any],
    manifest: dict[str, Any],
    descriptors: dict[str, Any] | None = None,
    shard_handle: Any = None,
) -> dict[str, Any]:
    arm, status, selection_index, relative, offset, length, row_sha = index_row
    descriptors = descriptors or {shard["path"]: shard for shard in manifest["shards"]}
    if relative not in descriptors:
        raise ValueError("selection index references an unauthenticated shard")
    shard = _confined(root, relative)
    if shard_handle is None:
        with shard.open("rb") as handle:
            handle.seek(offset)
            line = handle.read(length)
    else:
        shard_handle.seek(offset)
        line = shard_handle.read(length)
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
