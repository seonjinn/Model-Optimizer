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

"""Inspect bounded, filtered rows from a published B-prime/C/D selection artifact."""

from __future__ import annotations

import argparse
import json
import re
from hashlib import sha256
from pathlib import Path
from typing import TYPE_CHECKING, Any

from specdec_corpus_contracts import canonical_json, sha256_bytes

if TYPE_CHECKING:
    from collections.abc import Sequence

__all__ = ["inspect_prompt_manifest", "main"]

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_ARMS = ("B-prime", "C", "D")
_MAX_LIMIT = 1_000
_ROW_FIELDS = frozenset(
    {
        "prompt_uuid",
        "arm",
        "domain",
        "lane",
        "language",
        "context_bucket",
        "source_id",
        "source_revision",
        "source_file_sha256",
        "source_manifest_sha256",
        "source_file_path",
        "source_row_index",
        "candidate_rank",
        "candidate_rank_sha256",
        "selection_index",
        "status",
        "canonical_prompt",
    }
)


def inspect_prompt_manifest(
    path: Path,
    *,
    offset: int = 0,
    limit: int = 100,
    arm: str | None = None,
    domain: str | None = None,
    lane: str | None = None,
    language: str | None = None,
    prompt_uuid: str | None = None,
) -> dict[str, Any]:
    """Validate and return a deterministic bounded view of selected/reserve prompt rows."""
    if isinstance(offset, bool) or not isinstance(offset, int) or offset < 0:
        raise ValueError("offset must be a non-negative integer")
    if isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= _MAX_LIMIT:
        raise ValueError(f"limit must be between 1 and {_MAX_LIMIT}")
    if arm is not None and arm not in _ARMS:
        raise ValueError(f"arm must be one of {_ARMS}")
    if prompt_uuid is not None and _SHA256.fullmatch(prompt_uuid) is None:
        raise ValueError("prompt UUID filter must be an exact lowercase SHA-256")
    manifest = _read_and_validate_manifest(path)
    filters = {
        "arm": arm,
        "domain": domain,
        "lane": lane,
        "language": language,
        "prompt_uuid": prompt_uuid,
    }
    rows = [
        row
        for arm_name in _ARMS
        for row in manifest["arms"][arm_name]["rows"]
        if all(value is None or row[key] == value for key, value in filters.items())
    ]
    return {
        "schema_version": 1,
        "selection_sha256": manifest["selection_sha256"],
        "artifact_sha256": manifest["artifact_sha256"],
        "filters": filters,
        "total_matches": len(rows),
        "offset": offset,
        "limit": limit,
        "rows": rows[offset : offset + limit],
    }


def _read_and_validate_manifest(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"selection manifest is unreadable: {error}") from error
    if not isinstance(payload, dict):
        raise ValueError("selection manifest must be a JSON object")
    expected_root = {
        "schema_version",
        "selection_sha256",
        "paired_cd_sha256",
        "identity",
        "arms",
        "artifact_sha256",
    }
    if set(payload) != expected_root or payload.get("schema_version") != 1:
        raise ValueError("selection manifest has an invalid version or shape")
    artifact_sha256 = payload.get("artifact_sha256")
    if not isinstance(artifact_sha256, str) or _SHA256.fullmatch(artifact_sha256) is None:
        raise ValueError("selection manifest has no valid artifact digest")
    unsigned = dict(payload)
    del unsigned["artifact_sha256"]
    if sha256_bytes(canonical_json(unsigned)) != artifact_sha256:
        raise ValueError("selection artifact digest mismatch")
    for name in ("selection_sha256", "paired_cd_sha256"):
        if not isinstance(payload[name], str) or _SHA256.fullmatch(payload[name]) is None:
            raise ValueError(f"selection manifest has an invalid {name}")
    identity = payload["identity"]
    if not isinstance(identity, dict) or set(identity) != {
        "policy_sha256",
        "seed",
        "source_inventory_sha256",
        "baseline_receipt_sha256",
        "held_out_receipt_sha256",
    }:
        raise ValueError("selection manifest identity is malformed")
    for name in (
        "policy_sha256",
        "source_inventory_sha256",
        "baseline_receipt_sha256",
        "held_out_receipt_sha256",
    ):
        if not isinstance(identity[name], str) or _SHA256.fullmatch(identity[name]) is None:
            raise ValueError(f"selection manifest identity has an invalid {name}")
    if isinstance(identity["seed"], bool) or not isinstance(identity["seed"], int):
        raise ValueError("selection manifest identity has an invalid seed")
    arms = payload["arms"]
    if not isinstance(arms, dict) or set(arms) != set(_ARMS):
        raise ValueError("selection manifest arms are malformed")
    for arm in _ARMS:
        _validate_arm(arm, arms[arm])
    return payload


def _validate_arm(arm: str, value: Any) -> None:
    expected = {
        "primary_prompt_ids",
        "reserve_prompt_ids",
        "cell_counts",
        "lane_counts",
        "bucket_floors",
        "non_agentic_bucket_floors",
        "count_proof_sha256",
        "rows",
    }
    if not isinstance(value, dict) or set(value) != expected:
        raise ValueError(f"selection manifest arm {arm} is malformed")
    rows = value["rows"]
    if not isinstance(rows, list):
        raise ValueError(f"selection manifest arm {arm} rows must be a list")
    for row in rows:
        _validate_row(arm, row)
    primary = [row for row in rows if row["status"] == "primary"]
    reserve = [row for row in rows if row["status"] == "reserve"]
    if value["primary_prompt_ids"] != [row["prompt_uuid"] for row in primary]:
        raise ValueError(f"selection manifest arm {arm} primary order is inconsistent")
    if value["reserve_prompt_ids"] != [row["prompt_uuid"] for row in reserve]:
        raise ValueError(f"selection manifest arm {arm} reserve order is inconsistent")
    if [row["selection_index"] for row in primary] != list(range(len(primary))):
        raise ValueError(f"selection manifest arm {arm} primary indexes are inconsistent")
    if [row["selection_index"] for row in reserve] != list(range(len(reserve))):
        raise ValueError(f"selection manifest arm {arm} reserve indexes are inconsistent")
    if len({row["prompt_uuid"] for row in rows}) != len(rows):
        raise ValueError(f"selection manifest arm {arm} contains duplicate prompt UUIDs")
    if (
        not isinstance(value["count_proof_sha256"], str)
        or _SHA256.fullmatch(value["count_proof_sha256"]) is None
    ):
        raise ValueError(f"selection manifest arm {arm} count proof is malformed")
    for field in ("cell_counts", "lane_counts"):
        counts = value[field]
        if not isinstance(counts, dict) or any(
            not isinstance(key, str)
            or isinstance(count, bool)
            or not isinstance(count, int)
            or count <= 0
            for key, count in counts.items()
        ):
            raise ValueError(f"selection manifest arm {arm} {field} is malformed")


def _validate_row(arm: str, row: Any) -> None:
    if not isinstance(row, dict) or set(row) != _ROW_FIELDS or row.get("arm") != arm:
        raise ValueError(f"selection manifest arm {arm} has a malformed row")
    for name in (
        "prompt_uuid",
        "source_file_sha256",
        "source_manifest_sha256",
        "candidate_rank_sha256",
    ):
        if not isinstance(row[name], str) or _SHA256.fullmatch(row[name]) is None:
            raise ValueError(f"selection manifest row has an invalid {name}")
    for name in ("source_row_index", "candidate_rank", "selection_index"):
        if isinstance(row[name], bool) or not isinstance(row[name], int) or row[name] < 0:
            raise ValueError(f"selection manifest row has an invalid {name}")
    for name in (
        "domain",
        "lane",
        "language",
        "context_bucket",
        "source_id",
        "source_revision",
        "source_file_path",
    ):
        if not isinstance(row[name], str) or not row[name]:
            raise ValueError(f"selection manifest row has an invalid {name}")
    if row["status"] not in {"primary", "reserve"}:
        raise ValueError("selection manifest row has an invalid status")
    if not isinstance(row["canonical_prompt"], dict):
        raise ValueError("selection manifest row omits canonical prompt content")
    if sha256(canonical_json(row["canonical_prompt"])).hexdigest() != row["prompt_uuid"]:
        raise ValueError("selection manifest row canonical prompt identity mismatch")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("selection_manifest", nargs="?", type=Path)
    parser.add_argument("--manifest", dest="manifest_option", type=Path)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--arm", choices=_ARMS)
    parser.add_argument("--domain")
    parser.add_argument("--lane")
    parser.add_argument("--language")
    parser.add_argument("--prompt-uuid")
    args = parser.parse_args(argv)
    if (args.selection_manifest is None) == (args.manifest_option is None):
        parser.error("provide exactly one selection manifest path")
    path = args.selection_manifest or args.manifest_option
    result = inspect_prompt_manifest(
        path,
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
