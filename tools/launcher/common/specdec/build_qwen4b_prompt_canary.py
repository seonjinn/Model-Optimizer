# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Build small, immutable PTV3 target-synthesis and trace-replay canary lanes."""

from __future__ import annotations

import argparse
import hashlib
import heapq
import json
import os
from collections import Counter
from pathlib import Path
from typing import Any


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _decode_raw_row(row: dict[str, Any]) -> dict[str, Any]:
    if set(row) == {"raw_json"}:
        decoded = json.loads(row["raw_json"])
        if not isinstance(decoded, dict):
            raise ValueError("raw_json must encode an object")
        return decoded
    return row


def _messages(row: dict[str, Any]) -> list[dict[str, Any]]:
    value = row.get("messages") or row.get("conversations")
    if not isinstance(value, list) or not value:
        raise ValueError("row has no messages")
    if any(not isinstance(message, dict) for message in value):
        raise ValueError("row has a non-object message")
    return value


def _has_tool_trajectory(row: dict[str, Any]) -> bool:
    return bool(row.get("tools")) or any(
        message.get("role") == "tool" or bool(message.get("tool_calls"))
        for message in _messages(row)
    )


def _prompt_view(row: dict[str, Any]) -> dict[str, Any]:
    if _has_tool_trajectory(row):
        raise ValueError("tool trajectory cannot enter target synthesis")
    messages = [
        message
        for message in _messages(row)
        if message.get("role") in {"system", "developer", "user"}
    ]
    if not messages or not any(message.get("role") == "user" for message in messages):
        raise ValueError("target-synthesis prompt has no user turn")
    return {"messages": messages}


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True).encode()


def _prompt_identity(row: dict[str, Any]) -> str:
    return hashlib.sha256(_canonical_bytes(row)).hexdigest()


def _context_bucket(token_count: int) -> str:
    if token_count <= 4096:
        return "le4k"
    if token_count <= 16384:
        return "4k_16k"
    if token_count <= 32768:
        return "16k_32k"
    raise ValueError(f"conversation exceeds the 32K canary limit: {token_count}")


def _iter_parquet_rows(path: Path):
    import pyarrow.parquet as pq

    parquet = pq.ParquetFile(path)
    if parquet.schema_arrow.names != ["raw_json"]:
        raise ValueError(f"staged shard lacks explicit raw_json union schema: {path.name}")
    for batch in parquet.iter_batches(batch_size=4096, columns=["raw_json"]):
        for value in batch.column(0).to_pylist():
            yield _decode_raw_row({"raw_json": value})


def _bounded_candidates(path: Path, limit: int) -> list[dict[str, Any]]:
    heap: list[tuple[int, int, dict[str, Any]]] = []
    for serial, row in enumerate(_iter_parquet_rows(path)):
        score = int.from_bytes(hashlib.sha256(_canonical_bytes(row)).digest(), "big")
        entry = (-score, serial, row)
        if len(heap) < limit:
            heapq.heappush(heap, entry)
        elif entry > heap[0]:
            heapq.heapreplace(heap, entry)
    return [entry[2] for entry in sorted(heap, key=lambda value: -value[0])]


def _tokenize_with_assistant_mask(
    tokenizer: Any, row: dict[str, Any]
) -> tuple[list[int], list[int]]:
    encoded = tokenizer.apply_chat_template(
        _messages(row),
        tools=row.get("tools") or None,
        tokenize=True,
        add_generation_prompt=False,
        return_dict=True,
        return_assistant_tokens_mask=True,
    )
    input_ids = encoded.get("input_ids")
    loss_mask = encoded.get("assistant_masks", encoded.get("assistant_mask"))
    if (
        not isinstance(input_ids, list)
        or not isinstance(loss_mask, list)
        or len(input_ids) != len(loss_mask)
        or any(value not in (0, 1) for value in loss_mask)
    ):
        raise ValueError("tokenizer did not return aligned IDs and assistant mask")
    return input_ids, loss_mask


def _select_rows(
    records: list[dict[str, Any]],
    *,
    root: Path,
    tokenizer: Any,
    quota: int,
    response_source: str,
) -> tuple[list[dict[str, Any]], Counter[str], Counter[str], Counter[str]]:
    selected: list[dict[str, Any]] = []
    categories: Counter[str] = Counter()
    buckets: Counter[str] = Counter()
    exclusions: Counter[str] = Counter()
    for record in records:
        candidates = _bounded_candidates(root / record["path"], max(quota * 8, 64))
        accepted = 0
        for raw in candidates:
            try:
                if response_source == "target-synth":
                    row = _prompt_view(raw)
                else:
                    if not _has_tool_trajectory(raw):
                        raise ValueError("trace-replay row has no tool trajectory")
                    row = dict(raw)
                input_ids, loss_mask = _tokenize_with_assistant_mask(tokenizer, row)
                token_count = len(input_ids)
                bucket = _context_bucket(token_count)
                if response_source == "trace-replay" and sum(loss_mask) < 1:
                    raise ValueError("trace-replay row has no assistant loss tokens")
            except (KeyError, TypeError, ValueError) as error:
                exclusions[str(error)] += 1
                continue
            prompt_id = _prompt_identity(row)
            row["prompt_id"] = prompt_id
            row["_canary_provenance"] = {
                "category": record["category"],
                "context_bucket": bucket,
                "full_context_tokens": token_count,
                "response_source": response_source,
                "source_declared_response_source": record["response_source"],
                "source_id": record["source_id"],
                "source_revision": record["source_revision"],
                "source_file_sha256": record["sha256"],
                "full_assistant_tokens": sum(loss_mask),
            }
            if response_source == "trace-replay":
                row["_tokenized_trace"] = {"input_ids": input_ids, "loss_mask": loss_mask}
            selected.append(row)
            categories[record["category"]] += 1
            buckets[bucket] += 1
            accepted += 1
            if accepted == quota:
                break
        if accepted and accepted != quota:
            raise ValueError(
                f"{record['category']} yielded {accepted} valid {response_source} rows, need {quota}"
            )
    if not selected:
        raise ValueError(f"no rows selected for {response_source}")
    expected_categories = (
        {"swe", "code", "math", "science", "chat", "multilingual"}
        if response_source == "target-synth"
        else {"agentic_tool"}
    )
    if set(categories) != expected_categories or any(
        categories[category] != quota for category in expected_categories
    ):
        raise ValueError(
            f"{response_source} category coverage mismatch: {dict(sorted(categories.items()))}"
        )
    return selected, categories, buckets, exclusions


def _write_lane(
    destination: Path,
    rows: list[dict[str, Any]],
    *,
    lane: str,
    source_manifest_sha256: str,
    tokenizer_sha256: str,
    categories: Counter[str],
    buckets: Counter[str],
    exclusions: Counter[str],
) -> dict[str, Any]:
    destination.mkdir(parents=True, exist_ok=False)
    shard = destination / "canary.jsonl"
    with shard.open("w", encoding="utf-8") as output:
        for row in rows:
            output.write(
                json.dumps(
                    {"raw_json": json.dumps(row, ensure_ascii=False, sort_keys=True)},
                    sort_keys=True,
                )
                + "\n"
            )
        output.flush()
        os.fsync(output.fileno())
    manifest = {
        "schema_version": 1,
        "name": f"qwen3-4b-ptv3-{lane}-canary-v1",
        "format": "json",
        "compression": "none",
        "union_schema": {"raw_json": "string"},
        "row_count": len(rows),
        "file_count": 1,
        "source_manifest_sha256": source_manifest_sha256,
        "tokenizer_sha256": tokenizer_sha256,
        "category_counts": dict(sorted(categories.items())),
        "context_bucket_counts": dict(sorted(buckets.items())),
        "context_bucket_basis": "exact target tokenizer over full selected conversation",
        "excluded_candidate_counts_by_reason": dict(sorted(exclusions.items())),
        "selected_full_context_tokens": sum(
            row["_canary_provenance"]["full_context_tokens"] for row in rows
        ),
        "selected_assistant_tokens": sum(
            row["_canary_provenance"]["full_assistant_tokens"] for row in rows
        ),
        "selected_prompt_ids_sha256": hashlib.sha256(
            "\n".join(sorted(row["prompt_id"] for row in rows)).encode()
        ).hexdigest(),
        "files": [
            {"path": shard.name, "bytes": shard.stat().st_size, "sha256": _sha256_file(shard)}
        ],
    }
    temporary = destination / f".MANIFEST.json.tmp-{os.getpid()}"
    temporary.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, destination / "MANIFEST.json")
    return manifest


def main() -> int:
    """Build and atomically publish both canary lanes."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--source-manifest-sha256", required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--tokenizer-sha256", required=True)
    parser.add_argument("--rows-per-category", type=int, default=8)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    if args.rows_per_category < 1:
        raise ValueError("rows-per-category must be positive")
    raw = args.source_manifest.read_bytes()
    if hashlib.sha256(raw).hexdigest() != args.source_manifest_sha256:
        raise ValueError("source manifest SHA-256 mismatch")
    manifest = json.loads(raw)
    if manifest.get("schema_version") != 1 or not isinstance(manifest.get("files"), list):
        raise ValueError("invalid source manifest")

    from common.specdec.build_qwen4b_synthesis_manifest import tokenizer_snapshot_sha256
    from transformers import AutoTokenizer

    if tokenizer_snapshot_sha256(args.tokenizer) != args.tokenizer_sha256:
        raise ValueError("tokenizer SHA-256 mismatch")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True)
    results = {}
    for lane in ("target-synth", "trace-replay"):
        results[lane] = _select_rows(
            manifest["files"],
            root=args.source_manifest.parent,
            tokenizer=tokenizer,
            quota=args.rows_per_category,
            response_source=lane,
        )

    partial = args.output_root.with_name(f".{args.output_root.name}.partial-{os.getpid()}")
    if args.output_root.exists() or partial.exists():
        raise ValueError("output root or partial already exists")
    partial.mkdir(parents=True)
    try:
        lane_manifests = {}
        lane_prompt_ids = {}
        for lane, (rows, categories, buckets, exclusions) in results.items():
            lane_manifests[lane] = _write_lane(
                partial / lane,
                rows,
                lane=lane,
                source_manifest_sha256=args.source_manifest_sha256,
                tokenizer_sha256=args.tokenizer_sha256,
                categories=categories,
                buckets=buckets,
                exclusions=exclusions,
            )
            lane_prompt_ids[lane] = {row["prompt_id"] for row in rows}
        overlap = lane_prompt_ids["target-synth"] & lane_prompt_ids["trace-replay"]
        if overlap:
            raise ValueError("selected prompt IDs appear in more than one response lane")
        receipt = {
            "schema_version": 1,
            "source_manifest_sha256": args.source_manifest_sha256,
            "tokenizer_sha256": args.tokenizer_sha256,
            "selected_rows": {
                lane: manifest["row_count"] for lane, manifest in lane_manifests.items()
            },
            "selected_full_context_tokens": {
                lane: manifest["selected_full_context_tokens"]
                for lane, manifest in lane_manifests.items()
            },
            "selected_assistant_tokens": {
                lane: manifest["selected_assistant_tokens"]
                for lane, manifest in lane_manifests.items()
            },
            "prompt_id_overlap_count": 0,
            "lane_manifest_sha256": {
                lane: _sha256_file(partial / lane / "MANIFEST.json") for lane in lane_manifests
            },
        }
        (partial / "SELECTION_RECEIPT.json").write_text(
            json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        os.replace(partial, args.output_root)
    except BaseException:
        raise
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
