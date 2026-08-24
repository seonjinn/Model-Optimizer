# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Compare one authenticated real shard through serial and process candidate adapters."""

from __future__ import annotations

import argparse
import json
import os
import sqlite3
from collections import Counter
from pathlib import Path

import build_specdec_inventory as inventory_module
from build_qwen4b_bprime import baseline_exclusion_from_audit, held_out_exclusion_from_json
from build_specdec_inventory import (
    _CandidateShardTask,
    _authenticated_snapshot_tokenizer,
    _process_candidate_shard,
    build_candidate_inventory,
    sha256_file,
)
from specdec_corpus_contracts import canonical_json, sha256_bytes
from stage_ptv23_sources import SourceInventory, load_source_inventory


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-inventory", type=Path, required=True)
    parser.add_argument("--tokenizer-receipt", type=Path, required=True)
    parser.add_argument("--tokenizer-receipt-sha256", required=True)
    parser.add_argument("--baseline-audit", type=Path, required=True)
    parser.add_argument("--held-out-uuids", type=Path, required=True)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--scratch-root", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args()

    inventory = load_source_inventory(args.source_inventory)
    files = inventory_module._verified_candidate_files(inventory)
    if not 0 <= args.shard_index < len(files):
        raise ValueError("diagnostic shard index is out of bounds")
    source, descriptor, path = files[args.shard_index]
    snapshot, tokenizer = _authenticated_snapshot_tokenizer(
        args.tokenizer_receipt, args.tokenizer_receipt_sha256
    )
    baseline = baseline_exclusion_from_audit(args.baseline_audit)
    held_out = held_out_exclusion_from_json(args.held_out_uuids)
    diagnostic_manifest = b"qwen4b-real-shard-diagnostic-v1"
    mini = SourceInventory(
        schema_version=1,
        name="qwen4b-real-shard-diagnostic",
        sources=(source.__class__(**{**source.__dict__, "files": (descriptor,)}),),
        manifest_sha256=sha256_bytes(diagnostic_manifest),
        canonical_manifest=diagnostic_manifest,
        raw_counts={source.cell: 0},
        staged_root=inventory.staged_root,
    )
    serial = build_candidate_inventory(
        mini,
        tokenizer=tokenizer,
        tokenizer_sha256=snapshot.tokenizer_sha256,
        chat_template_sha256=snapshot.chat_template_sha256,
        baseline_exclusion=baseline,
        held_out_exclusion=held_out,
        storage_dir=args.scratch_root,
    )
    try:
        serial_summary = {
            "accepted_count": len(serial.rows),
            "quarantine_counts": dict(serial.quarantine_counts),
            "inventory_sha256": serial.inventory_sha256,
        }
    finally:
        serial.close()

    args.scratch_root.mkdir(parents=True, exist_ok=True)
    spool = args.scratch_root / "process-shard.sqlite3"
    inventory_module._PROCESS_TOKENIZER = tokenizer
    try:
        result = _process_candidate_shard(
            _CandidateShardTask(
                args.shard_index,
                source,
                descriptor,
                path,
                spool,
                mini.manifest_sha256,
                snapshot.tokenizer_sha256,
                snapshot.chat_template_sha256,
                4_096,
            )
        )
    finally:
        inventory_module._PROCESS_TOKENIZER = None
    historical = set(baseline.prompt_ids)
    heldout = set(held_out.prompt_ids)
    reasons: Counter[str] = Counter()
    accepted = 0
    first_payload: dict[str, object] | None = None
    with sqlite3.connect(spool) as connection:
        for (raw,) in connection.execute("SELECT payload FROM records ORDER BY source_row_index"):
            payload = json.loads(raw)
            prompt_id = payload.get("prompt_uuid")
            if prompt_id in historical:
                reasons["historical_exclusion"] += 1
            elif prompt_id in heldout:
                reasons["heldout_exclusion"] += 1
            elif payload.get("reason") is not None:
                reasons[str(payload["reason"])] += 1
            else:
                accepted += 1
                if first_payload is None:
                    first_payload = payload["candidate"]
    process_summary = {
        "accepted_count": accepted,
        "quarantine_counts": dict(sorted(reasons.items())),
        "row_count": result.row_count,
        "spool_sha256": result.spool_sha256,
        "first_candidate_prompt_uuid": None if first_payload is None else first_payload["prompt_uuid"],
    }
    payload = {
        "schema_version": 1,
        "source_commit": os.environ.get("SOURCE_COMMIT"),
        "source_inventory_sha256": inventory.manifest_sha256,
        "shard_index": args.shard_index,
        "source_file_path": descriptor.path,
        "source_file_sha256": descriptor.sha256,
        "serial": serial_summary,
        "process": process_summary,
        "counts_match": serial_summary["accepted_count"] == process_summary["accepted_count"]
        and serial_summary["quarantine_counts"] == process_summary["quarantine_counts"],
    }
    payload["receipt_sha256"] = sha256_bytes(canonical_json(payload))
    args.receipt.parent.mkdir(parents=True, exist_ok=True)
    with args.receipt.open("xb") as stream:
        stream.write(canonical_json(payload) + b"\n")
        stream.flush()
        os.fsync(stream.fileno())
    directory = os.open(args.receipt.parent, os.O_RDONLY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)
    print(json.dumps(payload, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
