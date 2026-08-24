# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Compare one authenticated real shard through serial and process candidate adapters."""

from __future__ import annotations

import argparse
import json
import os
import sqlite3
from pathlib import Path

import build_specdec_inventory as inventory_module
from build_qwen4b_bprime import baseline_exclusion_from_audit, held_out_exclusion_from_json
from build_specdec_inventory import (
    _authenticated_snapshot_tokenizer,
    _CandidateShardTask,
    _CandidateTokenizeTask,
    _create_candidate_database,
    _exclude_or_quarantine_classified_candidate,
    _merge_tokenized_candidate,
    _process_candidate_shard,
    _tokenize_candidate_shard,
    build_candidate_inventory,
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
    merge_database = args.scratch_root / "process-candidates.sqlite3"
    inventory_module._PROCESS_TOKENIZER = tokenizer
    try:
        result = _process_candidate_shard(
            _CandidateShardTask(
                args.shard_index,
                source,
                descriptor,
                path,
                args.scratch_root / f"staged{path.suffix}",
                spool,
                mini.manifest_sha256,
                snapshot.tokenizer_sha256,
                snapshot.chat_template_sha256,
                4_096,
            )
        )
        historical = set(baseline.prompt_ids)
        heldout = set(held_out.prompt_ids)
        reasons: dict[str, int] = {}
        with sqlite3.connect(spool) as connection:
            connection.execute(
                "CREATE TABLE selected(source_row_index INTEGER PRIMARY KEY,payload BLOB NOT NULL) "
                "WITHOUT ROWID"
            )
            for source_row_index, raw in connection.execute(
                "SELECT source_row_index,payload FROM records ORDER BY source_row_index"
            ):
                payload = json.loads(raw)
                if _exclude_or_quarantine_classified_candidate(
                    payload,
                    historical_prompt_ids=historical,
                    held_out_prompt_ids=heldout,
                    quarantine_counts=reasons,
                ):
                    continue
                connection.execute("INSERT INTO selected VALUES(?,?)", (source_row_index, raw))
            connection.commit()
        tokenization_result = _tokenize_candidate_shard(
            _CandidateTokenizeTask(args.shard_index, spool, source)
        )
    finally:
        inventory_module._PROCESS_TOKENIZER = None
    capacity = {}
    merged = _create_candidate_database(merge_database)
    accepted = 0
    first_prompt_uuid: str | None = None
    try:
        with sqlite3.connect(spool) as connection:
            for (raw,) in connection.execute(
                "SELECT payload FROM tokenized ORDER BY source_row_index"
            ):
                payload = json.loads(raw)
                if _merge_tokenized_candidate(
                    merged,
                    payload,
                    capacity=capacity,
                    quarantine_counts=reasons,
                ):
                    accepted += 1
                    if first_prompt_uuid is None:
                        first_prompt_uuid = str(payload["candidate"]["prompt_uuid"])
        merged.commit()
    finally:
        merged.close()
    process_summary = {
        "accepted_count": accepted,
        "quarantine_counts": dict(sorted(reasons.items())),
        "phase1_row_count": result.row_count,
        "phase2_row_count": tokenization_result.row_count,
        "phase1_spool_sha256": result.spool_sha256,
        "final_spool_sha256": inventory_module.sha256_file(spool),
        "first_candidate_prompt_uuid": first_prompt_uuid,
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
