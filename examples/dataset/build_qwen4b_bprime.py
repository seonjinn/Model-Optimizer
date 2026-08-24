# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Build and immutably publish the authenticated Task5 Qwen3-4B B-prime view."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from bprime_cd_policy import load_prompt_policy
from build_specdec_inventory import (
    ExclusionReceipt,
    build_candidate_inventory_from_snapshot,
    make_exclusion_receipt,
    sha256_file,
)
from select_bprime_cd_prompts import publish_bprime_prompt_view_bundle, select_bprime_prompt_view
from stage_ptv23_sources import load_source_inventory

_PTV2_REVISION = "5c89e01dd720ae0f4058445ed49c5fb68a03c76e"


def baseline_exclusion_from_audit(path: Path) -> ExclusionReceipt:
    """Rebuild the typed historical exclusion and authenticate the audit's root."""
    if path.is_symlink() or not path.is_file():
        raise ValueError("baseline audit must be a no-follow regular file")
    try:
        payload: Any = json.loads(path.read_bytes())
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError("baseline audit is invalid") from error
    if (
        not isinstance(payload, dict)
        or payload.get("source_revision") != _PTV2_REVISION
        or payload.get("row_count") != 1_300_000
        or not isinstance(payload.get("prompt_uuids"), list)
        or any(not isinstance(value, str) for value in payload["prompt_uuids"])
    ):
        raise ValueError("baseline audit does not describe the exact historical prefix")
    receipt = make_exclusion_receipt("baseline", tuple(payload["prompt_uuids"]))
    if receipt.receipt_sha256 != payload.get("receipt_sha256"):
        raise ValueError("baseline audit receipt identity does not reconcile")
    return receipt


def held_out_exclusion_from_json(path: Path) -> ExclusionReceipt:
    """Load an exact evaluator-held-out UUID array as a typed receipt."""
    if path.is_symlink() or not path.is_file():
        raise ValueError("held-out UUID set must be a no-follow regular file")
    try:
        payload: Any = json.loads(path.read_bytes())
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError("held-out UUID set is invalid") from error
    if not isinstance(payload, list) or any(not isinstance(value, str) for value in payload):
        raise ValueError("held-out UUID set must be a JSON string array")
    return make_exclusion_receipt("held-out", tuple(payload))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-inventory", type=Path, required=True)
    parser.add_argument("--tokenizer-receipt", type=Path, required=True)
    parser.add_argument("--tokenizer-receipt-sha256", required=True)
    parser.add_argument("--baseline-audit", type=Path, required=True)
    parser.add_argument("--held-out-uuids", type=Path, required=True)
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--storage-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    source_inventory = load_source_inventory(args.source_inventory)
    baseline = baseline_exclusion_from_audit(args.baseline_audit)
    held_out = held_out_exclusion_from_json(args.held_out_uuids)
    policy = load_prompt_policy(args.policy)
    candidates = build_candidate_inventory_from_snapshot(
        source_inventory,
        tokenizer_snapshot_receipt=args.tokenizer_receipt,
        tokenizer_snapshot_receipt_sha256=args.tokenizer_receipt_sha256,
        baseline_exclusion=baseline,
        held_out_exclusion=held_out,
        training_seq_len=policy.sequence_length,
        storage_dir=args.storage_dir,
    )
    bundle = None
    try:
        bundle = select_bprime_prompt_view(
            candidates,
            policy,
            source_inventory=source_inventory,
            tokenizer_snapshot_receipt=args.tokenizer_receipt,
            tokenizer_snapshot_receipt_sha256=args.tokenizer_receipt_sha256,
            baseline_receipt_sha256=baseline.receipt_sha256,
            held_out_receipt_sha256=held_out.receipt_sha256,
        )
        published = publish_bprime_prompt_view_bundle(bundle, args.output_dir)
    finally:
        if bundle is not None:
            bundle.close()
        candidates.close()
    manifest_sha256 = sha256_file(published.manifest_path)
    print(
        json.dumps(
            {
                "manifest": str(published.manifest_path),
                "manifest_sha256": manifest_sha256,
                "root_sha256": published.root_sha256,
                "row_count": published.row_count,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
