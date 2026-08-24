# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Build and immutably publish the authenticated Task5 Qwen3-4B B-prime view."""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass, fields
from pathlib import Path
from types import MappingProxyType
from typing import Any

from audit_ptv2_baseline import EXPECTED_BASELINE, BaselineAudit, BaselineExpectation
from bprime_cd_policy import load_prompt_policy
from build_specdec_inventory import (
    ExclusionReceipt,
    build_candidate_inventory_from_snapshot,
    make_exclusion_receipt,
    sha256_file,
)
from select_bprime_cd_prompts import publish_bprime_prompt_view_bundle, select_bprime_prompt_view
from specdec_corpus_contracts import canonical_json, sha256_bytes, sha256_canonical_json
from stage_ptv23_sources import load_source_inventory

_PTV2_REVISION = "5c89e01dd720ae0f4058445ed49c5fb68a03c76e"


@dataclass(frozen=True)
class AuthenticatedBaselineAudit:
    """One stable whole-audit receipt and its independently derived exclusion set."""

    payload: MappingProxyType[str, Any]
    receipt_sha256: str
    exclusion: ExclusionReceipt


def baseline_exclusion_from_audit(
    path: Path, *, expected: BaselineExpectation = EXPECTED_BASELINE
) -> ExclusionReceipt:
    """Rebuild the typed historical exclusion and authenticate the audit's root."""
    return authenticate_baseline_audit(path, expected=expected).exclusion


def authenticate_baseline_audit(
    path: Path, *, expected: BaselineExpectation = EXPECTED_BASELINE
) -> AuthenticatedBaselineAudit:
    """Authenticate one stable producer receipt without conflating its exclusion root."""
    if path.is_symlink() or not path.is_file():
        raise ValueError("baseline audit must be a no-follow regular file")
    try:
        before = os.lstat(path)
        raw = path.read_bytes()
        after = os.lstat(path)
        payload: Any = json.loads(raw)
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError("baseline audit is invalid") from error
    if (
        not isinstance(payload, dict)
        or (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
        != (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
        or raw != canonical_json(payload) + b"\n"
    ):
        raise ValueError("baseline audit is not a stable canonical receipt")
    receipt_sha256 = payload.get("receipt_sha256")
    body = {key: value for key, value in payload.items() if key != "receipt_sha256"}
    if (
        not isinstance(receipt_sha256, str)
        or receipt_sha256 != sha256_bytes(canonical_json(body))
    ):
        raise ValueError("baseline audit whole-receipt identity does not reconcile")
    genuine_keys = {field.name for field in fields(BaselineAudit)} | {"receipt_sha256"}
    if set(payload) == genuine_keys:
        prompt_ids = _validate_genuine_audit(payload, expected)
    else:
        prompt_ids = _validate_legacy_audit(payload, expected)
    exclusion = make_exclusion_receipt("baseline", tuple(prompt_ids))
    return AuthenticatedBaselineAudit(
        MappingProxyType(payload), receipt_sha256, exclusion
    )


def _validate_genuine_audit(payload: dict[str, Any], expected: BaselineExpectation) -> list[str]:
    prompt_ids = payload.get("exclusion_prompt_ids")
    occurrence_ids = payload.get("occurrence_prompt_ids")
    if (
        not isinstance(prompt_ids, list)
        or not isinstance(occurrence_ids, list)
        or any(not isinstance(value, str) for value in (*prompt_ids, *occurrence_ids))
        or prompt_ids != sorted(set(prompt_ids))
        or len(prompt_ids) != payload.get("unique_prompt_count")
        or sha256_canonical_json(tuple(prompt_ids)) != payload.get("exclusion_prompt_ids_sha256")
        or len(occurrence_ids) != payload.get("occurrence_count")
        or sha256_canonical_json(tuple(occurrence_ids))
        != payload.get("occurrence_prompt_ids_sha256")
    ):
        raise ValueError("baseline audit prompt identities do not reconcile")
    _validate_common_audit_fields(payload, expected, len(prompt_ids))
    return prompt_ids


def _validate_legacy_audit(payload: dict[str, Any], expected: BaselineExpectation) -> list[str]:
    required = {
        "duplicate_uuid_multiplicity",
        "files",
        "physical_row_count",
        "prompt_uuids",
        "receipt_sha256",
        "row_count",
        "selection_boundary",
        "selection_policy",
        "source_revision",
        "split_rows",
    }
    prompt_ids = payload.get("prompt_uuids")
    if (
        set(payload) != required
        or not isinstance(prompt_ids, list)
        or any(not isinstance(value, str) for value in prompt_ids)
        or prompt_ids != sorted(set(prompt_ids))
    ):
        raise ValueError("legacy baseline audit prompt identities do not reconcile")
    _validate_common_audit_fields(payload, expected, len(prompt_ids))
    return prompt_ids


def _validate_common_audit_fields(
    payload: dict[str, Any], expected: BaselineExpectation, unique_prompt_count: int
) -> None:
    boundary = payload.get("selection_boundary")
    files = payload.get("files")
    expected_rows = sum(expected.split_rows.values())
    if (
        payload.get("source_revision") != expected.source_revision
        or payload.get("row_count") != expected_rows
        or payload.get("split_rows") != expected.split_rows
        or not isinstance(files, list)
        or len(files) != expected.shard_count
        or unique_prompt_count != expected.unique_prompt_count
        or payload.get("physical_row_count") != expected.physical_row_count
        or payload.get("selection_policy") != "hf-streaming-sorted-parquet-take"
        or not isinstance(boundary, dict)
        or boundary.get("excluded_tail_rows") != expected.excluded_tail_rows
        or not isinstance(boundary.get("file"), str)
        or expected.excluded_tail_split not in boundary["file"]
    ):
        raise ValueError("baseline audit does not match the expected historical prefix")


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
