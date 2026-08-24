# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Build and publish the authenticated Task9 Qwen3-4B A-repair selection."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from hashlib import sha256
from pathlib import Path
from typing import TYPE_CHECKING

from audit_ptv2_baseline import (
    EXPECTED_BASELINE,
    BaselineAudit,
    BaselineExpectation,
    SelectionBoundary,
)
from build_qwen4b_bprime import (
    AuthenticatedBaselineAudit,
    authenticate_baseline_audit,
    held_out_exclusion_from_json,
)
from build_specdec_inventory import sha256_file
from promote_synthesis_reserve import load_prompt_view
from qwen3_4b_ptv2_study import (
    _authenticate_task5_bprime,
    iter_ptv2_staged_source_rows,
    load_ptv2_study_policy,
    select_authenticated_ptv2_study_views,
    write_ptv2_selection_receipt,
)
from specdec_corpus_contracts import SourceFile, sha256_canonical_json
from specdec_identity import ExclusionIndex
from stage_ptv23_sources import load_source_inventory

if TYPE_CHECKING:
    from collections.abc import Iterable

    from qwen3_4b_ptv2_study import PTV2StudySourceRow


def reconstruct_baseline_sequence(
    rows: Iterable[PTV2StudySourceRow],
    authenticated: AuthenticatedBaselineAudit,
    *,
    expected: BaselineExpectation = EXPECTED_BASELINE,
) -> BaselineAudit:
    """Bind a legacy unique-set receipt to the exact authenticated source occurrence order."""
    target = sum(expected.split_rows.values())
    occurrence_ids: list[str] = []
    split_counts: Counter[str] = Counter()
    for index, row in enumerate(rows):
        if index == target:
            break
        occurrence_ids.append(row.prompt_uuid)
        split = f"multilingual_{row.language}" if row.cell == "multilingual" else row.cell
        split_counts[split] += 1
    if len(occurrence_ids) != target:
        raise ValueError("baseline source is shorter than the exact historical segment")
    if dict(sorted(split_counts.items())) != expected.split_rows:
        raise ValueError("baseline source split counts do not match the authenticated audit")
    counts = Counter(occurrence_ids)
    exclusion_ids = tuple(sorted(counts))
    if exclusion_ids != authenticated.exclusion.prompt_ids:
        raise ValueError("baseline source exclusion set does not match the authenticated audit")
    duplicates = {key: value for key, value in sorted(counts.items()) if value > 1}
    payload = authenticated.payload
    if duplicates != payload.get("duplicate_uuid_multiplicity"):
        raise ValueError("baseline source duplicate multiplicity does not match the audit")
    file_records = payload.get("files")
    boundary_record = payload.get("selection_boundary")
    if not isinstance(file_records, list) or not isinstance(boundary_record, dict):
        raise ValueError("baseline audit topology is missing")
    files = tuple(SourceFile(str(item["path"]), int(item["bytes"]), str(item["sha256"])) for item in file_records)
    boundary = SelectionBoundary(
        str(boundary_record["file"]),
        int(boundary_record["rows_selected"]),
        int(boundary_record["rows_available"]),
        int(boundary_record["excluded_tail_rows"]),
    )
    occurrence_tuple = tuple(occurrence_ids)
    return BaselineAudit(
        source_revision=expected.source_revision,
        source_manifest_sha256=None,
        row_count=target,
        split_rows=dict(sorted(split_counts.items())),
        files=files,
        occurrence_count=target,
        unique_prompt_count=len(exclusion_ids),
        occurrence_prompt_ids=occurrence_tuple,
        occurrence_prompt_ids_sha256=sha256_canonical_json(occurrence_tuple),
        exclusion_prompt_ids=exclusion_ids,
        exclusion_prompt_ids_sha256=authenticated.exclusion.prompt_ids_sha256,
        duplicate_uuid_multiplicity=duplicates,
        physical_row_count=int(payload["physical_row_count"]),
        selection_policy=str(payload["selection_policy"]),
        selection_boundary=boundary,
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-inventory", type=Path, required=True)
    parser.add_argument("--baseline-audit", type=Path, required=True)
    parser.add_argument("--held-out-uuids", type=Path, required=True)
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--task5-manifest", type=Path, required=True)
    parser.add_argument("--task5-manifest-sha256", required=True)
    parser.add_argument("--work-root", type=Path, required=True)
    parser.add_argument("--selection-receipt-root", type=Path, required=True)
    args = parser.parse_args()

    policy = load_ptv2_study_policy(args.policy)
    inventory = load_source_inventory(args.source_inventory)
    authenticated = authenticate_baseline_audit(args.baseline_audit)
    held_out = held_out_exclusion_from_json(args.held_out_uuids)
    baseline = reconstruct_baseline_sequence(
        iter_ptv2_staged_source_rows(args.source_inventory, policy=policy),
        authenticated,
    )
    if sha256_file(args.task5_manifest) != args.task5_manifest_sha256:
        raise ValueError("caller-pinned Task5 manifest identity mismatch")
    task5_view = load_prompt_view(
        args.task5_manifest,
        expected_manifest_sha256=args.task5_manifest_sha256,
        arm="B-prime",
    )
    complement_sha256 = _authenticate_task5_bprime(
        args.task5_manifest,
        expected_manifest_sha256=args.task5_manifest_sha256,
        view=task5_view,
        policy=policy,
        source_manifest_sha256=inventory.manifest_sha256,
    )
    bundle = select_authenticated_ptv2_study_views(
        args.source_inventory,
        policy=policy,
        baseline=baseline,
        exclusions=ExclusionIndex(held_out=set(held_out.prompt_ids)),
        baseline_receipt=authenticated.exclusion,
        held_out_receipt=held_out,
        task5_manifest=args.task5_manifest,
        task5_manifest_sha256=args.task5_manifest_sha256,
        task5_arm="B-prime",
        output_root=args.work_root,
    )
    receipt = write_ptv2_selection_receipt(
        args.selection_receipt_root,
        bundle.a_repair,
        policy=policy,
        policy_path=args.policy,
        source_inventory_sha256=inventory.manifest_sha256,
        held_out_receipt_sha256=held_out.receipt_sha256,
        baseline_receipt_sha256=authenticated.exclusion.receipt_sha256,
        complement_selection_sha256=complement_sha256,
    )
    print(
        json.dumps(
            {
                "receipt": str(receipt),
                "receipt_sha256": sha256(receipt.read_bytes()).hexdigest(),
                "selection_sha256": bundle.a_repair.selection_sha256,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
