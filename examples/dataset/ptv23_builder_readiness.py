# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Prove whether the pinned PTV2/PTV3 inputs can satisfy B/C/D without renormalizing."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any

from specdec_corpus_contracts import canonical_json, sha256_bytes

_B_REQUIRED_DOMAINS = ("math", "code", "stem", "chat", "multilingual")
_CD_TARGET_CATEGORIES = frozenset({"swe", "math", "code", "science", "chat", "multilingual"})


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_bytes())
    if not isinstance(value, dict):
        raise ValueError(f"JSON input must be an object: {path}")
    return value


def _verify_audit(path: Path, expected_sha256: str, revision: str) -> dict[str, Any]:
    receipt = _read_object(path)
    embedded = receipt.pop("receipt_sha256", None)
    actual = sha256_bytes(canonical_json(receipt))
    if actual != expected_sha256 or embedded != expected_sha256:
        raise ValueError("audit receipt SHA-256 mismatch")
    if receipt.get("source_revision") != revision:
        raise ValueError("audit and requested PTV2 revisions differ")
    if receipt.get("selection_policy") != "hf-streaming-sorted-parquet-take":
        raise ValueError("unsupported prior selection policy")
    return receipt


def _domain_files(root: Path, domain: str) -> list[Path]:
    if domain == "multilingual":
        pattern = re.compile(r"multilingual_(?:ja|es|fr|it)-\d+(?:-of-\d+)?\.parquet$")
    else:
        pattern = re.compile(rf"{re.escape(domain)}-\d+(?:-of-\d+)?\.parquet$")
    return sorted(path for path in root.glob("*.parquet") if pattern.fullmatch(path.name))


def _ptv2_exhaustion_proof(
    audit: dict[str, Any], root: Path
) -> tuple[dict[str, int], dict[str, list[dict[str, Any]]]]:
    files = audit.get("files")
    boundary = audit.get("selection_boundary")
    if not isinstance(files, list) or not isinstance(boundary, dict):
        raise ValueError("audit receipt lacks files or selection boundary")
    boundary_name = boundary.get("file")
    ordered_names = [Path(str(record["path"])).name for record in files]
    if boundary_name not in ordered_names:
        raise ValueError("audit selection boundary is absent from its file inventory")
    boundary_index = ordered_names.index(str(boundary_name))
    audited_by_name = {Path(str(record["path"])).name: record for record in files}
    upper_bounds: dict[str, int] = {}
    evidence: dict[str, list[dict[str, Any]]] = {}
    for domain in _B_REQUIRED_DOMAINS:
        source_files = _domain_files(root, domain)
        domain_evidence = []
        fully_selected = bool(source_files)
        for source in source_files:
            record = audited_by_name.get(source.name)
            digest = _sha256_file(source)
            matched = (
                record is not None
                and Path(str(record["path"])).resolve() == source.resolve()
                and int(record["bytes"]) == source.stat().st_size
                and record["sha256"] == digest
            )
            position = ordered_names.index(source.name) if source.name in ordered_names else None
            selected_in_full = position is not None and (
                position < boundary_index
                or (position == boundary_index and int(boundary.get("excluded_tail_rows", -1)) == 0)
            )
            fully_selected = fully_selected and matched and selected_in_full
            domain_evidence.append(
                {
                    "path": str(source.resolve()),
                    "bytes": source.stat().st_size,
                    "sha256": digest,
                    "audited_identity_match": matched,
                    "fully_selected_by_prior": selected_in_full,
                }
            )
        evidence[domain] = domain_evidence
        if fully_selected:
            upper_bounds[domain] = 0
    return dict(sorted(upper_bounds.items())), evidence


def _verify_ptv3_inputs(prompt_root: Path, source_manifest_path: Path) -> dict[str, Any]:
    source_sha256 = _sha256_file(source_manifest_path)
    source = _read_object(source_manifest_path)
    source_files = source.get("files")
    if source.get("schema_version") != 1 or not isinstance(source_files, list):
        raise ValueError("invalid PTV3 source manifest")
    selection = _read_object(prompt_root / "SELECTION_RECEIPT.json")
    target_path = prompt_root / "target-synth" / "MANIFEST.json"
    trace_path = prompt_root / "trace-replay" / "MANIFEST.json"
    target = _read_object(target_path)
    trace = _read_object(trace_path)
    expected_lanes = selection.get("lane_manifest_sha256")
    if not isinstance(expected_lanes, dict):
        raise ValueError("PTV3 selection receipt lacks lane identities")
    if expected_lanes.get("target-synth") != _sha256_file(target_path):
        raise ValueError("target-synth manifest SHA-256 mismatch")
    if expected_lanes.get("trace-replay") != _sha256_file(trace_path):
        raise ValueError("trace-replay manifest SHA-256 mismatch")
    for value in (selection, target, trace):
        if value.get("source_manifest_sha256") != source_sha256:
            raise ValueError("PTV3 source manifest identity mismatch")
    target_categories = target.get("category_counts")
    if not isinstance(target_categories, dict) or not _CD_TARGET_CATEGORIES.issubset(
        target_categories
    ):
        raise ValueError("target-synth manifest lacks required C/D categories")
    interactive_sources = [
        record
        for record in source_files
        if record.get("category") == "swe"
        and (
            record.get("lane") == "interactive-swe-replay"
            or record.get("tool_lane") == "interactive-swe-replay"
        )
    ]
    generic_sources = [
        record
        for record in source_files
        if record.get("category") == "agentic_tool" and record.get("tool_lane") == "recorded-trace"
    ]
    return {
        "source_manifest_path": str(source_manifest_path.resolve()),
        "source_manifest_sha256": source_sha256,
        "selection_receipt_path": str((prompt_root / "SELECTION_RECEIPT.json").resolve()),
        "selection_receipt_sha256": _sha256_file(prompt_root / "SELECTION_RECEIPT.json"),
        "target_synth_manifest_sha256": _sha256_file(target_path),
        "target_synth_assistant_tokens": int(target.get("selected_assistant_tokens", 0)),
        "trace_replay_manifest_sha256": _sha256_file(trace_path),
        "generic_tool_replay_assistant_tokens": (
            int(trace.get("selected_assistant_tokens", 0)) if generic_sources else 0
        ),
        "interactive_swe_replay_source_count": len(interactive_sources),
        "generic_tool_replay_source_count": len(generic_sources),
    }


def assess_builder_readiness(
    *,
    audit_receipt: Path,
    expected_audit_receipt_sha256: str,
    ptv2_root: Path,
    ptv2_revision: str,
    ptv3_prompt_root: Path,
    ptv3_source_manifest: Path,
) -> dict[str, Any]:
    """Return immutable evidence for every exact-quota prerequisite."""
    if re.fullmatch(r"[0-9a-f]{40}", ptv2_revision) is None:
        raise ValueError("PTV2 revision must be an exact Git commit")
    audit = _verify_audit(audit_receipt, expected_audit_receipt_sha256, ptv2_revision)
    upper_bounds, source_evidence = _ptv2_exhaustion_proof(audit, ptv2_root)
    ptv3 = _verify_ptv3_inputs(ptv3_prompt_root, ptv3_source_manifest)

    b_blockers = [
        f"PTV2 {domain} unseen candidate upper bound is zero after prior UUID exclusion"
        for domain in ("math", "code", "chat")
        if upper_bounds.get(domain) == 0
    ]
    c_blockers = []
    if ptv3["target_synth_assistant_tokens"] == 0:
        c_blockers.append("PTV3 target-synth assistant-loss-token inventory is zero")
    d_blockers = list(c_blockers)
    if ptv3["interactive_swe_replay_source_count"] == 0:
        d_blockers.append("PTV3 interactive-SWE replay source is absent")
    if ptv3["generic_tool_replay_assistant_tokens"] == 0:
        d_blockers.append("PTV3 generic-tool replay assistant-loss-token inventory is zero")
    blockers = {"B": b_blockers, "C": c_blockers, "D": d_blockers}
    return {
        "schema_version": 1,
        "contract": "ptv23-complement-v1-exact-assistant-loss-token-readiness",
        "quota_policy": "exact-no-renormalization",
        "audit_receipt_path": str(audit_receipt.resolve()),
        "audit_receipt_sha256": expected_audit_receipt_sha256,
        "ptv2": {
            "root": str(ptv2_root.resolve()),
            "revision": ptv2_revision,
            "unseen_candidate_upper_bound": upper_bounds,
            "source_file_evidence": source_evidence,
        },
        "ptv3": ptv3,
        "ready": {arm: not reasons for arm, reasons in blockers.items()},
        "blockers": blockers,
    }


def write_readiness_receipt(payload: dict[str, Any], output: Path) -> None:
    if output.exists():
        raise FileExistsError(f"readiness receipt already exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    receipt = dict(payload)
    receipt["receipt_sha256"] = sha256_bytes(canonical_json(receipt))
    partial = output.with_name(f".{output.name}.partial-{os.getpid()}")
    if partial.exists():
        raise FileExistsError(f"partial readiness receipt already exists: {partial}")
    try:
        with partial.open("wb") as destination:
            destination.write(canonical_json(receipt) + b"\n")
            destination.flush()
            os.fsync(destination.fileno())
        os.rename(partial, output)
        directory_fd = os.open(output.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    except BaseException:
        partial.unlink(missing_ok=True)
        raise


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audit-receipt", type=Path, required=True)
    parser.add_argument("--audit-receipt-sha256", required=True)
    parser.add_argument("--ptv2-root", type=Path, required=True)
    parser.add_argument("--ptv2-revision", required=True)
    parser.add_argument("--ptv3-prompt-root", type=Path, required=True)
    parser.add_argument("--ptv3-source-manifest", type=Path, required=True)
    parser.add_argument("--output-receipt", type=Path, required=True)
    args = parser.parse_args()
    payload = assess_builder_readiness(
        audit_receipt=args.audit_receipt,
        expected_audit_receipt_sha256=args.audit_receipt_sha256,
        ptv2_root=args.ptv2_root,
        ptv2_revision=args.ptv2_revision,
        ptv3_prompt_root=args.ptv3_prompt_root,
        ptv3_source_manifest=args.ptv3_source_manifest,
    )
    write_readiness_receipt(payload, args.output_receipt)
    print(json.dumps({"ready": payload["ready"], "blockers": payload["blockers"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
