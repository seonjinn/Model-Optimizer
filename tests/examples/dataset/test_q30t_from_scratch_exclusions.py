# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Fail-closed held-out V2 exclusion contracts for Q30 corpus candidates."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
MODULE_DIR = ROOT / "examples/dataset"

sys.path.insert(0, str(MODULE_DIR))
try:
    from q30t_from_scratch_candidates import (  # pyright: ignore[reportMissingImports]
        PromptCandidate,
    )
    from q30t_from_scratch_exclusions import (  # pyright: ignore[reportMissingImports]
        HELDOUT_V2_CATALOG_ORDER,
        CandidateExclusion,
        ExclusionLedger,
        FromScratchExclusionError,
        HeldoutV2Bundle,
        apply_heldout_union,
        load_approved_heldout_v2,
    )
    from q30t_generation_prompt_identity import (  # pyright: ignore[reportMissingImports]
        PromptProvenance,
        generation_prompt_identity,
    )
    from specdec_corpus_contracts import (  # pyright: ignore[reportMissingImports]
        canonical_json,
        sha256_bytes,
    )
finally:
    sys.path.pop(0)


def _candidate(content: str) -> PromptCandidate:
    messages: list[dict[str, object]] = [{"role": "user", "content": content}]
    tools: list[dict[str, object]] = []
    identity = generation_prompt_identity(
        messages,
        tools,
        PromptProvenance(
            source_repository="fixture/repository",
            source_revision="a" * 40,
            source_file_sha256="b" * 64,
            source_split="train",
            source_row_index=0,
            reasoning_mode="reasoning_on",
            response_lane="math",
        ),
    )
    return PromptCandidate(
        arm="M-synth",
        lane="math",
        language="",
        identity=identity,
        request_bytes=canonical_json({"messages": messages, "tools": tools}),
        source_assistant=None,
        normalized_row_sha256=sha256_bytes(canonical_json({"messages": messages, "tools": tools})),
    )


def _bundle(
    *,
    prompt_catalogs: dict[str, tuple[str, ...]] | None = None,
    exact_catalogs: dict[str, tuple[str, ...]] | None = None,
    containment: dict[str, tuple[str, ...]] | None = None,
) -> HeldoutV2Bundle:
    prompt_catalogs = prompt_catalogs or {}
    exact_catalogs = exact_catalogs or {}
    containment = containment or {}
    populated = {
        catalog
        for catalogs in (*prompt_catalogs.values(), *exact_catalogs.values(), *containment.values())
        for catalog in catalogs
    }
    for index, catalog in enumerate(HELDOUT_V2_CATALOG_ORDER):
        if catalog not in populated:
            containment[f"fixture protected value {index}"] = (catalog,)
    return HeldoutV2Bundle(
        receipt_sha256="c" * 64,
        catalog_names=HELDOUT_V2_CATALOG_ORDER,
        prompt_uuids=frozenset(prompt_catalogs),
        exact_content_sha256s=frozenset(exact_catalogs),
        containment_index=containment,
        approval_identity_sha256="d" * 64,
        prompt_uuid_catalogs=prompt_catalogs,
        exact_content_catalogs=exact_catalogs,
    )


def _canonical_file(path: Path, value: object) -> str:
    path.write_bytes(
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
        + b"\n"
    )
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _v2_bundle_receipt(*, catalog_order: list[str]) -> dict[str, object]:
    body: dict[str, object] = {
        "schema": "q30t-held-out-bundle-v2",
        "scientific_identity": "fixture-scientific-identity",
        "proposal_sha256": "1" * 64,
        "catalog_receipt_descriptors": [
            {
                "path": f"{catalog}.json",
                "bytes": 1,
                "sha256": f"{index + 2:064x}",
                "media_type": "application/json",
            }
            for index, catalog in enumerate(catalog_order)
        ],
        "catalog_order": catalog_order,
        "union_prompt_count": len(catalog_order),
        "union_prompt_sha256": "8" * 64,
        "union_attribution_sha256": "9" * 64,
    }
    return body | {
        "self_sha256": hashlib.sha256(
            json.dumps(body, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
    }


def test_bundle_requires_the_exact_five_approved_nonempty_catalogs() -> None:
    with pytest.raises(FromScratchExclusionError, match="speed,math,code,swe,tool"):
        HeldoutV2Bundle(
            receipt_sha256="c" * 64,
            catalog_names=("speed", "math", "code", "swe"),
            prompt_uuids=frozenset(),
            exact_content_sha256s=frozenset(),
            containment_index={catalog: (catalog,) for catalog in ("speed", "math", "code", "swe")},
            approval_identity_sha256="d" * 64,
        )

    with pytest.raises(FromScratchExclusionError, match="nonempty"):
        HeldoutV2Bundle(
            receipt_sha256="c" * 64,
            catalog_names=HELDOUT_V2_CATALOG_ORDER,
            prompt_uuids=frozenset(),
            exact_content_sha256s=frozenset(),
            containment_index={
                catalog: (catalog,) for catalog in HELDOUT_V2_CATALOG_ORDER if catalog != "tool"
            },
            approval_identity_sha256="d" * 64,
        )


def test_prompt_uuid_exclusion_precedes_exact_and_containment() -> None:
    candidate = _candidate("protected prompt")
    heldout = _bundle(
        prompt_catalogs={candidate.identity.prompt_uuid: ("math",)},
        exact_catalogs={candidate.normalized_row_sha256: ("code",)},
        containment={"protected prompt": ("tool",)},
    )

    exclusion = apply_heldout_union(candidate, heldout)

    assert exclusion == CandidateExclusion(
        prompt_uuid=candidate.identity.prompt_uuid,
        catalog="math",
        reason="prompt_uuid",
        matched_identity=candidate.identity.prompt_uuid,
    )


def test_normalized_full_field_identity_precedes_containment() -> None:
    candidate = _candidate("protected full field")
    heldout = _bundle(
        exact_catalogs={candidate.normalized_row_sha256: ("swe",)},
        containment={"protected full field": ("code",)},
    )

    exclusion = apply_heldout_union(candidate, heldout)

    assert exclusion is not None
    assert exclusion.reason == "exact_content"
    assert exclusion.catalog == "swe"
    assert exclusion.matched_identity == candidate.normalized_row_sha256


def test_answer_or_patch_containment_excludes_a_longer_candidate_without_raw_evidence() -> None:
    protected = "return sorted(values)"
    candidate = _candidate(f"Implement it.\r\n{protected}\nThen explain.")
    heldout = _bundle(containment={protected: ("code",)})

    exclusion = apply_heldout_union(candidate, heldout)

    assert exclusion is not None
    assert exclusion.reason == "heldout_content_containment"
    assert exclusion.catalog == "code"
    assert exclusion.matched_identity == hashlib.sha256(protected.encode()).hexdigest()
    assert protected not in repr(exclusion)
    assert protected not in repr(heldout)


def test_containment_uses_fixed_catalog_order_and_preserves_case() -> None:
    candidate = _candidate("Answer: Exact Protected Text")
    heldout = _bundle(containment={"Exact Protected Text": ("math", "code")})

    exclusion = apply_heldout_union(candidate, heldout)

    assert exclusion is not None
    assert exclusion.catalog == "math"
    assert apply_heldout_union(_candidate("answer: exact protected text"), heldout) is None


def test_exclusion_ledger_rejects_raw_or_inconsistent_evidence() -> None:
    exclusion = CandidateExclusion(
        prompt_uuid="1" * 64,
        catalog="code",
        reason="heldout_content_containment",
        matched_identity="2" * 64,
    )
    ledger = ExclusionLedger(
        heldout_receipt_sha256="3" * 64,
        admitted_prompt_ids_sha256="4" * 64,
        excluded=(exclusion,),
        counts_by_catalog={catalog: int(catalog == "code") for catalog in HELDOUT_V2_CATALOG_ORDER},
    )
    assert ledger.counts_by_catalog["code"] == 1

    with pytest.raises(FromScratchExclusionError, match="counts"):
        ExclusionLedger(
            heldout_receipt_sha256="3" * 64,
            admitted_prompt_ids_sha256="4" * 64,
            excluded=(exclusion,),
            counts_by_catalog=dict.fromkeys(HELDOUT_V2_CATALOG_ORDER, 0),
        )


def test_production_loader_authenticates_receipt_before_parsing(tmp_path: Path) -> None:
    receipt = tmp_path / "heldout-root.json"
    receipt.write_bytes(b"not-json\n")

    with pytest.raises(FromScratchExclusionError, match="SHA-256 mismatch"):
        load_approved_heldout_v2(receipt, expected_receipt_sha256="0" * 64)


def test_production_loader_rejects_authenticated_legacy_v1(tmp_path: Path) -> None:
    receipt = tmp_path / "legacy.json"
    digest = _canonical_file(
        receipt,
        {
            "schema_version": "specdec-held-out-uuid-receipt-v1",
            "prompt_uuids": ["1" * 64],
            "prompt_uuids_sha256": "2" * 64,
            "receipt_sha256": "3" * 64,
        },
    )

    with pytest.raises(FromScratchExclusionError, match="legacy held-out V1"):
        load_approved_heldout_v2(receipt, expected_receipt_sha256=digest)


def test_production_loader_rejects_directory_layout_inference(tmp_path: Path) -> None:
    with pytest.raises(FromScratchExclusionError, match="explicit receipt file"):
        load_approved_heldout_v2(tmp_path, expected_receipt_sha256="0" * 64)


def test_production_loader_rejects_missing_or_extra_v2_catalogs_before_blocker(
    tmp_path: Path,
) -> None:
    receipt = tmp_path / "bundle.json"
    digest = _canonical_file(
        receipt,
        _v2_bundle_receipt(catalog_order=["speed", "math", "code", "swe"]),
    )
    with pytest.raises(FromScratchExclusionError, match="speed,math,code,swe,tool"):
        load_approved_heldout_v2(receipt, expected_receipt_sha256=digest)

    digest = _canonical_file(
        receipt,
        _v2_bundle_receipt(catalog_order=["speed", "math", "code", "swe", "tool", "unknown"]),
    )
    with pytest.raises(FromScratchExclusionError, match="speed,math,code,swe,tool"):
        load_approved_heldout_v2(receipt, expected_receipt_sha256=digest)


def test_production_loader_keeps_complete_v2_bundle_in_state_p(tmp_path: Path) -> None:
    receipt = tmp_path / "bundle.json"
    digest = _canonical_file(
        receipt,
        _v2_bundle_receipt(catalog_order=list(HELDOUT_V2_CATALOG_ORDER)),
    )

    with pytest.raises(
        FromScratchExclusionError,
        match="producer layout and trusted keyring verifier are unavailable",
    ):
        load_approved_heldout_v2(receipt, expected_receipt_sha256=digest)
