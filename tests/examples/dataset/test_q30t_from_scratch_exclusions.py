# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Fail-closed held-out V2 exclusion contracts for Q30 corpus candidates."""

from __future__ import annotations

import hashlib
import json
import sys
from dataclasses import asdict, replace
from pathlib import Path
from typing import cast

import pytest

ROOT = Path(__file__).resolve().parents[3]
MODULE_DIR = ROOT / "examples/dataset"

sys.path.insert(0, str(MODULE_DIR))
try:
    import q30t_from_scratch_exclusions as exclusions_module  # pyright: ignore[reportMissingImports]
    from q30t_from_scratch_candidates import (  # pyright: ignore[reportMissingImports]
        PromptCandidate,
    )
    from q30t_from_scratch_exclusions import (  # pyright: ignore[reportMissingImports]
        FROZEN_SCIENTIFIC_IDENTITY,
        HELDOUT_V2_CATALOG_ORDER,
        CandidateExclusion,
        CatalogName,
        ExclusionLedger,
        FromScratchExclusionError,
        HeldoutV2Atom,
        HeldoutV2Attribution,
        HeldoutV2Bundle,
        apply_heldout_union,
        load_approved_heldout_v2,
    )
    from q30t_generation_prompt_identity import (  # pyright: ignore[reportMissingImports]
        GenerationPromptIdentity,
        PromptProvenance,
        generation_prompt_identity,
    )
    from specdec_corpus_contracts import (  # pyright: ignore[reportMissingImports]
        canonical_json,
        sha256_bytes,
    )
    from specdec_identity import prompt_uuid  # pyright: ignore[reportMissingImports]
finally:
    sys.path.pop(0)


def _canonical_text(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _content_digest(
    *, domain_id: int, rule_id: int, shape_id: int, field_key: str, value: str
) -> str:
    preimage = [
        "q30t-held-out-content-v2",
        domain_id,
        rule_id,
        shape_id,
        field_key,
        value,
    ]
    return hashlib.sha256(_canonical_text(preimage).encode()).hexdigest()


def _provenance(*, source_row_index: int = 0) -> PromptProvenance:
    return PromptProvenance(
        source_repository="fixture/repository",
        source_revision="a" * 40,
        source_file_sha256="b" * 64,
        source_split="train",
        source_row_index=source_row_index,
        reasoning_mode="reasoning_on",
        response_lane="math",
    )


def _candidate(
    content: str,
    *,
    source_row_index: int = 0,
    metadata: dict[str, object] | None = None,
    tools: list[dict[str, object]] | None = None,
) -> PromptCandidate:
    message: dict[str, object] = {"role": "user", "content": content}
    if metadata is not None:
        message["metadata"] = metadata
    messages = [message]
    tool_list = tools or []
    identity = generation_prompt_identity(
        messages,
        tool_list,
        _provenance(source_row_index=source_row_index),
    )
    request_bytes = canonical_json({"messages": messages, "tools": tool_list})
    return PromptCandidate(
        arm="M-synth",
        lane="math",
        language="",
        identity=identity,
        request_bytes=request_bytes,
        source_assistant=None,
        normalized_row_sha256=sha256_bytes(request_bytes),
    )


def _attribution(catalog: CatalogName, *component_ids: str) -> HeldoutV2Attribution:
    contributors = component_ids or (f"fixture-{catalog}",)
    return HeldoutV2Attribution(
        catalog=catalog,
        contributing_component_ids=contributors,
    )


def _atom(
    catalog: CatalogName,
    value: str,
    *,
    domain_id: int = 0,
    rule_id: int = 0,
    shape_id: int = 0,
    field_key: str = "",
    component_ids: tuple[str, ...] | None = None,
) -> HeldoutV2Atom:
    contributors = component_ids or (f"fixture-{catalog}",)
    return HeldoutV2Atom(
        catalog=catalog,
        domain_id=domain_id,
        rule_id=rule_id,
        shape_id=shape_id,
        field_key=field_key,
        digest=_content_digest(
            domain_id=domain_id,
            rule_id=rule_id,
            shape_id=shape_id,
            field_key=field_key,
            value=value,
        ),
        value=value,
        contributing_component_ids=contributors,
    )


def _atom_sort_key(atom: HeldoutV2Atom) -> tuple[object, ...]:
    return (
        HELDOUT_V2_CATALOG_ORDER.index(atom.catalog),
        atom.domain_id,
        atom.rule_id,
        atom.shape_id,
        atom.field_key.encode(),
        atom.digest,
        atom.value.encode(),
    )


def _bundle(
    *,
    prompt_attributions: dict[str, tuple[HeldoutV2Attribution, ...]] | None = None,
    exact_attributions: dict[str, tuple[HeldoutV2Attribution, ...]] | None = None,
    atoms: tuple[HeldoutV2Atom, ...] = (),
) -> HeldoutV2Bundle:
    prompt_attributions = prompt_attributions or {}
    exact_attributions = exact_attributions or {}
    all_atoms = list(atoms)
    populated = {
        attribution.catalog
        for attributions in (*prompt_attributions.values(), *exact_attributions.values())
        for attribution in attributions
    } | {atom.catalog for atom in all_atoms}
    all_atoms.extend(
        _atom(catalog, f"fixture filler {catalog}")
        for catalog in HELDOUT_V2_CATALOG_ORDER
        if catalog not in populated
    )

    component_lists = {catalog: [f"fixture-{catalog}"] for catalog in HELDOUT_V2_CATALOG_ORDER}
    for attributions in (*prompt_attributions.values(), *exact_attributions.values()):
        for attribution in attributions:
            for component_id in attribution.contributing_component_ids:
                if component_id not in component_lists[attribution.catalog]:
                    component_lists[attribution.catalog].append(component_id)
    for atom in all_atoms:
        for component_id in atom.contributing_component_ids:
            if component_id not in component_lists[atom.catalog]:
                component_lists[atom.catalog].append(component_id)

    return HeldoutV2Bundle(
        scientific_identity=FROZEN_SCIENTIFIC_IDENTITY,
        receipt_sha256="c" * 64,
        catalog_names=HELDOUT_V2_CATALOG_ORDER,
        prompt_uuids=frozenset(prompt_attributions),
        exact_content_sha256s=frozenset(exact_attributions),
        containment_index=tuple(sorted(all_atoms, key=_atom_sort_key)),
        approval_identity_sha256="d" * 64,
        component_order={
            catalog: tuple(component_lists[catalog]) for catalog in HELDOUT_V2_CATALOG_ORDER
        },
        prompt_uuid_catalogs=prompt_attributions,
        exact_content_catalogs=exact_attributions,
    )


def _canonical_file(path: Path, value: object) -> str:
    path.write_bytes(_canonical_text(value).encode() + b"\n")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _v2_bundle_receipt(
    *,
    catalog_order: list[str] | None = None,
    scientific_identity: str = FROZEN_SCIENTIFIC_IDENTITY,
    descriptor_paths: list[str] | None = None,
) -> dict[str, object]:
    catalogs = catalog_order or list(HELDOUT_V2_CATALOG_ORDER)
    paths = descriptor_paths or [f"{catalog}.json" for catalog in catalogs]
    body: dict[str, object] = {
        "schema": "q30t-held-out-bundle-v2",
        "scientific_identity": scientific_identity,
        "proposal_sha256": "1" * 64,
        "catalog_receipt_descriptors": [
            {
                "path": path,
                "bytes": 1,
                "sha256": f"{index + 2:064x}",
                "media_type": "application/json",
            }
            for index, path in enumerate(paths)
        ],
        "catalog_order": catalogs,
        "union_prompt_count": len(catalogs),
        "union_prompt_sha256": "8" * 64,
        "union_attribution_sha256": "9" * 64,
    }
    return body | {"self_sha256": hashlib.sha256(_canonical_text(body).encode()).hexdigest()}


def _large_node_candidate(node_count: int) -> PromptCandidate:
    provenance = _provenance()
    messages: list[dict[str, object]] = [
        {
            "role": "user",
            "content": "bounded candidate",
            "metadata": {"nodes": [{} for _ in range(node_count)]},
        }
    ]
    tools: list[dict[str, object]] = []
    identity_bytes = canonical_json(
        {"messages": messages, "tools": tools, "provenance": asdict(provenance)}
    )
    request_bytes = canonical_json({"messages": messages, "tools": tools})
    return PromptCandidate(
        arm="M-synth",
        lane="math",
        language="",
        identity=GenerationPromptIdentity(
            prompt_uuid=sha256_bytes(identity_bytes),
            canonical_bytes=identity_bytes,
            provenance=provenance,
        ),
        request_bytes=request_bytes,
        source_assistant=None,
        normalized_row_sha256=sha256_bytes(request_bytes),
    )


def test_bundle_requires_the_exact_five_approved_nonempty_catalogs() -> None:
    with pytest.raises(FromScratchExclusionError, match="speed,math,code,swe,tool"):
        HeldoutV2Bundle(
            scientific_identity=FROZEN_SCIENTIFIC_IDENTITY,
            receipt_sha256="c" * 64,
            catalog_names=("speed", "math", "code", "swe"),
            prompt_uuids=frozenset(),
            exact_content_sha256s=frozenset(),
            containment_index=(),
            approval_identity_sha256="d" * 64,
            component_order={},
        )

    with pytest.raises(FromScratchExclusionError, match="nonempty"):
        HeldoutV2Bundle(
            scientific_identity=FROZEN_SCIENTIFIC_IDENTITY,
            receipt_sha256="c" * 64,
            catalog_names=HELDOUT_V2_CATALOG_ORDER,
            prompt_uuids=frozenset(),
            exact_content_sha256s=frozenset(),
            containment_index=tuple(
                _atom(catalog, catalog) for catalog in HELDOUT_V2_CATALOG_ORDER if catalog != "tool"
            ),
            approval_identity_sha256="d" * 64,
            component_order={
                catalog: (f"fixture-{catalog}",) for catalog in HELDOUT_V2_CATALOG_ORDER
            },
        )


def test_shared_prompt_uuid_matches_across_different_provenance() -> None:
    first = _candidate("shared protected prompt", source_row_index=1)
    second = _candidate("shared protected prompt", source_row_index=2)
    shared = prompt_uuid([{"role": "user", "content": "shared protected prompt"}], [])
    heldout = _bundle(prompt_attributions={shared: (_attribution("math"),)})

    first_exclusion = apply_heldout_union(first, heldout)
    second_exclusion = apply_heldout_union(second, heldout)

    assert first.identity.prompt_uuid != second.identity.prompt_uuid
    assert first_exclusion is not None
    assert second_exclusion is not None
    assert first_exclusion.prompt_uuid == shared
    assert second_exclusion.prompt_uuid == shared
    assert first_exclusion.matched_identity == shared
    assert second_exclusion.matched_identity == shared


def test_prompt_uuid_exclusion_precedes_exact_and_containment() -> None:
    candidate = _candidate("protected prompt")
    shared = prompt_uuid([{"role": "user", "content": "protected prompt"}], [])
    heldout = _bundle(
        prompt_attributions={shared: (_attribution("math"),)},
        exact_attributions={candidate.normalized_row_sha256: (_attribution("code"),)},
        atoms=(_atom("tool", "protected prompt", shape_id=1),),
    )

    exclusion = apply_heldout_union(candidate, heldout)

    assert exclusion is not None
    assert exclusion.reason == "prompt_uuid"
    assert exclusion.catalog == "math"
    assert exclusion.protected_domain == 8
    assert exclusion.protected_rule == 0
    assert exclusion.protected_shape == 0


def test_normalized_projected_row_identity_precedes_containment() -> None:
    candidate = _candidate("protected full field")
    heldout = _bundle(
        exact_attributions={candidate.normalized_row_sha256: (_attribution("swe"),)},
        atoms=(_atom("code", "protected full field", shape_id=1),),
    )

    exclusion = apply_heldout_union(candidate, heldout)

    assert exclusion is not None
    assert exclusion.reason == "exact_content"
    assert exclusion.catalog == "swe"
    assert exclusion.matched_identity == candidate.normalized_row_sha256
    assert exclusion.protected_domain == 7
    assert exclusion.protected_shape == 10


def test_typed_patch_containment_preserves_domain_digest_and_all_contributors() -> None:
    protected = "return sorted(values)"
    atom = _atom(
        "code",
        protected,
        domain_id=2,
        shape_id=3,
        component_ids=("code-primary", "code-secondary"),
    )
    heldout = _bundle(atoms=(atom,))

    exclusion = apply_heldout_union(
        _candidate(f"Implement it.\r\n{protected}\nThen explain."), heldout
    )

    assert exclusion is not None
    assert exclusion.reason == "heldout_content_containment"
    assert exclusion.catalog == "code"
    assert exclusion.matched_identity == atom.digest
    assert exclusion.protected_domain == 2
    assert exclusion.protected_rule == 0
    assert exclusion.protected_shape == 3
    assert exclusion.protected_field_key == ""
    assert exclusion.contributing_component_ids == ("code-primary", "code-secondary")
    assert protected not in repr(exclusion)
    assert protected not in repr(heldout)


def test_same_value_in_different_domains_has_distinct_evidence() -> None:
    problem = _atom("math", "domain separated value", domain_id=0, shape_id=0)
    code = _atom("code", "domain separated value", domain_id=4, shape_id=0)
    assert problem.digest != code.digest

    exclusion = apply_heldout_union(
        _candidate("domain separated value"), _bundle(atoms=(problem, code))
    )

    assert exclusion is not None
    assert exclusion.catalog == "math"
    assert exclusion.protected_domain == 0
    assert exclusion.matched_identity == problem.digest


def test_fixture_atom_rejects_a_shape_outside_its_frozen_domain_contract() -> None:
    with pytest.raises(FromScratchExclusionError, match="domain/shape"):
        _atom("math", "problem fragment", domain_id=0, shape_id=3)


def test_short_code_atom_matches_only_a_complete_normalized_line() -> None:
    atom = _atom("code", "x=1", domain_id=4, shape_id=4)
    heldout = _bundle(atoms=(atom,))

    assert apply_heldout_union(_candidate("prefix x=1 suffix"), heldout) is None
    exclusion = apply_heldout_union(_candidate("before\nx=1\nafter"), heldout)
    assert exclusion is not None
    assert exclusion.matched_identity == atom.digest


def test_short_numeric_answer_requires_an_authenticated_answer_context() -> None:
    answer = _atom("math", "42", domain_id=1, shape_id=2)
    heldout = _bundle(atoms=(answer,))

    assert apply_heldout_union(_candidate("There are 42 items."), heldout) is None
    exclusion = apply_heldout_union(_candidate("Final answer: 42"), heldout)
    assert exclusion is not None
    assert exclusion.matched_identity == answer.digest
    assert exclusion.protected_shape == 2


def test_projection_scans_object_keys_and_canonical_structured_tool_values() -> None:
    key_atom = _atom("tool", "dangerous_key", domain_id=5, shape_id=0)
    key_exclusion = apply_heldout_union(
        _candidate("inspect", metadata={"dangerous_key": 7}), _bundle(atoms=(key_atom,))
    )
    assert key_exclusion is not None
    assert key_exclusion.matched_identity == key_atom.digest

    structured_value = _canonical_text({"const": "dangerous", "type": "string"})
    structured_atom = _atom("tool", structured_value, domain_id=5, shape_id=5)
    tool = {
        "type": "function",
        "function": {
            "name": "shell",
            "description": "run command",
            "parameters": {
                "type": "object",
                "properties": {"mode": {"type": "string", "const": "dangerous"}},
            },
        },
    }
    structured_exclusion = apply_heldout_union(
        _candidate("inspect", tools=[tool]), _bundle(atoms=(structured_atom,))
    )
    assert structured_exclusion is not None
    assert structured_exclusion.matched_identity == structured_atom.digest


def test_projection_scans_authenticated_tool_key_value_shape() -> None:
    atom = _atom("tool", "dangerous", domain_id=5, shape_id=7, field_key="const")
    tool = {
        "type": "function",
        "function": {
            "name": "shell",
            "description": "run command",
            "parameters": {
                "type": "object",
                "properties": {"mode": {"type": "string", "const": "dangerous"}},
            },
        },
    }

    exclusion = apply_heldout_union(_candidate("inspect", tools=[tool]), _bundle(atoms=(atom,)))

    assert exclusion is not None
    assert exclusion.protected_shape == 7
    assert exclusion.protected_field_key == "const"


def test_projection_rejects_request_fields_outside_the_candidate_contract() -> None:
    candidate = _candidate("bounded request")
    request = cast("dict[str, object]", json.loads(candidate.request_bytes))
    request["unknown_source_only_field"] = "must not be silently ignored"
    malformed = replace(candidate, request_bytes=canonical_json(request))

    with pytest.raises(FromScratchExclusionError, match="projection boundary"):
        apply_heldout_union(malformed, _bundle())


def test_candidate_with_65537_structured_nodes_fails_closed() -> None:
    with pytest.raises(FromScratchExclusionError, match="structured nodes"):
        apply_heldout_union(_large_node_candidate(65_537), _bundle())


def test_matcher_work_budget_fails_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(exclusions_module, "_MAX_MATCHER_WORK_BYTES", 8)
    atom = _atom("code", "protected", domain_id=4, shape_id=3)

    with pytest.raises(FromScratchExclusionError, match="matcher work"):
        apply_heldout_union(_candidate("prefix protected suffix"), _bundle(atoms=(atom,)))


def test_atom_count_and_bytes_are_bounded_before_matching(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(exclusions_module, "_MAX_ATOMS_PER_CATALOG_RULE_SHAPE", 1)
    with pytest.raises(FromScratchExclusionError, match="atom count"):
        _bundle(
            atoms=(
                _atom("code", "first atom", domain_id=4),
                _atom("code", "second atom", domain_id=4),
            )
        )

    monkeypatch.setattr(exclusions_module, "_MAX_ATOMS_PER_CATALOG_RULE_SHAPE", 20_000_000)
    monkeypatch.setattr(exclusions_module, "_MAX_ATOM_BYTES_PER_CATALOG", 64)
    with pytest.raises(FromScratchExclusionError, match="atom bytes"):
        _bundle(atoms=(_atom("code", "x" * 64, domain_id=4),))


def test_unicode_version_mismatch_is_fatal_before_matching(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    candidate = _candidate("protected")
    heldout = _bundle(atoms=(_atom("code", "protected", domain_id=4),))
    monkeypatch.setattr(exclusions_module.unicodedata, "unidata_version", "16.0.0")

    with pytest.raises(FromScratchExclusionError, match=r"Unicode 15\.1\.0"):
        apply_heldout_union(candidate, heldout)


def test_exclusion_ledger_reconciles_typed_evidence() -> None:
    exclusion = CandidateExclusion(
        prompt_uuid="1" * 64,
        catalog="code",
        reason="heldout_content_containment",
        matched_identity="2" * 64,
        protected_domain=4,
        protected_rule=0,
        protected_shape=3,
        protected_field_key="",
        contributing_component_ids=("fixture-code",),
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


def test_production_loader_rejects_symlinked_parent_component(tmp_path: Path) -> None:
    real_parent = tmp_path / "real"
    real_parent.mkdir()
    receipt = real_parent / "legacy.json"
    digest = _canonical_file(
        receipt,
        {
            "schema_version": "specdec-held-out-uuid-receipt-v1",
            "prompt_uuids": ["1" * 64],
            "prompt_uuids_sha256": "2" * 64,
            "receipt_sha256": "3" * 64,
        },
    )
    linked_parent = tmp_path / "linked"
    linked_parent.symlink_to(real_parent, target_is_directory=True)

    with pytest.raises(FromScratchExclusionError, match="symlink or unsafe path component"):
        load_approved_heldout_v2(linked_parent / receipt.name, expected_receipt_sha256=digest)


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


def test_production_loader_requires_the_frozen_scientific_identity(tmp_path: Path) -> None:
    receipt = tmp_path / "bundle.json"
    digest = _canonical_file(
        receipt,
        _v2_bundle_receipt(scientific_identity="wrong-scientific-identity"),
    )

    with pytest.raises(FromScratchExclusionError, match="scientific identity"):
        load_approved_heldout_v2(receipt, expected_receipt_sha256=digest)


@pytest.mark.parametrize("path", ["a/./b", "a//b", ".", "a/../b", "a/"])
def test_production_loader_rejects_descriptor_path_before_normalization(
    tmp_path: Path, path: str
) -> None:
    receipt = tmp_path / "bundle.json"
    descriptor_paths = [path, "math.json", "code.json", "swe.json", "tool.json"]
    digest = _canonical_file(
        receipt,
        _v2_bundle_receipt(descriptor_paths=descriptor_paths),
    )

    with pytest.raises(FromScratchExclusionError, match="canonical relative"):
        load_approved_heldout_v2(receipt, expected_receipt_sha256=digest)


def test_production_loader_keeps_complete_v2_bundle_in_state_p(tmp_path: Path) -> None:
    receipt = tmp_path / "bundle.json"
    digest = _canonical_file(receipt, _v2_bundle_receipt())

    with pytest.raises(
        FromScratchExclusionError,
        match="producer layout and trusted keyring verifier are unavailable",
    ):
        load_approved_heldout_v2(receipt, expected_receipt_sha256=digest)
