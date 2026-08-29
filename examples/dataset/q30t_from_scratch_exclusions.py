# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Fail-closed held-out V2 exclusion contracts for Q30 corpus candidates."""

from __future__ import annotations

import json
import math
import os
import re
import stat
import unicodedata
from collections.abc import Callable, Iterator, Mapping
from dataclasses import asdict, dataclass, field
from hashlib import sha256
from pathlib import Path
from types import MappingProxyType
from typing import Any, Literal

from q30t_from_scratch_candidates import PromptCandidate
from specdec_identity import prompt_uuid as shared_prompt_uuid

__all__ = [
    "FROZEN_SCIENTIFIC_IDENTITY",
    "HELDOUT_V2_CATALOG_ORDER",
    "CandidateExclusion",
    "CatalogName",
    "ExclusionLedger",
    "FromScratchExclusionError",
    "HeldoutV2Atom",
    "HeldoutV2Attribution",
    "HeldoutV2Bundle",
    "apply_heldout_union",
    "load_approved_heldout_v2",
]

CatalogName = Literal["speed", "math", "code", "swe", "tool"]
ExclusionReason = Literal["prompt_uuid", "exact_content", "heldout_content_containment"]

FROZEN_SCIENTIFIC_IDENTITY = "ptv2-ptv3-complement-700k-qwen3-30ba3b-thinking-swe-heavy-v1"
HELDOUT_V2_CATALOG_ORDER: tuple[CatalogName, ...] = (
    "speed",
    "math",
    "code",
    "swe",
    "tool",
)

_BUNDLE_SCHEMA = "q30t-held-out-bundle-v2"
_LEGACY_SCHEMAS = frozenset(
    {
        "q30t-held-out-sources-v1",
        "specdec-held-out-uuid-receipt-v1",
        "q30t-held-out-provenance-v1",
        "q30t-held-out-review-v1",
    }
)
_MAX_RECEIPT_BYTES = 16 * 1024 * 1024
_MAX_DECODED_ROW_BYTES = 64 * 1024 * 1024
_MAX_SCALAR_UTF8_BYTES = 16 * 1024 * 1024
_MAX_NESTING_DEPTH = 64
_MAX_STRUCTURED_NODES = 65_536
_MAX_STRING_LEAVES = 65_536
_MAX_GENERATED_VIEW_COUNT = 65_536
_MAX_GENERATED_VIEW_BYTES = 256 * 1024 * 1024
_MAX_UNIQUE_SCANNED_VIEW_BYTES = 256 * 1024 * 1024
_MAX_MATCHER_WORK_BYTES = 1024 * 1024 * 1024
_MAX_ATOMS_PER_CATALOG_RULE_SHAPE = 20_000_000
_MAX_ATOM_BYTES_PER_CATALOG = 8 * 1024 * 1024 * 1024
_SUPPORTED_FIXTURE_SHAPES = frozenset(range(8)) | {10}
_NUMERIC_ANSWER = re.compile(
    r"[+-]?(?:[0-9]+(?:\.[0-9]+)?|[0-9]+/[0-9]+|[0-9]+(?:,[0-9]{3})+)(?:%|[A-Za-z]+)?"
)
_WHITE_SPACE = frozenset(
    chr(codepoint)
    for start, end in (
        (0x0009, 0x000D),
        (0x0020, 0x0020),
        (0x0085, 0x0085),
        (0x00A0, 0x00A0),
        (0x1680, 0x1680),
        (0x2000, 0x200A),
        (0x2028, 0x2029),
        (0x202F, 0x202F),
        (0x205F, 0x205F),
        (0x3000, 0x3000),
    )
    for codepoint in range(start, end + 1)
)


class FromScratchExclusionError(ValueError):
    """Held-out evidence or candidate exclusion data failed closed."""


@dataclass(frozen=True)
class HeldoutV2Attribution:
    """One catalog's fixed-order contributors to a protected identity."""

    catalog: CatalogName
    contributing_component_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        _require_catalog(self.catalog)
        _validate_component_ids(self.contributing_component_ids)


@dataclass(frozen=True)
class HeldoutV2Atom:
    """One authenticated V2 domain/rule/shape atom with hidden source bytes."""

    catalog: CatalogName
    domain_id: int
    rule_id: int
    shape_id: int
    field_key: str
    digest: str
    value: str = field(repr=False)
    contributing_component_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        _require_unicode_version()
        _require_catalog(self.catalog)
        if type(self.domain_id) is not int or self.domain_id not in range(8):
            raise FromScratchExclusionError("held-out atom domain ID is invalid")
        if type(self.rule_id) is not int or self.rule_id not in range(3):
            raise FromScratchExclusionError("held-out atom rule ID is invalid")
        if type(self.shape_id) is not int or self.shape_id not in range(11):
            raise FromScratchExclusionError("held-out atom shape ID is invalid")
        if self.shape_id not in _SUPPORTED_FIXTURE_SHAPES:
            raise FromScratchExclusionError(
                "fixture token/line window atoms require the absent producer adjacency index"
            )
        if self.rule_id != 0:
            raise FromScratchExclusionError(
                "fixture atoms support only authenticated full-rule shapes"
            )
        if self.shape_id == 7:
            if not self.field_key:
                raise FromScratchExclusionError("tool key/value atom requires a field key")
        elif self.field_key:
            raise FromScratchExclusionError("only a tool key/value atom may contain a field key")
        if not isinstance(self.value, str) or not self.value:
            raise FromScratchExclusionError("held-out atom value is empty")
        normalized_value = _containment_text(self.value)
        if normalized_value != self.value:
            raise FromScratchExclusionError("held-out atom value is not pre-normalized")
        if _normalize_text(self.field_key) != self.field_key:
            raise FromScratchExclusionError("held-out atom field key is not normalized")
        _validate_full_shape_contract(self)
        if len(self.field_key.encode("utf-8")) > 65_535:
            raise FromScratchExclusionError("held-out atom field key exceeds the byte cap")
        if len(self.value.encode("utf-8")) > _MAX_SCALAR_UTF8_BYTES:
            raise FromScratchExclusionError("held-out atom value exceeds the scalar byte cap")
        _require_sha256(self.digest, label="held-out atom")
        expected = _content_digest(
            domain_id=self.domain_id,
            rule_id=self.rule_id,
            shape_id=self.shape_id,
            field_key=self.field_key,
            value=self.value,
        )
        if self.digest != expected:
            raise FromScratchExclusionError("held-out atom digest does not reconcile")
        _validate_component_ids(self.contributing_component_ids)


@dataclass(frozen=True)
class HeldoutV2Bundle:
    """Validated in-memory identities; only the production loader can grant trust."""

    scientific_identity: str
    receipt_sha256: str
    catalog_names: tuple[CatalogName, ...]
    prompt_uuids: frozenset[str]
    exact_content_sha256s: frozenset[str]
    containment_index: tuple[HeldoutV2Atom, ...] = field(repr=False)
    approval_identity_sha256: str
    component_order: Mapping[CatalogName, tuple[str, ...]] = field(repr=False)
    prompt_uuid_catalogs: Mapping[str, tuple[HeldoutV2Attribution, ...]] = field(
        default_factory=dict, repr=False
    )
    exact_content_catalogs: Mapping[str, tuple[HeldoutV2Attribution, ...]] = field(
        default_factory=dict, repr=False
    )
    _atoms_by_catalog: Mapping[CatalogName, tuple[HeldoutV2Atom, ...]] = field(
        init=False, repr=False, compare=False
    )

    def __post_init__(self) -> None:
        _require_unicode_version()
        if self.scientific_identity != FROZEN_SCIENTIFIC_IDENTITY:
            raise FromScratchExclusionError("held-out scientific identity is invalid")
        _require_sha256(self.receipt_sha256, label="held-out receipt")
        _require_sha256(self.approval_identity_sha256, label="held-out approval identity")
        if self.catalog_names != HELDOUT_V2_CATALOG_ORDER:
            raise FromScratchExclusionError(
                "held-out bundle must contain catalogs speed,math,code,swe,tool in fixed order"
            )
        components = _validate_component_order(self.component_order)
        prompt_catalogs = _validate_identity_attributions(
            self.prompt_uuid_catalogs,
            component_order=components,
            label="prompt UUID",
        )
        exact_catalogs = _validate_identity_attributions(
            self.exact_content_catalogs,
            component_order=components,
            label="exact content",
        )
        if (
            not isinstance(self.prompt_uuids, frozenset)
            or frozenset(prompt_catalogs) != self.prompt_uuids
        ):
            raise FromScratchExclusionError(
                "held-out prompt UUID identities lack exact catalog attribution"
            )
        if (
            not isinstance(self.exact_content_sha256s, frozenset)
            or frozenset(exact_catalogs) != self.exact_content_sha256s
        ):
            raise FromScratchExclusionError(
                "held-out exact-content identities lack exact catalog attribution"
            )
        atoms, atoms_by_catalog = _validate_atoms(
            self.containment_index, component_order=components
        )
        populated = {
            attribution.catalog
            for attributions in (*prompt_catalogs.values(), *exact_catalogs.values())
            for attribution in attributions
        } | {atom.catalog for atom in atoms}
        if populated != set(HELDOUT_V2_CATALOG_ORDER):
            raise FromScratchExclusionError(
                "held-out catalogs speed,math,code,swe,tool must each be approved and nonempty"
            )
        object.__setattr__(self, "component_order", MappingProxyType(components))
        object.__setattr__(self, "prompt_uuid_catalogs", MappingProxyType(prompt_catalogs))
        object.__setattr__(self, "exact_content_catalogs", MappingProxyType(exact_catalogs))
        object.__setattr__(self, "containment_index", atoms)
        object.__setattr__(self, "_atoms_by_catalog", MappingProxyType(atoms_by_catalog))


@dataclass(frozen=True)
class CandidateExclusion:
    """Stable, raw-content-free evidence for one excluded candidate."""

    prompt_uuid: str
    catalog: CatalogName
    reason: ExclusionReason
    matched_identity: str
    protected_domain: int
    protected_rule: int
    protected_shape: int
    protected_field_key: str
    contributing_component_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        _require_sha256(self.prompt_uuid, label="excluded prompt UUID")
        _require_sha256(self.matched_identity, label="matched held-out identity")
        _require_catalog(self.catalog)
        if self.reason not in {
            "prompt_uuid",
            "exact_content",
            "heldout_content_containment",
        }:
            raise FromScratchExclusionError("excluded reason is invalid")
        if type(self.protected_domain) is not int or self.protected_domain not in range(9):
            raise FromScratchExclusionError("excluded protected domain is invalid")
        if type(self.protected_rule) is not int or self.protected_rule not in range(3):
            raise FromScratchExclusionError("excluded protected rule is invalid")
        if type(self.protected_shape) is not int or self.protected_shape not in range(11):
            raise FromScratchExclusionError("excluded protected shape is invalid")
        if self.protected_shape == 7:
            if not self.protected_field_key:
                raise FromScratchExclusionError("excluded tool key/value lacks its field key")
        elif self.protected_field_key:
            raise FromScratchExclusionError("excluded evidence has an invalid field key")
        _validate_component_ids(self.contributing_component_ids)


@dataclass(frozen=True)
class ExclusionLedger:
    """Aggregate stable exclusion evidence without benchmark text."""

    heldout_receipt_sha256: str
    admitted_prompt_ids_sha256: str
    excluded: tuple[CandidateExclusion, ...]
    counts_by_catalog: Mapping[str, int]

    def __post_init__(self) -> None:
        _require_sha256(self.heldout_receipt_sha256, label="ledger held-out receipt")
        _require_sha256(self.admitted_prompt_ids_sha256, label="ledger admitted prompts")
        if not isinstance(self.excluded, tuple) or any(
            not isinstance(item, CandidateExclusion) for item in self.excluded
        ):
            raise FromScratchExclusionError("ledger excluded events are invalid")
        if (
            not isinstance(self.counts_by_catalog, Mapping)
            or tuple(self.counts_by_catalog) != HELDOUT_V2_CATALOG_ORDER
        ):
            raise FromScratchExclusionError("ledger counts must use the exact catalog order")
        counts = dict(self.counts_by_catalog)
        if any(type(value) is not int or value < 0 for value in counts.values()):
            raise FromScratchExclusionError("ledger counts must be nonnegative integers")
        observed = dict.fromkeys(HELDOUT_V2_CATALOG_ORDER, 0)
        for exclusion in self.excluded:
            observed[exclusion.catalog] += 1
        if counts != observed:
            raise FromScratchExclusionError("ledger counts do not reconcile with exclusions")
        object.__setattr__(self, "counts_by_catalog", MappingProxyType(counts))


@dataclass(frozen=True)
class _CandidateViews:
    prompt_uuid: str
    text: tuple[str, ...]
    structured: tuple[str, ...]
    whole_lines: tuple[str, ...]
    key_values: Mapping[str, tuple[str, ...]]
    raw_record: str


@dataclass
class _TraversalCounters:
    structured_nodes: int = 0
    string_leaves: int = 0
    decoded_bytes: int = 0
    generated_count: int = 0
    generated_bytes: int = 0
    unique_bytes: int = 0


@dataclass
class _MatcherBudget:
    work_bytes: int = 0

    def charge(self, view: str, atom: HeldoutV2Atom) -> None:
        self.work_bytes += len(view.encode("utf-8")) + len(atom.value.encode("utf-8"))
        if self.work_bytes > _MAX_MATCHER_WORK_BYTES:
            raise FromScratchExclusionError("candidate matcher work exceeds the byte cap")


def load_approved_heldout_v2(root: Path, *, expected_receipt_sha256: str) -> HeldoutV2Bundle:
    """Authenticate an explicit root receipt and remain blocked without V2 approval."""
    raw = _read_authenticated_receipt(root, expected_receipt_sha256)
    payload = _load_canonical_object(raw)
    schema = payload.get("schema")
    legacy_schema = payload.get("schema_version")
    if legacy_schema in _LEGACY_SCHEMAS or schema in _LEGACY_SCHEMAS:
        raise FromScratchExclusionError(
            "legacy held-out V1 receipt cannot satisfy the held-out V2 consumer"
        )
    if schema == _BUNDLE_SCHEMA:
        _validate_v2_bundle_receipt(payload)
        raise FromScratchExclusionError(
            "held-out V2 producer layout and trusted keyring verifier are unavailable; "
            "the authenticated bundle remains in state P"
        )
    if schema in {
        "specdec-held-out-receipt-v2",
        "q30t-held-out-approval-v2",
        "q30t-held-out-index-v2",
        "q30t-held-out-atoms-v2",
        "q30t-held-out-attribution-v2",
    }:
        raise FromScratchExclusionError(
            "an individual held-out V2 artifact cannot establish the approved five-catalog union"
        )
    raise FromScratchExclusionError("held-out root receipt schema is unsupported")


def apply_heldout_union(
    candidate: PromptCandidate, heldout: HeldoutV2Bundle
) -> CandidateExclusion | None:
    """Scan the complete PromptCandidate projection in prompt/exact/atom order.

    The enforceable projection is exactly response-free identity messages and
    tools plus the optional H-native source assistant. Source-only fields that
    the PromptCandidate contract does not retain are unavailable and cannot be
    claimed as scanned by this fixture consumer.
    """
    _require_unicode_version()
    if not isinstance(candidate, PromptCandidate):
        raise FromScratchExclusionError("held-out candidate type is invalid")
    if not isinstance(heldout, HeldoutV2Bundle):
        raise FromScratchExclusionError("held-out bundle type is invalid")
    views = _candidate_views(candidate)

    prompt_attributions = heldout.prompt_uuid_catalogs.get(views.prompt_uuid)
    if prompt_attributions is not None:
        return _identity_exclusion(
            prompt_uuid=views.prompt_uuid,
            matched_identity=views.prompt_uuid,
            reason="prompt_uuid",
            domain=8,
            shape=0,
            attributions=prompt_attributions,
        )

    exact_attributions = heldout.exact_content_catalogs.get(candidate.normalized_row_sha256)
    if exact_attributions is not None:
        return _identity_exclusion(
            prompt_uuid=views.prompt_uuid,
            matched_identity=candidate.normalized_row_sha256,
            reason="exact_content",
            domain=7,
            shape=10,
            attributions=exact_attributions,
        )

    budget = _MatcherBudget()
    for catalog in HELDOUT_V2_CATALOG_ORDER:
        for atom in heldout._atoms_by_catalog[catalog]:
            if _atom_matches(atom, views, budget):
                return CandidateExclusion(
                    prompt_uuid=views.prompt_uuid,
                    catalog=atom.catalog,
                    reason="heldout_content_containment",
                    matched_identity=atom.digest,
                    protected_domain=atom.domain_id,
                    protected_rule=atom.rule_id,
                    protected_shape=atom.shape_id,
                    protected_field_key=atom.field_key,
                    contributing_component_ids=atom.contributing_component_ids,
                )
    return None


def _identity_exclusion(
    *,
    prompt_uuid: str,
    matched_identity: str,
    reason: ExclusionReason,
    domain: int,
    shape: int,
    attributions: tuple[HeldoutV2Attribution, ...],
) -> CandidateExclusion:
    contributors = tuple(
        component
        for attribution in attributions
        for component in attribution.contributing_component_ids
    )
    return CandidateExclusion(
        prompt_uuid=prompt_uuid,
        catalog=attributions[0].catalog,
        reason=reason,
        matched_identity=matched_identity,
        protected_domain=domain,
        protected_rule=0,
        protected_shape=shape,
        protected_field_key="",
        contributing_component_ids=contributors,
    )


def _candidate_views(candidate: PromptCandidate) -> _CandidateViews:
    if len(candidate.identity.canonical_bytes) > _MAX_DECODED_ROW_BYTES:
        raise FromScratchExclusionError("candidate decoded row exceeds the byte cap")
    payload = _load_canonical_value(candidate.identity.canonical_bytes, label="candidate identity")
    if not isinstance(payload, dict) or set(payload) != {"messages", "provenance", "tools"}:
        raise FromScratchExclusionError("candidate identity payload is invalid")
    if payload["provenance"] != asdict(candidate.identity.provenance):
        raise FromScratchExclusionError("candidate identity provenance does not reconcile")
    if sha256(candidate.identity.canonical_bytes).hexdigest() != candidate.identity.prompt_uuid:
        raise FromScratchExclusionError("candidate generation identity hash does not reconcile")
    expected_request = _canonical_json({"messages": payload["messages"], "tools": payload["tools"]})
    if candidate.arm == "H-native":
        if candidate.request_bytes or candidate.source_assistant is None:
            raise FromScratchExclusionError("candidate projection boundary is invalid")
    elif candidate.request_bytes != expected_request or candidate.source_assistant is not None:
        raise FromScratchExclusionError("candidate projection boundary is invalid")
    _require_sha256(candidate.normalized_row_sha256, label="candidate projected row")

    counters = _TraversalCounters()
    containers: list[dict[str, object] | list[object]] = []
    text_values: list[str] = []
    key_values: dict[str, list[object]] = {}
    projection = _normalize_candidate_value(
        {
            "messages": payload["messages"],
            "tools": payload["tools"],
            "source_assistant": candidate.source_assistant,
        },
        depth=0,
        counters=counters,
        containers=containers,
        text_values=text_values,
        key_values=key_values,
    )
    seen_views: set[tuple[str, int, str]] = set()
    text: list[str] = []
    for value in text_values:
        _preflight_containment_view(value, counters)
        normalized = _containment_text(value)
        if normalized:
            _append_unique_view(
                text,
                "text",
                normalized,
                counters=counters,
                seen=seen_views,
            )
    structured: list[str] = []
    for container in containers:
        _preflight_json_view(container, counters, containment=True)
        _append_unique_view(
            structured,
            "structured",
            _containment_text(_canonical_json(container).decode("utf-8")),
            counters=counters,
            seen=seen_views,
        )
    normalized_key_values: dict[str, tuple[str, ...]] = {}
    for key, values in key_values.items():
        recorded: list[str] = []
        for value in values:
            if isinstance(value, str):
                _preflight_containment_view(value, counters)
                normalized_value = _containment_text(value)
            else:
                _preflight_json_view(value, counters, containment=True)
                normalized_value = _containment_text(_canonical_json(value).decode("utf-8"))
            _append_unique_view(
                recorded,
                f"key-value:{key}",
                normalized_value,
                counters=counters,
                seen=seen_views,
            )
        normalized_key_values[key] = tuple(recorded)
    if not isinstance(projection, dict):
        raise FromScratchExclusionError("candidate projection is invalid")
    _append_joined_candidate_views(
        projection,
        text,
        counters=counters,
        seen=seen_views,
    )
    whole_lines: list[str] = []
    for value in text_values:
        for line_start, line_end in _iter_line_bounds(value):
            _preflight_containment_range(value, line_start, line_end, counters)
            line = value[line_start:line_end]
            normalized_line = _containment_text(line)
            if normalized_line:
                _append_unique_view(
                    whole_lines,
                    "whole-line",
                    normalized_line,
                    counters=counters,
                    seen=seen_views,
                )
    _preflight_json_view(
        projection,
        counters,
        containment=False,
        per_view_byte_cap=_MAX_DECODED_ROW_BYTES,
    )
    raw_record = _canonical_json(projection).decode("utf-8")
    _record_view("raw-record", raw_record, counters=counters, seen=seen_views)
    return _CandidateViews(
        prompt_uuid=shared_prompt_uuid(payload["messages"], payload["tools"]),
        text=tuple(text),
        structured=tuple(structured),
        whole_lines=tuple(whole_lines),
        key_values=MappingProxyType(normalized_key_values),
        raw_record=raw_record,
    )


def _normalize_candidate_value(
    value: object,
    *,
    depth: int,
    counters: _TraversalCounters,
    containers: list[dict[str, object] | list[object]],
    text_values: list[str],
    key_values: dict[str, list[object]],
) -> object:
    if depth > _MAX_NESTING_DEPTH:
        raise FromScratchExclusionError("candidate nesting depth exceeds the cap")
    if isinstance(value, dict):
        _charge_structured_node(counters)
        normalized: dict[str, object] = {}
        containers.append(normalized)
        for key, item in value.items():
            if not isinstance(key, str):
                raise FromScratchExclusionError("candidate object key is not a string")
            normalized_key = _normalize_text(key)
            _charge_decoded_bytes(counters, normalized_key)
            if normalized_key in normalized:
                raise FromScratchExclusionError("candidate object has duplicate normalized keys")
            text_values.append(normalized_key)
            normalized_item = _normalize_candidate_value(
                item,
                depth=depth + 1,
                counters=counters,
                containers=containers,
                text_values=text_values,
                key_values=key_values,
            )
            normalized[normalized_key] = normalized_item
            key_values.setdefault(normalized_key, []).append(normalized_item)
        return normalized
    if isinstance(value, list):
        _charge_structured_node(counters)
        normalized_list: list[object] = []
        containers.append(normalized_list)
        normalized_list.extend(
            _normalize_candidate_value(
                item,
                depth=depth + 1,
                counters=counters,
                containers=containers,
                text_values=text_values,
                key_values=key_values,
            )
            for item in value
        )
        return normalized_list
    if isinstance(value, str):
        counters.string_leaves += 1
        if counters.string_leaves > _MAX_STRING_LEAVES:
            raise FromScratchExclusionError("candidate string leaves exceed the cap")
        normalized_string = _normalize_text(value)
        _charge_decoded_bytes(counters, normalized_string)
        text_values.append(normalized_string)
        return normalized_string
    if value is None or isinstance(value, bool):
        return value
    if type(value) is int:
        return value
    if type(value) is float and math.isfinite(value):
        return value
    raise FromScratchExclusionError("candidate value type is unsupported")


def _charge_structured_node(counters: _TraversalCounters) -> None:
    counters.structured_nodes += 1
    if counters.structured_nodes > _MAX_STRUCTURED_NODES:
        raise FromScratchExclusionError("candidate structured nodes exceed the cap")


def _charge_decoded_bytes(counters: _TraversalCounters, value: str) -> None:
    size = len(value.encode("utf-8"))
    if size > _MAX_SCALAR_UTF8_BYTES:
        raise FromScratchExclusionError("candidate scalar exceeds the byte cap")
    counters.decoded_bytes += size
    if counters.decoded_bytes > _MAX_DECODED_ROW_BYTES:
        raise FromScratchExclusionError("candidate decoded row exceeds the byte cap")


def _preflight_containment_view(value: str, counters: _TraversalCounters) -> None:
    _preflight_containment_fragments((value,), counters)


def _preflight_containment_range(
    value: str, start: int, end: int, counters: _TraversalCounters
) -> None:
    _start_generated_view(counters)
    output_started = False
    pending_space = False
    for index in range(start, end):
        character = value[index]
        if character in _WHITE_SPACE:
            if output_started:
                pending_space = True
            continue
        if pending_space:
            _charge_generated_view_bytes(counters, 1)
            pending_space = False
        _charge_generated_view_bytes(counters, len(character.encode("utf-8")))
        output_started = True


def _iter_line_bounds(value: str) -> Iterator[tuple[int, int]]:
    start = 0
    while True:
        end = value.find("\n", start)
        if end < 0:
            yield start, len(value)
            return
        yield start, end
        start = end + 1


def _preflight_containment_fragments(
    fragments: tuple[str, ...], counters: _TraversalCounters
) -> None:
    _start_generated_view(counters)
    output_started = False
    pending_space = False
    for fragment_index, fragment in enumerate(fragments):
        if fragment_index:
            pending_space = output_started
        for character in fragment:
            if character in _WHITE_SPACE:
                if output_started:
                    pending_space = True
                continue
            if pending_space:
                _charge_generated_view_bytes(counters, 1)
                pending_space = False
            _charge_generated_view_bytes(counters, len(character.encode("utf-8")))
            output_started = True


def _preflight_json_view(
    value: object,
    counters: _TraversalCounters,
    *,
    containment: bool,
    per_view_byte_cap: int | None = None,
) -> None:
    _start_generated_view(counters)
    per_view_bytes = 0
    output_started = False
    pending_space = False
    for fragment in _canonical_json_fragments(value):
        if containment:
            for character in fragment:
                if character in _WHITE_SPACE:
                    if output_started:
                        pending_space = True
                    continue
                if pending_space:
                    per_view_bytes += 1
                    _charge_generated_view_bytes(counters, 1)
                    pending_space = False
                encoded_size = len(character.encode("utf-8"))
                per_view_bytes += encoded_size
                _charge_generated_view_bytes(counters, encoded_size)
                output_started = True
        else:
            encoded_size = len(fragment.encode("utf-8"))
            per_view_bytes += encoded_size
            _charge_generated_view_bytes(counters, encoded_size)
        if per_view_byte_cap is not None and per_view_bytes > per_view_byte_cap:
            raise FromScratchExclusionError("candidate decoded row exceeds the byte cap")


def _start_generated_view(counters: _TraversalCounters) -> None:
    counters.generated_count += 1
    if counters.generated_count > _MAX_GENERATED_VIEW_COUNT:
        raise FromScratchExclusionError("candidate generated view count exceeds the cap")


def _charge_generated_view_bytes(counters: _TraversalCounters, size: int) -> None:
    counters.generated_bytes += size
    if counters.generated_bytes > _MAX_GENERATED_VIEW_BYTES:
        raise FromScratchExclusionError("candidate generated views exceed the byte cap")


def _record_view(
    kind: str,
    value: str,
    *,
    counters: _TraversalCounters,
    seen: set[tuple[str, int, str]],
) -> bool:
    size = len(value.encode("utf-8"))
    identity = (kind, size, sha256(value.encode("utf-8")).hexdigest())
    if identity not in seen:
        seen.add(identity)
        counters.unique_bytes += size
        if counters.unique_bytes > _MAX_UNIQUE_SCANNED_VIEW_BYTES:
            raise FromScratchExclusionError("candidate unique scanned views exceed the byte cap")
        return True
    return False


def _append_unique_view(
    output: list[str],
    kind: str,
    value: str,
    *,
    counters: _TraversalCounters,
    seen: set[tuple[str, int, str]],
) -> None:
    if _record_view(kind, value, counters=counters, seen=seen):
        output.append(value)


def _append_joined_candidate_views(
    projection: dict[str, object],
    output: list[str],
    *,
    counters: _TraversalCounters,
    seen: set[tuple[str, int, str]],
) -> None:
    messages = projection.get("messages")
    tools = projection.get("tools")
    source_assistant = projection.get("source_assistant")
    if not isinstance(messages, list) or not isinstance(tools, list):
        raise FromScratchExclusionError("candidate joined-view projection is invalid")

    message_views: list[str] = []
    for message in messages:
        if not isinstance(message, dict):
            raise FromScratchExclusionError("candidate joined-view message is invalid")
        joined = _joined_containment_view(
            tuple(_string_leaf_values(message.get("content"))), counters
        )
        if joined:
            message_views.append(joined)
            _append_unique_view(
                output,
                "message-joined",
                joined,
                counters=counters,
                seen=seen,
            )
    if source_assistant is not None:
        if not isinstance(source_assistant, dict):
            raise FromScratchExclusionError("candidate joined-view assistant is invalid")
        joined = _joined_containment_view(
            tuple(_string_leaf_values(source_assistant.get("content"))), counters
        )
        if joined:
            message_views.append(joined)
            _append_unique_view(
                output,
                "message-joined",
                joined,
                counters=counters,
                seen=seen,
            )

    for tool in tools:
        if not isinstance(tool, dict):
            raise FromScratchExclusionError("candidate joined-view tool is invalid")
        joined = _joined_containment_view(tuple(_string_leaf_values(tool)), counters)
        if joined:
            _append_unique_view(
                output,
                "tool-joined",
                joined,
                counters=counters,
                seen=seen,
            )

    conversation = _joined_containment_view(tuple(message_views), counters)
    if conversation:
        _append_unique_view(
            output,
            "conversation-joined",
            conversation,
            counters=counters,
            seen=seen,
        )
    projected = _joined_containment_view(tuple(_string_leaf_values(projection)), counters)
    if projected:
        _append_unique_view(
            output,
            "projection-joined",
            projected,
            counters=counters,
            seen=seen,
        )


def _joined_containment_view(parts: tuple[str, ...], counters: _TraversalCounters) -> str:
    _preflight_containment_fragments(parts, counters)
    normalized_parts = tuple(
        normalized for part in parts if (normalized := _containment_text(part))
    )
    return " ".join(normalized_parts)


def _string_leaf_values(value: object) -> Iterator[str]:
    pending = [value]
    while pending:
        current = pending.pop()
        if isinstance(current, str):
            yield current
        elif isinstance(current, dict):
            pending.extend(reversed(tuple(current.values())))
        elif isinstance(current, list):
            pending.extend(reversed(current))


def _atom_matches(atom: HeldoutV2Atom, views: _CandidateViews, budget: _MatcherBudget) -> bool:
    if atom.shape_id == 0:
        return _match_equality(atom, (*views.text, *views.structured), budget)
    if atom.shape_id == 1:
        return _match_predicate(atom, views.text, budget, _lexical_contains)
    if atom.shape_id == 2:
        return _match_predicate(atom, views.text, budget, _answer_context_contains)
    if atom.shape_id == 3:
        return _match_predicate(
            atom, (*views.text, *views.structured), budget, lambda view, value: value in view
        )
    if atom.shape_id == 4:
        return _match_equality(atom, views.whole_lines, budget)
    if atom.shape_id == 5:
        return _match_predicate(atom, views.structured, budget, lambda view, value: value in view)
    if atom.shape_id == 6:
        return _match_predicate(atom, views.text, budget, lambda view, value: value in view)
    if atom.shape_id == 7:
        return _match_equality(atom, views.key_values.get(atom.field_key, ()), budget)
    if atom.shape_id == 10:
        budget.charge(views.raw_record, atom)
        return views.raw_record == atom.value
    raise FromScratchExclusionError("fixture atom shape is unsupported")


def _match_equality(atom: HeldoutV2Atom, views: tuple[str, ...], budget: _MatcherBudget) -> bool:
    for view in views:
        budget.charge(view, atom)
        if view == atom.value:
            return True
    return False


def _match_predicate(
    atom: HeldoutV2Atom,
    views: tuple[str, ...],
    budget: _MatcherBudget,
    predicate: Callable[[str, str], bool],
) -> bool:
    for view in views:
        budget.charge(view, atom)
        if predicate(view, atom.value):
            return True
    return False


def _validate_full_shape_contract(atom: HeldoutV2Atom) -> None:
    alphanumeric_count = sum(
        unicodedata.category(character).startswith(("L", "N")) for character in atom.value
    )
    non_whitespace_count = sum(character not in _WHITE_SPACE for character in atom.value)
    numeric_answer = _NUMERIC_ANSWER.fullmatch(atom.value) is not None
    allowed = False
    if atom.shape_id == 0:
        allowed = atom.domain_id in range(7)
    elif atom.shape_id == 1:
        allowed = (atom.domain_id in {0, 6} and alphanumeric_count >= 2) or (
            atom.domain_id == 1 and alphanumeric_count >= 2 and not numeric_answer
        )
    elif atom.shape_id == 2:
        allowed = atom.domain_id == 1 and (numeric_answer or alphanumeric_count < 2)
    elif atom.shape_id == 3:
        allowed = atom.domain_id in {2, 3, 4} and non_whitespace_count >= 4
    elif atom.shape_id == 4:
        allowed = (atom.domain_id in {2, 3, 4} and non_whitespace_count < 4) or (
            atom.domain_id == 6 and alphanumeric_count < 2
        )
    elif atom.shape_id == 5:
        allowed = atom.domain_id == 5 and _is_canonical_structured_json(atom.value)
    elif atom.shape_id == 6:
        allowed = atom.domain_id == 5 and len(atom.value.split(" ")) >= 2
    elif atom.shape_id == 7:
        allowed = atom.domain_id == 5
    elif atom.shape_id == 10:
        allowed = atom.domain_id == 7
    if not allowed:
        raise FromScratchExclusionError(
            "held-out fixture atom violates the frozen domain/shape contract"
        )


def _is_canonical_structured_json(value: str) -> bool:
    try:
        parsed = _load_canonical_value(value.encode("utf-8"), label="held-out tool atom")
    except FromScratchExclusionError:
        return False
    return isinstance(parsed, (dict, list)) and _canonical_json(parsed).decode("utf-8") == value


def _lexical_contains(view: str, value: str) -> bool:
    start = 0
    while (position := view.find(value, start)) >= 0:
        before = view[position - 1] if position else None
        end = position + len(value)
        after = view[end] if end < len(view) else None
        if _is_lexical_boundary(before) and _is_lexical_boundary(after):
            return True
        start = position + 1
    return False


def _is_lexical_boundary(character: str | None) -> bool:
    return character is None or (
        not unicodedata.category(character).startswith(("L", "N")) and character != "_"
    )


def _answer_context_contains(view: str, value: str) -> bool:
    contexts = (
        ("#### ", ""),
        ("answer ", ""),
        ("answer: ", ""),
        ("answer is ", ""),
        ("Answer ", ""),
        ("Answer: ", ""),
        ("Answer is ", ""),
        ("final answer ", ""),
        ("final answer: ", ""),
        ("Final answer ", ""),
        ("Final answer: ", ""),
        ("\\boxed{", "}"),
    )
    for prefix, suffix in contexts:
        template = f"{prefix}{value}{suffix}"
        # A prefix that opens with a letter is a word ("answer", "final answer"),
        # so it needs its own left boundary: without one, "unanswer: 42" contains
        # "answer: 42" and the value-side check passes vacuously, since the char
        # before the value is the prefix's own trailing space.
        prefix_is_lexical = bool(prefix) and not _is_lexical_boundary(prefix[0])
        start = 0
        while (position := view.find(template, start)) >= 0:
            value_start = position + len(prefix)
            value_end = value_start + len(value)
            head = view[position - 1] if position else None
            before = view[value_start - 1] if value_start else None
            after = view[value_end] if value_end < len(view) else None
            if (
                (not prefix_is_lexical or _is_lexical_boundary(head))
                and _is_lexical_boundary(before)
                and _is_lexical_boundary(after)
            ):
                return True
            start = position + 1
    return False


def _validate_component_order(
    value: Mapping[CatalogName, tuple[str, ...]],
) -> dict[CatalogName, tuple[str, ...]]:
    if not isinstance(value, Mapping) or tuple(value) != HELDOUT_V2_CATALOG_ORDER:
        raise FromScratchExclusionError("held-out component order is invalid")
    validated: dict[CatalogName, tuple[str, ...]] = {}
    for catalog in HELDOUT_V2_CATALOG_ORDER:
        components = value[catalog]
        _validate_component_ids(components)
        validated[catalog] = components
    return validated


def _validate_identity_attributions(
    value: Mapping[str, tuple[HeldoutV2Attribution, ...]],
    *,
    component_order: Mapping[CatalogName, tuple[str, ...]],
    label: str,
) -> dict[str, tuple[HeldoutV2Attribution, ...]]:
    if not isinstance(value, Mapping):
        raise FromScratchExclusionError(f"held-out {label} attribution is invalid")
    validated: dict[str, tuple[HeldoutV2Attribution, ...]] = {}
    for identity, attributions in value.items():
        _require_sha256(identity, label=f"held-out {label} identity")
        if (
            not isinstance(attributions, tuple)
            or not attributions
            or any(not isinstance(item, HeldoutV2Attribution) for item in attributions)
        ):
            raise FromScratchExclusionError(f"held-out {label} attribution is invalid")
        expected_order = tuple(
            sorted(attributions, key=lambda item: HELDOUT_V2_CATALOG_ORDER.index(item.catalog))
        )
        if attributions != expected_order or len({item.catalog for item in attributions}) != len(
            attributions
        ):
            raise FromScratchExclusionError(f"held-out {label} attribution is not in fixed order")
        for attribution in attributions:
            _require_component_subsequence(
                attribution.contributing_component_ids,
                component_order[attribution.catalog],
            )
        validated[identity] = attributions
    return validated


def _validate_atoms(
    value: tuple[HeldoutV2Atom, ...],
    *,
    component_order: Mapping[CatalogName, tuple[str, ...]],
) -> tuple[tuple[HeldoutV2Atom, ...], dict[CatalogName, tuple[HeldoutV2Atom, ...]]]:
    if not isinstance(value, tuple) or any(not isinstance(atom, HeldoutV2Atom) for atom in value):
        raise FromScratchExclusionError("held-out atom index is invalid")
    counts: dict[tuple[CatalogName, int, int], int] = {}
    byte_counts = dict.fromkeys(HELDOUT_V2_CATALOG_ORDER, 0)
    by_catalog: dict[CatalogName, list[HeldoutV2Atom]] = {
        catalog: [] for catalog in HELDOUT_V2_CATALOG_ORDER
    }
    # Caps run before the order check: the order check materializes an order key
    # per atom, so an over-cap index would pay for the whole index to be told it
    # is over the cap.
    for atom in value:
        _require_component_subsequence(
            atom.contributing_component_ids, component_order[atom.catalog]
        )
        count_key = (atom.catalog, atom.rule_id, atom.shape_id)
        counts[count_key] = counts.get(count_key, 0) + 1
        if counts[count_key] > _MAX_ATOMS_PER_CATALOG_RULE_SHAPE:
            raise FromScratchExclusionError("held-out atom count exceeds the cap")
        byte_counts[atom.catalog] += (
            45 + len(atom.field_key.encode("utf-8")) + len(atom.value.encode("utf-8"))
        )
        if byte_counts[atom.catalog] > _MAX_ATOM_BYTES_PER_CATALOG:
            raise FromScratchExclusionError("held-out atom bytes exceed the catalog cap")
        by_catalog[atom.catalog].append(atom)
    expected = tuple(sorted(value, key=_atom_order_key))
    if value != expected or len({_atom_order_key(atom) for atom in value}) != len(value):
        raise FromScratchExclusionError("held-out atoms are duplicate or not in fixed order")
    return value, {catalog: tuple(by_catalog[catalog]) for catalog in HELDOUT_V2_CATALOG_ORDER}


def _atom_order_key(atom: HeldoutV2Atom) -> tuple[object, ...]:
    return (
        HELDOUT_V2_CATALOG_ORDER.index(atom.catalog),
        atom.domain_id,
        atom.rule_id,
        atom.shape_id,
        atom.field_key.encode("utf-8"),
        atom.digest,
        atom.value.encode("utf-8"),
    )


def _validate_component_ids(value: object) -> None:
    if not isinstance(value, tuple) or not value or len(value) != len(set(value)):
        raise FromScratchExclusionError("held-out contributor component IDs are invalid")
    for component_id in value:
        if (
            not isinstance(component_id, str)
            or not component_id
            or component_id != component_id.lower()
        ):
            raise FromScratchExclusionError("held-out contributor component ID is invalid")
        try:
            encoded = component_id.encode("ascii")
        except UnicodeEncodeError as error:
            raise FromScratchExclusionError(
                "held-out contributor component ID is not lowercase ASCII"
            ) from error
        if any(byte <= 0x20 or byte >= 0x7F for byte in encoded):
            raise FromScratchExclusionError(
                "held-out contributor component ID is not lowercase ASCII"
            )


def _require_component_subsequence(value: tuple[str, ...], order: tuple[str, ...]) -> None:
    positions = {component: index for index, component in enumerate(order)}
    try:
        observed = tuple(positions[component] for component in value)
    except KeyError as error:
        raise FromScratchExclusionError(
            "held-out contributor is absent from component order"
        ) from error
    if observed != tuple(sorted(observed)):
        raise FromScratchExclusionError("held-out contributors are not in component order")


def _read_authenticated_receipt(path: Path, expected_sha256: str) -> bytes:
    _require_sha256(expected_sha256, label="expected held-out receipt")
    if not isinstance(path, Path) or not path.is_absolute():
        raise FromScratchExclusionError("held-out root must be an absolute explicit receipt file")
    file_descriptor, directory_descriptors, edge_names, opened = _open_componentwise(path)
    try:
        if not stat.S_ISREG(opened.st_mode) or opened.st_nlink != 1:
            raise FromScratchExclusionError(
                "held-out root must be an explicit receipt file and single-link regular file"
            )
        if opened.st_size > _MAX_RECEIPT_BYTES:
            raise FromScratchExclusionError("held-out root receipt exceeds the byte cap")
        retained = bytearray()
        while True:
            block = os.read(
                file_descriptor,
                min(1024 * 1024, _MAX_RECEIPT_BYTES + 1 - len(retained)),
            )
            if not block:
                break
            retained.extend(block)
            if len(retained) > _MAX_RECEIPT_BYTES:
                raise FromScratchExclusionError("held-out root receipt exceeds the byte cap")
        after = os.fstat(file_descriptor)
        final_named = os.stat(
            path.name,
            dir_fd=directory_descriptors[-1],
            follow_symlinks=False,
        )
        if (
            _file_identity(opened) != _file_identity(after)
            or _file_identity(after) != _file_identity(final_named)
            or len(retained) != opened.st_size
        ):
            raise FromScratchExclusionError("held-out root receipt changed while reading")
        _verify_directory_edges(directory_descriptors, edge_names)
    except FromScratchExclusionError:
        raise
    except OSError as error:
        raise FromScratchExclusionError("held-out root receipt changed while reading") from error
    finally:
        os.close(file_descriptor)
        for descriptor in reversed(directory_descriptors):
            os.close(descriptor)
    raw = bytes(retained)
    if sha256(raw).hexdigest() != expected_sha256:
        raise FromScratchExclusionError("held-out root receipt whole-file SHA-256 mismatch")
    return raw


def _open_componentwise(
    path: Path,
) -> tuple[int, list[int], list[str], os.stat_result]:
    raw = os.fspath(path)
    if not raw.startswith("/") or raw == "/" or "//" in raw:
        raise FromScratchExclusionError(
            "held-out explicit receipt contains a symlink or unsafe path component"
        )
    components = raw.split("/")[1:]
    if not components or any(component in {"", ".", ".."} for component in components):
        raise FromScratchExclusionError(
            "held-out explicit receipt contains a symlink or unsafe path component"
        )
    nofollow = getattr(os, "O_NOFOLLOW", 0)
    if not isinstance(nofollow, int) or nofollow == 0:
        raise FromScratchExclusionError("held-out receipt loading requires O_NOFOLLOW")
    directory_flags = (
        os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | nofollow | getattr(os, "O_CLOEXEC", 0)
    )
    file_flags = os.O_RDONLY | nofollow | getattr(os, "O_NONBLOCK", 0) | getattr(os, "O_CLOEXEC", 0)
    descriptors = [os.open("/", directory_flags)]
    edge_names: list[str] = []
    file_descriptor: int | None = None
    try:
        for component in components[:-1]:
            named = os.stat(component, dir_fd=descriptors[-1], follow_symlinks=False)
            child = os.open(component, directory_flags, dir_fd=descriptors[-1])
            opened = os.fstat(child)
            if not stat.S_ISDIR(named.st_mode) or _directory_identity(named) != _directory_identity(
                opened
            ):
                os.close(child)
                raise FromScratchExclusionError(
                    "held-out explicit receipt contains a symlink or unsafe path component"
                )
            descriptors.append(child)
            edge_names.append(component)
        named_file = os.stat(components[-1], dir_fd=descriptors[-1], follow_symlinks=False)
        file_descriptor = os.open(components[-1], file_flags, dir_fd=descriptors[-1])
        opened_file = os.fstat(file_descriptor)
        if _file_identity(named_file) != _file_identity(opened_file):
            raise FromScratchExclusionError("held-out explicit receipt changed while opening")
        return file_descriptor, descriptors, edge_names, opened_file
    except FromScratchExclusionError:
        if file_descriptor is not None:
            os.close(file_descriptor)
        for descriptor in reversed(descriptors):
            os.close(descriptor)
        raise
    except OSError as error:
        if file_descriptor is not None:
            os.close(file_descriptor)
        for descriptor in reversed(descriptors):
            os.close(descriptor)
        raise FromScratchExclusionError(
            "held-out explicit receipt contains a symlink or unsafe path component"
        ) from error


def _verify_directory_edges(descriptors: list[int], edge_names: list[str]) -> None:
    for parent, child, edge_name in zip(descriptors[:-1], descriptors[1:], edge_names, strict=True):
        named = os.stat(edge_name, dir_fd=parent, follow_symlinks=False)
        opened = os.fstat(child)
        if not stat.S_ISDIR(named.st_mode) or _directory_identity(named) != _directory_identity(
            opened
        ):
            raise FromScratchExclusionError("held-out receipt directory changed while reading")


def _load_canonical_object(raw: bytes) -> dict[str, Any]:
    payload = _load_canonical_value(raw.removesuffix(b"\n"), label="held-out root receipt")
    if not isinstance(payload, dict) or raw != _canonical_json(payload) + b"\n":
        raise FromScratchExclusionError("held-out root receipt is not canonical JSON")
    return payload


def _load_canonical_value(raw: bytes, *, label: str) -> Any:
    def reject_duplicate_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        parsed: dict[str, Any] = {}
        for key, value in pairs:
            if key in parsed:
                raise FromScratchExclusionError(f"{label} has duplicate JSON keys")
            parsed[key] = value
        return parsed

    def reject_nonfinite(value: str) -> None:
        raise FromScratchExclusionError(f"{label} has a nonfinite JSON value")

    try:
        payload: Any = json.loads(
            raw,
            object_pairs_hook=reject_duplicate_pairs,
            parse_constant=reject_nonfinite,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise FromScratchExclusionError(f"{label} JSON is invalid") from error
    if raw != _canonical_json(payload):
        raise FromScratchExclusionError(f"{label} is not canonical JSON")
    return payload


def _validate_v2_bundle_receipt(payload: dict[str, Any]) -> None:
    expected_fields = {
        "schema",
        "scientific_identity",
        "proposal_sha256",
        "catalog_receipt_descriptors",
        "catalog_order",
        "union_prompt_count",
        "union_prompt_sha256",
        "union_attribution_sha256",
        "self_sha256",
    }
    if set(payload) != expected_fields:
        raise FromScratchExclusionError("held-out V2 bundle fields are incomplete or unknown")
    if payload["scientific_identity"] != FROZEN_SCIENTIFIC_IDENTITY:
        raise FromScratchExclusionError("held-out V2 scientific identity is invalid")
    _require_sha256(payload["proposal_sha256"], label="held-out V2 proposal")
    _require_sha256(payload["union_prompt_sha256"], label="held-out V2 prompt union")
    _require_sha256(payload["union_attribution_sha256"], label="held-out V2 attribution union")
    if payload["catalog_order"] != list(HELDOUT_V2_CATALOG_ORDER):
        raise FromScratchExclusionError(
            "held-out V2 bundle must contain catalogs speed,math,code,swe,tool in fixed order"
        )
    descriptors = payload["catalog_receipt_descriptors"]
    if not isinstance(descriptors, list) or len(descriptors) != len(HELDOUT_V2_CATALOG_ORDER):
        raise FromScratchExclusionError(
            "held-out V2 bundle must contain catalogs speed,math,code,swe,tool exactly once"
        )
    for descriptor in descriptors:
        _validate_artifact_descriptor(descriptor)
    union_prompt_count = payload["union_prompt_count"]
    if type(union_prompt_count) is not int or union_prompt_count <= 0:
        raise FromScratchExclusionError("held-out V2 prompt union must be nonempty")
    self_sha256 = payload["self_sha256"]
    _require_sha256(self_sha256, label="held-out V2 bundle self hash")
    body = {key: value for key, value in payload.items() if key != "self_sha256"}
    if sha256(_canonical_json(body)).hexdigest() != self_sha256:
        raise FromScratchExclusionError("held-out V2 bundle self hash does not reconcile")


def _validate_artifact_descriptor(value: object) -> None:
    if not isinstance(value, dict) or set(value) != {
        "path",
        "bytes",
        "sha256",
        "media_type",
    }:
        raise FromScratchExclusionError("held-out V2 catalog receipt descriptor is invalid")
    relative_path = value["path"]
    if not isinstance(relative_path, str) or not relative_path:
        raise FromScratchExclusionError("held-out V2 descriptor path is invalid")
    components = relative_path.split("/")
    if (
        relative_path.startswith("/")
        or any(component in {"", ".", ".."} for component in components)
        or "/".join(components) != relative_path
    ):
        raise FromScratchExclusionError("held-out V2 descriptor path is not canonical relative")
    if type(value["bytes"]) is not int or value["bytes"] < 0:
        raise FromScratchExclusionError("held-out V2 descriptor byte count is invalid")
    _require_sha256(value["sha256"], label="held-out V2 descriptor")
    if not isinstance(value["media_type"], str) or not value["media_type"]:
        raise FromScratchExclusionError("held-out V2 descriptor media type is invalid")


def _require_unicode_version() -> None:
    if unicodedata.unidata_version != "15.1.0":
        raise FromScratchExclusionError("held-out normalization requires Unicode 15.1.0")


def _normalize_text(value: str) -> str:
    _require_unicode_version()
    if "\x00" in value or any(0xD800 <= ord(character) <= 0xDFFF for character in value):
        raise FromScratchExclusionError("held-out text contains invalid Unicode")
    return unicodedata.normalize("NFC", value.replace("\r\n", "\n").replace("\r", "\n"))


def _containment_text(value: str) -> str:
    normalized = _normalize_text(value)
    output: list[str] = []
    in_whitespace = False
    for character in normalized:
        if character in _WHITE_SPACE:
            if output:
                in_whitespace = True
            continue
        if in_whitespace:
            output.append(" ")
            in_whitespace = False
        output.append(character)
    return "".join(output)


def _content_digest(
    *, domain_id: int, rule_id: int, shape_id: int, field_key: str, value: str
) -> str:
    return sha256(
        _canonical_json(
            [
                "q30t-held-out-content-v2",
                domain_id,
                rule_id,
                shape_id,
                field_key,
                value,
            ]
        )
    ).hexdigest()


def _canonical_json(value: object) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode(
        "utf-8"
    )


def _canonical_json_fragments(value: object) -> Iterator[str]:
    encoder = json.JSONEncoder(
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    return encoder.iterencode(value)


def _require_catalog(value: object) -> None:
    if value not in HELDOUT_V2_CATALOG_ORDER:
        raise FromScratchExclusionError("held-out catalog is invalid")


def _is_sha256(value: object) -> bool:
    return (
        type(value) is str
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _require_sha256(value: object, *, label: str) -> None:
    if not _is_sha256(value):
        raise FromScratchExclusionError(f"{label} SHA-256 is invalid")


def _file_identity(
    status: os.stat_result,
) -> tuple[int, int, int, int, int, int, int]:
    return (
        status.st_dev,
        status.st_ino,
        status.st_mode,
        status.st_nlink,
        status.st_size,
        status.st_mtime_ns,
        status.st_ctime_ns,
    )


def _directory_identity(status: os.stat_result) -> tuple[int, int, int]:
    return status.st_dev, status.st_ino, stat.S_IFMT(status.st_mode)
