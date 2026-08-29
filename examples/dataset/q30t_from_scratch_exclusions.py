# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Fail-closed held-out V2 exclusion contracts for Q30 corpus candidates."""

from __future__ import annotations

import json
import os
import stat
import unicodedata
from collections.abc import Mapping
from dataclasses import dataclass, field
from hashlib import sha256
from pathlib import Path, PurePosixPath
from types import MappingProxyType
from typing import Any, Literal, cast

from q30t_from_scratch_candidates import PromptCandidate

__all__ = [
    "HELDOUT_V2_CATALOG_ORDER",
    "CandidateExclusion",
    "ExclusionLedger",
    "FromScratchExclusionError",
    "HeldoutV2Bundle",
    "apply_heldout_union",
    "load_approved_heldout_v2",
]

CatalogName = Literal["speed", "math", "code", "swe", "tool"]
ExclusionReason = Literal["prompt_uuid", "exact_content", "heldout_content_containment"]

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
_MAX_SCALAR_UTF8_BYTES = 16 * 1024 * 1024
_MAX_STRING_LEAVES = 65_536
_MAX_NESTING_DEPTH = 64
_MAX_CANDIDATE_VIEW_BYTES = 256 * 1024 * 1024
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
class HeldoutV2Bundle:
    """Validated in-memory identities; only the production loader can grant trust."""

    receipt_sha256: str
    catalog_names: tuple[CatalogName, ...]
    prompt_uuids: frozenset[str]
    exact_content_sha256s: frozenset[str]
    containment_index: Mapping[str, tuple[CatalogName, ...]] = field(repr=False)
    approval_identity_sha256: str
    prompt_uuid_catalogs: Mapping[str, tuple[CatalogName, ...]] = field(
        default_factory=dict, repr=False
    )
    exact_content_catalogs: Mapping[str, tuple[CatalogName, ...]] = field(
        default_factory=dict, repr=False
    )

    def __post_init__(self) -> None:
        _require_sha256(self.receipt_sha256, label="held-out receipt")
        _require_sha256(self.approval_identity_sha256, label="held-out approval identity")
        if self.catalog_names != HELDOUT_V2_CATALOG_ORDER:
            raise FromScratchExclusionError(
                "held-out bundle must contain catalogs speed,math,code,swe,tool in fixed order"
            )
        if not isinstance(self.prompt_uuids, frozenset) or any(
            not _is_sha256(value) for value in self.prompt_uuids
        ):
            raise FromScratchExclusionError("held-out prompt UUID identities are invalid")
        if not isinstance(self.exact_content_sha256s, frozenset) or any(
            not _is_sha256(value) for value in self.exact_content_sha256s
        ):
            raise FromScratchExclusionError("held-out exact-content identities are invalid")

        prompt_catalogs = _validate_identity_catalogs(
            self.prompt_uuid_catalogs, label="prompt UUID"
        )
        exact_catalogs = _validate_identity_catalogs(
            self.exact_content_catalogs, label="exact content"
        )
        if frozenset(prompt_catalogs) != self.prompt_uuids:
            raise FromScratchExclusionError(
                "held-out prompt UUID identities lack exact catalog attribution"
            )
        if frozenset(exact_catalogs) != self.exact_content_sha256s:
            raise FromScratchExclusionError(
                "held-out exact-content identities lack exact catalog attribution"
            )
        containment = _validate_containment_index(self.containment_index)

        populated = {
            catalog
            for catalogs in (
                *prompt_catalogs.values(),
                *exact_catalogs.values(),
                *containment.values(),
            )
            for catalog in catalogs
        }
        if populated != set(HELDOUT_V2_CATALOG_ORDER):
            raise FromScratchExclusionError(
                "held-out catalogs speed,math,code,swe,tool must each be approved and nonempty"
            )
        object.__setattr__(self, "prompt_uuid_catalogs", MappingProxyType(prompt_catalogs))
        object.__setattr__(self, "exact_content_catalogs", MappingProxyType(exact_catalogs))
        object.__setattr__(self, "containment_index", MappingProxyType(containment))


@dataclass(frozen=True)
class CandidateExclusion:
    """Stable, raw-content-free evidence for one excluded candidate."""

    prompt_uuid: str
    catalog: CatalogName
    reason: ExclusionReason
    matched_identity: str

    def __post_init__(self) -> None:
        _require_sha256(self.prompt_uuid, label="excluded prompt UUID")
        _require_sha256(self.matched_identity, label="matched held-out identity")
        if self.catalog not in HELDOUT_V2_CATALOG_ORDER:
            raise FromScratchExclusionError("excluded catalog is invalid")
        if self.reason not in {
            "prompt_uuid",
            "exact_content",
            "heldout_content_containment",
        }:
            raise FromScratchExclusionError("excluded reason is invalid")


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


def load_approved_heldout_v2(root: Path, *, expected_receipt_sha256: str) -> HeldoutV2Bundle:
    """Authenticate an explicit root receipt and fail closed without V2 approval support.

    The checked-in producer emits only legacy V1 receipts. The V2 design does
    not yet supply a producer-owned root layout or a trusted keyring verifier,
    so no authenticated receipt can currently be promoted from state P here.
    """
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
    """Apply prompt, full-row, then bounded exact containment exclusions."""
    if not isinstance(candidate, PromptCandidate):
        raise FromScratchExclusionError("held-out candidate type is invalid")
    if not isinstance(heldout, HeldoutV2Bundle):
        raise FromScratchExclusionError("held-out bundle type is invalid")

    prompt_identity = candidate.identity.prompt_uuid
    prompt_catalogs = heldout.prompt_uuid_catalogs.get(prompt_identity)
    if prompt_catalogs is not None:
        return CandidateExclusion(
            prompt_uuid=prompt_identity,
            catalog=prompt_catalogs[0],
            reason="prompt_uuid",
            matched_identity=prompt_identity,
        )

    exact_identity = candidate.normalized_row_sha256
    exact_catalogs = heldout.exact_content_catalogs.get(exact_identity)
    if exact_catalogs is not None:
        return CandidateExclusion(
            prompt_uuid=prompt_identity,
            catalog=exact_catalogs[0],
            reason="exact_content",
            matched_identity=exact_identity,
        )

    views = _candidate_containment_views(candidate)
    entries = sorted(
        heldout.containment_index.items(),
        key=lambda item: (
            HELDOUT_V2_CATALOG_ORDER.index(item[1][0]),
            item[0].encode("utf-8"),
        ),
    )
    for protected, catalogs in entries:
        if any(protected in view for view in views):
            return CandidateExclusion(
                prompt_uuid=prompt_identity,
                catalog=catalogs[0],
                reason="heldout_content_containment",
                matched_identity=sha256(protected.encode("utf-8")).hexdigest(),
            )
    return None


def _validate_identity_catalogs(
    value: Mapping[str, tuple[CatalogName, ...]], *, label: str
) -> dict[str, tuple[CatalogName, ...]]:
    if not isinstance(value, Mapping):
        raise FromScratchExclusionError(f"held-out {label} attribution is invalid")
    validated: dict[str, tuple[CatalogName, ...]] = {}
    for identity, catalogs in value.items():
        if not _is_sha256(identity):
            raise FromScratchExclusionError(f"held-out {label} identity is invalid")
        validated[identity] = _validate_catalog_tuple(catalogs, label=label)
    return validated


def _validate_containment_index(
    value: Mapping[str, tuple[CatalogName, ...]],
) -> dict[str, tuple[CatalogName, ...]]:
    if not isinstance(value, Mapping):
        raise FromScratchExclusionError("held-out containment index is invalid")
    validated: dict[str, tuple[CatalogName, ...]] = {}
    for protected, catalogs in value.items():
        if not isinstance(protected, str):
            raise FromScratchExclusionError("held-out containment value is invalid")
        normalized = _containment_text(protected)
        if not normalized or normalized != protected:
            raise FromScratchExclusionError(
                "held-out containment values must be nonempty and pre-normalized"
            )
        if len(protected.encode("utf-8")) > _MAX_SCALAR_UTF8_BYTES:
            raise FromScratchExclusionError("held-out containment value exceeds the byte cap")
        validated[protected] = _validate_catalog_tuple(catalogs, label="containment")
    return validated


def _validate_catalog_tuple(value: object, *, label: str) -> tuple[CatalogName, ...]:
    if not isinstance(value, tuple) or not value:
        raise FromScratchExclusionError(f"held-out {label} catalog attribution is empty")
    if any(
        not isinstance(catalog, str) or catalog not in HELDOUT_V2_CATALOG_ORDER for catalog in value
    ):
        raise FromScratchExclusionError(f"held-out {label} catalog attribution is invalid")
    catalogs = cast("tuple[CatalogName, ...]", value)
    if len(catalogs) != len(set(catalogs)) or catalogs != tuple(
        sorted(catalogs, key=HELDOUT_V2_CATALOG_ORDER.index)
    ):
        raise FromScratchExclusionError(
            f"held-out {label} catalog attribution is not in fixed order"
        )
    return catalogs


def _candidate_containment_views(candidate: PromptCandidate) -> tuple[str, ...]:
    try:
        identity_payload: Any = json.loads(candidate.identity.canonical_bytes)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise FromScratchExclusionError("candidate identity JSON is invalid") from error
    if not isinstance(identity_payload, dict) or set(identity_payload) != {
        "messages",
        "provenance",
        "tools",
    }:
        raise FromScratchExclusionError("candidate identity payload is invalid")
    scan_root = {
        "messages": identity_payload["messages"],
        "tools": identity_payload["tools"],
        "source_assistant": candidate.source_assistant,
    }
    stack: list[tuple[object, int]] = [(scan_root, 0)]
    views: list[str] = []
    total_bytes = 0
    while stack:
        value, depth = stack.pop()
        if depth > _MAX_NESTING_DEPTH:
            raise FromScratchExclusionError("candidate containment nesting depth exceeds the cap")
        if isinstance(value, dict):
            stack.extend((item, depth + 1) for item in reversed(tuple(value.values())))
        elif isinstance(value, list):
            stack.extend((item, depth + 1) for item in reversed(value))
        elif isinstance(value, str):
            if len(views) == _MAX_STRING_LEAVES:
                raise FromScratchExclusionError(
                    "candidate containment string leaves exceed the cap"
                )
            normalized = _containment_text(value)
            encoded_bytes = len(normalized.encode("utf-8"))
            if encoded_bytes > _MAX_SCALAR_UTF8_BYTES:
                raise FromScratchExclusionError("candidate containment scalar exceeds the byte cap")
            total_bytes += encoded_bytes
            if total_bytes > _MAX_CANDIDATE_VIEW_BYTES:
                raise FromScratchExclusionError("candidate containment views exceed the byte cap")
            if normalized:
                views.append(normalized)
        elif value is not None and not isinstance(value, (bool, int, float)):
            raise FromScratchExclusionError("candidate containment value type is unsupported")
    return tuple(views)


def _containment_text(value: str) -> str:
    if "\x00" in value or any(0xD800 <= ord(character) <= 0xDFFF for character in value):
        raise FromScratchExclusionError("held-out text contains invalid Unicode")
    normalized = unicodedata.normalize("NFC", value.replace("\r\n", "\n").replace("\r", "\n"))
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


def _read_authenticated_receipt(path: Path, expected_sha256: str) -> bytes:
    _require_sha256(expected_sha256, label="expected held-out receipt")
    if not isinstance(path, Path) or not path.is_absolute():
        raise FromScratchExclusionError("held-out root must be an absolute explicit receipt file")
    try:
        named = os.stat(path, follow_symlinks=False)
    except OSError as error:
        raise FromScratchExclusionError("held-out explicit receipt file is unavailable") from error
    if stat.S_ISDIR(named.st_mode):
        raise FromScratchExclusionError(
            "held-out V2 layout is unavailable; root must be an explicit receipt file"
        )
    if not stat.S_ISREG(named.st_mode) or named.st_nlink != 1:
        raise FromScratchExclusionError(
            "held-out explicit receipt must be a single-link regular file"
        )
    if named.st_size > _MAX_RECEIPT_BYTES:
        raise FromScratchExclusionError("held-out root receipt exceeds the byte cap")
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0)
    descriptor: int | None = None
    try:
        descriptor = os.open(path, flags)
        before = os.fstat(descriptor)
        if _file_identity(named) != _file_identity(before):
            raise FromScratchExclusionError("held-out root receipt changed while opening")
        retained = bytearray()
        while True:
            block = os.read(descriptor, min(1024 * 1024, _MAX_RECEIPT_BYTES + 1 - len(retained)))
            if not block:
                break
            retained.extend(block)
            if len(retained) > _MAX_RECEIPT_BYTES:
                raise FromScratchExclusionError("held-out root receipt exceeds the byte cap")
        after = os.fstat(descriptor)
        rebound = os.stat(path, follow_symlinks=False)
        if (
            _file_identity(before) != _file_identity(after)
            or _file_identity(after) != _file_identity(rebound)
            or len(retained) != before.st_size
        ):
            raise FromScratchExclusionError("held-out root receipt changed while reading")
    except FromScratchExclusionError:
        raise
    except OSError as error:
        raise FromScratchExclusionError("held-out explicit receipt file is unreadable") from error
    finally:
        if descriptor is not None:
            os.close(descriptor)
    raw = bytes(retained)
    if sha256(raw).hexdigest() != expected_sha256:
        raise FromScratchExclusionError("held-out root receipt whole-file SHA-256 mismatch")
    return raw


def _load_canonical_object(raw: bytes) -> dict[str, Any]:
    def reject_duplicate_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        parsed: dict[str, Any] = {}
        for key, value in pairs:
            if key in parsed:
                raise FromScratchExclusionError("held-out root receipt has duplicate JSON keys")
            parsed[key] = value
        return parsed

    def reject_nonfinite(value: str) -> None:
        raise FromScratchExclusionError(f"held-out root receipt has nonfinite JSON value {value}")

    try:
        payload: Any = json.loads(
            raw,
            object_pairs_hook=reject_duplicate_pairs,
            parse_constant=reject_nonfinite,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise FromScratchExclusionError("held-out root receipt JSON is invalid") from error
    if not isinstance(payload, dict) or raw != _canonical_json(payload) + b"\n":
        raise FromScratchExclusionError("held-out root receipt is not canonical JSON")
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
    if not isinstance(payload["scientific_identity"], str) or not payload["scientific_identity"]:
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
    if not isinstance(value, dict) or set(value) != {"path", "bytes", "sha256", "media_type"}:
        raise FromScratchExclusionError("held-out V2 catalog receipt descriptor is invalid")
    relative_path = value["path"]
    if not isinstance(relative_path, str) or not relative_path:
        raise FromScratchExclusionError("held-out V2 descriptor path is invalid")
    parsed_path = PurePosixPath(relative_path)
    if parsed_path.is_absolute() or ".." in parsed_path.parts or "." in parsed_path.parts:
        raise FromScratchExclusionError("held-out V2 descriptor path is not canonical relative")
    if type(value["bytes"]) is not int or value["bytes"] < 0:
        raise FromScratchExclusionError("held-out V2 descriptor byte count is invalid")
    _require_sha256(value["sha256"], label="held-out V2 descriptor")
    if not isinstance(value["media_type"], str) or not value["media_type"]:
        raise FromScratchExclusionError("held-out V2 descriptor media type is invalid")


def _canonical_json(value: object) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode(
        "utf-8"
    )


def _is_sha256(value: object) -> bool:
    return (
        type(value) is str
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _require_sha256(value: object, *, label: str) -> None:
    if not _is_sha256(value):
        raise FromScratchExclusionError(f"{label} SHA-256 is invalid")


def _file_identity(status: os.stat_result) -> tuple[int, int, int, int, int]:
    return status.st_dev, status.st_ino, status.st_size, status.st_mtime_ns, status.st_ctime_ns
