# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Select exact, deterministic B-prime/C/D prompt views and their reserves."""

from __future__ import annotations

import json
import re
from collections import defaultdict
from dataclasses import dataclass, replace
from hashlib import sha256
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Literal

from build_specdec_inventory import CandidateInventory, CandidatePrompt
from specdec_corpus_contracts import canonical_json, sha256_bytes

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping, Sequence

    from bprime_cd_policy import PromptPolicy

__all__ = [
    "PromptSelectionBlocked",
    "PromptSelectionBlockedError",
    "PromptView",
    "PromptViewBundle",
    "SelectedPrompt",
    "SelectionBlockedError",
    "build_policy_count_proofs",
    "prompt_view_manifest",
    "select_prompt_views",
]

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_BUCKET_ORDER = ("le4k", "4k_16k", "16k_32k")
_BPRIME_CELLS = {
    "stem": ("stem-science", "target-synth", "en"),
    "japanese": ("multilingual", "target-synth", "ja"),
    "spanish": ("multilingual", "target-synth", "es"),
    "french": ("multilingual", "target-synth", "fr"),
    "italian": ("multilingual", "target-synth", "it"),
}
_NON_AGENTIC_CELLS = (
    "math",
    "code",
    "stem-science",
    "multilingual",
    "instruction-chat",
)
_DEFAULT_RECEIPT_SHA256 = "0" * 64


@dataclass(frozen=True)
class SelectedPrompt:
    """One immutable primary or reserve selection with inspectable provenance."""

    prompt_uuid: str
    arm: str
    domain: str
    lane: str
    language: str
    context_bucket: str
    source_id: str
    source_revision: str
    source_file_sha256: str
    source_manifest_sha256: str
    source_file_path: str
    source_row_index: int
    candidate_rank: int
    candidate_rank_sha256: str
    selection_index: int
    status: Literal["primary", "reserve"]
    canonical_prompt_json: str

    @property
    def canonical_prompt(self) -> Any:
        """Return the complete canonical prompt represented by this row."""
        return json.loads(self.canonical_prompt_json)


@dataclass(frozen=True)
class PromptView:
    """One arm's exact ordered prompt list, reserves, and count proofs."""

    arm: str
    primary_rows: tuple[SelectedPrompt, ...]
    reserve_rows: tuple[SelectedPrompt, ...]
    cell_counts: Mapping[str, int]
    lane_counts: Mapping[str, int]
    bucket_floors: Mapping[str, Mapping[str, int]]
    non_agentic_bucket_floors: Mapping[str, Mapping[str, int]]
    count_proof_sha256: str

    @property
    def primary_prompt_ids(self) -> tuple[str, ...]:
        return tuple(row.prompt_uuid for row in self.primary_rows)

    @property
    def final_prompt_ids(self) -> tuple[str, ...]:
        """Alias for the final ordered primary UUIDs."""
        return self.primary_prompt_ids

    @property
    def final_ordered_uuids(self) -> tuple[str, ...]:
        return self.primary_prompt_ids

    @property
    def reserve_prompt_ids(self) -> tuple[str, ...]:
        return tuple(row.prompt_uuid for row in self.reserve_rows)

    @property
    def reserve_ordered_uuids(self) -> tuple[str, ...]:
        return self.reserve_prompt_ids

    @property
    def non_agentic_prompt_ids(self) -> frozenset[str]:
        return frozenset(
            row.prompt_uuid for row in self.primary_rows if row.domain != "swe-agentic-tool"
        )

    @property
    def agentic_prompt_ids(self) -> frozenset[str]:
        return frozenset(
            row.prompt_uuid for row in self.primary_rows if row.domain == "swe-agentic-tool"
        )


@dataclass(frozen=True)
class PromptViewBundle:
    """The exact B-prime/C/D selections bound to all input identities."""

    B_prime: PromptView
    C: PromptView
    D: PromptView
    policy_sha256: str
    seed: int
    source_inventory_sha256: str
    baseline_receipt_sha256: str
    held_out_receipt_sha256: str
    paired_cd_sha256: str
    selection_sha256: str

    @property
    def selection_digest(self) -> str:
        """Return the selection identity digest."""
        return self.selection_sha256

    @property
    def paired_cd_digest(self) -> str:
        """Return the paired C/D proof digest."""
        return self.paired_cd_sha256


class PromptSelectionBlockedError(ValueError):
    """Fatal exact-count shortfall carrying a canonical blocker receipt."""

    def __init__(self, receipt: Mapping[str, Any]) -> None:
        self.receipt = MappingProxyType(dict(receipt))
        super().__init__(
            "insufficient candidates for "
            f"{receipt['arm']}/{receipt['cell']}/{receipt['lane']}/{receipt['context_bucket']}: "
            f"need {receipt['required_candidate_count']}, have {receipt['available_candidate_count']}"
        )


PromptSelectionBlocked = PromptSelectionBlockedError
SelectionBlockedError = PromptSelectionBlockedError


def build_policy_count_proofs(policy: PromptPolicy) -> dict[str, Any]:
    """Build canonical large-count arithmetic proofs without allocating prompt rows."""
    proofs: dict[str, Any] = {}
    for arm_name, arm in policy.arms.items():
        cell_counts = {name: cell.prompt_count for name, cell in arm.cells.items()}
        lane_counts = dict(arm.lanes)
        cell_acquisition_counts = {
            name: _acquisition_count(count, policy) for name, count in cell_counts.items()
        }
        proof: dict[str, Any] = {
            "primary_count": arm.prompt_count,
            "acquisition_count": sum(cell_acquisition_counts.values()),
            "reserve_count": sum(cell_acquisition_counts.values()) - arm.prompt_count,
            "cell_counts": cell_counts,
            "cell_acquisition_counts": cell_acquisition_counts,
            "lane_counts": lane_counts,
            "lane_acquisition_counts": {
                name: _acquisition_count(count, policy) for name, count in lane_counts.items()
            },
        }
        if arm_name in {"C", "D"}:
            proof["non_agentic_count"] = sum(cell_counts[name] for name in _NON_AGENTIC_CELLS)
        proofs[arm_name] = proof
    paired_payload = {
        "C": {name: proofs["C"]["cell_counts"][name] for name in _NON_AGENTIC_CELLS},
        "D": {name: proofs["D"]["cell_counts"][name] for name in _NON_AGENTIC_CELLS},
        "reserve": {
            name: proofs["C"]["cell_acquisition_counts"][name] - proofs["C"]["cell_counts"][name]
            for name in _NON_AGENTIC_CELLS
        },
    }
    proofs["paired_cd_sha256"] = sha256_bytes(canonical_json(paired_payload))
    return proofs


def select_prompt_views(
    inventory: CandidateInventory,
    policy: PromptPolicy,
    *,
    baseline_receipt_sha256: str = _DEFAULT_RECEIPT_SHA256,
    held_out_receipt_sha256: str = _DEFAULT_RECEIPT_SHA256,
) -> PromptViewBundle:
    """Select deterministic exact-count B-prime/C/D primary and reserve prompt views."""
    _validate_digest(inventory.inventory_sha256, "source inventory")
    _validate_digest(policy.policy_sha256, "policy")
    _validate_digest(baseline_receipt_sha256, "baseline receipt")
    _validate_digest(held_out_receipt_sha256, "held-out receipt")
    rows = _validate_and_index_rows(inventory.rows, policy)

    b_primary: dict[str, list[SelectedPrompt]] = {}
    b_reserve: dict[str, list[SelectedPrompt]] = {}
    b_floors: dict[str, dict[str, int]] = {}
    for cell, (domain, lane, language) in _BPRIME_CELLS.items():
        eligible = _eligible(rows, domain=domain, lane=lane, language=language)
        primary, reserve, floors = _select_cell(
            eligible,
            quota=policy.count_for("B-prime", cell),
            arm="B-prime",
            cell=cell,
            policy=policy,
            inventory_sha256=inventory.inventory_sha256,
            baseline_receipt_sha256=baseline_receipt_sha256,
            held_out_receipt_sha256=held_out_receipt_sha256,
        )
        b_primary[cell], b_reserve[cell], b_floors[cell] = primary, reserve, floors
    b_view = _build_view("B-prime", b_primary, b_reserve, b_floors, policy, non_agentic_cells=())

    c_primary: dict[str, list[SelectedPrompt]] = {}
    c_reserve: dict[str, list[SelectedPrompt]] = {}
    c_floors: dict[str, dict[str, int]] = {}
    d_primary: dict[str, list[SelectedPrompt]] = {}
    d_reserve: dict[str, list[SelectedPrompt]] = {}
    d_floors: dict[str, dict[str, int]] = {}
    for cell in _NON_AGENTIC_CELLS:
        c_eligible = _eligible(rows, domain=cell, lane="target-synth")
        d_eligible = _eligible(rows, domain=cell, lane="target-synth")
        common_capacities = _common_bucket_capacities(c_eligible, d_eligible)
        primary, reserve, floors = _select_cell(
            c_eligible,
            quota=policy.count_for("C", cell),
            arm="C",
            cell=cell,
            policy=policy,
            inventory_sha256=inventory.inventory_sha256,
            baseline_receipt_sha256=baseline_receipt_sha256,
            held_out_receipt_sha256=held_out_receipt_sha256,
            capacity_vector=common_capacities,
        )
        c_primary[cell], c_reserve[cell], c_floors[cell] = primary, reserve, floors
        d_primary[cell] = [replace(row, arm="D") for row in primary]
        d_reserve[cell] = [replace(row, arm="D") for row in reserve]
        d_floors[cell] = dict(floors)

    c_agentic = _eligible(rows, domain="swe-agentic-tool", lane="agentless-swe")
    primary, reserve, floors = _select_cell(
        c_agentic,
        quota=policy.count_for("C", "swe-agentic-tool"),
        arm="C",
        cell="swe-agentic-tool",
        policy=policy,
        inventory_sha256=inventory.inventory_sha256,
        baseline_receipt_sha256=baseline_receipt_sha256,
        held_out_receipt_sha256=held_out_receipt_sha256,
    )
    c_primary["swe-agentic-tool"] = primary
    c_reserve["swe-agentic-tool"] = reserve
    c_floors["swe-agentic-tool"] = floors

    d_agentic_primary: list[SelectedPrompt] = []
    d_agentic_reserve: list[SelectedPrompt] = []
    d_agentic_floors: dict[str, int] = defaultdict(int)
    for lane, quota in policy.arms["D"].lanes.items():
        eligible = _eligible(rows, domain="swe-agentic-tool", lane=lane)
        primary, reserve, floors = _select_cell(
            eligible,
            quota=quota,
            arm="D",
            cell="swe-agentic-tool",
            policy=policy,
            inventory_sha256=inventory.inventory_sha256,
            baseline_receipt_sha256=baseline_receipt_sha256,
            held_out_receipt_sha256=held_out_receipt_sha256,
        )
        d_agentic_primary.extend(primary)
        d_agentic_reserve.extend(reserve)
        for bucket, count in floors.items():
            d_agentic_floors[bucket] += count
    d_primary["swe-agentic-tool"] = d_agentic_primary
    d_reserve["swe-agentic-tool"] = d_agentic_reserve
    d_floors["swe-agentic-tool"] = dict(d_agentic_floors)

    c_view = _build_view(
        "C", c_primary, c_reserve, c_floors, policy, non_agentic_cells=_NON_AGENTIC_CELLS
    )
    d_view = _build_view(
        "D", d_primary, d_reserve, d_floors, policy, non_agentic_cells=_NON_AGENTIC_CELLS
    )
    _validate_paired_views(c_view, d_view)
    paired_cd_sha256 = _paired_cd_sha256(c_view, d_view)
    identity = {
        "schema_version": 1,
        "policy_sha256": policy.policy_sha256,
        "seed": policy.seed,
        "source_inventory_sha256": inventory.inventory_sha256,
        "baseline_receipt_sha256": baseline_receipt_sha256,
        "held_out_receipt_sha256": held_out_receipt_sha256,
        "paired_cd_sha256": paired_cd_sha256,
        "arms": {
            view.arm: {
                "primary_prompt_ids": view.primary_prompt_ids,
                "reserve_prompt_ids": view.reserve_prompt_ids,
                "cell_counts": dict(view.cell_counts),
                "lane_counts": dict(view.lane_counts),
                "bucket_floors": _plain_floors(view.bucket_floors),
                "count_proof_sha256": view.count_proof_sha256,
            }
            for view in (b_view, c_view, d_view)
        },
    }
    return PromptViewBundle(
        B_prime=b_view,
        C=c_view,
        D=d_view,
        policy_sha256=policy.policy_sha256,
        seed=policy.seed,
        source_inventory_sha256=inventory.inventory_sha256,
        baseline_receipt_sha256=baseline_receipt_sha256,
        held_out_receipt_sha256=held_out_receipt_sha256,
        paired_cd_sha256=paired_cd_sha256,
        selection_sha256=sha256_bytes(canonical_json(identity)),
    )


def prompt_view_manifest(bundle: PromptViewBundle) -> dict[str, Any]:
    """Return a tamper-evident, canonical JSON selection artifact for inspection."""
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "selection_sha256": bundle.selection_sha256,
        "paired_cd_sha256": bundle.paired_cd_sha256,
        "identity": {
            "policy_sha256": bundle.policy_sha256,
            "seed": bundle.seed,
            "source_inventory_sha256": bundle.source_inventory_sha256,
            "baseline_receipt_sha256": bundle.baseline_receipt_sha256,
            "held_out_receipt_sha256": bundle.held_out_receipt_sha256,
        },
        "arms": {
            view.arm: _view_manifest_record(view) for view in (bundle.B_prime, bundle.C, bundle.D)
        },
    }
    manifest["artifact_sha256"] = sha256_bytes(canonical_json(manifest))
    return manifest


def _validate_digest(value: str, name: str) -> None:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ValueError(f"{name} digest must be an exact lowercase SHA-256")


def _validate_and_index_rows(
    raw_rows: Sequence[Any], policy: PromptPolicy
) -> tuple[CandidatePrompt, ...]:
    rows: list[CandidatePrompt] = []
    seen: set[str] = set()
    for row in raw_rows:
        if not isinstance(row, CandidatePrompt):
            raise TypeError("candidate inventory rows must be CandidatePrompt values")
        _validate_digest(row.prompt_uuid, "prompt UUID")
        if row.prompt_uuid in seen:
            raise ValueError(f"duplicate candidate prompt UUID: {row.prompt_uuid}")
        seen.add(row.prompt_uuid)
        if not policy.is_language_allowed(row.language):
            continue
        if sha256(row.canonical_bytes).hexdigest() != row.prompt_uuid:
            raise ValueError(f"candidate canonical prompt identity mismatch: {row.prompt_uuid}")
        rows.append(row)
    return tuple(rows)


def _eligible(
    rows: Sequence[CandidatePrompt],
    *,
    domain: str,
    lane: str,
    language: str | None = None,
) -> tuple[CandidatePrompt, ...]:
    return tuple(
        row
        for row in rows
        if row.domain == domain
        and row.lane == lane
        and (language is None or row.language == language)
    )


def _common_bucket_capacities(
    left: Sequence[CandidatePrompt], right: Sequence[CandidatePrompt]
) -> dict[str, int]:
    left_counts = _bucket_counts(left)
    right_counts = _bucket_counts(right)
    return {
        bucket: min(left_counts.get(bucket, 0), right_counts.get(bucket, 0))
        for bucket in sorted(set(left_counts) | set(right_counts), key=_bucket_key)
        if min(left_counts.get(bucket, 0), right_counts.get(bucket, 0)) > 0
    }


def _bucket_counts(rows: Iterable[CandidatePrompt]) -> dict[str, int]:
    counts: dict[str, int] = defaultdict(int)
    for row in rows:
        counts[row.context_bucket] += 1
    return dict(counts)


def _bucket_key(bucket: str) -> tuple[int, str]:
    try:
        return (_BUCKET_ORDER.index(bucket), bucket)
    except ValueError:
        return (len(_BUCKET_ORDER), bucket)


def _allocate_bucket_quotas(capacities: Mapping[str, int], quota: int) -> dict[str, int]:
    if quota <= 0:
        raise ValueError("cell quota must be positive")
    total = sum(capacities.values())
    if total < quota or not capacities:
        return {}
    floors = {bucket: quota * capacity // total for bucket, capacity in capacities.items()}
    remaining = quota - sum(floors.values())
    remainder_order = sorted(
        capacities,
        key=lambda bucket: (-(quota * capacities[bucket] % total), _bucket_key(bucket)),
    )
    for bucket in remainder_order[:remaining]:
        floors[bucket] += 1
    return {bucket: floors[bucket] for bucket in sorted(floors, key=_bucket_key) if floors[bucket]}


def _acquisition_count(quota: int, policy: PromptPolicy) -> int:
    return (
        quota * policy.reserve_numerator + policy.reserve_denominator - 1
    ) // policy.reserve_denominator


def _rank(candidate: CandidatePrompt, policy: PromptPolicy) -> str:
    fields = (
        policy.policy_sha256,
        str(policy.seed),
        candidate.domain,
        candidate.lane,
        candidate.language,
        candidate.context_bucket,
        candidate.source_id,
        candidate.source_revision,
        candidate.source_file_sha256,
        candidate.source_manifest_sha256,
        candidate.source_file_path,
        str(candidate.source_row_index),
        candidate.prompt_uuid,
    )
    return sha256("\0".join(fields).encode()).hexdigest()


def _select_cell(
    candidates: Sequence[CandidatePrompt],
    *,
    quota: int,
    arm: str,
    cell: str,
    policy: PromptPolicy,
    inventory_sha256: str,
    baseline_receipt_sha256: str,
    held_out_receipt_sha256: str,
    capacity_vector: Mapping[str, int] | None = None,
) -> tuple[list[SelectedPrompt], list[SelectedPrompt], dict[str, int]]:
    capacities = dict(capacity_vector or _bucket_counts(candidates))
    bucket_quotas = _allocate_bucket_quotas(capacities, quota)
    if not bucket_quotas:
        _raise_shortfall(
            arm=arm,
            cell=cell,
            lane=_single_lane(candidates),
            bucket="none",
            quota=quota,
            required=_acquisition_count(quota, policy),
            available=sum(capacities.values()),
            policy=policy,
            inventory_sha256=inventory_sha256,
            baseline_receipt_sha256=baseline_receipt_sha256,
            held_out_receipt_sha256=held_out_receipt_sha256,
        )
    by_bucket: dict[str, list[CandidatePrompt]] = defaultdict(list)
    for candidate in candidates:
        by_bucket[candidate.context_bucket].append(candidate)
    primary: list[SelectedPrompt] = []
    reserve: list[SelectedPrompt] = []
    for bucket, bucket_quota in bucket_quotas.items():
        ranked = sorted(by_bucket[bucket], key=lambda row: (_rank(row, policy), row.prompt_uuid))
        required = _acquisition_count(bucket_quota, policy)
        if len(ranked) < required:
            _raise_shortfall(
                arm=arm,
                cell=cell,
                lane=_single_lane(ranked or candidates),
                bucket=bucket,
                quota=bucket_quota,
                required=required,
                available=len(ranked),
                policy=policy,
                inventory_sha256=inventory_sha256,
                baseline_receipt_sha256=baseline_receipt_sha256,
                held_out_receipt_sha256=held_out_receipt_sha256,
            )
        reserve_count = required - bucket_quota
        for candidate_rank, candidate in enumerate(ranked):
            if candidate_rank >= required:
                break
            status: Literal["primary", "reserve"] = (
                "reserve" if candidate_rank < reserve_count else "primary"
            )
            selected = _selected_prompt(
                candidate,
                arm=arm,
                status=status,
                candidate_rank=candidate_rank,
                candidate_rank_sha256=_rank(candidate, policy),
            )
            (reserve if status == "reserve" else primary).append(selected)
    return primary, reserve, bucket_quotas


def _single_lane(candidates: Sequence[CandidatePrompt]) -> str:
    lanes = sorted({candidate.lane for candidate in candidates})
    return lanes[0] if len(lanes) == 1 else "mixed" if lanes else "unknown"


def _raise_shortfall(
    *,
    arm: str,
    cell: str,
    lane: str,
    bucket: str,
    quota: int,
    required: int,
    available: int,
    policy: PromptPolicy,
    inventory_sha256: str,
    baseline_receipt_sha256: str,
    held_out_receipt_sha256: str,
) -> None:
    receipt: dict[str, Any] = {
        "schema_version": 1,
        "status": "blocked",
        "reason": "insufficient_candidates",
        "arm": arm,
        "cell": cell,
        "lane": lane,
        "context_bucket": bucket,
        "quota_count": quota,
        "required_candidate_count": required,
        "available_candidate_count": available,
        "deficit": required - available,
        "policy_sha256": policy.policy_sha256,
        "seed": policy.seed,
        "source_inventory_sha256": inventory_sha256,
        "baseline_receipt_sha256": baseline_receipt_sha256,
        "held_out_receipt_sha256": held_out_receipt_sha256,
    }
    receipt["blocker_sha256"] = sha256_bytes(canonical_json(receipt))
    raise PromptSelectionBlockedError(receipt)


def _selected_prompt(
    candidate: CandidatePrompt,
    *,
    arm: str,
    status: Literal["primary", "reserve"],
    candidate_rank: int,
    candidate_rank_sha256: str,
) -> SelectedPrompt:
    return SelectedPrompt(
        prompt_uuid=candidate.prompt_uuid,
        arm=arm,
        domain=candidate.domain,
        lane=candidate.lane,
        language=candidate.language,
        context_bucket=candidate.context_bucket,
        source_id=candidate.source_id,
        source_revision=candidate.source_revision,
        source_file_sha256=candidate.source_file_sha256,
        source_manifest_sha256=candidate.source_manifest_sha256,
        source_file_path=candidate.source_file_path,
        source_row_index=candidate.source_row_index,
        candidate_rank=candidate_rank,
        candidate_rank_sha256=candidate_rank_sha256,
        selection_index=-1,
        status=status,
        canonical_prompt_json=candidate.canonical_bytes.decode("utf-8"),
    )


def _build_view(
    arm: str,
    primary_by_cell: Mapping[str, Sequence[SelectedPrompt]],
    reserve_by_cell: Mapping[str, Sequence[SelectedPrompt]],
    bucket_floors: Mapping[str, Mapping[str, int]],
    policy: PromptPolicy,
    *,
    non_agentic_cells: Sequence[str],
) -> PromptView:
    primary = _index_selection(
        sorted(
            (row for rows in primary_by_cell.values() for row in rows),
            key=_selected_order_key,
        )
    )
    reserve = _index_selection(
        sorted(
            (row for rows in reserve_by_cell.values() for row in rows),
            key=_selected_order_key,
        )
    )
    expected_count = policy.count_for(arm)
    if len(primary) != expected_count:
        raise AssertionError(f"{arm} exact-count proof failed: {len(primary)} != {expected_count}")
    all_ids = [row.prompt_uuid for row in (*primary, *reserve)]
    if len(set(all_ids)) != len(all_ids):
        raise ValueError(f"{arm} primary and reserve UUIDs are not globally unique")
    cell_counts = {name: policy.count_for(arm, name) for name in policy.arms[arm].cells}
    if arm == "C":
        lane_counts = {"agentless-swe": policy.count_for("C", "swe-agentic-tool")}
    else:
        lane_counts = dict(policy.arms[arm].lanes)
    frozen_floors = _freeze_floors(bucket_floors)
    non_agentic = _freeze_floors({cell: bucket_floors[cell] for cell in non_agentic_cells})
    proof_payload = {
        "arm": arm,
        "primary_count": len(primary),
        "reserve_count": len(reserve),
        "cell_counts": cell_counts,
        "lane_counts": lane_counts,
        "bucket_floors": _plain_floors(frozen_floors),
    }
    return PromptView(
        arm=arm,
        primary_rows=primary,
        reserve_rows=reserve,
        cell_counts=MappingProxyType(cell_counts),
        lane_counts=MappingProxyType(lane_counts),
        bucket_floors=frozen_floors,
        non_agentic_bucket_floors=non_agentic,
        count_proof_sha256=sha256_bytes(canonical_json(proof_payload)),
    )


def _selected_order_key(row: SelectedPrompt) -> tuple[str, str, str, str, str]:
    return (
        row.candidate_rank_sha256,
        row.prompt_uuid,
        row.domain,
        row.lane,
        row.context_bucket,
    )


def _index_selection(rows: Sequence[SelectedPrompt]) -> tuple[SelectedPrompt, ...]:
    return tuple(replace(row, selection_index=index) for index, row in enumerate(rows))


def _freeze_floors(
    floors: Mapping[str, Mapping[str, int]],
) -> Mapping[str, Mapping[str, int]]:
    return MappingProxyType(
        {
            cell: MappingProxyType(
                {bucket: values[bucket] for bucket in sorted(values, key=_bucket_key)}
            )
            for cell, values in sorted(floors.items())
        }
    )


def _plain_floors(floors: Mapping[str, Mapping[str, int]]) -> dict[str, dict[str, int]]:
    return {cell: dict(values) for cell, values in floors.items()}


def _validate_paired_views(c_view: PromptView, d_view: PromptView) -> None:
    if c_view.non_agentic_prompt_ids != d_view.non_agentic_prompt_ids:
        raise AssertionError("C/D non-agentic prompt UUID sets are not paired")
    if _plain_floors(c_view.non_agentic_bucket_floors) != _plain_floors(
        d_view.non_agentic_bucket_floors
    ):
        raise AssertionError("C/D non-agentic bucket floors are not paired")
    c_reserve = {row.prompt_uuid for row in c_view.reserve_rows if row.domain != "swe-agentic-tool"}
    d_reserve = {row.prompt_uuid for row in d_view.reserve_rows if row.domain != "swe-agentic-tool"}
    if c_reserve != d_reserve:
        raise AssertionError("C/D non-agentic reserve UUID sets are not paired")


def _paired_cd_sha256(c_view: PromptView, d_view: PromptView) -> str:
    payload = {
        "non_agentic_primary_prompt_ids": sorted(c_view.non_agentic_prompt_ids),
        "non_agentic_reserve_prompt_ids": sorted(
            row.prompt_uuid for row in c_view.reserve_rows if row.domain != "swe-agentic-tool"
        ),
        "bucket_floors": _plain_floors(c_view.non_agentic_bucket_floors),
        "C_count_proof_sha256": c_view.count_proof_sha256,
        "D_count_proof_sha256": d_view.count_proof_sha256,
    }
    return sha256_bytes(canonical_json(payload))


def _view_manifest_record(view: PromptView) -> dict[str, Any]:
    return {
        "primary_prompt_ids": list(view.primary_prompt_ids),
        "reserve_prompt_ids": list(view.reserve_prompt_ids),
        "cell_counts": dict(view.cell_counts),
        "lane_counts": dict(view.lane_counts),
        "bucket_floors": _plain_floors(view.bucket_floors),
        "non_agentic_bucket_floors": _plain_floors(view.non_agentic_bucket_floors),
        "count_proof_sha256": view.count_proof_sha256,
        "rows": [
            _selected_manifest_record(row) for row in (*view.primary_rows, *view.reserve_rows)
        ],
    }


def _selected_manifest_record(row: SelectedPrompt) -> dict[str, Any]:
    return {
        "prompt_uuid": row.prompt_uuid,
        "arm": row.arm,
        "domain": row.domain,
        "lane": row.lane,
        "language": row.language,
        "context_bucket": row.context_bucket,
        "source_id": row.source_id,
        "source_revision": row.source_revision,
        "source_file_sha256": row.source_file_sha256,
        "source_manifest_sha256": row.source_manifest_sha256,
        "source_file_path": row.source_file_path,
        "source_row_index": row.source_row_index,
        "candidate_rank": row.candidate_rank,
        "candidate_rank_sha256": row.candidate_rank_sha256,
        "selection_index": row.selection_index,
        "status": row.status,
        "canonical_prompt": row.canonical_prompt,
    }
