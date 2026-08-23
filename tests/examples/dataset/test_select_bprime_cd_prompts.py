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

from __future__ import annotations

import hashlib
import json
import sys
from collections import Counter
from dataclasses import replace
from pathlib import Path
from types import MappingProxyType

import pytest

ROOT = Path(__file__).resolve().parents[3]
MODULE_DIR = ROOT / "examples/dataset"

sys.path.insert(0, str(MODULE_DIR))
try:
    from bprime_cd_policy import ArmPolicy, PromptCell, PromptPolicy, load_prompt_policy
    from build_specdec_inventory import CandidateCell, CandidateInventory, CandidatePrompt
    from select_bprime_cd_prompts import (
        PromptSelectionBlocked,
        build_policy_count_proofs,
        select_prompt_views,
    )
finally:
    sys.path.pop(0)


def _policy() -> PromptPolicy:
    b_counts = {"stem": 40, "japanese": 25, "spanish": 25, "french": 25, "italian": 25}
    cd_counts = {
        "swe-agentic-tool": 120,
        "math": 80,
        "code": 40,
        "stem-science": 80,
        "multilingual": 60,
        "instruction-chat": 20,
    }
    arms = {
        "B-prime": ArmPolicy(
            140,
            MappingProxyType({key: PromptCell(value) for key, value in b_counts.items()}),
            MappingProxyType({}),
        ),
        "C": ArmPolicy(
            400,
            MappingProxyType({key: PromptCell(value) for key, value in cd_counts.items()}),
            MappingProxyType({}),
        ),
        "D": ArmPolicy(
            400,
            MappingProxyType({key: PromptCell(value) for key, value in cd_counts.items()}),
            MappingProxyType(
                {
                    "agentless-swe": 40,
                    "interactive-swe-replay": 40,
                    "generic-tool-replay": 40,
                }
            ),
        ),
    }
    return PromptPolicy(
        schema_version=1,
        seed=20260822,
        reserve_numerator=6,
        reserve_denominator=5,
        arms=MappingProxyType(arms),
        exposure_tokens=(256_000_000, 1_000_000_000),
        sequence_length=4_096,
        full_context_maximum=32_768,
        allowed_languages=frozenset({"en", "ja", "es", "fr", "it"}),
        denied_languages=frozenset({"de"}),
        policy_sha256="a" * 64,
    )


def _candidate(
    ordinal: int,
    *,
    domain: str,
    lane: str,
    language: str,
    bucket: str,
) -> CandidatePrompt:
    prompt = {"messages": [{"role": "user", "content": f"prompt-{ordinal}"}], "tools": []}
    canonical = json.dumps(prompt, sort_keys=True, separators=(",", ":")).encode()
    return CandidatePrompt(
        prompt_uuid=hashlib.sha256(canonical).hexdigest(),
        canonical_bytes=canonical,
        source_id=f"fixture/{domain}/{lane}",
        source_revision="b" * 40,
        source_file_sha256=hashlib.sha256(f"{domain}/{lane}".encode()).hexdigest(),
        source_row_index=ordinal,
        domain=domain,
        language=language,
        lane=lane,
        context_bucket=bucket,
        full_token_count=2 if bucket == "le4k" else 5_000,
        source_manifest_sha256="c" * 64,
        source_file_path=f"{domain}/{lane}.jsonl",
        input_ids=(1, 2),
        tokenizer_sha256="d" * 64,
        replay_valid=lane in {"interactive-swe-replay", "generic-tool-replay"},
    )


def _inventory(*, reverse: bool = False, drop_last: bool = False) -> CandidateInventory:
    specifications = [
        ("stem-science", "target-synth", "en", 120),
        ("multilingual", "target-synth", "ja", 30),
        ("multilingual", "target-synth", "es", 30),
        ("multilingual", "target-synth", "fr", 30),
        ("multilingual", "target-synth", "it", 30),
        ("math", "target-synth", "en", 96),
        ("code", "target-synth", "en", 48),
        ("instruction-chat", "target-synth", "en", 24),
        ("swe-agentic-tool", "agentless-swe", "en", 144),
        ("swe-agentic-tool", "interactive-swe-replay", "en", 48),
        ("swe-agentic-tool", "generic-tool-replay", "en", 48),
    ]
    rows: list[CandidatePrompt] = []
    ordinal = 0
    for domain, lane, language, count in specifications:
        for index in range(count):
            rows.append(
                _candidate(
                    ordinal,
                    domain=domain,
                    lane=lane,
                    language=language,
                    bucket=("le4k" if domain == "multilingual" or index % 2 == 0 else "4k_16k"),
                )
            )
            ordinal += 1
    if drop_last:
        rows.pop()
    capacity = Counter(
        CandidateCell(row.domain, row.lane, row.language, row.context_bucket) for row in rows
    )
    return CandidateInventory(
        rows=tuple(reversed(rows)) if reverse else tuple(rows),
        capacity=MappingProxyType(dict(capacity)),
        quarantine_counts=MappingProxyType({}),
        inventory_sha256=("e" if not drop_last else "f") * 64,
    )


def _all_rows(view):
    return (*view.primary_rows, *view.reserve_rows)


def test_bprime_has_exact_language_counts_reserves_and_stable_unique_order() -> None:
    policy = _policy()
    forward = select_prompt_views(_inventory(), policy)
    reverse = select_prompt_views(_inventory(reverse=True), policy)

    assert forward.B_prime.cell_counts == {
        "stem": 40,
        "japanese": 25,
        "spanish": 25,
        "french": 25,
        "italian": 25,
    }
    assert len(forward.B_prime.primary_prompt_ids) == 140
    assert len(forward.B_prime.reserve_prompt_ids) == 28
    assert Counter((row.domain, row.language) for row in forward.B_prime.primary_rows) == {
        ("stem-science", "en"): 40,
        ("multilingual", "ja"): 25,
        ("multilingual", "es"): 25,
        ("multilingual", "fr"): 25,
        ("multilingual", "it"): 25,
    }
    assert len({row.prompt_uuid for row in _all_rows(forward.B_prime)}) == 168
    assert all(row.language != "de" for row in _all_rows(forward.B_prime))
    assert forward.B_prime.primary_prompt_ids == reverse.B_prime.primary_prompt_ids
    assert forward.B_prime.reserve_prompt_ids == reverse.B_prime.reserve_prompt_ids
    assert forward.selection_sha256 == reverse.selection_sha256


def test_cd_views_are_paired_outside_agentic_lanes() -> None:
    bundle = select_prompt_views(_inventory(), _policy())

    assert len(bundle.C.primary_prompt_ids) == 400
    assert len(bundle.D.primary_prompt_ids) == 400
    assert (
        bundle.C.cell_counts
        == bundle.D.cell_counts
        == {
            "swe-agentic-tool": 120,
            "math": 80,
            "code": 40,
            "stem-science": 80,
            "multilingual": 60,
            "instruction-chat": 20,
        }
    )
    assert bundle.C.non_agentic_prompt_ids == bundle.D.non_agentic_prompt_ids
    assert bundle.C.non_agentic_bucket_floors == bundle.D.non_agentic_bucket_floors
    assert {
        row.prompt_uuid for row in bundle.C.reserve_rows if row.domain != "swe-agentic-tool"
    } == {row.prompt_uuid for row in bundle.D.reserve_rows if row.domain != "swe-agentic-tool"}
    assert bundle.C.agentic_prompt_ids != bundle.D.agentic_prompt_ids
    assert bundle.C.lane_counts == {"agentless-swe": 120}
    assert bundle.D.lane_counts == {
        "agentless-swe": 40,
        "interactive-swe-replay": 40,
        "generic-tool-replay": 40,
    }
    assert Counter(
        row.lane for row in bundle.D.primary_rows if row.domain == "swe-agentic-tool"
    ) == {
        "agentless-swe": 40,
        "interactive-swe-replay": 40,
        "generic-tool-replay": 40,
    }
    assert bundle.C.non_agentic_bucket_floors["math"] == {
        "le4k": 40,
        "4k_16k": 40,
    }
    assert len(bundle.C.reserve_prompt_ids) == len(bundle.D.reserve_prompt_ids) == 80
    assert bundle.paired_cd_sha256


def test_selection_identity_binds_seed_receipts_source_and_quota() -> None:
    inventory = _inventory()
    policy = _policy()
    original = select_prompt_views(inventory, policy)
    changed_seed = select_prompt_views(inventory, replace(policy, seed=policy.seed + 1))
    changed_heldout = select_prompt_views(inventory, policy, held_out_receipt_sha256="1" * 64)
    changed_baseline = select_prompt_views(inventory, policy, baseline_receipt_sha256="2" * 64)
    changed_source = select_prompt_views(replace(inventory, inventory_sha256="3" * 64), policy)
    b_arm = policy.arms["B-prime"]
    cells = dict(b_arm.cells)
    cells["stem"] = PromptCell(39)
    changed_b = replace(b_arm, prompt_count=139, cells=MappingProxyType(cells))
    changed_quota = select_prompt_views(
        inventory,
        replace(policy, arms=MappingProxyType({**policy.arms, "B-prime": changed_b})),
    )

    digests = {
        original.selection_sha256,
        changed_seed.selection_sha256,
        changed_heldout.selection_sha256,
        changed_baseline.selection_sha256,
        changed_source.selection_sha256,
        changed_quota.selection_sha256,
    }
    assert len(digests) == 6


def test_one_row_shortfall_is_fatal_and_emits_a_blocker_receipt() -> None:
    with pytest.raises(PromptSelectionBlocked) as caught:
        select_prompt_views(_inventory(drop_last=True), _policy())

    assert caught.value.receipt["status"] == "blocked"
    assert caught.value.receipt["reason"] == "insufficient_candidates"
    assert caught.value.receipt["arm"] == "D"
    assert caught.value.receipt["lane"] == "generic-tool-replay"
    assert caught.value.receipt["deficit"] == 1
    assert "partial_output" not in caught.value.receipt


def test_canonical_large_count_proof_uses_exact_policy_arithmetic_only() -> None:
    policy = load_prompt_policy(MODULE_DIR / "bprime_cd_prompt_policy.yaml")
    proof = build_policy_count_proofs(policy)

    assert proof["B-prime"]["primary_count"] == 700_000
    assert proof["C"]["primary_count"] == proof["D"]["primary_count"] == 2_000_000
    assert proof["B-prime"]["acquisition_count"] == 840_000
    assert proof["C"]["acquisition_count"] == proof["D"]["acquisition_count"] == 2_400_000
    assert sum(proof["B-prime"]["cell_counts"].values()) == 700_000
    assert sum(proof["D"]["lane_counts"].values()) == 600_000
    assert proof["C"]["non_agentic_count"] == proof["D"]["non_agentic_count"]
    assert proof["paired_cd_sha256"]
