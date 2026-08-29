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

import importlib.util
import json
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[3]
MODULE = ROOT / "examples/dataset/build_specdec_study_corpus.py"
POLICY = ROOT / "examples/dataset/ptv23_complement_study.yaml"


def _load():
    spec = importlib.util.spec_from_file_location("corpus", MODULE)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _rows() -> list[dict]:
    rows = []
    lanes = {
        "math": "target-synth",
        "code": "target-synth",
        "stem": "target-synth",
        "chat": "target-synth",
        "multilingual": "target-synth",
        "agentless-swe": "target-synth",
        "interactive-swe-replay": "recorded-trace",
        "generic-tool-replay": "recorded-trace",
    }
    for category, response_source in lanes.items():
        domain = (
            "swe"
            if category in {"agentless-swe", "interactive-swe-replay", "generic-tool-replay"}
            else category
        )
        rows.extend(
            [
                {
                    "prompt_id": f"{category}-{index}",
                    "pool": "ptv2"
                    if category
                    not in {"agentless-swe", "interactive-swe-replay", "generic-tool-replay"}
                    else "ptv3",
                    "category": domain,
                    "lane": category if domain == "swe" else "target-synth",
                    "language": "ja" if domain == "multilingual" else "en",
                    "context_bucket": "le4k",
                    "assistant_tokens": 1,
                    "source_id": category,
                    "source_revision": "a" * 40,
                    "license": "Apache-2.0",
                    "source_manifest_sha256": "b" * 64,
                    "source_file_path": f"{category}.jsonl",
                    "source_file_sha256": "c" * 64,
                    "full_token_count": 2,
                    "response_source": response_source,
                    "tool_lane": "recorded-trace"
                    if response_source == "recorded-trace"
                    else "none",
                    "tokenizer_sha256": "f" * 64,
                    "input_ids": [1, 2],
                    "loss_mask": [0, 1],
                }
                for index in range(400)
            ]
        )
    return rows


def test_b_c_d_exact_domain_and_d_lane_token_quotas() -> None:
    module = _load()
    policy = yaml.safe_load(POLICY.read_text())
    b = module.select_ptv23_arm(_rows(), config=policy, arm="B", target_assistant_tokens=1000)
    c = module.select_ptv23_arm(_rows(), config=policy, arm="C", target_assistant_tokens=1000)
    d = module.select_ptv23_arm(_rows(), config=policy, arm="D", target_assistant_tokens=1000)
    assert module.token_totals_by(b, "category") == {
        "chat": 200,
        "code": 250,
        "math": 250,
        "multilingual": 100,
        "stem": 200,
    }
    assert module.token_totals_by(c, "category") == {
        "chat": 50,
        "code": 150,
        "math": 250,
        "multilingual": 50,
        "stem": 150,
        "swe": 350,
    }
    assert module.token_totals_by(d, "lane") == {
        "agentless-swe": 150,
        "generic-tool-replay": 100,
        "interactive-swe-replay": 100,
        "target-synth": 650,
    }


def test_b_excludes_prior_uuid_and_german_before_quota_selection() -> None:
    module = _load()
    policy = yaml.safe_load(POLICY.read_text())
    rows = _rows()
    german = next(row for row in rows if row["category"] == "multilingual")
    german["language"] = "de"
    prior = rows[1]["prompt_id"]
    selected = module.select_ptv23_arm(
        rows,
        config=policy,
        arm="B",
        target_assistant_tokens=100,
        prior_prompt_ids={prior},
    )
    assert prior not in {row["prompt_id"] for row in selected}
    assert all(row["language"] != "de" for row in selected)


def test_materialized_rows_preserve_prompt_selection_indexes(tmp_path: Path) -> None:
    module = _load()
    row = _rows()[0]
    row.update(
        {
            "prompt_uuid": "1" * 64,
            "arm": "D",
            "candidate_rank": 7,
            "selection_index": 3,
            "selection_status": "primary",
            "canonical_prompt": {
                "messages": [{"role": "user", "content": "inspect me"}],
                "tools": [],
            },
        }
    )
    output = tmp_path / "corpus"

    module.materialize_parquet([row], output, rows_per_shard=1, tokenizer_sha256="f" * 64)

    import pyarrow.parquet as pq

    stored = pq.read_table(next(output.glob("*.parquet"))).to_pylist()[0]
    assert stored["prompt_uuid"] == "1" * 64
    assert stored["arm"] == "D"
    assert stored["domain"] == row["category"]
    assert stored["lane"] == row["lane"]
    assert stored["language"] == row["language"]
    assert stored["source_row_index"] == row.get("source_row_index")
    assert stored["candidate_rank"] == 7
    assert stored["selection_index"] == 3
    assert stored["selection_status"] == "primary"
    assert json.loads(stored["canonical_prompt"])["messages"][0]["content"] == "inspect me"


def test_every_prefix_of_the_selection_is_a_sample_of_the_blend() -> None:
    """A prefix of the corpus must hold the blend, not the first lane written.

    The trainer does not read a corpus whole. `sample_size` becomes a streaming
    `.take(n)` over shards sorted by filename, so whatever composition sits in
    the first n rows *is* the training mix. Selecting lane by lane and writing
    that order out would hand a resized run a single-lane corpus while every
    manifest still reported the full blend -- silently, because nothing
    downstream re-counts. This is the failure that cost the upstream PTv2 mix
    9,377 multilingual rows, and it is cheap to make structurally impossible.
    """
    module = _load()
    policy = yaml.safe_load(POLICY.read_text())
    selected = module.select_ptv23_arm(
        _rows(), config=policy, arm="C", target_assistant_tokens=1000
    )

    full = module.token_totals_by(selected, "category")
    prefix = module.token_totals_by(selected[: len(selected) // 10], "category")

    assert set(prefix) == set(full), f"prefix dropped categories: {sorted(set(full) - set(prefix))}"
    for category, tokens in full.items():
        share = prefix[category] / sum(prefix.values())
        expected = tokens / sum(full.values())
        assert abs(share - expected) < 0.15, f"{category}: {share:.3f} vs {expected:.3f}"


def test_the_selection_order_is_reproducible_across_input_orderings() -> None:
    """The permutation is seeded by the prompt, not by arrival order.

    A shuffle keyed on anything positional would make the corpus depend on the
    order the inventory happened to be written in, so two runs of the same
    config could train on differently-ordered -- and therefore, after a take,
    differently-composed -- corpora.
    """
    module = _load()
    policy = yaml.safe_load(POLICY.read_text())
    rows = _rows()
    forward = module.select_ptv23_arm(rows, config=policy, arm="C", target_assistant_tokens=1000)
    reversed_input = module.select_ptv23_arm(
        list(reversed(rows)), config=policy, arm="C", target_assistant_tokens=1000
    )
    assert [row["prompt_id"] for row in forward] == [row["prompt_id"] for row in reversed_input]


def test_a_ptv2_row_without_a_language_is_refused_rather_than_admitted() -> None:
    """Language exists on a row only because the normalizer put it there.

    The PTv2 parquet schema is uuid/license/generator/version/category/reasoning/
    messages -- there is no language column, and the language of a shard is
    carried solely by its filename. A filter that reads the field with .get()
    therefore treats "the normalizer dropped it" and "this row is in an allowed
    language" as the same case, and every denied-language row enters arm B
    silently. The build has to stop instead.
    """
    module = _load()
    policy = yaml.safe_load(POLICY.read_text())
    rows = _rows()
    for row in rows:
        if row["pool"] == "ptv2":
            del row["language"]
            break

    with pytest.raises(ValueError, match="carries no language"):
        module.select_ptv23_arm(rows, config=policy, arm="B", target_assistant_tokens=1000)


def test_the_denied_languages_come_from_the_policy_not_from_the_code() -> None:
    """ptv2_languages was declared in the study YAML and read by nothing.

    The deny list existed as documentation while the code matched a hard-coded
    "de", so editing the policy changed the study's stated design and not its
    output. The fixture's only multilingual language is ja; denying it has to
    empty that lane, and an empty lane against a 10% multilingual quota is a
    shortfall. The shortfall *is* the evidence the policy was read -- the same
    call with the shipped policy, which allows ja, fills the quota.
    """
    module = _load()
    policy = yaml.safe_load(POLICY.read_text())
    policy["ptv2_languages"] = {"allow": ["en"], "deny": ["ja"]}

    with pytest.raises(ValueError, match="quota shortfall for multilingual"):
        module.select_ptv23_arm(_rows(), config=policy, arm="B", target_assistant_tokens=1000)


def test_a_language_outside_the_allow_list_is_excluded_without_being_denied() -> None:
    """Allow and deny are separate gates, and only deny was ever implemented.

    The staged PTv2 tree is a subset of upstream, so a shard in a language the
    policy never listed is a real possibility. It must not ride in on the
    absence of an explicit denial.
    """
    module = _load()
    policy = yaml.safe_load(POLICY.read_text())
    policy["ptv2_languages"] = {"allow": ["en"], "deny": []}

    with pytest.raises(ValueError, match="quota shortfall for multilingual"):
        module.select_ptv23_arm(_rows(), config=policy, arm="B", target_assistant_tokens=1000)


def test_a_language_that_is_both_allowed_and_denied_is_a_policy_error() -> None:
    module = _load()
    policy = yaml.safe_load(POLICY.read_text())
    policy["ptv2_languages"] = {"allow": ["en", "ja"], "deny": ["ja"]}

    with pytest.raises(ValueError, match="both allows and denies"):
        module.select_ptv23_arm(_rows(), config=policy, arm="B", target_assistant_tokens=1000)


def test_the_builder_cli_accepts_the_study_policy_and_writes_a_corpus(tmp_path: Path) -> None:
    """The study's own selector was reachable from tests and from nothing else.

    main() called select_arm, which reads a pool/category schema this policy
    does not have, so the documented invocation died on KeyError: 'arms' before
    reading a single candidate -- and the domain quotas, the SWE replay lanes
    and the language policy were all dead code in production. Every unit test
    above called select_ptv23_arm directly, which is exactly why none of them
    noticed. This one goes through the entrypoint.
    """
    module = _load()
    inventory = tmp_path / "inventory.jsonl"
    inventory.write_text(
        "\n".join(json.dumps(row) for row in _rows()) + "\n",
        encoding="utf-8",
    )
    corpus = tmp_path / "corpus"

    exit_code = module.main(
        [
            "--inventory",
            str(inventory),
            "--config",
            str(POLICY),
            "--arm",
            "B",
            "--target-assistant-tokens",
            "1000",
            "--tokenizer-sha256",
            "f" * 64,
            "--output-manifest",
            str(corpus / "selection.json"),
            "--output-corpus",
            str(corpus),
        ]
    )

    assert exit_code == 0
    manifest = json.loads((corpus / "selection.json").read_text(encoding="utf-8"))
    assert manifest["arm"] == "B"
    assert manifest["unique_assistant_tokens"] == 1000
    assert sorted(manifest["category_assistant_tokens"]) == [
        "ptv2/chat",
        "ptv2/code",
        "ptv2/math",
        "ptv2/multilingual",
        "ptv2/stem",
    ]
    assert list(corpus.glob("train-*.parquet"))


def test_a_config_matching_neither_selector_schema_is_refused(tmp_path: Path) -> None:
    module = _load()
    with pytest.raises(ValueError, match="declares neither"):
        module.select_for_config(_rows(), config={"seed": 1}, arm="B", target_assistant_tokens=10)
