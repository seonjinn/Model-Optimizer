from __future__ import annotations

import importlib.util
from pathlib import Path

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
