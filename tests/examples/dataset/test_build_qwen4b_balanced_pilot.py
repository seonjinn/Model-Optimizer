# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import importlib.util
import json
import sys
from collections import Counter
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
MODULE_PATH = REPO_ROOT / "examples/dataset/build_qwen4b_balanced_pilot.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("build_qwen4b_balanced_pilot", MODULE_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _row(split: str, index: int, *, duplicate_prompt: str | None = None) -> dict:
    prompt = duplicate_prompt or f"{split} prompt {index}"
    return {
        "messages": [
            {"role": "system", "content": "You are useful."},
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": f"answer {index}"},
        ]
    }


def _all_split_rows(extra_per_split: int = 0) -> dict[str, list[dict]]:
    counts = {
        "chat": 48_286,
        "math": 25_000,
        "code": 20_000,
        "stem": 25_000,
        "multilingual_de": 2_000,
        "multilingual_es": 2_000,
        "multilingual_fr": 2_000,
        "multilingual_it": 2_000,
        "multilingual_ja": 2_000,
    }
    return {
        split: [_row(split, index) for index in range(count + extra_per_split)]
        for split, count in counts.items()
    }


def _category_counts(selection, arm: str) -> Counter[str]:
    return Counter(row.category for row in selection.rows_for(arm))


def test_quota_selection_uses_exact_historical_and_balanced_quotas() -> None:
    """A changed quota or category mapping must change the selected row counts."""
    module = _load_module()

    selection = module.select_pilot_rows(_all_split_rows(), config=module.PilotConfig())

    assert _category_counts(selection, "historical-proportion") == {
        "chat": 48_286,
        "math": 18_420,
        "code": 13_462,
        "stem": 19_832,
    }
    assert _category_counts(selection, "balanced") == {
        "math": 25_000,
        "code": 20_000,
        "stem": 25_000,
        "chat": 20_000,
        "multilingual": 10_000,
    }
    assert len(selection.rows_for("historical-proportion")) == 100_000
    assert len(selection.rows_for("balanced")) == 100_000


def test_quota_selection_uses_exact_multilingual_language_quotas() -> None:
    """A language-quota regression must be visible in the selected balanced rows."""
    module = _load_module()

    selection = module.select_pilot_rows(_all_split_rows(), config=module.PilotConfig())

    assert Counter(row.split for row in selection.rows_for("balanced")) == {
        "math": 25_000,
        "code": 20_000,
        "stem": 25_000,
        "chat": 20_000,
        "multilingual_de": 2_000,
        "multilingual_es": 2_000,
        "multilingual_fr": 2_000,
        "multilingual_it": 2_000,
        "multilingual_ja": 2_000,
    }


def test_parallel_selection_is_stable_when_input_is_reversed() -> None:
    """Dropping rank ordering or depending on worker order must change these IDs."""
    module = _load_module()
    rows = _all_split_rows(extra_per_split=5)
    reversed_rows = {split: list(reversed(values)) for split, values in rows.items()}

    one_worker = module.select_pilot_rows(rows, config=module.PilotConfig(workers=1))
    many_workers = module.select_pilot_rows(reversed_rows, config=module.PilotConfig(workers=96))

    for arm in ("historical-proportion", "balanced"):
        assert [row.prompt_uuid for row in one_worker.rows_for(arm)] == [
            row.prompt_uuid for row in many_workers.rows_for(arm)
        ]


def test_duplicate_refill_is_global_and_excludes_held_out_prompt() -> None:
    """Removing global dedup, held-out filtering, or quota refill must fail this case."""
    module = _load_module()
    rows = _all_split_rows(extra_per_split=3)
    rows["math"][0] = _row("math", 0, duplicate_prompt="chat prompt 0")
    held_out = module.prompt_uuid_from_messages(rows["math"][1]["messages"])

    selection = module.select_pilot_rows(
        rows, config=module.PilotConfig(), held_out_prompt_uuids={held_out}
    )

    for arm in ("historical-proportion", "balanced"):
        selected = selection.rows_for(arm)
        identities = [row.prompt_uuid for row in selected]
        assert len(identities) == len(set(identities))
        assert held_out not in identities
        assert len(selected) == 100_000
    assert _category_counts(selection, "balanced")["math"] == 25_000


def test_seed_change_changes_selected_ids() -> None:
    """Ignoring the seed in the rank preimage must leave these selections unchanged."""
    module = _load_module()
    rows = _all_split_rows(extra_per_split=20)

    baseline = module.select_pilot_rows(rows, config=module.PilotConfig(seed=20_260_822))
    changed = module.select_pilot_rows(rows, config=module.PilotConfig(seed=7))

    assert [row.prompt_uuid for row in baseline.rows_for("balanced")] != [
        row.prompt_uuid for row in changed.rows_for("balanced")
    ]


def test_held_out_uuid_json_array_is_loaded_and_validated(tmp_path: Path) -> None:
    """Accepting a non-array held-out receipt would bypass evaluator exclusion."""
    module = _load_module()
    held_out = "d" * 64
    receipt = tmp_path / "held-out.json"
    receipt.write_text(json.dumps([held_out]), encoding="utf-8")

    assert module.load_held_out_prompt_uuids(receipt) == {held_out}
    receipt.write_text(json.dumps({"uuid": held_out}), encoding="utf-8")
    with pytest.raises(module.PilotError, match="JSON UUID array"):
        module.load_held_out_prompt_uuids(receipt)


class _MaskTokenizer:
    """A deterministic stand-in for the pinned tokenizer's chat-template boundary."""

    chat_template = "qwen3-template"

    def apply_chat_template(self, messages, **kwargs):
        assert kwargs["return_assistant_tokens_mask"] is True
        input_ids = list(range(len(messages) * 3))
        mask = [0, 0, 0] + [1, 1, 1] * (len(messages) - 1)
        return {"input_ids": input_ids, "assistant_masks": mask}


class _FailingTokenizer(_MaskTokenizer):
    def apply_chat_template(self, messages, **kwargs):
        if messages[-1]["content"] == "answer 1":
            raise RuntimeError("simulated worker failure")
        return super().apply_chat_template(messages, **kwargs)


def _tiny_config(module, *, minimum_assistant_tokens: int = 1):
    quota = module.PilotQuota("chat", "chat", 2)
    return module.PilotConfig(
        historical_quotas=(quota,),
        balanced_quotas=(quota,),
        minimum_assistant_tokens=minimum_assistant_tokens,
        source_revision="c" * 40,
    )


def _tiny_rows() -> dict[str, list[dict]]:
    return {"chat": [_row("chat", index) for index in range(3)]}


def test_token_count_uses_exact_assistant_mask() -> None:
    """Counting all template IDs rather than the assistant mask must fail this test."""
    module = _load_module()

    assert module.count_assistant_tokens(_MaskTokenizer(), _tiny_rows()["chat"][0]["messages"]) == 6


def test_manifest_replay_binds_data_file_and_self_hash(tmp_path: Path) -> None:
    """Omitting file or self-hash replay must leave a tampered bundle accepted."""
    module = _load_module()
    output = tmp_path / "pilot"

    completion = module.build_pilot_bundles(
        _tiny_rows(),
        config=_tiny_config(module),
        tokenizer=_MaskTokenizer(),
        tokenizer_path="/tokenizer",
        tokenizer_sha256="a" * 64,
        output_root=output,
        producer_source_commit="b" * 40,
    )

    assert completion.output_root == output
    module.verify_pilot_completion(output)
    manifest = output / "balanced" / "MANIFEST.json"
    assert manifest.is_file()
    data = output / "balanced" / "data.jsonl"
    data.write_text(data.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    with pytest.raises(module.PilotError, match="data file"):
        module.verify_pilot_completion(output)


def test_publication_failure_for_insufficient_assistant_tokens_has_no_final_root(
    tmp_path: Path,
) -> None:
    """Removing the 16M-equivalent pre-install gate would publish this insufficient arm."""
    module = _load_module()
    output = tmp_path / "pilot"

    with pytest.raises(module.PilotError, match="assistant-token minimum"):
        module.build_pilot_bundles(
            _tiny_rows(),
            config=_tiny_config(module, minimum_assistant_tokens=13),
            tokenizer=_MaskTokenizer(),
            tokenizer_path="/tokenizer",
            tokenizer_sha256="a" * 64,
            output_root=output,
            producer_source_commit="b" * 40,
        )
    assert not output.exists()


def test_publication_failure_from_worker_has_no_final_root(tmp_path: Path) -> None:
    """Installing before worker results are replayed would leave a final partial root."""
    module = _load_module()
    output = tmp_path / "pilot"

    with pytest.raises(module.PilotError, match="token worker"):
        module.build_pilot_bundles(
            _tiny_rows(),
            config=_tiny_config(module),
            tokenizer=_FailingTokenizer(),
            tokenizer_path="/tokenizer",
            tokenizer_sha256="a" * 64,
            output_root=output,
            producer_source_commit="b" * 40,
            workers=2,
        )
    assert not output.exists()


def test_no_replace_publication_preserves_existing_destination(tmp_path: Path) -> None:
    """Replacing an existing root would destroy the first receipt-bound publication."""
    module = _load_module()
    output = tmp_path / "pilot"
    arguments = {
        "config": _tiny_config(module),
        "tokenizer": _MaskTokenizer(),
        "tokenizer_path": "/tokenizer",
        "tokenizer_sha256": "a" * 64,
        "output_root": output,
        "producer_source_commit": "b" * 40,
    }
    module.build_pilot_bundles(_tiny_rows(), **arguments)
    before = (output / "COMPLETE.json").read_bytes()

    with pytest.raises(module.PilotError, match="already exists"):
        module.build_pilot_bundles(_tiny_rows(), **arguments)
    assert (output / "COMPLETE.json").read_bytes() == before
