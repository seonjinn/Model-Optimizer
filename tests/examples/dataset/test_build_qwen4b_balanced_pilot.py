# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import errno
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
        "chat": 4,
        "math": 4,
        "code": 4,
        "stem": 4,
        "multilingual_de": 2,
        "multilingual_es": 2,
        "multilingual_fr": 2,
        "multilingual_it": 2,
        "multilingual_ja": 2,
    }
    return {
        split: [_row(split, index) for index in range(count + extra_per_split)]
        for split, count in counts.items()
    }


def _category_counts(selection, arm: str) -> Counter[str]:
    return Counter(row.category for row in selection.rows_for(arm))


def _scaled_config(module, *, seed: int = 20_260_822, workers: int = 1):
    return module.PilotConfig(
        seed=seed,
        workers=workers,
        historical_quotas=(
            module.PilotQuota("chat", "chat", 4),
            module.PilotQuota("math", "math", 2),
            module.PilotQuota("code", "code", 2),
            module.PilotQuota("stem", "stem", 2),
        ),
        balanced_quotas=(
            module.PilotQuota("math", "math", 1),
            module.PilotQuota("code", "code", 1),
            module.PilotQuota("stem", "stem", 1),
            module.PilotQuota("chat", "chat", 2),
            module.PilotQuota("multilingual", "multilingual_de", 1),
            module.PilotQuota("multilingual", "multilingual_es", 1),
            module.PilotQuota("multilingual", "multilingual_fr", 1),
            module.PilotQuota("multilingual", "multilingual_it", 1),
            module.PilotQuota("multilingual", "multilingual_ja", 1),
        ),
    )


def test_quota_selection_uses_exact_historical_and_balanced_quotas() -> None:
    """A changed quota or category mapping must change the selected row counts."""
    module = _load_module()

    config = _scaled_config(module)
    selection = module.select_pilot_rows(_all_split_rows(), config=config)

    assert tuple(quota.rows for quota in module.PilotConfig().historical_quotas) == (
        48_286,
        18_420,
        13_462,
        19_832,
    )

    assert _category_counts(selection, "historical-proportion") == {
        "chat": 4,
        "math": 2,
        "code": 2,
        "stem": 2,
    }
    assert _category_counts(selection, "balanced") == {
        "math": 1,
        "code": 1,
        "stem": 1,
        "chat": 2,
        "multilingual": 5,
    }
    assert len(selection.rows_for("historical-proportion")) == 10
    assert len(selection.rows_for("balanced")) == 10


def test_quota_selection_uses_exact_multilingual_language_quotas() -> None:
    """A language-quota regression must be visible in the selected balanced rows."""
    module = _load_module()

    selection = module.select_pilot_rows(_all_split_rows(), config=_scaled_config(module))

    assert Counter(row.split for row in selection.rows_for("balanced")) == {
        "math": 1,
        "code": 1,
        "stem": 1,
        "chat": 2,
        "multilingual_de": 1,
        "multilingual_es": 1,
        "multilingual_fr": 1,
        "multilingual_it": 1,
        "multilingual_ja": 1,
    }


def test_parallel_selection_is_stable_when_input_is_reversed() -> None:
    """Dropping rank ordering or depending on worker order must change these IDs."""
    module = _load_module()
    rows = _all_split_rows(extra_per_split=5)
    reversed_rows = {split: list(reversed(values)) for split, values in rows.items()}

    one_worker = module.select_pilot_rows(rows, config=_scaled_config(module, workers=1))
    many_workers = module.select_pilot_rows(
        reversed_rows, config=_scaled_config(module, workers=96)
    )

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
        rows, config=_scaled_config(module), held_out_prompt_uuids={held_out}
    )

    for arm in ("historical-proportion", "balanced"):
        selected = selection.rows_for(arm)
        identities = [row.prompt_uuid for row in selected]
        assert len(identities) == len(set(identities))
        assert held_out not in identities
        assert len(selected) == 10
    assert _category_counts(selection, "balanced")["math"] == 1


def test_seed_change_changes_selected_ids() -> None:
    """Ignoring the seed in the rank preimage must leave these selections unchanged."""
    module = _load_module()
    rows = _all_split_rows(extra_per_split=20)

    baseline = module.select_pilot_rows(rows, config=_scaled_config(module, seed=20_260_822))
    changed = module.select_pilot_rows(rows, config=_scaled_config(module, seed=7))

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

    completion = module._build_pilot_bundles_for_test(
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
        module._build_pilot_bundles_for_test(
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
        module._build_pilot_bundles_for_test(
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
    module._build_pilot_bundles_for_test(_tiny_rows(), **arguments)
    before = (output / "COMPLETE.json").read_bytes()

    with pytest.raises(module.PilotError, match="already exists"):
        module._build_pilot_bundles_for_test(_tiny_rows(), **arguments)
    assert (output / "COMPLETE.json").read_bytes() == before


def test_public_build_rejects_nonproduction_scientific_identity(tmp_path: Path) -> None:
    """Letting test-sized quotas reach the public builder would publish a false pilot."""
    module = _load_module()

    with pytest.raises(module.PilotError, match="frozen production identity"):
        module.build_pilot_bundles(
            _tiny_rows(),
            config=_tiny_config(module),
            tokenizer=_MaskTokenizer(),
            tokenizer_path="/tokenizer",
            tokenizer_sha256="a" * 64,
            output_root=tmp_path / "pilot",
            producer_source_commit="b" * 40,
        )


def test_held_out_uuid_rejects_uppercase_hex() -> None:
    """Case-insensitive held-out UUIDs must not silently miss canonical identities."""
    module = _load_module()

    with pytest.raises(module.PilotError, match="SHA-256"):
        module.select_pilot_rows({"chat": []}, held_out_prompt_uuids={"A" * 64})


def test_quota_rejects_boolean_row_count() -> None:
    """Treating True as one would alter the scientific quota contract."""
    module = _load_module()

    with pytest.raises(module.PilotError, match="positive integer"):
        module.PilotConfig(
            historical_quotas=(module.PilotQuota("chat", "chat", True),),
            balanced_quotas=(module.PilotQuota("chat", "chat", True),),
        )


def test_linux_noreplace_maps_racing_destination_to_pilot_error(
    monkeypatch, tmp_path: Path
) -> None:
    """An EEXIST race during the final install must never replace another publisher."""
    module = _load_module()
    output = tmp_path / "pilot"

    def race(_source, _destination):
        raise OSError(errno.EEXIST, "exists")

    monkeypatch.setattr(module, "_rename_noreplace", race)
    with pytest.raises(module.PilotError, match="already exists"):
        module._build_pilot_bundles_for_test(
            _tiny_rows(),
            config=_tiny_config(module),
            tokenizer=_MaskTokenizer(),
            tokenizer_path="/tokenizer",
            tokenizer_sha256="a" * 64,
            output_root=output,
            producer_source_commit="b" * 40,
        )
    assert not output.exists()


class _OneShotRows:
    def __init__(self, rows: list[dict]) -> None:
        self._rows = rows
        self.iterations = 0

    def __iter__(self):
        if self.iterations:
            raise AssertionError("source was replayed")
        self.iterations += 1
        yield from self._rows


def test_selection_spools_one_shot_input_privately_without_durable_partial(tmp_path: Path) -> None:
    """A selector that lists/replays source rows or leaks its spool must fail this test."""
    module = _load_module()
    quota = module.PilotQuota("chat", "chat", 2)
    config = module.PilotConfig(
        historical_quotas=(quota,), balanced_quotas=(quota,), source_revision="c" * 40
    )
    source = _OneShotRows([_row("chat", index) for index in range(5)])

    selection = module.select_pilot_rows({"chat": source}, config=config, scratch_root=tmp_path)

    assert len(selection.historical_proportion) == 2
    assert source.iterations == 1
    assert list(tmp_path.iterdir()) == []


def test_execution_metadata_cannot_override_worker_receipt(tmp_path: Path) -> None:
    """Allowing caller metadata to replace worker evidence makes the receipt untrustworthy."""
    module = _load_module()
    output = tmp_path / "pilot"
    module._build_pilot_bundles_for_test(
        _tiny_rows(),
        config=_tiny_config(module),
        tokenizer=_MaskTokenizer(),
        tokenizer_path="/tokenizer",
        tokenizer_sha256="a" * 64,
        output_root=output,
        producer_source_commit="b" * 40,
        workers=2,
        execution={"requested_workers": 999, "effective_workers": 999},
    )

    execution = json.loads((output / "balanced" / "EXECUTION.json").read_text())
    assert execution["requested_workers"] == 1
    assert execution["effective_workers"] == 2


def test_execution_receipt_tamper_is_rejected_independently(tmp_path: Path) -> None:
    """Skipping execution-descriptor replay accepts a modified allocation receipt."""
    module = _load_module()
    output = tmp_path / "pilot"
    module._build_pilot_bundles_for_test(
        _tiny_rows(),
        config=_tiny_config(module),
        tokenizer=_MaskTokenizer(),
        tokenizer_path="/tokenizer",
        tokenizer_sha256="a" * 64,
        output_root=output,
        producer_source_commit="b" * 40,
    )
    execution = output / "balanced" / "EXECUTION.json"
    execution.write_text('{"effective_workers":99}\n', encoding="utf-8")

    with pytest.raises(module.PilotError, match="execution receipt"):
        module.verify_pilot_completion(output)


def test_tokenization_keeps_inflight_work_bounded(monkeypatch) -> None:
    """Eagerly submitting every selected row must overflow this bounded executor."""
    module = _load_module()

    class _Future:
        def __init__(self, value: int) -> None:
            self._value = value

        def result(self) -> int:
            return self._value

    executors = []

    class _BoundedExecutor:
        def __init__(self, *, max_workers: int) -> None:
            self._max_workers = max_workers
            self._submitted = 0
            executors.append(self)

        def __enter__(self):
            return self

        def __exit__(self, *_args) -> None:
            return None

        def submit(self, function, row):
            self._submitted += 1
            if self._submitted > self._max_workers * 2:
                raise AssertionError("unbounded token work")
            return _Future(function(row))

    def complete_one(pending, **_kwargs):
        executors[0]._submitted -= 1
        return {next(iter(pending))}, set()

    monkeypatch.setattr(module, "ThreadPoolExecutor", _BoundedExecutor)
    monkeypatch.setattr(module, "wait", complete_one)
    rows = tuple(
        module.PilotRow(
            "a" * 63 + str(index), "chat", "chat", json.dumps(_row("chat", index)["messages"])
        )
        for index in range(6)
    )

    assert module._count_selected_rows(_MaskTokenizer(), rows, workers=2) == (6,) * 6
