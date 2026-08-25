# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import errno
import hashlib
import importlib.util
import json
import os
import sqlite3
import sys
import threading
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
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


class _Qwen3MaskTokenizer(_MaskTokenizer):
    eos_token_id = 151_645

    @staticmethod
    def convert_tokens_to_ids(token: str) -> int:
        return {"<|im_start|>": 151_644, "<|im_end|>": 151_645}[token]


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
    module._verify_bundle_root(output, production=False, tokenizer=_MaskTokenizer())
    manifest = output / "balanced" / "MANIFEST.json"
    assert manifest.is_file()
    data = output / "balanced" / "data.jsonl"
    data.write_text(data.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    with pytest.raises(module.PilotError, match="data file"):
        module._verify_bundle_root(output, production=False, tokenizer=_MaskTokenizer())


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
        module._verify_bundle_root(output, production=False, tokenizer=_MaskTokenizer())


def test_tokenization_keeps_inflight_work_bounded(monkeypatch, tmp_path: Path) -> None:
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
    import sqlite3

    connection = sqlite3.connect(tmp_path / "selected.sqlite")
    module._prepare_selection_database(connection)
    connection.executemany(
        "INSERT INTO selected VALUES (?, ?, ?, ?, ?, ?, ?)",
        [
            (
                "balanced",
                index,
                "a" * 63 + str(index),
                "chat",
                "chat",
                json.dumps(_row("chat", index)["messages"]),
                None,
            )
            for index in range(6)
        ],
    )

    assert (
        module._count_spooled_rows(
            connection, _MaskTokenizer(), "balanced", workers=2, training_sequence_length=4_096
        )
        == 36
    )
    assert (
        connection.execute("SELECT COUNT(*) FROM selected WHERE assistant_tokens = 6").fetchone()[0]
        == 6
    )
    connection.close()


def test_bundle_builder_streams_selected_bodies_without_materialized_selection(
    monkeypatch, tmp_path: Path
) -> None:
    """The builder must stream its verified output even if tuple selection is unavailable."""
    module = _load_module()

    def forbid_materialized_selection(*_args, **_kwargs):
        raise AssertionError("materialized selected messages")

    monkeypatch.setattr(module, "select_pilot_rows", forbid_materialized_selection)
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
    )

    assert (
        module._verify_bundle_root(output, production=False, tokenizer=_MaskTokenizer())[
            "balanced"
        ].row_count
        == 2
    )


def test_cross_split_duplicate_assignment_preserves_feasible_quotas() -> None:
    """Assigning a shared prompt greedily to chat must not starve math."""
    module = _load_module()
    shared_chat = _row("chat", 7, duplicate_prompt="a")
    shared_math = _row("math", 7, duplicate_prompt="a")
    chat_only = _row("chat", 0, duplicate_prompt="b")
    quotas = (
        module.PilotQuota("chat", "chat", 1),
        module.PilotQuota("math", "math", 1),
    )
    config = module.PilotConfig(
        historical_quotas=quotas,
        balanced_quotas=quotas,
        source_revision="c" * 40,
    )

    selection = module.select_pilot_rows(
        {"chat": [shared_chat, chat_only], "math": [shared_math]}, config=config
    )

    assert Counter(row.split for row in selection.historical_proportion) == {
        "chat": 1,
        "math": 1,
    }
    assert {row.messages[-1]["content"] for row in selection.historical_proportion} == {
        "answer 0",
        "answer 7",
    }


def test_capacitated_matching_uses_bounded_sql_operations(tmp_path: Path) -> None:
    """Restoring per-candidate or per-occupant SQL lookups must violate this bound."""
    module = _load_module()
    quotas = (
        module.PilotQuota("chat", "chat", 2_000),
        module.PilotQuota("math", "math", 2_000),
    )
    config = module.PilotConfig(
        historical_quotas=quotas,
        balanced_quotas=quotas,
        source_revision="c" * 40,
    )
    rows = {
        split: [_row(split, index, duplicate_prompt=f"shared {index}") for index in range(12_000)]
        for split in ("chat", "math")
    }
    connection = sqlite3.connect(tmp_path / "matching.sqlite")
    module._prepare_selection_database(connection)
    module._spool_candidates(connection, rows, config, set())
    selects: list[str] = []
    connection.set_trace_callback(
        lambda statement: (
            selects.append(statement)
            if statement.lstrip().upper().startswith(("SELECT", "WITH"))
            else None
        )
    )

    selected = module._select_arm_from_spool(
        connection, "balanced", quotas, config, materialize=False
    )

    connection.close()
    assert selected == ()
    assert len(selects) <= 3


def test_conflicting_response_tie_is_canonical_under_input_reversal() -> None:
    """A same-split prompt with two responses must publish input-order-independent bytes."""
    module = _load_module()
    first = _row("chat", 0, duplicate_prompt="same prompt")
    second = _row("chat", 1, duplicate_prompt="same prompt")
    quota = module.PilotQuota("chat", "chat", 1)
    config = module.PilotConfig(
        historical_quotas=(quota,),
        balanced_quotas=(quota,),
        source_revision="c" * 40,
    )

    forward = module.select_pilot_rows({"chat": [first, second]}, config=config)
    reverse = module.select_pilot_rows({"chat": [second, first]}, config=config)

    assert forward.historical_proportion == reverse.historical_proportion
    assert forward.historical_proportion[0].messages[-1]["content"] == "answer 0"


class _LongMaskTokenizer(_MaskTokenizer):
    def apply_chat_template(self, messages, **kwargs):
        assert kwargs["return_assistant_tokens_mask"] is True
        return {
            "input_ids": list(range(4_100)),
            "assistant_masks": [0] * 4_095 + [1] * 5,
        }


def test_token_count_matches_trainer_left_to_right_4096_truncation() -> None:
    """Assistant tokens beyond the trainer's first 4096 positions must not satisfy the gate."""
    module = _load_module()

    assert (
        module.count_assistant_tokens(_LongMaskTokenizer(), _tiny_rows()["chat"][0]["messages"])
        == 1
    )


def _rehash_receipts(module, output: Path, arm: str) -> None:
    manifest_path = output / arm / "MANIFEST.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["manifest_sha256"] = module._self_hash(manifest, "manifest_sha256")
    manifest_path.write_text(module.canonical_json(manifest) + "\n")
    complete_path = output / "COMPLETE.json"
    complete = json.loads(complete_path.read_text())
    complete["arms"][arm]["manifest_sha256"] = manifest["manifest_sha256"]
    complete["complete_sha256"] = module._self_hash(complete, "complete_sha256")
    complete_path.write_text(module.canonical_json(complete) + "\n")


def test_public_verifier_rejects_nonproduction_bundle_even_when_self_hashed(tmp_path: Path) -> None:
    """A valid tiny test receipt must never pass the production completion boundary."""
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

    with pytest.raises(module.PilotError, match="production"):
        module.verify_pilot_completion(output)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("seed", True, "seed"),
        ("arm", "balanced", "arm"),
        ("assistant_tokens", True, "assistant-token"),
        ("minimum_assistant_tokens", 0, "assistant-token"),
    ],
)
def test_semantic_manifest_tamper_fails_after_attacker_rehashes_receipts(
    tmp_path: Path, field: str, value: object, message: str
) -> None:
    """Self-hashes must not make semantically invalid typed manifest values acceptable."""
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
    manifest_path = output / "historical-proportion" / "MANIFEST.json"
    manifest = json.loads(manifest_path.read_text())
    manifest[field] = value
    manifest_path.write_text(module.canonical_json(manifest) + "\n")
    _rehash_receipts(module, output, "historical-proportion")

    with pytest.raises(module.PilotError, match=message):
        module._verify_bundle_root(output, production=False, tokenizer=_MaskTokenizer())


def test_cross_arm_tokenizer_pin_tamper_fails_after_full_rehash(tmp_path: Path) -> None:
    """Two independently valid manifests must not disagree on the tokenizer identity."""
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
    manifest_path = output / "balanced" / "MANIFEST.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["tokenizer"]["sha256"] = "f" * 64
    manifest_path.write_text(module.canonical_json(manifest) + "\n")
    _rehash_receipts(module, output, "balanced")

    with pytest.raises(module.PilotError, match="cross-arm"):
        module._verify_bundle_root(output, production=False, tokenizer=_MaskTokenizer())


def test_verifier_recomputes_assistant_tokens_after_full_attacker_rehash(
    monkeypatch, tmp_path: Path
) -> None:
    """Rehashing forged per-row token counts must not make fabricated evidence valid."""
    module = _load_module()
    output = tmp_path / "pilot"
    tokenizer_source = tmp_path / "tokenizer"
    tokenizer_source.mkdir()
    (tokenizer_source / "tokenizer.json").write_text('{"model":"Qwen3-4B"}\n')
    module._build_pilot_bundles_for_test(
        _tiny_rows(),
        config=_tiny_config(module),
        tokenizer=_MaskTokenizer(),
        tokenizer_path=tokenizer_source,
        tokenizer_sha256=_tree_digest(tokenizer_source),
        output_root=output,
        producer_source_commit="b" * 40,
    )
    arm = "balanced"
    data_path = output / arm / "data.jsonl"
    forged_rows = [json.loads(line) for line in data_path.read_text().splitlines()]
    for row in forged_rows:
        assert row["assistant_tokens"] == 6
        row["assistant_tokens"] = 1
    forged_data = "".join(module.canonical_json(row) + "\n" for row in forged_rows)
    data_path.write_text(forged_data, encoding="utf-8")
    token_digest = hashlib.sha256()
    for row in sorted(forged_rows, key=lambda value: value["prompt_uuid"]):
        token_digest.update(
            module.canonical_json([row["prompt_uuid"], row["assistant_tokens"]]).encode()
        )
        token_digest.update(b"\n")
    manifest_path = output / arm / "MANIFEST.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["data_file"] = {
        "path": "data.jsonl",
        "bytes": data_path.stat().st_size,
        "sha256": hashlib.sha256(data_path.read_bytes()).hexdigest(),
    }
    manifest["assistant_tokens"] = 2
    manifest["token_evidence_sha256"] = token_digest.hexdigest()
    manifest_path.write_text(module.canonical_json(manifest) + "\n")
    _rehash_receipts(module, output, arm)
    loaded_snapshots: list[Path] = []

    def load_snapshot(path: Path):
        loaded_snapshots.append(path)
        assert path != tokenizer_source
        return _Qwen3MaskTokenizer()

    monkeypatch.setattr(module, "_load_verified_qwen_tokenizer", load_snapshot)

    with pytest.raises(module.PilotError, match="assistant-token"):
        module._verify_bundle_root(output, production=False)
    assert len(loaded_snapshots) == 1
    assert not loaded_snapshots[0].exists()


def test_execution_parallelism_does_not_change_scientific_bundle_bytes(tmp_path: Path) -> None:
    """Changing CPU workers may change EXECUTION only, never data/manifest/completion identity."""
    module = _load_module()
    roots = [tmp_path / "serial", tmp_path / "parallel"]
    for root, workers in zip(roots, (1, 96), strict=True):
        module._build_pilot_bundles_for_test(
            _tiny_rows(),
            config=_tiny_config(module),
            tokenizer=_MaskTokenizer(),
            tokenizer_path="/tokenizer",
            tokenizer_sha256="a" * 64,
            output_root=root,
            producer_source_commit="b" * 40,
            workers=workers,
        )

    for relative in (
        Path("COMPLETE.json"),
        Path("balanced/data.jsonl"),
        Path("balanced/MANIFEST.json"),
        Path("historical-proportion/data.jsonl"),
        Path("historical-proportion/MANIFEST.json"),
    ):
        assert (roots[0] / relative).read_bytes() == (roots[1] / relative).read_bytes()
    assert (roots[0] / "balanced/EXECUTION.json").read_bytes() != (
        roots[1] / "balanced/EXECUTION.json"
    ).read_bytes()


def _tree_digest(root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(path for path in root.rglob("*") if path.is_file()):
        digest.update(path.relative_to(root).as_posix().encode())
        digest.update(b"\0")
        digest.update(hashlib.sha256(path.read_bytes()).digest())
    return digest.hexdigest()


def test_tokenizer_snapshot_is_private_nofollow_and_detached_from_source(tmp_path: Path) -> None:
    """Authentication and loading must use one private copy immune to later source mutation."""
    module = _load_module()
    source = tmp_path / "source"
    source.mkdir()
    (source / "tokenizer.json").write_text('{"version":"1"}\n')
    scratch = tmp_path / "scratch"
    scratch.mkdir()

    staged, digest = module._stage_verified_tokenizer_snapshot(
        source, _tree_digest(source), scratch
    )
    (source / "tokenizer.json").write_text('{"version":"forged"}\n')

    assert digest == _tree_digest(staged)
    assert (staged / "tokenizer.json").read_text() == '{"version":"1"}\n'
    assert staged.stat().st_mode & 0o777 == 0o700
    assert (staged / "tokenizer.json").stat().st_mode & 0o777 == 0o600


def test_tokenizer_snapshot_rejects_nested_symlink(tmp_path: Path) -> None:
    """Following a symlink in any tokenizer component would authenticate foreign bytes."""
    module = _load_module()
    source = tmp_path / "source"
    source.mkdir()
    outside = tmp_path / "outside.json"
    outside.write_text("foreign")
    (source / "nested").mkdir()
    (source / "nested/tokenizer.json").symlink_to(outside)
    scratch = tmp_path / "scratch"
    scratch.mkdir()

    with pytest.raises(module.PilotError, match=r"symlink|no-follow"):
        module._stage_verified_tokenizer_snapshot(source, "0" * 64, scratch)


def test_genuine_minimal_qwen_chat_template_marks_only_assistant_boundary(
    monkeypatch, tmp_path: Path
) -> None:
    """The authenticated real HF tokenizer must preserve Qwen ChatML generation masking."""
    monkeypatch.setenv("USE_TORCH", "0")
    monkeypatch.setenv("USE_TF", "0")
    monkeypatch.setenv("USE_FLAX", "0")
    import importlib.metadata

    class _DistributionMap(dict):
        def __getitem__(self, name: str) -> list[str]:
            return [name.replace("_", "-")]

    monkeypatch.setattr(importlib.metadata, "packages_distributions", _DistributionMap)
    pinned_versions = {
        "tqdm": "4.67.1",
        "regex": "2026.1.15",
        "packaging": "25.0",
        "filelock": "3.20.0",
        "numpy": "2.3.0",
        "tokenizers": "0.22.2",
        "huggingface-hub": "1.5.0",
        "safetensors": "0.8.0",
        "pyyaml": "6.0.2",
        "accelerate": "1.10.0",
    }

    def pinned_version(name: str) -> str:
        return pinned_versions.get(name.lower(), "1.0.0")

    monkeypatch.setattr(importlib.metadata, "version", pinned_version)
    from tokenizers import Tokenizer  # pyright: ignore[reportMissingImports]
    from tokenizers.models import WordLevel  # pyright: ignore[reportMissingImports]
    from tokenizers.pre_tokenizers import WhitespaceSplit  # pyright: ignore[reportMissingImports]
    from transformers import PreTrainedTokenizerFast  # pyright: ignore[reportMissingImports]

    module = _load_module()
    source = tmp_path / "qwen-tokenizer"
    backend = Tokenizer(
        WordLevel(
            {
                "[UNK]": 0,
                "<|im_start|>": 1,
                "<|im_end|>": 2,
                "system": 3,
                "user": 4,
                "assistant": 5,
                "You": 6,
                "are": 7,
                "useful.": 8,
                "question": 9,
                "answer": 10,
            },
            unk_token="[UNK]",
        )
    )
    backend.pre_tokenizer = WhitespaceSplit()
    template = (
        "{% for message in messages %}<|im_start|> {{ message['role'] }} "
        "{% if message['role'] == 'assistant' %}{% generation %}{{ message['content'] }}"
        "{% endgeneration %}{% else %}{{ message['content'] }}{% endif %} <|im_end|>\n"
        "{% endfor %}"
    )
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=backend,
        unk_token="[UNK]",
        chat_template=template,
    )
    tokenizer.save_pretrained(source)
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    snapshot, _digest = module._stage_verified_tokenizer_snapshot(
        source, _tree_digest(source), scratch
    )

    loaded = module._load_verified_qwen_tokenizer(snapshot)
    messages = [
        {"role": "system", "content": "You are useful."},
        {"role": "user", "content": "question"},
        {"role": "assistant", "content": "answer"},
    ]

    assert loaded.chat_template == template
    assert module.count_assistant_tokens(loaded, messages) == 1


def test_oci_preflight_authenticates_offline_qwen3_tokenizer_snapshot(
    monkeypatch, tmp_path: Path
) -> None:
    """The OCI gate must hash a detached tree and reject a non-Qwen tokenizer identity."""
    module = _load_module()
    source = tmp_path / "qwen3-4b-tokenizer"
    source.mkdir()
    (source / "tokenizer.json").write_text('{"model":"Qwen3-4B"}\n')
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    loaded_paths: list[Path] = []

    class _WrongTokenizer(_Qwen3MaskTokenizer):
        @staticmethod
        def convert_tokens_to_ids(token: str) -> int:
            return {"<|im_start|>": 1, "<|im_end|>": 2}[token]

    tokenizer_to_load: list[_Qwen3MaskTokenizer] = [_WrongTokenizer()]

    def load_snapshot(path: Path):
        loaded_paths.append(path)
        assert path != source
        assert (path / "tokenizer.json").read_text() == '{"model":"Qwen3-4B"}\n'
        return tokenizer_to_load[0]

    monkeypatch.setattr(module, "_load_verified_qwen_tokenizer", load_snapshot)
    with pytest.raises(module.PilotError, match="Qwen3 tokenizer identity"):
        module.preflight_qwen3_4b_tokenizer_snapshot(source, _tree_digest(source), scratch)
    tokenizer_to_load[0] = _Qwen3MaskTokenizer()
    evidence = module.preflight_qwen3_4b_tokenizer_snapshot(source, _tree_digest(source), scratch)

    assert evidence.sha256 == _tree_digest(source)
    assert evidence.im_start_token_id == 151_644
    assert evidence.im_end_token_id == 151_645
    assert evidence.probe_assistant_tokens == 6
    assert len(loaded_paths) == 2
    assert all(not path.exists() for path in loaded_paths)
    assert list(scratch.iterdir()) == []


def test_publication_fsyncs_entire_copied_tree_before_atomic_rename(
    monkeypatch, tmp_path: Path
) -> None:
    """Renaming a copied tree before syncing every inode can publish non-durable receipts."""
    module = _load_module()
    synced: set[tuple[int, int]] = set()
    real_fsync = os.fsync

    def record_fsync(descriptor: int) -> None:
        observed = os.fstat(descriptor)
        synced.add((observed.st_dev, observed.st_ino))
        real_fsync(descriptor)

    def assert_durable_then_rename(source: Path, destination: Path) -> None:
        copied = [source, *source.rglob("*")]
        expected = {
            (path.stat().st_dev, path.stat().st_ino)
            for path in copied
            if path.is_file() or path.is_dir()
        }
        assert expected <= synced
        source.rename(destination)

    monkeypatch.setattr(module.os, "fsync", record_fsync)
    monkeypatch.setattr(module, "_rename_noreplace", assert_durable_then_rename)
    module._build_pilot_bundles_for_test(
        _tiny_rows(),
        config=_tiny_config(module),
        tokenizer=_MaskTokenizer(),
        tokenizer_path="/tokenizer",
        tokenizer_sha256="a" * 64,
        output_root=tmp_path / "pilot",
        producer_source_commit="b" * 40,
    )


def test_public_failures_are_normalized_to_pilot_error(tmp_path: Path) -> None:
    """Filesystem and JSON scalar failures must not leak implementation exceptions."""
    module = _load_module()
    missing = tmp_path / "missing"

    with pytest.raises(module.PilotError):
        module.verify_pilot_completion(missing)
    with pytest.raises(module.PilotError):
        module.select_pilot_rows({"chat": iter([object()])}, config=_tiny_config(module))
    with pytest.raises(module.PilotError):
        module.load_held_out_prompt_uuids(None)

    class _ExplodingMessage(dict):
        def get(self, *_args, **_kwargs):
            raise RuntimeError("malformed mapping")

    with pytest.raises(module.PilotError):
        module.prompt_uuid_from_messages([_ExplodingMessage()])


@pytest.mark.skipif(not sys.platform.startswith("linux"), reason="renameat2 is Linux-only")
def test_real_linux_rename_noreplace_has_exactly_one_race_winner(tmp_path: Path) -> None:
    """The real kernel primitive must allow one publisher and return EEXIST to the loser."""
    module = _load_module()
    sources = [tmp_path / "one", tmp_path / "two"]
    for index, source in enumerate(sources):
        source.mkdir()
        (source / "value").write_text(str(index))
    destination = tmp_path / "published"
    barrier = threading.Barrier(2)

    def publish(source: Path) -> int:
        barrier.wait()
        try:
            module._rename_noreplace(source, destination)
        except OSError as error:
            return error.errno or -1
        return 0

    with ThreadPoolExecutor(max_workers=2) as executor:
        outcomes = list(executor.map(publish, sources))

    assert sorted(outcomes) == [0, errno.EEXIST]
    assert (destination / "value").read_text() in {"0", "1"}
