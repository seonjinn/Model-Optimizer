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
import importlib.util
import json
import shutil
import sys
from pathlib import Path
from types import MappingProxyType, SimpleNamespace
from typing import Any

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
MODULE_PATH = REPO_ROOT / "examples/dataset/build_specdec_inventory.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("build_specdec_inventory", MODULE_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    sys.path.insert(0, str(MODULE_PATH.parent))
    try:
        spec.loader.exec_module(module)
    finally:
        sys.path.pop(0)
    return module


class FakeTokenizer:
    def apply_chat_template(self, messages, **kwargs):
        assert kwargs["return_assistant_tokens_mask"] is True
        assert kwargs["return_dict"] is True
        tools = kwargs.get("tools")
        input_ids: list[int] = []
        mask: list[int] = []
        for message in messages:
            tokens = [len(str(message.get("content", ""))) + 1]
            if message.get("tool_calls"):
                tokens.append(99)
            input_ids.extend(tokens)
            mask.extend([int(message["role"] == "assistant")] * len(tokens))
        if tools:
            input_ids.insert(0, 77)
            mask.insert(0, 0)
        return {"input_ids": input_ids, "assistant_masks": mask}


class CandidateTokenizer:
    tokenizer_sha256 = "f" * 64

    def apply_chat_template(self, messages, **kwargs):
        assert kwargs["add_generation_prompt"] is True
        return {"input_ids": [index + 1 for index, _ in enumerate(messages)]}


class MappingCandidateTokenizer(CandidateTokenizer):
    def apply_chat_template(self, messages, **kwargs):
        return MappingProxyType(super().apply_chat_template(messages, **kwargs))


def test_candidate_tokenizer_accepts_transformers_mapping_contract() -> None:
    module = _load_module()

    assert module._candidate_tokenize(
        MappingCandidateTokenizer(),
        [{"role": "user", "content": "hello"}],
        [],
        add_generation_prompt=True,
    ) == (1,)


def test_candidate_process_pool_is_byte_identical_and_bounded(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _load_module()
    source = module.SourceIdentity(
        repository_id="nvidia/Nemotron-Post-Training-Dataset-v2",
        configuration="default",
        split="chat",
        revision=module.APPROVED_PTV2_REVISION,
        license_expression="ODC-BY-1.0",
        approved_use=True,
        cell="chat",
        lane="target-synth",
        files=(),
    )
    files = []
    excluded = {
        "messages": json.dumps(
            [
                {"role": "user", "content": "explode-before-tokenization"},
                {"role": "assistant", "content": "answer"},
            ]
        ),
        "tools": "[]",
        "language": ["excluded", "invalid"],
    }
    duplicate = {
        "messages": json.dumps(
            [
                {"role": "user", "content": "deduplicate-before-tokenization"},
                {"role": "assistant", "content": "answer"},
            ]
        ),
        "tools": "[]",
    }
    rejected_duplicate = {
        "messages": json.dumps(
            [
                {"role": "user", "content": "reject-duplicate-after-tokenization"},
                {"role": "assistant", "content": "answer"},
            ]
        ),
        "tools": "[]",
    }
    invalid_language = {
        "messages": json.dumps(
            [
                {"role": "user", "content": "invalid-language"},
                {"role": "assistant", "content": "answer"},
            ]
        ),
        "tools": "[]",
        "language": ["not", "a", "language"],
    }
    invalid_tools = {
        "messages": json.dumps(
            [
                {"role": "user", "content": "invalid-tools"},
                {"role": "assistant", "content": "answer"},
            ]
        ),
        "tools": '{"not":"a-list"}',
    }
    for index in range(201):
        path = tmp_path / f"shard-{index}.jsonl"
        rows = [
            {
                "messages": json.dumps(
                    [
                        {"role": "user", "content": f"question-{index}-{row}"},
                        {"role": "assistant", "content": "answer"},
                    ]
                ),
                "tools": "[]",
            }
            for row in range(3)
        ]
        if index == 0:
            rows[:0] = [
                excluded,
                duplicate,
                duplicate,
                rejected_duplicate,
                rejected_duplicate,
                invalid_language,
                invalid_tools,
            ]
        path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
        descriptor = module.SourceFile(path.name, path.stat().st_size, module.sha256_file(path))
        files.append((source, descriptor, path))
    inventory = SimpleNamespace(manifest_sha256="a" * 64, staged_root=tmp_path)
    monkeypatch.setattr(module, "_verified_candidate_files", lambda _inventory: files)
    monkeypatch.setattr(module, "_declared_candidate_files", lambda _inventory: (files, {}))
    monkeypatch.setattr(module, "_staged_tree_snapshot", lambda _root: {})
    excluded_messages, excluded_tools, *_ = module._ptv2_target_prompt_identity(excluded)
    excluded_uuid = module.sha256_bytes(
        module.canonicalize_prompt(excluded_messages, excluded_tools)
    )
    baseline = module.make_exclusion_receipt("baseline", (excluded_uuid,))
    held_out = module.make_exclusion_receipt("held-out", ())

    class ObservedTokenizer(CandidateTokenizer):
        def __init__(self, marker: Path) -> None:
            self.marker = marker

        def apply_chat_template(self, messages, **kwargs):
            if any(message.get("content") == "explode-before-tokenization" for message in messages):
                self.marker.with_suffix(".excluded").write_bytes(b"called")
                raise RuntimeError("excluded rows must not be tokenized")
            if any(
                message.get("content") == "deduplicate-before-tokenization" for message in messages
            ):
                with self.marker.open("ab") as stream:
                    stream.write(b"1")
            if any(
                message.get("content") == "reject-duplicate-after-tokenization"
                for message in messages
            ):
                with self.marker.with_suffix(".rejected").open("ab") as stream:
                    stream.write(b"1")
                raise ValueError("conversation exceeds the 32K inventory limit: fixture")
            return super().apply_chat_template(messages, **kwargs)

    serial_marker = tmp_path / "serial-tokenizer-calls"
    serial = module.build_candidate_inventory(
        inventory,
        tokenizer=ObservedTokenizer(serial_marker),
        tokenizer_sha256="f" * 64,
        baseline_exclusion=baseline,
        held_out_exclusion=held_out,
        storage_dir=tmp_path,
    )
    parallel = module.build_candidate_inventory(
        inventory,
        tokenizer=ObservedTokenizer(tmp_path / "parallel-tokenizer-calls"),
        tokenizer_sha256="f" * 64,
        baseline_exclusion=baseline,
        held_out_exclusion=held_out,
        storage_dir=tmp_path,
        workers=96,
        source_commit="1" * 40,
    )
    try:
        assert module.candidate_inventory_bytes(serial) == module.candidate_inventory_bytes(
            parallel
        )
        assert serial.inventory_sha256 == parallel.inventory_sha256
        assert parallel.execution_receipt is not None
        assert parallel.execution_receipt["effective_workers"] == 96
        assert parallel.execution_receipt["declared_shard_count"] == 201
        assert set(parallel.execution_receipt["thread_environment"].values()) == {"1"}
        assert parallel.diagnostic_receipt is not None
        diagnostics = parallel.diagnostic_receipt
        assert diagnostics["schema_version"] == 1
        assert diagnostics["source_commit"] == "1" * 40
        assert diagnostics["source_manifest_sha256"] == "a" * 64
        assert diagnostics["declared_shard_count"] == 201
        assert len(diagnostics["shards"]) == 201
        assert sum(shard["phase1_row_count"] for shard in diagnostics["shards"]) == 610
        assert sum(shard["accepted_count"] for shard in diagnostics["shards"]) == len(parallel.rows)
        first = diagnostics["shards"][0]
        assert first["classification_counts"]["historical_exclusion"] == 1
        assert first["tokenization_counts"]["context_too_long"] == 2
        assert first["accepted_count"] == 4
        assert first["raw_row_schema"]["messages"] == "str"
        assert first["raw_row_schema"]["tools"] == "str"
        assert (
            diagnostics["reason_exemplars"]["historical_exclusion"][0]["prompt_uuid"]
            == excluded_uuid
        )
        body = dict(diagnostics)
        claimed = body.pop("receipt_sha256")
        assert claimed == module.sha256_bytes(module.canonical_json(body))
        assert parallel.quarantine_counts["historical_exclusion"] == 1
        assert not (tmp_path / "parallel-tokenizer-calls.excluded").exists()
        assert serial_marker.read_bytes() == b"11"
        assert (tmp_path / "parallel-tokenizer-calls").read_bytes() == b"11"
        assert (tmp_path / "parallel-tokenizer-calls.rejected").read_bytes() == b"11"
        assert parallel.quarantine_counts["context_too_long"] == 2
        assert parallel.quarantine_counts["invalid_language"] == 1
        assert parallel.quarantine_counts["invalid_tools"] == 1
    finally:
        serial.close()
        parallel.close()


def test_candidate_worker_count_respects_all_bounds() -> None:
    module = _load_module()

    assert module.resolve_candidate_worker_count(
        requested_workers=96,
        declared_shard_count=201,
        environ={"SLURM_CPUS_PER_TASK": "48"},
    ) == (48, 48)


def test_candidate_diagnostic_compares_post_tokenization_accepted_rows() -> None:
    diagnostic = (MODULE_PATH.parent / "diagnose_qwen4b_candidate_shard.py").read_text(
        encoding="utf-8"
    )

    assert "_tokenize_candidate_shard(" in diagnostic
    assert "SELECT payload FROM tokenized ORDER BY source_row_index" in diagnostic
    assert '"phase2_row_count": tokenization_result.row_count' in diagnostic
    assert 'first_payload = payload["candidate"]' not in diagnostic
    assert '"encoded_is_dict": isinstance(encoded, dict)' in diagnostic
    assert '"encoded_is_mapping": isinstance(encoded, Mapping)' in diagnostic
    assert '"exception_message": str(error)' in diagnostic


def test_candidate_diagnostic_launcher_requires_explicit_shard() -> None:
    launcher = (
        MODULE_PATH.parents[2]
        / "tools/launcher/common/specdec/run_qwen4b_candidate_diagnostic.sbatch"
    ).read_text(encoding="utf-8")

    assert "DIAGNOSTIC_RECEIPT SHARD_INDEX" in launcher
    assert '--shard-index "$SHARD_INDEX"' in launcher


def _write_bound_source(tmp_path: Path, rows: list[dict]) -> Path:
    module = _load_module()
    data = tmp_path / "source.jsonl"
    data.write_text("".join(json.dumps(row) + "\n" for row in rows))
    manifest = tmp_path / "source-manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "files": [
                    {
                        "path": data.name,
                        "bytes": data.stat().st_size,
                        "sha256": module.sha256_file(data),
                    }
                ],
            }
        )
        + "\n"
    )
    return manifest


def test_inventory_pretokenizes_synthesis_and_binds_source_hashes(tmp_path: Path) -> None:
    module = _load_module()
    manifest = _write_bound_source(
        tmp_path,
        [
            {
                "prompt_id": "p1",
                "messages": [
                    {"role": "user", "content": "question"},
                    {"role": "assistant", "content": "answer"},
                ],
                "_synthesis_assistant_tokens": 1,
            }
        ],
    )
    spec = module.InventorySource(
        source_id="nvidia/Nemotron-Post-Training-Dataset-v2",
        source_revision="a" * 40,
        license="ODC-BY-1.0",
        pool="ptv2",
        category="math",
        language="en",
        response_source="target-synth",
        tool_lane="none",
        manifest_path=manifest,
    )

    rows = module.build_inventory_rows(
        [spec], tokenizer=FakeTokenizer(), tokenizer_sha256="f" * 64, training_seq_len=4096
    )

    assert rows[0]["assistant_tokens"] == 1
    assert rows[0]["full_token_count"] == 2
    assert rows[0]["context_bucket"] == "le4k"
    assert rows[0]["source_manifest_sha256"] == module.sha256_file(manifest)
    assert rows[0]["source_file_sha256"]
    assert rows[0]["response_source"] == "target-synth"


def test_inventory_preserves_tool_lane_and_exact_assistant_mask(tmp_path: Path) -> None:
    module = _load_module()
    manifest = _write_bound_source(
        tmp_path,
        [
            {
                "prompt_id": "tool-1",
                "tools": [{"type": "function", "function": {"name": "shell"}}],
                "messages": [
                    {"role": "user", "content": "run"},
                    {"role": "assistant", "content": "", "tool_calls": [{"id": "c1"}]},
                    {"role": "tool", "content": "ok", "tool_call_id": "c1"},
                    {"role": "assistant", "content": "done"},
                ],
            }
        ],
    )
    spec = module.InventorySource(
        source_id="nvidia/Nemotron-SFT-Agentic-v2",
        source_revision="b" * 40,
        license="NVIDIA-Open-Model-License",
        pool="ptv3",
        category="agentic_tool",
        language="en",
        response_source="trace-replay",
        tool_lane="recorded-trace",
        manifest_path=manifest,
    )

    rows = module.build_inventory_rows(
        [spec], tokenizer=FakeTokenizer(), tokenizer_sha256="e" * 64, training_seq_len=4096
    )

    assert rows[0]["tools"][0]["function"]["name"] == "shell"
    assert rows[0]["tool_lane"] == "recorded-trace"
    assert rows[0]["assistant_tokens"] == 3
    assert sum(rows[0]["loss_mask"]) == 3


def test_inventory_rejects_mutated_manifested_source(tmp_path: Path) -> None:
    module = _load_module()
    manifest = _write_bound_source(
        tmp_path,
        [{"prompt_id": "p1", "messages": [{"role": "assistant", "content": "a"}]}],
    )
    (tmp_path / "source.jsonl").write_text("mutated\n")
    spec = module.InventorySource(
        source_id="source",
        source_revision="c" * 40,
        license="license",
        pool="ptv2",
        category="math",
        language="en",
        response_source="target-synth",
        tool_lane="none",
        manifest_path=manifest,
    )

    with pytest.raises(ValueError, match="source file identity mismatch"):
        module.build_inventory_rows(
            [spec], tokenizer=FakeTokenizer(), tokenizer_sha256="d" * 64, training_seq_len=4096
        )


def test_tokenizer_digest_binds_exact_serialization_files(tmp_path: Path) -> None:
    module = _load_module()
    (tmp_path / "tokenizer.json").write_text('{"version":"1"}\n')
    (tmp_path / "tokenizer_config.json").write_text('{"chat_template":"x"}\n')
    (tmp_path / "model.safetensors").write_text("ignored weights\n")

    first = module.tokenizer_snapshot_sha256(tmp_path)
    (tmp_path / "model.safetensors").write_text("changed weights\n")
    assert module.tokenizer_snapshot_sha256(tmp_path) == first
    (tmp_path / "tokenizer.json").write_text('{"version":"2"}\n')
    assert module.tokenizer_snapshot_sha256(tmp_path) != first


def test_tokenizer_snapshot_receipt_authenticates_exact_files_and_template(tmp_path: Path) -> None:
    module = _load_module()
    snapshot = tmp_path / "tokenizer"
    snapshot.mkdir()
    (snapshot / "tokenizer.json").write_text('{"version":"1.0"}', encoding="utf-8")
    (snapshot / "tokenizer_config.json").write_text(
        json.dumps({"chat_template": "{{ messages }}"}), encoding="utf-8"
    )
    receipt_path = tmp_path / "TOKENIZER_SNAPSHOT.json"

    written = module.write_tokenizer_snapshot_receipt(snapshot, receipt_path)
    loaded = module.load_tokenizer_snapshot(receipt_path, written.receipt_sha256)

    assert loaded == written
    assert loaded.tokenizer_sha256 == module.tokenizer_snapshot_sha256(snapshot)
    assert (
        loaded.chat_template_sha256
        == hashlib.sha256(module.canonical_json("{{ messages }}")).hexdigest()
    )


def test_authenticated_tiny_tokenizer_snapshot_loads_without_caller_adapter(
    tmp_path: Path,
) -> None:
    module = _load_module()
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel

    snapshot_root = tmp_path / "tiny-tokenizer"
    snapshot_root.mkdir()
    Tokenizer(WordLevel({"[UNK]": 0, "hello": 1}, unk_token="[UNK]")).save(
        str(snapshot_root / "tokenizer.json")
    )
    (snapshot_root / "tokenizer_config.json").write_text(
        json.dumps(
            {
                "tokenizer_class": "PreTrainedTokenizerFast",
                "unk_token": "[UNK]",
                "chat_template": "{% for message in messages %}{{ message['content'] }}{% endfor %}",
            }
        ),
        encoding="utf-8",
    )
    receipt = tmp_path / "TINY_TOKENIZER_SNAPSHOT.json"
    written = module.write_tokenizer_snapshot_receipt(snapshot_root, receipt)

    loaded, tokenizer = module._authenticated_snapshot_tokenizer(receipt, written.receipt_sha256)

    assert loaded == written
    encoded = tokenizer.apply_chat_template([{"role": "user", "content": "hello"}], tokenize=True)
    assert encoded["input_ids"] == [1]


@pytest.mark.parametrize(
    "mutation", ["file", "template", "extra", "directory", "symlink", "receipt"]
)
def test_tokenizer_snapshot_receipt_rejects_forgery(tmp_path: Path, mutation: str) -> None:
    module = _load_module()
    snapshot = tmp_path / "tokenizer"
    snapshot.mkdir()
    tokenizer_json = snapshot / "tokenizer.json"
    tokenizer_json.write_text('{"version":"1.0"}', encoding="utf-8")
    config = snapshot / "tokenizer_config.json"
    config.write_text(json.dumps({"chat_template": "{{ messages }}"}), encoding="utf-8")
    receipt_path = tmp_path / "TOKENIZER_SNAPSHOT.json"
    written = module.write_tokenizer_snapshot_receipt(snapshot, receipt_path)

    expected = written.receipt_sha256
    if mutation == "file":
        tokenizer_json.write_text('{"version":"forged"}', encoding="utf-8")
    elif mutation == "template":
        config.write_text(json.dumps({"chat_template": "forged"}), encoding="utf-8")
    elif mutation == "extra":
        (snapshot / "undeclared.txt").write_text("extra", encoding="utf-8")
    elif mutation == "directory":
        (snapshot / "undeclared").mkdir()
    elif mutation == "symlink":
        (snapshot / "alias.json").symlink_to("tokenizer.json")
    else:
        expected = "f" * 64

    with pytest.raises(ValueError, match=r"tokenizer snapshot|receipt"):
        module.load_tokenizer_snapshot(receipt_path, expected)


def test_tokenizer_snapshot_rejects_symlinked_intermediate_directory(tmp_path: Path) -> None:
    module = _load_module()
    receipts = tmp_path / "receipts"
    receipts.mkdir()
    outside = tmp_path / "outside"
    snapshot = outside / "tokenizer"
    snapshot.mkdir(parents=True)
    (snapshot / "tokenizer.json").write_text('{"version":"1.0"}', encoding="utf-8")
    (snapshot / "tokenizer_config.json").write_text(
        json.dumps({"chat_template": "{{ messages }}"}), encoding="utf-8"
    )
    (receipts / "alias").symlink_to(outside, target_is_directory=True)

    with pytest.raises(ValueError, match=r"no-follow|symlink|snapshot path"):
        module.write_tokenizer_snapshot_receipt(
            receipts / "alias" / "tokenizer", receipts / "TOKENIZER_SNAPSHOT.json"
        )


def test_tokenizer_snapshot_rejects_root_inode_retarget_during_internal_load(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _load_module()
    snapshot = tmp_path / "tokenizer"
    snapshot.mkdir()
    (snapshot / "tokenizer.json").write_text('{"version":"1.0"}', encoding="utf-8")
    (snapshot / "tokenizer_config.json").write_text(
        json.dumps({"chat_template": "{{ messages }}"}), encoding="utf-8"
    )
    receipt = tmp_path / "TOKENIZER_SNAPSHOT.json"
    written = module.write_tokenizer_snapshot_receipt(snapshot, receipt)

    def retarget_root(_snapshot):
        original = tmp_path / "original-tokenizer"
        snapshot.rename(original)
        shutil.copytree(original, snapshot)
        return CandidateTokenizer()

    monkeypatch.setattr(module, "_load_tokenizer_from_snapshot", retarget_root)

    with pytest.raises(ValueError, match=r"root.*changed|inode"):
        module._authenticated_snapshot_tokenizer(receipt, written.receipt_sha256)


def test_inventory_reads_explicit_raw_json_parquet_shards(tmp_path: Path) -> None:
    module = _load_module()
    import pyarrow as pa  # pyright: ignore[reportMissingImports]
    import pyarrow.parquet as pq  # pyright: ignore[reportMissingImports]

    path = tmp_path / "raw.parquet"
    pq.write_table(
        pa.table(
            {
                "raw_json": [
                    json.dumps(
                        {
                            "prompt_id": "raw-1",
                            "messages": [
                                {"role": "user", "content": "q"},
                                {"role": "assistant", "content": "a"},
                            ],
                        }
                    )
                ]
            }
        ),
        path,
        compression="zstd",
    )

    assert next(iter(module._iter_rows(path)))["prompt_id"] == "raw-1"


def test_candidate_inventory_verifies_every_source_before_parsing_rows(tmp_path: Path) -> None:
    module = _load_module()
    valid = tmp_path / "valid.jsonl"
    valid.write_text("not-json\n")
    malformed = tmp_path / "malformed.jsonl"
    malformed.write_text('{"messages":[{"role":"user","content":"valid"}]}\n')
    first_staged = tmp_path / "sources/fixture/source" / ("a" * 40) / valid.name
    first_staged.parent.mkdir(parents=True)
    first_staged.write_bytes(valid.read_bytes())
    second_staged = tmp_path / "sources/fixture/source" / ("b" * 40) / malformed.name
    second_staged.parent.mkdir(parents=True)
    second_staged.write_bytes(malformed.read_bytes())
    canonical_manifest = b"{}\n"
    inventory = module.SourceInventory(
        schema_version=1,
        name="fixture",
        sources=(
            module.SourceIdentity(
                repository_id="fixture/source",
                configuration="default",
                split="valid",
                revision="a" * 40,
                license_expression="Apache-2.0",
                approved_use=True,
                cell="math",
                lane="target-synth",
                files=(
                    module.SourceFile(
                        "valid.jsonl", valid.stat().st_size, module.sha256_file(valid)
                    ),
                ),
            ),
            module.SourceIdentity(
                repository_id="fixture/source",
                configuration="default",
                split="malformed",
                revision="b" * 40,
                license_expression="Apache-2.0",
                approved_use=True,
                cell="math",
                lane="target-synth",
                files=(
                    module.SourceFile(
                        "malformed.jsonl",
                        malformed.stat().st_size + 1,
                        module.sha256_file(malformed),
                    ),
                ),
            ),
        ),
        manifest_sha256=hashlib.sha256(canonical_manifest).hexdigest(),
        canonical_manifest=canonical_manifest,
        raw_counts={"math": 2},
        staged_root=tmp_path,
    )

    with pytest.raises(ValueError, match="source file identity mismatch"):
        module.build_candidate_inventory(
            inventory,
            tokenizer=FakeTokenizer(),
            tokenizer_sha256="f" * 64,
            baseline_exclusion=module.make_exclusion_receipt("baseline", set()),
            held_out_exclusion=module.make_exclusion_receipt("held-out", set()),
        )


def test_candidate_inventory_normalizes_raw_ptv2_cells_for_bprime_selection(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A full Task3 PTV2 inventory reaches the genuine B-prime publisher."""
    pa = pytest.importorskip("pyarrow")
    pq = pytest.importorskip("pyarrow.parquet")
    module = _load_module()
    revision = module.APPROVED_PTV2_REVISION
    repository = "nvidia/Nemotron-Post-Training-Dataset-v2"
    sources = []
    splits = (
        "chat",
        "code",
        "math",
        "stem",
        "multilingual_ja",
        "multilingual_it",
        "multilingual_de",
        "multilingual_es",
        "multilingual_fr",
    )
    shard_counts = {
        "chat": 12,
        "math": 2,
        "code": 2,
        "stem": 2,
        "multilingual_de": 38,
        "multilingual_ja": 37,
        "multilingual_es": 33,
        "multilingual_fr": 37,
        "multilingual_it": 38,
    }
    for split in splits:
        count = shard_counts[split]
        files = []
        for file_index in range(count):
            relative = f"data/{split}/{file_index:03d}.parquet"
            local_file = tmp_path / "local" / repository / revision / relative
            local_file.parent.mkdir(parents=True, exist_ok=True)
            pq.write_table(
                pa.Table.from_pylist(
                    [
                        {
                            "messages": [
                                {
                                    "role": "user",
                                    "content": f"question-{split}-{file_index}",
                                },
                                {
                                    "role": "assistant",
                                    "content": f"answer-{split}-{file_index}",
                                },
                            ]
                        }
                    ]
                ),
                local_file,
            )
            files.append(
                module.SourceFile(
                    relative,
                    local_file.stat().st_size,
                    module.sha256_file(local_file),
                )
            )
        sources.append(
            module.SourceIdentity(
                repository,
                "default",
                split,
                revision,
                "CC-BY-4.0",
                True,
                split,
                "target-synth",
                tuple(files),
            )
        )
    plan_payload = {
        "schema_version": 1,
        "name": "ptv2-normalization",
        "sources": [
            {
                "repository_id": source.repository_id,
                "configuration": source.configuration,
                "split": source.split,
                "revision": source.revision,
                "license_expression": source.license_expression,
                "approved_use": source.approved_use,
                "cell": source.cell,
                "lane": source.lane,
                "files": [
                    {"path": file.path, "bytes": file.bytes, "sha256": file.sha256}
                    for file in source.files
                ],
            }
            for source in sources
        ],
    }
    plan_path = tmp_path / "source-plan.json"
    plan_path.write_text(json.dumps(plan_payload), encoding="utf-8")
    sys.path.insert(0, str(MODULE_PATH.parent))
    try:
        import stage_ptv23_sources as staging_module
    finally:
        sys.path.pop(0)
    inventory = staging_module.stage_source_inventory(
        staging_module.load_source_inventory(plan_path),
        durable_root=tmp_path / "durable",
        scratch_root=tmp_path / "scratch",
        local_source_root=tmp_path / "local",
    )

    tokenizer_root = tmp_path / "tokenizer"
    tokenizer_root.mkdir()
    (tokenizer_root / "tokenizer.json").write_text('{"version":"fixture"}', encoding="utf-8")
    (tokenizer_root / "tokenizer_config.json").write_text(
        json.dumps({"chat_template": "{{ messages }}"}), encoding="utf-8"
    )
    tokenizer_receipt = tmp_path / "TOKENIZER_SNAPSHOT.json"
    snapshot = module.write_tokenizer_snapshot_receipt(tokenizer_root, tokenizer_receipt)
    monkeypatch.setattr(
        module, "_load_tokenizer_from_snapshot", lambda _snapshot: CandidateTokenizer()
    )
    candidates = module.build_candidate_inventory(
        inventory,
        tokenizer=CandidateTokenizer(),
        tokenizer_sha256=snapshot.tokenizer_sha256,
        chat_template_sha256=snapshot.chat_template_sha256,
        baseline_exclusion=module.make_exclusion_receipt("baseline", ()),
        held_out_exclusion=module.make_exclusion_receipt("held-out", ()),
    )
    try:
        observed = {(row.domain, row.language) for row in candidates.rows}
        sys.path.insert(0, str(MODULE_PATH.parent))
        try:
            sys.modules.pop("bprime_cd_policy", None)
            sys.modules.pop("select_bprime_cd_prompts", None)
            import bprime_cd_policy as policy_module
            import select_bprime_cd_prompts as selection_module
        finally:
            sys.path.pop(0)
        cells = MappingProxyType(
            {
                name: policy_module.PromptCell(1)
                for name in ("stem", "japanese", "spanish", "french", "italian")
            }
        )
        policy = policy_module.PromptPolicy(
            1,
            20260822,
            1,
            1,
            MappingProxyType({"B-prime": policy_module.ArmPolicy(5, cells, MappingProxyType({}))}),
            (256_000_000, 1_000_000_000),
            4096,
            32768,
            frozenset({"en", "ja", "es", "fr", "it"}),
            frozenset({"de"}),
            "a" * 64,
        )
        bundle = selection_module.select_bprime_prompt_view(
            candidates,
            policy,
            source_inventory=inventory,
            tokenizer_snapshot_receipt=tokenizer_receipt,
            tokenizer_snapshot_receipt_sha256=snapshot.receipt_sha256,
            baseline_receipt_sha256=candidates.baseline_exclusion.receipt_sha256,
            held_out_receipt_sha256=candidates.held_out_exclusion.receipt_sha256,
        )
        try:
            published = selection_module.publish_bprime_prompt_view_bundle(
                bundle, tmp_path / "published-bprime", rows_per_shard=2
            )
        finally:
            bundle.close()
    finally:
        candidates.rows.close()

    assert observed == {
        ("instruction-chat", "en"),
        ("code", "en"),
        ("math", "en"),
        ("stem-science", "en"),
        ("multilingual", "ja"),
        ("multilingual", "it"),
        ("multilingual", "de"),
        ("multilingual", "es"),
        ("multilingual", "fr"),
    }
    assert {cell.arm_domain for cell in candidates.capacity} == {
        "instruction-chat",
        "code",
        "math",
        "stem-science",
        "multilingual",
    }
    selection_manifest = json.loads(published.manifest_path.read_bytes())
    assert selection_manifest["selection_mode"] == "B-prime-only"
    assert set(selection_manifest["arms"]) == {"B-prime"}


@pytest.mark.parametrize(
    ("token_count", "bucket"),
    [
        (4_096, "le4k"),
        (4_097, "4k_16k"),
        (16_384, "4k_16k"),
        (16_385, "16k_32k"),
        (32_768, "16k_32k"),
    ],
)
def test_context_bucket_exact_boundaries(token_count: int, bucket: str) -> None:
    module = _load_module()

    assert module._context_bucket(token_count) == bucket


def test_context_bucket_rejects_above_32k() -> None:
    module = _load_module()

    with pytest.raises(ValueError, match="32K inventory limit"):
        module._context_bucket(32_769)


def _ptv2_lane(module: Any, manifest: Path, **overrides: Any) -> Any:
    fields = {
        "source_id": "nvidia/Nemotron-Post-Training-Dataset-v2",
        "source_revision": "a" * 40,
        "license": "ODC-BY-1.0",
        "pool": "ptv2",
        "category": "math",
        "language": "en",
        "response_source": "target-synth",
        "tool_lane": "none",
        "manifest_path": manifest,
    }
    return module.InventorySource(**(fields | overrides))


def test_an_inventory_row_carries_the_language_its_lane_declares(tmp_path: Path) -> None:
    """Selection filters PTv2 by language, and the parquet has no language column.

    Language reaches a row only from the shard filename, so the lane is the
    finest granularity that knows it. If the row did not carry it, the arm-B
    language policy would have nothing to read.
    """
    module = _load_module()
    manifest = _write_bound_source(
        tmp_path,
        [
            {
                "prompt_id": "p1",
                "messages": [
                    {"role": "user", "content": "question"},
                    {"role": "assistant", "content": "answer"},
                ],
                "_synthesis_assistant_tokens": 1,
            }
        ],
    )
    rows = module.build_inventory_rows(
        [_ptv2_lane(module, manifest, language="ja")],
        tokenizer=FakeTokenizer(),
        tokenizer_sha256="f" * 64,
        training_seq_len=4096,
    )
    assert [row["language"] for row in rows] == ["ja"]


def test_a_lane_that_declares_no_language_is_refused_at_construction(tmp_path: Path) -> None:
    """The failure belongs where the lane is defined, not where it is selected.

    A lane that forgets its language would otherwise reach selection carrying
    nothing to filter on, and the whole point of the policy is that a denied
    language cannot enter arm B by omission.
    """
    module = _load_module()
    manifest = _write_bound_source(tmp_path, [{"prompt_id": "p1", "messages": []}])
    with pytest.raises(ValueError, match="not a recognized code"):
        _ptv2_lane(module, manifest, language="")


def test_a_language_that_is_not_a_known_code_is_refused(tmp_path: Path) -> None:
    """A free-text language would silently miss every allow and deny list."""
    module = _load_module()
    manifest = _write_bound_source(tmp_path, [{"prompt_id": "p1", "messages": []}])
    with pytest.raises(ValueError, match="not a recognized code"):
        _ptv2_lane(module, manifest, language="japanese")
