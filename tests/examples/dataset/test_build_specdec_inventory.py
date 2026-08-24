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
import sys
from pathlib import Path
from types import MappingProxyType

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
    remaining = 201
    for split_index, split in enumerate(splits):
        count = remaining // (len(splits) - split_index)
        remaining -= count
        files = []
        for file_index in range(count):
            relative = f"data/{split}/{file_index:03d}.jsonl"
            local_file = tmp_path / "local" / repository / revision / relative
            local_file.parent.mkdir(parents=True, exist_ok=True)
            local_file.write_text(
                json.dumps(
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
                )
                + "\n"
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
