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
import sqlite3
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from types import SimpleNamespace

import pytest
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import PreTrainedTokenizerFast

REPO_ROOT = Path(__file__).resolve().parents[3]
DATASET_ROOT = REPO_ROOT / "examples/dataset"
MODULE_PATH = DATASET_ROOT / "build_assistant_token_views.py"
sys.path.insert(0, str(DATASET_ROOT))


def _load_module():
    spec = importlib.util.spec_from_file_location("build_assistant_token_views", MODULE_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


CHAT_TEMPLATE = (
    "{% for message in messages %}"
    "{% if message['role'] == 'assistant' %}"
    "{% generation %}{{ message['content'] }}{% endgeneration %}"
    "{% else %}{{ message['content'] }}{% endif %}"
    "{% endfor %}"
)
CHAT_TEMPLATE_SHA256 = hashlib.sha256(CHAT_TEMPLATE.encode()).hexdigest()
TOKENIZER_SHA256 = "2" * 64
PAIRED_SHA256 = "9" * 64


@dataclass(frozen=True)
class _Record:
    prompt_uuid: str
    arm: str
    domain: str
    lane: str
    language: str
    context_bucket: str
    canonical_record_json: str
    request_sha256: str | None
    response_sha256: str | None
    selection_sha256: str
    paired_cd_sha256: str
    generation_identity_sha256: str
    attempt_or_replay_validation_sha256: str
    record_sha256: str


@dataclass(frozen=True)
class _GenerationIdentity:
    target_revision: str
    tokenizer_sha256: str
    chat_template_sha256: str
    runtime_sha256: str
    container_sha256: str
    source_selection_sha256: str
    thinking_mode: str
    temperature: float
    max_tokens: int
    max_total_length: int

    @property
    def sha256(self) -> str:
        return hashlib.sha256(_canonical(asdict(self))).hexdigest()


class _Tokenizer:
    chat_template = CHAT_TEMPLATE

    def apply_chat_template(self, messages, **kwargs):
        assert kwargs == {
            "tools": None,
            "tokenize": True,
            "add_generation_prompt": False,
            "return_dict": True,
            "return_assistant_tokens_mask": True,
        }
        encoded = json.loads(messages[-1]["content"])
        return {"input_ids": encoded["ids"], "assistant_masks": encoded["mask"]}


def _record(
    prompt_uuid: str,
    *,
    arm: str = "C",
    domain: str = "math",
    lane: str = "target-synth",
    ids: list[int],
    mask: list[int],
    paired_sha256: str = PAIRED_SHA256,
    selection_sha256: str = "5" * 64,
    generation_identity_sha256: str = "6" * 64,
    context_bucket: str = "le4k",
) -> _Record:
    canonical = json.dumps(
        {"messages": [{"role": "assistant", "content": json.dumps({"ids": ids, "mask": mask})}]},
        sort_keys=True,
        separators=(",", ":"),
    )
    values = {
        "prompt_uuid": prompt_uuid,
        "arm": arm,
        "domain": domain,
        "lane": lane,
        "language": "en",
        "context_bucket": context_bucket,
        "canonical_record_json": canonical,
        "request_sha256": "3" * 64,
        "response_sha256": "4" * 64,
        "selection_sha256": selection_sha256,
        "paired_cd_sha256": paired_sha256,
        "generation_identity_sha256": generation_identity_sha256,
        "attempt_or_replay_validation_sha256": "7" * 64,
    }
    return _Record(
        **values,
        record_sha256=hashlib.sha256(
            json.dumps(values, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest(),
    )


def _corpus(arm: str, records: list[_Record], *, digest: str) -> SimpleNamespace:
    return SimpleNamespace(
        arm=arm,
        records=records,
        corpus_sha256=digest,
        source_selection_sha256="5" * 64,
        generation_identity_sha256="6" * 64,
        generation_identity=SimpleNamespace(
            tokenizer_sha256=TOKENIZER_SHA256,
            chat_template_sha256=CHAT_TEMPLATE_SHA256,
        ),
    )


def _rows(view) -> list[dict]:
    return [json.loads(line) for line in Path(view.records_path).read_text().splitlines()]


def _canonical(value: object) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()


def test_ptv2_scientific_milestone_cannot_extend_past_the_2m_occurrence_pass() -> None:
    """A 256M receipt is invalid until both authenticated one-pass totals reach it."""
    module = _load_module()
    prefix = module.PTV2OnePassCorpus(
        strategy="A-repair",
        occurrence_count=2_000_000,
        trainer_epochs=1,
        assistant_tokens=200_000_000,
        tokenizer_sha256="1" * 64,
        chat_template_sha256="2" * 64,
        assistant_loss_target_sha256="3" * 64,
        training_config_sha256="4" * 64,
        source_response_root_sha256="5" * 64,
        ordered_occurrences_sha256="6" * 64,
    )
    balanced = module.PTV2OnePassCorpus(
        strategy="B-balanced",
        occurrence_count=2_000_000,
        trainer_epochs=1,
        assistant_tokens=240_000_000,
        tokenizer_sha256="1" * 64,
        chat_template_sha256="2" * 64,
        assistant_loss_target_sha256="3" * 64,
        training_config_sha256="4" * 64,
        source_response_root_sha256="7" * 64,
        ordered_occurrences_sha256="8" * 64,
    )

    with pytest.raises(module.ExposureViewError, match="one-pass does not reach 256M"):
        module.build_ptv2_study_exposures(prefix, balanced, scientific_tokens=256_000_000)


def test_ptv2_derivation_rejects_a_response_not_in_the_tokenized_conversation(
    tmp_path: Path,
) -> None:
    """Task 7 must tokenize the authenticated conversation, not an unrelated response string."""
    module = _load_module()
    index = tmp_path / "selection.sqlite3"
    conversation = _canonical(
        {
            "messages": [
                {"content": "prompt", "role": "user"},
                {"content": json.dumps({"ids": [1, 2], "mask": [0, 1]}), "role": "assistant"},
            ]
        }
    ).decode()
    response = _canonical({"content": "different", "role": "assistant"}).decode()
    occurrence = (0, "a" * 64, "b" * 64, 0, "math", 0)
    conversation_sha256 = hashlib.sha256(conversation.encode()).hexdigest()
    response_sha256 = hashlib.sha256(response.encode()).hexdigest()
    selected = (*occurrence, conversation_sha256, response_sha256)
    connection = sqlite3.connect(index)
    connection.executescript(
        "CREATE TABLE source_rows(source_identity_sha256 TEXT,source_row INTEGER,"
        "canonical_conversation TEXT,assistant_response TEXT);"
        "CREATE TABLE occurrences(strategy TEXT,ordinal INTEGER,prompt_uuid TEXT,"
        "source_identity_sha256 TEXT,source_row INTEGER,cell TEXT,reuse_index INTEGER,"
        "conversation_sha256 TEXT,assistant_response_sha256 TEXT);"
    )
    connection.execute(
        "INSERT INTO source_rows VALUES(?,?,?,?)", ("b" * 64, 0, conversation, response)
    )
    connection.execute(
        "INSERT INTO occurrences VALUES(?,?,?,?,?,?,?,?,?)", ("B-balanced", *selected)
    )
    connection.commit()
    connection.close()
    ordered = hashlib.sha256(_canonical(list(selected)) + b"\n").hexdigest()
    response_root = hashlib.sha256(
        _canonical(["b" * 64, 0, conversation_sha256, response_sha256]) + b"\n"
    ).hexdigest()
    view = SimpleNamespace(
        strategy="B-balanced",
        index_path=index,
        occurrence_count=1,
        ordered_occurrences_sha256=ordered,
        source_response_root_sha256=response_root,
        unique_prompt_count=1,
        natural_duplicate_count=0,
        constructed_repeat_count=0,
        trainer_epochs=1,
    )

    with pytest.raises(module.ExposureViewError, match="final assistant"):
        module.derive_ptv2_one_pass_corpus(
            view,
            tokenizer_sha256="1" * 64,
            chat_template_sha256=CHAT_TEMPLATE_SHA256,
            assistant_loss_target_sha256="2" * 64,
            training_config_sha256="3" * 64,
            tokenizer=_Tokenizer(),
            output_root=tmp_path / "tokenized",
        )


def _digest_lines(seed: object, rows: list[list[object]]) -> str:
    digest = hashlib.sha256(_canonical(seed))
    for row in rows:
        digest.update(_canonical(row) + b"\n")
    return digest.hexdigest()


def _write_selection_artifact(root: Path) -> tuple[Path, str, str, str]:
    root.mkdir(parents=True)
    index_path = root / "selection-index.sqlite3"
    connection = sqlite3.connect(index_path)
    connection.execute(
        "CREATE TABLE rows(arm TEXT,status TEXT,selection_index INTEGER,prompt_uuid TEXT,domain TEXT)"
    )
    selected_rows = [("C", "primary", 0, "shared", "math"), ("D", "primary", 0, "shared", "math")]
    connection.executemany("INSERT INTO rows VALUES(?,?,?,?,?)", selected_rows)
    connection.commit()
    connection.close()
    shard = root / "rows.jsonl"
    shard.write_bytes(b"{}\n{}\n")
    count_proofs = {"C": "a" * 64, "D": "b" * 64}
    arms = {
        arm: {
            "primary_count": 1,
            "reserve_count": 0,
            "cell_counts": {"math": 1},
            "lane_counts": {"target-synth": 1},
            "bucket_floors": {"math": {"le4k": 1}},
            "non_agentic_bucket_floors": {"math": {"le4k": 1}},
            "lane_bucket_floors": {},
            "count_proof_sha256": count_proofs[arm],
        }
        for arm in ("C", "D")
    }
    paired_seed = {
        "bucket_floors": arms["C"]["non_agentic_bucket_floors"],
        "C_count_proof_sha256": count_proofs["C"],
        "D_count_proof_sha256": count_proofs["D"],
    }
    paired_sha256 = _digest_lines(paired_seed, [["primary", "shared"]])
    identity = {
        "policy_sha256": "c" * 64,
        "seed": 42,
        "source_inventory_sha256": "d" * 64,
        "baseline_receipt_sha256": "e" * 64,
        "held_out_receipt_sha256": "f" * 64,
        "ptv2_revision": "1" * 40,
        "ptv2_allowlist_sha256": "1" * 64,
    }
    metadata = {
        "schema_version": 2,
        **identity,
        "paired_cd_sha256": paired_sha256,
        "arms": arms,
    }
    selection_sha256 = _digest_lines(
        metadata,
        [[arm, status, index, prompt] for arm, status, index, prompt, _ in selected_rows],
    )
    manifest = {
        "schema_version": 2,
        "selection_sha256": selection_sha256,
        "paired_cd_sha256": paired_sha256,
        "identity": identity,
        "arms": arms,
        "row_count": 2,
        "rows_per_shard": 2,
        "shards": [
            {
                "path": shard.name,
                "row_count": 2,
                "byte_count": shard.stat().st_size,
                "sha256": hashlib.sha256(shard.read_bytes()).hexdigest(),
            }
        ],
        "index": {
            "path": index_path.name,
            "sha256": hashlib.sha256(index_path.read_bytes()).hexdigest(),
        },
    }
    manifest["root_sha256"] = hashlib.sha256(_canonical(manifest)).hexdigest()
    manifest_path = root / "SELECTION_MANIFEST.json"
    manifest_path.write_bytes(_canonical(manifest) + b"\n")
    return (
        manifest_path,
        hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        selection_sha256,
        paired_sha256,
    )


def _write_response_artifact(
    root: Path,
    corpus: SimpleNamespace,
    record: _Record,
    *,
    artifact_paired_sha256: str | None = None,
) -> str:
    root.mkdir(parents=True)
    payload = _canonical(asdict(record)).decode()
    response_path = root / "responses.jsonl"
    response_path.write_bytes(payload.encode() + b"\n")
    index_path = root / "response-index.sqlite3"
    connection = sqlite3.connect(index_path)
    connection.executescript(
        "CREATE TABLE promoted(ordinal INTEGER,prompt_uuid TEXT,cell TEXT,assistant_tokens INTEGER,payload TEXT);"
        "CREATE TABLE attempts(ordinal INTEGER,payload TEXT);"
        "CREATE TABLE unused_reserve(ordinal INTEGER);"
    )
    connection.execute(
        "INSERT INTO promoted VALUES(0,?,?,1,?)", (record.prompt_uuid, "math", payload)
    )
    connection.commit()
    connection.close()
    stream_sha256 = hashlib.sha256(response_path.read_bytes()).hexdigest()
    history_sha256 = hashlib.sha256().hexdigest()
    histograms = {"math": {1: 1}}
    corpus_record = {
        "arm": corpus.arm,
        "generation_identity_sha256": corpus.generation_identity_sha256,
        "selection_sha256": record.selection_sha256,
        "paired_cd_sha256": artifact_paired_sha256 or record.paired_cd_sha256,
        "cell_counts": {"math": 1},
        "promoted_record_stream_sha256": stream_sha256,
        "attempt_history_stream_sha256": history_sha256,
        "unused_reserve_count": 0,
        "response_token_histograms": histograms,
    }
    corpus.corpus_sha256 = hashlib.sha256(_canonical(corpus_record)).hexdigest()
    receipt = {
        "schema_version": 1,
        "corpus_sha256": corpus.corpus_sha256,
        "generation_identity_sha256": corpus.generation_identity_sha256,
        "source_selection_sha256": corpus.source_selection_sha256,
        "selection_sha256": record.selection_sha256,
        "paired_cd_sha256": artifact_paired_sha256 or record.paired_cd_sha256,
        "promoted_count": 1,
        "cell_counts": {"math": 1},
        "unused_reserve_count": 0,
        "response_token_histograms": histograms,
        "promoted_record_stream_sha256": stream_sha256,
        "attempt_history_stream_sha256": history_sha256,
        "files": [
            {
                "path": path.name,
                "bytes": path.stat().st_size,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
            for path in (response_path, index_path)
        ],
    }
    (root / "PROMOTION.json").write_bytes(_canonical(receipt) + b"\n")
    promotion_path = root / "PROMOTION.json"
    files = [
        {
            "path": f"corpus/{path.name}",
            "bytes": path.stat().st_size,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
        for path in (response_path, index_path, promotion_path)
    ]
    identity = corpus.generation_identity
    completion = {
        "schema_version": 1,
        "generation_identity_sha256": corpus.generation_identity_sha256,
        "target_revision": identity.target_revision,
        "tokenizer_sha256": identity.tokenizer_sha256,
        "chat_template_sha256": identity.chat_template_sha256,
        "runtime_sha256": identity.runtime_sha256,
        "container_sha256": identity.container_sha256,
        "source_selection_sha256": identity.source_selection_sha256,
        "thinking_mode": identity.thinking_mode,
        "promotion_status": "passed",
        "corpus_sha256": corpus.corpus_sha256,
        "promotion_receipt_sha256": hashlib.sha256(promotion_path.read_bytes()).hexdigest(),
        "file_count": len(files),
        "files": files,
    }
    completion_path = root.parent / "completion.json"
    completion_path.write_bytes(_canonical(completion) + b"\n")
    return hashlib.sha256(completion_path.read_bytes()).hexdigest()


def test_exact_boundary_masks_only_late_assistant_positions(tmp_path: Path) -> None:
    module = _load_module()
    corpus = _corpus(
        "C",
        [
            _record("p0", ids=[90, 10, 91, 11, 12, 92], mask=[0, 1, 0, 1, 1, 0]),
            _record("p1", ids=[93, 13, 94, 14], mask=[0, 1, 0, 1]),
        ],
        digest="a" * 64,
    )

    result = module.build_exposure_views(
        corpus,
        _Tokenizer(),
        output_root=tmp_path / "views",
        boundaries={"exact": 3},
        seed=1,
        tokenizer_sha256=TOKENIZER_SHA256,
        chat_template_sha256=CHAT_TEMPLATE_SHA256,
        production=False,
    )
    rows = _rows(result["exact"])

    assert sum(sum(row["loss_mask"]) for row in rows) == 3
    assert rows[-1]["input_ids"] == [90, 10, 91, 11, 12, 92]
    assert rows[-1]["loss_mask"] == [0, 1, 0, 0, 0, 0]
    assert rows[-1]["cumulative_assistant_tokens"] == 3


def test_retokenization_rejects_a_zero_assistant_token_record(tmp_path: Path) -> None:
    module = _load_module()
    corpus = _corpus(
        "B-prime",
        [_record("empty", arm="B-prime", ids=[1, 2, 0], mask=[0, 0, 0], paired_sha256="0" * 64)],
        digest="b" * 64,
    )

    with pytest.raises(module.ExposureViewError, match="no assistant-owned tokens"):
        module.build_exposure_views(
            corpus,
            _Tokenizer(),
            output_root=tmp_path / "views",
            boundaries={"one-pass": None},
            seed=1,
            tokenizer_sha256=TOKENIZER_SHA256,
            chat_template_sha256=CHAT_TEMPLATE_SHA256,
            production=False,
        )


def test_trainer_window_masks_truncated_assistant_tokens(tmp_path: Path) -> None:
    module = _load_module()
    corpus = _corpus(
        "C",
        [_record("long", ids=[10, 11, 12, 13, 14], mask=[0, 1, 0, 1, 1])],
        digest="b" * 64,
    )

    view = module.build_exposure_views(
        corpus,
        _Tokenizer(),
        output_root=tmp_path / "views",
        boundaries={"one-pass": None},
        seed=1,
        tokenizer_sha256=TOKENIZER_SHA256,
        chat_template_sha256=CHAT_TEMPLATE_SHA256,
        training_sequence_length=3,
    )["one-pass"]

    assert view.assistant_tokens == 1
    assert _rows(view)[0]["input_ids"] == [10, 11, 12]
    assert _rows(view)[0]["loss_mask"] == [0, 1, 0]


def test_record_selection_digest_is_distinct_from_manifest_file_digest(tmp_path: Path) -> None:
    module = _load_module()
    corpus = _corpus(
        "C",
        [_record("p0", ids=[1], mask=[1], selection_sha256="8" * 64)],
        digest="3" * 64,
    )

    view = module.build_exposure_views(
        corpus,
        _Tokenizer(),
        output_root=tmp_path / "views",
        boundaries={"one-pass": None},
        seed=1,
        tokenizer_sha256=TOKENIZER_SHA256,
        chat_template_sha256=CHAT_TEMPLATE_SHA256,
    )["one-pass"]

    assert view.selection_sha256 == "8" * 64


def test_agentless_swe_uses_target_synthesis_tokenization(tmp_path: Path) -> None:
    module = _load_module()
    corpus = _corpus(
        "C",
        [
            _record(
                "agentless",
                domain="swe-agentic-tool",
                lane="agentless-swe",
                ids=[1, 2],
                mask=[0, 1],
            )
        ],
        digest="7" * 64,
    )

    view = module.build_exposure_views(
        corpus,
        _Tokenizer(),
        output_root=tmp_path / "views",
        boundaries={"one-pass": None},
        seed=1,
        tokenizer_sha256=TOKENIZER_SHA256,
        chat_template_sha256=CHAT_TEMPLATE_SHA256,
    )["one-pass"]

    assert view.assistant_tokens == 1


def test_epoch_order_is_deterministic_consumable_and_counts_only_mask_ones(
    tmp_path: Path,
) -> None:
    module = _load_module()
    records = [
        _record(
            f"p{index}",
            domain="swe" if index % 2 else "math",
            lane="target-synth",
            ids=[0, 100 + index, 200 + index, 300 + index],
            mask=[0, 1, 0, 0],
        )
        for index in range(3)
    ]
    corpus = _corpus("C", records, digest="c" * 64)

    first = module.build_exposure_views(
        corpus,
        _Tokenizer(),
        output_root=tmp_path / "first",
        boundaries={"repeated": 7},
        seed=29,
        tokenizer_sha256=TOKENIZER_SHA256,
        chat_template_sha256=CHAT_TEMPLATE_SHA256,
    )["repeated"]
    second = module.build_exposure_views(
        corpus,
        _Tokenizer(),
        output_root=tmp_path / "second",
        boundaries={"repeated": 7},
        seed=29,
        tokenizer_sha256=TOKENIZER_SHA256,
        chat_template_sha256=CHAT_TEMPLATE_SHA256,
    )["repeated"]

    first_rows = _rows(first)
    assert [row["prompt_uuid"] for row in first_rows] == [
        row["prompt_uuid"] for row in _rows(second)
    ]
    assert [row["epoch"] for row in first_rows] == [0, 0, 0, 1, 1, 1, 2]
    assert first.unique_prompt_count == 3
    assert first.repeated_prompt_count == 4
    assert first.assistant_tokens == 7
    assert all(sum(row["loss_mask"]) == 1 for row in first_rows)


def test_resume_rejects_an_identity_mismatch_and_reuses_authenticated_output(
    tmp_path: Path,
) -> None:
    module = _load_module()
    corpus = _corpus("C", [_record("p0", ids=[1, 2], mask=[0, 1])], digest="d" * 64)
    output = tmp_path / "views"
    first = module.build_exposure_views(
        corpus,
        _Tokenizer(),
        output_root=output,
        boundaries={"one-pass": None},
        seed=3,
        tokenizer_sha256=TOKENIZER_SHA256,
        chat_template_sha256=CHAT_TEMPLATE_SHA256,
        production=False,
    )
    resumed = module.build_exposure_views(
        corpus,
        _Tokenizer(),
        output_root=output,
        boundaries={"one-pass": None},
        seed=3,
        tokenizer_sha256=TOKENIZER_SHA256,
        chat_template_sha256=CHAT_TEMPLATE_SHA256,
    )

    assert resumed == first
    with pytest.raises(module.ExposureViewError, match="resume fingerprint mismatch"):
        module.build_exposure_views(
            corpus,
            _Tokenizer(),
            output_root=output,
            boundaries={"one-pass": None},
            seed=4,
            tokenizer_sha256=TOKENIZER_SHA256,
            chat_template_sha256=CHAT_TEMPLATE_SHA256,
            production=False,
        )


def test_resume_rederives_counts_and_digests_from_staged_rows(tmp_path: Path) -> None:
    module = _load_module()
    corpus = _corpus("C", [_record("p0", ids=[1, 2], mask=[0, 1])], digest="4" * 64)
    output = tmp_path / "views"
    view = module.build_exposure_views(
        corpus,
        _Tokenizer(),
        output_root=output,
        boundaries={"one-pass": None},
        seed=3,
        tokenizer_sha256=TOKENIZER_SHA256,
        chat_template_sha256=CHAT_TEMPLATE_SHA256,
    )["one-pass"]
    row = _rows(view)[0]
    row["cumulative_assistant_tokens"] = 99
    forged_records = json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n"
    Path(view.records_path).write_text(forged_records)
    receipt_path = Path(view.receipt_path)
    receipt = json.loads(receipt_path.read_text())
    receipt["records_bytes"] = len(forged_records.encode())
    receipt["records_sha256"] = hashlib.sha256(forged_records.encode()).hexdigest()
    receipt_without_digest = {
        key: value for key, value in receipt.items() if key != "receipt_sha256"
    }
    receipt["receipt_sha256"] = hashlib.sha256(
        json.dumps(
            receipt_without_digest,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()
    receipt_path.write_text(json.dumps(receipt, sort_keys=True, separators=(",", ":")) + "\n")

    with pytest.raises(module.ExposureViewError, match="reconciliation"):
        module.build_exposure_views(
            corpus,
            _Tokenizer(),
            output_root=output,
            boundaries={"one-pass": None},
            seed=3,
            tokenizer_sha256=TOKENIZER_SHA256,
            chat_template_sha256=CHAT_TEMPLATE_SHA256,
        )


def test_paired_views_use_exact_common_boundary_and_bind_pair_proof(tmp_path: Path) -> None:
    module = _load_module()
    corpus_c = _corpus(
        "C",
        [_record("c0", ids=[1, 2, 3, 4], mask=[0, 1, 1, 1])],
        digest="e" * 64,
    )
    corpus_d = _corpus(
        "D",
        [
            _record("d0", arm="D", ids=[4, 5, 6], mask=[0, 1, 1]),
            _record("d1", arm="D", ids=[7, 8, 9], mask=[0, 1, 1]),
        ],
        digest="f" * 64,
    )

    paired = module.build_paired_exposure_views(
        corpus_c,
        corpus_d,
        _Tokenizer(),
        output_root=tmp_path / "paired",
        fixed_boundaries=(2, 5),
        seed=11,
        tokenizer_sha256=TOKENIZER_SHA256,
        chat_template_sha256=CHAT_TEMPLATE_SHA256,
        production=False,
    )

    assert paired.common_boundary == 3
    assert paired.C_full_prompt_assistant_tokens == 3
    assert paired.D_full_prompt_assistant_tokens == 4
    assert paired.paired_selection_bucket_sha256 == PAIRED_SHA256
    assert paired.C["2"].assistant_tokens == paired.D["2"].assistant_tokens == 2
    assert paired.C["5"].assistant_tokens == paired.D["5"].assistant_tokens == 5
    assert paired.C["common"].assistant_tokens == paired.D["common"].assistant_tokens == 3
    assert paired.C["one-pass"].full_prompt_assistant_tokens == 3
    assert paired.D["one-pass"].full_prompt_assistant_tokens == 4


def test_paired_views_reject_mismatched_pair_bucket_proof(tmp_path: Path) -> None:
    module = _load_module()
    corpus_c = _corpus("C", [_record("c", ids=[1], mask=[1])], digest="1" * 64)
    corpus_d = _corpus(
        "D",
        [_record("d", arm="D", ids=[1], mask=[1], paired_sha256="8" * 64)],
        digest="2" * 64,
    )

    with pytest.raises(module.ExposureViewError, match="paired selection/bucket proof"):
        module.build_paired_exposure_views(
            corpus_c,
            corpus_d,
            _Tokenizer(),
            output_root=tmp_path / "paired",
            fixed_boundaries=(2,),
            seed=11,
            tokenizer_sha256=TOKENIZER_SHA256,
            chat_template_sha256=CHAT_TEMPLATE_SHA256,
            production=False,
        )


def test_production_paired_views_reject_runtime_screen_as_a_comparison(
    tmp_path: Path,
) -> None:
    module = _load_module()
    corpus_c = _corpus("C", [_record("c", ids=[1], mask=[1])], digest="1" * 64)
    corpus_d = _corpus("D", [_record("d", arm="D", ids=[1], mask=[1])], digest="2" * 64)

    with pytest.raises(module.ExposureViewError, match="production boundaries"):
        module.build_paired_exposure_views(
            corpus_c,
            corpus_d,
            _Tokenizer(),
            output_root=tmp_path / "paired",
            fixed_boundaries=(64_000_000,),
            seed=11,
            tokenizer_sha256=TOKENIZER_SHA256,
            chat_template_sha256=CHAT_TEMPLATE_SHA256,
            production=True,
        )


@pytest.mark.parametrize(
    ("boundaries", "sequence_length", "error"),
    [
        (
            {"256000000": 1, "1000000000": 2, "one-pass": None},
            4_096,
            "exact boundary values",
        ),
        (
            {"256000000": 256_000_000, "1000000000": 1_000_000_000, "one-pass": None},
            3,
            "sequence length",
        ),
        (
            {"256000000": 256_000_000, "1000000000": 1_000_000_000, "one-pass": 1},
            4_096,
            "one-pass",
        ),
        (
            {
                "256000000": 256_000_000,
                "1000000000": 1_000_000_000,
                "one-pass": None,
                "common": 2,
            },
            4_096,
            "paired builder",
        ),
    ],
)
def test_production_labels_require_exact_values_and_trainer_length(
    tmp_path: Path,
    boundaries: dict[str, int | None],
    sequence_length: int,
    error: str,
) -> None:
    module = _load_module()
    corpus = _corpus("C", [_record("p0", ids=[1], mask=[1])], digest="5" * 64)

    with pytest.raises(module.ExposureViewError, match=error):
        module.build_exposure_views(
            corpus,
            _Tokenizer(),
            output_root=tmp_path / "views",
            boundaries=boundaries,
            seed=1,
            tokenizer_sha256=TOKENIZER_SHA256,
            chat_template_sha256=CHAT_TEMPLATE_SHA256,
            training_sequence_length=sequence_length,
            production=True,
        )


def test_production_view_authenticates_task5_task6_and_tokenizer_artifacts(
    tmp_path: Path,
) -> None:
    module = _load_module()
    manifest_path, manifest_sha256, selection_sha256, paired_sha256 = _write_selection_artifact(
        tmp_path / "selection"
    )
    tokenizer_root = tmp_path / "tokenizer"
    backend = Tokenizer(WordLevel({"[UNK]": 0}, unk_token="[UNK]"))
    backend.pre_tokenizer = Whitespace()
    saved_tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=backend,
        unk_token="[UNK]",
        chat_template=CHAT_TEMPLATE,
    )
    saved_tokenizer.save_pretrained(tokenizer_root)
    template_path = tokenizer_root / "chat_template.jinja"
    template_file_sha256 = hashlib.sha256(template_path.read_bytes()).hexdigest()
    snapshot = hashlib.sha256()
    tokenizer_names = {
        "added_tokens.json",
        "chat_template.jinja",
        "merges.txt",
        "special_tokens_map.json",
        "tokenizer.json",
        "tokenizer.model",
        "tokenizer_config.json",
        "vocab.json",
        "vocab.txt",
    }
    for path in sorted(path for path in tokenizer_root.iterdir() if path.name in tokenizer_names):
        snapshot.update(path.name.encode())
        snapshot.update(b"\0")
        snapshot.update(bytes.fromhex(hashlib.sha256(path.read_bytes()).hexdigest()))
    tokenizer_digest = snapshot.hexdigest()
    identity = _GenerationIdentity(
        target_revision="1" * 40,
        tokenizer_sha256=tokenizer_digest,
        chat_template_sha256=template_file_sha256,
        runtime_sha256="2" * 64,
        container_sha256="3" * 64,
        source_selection_sha256=manifest_sha256,
        thinking_mode="off",
        temperature=0.0,
        max_tokens=128,
        max_total_length=4_096,
    )
    record = _record(
        "shared",
        arm="B",
        ids=[10, 11],
        mask=[0, 1],
        selection_sha256=selection_sha256,
        paired_sha256=paired_sha256,
        generation_identity_sha256=identity.sha256,
    )
    corpus = SimpleNamespace(
        arm="B",
        records=[record],
        corpus_sha256="0" * 64,
        source_selection_sha256=manifest_sha256,
        generation_identity_sha256=identity.sha256,
        generation_identity=identity,
    )
    response_root = tmp_path / "response/corpus"
    completion_sha256 = _write_response_artifact(
        response_root,
        corpus,
        record,
        artifact_paired_sha256=paired_sha256,
    )

    view = module._build_exposure_views(
        corpus,
        _Tokenizer(),
        output_root=tmp_path / "views",
        boundaries={"one-pass": None},
        seed=42,
        tokenizer_sha256=tokenizer_digest,
        chat_template_sha256=template_file_sha256,
        production=True,
        response_artifact_root=response_root,
        expected_task6_completion_sha256=completion_sha256,
        selection_manifest_path=manifest_path,
        tokenizer_root=tokenizer_root,
        _allow_partial_production_suite=True,
    )["one-pass"]

    assert view.purpose == "production-comparison"
    assert view.selection_sha256 == selection_sha256
    assert view.paired_selection_bucket_sha256 == "0" * 64
    assert view.task5_paired_selection_bucket_sha256 == paired_sha256
    assert view.task6_completion_sha256 == completion_sha256
    assert view.source_policy_sha256 == "c" * 64
    assert view.assistant_tokens == sum(_rows(view)[0]["loss_mask"])
    assert view.assistant_tokens != 1


def test_production_recomputes_complete_generation_identity() -> None:
    module = _load_module()
    identity = _GenerationIdentity(
        target_revision="1" * 40,
        tokenizer_sha256="2" * 64,
        chat_template_sha256="3" * 64,
        runtime_sha256="4" * 64,
        container_sha256="5" * 64,
        source_selection_sha256="6" * 64,
        thinking_mode="off",
        temperature=0.25,
        max_tokens=1_024,
        max_total_length=4_096,
    )
    corpus = SimpleNamespace(
        generation_identity=identity,
        generation_identity_sha256="0" * 64,
        source_selection_sha256=identity.source_selection_sha256,
    )

    with pytest.raises(module.ExposureViewError, match="generation identity digest"):
        module._authenticate_generation_identity(corpus)


def test_task6_completion_requires_an_external_expected_digest(tmp_path: Path) -> None:
    module = _load_module()
    published_root = tmp_path / "published"
    corpus_root = published_root / "corpus"
    corpus_root.mkdir(parents=True)
    (published_root / "completion.json").write_text("{}\n")

    with pytest.raises(module.ExposureViewError, match="completion identity mismatch"):
        module._authenticate_task6_completion(corpus_root, expected_sha256="0" * 64)


@pytest.mark.parametrize(
    ("d_prompt_uuid", "d_context_bucket"),
    [("reserve-replacement", "le4k"), ("shared", "gt4k")],
)
def test_final_paired_rows_reject_reserve_or_bucket_drift_without_materializing(
    tmp_path: Path,
    d_prompt_uuid: str,
    d_context_bucket: str,
) -> None:
    module = _load_module()

    def write_index(path: Path, prompt_uuid: str, context_bucket: str) -> None:
        payload = _canonical(
            {
                "prompt_uuid": prompt_uuid,
                "domain": "math",
                "context_bucket": context_bucket,
            }
        ).decode()
        connection = sqlite3.connect(path)
        connection.execute(
            "CREATE TABLE promoted(ordinal INTEGER,prompt_uuid TEXT,cell TEXT,"
            "assistant_tokens INTEGER,payload TEXT)"
        )
        connection.execute("INSERT INTO promoted VALUES(0,?,'math',1,?)", (prompt_uuid, payload))
        connection.commit()
        connection.close()

    c_index = tmp_path / "c.sqlite3"
    d_index = tmp_path / "d.sqlite3"
    write_index(c_index, "shared", "le4k")
    write_index(d_index, d_prompt_uuid, d_context_bucket)

    with pytest.raises(module.ExposureViewError, match="final C/D promoted rows"):
        module._authenticate_final_paired_rows(c_index, d_index)


def test_final_paired_rows_return_a_deterministic_stream_digest(tmp_path: Path) -> None:
    module = _load_module()

    def write_index(path: Path) -> None:
        connection = sqlite3.connect(path)
        connection.execute(
            "CREATE TABLE promoted(ordinal INTEGER,prompt_uuid TEXT,cell TEXT,"
            "assistant_tokens INTEGER,payload TEXT)"
        )
        for ordinal, prompt_uuid in enumerate(("second", "first")):
            payload = _canonical(
                {
                    "prompt_uuid": prompt_uuid,
                    "domain": "math",
                    "context_bucket": "le4k",
                }
            ).decode()
            connection.execute(
                "INSERT INTO promoted VALUES(?,?,'math',1,?)",
                (ordinal, prompt_uuid, payload),
            )
        connection.commit()
        connection.close()

    c_index = tmp_path / "c.sqlite3"
    d_index = tmp_path / "d.sqlite3"
    write_index(c_index)
    write_index(d_index)

    expected = hashlib.sha256(
        _canonical({"schema_version": 1, "scope": "final-promoted-non-agentic-cd"})
    )
    for prompt_uuid in ("first", "second"):
        expected.update(_canonical([prompt_uuid, "le4k"]) + b"\n")

    assert module._authenticate_final_paired_rows(c_index, d_index) == expected.hexdigest()
