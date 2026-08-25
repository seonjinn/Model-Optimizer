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
import os
import shutil
import sqlite3
import sys
from collections import Counter
from dataclasses import asdict, dataclass, replace
from itertools import pairwise
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

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
    tokenizer_sha256 = TOKENIZER_SHA256
    assistant_loss_target_sha256 = "2" * 64

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


class _FailingTask8Tokenizer(_Tokenizer):
    def apply_chat_template(self, messages, **kwargs):
        raise ValueError("injected Task8 worker failure")


def _injected_ptv2_reader_failure(database_path: Path, start: int, stop: int):
    raise RuntimeError("injected exposure reader failure")


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


def _write_ptv2_v2_corpus(
    module,
    root: Path,
    *,
    strategy: str,
    bucket_rows: list[tuple[str, str, int]],
    assistant_tokens_per_row: int = 4,
    historical_prefix_rows: int | None = None,
):
    if strategy == "H-historical-cyclic":
        raise AssertionError("H fixtures must use derive_ptv2_historical_corpus")
    bundle = root / f"{strategy.lower()}-tokenized"
    bundle.mkdir(parents=True)
    database = bundle / "records.sqlite3"
    connection = sqlite3.connect(database)
    connection.execute(
        "CREATE TABLE records("
        "ordinal INTEGER PRIMARY KEY,prompt_uuid TEXT NOT NULL,"
        "source_identity_sha256 TEXT NOT NULL,source_row INTEGER NOT NULL,"
        "cell TEXT NOT NULL,language TEXT NOT NULL,reuse_index INTEGER NOT NULL,"
        "input_ids_json TEXT NOT NULL,loss_mask_json TEXT NOT NULL,"
        "assistant_tokens INTEGER NOT NULL)"
    )
    histogram: dict[str, dict[str, int]] = {}
    occurrence_digest = hashlib.sha256()
    multiplicity_digest = hashlib.sha256()
    ordinal = 0
    for cell, language, count in bucket_rows:
        histogram.setdefault(cell, {})[language] = count * assistant_tokens_per_row
        for source_row in range(count):
            prompt_uuid = hashlib.sha256(
                f"{strategy}:{cell}:{language}:{source_row}".encode()
            ).hexdigest()
            source_identity = hashlib.sha256(f"source:{cell}:{language}".encode()).hexdigest()
            identity = [
                ordinal,
                prompt_uuid,
                source_identity,
                source_row,
                cell,
                language,
                0,
            ]
            occurrence_digest.update(_canonical(identity) + b"\n")
            multiplicity_digest.update(_canonical(identity[1:]) + b"\n")
            ids = list(range(assistant_tokens_per_row))
            mask = [1] * assistant_tokens_per_row
            connection.execute(
                "INSERT INTO records VALUES(?,?,?,?,?,?,?,?,?,?)",
                (
                    ordinal,
                    prompt_uuid,
                    source_identity,
                    source_row,
                    cell,
                    language,
                    0,
                    _canonical(ids).decode(),
                    _canonical(mask).decode(),
                    assistant_tokens_per_row,
                ),
            )
            ordinal += 1
    connection.commit()
    connection.close()
    database_sha256 = hashlib.sha256(database.read_bytes()).hexdigest()
    first_segment = ordinal if historical_prefix_rows is None else historical_prefix_rows
    if first_segment < 1 or first_segment > ordinal:
        raise AssertionError("historical prefix must be within the fixture corpus")
    second_segment = ordinal - first_segment
    first_steps = (first_segment + 511) // 512
    second_steps = (second_segment + 511) // 512
    receipt = {
        "schema_version": 2,
        "strategy": strategy,
        "occurrence_count": ordinal,
        "trainer_epochs": 1,
        "assistant_tokens": ordinal * assistant_tokens_per_row,
        "serialized_tokens": ordinal * assistant_tokens_per_row,
        "packed_sequence_lower_bound": 1,
        "unique_prompt_count": ordinal,
        "natural_duplicate_count": 0,
        "constructed_repeat_count": 0,
        "milestone_occurrences": [ordinal],
        "milestone_steps": [1],
        "segment_occurrences": [first_segment, second_segment],
        "segment_steps": [first_steps, second_steps],
        "cumulative_segment_steps": [first_steps, first_steps + second_steps],
        "segment_final_valid_occurrences": [first_segment % 512, second_segment % 512],
        "tokenizer_sha256": "1" * 64,
        "chat_template_sha256": "2" * 64,
        "assistant_loss_target_sha256": "3" * 64,
        "training_config_sha256": "4" * 64,
        "source_response_root_sha256": "5" * 64,
        "ordered_occurrences_sha256": occurrence_digest.hexdigest(),
        "selection_sha256": "6" * 64,
        "base_occurrence_multiplicity_sha256": multiplicity_digest.hexdigest(),
        "bucket_assistant_token_histogram": histogram,
        "database_path": "records.sqlite3",
        "database_sha256": database_sha256,
        "database_bytes": database.stat().st_size,
    }
    receipt_sha256 = hashlib.sha256(_canonical(receipt)).hexdigest()
    receipt_path = bundle / "TOKENIZED.json"
    receipt_path.write_bytes(_canonical(receipt | {"receipt_sha256": receipt_sha256}) + b"\n")
    return module.PTV2OnePassCorpus(
        strategy=strategy,
        occurrence_count=ordinal,
        trainer_epochs=1,
        assistant_tokens=ordinal * assistant_tokens_per_row,
        tokenizer_sha256="1" * 64,
        chat_template_sha256="2" * 64,
        assistant_loss_target_sha256="3" * 64,
        training_config_sha256="4" * 64,
        source_response_root_sha256="5" * 64,
        ordered_occurrences_sha256=occurrence_digest.hexdigest(),
        selection_sha256="6" * 64,
        unique_prompt_count=ordinal,
        serialized_tokens=ordinal * assistant_tokens_per_row,
        packed_sequence_lower_bound=1,
        tokenized_path=str(database),
        tokenized_sha256=database_sha256,
        receipt_path=str(receipt_path),
        receipt_sha256=receipt_sha256,
        milestone_occurrences=(ordinal,),
        milestone_steps=(1,),
        segment_occurrences=(first_segment, second_segment),
        segment_steps=(first_steps, second_steps),
        cumulative_segment_steps=(first_steps, first_steps + second_steps),
        segment_final_valid_occurrences=(first_segment % 512, second_segment % 512),
    )


def _ptv2_exposure_bundle(corpus, target_tokens: int) -> Path:
    return (
        Path(corpus.tokenized_path).parent.parent
        / f"{corpus.strategy.lower()}-exposures"
        / f"{corpus.strategy.lower()}-{target_tokens}-assistant-tokens-v2"
    )


def _resign_ptv2_tokenized_database(module, corpus, column: str, value: object):
    database = Path(corpus.tokenized_path)
    with sqlite3.connect(database) as connection:
        connection.execute(
            f"UPDATE records SET {column}=? WHERE ordinal=0", (_canonical(value).decode(),)
        )
        connection.commit()
    database_sha256 = hashlib.sha256(database.read_bytes()).hexdigest()
    receipt_path = Path(corpus.receipt_path)
    receipt = json.loads(receipt_path.read_bytes())
    receipt.pop("receipt_sha256")
    receipt["database_sha256"] = database_sha256
    receipt["database_bytes"] = database.stat().st_size
    receipt_sha256 = hashlib.sha256(_canonical(receipt)).hexdigest()
    receipt_path.write_bytes(_canonical(receipt | {"receipt_sha256": receipt_sha256}) + b"\n")
    return replace(corpus, tokenized_sha256=database_sha256, receipt_sha256=receipt_sha256)


def _resign_ptv2_exposure_bundle(bundle: Path, field: str, value: object) -> None:
    records_path = bundle / "records.jsonl"
    rows = [json.loads(line) for line in records_path.read_text().splitlines()]
    rows[0][field] = value
    records_path.write_bytes(b"".join(_canonical(row) + b"\n" for row in rows))
    records_sha256 = hashlib.sha256(records_path.read_bytes()).hexdigest()
    scientific_path = bundle / "SCIENTIFIC.json"
    scientific = json.loads(scientific_path.read_bytes())
    scientific.pop("receipt_sha256")
    scientific["records_bytes"] = records_path.stat().st_size
    scientific["records_sha256"] = records_sha256
    scientific_sha256 = hashlib.sha256(_canonical(scientific)).hexdigest()
    scientific_path.write_bytes(
        _canonical(scientific | {"receipt_sha256": scientific_sha256}) + b"\n"
    )
    execution_path = bundle / "EXECUTION.json"
    execution = json.loads(execution_path.read_bytes())
    execution.pop("receipt_sha256")
    execution["records_sha256"] = records_sha256
    execution["scientific_receipt_sha256"] = scientific_sha256
    execution_sha256 = hashlib.sha256(_canonical(execution)).hexdigest()
    execution_path.write_bytes(_canonical(execution | {"receipt_sha256": execution_sha256}) + b"\n")


def _resign_ptv2_scientific_receipt(bundle: Path, field: str, value: object) -> str:
    scientific_path = bundle / "SCIENTIFIC.json"
    scientific = json.loads(scientific_path.read_bytes())
    scientific.pop("receipt_sha256")
    scientific[field] = value
    scientific_sha256 = hashlib.sha256(_canonical(scientific)).hexdigest()
    scientific_path.write_bytes(
        _canonical(scientific | {"receipt_sha256": scientific_sha256}) + b"\n"
    )
    execution_path = bundle / "EXECUTION.json"
    execution = json.loads(execution_path.read_bytes())
    execution.pop("receipt_sha256")
    execution["scientific_receipt_sha256"] = scientific_sha256
    execution_sha256 = hashlib.sha256(_canonical(execution)).hexdigest()
    execution_path.write_bytes(_canonical(execution | {"receipt_sha256": execution_sha256}) + b"\n")
    return scientific_sha256


def test_ptv2_v2_interleaver_removes_ordinal_prefix_cell_language_confound(
    tmp_path: Path,
) -> None:
    module = _load_module()
    buckets = [
        ("chat", "en", 256),
        ("code", "en", 256),
        ("multilingual", "de", 256),
        ("multilingual", "ja", 256),
    ]
    corpus = _write_ptv2_v2_corpus(
        module, tmp_path / "input", strategy="B-balanced", bucket_rows=buckets
    )
    with sqlite3.connect(corpus.tokenized_path) as connection:
        legacy_prefix = Counter(
            connection.execute(
                "SELECT cell || '/' || language FROM records ORDER BY ordinal LIMIT 512"
            )
        )
    assert {value[0] for value in legacy_prefix} == {"chat/en", "code/en"}

    module._materialize_ptv2_exposure(corpus, 2_048, workers=1)

    bundle = _ptv2_exposure_bundle(corpus, 2_048)
    rows = [json.loads(line) for line in (bundle / "records.jsonl").read_text().splitlines()]
    realized = Counter((row["cell"], row["language"]) for row in rows)
    receipt = json.loads((bundle / "SCIENTIFIC.json").read_bytes())
    assert realized == Counter(
        {
            ("chat", "en"): 128,
            ("code", "en"): 128,
            ("multilingual", "de"): 128,
            ("multilingual", "ja"): 128,
        }
    )
    assert receipt["schema_version"] == 2
    assert receipt["target_bucket_assistant_tokens"] == {
        "chat": {"en": 512},
        "code": {"en": 512},
        "multilingual": {"de": 512, "ja": 512},
    }
    assert receipt["realized_bucket_assistant_tokens"] == receipt["target_bucket_assistant_tokens"]
    assert receipt["batch_boundary_count"] == 1
    assert receipt["max_prefix_discrepancy"]["denominator"] == 4_096


def test_ptv2_v2_exact_boundary_preserves_identity_and_trims_only_final_batch(
    tmp_path: Path,
) -> None:
    module = _load_module()
    corpus = _write_ptv2_v2_corpus(
        module,
        tmp_path / "input",
        strategy="A-repair",
        bucket_rows=[("math", "en", 512), ("stem", "en", 512)],
    )

    module._materialize_ptv2_exposure(corpus, 2_560, workers=1)

    bundle = _ptv2_exposure_bundle(corpus, 2_560)
    rows = [json.loads(line) for line in (bundle / "records.jsonl").read_text().splitlines()]
    assert len(rows) % 512 == 0
    assert sum(row["assistant_tokens"] for row in rows) == 2_560
    assert all(row["assistant_tokens"] > 0 for row in rows)
    assert all(row["loss_mask"] == [1, 1, 1, 1] for row in rows[:512])
    assert all(row["loss_mask"] == [1, 0, 0, 0] for row in rows[512:])
    assert Counter(row["base_ordinal"] for row in rows) == Counter(range(1_024))
    assert {row["cycle_index"] for row in rows} == {0}


def test_ptv2_v2_serial_and_parallel_publications_are_scientifically_identical(
    tmp_path: Path,
) -> None:
    module = _load_module()
    buckets = [("math", "en", 256), ("stem", "en", 256)]
    serial = _write_ptv2_v2_corpus(
        module, tmp_path / "serial", strategy="B-balanced", bucket_rows=buckets
    )
    parallel = _write_ptv2_v2_corpus(
        module, tmp_path / "parallel", strategy="B-balanced", bucket_rows=buckets
    )

    serial_sha = module._materialize_ptv2_exposure(serial, 1_024, workers=1)
    parallel_sha = module._materialize_ptv2_exposure(parallel, 1_024, workers=96)

    serial_bundle = _ptv2_exposure_bundle(serial, 1_024)
    parallel_bundle = _ptv2_exposure_bundle(parallel, 1_024)
    assert serial_sha == parallel_sha
    assert (serial_bundle / "records.jsonl").read_bytes() == (
        parallel_bundle / "records.jsonl"
    ).read_bytes()
    assert (serial_bundle / "SCIENTIFIC.json").read_bytes() == (
        parallel_bundle / "SCIENTIFIC.json"
    ).read_bytes()
    assert json.loads((parallel_bundle / "EXECUTION.json").read_bytes())["requested_workers"] == 96


def test_ptv2_historical_producer_binds_exact_a_prefix_and_cycles_it(tmp_path: Path) -> None:
    module = _load_module()
    prefix = _write_ptv2_v2_corpus(
        module,
        tmp_path / "study",
        strategy="A-repair",
        bucket_rows=[("chat", "en", 512), ("math", "en", 512)],
        assistant_tokens_per_row=2,
        historical_prefix_rows=512,
    )
    corpus = module.derive_ptv2_historical_corpus(prefix, tmp_path / "study")
    tokenized = json.loads(Path(corpus.receipt_path).read_bytes())

    with (
        sqlite3.connect(prefix.tokenized_path) as parent,
        sqlite3.connect(corpus.tokenized_path) as historical,
    ):
        columns = (
            "ordinal,prompt_uuid,source_identity_sha256,source_row,cell,language,reuse_index,"
            "input_ids_json,loss_mask_json,assistant_tokens"
        )
        parent_rows = parent.execute(
            f"SELECT {columns} FROM records WHERE ordinal<512 ORDER BY ordinal"
        ).fetchall()
        historical_rows = historical.execute(
            f"SELECT {columns} FROM records ORDER BY ordinal"
        ).fetchall()
    assert historical_rows == parent_rows
    assert corpus.occurrence_count == 512
    assert tokenized["historical_parent_receipt_sha256"] == prefix.receipt_sha256
    assert tokenized["historical_parent_tokenized_sha256"] == prefix.tokenized_sha256
    assert tokenized["historical_prefix_occurrence_count"] == 512

    module._materialize_ptv2_exposure(corpus, 2_048, workers=1)

    bundle = _ptv2_exposure_bundle(corpus, 2_048)
    rows = [json.loads(line) for line in (bundle / "records.jsonl").read_text().splitlines()]
    receipt = json.loads((bundle / "SCIENTIFIC.json").read_bytes())
    assert Counter(row["base_ordinal"] for row in rows) == Counter(dict.fromkeys(range(512), 2))
    assert {row["cycle_index"] for row in rows} == {0, 1}
    assert (
        receipt["base_occurrence_multiplicity_sha256"]
        == tokenized["base_occurrence_multiplicity_sha256"]
    )


def test_ptv2_historical_arm_cannot_be_rebound_to_an_unrelated_a_baseline(
    tmp_path: Path,
) -> None:
    module = _load_module()
    parent = _write_ptv2_v2_corpus(
        module,
        tmp_path / "parent",
        strategy="A-repair",
        bucket_rows=[("chat", "en", 512), ("math", "en", 512)],
        historical_prefix_rows=512,
    )
    unrelated = _write_ptv2_v2_corpus(
        module,
        tmp_path / "unrelated",
        strategy="A-repair",
        bucket_rows=[("code", "en", 512), ("stem", "en", 512)],
        historical_prefix_rows=512,
    )
    historical = module.derive_ptv2_historical_corpus(parent, tmp_path / "historical")

    with pytest.raises(module.ExposureViewError, match="historical parent"):
        module._require_ptv2_historical_parent(historical, unrelated)


@pytest.mark.parametrize(
    ("field", "forged"),
    [
        ("trainer_epochs", 2),
        ("serialized_tokens", 4_097),
        ("packed_sequence_lower_bound", 2),
        ("unique_prompt_count", 511),
        ("natural_duplicate_count", 1),
        ("constructed_repeat_count", 1),
        ("milestone_occurrences", [511]),
        ("milestone_steps", [2]),
        ("segment_occurrences", [511, 1]),
        ("segment_steps", [2, 1]),
        ("cumulative_segment_steps", [2, 3]),
        ("segment_final_valid_occurrences", [511, 1]),
        ("tokenizer_sha256", "a" * 64),
        ("chat_template_sha256", "b" * 64),
        ("assistant_loss_target_sha256", "c" * 64),
        ("training_config_sha256", "d" * 64),
        ("source_response_root_sha256", "e" * 64),
        ("ordered_occurrences_sha256", "f" * 64),
        ("selection_sha256", "0" * 64),
        ("database_path", "other.sqlite3"),
        ("database_bytes", 1),
    ],
)
def test_ptv2_source_rejects_self_rehashed_immutable_lineage_mismatch(
    tmp_path: Path, field: str, forged: object
) -> None:
    module = _load_module()
    corpus = _write_ptv2_v2_corpus(
        module,
        tmp_path / "input",
        strategy="B-balanced",
        bucket_rows=[("math", "en", 256), ("stem", "en", 256)],
    )
    receipt_path = Path(corpus.receipt_path)
    receipt = json.loads(receipt_path.read_bytes())
    receipt.pop("receipt_sha256")
    receipt[field] = forged
    forged_sha256 = hashlib.sha256(_canonical(receipt)).hexdigest()
    receipt_path.write_bytes(_canonical(receipt | {"receipt_sha256": forged_sha256}) + b"\n")
    forged_corpus = replace(corpus, receipt_sha256=forged_sha256)

    with pytest.raises(module.ExposureViewError, match="claimed corpus"):
        module._materialize_ptv2_exposure(forged_corpus, 1_024, workers=1)

    assert not _ptv2_exposure_bundle(forged_corpus, 1_024).exists()


def test_ptv2_source_file_entry_replacement_cannot_change_consumed_bytes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _load_module()
    corpus = _write_ptv2_v2_corpus(
        module,
        tmp_path / "input",
        strategy="B-balanced",
        bucket_rows=[("math", "en", 256), ("stem", "en", 256)],
    )
    source = Path(corpus.tokenized_path)
    authenticated = tmp_path / "authenticated.sqlite3"
    replacement = tmp_path / "replacement.sqlite3"
    shutil.copy2(source, authenticated)
    shutil.copy2(source, replacement)
    with sqlite3.connect(replacement) as connection:
        connection.execute("UPDATE records SET input_ids_json='[99,99,99,99]' WHERE ordinal=0")
        connection.commit()
    real_sha256_file = module._sha256_file

    def hash_then_replace(path: Path) -> str:
        if Path(path) == source:
            shutil.copy2(authenticated, source)
        digest = real_sha256_file(path)
        if Path(path) == source:
            temporary = tmp_path / "replacement-copy.sqlite3"
            shutil.copy2(replacement, temporary)
            os.replace(temporary, source)
        return digest

    monkeypatch.setattr(module, "_sha256_file", hash_then_replace)
    try:
        module._materialize_ptv2_exposure(corpus, 1_024, workers=1)
    except module.ExposureViewError:
        return

    rows = [
        json.loads(line)
        for line in (_ptv2_exposure_bundle(corpus, 1_024) / "records.jsonl")
        .read_text()
        .splitlines()
    ]
    assert rows[0]["input_ids"] == [0, 1, 2, 3]


@pytest.mark.parametrize(
    ("column", "value"),
    [
        ("input_ids_json", [False, 1, 2, 3]),
        ("input_ids_json", [0, 1.0, 2, 3]),
        ("input_ids_json", [0, "1", 2, 3]),
        ("loss_mask_json", [True, 1, 1, 1]),
    ],
)
def test_ptv2_source_rejects_non_exact_integer_token_payloads(
    tmp_path: Path, column: str, value: object
) -> None:
    module = _load_module()
    corpus = _write_ptv2_v2_corpus(
        module,
        tmp_path / "input",
        strategy="B-balanced",
        bucket_rows=[("math", "en", 256), ("stem", "en", 256)],
    )
    forged = _resign_ptv2_tokenized_database(module, corpus, column, value)

    with pytest.raises(module.ExposureViewError, match=r"identity|payload"):
        module._materialize_ptv2_exposure(forged, 1_024, workers=1)

    assert not _ptv2_exposure_bundle(forged, 1_024).exists()


@pytest.mark.parametrize(
    ("field", "value"),
    [("input_ids", [False, 1, 2, 3]), ("loss_mask", [True, 1, 1, 1])],
)
def test_ptv2_output_rejects_non_exact_integer_token_payloads(
    tmp_path: Path, field: str, value: object
) -> None:
    module = _load_module()
    corpus = _write_ptv2_v2_corpus(
        module,
        tmp_path / "input",
        strategy="B-balanced",
        bucket_rows=[("math", "en", 256), ("stem", "en", 256)],
    )
    module._materialize_ptv2_exposure(corpus, 1_024, workers=1)
    bundle = _ptv2_exposure_bundle(corpus, 1_024)
    _resign_ptv2_exposure_bundle(bundle, field, value)

    with pytest.raises(module.ExposureViewError, match="mask semantics"):
        module._validate_ptv2_exposure_bundle(bundle, corpus)


@pytest.mark.parametrize(
    ("field", "value"),
    [("unexpected", "self-hashed"), ("batch_boundary_count", True)],
)
def test_ptv2_scientific_receipt_requires_exact_schema2_json_types_and_keys(
    tmp_path: Path, field: str, value: object
) -> None:
    module = _load_module()
    corpus = _write_ptv2_v2_corpus(
        module,
        tmp_path / "input",
        strategy="B-balanced",
        bucket_rows=[("math", "en", 256), ("stem", "en", 256)],
    )
    module._materialize_ptv2_exposure(corpus, 1_024, workers=1)
    bundle = _ptv2_exposure_bundle(corpus, 1_024)
    _resign_ptv2_scientific_receipt(bundle, field, value)

    with pytest.raises(module.ExposureViewError, match="scientific receipt schema"):
        module._validate_ptv2_exposure_bundle(bundle, corpus)


def test_ptv2_adoption_requires_an_authenticated_execution_receipt(tmp_path: Path) -> None:
    module = _load_module()
    corpus = _write_ptv2_v2_corpus(
        module,
        tmp_path / "input",
        strategy="B-balanced",
        bucket_rows=[("math", "en", 256), ("stem", "en", 256)],
    )
    module._materialize_ptv2_exposure(corpus, 1_024, workers=1)
    bundle = _ptv2_exposure_bundle(corpus, 1_024)
    (bundle / "EXECUTION.json").unlink()

    with pytest.raises(module.ExposureViewError, match="execution"):
        module._materialize_ptv2_exposure(corpus, 1_024, workers=1)


def test_ptv2_v2_publication_is_no_replace_and_tamper_is_rejected(tmp_path: Path) -> None:
    module = _load_module()
    corpus = _write_ptv2_v2_corpus(
        module,
        tmp_path / "input",
        strategy="B-balanced",
        bucket_rows=[("math", "en", 256), ("stem", "en", 256)],
    )
    first_sha256 = module._materialize_ptv2_exposure(corpus, 1_024, workers=1)
    assert module._materialize_ptv2_exposure(corpus, 1_024, workers=1) == first_sha256

    bundle = _ptv2_exposure_bundle(corpus, 1_024)
    records = bundle / "records.jsonl"
    records.write_bytes(records.read_bytes() + b"{}\n")
    with pytest.raises(module.ExposureViewError, match="reconciliation"):
        module._validate_ptv2_exposure_bundle(bundle, corpus)


def test_ptv2_exposure_parent_fsync_failure_rolls_back_the_installed_inode(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _load_module()
    corpus = _write_ptv2_v2_corpus(
        module,
        tmp_path / "input",
        strategy="B-balanced",
        bucket_rows=[("math", "en", 256), ("stem", "en", 256)],
    )
    destination = _ptv2_exposure_bundle(corpus, 1_024)
    publication_root = destination.parent
    real_fsync = module._fsync_directory

    def fail_publication_parent(path: Path) -> None:
        if path == publication_root:
            raise OSError("injected exposure parent fsync failure")
        real_fsync(path)

    monkeypatch.setattr(module, "_fsync_directory", fail_publication_parent)
    with pytest.raises(OSError, match="injected exposure parent fsync failure"):
        module._materialize_ptv2_exposure(corpus, 1_024, workers=1)

    assert not destination.exists()


def test_ptv2_publication_rollback_never_deletes_a_replacement_inode(tmp_path: Path) -> None:
    module = _load_module()
    installed = tmp_path / "installed"
    installed.mkdir()
    (installed / "owned").write_text("installed")
    identity = os.lstat(installed)
    destination = tmp_path / "destination"
    os.rename(installed, destination)
    displaced = tmp_path / "displaced-installed"
    os.rename(destination, displaced)
    replacement = tmp_path / "replacement"
    replacement.mkdir()
    (replacement / "sentinel").write_text("caller-owned")
    os.rename(replacement, destination)

    module._rollback_ptv2_publication(destination, identity)

    assert (destination / "sentinel").read_text() == "caller-owned"
    assert (displaced / "owned").read_text() == "installed"


def test_ptv2_v2_parallel_reader_failure_publishes_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _load_module()
    corpus = _write_ptv2_v2_corpus(
        module,
        tmp_path / "input",
        strategy="B-balanced",
        bucket_rows=[("math", "en", 256), ("stem", "en", 256)],
    )
    monkeypatch.setattr(module, "_read_ptv2_source_range", _injected_ptv2_reader_failure)
    with pytest.raises(RuntimeError, match="injected exposure reader failure"):
        module._materialize_ptv2_exposure(corpus, 1_024, workers=96)

    assert not _ptv2_exposure_bundle(corpus, 1_024).exists()
    assert not (Path(corpus.tokenized_path).parent.parent / "b-balanced-exposures").exists()


def test_ptv2_six_arm_failure_retries_by_authenticating_completed_publications(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _load_module()
    study = tmp_path / "study"
    prefix = _write_ptv2_v2_corpus(
        module,
        study,
        strategy="A-repair",
        bucket_rows=[("math", "en", 512), ("stem", "en", 512)],
        historical_prefix_rows=512,
    )
    balanced = _write_ptv2_v2_corpus(
        module,
        study,
        strategy="B-balanced",
        bucket_rows=[("chat", "en", 512), ("code", "en", 512)],
        historical_prefix_rows=512,
    )
    historical = module.derive_ptv2_historical_corpus(prefix, study)
    corpora = (prefix, balanced, historical)
    real_rename = module._rename_no_replace
    exposure_publications = 0

    def fail_fourth_exposure(source: Path, destination: Path) -> None:
        nonlocal exposure_publications
        if destination.name.endswith("assistant-tokens-v2"):
            exposure_publications += 1
            if exposure_publications == 4:
                raise RuntimeError("injected fourth exposure publication failure")
        real_rename(source, destination)

    monkeypatch.setattr(module, "_rename_no_replace", fail_fourth_exposure)
    with pytest.raises(module.ExposureViewError, match="immutable"):
        module._materialize_ptv2_exposure_set(
            corpora,
            runtime_screen_tokens=1_024,
            scientific_tokens=2_048,
            workers=1,
        )
    completion = study / "ptv2-study-exposures" / "COMPLETE.json"
    assert not completion.exists()

    monkeypatch.setattr(module, "_rename_no_replace", real_rename)
    runtime, scientific, completion_sha256 = module._materialize_ptv2_exposure_set(
        corpora,
        runtime_screen_tokens=1_024,
        scientific_tokens=2_048,
        workers=1,
    )

    assert set(runtime) == {"A-repair", "B-balanced", "H-historical-cyclic"}
    assert set(scientific) == set(runtime)
    receipt = json.loads(completion.read_bytes())
    body = dict(receipt)
    assert body.pop("receipt_sha256") == completion_sha256
    assert completion_sha256 == hashlib.sha256(_canonical(body)).hexdigest()
    assert receipt["runtime_artifacts"] == runtime
    assert receipt["scientific_artifacts"] == scientific


def _write_ptv2_three_arm_study(module, root: Path):
    prefix = _write_ptv2_v2_corpus(
        module,
        root,
        strategy="A-repair",
        bucket_rows=[("math", "en", 512), ("stem", "en", 512)],
        historical_prefix_rows=512,
    )
    balanced = _write_ptv2_v2_corpus(
        module,
        root,
        strategy="B-balanced",
        bucket_rows=[("chat", "en", 512), ("code", "en", 512)],
        historical_prefix_rows=512,
    )
    historical = module.derive_ptv2_historical_corpus(prefix, root)
    return prefix, balanced, historical


def _resign_ptv2_completion(path: Path, mutate) -> str:
    completion = json.loads(path.read_bytes())
    completion.pop("receipt_sha256")
    mutate(completion)
    receipt_sha256 = hashlib.sha256(_canonical(completion)).hexdigest()
    path.write_bytes(_canonical(completion | {"receipt_sha256": receipt_sha256}) + b"\n")
    return receipt_sha256


def _coordinate_ptv2_execution_and_completion_tamper(
    module: Any, study: Path, corpora: tuple[Any, ...]
) -> str:
    completion_path = study / "ptv2-study-exposures/COMPLETE.json"
    completion = json.loads(completion_path.read_bytes())
    completion.pop("receipt_sha256")
    for phase, target in (("runtime", 1_024), ("scientific", 2_048)):
        for corpus in corpora:
            bundle = _ptv2_exposure_bundle(corpus, target)
            execution_path = bundle / "EXECUTION.json"
            execution = json.loads(execution_path.read_bytes())
            execution.pop("receipt_sha256")
            execution["started_at_ns"] += 1
            execution["finished_at_ns"] += 1
            execution_sha256 = hashlib.sha256(_canonical(execution)).hexdigest()
            raw = _canonical(execution | {"receipt_sha256": execution_sha256}) + b"\n"
            execution_path.write_bytes(raw)
            completion["execution_artifacts"][phase][corpus.strategy] = {
                "path": "EXECUTION.json",
                "bytes": len(raw),
                "sha256": hashlib.sha256(raw).hexdigest(),
                "receipt_sha256": execution_sha256,
                "requested_workers": execution["requested_workers"],
                "effective_workers": execution["effective_workers"],
                "started_at_ns": execution["started_at_ns"],
                "finished_at_ns": execution["finished_at_ns"],
                "elapsed_seconds": execution["elapsed_seconds"],
            }
    receipt_sha256 = hashlib.sha256(_canonical(completion)).hexdigest()
    completion_path.write_bytes(_canonical(completion | {"receipt_sha256": receipt_sha256}) + b"\n")
    return receipt_sha256


def test_ptv2_completion_replay_requires_a_caller_pinned_identity(tmp_path: Path) -> None:
    module = _load_module()
    study = tmp_path / "study"
    corpora = _write_ptv2_three_arm_study(module, study)
    module._materialize_ptv2_exposure_set(
        corpora,
        runtime_screen_tokens=1_024,
        scientific_tokens=2_048,
        workers=1,
    )

    with pytest.raises(module.ExposureViewError, match="caller-pinned"):
        module._materialize_ptv2_exposure_set(
            corpora,
            runtime_screen_tokens=1_024,
            scientific_tokens=2_048,
            workers=1,
        )


def test_ptv2_completion_same_identity_retry_is_idempotent(tmp_path: Path) -> None:
    module = _load_module()
    study = tmp_path / "study"
    corpora = _write_ptv2_three_arm_study(module, study)
    first = module._materialize_ptv2_exposure_set(
        corpora,
        runtime_screen_tokens=1_024,
        scientific_tokens=2_048,
        workers=1,
    )

    second = module._materialize_ptv2_exposure_set(
        corpora,
        runtime_screen_tokens=1_024,
        scientific_tokens=2_048,
        workers=1,
        expected_completion_receipt_sha256=first[2],
    )

    assert second == first


def test_ptv2_completion_pin_rejects_coordinated_execution_and_receipt_tamper(
    tmp_path: Path,
) -> None:
    module = _load_module()
    study = tmp_path / "study"
    corpora = _write_ptv2_three_arm_study(module, study)
    _, _, trusted_sha256 = module._materialize_ptv2_exposure_set(
        corpora,
        runtime_screen_tokens=1_024,
        scientific_tokens=2_048,
        workers=1,
    )
    forged_sha256 = _coordinate_ptv2_execution_and_completion_tamper(module, study, corpora)
    assert forged_sha256 != trusted_sha256

    with pytest.raises(module.ExposureViewError, match="completion identity"):
        module._materialize_ptv2_exposure_set(
            corpora,
            runtime_screen_tokens=1_024,
            scientific_tokens=2_048,
            workers=1,
            expected_completion_receipt_sha256=trusted_sha256,
        )


@pytest.mark.parametrize("mutation", ["extra-key", "bool-worker"])
def test_ptv2_completion_receipt_requires_exact_schema2_json_types_and_keys(
    tmp_path: Path, mutation: str
) -> None:
    module = _load_module()
    study = tmp_path / "study"
    corpora = _write_ptv2_three_arm_study(module, study)
    module._materialize_ptv2_exposure_set(
        corpora,
        runtime_screen_tokens=1_024,
        scientific_tokens=2_048,
        workers=1,
    )
    completion_path = study / "ptv2-study-exposures/COMPLETE.json"

    def mutate(completion: dict[str, Any]) -> None:
        if mutation == "extra-key":
            completion["unexpected"] = "self-hashed"
        else:
            completion["execution_artifacts"]["runtime"]["A-repair"]["requested_workers"] = True

    forged_sha256 = _resign_ptv2_completion(completion_path, mutate)

    with pytest.raises(module.ExposureViewError, match="completion receipt schema"):
        module._materialize_ptv2_exposure_set(
            corpora,
            runtime_screen_tokens=1_024,
            scientific_tokens=2_048,
            workers=1,
            expected_completion_receipt_sha256=forged_sha256,
        )


def test_ptv2_completion_parent_fsync_failure_rolls_back_only_complete_receipt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _load_module()
    study = tmp_path / "study"
    corpora = _write_ptv2_three_arm_study(module, study)
    completion_root = study / "ptv2-study-exposures"
    completion_path = completion_root / "COMPLETE.json"
    real_fsync = module._fsync_directory

    def fail_completion_parent(path: Path) -> None:
        if path == completion_root:
            raise OSError("injected completion parent fsync failure")
        real_fsync(path)

    monkeypatch.setattr(module, "_fsync_directory", fail_completion_parent)
    with pytest.raises(OSError, match="injected completion parent fsync failure"):
        module._materialize_ptv2_exposure_set(
            corpora,
            runtime_screen_tokens=1_024,
            scientific_tokens=2_048,
            workers=1,
        )

    assert not completion_path.exists()
    for corpus in corpora:
        assert _ptv2_exposure_bundle(corpus, 1_024).is_dir()
        assert _ptv2_exposure_bundle(corpus, 2_048).is_dir()


def test_ptv2_new_exposure_root_fsyncs_its_parent_entry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _load_module()
    study = tmp_path / "study"
    corpus = _write_ptv2_v2_corpus(
        module,
        study,
        strategy="A-repair",
        bucket_rows=[("math", "en", 512), ("stem", "en", 512)],
    )
    observed: list[Path] = []
    real_fsync = module._fsync_directory

    def record_fsync(path: Path) -> None:
        observed.append(path)
        real_fsync(path)

    monkeypatch.setattr(module, "_fsync_directory", record_fsync)
    module._materialize_ptv2_exposure(corpus, 1_024, workers=1)

    assert study in observed


def test_ptv2_new_completion_root_fsyncs_its_parent_entry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _load_module()
    study = tmp_path / "study"
    corpora = _write_ptv2_three_arm_study(module, study)
    for corpus in corpora:
        module._materialize_ptv2_exposure(corpus, 1_024, workers=1)
        module._materialize_ptv2_exposure(corpus, 2_048, workers=1)
    observed: list[Path] = []
    real_fsync = module._fsync_directory

    def record_fsync(path: Path) -> None:
        observed.append(path)
        real_fsync(path)

    monkeypatch.setattr(module, "_fsync_directory", record_fsync)
    module._materialize_ptv2_exposure_set(
        corpora,
        runtime_screen_tokens=1_024,
        scientific_tokens=2_048,
        workers=1,
    )

    assert study in observed


def test_ptv2_completion_refuses_a_four_artifact_ab_scope_before_publication(
    tmp_path: Path,
) -> None:
    module = _load_module()
    prefix, balanced, _ = _write_ptv2_three_arm_study(module, tmp_path / "study")

    with pytest.raises(module.ExposureViewError, match="exact A/B/H"):
        module._materialize_ptv2_exposure_set(
            (prefix, balanced),
            runtime_screen_tokens=1_024,
            scientific_tokens=2_048,
            workers=1,
        )

    assert not (tmp_path / "study/a-repair-exposures").exists()
    assert not (tmp_path / "study/b-balanced-exposures").exists()


def test_ptv2_incompatible_completion_is_rejected_before_h_publication(
    tmp_path: Path,
) -> None:
    module = _load_module()
    study = tmp_path / "study"
    corpora = _write_ptv2_three_arm_study(module, study)
    completion_root = study / "ptv2-study-exposures"
    completion_root.mkdir()
    legacy = {
        "schema_version": 1,
        "arms": ["A-repair", "B-balanced"],
        "runtime_screen_tokens": 1_024,
        "scientific_tokens": 2_048,
    }
    legacy_sha256 = hashlib.sha256(_canonical(legacy)).hexdigest()
    (completion_root / "COMPLETE.json").write_bytes(
        _canonical(legacy | {"receipt_sha256": legacy_sha256}) + b"\n"
    )

    with pytest.raises(module.ExposureViewError, match="scope"):
        module._materialize_ptv2_exposure_set(
            corpora,
            runtime_screen_tokens=1_024,
            scientific_tokens=2_048,
            workers=1,
        )

    assert not (study / "h-historical-cyclic-exposures").exists()


def test_ptv2_completion_symlink_is_rejected_before_target_read_or_h_publication(
    tmp_path: Path,
) -> None:
    module = _load_module()
    study = tmp_path / "study"
    corpora = _write_ptv2_three_arm_study(module, study)
    completion_root = study / "ptv2-study-exposures"
    completion_root.mkdir()
    target = tmp_path / "target.json"
    target.write_text("{}\n")
    (completion_root / "COMPLETE.json").symlink_to(target)

    with pytest.raises(module.ExposureViewError, match="unsafe"):
        module._materialize_ptv2_exposure_set(
            corpora,
            runtime_screen_tokens=1_024,
            scientific_tokens=2_048,
            workers=1,
        )

    assert not (study / "h-historical-cyclic-exposures").exists()


def test_ptv2_completion_detects_self_rehashed_execution_tamper(tmp_path: Path) -> None:
    module = _load_module()
    study = tmp_path / "study"
    corpora = _write_ptv2_three_arm_study(module, study)
    _, _, completion_sha256 = module._materialize_ptv2_exposure_set(
        corpora,
        runtime_screen_tokens=1_024,
        scientific_tokens=2_048,
        workers=1,
    )
    execution_path = _ptv2_exposure_bundle(corpora[1], 1_024) / "EXECUTION.json"
    execution = json.loads(execution_path.read_bytes())
    execution.pop("receipt_sha256")
    execution["requested_workers"] = 96
    execution["effective_workers"] = 96
    execution_sha256 = hashlib.sha256(_canonical(execution)).hexdigest()
    execution_path.write_bytes(_canonical(execution | {"receipt_sha256": execution_sha256}) + b"\n")

    with pytest.raises(module.ExposureViewError, match="execution evidence"):
        module._materialize_ptv2_exposure_set(
            corpora,
            runtime_screen_tokens=1_024,
            scientific_tokens=2_048,
            workers=1,
            expected_completion_receipt_sha256=completion_sha256,
        )


def test_ptv2_legacy_tokenized_receipt_is_read_only_for_new_publication(
    tmp_path: Path,
) -> None:
    module = _load_module()
    corpus = _write_ptv2_v2_corpus(
        module,
        tmp_path / "input",
        strategy="A-repair",
        bucket_rows=[("math", "en", 256), ("stem", "en", 256)],
    )
    receipt_path = Path(corpus.receipt_path)
    receipt = json.loads(receipt_path.read_bytes())
    receipt.pop("receipt_sha256")
    receipt["schema_version"] = 1
    receipt_sha256 = hashlib.sha256(_canonical(receipt)).hexdigest()
    receipt_path.write_bytes(_canonical(receipt | {"receipt_sha256": receipt_sha256}) + b"\n")
    legacy = replace(corpus, receipt_sha256=receipt_sha256)

    with pytest.raises(module.ExposureViewError, match="v2 tokenized receipt"):
        module._materialize_ptv2_exposure(legacy, 1_024, workers=1)

    assert not _ptv2_exposure_bundle(legacy, 1_024).exists()


def test_ptv2_scientific_cycle_requires_authenticated_v2_occurrences() -> None:
    """A scalar total alone cannot authorize cyclic exact-token publication."""
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

    with pytest.raises(module.ExposureViewError, match=r"one-pass.*256M"):
        module.build_ptv2_study_exposures(prefix, balanced, scientific_tokens=256_000_000)


def test_ptv2_ab_exposure_cannot_cycle_beyond_authenticated_one_pass(tmp_path: Path) -> None:
    module = _load_module()
    corpus = _write_ptv2_v2_corpus(
        module,
        tmp_path / "input",
        strategy="A-repair",
        bucket_rows=[("math", "en", 128), ("stem", "en", 128)],
    )
    assert corpus.assistant_tokens == 1_024

    with pytest.raises(module.ExposureViewError, match="one-pass reachability"):
        module._materialize_ptv2_exposure(corpus, 2_048, workers=1)

    assert not _ptv2_exposure_bundle(corpus, 2_048).exists()


def test_ptv2_authenticated_h_exposure_labels_its_explicit_cyclic_repeats(
    tmp_path: Path,
) -> None:
    module = _load_module()
    prefix = _write_ptv2_v2_corpus(
        module,
        tmp_path / "study",
        strategy="A-repair",
        bucket_rows=[("math", "en", 256), ("stem", "en", 256)],
        historical_prefix_rows=512,
    )
    historical = module.derive_ptv2_historical_corpus(prefix, tmp_path / "study")
    target_tokens = historical.assistant_tokens * 2

    module._materialize_ptv2_exposure(historical, target_tokens, workers=1)

    bundle = _ptv2_exposure_bundle(historical, target_tokens)
    receipt = json.loads((bundle / "SCIENTIFIC.json").read_bytes())
    records = [json.loads(line) for line in (bundle / "records.jsonl").read_text().splitlines()]
    assert receipt["repeat_policy"] == "authenticated-historical-cyclic-v1"
    assert {record["cycle_index"] for record in records} == {0, 1}


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
        "language TEXT,canonical_conversation TEXT,assistant_response TEXT);"
        "CREATE TABLE occurrences(strategy TEXT,ordinal INTEGER,prompt_uuid TEXT,"
        "source_identity_sha256 TEXT,source_row INTEGER,cell TEXT,reuse_index INTEGER,"
        "conversation_sha256 TEXT,assistant_response_sha256 TEXT);"
    )
    connection.execute(
        "INSERT INTO source_rows VALUES(?,?,?,?,?)", ("b" * 64, 0, "en", conversation, response)
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
        selection_sha256="c" * 64,
        source_response_root_sha256=response_root,
        unique_prompt_count=1,
        natural_duplicate_count=0,
        constructed_repeat_count=0,
        trainer_epochs=1,
    )

    with pytest.raises(module.ExposureViewError, match="final assistant"):
        module.derive_ptv2_one_pass_corpus(
            view,
            tokenizer_sha256=TOKENIZER_SHA256,
            chat_template_sha256=CHAT_TEMPLATE_SHA256,
            assistant_loss_target_sha256="2" * 64,
            training_config_sha256="3" * 64,
            tokenizer=_Tokenizer(),
            output_root=tmp_path / "tokenized",
        )


def test_ptv2_derivation_publishes_an_authenticated_token_bundle_on_apfs(tmp_path: Path) -> None:
    """A real tokenizer invocation writes durable Task7 token evidence without link-count assumptions."""
    module = _load_module()
    index = tmp_path / "selection.sqlite3"
    assistant = {"role": "assistant", "content": json.dumps({"ids": [1, 2, 3], "mask": [0, 1, 1]})}
    conversation = _canonical(
        {"messages": [{"role": "user", "content": "prompt"}, assistant]}
    ).decode()
    response = _canonical(assistant).decode()
    conversation_sha256 = hashlib.sha256(conversation.encode()).hexdigest()
    response_sha256 = hashlib.sha256(response.encode()).hexdigest()
    occurrence = (0, "a" * 64, "b" * 64, 0, "math", 0, conversation_sha256, response_sha256)
    connection = sqlite3.connect(index)
    connection.executescript(
        "CREATE TABLE source_rows("
        "source_identity_sha256 TEXT,source_row INTEGER,language TEXT,canonical_conversation TEXT,"
        "assistant_response TEXT);"
        "CREATE TABLE occurrences("
        "strategy TEXT,ordinal INTEGER,prompt_uuid TEXT,source_identity_sha256 TEXT,"
        "source_row INTEGER,cell TEXT,reuse_index INTEGER,conversation_sha256 TEXT,"
        "assistant_response_sha256 TEXT);"
    )
    connection.execute(
        "INSERT INTO source_rows VALUES(?,?,?,?,?)", ("b" * 64, 0, "en", conversation, response)
    )
    connection.execute(
        "INSERT INTO occurrences VALUES(?,?,?,?,?,?,?,?,?)", ("B-balanced", *occurrence)
    )
    connection.commit()
    connection.close()
    view = SimpleNamespace(
        strategy="B-balanced",
        index_path=index,
        occurrence_count=1,
        ordered_occurrences_sha256=hashlib.sha256(_canonical(list(occurrence)) + b"\n").hexdigest(),
        selection_sha256="c" * 64,
        source_response_root_sha256=hashlib.sha256(
            _canonical(["b" * 64, 0, conversation_sha256, response_sha256]) + b"\n"
        ).hexdigest(),
        unique_prompt_count=1,
        natural_duplicate_count=0,
        constructed_repeat_count=0,
        trainer_epochs=1,
    )

    corpus = module.derive_ptv2_one_pass_corpus(
        view,
        tokenizer_sha256=TOKENIZER_SHA256,
        chat_template_sha256=CHAT_TEMPLATE_SHA256,
        assistant_loss_target_sha256="2" * 64,
        training_config_sha256="3" * 64,
        tokenizer=_Tokenizer(),
        output_root=tmp_path / "tokenized",
    )

    assert corpus.assistant_tokens == 2
    assert Path(corpus.tokenized_path).is_file()
    assert Path(corpus.receipt_path).is_file()


def test_ptv2_task8_one_vs_96_is_byte_identical(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _load_module()
    monkeypatch.setenv("SLURM_CPUS_PER_TASK", "96")
    index = tmp_path / "selection-201.sqlite3"
    connection = sqlite3.connect(index)
    connection.executescript(
        "CREATE TABLE source_rows(source_identity_sha256 TEXT,source_row INTEGER,"
        "language TEXT,canonical_conversation TEXT,assistant_response TEXT);"
        "CREATE TABLE occurrences(strategy TEXT,ordinal INTEGER,prompt_uuid TEXT,"
        "source_identity_sha256 TEXT,source_row INTEGER,cell TEXT,reuse_index INTEGER,"
        "conversation_sha256 TEXT,assistant_response_sha256 TEXT);"
    )
    occurrence_digest = hashlib.sha256()
    response_digest = hashlib.sha256()
    for ordinal in range(201):
        assistant = {
            "role": "assistant",
            "content": json.dumps({"ids": [ordinal + 1, 2], "mask": [0, 1]}),
        }
        conversation = _canonical(
            {"messages": [{"role": "user", "content": str(ordinal)}, assistant], "tools": []}
        ).decode()
        response = _canonical(assistant).decode()
        source_identity = hashlib.sha256(f"source-{ordinal}".encode()).hexdigest()
        prompt = hashlib.sha256(f"prompt-{ordinal}".encode()).hexdigest()
        conversation_sha = hashlib.sha256(conversation.encode()).hexdigest()
        response_sha = hashlib.sha256(response.encode()).hexdigest()
        occurrence = (
            ordinal,
            prompt,
            source_identity,
            ordinal,
            "math",
            0,
            conversation_sha,
            response_sha,
        )
        connection.execute(
            "INSERT INTO source_rows VALUES(?,?,?,?,?)",
            (source_identity, ordinal, "en", conversation, response),
        )
        connection.execute(
            "INSERT INTO occurrences VALUES(?,?,?,?,?,?,?,?,?)",
            ("B-balanced", *occurrence),
        )
        occurrence_digest.update(_canonical(list(occurrence)) + b"\n")
        response_digest.update(
            _canonical([source_identity, ordinal, conversation_sha, response_sha]) + b"\n"
        )
    connection.commit()
    connection.close()
    original_index_bytes = index.stat().st_size
    original_index_sha256 = hashlib.sha256(index.read_bytes()).hexdigest()
    view = SimpleNamespace(
        strategy="B-balanced",
        index_path=index,
        occurrence_count=201,
        ordered_occurrences_sha256=occurrence_digest.hexdigest(),
        selection_sha256="c" * 64,
        source_response_root_sha256=response_digest.hexdigest(),
        unique_prompt_count=201,
        natural_duplicate_count=0,
        constructed_repeat_count=0,
        trainer_epochs=1,
    )
    common = {
        "tokenizer_sha256": TOKENIZER_SHA256,
        "chat_template_sha256": CHAT_TEMPLATE_SHA256,
        "assistant_loss_target_sha256": "2" * 64,
        "training_config_sha256": "3" * 64,
        "tokenizer": _Tokenizer(),
        "milestone_occurrences": (201,),
        "milestone_steps": (1,),
    }
    serial = module.derive_ptv2_one_pass_corpus(view, output_root=tmp_path / "serial", **common)
    real_pretokenize = module._pretokenize_task8

    def stage_then_mutate(**kwargs):
        result = real_pretokenize(**kwargs)
        with sqlite3.connect(index) as mutable:
            mutable.execute(
                "UPDATE source_rows SET canonical_conversation='mutated-after-stage' "
                "WHERE source_row=0"
            )
            mutable.commit()
        return result

    monkeypatch.setattr(module, "_pretokenize_task8", stage_then_mutate)
    parallel = module.derive_ptv2_one_pass_corpus(
        view,
        output_root=tmp_path / "parallel",
        work_root=tmp_path / "parallel-work",
        workers=96,
        source_commit="1" * 40,
        **common,
    )

    assert Path(serial.tokenized_path).read_bytes() == Path(parallel.tokenized_path).read_bytes()
    assert serial.tokenized_sha256 == parallel.tokenized_sha256
    bundle = tmp_path / "parallel/b-balanced-tokenized"
    execution_path = bundle / "EXECUTION_RECEIPT.json"
    assert execution_path.is_file()
    assert not (tmp_path / "parallel/b-balanced-tokenized.EXECUTION_RECEIPT.json").exists()
    execution = json.loads(execution_path.read_bytes())
    body = dict(execution)
    claimed = body.pop("receipt_sha256")
    assert claimed == hashlib.sha256(_canonical(body)).hexdigest()
    assert execution["source_commit"] == "1" * 40
    assert execution["selection_sha256"] == "c" * 64
    assert execution["declared_range_count"] == 201
    assert execution["effective_workers"] == 96
    assert execution["source_index_bytes"] == original_index_bytes
    assert execution["source_index_sha256"] == original_index_sha256
    assert execution["finished_at_ns"] >= (bundle / "records.sqlite3").stat().st_mtime_ns
    assert execution["elapsed_seconds"] >= execution["parallel_phase_elapsed_seconds"]
    assert execution["finished_at_ns"] >= execution["parallel_phase_finished_at_ns"]
    assert set(execution["thread_environment"].values()) == {"1"}
    tokenized = json.loads((bundle / "TOKENIZED.json").read_bytes())
    assert tokenized["execution_receipt"] == {
        "path": "EXECUTION_RECEIPT.json",
        "bytes": execution_path.stat().st_size,
        "sha256": hashlib.sha256(execution_path.read_bytes()).hexdigest(),
    }
    assert list((tmp_path / "parallel-work").iterdir()) == []


def test_ptv2_task8_exact_two_million_partition_is_201_source_order_ranges() -> None:
    module = _load_module()

    ranges = module._task8_ranges(2_000_000, 201)

    assert len(ranges) == 201
    assert ranges[0] == (0, 9_951)
    assert ranges[49] == (487_599, 497_550)
    assert ranges[50] == (497_550, 507_500)
    assert ranges[-1] == (1_990_050, 2_000_000)
    assert all(left[1] == right[0] for left, right in pairwise(ranges))
    assert module.resolve_task8_worker_count(
        96,
        occurrence_count=2_000_000,
        declared_ranges=201,
        environ={"SLURM_CPUS_PER_TASK": "96"},
    ) == (96, 96)


def test_ptv2_task8_worker_failure_cleans_spools_and_publishes_nothing(tmp_path: Path) -> None:
    module = _load_module()
    index = tmp_path / "selection.sqlite3"
    assistant = {"role": "assistant", "content": json.dumps({"ids": [1], "mask": [1]})}
    conversation = _canonical(
        {"messages": [{"role": "user", "content": "q"}, assistant], "tools": []}
    ).decode()
    response = _canonical(assistant).decode()
    conversation_sha = hashlib.sha256(conversation.encode()).hexdigest()
    response_sha = hashlib.sha256(response.encode()).hexdigest()
    connection = sqlite3.connect(index)
    connection.executescript(
        "CREATE TABLE source_rows(source_identity_sha256 TEXT,source_row INTEGER,"
        "language TEXT,canonical_conversation TEXT,assistant_response TEXT);"
        "CREATE TABLE occurrences(strategy TEXT,ordinal INTEGER,prompt_uuid TEXT,"
        "source_identity_sha256 TEXT,source_row INTEGER,cell TEXT,reuse_index INTEGER,"
        "conversation_sha256 TEXT,assistant_response_sha256 TEXT);"
    )
    connection.execute(
        "INSERT INTO source_rows VALUES(?,?,?,?,?)", ("b" * 64, 0, "en", conversation, response)
    )
    connection.execute(
        "INSERT INTO occurrences VALUES(?,?,?,?,?,?,?,?,?)",
        ("B-balanced", 0, "a" * 64, "b" * 64, 0, "math", 0, conversation_sha, response_sha),
    )
    connection.commit()
    connection.close()
    work = tmp_path / "work"

    with pytest.raises(ValueError, match="injected Task8 worker failure"):
        module._pretokenize_task8(
            index_path=index,
            strategy="B-balanced",
            occurrence_count=1,
            sequence_length=4_096,
            tokenizer=_FailingTask8Tokenizer(),
            work_root=work,
            workers=96,
            source_commit="1" * 40,
        )

    assert list(work.iterdir()) == []
    assert not (tmp_path / "b-balanced-tokenized").exists()


def test_ptv2_token_bundle_publish_is_atomic_no_replace_and_apfs_safe(tmp_path: Path) -> None:
    """The Task 7 bundle path must not rely on POSIX directory link counts."""
    module = _load_module()
    destination = tmp_path / "b-balanced-tokenized"
    temporary = module._prepare_ptv2_tokenized_bundle(tmp_path, destination)
    (temporary / "records.sqlite3").write_bytes(b"sqlite")
    (temporary / "TOKENIZED.json").write_bytes(b"{}\n")
    module._publish_ptv2_tokenized_bundle(temporary, destination)

    assert (destination / "records.sqlite3").read_bytes() == b"sqlite"
    with pytest.raises(module.ExposureViewError, match="immutable"):
        module._prepare_ptv2_tokenized_bundle(tmp_path, destination)


def test_ptv2_token_bundle_parent_fsync_failure_rolls_back_the_installed_inode(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _load_module()
    destination = tmp_path / "b-balanced-tokenized"
    temporary = module._prepare_ptv2_tokenized_bundle(tmp_path, destination)
    (temporary / "records.sqlite3").write_bytes(b"sqlite")
    (temporary / "TOKENIZED.json").write_bytes(b"{}\n")
    real_fsync = module._fsync_directory

    def fail_publication_parent(path: Path) -> None:
        if path == destination.parent:
            raise OSError("injected tokenized parent fsync failure")
        real_fsync(path)

    monkeypatch.setattr(module, "_fsync_directory", fail_publication_parent)
    with pytest.raises(OSError, match="injected tokenized parent fsync failure"):
        module._publish_ptv2_tokenized_bundle(temporary, destination)

    assert not destination.exists()


def test_ptv2_token_bundle_partial_requires_typed_recovery(tmp_path: Path) -> None:
    """A poisoned retry preserves the private namespace for explicit recovery."""
    module = _load_module()
    (tmp_path / ".b-balanced-tokenized.partial-poisoned").mkdir()

    with pytest.raises(module.PTV2TokenizedRecoveryError, match="typed recovery"):
        module._prepare_ptv2_tokenized_bundle(tmp_path, tmp_path / "b-balanced-tokenized")


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
    from tokenizers import Tokenizer  # pyright: ignore[reportMissingImports]
    from tokenizers.models import WordLevel  # pyright: ignore[reportMissingImports]
    from tokenizers.pre_tokenizers import Whitespace  # pyright: ignore[reportMissingImports]
    from transformers import PreTrainedTokenizerFast  # pyright: ignore[reportMissingImports]

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
