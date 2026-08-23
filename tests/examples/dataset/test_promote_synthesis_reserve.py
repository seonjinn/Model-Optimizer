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

"""Response validation and deterministic reserve-promotion contracts."""

from __future__ import annotations

import json
import sqlite3
import sys
from dataclasses import asdict, replace
from hashlib import sha256
from pathlib import Path
from types import MappingProxyType
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[3]
MODULE_DIR = ROOT / "examples/dataset"
sys.path.insert(0, str(MODULE_DIR))
try:
    from promote_synthesis_reserve import (
        GenerationAttempt,
        GenerationIdentity,
        ResponsePromotionError,
        ValidatedReplayRecord,
        load_prompt_view,
        promote_response_artifacts,
        promote_responses,
        validated_replay_record,
    )
    from select_bprime_cd_prompts import PromptView, SelectedPrompt
finally:
    sys.path.pop(0)


def _canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


IDENTITY = GenerationIdentity(
    target_revision="1" * 40,
    tokenizer_sha256="2" * 64,
    chat_template_sha256="3" * 64,
    runtime_sha256="4" * 64,
    container_sha256="5" * 64,
    source_selection_sha256="6" * 64,
    thinking_mode="on",
    temperature=0.0,
    max_tokens=4096,
    max_total_length=32768,
)


def _row(prompt_id: str, domain: str, rank: int, status: str) -> SelectedPrompt:
    prompt = {"messages": [{"role": "user", "content": f"question-{prompt_id}"}]}
    canonical_prompt = _canonical(prompt)
    return SelectedPrompt(
        prompt_uuid=prompt_id,
        arm="C",
        domain=domain,
        lane="target-synth",
        language="en",
        context_bucket="le4k",
        source_id="source",
        source_family="ptv3",
        source_repository_id="repo",
        source_configuration="default",
        source_split="train",
        source_revision="7" * 40,
        source_file_sha256="8" * 64,
        source_manifest_sha256="9" * 64,
        source_file_path="shard.jsonl",
        source_row_index=rank,
        candidate_rank=rank,
        candidate_rank_sha256="a" * 64,
        selection_index=rank,
        status=status,  # type: ignore[arg-type]
        canonical_prompt_json=canonical_prompt,
    )


PRIMARY = (
    _row("b" * 64, "math", 0, "primary"),
    _row("c" * 64, "math", 1, "primary"),
    _row("d" * 64, "code", 2, "primary"),
)
RESERVE = (
    _row("e" * 64, "math", 0, "reserve"),
    _row("f" * 64, "math", 1, "reserve"),
    _row("0" * 64, "code", 2, "reserve"),
)
VIEW = PromptView(
    arm="C",
    primary_rows=PRIMARY,
    reserve_rows=RESERVE,
    cell_counts=MappingProxyType({"math": 2, "code": 1}),
    lane_counts=MappingProxyType({"agentless-swe": 0}),
    bucket_floors=MappingProxyType({}),
    non_agentic_bucket_floors=MappingProxyType({}),
    lane_bucket_floors=MappingProxyType({}),
    count_proof_sha256="1" * 64,
)


def _attempt(
    row: SelectedPrompt,
    *,
    attempt_id: str | None = None,
    content: str = "answer",
    finish_reason: str = "stop",
    identity: GenerationIdentity = IDENTITY,
    response: dict[str, Any] | None = None,
) -> GenerationAttempt:
    request = {
        "messages": json.loads(row.canonical_prompt_json)["messages"],
        "generation_identity_sha256": identity.sha256,
    }
    response = response or {
        "message": {"role": "assistant", "content": content},
        "finish_reason": finish_reason,
        "completion_tokens": 4,
    }
    request_json = _canonical(request)
    response_json = _canonical(response)
    return GenerationAttempt(
        attempt_id=attempt_id or sha256(f"{row.prompt_uuid}:{content}".encode()).hexdigest(),
        prompt_uuid=row.prompt_uuid,
        generation_identity_sha256=identity.sha256,
        request_json=request_json,
        request_sha256=sha256(request_json.encode()).hexdigest(),
        response_json=response_json,
        response_sha256=sha256(response_json.encode()).hexdigest(),
    )


def test_generation_identity_rejects_wrong_target_runtime_mode_and_request() -> None:
    valid = _attempt(PRIMARY[0])
    mutations = [
        replace(valid, generation_identity_sha256="0" * 64),
        replace(valid, request_sha256="0" * 64),
        replace(valid, response_sha256="0" * 64),
    ]
    for attempt in mutations:
        with pytest.raises(ResponsePromotionError):
            promote_responses(VIEW, [attempt, _attempt(PRIMARY[1]), _attempt(PRIMARY[2])], IDENTITY)


@pytest.mark.parametrize(
    "response",
    [
        {
            "message": {"role": "assistant", "content": "answer"},
            "finish_reason": "length",
            "completion_tokens": 4,
        },
        {
            "message": {"role": "assistant", "content": ""},
            "finish_reason": "stop",
            "completion_tokens": 0,
        },
        {
            "message": {"role": "assistant", "content": "answer", "tool_calls": [{"id": "x"}]},
            "finish_reason": "stop",
            "completion_tokens": 4,
        },
        {
            "message": {"role": "tool", "content": "output"},
            "finish_reason": "stop",
            "completion_tokens": 4,
        },
    ],
)
def test_invalid_or_partial_target_response_is_a_failed_attempt(response: dict[str, Any]) -> None:
    corpus = promote_responses(
        VIEW,
        [
            _attempt(PRIMARY[0], response=response),
            _attempt(PRIMARY[1]),
            _attempt(PRIMARY[2]),
            _attempt(RESERVE[0]),
        ],
        IDENTITY,
    )

    assert PRIMARY[0].prompt_uuid in corpus.failed_prompt_ids
    assert RESERVE[0].prompt_uuid in corpus.promoted_ids


@pytest.mark.parametrize(
    "assistant_extension",
    [
        {"name": "extra"},
        {"function_call": {"name": "legacy", "arguments": "{}"}},
        {"tool_calls": []},
        {"tool_call_id": ""},
    ],
)
def test_target_response_rejects_open_or_present_tool_schema(
    assistant_extension: dict[str, Any],
) -> None:
    response = {
        "message": {"role": "assistant", "content": "answer"} | assistant_extension,
        "finish_reason": "stop",
        "completion_tokens": 4,
    }
    corpus = promote_responses(
        VIEW,
        [
            _attempt(PRIMARY[0], response=response),
            _attempt(PRIMARY[1]),
            _attempt(PRIMARY[2]),
            _attempt(RESERVE[0]),
        ],
        IDENTITY,
    )

    assert PRIMARY[0].prompt_uuid in corpus.failed_prompt_ids
    assert RESERVE[0].prompt_uuid in corpus.promoted_ids


def test_source_assistant_completion_is_rejected_before_promotion() -> None:
    prompt = json.loads(PRIMARY[0].canonical_prompt_json)
    prompt["messages"].append({"role": "assistant", "content": "source answer"})
    tainted = replace(PRIMARY[0], canonical_prompt_json=_canonical(prompt))
    view = replace(VIEW, primary_rows=(tainted, *PRIMARY[1:]))

    with pytest.raises(ResponsePromotionError, match="source assistant"):
        promote_responses(
            view, [_attempt(tainted), _attempt(PRIMARY[1]), _attempt(PRIMARY[2])], IDENTITY
        )


def test_missing_attempt_identity_and_conflicting_successes_are_fatal() -> None:
    missing = replace(_attempt(PRIMARY[0]), attempt_id="")
    with pytest.raises(ResponsePromotionError, match="attempt identity"):
        promote_responses(VIEW, [missing, _attempt(PRIMARY[1]), _attempt(PRIMARY[2])], IDENTITY)

    with pytest.raises(ResponsePromotionError, match="conflicting"):
        promote_responses(
            VIEW,
            [
                _attempt(PRIMARY[0], content="first"),
                _attempt(PRIMARY[0], content="second"),
                _attempt(PRIMARY[1]),
                _attempt(PRIMARY[2]),
            ],
            IDENTITY,
        )


def test_promotion_uses_only_same_cell_reserve_and_reports_unused() -> None:
    attempts = [
        _attempt(PRIMARY[0], finish_reason="length"),
        _attempt(PRIMARY[1]),
        _attempt(PRIMARY[2]),
        _attempt(RESERVE[0]),
        _attempt(RESERVE[1]),
        _attempt(RESERVE[2]),
    ]

    corpus = promote_responses(VIEW, attempts, IDENTITY)

    assert corpus.cell_counts == VIEW.cell_counts
    assert corpus.promoted_ids == (
        PRIMARY[1].prompt_uuid,
        PRIMARY[2].prompt_uuid,
        RESERVE[0].prompt_uuid,
    )
    assert corpus.unused_reserve_ids == (RESERVE[1].prompt_uuid, RESERVE[2].prompt_uuid)
    assert corpus.response_token_histograms == {"code": {4: 1}, "math": {4: 2}}
    assert len(corpus.records) == 3
    for record in corpus.records:
        assert record.prompt_uuid in corpus.promoted_ids
        assert record.arm == "C"
        assert record.domain in {"math", "code"}
        assert record.lane == "target-synth"
        assert record.context_bucket == "le4k"
        assert record.request_sha256 is not None
        assert record.response_sha256 is not None
        assert record.selection_sha256 == IDENTITY.source_selection_sha256
        assert record.generation_identity_sha256 == IDENTITY.sha256
        assert record.attempt_or_replay_validation_sha256
        without_digest = {
            key: value for key, value in vars(record).items() if key != "record_sha256"
        }
        assert record.record_sha256 == sha256(_canonical(without_digest).encode()).hexdigest()


def test_retries_are_idempotent_and_first_canonical_success_wins() -> None:
    retry = _attempt(PRIMARY[0])
    corpus = promote_responses(
        VIEW,
        [retry, retry, _attempt(PRIMARY[1]), _attempt(PRIMARY[2])],
        IDENTITY,
    )

    assert corpus.promoted_ids == tuple(row.prompt_uuid for row in PRIMARY)
    assert len(corpus.successful_attempts) == 3
    assert len(corpus.attempt_history[PRIMARY[0].prompt_uuid]) == 1


def test_corpus_digest_binds_complete_retry_and_failed_attempt_history() -> None:
    base_attempts = [_attempt(PRIMARY[0]), _attempt(PRIMARY[1]), _attempt(PRIMARY[2])]
    base = promote_responses(VIEW, base_attempts, IDENTITY)
    retry = replace(_attempt(PRIMARY[0]), attempt_id="1" * 64)
    with_retry = promote_responses(VIEW, [*base_attempts, retry], IDENTITY)
    failed = _attempt(RESERVE[0], attempt_id="2" * 64, finish_reason="length")
    with_failure = promote_responses(VIEW, [*base_attempts, failed], IDENTITY)

    assert base.corpus_sha256 != with_retry.corpus_sha256
    assert base.corpus_sha256 != with_failure.corpus_sha256


def test_cell_shortfall_fails_atomically_without_cross_cell_replacement() -> None:
    attempts = [
        _attempt(PRIMARY[0]),
        _attempt(PRIMARY[1], finish_reason="length"),
        _attempt(PRIMARY[2]),
        _attempt(RESERVE[2]),
    ]

    with pytest.raises(ResponsePromotionError, match=r"math.*quota"):
        promote_responses(VIEW, attempts, IDENTITY)


def test_d_replay_records_join_only_after_target_promotion_and_require_validation() -> None:
    replay_row = replace(
        _row("a" * 64, "swe-agentic-tool", 3, "primary"),
        arm="D",
        lane="generic-tool-replay",
        canonical_prompt_json=_canonical(
            {
                "tools": [
                    {
                        "type": "function",
                        "function": {"name": "shell", "parameters": {"type": "object"}},
                    }
                ],
                "messages": [
                    {"role": "user", "content": "use tool"},
                    {
                        "role": "assistant",
                        "content": "",
                        "tool_calls": [
                            {
                                "id": "call-1",
                                "type": "function",
                                "function": {"name": "shell", "arguments": "{}"},
                            }
                        ],
                    },
                    {
                        "role": "tool",
                        "tool_call_id": "call-1",
                        "name": "shell",
                        "content": "ok",
                    },
                    {"role": "assistant", "content": "done"},
                ],
            }
        ),
    )
    view = replace(
        VIEW,
        arm="D",
        primary_rows=(*PRIMARY, replay_row),
        cell_counts=MappingProxyType({"math": 2, "code": 1, "swe-agentic-tool": 1}),
    )
    replay = validated_replay_record(replay_row, assistant_tokens=3)

    corpus = promote_responses(
        view,
        [_attempt(PRIMARY[0]), _attempt(PRIMARY[1]), _attempt(PRIMARY[2])],
        IDENTITY,
        replay_records=[replay],
    )

    assert corpus.promoted_ids[-1] == replay_row.prompt_uuid
    assert corpus.response_token_histograms["swe-agentic-tool"] == {3: 1}
    with pytest.raises(ResponsePromotionError, match="validated replay"):
        promote_responses(
            view,
            [_attempt(PRIMARY[0]), _attempt(PRIMARY[1]), _attempt(PRIMARY[2])],
            IDENTITY,
        )


def test_d_replay_rejects_arbitrary_proof_and_plain_chat() -> None:
    replay_row = replace(
        _row("a" * 64, "swe-agentic-tool", 3, "primary"),
        arm="D",
        lane="generic-tool-replay",
        canonical_prompt_json=_canonical(
            {
                "messages": [
                    {"role": "user", "content": "not a tool trajectory"},
                    {"role": "assistant", "content": "plain answer"},
                ]
            }
        ),
    )
    view = replace(
        VIEW,
        arm="D",
        primary_rows=(*PRIMARY, replay_row),
        cell_counts=MappingProxyType({"math": 2, "code": 1, "swe-agentic-tool": 1}),
    )
    replay_json = replay_row.canonical_prompt_json
    forged = ValidatedReplayRecord(
        prompt_uuid=replay_row.prompt_uuid,
        canonical_record_json=replay_json,
        record_sha256=sha256(replay_json.encode()).hexdigest(),
        trajectory_validation_sha256="7" * 64,
        assistant_tokens=3,
    )

    with pytest.raises(ResponsePromotionError, match="trajectory"):
        promote_responses(
            view,
            [_attempt(PRIMARY[0]), _attempt(PRIMARY[1]), _attempt(PRIMARY[2])],
            IDENTITY,
            replay_records=[forged],
        )


def _write_published_view(tmp_path: Path) -> tuple[Path, str]:
    root = tmp_path / "selection"
    shards = root / "shards"
    shards.mkdir(parents=True)
    shard = shards / "rows-000001.jsonl"
    index = sqlite3.connect(root / "selection-index.sqlite3")
    index.execute(
        "CREATE TABLE rows(arm,status,selection_index,lane,prompt_uuid,shard_path,"
        "byte_offset,byte_length,row_sha256,domain,language,candidate_rank)"
    )
    with shard.open("wb") as output:
        for row in (*PRIMARY, *RESERVE):
            record = asdict(row)
            record["canonical_prompt"] = json.loads(record.pop("canonical_prompt_json"))
            line = _canonical(record).encode() + b"\n"
            offset = output.tell()
            output.write(line)
            index.execute(
                "INSERT INTO rows VALUES(?,?,?,?,?,?,?,?,?,?,?,?)",
                (
                    row.arm,
                    row.status,
                    row.selection_index,
                    row.lane,
                    row.prompt_uuid,
                    "shards/rows-000001.jsonl",
                    offset,
                    len(line),
                    sha256(line).hexdigest(),
                    row.domain,
                    row.language,
                    row.candidate_rank,
                ),
            )
    index.commit()
    index.close()
    index_path = root / "selection-index.sqlite3"
    manifest = {
        "schema_version": 2,
        "selection_sha256": "3" * 64,
        "paired_cd_sha256": "4" * 64,
        "row_count": 6,
        "shards": [
            {
                "path": "shards/rows-000001.jsonl",
                "row_count": 6,
                "byte_count": shard.stat().st_size,
                "sha256": sha256(shard.read_bytes()).hexdigest(),
            }
        ],
        "index": {
            "path": index_path.name,
            "sha256": sha256(index_path.read_bytes()).hexdigest(),
        },
        "arms": {
            "C": {
                "primary_count": 3,
                "reserve_count": 3,
                "cell_counts": {"math": 2, "code": 1},
                "lane_counts": {"agentless-swe": 0},
                "bucket_floors": {},
                "non_agentic_bucket_floors": {},
                "lane_bucket_floors": {},
                "count_proof_sha256": "5" * 64,
            }
        },
    }
    manifest["root_sha256"] = sha256(_canonical(manifest).encode()).hexdigest()
    manifest_path = root / "SELECTION_MANIFEST.json"
    manifest_path.write_text(_canonical(manifest) + "\n")
    return manifest_path, sha256(manifest_path.read_bytes()).hexdigest()


def test_production_promotion_streams_authenticated_task7_records(tmp_path: Path) -> None:
    manifest, manifest_sha256 = _write_published_view(tmp_path)
    identity = replace(IDENTITY, source_selection_sha256=manifest_sha256)
    view = load_prompt_view(manifest, expected_manifest_sha256=manifest_sha256, arm="C")
    attempts_root = tmp_path / "attempts/replicas/0"
    attempts_root.mkdir(parents=True)
    for replica in range(4):
        replica_root = tmp_path / f"attempts/replicas/{replica}"
        replica_root.mkdir(parents=True, exist_ok=True)
        (replica_root / f"gpu-utilization-replica-{replica}.json").write_text(
            _canonical({"replica": replica, "gpu_utilization": [f"{replica}, 1"]}) + "\n"
        )
    attempts = [
        _attempt(view.primary_rows[0], identity=identity),
        _attempt(view.primary_rows[1], identity=identity),
        _attempt(view.primary_rows[2], identity=identity),
    ]
    (attempts_root / "attempts-job-0.jsonl").write_text(
        "".join(_canonical(asdict(attempt)) + "\n" for attempt in attempts)
    )

    corpus = promote_response_artifacts(
        view,
        tmp_path / "attempts",
        identity,
        tmp_path / "promoted",
    )

    assert len(corpus.records) == 3
    assert getattr(corpus.records, "resident_record_count") == 0
    assert getattr(corpus.records, "_path").name == "response-index.sqlite3"
    assert all(record.selection_sha256 == "3" * 64 for record in corpus.records)
    assert all(record.paired_cd_sha256 == "4" * 64 for record in corpus.records)
    assert all(record.generation_identity_sha256 == identity.sha256 for record in corpus.records)
    assert (tmp_path / "promoted/PROMOTION.json").is_file()
