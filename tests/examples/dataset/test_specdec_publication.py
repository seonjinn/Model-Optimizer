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

"""Immutable SpecDec corpus publication contracts."""

from __future__ import annotations

import hashlib
import importlib
import json
import sqlite3
import sys
import tempfile
import tracemalloc
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

sys.path.insert(0, str(Path(__file__).parents[3] / "examples/dataset"))
publication = importlib.import_module("specdec_publication")


def _canonical(value: object) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode()


def test_ptv2_lineage_flows_from_source_to_selection_then_tokenization() -> None:
    """A genuine upstream source receipt need not contain downstream selection state."""

    def digest(character: str) -> str:
        return character * 64

    selection = {
        "selection_sha256": digest("1"),
        "source_inventory_sha256": digest("2"),
        "baseline_receipt_sha256": digest("3"),
        "held_out_receipt_sha256": digest("4"),
    }
    response = {"selection_sha256": digest("1"), "source_response_root_sha256": digest("5")}
    tokenized = {
        **response,
        "database_sha256": digest("6"),
        "tokenizer_sha256": digest("7"),
        "chat_template_sha256": digest("8"),
        "assistant_loss_target_sha256": digest("9"),
    }
    exposure = {
        **response,
        "tokenized_sha256": digest("6"),
        "tokenizer_sha256": digest("7"),
        "chat_template_sha256": digest("8"),
        "assistant_loss_target_sha256": digest("9"),
    }
    payloads = {
        "source": {"source_manifest_sha256": digest("2")},
        "selection": selection,
        "response": response,
        "tokenized": tokenized,
        "exposure": exposure,
        "rejection": {"selection_sha256": digest("1")},
    }

    publication._reconcile_ptv2_role_lineage(payloads)
    payloads["exposure"] = {**exposure, "tokenized_sha256": digest("a")}
    with pytest.raises(publication.PublicationError, match="tokenized root"):
        publication._reconcile_ptv2_role_lineage(payloads)


def _receipt(root: Path, role: str, records: list[dict[str, Any]]) -> publication.InputArtifact:
    root.mkdir(parents=True, exist_ok=True)
    payload_path = root / f"{role}.jsonl"
    payload_path.write_bytes(b"".join(_canonical(record) for record in records))
    body = {
        "role": role,
        "files": [
            {
                "path": payload_path.name,
                "bytes": payload_path.stat().st_size,
                "sha256": hashlib.sha256(payload_path.read_bytes()).hexdigest(),
            }
        ],
    }
    body["receipt_sha256"] = hashlib.sha256(_canonical(body)).hexdigest()
    receipt_path = root / f"{role}.json"
    receipt_path.write_bytes(_canonical(body))
    return publication.InputArtifact(
        role=role,
        receipt_path=receipt_path,
        receipt_sha256=hashlib.sha256(receipt_path.read_bytes()).hexdigest(),
    )


def _rows(count: int = 5) -> list[dict[str, Any]]:
    return [
        {
            "prompt_uuid": f"{index:064x}",
            "domain": "math",
            "lane": "target-synthesis",
            "context_bucket": "le4k",
            "input_ids": [index, index + 1],
            "loss_mask": [False, True],
            "assistant_tokens": 1,
            "rejection_reason": None,
        }
        for index in range(count)
    ]


def _bundle(root: Path, *, count: int = 5) -> publication.CorpusBundle:
    rows = _rows(count)
    artifacts = tuple(_receipt(root, role, rows[:1]) for role in publication.REQUIRED_ROLES)
    return publication.CorpusBundle(
        artifacts=artifacts,
        rows=lambda: iter(rows),
        prompt_count=count,
        assistant_token_count=count,
        quarantine_count=0,
        selection_manifest_sha256=artifacts[1].receipt_sha256,
        artifact_source_commit="a" * 40,
    )


def test_publish_bundle_normalizes_apfs_var_alias() -> None:
    """Resolved descendants remain relative to an unresolved macOS /var publication root."""
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        bundle = _bundle(root / "inputs", count=1)

        receipt = publication.publish_bundle(bundle, root / "published", "apfs-alias")

        assert receipt.published_path == str(root / "published")


def test_ptv2_selection_receipt_authenticates_policy_index_and_occurrence_shards(
    tmp_path: Path,
) -> None:
    """A PTV2 selection adapter must carry every disk-backed selection input into publication."""
    policy = tmp_path / "policy.yaml"
    index = tmp_path / "ptv2-study-index.sqlite3"
    shard = tmp_path / "a-prefix.jsonl"
    policy.write_bytes(b"seed: 20260822\nstrategy: B-balanced\n")
    connection = sqlite3.connect(index)
    connection.execute(
        "CREATE TABLE occurrences(ordinal INTEGER,prompt_uuid TEXT,source_identity_sha256 TEXT,"
        "source_row INTEGER,cell TEXT,reuse_index INTEGER,conversation_sha256 TEXT,"
        "assistant_response_sha256 TEXT,strategy TEXT)"
    )
    connection.execute(
        "CREATE TABLE source_rows(source_identity_sha256 TEXT,source_row INTEGER,cell TEXT,"
        "language TEXT,canonical_conversation TEXT,assistant_response TEXT)"
    )
    conversation = "conversation"
    response = "response"
    occurrence = (
        0,
        "a" * 64,
        "b" * 64,
        7,
        "math",
        0,
        hashlib.sha256(conversation.encode()).hexdigest(),
        hashlib.sha256(response.encode()).hexdigest(),
    )
    connection.execute(
        "INSERT INTO source_rows VALUES(?,?,?,?,?,?)",
        ("b" * 64, 7, "math", "", conversation, response),
    )
    connection.execute(
        "INSERT INTO occurrences VALUES(?,?,?,?,?,?,?,?,?)", (*occurrence, "B-balanced")
    )
    connection.commit()
    connection.close()
    shard.write_bytes(_canonical(list(occurrence)))
    ordered = hashlib.sha256(
        json.dumps(list(occurrence), sort_keys=True, separators=(",", ":")).encode() + b"\n"
    ).hexdigest()

    def descriptor(path: Path) -> dict[str, Any]:
        return {
            "path": path.name,
            "bytes": path.stat().st_size,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }

    trust_roots = {
        "source_inventory_sha256": "3" * 64,
        "held_out_receipt_sha256": "5" * 64,
    }
    selection_identity = {
        "strategy": "B-balanced",
        "policy_sha256": hashlib.sha256(
            publication._identity_json({"seed": 20260822, "strategy": "B-balanced"})
        ).hexdigest(),
        "occurrence_count": 1,
        "unique_prompt_count": 1,
        "cell_occurrence_counts": {
            "math": 1,
            "code": 0,
            "stem": 0,
            "chat": 0,
            "multilingual": 0,
        },
        "multilingual_occurrence_counts": {},
        "repair_complement_counts": {},
        "uuid_multiplicity_histogram": {1: 1},
        "source_occurrence_multiplicity_histogram": {1: 1},
        "ordered_occurrences_sha256": ordered,
        "ordered_prompt_uuids_sha256": hashlib.sha256(_canonical("a" * 64)).hexdigest(),
        "source_response_root_sha256": hashlib.sha256(
            _canonical(["b" * 64, 7, occurrence[6], occurrence[7]])
        ).hexdigest(),
        "occurrence_multiplicity_sha256": hashlib.sha256(
            _canonical(["a" * 64, "b" * 64, 7, 1])
        ).hexdigest(),
        "trust_root_sha256": hashlib.sha256(publication._identity_json(trust_roots)).hexdigest(),
    }
    payload = {
        "schema_version": 3,
        "selection_sha256": hashlib.sha256(
            publication._identity_json(selection_identity)
        ).hexdigest(),
        "selection_identity": selection_identity,
        "policy_sha256": hashlib.sha256(
            publication._identity_json({"seed": 20260822, "strategy": "B-balanced"})
        ).hexdigest(),
        "policy_file_sha256": hashlib.sha256(policy.read_bytes()).hexdigest(),
        "source_inventory_sha256": "3" * 64,
        "baseline_receipt_sha256": "4" * 64,
        "held_out_receipt_sha256": "5" * 64,
        "trust_roots": trust_roots,
        "strategy": "B-balanced",
        "occurrence_count": 1,
        "ordered_occurrences_sha256": ordered,
        "shard_semantic_sha256": ordered,
        "policy": descriptor(policy),
        "index": descriptor(index),
        "shards": [descriptor(shard)],
    }
    payload["root_sha256"] = hashlib.sha256(publication._identity_json(payload)).hexdigest()

    assert publication._role_file_descriptors("selection", payload) == [
        descriptor(shard),
        descriptor(index),
        descriptor(policy),
    ]
    publication._validate_ptv2_selection_policy(
        payload,
        [
            (shard.name, shard, shard.stat().st_size, descriptor(shard)["sha256"]),
            (index.name, index, index.stat().st_size, descriptor(index)["sha256"]),
            (policy.name, policy, policy.stat().st_size, descriptor(policy)["sha256"]),
        ],
    )

    incomplete = dict(payload)
    incomplete_identity = dict(selection_identity)
    incomplete_identity.pop("unique_prompt_count")
    incomplete["selection_identity"] = incomplete_identity
    incomplete["selection_sha256"] = hashlib.sha256(
        publication._identity_json(incomplete_identity)
    ).hexdigest()
    with pytest.raises(publication.PublicationError, match="exact schema"):
        publication._validate_ptv2_selection_policy(
            incomplete,
            [
                (shard.name, shard, shard.stat().st_size, descriptor(shard)["sha256"]),
                (index.name, index, index.stat().st_size, descriptor(index)["sha256"]),
                (policy.name, policy, policy.stat().st_size, descriptor(policy)["sha256"]),
            ],
        )

    forged_values = {
        "strategy": "A-repair",
        "policy_sha256": "9" * 64,
        "occurrence_count": 2,
        "unique_prompt_count": 2,
        "cell_occurrence_counts": {"math": 2},
        "multilingual_occurrence_counts": {"de": 1},
        "repair_complement_counts": {"stem": 1},
        "uuid_multiplicity_histogram": {2: 1},
        "source_occurrence_multiplicity_histogram": {2: 1},
        "ordered_occurrences_sha256": "9" * 64,
        "ordered_prompt_uuids_sha256": "9" * 64,
        "source_response_root_sha256": "9" * 64,
        "occurrence_multiplicity_sha256": "9" * 64,
        "trust_root_sha256": "9" * 64,
    }
    for field, forged_value in forged_values.items():
        forged = dict(payload)
        forged_identity = dict(selection_identity)
        forged_identity[field] = forged_value
        forged["selection_identity"] = forged_identity
        forged["selection_sha256"] = hashlib.sha256(
            publication._identity_json(forged_identity)
        ).hexdigest()
        with pytest.raises(publication.PublicationError, match="index semantics"):
            publication._validate_ptv2_selection_policy(
                forged,
                [
                    (shard.name, shard, shard.stat().st_size, descriptor(shard)["sha256"]),
                    (index.name, index, index.stat().st_size, descriptor(index)["sha256"]),
                    (policy.name, policy, policy.stat().st_size, descriptor(policy)["sha256"]),
                ],
            )

    connection = sqlite3.connect(index)
    connection.execute("UPDATE occurrences SET source_row=8")
    connection.commit()
    connection.close()
    with pytest.raises(publication.PublicationError, match="index semantics"):
        publication._validate_ptv2_selection_policy(
            payload,
            [
                (shard.name, shard, shard.stat().st_size, descriptor(shard)["sha256"]),
                (index.name, index, index.stat().st_size, descriptor(index)["sha256"]),
                (policy.name, policy, policy.stat().st_size, descriptor(policy)["sha256"]),
            ],
        )
    connection = sqlite3.connect(index)
    connection.execute("UPDATE occurrences SET source_row=7")
    connection.commit()
    connection.close()

    semantic_mismatch = dict(payload)
    semantic_mismatch["policy_sha256"] = "0" * 64
    semantic_mismatch["root_sha256"] = hashlib.sha256(
        publication._identity_json(
            {key: value for key, value in semantic_mismatch.items() if key != "root_sha256"}
        )
    ).hexdigest()
    with pytest.raises(publication.PublicationError, match="semantic policy"):
        publication._validate_ptv2_selection_policy(
            semantic_mismatch,
            [
                (shard.name, shard, shard.stat().st_size, descriptor(shard)["sha256"]),
                (index.name, index, index.stat().st_size, descriptor(index)["sha256"]),
                (policy.name, policy, policy.stat().st_size, descriptor(policy)["sha256"]),
            ],
        )

    malformed = dict(payload)
    malformed["selection_sha256"] = "not-a-digest"
    malformed["unknown"] = True
    malformed["root_sha256"] = hashlib.sha256(
        publication._identity_json(
            {key: value for key, value in malformed.items() if key != "root_sha256"}
        )
    ).hexdigest()
    with pytest.raises(publication.PublicationError, match="schema is incomplete"):
        publication._role_file_descriptors("selection", malformed)


def test_ptv2_schema_v3_receipts_publish_as_a_complete_bundle(tmp_path: Path) -> None:
    """Task 8 accepts genuine Task 9-shaped source, selection, token, and exposure roots."""
    base = _bundle(tmp_path / "base")

    def digest(value: bytes) -> str:
        return hashlib.sha256(value).hexdigest()

    def artifact(root: Path, role: str, payload: dict[str, Any]) -> publication.InputArtifact:
        path = root / f"{role}.json"
        path.write_bytes(_canonical(payload))
        return publication.InputArtifact(role, path, digest(path.read_bytes()))

    source_root = tmp_path / "source"
    source_root.mkdir()
    source_file = source_root / "source.jsonl"
    source_file.write_bytes(b"{}\n")
    source_root_sha = "1" * 64
    source = artifact(
        source_root,
        "source",
        {
            "schema_version": 1,
            "name": "ptv2",
            "source_manifest_sha256": source_root_sha,
            "complete": True,
            "raw_counts": {"math": 1},
            "files": [
                {
                    "repository_id": "owner/repo",
                    "configuration": "default",
                    "split": "math",
                    "revision": "a" * 40,
                    "license_expression": "Apache-2.0",
                    "approved_use": True,
                    "cell": "math",
                    "lane": "target-synthesis",
                    "source_path": source_file.name,
                    "staged_path": source_file.name,
                    "bytes": source_file.stat().st_size,
                    "sha256": digest(source_file.read_bytes()),
                }
            ],
        },
    )
    selection_root = tmp_path / "selection"
    selection_root.mkdir()
    policy = selection_root / "policy.yaml"
    policy.write_bytes(b"seed: 1\nstrategy: B-balanced\n")
    index = selection_root / "selection.sqlite3"
    connection = sqlite3.connect(index)
    connection.execute(
        "CREATE TABLE occurrences(ordinal INTEGER,prompt_uuid TEXT,source_identity_sha256 TEXT,"
        "source_row INTEGER,cell TEXT,reuse_index INTEGER,conversation_sha256 TEXT,"
        "assistant_response_sha256 TEXT,strategy TEXT)"
    )
    connection.execute(
        "CREATE TABLE source_rows(source_identity_sha256 TEXT,source_row INTEGER,cell TEXT,"
        "language TEXT,canonical_conversation TEXT,assistant_response TEXT)"
    )
    conversation = "conversation"
    response = "response"
    occurrence = (
        0,
        "a" * 64,
        "b" * 64,
        0,
        "math",
        0,
        digest(conversation.encode()),
        digest(response.encode()),
    )
    connection.execute(
        "INSERT INTO source_rows VALUES(?,?,?,?,?,?)",
        ("b" * 64, 0, "math", "", conversation, response),
    )
    connection.execute(
        "INSERT INTO occurrences VALUES(?,?,?,?,?,?,?,?,?)", (*occurrence, "B-balanced")
    )
    connection.commit()
    connection.close()
    shard = selection_root / "occurrences.jsonl"
    shard.write_bytes(_canonical(list(occurrence)))

    def descriptor(path: Path) -> dict[str, Any]:
        return {
            "path": path.name,
            "bytes": path.stat().st_size,
            "sha256": digest(path.read_bytes()),
        }

    ordered = digest(
        json.dumps(list(occurrence), sort_keys=True, separators=(",", ":")).encode() + b"\n"
    )
    policy_sha = digest(publication._identity_json({"seed": 1, "strategy": "B-balanced"}))
    trust_roots = {
        "source_inventory_sha256": source_root_sha,
        "held_out_receipt_sha256": "4" * 64,
    }
    selection_identity = {
        "strategy": "B-balanced",
        "policy_sha256": policy_sha,
        "occurrence_count": 1,
        "unique_prompt_count": 1,
        "cell_occurrence_counts": {
            "math": 1,
            "code": 0,
            "stem": 0,
            "chat": 0,
            "multilingual": 0,
        },
        "multilingual_occurrence_counts": {},
        "repair_complement_counts": {},
        "uuid_multiplicity_histogram": {1: 1},
        "source_occurrence_multiplicity_histogram": {1: 1},
        "ordered_occurrences_sha256": ordered,
        "ordered_prompt_uuids_sha256": digest(_canonical("a" * 64)),
        "source_response_root_sha256": digest(
            _canonical(["b" * 64, 0, occurrence[6], occurrence[7]])
        ),
        "occurrence_multiplicity_sha256": digest(_canonical(["a" * 64, "b" * 64, 0, 1])),
        "trust_root_sha256": digest(publication._identity_json(trust_roots)),
    }
    selection_body = {
        "schema_version": 3,
        "selection_sha256": digest(publication._identity_json(selection_identity)),
        "selection_identity": selection_identity,
        "policy_sha256": policy_sha,
        "policy_file_sha256": digest(policy.read_bytes()),
        "source_inventory_sha256": source_root_sha,
        "baseline_receipt_sha256": "3" * 64,
        "held_out_receipt_sha256": "4" * 64,
        "trust_roots": trust_roots,
        "strategy": "B-balanced",
        "occurrence_count": 1,
        "ordered_occurrences_sha256": ordered,
        "shard_semantic_sha256": ordered,
        "policy": descriptor(policy),
        "index": descriptor(index),
        "shards": [descriptor(shard)],
    }
    selection_body["root_sha256"] = digest(publication._identity_json(selection_body))
    selection = artifact(selection_root, "selection", selection_body)
    response_root = "5" * 64
    response = _receipt(tmp_path / "response", "response", _rows(1))
    response_payload = {
        key: value
        for key, value in json.loads(response.receipt_path.read_bytes()).items()
        if key != "receipt_sha256"
    } | {
        "selection_sha256": selection_body["selection_sha256"],
        "source_response_root_sha256": response_root,
    }
    response = artifact(response.receipt_path.parent, "response", response_payload)
    token_root = tmp_path / "token"
    token_root.mkdir()
    database = token_root / "records.sqlite3"
    database.write_bytes(b"sqlite")
    tokenized = artifact(
        token_root,
        "tokenized",
        {
            "schema_version": 1,
            "strategy": "B-balanced",
            "occurrence_count": 1,
            "assistant_tokens": 1,
            "database_path": database.name,
            "database_bytes": database.stat().st_size,
            "database_sha256": digest(database.read_bytes()),
            "selection_sha256": selection_body["selection_sha256"],
            "source_response_root_sha256": response_root,
            "tokenizer_sha256": "6" * 64,
            "chat_template_sha256": "7" * 64,
            "assistant_loss_target_sha256": "8" * 64,
        },
    )
    exposure_root = tmp_path / "exposure"
    exposure_root.mkdir()
    records = exposure_root / "records.jsonl"
    records.write_bytes(_canonical(_rows(1)[0]))
    exposure = artifact(
        exposure_root,
        "exposure",
        {
            "schema_version": 1,
            "strategy": "B-balanced",
            "target_assistant_tokens": 1,
            "records_path": records.name,
            "records_bytes": records.stat().st_size,
            "records_sha256": digest(records.read_bytes()),
            "row_count": 1,
            "tokenized_sha256": digest(database.read_bytes()),
            "selection_sha256": selection_body["selection_sha256"],
            "source_response_root_sha256": response_root,
            "tokenizer_sha256": "6" * 64,
            "chat_template_sha256": "7" * 64,
            "assistant_loss_target_sha256": "8" * 64,
        },
    )
    rejection = _receipt(tmp_path / "rejection", "rejection", _rows(1))
    rejection_payload = {
        key: value
        for key, value in json.loads(rejection.receipt_path.read_bytes()).items()
        if key != "receipt_sha256"
    } | {"selection_sha256": selection_body["selection_sha256"]}
    rejection = artifact(rejection.receipt_path.parent, "rejection", rejection_payload)
    bundle = publication.CorpusBundle(
        artifacts=(source, selection, response, tokenized, exposure, rejection),
        rows=base.rows,
        prompt_count=base.prompt_count,
        assistant_token_count=base.assistant_token_count,
        quarantine_count=base.quarantine_count,
        selection_manifest_sha256=selection.receipt_sha256,
        artifact_source_commit="a" * 40,
    )

    published = publication.publish_bundle(bundle, tmp_path / "published", "ptv2-v3")
    assert Path(published.published_path).is_dir()


@pytest.mark.parametrize(
    "phase",
    [
        "shard_open",
        "shard_temporary_fsynced",
        "shard_closed",
        "manifest",
        "directory_fsync",
        "rename",
    ],
)
def test_interruption_preserves_partial_and_retry_resumes_verified_shards(
    tmp_path: Path, phase: str
) -> None:
    bundle = _bundle(tmp_path / "inputs")
    destination = tmp_path / "published"
    seen: list[str] = []

    def interrupt(current: str) -> None:
        seen.append(current)
        if current == phase:
            raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt) as captured:
        publication.publish_bundle(
            bundle, destination, "job-42", rows_per_shard=2, _phase_hook=interrupt
        )
    assert not destination.exists()
    state = publication.publication_recovery_state(captured.value)
    assert state.partial_path.exists()
    assert state.destination_observation.status == "absent"

    receipt = publication.publish_bundle(bundle, destination, "job-42", rows_per_shard=2)
    assert receipt.file_count >= 9
    assert destination.is_dir()
    assert (
        sum(pq.read_metadata(path).num_rows for path in destination.glob("shards/*.parquet")) == 5
    )
    assert "rename" in seen or phase != "rename"


def test_stale_partial_is_quarantined_and_collision_is_fail_closed(tmp_path: Path) -> None:
    bundle = _bundle(tmp_path / "first")
    destination = tmp_path / "published"

    def interrupt(phase: str) -> None:
        if phase == "shard_closed":
            raise RuntimeError("stop")

    with pytest.raises(RuntimeError):
        publication.publish_bundle(
            bundle, destination, "same-job", rows_per_shard=2, _phase_hook=interrupt
        )
    stale_partial = next(tmp_path.glob(".published.partial-same-job"))
    changed = _bundle(tmp_path / "second", count=3)
    receipt = publication.publish_bundle(changed, destination, "same-job", rows_per_shard=2)
    quarantines = list(tmp_path.glob(".published.quarantine-same-job-*"))
    assert receipt.artifact_id
    assert quarantines and not stale_partial.exists()

    with pytest.raises(FileExistsError, match="immutable publication destination"):
        publication.publish_bundle(changed, destination, "other-job", rows_per_shard=2)


def test_resumed_shard_rejects_forged_state_and_changed_parquet_semantics(tmp_path: Path) -> None:
    """A mutable state file cannot bless Parquet rows that differ from the current batch."""
    bundle = _bundle(tmp_path / "inputs")
    destination = tmp_path / "published"

    def interrupt(phase: str) -> None:
        if phase == "shard_closed":
            raise RuntimeError("stop")

    with pytest.raises(RuntimeError):
        publication.publish_bundle(
            bundle, destination, "resume", rows_per_shard=2, _phase_hook=interrupt
        )
    partial = tmp_path / ".published.partial-resume"
    shard = partial / "shards/part-000000.parquet"
    changed_rows = [_rows(2)[0] | {"prompt_uuid": "f" * 64}, _rows(2)[1]]
    normalized = [publication._normalize_row(row) for row in changed_rows]
    pq.write_table(pa.Table.from_pylist(normalized, schema=publication._SCHEMA), shard)
    state_path = partial / "SHARD_STATE-000000.json"
    state = json.loads(state_path.read_bytes())
    state["shard"]["bytes"] = shard.stat().st_size
    state["shard"]["sha256"] = hashlib.sha256(shard.read_bytes()).hexdigest()
    forged_rows_sha256 = hashlib.sha256(b"".join(_canonical(row) for row in normalized)).hexdigest()
    state["canonical_rows_sha256"] = forged_rows_sha256
    state["shard"]["canonical_rows_sha256"] = forged_rows_sha256
    state_path.write_bytes(_canonical(state))

    with pytest.raises(publication.PublicationError, match=r"resumed shard.*current batch"):
        publication.publish_bundle(bundle, destination, "resume", rows_per_shard=2)
    assert not destination.exists()


def test_interruption_after_parquet_install_before_state_is_resumable(tmp_path: Path) -> None:
    """An authenticated orphan shard is adopted after interruption before its state write."""
    bundle = _bundle(tmp_path / "inputs")
    destination = tmp_path / "published"

    def interrupt(phase: str) -> None:
        if phase == "shard_installed":
            raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        publication.publish_bundle(
            bundle, destination, "orphan", rows_per_shard=2, _phase_hook=interrupt
        )
    partial = tmp_path / ".published.partial-orphan"
    assert (partial / "shards/part-000000.parquet").is_file()
    assert not (partial / "SHARD_STATE-000000.json").exists()

    publication.publish_bundle(bundle, destination, "orphan", rows_per_shard=2)
    assert destination.is_dir()


def test_resume_quarantines_nested_symlink_before_writing(tmp_path: Path) -> None:
    """Resume validates the complete partial tree no-follow before copying or sharding."""
    bundle = _bundle(tmp_path / "inputs")
    destination = tmp_path / "published"

    def interrupt(phase: str) -> None:
        if phase == "shard_closed":
            raise RuntimeError("stop")

    with pytest.raises(RuntimeError):
        publication.publish_bundle(
            bundle, destination, "symlink", rows_per_shard=2, _phase_hook=interrupt
        )
    partial = tmp_path / ".published.partial-symlink"
    preserved_inputs = tmp_path / "preserved-inputs"
    (partial / "inputs").rename(preserved_inputs)
    sentinel = tmp_path / "sentinel"
    sentinel.mkdir()
    (sentinel / "keep").write_bytes(b"keep")
    (partial / "inputs").symlink_to(sentinel, target_is_directory=True)

    publication.publish_bundle(bundle, destination, "symlink", rows_per_shard=2)
    assert (sentinel / "keep").read_bytes() == b"keep"
    quarantines = list(tmp_path.glob(".published.quarantine-symlink-*"))
    assert quarantines and (quarantines[0] / "inputs").is_symlink()


@pytest.mark.parametrize("mutation", ["missing", "changed", "symlink"])
def test_input_receipts_reject_corrupt_or_unlisted_files(tmp_path: Path, mutation: str) -> None:
    bundle = _bundle(tmp_path / "inputs")
    artifact = bundle.artifacts[0]
    declared = artifact.receipt_path.parent / "source.jsonl"
    if mutation == "missing":
        declared.unlink()
    elif mutation == "changed":
        declared.write_bytes(b"changed")
    else:
        target = tmp_path / "target"
        target.write_bytes(declared.read_bytes())
        declared.unlink()
        declared.symlink_to(target)

    with pytest.raises(publication.PublicationError):
        publication.publish_bundle(bundle, tmp_path / "published", "job")


def test_complete_role_set_and_count_reconciliation_are_required(tmp_path: Path) -> None:
    bundle = _bundle(tmp_path / "inputs")
    incomplete = publication.CorpusBundle(
        artifacts=tuple(artifact for artifact in bundle.artifacts if artifact.role != "selection"),
        rows=bundle.rows,
        prompt_count=bundle.prompt_count,
        assistant_token_count=bundle.assistant_token_count,
        quarantine_count=bundle.quarantine_count,
        selection_manifest_sha256=bundle.selection_manifest_sha256,
        artifact_source_commit=bundle.artifact_source_commit,
    )
    with pytest.raises(publication.PublicationError, match="artifact roles"):
        publication.publish_bundle(incomplete, tmp_path / "published", "job")

    wrong_count = publication.CorpusBundle(
        artifacts=bundle.artifacts,
        rows=bundle.rows,
        prompt_count=99,
        assistant_token_count=bundle.assistant_token_count,
        quarantine_count=bundle.quarantine_count,
        selection_manifest_sha256=bundle.selection_manifest_sha256,
        artifact_source_commit=bundle.artifact_source_commit,
    )
    with pytest.raises(publication.PublicationError, match="prompt count"):
        publication.publish_bundle(wrong_count, tmp_path / "published", "job-2")


def test_task5_selection_and_task7_tokenized_exposure_receipts_publish_together(
    tmp_path: Path,
) -> None:
    """Production Task 5/7 receipt schemas authenticate their native declared artifacts."""
    base = _bundle(tmp_path / "generic")
    source_root = tmp_path / "source"
    source_file = source_root / "sources/repo/data.jsonl"
    source_file.parent.mkdir(parents=True)
    source_file.write_bytes(_canonical({"messages": []}))
    source_body = {
        "schema_version": 1,
        "name": "ptv23",
        "source_manifest_sha256": "1" * 64,
        "complete": True,
        "raw_counts": {"math": 1},
        "files": [
            {
                "repository_id": "owner/repo",
                "configuration": "default",
                "split": "train",
                "revision": "a" * 40,
                "license_expression": "Apache-2.0",
                "approved_use": True,
                "cell": "math",
                "lane": "target-synthesis",
                "source_path": "data.jsonl",
                "staged_path": "sources/repo/data.jsonl",
                "bytes": source_file.stat().st_size,
                "sha256": hashlib.sha256(source_file.read_bytes()).hexdigest(),
            }
        ],
    }
    source_receipt = source_root / "SOURCE_INVENTORY.json"
    source_receipt.write_text(json.dumps(source_body, indent=2, sort_keys=True) + "\n")
    source_artifact = publication.InputArtifact(
        "source",
        source_receipt,
        hashlib.sha256(source_receipt.read_bytes()).hexdigest(),
    )
    selection_root = tmp_path / "selection"
    (selection_root / "shards").mkdir(parents=True)
    selection_shard = selection_root / "shards/rows-000001.jsonl"
    selection_shard.write_bytes(_canonical({"prompt_uuid": "1" * 64}))
    selection_index = selection_root / "selection-index.sqlite3"
    selection_index.write_bytes(b"sqlite-index")
    selection_body = {
        "schema_version": 2,
        "selection_sha256": "2" * 64,
        "paired_cd_sha256": "3" * 64,
        "identity": {"policy_sha256": "4" * 64},
        "arms": {},
        "row_count": 1,
        "rows_per_shard": 1,
        "shards": [
            {
                "path": "shards/rows-000001.jsonl",
                "row_count": 1,
                "byte_count": selection_shard.stat().st_size,
                "sha256": hashlib.sha256(selection_shard.read_bytes()).hexdigest(),
            }
        ],
        "index": {
            "path": selection_index.name,
            "sha256": hashlib.sha256(selection_index.read_bytes()).hexdigest(),
        },
    }
    selection_body["root_sha256"] = hashlib.sha256(
        json.dumps(selection_body, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    selection_receipt = selection_root / "SELECTION_MANIFEST.json"
    selection_receipt.write_bytes(_canonical(selection_body))
    selection_artifact = publication.InputArtifact(
        "selection",
        selection_receipt,
        hashlib.sha256(selection_receipt.read_bytes()).hexdigest(),
    )

    task7_root = tmp_path / "task7"
    task7_root.mkdir()
    database = task7_root / "tokenized.sqlite3"
    database.write_bytes(b"sqlite-tokenized")
    tokenized_body = {
        "schema_version": 1,
        "arm": "B",
        "selection_sha256": "2" * 64,
        "record_count": 1,
        "total_assistant_tokens": 1,
        "database_path": str(database),
        "database_bytes": database.stat().st_size,
        "database_sha256": hashlib.sha256(database.read_bytes()).hexdigest(),
        "resume_fingerprint": "5" * 64,
    }
    tokenized_body["receipt_sha256"] = hashlib.sha256(
        json.dumps(tokenized_body, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    tokenized_receipt = task7_root / "TOKENIZED.json"
    tokenized_receipt.write_bytes(_canonical(tokenized_body))
    tokenized_artifact = publication.InputArtifact(
        "tokenized",
        tokenized_receipt,
        hashlib.sha256(tokenized_receipt.read_bytes()).hexdigest(),
    )

    records = task7_root / "one-pass.jsonl"
    records.write_bytes(_canonical(_rows(1)[0]))
    exposure_body = {
        "schema_version": 1,
        "name": "one-pass",
        "selection_sha256": "2" * 64,
        "assistant_tokens": 1,
        "row_count": 1,
        "records_path": str(records),
        "records_bytes": records.stat().st_size,
        "records_sha256": hashlib.sha256(records.read_bytes()).hexdigest(),
        "resume_fingerprint": "5" * 64,
    }
    exposure_body["receipt_sha256"] = hashlib.sha256(
        json.dumps(exposure_body, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    exposure_receipt = task7_root / "one-pass.json"
    exposure_receipt.write_bytes(_canonical(exposure_body))
    exposure_artifact = publication.InputArtifact(
        "exposure",
        exposure_receipt,
        hashlib.sha256(exposure_receipt.read_bytes()).hexdigest(),
    )
    replacements = {
        "source": source_artifact,
        "selection": selection_artifact,
        "tokenized": tokenized_artifact,
        "exposure": exposure_artifact,
    }
    artifacts = tuple(replacements.get(artifact.role, artifact) for artifact in base.artifacts)
    bundle = publication.CorpusBundle(
        artifacts=artifacts,
        rows=base.rows,
        prompt_count=base.prompt_count,
        assistant_token_count=base.assistant_token_count,
        quarantine_count=base.quarantine_count,
        selection_manifest_sha256=selection_artifact.receipt_sha256,
        artifact_source_commit=base.artifact_source_commit,
    )

    publication.publish_bundle(bundle, tmp_path / "published", "native", rows_per_shard=2)
    assert (tmp_path / "published/inputs/source/files/sources/repo/data.jsonl").is_file()
    assert (tmp_path / "published/inputs/selection/files/shards/rows-000001.jsonl").is_file()
    assert (tmp_path / "published/inputs/tokenized/files/tokenized.sqlite3").is_file()
    assert (tmp_path / "published/inputs/exposure/files/one-pass.jsonl").is_file()


def test_rename_races_preserve_every_observed_inode(tmp_path: Path, monkeypatch: Any) -> None:
    """An ambiguous rename never causes the publisher to delete either pathname."""
    bundle = _bundle(tmp_path / "inputs")
    destination = tmp_path / "published"
    preserved = tmp_path / "preserved-partial"
    rename = publication._rename_no_replace

    def racing_rename(source: Path, target: Path) -> None:
        if target != destination:
            rename(source, target)
            return
        source.rename(preserved)
        target.mkdir()
        (target / "sentinel").write_bytes(b"winner")
        raise FileExistsError(target)

    monkeypatch.setattr(publication, "_rename_no_replace", racing_rename)
    with pytest.raises(FileExistsError) as captured:
        publication.publish_bundle(bundle, destination, "race", rows_per_shard=2)
    state = publication.publication_recovery_state(captured.value)
    assert (destination / "sentinel").read_bytes() == b"winner"
    assert preserved.is_dir()
    assert state.partial_observation.status == "absent"
    assert state.destination_observation.status == "present"


def test_successful_but_raised_rename_and_parent_fsync_are_typed(
    tmp_path: Path, monkeypatch: Any
) -> None:
    """Post-rename ambiguity carries the installed inode and leaves it untouched."""
    bundle = _bundle(tmp_path / "inputs")
    destination = tmp_path / "published"
    rename = publication._rename_no_replace

    def ambiguous_rename(source: Path, target: Path) -> None:
        rename(source, target)
        if target == destination:
            raise RuntimeError("helper lost acknowledgement")

    monkeypatch.setattr(publication, "_rename_no_replace", ambiguous_rename)
    with pytest.raises(RuntimeError) as captured:
        publication.publish_bundle(bundle, destination, "ambiguous", rows_per_shard=2)
    state = publication.publication_recovery_state(captured.value)
    assert destination.is_dir()
    assert state.destination_observation.identity == state.expected_partial_identity

    second = tmp_path / "second"
    monkeypatch.setattr(publication, "_rename_no_replace", rename)
    fsync = publication._fsync_directory

    def failing_parent_fsync(path: Path) -> None:
        if path == tmp_path:
            raise OSError("durability unknown")
        fsync(path)

    monkeypatch.setattr(publication, "_fsync_directory", failing_parent_fsync)
    with pytest.raises(OSError) as fsync_captured:
        publication.publish_bundle(bundle, second, "fsync", rows_per_shard=2)
    fsync_state = publication.publication_recovery_state(fsync_captured.value)
    assert second.is_dir()
    assert fsync_state.phase is publication.PublicationPhase.PARENT_FSYNC
    assert fsync_state.destination_observation.identity == fsync_state.expected_partial_identity


def test_quarantine_acknowledgement_failure_records_new_path(
    tmp_path: Path, monkeypatch: Any
) -> None:
    """Recovery identifies a stale partial after quarantine rename succeeds ambiguously."""
    original = _bundle(tmp_path / "first")
    destination = tmp_path / "published"

    def interrupt(phase: str) -> None:
        if phase == "shard_closed":
            raise RuntimeError("stop")

    with pytest.raises(RuntimeError):
        publication.publish_bundle(
            original, destination, "quarantine", rows_per_shard=2, _phase_hook=interrupt
        )
    changed = _bundle(tmp_path / "second", count=3)
    rename = publication._rename_no_replace

    def ambiguous_quarantine(source: Path, target: Path) -> None:
        rename(source, target)
        raise OSError("quarantine acknowledgement lost")

    monkeypatch.setattr(publication, "_rename_no_replace", ambiguous_quarantine)
    with pytest.raises(OSError) as captured:
        publication.publish_bundle(changed, destination, "quarantine", rows_per_shard=2)
    state = publication.publication_recovery_state(captured.value)
    assert state.phase is publication.PublicationPhase.QUARANTINE
    assert state.quarantine_path is not None
    assert state.quarantine_observation is not None
    assert state.quarantine_observation.status == "present"
    assert state.quarantine_observation.identity == state.expected_partial_identity


def test_streaming_publication_has_bounded_memory_and_byte_idempotent_receipt(
    tmp_path: Path,
) -> None:
    count = 1_000
    base = _bundle(tmp_path / "inputs", count=1)

    def stream() -> Any:
        for index in range(count):
            yield _rows(1)[0] | {"prompt_uuid": f"{index:064x}"}

    bundle = publication.CorpusBundle(
        artifacts=base.artifacts,
        rows=stream,
        prompt_count=count,
        assistant_token_count=count,
        quarantine_count=0,
        selection_manifest_sha256=base.selection_manifest_sha256,
        artifact_source_commit=base.artifact_source_commit,
    )
    tracemalloc.start()
    first = publication.publish_bundle(bundle, tmp_path / "one", "one", rows_per_shard=127)
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    second = publication.publish_bundle(bundle, tmp_path / "two", "two", rows_per_shard=127)
    assert peak < 32 * 1024 * 1024
    assert first.artifact_id == second.artifact_id
    assert first.corpus_manifest_sha256 == second.corpus_manifest_sha256
    assert (tmp_path / "one/PUBLICATION.json").read_bytes() == (
        tmp_path / "two/PUBLICATION.json"
    ).read_bytes()
    assert json.loads((tmp_path / "one/PUBLICATION.json").read_bytes())[
        "artifact_source_commit"
    ] == ("a" * 40)
