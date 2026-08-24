# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Build the exact, source-ordered 200-step A-repair canary corpus."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sqlite3
import stat
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from common.specdec.qwen4b_b_atomic import atomic_publish_directory

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping

__all__ = [
    "A_COMPLEMENT_QUOTAS",
    "A_HISTORICAL_BOUNDARY",
    "A_HISTORICAL_QUOTA",
    "build_a_canary_occurrences",
    "materialize_a_canary",
    "resolve_worker_count",
]

A_HISTORICAL_BOUNDARY = 1_300_000
A_HISTORICAL_QUOTA = 66_560
A_COMPLEMENT_QUOTAS = {
    "stem": 10_240,
    "ja": 6_400,
    "es": 6_400,
    "fr": 6_400,
    "it": 6_400,
    "de": 0,
}
_A_TOTAL = A_HISTORICAL_QUOTA + sum(A_COMPLEMENT_QUOTAS.values())
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_COMMIT = re.compile(r"^[0-9a-f]{40}$")


@dataclass(frozen=True)
class Task8Shard:
    """One authenticated, source-ordered Task8 shard."""

    path: Path
    relative_path: str
    rows: int
    bytes: int
    sha256: str


@dataclass(frozen=True)
class Task8APublication:
    """Authenticated A-repair publication and its ordered shards."""

    root: Path
    publication_sha256: str
    corpus_manifest_sha256: str
    selection_receipt_sha256: str
    source_commit: str
    tokenizer_sha256: str
    chat_template_sha256: str
    selection_index_sha256: str
    tokenized_database_sha256: str
    shards: tuple[Task8Shard, ...]


@dataclass(frozen=True)
class ACanaryBuildArtifact:
    """Immutable A canary builder result."""

    root: Path
    output_path: Path
    receipt_path: Path
    output_sha256: str
    receipt_sha256: str


@dataclass(frozen=True)
class _WorkerTask:
    index: int
    start_ordinal: int
    shard: Task8Shard
    spool_path: Path


@dataclass(frozen=True)
class _WorkerResult:
    index: int
    start_ordinal: int
    row_count: int
    bytes: int
    sha256: str
    spool_path: str
    worker_pid: int


def resolve_worker_count(
    *,
    requested_workers: int | None,
    declared_shard_count: int,
    environ: Mapping[str, str] | None = None,
) -> tuple[int, int]:
    """Bound CPU parallelism by the Slurm allocation and shard count."""
    if not 1 <= declared_shard_count <= 201:
        raise ValueError("A builder requires between 1 and 201 declared shards")
    environment = os.environ if environ is None else environ
    raw_cpus = environment.get("SLURM_CPUS_PER_TASK")
    try:
        allocated = int(raw_cpus) if raw_cpus is not None else (os.cpu_count() or 1)
    except ValueError as error:
        raise ValueError("SLURM_CPUS_PER_TASK must be a positive integer") from error
    if allocated < 1:
        raise ValueError("A builder requires a positive CPU allocation")
    requested = allocated if requested_workers is None else requested_workers
    if requested < 1:
        raise ValueError("A builder requires at least one worker")
    return min(requested, allocated, declared_shard_count, 201), allocated


def build_a_canary_occurrences(
    rows: Iterable[Mapping[str, object]],
    *,
    historical_boundary: int = A_HISTORICAL_BOUNDARY,
    historical_quota: int = A_HISTORICAL_QUOTA,
    complement_quotas: Mapping[str, int] = A_COMPLEMENT_QUOTAS,
) -> list[dict[str, object]]:
    """Select historical prefix plus repair complement in global source order."""
    if historical_boundary < historical_quota or historical_quota < 1:
        raise ValueError("A historical boundary/quota is invalid")
    quotas = _validated_quotas(complement_quotas)
    selected: list[dict[str, object]] = []
    counts = dict.fromkeys(quotas, 0)
    for ordinal, raw in enumerate(rows):
        row = _validated_task8_row(raw)
        if ordinal < historical_boundary:
            if ordinal < historical_quota:
                selected.append(_streaming_entry(row))
            continue
        category = _complement_category(row)
        if category is None:
            raise ValueError("A repair complement contains C/D contamination")
        if category == "de":
            raise ValueError("A repair German complement is forbidden")
        if counts[category] < quotas[category]:
            selected.append(_streaming_entry(row))
            counts[category] += 1
    missing = {name: quotas[name] - counts[name] for name in quotas if counts[name] != quotas[name]}
    if missing:
        raise ValueError(f"A repair complement quota mismatch: {missing}")
    expected = historical_quota + sum(quotas.values())
    if len(selected) != expected:
        raise AssertionError(f"A canary selection must contain {expected} occurrences")
    return selected


def load_task8_a_publication(
    root: Path,
    *,
    expected_publication_sha256: str,
    expected_selection_receipt_sha256: str,
    source_commit: str,
    _expected_occurrences: int = 2_000_000,
    _expected_shards: int = 201,
    _expected_repair: Mapping[str, int] | None = None,
) -> Task8APublication:
    """Authenticate Task9 schema-v3 A-repair through the Task8 publication."""
    _require_digest(expected_publication_sha256, "publication")
    _require_digest(expected_selection_receipt_sha256, "selection receipt")
    if _COMMIT.fullmatch(source_commit) is None:
        raise ValueError("Task8 source commit must be exact")
    publication_raw = _stable_file_bytes(root / "PUBLICATION.json")
    if hashlib.sha256(publication_raw).hexdigest() != expected_publication_sha256:
        raise ValueError("Task8 publication SHA-256 mismatch")
    publication = _canonical_object(publication_raw, "Task8 publication")
    if (
        publication.get("schema") != "modelopt-specdec-publication-receipt-v1"
        or publication.get("artifact_source_commit") != source_commit
        or publication.get("selection_manifest_sha256") != expected_selection_receipt_sha256
    ):
        raise ValueError("Task8 A publication lineage mismatch")
    manifest_raw = _stable_file_bytes(root / "CORPUS_MANIFEST.json")
    manifest_sha = hashlib.sha256(manifest_raw).hexdigest()
    if manifest_sha != publication.get("corpus_manifest_sha256"):
        raise ValueError("Task8 corpus manifest SHA-256 mismatch")
    manifest = _canonical_object(manifest_raw, "Task8 corpus manifest")
    if (
        manifest.get("schema") != "modelopt-specdec-corpus-manifest-v1"
        or manifest.get("artifact_source_commit") != source_commit
        or manifest.get("selection_manifest_sha256") != expected_selection_receipt_sha256
    ):
        raise ValueError("Task8 corpus manifest lineage mismatch")
    selection_raw = _stable_file_bytes(root / "inputs/selection/receipt.json")
    if hashlib.sha256(selection_raw).hexdigest() != expected_selection_receipt_sha256:
        raise ValueError("Task9 A selection SHA-256 mismatch")
    selection = _canonical_object(selection_raw, "Task9 A selection")
    identity = selection.get("selection_identity")
    expected_repair = dict(_expected_repair) if _expected_repair is not None else {
        "stem": 200_000,
        "ja": 125_000,
        "es": 125_000,
        "fr": 125_000,
        "it": 125_000,
        "de": 0,
    }
    if (
        selection.get("schema_version") != 3
        or selection.get("strategy") != "A-repair"
        or not isinstance(identity, dict)
        or identity.get("strategy") != "A-repair"
        or identity.get("occurrence_count") != _expected_occurrences
        or identity.get("repair_complement_counts") != expected_repair
    ):
        raise ValueError("Task9 schema-v3 A selection semantics are invalid")
    descriptors = manifest.get("shards")
    files = manifest.get("files")
    if not isinstance(descriptors, list) or len(descriptors) != _expected_shards or not isinstance(files, list):
        raise ValueError(f"Task8 A publication must declare exactly {_expected_shards} shards")
    declared_files = {
        item.get("path"): (item.get("bytes"), item.get("sha256"))
        for item in files
        if isinstance(item, dict)
    }
    shards: list[Task8Shard] = []
    for item in descriptors:
        if not isinstance(item, dict):
            raise ValueError("Task8 shard descriptor must be an object")
        relative = _safe_relative(item.get("path"))
        size, rows, digest = item.get("bytes"), item.get("row_count"), item.get("sha256")
        if (
            not relative.startswith("shards/")
            or Path(relative).suffix not in {".jsonl", ".parquet"}
            or isinstance(size, bool)
            or not isinstance(size, int)
            or size < 0
            or isinstance(rows, bool)
            or not isinstance(rows, int)
            or rows < 0
            or not isinstance(digest, str)
            or _SHA256.fullmatch(digest) is None
            or declared_files.get(relative) != (size, digest)
        ):
            raise ValueError("Task8 shard descriptor is invalid")
        shards.append(Task8Shard(root / relative, relative, rows, size, digest))
    if sum(shard.rows for shard in shards) != _expected_occurrences:
        raise ValueError(f"Task8 A source must contain exactly {_expected_occurrences} occurrences")
    actual = set((root / "shards").glob("*"))
    if actual != {shard.path for shard in shards}:
        raise ValueError("Task8 A publication has missing or orphan shards")
    tokenizer_sha, template_sha, selection_index_sha, tokenized_sha = _replay_task9_task8(
        root,
        selection,
        tuple(shards),
        expected_occurrences=_expected_occurrences,
    )
    return Task8APublication(
        root,
        expected_publication_sha256,
        manifest_sha,
        expected_selection_receipt_sha256,
        source_commit,
        tokenizer_sha,
        template_sha,
        selection_index_sha,
        tokenized_sha,
        tuple(shards),
    )


def _replay_task9_task8(
    root: Path,
    selection: Mapping[str, Any],
    shards: tuple[Task8Shard, ...],
    *,
    expected_occurrences: int,
) -> tuple[str, str, str, str]:
    """Replay Task9 selection/tokenization against every Task8 row in exact ordinal order."""
    selection_copy_root = root / "inputs/selection/files"
    index_path, index_sha = _authenticated_declared_file(
        selection_copy_root, selection.get("index"), "Task9 selection index"
    )
    for descriptor in selection.get("shards", ()):
        _authenticated_declared_file(selection_copy_root, descriptor, "Task9 occurrence shard")
    selection_without_root = dict(selection)
    root_claim = selection_without_root.pop("root_sha256", None)
    if root_claim != _sha256_json(selection_without_root):
        raise ValueError("Task9 A selection root identity mismatch")

    tokenized_raw = _stable_file_bytes(root / "inputs/tokenized/receipt.json")
    tokenized_receipt = _canonical_object(tokenized_raw, "Task9 tokenized receipt")
    tokenized_claim = tokenized_receipt.pop("receipt_sha256", None)
    if tokenized_claim not in {
        _sha256_json(tokenized_receipt),
        hashlib.sha256((_canonical_json(tokenized_receipt) + "\n").encode()).hexdigest(),
    }:
        raise ValueError("Task9 tokenized receipt self identity mismatch")
    tokenized = tokenized_receipt.get("identity", tokenized_receipt)
    if not isinstance(tokenized, dict):
        raise ValueError("Task9 tokenized identity is invalid")
    database_descriptor = {
        "path": tokenized.get("database_path"),
        "bytes": tokenized.get("database_bytes"),
        "sha256": tokenized.get("database_sha256"),
    }
    tokenized_path, tokenized_sha = _authenticated_declared_file(
        root / "inputs/tokenized/files", database_descriptor, "Task9 tokenized database"
    )
    identity = selection.get("selection_identity")
    if (
        tokenized.get("schema_version") != 1
        or tokenized.get("strategy") != "A-repair"
        or tokenized.get("occurrence_count") != expected_occurrences
        or tokenized.get("selection_sha256") != selection.get("selection_sha256")
        or not isinstance(identity, dict)
        or tokenized.get("ordered_occurrences_sha256")
        != identity.get("ordered_occurrences_sha256")
        or tokenized.get("source_response_root_sha256")
        != identity.get("source_response_root_sha256")
    ):
        raise ValueError("Task9 tokenized lineage does not match A selection")
    tokenizer_sha = tokenized.get("tokenizer_sha256")
    template_sha = tokenized.get("chat_template_sha256")
    if not isinstance(tokenizer_sha, str) or not isinstance(template_sha, str):
        raise ValueError("Task9 tokenized tokenizer/template identity is invalid")
    _require_digest(tokenizer_sha, "tokenizer")
    _require_digest(template_sha, "chat template")

    occurrence_digest = hashlib.sha256()
    response_digest = hashlib.sha256()
    selection_db = sqlite3.connect(f"file:{index_path}?mode=ro&immutable=1", uri=True)
    token_db = sqlite3.connect(f"file:{tokenized_path}?mode=ro&immutable=1", uri=True)
    try:
        selection_rows = selection_db.execute(
            "SELECT o.ordinal,o.prompt_uuid,o.source_identity_sha256,o.source_row,o.cell,"
            "o.reuse_index,o.conversation_sha256,o.assistant_response_sha256,"
            "s.canonical_conversation,s.assistant_response,s.language FROM occurrences o "
            "JOIN source_rows s ON o.source_identity_sha256=s.source_identity_sha256 "
            "AND o.source_row=s.source_row WHERE o.strategy='A-repair' ORDER BY o.ordinal"
        )
        token_rows = token_db.execute(
            "SELECT ordinal,prompt_uuid,input_ids_json,loss_mask_json,assistant_tokens "
            "FROM records ORDER BY ordinal"
        )
        task8_rows = (
            row
            for shard in shards
            for row in _authenticated_shard_rows(shard)
        )
        count = 0
        for count, triple in enumerate(zip(selection_rows, token_rows, task8_rows, strict=True), start=1):
            occurrence, tokenized_row, task8_raw = triple
            ordinal = count - 1
            if occurrence[0] != ordinal or tokenized_row[0] != ordinal:
                raise ValueError("Task9/Task8 occurrence ordinal was reordered")
            occurrence_identity = list(occurrence[:8])
            conversation, response = str(occurrence[8]), str(occurrence[9])
            if (
                hashlib.sha256(conversation.encode()).hexdigest() != occurrence[6]
                or hashlib.sha256(response.encode()).hexdigest() != occurrence[7]
            ):
                raise ValueError("Task9 source conversation/response identity mismatch")
            task8 = _validated_task8_row(task8_raw)
            producer = json.loads(str(task8["record_json"]))
            if (
                task8["prompt_uuid"] != occurrence[1]
                or task8["domain"] != occurrence[4]
                or tokenized_row[1] != occurrence[1]
                or task8["input_ids"] != json.loads(tokenized_row[2])
                or task8["loss_mask"] != json.loads(tokenized_row[3])
                or task8["assistant_tokens"] != tokenized_row[4]
                or not _producer_matches_source(
                    producer,
                    conversation=conversation,
                    response=response,
                )
            ):
                raise ValueError("Task8 row does not replay the exact Task9 occurrence")
            occurrence_digest.update((_canonical_json(occurrence_identity) + "\n").encode())
            response_digest.update(
                (_canonical_json([occurrence[2], occurrence[3], occurrence[6], occurrence[7]]) + "\n").encode()
            )
        if count != expected_occurrences:
            raise ValueError("Task9/Task8 replay occurrence count mismatch")
    except sqlite3.Error as error:
        raise ValueError("Task9/Task8 replay database is invalid") from error
    finally:
        selection_db.close()
        token_db.close()
    if (
        occurrence_digest.hexdigest() != identity.get("ordered_occurrences_sha256")
        or response_digest.hexdigest() != identity.get("source_response_root_sha256")
    ):
        raise ValueError("Task9/Task8 replay semantic root mismatch")
    return tokenizer_sha, template_sha, index_sha, tokenized_sha


def _authenticated_declared_file(root: Path, descriptor: object, label: str) -> tuple[Path, str]:
    if not isinstance(descriptor, dict):
        raise ValueError(f"{label} descriptor is invalid")
    relative = _safe_relative(descriptor.get("path"))
    size, digest = descriptor.get("bytes"), descriptor.get("sha256")
    if isinstance(size, bool) or not isinstance(size, int) or not isinstance(digest, str):
        raise ValueError(f"{label} descriptor is invalid")
    _require_digest(digest, label)
    path = root / relative
    raw = _stable_file_bytes(path)
    if len(raw) != size or hashlib.sha256(raw).hexdigest() != digest:
        raise ValueError(f"{label} bytes changed")
    return path, digest


def _authenticated_shard_rows(shard: Task8Shard) -> Iterable[dict[str, object]]:
    raw = _stable_file_bytes(shard.path)
    if len(raw) != shard.bytes or hashlib.sha256(raw).hexdigest() != shard.sha256:
        raise ValueError("Task8 shard content identity mismatch")
    rows = list(_decode_rows(raw, shard.path.suffix))
    if len(rows) != shard.rows:
        raise ValueError("Task8 shard row count changed")
    yield from rows


def _producer_matches_source(
    producer: Mapping[str, object],
    *,
    conversation: str,
    response: str,
) -> bool:
    messages = producer.get("messages")
    tools = producer.get("tools", [])
    if not isinstance(messages, list) or not isinstance(tools, list):
        return False
    assistants = [message for message in messages if isinstance(message, dict) and message.get("role") == "assistant"]
    if not assistants:
        return False
    return (
        _canonical_json({"messages": messages, "tools": tools}) == conversation
        and _canonical_json(assistants[-1]) == response
    )


def materialize_a_canary(
    publication: Task8APublication,
    *,
    output_root: Path,
    scratch_root: Path,
    job_id: str,
    requested_workers: int | None = 96,
) -> ACanaryBuildArtifact:
    """Parallel-authenticate shards, then emit a deterministic source-order view."""
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", job_id):
        raise ValueError("A builder job_id must be safe")
    if output_root.exists() or os.path.lexists(output_root):
        raise FileExistsError(f"A builder output already exists: {output_root}")
    workers, allocated = resolve_worker_count(
        requested_workers=requested_workers,
        declared_shard_count=len(publication.shards),
    )
    output_root.parent.mkdir(parents=True, exist_ok=True)
    scratch_root.mkdir(mode=0o700, parents=True, exist_ok=False)
    offsets: list[int] = []
    offset = 0
    for shard in publication.shards:
        offsets.append(offset)
        offset += shard.rows
    tasks = tuple(
        _WorkerTask(index, offsets[index], shard, scratch_root / f"{index:03d}.jsonl")
        for index, shard in enumerate(publication.shards)
    )
    results: dict[int, _WorkerResult] = {}
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(_process_shard, task): task.index for task in tasks}
        for future in as_completed(futures):
            result = future.result()
            results[result.index] = result
    ordered = tuple(results[index] for index in range(len(tasks)))
    if len(ordered) != len(tasks):
        raise RuntimeError("A builder did not receive every shard result")
    selected = build_a_canary_occurrences(
        row for result in ordered for row in _iter_jsonl(Path(result.spool_path))
    )
    partial = output_root.with_name(f".{output_root.name}.partial-{job_id}")
    partial.mkdir(mode=0o700)
    output = partial / "canary.jsonl"
    payload = b"".join((_canonical_json(row) + "\n").encode() for row in selected)
    output.write_bytes(payload)
    output_sha = hashlib.sha256(payload).hexdigest()
    receipt: dict[str, Any] = {
        "schema_version": 1,
        "producer": "qwen4b-a-repair-builder-v1",
        "source_commit": publication.source_commit,
        "source_projection_sha256": publication.selection_receipt_sha256,
        "source_task8_publication_sha256": publication.publication_sha256,
        "source_corpus_manifest_sha256": publication.corpus_manifest_sha256,
        "source_selection_index_sha256": publication.selection_index_sha256,
        "source_tokenized_database_sha256": publication.tokenized_database_sha256,
        "tokenizer_sha256": publication.tokenizer_sha256,
        "chat_template_sha256": publication.chat_template_sha256,
        "declared_shard_count": len(publication.shards),
        "source_row_count": sum(result.row_count for result in ordered),
        "historical_source_order_occurrences": A_HISTORICAL_QUOTA,
        "complement_occurrences": sum(A_COMPLEMENT_QUOTAS.values()),
        "complement_quotas": A_COMPLEMENT_QUOTAS,
        "occurrence_count": len(selected),
        "output_path": "canary.jsonl",
        "output_bytes": len(payload),
        "output_sha256": output_sha,
        "execution": {
            "allocated_cpus": allocated,
            "effective_workers": workers,
            "process_pool": True,
            "threads_per_worker": 1,
        },
        "shards": [
            asdict(result) | {"spool_path": Path(result.spool_path).name} for result in ordered
        ],
    }
    receipt["receipt_sha256"] = _sha256_json(receipt)
    receipt_path = partial / "BUILD_RECEIPT.json"
    receipt_path.write_text(_canonical_json(receipt) + "\n")
    receipt_sha = hashlib.sha256(receipt_path.read_bytes()).hexdigest()
    atomic_publish_directory(partial, output_root)
    return ACanaryBuildArtifact(
        output_root,
        output_root / "canary.jsonl",
        output_root / "BUILD_RECEIPT.json",
        output_sha,
        receipt_sha,
    )


def _process_shard(task: _WorkerTask) -> _WorkerResult:
    for variable in (
        "ARROW_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
    ):
        os.environ[variable] = "1"
    raw = _stable_file_bytes(task.shard.path)
    if len(raw) != task.shard.bytes or hashlib.sha256(raw).hexdigest() != task.shard.sha256:
        raise ValueError("Task8 shard content identity mismatch")
    rows = list(_decode_rows(raw, task.shard.path.suffix))
    if len(rows) != task.shard.rows:
        raise ValueError("Task8 shard row count changed")
    with task.spool_path.open("x", encoding="utf-8") as stream:
        for row in rows:
            stream.write(_canonical_json(row) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    return _WorkerResult(
        task.index,
        task.start_ordinal,
        len(rows),
        len(raw),
        hashlib.sha256(raw).hexdigest(),
        str(task.spool_path),
        os.getpid(),
    )


def _decode_rows(raw: bytes, suffix: str) -> Iterable[dict[str, object]]:
    if suffix == ".jsonl":
        for line in raw.splitlines():
            if line.strip():
                row = json.loads(line)
                if not isinstance(row, dict):
                    raise ValueError("Task8 shard row must be an object")
                yield row
        return
    if suffix == ".parquet":
        try:
            import pyarrow as pa  # pyright: ignore[reportMissingImports]
            import pyarrow.parquet as pq  # pyright: ignore[reportMissingImports]
        except ImportError as error:
            raise ValueError("pyarrow is required for Parquet Task8 shards") from error
        pa.set_cpu_count(1)
        pa.set_io_thread_count(1)
        import io

        yield from pq.read_table(io.BytesIO(raw), use_threads=False).to_pylist()
        return
    raise ValueError("unsupported Task8 shard type")


def _validated_task8_row(raw: Mapping[str, object]) -> dict[str, object]:
    row = dict(raw)
    identifier = row.get("prompt_uuid")
    record_json = row.get("record_json")
    input_ids = row.get("input_ids")
    loss_mask = row.get("loss_mask")
    assistant_tokens = row.get("assistant_tokens")
    if not isinstance(identifier, str) or not identifier or not isinstance(record_json, str):
        raise ValueError("Task8 A row identity is invalid")
    try:
        producer = json.loads(record_json)
    except json.JSONDecodeError as error:
        raise ValueError("Task8 A record_json is invalid") from error
    if (
        not isinstance(producer, dict)
        or _canonical_json(producer) != record_json
        or producer.get("prompt_uuid") != identifier
        or producer.get("domain") != row.get("domain")
        or producer.get("input_ids") != input_ids
        or producer.get("loss_mask") != loss_mask
        or producer.get("assistant_tokens") != assistant_tokens
    ):
        raise ValueError("Task8 A row is not source-native")
    if (
        not isinstance(input_ids, list)
        or not input_ids
        or not all(isinstance(token, int) and not isinstance(token, bool) for token in input_ids)
        or not isinstance(loss_mask, list)
        or len(loss_mask) != len(input_ids)
        or not all(isinstance(value, bool) for value in loss_mask)
        or not isinstance(assistant_tokens, int)
        or isinstance(assistant_tokens, bool)
        or assistant_tokens <= 0
        or sum(loss_mask) != assistant_tokens
        or row.get("rejection_reason") is not None
    ):
        raise ValueError("Task8 A row lacks a source-native assistant target")
    return row


def _streaming_entry(row: Mapping[str, object]) -> dict[str, object]:
    producer = json.loads(str(row["record_json"]))
    messages = producer.get("messages")
    tools = producer.get("tools", [])
    if (
        not isinstance(messages, list)
        or not messages
        or any(
            not isinstance(message, dict)
            or message.get("role") not in {"system", "developer", "user", "assistant", "tool"}
            or not isinstance(message.get("content"), str)
            for message in messages
        )
        or not isinstance(tools, list)
    ):
        raise ValueError("Task8 A record_json lacks canonical streaming conversations")
    loss_mask = row["loss_mask"]
    if not isinstance(loss_mask, list):
        raise ValueError("Task8 A loss mask is invalid")
    return {
        "conversation_id": row["prompt_uuid"],
        "messages": messages,
        "tools": tools,
        "input_ids": row["input_ids"],
        "loss_mask": [int(value) for value in loss_mask],
    }


def _complement_category(row: Mapping[str, object]) -> str | None:
    domain = str(row.get("domain", "")).lower()
    producer = json.loads(str(row["record_json"]))
    language = str(producer.get("language", "")).lower()
    if domain == "stem" and not language:
        return "stem"
    if domain == "multilingual" and language in A_COMPLEMENT_QUOTAS:
        return language
    return None


def _validated_quotas(raw: Mapping[str, int]) -> dict[str, int]:
    quotas = dict(raw)
    if set(quotas) != set(A_COMPLEMENT_QUOTAS) or any(
        isinstance(value, bool) or not isinstance(value, int) or value < 0
        for value in quotas.values()
    ):
        raise ValueError("A complement quotas are invalid")
    return quotas


def _iter_jsonl(path: Path) -> Iterable[dict[str, object]]:
    for line in path.read_text().splitlines():
        if line:
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError("spooled Task8 row must be an object")
            yield row


def _stable_file_bytes(path: Path) -> bytes:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    with os.fdopen(descriptor, "rb") as stream:
        before = os.fstat(stream.fileno())
        namespace_before = os.lstat(path)
        raw = stream.read()
        after = os.fstat(stream.fileno())
    namespace_after = os.lstat(path)

    def signature(value: os.stat_result) -> tuple[int, int, int, int]:
        return value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns

    if (
        not stat.S_ISREG(before.st_mode)
        or (namespace_before.st_dev, namespace_before.st_ino) != (before.st_dev, before.st_ino)
        or (namespace_after.st_dev, namespace_after.st_ino) != (before.st_dev, before.st_ino)
        or signature(before) != signature(after)
    ):
        raise ValueError("artifact is not an inode-stable regular file")
    return raw


def _canonical_object(raw: bytes, label: str) -> dict[str, Any]:
    value = json.loads(raw)
    if not isinstance(value, dict) or raw != (_canonical_json(value) + "\n").encode():
        raise ValueError(f"{label} must be a canonical object")
    return value


def _safe_relative(value: object) -> str:
    if not isinstance(value, str):
        raise ValueError("Task8 shard path must be a string")
    path = Path(value)
    if path.is_absolute() or ".." in path.parts or value != path.as_posix():
        raise ValueError("Task8 shard path is unsafe")
    return value


def _require_digest(value: str, label: str) -> None:
    if _SHA256.fullmatch(value) is None:
        raise ValueError(f"Task8 {label} SHA-256 must be exact")


def _canonical_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _sha256_json(value: dict[str, Any]) -> str:
    return hashlib.sha256(_canonical_json(value).encode()).hexdigest()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task8-root", type=Path, required=True)
    parser.add_argument("--publication-sha256", required=True)
    parser.add_argument("--selection-sha256", required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--scratch-root", type=Path, required=True)
    parser.add_argument("--job-id", required=True)
    parser.add_argument("--workers", type=int, default=96)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    publication = load_task8_a_publication(
        args.task8_root,
        expected_publication_sha256=args.publication_sha256,
        expected_selection_receipt_sha256=args.selection_sha256,
        source_commit=args.source_commit,
    )
    artifact = materialize_a_canary(
        publication,
        output_root=args.output_root,
        scratch_root=args.scratch_root,
        job_id=args.job_id,
        requested_workers=args.workers,
    )
    print(
        _canonical_json(
            {"output": str(artifact.output_path), "receipt": str(artifact.receipt_path)}
        )
    )


if __name__ == "__main__":
    main()
