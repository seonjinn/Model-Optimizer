# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Materialize and publish one authenticated Qwen3-4B PTV2 Task8 arm."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sqlite3
import stat
import sys
from collections import Counter
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from itertools import zip_longest
from pathlib import Path, PurePosixPath
from types import SimpleNamespace
from typing import Any

import yaml

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_COMMIT = re.compile(r"^[0-9a-f]{40}$")
_MAX_ROLE_FILES = 201
_MAX_RECEIPT_BYTES = 1024 * 1024
_MAX_RECORD_BYTES = 4096
_MAX_LABEL_LENGTH = 128
_RESPONSE_KEYS = frozenset(
    {
        "schema_version",
        "role",
        "source_commit",
        "selection_sha256",
        "source_response_root_sha256",
        "occurrence_count",
        "files",
        "receipt_sha256",
    }
)
_REJECTION_KEYS = frozenset(
    {
        "schema_version",
        "role",
        "source_commit",
        "selection_sha256",
        "occurrence_count",
        "rejection_count",
        "reason_counts",
        "files",
        "receipt_sha256",
    }
)
_RESPONSE_RECORD_KEYS = frozenset(
    {
        "ordinal",
        "prompt_uuid",
        "source_identity_sha256",
        "source_row",
        "conversation_sha256",
        "assistant_response_sha256",
    }
)
_REJECTION_RECORD_KEYS = frozenset(
    {
        "ordinal",
        "prompt_uuid",
        "domain",
        "lane",
        "context_bucket",
        "source_identity_sha256",
        "source_row",
        "rejection_reason",
    }
)


class Task8BuildError(ValueError):
    """A Task8 input or publication contract is incomplete or invalid."""


@dataclass(frozen=True)
class _RoleEvidence:
    rejection_count: int
    rejection_reason_counts: Mapping[str, int]
    rejection_files: tuple[Path, ...]


def publication_rows_per_shard(occurrence_count: int) -> int:
    """Return the stable row width that yields at most 201 source-order shards."""
    if (
        isinstance(occurrence_count, bool)
        or not isinstance(occurrence_count, int)
        or occurrence_count < 1
    ):
        raise Task8BuildError("publication occurrence count must be positive")
    return (occurrence_count + 200) // 201


@dataclass(frozen=True)
class _BoundTokenizer:
    tokenizer: Any
    tokenizer_sha256: str
    chat_template: str
    assistant_loss_target_sha256: str

    def apply_chat_template(self, messages: Any, **kwargs: Any) -> Any:
        return self.tokenizer.apply_chat_template(messages, **kwargs)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_json(value: object) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()


def _require_digest(value: str, label: str) -> None:
    if _SHA256.fullmatch(value) is None:
        raise Task8BuildError(f"{label} must be a lowercase SHA-256")


def _canonical_absolute_path(value: os.PathLike[str] | str, label: str) -> Path:
    raw = os.fspath(value)
    if not raw.startswith("/") or raw == "/" or "//" in raw:
        raise Task8BuildError(f"{label} must be a canonical absolute path")
    components = raw.split("/")[1:]
    if not components or any(component in {"", ".", ".."} for component in components):
        raise Task8BuildError(f"{label} must be a canonical absolute path")
    canonical = PurePosixPath(raw).as_posix()
    if raw != canonical:
        raise Task8BuildError(f"{label} must be a canonical absolute path")
    return Path(raw)


def _open_directory_nofollow(path: Path, label: str) -> int:
    canonical = _canonical_absolute_path(path, label)
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open("/", flags)
    try:
        for component in canonical.parts[1:]:
            child = os.open(component, flags, dir_fd=descriptor)
            os.close(descriptor)
            descriptor = child
    except OSError as error:
        os.close(descriptor)
        raise Task8BuildError(f"{label} contains a symlink or unsafe ancestor") from error
    return descriptor


def _open_regular_nofollow(path: Path, label: str) -> int:
    canonical = _canonical_absolute_path(path, label)
    parent = _open_directory_nofollow(canonical.parent, f"{label} parent")
    try:
        descriptor = os.open(
            canonical.name,
            os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0),
            dir_fd=parent,
        )
    except OSError as error:
        raise Task8BuildError(f"{label} is missing or unsafe") from error
    finally:
        os.close(parent)
    if not stat.S_ISREG(os.fstat(descriptor).st_mode):
        os.close(descriptor)
        raise Task8BuildError(f"{label} must be a no-follow regular file")
    return descriptor


def _read_regular_nofollow(path: Path, label: str, *, max_bytes: int | None = None) -> bytes:
    descriptor = _open_regular_nofollow(path, label)
    if max_bytes is not None and os.fstat(descriptor).st_size > max_bytes:
        os.close(descriptor)
        raise Task8BuildError(f"{label} exceeds its bounded size")
    with os.fdopen(descriptor, "rb") as stream:
        return stream.read()


def validate_durable_publication_root(
    publication_root: os.PathLike[str] | str,
    durable_prefix: os.PathLike[str] | str,
) -> Path:
    """Validate a missing canonical destination below a no-follow durable parent."""
    destination = _canonical_absolute_path(publication_root, "publication root")
    prefix = _canonical_absolute_path(durable_prefix, "durable prefix")
    if destination == prefix or destination.parts[: len(prefix.parts)] != prefix.parts:
        raise Task8BuildError("publication root is outside the durable dataset-study root")
    descriptor = _open_directory_nofollow(destination.parent, "publication root parent")
    os.close(descriptor)
    if os.path.lexists(destination):
        raise Task8BuildError("publication root must be an immutable missing destination")
    return destination


def _read_receipt(path: Path, expected_sha256: str, label: str) -> dict[str, Any]:
    _require_digest(expected_sha256, f"{label} receipt SHA-256")
    raw = _read_regular_nofollow(path, f"{label} receipt", max_bytes=_MAX_RECEIPT_BYTES)
    if hashlib.sha256(raw).hexdigest() != expected_sha256:
        raise Task8BuildError(f"{label} receipt SHA-256 mismatch")
    try:
        payload = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise Task8BuildError(f"{label} receipt is not JSON") from error
    if not isinstance(payload, dict):
        raise Task8BuildError(f"{label} receipt must be an object")
    return payload


def _stage_schema2_exposure_input(
    bundle: Path,
    work_root: Path,
    *,
    strategy: str,
    scientific_receipt_sha256: str,
) -> tuple[Path, str]:
    """Stage a generic role receipt over all three immutable schema-v2 bundle files."""
    _require_digest(scientific_receipt_sha256, "scientific exposure receipt")
    wrapper_root = work_root / "exposure-publication-input"
    wrapper_root.mkdir(mode=0o700)
    descriptors: list[dict[str, object]] = []
    for name in ("SCIENTIFIC.json", "records.jsonl", "EXECUTION.json"):
        source = bundle / name
        before = os.lstat(source)
        if source.is_symlink() or not stat.S_ISREG(before.st_mode):
            raise Task8BuildError("schema2 exposure bundle contains an unsafe file")
        destination = wrapper_root / name
        try:
            os.link(source, destination, follow_symlinks=False)
        except OSError as error:
            raise Task8BuildError(
                "schema2 exposure wrapper requires one node-local immutable filesystem"
            ) from error
        linked = os.lstat(destination)
        after = os.lstat(source)
        if (before.st_dev, before.st_ino) != (linked.st_dev, linked.st_ino) or (
            before.st_dev,
            before.st_ino,
            before.st_size,
        ) != (after.st_dev, after.st_ino, after.st_size):
            raise Task8BuildError("schema2 exposure bundle changed during authenticated staging")
        descriptors.append(
            {"path": name, "bytes": linked.st_size, "sha256": _sha256_file(destination)}
        )
    scientific = json.loads((wrapper_root / "SCIENTIFIC.json").read_bytes())
    execution = json.loads((wrapper_root / "EXECUTION.json").read_bytes())
    if (
        not isinstance(scientific, dict)
        or scientific.get("receipt_sha256") != scientific_receipt_sha256
        or not isinstance(execution, dict)
        or execution.get("scientific_receipt_sha256") != scientific_receipt_sha256
    ):
        raise Task8BuildError("schema2 exposure wrapper lineage is inconsistent")
    body = {
        "schema_version": 2,
        "role": "exposure",
        "contract": "ptv2-schema2-exposure-publication-v1",
        "strategy": strategy,
        "scientific_receipt_sha256": scientific_receipt_sha256,
        "identity": {key: value for key, value in scientific.items() if key != "receipt_sha256"},
        "files": descriptors,
    }
    self_sha256 = hashlib.sha256(_canonical_json(body)).hexdigest()
    wrapper = wrapper_root / "EXPOSURE.json"
    raw = _canonical_json(body | {"receipt_sha256": self_sha256}) + b"\n"
    with wrapper.open("xb") as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())
    descriptor = os.open(wrapper_root, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    return wrapper, hashlib.sha256(raw).hexdigest()


def _stage_schema2_tokenized_input(
    corpus: Any,
    work_root: Path,
) -> tuple[Path, str]:
    bundle = Path(corpus.receipt_path).parent
    wrapper_root = work_root / "tokenized-publication-input"
    wrapper_root.mkdir(mode=0o700)
    names = ["TOKENIZED.json", "records.sqlite3"]
    if (bundle / "EXECUTION_RECEIPT.json").is_file():
        names.append("EXECUTION_RECEIPT.json")
    descriptors: list[dict[str, object]] = []
    for name in names:
        source = bundle / name
        before = os.lstat(source)
        if source.is_symlink() or not stat.S_ISREG(before.st_mode):
            raise Task8BuildError("schema2 tokenized bundle contains an unsafe file")
        destination = wrapper_root / name
        try:
            os.link(source, destination, follow_symlinks=False)
        except OSError as error:
            raise Task8BuildError(
                "schema2 tokenized wrapper requires one node-local immutable filesystem"
            ) from error
        linked = os.lstat(destination)
        after = os.lstat(source)
        if (before.st_dev, before.st_ino) != (linked.st_dev, linked.st_ino) or (
            before.st_dev,
            before.st_ino,
            before.st_size,
        ) != (after.st_dev, after.st_ino, after.st_size):
            raise Task8BuildError("schema2 tokenized bundle changed during authenticated staging")
        descriptors.append(
            {"path": name, "bytes": linked.st_size, "sha256": _sha256_file(destination)}
        )
    tokenized = json.loads((wrapper_root / "TOKENIZED.json").read_bytes())
    if (
        not isinstance(tokenized, dict)
        or tokenized.get("schema_version") != 2
        or tokenized.get("receipt_sha256") != corpus.receipt_sha256
        or tokenized.get("database_sha256") != corpus.tokenized_sha256
    ):
        raise Task8BuildError("schema2 tokenized wrapper lineage is inconsistent")
    body = {
        "schema_version": 2,
        "role": "tokenized",
        "contract": "ptv2-schema2-tokenized-publication-v1",
        "strategy": corpus.strategy,
        "tokenized_receipt_sha256": corpus.receipt_sha256,
        "identity": {key: value for key, value in tokenized.items() if key != "receipt_sha256"},
        "files": descriptors,
    }
    if "execution_receipt" in tokenized:
        body["execution_receipt"] = next(
            descriptor
            for descriptor in descriptors
            if descriptor["path"] == "EXECUTION_RECEIPT.json"
        )
    self_sha256 = hashlib.sha256(_canonical_json(body)).hexdigest()
    wrapper = wrapper_root / "TOKENIZED_INPUT.json"
    raw = _canonical_json(body | {"receipt_sha256": self_sha256}) + b"\n"
    with wrapper.open("xb") as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())
    descriptor = os.open(wrapper_root, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    return wrapper, hashlib.sha256(raw).hexdigest()


def _authenticate_declared_files(
    receipt_path: Path, payload: Mapping[str, Any], label: str
) -> tuple[Path, ...]:
    descriptors = payload.get("files")
    if not isinstance(descriptors, list) or not descriptors or len(descriptors) > _MAX_ROLE_FILES:
        raise Task8BuildError(f"{label} receipt has no declared files")
    authenticated: list[Path] = []
    for descriptor in descriptors:
        if not isinstance(descriptor, Mapping):
            raise Task8BuildError(f"{label} declared file descriptor is malformed")
        authenticated.append(
            _authenticate_descriptor(receipt_path, descriptor, f"{label} declared file")
        )
    return tuple(authenticated)


def _authenticate_descriptor(receipt_path: Path, descriptor: Mapping[str, Any], label: str) -> Path:
    raw_relative = descriptor.get("path", descriptor.get("staged_path"))
    relative = PurePosixPath(str(raw_relative or ""))
    if relative.is_absolute() or ".." in relative.parts or not relative.parts:
        raise Task8BuildError(f"{label} path is unsafe")
    directory_flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
    file_flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    parent_descriptor = _open_directory_nofollow(receipt_path.parent, f"{label} parent")
    file_descriptor = -1
    try:
        for component in relative.parts[:-1]:
            try:
                child = os.open(component, directory_flags, dir_fd=parent_descriptor)
            except OSError as error:
                raise Task8BuildError(f"{label} contains a symlink or non-directory") from error
            os.close(parent_descriptor)
            parent_descriptor = child
        try:
            file_descriptor = os.open(relative.parts[-1], file_flags, dir_fd=parent_descriptor)
        except OSError as error:
            raise Task8BuildError(f"{label} is missing or unsafe") from error
        observed = os.fstat(file_descriptor)
        digest = descriptor.get("sha256")
        content_digest = hashlib.sha256()
        with os.fdopen(os.dup(file_descriptor), "rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                content_digest.update(block)
        if (
            not stat.S_ISREG(observed.st_mode)
            or observed.st_size != descriptor.get("bytes")
            or not isinstance(digest, str)
            or _SHA256.fullmatch(digest) is None
            or content_digest.hexdigest() != digest
        ):
            raise Task8BuildError(f"{label} authentication failed")
    finally:
        if file_descriptor >= 0:
            os.close(file_descriptor)
        os.close(parent_descriptor)
    return receipt_path.parent.joinpath(*relative.parts)


def _identity(payload: Mapping[str, Any]) -> Mapping[str, Any]:
    nested = payload.get("identity")
    return nested if isinstance(nested, Mapping) else payload


def _canonical_records(paths: tuple[Path, ...], label: str) -> Iterator[dict[str, Any]]:
    for path in paths:
        descriptor = _open_regular_nofollow(path, f"{label} records")
        with os.fdopen(descriptor, "rb") as stream:
            for line_number, raw in enumerate(stream, start=1):
                if len(raw) > _MAX_RECORD_BYTES:
                    raise Task8BuildError(f"{label} record exceeds its bounded size")
                try:
                    record = json.loads(raw)
                except (UnicodeDecodeError, json.JSONDecodeError) as error:
                    raise Task8BuildError(
                        f"{label} record {line_number} is not canonical JSON"
                    ) from error
                if not isinstance(record, dict) or raw != _canonical_json(record) + b"\n":
                    raise Task8BuildError(f"{label} record schema is invalid")
                yield record


def _response_record_identity(record: Mapping[str, Any], ordinal: int) -> tuple[Any, ...]:
    if (
        set(record) != _RESPONSE_RECORD_KEYS
        or record.get("ordinal") != ordinal
        or isinstance(record.get("source_row"), bool)
        or not isinstance(record.get("source_row"), int)
        or record["source_row"] < 0
        or any(
            not isinstance(record.get(key), str) or _SHA256.fullmatch(record[key]) is None
            for key in (
                "prompt_uuid",
                "source_identity_sha256",
                "conversation_sha256",
                "assistant_response_sha256",
            )
        )
    ):
        raise Task8BuildError("response record schema is invalid")
    return (
        record["ordinal"],
        record["prompt_uuid"],
        record["source_identity_sha256"],
        record["source_row"],
        record["conversation_sha256"],
        record["assistant_response_sha256"],
    )


def _validate_response_records(view: Any, paths: tuple[Path, ...]) -> tuple[int, str]:
    selection = sqlite3.connect(f"file:{Path(view.index_path)}?mode=ro", uri=True)
    digest = hashlib.sha256()
    count = 0
    try:
        expected_rows = selection.execute(
            "SELECT ordinal,prompt_uuid,source_identity_sha256,source_row,"
            "conversation_sha256,assistant_response_sha256 FROM occurrences "
            "WHERE strategy=? ORDER BY ordinal",
            (view.strategy,),
        )
        sentinel = object()
        for record, expected in zip_longest(
            _canonical_records(paths, "response"), expected_rows, fillvalue=sentinel
        ):
            if record is sentinel or expected is sentinel:
                raise Task8BuildError("response record count does not match Task9 selection")
            assert isinstance(record, Mapping) and isinstance(expected, tuple)
            identity = _response_record_identity(record, count)
            if identity != expected:
                raise Task8BuildError("response record identity does not match Task9 selection")
            digest.update(
                _canonical_json([identity[2], identity[3], identity[4], identity[5]]) + b"\n"
            )
            count += 1
    finally:
        selection.close()
    return count, digest.hexdigest()


def _validated_rejection_record(record: Mapping[str, Any], ordinal: int) -> dict[str, Any]:
    if (
        set(record) != _REJECTION_RECORD_KEYS
        or record.get("ordinal") != ordinal
        or isinstance(record.get("source_row"), bool)
        or not isinstance(record.get("source_row"), int)
        or record["source_row"] < 0
        or any(
            not isinstance(record.get(key), str)
            or not record[key]
            or len(record[key]) > _MAX_LABEL_LENGTH
            for key in ("domain", "lane", "context_bucket", "rejection_reason")
        )
        or any(
            not isinstance(record.get(key), str) or _SHA256.fullmatch(record[key]) is None
            for key in ("prompt_uuid", "source_identity_sha256")
        )
    ):
        raise Task8BuildError("rejection record schema is invalid")
    return dict(record)


def _validate_rejection_records(
    paths: tuple[Path, ...],
) -> tuple[int, dict[str, int]]:
    counts: Counter[str] = Counter()
    count = 0
    for record in _canonical_records(paths, "rejection"):
        validated = _validated_rejection_record(record, count)
        counts[validated["rejection_reason"]] += 1
        count += 1
    return count, dict(sorted(counts.items()))


def validate_required_role_receipts(
    *,
    view: Any,
    task9_source_commit: str,
    response_receipt: Path,
    response_receipt_sha256: str,
    rejection_receipt: Path,
    rejection_receipt_sha256: str,
) -> _RoleEvidence:
    """Fail before tokenization unless genuine response and rejection roots are pinned."""
    response_payload = _read_receipt(Path(response_receipt), response_receipt_sha256, "response")
    rejection_payload = _read_receipt(
        Path(rejection_receipt), rejection_receipt_sha256, "rejection"
    )
    response_files = _authenticate_declared_files(
        Path(response_receipt), response_payload, "response"
    )
    rejection_files = _authenticate_declared_files(
        Path(rejection_receipt), rejection_payload, "rejection"
    )
    response = _identity(response_payload)
    rejection = _identity(rejection_payload)
    selection_sha256 = getattr(view, "selection_sha256", None)
    response_root = getattr(view, "source_response_root_sha256", None)
    response_claimed = response_payload.get("receipt_sha256")
    rejection_claimed = rejection_payload.get("receipt_sha256")
    response_without_claim = {
        key: value for key, value in response_payload.items() if key != "receipt_sha256"
    }
    rejection_without_claim = {
        key: value for key, value in rejection_payload.items() if key != "receipt_sha256"
    }
    reason_counts = rejection.get("reason_counts")
    rejection_count = rejection.get("rejection_count")
    if (
        set(response_payload) != _RESPONSE_KEYS
        or response.get("schema_version") != 1
        or response.get("role") != "response"
        or response.get("source_commit") != task9_source_commit
        or _COMMIT.fullmatch(task9_source_commit) is None
        or response.get("occurrence_count") != getattr(view, "occurrence_count", None)
        or response_claimed != hashlib.sha256(_canonical_json(response_without_claim)).hexdigest()
    ):
        raise Task8BuildError("response receipt schema or identity is invalid")
    if (
        set(rejection_payload) != _REJECTION_KEYS
        or rejection.get("schema_version") != 1
        or rejection.get("role") != "rejection"
        or rejection.get("source_commit") != task9_source_commit
        or rejection.get("occurrence_count") != getattr(view, "occurrence_count", None)
        or isinstance(rejection_count, bool)
        or not isinstance(rejection_count, int)
        or rejection_count < 0
        or rejection_count > getattr(view, "occurrence_count", -1)
        or not isinstance(reason_counts, Mapping)
        or len(reason_counts) > _MAX_LABEL_LENGTH
        or any(
            not isinstance(reason, str)
            or not reason
            or len(reason) > _MAX_LABEL_LENGTH
            or isinstance(count, bool)
            or not isinstance(count, int)
            or count < 1
            for reason, count in reason_counts.items()
        )
        or sum(reason_counts.values()) != rejection_count
        or rejection_claimed != hashlib.sha256(_canonical_json(rejection_without_claim)).hexdigest()
    ):
        raise Task8BuildError("rejection receipt schema or identity is invalid")
    if response.get("selection_sha256") != selection_sha256:
        raise Task8BuildError("response receipt is not bound to the Task9 selection")
    if response.get("source_response_root_sha256") != response_root:
        raise Task8BuildError("response receipt does not bind Task9 source-native responses")
    if rejection.get("selection_sha256") != selection_sha256:
        raise Task8BuildError("rejection receipt is not bound to the Task9 selection")
    response_count, computed_response_root = _validate_response_records(view, response_files)
    if response_count != response.get("occurrence_count"):
        raise Task8BuildError("response record count does not reconcile with its receipt")
    if computed_response_root != response_root:
        raise Task8BuildError("response records do not reproduce the Task9 response root")
    observed_rejections, observed_reasons = _validate_rejection_records(rejection_files)
    if observed_rejections != rejection_count or observed_reasons != dict(reason_counts):
        raise Task8BuildError("rejection records do not reconcile with their receipt")
    return _RoleEvidence(observed_rejections, observed_reasons, rejection_files)


def _publication_rows(
    view: Any,
    tokenized_database: Path,
    rejection_files: tuple[Path, ...],
) -> Iterator[dict[str, Any]]:
    selection = sqlite3.connect(f"file:{Path(view.index_path)}?mode=ro", uri=True)
    tokenized = sqlite3.connect(f"file:{tokenized_database}?mode=ro", uri=True)
    try:
        selected_rows = selection.execute(
            "SELECT occurrences.ordinal,occurrences.prompt_uuid,occurrences.cell,"
            "source_rows.language FROM occurrences JOIN source_rows "
            "ON occurrences.source_identity_sha256=source_rows.source_identity_sha256 "
            "AND occurrences.source_row=source_rows.source_row "
            "WHERE occurrences.strategy=? ORDER BY occurrences.ordinal",
            (view.strategy,),
        )
        token_rows = tokenized.execute(
            "SELECT ordinal,prompt_uuid,input_ids_json,loss_mask_json,assistant_tokens "
            "FROM records ORDER BY ordinal"
        )
        count = 0
        sentinel = object()
        for selected, tokens in zip_longest(selected_rows, token_rows, fillvalue=sentinel):
            if selected is sentinel or tokens is sentinel:
                raise Task8BuildError("selection and tokenized row counts differ")
            assert isinstance(selected, tuple) and isinstance(tokens, tuple)
            ordinal, prompt_uuid, cell, language = selected
            token_ordinal, token_prompt_uuid, ids_json, mask_json, assistant_tokens = tokens
            if (ordinal, prompt_uuid) != (token_ordinal, token_prompt_uuid):
                raise Task8BuildError("selection and tokenized row identities differ")
            input_ids = json.loads(ids_json)
            loss_mask = [bool(value) for value in json.loads(mask_json)]
            count += 1
            yield {
                "prompt_uuid": prompt_uuid,
                "domain": cell,
                "lane": "source-native",
                "context_bucket": "ptv2-one-pass",
                "language": language,
                "input_ids": input_ids,
                "loss_mask": loss_mask,
                "assistant_tokens": assistant_tokens,
                "rejection_reason": None,
            }
        if count != view.occurrence_count:
            raise Task8BuildError("published row count does not match Task9 selection")
    finally:
        tokenized.close()
        selection.close()
    for ordinal, record in enumerate(_canonical_records(rejection_files, "rejection")):
        rejected = _validated_rejection_record(record, ordinal)
        yield {
            "prompt_uuid": rejected["prompt_uuid"],
            "domain": rejected["domain"],
            "lane": rejected["lane"],
            "context_bucket": rejected["context_bucket"],
            "language": "",
            "input_ids": [],
            "loss_mask": [],
            "assistant_tokens": 0,
            "rejection_reason": rejected["rejection_reason"],
        }


def _task8_production_exposure_target(view: Any) -> int:
    """Return the exact reachable scientific boundary for a production 2M selection."""
    occurrence_count = getattr(view, "occurrence_count", None)
    if occurrence_count != 2_000_000 or isinstance(occurrence_count, bool):
        raise Task8BuildError("production Task8 selection must contain exactly 2M occurrences")
    return 256_000_000


def materialize_task8_publication(
    *,
    view: Any,
    task9_source_commit: str,
    tokenizer: Any,
    tokenizer_sha256: str,
    chat_template_sha256: str,
    assistant_loss_target_sha256: str,
    training_config_sha256: str,
    source_receipt: Path,
    source_receipt_sha256: str,
    selection_receipt: Path,
    selection_receipt_sha256: str,
    response_receipt: Path,
    response_receipt_sha256: str,
    rejection_receipt: Path,
    rejection_receipt_sha256: str,
    work_root: Path,
    materialization_root: Path,
    publication_root: Path,
    source_commit: str,
    job_id: str,
    workers: int = 96,
) -> Any:
    """Build token/exposure artifacts and atomically publish all six authenticated roles."""
    from build_assistant_token_views import (  # pyright: ignore[reportMissingImports]
        _materialize_ptv2_exposure,
        derive_ptv2_one_pass_corpus,
    )
    from specdec_publication import (  # pyright: ignore[reportMissingImports]
        CorpusBundle,
        InputArtifact,
        publish_bundle,
    )

    source_payload = _read_receipt(Path(source_receipt), source_receipt_sha256, "source")
    _authenticate_declared_files(Path(source_receipt), source_payload, "source")
    _read_receipt(Path(selection_receipt), selection_receipt_sha256, "selection")
    role_evidence = validate_required_role_receipts(
        view=view,
        task9_source_commit=task9_source_commit,
        response_receipt=response_receipt,
        response_receipt_sha256=response_receipt_sha256,
        rejection_receipt=rejection_receipt,
        rejection_receipt_sha256=rejection_receipt_sha256,
    )
    for digest, label in (
        (tokenizer_sha256, "tokenizer"),
        (chat_template_sha256, "chat template"),
        (assistant_loss_target_sha256, "assistant loss target"),
        (training_config_sha256, "training config"),
        (source_receipt_sha256, "source receipt"),
        (selection_receipt_sha256, "selection receipt"),
    ):
        _require_digest(digest, label)
    if _COMMIT.fullmatch(source_commit) is None:
        raise Task8BuildError("source commit must be exact")
    if _COMMIT.fullmatch(task9_source_commit) is None:
        raise Task8BuildError("Task9 source commit must be exact")
    if workers < 1:
        raise Task8BuildError("Task8 workers must be positive")
    work_root.mkdir(mode=0o700, parents=True, exist_ok=False)
    corpus = derive_ptv2_one_pass_corpus(
        view,
        tokenizer_sha256=tokenizer_sha256,
        chat_template_sha256=chat_template_sha256,
        assistant_loss_target_sha256=assistant_loss_target_sha256,
        training_config_sha256=training_config_sha256,
        tokenizer=tokenizer,
        output_root=materialization_root,
        work_root=work_root,
        workers=workers,
        source_commit=source_commit,
    )
    exposure_target = _task8_production_exposure_target(view)
    scientific_receipt_sha256 = _materialize_ptv2_exposure(corpus, exposure_target, workers=workers)
    exposure_root = (
        Path(corpus.tokenized_path).parent.parent / f"{corpus.strategy.lower()}-exposures"
    )
    exposure_bundle = exposure_root / (
        f"{corpus.strategy.lower()}-{exposure_target}-assistant-tokens-v2"
    )
    exposure_receipt, exposure_receipt_sha256 = _stage_schema2_exposure_input(
        exposure_bundle,
        work_root,
        strategy=corpus.strategy,
        scientific_receipt_sha256=scientific_receipt_sha256,
    )
    tokenized_receipt, tokenized_receipt_sha256 = _stage_schema2_tokenized_input(corpus, work_root)
    artifacts = (
        InputArtifact("source", Path(source_receipt), source_receipt_sha256),
        InputArtifact("selection", Path(selection_receipt), selection_receipt_sha256),
        InputArtifact("response", Path(response_receipt), response_receipt_sha256),
        InputArtifact("tokenized", tokenized_receipt, tokenized_receipt_sha256),
        InputArtifact("exposure", exposure_receipt, exposure_receipt_sha256),
        InputArtifact("rejection", Path(rejection_receipt), rejection_receipt_sha256),
    )
    return publish_bundle(
        CorpusBundle(
            artifacts=artifacts,
            rows=lambda: _publication_rows(
                view, Path(corpus.tokenized_path), role_evidence.rejection_files
            ),
            prompt_count=corpus.occurrence_count,
            assistant_token_count=corpus.assistant_tokens,
            quarantine_count=role_evidence.rejection_count,
            selection_manifest_sha256=selection_receipt_sha256,
            artifact_source_commit=source_commit,
        ),
        publication_root,
        job_id,
        rows_per_shard=publication_rows_per_shard(
            corpus.occurrence_count + role_evidence.rejection_count
        ),
    )


_PTV2_BASE_RECEIPT_KEYS = frozenset(
    {
        "schema_version",
        "strategy",
        "occurrence_count",
        "trainer_epochs",
        "assistant_tokens",
        "serialized_tokens",
        "packed_sequence_lower_bound",
        "unique_prompt_count",
        "natural_duplicate_count",
        "constructed_repeat_count",
        "milestone_occurrences",
        "milestone_steps",
        "segment_occurrences",
        "segment_steps",
        "cumulative_segment_steps",
        "segment_final_valid_occurrences",
        "tokenizer_sha256",
        "chat_template_sha256",
        "assistant_loss_target_sha256",
        "training_config_sha256",
        "source_response_root_sha256",
        "ordered_occurrences_sha256",
        "selection_sha256",
        "base_occurrence_multiplicity_sha256",
        "bucket_assistant_token_histogram",
        "database_path",
        "database_sha256",
        "database_bytes",
        "receipt_sha256",
    }
)
_PTV2_PARALLEL_RECEIPT_KEYS = frozenset(
    {"execution_receipt", "source_index_bytes", "source_index_sha256"}
)
_PTV2_HISTORICAL_RECEIPT_KEYS = frozenset(
    {
        "historical_parent_receipt_sha256",
        "historical_parent_tokenized_sha256",
        "historical_prefix_occurrence_count",
        "historical_prefix_record_stream_sha256",
    }
)


def _exact_positive_int(value: object, label: str, *, allow_zero: bool = False) -> int:
    minimum = 0 if allow_zero else 1
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise Task8BuildError(f"{label} must be an exact integer")
    return value


def _exact_int_tuple(value: object, length: int, label: str) -> tuple[int, ...]:
    if (
        not isinstance(value, list)
        or len(value) != length
        or any(isinstance(item, bool) or not isinstance(item, int) or item < 0 for item in value)
    ):
        raise Task8BuildError(f"{label} must be an exact integer list")
    return tuple(value)


def _load_pinned_ptv2_corpus(
    *,
    receipt_path: Path,
    receipt_file_sha256: str,
    expected_strategy: str,
    study_root: Path,
) -> Any:
    """Load one immutable schema-v2 tokenized corpus from a caller-pinned file identity."""
    from build_assistant_token_views import (  # pyright: ignore[reportMissingImports]
        PTV2OnePassCorpus,
    )

    _require_digest(receipt_file_sha256, f"{expected_strategy} tokenized receipt file")
    root = _canonical_absolute_path(study_root, "PTV2 study root")
    expected_path = root / f"{expected_strategy.lower()}-tokenized" / "TOKENIZED.json"
    if receipt_path != expected_path:
        raise Task8BuildError(f"{expected_strategy} tokenized receipt path is outside its arm root")
    raw = _read_regular_nofollow(
        receipt_path, f"{expected_strategy} tokenized receipt", max_bytes=_MAX_RECEIPT_BYTES
    )
    if hashlib.sha256(raw).hexdigest() != receipt_file_sha256:
        raise Task8BuildError(f"{expected_strategy} tokenized receipt file SHA-256 mismatch")
    try:
        payload = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise Task8BuildError(f"{expected_strategy} tokenized receipt is not JSON") from error
    if not isinstance(payload, dict):
        raise Task8BuildError(f"{expected_strategy} tokenized receipt must be an object")
    optional = (
        _PTV2_HISTORICAL_RECEIPT_KEYS
        if expected_strategy == "H-historical-cyclic"
        else _PTV2_PARALLEL_RECEIPT_KEYS
        if "execution_receipt" in payload
        else frozenset()
    )
    if set(payload) != _PTV2_BASE_RECEIPT_KEYS | optional:
        raise Task8BuildError(f"{expected_strategy} tokenized receipt schema is not exact")
    claimed = payload.get("receipt_sha256")
    body = {key: value for key, value in payload.items() if key != "receipt_sha256"}
    if (
        payload.get("schema_version") != 2
        or payload.get("strategy") != expected_strategy
        or not isinstance(claimed, str)
        or _SHA256.fullmatch(claimed) is None
        or hashlib.sha256(_canonical_json(body)).hexdigest() != claimed
        or raw != _canonical_json(payload) + b"\n"
    ):
        raise Task8BuildError(f"{expected_strategy} tokenized receipt authentication failed")
    integer_fields = (
        "occurrence_count",
        "trainer_epochs",
        "assistant_tokens",
        "serialized_tokens",
        "packed_sequence_lower_bound",
        "unique_prompt_count",
        "natural_duplicate_count",
        "constructed_repeat_count",
        "database_bytes",
    )
    integers = {
        name: _exact_positive_int(
            payload.get(name),
            f"{expected_strategy} {name}",
            allow_zero=name
            in {"trainer_epochs", "natural_duplicate_count", "constructed_repeat_count"},
        )
        for name in integer_fields
    }
    digest_fields = (
        "tokenizer_sha256",
        "chat_template_sha256",
        "assistant_loss_target_sha256",
        "training_config_sha256",
        "source_response_root_sha256",
        "ordered_occurrences_sha256",
        "selection_sha256",
        "base_occurrence_multiplicity_sha256",
        "database_sha256",
    )
    for name in digest_fields:
        value = payload.get(name)
        if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
            raise Task8BuildError(f"{expected_strategy} {name} is invalid")
    if payload.get("database_path") != "records.sqlite3":
        raise Task8BuildError(f"{expected_strategy} tokenized database path is invalid")
    database = _authenticate_descriptor(
        receipt_path,
        {
            "path": "records.sqlite3",
            "bytes": integers["database_bytes"],
            "sha256": payload["database_sha256"],
        },
        f"{expected_strategy} tokenized database",
    )
    if optional == _PTV2_PARALLEL_RECEIPT_KEYS:
        execution = payload.get("execution_receipt")
        source_bytes = payload.get("source_index_bytes")
        source_sha256 = payload.get("source_index_sha256")
        if (
            not isinstance(execution, Mapping)
            or isinstance(source_bytes, bool)
            or not isinstance(source_bytes, int)
            or source_bytes < 1
            or not isinstance(source_sha256, str)
            or _SHA256.fullmatch(source_sha256) is None
        ):
            raise Task8BuildError(f"{expected_strategy} parallel lineage is malformed")
        _authenticate_descriptor(receipt_path, execution, f"{expected_strategy} execution receipt")
    raw_milestones = payload.get("milestone_occurrences")
    milestone_count = len(raw_milestones) if isinstance(raw_milestones, list) else -1
    milestones = _exact_int_tuple(
        raw_milestones, milestone_count, f"{expected_strategy} milestone occurrences"
    )
    steps = _exact_int_tuple(
        payload.get("milestone_steps"), len(milestones), f"{expected_strategy} milestone steps"
    )
    if not milestones or any(value < 1 for value in (*milestones, *steps)):
        raise Task8BuildError(f"{expected_strategy} milestone schedule is invalid")
    segment_occurrences = _exact_int_tuple(
        payload.get("segment_occurrences"), 2, f"{expected_strategy} segment occurrences"
    )
    segment_steps = _exact_int_tuple(
        payload.get("segment_steps"), 2, f"{expected_strategy} segment steps"
    )
    cumulative_steps = _exact_int_tuple(
        payload.get("cumulative_segment_steps"),
        2,
        f"{expected_strategy} cumulative segment steps",
    )
    final_valid = _exact_int_tuple(
        payload.get("segment_final_valid_occurrences"),
        2,
        f"{expected_strategy} final valid occurrences",
    )
    historical_values: dict[str, str | None] = {
        "historical_parent_receipt_sha256": None,
        "historical_parent_tokenized_sha256": None,
        "historical_prefix_record_stream_sha256": None,
    }
    if expected_strategy == "H-historical-cyclic":
        if payload.get("historical_prefix_occurrence_count") != integers["occurrence_count"]:
            raise Task8BuildError("historical prefix occurrence count is inconsistent")
        for name in historical_values:
            value = payload.get(name)
            if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
                raise Task8BuildError(f"historical {name} is invalid")
            historical_values[name] = value
    return PTV2OnePassCorpus(
        strategy=expected_strategy,
        occurrence_count=integers["occurrence_count"],
        trainer_epochs=integers["trainer_epochs"],
        assistant_tokens=integers["assistant_tokens"],
        tokenizer_sha256=str(payload["tokenizer_sha256"]),
        chat_template_sha256=str(payload["chat_template_sha256"]),
        assistant_loss_target_sha256=str(payload["assistant_loss_target_sha256"]),
        training_config_sha256=str(payload["training_config_sha256"]),
        source_response_root_sha256=str(payload["source_response_root_sha256"]),
        ordered_occurrences_sha256=str(payload["ordered_occurrences_sha256"]),
        selection_sha256=str(payload["selection_sha256"]),
        unique_prompt_count=integers["unique_prompt_count"],
        natural_duplicate_count=integers["natural_duplicate_count"],
        constructed_repeat_count=integers["constructed_repeat_count"],
        serialized_tokens=integers["serialized_tokens"],
        packed_sequence_lower_bound=integers["packed_sequence_lower_bound"],
        milestone_occurrences=milestones,
        milestone_steps=steps,
        segment_occurrences=segment_occurrences,
        segment_steps=segment_steps,
        cumulative_segment_steps=cumulative_steps,
        segment_final_valid_occurrences=final_valid,
        tokenized_path=str(database),
        tokenized_sha256=str(payload["database_sha256"]),
        receipt_path=str(receipt_path),
        receipt_sha256=claimed,
        **historical_values,
    )


def _verify_existing_historical_derivation(prefix: Any, historical: Any) -> None:
    """Prove that an adopted H database is byte-for-byte the authenticated A prefix."""
    if (
        historical.historical_parent_receipt_sha256 != prefix.receipt_sha256
        or historical.historical_parent_tokenized_sha256 != prefix.tokenized_sha256
        or historical.occurrence_count != prefix.segment_occurrences[0]
    ):
        raise Task8BuildError("historical corpus is not derived from the pinned A baseline")
    left = sqlite3.connect(f"file:{Path(prefix.tokenized_path)}?mode=ro", uri=True)
    right = sqlite3.connect(f"file:{Path(historical.tokenized_path)}?mode=ro", uri=True)
    digest = hashlib.sha256()
    try:
        query = (
            "SELECT ordinal,prompt_uuid,source_identity_sha256,source_row,cell,language,"
            "reuse_index,input_ids_json,loss_mask_json,assistant_tokens FROM records "
            "WHERE ordinal<? ORDER BY ordinal"
        )
        expected = left.execute(query, (historical.occurrence_count,))
        observed = right.execute(query, (historical.occurrence_count + 1,))
        count = 0
        for a_row, h_row in zip_longest(expected, observed, fillvalue=None):
            if a_row is None or h_row is None or a_row != h_row:
                raise Task8BuildError("historical corpus differs from the pinned A prefix")
            digest.update(_canonical_json(list(a_row)) + b"\n")
            count += 1
        if (
            count != historical.occurrence_count
            or digest.hexdigest() != historical.historical_prefix_record_stream_sha256
        ):
            raise Task8BuildError("historical corpus record-stream lineage is invalid")
    finally:
        right.close()
        left.close()


def materialize_task8_ab_h_completion(
    *,
    a_tokenized_receipt: Path,
    a_tokenized_receipt_sha256: str,
    b_tokenized_receipt: Path,
    b_tokenized_receipt_sha256: str,
    study_root: Path,
    workers: int = 96,
    expected_completion_receipt_sha256: str | None = None,
) -> Any:
    """Build or authenticate the complete six-bundle A/B/H Task8 exposure study."""
    from build_assistant_token_views import (  # pyright: ignore[reportMissingImports]
        build_ptv2_study_exposures,
        derive_ptv2_historical_corpus,
    )

    root = _canonical_absolute_path(study_root, "PTV2 study root")
    prefix = _load_pinned_ptv2_corpus(
        receipt_path=Path(a_tokenized_receipt),
        receipt_file_sha256=a_tokenized_receipt_sha256,
        expected_strategy="A-repair",
        study_root=root,
    )
    balanced = _load_pinned_ptv2_corpus(
        receipt_path=Path(b_tokenized_receipt),
        receipt_file_sha256=b_tokenized_receipt_sha256,
        expected_strategy="B-balanced",
        study_root=root,
    )
    historical_receipt = root / "h-historical-cyclic-tokenized" / "TOKENIZED.json"
    if os.path.lexists(historical_receipt.parent):
        raw_h = _read_regular_nofollow(
            historical_receipt, "historical tokenized receipt", max_bytes=_MAX_RECEIPT_BYTES
        )
        historical = _load_pinned_ptv2_corpus(
            receipt_path=historical_receipt,
            receipt_file_sha256=hashlib.sha256(raw_h).hexdigest(),
            expected_strategy="H-historical-cyclic",
            study_root=root,
        )
        _verify_existing_historical_derivation(prefix, historical)
    else:
        historical = derive_ptv2_historical_corpus(prefix, root)
    return build_ptv2_study_exposures(
        prefix,
        balanced,
        historical,
        scientific_tokens=256_000_000,
        runtime_screen_tokens=64_000_000,
        workers=workers,
        expected_completion_receipt_sha256=expected_completion_receipt_sha256,
    )


def _validate_typed_task9_receipt(
    receipt_path: Path, payload: Mapping[str, Any], task9_source_commit: str
) -> None:
    from specdec_publication import (  # pyright: ignore[reportMissingImports]
        PublicationError,
        _role_file_descriptors,
        _validate_ptv2_selection_policy,
        _validate_task9_execution_receipt,
    )

    claimed_root = payload.get("root_sha256")
    root_body = {key: value for key, value in payload.items() if key != "root_sha256"}
    if claimed_root != hashlib.sha256(_canonical_json(root_body)).hexdigest():
        raise Task8BuildError("Task9 selection root digest is invalid")
    try:
        descriptors = _role_file_descriptors("selection", payload)
        files: list[tuple[str, Path, int, str]] = []
        for descriptor in descriptors:
            if not isinstance(descriptor, Mapping):
                raise Task8BuildError("Task9 declared file descriptor is malformed")
            path = _authenticate_descriptor(receipt_path, descriptor, "Task9 declared file")
            relative = str(descriptor.get("path", descriptor.get("staged_path", "")))
            size = descriptor.get("bytes")
            digest = descriptor.get("sha256")
            if not isinstance(size, int) or isinstance(size, bool) or not isinstance(digest, str):
                raise Task8BuildError("Task9 declared file identity is malformed")
            files.append((relative, path, size, digest))
        _validate_ptv2_selection_policy(payload, files)
        _validate_task9_execution_receipt(payload, files)
    except PublicationError as error:
        raise Task8BuildError("Task9 typed selection authentication failed") from error
    descriptor = payload["execution_receipt"]
    execution_path = receipt_path.parent / str(descriptor["path"])
    execution = _read_receipt(
        execution_path,
        str(descriptor["sha256"]),
        "Task9 execution",
    )
    if execution.get("source_commit") != task9_source_commit:
        raise Task8BuildError("Task9 execution source commit does not match its pinned producer")


def load_task9_selection(receipt_path: Path, receipt_sha256: str, task9_source_commit: str) -> Any:
    """Authenticate an exact schema-v3 Task9 selection and reconstruct its materialized view."""
    payload = _read_receipt(receipt_path, receipt_sha256, "selection")
    identity = payload.get("selection_identity")
    descriptor = payload.get("index")
    if (
        payload.get("schema_version") != 3
        or payload.get("strategy") not in {"A-repair", "B-balanced"}
        or payload.get("occurrence_count") != 2_000_000
        or not isinstance(payload.get("execution_receipt"), Mapping)
        or not isinstance(identity, dict)
        or not isinstance(descriptor, dict)
    ):
        raise Task8BuildError("Task8 requires an exact 2M schema-v3 Task9 selection")
    index_path = _authenticate_descriptor(receipt_path, descriptor, "Task9 selection index")
    policy_descriptor = payload.get("policy")
    if not isinstance(policy_descriptor, dict):
        raise Task8BuildError("Task9 selection policy descriptor is missing")
    policy_path = _authenticate_descriptor(receipt_path, policy_descriptor, "Task9 policy")
    try:
        policy = yaml.safe_load(_read_regular_nofollow(policy_path, "Task9 policy"))
    except (OSError, yaml.YAMLError) as error:
        raise Task8BuildError("Task9 selection policy is unreadable") from error
    if not isinstance(policy, dict) or policy.get("trainer_epochs") != 1:
        raise Task8BuildError("Task9 selection must bind one trainer epoch")
    _validate_typed_task9_receipt(receipt_path, payload, task9_source_commit)
    connection = sqlite3.connect(f"file:{index_path}?mode=ro", uri=True)
    try:
        constructed = int(
            connection.execute(
                "SELECT count(*) FROM occurrences WHERE strategy=? AND reuse_index>0",
                (payload["strategy"],),
            ).fetchone()[0]
        )
    finally:
        connection.close()
    unique = identity.get("unique_prompt_count")
    if not isinstance(unique, int) or isinstance(unique, bool):
        raise Task8BuildError("Task9 unique prompt count is invalid")
    natural = payload["occurrence_count"] - unique - constructed
    if natural < 0:
        raise Task8BuildError("Task9 multiplicity counts do not reconcile")
    return SimpleNamespace(
        strategy=payload["strategy"],
        occurrence_count=payload["occurrence_count"],
        trainer_epochs=1,
        unique_prompt_count=unique,
        natural_duplicate_count=natural,
        constructed_repeat_count=constructed,
        index_path=index_path,
        ordered_occurrences_sha256=identity.get("ordered_occurrences_sha256"),
        source_response_root_sha256=identity.get("source_response_root_sha256"),
        selection_sha256=payload.get("selection_sha256"),
    )


def _load_bound_tokenizer(
    receipt_path: Path,
    receipt_sha256: str,
    assistant_loss_target_sha256: str,
) -> tuple[_BoundTokenizer, str, str]:
    from build_specdec_inventory import (  # pyright: ignore[reportMissingImports]
        load_tokenizer_snapshot,
    )
    from transformers import AutoTokenizer  # pyright: ignore[reportMissingImports]

    snapshot = load_tokenizer_snapshot(receipt_path, receipt_sha256)
    tokenizer = AutoTokenizer.from_pretrained(
        snapshot.root, local_files_only=True, trust_remote_code=False
    )
    return (
        _BoundTokenizer(
            tokenizer,
            snapshot.tokenizer_sha256,
            snapshot.chat_template_canonical_json,
            assistant_loss_target_sha256,
        ),
        snapshot.tokenizer_sha256,
        snapshot.chat_template_sha256,
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in (
        "source-receipt",
        "source-receipt-sha256",
        "selection-receipt",
        "selection-receipt-sha256",
        "response-receipt",
        "response-receipt-sha256",
        "rejection-receipt",
        "rejection-receipt-sha256",
        "tokenizer-receipt",
        "tokenizer-receipt-sha256",
        "assistant-loss-target-sha256",
        "training-config-sha256",
        "work-root",
        "materialization-root",
        "publication-root",
        "task9-source-commit",
        "source-commit",
        "job-id",
    ):
        parser.add_argument(f"--{name}", required=True)
    parser.add_argument("--workers", type=int, default=96)
    return parser


def _ab_h_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Materialize authenticated A/B/H Task8 views")
    for name in (
        "a-tokenized-receipt",
        "a-tokenized-receipt-sha256",
        "b-tokenized-receipt",
        "b-tokenized-receipt-sha256",
        "study-root",
    ):
        parser.add_argument(f"--{name}", required=True)
    parser.add_argument("--expected-completion-receipt-sha256")
    parser.add_argument("--workers", type=int, default=96)
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the fail-closed Task8 producer from command-line arguments."""
    effective_argv = sys.argv[1:] if argv is None else argv
    if effective_argv and effective_argv[0] == "--validate-publication-root":
        if len(effective_argv) != 3:
            raise Task8BuildError("publication root validation requires path and prefix")
        validate_durable_publication_root(effective_argv[1], effective_argv[2])
        return 0
    if effective_argv and effective_argv[0] == "--materialize-ab-h":
        args = _ab_h_parser().parse_args(effective_argv[1:])
        views = materialize_task8_ab_h_completion(
            a_tokenized_receipt=Path(args.a_tokenized_receipt),
            a_tokenized_receipt_sha256=args.a_tokenized_receipt_sha256,
            b_tokenized_receipt=Path(args.b_tokenized_receipt),
            b_tokenized_receipt_sha256=args.b_tokenized_receipt_sha256,
            study_root=Path(args.study_root),
            workers=args.workers,
            expected_completion_receipt_sha256=args.expected_completion_receipt_sha256,
        )
        print(
            json.dumps(
                {
                    "completion_receipt_path": views.completion_receipt_path,
                    "completion_receipt_sha256": views.completion_receipt_sha256,
                    "runtime_artifacts": dict(views.runtime_artifacts),
                    "scientific_artifacts": dict(views.scientific_artifacts),
                },
                sort_keys=True,
                separators=(",", ":"),
            )
        )
        return 0
    args = _parser().parse_args(effective_argv)
    selection_receipt = Path(args.selection_receipt)
    view = load_task9_selection(
        selection_receipt, args.selection_receipt_sha256, args.task9_source_commit
    )
    validate_required_role_receipts(
        view=view,
        task9_source_commit=args.task9_source_commit,
        response_receipt=Path(args.response_receipt),
        response_receipt_sha256=args.response_receipt_sha256,
        rejection_receipt=Path(args.rejection_receipt),
        rejection_receipt_sha256=args.rejection_receipt_sha256,
    )
    tokenizer, tokenizer_sha256, template_sha256 = _load_bound_tokenizer(
        Path(args.tokenizer_receipt),
        args.tokenizer_receipt_sha256,
        args.assistant_loss_target_sha256,
    )
    receipt = materialize_task8_publication(
        view=view,
        task9_source_commit=args.task9_source_commit,
        tokenizer=tokenizer,
        tokenizer_sha256=tokenizer_sha256,
        chat_template_sha256=template_sha256,
        assistant_loss_target_sha256=args.assistant_loss_target_sha256,
        training_config_sha256=args.training_config_sha256,
        source_receipt=Path(args.source_receipt),
        source_receipt_sha256=args.source_receipt_sha256,
        selection_receipt=selection_receipt,
        selection_receipt_sha256=args.selection_receipt_sha256,
        response_receipt=Path(args.response_receipt),
        response_receipt_sha256=args.response_receipt_sha256,
        rejection_receipt=Path(args.rejection_receipt),
        rejection_receipt_sha256=args.rejection_receipt_sha256,
        work_root=Path(args.work_root),
        materialization_root=Path(args.materialization_root),
        publication_root=Path(args.publication_root),
        source_commit=args.source_commit,
        job_id=args.job_id,
        workers=args.workers,
    )
    print(json.dumps(receipt.__dict__, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
