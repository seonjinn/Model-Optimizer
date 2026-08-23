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

"""Publish complete, content-addressed SpecDec corpora without replacement."""

from __future__ import annotations

import ctypes
import errno
import hashlib
import json
import os
import platform
import re
import shutil
import stat
from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass
from enum import Enum
from pathlib import Path, PurePosixPath
from typing import Any, Literal
from uuid import uuid4

import pyarrow as pa
import pyarrow.parquet as pq

__all__ = [
    "REQUIRED_ROLES",
    "CorpusBundle",
    "InputArtifact",
    "PublicationError",
    "PublicationPhase",
    "PublicationReceipt",
    "PublicationRecoveryState",
    "publication_recovery_state",
    "publish_bundle",
]

REQUIRED_ROLES = ("source", "selection", "response", "tokenized", "exposure", "rejection")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_JOB_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
_BUFFER_SIZE = 1024 * 1024
_SCHEMA = pa.schema(
    [
        ("prompt_uuid", pa.string()),
        ("domain", pa.string()),
        ("lane", pa.string()),
        ("context_bucket", pa.string()),
        ("input_ids", pa.list_(pa.int64())),
        ("loss_mask", pa.list_(pa.bool_())),
        ("assistant_tokens", pa.int64()),
        ("rejection_reason", pa.string()),
        ("record_json", pa.string()),
    ]
)


class PublicationError(ValueError):
    """A corpus input, partial, or publication failed closed."""


@dataclass(frozen=True)
class InputArtifact:
    """One digest-pinned canonical receipt and its transitively declared files."""

    role: str
    receipt_path: Path
    receipt_sha256: str


@dataclass(frozen=True)
class CorpusBundle:
    """The complete receipt set and replayable record stream for one corpus."""

    artifacts: tuple[InputArtifact, ...]
    rows: Callable[[], Iterator[Mapping[str, Any]]]
    prompt_count: int
    assistant_token_count: int
    quarantine_count: int
    selection_manifest_sha256: str
    artifact_source_commit: str


@dataclass(frozen=True)
class PublicationReceipt:
    """The stable content identity of one atomically published corpus."""

    artifact_id: str
    corpus_manifest_sha256: str
    selection_manifest_sha256: str
    file_count: int
    total_bytes: int
    published_path: str


class PublicationPhase(str, Enum):
    """Last attempted operation when publication recovery became necessary."""

    PARTIAL_SETUP = "partial_setup"
    SHARD_WRITE = "shard_write"
    MANIFEST = "manifest"
    DIRECTORY_FSYNC = "directory_fsync"
    RENAME = "rename"
    PARENT_FSYNC = "parent_fsync"
    QUARANTINE = "quarantine"


@dataclass(frozen=True)
class PathObservation:
    """A no-follow point-in-time pathname observation."""

    path: Path
    status: Literal["present", "absent", "unavailable"]
    device: int | None
    inode: int | None
    mode: int | None
    error: str | None

    @property
    def identity(self) -> tuple[int, int] | None:
        if self.device is None or self.inode is None:
            return None
        return self.device, self.inode


@dataclass(frozen=True)
class PublicationRecoveryState:
    """Typed evidence for a preserved partial or ambiguous published path."""

    phase: PublicationPhase
    partial_path: Path
    destination_path: Path
    expected_artifact_id: str | None
    expected_partial_identity: tuple[int, int] | None
    partial_observation: PathObservation
    destination_observation: PathObservation
    recovery_required: Literal[True] = True


def publication_recovery_state(error: BaseException) -> PublicationRecoveryState:
    """Return typed recovery evidence carried by a publication exception."""
    state = getattr(error, "recovery_state", None)
    if not isinstance(state, PublicationRecoveryState):
        raise ValueError("exception does not carry publication recovery state")
    return state


@dataclass(frozen=True)
class _AuthenticatedArtifact:
    role: str
    receipt_path: Path
    receipt_sha256: str
    files: tuple[tuple[str, Path, int, str], ...]


def publish_bundle(
    bundle: CorpusBundle,
    destination: Path,
    job_id: str,
    *,
    rows_per_shard: int = 10_000,
    _phase_hook: Callable[[str], None] | None = None,
) -> PublicationReceipt:
    """Stream, verify, and atomically publish a complete immutable corpus bundle."""
    destination = Path(destination)
    _validate_options(bundle, destination, job_id, rows_per_shard)
    if os.path.lexists(destination):
        raise FileExistsError(f"immutable publication destination already exists: {destination}")
    artifacts = _authenticate_artifacts(bundle)
    row_identity = _preflight_rows(bundle)
    identity_payload = {
        "schema": "modelopt-specdec-publication-input-v1",
        "artifacts": [
            {"role": artifact.role, "receipt_sha256": artifact.receipt_sha256}
            for artifact in artifacts
        ],
        "artifact_source_commit": bundle.artifact_source_commit,
        "assistant_token_count": bundle.assistant_token_count,
        "prompt_count": bundle.prompt_count,
        "quarantine_count": bundle.quarantine_count,
        "rows_sha256": row_identity,
        "selection_manifest_sha256": bundle.selection_manifest_sha256,
    }
    input_identity = _sha256_bytes(_canonical_json(identity_payload))
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_name(f".{destination.name}.partial-{job_id}")
    phase = PublicationPhase.PARTIAL_SETUP
    expected_partial_identity: tuple[int, int] | None = None
    artifact_id: str | None = None
    try:
        if os.path.lexists(partial):
            if not _valid_partial_identity(partial, input_identity):
                phase = PublicationPhase.QUARANTINE
                _quarantine_partial(partial, destination, job_id)
        if not os.path.lexists(partial):
            partial.mkdir(mode=0o750)
            metadata = os.lstat(partial)
            expected_partial_identity = (metadata.st_dev, metadata.st_ino)
            _write_exclusive_durable(
                partial / "PARTIAL_IDENTITY.json",
                _canonical_json(identity_payload | {"input_identity": input_identity}),
            )
        else:
            observed = os.lstat(partial)
            expected_partial_identity = (observed.st_dev, observed.st_ino)

        _copy_authenticated_inputs(partial, artifacts)
        phase = PublicationPhase.SHARD_WRITE
        shard_descriptors = _write_or_resume_shards(
            partial,
            bundle,
            input_identity=input_identity,
            rows_per_shard=rows_per_shard,
            phase_hook=_phase_hook,
        )
        phase = PublicationPhase.MANIFEST
        _call_hook(_phase_hook, "manifest")
        payload_files = _payload_file_records(partial)
        manifest_payload = {
            "schema": "modelopt-specdec-corpus-manifest-v1",
            "input_identity": input_identity,
            "artifact_source_commit": bundle.artifact_source_commit,
            "selection_manifest_sha256": bundle.selection_manifest_sha256,
            "prompt_count": bundle.prompt_count,
            "assistant_token_count": bundle.assistant_token_count,
            "quarantine_count": bundle.quarantine_count,
            "rows_sha256": row_identity,
            "shards": shard_descriptors,
            "file_count": len(payload_files),
            "total_bytes": sum(record["bytes"] for record in payload_files),
            "files": payload_files,
        }
        manifest_bytes = _canonical_json(manifest_payload)
        manifest_sha256 = _sha256_bytes(manifest_bytes)
        artifact_id = _sha256_bytes(
            _canonical_json(
                {
                    "artifact_source_commit": bundle.artifact_source_commit,
                    "corpus_manifest_sha256": manifest_sha256,
                }
            )
        )
        publication_payload = {
            "artifact_id": artifact_id,
            "artifact_source_commit": bundle.artifact_source_commit,
            "corpus_manifest_sha256": manifest_sha256,
            "selection_manifest_sha256": bundle.selection_manifest_sha256,
            "file_count": len(payload_files),
            "total_bytes": sum(record["bytes"] for record in payload_files),
            "published_path": ".",
            "schema": "modelopt-specdec-publication-receipt-v1",
        }
        _write_or_verify(partial / "CORPUS_MANIFEST.json", manifest_bytes)
        _write_or_verify(partial / "PUBLICATION.json", _canonical_json(publication_payload))
        _verify_complete_partial(partial, manifest_sha256, publication_payload)
        phase = PublicationPhase.DIRECTORY_FSYNC
        _call_hook(_phase_hook, "directory_fsync")
        _fsync_tree_directories(partial)
        phase = PublicationPhase.RENAME
        _call_hook(_phase_hook, "rename")
        if _observe(partial).identity != expected_partial_identity:
            raise PublicationError("publication partial inode changed before rename")
        _rename_no_replace(partial, destination)
        if _observe(destination).identity != expected_partial_identity:
            raise PublicationError("published destination inode does not match created partial")
        phase = PublicationPhase.PARENT_FSYNC
        _fsync_directory(destination.parent)
    except BaseException as error:
        state = PublicationRecoveryState(
            phase=phase,
            partial_path=partial,
            destination_path=destination,
            expected_artifact_id=artifact_id,
            expected_partial_identity=expected_partial_identity,
            partial_observation=_observe(partial),
            destination_observation=_observe(destination),
        )
        setattr(error, "recovery_state", state)
        raise
    return PublicationReceipt(
        artifact_id=publication_payload["artifact_id"],
        corpus_manifest_sha256=publication_payload["corpus_manifest_sha256"],
        selection_manifest_sha256=publication_payload["selection_manifest_sha256"],
        file_count=publication_payload["file_count"],
        total_bytes=publication_payload["total_bytes"],
        published_path=str(destination),
    )


def _validate_options(
    bundle: CorpusBundle, destination: Path, job_id: str, rows_per_shard: int
) -> None:
    if not destination.name or destination.name in {".", ".."}:
        raise PublicationError("destination must name one artifact directory")
    if _JOB_ID.fullmatch(job_id) is None:
        raise PublicationError("job_id must be a safe non-empty name")
    if (
        isinstance(rows_per_shard, bool)
        or not isinstance(rows_per_shard, int)
        or rows_per_shard < 1
    ):
        raise PublicationError("rows_per_shard must be a positive integer")
    for label, value in (
        ("prompt_count", bundle.prompt_count),
        ("assistant_token_count", bundle.assistant_token_count),
        ("quarantine_count", bundle.quarantine_count),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise PublicationError(f"{label} must be a nonnegative integer")
    _require_digest("selection manifest", bundle.selection_manifest_sha256)
    if not re.fullmatch(r"[0-9a-f]{40}", bundle.artifact_source_commit):
        raise PublicationError("artifact source commit must be an exact lowercase Git SHA")
    if not callable(bundle.rows):
        raise PublicationError("rows must be a replayable iterator factory")


def _authenticate_artifacts(bundle: CorpusBundle) -> tuple[_AuthenticatedArtifact, ...]:
    by_role = {artifact.role: artifact for artifact in bundle.artifacts}
    if len(by_role) != len(bundle.artifacts) or set(by_role) != set(REQUIRED_ROLES):
        raise PublicationError(f"artifact roles must be exactly {REQUIRED_ROLES}")
    authenticated: list[_AuthenticatedArtifact] = []
    allowed_by_root: dict[Path, set[Path]] = {}
    for role in REQUIRED_ROLES:
        artifact = by_role[role]
        _require_digest(f"{artifact.role} receipt", artifact.receipt_sha256)
        receipt = artifact.receipt_path
        if receipt.is_symlink() or not receipt.is_file():
            raise PublicationError(f"{artifact.role} receipt is not a regular file")
        raw = receipt.read_bytes()
        if _sha256_bytes(raw) != artifact.receipt_sha256:
            raise PublicationError(f"{artifact.role} receipt SHA-256 mismatch")
        payload = _canonical_document(raw, f"{artifact.role} receipt")
        claimed = payload.get("receipt_sha256")
        without_claim = {key: value for key, value in payload.items() if key != "receipt_sha256"}
        if claimed is not None and claimed != _sha256_bytes(_canonical_json(without_claim)):
            raise PublicationError(f"{artifact.role} self digest mismatch")
        if payload.get("role") not in {None, artifact.role}:
            raise PublicationError(f"{artifact.role} receipt role mismatch")
        descriptors = payload.get("files")
        if not isinstance(descriptors, list) or not descriptors:
            raise PublicationError(f"{artifact.role} receipt has no declared files")
        files: list[tuple[str, Path, int, str]] = []
        root = receipt.parent.resolve(strict=True)
        allowed = allowed_by_root.setdefault(root, set())
        allowed.add(receipt.resolve(strict=True))
        for descriptor in descriptors:
            relative, size, digest = _file_descriptor(descriptor, artifact.role)
            unresolved = receipt.parent / relative
            if unresolved.is_symlink():
                raise PublicationError(f"{artifact.role} declared file is a symlink")
            try:
                path = unresolved.resolve(strict=True)
                metadata = path.stat(follow_symlinks=False)
            except OSError as error:
                raise PublicationError(f"{artifact.role} declared file is missing") from error
            if (
                not path.is_relative_to(root)
                or not stat.S_ISREG(metadata.st_mode)
                or metadata.st_size != size
                or _sha256_file(path) != digest
            ):
                raise PublicationError(f"{artifact.role} declared file authentication failed")
            allowed.add(path)
            files.append((relative, path, size, digest))
        authenticated.append(
            _AuthenticatedArtifact(artifact.role, receipt, artifact.receipt_sha256, tuple(files))
        )
    for root, allowed in allowed_by_root.items():
        actual = set(_regular_tree_files(root))
        if actual != allowed:
            raise PublicationError("input receipt tree contains missing or extra files")
    if bundle.selection_manifest_sha256 != authenticated[1].receipt_sha256:
        raise PublicationError("selection manifest SHA-256 is not the selection receipt")
    return tuple(authenticated)


def _preflight_rows(bundle: CorpusBundle) -> str:
    digest = hashlib.sha256()
    prompts = tokens = quarantines = 0
    for row in bundle.rows():
        normalized = _normalize_row(row)
        digest.update(_canonical_json(normalized))
        if normalized["rejection_reason"] is None:
            prompts += 1
        else:
            quarantines += 1
        tokens += normalized["assistant_tokens"]
    if prompts != bundle.prompt_count:
        raise PublicationError("prompt count does not reconcile with streamed rows")
    if tokens != bundle.assistant_token_count:
        raise PublicationError("assistant token count does not reconcile with streamed rows")
    if quarantines != bundle.quarantine_count:
        raise PublicationError("quarantine count does not reconcile with streamed rows")
    return digest.hexdigest()


def _write_or_resume_shards(
    partial: Path,
    bundle: CorpusBundle,
    *,
    input_identity: str,
    rows_per_shard: int,
    phase_hook: Callable[[str], None] | None,
) -> list[dict[str, Any]]:
    shards_dir = partial / "shards"
    shards_dir.mkdir(exist_ok=True)
    prior = _load_shard_states(partial, input_identity)
    descriptors: list[dict[str, Any]] = []
    batch: list[dict[str, Any]] = []
    shard_index = 0
    for row in bundle.rows():
        batch.append(_normalize_row(row))
        if len(batch) == rows_per_shard:
            descriptors.append(
                _materialize_shard(
                    partial, shards_dir, shard_index, batch, prior, input_identity, phase_hook
                )
            )
            shard_index += 1
            batch = []
    if batch:
        descriptors.append(
            _materialize_shard(
                partial, shards_dir, shard_index, batch, prior, input_identity, phase_hook
            )
        )
        shard_index += 1
    if set(prior) - set(range(shard_index)):
        raise PublicationError("partial contains shards beyond the current input stream")
    return descriptors


def _materialize_shard(
    partial: Path,
    shards_dir: Path,
    index: int,
    rows: list[dict[str, Any]],
    prior: dict[int, dict[str, Any]],
    input_identity: str,
    phase_hook: Callable[[str], None] | None,
) -> dict[str, Any]:
    name = f"part-{index:06d}.parquet"
    path = shards_dir / name
    if index in prior:
        descriptor = prior[index]["shard"]
        _verify_file_record(partial, descriptor)
        if descriptor["row_count"] != len(rows):
            raise PublicationError("resumed shard row count mismatch")
        return descriptor
    _call_hook(phase_hook, "shard_open")
    if os.path.lexists(path):
        raise PublicationError("partial has an unauthenticated shard")
    table = pa.Table.from_pylist(rows, schema=_SCHEMA)
    with path.open("xb") as output:
        pq.write_table(table, output, compression="zstd")
        output.flush()
        os.fsync(output.fileno())
    descriptor = {
        "path": f"shards/{name}",
        "bytes": path.stat().st_size,
        "sha256": _sha256_file(path),
        "row_count": len(rows),
    }
    _verify_file_record(partial, descriptor)
    state = {
        "schema": "modelopt-specdec-partial-shard-v1",
        "input_identity": input_identity,
        "shard_index": index,
        "shard": descriptor,
    }
    _write_exclusive_durable(partial / f"SHARD_STATE-{index:06d}.json", _canonical_json(state))
    _call_hook(phase_hook, "shard_closed")
    return descriptor


def _load_shard_states(partial: Path, input_identity: str) -> dict[int, dict[str, Any]]:
    states: dict[int, dict[str, Any]] = {}
    for path in sorted(partial.glob("SHARD_STATE-*.json")):
        payload = _canonical_document(path.read_bytes(), "partial shard state")
        index = payload.get("shard_index")
        if (
            payload.get("schema") != "modelopt-specdec-partial-shard-v1"
            or payload.get("input_identity") != input_identity
            or not isinstance(index, int)
            or index != len(states)
        ):
            raise PublicationError("partial shard state does not match input identity")
        states[index] = payload
    declared = {Path(state["shard"]["path"]).name for state in states.values()}
    actual = {path.name for path in (partial / "shards").glob("*")}
    if actual != declared:
        raise PublicationError("partial contains unauthenticated shard files")
    return states


def _normalize_row(row: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(row, Mapping):
        raise PublicationError("corpus row must be a mapping")
    required = ("prompt_uuid", "domain", "lane", "context_bucket", "input_ids", "loss_mask")
    if any(not isinstance(row.get(field), str) for field in required[:4]):
        raise PublicationError("corpus row identity fields must be strings")
    input_ids = row.get("input_ids")
    loss_mask = row.get("loss_mask")
    if (
        not isinstance(input_ids, list)
        or not isinstance(loss_mask, list)
        or len(input_ids) != len(loss_mask)
        or any(isinstance(value, bool) or not isinstance(value, int) for value in input_ids)
        or any(not isinstance(value, bool) for value in loss_mask)
    ):
        raise PublicationError("input IDs and loss mask must be aligned typed lists")
    assistant_tokens = row.get("assistant_tokens")
    if assistant_tokens != sum(loss_mask):
        raise PublicationError("assistant token count does not match loss mask")
    rejection = row.get("rejection_reason")
    if rejection is not None and not isinstance(rejection, str):
        raise PublicationError("rejection reason must be a string or null")
    canonical = json.dumps(dict(row), sort_keys=True, separators=(",", ":"))
    return {
        "prompt_uuid": row["prompt_uuid"],
        "domain": row["domain"],
        "lane": row["lane"],
        "context_bucket": row["context_bucket"],
        "input_ids": input_ids,
        "loss_mask": loss_mask,
        "assistant_tokens": assistant_tokens,
        "rejection_reason": rejection,
        "record_json": canonical,
    }


def _copy_authenticated_inputs(
    partial: Path, artifacts: tuple[_AuthenticatedArtifact, ...]
) -> None:
    for artifact in artifacts:
        root = partial / "inputs" / artifact.role
        root.mkdir(parents=True, exist_ok=True)
        _copy_or_verify(artifact.receipt_path, root / "receipt.json", artifact.receipt_sha256)
        for relative, source, _size, digest in artifact.files:
            destination = root / "files" / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            _copy_or_verify(source, destination, digest)


def _copy_or_verify(source: Path, destination: Path, digest: str) -> None:
    if os.path.lexists(destination):
        if (
            destination.is_symlink()
            or not destination.is_file()
            or _sha256_file(destination) != digest
        ):
            raise PublicationError("existing partial input copy is not immutable")
        return
    with source.open("rb") as input_stream, destination.open("xb") as output:
        shutil.copyfileobj(input_stream, output, _BUFFER_SIZE)
        output.flush()
        os.fsync(output.fileno())
    if _sha256_file(destination) != digest:
        raise PublicationError("reread input copy SHA-256 mismatch")


def _payload_file_records(partial: Path) -> list[dict[str, Any]]:
    excluded = {"CORPUS_MANIFEST.json", "PUBLICATION.json"}
    records: list[dict[str, Any]] = []
    for path in _regular_tree_files(partial):
        relative = path.relative_to(partial).as_posix()
        if relative in excluded:
            continue
        records.append(
            {"path": relative, "bytes": path.stat().st_size, "sha256": _sha256_file(path)}
        )
    records.sort(key=lambda record: str(record["path"]))
    return records


def _verify_complete_partial(partial: Path, manifest_sha256: str, receipt: dict[str, Any]) -> None:
    manifest_path = partial / "CORPUS_MANIFEST.json"
    if _sha256_file(manifest_path) != manifest_sha256:
        raise PublicationError("corpus manifest reread failed")
    manifest = _canonical_document(manifest_path.read_bytes(), "corpus manifest")
    expected_paths = [record["path"] for record in manifest["files"]]
    actual_payload = _payload_file_records(partial)
    if actual_payload != manifest["files"] or len(expected_paths) != len(set(expected_paths)):
        raise PublicationError("corpus manifest file reconciliation failed")
    for record in manifest["files"]:
        _verify_file_record(partial, record)
    publication_path = partial / "PUBLICATION.json"
    if _canonical_document(publication_path.read_bytes(), "publication receipt") != receipt:
        raise PublicationError("publication receipt reread failed")
    if sum(shard["row_count"] for shard in manifest["shards"]) != (
        manifest["prompt_count"] + manifest["quarantine_count"]
    ):
        raise PublicationError("published shard row count reconciliation failed")


def _verify_file_record(root: Path, record: Mapping[str, Any]) -> None:
    relative = _safe_relative(record.get("path"), "file record")
    unresolved = root / relative
    if unresolved.is_symlink():
        raise PublicationError("published file cannot be a symlink")
    try:
        path = unresolved.resolve(strict=True)
        metadata = path.stat(follow_symlinks=False)
    except OSError as error:
        raise PublicationError("published file is missing") from error
    if (
        not path.is_relative_to(root.resolve(strict=True))
        or not stat.S_ISREG(metadata.st_mode)
        or metadata.st_size != record.get("bytes")
        or _sha256_file(path) != record.get("sha256")
    ):
        raise PublicationError("published file digest reconciliation failed")


def _valid_partial_identity(partial: Path, expected: str) -> bool:
    if partial.is_symlink() or not partial.is_dir():
        return False
    identity = partial / "PARTIAL_IDENTITY.json"
    try:
        payload = _canonical_document(identity.read_bytes(), "partial identity")
    except (OSError, PublicationError):
        return False
    return payload.get("input_identity") == expected


def _quarantine_partial(partial: Path, destination: Path, job_id: str) -> None:
    before = _observe(partial)
    quarantine = destination.with_name(f".{destination.name}.quarantine-{job_id}-{uuid4().hex}")
    if before.identity is None:
        raise PublicationError("stale partial cannot be safely observed")
    _rename_no_replace(partial, quarantine)
    if _observe(quarantine).identity != before.identity:
        raise PublicationError("quarantined partial inode does not match observed partial")
    _fsync_directory(destination.parent)


def _file_descriptor(value: object, role: str) -> tuple[str, int, str]:
    if not isinstance(value, dict):
        raise PublicationError(f"{role} file descriptor is malformed")
    relative = _safe_relative(value.get("path"), f"{role} file")
    size = value.get("bytes", value.get("byte_count", value.get("size")))
    digest = value.get("sha256")
    if isinstance(size, bool) or not isinstance(size, int) or size < 0:
        raise PublicationError(f"{role} file descriptor has invalid bytes")
    if not isinstance(digest, str):
        raise PublicationError(f"{role} file descriptor has invalid SHA-256")
    _require_digest(f"{role} file", digest)
    return relative, size, digest


def _safe_relative(value: object, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise PublicationError(f"{label} path must be a non-empty string")
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts or path.as_posix() != value:
        raise PublicationError(f"{label} path is unsafe")
    return value


def _regular_tree_files(root: Path) -> Iterator[Path]:
    for directory, directory_names, file_names in os.walk(root, followlinks=False):
        base = Path(directory)
        for name in directory_names:
            path = base / name
            if path.is_symlink():
                raise PublicationError(f"tree contains symlink directory: {path}")
        for name in file_names:
            path = base / name
            metadata = path.stat(follow_symlinks=False)
            if path.is_symlink() or not stat.S_ISREG(metadata.st_mode):
                raise PublicationError(f"tree contains non-regular file: {path}")
            yield path.resolve(strict=True)


def _canonical_document(raw: bytes, label: str) -> dict[str, Any]:
    try:
        payload = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise PublicationError(f"{label} is not JSON") from error
    if not isinstance(payload, dict) or raw != _canonical_json(payload):
        raise PublicationError(f"{label} is not canonical JSON")
    return payload


def _write_exclusive_durable(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as output:
        output.write(data)
        output.flush()
        os.fsync(output.fileno())


def _write_or_verify(path: Path, data: bytes) -> None:
    if os.path.lexists(path):
        if path.is_symlink() or not path.is_file() or path.read_bytes() != data:
            raise PublicationError(f"existing immutable metadata differs: {path.name}")
        return
    _write_exclusive_durable(path, data)
    if path.read_bytes() != data:
        raise PublicationError(f"metadata reread failed: {path.name}")


def _fsync_tree_directories(root: Path) -> None:
    directories = [Path(directory) for directory, _, _ in os.walk(root, followlinks=False)]
    for directory in reversed(directories):
        _fsync_directory(directory)


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _rename_no_replace(source: Path, destination: Path) -> None:
    library = ctypes.CDLL(None, use_errno=True)
    source_bytes = os.fsencode(source)
    destination_bytes = os.fsencode(destination)
    if platform.system() == "Linux":
        try:
            rename = library.renameat2
        except AttributeError as error:
            raise PublicationError("atomic no-replace rename is unavailable") from error
        rename.argtypes = [
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_uint,
        ]
        rename.restype = ctypes.c_int
        result = rename(-100, source_bytes, -100, destination_bytes, 1)
    elif platform.system() == "Darwin":
        rename = library.renamex_np
        rename.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_uint]
        rename.restype = ctypes.c_int
        result = rename(source_bytes, destination_bytes, 0x00000004)
    else:
        raise PublicationError(f"atomic no-replace rename is unsupported on {platform.system()}")
    if result == 0:
        return
    error_number = ctypes.get_errno()
    if error_number == errno.EEXIST:
        raise FileExistsError(error_number, os.strerror(error_number), destination)
    raise PublicationError(f"atomic no-replace rename failed: {os.strerror(error_number)}")


def _observe(path: Path) -> PathObservation:
    try:
        metadata = os.lstat(path)
    except FileNotFoundError:
        return PathObservation(path, "absent", None, None, None, None)
    except OSError as error:
        return PathObservation(path, "unavailable", None, None, None, str(error))
    return PathObservation(
        path, "present", metadata.st_dev, metadata.st_ino, metadata.st_mode, None
    )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(_BUFFER_SIZE), b""):
            digest.update(block)
    return digest.hexdigest()


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _require_digest(label: str, value: object) -> None:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise PublicationError(f"{label} SHA-256 is invalid")


def _canonical_json(value: object) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode()


def _call_hook(hook: Callable[[str], None] | None, phase: str) -> None:
    if hook is not None:
        hook(phase)
