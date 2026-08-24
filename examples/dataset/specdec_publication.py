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
import math
import os
import platform
import re
import shutil
import sqlite3
import stat
from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass
from enum import Enum
from pathlib import Path, PurePosixPath
from typing import Any, Literal
from uuid import uuid4

import pyarrow as pa
import pyarrow.parquet as pq
import yaml

__all__ = [
    "REQUIRED_ROLES",
    "CorpusBundle",
    "InputArtifact",
    "PublicationError",
    "PublicationPhase",
    "PublicationReceipt",
    "PublicationRecoveryState",
    "Task5ExecutionReceiptIdentity",
    "publication_recovery_state",
    "publish_bundle",
    "validate_task5_execution_receipt",
]

REQUIRED_ROLES = ("source", "selection", "response", "tokenized", "exposure", "rejection")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_JOB_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
_BUFFER_SIZE = 1024 * 1024
_PARALLEL_THREAD_ENVIRONMENT = frozenset(
    {
        "ARROW_NUM_THREADS",
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    }
)
_PTV2_SELECTION_IDENTITY_KEYS = frozenset(
    {
        "strategy",
        "policy_sha256",
        "occurrence_count",
        "unique_prompt_count",
        "cell_occurrence_counts",
        "multilingual_occurrence_counts",
        "repair_complement_counts",
        "uuid_multiplicity_histogram",
        "source_occurrence_multiplicity_histogram",
        "ordered_occurrences_sha256",
        "ordered_prompt_uuids_sha256",
        "source_response_root_sha256",
        "occurrence_multiplicity_sha256",
        "trust_root_sha256",
    }
)
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


@dataclass(frozen=True)
class Task5ExecutionReceiptIdentity:
    """Authenticated lineage roots carried by a Task5 p96 execution receipt."""

    source_commit: str
    source_manifest_sha256: str
    candidate_inventory_sha256: str
    selection_sha256: str
    receipt_sha256: str


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
    quarantine_path: Path | None
    quarantine_observation: PathObservation | None
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
    quarantine_path: Path | None = None
    try:
        if os.path.lexists(partial):
            expected_partial_identity = _observe(partial).identity
            if not _valid_partial_identity(partial, input_identity) or not _partial_tree_safe(
                partial, artifacts
            ):
                phase = PublicationPhase.QUARANTINE
                quarantine_path = destination.with_name(
                    f".{destination.name}.quarantine-{job_id}-{uuid4().hex}"
                )
                _quarantine_partial(partial, quarantine_path, destination.parent)
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
            quarantine_path=quarantine_path,
            quarantine_observation=(None if quarantine_path is None else _observe(quarantine_path)),
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
    payloads: dict[str, dict[str, Any]] = {}
    for role in REQUIRED_ROLES:
        artifact = by_role[role]
        _require_digest(f"{artifact.role} receipt", artifact.receipt_sha256)
        receipt = artifact.receipt_path
        if receipt.is_symlink() or not receipt.is_file():
            raise PublicationError(f"{artifact.role} receipt is not a regular file")
        raw = receipt.read_bytes()
        if _sha256_bytes(raw) != artifact.receipt_sha256:
            raise PublicationError(f"{artifact.role} receipt SHA-256 mismatch")
        payload = _receipt_document(raw, artifact.role)
        payloads[artifact.role] = payload
        claimed = payload.get("receipt_sha256")
        without_claim = {key: value for key, value in payload.items() if key != "receipt_sha256"}
        canonical_identity = json.dumps(
            without_claim, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        ).encode()
        if claimed is not None and claimed not in {
            _sha256_bytes(canonical_identity),
            _sha256_bytes(canonical_identity + b"\n"),
        }:
            raise PublicationError(f"{artifact.role} self digest mismatch")
        if payload.get("role") not in {None, artifact.role}:
            raise PublicationError(f"{artifact.role} receipt role mismatch")
        descriptors = _role_file_descriptors(artifact.role, payload)
        files: list[tuple[str, Path, int, str]] = []
        root = receipt.parent.resolve(strict=True)
        for descriptor in descriptors:
            relative, size, digest = _file_descriptor(descriptor, artifact.role, root)
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
            files.append((relative, path, size, digest))
        if artifact.role == "selection" and payload.get("schema_version") == 3:
            _validate_ptv2_selection_policy(payload, files)
            if "execution_receipt" in payload:
                _validate_task9_execution_receipt(payload, files)
        if (
            artifact.role == "tokenized"
            and payload.get("schema_version") == 1
            and payload.get("strategy") in {"A-repair", "B-balanced"}
            and "execution_receipt" in payload
        ):
            _validate_task8_execution_receipt(payload, files)
        if (
            artifact.role == "selection"
            and payload.get("schema_version") == 2
            and payload.get("selection_mode") == "B-prime-only"
        ):
            validate_task5_execution_receipt(payload, artifact.receipt_path.parent)
        authenticated.append(
            _AuthenticatedArtifact(artifact.role, receipt, artifact.receipt_sha256, tuple(files))
        )
    if bundle.selection_manifest_sha256 != authenticated[1].receipt_sha256:
        raise PublicationError("selection manifest SHA-256 is not the selection receipt")
    if payloads["selection"].get("schema_version") == 3:
        _reconcile_ptv2_role_lineage(payloads)
    return tuple(authenticated)


def _reconcile_ptv2_role_lineage(payloads: Mapping[str, Mapping[str, Any]]) -> None:
    """Reconcile Task 9 roots in their forward data-dependency direction."""
    selection = payloads["selection"]
    source = payloads["source"]
    if source.get("source_manifest_sha256") != selection["source_inventory_sha256"]:
        raise PublicationError("PTV2 selection is not bound to the authenticated source inventory")
    expected_selection = selection["selection_sha256"]

    def identity(role: str) -> Mapping[str, Any]:
        payload = payloads[role]
        value = payload.get("identity")
        return value if isinstance(value, Mapping) else payload

    response = identity("response")
    tokenized = identity("tokenized")
    exposure = identity("exposure")
    rejection = identity("rejection")
    for role, payload in (
        ("response", response),
        ("tokenized", tokenized),
        ("exposure", exposure),
        ("rejection", rejection),
    ):
        if payload.get("selection_sha256") != expected_selection:
            raise PublicationError(f"PTV2 {role} receipt is not bound to selection")
    if "execution_receipt" in tokenized and (
        tokenized.get("source_index_sha256") != selection["index"].get("sha256")
        or tokenized.get("source_index_bytes") != selection["index"].get("bytes")
    ):
        raise PublicationError("PTV2 Task8 source index does not reconcile Task9 selection")
    response_root = response.get("source_response_root_sha256")
    if not isinstance(response_root, str) or _SHA256.fullmatch(response_root) is None:
        raise PublicationError("PTV2 response receipt has no source response root")
    for role, payload in (("tokenized", tokenized), ("exposure", exposure)):
        if payload.get("source_response_root_sha256") != response_root:
            raise PublicationError(f"PTV2 {role} receipt does not reconcile response root")
    for key in ("tokenizer_sha256", "chat_template_sha256", "assistant_loss_target_sha256"):
        value = tokenized.get(key)
        if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
            raise PublicationError(f"PTV2 tokenized receipt has no {key}")
        if exposure.get(key) != value:
            raise PublicationError(f"PTV2 exposure receipt does not reconcile {key}")
    tokenized_root = tokenized.get("database_sha256")
    if not isinstance(tokenized_root, str) or _SHA256.fullmatch(tokenized_root) is None:
        raise PublicationError("PTV2 tokenized receipt has no database root")
    if exposure.get("tokenized_sha256") != tokenized_root:
        raise PublicationError("PTV2 exposure receipt does not reconcile tokenized root")


def _validate_ptv2_selection_policy(
    payload: Mapping[str, Any], files: list[tuple[str, Path, int, str]]
) -> None:
    """Bind a semantic policy digest to the separately authenticated YAML bytes."""
    policy_file = next((item for item in files if item[0] == payload["policy"]["path"]), None)
    if policy_file is None:
        raise PublicationError("PTV2 policy descriptor was not authenticated")
    if policy_file[3] != payload["policy_file_sha256"]:
        raise PublicationError("PTV2 policy bytes do not match policy_file_sha256")
    try:
        decoded_policy = yaml.safe_load(policy_file[1].read_bytes())
    except (OSError, yaml.YAMLError) as error:
        raise PublicationError("PTV2 policy bytes are not valid YAML") from error
    if not isinstance(decoded_policy, dict):
        raise PublicationError("PTV2 semantic policy must be a mapping")
    if _sha256_bytes(_identity_json(decoded_policy)) != payload["policy_sha256"]:
        raise PublicationError("PTV2 semantic policy does not match policy_sha256")
    identity = payload["selection_identity"]
    if not isinstance(identity, Mapping) or set(identity) != _PTV2_SELECTION_IDENTITY_KEYS:
        raise PublicationError("PTV2 selection identity does not have the exact schema")
    if _sha256_bytes(_identity_json(identity)) != payload["selection_sha256"]:
        raise PublicationError("PTV2 selection identity does not match selection_sha256")
    trust_roots = payload["trust_roots"]
    expected_trust_roots = {
        "source_inventory_sha256": payload["source_inventory_sha256"],
        "held_out_receipt_sha256": payload["held_out_receipt_sha256"],
    }
    if payload["strategy"] == "A-repair":
        expected_trust_roots |= {
            "baseline_receipt_sha256": payload["baseline_receipt_sha256"],
            "complement_selection_sha256": payload["complement_selection_sha256"],
        }
    if not isinstance(trust_roots, Mapping) or dict(trust_roots) != expected_trust_roots:
        raise PublicationError("PTV2 selection trust-root preimage is invalid")
    for value in trust_roots.values():
        if not isinstance(value, str) or _SHA256.fullmatch(value) is None or value == "0" * 64:
            raise PublicationError("PTV2 selection trust-root preimage is invalid")
    index_file = next((item for item in files if item[0] == payload["index"]["path"]), None)
    if index_file is None:
        raise PublicationError("PTV2 selection index was not authenticated")
    connection: sqlite3.Connection | None = None
    try:
        connection = sqlite3.connect(f"file:{index_file[1]}?mode=ro", uri=True)
        rows = connection.execute(
            "SELECT occurrences.ordinal,occurrences.prompt_uuid,occurrences.source_identity_sha256,"
            "occurrences.source_row,occurrences.cell,occurrences.reuse_index,"
            "occurrences.conversation_sha256,occurrences.assistant_response_sha256,"
            "source_rows.canonical_conversation,source_rows.assistant_response,"
            "source_rows.language FROM occurrences "
            "JOIN source_rows ON occurrences.source_identity_sha256=source_rows.source_identity_sha256 "
            "AND occurrences.source_row=source_rows.source_row WHERE occurrences.strategy=? "
            "ORDER BY occurrences.ordinal",
            (payload["strategy"],),
        )
        semantic = hashlib.sha256()
        prompt_semantic = hashlib.sha256()
        response_semantic = hashlib.sha256()
        count = 0
        cell_counts: dict[str, int] = {}
        language_counts: dict[str, int] = {}
        repair_counts = dict.fromkeys(decoded_policy.get("repair", {}).get("complement", {}), 0)
        historical_count = (
            decoded_policy.get("repair", {}).get("historical", {}).get("occurrences", 0)
        )
        for row in rows:
            occurrence = row[:8]
            if (
                _sha256_bytes(str(row[8]).encode()) != occurrence[6]
                or _sha256_bytes(str(row[9]).encode()) != occurrence[7]
            ):
                raise PublicationError("PTV2 source-row conversation or response hash mismatch")
            semantic.update(_identity_json(list(occurrence)))
            semantic.update(b"\n")
            prompt_semantic.update(_identity_json(occurrence[1]))
            prompt_semantic.update(b"\n")
            response_semantic.update(
                _identity_json([occurrence[2], occurrence[3], occurrence[6], occurrence[7]])
            )
            response_semantic.update(b"\n")
            cell = str(occurrence[4])
            language = str(row[10])
            cell_counts[cell] = cell_counts.get(cell, 0) + 1
            if cell == "multilingual":
                language_counts[language] = language_counts.get(language, 0) + 1
            if payload["strategy"] == "A-repair" and occurrence[0] >= historical_count:
                repair_key = "stem" if cell == "stem" else language
                if repair_key in repair_counts:
                    repair_counts[repair_key] += 1
            count += 1
        unique_count = int(
            connection.execute(
                "SELECT count(DISTINCT prompt_uuid) FROM occurrences WHERE strategy=?",
                (payload["strategy"],),
            ).fetchone()[0]
        )
        multiplicity_semantic = hashlib.sha256()
        for row in connection.execute(
            "SELECT prompt_uuid,source_identity_sha256,source_row,count(*) FROM occurrences "
            "WHERE strategy=? GROUP BY prompt_uuid,source_identity_sha256,source_row "
            "ORDER BY prompt_uuid,source_identity_sha256,source_row",
            (payload["strategy"],),
        ):
            multiplicity_semantic.update(_identity_json(list(row)))
            multiplicity_semantic.update(b"\n")
        uuid_histogram = {
            int(multiplicity): int(row_count)
            for multiplicity, row_count in connection.execute(
                "SELECT multiplicity,count(*) FROM (SELECT prompt_uuid,count(*) AS multiplicity "
                "FROM occurrences WHERE strategy=? GROUP BY prompt_uuid) "
                "GROUP BY multiplicity ORDER BY multiplicity",
                (payload["strategy"],),
            )
        }
        source_histogram = {
            int(multiplicity): int(row_count)
            for multiplicity, row_count in connection.execute(
                "SELECT multiplicity,count(*) FROM (SELECT prompt_uuid,source_identity_sha256,"
                "source_row,count(*) AS multiplicity FROM occurrences WHERE strategy=? "
                "GROUP BY prompt_uuid,source_identity_sha256,source_row) "
                "GROUP BY multiplicity ORDER BY multiplicity",
                (payload["strategy"],),
            )
        }
    except sqlite3.Error as error:
        raise PublicationError("PTV2 selection index semantics are invalid") from error
    finally:
        if connection is not None:
            connection.close()
    if (
        count != payload["occurrence_count"]
        or semantic.hexdigest() != payload["ordered_occurrences_sha256"]
    ):
        raise PublicationError("PTV2 selection index semantics do not match its receipt")
    language_keys = decoded_policy.get("balanced", {}).get("multilingual_occurrences", {})
    recomputed_identity = {
        "strategy": payload["strategy"],
        "policy_sha256": payload["policy_sha256"],
        "occurrence_count": count,
        "unique_prompt_count": unique_count,
        "cell_occurrence_counts": {
            cell: cell_counts.get(cell, 0)
            for cell in ("math", "code", "stem", "chat", "multilingual")
        },
        "multilingual_occurrence_counts": {
            language: language_counts.get(language, 0) for language in language_keys
        },
        "repair_complement_counts": repair_counts if payload["strategy"] == "A-repair" else {},
        "uuid_multiplicity_histogram": uuid_histogram,
        "source_occurrence_multiplicity_histogram": source_histogram,
        "ordered_occurrences_sha256": semantic.hexdigest(),
        "ordered_prompt_uuids_sha256": prompt_semantic.hexdigest(),
        "source_response_root_sha256": response_semantic.hexdigest(),
        "occurrence_multiplicity_sha256": multiplicity_semantic.hexdigest(),
        "trust_root_sha256": _sha256_bytes(_identity_json(trust_roots)),
    }
    if _identity_json(recomputed_identity) != _identity_json(identity):
        raise PublicationError("PTV2 selection identity does not match index semantics")
    shard_semantic = hashlib.sha256()
    shard_count = 0
    occurrence_paths = {descriptor["path"] for descriptor in payload["shards"]}
    for _, path, _, _ in files:
        if path.name not in occurrence_paths:
            continue
        with path.open("rb") as stream:
            for raw in stream:
                try:
                    occurrence = json.loads(raw)
                except json.JSONDecodeError as error:
                    raise PublicationError("PTV2 occurrence shard is not JSONL") from error
                if not isinstance(occurrence, list) or len(occurrence) != 8:
                    raise PublicationError("PTV2 occurrence shard has invalid row shape")
                shard_semantic.update(_identity_json(occurrence))
                shard_semantic.update(b"\n")
                shard_count += 1
    if (
        shard_count != count
        or shard_semantic.hexdigest() != payload["shard_semantic_sha256"]
        or payload["shard_semantic_sha256"] != payload["ordered_occurrences_sha256"]
    ):
        raise PublicationError("PTV2 occurrence shards do not match selection index semantics")


def validate_task5_execution_receipt(
    selection: Mapping[str, Any], publication_root: Path
) -> Task5ExecutionReceiptIdentity:
    """Authenticate and semantically reconcile one Task5 p96 execution receipt."""
    descriptor = selection.get("execution_receipt")
    if (
        not isinstance(descriptor, Mapping)
        or set(descriptor) != {"path", "byte_count", "sha256"}
        or descriptor.get("path") != "EXECUTION_RECEIPT.json"
    ):
        raise PublicationError("Task5 selection execution receipt is missing")
    try:
        root = publication_root.resolve(strict=True)
        relative, expected_bytes, expected_sha256 = _file_descriptor(
            descriptor, "selection", root
        )
        unresolved = publication_root / relative
        if unresolved.is_symlink():
            raise PublicationError("Task5 execution receipt must not be a symlink")
        execution_path = unresolved.resolve(strict=True)
        descriptor_fd = os.open(
            execution_path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
        )
        try:
            initial = os.fstat(descriptor_fd)
            with os.fdopen(os.dup(descriptor_fd), "rb") as stream:
                raw = stream.read()
            final = os.fstat(descriptor_fd)
            pathname = os.lstat(execution_path)
        finally:
            os.close(descriptor_fd)
    except (OSError, ValueError) as error:
        raise PublicationError("Task5 execution receipt was not authenticated") from error
    if (
        not execution_path.is_relative_to(root)
        or not stat.S_ISREG(initial.st_mode)
        or initial.st_size != expected_bytes
        or (initial.st_dev, initial.st_ino, initial.st_size, initial.st_mtime_ns)
        != (final.st_dev, final.st_ino, final.st_size, final.st_mtime_ns)
        or stat.S_ISLNK(pathname.st_mode)
        or (pathname.st_dev, pathname.st_ino) != (final.st_dev, final.st_ino)
        or hashlib.sha256(raw).hexdigest() != expected_sha256
    ):
        raise PublicationError("Task5 execution receipt was not authenticated")
    try:
        execution = json.loads(raw)
    except (OSError, json.JSONDecodeError) as error:
        raise PublicationError("Task5 execution receipt is invalid") from error
    if not isinstance(execution, dict) or raw != _identity_json(execution) + b"\n":
        raise PublicationError("Task5 execution receipt is not canonical")
    claimed = execution.pop("receipt_sha256", None)
    required = {
        "schema_version",
        "source_commit",
        "source_manifest_sha256",
        "declared_shard_count",
        "allocated_cpus",
        "requested_workers",
        "effective_workers",
        "threads_per_worker",
        "thread_environment",
        "started_at_ns",
        "finished_at_ns",
        "elapsed_seconds",
        "accepted_count",
        "quarantine_counts",
        "shards",
        "tokenization_shards",
        "candidate_inventory_sha256",
        "selection_sha256",
    }
    identity = selection.get("identity")
    thread_environment = execution.get("thread_environment")
    shards = execution.get("shards")
    tokenization_shards = execution.get("tokenization_shards")
    started = execution.get("started_at_ns")
    finished = execution.get("finished_at_ns")
    elapsed = execution.get("elapsed_seconds")
    quarantine = execution.get("quarantine_counts")
    source_manifest_sha256 = execution.get("source_manifest_sha256")
    candidate_inventory_sha256 = execution.get("candidate_inventory_sha256")
    selection_source_inventory_sha256 = (
        identity.get("source_inventory_sha256") if isinstance(identity, Mapping) else None
    )
    selection_source_manifest_sha256 = (
        identity.get("source_manifest_sha256") if isinstance(identity, Mapping) else None
    )
    selection_candidate_inventory_sha256 = (
        identity.get("candidate_inventory_sha256") if isinstance(identity, Mapping) else None
    )
    if (
        set(execution) != required
        or claimed != _sha256_bytes(_identity_json(execution))
        or execution.get("schema_version") != 1
        or not isinstance(execution.get("source_commit"), str)
        or re.fullmatch(r"[0-9a-f]{40}", execution["source_commit"]) is None
        or not isinstance(identity, Mapping)
        or not _is_sha256(source_manifest_sha256)
        or not _is_sha256(candidate_inventory_sha256)
        or not _is_sha256(selection_source_inventory_sha256)
        or not _is_sha256(selection_source_manifest_sha256)
        or not _is_sha256(selection_candidate_inventory_sha256)
        or source_manifest_sha256 != selection_source_manifest_sha256
        or candidate_inventory_sha256 != selection_candidate_inventory_sha256
        or candidate_inventory_sha256 != selection_source_inventory_sha256
        or execution.get("selection_sha256") != selection.get("selection_sha256")
        or not _is_sha256(execution.get("selection_sha256"))
        or execution.get("effective_workers") != 96
        or execution.get("allocated_cpus") != 96
        or execution.get("requested_workers") != 96
        or execution.get("declared_shard_count") != 201
        or execution.get("threads_per_worker") != 1
        or not isinstance(thread_environment, dict)
        or set(thread_environment) != _PARALLEL_THREAD_ENVIRONMENT
        or set(thread_environment.values()) != {"1"}
        or not isinstance(started, int)
        or isinstance(started, bool)
        or started < 1
        or not isinstance(finished, int)
        or isinstance(finished, bool)
        or finished < started
        or not isinstance(elapsed, (int, float))
        or isinstance(elapsed, bool)
        or elapsed < 0
        or not isinstance(execution.get("accepted_count"), int)
        or isinstance(execution.get("accepted_count"), bool)
        or not isinstance(selection.get("row_count"), int)
        or isinstance(selection.get("row_count"), bool)
        or execution["accepted_count"] < selection.get("row_count", 0)
        or not isinstance(quarantine, dict)
        or any(
            not isinstance(key, str)
            or isinstance(value, bool)
            or not isinstance(value, int)
            or value < 0
            for key, value in quarantine.items()
        )
        or not _parallel_shards_reconcile(shards, include_spool_identity=True)
        or not _parallel_shards_reconcile(tokenization_shards, include_spool_identity=False)
    ):
        raise PublicationError("Task5 execution receipt does not reconcile")
    assert isinstance(claimed, str)
    assert isinstance(source_manifest_sha256, str)
    assert isinstance(candidate_inventory_sha256, str)
    selection_sha256 = execution["selection_sha256"]
    assert isinstance(selection_sha256, str)
    source_commit = execution["source_commit"]
    assert isinstance(source_commit, str)
    return Task5ExecutionReceiptIdentity(
        source_commit,
        source_manifest_sha256,
        candidate_inventory_sha256,
        selection_sha256,
        claimed,
    )


def _validate_task9_execution_receipt(
    selection: Mapping[str, Any], files: list[tuple[str, Path, int, str]]
) -> None:
    descriptor = selection.get("execution_receipt")
    if not isinstance(descriptor, Mapping) or not isinstance(descriptor.get("path"), str):
        raise PublicationError("Task9 execution receipt descriptor is malformed")
    execution_file = next((item for item in files if item[0] == descriptor["path"]), None)
    if execution_file is None:
        raise PublicationError("Task9 execution receipt was not authenticated")
    try:
        raw = execution_file[1].read_bytes()
        execution = json.loads(raw)
    except (OSError, json.JSONDecodeError) as error:
        raise PublicationError("Task9 execution receipt is invalid") from error
    if not isinstance(execution, dict) or raw != _identity_json(execution) + b"\n":
        raise PublicationError("Task9 execution receipt is not canonical")
    claimed = execution.pop("receipt_sha256", None)
    required = {
        "schema_version",
        "source_commit",
        "source_inventory_sha256",
        "declared_shard_count",
        "allocated_cpus",
        "requested_workers",
        "effective_workers",
        "threads_per_worker",
        "thread_environment",
        "started_at_ns",
        "finished_at_ns",
        "elapsed_seconds",
        "shards",
        "selection_sha256",
    }
    thread_environment = execution.get("thread_environment")
    started = execution.get("started_at_ns")
    finished = execution.get("finished_at_ns")
    elapsed = execution.get("elapsed_seconds")
    shards = execution.get("shards")
    if (
        set(execution) != required
        or claimed != _sha256_bytes(_identity_json(execution))
        or execution.get("schema_version") != 1
        or not isinstance(execution.get("source_commit"), str)
        or re.fullmatch(r"[0-9a-f]{40}", execution["source_commit"]) is None
        or not _is_sha256(execution.get("source_inventory_sha256"))
        or not _is_sha256(selection.get("source_inventory_sha256"))
        or execution.get("source_inventory_sha256") != selection.get("source_inventory_sha256")
        or execution.get("selection_sha256") != selection.get("selection_sha256")
        or not _is_sha256(execution.get("selection_sha256"))
        or execution.get("declared_shard_count") != 201
        or execution.get("allocated_cpus") != 96
        or execution.get("requested_workers") != 96
        or execution.get("effective_workers") != 96
        or execution.get("threads_per_worker") != 1
        or not isinstance(thread_environment, dict)
        or set(thread_environment) != _PARALLEL_THREAD_ENVIRONMENT
        or set(thread_environment.values()) != {"1"}
        or not isinstance(started, int)
        or isinstance(started, bool)
        or started < 1
        or not isinstance(finished, int)
        or isinstance(finished, bool)
        or finished < started
        or not isinstance(elapsed, (int, float))
        or isinstance(elapsed, bool)
        or elapsed < 0
        or not _parallel_shards_reconcile(shards, include_spool_identity=True)
    ):
        raise PublicationError("Task9 execution receipt does not reconcile")


def _is_sha256(value: object) -> bool:
    return isinstance(value, str) and _SHA256.fullmatch(value) is not None


def _is_nonnegative_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


def _is_nonnegative_number(value: object) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
        and value >= 0
    )


def _parallel_shards_reconcile(value: object, *, include_spool_identity: bool) -> bool:
    if not isinstance(value, list) or len(value) != 201:
        return False
    common_keys = {"index", "row_count", "elapsed_seconds", "worker_pid"}
    expected_keys = (
        common_keys | {"spool_path", "spool_bytes", "spool_sha256"}
        if include_spool_identity
        else common_keys
    )
    seen: set[int] = set()
    for item in value:
        if not isinstance(item, dict) or set(item) != expected_keys:
            return False
        index = item.get("index")
        if not isinstance(index, int) or isinstance(index, bool) or not 0 <= index < 201:
            return False
        if index in seen:
            return False
        seen.add(index)
        if (
            not _is_nonnegative_int(item.get("row_count"))
            or not _is_nonnegative_number(item.get("elapsed_seconds"))
            or not isinstance(item.get("worker_pid"), int)
            or isinstance(item.get("worker_pid"), bool)
            or item["worker_pid"] < 1
        ):
            return False
        if include_spool_identity and (
            item.get("spool_path") != f"shard-{index:03d}.sqlite3"
            or not isinstance(item.get("spool_bytes"), int)
            or isinstance(item.get("spool_bytes"), bool)
            or item["spool_bytes"] < 1
            or not _is_sha256(item.get("spool_sha256"))
        ):
            return False
    return seen == set(range(201))


def _validate_task8_execution_receipt(
    tokenized: Mapping[str, Any], files: list[tuple[str, Path, int, str]]
) -> None:
    """Authenticate the exact 2M/201-range Task8 p96 execution lineage."""
    descriptor = tokenized.get("execution_receipt")
    if not isinstance(descriptor, dict):
        raise PublicationError("Task8 execution receipt descriptor is malformed")
    execution_file = next((item for item in files if item[0] == descriptor.get("path")), None)
    if execution_file is None:
        raise PublicationError("Task8 execution receipt was not authenticated")
    execution = _canonical_document(execution_file[1].read_bytes(), "Task8 execution receipt")
    expected_keys = {
        "schema_version",
        "source_commit",
        "declared_range_count",
        "occurrence_count",
        "allocated_cpus",
        "requested_workers",
        "effective_workers",
        "threads_per_worker",
        "thread_environment",
        "source_index_bytes",
        "source_index_sha256",
        "source_stage_elapsed_seconds",
        "started_at_ns",
        "parallel_phase_finished_at_ns",
        "parallel_phase_elapsed_seconds",
        "finished_at_ns",
        "elapsed_seconds",
        "ranges",
        "selection_sha256",
        "ordered_occurrences_sha256",
        "source_response_root_sha256",
        "tokenizer_sha256",
        "chat_template_sha256",
        "assistant_loss_target_sha256",
        "tokenized_sha256",
        "receipt_sha256",
    }
    claimed = execution.get("receipt_sha256")
    body = {key: value for key, value in execution.items() if key != "receipt_sha256"}
    thread_environment = execution.get("thread_environment")
    lineage = {
        "source_index_sha256": tokenized.get("source_index_sha256"),
        "selection_sha256": tokenized.get("selection_sha256"),
        "ordered_occurrences_sha256": tokenized.get("ordered_occurrences_sha256"),
        "source_response_root_sha256": tokenized.get("source_response_root_sha256"),
        "tokenizer_sha256": tokenized.get("tokenizer_sha256"),
        "chat_template_sha256": tokenized.get("chat_template_sha256"),
        "assistant_loss_target_sha256": tokenized.get("assistant_loss_target_sha256"),
        "tokenized_sha256": tokenized.get("database_sha256"),
    }
    started = execution.get("started_at_ns")
    parallel_finished = execution.get("parallel_phase_finished_at_ns")
    finished = execution.get("finished_at_ns")
    if (
        set(execution) != expected_keys
        or claimed != _sha256_bytes(_identity_json(body))
        or execution.get("schema_version") != 1
        or not isinstance(execution.get("source_commit"), str)
        or re.fullmatch(r"[0-9a-f]{40}", execution["source_commit"]) is None
        or execution.get("declared_range_count") != 201
        or execution.get("occurrence_count") != 2_000_000
        or execution.get("allocated_cpus") != 96
        or execution.get("requested_workers") != 96
        or execution.get("effective_workers") != 96
        or execution.get("threads_per_worker") != 1
        or not isinstance(thread_environment, dict)
        or set(thread_environment) != _PARALLEL_THREAD_ENVIRONMENT
        or set(thread_environment.values()) != {"1"}
        or not _is_nonnegative_int(execution.get("source_index_bytes"))
        or execution["source_index_bytes"] < 1
        or not _is_nonnegative_number(execution.get("source_stage_elapsed_seconds"))
        or not _is_nonnegative_number(execution.get("parallel_phase_elapsed_seconds"))
        or not _is_nonnegative_number(execution.get("elapsed_seconds"))
        or not isinstance(started, int)
        or isinstance(started, bool)
        or started < 0
        or not isinstance(finished, int)
        or isinstance(finished, bool)
        or finished < 0
        or not isinstance(parallel_finished, int)
        or isinstance(parallel_finished, bool)
        or parallel_finished < started
        or parallel_finished > finished
        or finished < started
        or execution["parallel_phase_elapsed_seconds"] > execution["elapsed_seconds"]
        or not _is_nonnegative_int(tokenized.get("source_index_bytes"))
        or execution.get("source_index_bytes") != tokenized.get("source_index_bytes")
        or any(not _is_sha256(value) for value in lineage.values())
        or any(execution.get(key) != value for key, value in lineage.items())
    ):
        raise PublicationError("Task8 execution lineage is invalid")
    ranges = execution.get("ranges")
    if not isinstance(ranges, list) or len(ranges) != 201:
        raise PublicationError("Task8 execution range topology is invalid")
    quotient, remainder = divmod(2_000_000, 201)
    start = 0
    range_keys = {
        "index",
        "start",
        "stop",
        "row_count",
        "spool_path",
        "spool_bytes",
        "spool_sha256",
        "elapsed_seconds",
        "worker_pid",
    }
    for index, item in enumerate(ranges):
        stop = start + quotient + int(index < remainder)
        if (
            not isinstance(item, dict)
            or set(item) != range_keys
            or item.get("index") != index
            or item.get("start") != start
            or item.get("stop") != stop
            or item.get("row_count") != stop - start
            or item.get("spool_path") != f"range-{index:03d}.sqlite3"
            or not _is_nonnegative_int(item.get("spool_bytes"))
            or item["spool_bytes"] < 1
            or not _is_sha256(item.get("spool_sha256"))
            or not _is_nonnegative_number(item.get("elapsed_seconds"))
            or not _is_nonnegative_int(item.get("worker_pid"))
            or item["worker_pid"] < 1
        ):
            raise PublicationError("Task8 execution range topology is invalid")
        start = stop
    if start != 2_000_000:
        raise PublicationError("Task8 execution range topology is invalid")


def _role_file_descriptors(role: str, payload: dict[str, Any]) -> list[dict[str, Any]]:
    if role == "selection" and payload.get("schema_version") == 2:
        root_sha256 = payload.get("root_sha256")
        root_record = {key: value for key, value in payload.items() if key != "root_sha256"}
        if root_sha256 != _sha256_bytes(_identity_json(root_record)):
            raise PublicationError("selection manifest root SHA-256 mismatch")
        shards = payload.get("shards")
        index = payload.get("index")
        if not isinstance(shards, list) or not shards or not isinstance(index, dict):
            raise PublicationError("selection manifest declared files are malformed")
        execution = payload.get("execution_receipt")
        if payload.get("selection_mode") == "B-prime-only" and not isinstance(execution, dict):
            raise PublicationError("Task5 selection execution receipt is missing")
        return [*shards, index, *([execution] if isinstance(execution, dict) else [])]
    if role == "selection" and payload.get("schema_version") == 3:
        root_sha256 = payload.get("root_sha256")
        root_record = {key: value for key, value in payload.items() if key != "root_sha256"}
        if root_sha256 != _sha256_bytes(_identity_json(root_record)):
            raise PublicationError("PTV2 selection receipt root SHA-256 mismatch")
        required = {
            "selection_sha256",
            "selection_identity",
            "policy_sha256",
            "policy_file_sha256",
            "source_inventory_sha256",
            "held_out_receipt_sha256",
            "trust_roots",
            "strategy",
            "occurrence_count",
            "ordered_occurrences_sha256",
            "shard_semantic_sha256",
            "policy",
            "index",
            "shards",
        }
        strategy = payload.get("strategy")
        strategy_fields = (
            {"baseline_receipt_sha256", "complement_selection_sha256"}
            if strategy == "A-repair"
            else set()
        )
        execution_fields = {"execution_receipt"} if "execution_receipt" in payload else set()
        if strategy not in {"A-repair", "B-balanced"} or set(
            payload
        ) != required | strategy_fields | execution_fields | {
            "schema_version",
            "root_sha256",
        }:
            raise PublicationError("PTV2 selection receipt schema is incomplete")
        policy = payload["policy"]
        index = payload["index"]
        shards = payload["shards"]
        if (
            not isinstance(policy, dict)
            or not isinstance(index, dict)
            or not isinstance(shards, list)
            or not shards
        ):
            raise PublicationError("PTV2 selection receipt declared files are malformed")
        digest_keys = (
            "root_sha256",
            "selection_sha256",
            "policy_sha256",
            "policy_file_sha256",
            "source_inventory_sha256",
            "held_out_receipt_sha256",
            "ordered_occurrences_sha256",
            *sorted(strategy_fields),
        )
        for key in digest_keys:
            value = payload[key]
            if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
                raise PublicationError(f"PTV2 selection receipt {key} is not a SHA-256")
        execution = payload.get("execution_receipt")
        if execution is not None and not isinstance(execution, dict):
            raise PublicationError("PTV2 execution receipt descriptor is malformed")
        return [*shards, index, policy, *([execution] if execution is not None else [])]
    if role == "tokenized" and payload.get("schema_version") == 1:
        if payload.get("strategy") in {"A-repair", "B-balanced"}:
            exact_two_million = payload.get("occurrence_count") == 2_000_000
            if exact_two_million and "execution_receipt" not in payload:
                raise PublicationError(
                    "Task8 execution receipt is required for exact 2M publication"
                )
            required = {
                "strategy",
                "occurrence_count",
                "assistant_tokens",
                "database_path",
                "database_bytes",
                "database_sha256",
                "selection_sha256",
                "source_response_root_sha256",
                "tokenizer_sha256",
                "chat_template_sha256",
                "assistant_loss_target_sha256",
            }
            if "execution_receipt" in payload:
                required |= {"source_index_bytes", "source_index_sha256"}
            if not required.issubset(payload):
                raise PublicationError("PTV2 tokenized receipt schema is incomplete")
            execution = payload.get("execution_receipt")
            if execution is not None and not isinstance(execution, dict):
                raise PublicationError("Task8 execution receipt descriptor is malformed")
            return [
                {
                    "path": payload["database_path"],
                    "bytes": payload["database_bytes"],
                    "sha256": payload["database_sha256"],
                },
                *([execution] if execution is not None else []),
            ]
        required = {
            "database_path",
            "database_bytes",
            "database_sha256",
            "record_count",
            "total_assistant_tokens",
            "selection_sha256",
            "resume_fingerprint",
        }
        if not required.issubset(payload):
            raise PublicationError("tokenized receipt schema is incomplete")
        return [
            {
                "path": payload["database_path"],
                "bytes": payload["database_bytes"],
                "sha256": payload["database_sha256"],
            }
        ]
    if role == "exposure" and payload.get("schema_version") == 1:
        if payload.get("strategy") in {"A-repair", "B-balanced"}:
            required = {
                "strategy",
                "target_assistant_tokens",
                "records_path",
                "records_bytes",
                "records_sha256",
                "row_count",
                "tokenized_sha256",
                "selection_sha256",
                "source_response_root_sha256",
                "tokenizer_sha256",
                "chat_template_sha256",
                "assistant_loss_target_sha256",
            }
            if not required.issubset(payload):
                raise PublicationError("PTV2 exposure receipt schema is incomplete")
            records_path = payload["records_path"]
            if not isinstance(records_path, str):
                raise PublicationError("PTV2 exposure records path is malformed")
            return [
                {
                    "path": records_path,
                    "bytes": payload["records_bytes"],
                    "sha256": payload["records_sha256"],
                }
            ]
        required = {
            "records_path",
            "records_bytes",
            "records_sha256",
            "row_count",
            "assistant_tokens",
            "selection_sha256",
            "resume_fingerprint",
        }
        if not required.issubset(payload):
            raise PublicationError("exposure receipt schema is incomplete")
        return [
            {
                "path": payload["records_path"],
                "bytes": payload["records_bytes"],
                "sha256": payload["records_sha256"],
            }
        ]
    descriptors = payload.get("files")
    if not isinstance(descriptors, list) or not descriptors:
        raise PublicationError(f"{role} receipt has no declared files")
    normalized: list[dict[str, Any]] = []
    for descriptor in descriptors:
        if not isinstance(descriptor, dict):
            raise PublicationError(f"{role} file descriptor is malformed")
        if "path" not in descriptor and "staged_path" in descriptor:
            descriptor = descriptor | {"path": descriptor["staged_path"]}
        normalized.append(descriptor)
    return normalized


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
        _verify_parquet_rows(path, rows, descriptor, resumed=True)
        return descriptor
    _call_hook(phase_hook, "shard_open")
    if os.path.lexists(path):
        descriptor = _parquet_descriptor(partial, path, rows)
        state = _shard_state(input_identity, index, descriptor)
        _write_exclusive_durable(partial / f"SHARD_STATE-{index:06d}.json", _canonical_json(state))
        _call_hook(phase_hook, "shard_closed")
        return descriptor
    table = pa.Table.from_pylist(rows, schema=_SCHEMA)
    temporary = shards_dir / f".{name}.partial-{uuid4().hex}"
    with temporary.open("xb") as output:
        pq.write_table(table, output, compression="zstd")
        output.flush()
        os.fsync(output.fileno())
    _call_hook(phase_hook, "shard_temporary_fsynced")
    _rename_no_replace(temporary, path)
    _call_hook(phase_hook, "shard_installed")
    descriptor = _parquet_descriptor(partial, path, rows)
    state = _shard_state(input_identity, index, descriptor)
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
            or not isinstance(payload.get("shard"), dict)
            or payload.get("canonical_rows_sha256") != payload["shard"].get("canonical_rows_sha256")
        ):
            raise PublicationError("partial shard state does not match input identity")
        states[index] = payload
    declared = {Path(state["shard"]["path"]).name for state in states.values()}
    actual = {path.name for path in (partial / "shards").glob("*")}
    allowed_orphan = f"part-{len(states):06d}.parquet"
    if not declared.issubset(actual) or actual - declared not in (set(), {allowed_orphan}):
        raise PublicationError("partial contains unauthenticated shard files")
    return states


def _shard_state(input_identity: str, index: int, descriptor: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema": "modelopt-specdec-partial-shard-v1",
        "input_identity": input_identity,
        "shard_index": index,
        "canonical_rows_sha256": descriptor["canonical_rows_sha256"],
        "shard": descriptor,
    }


def _parquet_descriptor(partial: Path, path: Path, rows: list[dict[str, Any]]) -> dict[str, Any]:
    descriptor = {
        "path": path.relative_to(partial).as_posix(),
        "bytes": path.stat().st_size,
        "sha256": _sha256_file(path),
        "row_count": len(rows),
        "assistant_token_count": sum(int(row["assistant_tokens"]) for row in rows),
        "quarantine_count": sum(row["rejection_reason"] is not None for row in rows),
        "canonical_rows_sha256": _canonical_rows_sha256(rows),
    }
    _verify_file_record(partial, descriptor)
    _verify_parquet_rows(path, rows, descriptor, resumed=False)
    return descriptor


def _verify_parquet_rows(
    path: Path,
    expected_rows: list[dict[str, Any]],
    descriptor: Mapping[str, Any],
    *,
    resumed: bool,
) -> None:
    try:
        table = pq.read_table(path)
    except Exception as error:
        raise PublicationError("published Parquet shard cannot be reread") from error
    actual_rows = table.to_pylist()
    actual_tokens = sum(int(row["assistant_tokens"]) for row in actual_rows)
    actual_quarantines = sum(row["rejection_reason"] is not None for row in actual_rows)
    actual_digest = _canonical_rows_sha256(actual_rows)
    if (
        table.schema != _SCHEMA
        or actual_rows != expected_rows
        or descriptor.get("row_count") != len(actual_rows)
        or descriptor.get("assistant_token_count") != actual_tokens
        or descriptor.get("quarantine_count") != actual_quarantines
        or descriptor.get("canonical_rows_sha256") != actual_digest
    ):
        prefix = "resumed shard" if resumed else "Parquet shard"
        raise PublicationError(f"{prefix} semantics do not match current batch")


def _canonical_rows_sha256(rows: list[dict[str, Any]]) -> str:
    digest = hashlib.sha256()
    for row in rows:
        digest.update(_canonical_json(row))
    return digest.hexdigest()


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
    normalized_root = partial.resolve(strict=True)
    for path in _regular_tree_files(normalized_root):
        relative = path.relative_to(normalized_root).as_posix()
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


def _partial_tree_safe(partial: Path, artifacts: tuple[_AuthenticatedArtifact, ...]) -> bool:
    """Validate a resumable partial without following or opening nested symlinks."""
    input_files = {Path("inputs") / artifact.role / "receipt.json" for artifact in artifacts} | {
        Path("inputs") / artifact.role / "files" / relative
        for artifact in artifacts
        for relative, _source, _size, _digest in artifact.files
    }
    input_directories = {
        parent for path in input_files for parent in path.parents if parent != Path(".")
    }
    try:
        root_metadata = os.lstat(partial)
        if not stat.S_ISDIR(root_metadata.st_mode):
            return False
        for directory, directory_names, file_names in os.walk(partial, followlinks=False):
            base = Path(directory)
            for name in directory_names:
                path = base / name
                metadata = os.lstat(path)
                relative = path.relative_to(partial)
                if not stat.S_ISDIR(metadata.st_mode) or not _allowed_partial_directory(
                    relative, input_directories
                ):
                    return False
            for name in file_names:
                path = base / name
                metadata = os.lstat(path)
                relative = path.relative_to(partial)
                if not stat.S_ISREG(metadata.st_mode) or not _allowed_partial_file(
                    relative, input_files
                ):
                    return False
    except OSError:
        return False
    return True


def _allowed_partial_directory(relative: Path, input_directories: set[Path]) -> bool:
    return relative in input_directories or relative == Path("shards")


def _allowed_partial_file(relative: Path, input_files: set[Path]) -> bool:
    if len(relative.parts) == 1:
        return relative.name in {
            "PARTIAL_IDENTITY.json",
            "CORPUS_MANIFEST.json",
            "PUBLICATION.json",
        } or bool(re.fullmatch(r"SHARD_STATE-[0-9]{6}\.json", relative.name))
    if relative.parts[0] == "inputs":
        return relative in input_files
    return (
        len(relative.parts) == 2
        and relative.parts[0] == "shards"
        and re.fullmatch(r"part-[0-9]{6}\.parquet", relative.name) is not None
    )


def _quarantine_partial(partial: Path, quarantine: Path, parent: Path) -> None:
    before = _observe(partial)
    if before.identity is None:
        raise PublicationError("stale partial cannot be safely observed")
    _rename_no_replace(partial, quarantine)
    if _observe(quarantine).identity != before.identity:
        raise PublicationError("quarantined partial inode does not match observed partial")
    _fsync_directory(parent)


def _file_descriptor(value: object, role: str, root: Path) -> tuple[str, int, str]:
    if not isinstance(value, dict):
        raise PublicationError(f"{role} file descriptor is malformed")
    raw_path = value.get("path")
    if not isinstance(raw_path, str) or not raw_path:
        raise PublicationError(f"{role} file path must be a non-empty string")
    descriptor_path = Path(raw_path)
    if descriptor_path.is_absolute():
        try:
            relative = descriptor_path.resolve(strict=False).relative_to(root).as_posix()
        except ValueError as error:
            raise PublicationError(f"{role} file path escapes its receipt root") from error
    else:
        relative = _safe_relative(raw_path, f"{role} file")
    size = value.get("bytes", value.get("byte_count", value.get("size")))
    digest = value.get("sha256")
    unresolved = root / relative
    if size is None and not unresolved.is_symlink() and unresolved.is_file():
        size = unresolved.stat(follow_symlinks=False).st_size
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


def _receipt_document(raw: bytes, role: str) -> dict[str, Any]:
    try:
        payload = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise PublicationError(f"{role} receipt is not JSON") from error
    if not isinstance(payload, dict):
        raise PublicationError(f"{role} receipt is not a JSON object")
    allowed_encodings = {_canonical_json(payload)}
    if role == "source":
        allowed_encodings.add((json.dumps(payload, indent=2, sort_keys=True) + "\n").encode())
    if raw not in allowed_encodings:
        raise PublicationError(f"{role} receipt is not deterministically encoded JSON")
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


def _identity_json(value: object) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()


def _call_hook(hook: Callable[[str], None] | None, phase: str) -> None:
    if hook is not None:
        hook(phase)
