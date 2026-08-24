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

"""Build a hash-bound, pretokenized SpecDec inventory from synthesis or trace shards."""

import argparse
import hashlib
import json
import os
import re
import shutil
import sqlite3
import stat
import tempfile
import time
import weakref
from collections.abc import Mapping, Sequence
from concurrent.futures import ProcessPoolExecutor, as_completed
from copy import deepcopy
from dataclasses import asdict, dataclass
from multiprocessing import get_context
from pathlib import Path
from types import MappingProxyType
from typing import Any, Literal, Protocol, overload

import yaml
from specdec_corpus_contracts import (
    CanonicalPrompt,
    canonical_json,
    sha256_bytes,
    sha256_canonical_json,
)
from specdec_identity import UUIDCollisionError, canonicalize_prompt
from stage_ptv23_sources import SourceFile, SourceIdentity, SourceInventory
from trajectory_schema import TrajectoryValidationError, validate_trajectory

__all__ = [
    "APPROVED_PTV2_ALLOWLIST_SHA256",
    "APPROVED_PTV2_REVISION",
    "CandidateCell",
    "CandidateInventory",
    "CandidatePrompt",
    "CandidateTokenizer",
    "DiskBackedCandidateRows",
    "ExclusionProof",
    "ExclusionReceipt",
    "InventorySource",
    "SourceFile",
    "SourceIdentity",
    "SourceInventory",
    "TokenizerSnapshot",
    "TokenizerSnapshotFile",
    "UUIDCollisionError",
    "build_candidate_inventory",
    "build_candidate_inventory_from_snapshot",
    "build_inventory_rows",
    "candidate_inventory_bytes",
    "candidate_inventory_sha256",
    "is_approved_ptv2_source",
    "iter_candidate_inventory_bytes",
    "load_tokenizer_snapshot",
    "make_exclusion_receipt",
    "sha256_file",
    "tokenizer_snapshot_sha256",
    "validate_approved_ptv2_topology",
    "verify_candidate_inventory_membership",
    "write_tokenizer_snapshot_receipt",
]

_SHA = re.compile(r"^[0-9a-f]{40}(?:[0-9a-f]{24})?$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_REVISION = re.compile(r"^[0-9a-f]{40}$")
_PTV2_REVISION = "5c89e01dd720ae0f4058445ed49c5fb68a03c76e"
APPROVED_PTV2_REVISION = _PTV2_REVISION
_APPROVED_PTV2_SOURCES = frozenset(
    (
        "nvidia/Nemotron-Post-Training-Dataset-v2",
        "default",
        split,
        _PTV2_REVISION,
    )
    for split in (
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
)
APPROVED_PTV2_ALLOWLIST_SHA256 = sha256_bytes(canonical_json(sorted(_APPROVED_PTV2_SOURCES)))
_APPROVED_PTV2_SHARD_COUNTS = MappingProxyType(
    {
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
)
_LANGUAGE_ALIASES = {
    "english": "en",
    "french": "fr",
    "german": "de",
    "italian": "it",
    "japanese": "ja",
    "spanish": "es",
}
_LANGUAGE_CODES = frozenset({*_LANGUAGE_ALIASES.values(), "zh"})
_APPROVED_SOURCE_LANGUAGES = {
    ("nvidia/Nemotron-SFT-SWE-v2", "agentless"): "en",
    ("nvidia/Nemotron-SFT-SWE-v2", "openhands_swe"): "en",
    ("nvidia/Nemotron-SWE-v1", "r2e_gym"): "en",
    ("nvidia/Nemotron-Agentic-v1", "interactive_agent"): "en",
    ("nvidia/Nemotron-Agentic-v1", "tool_calling"): "en",
    ("nvidia/Nemotron-SFT-Competitive-Programming-v2", "exercism"): "en",
    ("nvidia/Nemotron-Math-v2", "low"): "en",
    ("nvidia/Nemotron-Science-v1", "RQA"): "en",
    ("nvidia/Nemotron-SFT-Instruction-Following-Chat-v2", "reasoning_off"): "en",
    ("nvidia/Nemotron-SFT-Multilingual-v1", "stem_zh"): "zh",
}
_CANDIDATE_QUARANTINE_CODES = frozenset(
    {
        "context_too_long",
        "invalid_language",
        "invalid_row",
        "invalid_tokenization",
        "invalid_tools",
        "missing_messages",
        "wrong_lane",
    }
)
_TOKENIZER_FILE_NAMES = frozenset(
    {
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
)


@dataclass(frozen=True)
class TokenizerSnapshotFile:
    """One exact regular file in an immutable tokenizer snapshot."""

    path: str
    bytes: int
    sha256: str


@dataclass(frozen=True)
class TokenizerSnapshot:
    """Authenticated tokenizer files plus the effective chat-template identity."""

    root: Path
    files: tuple[TokenizerSnapshotFile, ...]
    tokenizer_sha256: str
    chat_template_sha256: str
    chat_template_canonical_json: str
    root_sha256: str
    receipt_sha256: str
    root_device: int
    root_inode: int


@dataclass(frozen=True)
class CandidateCell:
    """One capacity cell after canonical exclusion and replay validation."""

    arm_domain: str
    lane: str
    language: str
    context_bucket: str


class CandidateTokenizer(Protocol):
    """Tokenizer loaded from an authenticated local snapshot."""

    def apply_chat_template(self, messages: Any, **kwargs: Any) -> Any:
        """Return token IDs for the supplied canonical prompt."""
        ...


@dataclass(frozen=True)
class ExclusionProof:
    """Receipt identity and exact prompt-set reconciliation used by an inventory."""

    receipt_sha256: str
    prompt_ids_sha256: str
    prompt_id_count: int
    excluded_candidate_count: int


@dataclass(frozen=True)
class ExclusionReceipt:
    """Typed, self-authenticating exclusion prompt-set receipt."""

    kind: Literal["baseline", "held-out"]
    prompt_ids: tuple[str, ...]
    prompt_ids_sha256: str
    prompt_id_count: int
    receipt_sha256: str


def make_exclusion_receipt(
    kind: Literal["baseline", "held-out"], prompt_ids: Sequence[str]
) -> ExclusionReceipt:
    """Canonicalize an exact exclusion set into its typed content receipt."""
    if kind not in {"baseline", "held-out"}:
        raise ValueError("exclusion receipt kind must be baseline or held-out")
    ordered = tuple(sorted(prompt_ids))
    if len(set(ordered)) != len(ordered) or any(
        _SHA256.fullmatch(value) is None for value in ordered
    ):
        raise ValueError("exclusion receipt prompt IDs must be unique lowercase SHA-256 values")
    prompt_ids_sha256 = sha256_canonical_json(ordered)
    payload = {
        "schema_version": 1,
        "kind": kind,
        "prompt_ids": ordered,
        "prompt_ids_sha256": prompt_ids_sha256,
        "prompt_id_count": len(ordered),
    }
    return ExclusionReceipt(
        kind,
        ordered,
        prompt_ids_sha256,
        len(ordered),
        sha256_bytes(canonical_json(payload)),
    )


def _validate_exclusion_receipt(
    receipt: ExclusionReceipt, expected_kind: Literal["baseline", "held-out"]
) -> set[str]:
    if not isinstance(receipt, ExclusionReceipt) or receipt.kind != expected_kind:
        raise ValueError(f"{expected_kind} exclusion receipt has the wrong typed identity")
    expected = make_exclusion_receipt(expected_kind, receipt.prompt_ids)
    if receipt != expected or receipt.receipt_sha256 == "0" * 64:
        raise ValueError(f"{expected_kind} exclusion receipt content reconciliation mismatch")
    return set(receipt.prompt_ids)


def is_approved_ptv2_source(
    repository_id: str, configuration: str, split: str, revision: str
) -> bool:
    """Return whether a structured identity is in the immutable PTV2 allowlist."""
    return (
        repository_id,
        configuration,
        split,
        revision,
    ) in _APPROVED_PTV2_SOURCES


def _approved_ptv2_source(source: SourceIdentity) -> bool:
    return is_approved_ptv2_source(
        source.repository_id, source.configuration, source.split, source.revision
    )


def validate_approved_ptv2_topology(source_inventory: SourceInventory) -> SourceInventory:
    """Require the exact staged PTV2 repository, split, lane, and Parquet topology."""
    if not isinstance(source_inventory, SourceInventory) or source_inventory.staged_root is None:
        raise ValueError("approved PTV2 source topology requires a staged Task 3 inventory")
    observed: dict[str, int] = {}
    for source in source_inventory.sources:
        if (
            source.repository_id != "nvidia/Nemotron-Post-Training-Dataset-v2"
            or source.configuration != "default"
            or source.revision != APPROVED_PTV2_REVISION
            or source.approved_use is not True
            or source.lane != "target-synth"
            or source.cell != source.split
            or source.split not in _APPROVED_PTV2_SHARD_COUNTS
            or any(Path(file.path).suffix != ".parquet" for file in source.files)
        ):
            raise ValueError("approved PTV2 source topology identity mismatch")
        if source.split in observed:
            raise ValueError("approved PTV2 source topology repeats a split")
        observed[source.split] = len(source.files)
    if observed != dict(_APPROVED_PTV2_SHARD_COUNTS):
        raise ValueError("approved PTV2 source topology shard counts mismatch")
    return source_inventory


@dataclass(frozen=True)
class CandidatePrompt(CanonicalPrompt):
    """A canonical prompt plus deterministic selection and tokenization metadata."""

    lane: str
    context_bucket: str
    full_token_count: int
    source_manifest_sha256: str
    source_file_path: str
    input_ids: tuple[int, ...]
    tokenizer_sha256: str
    replay_valid: bool
    source_family: Literal["ptv2", "ptv3"]
    source_repository_id: str
    source_configuration: str
    source_split: str
    source_conversation_sha256: str | None = None
    source_response_sha256: str | None = None
    chat_template_sha256: str | None = None

    @property
    def arm_domain(self) -> str:
        return self.domain


_CANDIDATE_COLUMNS = (
    "prompt_uuid",
    "canonical_bytes",
    "source_id",
    "source_revision",
    "source_file_sha256",
    "source_row_index",
    "domain",
    "language",
    "lane",
    "context_bucket",
    "full_token_count",
    "source_manifest_sha256",
    "source_file_path",
    "input_ids",
    "tokenizer_sha256",
    "replay_valid",
    "source_family",
    "source_repository_id",
    "source_configuration",
    "source_split",
    "source_conversation_sha256",
    "source_response_sha256",
    "chat_template_sha256",
)


class DiskBackedCandidateRows(Sequence[CandidatePrompt]):
    """A deterministic, lazy prompt sequence backed by an ordered SQLite index."""

    resident_row_count = 0

    def __init__(
        self,
        storage_path: Path,
        count: int,
        source_manifest_sha256: str,
        tokenizer_sha256: str,
        storage_root: Path,
    ) -> None:
        self.storage_path = storage_path
        self._count = count
        self.source_manifest_sha256 = source_manifest_sha256
        self.tokenizer_sha256 = tokenizer_sha256
        self._closed = False
        self._finalizer = weakref.finalize(self, shutil.rmtree, storage_root, True)

    def close(self) -> None:
        """Release and remove the temporary storage owned by this sequence."""
        if self._closed:
            return
        self._closed = True
        self._finalizer()

    def _require_open(self) -> None:
        if self._closed:
            raise RuntimeError("candidate inventory is closed")

    def __len__(self) -> int:
        return self._count

    @staticmethod
    def _candidate(record: tuple[Any, ...]) -> CandidatePrompt:
        values = dict(zip(_CANDIDATE_COLUMNS, record))
        return CandidatePrompt(
            prompt_uuid=values["prompt_uuid"],
            canonical_bytes=values["canonical_bytes"],
            source_id=values["source_id"],
            source_revision=values["source_revision"],
            source_file_sha256=values["source_file_sha256"],
            source_row_index=values["source_row_index"],
            domain=values["domain"],
            language=values["language"],
            lane=values["lane"],
            context_bucket=values["context_bucket"],
            full_token_count=values["full_token_count"],
            source_manifest_sha256=values["source_manifest_sha256"],
            source_file_path=values["source_file_path"],
            input_ids=tuple(json.loads(values["input_ids"])),
            tokenizer_sha256=values["tokenizer_sha256"],
            replay_valid=bool(values["replay_valid"]),
            source_family=values["source_family"],
            source_repository_id=values["source_repository_id"],
            source_configuration=values["source_configuration"],
            source_split=values["source_split"],
            source_conversation_sha256=values["source_conversation_sha256"],
            source_response_sha256=values["source_response_sha256"],
            chat_template_sha256=values["chat_template_sha256"],
        )

    def __iter__(self):
        self._require_open()
        query = f"SELECT {', '.join(_CANDIDATE_COLUMNS)} FROM candidates ORDER BY prompt_uuid"
        with sqlite3.connect(self.storage_path) as connection:
            for record in connection.execute(query):
                yield self._candidate(record)

    def _at(self, index: int) -> CandidatePrompt:
        self._require_open()
        if index < 0:
            index += self._count
        if index < 0 or index >= self._count:
            raise IndexError(index)
        query = (
            f"SELECT {', '.join(_CANDIDATE_COLUMNS)} FROM candidates "
            "ORDER BY prompt_uuid LIMIT 1 OFFSET ?"
        )
        with sqlite3.connect(self.storage_path) as connection:
            record = connection.execute(query, (index,)).fetchone()
        if record is None:
            raise IndexError(index)
        return self._candidate(record)

    @overload
    def __getitem__(self, index: int) -> CandidatePrompt: ...

    @overload
    def __getitem__(self, index: slice) -> tuple[CandidatePrompt, ...]: ...

    def __getitem__(self, index: int | slice) -> CandidatePrompt | tuple[CandidatePrompt, ...]:
        if isinstance(index, slice):
            start, stop, step = index.indices(self._count)
            return tuple(self._at(position) for position in range(start, stop, step))
        return self._at(index)


@dataclass(frozen=True, eq=False)
class CandidateInventory:
    """Canonical candidates, usable capacity, and stable rejection receipts."""

    rows: Sequence[CandidatePrompt]
    capacity: Mapping[CandidateCell, int]
    quarantine_counts: Mapping[str, int]
    inventory_sha256: str
    baseline_exclusion: ExclusionProof
    held_out_exclusion: ExclusionProof
    ptv2_revision: str
    ptv2_allowlist_sha256: str
    execution_receipt: Mapping[str, Any] | None = None
    diagnostic_receipt: Mapping[str, Any] | None = None

    def close(self) -> None:
        """Release temporary row storage after all sequence consumers finish."""
        if isinstance(self.rows, DiskBackedCandidateRows):
            self.rows.close()

    def __enter__(self) -> "CandidateInventory":
        if isinstance(self.rows, DiskBackedCandidateRows):
            self.rows._require_open()
        return self

    def __exit__(self, exc_type: object, exc_value: object, traceback: object) -> None:
        self.close()

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, CandidateInventory):
            return NotImplemented
        return (
            self.inventory_sha256 == other.inventory_sha256
            and self.capacity == other.capacity
            and self.quarantine_counts == other.quarantine_counts
            and self.baseline_exclusion == other.baseline_exclusion
            and self.held_out_exclusion == other.held_out_exclusion
            and self.ptv2_revision == other.ptv2_revision
            and self.ptv2_allowlist_sha256 == other.ptv2_allowlist_sha256
        )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tokenizer_snapshot_sha256(root: Path) -> str:
    """Hash tokenizer serialization files without hashing model weights."""
    files = sorted(
        path for path in root.iterdir() if path.is_file() and path.name in _TOKENIZER_FILE_NAMES
    )
    if not files:
        raise ValueError("tokenizer snapshot contains no serialization files")
    digest = hashlib.sha256()
    for path in files:
        digest.update(path.name.encode())
        digest.update(b"\0")
        digest.update(bytes.fromhex(sha256_file(path)))
    return digest.hexdigest()


def _tokenizer_chat_template(root: Path) -> Any:
    template_path = root / "chat_template.jinja"
    if template_path.is_file():
        return template_path.read_text(encoding="utf-8")
    config_path = root / "tokenizer_config.json"
    if not config_path.is_file():
        raise ValueError("tokenizer snapshot has no chat template")
    try:
        config = json.loads(config_path.read_bytes())
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("tokenizer snapshot config is invalid") from error
    template = config.get("chat_template") if isinstance(config, dict) else None
    if template is None:
        raise ValueError("tokenizer snapshot has no chat template")
    return template


def _snapshot_file_records(root: Path) -> tuple[TokenizerSnapshotFile, ...]:
    try:
        tree = _staged_tree_snapshot(root)
    except (OSError, ValueError) as error:
        raise ValueError("tokenizer snapshot physical file set is invalid") from error
    actual_files = {
        relative
        for relative, identity in tree.items()
        if relative != "." and stat.S_ISREG(identity[0])
    }
    if (
        set(tree) != {".", *actual_files}
        or not actual_files
        or any(
            "/" in relative or relative not in _TOKENIZER_FILE_NAMES for relative in actual_files
        )
    ):
        raise ValueError("tokenizer snapshot has an unsupported physical file set")
    records: list[TokenizerSnapshotFile] = []
    for relative in sorted(actual_files):
        path = root / relative
        observed = os.lstat(path)
        descriptor = SourceFile(relative, observed.st_size, sha256_file(path))
        file_descriptor, initial = _open_verified_source_fd(path, descriptor)
        try:
            _verify_source_fd_stable(file_descriptor, initial, path, descriptor)
        finally:
            os.close(file_descriptor)
        records.append(TokenizerSnapshotFile(relative, descriptor.bytes, descriptor.sha256))
    try:
        final_tree = _staged_tree_snapshot(root)
    except (OSError, ValueError) as error:
        raise ValueError("tokenizer snapshot physical file set is invalid") from error
    if final_tree != tree:
        raise ValueError("tokenizer snapshot changed during authentication")
    return tuple(records)


def _tokenizer_snapshot_payload(root_name: str, root: Path) -> dict[str, Any]:
    try:
        initial_tree = _staged_tree_snapshot(root)
    except (OSError, ValueError) as error:
        raise ValueError("tokenizer snapshot physical file set is invalid") from error
    files = _snapshot_file_records(root)
    canonical_template = canonical_json(_tokenizer_chat_template(root)).decode("utf-8")
    payload = {
        "schema_version": 1,
        "kind": "specdec-tokenizer-snapshot",
        "snapshot_path": root_name,
        "files": [asdict(record) for record in files],
        "tokenizer_sha256": tokenizer_snapshot_sha256(root),
        "chat_template_canonical_json": canonical_template,
        "chat_template_sha256": sha256_bytes(canonical_template.encode("utf-8")),
    }
    try:
        final_tree = _staged_tree_snapshot(root)
    except (OSError, ValueError) as error:
        raise ValueError("tokenizer snapshot physical file set is invalid") from error
    if final_tree != initial_tree:
        raise ValueError("tokenizer snapshot changed during authentication")
    return payload


def _snapshot_root_identity(parent: Path, relative: Path) -> tuple[int, int]:
    """Open every snapshot directory component without following symlinks."""
    if (
        not parent.is_absolute()
        or relative.is_absolute()
        or not relative.parts
        or ".." in relative.parts
    ):
        raise ValueError("tokenizer snapshot path is not lexical and relative")
    if not hasattr(os, "O_NOFOLLOW") or not hasattr(os, "O_DIRECTORY"):
        raise ValueError("tokenizer snapshot requires no-follow directory traversal")
    flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
    try:
        descriptor = os.open(parent.anchor, flags)
    except OSError as error:
        raise ValueError(
            "tokenizer snapshot filesystem root is not a no-follow directory"
        ) from error
    try:
        components = (*parent.parts[1:], *relative.parts)
        for component in components:
            try:
                child = os.open(component, flags, dir_fd=descriptor)
            except OSError as error:
                raise ValueError(
                    "tokenizer snapshot path contains a symlink or non-directory component"
                ) from error
            os.close(descriptor)
            descriptor = child
        observed = os.fstat(descriptor)
        if not stat.S_ISDIR(observed.st_mode):
            raise ValueError("tokenizer snapshot root is not a directory")
        return observed.st_dev, observed.st_ino
    finally:
        os.close(descriptor)


def write_tokenizer_snapshot_receipt(root: Path, receipt_path: Path) -> TokenizerSnapshot:
    """Publish a typed immutable receipt for a dedicated local tokenizer snapshot."""
    receipt_parent = receipt_path.parent.absolute()
    lexical_root = root.absolute()
    try:
        root_name = lexical_root.relative_to(receipt_parent).as_posix()
    except ValueError as error:
        raise ValueError("tokenizer snapshot must be below its receipt directory") from error
    initial_root_identity = _snapshot_root_identity(receipt_parent, Path(root_name))
    payload = _tokenizer_snapshot_payload(root_name, lexical_root)
    if _snapshot_root_identity(receipt_parent, Path(root_name)) != initial_root_identity:
        raise ValueError("tokenizer snapshot root changed during authentication")
    payload["root_sha256"] = sha256_bytes(canonical_json(payload))
    encoded = canonical_json(payload) + b"\n"
    try:
        with receipt_path.open("xb") as receipt:
            receipt.write(encoded)
            receipt.flush()
            os.fsync(receipt.fileno())
    except FileExistsError as error:
        raise ValueError("tokenizer snapshot receipt already exists") from error
    return load_tokenizer_snapshot(receipt_path, sha256_bytes(encoded))


def load_tokenizer_snapshot(receipt_path: Path, expected_receipt_sha256: str) -> TokenizerSnapshot:
    """Load and independently authenticate a caller-pinned tokenizer snapshot receipt."""
    if _SHA256.fullmatch(expected_receipt_sha256) is None:
        raise ValueError("tokenizer snapshot receipt SHA-256 is invalid")
    try:
        observed = os.lstat(receipt_path)
        receipt_descriptor = SourceFile(
            receipt_path.name, observed.st_size, expected_receipt_sha256
        )
        descriptor, initial = _open_verified_source_fd(receipt_path, receipt_descriptor)
        try:
            os.lseek(descriptor, 0, os.SEEK_SET)
            with os.fdopen(os.dup(descriptor), "rb") as stream:
                raw = stream.read()
            _verify_source_fd_stable(descriptor, initial, receipt_path, receipt_descriptor)
        finally:
            os.close(descriptor)
    except (OSError, ValueError) as error:
        raise ValueError("tokenizer snapshot receipt SHA-256 mismatch") from error
    receipt_sha256 = sha256_bytes(raw)
    try:
        payload = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("tokenizer snapshot receipt is invalid") from error
    if raw != canonical_json(payload) + b"\n" or not isinstance(payload, dict):
        raise ValueError("tokenizer snapshot receipt is not canonical")
    required = {
        "schema_version",
        "kind",
        "snapshot_path",
        "files",
        "tokenizer_sha256",
        "chat_template_canonical_json",
        "chat_template_sha256",
        "root_sha256",
    }
    if (
        set(payload) != required
        or payload.get("schema_version") != 1
        or payload.get("kind") != "specdec-tokenizer-snapshot"
    ):
        raise ValueError("tokenizer snapshot receipt schema mismatch")
    root_record = {key: value for key, value in payload.items() if key != "root_sha256"}
    if payload["root_sha256"] != sha256_bytes(canonical_json(root_record)):
        raise ValueError("tokenizer snapshot receipt root mismatch")
    relative = Path(str(payload["snapshot_path"]))
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError("tokenizer snapshot path is not lexical and relative")
    receipt_parent = receipt_path.parent.absolute()
    root = receipt_parent / relative
    initial_root_identity = _snapshot_root_identity(receipt_parent, relative)
    expected_payload = _tokenizer_snapshot_payload(relative.as_posix(), root)
    if _snapshot_root_identity(receipt_parent, relative) != initial_root_identity:
        raise ValueError("tokenizer snapshot root changed during authentication")
    if expected_payload != root_record:
        raise ValueError("tokenizer snapshot physical identity mismatch")
    records = tuple(TokenizerSnapshotFile(**record) for record in payload["files"])
    return TokenizerSnapshot(
        root,
        records,
        str(payload["tokenizer_sha256"]),
        str(payload["chat_template_sha256"]),
        str(payload["chat_template_canonical_json"]),
        str(payload["root_sha256"]),
        receipt_sha256,
        initial_root_identity[0],
        initial_root_identity[1],
    )


def _load_tokenizer_from_snapshot(snapshot: TokenizerSnapshot) -> CandidateTokenizer:
    """Load only the locally authenticated serialization represented by ``snapshot``."""
    try:
        from transformers import AutoTokenizer  # pyright: ignore[reportMissingImports]

        tokenizer = AutoTokenizer.from_pretrained(
            snapshot.root,
            local_files_only=True,
            trust_remote_code=False,
        )
    except Exception as error:
        raise ValueError("authenticated tokenizer snapshot cannot be loaded locally") from error
    try:
        loaded_template = canonical_json(getattr(tokenizer, "chat_template"))
    except (TypeError, ValueError) as error:
        raise ValueError("loaded tokenizer chat template is invalid") from error
    if sha256_bytes(loaded_template) != snapshot.chat_template_sha256:
        raise ValueError("loaded tokenizer chat template identity mismatch")
    return tokenizer


def _authenticated_snapshot_tokenizer(
    receipt_path: Path, expected_receipt_sha256: str
) -> tuple[TokenizerSnapshot, CandidateTokenizer]:
    snapshot = load_tokenizer_snapshot(receipt_path, expected_receipt_sha256)
    tokenizer = _load_tokenizer_from_snapshot(snapshot)
    if load_tokenizer_snapshot(receipt_path, expected_receipt_sha256) != snapshot:
        raise ValueError("tokenizer snapshot root or contents changed while loading")
    return snapshot, tokenizer


@dataclass(frozen=True)
class InventorySource:
    """One immutable raw/synthesis or recorded-trace input lane."""

    source_id: str
    source_revision: str
    license: str
    pool: str
    category: str
    response_source: str
    tool_lane: str
    manifest_path: Path

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "manifest_path", Path(self.manifest_path).expanduser().resolve(strict=False)
        )
        if not self.source_id or not self.license or not self.category:
            raise ValueError("source identity, license, and category are required")
        if _SHA.fullmatch(self.source_revision) is None:
            raise ValueError("source_revision must be a pinned commit digest")
        if self.pool not in {"ptv2", "ptv3"}:
            raise ValueError("inventory pool must be ptv2 or ptv3")
        if self.response_source not in {"target-synth", "trace-replay"}:
            raise ValueError("response_source must be target-synth or trace-replay")
        expected_lane = "none" if self.response_source == "target-synth" else "recorded-trace"
        if self.tool_lane != expected_lane:
            raise ValueError("tool lane does not match the response source")


def _verified_files(manifest_path: Path) -> tuple[str, list[Path]]:
    raw = manifest_path.read_bytes()
    manifest_sha256 = hashlib.sha256(raw).hexdigest()
    payload = json.loads(raw)
    records = payload.get("files") if isinstance(payload, dict) else None
    if (
        not isinstance(payload, dict)
        or payload.get("schema_version") != 1
        or not isinstance(records, list)
        or not records
    ):
        raise ValueError("source manifest must contain a non-empty version-1 file set")
    root = manifest_path.parent.resolve(strict=True)
    files: list[Path] = []
    for record in records:
        relative = record.get("path") if isinstance(record, dict) else None
        if not isinstance(relative, str) or not relative:
            raise ValueError("source manifest has an invalid file record")
        path = (manifest_path.parent / relative).resolve(strict=True)
        if not path.is_relative_to(root) or not path.is_file():
            raise ValueError(f"source file escapes manifest root: {relative}")
        if path.stat().st_size != record.get("bytes") or sha256_file(path) != record.get("sha256"):
            raise ValueError(f"source file identity mismatch: {relative}")
        files.append(path)
    return manifest_sha256, files


def _iter_rows(path: Path):
    if path.suffix == ".parquet":
        import pyarrow.parquet as pq  # pyright: ignore[reportMissingImports]

        for row in pq.read_table(path).to_pylist():
            yield json.loads(row["raw_json"]) if set(row) == {"raw_json"} else row
        return
    with path.open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            if line.strip():
                try:
                    yield json.loads(line)
                except json.JSONDecodeError as error:
                    raise ValueError(f"invalid JSON row: {path}:{line_number}") from error


def _iter_candidate_rows(path: Path):
    if path.suffix == ".parquet":
        import pyarrow.parquet as pq  # pyright: ignore[reportMissingImports]

        row_index = 0
        for batch in pq.ParquetFile(path).iter_batches(batch_size=1_024):
            for raw_row in batch.to_pylist():
                value: Any = raw_row
                if isinstance(raw_row, dict) and set(raw_row) == {"raw_json"}:
                    encoded = raw_row["raw_json"]
                    try:
                        value = json.loads(encoded) if isinstance(encoded, str | bytes) else None
                    except (UnicodeDecodeError, json.JSONDecodeError):
                        value = None
                yield row_index, value if isinstance(value, dict) else None
                row_index += 1
        return
    with path.open("rb") as source:
        row_index = 0
        for raw_line in source:
            if not raw_line.strip():
                continue
            try:
                value = json.loads(raw_line.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError):
                value = None
            yield row_index, value if isinstance(value, dict) else None
            row_index += 1


def _iter_candidate_rows_fd(descriptor: int, suffix: str):
    """Stream rows through a stable no-follow file descriptor."""
    os.lseek(descriptor, 0, os.SEEK_SET)
    with os.fdopen(os.dup(descriptor), "rb") as stream:
        if suffix == ".parquet":
            import pyarrow.parquet as pq  # pyright: ignore[reportMissingImports]

            row_index = 0
            for batch in pq.ParquetFile(stream).iter_batches(batch_size=1_024):
                for raw_row in batch.to_pylist():
                    value: Any = raw_row
                    if isinstance(raw_row, dict) and set(raw_row) == {"raw_json"}:
                        encoded = raw_row["raw_json"]
                        try:
                            value = (
                                json.loads(encoded) if isinstance(encoded, str | bytes) else None
                            )
                        except (UnicodeDecodeError, json.JSONDecodeError):
                            value = None
                    yield row_index, value if isinstance(value, dict) else None
                    row_index += 1
            return
        row_index = 0
        for raw_line in stream:
            if not raw_line.strip():
                continue
            try:
                value = json.loads(raw_line.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError):
                value = None
            yield row_index, value if isinstance(value, dict) else None
            row_index += 1


def _sha256_fd(descriptor: int) -> str:
    digest = hashlib.sha256()
    os.lseek(descriptor, 0, os.SEEK_SET)
    with os.fdopen(os.dup(descriptor), "rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _stable_stat_identity(value: os.stat_result) -> tuple[int, int, int, int, int]:
    return value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns, value.st_ctime_ns


def _staged_tree_snapshot(root: Path) -> dict[str, tuple[int, int, int, int, int, int]]:
    """Capture lexical staged-tree identities while rejecting links and special files."""
    root_stat = os.lstat(root)
    if stat.S_ISLNK(root_stat.st_mode) or not stat.S_ISDIR(root_stat.st_mode):
        raise ValueError("authenticated physical shard root is not a no-follow directory")
    snapshot = {
        ".": (root_stat.st_mode, *_stable_stat_identity(root_stat)),
    }
    for path in root.rglob("*"):
        observed = os.lstat(path)
        relative = path.relative_to(root).as_posix()
        if stat.S_ISLNK(observed.st_mode):
            raise ValueError("authenticated physical shard set contains a symlink")
        if not stat.S_ISREG(observed.st_mode) and not stat.S_ISDIR(observed.st_mode):
            raise ValueError("authenticated physical shard set contains a non-regular entry")
        snapshot[relative] = (observed.st_mode, *_stable_stat_identity(observed))
    return snapshot


def _expected_staged_tree_entries(root: Path, files: Sequence[Path]) -> frozenset[str]:
    """Return every lexical file and ancestor directory permitted below ``root``."""
    entries = {"."}
    for path in files:
        relative = path.relative_to(root)
        entries.add(relative.as_posix())
        entries.update(parent.as_posix() for parent in relative.parents if parent != Path("."))
    return frozenset(entries)


def _open_verified_source_fd(path: Path, source_file: SourceFile) -> tuple[int, os.stat_result]:
    flags = os.O_RDONLY
    if not hasattr(os, "O_NOFOLLOW"):
        raise ValueError("stable source authentication requires O_NOFOLLOW")
    flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        raise ValueError(
            f"source file is not a no-follow regular file: {source_file.path}"
        ) from error
    initial = os.fstat(descriptor)
    if (
        not stat.S_ISREG(initial.st_mode)
        or initial.st_size != source_file.bytes
        or _sha256_fd(descriptor) != source_file.sha256
    ):
        os.close(descriptor)
        raise ValueError(f"source file stable identity mismatch: {source_file.path}")
    return descriptor, initial


def _verify_source_fd_stable(
    descriptor: int, initial: os.stat_result, path: Path, source_file: SourceFile
) -> None:
    final = os.fstat(descriptor)
    try:
        pathname = os.lstat(path)
    except OSError as error:
        raise ValueError(
            f"source file changed during authentication: {source_file.path}"
        ) from error
    if (
        _stable_stat_identity(final) != _stable_stat_identity(initial)
        or stat.S_ISLNK(pathname.st_mode)
        or not stat.S_ISREG(pathname.st_mode)
        or (pathname.st_dev, pathname.st_ino) != (final.st_dev, final.st_ino)
        or _sha256_fd(descriptor) != source_file.sha256
    ):
        raise ValueError(f"source file changed during authentication: {source_file.path}")


def _context_bucket(token_count: int) -> str:
    if token_count <= 4096:
        return "le4k"
    if token_count <= 16384:
        return "4k_16k"
    if token_count <= 32768:
        return "16k_32k"
    raise ValueError(f"conversation exceeds the 32K inventory limit: {token_count}")


def _source_key(source: SourceIdentity) -> tuple[str, ...]:
    return (
        source.repository_id,
        source.configuration,
        source.split,
        source.revision,
        source.cell,
        source.lane,
    )


def _staged_path(root: Path, source: SourceIdentity, file: SourceFile) -> Path:
    return root / "sources" / source.repository_id / source.revision / file.path


def _verified_candidate_files(
    inventory: SourceInventory,
) -> list[tuple[SourceIdentity, SourceFile, Path]]:
    if inventory.schema_version != 1 or inventory.staged_root is None:
        raise ValueError("candidate inventory requires a staged version-1 SourceInventory")
    root = inventory.staged_root.resolve(strict=True)
    if not root.is_dir():
        raise ValueError("staged source root is not a directory")
    if (
        _SHA256.fullmatch(inventory.manifest_sha256) is None
        or hashlib.sha256(inventory.canonical_manifest).hexdigest() != inventory.manifest_sha256
    ):
        raise ValueError("source inventory manifest identity mismatch")
    verified: list[tuple[SourceIdentity, SourceFile, Path]] = []
    logical_files: set[tuple[str, str, str]] = set()
    for source in sorted(inventory.sources, key=_source_key):
        if not source.approved_use or _REVISION.fullmatch(source.revision) is None:
            raise ValueError(f"unapproved or unpinned source identity: {source.repository_id}")
        if source.lane not in {
            "target-synth",
            "agentless-swe",
            "interactive-swe-replay",
            "generic-tool-replay",
        }:
            raise ValueError(f"unsupported source lane: {source.lane}")
        for file in sorted(source.files, key=lambda descriptor: descriptor.path):
            key = (source.repository_id, source.revision, file.path)
            if key in logical_files:
                raise ValueError(f"duplicate logical source file: {key}")
            logical_files.add(key)
            path = _staged_path(root, source, file).resolve(strict=False)
            if (
                not path.is_relative_to(root)
                or not path.is_file()
                or path.stat().st_size != file.bytes
                or sha256_file(path) != file.sha256
            ):
                raise ValueError(
                    f"source file identity mismatch: {source.repository_id}:{file.path}"
                )
            verified.append((source, file, path))
    return verified


def _normalize_language(value: Any, source: SourceIdentity) -> str:
    if value is None:
        if _approved_ptv2_source(source):
            value = (
                source.split.removeprefix("multilingual_")
                if source.split.startswith("multilingual_")
                else "en"
            )
        else:
            value = _APPROVED_SOURCE_LANGUAGES.get((source.repository_id, source.split))
    if not isinstance(value, str) or not value.strip():
        raise ValueError("invalid_language")
    normalized = value.strip().lower().replace("_", "-")
    normalized = _LANGUAGE_ALIASES.get(normalized, normalized.split("-", maxsplit=1)[0])
    if normalized not in _LANGUAGE_CODES:
        raise ValueError("invalid_language")
    return normalized


def _candidate_domain(source: SourceIdentity) -> str:
    if not _approved_ptv2_source(source):
        return source.cell
    expected_cell = source.split
    if source.cell != expected_cell:
        raise ValueError(
            f"approved PTV2 source cell does not match split: {source.cell!r} != {expected_cell!r}"
        )
    return {
        "chat": "instruction-chat",
        "stem": "stem-science",
    }.get(source.cell, "multilingual" if source.cell.startswith("multilingual_") else source.cell)


def _row_messages(row: dict[str, Any]) -> list[dict[str, Any]]:
    messages = row.get("messages") or row.get("conversations")
    if isinstance(messages, str):
        try:
            messages = json.loads(messages)
        except json.JSONDecodeError as error:
            raise ValueError("missing_messages") from error
    if (
        not isinstance(messages, list)
        or not messages
        or not all(isinstance(message, dict) for message in messages)
    ):
        raise ValueError("missing_messages")
    return deepcopy(messages)


def _row_tools(row: dict[str, Any]) -> list[dict[str, Any]]:
    tools = row.get("tools") or []
    if isinstance(tools, str):
        try:
            tools = json.loads(tools)
        except json.JSONDecodeError as error:
            raise ValueError("invalid_tools") from error
    if not isinstance(tools, list) or not all(isinstance(tool, dict) for tool in tools):
        raise ValueError("invalid_tools")
    return deepcopy(tools)


def _target_prompt(row: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    messages = _row_messages(row)
    tools = _row_tools(row)
    if messages[-1].get("role") == "assistant":
        messages.pop()
    if not messages:
        raise ValueError("missing_messages")
    has_tool_exchange = bool(tools) or any(
        message.get("role") == "tool" or message.get("tool_calls") for message in messages
    )
    if has_tool_exchange:
        raise ValueError("wrong_lane")
    return messages, tools


def _ptv2_target_prompt_identity(
    row: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], str, str]:
    """Extract a PTV2 prompt and bind its source-native terminal response."""
    messages = _row_messages(row)
    tools = _row_tools(row)
    if messages[-1].get("role") != "assistant":
        raise ValueError("PTV2 source-native assistant response is missing")
    source_conversation = canonical_json({"messages": messages, "tools": tools})
    source_response = canonical_json(messages[-1])
    prompt_messages = messages[:-1]
    if not prompt_messages:
        raise ValueError("missing_messages")
    has_tool_exchange = bool(tools) or any(
        message.get("role") == "tool" or message.get("tool_calls") for message in prompt_messages
    )
    if has_tool_exchange:
        raise ValueError("wrong_lane")
    return (
        prompt_messages,
        tools,
        sha256_bytes(source_conversation),
        sha256_bytes(source_response),
    )


def _candidate_tokenize(
    tokenizer: Any,
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]],
    *,
    add_generation_prompt: bool,
) -> tuple[int, ...]:
    encoded = tokenizer.apply_chat_template(
        messages,
        tools=tools or None,
        tokenize=True,
        add_generation_prompt=add_generation_prompt,
        return_dict=True,
    )
    input_ids = encoded.get("input_ids") if isinstance(encoded, Mapping) else encoded
    if not isinstance(input_ids, list) or any(
        isinstance(token, bool) or not isinstance(token, int) for token in input_ids
    ):
        raise ValueError("invalid_tokenization")
    return tuple(input_ids)


def _quarantine(counts: dict[str, int], reason: str) -> None:
    counts[reason] = counts.get(reason, 0) + 1


def _create_candidate_database(path: Path) -> sqlite3.Connection:
    connection = sqlite3.connect(path)
    connection.execute("PRAGMA journal_mode=DELETE")
    connection.execute("PRAGMA synchronous=NORMAL")
    connection.execute(
        """
        CREATE TABLE candidates (
            prompt_uuid TEXT PRIMARY KEY,
            canonical_bytes BLOB NOT NULL,
            source_id TEXT NOT NULL,
            source_revision TEXT NOT NULL,
            source_file_sha256 TEXT NOT NULL,
            source_row_index INTEGER NOT NULL,
            domain TEXT NOT NULL,
            language TEXT NOT NULL,
            lane TEXT NOT NULL,
            context_bucket TEXT NOT NULL,
            full_token_count INTEGER NOT NULL,
            source_manifest_sha256 TEXT NOT NULL,
            source_file_path TEXT NOT NULL,
            input_ids TEXT NOT NULL,
            tokenizer_sha256 TEXT NOT NULL,
            replay_valid INTEGER NOT NULL,
            source_family TEXT NOT NULL CHECK (source_family IN ('ptv2', 'ptv3')),
            source_repository_id TEXT NOT NULL,
            source_configuration TEXT NOT NULL,
            source_split TEXT NOT NULL,
            source_conversation_sha256 TEXT,
            source_response_sha256 TEXT,
            chat_template_sha256 TEXT
        ) WITHOUT ROWID
        """
    )
    return connection


def _insert_candidate(connection: sqlite3.Connection, candidate: CandidatePrompt) -> None:
    connection.execute(
        f"INSERT INTO candidates ({', '.join(_CANDIDATE_COLUMNS)}) "
        f"VALUES ({', '.join('?' for _ in _CANDIDATE_COLUMNS)})",
        (
            candidate.prompt_uuid,
            candidate.canonical_bytes,
            candidate.source_id,
            candidate.source_revision,
            candidate.source_file_sha256,
            candidate.source_row_index,
            candidate.domain,
            candidate.language,
            candidate.lane,
            candidate.context_bucket,
            candidate.full_token_count,
            candidate.source_manifest_sha256,
            candidate.source_file_path,
            json.dumps(candidate.input_ids, separators=(",", ":")),
            candidate.tokenizer_sha256,
            int(candidate.replay_valid),
            candidate.source_family,
            candidate.source_repository_id,
            candidate.source_configuration,
            candidate.source_split,
            candidate.source_conversation_sha256,
            candidate.source_response_sha256,
            candidate.chat_template_sha256,
        ),
    )


def _candidate_record(prompt: CandidatePrompt) -> dict[str, Any]:
    record: dict[str, Any] = {
        "type": "candidate",
        "prompt_uuid": prompt.prompt_uuid,
        "canonical_prompt": json.loads(prompt.canonical_bytes),
        "source_id": prompt.source_id,
        "source_revision": prompt.source_revision,
        "source_file_sha256": prompt.source_file_sha256,
        "source_row_index": prompt.source_row_index,
        "source_manifest_sha256": prompt.source_manifest_sha256,
        "source_file_path": prompt.source_file_path,
        "arm_domain": prompt.arm_domain,
        "lane": prompt.lane,
        "language": prompt.language,
        "context_bucket": prompt.context_bucket,
        "full_token_count": prompt.full_token_count,
        "input_ids": prompt.input_ids,
        "tokenizer_sha256": prompt.tokenizer_sha256,
        "replay_valid": prompt.replay_valid,
        "source_family": prompt.source_family,
        "source_repository_id": prompt.source_repository_id,
        "source_configuration": prompt.source_configuration,
        "source_split": prompt.source_split,
        "source_conversation_sha256": prompt.source_conversation_sha256,
        "source_response_sha256": prompt.source_response_sha256,
    }
    if prompt.chat_template_sha256 is not None:
        record["chat_template_sha256"] = prompt.chat_template_sha256
    return record


def _capacity_records(capacity: Mapping[CandidateCell, int]) -> list[dict[str, Any]]:
    return [
        {
            "arm_domain": cell.arm_domain,
            "lane": cell.lane,
            "language": cell.language,
            "context_bucket": cell.context_bucket,
            "count": count,
        }
        for cell, count in capacity.items()
    ]


def _identity_chunks(
    rows: Sequence[CandidatePrompt],
    source_manifest_sha256: str,
    tokenizer_sha256: str,
    capacity: Mapping[CandidateCell, int],
    quarantine_counts: Mapping[str, int],
    baseline_exclusion: ExclusionProof,
    held_out_exclusion: ExclusionProof,
    ptv2_revision: str,
    ptv2_allowlist_sha256: str,
):
    yield (
        canonical_json(
            {
                "type": "metadata",
                "schema_version": 1,
                "source_manifest_sha256": source_manifest_sha256,
                "tokenizer_sha256": tokenizer_sha256,
                "baseline_exclusion": asdict(baseline_exclusion),
                "held_out_exclusion": asdict(held_out_exclusion),
                "ptv2_revision": ptv2_revision,
                "ptv2_allowlist_sha256": ptv2_allowlist_sha256,
            }
        )
        + b"\n"
    )
    for prompt in rows:
        yield canonical_json(_candidate_record(prompt)) + b"\n"
    yield canonical_json({"type": "capacity", "cells": _capacity_records(capacity)}) + b"\n"
    yield canonical_json({"type": "quarantine", "counts": dict(quarantine_counts)}) + b"\n"


_THREAD_ENVIRONMENT = (
    "ARROW_NUM_THREADS",
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
)
_PROCESS_TOKENIZER: Any | None = None


@dataclass(frozen=True)
class _CandidateShardTask:
    index: int
    source: SourceIdentity
    descriptor: SourceFile
    path: Path
    staged_path: Path
    spool_path: Path
    source_manifest_sha256: str
    tokenizer_sha256: str
    chat_template_sha256: str | None
    training_seq_len: int


@dataclass(frozen=True)
class _CandidateShardResult:
    index: int
    spool_path: Path
    row_count: int
    spool_bytes: int
    spool_sha256: str
    elapsed_seconds: float
    worker_pid: int


@dataclass(frozen=True)
class _CandidateTokenizeTask:
    index: int
    spool_path: Path
    source: SourceIdentity


@dataclass(frozen=True)
class _CandidateTokenizeResult:
    index: int
    row_count: int
    elapsed_seconds: float
    worker_pid: int


def resolve_candidate_worker_count(
    *, requested_workers: int, declared_shard_count: int, environ: Mapping[str, str] | None = None
) -> tuple[int, int]:
    """Bound process workers by the allocation and the exact declared shard set."""
    if isinstance(requested_workers, bool) or requested_workers < 1:
        raise ValueError("candidate workers must be a positive integer")
    if not 1 <= declared_shard_count <= 201:
        raise ValueError("candidate processing requires between 1 and 201 shards")
    environment = os.environ if environ is None else environ
    raw_allocation = environment.get("SLURM_CPUS_PER_TASK")
    try:
        allocation = int(raw_allocation) if raw_allocation is not None else (os.cpu_count() or 1)
    except ValueError as error:
        raise ValueError("SLURM_CPUS_PER_TASK must be a positive integer") from error
    if allocation < 1:
        raise ValueError("candidate processing requires a positive CPU allocation")
    return min(requested_workers, allocation, declared_shard_count, 96), allocation


def _declared_candidate_files(
    inventory: SourceInventory,
) -> tuple[list[tuple[SourceIdentity, SourceFile, Path]], dict[str, tuple[int, ...]]]:
    """Authenticate the namespace cheaply; workers authenticate shard bytes in parallel."""
    if inventory.schema_version != 1 or inventory.staged_root is None:
        raise ValueError("candidate inventory requires a staged version-1 SourceInventory")
    root = inventory.staged_root.resolve(strict=True)
    if (
        _SHA256.fullmatch(inventory.manifest_sha256) is None
        or hashlib.sha256(inventory.canonical_manifest).hexdigest() != inventory.manifest_sha256
    ):
        raise ValueError("source inventory manifest identity mismatch")
    sources_root = root / "sources"
    initial_tree = _staged_tree_snapshot(sources_root)
    declared: list[tuple[SourceIdentity, SourceFile, Path]] = []
    logical_files: set[tuple[str, str, str]] = set()
    for source in sorted(inventory.sources, key=_source_key):
        if not source.approved_use or _REVISION.fullmatch(source.revision) is None:
            raise ValueError(f"unapproved or unpinned source identity: {source.repository_id}")
        if source.lane not in {
            "target-synth",
            "agentless-swe",
            "interactive-swe-replay",
            "generic-tool-replay",
        }:
            raise ValueError(f"unsupported source lane: {source.lane}")
        for descriptor in sorted(source.files, key=lambda item: item.path):
            key = (source.repository_id, source.revision, descriptor.path)
            if key in logical_files:
                raise ValueError(f"duplicate logical source file: {key}")
            logical_files.add(key)
            path = _staged_path(root, source, descriptor)
            try:
                observed = os.lstat(path)
            except OSError as error:
                raise ValueError(f"source file is missing: {descriptor.path}") from error
            if not stat.S_ISREG(observed.st_mode) or observed.st_size != descriptor.bytes:
                raise ValueError(f"source file identity mismatch: {descriptor.path}")
            declared.append((source, descriptor, path))
    if set(initial_tree) != _expected_staged_tree_entries(
        sources_root, [path for _source, _descriptor, path in declared]
    ):
        raise ValueError("authenticated physical shard set has undeclared entries")
    return declared, initial_tree


def _initialize_candidate_worker() -> None:
    for name in _THREAD_ENVIRONMENT:
        os.environ[name] = "1"


def _candidate_quarantine_payload(error: ValueError | TrajectoryValidationError) -> dict[str, str]:
    reason = error.reason if isinstance(error, TrajectoryValidationError) else str(error)
    if reason.startswith("conversation exceeds the 32K inventory limit"):
        reason = "context_too_long"
    if reason not in _CANDIDATE_QUARANTINE_CODES:
        raise error
    return {"reason": reason}


def _classify_candidate_row(
    *,
    source: SourceIdentity,
    descriptor: SourceFile,
    source_manifest_sha256: str,
    tokenizer_sha256: str,
    chat_template_sha256: str | None,
    training_seq_len: int,
    row_index: int,
    raw_row: Mapping[str, Any] | None,
    tokenizer: Any,
) -> dict[str, Any]:
    if raw_row is None:
        return {"reason": "invalid_row"}
    row = dict(raw_row)
    source_id = f"{source.repository_id}:{source.configuration}:{source.split}"
    try:
        source_conversation_sha256: str | None = None
        source_response_sha256: str | None = None
        if source.lane in {"interactive-swe-replay", "generic-tool-replay"}:
            validation = validate_trajectory(
                row,
                source_id=f"{source_id}:{row_index}",
                lane=source.lane,
                tokenizer=tokenizer,
                training_seq_len=training_seq_len,
            )
            messages = validation.canonical["messages"]
            tools = validation.canonical["tools"]
            canonical_bytes = validation.canonical_bytes
            replay_valid = True
        else:
            if _approved_ptv2_source(source):
                messages, tools, source_conversation_sha256, source_response_sha256 = (
                    _ptv2_target_prompt_identity(row)
                )
            else:
                messages, tools = _target_prompt(row)
            canonical_bytes = canonicalize_prompt(messages, tools)
            replay_valid = False
        prompt_uuid = sha256_bytes(canonical_bytes)
    except TrajectoryValidationError as error:
        return _candidate_quarantine_payload(error)
    except ValueError as error:
        return _candidate_quarantine_payload(error)
    candidate = CandidatePrompt(
        prompt_uuid=prompt_uuid,
        canonical_bytes=canonical_bytes,
        source_id=source_id,
        source_revision=source.revision,
        source_file_sha256=descriptor.sha256,
        source_row_index=row_index,
        domain=_candidate_domain(source),
        language="",
        lane=source.lane,
        context_bucket="",
        full_token_count=0,
        source_manifest_sha256=source_manifest_sha256,
        source_file_path=descriptor.path,
        input_ids=(),
        tokenizer_sha256=tokenizer_sha256,
        replay_valid=replay_valid,
        source_family="ptv2" if _approved_ptv2_source(source) else "ptv3",
        source_repository_id=source.repository_id,
        source_configuration=source.configuration,
        source_split=source.split,
        source_conversation_sha256=source_conversation_sha256,
        source_response_sha256=source_response_sha256,
        chat_template_sha256=chat_template_sha256,
    )
    record = _candidate_record(candidate)
    record.pop("language")
    record["raw_language"] = row.get("language")
    return {"prompt_uuid": prompt_uuid, "pretoken_candidate": record}


def _spool_candidate_payload(
    task: _CandidateShardTask, row_index: int, raw_row: Mapping[str, Any] | None
) -> dict[str, Any]:
    assert _PROCESS_TOKENIZER is not None
    return _classify_candidate_row(
        source=task.source,
        descriptor=task.descriptor,
        source_manifest_sha256=task.source_manifest_sha256,
        tokenizer_sha256=task.tokenizer_sha256,
        chat_template_sha256=task.chat_template_sha256,
        training_seq_len=task.training_seq_len,
        row_index=row_index,
        raw_row=raw_row,
        tokenizer=_PROCESS_TOKENIZER,
    )


def _stage_authenticated_source_once(
    source_path: Path, staged_path: Path, source_file: SourceFile
) -> None:
    """Authenticate one shared-storage stream while copying it to node-local storage."""
    source_flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(source_path, source_flags)
    initial = os.fstat(descriptor)
    if not stat.S_ISREG(initial.st_mode) or initial.st_size != source_file.bytes:
        os.close(descriptor)
        raise ValueError(f"source file stable identity mismatch: {source_file.path}")
    stage_flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
    staged_descriptor = os.open(staged_path, stage_flags, 0o600)
    digest = hashlib.sha256()
    try:
        with os.fdopen(os.dup(descriptor), "rb") as source_stream, os.fdopen(
            staged_descriptor, "wb"
        ) as staged_stream:
            for chunk in iter(lambda: source_stream.read(8 * 1024 * 1024), b""):
                digest.update(chunk)
                staged_stream.write(chunk)
            staged_stream.flush()
            os.fsync(staged_stream.fileno())
        final = os.fstat(descriptor)
        pathname = os.lstat(source_path)
        if (
            digest.hexdigest() != source_file.sha256
            or _stable_stat_identity(final) != _stable_stat_identity(initial)
            or stat.S_ISLNK(pathname.st_mode)
            or (pathname.st_dev, pathname.st_ino) != (final.st_dev, final.st_ino)
        ):
            raise ValueError(f"source file changed during authentication: {source_file.path}")
    except BaseException:
        os.close(descriptor)
        staged_path.unlink(missing_ok=True)
        raise
    os.close(descriptor)


def _process_candidate_shard(task: _CandidateShardTask) -> _CandidateShardResult:
    started = time.monotonic_ns()
    _stage_authenticated_source_once(task.path, task.staged_path, task.descriptor)
    staged_initial = os.lstat(task.staged_path)
    connection = sqlite3.connect(task.spool_path)
    connection.execute("PRAGMA journal_mode=DELETE")
    connection.execute("PRAGMA synchronous=NORMAL")
    connection.execute(
        "CREATE TABLE records (source_row_index INTEGER PRIMARY KEY, payload BLOB NOT NULL) WITHOUT ROWID"
    )
    row_count = 0
    try:
        for row_index, raw_row in _iter_candidate_rows(task.staged_path):
            payload = _spool_candidate_payload(task, row_index, raw_row)
            connection.execute(
                "INSERT INTO records VALUES (?, ?)", (row_index, canonical_json(payload))
            )
            row_count += 1
            if row_count % 10_000 == 0:
                connection.commit()
        connection.commit()
        staged_final = os.lstat(task.staged_path)
        if (
            _stable_stat_identity(staged_final) != _stable_stat_identity(staged_initial)
            or sha256_file(task.staged_path) != task.descriptor.sha256
        ):
            raise ValueError("node-local staged source changed during candidate processing")
    except BaseException:
        connection.close()
        task.spool_path.unlink(missing_ok=True)
        raise
    connection.close()
    return _CandidateShardResult(
        task.index,
        task.spool_path,
        row_count,
        task.spool_path.stat().st_size,
        sha256_file(task.spool_path),
        round((time.monotonic_ns() - started) / 1_000_000_000, 6),
        os.getpid(),
    )


def _run_candidate_shard_workers(
    tasks: tuple[_CandidateShardTask, ...], workers: int
) -> tuple[_CandidateShardResult, ...]:
    results: dict[int, _CandidateShardResult] = {}
    with ProcessPoolExecutor(
        max_workers=workers,
        mp_context=get_context("fork"),
        initializer=_initialize_candidate_worker,
    ) as executor:
        futures = {executor.submit(_process_candidate_shard, task): task.index for task in tasks}
        try:
            for future in as_completed(futures):
                result = future.result()
                results[result.index] = result
        except BaseException:
            for future in futures:
                future.cancel()
            raise
    if len(results) != len(tasks):
        raise RuntimeError("candidate processing did not return every declared shard")
    return tuple(results[index] for index in range(len(tasks)))


def _tokenize_candidate_payload(
    payload: Mapping[str, Any], *, tokenizer: Any, source: SourceIdentity
) -> dict[str, Any]:
    if payload.get("reason") is not None:
        return dict(payload)
    record_value = payload.get("pretoken_candidate")
    if not isinstance(record_value, Mapping):
        raise ValueError("candidate identity is missing before tokenization")
    record = dict(record_value)
    canonical_prompt = record.get("canonical_prompt")
    if not isinstance(canonical_prompt, dict):
        raise ValueError("candidate canonical prompt is missing before tokenization")
    raw_language = record.pop("raw_language", None)
    try:
        record["language"] = _normalize_language(raw_language, source)
        input_ids = _candidate_tokenize(
            tokenizer,
            canonical_prompt["messages"],
            canonical_prompt["tools"],
            add_generation_prompt=not record["replay_valid"],
        )
        record["input_ids"] = input_ids
        record["full_token_count"] = len(input_ids)
        record["context_bucket"] = _context_bucket(len(input_ids))
    except (TrajectoryValidationError, ValueError) as error:
        return _candidate_quarantine_payload(error)
    return {"candidate": record}


def _tokenize_candidate_shard(task: _CandidateTokenizeTask) -> _CandidateTokenizeResult:
    started = time.monotonic_ns()
    assert _PROCESS_TOKENIZER is not None
    connection = sqlite3.connect(task.spool_path)
    connection.execute(
        "CREATE TABLE tokenized(source_row_index INTEGER PRIMARY KEY,payload BLOB NOT NULL) "
        "WITHOUT ROWID"
    )
    row_count = 0
    try:
        for source_row_index, encoded in connection.execute(
            "SELECT source_row_index,payload FROM selected ORDER BY source_row_index"
        ):
            payload = json.loads(encoded)
            output = _tokenize_candidate_payload(
                payload, tokenizer=_PROCESS_TOKENIZER, source=task.source
            )
            connection.execute(
                "INSERT INTO tokenized VALUES(?,?)",
                (source_row_index, canonical_json(output)),
            )
            row_count += 1
            if row_count % 10_000 == 0:
                connection.commit()
        connection.commit()
    except BaseException:
        connection.close()
        raise
    connection.close()
    return _CandidateTokenizeResult(
        task.index,
        row_count,
        round((time.monotonic_ns() - started) / 1_000_000_000, 6),
        os.getpid(),
    )


def _run_candidate_tokenizers(
    tasks: tuple[_CandidateTokenizeTask, ...], workers: int
) -> tuple[_CandidateTokenizeResult, ...]:
    results: dict[int, _CandidateTokenizeResult] = {}
    with ProcessPoolExecutor(
        max_workers=workers,
        mp_context=get_context("fork"),
        initializer=_initialize_candidate_worker,
    ) as executor:
        futures = {executor.submit(_tokenize_candidate_shard, task): task.index for task in tasks}
        try:
            for future in as_completed(futures):
                result = future.result()
                results[result.index] = result
        except BaseException:
            for future in futures:
                future.cancel()
            raise
    if len(results) != len(tasks):
        raise RuntimeError("candidate tokenization did not return every declared shard")
    return tuple(results[index] for index in range(len(tasks)))


def _candidate_from_record(record: Mapping[str, Any]) -> CandidatePrompt:
    return CandidatePrompt(
        prompt_uuid=record["prompt_uuid"],
        canonical_bytes=canonical_json(record["canonical_prompt"]),
        source_id=record["source_id"],
        source_revision=record["source_revision"],
        source_file_sha256=record["source_file_sha256"],
        source_row_index=record["source_row_index"],
        domain=record["arm_domain"],
        language=record["language"],
        lane=record["lane"],
        context_bucket=record["context_bucket"],
        full_token_count=record["full_token_count"],
        source_manifest_sha256=record["source_manifest_sha256"],
        source_file_path=record["source_file_path"],
        input_ids=tuple(record["input_ids"]),
        tokenizer_sha256=record["tokenizer_sha256"],
        replay_valid=record["replay_valid"],
        source_family=record["source_family"],
        source_repository_id=record["source_repository_id"],
        source_configuration=record["source_configuration"],
        source_split=record["source_split"],
        source_conversation_sha256=record["source_conversation_sha256"],
        source_response_sha256=record["source_response_sha256"],
        chat_template_sha256=record.get("chat_template_sha256"),
    )


def _exclude_or_quarantine_classified_candidate(
    payload: Mapping[str, Any],
    *,
    historical_prompt_ids: set[str],
    held_out_prompt_ids: set[str],
    quarantine_counts: dict[str, int],
) -> bool:
    reason = _classified_candidate_reason(
        payload,
        historical_prompt_ids=historical_prompt_ids,
        held_out_prompt_ids=held_out_prompt_ids,
    )
    if reason is None:
        return False
    _quarantine(quarantine_counts, reason)
    return True


def _classified_candidate_reason(
    payload: Mapping[str, Any],
    *,
    historical_prompt_ids: set[str],
    held_out_prompt_ids: set[str],
) -> str | None:
    """Return the exact phase-one exclusion/quarantine reason without mutating counts."""
    prompt_id = payload.get("prompt_uuid")
    if prompt_id in historical_prompt_ids:
        return "historical_exclusion"
    if prompt_id in held_out_prompt_ids:
        return "heldout_exclusion"
    reason = payload.get("reason")
    if reason is not None:
        if not isinstance(reason, str) or reason not in _CANDIDATE_QUARANTINE_CODES:
            raise ValueError("candidate classification quarantine reason is invalid")
        return reason
    if not isinstance(prompt_id, str) or not isinstance(payload.get("pretoken_candidate"), dict):
        raise ValueError("candidate identity is missing before tokenization")
    return None


def _diagnostic_exemplar(
    exemplars: dict[str, list[dict[str, Any]]],
    reason: str,
    *,
    shard_index: int,
    source_row_index: int,
    payload: Mapping[str, Any],
) -> None:
    bucket = exemplars.setdefault(reason, [])
    if len(bucket) >= 3:
        return
    prompt_uuid = payload.get("prompt_uuid")
    candidate = payload.get("candidate") or payload.get("pretoken_candidate")
    if not isinstance(prompt_uuid, str) and isinstance(candidate, Mapping):
        prompt_uuid = candidate.get("prompt_uuid")
    bucket.append(
        {
            "shard_index": shard_index,
            "source_row_index": source_row_index,
            "prompt_uuid": prompt_uuid if isinstance(prompt_uuid, str) else None,
            "payload_keys": sorted(str(key) for key in payload),
        }
    )


def _raw_row_schema(path: Path) -> dict[str, Any]:
    for _row_index, row in _iter_candidate_rows(path):
        if row is None:
            return {"row_type": "null"}
        schema: dict[str, Any] = {
            str(key): type(value).__name__ for key, value in sorted(row.items())
        }
        record_json = row.get("record_json")
        if isinstance(record_json, str):
            try:
                decoded = json.loads(record_json)
            except json.JSONDecodeError:
                schema["record_json_decoded"] = "invalid_json"
            else:
                schema["record_json_decoded"] = type(decoded).__name__
                if isinstance(decoded, dict):
                    schema["record_json_fields"] = {
                        str(key): type(value).__name__
                        for key, value in sorted(decoded.items())
                    }
        return schema
    return {"row_type": "empty"}


def _merge_tokenized_candidate(
    connection: sqlite3.Connection,
    payload: Mapping[str, Any],
    *,
    capacity: dict[CandidateCell, int],
    quarantine_counts: dict[str, int],
) -> bool:
    reason = payload.get("reason")
    if reason is not None:
        if not isinstance(reason, str) or reason not in _CANDIDATE_QUARANTINE_CODES:
            raise ValueError("candidate tokenization quarantine reason is invalid")
        _quarantine(quarantine_counts, reason)
        return False
    record = payload.get("candidate")
    if not isinstance(record, Mapping):
        raise ValueError("tokenized candidate payload is missing")
    candidate = _candidate_from_record(record)
    previous = connection.execute(
        "SELECT canonical_bytes FROM candidates WHERE prompt_uuid = ?", (candidate.prompt_uuid,)
    ).fetchone()
    if previous is not None:
        if previous[0] != candidate.canonical_bytes:
            raise UUIDCollisionError(f"UUID collision: {candidate.prompt_uuid}")
        _quarantine(quarantine_counts, "duplicate_prompt_uuid")
        return False
    _insert_candidate(connection, candidate)
    cell = CandidateCell(
        candidate.arm_domain,
        candidate.lane,
        candidate.language,
        candidate.context_bucket,
    )
    capacity[cell] = capacity.get(cell, 0) + 1
    return True


def _finalize_candidate_inventory(
    *,
    source_inventory: SourceInventory,
    tokenizer_sha256: str,
    baseline_exclusion: ExclusionReceipt,
    held_out_exclusion: ExclusionReceipt,
    database_path: Path,
    storage_root: Path,
    accepted_count: int,
    capacity: dict[CandidateCell, int],
    quarantine_counts: dict[str, int],
    execution_receipt: Mapping[str, Any] | None = None,
    diagnostic_receipt: Mapping[str, Any] | None = None,
) -> CandidateInventory:
    capacity = dict(
        sorted(
            capacity.items(),
            key=lambda item: (
                item[0].arm_domain,
                item[0].lane,
                item[0].language,
                item[0].context_bucket,
            ),
        )
    )
    quarantine_counts = dict(sorted(quarantine_counts.items()))
    baseline_proof = ExclusionProof(
        baseline_exclusion.receipt_sha256,
        baseline_exclusion.prompt_ids_sha256,
        baseline_exclusion.prompt_id_count,
        quarantine_counts.get("historical_exclusion", 0),
    )
    held_out_proof = ExclusionProof(
        held_out_exclusion.receipt_sha256,
        held_out_exclusion.prompt_ids_sha256,
        held_out_exclusion.prompt_id_count,
        quarantine_counts.get("heldout_exclusion", 0),
    )
    rows = DiskBackedCandidateRows(
        database_path,
        accepted_count,
        source_inventory.manifest_sha256,
        tokenizer_sha256,
        storage_root,
    )
    digest = hashlib.sha256()
    for chunk in _identity_chunks(
        rows,
        rows.source_manifest_sha256,
        rows.tokenizer_sha256,
        capacity,
        quarantine_counts,
        baseline_proof,
        held_out_proof,
        _PTV2_REVISION,
        APPROVED_PTV2_ALLOWLIST_SHA256,
    ):
        digest.update(chunk)
    return CandidateInventory(
        rows=rows,
        capacity=MappingProxyType(capacity),
        quarantine_counts=MappingProxyType(quarantine_counts),
        inventory_sha256=digest.hexdigest(),
        baseline_exclusion=baseline_proof,
        held_out_exclusion=held_out_proof,
        ptv2_revision=_PTV2_REVISION,
        ptv2_allowlist_sha256=APPROVED_PTV2_ALLOWLIST_SHA256,
        execution_receipt=execution_receipt,
        diagnostic_receipt=diagnostic_receipt,
    )


def _build_candidate_inventory_parallel(
    source_inventory: SourceInventory,
    *,
    tokenizer: Any,
    tokenizer_sha256: str,
    chat_template_sha256: str | None,
    baseline_exclusion: ExclusionReceipt,
    held_out_exclusion: ExclusionReceipt,
    training_seq_len: int,
    storage_dir: Path | None,
    workers: int,
    source_commit: str,
) -> CandidateInventory:
    historical_prompt_ids = _validate_exclusion_receipt(baseline_exclusion, "baseline")
    held_out_prompt_ids = _validate_exclusion_receipt(held_out_exclusion, "held-out")
    files, initial_tree = _declared_candidate_files(source_inventory)
    effective_workers, allocated_cpus = resolve_candidate_worker_count(
        requested_workers=workers, declared_shard_count=len(files)
    )
    storage_parent = storage_dir or Path(tempfile.gettempdir())
    storage_parent.mkdir(parents=True, exist_ok=True)
    storage_root = Path(tempfile.mkdtemp(prefix="specdec-candidates-", dir=storage_parent))
    spool_root = storage_root / "shards"
    spool_root.mkdir(mode=0o700)
    stage_root = storage_root / "staged"
    stage_root.mkdir(mode=0o700)
    database_path = storage_root / "candidates.sqlite3"
    connection = _create_candidate_database(database_path)
    started_wall_ns = time.time_ns()
    started_monotonic_ns = time.monotonic_ns()
    global _PROCESS_TOKENIZER
    _PROCESS_TOKENIZER = tokenizer
    _initialize_candidate_worker()
    thread_environment = {name: os.environ.get(name) for name in _THREAD_ENVIRONMENT}
    if set(thread_environment.values()) != {"1"}:
        raise ValueError("parallel candidate thread caps did not reconcile")
    tasks = tuple(
        _CandidateShardTask(
            index,
            source,
            descriptor,
            path,
            stage_root / f"shard-{index:03d}{path.suffix}",
            spool_root / f"shard-{index:03d}.sqlite3",
            source_inventory.manifest_sha256,
            tokenizer_sha256,
            chat_template_sha256,
            training_seq_len,
        )
        for index, (source, descriptor, path) in enumerate(files)
    )
    capacity: dict[CandidateCell, int] = {}
    quarantine_counts: dict[str, int] = {}
    diagnostic_shards: dict[int, dict[str, Any]] = {}
    reason_exemplars: dict[str, list[dict[str, Any]]] = {}
    accepted_count = 0
    try:
        results = _run_candidate_shard_workers(tasks, effective_workers)
        assert source_inventory.staged_root is not None
        sources_root = source_inventory.staged_root.resolve(strict=True) / "sources"
        if _staged_tree_snapshot(sources_root) != initial_tree:
            raise ValueError("authenticated physical shard set changed during processing")
        for result in results:
            task = tasks[result.index]
            observed = os.lstat(result.spool_path)
            if (
                not stat.S_ISREG(observed.st_mode)
                or observed.st_size != result.spool_bytes
                or sha256_file(result.spool_path) != result.spool_sha256
            ):
                raise ValueError("candidate worker spool changed before deterministic merge")
            with sqlite3.connect(result.spool_path) as shard:
                shard.execute(
                    "CREATE TABLE selected(source_row_index INTEGER PRIMARY KEY,payload BLOB NOT NULL) "
                    "WITHOUT ROWID"
                )
                classification_counts: dict[str, int] = {}
                selected_count = 0
                for source_row_index, raw_payload in shard.execute(
                    "SELECT source_row_index,payload FROM records ORDER BY source_row_index"
                ):
                    payload = json.loads(raw_payload)
                    classification_reason = _classified_candidate_reason(
                        payload,
                        historical_prompt_ids=historical_prompt_ids,
                        held_out_prompt_ids=held_out_prompt_ids,
                    )
                    if classification_reason is not None:
                        _quarantine(classification_counts, classification_reason)
                        _diagnostic_exemplar(
                            reason_exemplars,
                            classification_reason,
                            shard_index=result.index,
                            source_row_index=source_row_index,
                            payload=payload,
                        )
                    if _exclude_or_quarantine_classified_candidate(
                        payload,
                        historical_prompt_ids=historical_prompt_ids,
                        held_out_prompt_ids=held_out_prompt_ids,
                        quarantine_counts=quarantine_counts,
                    ):
                        continue
                    shard.execute(
                        "INSERT INTO selected VALUES(?,?)", (source_row_index, raw_payload)
                    )
                    selected_count += 1
                shard.commit()
            diagnostic_shards[result.index] = {
                "index": result.index,
                "source_file_path": task.descriptor.path,
                "source_file_sha256": task.descriptor.sha256,
                "phase1_row_count": result.row_count,
                "selected_for_tokenization_count": selected_count,
                "classification_counts": dict(sorted(classification_counts.items())),
                "raw_row_schema": _raw_row_schema(task.staged_path),
            }
        connection.commit()
        tokenization_results = _run_candidate_tokenizers(
            tuple(
                _CandidateTokenizeTask(task.index, task.spool_path, task.source) for task in tasks
            ),
            effective_workers,
        )
        for task in tasks:
            tokenization_counts: dict[str, int] = {}
            shard_accepted_count = 0
            with sqlite3.connect(task.spool_path) as shard:
                for source_row_index, raw_payload, classified_payload in shard.execute(
                    "SELECT tokenized.source_row_index,tokenized.payload,selected.payload "
                    "FROM tokenized JOIN selected USING(source_row_index) "
                    "ORDER BY tokenized.source_row_index"
                ):
                    payload = json.loads(raw_payload)
                    merged = _merge_tokenized_candidate(
                        connection,
                        payload,
                        capacity=capacity,
                        quarantine_counts=quarantine_counts,
                    )
                    if merged:
                        accepted_count += 1
                        shard_accepted_count += 1
                        if accepted_count % 10_000 == 0:
                            connection.commit()
                    else:
                        reason = payload.get("reason")
                        if reason is None:
                            reason = "duplicate_prompt_uuid"
                        if not isinstance(reason, str):
                            raise ValueError("candidate diagnostic reason is invalid")
                        _quarantine(tokenization_counts, reason)
                        exemplar_payload = payload
                        if payload.get("candidate") is None:
                            exemplar_payload = json.loads(classified_payload)
                        _diagnostic_exemplar(
                            reason_exemplars,
                            reason,
                            shard_index=task.index,
                            source_row_index=source_row_index,
                            payload=exemplar_payload,
                        )
            diagnostic_shards[task.index].update(
                {
                    "phase2_row_count": tokenization_results[task.index].row_count,
                    "accepted_count": shard_accepted_count,
                    "tokenization_counts": dict(sorted(tokenization_counts.items())),
                }
            )
        connection.commit()
        connection.close()
        finished_wall_ns = time.time_ns()
        execution: dict[str, Any] = {
            "schema_version": 1,
            "source_commit": source_commit,
            "source_manifest_sha256": source_inventory.manifest_sha256,
            "declared_shard_count": len(files),
            "allocated_cpus": allocated_cpus,
            "requested_workers": workers,
            "effective_workers": effective_workers,
            "threads_per_worker": 1,
            "thread_environment": thread_environment,
            "accepted_count": accepted_count,
            "quarantine_counts": dict(sorted(quarantine_counts.items())),
            "started_at_ns": started_wall_ns,
            "finished_at_ns": finished_wall_ns,
            "elapsed_seconds": round(
                (time.monotonic_ns() - started_monotonic_ns) / 1_000_000_000, 6
            ),
            "shards": [asdict(result) | {"spool_path": result.spool_path.name} for result in results],
            "tokenization_shards": [asdict(result) for result in tokenization_results],
        }
        execution["receipt_sha256"] = sha256_bytes(canonical_json(execution))
        diagnostic: dict[str, Any] = {
            "schema_version": 1,
            "source_commit": source_commit,
            "source_manifest_sha256": source_inventory.manifest_sha256,
            "tokenizer_sha256": tokenizer_sha256,
            "chat_template_sha256": chat_template_sha256,
            "baseline_exclusion": {
                "receipt_sha256": baseline_exclusion.receipt_sha256,
                "prompt_ids_sha256": baseline_exclusion.prompt_ids_sha256,
                "prompt_id_count": baseline_exclusion.prompt_id_count,
            },
            "held_out_exclusion": {
                "receipt_sha256": held_out_exclusion.receipt_sha256,
                "prompt_ids_sha256": held_out_exclusion.prompt_ids_sha256,
                "prompt_id_count": held_out_exclusion.prompt_id_count,
            },
            "declared_shard_count": len(files),
            "allocated_cpus": allocated_cpus,
            "requested_workers": workers,
            "effective_workers": effective_workers,
            "accepted_count": accepted_count,
            "quarantine_counts": dict(sorted(quarantine_counts.items())),
            "reason_exemplars": dict(sorted(reason_exemplars.items())),
            "shards": [diagnostic_shards[index] for index in range(len(files))],
        }
        diagnostic["receipt_sha256"] = sha256_bytes(canonical_json(diagnostic))
        return _finalize_candidate_inventory(
            source_inventory=source_inventory,
            tokenizer_sha256=tokenizer_sha256,
            baseline_exclusion=baseline_exclusion,
            held_out_exclusion=held_out_exclusion,
            database_path=database_path,
            storage_root=storage_root,
            accepted_count=accepted_count,
            capacity=capacity,
            quarantine_counts=quarantine_counts,
            execution_receipt=MappingProxyType(execution),
            diagnostic_receipt=MappingProxyType(diagnostic),
        )
    except BaseException:
        connection.close()
        shutil.rmtree(storage_root, ignore_errors=True)
        raise
    finally:
        _PROCESS_TOKENIZER = None


def build_candidate_inventory(
    source_inventory: SourceInventory,
    *,
    tokenizer: Any,
    tokenizer_sha256: str,
    chat_template_sha256: str | None = None,
    baseline_exclusion: ExclusionReceipt,
    held_out_exclusion: ExclusionReceipt,
    training_seq_len: int = 4_096,
    storage_dir: Path | None = None,
    workers: int = 1,
    source_commit: str | None = None,
) -> CandidateInventory:
    """Build canonical B-prime/C/D candidates from one verified staged inventory."""
    if (
        _SHA256.fullmatch(tokenizer_sha256) is None
        or (chat_template_sha256 is not None and _SHA256.fullmatch(chat_template_sha256) is None)
        or training_seq_len < 1
        or isinstance(workers, bool)
        or not isinstance(workers, int)
        or workers < 1
    ):
        raise ValueError("tokenizer and training sequence length must be pinned")
    if workers > 1:
        resolved_commit = source_commit or os.environ.get("SOURCE_COMMIT", "")
        if _REVISION.fullmatch(resolved_commit) is None:
            raise ValueError("parallel candidate processing requires an exact source commit")
        return _build_candidate_inventory_parallel(
            source_inventory,
            tokenizer=tokenizer,
            tokenizer_sha256=tokenizer_sha256,
            chat_template_sha256=chat_template_sha256,
            baseline_exclusion=baseline_exclusion,
            held_out_exclusion=held_out_exclusion,
            training_seq_len=training_seq_len,
            storage_dir=storage_dir,
            workers=workers,
            source_commit=resolved_commit,
        )
    historical_prompt_ids = _validate_exclusion_receipt(baseline_exclusion, "baseline")
    held_out_prompt_ids = _validate_exclusion_receipt(held_out_exclusion, "held-out")
    files = _verified_candidate_files(source_inventory)
    capacity: dict[CandidateCell, int] = {}
    quarantine_counts: dict[str, int] = {}
    storage_parent = storage_dir or Path(tempfile.gettempdir())
    storage_parent.mkdir(parents=True, exist_ok=True)
    storage_root = Path(tempfile.mkdtemp(prefix="specdec-candidates-", dir=storage_parent))
    database_path = storage_root / "candidates.sqlite3"
    connection = _create_candidate_database(database_path)
    accepted_count = 0
    try:
        for source, descriptor, path in files:
            for row_index, raw_row in _iter_candidate_rows(path):
                classified = _classify_candidate_row(
                    source=source,
                    descriptor=descriptor,
                    source_manifest_sha256=source_inventory.manifest_sha256,
                    tokenizer_sha256=tokenizer_sha256,
                    chat_template_sha256=chat_template_sha256,
                    training_seq_len=training_seq_len,
                    row_index=row_index,
                    raw_row=raw_row,
                    tokenizer=tokenizer,
                )
                if _exclude_or_quarantine_classified_candidate(
                    classified,
                    historical_prompt_ids=historical_prompt_ids,
                    held_out_prompt_ids=held_out_prompt_ids,
                    quarantine_counts=quarantine_counts,
                ):
                    continue
                tokenized = _tokenize_candidate_payload(
                    classified, tokenizer=tokenizer, source=source
                )
                if _merge_tokenized_candidate(
                    connection,
                    tokenized,
                    capacity=capacity,
                    quarantine_counts=quarantine_counts,
                ):
                    accepted_count += 1
                    if accepted_count % 10_000 == 0:
                        connection.commit()
        connection.commit()
    except BaseException:
        connection.close()
        shutil.rmtree(storage_root)
        raise
    connection.close()
    return _finalize_candidate_inventory(
        source_inventory=source_inventory,
        tokenizer_sha256=tokenizer_sha256,
        baseline_exclusion=baseline_exclusion,
        held_out_exclusion=held_out_exclusion,
        database_path=database_path,
        storage_root=storage_root,
        accepted_count=accepted_count,
        capacity=capacity,
        quarantine_counts=quarantine_counts,
    )


def build_candidate_inventory_from_snapshot(
    source_inventory: SourceInventory,
    *,
    tokenizer_snapshot_receipt: Path,
    tokenizer_snapshot_receipt_sha256: str,
    baseline_exclusion: ExclusionReceipt,
    held_out_exclusion: ExclusionReceipt,
    training_seq_len: int = 4_096,
    storage_dir: Path | None = None,
    workers: int = 1,
    source_commit: str | None = None,
) -> CandidateInventory:
    """Build candidates with a tokenizer loaded from a caller-pinned local snapshot."""
    snapshot, tokenizer = _authenticated_snapshot_tokenizer(
        tokenizer_snapshot_receipt, tokenizer_snapshot_receipt_sha256
    )
    inventory = build_candidate_inventory(
        source_inventory,
        tokenizer=tokenizer,
        tokenizer_sha256=snapshot.tokenizer_sha256,
        chat_template_sha256=snapshot.chat_template_sha256,
        baseline_exclusion=baseline_exclusion,
        held_out_exclusion=held_out_exclusion,
        training_seq_len=training_seq_len,
        storage_dir=storage_dir,
        workers=workers,
        source_commit=source_commit,
    )
    if (
        load_tokenizer_snapshot(tokenizer_snapshot_receipt, tokenizer_snapshot_receipt_sha256)
        != snapshot
    ):
        inventory.close()
        raise ValueError("tokenizer snapshot changed while building candidates")
    return inventory


def verify_candidate_inventory_membership(
    inventory: CandidateInventory,
    source_inventory: SourceInventory,
    *,
    tokenizer_snapshot_receipt: Path,
    tokenizer_snapshot_receipt_sha256: str,
) -> None:
    """Rejoin every candidate to its authenticated staged physical source row."""
    snapshot, tokenizer = _authenticated_snapshot_tokenizer(
        tokenizer_snapshot_receipt, tokenizer_snapshot_receipt_sha256
    )
    tokenizer_sha256 = snapshot.tokenizer_sha256
    files = _verified_candidate_files(source_inventory)
    assert source_inventory.staged_root is not None
    staged_root = source_inventory.staged_root.resolve(strict=True)
    sources_root = staged_root / "sources"
    initial_tree = _staged_tree_snapshot(sources_root)
    expected_tree = _expected_staged_tree_entries(
        sources_root, [path for _source, _source_file, path in files]
    )
    if set(initial_tree) != expected_tree:
        raise ValueError("staged Task 3 physical shard set does not match its receipt")

    descriptor, temporary_name = tempfile.mkstemp(prefix="bprime-membership-", suffix=".sqlite3")
    os.close(descriptor)
    database = Path(temporary_name)
    connection = sqlite3.connect(database)
    try:
        connection.executescript(
            "CREATE TABLE wanted("
            "repository_id TEXT,configuration TEXT,split TEXT,revision TEXT,file_path TEXT,"
            "source_row INTEGER,prompt_uuid TEXT,canonical_bytes BLOB,source_id TEXT,language TEXT,"
            "domain TEXT,lane TEXT,context_bucket TEXT,full_token_count INTEGER,input_ids TEXT,"
            "tokenizer_sha256 TEXT,chat_template_sha256 TEXT,source_conversation_sha256 TEXT,"
            "source_response_sha256 TEXT,"
            "PRIMARY KEY(repository_id,configuration,split,revision,file_path,source_row));"
        )
        observed_capacity: dict[CandidateCell, int] = {}
        for candidate in inventory.rows:
            if not isinstance(candidate, CandidatePrompt):
                raise TypeError("candidate inventory rows must be CandidatePrompt values")
            row = candidate
            inserted = connection.execute(
                "INSERT OR IGNORE INTO wanted(repository_id,configuration,split,revision,file_path,"
                "source_row,prompt_uuid,canonical_bytes,source_id,language,domain,lane,context_bucket,"
                "full_token_count,input_ids,tokenizer_sha256,chat_template_sha256,"
                "source_conversation_sha256,source_response_sha256) "
                "VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (
                    row.source_repository_id,
                    row.source_configuration,
                    row.source_split,
                    row.source_revision,
                    row.source_file_path,
                    row.source_row_index,
                    row.prompt_uuid,
                    row.canonical_bytes,
                    row.source_id,
                    row.language,
                    row.domain,
                    row.lane,
                    row.context_bucket,
                    row.full_token_count,
                    json.dumps(row.input_ids, separators=(",", ":")),
                    row.tokenizer_sha256,
                    row.chat_template_sha256,
                    row.source_conversation_sha256,
                    row.source_response_sha256,
                ),
            )
            if inserted.rowcount != 1:
                raise ValueError("candidate inventory repeats a physical Task 3 row")
            cell = CandidateCell(row.domain, row.lane, row.language, row.context_bucket)
            observed_capacity[cell] = observed_capacity.get(cell, 0) + 1
        connection.commit()
        if observed_capacity != dict(inventory.capacity):
            raise ValueError("candidate inventory capacity does not reconcile with its rows")

        matched = 0
        for source, _file, path in files:
            wanted_rows = iter(
                connection.execute(
                    "SELECT source_row,prompt_uuid,canonical_bytes,source_id,language,domain,lane,"
                    "context_bucket,full_token_count,input_ids,tokenizer_sha256,"
                    "chat_template_sha256,"
                    "source_conversation_sha256,source_response_sha256 "
                    "FROM wanted WHERE repository_id=? AND configuration=? AND split=? "
                    "AND revision=? AND file_path=? ORDER BY source_row",
                    (
                        source.repository_id,
                        source.configuration,
                        source.split,
                        source.revision,
                        _file.path,
                    ),
                )
            )
            wanted = next(wanted_rows, None)
            file_descriptor, initial_stat = _open_verified_source_fd(path, _file)
            try:
                if wanted is not None:
                    for row_index, raw_row in _iter_candidate_rows_fd(file_descriptor, path.suffix):
                        if row_index < wanted[0]:
                            continue
                        if row_index != wanted[0] or raw_row is None:
                            raise ValueError("candidate does not match its physical Task 3 row")
                        try:
                            raw = dict(raw_row)
                            (
                                messages,
                                tools,
                                source_conversation_sha256,
                                source_response_sha256,
                            ) = _ptv2_target_prompt_identity(raw)
                            canonical_bytes = canonicalize_prompt(messages, tools)
                            language = _normalize_language(raw.get("language"), source)
                            input_ids = _candidate_tokenize(
                                tokenizer, messages, tools, add_generation_prompt=True
                            )
                            context_bucket = _context_bucket(len(input_ids))
                        except (TypeError, ValueError) as error:
                            raise ValueError(
                                "candidate does not match its physical Task 3 row or tokenization"
                            ) from error
                        expected = (
                            sha256_bytes(canonical_bytes),
                            canonical_bytes,
                            f"{source.repository_id}:{source.configuration}:{source.split}",
                            language,
                            _candidate_domain(source),
                            source.lane,
                            context_bucket,
                            len(input_ids),
                            json.dumps(input_ids, separators=(",", ":")),
                            tokenizer_sha256,
                            snapshot.chat_template_sha256,
                            source_conversation_sha256,
                            source_response_sha256,
                        )
                        if wanted[1:] != expected:
                            raise ValueError(
                                "candidate does not match its physical Task 3 row or tokenization"
                            )
                        matched += 1
                        wanted = next(wanted_rows, None)
                        if wanted is None:
                            break
                _verify_source_fd_stable(file_descriptor, initial_stat, path, _file)
            finally:
                os.close(file_descriptor)
            if wanted is not None:
                raise ValueError("candidate is absent from its physical Task 3 row")
        expected_count = int(connection.execute("SELECT count(*) FROM wanted").fetchone()[0])
        if matched != expected_count:
            raise ValueError("candidate is absent from its physical Task 3 row")
        if _staged_tree_snapshot(sources_root) != initial_tree:
            raise ValueError("staged Task 3 source tree changed during authentication")
        if (
            load_tokenizer_snapshot(tokenizer_snapshot_receipt, tokenizer_snapshot_receipt_sha256)
            != snapshot
        ):
            raise ValueError("tokenizer snapshot changed during physical membership verification")
    finally:
        connection.close()
        database.unlink(missing_ok=True)


def iter_candidate_inventory_bytes(inventory: CandidateInventory):
    """Yield deterministic inventory bytes without materializing the inventory."""
    if not inventory.rows:
        raise ValueError("candidate inventory identity requires at least one row")
    first = inventory.rows[0]
    source_manifest_sha256 = (
        inventory.rows.source_manifest_sha256
        if isinstance(inventory.rows, DiskBackedCandidateRows)
        else first.source_manifest_sha256
    )
    tokenizer_sha256 = (
        inventory.rows.tokenizer_sha256
        if isinstance(inventory.rows, DiskBackedCandidateRows)
        else first.tokenizer_sha256
    )
    ordered_rows: Sequence[CandidatePrompt] = (
        inventory.rows
        if isinstance(inventory.rows, DiskBackedCandidateRows)
        else tuple(sorted(inventory.rows, key=lambda row: row.prompt_uuid))
    )
    yield from _identity_chunks(
        ordered_rows,
        source_manifest_sha256,
        tokenizer_sha256,
        inventory.capacity,
        inventory.quarantine_counts,
        inventory.baseline_exclusion,
        inventory.held_out_exclusion,
        inventory.ptv2_revision,
        inventory.ptv2_allowlist_sha256,
    )


def candidate_inventory_bytes(inventory: CandidateInventory) -> bytes:
    """Materialize deterministic bytes for small tests; production callers should iterate."""
    return b"".join(iter_candidate_inventory_bytes(inventory))


def candidate_inventory_sha256(inventory: CandidateInventory) -> str:
    """Recompute the complete streamed inventory identity without materializing it."""
    digest = hashlib.sha256()
    for chunk in iter_candidate_inventory_bytes(inventory):
        digest.update(chunk)
    return digest.hexdigest()


def _tokenize(tokenizer: Any, row: dict[str, Any]) -> tuple[list[int], list[int]]:
    messages = row.get("messages") or row.get("conversations")
    if not isinstance(messages, list) or not messages:
        raise ValueError("inventory row has no messages")
    encoded = tokenizer.apply_chat_template(
        messages,
        tools=row.get("tools") or None,
        tokenize=True,
        add_generation_prompt=False,
        return_dict=True,
        return_assistant_tokens_mask=True,
    )
    input_ids = encoded.get("input_ids")
    loss_mask = encoded.get("assistant_masks", encoded.get("assistant_mask"))
    if (
        not isinstance(input_ids, list)
        or not isinstance(loss_mask, list)
        or len(input_ids) != len(loss_mask)
        or any(value not in (0, 1) for value in loss_mask)
    ):
        raise ValueError("tokenizer did not return aligned IDs and assistant mask")
    return input_ids, loss_mask


def build_inventory_rows(
    sources: list[InventorySource],
    *,
    tokenizer: Any,
    tokenizer_sha256: str,
    training_seq_len: int,
) -> list[dict[str, Any]]:
    """Verify and pretokenize all source rows without trusting caller metadata."""
    if _SHA256.fullmatch(tokenizer_sha256) is None or training_seq_len < 1:
        raise ValueError("tokenizer digest and training sequence length must be pinned")
    inventory: list[dict[str, Any]] = []
    seen: set[str] = set()
    for spec in sources:
        manifest_sha256, files = _verified_files(spec.manifest_path)
        for source_file in files:
            source_file_sha256 = sha256_file(source_file)
            for index, raw_row in enumerate(_iter_rows(source_file)):
                row = dict(raw_row)
                prompt_id = str(
                    row.get("prompt_id")
                    or hashlib.sha256(
                        json.dumps(
                            row.get("messages") or row.get("conversations"),
                            ensure_ascii=False,
                            separators=(",", ":"),
                            sort_keys=True,
                        ).encode()
                    ).hexdigest()
                )
                if prompt_id in seen:
                    raise ValueError(f"duplicate prompt_id {prompt_id}")
                seen.add(prompt_id)
                has_tools = bool(row.get("tools")) or any(
                    message.get("role") == "tool" or message.get("tool_calls")
                    for message in row.get("messages", [])
                )
                if has_tools != (spec.tool_lane == "recorded-trace"):
                    raise ValueError(f"prompt {prompt_id} is routed to the wrong tool lane")
                input_ids, full_mask = _tokenize(tokenizer, row)
                full_token_count = len(input_ids)
                training_ids = input_ids[:training_seq_len]
                training_mask = full_mask[:training_seq_len]
                assistant_tokens = sum(training_mask)
                if assistant_tokens < 1:
                    continue
                inventory.append(
                    {
                        "prompt_id": prompt_id,
                        "pool": spec.pool,
                        "category": spec.category,
                        "context_bucket": _context_bucket(full_token_count),
                        "full_token_count": full_token_count,
                        "full_assistant_tokens": sum(full_mask),
                        "assistant_tokens": assistant_tokens,
                        "source_id": spec.source_id,
                        "source_revision": spec.source_revision,
                        "license": spec.license,
                        "source_manifest_sha256": manifest_sha256,
                        "source_file_path": str(source_file.relative_to(spec.manifest_path.parent)),
                        "source_file_sha256": source_file_sha256,
                        "source_row_index": index,
                        "response_source": spec.response_source,
                        "tool_lane": spec.tool_lane,
                        "tokenizer_sha256": tokenizer_sha256,
                        "input_ids": training_ids,
                        "loss_mask": training_mask,
                        "messages": row.get("messages") or row.get("conversations"),
                        "tools": row.get("tools"),
                    }
                )
    return inventory


def _write_jsonl_atomic(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(f".{path.name}.partial-{os.getpid()}")
    with partial.open("w", encoding="utf-8") as output:
        for row in rows:
            output.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        output.flush()
        os.fsync(output.fileno())
    os.replace(partial, path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sources-config", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--tokenizer-sha256", required=True)
    parser.add_argument("--training-seq-len", type=int, default=4096)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    from transformers import AutoTokenizer  # pyright: ignore[reportMissingImports]

    config = yaml.safe_load(args.sources_config.read_text(encoding="utf-8"))
    sources = [
        InventorySource(
            **(record | {"manifest_path": args.sources_config.parent / record["manifest_path"]})
        )
        for record in config["sources"]
    ]
    actual_tokenizer_sha256 = tokenizer_snapshot_sha256(args.tokenizer)
    if actual_tokenizer_sha256 != args.tokenizer_sha256:
        raise ValueError("tokenizer snapshot SHA-256 mismatch")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True)
    rows = build_inventory_rows(
        sources,
        tokenizer=tokenizer,
        tokenizer_sha256=args.tokenizer_sha256,
        training_seq_len=args.training_seq_len,
    )
    _write_jsonl_atomic(args.output, rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
