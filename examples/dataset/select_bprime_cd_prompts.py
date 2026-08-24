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

"""Select exact, deterministic B-prime/C/D prompt views and their reserves."""

from __future__ import annotations

import json
import os
import re
import sqlite3
import tempfile
from collections import defaultdict
from collections.abc import Iterable, Iterator, Mapping, Sequence
from contextlib import suppress
from dataclasses import dataclass, replace
from enum import Enum
from hashlib import sha256
from pathlib import Path
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Literal, cast, overload

from build_specdec_inventory import (
    APPROVED_PTV2_ALLOWLIST_SHA256,
    APPROVED_PTV2_REVISION,
    CandidateInventory,
    CandidatePrompt,
    candidate_inventory_sha256,
    is_approved_ptv2_source,
    verify_candidate_inventory_membership,
)
from specdec_corpus_contracts import canonical_json, sha256_bytes
from stage_ptv23_sources import (
    SourceInventory,
    _fsync_directory,
    _rename_no_replace,
    _write_bytes_durable,
    load_source_inventory,
)

if TYPE_CHECKING:
    from bprime_cd_policy import PromptPolicy

__all__ = [
    "BPrimePromptViewBundle",
    "DiskBackedSelectedRows",
    "PromptPublicationArtifactIdentity",
    "PromptPublicationDurabilityError",
    "PromptPublicationInodeSnapshot",
    "PromptPublicationPhase",
    "PromptPublicationRecoveryState",
    "PromptSelectionBlocked",
    "PromptSelectionBlockedError",
    "PromptView",
    "PromptViewBundle",
    "PublishedPromptViews",
    "SelectedPrompt",
    "SelectionBlockedError",
    "build_policy_count_proofs",
    "prompt_publication_recovery_state",
    "publish_bprime_prompt_view_bundle",
    "publish_prompt_view_bundle",
    "select_bprime_prompt_view",
    "select_prompt_views",
]


_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_BUCKET_ORDER = ("le4k", "4k_16k", "16k_32k")
_BPRIME_CELLS = {
    "stem": ("stem-science", "target-synth", "en"),
    "japanese": ("multilingual", "target-synth", "ja"),
    "spanish": ("multilingual", "target-synth", "es"),
    "french": ("multilingual", "target-synth", "fr"),
    "italian": ("multilingual", "target-synth", "it"),
}
_NON_AGENTIC_CELLS = (
    "math",
    "code",
    "stem-science",
    "multilingual",
    "instruction-chat",
)


@dataclass(frozen=True)
class SelectedPrompt:
    """One immutable primary or reserve selection with inspectable provenance."""

    prompt_uuid: str
    arm: str
    domain: str
    lane: str
    language: str
    context_bucket: str
    source_id: str
    source_family: Literal["ptv2", "ptv3"]
    source_repository_id: str
    source_configuration: str
    source_split: str
    source_revision: str
    source_file_sha256: str
    source_manifest_sha256: str
    source_file_path: str
    source_row_index: int
    candidate_rank: int
    candidate_rank_sha256: str
    selection_index: int
    status: Literal["primary", "reserve"]
    canonical_prompt_json: str
    source_conversation_sha256: str | None = None
    source_response_sha256: str | None = None
    tokenizer_sha256: str | None = None
    chat_template_sha256: str | None = None

    @property
    def canonical_prompt(self) -> Any:
        """Return the complete canonical prompt represented by this row."""
        return json.loads(self.canonical_prompt_json)


@dataclass(frozen=True)
class PromptView:
    """One arm's exact ordered prompt list, reserves, and count proofs."""

    arm: str
    primary_rows: Sequence[SelectedPrompt]
    reserve_rows: Sequence[SelectedPrompt]
    cell_counts: Mapping[str, int]
    lane_counts: Mapping[str, int]
    bucket_floors: Mapping[str, Mapping[str, int]]
    non_agentic_bucket_floors: Mapping[str, Mapping[str, int]]
    lane_bucket_floors: Mapping[str, Mapping[str, int]]
    count_proof_sha256: str
    tokenizer_sha256: str | None = None
    chat_template_sha256: str | None = None

    @property
    def primary_prompt_ids(self) -> Sequence[str]:
        return _MappedSequence(self.primary_rows, "prompt_uuid")

    @property
    def final_prompt_ids(self) -> Sequence[str]:
        """Alias for the final ordered primary UUIDs."""
        return self.primary_prompt_ids

    @property
    def final_ordered_uuids(self) -> Sequence[str]:
        return self.primary_prompt_ids

    @property
    def reserve_prompt_ids(self) -> Sequence[str]:
        return _MappedSequence(self.reserve_rows, "prompt_uuid")

    @property
    def reserve_ordered_uuids(self) -> Sequence[str]:
        return self.reserve_prompt_ids

    @property
    def non_agentic_prompt_ids(self) -> frozenset[str]:
        return frozenset(
            row.prompt_uuid for row in self.primary_rows if row.domain != "swe-agentic-tool"
        )

    @property
    def agentic_prompt_ids(self) -> frozenset[str]:
        return frozenset(
            row.prompt_uuid for row in self.primary_rows if row.domain == "swe-agentic-tool"
        )


@dataclass(frozen=True)
class PromptViewBundle:
    """The exact B-prime/C/D selections bound to all input identities."""

    B_prime: PromptView
    C: PromptView
    D: PromptView
    policy_sha256: str
    seed: int
    source_inventory_sha256: str
    baseline_receipt_sha256: str
    held_out_receipt_sha256: str
    ptv2_revision: str
    ptv2_allowlist_sha256: str
    reserve_numerator: int
    reserve_denominator: int
    paired_cd_sha256: str
    selection_sha256: str
    _storage: _SelectionStorage

    @property
    def selection_digest(self) -> str:
        """Return the selection identity digest."""
        return self.selection_sha256

    @property
    def paired_cd_digest(self) -> str:
        """Return the paired C/D proof digest."""
        return self.paired_cd_sha256

    def close(self) -> None:
        """Release the temporary disk-backed selection database."""
        self._storage.close()

    def __enter__(self) -> PromptViewBundle:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


@dataclass(frozen=True)
class BPrimePromptViewBundle:
    """A PTV2-only B-prime selection with no C/D publication surface."""

    B_prime: PromptView
    policy_sha256: str
    seed: int
    source_inventory_sha256: str
    baseline_receipt_sha256: str
    held_out_receipt_sha256: str
    ptv2_revision: str
    ptv2_allowlist_sha256: str
    reserve_numerator: int
    reserve_denominator: int
    paired_cd_sha256: str
    selection_sha256: str
    _storage: _SelectionStorage
    tokenizer_sha256: str
    chat_template_sha256: str

    @property
    def selection_digest(self) -> str:
        return self.selection_sha256

    def close(self) -> None:
        self._storage.close()

    def __enter__(self) -> BPrimePromptViewBundle:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


@dataclass(frozen=True)
class PublishedPromptViews:
    """Paths and trusted root identity of a sharded published selection."""

    manifest_path: Path
    index_path: Path
    root_sha256: str
    row_count: int


class PromptPublicationPhase(str, Enum):
    """The last publication operation attempted before recovery became required."""

    PARTIAL_SETUP = "partial_setup"
    SHARD_WRITE = "shard_write"
    INDEX_FINALIZE = "index_finalize"
    MANIFEST_WRITE = "manifest_write"
    STAGING_FSYNC = "staging_fsync"
    RENAME = "rename"
    PARENT_FSYNC = "parent_fsync"


@dataclass(frozen=True)
class PromptPublicationArtifactIdentity:
    """The inode created for an artifact and its authenticated root when available."""

    device: int
    inode: int
    root_sha256: str | None

    @property
    def receipt(self) -> dict[str, Any]:
        return {
            "device": self.device,
            "inode": self.inode,
            "root_sha256": self.root_sha256,
        }


@dataclass(frozen=True)
class PromptPublicationInodeSnapshot:
    """One no-follow, point-in-time pathname observation."""

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

    @property
    def receipt(self) -> dict[str, Any]:
        return {
            "path": str(self.path),
            "status": self.status,
            "device": self.device,
            "inode": self.inode,
            "mode": self.mode,
            "error": self.error,
        }


@dataclass(frozen=True)
class PromptPublicationRecoveryState:
    """Typed evidence retained after any failure following partial creation."""

    phase: PromptPublicationPhase
    partial_path: Path
    destination_path: Path
    expected_artifact_identity: PromptPublicationArtifactIdentity | None
    partial_observation: PromptPublicationInodeSnapshot
    destination_observation: PromptPublicationInodeSnapshot
    recovery_required: Literal[True] = True

    @property
    def state(self) -> Literal["publication_recovery_required"]:
        return "publication_recovery_required"

    @property
    def verification_instructions(self) -> str:
        expected = self.expected_artifact_identity
        if expected is None:
            expected_text = "No expected artifact inode was available."
        elif expected.root_sha256 is None:
            expected_text = (
                f"The created partial used device {expected.device}, inode {expected.inode}; "
                "its authenticated root was not complete."
            )
        else:
            expected_text = (
                f"The created artifact used device {expected.device}, inode {expected.inode}, "
                f"and root SHA-256 {expected.root_sha256}."
            )
        return (
            f"Preserve {self.partial_path} and {self.destination_path}. Independently acquire "
            f"exclusive control of {self.destination_path.parent}, then inspect both paths without "
            f"following symlinks. {expected_text} Treat the recorded inode observations as "
            "point-in-time evidence, authenticate any manifest before reconciliation, and do not "
            "delete either pathname from this recovery state. Publication retries fail while the "
            "destination exists."
        )

    @property
    def receipt(self) -> dict[str, Any]:
        expected = self.expected_artifact_identity
        return {
            "state": self.state,
            "recovery_required": self.recovery_required,
            "phase": self.phase.value,
            "partial_path": str(self.partial_path),
            "destination_path": str(self.destination_path),
            "expected_artifact_identity": None if expected is None else expected.receipt,
            "observations": {
                "partial": self.partial_observation.receipt,
                "destination": self.destination_observation.receipt,
            },
            "verification_instructions": self.verification_instructions,
        }


class PromptPublicationDurabilityError(RuntimeError):
    """A published path requires independent durability recovery."""

    def __init__(self, recovery_state: PromptPublicationRecoveryState) -> None:
        expected = recovery_state.expected_artifact_identity
        if expected is None or expected.root_sha256 is None:
            raise ValueError("durability recovery requires a complete expected artifact identity")
        self.recovery_state = recovery_state
        self.destination = recovery_state.destination_path
        self.state = "durability_unconfirmed_recovery_required"
        self.recovery_required = recovery_state.recovery_required
        self.expected_device = expected.device
        self.expected_inode = expected.inode
        self.expected_identity = (expected.device, expected.inode)
        self.expected_root_sha256 = expected.root_sha256
        self.observed_identity = recovery_state.destination_observation.identity
        self.verification_instructions = recovery_state.verification_instructions
        self.receipt = recovery_state.receipt
        super().__init__(
            f"publication durability is unconfirmed for {self.destination}; "
            "independent recovery is required"
        )


def prompt_publication_recovery_state(error: BaseException) -> PromptPublicationRecoveryState:
    """Return typed recovery evidence carried by a publication exception."""
    recovery_state = getattr(error, "recovery_state", None)
    if not isinstance(recovery_state, PromptPublicationRecoveryState):
        raise ValueError("exception does not carry prompt publication recovery state")
    return recovery_state


class _SelectionStorage:
    def __init__(self) -> None:
        self._temporary = tempfile.TemporaryDirectory(prefix="specdec-selection-")
        self.path = Path(self._temporary.name) / "selection.sqlite3"
        self.closed = False
        self._working_connection: sqlite3.Connection | None = None

    def connect(self) -> sqlite3.Connection:
        if self.closed:
            raise RuntimeError("selection storage is closed")
        return sqlite3.connect(self.path)

    def close(self) -> None:
        if not self.closed:
            if self._working_connection is not None:
                self._working_connection.close()
                self._working_connection = None
            self.closed = True
            self._temporary.cleanup()

    def open_working_connection(self) -> sqlite3.Connection:
        """Open the single build connection owned by this storage lifetime."""
        self._working_connection = self.connect()
        return self._working_connection

    def release_working_connection(self) -> None:
        if self._working_connection is not None:
            self._working_connection.close()
            self._working_connection = None


class DiskBackedSelectedRows(Sequence[SelectedPrompt]):
    """A zero-resident sequence backed by the externally ordered SQLite result."""

    resident_row_count = 0

    def __init__(self, storage: _SelectionStorage, arm: str, status: str) -> None:
        self._storage = storage
        self._arm = arm
        self._status = status

    def __len__(self) -> int:
        with self._storage.connect() as connection:
            return int(
                connection.execute(
                    "SELECT count(*) FROM selected WHERE arm=? AND status=?",
                    (self._arm, self._status),
                ).fetchone()[0]
            )

    @overload
    def __getitem__(self, index: int) -> SelectedPrompt: ...

    @overload
    def __getitem__(self, index: slice) -> list[SelectedPrompt]: ...

    def __getitem__(self, index: int | slice) -> SelectedPrompt | list[SelectedPrompt]:
        if isinstance(index, slice):
            start, stop, step = index.indices(len(self))
            return [self[position] for position in range(start, stop, step)]
        if index < 0:
            index += len(self)
        if index < 0:
            raise IndexError(index)
        with self._storage.connect() as connection:
            row = connection.execute(
                "SELECT payload,arm,status,candidate_rank,rank,selection_index FROM selected "
                "WHERE arm=? AND status=? AND selection_index=?",
                (self._arm, self._status, index),
            ).fetchone()
        if row is None:
            raise IndexError(index)
        return _selected_from_db_row(row)

    def __iter__(self) -> Iterator[SelectedPrompt]:
        with self._storage.connect() as connection:
            cursor = connection.execute(
                "SELECT payload,arm,status,candidate_rank,rank,selection_index FROM selected "
                "WHERE arm=? AND status=? ORDER BY selection_index",
                (self._arm, self._status),
            )
            for row in cursor:
                yield _selected_from_db_row(row)


class _MappedSequence(Sequence[str]):
    def __init__(self, rows: Sequence[SelectedPrompt], field: str) -> None:
        self._rows = rows
        self._field = field

    def __len__(self) -> int:
        return len(self._rows)

    def __getitem__(self, index: int | slice) -> str | list[str]:
        value = self._rows[index]
        if isinstance(value, list):
            return [str(getattr(row, self._field)) for row in value]
        return str(getattr(value, self._field))

    def __iter__(self) -> Iterator[str]:
        return (str(getattr(row, self._field)) for row in self._rows)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Sequence):
            return False
        return len(self) == len(other) and all(left == right for left, right in zip(self, other))


class PromptSelectionBlockedError(ValueError):
    """Fatal exact-count shortfall carrying a canonical blocker receipt."""

    def __init__(self, receipt: Mapping[str, Any]) -> None:
        self.receipt = MappingProxyType(dict(receipt))
        super().__init__(
            "insufficient candidates for "
            f"{receipt['arm']}/{receipt['cell']}/{receipt['lane']}/{receipt['context_bucket']}: "
            f"need {receipt['required_candidate_count']}, have {receipt['available_candidate_count']}"
        )


PromptSelectionBlocked = PromptSelectionBlockedError
SelectionBlockedError = PromptSelectionBlockedError


def build_policy_count_proofs(policy: PromptPolicy) -> dict[str, Any]:
    """Build canonical large-count arithmetic proofs without allocating prompt rows."""
    proofs: dict[str, Any] = {}
    for arm_name, arm in policy.arms.items():
        cell_counts = {name: cell.prompt_count for name, cell in arm.cells.items()}
        lane_counts = dict(arm.lanes)
        cell_acquisition_lower_bounds = {
            name: _acquisition_count(count, policy) for name, count in cell_counts.items()
        }
        proof: dict[str, Any] = {
            "primary_count": arm.prompt_count,
            "acquisition_lower_bound_count": sum(cell_acquisition_lower_bounds.values()),
            "reserve_lower_bound_count": sum(cell_acquisition_lower_bounds.values())
            - arm.prompt_count,
            "cell_counts": cell_counts,
            "cell_acquisition_lower_bound_counts": cell_acquisition_lower_bounds,
            "lane_counts": lane_counts,
            "lane_acquisition_counts": {
                name: _acquisition_count(count, policy) for name, count in lane_counts.items()
            },
        }
        if arm_name in {"C", "D"}:
            proof["non_agentic_count"] = sum(cell_counts[name] for name in _NON_AGENTIC_CELLS)
        proofs[arm_name] = proof
    paired_payload = {
        "C": {name: proofs["C"]["cell_counts"][name] for name in _NON_AGENTIC_CELLS},
        "D": {name: proofs["D"]["cell_counts"][name] for name in _NON_AGENTIC_CELLS},
        "reserve": {
            name: proofs["C"]["cell_acquisition_lower_bound_counts"][name]
            - proofs["C"]["cell_counts"][name]
            for name in _NON_AGENTIC_CELLS
        },
    }
    proofs["paired_cd_sha256"] = sha256_bytes(canonical_json(paired_payload))
    return proofs


def select_prompt_views(
    inventory: CandidateInventory,
    policy: PromptPolicy,
    *,
    baseline_receipt_sha256: str,
    held_out_receipt_sha256: str,
) -> PromptViewBundle:
    """Select exact views and deterministically release storage on every failed exit."""
    storage = _SelectionStorage()
    try:
        return _select_prompt_views_impl(
            inventory,
            policy,
            baseline_receipt_sha256=baseline_receipt_sha256,
            held_out_receipt_sha256=held_out_receipt_sha256,
            storage=storage,
        )
    except BaseException:
        storage.close()
        raise


def select_bprime_prompt_view(
    inventory: CandidateInventory,
    policy: PromptPolicy,
    *,
    source_inventory: SourceInventory,
    tokenizer_snapshot_receipt: Path,
    tokenizer_snapshot_receipt_sha256: str,
    baseline_receipt_sha256: str,
    held_out_receipt_sha256: str,
) -> BPrimePromptViewBundle:
    """Select the authenticated PTV2-only B-prime complement without C/D."""
    storage = _SelectionStorage()
    try:
        _validate_digest(inventory.inventory_sha256, "source inventory")
        if candidate_inventory_sha256(inventory) != inventory.inventory_sha256:
            raise ValueError("source inventory streamed identity mismatch")
        if (
            inventory.ptv2_revision != APPROVED_PTV2_REVISION
            or inventory.ptv2_allowlist_sha256 != APPROVED_PTV2_ALLOWLIST_SHA256
        ):
            raise ValueError("source inventory PTV2 allowlist identity mismatch")
        _validate_digest(policy.policy_sha256, "policy")
        _validate_exclusion_receipts(inventory, baseline_receipt_sha256, held_out_receipt_sha256)
        authenticated_source_inventory = _validate_bprime_source_inventory(source_inventory)
        _validate_bprime_only_rows(inventory.rows, authenticated_source_inventory)
        verify_candidate_inventory_membership(
            inventory,
            authenticated_source_inventory,
            tokenizer_snapshot_receipt=tokenizer_snapshot_receipt,
            tokenizer_snapshot_receipt_sha256=tokenizer_snapshot_receipt_sha256,
        )
        first = inventory.rows[0]
        if not isinstance(first, CandidatePrompt) or first.chat_template_sha256 is None:
            raise ValueError("B-prime candidates lack authenticated tokenizer/template identity")
        tokenizer_sha256 = first.tokenizer_sha256
        chat_template_sha256 = first.chat_template_sha256
        connection = storage.open_working_connection()
        _create_selection_schema(connection)
        _spool_candidates(connection, inventory.rows, policy)
        floors_by_cell: dict[str, dict[str, int]] = {}
        for cell, (domain, lane, language) in _BPRIME_CELLS.items():
            floors_by_cell[cell] = _select_sql_cell(
                connection,
                quota=policy.count_for("B-prime", cell),
                arm="B-prime",
                cell=cell,
                domain=domain,
                lane=lane,
                language=language,
                source_family="ptv2",
                source_revision=inventory.ptv2_revision,
                approved_ptv2=True,
                policy=policy,
                inventory_sha256=inventory.inventory_sha256,
                baseline_receipt_sha256=baseline_receipt_sha256,
                held_out_receipt_sha256=held_out_receipt_sha256,
            )
        _assign_selection_indexes(connection)
        connection.commit()
        view = replace(
            _build_disk_view(storage, "B-prime", floors_by_cell, policy, (), {}),
            tokenizer_sha256=tokenizer_sha256,
            chat_template_sha256=chat_template_sha256,
        )
        paired_cd_sha256 = "0" * 64
        metadata = {
            "schema_version": 2,
            "selection_mode": "B-prime-only",
            "policy_sha256": policy.policy_sha256,
            "seed": policy.seed,
            "source_inventory_sha256": inventory.inventory_sha256,
            "baseline_receipt_sha256": baseline_receipt_sha256,
            "held_out_receipt_sha256": held_out_receipt_sha256,
            "ptv2_revision": inventory.ptv2_revision,
            "ptv2_allowlist_sha256": inventory.ptv2_allowlist_sha256,
            "paired_cd_sha256": paired_cd_sha256,
            "tokenizer_sha256": tokenizer_sha256,
            "chat_template_sha256": chat_template_sha256,
            "arms": {view.arm: _compact_view_record(view)},
        }
        selection_sha256 = _stream_selection_digest(connection, metadata)
        storage.release_working_connection()
        return BPrimePromptViewBundle(
            B_prime=view,
            policy_sha256=policy.policy_sha256,
            seed=policy.seed,
            source_inventory_sha256=inventory.inventory_sha256,
            baseline_receipt_sha256=baseline_receipt_sha256,
            held_out_receipt_sha256=held_out_receipt_sha256,
            ptv2_revision=inventory.ptv2_revision,
            ptv2_allowlist_sha256=inventory.ptv2_allowlist_sha256,
            reserve_numerator=policy.reserve_numerator,
            reserve_denominator=policy.reserve_denominator,
            paired_cd_sha256=paired_cd_sha256,
            selection_sha256=selection_sha256,
            _storage=storage,
            tokenizer_sha256=tokenizer_sha256,
            chat_template_sha256=chat_template_sha256,
        )
    except BaseException:
        storage.close()
        raise


def _validate_bprime_source_inventory(source_inventory: SourceInventory) -> SourceInventory:
    if not isinstance(source_inventory, SourceInventory) or source_inventory.staged_root is None:
        raise ValueError("B-prime-only selection requires a staged Task 3 source inventory")
    receipt = source_inventory.staged_root / "SOURCE_INVENTORY.json"
    try:
        authenticated = load_source_inventory(receipt)
    except (OSError, ValueError) as error:
        raise ValueError("B-prime-only staged Task 3 source inventory is invalid") from error
    if authenticated != source_inventory:
        raise ValueError("B-prime-only Task 3 source inventory identity mismatch")
    expected_splits = {
        "chat",
        "code",
        "math",
        "stem",
        "multilingual_ja",
        "multilingual_it",
        "multilingual_de",
        "multilingual_es",
        "multilingual_fr",
    }
    if (
        len(authenticated.sources) != len(expected_splits)
        or {source.split for source in authenticated.sources} != expected_splits
        or sum(len(source.files) for source in authenticated.sources) != 201
    ):
        raise ValueError("B-prime-only selection requires exactly 201 PTV2 source shards")
    if any(
        source.lane != "target-synth"
        or source.cell != source.split
        or not is_approved_ptv2_source(
            source.repository_id, source.configuration, source.split, source.revision
        )
        for source in authenticated.sources
    ):
        raise ValueError("B-prime-only selection requires a PTV2-only approved inventory")
    return authenticated


def _validate_bprime_only_rows(rows: Iterable[Any], source_inventory: SourceInventory) -> None:
    canonical = {
        "chat": ("instruction-chat", "en"),
        "code": ("code", "en"),
        "math": ("math", "en"),
        "stem": ("stem-science", "en"),
        "multilingual_ja": ("multilingual", "ja"),
        "multilingual_it": ("multilingual", "it"),
        "multilingual_de": ("multilingual", "de"),
        "multilingual_es": ("multilingual", "es"),
        "multilingual_fr": ("multilingual", "fr"),
    }
    source_files = {
        (
            source.repository_id,
            source.configuration,
            source.split,
            source.revision,
            file.path,
            file.sha256,
        )
        for source in source_inventory.sources
        for file in source.files
    }
    for row in rows:
        if not isinstance(row, CandidatePrompt):
            raise TypeError("candidate inventory rows must be CandidatePrompt values")
        if row.source_family != "ptv2" or not is_approved_ptv2_source(
            row.source_repository_id,
            row.source_configuration,
            row.source_split,
            row.source_revision,
        ):
            raise ValueError("B-prime-only selection requires a PTV2-only approved inventory")
        expected = canonical.get(row.source_split)
        if expected is None or row.domain != expected[0] or row.lane != "target-synth":
            raise ValueError("PTV2 candidate does not use the canonical cell and lane")
        if row.language != expected[1]:
            raise ValueError("PTV2 candidate language does not match its source split")
        _validate_digest(row.source_conversation_sha256 or "", "source conversation")
        _validate_digest(row.source_response_sha256 or "", "source response")
        if (
            row.source_manifest_sha256 != source_inventory.manifest_sha256
            or (
                row.source_repository_id,
                row.source_configuration,
                row.source_split,
                row.source_revision,
                row.source_file_path,
                row.source_file_sha256,
            )
            not in source_files
        ):
            raise ValueError("PTV2 candidate source digest does not match Task 3 inventory")


def _select_prompt_views_impl(
    inventory: CandidateInventory,
    policy: PromptPolicy,
    *,
    baseline_receipt_sha256: str,
    held_out_receipt_sha256: str,
    storage: _SelectionStorage,
) -> PromptViewBundle:
    """Select exact views using SQLite for external ordering and bounded resident memory."""
    _validate_digest(inventory.inventory_sha256, "source inventory")
    if candidate_inventory_sha256(inventory) != inventory.inventory_sha256:
        raise ValueError("source inventory streamed identity mismatch")
    if (
        inventory.ptv2_revision != APPROVED_PTV2_REVISION
        or inventory.ptv2_allowlist_sha256 != APPROVED_PTV2_ALLOWLIST_SHA256
    ):
        raise ValueError("source inventory PTV2 allowlist identity mismatch")
    _validate_digest(policy.policy_sha256, "policy")
    _validate_exclusion_receipts(inventory, baseline_receipt_sha256, held_out_receipt_sha256)
    connection = storage.open_working_connection()
    _create_selection_schema(connection)
    try:
        _spool_candidates(connection, inventory.rows, policy)
    except BaseException:
        storage.release_working_connection()
        raise

    b_floors: dict[str, dict[str, int]] = {}
    for cell, (domain, lane, language) in _BPRIME_CELLS.items():
        floors = _select_sql_cell(
            connection,
            quota=policy.count_for("B-prime", cell),
            arm="B-prime",
            cell=cell,
            domain=domain,
            lane=lane,
            language=language,
            source_family="ptv2",
            source_revision=inventory.ptv2_revision,
            approved_ptv2=True,
            policy=policy,
            inventory_sha256=inventory.inventory_sha256,
            baseline_receipt_sha256=baseline_receipt_sha256,
            held_out_receipt_sha256=held_out_receipt_sha256,
        )
        b_floors[cell] = floors

    c_floors: dict[str, dict[str, int]] = {}
    d_floors: dict[str, dict[str, int]] = {}
    for cell in _NON_AGENTIC_CELLS:
        floors = _select_sql_cell(
            connection,
            quota=policy.count_for("C", cell),
            arm="C",
            cell=cell,
            domain=cell,
            lane="target-synth",
            policy=policy,
            inventory_sha256=inventory.inventory_sha256,
            baseline_receipt_sha256=baseline_receipt_sha256,
            held_out_receipt_sha256=held_out_receipt_sha256,
        )
        _clone_selected_cell(connection, source_arm="C", target_arm="D", domain=cell)
        c_floors[cell] = floors
        d_floors[cell] = dict(floors)

    floors = _select_sql_cell(
        connection,
        quota=policy.count_for("C", "swe-agentic-tool"),
        arm="C",
        cell="swe-agentic-tool",
        domain="swe-agentic-tool",
        lane="agentless-swe",
        policy=policy,
        inventory_sha256=inventory.inventory_sha256,
        baseline_receipt_sha256=baseline_receipt_sha256,
        held_out_receipt_sha256=held_out_receipt_sha256,
    )
    c_floors["swe-agentic-tool"] = floors

    d_agentic_floors: dict[str, int] = defaultdict(int)
    d_lane_floors: dict[str, dict[str, int]] = {}
    for lane, quota in policy.arms["D"].lanes.items():
        floors = _select_sql_cell(
            connection,
            quota=quota,
            arm="D",
            cell="swe-agentic-tool",
            domain="swe-agentic-tool",
            lane=lane,
            policy=policy,
            inventory_sha256=inventory.inventory_sha256,
            baseline_receipt_sha256=baseline_receipt_sha256,
            held_out_receipt_sha256=held_out_receipt_sha256,
        )
        for bucket, count in floors.items():
            d_agentic_floors[bucket] += count
        d_lane_floors[lane] = floors
    d_floors["swe-agentic-tool"] = dict(d_agentic_floors)
    _assign_selection_indexes(connection)
    connection.commit()
    b_view = _build_disk_view(storage, "B-prime", b_floors, policy, (), {})
    c_view = _build_disk_view(
        storage,
        "C",
        c_floors,
        policy,
        _NON_AGENTIC_CELLS,
        {"agentless-swe": c_floors["swe-agentic-tool"]},
    )
    d_view = _build_disk_view(storage, "D", d_floors, policy, _NON_AGENTIC_CELLS, d_lane_floors)
    _validate_paired_sql(connection)
    paired_cd_sha256 = _stream_paired_digest(connection, c_view, d_view)
    metadata = {
        "schema_version": 2,
        "policy_sha256": policy.policy_sha256,
        "seed": policy.seed,
        "source_inventory_sha256": inventory.inventory_sha256,
        "baseline_receipt_sha256": baseline_receipt_sha256,
        "held_out_receipt_sha256": held_out_receipt_sha256,
        "ptv2_revision": inventory.ptv2_revision,
        "ptv2_allowlist_sha256": inventory.ptv2_allowlist_sha256,
        "paired_cd_sha256": paired_cd_sha256,
        "arms": {v.arm: _compact_view_record(v) for v in (b_view, c_view, d_view)},
    }
    selection_sha256 = _stream_selection_digest(connection, metadata)
    storage.release_working_connection()
    return PromptViewBundle(
        B_prime=b_view,
        C=c_view,
        D=d_view,
        policy_sha256=policy.policy_sha256,
        seed=policy.seed,
        source_inventory_sha256=inventory.inventory_sha256,
        baseline_receipt_sha256=baseline_receipt_sha256,
        held_out_receipt_sha256=held_out_receipt_sha256,
        ptv2_revision=inventory.ptv2_revision,
        ptv2_allowlist_sha256=inventory.ptv2_allowlist_sha256,
        reserve_numerator=policy.reserve_numerator,
        reserve_denominator=policy.reserve_denominator,
        paired_cd_sha256=paired_cd_sha256,
        selection_sha256=selection_sha256,
        _storage=storage,
    )


def _create_selection_schema(connection: sqlite3.Connection) -> None:
    connection.executescript(
        """
        PRAGMA journal_mode=OFF;
        PRAGMA synchronous=OFF;
        PRAGMA temp_store=FILE;
        CREATE TABLE candidates(
          prompt_uuid TEXT PRIMARY KEY, domain TEXT NOT NULL, lane TEXT NOT NULL,
          language TEXT NOT NULL, bucket TEXT NOT NULL, source_family TEXT NOT NULL,
          source_revision TEXT NOT NULL, approved_ptv2 INTEGER NOT NULL,
          rank TEXT NOT NULL, payload TEXT NOT NULL
        );
        CREATE INDEX candidates_cell ON candidates(
          domain,lane,language,source_family,source_revision,approved_ptv2,bucket,rank,prompt_uuid
        );
        CREATE TABLE selected(
          arm TEXT NOT NULL, status TEXT NOT NULL, cell TEXT NOT NULL,
          prompt_uuid TEXT NOT NULL, domain TEXT NOT NULL, lane TEXT NOT NULL,
          language TEXT NOT NULL, bucket TEXT NOT NULL, candidate_rank INTEGER NOT NULL,
          rank TEXT NOT NULL, selection_index INTEGER NOT NULL DEFAULT -1, payload TEXT NOT NULL,
          PRIMARY KEY(arm,status,prompt_uuid)
        );
        CREATE INDEX selected_order ON selected(arm,status,selection_index);
        CREATE INDEX selected_filter ON selected(arm,domain,lane,language,prompt_uuid,status,selection_index);
        """
    )


def _candidate_payload(candidate: CandidatePrompt) -> str:
    payload: dict[str, Any] = {
        "prompt_uuid": candidate.prompt_uuid,
        "domain": candidate.domain,
        "lane": candidate.lane,
        "language": candidate.language,
        "context_bucket": candidate.context_bucket,
        "source_id": candidate.source_id,
        "source_family": candidate.source_family,
        "source_repository_id": candidate.source_repository_id,
        "source_configuration": candidate.source_configuration,
        "source_split": candidate.source_split,
        "source_revision": candidate.source_revision,
        "source_file_sha256": candidate.source_file_sha256,
        "source_manifest_sha256": candidate.source_manifest_sha256,
        "source_file_path": candidate.source_file_path,
        "source_row_index": candidate.source_row_index,
        "source_conversation_sha256": candidate.source_conversation_sha256,
        "source_response_sha256": candidate.source_response_sha256,
        "canonical_prompt_json": candidate.canonical_bytes.decode("utf-8"),
    }
    if candidate.chat_template_sha256 is not None:
        payload["tokenizer_sha256"] = candidate.tokenizer_sha256
        payload["chat_template_sha256"] = candidate.chat_template_sha256
    return json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _spool_candidates(
    connection: sqlite3.Connection, raw_rows: Iterable[Any], policy: PromptPolicy
) -> None:
    for row in raw_rows:
        if not isinstance(row, CandidatePrompt):
            raise TypeError("candidate inventory rows must be CandidatePrompt values")
        _validate_digest(row.prompt_uuid, "prompt UUID")
        if sha256(row.canonical_bytes).hexdigest() != row.prompt_uuid:
            raise ValueError(f"candidate canonical prompt identity mismatch: {row.prompt_uuid}")
        if not policy.is_language_allowed(row.language):
            continue
        try:
            connection.execute(
                "INSERT INTO candidates VALUES(?,?,?,?,?,?,?,?,?,?)",
                (
                    row.prompt_uuid,
                    row.domain,
                    row.lane,
                    row.language,
                    row.context_bucket,
                    row.source_family,
                    row.source_revision,
                    int(
                        is_approved_ptv2_source(
                            row.source_repository_id,
                            row.source_configuration,
                            row.source_split,
                            row.source_revision,
                        )
                    ),
                    _rank(row, policy),
                    _candidate_payload(row),
                ),
            )
        except sqlite3.IntegrityError as error:
            raise ValueError(f"duplicate candidate prompt UUID: {row.prompt_uuid}") from error
    connection.commit()


def _candidate_where(
    *,
    domain: str,
    lane: str,
    language: str | None,
    source_family: str | None,
    source_revision: str | None,
    approved_ptv2: bool | None,
) -> tuple[str, list[str]]:
    clauses = ["domain=?", "lane=?"]
    values = [domain, lane]
    for column, value in (
        ("language", language),
        ("source_family", source_family),
        ("source_revision", source_revision),
        ("approved_ptv2", None if approved_ptv2 is None else str(int(approved_ptv2))),
    ):
        if value is not None:
            clauses.append(f"{column}=?")
            values.append(value)
    return " AND ".join(clauses), values


def _select_sql_cell(
    connection: sqlite3.Connection,
    *,
    quota: int,
    arm: str,
    cell: str,
    domain: str,
    lane: str,
    policy: PromptPolicy,
    inventory_sha256: str,
    baseline_receipt_sha256: str,
    held_out_receipt_sha256: str,
    language: str | None = None,
    source_family: str | None = None,
    source_revision: str | None = None,
    approved_ptv2: bool | None = None,
) -> dict[str, int]:
    where, values = _candidate_where(
        domain=domain,
        lane=lane,
        language=language,
        source_family=source_family,
        source_revision=source_revision,
        approved_ptv2=approved_ptv2,
    )
    capacities = {
        str(bucket): int(count)
        for bucket, count in connection.execute(
            f"SELECT bucket,count(*) FROM candidates WHERE {where} GROUP BY bucket", values
        )
    }
    floors = _allocate_bucket_quotas(capacities, quota)
    if not floors:
        _raise_shortfall(
            arm=arm,
            cell=cell,
            lane=lane,
            bucket="none",
            quota=quota,
            required=_acquisition_count(quota, policy),
            available=sum(capacities.values()),
            policy=policy,
            inventory_sha256=inventory_sha256,
            baseline_receipt_sha256=baseline_receipt_sha256,
            held_out_receipt_sha256=held_out_receipt_sha256,
        )
    for bucket, bucket_quota in floors.items():
        required = _acquisition_count(bucket_quota, policy)
        rows = connection.execute(
            f"SELECT prompt_uuid,domain,lane,language,bucket,rank,payload FROM candidates "
            f"WHERE {where} AND bucket=? ORDER BY rank,prompt_uuid LIMIT ?",
            (*values, bucket, required),
        )
        selected = 0
        reserve_count = required - bucket_quota
        for candidate_rank, row in enumerate(rows):
            status = "reserve" if candidate_rank < reserve_count else "primary"
            connection.execute(
                "INSERT INTO selected(arm,status,cell,prompt_uuid,domain,lane,language,bucket,"
                "candidate_rank,rank,payload) VALUES(?,?,?,?,?,?,?,?,?,?,?)",
                (
                    arm,
                    status,
                    cell,
                    row[0],
                    row[1],
                    row[2],
                    row[3],
                    row[4],
                    candidate_rank,
                    row[5],
                    row[6],
                ),
            )
            selected += 1
        if selected != required:
            _raise_shortfall(
                arm=arm,
                cell=cell,
                lane=lane,
                bucket=bucket,
                quota=bucket_quota,
                required=required,
                available=selected,
                policy=policy,
                inventory_sha256=inventory_sha256,
                baseline_receipt_sha256=baseline_receipt_sha256,
                held_out_receipt_sha256=held_out_receipt_sha256,
            )
    return floors


def _clone_selected_cell(
    connection: sqlite3.Connection, *, source_arm: str, target_arm: str, domain: str
) -> None:
    connection.execute(
        "INSERT INTO selected(arm,status,cell,prompt_uuid,domain,lane,language,bucket,"
        "candidate_rank,rank,payload) SELECT ?,status,cell,prompt_uuid,domain,lane,language,bucket,"
        "candidate_rank,rank,payload FROM selected WHERE arm=? AND domain=?",
        (target_arm, source_arm, domain),
    )


def _assign_selection_indexes(connection: sqlite3.Connection) -> None:
    for arm in ("B-prime", "C", "D"):
        for status in ("primary", "reserve"):
            cursor = connection.execute(
                "SELECT rowid FROM selected WHERE arm=? AND status=? "
                "ORDER BY rank,prompt_uuid,domain,lane,bucket",
                (arm, status),
            )
            for index, (rowid,) in enumerate(cursor):
                connection.execute(
                    "UPDATE selected SET selection_index=? WHERE rowid=?", (index, rowid)
                )


def _selected_from_db_row(row: Sequence[Any]) -> SelectedPrompt:
    value = json.loads(row[0])
    status = str(row[2])
    if status not in {"primary", "reserve"}:
        raise ValueError(f"invalid stored selection status: {status}")
    return SelectedPrompt(
        **value,
        arm=str(row[1]),
        status=cast("Literal['primary', 'reserve']", status),
        candidate_rank=int(row[3]),
        candidate_rank_sha256=str(row[4]),
        selection_index=int(row[5]),
    )


def _build_disk_view(
    storage: _SelectionStorage,
    arm: str,
    bucket_floors: Mapping[str, Mapping[str, int]],
    policy: PromptPolicy,
    non_agentic_cells: Sequence[str],
    lane_bucket_floors: Mapping[str, Mapping[str, int]],
) -> PromptView:
    primary = DiskBackedSelectedRows(storage, arm, "primary")
    reserve = DiskBackedSelectedRows(storage, arm, "reserve")
    expected = policy.count_for(arm)
    if len(primary) != expected:
        raise AssertionError(f"{arm} exact-count proof failed: {len(primary)} != {expected}")
    cell_counts = {name: policy.count_for(arm, name) for name in policy.arms[arm].cells}
    lane_counts = (
        {"agentless-swe": policy.count_for("C", "swe-agentic-tool")}
        if arm == "C"
        else dict(policy.arms[arm].lanes)
    )
    floors = _freeze_floors(bucket_floors)
    non_agentic = _freeze_floors({name: bucket_floors[name] for name in non_agentic_cells})
    lane_floors = _freeze_floors(lane_bucket_floors)
    proof = {
        "arm": arm,
        "primary_count": len(primary),
        "reserve_count": len(reserve),
        "cell_counts": cell_counts,
        "lane_counts": lane_counts,
        "bucket_floors": _plain_floors(floors),
        "lane_bucket_floors": _plain_floors(lane_floors),
        "acquisition_count": len(primary) + len(reserve),
    }
    return PromptView(
        arm,
        primary,
        reserve,
        MappingProxyType(cell_counts),
        MappingProxyType(lane_counts),
        floors,
        non_agentic,
        lane_floors,
        sha256_bytes(canonical_json(proof)),
    )


def _compact_view_record(view: PromptView) -> dict[str, Any]:
    return {
        "primary_count": len(view.primary_rows),
        "reserve_count": len(view.reserve_rows),
        "cell_counts": dict(view.cell_counts),
        "lane_counts": dict(view.lane_counts),
        "bucket_floors": _plain_floors(view.bucket_floors),
        "non_agentic_bucket_floors": _plain_floors(view.non_agentic_bucket_floors),
        "lane_bucket_floors": _plain_floors(view.lane_bucket_floors),
        "count_proof_sha256": view.count_proof_sha256,
    }


def _validate_paired_sql(connection: sqlite3.Connection) -> None:
    mismatch = connection.execute(
        "SELECT count(*) FROM ("
        "SELECT prompt_uuid,status FROM selected WHERE arm='C' AND domain!='swe-agentic-tool' "
        "EXCEPT SELECT prompt_uuid,status FROM selected WHERE arm='D' AND domain!='swe-agentic-tool'"
        ")"
    ).fetchone()[0]
    if mismatch:
        raise AssertionError("C/D non-agentic prompt UUID sets are not paired")


def _stream_paired_digest(
    connection: sqlite3.Connection, c_view: PromptView, d_view: PromptView
) -> str:
    digest = sha256(
        canonical_json(
            {
                "bucket_floors": _plain_floors(c_view.non_agentic_bucket_floors),
                "C_count_proof_sha256": c_view.count_proof_sha256,
                "D_count_proof_sha256": d_view.count_proof_sha256,
            }
        )
    )
    for row in connection.execute(
        "SELECT status,prompt_uuid FROM selected WHERE arm='C' AND domain!='swe-agentic-tool' "
        "ORDER BY status,prompt_uuid"
    ):
        digest.update(canonical_json(list(row)))
        digest.update(b"\n")
    return digest.hexdigest()


def _stream_selection_digest(connection: sqlite3.Connection, metadata: Mapping[str, Any]) -> str:
    digest = sha256(canonical_json(metadata))
    for row in connection.execute(
        "SELECT arm,status,selection_index,prompt_uuid FROM selected "
        "ORDER BY arm,status,selection_index"
    ):
        digest.update(canonical_json(list(row)))
        digest.update(b"\n")
    return digest.hexdigest()


def _observe_publication_inode(path: Path) -> PromptPublicationInodeSnapshot:
    try:
        observation = os.lstat(path)
    except FileNotFoundError:
        return PromptPublicationInodeSnapshot(path, "absent", None, None, None, None)
    except OSError as error:
        return PromptPublicationInodeSnapshot(
            path,
            "unavailable",
            None,
            None,
            None,
            f"{type(error).__name__}: {error}",
        )
    return PromptPublicationInodeSnapshot(
        path,
        "present",
        observation.st_dev,
        observation.st_ino,
        observation.st_mode,
        None,
    )


def publish_prompt_view_bundle(
    bundle: PromptViewBundle, output_dir: Path, *, rows_per_shard: int = 10_000
) -> PublishedPromptViews:
    """Publish the full B-prime/C/D selection bundle."""
    return _publish_prompt_view_bundle(
        bundle,
        output_dir,
        rows_per_shard=rows_per_shard,
        views=(bundle.B_prime, bundle.C, bundle.D),
        selection_mode=None,
    )


def publish_bprime_prompt_view_bundle(
    bundle: BPrimePromptViewBundle, output_dir: Path, *, rows_per_shard: int = 10_000
) -> PublishedPromptViews:
    """Publish an authenticated PTV2-only B-prime complement."""
    return _publish_prompt_view_bundle(
        bundle,
        output_dir,
        rows_per_shard=rows_per_shard,
        views=(bundle.B_prime,),
        selection_mode="B-prime-only",
    )


def _publish_prompt_view_bundle(
    bundle: PromptViewBundle | BPrimePromptViewBundle,
    output_dir: Path,
    *,
    rows_per_shard: int,
    views: tuple[PromptView, ...],
    selection_mode: str | None,
) -> PublishedPromptViews:
    """Stream canonical rows to hashed JSONL shards plus a compact indexed root manifest."""
    if (
        isinstance(rows_per_shard, bool)
        or not isinstance(rows_per_shard, int)
        or rows_per_shard < 1
    ):
        raise ValueError("rows_per_shard must be a positive integer")
    output_dir = Path(output_dir)
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    partial: Path | None = None
    index: sqlite3.Connection | None = None
    shards: list[dict[str, Any]] = []
    row_count = 0
    shard_number = 0
    shard_file: Any = None
    shard_hasher = sha256()
    shard_count = 0
    shard_path = ""
    phase = PromptPublicationPhase.PARTIAL_SETUP
    expected_artifact_identity: PromptPublicationArtifactIdentity | None = None
    root_sha256: str | None = None

    def finish_shard() -> None:
        nonlocal shard_file, shard_hasher, shard_count
        if shard_file is None:
            return
        shard_file.flush()
        os.fsync(shard_file.fileno())
        byte_count = shard_file.tell()
        shard_file.close()
        shards.append(
            {
                "path": shard_path,
                "row_count": shard_count,
                "byte_count": byte_count,
                "sha256": shard_hasher.hexdigest(),
            }
        )
        shard_file = None

    try:
        partial = Path(
            tempfile.mkdtemp(prefix=f".{output_dir.name}.partial-", dir=output_dir.parent)
        )
        created_identity = os.lstat(partial)
        expected_artifact_identity = PromptPublicationArtifactIdentity(
            created_identity.st_dev,
            created_identity.st_ino,
            None,
        )
        shards_dir = partial / "shards"
        index_path = partial / "selection-index.sqlite3"
        shards_dir.mkdir()
        index = sqlite3.connect(index_path)
        index.executescript(
            """
            PRAGMA journal_mode=OFF;
            PRAGMA synchronous=FULL;
            CREATE TABLE rows(
              global_index INTEGER PRIMARY KEY, arm TEXT NOT NULL, status TEXT NOT NULL,
              selection_index INTEGER NOT NULL, domain TEXT NOT NULL, lane TEXT NOT NULL,
              language TEXT NOT NULL, prompt_uuid TEXT NOT NULL, context_bucket TEXT NOT NULL,
              source_family TEXT NOT NULL, source_repository_id TEXT NOT NULL,
              source_configuration TEXT NOT NULL, source_split TEXT NOT NULL,
              source_id TEXT NOT NULL, source_revision TEXT NOT NULL,
              source_file_sha256 TEXT NOT NULL, source_manifest_sha256 TEXT NOT NULL,
              source_file_path TEXT NOT NULL, source_row_index INTEGER NOT NULL,
              candidate_rank INTEGER NOT NULL, candidate_rank_sha256 TEXT NOT NULL,
              shard_path TEXT NOT NULL, byte_offset INTEGER NOT NULL,
              byte_length INTEGER NOT NULL, row_sha256 TEXT NOT NULL
            );
            CREATE INDEX rows_filter ON rows(
              arm,domain,lane,language,prompt_uuid,status,selection_index
            );
            """
        )
        phase = PromptPublicationPhase.SHARD_WRITE
        for view in views:
            for rows in (view.primary_rows, view.reserve_rows):
                for selected in rows:
                    if shard_file is None or shard_count == rows_per_shard:
                        finish_shard()
                        shard_number += 1
                        shard_count = 0
                        shard_hasher = sha256()
                        shard_path = f"shards/rows-{shard_number:06d}.jsonl"
                        shard_file = (partial / shard_path).open("wb")
                    _validate_selected_rank(selected, bundle)
                    record = _selected_manifest_record(selected)
                    line = canonical_json(record) + b"\n"
                    offset = shard_file.tell()
                    shard_file.write(line)
                    shard_hasher.update(line)
                    index_values = (
                        row_count,
                        selected.arm,
                        selected.status,
                        selected.selection_index,
                        selected.domain,
                        selected.lane,
                        selected.language,
                        selected.prompt_uuid,
                        selected.context_bucket,
                        selected.source_family,
                        selected.source_repository_id,
                        selected.source_configuration,
                        selected.source_split,
                        selected.source_id,
                        selected.source_revision,
                        selected.source_file_sha256,
                        selected.source_manifest_sha256,
                        selected.source_file_path,
                        selected.source_row_index,
                        selected.candidate_rank,
                        selected.candidate_rank_sha256,
                        shard_path,
                        offset,
                        len(line),
                        sha256(line).hexdigest(),
                    )
                    index.execute(
                        f"INSERT INTO rows VALUES({','.join('?' for _ in index_values)})",
                        index_values,
                    )
                    row_count += 1
                    shard_count += 1
        phase = PromptPublicationPhase.INDEX_FINALIZE
        finish_shard()
        index.commit()
        index.execute("VACUUM")
        index.close()
        index = None
        with index_path.open("rb") as index_stream:
            os.fsync(index_stream.fileno())
        index_sha256 = _file_sha256(index_path)
        phase = PromptPublicationPhase.MANIFEST_WRITE
        manifest: dict[str, Any] = {
            "schema_version": 2,
            "selection_sha256": bundle.selection_sha256,
            "paired_cd_sha256": bundle.paired_cd_sha256,
            "identity": {
                "policy_sha256": bundle.policy_sha256,
                "seed": bundle.seed,
                "source_inventory_sha256": bundle.source_inventory_sha256,
                "baseline_receipt_sha256": bundle.baseline_receipt_sha256,
                "held_out_receipt_sha256": bundle.held_out_receipt_sha256,
                "ptv2_revision": bundle.ptv2_revision,
                "ptv2_allowlist_sha256": bundle.ptv2_allowlist_sha256,
                "reserve_numerator": bundle.reserve_numerator,
                "reserve_denominator": bundle.reserve_denominator,
            },
            "arms": {view.arm: _compact_view_record(view) for view in views},
            "row_count": row_count,
            "rows_per_shard": rows_per_shard,
            "shards": shards,
            "index": {"path": "selection-index.sqlite3", "sha256": index_sha256},
        }
        if isinstance(bundle, BPrimePromptViewBundle):
            manifest["identity"]["tokenizer_sha256"] = bundle.tokenizer_sha256
            manifest["identity"]["chat_template_sha256"] = bundle.chat_template_sha256
        if selection_mode is not None:
            manifest["selection_mode"] = selection_mode
        root_sha256 = sha256_bytes(canonical_json(manifest))
        manifest["root_sha256"] = root_sha256
        expected_artifact_identity = PromptPublicationArtifactIdentity(
            expected_artifact_identity.device,
            expected_artifact_identity.inode,
            root_sha256,
        )
        manifest_path = partial / "SELECTION_MANIFEST.json"
        _write_bytes_durable(manifest_path, canonical_json(manifest) + b"\n")
        phase = PromptPublicationPhase.STAGING_FSYNC
        _fsync_directory(shards_dir)
        _fsync_directory(partial)
        phase = PromptPublicationPhase.RENAME
        current_partial = _observe_publication_inode(partial)
        if current_partial.identity != (
            expected_artifact_identity.device,
            expected_artifact_identity.inode,
        ):
            raise RuntimeError("publication partial inode changed before rename")
        _rename_no_replace(partial, output_dir)
        installed_artifact = _observe_publication_inode(output_dir)
        if installed_artifact.identity != (
            expected_artifact_identity.device,
            expected_artifact_identity.inode,
        ):
            raise RuntimeError("published destination inode does not match created artifact")
        phase = PromptPublicationPhase.PARENT_FSYNC
        _fsync_directory(output_dir.parent)
    except BaseException as error:
        if shard_file is not None and not shard_file.closed:
            with suppress(BaseException):
                shard_file.close()
        if index is not None:
            with suppress(BaseException):
                index.close()
        if partial is None:
            raise
        recovery_state = PromptPublicationRecoveryState(
            phase=phase,
            partial_path=partial,
            destination_path=output_dir,
            expected_artifact_identity=expected_artifact_identity,
            partial_observation=_observe_publication_inode(partial),
            destination_observation=_observe_publication_inode(output_dir),
        )
        if phase is PromptPublicationPhase.PARENT_FSYNC and isinstance(error, Exception):
            raise PromptPublicationDurabilityError(recovery_state) from error
        setattr(error, "recovery_state", recovery_state)
        raise
    if root_sha256 is None:
        raise AssertionError("successful publication lacks an authenticated root")
    return PublishedPromptViews(
        output_dir / "SELECTION_MANIFEST.json",
        output_dir / "selection-index.sqlite3",
        root_sha256,
        row_count,
    )


def _validate_selected_rank(
    row: SelectedPrompt, bundle: PromptViewBundle | BPrimePromptViewBundle
) -> None:
    fields = (
        bundle.policy_sha256,
        str(bundle.seed),
        row.domain,
        row.lane,
        row.language,
        row.context_bucket,
        row.source_id,
        row.source_revision,
        row.source_file_sha256,
        row.source_manifest_sha256,
        row.source_file_path,
        str(row.source_row_index),
        row.prompt_uuid,
    )
    if sha256("\0".join(fields).encode()).hexdigest() != row.candidate_rank_sha256:
        raise ValueError("selected row candidate ranking proof mismatch")


def _file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_digest(value: str, name: str) -> None:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ValueError(f"{name} digest must be an exact lowercase SHA-256")


def _validate_exclusion_receipts(
    inventory: CandidateInventory,
    baseline_receipt_sha256: str,
    held_out_receipt_sha256: str,
) -> None:
    for label, supplied, proof, quarantine_key in (
        (
            "baseline",
            baseline_receipt_sha256,
            inventory.baseline_exclusion,
            "historical_exclusion",
        ),
        (
            "held-out",
            held_out_receipt_sha256,
            inventory.held_out_exclusion,
            "heldout_exclusion",
        ),
    ):
        _validate_digest(supplied, f"{label} receipt")
        if supplied == "0" * 64:
            raise ValueError(f"{label} receipt digest must be nonzero")
        if supplied != proof.receipt_sha256:
            raise ValueError(f"{label} receipt does not match candidate inventory")
        _validate_digest(proof.prompt_ids_sha256, f"{label} exclusion prompt IDs")
        if proof.prompt_id_count < 0 or proof.excluded_candidate_count < 0:
            raise ValueError(f"{label} exclusion reconciliation has invalid counts")
        if proof.excluded_candidate_count != inventory.quarantine_counts.get(quarantine_key, 0):
            raise ValueError(f"{label} exclusion reconciliation does not match quarantine")


def _bucket_key(bucket: str) -> tuple[int, str]:
    try:
        return (_BUCKET_ORDER.index(bucket), bucket)
    except ValueError:
        return (len(_BUCKET_ORDER), bucket)


def _allocate_bucket_quotas(capacities: Mapping[str, int], quota: int) -> dict[str, int]:
    if quota <= 0:
        raise ValueError("cell quota must be positive")
    total = sum(capacities.values())
    if total < quota or not capacities:
        return {}
    floors = {bucket: quota * capacity // total for bucket, capacity in capacities.items()}
    remaining = quota - sum(floors.values())
    remainder_order = sorted(
        capacities,
        key=lambda bucket: (-(quota * capacities[bucket] % total), _bucket_key(bucket)),
    )
    for bucket in remainder_order[:remaining]:
        floors[bucket] += 1
    return {bucket: floors[bucket] for bucket in sorted(floors, key=_bucket_key) if floors[bucket]}


def _acquisition_count(quota: int, policy: PromptPolicy) -> int:
    return (
        quota * policy.reserve_numerator + policy.reserve_denominator - 1
    ) // policy.reserve_denominator


def _rank(candidate: CandidatePrompt, policy: PromptPolicy) -> str:
    fields = (
        policy.policy_sha256,
        str(policy.seed),
        candidate.domain,
        candidate.lane,
        candidate.language,
        candidate.context_bucket,
        candidate.source_id,
        candidate.source_revision,
        candidate.source_file_sha256,
        candidate.source_manifest_sha256,
        candidate.source_file_path,
        str(candidate.source_row_index),
        candidate.prompt_uuid,
    )
    return sha256("\0".join(fields).encode()).hexdigest()


def _raise_shortfall(
    *,
    arm: str,
    cell: str,
    lane: str,
    bucket: str,
    quota: int,
    required: int,
    available: int,
    policy: PromptPolicy,
    inventory_sha256: str,
    baseline_receipt_sha256: str,
    held_out_receipt_sha256: str,
) -> None:
    receipt: dict[str, Any] = {
        "schema_version": 1,
        "status": "blocked",
        "reason": "insufficient_candidates",
        "arm": arm,
        "cell": cell,
        "lane": lane,
        "context_bucket": bucket,
        "quota_count": quota,
        "required_candidate_count": required,
        "available_candidate_count": available,
        "deficit": required - available,
        "policy_sha256": policy.policy_sha256,
        "seed": policy.seed,
        "source_inventory_sha256": inventory_sha256,
        "baseline_receipt_sha256": baseline_receipt_sha256,
        "held_out_receipt_sha256": held_out_receipt_sha256,
    }
    receipt["blocker_sha256"] = sha256_bytes(canonical_json(receipt))
    raise PromptSelectionBlockedError(receipt)


def _freeze_floors(
    floors: Mapping[str, Mapping[str, int]],
) -> Mapping[str, Mapping[str, int]]:
    return MappingProxyType(
        {
            cell: MappingProxyType(
                {bucket: values[bucket] for bucket in sorted(values, key=_bucket_key)}
            )
            for cell, values in sorted(floors.items())
        }
    )


def _plain_floors(floors: Mapping[str, Mapping[str, int]]) -> dict[str, dict[str, int]]:
    return {cell: dict(values) for cell, values in floors.items()}


def _selected_manifest_record(row: SelectedPrompt) -> dict[str, Any]:
    record: dict[str, Any] = {
        "prompt_uuid": row.prompt_uuid,
        "arm": row.arm,
        "domain": row.domain,
        "lane": row.lane,
        "language": row.language,
        "context_bucket": row.context_bucket,
        "source_id": row.source_id,
        "source_family": row.source_family,
        "source_repository_id": row.source_repository_id,
        "source_configuration": row.source_configuration,
        "source_split": row.source_split,
        "source_revision": row.source_revision,
        "source_file_sha256": row.source_file_sha256,
        "source_manifest_sha256": row.source_manifest_sha256,
        "source_file_path": row.source_file_path,
        "source_row_index": row.source_row_index,
        "candidate_rank": row.candidate_rank,
        "candidate_rank_sha256": row.candidate_rank_sha256,
        "selection_index": row.selection_index,
        "status": row.status,
        "source_conversation_sha256": row.source_conversation_sha256,
        "source_response_sha256": row.source_response_sha256,
        "canonical_prompt": row.canonical_prompt,
    }
    if row.chat_template_sha256 is not None:
        record["tokenizer_sha256"] = row.tokenizer_sha256
        record["chat_template_sha256"] = row.chat_template_sha256
    return record
