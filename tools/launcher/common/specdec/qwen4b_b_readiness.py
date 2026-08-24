# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Independent B-balanced readiness validation for the Qwen3-4B canary."""

from __future__ import annotations

import hashlib
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

__all__ = [
    "BReadinessInputs",
    "BReadinessReceipt",
    "Task9BalancedView",
    "assess_b_readiness",
    "load_task9_balanced_view",
    "write_b_readiness_receipt",
]

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_COMMIT = re.compile(r"^[0-9a-f]{40}$")
_CELL_QUOTAS = {
    "math": 500_000,
    "code": 400_000,
    "stem": 500_000,
    "chat": 400_000,
    "multilingual": 200_000,
}
_LANGUAGE_QUOTAS = {language: 40_000 for language in ("de", "ja", "es", "fr", "it")}
_DATA_SUFFIXES = frozenset({".jsonl", ".parquet"})


@dataclass(frozen=True)
class Task9BalancedView:
    """Typed Task9 B-publication projection; this intentionally has no A fields."""

    strategy: str
    occurrence_count: int
    cell_occurrence_counts: dict[str, int]
    multilingual_occurrence_counts: dict[str, int]
    declared_shards: tuple[str, ...]
    shard_root: Path
    trainer_epochs: int
    global_batch_size: int
    segment_occurrences: tuple[int, int]
    segment_steps: tuple[int, int]
    cumulative_steps: tuple[int, int]
    segment_final_valid_occurrences: tuple[int, int]
    source_native_responses: bool
    publication_sha256: str
    destination_sha256: str
    tokenizer_sha256: str
    chat_template_sha256: str
    assistant_loss_mask_sha256: str


@dataclass(frozen=True)
class BReadinessInputs:
    """The B-only inputs that must be bound before a bounded canary can run."""

    task9_b_view: Task9BalancedView
    source_commit: str


@dataclass(frozen=True)
class BReadinessReceipt:
    """Immutable result of B-only readiness validation."""

    ready: bool
    blocker_codes: tuple[str, ...]
    declared_shard_count: int
    source_commit: str
    task9_publication_sha256: str
    destination_sha256: str

    def as_dict(self) -> dict[str, Any]:
        """Return the canonical JSON-compatible receipt body."""
        return {
            "schema_version": 1,
            "arm": "B-balanced",
            "ready": self.ready,
            "blocker_codes": list(self.blocker_codes),
            "declared_shard_count": self.declared_shard_count,
            "source_commit": self.source_commit,
            "task9_publication_sha256": self.task9_publication_sha256,
            "destination_sha256": self.destination_sha256,
        }


def load_task9_balanced_view(path: Path) -> Task9BalancedView:
    """Load the documented Task9 JSON projection without importing Task9 internals."""
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("Task9 B projection must be a JSON object")
    try:
        return Task9BalancedView(
            strategy=_string(raw, "strategy"),
            occurrence_count=_integer(raw, "occurrence_count"),
            cell_occurrence_counts=_integer_mapping(raw, "cell_occurrence_counts"),
            multilingual_occurrence_counts=_integer_mapping(
                raw, "multilingual_occurrence_counts"
            ),
            declared_shards=_string_tuple(raw, "declared_shards"),
            shard_root=Path(_string(raw, "shard_root")),
            trainer_epochs=_integer(raw, "trainer_epochs"),
            global_batch_size=_integer(raw, "global_batch_size"),
            segment_occurrences=_integer_pair(raw, "segment_occurrences"),
            segment_steps=_integer_pair(raw, "segment_steps"),
            cumulative_steps=_integer_pair(raw, "cumulative_steps"),
            segment_final_valid_occurrences=_integer_pair(raw, "segment_final_valid_occurrences"),
            source_native_responses=_boolean(raw, "source_native_responses"),
            publication_sha256=_string(raw, "publication_sha256"),
            destination_sha256=_string(raw, "destination_sha256"),
            tokenizer_sha256=_string(raw, "tokenizer_sha256"),
            chat_template_sha256=_string(raw, "chat_template_sha256"),
            assistant_loss_mask_sha256=_string(raw, "assistant_loss_mask_sha256"),
        )
    except KeyError as error:
        raise ValueError(f"Task9 B projection lacks {error.args[0]}") from error


def assess_b_readiness(inputs: BReadinessInputs) -> BReadinessReceipt:
    """Validate B independently, emitting stable fail-closed blocker codes."""
    view = inputs.task9_b_view
    blockers: list[str] = []
    _require(view.strategy == "B-balanced", "B_STRATEGY", blockers)
    _require(view.occurrence_count == 2_000_000, "B_OCCURRENCE_COUNT", blockers)
    _require(view.cell_occurrence_counts == _CELL_QUOTAS, "B_CELL_QUOTA", blockers)
    _require(view.multilingual_occurrence_counts == _LANGUAGE_QUOTAS, "B_LANGUAGE_QUOTA", blockers)
    _require(len(view.declared_shards) == 201, "B_DECLARED_SHARD_COUNT", blockers)
    _require(len(set(view.declared_shards)) == len(view.declared_shards), "B_DUPLICATE_SHARD", blockers)
    _require(_declared_paths_are_safe(view.declared_shards), "B_INVALID_SHARD_PATH", blockers)
    _require(_declared_shards_exist(view), "B_MISSING_DECLARED_SHARD", blockers)
    _require(_has_no_orphan_shards(view), "B_ORPHAN_SHARD", blockers)
    _require(view.trainer_epochs == 1, "B_TRAINER_EPOCHS", blockers)
    _require(view.global_batch_size == 512, "B_GLOBAL_BATCH_SIZE", blockers)
    _require(view.segment_occurrences == (1_300_000, 700_000), "B_SEGMENT_OCCURRENCES", blockers)
    _require(view.segment_steps == (2_540, 1_368), "B_SEGMENT_STEPS", blockers)
    _require(view.cumulative_steps == (2_540, 3_908), "B_SCHEDULE", blockers)
    _require(view.segment_final_valid_occurrences == (32, 96), "B_SEGMENT_FINAL_VALID", blockers)
    _require(view.source_native_responses, "B_SOURCE_NATIVE_RESPONSES", blockers)
    _require(_COMMIT.fullmatch(inputs.source_commit) is not None, "B_SOURCE_COMMIT", blockers)
    _require(
        all(
            _SHA256.fullmatch(value) is not None
            for value in (
                view.publication_sha256,
                view.destination_sha256,
                view.tokenizer_sha256,
                view.chat_template_sha256,
                view.assistant_loss_mask_sha256,
            )
        ),
        "B_RECEIPT_IDENTITY",
        blockers,
    )
    return BReadinessReceipt(
        ready=not blockers,
        blocker_codes=tuple(blockers),
        declared_shard_count=len(view.declared_shards),
        source_commit=inputs.source_commit,
        task9_publication_sha256=view.publication_sha256,
        destination_sha256=view.destination_sha256,
    )


def write_b_readiness_receipt(path: Path, receipt: BReadinessReceipt) -> None:
    """Atomically write an immutable B-only readiness receipt."""
    if path.exists():
        raise FileExistsError(f"readiness receipt already exists: {path}")
    payload = receipt.as_dict()
    payload["receipt_sha256"] = _sha256_json(payload)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.partial-{os.getpid()}")
    temporary.write_text(json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n")
    os.replace(temporary, path)


def _declared_paths_are_safe(shards: tuple[str, ...]) -> bool:
    return all(
        bool(name)
        and not Path(name).is_absolute()
        and ".." not in Path(name).parts
        and Path(name).suffix in _DATA_SUFFIXES
        for name in shards
    )


def _declared_shards_exist(view: Task9BalancedView) -> bool:
    return view.shard_root.is_dir() and all((view.shard_root / shard).is_file() for shard in view.declared_shards)


def _has_no_orphan_shards(view: Task9BalancedView) -> bool:
    if not view.shard_root.is_dir():
        return False
    actual = {
        path.relative_to(view.shard_root).as_posix()
        for path in view.shard_root.rglob("*")
        if path.is_file() and path.suffix in _DATA_SUFFIXES
    }
    return actual == set(view.declared_shards)


def _require(condition: bool, blocker: str, blockers: list[str]) -> None:
    if not condition:
        blockers.append(blocker)


def _sha256_json(payload: dict[str, Any]) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _string(payload: dict[str, Any], key: str) -> str:
    value = payload[key]
    if not isinstance(value, str):
        raise ValueError(f"Task9 B projection {key} must be a string")
    return value


def _integer(payload: dict[str, Any], key: str) -> int:
    value = payload[key]
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(f"Task9 B projection {key} must be an integer")
    return value


def _integer_mapping(payload: dict[str, Any], key: str) -> dict[str, int]:
    value = payload[key]
    if not isinstance(value, dict) or not all(isinstance(name, str) and isinstance(count, int) and not isinstance(count, bool) for name, count in value.items()):
        raise ValueError(f"Task9 B projection {key} must be a string/integer mapping")
    return value


def _string_tuple(payload: dict[str, Any], key: str) -> tuple[str, ...]:
    value = payload[key]
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise ValueError(f"Task9 B projection {key} must be a string list")
    return tuple(value)


def _integer_pair(payload: dict[str, Any], key: str) -> tuple[int, int]:
    value = payload[key]
    if not isinstance(value, list) or len(value) != 2 or any(not isinstance(item, int) or isinstance(item, bool) for item in value):
        raise ValueError(f"Task9 B projection {key} must be a two-integer list")
    return (value[0], value[1])


def _boolean(payload: dict[str, Any], key: str) -> bool:
    value = payload[key]
    if not isinstance(value, bool):
        raise ValueError(f"Task9 B projection {key} must be boolean")
    return value
