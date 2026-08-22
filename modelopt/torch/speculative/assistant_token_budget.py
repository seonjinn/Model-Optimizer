# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""Deterministic, checkpointable assistant-token budget accounting."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from pathlib import Path


def local_token_allowance(counts: list[int], *, rank: int, remaining: int) -> int:
    """Allocate a global remaining budget deterministically in ascending rank order."""
    if not counts or any(type(count) is not int or count < 0 for count in counts):
        raise ValueError("rank token counts must be non-negative integers")
    if not 0 <= rank < len(counts):
        raise ValueError("rank is outside the gathered token counts")
    if type(remaining) is not int or remaining < 0:
        raise ValueError("remaining token budget must be a non-negative integer")
    prefix = sum(counts[:rank])
    return min(counts[rank], max(0, remaining - prefix))


def trim_binary_mask(mask: list[int], keep: int) -> list[int]:
    """Keep the first ``keep`` active positions of a row-major binary mask."""
    if any(value not in (0, 1) for value in mask):
        raise ValueError("assistant loss mask must be binary")
    if type(keep) is not int or not 0 <= keep <= sum(mask):
        raise ValueError("assistant loss-mask retention is out of range")
    remaining = keep
    trimmed: list[int] = []
    for value in mask:
        retain = int(value == 1 and remaining > 0)
        trimmed.append(retain)
        remaining -= retain
    return trimmed


class AssistantTokenBudgetController:
    """Track globally accepted tokens and persist only optimizer-step commits."""

    def __init__(self, *, target: int, training_fingerprint: str) -> None:
        """Initialize an empty counter for one immutable target and run identity."""
        if type(target) is not int or target < 1:
            raise ValueError("assistant-token target must be positive")
        if not training_fingerprint:
            raise ValueError("training fingerprint must be non-empty")
        self.target = target
        self.training_fingerprint = training_fingerprint
        self.committed = 0
        self.pending = 0
        self.global_step = 0

    @property
    def remaining(self) -> int:
        """Return tokens still available to the current target, including pending work."""
        return self.target - self.committed - self.pending

    @property
    def reached_target(self) -> bool:
        """Return whether committed optimizer steps exactly reached the target."""
        return self.committed == self.target and self.pending == 0

    def record_microbatch(self, globally_accepted: int) -> None:
        """Record accepted tokens provisionally until the optimizer step succeeds."""
        if type(globally_accepted) is not int or not 0 <= globally_accepted <= self.remaining:
            raise ValueError("accepted assistant-token count exceeds the remaining budget")
        self.pending += globally_accepted

    def commit_step(self, global_step: int) -> None:
        """Commit the current accumulation window after an optimizer step completes."""
        if type(global_step) is not int or global_step <= self.global_step:
            raise ValueError("assistant-token state requires an increasing global step")
        self.committed += self.pending
        self.pending = 0
        self.global_step = global_step

    def state_dict(self) -> dict[str, Any]:
        """Return the checkpoint payload; pending microbatches are never persisted."""
        if self.pending:
            raise ValueError("cannot checkpoint an uncommitted assistant-token accumulation")
        return {
            "schema_version": 1,
            "training_fingerprint": self.training_fingerprint,
            "committed_assistant_tokens": self.committed,
            "global_step": self.global_step,
        }

    def load_checkpoint(self, path: Path, *, expected_global_step: int) -> None:
        """Restore a counter only when checkpoint identity and trainer step agree."""
        payload = json.loads(path.read_text(encoding="utf-8"))
        expected = {
            "schema_version": 1,
            "training_fingerprint": self.training_fingerprint,
            "global_step": expected_global_step,
        }
        if not isinstance(payload, dict) or any(
            payload.get(field) != value for field, value in expected.items()
        ):
            raise ValueError("assistant-token checkpoint identity mismatch")
        committed = payload.get("committed_assistant_tokens")
        if type(committed) is not int or not 0 <= committed <= self.target:
            raise ValueError("assistant-token checkpoint count is invalid for the target")
        self.committed = committed
        self.pending = 0
        self.global_step = expected_global_step
