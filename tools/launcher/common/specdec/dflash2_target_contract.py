# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Exact target-family contracts shared by DFlash2 training and serve gates."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path

__all__ = [
    "DFlash2TargetSpec",
    "dflash2_target_spec",
    "validate_dflash2_target_snapshot",
]


@dataclass(frozen=True)
class DFlash2TargetSpec:
    """Immutable snapshot and topology identity for one supported target."""

    revision: str
    capture_ids: tuple[int, ...]
    serve_tp: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    intermediate_size: int


_TARGET_SPECS = {
    "q30-base": DFlash2TargetSpec(
        "ad44e777bcd18fa416d9da3bd8f70d33ebb85d39",
        (2, 13, 24, 35, 46, 48),
        2,
        32,
        4,
        128,
        6144,
    ),
    "q30-thinking": DFlash2TargetSpec(
        "144afc2f379b542fdd4e85a1fcd5e1f79112d95d",
        (2, 13, 24, 35, 46, 48),
        2,
        32,
        4,
        128,
        6144,
    ),
    "q235-base": DFlash2TargetSpec(
        "8efa61729e24bd65b1d152b5ab5409052aa80e65",
        (2, 25, 47, 69, 92, 94),
        4,
        64,
        4,
        128,
        12288,
    ),
    "q235-thinking": DFlash2TargetSpec(
        "6cbffae6d8e28b986a6b17bd36f42f9fa0f1f0a5",
        (2, 25, 47, 69, 92, 94),
        4,
        64,
        4,
        128,
        12288,
    ),
}


def dflash2_target_spec(target_label: str) -> DFlash2TargetSpec:
    """Return the exact target contract or fail closed for an unknown label."""
    try:
        return _TARGET_SPECS[target_label]
    except KeyError as error:
        raise ValueError(f"unsupported DFlash2 target label: {target_label}") from error


def validate_dflash2_target_snapshot(
    target_path: Path,
    target_label: str,
) -> DFlash2TargetSpec:
    """Bind a target label to its exact snapshot revision and architecture dimensions."""
    spec = dflash2_target_spec(target_label)
    snapshot = json.loads((target_path / "snapshot-manifest.json").read_text())
    if snapshot.get("source_identity") != spec.revision:
        raise ValueError("DFlash2 target snapshot revision mismatch")
    config = json.loads((target_path / "config.json").read_text())
    actual_dims = tuple(
        config.get(name)
        for name in (
            "num_attention_heads",
            "num_key_value_heads",
            "head_dim",
            "intermediate_size",
        )
    )
    expected_dims = (
        spec.num_attention_heads,
        spec.num_key_value_heads,
        spec.head_dim,
        spec.intermediate_size,
    )
    if actual_dims != expected_dims:
        raise ValueError("DFlash2 target dimensions mismatch")
    return spec
