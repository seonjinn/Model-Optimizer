# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Materialize the exact scaled B-balanced source-native canary view."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

__all__ = [
    "CANARY_CELL_QUOTAS",
    "CANARY_LANGUAGE_QUOTAS",
    "build_canary_occurrences",
    "write_canary_occurrences",
]

CANARY_CELL_QUOTAS = {
    "math": 25_600,
    "code": 20_480,
    "stem": 25_600,
    "chat": 20_480,
    "multilingual": 10_240,
}
CANARY_LANGUAGE_QUOTAS = {language: 2_048 for language in ("de", "ja", "es", "fr", "it")}


def build_canary_occurrences(rows: Iterable[Mapping[str, object]], *, seed: int) -> list[dict[str, object]]:
    """Deterministically select the exact B-scaled cell and language quota view."""
    candidates: dict[tuple[str, str], list[dict[str, object]]] = {}
    for ordinal, raw in enumerate(rows):
        row = dict(raw)
        cell = _required_string(row, "cell").lower()
        language = _required_string(row, "language").lower()
        if cell not in CANARY_CELL_QUOTAS:
            raise ValueError(f"unsupported B canary cell: {cell}")
        if cell == "multilingual":
            if language not in CANARY_LANGUAGE_QUOTAS:
                raise ValueError(f"unsupported B canary language: {language}")
        elif language:
            raise ValueError("non-multilingual B canary rows must have an empty language")
        if row.get("source_native") is not True:
            raise ValueError("B canary requires source-native completions")
        identifier = _required_string(row, "uuid")
        row["cell"] = cell
        row["language"] = language
        row["_rank"] = _rank(seed, ordinal, identifier)
        candidates.setdefault((cell, language), []).append(row)

    selected: list[dict[str, object]] = []
    for cell, quota in CANARY_CELL_QUOTAS.items():
        if cell == "multilingual":
            for language, language_quota in CANARY_LANGUAGE_QUOTAS.items():
                selected.extend(_take_exact(candidates.get((cell, language), []), language_quota, f"{cell}/{language}"))
        else:
            selected.extend(_take_exact(candidates.get((cell, ""), []), quota, cell))
    selected.sort(key=lambda row: str(row["_rank"]))
    for row in selected:
        row.pop("_rank")
    if len(selected) != 102_400:
        raise AssertionError("B canary selection must contain 102400 occurrences")
    return selected


def write_canary_occurrences(path: Path, rows: Iterable[Mapping[str, object]]) -> None:
    """Atomically write canonical JSONL B canary occurrences."""
    if path.exists():
        raise FileExistsError(f"B canary output already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.partial-{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as destination:
        for row in rows:
            destination.write(json.dumps(dict(row), sort_keys=True, separators=(",", ":")) + "\n")
    os.replace(temporary, path)


def _take_exact(rows: list[dict[str, object]], quota: int, label: str) -> list[dict[str, object]]:
    if len(rows) < quota:
        raise ValueError(f"B canary {label} has {len(rows)} candidates, requires {quota}")
    return sorted(rows, key=lambda row: str(row["_rank"]))[:quota]


def _rank(seed: int, ordinal: int, identifier: str) -> str:
    return hashlib.sha256(f"{seed}:{ordinal}:{identifier}".encode()).hexdigest()


def _required_string(row: Mapping[str, object], key: str) -> str:
    value = row.get(key)
    if not isinstance(value, str):
        raise ValueError(f"B canary row {key} must be a string")
    return value


def main() -> int:
    """Build a B canary JSONL file from an authenticated Task9 B occurrence stream."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    args = parser.parse_args()
    with args.input.open(encoding="utf-8") as source:
        rows = [json.loads(line) for line in source if line.strip()]
    write_canary_occurrences(args.output, build_canary_occurrences(rows, seed=args.seed))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
