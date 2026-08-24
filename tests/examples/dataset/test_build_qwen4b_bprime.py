# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Production Task5 B-prime builder contracts."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

MODULE_DIR = Path(__file__).resolve().parents[3] / "examples/dataset"
sys.path.insert(0, str(MODULE_DIR))
try:
    from build_qwen4b_bprime import baseline_exclusion_from_audit
    from build_specdec_inventory import make_exclusion_receipt
finally:
    sys.path.pop(0)


def test_baseline_exclusion_replays_the_audit_receipt(tmp_path: Path) -> None:
    prompt_ids = ("1" * 64, "2" * 64)
    expected = make_exclusion_receipt("baseline", prompt_ids)
    path = tmp_path / "audit.json"
    path.write_text(
        json.dumps(
            {
                "source_revision": "5c89e01dd720ae0f4058445ed49c5fb68a03c76e",
                "row_count": 1_300_000,
                "prompt_uuids": list(reversed(prompt_ids)),
                "receipt_sha256": expected.receipt_sha256,
            }
        )
    )

    assert baseline_exclusion_from_audit(path) == expected


def test_baseline_exclusion_rejects_a_self_inconsistent_audit(tmp_path: Path) -> None:
    path = tmp_path / "audit.json"
    path.write_text(
        json.dumps(
            {
                "source_revision": "5c89e01dd720ae0f4058445ed49c5fb68a03c76e",
                "row_count": 1_300_000,
                "prompt_uuids": ["1" * 64],
                "receipt_sha256": "0" * 64,
            }
        )
    )

    with pytest.raises(ValueError, match="receipt identity"):
        baseline_exclusion_from_audit(path)
