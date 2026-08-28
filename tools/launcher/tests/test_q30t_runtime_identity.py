# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Behavioral tests for canonical Q30 runtime-tree identity evidence."""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

import pytest
from common.specdec import q30t_runtime_identity as module
from common.specdec.q30t_runtime_identity import runtime_tree_identity

if TYPE_CHECKING:
    from pathlib import Path


def test_shared_runtime_identity_matches_frozen_legacy_digest(tmp_path: Path) -> None:
    """A legacy-compatible runtime fixture preserves its approved digest."""
    runtime = tmp_path / "runtime"
    (runtime / "bin").mkdir(parents=True)
    (runtime / "bin/python3.12").write_bytes(b"python")
    (runtime / "bin/python").symlink_to("python3.12")
    observed = runtime_tree_identity(runtime)
    assert observed.sha256 == "3992ed625fc893f39e1a30148e9d9df90ab23cbe2bb0a4ed97d15842267ae673"
    assert observed.file_count == 1
    assert observed.symlink_count == 1
    assert observed.total_regular_bytes == 6


def test_runtime_identity_rejects_fifo(tmp_path: Path) -> None:
    """A runtime tree cannot silently digest a FIFO as a regular entry."""
    runtime = tmp_path / "runtime"
    runtime.mkdir()
    os.mkfifo(runtime / "controller.fifo")
    with pytest.raises(ValueError, match="unsupported entry"):
        runtime_tree_identity(runtime)


def test_runtime_identity_rejects_regular_file_rebind(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A pathname mutation after hashing cannot reuse an old file digest."""
    runtime = tmp_path / "runtime"
    runtime.mkdir()
    entry = runtime / "pyvenv.cfg"
    entry.write_text("home=/runtime\n")
    monkeypatch.setattr(module, "_POST_FILE_HASH_HOOK", lambda: entry.write_text("changed\n"))
    with pytest.raises(ValueError, match="changed while hashing"):
        runtime_tree_identity(runtime)


def test_runtime_identity_changes_when_symlink_target_changes(tmp_path: Path) -> None:
    """The digest includes the symlink target rather than only its pathname."""
    runtime = tmp_path / "runtime"
    runtime.mkdir()
    link = runtime / "python"
    link.symlink_to("python3.12")
    before = runtime_tree_identity(runtime).sha256
    link.unlink()
    link.symlink_to("python3")
    assert runtime_tree_identity(runtime).sha256 != before
