# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Behavioral tests for canonical Q30 runtime-tree identity evidence."""

from __future__ import annotations

import os
from pathlib import Path

import pytest
from common.specdec import q30t_runtime_identity as module
from common.specdec.q30t_runtime_identity import runtime_tree_identity


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


def test_runtime_identity_rejects_root_alias_before_opening(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The double-slash filesystem-root alias cannot start a tree traversal."""

    def unexpected_open(*args: object, **kwargs: object) -> int:
        raise AssertionError(f"runtime identity opened root alias: {args}, {kwargs}")

    monkeypatch.setattr(module.os, "open", unexpected_open)
    with pytest.raises(ValueError, match="absolute non-root"):
        runtime_tree_identity(Path("//"))


def test_runtime_identity_rejects_non_utf8_entry_path() -> None:
    """A surrogateescaped entry name cannot escape as an encoding error."""
    with pytest.raises(ValueError, match="UTF-8"):
        module._require_canonical_relative_path(os.fsdecode(b"\xff"))


def test_runtime_identity_rejects_non_utf8_symlink_target(tmp_path: Path) -> None:
    """A surrogateescaped symlink target cannot escape as an encoding error."""
    runtime = tmp_path / "runtime"
    runtime.mkdir()
    os.symlink(b"\xff", os.fsencode(runtime / "python"))
    with pytest.raises(ValueError, match="UTF-8"):
        runtime_tree_identity(runtime)
