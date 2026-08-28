# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Lustre-compatible immutable publication boundary tests."""

from __future__ import annotations

import ctypes
import errno
from typing import TYPE_CHECKING

import pytest
from common.specdec import qwen4b_b_atomic as publication

if TYPE_CHECKING:
    import os
    from pathlib import Path


class _UnsupportedRename:
    argtypes: object = None
    restype: object = None

    def __call__(self, *_args: object) -> int:
        ctypes.set_errno(errno.EINVAL)
        return -1


class _UnsupportedRenameLibrary:
    renameat2 = _UnsupportedRename()


def test_file_publication_survives_lustre_renameat2_einval(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An unsupported Lustre rename flag cannot block immutable file publication."""
    destination = tmp_path / "receipt.json"
    monkeypatch.setattr(publication.platform, "system", lambda: "Linux")
    monkeypatch.setattr(
        publication.ctypes, "CDLL", lambda *_args, **_kwargs: _UnsupportedRenameLibrary()
    )

    publication.atomic_publish_bytes(destination, b"receipt\n", job_id="lustre")

    assert destination.read_bytes() == b"receipt\n"
    assert (tmp_path / ".receipt.json.partial-lustre").read_bytes() == b"receipt\n"


def test_file_fallback_rejects_same_size_source_mutation_after_reread(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Fallback publication cannot authenticate bytes changed after the staged reread."""
    destination = tmp_path / "receipt.json"
    partial = tmp_path / ".receipt.json.partial-lustre"
    original_copy = publication._copy_file_with_exclusive_destination

    monkeypatch.setattr(publication.platform, "system", lambda: "Linux")
    monkeypatch.setattr(
        publication.ctypes, "CDLL", lambda *_args, **_kwargs: _UnsupportedRenameLibrary()
    )

    def mutate_before_copy(
        source: Path,
        target: Path,
        *,
        expected_sha256: str,
        expected_source_identity: tuple[int, int],
        expected_parent_identity: tuple[int, int],
        parent_fd: int,
    ) -> tuple[int, int]:
        assert source == partial
        source.write_bytes(b"forged!!\n")
        return original_copy(
            source,
            target,
            expected_sha256=expected_sha256,
            expected_source_identity=expected_source_identity,
            expected_parent_identity=expected_parent_identity,
            parent_fd=parent_fd,
        )

    monkeypatch.setattr(publication, "_copy_file_with_exclusive_destination", mutate_before_copy)

    with pytest.raises(publication.Task10PublicationError, match=r"changed|content"):
        publication.atomic_publish_bytes(destination, b"trusted!\n", job_id="lustre")

    assert partial.read_bytes() == b"forged!!\n"


def test_file_fallback_rejects_parent_replacement_at_final_durability(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A same-UID parent swap cannot redirect the final durability boundary."""
    parent = tmp_path / "publication"
    parent.mkdir()
    destination = parent / "receipt.json"
    moved_parent = tmp_path / "moved-publication"
    monkeypatch.setattr(publication.platform, "system", lambda: "Linux")
    monkeypatch.setattr(
        publication.ctypes, "CDLL", lambda *_args, **_kwargs: _UnsupportedRenameLibrary()
    )
    expected_parent = parent.stat()
    expected_parent_identity = (expected_parent.st_dev, expected_parent.st_ino)
    real_fsync = publication.os.fsync
    attacked = False

    def replace_parent(descriptor: int) -> None:
        nonlocal attacked
        real_fsync(descriptor)
        status = publication.os.fstat(descriptor)
        if not attacked and (status.st_dev, status.st_ino) == expected_parent_identity:
            attacked = True
            parent.rename(moved_parent)
            parent.mkdir()
            (parent / destination.name).write_bytes(b"foreign\n")

    monkeypatch.setattr(publication.os, "fsync", replace_parent)

    with pytest.raises(publication.Task10PublicationError, match=r"parent|rebound"):
        publication.atomic_publish_bytes(destination, b"receipt\n", job_id="replacement")

    assert attacked
    assert destination.read_bytes() == b"foreign\n"
    assert (moved_parent / destination.name).read_bytes() == b"receipt\n"


def test_file_publication_parent_swap_before_partial_create_never_mutates_replacement(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Partial creation remains under one held authenticated parent after a pathname swap."""
    parent = tmp_path / "publication"
    parent.mkdir()
    destination = parent / "receipt.json"
    partial = parent / ".receipt.json.partial-parent-swap"
    displaced = tmp_path / "publication-displaced"
    real_open = publication.os.open
    attacked = False

    def replace_parent_before_partial_create(
        path: str | bytes | Path,
        flags: int,
        mode: int = 0o777,
        *,
        dir_fd: int | None = None,
    ) -> int:
        nonlocal attacked
        if not attacked and (path == partial or (path == partial.name and dir_fd is not None)):
            attacked = True
            parent.rename(displaced)
            parent.mkdir()
        return real_open(path, flags, mode, dir_fd=dir_fd)

    monkeypatch.setattr(publication.os, "open", replace_parent_before_partial_create)

    with pytest.raises(publication.Task10PublicationError, match=r"parent|rebound"):
        publication.atomic_publish_bytes(destination, b"receipt\n", job_id="parent-swap")

    assert attacked
    assert not partial.exists()
    assert not destination.exists()


def test_directory_publication_survives_lustre_renameat2_einval(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A complete bundle remains adoptable when Lustre rejects no-replace rename."""
    partial = tmp_path / ".bundle.partial"
    partial.mkdir()
    (partial / "MANIFEST.json").write_bytes(b"manifest\n")
    nested = partial / "nested"
    nested.mkdir()
    (nested / "DATA.jsonl").write_bytes(b"data\n")
    destination = tmp_path / "bundle"
    monkeypatch.setattr(publication.platform, "system", lambda: "Linux")
    monkeypatch.setattr(
        publication.ctypes, "CDLL", lambda *_args, **_kwargs: _UnsupportedRenameLibrary()
    )

    publication.atomic_publish_directory(partial, destination)

    assert (partial / "MANIFEST.json").read_bytes() == b"manifest\n"
    assert (destination / "MANIFEST.json").read_bytes() == b"manifest\n"
    assert (destination / "nested" / "DATA.jsonl").read_bytes() == b"data\n"


def test_directory_publication_rejects_a_caller_bound_partial_replacement(tmp_path: Path) -> None:
    """A caller-bound partial inode cannot be silently redefined before publication."""
    partial = tmp_path / ".bundle.partial"
    partial.mkdir()
    (partial / "MANIFEST.json").write_bytes(b"trusted\n")
    expected = partial.stat()
    displaced = tmp_path / "displaced-partial"
    partial.rename(displaced)
    partial.mkdir()
    (partial / "MANIFEST.json").write_bytes(b"foreign\n")
    destination = tmp_path / "bundle"

    with pytest.raises(publication.Task10PublicationError, match=r"partial.*identity"):
        publication.atomic_publish_directory(
            partial,
            destination,
            expected_directory_identity=(expected.st_dev, expected.st_ino),
        )

    assert not destination.exists()
    assert (partial / "MANIFEST.json").read_bytes() == b"foreign\n"


def test_directory_publication_rejects_a_caller_bound_parent_replacement(tmp_path: Path) -> None:
    """A caller-bound parent inode rejects publication through a replacement pathname."""
    parent = tmp_path / "publication"
    parent.mkdir()
    expected_parent = parent.stat()
    displaced = tmp_path / "displaced-publication"
    parent.rename(displaced)
    parent.mkdir()
    partial = parent / ".bundle.partial"
    partial.mkdir()
    (partial / "MANIFEST.json").write_bytes(b"foreign\n")
    partial_status = partial.stat()
    destination = parent / "bundle"

    with pytest.raises(publication.Task10PublicationError, match=r"parent.*identity"):
        publication.atomic_publish_directory(
            partial,
            destination,
            expected_directory_identity=(partial_status.st_dev, partial_status.st_ino),
            expected_parent_identity=(expected_parent.st_dev, expected_parent.st_ino),
        )

    assert not destination.exists()
    assert (partial / "MANIFEST.json").read_bytes() == b"foreign\n"


def test_directory_publication_rejects_parent_swap_before_any_reservation_write(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A caller-bound parent is rechecked inside the reservation boundary before mkdir."""
    parent = tmp_path / "publication"
    parent.mkdir()
    partial = parent / ".bundle.partial"
    partial.mkdir()
    (partial / "MANIFEST.json").write_bytes(b"trusted\n")
    destination = parent / "bundle"
    expected_parent = parent.stat()
    expected_partial = partial.stat()
    displaced = tmp_path / "publication-displaced"
    original_reservation = publication._rename_directory_with_reservation
    attacked = False

    def replace_parent_before_reservation(
        source: Path,
        target: Path,
        **kwargs: object,
    ) -> tuple[int, int]:
        nonlocal attacked
        attacked = True
        parent.rename(displaced)
        parent.mkdir()
        (displaced / source.name).rename(parent / source.name)
        return original_reservation(source, target, **kwargs)

    monkeypatch.setattr(
        publication,
        "_rename_directory_with_reservation",
        replace_parent_before_reservation,
    )

    with pytest.raises(
        publication.Task10PublicationError,
        match=r"parent.*(?:changed|rebound)|identity",
    ):
        publication.atomic_publish_directory(
            partial,
            destination,
            expected_directory_identity=(expected_partial.st_dev, expected_partial.st_ino),
            expected_parent_identity=(expected_parent.st_dev, expected_parent.st_ino),
        )

    assert attacked
    assert not destination.exists()
    assert not (destination / publication.Q30_DIRECTORY_COMPLETION_MARKER).exists()


def test_directory_fallback_rejects_same_size_source_mutation_after_fsync(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A directory fallback cannot redefine authenticated source bytes during copying."""
    partial = tmp_path / ".bundle.partial"
    partial.mkdir()
    source = partial / "MANIFEST.json"
    source.write_bytes(b"trusted!\n")
    destination = tmp_path / "bundle"
    original_copy = publication._copy_directory_at
    mutated = False

    def mutate_before_copy(source_fd: int, destination_fd: int, *, root: bool = False) -> None:
        nonlocal mutated
        if root and not mutated:
            mutated = True
            source.write_bytes(b"forged!!\n")
        original_copy(source_fd, destination_fd, root=root)

    monkeypatch.setattr(publication, "_copy_directory_at", mutate_before_copy)

    with pytest.raises(publication.Task10PublicationError, match=r"changed|copy"):
        publication.atomic_publish_directory(partial, destination)

    assert mutated is True


def test_directory_publication_rejects_partial_root_replacement_before_digest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Every partial-root reopen must remain bound to the initially observed inode."""
    partial = tmp_path / ".bundle.partial"
    partial.mkdir()
    (partial / "MANIFEST.json").write_bytes(b"trusted")
    displaced = tmp_path / "displaced"
    forged = tmp_path / "forged"
    forged.mkdir()
    (forged / "MANIFEST.json").write_bytes(b"forged!")
    destination = tmp_path / "bundle"
    original_digest = publication._directory_content_sha256
    swapped = False

    def replace_before_digest(path: Path) -> str:
        nonlocal swapped
        if path == partial and not swapped:
            swapped = True
            partial.rename(displaced)
            forged.rename(partial)
        return original_digest(path)

    monkeypatch.setattr(publication, "_directory_content_sha256", replace_before_digest)

    with pytest.raises(publication.Task10PublicationError, match=r"partial|root|changed"):
        publication.atomic_publish_directory(partial, destination)

    assert swapped is True
    assert (displaced / "MANIFEST.json").read_bytes() == b"trusted"


def test_directory_content_digest_frames_file_lengths(tmp_path: Path) -> None:
    """File bytes cannot absorb the encoded header of a following tree entry."""
    first = tmp_path / "first"
    second = tmp_path / "second"
    first.mkdir()
    second.mkdir()
    mode = 0o644
    encoded_b = (1).to_bytes(8, "big") + b"b" + mode.to_bytes(4, "big") + b"F"
    (first / "a").write_bytes(b"alpha" + encoded_b + b"beta")
    (second / "a").write_bytes(b"alpha")
    (second / "b").write_bytes(b"beta")
    for path in (first / "a", second / "a", second / "b"):
        path.chmod(mode)

    assert publication._directory_content_sha256(first) != publication._directory_content_sha256(
        second
    )


def test_directory_fallback_never_replaces_a_same_uid_reservation_swap(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A replacement after reservation authentication is preserved and rejected."""
    partial = tmp_path / ".bundle.partial"
    partial.mkdir()
    (partial / "MANIFEST.json").write_bytes(b"manifest\n")
    destination = tmp_path / "bundle"
    displaced = tmp_path / "displaced-reservation"
    monkeypatch.setattr(publication.platform, "system", lambda: "Linux")
    monkeypatch.setattr(
        publication.ctypes, "CDLL", lambda *_args, **_kwargs: _UnsupportedRenameLibrary()
    )
    real_open = publication.os.open
    real_rename = publication.os.rename
    replacement_identity: tuple[int, int] | None = None

    def replace_before_copy(
        path: str | bytes | Path,
        flags: int,
        mode: int = 0o777,
        *,
        dir_fd: int | None = None,
    ) -> int:
        nonlocal replacement_identity
        if path == "MANIFEST.json" and dir_fd is not None and destination.exists():
            held = publication.os.fstat(dir_fd)
            named = destination.stat()
            if (held.st_dev, held.st_ino) == (named.st_dev, named.st_ino):
                real_rename(destination, displaced)
                destination.mkdir()
                status = destination.stat()
                replacement_identity = (status.st_dev, status.st_ino)
        return real_open(path, flags, mode, dir_fd=dir_fd)

    monkeypatch.setattr(publication.os, "open", replace_before_copy)

    with pytest.raises(publication.Task10PublicationError, match=r"changed|rebound|replacement"):
        publication.atomic_publish_directory(partial, destination)

    if replacement_identity is None:
        pytest.fail("publication never copied through its held destination descriptor")
    status = destination.stat()
    assert (status.st_dev, status.st_ino) == replacement_identity


def test_directory_fallback_never_deletes_a_replaced_partial_after_commit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Committed publication retains recovery input instead of deleting a rebound pathname."""
    partial = tmp_path / ".bundle.partial"
    partial.mkdir()
    (partial / "MANIFEST.json").write_bytes(b"manifest\n")
    displaced = tmp_path / "displaced-partial"
    destination = tmp_path / "bundle"
    monkeypatch.setattr(publication.platform, "system", lambda: "Linux")
    monkeypatch.setattr(
        publication.ctypes, "CDLL", lambda *_args, **_kwargs: _UnsupportedRenameLibrary()
    )
    real_fsync = publication.os.fsync
    attacked = False
    replacement_identity: tuple[int, int] | None = None

    def replace_partial_after_commit(descriptor: int) -> None:
        nonlocal attacked, replacement_identity
        real_fsync(descriptor)
        if (
            not attacked
            and (destination / "MANIFEST.json").is_file()
            and (destination / publication.Q30_DIRECTORY_COMPLETION_MARKER).is_file()
            and (destination / publication.Q30_DIRECTORY_COMPLETION_MARKER).read_bytes()
            == b"q30-directory-publication-complete-v1\n"
            and partial.exists()
        ):
            attacked = True
            partial.rename(displaced)
            partial.mkdir()
            (partial / "foreign").write_bytes(b"foreign\n")
            status = partial.stat()
            replacement_identity = (status.st_dev, status.st_ino)

    monkeypatch.setattr(publication.os, "fsync", replace_partial_after_commit)

    publication.atomic_publish_directory(partial, destination)

    assert replacement_identity is not None
    status = partial.stat()
    assert (status.st_dev, status.st_ino) == replacement_identity
    assert (partial / "foreign").read_bytes() == b"foreign\n"
    assert (destination / "MANIFEST.json").read_bytes() == b"manifest\n"


def test_directory_fallback_never_unlinks_a_replaced_completion_marker(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A marker replacement after pathname authentication is never deleted."""
    partial = tmp_path / ".bundle.partial"
    partial.mkdir()
    (partial / "MANIFEST.json").write_bytes(b"manifest\n")
    destination = tmp_path / "bundle"
    marker = ".q30-publication-incomplete"
    displaced_marker = tmp_path / ".q30-publication-displaced"
    monkeypatch.setattr(publication.platform, "system", lambda: "Linux")
    monkeypatch.setattr(
        publication.ctypes, "CDLL", lambda *_args, **_kwargs: _UnsupportedRenameLibrary()
    )
    real_stat = publication.os.stat
    attacked = False

    def replace_after_marker_stat(
        path: str | bytes | Path,
        *,
        dir_fd: int | None = None,
        follow_symlinks: bool = True,
    ) -> os.stat_result:
        nonlocal attacked
        status = real_stat(path, dir_fd=dir_fd, follow_symlinks=follow_symlinks)
        if path == marker and dir_fd is not None and not attacked:
            attacked = True
            publication.os.rename(
                marker,
                displaced_marker,
                src_dir_fd=dir_fd,
            )
            descriptor = publication.os.open(
                marker,
                publication.os.O_WRONLY | publication.os.O_CREAT | publication.os.O_EXCL,
                0o600,
                dir_fd=dir_fd,
            )
            try:
                publication.os.write(descriptor, b"foreign\n")
                publication.os.fsync(descriptor)
            finally:
                publication.os.close(descriptor)
        return status

    monkeypatch.setattr(publication.os, "stat", replace_after_marker_stat)

    with pytest.raises(publication.Task10PublicationError, match=r"marker|changed|rebound"):
        publication.atomic_publish_directory(partial, destination)

    assert attacked
    assert (destination / marker).read_bytes() == b"foreign\n"
    assert displaced_marker.exists()


def test_file_fallback_never_unlinks_a_replaced_partial(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A replacement after final source stat remains untouched recovery evidence."""
    destination = tmp_path / "receipt.json"
    partial = tmp_path / ".receipt.json.partial-replacement"
    displaced = tmp_path / "displaced-partial"
    monkeypatch.setattr(publication.platform, "system", lambda: "Linux")
    monkeypatch.setattr(
        publication.ctypes, "CDLL", lambda *_args, **_kwargs: _UnsupportedRenameLibrary()
    )
    real_stat = publication.os.stat
    source_stats = 0
    replacement_identity: tuple[int, int] | None = None

    def replace_after_source_stat(
        path: str | bytes | Path,
        *,
        dir_fd: int | None = None,
        follow_symlinks: bool = True,
    ) -> os.stat_result:
        nonlocal source_stats, replacement_identity
        status = real_stat(path, dir_fd=dir_fd, follow_symlinks=follow_symlinks)
        if path == partial.name and dir_fd is not None:
            source_stats += 1
            if source_stats == 2:
                partial.rename(displaced)
                partial.write_bytes(b"foreign\n")
                replacement = partial.stat()
                replacement_identity = (replacement.st_dev, replacement.st_ino)
        return status

    monkeypatch.setattr(publication.os, "stat", replace_after_source_stat)

    with pytest.raises(publication.Task10PublicationError, match=r"changed|rebound|stable"):
        publication.atomic_publish_bytes(destination, b"receipt\n", job_id="replacement")

    assert replacement_identity is not None
    status = partial.stat()
    assert (status.st_dev, status.st_ino) == replacement_identity
    assert partial.read_bytes() == b"foreign\n"
    assert not destination.exists()
