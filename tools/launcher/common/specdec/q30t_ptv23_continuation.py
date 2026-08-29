# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Qwen3-30B-A3B Thinking experimental 700K continuation contract."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import stat
import subprocess
import sys
from dataclasses import dataclass, fields, replace
from pathlib import Path
from types import MappingProxyType
from typing import TYPE_CHECKING, Literal, Self

from common.specdec.q30t_parent_receipt import (
    Q30TParentReceipt,
    load_q30t_parent_receipt,
    require_matching_historical_lineage,
)
from common.specdec.q30t_tokenizer_receipt import load_q30t_tokenizer_receipt

if TYPE_CHECKING:
    from collections.abc import Mapping

__all__ = [
    "ExposureSchedule",
    "Method",
    "Q30TContinuationContract",
    "Q30TTrainingLaunch",
    "build_q30_node_local_training_command",
    "build_q30_training_command",
    "load_q30_launch_descriptor",
    "require_q30_parent_pair_for_shared_dataset",
    "write_q30_launch_descriptor",
]

Method = Literal["DFlash", "DSpark"]
Stage = Literal["canary", "full"]
TARGET_MODEL = "Qwen/Qwen3-30B-A3B-Thinking-2507"
_CAPTURE_IDS = "[2,13,24,35,46,48]"
_APPROVED_SCHEDULE = (700_000, 1_368, 512, 96, 20)
_DATASET_IDENTITY = "ptv2-ptv3-complement-700k-qwen3-30ba3b-thinking-swe-heavy-v1"
_APPROVED_QUOTAS = {
    "ptv2_stem": 200_000,
    "ptv2_multilingual_ja": 25_000,
    "ptv2_multilingual_es": 25_000,
    "ptv2_multilingual_fr": 25_000,
    "ptv2_multilingual_it": 25_000,
    "ptv3_swe_v3": 180_000,
    "ptv3_interactive_agentic_swe": 60_000,
    "ptv3_general_tool_trajectories": 160_000,
}
APPROVED_Q30T_DATASET_COMPLETION_RECEIPT_FILE_SHA256S: frozenset[str] = frozenset()
APPROVED_Q30T_RUNTIME_TREE_SHA256_BY_ARCHIVE_SHA256: Mapping[str, str] = MappingProxyType({})
APPROVED_Q30T_PRODUCTION_RUNTIME_IMAGE_SHA256S: frozenset[str] = frozenset()
APPROVED_Q30T_SOURCE_BY_ARCHIVE_SHA256: Mapping[str, tuple[str, str, str]] = MappingProxyType({})


def _is_lower_hex(value: object, length: int) -> bool:
    return (
        type(value) is str
        and len(value) == length
        and all(character in "0123456789abcdef" for character in value)
    )


def _is_exact_int(value: object) -> bool:
    return type(value) is int


def _paths_overlap(first: Path, second: Path) -> bool:
    first_parts = first.parts
    second_parts = second.parts
    prefix_length = min(len(first_parts), len(second_parts))
    return first_parts[:prefix_length] == second_parts[:prefix_length]


def _require_canonical_nofollow_path(path: Path, *, label: str) -> None:
    """Reject lexical aliases and symlinked existing components before path comparison."""
    rendered = str(path)
    if (
        not path.is_absolute()
        or rendered != os.path.normpath(rendered)
        or any(component in {".", ".."} for component in path.parts)
    ):
        raise ValueError(f"{label} must use canonical absolute path components")
    nofollow = getattr(os, "O_NOFOLLOW", 0)
    if not isinstance(nofollow, int) or nofollow == 0:
        raise ValueError("platform lacks descriptor-safe O_NOFOLLOW path validation")
    directory_flags = os.O_RDONLY | os.O_DIRECTORY | nofollow
    descriptor = os.open("/", directory_flags)
    try:
        for index, component in enumerate(path.parts[1:]):
            is_last = index == len(path.parts[1:]) - 1
            flags = os.O_RDONLY | nofollow | getattr(os, "O_NONBLOCK", 0)
            if not is_last:
                flags |= os.O_DIRECTORY
            try:
                child = os.open(component, flags, dir_fd=descriptor)
            except FileNotFoundError:
                return
            except OSError as error:
                raise ValueError(
                    f"{label} contains an unreadable or symlinked component"
                ) from error
            os.close(descriptor)
            descriptor = child
    finally:
        os.close(descriptor)


def _require_pyxis_path(path: Path, *, label: str) -> None:
    """Reject characters that delimit or control a Pyxis mount specification."""
    if any(
        character in {",", ":"} or ord(character) < 32 or ord(character) == 127
        for character in str(path)
    ):
        raise ValueError(f"{label} contains a forbidden Pyxis path delimiter")


def _stable_file(path: Path, *, label: str, retain: bool = False) -> tuple[bytes | None, str]:
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        raise ValueError(f"{label} is unreadable: {path}") from error
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise ValueError(f"{label} must be a regular file: {path}")
        content = bytearray() if retain else None
        digest = hashlib.sha256()
        size = 0
        while block := os.read(descriptor, 8 * 1024 * 1024):
            size += len(block)
            if content is not None:
                content.extend(block)
            digest.update(block)
        after = os.fstat(descriptor)

        def identity(status: os.stat_result) -> tuple[int, int, int, int, int, int]:
            return (
                status.st_dev,
                status.st_ino,
                status.st_mode,
                status.st_size,
                status.st_mtime_ns,
                status.st_ctime_ns,
            )

        named = path.stat(follow_symlinks=False)
        if (
            identity(before) != identity(after)
            or identity(after) != identity(named)
            or size != before.st_size
        ):
            raise ValueError(f"{label} changed while reading: {path}")
        return bytes(content) if content is not None else None, digest.hexdigest()
    finally:
        os.close(descriptor)


def _require_file_sha256(
    path: Path, expected_sha256: str, *, label: str, retain: bool = False
) -> bytes | None:
    raw, observed = _stable_file(path, label=label, retain=retain)
    if observed != expected_sha256:
        raise ValueError(f"{label} SHA-256 mismatch")
    return raw


def _tree_sha256(root: Path) -> str:
    """Hash regular files and symlink targets in one staged runtime tree."""
    if root.is_symlink() or not root.is_dir():
        raise ValueError("staged runtime must be a no-follow directory")
    entries: list[list[object]] = []
    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root).as_posix()
        if path.is_symlink():
            entries.append([relative, "symlink", os.readlink(path)])
        elif path.is_file():
            _, digest = _stable_file(path, label=f"runtime entry {relative}")
            entries.append([relative, "regular", path.stat().st_size, digest])
        elif not path.is_dir():
            raise ValueError("staged runtime contains an unsupported entry")
    return hashlib.sha256(
        json.dumps(entries, ensure_ascii=False, separators=(",", ":")).encode()
    ).hexdigest()


def _model_tree_sha256(root: Path) -> str:
    """Hash the exact canonical regular-file evidence used by Q30 model receipts."""
    if root.is_symlink() or not root.is_dir():
        raise ValueError("Q30 model tree must be a no-follow directory")
    entries: list[dict[str, object]] = []
    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root).as_posix()
        if path.is_symlink() or (not path.is_file() and not path.is_dir()):
            raise ValueError("Q30 model tree contains a non-regular entry")
        if path.is_dir():
            continue
        metadata = path.stat(follow_symlinks=False)
        if metadata.st_nlink != 1:
            raise ValueError("Q30 model tree regular files must have one link")
        _, digest = _stable_file(path, label=f"Q30 model entry {relative}")
        entries.append(
            {
                "path": relative,
                "sha256": digest,
                "size": metadata.st_size,
                "type": "regular",
            }
        )
    if not entries:
        raise ValueError("Q30 model tree evidence is empty")
    return hashlib.sha256(_canonical_json(entries)).hexdigest()


def _canonical_json(value: object) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()


def _require_dataset_identity(manifest: object) -> None:
    """Require the one reviewed Q30 Thinking SWE-heavy 700K study arm."""
    if (
        not isinstance(manifest, dict)
        or manifest.get("scientific_identity") != _DATASET_IDENTITY
        or type(manifest.get("row_count")) is not int
        or manifest.get("row_count") != 700_000
        or manifest.get("quotas") != _APPROVED_QUOTAS
        or manifest.get("category_counts") != _APPROVED_QUOTAS
    ):
        raise ValueError("exact SWE-heavy 700K dataset identity is invalid")


def _source_git(source_checkout_path: Path, *arguments: str) -> bytes:
    environment = {
        "GIT_CONFIG_GLOBAL": "/dev/null",
        "GIT_CONFIG_NOSYSTEM": "1",
        "GIT_NO_REPLACE_OBJECTS": "1",
        "LC_ALL": "C",
        "PATH": "/usr/bin:/bin",
    }
    git = (
        "/usr/bin/git",
        "-c",
        "core.fsmonitor=false",
        "-c",
        "core.hooksPath=/dev/null",
        "-C",
        str(source_checkout_path),
    )
    try:
        return subprocess.run(
            (*git, *arguments), check=True, capture_output=True, env=environment
        ).stdout
    except (OSError, subprocess.CalledProcessError) as error:
        raise ValueError("source checkout commit is unavailable") from error


def _git_blob_oid(content: bytes) -> str:
    digest = hashlib.sha1(usedforsecurity=False)
    digest.update(f"blob {len(content)}\0".encode())
    digest.update(content)
    return digest.hexdigest()


def _source_regular_file_identity(path: Path) -> tuple[str, str, int, bool]:
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        raise ValueError("source checkout contains an unreadable tracked file") from error
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1:
            raise ValueError("source checkout tracked entries must be single-link regular files")
        git_digest = hashlib.sha1(usedforsecurity=False)
        git_digest.update(f"blob {before.st_size}\0".encode())
        content_digest = hashlib.sha256()
        size = 0
        while block := os.read(descriptor, 1024 * 1024):
            size += len(block)
            git_digest.update(block)
            content_digest.update(block)
        after = os.fstat(descriptor)
        named = path.stat(follow_symlinks=False)

        def identity(status: os.stat_result) -> tuple[int, int, int, int, int, int, int]:
            return (
                status.st_dev,
                status.st_ino,
                status.st_mode,
                status.st_nlink,
                status.st_size,
                status.st_mtime_ns,
                status.st_ctime_ns,
            )

        if identity(before) != identity(after) or identity(after) != identity(named):
            raise ValueError("source checkout tracked file changed while authenticating")
        if size != before.st_size:
            raise ValueError("source checkout tracked file size changed while authenticating")
        return (
            git_digest.hexdigest(),
            content_digest.hexdigest(),
            size,
            bool(before.st_mode & 0o111),
        )
    finally:
        os.close(descriptor)


def _source_checkout_identity(
    source_checkout_path: Path, *, expected_commit: str
) -> tuple[str, str]:
    """Verify exact commit contents without trusting the mutable Git index or status."""
    if source_checkout_path.is_symlink() or not source_checkout_path.is_dir():
        raise ValueError("source checkout must be a no-follow directory")
    head = _source_git(source_checkout_path, "rev-parse", "--verify", "HEAD").decode().strip()
    if head != expected_commit:
        raise ValueError("source checkout commit does not match the contract")
    tree_oid = (
        _source_git(source_checkout_path, "rev-parse", "--verify", f"{expected_commit}^{{tree}}")
        .decode()
        .strip()
    )
    if not _is_lower_hex(tree_oid, 40):
        raise ValueError("source checkout Git tree identity is invalid")

    index_records = _source_git(source_checkout_path, "ls-files", "-v", "-z").split(b"\0")
    if any(record and not record.startswith(b"H ") for record in index_records):
        raise ValueError("source checkout has forbidden Git index flags")

    tracked_files: set[str] = set()
    gitlinks: set[str] = set()
    inventory: list[dict[str, object]] = []
    tree_records = _source_git(
        source_checkout_path, "ls-tree", "-r", "-z", "--full-tree", expected_commit
    ).split(b"\0")
    for record in tree_records:
        if not record:
            continue
        metadata, separator, raw_path = record.partition(b"\t")
        parts = metadata.split(b" ")
        if not separator or len(parts) != 3:
            raise ValueError("source commit tree inventory is malformed")
        mode, object_type, raw_oid = (part.decode() for part in parts)
        relative_text = os.fsdecode(raw_path)
        relative = Path(relative_text)
        if (
            relative.is_absolute()
            or relative_text != relative.as_posix()
            or any(component in {"", ".", "..", ".git"} for component in relative.parts)
        ):
            raise ValueError("source commit tree contains an unsafe path")
        path = source_checkout_path / relative
        if mode == "160000" and object_type == "commit":
            if path.is_symlink() or not path.is_dir() or any(path.iterdir()):
                raise ValueError(
                    "source checkout Git links must be uninitialized empty directories"
                )
            gitlinks.add(relative_text)
            inventory.append(
                {"git_object_oid": raw_oid, "mode": mode, "path": relative_text, "type": "gitlink"}
            )
            continue
        if object_type != "blob" or mode not in {"100644", "100755", "120000"}:
            raise ValueError("source commit tree contains an unsupported entry")
        tracked_files.add(relative_text)
        if mode == "120000":
            before = path.lstat()
            if not stat.S_ISLNK(before.st_mode):
                raise ValueError("source checkout tracked symlink type mismatch")
            content = os.fsencode(os.readlink(path))
            after = path.lstat()
            if (before.st_dev, before.st_ino, before.st_mtime_ns, before.st_ctime_ns) != (
                after.st_dev,
                after.st_ino,
                after.st_mtime_ns,
                after.st_ctime_ns,
            ):
                raise ValueError("source checkout symlink changed while authenticating")
            observed_oid = _git_blob_oid(content)
            content_sha256 = hashlib.sha256(content).hexdigest()
            size = len(content)
            entry_type = "symlink"
        else:
            observed_oid, content_sha256, size, executable = _source_regular_file_identity(path)
            if executable != (mode == "100755"):
                raise ValueError("source checkout executable mode differs from the commit")
            entry_type = "regular"
        if observed_oid != raw_oid:
            raise ValueError("source checkout content differs from the authenticated commit tree")
        inventory.append(
            {
                "git_object_oid": raw_oid,
                "mode": mode,
                "path": relative_text,
                "sha256": content_sha256,
                "size": size,
                "type": entry_type,
            }
        )

    observed_files: set[str] = set()
    for directory, directories, files in os.walk(source_checkout_path, followlinks=False):
        directory_path = Path(directory)
        if directory_path == source_checkout_path:
            directories[:] = [name for name in directories if name != ".git"]
            files = [name for name in files if name != ".git"]
        for name in tuple(directories):
            path = directory_path / name
            if path.is_symlink():
                observed_files.add(path.relative_to(source_checkout_path).as_posix())
                directories.remove(name)
        observed_files.update(
            (directory_path / name).relative_to(source_checkout_path).as_posix() for name in files
        )
    if observed_files != tracked_files:
        raise ValueError(
            "source checkout contains files outside the authenticated commit inventory"
        )
    if any(not (source_checkout_path / gitlink).is_dir() for gitlink in gitlinks):
        raise ValueError("source checkout Git link inventory is incomplete")
    return tree_oid, hashlib.sha256(_canonical_json(inventory)).hexdigest()


def _source_head(source_checkout_path: Path) -> str:
    """Return HEAD only after exact worktree authentication against that commit."""
    head = _source_git(source_checkout_path, "rev-parse", "--verify", "HEAD").decode().strip()
    _source_checkout_identity(source_checkout_path, expected_commit=head)
    return head


@dataclass(frozen=True)
class ExposureSchedule:
    """Exact approved one-pass schedule, including its partial final optimizer step."""

    consumed_examples: int = 700_000
    full_steps: int = 1_368
    nominal_global_batch_size: int = 512
    final_global_batch_size: int = 96
    canary_steps: int = 20

    def validate(self, *, trainer_ranks: int) -> None:
        """Reject any schedule other than the frozen approved scientific identity."""
        values = (
            self.consumed_examples,
            self.full_steps,
            self.nominal_global_batch_size,
            self.final_global_batch_size,
            self.canary_steps,
        )
        if not _is_exact_int(trainer_ranks) or trainer_ranks < 1:
            raise ValueError("trainer ranks must be a positive exact integer")
        if not all(_is_exact_int(value) for value in values):
            raise ValueError("exposure schedule values must be exact integers")
        if values != _APPROVED_SCHEDULE:
            raise ValueError("exposure schedule must equal the approved exact 700K schedule")
        if self.nominal_global_batch_size % trainer_ranks:
            raise ValueError("nominal batch must divide across trainer ranks")
        if self.final_global_batch_size % trainer_ranks:
            raise ValueError("final batch must divide across trainer ranks")

    def examples_for_step(self, step: int) -> int:
        """Return the exact example exposure for one optimizer step."""
        if not _is_exact_int(step) or not 1 <= step <= self.full_steps:
            raise ValueError("step is outside the exposure schedule")
        return self.final_global_batch_size if step == self.full_steps else 512


@dataclass(frozen=True)
class Q30TContinuationContract:
    """Authenticated identity and actual launcher inputs for one Q30 canary."""

    target_model: str
    target_revision: str
    target_tree_sha256: str
    target_checkpoint_path: Path
    method: Method
    block_size: int
    dataset_path: Path
    dataset_sha256: str
    ordered_prompt_uuids_sha256: str
    dataset_manifest_path: Path
    dataset_manifest_file_sha256: str
    dataset_completion_receipt_path: Path
    dataset_completion_receipt_file_sha256: str
    historical_receipt_file_sha256: str
    historical_ordered_prompt_uuids_sha256: str
    dataset_records: int
    tokenizer_receipt_path: Path
    tokenizer_receipt_file_sha256: str
    parent_checkpoint_path: Path
    parent_checkpoint_tree_sha256: str
    parent_receipt_path: Path
    parent_receipt_file_sha256: str
    source_checkout_path: Path
    source_commit: str
    source_archive_path: Path
    source_archive_sha256: str
    parent_source_commit: str
    target_archive_path: Path
    target_archive_sha256: str
    parent_archive_path: Path
    parent_archive_sha256: str
    runtime_archive_path: Path
    runtime_archive_sha256: str
    runtime_path: Path
    runtime_tree_sha256: str
    runtime_image_path: Path
    runtime_image_sha256: str
    training_entrypoint: Path
    training_entrypoint_sha256: str
    method_config_path: Path
    method_config_sha256: str
    output_root: Path
    run_name: str
    schedule: ExposureSchedule = ExposureSchedule()
    nodes: int = 16
    serving_nodes: int = 8
    trainer_nodes: int = 8
    gpus_per_node: int = 4
    trainer_ranks: int = 32
    target_tensor_parallel_size: int = 2
    per_device_train_batch_size: int = 4
    gradient_accumulation_steps: int = 4

    def __post_init__(self) -> None:
        if self.target_model != TARGET_MODEL:
            raise ValueError("contract must use the Q30 Thinking target")
        if self.method not in ("DFlash", "DSpark"):
            raise ValueError("contract method must be DFlash or DSpark")
        if not _is_exact_int(self.block_size) or self.block_size != 8:
            raise ValueError("contract block size must be 8")
        if not _is_exact_int(self.dataset_records) or self.dataset_records != 700_000:
            raise ValueError("contract dataset must contain exact 700K records")
        topology = (
            self.nodes,
            self.serving_nodes,
            self.trainer_nodes,
            self.gpus_per_node,
            self.trainer_ranks,
            self.target_tensor_parallel_size,
            self.per_device_train_batch_size,
            self.gradient_accumulation_steps,
        )
        if not all(_is_exact_int(value) for value in topology) or topology != (
            16,
            8,
            8,
            4,
            32,
            2,
            4,
            4,
        ):
            raise ValueError("contract must preserve the proven 16-node topology")
        paths = (
            self.target_checkpoint_path,
            self.dataset_path,
            self.dataset_manifest_path,
            self.dataset_completion_receipt_path,
            self.tokenizer_receipt_path,
            self.parent_checkpoint_path,
            self.parent_receipt_path,
            self.source_checkout_path,
            self.source_archive_path,
            self.target_archive_path,
            self.parent_archive_path,
            self.runtime_archive_path,
            self.runtime_path,
            self.runtime_image_path,
            self.training_entrypoint,
            self.method_config_path,
            self.output_root,
        )
        if any(not isinstance(path, Path) or not path.is_absolute() for path in paths):
            raise ValueError("contract paths must be absolute pathlib Paths")
        for path in paths:
            _require_canonical_nofollow_path(path, label="contract path")
            _require_pyxis_path(path, label="contract path")
        for digest in (
            self.target_tree_sha256,
            self.dataset_sha256,
            self.ordered_prompt_uuids_sha256,
            self.dataset_manifest_file_sha256,
            self.dataset_completion_receipt_file_sha256,
            self.historical_receipt_file_sha256,
            self.historical_ordered_prompt_uuids_sha256,
            self.tokenizer_receipt_file_sha256,
            self.parent_receipt_file_sha256,
            self.parent_checkpoint_tree_sha256,
            self.source_archive_sha256,
            self.target_archive_sha256,
            self.parent_archive_sha256,
            self.runtime_archive_sha256,
            self.runtime_tree_sha256,
            self.runtime_image_sha256,
            self.training_entrypoint_sha256,
            self.method_config_sha256,
        ):
            if not _is_lower_hex(digest, 64):
                raise ValueError("contract SHA-256 values must be lowercase hexadecimal")
        for commit in (self.target_revision, self.source_commit, self.parent_source_commit):
            if not _is_lower_hex(commit, 40):
                raise ValueError("contract revisions must be lowercase 40-character Git SHAs")
        expected_config = f"{self.method.lower()}.yaml"
        if self.method_config_path.name != expected_config:
            raise ValueError("method config does not match DFlash/DSpark")
        if not self.run_name or any(character.isspace() for character in self.run_name):
            raise ValueError("run name must be a non-empty space-free identity")
        immutable_paths = (
            self.target_checkpoint_path,
            self.dataset_path,
            self.dataset_manifest_path,
            self.dataset_completion_receipt_path,
            self.tokenizer_receipt_path,
            self.parent_checkpoint_path,
            self.parent_receipt_path,
            self.source_checkout_path,
            self.source_archive_path,
            self.target_archive_path,
            self.parent_archive_path,
            self.runtime_archive_path,
            self.runtime_path,
            self.runtime_image_path,
        )
        for immutable_path in immutable_paths:
            if _paths_overlap(self.output_root, immutable_path):
                raise ValueError("output root must be disjoint from every immutable input")
        if any(_paths_overlap(self.output_root.parent, path) for path in immutable_paths):
            raise ValueError("RW output mount source must be disjoint from every immutable input")
        self.schedule.validate(trainer_ranks=self.trainer_ranks)

    def replace(self, **changes: object) -> Self:
        """Return a validated contract with selected fields replaced."""
        return replace(self, **changes)


@dataclass(frozen=True)
class Q30TTrainingLaunch:
    """One authenticated process invocation for the real streaming launcher."""

    command: tuple[str, ...]
    environment: Mapping[str, str]
    working_directory: Path
    container_image: Path
    nodes: int
    gpus_per_node: int


def _authenticate_dataset_bundle(contract: Q30TContinuationContract) -> None:
    """Recompute DATA semantics and require a reviewed full-replay completion receipt."""
    manifest_raw = _require_file_sha256(
        contract.dataset_manifest_path,
        contract.dataset_manifest_file_sha256,
        label="dataset manifest",
        retain=True,
    )
    receipt_raw = _require_file_sha256(
        contract.dataset_completion_receipt_path,
        contract.dataset_completion_receipt_file_sha256,
        label="dataset completion receipt",
        retain=True,
    )
    assert manifest_raw is not None and receipt_raw is not None
    try:
        manifest = json.loads(manifest_raw)
        receipt = json.loads(receipt_raw)
    except json.JSONDecodeError as error:
        raise ValueError("exact 700K dataset metadata is invalid") from error
    _require_dataset_identity(manifest)
    manifest_keys = {
        "schema_version",
        "scientific_identity",
        "row_count",
        "quotas",
        "category_counts",
        "data",
        "ordered_prompt_uuids_sha256",
        "duplicate_uuid_multiplicity",
        "source_occurrences_sha256",
        "selected_token_evidence_sha256",
        "capacity_receipt_file_sha256",
        "capacity",
        "exclusions",
        "historical",
        "held_out",
        "trust",
        "manifest_sha256",
    }
    if (
        not isinstance(manifest, dict)
        or set(manifest) != manifest_keys
        or manifest_raw != _canonical_json(manifest) + b"\n"
        or manifest.get("schema_version") != "ptv2-ptv3-complement-bundle-v1"
        or manifest.get("duplicate_uuid_multiplicity") != {}
    ):
        raise ValueError("exact 700K dataset manifest identity is invalid")
    manifest_body = {name: value for name, value in manifest.items() if name != "manifest_sha256"}
    if (
        manifest.get("manifest_sha256")
        != hashlib.sha256(_canonical_json(manifest_body)).hexdigest()
    ):
        raise ValueError("exact 700K dataset manifest hash does not reconcile")
    historical = manifest.get("historical")
    if (
        not isinstance(historical, dict)
        or historical.get("occurrence_count") != 1_300_000
        or historical.get("file_sha256") != contract.historical_receipt_file_sha256
        or historical.get("ordered_prompt_uuids_sha256")
        != contract.historical_ordered_prompt_uuids_sha256
    ):
        raise ValueError("exact 700K dataset historical non-overlap root does not reconcile")
    descriptor = manifest.get("data")
    if (
        not isinstance(descriptor, dict)
        or set(descriptor) != {"path", "bytes", "sha256"}
        or contract.dataset_manifest_path.parent / str(descriptor.get("path"))
        != contract.dataset_path
        or descriptor.get("sha256") != contract.dataset_sha256
        or descriptor.get("bytes") != contract.dataset_path.stat().st_size
    ):
        raise ValueError("exact 700K dataset file identity does not reconcile")
    counts: dict[str, int] = {}
    category_order: list[str] = []
    observed_uuids: set[str] = set()
    ordered = hashlib.sha256(b"[")
    row_count = 0
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor_fd = os.open(contract.dataset_path, flags)
    except OSError as error:
        raise ValueError("exact 700K dataset is unreadable") from error
    with os.fdopen(descriptor_fd, "rb") as stream:
        before = os.fstat(stream.fileno())
        for raw_line in stream:
            try:
                record = json.loads(raw_line)
            except json.JSONDecodeError as error:
                raise ValueError("exact 700K dataset row JSON is invalid") from error
            if (
                not isinstance(record, dict)
                or raw_line != _canonical_json(record) + b"\n"
                or not isinstance(record.get("category"), str)
                or not _is_lower_hex(record.get("prompt_uuid"), 64)
                or record["prompt_uuid"] in observed_uuids
            ):
                raise ValueError("exact 700K dataset row identity is invalid")
            category = record["category"]
            prompt_uuid = record["prompt_uuid"]
            observed_uuids.add(prompt_uuid)
            counts[category] = counts.get(category, 0) + 1
            if not category_order or category_order[-1] != category:
                category_order.append(category)
            ordered.update((b"" if row_count == 0 else b",") + _canonical_json(prompt_uuid))
            row_count += 1
        after = os.fstat(stream.fileno())
    ordered.update(b"]")
    if (
        (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
        != (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
        or row_count != 700_000
        or counts != _APPROVED_QUOTAS
        or tuple(category_order) != tuple(_APPROVED_QUOTAS)
        or ordered.hexdigest() != contract.ordered_prompt_uuids_sha256
        or manifest.get("ordered_prompt_uuids_sha256") != ordered.hexdigest()
    ):
        raise ValueError("exact 700K dataset semantic evidence does not reconcile")
    receipt_keys = {
        "schema_version",
        "scientific_identity",
        "output_root",
        "row_count",
        "manifest_file_sha256",
        "manifest_sha256",
        "runtime_sha256",
        "source_commit",
        "receipt_sha256",
    }
    receipt_body = (
        {name: value for name, value in receipt.items() if name != "receipt_sha256"}
        if isinstance(receipt, dict)
        else {}
    )
    if (
        not isinstance(receipt, dict)
        or set(receipt) != receipt_keys
        or receipt_raw != _canonical_json(receipt) + b"\n"
        or receipt.get("schema_version") != "ptv2-ptv3-complement-completion-v1"
        or receipt.get("scientific_identity") != _DATASET_IDENTITY
        or receipt.get("output_root") != str(contract.dataset_manifest_path.parent)
        or receipt.get("row_count") != 700_000
        or receipt.get("manifest_file_sha256") != contract.dataset_manifest_file_sha256
        or receipt.get("manifest_sha256") != manifest.get("manifest_sha256")
        or receipt.get("receipt_sha256")
        != hashlib.sha256(_canonical_json(receipt_body)).hexdigest()
    ):
        raise ValueError("exact 700K dataset completion receipt does not reconcile")
    if (
        contract.dataset_completion_receipt_file_sha256
        not in APPROVED_Q30T_DATASET_COMPLETION_RECEIPT_FILE_SHA256S
    ):
        raise ValueError("exact 700K dataset full-replay completion receipt is not reviewed")


def _parent_for_contract(contract: Q30TContinuationContract) -> Q30TParentReceipt:
    parent = load_q30t_parent_receipt(
        contract.parent_receipt_path,
        expected_sha256=contract.parent_receipt_file_sha256,
    )
    expected = (
        contract.method,
        contract.target_model,
        contract.target_revision,
        contract.target_tree_sha256,
        contract.block_size,
        contract.parent_checkpoint_path,
        contract.parent_checkpoint_tree_sha256,
        contract.historical_receipt_file_sha256,
        contract.historical_ordered_prompt_uuids_sha256,
        contract.parent_source_commit,
        contract.runtime_archive_sha256,
        contract.parent_receipt_file_sha256,
    )
    observed = (
        parent.method,
        parent.target_repository,
        parent.target_revision,
        parent.target_tree_sha256,
        parent.block_size,
        parent.checkpoint_path,
        parent.checkpoint_tree_sha256,
        parent.historical_receipt_file_sha256,
        parent.historical_ordered_prompt_uuids_sha256,
        parent.source_commit,
        parent.runtime_sha256,
        parent.receipt_file_sha256,
    )
    if observed != expected or parent.global_step != 25_391:
        raise ValueError("Q30 parent receipt does not match the continuation contract")
    return parent


def _authenticate_contract(contract: Q30TContinuationContract) -> tuple[str, str]:
    if contract.output_root.exists():
        raise ValueError("output root must be fresh and absent before canary launch")
    if not contract.source_checkout_path.is_dir():
        raise ValueError("source checkout path is unavailable")
    source_tree_oid, source_tree_sha256 = _source_checkout_identity(
        contract.source_checkout_path, expected_commit=contract.source_commit
    )
    if APPROVED_Q30T_SOURCE_BY_ARCHIVE_SHA256.get(contract.source_archive_sha256) != (
        contract.source_commit,
        source_tree_oid,
        source_tree_sha256,
    ):
        raise ValueError(
            "source commit/tree inventory has no independently approved source archive"
        )
    if not contract.parent_checkpoint_path.is_dir():
        raise ValueError("parent checkpoint path is unavailable")
    if not contract.runtime_path.is_dir():
        raise ValueError("staged ModelOpt runtime path is unavailable")
    if (
        APPROVED_Q30T_RUNTIME_TREE_SHA256_BY_ARCHIVE_SHA256.get(contract.runtime_archive_sha256)
        != contract.runtime_tree_sha256
    ):
        raise ValueError("runtime archive has no reviewed extracted runtime tree")
    if contract.runtime_image_sha256 not in APPROVED_Q30T_PRODUCTION_RUNTIME_IMAGE_SHA256S:
        raise ValueError(
            "runtime image is not an externally reviewed immutable production runtime image"
        )
    if _tree_sha256(contract.runtime_path) != contract.runtime_tree_sha256:
        raise ValueError("staged runtime tree SHA-256 mismatch")
    try:
        training_entrypoint_relative = contract.training_entrypoint.relative_to(
            contract.source_checkout_path
        )
        method_config_relative = contract.method_config_path.relative_to(
            contract.source_checkout_path
        )
    except ValueError as error:
        raise ValueError(
            "training entrypoint and method config must belong to source checkout"
        ) from error
    if training_entrypoint_relative != Path(
        "tools/launcher/common/eagle3/train_eagle_streaming.sh"
    ) or method_config_relative != Path(
        f"modelopt_recipes/general/speculative_decoding/{contract.method.lower()}.yaml"
    ):
        raise ValueError("training entrypoint and method config must use canonical source paths")
    _require_file_sha256(contract.dataset_path, contract.dataset_sha256, label="dataset")
    _authenticate_dataset_bundle(contract)
    tokenizer = load_q30t_tokenizer_receipt(
        contract.tokenizer_receipt_path,
        expected_sha256=contract.tokenizer_receipt_file_sha256,
    )
    expected_tokenizer = (
        contract.target_model,
        contract.target_revision,
        str(contract.target_checkpoint_path),
        contract.target_tree_sha256,
    )
    observed_tokenizer = tuple(
        tokenizer.get(name)
        for name in ("repository", "revision", "snapshot_path", "snapshot_tree_sha256")
    )
    if observed_tokenizer != expected_tokenizer:
        raise ValueError("Q30 tokenizer receipt does not match the target snapshot")
    _parent_for_contract(contract)
    _require_file_sha256(
        contract.source_archive_path,
        contract.source_archive_sha256,
        label="source archive",
    )
    _require_file_sha256(
        contract.target_archive_path,
        contract.target_archive_sha256,
        label="target archive",
    )
    _require_file_sha256(
        contract.parent_archive_path,
        contract.parent_archive_sha256,
        label="parent archive",
    )
    _require_file_sha256(
        contract.runtime_archive_path,
        contract.runtime_archive_sha256,
        label="runtime archive",
    )
    _require_file_sha256(
        contract.runtime_image_path,
        contract.runtime_image_sha256,
        label="runtime image",
    )
    _require_file_sha256(
        contract.training_entrypoint,
        contract.training_entrypoint_sha256,
        label="training entrypoint",
    )
    _require_file_sha256(
        contract.method_config_path,
        contract.method_config_sha256,
        label="method config",
    )
    return source_tree_oid, source_tree_sha256


def require_q30_parent_pair_for_shared_dataset(
    dflash: Q30TContinuationContract, dspark: Q30TContinuationContract
) -> None:
    """Authenticate both parents before sharing one exclusion-filtered DATA artifact."""
    dflash_parent = _parent_for_contract(dflash)
    dspark_parent = _parent_for_contract(dspark)
    require_matching_historical_lineage(dflash_parent, dspark_parent)


def _run_identity(method: Method, stage: Stage) -> str:
    return f"q30t-{method.lower()}-b8-ptv23-swe-heavy-700k-{stage}"


def _override(name: str, value: object) -> str:
    return f"{name}={value}"


def build_q30_node_local_training_command(
    *,
    method: Method,
    local_root: Path,
    output_root: Path,
    run_name: str,
    stage: Stage | None = None,
) -> tuple[str, ...]:
    """Build the exact post-attestation ``train_eagle_streaming.sh`` interface."""
    if method not in ("DFlash", "DSpark"):
        raise ValueError("node-local method must be DFlash or DSpark")
    for path, label in ((local_root, "node-local root"), (output_root, "output root")):
        _require_canonical_nofollow_path(path, label=label)
    resolved_stage: Stage = stage or ("full" if run_name.endswith("-full") else "canary")
    if run_name != _run_identity(method, resolved_stage) or output_root.name != run_name:
        raise ValueError("node-local run identity must be method/stage bound")
    steps = 20 if resolved_stage == "canary" else 1_368
    source = local_root / "source"
    entrypoint = source / "tools/launcher/common/eagle3/train_eagle_streaming.sh"
    config = source / f"modelopt_recipes/general/speculative_decoding/{method.lower()}.yaml"
    if not entrypoint.is_file() or not config.is_file():
        raise ValueError("node-local trainer entrypoint or method config is absent")
    common = (
        _override("model.model_name_or_path", local_root / "parent"),
        _override("model.tokenizer_name_or_path", local_root / "target"),
        _override("model.use_fake_base_for_offline", "false"),
        _override("model.initialization_policy", "converted-weights-only"),
        _override("data.mode", "streaming"),
        _override("data.data_path", local_root / "DATA.jsonl"),
        _override("data.sample_size", -1),
        _override("training.output_dir", output_root),
        _override("training.max_steps", steps),
        _override("training.save_steps", steps),
        _override("training.save_total_limit", 1),
        _override("training.training_seq_len", 4096),
        _override("training.answer_only_loss", "false"),
        _override("training.num_train_epochs", 1),
        _override("training.dataloader_drop_last", str(resolved_stage == "canary").lower()),
        _override("training.seed", 42),
        _override("training.data_seed", 42),
        _override("training.per_device_train_batch_size", 4),
        _override("training.gradient_accumulation_steps", 4),
        _override("training.dataloader_num_workers", 0),
        _override("training.learning_rate", "0.00006"),
        _override("training.warmup_steps", 55),
        _override("training.lr_scheduler_type", "linear"),
        _override("training.report_to", "wandb"),
        _override("training.run_name", run_name),
        _override("dflash.dflash_block_size", 8),
        _override("dflash.dflash_num_anchors", 512),
        _override("dflash.dflash_mask_token_id", 151669),
        _override("dflash.dflash_architecture_config.num_hidden_layers", 5),
        _override("dflash.dflash_architecture_config.num_attention_heads", 32),
        _override("dflash.dflash_architecture_config.num_key_value_heads", 4),
        _override("dflash.dflash_architecture_config.head_dim", 128),
        _override("dflash.dflash_architecture_config.intermediate_size", 6144),
    )
    if resolved_stage == "full":
        common += (_override("training.exact_exposure_count", 700_000),)
    if method == "DFlash":
        specific = (
            _override("dflash.dflash_loss_objective", "dpace"),
            _override("dflash.dflash_loss_decay_factor", 0),
            _override("dflash.dflash_dpace_alpha", "0.5"),
        )
    else:
        specific = (
            _override("dflash.dflash_loss_objective", "decay"),
            _override("dflash.dflash_loss_decay_factor", 4),
            _override("dflash.dflash_self_logit_distillation", "false"),
            _override("dflash.dflash_architecture_config.projector_type", "dspark"),
        )
    return ("/bin/bash", str(entrypoint), "--config", str(config), *common, *specific)


def build_q30_training_command(
    contract: Q30TContinuationContract, *, stage: Stage
) -> Q30TTrainingLaunch:
    """Build an authenticated canary or exact-full streaming invocation."""
    contract.__post_init__()
    if stage not in ("canary", "full"):
        raise ValueError("stage must be canary or full")
    expected_run_identity = _run_identity(contract.method, stage)
    if (
        contract.run_name != expected_run_identity
        or contract.output_root.name != expected_run_identity
    ):
        raise ValueError("run name and output root must be bound to method and stage")
    source_tree_oid, source_tree_sha256 = _authenticate_contract(contract)
    runner = contract.source_checkout_path / (
        "tools/launcher/common/specdec/run_q30t_ptv23_continuation.sbatch"
    )
    keeper_tool = contract.source_checkout_path / (
        "tools/launcher/common/specdec/ptv23_node_keeper.py"
    )
    if not runner.is_file() or not keeper_tool.is_file():
        raise ValueError("authenticated source checkout lacks the Q30 controller or node keeper")
    _, keeper_tool_sha256 = _stable_file(keeper_tool, label="node keeper")
    training_entrypoint_relative = contract.training_entrypoint.relative_to(
        contract.source_checkout_path
    )
    method_config_relative = contract.method_config_path.relative_to(contract.source_checkout_path)
    control_root = contract.output_root.parent / f".{contract.run_name}.controller"
    canary_receipt = contract.output_root.parent / f"{contract.run_name}.receipt.json"
    for derived_path, label in (
        (control_root, "controller root"),
        (canary_receipt, "canary receipt"),
    ):
        _require_canonical_nofollow_path(derived_path, label=label)
        if derived_path.exists() or derived_path.is_symlink():
            raise ValueError(f"{label} must be fresh and absent")
    environment = MappingProxyType(
        {
            "CANARY_RECEIPT": str(canary_receipt),
            "CONTROL_ROOT": str(control_root),
            "DATASET": str(contract.dataset_path),
            "DATASET_SHA256": contract.dataset_sha256,
            "EXPECTED_EXPOSURE_COUNT": str(
                contract.schedule.consumed_examples
                if stage == "full"
                else contract.schedule.canary_steps * contract.schedule.nominal_global_batch_size
            ),
            "EXPECTED_FINAL_GLOBAL_BATCH_SIZE": str(
                contract.schedule.final_global_batch_size
                if stage == "full"
                else contract.schedule.nominal_global_batch_size
            ),
            "EXPECTED_STEPS": str(
                contract.schedule.full_steps if stage == "full" else contract.schedule.canary_steps
            ),
            "KEEPER_TOOL": str(keeper_tool),
            "KEEPER_TOOL_SHA256": keeper_tool_sha256,
            "METHOD": contract.method,
            "METHOD_CONFIG_RELATIVE": str(method_config_relative),
            "ORDERED_PROMPT_UUIDS_SHA256": contract.ordered_prompt_uuids_sha256,
            "OUTPUT_ROOT": str(contract.output_root),
            "PARENT_ARCHIVE": str(contract.parent_archive_path),
            "PARENT_ARCHIVE_SHA256": contract.parent_archive_sha256,
            "PARENT_TREE_SHA256": contract.parent_checkpoint_tree_sha256,
            "RUNTIME_ARCHIVE": str(contract.runtime_archive_path),
            "RUNTIME_ARCHIVE_SHA256": contract.runtime_archive_sha256,
            "RUNTIME_IMAGE": str(contract.runtime_image_path),
            "RUNTIME_IMAGE_SHA256": contract.runtime_image_sha256,
            "RUNTIME_TREE_SHA256": contract.runtime_tree_sha256,
            "RUN_NAME": contract.run_name,
            "SOURCE_ARCHIVE": str(contract.source_archive_path),
            "SOURCE_ARCHIVE_SHA256": contract.source_archive_sha256,
            "SOURCE_COMMIT": contract.source_commit,
            "SOURCE_TREE_OID": source_tree_oid,
            "SOURCE_TREE_SHA256": source_tree_sha256,
            "STAGE": stage,
            "TARGET_ARCHIVE": str(contract.target_archive_path),
            "TARGET_ARCHIVE_SHA256": contract.target_archive_sha256,
            "TARGET_MODEL": contract.target_model,
            "TARGET_TREE_SHA256": contract.target_tree_sha256,
            "TRAINING_ENTRYPOINT_RELATIVE": str(training_entrypoint_relative),
        }
    )
    return Q30TTrainingLaunch(
        command=("/bin/bash", str(runner)),
        environment=environment,
        working_directory=contract.source_checkout_path,
        container_image=contract.runtime_image_path,
        nodes=contract.nodes,
        gpus_per_node=contract.gpus_per_node,
    )


_CONTRACT_PATH_FIELDS = frozenset(
    {
        "dataset_completion_receipt_path",
        "dataset_manifest_path",
        "dataset_path",
        "method_config_path",
        "output_root",
        "parent_archive_path",
        "parent_checkpoint_path",
        "parent_receipt_path",
        "runtime_archive_path",
        "runtime_image_path",
        "runtime_path",
        "source_archive_path",
        "source_checkout_path",
        "target_archive_path",
        "target_checkpoint_path",
        "tokenizer_receipt_path",
        "training_entrypoint",
    }
)
_CONTRACT_FIELD_NAMES = tuple(field.name for field in fields(Q30TContinuationContract))


def _contract_payload(contract: Q30TContinuationContract) -> dict[str, object]:
    payload: dict[str, object] = {}
    for name in _CONTRACT_FIELD_NAMES:
        value = getattr(contract, name)
        if name in _CONTRACT_PATH_FIELDS:
            payload[name] = str(value)
        elif name == "schedule":
            payload[name] = {
                field.name: getattr(value, field.name) for field in fields(ExposureSchedule)
            }
        else:
            payload[name] = value
    return payload


def _contract_from_payload(payload: object) -> Q30TContinuationContract:
    if not isinstance(payload, dict) or set(payload) != set(_CONTRACT_FIELD_NAMES):
        raise ValueError("Q30 launch descriptor contract fields are invalid")
    values = dict(payload)
    for name in _CONTRACT_PATH_FIELDS:
        value = values[name]
        if type(value) is not str:
            raise ValueError("Q30 launch descriptor contract path is invalid")
        values[name] = Path(value)
    schedule = values["schedule"]
    schedule_fields = tuple(field.name for field in fields(ExposureSchedule))
    if not isinstance(schedule, dict) or set(schedule) != set(schedule_fields):
        raise ValueError("Q30 launch descriptor schedule is invalid")
    values["schedule"] = ExposureSchedule(**schedule)
    try:
        return Q30TContinuationContract(**values)
    except TypeError as error:
        raise ValueError("Q30 launch descriptor contract types are invalid") from error


def write_q30_launch_descriptor(
    contract: Q30TContinuationContract, *, stage: Stage, path: Path
) -> str:
    """Authenticate and exclusively publish the sole canonical scheduler descriptor."""
    build_q30_training_command(contract, stage=stage)
    _require_canonical_nofollow_path(path, label="launch descriptor")
    payload = {
        "contract": _contract_payload(contract),
        "schema_version": "q30t-ptv23-launch-descriptor-v1",
        "stage": stage,
    }
    raw = _canonical_json(payload) + b"\n"
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags, 0o600)
    except OSError as error:
        raise ValueError("Q30 launch descriptor path must be fresh") from error
    try:
        if os.write(descriptor, raw) != len(raw):
            raise ValueError("Q30 launch descriptor write was incomplete")
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    return hashlib.sha256(raw).hexdigest()


def load_q30_launch_descriptor(
    path: Path, *, expected_sha256: str, expected_source_checkout: Path
) -> Q30TTrainingLaunch:
    """Replay one canonical descriptor through the complete continuation authenticator."""
    if not _is_lower_hex(expected_sha256, 64):
        raise ValueError("Q30 launch descriptor SHA-256 is invalid")
    raw, observed_sha256 = _stable_file(path, label="launch descriptor", retain=True)
    assert raw is not None
    if observed_sha256 != expected_sha256:
        raise ValueError("Q30 launch descriptor SHA-256 mismatch")
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as error:
        raise ValueError("Q30 launch descriptor JSON is invalid") from error
    if (
        not isinstance(payload, dict)
        or set(payload) != {"contract", "schema_version", "stage"}
        or payload.get("schema_version") != "q30t-ptv23-launch-descriptor-v1"
        or payload.get("stage") not in ("canary", "full")
        or raw != _canonical_json(payload) + b"\n"
    ):
        raise ValueError("Q30 launch descriptor identity is invalid")
    contract = _contract_from_payload(payload["contract"])
    _require_canonical_nofollow_path(expected_source_checkout, label="expected source checkout")
    if contract.source_checkout_path != expected_source_checkout:
        raise ValueError("Q30 launch descriptor source checkout is substituted")
    return build_q30_training_command(contract, stage=payload["stage"])


def _write_launch_environment(path: Path, launch: Q30TTrainingLaunch) -> None:
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags, 0o600)
    try:
        for name, value in sorted(launch.environment.items()):
            record = name.encode() + b"\0" + value.encode() + b"\0"
            if os.write(descriptor, record) != len(record):
                raise ValueError("Q30 launch environment write was incomplete")
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("replay-launch-descriptor",))
    parser.add_argument("--descriptor", type=Path, required=True)
    parser.add_argument("--sha256", required=True)
    parser.add_argument("--source-checkout", type=Path, required=True)
    parser.add_argument("--environment-output", type=Path, required=True)
    arguments = parser.parse_args(argv)
    launch = load_q30_launch_descriptor(
        arguments.descriptor,
        expected_sha256=arguments.sha256,
        expected_source_checkout=arguments.source_checkout,
    )
    _write_launch_environment(arguments.environment_output, launch)
    return 0


if __name__ == "__main__":
    sys.exit(_main())
