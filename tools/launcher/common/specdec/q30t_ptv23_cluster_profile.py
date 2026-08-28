# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Finalize and replay the external Q30 Ptyche runtime profile."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import stat
import subprocess
import sys
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any, Literal, cast

__all__ = [
    "Q30TRuntimeClusterProfile",
    "finalize_ptyche_runtime_profile",
    "load_ptyche_runtime_profile",
    "main",
]

_SCHEMA_VERSION = "q30t-runtime-cluster-profile-v1"
_CLUSTER = "ptyche"
_ACCOUNT = "coreai_dlalgo_llm"
_PARTITION = "36x2-a01r"
_SCRATCH_ROOT = "/raid/scratch"
_ARCHIVE_PATH = (
    "/lustre/fsw/coreai_dlalgo_llm/users/sna/modelopt-qwen3-drafter-training/"
    "assets/q235-training-prereqs-vllm0271-v1/runtime/"
    "modelopt-vllm-0.27.1-py312-aarch64-symlinks.tar.zst"
)
_ARCHIVE_SHA256 = "4a20aee61f290c48bed22a84b4a0ae0cbdc54e3e3910854d253188c8854f5dc9"
_IMAGE_PATH = (
    "/lustre/fsw/coreai_dlalgo_llm/users/sna/modelopt-qwen3-drafter-training/"
    "assets/q235-training-prereqs-vllm0271-v1/image/"
    "vllm_openai_v0271_aarch64_20260813_2688476.sqsh"
)
_IMAGE_SHA256 = "e7be53f2754097c88f7c801da92f6d94794ec4d78d9df937fcd315a6994297f0"
_SOURCE_ROOT = Path("/home")
_EXPECTED_DURABLE_RECEIPT_ROOT = Path(
    "/lustre/fsw/coreai_dlalgo_llm/users/sna/modelopt-qwen3-drafter-training/"
    "receipts/q30t-swe-heavy-700k-v1/runtime-qualification"
)
_TOOL_PATHS = {
    "archive_producer_sha256": Path(
        "tools/launcher/common/specdec/q30t_runtime_archive_receipt.py"
    ),
    "keeper_tool_sha256": Path("tools/launcher/common/specdec/ptv23_node_keeper.py"),
    "attestation_tool_sha256": Path("tools/launcher/common/specdec/ptv23_runtime_attestation.py"),
    "probe_runner_sha256": Path("tools/launcher/common/specdec/probe_ptv23_pyxis_keeper.sbatch"),
}
_READ_BLOCK_BYTES = 1024 * 1024
_MAX_PROFILE_BYTES = 256 * 1024
_MAX_TOOL_BYTES = 16 * 1024 * 1024


def _no_op_hook() -> None:
    return None


def _no_op_path_hook(_: Path) -> None:
    return None


_POST_SOURCE_AUTHENTICATION_HOOK = _no_op_hook
_POST_TOOL_OPEN_HOOK = _no_op_path_hook


class _ProfileOutputExistsError(FileExistsError):
    pass


def _canonical_json(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def _is_lower_hex(value: object, length: int) -> bool:
    return (
        type(value) is str
        and len(value) == length
        and all(character in "0123456789abcdef" for character in value)
    )


def _full_identity(status: os.stat_result) -> tuple[int, int, int, int, int, int, int]:
    return (
        status.st_dev,
        status.st_ino,
        status.st_mode,
        status.st_nlink,
        status.st_size,
        status.st_mtime_ns,
        status.st_ctime_ns,
    )


def _directory_identity(status: os.stat_result) -> tuple[int, int]:
    return status.st_dev, status.st_ino


def _require_string(payload: dict[str, object], field: str) -> str:
    value = payload[field]
    if type(value) is not str:
        raise ValueError(f"profile {field.replace('_', ' ')} must be a string")
    return value


def _require_exact_int(payload: dict[str, object], field: str, expected: int) -> int:
    value = payload[field]
    if type(value) is not int or value != expected:
        raise ValueError(f"profile {field.replace('_', ' ')} must equal {expected}")
    return value


def _require_exact_string(payload: dict[str, object], field: str, expected: str) -> str:
    value = _require_string(payload, field)
    if value != expected:
        raise ValueError(f"profile {field.replace('_', ' ')} is not the canonical value")
    return value


def _require_digest(payload: dict[str, object], field: str, length: int = 64) -> str:
    value = payload[field]
    if not _is_lower_hex(value, length):
        raise ValueError(f"profile {field.replace('_', ' ')} is not lowercase hexadecimal")
    return cast("str", value)


def _require_canonical_absolute(path: Path, label: str) -> None:
    text = str(path)
    if (
        not path.is_absolute()
        or "\x00" in text
        or any(part in {"", ".", ".."} for part in path.parts[1:])
        or os.path.normpath(text) != text
    ):
        raise ValueError(f"{label} must be a canonical absolute path")


def _require_source_path(path: Path) -> None:
    _require_canonical_absolute(path, "source checkout")
    try:
        relative = path.relative_to(_SOURCE_ROOT)
    except ValueError as error:
        raise ValueError("source checkout must be under /home") from error
    if not relative.parts:
        raise ValueError("source checkout must be below /home")


def _require_receipt_root(path: Path) -> None:
    _require_canonical_absolute(path, "durable receipt root")
    if path != _EXPECTED_DURABLE_RECEIPT_ROOT:
        raise ValueError("durable receipt root is not the exact /lustre qualification root")


def _require_profile_output(path: Path, durable_receipt_root: Path) -> None:
    _require_canonical_absolute(path, "profile output")
    if path.parent != durable_receipt_root or path.name in {"", ".", ".."}:
        raise ValueError("profile output must be one file directly under the durable receipt root")


@dataclass(frozen=True)
class Q30TRuntimeClusterProfile:
    """Exact external cluster and runtime identity used by Q30 qualification."""

    schema_version: Literal["q30t-runtime-cluster-profile-v1"]
    cluster: Literal["ptyche"]
    account: Literal["coreai_dlalgo_llm"]
    partition: Literal["36x2-a01r"]
    source_checkout: str
    source_commit: str
    durable_receipt_root: str
    scratch_root: Literal["/raid/scratch"]
    archive_path: str
    archive_sha256: str
    image_path: str
    image_sha256: str
    archive_producer_sha256: str
    keeper_tool_sha256: str
    attestation_tool_sha256: str
    probe_runner_sha256: str
    probe_node_counts: tuple[Literal[1], Literal[2]]
    training_nodes: Literal[16]
    training_segment: Literal[16]
    gpus_per_node: Literal[4]
    one_node_walltime: Literal["00:20:00"]
    two_node_walltime: Literal["00:30:00"]
    receipt_sha256: str

    def canonical_body_bytes(self) -> bytes:
        """Return canonical bytes for the self-hashed profile body."""
        payload = asdict(self)
        del payload["receipt_sha256"]
        return _canonical_json(payload)

    def canonical_bytes(self) -> bytes:
        """Return canonical JSON bytes without the publication newline."""
        return _canonical_json(asdict(self))

    def verify_self_hash(self) -> None:
        """Reject a profile whose receipt hash does not bind its complete body."""
        observed = hashlib.sha256(self.canonical_body_bytes()).hexdigest()
        if self.receipt_sha256 != observed:
            raise ValueError("profile self-hash mismatch")

    @classmethod
    def from_dict(cls, value: object) -> Q30TRuntimeClusterProfile:
        """Validate an untrusted JSON object against the exact profile schema."""
        if type(value) is not dict:
            raise ValueError("profile JSON must be an object")
        payload = cast("dict[str, object]", value)
        expected_keys = {field.name for field in fields(cls)}
        if set(payload) != expected_keys or any(type(key) is not str for key in payload):
            raise ValueError("profile keys do not match the exact schema")

        schema_version = _require_exact_string(payload, "schema_version", _SCHEMA_VERSION)
        cluster = _require_exact_string(payload, "cluster", _CLUSTER)
        account = _require_exact_string(payload, "account", _ACCOUNT)
        partition = _require_exact_string(payload, "partition", _PARTITION)
        source_checkout = _require_string(payload, "source_checkout")
        _require_source_path(Path(source_checkout))
        source_commit = _require_digest(payload, "source_commit", 40)
        durable_receipt_root = _require_exact_string(
            payload, "durable_receipt_root", str(_EXPECTED_DURABLE_RECEIPT_ROOT)
        )
        _require_receipt_root(Path(durable_receipt_root))
        scratch_root = _require_exact_string(payload, "scratch_root", _SCRATCH_ROOT)
        archive_path = _require_exact_string(payload, "archive_path", _ARCHIVE_PATH)
        archive_sha256 = _require_exact_string(payload, "archive_sha256", _ARCHIVE_SHA256)
        image_path = _require_exact_string(payload, "image_path", _IMAGE_PATH)
        image_sha256 = _require_exact_string(payload, "image_sha256", _IMAGE_SHA256)
        archive_producer_sha256 = _require_digest(payload, "archive_producer_sha256")
        keeper_tool_sha256 = _require_digest(payload, "keeper_tool_sha256")
        attestation_tool_sha256 = _require_digest(payload, "attestation_tool_sha256")
        probe_runner_sha256 = _require_digest(payload, "probe_runner_sha256")
        node_counts = payload["probe_node_counts"]
        if (
            type(node_counts) is not list
            or len(node_counts) != 2
            or type(node_counts[0]) is not int
            or type(node_counts[1]) is not int
            or node_counts != [1, 2]
        ):
            raise ValueError("profile probe node counts must equal [1,2]")
        training_nodes = _require_exact_int(payload, "training_nodes", 16)
        training_segment = _require_exact_int(payload, "training_segment", 16)
        gpus_per_node = _require_exact_int(payload, "gpus_per_node", 4)
        one_node_walltime = _require_exact_string(payload, "one_node_walltime", "00:20:00")
        two_node_walltime = _require_exact_string(payload, "two_node_walltime", "00:30:00")
        receipt_sha256 = _require_digest(payload, "receipt_sha256")
        profile = cls(
            schema_version=cast("Literal['q30t-runtime-cluster-profile-v1']", schema_version),
            cluster=cast("Literal['ptyche']", cluster),
            account=cast("Literal['coreai_dlalgo_llm']", account),
            partition=cast("Literal['36x2-a01r']", partition),
            source_checkout=source_checkout,
            source_commit=source_commit,
            durable_receipt_root=durable_receipt_root,
            scratch_root=cast("Literal['/raid/scratch']", scratch_root),
            archive_path=archive_path,
            archive_sha256=archive_sha256,
            image_path=image_path,
            image_sha256=image_sha256,
            archive_producer_sha256=archive_producer_sha256,
            keeper_tool_sha256=keeper_tool_sha256,
            attestation_tool_sha256=attestation_tool_sha256,
            probe_runner_sha256=probe_runner_sha256,
            probe_node_counts=(1, 2),
            training_nodes=cast("Literal[16]", training_nodes),
            training_segment=cast("Literal[16]", training_segment),
            gpus_per_node=cast("Literal[4]", gpus_per_node),
            one_node_walltime=cast("Literal['00:20:00']", one_node_walltime),
            two_node_walltime=cast("Literal['00:30:00']", two_node_walltime),
            receipt_sha256=receipt_sha256,
        )
        profile.verify_self_hash()
        return profile


def _run_git(source_checkout: Path, held_source_fd: int, *arguments: str) -> bytes:
    _assert_held_directory(source_checkout, held_source_fd, "source checkout")
    command = (
        "/usr/bin/git",
        "-c",
        "core.fsmonitor=false",
        "-c",
        "core.hooksPath=/dev/null",
        "-C",
        str(source_checkout),
        *arguments,
    )
    environment = {
        "GIT_CONFIG_GLOBAL": "/dev/null",
        "GIT_CONFIG_NOSYSTEM": "1",
        "GIT_NO_REPLACE_OBJECTS": "1",
        "LC_ALL": "C",
        "PATH": "/usr/bin:/bin",
    }
    try:
        result = subprocess.run(command, check=True, capture_output=True, env=environment)
    except (OSError, subprocess.CalledProcessError) as error:
        raise ValueError("source checkout Git provenance is unavailable") from error
    _assert_held_directory(source_checkout, held_source_fd, "source checkout")
    return result.stdout


def _assert_held_directory(path: Path, descriptor: int, label: str) -> None:
    held = os.fstat(descriptor)
    try:
        named = os.stat(path, follow_symlinks=False)
    except OSError as error:
        raise ValueError(f"{label} pathname rebound") from error
    if (
        not stat.S_ISDIR(held.st_mode)
        or not stat.S_ISDIR(named.st_mode)
        or _directory_identity(held) != _directory_identity(named)
    ):
        raise ValueError(f"{label} pathname rebound")


def _open_held_directory(path: Path, label: str) -> int:
    flags = (
        os.O_RDONLY
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_NONBLOCK", 0)
    )
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        raise ValueError(f"{label} must be an existing no-follow directory") from error
    try:
        _assert_held_directory(path, descriptor, label)
    except BaseException:
        os.close(descriptor)
        raise
    return descriptor


def _authenticate_git_state(source_checkout: Path, held_source_fd: int, source_commit: str) -> None:
    if not _is_lower_hex(source_commit, 40):
        raise ValueError("source commit must be a lowercase 40-character Git identity")
    top_level = (
        _run_git(source_checkout, held_source_fd, "rev-parse", "--show-toplevel").decode().strip()
    )
    if top_level != str(source_checkout):
        raise ValueError("source checkout is not the exact Git top level")
    head = (
        _run_git(source_checkout, held_source_fd, "rev-parse", "--verify", "HEAD").decode().strip()
    )
    if head != source_commit:
        raise ValueError("source checkout HEAD does not match source commit")
    replace_refs = _run_git(
        source_checkout,
        held_source_fd,
        "for-each-ref",
        "--format=%(refname)",
        "refs/replace",
    )
    if replace_refs.strip():
        raise ValueError("source checkout contains forbidden Git replace refs")
    status = _run_git(
        source_checkout,
        held_source_fd,
        "status",
        "--porcelain=v1",
        "--untracked-files=all",
    )
    if status:
        raise ValueError("source checkout must be clean")
    upstream = (
        _run_git(source_checkout, held_source_fd, "rev-parse", "--verify", "@{upstream}")
        .decode()
        .strip()
    )
    if upstream != source_commit:
        raise ValueError("source checkout HEAD must equal its live pushed upstream")


def _stable_tool_identity(path: Path) -> tuple[str, str, bool]:
    parent_flags = (
        os.O_RDONLY
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_NONBLOCK", 0)
    )
    file_flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0)
    try:
        parent_fd = os.open(path.parent, parent_flags)
    except OSError as error:
        raise ValueError(
            f"required profile tool directory is unavailable: {path.parent}"
        ) from error
    descriptor: int | None = None
    try:
        parent_before = os.fstat(parent_fd)
        absolute_parent_before = os.stat(path.parent, follow_symlinks=False)
        named_before = os.stat(path.name, dir_fd=parent_fd, follow_symlinks=False)
        descriptor = os.open(path.name, file_flags, dir_fd=parent_fd)
        opened_before = os.fstat(descriptor)
        if (
            _directory_identity(parent_before) != _directory_identity(absolute_parent_before)
            or _full_identity(named_before) != _full_identity(opened_before)
            or not stat.S_ISREG(opened_before.st_mode)
            or opened_before.st_nlink != 1
            or opened_before.st_size > _MAX_TOOL_BYTES
        ):
            raise ValueError(f"required profile tool is not a stable single-link file: {path}")
        _POST_TOOL_OPEN_HOOK(path)
        content_digest = hashlib.sha256()
        git_digest = hashlib.sha1(usedforsecurity=False)
        git_digest.update(f"blob {opened_before.st_size}\0".encode())
        size = 0
        while block := os.read(descriptor, _READ_BLOCK_BYTES):
            size += len(block)
            if size > _MAX_TOOL_BYTES:
                raise ValueError(f"required profile tool exceeds size limit: {path}")
            content_digest.update(block)
            git_digest.update(block)
        opened_after = os.fstat(descriptor)
        named_after = os.stat(path.name, dir_fd=parent_fd, follow_symlinks=False)
        parent_after = os.fstat(parent_fd)
        absolute_parent_after = os.stat(path.parent, follow_symlinks=False)
        if (
            size != opened_before.st_size
            or _full_identity(opened_before) != _full_identity(opened_after)
            or _full_identity(opened_after) != _full_identity(named_after)
            or _directory_identity(parent_before) != _directory_identity(parent_after)
            or _directory_identity(parent_after) != _directory_identity(absolute_parent_after)
        ):
            raise ValueError(f"required profile tool changed while hashing: {path}")
        return (
            content_digest.hexdigest(),
            git_digest.hexdigest(),
            bool(opened_before.st_mode & 0o111),
        )
    except OSError as error:
        raise ValueError(f"required profile tool is unavailable: {path}") from error
    finally:
        if descriptor is not None:
            os.close(descriptor)
        os.close(parent_fd)


def _authenticated_tool_sha256(
    source_checkout: Path,
    held_source_fd: int,
    source_commit: str,
    relative_path: Path,
) -> str:
    raw_entry = _run_git(
        source_checkout,
        held_source_fd,
        "ls-tree",
        "-z",
        source_commit,
        "--",
        relative_path.as_posix(),
    )
    records = [record for record in raw_entry.split(b"\0") if record]
    if len(records) != 1:
        raise ValueError(
            f"required profile tool is absent from authenticated commit: {relative_path}"
        )
    metadata, separator, raw_path = records[0].partition(b"\t")
    parts = metadata.split(b" ")
    if (
        not separator
        or raw_path != os.fsencode(relative_path.as_posix())
        or len(parts) != 3
        or parts[0] not in {b"100644", b"100755"}
        or parts[1] != b"blob"
        or not _is_lower_hex(parts[2].decode(errors="replace"), 40)
    ):
        raise ValueError(f"required profile tool has an invalid Git identity: {relative_path}")
    content_sha256, observed_oid, executable = _stable_tool_identity(
        source_checkout / relative_path
    )
    expected_oid = parts[2].decode()
    expected_executable = parts[0] == b"100755"
    if observed_oid != expected_oid or executable != expected_executable:
        raise ValueError(
            f"required profile tool differs from the authenticated commit: {relative_path}"
        )
    _assert_held_directory(source_checkout, held_source_fd, "source checkout")
    return content_sha256


def _stable_single_link_bytes_at(
    parent_fd: int, name: str, maximum_bytes: int
) -> tuple[bytes, str]:
    file_flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0)
    descriptor: int | None = None
    try:
        parent_before = os.fstat(parent_fd)
        named_before = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
        descriptor = os.open(name, file_flags, dir_fd=parent_fd)
        opened_before = os.fstat(descriptor)
        if (
            _full_identity(named_before) != _full_identity(opened_before)
            or not stat.S_ISREG(opened_before.st_mode)
            or opened_before.st_nlink != 1
            or opened_before.st_size > maximum_bytes
        ):
            raise ValueError("profile must be a bounded single-link regular file")
        retained = bytearray()
        digest = hashlib.sha256()
        while block := os.read(descriptor, _READ_BLOCK_BYTES):
            retained.extend(block)
            digest.update(block)
            if len(retained) > maximum_bytes:
                raise ValueError("profile is too large")
        opened_after = os.fstat(descriptor)
        named_after = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
        parent_after = os.fstat(parent_fd)
        if (
            len(retained) != opened_before.st_size
            or _full_identity(opened_before) != _full_identity(opened_after)
            or _full_identity(opened_after) != _full_identity(named_after)
            or _directory_identity(parent_before) != _directory_identity(parent_after)
        ):
            raise ValueError("profile changed while reading")
        return bytes(retained), digest.hexdigest()
    except OSError as error:
        raise ValueError("profile is unreadable") from error
    finally:
        if descriptor is not None:
            os.close(descriptor)


def _stable_single_link_bytes(path: Path, maximum_bytes: int) -> tuple[bytes, str]:
    parent_fd = _open_held_directory(path.parent, "profile directory")
    try:
        result = _stable_single_link_bytes_at(parent_fd, path.name, maximum_bytes)
        _assert_held_directory(path.parent, parent_fd, "profile directory")
        return result
    finally:
        os.close(parent_fd)


def _parse_profile_bytes(raw: bytes) -> Q30TRuntimeClusterProfile:
    def reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("profile contains duplicate keys")
            result[key] = value
        return result

    try:
        payload = json.loads(raw, object_pairs_hook=reject_duplicate_keys)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("profile is not valid UTF-8 JSON") from error
    profile = Q30TRuntimeClusterProfile.from_dict(payload)
    if raw != profile.canonical_bytes() + b"\n":
        raise ValueError("profile bytes are not canonical newline-terminated JSON")
    return profile


def load_ptyche_runtime_profile(path: Path, expected_file_sha256: str) -> Q30TRuntimeClusterProfile:
    """Load one exact, canonical, self-hashed Ptyche runtime profile."""
    _require_canonical_absolute(path, "profile path")
    if not _is_lower_hex(expected_file_sha256, 64):
        raise ValueError("expected profile whole-file SHA-256 is invalid")
    raw, observed_sha256 = _stable_single_link_bytes(path, _MAX_PROFILE_BYTES)
    if observed_sha256 != expected_file_sha256:
        raise ValueError("profile whole-file SHA-256 mismatch")
    profile = _parse_profile_bytes(raw)
    _require_profile_output(path, Path(profile.durable_receipt_root))
    return profile


def _new_profile(
    *,
    source_checkout: Path,
    source_commit: str,
    durable_receipt_root: Path,
    tool_sha256s: dict[str, str],
) -> Q30TRuntimeClusterProfile:
    body: dict[str, object] = {
        "schema_version": _SCHEMA_VERSION,
        "cluster": _CLUSTER,
        "account": _ACCOUNT,
        "partition": _PARTITION,
        "source_checkout": str(source_checkout),
        "source_commit": source_commit,
        "durable_receipt_root": str(durable_receipt_root),
        "scratch_root": _SCRATCH_ROOT,
        "archive_path": _ARCHIVE_PATH,
        "archive_sha256": _ARCHIVE_SHA256,
        "image_path": _IMAGE_PATH,
        "image_sha256": _IMAGE_SHA256,
        **tool_sha256s,
        "probe_node_counts": [1, 2],
        "training_nodes": 16,
        "training_segment": 16,
        "gpus_per_node": 4,
        "one_node_walltime": "00:20:00",
        "two_node_walltime": "00:30:00",
    }
    body["receipt_sha256"] = hashlib.sha256(_canonical_json(body)).hexdigest()
    return Q30TRuntimeClusterProfile.from_dict(body)


def _adopt_exact_profile_at(
    receipt_root: Path,
    receipt_root_fd: int,
    output_name: str,
    expected_bytes: bytes,
) -> Q30TRuntimeClusterProfile:
    _assert_held_directory(receipt_root, receipt_root_fd, "durable receipt root")
    try:
        observed_bytes, _ = _stable_single_link_bytes_at(
            receipt_root_fd, output_name, _MAX_PROFILE_BYTES
        )
        observed = _parse_profile_bytes(observed_bytes)
    except ValueError as error:
        raise ValueError("foreign profile output already exists") from error
    if observed_bytes != expected_bytes:
        raise ValueError("foreign profile output already exists")
    _assert_held_directory(receipt_root, receipt_root_fd, "durable receipt root")
    return observed


def _write_all(descriptor: int, payload: bytes) -> None:
    view = memoryview(payload)
    while view:
        written = os.write(descriptor, view)
        if written < 1:
            raise OSError("profile publication write stalled")
        view = view[written:]


def _publish_profile_bytes_at(
    *,
    receipt_root: Path,
    receipt_root_fd: int,
    output_name: str,
    payload: bytes,
    job_id: str,
) -> None:
    _assert_held_directory(receipt_root, receipt_root_fd, "durable receipt root")
    try:
        os.stat(output_name, dir_fd=receipt_root_fd, follow_symlinks=False)
    except FileNotFoundError:
        pass
    else:
        raise _ProfileOutputExistsError(output_name)

    partial_name = f".{output_name}.partial-{job_id}"
    flags = (
        os.O_WRONLY
        | os.O_CREAT
        | os.O_EXCL
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_NONBLOCK", 0)
    )
    descriptor = os.open(partial_name, flags, 0o600, dir_fd=receipt_root_fd)
    try:
        partial_before = os.fstat(descriptor)
        _write_all(descriptor, payload)
        os.fsync(descriptor)
        partial_after = os.fstat(descriptor)
        if (
            _full_identity(partial_before)[:4] != _full_identity(partial_after)[:4]
            or partial_after.st_size != len(payload)
            or partial_after.st_nlink != 1
        ):
            raise ValueError("profile partial changed while writing")
    finally:
        os.close(descriptor)

    reread, reread_sha256 = _stable_single_link_bytes_at(
        receipt_root_fd, partial_name, _MAX_PROFILE_BYTES
    )
    if reread != payload or reread_sha256 != hashlib.sha256(payload).hexdigest():
        raise ValueError("profile partial differs from canonical payload")
    _assert_held_directory(receipt_root, receipt_root_fd, "durable receipt root")
    try:
        os.link(
            partial_name,
            output_name,
            src_dir_fd=receipt_root_fd,
            dst_dir_fd=receipt_root_fd,
            follow_symlinks=False,
        )
    except FileExistsError as error:
        raise _ProfileOutputExistsError(output_name) from error
    os.unlink(partial_name, dir_fd=receipt_root_fd)
    installed, installed_sha256 = _stable_single_link_bytes_at(
        receipt_root_fd, output_name, _MAX_PROFILE_BYTES
    )
    if installed != payload or installed_sha256 != hashlib.sha256(payload).hexdigest():
        raise ValueError("published profile differs from canonical payload")
    os.fsync(receipt_root_fd)
    _assert_held_directory(receipt_root, receipt_root_fd, "durable receipt root")


def finalize_ptyche_runtime_profile(
    *,
    source_checkout: Path,
    source_commit: str,
    durable_receipt_root: Path,
    output_path: Path,
    job_id: str,
) -> Q30TRuntimeClusterProfile:
    """Authenticate one pushed checkout and publish its exact external profile."""
    _require_source_path(source_checkout)
    _require_receipt_root(durable_receipt_root)
    _require_profile_output(output_path, durable_receipt_root)
    if re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", job_id) is None:
        raise ValueError("profile publication job ID is unsafe")

    source_fd = _open_held_directory(source_checkout, "source checkout")
    receipt_root_fd: int | None = None
    try:
        receipt_root_fd = _open_held_directory(durable_receipt_root, "durable receipt root")
        _authenticate_git_state(source_checkout, source_fd, source_commit)
        _POST_SOURCE_AUTHENTICATION_HOOK()
        _assert_held_directory(source_checkout, source_fd, "source checkout")
        tool_sha256s = {
            field: _authenticated_tool_sha256(
                source_checkout, source_fd, source_commit, relative_path
            )
            for field, relative_path in _TOOL_PATHS.items()
        }
        _authenticate_git_state(source_checkout, source_fd, source_commit)
        _assert_held_directory(durable_receipt_root, receipt_root_fd, "durable receipt root")
        profile = _new_profile(
            source_checkout=source_checkout,
            source_commit=source_commit,
            durable_receipt_root=durable_receipt_root,
            tool_sha256s=tool_sha256s,
        )
        payload = profile.canonical_bytes() + b"\n"
        try:
            os.stat(output_path.name, dir_fd=receipt_root_fd, follow_symlinks=False)
        except FileNotFoundError:
            pass
        else:
            return _adopt_exact_profile_at(
                durable_receipt_root, receipt_root_fd, output_path.name, payload
            )
        try:
            _publish_profile_bytes_at(
                receipt_root=durable_receipt_root,
                receipt_root_fd=receipt_root_fd,
                output_name=output_path.name,
                payload=payload,
                job_id=job_id,
            )
        except _ProfileOutputExistsError:
            return _adopt_exact_profile_at(
                durable_receipt_root, receipt_root_fd, output_path.name, payload
            )
        published = _adopt_exact_profile_at(
            durable_receipt_root, receipt_root_fd, output_path.name, payload
        )
        if published != profile:
            raise ValueError("published profile differs from authenticated profile")
        return published
    finally:
        if receipt_root_fd is not None:
            os.close(receipt_root_fd)
        os.close(source_fd)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    finalize = commands.add_parser("finalize")
    finalize.add_argument("--source-checkout", type=Path, required=True)
    finalize.add_argument("--source-commit", required=True)
    finalize.add_argument("--durable-receipt-root", type=Path, required=True)
    finalize.add_argument("--output", type=Path, required=True)
    finalize.add_argument("--job-id", required=True)
    verify = commands.add_parser("verify")
    verify.add_argument("--profile", type=Path, required=True)
    verify.add_argument("--profile-sha256", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the profile finalizer or strict replay command."""
    if sys.version_info < (3, 12):
        raise RuntimeError("Q30 runtime qualification requires Python 3.12+")
    arguments = _parser().parse_args(argv)
    if arguments.command == "finalize":
        finalize_ptyche_runtime_profile(
            source_checkout=arguments.source_checkout,
            source_commit=arguments.source_commit,
            durable_receipt_root=arguments.durable_receipt_root,
            output_path=arguments.output,
            job_id=arguments.job_id,
        )
        _, whole_file_sha256 = _stable_single_link_bytes(arguments.output, _MAX_PROFILE_BYTES)
    else:
        load_ptyche_runtime_profile(arguments.profile, arguments.profile_sha256)
        whole_file_sha256 = arguments.profile_sha256
    print(whole_file_sha256)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
