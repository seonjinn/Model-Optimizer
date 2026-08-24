# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Immutable artifact and staged-runtime verification for DFlash2 jobs."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import stat
import subprocess  # nosec B404 - Git is invoked with fixed argv during receipt creation.
from pathlib import Path
from typing import Any

_FULL_SHA = re.compile(r"^[0-9a-f]{40}$")
_FULL_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_GIT = shutil.which("git")


def _stable_bytes(path: Path) -> bytes:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    with os.fdopen(descriptor, "rb") as stream:
        before = os.fstat(stream.fileno())
        raw = stream.read()
        after = os.fstat(stream.fileno())
    if not stat.S_ISREG(before.st_mode) or (
        before.st_dev,
        before.st_ino,
        before.st_size,
        before.st_mtime_ns,
    ) != (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns):
        raise ValueError(f"artifact is not an inode-stable regular file: {path}")
    return raw


def _canonical(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _sha_json(value: object) -> str:
    return hashlib.sha256(_canonical(value).encode()).hexdigest()


def artifact_tree_sha256(path: Path) -> str:
    """Hash a regular file or a symlink-free directory tree deterministically."""
    root = path.resolve(strict=True)
    if root.is_file() and not root.is_symlink():
        return hashlib.sha256(_stable_bytes(root)).hexdigest()
    if not root.is_dir() or root.is_symlink():
        raise ValueError("artifact must be a regular file or directory")
    files = sorted(item for item in root.rglob("*") if item.is_file())
    if not files or any(item.is_symlink() for item in root.rglob("*")):
        raise ValueError("artifact tree must be non-empty and symlink-free")
    digest = hashlib.sha256()
    for item in files:
        digest.update(item.relative_to(root).as_posix().encode())
        digest.update(b"\0")
        digest.update(bytes.fromhex(hashlib.sha256(_stable_bytes(item)).hexdigest()))
    return digest.hexdigest()


def _dataset_occurrence_count(path: Path) -> int:
    root = path.resolve(strict=True)
    candidates = (
        [root] if root.is_file() else sorted(item for item in root.rglob("*") if item.is_file())
    )
    data_files = [item for item in candidates if item.suffix in {".jsonl", ".parquet"}]
    if not data_files:
        raise ValueError("Nemotron dataset has no supported JSONL or Parquet shards")
    total = 0
    for item in data_files:
        if item.suffix == ".jsonl":
            with item.open("rb") as stream:
                total += sum(bool(line.strip()) for line in stream)
            continue
        try:
            import pyarrow.parquet as parquet
        except ImportError as error:
            raise ValueError("PyArrow is required to attest Parquet row counts") from error
        total += parquet.ParquetFile(item).metadata.num_rows
    return total


def write_artifact_receipt(
    path: Path,
    artifact_path: Path,
    *,
    kind: str,
    occurrence_count: int | None = None,
) -> str:
    """Publish a self-hashed receipt for exact target or dataset bytes."""
    if kind not in {"target", "dataset"}:
        raise ValueError("artifact receipt kind is invalid")
    if kind == "dataset" and occurrence_count != 1_300_000:
        raise ValueError("Nemotron dataset receipt must attest exactly 1,300,000 occurrences")
    if kind == "target" and occurrence_count is not None:
        raise ValueError("target receipt cannot contain an occurrence count")
    if kind == "dataset" and _dataset_occurrence_count(artifact_path) != occurrence_count:
        raise ValueError("Nemotron dataset occurrence count does not match the claimed 1,300,000")
    body: dict[str, Any] = {
        "schema_version": 1,
        "producer": "dflash2-artifact-receipt-v1",
        "kind": kind,
        "artifact_path": str(artifact_path.resolve(strict=True)),
        "artifact_sha256": artifact_tree_sha256(artifact_path),
        "occurrence_count": occurrence_count,
    }
    body["receipt_sha256"] = _sha_json(body)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        stream.write(_canonical(body) + "\n")
    return hashlib.sha256(_stable_bytes(path)).hexdigest()


def validate_artifact_receipt(
    receipt_path: Path,
    *,
    expected_receipt_sha256: str,
    artifact_path: Path,
    expected_artifact_sha256: str,
    kind: str,
) -> None:
    """Rehash an artifact and require its exact externally pinned receipt."""
    raw = _stable_bytes(receipt_path)
    if hashlib.sha256(raw).hexdigest() != expected_receipt_sha256:
        raise ValueError("artifact receipt bytes mismatch")
    body = json.loads(raw)
    claim = body.pop("receipt_sha256", None) if isinstance(body, dict) else None
    expected = {
        "schema_version": 1,
        "producer": "dflash2-artifact-receipt-v1",
        "kind": kind,
        "artifact_path": str(artifact_path.resolve(strict=True)),
        "artifact_sha256": expected_artifact_sha256,
        "occurrence_count": 1_300_000 if kind == "dataset" else None,
    }
    if claim != _sha_json(body) or body != expected:
        raise ValueError("artifact receipt identity mismatch")
    if artifact_tree_sha256(artifact_path) != expected_artifact_sha256:
        raise ValueError("artifact bytes mismatch")


def _git(repo: Path, *arguments: str) -> subprocess.CompletedProcess[str]:
    if _GIT is None:
        raise ValueError("Git is required to build the vLLM runtime receipt")
    return subprocess.run(  # nosec B603 - arguments are passed directly without a shell.
        [_GIT, "-C", str(repo), *arguments],
        check=False,
        capture_output=True,
        text=True,
        timeout=10,
    )


def _verified_checkout(package_path: Path, expected_commit: str, required_ancestor: str) -> Path:
    if not _FULL_SHA.fullmatch(expected_commit) or not _FULL_SHA.fullmatch(required_ancestor):
        raise ValueError("vLLM commits must be exact lowercase 40-character SHAs")
    package = package_path.resolve(strict=True)
    repo = next(
        (candidate for candidate in (package, *package.parents) if (candidate / ".git").exists()),
        None,
    )
    if repo is None:
        raise ValueError("vLLM receipt must be built from a Git checkout")
    head = _git(repo, "rev-parse", "HEAD")
    if head.returncode != 0 or head.stdout.strip() != expected_commit:
        raise ValueError(f"vLLM runtime does not match expected commit {expected_commit}")
    if _git(repo, "merge-base", "--is-ancestor", required_ancestor, expected_commit).returncode:
        raise ValueError(
            f"vLLM runtime does not contain required DFlash2 commit {required_ancestor}"
        )
    top_level = _git(repo, "rev-parse", "--show-toplevel")
    if top_level.returncode:
        raise ValueError("vLLM Git top-level is unavailable")
    repository = Path(top_level.stdout.strip()).resolve(strict=True)
    relative_package = package.relative_to(repository)
    status = _git(
        repository,
        "status",
        "--porcelain",
        "--untracked-files=all",
        "--",
        str(relative_package),
    )
    tracked = _git(repository, "ls-files", "-z", "--", str(relative_package))
    actual_files = {
        (relative_package / descriptor["path"]).as_posix() for descriptor in _runtime_files(package)
    }
    tracked_files = {item for item in tracked.stdout.split("\0") if item}
    if status.returncode or status.stdout or tracked.returncode or actual_files != tracked_files:
        raise ValueError("vLLM receipt requires clean tracked checkout bytes")
    return package


def _runtime_files(package: Path) -> list[dict[str, Any]]:
    files = sorted(
        item
        for item in package.rglob("*")
        if item.is_file() and "__pycache__" not in item.parts and item.suffix != ".pyc"
    )
    if not files or any(item.is_symlink() for item in files):
        raise ValueError("vLLM package runtime files are invalid")
    return [
        {
            "path": item.relative_to(package).as_posix(),
            "bytes": item.stat().st_size,
            "sha256": hashlib.sha256(_stable_bytes(item)).hexdigest(),
        }
        for item in files
    ]


def write_vllm_runtime_receipt(
    path: Path,
    package_path: Path,
    expected_commit: str,
    required_ancestor: str,
) -> str:
    """Bind Git ancestry once, before a relocatable runtime archive is built."""
    package = _verified_checkout(package_path, expected_commit, required_ancestor)
    files = _runtime_files(package)
    body: dict[str, Any] = {
        "schema_version": 1,
        "producer": "dflash2-vllm-runtime-receipt-v1",
        "vllm_commit": expected_commit,
        "required_pr52816_commit": required_ancestor,
        "runtime_files": files,
        "runtime_files_sha256": _sha_json(files),
    }
    body["receipt_sha256"] = _sha_json(body)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        stream.write(_canonical(body) + "\n")
    return hashlib.sha256(_stable_bytes(path)).hexdigest()


def verify_vllm_runtime(
    package_path: Path,
    receipt_path: Path,
    expected_receipt_sha256: str,
    expected_commit: str,
    required_ancestor: str,
) -> str:
    """Verify staged vLLM bytes and build-time PR 52816 provenance without Git."""
    if not _FULL_SHA256.fullmatch(expected_receipt_sha256):
        raise ValueError("vLLM runtime receipt SHA-256 must be exact")
    if not receipt_path.is_file():
        raise ValueError("vLLM runtime receipt is missing")
    raw = _stable_bytes(receipt_path)
    if hashlib.sha256(raw).hexdigest() != expected_receipt_sha256:
        raise ValueError("vLLM runtime receipt bytes mismatch")
    body = json.loads(raw)
    claim = body.pop("receipt_sha256", None) if isinstance(body, dict) else None
    files = body.get("runtime_files") if isinstance(body, dict) else None
    if (
        claim != _sha_json(body)
        or body.get("schema_version") != 1
        or body.get("producer") != "dflash2-vllm-runtime-receipt-v1"
        or body.get("vllm_commit") != expected_commit
        or body.get("required_pr52816_commit") != required_ancestor
        or not isinstance(files, list)
        or body.get("runtime_files_sha256") != _sha_json(files)
    ):
        raise ValueError("vLLM runtime receipt identity mismatch")
    package = package_path.resolve(strict=True)
    for descriptor in files:
        if not isinstance(descriptor, dict) or set(descriptor) != {"path", "bytes", "sha256"}:
            raise ValueError("vLLM runtime file descriptor is invalid")
        relative = Path(str(descriptor["path"]))
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("vLLM runtime file path is invalid")
    declared_paths = {descriptor["path"] for descriptor in files}
    runtime_paths = {descriptor["path"] for descriptor in _runtime_files(package)}
    if declared_paths != runtime_paths:
        raise ValueError("vLLM runtime file set mismatch")
    for descriptor in files:
        relative = Path(str(descriptor["path"]))
        artifact = package / relative
        raw_file = _stable_bytes(artifact)
        if (
            len(raw_file) != descriptor["bytes"]
            or hashlib.sha256(raw_file).hexdigest() != descriptor["sha256"]
        ):
            raise ValueError(f"vLLM runtime file bytes mismatch: {relative}")
    return expected_commit


def main() -> None:
    """Build immutable DFlash2 provenance receipts before archive staging."""
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    vllm = subparsers.add_parser("vllm-receipt")
    vllm.add_argument("--package", type=Path, required=True)
    vllm.add_argument("--output", type=Path, required=True)
    vllm.add_argument("--expected-commit", required=True)
    vllm.add_argument("--required-ancestor", required=True)
    artifact = subparsers.add_parser("artifact-receipt")
    artifact.add_argument("--artifact", type=Path, required=True)
    artifact.add_argument("--output", type=Path, required=True)
    artifact.add_argument("--kind", choices=("target", "dataset"), required=True)
    artifact.add_argument("--occurrence-count", type=int)
    args = parser.parse_args()
    if args.command == "vllm-receipt":
        receipt_sha256 = write_vllm_runtime_receipt(
            args.output,
            args.package,
            args.expected_commit,
            args.required_ancestor,
        )
    else:
        receipt_sha256 = write_artifact_receipt(
            args.output,
            args.artifact,
            kind=args.kind,
            occurrence_count=args.occurrence_count,
        )
    print(receipt_sha256)


if __name__ == "__main__":
    main()
