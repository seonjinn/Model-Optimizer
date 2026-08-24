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
_DFLASH2_FEATURE_PATHS = (
    "modelopt/torch/export/plugins/hf_spec_export.py",
    "modelopt/torch/speculative/config.py",
    "modelopt/torch/speculative/dflash/conversion.py",
    "modelopt/torch/speculative/plugins/__init__.py",
    "modelopt/torch/speculative/plugins/hf_dflash.py",
    "modelopt/torch/speculative/plugins/hf_dflash2.py",
    "modelopt/torch/speculative/plugins/modeling_dflash.py",
    "modelopt/torch/speculative/plugins/modeling_dflash2.py",
    "modelopt_recipes/general/speculative_decoding/dflash2.yaml",
)


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


def materialize_dataset_view(source_path: Path, output_path: Path, receipt_path: Path) -> str:
    """Replace an ordered symlink dataset view with exact same-filesystem hardlinks."""
    source = source_path.resolve(strict=True)
    if not source.is_dir() or source.is_symlink():
        raise ValueError("dataset source view must be a directory")
    if output_path.exists() or receipt_path.exists():
        raise FileExistsError("dataset materialization outputs already exist")
    entries = sorted(source_path.iterdir())
    if not entries or any(
        not entry.is_symlink() or entry.suffix not in {".jsonl", ".parquet"} for entry in entries
    ):
        raise ValueError("dataset source view must contain only JSONL or Parquet symlinks")

    descriptors: list[dict[str, Any]] = []
    temporary = output_path.with_name(f".{output_path.name}.partial-{os.getpid()}")
    temporary.parent.mkdir(parents=True, exist_ok=True)
    temporary.mkdir()
    try:
        for entry in entries:
            target = entry.resolve(strict=True)
            if not target.is_file() or target.is_symlink():
                raise ValueError(f"dataset source target is not a regular file: {entry.name}")
            raw = _stable_bytes(target)
            os.link(target, temporary / entry.name, follow_symlinks=False)
            descriptors.append(
                {
                    "path": entry.name,
                    "source_target": str(target),
                    "bytes": len(raw),
                    "sha256": hashlib.sha256(raw).hexdigest(),
                }
            )
        logical_sha256 = artifact_tree_sha256(temporary)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        temporary.rename(output_path)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise

    body: dict[str, Any] = {
        "schema_version": 1,
        "producer": "dflash2-dataset-hardlink-materialization-v1",
        "source_path": str(source),
        "output_path": str(output_path.resolve(strict=True)),
        "storage": "hardlink",
        "ordered_files": descriptors,
        "source_tree_sha256": logical_sha256,
        "output_tree_sha256": artifact_tree_sha256(output_path),
    }
    body["receipt_sha256"] = _sha_json(body)
    receipt_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with receipt_path.open("x", encoding="utf-8") as stream:
            stream.write(_canonical(body) + "\n")
    except BaseException:
        shutil.rmtree(output_path, ignore_errors=True)
        raise
    return hashlib.sha256(_stable_bytes(receipt_path)).hexdigest()


def _dataset_layout(
    path: Path, admitted_prefix_count: int
) -> tuple[int, str, list[dict[str, Any]]]:
    root = path.resolve(strict=True)
    candidates = (
        [root] if root.is_file() else sorted(item for item in root.glob("*") if item.is_file())
    )
    jsonl_files = [item for item in candidates if item.suffix == ".jsonl"]
    parquet_files = [item for item in candidates if item.suffix == ".parquet"]
    if jsonl_files and parquet_files:
        raise ValueError("Nemotron dataset cannot mix JSONL and Parquet shards")
    data_files = jsonl_files or parquet_files
    if not data_files:
        raise ValueError("Nemotron dataset has no supported JSONL or Parquet shards")
    total = 0
    remaining = admitted_prefix_count
    sources: list[dict[str, Any]] = []
    for item in data_files:
        if item.suffix == ".jsonl":
            with item.open("rb") as stream:
                physical = sum(bool(line.strip()) for line in stream)
        else:
            try:
                import pyarrow.parquet as parquet
            except ImportError as error:
                raise ValueError("PyArrow is required to attest Parquet row counts") from error
            physical = parquet.ParquetFile(item).metadata.num_rows
        admitted = min(remaining, physical)
        remaining -= admitted
        total += physical
        sources.append(
            {
                "path": item.name if root.is_file() else item.relative_to(root).as_posix(),
                "sha256": hashlib.sha256(_stable_bytes(item)).hexdigest(),
                "physical_occurrences": physical,
                "admitted_occurrences": admitted,
            }
        )
    if remaining:
        raise ValueError("Nemotron dataset occurrence count is smaller than the admitted prefix")
    return total, _sha_json(sources), sources


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
    dataset_layout = None
    if kind == "dataset":
        dataset_layout = _dataset_layout(artifact_path, occurrence_count)
    schema_version = 2 if kind == "dataset" else 1
    body: dict[str, Any] = {
        "schema_version": schema_version,
        "producer": f"dflash2-artifact-receipt-v{schema_version}",
        "kind": kind,
        "artifact_path": str(artifact_path.resolve(strict=True)),
        "artifact_sha256": artifact_tree_sha256(artifact_path),
    }
    if dataset_layout is None:
        body["occurrence_count"] = None
    else:
        physical, admitted_order_sha256, ordered_sources = dataset_layout
        body.update(
            physical_occurrence_count=physical,
            admitted_prefix_count=occurrence_count,
            admitted_order_policy="huggingface-streaming-take-prefix-v1",
            admitted_order_sha256=admitted_order_sha256,
            ordered_sources=ordered_sources,
        )
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
    verify_dataset_rows: bool = True,
) -> None:
    """Rehash an artifact and require its exact externally pinned receipt."""
    raw = _stable_bytes(receipt_path)
    if hashlib.sha256(raw).hexdigest() != expected_receipt_sha256:
        raise ValueError("artifact receipt bytes mismatch")
    body = json.loads(raw)
    claim = body.pop("receipt_sha256", None) if isinstance(body, dict) else None
    expected: dict[str, Any] = {
        "schema_version": 2 if kind == "dataset" else 1,
        "producer": f"dflash2-artifact-receipt-v{2 if kind == 'dataset' else 1}",
        "kind": kind,
        "artifact_path": str(artifact_path.resolve(strict=True)),
        "artifact_sha256": expected_artifact_sha256,
    }
    if kind == "dataset":
        if verify_dataset_rows:
            physical, admitted_order_sha256, ordered_sources = _dataset_layout(
                artifact_path, 1_300_000
            )
        else:
            ordered_sources = body.get("ordered_sources")
            if not isinstance(ordered_sources, list):
                raise ValueError("dataset receipt ordered sources are invalid")
            root = artifact_path.resolve(strict=True)
            candidates = (
                [root]
                if root.is_file()
                else sorted(item for item in root.glob("*") if item.is_file())
            )
            data_files = [item for item in candidates if item.suffix in {".jsonl", ".parquet"}]
            if len(data_files) != len(ordered_sources):
                raise ValueError("dataset receipt source file set mismatch")
            remaining = 1_300_000
            physical = 0
            for item, descriptor in zip(data_files, ordered_sources, strict=True):
                if not isinstance(descriptor, dict) or set(descriptor) != {
                    "path",
                    "sha256",
                    "physical_occurrences",
                    "admitted_occurrences",
                }:
                    raise ValueError("dataset receipt source descriptor is invalid")
                expected_path = item.name if root.is_file() else item.relative_to(root).as_posix()
                item_physical = descriptor["physical_occurrences"]
                item_admitted = descriptor["admitted_occurrences"]
                if (
                    descriptor["path"] != expected_path
                    or descriptor["sha256"] != hashlib.sha256(_stable_bytes(item)).hexdigest()
                    or not isinstance(item_physical, int)
                    or isinstance(item_physical, bool)
                    or item_physical < 1
                    or not isinstance(item_admitted, int)
                    or isinstance(item_admitted, bool)
                    or item_admitted != min(remaining, item_physical)
                ):
                    raise ValueError("dataset receipt source identity mismatch")
                remaining -= item_admitted
                physical += item_physical
            if remaining:
                raise ValueError("dataset receipt admitted prefix is incomplete")
            admitted_order_sha256 = _sha_json(ordered_sources)
        expected.update(
            physical_occurrence_count=physical,
            admitted_prefix_count=1_300_000,
            admitted_order_policy="huggingface-streaming-take-prefix-v1",
            admitted_order_sha256=admitted_order_sha256,
            ordered_sources=ordered_sources,
        )
    else:
        expected["occurrence_count"] = None
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


def validate_dflash2_source_checkout(
    source_path: Path,
    expected_head: str,
    feature_base: str,
) -> str:
    """Bind a clean launcher HEAD to unchanged DFlash2 feature blobs from its ancestor."""
    if not _FULL_SHA.fullmatch(expected_head) or not _FULL_SHA.fullmatch(feature_base):
        raise ValueError("DFlash2 source commits must be exact lowercase SHAs")
    source = source_path.resolve(strict=True)
    top_level = _git(source, "rev-parse", "--show-toplevel")
    if top_level.returncode or Path(top_level.stdout.strip()).resolve() != source:
        raise ValueError("DFlash2 source path must be the Git top-level")
    if _git(source, "rev-parse", "HEAD").stdout.strip() != expected_head:
        raise ValueError("DFlash2 launcher checkout HEAD mismatch")
    status = _git(source, "status", "--porcelain", "--untracked-files=all")
    if status.returncode or status.stdout:
        raise ValueError("DFlash2 launcher checkout must be clean")
    if _git(source, "merge-base", "--is-ancestor", feature_base, expected_head).returncode:
        raise ValueError("DFlash2 feature base is not an ancestor of launcher HEAD")
    if _git(
        source, "diff", "--quiet", feature_base, expected_head, "--", *_DFLASH2_FEATURE_PATHS
    ).returncode:
        raise ValueError("DFlash2 feature files differ from the pinned feature base")
    descriptors = []
    for relative in _DFLASH2_FEATURE_PATHS:
        blob = _git(source, "rev-parse", f"{expected_head}:{relative}")
        if blob.returncode or not _FULL_SHA.fullmatch(blob.stdout.strip()):
            raise ValueError(f"DFlash2 feature file is missing: {relative}")
        descriptors.append({"path": relative, "blob": blob.stdout.strip()})
    return _sha_json(descriptors)


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
    *,
    runtime_package_path: Path | None = None,
) -> str:
    """Bind clean PR source plus the complete installed runtime file set."""
    source_package = _verified_checkout(package_path, expected_commit, required_ancestor)
    runtime_package = (
        source_package
        if runtime_package_path is None
        else runtime_package_path.resolve(strict=True)
    )
    source_files = _runtime_files(source_package)
    runtime_files = _runtime_files(runtime_package)
    runtime_by_path = {descriptor["path"]: descriptor for descriptor in runtime_files}
    if any(runtime_by_path.get(descriptor["path"]) != descriptor for descriptor in source_files):
        raise ValueError("vLLM runtime does not preserve exact tracked source bytes")
    body: dict[str, Any] = {
        "schema_version": 2,
        "producer": "dflash2-vllm-runtime-receipt-v2",
        "vllm_commit": expected_commit,
        "required_pr52816_commit": required_ancestor,
        "source_files": source_files,
        "source_files_sha256": _sha_json(source_files),
        "runtime_files": runtime_files,
        "runtime_files_sha256": _sha_json(runtime_files),
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
    source_files = body.get("source_files") if isinstance(body, dict) else None
    if (
        claim != _sha_json(body)
        or body.get("schema_version") != 2
        or body.get("producer") != "dflash2-vllm-runtime-receipt-v2"
        or body.get("vllm_commit") != expected_commit
        or body.get("required_pr52816_commit") != required_ancestor
        or not isinstance(files, list)
        or not isinstance(source_files, list)
        or body.get("source_files_sha256") != _sha_json(source_files)
        or body.get("runtime_files_sha256") != _sha_json(files)
    ):
        raise ValueError("vLLM runtime receipt identity mismatch")
    package = package_path.resolve(strict=True)
    for descriptor in [*source_files, *files]:
        if not isinstance(descriptor, dict) or set(descriptor) != {"path", "bytes", "sha256"}:
            raise ValueError("vLLM runtime file descriptor is invalid")
        relative = Path(str(descriptor["path"]))
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("vLLM runtime file path is invalid")
    runtime_by_path = {descriptor["path"]: descriptor for descriptor in files}
    if any(runtime_by_path.get(descriptor["path"]) != descriptor for descriptor in source_files):
        raise ValueError("vLLM runtime source provenance mismatch")
    declared_paths = set(runtime_by_path)
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
    vllm.add_argument("--runtime-package", type=Path)
    vllm.add_argument("--output", type=Path, required=True)
    vllm.add_argument("--expected-commit", required=True)
    vllm.add_argument("--required-ancestor", required=True)
    artifact = subparsers.add_parser("artifact-receipt")
    artifact.add_argument("--artifact", type=Path, required=True)
    artifact.add_argument("--output", type=Path, required=True)
    artifact.add_argument("--kind", choices=("target", "dataset"), required=True)
    artifact.add_argument("--occurrence-count", type=int)
    materialize = subparsers.add_parser("materialize-dataset")
    materialize.add_argument("--source", type=Path, required=True)
    materialize.add_argument("--output", type=Path, required=True)
    materialize.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "vllm-receipt":
        receipt_sha256 = write_vllm_runtime_receipt(
            args.output,
            args.package,
            args.expected_commit,
            args.required_ancestor,
            runtime_package_path=args.runtime_package,
        )
    elif args.command == "artifact-receipt":
        receipt_sha256 = write_artifact_receipt(
            args.output,
            args.artifact,
            kind=args.kind,
            occurrence_count=args.occurrence_count,
        )
    else:
        receipt_sha256 = materialize_dataset_view(args.source, args.output, args.receipt)
    print(receipt_sha256)


if __name__ == "__main__":
    main()
