# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Immutable artifact and staged-runtime verification for DFlash2 jobs."""

from __future__ import annotations

import argparse
import errno
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
_FLASHMLA_REQUIRED_COMMIT = "a8f794d1251cbfd88a5011445dd5582289c727e4"
_FLASHMLA_SOURCE_INTERFACE = "flash_mla/flash_mla_interface.py"
_FLASHMLA_RUNTIME_INTERFACE = "third_party/flashmla/flash_mla_interface.py"
_FLASHMLA_VLLM_RECIPE = "cmake/external_projects/flashmla.cmake"
_FLASHMLA_BUILD_MANIFEST = "dflash2-flashmla-build-manifest.json"
_FLASHMLA_CONFIGURE_LOG = "dflash2-flashmla-cmake-configure.log"
_FLASHMLA_EXTENSION_STEMS = ("_flashmla_C", "_flashmla_extension_C")
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


def _file_descriptor(path: Path, name: str) -> dict[str, Any]:
    handle = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    digest = hashlib.sha256()
    total = 0
    with os.fdopen(handle, "rb") as stream:
        before = os.fstat(stream.fileno())
        while chunk := stream.read(16 * 1024 * 1024):
            digest.update(chunk)
            total += len(chunk)
        after = os.fstat(stream.fileno())
    if not stat.S_ISREG(before.st_mode) or (
        before.st_dev,
        before.st_ino,
        before.st_size,
        before.st_mtime_ns,
    ) != (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns):
        raise ValueError(f"artifact is not an inode-stable regular file: {path}")
    return {"path": name, "bytes": total, "sha256": digest.hexdigest()}


def _valid_file_descriptor(value: object, expected_path: str | None = None) -> bool:
    return bool(
        isinstance(value, dict)
        and set(value) == {"path", "bytes", "sha256"}
        and isinstance(value.get("path"), str)
        and (expected_path is None or value.get("path") == expected_path)
        and isinstance(value.get("bytes"), int)
        and value["bytes"] >= 0
        and _FULL_SHA256.fullmatch(str(value.get("sha256", "")))
    )


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
    storage_modes: set[str] = set()
    temporary = output_path.with_name(f".{output_path.name}.partial-{os.getpid()}")
    temporary.parent.mkdir(parents=True, exist_ok=True)
    temporary.mkdir()
    try:
        for entry in entries:
            target = entry.resolve(strict=True)
            if not target.is_file() or target.is_symlink():
                raise ValueError(f"dataset source target is not a regular file: {entry.name}")
            raw = _stable_bytes(target)
            materialized = temporary / entry.name
            try:
                os.link(target, materialized, follow_symlinks=False)
                storage = "hardlink"
            except OSError as error:
                if error.errno != errno.EXDEV:
                    raise
                shutil.copyfile(target, materialized, follow_symlinks=False)
                storage = "copy"
            materialized_raw = _stable_bytes(materialized)
            if materialized_raw != raw:
                raise ValueError(f"materialized dataset bytes mismatch: {entry.name}")
            storage_modes.add(storage)
            descriptors.append(
                {
                    "path": entry.name,
                    "source_target": str(target),
                    "bytes": len(raw),
                    "sha256": hashlib.sha256(raw).hexdigest(),
                    "storage": storage,
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
        "storage": next(iter(storage_modes)) if len(storage_modes) == 1 else "mixed",
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


def _verified_commit_file(
    repository: Path,
    relative_path: str,
    expected_commit: str,
    label: str,
) -> Path:
    artifact = repository / relative_path
    status = _git(
        repository,
        "status",
        "--porcelain",
        "--untracked-files=all",
        "--",
        relative_path,
    )
    expected_blob = _git(repository, "rev-parse", f"{expected_commit}:{relative_path}")
    working_blob = _git(repository, "hash-object", str(artifact))
    if (
        status.returncode
        or status.stdout
        or expected_blob.returncode
        or working_blob.returncode
        or not _FULL_SHA.fullmatch(expected_blob.stdout.strip())
        or working_blob.stdout.strip() != expected_blob.stdout.strip()
    ):
        raise ValueError(f"{label} must match exact bytes from {expected_commit}")
    return artifact


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


def _flashmla_extension_descriptors(runtime_package: Path) -> list[dict[str, Any]]:
    extensions: list[Path] = []
    for stem in _FLASHMLA_EXTENSION_STEMS:
        matches = sorted(runtime_package.glob(f"{stem}*.so"))
        if len(matches) != 1:
            raise ValueError("runtime must contain the exact FlashMLA extension pair")
        extensions.extend(matches)
    return [_file_descriptor(item, item.name) for item in extensions]


def _flashmla_configure_evidence(configure_log: Path) -> dict[str, str]:
    configure_text = _stable_bytes(configure_log).decode("utf-8")
    patterns = {
        "cmake_path": r"(?m)^cmake_path=(\S*/runtime/bin/cmake)$",
        "cmake_version": r"(?m)^cmake version (3\.31\.6)$",
        "ninja_path": r"(?m)^ninja_path=(\S*/runtime/bin/ninja)$",
        "ninja_version": r"(?m)^(1\.13\.0)$",
        "cuda_architectures": r"CUDA target architectures:.*(10\.0[af])",
        "flashmla_architectures": r"FlashMLA CUDA architectures:.*(10\.0[af])",
    }
    evidence: dict[str, str] = {}
    for name, pattern in patterns.items():
        match = re.search(pattern, configure_text)
        if match is None:
            raise ValueError("FlashMLA configure evidence is incomplete")
        evidence[name] = match.group(1)
    return evidence


def write_flashmla_configure_preflight(
    path: Path,
    vllm_package_path: Path,
    flashmla_package_path: Path,
    base_runtime_path: Path,
    image_path: Path,
    builder_path: Path,
    configure_log_path: Path,
    vllm_commit: str,
    flashmla_commit: str,
    slurm_job_id: str,
) -> str:
    """Publish configure-only evidence before attempting a source build."""
    if not slurm_job_id.isdigit():
        raise ValueError("FlashMLA configure preflight requires a numeric Slurm job ID")
    vllm_package = _verified_checkout(vllm_package_path, vllm_commit, vllm_commit)
    flashmla_package = _verified_checkout(
        flashmla_package_path,
        flashmla_commit,
        flashmla_commit,
    )
    recipe = _verified_commit_file(
        vllm_package.parent,
        _FLASHMLA_VLLM_RECIPE,
        vllm_commit,
        "vLLM FlashMLA build recipe",
    )
    submodules = _git(flashmla_package.parent, "submodule", "status", "--recursive")
    submodule_lines = [line for line in submodules.stdout.splitlines() if line]
    if submodules.returncode or any(line[0] != " " for line in submodule_lines):
        raise ValueError("FlashMLA configure requires initialized exact submodules")
    configure_log = configure_log_path.resolve(strict=True)
    body: dict[str, Any] = {
        "schema_version": 1,
        "producer": "dflash2-flashmla-configure-preflight-v1",
        "slurm_job_id": slurm_job_id,
        "vllm_commit": vllm_commit,
        "flashmla_commit": flashmla_commit,
        "vllm_build_recipe": _file_descriptor(recipe, _FLASHMLA_VLLM_RECIPE),
        "flashmla_source_interface": _file_descriptor(
            flashmla_package.parent / _FLASHMLA_SOURCE_INTERFACE,
            _FLASHMLA_SOURCE_INTERFACE,
        ),
        "flashmla_submodules": submodule_lines,
        "base_runtime": _file_descriptor(
            base_runtime_path.resolve(strict=True), str(base_runtime_path.resolve(strict=True))
        ),
        "container_image": _file_descriptor(
            image_path.resolve(strict=True), str(image_path.resolve(strict=True))
        ),
        "builder": _file_descriptor(builder_path.resolve(strict=True), str(builder_path.resolve(strict=True))),
        "configure_log": _file_descriptor(configure_log, _FLASHMLA_CONFIGURE_LOG),
        "configure_evidence": _flashmla_configure_evidence(configure_log),
    }
    body["receipt_sha256"] = _sha_json(body)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        stream.write(_canonical(body) + "\n")
    return hashlib.sha256(_stable_bytes(path)).hexdigest()


def write_flashmla_build_manifest(
    path: Path,
    runtime_package_path: Path,
    vllm_package_path: Path,
    flashmla_package_path: Path,
    base_runtime_path: Path,
    image_path: Path,
    builder_path: Path,
    configure_log_path: Path,
    vllm_commit: str,
    flashmla_commit: str,
) -> str:
    """Bind exact source-build inputs to both installed FlashMLA extensions."""
    vllm_package = _verified_checkout(vllm_package_path, vllm_commit, vllm_commit)
    flashmla_package = _verified_checkout(
        flashmla_package_path,
        flashmla_commit,
        flashmla_commit,
    )
    recipe = _verified_commit_file(
        vllm_package.parent,
        _FLASHMLA_VLLM_RECIPE,
        vllm_commit,
        "vLLM FlashMLA build recipe",
    )
    flashmla_repository = flashmla_package.parent
    submodules = _git(flashmla_repository, "submodule", "status", "--recursive")
    submodule_lines = [line for line in submodules.stdout.splitlines() if line]
    if submodules.returncode or any(line[0] != " " for line in submodule_lines):
        raise ValueError("FlashMLA build requires initialized exact submodules")
    runtime_package = runtime_package_path.resolve(strict=True)
    configure_log = configure_log_path.resolve(strict=True)
    configure_evidence = _flashmla_configure_evidence(configure_log)
    body: dict[str, Any] = {
        "schema_version": 1,
        "producer": "dflash2-flashmla-source-build-v1",
        "vllm_commit": vllm_commit,
        "flashmla_commit": flashmla_commit,
        "vllm_build_recipe": _file_descriptor(recipe, _FLASHMLA_VLLM_RECIPE),
        "flashmla_source_interface": _file_descriptor(
            flashmla_package.parent / _FLASHMLA_SOURCE_INTERFACE,
            _FLASHMLA_SOURCE_INTERFACE,
        ),
        "flashmla_submodules": submodule_lines,
        "base_runtime": _file_descriptor(
            base_runtime_path.resolve(strict=True), str(base_runtime_path.resolve(strict=True))
        ),
        "container_image": _file_descriptor(
            image_path.resolve(strict=True), str(image_path.resolve(strict=True))
        ),
        "builder": _file_descriptor(builder_path.resolve(strict=True), str(builder_path.resolve(strict=True))),
        "configure_log": _file_descriptor(configure_log, _FLASHMLA_CONFIGURE_LOG),
        "configure_evidence": configure_evidence,
        "cmake_targets": list(_FLASHMLA_EXTENSION_STEMS),
        "extension_binaries": _flashmla_extension_descriptors(runtime_package),
    }
    body["receipt_sha256"] = _sha_json(body)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        stream.write(_canonical(body) + "\n")
    return hashlib.sha256(_stable_bytes(path)).hexdigest()


def write_vllm_runtime_receipt(
    path: Path,
    package_path: Path,
    expected_commit: str,
    required_ancestor: str,
    *,
    runtime_package_path: Path | None = None,
    flashmla_package_path: Path | None = None,
    flashmla_expected_commit: str = _FLASHMLA_REQUIRED_COMMIT,
    flashmla_build_manifest_path: Path | None = None,
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
        "vllm_commit": expected_commit,
        "required_pr52816_commit": required_ancestor,
        "source_files": source_files,
        "source_files_sha256": _sha_json(source_files),
        "runtime_files": runtime_files,
        "runtime_files_sha256": _sha_json(runtime_files),
    }
    if runtime_package_path is None:
        if flashmla_package_path is not None or flashmla_build_manifest_path is not None:
            raise ValueError("source-only vLLM receipts cannot claim FlashMLA runtime provenance")
        body.update(
            schema_version=2,
            producer="dflash2-vllm-runtime-receipt-v2",
        )
    else:
        if flashmla_package_path is None:
            raise ValueError("staged vLLM runtime requires exact FlashMLA provenance")
        flashmla = _verified_checkout(
            flashmla_package_path,
            flashmla_expected_commit,
            flashmla_expected_commit,
        )
        source_interface = flashmla.parent / _FLASHMLA_SOURCE_INTERFACE
        source_raw = _stable_bytes(source_interface)
        needle = b"flash_mla_cuda = torch.ops._flashmla_C"
        if source_raw.count(needle) != 1:
            raise ValueError("FlashMLA source interface cannot be transformed deterministically")
        generated_raw = source_raw.replace(
            needle,
            b"import vllm._flashmla_C\nflash_mla_cuda = torch.ops._flashmla_C",
        )
        generated_interface = runtime_package / _FLASHMLA_RUNTIME_INTERFACE
        if _stable_bytes(generated_interface) != generated_raw:
            raise ValueError("staged FlashMLA interface does not match exact generated bytes")
        vllm_repository = source_package.parent
        recipe = _verified_commit_file(
            vllm_repository,
            _FLASHMLA_VLLM_RECIPE,
            expected_commit,
            "vLLM FlashMLA build recipe",
        )
        if flashmla_build_manifest_path is None:
            raise ValueError("staged vLLM runtime requires a FlashMLA source-build manifest")
        build_manifest_path = flashmla_build_manifest_path.resolve(strict=True)
        build_manifest_raw = _stable_bytes(build_manifest_path)
        build_manifest = json.loads(build_manifest_raw)
        build_claim = (
            build_manifest.pop("receipt_sha256", None)
            if isinstance(build_manifest, dict)
            else None
        )
        extension_binaries = _flashmla_extension_descriptors(runtime_package)
        if (
            build_claim != _sha_json(build_manifest)
            or build_manifest.get("schema_version") != 1
            or build_manifest.get("producer") != "dflash2-flashmla-source-build-v1"
            or build_manifest.get("vllm_commit") != expected_commit
            or build_manifest.get("flashmla_commit") != flashmla_expected_commit
            or build_manifest.get("vllm_build_recipe")
            != _file_descriptor(recipe, _FLASHMLA_VLLM_RECIPE)
            or build_manifest.get("flashmla_source_interface")
            != _file_descriptor(source_interface, _FLASHMLA_SOURCE_INTERFACE)
            or build_manifest.get("cmake_targets") != list(_FLASHMLA_EXTENSION_STEMS)
            or build_manifest.get("extension_binaries") != extension_binaries
            or any(
                not _valid_file_descriptor(build_manifest.get(name))
                for name in ("base_runtime", "container_image", "builder")
            )
            or not _valid_file_descriptor(
                build_manifest.get("configure_log"), _FLASHMLA_CONFIGURE_LOG
            )
            or not isinstance(build_manifest.get("flashmla_submodules"), list)
        ):
            raise ValueError("FlashMLA source-build manifest identity mismatch")

        body.update(
            schema_version=4,
            producer="dflash2-vllm-runtime-receipt-v4",
            flashmla_commit=flashmla_expected_commit,
            flashmla_source_interface=_file_descriptor(
                source_interface, _FLASHMLA_SOURCE_INTERFACE
            ),
            flashmla_generated_interface=_file_descriptor(
                generated_interface, _FLASHMLA_RUNTIME_INTERFACE
            ),
            flashmla_vllm_build_recipe=_file_descriptor(recipe, _FLASHMLA_VLLM_RECIPE),
            flashmla_build_manifest=_file_descriptor(
                build_manifest_path, _FLASHMLA_BUILD_MANIFEST
            ),
            flashmla_extension_binaries=extension_binaries,
        )
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
    *,
    expected_flashmla_commit: str = _FLASHMLA_REQUIRED_COMMIT,
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
    schema_version = body.get("schema_version") if isinstance(body, dict) else None
    producer = body.get("producer") if isinstance(body, dict) else None
    if (
        claim != _sha_json(body)
        or schema_version not in {2, 4}
        or producer != f"dflash2-vllm-runtime-receipt-v{schema_version}"
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
    if schema_version == 2:
        if files != source_files:
            raise ValueError("staged vLLM runtime requires FlashMLA provenance receipt v4")
    else:
        flashmla_fields = (
            ("flashmla_source_interface", _FLASHMLA_SOURCE_INTERFACE),
            ("flashmla_generated_interface", _FLASHMLA_RUNTIME_INTERFACE),
            ("flashmla_vllm_build_recipe", _FLASHMLA_VLLM_RECIPE),
        )
        if body.get("flashmla_commit") != expected_flashmla_commit:
            raise ValueError("FlashMLA runtime commit mismatch")
        for name, expected_path in flashmla_fields:
            descriptor = body.get(name)
            if (
                not _valid_file_descriptor(descriptor, expected_path)
            ):
                raise ValueError("FlashMLA runtime provenance descriptor mismatch")
        if runtime_by_path.get(_FLASHMLA_RUNTIME_INTERFACE) != body.get(
            "flashmla_generated_interface"
        ):
            raise ValueError("FlashMLA generated interface is not bound to runtime bytes")
        build_descriptor = body.get("flashmla_build_manifest")
        extension_binaries = body.get("flashmla_extension_binaries")
        if (
            not _valid_file_descriptor(build_descriptor, _FLASHMLA_BUILD_MANIFEST)
            or not isinstance(extension_binaries, list)
            or len(extension_binaries) != len(_FLASHMLA_EXTENSION_STEMS)
            or any(not _valid_file_descriptor(item) for item in extension_binaries)
        ):
            raise ValueError("FlashMLA source-build provenance is invalid")
        build_path = receipt_path.parent / _FLASHMLA_BUILD_MANIFEST
        build_raw = _stable_bytes(build_path)
        if (
            len(build_raw) != build_descriptor.get("bytes")
            or hashlib.sha256(build_raw).hexdigest() != build_descriptor.get("sha256")
        ):
            raise ValueError("FlashMLA source-build manifest bytes mismatch")
        build_manifest = json.loads(build_raw)
        build_claim = (
            build_manifest.pop("receipt_sha256", None)
            if isinstance(build_manifest, dict)
            else None
        )
        if (
            build_claim != _sha_json(build_manifest)
            or build_manifest.get("schema_version") != 1
            or build_manifest.get("producer") != "dflash2-flashmla-source-build-v1"
            or build_manifest.get("vllm_commit") != expected_commit
            or build_manifest.get("flashmla_commit") != expected_flashmla_commit
            or build_manifest.get("cmake_targets") != list(_FLASHMLA_EXTENSION_STEMS)
            or build_manifest.get("extension_binaries") != extension_binaries
            or any(
                not _valid_file_descriptor(build_manifest.get(name))
                for name in ("base_runtime", "container_image", "builder")
            )
            or not _valid_file_descriptor(
                build_manifest.get("configure_log"), _FLASHMLA_CONFIGURE_LOG
            )
            or not isinstance(build_manifest.get("flashmla_submodules"), list)
        ):
            raise ValueError("FlashMLA source-build manifest identity mismatch")
        configure_descriptor = build_manifest["configure_log"]
        configure_raw = _stable_bytes(receipt_path.parent / _FLASHMLA_CONFIGURE_LOG)
        configure_text = configure_raw.decode("utf-8")
        if (
            len(configure_raw) != configure_descriptor["bytes"]
            or hashlib.sha256(configure_raw).hexdigest() != configure_descriptor["sha256"]
            or not re.search(r"CUDA target architectures:.*10\.0[af]", configure_text)
            or not re.search(r"FlashMLA CUDA architectures:.*10\.0[af]", configure_text)
        ):
            raise ValueError("FlashMLA CMake configure evidence mismatch")
        for descriptor in extension_binaries:
            if runtime_by_path.get(descriptor.get("path")) != descriptor:
                raise ValueError("FlashMLA extension binary is not bound to runtime bytes")
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
    vllm.add_argument("--flashmla-package", type=Path)
    vllm.add_argument("--flashmla-expected-commit", default=_FLASHMLA_REQUIRED_COMMIT)
    vllm.add_argument("--flashmla-build-manifest", type=Path)
    flashmla_build = subparsers.add_parser("flashmla-build-manifest")
    flashmla_build.add_argument("--output", type=Path, required=True)
    flashmla_build.add_argument("--runtime-package", type=Path, required=True)
    flashmla_build.add_argument("--vllm-package", type=Path, required=True)
    flashmla_build.add_argument("--flashmla-package", type=Path, required=True)
    flashmla_build.add_argument("--base-runtime", type=Path, required=True)
    flashmla_build.add_argument("--image", type=Path, required=True)
    flashmla_build.add_argument("--builder", type=Path, required=True)
    flashmla_build.add_argument("--configure-log", type=Path, required=True)
    flashmla_build.add_argument("--vllm-commit", required=True)
    flashmla_build.add_argument("--flashmla-commit", required=True)
    configure = subparsers.add_parser("flashmla-configure-preflight")
    configure.add_argument("--output", type=Path, required=True)
    configure.add_argument("--vllm-package", type=Path, required=True)
    configure.add_argument("--flashmla-package", type=Path, required=True)
    configure.add_argument("--base-runtime", type=Path, required=True)
    configure.add_argument("--image", type=Path, required=True)
    configure.add_argument("--builder", type=Path, required=True)
    configure.add_argument("--configure-log", type=Path, required=True)
    configure.add_argument("--vllm-commit", required=True)
    configure.add_argument("--flashmla-commit", required=True)
    configure.add_argument("--slurm-job-id", required=True)
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
            flashmla_package_path=args.flashmla_package,
            flashmla_expected_commit=args.flashmla_expected_commit,
            flashmla_build_manifest_path=args.flashmla_build_manifest,
        )
    elif args.command == "flashmla-build-manifest":
        receipt_sha256 = write_flashmla_build_manifest(
            args.output,
            args.runtime_package,
            args.vllm_package,
            args.flashmla_package,
            args.base_runtime,
            args.image,
            args.builder,
            args.configure_log,
            args.vllm_commit,
            args.flashmla_commit,
        )
    elif args.command == "flashmla-configure-preflight":
        receipt_sha256 = write_flashmla_configure_preflight(
            args.output,
            args.vllm_package,
            args.flashmla_package,
            args.base_runtime,
            args.image,
            args.builder,
            args.configure_log,
            args.vllm_commit,
            args.flashmla_commit,
            args.slurm_job_id,
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
