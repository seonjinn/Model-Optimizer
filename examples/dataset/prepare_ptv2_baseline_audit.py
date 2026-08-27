# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Authenticate and adapt the historical PTV2 asset for baseline auditing."""

from __future__ import annotations

import argparse
import hashlib
import os
import re
import stat
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING

from specdec_corpus_contracts import canonical_json, sha256_bytes

if TYPE_CHECKING:
    from collections.abc import Sequence

__all__ = ["AssetEvidence", "PreparationError", "prepare_and_audit"]

SOURCE_REVISION = "5c89e01dd720ae0f4058445ed49c5fb68a03c76e"
SHARD_COUNT = 26
_SHA256 = re.compile(r"[0-9a-f]{64}")


class PreparationError(ValueError):
    """The historical asset or requested audit preparation is invalid."""


@dataclass(frozen=True)
class AssetEvidence:
    """Caller-pinned identities for the historical asset's trust receipts."""

    identity_path: Path
    identity_sha256: str
    completion_path: Path
    completion_sha256: str
    sha256_list_path: Path
    sha256_list_sha256: str


def _require_digest(value: str, label: str) -> None:
    if _SHA256.fullmatch(value) is None:
        raise PreparationError(f"{label} must be lowercase SHA-256")


def _hash_regular_file(path: Path, label: str) -> tuple[int, str]:
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        raise PreparationError(f"{label} is not an available regular file: {path}") from error
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise PreparationError(f"{label} is not a regular file: {path}")
        digest = hashlib.sha256()
        with os.fdopen(descriptor, "rb", closefd=False) as source:
            for chunk in iter(lambda: source.read(8 * 1024 * 1024), b""):
                digest.update(chunk)
        after = os.fstat(descriptor)
        if _stat_identity(before) != _stat_identity(after):
            raise PreparationError(f"{label} changed while it was authenticated: {path}")
        return before.st_size, digest.hexdigest()
    finally:
        os.close(descriptor)


def _stat_identity(value: os.stat_result) -> tuple[int, int, int, int, int]:
    return value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns, value.st_ctime_ns


def _relative_asset_path(path: Path, asset_root: Path, label: str) -> str:
    try:
        return path.relative_to(asset_root).as_posix()
    except ValueError as error:
        raise PreparationError(f"{label} must be inside asset root") from error


def _authenticate_evidence(asset_root: Path, evidence: AssetEvidence) -> dict[str, object]:
    records: dict[str, object] = {"asset_root": str(asset_root)}
    for key, path, expected in (
        ("identity", evidence.identity_path, evidence.identity_sha256),
        ("completion", evidence.completion_path, evidence.completion_sha256),
        ("sha256_list", evidence.sha256_list_path, evidence.sha256_list_sha256),
    ):
        _require_digest(expected, f"{key} digest")
        _, actual = _hash_regular_file(path, key)
        if actual != expected:
            raise PreparationError(f"{key} digest mismatch: {actual} != {expected}")
        records[key] = {
            "path": _relative_asset_path(path, asset_root, key),
            "sha256": actual,
        }
    return records


def _read_sha256_list(path: Path, source_root: Path, expected_sha256: str) -> dict[Path, str]:
    try:
        payload = path.read_bytes()
        lines = payload.decode().splitlines()
    except (OSError, UnicodeDecodeError) as error:
        raise PreparationError("SHA-256 list is not UTF-8 text") from error
    if sha256_bytes(payload) != expected_sha256:
        raise PreparationError("SHA-256 list changed after its evidence check")
    records: dict[Path, str] = {}
    for line_number, line in enumerate(lines, 1):
        if not line:
            continue
        fields = line.split(maxsplit=1)
        if len(fields) != 2 or _SHA256.fullmatch(fields[0]) is None:
            raise PreparationError(f"invalid SHA-256 list line {line_number}")
        name = fields[1].removeprefix("*")
        relative = PurePosixPath(name)
        if relative.is_absolute() or not relative.parts or ".." in relative.parts:
            raise PreparationError(f"unsafe SHA-256 list path on line {line_number}")
        candidate = source_root.joinpath(*relative.parts)
        if candidate in records:
            raise PreparationError(f"duplicate SHA-256 list path: {name}")
        records[candidate] = fields[0]
    return records


def _authenticate_sources(
    asset_root: Path,
    source_root: Path,
    sha256_list_path: Path,
    sha256_list_sha256: str,
) -> list[dict[str, object]]:
    if source_root.is_symlink() or not source_root.is_dir():
        raise PreparationError("source root must be a regular directory")
    _relative_asset_path(source_root, asset_root, "source root")
    source_files = sorted(source_root.glob("*.parquet"))
    if len(source_files) != SHARD_COUNT:
        raise PreparationError(
            f"Parquet shard count mismatch: {len(source_files)} != {SHARD_COUNT}"
        )
    listed = _read_sha256_list(sha256_list_path, source_root, sha256_list_sha256)
    if set(listed) != set(source_files):
        raise PreparationError("SHA-256 list does not exactly match the 26 Parquet shards")
    authenticated: list[dict[str, object]] = []
    for path in source_files:
        size, actual = _hash_regular_file(path, "Parquet shard")
        if actual != listed[path]:
            raise PreparationError(f"Parquet shard digest mismatch: {path.name}")
        authenticated.append({"name": path.name, "bytes": size, "sha256": actual})
    return authenticated


def prepare_and_audit(
    *,
    asset_root: Path,
    source_root: Path,
    evidence: AssetEvidence,
    work_root: Path,
    audit_script: Path,
    output: Path,
) -> None:
    """Authenticate the asset, create its symlink view, and run the canonical audit."""
    asset_root = asset_root.resolve(strict=True)
    source_root = source_root.resolve(strict=True)
    evidence = AssetEvidence(
        evidence.identity_path.resolve(strict=True),
        evidence.identity_sha256,
        evidence.completion_path.resolve(strict=True),
        evidence.completion_sha256,
        evidence.sha256_list_path.resolve(strict=True),
        evidence.sha256_list_sha256,
    )
    asset_evidence = _authenticate_evidence(asset_root, evidence)
    source_files = _authenticate_sources(
        asset_root, source_root, evidence.sha256_list_path, evidence.sha256_list_sha256
    )
    if output.exists():
        raise PreparationError(f"audit output already exists: {output}")
    try:
        work_root.mkdir(mode=0o700)
    except OSError as error:
        raise PreparationError(f"work root must be a new directory: {work_root}") from error
    view_root = work_root / "data"
    view_root.mkdir()
    for record in source_files:
        (view_root / str(record["name"])).symlink_to(source_root / str(record["name"]))
    manifest_path = work_root / "SOURCE_MANIFEST.json"
    manifest_bytes = canonical_json(
        {
            "source_revision": SOURCE_REVISION,
            "files": source_files,
            "asset_evidence": asset_evidence,
        }
    )
    manifest_path.write_bytes(manifest_bytes)
    command = [
        sys.executable,
        str(audit_script),
        "--root",
        str(view_root),
        "--source-manifest",
        str(manifest_path),
        "--source-manifest-sha256",
        sha256_bytes(manifest_bytes),
        "--output",
        str(output),
    ]
    subprocess.run(command, check=True)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--asset-root", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--identity-path", type=Path, required=True)
    parser.add_argument("--identity-sha256", required=True)
    parser.add_argument("--completion-path", type=Path, required=True)
    parser.add_argument("--completion-sha256", required=True)
    parser.add_argument("--sha256-list-path", type=Path, required=True)
    parser.add_argument("--sha256-list-sha256", required=True)
    parser.add_argument("--work-root", type=Path, required=True)
    parser.add_argument("--audit-script", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    prepare_and_audit(
        asset_root=args.asset_root,
        source_root=args.source_root,
        evidence=AssetEvidence(
            args.identity_path,
            args.identity_sha256,
            args.completion_path,
            args.completion_sha256,
            args.sha256_list_path,
            args.sha256_list_sha256,
        ),
        work_root=args.work_root,
        audit_script=args.audit_script,
        output=args.output,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
