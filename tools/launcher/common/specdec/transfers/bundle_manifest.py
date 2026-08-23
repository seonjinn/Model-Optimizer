# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Create and verify content-addressed SpecDec transfer bundles."""

from __future__ import annotations

import argparse
import ctypes
import errno
import hashlib
import json
import os
import platform
import shutil
import stat
import subprocess  # nosec B404 - fixed argv invokes the resolved git executable only
import sys
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any
from uuid import uuid4

__all__ = [
    "BundleError",
    "FileRecord",
    "atomic_install",
    "create_manifest",
    "load_manifest",
    "main",
    "verify_completion",
    "verify_launcher",
    "verify_tree",
    "write_completion",
]

_MANIFEST_SCHEMA = "modelopt-specdec-manifest-v2"
_COMPLETION_SCHEMA = "modelopt-specdec-completion-v1"
_SHA256_BUFFER_SIZE = 8 * 1024 * 1024


class BundleError(Exception):
    """A transfer bundle failed validation or installation."""


@dataclass(frozen=True)
class FileRecord:
    """One immutable regular file in a bundle."""

    path: str
    sha256: str
    size: int

    def as_dict(self) -> dict[str, str | int]:
        """Return the canonical JSON representation."""
        return {"path": self.path, "sha256": self.sha256, "size": self.size}


def main(argv: list[str] | None = None) -> int:
    """Run one bundle-manifest operation."""
    parser = _parser()
    arguments = parser.parse_args(argv)
    try:
        if arguments.command == "create":
            digest = create_manifest(
                arguments.source,
                arguments.durable_root,
                arguments.output,
                arguments.artifact_source_commit,
            )
            print(digest)
        elif arguments.command == "verify-manifest":
            load_manifest(
                arguments.manifest,
                arguments.expected_sha256,
                arguments.expected_artifact_source_commit,
            )
        elif arguments.command == "verify-tree":
            verify_tree(arguments.manifest, arguments.root)
        elif arguments.command == "write-completion":
            write_completion(
                arguments.output,
                arguments.artifact_id,
                arguments.manifest_sha256,
                arguments.artifact_source_commit,
            )
        elif arguments.command == "verify-completion":
            verify_completion(
                arguments.completion,
                arguments.artifact_id,
                arguments.manifest_sha256,
                arguments.artifact_source_commit,
            )
        elif arguments.command == "install":
            print(atomic_install(arguments.manifest, arguments.partial, arguments.destination))
        elif arguments.command == "verify-launcher":
            verify_launcher(arguments.root, arguments.expected_commit)
        else:
            raise AssertionError(f"unhandled command: {arguments.command}")
    except BundleError as error:
        print(error, file=sys.stderr)
        return 2
    return 0


def create_manifest(
    source: Path,
    durable_root: Path,
    output: Path,
    artifact_source_commit: str,
) -> str:
    """Write a stable manifest for one regular-file tree and return its digest."""
    source = source.resolve(strict=True)
    durable_root = durable_root.resolve(strict=True)
    if source == durable_root:
        raise BundleError("local bundle path cannot equal the profile durable root")
    if not source.is_relative_to(durable_root):
        raise BundleError("local bundle path must be under the profile durable root")
    if not source.is_dir():
        raise BundleError("upload source must be a directory")

    publication = _publication_binding(source, artifact_source_commit)

    records: list[FileRecord] = []
    for directory, directory_names, file_names in os.walk(source, followlinks=False):
        directory_path = Path(directory)
        for directory_name in directory_names:
            candidate = directory_path / directory_name
            if candidate.is_symlink():
                _validate_symlink(candidate, durable_root)
                raise BundleError("symlinked directories are not supported; materialize them first")
        for file_name in file_names:
            candidate = directory_path / file_name
            relative = candidate.relative_to(source).as_posix()
            if candidate.is_symlink():
                _validate_symlink(candidate, durable_root)
                raise BundleError("symlink files are not supported; materialize them first")
            metadata = candidate.stat(follow_symlinks=False)
            if not stat.S_ISREG(metadata.st_mode):
                raise BundleError(f"bundle entry is not a regular file: {relative}")
            records.append(FileRecord(relative, _sha256(candidate), metadata.st_size))
    records.sort(key=lambda record: record.path)
    payload = {
        "file_count": len(records),
        "files": [record.as_dict() for record in records],
        "artifact_source_commit": _commit(artifact_source_commit),
        "schema": _MANIFEST_SCHEMA,
        "total_bytes": sum(record.size for record in records),
        **({"publication": publication} if publication is not None else {}),
    }
    data = _canonical_json(payload)
    _atomic_write(output, data)
    return hashlib.sha256(data).hexdigest()


def load_manifest(
    path: Path,
    expected_sha256: str | None = None,
    expected_artifact_source_commit: str | None = None,
) -> tuple[FileRecord, ...]:
    """Load a canonical manifest and validate its schema, digest, and totals."""
    try:
        data = path.read_bytes()
    except OSError as error:
        raise BundleError(f"cannot read manifest: {error}") from error
    actual_sha256 = hashlib.sha256(data).hexdigest()
    if expected_sha256 is not None and actual_sha256 != expected_sha256:
        raise BundleError("manifest SHA-256 mismatch")
    try:
        payload = json.loads(data)
    except json.JSONDecodeError as error:
        raise BundleError(f"invalid manifest JSON: {error}") from error
    required_fields = {
        "file_count",
        "files",
        "artifact_source_commit",
        "schema",
        "total_bytes",
    }
    if not isinstance(payload, dict) or set(payload) not in (
        required_fields,
        required_fields | {"publication"},
    ):
        raise BundleError("invalid manifest fields")
    if payload["schema"] != _MANIFEST_SCHEMA or not isinstance(payload["files"], list):
        raise BundleError("invalid manifest schema")
    if (
        not isinstance(payload["file_count"], int)
        or isinstance(payload["file_count"], bool)
        or payload["file_count"] < 0
        or not isinstance(payload["total_bytes"], int)
        or isinstance(payload["total_bytes"], bool)
        or payload["total_bytes"] < 0
    ):
        raise BundleError("invalid manifest totals")
    source_commit = _commit(payload["artifact_source_commit"])
    if (
        expected_artifact_source_commit is not None
        and source_commit != expected_artifact_source_commit
    ):
        raise BundleError("manifest artifact source commit mismatch")
    if "publication" in payload:
        _validate_publication_binding(payload["publication"], source_commit)

    records: list[FileRecord] = []
    for raw_record in payload["files"]:
        if not isinstance(raw_record, dict) or set(raw_record) != {"path", "sha256", "size"}:
            raise BundleError("invalid manifest file record")
        record = _file_record(raw_record)
        records.append(record)
    if records != sorted(records, key=lambda record: record.path):
        raise BundleError("manifest file records are not sorted")
    if len({record.path for record in records}) != len(records):
        raise BundleError("manifest contains duplicate paths")
    if payload["file_count"] != len(records):
        raise BundleError("manifest file count mismatch")
    if payload["total_bytes"] != sum(record.size for record in records):
        raise BundleError("manifest total byte count mismatch")
    if data != _canonical_json(payload):
        raise BundleError("manifest is not canonical JSON")
    return tuple(records)


def verify_tree(manifest: Path, root: Path) -> None:
    """Verify that a local tree exactly matches every manifest file."""
    records = load_manifest(manifest)
    if root.is_symlink():
        raise BundleError("bundle tree cannot be a symlink")
    root = root.resolve(strict=True)
    if not root.is_dir():
        raise BundleError("bundle tree is not a directory")
    actual_paths: list[str] = []
    for directory, directory_names, file_names in os.walk(root, followlinks=False):
        directory_path = Path(directory)
        for name in directory_names:
            if (directory_path / name).is_symlink():
                raise BundleError(
                    f"downloaded tree contains a symlink: {(directory_path / name).relative_to(root)}"
                )
        for name in file_names:
            candidate = directory_path / name
            relative = candidate.relative_to(root).as_posix()
            metadata = candidate.stat(follow_symlinks=False)
            if not stat.S_ISREG(metadata.st_mode):
                raise BundleError(f"downloaded tree entry is not a regular file: {relative}")
            actual_paths.append(relative)
    expected_paths = [record.path for record in records]
    if sorted(actual_paths) != expected_paths:
        raise BundleError("downloaded file set does not match manifest")
    for record in records:
        candidate = root / record.path
        if candidate.stat().st_size != record.size:
            raise BundleError(f"size mismatch for {record.path}")
        if _sha256(candidate) != record.sha256:
            raise BundleError(f"SHA-256 mismatch for {record.path}")


def write_completion(
    output: Path,
    artifact_id: str,
    manifest_sha256: str,
    artifact_source_commit: str | None = None,
) -> None:
    """Write the stable completion marker published after payload verification."""
    payload = {
        "artifact_id": artifact_id,
        **(
            {"artifact_source_commit": _commit(artifact_source_commit)}
            if artifact_source_commit is not None
            else {}
        ),
        "complete": True,
        "manifest_sha256": manifest_sha256,
        "schema": _COMPLETION_SCHEMA,
    }
    _atomic_write(output, _canonical_json(payload))


def verify_completion(
    path: Path,
    artifact_id: str,
    manifest_sha256: str,
    artifact_source_commit: str | None = None,
) -> None:
    """Verify a remote completion marker against the requested content address."""
    try:
        data = path.read_bytes()
        payload = json.loads(data)
    except (OSError, json.JSONDecodeError) as error:
        raise BundleError(f"invalid remote completion marker: {error}") from error
    expected = {
        "artifact_id": artifact_id,
        **(
            {"artifact_source_commit": _commit(artifact_source_commit)}
            if artifact_source_commit is not None
            else {}
        ),
        "complete": True,
        "manifest_sha256": manifest_sha256,
        "schema": _COMPLETION_SCHEMA,
    }
    if payload != expected or data != _canonical_json(expected):
        if (
            artifact_source_commit is not None
            and isinstance(payload, dict)
            and payload.get("artifact_source_commit") != artifact_source_commit
        ):
            raise BundleError("remote completion marker artifact source commit mismatch")
        raise BundleError("remote completion marker does not match requested content")


def atomic_install(manifest: Path, partial: Path, destination: Path) -> str:
    """Atomically install a verified sibling directory without replacing a destination."""
    verify_tree(manifest, partial)
    if partial.parent.resolve() != destination.parent.resolve():
        raise BundleError("download partial must be a sibling of the final destination")
    try:
        _rename_no_replace(partial, destination)
    except FileExistsError:
        try:
            verify_tree(manifest, destination)
        except (BundleError, OSError) as error:
            raise BundleError("conflicting final destination") from error
        return "reused"
    return "installed"


def verify_launcher(root: Path, expected_commit: str) -> None:
    """Require the executing transfer checkout to be exact and clean."""
    expected_commit = _commit(expected_commit)
    git_binary = shutil.which("git")
    if git_binary is None:
        raise BundleError("cannot verify launcher checkout: git is unavailable")
    try:
        actual_commit = subprocess.run(  # nosec B603 - fixed git arguments, no shell
            [git_binary, "-C", str(root), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        status = subprocess.run(  # nosec B603 - fixed git arguments, no shell
            [git_binary, "-C", str(root), "status", "--porcelain", "--untracked-files=all"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout
    except (OSError, subprocess.CalledProcessError) as error:
        raise BundleError("cannot verify launcher checkout") from error
    if actual_commit != expected_commit:
        raise BundleError(
            f"launcher checkout does not match expected commit: {expected_commit}, got {actual_commit}"
        )
    if status:
        raise BundleError("launcher checkout is not clean")


def _rename_no_replace(source: Path, destination: Path) -> None:
    system = platform.system()
    library = ctypes.CDLL(None, use_errno=True)
    source_bytes = os.fsencode(source)
    destination_bytes = os.fsencode(destination)
    if system == "Linux":
        try:
            rename = library.renameat2
        except AttributeError as error:
            raise BundleError("atomic no-replace rename is unavailable") from error
        rename.argtypes = [
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_uint,
        ]
        rename.restype = ctypes.c_int
        result = rename(-100, source_bytes, -100, destination_bytes, 1)
    elif system == "Darwin":
        rename = library.renamex_np
        rename.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_uint]
        rename.restype = ctypes.c_int
        result = rename(source_bytes, destination_bytes, 0x00000004)
    else:
        raise BundleError(f"atomic no-replace rename is unsupported on {system}")
    if result == 0:
        return
    error_number = ctypes.get_errno()
    if error_number == errno.EEXIST:
        raise FileExistsError(error_number, os.strerror(error_number), destination)
    raise BundleError(f"atomic no-replace rename failed: {os.strerror(error_number)}")


def _validate_symlink(path: Path, durable_root: Path) -> None:
    try:
        target = path.resolve(strict=True)
    except OSError as error:
        raise BundleError(f"invalid symlink target: {path}") from error
    if not target.is_relative_to(durable_root):
        raise BundleError(f"symlink target escapes durable root: {path}")


def _publication_binding(source: Path, artifact_source_commit: str) -> dict[str, str] | None:
    receipt_path = source / "PUBLICATION.json"
    if not receipt_path.exists():
        return None
    if receipt_path.is_symlink() or not receipt_path.is_file():
        raise BundleError("publication receipt is not a regular file")
    data = receipt_path.read_bytes()
    try:
        payload = json.loads(data)
    except json.JSONDecodeError as error:
        raise BundleError("publication receipt is not JSON") from error
    if not isinstance(payload, dict) or data != _canonical_json(payload):
        raise BundleError("publication receipt is not canonical JSON")
    _validate_publication_binding(payload, artifact_source_commit)
    corpus_manifest = source / "CORPUS_MANIFEST.json"
    if (
        corpus_manifest.is_symlink()
        or not corpus_manifest.is_file()
        or _sha256(corpus_manifest) != payload["corpus_manifest_sha256"]
    ):
        raise BundleError("publication corpus manifest identity mismatch")
    return {
        "artifact_id": payload["artifact_id"],
        "corpus_manifest_sha256": payload["corpus_manifest_sha256"],
        "selection_manifest_sha256": payload["selection_manifest_sha256"],
    }


def _validate_publication_binding(value: object, artifact_source_commit: str) -> None:
    if not isinstance(value, dict):
        raise BundleError("invalid publication binding")
    expected_keys = {
        "artifact_id",
        "corpus_manifest_sha256",
        "selection_manifest_sha256",
    }
    if set(value) == expected_keys:
        binding = value
    else:
        receipt_keys = expected_keys | {
            "artifact_source_commit",
            "file_count",
            "published_path",
            "schema",
            "total_bytes",
        }
        if set(value) != receipt_keys:
            raise BundleError("invalid publication receipt fields")
        if value["schema"] != "modelopt-specdec-publication-receipt-v1":
            raise BundleError("invalid publication receipt schema")
        if value["published_path"] != ".":
            raise BundleError("publication receipt path is not transferable")
        if value["artifact_source_commit"] != artifact_source_commit:
            raise BundleError("publication artifact source commit mismatch")
        binding = value
    for field in expected_keys:
        digest = binding.get(field)
        if (
            not isinstance(digest, str)
            or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
        ):
            raise BundleError(f"invalid publication {field}")
    expected_artifact_id = hashlib.sha256(
        _canonical_json(
            {
                "artifact_source_commit": artifact_source_commit,
                "corpus_manifest_sha256": binding["corpus_manifest_sha256"],
            }
        )
    ).hexdigest()
    if binding["artifact_id"] != expected_artifact_id:
        raise BundleError("publication artifact identity mismatch")


def _file_record(raw_record: dict[str, Any]) -> FileRecord:
    path = raw_record["path"]
    digest = raw_record["sha256"]
    size = raw_record["size"]
    if not isinstance(path, str) or not path:
        raise BundleError("manifest path must be a non-empty string")
    pure_path = PurePosixPath(path)
    if pure_path.is_absolute() or ".." in pure_path.parts or pure_path.as_posix() != path:
        raise BundleError(f"unsafe manifest path: {path}")
    if (
        not isinstance(digest, str)
        or len(digest) != 64
        or any(character not in "0123456789abcdef" for character in digest)
    ):
        raise BundleError(f"invalid SHA-256 for {path}")
    if not isinstance(size, int) or isinstance(size, bool) or size < 0:
        raise BundleError(f"invalid size for {path}")
    return FileRecord(path, digest, size)


def _commit(value: object) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 40
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise BundleError("invalid commit")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(_SHA256_BUFFER_SIZE), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_json(payload: object) -> bytes:
    return (json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n").encode()


def _atomic_write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if os.path.lexists(path):
        if path.is_symlink() or not path.is_file() or path.read_bytes() != data:
            raise BundleError(f"immutable output already exists with different bytes: {path}")
        return
    temporary = path.with_name(f".{path.name}.partial-{uuid4().hex}")
    with temporary.open("xb") as output:
        output.write(data)
        output.flush()
        os.fsync(output.fileno())
    try:
        _rename_no_replace(temporary, path)
    except FileExistsError as error:
        if path.is_symlink() or not path.is_file() or path.read_bytes() != data:
            raise BundleError(
                f"immutable output collision; partial preserved at {temporary}"
            ) from error
    descriptor = os.open(path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    create = commands.add_parser("create")
    create.add_argument("--source", type=Path, required=True)
    create.add_argument("--durable-root", type=Path, required=True)
    create.add_argument("--output", type=Path, required=True)
    create.add_argument("--artifact-source-commit", required=True)
    verify_manifest = commands.add_parser("verify-manifest")
    verify_manifest.add_argument("--manifest", type=Path, required=True)
    verify_manifest.add_argument("--expected-sha256", required=True)
    verify_manifest.add_argument("--expected-artifact-source-commit")
    verify_tree_parser = commands.add_parser("verify-tree")
    verify_tree_parser.add_argument("--manifest", type=Path, required=True)
    verify_tree_parser.add_argument("--root", type=Path, required=True)
    write_completion_parser = commands.add_parser("write-completion")
    write_completion_parser.add_argument("--output", type=Path, required=True)
    write_completion_parser.add_argument("--artifact-id", required=True)
    write_completion_parser.add_argument("--manifest-sha256", required=True)
    write_completion_parser.add_argument("--artifact-source-commit", required=True)
    verify_completion_parser = commands.add_parser("verify-completion")
    verify_completion_parser.add_argument("--completion", type=Path, required=True)
    verify_completion_parser.add_argument("--artifact-id", required=True)
    verify_completion_parser.add_argument("--manifest-sha256", required=True)
    verify_completion_parser.add_argument("--artifact-source-commit", required=True)
    install = commands.add_parser("install")
    install.add_argument("--manifest", type=Path, required=True)
    install.add_argument("--partial", type=Path, required=True)
    install.add_argument("--destination", type=Path, required=True)
    verify_launcher_parser = commands.add_parser("verify-launcher")
    verify_launcher_parser.add_argument("--root", type=Path, required=True)
    verify_launcher_parser.add_argument("--expected-commit", required=True)
    return parser


if __name__ == "__main__":
    raise SystemExit(main())
