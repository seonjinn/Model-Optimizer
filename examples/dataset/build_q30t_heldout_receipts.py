# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Build and verify canonical Q30 held-out consumer and provenance receipts."""

from __future__ import annotations

import argparse
import json
import os
import stat
import sys
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Sequence

try:
    from common.specdec.qwen4b_b_atomic import (  # pyright: ignore[reportMissingImports]
        atomic_publish_bytes,
    )
except ModuleNotFoundError:
    # Direct script execution places only examples/dataset on sys.path.
    _REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(_REPOSITORY_ROOT))
    try:
        from tools.launcher.common.specdec.qwen4b_b_atomic import atomic_publish_bytes
    finally:
        sys.path.pop(0)

__all__ = [
    "HELD_OUT_ORDER",
    "HELD_OUT_SOURCES_SCHEMA",
    "HeldOutError",
    "HeldOutSource",
    "build_all",
    "build_heldout_provenance",
    "build_heldout_receipt",
    "load_heldout_source",
    "load_sources_manifest",
    "main",
    "publish_receipt",
    "stable_single_link_regular_bytes",
    "verify_heldout_provenance",
    "verify_heldout_receipt",
]

HELD_OUT_ORDER = ("speed", "math", "code", "swe", "tool")
HELD_OUT_SOURCES_SCHEMA = "q30t-held-out-sources-v1"
_CONSUMER_SCHEMA = "specdec-held-out-uuid-receipt-v1"
_PROVENANCE_SCHEMA = "q30t-held-out-provenance-v1"
_MAX_MANIFEST_BYTES = 4 * 1024 * 1024
_MAX_HELD_OUT_BYTES = 256 * 1024 * 1024
_READ_BLOCK_BYTES = 1024 * 1024


class HeldOutError(ValueError):
    """Held-out evidence failed canonical validation or authentication."""


@dataclass(frozen=True)
class HeldOutSource:
    """One caller-pinned external held-out UUID source."""

    name: str
    path: Path
    file_sha256: str
    repository: str
    revision: str
    variant: str
    prompt_uuids: tuple[str, ...]


def _canonical_json(value: object) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode(
        "utf-8"
    )


def _is_lower_hex(value: object, length: int) -> bool:
    return (
        type(value) is str
        and len(value) == length
        and all(character in "0123456789abcdef" for character in value)
    )


def _identity(status: os.stat_result) -> tuple[int, int, int, int, int]:
    return status.st_dev, status.st_ino, status.st_size, status.st_mtime_ns, status.st_ctime_ns


def stable_single_link_regular_bytes(path: Path, *, max_bytes: int) -> bytes:
    """Read bounded bytes while binding one no-follow path to one immutable inode."""
    if type(max_bytes) is not int or max_bytes < 0:
        raise ValueError("held-out retention limit is invalid")
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0)
    directory_flags = flags | getattr(os, "O_DIRECTORY", 0)
    parent_fd: int | None = None
    descriptor: int | None = None
    try:
        named_parent = os.stat(path.parent, follow_symlinks=False)
        parent_fd = os.open(path.parent, directory_flags)
        parent_before = os.fstat(parent_fd)
        if _identity(named_parent) != _identity(parent_before) or not stat.S_ISDIR(
            parent_before.st_mode
        ):
            raise HeldOutError(f"held-out input parent changed while opening: {path}")
        named_before = os.stat(path.name, dir_fd=parent_fd, follow_symlinks=False)
        descriptor = os.open(path.name, flags, dir_fd=parent_fd)
    except HeldOutError:
        if descriptor is not None:
            os.close(descriptor)
        if parent_fd is not None:
            os.close(parent_fd)
        raise
    except OSError as error:
        if descriptor is not None:
            os.close(descriptor)
        if parent_fd is not None:
            os.close(parent_fd)
        raise HeldOutError(f"held-out input is unreadable: {path}") from error
    try:
        before = os.fstat(descriptor)
        if (
            _identity(named_before) != _identity(before)
            or not stat.S_ISREG(before.st_mode)
            or before.st_nlink != 1
        ):
            raise HeldOutError(f"held-out input is not a single-link regular file: {path}")
        if before.st_size > max_bytes:
            raise HeldOutError(f"held-out input exceeds the retention limit: {path}")
        retained = bytearray()
        while True:
            block = os.read(descriptor, min(_READ_BLOCK_BYTES, max_bytes + 1 - len(retained)))
            if not block:
                break
            retained.extend(block)
            if len(retained) > max_bytes:
                raise HeldOutError(f"held-out input exceeds the retention limit: {path}")
        after = os.fstat(descriptor)
        named_after = os.stat(path.name, dir_fd=parent_fd, follow_symlinks=False)
        parent_after = os.fstat(parent_fd)
        absolute_parent = os.stat(path.parent, follow_symlinks=False)
        absolute_file = os.stat(path, follow_symlinks=False)
        if (
            _identity(before) != _identity(after)
            or _identity(after) != _identity(named_after)
            or _identity(after) != _identity(absolute_file)
            or _identity(parent_before) != _identity(parent_after)
            or _identity(parent_after) != _identity(absolute_parent)
            or after.st_nlink != 1
            or len(retained) != before.st_size
        ):
            raise HeldOutError(f"held-out input changed while reading: {path}")
        return bytes(retained)
    except OSError as error:
        raise HeldOutError(f"held-out input changed while reading: {path}") from error
    finally:
        os.close(descriptor)
        os.close(parent_fd)


def _load_canonical_object(raw: bytes, *, label: str) -> dict[str, Any]:
    try:
        payload: Any = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise HeldOutError(f"{label} JSON is invalid") from error
    if not isinstance(payload, dict) or raw != _canonical_json(payload) + b"\n":
        raise HeldOutError(f"{label} is not canonical")
    return payload


def _validated_prompt_uuids(value: object, *, label: str) -> tuple[str, ...]:
    if not isinstance(value, list) or not value:
        raise HeldOutError(f"{label} cannot be empty")
    if any(not _is_lower_hex(item, 64) for item in value):
        raise HeldOutError(f"{label} UUID must be lowercase 64-hex")
    if value != sorted(value):
        raise HeldOutError(f"{label} UUIDs must be sorted")
    if len(value) != len(set(value)):
        raise HeldOutError(f"{label} UUIDs contain a duplicate")
    return tuple(value)


def load_heldout_source(path: Path, expected_sha256: str) -> tuple[str, ...]:
    """Authenticate one caller-pinned canonical held-out UUID source."""
    if not _is_lower_hex(expected_sha256, 64):
        raise HeldOutError("held-out source caller SHA-256 is invalid")
    raw = stable_single_link_regular_bytes(path, max_bytes=_MAX_HELD_OUT_BYTES)
    if sha256(raw).hexdigest() != expected_sha256:
        raise HeldOutError("held-out source whole-file SHA-256 mismatch")
    payload = _load_canonical_object(raw, label="held-out source")
    if set(payload) != {"prompt_uuids"}:
        raise HeldOutError("held-out source is not canonical")
    return _validated_prompt_uuids(payload["prompt_uuids"], label="held-out source")


def load_sources_manifest(path: Path, expected_sha256: str) -> tuple[HeldOutSource, ...]:
    """Load the exact five externally supplied held-out source identities."""
    if not _is_lower_hex(expected_sha256, 64):
        raise HeldOutError("held-out sources manifest caller SHA-256 is invalid")
    raw = stable_single_link_regular_bytes(path, max_bytes=_MAX_MANIFEST_BYTES)
    if sha256(raw).hexdigest() != expected_sha256:
        raise HeldOutError("held-out sources manifest whole-file SHA-256 mismatch")
    payload = _load_canonical_object(raw, label="held-out sources manifest")
    if set(payload) != {"schema_version", "sources"} or payload["schema_version"] != (
        HELD_OUT_SOURCES_SCHEMA
    ):
        raise HeldOutError("held-out sources manifest identity is invalid")
    records = payload["sources"]
    if not isinstance(records, list):
        raise HeldOutError("held-out sources manifest sources are invalid")
    expected_keys = {"name", "path", "sha256", "repository", "revision", "variant"}
    parsed: list[tuple[str, Path, str, str, str, str]] = []
    names: list[str] = []
    for record in records:
        if not isinstance(record, dict) or set(record) != expected_keys:
            raise HeldOutError("held-out source manifest entry is invalid")
        name = record["name"]
        source_path_value = record["path"]
        repository = record["repository"]
        revision = record["revision"]
        variant = record["variant"]
        file_sha256 = record["sha256"]
        if type(name) is not str or name not in HELD_OUT_ORDER:
            raise HeldOutError("held-out source name is invalid")
        if name in names:
            raise HeldOutError("held-out source manifest contains a duplicate name")
        if type(source_path_value) is not str or not Path(source_path_value).is_absolute():
            raise HeldOutError("held-out source path must be absolute")
        if type(repository) is not str or not repository:
            raise HeldOutError("held-out source repository is empty")
        if type(variant) is not str or not variant:
            raise HeldOutError("held-out source variant is empty")
        if not (_is_lower_hex(revision, 40) or _is_lower_hex(revision, 64)):
            raise HeldOutError("held-out source revision must be lowercase 40- or 64-hex")
        if not _is_lower_hex(file_sha256, 64):
            raise HeldOutError("held-out source SHA-256 is invalid")
        parsed.append((name, Path(source_path_value), file_sha256, repository, revision, variant))
        names.append(name)
    if set(names) != set(HELD_OUT_ORDER) or len(names) != len(HELD_OUT_ORDER):
        raise HeldOutError("held-out sources manifest must contain the exact five names")
    sources = [
        HeldOutSource(
            name=name,
            path=source_path,
            file_sha256=file_sha256,
            repository=repository,
            revision=revision,
            variant=variant,
            prompt_uuids=load_heldout_source(source_path, file_sha256),
        )
        for name, source_path, file_sha256, repository, revision, variant in parsed
    ]
    return tuple(sorted(sources, key=lambda source: HELD_OUT_ORDER.index(source.name)))


def verify_heldout_receipt(raw: bytes) -> tuple[str, ...]:
    """Verify canonical consumer receipt bytes and return their UUID sequence."""
    payload = _load_canonical_object(raw, label="held-out consumer receipt")
    if (
        set(payload)
        != {
            "schema_version",
            "prompt_uuids",
            "prompt_uuids_sha256",
            "receipt_sha256",
        }
        or payload["schema_version"] != _CONSUMER_SCHEMA
    ):
        raise HeldOutError("held-out consumer receipt identity is invalid")
    prompt_uuids = _validated_prompt_uuids(
        payload["prompt_uuids"], label="held-out consumer receipt"
    )
    body = {key: value for key, value in payload.items() if key != "receipt_sha256"}
    if (
        payload["prompt_uuids_sha256"] != sha256(_canonical_json(list(prompt_uuids))).hexdigest()
        or payload["receipt_sha256"] != sha256(_canonical_json(body)).hexdigest()
    ):
        raise HeldOutError("held-out consumer receipt hashes do not reconcile")
    return prompt_uuids


def build_heldout_receipt(source: HeldOutSource) -> bytes:
    """Build the existing consumer schema without embedding provenance fields."""
    prompt_uuids = _validated_prompt_uuids(list(source.prompt_uuids), label="held-out source")
    body = {
        "schema_version": _CONSUMER_SCHEMA,
        "prompt_uuids": list(prompt_uuids),
        "prompt_uuids_sha256": sha256(_canonical_json(list(prompt_uuids))).hexdigest(),
    }
    return (
        _canonical_json(body | {"receipt_sha256": sha256(_canonical_json(body)).hexdigest()})
        + b"\n"
    )


def build_heldout_provenance(source: HeldOutSource, receipt: bytes) -> bytes:
    """Build source lineage separately while cross-binding the consumer bytes."""
    receipt_prompt_uuids = verify_heldout_receipt(receipt)
    if receipt_prompt_uuids != source.prompt_uuids:
        raise HeldOutError("held-out provenance consumer receipt does not match its source")
    body = {
        "schema_version": _PROVENANCE_SCHEMA,
        "name": source.name,
        "repository": source.repository,
        "revision": source.revision,
        "variant": source.variant,
        "source_path": str(source.path),
        "source_file_sha256": source.file_sha256,
        "prompt_uuids_sha256": sha256(_canonical_json(list(receipt_prompt_uuids))).hexdigest(),
        "consumer_receipt_file_sha256": sha256(receipt).hexdigest(),
    }
    return (
        _canonical_json(body | {"receipt_sha256": sha256(_canonical_json(body)).hexdigest()})
        + b"\n"
    )


def verify_heldout_provenance(
    raw: bytes, *, receipt: bytes, source: HeldOutSource
) -> dict[str, object]:
    """Verify provenance bytes against both their exact consumer and source."""
    prompt_uuids = verify_heldout_receipt(receipt)
    if prompt_uuids != source.prompt_uuids:
        raise HeldOutError("held-out provenance consumer receipt does not match its source")
    payload = _load_canonical_object(raw, label="held-out provenance receipt")
    body = {
        "schema_version": _PROVENANCE_SCHEMA,
        "name": source.name,
        "repository": source.repository,
        "revision": source.revision,
        "variant": source.variant,
        "source_path": str(source.path),
        "source_file_sha256": source.file_sha256,
        "prompt_uuids_sha256": sha256(_canonical_json(list(prompt_uuids))).hexdigest(),
        "consumer_receipt_file_sha256": sha256(receipt).hexdigest(),
    }
    expected = body | {"receipt_sha256": sha256(_canonical_json(body)).hexdigest()}
    if payload != expected:
        raise HeldOutError("held-out provenance receipt does not reconcile")
    return payload


def _adopt_exact_receipt(destination: Path, payload: bytes) -> None:
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0)
    directory_flags = flags | getattr(os, "O_DIRECTORY", 0)
    parent_fd: int | None = None
    descriptor: int | None = None
    final_descriptor: int | None = None
    try:
        named_parent = os.stat(destination.parent, follow_symlinks=False)
        parent_fd = os.open(destination.parent, directory_flags)
        parent_before = os.fstat(parent_fd)
        if _identity(named_parent) != _identity(parent_before) or not stat.S_ISDIR(
            parent_before.st_mode
        ):
            raise FileExistsError(
                f"held-out destination parent cannot be adopted: {destination.parent}"
            )
        named_before = os.stat(destination.name, dir_fd=parent_fd, follow_symlinks=False)
        descriptor = os.open(destination.name, flags, dir_fd=parent_fd)
        before = os.fstat(descriptor)
        if (
            _identity(named_before) != _identity(before)
            or not stat.S_ISREG(before.st_mode)
            or before.st_nlink != 1
            or before.st_size != len(payload)
        ):
            raise FileExistsError(f"held-out destination cannot be adopted: {destination}")
        retained = bytearray()
        while len(retained) <= len(payload):
            block = os.read(
                descriptor,
                min(_READ_BLOCK_BYTES, len(payload) + 1 - len(retained)),
            )
            if not block:
                break
            retained.extend(block)
        after = os.fstat(descriptor)
        rebound = os.stat(destination.name, dir_fd=parent_fd, follow_symlinks=False)
        if (
            bytes(retained) != payload
            or _identity(before) != _identity(after)
            or _identity(after) != _identity(rebound)
            or after.st_nlink != 1
        ):
            raise FileExistsError(f"held-out destination differs or changed: {destination}")
        os.fsync(descriptor)
        os.fsync(parent_fd)
        final_parent = os.fstat(parent_fd)
        absolute_parent = os.stat(destination.parent, follow_symlinks=False)
        final_named = os.stat(destination.name, dir_fd=parent_fd, follow_symlinks=False)
        final_descriptor = os.open(destination.name, flags, dir_fd=parent_fd)
        final_opened = os.fstat(final_descriptor)
        final_absolute = os.stat(destination, follow_symlinks=False)
        if (
            _identity(final_parent) != _identity(parent_before)
            or _identity(absolute_parent) != _identity(parent_before)
            or _identity(final_named) != _identity(after)
            or _identity(final_opened) != _identity(after)
            or _identity(final_absolute) != _identity(after)
            or final_opened.st_nlink != 1
        ):
            raise FileExistsError(f"held-out destination parent or file rebound: {destination}")
    except FileExistsError:
        raise
    except OSError as error:
        raise FileExistsError(f"held-out destination cannot be adopted: {destination}") from error
    finally:
        if final_descriptor is not None:
            os.close(final_descriptor)
        if descriptor is not None:
            os.close(descriptor)
        if parent_fd is not None:
            os.close(parent_fd)


def publish_receipt(destination: Path, payload: bytes, *, job_id: str) -> None:
    """Publish or adopt exact immutable receipt bytes without replacement."""
    if os.path.lexists(destination):
        _adopt_exact_receipt(destination, payload)
        return
    try:
        atomic_publish_bytes(destination, payload, job_id=job_id)
    except Exception:
        if not os.path.lexists(destination):
            raise
    _adopt_exact_receipt(destination, payload)


def build_all(
    sources: tuple[HeldOutSource, ...], output_root: Path, *, job_id: str
) -> dict[str, dict[str, object]]:
    """Publish all five receipt pairs in the fixed consumer name order."""
    if not output_root.is_absolute():
        raise HeldOutError("held-out receipt output root must be absolute")
    names = [source.name for source in sources]
    if len(names) != len(set(names)):
        raise HeldOutError("held-out source set contains a duplicate name")
    if set(names) != set(HELD_OUT_ORDER) or len(names) != len(HELD_OUT_ORDER):
        raise HeldOutError("held-out source set must contain the exact five names")
    by_name = {source.name: source for source in sources}
    outputs: dict[str, dict[str, object]] = {}
    for name in HELD_OUT_ORDER:
        source = by_name[name]
        receipt = build_heldout_receipt(source)
        provenance = build_heldout_provenance(source, receipt)
        receipt_path = output_root / f"{name}.json"
        provenance_path = output_root / f"{name}.provenance.json"
        publish_receipt(receipt_path, receipt, job_id=f"{job_id}-{name}-consumer")
        publish_receipt(provenance_path, provenance, job_id=f"{job_id}-{name}-provenance")
        outputs[name] = {
            "receipt_path": str(receipt_path),
            "receipt_file_sha256": sha256(receipt).hexdigest(),
            "provenance_path": str(provenance_path),
            "provenance_file_sha256": sha256(provenance).hexdigest(),
        }
    return outputs


def _verify_all(sources: tuple[HeldOutSource, ...], receipt_root: Path) -> list[dict[str, object]]:
    if not receipt_root.is_absolute():
        raise HeldOutError("held-out receipt root must be absolute")
    names = [source.name for source in sources]
    if tuple(names) != HELD_OUT_ORDER:
        raise HeldOutError("held-out sources are not in the fixed name order")
    receipts: list[dict[str, object]] = []
    for source in sources:
        receipt_path = receipt_root / f"{source.name}.json"
        provenance_path = receipt_root / f"{source.name}.provenance.json"
        receipt = stable_single_link_regular_bytes(receipt_path, max_bytes=_MAX_HELD_OUT_BYTES)
        provenance = stable_single_link_regular_bytes(
            provenance_path, max_bytes=_MAX_MANIFEST_BYTES
        )
        prompt_uuids = verify_heldout_receipt(receipt)
        verify_heldout_provenance(provenance, receipt=receipt, source=source)
        if receipt != build_heldout_receipt(source) or provenance != build_heldout_provenance(
            source, receipt
        ):
            raise HeldOutError("held-out published bytes differ from their source")
        receipts.append(
            {
                "name": source.name,
                "prompt_uuid_count": len(prompt_uuids),
                "prompt_uuids_sha256": sha256(_canonical_json(list(prompt_uuids))).hexdigest(),
                "consumer_receipt_file_sha256": sha256(receipt).hexdigest(),
                "provenance_receipt_file_sha256": sha256(provenance).hexdigest(),
            }
        )
    return receipts


def _arguments(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build or verify canonical Q30 held-out receipt pairs."
    )
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("build", "verify"):
        command = commands.add_parser(name)
        command.add_argument("--sources-manifest", type=Path, required=True)
        command.add_argument("--sources-manifest-sha256", required=True)
        if name == "build":
            command.add_argument("--output-root", type=Path, required=True)
        else:
            command.add_argument("--receipt-root", type=Path, required=True)
            command.add_argument("--review-output", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Build immutable pairs or verify them into a separate review receipt."""
    arguments = _arguments(argv)
    sources = load_sources_manifest(arguments.sources_manifest, arguments.sources_manifest_sha256)
    job_id = os.environ.get("SLURM_JOB_ID", str(os.getpid()))
    if arguments.command == "build":
        outputs = build_all(sources, arguments.output_root, job_id=job_id)
        print(json.dumps(outputs, sort_keys=True, separators=(",", ":")))
        return 0
    if not arguments.review_output.is_absolute():
        raise HeldOutError("held-out review output must be absolute")
    receipts = _verify_all(sources, arguments.receipt_root)
    body = {
        "schema_version": "q30t-held-out-review-v1",
        "sources_manifest_file_sha256": arguments.sources_manifest_sha256,
        "receipts": receipts,
    }
    review = (
        _canonical_json(body | {"receipt_sha256": sha256(_canonical_json(body)).hexdigest()})
        + b"\n"
    )
    publish_receipt(arguments.review_output, review, job_id=f"{job_id}-review")
    print(
        json.dumps(
            {
                "path": str(arguments.review_output),
                "sha256": sha256(review).hexdigest(),
            },
            sort_keys=True,
            separators=(",", ":"),
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
