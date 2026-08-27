# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Build and verify immutable Qwen3-30B-A3B-Thinking tokenizer receipts."""

from __future__ import annotations

import json
import os
import stat
from hashlib import sha256
from pathlib import Path
from typing import Any

__all__ = [
    "Q30T_TOKENIZER_REPOSITORY",
    "Q30T_TOKENIZER_TRUST_SCHEMA",
    "build_q30t_tokenizer_receipt",
    "snapshot_tree_sha256",
    "verify_q30t_tokenizer_receipt",
]

Q30T_TOKENIZER_REPOSITORY = "Qwen/Qwen3-30B-A3B-Thinking-2507"
Q30T_TOKENIZER_TRUST_SCHEMA = "qwen3-30ba3b-thinking-tokenizer-trust-v1"
_IM_START = "<|im_start|>"
_IM_END = "<|im_end|>"


def _canonical_json(value: object) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _is_lower_hex(value: object, length: int) -> bool:
    return isinstance(value, str) and len(value) == length and all(
        character in "0123456789abcdef" for character in value
    )


def _identity(status: os.stat_result) -> tuple[int, int, int, int]:
    return status.st_dev, status.st_ino, status.st_size, status.st_mtime_ns


def _read_regular_at(directory_fd: int, name: str, display_path: str) -> bytes:
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(name, flags, dir_fd=directory_fd)
    except OSError as error:
        raise ValueError(f"Q30 tokenizer snapshot input is unreadable: {display_path}") from error
    with os.fdopen(descriptor, "rb") as stream:
        before = os.fstat(stream.fileno())
        if not stat.S_ISREG(before.st_mode):
            raise ValueError(f"Q30 tokenizer snapshot input is not regular: {display_path}")
        raw = stream.read()
        after = os.fstat(stream.fileno())
    if _identity(before) != _identity(after) or len(raw) != before.st_size:
        raise ValueError(f"Q30 tokenizer snapshot input changed while reading: {display_path}")
    return raw


def _walk_snapshot(snapshot: Path) -> tuple[list[list[str]], dict[str, bytes]]:
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        root_fd = os.open(snapshot, flags)
    except OSError as error:
        raise ValueError(f"Q30 tokenizer snapshot is unreadable: {snapshot}") from error
    entries: list[list[str]] = []
    files: dict[str, bytes] = {}

    def walk(directory_fd: int, relative: str) -> None:
        before = os.fstat(directory_fd)
        if not stat.S_ISDIR(before.st_mode):
            raise ValueError("Q30 tokenizer snapshot directory is invalid")
        with os.scandir(directory_fd) as scan:
            names = sorted(entry.name for entry in scan)
        for name in names:
            path = f"{relative}/{name}" if relative else name
            try:
                status = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
            except OSError as error:
                raise ValueError(f"Q30 tokenizer snapshot entry is unreadable: {path}") from error
            if stat.S_ISLNK(status.st_mode):
                raise ValueError(f"Q30 tokenizer snapshot cannot contain symlinks: {path}")
            if stat.S_ISREG(status.st_mode):
                raw = _read_regular_at(directory_fd, name, path)
                entries.append([path, sha256(raw).hexdigest()])
                files[path] = raw
                continue
            if not stat.S_ISDIR(status.st_mode):
                raise ValueError(f"Q30 tokenizer snapshot has unsupported entry: {path}")
            try:
                child_fd = os.open(name, flags, dir_fd=directory_fd)
            except OSError as error:
                raise ValueError(f"Q30 tokenizer snapshot directory is unreadable: {path}") from error
            try:
                walk(child_fd, path)
            finally:
                os.close(child_fd)
        after = os.fstat(directory_fd)
        if _identity(before) != _identity(after):
            raise ValueError("Q30 tokenizer snapshot changed while walking")

    try:
        walk(root_fd, "")
    finally:
        os.close(root_fd)
    if not entries:
        raise ValueError("Q30 tokenizer snapshot is empty")
    return entries, files


def snapshot_tree_sha256(snapshot: Path) -> str:
    """Return the descriptor-stable SHA-256 tree identity for one tokenizer snapshot."""
    entries, _ = _walk_snapshot(snapshot)
    return sha256(_canonical_json(entries)).hexdigest()


def _json_object(raw: bytes, name: str) -> dict[str, Any]:
    try:
        value: Any = json.loads(raw)
    except json.JSONDecodeError as error:
        raise ValueError(f"Q30 tokenizer {name} JSON is invalid") from error
    if not isinstance(value, dict):
        raise ValueError(f"Q30 tokenizer {name} JSON is not an object")
    return value


def _training_chat_template(official_template: str) -> str:
    assistant = '    {%- elif message.role == "assistant" %}\n'
    header = "'<|im_start|>' + message.role + '\\n' + "
    thinking_header = "'<|im_start|>' + message.role + '\\n<think>\\n' + "
    im_end = "        {{- '<|im_end|>\\n' }}\n"
    tool = '    {%- elif message.role == "tool" %}'
    boundary = im_end + tool
    if official_template.count(assistant) != 1 or official_template.count(boundary) != 1:
        raise ValueError("Q30 official chat template lacks approved assistant anchors")
    before, remainder = official_template.split(assistant, 1)
    assistant_block, after = remainder.split(boundary, 1)
    if assistant_block.count(header) < 1 or assistant_block.count(thinking_header) > 1:
        raise ValueError("Q30 official chat template lacks the approved assistant header")
    assistant_block = assistant_block.replace(thinking_header, "'<think>\\n' + ", 1)
    assistant_block = assistant_block.replace(header, "")
    return (
        before
        + assistant
        + "        {{- '<|im_start|>' + message.role + '\\n' }}\n"
        + "        {%- generation %}\n"
        + assistant_block
        + im_end
        + "        {%- endgeneration %}\n"
        + tool
        + after
    )


def _special_token_ids(tokenizer_json: dict[str, Any]) -> tuple[int, int]:
    added_tokens = tokenizer_json.get("added_tokens")
    if not isinstance(added_tokens, list):
        raise ValueError("Q30 tokenizer special-token inventory is invalid")
    values: dict[str, int] = {}
    for token in added_tokens:
        if not isinstance(token, dict):
            raise ValueError("Q30 tokenizer special-token inventory is invalid")
        content = token.get("content")
        identifier = token.get("id")
        if content in (_IM_START, _IM_END):
            if type(identifier) is not int or identifier < 0 or content in values:
                raise ValueError("Q30 tokenizer special-token inventory is invalid")
            values[content] = identifier
    if set(values) != {_IM_START, _IM_END}:
        raise ValueError("Q30 tokenizer special tokens are missing")
    return values[_IM_START], values[_IM_END]


def _derived_snapshot_evidence(snapshot: Path) -> dict[str, object]:
    entries, files = _walk_snapshot(snapshot)
    try:
        config = _json_object(files["tokenizer_config.json"], "config")
        tokenizer_json = _json_object(files["tokenizer.json"], "vocabulary")
    except KeyError as error:
        raise ValueError(f"Q30 tokenizer snapshot lacks {error.args[0]}") from error
    official_template = config.get("chat_template")
    if not isinstance(official_template, str) or not official_template:
        raise ValueError("Q30 tokenizer has no official chat template")
    training_template = _training_chat_template(official_template)
    im_start_token_id, im_end_token_id = _special_token_ids(tokenizer_json)
    return {
        "snapshot_tree_sha256": sha256(_canonical_json(entries)).hexdigest(),
        "chat_template_sha256": sha256(official_template.encode("utf-8")).hexdigest(),
        "training_chat_template_sha256": sha256(training_template.encode("utf-8")).hexdigest(),
        "im_start_token_id": im_start_token_id,
        "im_end_token_id": im_end_token_id,
    }


def build_q30t_tokenizer_receipt(snapshot: Path, repository: str, revision: str) -> bytes:
    """Build canonical receipt bytes from one exact Q30 Thinking tokenizer snapshot."""
    if repository != Q30T_TOKENIZER_REPOSITORY or not _is_lower_hex(revision, 40):
        raise ValueError("Q30 tokenizer target identity is invalid")
    body: dict[str, object] = {
        "schema_version": Q30T_TOKENIZER_TRUST_SCHEMA,
        "repository": repository,
        "revision": revision,
        "snapshot_path": str(snapshot),
    }
    body.update(_derived_snapshot_evidence(snapshot))
    payload = body | {"receipt_sha256": sha256(_canonical_json(body)).hexdigest()}
    return _canonical_json(payload) + b"\n"


def verify_q30t_tokenizer_receipt(receipt: bytes) -> dict[str, object]:
    """Recompute and validate every Q30 tokenizer receipt identity from disk."""
    try:
        payload: Any = json.loads(receipt)
    except json.JSONDecodeError as error:
        raise ValueError("Q30 tokenizer receipt JSON is invalid") from error
    expected_keys = {
        "schema_version",
        "repository",
        "revision",
        "snapshot_path",
        "snapshot_tree_sha256",
        "chat_template_sha256",
        "training_chat_template_sha256",
        "im_start_token_id",
        "im_end_token_id",
        "receipt_sha256",
    }
    if (
        not isinstance(payload, dict)
        or receipt != _canonical_json(payload) + b"\n"
        or set(payload) != expected_keys
        or payload["schema_version"] != Q30T_TOKENIZER_TRUST_SCHEMA
        or payload["repository"] != Q30T_TOKENIZER_REPOSITORY
        or not _is_lower_hex(payload["revision"], 40)
        or not isinstance(payload["snapshot_path"], str)
        or not Path(payload["snapshot_path"]).is_absolute()
    ):
        raise ValueError("Q30 tokenizer receipt identity is invalid")
    body = {name: value for name, value in payload.items() if name != "receipt_sha256"}
    if payload["receipt_sha256"] != sha256(_canonical_json(body)).hexdigest():
        raise ValueError("Q30 tokenizer receipt hash does not reconcile")
    observed = _derived_snapshot_evidence(Path(payload["snapshot_path"]))
    if any(payload[name] != value for name, value in observed.items()):
        raise ValueError("Q30 tokenizer receipt evidence does not reconcile")
    return payload
