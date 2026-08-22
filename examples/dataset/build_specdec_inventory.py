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

"""Build a hash-bound, pretokenized SpecDec inventory from synthesis or trace shards."""

import argparse
import hashlib
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

_SHA = re.compile(r"^[0-9a-f]{40}(?:[0-9a-f]{24})?$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tokenizer_snapshot_sha256(root: Path) -> str:
    """Hash tokenizer serialization files without hashing model weights."""
    names = {
        "added_tokens.json",
        "chat_template.jinja",
        "merges.txt",
        "special_tokens_map.json",
        "tokenizer.json",
        "tokenizer.model",
        "tokenizer_config.json",
        "vocab.json",
        "vocab.txt",
    }
    files = sorted(path for path in root.iterdir() if path.is_file() and path.name in names)
    if not files:
        raise ValueError("tokenizer snapshot contains no serialization files")
    digest = hashlib.sha256()
    for path in files:
        digest.update(path.name.encode())
        digest.update(b"\0")
        digest.update(bytes.fromhex(sha256_file(path)))
    return digest.hexdigest()


@dataclass(frozen=True)
class InventorySource:
    """One immutable raw/synthesis or recorded-trace input lane."""

    source_id: str
    source_revision: str
    license: str
    pool: str
    category: str
    response_source: str
    tool_lane: str
    manifest_path: Path

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "manifest_path", Path(self.manifest_path).expanduser().resolve(strict=False)
        )
        if not self.source_id or not self.license or not self.category:
            raise ValueError("source identity, license, and category are required")
        if _SHA.fullmatch(self.source_revision) is None:
            raise ValueError("source_revision must be a pinned commit digest")
        if self.pool not in {"ptv2", "ptv3"}:
            raise ValueError("inventory pool must be ptv2 or ptv3")
        if self.response_source not in {"target-synth", "trace-replay"}:
            raise ValueError("response_source must be target-synth or trace-replay")
        expected_lane = "none" if self.response_source == "target-synth" else "recorded-trace"
        if self.tool_lane != expected_lane:
            raise ValueError("tool lane does not match the response source")


def _verified_files(manifest_path: Path) -> tuple[str, list[Path]]:
    raw = manifest_path.read_bytes()
    manifest_sha256 = hashlib.sha256(raw).hexdigest()
    payload = json.loads(raw)
    records = payload.get("files") if isinstance(payload, dict) else None
    if (
        not isinstance(payload, dict)
        or payload.get("schema_version") != 1
        or not isinstance(records, list)
        or not records
    ):
        raise ValueError("source manifest must contain a non-empty version-1 file set")
    root = manifest_path.parent.resolve(strict=True)
    files: list[Path] = []
    for record in records:
        relative = record.get("path") if isinstance(record, dict) else None
        if not isinstance(relative, str) or not relative:
            raise ValueError("source manifest has an invalid file record")
        path = (manifest_path.parent / relative).resolve(strict=True)
        if not path.is_relative_to(root) or not path.is_file():
            raise ValueError(f"source file escapes manifest root: {relative}")
        if path.stat().st_size != record.get("bytes") or sha256_file(path) != record.get("sha256"):
            raise ValueError(f"source file identity mismatch: {relative}")
        files.append(path)
    return manifest_sha256, files


def _iter_rows(path: Path):
    if path.suffix == ".parquet":
        import pyarrow.parquet as pq

        yield from pq.read_table(path).to_pylist()
        return
    with path.open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            if line.strip():
                try:
                    yield json.loads(line)
                except json.JSONDecodeError as error:
                    raise ValueError(f"invalid JSON row: {path}:{line_number}") from error


def _context_bucket(token_count: int) -> str:
    if token_count <= 4096:
        return "le4k"
    if token_count <= 16384:
        return "4k_16k"
    if token_count <= 32768:
        return "16k_32k"
    raise ValueError(f"conversation exceeds the 32K inventory limit: {token_count}")


def _tokenize(tokenizer: Any, row: dict[str, Any]) -> tuple[list[int], list[int]]:
    messages = row.get("messages") or row.get("conversations")
    if not isinstance(messages, list) or not messages:
        raise ValueError("inventory row has no messages")
    encoded = tokenizer.apply_chat_template(
        messages,
        tools=row.get("tools") or None,
        tokenize=True,
        add_generation_prompt=False,
        return_dict=True,
        return_assistant_tokens_mask=True,
    )
    input_ids = encoded.get("input_ids")
    loss_mask = encoded.get("assistant_masks", encoded.get("assistant_mask"))
    if (
        not isinstance(input_ids, list)
        or not isinstance(loss_mask, list)
        or len(input_ids) != len(loss_mask)
        or any(value not in (0, 1) for value in loss_mask)
    ):
        raise ValueError("tokenizer did not return aligned IDs and assistant mask")
    return input_ids, loss_mask


def build_inventory_rows(
    sources: list[InventorySource],
    *,
    tokenizer: Any,
    tokenizer_sha256: str,
    training_seq_len: int,
) -> list[dict[str, Any]]:
    """Verify and pretokenize all source rows without trusting caller metadata."""
    if _SHA256.fullmatch(tokenizer_sha256) is None or training_seq_len < 1:
        raise ValueError("tokenizer digest and training sequence length must be pinned")
    inventory: list[dict[str, Any]] = []
    seen: set[str] = set()
    for spec in sources:
        manifest_sha256, files = _verified_files(spec.manifest_path)
        for source_file in files:
            source_file_sha256 = sha256_file(source_file)
            for index, raw_row in enumerate(_iter_rows(source_file)):
                row = dict(raw_row)
                prompt_id = str(
                    row.get("prompt_id")
                    or hashlib.sha256(
                        json.dumps(
                            row.get("messages") or row.get("conversations"),
                            ensure_ascii=False,
                            separators=(",", ":"),
                            sort_keys=True,
                        ).encode()
                    ).hexdigest()
                )
                if prompt_id in seen:
                    raise ValueError(f"duplicate prompt_id {prompt_id}")
                seen.add(prompt_id)
                has_tools = bool(row.get("tools")) or any(
                    message.get("role") == "tool" or message.get("tool_calls")
                    for message in row.get("messages", [])
                )
                if has_tools != (spec.tool_lane == "recorded-trace"):
                    raise ValueError(f"prompt {prompt_id} is routed to the wrong tool lane")
                input_ids, full_mask = _tokenize(tokenizer, row)
                full_token_count = len(input_ids)
                training_ids = input_ids[:training_seq_len]
                training_mask = full_mask[:training_seq_len]
                assistant_tokens = sum(training_mask)
                if assistant_tokens < 1:
                    continue
                inventory.append(
                    {
                        "prompt_id": prompt_id,
                        "pool": spec.pool,
                        "category": spec.category,
                        "context_bucket": _context_bucket(full_token_count),
                        "full_token_count": full_token_count,
                        "full_assistant_tokens": sum(full_mask),
                        "assistant_tokens": assistant_tokens,
                        "source_id": spec.source_id,
                        "source_revision": spec.source_revision,
                        "license": spec.license,
                        "source_manifest_sha256": manifest_sha256,
                        "source_file_path": str(source_file.relative_to(spec.manifest_path.parent)),
                        "source_file_sha256": source_file_sha256,
                        "source_row_index": index,
                        "response_source": spec.response_source,
                        "tool_lane": spec.tool_lane,
                        "tokenizer_sha256": tokenizer_sha256,
                        "input_ids": training_ids,
                        "loss_mask": training_mask,
                        "messages": row.get("messages") or row.get("conversations"),
                        "tools": row.get("tools"),
                    }
                )
    return inventory


def _write_jsonl_atomic(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(f".{path.name}.partial-{os.getpid()}")
    with partial.open("w", encoding="utf-8") as output:
        for row in rows:
            output.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        output.flush()
        os.fsync(output.fileno())
    os.replace(partial, path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sources-config", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--tokenizer-sha256", required=True)
    parser.add_argument("--training-seq-len", type=int, default=4096)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    from transformers import AutoTokenizer

    config = yaml.safe_load(args.sources_config.read_text(encoding="utf-8"))
    sources = [
        InventorySource(
            **(record | {"manifest_path": args.sources_config.parent / record["manifest_path"]})
        )
        for record in config["sources"]
    ]
    actual_tokenizer_sha256 = tokenizer_snapshot_sha256(args.tokenizer)
    if actual_tokenizer_sha256 != args.tokenizer_sha256:
        raise ValueError("tokenizer snapshot SHA-256 mismatch")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True)
    rows = build_inventory_rows(
        sources,
        tokenizer=tokenizer,
        tokenizer_sha256=args.tokenizer_sha256,
        training_seq_len=args.training_seq_len,
    )
    _write_jsonl_atomic(args.output, rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
