# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""OpenAI-compatible client for querying LLM inference servers.

Used by TRT-LLM and vLLM query scripts to send prompts to a running server,
collect responses, and optionally save them to disk for downstream pipelines
(e.g., EAGLE3 data synthesis).
"""

# ruff: noqa: D101, D102, D103, D107
import argparse
import hashlib
import json
import os
import re
from copy import deepcopy
from pathlib import Path
from typing import Any

from datasets import load_dataset

early_termination = False
args: argparse.Namespace
llm: "LLM"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_thinking_mode(mode: str, row: dict[str, Any]) -> bool:
    if mode == "on":
        return True
    if mode == "off":
        return False
    if mode == "source":
        return bool(row.get("enable_thinking", True))
    raise ValueError(f"Unsupported thinking mode: {mode!r}")


def prepare_generation_messages(
    row: dict[str, Any],
    mode: str,
    *,
    shard_id: int,
    reject_tool_trajectories: bool = False,
) -> list[dict[str, Any]]:
    del shard_id
    messages = row.get("messages") or row.get("conversations")
    if messages is None:
        raise ValueError(
            "No conversations or messages in the data. Only OAI chat data is supported."
        )
    if reject_tool_trajectories and (
        row.get("tools")
        or any(message.get("role") == "tool" or message.get("tool_calls") for message in messages)
    ):
        raise ValueError("target synthesis received a tool trajectory; use trace replay")
    prepared = deepcopy(messages)
    if not resolve_thinking_mode(mode, row):
        for message in prepared:
            if message["role"] == "user":
                message["content"] = f"{message['content']} /no_think"
    return prepared


def build_shard_metadata(
    *,
    output_path: Path,
    shard_id: int,
    num_shards: int,
    thinking_mode: str,
    source_id: str,
    target_revision: str,
    temperature: float,
    max_tokens: int | None,
    max_total_length: int | None,
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "shard_id": shard_id,
        "num_shards": num_shards,
        "thinking_mode": thinking_mode,
        "source_id": source_id,
        "target_revision": target_revision,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "max_total_length": max_total_length,
        "output_file": output_path.name,
        "output_sha256": sha256_file(output_path),
        "output_bytes": output_path.stat().st_size,
    }


def verify_completed_shard(
    output_path: Path,
    metadata_path: Path,
    done_path: Path,
    *,
    expected: dict[str, Any],
) -> None:
    """Rehash an identity-bound shard before treating it as resumable."""
    if not output_path.is_file() or not metadata_path.is_file() or not done_path.is_file():
        raise ValueError(f"incomplete synthesis shard state: {output_path.name}")
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"completed shard identity mismatch: {output_path.name}") from error
    identity_matches = (
        isinstance(metadata, dict)
        and metadata.get("schema_version") == 1
        and all(metadata.get(key) == value for key, value in expected.items())
        and metadata.get("output_file") == output_path.name
        and metadata.get("output_bytes") == output_path.stat().st_size
        and metadata.get("output_sha256") == sha256_file(output_path)
        and done_path.read_text(encoding="utf-8") == "done\n"
    )
    if not identity_matches:
        raise ValueError(f"completed shard identity mismatch: {output_path.name}")


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    partial = path.with_name(f".{path.name}.partial-{os.getpid()}")
    partial.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(partial, path)


def resolve_local_dataset(path: Path) -> tuple[str, list[str]]:
    """Resolve one local file or sharded directory into a datasets input."""
    if path.is_file():
        if path.suffix.lower() == ".json":
            try:
                manifest = json.loads(path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                manifest = None
            if (
                isinstance(manifest, dict)
                and manifest.get("schema_version") == 1
                and isinstance(manifest.get("files"), list)
            ):
                files = []
                root = path.parent.resolve(strict=True)
                for record in manifest["files"]:
                    candidate = (path.parent / record["path"]).resolve(strict=True)
                    if not candidate.is_relative_to(root) or not candidate.is_file():
                        raise ValueError(f"manifest shard escapes data root: {record['path']}")
                    if candidate.stat().st_size != record.get("bytes") or sha256_file(
                        candidate
                    ) != record.get("sha256"):
                        raise ValueError(f"manifest shard identity mismatch: {record['path']}")
                    files.append(str(candidate))
                if not files or manifest.get("file_count") != len(files):
                    raise ValueError("manifest has no complete shard set")
                return str(manifest.get("format") or "json"), files
        dataset_format = "parquet" if path.suffix.lower() == ".parquet" else "json"
        return dataset_format, [str(path)]
    if not path.is_dir():
        raise ValueError(f"local dataset does not exist: {path}")
    parquet = sorted(path.glob("*.parquet"))
    json_files = sorted((*path.glob("*.jsonl"), *path.glob("*.json")))
    json_files = [candidate for candidate in json_files if candidate.name != "MANIFEST.json"]
    if parquet and json_files:
        raise ValueError(f"local dataset mixes Parquet and JSON shards: {path}")
    files = parquet or json_files
    if not files:
        raise ValueError(f"local dataset has no supported shards: {path}")
    return ("parquet" if parquet else "json"), [str(candidate) for candidate in files]


def _strip_thinking(content: str) -> str:
    """Strip <think>...</think> blocks from assistant message content.

    Used to clean intermediate assistant turns before they are appended to the
    context for the next generation step.  Only the final assistant turn in a
    multi-turn conversation should retain the full reasoning trace.
    """
    return re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL).strip()


class LLM:
    def __init__(self, args):
        from openai import OpenAI

        self.args = args
        self._pid = os.getpid()
        self._client_type = OpenAI
        self.client = OpenAI(base_url=args.base_url)
        self.last_completion_tokens: int | None = None
        self.generate(messages=[{"role": "user", "content": "Hello! /no_think"}], verbose=True)

    def _ensure_client(self):
        """Reinitialize the HTTP client if we've been forked into a new process.

        datasets.map(num_proc>1) forks worker processes that inherit the parent's
        connection pool.  Reusing inherited sockets across processes causes
        "Invalid HTTP request" errors.  Creating a fresh client per-process avoids this.
        """
        if os.getpid() != self._pid:
            self._pid = os.getpid()
            self.client = self._client_type(base_url=self.args.base_url)

    def generate(self, messages, verbose=False, **chat_template_kwargs):
        global early_termination
        self._ensure_client()
        try:
            completion = self.client.chat.completions.create(
                model=self.args.model,
                messages=messages,
                temperature=self.args.temperature,
                max_tokens=self.args.max_tokens,
            )
            usage = getattr(completion, "usage", None)
            completion_tokens = getattr(usage, "completion_tokens", None)
            self.last_completion_tokens = (
                completion_tokens
                if isinstance(completion_tokens, int) and not isinstance(completion_tokens, bool)
                else None
            )
            new_message = completion.choices[0].message.content
            if verbose:
                for msg in messages:
                    print("[OLD] {:10}: {:64}".format(msg["role"], msg["content"]))
                print("[NEW] {:10}: {:64}\n\n".format("assistant", new_message))

            new_message = {"role": "assistant", "content": new_message}
        except Exception as e:
            print(e)
            if "Connection error" in str(e):
                early_termination = True
            raise  # always propagate so datasets.map() halts the shard

        return new_message


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="query")
    parser.add_argument("base_url", type=str, help="url to the OpenAI compatible API.")
    parser.add_argument("model", type=str, help="model name")
    parser.add_argument(
        "--data", type=str, default=None, help="path to OAI chat data (local or HF hub)"
    )
    parser.add_argument("--data-split", type=str, default="train", help="HF dataset split")
    parser.add_argument(
        "--save", type=str, default=None, help="path to store the generated output."
    )
    parser.add_argument("--num-shards", type=int, default=1000, help="number of shards.")
    parser.add_argument(
        "--strict-num-shards",
        action="store_true",
        help="Preserve the configured shard topology instead of reducing small datasets.",
    )
    parser.add_argument("--shard-id", type=int, default=None, help="single shard id to process.")
    parser.add_argument("--shard-id-begin", type=int, default=0, help="the shard id to start.")
    parser.add_argument(
        "--shard-id-step", type=int, default=1, help="the step that the shard id progress."
    )
    parser.add_argument(
        "--num-samples",
        "--num_samples",
        type=int,
        default=None,
        help="maximum samples to process.",
    )
    parser.add_argument(
        "--num-proc", type=int, default=32, help="number of processes (concurrency)."
    )
    parser.add_argument("--temperature", type=float, default=0.0, help="temperature.")
    parser.add_argument(
        "--max-tokens", type=int, default=None, help="maximum tokens to generate per response."
    )
    parser.add_argument(
        "--max-total-length",
        type=int,
        default=8192,
        help="maximum total length (prompt + output). Stops synthesizing remaining turns "
        "when context exceeds this limit.",
    )
    parser.add_argument(
        "--thinking-mode",
        choices=["on", "off", "source"],
        default="source",
        help="Force thinking on/off or honor each source row's enable_thinking value.",
    )
    parser.add_argument(
        "--source-id",
        default=None,
        help="Immutable source manifest identifier recorded in each shard sidecar.",
    )
    parser.add_argument(
        "--target-revision",
        default=None,
        help="Immutable target revision recorded in each shard sidecar.",
    )
    parser.add_argument(
        "--reject-tool-trajectories",
        action="store_true",
        help="Fail if target synthesis receives tools, tool calls, or tool results.",
    )
    parser.add_argument(
        "--record-assistant-tokens",
        action="store_true",
        help="Record exact API-reported completion tokens in each synthesized row.",
    )
    return parser


def synthesize(data):
    messages = prepare_generation_messages(
        data,
        args.thinking_mode,
        shard_id=-1,
        reject_tool_trajectories=args.reject_tool_trajectories,
    )
    enable_thinking = resolve_thinking_mode(args.thinking_mode, data)

    current_messages = []
    output_messages = []
    assistant_tokens = 0
    max_total = args.max_total_length

    for msg in messages:
        role = msg["role"]
        if role == "system":
            current_messages.append(msg)
            output_messages.append(msg)
        elif role == "user":
            current_messages.append(msg)

            # Estimate context length; stop if remaining budget is too small.
            if max_total is not None and args.max_tokens is not None:
                ctx_chars = sum(len(m.get("content", "")) for m in current_messages)
                est_tokens = ctx_chars // 3  # rough char-to-token estimate
                if est_tokens + args.max_tokens > max_total:
                    # Drop this user turn — context too long for another generation
                    current_messages.pop()
                    break

            output_messages.append(msg)

            new_message = llm.generate(current_messages, verbose=False)
            if new_message is None:
                break
            if args.record_assistant_tokens:
                completion_tokens = llm.last_completion_tokens
                if (
                    not isinstance(completion_tokens, int)
                    or isinstance(completion_tokens, bool)
                    or completion_tokens < 0
                ):
                    raise ValueError("generation response lacks exact completion-token usage")
                assistant_tokens += completion_tokens
            output_messages.append(new_message)

            if enable_thinking:
                # Append a thinking-stripped copy as context for the next turn.
                # Multi-turn reasoning: only the *last* assistant turn should
                # retain the full <think>...</think> trace; prior turns are
                # already resolved and the trace would distract the model.
                # The full trace is restored to the last turn after the loop.
                stripped = {
                    "role": "assistant",
                    "content": _strip_thinking(new_message["content"]),
                }
                current_messages.append(stripped)
            else:
                current_messages.append(new_message)
        elif role == "developer":
            # Map developer-role messages to system per OpenAI schema conventions.
            mapped = {"role": "system", "content": msg["content"]}
            current_messages.append(mapped)
            output_messages.append(mapped)
        elif role == "assistant":
            # Original assistant messages are not used — the model generates fresh responses.
            pass
        elif role == "tool":
            # Tool turns are not sent to the generation model — skip them.
            pass
        else:
            raise ValueError(f"Unexpected message role {role!r} in conversation.")

    result = {"messages": output_messages}
    if args.record_assistant_tokens:
        result["_synthesis_assistant_tokens"] = assistant_tokens
    return result


def main(argv: list[str] | None = None) -> int:
    global args, llm
    parser = build_parser()
    args = parser.parse_args(argv)
    llm = LLM(args)
    if args.data is None:
        return 0
    if args.save is None:
        parser.error("--save is required when --data is provided")

    if os.path.exists(args.data):
        fmt, data_files = resolve_local_dataset(Path(args.data))
        dataset = load_dataset(fmt, data_files={"train": data_files}, split=args.data_split)
    else:
        dataset = load_dataset(args.data, split=args.data_split)

    if args.strict_num_shards and args.num_shards > len(dataset):
        parser.error("--strict-num-shards requires at least one row per shard")
    if (
        not args.strict_num_shards
        and args.shard_id is None
        and args.num_shards * 100 > len(dataset)
    ):
        args.num_shards = max(1, min(16, len(dataset) // 100))
    if args.num_samples is not None:
        dataset = dataset.select(range(min(args.num_samples, len(dataset))))
    if args.shard_id is not None and not (0 <= args.shard_id < args.num_shards):
        parser.error(f"--shard-id {args.shard_id} out of range [0, {args.num_shards})")

    print(f"Create save dir: {args.save}")
    os.makedirs(args.save, exist_ok=True)
    shard_ids = (
        [args.shard_id]
        if args.shard_id is not None
        else range(args.shard_id_begin, args.num_shards, args.shard_id_step)
    )

    for shard_id in shard_ids:
        if args.shard_id is None:
            file_path = Path(args.save) / f"train-{shard_id + 1:05}-{args.num_shards:05}.jsonl"
        else:
            file_path = Path(args.save) / f"shard_{shard_id}.jsonl"
        metadata_path = file_path.with_suffix(file_path.suffix + ".metadata.json")
        done_path = file_path.with_suffix(file_path.suffix + ".done")
        if file_path.exists() or metadata_path.exists() or done_path.exists():
            verify_completed_shard(
                file_path,
                metadata_path,
                done_path,
                expected={
                    "shard_id": shard_id,
                    "num_shards": args.num_shards,
                    "thinking_mode": args.thinking_mode,
                    "source_id": args.source_id or args.data,
                    "target_revision": args.target_revision or args.model,
                    "temperature": args.temperature,
                    "max_tokens": args.max_tokens,
                    "max_total_length": args.max_total_length,
                },
            )
            continue

        shard = dataset.shard(num_shards=args.num_shards, index=shard_id)
        print(len(shard), file_path)
        num_proc = min(args.num_proc, len(shard))
        updated_shard = shard.map(synthesize, num_proc=num_proc)
        updated_shard.to_json(str(file_path))
        metadata = build_shard_metadata(
            output_path=file_path,
            shard_id=shard_id,
            num_shards=args.num_shards,
            thinking_mode=args.thinking_mode,
            source_id=args.source_id or args.data,
            target_revision=args.target_revision or args.model,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            max_total_length=args.max_total_length,
        )
        _write_json_atomic(metadata_path, metadata)
        done_path.write_text("done\n", encoding="utf-8")
        print(updated_shard[0])

        if early_termination:
            print("Terminate earlier due to server connection error!")
            break
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
