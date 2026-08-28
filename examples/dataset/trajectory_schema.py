# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""Canonical schema and integrity checks for tool-using training trajectories."""

from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from dataclasses import dataclass
from typing import Any, NoReturn

from specdec_corpus_contracts import canonical_json
from specdec_identity import normalize_storage_fields

__all__ = [
    "TrajectoryValidation",
    "TrajectoryValidationError",
    "canonicalize_trajectory",
    "normalize_trajectory_for_dataset",
    "quarantine_reason",
    "trajectory_digest",
    "validate_trajectory",
]

_REPLAY_LANES = frozenset({"generic-tool-replay", "interactive-swe-replay"})


class TrajectoryValidationError(ValueError):
    """A replay trajectory rejection with a stable quarantine code."""

    def __init__(self, reason: str, detail: str) -> None:
        self.reason = reason
        super().__init__(detail)


@dataclass(frozen=True)
class TrajectoryValidation:
    """Canonical replay data proven safe for deterministic truncation."""

    canonical: dict[str, Any]
    canonical_bytes: bytes
    tool_call_count: int


def _reject(reason: str, detail: str) -> NoReturn:
    raise TrajectoryValidationError(reason, detail)


def _validate_tool_call(call: Any, source_id: str, declared_functions: set[str]) -> tuple[str, str]:
    if not isinstance(call, dict):
        _reject("invalid_tool_call", f"{source_id}: assistant tool call is not a mapping")
    call_id = str(call.get("id") or call.get("tool_call_id") or "")
    if not call_id:
        _reject("missing_tool_call_id", f"{source_id}: assistant tool call has no ID")
    call_type = call.get("type", "function")
    if call_type != "function":
        _reject(
            "unsupported_tool_call_type",
            f"{source_id}: tool call {call_id} has unsupported type {call_type!r}",
        )
    function = call.get("function")
    if not isinstance(function, dict) or not function.get("name"):
        _reject("missing_function_name", f"{source_id}: tool call {call_id} has no function name")
    name = str(function["name"])
    if name not in declared_functions:
        _reject("undeclared_function", f"{source_id}: undeclared function {name}")
    arguments = function.get("arguments", "{}")
    if not isinstance(arguments, str):
        _reject("malformed_arguments", f"{source_id}: malformed arguments for {call_id}")
    try:
        parsed_arguments = json.loads(arguments)
    except json.JSONDecodeError:
        _reject("malformed_arguments", f"{source_id}: malformed arguments for {call_id}")
    if not isinstance(parsed_arguments, dict):
        _reject("malformed_arguments", f"{source_id}: malformed arguments for {call_id}")
    return call_id, name


def _declared_functions(tools: list[Any], source_id: str) -> set[str]:
    declared: set[str] = set()
    for index, tool in enumerate(tools):
        if not isinstance(tool, dict) or not isinstance(tool.get("function"), dict):
            _reject(
                "invalid_tool_declaration", f"{source_id}: tool declaration {index} is malformed"
            )
        function = tool["function"]
        name = function.get("name")
        if not isinstance(name, str) or not name:
            _reject(
                "invalid_tool_declaration", f"{source_id}: tool declaration {index} has no name"
            )
        if name in declared:
            _reject("duplicate_tool_declaration", f"{source_id}: duplicate tool declaration {name}")
        declared.add(name)
    return declared


def _validated_tool_calls_by_message(
    messages: list[dict[str, Any]], source_id: str
) -> list[list[dict[str, Any]]]:
    validated: list[list[dict[str, Any]]] = []
    for index, message in enumerate(messages):
        raw_calls = message.get("tool_calls")
        if raw_calls is None:
            validated.append([])
            continue
        if not isinstance(raw_calls, list):
            _reject(
                "invalid_tool_calls",
                f"{source_id}: message {index} tool_calls must be a list",
            )
        calls: list[dict[str, Any]] = []
        for call in raw_calls:
            if not isinstance(call, dict):
                _reject(
                    "invalid_tool_call",
                    f"{source_id}: message {index} tool call is not a mapping",
                )
            calls.append(call)
        validated.append(calls)
    return validated


def _storage_normalized_trajectory(
    example: dict[str, Any], source_id: str
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], bytes]:
    messages = example.get("messages") if "messages" in example else example.get("conversations")
    if not isinstance(messages, list) or not messages:
        _reject("missing_messages", f"{source_id}: trajectory has no messages")
    if not all(isinstance(message, dict) for message in messages):
        _reject("invalid_message", f"{source_id}: trajectory message is not a mapping")
    tools = example.get("tools")
    if tools is None:
        tools = []
    if not isinstance(tools, list):
        _reject("invalid_tool_declarations", f"{source_id}: tools must be a list")
    canonical_bytes = canonical_json(
        {
            "messages": normalize_storage_fields(deepcopy(messages)),
            "tools": normalize_storage_fields(deepcopy(tools)),
        }
    )
    payload = json.loads(canonical_bytes)
    return payload["messages"], payload["tools"], canonical_bytes


def _validate_referential_integrity(
    messages: list[dict[str, Any]], tools: list[dict[str, Any]], source_id: str
) -> int:
    calls_by_message = _validated_tool_calls_by_message(messages, source_id)
    declared_functions = _declared_functions(tools, source_id)
    pending: dict[str, str] = {}
    seen_calls: set[str] = set()
    tool_call_count = 0

    for index, message in enumerate(messages):
        role = message.get("role")
        if role != "tool" and pending:
            unresolved = ", ".join(sorted(pending))
            _reject(
                "unresolved_tool_call",
                f"{source_id}: unresolved tool call(s) {unresolved} before message {index}",
            )
        if role in {"system", "user", "developer"}:
            continue
        if role == "assistant":
            calls = calls_by_message[index]
            for call in calls:
                call_id, function_name = _validate_tool_call(call, source_id, declared_functions)
                if call_id in seen_calls:
                    _reject(
                        "duplicate_tool_call_id",
                        f"{source_id}: duplicate tool call ID {call_id}",
                    )
                seen_calls.add(call_id)
                pending[call_id] = function_name
                tool_call_count += 1
            continue
        if role == "tool":
            call_id = str(message.get("tool_call_id") or message.get("id") or "")
            if not call_id or call_id not in pending:
                _reject(
                    "orphan_tool_result",
                    f"{source_id}: tool result references missing call {call_id!r}",
                )
            result_name = message.get("name")
            if result_name is not None and str(result_name) != pending[call_id]:
                _reject(
                    "tool_result_name_mismatch",
                    f"{source_id}: tool result {call_id} names the wrong function",
                )
            del pending[call_id]
            continue
        _reject(
            "unsupported_role", f"{source_id}: unsupported message role {role!r} at index {index}"
        )

    if pending:
        unresolved = ", ".join(sorted(pending))
        _reject(
            "unresolved_tool_call",
            f"{source_id}: unresolved tool call(s) {unresolved} at end of trajectory",
        )
    return tool_call_count


def _validated_native_trajectory(
    example: dict[str, Any], source_id: str
) -> tuple[dict[str, Any], bytes, int]:
    messages, tools, canonical_bytes = _storage_normalized_trajectory(example, source_id)
    tool_call_count = _validate_referential_integrity(messages, tools, source_id)
    canonical = {"source_id": source_id, "tools": tools, "messages": messages}
    return canonical, canonical_bytes, tool_call_count


def canonicalize_trajectory(example: dict[str, Any], *, source_id: str) -> dict[str, Any]:
    canonical, _, _ = _validated_native_trajectory(example, source_id)
    return canonical


def trajectory_digest(trajectory: dict[str, Any]) -> str:
    payload = {key: value for key, value in trajectory.items() if key != "trajectory_sha256"}
    encoded = json.dumps(payload, ensure_ascii=False, separators=(",", ":"), sort_keys=True).encode(
        "utf-8"
    )
    return hashlib.sha256(encoded).hexdigest()


def _token_count(
    tokenizer: Any, messages: list[dict[str, Any]], tools: list[dict[str, Any]]
) -> int:
    encoded = tokenizer.apply_chat_template(
        messages,
        tools=tools or None,
        tokenize=True,
        add_generation_prompt=False,
        return_dict=True,
    )
    input_ids = encoded.get("input_ids") if isinstance(encoded, dict) else encoded
    if not isinstance(input_ids, list):
        _reject("invalid_tokenization", "tokenizer did not return input IDs")
    return len(input_ids)


def _tool_transactions(messages: list[dict[str, Any]]) -> list[tuple[int, int]]:
    transactions: list[tuple[int, int]] = []
    pending: set[str] = set()
    start = -1
    for index, message in enumerate(messages):
        raw_calls = message.get("tool_calls") if message.get("role") == "assistant" else None
        calls = raw_calls if isinstance(raw_calls, list) else []
        if calls:
            start = index
            pending = {
                str(call.get("id") or call.get("tool_call_id"))
                for call in calls
                if isinstance(call, dict)
            }
        if message["role"] == "tool":
            call_id = str(message.get("tool_call_id") or message.get("id"))
            pending.remove(call_id)
            if not pending:
                transactions.append((start, index))
    return transactions


def _validate_reasoning_mode(
    example: dict[str, Any], source_id: str, expected_mode: str | None
) -> None:
    declared_mode = example.get("reasoning_mode")
    valid_modes = {None, "reasoning_on", "reasoning_off"}
    if declared_mode not in valid_modes or expected_mode not in valid_modes:
        mode = declared_mode if declared_mode not in valid_modes else expected_mode
        _reject("reasoning_mode_mismatch", f"{source_id}: unsupported reasoning mode {mode!r}")
    if expected_mode is not None and declared_mode not in {None, expected_mode}:
        _reject(
            "reasoning_mode_mismatch",
            f"{source_id}: declared reasoning mode does not match {expected_mode}",
        )
    mode = expected_mode or declared_mode
    messages = example.get("messages") or example.get("conversations") or []
    has_reasoning = any(
        isinstance(message, dict)
        and isinstance(message.get("reasoning_content"), str)
        and bool(message["reasoning_content"])
        for message in messages
    )
    if (mode == "reasoning_on" and not has_reasoning) or (
        mode == "reasoning_off" and has_reasoning
    ):
        _reject(
            "reasoning_mode_mismatch",
            f"{source_id}: reasoning content does not match declared mode {mode}",
        )


def validate_trajectory(
    example: dict[str, Any],
    *,
    source_id: str,
    lane: str,
    tokenizer: Any | None = None,
    training_seq_len: int = 4_096,
    reasoning_mode: str | None = None,
) -> TrajectoryValidation:
    """Canonicalize and validate one replay-lane tool trajectory."""
    if lane not in _REPLAY_LANES:
        _reject("unsupported_replay_lane", f"{source_id}: unsupported replay lane {lane!r}")
    if training_seq_len < 1:
        raise ValueError("training_seq_len must be positive")
    _validate_reasoning_mode(example, source_id, reasoning_mode)
    canonical, canonical_bytes, tool_call_count = _validated_native_trajectory(example, source_id)
    if tool_call_count == 0:
        _reject("no_assistant_tool_call", f"{source_id}: replay has no assistant tool call")
    if tokenizer is not None:
        for start, end in _tool_transactions(canonical["messages"]):
            before = _token_count(tokenizer, canonical["messages"][:start], canonical["tools"])
            after = _token_count(tokenizer, canonical["messages"][: end + 1], canonical["tools"])
            if before < training_seq_len < after:
                _reject(
                    "split_tool_transaction",
                    f"{source_id}: {training_seq_len}-token cut splits a tool transaction",
                )
    return TrajectoryValidation(
        canonical=canonical,
        canonical_bytes=canonical_bytes,
        tool_call_count=tool_call_count,
    )


def quarantine_reason(
    example: dict[str, Any],
    *,
    source_id: str,
    lane: str,
    tokenizer: Any | None = None,
    training_seq_len: int = 4_096,
    reasoning_mode: str | None = None,
) -> str | None:
    """Return the stable quarantine code for a replay, or ``None`` when valid."""
    try:
        validate_trajectory(
            example,
            source_id=source_id,
            lane=lane,
            tokenizer=tokenizer,
            training_seq_len=training_seq_len,
            reasoning_mode=reasoning_mode,
        )
    except TrajectoryValidationError as error:
        return error.reason
    return None


def normalize_trajectory_for_dataset(example: dict[str, Any], *, source_id: str) -> dict[str, Any]:
    canonical = canonicalize_trajectory(example, source_id=source_id)
    canonical["trajectory_sha256"] = trajectory_digest(canonical)
    return canonical
