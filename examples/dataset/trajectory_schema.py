# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""Canonical schema and integrity checks for tool-using training trajectories."""

from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from dataclasses import dataclass
from typing import Any, NoReturn

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
    tool_call_count: int


def _reject(reason: str, detail: str) -> NoReturn:
    raise TrajectoryValidationError(reason, detail)


def _canonical_tool_call(
    call: dict[str, Any], source_id: str, declared_functions: set[str]
) -> dict[str, Any]:
    call_id = str(call.get("id") or call.get("tool_call_id") or "")
    if not call_id:
        _reject("missing_tool_call_id", f"{source_id}: assistant tool call has no ID")
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
    return {
        "id": call_id,
        "type": str(call.get("type") or "function"),
        "function": deepcopy(function),
    }


def canonicalize_trajectory(example: dict[str, Any], *, source_id: str) -> dict[str, Any]:
    messages = example.get("messages") or example.get("conversations")
    if not isinstance(messages, list) or not messages:
        _reject("missing_messages", f"{source_id}: trajectory has no messages")

    tools = deepcopy(example.get("tools") or [])
    if not isinstance(tools, list):
        _reject("invalid_tool_declarations", f"{source_id}: tools must be a list")
    declared_functions = {
        str(tool["function"]["name"])
        for tool in tools
        if isinstance(tool, dict)
        and isinstance(tool.get("function"), dict)
        and tool["function"].get("name")
    }
    normalized: list[dict[str, Any]] = []
    pending: dict[str, str] = {}
    seen_calls: set[str] = set()

    for index, original in enumerate(messages):
        if not isinstance(original, dict):
            _reject("invalid_message", f"{source_id}: message {index} is not a mapping")
        role = original.get("role")
        if role != "tool" and pending:
            unresolved = ", ".join(sorted(pending))
            _reject(
                "unresolved_tool_call",
                f"{source_id}: unresolved tool call(s) {unresolved} before message {index}",
            )

        if role in {"system", "user", "developer"}:
            normalized.append(
                {
                    "role": "system" if role == "developer" else role,
                    "content": original.get("content") or "",
                }
            )
            continue

        if role == "assistant":
            message: dict[str, Any] = {
                "role": "assistant",
                "content": original.get("content") or "",
            }
            if "reasoning_content" in original:
                message["reasoning_content"] = original.get("reasoning_content") or ""
            calls = [
                _canonical_tool_call(call, source_id, declared_functions)
                for call in (original.get("tool_calls") or [])
            ]
            for call in calls:
                call_id = call["id"]
                if call_id in seen_calls:
                    _reject(
                        "duplicate_tool_call_id",
                        f"{source_id}: duplicate tool call ID {call_id}",
                    )
                seen_calls.add(call_id)
                pending[call_id] = str(call["function"]["name"])
            if calls:
                if not tools:
                    _reject(
                        "missing_tool_declarations",
                        f"{source_id}: tool calls are present but top-level tools are missing",
                    )
                message["tool_calls"] = calls
            normalized.append(message)
            continue

        if role == "tool":
            call_id = str(original.get("tool_call_id") or original.get("id") or "")
            if not call_id or call_id not in pending:
                _reject(
                    "orphan_tool_result",
                    f"{source_id}: tool result references missing call {call_id!r}",
                )
            result_name = original.get("name")
            if result_name is not None and str(result_name) != pending[call_id]:
                _reject(
                    "tool_result_name_mismatch",
                    f"{source_id}: tool result {call_id} names the wrong function",
                )
            message = {
                "role": "tool",
                "content": original.get("content") or "",
                "tool_call_id": call_id,
            }
            if original.get("name"):
                message["name"] = original["name"]
            normalized.append(message)
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

    return {"source_id": source_id, "tools": tools, "messages": normalized}


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
        calls = message.get("tool_calls") or []
        if calls:
            start = index
            pending = {str(call["id"]) for call in calls}
        if message["role"] == "tool":
            pending.remove(str(message["tool_call_id"]))
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
    canonical = canonicalize_trajectory(example, source_id=source_id)
    tool_call_count = sum(len(message.get("tool_calls") or []) for message in canonical["messages"])
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
    return TrajectoryValidation(canonical=canonical, tool_call_count=tool_call_count)


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
