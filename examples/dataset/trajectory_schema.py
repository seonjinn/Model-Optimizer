# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""Canonical schema and integrity checks for tool-using training trajectories."""

from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from typing import Any


def _canonical_tool_call(call: dict[str, Any], source_id: str) -> dict[str, Any]:
    call_id = str(call.get("id") or call.get("tool_call_id") or "")
    if not call_id:
        raise ValueError(f"{source_id}: assistant tool call has no ID")
    function = call.get("function")
    if not isinstance(function, dict) or not function.get("name"):
        raise ValueError(f"{source_id}: tool call {call_id} has no function name")
    return {
        "id": call_id,
        "type": str(call.get("type") or "function"),
        "function": deepcopy(function),
    }


def canonicalize_trajectory(example: dict[str, Any], *, source_id: str) -> dict[str, Any]:
    messages = example.get("messages") or example.get("conversations")
    if not isinstance(messages, list) or not messages:
        raise ValueError(f"{source_id}: trajectory has no messages")

    tools = deepcopy(example.get("tools") or [])
    normalized: list[dict[str, Any]] = []
    pending: set[str] = set()
    seen_calls: set[str] = set()

    for index, original in enumerate(messages):
        if not isinstance(original, dict):
            raise ValueError(f"{source_id}: message {index} is not a mapping")
        role = original.get("role")
        if role != "tool" and pending:
            unresolved = ", ".join(sorted(pending))
            raise ValueError(
                f"{source_id}: unresolved tool call(s) {unresolved} before message {index}"
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
                _canonical_tool_call(call, source_id) for call in (original.get("tool_calls") or [])
            ]
            for call in calls:
                call_id = call["id"]
                if call_id in seen_calls:
                    raise ValueError(f"{source_id}: duplicate tool call ID {call_id}")
                seen_calls.add(call_id)
                pending.add(call_id)
            if calls:
                if not tools:
                    raise ValueError(
                        f"{source_id}: tool calls are present but top-level tools are missing"
                    )
                message["tool_calls"] = calls
            normalized.append(message)
            continue

        if role == "tool":
            call_id = str(original.get("tool_call_id") or original.get("id") or "")
            if not call_id or call_id not in pending:
                raise ValueError(f"{source_id}: tool result references missing call {call_id!r}")
            message = {
                "role": "tool",
                "content": original.get("content") or "",
                "tool_call_id": call_id,
            }
            if original.get("name"):
                message["name"] = original["name"]
            normalized.append(message)
            pending.remove(call_id)
            continue

        raise ValueError(f"{source_id}: unsupported message role {role!r} at index {index}")

    if pending:
        unresolved = ", ".join(sorted(pending))
        raise ValueError(f"{source_id}: unresolved tool call(s) {unresolved} at end of trajectory")

    return {"source_id": source_id, "tools": tools, "messages": normalized}


def trajectory_digest(trajectory: dict[str, Any]) -> str:
    payload = {key: value for key, value in trajectory.items() if key != "trajectory_sha256"}
    encoded = json.dumps(payload, ensure_ascii=False, separators=(",", ":"), sort_keys=True).encode(
        "utf-8"
    )
    return hashlib.sha256(encoded).hexdigest()


def normalize_trajectory_for_dataset(example: dict[str, Any], *, source_id: str) -> dict[str, Any]:
    canonical = canonicalize_trajectory(example, source_id=source_id)
    canonical["trajectory_sha256"] = trajectory_digest(canonical)
    return canonical
