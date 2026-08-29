# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Construct transaction-safe target-generation prefixes for Q30 agentic rows."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Literal, cast

from q30t_from_scratch_candidates import PromptCandidate
from q30t_generation_prompt_identity import generation_prompt_identity
from specdec_corpus_contracts import canonical_json, sha256_bytes
from specdec_identity import normalize_storage_fields
from trajectory_schema import TrajectoryValidationError, validate_trajectory

__all__ = [
    "AgenticGenerationPrefix",
    "AgenticPrefixError",
    "CanonicalToolCall",
    "accept_target_action",
    "agentic_generation_prefixes",
    "canonical_tool_call",
    "recorded_result_matches",
]

_REPLAY_LANE_BY_RESPONSE_LANE = {
    "general_tool": "generic-tool-replay",
    "interactive_swe": "interactive-swe-replay",
}


class AgenticPrefixError(ValueError):
    """An agentic action or source transaction cannot be used safely."""


@dataclass(frozen=True)
class CanonicalToolCall:
    """One function call with identity-independent canonical arguments."""

    call_id: str
    function_name: str
    arguments_bytes: bytes

    @property
    def function_bytes(self) -> bytes:
        """Return the function-and-arguments identity, excluding the call ID."""
        return canonical_json(
            {
                "name": self.function_name,
                "arguments": json.loads(self.arguments_bytes),
            }
        )


@dataclass(frozen=True)
class AgenticGenerationPrefix:
    """One response-free request plus source evidence for its next action."""

    candidate: PromptCandidate
    source_next_call: CanonicalToolCall | None
    recorded_result: dict[str, object] | None
    executable_environment: bool


def canonical_tool_call(call: object) -> CanonicalToolCall:
    """Normalize one JSON-object function call without trusting its call ID."""
    if not isinstance(call, Mapping):
        raise AgenticPrefixError("tool call must be a mapping")
    call_id = _call_id(call)
    call_type = call.get("type", "function")
    if call_type != "function":
        raise AgenticPrefixError(f"unsupported tool call type: {call_type!r}")
    function = call.get("function")
    if not isinstance(function, Mapping):
        raise AgenticPrefixError("tool call function must be a mapping")
    function_name = function.get("name")
    if not isinstance(function_name, str) or not function_name:
        raise AgenticPrefixError("tool call function name must be a non-empty string")
    arguments = function.get("arguments", {})
    if isinstance(arguments, str):
        try:
            parsed_arguments = json.loads(arguments)
        except json.JSONDecodeError as error:
            raise AgenticPrefixError("tool call arguments are not valid JSON") from error
    else:
        parsed_arguments = arguments
    if not isinstance(parsed_arguments, Mapping):
        raise AgenticPrefixError("tool call arguments must be a JSON object")
    try:
        arguments_bytes = canonical_json(dict(parsed_arguments))
    except (TypeError, ValueError) as error:
        raise AgenticPrefixError("tool call arguments are not JSON-serializable") from error
    return CanonicalToolCall(
        call_id=call_id,
        function_name=function_name,
        arguments_bytes=arguments_bytes,
    )


def recorded_result_matches(
    target_call: CanonicalToolCall, source_call: CanonicalToolCall
) -> bool:
    """Return whether a recorded result belongs to the generated function call."""
    return target_call.function_bytes == source_call.function_bytes


def accept_target_action(
    target_call: CanonicalToolCall,
    source_call: CanonicalToolCall,
    *,
    executable: bool,
) -> Literal["execute", "reuse-recorded"]:
    """Choose execution or proven-safe recorded-result reuse for a target call."""
    if type(executable) is not bool:
        raise AgenticPrefixError("executable must be a boolean")
    if executable:
        return "execute"
    if recorded_result_matches(target_call, source_call):
        return "reuse-recorded"
    raise AgenticPrefixError("divergent target action requires an executable environment")


def agentic_generation_prefixes(
    candidate: PromptCandidate,
    source_trajectory: Mapping[str, object],
    *,
    executable_environment: bool,
) -> tuple[AgenticGenerationPrefix, ...]:
    """Slice each assistant action after complete prior source transactions."""
    if type(executable_environment) is not bool:
        raise AgenticPrefixError("executable_environment must be a boolean")
    if candidate.identity.provenance.response_lane != candidate.lane:
        raise AgenticPrefixError("candidate response lane does not match authenticated provenance")
    replay_lane = _REPLAY_LANE_BY_RESPONSE_LANE.get(candidate.lane)
    if replay_lane is None:
        raise AgenticPrefixError(f"candidate response lane is not agentic: {candidate.lane!r}")
    trajectory = _decoded_trajectory(source_trajectory)
    _reconcile_candidate_source(candidate, trajectory)
    try:
        validation = validate_trajectory(
            trajectory,
            source_id=f"q30t-agentic:{candidate.identity.prompt_uuid}",
            lane=replay_lane,
            reasoning_mode=candidate.identity.provenance.reasoning_mode,
        )
    except TrajectoryValidationError as error:
        raise AgenticPrefixError(f"invalid source trajectory: {error}") from error

    messages = cast("list[dict[str, object]]", validation.canonical["messages"])
    tools = cast("list[dict[str, object]]", validation.canonical["tools"])
    prefixes: list[AgenticGenerationPrefix] = []
    for index, message in enumerate(messages):
        if message.get("role") != "assistant":
            continue
        raw_calls = message.get("tool_calls")
        calls = cast("list[object]", raw_calls) if isinstance(raw_calls, list) else []
        if len(calls) > 1:
            raise AgenticPrefixError(
                "ambiguous source action has multiple tool calls; recorded-result reuse "
                "requires one call"
            )
        source_next_call = canonical_tool_call(calls[0]) if calls else None
        recorded_result: dict[str, object] | None = None
        if source_next_call is not None:
            if index + 1 >= len(messages) or messages[index + 1].get("role") != "tool":
                raise AgenticPrefixError("source tool call has no adjacent recorded result")
            recorded_result = dict(messages[index + 1])
        prefix_candidate = _prefix_candidate(candidate, messages[:index], tools)
        prefixes.append(
            AgenticGenerationPrefix(
                candidate=prefix_candidate,
                source_next_call=source_next_call,
                recorded_result=recorded_result,
                executable_environment=executable_environment,
            )
        )
    return tuple(prefixes)


def _call_id(call: Mapping[str, object]) -> str:
    id_value = str(call.get("id") or "")
    tool_call_id_value = str(call.get("tool_call_id") or "")
    if id_value and tool_call_id_value and id_value != tool_call_id_value:
        raise AgenticPrefixError("tool call has ambiguous ID aliases")
    call_id = id_value or tool_call_id_value
    if not call_id:
        raise AgenticPrefixError("tool call ID must be non-empty")
    return call_id


def _decoded_trajectory(source_trajectory: Mapping[str, object]) -> dict[str, object]:
    trajectory = dict(source_trajectory)
    for field in ("messages", "conversations", "tools"):
        value = trajectory.get(field)
        if not isinstance(value, str):
            continue
        try:
            trajectory[field] = cast("object", json.loads(value))
        except json.JSONDecodeError as error:
            raise AgenticPrefixError(f"source {field} is not valid JSON") from error
    if trajectory.get("tools") is None:
        trajectory["tools"] = []
    return trajectory


def _reconcile_candidate_source(
    candidate: PromptCandidate, trajectory: dict[str, object]
) -> None:
    messages = trajectory.get("messages")
    tools = trajectory.get("tools")
    if not isinstance(messages, list) or any(not isinstance(item, dict) for item in messages):
        raise AgenticPrefixError("source trajectory messages must be a list of mappings")
    if not isinstance(tools, list) or any(not isinstance(item, dict) for item in tools):
        raise AgenticPrefixError("source trajectory tools must be a list of mappings")
    normalized_row = normalize_storage_fields({"messages": messages, "tools": tools})
    normalized_row_sha256 = sha256_bytes(canonical_json(normalized_row))
    if normalized_row_sha256 != candidate.normalized_row_sha256:
        raise AgenticPrefixError("candidate is not bound to the supplied source trajectory")
    source_prefix = messages[:-1] if messages[-1].get("role") == "assistant" else messages
    try:
        expected_identity = generation_prompt_identity(
            source_prefix,
            tools,
            candidate.identity.provenance,
        )
    except ValueError as error:
        raise AgenticPrefixError(f"invalid source trajectory prefix: {error}") from error
    expected_request_bytes = _request_bytes(expected_identity.canonical_bytes)
    if (
        candidate.source_assistant is not None
        or candidate.identity != expected_identity
        or candidate.request_bytes != expected_request_bytes
    ):
        raise AgenticPrefixError("candidate does not match the supplied source trajectory prefix")


def _prefix_candidate(
    candidate: PromptCandidate,
    messages: list[dict[str, object]],
    tools: list[dict[str, object]],
) -> PromptCandidate:
    identity = generation_prompt_identity(messages, tools, candidate.identity.provenance)
    return PromptCandidate(
        arm=candidate.arm,
        lane=candidate.lane,
        language=candidate.language,
        identity=identity,
        request_bytes=_request_bytes(identity.canonical_bytes),
        source_assistant=None,
        normalized_row_sha256=candidate.normalized_row_sha256,
    )


def _request_bytes(identity_bytes: bytes) -> bytes:
    payload = json.loads(identity_bytes)
    if not isinstance(payload, dict) or set(payload) != {"messages", "tools", "provenance"}:
        raise AssertionError("generation prompt identity payload is invalid")
    return canonical_json({"messages": payload["messages"], "tools": payload["tools"]})
