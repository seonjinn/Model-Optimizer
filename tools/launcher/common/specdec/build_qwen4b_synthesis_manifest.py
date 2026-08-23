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

"""Immutable manifests for Qwen3-4B response synthesis and trace replay."""

from __future__ import annotations

import json
import os
import re
import tempfile
from dataclasses import asdict, dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any

__all__ = [
    "Qwen4BSynthesisManifest",
    "SynthesisInputs",
    "SynthesisSettings",
    "load_synthesis_manifest",
    "tokenizer_snapshot_sha256",
    "verify_data_manifest",
    "verify_synthesis_completion",
    "write_synthesis_manifest",
]

_SHA = re.compile(r"^[0-9a-f]{40}$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_ARMS = frozenset({"B", "C", "D"})
_THINKING_MODES = frozenset({"on", "off"})
_RESPONSE_SOURCES = frozenset({"target-synth", "trace-replay"})


def _path_under(name: str, value: str, root: str) -> str:
    if not value or not Path(value).is_absolute():
        raise ValueError(f"{name} must be an absolute path under {root}")
    normalized = Path(os.path.abspath(value))
    resolved = normalized.resolve(strict=False)
    resolved_root = Path(root).resolve(strict=False)
    if not resolved.is_relative_to(resolved_root) or resolved == resolved_root:
        raise ValueError(f"{name} must be a strict descendant of {root}")
    return str(normalized)


def _validate_sha(name: str, value: str, pattern: re.Pattern[str]) -> None:
    if not pattern.fullmatch(value):
        raise ValueError(f"{name} must be an exact lowercase digest")


def _sha256_file(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tokenizer_snapshot_sha256(root: Path) -> str:
    """Hash the tokenizer serialization independently of model weights."""
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
        raise ValueError("target snapshot contains no tokenizer files")
    digest = sha256()
    for path in files:
        digest.update(path.name.encode())
        digest.update(b"\0")
        digest.update(bytes.fromhex(_sha256_file(path)))
    return digest.hexdigest()


def _verified_files(root: Path, records: Any, *, label: str) -> tuple[Path, ...]:
    if not isinstance(records, list) or not records:
        raise ValueError(f"{label} manifest must contain files")
    verified: list[Path] = []
    seen: set[str] = set()
    resolved_root = root.resolve(strict=True)
    for record in records:
        if not isinstance(record, dict):
            raise ValueError(f"invalid {label} file record")
        relative = record.get("path")
        if not isinstance(relative, str) or not relative or relative in seen:
            raise ValueError(f"invalid or duplicate {label} file path")
        seen.add(relative)
        candidate = root / relative
        try:
            resolved = candidate.resolve(strict=True)
        except OSError as error:
            raise ValueError(f"{label} file is missing: {relative}") from error
        if not resolved.is_relative_to(resolved_root) or not resolved.is_file():
            raise ValueError(f"{label} file escapes its root: {relative}")
        if record.get("bytes") != resolved.stat().st_size or record.get("sha256") != _sha256_file(
            resolved
        ):
            raise ValueError(f"{label} file identity mismatch: {relative}")
        verified.append(resolved)
    return tuple(verified)


def verify_data_manifest(path: Path, expected_sha256: str | None = None) -> tuple[Path, ...]:
    """Verify a materialized prompt/corpus manifest and return its exact shards."""
    data = path.read_bytes()
    if expected_sha256 is not None and sha256(data).hexdigest() != expected_sha256:
        raise ValueError("data manifest SHA-256 mismatch")
    payload = json.loads(data)
    if not isinstance(payload, dict) or payload.get("schema_version") != 1:
        raise ValueError("unsupported data manifest")
    files = _verified_files(path.parent, payload.get("files"), label="shard")
    if payload.get("file_count") != len(files):
        raise ValueError("data manifest file count mismatch")
    if not isinstance(payload.get("row_count"), int) or payload["row_count"] < 1:
        raise ValueError("data manifest row count must be positive")
    return files


def verify_synthesis_completion(
    root: Path,
    experiment_id: str,
    *,
    expected_output_token_budget: int,
    expected_tokenizer_sha256: str,
    expected_file_count: int,
    expected_generation_identity_sha256: str | None = None,
) -> dict[str, Any]:
    """Revalidate immutable synthesis output before reuse."""
    payload = json.loads((root / "completion.json").read_text(encoding="utf-8"))
    if payload.get("schema_version") != 1 or payload.get("experiment_id") != experiment_id:
        raise ValueError("synthesis completion identity mismatch")
    files = _verified_files(root, payload.get("files"), label="synthesis")
    immutable_contract = {
        "requested_output_token_budget": expected_output_token_budget,
        "tokenizer_sha256": expected_tokenizer_sha256,
        "file_count": expected_file_count,
    }
    if expected_generation_identity_sha256 is not None:
        immutable_contract["generation_identity_sha256"] = expected_generation_identity_sha256
        immutable_contract["promotion_status"] = "passed"
    if (
        any(payload.get(field) != value for field, value in immutable_contract.items())
        or len(files) != expected_file_count
    ):
        raise ValueError("immutable synthesis contract mismatch")
    requested = payload.get("requested_output_token_budget")
    actual = payload.get("actual_assistant_tokens")
    if (
        not isinstance(requested, int)
        or isinstance(requested, bool)
        or requested < 1
        or not isinstance(actual, int)
        or isinstance(actual, bool)
        or actual < requested
    ):
        raise ValueError("synthesis assistant-token budget mismatch")
    return payload


@dataclass(frozen=True)
class SynthesisInputs:
    """Pinned source, model, corpus, and output locations for one synthesis arm."""

    source_path: str
    source_sha: str
    image_path: str
    image_sha256: str
    runtime_archive_path: str
    runtime_archive_sha256: str
    target_path: str
    target_revision: str
    tokenizer_sha256: str
    prompt_manifest_path: str
    prompt_manifest_sha256: str
    output_root: str
    trace_manifest_path: str | None = None
    trace_manifest_sha256: str | None = None
    chat_template_sha256: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "source_path", _path_under("source_path", self.source_path, "/home")
        )
        for name in (
            "image_path",
            "runtime_archive_path",
            "target_path",
            "prompt_manifest_path",
            "output_root",
        ):
            object.__setattr__(self, name, _path_under(name, getattr(self, name), "/lustre"))
        if self.trace_manifest_path is not None:
            object.__setattr__(
                self,
                "trace_manifest_path",
                _path_under("trace_manifest_path", self.trace_manifest_path, "/lustre"),
            )
        _validate_sha("source_sha", self.source_sha, _SHA)
        _validate_sha("target_revision", self.target_revision, _SHA)
        for name in (
            "image_sha256",
            "runtime_archive_sha256",
            "tokenizer_sha256",
            "prompt_manifest_sha256",
        ):
            _validate_sha(name, getattr(self, name), _SHA256)
        if self.chat_template_sha256 is not None:
            _validate_sha("chat_template_sha256", self.chat_template_sha256, _SHA256)
        if (self.trace_manifest_path is None) != (self.trace_manifest_sha256 is None):
            raise ValueError("trace manifest path and SHA-256 must be provided together")
        if self.trace_manifest_sha256 is not None:
            _validate_sha("trace_manifest_sha256", self.trace_manifest_sha256, _SHA256)


@dataclass(frozen=True)
class SynthesisSettings:
    """Generation and scheduler settings for a four-GPU synthesis allocation."""

    account: str
    partition: str
    replicas: int
    tensor_parallel_size: int
    num_shards: int
    output_token_budget: int
    temperature: float
    max_tokens: int
    max_total_length: int

    def __post_init__(self) -> None:
        if self.account != "nemotron_sw_post":
            raise ValueError("Qwen3-4B study must not use nemotron_n3_post")
        if not self.partition.strip():
            raise ValueError("partition must be non-empty")
        if self.replicas != 4:
            raise ValueError("synthesis requires four replicas to own every GPU")
        if self.tensor_parallel_size != 1:
            raise ValueError("Qwen3-4B synthesis requires TP1")
        if self.num_shards < self.replicas or self.num_shards % self.replicas:
            raise ValueError("num_shards must be positive and divisible by four replicas")
        if min(self.output_token_budget, self.max_tokens, self.max_total_length) < 1:
            raise ValueError("token budgets must be positive")
        if self.max_tokens > self.max_total_length:
            raise ValueError("max_tokens cannot exceed max_total_length")
        if self.temperature < 0:
            raise ValueError("temperature must be non-negative")


@dataclass(frozen=True)
class Qwen4BSynthesisManifest:
    """One target mode, corpus arm, and response-source synthesis experiment."""

    arm: str
    thinking_mode: str
    response_source: str
    inputs: SynthesisInputs
    settings: SynthesisSettings

    def __post_init__(self) -> None:
        if self.arm not in _ARMS:
            raise ValueError(f"unsupported study arm: {self.arm}")
        if self.thinking_mode not in _THINKING_MODES:
            raise ValueError(f"unsupported thinking mode: {self.thinking_mode}")
        if self.response_source not in _RESPONSE_SOURCES:
            raise ValueError(f"unsupported response source: {self.response_source}")
        if self.response_source == "trace-replay" and self.inputs.trace_manifest_path is None:
            raise ValueError("trace replay requires a trace manifest")

    @property
    def experiment_id(self) -> str:
        """Return a stable identity over every immutable synthesis input."""
        return sha256(_canonical_bytes(asdict(self))).hexdigest()[:16]

    @property
    def generation_identity_sha256(self) -> str:
        """Bind every input that can change target-native completion bytes."""
        if self.inputs.chat_template_sha256 is None:
            raise ValueError("target synthesis requires an exact chat-template identity")
        identity = {
            "target_revision": self.inputs.target_revision,
            "tokenizer_sha256": self.inputs.tokenizer_sha256,
            "chat_template_sha256": self.inputs.chat_template_sha256,
            "runtime_sha256": self.inputs.runtime_archive_sha256,
            "container_sha256": self.inputs.image_sha256,
            "source_selection_sha256": self.inputs.prompt_manifest_sha256,
            "thinking_mode": self.thinking_mode,
            "temperature": self.settings.temperature,
            "max_tokens": self.settings.max_tokens,
            "max_total_length": self.settings.max_total_length,
        }
        return sha256(_canonical_bytes(identity)).hexdigest()


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def write_synthesis_manifest(path: Path, manifests: list[Qwen4BSynthesisManifest]) -> None:
    """Atomically write canonical manifests with derived experiment identities."""
    if not manifests:
        raise ValueError("at least one synthesis manifest is required")
    experiment_ids = [manifest.experiment_id for manifest in manifests]
    if len(set(experiment_ids)) != len(experiment_ids):
        raise ValueError("synthesis experiment IDs must be unique")
    payload = [
        asdict(manifest) | {"experiment_id": manifest.experiment_id} for manifest in manifests
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", dir=path.parent, prefix=f".{path.name}.", delete=False, encoding="utf-8"
    ) as output:
        json.dump(payload, output, indent=2, sort_keys=True)
        output.write("\n")
        temporary = Path(output.name)
    os.replace(temporary, path)


def load_synthesis_manifest(path: Path) -> list[Qwen4BSynthesisManifest]:
    """Load and verify a canonical synthesis manifest file."""
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, list) or not raw:
        raise ValueError("synthesis manifest must be a non-empty list")
    manifests: list[Qwen4BSynthesisManifest] = []
    for entry in raw:
        expected_id = entry.pop("experiment_id", None)
        manifest = Qwen4BSynthesisManifest(
            arm=entry["arm"],
            thinking_mode=entry["thinking_mode"],
            response_source=entry["response_source"],
            inputs=SynthesisInputs(**entry["inputs"]),
            settings=SynthesisSettings(**entry["settings"]),
        )
        if expected_id != manifest.experiment_id:
            raise ValueError("synthesis experiment identity mismatch")
        manifests.append(manifest)
    return manifests
