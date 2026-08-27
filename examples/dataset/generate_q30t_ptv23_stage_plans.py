# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Generate immutable Q30T source-stage plans from official Hugging Face metadata."""

from __future__ import annotations

import argparse
import json
import os
import re
import tempfile
import urllib.request
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

PTV2_REPOSITORY = "nvidia/Nemotron-Post-Training-Dataset-v2"
PTV2_REVISION = "5c89e01dd720ae0f4058445ed49c5fb68a03c76e"
PTV2_LICENSE = "CC-BY-4.0"
SWE_V3_REPOSITORY = "nvidia/Nemotron-SFT-SWE-v3"
SWE_V3_REVISION = "3f73de64c1fe928a8f538fe45ccc10c228cc4c6a"
SWE_V3_LICENSE = "CC-BY-4.0"
AGENTIC_REPOSITORY = "nvidia/Nemotron-Agentic-v1"
AGENTIC_REVISION = "650d590978ca35c8f1ecea2faf136e5fac421b62"
_TRACE_IDENTITIES: tuple[tuple[str, str, str, str], ...] = (
    (
        "nvidia/Nemotron-SFT-SWE-v2",
        "bd151f3f2d89c4804dda0083d912bd9f6a0a9fb7",
        "data/swe.jsonl",
        "CC-BY-4.0; Apache-2.0; MIT; BSD-3-Clause; BSD-2-Clause",
    ),
    (
        "nvidia/Nemotron-SWE-v1",
        "0fe17a965b297a9c943a59050a14c42d5f0083ce",
        "data/r2e_gym.jsonl",
        "CC-BY-4.0; BSD-3-Clause for indicated subsets; Apache-2.0; MIT",
    ),
    (
        AGENTIC_REPOSITORY,
        AGENTIC_REVISION,
        "data/interactive_agent.jsonl",
        "CC-BY-4.0; Apache-2.0 for Glaive-derived records",
    ),
    (
        AGENTIC_REPOSITORY,
        AGENTIC_REVISION,
        "data/tool_calling.jsonl",
        "CC-BY-4.0; Apache-2.0 for Glaive-derived records",
    ),
)
PTYCHE_CONTAINER_PATH = (
    "/lustre/fsw/coreai_dlalgo_llm/users/sna/modelopt-qwen3-drafter-training/"
    "assets/q235-training-prereqs-vllm0271-v1/image/"
    "vllm_openai_v0271_aarch64_20260813_2688476.sqsh"
)
PTYCHE_CONTAINER_SHA256 = "e7be53f2754097c88f7c801da92f6d94794ec4d78d9df937fcd315a6994297f0"
PTV2_SPLIT_COUNTS: Mapping[str, int] = {
    "chat": 12,
    "math": 2,
    "code": 2,
    "stem": 2,
    "multilingual_de": 38,
    "multilingual_ja": 37,
    "multilingual_es": 33,
    "multilingual_fr": 37,
    "multilingual_it": 38,
}
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_REDACTED_SHA256 = "*" * 64


def _positive_int(value: object, *, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"invalid {field}")
    return value


def _lfs_identity(record: Mapping[str, Any]) -> tuple[int, str]:
    lfs = record.get("lfs")
    if not isinstance(lfs, dict):
        raise ValueError("official tree record is missing LFS metadata")
    size = _positive_int(record.get("size"), field="source size")
    if _positive_int(lfs.get("size"), field="LFS size") != size:
        raise ValueError("official tree record has mismatched LFS size")
    return size, str(lfs.get("oid"))


def _parquet_records(tree: object) -> list[Mapping[str, Any]]:
    if not isinstance(tree, list):
        raise ValueError("official tree response is not a list")
    records: list[Mapping[str, Any]] = []
    for record in tree:
        if not isinstance(record, dict):
            raise ValueError("official tree response contains a non-record")
        path = record.get("path")
        if record.get("type") == "file" and isinstance(path, str) and path.endswith(".parquet"):
            records.append(record)
    return records


def _expected_swe_paths() -> list[str]:
    return [f"data/train-{index:05d}-of-00096.parquet" for index in range(96)]


def _expected_ptv2_paths() -> dict[str, list[str]]:
    return {
        split: [f"data/{split}-{index:05d}-of-{count:05d}.parquet" for index in range(count)]
        for split, count in PTV2_SPLIT_COUNTS.items()
    }


def _records_by_exact_path(
    tree: object, expected_paths: Sequence[str], *, expected_count: int
) -> dict[str, Mapping[str, Any]]:
    parquet = _parquet_records(tree)
    if len(parquet) != expected_count:
        raise ValueError(f"official tree must contain exactly {expected_count} Parquet files")
    by_path = {str(record["path"]): record for record in parquet}
    if len(by_path) != len(parquet) or set(by_path) != set(expected_paths):
        raise ValueError("official tree Parquet paths do not match the approved inventory")
    return by_path


def _trace_records(sources: Mapping[str, Any]) -> list[dict[str, Any]]:
    known = sources.get("known_pinned_sources")
    if not isinstance(known, list):
        raise ValueError("Q30 source requirements are missing known_pinned_sources")
    expected = {(repository, revision, path) for repository, revision, path, _ in _TRACE_IDENTITIES}
    selected: dict[tuple[str, str, str], Mapping[str, Any]] = {}
    for record in known:
        if isinstance(record, dict):
            identity = (
                str(record.get("repository")),
                str(record.get("revision")),
                str(record.get("path")),
            )
            if identity in expected:
                selected[identity] = record
    if set(selected) != expected:
        raise ValueError("Q30 source requirements lack the exact four trace sources")
    output: list[dict[str, Any]] = []
    for repository, revision, path, license_expression in _TRACE_IDENTITIES:
        source = selected[(repository, revision, path)]
        size = _positive_int(source.get("bytes"), field="trace source size")
        sha256 = str(source.get("sha256"))
        if _SHA256.fullmatch(sha256) is None:
            raise ValueError("trace source is missing an exact SHA256")
        category = source.get("category")
        if category not in {
            "ptv3_interactive_agentic_swe",
            "ptv3_general_tool_trajectories",
        }:
            raise ValueError("trace source category is not approved")
        output.append(
            {
                "source_id": repository,
                "revision": revision,
                "license": license_expression,
                "pool": "ptv3",
                "category": category,
                "response_source": "trace-replay",
                "tool_lane": "recorded-trace",
                "path": path,
                "bytes": size,
                "sha256": sha256,
            }
        )
    return output


def build_ptv3_stage_plan(
    swe_v3_tree: object,
    sources: Mapping[str, Any],
    *,
    container: Mapping[str, Any],
) -> dict[str, Any]:
    """Build the 96-shard SWE-v3 plus four exact replay-source stage plan."""
    expected = _expected_swe_paths()
    by_path = _records_by_exact_path(swe_v3_tree, expected, expected_count=96)
    files: list[dict[str, Any]] = []
    for path in expected:
        size, sha256 = _lfs_identity(by_path[path])
        if _SHA256.fullmatch(sha256) is None:
            raise ValueError("SWE-v3 source is missing an exact LFS SHA256")
        files.append(
            {
                "source_id": SWE_V3_REPOSITORY,
                "revision": SWE_V3_REVISION,
                "license": SWE_V3_LICENSE,
                "pool": "ptv3",
                "category": "ptv3_swe_v3",
                "response_source": "trace-replay",
                "tool_lane": "recorded-trace",
                "path": path,
                "bytes": size,
                "sha256": sha256,
            }
        )
    files.extend(_trace_records(sources))
    plan = {
        "schema_version": 1,
        "name": "qwen3-30ba3b-thinking-ptv3-stage-subset-v1",
        "container": dict(container),
        "files": files,
    }
    validate_ptv3_stage_plan(plan)
    return plan


def build_ptv2_stage_inventory(tree: object) -> dict[str, Any]:
    """Record the complete gated PTV2 path/size input to the cluster hash bootstrap."""
    expected_by_split = _expected_ptv2_paths()
    expected = [path for paths in expected_by_split.values() for path in paths]
    parquet = _parquet_records(tree)
    by_path = {str(record["path"]): record for record in parquet}
    allowed_extra = {"data/multilingual-00000-of-00001.parquet"}
    if len(by_path) != len(parquet) or set(by_path) - set(expected) != allowed_extra:
        raise ValueError("official PTV2 tree has an unexpected Parquet inventory")
    if not set(expected).issubset(by_path):
        raise ValueError("official tree must contain exactly 201 approved PTV2 Parquet files")
    observed_sha256 = [_lfs_identity(by_path[path])[1] for path in expected]
    redacted = [sha256 == _REDACTED_SHA256 for sha256 in observed_sha256]
    if any(redacted) and not all(redacted):
        raise ValueError("PTV2 LFS SHA256 metadata must be uniformly redacted or exact")
    if not all(redacted) and any(_SHA256.fullmatch(value) is None for value in observed_sha256):
        raise ValueError("PTV2 source is missing an exact LFS SHA256")
    sources: list[dict[str, Any]] = []
    total_bytes = 0
    for split, paths in expected_by_split.items():
        files: list[dict[str, Any]] = []
        for path in paths:
            size, sha256 = _lfs_identity(by_path[path])
            total_bytes += size
            files.append(
                {
                    "path": path,
                    "bytes": size,
                    "sha256": None if all(redacted) else sha256,
                }
            )
        sources.append({"split": split, "files": files})
    inventory = {
        "schema_version": 1,
        "name": "qwen3-30ba3b-thinking-ptv2-full201-stage-inventory-v1",
        "repository": PTV2_REPOSITORY,
        "revision": PTV2_REVISION,
        "configuration": "default",
        "license": PTV2_LICENSE,
        "file_count": 201,
        "total_bytes": total_bytes,
        "source_sha256_status": (
            "gated-redacted-require-bootstrap" if all(redacted) else "official-lfs-pinned"
        ),
        "bootstrap_entrypoint": "examples/dataset/bootstrap_ptv2_source_manifest.py",
        "sources": sources,
    }
    validate_ptv2_stage_inventory(inventory)
    return inventory


def validate_ptv3_stage_plan(plan: Mapping[str, Any]) -> None:
    """Validate the immutable aggregate and every source identity in a Q30T PTV3 plan."""
    if set(plan) != {"schema_version", "name", "container", "files"}:
        raise ValueError("PTV3 plan has invalid fields")
    if plan.get("schema_version") != 1:
        raise ValueError("PTV3 plan has invalid schema")
    files = plan.get("files")
    if not isinstance(files, list) or len(files) != 100:
        raise ValueError("PTV3 plan must contain exactly 100 source files")
    swe = files[:96]
    if [record.get("path") for record in swe if isinstance(record, dict)] != _expected_swe_paths():
        raise ValueError("PTV3 plan SWE-v3 paths are not exact source order")
    identities: set[tuple[str, str, str]] = set()
    for record in files:
        if not isinstance(record, dict):
            raise ValueError("PTV3 plan contains a non-record")
        size = _positive_int(record.get("bytes"), field="PTV3 source size")
        del size
        sha256 = str(record.get("sha256"))
        if _SHA256.fullmatch(sha256) is None:
            raise ValueError("PTV3 plan source is missing an exact SHA256")
        identity = (
            str(record.get("source_id")),
            str(record.get("revision")),
            str(record.get("path")),
        )
        if identity in identities:
            raise ValueError("PTV3 plan contains duplicate source identities")
        identities.add(identity)
    if any(
        record.get("source_id") != SWE_V3_REPOSITORY
        or record.get("revision") != SWE_V3_REVISION
        or record.get("license") != SWE_V3_LICENSE
        or record.get("category") != "ptv3_swe_v3"
        for record in swe
    ):
        raise ValueError("PTV3 plan SWE-v3 identity is not pinned")
    trace_identities = {
        (record.get("source_id"), record.get("revision"), record.get("path"))
        for record in files[96:]
    }
    if trace_identities != {
        (repository, revision, path) for repository, revision, path, _ in _TRACE_IDENTITIES
    }:
        raise ValueError("PTV3 plan trace-source identity is not pinned")
    if any(
        record.get("response_source") != "trace-replay"
        or record.get("tool_lane") != "recorded-trace"
        for record in files
    ):
        raise ValueError("PTV3 plan response/tool lane is not preserved")


def validate_ptv2_stage_inventory(inventory: Mapping[str, Any]) -> None:
    """Validate the complete PTV2 source shape and explicit gated-hash state."""
    expected_fields = {
        "schema_version",
        "name",
        "repository",
        "revision",
        "configuration",
        "license",
        "file_count",
        "total_bytes",
        "source_sha256_status",
        "bootstrap_entrypoint",
        "sources",
    }
    if set(inventory) != expected_fields:
        raise ValueError("PTV2 inventory has invalid fields")
    if (
        inventory.get("schema_version") != 1
        or inventory.get("repository") != PTV2_REPOSITORY
        or inventory.get("revision") != PTV2_REVISION
        or inventory.get("configuration") != "default"
        or inventory.get("license") != PTV2_LICENSE
        or inventory.get("file_count") != 201
        or inventory.get("bootstrap_entrypoint")
        != "examples/dataset/bootstrap_ptv2_source_manifest.py"
    ):
        raise ValueError("PTV2 inventory identity is invalid")
    sources = inventory.get("sources")
    if not isinstance(sources, list) or [source.get("split") for source in sources] != list(
        PTV2_SPLIT_COUNTS
    ):
        raise ValueError("PTV2 inventory split order is invalid")
    expected_by_split = _expected_ptv2_paths()
    total_bytes = 0
    total_files = 0
    all_sha256: list[object] = []
    for source in sources:
        split = str(source["split"])
        files = source.get("files")
        if (
            not isinstance(files, list)
            or [record.get("path") for record in files] != (expected_by_split[split])
        ):
            raise ValueError("PTV2 inventory file paths are invalid")
        for record in files:
            total_bytes += _positive_int(record.get("bytes"), field="PTV2 source size")
            all_sha256.append(record.get("sha256"))
        total_files += len(files)
    if total_files != 201 or inventory.get("total_bytes") != total_bytes:
        raise ValueError("PTV2 inventory aggregate is invalid")
    status = inventory.get("source_sha256_status")
    if status == "gated-redacted-require-bootstrap":
        if any(value is not None for value in all_sha256):
            raise ValueError("PTV2 gated inventory must not claim source SHA256 values")
    elif status == "official-lfs-pinned":
        if any(_SHA256.fullmatch(str(value)) is None for value in all_sha256):
            raise ValueError("PTV2 pinned inventory has an invalid source SHA256")
    else:
        raise ValueError("PTV2 inventory has an invalid SHA256 status")


def _fetch_tree(repository: str, revision: str, *, token: str | None) -> object:
    url = (
        f"https://huggingface.co/api/datasets/{repository}/tree/{revision}/data"
        "?recursive=true&limit=1000"
    )
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    request = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(request, timeout=120) as response:
        return json.load(response)


def _write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2, sort_keys=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def main() -> int:
    """Fetch official metadata and publish deterministic checked-in stage inputs."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--sources", required=True, type=Path)
    parser.add_argument("--ptv3-output", required=True, type=Path)
    parser.add_argument("--ptv2-output", required=True, type=Path)
    arguments = parser.parse_args()
    sources = json.loads(arguments.sources.read_text(encoding="utf-8"))
    container = {"path": PTYCHE_CONTAINER_PATH, "sha256": PTYCHE_CONTAINER_SHA256}
    token = os.environ.get("HF_TOKEN")
    ptv3 = build_ptv3_stage_plan(
        _fetch_tree(SWE_V3_REPOSITORY, SWE_V3_REVISION, token=token),
        sources,
        container=container,
    )
    ptv2 = build_ptv2_stage_inventory(_fetch_tree(PTV2_REPOSITORY, PTV2_REVISION, token=token))
    _write_json_atomic(arguments.ptv3_output, ptv3)
    _write_json_atomic(arguments.ptv2_output, ptv2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
