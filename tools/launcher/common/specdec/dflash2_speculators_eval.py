# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Fail-closed contracts for the matched Q30 DFlash2 step-4166 evaluation."""

from __future__ import annotations

import argparse
import collections
import csv
import hashlib
import json
import math
import os
import re
import subprocess
import tempfile
import urllib.request
from datetime import datetime
from pathlib import Path
from typing import Any

try:
    from common.specdec.dflash2_runtime_contract import (
        artifact_tree_sha256,
        validate_artifact_receipt,
    )
    from common.specdec.dflash2_target_contract import validate_dflash2_target_snapshot
except ModuleNotFoundError:  # Direct script execution from common/specdec.
    from dflash2_runtime_contract import artifact_tree_sha256, validate_artifact_receipt
    from dflash2_target_contract import validate_dflash2_target_snapshot

DATASET_ID = "RedHatAI/speculator_benchmarks"
DATASET_REVISION = "2ae86affa2cb97a972b7fc681dd51c04fbff083e"
STANDARD_SUBSETS = (
    "HumanEval",
    "math_reasoning",
    "qa",
    "question",
    "rag",
    "summarization",
    "tool_call",
    "translation",
    "writing",
)
EVALUATION_REQUESTS_PER_SUBSET = 200
CORRECTNESS_REQUESTS_PER_SUBSET = EVALUATION_REQUESTS_PER_SUBSET
EVALUATION_STEP = 4166
DFLASH2_BLOCK_SIZE = 8
DFLASH2_SPECULATIVE_TOKENS = 7
_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")
_GIT_SHA_PATTERN = re.compile(r"[0-9a-f]{40}")
_CONTROL_MANIFEST_KEYS = {
    "status",
    "recorded_at",
    "method",
    "block_size",
    "num_speculative_tokens",
    "target_model",
    "draft_model",
    "speculators_repo",
    "speculators_sha",
    "modelopt_repo",
    "modelopt_sha",
    "modelopt_dirty",
    "runtime",
    "runtimes",
    "artifact_identity",
    "container",
    "dataset",
    "provenance_error",
    "slurm_job_id",
    "launcher_config",
    "config_sha256",
    "versions",
    "server_args",
    "evaluator_args",
    "evaluation",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _is_sha256(value: object) -> bool:
    return isinstance(value, str) and _SHA256_PATTERN.fullmatch(value) is not None


def _canonical(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _sha_json(value: object) -> str:
    return hashlib.sha256(_canonical(value).encode()).hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _atomic_json(path: Path, payload: dict[str, Any], *, no_replace: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if no_replace and path.exists():
        raise FileExistsError(f"output already exists: {path}")
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w") as stream:
            stream.write(json.dumps(payload, indent=2, sort_keys=True) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        if no_replace and path.exists():
            raise FileExistsError(f"output already exists: {path}")
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _read_prompt_prefix(path: Path, limit: int) -> list[tuple[int, str]]:
    prompts: list[tuple[int, str]] = []
    with path.open() as stream:
        for source_row, line in enumerate(stream):
            if source_row >= limit:
                break
            row = json.loads(line)
            prompt = row.get("prompt") if isinstance(row, dict) else None
            if not isinstance(prompt, str) or not prompt:
                raise ValueError(f"prompt {source_row} is not a non-empty string: {path}")
            recorded_source_row = row.get("_specdec_source_row", source_row)
            if (
                isinstance(recorded_source_row, bool)
                or not isinstance(recorded_source_row, int)
                or recorded_source_row < 0
            ):
                raise ValueError(f"invalid source row {source_row}: {path}")
            prompts.append((recorded_source_row, prompt))
    if not prompts:
        raise ValueError(f"no prompts: {path}")
    return prompts


def materialize_prompt_set(
    source_manifest_path: Path,
    source_hf_home: Path,
    output_hf_home: Path,
    output_manifest_path: Path,
) -> dict[str, Any]:
    """Materialize an authenticated source-order cycle of exactly 200 rows/subset."""
    source = compute_prompt_set(source_manifest_path, source_hf_home)
    if output_manifest_path.exists() or output_hf_home.exists():
        raise FileExistsError("matched prompt output already exists")
    snapshot = (
        output_hf_home
        / "hub/datasets--RedHatAI--speculator_benchmarks/snapshots"
        / DATASET_REVISION
    )
    snapshot.mkdir(parents=True)
    source_manifest = _load_json(source_manifest_path)
    source_files = source_manifest["files"]
    files: dict[str, dict[str, object]] = {}
    for subset in STANDARD_SUBSETS:
        source_path = Path(str(source_files[subset]["path"]))
        rows: list[dict[str, Any]] = []
        with source_path.open() as stream:
            for index, line in enumerate(stream):
                if index >= EVALUATION_REQUESTS_PER_SUBSET:
                    break
                row = json.loads(line)
                if not isinstance(row, dict) or not isinstance(row.get("prompt"), str):
                    raise ValueError(f"invalid source prompt row: {subset}[{index}]")
                rows.append(row)
        if not rows:
            raise ValueError(f"empty source subset: {subset}")
        output_path = snapshot / f"{subset}.jsonl"
        with output_path.open("x") as stream:
            for ordinal in range(EVALUATION_REQUESTS_PER_SUBSET):
                source_row = ordinal % len(rows)
                derived = {**rows[source_row], "_specdec_source_row": source_row}
                stream.write(_canonical(derived) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        files[subset] = {
            "path": str(output_path.resolve(strict=True)),
            "sha256": _sha256(output_path),
        }
    payload: dict[str, Any] = {
        "dataset_id": DATASET_ID,
        "revision": DATASET_REVISION,
        "hf_home": str(output_hf_home.resolve(strict=True)),
        "files": files,
        "derivation": {
            "producer": "speculator-benchmarks-source-order-cycle-v1",
            "requests_per_subset": EVALUATION_REQUESTS_PER_SUBSET,
            "source_manifest_path": str(source_manifest_path.resolve(strict=True)),
            "source_manifest_sha256": _sha256(source_manifest_path),
            "source_hf_home": str(source_hf_home.resolve(strict=True)),
            "source_prompt_sha256": source["prompt_sha256"],
            "source_files": source["files"],
        },
    }
    _atomic_json(output_manifest_path, payload, no_replace=True)
    compute_prompt_set(output_manifest_path, output_hf_home)
    return payload


def compute_prompt_set(
    dataset_manifest_path: Path,
    hf_home: Path,
    *,
    requests_per_subset: int = EVALUATION_REQUESTS_PER_SUBSET,
) -> dict[str, Any]:
    """Bind the ordered prompt prefix consumed by every matched cell."""
    if requests_per_subset != EVALUATION_REQUESTS_PER_SUBSET:
        raise ValueError("matched evaluation requires exactly 200 requests per subset")
    manifest = _load_json(dataset_manifest_path)
    recorded_home = Path(str(manifest.get("hf_home", ""))).resolve(strict=False)
    active_home = hf_home.resolve(strict=False)
    if manifest.get("dataset_id") != DATASET_ID:
        raise ValueError("dataset identity mismatch")
    if manifest.get("revision") != DATASET_REVISION:
        raise ValueError("dataset revision mismatch")
    if recorded_home != active_home:
        raise ValueError("dataset HF_HOME mismatch")
    files = manifest.get("files")
    if not isinstance(files, dict) or set(files) != set(STANDARD_SUBSETS):
        raise ValueError("dataset manifest must contain exactly the nine subsets")
    source_prompt_set: dict[str, Any] | None = None
    derivation = manifest.get("derivation")
    if derivation is not None:
        if (
            not isinstance(derivation, dict)
            or derivation.get("producer") != "speculator-benchmarks-source-order-cycle-v1"
            or derivation.get("requests_per_subset") != requests_per_subset
        ):
            raise ValueError("invalid matched-prompt derivation")
        source_manifest = Path(str(derivation.get("source_manifest_path", "")))
        source_home = Path(str(derivation.get("source_hf_home", "")))
        if (
            not source_manifest.is_file()
            or _sha256(source_manifest) != derivation.get("source_manifest_sha256")
            or source_manifest.resolve(strict=True) == dataset_manifest_path.resolve(strict=True)
        ):
            raise ValueError("matched-prompt source manifest mismatch")
        source_prompt_set = compute_prompt_set(source_manifest, source_home)
        if (
            derivation.get("source_prompt_sha256") != source_prompt_set["prompt_sha256"]
            or derivation.get("source_files") != source_prompt_set["files"]
        ):
            raise ValueError("matched-prompt source provenance mismatch")
    snapshot = (
        active_home / "hub/datasets--RedHatAI--speculator_benchmarks/snapshots" / DATASET_REVISION
    ).resolve(strict=False)
    prompts: list[dict[str, object]] = []
    file_descriptors: dict[str, dict[str, object]] = {}
    for subset in STANDARD_SUBSETS:
        entry = files[subset]
        if not isinstance(entry, dict):
            raise ValueError(f"invalid dataset entry: {subset}")
        path = Path(str(entry.get("path", "")))
        if path.resolve(strict=False) != (snapshot / f"{subset}.jsonl").resolve(strict=False):
            raise ValueError(f"dataset path mismatch: {subset}")
        expected_sha = entry.get("sha256")
        if not path.is_file() or not isinstance(expected_sha, str) or _sha256(path) != expected_sha:
            raise ValueError(f"dataset file hash mismatch: {subset}")
        available_prompts = _read_prompt_prefix(path, requests_per_subset)
        for ordinal in range(requests_per_subset):
            selected_row, prompt = available_prompts[ordinal % len(available_prompts)]
            prompts.append(
                {
                    "subset": subset,
                    "index": ordinal,
                    "source_row": selected_row,
                    "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
                }
            )
        file_descriptors[subset] = {
            "path": str(path.resolve(strict=True)),
            "sha256": expected_sha,
            "available_rows": len(available_prompts),
        }
    if source_prompt_set is not None and prompts != source_prompt_set["ordered_prompts"]:
        raise ValueError("materialized prompt schedule does not match its source")
    return {
        "dataset_id": DATASET_ID,
        "revision": DATASET_REVISION,
        "manifest_path": str(dataset_manifest_path.resolve(strict=True)),
        "manifest_sha256": _sha256(dataset_manifest_path),
        "subsets": list(STANDARD_SUBSETS),
        "requests_per_subset": requests_per_subset,
        "total_requests": len(prompts),
        "prompt_sha256": _sha_json(prompts),
        "ordered_prompts": prompts,
        "files": file_descriptors,
    }


def _regular_file_hashes(root: Path) -> dict[str, str]:
    return {
        path.relative_to(root).as_posix(): _sha256(path)
        for path in sorted(root.rglob("*"))
        if path.is_file() and not path.is_symlink()
    }


def validate_milestone_export(
    milestone_manifest_path: Path,
    export_path: Path,
    *,
    expected_step: int = EVALUATION_STEP,
) -> dict[str, Any]:
    """Authenticate the exact lifecycle-preserved export at step 4166."""
    if expected_step != EVALUATION_STEP:
        raise ValueError("matched DFlash2 evaluation requires step 4166")
    manifest = _load_json(milestone_manifest_path)
    milestone = milestone_manifest_path.parent
    exact_model = milestone / "exact-model"
    expected_hashes = manifest.get("exact_model_sha256")
    if (
        manifest.get("exact_model_step") != expected_step
        or not exact_model.is_symlink()
        or exact_model.resolve(strict=True) != export_path.resolve(strict=True)
        or not isinstance(expected_hashes, dict)
    ):
        raise ValueError("milestone export identity mismatch")
    actual_hashes = _regular_file_hashes(export_path.resolve(strict=True))
    if actual_hashes != expected_hashes:
        raise ValueError("milestone export file hashes mismatch")
    config = _load_json(export_path / "config.json")
    architectures = config.get("architectures")
    if not isinstance(architectures, list) or "DFlash2DraftModel" not in architectures:
        raise ValueError("step-4166 export is not DFlash2")
    dflash_config = config.get("dflash_config")
    if (
        config.get("block_size") != DFLASH2_BLOCK_SIZE
        or config.get("num_attention_heads") != 32
        or config.get("num_key_value_heads") != 4
        or config.get("head_dim") != 128
        or config.get("intermediate_size") != 6144
        or not isinstance(dflash_config, dict)
        or dflash_config.get("projector_type") != "dflash2"
    ):
        raise ValueError("step-4166 export is not the Q30 Base B8 DFlash2 architecture")
    return {
        "step": expected_step,
        "milestone_manifest_path": str(milestone_manifest_path.resolve(strict=True)),
        "milestone_manifest_sha256": _sha256(milestone_manifest_path),
        "export_path": str(export_path.resolve(strict=True)),
        "export_tree_sha256": artifact_tree_sha256(export_path),
        "export_file_sha256": actual_hashes,
    }


def build_artifact_identity(
    *,
    target_path: Path,
    target_sha256: str,
    target_receipt_path: Path,
    target_receipt_sha256: str,
    export_path: Path,
    milestone_manifest_path: Path,
    dataset_manifest_path: Path,
    hf_home: Path,
    client_runtime_archive: Path,
    client_runtime_archive_sha256: str,
    server_runtime_archive: Path,
    server_runtime_archive_sha256: str,
    server_runtime_receipt_sha256: str,
) -> dict[str, Any]:
    """Rehash every immutable scientific input shared by C1 and C32."""
    target_spec = validate_dflash2_target_snapshot(target_path, "q30-base")
    validate_artifact_receipt(
        target_receipt_path,
        expected_receipt_sha256=target_receipt_sha256,
        artifact_path=target_path,
        expected_artifact_sha256=target_sha256,
        kind="target",
    )
    export = validate_milestone_export(milestone_manifest_path, export_path)
    prompt_set = compute_prompt_set(dataset_manifest_path, hf_home)
    for path, expected, label in (
        (client_runtime_archive, client_runtime_archive_sha256, "client"),
        (server_runtime_archive, server_runtime_archive_sha256, "server"),
    ):
        if not _is_sha256(expected) or _sha256(path) != expected:
            raise ValueError(f"{label} runtime archive SHA-256 mismatch")
    if not _is_sha256(server_runtime_receipt_sha256):
        raise ValueError("server runtime receipt SHA-256 must be exact")
    payload = {
        "schema_version": 1,
        "producer": "q30-dflash2-s4166-speculators-inputs-v1",
        "target": {
            "path": str(target_path.resolve(strict=True)),
            "artifact_sha256": target_sha256,
            "receipt_path": str(target_receipt_path.resolve(strict=True)),
            "receipt_sha256": target_receipt_sha256,
            "label": "q30-base",
            "revision": target_spec.revision,
            "capture_ids": list(target_spec.capture_ids),
            "serve_tp": target_spec.serve_tp,
        },
        "draft": export,
        "dataset": prompt_set,
        "runtimes": {
            "client": {
                "archive_path": str(client_runtime_archive.resolve(strict=True)),
                "archive_sha256": client_runtime_archive_sha256,
            },
            "server": {
                "archive_path": str(server_runtime_archive.resolve(strict=True)),
                "archive_sha256": server_runtime_archive_sha256,
                "receipt_sha256": server_runtime_receipt_sha256,
                "vllm_commit": "b389ac29465b33f9e9c534df221ea3c129e9793f",
                "profile_capacity_patch_sha256": (
                    "540fbb17ff1b1f388435c924b73a56c997b88642608e1fa466e8874dd6e59212"
                ),
                "flashmla_commit": "a8f794d1251cbfd88a5011445dd5582289c727e4",
            },
        },
        "sampling": {
            "temperature": 0,
            "top_p": 1,
            "correctness": {
                "seed": 42,
                "max_tokens": 64,
                "logprobs": 0,
                "return_tokens_as_token_ids": True,
            },
        },
        "matrix": {
            "concurrencies": [1, 32],
            "requests_per_subset": 200,
            "tensor_parallel_size": {"baseline": 2, "dflash2": 2},
            "subsets": list(STANDARD_SUBSETS),
            "dflash2": {"block_size": 8, "num_speculative_tokens": 7},
        },
    }
    payload["receipt_sha256"] = _sha_json(payload)
    return payload


def _load_ledger(path: Path, requests_per_subset: int) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open() as stream:
        for line in stream:
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError("output ledger row must be an object")
            records.append(value)
    expected = [
        (subset, index) for subset in STANDARD_SUBSETS for index in range(requests_per_subset)
    ]
    actual = [(row.get("subset"), row.get("index")) for row in records]
    if actual != expected:
        raise ValueError("output ledger request order/count mismatch")
    for row in records:
        if (
            set(row)
            != {
                "subset",
                "index",
                "source_row",
                "prompt_sha256",
                "output_sha256",
                "output_text",
                "output_tokens",
                "request_sha256",
                "seed",
                "finish_reason",
            }
            or not all(
                _is_sha256(row.get(name))
                for name in ("prompt_sha256", "output_sha256", "request_sha256")
            )
            or isinstance(row.get("source_row"), bool)
            or not isinstance(row.get("source_row"), int)
            or row["source_row"] < 0
            or not isinstance(row.get("output_text"), str)
            or not isinstance(row.get("output_tokens"), list)
            or not all(isinstance(token, str) for token in row["output_tokens"])
            or row.get("seed") != 42
            or hashlib.sha256(row["output_text"].encode()).hexdigest()
            != row["output_sha256"]
            or row.get("finish_reason") not in {"stop", "length"}
        ):
            raise ValueError("output ledger row schema mismatch")
    return records


def _common_prefix_length(left: list[str], right: list[str]) -> int:
    length = 0
    for expected, actual in zip(left, right):
        if expected != actual:
            break
        length += 1
    return length


def _within_server_repeat_metrics(rows: list[dict[str, Any]]) -> dict[str, int | float]:
    groups: dict[tuple[str, int, str], list[dict[str, Any]]] = collections.defaultdict(list)
    for row in rows:
        groups[(row["subset"], row["source_row"], row["prompt_sha256"])].append(row)
    repeats = [values for values in groups.values() if len(values) > 1]
    token_prefixes = [
        _common_prefix_length(values[0]["output_tokens"], row["output_tokens"])
        for values in repeats
        for row in values[1:]
    ]
    character_prefixes = [
        _common_prefix_length(list(values[0]["output_text"]), list(row["output_text"]))
        for values in repeats
        for row in values[1:]
    ]
    return {
        "repeat_groups": len(repeats),
        "divergent_text_repeat_groups": sum(
            len({row["output_sha256"] for row in values}) > 1 for values in repeats
        ),
        "divergent_token_repeat_groups": sum(
            len({_canonical(row["output_tokens"]) for row in values}) > 1 for values in repeats
        ),
        "mean_common_token_prefix": (
            sum(token_prefixes) / len(token_prefixes) if token_prefixes else 0
        ),
        "mean_common_character_prefix": (
            sum(character_prefixes) / len(character_prefixes) if character_prefixes else 0
        ),
    }


def summarize_target_control(
    left_path: Path,
    right_path: Path,
    *,
    requests_per_subset: int = CORRECTNESS_REQUESTS_PER_SUBSET,
    expected_prompt_schedule: list[dict[str, object]] | None = None,
    expected_request_sha256: list[str] | None = None,
) -> dict[str, Any]:
    """Quantify target-only runtime nondeterminism without authorizing speed metrics."""
    left = _load_ledger(left_path, requests_per_subset)
    right = _load_ledger(right_path, requests_per_subset)
    left_schedule = [
        {
            "subset": row["subset"],
            "index": row["index"],
            "source_row": row["source_row"],
            "prompt_sha256": row["prompt_sha256"],
        }
        for row in left
    ]
    right_schedule = [
        {
            "subset": row["subset"],
            "index": row["index"],
            "source_row": row["source_row"],
            "prompt_sha256": row["prompt_sha256"],
        }
        for row in right
    ]
    if left_schedule != right_schedule or (
        expected_prompt_schedule is not None and left_schedule != expected_prompt_schedule
    ):
        raise ValueError("target control prompt schedule mismatch")
    left_request_sha256 = [row["request_sha256"] for row in left]
    if left_request_sha256 != [row["request_sha256"] for row in right] or (
        expected_request_sha256 is not None
        and left_request_sha256 != expected_request_sha256
    ):
        raise ValueError("target control request fingerprint mismatch")

    subset_metrics: dict[str, dict[str, object]] = {}
    token_prefixes: list[int] = []
    text_prefixes: list[int] = []
    exact_text_total = 0
    exact_token_total = 0
    for subset in STANDARD_SUBSETS:
        pairs = [
            (expected, actual)
            for expected, actual in zip(left, right, strict=True)
            if expected["subset"] == subset
        ]
        exact_text = sum(
            expected["output_text"] == actual["output_text"] for expected, actual in pairs
        )
        exact_tokens = sum(
            expected["output_tokens"] == actual["output_tokens"]
            for expected, actual in pairs
        )
        prefixes = [
            _common_prefix_length(expected["output_tokens"], actual["output_tokens"])
            for expected, actual in pairs
        ]
        char_prefixes = [
            _common_prefix_length(list(expected["output_text"]), list(actual["output_text"]))
            for expected, actual in pairs
        ]
        exact_text_total += exact_text
        exact_token_total += exact_tokens
        token_prefixes.extend(prefixes)
        text_prefixes.extend(char_prefixes)
        subset_metrics[subset] = {
            "requests": len(pairs),
            "exact_text_matches": exact_text,
            "exact_token_matches": exact_tokens,
            "mean_common_token_prefix": sum(prefixes) / len(prefixes),
            "mean_common_character_prefix": sum(char_prefixes) / len(char_prefixes),
        }
    return {
        "schema_version": 1,
        "producer": "q30-target-target-runtime-control-v1",
        "status": "diagnostic-only",
        "claim_scope": "no training-quality or speedup claim",
        "allocation_evidence_scope": (
            "live SLURM/GPU origin checked at creation; offline verification is tamper replay"
        ),
        "requests": len(left),
        "sampling": {"temperature": 0, "top_p": 1, "seed": 42},
        "request_set_sha256": _sha_json(left_request_sha256),
        "left_ledger_sha256": _sha256(left_path),
        "right_ledger_sha256": _sha256(right_path),
        "cross_server": {
            "exact_text_matches": exact_text_total,
            "exact_token_matches": exact_token_total,
            "mean_common_token_prefix": sum(token_prefixes) / len(token_prefixes),
            "mean_common_character_prefix": sum(text_prefixes) / len(text_prefixes),
            "subsets": subset_metrics,
        },
        "within_server": {
            "left": _within_server_repeat_metrics(left),
            "right": _within_server_repeat_metrics(right),
        },
        "task_correctness": {
            "status": "not-claimed",
            "reason": "performance prompts do not provide a safe uniform semantic scorer",
        },
    }


def validate_output_equivalence(
    baseline_path: Path,
    dflash2_path: Path,
    *,
    requests_per_subset: int = CORRECTNESS_REQUESTS_PER_SUBSET,
    artifact_identity_sha256: str | None = None,
    expected_prompt_schedule: list[dict[str, object]] | None = None,
) -> dict[str, Any]:
    """Require deterministic target output identity before speed metrics are publishable."""
    if requests_per_subset != CORRECTNESS_REQUESTS_PER_SUBSET:
        raise ValueError("correctness gate requires all 200 prompts per subset")
    baseline = _load_ledger(baseline_path, requests_per_subset)
    dflash2 = _load_ledger(dflash2_path, requests_per_subset)
    if expected_prompt_schedule is not None:
        actual_schedule = [
            {
                "subset": row["subset"],
                "index": row["index"],
                "source_row": row["source_row"],
                "prompt_sha256": row["prompt_sha256"],
            }
            for row in baseline
        ]
        if actual_schedule != expected_prompt_schedule:
            raise ValueError("captured prompt ledger does not match artifact identity")
    for expected, actual in zip(baseline, dflash2, strict=True):
        if expected != actual:
            raise ValueError(f"target output mismatch: {expected['subset']}[{expected['index']}]")
    if artifact_identity_sha256 is not None and not _is_sha256(artifact_identity_sha256):
        raise ValueError("artifact identity SHA-256 must be exact")
    return {
        "status": "passed",
        "matched_requests": len(baseline),
        "requests_per_subset": requests_per_subset,
        "baseline_ledger_sha256": _sha256(baseline_path),
        "dflash2_ledger_sha256": _sha256(dflash2_path),
        "output_set_sha256": _sha_json(baseline),
        "artifact_identity_sha256": artifact_identity_sha256,
    }


def _read_csv_by_subset(path: Path) -> dict[str, dict[str, str]]:
    with path.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) != len(STANDARD_SUBSETS) or {row.get("subset") for row in rows} != set(
        STANDARD_SUBSETS
    ):
        raise ValueError(f"expected exactly nine subset rows: {path}")
    return {row["subset"]: row for row in rows}


def _positive(row: dict[str, str], name: str) -> float:
    try:
        value = float(row[name])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"invalid {name}") from error
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"invalid {name}")
    return value


def _nonnegative(row: dict[str, str], name: str) -> float:
    try:
        value = float(row[name])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"invalid {name}") from error
    if not math.isfinite(value) or value < 0:
        raise ValueError(f"invalid {name}")
    return value


def _file_descriptor(path: Path) -> dict[str, object]:
    resolved = path.resolve(strict=True)
    return {"path": str(resolved), "bytes": resolved.stat().st_size, "sha256": _sha256(resolved)}


def _validate_file_descriptor(value: object) -> Path:
    if not isinstance(value, dict) or set(value) != {"path", "bytes", "sha256"}:
        raise ValueError("invalid evidence file descriptor")
    path = Path(str(value["path"])).resolve(strict=True)
    if (
        not isinstance(value["bytes"], int)
        or isinstance(value["bytes"], bool)
        or value["bytes"] != path.stat().st_size
        or not _is_sha256(value["sha256"])
        or value["sha256"] != _sha256(path)
    ):
        raise ValueError("evidence file descriptor mismatch")
    return path


def build_target_control_allocation_receipt(
    output_path: Path,
    *,
    slurm_job_id: str,
    slurm_job_num_nodes: int,
    slurm_job_nodelist: str,
    gpu_count: int,
) -> dict[str, Any]:
    """Publish job-local proof of the intended one-node disjoint TP2+TP2 allocation."""
    if (
        re.fullmatch(r"[1-9][0-9]*", slurm_job_id) is None
        or slurm_job_num_nodes != 1
        or not slurm_job_nodelist
        or gpu_count != 4
    ):
        raise ValueError("target control allocation must be one node with exactly four GPUs")
    payload: dict[str, Any] = {
        "schema_version": 1,
        "producer": "q30-target-control-allocation-v1",
        "slurm_job_id": slurm_job_id,
        "slurm_job_num_nodes": slurm_job_num_nodes,
        "slurm_job_nodelist": slurm_job_nodelist,
        "gpu_count": gpu_count,
        "cell_visible_devices": {"left": "0,1", "right": "2,3"},
    }
    payload["receipt_sha256"] = _sha_json(payload)
    _atomic_json(output_path, payload, no_replace=True)
    return payload


def validate_target_control_allocation_receipt(path: Path) -> dict[str, Any]:
    """Replay the exact one-node, four-GPU allocation claim."""
    payload = _load_json(path)
    expected_keys = {
        "schema_version",
        "producer",
        "slurm_job_id",
        "slurm_job_num_nodes",
        "slurm_job_nodelist",
        "gpu_count",
        "cell_visible_devices",
        "receipt_sha256",
    }
    unsigned = {key: value for key, value in payload.items() if key != "receipt_sha256"}
    if (
        set(payload) != expected_keys
        or payload.get("schema_version") != 1
        or payload.get("producer") != "q30-target-control-allocation-v1"
        or payload.get("receipt_sha256") != _sha_json(unsigned)
        or re.fullmatch(r"[1-9][0-9]*", str(payload.get("slurm_job_id", ""))) is None
        or payload.get("slurm_job_num_nodes") != 1
        or not isinstance(payload.get("slurm_job_nodelist"), str)
        or not payload.get("slurm_job_nodelist")
        or payload.get("gpu_count") != 4
        or payload.get("cell_visible_devices") != {"left": "0,1", "right": "2,3"}
    ):
        raise ValueError("invalid target control allocation receipt")
    return payload


def _query_current_allocation() -> dict[str, Any]:
    try:
        gpu_lines = subprocess.run(
            ["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.splitlines()
        node_count = int(os.environ["SLURM_JOB_NUM_NODES"])
        job_id = os.environ["SLURM_JOB_ID"]
        nodelist = os.environ["SLURM_JOB_NODELIST"]
    except (KeyError, ValueError, subprocess.CalledProcessError) as error:
        raise ValueError("allocation receipt must be produced inside the active GPU job") from error
    return {
        "slurm_job_id": job_id,
        "slurm_job_num_nodes": node_count,
        "slurm_job_nodelist": nodelist,
        "gpu_count": len(gpu_lines),
        "cell_visible_devices": {"left": "0,1", "right": "2,3"},
    }


def _publish_current_allocation_receipt(output_path: Path) -> dict[str, Any]:
    current = _query_current_allocation()
    return build_target_control_allocation_receipt(
        output_path,
        slurm_job_id=str(current["slurm_job_id"]),
        slurm_job_num_nodes=int(current["slurm_job_num_nodes"]),
        slurm_job_nodelist=str(current["slurm_job_nodelist"]),
        gpu_count=int(current["gpu_count"]),
    )


def _git_output(repo: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(repo), *args],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _validate_clean_repo(path_value: object, sha_value: object, name: str) -> Path:
    path = Path(str(path_value)).resolve(strict=True)
    if (
        not path.is_dir()
        or not isinstance(sha_value, str)
        or _GIT_SHA_PATTERN.fullmatch(sha_value) is None
    ):
        raise ValueError(f"invalid {name} source checkout")
    try:
        head = _git_output(path, "rev-parse", "HEAD")
        dirty = _git_output(path, "status", "--porcelain")
    except subprocess.CalledProcessError as error:
        raise ValueError(f"invalid {name} source checkout") from error
    if head != sha_value or dirty:
        raise ValueError(f"{name} source checkout mismatch")
    return path


def _validate_control_launcher(path_value: object, image: Path, expected_sha: object) -> Path:
    path = Path(str(path_value)).resolve(strict=True)
    if not _is_sha256(expected_sha) or _sha256(path) != expected_sha:
        raise ValueError("target control launcher config mismatch")
    expected = (
        "pipeline:\n  task_0:\n    slurm_config:\n"
        f"      container: {image.resolve(strict=True)}\n"
    )
    if path.read_text() != expected:
        raise ValueError("target control launcher config is not canonical")
    return path


def _validate_artifact_identity(path: Path) -> dict[str, Any]:
    identity = _load_json(path)
    if (
        identity.get("producer") != "q30-dflash2-s4166-speculators-inputs-v1"
        or identity.get("schema_version") != 1
        or identity.get("receipt_sha256") != _sha_json(
            {key: value for key, value in identity.items() if key != "receipt_sha256"}
        )
    ):
        raise ValueError("artifact identity receipt mismatch")
    target = identity.get("target")
    draft = identity.get("draft")
    dataset = identity.get("dataset")
    runtimes = identity.get("runtimes")
    if not all(isinstance(value, dict) for value in (target, draft, dataset, runtimes)):
        raise ValueError("artifact identity schema mismatch")
    assert isinstance(target, dict)
    assert isinstance(draft, dict)
    assert isinstance(dataset, dict)
    assert isinstance(runtimes, dict)
    client = runtimes.get("client")
    server = runtimes.get("server")
    if not isinstance(client, dict) or not isinstance(server, dict):
        raise ValueError("artifact identity runtime schema mismatch")
    dataset_manifest_path = Path(str(dataset.get("manifest_path", "")))
    dataset_manifest = _load_json(dataset_manifest_path)
    rebuilt = build_artifact_identity(
        target_path=Path(str(target.get("path", ""))),
        target_sha256=str(target.get("artifact_sha256", "")),
        target_receipt_path=Path(str(target.get("receipt_path", ""))),
        target_receipt_sha256=str(target.get("receipt_sha256", "")),
        export_path=Path(str(draft.get("export_path", ""))),
        milestone_manifest_path=Path(str(draft.get("milestone_manifest_path", ""))),
        dataset_manifest_path=dataset_manifest_path,
        hf_home=Path(str(dataset_manifest.get("hf_home", ""))),
        client_runtime_archive=Path(str(client.get("archive_path", ""))),
        client_runtime_archive_sha256=str(client.get("archive_sha256", "")),
        server_runtime_archive=Path(str(server.get("archive_path", ""))),
        server_runtime_archive_sha256=str(server.get("archive_sha256", "")),
        server_runtime_receipt_sha256=str(server.get("receipt_sha256", "")),
    )
    if identity != rebuilt:
        raise ValueError("artifact identity replay mismatch")
    return identity


def _validate_target_control_manifest(
    path: Path, artifact_identity_path: Path, identity: dict[str, Any]
) -> tuple[dict[str, Any], Path, dict[str, Any], Path]:
    manifest = _load_json(path)
    evaluation = manifest.get("evaluation")
    server_args = manifest.get("server_args")
    target = identity["target"]
    dataset = identity["dataset"]
    runtimes = identity["runtimes"]
    artifact_sha256 = _sha256(artifact_identity_path)
    config_sha256 = manifest.get("config_sha256")
    artifact_binding = manifest.get("artifact_identity")
    container = manifest.get("container")
    manifest_dataset = manifest.get("dataset")
    runtimes_value = manifest.get("runtimes")
    versions = manifest.get("versions")
    evaluation_expected = {
        "dataset": DATASET_ID,
        "subsets": list(STANDARD_SUBSETS),
        "temperature": 0,
        "top_p": 1,
        "mode": "throughput",
        "max_concurrency": 1,
        "max_requests": 200,
        "tensor_parallel_size": 2,
    }
    try:
        recorded_at = datetime.fromisoformat(str(manifest.get("recorded_at", "")))
    except ValueError as error:
        raise ValueError("target control recorded_at is invalid") from error
    if (
        set(manifest) != _CONTROL_MANIFEST_KEYS
        or manifest.get("method") != "baseline"
        or manifest.get("status") != "success"
        or manifest.get("block_size") != 0
        or manifest.get("num_speculative_tokens") != 0
        or manifest.get("draft_model") is not None
        or manifest.get("target_model") != target["path"]
        or manifest.get("modelopt_dirty") is not False
        or manifest.get("provenance_error") is not None
        or manifest.get("evaluator_args") != []
        or recorded_at.tzinfo is None
        or not isinstance(evaluation, dict)
        or evaluation != evaluation_expected
        or not isinstance(server_args, list)
        or not all(isinstance(value, str) for value in server_args)
        or any(value == "--speculative-config" or value.startswith("--speculative-config=") for value in server_args)
        or server_args[:6]
        != [
            "-m",
            "vllm.entrypoints.cli.main",
            "serve",
            target["path"],
            "--tensor-parallel-size",
            "2",
        ]
        or len(server_args) != 8
        or server_args[6] != "--port"
        or server_args[7] not in {"8000", "8010"}
        or not isinstance(config_sha256, dict)
        or set(config_sha256) != {"launcher", "target"}
        or not all(_is_sha256(value) for value in config_sha256.values())
        or config_sha256["target"] != _sha256(Path(target["path"]) / "config.json")
        or artifact_binding
        != {"path": str(artifact_identity_path.resolve(strict=True)), "sha256": artifact_sha256}
        or not isinstance(container, dict)
        or not isinstance(manifest_dataset, dict)
        or not isinstance(runtimes_value, dict)
        or set(runtimes_value) != {"client", "server"}
        or manifest.get("runtime") != runtimes_value.get("client")
        or not all(isinstance(value, str) and value for value in runtimes_value.values())
        or not isinstance(versions, dict)
        or set(versions) != {"python", "vllm", "guidellm"}
        or not all(isinstance(value, str) and value for value in versions.values())
        or re.fullmatch(r"[1-9][0-9]*", str(manifest.get("slurm_job_id", ""))) is None
    ):
        raise ValueError("target control manifest is not exact baseline C1")

    _validate_clean_repo(manifest.get("modelopt_repo"), manifest.get("modelopt_sha"), "ModelOpt")
    _validate_clean_repo(
        manifest.get("speculators_repo"), manifest.get("speculators_sha"), "Speculators"
    )

    container_identity_path = Path(str(container.get("identity_path", "")))
    container_identity = _load_json(container_identity_path)
    container_path = Path(str(container.get("path", ""))).resolve(strict=True)
    if (
        container.get("identity_sha256") != _sha256(container_identity_path)
        or container_identity
        != {
            "path": container.get("path"),
            "sha256": container.get("sha256"),
            "size_bytes": container.get("size_bytes"),
        }
        or not _is_sha256(container.get("sha256"))
        or not isinstance(container.get("size_bytes"), int)
        or isinstance(container.get("size_bytes"), bool)
        or not container_path.is_file()
        or container_path.stat().st_size != container.get("size_bytes")
        or _sha256(container_path) != container.get("sha256")
    ):
        raise ValueError("target control container identity mismatch")
    launcher_path = _validate_control_launcher(
        manifest.get("launcher_config"), container_path, config_sha256["launcher"]
    )
    expected_dataset_files = {
        subset: {
            "path": dataset["files"][subset]["path"],
            "sha256": dataset["files"][subset]["sha256"],
        }
        for subset in STANDARD_SUBSETS
    }
    dataset_manifest = _load_json(Path(dataset["manifest_path"]))
    if manifest_dataset != {
        "dataset_id": dataset["dataset_id"],
        "revision": dataset["revision"],
        "hf_home": dataset_manifest["hf_home"],
        "manifest_path": dataset["manifest_path"],
        "manifest_sha256": dataset["manifest_sha256"],
        "files": expected_dataset_files,
    }:
        raise ValueError("target control dataset provenance mismatch")

    fingerprint_path = path.parent / "input-fingerprint.json"
    fingerprint = _load_json(fingerprint_path)
    inputs = fingerprint.get("inputs")
    if (
        fingerprint.get("schema_version") != 1
        or not isinstance(inputs, dict)
        or fingerprint.get("sha256") != _sha_json(inputs)
    ):
        raise ValueError("target control input fingerprint self-hash mismatch")
    expected_inputs = {
        "target_config_sha256": config_sha256["target"],
        "draft_config_sha256": None,
        "dataset": {
            "revision": dataset["revision"],
            "manifest_sha256": dataset["manifest_sha256"],
        },
        "image": {
            "sha256": container["sha256"],
            "identity_sha256": container["identity_sha256"],
        },
        "runtimes": {
            "client": {"sha256": runtimes["client"]["archive_sha256"]},
            "server": {
                "sha256": runtimes["server"]["archive_sha256"],
                "receipt_sha256": runtimes["server"]["receipt_sha256"],
            },
        },
        "artifact_identity_sha256": artifact_sha256,
        "source": {
            "modelopt_sha": manifest.get("modelopt_sha"),
            "speculators_sha": manifest.get("speculators_sha"),
        },
        "launcher_config_sha256": config_sha256["launcher"],
        "evaluation": {
            "method": "baseline",
            "block_size": 0,
            "num_speculative_tokens": 0,
            "max_concurrency": 1,
            "max_requests": 200,
            "mode": "throughput",
            "tensor_parallel_size": 2,
        },
    }
    if (
        inputs != expected_inputs
    ):
        raise ValueError("target control input fingerprint mismatch")
    return manifest, fingerprint_path, fingerprint, launcher_path


def build_target_control_receipt(
    left_path: Path,
    right_path: Path,
    left_manifest_path: Path,
    right_manifest_path: Path,
    artifact_identity_path: Path,
    allocation_receipt_path: Path,
    output_path: Path,
) -> dict[str, Any]:
    """Publish a self-hashed, replayable target-only nondeterminism receipt."""
    identity = _validate_artifact_identity(artifact_identity_path)
    dataset = identity.get("dataset")
    if not isinstance(dataset, dict) or not isinstance(dataset.get("ordered_prompts"), list):
        raise ValueError("artifact identity has no ordered prompt schedule")
    manifests = [left_manifest_path, right_manifest_path]
    manifest_evidence = [
        _validate_target_control_manifest(path, artifact_identity_path, identity)
        for path in manifests
    ]
    manifest_payloads = [evidence[0] for evidence in manifest_evidence]
    fingerprint_paths = [evidence[1] for evidence in manifest_evidence]
    fingerprint_payloads = [evidence[2] for evidence in manifest_evidence]
    launcher_paths = [evidence[3] for evidence in manifest_evidence]
    for name in (
        "config_sha256",
        "container",
        "dataset",
        "modelopt_sha",
        "speculators_sha",
        "target_model",
        "versions",
        "artifact_identity",
        "runtimes",
    ):
        if manifest_payloads[0].get(name) != manifest_payloads[1].get(name):
            raise ValueError(f"target control manifest mismatch: {name}")
    if fingerprint_payloads[0].get("inputs") != fingerprint_payloads[1].get("inputs"):
        raise ValueError("target control input fingerprint mismatch between cells")
    allocation = validate_target_control_allocation_receipt(allocation_receipt_path)
    current_allocation = _query_current_allocation()
    for name in (
        "slurm_job_id",
        "slurm_job_num_nodes",
        "slurm_job_nodelist",
        "gpu_count",
        "cell_visible_devices",
    ):
        if allocation.get(name) != current_allocation.get(name):
            raise ValueError(f"target control live allocation mismatch: {name}")
    if any(manifest.get("slurm_job_id") != allocation["slurm_job_id"] for manifest in manifest_payloads):
        raise ValueError("target control allocation job mismatch")
    ports = {manifest["server_args"][-1] for manifest in manifest_payloads}
    if ports != {"8000", "8010"}:
        raise ValueError("target control server ports are not isolated")
    payload = summarize_target_control(
        left_path,
        right_path,
        expected_prompt_schedule=dataset["ordered_prompts"],
        expected_request_sha256=_expected_request_hashes(identity),
    )
    payload["artifact_identity"] = _file_descriptor(artifact_identity_path)
    payload["manifests"] = [_file_descriptor(path) for path in manifests]
    payload["input_fingerprints"] = [
        _file_descriptor(path) for path in fingerprint_paths
    ]
    payload["launcher_configs"] = [_file_descriptor(path) for path in launcher_paths]
    payload["allocation_receipt"] = _file_descriptor(allocation_receipt_path)
    payload["ledgers"] = [_file_descriptor(left_path), _file_descriptor(right_path)]
    payload["receipt_sha256"] = _sha_json(payload)
    _atomic_json(output_path, payload, no_replace=True)
    return payload


def validate_target_control_receipt(path: Path) -> dict[str, Any]:
    """Rehash all target-control evidence and replay every reported metric."""
    payload = _load_json(path)
    claim = payload.pop("receipt_sha256", None)
    if claim != _sha_json(payload):
        raise ValueError("target control receipt self-hash mismatch")
    artifact_path = _validate_file_descriptor(payload.get("artifact_identity"))
    manifest_values = payload.get("manifests")
    ledger_values = payload.get("ledgers")
    if (
        not isinstance(manifest_values, list)
        or len(manifest_values) != 2
        or not isinstance(ledger_values, list)
        or len(ledger_values) != 2
    ):
        raise ValueError("target control evidence descriptor mismatch")
    manifests = [_validate_file_descriptor(value) for value in manifest_values]
    ledgers = [_validate_file_descriptor(value) for value in ledger_values]
    identity = _validate_artifact_identity(artifact_path)
    dataset = identity.get("dataset")
    if not isinstance(dataset, dict) or not isinstance(dataset.get("ordered_prompts"), list):
        raise ValueError("artifact identity has no ordered prompt schedule")
    fingerprint_values = payload.get("input_fingerprints")
    if not isinstance(fingerprint_values, list) or len(fingerprint_values) != 2:
        raise ValueError("target control input fingerprint descriptor mismatch")
    fingerprint_paths = [_validate_file_descriptor(value) for value in fingerprint_values]
    launcher_values = payload.get("launcher_configs")
    if not isinstance(launcher_values, list) or len(launcher_values) != 2:
        raise ValueError("target control launcher config descriptor mismatch")
    launcher_paths = [_validate_file_descriptor(value) for value in launcher_values]
    allocation_path = _validate_file_descriptor(payload.get("allocation_receipt"))
    allocation = validate_target_control_allocation_receipt(allocation_path)
    manifest_evidence = [
        _validate_target_control_manifest(item, artifact_path, identity) for item in manifests
    ]
    manifest_payloads = [evidence[0] for evidence in manifest_evidence]
    if [evidence[1] for evidence in manifest_evidence] != fingerprint_paths:
        raise ValueError("target control input fingerprint path mismatch")
    if [evidence[3] for evidence in manifest_evidence] != launcher_paths:
        raise ValueError("target control launcher config path mismatch")
    for name in (
        "config_sha256",
        "container",
        "dataset",
        "modelopt_sha",
        "speculators_sha",
        "target_model",
        "versions",
        "artifact_identity",
        "runtimes",
    ):
        if manifest_payloads[0].get(name) != manifest_payloads[1].get(name):
            raise ValueError(f"target control manifest mismatch: {name}")
    if manifest_evidence[0][2].get("inputs") != manifest_evidence[1][2].get("inputs"):
        raise ValueError("target control input fingerprint mismatch between cells")
    if {manifest["server_args"][-1] for manifest in manifest_payloads} != {"8000", "8010"}:
        raise ValueError("target control server ports are not isolated")
    if any(manifest.get("slurm_job_id") != allocation["slurm_job_id"] for manifest in manifest_payloads):
        raise ValueError("target control allocation job mismatch")
    replayed = summarize_target_control(
        ledgers[0],
        ledgers[1],
        expected_prompt_schedule=dataset["ordered_prompts"],
        expected_request_sha256=_expected_request_hashes(identity),
    )
    replayed["artifact_identity"] = payload["artifact_identity"]
    replayed["manifests"] = payload["manifests"]
    replayed["input_fingerprints"] = payload["input_fingerprints"]
    replayed["launcher_configs"] = payload["launcher_configs"]
    replayed["allocation_receipt"] = payload["allocation_receipt"]
    replayed["ledgers"] = payload["ledgers"]
    if payload != replayed:
        raise ValueError("target control receipt replay mismatch")
    return {**payload, "receipt_sha256": claim}


def _metric_evidence(
    baseline_run: Path,
    dflash2_run: Path,
    *,
    concurrency: int,
    artifact_identity_sha256: str,
) -> dict[str, object]:
    if concurrency not in {1, 32} or not _is_sha256(artifact_identity_sha256):
        raise ValueError("invalid matched metric evidence request")
    result: dict[str, object] = {}
    for method, run in (("baseline", baseline_run), ("dflash2", dflash2_run)):
        manifest_path = run / "manifest.json"
        fingerprint_path = run / "input-fingerprint.json"
        manifest = _load_json(manifest_path)
        fingerprint = _load_json(fingerprint_path)
        inputs = fingerprint.get("inputs")
        evaluation = manifest.get("evaluation")
        artifact = manifest.get("artifact_identity")
        if (
            manifest.get("status") != "success"
            or manifest.get("method") != method
            or not isinstance(evaluation, dict)
            or evaluation.get("max_concurrency") != concurrency
            or evaluation.get("max_requests") != EVALUATION_REQUESTS_PER_SUBSET
            or evaluation.get("tensor_parallel_size") != 2
            or evaluation.get("temperature") != 0
            or evaluation.get("top_p") != 1
            or not isinstance(artifact, dict)
            or artifact.get("sha256") != artifact_identity_sha256
            or not isinstance(inputs, dict)
            or fingerprint.get("sha256") != _sha_json(inputs)
            or inputs.get("artifact_identity_sha256") != artifact_identity_sha256
        ):
            raise ValueError(f"invalid {method} cell evidence")
        descriptors = {
            "manifest": _file_descriptor(manifest_path),
            "input_fingerprint": _file_descriptor(fingerprint_path),
            "performance_csv": _file_descriptor(run / "perf_results.csv"),
        }
        if method == "dflash2":
            descriptors["acceptance_csv"] = _file_descriptor(run / "acceptance.csv")
        result[method] = descriptors
    return result


def summarize_pair(
    baseline_run: Path,
    dflash2_run: Path,
    *,
    tensor_parallel_size: int,
) -> dict[str, Any]:
    """Compute matched per-subset and macro metrics only after correctness passes."""
    if tensor_parallel_size != 2:
        raise ValueError("Q30 matched evaluation requires TP2+TP2")
    baseline = _read_csv_by_subset(baseline_run / "perf_results.csv")
    draft = _read_csv_by_subset(dflash2_run / "perf_results.csv")
    acceptance = _read_csv_by_subset(dflash2_run / "acceptance.csv")
    rows: list[dict[str, Any]] = []
    for subset in STANDARD_SUBSETS:
        baseline_tps = _positive(baseline[subset], "output_tps_median")
        draft_tps = _positive(draft[subset], "output_tps_median")
        baseline_latency = _positive(baseline[subset], "latency_median_s")
        draft_latency = _positive(draft[subset], "latency_median_s")
        accepted = _nonnegative(acceptance[subset], "num_accepted_tokens")
        drafted = _positive(acceptance[subset], "num_draft_tokens")
        mean_length = _positive(acceptance[subset], "acceptance_length")
        rows.append(
            {
                "subset": subset,
                "output_tps_per_gpu_baseline": baseline_tps / tensor_parallel_size,
                "output_tps_per_gpu_dflash2": draft_tps / tensor_parallel_size,
                "output_tps_speedup": draft_tps / baseline_tps,
                "latency_s_baseline": baseline_latency,
                "latency_s_dflash2": draft_latency,
                "latency_speedup": baseline_latency / draft_latency,
                "acceptance_rate": accepted / drafted,
                "mean_accepted_length": mean_length,
            }
        )

    def macro(name: str) -> float:
        return sum(float(row[name]) for row in rows) / len(rows)

    return {
        "method": "dflash2",
        "block_size": DFLASH2_BLOCK_SIZE,
        "num_speculative_tokens": DFLASH2_SPECULATIVE_TOKENS,
        "tensor_parallel_size": tensor_parallel_size,
        "subsets": rows,
        "aggregate": {
            name: macro(name)
            for name in (
                "output_tps_per_gpu_baseline",
                "output_tps_per_gpu_dflash2",
                "output_tps_speedup",
                "latency_s_baseline",
                "latency_s_dflash2",
                "latency_speedup",
                "acceptance_rate",
                "mean_accepted_length",
            )
        },
    }


def validate_report_receipt(path: Path) -> dict[str, Any]:
    """Replay a published metric receipt from its exact bound cell artifacts."""
    payload = _load_json(path)
    claim = payload.pop("receipt_sha256", None)
    if claim != _sha_json(payload):
        raise ValueError("metric report self-hash mismatch")
    artifact_path = _validate_file_descriptor(payload.get("artifact_identity"))
    correctness_path = _validate_file_descriptor(payload.get("correctness_receipt"))
    artifact_sha = _sha256(artifact_path)
    correctness = _load_json(correctness_path)
    correctness_claim = correctness.pop("receipt_sha256", None)
    if (
        correctness_claim != _sha_json(correctness)
        or correctness.get("status") != "passed"
        or correctness.get("artifact_identity_sha256") != artifact_sha
    ):
        raise ValueError("metric report correctness gate mismatch")
    evidence = payload.get("evidence")
    if not isinstance(evidence, dict) or set(evidence) != {"baseline", "dflash2"}:
        raise ValueError("metric report evidence is incomplete")
    runs: dict[str, Path] = {}
    for method in ("baseline", "dflash2"):
        descriptors = evidence[method]
        expected = {"manifest", "input_fingerprint", "performance_csv"}
        if method == "dflash2":
            expected.add("acceptance_csv")
        if not isinstance(descriptors, dict) or set(descriptors) != expected:
            raise ValueError(f"invalid {method} metric evidence")
        paths = {name: _validate_file_descriptor(value) for name, value in descriptors.items()}
        runs[method] = paths["manifest"].parent
    baseline_manifest = _load_json(runs["baseline"] / "manifest.json")
    evaluation = baseline_manifest.get("evaluation")
    if not isinstance(evaluation, dict) or evaluation.get("max_concurrency") not in {1, 32}:
        raise ValueError("metric report concurrency is invalid")
    concurrency = int(evaluation["max_concurrency"])
    replayed_evidence = _metric_evidence(
        runs["baseline"],
        runs["dflash2"],
        concurrency=concurrency,
        artifact_identity_sha256=artifact_sha,
    )
    replayed_metrics = summarize_pair(runs["baseline"], runs["dflash2"], tensor_parallel_size=2)
    for key, value in replayed_metrics.items():
        if payload.get(key) != value:
            raise ValueError(f"metric report replay mismatch: {key}")
    if payload.get("evidence") != replayed_evidence:
        raise ValueError("metric report evidence replay mismatch")
    return {**payload, "receipt_sha256": claim}


def _completion_request_body(model: str, subset: str, index: int, prompt: str) -> bytes:
    return json.dumps(
        {
            "model": model,
            "prompt": prompt,
            "max_tokens": 64,
            "temperature": 0,
            "top_p": 1,
            "seed": 42,
            "logprobs": 0,
            "return_tokens_as_token_ids": True,
            "request_id": f"specdec-correctness-{subset}-{index}",
        }
    ).encode()


def _expected_request_hashes(identity: dict[str, Any]) -> list[str]:
    target = identity.get("target")
    dataset = identity.get("dataset")
    if not isinstance(target, dict) or not isinstance(dataset, dict):
        raise ValueError("artifact identity target/dataset mismatch")
    model = target.get("path")
    files = dataset.get("files")
    if not isinstance(model, str) or not isinstance(files, dict):
        raise ValueError("artifact identity request inputs mismatch")
    hashes: list[str] = []
    for subset in STANDARD_SUBSETS:
        entry = files.get(subset)
        if not isinstance(entry, dict):
            raise ValueError("artifact identity request dataset mismatch")
        prompts = _read_prompt_prefix(Path(str(entry.get("path", ""))), 200)
        if len(prompts) != 200:
            raise ValueError("target control requires durable exact-200 prompt files")
        for index, (_, prompt) in enumerate(prompts):
            hashes.append(hashlib.sha256(_completion_request_body(model, subset, index, prompt)).hexdigest())
    return hashes


def capture_outputs(
    dataset_manifest_path: Path,
    hf_home: Path,
    output_path: Path,
    *,
    endpoint: str,
    model: str,
    requests_per_subset: int = CORRECTNESS_REQUESTS_PER_SUBSET,
    max_tokens: int = 64,
) -> None:
    """Capture a small deterministic ledger using the same pinned prompt snapshot."""
    if requests_per_subset != CORRECTNESS_REQUESTS_PER_SUBSET or max_tokens != 64:
        raise ValueError("correctness capture requires 200 prompts/subset and max_tokens=64")
    provenance = compute_prompt_set(
        dataset_manifest_path,
        hf_home,
        requests_per_subset=EVALUATION_REQUESTS_PER_SUBSET,
    )
    files = provenance["files"]
    if not isinstance(files, dict):
        raise ValueError("invalid prompt provenance")
    if output_path.exists():
        raise FileExistsError(f"output ledger exists: {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{output_path.name}.", dir=output_path.parent)
    try:
        with os.fdopen(descriptor, "w") as stream:
            for subset in STANDARD_SUBSETS:
                entry = files[subset]
                assert isinstance(entry, dict)
                prompts = _read_prompt_prefix(Path(str(entry["path"])), requests_per_subset)
                for index in range(requests_per_subset):
                    source_row, prompt = prompts[index % len(prompts)]
                    body = _completion_request_body(model, subset, index, prompt)
                    request = urllib.request.Request(
                        endpoint.rstrip("/") + "/completions",
                        data=body,
                        headers={"Content-Type": "application/json"},
                        method="POST",
                    )
                    with urllib.request.urlopen(request, timeout=600) as response:
                        result = json.loads(response.read())
                    choice = result["choices"][0]
                    text = choice["text"]
                    finish = choice["finish_reason"]
                    logprobs = choice.get("logprobs")
                    tokens = logprobs.get("tokens") if isinstance(logprobs, dict) else None
                    if (
                        not isinstance(text, str)
                        or not isinstance(tokens, list)
                        or not all(isinstance(token, str) for token in tokens)
                        or finish not in {"stop", "length"}
                    ):
                        raise ValueError("invalid completion response")
                    record = {
                        "subset": subset,
                        "index": index,
                        "source_row": source_row,
                        "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
                        "output_sha256": hashlib.sha256(text.encode()).hexdigest(),
                        "output_text": text,
                        "output_tokens": tokens,
                        "request_sha256": hashlib.sha256(body).hexdigest(),
                        "seed": 42,
                        "finish_reason": finish,
                    }
                    stream.write(_canonical(record) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, output_path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def main() -> None:
    """Dispatch matched-evaluation artifact operations."""
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)

    prompts = commands.add_parser("prompt-set")
    prompts.add_argument("--dataset-manifest", required=True)
    prompts.add_argument("--hf-home", required=True)
    prompts.add_argument("--output", required=True)

    materialize = commands.add_parser("materialize-prompts")
    materialize.add_argument("--source-manifest", required=True)
    materialize.add_argument("--source-hf-home", required=True)
    materialize.add_argument("--output-hf-home", required=True)
    materialize.add_argument("--output-manifest", required=True)

    capture = commands.add_parser("capture-outputs")
    capture.add_argument("--dataset-manifest", required=True)
    capture.add_argument("--hf-home", required=True)
    capture.add_argument("--output", required=True)
    capture.add_argument("--endpoint", required=True)
    capture.add_argument("--model", required=True)

    compare = commands.add_parser("compare-outputs")
    compare.add_argument("--baseline", required=True)
    compare.add_argument("--dflash2", required=True)
    compare.add_argument("--output", required=True)
    compare.add_argument("--artifact-identity", required=True)

    control = commands.add_parser("analyze-control")
    control.add_argument("--left", required=True)
    control.add_argument("--right", required=True)
    control.add_argument("--left-manifest", required=True)
    control.add_argument("--right-manifest", required=True)
    control.add_argument("--artifact-identity", required=True)
    control.add_argument("--allocation-receipt", required=True)
    control.add_argument("--output", required=True)

    allocation = commands.add_parser("allocation-receipt")
    allocation.add_argument("--output", required=True)

    summary = commands.add_parser("summarize")
    summary.add_argument("--baseline-run", required=True)
    summary.add_argument("--dflash2-run", required=True)
    summary.add_argument("--correctness-receipt", required=True)
    summary.add_argument("--output", required=True)
    summary.add_argument("--artifact-identity", required=True)
    summary.add_argument("--concurrency", required=True, type=int, choices=(1, 32))

    identity = commands.add_parser("artifact-identity")
    identity.add_argument("--output", required=True)
    identity.add_argument("--target", required=True)
    identity.add_argument("--target-sha256", required=True)
    identity.add_argument("--target-receipt", required=True)
    identity.add_argument("--target-receipt-sha256", required=True)
    identity.add_argument("--export", required=True)
    identity.add_argument("--milestone-manifest", required=True)
    identity.add_argument("--dataset-manifest", required=True)
    identity.add_argument("--hf-home", required=True)
    identity.add_argument("--client-runtime-archive", required=True)
    identity.add_argument("--client-runtime-archive-sha256", required=True)
    identity.add_argument("--server-runtime-archive", required=True)
    identity.add_argument("--server-runtime-archive-sha256", required=True)
    identity.add_argument("--server-runtime-receipt-sha256", required=True)

    milestone = commands.add_parser("verify-milestone")
    milestone.add_argument("--manifest", required=True)
    milestone.add_argument("--export", required=True)

    verify_report = commands.add_parser("verify-report")
    verify_report.add_argument("--report", required=True)

    verify_control = commands.add_parser("verify-control")
    verify_control.add_argument("--receipt", required=True)

    args = parser.parse_args()
    if args.command == "prompt-set":
        payload = compute_prompt_set(Path(args.dataset_manifest), Path(args.hf_home))
        payload["receipt_sha256"] = _sha_json(payload)
        _atomic_json(Path(args.output), payload, no_replace=True)
    elif args.command == "materialize-prompts":
        materialize_prompt_set(
            Path(args.source_manifest),
            Path(args.source_hf_home),
            Path(args.output_hf_home),
            Path(args.output_manifest),
        )
    elif args.command == "capture-outputs":
        capture_outputs(
            Path(args.dataset_manifest),
            Path(args.hf_home),
            Path(args.output),
            endpoint=args.endpoint,
            model=args.model,
        )
    elif args.command == "compare-outputs":
        artifact_identity_sha256 = _sha256(Path(args.artifact_identity))
        identity = _load_json(Path(args.artifact_identity))
        dataset = identity.get("dataset")
        if not isinstance(dataset, dict) or not isinstance(dataset.get("ordered_prompts"), list):
            raise ValueError("artifact identity has no ordered prompt schedule")
        payload = validate_output_equivalence(
            Path(args.baseline),
            Path(args.dflash2),
            artifact_identity_sha256=artifact_identity_sha256,
            expected_prompt_schedule=dataset["ordered_prompts"],
        )
        payload["receipt_sha256"] = _sha_json(payload)
        _atomic_json(Path(args.output), payload, no_replace=True)
    elif args.command == "analyze-control":
        build_target_control_receipt(
            Path(args.left),
            Path(args.right),
            Path(args.left_manifest),
            Path(args.right_manifest),
            Path(args.artifact_identity),
            Path(args.allocation_receipt),
            Path(args.output),
        )
    elif args.command == "allocation-receipt":
        _publish_current_allocation_receipt(Path(args.output))
    elif args.command == "summarize":
        correctness = _load_json(Path(args.correctness_receipt))
        claim = correctness.pop("receipt_sha256", None)
        artifact_identity_sha256 = _sha256(Path(args.artifact_identity))
        if (
            correctness.get("status") != "passed"
            or correctness.get("artifact_identity_sha256") != artifact_identity_sha256
            or claim != _sha_json(correctness)
        ):
            raise ValueError("correctness receipt is not a passing self-hashed receipt")
        payload = summarize_pair(
            Path(args.baseline_run), Path(args.dflash2_run), tensor_parallel_size=2
        )
        payload["evidence"] = _metric_evidence(
            Path(args.baseline_run),
            Path(args.dflash2_run),
            concurrency=args.concurrency,
            artifact_identity_sha256=artifact_identity_sha256,
        )
        payload["correctness_receipt"] = _file_descriptor(Path(args.correctness_receipt))
        payload["artifact_identity"] = _file_descriptor(Path(args.artifact_identity))
        payload["receipt_sha256"] = _sha_json(payload)
        _atomic_json(Path(args.output), payload, no_replace=True)
    elif args.command == "artifact-identity":
        payload = build_artifact_identity(
            target_path=Path(args.target),
            target_sha256=args.target_sha256,
            target_receipt_path=Path(args.target_receipt),
            target_receipt_sha256=args.target_receipt_sha256,
            export_path=Path(args.export),
            milestone_manifest_path=Path(args.milestone_manifest),
            dataset_manifest_path=Path(args.dataset_manifest),
            hf_home=Path(args.hf_home),
            client_runtime_archive=Path(args.client_runtime_archive),
            client_runtime_archive_sha256=args.client_runtime_archive_sha256,
            server_runtime_archive=Path(args.server_runtime_archive),
            server_runtime_archive_sha256=args.server_runtime_archive_sha256,
            server_runtime_receipt_sha256=args.server_runtime_receipt_sha256,
        )
        _atomic_json(Path(args.output), payload, no_replace=True)
    elif args.command == "verify-milestone":
        print(_canonical(validate_milestone_export(Path(args.manifest), Path(args.export))))
    elif args.command == "verify-report":
        print(_canonical(validate_report_receipt(Path(args.report))))
    else:
        print(_canonical(validate_target_control_receipt(Path(args.receipt))))


if __name__ == "__main__":
    main()
