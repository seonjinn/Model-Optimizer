# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Fail-closed contracts for the matched Q30 DFlash2 step-4166 evaluation."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import re
import tempfile
import urllib.request
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
        "sampling": {"temperature": 0, "top_p": 1},
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
                "finish_reason",
            }
            or not all(_is_sha256(row.get(name)) for name in ("prompt_sha256", "output_sha256"))
            or isinstance(row.get("source_row"), bool)
            or not isinstance(row.get("source_row"), int)
            or row["source_row"] < 0
            or row.get("finish_reason") not in {"stop", "length"}
        ):
            raise ValueError("output ledger row schema mismatch")
    return records


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
                    body = json.dumps(
                        {
                            "model": model,
                            "prompt": prompt,
                            "max_tokens": max_tokens,
                            "temperature": 0,
                            "top_p": 1,
                        }
                    ).encode()
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
                    if not isinstance(text, str) or finish not in {"stop", "length"}:
                        raise ValueError("invalid completion response")
                    record = {
                        "subset": subset,
                        "index": index,
                        "source_row": source_row,
                        "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
                        "output_sha256": hashlib.sha256(text.encode()).hexdigest(),
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
    else:
        print(_canonical(validate_report_receipt(Path(args.report))))


if __name__ == "__main__":
    main()
