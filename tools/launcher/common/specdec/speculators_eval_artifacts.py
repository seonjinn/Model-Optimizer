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

"""Validate Speculators acceptance output and atomically record run provenance."""

from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import json
import math
import os
import tempfile
from datetime import datetime, timezone
from itertools import pairwise
from pathlib import Path
from typing import TypeGuard

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
ACCEPTANCE_TOLERANCE = 1e-6
DATASET_ID = "RedHatAI/speculator_benchmarks"
PERF_COLUMNS = (
    "subset",
    "strategy",
    "max_concurrency",
    "request_budget",
    "completed_requests",
    "rps_median",
    "latency_median_s",
    "itl_median_ms",
    "ttft_median_ms",
    "output_tps_median",
    "total_output_tokens",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _is_lower_hex(value: object, length: int) -> TypeGuard[str]:
    return (
        isinstance(value, str)
        and len(value) == length
        and all(character in "0123456789abcdef" for character in value)
    )


def _finite_float(row: dict[str, str], column: str) -> float:
    try:
        value = float(row[column])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"{row.get('subset', '<unknown>')}: invalid {column}") from error
    if not math.isfinite(value):
        raise ValueError(f"{row.get('subset', '<unknown>')}: non-finite {column}")
    return value


def _nonnegative_int(row: dict[str, str], column: str) -> int:
    value = _finite_float(row, column)
    if not value.is_integer():
        raise ValueError(f"{row.get('subset', '<unknown>')}: fractional {column}")
    integer = int(value)
    if integer < 0:
        raise ValueError(f"{row.get('subset', '<unknown>')}: negative {column}")
    return integer


def validate_acceptance(csv_path: Path, num_speculative_tokens: int) -> None:
    """Require a complete nine-subset acceptance table for the configured K."""
    if not csv_path.is_file() or csv_path.stat().st_size == 0:
        raise ValueError(f"missing or empty acceptance CSV: {csv_path}")

    with csv_path.open(newline="") as file:
        reader = csv.DictReader(file)
        expected_positions = {
            f"acceptance_at_pos_{position}" for position in range(num_speculative_tokens)
        }
        actual_positions = {
            column
            for column in (reader.fieldnames or ())
            if column.startswith("acceptance_at_pos_")
        }
        if actual_positions != expected_positions:
            raise ValueError(
                f"acceptance positions mismatch: expected {sorted(expected_positions)}, "
                f"got {sorted(actual_positions)}"
            )
        rows = list(reader)

    subsets = [row.get("subset", "") for row in rows]
    if len(rows) != len(STANDARD_SUBSETS) or set(subsets) != set(STANDARD_SUBSETS):
        raise ValueError(f"expected exactly the nine standard subsets, got {subsets}")
    for row in rows:
        _validate_acceptance_row(row, num_speculative_tokens)


def _validate_acceptance_row(row: dict[str, str], num_speculative_tokens: int) -> None:
    subset = row["subset"]
    drafts = _nonnegative_int(row, "num_drafts")
    draft_tokens = _nonnegative_int(row, "num_draft_tokens")
    accepted_tokens = _nonnegative_int(row, "num_accepted_tokens")
    acceptance_length = _finite_float(row, "acceptance_length")
    if drafts == 0:
        raise ValueError(f"{subset}: num_drafts must be > 0")
    if draft_tokens != drafts * num_speculative_tokens:
        raise ValueError(f"{subset}: num_draft_tokens must equal num_drafts * K")
    if accepted_tokens > draft_tokens:
        raise ValueError(f"{subset}: accepted tokens exceed drafted tokens")

    rates = [
        _finite_float(row, f"acceptance_at_pos_{position}")
        for position in range(num_speculative_tokens)
    ]
    if any(rate < 0 or rate > 1 for rate in rates):
        raise ValueError(f"{subset}: position acceptance must be within [0, 1]")
    if any(current > previous + ACCEPTANCE_TOLERANCE for previous, current in pairwise(rates)):
        raise ValueError(f"{subset}: position acceptance must be monotone nonincreasing")

    counter_length = 1 + accepted_tokens / drafts
    position_length = 1 + sum(rates)
    if not math.isclose(acceptance_length, counter_length, abs_tol=ACCEPTANCE_TOLERANCE, rel_tol=0):
        raise ValueError(f"{subset}: acceptance_length disagrees with accepted-token counters")
    if not math.isclose(
        acceptance_length, position_length, abs_tol=ACCEPTANCE_TOLERANCE, rel_tol=0
    ):
        raise ValueError(f"{subset}: acceptance_length disagrees with position rates")


def validate_performance(csv_path: Path) -> None:
    """Require finite task-wise performance rows for all standard subsets."""
    if not csv_path.is_file() or csv_path.stat().st_size == 0:
        raise ValueError(f"missing or empty performance CSV: {csv_path}")
    with csv_path.open(newline="") as file:
        reader = csv.DictReader(file)
        if not set(PERF_COLUMNS).issubset(reader.fieldnames or ()):
            raise ValueError("performance CSV is missing required columns")
        rows = list(reader)
    subsets = {row.get("subset", "") for row in rows}
    if len(rows) != len(STANDARD_SUBSETS) or subsets != set(STANDARD_SUBSETS):
        raise ValueError(f"expected exactly the nine standard subsets, got {sorted(subsets)}")
    concurrencies: set[int] = set()
    budgets: set[int] = set()
    for row in rows:
        concurrency, budget = _validate_performance_row(row)
        concurrencies.add(concurrency)
        budgets.add(budget)
    if len(concurrencies) != 1 or len(budgets) != 1:
        raise ValueError("performance rows must use one concurrency and request budget")


def _validate_performance_row(
    row: dict[str, str],
    expected_concurrency: int | None = None,
    expected_budget: int | None = None,
) -> tuple[int, int]:
    if row.get("strategy") != "throughput":
        raise ValueError(f"{row.get('subset', '<unknown>')}: strategy must be throughput")
    for column in PERF_COLUMNS[2:]:
        if _finite_float(row, column) < 0:
            raise ValueError(f"{row['subset']}: negative {column}")
    concurrency = _nonnegative_int(row, "max_concurrency")
    budget = _nonnegative_int(row, "request_budget")
    completed = _nonnegative_int(row, "completed_requests")
    if concurrency not in {1, 8, 32, 128}:
        raise ValueError(f"{row['subset']}: invalid max_concurrency")
    if budget == 0 or completed != budget:
        raise ValueError(f"{row['subset']}: completed requests do not match budget")
    if expected_concurrency is not None and concurrency != expected_concurrency:
        raise ValueError(f"{row['subset']}: max_concurrency mismatch")
    if expected_budget is not None and budget != expected_budget:
        raise ValueError(f"{row['subset']}: request budget mismatch")
    return concurrency, budget


def _successful_median(metrics: dict[str, object], metric_name: str) -> float:
    metric = metrics.get(metric_name)
    if not isinstance(metric, dict):
        raise ValueError(f"missing GuideLLM metric: {metric_name}")
    successful = metric.get("successful")
    if not isinstance(successful, dict):
        raise ValueError(f"missing successful GuideLLM metric: {metric_name}")
    try:
        value = float(successful["median"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"invalid GuideLLM median: {metric_name}") from error
    if not math.isfinite(value) or value < 0:
        raise ValueError(f"invalid GuideLLM median: {metric_name}")
    return value


def append_fixed_performance(
    json_path: Path,
    csv_path: Path,
    subset: str,
    max_concurrency: int,
    max_requests: int,
) -> None:
    """Append one validated fixed-concurrency GuideLLM throughput result."""
    if subset not in STANDARD_SUBSETS:
        raise ValueError(f"unexpected subset: {subset}")
    data = _load_json(json_path)
    benchmarks = data.get("benchmarks")
    if not isinstance(benchmarks, list) or len(benchmarks) != 1:
        raise ValueError("fixed throughput output must contain exactly one benchmark")
    benchmark = benchmarks[0]
    if not isinstance(benchmark, dict):
        raise ValueError("invalid GuideLLM benchmark")
    config = benchmark.get("config")
    state = benchmark.get("scheduler_state")
    metrics = benchmark.get("metrics")
    if not isinstance(config, dict) or not isinstance(state, dict) or not isinstance(metrics, dict):
        raise ValueError("incomplete GuideLLM benchmark")
    strategy = config.get("strategy")
    if not isinstance(strategy, dict) or strategy.get("type_") != "throughput":
        raise ValueError("GuideLLM benchmark is not fixed throughput")
    if strategy.get("max_concurrency") != max_concurrency:
        raise ValueError("GuideLLM max concurrency does not match configuration")
    successful = state.get("successful_requests")
    if successful != max_requests:
        raise ValueError("GuideLLM completed requests do not match the request budget")
    if state.get("errored_requests") != 0 or state.get("cancelled_requests") != 0:
        raise ValueError("GuideLLM benchmark contains failed requests")

    text = metrics.get("text")
    try:
        output = text["tokens"]["output"]["successful"]  # type: ignore[index]
        total_output_tokens = float(output["total_sum"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("invalid GuideLLM total output tokens") from error
    if not math.isfinite(total_output_tokens) or total_output_tokens < 0:
        raise ValueError("invalid GuideLLM total output tokens")

    row: dict[str, object] = {
        "subset": subset,
        "strategy": "throughput",
        "max_concurrency": max_concurrency,
        "request_budget": max_requests,
        "completed_requests": successful,
        "rps_median": _successful_median(metrics, "requests_per_second"),
        "latency_median_s": _successful_median(metrics, "request_latency"),
        "itl_median_ms": _successful_median(metrics, "inter_token_latency_ms"),
        "ttft_median_ms": _successful_median(metrics, "time_to_first_token_ms"),
        "output_tps_median": _successful_median(metrics, "output_tokens_per_second"),
        "total_output_tokens": total_output_tokens,
    }
    existing: list[dict[str, str]] = []
    if csv_path.exists():
        with csv_path.open(newline="") as file:
            reader = csv.DictReader(file)
            if tuple(reader.fieldnames or ()) != PERF_COLUMNS:
                raise ValueError("performance CSV schema mismatch")
            existing = list(reader)
        if any(item.get("subset") == subset for item in existing):
            raise ValueError(f"duplicate performance subset: {subset}")
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{csv_path.name}.", dir=csv_path.parent)
    try:
        with os.fdopen(fd, "w", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=PERF_COLUMNS)
            writer.writeheader()
            writer.writerows(existing)
            writer.writerow(row)
            file.flush()
            os.fsync(file.fileno())
        os.replace(temporary, csv_path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def validate_fixed_subset(
    subset_dir: Path,
    subset: str,
    method: str,
    num_speculative_tokens: int,
    max_concurrency: int,
    max_requests: int,
) -> None:
    """Validate one durable subset so a resumed job can safely skip it."""
    perf_path = subset_dir / "perf_results.csv"
    with perf_path.open(newline="") as file:
        perf_reader = csv.DictReader(file)
        if tuple(perf_reader.fieldnames or ()) != PERF_COLUMNS:
            raise ValueError("performance CSV schema mismatch")
        perf_rows = list(perf_reader)
    if len(perf_rows) != 1 or perf_rows[0].get("subset") != subset:
        raise ValueError(f"{subset}: expected exactly one performance row")
    _validate_performance_row(perf_rows[0], max_concurrency, max_requests)
    if method == "baseline":
        if (subset_dir / "acceptance.csv").exists():
            raise ValueError(f"{subset}: baseline must not have acceptance output")
        return
    with (subset_dir / "acceptance.csv").open(newline="") as file:
        acceptance_reader = csv.DictReader(file)
        rows = list(acceptance_reader)
    expected_positions = {
        f"acceptance_at_pos_{position}" for position in range(num_speculative_tokens)
    }
    actual_positions = {
        column
        for column in (acceptance_reader.fieldnames or ())
        if column.startswith("acceptance_at_pos_")
    }
    if len(rows) != 1 or rows[0].get("subset") != subset:
        raise ValueError(f"{subset}: expected exactly one acceptance row")
    if actual_positions != expected_positions:
        raise ValueError(f"{subset}: acceptance positions mismatch")
    _validate_acceptance_row(rows[0], num_speculative_tokens)


def consolidate_fixed_results(
    run_dir: Path,
    method: str,
    num_speculative_tokens: int,
    max_concurrency: int,
    max_requests: int,
) -> None:
    """Atomically consolidate nine validated durable subset results."""
    perf_rows: list[dict[str, str]] = []
    acceptance_rows: list[dict[str, str]] = []
    acceptance_fields: list[str] | None = None
    for subset in STANDARD_SUBSETS:
        subset_dir = run_dir / "subsets" / subset
        validate_fixed_subset(
            subset_dir,
            subset,
            method,
            num_speculative_tokens,
            max_concurrency,
            max_requests,
        )
        with (subset_dir / "perf_results.csv").open(newline="") as file:
            perf_rows.extend(csv.DictReader(file))
        if method != "baseline":
            with (subset_dir / "acceptance.csv").open(newline="") as file:
                reader = csv.DictReader(file)
                fields = list(reader.fieldnames or ())
                if acceptance_fields is None:
                    acceptance_fields = fields
                elif fields != acceptance_fields:
                    raise ValueError("acceptance CSV schema mismatch across subsets")
                acceptance_rows.extend(reader)

    outputs: list[tuple[Path, tuple[str, ...] | list[str], list[dict[str, str]]]] = [
        (run_dir / "perf_results.csv", PERF_COLUMNS, perf_rows)
    ]
    if method != "baseline":
        assert acceptance_fields is not None
        outputs.append((run_dir / "acceptance.csv", acceptance_fields, acceptance_rows))
    for output, fields, rows in outputs:
        fd, temporary = tempfile.mkstemp(prefix=f".{output.name}.", dir=output.parent)
        try:
            with os.fdopen(fd, "w", newline="") as file:
                writer = csv.DictWriter(file, fieldnames=fields)
                writer.writeheader()
                writer.writerows(rows)
                file.flush()
                os.fsync(file.fileno())
            os.replace(temporary, output)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)


def _load_json(path: Path) -> dict[str, object]:
    with path.open() as file:
        value = json.load(file)
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _dataset_provenance(manifest_path: Path, hf_home: Path) -> dict[str, object]:
    manifest = _load_json(manifest_path)
    revision = manifest.get("revision")
    files = manifest.get("files")
    recorded_home = Path(str(manifest.get("hf_home", ""))).resolve()
    resolved_home = hf_home.resolve()
    if manifest.get("dataset_id") != DATASET_ID:
        raise ValueError(f"dataset identity must be {DATASET_ID}")
    if not _is_lower_hex(revision, 40):
        raise ValueError("dataset revision must be a pinned 40-character commit")
    if recorded_home != resolved_home:
        raise ValueError("dataset manifest HF_HOME does not match the active HF_HOME")
    if not isinstance(files, dict) or set(files) != set(STANDARD_SUBSETS):
        raise ValueError("dataset manifest must contain exactly the nine standard subsets")
    snapshot_root = (
        resolved_home / "hub" / "datasets--RedHatAI--speculator_benchmarks" / "snapshots" / revision
    ).resolve()

    verified_files: dict[str, dict[str, str]] = {}
    for subset in STANDARD_SUBSETS:
        entry = files[subset]
        if not isinstance(entry, dict):
            raise ValueError(f"invalid dataset file entry: {subset}")
        path = Path(os.path.abspath(str(entry.get("path", ""))))
        if path != snapshot_root / f"{subset}.jsonl":
            raise ValueError(f"dataset file is not the pinned subset JSONL: {path}")
        if path.is_symlink():
            blobs_root = (snapshot_root.parents[1] / "blobs").resolve()
            try:
                path.resolve(strict=True).relative_to(blobs_root)
            except (FileNotFoundError, ValueError) as error:
                raise ValueError(
                    f"dataset snapshot link escapes its blobs directory: {path}"
                ) from error
        expected_sha = entry.get("sha256")
        if not path.is_file() or not _is_lower_hex(expected_sha, 64):
            raise ValueError(f"invalid staged dataset file: {path}")
        if _sha256(path) != expected_sha:
            raise ValueError(f"dataset file hash mismatch: {path}")
        verified_files[subset] = {"path": str(path), "sha256": expected_sha}
    return {
        "dataset_id": DATASET_ID,
        "revision": revision,
        "hf_home": str(resolved_home),
        "manifest_path": str(manifest_path.resolve()),
        "manifest_sha256": _sha256(manifest_path),
        "files": verified_files,
    }


def _resolved_slurm_container(launcher_config: Path) -> Path:
    keys: list[tuple[int, str]] = []
    for raw_line in launcher_config.read_text().splitlines():
        content = raw_line.lstrip()
        if not content or content.startswith("#") or ":" not in content:
            continue
        indentation = len(raw_line) - len(content)
        key, value = content.split(":", 1)
        while keys and keys[-1][0] >= indentation:
            keys.pop()
        keys.append((indentation, key.strip()))
        if [item[1] for item in keys] == ["pipeline", "task_0", "slurm_config", "container"]:
            scalar = value.strip()
            if not scalar:
                break
            if scalar[0:1] in {'"', "'"}:
                scalar = ast.literal_eval(scalar)
            return Path(scalar).resolve()
    raise ValueError("resolved launcher config has no pipeline.task_0.slurm_config.container")


def _container_provenance(
    identity_path: Path, image_path: Path, launcher_config: Path
) -> dict[str, object]:
    identity = _load_json(identity_path)
    resolved_image = image_path.resolve()
    if Path(str(identity.get("path", ""))).resolve() != resolved_image:
        raise ValueError("container identity path does not match the configured image")
    digest = identity.get("sha256")
    size_bytes = identity.get("size_bytes")
    if not _is_lower_hex(digest, 64):
        raise ValueError("container identity must contain a 64-character SHA256")
    if not isinstance(size_bytes, int) or not resolved_image.is_file():
        raise ValueError(f"container image is missing: {resolved_image}")
    if resolved_image.stat().st_size != size_bytes:
        raise ValueError("container image size does not match its identity sidecar")
    if _sha256(resolved_image) != digest:
        raise ValueError("container image hash mismatch")
    if _resolved_slurm_container(launcher_config) != resolved_image:
        raise ValueError("resolved Slurm container does not match the configured image")
    return {
        "path": str(resolved_image),
        "sha256": digest,
        "size_bytes": size_bytes,
        "identity_path": str(identity_path.resolve()),
        "identity_sha256": _sha256(identity_path),
    }


def verify_inputs(args: argparse.Namespace) -> None:
    """Verify the offline dataset snapshot and immutable container identity."""
    _dataset_provenance(Path(args.dataset_manifest), Path(args.hf_home))
    _container_provenance(
        Path(args.container_identity), Path(args.container_image), Path(args.launcher_config)
    )


def print_dataset_paths(args: argparse.Namespace) -> None:
    """Print verified subset/path pairs for sequential local evaluation."""
    dataset = _dataset_provenance(Path(args.dataset_manifest), Path(args.hf_home))
    files = dataset["files"]
    assert isinstance(files, dict)
    for subset in STANDARD_SUBSETS:
        entry = files[subset]
        assert isinstance(entry, dict)
        print(f"{subset}\t{entry['path']}")


def write_manifest(args: argparse.Namespace) -> None:
    """Atomically write immutable inputs and the run's final status."""
    config_paths = {
        "target": Path(args.target_model) / "config.json",
        "launcher": Path(args.launcher_config),
    }
    if args.draft_model:
        config_paths["draft"] = Path(args.draft_model) / "config.json"
    try:
        container = _container_provenance(
            Path(args.container_identity),
            Path(args.container_image),
            Path(args.launcher_config),
        )
        dataset = _dataset_provenance(Path(args.dataset_manifest), Path(args.hf_home))
    except ValueError as error:
        if args.status == "success":
            raise
        container = {
            "path": str(Path(args.container_image).resolve()),
            "identity_path": str(Path(args.container_identity).resolve()),
        }
        dataset = {
            "manifest_path": str(Path(args.dataset_manifest).resolve()),
            "hf_home": str(Path(args.hf_home).resolve()),
        }
        provenance_error = str(error)
    else:
        provenance_error = None

    payload = {
        "status": args.status,
        "recorded_at": datetime.now(timezone.utc).isoformat(),
        "method": args.method,
        "block_size": args.block_size,
        "num_speculative_tokens": args.num_speculative_tokens,
        "target_model": args.target_model,
        "draft_model": args.draft_model or None,
        "speculators_repo": args.speculators_repo,
        "speculators_sha": args.speculators_sha,
        "modelopt_repo": args.modelopt_repo,
        "modelopt_sha": args.modelopt_sha,
        "modelopt_dirty": False if args.modelopt_dirty == "false" else None,
        "runtime": args.runtime,
        "container": container,
        "dataset": dataset,
        "provenance_error": provenance_error,
        "slurm_job_id": args.slurm_job_id,
        "launcher_config": args.launcher_config,
        "config_sha256": {name: _sha256(path) for name, path in config_paths.items()},
        "versions": {
            "python": args.python_version,
            "vllm": args.vllm_version,
            "guidellm": args.guidellm_version,
        },
        "server_args": args.server_arg,
        "evaluator_args": args.evaluator_arg,
        "evaluation": {
            "dataset": DATASET_ID,
            "subsets": list(STANDARD_SUBSETS),
            "temperature": 0,
            "top_p": 1,
            "mode": args.evaluation_mode,
            "max_concurrency": args.max_concurrency,
            "max_requests": args.max_requests,
            "tensor_parallel_size": args.tensor_parallel_size,
        },
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{output.name}.", dir=output.parent)
    try:
        with os.fdopen(fd, "w") as file:
            json.dump(payload, file, indent=2, sort_keys=True)
            file.write("\n")
            file.flush()
            os.fsync(file.fileno())
        os.replace(temporary, output)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def main() -> None:
    """Dispatch CSV validation or manifest generation."""
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    validate = subparsers.add_parser("validate")
    validate.add_argument("--csv", required=True)
    validate.add_argument("--num-speculative-tokens", type=int, required=True)

    validate_perf = subparsers.add_parser("validate-perf")
    validate_perf.add_argument("--csv", required=True)

    append_perf = subparsers.add_parser("append-fixed-perf")
    append_perf.add_argument("--json", required=True)
    append_perf.add_argument("--csv", required=True)
    append_perf.add_argument("--subset", required=True)
    append_perf.add_argument("--max-concurrency", type=int, required=True)
    append_perf.add_argument("--max-requests", type=int, required=True)

    validate_subset = subparsers.add_parser("validate-fixed-subset")
    validate_subset.add_argument("--dir", required=True)
    validate_subset.add_argument("--subset", required=True)
    validate_subset.add_argument("--method", required=True)
    validate_subset.add_argument("--num-speculative-tokens", type=int, required=True)
    validate_subset.add_argument("--max-concurrency", type=int, required=True)
    validate_subset.add_argument("--max-requests", type=int, required=True)

    consolidate = subparsers.add_parser("consolidate-fixed")
    consolidate.add_argument("--run-dir", required=True)
    consolidate.add_argument("--method", required=True)
    consolidate.add_argument("--num-speculative-tokens", type=int, required=True)
    consolidate.add_argument("--max-concurrency", type=int, required=True)
    consolidate.add_argument("--max-requests", type=int, required=True)

    verify = subparsers.add_parser("verify-inputs")
    verify.add_argument("--dataset-manifest", required=True)
    verify.add_argument("--hf-home", required=True)
    verify.add_argument("--container-identity", required=True)
    verify.add_argument("--container-image", required=True)
    verify.add_argument("--launcher-config", required=True)

    dataset_paths = subparsers.add_parser("dataset-paths")
    dataset_paths.add_argument("--dataset-manifest", required=True)
    dataset_paths.add_argument("--hf-home", required=True)

    manifest = subparsers.add_parser("manifest")
    manifest.add_argument("--output", required=True)
    manifest.add_argument("--status", choices=("success", "failed"), required=True)
    manifest.add_argument("--method", required=True)
    manifest.add_argument("--block-size", type=int, required=True)
    manifest.add_argument("--num-speculative-tokens", type=int, required=True)
    manifest.add_argument("--target-model", required=True)
    manifest.add_argument("--draft-model", required=True)
    manifest.add_argument("--speculators-repo", required=True)
    manifest.add_argument("--speculators-sha", required=True)
    manifest.add_argument("--modelopt-repo", required=True)
    manifest.add_argument("--modelopt-sha", required=True)
    manifest.add_argument("--modelopt-dirty", choices=("false", "unknown"), required=True)
    manifest.add_argument("--runtime", required=True)
    manifest.add_argument("--container-image", required=True)
    manifest.add_argument("--container-identity", required=True)
    manifest.add_argument("--dataset-manifest", required=True)
    manifest.add_argument("--hf-home", required=True)
    manifest.add_argument("--slurm-job-id", required=True)
    manifest.add_argument("--launcher-config", required=True)
    manifest.add_argument("--python-version", required=True)
    manifest.add_argument("--vllm-version", required=True)
    manifest.add_argument("--guidellm-version", required=True)
    manifest.add_argument("--server-arg", action="append", default=[])
    manifest.add_argument("--evaluator-arg", action="append", default=[])
    manifest.add_argument("--max-concurrency", type=int, required=True)
    manifest.add_argument("--max-requests", type=int, required=True)
    manifest.add_argument("--tensor-parallel-size", type=int, required=True)
    manifest.add_argument("--evaluation-mode", choices=("throughput", "sweep"), required=True)
    args = parser.parse_args()

    if args.command == "validate":
        validate_acceptance(Path(args.csv), args.num_speculative_tokens)
    elif args.command == "validate-perf":
        validate_performance(Path(args.csv))
    elif args.command == "append-fixed-perf":
        append_fixed_performance(
            Path(args.json),
            Path(args.csv),
            args.subset,
            args.max_concurrency,
            args.max_requests,
        )
    elif args.command == "validate-fixed-subset":
        validate_fixed_subset(
            Path(args.dir),
            args.subset,
            args.method,
            args.num_speculative_tokens,
            args.max_concurrency,
            args.max_requests,
        )
    elif args.command == "consolidate-fixed":
        consolidate_fixed_results(
            Path(args.run_dir),
            args.method,
            args.num_speculative_tokens,
            args.max_concurrency,
            args.max_requests,
        )
    elif args.command == "verify-inputs":
        verify_inputs(args)
    elif args.command == "dataset-paths":
        print_dataset_paths(args)
    else:
        write_manifest(args)


if __name__ == "__main__":
    main()
