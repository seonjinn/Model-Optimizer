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
    try:
        value = int(row[column])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"{row.get('subset', '<unknown>')}: invalid {column}") from error
    if value < 0:
        raise ValueError(f"{row.get('subset', '<unknown>')}: negative {column}")
    return value


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
        if not math.isclose(
            acceptance_length, counter_length, abs_tol=ACCEPTANCE_TOLERANCE, rel_tol=0
        ):
            raise ValueError(f"{subset}: acceptance_length disagrees with accepted-token counters")
        if not math.isclose(
            acceptance_length, position_length, abs_tol=ACCEPTANCE_TOLERANCE, rel_tol=0
        ):
            raise ValueError(f"{subset}: acceptance_length disagrees with position rates")


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
        resolved_home / "datasets--RedHatAI--speculator_benchmarks" / "snapshots" / revision
    ).resolve()

    verified_files: dict[str, dict[str, str]] = {}
    for subset in STANDARD_SUBSETS:
        entry = files[subset]
        if not isinstance(entry, dict):
            raise ValueError(f"invalid dataset file entry: {subset}")
        path = Path(str(entry.get("path", ""))).resolve()
        try:
            path.relative_to(snapshot_root)
        except ValueError as error:
            raise ValueError(f"dataset file is outside the pinned snapshot: {path}") from error
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


def _container_provenance(identity_path: Path, image_path: Path) -> dict[str, object]:
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
    _container_provenance(Path(args.container_identity), Path(args.container_image))


def write_manifest(args: argparse.Namespace) -> None:
    """Atomically write immutable inputs and the run's final status."""
    config_paths = {
        "target": Path(args.target_model) / "config.json",
        "draft": Path(args.draft_model) / "config.json",
        "launcher": Path(args.launcher_config),
    }
    try:
        container = _container_provenance(Path(args.container_identity), Path(args.container_image))
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
        "draft_model": args.draft_model,
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

    verify = subparsers.add_parser("verify-inputs")
    verify.add_argument("--dataset-manifest", required=True)
    verify.add_argument("--hf-home", required=True)
    verify.add_argument("--container-identity", required=True)
    verify.add_argument("--container-image", required=True)

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
    args = parser.parse_args()

    if args.command == "validate":
        validate_acceptance(Path(args.csv), args.num_speculative_tokens)
    elif args.command == "verify-inputs":
        verify_inputs(args)
    else:
        write_manifest(args)


if __name__ == "__main__":
    main()
