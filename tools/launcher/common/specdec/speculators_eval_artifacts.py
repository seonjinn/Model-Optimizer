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
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path

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


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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
    if any(float(row.get("num_drafts", "0")) <= 0 for row in rows):
        raise ValueError("every subset must report num_drafts > 0")


def write_manifest(args: argparse.Namespace) -> None:
    """Atomically write immutable inputs and the run's final status."""
    config_paths = {
        "target": Path(args.target_model) / "config.json",
        "draft": Path(args.draft_model) / "config.json",
        "launcher": Path(args.launcher_config),
    }
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
        "runtime": args.runtime,
        "container_image": args.container_image,
        "slurm_job_id": args.slurm_job_id,
        "launcher_config": args.launcher_config,
        "config_sha256": {name: _sha256(path) for name, path in config_paths.items()},
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
    manifest.add_argument("--runtime", required=True)
    manifest.add_argument("--container-image", required=True)
    manifest.add_argument("--slurm-job-id", required=True)
    manifest.add_argument("--launcher-config", required=True)
    args = parser.parse_args()

    if args.command == "validate":
        validate_acceptance(Path(args.csv), args.num_speculative_tokens)
    else:
        write_manifest(args)


if __name__ == "__main__":
    main()
