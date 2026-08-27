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

"""Contracts for selective, content-addressed Hugging Face dataset staging."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
from pathlib import Path

import pyarrow.parquet as pq
import pytest
from common.specdec.stage_hf_subset import load_subset_plan, materialize_subset, verify_completion

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _plan(tmp_path: Path, source: Path) -> Path:
    plan = tmp_path / "plan.json"
    plan.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "name": "fixture",
                "container": {
                    "path": "/lustre/container.sqsh",
                    "sha256": "e" * 64,
                },
                "files": [
                    {
                        "source_id": "org/repo",
                        "revision": "a" * 40,
                        "license": "Apache-2.0",
                        "pool": "ptv3",
                        "category": "science_reasoning",
                        "response_source": "target-synth",
                        "tool_lane": "none",
                        "path": "data/sample.jsonl",
                        "bytes": source.stat().st_size,
                        "sha256": _sha256(source),
                    }
                ],
            }
        )
        + "\n"
    )
    return plan


def test_materialize_subset_converts_jsonl_to_zstd_parquet_and_receipts(
    tmp_path: Path,
) -> None:
    """JSONL staging produces compressed rows and complete hash receipts."""
    source = tmp_path / "source.jsonl"
    source.write_text(
        '{"messages":[{"role":"user","content":"q1"}]}\n'
        '{"messages":[{"role":"user","content":"q2"}]}\n'
    )
    plan_path = _plan(tmp_path, source)
    output = tmp_path / "published"

    manifest = materialize_subset(
        load_subset_plan(plan_path),
        output,
        fetch=lambda _record, destination: destination.write_bytes(source.read_bytes()),
    )

    assert manifest["file_count"] == 1
    assert manifest["row_count"] == 2
    assert manifest["files"][0]["source_sha256"] == _sha256(source)
    shard = output / manifest["files"][0]["path"]
    metadata = pq.read_metadata(shard)
    assert metadata.metadata[b"ARROW:schema"]
    assert metadata.row_group(0).column(0).compression == "ZSTD"
    assert [
        json.loads(row["raw_json"])["messages"][0]["content"]
        for row in pq.read_table(shard).to_pylist()
    ] == ["q1", "q2"]
    verify_completion(output, expected_plan_sha256=_sha256(plan_path))


def test_materialize_subset_rejects_source_lfs_hash_mismatch(tmp_path: Path) -> None:
    """A mismatched LFS object is never published."""
    source = tmp_path / "source.jsonl"
    source.write_text('{"messages":[]}\n')
    plan_path = _plan(tmp_path, source)
    payload = json.loads(plan_path.read_text())
    payload["files"][0]["sha256"] = "f" * 64
    plan_path.write_text(json.dumps(payload) + "\n")

    with pytest.raises(ValueError, match="source file identity mismatch"):
        materialize_subset(
            load_subset_plan(plan_path),
            tmp_path / "published",
            fetch=lambda _record, destination: destination.write_bytes(source.read_bytes()),
        )

    assert not (tmp_path / "published").exists()


def test_completed_subset_is_rehashed_before_reuse(tmp_path: Path) -> None:
    """A completed subset is reused only after every shard is rehashed."""
    source = tmp_path / "source.jsonl"
    source.write_text('{"messages":[{"role":"user","content":"q"}]}\n')
    plan_path = _plan(tmp_path, source)
    plan = load_subset_plan(plan_path)
    output = tmp_path / "published"
    fetch_calls = 0

    def fetch(_record: dict, destination: Path) -> None:
        nonlocal fetch_calls
        fetch_calls += 1
        destination.write_bytes(source.read_bytes())

    materialize_subset(plan, output, fetch=fetch)
    materialize_subset(plan, output, fetch=fetch)
    assert fetch_calls == 1

    shard = next(output.glob("*.parquet"))
    shard.write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="published subset file identity mismatch"):
        materialize_subset(plan, output, fetch=fetch)


def test_qwen4b_ptv3_plan_is_selective_and_fully_content_pinned() -> None:
    """The study stages only the seven approved, immutable PTV3 files."""
    plan = load_subset_plan(REPOSITORY_ROOT / "examples/dataset/qwen3_4b_ptv3_subset_v1.json")

    assert len(plan["files"]) == 7
    assert sum(record["bytes"] for record in plan["files"]) == 20_620_619_533
    assert {record["category"] for record in plan["files"]} == {
        "agentic_tool",
        "chat",
        "code",
        "math",
        "multilingual",
        "science",
        "swe",
    }
    assert all(record["revision"] != "main" for record in plan["files"])


@pytest.mark.parametrize(
    ("explicit_gpu_flag", "expected_gpu_arg"),
    [(False, None), (True, "--gpus-per-node=4")],
)
def test_subset_submitter_honors_the_profile_gpu_flag_contract(
    tmp_path: Path,
    explicit_gpu_flag: bool,
    expected_gpu_arg: str | None,
) -> None:
    """The real dry-run omits unsupported GPU flags on exclusive partitions."""
    profile = tmp_path / "profile.yaml"
    profile.write_text(
        "\n".join(
            [
                "name: fixture",
                f"modelopt_commit: {'a' * 40}",
                "ssh_host: fixture",
                "account: fixture-account",
                "partition: fixture-partition",
                "fallback_partition: null",
                "durable_root: /lustre/fixture",
                "scratch_candidates:",
                "  - /raid/scratch",
                "training_nodes: 16",
                "training_segment: 16",
                "evaluation_nodes: 1",
                "evaluation_segment: 1",
                "gpus_per_node: 4",
                f"explicit_gpu_flag: {str(explicit_gpu_flag).lower()}",
                "walltime: '01:00:00'",
            ]
        )
        + "\n"
    )
    readiness = tmp_path / "readiness.json"
    readiness.write_text(
        json.dumps(
            {
                "profile": "fixture",
                "account": "fixture-account",
                "partition": "fixture-partition",
                "scratch_root": "/raid/scratch",
                "pyxis_available": True,
                "architecture": "aarch64",
                "gpu_count": 4,
            }
        )
        + "\n"
    )
    source = tmp_path / "source.jsonl"
    source.write_text('{"messages":[]}\n')
    plan = _plan(tmp_path, source)
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    calls = tmp_path / "sbatch.calls"
    _write_executable(
        fake_bin / "git",
        """#!/bin/sh
case "$*" in
  *"rev-parse --show-toplevel"*) printf '%s\\n' "$FAKE_REPO_ROOT" ;;
  *"rev-parse HEAD"*) printf '%040d\\n' 0 ;;
  *"status --porcelain"*) : ;;
  *) exit 2 ;;
esac
""",
    )
    _write_executable(fake_bin / "mkdir", "#!/bin/sh\nexit 0\n")
    _write_executable(
        fake_bin / "sbatch",
        '#!/bin/sh\nprintf "%s\\n" "$*" >> "$SBATCH_CALLS"\n',
    )
    submitter = REPOSITORY_ROOT / "tools/launcher/common/specdec/submit_hf_subset_stage.sh"
    bash3_mapfile_compat = """
mapfile() {
  identity=()
  while IFS= read -r line; do identity+=("$line"); done
}
export -f mapfile
exec "$@"
"""
    result = subprocess.run(
        [
            "/bin/bash",
            "-c",
            bash3_mapfile_compat,
            "subset-submit-test",
            str(submitter),
            "--profile",
            str(profile),
            "--readiness",
            str(readiness),
            "--plan",
            str(plan),
            "--output-root",
            "/lustre/fixture/output",
            "--dry-run",
        ],
        check=False,
        capture_output=True,
        text=True,
        env={
            **os.environ,
            "DRAFTER_LAUNCHER_ROOT": str(REPOSITORY_ROOT / "tools/launcher"),
            "FAKE_REPO_ROOT": str(REPOSITORY_ROOT),
            "PATH": f"{fake_bin}:{os.environ['PATH']}",
            "SBATCH_CALLS": str(calls),
        },
    )

    assert result.returncode == 0, result.stderr
    sbatch_args = calls.read_text()
    if expected_gpu_arg is None:
        assert "--gpus-per-node" not in sbatch_args
    else:
        assert expected_gpu_arg in sbatch_args
    assert "--time=01:00:00" in sbatch_args
    assert "--test-only" in sbatch_args


def _write_executable(path: Path, contents: str) -> None:
    path.write_text(contents)
    path.chmod(0o755)
