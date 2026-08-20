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

"""Behavioral contracts for the pinned Speculators acceptance evaluator."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

_LAUNCHER_DIR = Path(__file__).resolve().parents[1]
_WRAPPER = _LAUNCHER_DIR / "common/specdec/run_speculators_eval.sh"
_RECIPE = _LAUNCHER_DIR / "examples/Qwen/Qwen3-30B-A3B/speculators_eval.yaml"
_SPECULATORS_SHA = "0b08a89a83b92007be63f128e01497455b0209df"
_SUBSETS = (
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


def _write_executable(path: Path, body: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(body)
    path.chmod(0o755)


def _make_harness(
    tmp_path: Path,
    *,
    method: str = "dspark",
    block_size: int = 8,
    num_spec_tokens: int = 8,
    repo_sha: str = _SPECULATORS_SHA,
    evaluator_exit: int = 0,
    csv_positions: int | None = None,
    drafts: int = 10,
    spec_metrics_ready: bool = True,
) -> tuple[dict[str, str], Path, Path]:
    runtime = tmp_path / "runtime"
    runtime_bin = runtime / "bin"
    repo = tmp_path / "shared-speculators"
    evaluator = repo / "scripts/evaluate/evaluate.py"
    evaluator.parent.mkdir(parents=True)
    evaluator.write_text("# staged evaluator fixture\n")
    (repo / ".git").mkdir()

    target = tmp_path / "target"
    draft = tmp_path / "draft"
    target.mkdir()
    draft.mkdir()
    (target / "config.json").write_text('{"model_type":"qwen3_moe"}\n')
    (draft / "config.json").write_text('{"architectures":["Qwen3DSparkModel"]}\n')
    launcher_config = tmp_path / "resolved.yaml"
    launcher_config.write_text("job_name: fixture\n")
    hf_home = tmp_path / "shared-cache"
    hf_home.mkdir()
    output_root = tmp_path / "results"
    invocation_log = tmp_path / "invocations.log"
    server_pid_file = tmp_path / "server.pid"
    positions = num_spec_tokens if csv_positions is None else csv_positions
    acceptance_fixture = tmp_path / "acceptance-fixture.csv"
    acceptance_fixture.write_text(_acceptance_csv_text(positions, drafts))

    activate = runtime_bin / "activate"
    activate.parent.mkdir(parents=True)
    activate.write_text(f'export PATH="{runtime_bin}:$PATH"\n')
    _write_executable(runtime_bin / "guidellm", "#!/bin/bash\nexit 0\n")

    _write_executable(
        runtime_bin / "git",
        f"""#!/bin/bash
printf 'git %s\\n' "$*" >> "{invocation_log}"
if [[ "$*" == *" rev-parse HEAD" ]]; then
  printf '%s\\n' "{repo_sha}"
elif [[ "$*" == *" status --porcelain" ]]; then
  exit 0
else
  printf 'unexpected git command: %s\\n' "$*" >&2
  exit 97
fi
""",
    )
    _write_executable(
        runtime_bin / "pip",
        f"""#!/bin/bash
printf 'pip %s\\n' "$*" >> "{invocation_log}"
exit 98
""",
    )
    _write_executable(
        runtime_bin / "curl",
        f"""#!/bin/bash
printf 'curl %s\\n' "$*" >> "{invocation_log}"
case "$*" in
  *'/health'*) exit 0 ;;
  *'/v1/models'*) printf '{{"data":[{{"id":"target-model"}}]}}\\n' ;;
  *'/metrics'*) printf '{"vllm:spec_decode_num_drafts 1" if spec_metrics_ready else "vllm:num_requests 1"}\\n' ;;
  *) exit 96 ;;
esac
""",
    )
    _write_executable(
        runtime_bin / "python3",
        f"""#!/bin/bash
printf 'python3 %s\\n' "$*" >> "{invocation_log}"
if [[ "$1" == "-m" && "$2" == "vllm.entrypoints.cli.main" ]]; then
  printf '%s\\n' "$$" > "{server_pid_file}"
  trap 'exit 0' TERM INT
  while true; do sleep 1; done
elif [[ "$1" == "{evaluator}" ]]; then
  [[ "${{HF_HUB_OFFLINE:-}}" == "1" ]] || exit 91
  [[ "${{HF_DATASETS_OFFLINE:-}}" == "1" ]] || exit 92
  [[ "${{PYTHONPATH%%:*}}" == "{repo / "src"}" ]] || exit 90
  [[ "$2" == "--target" ]] || exit 89
  [[ "${{@: -1}}" == "throughput" ]] || exit 88
  [[ "$*" == *'--dataset RedHatAI/speculator_benchmarks'* ]] || exit 93
  [[ "$*" == *'--gen-kwargs {{"temperature":0}}'* ]] || exit 94
  [[ "$*" == *'--max-concurrency 128'* ]] || exit 87
  [[ "$*" == *'--max-requests 200'* ]] || exit 86
  while [[ $# -gt 0 ]]; do
    if [[ "$1" == "--output-dir" ]]; then
      shift
      output_dir="$1"
      break
    fi
    shift
  done
  mkdir -p "$output_dir"
  cp "{acceptance_fixture}" "$output_dir/acceptance.csv"
  exit {evaluator_exit}
else
  exec "{sys.executable}" "$@"
fi
""",
    )

    env = {
        **os.environ,
        "SPECULATORS_RUNTIME": str(runtime),
        "SPECULATORS_REPO": str(repo),
        "HF_MODEL_CKPT": str(target),
        "DRAFT_MODEL": str(draft),
        "SPEC_METHOD": method,
        "DFLASH_BLOCK_SIZE": str(block_size),
        "NUM_SPEC_TOKENS": str(num_spec_tokens),
        "TP_SIZE": "2",
        "VLLM_PORT": "8123",
        "SERVE_READY_TIMEOUT": "3",
        "HF_HOME": str(hf_home),
        "EVAL_OUTPUT_ROOT": str(output_root),
        "EVAL_RUN_ID": "fixture-run",
        "EVAL_CONFIG_PATH": str(launcher_config),
        "CONTAINER_IMAGE": "/shared/containers/vllm-speculators@sha256:abc.sqsh",
        "SLURM_JOB_ID": "12345",
        "INVOCATION_LOG": str(invocation_log),
    }
    return env, output_root / "fixture-run", server_pid_file


def _acceptance_csv_text(positions: int, drafts: int) -> str:
    position_columns = ",".join(f"acceptance_at_pos_{index}" for index in range(positions))
    header = (
        "subset,num_drafts,num_draft_tokens,num_accepted_tokens,acceptance_length,"
        f"{position_columns}\n"
    )
    return header + "".join(
        f"{subset},{drafts},80,40,4.0," + ",".join("0.5" for _ in range(positions)) + "\n"
        for subset in _SUBSETS
    )


def _run(env: dict[str, str], tmp_path: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["bash", str(_WRAPPER)],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=15,
        check=False,
    )


@pytest.mark.parametrize(
    ("method", "block_size", "num_spec_tokens"),
    [("dflash", 8, 7), ("dflash", 16, 15), ("dspark", 8, 8), ("dspark", 16, 16)],
)
def test_valid_method_block_mapping_runs_all_subsets_and_writes_provenance(
    tmp_path: Path, method: str, block_size: int, num_spec_tokens: int
) -> None:
    """A valid matrix cell must yield a validated nine-subset result and exact manifest."""
    env, run_dir, server_pid_file = _make_harness(
        tmp_path,
        method=method,
        block_size=block_size,
        num_spec_tokens=num_spec_tokens,
    )

    result = _run(env, tmp_path)

    assert result.returncode == 0, result.stderr
    manifest = json.loads((run_dir / "manifest.json").read_text())
    assert manifest["status"] == "success"
    assert manifest["method"] == method
    assert manifest["block_size"] == block_size
    assert manifest["num_speculative_tokens"] == num_spec_tokens
    assert manifest["speculators_sha"] == _SPECULATORS_SHA
    assert manifest["slurm_job_id"] == "12345"
    assert manifest["container_image"].endswith("@sha256:abc.sqsh")
    assert len(manifest["config_sha256"]) == 3
    assert all(len(value) == 64 for value in manifest["config_sha256"].values())
    assert (run_dir / "acceptance.csv").stat().st_size > 0
    server_pid = int(server_pid_file.read_text())
    assert subprocess.run(["kill", "-0", str(server_pid)], check=False).returncode != 0
    invocations = (tmp_path / "invocations.log").read_text()
    assert "pip " not in invocations
    assert "git clone" not in invocations
    assert "/health" in invocations
    assert "/v1/models" in invocations
    assert "/metrics" in invocations


@pytest.mark.parametrize(
    ("method", "block_size", "num_spec_tokens"),
    [("eagle", 8, 8), ("dflash", 8, 8), ("dspark", 8, 7), ("dspark", 12, 12)],
)
def test_invalid_method_or_block_mapping_fails_before_server_start(
    tmp_path: Path, method: str, block_size: int, num_spec_tokens: int
) -> None:
    """Unsupported methods and uncoupled B/K values must never start vLLM."""
    env, run_dir, server_pid_file = _make_harness(
        tmp_path,
        method=method,
        block_size=block_size,
        num_spec_tokens=num_spec_tokens,
    )

    result = _run(env, tmp_path)

    assert result.returncode != 0
    assert not server_pid_file.exists()
    manifest = json.loads((run_dir / "manifest.json").read_text())
    assert manifest["status"] == "failed"


def test_speculators_sha_mismatch_fails_before_server_start(tmp_path: Path) -> None:
    """A stale shared checkout must be rejected before allocating the serving process."""
    env, run_dir, server_pid_file = _make_harness(tmp_path, repo_sha="deadbeef")

    result = _run(env, tmp_path)

    assert result.returncode != 0
    assert "Speculators SHA mismatch" in result.stderr
    assert not server_pid_file.exists()
    assert json.loads((run_dir / "manifest.json").read_text())["status"] == "failed"


def test_readiness_requires_speculative_metrics_before_evaluation(tmp_path: Path) -> None:
    """A generic metrics response must not be mistaken for a SpecDec-ready server."""
    env, run_dir, _ = _make_harness(tmp_path, spec_metrics_ready=False)

    result = _run(env, tmp_path)

    assert result.returncode == 3
    invocations = (tmp_path / "invocations.log").read_text()
    assert "acceptance-fixture.csv" not in invocations
    assert json.loads((run_dir / "manifest.json").read_text())["status"] == "failed"


def test_evaluator_failure_is_preserved_and_server_is_cleaned_up(tmp_path: Path) -> None:
    """An upstream evaluator error must fail the job without leaking the vLLM server."""
    env, run_dir, server_pid_file = _make_harness(tmp_path, evaluator_exit=23)

    result = _run(env, tmp_path)

    assert result.returncode == 23
    assert json.loads((run_dir / "manifest.json").read_text())["status"] == "failed"
    server_pid = int(server_pid_file.read_text())
    assert subprocess.run(["kill", "-0", str(server_pid)], check=False).returncode != 0


@pytest.mark.parametrize(("csv_positions", "drafts"), [(7, 10), (8, 0)])
def test_invalid_acceptance_csv_fails_the_job(
    tmp_path: Path, csv_positions: int, drafts: int
) -> None:
    """Missing position columns and zero-draft results must not be publishable."""
    env, run_dir, _ = _make_harness(
        tmp_path,
        csv_positions=csv_positions,
        drafts=drafts,
    )

    result = _run(env, tmp_path)

    assert result.returncode != 0
    assert json.loads((run_dir / "manifest.json").read_text())["status"] == "failed"


def test_speculators_eval_yaml_typed_resolves_without_internal_paths(tmp_path: Path) -> None:
    """The generic recipe must resolve through the real launcher and remain deployment-neutral."""
    resolved_path = tmp_path / "resolved.yaml"
    result = subprocess.run(
        [
            "uv",
            "run",
            "--frozen",
            "python",
            "launch.py",
            "--yaml",
            str(_RECIPE.relative_to(_LAUNCHER_DIR)),
            "--to-yaml",
            str(resolved_path),
        ],
        cwd=_LAUNCHER_DIR,
        capture_output=True,
        text=True,
        timeout=45,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    raw = _RECIPE.read_text()
    assert "/lustre/" not in raw
    assert ":latest" not in raw
    resolved = yaml.safe_load(resolved_path.read_text())
    global_vars = resolved["pipeline"]["global_vars"]
    assert global_vars["_target_"] == "modelopt_launcher.core.GlobalVariables"
    assert resolved["pipeline"]["task_0"]["script"] == "common/specdec/run_speculators_eval.sh"
