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

import hashlib
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import TYPE_CHECKING

import pytest
import yaml

if TYPE_CHECKING:
    from collections.abc import Callable

_LAUNCHER_DIR = Path(__file__).resolve().parents[1]
_WRAPPER = _LAUNCHER_DIR / "common/specdec/run_speculators_eval.sh"
_RECIPE = _LAUNCHER_DIR / "examples/Qwen/Qwen3-30B-A3B/speculators_eval.yaml"
_SPECULATORS_SHA = "0b08a89a83b92007be63f128e01497455b0209df"
_MODELOPT_SHA = "a" * 40
_DATASET_REVISION = "b" * 40
_IMAGE_SHA256 = hashlib.sha256(b"staged image fixture").hexdigest()
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


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


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
    preexisting_health: bool = False,
    server_exit: int | None = None,
    stale_after_spawn: bool = False,
    evaluator_block: bool = False,
) -> tuple[dict[str, str], Path, Path]:
    runtime = tmp_path / "runtime"
    runtime_bin = runtime / "bin"
    repo = tmp_path / "shared-speculators"
    evaluator = repo / "scripts/evaluate/evaluate.py"
    evaluator.parent.mkdir(parents=True)
    evaluator.write_text("# staged evaluator fixture\n")
    (repo / ".git").mkdir()
    modelopt_repo = tmp_path / "modelopt"
    (modelopt_repo / ".git").mkdir(parents=True)

    target = tmp_path / "target"
    draft = tmp_path / "draft"
    target.mkdir()
    draft.mkdir()
    (target / "config.json").write_text('{"model_type":"qwen3_moe"}\n')
    (draft / "config.json").write_text('{"architectures":["Qwen3DSparkModel"]}\n')
    launcher_config = tmp_path / "resolved.yaml"
    hf_home = tmp_path / "shared-cache"
    hf_home.mkdir()
    dataset_files = {}
    for subset in _SUBSETS:
        dataset_file = (
            hf_home
            / "hub"
            / "datasets--RedHatAI--speculator_benchmarks"
            / "snapshots"
            / _DATASET_REVISION
            / f"{subset}.jsonl"
        )
        dataset_file.parent.mkdir(parents=True, exist_ok=True)
        dataset_blob = dataset_file.parents[2] / "blobs" / subset
        dataset_blob.parent.mkdir(parents=True, exist_ok=True)
        dataset_blob.write_text(json.dumps({"prompt": f"fixture:{subset}"}) + "\n")
        dataset_file.symlink_to(dataset_blob)
        dataset_files[subset] = {"path": str(dataset_file), "sha256": _sha256(dataset_file)}
    dataset_manifest = tmp_path / "dataset-manifest.json"
    dataset_manifest.write_text(
        json.dumps(
            {
                "dataset_id": "RedHatAI/speculator_benchmarks",
                "revision": _DATASET_REVISION,
                "hf_home": str(hf_home),
                "files": dataset_files,
            }
        )
    )
    container_image = tmp_path / "vllm-speculators.sqsh"
    container_image.write_bytes(b"staged image fixture")
    launcher_config.write_text(
        f"pipeline:\n  task_0:\n    slurm_config:\n      container: {container_image}\n"
    )
    container_identity = tmp_path / "vllm-speculators.sqsh.identity.json"
    container_identity.write_text(
        json.dumps(
            {
                "path": str(container_image),
                "sha256": _IMAGE_SHA256,
                "size_bytes": container_image.stat().st_size,
            }
        )
    )
    output_root = tmp_path / "results"
    invocation_log = tmp_path / "invocations.log"
    server_pid_file = tmp_path / "server.pid"
    evaluator_ready = tmp_path / "evaluator.ready"
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
  if [[ "$*" == *"{modelopt_repo}"* ]]; then
    printf '%s\\n' "{_MODELOPT_SHA}"
  else
    printf '%s\\n' "{repo_sha}"
  fi
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
live=0
if [[ -f "{server_pid_file}" ]] && kill -0 "$(cat "{server_pid_file}")" 2>/dev/null; then
  live=1
fi
if [[ "{int(preexisting_health)}" == "1" ]]; then
  live=1
fi
if [[ "{int(stale_after_spawn)}" == "1" && -f "{server_pid_file}" ]]; then
  live=1
fi
[[ "$live" == "1" ]] || exit 7
case "$*" in
  *'/health'*) exit 0 ;;
  *'/v1/models'*) printf '{{"data":[{{"id":"{target}"}}]}}\\n' ;;
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
  {f"exit {server_exit}" if server_exit is not None else ":"}
  trap 'exit 0' TERM INT
  while true; do sleep 1; done
elif [[ "$1" == "{evaluator}" ]]; then
  [[ "${{HF_HUB_OFFLINE:-}}" == "1" ]] || exit 91
  [[ "${{HF_DATASETS_OFFLINE:-}}" == "1" ]] || exit 92
  [[ "${{PYTHONPATH%%:*}}" == "{repo / "src"}" ]] || exit 90
  [[ "$2" == "--target" ]] || exit 89
  [[ "${{@: -1}}" == "throughput" ]] || exit 88
  [[ "$*" == *'--gen-kwargs {{"temperature":0}}'* ]] || exit 94
  [[ "$*" == *'--max-concurrency 128'* ]] || exit 87
  [[ "$*" == *'--max-requests 200'* ]] || exit 86
  while [[ $# -gt 0 ]]; do
    case "$1" in
      --dataset) shift; dataset_path="$1" ;;
      --subsets) shift; subset="$1" ;;
      --output-dir) shift; output_dir="$1" ;;
    esac
    shift
  done
  [[ -f "$dataset_path" ]] || exit 93
  expected_suffix="/hub/datasets--RedHatAI--speculator_benchmarks/snapshots/{_DATASET_REVISION}/${{subset}}.jsonl"
  [[ "$dataset_path" == *"$expected_suffix" ]] || exit 85
  mkdir -p "$output_dir"
  awk -F, -v subset="$subset" 'NR == 1 || $1 == subset' "{acceptance_fixture}" > "$output_dir/.acceptance-$subset.csv"
  if [[ -f "$output_dir/acceptance.csv" ]]; then
    tail -n +2 "$output_dir/.acceptance-$subset.csv" >> "$output_dir/acceptance.csv"
    rm "$output_dir/.acceptance-$subset.csv"
  else
    mv "$output_dir/.acceptance-$subset.csv" "$output_dir/acceptance.csv"
  fi
  touch "{evaluator_ready}"
  {"while true; do sleep 1; done" if evaluator_block else ":"}
  exit {evaluator_exit}
elif [[ "$1" == "--version" ]]; then
  printf 'Python 3.12.9\\n'
elif [[ "$1" == "-c" && "$2" == *'importlib.metadata'* ]]; then
  if [[ "$2" == *'vllm'* ]]; then printf '0.27.1\\n'; else printf '0.4.0\\n'; fi
else
  exec "{sys.executable}" "$@"
fi
""",
    )

    env = {
        **os.environ,
        "SPECULATORS_RUNTIME": str(runtime),
        "SPECULATORS_REPO": str(repo),
        "MODELOPT_REPO": str(modelopt_repo),
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
        "DATASET_MANIFEST_PATH": str(dataset_manifest),
        "CONTAINER_IMAGE": str(container_image),
        "CONTAINER_IDENTITY_PATH": str(container_identity),
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
    rate = 0.4
    draft_tokens = drafts * positions
    accepted_tokens = int(drafts * positions * rate)
    acceptance_length = 1 + accepted_tokens / drafts if drafts else 1
    return header + "".join(
        f"{subset},{float(drafts)},{float(draft_tokens)},{float(accepted_tokens)},"
        f"{acceptance_length}," + ",".join(str(rate) for _ in range(positions)) + "\n"
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
    assert manifest["modelopt_sha"] == _MODELOPT_SHA
    assert manifest["modelopt_dirty"] is False
    assert manifest["slurm_job_id"] == "12345"
    assert manifest["container"]["sha256"] == _IMAGE_SHA256
    assert manifest["dataset"]["revision"] == _DATASET_REVISION
    assert set(manifest["dataset"]["files"]) == set(_SUBSETS)
    assert manifest["versions"] == {
        "guidellm": "0.4.0",
        "python": "Python 3.12.9",
        "vllm": "0.27.1",
    }
    assert manifest["evaluation"]["temperature"] == 0
    assert manifest["evaluation"]["max_concurrency"] == 128
    assert manifest["evaluation"]["max_requests"] == 200
    assert manifest["evaluation"]["tensor_parallel_size"] == 2
    assert manifest["server_args"][0:3] == [
        "-m",
        "vllm.entrypoints.cli.main",
        "serve",
    ]
    assert manifest["evaluator_args"][-1] == "throughput"
    assert manifest["evaluator_args"].count("--invocation") == len(_SUBSETS)
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
    assert "--dataset RedHatAI/speculator_benchmarks" not in invocations
    for subset in _SUBSETS:
        dataset_path = manifest["dataset"]["files"][subset]["path"]
        assert f"--dataset {dataset_path}" in invocations
        assert dataset_path in manifest["evaluator_args"]


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


@pytest.mark.parametrize("identity", ["dataset", "container"])
def test_staged_identity_mismatch_fails_before_server_start(tmp_path: Path, identity: str) -> None:
    """A lying snapshot revision or image path sidecar must fail before serving."""
    env, run_dir, server_pid_file = _make_harness(tmp_path)
    identity_path = Path(
        env["DATASET_MANIFEST_PATH"] if identity == "dataset" else env["CONTAINER_IDENTITY_PATH"]
    )
    payload = json.loads(identity_path.read_text())
    if identity == "dataset":
        payload["revision"] = "d" * 40
    else:
        payload["path"] = str(tmp_path / "different.sqsh")
    identity_path.write_text(json.dumps(payload))

    result = _run(env, tmp_path)

    assert result.returncode != 0
    assert not server_pid_file.exists()
    assert json.loads((run_dir / "manifest.json").read_text())["status"] == "failed"


def test_same_size_container_tamper_fails_before_server_start(tmp_path: Path) -> None:
    """The recorded digest must be checked against the actual staged image bytes."""
    env, run_dir, server_pid_file = _make_harness(tmp_path)
    image = Path(env["CONTAINER_IMAGE"])
    image.write_bytes(b"x" * image.stat().st_size)

    result = _run(env, tmp_path)

    assert result.returncode != 0
    assert "container image hash mismatch" in result.stderr
    assert not server_pid_file.exists()
    assert json.loads((run_dir / "manifest.json").read_text())["status"] == "failed"


def test_resolved_slurm_container_mismatch_fails_before_server_start(tmp_path: Path) -> None:
    """Provenance must describe the exact image selected in the resolved Slurm config."""
    env, run_dir, server_pid_file = _make_harness(tmp_path)
    Path(env["EVAL_CONFIG_PATH"]).write_text(
        f"pipeline:\n  task_0:\n    slurm_config:\n      container: {tmp_path / 'different.sqsh'}\n"
    )

    result = _run(env, tmp_path)

    assert result.returncode != 0
    assert "resolved Slurm container does not match" in result.stderr
    assert not server_pid_file.exists()
    assert json.loads((run_dir / "manifest.json").read_text())["status"] == "failed"


def test_preexisting_health_endpoint_is_rejected_before_server_spawn(tmp_path: Path) -> None:
    """An occupied serving port must fail before the owned vLLM process is launched."""
    env, run_dir, server_pid_file = _make_harness(tmp_path, preexisting_health=True)

    result = _run(env, tmp_path)

    assert result.returncode != 0
    assert "already serving health" in result.stderr
    assert not server_pid_file.exists()
    assert json.loads((run_dir / "manifest.json").read_text())["status"] == "failed"


def test_owned_server_exit_zero_cannot_attach_to_stale_server(tmp_path: Path) -> None:
    """Loss of the owned PID must fail even if another server then answers every probe."""
    env, run_dir, _ = _make_harness(tmp_path, server_exit=0, stale_after_spawn=True)

    result = _run(env, tmp_path)

    assert result.returncode != 0
    assert "owned vLLM server exited" in result.stderr
    assert json.loads((run_dir / "manifest.json").read_text())["status"] == "failed"


def test_term_signal_writes_failed_manifest_and_exits_143(tmp_path: Path) -> None:
    """Slurm TERM cancellation must never be reported as a successful evaluation."""
    env, run_dir, _ = _make_harness(tmp_path, evaluator_block=True)
    process = subprocess.Popen(
        ["bash", str(_WRAPPER)],
        cwd=tmp_path,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        start_new_session=True,
    )
    evaluator_ready = tmp_path / "evaluator.ready"
    deadline = time.monotonic() + 10
    while not evaluator_ready.exists() and process.poll() is None and time.monotonic() < deadline:
        time.sleep(0.05)
    assert evaluator_ready.exists(), process.communicate(timeout=2)

    os.killpg(process.pid, signal.SIGTERM)
    _, stderr = process.communicate(timeout=10)

    assert process.returncode == 143, stderr
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


def test_acceptance_csv_accepts_upstream_integral_float_counters(tmp_path: Path) -> None:
    """Prometheus counters serialized by upstream as 10.0 remain exact integers."""
    csv_path = tmp_path / "acceptance.csv"
    csv_path.write_text(_acceptance_csv_text(8, 10))

    result = subprocess.run(
        [
            sys.executable,
            str(_LAUNCHER_DIR / "common/specdec/speculators_eval_artifacts.py"),
            "validate",
            "--csv",
            str(csv_path),
            "--num-speculative-tokens",
            "8",
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize(
    "mutation",
    [
        lambda text: text.replace(",0.4,", ",nan,", 1),
        lambda text: text.replace(",0.4,", ",not-a-number,", 1),
        lambda text: text.replace(",0.4,", ",1.1,", 1),
        lambda text: text.replace(",10.0,80.0,", ",10.5,80.0,", 1),
        lambda text: text.replace(",10.0,80.0,", ",nan,80.0,", 1),
        lambda text: text.replace(",80.0,32.0,", ",79.0,32.0,", 1),
        lambda text: text.replace(",80.0,32.0,", ",80.0,31.0,", 1),
        lambda text: text.replace(",32.0,4.2,", ",32.0,4.3,", 1),
        lambda text: text.replace(",0.4,0.4,", ",0.2,0.6,", 1),
        lambda text: text.replace(",80.0,32.0,", ",80.0,-1.0,", 1),
    ],
)
def test_acceptance_csv_rejects_invalid_numeric_contract(
    tmp_path: Path, mutation: Callable[[str], str]
) -> None:
    """Corrupt counters, rates, monotonicity, and acceptance formulas must be rejected."""
    csv_path = tmp_path / "acceptance.csv"
    csv_path.write_text(mutation(_acceptance_csv_text(8, 10)))

    result = subprocess.run(
        [
            sys.executable,
            str(_LAUNCHER_DIR / "common/specdec/speculators_eval_artifacts.py"),
            "validate",
            "--csv",
            str(csv_path),
            "--num-speculative-tokens",
            "8",
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode != 0


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
    assert global_vars["eval_config"] == "/path/to/resolved-launch.yaml"
    assert resolved["pipeline"]["task_0"]["script"] == "common/specdec/run_speculators_eval.sh"
