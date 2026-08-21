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

import csv
import hashlib
import json
import os
import signal
import subprocess
import sys
import tarfile
import time
from pathlib import Path
from typing import TYPE_CHECKING

import pytest
import yaml

if TYPE_CHECKING:
    from collections.abc import Callable

_LAUNCHER_DIR = Path(__file__).resolve().parents[1]
_WRAPPER = _LAUNCHER_DIR / "common/specdec/run_speculators_eval.sh"
_PAIR_RUNNER = _LAUNCHER_DIR / "common/specdec/run_speculators_eval_pair.sbatch"
_RECIPE = _LAUNCHER_DIR / "examples/Qwen/Qwen3-30B-A3B/speculators_eval.yaml"
_RUNTIME_STAGER = _LAUNCHER_DIR / "common/specdec/stage_speculators_eval_runtime.sh"
_HF_STAGER = _LAUNCHER_DIR / "common/specdec/stage_hf_model.sh"
_MODELOPT_RUNTIME_STAGER = _LAUNCHER_DIR / "common/specdec/stage_relocatable_runtime_archive.sh"
_MODELOPT_RUNTIME_PROBE = _LAUNCHER_DIR / "common/specdec/probe_relocatable_runtime.sh"
_SPECULATORS_SHA = "0b08a89a83b92007be63f128e01497455b0209df"
_MODELOPT_SHA = "a" * 40
_DATASET_REVISION = "b" * 40
_IMAGE_SHA256 = hashlib.sha256(b"staged image fixture").hexdigest()
_RUNTIME_SHA256 = "c" * 64
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


def _write_cluster_contract(tmp_path: Path, *, name: str = "lyris") -> tuple[Path, Path, Path]:
    durable = tmp_path / "durable"
    scratch = tmp_path / "scratch"
    durable.mkdir()
    scratch.mkdir()
    profile = tmp_path / f"{name}.yaml"
    profile.write_text(
        "\n".join(
            (
                f"name: {name}",
                "modelopt_commit: e3febcbe1319f018eea81fa4d42e2e36cb54494e",
                f"ssh_host: login-{name}",
                "account: coreai_dlalgo_llm",
                "partition: gb200",
                "fallback_partition: null",
                f"durable_root: {durable}",
                "scratch_candidates:",
                f"  - {scratch}",
                "training_nodes: 4",
                "training_segment: 4",
                "evaluation_nodes: 1",
                "evaluation_segment: 1",
                "gpus_per_node: 4",
                "explicit_gpu_flag: false",
                'walltime: "05:00:00"',
                "",
            )
        )
    )
    receipt = tmp_path / f"{name}-readiness.json"
    receipt.write_text(
        json.dumps(
            {
                "profile": name,
                "scratch_root": str(scratch),
                "account": "coreai_dlalgo_llm",
                "partition": "gb200",
                "pyxis_available": True,
                "gpu_count": 4,
                "architecture": "aarch64",
                "hostname": f"{name}0001",
                "timestamp": "2026-08-21T00:00:00Z",
            }
        )
    )
    return profile, receipt, durable


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
    max_concurrency: int = 32,
    max_requests: int = 200,
    eval_mode: str = "throughput",
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
    throughput_fixture = tmp_path / "throughput-fixture.json"
    throughput_fixture.write_text(
        json.dumps(
            {
                "benchmarks": [
                    {
                        "config": {
                            "strategy": {
                                "type_": "throughput",
                                "max_concurrency": max_concurrency,
                            }
                        },
                        "scheduler_state": {
                            "successful_requests": max_requests,
                            "errored_requests": 0,
                            "cancelled_requests": 0,
                        },
                        "metrics": {
                            "request_totals": {
                                "successful": max_requests,
                                "errored": 0,
                                "incomplete": 0,
                                "total": max_requests,
                            },
                            "requests_per_second": {"successful": {"median": 1.0}},
                            "request_latency": {"successful": {"median": 0.5}},
                            "inter_token_latency_ms": {"successful": {"median": 2.0}},
                            "time_to_first_token_ms": {"successful": {"median": 10.0}},
                            "output_tokens_per_second": {"successful": {"median": 100.0}},
                            "text": {"tokens": {"output": {"successful": {"total_sum": 2048.0}}}},
                        },
                    }
                ]
            }
        )
    )

    activate = runtime_bin / "activate"
    activate.parent.mkdir(parents=True)
    activate.write_text(f'export PATH="{runtime_bin}:$PATH"\n')
    (runtime / ".archive.sha256").write_text(f"{_RUNTIME_SHA256}\n")
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
  [[ "${{@: -1}}" == "{eval_mode}" ]] || exit 88
  [[ "$*" == *'--gen-kwargs {{"temperature":0,"top_p":1}}'* ]] || exit 94
  [[ "$*" == *'--max-concurrency {max_concurrency}'* ]] || exit 87
  [[ "$*" == *'--max-requests {max_requests}'* ]] || exit 86
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
  if [[ "{method}" != "baseline" ]]; then
    awk -F, -v subset="$subset" 'NR == 1 || $1 == subset' "{acceptance_fixture}" > "$output_dir/.acceptance-$subset.csv"
    if [[ -f "$output_dir/acceptance.csv" ]]; then
      tail -n +2 "$output_dir/.acceptance-$subset.csv" >> "$output_dir/acceptance.csv"
      rm "$output_dir/.acceptance-$subset.csv"
    else
      mv "$output_dir/.acceptance-$subset.csv" "$output_dir/acceptance.csv"
    fi
  fi
  mkdir -p "$output_dir/artifacts"
  cp "{throughput_fixture}" "$output_dir/artifacts/run_${{subset}}.json"
  if [[ "{eval_mode}" == "sweep" ]]; then
    if [[ ! -f "$output_dir/perf_results.csv" ]]; then
      printf '%s%s\n' \
        'subset,strategy,target_rate,rps_median,latency_median_s,itl_median_ms,' \
        'ttft_median_ms,output_tps_median,total_output_tokens' \
        > "$output_dir/perf_results.csv"
    fi
    printf '%s,constant,1,1.0,0.5,2.0,10.0,100.0,2048\n' "$subset" >> "$output_dir/perf_results.csv"
  fi
  touch "{evaluator_ready}"
  {"while true; do sleep 1; done" if evaluator_block else ":"}
  exit {1 if method == "baseline" and eval_mode == "sweep" else evaluator_exit}
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
        "MAX_CONCURRENCY": str(max_concurrency),
        "MAX_REQUESTS": str(max_requests),
        "EVAL_MODE": eval_mode,
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


def test_runtime_archive_stages_once_under_job_local_scratch(tmp_path: Path) -> None:
    """A pinned evaluator runtime must be relocated into job-local scratch without installs."""
    archived_runtime = tmp_path / "archived-runtime"
    _write_executable(archived_runtime / "bin/python3", "#!/bin/bash\nexit 0\n")
    _write_executable(archived_runtime / "bin/guidellm", "#!/bin/bash\nexit 0\n")
    (archived_runtime / "bin/activate").write_text(
        f'export VIRTUAL_ENV="{archived_runtime}"\nexport PATH="$VIRTUAL_ENV/bin:$PATH"\n'
    )
    archive = tmp_path / "runtime.tar.gz"
    with tarfile.open(archive, "w:gz") as file:
        for path in archived_runtime.rglob("*"):
            file.add(path, arcname=path.relative_to(archived_runtime))
    scratch_root = tmp_path / "raid/scratch"
    destination = scratch_root / "12345/speculators-runtime"
    env = {
        **os.environ,
        "SLURM_JOB_ID": "12345",
        "MARS_SCRATCH_ROOT": str(scratch_root),
        "SPECULATORS_RUNTIME_ARCHIVE": str(archive),
        "SPECULATORS_RUNTIME_ARCHIVE_SHA256": _sha256(archive),
        "SPECULATORS_RUNTIME": str(destination),
    }

    first = subprocess.run(
        ["bash", str(_RUNTIME_STAGER)], env=env, capture_output=True, text=True, check=False
    )
    second = subprocess.run(
        ["bash", str(_RUNTIME_STAGER)], env=env, capture_output=True, text=True, check=False
    )

    assert first.returncode == 0, first.stderr
    assert second.returncode == 0, second.stderr
    assert str(destination) in (destination / "bin/activate").read_text()
    assert first.stdout.strip() == str(destination)
    assert second.stdout.strip() == str(destination)


def test_runtime_archive_stager_rejects_unpinned_or_nonlocal_destination(tmp_path: Path) -> None:
    """The stager must fail before extraction when checksum or MARS placement is invalid."""
    archive = tmp_path / "runtime.tar.gz"
    archive.write_bytes(b"fixture")
    env = {
        **os.environ,
        "SLURM_JOB_ID": "12345",
        "MARS_SCRATCH_ROOT": str(tmp_path / "raid/scratch"),
        "SPECULATORS_RUNTIME_ARCHIVE": str(archive),
        "SPECULATORS_RUNTIME_ARCHIVE_SHA256": "0" * 64,
        "SPECULATORS_RUNTIME": str(tmp_path / "lustre/runtime"),
    }

    result = subprocess.run(
        ["bash", str(_RUNTIME_STAGER)], env=env, capture_output=True, text=True, check=False
    )

    assert result.returncode != 0
    assert "must be job-local" in result.stderr
    assert not Path(env["SPECULATORS_RUNTIME"]).exists()


def test_runtime_stager_accepts_matching_profile_readiness_contract(tmp_path: Path) -> None:
    """Evaluator runtime extraction uses the compute-verified profile scratch root."""
    profile, receipt, durable = _write_cluster_contract(tmp_path)
    archived_runtime = tmp_path / "archived-runtime"
    _write_executable(archived_runtime / "bin/python3", "#!/bin/bash\nexit 0\n")
    _write_executable(archived_runtime / "bin/guidellm", "#!/bin/bash\nexit 0\n")
    (archived_runtime / "bin/activate").write_text(
        f'export VIRTUAL_ENV="{archived_runtime}"\nexport PATH="$VIRTUAL_ENV/bin:$PATH"\n'
    )
    archive = durable / "runtime.tar.gz"
    with tarfile.open(archive, "w:gz") as file:
        for path in archived_runtime.rglob("*"):
            file.add(path, arcname=path.relative_to(archived_runtime))
    scratch = tmp_path / "scratch"
    destination = scratch / "12345/speculators-runtime"
    env = {
        **os.environ,
        "SLURM_JOB_ID": "12345",
        "MARS_SCRATCH_ROOT": str(scratch),
        "SPECULATORS_RUNTIME_ARCHIVE": str(archive),
        "SPECULATORS_RUNTIME_ARCHIVE_SHA256": _sha256(archive),
        "SPECULATORS_RUNTIME": str(destination),
        "CLUSTER_PROFILE": str(profile),
        "CLUSTER_READINESS_RECEIPT": str(receipt),
    }

    result = subprocess.run(
        ["bash", str(_RUNTIME_STAGER)], env=env, capture_output=True, text=True, check=False
    )

    assert result.returncode == 0, result.stderr
    assert destination.is_dir()


def test_runtime_stager_rejects_mismatched_readiness_before_extraction(tmp_path: Path) -> None:
    """A receipt for another profile cannot authorize node-local extraction."""
    profile, receipt, durable = _write_cluster_contract(tmp_path)
    payload = json.loads(receipt.read_text())
    payload["profile"] = "ptyche"
    receipt.write_text(json.dumps(payload))
    archive = durable / "runtime.tar.gz"
    archive.write_bytes(b"fixture")
    destination = tmp_path / "scratch/12345/speculators-runtime"
    env = {
        **os.environ,
        "SLURM_JOB_ID": "12345",
        "MARS_SCRATCH_ROOT": str(tmp_path / "scratch"),
        "SPECULATORS_RUNTIME_ARCHIVE": str(archive),
        "SPECULATORS_RUNTIME_ARCHIVE_SHA256": _sha256(archive),
        "SPECULATORS_RUNTIME": str(destination),
        "CLUSTER_PROFILE": str(profile),
        "CLUSTER_READINESS_RECEIPT": str(receipt),
    }

    result = subprocess.run(
        ["bash", str(_RUNTIME_STAGER)], env=env, capture_output=True, text=True, check=False
    )

    assert result.returncode != 0
    assert "readiness" in result.stderr.lower()
    assert not destination.exists()


def test_profile_aware_staging_entrypoints_share_the_readiness_contract() -> None:
    """All staging/probe entrypoints consume one profile and readiness receipt contract."""
    for script in (_HF_STAGER, _MODELOPT_RUNTIME_STAGER, _MODELOPT_RUNTIME_PROBE):
        text = script.read_text()
        assert "--cluster-profile" in text
        assert "--readiness-receipt" in text
        assert "load_cluster_profile" in text
        assert "validate_scratch_root" in text
        assert "scheduler_gpu_args" in text


def test_profile_aware_eval_rejects_output_outside_cluster_namespace(tmp_path: Path) -> None:
    """A profile-qualified evaluator cannot write into another cluster's result root."""
    profile, receipt, _ = _write_cluster_contract(tmp_path)
    env, run_dir, server_pid_file = _make_harness(tmp_path / "harness")
    env.update(
        {
            "CLUSTER_PROFILE": str(profile),
            "CLUSTER_READINESS_RECEIPT": str(receipt),
            "MARS_SCRATCH_ROOT": str(tmp_path / "scratch"),
        }
    )

    result = _run(env, tmp_path)

    assert result.returncode != 0
    assert "durable_root" in result.stderr
    assert not server_pid_file.exists()
    assert not run_dir.exists()


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
    fingerprint = json.loads((run_dir / "input-fingerprint.json").read_text())
    assert fingerprint["schema_version"] == 1
    assert len(fingerprint["sha256"]) == 64
    assert fingerprint["inputs"]["dataset"]["revision"] == _DATASET_REVISION
    assert fingerprint["inputs"]["image"]["sha256"] == _IMAGE_SHA256
    assert fingerprint["inputs"]["runtime"]["sha256"] == _RUNTIME_SHA256
    assert fingerprint["inputs"]["source"] == {
        "modelopt_sha": _MODELOPT_SHA,
        "speculators_sha": _SPECULATORS_SHA,
    }
    assert len(fingerprint["inputs"]["target_config_sha256"]) == 64
    assert len(fingerprint["inputs"]["draft_config_sha256"]) == 64
    assert set(manifest["dataset"]["files"]) == set(_SUBSETS)
    assert manifest["versions"] == {
        "guidellm": "0.4.0",
        "python": "Python 3.12.9",
        "vllm": "0.27.1",
    }
    assert manifest["evaluation"]["temperature"] == 0
    assert manifest["evaluation"]["top_p"] == 1
    assert manifest["evaluation"]["max_concurrency"] == 32
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
    performance_rows = list(csv.DictReader((run_dir / "perf_results.csv").open()))
    assert len(performance_rows) == len(_SUBSETS)
    assert {row["subset"] for row in performance_rows} == set(_SUBSETS)
    assert {row["strategy"] for row in performance_rows} == {"throughput"}
    assert {int(row["max_concurrency"]) for row in performance_rows} == {32}
    assert {int(row["completed_requests"]) for row in performance_rows} == {200}
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


@pytest.mark.parametrize("max_concurrency", [1, 8, 32, 128])
def test_fixed_throughput_uses_configured_concurrency_and_validates_performance(
    tmp_path: Path, max_concurrency: int
) -> None:
    """Each approved concurrency must produce task-wise sweep performance."""
    env, run_dir, _ = _make_harness(
        tmp_path,
        max_concurrency=max_concurrency,
        max_requests=512 if max_concurrency == 128 else 200,
        eval_mode="throughput",
    )

    result = _run(env, tmp_path)

    assert result.returncode == 0, result.stderr
    manifest = json.loads((run_dir / "manifest.json").read_text())
    assert manifest["evaluation"]["mode"] == "throughput"
    assert manifest["evaluation"]["max_concurrency"] == max_concurrency
    assert manifest["evaluation"]["max_requests"] == (512 if max_concurrency == 128 else 200)
    assert manifest["evaluator_args"][-1] == "throughput"
    assert (run_dir / "acceptance.csv").is_file()
    assert (run_dir / "perf_results.csv").is_file()


def test_baseline_throughput_omits_speculation_and_requires_only_performance(
    tmp_path: Path,
) -> None:
    """The AR baseline must use the same fixed throughput without acceptance."""
    env, run_dir, _ = _make_harness(
        tmp_path,
        method="baseline",
        block_size=0,
        num_spec_tokens=0,
        max_concurrency=8,
        eval_mode="throughput",
        spec_metrics_ready=False,
    )
    env.pop("DRAFT_MODEL")

    result = _run(env, tmp_path)

    assert result.returncode == 0, result.stderr
    manifest = json.loads((run_dir / "manifest.json").read_text())
    assert manifest["method"] == "baseline"
    assert manifest["draft_model"] is None
    assert manifest["config_sha256"].keys() == {"launcher", "target"}
    assert (run_dir / "perf_results.csv").is_file()
    assert not (run_dir / "acceptance.csv").exists()
    assert "--speculative-config" not in manifest["server_args"]


def test_paired_evaluator_uses_full_node_without_reintroducing_sweep() -> None:
    """Two TP2 cells share one four-GPU node and use the bounded throughput adapter."""
    runner = _PAIR_RUNNER.read_text()

    for required in (
        "#SBATCH --nodes=1",
        "#SBATCH --gpus-per-node=4",
        "#SBATCH --segment=1",
        'JOB_ROOT="${MARS_SCRATCH_ROOT%/}/${SLURM_JOB_ID}"',
        'readonly JOB_RUNTIME="${JOB_ROOT}/speculators-runtime"',
        "EVAL_MODE=throughput",
        "srun --exclusive --nodes=1 --ntasks=1 --gpus=2",
        'run_cell "${CELL_A}" 8000',
        'run_cell "${CELL_B}" 8010',
        'SPECULATORS_RUNTIME_ARCHIVE_SHA256="${RUNTIME_ARCHIVE_SHA256}"',
        "CLUSTER_PROFILE CLUSTER_READINESS_RECEIPT",
        'CLUSTER_PROFILE="${CLUSTER_PROFILE}"',
        'CLUSTER_READINESS_RECEIPT="${CLUSTER_READINESS_RECEIPT}"',
        "${CLUSTER_PROFILE}:${CLUSTER_PROFILE}",
        "${CLUSTER_READINESS_RECEIPT}:${CLUSTER_READINESS_RECEIPT}",
        'MARS_SCRATCH_ROOT="${MARS_SCRATCH_ROOT}"',
        'validate_label "${PAIR_LABEL}"',
        'validate_label "${label}"',
        "wait -n -p completed_pid",
        "terminate_children",
        "trap terminate_children EXIT",
        "trap 'terminate_children; exit 130' INT",
        "trap 'terminate_children; exit 143' TERM",
    ):
        assert required in runner
    assert "EVAL_MODE=sweep" not in runner
    assert "readonly SPECULATORS_RUNTIME=" not in runner
    assert "pip install" not in runner
    assert "git clone" not in runner


def test_resume_skips_atomically_validated_completed_subsets(tmp_path: Path) -> None:
    """A retry must reuse durable subset outputs without issuing duplicate requests."""
    env, run_dir, _ = _make_harness(tmp_path)
    evaluator_prefix = f"python3 {env['SPECULATORS_REPO']}/scripts/evaluate/evaluate.py "

    first = _run(env, tmp_path)
    first_evaluations = sum(
        line.startswith(evaluator_prefix)
        for line in (tmp_path / "invocations.log").read_text().splitlines()
    )
    second = _run(env, tmp_path)
    second_evaluations = sum(
        line.startswith(evaluator_prefix)
        for line in (tmp_path / "invocations.log").read_text().splitlines()
    )

    assert first.returncode == 0, first.stderr
    assert second.returncode == 0, second.stderr
    assert first_evaluations == len(_SUBSETS)
    assert second_evaluations == first_evaluations
    assert "Reusing validated completed subset" in second.stdout
    assert (run_dir / "perf_results.csv").is_file()


@pytest.mark.parametrize("mutated_input", ["target", "runtime"])
def test_resume_rejects_immutable_input_fingerprint_change(
    tmp_path: Path, mutated_input: str
) -> None:
    """A run ID cannot reuse subsets after any immutable staged input changes."""
    env, _, _ = _make_harness(tmp_path)
    evaluator_prefix = f"python3 {env['SPECULATORS_REPO']}/scripts/evaluate/evaluate.py "

    first = _run(env, tmp_path)
    first_evaluations = sum(
        line.startswith(evaluator_prefix)
        for line in (tmp_path / "invocations.log").read_text().splitlines()
    )
    if mutated_input == "target":
        (Path(env["HF_MODEL_CKPT"]) / "config.json").write_text(
            '{"model_type":"qwen3_moe","changed":true}\n'
        )
    else:
        (Path(env["SPECULATORS_RUNTIME"]) / ".archive.sha256").write_text(f"{'d' * 64}\n")

    second = _run(env, tmp_path)
    second_evaluations = sum(
        line.startswith(evaluator_prefix)
        for line in (tmp_path / "invocations.log").read_text().splitlines()
    )

    assert first.returncode == 0, first.stderr
    assert second.returncode != 0
    assert "immutable input fingerprint mismatch" in second.stderr
    assert second_evaluations == first_evaluations


def test_sweep_mode_is_rejected_before_server_start(tmp_path: Path) -> None:
    """The fixed-throughput wrapper must reject the incompatible upstream sweep path."""
    env, run_dir, server_pid_file = _make_harness(tmp_path, eval_mode="sweep")

    result = _run(env, tmp_path)

    assert result.returncode != 0
    assert "unsupported EVAL_MODE: sweep" in result.stderr
    assert not server_pid_file.exists()
    assert json.loads((run_dir / "manifest.json").read_text())["status"] == "failed"


@pytest.mark.parametrize("max_concurrency", [0, 2, 64])
def test_unapproved_concurrency_fails_before_server_start(
    tmp_path: Path, max_concurrency: int
) -> None:
    """Only the formal 1/8/32/128 concurrency cells may launch a server."""
    env, run_dir, server_pid_file = _make_harness(tmp_path, max_concurrency=max_concurrency)

    result = _run(env, tmp_path)

    assert result.returncode != 0
    assert "invalid MAX_CONCURRENCY" in result.stderr
    assert not server_pid_file.exists()
    assert json.loads((run_dir / "manifest.json").read_text())["status"] == "failed"


def test_concurrency_128_rejects_underfilled_request_budget(tmp_path: Path) -> None:
    """The saturation cell must not silently benchmark fewer than four waves."""
    env, run_dir, server_pid_file = _make_harness(
        tmp_path, max_concurrency=128, max_requests=200, eval_mode="sweep"
    )

    result = _run(env, tmp_path)

    assert result.returncode != 0
    assert "MAX_REQUESTS must be at least 512" in result.stderr
    assert not server_pid_file.exists()
    assert json.loads((run_dir / "manifest.json").read_text())["status"] == "failed"


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
    assert not any((run_dir / "subsets").iterdir())


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
