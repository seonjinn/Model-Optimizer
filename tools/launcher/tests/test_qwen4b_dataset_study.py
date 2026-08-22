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

"""Contracts for the Qwen3-4B speculative-decoding dataset study."""

from __future__ import annotations

import json
import os
import subprocess
from dataclasses import replace
from pathlib import Path

import pytest
from common.specdec.build_qwen4b_synthesis_manifest import (
    Qwen4BSynthesisManifest,
    SynthesisInputs,
    SynthesisSettings,
    load_synthesis_manifest,
    tokenizer_snapshot_sha256,
    verify_data_manifest,
    verify_synthesis_completion,
    write_synthesis_manifest,
)
from common.specdec.qwen4b_study_manifest import (
    Qwen4BStudyExperiment,
    Qwen4BStudyInputs,
    Qwen4BStudySchedule,
    _directory_sha256,
    load_study_manifest,
    readable_study_job_name,
    validate_canary_receipt,
    validate_milestone_receipt,
    validate_pmon_logs,
    validate_training_target,
    write_study_manifest,
)


def _manifest() -> Qwen4BSynthesisManifest:
    return Qwen4BSynthesisManifest(
        arm="C",
        thinking_mode="on",
        response_source="target-synth",
        inputs=SynthesisInputs(
            source_path="/home/sna/ModelOpt",
            source_sha="a" * 40,
            image_path="/lustre/images/vllm.sqsh",
            image_sha256="b" * 64,
            runtime_archive_path="/lustre/runtime/modelopt.tar.zst",
            runtime_archive_sha256="c" * 64,
            target_path="/lustre/models/Qwen3-4B",
            target_revision="1cfa9a7208912126459214e8b04321603b3df60c",
            tokenizer_sha256="d" * 64,
            prompt_manifest_path="/lustre/study/prompts/C.json",
            prompt_manifest_sha256="e" * 64,
            trace_manifest_path="/lustre/study/traces/C.json",
            trace_manifest_sha256="f" * 64,
            output_root="/lustre/study/synthesis/C-thinking-on",
        ),
        settings=SynthesisSettings(
            account="nemotron_sw_post",
            partition="batch",
            replicas=4,
            tensor_parallel_size=1,
            num_shards=256,
            output_token_budget=256_000_000,
            temperature=0.0,
            max_tokens=4096,
            max_total_length=32768,
        ),
    )


def test_readiness_receipt_requires_canonical_pyxis_available_field() -> None:
    """Legacy pyxis aliases cannot satisfy the immutable readiness gate."""
    from common.specdec.qwen4b_study_manifest import validate_readiness_receipt

    expected = {
        "profile": "oci-hsg",
        "account": "nemotron_sw_post",
        "partition": "batch",
        "scratch_root": "/raid/scratch",
        "pyxis_available": True,
        "architecture": "aarch64",
        "gpu_count": 4,
    }

    validate_readiness_receipt(
        expected,
        profile="oci-hsg",
        account="nemotron_sw_post",
        partition="batch",
        scratch_root=Path("/raid/scratch"),
    )
    with pytest.raises(ValueError, match="readiness receipt"):
        validate_readiness_receipt(
            expected | {"pyxis_available": False, "pyxis": True},
            profile="oci-hsg",
            account="nemotron_sw_post",
            partition="batch",
            scratch_root=Path("/raid/scratch"),
        )


def test_synthesis_identity_separates_thinking_modes_and_response_sources() -> None:
    """Mode and target-native versus replay provenance cannot share output identity."""
    manifest = _manifest()

    assert manifest.experiment_id != replace(manifest, thinking_mode="off").experiment_id
    assert manifest.experiment_id != replace(manifest, response_source="trace-replay").experiment_id


def test_synthesis_manifest_round_trip_is_canonical_and_immutable(tmp_path: Path) -> None:
    """A written synthesis manifest round-trips without losing pinned inputs."""
    path = tmp_path / "manifest.json"
    manifest = _manifest()

    write_synthesis_manifest(path, [manifest])

    assert load_synthesis_manifest(path) == [manifest]
    payload = json.loads(path.read_text())
    assert payload[0]["experiment_id"] == manifest.experiment_id
    assert payload[0]["settings"]["output_token_budget"] == 256_000_000


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("account", "nemotron_n3_post", "nemotron_n3_post"),
        ("replicas", 2, "four replicas"),
        ("tensor_parallel_size", 2, "TP1"),
        ("num_shards", 255, "divisible"),
    ],
)
def test_synthesis_settings_reject_idle_or_unapproved_allocations(
    field: str, value: str | int, message: str
) -> None:
    """Study synthesis must use all four GPUs and stay off the protected account."""
    with pytest.raises(ValueError, match=message):
        replace(_manifest().settings, **{field: value})


def test_trace_replay_requires_a_trace_manifest() -> None:
    """Recorded tool trajectories are explicit inputs, never implicit target synthesis."""
    inputs = replace(
        _manifest().inputs,
        trace_manifest_path=None,
        trace_manifest_sha256=None,
    )

    with pytest.raises(ValueError, match="trace manifest"):
        replace(_manifest(), response_source="trace-replay", inputs=inputs)


def test_prompt_manifest_selects_and_verifies_exact_shards(tmp_path: Path) -> None:
    """Prompt manifests select only their content-addressed shard set."""
    shard = tmp_path / "selected.jsonl"
    shard.write_text('{"messages": []}\n')
    (tmp_path / "unselected.jsonl").write_text('{"messages": [{"role": "user"}]}\n')
    digest = __import__("hashlib").sha256(shard.read_bytes()).hexdigest()
    manifest = tmp_path / "MANIFEST.json"
    manifest.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "format": "json",
                "row_count": 1,
                "file_count": 1,
                "files": [{"path": shard.name, "bytes": shard.stat().st_size, "sha256": digest}],
            }
        )
        + "\n"
    )

    selected = verify_data_manifest(manifest)

    assert selected == (shard,)


def test_prompt_manifest_rejects_changed_shard(tmp_path: Path) -> None:
    """A mutated prompt shard invalidates the immutable manifest."""
    shard = tmp_path / "selected.jsonl"
    shard.write_text("original\n")
    manifest = tmp_path / "MANIFEST.json"
    manifest.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "format": "json",
                "row_count": 1,
                "file_count": 1,
                "files": [
                    {
                        "path": shard.name,
                        "bytes": shard.stat().st_size,
                        "sha256": __import__("hashlib").sha256(shard.read_bytes()).hexdigest(),
                    }
                ],
            }
        )
        + "\n"
    )
    shard.write_text("changed\n")

    with pytest.raises(ValueError, match="identity mismatch"):
        verify_data_manifest(manifest)


def test_completed_synthesis_is_rehashed_before_reuse(tmp_path: Path) -> None:
    """Completion identity alone cannot authorize reuse of mutated output."""
    shards = tmp_path / "shards"
    shards.mkdir()
    shard = shards / "shard.jsonl"
    shard.write_text("ok\n")
    completion = {
        "schema_version": 1,
        "experiment_id": "experiment",
        "actual_assistant_tokens": 1,
        "requested_output_token_budget": 1,
        "tokenizer_sha256": "tokenizer",
        "file_count": 1,
        "files": [
            {
                "path": "shards/shard.jsonl",
                "bytes": shard.stat().st_size,
                "sha256": __import__("hashlib").sha256(shard.read_bytes()).hexdigest(),
            }
        ],
    }
    (tmp_path / "completion.json").write_text(json.dumps(completion) + "\n")
    verify_synthesis_completion(
        tmp_path,
        "experiment",
        expected_output_token_budget=1,
        expected_tokenizer_sha256="tokenizer",
        expected_file_count=1,
    )
    shard.write_text("corrupt\n")

    with pytest.raises(ValueError, match="synthesis file identity mismatch"):
        verify_synthesis_completion(
            tmp_path,
            "experiment",
            expected_output_token_budget=1,
            expected_tokenizer_sha256="tokenizer",
            expected_file_count=1,
        )


def test_synthesis_completion_requires_at_least_the_requested_exact_count(
    tmp_path: Path,
) -> None:
    """Exact token counts may exceed but never fall below the requested supply."""
    shards = tmp_path / "shards"
    shards.mkdir()
    shard = shards / "shard.jsonl"
    shard.write_text("ok\n")
    completion = {
        "schema_version": 1,
        "experiment_id": "experiment",
        "actual_assistant_tokens": 2,
        "requested_output_token_budget": 1,
        "tokenizer_sha256": "tokenizer",
        "file_count": 1,
        "files": [
            {
                "path": "shards/shard.jsonl",
                "bytes": shard.stat().st_size,
                "sha256": __import__("hashlib").sha256(shard.read_bytes()).hexdigest(),
            }
        ],
    }
    (tmp_path / "completion.json").write_text(json.dumps(completion) + "\n")
    verify_synthesis_completion(
        tmp_path,
        "experiment",
        expected_output_token_budget=1,
        expected_tokenizer_sha256="tokenizer",
        expected_file_count=1,
    )

    completion["actual_assistant_tokens"] = 0
    (tmp_path / "completion.json").write_text(json.dumps(completion) + "\n")
    with pytest.raises(ValueError, match="assistant-token budget mismatch"):
        verify_synthesis_completion(
            tmp_path,
            "experiment",
            expected_output_token_budget=1,
            expected_tokenizer_sha256="tokenizer",
            expected_file_count=1,
        )


def test_synthesis_completion_cannot_self_report_a_smaller_immutable_contract(
    tmp_path: Path,
) -> None:
    """Reuse must compare completion metadata with the immutable synthesis manifest."""
    shards = tmp_path / "shards"
    shards.mkdir()
    shard = shards / "shard.jsonl"
    shard.write_text("ok\n")
    completion = {
        "schema_version": 1,
        "experiment_id": "experiment",
        "actual_assistant_tokens": 1,
        "requested_output_token_budget": 1,
        "tokenizer_sha256": "wrong-tokenizer",
        "file_count": 1,
        "files": [
            {
                "path": "shards/shard.jsonl",
                "bytes": shard.stat().st_size,
                "sha256": __import__("hashlib").sha256(shard.read_bytes()).hexdigest(),
            }
        ],
    }
    (tmp_path / "completion.json").write_text(json.dumps(completion) + "\n")

    with pytest.raises(ValueError, match="immutable synthesis contract"):
        verify_synthesis_completion(
            tmp_path,
            "experiment",
            expected_output_token_budget=256_000_000,
            expected_tokenizer_sha256="expected-tokenizer",
            expected_file_count=256,
        )


def test_tokenizer_digest_excludes_model_weights_and_is_content_addressed(tmp_path: Path) -> None:
    """Tokenizer identity is independent of large model-weight files."""
    (tmp_path / "tokenizer.json").write_text("tokenizer-v1\n")
    (tmp_path / "tokenizer_config.json").write_text("{}\n")
    weights = tmp_path / "model.safetensors"
    weights.write_text("weights-v1\n")

    original = tokenizer_snapshot_sha256(tmp_path)
    weights.write_text("weights-v2\n")
    assert tokenizer_snapshot_sha256(tmp_path) == original
    (tmp_path / "tokenizer.json").write_text("tokenizer-v2\n")
    assert tokenizer_snapshot_sha256(tmp_path) != original


def test_runner_owns_all_gpus_and_pins_each_shard_to_one_replica() -> None:
    """The synthesis runner launches four supervised TP1 cells with disjoint shards."""
    runner = (
        Path(__file__).resolve().parents[1] / "common/specdec/run_qwen4b_synthesis.sbatch"
    ).read_text()

    assert "for ((replica = 0; replica < REPLICAS; replica++))" in runner
    assert "--gpus=1" in runner
    assert "--exclusive" in runner
    assert '--shard-id-begin="$replica"' in runner
    assert '--shard-id-step="$REPLICAS"' in runner
    assert '--tensor-parallel-size "$SERVE_TP"' in runner
    assert "wait -n" in runner
    assert "kill_children" in runner
    assert "verify_data_manifest" in runner
    assert "tokenizer_snapshot_sha256" in runner
    assert "verify_synthesis_completion" in runner
    assert '--data "$job_root/input/prompts/$PROMPT_MANIFEST_NAME"' in runner
    assert "--reject-tool-trajectories" in runner
    assert "--record-assistant-tokens" in runner
    assert "--strict-num-shards" in runner
    assert 'SYNTHESIS_WORK_ROOT="${OUTPUT_ROOT}.work-${EXPERIMENT_ID}"' in runner
    assert 'replica_output="$SYNTHESIS_WORK_ROOT/replicas/$replica"' in runner


def _study_experiment() -> Qwen4BStudyExperiment:
    return Qwen4BStudyExperiment(
        arm="C",
        thinking_mode="on",
        method="dflash",
        block_size=8,
        inputs=Qwen4BStudyInputs(
            source_path="/home/sna/ModelOpt",
            source_sha="a" * 40,
            image_path="/lustre/images/vllm.sqsh",
            image_sha256="b" * 64,
            runtime_archive_path="/lustre/runtime/modelopt.tar.zst",
            runtime_archive_sha256="c" * 64,
            target_path="/lustre/models/Qwen3-4B",
            target_revision="1cfa9a7208912126459214e8b04321603b3df60c",
            tokenizer_sha256="9" * 64,
            corpus_path="/lustre/study/corpora/C-thinking-on",
            corpus_manifest_path="/lustre/study/corpora/C-thinking-on/MANIFEST.json",
            corpus_manifest_sha256="d" * 64,
            output_root="/lustre/study/training/C-thinking-on-dflash-b8",
        ),
        schedule=Qwen4BStudySchedule(
            account="nemotron_sw_post",
            partition="batch",
            assistant_token_budget=256_000_000,
            corpus_rows=100_000,
            max_steps=782,
            canary_steps=20,
            assistant_token_milestones=(64_000_000, 128_000_000, 256_000_000),
            assistant_token_milestone_step_ceilings=(390, 650, 782),
        ),
    )


def test_study_topology_uses_every_gpu_at_gbs128() -> None:
    """One serve node and one trainer node occupy eight GPUs without changing GBS."""
    experiment = _study_experiment()

    assert experiment.nodes == 2
    assert experiment.serve_nodes == 1
    assert experiment.trainer_nodes == 1
    assert experiment.serve_replicas_per_node == 4
    assert experiment.trainer_world_size == 4
    assert experiment.global_batch_size == 128


def test_token_milestones_have_strict_increasing_step_safety_ceilings() -> None:
    """Every exact token milestone has one increasing optimizer-step ceiling."""
    experiment = _study_experiment()

    with pytest.raises(ValueError, match="step ceilings"):
        replace(
            experiment.schedule,
            assistant_token_milestone_step_ceilings=(390, 390, 782),
        )


def test_training_target_binds_token_milestone_to_step_safety_ceiling() -> None:
    """A production launch cannot mix a token target with another step ceiling."""
    experiment = _study_experiment()

    validate_training_target(experiment, experiment.schedule.canary_steps, None)
    validate_training_target(experiment, 390, 64_000_000)

    with pytest.raises(ValueError, match="milestone/step ceiling mismatch"):
        validate_training_target(experiment, 650, 64_000_000)
    with pytest.raises(ValueError, match="requires an assistant-token target"):
        validate_training_target(experiment, 390, None)


def test_milestone_receipt_is_identity_and_token_bound(tmp_path: Path) -> None:
    """Durable milestone receipts bind exact exposure and experiment identity."""
    experiment = _study_experiment()
    output_root = tmp_path / "output"
    object.__setattr__(experiment.inputs, "output_root", str(output_root))
    checkpoint = output_root / "checkpoint-200"
    export = output_root / "exported-assistant-tokens-64000000"
    checkpoint.mkdir(parents=True)
    export.mkdir()
    (checkpoint / "trainer_state.json").write_text('{"global_step": 200}\n')
    (export / "model.safetensors").write_text("weights\n")
    receipt = tmp_path / "milestone.json"
    receipt.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "experiment_id": experiment.experiment_id,
                "committed_assistant_tokens": 64_000_000,
                "global_step": 200,
                "checkpoint_path": str(checkpoint),
                "checkpoint_sha256": _directory_sha256(checkpoint),
                "export_path": str(export),
                "export_sha256": _directory_sha256(export),
                "checkpoint_verified": True,
                "export_verified": True,
                "status": "passed",
            }
        )
        + "\n"
    )

    validate_milestone_receipt(receipt, experiment, 64_000_000)

    (export / "model.safetensors").write_text("changed\n")
    with pytest.raises(ValueError, match="artifact"):
        validate_milestone_receipt(receipt, experiment, 64_000_000)

    assert experiment.capture_ids == (2, 18, 33, 36)


def test_study_paths_must_be_strict_descendants_of_cluster_roots() -> None:
    """Broad shared roots cannot be selected as mutable study destinations."""
    with pytest.raises(ValueError, match="strict descendant"):
        replace(_study_experiment().inputs, output_root="/lustre")


def test_study_schedule_requires_ordered_token_milestones_and_sw_account() -> None:
    """The screen stays off N3 and records comparable assistant-token boundaries."""
    with pytest.raises(ValueError, match="nemotron_sw_post"):
        replace(_study_experiment().schedule, account="nemotron_n3_post")
    with pytest.raises(ValueError, match="final assistant-token milestone"):
        replace(
            _study_experiment().schedule,
            assistant_token_milestones=(64_000_000, 128_000_000),
        )


def test_study_identity_and_job_name_expose_arm_mode_method_and_topology() -> None:
    """Scheduler and durable identities remain readable and collision-resistant."""
    experiment = _study_experiment()

    assert readable_study_job_name(experiment) == "q4b-thon-c-df-b8-n2-t256m"
    assert experiment.experiment_id != replace(experiment, arm="D").experiment_id
    assert experiment.experiment_id != replace(experiment, thinking_mode="off").experiment_id


def test_study_manifest_round_trip_is_canonical_and_identity_checked(tmp_path: Path) -> None:
    """Training jobs consume a canonical manifest whose identity cannot be edited."""
    path = tmp_path / "study.json"
    experiment = _study_experiment()

    write_study_manifest(path, [experiment])

    assert load_study_manifest(path) == [experiment]
    payload = json.loads(path.read_text())
    payload[0]["arm"] = "D"
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="identity mismatch"):
        load_study_manifest(path)


def test_production_steps_require_matching_successful_canary_receipt(tmp_path: Path) -> None:
    """Only an identity-bound, fully passed canary authorizes production."""
    experiment = _study_experiment()
    validate_canary_receipt(None, experiment, experiment.schedule.canary_steps)

    with pytest.raises(ValueError, match="canary receipt is required"):
        validate_canary_receipt(None, experiment, experiment.schedule.max_steps)

    receipt = tmp_path / "canary.json"
    output_root = tmp_path / "output"
    object.__setattr__(experiment.inputs, "output_root", str(output_root))
    checkpoint = output_root / "canary/checkpoint-20"
    export = output_root / "canary/exported-checkpoint-20"
    ownership = output_root / "canary/control/gpu-ownership.json"
    logs = output_root / "canary/logs"
    checkpoint.mkdir(parents=True)
    export.mkdir()
    ownership.parent.mkdir(parents=True)
    logs.mkdir()
    (checkpoint / "trainer_state.json").write_text('{"global_step": 20}\n')
    (export / "model.safetensors").write_text("weights\n")
    pmon_hashes = {}
    for node in range(2):
        pmon = logs / f"nvidia-smi-pmon-node-{node}-42.log"
        pmon.write_text("activity\n")
        pmon_hashes[pmon.name] = __import__("hashlib").sha256(pmon.read_bytes()).hexdigest()
    ownership.write_text(
        json.dumps({"nodes": 2, "nonzero_sm_activity": True, "pmon_sha256": pmon_hashes}) + "\n"
    )
    receipt.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "experiment_id": experiment.experiment_id,
                "completed_steps": experiment.schedule.canary_steps,
                "checkpoint_verified": True,
                "export_verified": True,
                "all_gpus_active": True,
                "checkpoint_path": str(checkpoint),
                "checkpoint_sha256": _directory_sha256(checkpoint),
                "export_path": str(export),
                "export_sha256": _directory_sha256(export),
                "gpu_ownership_path": str(ownership),
                "gpu_ownership_sha256": __import__("hashlib")
                .sha256(ownership.read_bytes())
                .hexdigest(),
                "status": "passed",
            }
        )
        + "\n"
    )
    validate_canary_receipt(receipt, experiment, experiment.schedule.max_steps)
    validate_canary_receipt(receipt, experiment, 390)

    payload = json.loads(receipt.read_text())
    payload["all_gpus_active"] = False
    receipt.write_text(json.dumps(payload) + "\n")
    with pytest.raises(ValueError, match="canary receipt failed"):
        validate_canary_receipt(receipt, experiment, experiment.schedule.max_steps)


def test_pmon_gate_selects_current_job_and_requires_nonzero_sm_activity(tmp_path: Path) -> None:
    """The canary uses current-job logs and requires real activity on every GPU."""
    header = "#Date Time gpu pid type sm mem enc dec jpg ofa command\n"
    active = "".join(
        f"2026/08/22 12:00:00 {gpu} {100 + gpu} C 1 2 - - - - python\n" for gpu in range(4)
    )
    for node in range(2):
        (tmp_path / f"nvidia-smi-pmon-node-{node}-42.log").write_text(header + active)
    (tmp_path / "nvidia-smi-pmon-node-0-41.log").write_text("stale\n")

    evidence = validate_pmon_logs(tmp_path, "42")

    assert set(evidence) == {
        "nvidia-smi-pmon-node-0-42.log",
        "nvidia-smi-pmon-node-1-42.log",
    }
    assert all(value == [0, 1, 2, 3] for value in evidence.values())

    inactive = "".join(
        f"2026/08/22 12:00:00 {gpu} {100 + gpu} C {0 if gpu == 3 else 1} 2 - - - - python\n"
        for gpu in range(4)
    )
    (tmp_path / "nvidia-smi-pmon-node-1-42.log").write_text(header + inactive)
    with pytest.raises(ValueError, match="nonzero SM activity"):
        validate_pmon_logs(tmp_path, "42")


def test_training_runner_encodes_full_gpu_requeue_and_assistant_loss_contract() -> None:
    """The Qwen4B runner must occupy both nodes and preserve resumable state."""
    runner = (
        Path(__file__).resolve().parents[1] / "common/specdec/run_qwen4b_study_training.sbatch"
    ).read_text()

    assert "GLOBAL_BATCH_SIZE=128" in runner
    assert "SERVE_REPLICAS_PER_NODE=4" in runner
    assert "EXPECTED_RNG_STATES=4" in runner
    assert "training.answer_only_loss=true" in runner
    assert "training.save_total_limit=2" in runner
    assert "drafter_requeue_lifecycle.sh" in runner
    assert 'training.save_steps="${SAVE_STEPS}"' in runner
    assert "nvidia-smi pmon" in runner
    assert "gpu-ownership.json" in runner
    assert "observed_gpu_indices" in runner
    ownership_receipt = runner.split("$SLURM_JOB_ID\" <<'PY'", 1)[1].split(
        "drafter_mark_training_complete", 1
    )[0]
    assert "import hashlib" in ownership_receipt
    assert "assistant-token-budget.json" in runner
    assert '"tokenizer_sha256": sys.argv[8]' in runner
    assert "verify_data_manifest" in runner
    assert "validate_pmon_logs" in runner
    assert "canary-${EXPERIMENT_ID}.json" in runner
    assert "MODELOPT_ASSISTANT_TOKEN_TARGET" in runner
    assert "assistant-token-milestone-${ASSISTANT_TOKEN_TARGET}.json" in runner
    assert 'OUTPUT_ROOT="$STUDY_OUTPUT_ROOT/canary"' in runner
    assert 'DEFAULT_WANDB_RUN_ID="q4b-${EXPERIMENT_ID}-canary"' in runner
    assert 'DEFAULT_WANDB_RUN_ID="q4b-${EXPERIMENT_ID}-production"' in runner
    assert 'export WANDB_RUN_ID="$DEFAULT_WANDB_RUN_ID"' in runner
    assert "${WANDB_RUN_ID:-$DEFAULT_WANDB_RUN_ID}" not in runner
    assert "MANIFEST_SHA256" in runner.split("for name in", 1)[1].split("; do", 1)[0]
    assert '"$MANIFEST_PATH:$MANIFEST_SHA256:study manifest"' in runner
    assert 'sha256sum "$path"' in runner


def test_submitter_runs_test_only_before_real_submission_and_writes_receipt(
    tmp_path: Path,
) -> None:
    """A study job is never submitted before the scheduler validates its allocation."""
    manifest_path = tmp_path / "study.json"
    write_study_manifest(manifest_path, [_study_experiment()])
    profile = tmp_path / "profile.yaml"
    profile.write_text(
        "\n".join(
            [
                "name: oci-hsg",
                f"modelopt_commit: {'a' * 40}",
                "ssh_host: host02",
                "account: nemotron_sw_post",
                "partition: batch",
                "fallback_partition: null",
                "durable_root: /lustre/study",
                "scratch_candidates: [/raid/scratch]",
                "training_nodes: 4",
                "training_segment: 4",
                "evaluation_nodes: 1",
                "evaluation_segment: 1",
                "gpus_per_node: 4",
                "explicit_gpu_flag: true",
                'walltime: "03:55:00"',
            ]
        )
        + "\n"
    )
    receipt = tmp_path / "readiness.json"
    receipt.write_text(
        json.dumps(
            {
                "profile": "oci-hsg",
                "account": "nemotron_sw_post",
                "partition": "batch",
                "scratch_root": "/raid/scratch",
                "pyxis_available": True,
                "architecture": "aarch64",
                "gpu_count": 4,
            }
        )
        + "\n"
    )
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    calls = tmp_path / "sbatch.calls"
    sbatch = fake_bin / "sbatch"
    sbatch.write_text(
        "#!/usr/bin/env bash\n"
        'printf "%s\\n" "$*" >>"$SBATCH_CALLS"\n'
        'if [[ " $* " == *" --test-only "* ]]; then echo "job 9001"; else echo "9002"; fi\n'
    )
    sbatch.chmod(0o755)
    output_receipt = tmp_path / "submission.jsonl"
    experiment = _study_experiment()
    canary_receipt = tmp_path / "canary.json"
    canary_receipt.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "experiment_id": experiment.experiment_id,
                "completed_steps": experiment.schedule.canary_steps,
                "checkpoint_verified": True,
                "export_verified": True,
                "all_gpus_active": True,
                "status": "passed",
            }
        )
        + "\n"
    )
    submitter = Path(__file__).resolve().parents[1] / "common/specdec/submit_qwen4b_study.sh"
    assert "--assistant-token-target" in submitter.read_text()
    env = os.environ | {
        "PATH": f"{fake_bin}:{os.environ['PATH']}",
        "SBATCH_CALLS": str(calls),
    }

    completed = subprocess.run(
        [
            str(submitter),
            "--manifest",
            str(manifest_path),
            "--index",
            "0",
            "--profile",
            str(profile),
            "--readiness",
            str(receipt),
            "--receipt",
            str(output_receipt),
            "--canary-receipt",
            str(canary_receipt),
            "--max-steps",
            str(experiment.schedule.canary_steps),
            "--save-steps",
            "10",
        ],
        cwd=Path(__file__).resolve().parents[1],
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    call_lines = calls.read_text().splitlines()
    assert "--test-only" in call_lines[0]
    assert "--test-only" not in call_lines[1]
    assert "--account=nemotron_sw_post" in call_lines[1]
    assert "--nodes=2" in call_lines[1]
    assert "--gpus-per-node=4" in call_lines[1]
    record = json.loads(output_receipt.read_text())
    assert record["job_id"] == "9002"
    assert record["experiment_id"] == _study_experiment().experiment_id
    assert record["test_only_passed"] is True

    blocked_parent = tmp_path / "receipt-parent-is-a-file"
    blocked_parent.write_text("not a directory\n")
    before = calls.read_text().splitlines()
    blocked = subprocess.run(
        [
            str(submitter),
            "--manifest",
            str(manifest_path),
            "--index",
            "0",
            "--profile",
            str(profile),
            "--readiness",
            str(receipt),
            "--receipt",
            str(blocked_parent / "submission.json"),
            "--max-steps",
            str(experiment.schedule.canary_steps),
            "--save-steps",
            "10",
        ],
        cwd=Path(__file__).resolve().parents[1],
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )

    assert blocked.returncode != 0
    assert calls.read_text().splitlines() == before

    receipt_is_directory = tmp_path / "submission-receipt-directory"
    receipt_is_directory.mkdir()
    blocked = subprocess.run(
        [
            str(submitter),
            "--manifest",
            str(manifest_path),
            "--index",
            "0",
            "--profile",
            str(profile),
            "--readiness",
            str(receipt),
            "--receipt",
            str(receipt_is_directory),
            "--max-steps",
            str(experiment.schedule.canary_steps),
            "--save-steps",
            "10",
        ],
        cwd=Path(__file__).resolve().parents[1],
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )

    assert blocked.returncode != 0
    assert calls.read_text().splitlines() == before


def test_html_report_discloses_corpus_weights_and_preliminary_status() -> None:
    """The shareable report must not present planned Qwen4B arms as results."""
    report = (
        Path(__file__).resolve().parents[3] / "reports/qwen3-4b-hybrid-study/index.html"
    ).read_text()

    assert "Balanced PTV2" in report
    assert "75% PTV2 / 25% PTV3" in report
    assert "50% PTV2 / 50% PTV3" in report
    assert "256M assistant tokens" in report
    assert "Thinking ON" in report
    assert "PLANNED — no benchmark result yet" in report
    assert "make_nemotron_ptv2_dataset.py" in report
    assert "make_nemotron_ptv3_dataset.py" in report
