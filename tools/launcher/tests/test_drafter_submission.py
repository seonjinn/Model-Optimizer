# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Contracts for the reusable OCI-HSG drafter workflow."""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest
from common.specdec.drafter_job_manifest import (
    DrafterExperiment,
    PinnedPaths,
    SlurmSettings,
    TargetTopology,
    canonical_manifest,
    load_manifest,
    speculative_tokens,
    validate_topology,
    write_manifest,
)

_LAUNCHER_DIR = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize(
    ("nodes", "segment", "role"),
    [(4, 4, "training"), (1, 1, "evaluation")],
)
def test_validate_topology_accepts_supported_oci_segments(
    nodes: int, segment: int, role: str
) -> None:
    """Training and evaluation allocations stay within one NVL72 segment."""
    validate_topology(nodes, segment, role)


@pytest.mark.parametrize(
    ("nodes", "segment", "role"),
    [(4, 3, "training"), (19, 19, "training"), (1, 2, "evaluation")],
)
def test_validate_topology_rejects_invalid_segments(nodes: int, segment: int, role: str) -> None:
    """A segment must divide nodes and cannot exceed the 18-node block size."""
    with pytest.raises(ValueError):
        validate_topology(nodes, segment, role)


@pytest.mark.parametrize(
    ("method", "block_size", "expected"),
    [("dflash", 8, 7), ("dflash", 16, 15), ("dspark", 8, 8), ("dspark", 16, 16)],
)
def test_speculative_tokens_uses_the_pinned_b_k_matrix(
    method: str, block_size: int, expected: int
) -> None:
    """Every public evaluator run uses the corresponding DFlash/DSpark horizon."""
    assert speculative_tokens(method, block_size) == expected


def _experiment() -> DrafterExperiment:
    return DrafterExperiment(
        target="Qwen/Qwen3-30B-A3B",
        dataset="open-perfectblend",
        method="dflash",
        block_size=8,
        cumulative_max_steps=(500, 1000),
        run_name="qwen3-30b-dflash-b8",
        topology=TargetTopology(
            target_kind="qwen3-30b-a3b",
            capture_ids=(2, 13, 24, 35, 46, 48),
            serve_tp=2,
            per_device_train_batch_size=4,
            gradient_accumulation_steps=16,
            num_attention_heads=32,
            num_key_value_heads=4,
            head_dim=128,
            intermediate_size=6144,
        ),
        paths=PinnedPaths(
            source_path="/home/user/ModelOpt",
            source_sha="a" * 40,
            image_path="/lustre/images/vllm.sqsh",
            runtime_archive_path="/lustre/runtimes/modelopt-runtime.tar.zst",
            runtime_archive_sha256="b" * 64,
            target_path="/lustre/models/Qwen3-30B-A3B",
            dataset_path="/lustre/datasets/open-perfectblend.parquet",
            output_root="/lustre/results/qwen3-30b-dflash-b8",
        ),
        slurm=SlurmSettings(
            account="nemotron_n3_post",
            partition="batch",
            nodes=4,
            gpus_per_node=4,
            segment=4,
        ),
    )


def test_canonical_manifest_is_stable_and_written_atomically(tmp_path: Path) -> None:
    """Canonical manifests are deterministic durable inputs for later submitters."""
    experiment = _experiment()
    expected = canonical_manifest((experiment,))
    output = tmp_path / "manifest.json"

    write_manifest(output, (experiment,))

    assert output.read_text() == expected
    assert json.loads(expected)["experiments"][0]["num_speculative_tokens"] == 7
    assert json.loads(expected)["experiments"][0]["sample_size"] == 1_300_000
    assert not list(tmp_path.glob(".manifest.json.*"))


def test_written_manifest_round_trips_tuple_topology_fields(tmp_path: Path) -> None:
    """JSON list encoding must restore immutable topology tuples when loaded."""
    experiment = _experiment()
    output = tmp_path / "manifest.json"

    write_manifest(output, (experiment,))

    assert load_manifest(output) == (experiment,)


@pytest.mark.parametrize(
    ("field", "value"),
    [("source_path", "relative/source"), ("source_sha", "short"), ("image_path", "")],
)
def test_pinned_paths_require_exact_absolute_values(field: str, value: str) -> None:
    """A mutable path or non-commit source revision cannot enter a manifest."""
    values = _experiment().paths.__dict__.copy()
    values[field] = value
    with pytest.raises(ValueError):
        PinnedPaths(**values)


def test_pinned_paths_reject_normalized_traversal_outside_the_declared_root() -> None:
    """Canonical path validation closes lexical ``..`` containment escapes."""
    values = _experiment().paths.__dict__.copy()
    values["target_path"] = "/lustre/models/../../home/other"
    with pytest.raises(ValueError):
        PinnedPaths(**values)


@pytest.mark.parametrize(
    ("kind", "per_device", "accumulation", "capture_ids", "serve_tp"),
    [
        ("qwen3-30b-a3b", 4, 16, (2, 13, 24, 35, 46, 48), 2),
        ("qwen3-235b-a22b", 2, 32, (2, 25, 47, 69, 92, 94), 4),
    ],
)
def test_target_topology_pins_capture_ids_and_global_batch_arithmetic(
    kind: str,
    per_device: int,
    accumulation: int,
    capture_ids: tuple[int, ...],
    serve_tp: int,
) -> None:
    """Both targets resolve to world-size eight and an exact global batch of 512."""
    topology = TargetTopology.for_kind(kind)

    assert topology.capture_ids == capture_ids
    assert topology.serve_tp == serve_tp
    assert topology.per_device_train_batch_size == per_device
    assert topology.gradient_accumulation_steps == accumulation
    assert topology.per_device_train_batch_size * topology.gradient_accumulation_steps * 8 == 512


def test_run_name_cannot_be_blank() -> None:
    """Receipts and output namespaces always have an explicit identity."""
    values = _experiment().__dict__.copy()
    values["run_name"] = "  "
    with pytest.raises(ValueError):
        DrafterExperiment(**values)


def test_model_staging_is_pinned_and_node_local_until_completion() -> None:
    """Staging uses one segment, a pinned revision, and only durable Lustre output."""
    script = (_LAUNCHER_DIR / "common/specdec/stage_hf_model.sh").read_text()

    for required in (
        "--segment=1",
        "--revision",
        "/raid/scratch",
        "HF_HOME",
        "HF_HUB_CACHE",
        "LOCK",
        "snapshot_download",
        "sbatch --test-only",
        "squeue -h -n",
        "xargs -0 -r -P 4",
        "snapshot-manifest.json",
        "completion.json",
    ):
        assert required in script
    assert "pip install" not in script
    assert "git clone" not in script
    assert "find /lustre" not in script
    assert "rm -rf /lustre" not in script


def test_runtime_archive_staging_is_bounded_and_atomically_published() -> None:
    """The legacy Lustre venv becomes one checksummed archive without a rebuild."""
    script = (_LAUNCHER_DIR / "common/specdec/stage_relocatable_runtime_archive.sh").read_text()
    assert 'SCRATCH_ROOT="/raid/scratch/${USER}"' in script

    for required in (
        "--segment=1",
        "/raid/scratch",
        "tar --dereference --create",
        "tar --list",
        'tar --list --use-compress-program=zstd --file="$archive" >"$listing"',
        "grep -q '/bin/activate$' \"$listing\"",
        "sha256sum",
        ".provenance.json",
        ".partial-",
        'mv "$temporary" "$OUTPUT_ARCHIVE"',
        'tar --dereference --create --use-compress-program=zstd --file="$archive" -C "$SOURCE_RUNTIME" .',
        'submitted="$(sbatch --parsable',
        '--account="$ACCOUNT"',
        '--partition="$PARTITION"',
        "--gpus-per-node=4",
    ):
        assert required in script
    for forbidden in ("pip install", "git clone", "find /lustre", "rm -rf"):
        assert forbidden not in script


def test_training_wave_batches_scheduler_history_with_parseable_output() -> None:
    """One scheduler snapshot must cover every tuple in a submission wave."""
    submitter = (_LAUNCHER_DIR / "common/specdec/submit_drafter_training_wave.sh").read_text()

    assert 'squeue -h -u "$USER"' in submitter
    assert 'sacct -X -n -P -u "$USER"' in submitter
    assert "JobName%64,JobIDRaw" in submitter
    assert "--parsable2" in submitter or "-P" in submitter
    assert 'exports="ALL,SCHEDULER_JOBS_SNAPSHOT=,MANIFEST_PATH=' in submitter


def test_training_wave_writes_slurm_logs_beside_durable_experiment_outputs() -> None:
    """Formal jobs must not leave default Slurm logs in the home checkout."""
    submitter = (_LAUNCHER_DIR / "common/specdec/submit_drafter_training_wave.sh").read_text()

    assert "experiment.paths.output_root" in submitter
    assert 'mkdir -p "$output_root/logs"' in submitter
    assert '--output="${output_root}/logs/slurm-%j.out"' in submitter
    assert '--error="${output_root}/logs/slurm-%j.err"' in submitter


def test_training_wave_uses_the_fixed_four_node_streaming_topology() -> None:
    """A wave renders unique tuple jobs with node-local mutable runtime state."""
    submitter = (_LAUNCHER_DIR / "common/specdec/submit_drafter_training_wave.sh").read_text()
    runner = (_LAUNCHER_DIR / "common/specdec/run_drafter_training.sbatch").read_text()

    for required in (
        "-N4",
        "--segment=4",
        "sbatch --test-only",
        "squeue -h -n",
        "duplicate training tuple",
        "sacct -X -n --name",
        "identity",
        "--dependency=afterok:",
        "receipt",
    ):
        assert required in submitter
    for required in (
        "#SBATCH -N 4",
        "#SBATCH --segment=4",
        "SERVE_NODES=2",
        "EAGLE_CAPTURE_IDS",
        "IMAGE_PATH",
        "--container-image",
        "gradient_accumulation_steps",
        "num_attention_heads",
        "GLOBAL_BATCH_SIZE=512",
        "TRAINER_NODES=2",
        "GPUS_PER_NODE=4",
        "PER_DEVICE_TRAIN_BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS * TRAINER_NODES * GPUS_PER_NODE",
        "/home",
        "rev-parse HEAD",
        "status --porcelain",
        "/raid/scratch/${SLURM_JOB_ID}",
        "RUNTIME_ARCHIVE",
        "tar --extract",
        "HF_HOME",
        "TRITON_CACHE_DIR",
        "XDG_CACHE_HOME",
        "SQLITE_TMPDIR",
        "srun --nodes=4 --ntasks=4 --ntasks-per-node=1",
        "training.output_dir",
        "LOSS_OBJECTIVE=dpace",
        "LOSS_OBJECTIVE=decay",
    ):
        assert required in runner
    assert "/lustre/.cache" not in runner
    assert "pip install" not in runner
    assert "git clone" not in runner


def test_training_runner_relocates_runtime_and_stages_only_role_inputs() -> None:
    """The pinned runtime and mounts remain valid after per-node extraction."""
    runner = (_LAUNCHER_DIR / "common/specdec/run_drafter_training.sbatch").read_text()

    for required in (
        "VIRTUAL_ENV",
        "sed -i",
        "PYTHONPATH",
        "--no-container-mount-home",
        "${OUTPUT_ROOT}:${OUTPUT_ROOT}",
        "SLURM_NODEID < SERVE_NODES",
        "WANDB_PROJECT",
        "WANDB_RUN_GROUP",
        "WANDB_CACHE_DIR",
        "report_to=wandb",
        'training.run_name="${RUN_NAME}-s${MAX_STEPS}"',
        "SERVE_BLOCK_SIZE=32",
        "SERVE_MAX_MODEL_LEN=8192",
        "SERVE_READY_TIMEOUT=1800",
        "HS_POOL_SLOTS=32",
        "SERVE_MAX_NUM_SEQS=16",
        'training.save_steps="${MAX_STEPS}"',
        "training.save_total_limit=2",
        'cp -a "$SOURCE_PATH"',
        'cp -aL "$TARGET_PATH"',
        'cp -aL "$DATASET_PATH"',
        "import accelerate, datasets, modelopt, wandb",
        "is_relative_to",
        "modules/Model-Optimizer/modelopt_recipes/general/speculative_decoding/${METHOD}.yaml",
        "dflash.dflash_mask_token_id=151669",
        'data.sample_size="${SAMPLE_SIZE}"',
        "${CONTROL_ROOT}:/scratchspace",
    ):
        assert required in runner
    assert "training.global_batch_size=512" not in runner
    assert "EXTRA_MODELOPT_DOTLIST" not in runner


def test_submitter_exports_the_home_launcher_root_to_the_spooled_runner() -> None:
    """A copied Slurm batch script must not derive imports from its spool directory."""
    submitter = (_LAUNCHER_DIR / "common/specdec/submit_drafter_training_wave.sh").read_text()
    runner = (_LAUNCHER_DIR / "common/specdec/run_drafter_training.sbatch").read_text()

    assert "LAUNCHER_ROOT=${LAUNCHER_ROOT}" in submitter
    assert 'for name in MANIFEST_PATH EXPERIMENT_INDEX MAX_STEPS LAUNCHER_ROOT' in runner
    assert 'LAUNCHER_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"' not in runner


def test_training_runner_stages_pattern_packager_layout_and_shared_control_dir() -> None:
    """The staged source resolves recipe paths and all ranks rendezvous in one control mount."""
    runner = (_LAUNCHER_DIR / "common/specdec/run_drafter_training.sbatch").read_text()
    modelopt_link = _LAUNCHER_DIR / "modules/Model-Optimizer"

    for required in (
        'cp -a "$SOURCE_PATH"',
        'CONTROL_ROOT="${OUTPUT_ROOT}/control"',
        'mkdir -p "$OUTPUT_ROOT" "$CONTROL_ROOT"',
        "${CONTROL_ROOT}:/scratchspace",
    ):
        assert required in runner
    assert modelopt_link.is_symlink()
    assert modelopt_link.readlink() == Path("../../..")
    assert 'cp -aL "$SOURCE_PATH"' not in runner
    assert 'ln -s .. "$node_root/source/modules/Model-Optimizer"' not in runner


def test_host_staging_preserves_one_bounded_diagnostic_log_per_node() -> None:
    """A failed staging command must remain attributable without a shared log funnel."""
    runner = (_LAUNCHER_DIR / "common/specdec/run_drafter_training.sbatch").read_text()

    assert 'STAGE_LOG_ROOT="${OUTPUT_ROOT}/logs/stage-${SLURM_JOB_ID}"' in runner
    assert 'mkdir -p "$STAGE_LOG_ROOT"' in runner
    assert 'exec >"${STAGE_LOG_ROOT}/node-${SLURM_NODEID}.log" 2>&1' in runner
    assert "set -x" in runner


def test_missing_optional_target_sidecar_does_not_fail_host_staging() -> None:
    """An absent final optional tokenizer file must still leave the subshell successful."""
    runner = (_LAUNCHER_DIR / "common/specdec/run_drafter_training.sbatch").read_text()

    assert 'if [[ -e "$TARGET_PATH/$sidecar" ]]; then' in runner
    assert '[[ -e "$TARGET_PATH/$sidecar" ]] && cp' not in runner
    assert 'echo "host staging complete: node=${SLURM_NODEID}"' in runner


def test_runtime_relocation_accepts_exported_and_quoted_activate_assignments() -> None:
    """OCI virtualenv activation lines may use export and shell quotes."""
    parser = re.compile(r"^\s*export\s+VIRTUAL_ENV=(.*)$")
    cygwin = parser.match("VIRTUAL_ENV=$(cygpath /lustre/wrong)")
    unquoted = parser.match("export VIRTUAL_ENV=/lustre/runtime")
    quoted = parser.match('export VIRTUAL_ENV="/lustre/runtime"')
    assert cygwin is None
    assert unquoted and unquoted.group(1) == "/lustre/runtime"
    assert quoted and quoted.group(1).strip("\"'") == "/lustre/runtime"

    for script_name in (
        "run_drafter_training.sbatch",
        "probe_relocatable_runtime.sh",
        "submit_drafter_resume_chain.sh",
    ):
        script = (_LAUNCHER_DIR / f"common/specdec/{script_name}").read_text()
        assert "export[[:space:]]+VIRTUAL_ENV=(.*)$" in script
        assert 'old_venv="${old_venv%\\"}"' in script


def test_training_mounts_existing_node_local_scratch_parent() -> None:
    """Pyxis must not bind a job-specific scratch path before it is created."""
    runner = (_LAUNCHER_DIR / "common/specdec/run_drafter_training.sbatch").read_text()
    probe = (_LAUNCHER_DIR / "common/specdec/probe_relocatable_runtime.sh").read_text()

    assert "/raid/scratch:/raid/scratch" in runner
    assert "${SCRATCH_JOB_ROOT}:${SCRATCH_JOB_ROOT}" not in runner
    assert "/raid/scratch:/raid/scratch" in probe


def test_streaming_serve_logs_stay_on_node_local_scratch() -> None:
    """Shared scratchspace carries only the few cross-node rendezvous files."""
    runner = (_LAUNCHER_DIR / "common/specdec/run_drafter_training.sbatch").read_text()
    streaming = (_LAUNCHER_DIR / "common/eagle3/train_eagle_streaming.sh").read_text()

    assert 'SERVE_LOG_DIR="${SCRATCH_JOB_ROOT}/node-${SLURM_NODEID}/logs"' in runner
    assert "${SERVE_LOG_DIR:-/scratchspace}/vllm_serve.${NODEID}.log" in streaming


def test_cumulative_runner_writes_a_checkpoint_for_every_resume_boundary() -> None:
    """Each same-output cumulative wave leaves get_last_checkpoint() an exact resume."""
    runner = (_LAUNCHER_DIR / "common/specdec/run_drafter_training.sbatch").read_text()

    assert 'training.output_dir="${OUTPUT_ROOT}"' in runner
    assert 'training.save_steps="${MAX_STEPS}"' in runner
    assert "training.save_total_limit=2" in runner


def test_training_runner_preserves_proven_production_training_semantics() -> None:
    """Production waves must not inherit incompatible generic recipe defaults."""
    runner = (_LAUNCHER_DIR / "common/specdec/run_drafter_training.sbatch").read_text()

    for required in (
        "training.training_seq_len=4096",
        "training.answer_only_loss=false",
        "training.seed=42",
        'mkdir -p "$OUTPUT_ROOT"',
    ):
        assert required in runner


def test_runtime_probe_verifies_a_relocated_bundle_in_the_pinned_container() -> None:
    """The reusable preflight verifies imports after the archive is moved to scratch."""
    script = (_LAUNCHER_DIR / "common/specdec/probe_relocatable_runtime.sh").read_text()

    for required in (
        "/home",
        "/lustre",
        "/raid/scratch",
        "--container-image",
        "tar --extract",
        "VIRTUAL_ENV",
        "sed -i",
        "PYTHONPATH",
        'cp -a "$SOURCE_PATH"',
        "import accelerate, datasets, modelopt, wandb",
        "is_relative_to",
        "--runtime-sha256",
        "sha256sum",
        '--account="$ACCOUNT"',
        '--partition="$PARTITION"',
        "--gpus-per-node=4",
        "--segment=1",
        "--job-name=modelopt-runtime-probe",
        '--output="$PROBE_LOG"',
        '--error="$PROBE_LOG"',
        "${RUNTIME_ARCHIVE_ROOT}:${RUNTIME_ARCHIVE_ROOT}",
        "runtime-probe-${SLURM_JOB_ID}.trace",
        'exec >>"$TRACE_LOG" 2>&1',
    ):
        assert required in script
    assert "pip install" not in script
    assert "git clone" not in script
    assert "${SCRIPT_PATH}:${SCRIPT_PATH}" not in script
    assert "--no-container-mount-home" in script
    assert "${SOURCE_PATH}:${SOURCE_PATH}" in script
    assert 'cp -aL "$SOURCE_PATH"' not in script


def test_training_manifest_pins_and_verifies_runtime_archive_bytes() -> None:
    """A mutable archive path cannot silently change the production runtime."""
    manifest = PinnedPaths.__dataclass_fields__
    runner = (_LAUNCHER_DIR / "common/specdec/run_drafter_training.sbatch").read_text()

    assert "runtime_archive_sha256" in manifest
    assert "RUNTIME_ARCHIVE_SHA256" in runner
    assert 'sha256sum "$RUNTIME_ARCHIVE"' in runner


def test_node_local_input_staging_dereferences_hf_blob_symlinks() -> None:
    """HF cache links are materialized before their Lustre backing paths disappear."""
    runner = (_LAUNCHER_DIR / "common/specdec/run_drafter_training.sbatch").read_text()

    assert "srun --nodes=4 --ntasks=4 --ntasks-per-node=1 bash -c '\n" in runner
    for source in ("TARGET_PATH", "DATASET_PATH"):
        assert f'cp -aL "${source}"' in runner
        assert f'cp -a "${source}"' not in runner
    assert 'cp -a "$SOURCE_PATH"' in runner
    assert 'cp -aL "$SOURCE_PATH"' not in runner


def test_resume_chain_gates_each_cumulative_wave_on_public_acceptance() -> None:
    """Only a successful nine-subset evaluation releases the next stable-output wave."""
    script = (_LAUNCHER_DIR / "common/specdec/submit_drafter_resume_chain.sh").read_text()
    readme = (_LAUNCHER_DIR / "examples/Qwen/Qwen3-30B-A3B/README.md").read_text()

    for required in (
        "afterok",
        "--segment=1",
        "--nodes=1",
        "sbatch --test-only",
        "acceptance.csv",
        "HumanEval,math_reasoning,qa,question,rag,summarization,tool_call,translation,writing",
        "--time=03:55:00",
        "squeue -h -n",
        "sacct -X -n --name",
        "--run-evaluation",
        "EXPERIMENT_IDENTITY",
        "JobName%64,JobIDRaw",
        "ALL,SCHEDULER_JOBS_SNAPSHOT=,EVAL_IMAGE=",
        "receipt",
    ):
        assert required in script
    assert "--wrap" not in script
    for required in (
        "Qwen3-30B-A3B",
        "Qwen3-235B-A22B",
        "Thinking",
        "DFlash B8",
        "DSpark B16",
        "--segment=4",
        "--segment=1",
        "/raid/scratch",
        "receipt",
        "Q30 uses per-device batch 4 with gradient accumulation 16",
        "Q235 uses per-device batch 2 with gradient accumulation 32",
    ):
        assert required in readme
    assert "512 / (2 * 4) = 64" not in readme
