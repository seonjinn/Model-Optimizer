# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Contracts for the reusable OCI-HSG drafter workflow."""

from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import textwrap
from dataclasses import replace
from pathlib import Path

import pytest
from common.specdec.build_drafter_full_manifest import (
    full_convergence_boundaries,
    legacy_q30_seed_expectations,
    readable_job_name,
    select_experiments,
    validate_legacy_seed_identity,
)
from common.specdec.drafter_job_manifest import (
    DrafterExperiment,
    PinnedPaths,
    SlurmSettings,
    TargetTopology,
    canonical_manifest,
    legacy_training_fingerprint,
    load_manifest,
    speculative_tokens,
    topology_v2_training_fingerprint,
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
            gradient_accumulation_steps=32,
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
            nodes=2,
            gpus_per_node=4,
            segment=2,
        ),
    )


def test_canonical_manifest_is_stable_and_written_atomically(tmp_path: Path) -> None:
    """Canonical manifests are deterministic durable inputs for later submitters."""
    experiment = _experiment()
    expected = canonical_manifest((experiment,))
    output = tmp_path / "manifest.json"

    write_manifest(output, (experiment,))

    assert output.read_text() == expected
    assert "image_sha256" not in expected
    assert json.loads(expected)["experiments"][0]["num_speculative_tokens"] == 7
    assert json.loads(expected)["experiments"][0]["sample_size"] == 1_300_000
    assert not list(tmp_path.glob(".manifest.json.*"))


def test_legacy_q30_template_migration_is_explicit_and_target_scoped(tmp_path: Path) -> None:
    """Only the builder opt-in can translate the former Q30 world-eight topology."""
    document = json.loads(canonical_manifest((_experiment(),)))
    entry = document["experiments"][0]
    entry["topology"]["gradient_accumulation_steps"] = 16
    entry["slurm"]["nodes"] = 4
    entry["slurm"]["segment"] = 4
    path = tmp_path / "legacy.json"
    path.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n")

    with pytest.raises(ValueError):
        load_manifest(path)
    migrated = load_manifest(path, migrate_legacy_q30_topology=True)

    assert migrated[0].topology.gradient_accumulation_steps == 32
    assert migrated[0].slurm.nodes == 2
    assert migrated[0].slurm.segment == 2


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
    ("kind", "per_device", "accumulation", "capture_ids", "serve_tp", "trainer_world"),
    [
        ("qwen3-30b-a3b", 4, 32, (2, 13, 24, 35, 46, 48), 2, 4),
        ("qwen3-235b-a22b", 2, 32, (2, 25, 47, 69, 92, 94), 4, 8),
    ],
)
def test_target_topology_pins_capture_ids_and_global_batch_arithmetic(
    kind: str,
    per_device: int,
    accumulation: int,
    capture_ids: tuple[int, ...],
    serve_tp: int,
    trainer_world: int,
) -> None:
    """Both targets resolve to world-size eight and an exact global batch of 512."""
    topology = TargetTopology.for_kind(kind)

    assert topology.capture_ids == capture_ids
    assert topology.serve_tp == serve_tp
    assert topology.per_device_train_batch_size == per_device
    assert topology.gradient_accumulation_steps == accumulation
    assert (
        topology.per_device_train_batch_size * topology.gradient_accumulation_steps * trainer_world
        == 512
    )


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
        "--source-dir",
        "--source-id",
        'cp -aL "$SOURCE_DIR"/. "$local_snapshot"/',
        "sbatch --test-only",
        "squeue -h -n",
        "xargs -0 -r -P 4",
        "snapshot-manifest.json",
        "completion.json",
        'mv -T --no-clobber "$partial" "$ARTIFACT_DIR"',
    ):
        assert required in script
    assert "pip install" not in script
    assert "git clone" not in script
    assert "find /lustre" not in script
    assert "rm -rf /lustre" not in script


def test_model_staging_fails_closed_and_bounds_shared_filesystem_operations() -> None:
    """Local sources are copied once through scratch without replacing durable trees."""
    script = (_LAUNCHER_DIR / "common/specdec/stage_hf_model.sh").read_text()

    for required in (
        '[[ "$SOURCE_DIR" == /lustre/* ]]',
        '[[ "$ARTIFACT_DIR" == /lustre/* ]]',
        '[[ "$SCRATCH_ROOT" == /raid/scratch/* ]]',
        'SOURCE_CANONICAL="$(realpath -m -- "$SOURCE_DIR")"',
        'ARTIFACT_CANONICAL="$(realpath -m -- "$ARTIFACT_DIR")"',
        '[[ "$SOURCE_CANONICAL" == /lustre/* ]]',
        '[[ "$ARTIFACT_CANONICAL" == /lustre/* ]]',
        '[[ "$ARTIFACT_CANONICAL" != "$SOURCE_CANONICAL"/* ]]',
        '[[ "$SOURCE_CANONICAL" != "$ARTIFACT_CANONICAL"/* ]]',
        '[[ ! -e "$ARTIFACT_DIR" && ! -L "$ARTIFACT_DIR" ]]',
        '[[ ! -e "$partial" && ! -L "$partial" ]]',
        'find "$local_snapshot" -type f -print0',
        '[[ ! -e "$partial" ]] || {',
    ):
        assert required in script
    assert script.count('cp -aL "$SOURCE_DIR"/. "$local_snapshot"/') == 1
    assert 'rm -rf "$partial"' not in script
    assert 'rm -rf "$ARTIFACT_DIR"' not in script


def test_model_staging_rejects_broken_destinations_and_cleans_failed_publishes() -> None:
    """Failure cleanup covers errexit, and lexical symlinks cannot evade conflict checks."""
    script = (_LAUNCHER_DIR / "common/specdec/stage_hf_model.sh").read_text()

    symlink_guard = '[[ ! -L "$ARTIFACT_DIR" ]] || {'
    canonicalization = 'ARTIFACT_CANONICAL="$(realpath -m -- "$ARTIFACT_DIR")"'
    assert symlink_guard in script
    assert script.index(symlink_guard) < script.index(canonicalization)
    assert "run_stage() (" in script
    assert "trap cleanup EXIT" in script
    assert "trap cleanup RETURN" not in script


def test_model_staging_copy_pipeline_materializes_nested_files(tmp_path: Path) -> None:
    """The exact xargs pipeline passes each discovered file as the child source argument."""
    source = tmp_path / "snapshot"
    artifact = tmp_path / "partial"
    source_file = source / "nested" / "weights.bin"
    source_file.parent.mkdir(parents=True)
    artifact.mkdir()
    source_file.write_bytes(b"weights")
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    fake_cp = fake_bin / "cp"
    fake_cp.write_text(
        "#!/usr/bin/env bash\n"
        "set -euo pipefail\n"
        "[[ $1 == --reflink=auto ]] && shift\n"
        "[[ $1 == --preserve=mode,timestamps ]] && shift\n"
        'exec /bin/cp "$@"\n'
    )
    fake_cp.chmod(0o755)
    script = (_LAUNCHER_DIR / "common/specdec/stage_hf_model.sh").read_text()
    start = script.index('    find "$local_snapshot" -type f -print0')
    end = script.index("\n    printf ", start)
    pipeline = textwrap.dedent(script[start:end])

    completed = subprocess.run(
        ["bash", "-c", f"set -euo pipefail\n{pipeline}"],
        capture_output=True,
        check=False,
        env={
            **os.environ,
            "PATH": f"{fake_bin}:{os.environ['PATH']}",
            "local_snapshot": str(source),
            "partial": str(artifact),
        },
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    assert (artifact / "nested" / "weights.bin").read_bytes() == b"weights"


def test_runtime_archive_staging_is_bounded_and_atomically_published() -> None:
    """The legacy Lustre venv becomes one checksummed archive without a rebuild."""
    script = (_LAUNCHER_DIR / "common/specdec/stage_relocatable_runtime_archive.sh").read_text()
    assert 'SCRATCH_ROOT="/raid/scratch/${USER}"' in script

    for required in (
        "--segment=1",
        "/raid/scratch",
        "tar --create",
        "tar --list",
        'tar --list --use-compress-program=zstd --file="$archive" >"$listing"',
        "grep -q '/bin/activate$' \"$listing\"",
        "sha256sum",
        ".provenance.json",
        ".partial-",
        'mv "$temporary" "$OUTPUT_ARCHIVE"',
        'tar --create --use-compress-program=zstd --file="$archive" -C "$SOURCE_RUNTIME" .',
        'submitted="$(sbatch --parsable',
        '--account="$ACCOUNT"',
        '--partition="$PARTITION"',
        "--gpus-per-node=4",
    ):
        assert required in script
    for forbidden in ("pip install", "git clone", "find /lustre", "rm -rf"):
        assert forbidden not in script
    assert "tar --dereference" not in script


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


def test_training_wave_dry_run_does_not_create_experiment_output_namespace() -> None:
    """Scheduler test-only must not poison a later atomic checkpoint seed."""
    submitter = (_LAUNCHER_DIR / "common/specdec/submit_drafter_training_wave.sh").read_text()

    dry_run = submitter.index('if [[ "$DRY_RUN" -eq 1 ]]')
    create_logs = submitter.index('mkdir -p "$output_root/logs"')

    assert dry_run < create_logs


def test_training_wave_uses_the_target_specific_streaming_topology() -> None:
    """Q30 uses 2 nodes while Q235 retains the proven 4-node allocation."""
    submitter = (_LAUNCHER_DIR / "common/specdec/submit_drafter_training_wave.sh").read_text()
    runner = (_LAUNCHER_DIR / "common/specdec/run_drafter_training.sbatch").read_text()

    for required in (
        "experiment.slurm.nodes",
        "experiment.slurm.segment",
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
        "ALLOCATED_NODES",
        "SERVE_NODES=$((ALLOCATED_NODES / 2))",
        "TRAINER_NODES=$((ALLOCATED_NODES - SERVE_NODES))",
        "EAGLE_CAPTURE_IDS",
        "IMAGE_PATH",
        "--container-image",
        "gradient_accumulation_steps",
        "num_attention_heads",
        "GLOBAL_BATCH_SIZE=512",
        "TRAINER_NODES",
        'GPUS_PER_NODE="${manifest_values[24]}"',
        "PER_DEVICE_TRAIN_BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS * TRAINER_NODES * GPUS_PER_NODE",
        "/home",
        "rev-parse HEAD",
        "status --porcelain",
        'SCRATCH_JOB_ROOT="${SCRATCH_ROOT%/}/${SLURM_JOB_ID}"',
        "RUNTIME_ARCHIVE",
        "tar --extract",
        "HF_HOME",
        "TRITON_CACHE_DIR",
        "XDG_CACHE_HOME",
        "SQLITE_TMPDIR",
        'srun --nodes="$ALLOCATED_NODES" --ntasks="$ALLOCATED_NODES" --ntasks-per-node=1',
        "training.output_dir",
        "LOSS_OBJECTIVE=dpace",
        "LOSS_OBJECTIVE=decay",
    ):
        assert required in runner
    assert "/lustre/.cache" not in runner
    assert "pip install" not in runner
    assert "git clone" not in runner


def test_q30_manifest_topology_is_two_nodes_and_uses_a_distinct_output_namespace() -> None:
    """The builder never points a world-four Q30 run at legacy world-eight output."""
    module = (_LAUNCHER_DIR / "common/specdec/build_drafter_full_manifest.py").read_text()
    manifest = (_LAUNCHER_DIR / "common/specdec/drafter_job_manifest.py").read_text()

    for required in (
        '"qwen3-30b-a3b": {"nodes": 2, "segment": 2}',
        '"qwen3-235b-a22b": {"nodes": 4, "segment": 4}',
        'gradient_accumulation_steps": 32',
    ):
        assert required in manifest
    assert 'suffix = "-2n"' in module
    assert "run_name.endswith(suffix)" in module
    assert "output_root.endswith(suffix)" in module


def test_training_fingerprint_includes_batch_and_scheduler_topology() -> None:
    """A topology migration cannot silently reuse an incompatible identity marker."""
    runner = (_LAUNCHER_DIR / "common/specdec/run_drafter_training.sbatch").read_text()

    for required in (
        "experiment.topology.per_device_train_batch_size",
        "experiment.topology.gradient_accumulation_steps",
        "experiment.slurm.nodes",
        "experiment.slurm.gpus_per_node",
        "experiment.slurm.segment",
    ):
        assert required in runner
    assert 'identity["fingerprint_schema"] = schema' in runner
    assert 'if target_kind != "qwen3-30b-a3b"' in runner


def test_q235_fingerprint_remains_the_exact_legacy_hash() -> None:
    """The topology migration must not invalidate any healthy Q235 output."""
    q235 = replace(
        _experiment(),
        target="q235-base",
        topology=TargetTopology.for_kind("qwen3-235b-a22b"),
        slurm=replace(_experiment().slurm, nodes=4, segment=4),
    )
    payload = (
        q235.target,
        q235.dataset,
        q235.method,
        q235.block_size,
        q235.run_name,
        q235.paths.target_path,
        q235.paths.dataset_path,
        q235.paths.output_root,
        q235.topology.target_kind,
        q235.topology.capture_ids,
        q235.topology.serve_tp,
    )
    expected = hashlib.sha256(json.dumps(payload, separators=(",", ":")).encode()).hexdigest()

    assert legacy_training_fingerprint(q235) == expected


def test_q30_topology_v2_fingerprint_binds_all_execution_inputs() -> None:
    """Changing image, runtime, or sampled data invalidates a Q30 training identity."""
    experiment = replace(_experiment(), paths=replace(_experiment().paths, image_sha256="c" * 64))
    original = topology_v2_training_fingerprint(experiment, "oci-hsg")

    variants = (
        replace(
            experiment,
            paths=replace(experiment.paths, image_path="/lustre/images/other.sqsh"),
        ),
        replace(
            experiment,
            paths=replace(
                experiment.paths,
                runtime_archive_path="/lustre/runtimes/other.tar.zst",
            ),
        ),
        replace(
            experiment,
            paths=replace(experiment.paths, runtime_archive_sha256="d" * 64),
        ),
    )
    for variant in variants:
        assert topology_v2_training_fingerprint(variant, "oci-hsg") != original
    changed_image = replace(experiment, paths=replace(experiment.paths, image_sha256="d" * 64))
    assert topology_v2_training_fingerprint(changed_image, "oci-hsg") != original
    changed_sample = replace(experiment)
    object.__setattr__(changed_sample, "sample_size", experiment.sample_size - 1)
    assert topology_v2_training_fingerprint(changed_sample, "oci-hsg") != original


def test_q30_topology_v2_requires_builder_pinned_image_digest() -> None:
    """Legacy manifests remain readable, but a new Q30 identity cannot omit image bytes."""
    with pytest.raises(ValueError, match="pinned image SHA-256"):
        topology_v2_training_fingerprint(_experiment(), "oci-hsg")

    builder = (_LAUNCHER_DIR / "common/specdec/build_drafter_full_manifest.py").read_text()
    runner = (_LAUNCHER_DIR / "common/specdec/run_drafter_training.sbatch").read_text()
    assert 'parser.add_argument("--image-sha256", required=True)' in builder
    assert "image_sha256=image_sha256" in builder
    assert 'sha256sum "$IMAGE_PATH"' in runner


def test_q30_checkpoint_seed_is_explicit_atomic_and_accepts_extra_rng_ranks() -> None:
    """Legacy world-eight checkpoints seed a new root without mutating old output."""
    chain = (_LAUNCHER_DIR / "common/specdec/submit_drafter_full_chain.sh").read_text()
    lifecycle = (_LAUNCHER_DIR / "common/specdec/drafter_requeue_lifecycle.sh").read_text()

    for required in (
        "--seed-q30-from-legacy",
        'legacy_output_root="${output_root%-2n}"',
        'temporary = Path(f"{output_root}.seed-partial-',
        "os.link",
        "shutil.copy2",
        "os.rename(temporary, output_root)",
        '"source_checkpoint"',
        'legacy_identity_path = legacy_output_root / "control/training-identity.json"',
        '"source_identity_sha256"',
        '"checkpoint_files_sha256"',
        '"source_experiment_id"',
        '"source_training_fingerprint"',
        '"selected_experiment_tuple"',
        "validate_legacy_seed_identity(",
        'verify_seeded_output "$output_root" "$legacy_output_root" "$source_sha"',
    ):
        assert required in chain
    assert "for rank in range(expected_rng_states)" in lifecycle
    assert (
        "len(" not in lifecycle[lifecycle.index("for rank in range(expected_rng_states)") :][:300]
    )


@pytest.mark.parametrize(
    ("field", "wrong"),
    [("experiment_id", "wrong-id"), ("training_fingerprint", "0" * 64)],
)
def test_q30_seed_rejects_wrong_legacy_experiment_identity(field: str, wrong: str) -> None:
    """A neighboring Q30 checkpoint cannot seed the selected experiment tuple."""
    identity = {
        "experiment_id": "legacy-id",
        "source_sha": "a" * 40,
        "training_fingerprint": "b" * 64,
    }
    identity[field] = wrong

    with pytest.raises(ValueError):
        validate_legacy_seed_identity(identity, "legacy-id", "b" * 64, "a" * 40)


def test_q30_seed_expectations_bind_current_tuple_to_legacy_namespace() -> None:
    """The selected new run derives one exact old identity instead of trusting its path."""
    experiment = replace(
        _experiment(),
        run_name=f"{_experiment().run_name}-2n",
        paths=replace(
            _experiment().paths,
            output_root=f"{_experiment().paths.output_root}-2n",
        ),
    )

    legacy_id, legacy_fingerprint, selected_tuple = legacy_q30_seed_expectations(experiment)

    assert legacy_id == _experiment().experiment_id
    assert legacy_fingerprint == legacy_training_fingerprint(_experiment())
    assert json.loads(selected_tuple) == ["Qwen/Qwen3-30B-A3B", "open-perfectblend", "dflash", 8]


def test_q30_seed_accepts_the_adopted_legacy_source_lineage() -> None:
    """An adopted checkpoint binds to its original source SHA, not the adopting checkout."""
    identity = {
        "adopted_from_source_sha": "a" * 40,
        "experiment_id": "legacy-id",
        "source_sha": "c" * 40,
        "training_fingerprint": "b" * 64,
    }

    validate_legacy_seed_identity(identity, "legacy-id", "b" * 64, "a" * 40)

    with pytest.raises(ValueError):
        validate_legacy_seed_identity(identity, "legacy-id", "b" * 64, "c" * 40)


def test_training_wave_renders_scheduler_flags_from_an_immutable_cluster_profile() -> None:
    """OCI keeps its GPU request while exclusive GB200 profiles omit it."""
    submitter = (_LAUNCHER_DIR / "common/specdec/submit_drafter_training_wave.sh").read_text()
    runner = (_LAUNCHER_DIR / "common/specdec/run_drafter_training.sbatch").read_text()

    for required in (
        "--cluster-profile",
        "--readiness-receipt",
        "DEFAULT_CLUSTER_PROFILE",
        "load_cluster_profile",
        "scheduler_gpu_args",
        'CLUSTER_PROFILE_SHA256="$(sha256sum "$CLUSTER_PROFILE"',
        "profile.training_nodes",
        "profile.training_segment",
        "profile.walltime",
    ):
        assert required in submitter
    assert "--gpus-per-node=4" not in runner
    assert 'scheduler_args+=("--gpus-per-node=${gpus_per_node}")' in submitter


def test_cluster_profile_allocation_is_a_capacity_not_an_exact_node_count() -> None:
    """One profile can safely schedule both 2-node Q30 and 4-node Q235 experiments."""
    submitter = (_LAUNCHER_DIR / "common/specdec/submit_drafter_training_wave.sh").read_text()

    assert "experiment.slurm.nodes > profile.training_nodes" in submitter
    assert "profile.training_nodes % experiment.slurm.nodes" in submitter
    assert "experiment.slurm.segment > profile.training_segment" in submitter
    assert "experiment.slurm.nodes != profile.training_nodes" not in submitter


def test_non_oci_training_requires_a_pinned_readiness_receipt_and_local_scratch() -> None:
    """A stale or Lustre-backed readiness artifact must fail before GPU startup."""
    submitter = (_LAUNCHER_DIR / "common/specdec/submit_drafter_training_wave.sh").read_text()
    runner = (_LAUNCHER_DIR / "common/specdec/run_drafter_training.sbatch").read_text()

    for required in (
        '[[ "$CLUSTER_NAME" == "oci-hsg" || -n "$READINESS_RECEIPT" ]]',
        'READINESS_RECEIPT_SHA256="$(sha256sum "$READINESS_RECEIPT"',
        "validate_scratch_root",
        'receipt["profile"] != profile.name',
        'receipt["scratch_root"]',
        'scratch.is_relative_to(Path("/lustre"))',
    ):
        assert required in submitter
    for required in (
        "CLUSTER_PROFILE_PATH",
        "CLUSTER_PROFILE_SHA256",
        "READINESS_RECEIPT_SHA256",
        'SCRATCH_JOB_ROOT="${SCRATCH_ROOT%/}/${SLURM_JOB_ID}"',
        '"$SCRATCH_ROOT" != /lustre*',
        "${SCRATCH_ROOT}:${SCRATCH_ROOT}",
    ):
        assert required in runner


def test_non_oci_training_namespaces_scheduler_wandb_and_receipt_identity() -> None:
    """Independent clusters cannot share a logical job or W&B resume identity."""
    submitter = (_LAUNCHER_DIR / "common/specdec/submit_drafter_training_wave.sh").read_text()
    runner = (_LAUNCHER_DIR / "common/specdec/run_drafter_training.sbatch").read_text()

    for required in (
        'cluster_tuple_identity="${CLUSTER_NAME}:${identity}:${boundary}"',
        'wandb_run_id="sd-${identity}-${CLUSTER_NAME}"',
        "CLUSTER_NAME=${CLUSTER_NAME}",
        '"cluster":"%s"',
    ):
        assert required in submitter
    assert 'WANDB_RUN_GROUP="${RUN_NAME}-${CLUSTER_NAME}"' in runner
    assert 'TRAINING_RUN_NAME="${RUN_NAME}-${CLUSTER_NAME}"' in runner
    assert 'training.run_name="${TRAINING_RUN_NAME}-s${MAX_STEPS}"' in runner


def test_runner_exports_computed_serve_nodes_and_cluster_name_to_child_shell() -> None:
    """The child shell must see the target-specific role split and cluster namespace."""
    runner = (_LAUNCHER_DIR / "common/specdec/run_drafter_training.sbatch").read_text()

    assert "export SERVE_NODES CLUSTER_NAME" in runner
    assert "export SERVE_NODES=2" not in runner


def test_full_chain_forwards_profile_and_readiness_to_every_wave() -> None:
    """All dependent stages use the same immutable cluster placement receipt."""
    chain = (_LAUNCHER_DIR / "common/specdec/submit_drafter_full_chain.sh").read_text()

    for required in (
        "--cluster-profile",
        "--readiness-receipt",
        '--cluster-profile "$CLUSTER_PROFILE"',
        '--readiness-receipt "$READINESS_RECEIPT"',
        'CHAIN_CLUSTER_NAME="oci-hsg"',
        'tuple_identity="${CHAIN_CLUSTER_NAME}:${identity}:${boundary}"',
    ):
        assert required in chain


def test_full_chain_exact_experiment_filter_selects_only_requested_chains() -> None:
    """Three exact experiment IDs render only their nine cumulative stages."""
    base = replace(_experiment(), cumulative_max_steps=(4166, 14500, 25391))
    experiments = (
        base,
        replace(base, target="q30-thinking"),
        replace(base, dataset="nemo-direct"),
        replace(base, method="dspark"),
    )
    selected_ids = {experiment.experiment_id for experiment in experiments[:3]}

    selected = select_experiments(experiments, selected_ids)

    assert {experiment.experiment_id for experiment in selected} == selected_ids
    assert sum(len(experiment.cumulative_max_steps) for experiment in selected) == 9
    assert experiments[3].experiment_id not in selected_ids

    chain = (_LAUNCHER_DIR / "common/specdec/submit_drafter_full_chain.sh").read_text()
    assert "--experiment-id" in chain
    assert "selected_ids = set(sys.argv[2:])" in chain


def test_training_runner_relocates_runtime_and_stages_only_role_inputs() -> None:
    """The pinned runtime and mounts remain valid after per-node extraction."""
    runner = (_LAUNCHER_DIR / "common/specdec/run_drafter_training.sbatch").read_text()

    for required in (
        "VIRTUAL_ENV",
        "sed -i",
        "PYTHONPATH",
        "--no-container-mount-home",
        "${OUTPUT_ROOT}:${OUTPUT_ROOT}",
        "SLURM_NODEID >= SERVE_NODES",
        "WANDB_PROJECT",
        "WANDB_RUN_GROUP",
        "WANDB_CACHE_DIR",
        "report_to=wandb",
        'TRAINING_RUN_NAME="$RUN_NAME"',
        'training.run_name="${TRAINING_RUN_NAME}-s${MAX_STEPS}"',
        "SERVE_BLOCK_SIZE=32",
        "SERVE_MAX_MODEL_LEN=8192",
        "SERVE_READY_TIMEOUT=1800",
        "HS_POOL_SLOTS=32",
        "SERVE_MAX_NUM_SEQS=16",
        'training.save_steps="${SAVE_STEPS}"',
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
    assert "for name in MANIFEST_PATH EXPERIMENT_INDEX MAX_STEPS LAUNCHER_ROOT" in runner
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
    assert 'cd "$node_root/source/tools/launcher"' in runner
    assert 'cd "$node_root/source"' not in runner
    assert "exec bash common/eagle3/train_eagle_streaming.sh" in runner
    assert "exec bash tools/launcher/common/eagle3/train_eagle_streaming.sh" not in runner


def test_host_staging_preserves_one_bounded_diagnostic_log_per_node() -> None:
    """A failed staging command must remain attributable without a shared log funnel."""
    runner = (_LAUNCHER_DIR / "common/specdec/run_drafter_training.sbatch").read_text()

    assert 'STAGE_LOG_ROOT="${OUTPUT_ROOT}/logs/stage-${SLURM_JOB_ID}"' in runner
    assert 'mkdir -p "$STAGE_LOG_ROOT"' in runner
    assert 'exec >"${STAGE_LOG_ROOT}/node-${SLURM_NODEID}.log" 2>&1' in runner
    assert "set -x" in runner


def test_requeued_host_staging_reuses_only_a_completed_node_local_copy() -> None:
    """Same-job-ID restart cannot nest or merge staged source/model/dataset trees."""
    runner = (_LAUNCHER_DIR / "common/specdec/run_drafter_training.sbatch").read_text()

    for required in (
        'stage_marker="$node_root/staging.complete"',
        'stage_fingerprint="${SOURCE_SHA}|${RUNTIME_ARCHIVE_SHA256}|${TARGET_PATH}|${DATASET_PATH}|${stage_role}"',
        'if [[ -f "$stage_marker" && "$(<"$stage_marker")" == "$stage_fingerprint" ]]',
        '[[ "$node_root" == "${SCRATCH_ROOT%/}/${SLURM_JOB_ID}/node-${SLURM_NODEID}" ]]',
        'rm -rf -- "$node_root"',
        'printf "%s\\n" "$stage_fingerprint" >"$stage_marker_tmp"',
        'mv "$stage_marker_tmp" "$stage_marker"',
    ):
        assert required in runner


def test_missing_optional_target_sidecar_does_not_fail_host_staging() -> None:
    """Every role gets the complete target checkpoint without optional-sidecar branching."""
    runner = (_LAUNCHER_DIR / "common/specdec/run_drafter_training.sbatch").read_text()

    target_copy = 'cp -aL "$TARGET_PATH" "$node_root/input/target"'
    assert runner.count(target_copy) == 1
    assert '[[ -e "$TARGET_PATH/$sidecar" ]] && cp' not in runner
    assert "for sidecar in config.json" not in runner
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

    assert "${SCRATCH_ROOT}:${SCRATCH_ROOT}" in runner
    assert "${SCRATCH_JOB_ROOT}:${SCRATCH_JOB_ROOT}" not in runner
    assert "/raid/scratch:/raid/scratch" in probe


def test_streaming_serve_logs_stay_on_node_local_scratch() -> None:
    """Shared scratchspace carries only the few cross-node rendezvous files."""
    runner = (_LAUNCHER_DIR / "common/specdec/run_drafter_training.sbatch").read_text()
    streaming = (_LAUNCHER_DIR / "common/eagle3/train_eagle_streaming.sh").read_text()

    assert 'SERVE_LOG_DIR="${SCRATCH_JOB_ROOT}/node-${SLURM_NODEID}/logs"' in runner
    assert "${SERVE_LOG_DIR:-/scratchspace}/vllm_serve.${NODEID}.${replica}.log" in streaming


def test_cumulative_runner_writes_a_checkpoint_for_every_resume_boundary() -> None:
    """Each same-output cumulative wave leaves get_last_checkpoint() an exact resume."""
    runner = (_LAUNCHER_DIR / "common/specdec/run_drafter_training.sbatch").read_text()

    assert 'training.output_dir="${OUTPUT_ROOT}"' in runner
    assert 'SAVE_STEPS="${SAVE_STEPS:-$MAX_STEPS}"' in runner
    assert 'training.save_steps="${SAVE_STEPS}"' in runner
    assert "training.save_total_limit=2" in runner


def test_self_requeue_is_opt_in_and_configures_slurm_signal_delivery() -> None:
    """Production waves change lifecycle only when explicitly requested."""
    submitter = (_LAUNCHER_DIR / "common/specdec/submit_drafter_training_wave.sh").read_text()

    assert "DEFAULT_REQUEUE_SAVE_STEPS=50" in submitter
    assert "MAX_REQUEUES=50" in submitter
    assert 'elif [[ "$SELF_REQUEUE" -eq 1 ]]' in submitter
    assert 'save_steps="$DEFAULT_REQUEUE_SAVE_STEPS"' in submitter
    assert 'save_steps="$boundary"' in submitter
    for required in (
        "--self-requeue",
        "--save-steps",
        "--max-requeues",
        "--requeue-signal-lead",
        "--requeue",
        '"--signal=B:USR1@${REQUEUE_SIGNAL_LEAD}"',
        "SELF_REQUEUE=${SELF_REQUEUE}",
        "SAVE_STEPS=${save_steps}",
        "MAX_REQUEUES=${MAX_REQUEUES}",
        "WANDB_RUN_ID=${wandb_run_id}",
    ):
        assert required in submitter


def test_full_convergence_boundaries_keep_every_family_below_restart_budget() -> None:
    """Every target family uses the approved bounded stage schedule."""
    assert full_convergence_boundaries("qwen3-30b-a3b", 8) == (4166, 14500, 25391)
    assert full_convergence_boundaries("qwen3-30b-a3b", 16) == (4166, 14500, 25391)
    assert full_convergence_boundaries("qwen3-235b-a22b", 8) == (4166, 14500, 25391)
    assert full_convergence_boundaries("qwen3-235b-a22b", 16) == (
        4166,
        10500,
        17000,
        23500,
        25391,
    )


def test_full_chain_job_names_are_readable_unique_and_bounded() -> None:
    """Readable scheduler names encode every dimension without collisions."""
    experiments = [
        readable_job_name(target, dataset, method, block_size, 25391)
        for target in ("q30-base", "q30-thinking", "q235-base", "q235-thinking")
        for dataset in ("opb-direct", "nemo-direct")
        for method in ("dflash", "dspark")
        for block_size in (8, 16)
    ]
    assert len(experiments) == len(set(experiments)) == 32
    assert max(map(len, experiments)) <= 64
    assert "q235t" in readable_job_name("q235-thinking", "nemo-direct", "dspark", 16, 4166)


def test_full_chain_submitter_presubmits_safe_afterok_stages() -> None:
    """The orchestrator pins lifecycle settings and dependency submission."""
    chain = (_LAUNCHER_DIR / "common/specdec/submit_drafter_full_chain.sh").read_text()
    wave = (_LAUNCHER_DIR / "common/specdec/submit_drafter_training_wave.sh").read_text()

    for required in (
        "squeue -h -u",
        "sacct -X -n -P -u",
        "SCHEDULER_JOBS_SNAPSHOT",
        "--dependency",
        "--save-steps 50",
        "--self-requeue",
        "--max-requeues 50",
        "--experiment-index",
        "--job-name",
    ):
        assert required in chain
    assert 'wandb_run_id="sd-${identity}"' in wave


def test_self_requeue_runner_uses_checkpoint_gated_lifecycle() -> None:
    """The runner resumes one W&B identity and requeues only through the tested helper."""
    runner = (_LAUNCHER_DIR / "common/specdec/run_drafter_training.sbatch").read_text()

    for required in (
        'SAVE_STEPS="${SAVE_STEPS:-$MAX_STEPS}"',
        'training.save_steps="${SAVE_STEPS}"',
        'export WANDB_RUN_ID="${WANDB_RUN_ID}"',
        "export WANDB_RESUME=allow",
        'source "$LAUNCHER_ROOT/common/specdec/drafter_requeue_lifecycle.sh"',
        "drafter_requeue_init",
        "drafter_run_requeueable_step",
        "drafter_mark_training_complete",
    ):
        assert required in runner
    assert runner.index("drafter_requeue_init") < runner.index("host staging complete")
    assert runner.count("drafter_run_requeueable_step") == 2


def test_training_runner_preserves_proven_production_training_semantics() -> None:
    """Production waves must not inherit incompatible generic recipe defaults."""
    runner = (_LAUNCHER_DIR / "common/specdec/run_drafter_training.sbatch").read_text()

    for required in (
        "training.training_seq_len=4096",
        "training.answer_only_loss=false",
        "training.seed=42",
        'mkdir -p "$OUTPUT_ROOT"',
        'SERVE_MODEL_NAME="modelopt-${RUN_NAME}"',
    ):
        assert required in runner

    streaming = (_LAUNCHER_DIR / "common/eagle3/train_eagle_streaming.sh").read_text()
    assert 'SERVED_MODEL_NAME="${SERVE_MODEL_NAME:-$HF_MODEL_CKPT}"' in streaming
    assert '--served-model-name "$SERVED_MODEL_NAME"' in streaming
    assert 'data.streaming_model_name="$SERVED_MODEL_NAME"' in streaming


def test_training_runner_fills_every_serve_gpu_with_tp_replicas() -> None:
    """Q30 TP2 uses two replicas per four-GPU serve node; Q235 TP4 stays at one."""
    runner = (_LAUNCHER_DIR / "common/specdec/run_drafter_training.sbatch").read_text()

    for required in (
        "GPUS_PER_NODE % SERVE_TP",
        "SERVE_REPLICAS_PER_NODE=$((GPUS_PER_NODE / SERVE_TP))",
        "SERVE_REPLICAS_PER_NODE * SERVE_TP == GPUS_PER_NODE",
        "export SERVE_REPLICAS_PER_NODE",
    ):
        assert required in runner


def test_streaming_serve_replicas_have_disjoint_devices_and_ports() -> None:
    """Every local replica owns one TP-sized GPU slice and unique API/sidecar ports."""
    streaming = (_LAUNCHER_DIR / "common/eagle3/train_eagle_streaming.sh").read_text()

    for required in (
        "SERVE_REPLICAS_PER_NODE",
        "replica * SERVE_TP",
        "replica_api_port=$((SERVE_PORT + replica))",
        "replica_sidecar_port=$((HS_SIDECAR_PORT + replica))",
        'launch_vllm "0.0.0.0" "$SERVE_TP" "$replica_cvd"',
        "vllm_serve.${NODEID}.${replica}.log",
    ):
        assert required in streaming


def test_streaming_rendezvous_publishes_and_checks_every_replica() -> None:
    """Trainers must not start until every per-node replica endpoint is healthy."""
    streaming = (_LAUNCHER_DIR / "common/eagle3/train_eagle_streaming.sh").read_text()

    for required in (
        "${SERVE_ADDR_FILE}.${NODEID}.${replica}",
        'mv "$addr_tmp" "$addr_file"',
        "for ((replica = 0; replica < SERVE_REPLICAS_PER_NODE; replica++))",
        "${SERVE_ADDR_FILE}.${s}.${replica}",
        'wait_vllm_ready "$surl"',
    ):
        assert required in streaming


def test_requeued_runner_clears_exact_persistent_rendezvous_before_launch() -> None:
    """Trainers cannot consume stale same-job-ID addresses after a new allocation starts."""
    runner = (_LAUNCHER_DIR / "common/specdec/run_drafter_training.sbatch").read_text()

    for required in (
        'rm -f "$CONTROL_ROOT/.training_done.${SLURM_JOB_ID}"',
        'rm -f "$CONTROL_ROOT/.trainer_addr.${SLURM_JOB_ID}"',
        'rm -f "$CONTROL_ROOT/.serve_addr.${SLURM_JOB_ID}.${serve_node}.${replica}"',
    ):
        assert required in runner
    assert runner.index(".serve_addr.${SLURM_JOB_ID}") < runner.index("training_step=(srun")


def test_streaming_serve_supervisor_cleans_up_and_fails_on_child_death() -> None:
    """A replica failure must fail the serve task and terminate all sibling replicas."""
    streaming = (_LAUNCHER_DIR / "common/eagle3/train_eagle_streaming.sh").read_text()

    assert "SERVE_PIDS=()" in streaming
    assert 'for pid in "${SERVE_PIDS[@]}"' in streaming
    assert 'if ! kill -0 "$pid"' in streaming
    assert "ERROR: vllm serve replica" in streaming


def test_training_step_terminates_all_node_tasks_when_a_replica_dies() -> None:
    """A nonzero serve task must stop trainers before they use a partial endpoint set."""
    runner = (_LAUNCHER_DIR / "common/specdec/run_drafter_training.sbatch").read_text()

    execution_srun = next(line for line in runner.splitlines() if "--container-image" in line)
    assert "--kill-on-bad-exit=1" in execution_srun


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

    assert (
        'srun --nodes="$ALLOCATED_NODES" --ntasks="$ALLOCATED_NODES" --ntasks-per-node=1 bash -c \'\n'
        in runner
    )
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
