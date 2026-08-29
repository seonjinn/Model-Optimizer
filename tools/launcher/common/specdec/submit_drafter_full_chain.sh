#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Pre-submit the complete 32-run paper-scale matrix as bounded afterok chains.
set -euo pipefail

# Shared parallel storage is mounted at /lustre on OCI-HSG, Ptyche and Lyris
# and at /scratch/fsw on AWS-CMH and OCI-AGA, so requiring one prefix refuses
# to run on two of the five clusters. Name the storage a durable artifact must
# NOT live on instead -- node-local scratch that vanishes with the job, and the
# NFS home the MARS guidance reserves for source.
is_durable_path() {
    case "${1:-}" in
        /home/*|/raid/*|/tmp/*|/var/*|/cm/*|/dev/shm/*) return 1 ;;
        /*) return 0 ;;
        *) return 1 ;;
    esac
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LAUNCHER_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
WAVE_SUBMITTER="${SCRIPT_DIR}/submit_drafter_training_wave.sh"
MANIFEST=""
RECEIPT=""
CLUSTER_PROFILE=""
READINESS_RECEIPT=""
DRY_RUN=0
TEST_DEPENDENCY_JOB=""
LEGACY_SOURCE_SHA=""
SEED_Q30_FROM_LEGACY=0
EXPERIMENT_IDS=()

usage() {
    echo "usage: $0 --manifest /home/.../manifest.json --receipt /lustre/.../submission.jsonl [--experiment-id ID ...] [--cluster-profile /home/.../profile.yaml --readiness-receipt /lustre/.../profile-probe.json] [--seed-q30-from-legacy --legacy-source-sha SHA] [--dry-run --test-dependency-job COMPLETED_JOBID]" >&2
    exit 2
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --manifest) MANIFEST="$2"; shift 2 ;;
        --receipt) RECEIPT="$2"; shift 2 ;;
        --cluster-profile) CLUSTER_PROFILE="$2"; shift 2 ;;
        --readiness-receipt) READINESS_RECEIPT="$2"; shift 2 ;;
        --dry-run) DRY_RUN=1; shift ;;
        --test-dependency-job) TEST_DEPENDENCY_JOB="$2"; shift 2 ;;
        --legacy-source-sha) LEGACY_SOURCE_SHA="$2"; shift 2 ;;
        --seed-q30-from-legacy) SEED_Q30_FROM_LEGACY=1; shift ;;
        --experiment-id) EXPERIMENT_IDS+=("$2"); shift 2 ;;
        *) usage ;;
    esac
done
[[ "$MANIFEST" == /home/* && -f "$MANIFEST" ]] && is_durable_path "$RECEIPT" || usage
[[ -z "$TEST_DEPENDENCY_JOB" || "$TEST_DEPENDENCY_JOB" =~ ^[1-9][0-9]*$ ]] || usage
[[ -z "$LEGACY_SOURCE_SHA" || "$LEGACY_SOURCE_SHA" =~ ^[0-9a-f]{40}$ ]] || usage
[[ "$DRY_RUN" -eq 0 || -n "$TEST_DEPENDENCY_JOB" ]] || usage
[[ "$SEED_Q30_FROM_LEGACY" -eq 0 || -n "$LEGACY_SOURCE_SHA" ]] || usage

CHAIN_CLUSTER_NAME="oci-hsg"
if [[ -n "$CLUSTER_PROFILE" ]]; then
    [[ "$CLUSTER_PROFILE" == /home/* && -f "$CLUSTER_PROFILE" ]] || usage
    CHAIN_CLUSTER_NAME="$(PYTHONPATH="$LAUNCHER_ROOT${PYTHONPATH:+:$PYTHONPATH}" python3 - "$CLUSTER_PROFILE" <<'PY'
import sys
from pathlib import Path

from common.specdec.cluster_profile import load_cluster_profile

print(load_cluster_profile(Path(sys.argv[1])).name)
PY
)"
fi

mkdir -p "$(dirname "$RECEIPT")"
scheduler_jobs="$({
    squeue -h -u "$USER" -o "%j|%A|%k|%T"
    sacct -X -n -P -u "$USER" -S today --format=JobName%64,JobIDRaw,Comment,State
} || true)"

render_plan() {
    PYTHONPATH="$LAUNCHER_ROOT${PYTHONPATH:+:$PYTHONPATH}" python3 - "$MANIFEST" "$SEED_Q30_FROM_LEGACY" "${EXPERIMENT_IDS[@]}" <<'PY'
import sys
from pathlib import Path

from common.specdec.build_drafter_full_manifest import (
    legacy_q30_seed_expectations,
    readable_job_name,
    select_experiments,
)
from common.specdec.drafter_job_manifest import load_manifest

experiments = load_manifest(Path(sys.argv[1]))
if len(experiments) != 32:
    raise SystemExit("full chain manifest must contain exactly 32 experiments")
seed_q30_from_legacy = bool(int(sys.argv[2]))
selected_ids = set(sys.argv[3:])
selected = select_experiments(experiments, selected_ids)
selected_by_id = {experiment.experiment_id for experiment in selected}
for index, experiment in enumerate(experiments):
    if experiment.experiment_id not in selected_by_id:
        continue
    for boundary in experiment.cumulative_max_steps:
        legacy_id = legacy_fingerprint = selected_tuple = ""
        if seed_q30_from_legacy and experiment.topology.target_kind == "qwen3-30b-a3b":
            legacy_id, legacy_fingerprint, selected_tuple = legacy_q30_seed_expectations(
                experiment
            )
        expected_rng_states = (experiment.slurm.nodes // 2) * experiment.slurm.gpus_per_node
        print(
            "\t".join(
                (
                    str(index),
                    experiment.experiment_id,
                    str(boundary),
                    readable_job_name(
                        experiment.target,
                        experiment.dataset,
                        experiment.method,
                        experiment.block_size,
                        boundary,
                        nodes=experiment.slurm.nodes,
                    ),
                    experiment.paths.output_root,
                    experiment.topology.target_kind,
                    str(expected_rng_states),
                    legacy_id,
                    legacy_fingerprint,
                    selected_tuple,
                )
            )
        )
PY
}

verify_seeded_output() {
    local output_root="$1" legacy_output_root="$2" source_sha="$3"
    local expected_legacy_id="$4" expected_legacy_fingerprint="$5" selected_tuple="$6"
    PYTHONPATH="$LAUNCHER_ROOT${PYTHONPATH:+:$PYTHONPATH}" python3 - "$output_root" "$legacy_output_root" "$source_sha" "$expected_legacy_id" "$expected_legacy_fingerprint" "$selected_tuple" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

output_root = Path(sys.argv[1])
legacy_output_root = Path(sys.argv[2])
source_sha = sys.argv[3]
expected_legacy_id = sys.argv[4]
expected_legacy_fingerprint = sys.argv[5]
selected_tuple = sys.argv[6]
provenance_path = output_root / "control/checkpoint-seed.json"
legacy_identity_path = legacy_output_root / "control/training-identity.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def regular_file_hashes(root: Path) -> dict[str, str]:
    return {
        str(path.relative_to(root)): sha256(path)
        for path in sorted(root.rglob("*"))
        if path.is_file() and not path.is_symlink()
    }


try:
    provenance = json.loads(provenance_path.read_text())
    legacy_identity = json.loads(legacy_identity_path.read_text())
except (OSError, json.JSONDecodeError) as error:
    raise SystemExit(f"invalid Q30 seed provenance: {error}") from error
from common.specdec.build_drafter_full_manifest import validate_legacy_seed_identity

try:
    validate_legacy_seed_identity(
        legacy_identity, expected_legacy_id, expected_legacy_fingerprint, source_sha
    )
except ValueError as error:
    raise SystemExit(str(error)) from error
identity_source_sha = legacy_identity.get(
    "adopted_from_source_sha", legacy_identity.get("source_sha")
)
if identity_source_sha != source_sha:
    raise SystemExit("legacy Q30 identity source SHA mismatch")
source_checkpoint = Path(provenance.get("source_checkpoint", ""))
destination = output_root / source_checkpoint.name
expected = {
    "fingerprint_schema": "topology-v2",
    "source_output_root": str(legacy_output_root),
    "source_sha": identity_source_sha,
    "source_identity_sha256": sha256(legacy_identity_path),
    "source_experiment_id": expected_legacy_id,
    "source_training_fingerprint": expected_legacy_fingerprint,
    "selected_experiment_tuple": selected_tuple,
}
if any(provenance.get(key) != value for key, value in expected.items()):
    raise SystemExit("Q30 checkpoint seed provenance mismatch")
if source_checkpoint.parent != legacy_output_root or not destination.is_dir():
    raise SystemExit("Q30 checkpoint seed path mismatch")
checkpoint_hashes = provenance.get("checkpoint_files_sha256")
if not checkpoint_hashes or regular_file_hashes(destination) != checkpoint_hashes:
    raise SystemExit("Q30 checkpoint seed checksum mismatch")
if source_checkpoint.is_dir() and regular_file_hashes(source_checkpoint) != checkpoint_hashes:
    raise SystemExit("legacy Q30 source checkpoint changed after seeding")
PY
    EXPECTED_RNG_STATES=4 bash "${SCRIPT_DIR}/drafter_requeue_lifecycle.sh" latest-complete "$output_root" >/dev/null
}

seed_q30_checkpoint() {
    local output_root="$1" source_sha="$2" expected_legacy_id="$3"
    local expected_legacy_fingerprint="$4" selected_tuple="$5"
    local legacy_output_root="${output_root%-2n}"
    [[ "$legacy_output_root" != "$output_root" ]] || {
        echo "Q30 seed output must use the -2n namespace: $output_root" >&2
        return 2
    }
    if [[ -e "$output_root" ]]; then
        verify_seeded_output "$output_root" "$legacy_output_root" "$source_sha" "$expected_legacy_id" "$expected_legacy_fingerprint" "$selected_tuple"
        return 0
    fi
    local checkpoint_record source_checkpoint checkpoint_step
    checkpoint_record="$(EXPECTED_RNG_STATES=8 bash "${SCRIPT_DIR}/drafter_requeue_lifecycle.sh" latest-complete "$legacy_output_root")" || {
        echo "legacy Q30 output has no complete world-eight checkpoint: $legacy_output_root" >&2
        return 1
    }
    IFS=$'\t' read -r source_checkpoint checkpoint_step <<<"$checkpoint_record"
    PYTHONPATH="$LAUNCHER_ROOT${PYTHONPATH:+:$PYTHONPATH}" python3 - "$source_checkpoint" "$legacy_output_root" "$output_root" "$checkpoint_step" "$source_sha" "$expected_legacy_id" "$expected_legacy_fingerprint" "$selected_tuple" <<'PY'
import errno
import hashlib
import json
import os
import shutil
import sys
from pathlib import Path

source_checkpoint = Path(sys.argv[1])
legacy_output_root = Path(sys.argv[2])
output_root = Path(sys.argv[3])
checkpoint_step = int(sys.argv[4])
source_sha = sys.argv[5]
expected_legacy_id = sys.argv[6]
expected_legacy_fingerprint = sys.argv[7]
selected_tuple = sys.argv[8]
temporary = Path(f"{output_root}.seed-partial-{os.getpid()}")
legacy_identity_path = legacy_output_root / "control/training-identity.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def regular_file_hashes(root: Path) -> dict[str, str]:
    return {
        str(path.relative_to(root)): sha256(path)
        for path in sorted(root.rglob("*"))
        if path.is_file() and not path.is_symlink()
    }


try:
    legacy_identity = json.loads(legacy_identity_path.read_text())
except (OSError, json.JSONDecodeError) as error:
    raise SystemExit(f"invalid legacy Q30 training identity: {error}") from error
from common.specdec.build_drafter_full_manifest import validate_legacy_seed_identity

try:
    validate_legacy_seed_identity(
        legacy_identity, expected_legacy_id, expected_legacy_fingerprint, source_sha
    )
except ValueError as error:
    raise SystemExit(str(error)) from error
identity_source_sha = legacy_identity.get(
    "adopted_from_source_sha", legacy_identity.get("source_sha")
)
if identity_source_sha != source_sha:
    raise SystemExit("legacy Q30 identity source SHA mismatch")
if output_root.exists() or temporary.exists():
    raise SystemExit("checkpoint seed destination already exists")
temporary.mkdir(parents=True)
destination = temporary / source_checkpoint.name
storage: dict[str, str] = {}
try:
    for source in sorted(source_checkpoint.rglob("*")):
        relative = source.relative_to(source_checkpoint)
        target = destination / relative
        if source.is_symlink():
            target.parent.mkdir(parents=True, exist_ok=True)
            os.symlink(os.readlink(source), target)
        elif source.is_dir():
            target.mkdir(parents=True, exist_ok=True)
        elif source.is_file():
            target.parent.mkdir(parents=True, exist_ok=True)
            try:
                os.link(source, target)
                storage[str(relative)] = "hardlink"
            except OSError as error:
                if error.errno not in {errno.EXDEV, errno.EPERM, errno.EMLINK}:
                    raise
                shutil.copy2(source, target)
                if sha256(source) != sha256(target):
                    raise RuntimeError(f"checkpoint copy integrity mismatch: {source}")
                storage[str(relative)] = "copy"
    state = json.loads((destination / "trainer_state.json").read_text())
    if int(state["global_step"]) != checkpoint_step:
        raise RuntimeError("seed checkpoint step mismatch")
    required = ("optimizer.pt", "scheduler.pt", *(f"rng_state_{rank}.pth" for rank in range(4)))
    if any(not (destination / name).is_file() or (destination / name).stat().st_size == 0 for name in required):
        raise RuntimeError("seed checkpoint is not resumable by the world-four trainer")
    weight_names = (
        "model.safetensors",
        "model.safetensors.index.json",
        "pytorch_model.bin",
        "pytorch_model.bin.index.json",
    )
    if not any((destination / name).is_file() and (destination / name).stat().st_size > 0 for name in weight_names):
        raise RuntimeError("seed checkpoint has no model weights")
    control = temporary / "control"
    control.mkdir()
    provenance = {
        "checkpoint_files_sha256": regular_file_hashes(destination),
        "checkpoint_step": checkpoint_step,
        "fingerprint_schema": "topology-v2",
        "source_checkpoint": str(source_checkpoint),
        "source_identity_sha256": sha256(legacy_identity_path),
        "source_experiment_id": expected_legacy_id,
        "source_output_root": str(legacy_output_root),
        "source_sha": identity_source_sha,
        "source_training_fingerprint": expected_legacy_fingerprint,
        "selected_experiment_tuple": selected_tuple,
        "storage": storage,
    }
    (control / "checkpoint-seed.json").write_text(json.dumps(provenance, sort_keys=True) + "\n")
    os.rename(temporary, output_root)
except BaseException:
    shutil.rmtree(temporary, ignore_errors=True)
    raise
PY
    verify_seeded_output "$output_root" "$legacy_output_root" "$source_sha" "$expected_legacy_id" "$expected_legacy_fingerprint" "$selected_tuple"
}

previous_index=""
previous_job=""
legacy_step=""
while IFS=$'\t' read -r index identity boundary job_name output_root target_kind expected_rng_states expected_legacy_id expected_legacy_fingerprint selected_tuple; do
    if [[ "$index" != "$previous_index" ]]; then
        previous_index="$index"
        previous_job=""
        legacy_step=""
        if [[ "$target_kind" == "qwen3-30b-a3b" && "$SEED_Q30_FROM_LEGACY" -eq 1 && "$DRY_RUN" -eq 0 && ! -f "$output_root/control/training-identity.json" ]]; then
            seed_q30_checkpoint "$output_root" "$LEGACY_SOURCE_SHA" "$expected_legacy_id" "$expected_legacy_fingerprint" "$selected_tuple"
        fi
        if [[ ! -f "$output_root/control/training-identity.json" ]]; then
            checkpoint_record="$(EXPECTED_RNG_STATES="$expected_rng_states" bash "${SCRIPT_DIR}/drafter_requeue_lifecycle.sh" latest-complete "$output_root" 2>/dev/null || true)"
            if [[ -n "$checkpoint_record" ]]; then
                [[ -n "$LEGACY_SOURCE_SHA" ]] || {
                    echo "legacy checkpoint requires --legacy-source-sha: $output_root" >&2
                    exit 1
                }
                legacy_step="${checkpoint_record##*$'\t'}"
            elif compgen -G "$output_root/checkpoint-*" >/dev/null; then
                echo "legacy output has no complete resumable checkpoint: $output_root" >&2
                exit 1
            fi
        fi
    fi
    args=(
        --manifest "$MANIFEST"
        --receipt "$RECEIPT"
        --experiment-index "$index"
        --max-steps "$boundary"
        --job-name "$job_name"
        --save-steps 50
        --self-requeue
        --max-requeues 50
        --requeue-signal-lead 300
    )
    if [[ -n "$CLUSTER_PROFILE" ]]; then
        args+=(--cluster-profile "$CLUSTER_PROFILE")
    fi
    if [[ -n "$READINESS_RECEIPT" ]]; then
        args+=(--readiness-receipt "$READINESS_RECEIPT")
    fi
    if [[ -n "$legacy_step" ]]; then
        args+=(
            --legacy-adoption-source-sha "$LEGACY_SOURCE_SHA"
            --legacy-adoption-checkpoint-step "$legacy_step"
        )
    fi
    if [[ "$DRY_RUN" -eq 1 ]]; then
        args+=(--dry-run)
        [[ -z "$previous_job" ]] || args+=(--dependency "$TEST_DEPENDENCY_JOB")
    elif [[ -n "$previous_job" ]]; then
        args+=(--dependency "$previous_job")
    fi
    SCHEDULER_JOBS_SNAPSHOT="$scheduler_jobs" bash "$WAVE_SUBMITTER" "${args[@]}"
    if [[ "$DRY_RUN" -eq 0 ]]; then
        tuple_identity="${identity}:${boundary}"
        if [[ "$CHAIN_CLUSTER_NAME" != "oci-hsg" ]]; then
            tuple_identity="${CHAIN_CLUSTER_NAME}:${identity}:${boundary}"
        fi
        previous_job="$(python3 - "$RECEIPT" "$tuple_identity" <<'PY'
import json
import sys
from pathlib import Path

for line in reversed(Path(sys.argv[1]).read_text().splitlines()):
    record = json.loads(line)
    if record.get("tuple_identity") == sys.argv[2] and record.get("job_id"):
        print(record["job_id"])
        break
PY
)"
        [[ -n "$previous_job" ]] || {
            echo "submission receipt lacks job ID for ${tuple_identity}" >&2
            exit 1
        }
    else
        previous_job="$TEST_DEPENDENCY_JOB"
    fi
done < <(render_plan)
