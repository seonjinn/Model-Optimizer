#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Render and submit unique four-node cumulative drafter training waves.
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
RUNNER="${SCRIPT_DIR}/run_drafter_training.sbatch"
DEFAULT_CLUSTER_PROFILE="${SCRIPT_DIR}/profiles/oci-hsg.yaml"
MANIFEST=""
RECEIPT=""
CLUSTER_PROFILE="$DEFAULT_CLUSTER_PROFILE"
READINESS_RECEIPT=""
DEPENDENCY=""
ONLY_STEP=""
EXPERIMENT_INDEX=""
JOB_NAME_OVERRIDE=""
LEGACY_ADOPTION_SOURCE_SHA=""
LEGACY_ADOPTION_CHECKPOINT_STEP=""
SAVE_STEPS=""
DEFAULT_REQUEUE_SAVE_STEPS=50
SELF_REQUEUE=0
MAX_REQUEUES=50
REQUEUE_SIGNAL_LEAD=300
DRY_RUN=0

usage() {
    echo "usage: $0 --manifest /home/.../manifest.json --receipt /lustre/.../receipt.jsonl [--cluster-profile /home/.../profile.yaml] [--readiness-receipt /lustre/.../profile-probe.json] [--dependency JOBID] [--max-steps N] [--save-steps N] [--self-requeue] [--max-requeues N] [--requeue-signal-lead SECONDS] [--experiment-index N] [--job-name NAME] [--dry-run]" >&2
    exit 2
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --manifest) MANIFEST="$2"; shift 2 ;;
        --receipt) RECEIPT="$2"; shift 2 ;;
        --cluster-profile) CLUSTER_PROFILE="$2"; shift 2 ;;
        --readiness-receipt) READINESS_RECEIPT="$2"; shift 2 ;;
        --dependency) DEPENDENCY="$2"; shift 2 ;;
        --max-steps) ONLY_STEP="$2"; shift 2 ;;
        --save-steps) SAVE_STEPS="$2"; shift 2 ;;
        --self-requeue) SELF_REQUEUE=1; shift ;;
        --max-requeues) MAX_REQUEUES="$2"; shift 2 ;;
        --requeue-signal-lead) REQUEUE_SIGNAL_LEAD="$2"; shift 2 ;;
        --experiment-index) EXPERIMENT_INDEX="$2"; shift 2 ;;
        --job-name) JOB_NAME_OVERRIDE="$2"; shift 2 ;;
        --legacy-adoption-source-sha) LEGACY_ADOPTION_SOURCE_SHA="$2"; shift 2 ;;
        --legacy-adoption-checkpoint-step) LEGACY_ADOPTION_CHECKPOINT_STEP="$2"; shift 2 ;;
        --dry-run) DRY_RUN=1; shift ;;
        --) echo "extra ModelOpt dotlist arguments are not accepted for pinned production manifests" >&2; exit 2 ;;
        *) usage ;;
    esac
done
[[ "$MANIFEST" == /home/* && -f "$MANIFEST" ]] && is_durable_path "$RECEIPT" || usage
[[ "$RUNNER" == /home/* ]] || { echo "runner must be in /home source" >&2; exit 2; }
[[ "$CLUSTER_PROFILE" == /home/* && -f "$CLUSTER_PROFILE" ]] || usage
[[ -z "$ONLY_STEP" || "$ONLY_STEP" =~ ^[1-9][0-9]*$ ]] || usage
[[ -z "$SAVE_STEPS" || "$SAVE_STEPS" =~ ^[1-9][0-9]*$ ]] || usage
[[ "$MAX_REQUEUES" =~ ^[1-9][0-9]*$ ]] || usage
[[ "$REQUEUE_SIGNAL_LEAD" =~ ^[1-9][0-9]*$ ]] || usage
[[ -z "$EXPERIMENT_INDEX" || "$EXPERIMENT_INDEX" =~ ^[0-9]+$ ]] || usage
[[ -z "$JOB_NAME_OVERRIDE" || ( -n "$ONLY_STEP" && -n "$EXPERIMENT_INDEX" && ${#JOB_NAME_OVERRIDE} -le 64 && "$JOB_NAME_OVERRIDE" =~ ^[a-zA-Z0-9._-]+$ ) ]] || usage
[[ ( -z "$LEGACY_ADOPTION_SOURCE_SHA" && -z "$LEGACY_ADOPTION_CHECKPOINT_STEP" ) || ( "$LEGACY_ADOPTION_SOURCE_SHA" =~ ^[0-9a-f]{40}$ && "$LEGACY_ADOPTION_CHECKPOINT_STEP" =~ ^[1-9][0-9]*$ ) ]] || usage

CLUSTER_PROFILE_SHA256="$(sha256sum "$CLUSTER_PROFILE" | cut -d' ' -f1)"
READINESS_RECEIPT_SHA256=""
profile_data="$(PYTHONPATH="$LAUNCHER_ROOT${PYTHONPATH:+:$PYTHONPATH}" python3 - "$CLUSTER_PROFILE" "$READINESS_RECEIPT" "$MANIFEST" <<'PY'
import json
import sys
from pathlib import Path

from common.specdec.cluster_profile import (
    load_cluster_profile,
    scheduler_gpu_args,
    validate_scratch_root,
)
from common.specdec.drafter_job_manifest import load_manifest

profile = load_cluster_profile(Path(sys.argv[1]))
receipt_path = Path(sys.argv[2]) if sys.argv[2] else None
experiments = load_manifest(Path(sys.argv[3]))
if any(experiment.paths.source_sha != profile.modelopt_commit for experiment in experiments):
    raise SystemExit("manifest source SHA does not match the cluster profile")
if any(
    experiment.slurm.nodes > profile.training_nodes
    or profile.training_nodes % experiment.slurm.nodes
    or experiment.slurm.gpus_per_node != profile.gpus_per_node
    or experiment.slurm.segment > profile.training_segment
    for experiment in experiments
):
    raise SystemExit("manifest topology exceeds or cannot divide the cluster profile capacity")

if receipt_path is None:
    if profile.name != "oci-hsg":
        raise SystemExit("non-OCI training requires a readiness receipt")
    scratch = profile.scratch_candidates[0]
else:
    if not receipt_path.is_file() or not receipt_path.resolve().is_relative_to(
        profile.durable_root.resolve()
    ):
        raise SystemExit("readiness receipt must be under the profile durable root")
    receipt = json.loads(receipt_path.read_text())
    if receipt["profile"] != profile.name:
        raise SystemExit("readiness profile mismatch")
    if receipt.get("account") != profile.account or receipt.get("partition") != profile.partition:
        raise SystemExit("readiness scheduler settings mismatch")
    if (
        receipt.get("pyxis_available") is not True
        or receipt.get("gpu_count") != profile.gpus_per_node
        or receipt.get("architecture") != "aarch64"
    ):
        raise SystemExit("readiness hardware or Pyxis gate failed")
    scratch = Path(receipt["scratch_root"])
    validate_scratch_root(profile, scratch)
# Shared parallel storage is mounted at /lustre on OCI-HSG, Ptyche and Lyris and
# at /scratch/fsw on AWS-CMH and OCI-AGA. Naming only Lustre would let a shared
# path through as a scratch root on the two clusters that mount it elsewhere.
SHARED_STORAGE_ROOTS = (Path("/lustre"), Path("/scratch/fsw"))
if not scratch.is_absolute() or any(
    scratch.is_relative_to(root) for root in SHARED_STORAGE_ROOTS
):
    raise SystemExit("mutable training state cannot use shared parallel storage")

print(profile.name)
print(profile.account)
print(profile.partition)
print(profile.training_nodes)
print(profile.training_segment)
print(profile.walltime)
print(profile.gpus_per_node)
print("true" if scheduler_gpu_args(profile) else "false")
print(scratch)
PY
)"
mapfile -t profile_values <<<"$profile_data"
(( ${#profile_values[@]} == 9 )) || { echo "invalid cluster profile data" >&2; exit 2; }
CLUSTER_NAME="${profile_values[0]}"
PROFILE_ACCOUNT="${profile_values[1]}"
PROFILE_PARTITION="${profile_values[2]}"
SCRATCH_ROOT="${profile_values[8]}"
[[ "$CLUSTER_NAME" == "oci-hsg" || -n "$READINESS_RECEIPT" ]] || usage
if [[ -n "$READINESS_RECEIPT" ]]; then
    READINESS_RECEIPT_SHA256="$(sha256sum "$READINESS_RECEIPT" | cut -d' ' -f1)"
fi
render_waves() {
    PYTHONPATH="$LAUNCHER_ROOT${PYTHONPATH:+:$PYTHONPATH}" python3 - "$MANIFEST" "$ONLY_STEP" "$EXPERIMENT_INDEX" <<'PY'
import sys
from pathlib import Path
from common.specdec.drafter_job_manifest import load_manifest

selected = sys.argv[2]
selected_index = sys.argv[3]
for index, experiment in enumerate(load_manifest(Path(sys.argv[1]))):
    if selected_index and index != int(selected_index):
        continue
    for boundary in experiment.cumulative_max_steps:
        if selected and boundary != int(selected):
            continue
        print(
            f"{index}\t{experiment.experiment_id}\t{boundary}\t{experiment.run_name}"
            f"\t{experiment.slurm.account}\t{experiment.slurm.partition}"
            f"\t{experiment.slurm.nodes}\t{experiment.slurm.segment}"
            f"\t{experiment.slurm.gpus_per_node}\t{experiment.paths.output_root}"
        )
PY
}

declare -A identities=()
if [[ -n "${SCHEDULER_JOBS_SNAPSHOT+x}" ]]; then
    scheduler_jobs="$SCHEDULER_JOBS_SNAPSHOT"
else
    scheduler_jobs="$(
        {
            squeue -h -u "$USER" -o "%j|%A|%k|%T"
            sacct -X -n -P -u "$USER" -S today --format=JobName%64,JobIDRaw,Comment,State
        } || true
    )"
fi
while IFS=$'\t' read -r index identity boundary run_name _manifest_account _manifest_partition nodes segment gpus_per_node output_root; do
    [[ -n "$index" ]] || continue
    tuple_identity="${identity}:${boundary}"
    [[ -z "${identities[$tuple_identity]:-}" ]] || { echo "duplicate training tuple: $tuple_identity" >&2; exit 2; }
    identities[$tuple_identity]=1
    job_name="${JOB_NAME_OVERRIDE:-${run_name}-s${boundary}}"
    # An override that names a different arm than the manifest row silently
    # trains the wrong experiment under a convincing name.
    [[ -z "$JOB_NAME_OVERRIDE" || "$job_name" == *"$run_name"* ]] || {
        echo "job name $job_name does not name the run it would train: $run_name" >&2
        exit 2
    }
    cluster_tuple_identity="${CLUSTER_NAME}:${identity}:${boundary}"
    tuple_identity="$cluster_tuple_identity"
    if [[ "$CLUSTER_NAME" == "oci-hsg" ]]; then
        tuple_identity="${identity}:${boundary}"
    else
        suffix="-${CLUSTER_NAME}"
        job_name="${job_name:0:$((64 - ${#suffix}))}${suffix}"
    fi
    receipt_job="$(python3 - "$RECEIPT" "$tuple_identity" <<'PY'
import json, sys
from pathlib import Path
path = Path(sys.argv[1])
if path.exists():
    for line in reversed(path.read_text().splitlines()):
        record = json.loads(line)
        if record.get("tuple_identity") == sys.argv[2] and record.get("job_id"):
            print(record["job_id"])
            break
PY
)"
    if [[ -n "$receipt_job" ]]; then
        receipt_state="$(awk -F'|' -v job="$receipt_job" '$2 == job {print $4; exit}' <<<"$scheduler_jobs")"
        if [[ "$receipt_state" =~ ^(PENDING|RUNNING|COMPLETING|CONFIGURING|REQUEUED|COMPLETED)$ ]]; then
            printf '{"job_id":"%s","status":"receipt","tuple_identity":"%s"}\n' "$receipt_job" "$tuple_identity" >>"$RECEIPT"
            continue
        fi
        printf '{"job_id":"%s","status":"retrying-stale-receipt","tuple_identity":"%s","scheduler_state":"%s"}\n' "$receipt_job" "$tuple_identity" "$receipt_state" >>"$RECEIPT"
    fi
    existing="$(awk -F'|' -v tuple="$tuple_identity" '$3 == tuple && ($4 == "" || $4 ~ /^(PENDING|RUNNING|COMPLETING|CONFIGURING|REQUEUED|COMPLETED)$/) {print $2; exit}' <<<"$scheduler_jobs")"
    if [[ -n "$existing" ]]; then
        printf '{"job_id":"%s","status":"already-known","tuple_identity":"%s","max_steps":%s}\n' "$existing" "$tuple_identity" "$boundary" >>"$RECEIPT"
        continue
    fi
    if [[ -n "$SAVE_STEPS" ]]; then
        save_steps="$SAVE_STEPS"
    elif [[ "$SELF_REQUEUE" -eq 1 ]]; then
        save_steps="$DEFAULT_REQUEUE_SAVE_STEPS"
    else
        save_steps="$boundary"
    fi
    exports="ALL,SCHEDULER_JOBS_SNAPSHOT=,MANIFEST_PATH=${MANIFEST},EXPERIMENT_INDEX=${index},MAX_STEPS=${boundary},SAVE_STEPS=${save_steps},SELF_REQUEUE=${SELF_REQUEUE},LAUNCHER_ROOT=${LAUNCHER_ROOT},CLUSTER_NAME=${CLUSTER_NAME},CLUSTER_PROFILE_PATH=${CLUSTER_PROFILE},CLUSTER_PROFILE_SHA256=${CLUSTER_PROFILE_SHA256},READINESS_RECEIPT=${READINESS_RECEIPT},READINESS_RECEIPT_SHA256=${READINESS_RECEIPT_SHA256},SCRATCH_ROOT=${SCRATCH_ROOT}"
    if [[ -n "$LEGACY_ADOPTION_SOURCE_SHA" ]]; then
        exports+=",LEGACY_ADOPTION_SOURCE_SHA=${LEGACY_ADOPTION_SOURCE_SHA},LEGACY_ADOPTION_CHECKPOINT_STEP=${LEGACY_ADOPTION_CHECKPOINT_STEP}"
    fi
    requeue_args=()
    if [[ "$SELF_REQUEUE" -eq 1 ]]; then
        wandb_run_id="sd-${identity}"
        if [[ "$CLUSTER_NAME" != "oci-hsg" ]]; then
            wandb_run_id="sd-${identity}-${CLUSTER_NAME}"
        fi
        exports+=",MAX_REQUEUES=${MAX_REQUEUES},WANDB_RUN_ID=${wandb_run_id}"
        requeue_args=(--requeue "--signal=B:USR1@${REQUEUE_SIGNAL_LEAD}")
    fi
    account="$PROFILE_ACCOUNT"
    partition="$PROFILE_PARTITION"
    scheduler_args=(--account="$account" --partition="$partition" "--nodes=${nodes}" --ntasks-per-node=1 "--segment=${segment}" "--time=${profile_values[5]}")
    if [[ "${profile_values[7]}" == "true" ]]; then
        [[ "$gpus_per_node" == "${profile_values[6]}" ]] || { echo "manifest/profile GPU mismatch" >&2; exit 2; }
        scheduler_args+=("--gpus-per-node=${gpus_per_node}")
    fi
    args=("${scheduler_args[@]}" --job-name="$job_name" --comment="$tuple_identity" --export="$exports" --output="${output_root}/logs/slurm-%j.out" --error="${output_root}/logs/slurm-%j.err" "${requeue_args[@]}")
    echo "submitting index=$index run=$run_name experiment=$identity job-name=$job_name"
    [[ -z "$DEPENDENCY" ]] || args+=(--dependency=afterok:"$DEPENDENCY")
    sbatch --test-only "${args[@]}" "$RUNNER"
    if [[ "$DRY_RUN" -eq 1 ]]; then
        printf '{"cluster":"%s","job_name":"%s","status":"test-only","tuple_identity":"%s","max_steps":%s}\n' "$CLUSTER_NAME" "$job_name" "$tuple_identity" "$boundary" >>"$RECEIPT"
        continue
    fi
    mkdir -p "$output_root/logs"
    submitted="$(sbatch --parsable "${args[@]}" "$RUNNER" || true)"
    job_id="${submitted%%;*}"
    if [[ -z "$job_id" ]]; then
        job_id="$(squeue -h -n "$job_name" -o "%A|%k" | awk -F'|' -v tuple="$tuple_identity" '$2 == tuple {print $1; exit}' || true)"
        [[ -n "$job_id" ]] || job_id="$(sacct -X -n --name "$job_name" -P --format=JobIDRaw,Comment,State | awk -F'|' -v tuple="$tuple_identity" '$2 == tuple && $3 ~ /^(PENDING|RUNNING|COMPLETING|CONFIGURING|REQUEUED|COMPLETED)$/ {print $1; exit}' || true)"
    fi
    [[ -n "$job_id" ]] || { echo "scheduler did not confirm training submission: $job_name" >&2; exit 1; }
    printf '{"cluster":"%s","job_id":"%s","status":"submitted","tuple_identity":"%s","max_steps":%s,"job_name":"%s"}\n' "$CLUSTER_NAME" "$job_id" "$tuple_identity" "$boundary" "$job_name" >>"$RECEIPT"
done < <(render_waves)
