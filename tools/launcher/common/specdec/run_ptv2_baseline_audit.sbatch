#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=400G
#SBATCH --gpus-per-node=4
#SBATCH --time=00:45:00

set -euo pipefail

for name in \
    SOURCE_PATH SOURCE_SHA BASELINE_ROOT BASELINE_SOURCE_ROOT \
    BASELINE_IDENTITY_PATH BASELINE_IDENTITY_SHA256 \
    BASELINE_COMPLETION_PATH BASELINE_COMPLETION_SHA256 \
    BASELINE_SHA256_LIST_PATH BASELINE_SHA256_LIST_SHA256 \
    OUTPUT_PATH IMAGE_PATH RUNTIME_ARCHIVE RUNTIME_ARCHIVE_SHA256 SCRATCH_ROOT; do
    [[ -n "${!name:-}" ]] || { echo "missing $name" >&2; exit 2; }
done
[[ "${SLURM_NNODES:-}" == 1 && "$BASELINE_ROOT" == /lustre/* && "$OUTPUT_PATH" == /lustre/* ]] || {
    echo "invalid audit topology or paths" >&2
    exit 2
}
[[ "$BASELINE_SOURCE_ROOT" == /lustre/* && -d "$BASELINE_SOURCE_ROOT" ]] || {
    echo "baseline source root is unavailable" >&2
    exit 2
}
[[ ! -e "$OUTPUT_PATH" ]] || { echo "immutable audit receipt already exists" >&2; exit 2; }
[[ "$(git -C "$SOURCE_PATH" rev-parse HEAD)" == "$SOURCE_SHA" && -z "$(git -C "$SOURCE_PATH" status --porcelain)" ]] || {
    echo "source checkout identity mismatch" >&2
    exit 2
}
[[ -f "$RUNTIME_ARCHIVE" && "$(sha256sum "$RUNTIME_ARCHIVE" | cut -d' ' -f1)" == "$RUNTIME_ARCHIVE_SHA256" ]] || {
    echo "runtime archive identity mismatch" >&2
    exit 2
}

JOB_ROOT="${SCRATCH_ROOT%/}/${USER}/${SLURM_JOB_ID}/ptv2-baseline-audit"
export JOB_ROOT
mkdir -p "$JOB_ROOT/runtime"
trap 'rm -rf -- "$JOB_ROOT"' EXIT
tar --extract --file="$RUNTIME_ARCHIVE" --directory="$JOB_ROOT/runtime"
old_venv=$(sed -nE 's/^[[:space:]]*export[[:space:]]+VIRTUAL_ENV=(.*)$/\1/p' "$JOB_ROOT/runtime/bin/activate" | head -n 1)
old_venv="${old_venv#\"}"
old_venv="${old_venv%\"}"
[[ -n "$old_venv" ]] || { echo "runtime archive has no VIRTUAL_ENV" >&2; exit 2; }
grep -IlZ "$old_venv" "$JOB_ROOT/runtime/bin"/* "$JOB_ROOT/runtime/pyvenv.cfg" 2>/dev/null \
    | xargs -0 -r sed -i "s|$old_venv|$JOB_ROOT/runtime|g"

# The container shell expands the exported paths after srun starts.
# shellcheck disable=SC2016
srun --nodes=1 --ntasks=1 --gpus=4 --cpus-per-task=64 --mem=400G --kill-on-bad-exit=1 \
    --no-container-mount-home --container-image="$IMAGE_PATH" \
    --container-mounts="$JOB_ROOT:$JOB_ROOT,$SOURCE_PATH:$SOURCE_PATH,$BASELINE_ROOT:$BASELINE_ROOT,$BASELINE_SOURCE_ROOT:$BASELINE_SOURCE_ROOT,$(dirname "$OUTPUT_PATH"):$(dirname "$OUTPUT_PATH")" \
    bash -c 'export VIRTUAL_ENV="$JOB_ROOT/runtime"; export PATH="$VIRTUAL_ENV/bin:$PATH"; exec python3 "$SOURCE_PATH/examples/dataset/prepare_ptv2_baseline_audit.py" --asset-root "$BASELINE_ROOT" --source-root "$BASELINE_SOURCE_ROOT" --identity-path "$BASELINE_IDENTITY_PATH" --identity-sha256 "$BASELINE_IDENTITY_SHA256" --completion-path "$BASELINE_COMPLETION_PATH" --completion-sha256 "$BASELINE_COMPLETION_SHA256" --sha256-list-path "$BASELINE_SHA256_LIST_PATH" --sha256-list-sha256 "$BASELINE_SHA256_LIST_SHA256" --work-root "$JOB_ROOT/baseline-preparation" --audit-script "$SOURCE_PATH/examples/dataset/audit_ptv2_baseline.py" --output "$OUTPUT_PATH"'
