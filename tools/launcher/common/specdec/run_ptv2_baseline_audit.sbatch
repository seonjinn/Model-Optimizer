#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=400G
#SBATCH --time=00:45:00

set -euo pipefail

for name in SOURCE_PATH BASELINE_ROOT OUTPUT_PATH IMAGE_PATH; do
    [[ -n "${!name:-}" ]] || { echo "missing $name" >&2; exit 2; }
done
[[ "${SLURM_NNODES:-}" == 1 && "$BASELINE_ROOT" == /lustre/* && "$OUTPUT_PATH" == /lustre/* ]] || {
    echo "invalid audit topology or paths" >&2
    exit 2
}
[[ ! -e "$OUTPUT_PATH" ]] || { echo "immutable audit receipt already exists" >&2; exit 2; }

srun --nodes=1 --ntasks=1 --gpus=4 --cpus-per-task=64 --mem=400G --kill-on-bad-exit=1 \
    --no-container-mount-home --container-image="$IMAGE_PATH" \
    --container-mounts="$SOURCE_PATH:$SOURCE_PATH,$BASELINE_ROOT:$BASELINE_ROOT,$(dirname "$OUTPUT_PATH"):$(dirname "$OUTPUT_PATH")" \
    python3 "$SOURCE_PATH/examples/dataset/audit_ptv2_baseline.py" \
        --root "$BASELINE_ROOT" --output "$OUTPUT_PATH"
