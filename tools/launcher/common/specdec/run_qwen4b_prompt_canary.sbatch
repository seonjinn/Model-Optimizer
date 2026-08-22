#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --segment=1
#SBATCH --time=01:55:00

set -euo pipefail

for name in SOURCE_PATH SOURCE_SHA IMAGE_PATH IMAGE_SHA256 RUNTIME_ARCHIVE RUNTIME_ARCHIVE_SHA256 SOURCE_MANIFEST SOURCE_MANIFEST_SHA256 TOKENIZER_PATH TOKENIZER_SHA256 OUTPUT_ROOT ROWS_PER_CATEGORY SCRATCH_ROOT; do
    [[ -n "${!name:-}" ]] || { echo "missing required environment: $name" >&2; exit 2; }
done
[[ "${SLURM_NNODES:-}" == 1 && "$SOURCE_PATH" == /home/* && "$OUTPUT_ROOT" == /lustre/* ]] || {
    echo "invalid prompt-canary topology or paths" >&2
    exit 2
}
[[ "$SCRATCH_ROOT" == /* && "$SCRATCH_ROOT" != /home* && "$SCRATCH_ROOT" != /lustre* ]] || {
    echo "scratch root must be node-local" >&2
    exit 2
}
[[ "$(git -C "$SOURCE_PATH" rev-parse HEAD)" == "$SOURCE_SHA" && -z "$(git -C "$SOURCE_PATH" status --porcelain)" ]] || {
    echo "source checkout identity mismatch" >&2
    exit 2
}
for specification in \
    "$IMAGE_PATH:$IMAGE_SHA256:image" \
    "$RUNTIME_ARCHIVE:$RUNTIME_ARCHIVE_SHA256:runtime archive" \
    "$SOURCE_MANIFEST:$SOURCE_MANIFEST_SHA256:source manifest"; do
    IFS=: read -r path expected label <<<"$specification"
    [[ -f "$path" && "$(sha256sum "$path" | cut -d' ' -f1)" == "$expected" ]] || {
        echo "$label identity mismatch" >&2
        exit 2
    }
done
[[ -d "$TOKENIZER_PATH" && ! -e "$OUTPUT_ROOT" ]] || {
    echo "tokenizer is missing or immutable output already exists" >&2
    exit 2
}

JOB_ROOT="${SCRATCH_ROOT%/}/${USER}/${SLURM_JOB_ID}/qwen4b-prompt-canary"
[[ "$JOB_ROOT" == "${SCRATCH_ROOT%/}/${USER}/${SLURM_JOB_ID}/"* ]] || {
    echo "unsafe node-local job root" >&2
    exit 2
}
export JOB_ROOT
rm -rf -- "$JOB_ROOT"
mkdir -p "$JOB_ROOT/runtime"
trap 'rm -rf -- "$JOB_ROOT"' EXIT
tar --extract --file="$RUNTIME_ARCHIVE" --directory="$JOB_ROOT/runtime"
old_venv=$(sed -nE 's/^[[:space:]]*export[[:space:]]+VIRTUAL_ENV=(.*)$/\1/p' "$JOB_ROOT/runtime/bin/activate" | head -n 1)
old_venv="${old_venv#\"}"
old_venv="${old_venv%\"}"
[[ -n "$old_venv" ]] || { echo "runtime archive has no VIRTUAL_ENV" >&2; exit 2; }
grep -IlZ "$old_venv" "$JOB_ROOT/runtime/bin"/* "$JOB_ROOT/runtime/pyvenv.cfg" 2>/dev/null \
    | xargs -0 -r sed -i "s|$old_venv|$JOB_ROOT/runtime|g"

srun --nodes=1 --ntasks=1 --gpus=4 --cpus-per-task=64 --mem=400G \
    --kill-on-bad-exit=1 --no-container-mount-home \
    --container-image="$IMAGE_PATH" \
    --container-mounts="$JOB_ROOT:$JOB_ROOT,$SOURCE_PATH:$SOURCE_PATH,$(dirname "$SOURCE_MANIFEST"):$(dirname "$SOURCE_MANIFEST"),$TOKENIZER_PATH:$TOKENIZER_PATH,$(dirname "$OUTPUT_ROOT"):$(dirname "$OUTPUT_ROOT")" \
    bash -c '
        set -euo pipefail
        export VIRTUAL_ENV="$JOB_ROOT/runtime"
        export PATH="$VIRTUAL_ENV/bin:$PATH"
        export PYTHONPATH="$SOURCE_PATH/tools/launcher"
        exec python3 "$SOURCE_PATH/tools/launcher/common/specdec/build_qwen4b_prompt_canary.py" \
            --source-manifest "$SOURCE_MANIFEST" \
            --source-manifest-sha256 "$SOURCE_MANIFEST_SHA256" \
            --tokenizer "$TOKENIZER_PATH" \
            --tokenizer-sha256 "$TOKENIZER_SHA256" \
            --rows-per-category "$ROWS_PER_CATEGORY" \
            --output-root "$OUTPUT_ROOT"
    '

echo "prompt canary complete: $OUTPUT_ROOT"
