#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Check a staged PTV2 directory inside the runtime the trainer actually uses.
#
# The checker imports the loader out of the pinned modelopt checkout, so the
# answer it gives is the answer training would get. That needs the runtime
# archive rather than the container's own site-packages, which is why this
# mirrors run_ptv2_baseline_audit.sbatch's extract-and-relocate dance.
#
# Usage:
#   CHECK_CLUSTER=lyris DATASET_ROOT=... TOKENIZER_PATH=... OUTPUT_PATH=... \
#   SOURCE_PATH=... IMAGE_PATH=... RUNTIME_ARCHIVE=... SCRATCH_ROOT=/raid/scratch \
#   sbatch --account=... --partition=... run_ptv2_corpus_check.sbatch

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --time=00:40:00

set -euo pipefail

for name in CHECK_CLUSTER DATASET_ROOT OUTPUT_PATH SOURCE_PATH IMAGE_PATH RUNTIME_ARCHIVE SCRATCH_ROOT; do
    [[ -n "${!name:-}" ]] || { echo "missing $name" >&2; exit 2; }
done
[[ "${SLURM_NNODES:-}" == 1 ]] || { echo "this check runs on one node" >&2; exit 2; }
[[ -d "$DATASET_ROOT" ]] || { echo "dataset root is unavailable: $DATASET_ROOT" >&2; exit 2; }
[[ -f "$IMAGE_PATH" ]] || { echo "image is unavailable: $IMAGE_PATH" >&2; exit 2; }
[[ -f "$RUNTIME_ARCHIVE" ]] || { echo "runtime archive is unavailable: $RUNTIME_ARCHIVE" >&2; exit 2; }
# The checker imports from this checkout, so a dirty tree would make the result
# describe code that is not what the pinned runs use.
[[ -z "$(git -C "$SOURCE_PATH" status --porcelain)" ]] || {
    echo "source checkout is dirty: $SOURCE_PATH" >&2
    exit 2
}

CHECKER="${CHECKER:-$SLURM_SUBMIT_DIR/check_ptv2_corpus.py}"
[[ -r "$CHECKER" ]] || { echo "checker not found: $CHECKER" >&2; exit 2; }

# modelopt/__init__.py resolves __version__ through importlib.metadata, so the
# checkout is not importable from PYTHONPATH alone. run_drafter_training.sbatch
# writes the metadata beside a staged copy rather than installing anything;
# copy that so the checker imports the same loader the trainer does.
SOURCE_DISTRIBUTION_VERSION="$(
    describe="$(git -C "$SOURCE_PATH" describe --tags --long --match "[0-9]*" 2>/dev/null || true)"
    if [[ "$describe" =~ ^([0-9][^-]*)-([0-9]+)-g([0-9a-f]+)$ ]]; then
        printf "%s.dev%s+g%s" "${BASH_REMATCH[1]}" "${BASH_REMATCH[2]}" "${BASH_REMATCH[3]}"
    else
        sed -nE 's/^fallback_version[[:space:]]*=[[:space:]]*"([^"]+)".*/\1/p' \
            "$SOURCE_PATH/pyproject.toml" | head -n 1
    fi
)"
[[ -n "$SOURCE_DISTRIBUTION_VERSION" ]] || {
    echo "cannot determine the nvidia-modelopt distribution version" >&2
    exit 2
}
echo "staged modelopt distribution version: $SOURCE_DISTRIBUTION_VERSION"

JOB_ROOT="${SCRATCH_ROOT%/}/${USER}/${SLURM_JOB_ID}/ptv2-corpus-check"
export JOB_ROOT
mkdir -p "$JOB_ROOT/runtime"
trap 'rm -rf -- "$JOB_ROOT"' EXIT
# Staging copies this tree onto every node of every arm at once. .git is three
# quarters of the bytes and the job never reads it. Worse, any git command in
# the pinned tree creates and removes .git/index.lock, and a copy that
# enumerates the directory before the lock is gone but opens it after fails
# with ENOENT, which kills the staging task and takes the whole step down with
# it. Excluding .git removes both the cost and the race. Keep apostrophes out
# of this block: in some callers it sits inside a single-quoted bash -c body.
mkdir -p "$JOB_ROOT/source"
tar --create --file=- --exclude=./.git --directory="$SOURCE_PATH" . \
    | tar --extract --file=- --directory="$JOB_ROOT/source"
egg_info="$JOB_ROOT/source/nvidia_modelopt.egg-info"
mkdir -p "$egg_info"
printf "Metadata-Version: 2.4\nName: nvidia-modelopt\nVersion: %s\n" \
    "$SOURCE_DISTRIBUTION_VERSION" >"$egg_info/PKG-INFO"
printf "modelopt\n" >"$egg_info/top_level.txt"
tar --extract --file="$RUNTIME_ARCHIVE" --directory="$JOB_ROOT/runtime"
old_venv=$(sed -nE 's/^[[:space:]]*export[[:space:]]+VIRTUAL_ENV=(.*)$/\1/p' "$JOB_ROOT/runtime/bin/activate" | head -n 1)
old_venv="${old_venv#\"}"
old_venv="${old_venv%\"}"
[[ -n "$old_venv" ]] || { echo "runtime archive has no VIRTUAL_ENV" >&2; exit 2; }
grep -IlZ "$old_venv" "$JOB_ROOT/runtime/bin"/* "$JOB_ROOT/runtime/pyvenv.cfg" 2>/dev/null \
    | xargs -0 -r sed -i "s|$old_venv|$JOB_ROOT/runtime|g"
cp -- "$CHECKER" "$JOB_ROOT/check_ptv2_corpus.py"

mkdir -p "$(dirname "$OUTPUT_PATH")"

CHECK_EXTRA=()
MOUNTS="$JOB_ROOT:$JOB_ROOT,$DATASET_ROOT:$DATASET_ROOT,$(dirname "$OUTPUT_PATH"):$(dirname "$OUTPUT_PATH")"
if [[ -n "${TOKENIZER_PATH:-}" ]]; then
    # A hub snapshot is a farm of symlinks into ../../blobs, so mounting the
    # snapshot directory alone hands the container dangling links and the fast
    # tokenizer silently falls back to a slow one it cannot build. Dereference
    # the small files into the job root instead of mounting the 16 GB repo.
    mkdir -p "$JOB_ROOT/tokenizer"
    for item in "$TOKENIZER_PATH"/*; do
        case "$item" in
            *.safetensors|*.bin|*.pt) continue ;;
        esac
        cp -L --no-preserve=all -- "$item" "$JOB_ROOT/tokenizer/"
    done
    CHECK_EXTRA+=(--tokenizer-path "$JOB_ROOT/tokenizer")
fi
# Compute nodes have no egress here, so the submitter fetches the published
# sizes on the login node and the checker reads them from the job root.
if [[ -n "${PUBLISHED_SIZES:-}" ]]; then
    cp -- "$PUBLISHED_SIZES" "$JOB_ROOT/published_sizes.json"
    CHECK_EXTRA+=(--published-sizes "$JOB_ROOT/published_sizes.json")
fi

export CHECK_CLUSTER DATASET_ROOT OUTPUT_PATH
export CHECK_ARGS="${CHECK_EXTRA[*]:-}"
export CHECK_ROWS_PER_SHARD="${CHECK_ROWS_PER_SHARD:-64}"
export CHECK_MAX_SEQ_LEN="${CHECK_MAX_SEQ_LEN:-4096}"

# The container shell expands the exported paths after srun starts.
# shellcheck disable=SC2016
srun --nodes=1 --ntasks=1 --cpus-per-task=32 --kill-on-bad-exit=1 \
    --no-container-mount-home --container-image="$IMAGE_PATH" \
    --container-mounts="$MOUNTS" \
    bash -c 'export VIRTUAL_ENV="$JOB_ROOT/runtime"; export PATH="$VIRTUAL_ENV/bin:$PATH"; export PYTHONPATH="$JOB_ROOT/source${PYTHONPATH:+:$PYTHONPATH}"; export HF_HUB_OFFLINE=1; exec python3 "$JOB_ROOT/check_ptv2_corpus.py" --dataset-root "$DATASET_ROOT" --output "$OUTPUT_PATH" --rows-per-shard "$CHECK_ROWS_PER_SHARD" --max-seq-len "$CHECK_MAX_SEQ_LEN" $CHECK_ARGS'
