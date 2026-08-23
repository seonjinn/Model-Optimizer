#!/usr/bin/env bash
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

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --segment=1
#SBATCH --time=03:55:00

set -euo pipefail

for name in MANIFEST_PATH SYNTHESIS_INDEX LAUNCHER_ROOT CLUSTER_PROFILE_PATH CLUSTER_PROFILE_SHA256 READINESS_RECEIPT READINESS_RECEIPT_SHA256 SCRATCH_ROOT; do
    [[ -n "${!name:-}" ]] || { echo "missing required environment: $name" >&2; exit 2; }
done
[[ "${SLURM_NNODES:-}" == 1 ]] || { echo "synthesis requires exactly one node" >&2; exit 2; }
[[ "$SCRATCH_ROOT" == /* && "$SCRATCH_ROOT" != /home* && "$SCRATCH_ROOT" != /lustre* ]] || {
    echo "scratch root must be node-local" >&2
    exit 2
}
for specification in \
    "$CLUSTER_PROFILE_PATH:$CLUSTER_PROFILE_SHA256:cluster profile" \
    "$READINESS_RECEIPT:$READINESS_RECEIPT_SHA256:readiness receipt"; do
    IFS=: read -r path expected label <<<"$specification"
    [[ -f "$path" && "$(sha256sum "$path" | cut -d' ' -f1)" == "$expected" ]] || {
        echo "$label identity mismatch" >&2
        exit 2
    }
done

mapfile -t values < <(PYTHONPATH="$LAUNCHER_ROOT${PYTHONPATH:+:$PYTHONPATH}" python3 - "$MANIFEST_PATH" "$SYNTHESIS_INDEX" "$CLUSTER_PROFILE_PATH" "$READINESS_RECEIPT" "$SCRATCH_ROOT" <<'PY'
import json
import sys
from pathlib import Path

from common.specdec.build_qwen4b_synthesis_manifest import load_synthesis_manifest
from common.specdec.cluster_profile import load_cluster_profile, validate_scratch_root
from common.specdec.qwen4b_study_manifest import validate_readiness_receipt

manifest = load_synthesis_manifest(Path(sys.argv[1]))[int(sys.argv[2])]
profile = load_cluster_profile(Path(sys.argv[3]))
receipt = json.loads(Path(sys.argv[4]).read_text())
scratch_root = Path(sys.argv[5])
validate_scratch_root(profile, scratch_root)
if manifest.inputs.source_sha != profile.modelopt_commit:
    raise SystemExit("source/profile commit mismatch")
if manifest.settings.account != profile.account or manifest.settings.partition != profile.partition:
    raise SystemExit("manifest/profile scheduler mismatch")
if profile.gpus_per_node != manifest.settings.replicas:
    raise SystemExit("manifest/profile GPU mismatch")
validate_readiness_receipt(
    receipt,
    profile=profile.name,
    account=profile.account,
    partition=profile.partition,
    scratch_root=scratch_root,
)
if manifest.response_source != "target-synth":
    raise SystemExit("trace-replay is a corpus input and must not allocate synthesis GPUs")

for value in (
    manifest.experiment_id,
    manifest.arm,
    manifest.thinking_mode,
    manifest.inputs.source_path,
    manifest.inputs.source_sha,
    manifest.inputs.image_path,
    manifest.inputs.image_sha256,
    manifest.inputs.runtime_archive_path,
    manifest.inputs.runtime_archive_sha256,
    manifest.inputs.target_path,
    manifest.inputs.target_revision,
    manifest.inputs.tokenizer_sha256,
    manifest.inputs.chat_template_sha256,
    manifest.inputs.prompt_manifest_path,
    manifest.inputs.prompt_manifest_sha256,
    manifest.inputs.output_root,
    manifest.settings.replicas,
    manifest.settings.tensor_parallel_size,
    manifest.settings.num_shards,
    manifest.settings.output_token_budget,
    manifest.settings.temperature,
    manifest.settings.max_tokens,
    manifest.settings.max_total_length,
    manifest.generation_identity_sha256,
):
    print(value)
PY
)
(( ${#values[@]} == 24 )) || { echo "synthesis manifest entry is unavailable" >&2; exit 2; }
EXPERIMENT_ID="${values[0]}"
ARM="${values[1]}"
THINKING_MODE="${values[2]}"
SOURCE_PATH="${values[3]}"
SOURCE_SHA="${values[4]}"
IMAGE_PATH="${values[5]}"
IMAGE_SHA256="${values[6]}"
RUNTIME_ARCHIVE="${values[7]}"
RUNTIME_ARCHIVE_SHA256="${values[8]}"
TARGET_PATH="${values[9]}"
TARGET_REVISION="${values[10]}"
TOKENIZER_SHA256="${values[11]}"
CHAT_TEMPLATE_SHA256="${values[12]}"
PROMPT_MANIFEST="${values[13]}"
PROMPT_MANIFEST_SHA256="${values[14]}"
OUTPUT_ROOT="${values[15]}"
REPLICAS="${values[16]}"
SERVE_TP="${values[17]}"
NUM_SHARDS="${values[18]}"
OUTPUT_TOKEN_BUDGET="${values[19]}"
TEMPERATURE="${values[20]}"
MAX_TOKENS="${values[21]}"
MAX_TOTAL_LENGTH="${values[22]}"
GENERATION_IDENTITY_SHA256="${values[23]}"
readonly EXPERIMENT_ID ARM THINKING_MODE SOURCE_PATH SOURCE_SHA IMAGE_PATH IMAGE_SHA256
readonly RUNTIME_ARCHIVE RUNTIME_ARCHIVE_SHA256 TARGET_PATH TARGET_REVISION TOKENIZER_SHA256
readonly CHAT_TEMPLATE_SHA256 GENERATION_IDENTITY_SHA256
readonly PROMPT_MANIFEST PROMPT_MANIFEST_SHA256 OUTPUT_ROOT REPLICAS SERVE_TP NUM_SHARDS
readonly OUTPUT_TOKEN_BUDGET TEMPERATURE MAX_TOKENS MAX_TOTAL_LENGTH
export ARM THINKING_MODE TARGET_REVISION NUM_SHARDS SERVE_TP REPLICAS TEMPERATURE MAX_TOKENS MAX_TOTAL_LENGTH

[[ "$REPLICAS" == 4 && "$SERVE_TP" == 1 ]] || { echo "synthesis must use four TP1 replicas" >&2; exit 2; }
[[ "$OUTPUT_ROOT" == /lustre/* && "$SOURCE_PATH" == /home/* ]] || { echo "invalid durable/source paths" >&2; exit 2; }
[[ "$(git -C "$SOURCE_PATH" rev-parse HEAD)" == "$SOURCE_SHA" && -z "$(git -C "$SOURCE_PATH" status --porcelain)" ]] || {
    echo "source checkout identity mismatch" >&2
    exit 2
}
for specification in \
    "$IMAGE_PATH:$IMAGE_SHA256:image" \
    "$RUNTIME_ARCHIVE:$RUNTIME_ARCHIVE_SHA256:runtime archive" \
    "$PROMPT_MANIFEST:$PROMPT_MANIFEST_SHA256:prompt manifest"; do
    IFS=: read -r path expected label <<<"$specification"
    [[ -f "$path" && "$(sha256sum "$path" | cut -d' ' -f1)" == "$expected" ]] || {
        echo "$label identity mismatch" >&2
        exit 2
    }
done
[[ -d "$TARGET_PATH" ]] || { echo "target snapshot is missing" >&2; exit 2; }
PYTHONPATH="$LAUNCHER_ROOT${PYTHONPATH:+:$PYTHONPATH}" python3 - \
    "$PROMPT_MANIFEST" "$PROMPT_MANIFEST_SHA256" "$TARGET_PATH" \
    "$TARGET_REVISION" "$TOKENIZER_SHA256" <<'PY'
import json
import sys
from pathlib import Path

from common.specdec.build_qwen4b_synthesis_manifest import (
    tokenizer_snapshot_sha256,
    verify_data_manifest,
)

prompt_manifest = Path(sys.argv[1])
verify_data_manifest(prompt_manifest, sys.argv[2])
target = Path(sys.argv[3])
try:
    snapshot = json.loads((target / "snapshot-manifest.json").read_text())
except (FileNotFoundError, json.JSONDecodeError) as error:
    raise SystemExit("target snapshot manifest is missing or invalid") from error
if snapshot.get("source_identity") != sys.argv[4]:
    raise SystemExit("target snapshot revision mismatch")
if tokenizer_snapshot_sha256(target) != sys.argv[5]:
    raise SystemExit("target tokenizer identity mismatch")
PY
[[ -f "$TARGET_PATH/chat_template.jinja" ]] || { echo "target chat template is missing" >&2; exit 2; }
[[ "$(sha256sum "$TARGET_PATH/chat_template.jinja" | cut -d' ' -f1)" == "$CHAT_TEMPLATE_SHA256" ]] || {
    echo "target chat template identity mismatch" >&2
    exit 2
}

if [[ -f "$OUTPUT_ROOT/completion.json" ]]; then
    PYTHONPATH="$LAUNCHER_ROOT${PYTHONPATH:+:$PYTHONPATH}" python3 - \
        "$OUTPUT_ROOT" "$EXPERIMENT_ID" "$OUTPUT_TOKEN_BUDGET" "$TOKENIZER_SHA256" \
        "7" "$GENERATION_IDENTITY_SHA256" <<'PY'
import sys
from pathlib import Path

from common.specdec.build_qwen4b_synthesis_manifest import verify_synthesis_completion

verify_synthesis_completion(
    Path(sys.argv[1]),
    sys.argv[2],
    expected_output_token_budget=int(sys.argv[3]),
    expected_tokenizer_sha256=sys.argv[4],
    expected_file_count=int(sys.argv[5]),
    expected_generation_identity_sha256=sys.argv[6],
)
PY
    echo "reusing completed synthesis output: $OUTPUT_ROOT"
    exit 0
fi
[[ ! -e "$OUTPUT_ROOT" ]] || { echo "incomplete synthesis output requires quarantine" >&2; exit 2; }

PROMPT_DATA_ROOT="$(dirname "$PROMPT_MANIFEST")"
PROMPT_MANIFEST_NAME="$(basename "$PROMPT_MANIFEST")"
ATTEMPT_ROOT="${OUTPUT_ROOT}.attempts-${EXPERIMENT_ID}"
PROMOTED_PARTIAL="${OUTPUT_ROOT}.promotion-partial-${SLURM_JOB_ID}"
readonly ATTEMPT_ROOT PROMOTED_PARTIAL
[[ "$ATTEMPT_ROOT" == /lustre/*.attempts-"$EXPERIMENT_ID" ]] || {
    echo "invalid durable synthesis attempt root" >&2
    exit 2
}
JOB_ROOT="${SCRATCH_ROOT%/}/${USER}/${SLURM_JOB_ID}/qwen4b-synthesis-${EXPERIMENT_ID}"
[[ "$JOB_ROOT" == "${SCRATCH_ROOT%/}/${USER}/${SLURM_JOB_ID}/"* ]] || { echo "unsafe scratch path" >&2; exit 2; }
rm -rf -- "$JOB_ROOT"
mkdir -p "$JOB_ROOT/input" "$JOB_ROOT/runtime" "$JOB_ROOT/replicas" "$JOB_ROOT/logs"
mkdir -p "$ATTEMPT_ROOT/replicas"
cp -rP --no-preserve=all -- "$SOURCE_PATH" "$JOB_ROOT/source"
cp -rL --no-preserve=all -- "$TARGET_PATH" "$JOB_ROOT/input/target"
cp -rL --no-preserve=all -- "$PROMPT_DATA_ROOT" "$JOB_ROOT/input/prompts"
python3 - "$JOB_ROOT/input/generation-identity.json" "$TARGET_REVISION" "$TOKENIZER_SHA256" \
    "$CHAT_TEMPLATE_SHA256" "$RUNTIME_ARCHIVE_SHA256" "$IMAGE_SHA256" \
    "$PROMPT_MANIFEST_SHA256" "$THINKING_MODE" "$TEMPERATURE" "$MAX_TOKENS" \
    "$MAX_TOTAL_LENGTH" <<'PY'
import json
import sys
from pathlib import Path

Path(sys.argv[1]).write_text(
    json.dumps(
        {
            "target_revision": sys.argv[2],
            "tokenizer_sha256": sys.argv[3],
            "chat_template_sha256": sys.argv[4],
            "runtime_sha256": sys.argv[5],
            "container_sha256": sys.argv[6],
            "source_selection_sha256": sys.argv[7],
            "thinking_mode": sys.argv[8],
            "temperature": float(sys.argv[9]),
            "max_tokens": int(sys.argv[10]),
            "max_total_length": int(sys.argv[11]),
        },
        sort_keys=True,
    )
    + "\n"
)
PY
tar --extract --file="$RUNTIME_ARCHIVE" --directory="$JOB_ROOT/runtime"
old_venv=$(sed -nE 's/^[[:space:]]*export[[:space:]]+VIRTUAL_ENV=(.*)$/\1/p' "$JOB_ROOT/runtime/bin/activate" | head -n 1)
old_venv="${old_venv#\"}"
old_venv="${old_venv%\"}"
[[ -n "$old_venv" ]] || { echo "runtime archive has no VIRTUAL_ENV" >&2; exit 2; }
grep -IlZ "$old_venv" "$JOB_ROOT/runtime/bin"/* "$JOB_ROOT/runtime/pyvenv.cfg" 2>/dev/null \
    | xargs -0 -r sed -i "s|$old_venv|$JOB_ROOT/runtime|g"

CHILD_PIDS=()
kill_children() {
    local pid
    for pid in "${CHILD_PIDS[@]:-}"; do
        kill -TERM "$pid" 2>/dev/null || true
    done
    wait "${CHILD_PIDS[@]:-}" 2>/dev/null || true
}
trap kill_children EXIT INT TERM

OUTPUT_PARENT="$(dirname "$OUTPUT_ROOT")"
CONTAINER_MOUNTS="${SCRATCH_ROOT}:${SCRATCH_ROOT},${OUTPUT_PARENT}:${OUTPUT_PARENT}"
export PROMPT_MANIFEST_NAME PROMPT_MANIFEST_SHA256 ATTEMPT_ROOT
# shellcheck disable=SC2016
for ((replica = 0; replica < REPLICAS; replica++)); do
    replica_root="$JOB_ROOT/replicas/$replica"
    replica_output="$ATTEMPT_ROOT/replicas/$replica"
    mkdir -p "$replica_root"
    srun --nodes=1 --ntasks=1 --exclusive --gpus=1 --cpus-per-task=24 --mem=200G \
        --kill-on-bad-exit=1 --no-container-mount-home \
        --container-image="$IMAGE_PATH" --container-mounts="$CONTAINER_MOUNTS" \
        bash -c '
            set -euo pipefail
            replica="$1"
            replica_root="$2"
            job_root="$3"
            replica_output="$4"
            port=$((8100 + replica))
            export VIRTUAL_ENV="$job_root/runtime"
            export PATH="$VIRTUAL_ENV/bin:$PATH"
            export PYTHONPATH="$job_root/source${PYTHONPATH:+:$PYTHONPATH}"
            export HF_HOME="$replica_root/hf" XDG_CACHE_HOME="$replica_root/xdg"
            export TRITON_CACHE_DIR="$replica_root/triton" TMPDIR="$replica_root/tmp"
            mkdir -p "$HF_HOME" "$XDG_CACHE_HOME" "$TRITON_CACHE_DIR" "$TMPDIR"
            vllm serve "$job_root/input/target" \
                --host 127.0.0.1 --port "$port" --served-model-name qwen3-4b-study \
                --tensor-parallel-size "$SERVE_TP" --max-model-len "$MAX_TOTAL_LENGTH" \
                >"$replica_root/server.log" 2>&1 &
            server_pid=$!
            trap '\''kill -TERM "$server_pid" 2>/dev/null || true; wait "$server_pid" 2>/dev/null || true'\'' EXIT INT TERM
            for _ in $(seq 1 900); do
                curl --fail --silent "http://127.0.0.1:${port}/health" >/dev/null && break
                kill -0 "$server_pid" 2>/dev/null || { echo "vLLM exited before readiness" >&2; exit 1; }
                sleep 2
            done
            curl --fail --silent "http://127.0.0.1:${port}/health" >/dev/null
            python "$job_root/source/examples/dataset/promote_synthesis_reserve.py" \
                generate-attempts \
                --selection-manifest "$job_root/input/prompts/$PROMPT_MANIFEST_NAME" \
                --selection-manifest-sha256 "$PROMPT_MANIFEST_SHA256" --arm "$ARM" \
                --identity "$job_root/input/generation-identity.json" \
                --base-url "http://127.0.0.1:${port}/v1" --model qwen3-4b-study \
                --output "$replica_output/attempts-${SLURM_JOB_ID}-${replica}.jsonl" \
                --num-shards "$NUM_SHARDS" --shard-id-begin "$replica" \
                --shard-id-step "$REPLICAS" \
                --attempt-namespace "${SLURM_JOB_ID}-${replica}"
            python - "$replica_output/gpu-utilization-replica-${replica}.json" "$replica" <<'PY'
import json
import subprocess
import sys
from pathlib import Path

output = subprocess.run(
    ["nvidia-smi", "--query-gpu=index,utilization.gpu", "--format=csv,noheader,nounits"],
    check=True,
    capture_output=True,
    text=True,
).stdout
Path(sys.argv[-2]).write_text(
    json.dumps({"replica": int(sys.argv[-1]), "gpu_utilization": output.splitlines()}, sort_keys=True)
    + "\n"
)
PY
        ' bash "$replica" "$replica_root" "$JOB_ROOT" "$replica_output" \
        >"$JOB_ROOT/logs/replica-${replica}.log" 2>&1 &
    CHILD_PIDS+=("$!")
done

for ((finished = 0; finished < REPLICAS; finished++)); do
    if ! wait -n; then
        echo "a synthesis replica failed" >&2
        exit 1
    fi
done
CHILD_PIDS=()
trap - EXIT INT TERM

partial="$PROMOTED_PARTIAL"
[[ ! -e "$partial" ]] || { echo "synthesis partial already exists: $partial" >&2; exit 2; }
mkdir -p "$partial"
srun --nodes=1 --ntasks=1 --exclusive --gpus=1 --cpus-per-task=24 --mem=200G \
    --kill-on-bad-exit=1 --no-container-mount-home \
    --container-image="$IMAGE_PATH" --container-mounts="$CONTAINER_MOUNTS" \
    bash -c '
        set -euo pipefail
        export VIRTUAL_ENV="$1/runtime"
        export PATH="$VIRTUAL_ENV/bin:$PATH"
        export PYTHONPATH="$1/source${PYTHONPATH:+:$PYTHONPATH}"
        python "$1/source/examples/dataset/promote_synthesis_reserve.py" promote-attempts \
            --selection-manifest "$1/input/prompts/$PROMPT_MANIFEST_NAME" \
            --selection-manifest-sha256 "$PROMPT_MANIFEST_SHA256" --arm "$ARM" \
            --identity "$1/input/generation-identity.json" --attempts-root "$2" \
            --output "$3/corpus" --tokenizer "$1/input/target"
    ' bash "$JOB_ROOT" "$ATTEMPT_ROOT" "$partial"
python3 - "$partial" "$EXPERIMENT_ID" "$MANIFEST_PATH" "$OUTPUT_TOKEN_BUDGET" \
    "$TOKENIZER_SHA256" "$GENERATION_IDENTITY_SHA256" "$TARGET_REVISION" \
    "$CHAT_TEMPLATE_SHA256" "$RUNTIME_ARCHIVE_SHA256" "$IMAGE_SHA256" \
    "$PROMPT_MANIFEST_SHA256" "$THINKING_MODE" <<'PY'
import hashlib
import json
import os
import sys
from pathlib import Path

root = Path(sys.argv[1])
promotion_path = root / "corpus/PROMOTION.json"
promotion = json.loads(promotion_path.read_text())
files = []
for record in promotion.get("files", []):
    path = root / "corpus" / record["path"]
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if path.stat().st_size != record["bytes"] or digest != record["sha256"]:
        raise SystemExit(f"promoted corpus identity mismatch: {path.name}")
    files.append(
        {
            "path": f"corpus/{record['path']}",
            "bytes": path.stat().st_size,
            "sha256": digest,
        }
    )
files.append(
    {
        "path": "corpus/PROMOTION.json",
        "bytes": promotion_path.stat().st_size,
        "sha256": hashlib.sha256(promotion_path.read_bytes()).hexdigest(),
    }
)
actual_assistant_tokens = sum(
    int(token_count) * int(count)
    for histogram in promotion["response_token_histograms"].values()
    for token_count, count in histogram.items()
)
requested_assistant_tokens = int(sys.argv[4])
if actual_assistant_tokens < requested_assistant_tokens:
    raise SystemExit(
        f"synthesis produced {actual_assistant_tokens} assistant tokens, "
        f"below requested {requested_assistant_tokens}"
    )
completion = {
    "schema_version": 1,
    "experiment_id": sys.argv[2],
    "manifest_path": sys.argv[3],
    "requested_output_token_budget": requested_assistant_tokens,
    "actual_assistant_tokens": actual_assistant_tokens,
    "tokenizer_sha256": sys.argv[5],
    "generation_identity_sha256": sys.argv[6],
    "target_revision": sys.argv[7],
    "chat_template_sha256": sys.argv[8],
    "runtime_sha256": sys.argv[9],
    "container_sha256": sys.argv[10],
    "source_selection_sha256": sys.argv[11],
    "thinking_mode": sys.argv[12],
    "promotion_status": "passed",
    "corpus_sha256": promotion["corpus_sha256"],
    "promotion_receipt_sha256": hashlib.sha256(promotion_path.read_bytes()).hexdigest(),
    "file_count": len(files),
    "files": files,
}
temporary = root / f".completion.json.tmp-{os.getpid()}"
temporary.write_text(json.dumps(completion, indent=2, sort_keys=True) + "\n")
os.replace(temporary, root / "completion.json")
PY
PYTHONPATH="$LAUNCHER_ROOT${PYTHONPATH:+:$PYTHONPATH}" python3 - \
    "$partial" "$OUTPUT_ROOT" <<'PY'
import hashlib
import sys
from pathlib import Path

from common.specdec.build_qwen4b_synthesis_manifest import publish_synthesis_output

partial = Path(sys.argv[1])
completion = partial / "completion.json"
publish_synthesis_output(
    partial,
    Path(sys.argv[2]),
    expected_completion_sha256=hashlib.sha256(completion.read_bytes()).hexdigest(),
)
PY
echo "synthesis complete: $OUTPUT_ROOT"
