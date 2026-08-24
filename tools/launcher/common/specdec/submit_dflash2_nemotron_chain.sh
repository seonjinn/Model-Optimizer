#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Submit one independent 20 -> 4166 -> 14500 -> 25391 chain per target.
set -euo pipefail

usage() {
    echo "usage: $0 --manifest PATH --receipt PATH --cluster-profile PATH [--target q30-base|q235-base] [--readiness-receipt PATH] [--dry-run]" >&2
    exit 2
}

MANIFEST="" RECEIPT="" CLUSTER_PROFILE="" TARGET="" READINESS_RECEIPT="" DRY_RUN=0
while (( $# )); do
    case "$1" in
        --manifest) MANIFEST="${2:-}"; shift 2 ;;
        --receipt) RECEIPT="${2:-}"; shift 2 ;;
        --cluster-profile) CLUSTER_PROFILE="${2:-}"; shift 2 ;;
        --target) TARGET="${2:-}"; shift 2 ;;
        --readiness-receipt) READINESS_RECEIPT="${2:-}"; shift 2 ;;
        --dry-run) DRY_RUN=1; shift ;;
        *) usage ;;
    esac
done
is_launcher_path() {
    [[ "$1" == /home/* || "$1" == /project/coreai_dlalgo_llm/users/sna/* ]]
}
is_launcher_path "$MANIFEST" && [[ -f "$MANIFEST" ]] || usage
if [[ "$RECEIPT" != /lustre/* ]] || ! is_launcher_path "$CLUSTER_PROFILE"; then
    usage
fi
[[ -f "$CLUSTER_PROFILE" ]] || usage
[[ -z "$TARGET" || "$TARGET" == "q30-base" || "$TARGET" == "q235-base" ]] || usage

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
LAUNCHER_ROOT="$(cd -- "$SCRIPT_DIR/../.." && pwd)"
readonly SCRIPT_DIR LAUNCHER_ROOT
readonly WAVE_SUBMITTER="$SCRIPT_DIR/submit_drafter_training_wave.sh"
mapfile -t experiment_records < <(
    PYTHONPATH="$LAUNCHER_ROOT${PYTHONPATH:+:$PYTHONPATH}" python3 - "$MANIFEST" "$TARGET" <<'PY'
import sys
from pathlib import Path
from common.specdec.drafter_job_manifest import load_manifest

experiments = load_manifest(Path(sys.argv[1]))
if len(experiments) != 2 or {item.target for item in experiments} != {"q30-base", "q235-base"}:
    raise SystemExit("DFlash2 chain requires the exact Q30/Q235 matrix")
for index, experiment in enumerate(experiments):
    if sys.argv[2] and experiment.target != sys.argv[2]:
        continue
    if (
        experiment.method != "dflash2"
        or experiment.slurm.nodes != 16
        or experiment.slurm.segment != 16
        or experiment.cumulative_max_steps != (20, 4166, 14500, 25391)
    ):
        raise SystemExit("DFlash2 chain manifest is not the native 16-node canary/full plan")
    print(f"{index}\t{experiment.experiment_id}")
PY
)
expected_records=2
[[ -z "$TARGET" ]] || expected_records=1
(( ${#experiment_records[@]} == expected_records )) || exit 2

for experiment_record in "${experiment_records[@]}"; do
    IFS=$'\t' read -r experiment_index experiment_id <<<"$experiment_record"
    previous_job_id=""
    for max_steps in 20 4166 14500 25391; do
        command=(
            "$WAVE_SUBMITTER" --manifest "$MANIFEST" --receipt "$RECEIPT"
            --cluster-profile "$CLUSTER_PROFILE" --experiment-index "$experiment_index"
            --max-steps "$max_steps"
        )
        [[ -z "$READINESS_RECEIPT" ]] || command+=(--readiness-receipt "$READINESS_RECEIPT")
        [[ -z "$previous_job_id" ]] || command+=(--dependency "$previous_job_id")
        if (( DRY_RUN )); then
            command+=(--dry-run)
        fi
        "${command[@]}"
        if (( ! DRY_RUN )); then
            previous_job_id="$(python3 - "$RECEIPT" "$experiment_id" "$max_steps" <<'PY'
import json
import sys
from pathlib import Path

for line in reversed(Path(sys.argv[1]).read_text().splitlines()):
    record = json.loads(line)
    identity = str(record.get("tuple_identity", ""))
    expected_suffix = f"{sys.argv[2]}:{sys.argv[3]}"
    if record.get("job_id") and (identity == expected_suffix or identity.endswith(f":{expected_suffix}")):
        print(record["job_id"])
        break
PY
)"
            [[ "$previous_job_id" =~ ^[0-9]+$ ]] || {
                echo "DFlash2 chain could not bind submitted job ${experiment_index}/${max_steps}" >&2
                exit 1
            }
        fi
    done
done
