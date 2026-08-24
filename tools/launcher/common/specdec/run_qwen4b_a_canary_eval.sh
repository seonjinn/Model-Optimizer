#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

for name in A_CANARY_EXPORT A_CANARY_EVALUATION_RECEIPT A_CANARY_EVAL_OUTPUT \
    LAUNCHER_ROOT SLURM_JOB_ID; do
    [[ -n "${!name:-}" ]] || { echo "missing evaluator environment: $name" >&2; exit 2; }
done
[[ "$SLURM_JOB_ID" =~ ^[0-9]+$ ]] || exit 2
readonly WRAPPER="$LAUNCHER_ROOT/common/specdec/run_speculators_eval.sh"
readonly RUN_ID="a-repair-${SLURM_JOB_ID}"

export DRAFT_MODEL="$A_CANARY_EXPORT"
export SPEC_METHOD=dflash DFLASH_BLOCK_SIZE=8 NUM_SPEC_TOKENS=7
export MAX_CONCURRENCY="${MAX_CONCURRENCY:-1}" MAX_REQUESTS="${MAX_REQUESTS:-4}"
export EVAL_MODE=throughput EVAL_RUN_ID="$RUN_ID" EVAL_OUTPUT_ROOT="$A_CANARY_EVAL_OUTPUT"
bash "$WRAPPER"

PYTHONPATH="$LAUNCHER_ROOT${PYTHONPATH:+:$PYTHONPATH}" python3 - \
    "$A_CANARY_EVAL_OUTPUT/$RUN_ID" "$A_CANARY_EVALUATION_RECEIPT" \
    "$SLURM_JOB_ID" "$A_CANARY_EXPORT" <<'PY'
import csv
import hashlib
import json
import math
import sys
from pathlib import Path

from common.specdec.qwen4b_a_canary_manifest import _directory_sha256
from common.specdec.qwen4b_b_atomic import atomic_publish_bytes

run_root, receipt_path, job_id, export_path = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3], Path(sys.argv[4])
manifest = json.loads((run_root / "manifest.json").read_bytes())
if manifest.get("status") != "success" or str(manifest.get("slurm_job_id")) != job_id:
    raise ValueError("Speculators evaluator did not complete in the current job")
with (run_root / "perf_results.csv").open(newline="") as stream:
    perf = list(csv.DictReader(stream))
with (run_root / "acceptance.csv").open(newline="") as stream:
    acceptance = list(csv.DictReader(stream))
if not perf or not acceptance:
    raise ValueError("Speculators evaluator produced no completed subsets")
completed = sum(int(float(row["completed_requests"])) for row in perf)
metric_values = [float(row["output_tps_median"]) for row in perf]
positions = [
    float(value)
    for row in acceptance
    for key, value in row.items()
    if key.startswith("acceptance_at_pos_") and value not in (None, "")
]
if completed <= 0 or not metric_values or not positions or not all(map(math.isfinite, metric_values + positions)):
    raise ValueError("Speculators evaluator metrics are incomplete")
body = {
    "schema_version": 1,
    "slurm_job_id": job_id,
    "status": "passed",
    "evaluator": "specdec-bench-v1",
    "export_sha256": _directory_sha256(export_path),
    "completed_requests": completed,
    "metrics": {
        "mean_output_tps": sum(metric_values) / len(metric_values),
        "mean_position_acceptance": sum(positions) / len(positions),
    },
}
body["receipt_sha256"] = hashlib.sha256(
    json.dumps(body, sort_keys=True, separators=(",", ":")).encode()
).hexdigest()
atomic_publish_bytes(
    receipt_path,
    (json.dumps(body, sort_keys=True, separators=(",", ":")) + "\n").encode(),
    job_id=job_id,
)
PY
