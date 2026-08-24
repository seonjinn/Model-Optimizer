#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

for name in B_CANARY_EXPORT B_CANARY_EVALUATION_RECEIPT B_CANARY_EVAL_OUTPUT \
    B_CANARY_CHECKPOINT B_CANARY_MANIFEST REPO_ROOT LAUNCHER_ROOT SLURM_JOB_ID; do
    [[ -n "${!name:-}" ]] || { echo "missing evaluator environment: $name" >&2; exit 2; }
done
[[ "$SLURM_JOB_ID" =~ ^[0-9]+$ ]] || exit 2
readonly WRAPPER="$LAUNCHER_ROOT/common/specdec/run_speculators_eval.sh"
readonly RUN_ID="b-balanced-${SLURM_JOB_ID}"

export DRAFT_MODEL="$B_CANARY_EXPORT"
export SPEC_METHOD=dflash DFLASH_BLOCK_SIZE=8 NUM_SPEC_TOKENS=7
export MAX_CONCURRENCY="${MAX_CONCURRENCY:-1}" MAX_REQUESTS="${MAX_REQUESTS:-4}"
export EVAL_MODE=throughput EVAL_RUN_ID="$RUN_ID" EVAL_OUTPUT_ROOT="$B_CANARY_EVAL_OUTPUT"
bash "$WRAPPER"

PYTHONPATH="$LAUNCHER_ROOT${PYTHONPATH:+:$PYTHONPATH}" python3 - \
    "$B_CANARY_EVAL_OUTPUT/$RUN_ID" "$B_CANARY_EVALUATION_RECEIPT" \
    "$SLURM_JOB_ID" "$B_CANARY_EXPORT" "$B_CANARY_CHECKPOINT/checkpoint-200" \
    "$B_CANARY_MANIFEST" "$REPO_ROOT" <<'PY'
import csv
import hashlib
import json
import math
import sys
from pathlib import Path

from common.specdec.qwen4b_b_atomic import atomic_publish_bytes
from common.specdec.qwen4b_b_canary_manifest import _directory_sha256, load_b_canary_manifest

run_root, receipt_path, job_id, export_path = (
    Path(sys.argv[1]),
    Path(sys.argv[2]),
    sys.argv[3],
    Path(sys.argv[4]),
)
checkpoint_path, manifest_path, repo_root = Path(sys.argv[5]), Path(sys.argv[6]), Path(sys.argv[7])
runtime = load_b_canary_manifest(manifest_path).runtime
exporter = repo_root / "examples/speculative_decoding/scripts/export_hf_checkpoint.py"
exporter_raw = exporter.read_bytes()
if (
    hashlib.sha256(exporter_raw).hexdigest() != runtime.exporter_sha256
    or b"load_vlm_or_llm(args.model_path" not in exporter_raw
):
    raise ValueError("bound B exporter does not prove checkpoint reload")
manifest = json.loads((run_root / "manifest.json").read_bytes())
if manifest.get("status") != "success" or str(manifest.get("slurm_job_id")) != job_id:
    raise ValueError("Speculators evaluator did not complete in the current B job")
with (run_root / "perf_results.csv").open(newline="") as stream:
    perf = list(csv.DictReader(stream))
with (run_root / "acceptance.csv").open(newline="") as stream:
    acceptance = list(csv.DictReader(stream))
if not perf or not acceptance:
    raise ValueError("Speculators evaluator produced no completed B subsets")
completed = sum(int(float(row["completed_requests"])) for row in perf)
throughput = [float(row["output_tps_median"]) for row in perf]
positions = [
    float(value)
    for row in acceptance
    for key, value in row.items()
    if key.startswith("acceptance_at_pos_") and value not in (None, "")
]
if completed <= 0 or not throughput or not positions or not all(
    map(math.isfinite, throughput + positions)
):
    raise ValueError("Speculators evaluator B metrics are incomplete")
body = {
    "schema_version": 1,
    "slurm_job_id": job_id,
    "status": "passed",
    "evaluator": "specdec-bench-v1",
    "export_sha256": _directory_sha256(export_path),
    "checkpoint_path": str(checkpoint_path),
    "checkpoint_sha256": _directory_sha256(checkpoint_path),
    "exporter_path": "examples/speculative_decoding/scripts/export_hf_checkpoint.py",
    "exporter_sha256": runtime.exporter_sha256,
    "checkpoint_reloaded_by_exporter": True,
    "completed_requests": completed,
    "metrics": {
        "mean_output_tps": sum(throughput) / len(throughput),
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
