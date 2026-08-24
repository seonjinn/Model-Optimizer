#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2

set -euo pipefail

required=(REPO_ROOT SOURCE_COMMIT TASK5_PARENT_JOB_ID SOURCE_INVENTORY BASELINE_AUDIT
    HELD_OUT_UUIDS POLICY TASK5_MANIFEST OUTPUT_ROOT IMAGE_PATH IMAGE_SHA256 ACCOUNT
    PARTITION WALLTIME TASK9_A_CONTROLLER_RECEIPT)
for name in "${required[@]}"; do
    [[ -n "${!name:-}" ]] || { echo "missing required environment: $name" >&2; exit 2; }
done
[[ "${SLURM_JOB_ID:-}" =~ ^[0-9]+$ && "${SLURM_CPUS_PER_TASK:-}" == 2 ]] || exit 2
[[ "$TASK5_PARENT_JOB_ID" =~ ^[0-9]+$ && "$SOURCE_COMMIT" =~ ^[0-9a-f]{40}$ ]] || exit 2
[[ "$(git -C "$REPO_ROOT" rev-parse HEAD)" == "$SOURCE_COMMIT" ]] || exit 2
[[ -z "$(git -C "$REPO_ROOT" status --porcelain)" ]] || exit 2
[[ -f "$TASK5_MANIFEST" && ! -L "$TASK5_MANIFEST" ]] || exit 2
[[ ! -e "$TASK9_A_CONTROLLER_RECEIPT" && ! -L "$TASK9_A_CONTROLLER_RECEIPT" ]] || exit 2

task5_sha256="$(sha256sum "$TASK5_MANIFEST" | cut -d' ' -f1)"
submitter="$REPO_ROOT/tools/launcher/common/specdec/submit_qwen4b_task9_a.sh"
submission="$($submitter --source-inventory "$SOURCE_INVENTORY" \
    --baseline-audit "$BASELINE_AUDIT" --held-out-uuids "$HELD_OUT_UUIDS" \
    --policy "$POLICY" --task5-manifest "$TASK5_MANIFEST" \
    --task5-manifest-sha256 "$task5_sha256" --output-root "$OUTPUT_ROOT" \
    --image "$IMAGE_PATH" --image-sha256 "$IMAGE_SHA256" --account "$ACCOUNT" \
    --partition "$PARTITION" --time "$WALLTIME")"
task9_job_id="${submission##*$'\n'}"
[[ "$task9_job_id" =~ ^[0-9]+$ ]] || { printf '%s\n' "$submission" >&2; exit 1; }

mkdir -p "$(dirname "$TASK9_A_CONTROLLER_RECEIPT")"
python3 - "$TASK9_A_CONTROLLER_RECEIPT" "$SOURCE_COMMIT" "$TASK5_PARENT_JOB_ID" \
    "$task5_sha256" "$task9_job_id" "$OUTPUT_ROOT" <<'PY'
import hashlib
import json
import os
import sys
from pathlib import Path

receipt = Path(sys.argv[1])
payload = {
    "schema_version": 1,
    "source_commit": sys.argv[2],
    "task5_parent_job_id": sys.argv[3],
    "task5_manifest_sha256": sys.argv[4],
    "task9_a_job_id": sys.argv[5],
    "task9_a_output_root": sys.argv[6],
}
raw = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
payload["receipt_sha256"] = hashlib.sha256(raw).hexdigest()
encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode() + b"\n"
with open(receipt, "xb") as stream:
    stream.write(encoded)
    stream.flush()
    os.fsync(stream.fileno())
directory = os.open(receipt.parent, os.O_RDONLY)
try:
    os.fsync(directory)
finally:
    os.close(directory)
PY
echo "$task9_job_id"
