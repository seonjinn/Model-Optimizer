#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

usage() {
    echo "usage: $0 --test-only --manifest PATH --readiness PATH --receipt PATH" >&2
    exit 2
}

test_only=false
manifest_path=""
readiness_path=""
receipt_path=""
while (( $# )); do
    case "$1" in
        --test-only) test_only=true; shift ;;
        --manifest) manifest_path="${2:-}"; shift 2 ;;
        --readiness) readiness_path="${2:-}"; shift 2 ;;
        --receipt) receipt_path="${2:-}"; shift 2 ;;
        *) usage ;;
    esac
done

"$test_only" || { echo "production submission is not supported by the B canary submitter" >&2; exit 2; }
[[ -f "$manifest_path" && -f "$readiness_path" && -n "$receipt_path" ]] || usage

launcher_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
runner="$launcher_root/common/specdec/run_qwen4b_b_canary.sbatch"
readiness_sha256="$(sha256sum "$readiness_path" | cut -d' ' -f1)"
settings=()
while IFS= read -r setting; do
    settings+=("$setting")
done < <(PYTHONPATH="$launcher_root${PYTHONPATH:+:$PYTHONPATH}" python3 - \
    "$manifest_path" "$readiness_path" <<'PY'
import json
import sys
from pathlib import Path

from common.specdec.qwen4b_b_canary_manifest import load_b_canary_manifest

manifest = load_b_canary_manifest(Path(sys.argv[1]))
readiness = json.loads(Path(sys.argv[2]).read_text())
if readiness.get("ready") is not True:
    raise SystemExit("B readiness receipt is blocked")
if readiness.get("receipt_sha256") != manifest.readiness_receipt_sha256:
    raise SystemExit("B readiness manifest identity mismatch")
for value in (manifest.topology.account, manifest.topology.partition, manifest.source_commit):
    print(value)
PY
)
(( ${#settings[@]} == 3 )) || { echo "unable to resolve B canary submission" >&2; exit 2; }
account="${settings[0]}"
partition="${settings[1]}"
source_commit="${settings[2]}"
case "$account" in
    nemotron_sw_post|nemotron_n4_post) ;;
    *) echo "unapproved B canary account" >&2; exit 2 ;;
esac

sbatch --test-only --account="$account" --partition="$partition" --nodes=16 --segment=16 \
    --ntasks-per-node=4 --gpus-per-node=4 --cpus-per-task=96 "$runner" >/dev/null

receipt_parent="$(dirname "$receipt_path")"
mkdir -p "$receipt_parent"
[[ ! -e "$receipt_path" ]] || { echo "submission receipt already exists" >&2; exit 2; }
python3 - "$receipt_path" "$manifest_path" "$readiness_path" "$readiness_sha256" "$source_commit" <<'PY'
import hashlib
import json
import os
import sys
from pathlib import Path

path = Path(sys.argv[1])
record = {
    "manifest_path": sys.argv[2],
    "manifest_sha256": hashlib.sha256(Path(sys.argv[2]).read_bytes()).hexdigest(),
    "readiness_path": sys.argv[3],
    "readiness_sha256": sys.argv[4],
    "source_commit": sys.argv[5],
    "test_only_passed": True,
    "production_submitted": False,
}
temporary = path.with_name(f".{path.name}.partial-{os.getpid()}")
temporary.write_text(json.dumps(record, sort_keys=True) + "\n")
os.replace(temporary, path)
PY
