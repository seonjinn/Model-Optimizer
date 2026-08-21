#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Validate one drafter cluster profile without placing a real workload in the queue.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LAUNCHER_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
MODE="outer"
PROFILE=""
OUTPUT=""
DRY_RUN=0

usage() {
    echo "usage: $0 --profile PATH --output PATH [--dry-run] [--inside]" >&2
    exit 2
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --profile) PROFILE="$2"; shift 2 ;;
        --output) OUTPUT="$2"; shift 2 ;;
        --dry-run) DRY_RUN=1; shift ;;
        --inside) MODE="inside"; shift ;;
        *) usage ;;
    esac
done

[[ -n "$PROFILE" && -n "$OUTPUT" ]] || usage

profile_data="$(PYTHONPATH="$LAUNCHER_ROOT${PYTHONPATH:+:$PYTHONPATH}" python3 - "$PROFILE" "$OUTPUT" <<'PY'
import sys
from pathlib import Path

from common.specdec.cluster_profile import load_cluster_profile, render_probe_sbatch

profile_path = Path(sys.argv[1]).resolve()
output = Path(sys.argv[2]).resolve()
profile = load_cluster_profile(profile_path)
if not output.is_relative_to(profile.durable_root):
    raise ValueError("output must be under the profile durable_root")
print(profile_path)
print(output)
print(profile.name)
print(profile.account)
print(profile.partition)
for argument in render_probe_sbatch(profile):
    print(argument)
PY
)"
profile_lines=()
while IFS= read -r profile_line; do
    profile_lines+=("$profile_line")
done <<<"$profile_data"
[[ "${#profile_lines[@]}" -gt 6 ]] || { echo "invalid profile data" >&2; exit 1; }
PROFILE="${profile_lines[0]}"
OUTPUT="${profile_lines[1]}"
PROFILE_NAME="${profile_lines[2]}"
ACCOUNT="${profile_lines[3]}"
PARTITION="${profile_lines[4]}"
SBATCH_ARGS=("${profile_lines[@]:5}")

if [[ "$MODE" == "outer" ]]; then
    sacctmgr --noheader --parsable2 show assoc where "user=$USER" "account=$ACCOUNT" format=Account,Partition \
        | awk -F'|' -v account="$ACCOUNT" -v partition="$PARTITION" '$1 == account && ($2 == partition || $2 == "") { found = 1 } END { exit !found }'
    scontrol show partition "$PARTITION" >/dev/null
    sbatch --test-only "${SBATCH_ARGS[@]}" "$0" --inside --profile "$PROFILE" --output "$OUTPUT"
    [[ "$DRY_RUN" -eq 0 ]] || exit 0
    submitted="$(sbatch --parsable "${SBATCH_ARGS[@]}" "$0" --inside --profile "$PROFILE" --output "$OUTPUT" || true)"
    job_id="${submitted%%;*}"
    if [[ -z "$job_id" ]]; then
        job_id="$(squeue -h -n "drafter-profile-probe-${PROFILE_NAME}" -o "%A" | head -n 1 || true)"
    fi
    [[ "$job_id" =~ ^[0-9]+$ ]] || { echo "scheduler did not confirm profile probe submission" >&2; exit 1; }
    printf '%s\n' "$job_id"
    exit 0
fi

scratch_root="$(PYTHONPATH="$LAUNCHER_ROOT${PYTHONPATH:+:$PYTHONPATH}" python3 - "$PROFILE" <<'PY'
import os
import sys
import tempfile
from pathlib import Path

from common.specdec.cluster_profile import load_cluster_profile, select_scratch_root

profile = load_cluster_profile(Path(sys.argv[1]))

def writable(path: Path) -> bool:
    try:
        path.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=path):
            pass
    except OSError:
        return False
    return True

print(select_scratch_root(profile.scratch_candidates, os.environ, writable))
PY
)"
mkdir -p "$scratch_root"
[[ "$(nvidia-smi -L | wc -l | tr -d ' ')" == "4" ]] || { echo "expected four visible GPUs" >&2; exit 1; }
case "$(uname -m)" in
    aarch64) ;;
    *) echo "expected aarch64 compute node" >&2; exit 1 ;;
esac
srun --help | grep -q -- '--container-image' || { echo "Pyxis is unavailable" >&2; exit 1; }

mkdir -p "$(dirname "$OUTPUT")"
receipt_tmp="$(mktemp "${OUTPUT}.partial.XXXXXX")"
trap 'rm -f "$receipt_tmp"' EXIT
python3 - "$receipt_tmp" "$PROFILE_NAME" "$scratch_root" "$ACCOUNT" "$PARTITION" <<'PY'
import json
import socket
import sys
from datetime import UTC, datetime
from pathlib import Path

receipt = {
    "profile": sys.argv[2],
    "scratch_root": sys.argv[3],
    "account": sys.argv[4],
    "partition": sys.argv[5],
    "pyxis_available": True,
    "gpu_count": 4,
    "architecture": "aarch64",
    "hostname": socket.gethostname(),
    "timestamp": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
}
Path(sys.argv[1]).write_text(json.dumps(receipt, sort_keys=True) + "\n")
PY
mv "$receipt_tmp" "$OUTPUT"
trap - EXIT
echo "cluster readiness receipt: $OUTPUT"
