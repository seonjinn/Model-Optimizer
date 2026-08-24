#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

usage() {
    echo "usage: $0 --repo-root PATH --submission-receipt PATH --job-completion PATH --scheduler-observation PATH --authorization PATH" >&2
    exit 2
}

repo_root="" submission="" completion="" observation="" authorization=""
while (( $# )); do
    case "$1" in
        --repo-root) repo_root="${2:-}"; shift 2 ;;
        --submission-receipt) submission="${2:-}"; shift 2 ;;
        --job-completion) completion="${2:-}"; shift 2 ;;
        --scheduler-observation) observation="${2:-}"; shift 2 ;;
        --authorization) authorization="${2:-}"; shift 2 ;;
        *) usage ;;
    esac
done
for path in "$repo_root" "$submission" "$completion" "$observation" "$authorization"; do
    [[ "$path" == /* ]] || usage
done
[[ -d "$repo_root" && -f "$submission" && -f "$completion" ]] || usage
[[ ! -e "$observation" && ! -L "$observation" ]] || exit 2
[[ ! -e "$authorization" && ! -L "$authorization" ]] || exit 2

source_commit="$(python3 - "$submission" <<'PY'
import json
import sys
from pathlib import Path
print(json.loads(Path(sys.argv[1]).read_bytes())["source_commit"])
PY
)"
job_id="$(python3 - "$submission" <<'PY'
import json
import sys
from pathlib import Path
print(json.loads(Path(sys.argv[1]).read_bytes())["expected_gpu_job_id"])
PY
)"
[[ "$source_commit" =~ ^[0-9a-f]{40}$ && "$job_id" =~ ^[0-9]+$ ]] || exit 2
[[ "$(git -C "$repo_root" rev-parse HEAD)" == "$source_commit" ]] || exit 2
[[ -z "$(git -C "$repo_root" status --porcelain)" ]] || exit 2

# The controller, not the caller, obtains the scheduler observation.
sacct_output="$(sacct -X -j "$job_id" -n -P \
    -o JobIDRaw,State,ExitCode,Account,JobName,Start,End,Comment,StdOut)"
readonly sacct_output
PYTHONPATH="$repo_root/tools/launcher${PYTHONPATH:+:$PYTHONPATH}" python3 - \
    "$authorization" "$observation" "$submission" "$completion" "$sacct_output" <<'PY'
import sys
from pathlib import Path

from common.specdec.qwen4b_a_canary_manifest import finalize_a_authorization

finalize_a_authorization(
    Path(sys.argv[1]),
    Path(sys.argv[2]),
    submission_receipt_path=Path(sys.argv[3]),
    job_completion_path=Path(sys.argv[4]),
    sacct_output=sys.argv[5],
)
PY
