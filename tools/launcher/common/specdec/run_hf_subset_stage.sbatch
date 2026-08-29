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

set -euo pipefail

# Shared parallel storage is mounted at /lustre on OCI-HSG, Ptyche and Lyris
# and at /scratch/fsw on AWS-CMH and OCI-AGA, so requiring one prefix refuses
# to run on two of the five clusters. Name the storage a durable artifact must
# NOT live on instead -- node-local scratch that vanishes with the job, and the
# NFS home the MARS guidance reserves for source.
is_durable_path() {
    case "${1:-}" in
        /home/*|/raid/*|/tmp/*|/var/*|/cm/*|/dev/shm/*) return 1 ;;
        /*) return 0 ;;
        *) return 1 ;;
    esac
}

required=(REPO_ROOT SOURCE_COMMIT PROFILE PROFILE_SHA256 PLAN_PATH PLAN_SHA256 IMAGE IMAGE_SHA256 OUTPUT_ROOT READINESS_RECEIPT READINESS_RECEIPT_SHA256)
for name in "${required[@]}"; do
    [[ -n "${!name:-}" ]] || { echo "missing environment variable: $name" >&2; exit 2; }
done
[[ "$REPO_ROOT" == /home/* && "$PLAN_PATH" == "$REPO_ROOT"/* ]] || exit 2
for durable in OUTPUT_ROOT READINESS_RECEIPT IMAGE; do
    is_durable_path "${!durable}" || { echo "$durable must be on shared storage, got: ${!durable}" >&2; exit 2; }
done
[[ "$SOURCE_COMMIT" =~ ^[0-9a-f]{40}$ && "$PLAN_SHA256" =~ ^[0-9a-f]{64}$ ]] || exit 2
[[ "${SLURM_JOB_ID:-}" =~ ^[0-9]+$ ]] || { echo "invalid Slurm job identity" >&2; exit 2; }

[[ "$(git -C "$REPO_ROOT" rev-parse HEAD)" == "$SOURCE_COMMIT" ]] || { echo "source commit mismatch" >&2; exit 2; }
[[ -z "$(git -C "$REPO_ROOT" status --porcelain)" ]] || { echo "source checkout is dirty" >&2; exit 2; }
[[ "$(sha256sum "$PLAN_PATH" | cut -d' ' -f1)" == "$PLAN_SHA256" ]] || { echo "plan SHA-256 mismatch" >&2; exit 2; }
[[ "$(sha256sum "$PROFILE" | cut -d' ' -f1)" == "$PROFILE_SHA256" ]] || { echo "profile SHA-256 mismatch" >&2; exit 2; }
[[ "$(sha256sum "$IMAGE" | cut -d' ' -f1)" == "$IMAGE_SHA256" ]] || { echo "container SHA-256 mismatch" >&2; exit 2; }
[[ "$(sha256sum "$READINESS_RECEIPT" | cut -d' ' -f1)" == "$READINESS_RECEIPT_SHA256" ]] || { echo "readiness receipt SHA-256 mismatch" >&2; exit 2; }
PYTHONPATH="$REPO_ROOT/tools/launcher${PYTHONPATH:+:$PYTHONPATH}" python3 - "$PROFILE" "$READINESS_RECEIPT" <<'PY'
import json
import sys
from pathlib import Path

from common.specdec.cluster_profile import load_cluster_profile
from common.specdec.qwen4b_study_manifest import validate_readiness_receipt

profile = load_cluster_profile(Path(sys.argv[1]))
receipt = json.loads(Path(sys.argv[2]).read_text())
validate_readiness_receipt(
    receipt,
    profile=profile.name,
    account=profile.account,
    partition=profile.partition,
    scratch_root=Path(receipt.get("scratch_root", "")),
)
PY

scratch="/raid/scratch/${USER}/${SLURM_JOB_ID}"
mkdir -p "$scratch" "$(dirname "$OUTPUT_ROOT")"
trap 'rm -rf -- "$scratch"' EXIT

mounts="$REPO_ROOT:$REPO_ROOT,$scratch:$scratch,$(dirname "$OUTPUT_ROOT"):$(dirname "$OUTPUT_ROOT")"
srun --container-image="$IMAGE" --container-mounts="$mounts" --container-workdir="$REPO_ROOT" \
    env PYTHONPATH="$REPO_ROOT/tools/launcher" python3 -m common.specdec.stage_hf_subset \
    --plan "$PLAN_PATH" --output-root "$OUTPUT_ROOT" --scratch-root "$scratch"
