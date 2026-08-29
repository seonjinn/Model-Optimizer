#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#SBATCH --job-name=q30t-ptv23-canary
#SBATCH --nodes=16
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=96
#SBATCH --time=05:00:00

set -euo pipefail
umask 077

[[ "$#" == 4 ]] || {
    echo "Q30 controller requires descriptor, SHA-256, receipt, and source checkout args" >&2
    exit 2
}
launch_descriptor="$1"
launch_descriptor_sha256="$2"
requested_canary_receipt="$3"
requested_source_checkout="$4"
[[ "$launch_descriptor" == /* && "$launch_descriptor_sha256" =~ ^[0-9a-f]{64}$ \
    && "$requested_canary_receipt" == /* && "$requested_source_checkout" == /* \
    && -d "$requested_source_checkout" && ! -L "$requested_source_checkout" ]] || {
    echo "invalid Q30 launch descriptor arguments" >&2
    exit 2
}
source_checkout="$(cd -- "$requested_source_checkout" && pwd -P)"
[[ "$source_checkout" == "$requested_source_checkout" ]] || {
    echo "Q30 source checkout argument must be canonical" >&2
    exit 2
}
replay_dir="$(/usr/bin/mktemp -d "${TMPDIR:-/tmp}/q30-launch-replay.XXXXXX")"
replay_environment="$replay_dir/environment.bin"
bootstrap_commit_file="$replay_dir/source-commit"
bootstrap_root="$replay_dir/bootstrap"
bootstrap_publisher_oid_file="$replay_dir/publisher-oid"
trap '/bin/rm -rf -- "${replay_dir:?}"' EXIT
/usr/bin/python3 -I - "$launch_descriptor" "$launch_descriptor_sha256" "$source_checkout" \
    "$bootstrap_commit_file" <<'PY'
import hashlib
import json
import os
import stat
import sys
from pathlib import Path

descriptor_path = Path(sys.argv[1])
expected_sha256 = sys.argv[2]
source_checkout = Path(sys.argv[3])
output_path = Path(sys.argv[4])
flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
descriptor = os.open(descriptor_path, flags)
try:
    before = os.fstat(descriptor)
    if not stat.S_ISREG(before.st_mode) or before.st_size > 16 * 1024 * 1024:
        raise SystemExit("Q30 bootstrap descriptor is not a bounded regular file")
    raw = b""
    while chunk := os.read(descriptor, 1024 * 1024):
        raw += chunk
    after = os.fstat(descriptor)
finally:
    os.close(descriptor)
if (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns) != (
    after.st_dev,
    after.st_ino,
    after.st_size,
    after.st_mtime_ns,
):
    raise SystemExit("Q30 bootstrap descriptor changed while reading")
if hashlib.sha256(raw).hexdigest() != expected_sha256:
    raise SystemExit("Q30 bootstrap descriptor SHA-256 mismatch")
try:
    payload = json.loads(raw)
except json.JSONDecodeError as error:
    raise SystemExit("Q30 bootstrap descriptor JSON is invalid") from error
canonical = (json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n").encode()
if (
    not isinstance(payload, dict)
    or set(payload) != {"contract", "schema_version", "stage"}
    or payload.get("schema_version") != "q30t-ptv23-launch-descriptor-v1"
    or payload.get("stage") not in {"canary", "full"}
    or raw != canonical
):
    raise SystemExit("Q30 bootstrap descriptor identity is invalid")
contract = payload.get("contract")
if not isinstance(contract, dict):
    raise SystemExit("Q30 bootstrap descriptor contract is invalid")
commit = contract.get("source_commit")
checkout = contract.get("source_checkout_path")
if (
    not isinstance(commit, str)
    or len(commit) != 40
    or any(character not in "0123456789abcdef" for character in commit)
    or checkout != str(source_checkout)
):
    raise SystemExit("Q30 bootstrap source identity is invalid")
output = os.open(
    output_path,
    os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
    0o600,
)
try:
    encoded = (commit + "\n").encode()
    if os.write(output, encoded) != len(encoded):
        raise SystemExit("Q30 bootstrap source identity write failed")
    os.fsync(output)
finally:
    os.close(output)
PY
IFS= read -r bootstrap_commit < "$bootstrap_commit_file"
[[ "$bootstrap_commit" =~ ^[0-9a-f]{40}$ ]] || {
    echo "invalid Q30 bootstrap source commit" >&2
    exit 2
}
GIT_NO_REPLACE_OBJECTS=1 /usr/bin/git -C "$source_checkout" \
    cat-file -e "${bootstrap_commit}^{commit}"
bootstrap_tree="$replay_dir/bootstrap-tree"
GIT_NO_REPLACE_OBJECTS=1 /usr/bin/git -C "$source_checkout" \
    ls-tree -r -z "$bootstrap_commit" \
    -- tools/launcher/common/specdec > "$bootstrap_tree"
/usr/bin/python3 -I - "$bootstrap_tree" "$bootstrap_root" "$source_checkout" \
    "$bootstrap_publisher_oid_file" <<'PY'
import hashlib
import os
import stat
import subprocess
import sys
from pathlib import Path, PurePosixPath

tree_path, bootstrap_root, source_checkout, publisher_oid_path = map(Path, sys.argv[1:])
bootstrap_root.mkdir(mode=0o700)
records = tree_path.read_bytes().split(b"\0")
seen: set[str] = set()
module_bound = False
publisher_oid = None
for record in records:
    if not record:
        continue
    metadata, separator, raw_path = record.partition(b"\t")
    fields = metadata.split(b" ")
    if not separator or len(fields) != 3:
        raise SystemExit("Q30 bootstrap tree record is malformed")
    mode, object_type, raw_oid = (field.decode("ascii") for field in fields)
    relative_text = os.fsdecode(raw_path)
    relative = PurePosixPath(relative_text)
    if (
        mode not in {"100644", "100755"}
        or object_type != "blob"
        or len(raw_oid) != 40
        or any(character not in "0123456789abcdef" for character in raw_oid)
        or relative.is_absolute()
        or relative.as_posix() != relative_text
        or any(component in {"", ".", ".."} for component in relative.parts)
        or any(ord(character) < 32 or ord(character) == 127 for character in relative_text)
        or relative.parts[:4] != ("tools", "launcher", "common", "specdec")
        or relative_text in seen
    ):
        raise SystemExit("Q30 bootstrap tree contains an unsafe non-regular entry")
    seen.add(relative_text)
    blob = subprocess.run(
        ["/usr/bin/git", "-C", str(source_checkout), "cat-file", "blob", raw_oid],
        check=True,
        env={"GIT_NO_REPLACE_OBJECTS": "1", "PATH": "/usr/bin:/bin"},
        stdout=subprocess.PIPE,
    ).stdout
    observed_oid = hashlib.sha1(
        b"blob " + str(len(blob)).encode("ascii") + b"\0" + blob
    ).hexdigest()
    if observed_oid != raw_oid:
        raise SystemExit("Q30 bootstrap blob content differs from its Git object ID")
    destination = bootstrap_root.joinpath(*relative.parts)
    destination.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    descriptor = os.open(
        destination,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
        0o700 if mode == "100755" else 0o600,
    )
    try:
        view = memoryview(blob)
        while view:
            written = os.write(descriptor, view)
            if written <= 0:
                raise SystemExit("Q30 bootstrap blob write made no progress")
            view = view[written:]
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    module_bound |= relative_text == "tools/launcher/common/specdec/q30t_ptv23_continuation.py"
    if relative_text == "tools/launcher/common/specdec/qwen4b_b_atomic.py":
        publisher_oid = raw_oid
if not module_bound or publisher_oid is None:
    raise SystemExit("Q30 bootstrap authenticator or completion publisher is absent")
descriptor = os.open(
    publisher_oid_path,
    os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
    0o600,
)
try:
    view = memoryview((publisher_oid + "\n").encode("ascii"))
    while view:
        written = os.write(descriptor, view)
        if written <= 0:
            raise SystemExit("Q30 bootstrap publisher identity write made no progress")
        view = view[written:]
    os.fsync(descriptor)
finally:
    os.close(descriptor)
PY
/usr/bin/python3 -I - "$bootstrap_root/tools/launcher" \
    replay-launch-descriptor \
    --descriptor "$launch_descriptor" \
    --sha256 "$launch_descriptor_sha256" \
    --source-checkout "$source_checkout" \
    --environment-output "$replay_environment" <<'PY'
import runpy
import sys

launcher_root = sys.argv.pop(1)
sys.path.insert(0, launcher_root)
sys.argv[0] = "common.specdec.q30t_ptv23_continuation"
runpy.run_module("common.specdec.q30t_ptv23_continuation", run_name="__main__")
PY
while IFS= read -r -d '' name && IFS= read -r -d '' value; do
    [[ "$name" =~ ^[A-Z][A-Z0-9_]*$ ]] || { echo "invalid launch environment name" >&2; exit 2; }
    printf -v "$name" '%s' "$value"
    export "${name?}"
done < "$replay_environment"
/bin/rm -f -- "$replay_environment"
IFS= read -r bootstrap_publisher_oid < "$bootstrap_publisher_oid_file"
[[ "$bootstrap_publisher_oid" =~ ^[0-9a-f]{40}$ ]] || {
    echo "invalid Q30 bootstrap completion publisher identity" >&2
    exit 2
}
bootstrap_publisher_path="$bootstrap_root/tools/launcher/common/specdec/qwen4b_b_atomic.py"
[[ "${CANARY_RECEIPT:-}" == "$requested_canary_receipt" ]] || {
    echo "canary receipt argument does not match authenticated descriptor" >&2
    exit 2
}

required=(
    STAGE METHOD RUN_NAME SOURCE_ARCHIVE SOURCE_ARCHIVE_SHA256 SOURCE_COMMIT
    SOURCE_TREE_OID SOURCE_TREE_SHA256
    TARGET_ARCHIVE TARGET_ARCHIVE_SHA256 TARGET_TREE_SHA256
    PARENT_ARCHIVE PARENT_ARCHIVE_SHA256 PARENT_TREE_SHA256
    RUNTIME_ARCHIVE RUNTIME_ARCHIVE_SHA256 RUNTIME_TREE_SHA256
    RUNTIME_IMAGE RUNTIME_IMAGE_SHA256 DATASET DATASET_SHA256
    ORDERED_PROMPT_UUIDS_SHA256 EXPECTED_EXPOSURE_COUNT
    EXPECTED_FINAL_GLOBAL_BATCH_SIZE EXPECTED_STEPS
    TRAINING_ENTRYPOINT_RELATIVE METHOD_CONFIG_RELATIVE OUTPUT_ROOT CONTROL_ROOT
    CANARY_RECEIPT KEEPER_TOOL KEEPER_TOOL_SHA256 TARGET_MODEL
)
for name in "${required[@]}"; do
    [[ -n "${!name:-}" ]] || { echo "missing Q30 controller input: $name" >&2; exit 2; }
done
case "$STAGE:$EXPECTED_STEPS:$EXPECTED_EXPOSURE_COUNT:$EXPECTED_FINAL_GLOBAL_BATCH_SIZE" in
    canary:20:10240:512|full:1368:700000:96) ;;
    *) echo "invalid Q30 stage schedule" >&2; exit 2 ;;
esac
[[ "$TRAINING_ENTRYPOINT_RELATIVE" == tools/launcher/common/eagle3/train_eagle_streaming.sh ]] || {
    echo "unexpected training entrypoint" >&2
    exit 2
}
case "$METHOD" in
    DFlash) expected_method_config="modelopt_recipes/general/speculative_decoding/dflash.yaml" ;;
    DSpark) expected_method_config="modelopt_recipes/general/speculative_decoding/dspark.yaml" ;;
    *) echo "unexpected method" >&2; exit 2 ;;
esac
[[ "$METHOD_CONFIG_RELATIVE" == "$expected_method_config" ]] || {
    echo "unexpected method config" >&2
    exit 2
}
[[ "${SLURM_NNODES:-}" == 16 && "${SLURM_GPUS_ON_NODE:-4}" == 4 ]] || {
    echo "Q30 controller requires exactly 16 nodes and 4 GPUs per node" >&2
    exit 2
}
[[ "${SLURM_JOB_ID:-}" =~ ^[0-9]+$ ]] || { echo "invalid Slurm job identity" >&2; exit 2; }
for digest_name in SOURCE_ARCHIVE_SHA256 SOURCE_TREE_SHA256 TARGET_ARCHIVE_SHA256 TARGET_TREE_SHA256 \
    PARENT_ARCHIVE_SHA256 PARENT_TREE_SHA256 RUNTIME_ARCHIVE_SHA256 RUNTIME_TREE_SHA256 \
    RUNTIME_IMAGE_SHA256 DATASET_SHA256 ORDERED_PROMPT_UUIDS_SHA256 KEEPER_TOOL_SHA256; do
    [[ "${!digest_name}" =~ ^[0-9a-f]{64}$ ]] || { echo "invalid $digest_name" >&2; exit 2; }
done
[[ "$SOURCE_COMMIT" =~ ^[0-9a-f]{40}$ ]] || { echo "invalid source commit" >&2; exit 2; }
[[ "$SOURCE_TREE_OID" =~ ^[0-9a-f]{40}$ ]] || { echo "invalid source tree OID" >&2; exit 2; }
for path_name in SOURCE_ARCHIVE TARGET_ARCHIVE PARENT_ARCHIVE RUNTIME_ARCHIVE RUNTIME_IMAGE \
    DATASET OUTPUT_ROOT CONTROL_ROOT CANARY_RECEIPT KEEPER_TOOL; do
    [[ "${!path_name}" == /* ]] || { echo "$path_name must be absolute" >&2; exit 2; }
done
[[ ! -e "$OUTPUT_ROOT" && ! -L "$OUTPUT_ROOT" && ! -e "$CONTROL_ROOT" && ! -L "$CONTROL_ROOT" ]] || {
    echo "Q30 canary output and controller roots must be fresh" >&2
    exit 2
}
[[ ! -e "$CANARY_RECEIPT" && ! -L "$CANARY_RECEIPT" ]] || {
    echo "Q30 canary receipt must be fresh" >&2
    exit 2
}
# The allocation starts with sbatch --export=NONE. Only after descriptor replay and
# validation may its authenticated environment cross into nested Slurm steps.
export SLURM_EXPORT_ENV=ALL
mkdir -m 0700 "$CONTROL_ROOT"
mkdir -m 0700 "$CONTROL_ROOT/rendezvous"
mkdir -p "$(dirname "$OUTPUT_ROOT")" "$(dirname "$CANARY_RECEIPT")"

scratch_root="/raid/scratch/${USER}/ptv23-node-keeper-${SLURM_JOB_ID}"
anchor_root="$scratch_root/anchors"
training_scratch_root="/raid/scratch/${USER}/q30t-training-${SLURM_JOB_ID}"
stop_file="$CONTROL_ROOT/stop-keepers"
container_name="q30t-ptv23-${SLURM_JOB_ID}"
container_sentinel="/q30-container-sentinel.json"
export scratch_root anchor_root training_scratch_root stop_file container_name container_sentinel

keeper_task() {
    set -euo pipefail
    python3 -I - <<'PY'
import os
if not isinstance(getattr(os, "O_TMPFILE", None), int):
    raise SystemExit("node-local keeper requires O_TMPFILE")
PY
    local node="${SLURMD_NODENAME:?}"
    local local_root="$scratch_root"
    [[ ! -e "$local_root" && ! -L "$local_root" ]] || {
        echo "node-local keeper root is not fresh" >&2
        exit 2
    }
    mkdir -p "$local_root"
    chmod 0700 "$local_root"
    [[ ! -e "$training_scratch_root" && ! -L "$training_scratch_root" ]] || {
        echo "node-local training root is not fresh" >&2
        exit 2
    }
    mkdir -m 0700 "$training_scratch_root"
    local local_tool="$local_root/ptv23_node_keeper.py"
    cp --reflink=never -- "$KEEPER_TOOL" "$local_tool"
    [[ "$(sha256sum "$local_tool" | cut -d' ' -f1)" == "$KEEPER_TOOL_SHA256" ]] || {
        echo "keeper bootstrap copy mismatch" >&2
        exit 2
    }
    local fifo="$local_root/controller.fifo"
    mkfifo -m 0600 "$fifo"
    local plan="$local_root/plan.json"
    local receipt="$CONTROL_ROOT/keeper.${node}.json"
    KEEPER_PLAN="$plan" KEEPER_FIFO="$fifo" KEEPER_NODE="$node" python3 -I - <<'PY'
import importlib.util
import os
import sys
from pathlib import Path

tool = Path(os.environ["scratch_root"]) / "ptv23_node_keeper.py"
spec = importlib.util.spec_from_file_location("q30_keeper", tool)
if spec is None or spec.loader is None:
    raise SystemExit("keeper bootstrap import unavailable")
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)
sources = {
    "dataset": (os.environ["DATASET"], os.environ["DATASET_SHA256"]),
    "parent-archive": (os.environ["PARENT_ARCHIVE"], os.environ["PARENT_ARCHIVE_SHA256"]),
    "runtime-archive": (os.environ["RUNTIME_ARCHIVE"], os.environ["RUNTIME_ARCHIVE_SHA256"]),
    "runtime-image": (os.environ["RUNTIME_IMAGE"], os.environ["RUNTIME_IMAGE_SHA256"]),
    "source-archive": (os.environ["SOURCE_ARCHIVE"], os.environ["SOURCE_ARCHIVE_SHA256"]),
    "target-archive": (os.environ["TARGET_ARCHIVE"], os.environ["TARGET_ARCHIVE_SHA256"]),
}
roots = tuple(sorted({Path(path).parent for path, _ in sources.values()}, key=str))
plan = module.KeeperPlan(
    schema_version="ptv23-node-keeper-v1",
    job_id=os.environ["SLURM_JOB_ID"],
    node_name=os.environ["KEEPER_NODE"],
    scratch_root=Path(os.environ["scratch_root"]),
    directory_roots=roots,
    anchor_root=Path(os.environ["anchor_root"]),
    fifo_path=Path(os.environ["KEEPER_FIFO"]),
    sources=tuple(
        module.KeeperSource(name, Path(path), digest)
        for name, (path, digest) in sorted(sources.items())
    ),
)
Path(os.environ["KEEPER_PLAN"]).write_bytes(plan.canonical_bytes() + b"\n")
PY
    python3 -I "$local_tool" serve --plan "$plan" --receipt "$receipt" &
    local keeper_pid=$!
    for _ in $(seq 1 1800); do [[ -f "$receipt" ]] && break; kill -0 "$keeper_pid" 2>/dev/null || exit 2; sleep 1; done
    [[ -f "$receipt" ]] || { echo "keeper readiness timed out" >&2; exit 2; }
    exec 9>"$fifo"
    KEEPER_RECEIPT="$receipt" python3 -I - <<'PY'
import importlib.util
import json
import os
import sys
from pathlib import Path

tool = Path(os.environ["scratch_root"]) / "ptv23_node_keeper.py"
spec = importlib.util.spec_from_file_location("q30_keeper_validate", tool)
if spec is None or spec.loader is None:
    raise SystemExit("keeper validation import unavailable")
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)
receipt = module.validate_keeper_receipt(Path(os.environ["KEEPER_RECEIPT"]))
evidence = {
    "job_id": receipt.job_id,
    "node_name": receipt.node_name,
    "receipt_sha256": receipt.receipt_sha256,
    "schema_version": "q30t-node-evidence-v1",
}
target = Path(os.environ["CONTROL_ROOT"]) / f"node-evidence.keeper.{receipt.node_name}.json"
target.write_text(json.dumps(evidence, sort_keys=True, separators=(",", ":")) + "\n")
PY
    while [[ ! -f "$stop_file" ]]; do kill -0 "$keeper_pid" 2>/dev/null || exit 2; sleep 2; done
    printf 'stop\n' >&9
    exec 9>&-
    wait "$keeper_pid"
}
keeper_task_command="$(declare -f keeper_task)"$'\nkeeper_task'

srun --overlap --nodes=16 --ntasks=16 --ntasks-per-node=1 \
    --cpus-per-task=2 --kill-on-bad-exit=1 /bin/bash -c "$keeper_task_command" &
keeper_step_pid=$!
probe_step_pid=""
training_step_pid=""
reconciliation_step_pid=""
cleanup_steps() {
    if [[ -n "$probe_step_pid" ]] && kill -0 "$probe_step_pid" 2>/dev/null; then
        kill -TERM -- "-$probe_step_pid" 2>/dev/null || true
        wait "$probe_step_pid" 2>/dev/null || true
    fi
    if [[ -n "$training_step_pid" ]] && kill -0 "$training_step_pid" 2>/dev/null; then
        kill -TERM -- "-$training_step_pid" 2>/dev/null || true
        wait "$training_step_pid" 2>/dev/null || true
    fi
    if [[ -n "$reconciliation_step_pid" ]] && kill -0 "$reconciliation_step_pid" 2>/dev/null; then
        kill -TERM -- "-$reconciliation_step_pid" 2>/dev/null || true
        wait "$reconciliation_step_pid" 2>/dev/null || true
    fi
    if [[ -n "$keeper_step_pid" ]]; then
        touch "$stop_file" 2>/dev/null || true
        wait "$keeper_step_pid" 2>/dev/null || true
    fi
    /bin/rm -rf -- "${replay_dir:?}"
}
trap cleanup_steps EXIT

keeper_ready=0
for _ in $(seq 1 1800); do
    shopt -s nullglob
    keeper_evidence_paths=("$CONTROL_ROOT"/node-evidence.keeper.*.json)
    shopt -u nullglob
    if ((${#keeper_evidence_paths[@]} > 16)); then
        echo "keeper evidence exceeds exactly 16 nodes" >&2
        exit 2
    fi
    if ((${#keeper_evidence_paths[@]} == 16)); then
        keeper_ready=1
        break
    fi
    if ! kill -0 "$keeper_step_pid" 2>/dev/null; then
        wait "$keeper_step_pid" 2>/dev/null || true
        keeper_step_pid=""
        echo "keeper exited before all 16 readiness receipts; completion suppressed" >&2
        exit 2
    fi
    sleep 1
done
[[ "$keeper_ready" == 1 ]] || {
    echo "keeper evidence did not reach exactly 16 unique nodes" >&2
    exit 2
}

python3 - "$CONTROL_ROOT" "$SLURM_JOB_ID" <<'PY'
import json
import sys
from pathlib import Path

root, job = Path(sys.argv[1]), sys.argv[2]
paths = sorted(root.glob("node-evidence.keeper.*.json"))
if len(paths) != 16:
    raise SystemExit("keeper evidence did not reach exactly 16 unique nodes")
records = [json.loads(path.read_text()) for path in paths]
nodes = {record.get("node_name") for record in records}
if len(nodes) != 16 or any(record.get("job_id") != job for record in records):
    raise SystemExit("keeper evidence did not reconcile exactly 16 unique nodes")
PY

python3 - "$anchor_root" "$training_scratch_root" "$(dirname "$OUTPUT_ROOT")" \
    "$CONTROL_ROOT" /q30/input /scratchspace <<'PY'
import sys

for path in sys.argv[1:]:
    if not path.startswith("/") or any(
        character in {",", ":"} or ord(character) < 32 or ord(character) == 127
        for character in path
    ):
        raise SystemExit("interpolated Pyxis mount path is unsafe")
PY

mounts="${anchor_root}/source-archive:/q30/input/source.tar.zst:ro,${anchor_root}/target-archive:/q30/input/target.tar.zst:ro,${anchor_root}/parent-archive:/q30/input/parent.tar.zst:ro,${anchor_root}/runtime-archive:/q30/input/runtime.tar.zst:ro,${anchor_root}/runtime-image:/q30/input/runtime.sqsh:ro,${anchor_root}/dataset:/q30/input/DATA.jsonl:ro,${training_scratch_root}:${training_scratch_root},$(dirname "$OUTPUT_ROOT"):$(dirname "$OUTPUT_ROOT"),${CONTROL_ROOT}:${CONTROL_ROOT},${CONTROL_ROOT}/rendezvous:/scratchspace"

container_probe_task() {
    set -euo pipefail
    local node="${SLURMD_NODENAME:?}"
    local local_root="$training_scratch_root/run"
    [[ ! -e "$local_root" && ! -L "$local_root" ]] || exit 2
    mkdir -m 0700 "$local_root"
    for name in source target parent runtime; do mkdir -m 0700 "$local_root/$name"; done
    [[ "$(sha256sum /q30/input/runtime.sqsh | cut -d' ' -f1)" == "$RUNTIME_IMAGE_SHA256" ]] || {
        echo "whole SQSH identity mismatch" >&2
        exit 2
    }
    tar --zstd -xf /q30/input/source.tar.zst -C "$local_root/source"
    tar --zstd -xf /q30/input/target.tar.zst -C "$local_root/target"
    tar --zstd -xf /q30/input/parent.tar.zst -C "$local_root/parent"
    tar --zstd -xf /q30/input/runtime.tar.zst -C "$local_root/runtime"
    cp --reflink=never -- /q30/input/DATA.jsonl "$local_root/DATA.jsonl"
    [[ "$(sha256sum "$local_root/DATA.jsonl" | cut -d' ' -f1)" == "$DATASET_SHA256" ]] || exit 2
    PYTHONPATH="$local_root/source/tools/launcher" python3 - "$local_root" <<'PY'
import os
import sys
from pathlib import Path
from common.specdec.q30t_ptv23_continuation import (
    _model_tree_sha256,
    _source_checkout_identity,
    _tree_sha256,
)

root = Path(sys.argv[1])
source_tree_oid, source_tree_sha256 = _source_checkout_identity(
    root / "source", expected_commit=os.environ["SOURCE_COMMIT"]
)
if (
    source_tree_oid != os.environ["SOURCE_TREE_OID"]
    or source_tree_sha256 != os.environ["SOURCE_TREE_SHA256"]
):
    raise SystemExit("extracted source commit/tree inventory mismatch")
for name, expected in (
    ("target", os.environ["TARGET_TREE_SHA256"]),
    ("parent", os.environ["PARENT_TREE_SHA256"]),
):
    if _model_tree_sha256(root / name) != expected:
        raise SystemExit(f"{name} runtime attestation mismatch")
if _tree_sha256(root / "runtime") != os.environ["RUNTIME_TREE_SHA256"]:
    raise SystemExit("runtime attestation mismatch")
PY
    [[ -x "$local_root/runtime/bin/python" ]] || { echo "runtime attestation missing Python" >&2; exit 2; }
    local imported
    imported="$(PYTHONPATH="$local_root/source" "$local_root/runtime/bin/python" - <<'PY'
import modelopt
import modelopt.torch.speculative

print(modelopt.__file__)
PY
)"
    [[ "$imported" == "$local_root/source"/* ]] || { echo "modelopt.__file__ import attestation failed" >&2; exit 2; }
    PROBE_NODE="$node" python3 -I - <<'PY'
import hashlib
import json
import os
from pathlib import Path

body = {
    "job_id": os.environ["SLURM_JOB_ID"],
    "node_name": os.environ["PROBE_NODE"],
    "runtime_image_sha256": os.environ["RUNTIME_IMAGE_SHA256"],
    "runtime_tree_sha256": os.environ["RUNTIME_TREE_SHA256"],
    "schema_version": "q30t-named-container-sentinel-v1",
    "source_archive_sha256": os.environ["SOURCE_ARCHIVE_SHA256"],
    "source_commit": os.environ["SOURCE_COMMIT"],
    "source_tree_oid": os.environ["SOURCE_TREE_OID"],
    "source_tree_sha256": os.environ["SOURCE_TREE_SHA256"],
}
raw = (json.dumps(body, sort_keys=True, separators=(",", ":")) + "\n").encode()
sentinel = Path(os.environ["container_sentinel"])
with sentinel.open("xb") as stream:
    stream.write(raw)
    stream.flush()
    os.fsync(stream.fileno())
evidence = dict(body)
evidence["schema_version"] = "q30t-named-container-evidence-v1"
evidence["sentinel_sha256"] = hashlib.sha256(raw).hexdigest()
target = Path(os.environ["CONTROL_ROOT"]) / f"node-evidence.container.{body['node_name']}.json"
with target.open("x") as stream:
    json.dump(evidence, stream, sort_keys=True, separators=(",", ":"))
    stream.write("\n")
PY
}
container_probe_command="$(declare -f container_probe_task)"$'\ncontainer_probe_task'

set -m
srun --overlap --nodes=16 --ntasks=16 --ntasks-per-node=1 --gpus-per-task=4 \
    --cpus-per-task=96 --kill-on-bad-exit=1 --no-container-mount-home \
    --container-name="$container_name" --container-image="${anchor_root}/runtime-image" \
    --container-mounts="$mounts" /bin/bash -c "$container_probe_command" &
probe_step_pid=$!
set +m

while kill -0 "$keeper_step_pid" 2>/dev/null && kill -0 "$probe_step_pid" 2>/dev/null; do
    sleep 1
done
if ! kill -0 "$keeper_step_pid" 2>/dev/null; then
    kill -TERM -- "-$probe_step_pid" 2>/dev/null || true
    wait "$probe_step_pid" 2>/dev/null || true
    wait "$keeper_step_pid" 2>/dev/null || true
    keeper_step_pid=""
    echo "keeper exited during container probe; completion suppressed" >&2
    exit 2
fi
set +e
wait "$probe_step_pid"
probe_status=$?
set -e
probe_step_pid=""
if [[ "$probe_status" -ne 0 ]]; then
    touch "$stop_file"
    wait "$keeper_step_pid" 2>/dev/null || true
    keeper_step_pid=""
    echo "container probe failed; completion suppressed" >&2
    exit "$probe_status"
fi

python3 - "$CONTROL_ROOT" "$SLURM_JOB_ID" <<'PY'
import hashlib
import json
import os
import sys
from pathlib import Path

root, job = Path(sys.argv[1]), sys.argv[2]
paths = sorted(root.glob("node-evidence.container.*.json"))
if len(paths) != 16:
    raise SystemExit("named-container evidence must reconcile exactly 16 nodes")
records = [json.loads(path.read_text()) for path in paths]
nodes = {record.get("node_name") for record in records}
if len(nodes) != 16 or any(
    set(record)
    != {
        "job_id",
        "node_name",
        "runtime_image_sha256",
        "runtime_tree_sha256",
        "schema_version",
        "sentinel_sha256",
        "source_archive_sha256",
        "source_commit",
        "source_tree_oid",
        "source_tree_sha256",
    }
    or record.get("schema_version") != "q30t-named-container-evidence-v1"
    or record.get("job_id") != job
    or record.get("runtime_image_sha256") != os.environ["RUNTIME_IMAGE_SHA256"]
    or record.get("runtime_tree_sha256") != os.environ["RUNTIME_TREE_SHA256"]
    or record.get("source_archive_sha256") != os.environ["SOURCE_ARCHIVE_SHA256"]
    or record.get("source_commit") != os.environ["SOURCE_COMMIT"]
    or record.get("source_tree_oid") != os.environ["SOURCE_TREE_OID"]
    or record.get("source_tree_sha256") != os.environ["SOURCE_TREE_SHA256"]
    or not isinstance(record.get("sentinel_sha256"), str)
    or len(record["sentinel_sha256"]) != 64
    or any(character not in "0123456789abcdef" for character in record["sentinel_sha256"])
    for record in records
):
    raise SystemExit("named-container source/runtime/sentinel identity does not reconcile")
ordered = sorted([record["node_name"], record["sentinel_sha256"]] for record in records)
body = {
    "job_id": job,
    "node_count": len(ordered),
    "ordered_node_sentinel_sha256": hashlib.sha256(
        json.dumps(ordered, separators=(",", ":")).encode()
    ).hexdigest(),
    "schema_version": "q30t-named-container-reconciliation-v1",
}
with (root / "container-evidence.json").open("x") as stream:
    json.dump(body, stream, sort_keys=True, separators=(",", ":"))
    stream.write("\n")
    stream.flush()
    os.fsync(stream.fileno())
PY

training_task() {
    set -euo pipefail
    local node="${SLURMD_NODENAME:?}"
    local local_root="$training_scratch_root/run"
    PROBE_NODE="$node" python3 -I - <<'PY'
import hashlib
import json
import os
from pathlib import Path
sentinel = Path(os.environ["container_sentinel"])
raw = sentinel.read_bytes()
body = json.loads(raw)
expected = {
    "job_id": os.environ["SLURM_JOB_ID"],
    "node_name": os.environ["PROBE_NODE"],
    "runtime_image_sha256": os.environ["RUNTIME_IMAGE_SHA256"],
    "runtime_tree_sha256": os.environ["RUNTIME_TREE_SHA256"],
    "schema_version": "q30t-named-container-sentinel-v1",
    "source_archive_sha256": os.environ["SOURCE_ARCHIVE_SHA256"],
    "source_commit": os.environ["SOURCE_COMMIT"],
    "source_tree_oid": os.environ["SOURCE_TREE_OID"],
    "source_tree_sha256": os.environ["SOURCE_TREE_SHA256"],
}
if body != expected or raw != (json.dumps(expected, sort_keys=True, separators=(",", ":")) + "\n").encode():
    raise SystemExit("named container sentinel changed before training")
evidence = Path(os.environ["CONTROL_ROOT"]) / f"node-evidence.container.{body['node_name']}.json"
record = json.loads(evidence.read_text())
if record.get("sentinel_sha256") != hashlib.sha256(raw).hexdigest():
    raise SystemExit("named container sentinel receipt mismatch")
PY
    python3 - "$CONTROL_ROOT/node-evidence.training.${node}.json" "$node" <<'PY'
import json
import os
import sys
from pathlib import Path
record = {
    "job_id": os.environ["SLURM_JOB_ID"],
    "node_name": sys.argv[2],
    "runtime_tree_sha256": os.environ["RUNTIME_TREE_SHA256"],
    "schema_version": "q30t-training-node-evidence-v1",
    "source_commit": os.environ["SOURCE_COMMIT"],
    "source_tree_oid": os.environ["SOURCE_TREE_OID"],
    "source_tree_sha256": os.environ["SOURCE_TREE_SHA256"],
}
Path(sys.argv[1]).write_text(json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n")
PY
    for _ in $(seq 1 1800); do
        [[ "$(find "$CONTROL_ROOT" -maxdepth 1 -name 'node-evidence.training.*.json' | wc -l | tr -d ' ')" == 16 ]] && break
        sleep 1
    done
    [[ "$(find "$CONTROL_ROOT" -maxdepth 1 -name 'node-evidence.training.*.json' | wc -l | tr -d ' ')" == 16 ]] || {
        echo "runtime attestation barrier did not reach exactly 16 nodes" >&2
        exit 2
    }
    export HF_MODEL_CKPT="$local_root/target"
    export MODELOPT_RUNTIME="$local_root/runtime"
    export PYTHONPATH="$local_root/source"
    export EAGLE_CAPTURE_IDS='[2,13,24,35,46,48]'
    export SERVE_NODES=8 SERVE_TP=2 SERVE_REPLICAS_PER_NODE=2
    export SERVE_BLOCK_SIZE=32 SERVE_MAX_MODEL_LEN=8192 SERVE_READY_TIMEOUT=1800
    export STREAMING_NUM_WORKERS=0 \
        EXPORT_PATH="$OUTPUT_ROOT/exported-checkpoint-${EXPECTED_STEPS}"
    export Q30_STRICT_EXPOSURE=1 Q30_EXPOSURE_EVIDENCE_DIR=/scratchspace/exposure-evidence
    cd "$local_root/source/tools/launcher"
    local command_file="$local_root/trainer-command.bin"
    PYTHONPATH="$local_root/source/tools/launcher" "$local_root/runtime/bin/python" - \
        "$METHOD" "$local_root" "$OUTPUT_ROOT" "$RUN_NAME" "$STAGE" \
        > "$command_file" <<'PY'
import sys
from pathlib import Path
from common.specdec.q30t_ptv23_continuation import build_q30_node_local_training_command

command = build_q30_node_local_training_command(
    method=sys.argv[1],
    local_root=Path(sys.argv[2]),
    output_root=Path(sys.argv[3]),
    run_name=sys.argv[4],
    stage=sys.argv[5],
)
for argument in command:
    sys.stdout.buffer.write(argument.encode() + b"\0")
PY
    local -a command=()
    mapfile -d '' -t command < "$command_file"
    rm -f -- "$command_file"
    [[ "${#command[@]}" -gt 4 && "${command[0]}" == /bin/bash ]] || {
        echo "node-local trainer command is invalid" >&2
        exit 2
    }
    "${command[@]}"
    if [[ "${SLURM_NODEID:?}" == "$SERVE_NODES" ]]; then
        "$local_root/runtime/bin/python" - \
            "$local_root/parent" "$EXPORT_PATH/model.safetensors" \
            "$CONTROL_ROOT/export-validation.json" "$METHOD" <<'PY'
# Q30_EXPORT_SAFETENSORS_VALIDATOR
import hashlib
import json
import os
import stat
import sys
from pathlib import Path


DTYPE_BYTES = {
    "BOOL": 1,
    "F8_E4M3": 1,
    "F8_E5M2": 1,
    "I8": 1,
    "U8": 1,
    "BF16": 2,
    "F16": 2,
    "I16": 2,
    "U16": 2,
    "F32": 4,
    "I32": 4,
    "U32": 4,
    "F64": 8,
    "I64": 8,
    "U64": 8,
}


def open_regular(path: Path) -> tuple[int, os.stat_result]:
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    metadata = os.fstat(fd)
    if not stat.S_ISREG(metadata.st_mode):
        os.close(fd)
        raise SystemExit(f"safetensors input is not a regular file: {path}")
    return fd, metadata


def exact_json(path: Path) -> object:
    fd, before = open_regular(path)
    try:
        raw = b""
        while chunk := os.read(fd, 1024 * 1024):
            raw += chunk
        after = os.fstat(fd)
    finally:
        os.close(fd)
    if (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns) != (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
    ):
        raise SystemExit(f"metadata changed while reading: {path}")
    try:
        return json.loads(raw)
    except json.JSONDecodeError as error:
        raise SystemExit(f"invalid JSON metadata: {path}") from error


def multiply(shape: object) -> int:
    if not isinstance(shape, list) or any(type(dimension) is not int or dimension < 0 for dimension in shape):
        raise SystemExit("invalid safetensors shape")
    result = 1
    for dimension in shape:
        result *= dimension
    return result


def parse_safetensors(path: Path) -> tuple[int, os.stat_result, dict[str, dict[str, object]], int]:
    fd, metadata = open_regular(path)
    try:
        if metadata.st_size < 10:
            raise SystemExit(f"truncated safetensors file: {path}")
        header_length_raw = os.pread(fd, 8, 0)
        if len(header_length_raw) != 8:
            raise SystemExit(f"truncated safetensors header length: {path}")
        header_length = int.from_bytes(header_length_raw, "little")
        if header_length < 2 or header_length > metadata.st_size - 8:
            raise SystemExit(f"invalid safetensors header length: {path}")
        header_raw = os.pread(fd, header_length, 8)
        if len(header_raw) != header_length:
            raise SystemExit(f"truncated safetensors header: {path}")
        try:
            header = json.loads(header_raw)
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise SystemExit(f"invalid safetensors header: {path}") from error
        if not isinstance(header, dict):
            raise SystemExit(f"safetensors header must be an object: {path}")
        tensors: dict[str, dict[str, object]] = {}
        intervals: list[tuple[int, int, str]] = []
        payload_size = metadata.st_size - 8 - header_length
        for name, descriptor in header.items():
            if name == "__metadata__":
                if not isinstance(descriptor, dict):
                    raise SystemExit("invalid safetensors __metadata__")
                continue
            if not isinstance(name, str) or not name or not isinstance(descriptor, dict):
                raise SystemExit("invalid safetensors tensor descriptor")
            if set(descriptor) != {"data_offsets", "dtype", "shape"}:
                raise SystemExit(f"unexpected safetensors descriptor fields: {name}")
            offsets = descriptor["data_offsets"]
            dtype = descriptor["dtype"]
            if (
                not isinstance(offsets, list)
                or len(offsets) != 2
                or any(type(offset) is not int for offset in offsets)
                or not 0 <= offsets[0] <= offsets[1] <= payload_size
                or not isinstance(dtype, str)
                or dtype not in DTYPE_BYTES
            ):
                raise SystemExit(f"invalid safetensors tensor descriptor: {name}")
            expected_size = multiply(descriptor["shape"]) * DTYPE_BYTES[dtype]
            if offsets[1] - offsets[0] != expected_size:
                raise SystemExit(f"safetensors tensor byte size mismatch: {name}")
            intervals.append((offsets[0], offsets[1], name))
            tensors[name] = descriptor
        if not tensors:
            raise SystemExit(f"safetensors file has no tensors: {path}")
        cursor = 0
        for start, end, name in sorted(intervals):
            if start != cursor:
                raise SystemExit(f"safetensors payload has a gap or overlap before: {name}")
            cursor = end
        if cursor != payload_size:
            raise SystemExit(f"safetensors payload is not fully described: {path}")
        return fd, metadata, tensors, 8 + header_length
    except BaseException:
        os.close(fd)
        raise


def tensor_sha256(fd: int, payload_start: int, descriptor: dict[str, object]) -> str:
    offsets = descriptor["data_offsets"]
    assert isinstance(offsets, list)
    position = payload_start + int(offsets[0])
    remaining = int(offsets[1]) - int(offsets[0])
    digest = hashlib.sha256()
    while remaining:
        chunk = os.pread(fd, min(remaining, 1024 * 1024), position)
        if not chunk:
            raise SystemExit("safetensors tensor data changed or was truncated")
        digest.update(chunk)
        position += len(chunk)
        remaining -= len(chunk)
    return digest.hexdigest()


def inventory(
    root: Path, *, draft_only: bool, allow_index: bool
) -> tuple[dict[str, tuple[str, list[int], str]], list[int]]:
    index_path = root / "model.safetensors.index.json"
    if index_path.is_symlink():
        raise SystemExit("safetensors index must not be aliased")
    if allow_index and index_path.exists():
        index = exact_json(index_path)
        if not isinstance(index, dict) or not isinstance(index.get("weight_map"), dict):
            raise SystemExit("invalid parent safetensors index")
        weight_map = index["weight_map"]
        if any(not isinstance(name, str) or Path(name).name != name for name in weight_map.values()):
            raise SystemExit("invalid parent safetensors shard path")
        shard_names = sorted(set(weight_map.values()))
    else:
        if not allow_index and index_path.exists():
            raise SystemExit("export must use one exact model.safetensors file")
        weight_map = None
        shard_names = ["model.safetensors"]
    result: dict[str, tuple[str, list[int], str]] = {}
    descriptors_seen: set[str] = set()
    opened: list[int] = []
    try:
        for shard_name in shard_names:
            fd, metadata, tensors, payload_start = parse_safetensors(root / shard_name)
            opened.append(fd)
            for name, descriptor in tensors.items():
                if weight_map is not None and weight_map.get(name) != shard_name:
                    continue
                descriptors_seen.add(name)
                if draft_only and (
                    "dflash_module." not in name or "rotary_emb." in name
                ):
                    continue
                dtype = descriptor["dtype"]
                shape = descriptor["shape"]
                assert isinstance(dtype, str) and isinstance(shape, list)
                result[name] = (dtype, shape, tensor_sha256(fd, payload_start, descriptor))
            after = os.fstat(fd)
            if (metadata.st_dev, metadata.st_ino, metadata.st_size, metadata.st_mtime_ns) != (
                after.st_dev,
                after.st_ino,
                after.st_size,
                after.st_mtime_ns,
            ):
                raise SystemExit(f"safetensors input changed while validating: {shard_name}")
        if weight_map is not None and set(weight_map) != descriptors_seen:
            raise SystemExit("parent safetensors index does not exactly match its shards")
        return result, opened
    except BaseException:
        for fd in opened:
            os.close(fd)
        raise


parent_root, export_file, receipt_path, method = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3]), sys.argv[4]
if method not in {"DFlash", "DSpark"}:
    raise SystemExit("unsupported Q30 continuation method")
parent_all, parent_fds = inventory(parent_root, draft_only=True, allow_index=True)
try:
    parent = {
        name.split("dflash_module.", 1)[1]: value
        for name, value in parent_all.items()
        if "dflash_module." in name and "rotary_emb." not in name
    }
finally:
    for fd in parent_fds:
        os.close(fd)
if method == "DSpark":
    for source_name, export_name in {
        "markov_w1.weight": "markov_head.markov_w1.weight",
        "markov_w2.weight": "markov_head.markov_w2.weight",
        "confidence_proj.weight": "confidence_head.proj.weight",
        "confidence_proj.bias": "confidence_head.proj.bias",
    }.items():
        if source_name in parent:
            parent[export_name] = parent.pop(source_name)
export_all, export_fds = inventory(export_file.parent, draft_only=False, allow_index=False)
try:
    if export_file.name != "model.safetensors" or set(export_all) != set(parent):
        raise SystemExit("exported drafter tensor inventory does not exactly match authenticated parent")
    required = {"fc.weight", "hidden_norm.weight", "norm.weight"}
    required.update(f"layers.{layer}.self_attn.q_proj.weight" for layer in range(5))
    if method == "DSpark":
        required.update(
            {"markov_head.markov_w1.weight", "markov_head.markov_w2.weight"}
        )
    if not required.issubset(export_all):
        raise SystemExit("exported drafter tensor inventory is missing required architecture tensors")
    changed = 0
    canonical_inventory: list[list[object]] = []
    for name in sorted(parent):
        parent_dtype, parent_shape, parent_digest = parent[name]
        export_dtype, export_shape, export_digest = export_all[name]
        if parent_dtype != export_dtype or parent_shape != export_shape:
            raise SystemExit(f"exported drafter tensor metadata differs from parent: {name}")
        changed += export_digest != parent_digest
        canonical_inventory.append([name, export_dtype, export_shape, export_digest])
    if changed == 0:
        raise SystemExit("exported drafter state differs from authenticated parent by zero tensors")
finally:
    for fd in export_fds:
        os.close(fd)
body = {
    "changed_tensor_count": changed,
    "inventory_sha256": hashlib.sha256(
        json.dumps(canonical_inventory, separators=(",", ":")).encode()
    ).hexdigest(),
    "method": method,
    "schema_version": "q30t-export-validation-v1",
    "tensor_count": len(canonical_inventory),
}
payload = (json.dumps(body, sort_keys=True, separators=(",", ":")) + "\n").encode()
fd = os.open(receipt_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
try:
    os.write(fd, payload)
    os.fsync(fd)
finally:
    os.close(fd)
# Q30_EXPORT_SAFETENSORS_VALIDATOR_END
PY
    else
        for _ in $(seq 1 1800); do
            [[ -f "$CONTROL_ROOT/export-validation.json" ]] && break
            sleep 1
        done
        [[ -f "$CONTROL_ROOT/export-validation.json" ]] || {
            echo "rank-0 export validation receipt did not appear" >&2
            exit 2
        }
    fi
}
training_task_command="$(declare -f training_task)"$'\ntraining_task'

set -m
srun --overlap --nodes=16 --ntasks=16 --ntasks-per-node=1 --gpus-per-task=4 \
    --cpus-per-task=96 --kill-on-bad-exit=1 --no-container-mount-home \
    --container-name="$container_name" --container-mounts="$mounts" \
    /bin/bash -c "$training_task_command" &
training_step_pid=$!
set +m

while kill -0 "$keeper_step_pid" 2>/dev/null && kill -0 "$training_step_pid" 2>/dev/null; do
    sleep 1
done
if ! kill -0 "$keeper_step_pid" 2>/dev/null; then
    kill -TERM -- "-$training_step_pid" 2>/dev/null || true
    wait "$training_step_pid" 2>/dev/null || true
    wait "$keeper_step_pid" 2>/dev/null || true
    echo "node keeper exited before training completed; completion suppressed" >&2
    exit 2
fi
set +e
wait "$training_step_pid"
training_status=$?
set -e
if [[ "$training_status" -ne 0 ]]; then
    touch "$stop_file"
    wait "$keeper_step_pid" 2>/dev/null || true
    echo "training step failed; completion suppressed" >&2
    exit "$training_status"
fi
training_step_pid=""

# Q30_EXPOSURE_RECONCILIATION
set -m
srun --overlap --nodes=1 --ntasks=1 --ntasks-per-node=1 \
    --cpus-per-task=1 --kill-on-bad-exit=1 --no-container-mount-home \
    --container-name="$container_name" --container-mounts="$mounts" \
    "$training_scratch_root/run/runtime/bin/python" - \
    "$CONTROL_ROOT/rendezvous/exposure-evidence" "$CONTROL_ROOT/exposure-evidence.json" \
    "$training_scratch_root/run/DATA.jsonl" "$DATASET_SHA256" \
    "$ORDERED_PROMPT_UUIDS_SHA256" <<'PY' &
import hashlib
import json
import os
import stat
import sys
from importlib.metadata import PackageNotFoundError, version
from itertools import chain, islice
from pathlib import Path

from accelerate.data_loader import BatchSamplerShard, SeedableRandomSampler
from torch.utils.data import BatchSampler

root, receipt_path, dataset_path = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
expected_dataset_sha256, expected_ordered_uuids_sha256 = sys.argv[4:6]
flags = os.O_RDONLY | os.O_NOFOLLOW
try:
    descriptor = os.open(dataset_path, flags)
except OSError as error:
    raise SystemExit("authenticated DATA is unreadable") from error
try:
    before = os.fstat(descriptor)
    if not stat.S_ISREG(before.st_mode):
        raise SystemExit("authenticated DATA must remain a regular file")
    dataset_digest = hashlib.sha256()
    ordered_uuid_digest = hashlib.sha256(b"[")
    dataset_uuids: list[str] = []
    with os.fdopen(descriptor, "rb", closefd=False) as stream:
        for index, line in enumerate(stream):
            dataset_digest.update(line)
            try:
                row = json.loads(line)
            except (UnicodeDecodeError, json.JSONDecodeError) as error:
                raise SystemExit("authenticated DATA contains invalid JSON") from error
            prompt_uuid = row.get("prompt_uuid") if isinstance(row, dict) else None
            if (
                not isinstance(prompt_uuid, str)
                or len(prompt_uuid) != 64
                or any(character not in "0123456789abcdef" for character in prompt_uuid)
            ):
                raise SystemExit("authenticated DATA prompt UUID inventory is invalid")
            if index:
                ordered_uuid_digest.update(b",")
            ordered_uuid_digest.update(
                json.dumps(prompt_uuid, ensure_ascii=False, separators=(",", ":")).encode()
            )
            dataset_uuids.append(prompt_uuid)
    ordered_uuid_digest.update(b"]")
    after = os.fstat(descriptor)
finally:
    os.close(descriptor)
if (
    len(dataset_uuids) != 700_000
    or (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns, before.st_ctime_ns)
    != (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns, after.st_ctime_ns)
    or dataset_digest.hexdigest() != expected_dataset_sha256
    or ordered_uuid_digest.hexdigest() != expected_ordered_uuids_sha256
):
    raise SystemExit("authenticated DATA identity changed before exposure reconciliation")

try:
    accelerate_version = version("accelerate")
except PackageNotFoundError as error:
    raise SystemExit("pinned Accelerate runtime is unavailable") from error
if accelerate_version != "1.14.0":
    raise SystemExit(f"unapproved Accelerate sampler implementation: {accelerate_version}")
stage = os.environ.get("STAGE", "canary")
expected_exposure_count = int(os.environ.get("EXPECTED_EXPOSURE_COUNT", "10240"))
expected_steps = int(os.environ.get("EXPECTED_STEPS", "20"))
expected_final_global_batch_size = int(
    os.environ.get("EXPECTED_FINAL_GLOBAL_BATCH_SIZE", "512")
)
if (stage, expected_exposure_count, expected_steps, expected_final_global_batch_size) not in {
    ("canary", 10_240, 20, 512),
    ("full", 700_000, 1_368, 96),
}:
    raise SystemExit("strict exposure schedule environment is invalid")
local_batch_size = 4
expected_indices_by_rank: list[list[int]] = []
for rank in range(32):
    sampler = SeedableRandomSampler(range(700_000), data_seed=42)
    if stage == "full":
        source = iter(sampler)
        expected: list[int] = []
        for _ in range(5_468):
            global_microbatch = list(islice(source, 128))
            expected.extend(global_microbatch[rank * 4 : (rank + 1) * 4])
        final_global_microbatch = list(source)
        if len(final_global_microbatch) != 96:
            raise SystemExit("pinned full sampler did not produce exact final global batch 96")
        expected.extend(final_global_microbatch[rank * 3 : (rank + 1) * 3])
        expected_indices_by_rank.append(expected)
        continue
    else:
        batches = BatchSampler(sampler, batch_size=local_batch_size, drop_last=False)
    sharded_batches = BatchSamplerShard(
        batches,
        num_processes=32,
        process_index=rank,
        split_batches=False,
        even_batches=True,
    )
    expected_indices_by_rank.append(
        list(islice(chain.from_iterable(sharded_batches), expected_exposure_count // 32))
    )
paths = sorted(root.glob("rank-*.jsonl"))
expected_names = [f"rank-{rank:05d}.jsonl" for rank in range(32)]
if [path.name for path in paths] != expected_names:
    raise SystemExit("strict exposure evidence must contain exactly 32 trainer ranks")
ordered: list[list[object]] = []
seen_indices: set[int] = set()
seen_uuids: set[str] = set()
for rank, path in enumerate(paths):
    if path.is_symlink() or not path.is_file():
        raise SystemExit("strict exposure evidence file is aliased")
    raw = path.read_bytes()
    lines = raw.splitlines(keepends=True)
    if len(lines) != expected_exposure_count // 32:
        raise SystemExit(
            f"strict exposure rank {rank} must contain exactly "
            f"{expected_exposure_count // 32} occurrences"
        )
    for occurrence, line in enumerate(lines):
        record = json.loads(line)
        expected_index = expected_indices_by_rank[rank][occurrence]
        if (
            not isinstance(record, dict)
            or set(record) != {"dataset_index", "prompt_uuid"}
            or line != (json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n").encode()
            or type(record["dataset_index"]) is not int
            or not 0 <= record["dataset_index"] < 700_000
            or not isinstance(record["prompt_uuid"], str)
            or len(record["prompt_uuid"]) != 64
            or any(character not in "0123456789abcdef" for character in record["prompt_uuid"])
            or record["dataset_index"] != expected_index
            or record["prompt_uuid"] != dataset_uuids[expected_index]
            or record["dataset_index"] in seen_indices
            or record["prompt_uuid"] in seen_uuids
        ):
            raise SystemExit("strict exposure occurrence/UUID evidence is invalid or duplicated")
        seen_indices.add(record["dataset_index"])
        seen_uuids.add(record["prompt_uuid"])
        ordered.append([rank, record["dataset_index"], record["prompt_uuid"]])
if len(ordered) != expected_exposure_count:
    raise SystemExit("strict exposure evidence must reconcile the exact scheduled exposure")
body = {
    "accelerate_version": accelerate_version,
    "dataset_sha256": expected_dataset_sha256,
    "job_id": os.environ["SLURM_JOB_ID"],
    "occurrence_count": len(ordered),
    "ordered_prompt_uuids_sha256": expected_ordered_uuids_sha256,
    "ordered_rank_occurrence_uuid_sha256": hashlib.sha256(
        json.dumps(ordered, separators=(",", ":")).encode()
    ).hexdigest(),
    "rank_count": 32,
    "sampler_batch_size": local_batch_size,
    "sampler_epoch": 0,
    "sampler_num_replicas": 32,
    "sampler_seed": 42,
    "schema_version": (
        "q30t-strict-exposure-evidence-v4"
        if stage == "full"
        else "q30t-strict-exposure-evidence-v3"
    ),
}
if stage == "full":
    body.update(
        {
            "final_global_batch_size": expected_final_global_batch_size,
            "final_local_batch_size": expected_final_global_batch_size // 32,
            "optimizer_steps": expected_steps,
        }
    )
with receipt_path.open("x") as stream:
    json.dump(body, stream, sort_keys=True, separators=(",", ":"))
    stream.write("\n")
    stream.flush()
    os.fsync(stream.fileno())
PY
reconciliation_step_pid=$!
set +m

while kill -0 "$keeper_step_pid" 2>/dev/null \
    && kill -0 "$reconciliation_step_pid" 2>/dev/null; do
    sleep 1
done
if ! kill -0 "$keeper_step_pid" 2>/dev/null; then
    kill -TERM -- "-$reconciliation_step_pid" 2>/dev/null || true
    wait "$reconciliation_step_pid" 2>/dev/null || true
    wait "$keeper_step_pid" 2>/dev/null || true
    reconciliation_step_pid=""
    keeper_step_pid=""
    echo "keeper exited during exposure reconciliation; completion suppressed" >&2
    exit 2
fi
set +e
wait "$reconciliation_step_pid"
reconciliation_status=$?
set -e
reconciliation_step_pid=""
if [[ "$reconciliation_status" -ne 0 ]]; then
    touch "$stop_file"
    wait "$keeper_step_pid" 2>/dev/null || true
    keeper_step_pid=""
    echo "exposure reconciliation failed; completion suppressed" >&2
    exit "$reconciliation_status"
fi

# Keep anchor-backed mounts alive through reconciliation, then successfully reap
# their keepers before the final completion receipt can be minted.
touch "$stop_file"
set +e
wait "$keeper_step_pid"
keeper_status=$?
set -e
keeper_step_pid=""
if [[ "$keeper_status" -ne 0 ]]; then
    echo "node keeper failed after exposure reconciliation; completion suppressed" >&2
    exit "$keeper_status"
fi

/usr/bin/python3 -I -B - "$CANARY_RECEIPT" \
    "$bootstrap_publisher_path" "$bootstrap_publisher_oid" <<'PY'
import hashlib
import json
import os
import stat
import sys
import types
from pathlib import Path

publisher_path = Path(sys.argv[2])
publisher_oid = sys.argv[3]
descriptor = os.open(publisher_path, os.O_RDONLY | os.O_NOFOLLOW)
try:
    before = os.fstat(descriptor)
    if not stat.S_ISREG(before.st_mode):
        raise SystemExit("authenticated completion publisher is not a regular file")
    publisher_source = b""
    while chunk := os.read(descriptor, 1024 * 1024):
        publisher_source += chunk
    after = os.fstat(descriptor)
finally:
    os.close(descriptor)
publisher_identity = (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
if publisher_identity != (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns):
    raise SystemExit("authenticated completion publisher changed while reading")
observed_oid = hashlib.sha1(
    b"blob " + str(len(publisher_source)).encode("ascii") + b"\0" + publisher_source
).hexdigest()
if observed_oid != publisher_oid:
    raise SystemExit("authenticated completion publisher differs from its Git object ID")
publisher_module = types.ModuleType("_q30_commit_bound_atomic_publisher")
publisher_module.__file__ = str(publisher_path)
sys.modules[publisher_module.__name__] = publisher_module
exec(compile(publisher_source, str(publisher_path), "exec"), publisher_module.__dict__)
atomic_publish_bytes = publisher_module.atomic_publish_bytes


def identity(metadata: os.stat_result) -> tuple[int, int]:
    return metadata.st_dev, metadata.st_ino


def adopt_exact_receipt(output: Path, expected: bytes) -> None:
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0)
    directory_flags = flags | getattr(os, "O_DIRECTORY", 0)
    try:
        parent_fd = os.open(output.parent, directory_flags)
    except OSError as error:
        raise FileExistsError(f"Q30 completion receipt parent cannot be adopted: {output.parent}") from error
    try:
        parent_before = os.fstat(parent_fd)
        try:
            named = os.stat(output.name, dir_fd=parent_fd, follow_symlinks=False)
            descriptor = os.open(output.name, flags, dir_fd=parent_fd)
        except OSError as error:
            raise FileExistsError(f"Q30 completion receipt cannot be adopted: {output}") from error
        try:
            before = os.fstat(descriptor)
            if (
                identity(named) != identity(before)
                or not stat.S_ISREG(before.st_mode)
                or before.st_nlink != 1
            ):
                raise FileExistsError(f"Q30 completion receipt is not immutable: {output}")
            retained = bytearray()
            while block := os.read(descriptor, 1024 * 1024):
                retained.extend(block)
                if len(retained) > len(expected):
                    break
            after = os.fstat(descriptor)
            rebound = os.stat(output.name, dir_fd=parent_fd, follow_symlinks=False)
            if identity(before) != identity(after) or identity(after) != identity(rebound):
                raise FileExistsError(f"Q30 completion receipt changed while adopting: {output}")
            if bytes(retained) != expected:
                raise FileExistsError(f"Q30 completion receipt differs: {output}")
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        if identity(os.fstat(parent_fd)) != identity(parent_before):
            raise FileExistsError(f"Q30 completion receipt parent changed: {output.parent}")
        os.fsync(parent_fd)
        final_parent = os.fstat(parent_fd)
        try:
            final_named = os.stat(output.name, dir_fd=parent_fd, follow_symlinks=False)
            final_descriptor = os.open(output.name, flags, dir_fd=parent_fd)
        except OSError as error:
            raise FileExistsError(
                f"Q30 completion receipt cannot be rebound after durability: {output}"
            ) from error
        try:
            final_opened = os.fstat(final_descriptor)
            absolute = os.stat(output, follow_symlinks=False)
            if (
                identity(final_parent) != identity(parent_before)
                or identity(final_named) != identity(after)
                or identity(final_opened) != identity(after)
                or identity(absolute) != identity(after)
                or final_opened.st_nlink != 1
            ):
                raise FileExistsError(
                    f"Q30 completion receipt changed or rebound after durability: {output}"
                )
        finally:
            os.close(final_descriptor)
    finally:
        os.close(parent_fd)


def publish_or_adopt_receipt(output: Path, payload: bytes) -> None:
    if os.path.lexists(output):
        adopt_exact_receipt(output, payload)
        return
    try:
        atomic_publish_bytes(
            output,
            payload,
            job_id=f"{os.environ['SLURM_JOB_ID']}-{os.getpid()}",
        )
    except Exception:
        if not os.path.lexists(output):
            raise
        adopt_exact_receipt(output, payload)
        return
    adopt_exact_receipt(output, payload)


def stable_file_sha256(path: Path) -> str:
    flags = os.O_RDONLY | os.O_NOFOLLOW
    fd = os.open(path, flags)
    try:
        before = os.fstat(fd)
        if not stat.S_ISREG(before.st_mode):
            raise SystemExit(f"completion evidence is not a regular file: {path}")
        digest = hashlib.sha256()
        while chunk := os.read(fd, 1024 * 1024):
            digest.update(chunk)
        after = os.fstat(fd)
        if (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns) != (
            after.st_dev,
            after.st_ino,
            after.st_size,
            after.st_mtime_ns,
        ):
            raise SystemExit(f"completion evidence changed while hashing: {path}")
        return digest.hexdigest()
    finally:
        os.close(fd)


def stable_tree_sha256(root: Path) -> str:
    if not root.is_dir() or root.is_symlink():
        raise SystemExit(f"completion evidence directory is absent or aliased: {root}")
    entries: list[dict[str, str | int]] = []
    for path in sorted(root.rglob("*"), key=lambda item: item.relative_to(root).as_posix()):
        relative = path.relative_to(root).as_posix()
        metadata = path.lstat()
        if stat.S_ISDIR(metadata.st_mode):
            entries.append({"path": relative, "type": "directory"})
        elif stat.S_ISREG(metadata.st_mode):
            entries.append(
                {
                    "path": relative,
                    "sha256": stable_file_sha256(path),
                    "size": metadata.st_size,
                    "type": "file",
                }
            )
        else:
            raise SystemExit(f"unsupported completion evidence entry: {path}")
    if not any(entry["type"] == "file" for entry in entries):
        raise SystemExit(f"completion evidence directory has no files: {root}")
    canonical = json.dumps(entries, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(canonical).hexdigest()


output_root = Path(os.environ["OUTPUT_ROOT"])
expected_steps = int(os.environ.get("EXPECTED_STEPS", "20"))
expected_exposure_count = int(os.environ.get("EXPECTED_EXPOSURE_COUNT", "10240"))
expected_final_global_batch_size = int(
    os.environ.get("EXPECTED_FINAL_GLOBAL_BATCH_SIZE", "512")
)
stage = os.environ.get("STAGE", "canary")
if (stage, expected_steps, expected_exposure_count, expected_final_global_batch_size) not in {
    ("canary", 20, 10_240, 512),
    ("full", 1_368, 700_000, 96),
}:
    raise SystemExit("completion evidence schedule is invalid")
checkpoint = output_root / f"checkpoint-{expected_steps}"
trainer_state = checkpoint / "trainer_state.json"
if trainer_state.is_symlink():
    raise SystemExit("trainer_state.json must not be a symlink")
try:
    trainer_state_body = json.loads(trainer_state.read_bytes())
except (OSError, json.JSONDecodeError) as error:
    raise SystemExit("fresh final trainer_state.json is unavailable") from error
if (
    type(trainer_state_body.get("global_step")) is not int
    or trainer_state_body.get("global_step") != expected_steps
):
    raise SystemExit("fresh trainer_state global_step does not equal the exact scheduled step")
export_path = output_root / f"exported-checkpoint-{expected_steps}"
try:
    export_model_sha256 = stable_file_sha256(export_path / "model.safetensors")
except OSError as error:
    raise SystemExit(f"export completion evidence has no model weight file: {export_path}") from error
exposure_receipt = Path(os.environ["CONTROL_ROOT"]) / "exposure-evidence.json"
try:
    exposure_body = json.loads(exposure_receipt.read_bytes())
except (OSError, json.JSONDecodeError) as error:
    raise SystemExit("strict exposure evidence receipt is unavailable") from error
if (
    not isinstance(exposure_body, dict)
    or set(exposure_body)
    != ({
        "accelerate_version",
        "dataset_sha256",
        "job_id",
        "occurrence_count",
        "ordered_prompt_uuids_sha256",
        "ordered_rank_occurrence_uuid_sha256",
        "rank_count",
        "sampler_batch_size",
        "sampler_epoch",
        "sampler_num_replicas",
        "sampler_seed",
        "schema_version",
    } | ({
        "final_global_batch_size",
        "final_local_batch_size",
        "optimizer_steps",
    } if stage == "full" else set()))
    or exposure_body.get("schema_version")
    != (
        "q30t-strict-exposure-evidence-v4"
        if stage == "full"
        else "q30t-strict-exposure-evidence-v3"
    )
    or exposure_body.get("accelerate_version") != "1.14.0"
    or exposure_body.get("rank_count") != 32
    or exposure_body.get("occurrence_count") != expected_exposure_count
    or exposure_body.get("sampler_seed") != 42
    or exposure_body.get("sampler_batch_size") != 4
    or exposure_body.get("sampler_epoch") != 0
    or exposure_body.get("sampler_num_replicas") != 32
    or exposure_body.get("dataset_sha256") != os.environ["DATASET_SHA256"]
    or exposure_body.get("ordered_prompt_uuids_sha256")
    != os.environ["ORDERED_PROMPT_UUIDS_SHA256"]
    or (
        stage == "full"
        and (
            exposure_body.get("optimizer_steps") != expected_steps
            or exposure_body.get("final_global_batch_size")
            != expected_final_global_batch_size
            or exposure_body.get("final_local_batch_size")
            != expected_final_global_batch_size // 32
        )
    )
):
    raise SystemExit("strict exposure evidence receipt does not reconcile")
export_validation_receipt = Path(os.environ["CONTROL_ROOT"]) / "export-validation.json"
try:
    export_validation_raw = export_validation_receipt.read_bytes()
    export_validation_body = json.loads(export_validation_raw)
except (OSError, json.JSONDecodeError) as error:
    raise SystemExit("export validation receipt is unavailable") from error
if (
    not isinstance(export_validation_body, dict)
    or set(export_validation_body)
    != {
        "changed_tensor_count",
        "inventory_sha256",
        "method",
        "schema_version",
        "tensor_count",
    }
    or export_validation_raw
    != (
        json.dumps(export_validation_body, sort_keys=True, separators=(",", ":")) + "\n"
    ).encode()
    or export_validation_body.get("schema_version") != "q30t-export-validation-v1"
    or export_validation_body.get("method") != os.environ["METHOD"]
    or type(export_validation_body.get("tensor_count")) is not int
    or export_validation_body["tensor_count"] < (10 if os.environ["METHOD"] == "DSpark" else 8)
    or type(export_validation_body.get("changed_tensor_count")) is not int
    or not 1 <= export_validation_body["changed_tensor_count"] <= export_validation_body["tensor_count"]
    or not isinstance(export_validation_body.get("inventory_sha256"), str)
    or len(export_validation_body["inventory_sha256"]) != 64
    or any(character not in "0123456789abcdef" for character in export_validation_body["inventory_sha256"])
):
    raise SystemExit("export validation receipt does not reconcile")
container_evidence_receipt = Path(os.environ["CONTROL_ROOT"]) / "container-evidence.json"
try:
    container_evidence_raw = container_evidence_receipt.read_bytes()
    container_evidence_body = json.loads(container_evidence_raw)
except (OSError, json.JSONDecodeError) as error:
    raise SystemExit("named-container reconciliation receipt is unavailable") from error
if (
    set(container_evidence_body)
    != {
        "job_id",
        "node_count",
        "ordered_node_sentinel_sha256",
        "schema_version",
    }
    or container_evidence_raw
    != (json.dumps(container_evidence_body, sort_keys=True, separators=(",", ":")) + "\n").encode()
    or container_evidence_body.get("schema_version")
    != "q30t-named-container-reconciliation-v1"
    or container_evidence_body.get("job_id") != os.environ["SLURM_JOB_ID"]
    or container_evidence_body.get("node_count") != 16
    or not isinstance(container_evidence_body.get("ordered_node_sentinel_sha256"), str)
    or len(container_evidence_body["ordered_node_sentinel_sha256"]) != 64
    or any(
        character not in "0123456789abcdef"
        for character in container_evidence_body["ordered_node_sentinel_sha256"]
    )
):
    raise SystemExit("named-container reconciliation receipt does not reconcile")
body = {
    "checkpoint_path": str(checkpoint),
    "checkpoint_tree_sha256": stable_tree_sha256(checkpoint),
    "container_evidence_sha256": stable_file_sha256(container_evidence_receipt),
    "dataset_sha256": os.environ["DATASET_SHA256"],
    "export_path": str(export_path),
    "export_model_sha256": export_model_sha256,
    "export_tree_sha256": stable_tree_sha256(export_path),
    "export_validation_inventory_sha256": export_validation_body["inventory_sha256"],
    "export_validation_sha256": stable_file_sha256(export_validation_receipt),
    "exposure_evidence_sha256": stable_file_sha256(exposure_receipt),
    "exposure_ordered_rank_occurrence_uuid_sha256": exposure_body[
        "ordered_rank_occurrence_uuid_sha256"
    ],
    "job_id": os.environ["SLURM_JOB_ID"],
    "method": os.environ["METHOD"],
    "method_config_relative": os.environ["METHOD_CONFIG_RELATIVE"],
    "ordered_prompt_uuids_sha256": os.environ["ORDERED_PROMPT_UUIDS_SHA256"],
    "output_root": os.environ["OUTPUT_ROOT"],
    "parent_archive_sha256": os.environ["PARENT_ARCHIVE_SHA256"],
    "parent_tree_sha256": os.environ["PARENT_TREE_SHA256"],
    "run_name": os.environ["RUN_NAME"],
    "runtime_archive_sha256": os.environ["RUNTIME_ARCHIVE_SHA256"],
    "runtime_image_sha256": os.environ["RUNTIME_IMAGE_SHA256"],
    "runtime_tree_sha256": os.environ["RUNTIME_TREE_SHA256"],
    "schema_version": f"q30t-ptv23-{stage}-completion-v1",
    "source_archive_sha256": os.environ["SOURCE_ARCHIVE_SHA256"],
    "source_commit": os.environ["SOURCE_COMMIT"],
    "source_tree_oid": os.environ["SOURCE_TREE_OID"],
    "source_tree_sha256": os.environ["SOURCE_TREE_SHA256"],
    "stage": os.environ["STAGE"],
    "steps": expected_steps,
    "target_archive_sha256": os.environ["TARGET_ARCHIVE_SHA256"],
    "target_model": os.environ["TARGET_MODEL"],
    "target_tree_sha256": os.environ["TARGET_TREE_SHA256"],
    "trainer_state_sha256": stable_file_sha256(trainer_state),
    "training_entrypoint_relative": os.environ["TRAINING_ENTRYPOINT_RELATIVE"],
}
payload = (json.dumps(body, sort_keys=True, separators=(",", ":")) + "\n").encode()
publish_or_adopt_receipt(Path(sys.argv[1]), payload)
PY
/bin/rm -rf -- "${replay_dir:?}"
trap - EXIT
