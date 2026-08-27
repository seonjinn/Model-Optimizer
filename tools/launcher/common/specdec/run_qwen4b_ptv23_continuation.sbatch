#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#SBATCH --job-name=q4-ptv23-continuation
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --time=02:00:00
set -euo pipefail
IMMUTABLE_CONTRACT_TOOL_SHA256="6751546191dedda988ba04b00db8fef8c30b9f6f431cd1c93e9e530e1dbd94de"
RUNTIME_ATTESTATION_IMPLEMENTED=0

: "${STAGE:?} ${CONTRACT:?} ${CONTRACT_SHA256:?} ${CONTRACT_TOOL:?}"
: "${CONTRACT_TOOL_SHA256:?}"
[[ "$CONTRACT_TOOL_SHA256" == "$IMMUTABLE_CONTRACT_TOOL_SHA256" ]]
((RUNTIME_ATTESTATION_IMPLEMENTED == 1)) || {
    echo "runtime image attestation is not implemented; runner is fail-closed" >&2
    exit 1
}
: "${TRAINING_ENTRYPOINT:?} ${TRAINING_ENTRYPOINT_SHA256:?}"
: "${IMAGE:?} ${IMAGE_SHA256:?}"
CANARY_STEPS=20
RESUME_MODE="weights-only"
OPTIMIZER_STATE="fresh"
SCHEDULER_STATE="fresh"
MAX_STEPS="${FULL_STEPS:-0}"
[[ "$STAGE" == "canary" ]] && MAX_STEPS="$CANARY_STEPS"
if [[ "$STAGE" == "canary" ]]; then
    : "${CANARY_RECEIPT:?} ${CANARY_EVIDENCE:?}"
fi
if [[ "$STAGE" == "full" ]]; then
    : "${CANARY_RECEIPT:?} ${CANARY_RECEIPT_SHA256:?}"
    : "${FULL_COMPLETION_EVIDENCE:?} ${FULL_COMPLETION_RECEIPT:?}"
    ((MAX_STEPS > CANARY_STEPS))
fi
RUNNER_STAGE_ROOT="$(mktemp -d "${SLURM_TMPDIR:-/tmp}/q4-continuation.XXXXXX")"
trap 'rm -rf -- "$RUNNER_STAGE_ROOT"' EXIT
export RUNNER_STAGE_ROOT MAX_STEPS RESUME_MODE OPTIMIZER_STATE SCHEDULER_STATE
python3 -I - <<'PY'
import hashlib
import json
import os
import stat
import subprocess
import sys

nofollow = getattr(os, "O_NOFOLLOW", 0)
directory_flags = os.O_RDONLY | os.O_DIRECTORY | nofollow
stage_root = os.environ["RUNNER_STAGE_ROOT"]
descriptor_prefix = f"/proc/{os.getpid()}/fd"


def _write_all(descriptor: int, block: bytes) -> None:
    remaining = memoryview(block)
    while remaining:
        written = os.write(descriptor, remaining)
        if written <= 0:
            raise SystemExit("anonymous staged source copy made no progress")
        remaining = remaining[written:]


def copy_to_anonymous_file(source: str, expected: str) -> int:
    source_fd = os.open(source, os.O_RDONLY | nofollow)
    tmpfile_flag = getattr(os, "O_TMPFILE", None)
    if tmpfile_flag is None:
        os.close(source_fd)
        raise SystemExit("node-local staging requires Python O_TMPFILE support")
    try:
        output_fd = os.open(stage_root, os.O_RDWR | tmpfile_flag, 0o600)
    except OSError as error:
        os.close(source_fd)
        raise SystemExit(
            f"node-local staging root does not support O_TMPFILE: {stage_root}: {error}"
        ) from error
    digest = hashlib.sha256()
    try:
        before = os.fstat(source_fd)
        if not stat.S_ISREG(before.st_mode):
            raise SystemExit("staged source is not a regular file")
        while block := os.read(source_fd, 8 * 1024 * 1024):
            digest.update(block)
            _write_all(output_fd, block)
        after = os.fstat(source_fd)
        if (
            (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
            != (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
            or digest.hexdigest() != expected
        ):
            raise SystemExit("staged source identity mismatch")
        os.fsync(output_fd)
        staged_stat = os.fstat(output_fd)
        os.lseek(output_fd, 0, os.SEEK_SET)
        staged_digest = hashlib.sha256()
        while block := os.read(output_fd, 8 * 1024 * 1024):
            staged_digest.update(block)
        if staged_stat.st_size != before.st_size or staged_digest.hexdigest() != expected:
            raise SystemExit("staged source copy mismatch")
        os.lseek(output_fd, 0, os.SEEK_SET)
        os.set_inheritable(output_fd, True)
        return output_fd
    except BaseException:
        os.close(output_fd)
        raise
    finally:
        os.close(source_fd)


def verify_pyxis_visible_descriptors(descriptors: tuple[int, ...]) -> None:
    if os.environ.get("SLURM_NNODES", "1") != "1":
        raise SystemExit(
            "multi-node Pyxis descriptor staging is not implemented; refusing launch"
        )
    if not os.path.isdir(descriptor_prefix):
        raise SystemExit("procfs keeper descriptor directory is unavailable")
    probe = (
        "import os,sys; "
        "descriptors=[os.open(path,os.O_RDONLY) for path in sys.argv[1:]]; "
        "[os.fstat(descriptor) for descriptor in descriptors]; "
        "[os.close(descriptor) for descriptor in descriptors]"
    )
    paths = [f"{descriptor_prefix}/{descriptor}" for descriptor in descriptors]
    try:
        subprocess.run(
            [sys.executable, "-I", "-c", probe, *paths],
            check=True,
            capture_output=True,
        )
    except (OSError, subprocess.CalledProcessError) as error:
        raise SystemExit(
            "procfs keeper descriptors are not visible across the process boundary"
        ) from error


def validate_mount_target(target: str) -> None:
    if (
        not os.path.isabs(target)
        or "," in target
        or ":" in target
        or any(ord(character) < 32 or ord(character) == 127 for character in target)
    ):
        raise SystemExit("contract mount target is unsafe for Pyxis grammar")


def open_directory(path: str, *, create: bool) -> int:
    if not os.path.isabs(path):
        raise SystemExit("mount root must be absolute")
    descriptor = os.open("/", directory_flags)
    try:
        for component in (part for part in path.split("/") if part):
            try:
                child = os.open(component, directory_flags, dir_fd=descriptor)
            except FileNotFoundError:
                if not create:
                    raise
                os.mkdir(component, mode=0o700, dir_fd=descriptor)
                child = os.open(component, directory_flags, dir_fd=descriptor)
            os.close(descriptor)
            descriptor = child
        os.set_inheritable(descriptor, True)
        return descriptor
    except BaseException:
        os.close(descriptor)
        raise


contract_fd = copy_to_anonymous_file(os.environ["CONTRACT"], os.environ["CONTRACT_SHA256"])
tool_fd = copy_to_anonymous_file(
    os.environ["CONTRACT_TOOL"], os.environ["CONTRACT_TOOL_SHA256"]
)
contract_source = f"{descriptor_prefix}/{contract_fd}"
tool_source = f"{descriptor_prefix}/{tool_fd}"
preflight = subprocess.run(
    [
        sys.executable,
        "-I",
        tool_source,
        "preflight-contract",
        "--contract",
        contract_source,
        "--contract-sha256",
        os.environ["CONTRACT_SHA256"],
        "--stage",
        os.environ["STAGE"],
    ],
    check=True,
    capture_output=True,
    text=True,
    pass_fds=(contract_fd, tool_fd),
)
plan = json.loads(preflight.stdout)
if plan.get("schema_version") != "qwen4b-ptv23-mount-plan-v1":
    raise SystemExit("trusted preflight returned an invalid mount plan")

stage = os.environ["STAGE"]
expected_evidence = (
    os.environ["CANARY_EVIDENCE"] if stage == "canary" else os.environ["FULL_COMPLETION_EVIDENCE"]
)
expected_receipt = (
    os.environ["CANARY_RECEIPT"] if stage == "canary" else os.environ["FULL_COMPLETION_RECEIPT"]
)
if (
    plan["training_entrypoint_path"] != os.environ["TRAINING_ENTRYPOINT"]
    or plan["runtime_image_path"] != os.environ["IMAGE"]
    or plan["checkpoint_output_root"] != os.environ["CHECKPOINT_OUTPUT_ROOT"]
    or plan["evidence_path"] != expected_evidence
    or plan["receipt_path"] != expected_receipt
):
    raise SystemExit("contract mount plan disagrees with submission arguments")

for target in (*plan["readonly_roots"], *plan["writable_roots"]):
    validate_mount_target(target)
readonly = [(open_directory(path, create=False), path) for path in plan["readonly_roots"]]
writable = [(open_directory(path, create=True), path) for path in plan["writable_roots"]]
image_fd = copy_to_anonymous_file(os.environ["IMAGE"], os.environ["IMAGE_SHA256"])
verify_pyxis_visible_descriptors(
    (contract_fd, tool_fd, image_fd, *(descriptor for descriptor, _ in readonly), *(descriptor for descriptor, _ in writable))
)

contract_target = "/tmp/q4-auth-contract.json"
tool_target = "/tmp/q4-auth-contract-tool.py"
mounts = [
    *(f"{descriptor_prefix}/{descriptor}:{target}:ro" for descriptor, target in readonly),
    *(f"{descriptor_prefix}/{descriptor}:{target}:rw" for descriptor, target in writable),
    f"{descriptor_prefix}/{contract_fd}:{contract_target}:ro",
    f"{descriptor_prefix}/{tool_fd}:{tool_target}:ro",
]
launch = [
    "launch",
    "--contract",
    contract_target,
    "--contract-sha256",
    os.environ["CONTRACT_SHA256"],
    "--training-entrypoint",
    os.environ["TRAINING_ENTRYPOINT"],
    "--training-entrypoint-sha256",
    os.environ["TRAINING_ENTRYPOINT_SHA256"],
    "--runtime-image",
    os.environ["IMAGE"],
    "--runtime-image-sha256",
    os.environ["IMAGE_SHA256"],
    "--checkpoint-output-root",
    plan["checkpoint_output_root"],
    "--stage",
    stage,
    "--max-steps",
    os.environ["MAX_STEPS"],
    "--resume-mode",
    os.environ["RESUME_MODE"],
    "--optimizer-state",
    os.environ["OPTIMIZER_STATE"],
    "--scheduler-state",
    os.environ["SCHEDULER_STATE"],
]
if os.environ.get("CANARY_RECEIPT"):
    launch += [
        "--canary-receipt",
        os.environ["CANARY_RECEIPT"],
        "--canary-receipt-sha256",
        os.environ.get("CANARY_RECEIPT_SHA256", ""),
    ]
if os.environ.get("CANARY_EVIDENCE"):
    launch += ["--canary-evidence", os.environ["CANARY_EVIDENCE"]]
if stage == "full":
    launch += [
        "--full-completion-evidence",
        os.environ["FULL_COMPLETION_EVIDENCE"],
        "--full-completion-receipt",
        os.environ["FULL_COMPLETION_RECEIPT"],
    ]

command = [
    "srun",
    "--export=NONE",
    "--no-container-mount-home",
    f"--container-image={descriptor_prefix}/{image_fd}",
    f"--container-mounts={','.join(mounts)}",
    "python",
    "-I",
    tool_target,
    *launch,
]
subprocess.run(command, check=True, env=os.environ)
PY
