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

"""Behavioral contracts for reusable SpecDec cluster entrypoints."""

from __future__ import annotations

import hashlib
import json
import os
import shlex
import shutil
import subprocess
from pathlib import Path

import pytest

_LAUNCHER_DIR = Path(__file__).resolve().parents[1]
_CLUSTER_EXAMPLES = _LAUNCHER_DIR / "examples/clusters"
_TRANSFERS = _LAUNCHER_DIR / "common/specdec/transfers"
_BASH: str = shutil.which("bash") or "/bin/bash"
_HEAD = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=_LAUNCHER_DIR, text=True).strip()
_ARTIFACT_SOURCE_COMMIT = "9" * 40


@pytest.mark.parametrize("cluster", ["oci-hsg", "lyris", "ptyche"])
def test_cluster_wrapper_selects_profile_and_forwards_artifact_paths(cluster: str) -> None:
    """A server wrapper preserves user paths while selecting only its own profile."""
    wrapper = _CLUSTER_EXAMPLES / cluster / "specdec.sh"
    readiness = "/lustre/project/readiness/cluster.json"
    result = subprocess.run(
        [
            _BASH,
            str(wrapper),
            "--print-command",
            "full-chain",
            "--manifest",
            "/home/user/manifests/qwen.json",
            "--receipt",
            "/lustre/project/receipts/full.jsonl",
            "--readiness-receipt",
            readiness,
            "--experiment-id",
            "drafter-a4e90af908c82df7",
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    command = shlex.split(result.stdout)
    assert command[0] == str(_LAUNCHER_DIR / "common/specdec/submit_drafter_full_chain.sh")
    assert command[1:3] == [
        "--cluster-profile",
        str(_LAUNCHER_DIR / f"common/specdec/profiles/{cluster}.yaml"),
    ]
    assert command[command.index("--readiness-receipt") + 1] == readiness
    assert command[-2:] == ["--experiment-id", "drafter-a4e90af908c82df7"]


def test_training_entrypoint_fails_closed_without_readiness_receipt() -> None:
    """A training submission cannot bypass cluster readiness qualification."""
    result = subprocess.run(
        [
            _BASH,
            str(_CLUSTER_EXAMPLES / "oci-hsg/specdec.sh"),
            "--print-command",
            "training-wave",
            "--manifest",
            "/home/user/manifest.json",
            "--receipt",
            "/lustre/project/receipt.jsonl",
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 2
    assert "--readiness-receipt is required" in result.stderr


def test_evaluator_pair_is_not_claimed_portable_by_cluster_wrapper() -> None:
    """Profiles cannot make OCI-specific static evaluator directives portable to Lyris."""
    result = subprocess.run(
        [
            _BASH,
            str(_CLUSTER_EXAMPLES / "lyris/specdec.sh"),
            "--print-command",
            "evaluator-pair",
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 2
    assert "unsupported action" in result.stderr


def test_cluster_upload_action_delegates_to_profile_aware_submitter() -> None:
    """The public entrypoint schedules transfers instead of running rclone on the login node."""
    result = subprocess.run(
        [
            _BASH,
            str(_CLUSTER_EXAMPLES / "oci-hsg/specdec.sh"),
            "--print-command",
            "upload",
            "--artifact-id",
            "checkpoint-a",
            "--artifact-source-commit",
            _ARTIFACT_SOURCE_COMMIT,
            "--source",
            "/lustre/project/checkpoint-a",
            "--remote-root",
            "remote:project/specdec",
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    command = shlex.split(result.stdout)
    assert command[0] == str(_TRANSFERS / "submit_transfer.sh")
    assert command[1:4] == [
        "--cluster-profile",
        str(_LAUNCHER_DIR / "common/specdec/profiles/oci-hsg.yaml"),
        "upload",
    ]


def test_transfer_requires_slurm_allocation_before_rclone(tmp_path: Path) -> None:
    """Bulk transfer cannot run inline from a login shell."""
    context = _transfer_context(tmp_path)
    result = _run_transfer(context, "upload", slurm=False)

    assert result.returncode == 2
    assert "inside an approved one-node Slurm allocation" in result.stderr
    assert not context.calls.exists()


def test_transfer_rejects_launcher_commit_mismatch(tmp_path: Path) -> None:
    """The executing transfer tooling must match its separately pinned launcher revision."""
    context = _transfer_context(tmp_path)
    result = _run_transfer(context, "upload", launcher_commit="f" * 40)

    assert result.returncode == 2
    assert "launcher checkout does not match" in result.stderr
    assert not context.calls.exists()


@pytest.mark.parametrize(
    ("allocation_env", "message"),
    [
        ({"SLURM_JOB_NUM_NODES": "2"}, "one-node Slurm allocation"),
        ({"SLURM_JOB_PARTITION": "batch"}, "partition is not approved"),
        ({"SLURM_CPUS_PER_TASK": "2"}, "CPU count does not match"),
        ({"SLURM_NTASKS": "2"}, "exactly one task"),
        ({"SLURM_JOB_ACCOUNT": "wrong-account"}, "account is not approved"),
        ({"SLURM_JOB_ACCOUNT": ""}, "account is not approved"),
        ({"SLURM_JOB_ACCOUNT": None}, "account is not approved"),
    ],
)
def test_transfer_rejects_wrong_allocation_shape(
    tmp_path: Path, allocation_env: dict[str, str | None], message: str
) -> None:
    """The low-level worker accepts only its one-node approved transfer allocation."""
    context = _transfer_context(tmp_path)
    result = _run_transfer(context, "upload", allocation_env=allocation_env)

    assert result.returncode == 2
    assert message in result.stderr
    assert not context.calls.exists()


def test_upload_writes_content_manifest_and_reuses_exact_completed_bundle(
    tmp_path: Path,
) -> None:
    """A stable file manifest binds completion and makes identical retries idempotent."""
    context = _transfer_context(tmp_path)
    (context.source / "nested").mkdir()
    (context.source / "nested/config.json").write_text('{"a": 1}\n')
    first = _run_transfer(context, "upload")

    assert first.returncode == 0, first.stderr
    digest, remote_bundle = _only_remote_bundle(context)
    manifest_bytes = (remote_bundle / "manifest.json").read_bytes()
    manifest = json.loads(manifest_bytes)
    completion = json.loads((remote_bundle / "completion.json").read_text())
    assert hashlib.sha256(manifest_bytes).hexdigest() == digest
    assert completion == {
        "artifact_id": context.artifact_id,
        "artifact_source_commit": _ARTIFACT_SOURCE_COMMIT,
        "complete": True,
        "manifest_sha256": digest,
        "schema": "modelopt-specdec-completion-v1",
    }
    assert manifest["file_count"] == 2
    assert manifest["artifact_source_commit"] == _ARTIFACT_SOURCE_COMMIT
    assert manifest["total_bytes"] == sum(record["size"] for record in manifest["files"])
    assert [record["path"] for record in manifest["files"]] == [
        "nested/config.json",
        "weights.bin",
    ]
    initial_payload_copies = _payload_copy_count(context.calls)

    second = _run_transfer(context, "upload", job_id="1002")

    assert second.returncode == 0, second.stderr
    assert "reused completed bundle" in second.stdout
    assert _payload_copy_count(context.calls) == initial_payload_copies
    local_completion = (
        context.durable / "transfers" / context.artifact_id / digest / "completion.json"
    )
    assert local_completion.read_bytes() == (remote_bundle / "completion.json").read_bytes()


def test_same_artifact_id_with_concurrent_different_content_uses_distinct_digest_paths(
    tmp_path: Path,
) -> None:
    """Concurrent publications with one semantic ID cannot overwrite different contents."""
    first = _transfer_context(tmp_path / "first", artifact_id="shared-id")
    second = _transfer_context(tmp_path / "second", artifact_id="shared-id")
    second.remote = first.remote
    second.calls = first.calls
    (first.source / "weights.bin").write_bytes(b"first")
    (second.source / "weights.bin").write_bytes(b"second")

    processes = [
        _start_transfer(first, "upload", job_id="2001"),
        _start_transfer(second, "upload", job_id="2002"),
    ]
    results = [(*process.communicate(timeout=20), process.returncode) for process in processes]

    assert [result[2] for result in results] == [0, 0], results
    artifact_root = first.remote / "project/specdec/shared-id"
    digest_roots = sorted(path for path in artifact_root.iterdir() if path.is_dir())
    assert len(digest_roots) == 2
    for digest_root in digest_roots:
        completion = json.loads((digest_root / "completion.json").read_text())
        assert completion["manifest_sha256"] == digest_root.name


def test_upload_distinguishes_missing_marker_from_authentication_failure(tmp_path: Path) -> None:
    """A remote transport error cannot be mistaken for an unpublished artifact."""
    context = _transfer_context(tmp_path)
    result = _run_transfer(context, "upload", extra_env={"FAKE_RCLONE_AUTH_FAIL": "1"})

    assert result.returncode == 17
    assert "could not verify remote artifact state" in result.stderr
    assert _operations(context.calls) == ["copyto"]
    assert not list((context.durable / "transfers").glob("**/completion.json"))


def test_upload_rejects_symlink_that_escapes_durable_tree(tmp_path: Path) -> None:
    """Materializing a symlink cannot read bytes outside the profile's durable tree."""
    context = _transfer_context(tmp_path)
    outside = tmp_path / "secret.bin"
    outside.write_bytes(b"secret")
    (context.source / "escape.bin").symlink_to(outside)

    result = _run_transfer(context, "upload")

    assert result.returncode == 2
    assert "symlink target escapes durable root" in result.stderr
    assert not context.calls.exists()


def test_upload_rejects_top_level_source_symlink(tmp_path: Path) -> None:
    """The source itself cannot bypass symlink rejection by resolving before traversal."""
    context = _transfer_context(tmp_path)
    source_link = context.source.parent / "source-link"
    source_link.symlink_to(context.source, target_is_directory=True)
    context.source = source_link

    result = _run_transfer(context, "upload")

    assert result.returncode == 2
    assert "local bundle path cannot be a symlink" in result.stderr
    assert not context.calls.exists()


def test_launcher_verifier_rejects_dirty_checkout(tmp_path: Path) -> None:
    """Transfer provenance cannot be emitted from a checkout with local changes."""
    checkout = tmp_path / "checkout"
    checkout.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=checkout, check=True)
    subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=checkout, check=True)
    subprocess.run(["git", "config", "user.name", "Test"], cwd=checkout, check=True)
    tracked = checkout / "tracked.txt"
    tracked.write_text("clean\n")
    subprocess.run(["git", "add", "tracked.txt"], cwd=checkout, check=True)
    subprocess.run(["git", "commit", "-qm", "initial"], cwd=checkout, check=True)
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=checkout, text=True).strip()
    tracked.write_text("dirty\n")

    result = subprocess.run(
        [
            "python3",
            str(_TRANSFERS / "bundle_manifest.py"),
            "verify-launcher",
            "--root",
            str(checkout),
            "--expected-commit",
            commit,
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 2
    assert "launcher checkout is not clean" in result.stderr


def test_oci_transfer_submitter_uses_cpu_partition_and_test_only(
    tmp_path: Path,
) -> None:
    """OCI transfers are scheduled on one CPU datamover node after scheduler validation."""
    context = _transfer_context(tmp_path)
    command_dir = tmp_path / "bin"
    command_dir.mkdir()
    calls = tmp_path / "sbatch.calls"
    sbatch = command_dir / "sbatch"
    sbatch.write_text(
        "#!/usr/bin/env bash\n"
        'printf "%s\\n" "$*" >> "$SBATCH_CALLS"\n'
        '[[ "$1" == "--test-only" ]] && exit 0\n'
        'printf "8123\\n"\n'
    )
    sbatch.chmod(0o755)
    environment = {
        **os.environ,
        "PATH": f"{command_dir}{os.pathsep}{os.environ['PATH']}",
        "SBATCH_CALLS": str(calls),
    }

    result = subprocess.run(
        [
            _BASH,
            str(_TRANSFERS / "submit_transfer.sh"),
            "--cluster-profile",
            str(context.profile),
            "upload",
            "--artifact-id",
            context.artifact_id,
            "--artifact-source-commit",
            _ARTIFACT_SOURCE_COMMIT,
            "--source",
            str(context.source),
            "--remote-root",
            "fake:project/specdec",
        ],
        check=False,
        capture_output=True,
        text=True,
        env=environment,
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "8123"
    submitted = calls.read_text().splitlines()
    assert len(submitted) == 2
    assert submitted[0].startswith("--test-only ")
    assert submitted[1].startswith("--parsable ")
    for command in submitted:
        assert "--account=account" in command
        assert "--partition=cpu_datamover" in command
        assert "--nodes=1" in command
        assert "--cpus-per-task=4" in command
        assert "--gpus" not in command
        assert f"--launcher-commit={_HEAD}" in command
        assert str(context.durable / "transfers/logs") in command


def test_lyris_transfer_submitter_uses_one_partition_managed_node(
    tmp_path: Path,
) -> None:
    """Lyris transfers use one GB200 node without requesting explicit GPUs."""
    context = _transfer_context(tmp_path)
    _write_profile(context, _HEAD, name="lyris")
    command_dir = tmp_path / "bin"
    command_dir.mkdir()
    calls = tmp_path / "sbatch.calls"
    sbatch = command_dir / "sbatch"
    sbatch.write_text(
        "#!/usr/bin/env bash\n"
        'printf "%s\\n" "$*" >> "$SBATCH_CALLS"\n'
        '[[ "$1" == "--test-only" ]] && exit 0\n'
        'printf "8124\\n"\n'
    )
    sbatch.chmod(0o755)
    environment = {
        **os.environ,
        "PATH": f"{command_dir}{os.pathsep}{os.environ['PATH']}",
        "SBATCH_CALLS": str(calls),
    }

    result = subprocess.run(
        [
            _BASH,
            str(_TRANSFERS / "submit_transfer.sh"),
            "--cluster-profile",
            str(context.profile),
            "upload",
            "--artifact-id",
            context.artifact_id,
            "--artifact-source-commit",
            _ARTIFACT_SOURCE_COMMIT,
            "--source",
            str(context.source),
            "--remote-root",
            "fake:project/specdec",
        ],
        check=False,
        capture_output=True,
        text=True,
        env=environment,
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "8124"
    submitted = calls.read_text().splitlines()
    assert len(submitted) == 2
    for command in submitted:
        assert "--account=account" in command
        assert "--partition=gb200" in command
        assert "--nodes=1" in command
        assert "--cpus-per-task=16" in command
        assert "--gpus" not in command


@pytest.mark.parametrize("cluster", ["ptyche"])
def test_transfer_submitter_fails_closed_without_approved_partition(
    tmp_path: Path, cluster: str
) -> None:
    """Clusters without a reviewed transfer partition never fall back to GPU batch."""
    context = _transfer_context(tmp_path)
    _write_profile(context, _HEAD, name=cluster)

    result = subprocess.run(
        [
            _BASH,
            str(_TRANSFERS / "submit_transfer.sh"),
            "--cluster-profile",
            str(context.profile),
            "upload",
            "--artifact-id",
            context.artifact_id,
            "--artifact-source-commit",
            _ARTIFACT_SOURCE_COMMIT,
            "--source",
            str(context.source),
            "--remote-root",
            "fake:project/specdec",
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 2
    assert "no approved transfer partition" in result.stderr


def test_download_verifies_manifest_bytes_then_atomically_installs(tmp_path: Path) -> None:
    """A valid download is hashed locally and atomically installed from a sibling partial."""
    context = _transfer_context(tmp_path)
    uploaded = _run_transfer(context, "upload")
    assert uploaded.returncode == 0, uploaded.stderr
    digest, _ = _only_remote_bundle(context)

    downloaded = _run_transfer(context, "download", manifest_sha256=digest, job_id="3001")

    assert downloaded.returncode == 0, downloaded.stderr
    assert (context.destination / "weights.bin").read_bytes() == b"weights"
    assert not list(context.destination.parent.glob(f"{context.destination.name}.partial-*"))
    local_marker = context.durable / "transfers" / context.artifact_id / digest / "completion.json"
    assert local_marker.is_file()


def test_download_requires_artifact_source_commit_before_remote_access(tmp_path: Path) -> None:
    """Corpus completion and manifest verification require the producing source commit."""
    context = _transfer_context(tmp_path)
    result = _run_transfer(
        context,
        "download",
        manifest_sha256="a" * 64,
        artifact_source_commit=None,
    )

    assert result.returncode == 2
    assert not context.calls.exists()


def test_download_corruption_never_reaches_final_destination(tmp_path: Path) -> None:
    """A payload changed after transfer fails local manifest verification and is discarded."""
    context = _transfer_context(tmp_path)
    uploaded = _run_transfer(context, "upload")
    assert uploaded.returncode == 0, uploaded.stderr
    digest, _ = _only_remote_bundle(context)

    downloaded = _run_transfer(
        context,
        "download",
        manifest_sha256=digest,
        job_id="3002",
        extra_env={"FAKE_RCLONE_CORRUPT_DOWNLOAD": "1"},
    )

    assert downloaded.returncode != 0
    assert "mismatch for weights.bin" in downloaded.stderr
    assert not context.destination.exists()
    assert not list(context.destination.parent.glob(f"{context.destination.name}.partial-*"))


def test_download_rejects_manifest_digest_corruption_before_payload_copy(
    tmp_path: Path,
) -> None:
    """The marker-bound manifest is verified before any bulk payload download."""
    context = _transfer_context(tmp_path)
    uploaded = _run_transfer(context, "upload")
    assert uploaded.returncode == 0, uploaded.stderr
    digest, remote_bundle = _only_remote_bundle(context)
    manifest = json.loads((remote_bundle / "manifest.json").read_text())
    manifest["total_bytes"] += 1
    (remote_bundle / "manifest.json").write_text(json.dumps(manifest))
    payload_downloads = _payload_download_count(context.calls)

    downloaded = _run_transfer(context, "download", manifest_sha256=digest, job_id="3007")

    assert downloaded.returncode == 2
    assert "manifest SHA-256 mismatch" in downloaded.stderr
    assert _payload_download_count(context.calls) == payload_downloads
    assert not context.destination.exists()


def test_download_preserves_remote_authentication_error_status(tmp_path: Path) -> None:
    """A download reports remote authentication failure instead of claiming incompleteness."""
    context = _transfer_context(tmp_path)
    uploaded = _run_transfer(context, "upload")
    assert uploaded.returncode == 0, uploaded.stderr
    digest, _ = _only_remote_bundle(context)

    downloaded = _run_transfer(
        context,
        "download",
        manifest_sha256=digest,
        job_id="3008",
        extra_env={"FAKE_RCLONE_AUTH_FAIL": "1"},
    )

    assert downloaded.returncode == 17
    assert "could not fetch remote completion marker" in downloaded.stderr
    assert not context.destination.exists()


def test_download_reuses_exact_final_but_rejects_conflicting_final(tmp_path: Path) -> None:
    """Completed local content is reusable, while conflicting content is never overwritten."""
    context = _transfer_context(tmp_path)
    uploaded = _run_transfer(context, "upload")
    assert uploaded.returncode == 0, uploaded.stderr
    digest, _ = _only_remote_bundle(context)
    first = _run_transfer(context, "download", manifest_sha256=digest, job_id="3003")
    assert first.returncode == 0, first.stderr
    payload_downloads = _payload_download_count(context.calls)

    reused = _run_transfer(context, "download", manifest_sha256=digest, job_id="3004")
    assert reused.returncode == 0, reused.stderr
    assert "reused final destination" in reused.stdout
    assert _payload_download_count(context.calls) == payload_downloads

    (context.destination / "weights.bin").write_bytes(b"conflict")
    conflict = _run_transfer(context, "download", manifest_sha256=digest, job_id="3005")
    assert conflict.returncode == 2
    assert "conflicting final destination" in conflict.stderr
    assert (context.destination / "weights.bin").read_bytes() == b"conflict"


def test_atomic_install_does_not_clobber_destination_created_during_download(
    tmp_path: Path,
) -> None:
    """A destination appearing after transfer wins without being replaced by the partial tree."""
    context = _transfer_context(tmp_path)
    uploaded = _run_transfer(context, "upload")
    assert uploaded.returncode == 0, uploaded.stderr
    digest, _ = _only_remote_bundle(context)

    result = _run_transfer(
        context,
        "download",
        manifest_sha256=digest,
        job_id="3006",
        extra_env={"FAKE_RCLONE_CREATE_CONFLICT": str(context.destination)},
    )

    assert result.returncode == 2
    assert "conflicting final destination" in result.stderr
    assert (context.destination / "owner.txt").read_text() == "other process\n"
    assert not list(context.destination.parent.glob(f"{context.destination.name}.partial-*"))


@pytest.mark.parametrize("direction", ["upload", "download"])
def test_transfer_rejects_local_path_equal_to_durable_root(tmp_path: Path, direction: str) -> None:
    """A transfer can never treat the profile's entire durable tree as one artifact."""
    context = _transfer_context(tmp_path)
    context.source = context.durable
    context.destination = context.durable
    result = _run_transfer(
        context,
        direction,
        manifest_sha256="0" * 64 if direction == "download" else None,
    )

    assert result.returncode == 2
    assert "cannot equal the profile durable root" in result.stderr
    assert not context.calls.exists()


class _TransferContext:
    def __init__(self, root: Path, artifact_id: str) -> None:
        self.root = root
        self.durable = root / "durable"
        self.scratch = root / "scratch"
        self.source = self.durable / "artifacts/source"
        self.destination = self.durable / "artifacts/downloaded"
        self.remote = root / "remote"
        self.calls = root / "rclone.calls"
        self.artifact_id = artifact_id
        self.profile = root / "profile.yaml"
        self.rclone = root / "rclone"


def _transfer_context(
    root: Path,
    *,
    artifact_id: str = "q30-opb-dflash-b8",
    commit: str = _HEAD,
) -> _TransferContext:
    context = _TransferContext(root, artifact_id)
    for directory in (context.source, context.scratch, context.remote):
        directory.mkdir(parents=True, exist_ok=True)
    (context.source / "weights.bin").write_bytes(b"weights")
    _write_profile(context, commit)
    _write_fake_rclone(context)
    return context


def _write_profile(
    context: _TransferContext,
    commit: str,
    *,
    name: str = "oci-hsg",
) -> None:
    context.profile.write_text(
        "\n".join(
            (
                f"name: {name}",
                f"modelopt_commit: {commit}",
                "ssh_host: login-oci",
                "account: account",
                "partition: batch",
                "fallback_partition: null",
                f"durable_root: {context.durable}",
                "scratch_candidates:",
                f"  - {context.scratch}",
                "training_nodes: 4",
                "training_segment: 4",
                "evaluation_nodes: 1",
                "evaluation_segment: 1",
                "gpus_per_node: 4",
                "explicit_gpu_flag: true",
                'walltime: "03:55:00"',
            )
        )
        + "\n"
    )


def _write_fake_rclone(context: _TransferContext) -> None:
    context.rclone.write_text(
        """#!/usr/bin/env bash
set -euo pipefail
printf '%q ' "$@" >> "$RCLONE_CALLS"
printf '\n' >> "$RCLONE_CALLS"

remote_path() {
    printf '%s/%s' "$FAKE_RCLONE_REMOTE" "${1#*:}"
}

operation="$1"
source_path="$2"
destination_path="$3"
source_remote=0
destination_remote=0
[[ "$source_path" == *:* ]] && source_remote=1
[[ "$destination_path" == *:* ]] && destination_remote=1
[[ "$source_remote" -eq 0 ]] || source_path="$(remote_path "$source_path")"
[[ "$destination_remote" -eq 0 ]] || destination_path="$(remote_path "$destination_path")"

if [[ "$operation" == copyto && "$source_remote" -eq 1 && "$2" == *completion.json \
    && "${FAKE_RCLONE_AUTH_FAIL:-0}" == 1 ]]; then
    exit 17
fi

case "$operation" in
    copyto)
        [[ -f "$source_path" ]] || exit 3
        mkdir -p "$(dirname "$destination_path")"
        cp "$source_path" "$destination_path"
        ;;
    copy)
        [[ -d "$source_path" ]] || exit 3
        mkdir -p "$destination_path"
        cp -RL "$source_path"/. "$destination_path"/
        if [[ "$source_remote" -eq 1 && "${FAKE_RCLONE_CORRUPT_DOWNLOAD:-0}" == 1 ]]; then
            first_file="$(find "$destination_path" -type f | sort | head -n 1)"
            printf corrupt >> "$first_file"
        fi
        if [[ "$source_remote" -eq 1 && -n "${FAKE_RCLONE_CREATE_CONFLICT:-}" ]]; then
            mkdir -p "$FAKE_RCLONE_CREATE_CONFLICT"
            printf 'other process\n' > "$FAKE_RCLONE_CREATE_CONFLICT/owner.txt"
        fi
        ;;
    check)
        diff -qr "$source_path" "$destination_path" >/dev/null || exit 9
        ;;
    *) exit 64 ;;
esac
"""
    )
    context.rclone.chmod(0o755)


def _run_transfer(
    context: _TransferContext,
    direction: str,
    *,
    manifest_sha256: str | None = None,
    job_id: str = "1001",
    slurm: bool = True,
    extra_env: dict[str, str] | None = None,
    launcher_commit: str = _HEAD,
    allocation_env: dict[str, str | None] | None = None,
    artifact_source_commit: str | None = _ARTIFACT_SOURCE_COMMIT,
) -> subprocess.CompletedProcess[str]:
    process = _start_transfer(
        context,
        direction,
        manifest_sha256=manifest_sha256,
        job_id=job_id,
        slurm=slurm,
        extra_env=extra_env,
        launcher_commit=launcher_commit,
        allocation_env=allocation_env,
        artifact_source_commit=artifact_source_commit,
    )
    stdout, stderr = process.communicate(timeout=20)
    return subprocess.CompletedProcess(process.args, process.returncode, stdout, stderr)


def _start_transfer(
    context: _TransferContext,
    direction: str,
    *,
    manifest_sha256: str | None = None,
    job_id: str = "1001",
    slurm: bool = True,
    extra_env: dict[str, str] | None = None,
    launcher_commit: str = _HEAD,
    allocation_env: dict[str, str | None] | None = None,
    artifact_source_commit: str | None = _ARTIFACT_SOURCE_COMMIT,
) -> subprocess.Popen[str]:
    local_flag = "--source" if direction == "upload" else "--destination"
    local_path = context.source if direction == "upload" else context.destination
    arguments = [
        _BASH,
        str(_TRANSFERS / f"{direction}_bundle.sh"),
        "--cluster-profile",
        str(context.profile),
        "--artifact-id",
        context.artifact_id,
        local_flag,
        str(local_path),
        "--remote-root",
        "fake:project/specdec",
        "--rclone-bin",
        str(context.rclone),
        "--launcher-commit",
        launcher_commit,
    ]
    if artifact_source_commit is not None:
        arguments.extend(("--artifact-source-commit", artifact_source_commit))
    if manifest_sha256 is not None:
        arguments.extend(("--manifest-sha256", manifest_sha256))
    environment = {
        **os.environ,
        "FAKE_RCLONE_REMOTE": str(context.remote),
        "RCLONE_CALLS": str(context.calls),
        **(extra_env or {}),
    }
    if slurm:
        environment.update(
            {
                "SLURM_JOB_ID": job_id,
                "SLURM_JOB_NODELIST": "node001",
                "SLURM_JOB_NUM_NODES": "1",
                "SLURM_JOB_PARTITION": "cpu_datamover",
                "SLURM_CPUS_PER_TASK": "4",
                "SLURM_NTASKS": "1",
                "SLURM_JOB_ACCOUNT": "account",
            }
        )
        for key, value in (allocation_env or {}).items():
            if value is None:
                environment.pop(key, None)
            else:
                environment[key] = value
    else:
        environment.pop("SLURM_JOB_ID", None)
        environment.pop("SLURM_JOB_NODELIST", None)
    return subprocess.Popen(
        arguments,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=environment,
    )


def _only_remote_bundle(context: _TransferContext) -> tuple[str, Path]:
    artifact_root = context.remote / "project/specdec" / context.artifact_id
    digest_roots = [path for path in artifact_root.iterdir() if path.is_dir()]
    assert len(digest_roots) == 1
    return digest_roots[0].name, digest_roots[0]


def _operations(calls: Path) -> list[str]:
    return [shlex.split(line)[0] for line in calls.read_text().splitlines()]


def _payload_copy_count(calls: Path) -> int:
    return sum(
        operation[0] == "copy" and operation[2].endswith("/payload")
        for operation in map(shlex.split, calls.read_text().splitlines())
    )


def _payload_download_count(calls: Path) -> int:
    return sum(
        operation[0] == "copy" and operation[1].endswith("/payload")
        for operation in map(shlex.split, calls.read_text().splitlines())
    )
