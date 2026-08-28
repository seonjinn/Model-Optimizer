# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Behavioral review gates for the 16-node Q30 continuation controller."""

from __future__ import annotations

import hashlib
import json
import os
import resource
import shutil
import signal
import struct
import subprocess
import sys
from pathlib import Path
from typing import cast

import pytest

_REPO_ROOT = Path(__file__).parents[3]
_RUNNER = _REPO_ROOT / "tools/launcher/common/specdec/run_q30t_ptv23_continuation.sbatch"
_SUBMIT = _REPO_ROOT / "tools/launcher/common/specdec/submit_q30t_ptv23_continuation.sh"


def _copy_commit_bound_atomic_publisher(specdec_root: Path) -> None:
    shutil.copyfile(
        _REPO_ROOT / "tools/launcher/common/specdec/qwen4b_b_atomic.py",
        specdec_root / "qwen4b_b_atomic.py",
    )


def test_spooled_runner_uses_fixed_trusted_bootstrap_binaries(tmp_path: Path) -> None:
    """Untrusted PATH entries cannot run before descriptor authentication."""
    checkout = tmp_path / "checkout"
    module = checkout / "tools/launcher/common/specdec/q30t_ptv23_continuation.py"
    module.parent.mkdir(parents=True)
    module.write_text("raise SystemExit(72)\n")
    _copy_commit_bound_atomic_publisher(module.parent)
    subprocess.run(("git", "init", "-q", str(checkout)), check=True)
    subprocess.run(("git", "-C", str(checkout), "add", "."), check=True)
    subprocess.run(
        (
            "git",
            "-C",
            str(checkout),
            "-c",
            "user.name=Q30 Test",
            "-c",
            "user.email=q30@example.invalid",
            "commit",
            "-qm",
            "trusted authenticator",
        ),
        check=True,
    )
    commit = subprocess.run(
        ("git", "-C", str(checkout), "rev-parse", "HEAD"),
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    descriptor = tmp_path / "launch.json"
    descriptor_payload = {
        "contract": {"source_checkout_path": str(checkout), "source_commit": commit},
        "schema_version": "q30t-ptv23-launch-descriptor-v1",
        "stage": "canary",
    }
    descriptor.write_text(
        json.dumps(descriptor_payload, sort_keys=True, separators=(",", ":")) + "\n"
    )
    descriptor_sha256 = hashlib.sha256(descriptor.read_bytes()).hexdigest()
    fake_bin = tmp_path / "untrusted-bin"
    fake_bin.mkdir()
    marker_root = tmp_path / "path-substitution"
    marker_root.mkdir()
    for command in ("bash", "python3", "git", "tar"):
        executable = fake_bin / command
        executable.write_text(f"#!/bin/sh\nprintf substituted > {marker_root / command}\nexit 91\n")
        executable.chmod(0o755)

    result = subprocess.run(
        (
            str(_RUNNER),
            str(descriptor),
            descriptor_sha256,
            str(tmp_path / "receipt.json"),
            str(checkout),
        ),
        env=os.environ | {"PATH": f"{fake_bin}:{os.environ['PATH']}"},
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 72, result.stderr
    assert not list(marker_root.iterdir())


def test_spooled_runner_bootstraps_replay_from_descriptor_commit_not_mutable_checkout(
    tmp_path: Path,
) -> None:
    """A queued worktree edit cannot replace the descriptor authenticator."""
    checkout = tmp_path / "checkout"
    module = checkout / "tools/launcher/common/specdec/q30t_ptv23_continuation.py"
    module.parent.mkdir(parents=True)
    module.write_text("raise SystemExit(72)\n")
    _copy_commit_bound_atomic_publisher(module.parent)
    subprocess.run(("git", "init", "-q", str(checkout)), check=True)
    subprocess.run(("git", "-C", str(checkout), "add", "."), check=True)
    subprocess.run(
        (
            "git",
            "-C",
            str(checkout),
            "-c",
            "user.name=Q30 Test",
            "-c",
            "user.email=q30@example.invalid",
            "commit",
            "-qm",
            "trusted authenticator",
        ),
        check=True,
    )
    commit = subprocess.run(
        ("git", "-C", str(checkout), "rev-parse", "HEAD"),
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    marker = tmp_path / "mutable-authenticator-ran"
    module.write_text(
        "from pathlib import Path\n"
        f"Path({str(marker)!r}).write_text('bypass')\n"
        "raise SystemExit(73)\n"
    )
    descriptor = tmp_path / "launch.json"
    descriptor_payload = {
        "contract": {"source_checkout_path": str(checkout), "source_commit": commit},
        "schema_version": "q30t-ptv23-launch-descriptor-v1",
        "stage": "canary",
    }
    descriptor.write_text(
        json.dumps(descriptor_payload, sort_keys=True, separators=(",", ":")) + "\n"
    )
    descriptor_sha256 = hashlib.sha256(descriptor.read_bytes()).hexdigest()

    result = subprocess.run(
        (
            "bash",
            str(_RUNNER),
            str(descriptor),
            descriptor_sha256,
            str(tmp_path / "receipt.json"),
            str(checkout),
        ),
        env=os.environ | {"SLURM_GPUS_ON_NODE": "4", "SLURM_JOB_ID": "123", "SLURM_NNODES": "16"},
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 72, result.stderr
    assert not marker.exists()


def test_spooled_runner_materializes_exact_blobs_despite_mutable_archive_attributes(
    tmp_path: Path,
) -> None:
    """Git info attributes cannot omit the authenticator and expose a user-site substitute."""
    checkout = tmp_path / "checkout"
    module = checkout / "tools/launcher/common/specdec/q30t_ptv23_continuation.py"
    module.parent.mkdir(parents=True)
    module.write_text("raise SystemExit(72)\n")
    (module.parent / "retained.py").write_text("RETAINED = True\n")
    _copy_commit_bound_atomic_publisher(module.parent)
    subprocess.run(("git", "init", "-q", str(checkout)), check=True)
    subprocess.run(("git", "-C", str(checkout), "add", "."), check=True)
    subprocess.run(
        (
            "git",
            "-C",
            str(checkout),
            "-c",
            "user.name=Q30 Test",
            "-c",
            "user.email=q30@example.invalid",
            "commit",
            "-qm",
            "trusted authenticator",
        ),
        check=True,
    )
    commit = subprocess.run(
        ("git", "-C", str(checkout), "rev-parse", "HEAD"),
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    info_attributes = checkout / ".git/info/attributes"
    info_attributes.write_text(
        "tools/launcher/common/specdec/q30t_ptv23_continuation.py export-ignore\n"
    )
    marker = tmp_path / "user-site-substitute-ran"
    system_version = subprocess.run(
        (
            "/usr/bin/python3",
            "-c",
            "import sys;print(f'{sys.version_info[0]}.{sys.version_info[1]}')",
        ),
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    substitute = (
        tmp_path
        / "userbase/lib"
        / f"python{system_version}/site-packages/common/specdec/q30t_ptv23_continuation.py"
    )
    substitute.parent.mkdir(parents=True)
    substitute.write_text(
        "from pathlib import Path\n"
        f"Path({str(marker)!r}).write_text('substituted')\n"
        "raise SystemExit(73)\n"
    )
    descriptor = tmp_path / "launch.json"
    descriptor_payload = {
        "contract": {"source_checkout_path": str(checkout), "source_commit": commit},
        "schema_version": "q30t-ptv23-launch-descriptor-v1",
        "stage": "canary",
    }
    descriptor.write_text(
        json.dumps(descriptor_payload, sort_keys=True, separators=(",", ":")) + "\n"
    )
    descriptor_sha256 = hashlib.sha256(descriptor.read_bytes()).hexdigest()

    result = subprocess.run(
        (
            str(_RUNNER),
            str(descriptor),
            descriptor_sha256,
            str(tmp_path / "receipt.json"),
            str(checkout),
        ),
        env=os.environ | {"PYTHONUSERBASE": str(tmp_path / "userbase")},
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 72, result.stderr
    assert not marker.exists()


def test_q30_controller_is_a_real_16_node_keeper_then_pyxis_consumer() -> None:
    """The reviewed controller has both keeper and authenticated Pyxis training phases."""
    assert _RUNNER.is_file(), "Q30 controller is missing"
    source = _RUNNER.read_text()

    assert "#SBATCH --nodes=16" in source
    assert "#SBATCH --gpus-per-node=4" in source
    assert source.count("srun") >= 2
    assert "ptv23_node_keeper.py" in source
    assert "--overlap" in source
    assert "--no-container-mount-home" in source
    assert "--container-image=" in source
    assert "validate_keeper_receipt" in source
    assert "modelopt.__file__" in source
    assert "node-evidence" in source
    assert "TRAINING_ENTRYPOINT" in source
    assert "SOURCE_ARCHIVE" in source
    assert "TARGET_ARCHIVE" in source
    assert "PARENT_ARCHIVE" in source
    assert "RUNTIME_ARCHIVE" in source
    assert "export -f" not in source
    assert "declare -f keeper_task" in source
    assert "declare -f training_task" in source
    assert "rendezvous:/scratchspace" in source
    assert "trainer_state.json" in source
    assert 'trainer_state_body.get("global_step") != expected_steps' in source
    assert "checkpoint_tree_sha256" in source
    assert "O_EXCL" in source
    assert "--container-name=" in source
    assert "q30-container-sentinel" in source
    assert "node-evidence.container" in source
    assert 'kill -TERM -- "-$training_step_pid"' in source
    assert "Q30_STRICT_EXPOSURE" in source
    assert "exposure-evidence" in source


@pytest.mark.parametrize(
    ("submit_mode", "run_name", "receipt_flag"),
    [
        ("--submit-canary", "q30t-dflash-b8-ptv23-swe-heavy-700k-canary", "--canary-receipt"),
        ("--submit-full", "q30t-dflash-b8-ptv23-swe-heavy-700k-full", "--full-receipt"),
    ],
)
def test_q30_submitter_checks_clean_pushed_commit_test_only_and_fresh_receipt(
    tmp_path: Path, submit_mode: str, run_name: str, receipt_flag: str
) -> None:
    """Submission is gated by Git remote identity, Slurm test-only, and fresh evidence."""
    assert _SUBMIT.is_file(), "Q30 submitter is missing"
    source = _SUBMIT.read_text()
    for required in (
        "status --porcelain",
        "rev-parse HEAD",
        "@{upstream}",
        "--test-only",
        "--parsable",
        "--nodes=16",
        "--gpus-per-node=4",
        "--canary-receipt",
        'git -C "$repo_root"',
        "--launch-descriptor",
        "--launch-descriptor-sha256",
        "--export=NONE",
    ):
        assert required in source
    assert "--export=ALL" not in source

    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    calls = tmp_path / "sbatch.calls"
    fake_sbatch = fake_bin / "sbatch"
    fake_sbatch.write_text(
        "#!/bin/bash\n"
        "set -euo pipefail\n"
        f"printf '%s\\n' \"$*\" >> {calls}\n"
        '[[ "$*" == *--test-only* ]] && exit 0\n'
        "echo 12345\n"
    )
    fake_sbatch.chmod(0o755)
    fake_git = fake_bin / "git"
    fake_git.write_text(
        "#!/bin/bash\n"
        "set -euo pipefail\n"
        'case "$*" in\n'
        f"  *'rev-parse --show-toplevel'*) echo {_REPO_ROOT} ;;\n"
        "  *'status --porcelain'*) exit 0 ;;\n"
        "  *'rev-parse HEAD'*) echo aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa ;;\n"
        "  *'rev-parse @{upstream}'*) echo aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa ;;\n"
        "  *) exit 2 ;;\n"
        "esac\n"
    )
    fake_git.chmod(0o755)
    fake_python = fake_bin / "python3"
    fake_python.write_text(
        "#!/bin/bash\n"
        "set -euo pipefail\n"
        "while (($#)); do\n"
        '  if [[ "$1" == --environment-output ]]; then\n'
        f"    printf 'RUN_NAME\\0{run_name}\\0METHOD\\0DFlash\\0' > \"$2\"\n"
        "    exit 0\n"
        "  fi\n"
        "  shift\n"
        "done\n"
        "exit 2\n"
    )
    fake_python.chmod(0o755)
    receipt = tmp_path / "completion.json"
    log = tmp_path / "training.log"
    descriptor = tmp_path / "launch.json"
    descriptor.write_text("{}\n")
    environment = os.environ | {
        "PATH": f"{fake_bin}:{os.environ['PATH']}",
        "Q30T_SKIP_LOCAL_CONTRACT_AUTH": "0",
    }

    result = subprocess.run(
        (
            str(_SUBMIT),
            submit_mode,
            "--account",
            "test-account",
            "--output",
            str(log),
            receipt_flag,
            str(receipt),
            "--launch-descriptor",
            str(descriptor),
            "--launch-descriptor-sha256",
            "0" * 64,
        ),
        cwd=tmp_path,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    call_lines = calls.read_text().splitlines()
    assert call_lines[0].find("--test-only") >= 0
    assert "--test-only" not in call_lines[1]
    expected_job_name = f"--job-name={run_name}"
    assert expected_job_name in call_lines[0]
    assert expected_job_name in call_lines[1]
    assert all(line.endswith(f" {receipt} {_REPO_ROOT}") for line in call_lines)
    assert not receipt.exists()


def test_q30_spooled_runner_uses_authenticated_positional_repository_root(tmp_path: Path) -> None:
    """Slurm's copied script location cannot replace the descriptor-bound checkout."""
    spooled_runner = tmp_path / "slurm_script"
    spooled_runner.write_bytes(_RUNNER.read_bytes())
    spooled_runner.chmod(0o755)
    receipt = tmp_path / "receipt.json"
    observed_source_checkout = tmp_path / "observed-source-checkout"
    checkout = tmp_path / "checkout"
    module = checkout / "tools/launcher/common/specdec/q30t_ptv23_continuation.py"
    module.parent.mkdir(parents=True)
    module.write_text(
        "import os\n"
        "import sys\n"
        "from pathlib import Path\n"
        "index = sys.argv.index('--source-checkout')\n"
        "Path(os.environ['Q30_OBSERVED_CHECKOUT']).write_text(sys.argv[index + 1])\n"
        "raise SystemExit(73)\n"
    )
    _copy_commit_bound_atomic_publisher(module.parent)
    subprocess.run(("git", "init", "-q", str(checkout)), check=True)
    subprocess.run(("git", "-C", str(checkout), "add", "."), check=True)
    subprocess.run(
        (
            "git",
            "-C",
            str(checkout),
            "-c",
            "user.name=Q30 Test",
            "-c",
            "user.email=q30@example.invalid",
            "commit",
            "-qm",
            "trusted positional replay",
        ),
        check=True,
    )
    commit = subprocess.run(
        ("git", "-C", str(checkout), "rev-parse", "HEAD"),
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    descriptor = tmp_path / "launch.json"
    descriptor.write_text(
        json.dumps(
            {
                "contract": {
                    "source_checkout_path": str(checkout),
                    "source_commit": commit,
                },
                "schema_version": "q30t-ptv23-launch-descriptor-v1",
                "stage": "canary",
            },
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    )
    descriptor_sha256 = hashlib.sha256(descriptor.read_bytes()).hexdigest()

    result = subprocess.run(
        (
            "bash",
            str(spooled_runner),
            str(descriptor),
            descriptor_sha256,
            str(receipt),
            str(checkout),
        ),
        env=os.environ | {"Q30_OBSERVED_CHECKOUT": str(observed_source_checkout)},
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 73, result.stderr
    assert observed_source_checkout.read_text() == str(checkout)


def _completion_evidence_program() -> str:
    source = _RUNNER.read_text()
    marker = '    "$bootstrap_publisher_path" "$bootstrap_publisher_oid" <<\'PY\'\n'
    return source.rsplit(marker, 1)[1].split("\nPY\n", 1)[0]


def _completion_evidence_command(receipt: Path) -> tuple[str, str, str, str, str]:
    publisher = _REPO_ROOT / "tools/launcher/common/specdec/qwen4b_b_atomic.py"
    payload = publisher.read_bytes()
    publisher_oid = hashlib.sha1(
        b"blob " + str(len(payload)).encode() + b"\0" + payload
    ).hexdigest()
    return ("python3", "-", str(receipt), str(publisher), publisher_oid)


def _export_validation_program() -> str:
    source = _RUNNER.read_text()
    marker = "# Q30_EXPORT_SAFETENSORS_VALIDATOR\n"
    return source.split(marker, 1)[1].split("\n# Q30_EXPORT_SAFETENSORS_VALIDATOR_END", 1)[0]


def _exposure_evidence_program() -> str:
    source = _RUNNER.read_text()
    marker = "# Q30_EXPOSURE_RECONCILIATION\n"
    return source.split(marker, 1)[1].split("<<'PY' &\n", 1)[1].split("\nPY\n", 1)[0]


def _accelerate_seed42_indices_by_rank() -> list[list[int]]:
    """Exercise the runtime's pinned sampler and non-split batch-shard implementation."""
    accelerate = pytest.importorskip("accelerate")
    assert accelerate.__version__ == "1.14.0"
    from accelerate.data_loader import BatchSamplerShard, SeedableRandomSampler
    from torch.utils.data import BatchSampler

    indices_by_rank: list[list[int]] = []
    for rank in range(32):
        sampler = SeedableRandomSampler(range(700_000), data_seed=42)
        batches = BatchSampler(sampler, batch_size=4, drop_last=False)
        sharded_batches = BatchSamplerShard(
            batches,
            num_processes=32,
            process_index=rank,
            split_batches=False,
            even_batches=True,
        )
        rank_indices: list[int] = []
        for batch in sharded_batches:
            if batch is None:
                raise AssertionError("non-split fixed-size batch shard yielded no batch")
            rank_indices.extend(cast("list[int]", batch))
            if len(rank_indices) == 320:
                break
        indices_by_rank.append(rank_indices)
    return indices_by_rank


def test_q30_exposure_receipt_recomputes_data_uuid_and_seed42_sampler_order(
    tmp_path: Path,
) -> None:
    """Unique evidence still fails unless DATA identity and sampler occurrence agree."""
    dataset = tmp_path / "DATA.jsonl"
    ordered_uuid_digest = hashlib.sha256(b"[")
    with dataset.open("wb") as stream:
        for index in range(700_000):
            prompt_uuid = f"{index:064x}"
            stream.write(
                (json.dumps({"prompt_uuid": prompt_uuid}, separators=(",", ":")) + "\n").encode()
            )
            if index:
                ordered_uuid_digest.update(b",")
            ordered_uuid_digest.update(json.dumps(prompt_uuid).encode())
    ordered_uuid_digest.update(b"]")
    dataset_sha256 = hashlib.sha256(dataset.read_bytes()).hexdigest()
    expected_by_rank = _accelerate_seed42_indices_by_rank()
    assert all(len(rank_indices) == 320 for rank_indices in expected_by_rank)
    evidence_root = tmp_path / "evidence"
    evidence_root.mkdir()

    for tampering in (None, "uuid-at-index", "sampler-order", "accelerate-version"):
        records_by_rank = [
            [{"dataset_index": index, "prompt_uuid": f"{index:064x}"} for index in rank_indices]
            for rank_indices in expected_by_rank
        ]
        if tampering == "uuid-at-index":
            records_by_rank[0][0]["prompt_uuid"], records_by_rank[0][1]["prompt_uuid"] = (
                records_by_rank[0][1]["prompt_uuid"],
                records_by_rank[0][0]["prompt_uuid"],
            )
        elif tampering == "sampler-order":
            records_by_rank[0][0], records_by_rank[0][1] = (
                records_by_rank[0][1],
                records_by_rank[0][0],
            )
        for rank, records in enumerate(records_by_rank):
            (evidence_root / f"rank-{rank:05d}.jsonl").write_text(
                "".join(
                    json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n"
                    for record in records
                )
            )
        receipt = tmp_path / f"{tampering or 'valid'}.json"
        subprocess_environment = os.environ | {"SLURM_JOB_ID": "12345"}
        if tampering == "accelerate-version":
            fake_site = tmp_path / "fake-site"
            dist_info = fake_site / "accelerate-9.9.9.dist-info"
            dist_info.mkdir(parents=True)
            (dist_info / "METADATA").write_text("Name: accelerate\nVersion: 9.9.9\n")
            subprocess_environment["PYTHONPATH"] = os.pathsep.join(
                (str(fake_site), subprocess_environment.get("PYTHONPATH", ""))
            )
        result = subprocess.run(
            (
                sys.executable,
                "-",
                str(evidence_root),
                str(receipt),
                str(dataset),
                dataset_sha256,
                ordered_uuid_digest.hexdigest(),
            ),
            input=_exposure_evidence_program(),
            env=subprocess_environment,
            text=True,
            capture_output=True,
            check=False,
        )

        if tampering is None:
            assert result.returncode == 0, result.stderr
            body = json.loads(receipt.read_text())
            assert body["accelerate_version"] == "1.14.0"
            assert body["schema_version"] == "q30t-strict-exposure-evidence-v3"
        else:
            assert result.returncode != 0, tampering
            assert not receipt.exists()
            if tampering == "accelerate-version":
                assert "unapproved Accelerate sampler implementation: 9.9.9" in result.stderr


def test_q30_full_exposure_receipt_proves_exact_final96_and_optimizer_step_1368(
    tmp_path: Path,
) -> None:
    """Full evidence must reconcile all 700K occurrences and the local-3 final batch."""
    accelerate = pytest.importorskip("accelerate")
    assert accelerate.__version__ == "1.14.0"
    from accelerate.data_loader import SeedableRandomSampler

    from modelopt.torch.speculative.plugins.hf_streaming_dataset import Q30ExactBatchSampler

    dataset = tmp_path / "DATA.jsonl"
    ordered_uuid_digest = hashlib.sha256(b"[")
    dataset_digest = hashlib.sha256()
    with dataset.open("wb") as stream:
        for index in range(700_000):
            prompt_uuid = f"{index:064x}"
            line = (json.dumps({"prompt_uuid": prompt_uuid}, separators=(",", ":")) + "\n").encode()
            stream.write(line)
            dataset_digest.update(line)
            if index:
                ordered_uuid_digest.update(b",")
            ordered_uuid_digest.update(json.dumps(prompt_uuid).encode())
    ordered_uuid_digest.update(b"]")
    evidence_root = tmp_path / "evidence"
    evidence_root.mkdir()
    streams = [(evidence_root / f"rank-{rank:05d}.jsonl").open("w") for rank in range(32)]
    try:
        sampler = SeedableRandomSampler(range(700_000), data_seed=42)
        batches = Q30ExactBatchSampler(
            sampler,
            local_batch_size=4,
            trainer_ranks=32,
            exact_exposure_count=700_000,
        )
        for batch_index, batch in enumerate(batches):
            rank = batch_index % 32
            for index in batch:
                streams[rank].write(
                    json.dumps(
                        {"dataset_index": index, "prompt_uuid": f"{index:064x}"},
                        sort_keys=True,
                        separators=(",", ":"),
                    )
                    + "\n"
                )
    finally:
        for stream in streams:
            stream.close()
    receipt = tmp_path / "full-exposure.json"

    result = subprocess.run(
        (
            sys.executable,
            "-",
            str(evidence_root),
            str(receipt),
            str(dataset),
            dataset_digest.hexdigest(),
            ordered_uuid_digest.hexdigest(),
        ),
        input=_exposure_evidence_program(),
        env=os.environ
        | {
            "EXPECTED_EXPOSURE_COUNT": "700000",
            "EXPECTED_FINAL_GLOBAL_BATCH_SIZE": "96",
            "EXPECTED_STEPS": "1368",
            "SLURM_JOB_ID": "12345",
            "STAGE": "full",
        },
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    body = json.loads(receipt.read_text())
    assert body["schema_version"] == "q30t-strict-exposure-evidence-v4"
    assert body["occurrence_count"] == 700_000
    assert body["optimizer_steps"] == 1_368
    assert body["final_global_batch_size"] == 96
    assert body["final_local_batch_size"] == 3


def _write_safetensors(path: Path, tensors: dict[str, float]) -> None:
    offset = 0
    header: dict[str, object] = {}
    payload = bytearray()
    for name, value in sorted(tensors.items()):
        raw = struct.pack("<f", value)
        header[name] = {"data_offsets": [offset, offset + len(raw)], "dtype": "F32", "shape": [1]}
        payload.extend(raw)
        offset += len(raw)
    encoded = json.dumps(header, separators=(",", ":")).encode()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(len(encoded).to_bytes(8, "little") + encoded + payload)


def test_q30_export_validator_requires_complete_changed_drafter_safetensors(
    tmp_path: Path,
) -> None:
    """A canary export must be a complete draft tensor inventory changed from its parent."""
    parent = tmp_path / "parent/model.safetensors"
    exported = tmp_path / "export/model.safetensors"
    parent_tensors = {
        "model.dflash_module.fc.weight": 1.0,
        "model.dflash_module.hidden_norm.weight": 1.0,
        "model.dflash_module.norm.weight": 1.0,
        **{
            f"model.dflash_module.layers.{layer}.self_attn.q_proj.weight": 1.0 for layer in range(5)
        },
    }
    exported_tensors = {
        name.split("dflash_module.", 1)[1]: (
            2.0 if name.endswith("layers.4.self_attn.q_proj.weight") else value
        )
        for name, value in parent_tensors.items()
    }
    _write_safetensors(parent, parent_tensors)
    _write_safetensors(exported, exported_tensors)
    receipt = tmp_path / "validation.json"

    valid = subprocess.run(
        ("python3", "-", str(parent.parent), str(exported), str(receipt), "DFlash"),
        input=_export_validation_program(),
        text=True,
        capture_output=True,
        check=False,
    )

    assert valid.returncode == 0, valid.stderr
    body = json.loads(receipt.read_text())
    assert body["tensor_count"] == 8
    assert body["changed_tensor_count"] == 1

    receipt.unlink()
    _write_safetensors(
        exported,
        {name.split("dflash_module.", 1)[1]: value for name, value in parent_tensors.items()},
    )
    unchanged = subprocess.run(
        ("python3", "-", str(parent.parent), str(exported), str(receipt), "DFlash"),
        input=_export_validation_program(),
        text=True,
        capture_output=True,
        check=False,
    )

    assert unchanged.returncode != 0
    assert "differs from authenticated parent" in unchanged.stderr
    assert not receipt.exists()

    dspark_parent = tmp_path / "dspark-parent/model.safetensors"
    dspark_export = tmp_path / "dspark-export/model.safetensors"
    dspark_parent_tensors = parent_tensors | {
        "model.dflash_module.markov_w1.weight": 1.0,
        "model.dflash_module.markov_w2.weight": 1.0,
    }
    dspark_export_tensors = {
        name.split("dflash_module.", 1)[1]: value for name, value in dspark_parent_tensors.items()
    }
    dspark_export_tensors["markov_head.markov_w1.weight"] = dspark_export_tensors.pop(
        "markov_w1.weight"
    )
    dspark_export_tensors["markov_head.markov_w2.weight"] = 2.0
    del dspark_export_tensors["markov_w2.weight"]
    _write_safetensors(dspark_parent, dspark_parent_tensors)
    _write_safetensors(dspark_export, dspark_export_tensors)
    dspark_receipt = tmp_path / "dspark-validation.json"

    dspark = subprocess.run(
        (
            "python3",
            "-",
            str(dspark_parent.parent),
            str(dspark_export),
            str(dspark_receipt),
            "DSpark",
        ),
        input=_export_validation_program(),
        text=True,
        capture_output=True,
        check=False,
    )

    assert dspark.returncode == 0, dspark.stderr
    assert json.loads(dspark_receipt.read_text())["tensor_count"] == 10


def test_q30_completion_evidence_executes_and_rejects_unproven_step(
    tmp_path: Path,
) -> None:
    """The controller's actual receipt program proves checkpoint-20 and exported bytes."""
    output = tmp_path / "output"
    checkpoint = output / "checkpoint-20"
    exported = output / "exported-checkpoint-20"
    checkpoint.mkdir(parents=True)
    exported.mkdir()
    control = tmp_path / "control"
    control.mkdir()
    (checkpoint / "trainer_state.json").write_text('{"global_step":20}\n')
    (checkpoint / "model.safetensors").write_bytes(b"checkpoint")
    (exported / "config.json").write_text("{}\n")
    receipt = tmp_path / "canary.json"
    digest = "0" * 64
    environment = os.environ | {
        "DATASET_SHA256": digest,
        "EXPECTED_EXPOSURE_COUNT": "10240",
        "EXPECTED_FINAL_GLOBAL_BATCH_SIZE": "512",
        "EXPECTED_STEPS": "20",
        "CONTROL_ROOT": str(control),
        "METHOD": "DFlash",
        "METHOD_CONFIG_RELATIVE": "modelopt_recipes/general/speculative_decoding/dflash.yaml",
        "ORDERED_PROMPT_UUIDS_SHA256": digest,
        "OUTPUT_ROOT": str(output),
        "PARENT_ARCHIVE_SHA256": digest,
        "PARENT_TREE_SHA256": digest,
        "RUN_NAME": "q30t-dflash-b8-ptv23-swe-heavy-700k-canary",
        "RUNTIME_ARCHIVE_SHA256": digest,
        "RUNTIME_IMAGE_SHA256": digest,
        "RUNTIME_TREE_SHA256": digest,
        "SLURM_JOB_ID": "12345",
        "SOURCE_ARCHIVE_SHA256": digest,
        "SOURCE_COMMIT": "a" * 40,
        "SOURCE_TREE_OID": "b" * 40,
        "SOURCE_TREE_SHA256": digest,
        "STAGE": "canary",
        "TARGET_ARCHIVE_SHA256": digest,
        "TARGET_MODEL": "Qwen/Qwen3-30B-A3B-Thinking-2507",
        "TARGET_TREE_SHA256": digest,
        "TRAINING_ENTRYPOINT_RELATIVE": "tools/launcher/common/eagle3/train_eagle_streaming.sh",
    }
    (control / "exposure-evidence.json").write_text(
        json.dumps(
            {
                "accelerate_version": "1.14.0",
                "dataset_sha256": digest,
                "job_id": "12345",
                "occurrence_count": 10_240,
                "ordered_prompt_uuids_sha256": digest,
                "ordered_rank_occurrence_uuid_sha256": digest,
                "rank_count": 32,
                "sampler_batch_size": 4,
                "sampler_epoch": 0,
                "sampler_num_replicas": 32,
                "sampler_seed": 42,
                "schema_version": "q30t-strict-exposure-evidence-v3",
            },
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    )

    config_only = subprocess.run(
        _completion_evidence_command(receipt),
        input=_completion_evidence_program(),
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )

    assert config_only.returncode != 0
    assert "model weight" in config_only.stderr
    assert not receipt.exists()
    (exported / "model.safetensors").write_bytes(b"exported")
    unvalidated = subprocess.run(
        _completion_evidence_command(receipt),
        input=_completion_evidence_program(),
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )

    assert unvalidated.returncode != 0
    assert "export validation" in unvalidated.stderr
    assert not receipt.exists()
    (control / "export-validation.json").write_text(
        json.dumps(
            {
                "changed_tensor_count": 1,
                "inventory_sha256": digest,
                "method": "DFlash",
                "schema_version": "q30t-export-validation-v1",
                "tensor_count": 8,
            },
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    )
    (control / "container-evidence.json").write_text(
        json.dumps(
            {
                "job_id": "12345",
                "node_count": 16,
                "ordered_node_sentinel_sha256": digest,
                "schema_version": "q30t-named-container-reconciliation-v1",
            },
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    )
    proven = subprocess.run(
        _completion_evidence_command(receipt),
        input=_completion_evidence_program(),
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )

    assert proven.returncode == 0, proven.stderr
    body = json.loads(receipt.read_text())
    assert body["steps"] == 20
    assert body["checkpoint_tree_sha256"]
    assert body["export_model_sha256"]
    assert body["export_tree_sha256"]
    assert body["export_validation_sha256"]
    for field in (
        "dataset_sha256",
        "parent_archive_sha256",
        "parent_tree_sha256",
        "runtime_archive_sha256",
        "runtime_image_sha256",
        "runtime_tree_sha256",
        "source_archive_sha256",
        "source_commit",
        "source_tree_oid",
        "source_tree_sha256",
        "target_archive_sha256",
        "target_tree_sha256",
    ):
        assert body[field] == environment[field.upper()]

    receipt.unlink()

    def limit_completion_write() -> None:
        signal.signal(signal.SIGXFSZ, signal.SIG_IGN)
        resource.setrlimit(resource.RLIMIT_FSIZE, (256, 256))

    interrupted = subprocess.run(
        _completion_evidence_command(receipt),
        input=_completion_evidence_program(),
        env=environment | {"PYTHONDONTWRITEBYTECODE": "1"},
        text=True,
        capture_output=True,
        check=False,
        preexec_fn=limit_completion_write,
    )

    assert interrupted.returncode != 0
    assert not receipt.exists(), "an interrupted receipt write must not occupy the final pathname"

    retried = subprocess.run(
        _completion_evidence_command(receipt),
        input=_completion_evidence_program(),
        env=environment | {"PYTHONDONTWRITEBYTECODE": "1"},
        text=True,
        capture_output=True,
        check=False,
    )

    assert retried.returncode == 0, retried.stderr
    assert json.loads(receipt.read_text()) == body

    adopted = subprocess.run(
        _completion_evidence_command(receipt),
        input=_completion_evidence_program(),
        env=environment | {"PYTHONDONTWRITEBYTECODE": "1"},
        text=True,
        capture_output=True,
        check=False,
    )

    assert adopted.returncode == 0, adopted.stderr
    assert json.loads(receipt.read_text()) == body

    receipt.write_bytes(b"foreign\n")
    mismatched = subprocess.run(
        _completion_evidence_command(receipt),
        input=_completion_evidence_program(),
        env=environment | {"PYTHONDONTWRITEBYTECODE": "1"},
        text=True,
        capture_output=True,
        check=False,
    )

    assert mismatched.returncode != 0
    assert receipt.read_bytes() == b"foreign\n"

    receipt.unlink()
    (checkpoint / "trainer_state.json").write_text('{"global_step":19}\n')
    unproven = subprocess.run(
        _completion_evidence_command(receipt),
        input=_completion_evidence_program(),
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )

    assert unproven.returncode != 0
    assert "global_step" in unproven.stderr
    assert not receipt.exists()


@pytest.mark.parametrize(
    "keeper_failure",
    [
        "none",
        "before_readiness",
        "during_probe",
        "during_training",
        "during_reconciliation",
        "after_training",
        "stop_failure",
    ],
)
def test_q30_controller_executes_mocked_keeper_and_training_phases(
    tmp_path: Path,
    keeper_failure: str,
) -> None:
    """The spooled controller consumes shared rendezvous and publishes proven evidence."""
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    fake_python = fake_bin / "python3"
    fake_python.write_text(
        "#!/bin/bash\n"
        "set -euo pipefail\n"
        'if [[ "${Q30_FAKE_KEEPER_FAILURE:-none}" == after_training '
        '&& "${2:-}" == "${CANARY_RECEIPT:-}" ]]; then\n'
        '  touch "$Q30_FAKE_COMPLETION_STARTED"\n'
        '  [[ -f "$Q30_FAKE_KEEPER_STOPPED" ]] || exit 58\n'
        "  sleep 0.1\n"
        "fi\n"
        f'exec {sys.executable} "$@"\n'
    )
    fake_python.chmod(0o755)
    fake_srun = fake_bin / "srun"
    fake_srun.write_text(
        "#!/bin/bash\n"
        "set -euo pipefail\n"
        'case "$*" in\n'
        '  *"validate_keeper_receipt"*) phase=keeper ;;\n'
        '  *"q30t-named-container-evidence-v1"*) phase=probe ;;\n'
        '  *"exposure-evidence.json"*) phase=reconciliation ;;\n'
        "  *) phase=training ;;\n"
        "esac\n"
        'printf "%s:%s\\n" "$phase" "${SLURM_EXPORT_ENV:-missing}" '
        '>> "$Q30_FAKE_SRUN_PHASES"\n'
        'if [[ "$*" == *"validate_keeper_receipt"* ]]; then\n'
        "  python3 - <<'PY'\n"
        "import json, os\n"
        "from pathlib import Path\n"
        "root = Path(os.environ['CONTROL_ROOT'])\n"
        "count = 15 if os.environ.get('Q30_FAKE_KEEPER_FAILURE') == 'before_readiness' else 16\n"
        "for index in range(count):\n"
        "    node = f'node-{index:02d}'\n"
        "    body = {'job_id': os.environ['SLURM_JOB_ID'], 'node_name': node}\n"
        "    (root / f'node-evidence.keeper.{node}.json').write_text(json.dumps(body))\n"
        "PY\n"
        '  if [[ "${Q30_FAKE_KEEPER_FAILURE:-none}" == before_readiness ]]; then exit 54; fi\n'
        '  if [[ "${Q30_FAKE_KEEPER_FAILURE:-none}" == during_training ]]; then\n'
        '    while [[ ! -f "$Q30_FAKE_TRAINING_PID" ]]; do sleep 0.01; done\n'
        "    exit 55\n"
        "  fi\n"
        '  if [[ "${Q30_FAKE_KEEPER_FAILURE:-none}" == during_probe ]]; then\n'
        '    while [[ ! -f "$Q30_FAKE_PROBE_STARTED" ]]; do sleep 0.01; done\n'
        '    touch "$Q30_FAKE_KEEPER_DIED"\n'
        "    exit 64\n"
        "  fi\n"
        '  if [[ "${Q30_FAKE_KEEPER_FAILURE:-none}" == after_training ]]; then\n'
        '    while [[ ! -f "$Q30_FAKE_COMPLETION_STARTED" '
        '&& ! -f "$CONTROL_ROOT/stop-keepers" ]]; do sleep 0.01; done\n'
        '    [[ -f "$Q30_FAKE_COMPLETION_STARTED" ]] && exit 56\n'
        '    touch "$Q30_FAKE_KEEPER_STOPPED"\n'
        "    exit 0\n"
        "  fi\n"
        '  if [[ "${Q30_FAKE_KEEPER_FAILURE:-none}" == during_reconciliation ]]; then\n'
        '    while [[ ! -f "$Q30_FAKE_RECONCILIATION_STARTED" '
        '&& ! -f "$CONTROL_ROOT/stop-keepers" ]]; do sleep 0.01; done\n'
        '    if [[ -f "$Q30_FAKE_RECONCILIATION_STARTED" ]]; then\n'
        '      touch "$Q30_FAKE_KEEPER_DIED"\n'
        "      exit 62\n"
        "    fi\n"
        "    exit 0\n"
        "  fi\n"
        '  if [[ "${Q30_FAKE_KEEPER_FAILURE:-none}" == stop_failure ]]; then\n'
        '    while [[ ! -f "$CONTROL_ROOT/stop-keepers" ]]; do sleep 0.01; done\n'
        "    exit 57\n"
        "  fi\n"
        '  while [[ ! -f "$CONTROL_ROOT/stop-keepers" ]]; do sleep 0.01; done\n'
        "  exit 0\n"
        "fi\n"
        'if [[ "$*" == *"exposure-evidence.json"* ]]; then\n'
        '  [[ "$*" == *"--container-mounts="* ]] || exit 59\n'
        '  [[ ! -f "$CONTROL_ROOT/stop-keepers" ]] || exit 61\n'
        '  touch "$Q30_FAKE_RECONCILIATION_STARTED"\n'
        '  if [[ "${Q30_FAKE_KEEPER_FAILURE:-none}" == during_reconciliation ]]; then\n'
        '    while [[ ! -f "$Q30_FAKE_KEEPER_DIED" ]]; do sleep 0.01; done\n'
        "    while true; do sleep 0.05; done\n"
        "  fi\n"
        "  python3 - <<'PY'\n"
        "import json, os\n"
        "from pathlib import Path\n"
        "body = {'accelerate_version': '1.14.0', "
        "'dataset_sha256': os.environ['DATASET_SHA256'], 'job_id': "
        "os.environ['SLURM_JOB_ID'], 'occurrence_count': 10240, "
        "'ordered_prompt_uuids_sha256': os.environ['ORDERED_PROMPT_UUIDS_SHA256'], "
        "'ordered_rank_occurrence_uuid_sha256': '0' * 64, 'rank_count': 32, "
        "'sampler_batch_size': 4, 'sampler_epoch': 0, 'sampler_num_replicas': 32, "
        "'sampler_seed': 42, "
        "'schema_version': 'q30t-strict-exposure-evidence-v3'}\n"
        "Path(os.environ['CONTROL_ROOT'], 'exposure-evidence.json').write_text(\n"
        "    json.dumps(body, sort_keys=True, separators=(',', ':')) + '\\n'\n"
        ")\n"
        "PY\n"
        "  exit 0\n"
        "fi\n"
        '[[ "$*" == *"${CONTROL_ROOT}/rendezvous:/scratchspace"* ]] || exit 41\n'
        'if [[ "$*" == *"q30t-named-container-evidence-v1"* ]]; then\n'
        '  [[ "$*" == *"--container-image="* ]] || exit 42\n'
        '  if [[ "${Q30_FAKE_KEEPER_FAILURE:-none}" == during_probe ]]; then\n'
        '    touch "$Q30_FAKE_PROBE_STARTED"\n'
        '    while [[ ! -f "$Q30_FAKE_KEEPER_DIED" ]]; do sleep 0.01; done\n'
        "    while true; do sleep 0.05; done\n"
        "  fi\n"
        "  python3 - <<'PY'\n"
        "import hashlib, json, os\n"
        "from pathlib import Path\n"
        "root = Path(os.environ['CONTROL_ROOT'])\n"
        "for index in range(16):\n"
        "    node = f'node-{index:02d}'\n"
        "    sentinel = (json.dumps({'job_id': os.environ['SLURM_JOB_ID'], 'node_name': node, "
        "'runtime_image_sha256': os.environ['RUNTIME_IMAGE_SHA256'], "
        "'runtime_tree_sha256': os.environ['RUNTIME_TREE_SHA256'], "
        "'schema_version': 'q30t-named-container-sentinel-v1', "
        "'source_archive_sha256': os.environ['SOURCE_ARCHIVE_SHA256'], "
        "'source_commit': os.environ['SOURCE_COMMIT'], "
        "'source_tree_oid': os.environ['SOURCE_TREE_OID'], "
        "'source_tree_sha256': os.environ['SOURCE_TREE_SHA256']}, "
        "sort_keys=True, separators=(',', ':')) + '\\n').encode()\n"
        "    body = {'job_id': os.environ['SLURM_JOB_ID'], 'node_name': node, "
        "'runtime_image_sha256': os.environ['RUNTIME_IMAGE_SHA256'], "
        "'runtime_tree_sha256': os.environ['RUNTIME_TREE_SHA256'], "
        "'schema_version': 'q30t-named-container-evidence-v1', "
        "'sentinel_sha256': hashlib.sha256(sentinel).hexdigest(), "
        "'source_archive_sha256': os.environ['SOURCE_ARCHIVE_SHA256'], "
        "'source_commit': os.environ['SOURCE_COMMIT'], "
        "'source_tree_oid': os.environ['SOURCE_TREE_OID'], "
        "'source_tree_sha256': os.environ['SOURCE_TREE_SHA256']}\n"
        "    (root / f'node-evidence.container.{node}.json').write_text(json.dumps(body))\n"
        "PY\n"
        "  exit 0\n"
        "fi\n"
        '[[ "$*" == *"--container-name="* && "$*" != *"--container-image="* ]] || exit 43\n'
        'if [[ "${Q30_FAKE_KEEPER_FAILURE:-none}" == during_training ]]; then\n'
        '  printf \'%s\\n\' "$$" > "$Q30_FAKE_TRAINING_PID"\n'
        "  while true; do sleep 0.05; done\n"
        "fi\n"
        'mkdir -p "$OUTPUT_ROOT/checkpoint-20" "$OUTPUT_ROOT/exported-checkpoint-20"\n'
        'printf \'{"global_step":20}\\n\' > "$OUTPUT_ROOT/checkpoint-20/trainer_state.json"\n'
        'printf checkpoint > "$OUTPUT_ROOT/checkpoint-20/model.safetensors"\n'
        "printf '{}\\n' > \"$OUTPUT_ROOT/exported-checkpoint-20/config.json\"\n"
        'printf exported > "$OUTPUT_ROOT/exported-checkpoint-20/model.safetensors"\n'
        "python3 - <<'PY'\n"
        "import json, os\n"
        "from pathlib import Path\n"
        "root = Path(os.environ['CONTROL_ROOT'])\n"
        "evidence = root / 'rendezvous/exposure-evidence'\n"
        "evidence.mkdir(parents=True, exist_ok=True)\n"
        "for rank in range(32):\n"
        "    with (evidence / f'rank-{rank:05d}.jsonl').open('w') as stream:\n"
        "        for offset in range(320):\n"
        "            index = rank * 320 + offset\n"
        "            body = {'dataset_index': index, 'prompt_uuid': f'{index:064x}'}\n"
        "            stream.write(json.dumps(body, sort_keys=True, separators=(',', ':')) + '\\n')\n"
        "validation = {'changed_tensor_count': 1, 'inventory_sha256': '0' * 64, "
        "'method': os.environ['METHOD'], 'schema_version': 'q30t-export-validation-v1', "
        "'tensor_count': 8}\n"
        "(root / 'export-validation.json').write_text(\n"
        "    json.dumps(validation, sort_keys=True, separators=(',', ':')) + '\\n'\n"
        ")\n"
        "PY\n"
    )
    fake_srun.chmod(0o755)
    checkout = tmp_path / "bootstrap-checkout"
    bootstrap_module = checkout / "tools/launcher/common/specdec/q30t_ptv23_continuation.py"
    bootstrap_module.parent.mkdir(parents=True)
    bootstrap_module.write_text(
        "import os\n"
        "import shutil\n"
        "import sys\n"
        "index = sys.argv.index('--environment-output')\n"
        "shutil.copyfile(os.environ['REPLAY_ENVIRONMENT_FIXTURE'], sys.argv[index + 1])\n"
    )
    shutil.copyfile(
        _REPO_ROOT / "tools/launcher/common/specdec/qwen4b_b_atomic.py",
        bootstrap_module.parent / "qwen4b_b_atomic.py",
    )
    subprocess.run(("git", "init", "-q", str(checkout)), check=True)
    subprocess.run(("git", "-C", str(checkout), "add", "."), check=True)
    subprocess.run(
        (
            "git",
            "-C",
            str(checkout),
            "-c",
            "user.name=Q30 Test",
            "-c",
            "user.email=q30@example.invalid",
            "commit",
            "-qm",
            "trusted mocked replay",
        ),
        check=True,
    )
    bootstrap_commit = subprocess.run(
        ("git", "-C", str(checkout), "rev-parse", "HEAD"),
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    inputs = tmp_path / "inputs"
    inputs.mkdir()
    input_names = (
        "source.tar.zst",
        "target.tar.zst",
        "parent.tar.zst",
        "runtime.tar.zst",
        "runtime.sqsh",
        "DATA.jsonl",
        "ptv23_node_keeper.py",
    )
    for name in input_names:
        (inputs / name).write_bytes(b"fixture")
    output = tmp_path / "q30t-dflash-b8-ptv23-swe-heavy-700k-canary"
    control = tmp_path / "control"
    receipt = tmp_path / "receipt.json"
    digest = "0" * 64
    environment = os.environ | {
        "CANARY_RECEIPT": str(receipt),
        "CONTROL_ROOT": str(control),
        "DATASET": str(inputs / "DATA.jsonl"),
        "DATASET_SHA256": digest,
        "EXPECTED_EXPOSURE_COUNT": "10240",
        "EXPECTED_FINAL_GLOBAL_BATCH_SIZE": "512",
        "EXPECTED_STEPS": "20",
        "KEEPER_TOOL": str(inputs / "ptv23_node_keeper.py"),
        "KEEPER_TOOL_SHA256": digest,
        "METHOD": "DFlash",
        "METHOD_CONFIG_RELATIVE": "modelopt_recipes/general/speculative_decoding/dflash.yaml",
        "ORDERED_PROMPT_UUIDS_SHA256": digest,
        "OUTPUT_ROOT": str(output),
        "PARENT_ARCHIVE": str(inputs / "parent.tar.zst"),
        "PARENT_ARCHIVE_SHA256": digest,
        "PARENT_TREE_SHA256": digest,
        "PATH": f"{fake_bin}:{os.environ['PATH']}",
        "RUN_NAME": output.name,
        "RUNTIME_ARCHIVE": str(inputs / "runtime.tar.zst"),
        "RUNTIME_ARCHIVE_SHA256": digest,
        "RUNTIME_IMAGE": str(inputs / "runtime.sqsh"),
        "RUNTIME_IMAGE_SHA256": digest,
        "RUNTIME_TREE_SHA256": digest,
        "SLURM_GPUS_ON_NODE": "4",
        "SLURM_JOB_ID": "12345",
        "SLURM_NNODES": "16",
        "SOURCE_ARCHIVE": str(inputs / "source.tar.zst"),
        "SOURCE_ARCHIVE_SHA256": digest,
        "SOURCE_COMMIT": bootstrap_commit,
        "SOURCE_TREE_OID": "b" * 40,
        "SOURCE_TREE_SHA256": digest,
        "STAGE": "canary",
        "TARGET_ARCHIVE": str(inputs / "target.tar.zst"),
        "TARGET_ARCHIVE_SHA256": digest,
        "TARGET_MODEL": "Qwen/Qwen3-30B-A3B-Thinking-2507",
        "TARGET_TREE_SHA256": digest,
        "TRAINING_ENTRYPOINT_RELATIVE": "tools/launcher/common/eagle3/train_eagle_streaming.sh",
        "USER": "test-user",
        "Q30_FAKE_KEEPER_FAILURE": keeper_failure,
        "Q30_FAKE_TRAINING_PID": str(tmp_path / "training-pid"),
        "Q30_FAKE_PROBE_STARTED": str(tmp_path / "probe-started"),
        "Q30_FAKE_COMPLETION_STARTED": str(tmp_path / "completion-started"),
        "Q30_FAKE_KEEPER_STOPPED": str(tmp_path / "keeper-stopped"),
        "Q30_FAKE_KEEPER_DIED": str(tmp_path / "keeper-died"),
        "Q30_FAKE_RECONCILIATION_STARTED": str(tmp_path / "reconciliation-started"),
        "Q30_FAKE_SRUN_PHASES": str(tmp_path / "srun-phases"),
    }
    replay_fixture = tmp_path / "replay-environment.bin"
    replay_fixture.write_bytes(
        b"".join(
            name.encode() + b"\0" + value.encode() + b"\0"
            for name, value in sorted(environment.items())
            if name
            in {
                "CANARY_RECEIPT",
                "CONTROL_ROOT",
                "DATASET",
                "DATASET_SHA256",
                "EXPECTED_EXPOSURE_COUNT",
                "EXPECTED_FINAL_GLOBAL_BATCH_SIZE",
                "EXPECTED_STEPS",
                "KEEPER_TOOL",
                "KEEPER_TOOL_SHA256",
                "METHOD",
                "METHOD_CONFIG_RELATIVE",
                "ORDERED_PROMPT_UUIDS_SHA256",
                "OUTPUT_ROOT",
                "PARENT_ARCHIVE",
                "PARENT_ARCHIVE_SHA256",
                "PARENT_TREE_SHA256",
                "RUN_NAME",
                "RUNTIME_ARCHIVE",
                "RUNTIME_ARCHIVE_SHA256",
                "RUNTIME_IMAGE",
                "RUNTIME_IMAGE_SHA256",
                "RUNTIME_TREE_SHA256",
                "SOURCE_ARCHIVE",
                "SOURCE_ARCHIVE_SHA256",
                "SOURCE_COMMIT",
                "SOURCE_TREE_OID",
                "SOURCE_TREE_SHA256",
                "STAGE",
                "TARGET_ARCHIVE",
                "TARGET_ARCHIVE_SHA256",
                "TARGET_MODEL",
                "TARGET_TREE_SHA256",
                "TRAINING_ENTRYPOINT_RELATIVE",
            }
        )
    )
    environment["REPLAY_ENVIRONMENT_FIXTURE"] = str(replay_fixture)
    descriptor = tmp_path / "launch-descriptor.json"
    descriptor.write_text(
        json.dumps(
            {
                "contract": {
                    "source_checkout_path": str(checkout),
                    "source_commit": bootstrap_commit,
                },
                "schema_version": "q30t-ptv23-launch-descriptor-v1",
                "stage": "canary",
            },
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    )
    descriptor_sha256 = hashlib.sha256(descriptor.read_bytes()).hexdigest()

    result = subprocess.run(
        (
            "bash",
            str(_RUNNER),
            str(descriptor),
            descriptor_sha256,
            str(receipt),
            str(checkout),
        ),
        env=environment,
        text=True,
        capture_output=True,
        check=False,
        timeout=10,
    )

    if keeper_failure in {
        "before_readiness",
        "during_probe",
        "during_training",
        "during_reconciliation",
        "stop_failure",
    }:
        assert result.returncode != 0
        if keeper_failure == "before_readiness":
            assert "keeper exited before all 16 readiness receipts" in result.stderr
        if keeper_failure == "during_training":
            assert "keeper exited before training completed" in result.stderr
            training_pid = int((tmp_path / "training-pid").read_text())
            with pytest.raises(ProcessLookupError):
                os.kill(training_pid, 0)
        if keeper_failure == "during_probe":
            assert "keeper exited during container probe" in result.stderr
        if keeper_failure == "during_reconciliation":
            assert "keeper exited during exposure reconciliation" in result.stderr
        assert not receipt.exists()
        return

    assert result.returncode == 0, result.stderr
    if keeper_failure == "after_training":
        assert (tmp_path / "keeper-stopped").exists()
    body = json.loads(receipt.read_text())
    assert body["steps"] == 20
    assert body["container_evidence_sha256"]
    assert (tmp_path / "srun-phases").read_text().splitlines() == [
        "keeper:ALL",
        "probe:ALL",
        "training:ALL",
        "reconciliation:ALL",
    ]


@pytest.mark.parametrize(
    ("missing_evidence", "expected"),
    [("keeper", "keeper"), ("runtime", "runtime"), ("node", "node")],
)
def test_q30_controller_fails_closed_without_required_platform_evidence(
    missing_evidence: str, expected: str
) -> None:
    """No platform primitive may silently degrade to mutable shared-path execution."""
    assert _RUNNER.is_file(), "Q30 controller is missing"
    source = _RUNNER.read_text()
    guard = {
        "keeper": "O_TMPFILE",
        "runtime": "runtime attestation",
        "node": "exactly 16",
    }[missing_evidence]
    assert guard in source, f"missing fail-closed {expected} evidence guard"
