# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Behavioral tests for the Q30 Thinking 700K continuation launch contract."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
from dataclasses import replace
from typing import TYPE_CHECKING, cast

import pytest
from common.specdec import q30t_ptv23_continuation as continuation
from common.specdec.q30t_parent_receipt import Q30TParentReceipt
from common.specdec.q30t_ptv23_continuation import (
    ExposureSchedule,
    Method,
    Q30TContinuationContract,
    build_q30_node_local_training_command,
    build_q30_training_command,
    require_q30_parent_pair_for_shared_dataset,
)

if TYPE_CHECKING:
    from pathlib import Path


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write(path: Path, content: bytes) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    return path


def _contract(tmp_path: Path, *, method: Method = "DFlash") -> Q30TContinuationContract:
    source = tmp_path / "source"
    entrypoint = _write(
        source / "tools/launcher/common/eagle3/train_eagle_streaming.sh",
        b"#!/bin/bash\nexit 0\n",
    )
    config = _write(
        source / f"modelopt_recipes/general/speculative_decoding/{method.lower()}.yaml",
        b"metadata: {}\n",
    )
    _write(
        source / "tools/launcher/common/specdec/run_q30t_ptv23_continuation.sbatch",
        b"#!/bin/bash\nexit 0\n",
    )
    _write(
        source / "tools/launcher/common/specdec/ptv23_node_keeper.py",
        b"# authenticated keeper fixture\n",
    )
    dataset = _write(tmp_path / "dataset/DATA.jsonl", b'{"uuid":"new"}\n')
    dataset_manifest = _write(tmp_path / "dataset/MANIFEST.json", b"manifest\n")
    dataset_completion = _write(tmp_path / "dataset/COMPLETION.json", b"completion\n")
    tokenizer_receipt = _write(tmp_path / "tokenizer/TOKENIZER.json", b"tokenizer-receipt\n")
    parent_receipt = _write(tmp_path / f"parent/{method}.json", b"parent-receipt\n")
    source_archive = _write(tmp_path / "archives/source.tar.zst", b"source-archive\n")
    target_archive = _write(tmp_path / "archives/target.tar.zst", b"target-archive\n")
    parent_archive = _write(
        tmp_path / f"archives/{method.lower()}-parent.tar.zst", b"parent-archive\n"
    )
    runtime_archive = _write(tmp_path / "runtime/runtime.tar.zst", b"runtime\n")
    runtime_image = _write(tmp_path / "image/image.sqsh", b"image\n")
    runtime_path = tmp_path / "node-local/runtime"
    runtime_path.mkdir(parents=True)
    target = tmp_path / "target/model"
    target.mkdir(parents=True)
    parent_checkpoint = tmp_path / f"parent/{method}/resume-checkpoint-025391"
    parent_checkpoint.mkdir(parents=True)
    subprocess.run(("git", "init", "-q"), cwd=source, check=True)
    subprocess.run(("git", "add", "."), cwd=source, check=True)
    subprocess.run(
        (
            "git",
            "-c",
            "user.name=Q30 test",
            "-c",
            "user.email=q30@example.com",
            "-c",
            "commit.gpgsign=false",
            "commit",
            "-qm",
            "fixture",
        ),
        cwd=source,
        check=True,
    )
    source_commit = subprocess.check_output(
        ("git", "rev-parse", "HEAD"), cwd=source, text=True
    ).strip()
    return Q30TContinuationContract(
        target_model="Qwen/Qwen3-30B-A3B-Thinking-2507",
        target_revision="a" * 40,
        target_tree_sha256="b" * 64,
        target_checkpoint_path=target,
        method=method,
        block_size=8,
        dataset_path=dataset,
        dataset_sha256=_sha(dataset),
        ordered_prompt_uuids_sha256="c" * 64,
        dataset_manifest_path=dataset_manifest,
        dataset_manifest_file_sha256=_sha(dataset_manifest),
        dataset_completion_receipt_path=dataset_completion,
        dataset_completion_receipt_file_sha256=_sha(dataset_completion),
        historical_receipt_file_sha256="d" * 64,
        historical_ordered_prompt_uuids_sha256="e" * 64,
        dataset_records=700_000,
        tokenizer_receipt_path=tokenizer_receipt,
        tokenizer_receipt_file_sha256=_sha(tokenizer_receipt),
        parent_checkpoint_path=parent_checkpoint,
        parent_checkpoint_tree_sha256="1" * 64,
        parent_receipt_path=parent_receipt,
        parent_receipt_file_sha256=_sha(parent_receipt),
        source_checkout_path=source,
        source_commit=source_commit,
        source_archive_path=source_archive,
        source_archive_sha256=_sha(source_archive),
        parent_source_commit="f" * 40,
        target_archive_path=target_archive,
        target_archive_sha256=_sha(target_archive),
        parent_archive_path=parent_archive,
        parent_archive_sha256=_sha(parent_archive),
        runtime_archive_path=runtime_archive,
        runtime_archive_sha256=_sha(runtime_archive),
        runtime_path=runtime_path,
        runtime_tree_sha256=continuation._tree_sha256(  # pyright: ignore[reportPrivateUsage]
            runtime_path
        ),
        runtime_image_path=runtime_image,
        runtime_image_sha256=_sha(runtime_image),
        training_entrypoint=entrypoint,
        training_entrypoint_sha256=_sha(entrypoint),
        method_config_path=config,
        method_config_sha256=_sha(config),
        output_root=(tmp_path / f"output/q30t-{method.lower()}-b8-ptv23-swe-heavy-700k-canary"),
        run_name=f"q30t-{method.lower()}-b8-ptv23-swe-heavy-700k-canary",
    )


def _parent(contract: Q30TContinuationContract) -> Q30TParentReceipt:
    return Q30TParentReceipt(
        method=contract.method,
        target_repository=contract.target_model,
        target_revision=contract.target_revision,
        target_tree_sha256=contract.target_tree_sha256,
        variant="Thinking-2507",
        block_size=8,
        global_step=25391,
        checkpoint_path=contract.parent_checkpoint_path,
        checkpoint_tree_sha256="1" * 64,
        modelopt_state_sha256="2" * 64,
        historical_receipt_file_sha256=contract.historical_receipt_file_sha256,
        historical_ordered_prompt_uuids_sha256=contract.historical_ordered_prompt_uuids_sha256,
        source_commit=contract.parent_source_commit,
        runtime_sha256=contract.runtime_archive_sha256,
        receipt_file_sha256=contract.parent_receipt_file_sha256,
    )


def _authorize_contract(
    monkeypatch: pytest.MonkeyPatch,
    contract: Q30TContinuationContract,
    *,
    authorize_dataset: bool = True,
    authorize_runtime_image: bool = True,
    authorize_source: bool = True,
) -> None:
    monkeypatch.setattr(
        continuation,
        "APPROVED_Q30T_RUNTIME_TREE_SHA256_BY_ARCHIVE_SHA256",
        {contract.runtime_archive_sha256: contract.runtime_tree_sha256},
    )
    if authorize_runtime_image:
        monkeypatch.setattr(
            continuation,
            "APPROVED_Q30T_PRODUCTION_RUNTIME_IMAGE_SHA256S",
            frozenset({contract.runtime_image_sha256}),
            raising=False,
        )
    if authorize_source and hasattr(continuation, "_source_checkout_identity"):
        source_tree_oid, source_tree_sha256 = continuation._source_checkout_identity(  # pyright: ignore[reportPrivateUsage]
            contract.source_checkout_path,
            expected_commit=contract.source_commit,
        )
        monkeypatch.setattr(
            continuation,
            "APPROVED_Q30T_SOURCE_BY_ARCHIVE_SHA256",
            {
                contract.source_archive_sha256: (
                    contract.source_commit,
                    source_tree_oid,
                    source_tree_sha256,
                )
            },
            raising=False,
        )
    monkeypatch.setattr(
        continuation,
        "load_q30t_parent_receipt",
        lambda path, expected_sha256: _parent(contract),
    )
    monkeypatch.setattr(
        continuation,
        "verify_q30t_tokenizer_receipt",
        lambda raw: {
            "repository": contract.target_model,
            "revision": contract.target_revision,
            "snapshot_path": str(contract.target_checkpoint_path),
            "snapshot_tree_sha256": contract.target_tree_sha256,
        },
    )
    if authorize_dataset:
        monkeypatch.setattr(continuation, "_authenticate_dataset_bundle", lambda contract: None)


def test_q30_schedule_is_exactly_the_approved_700k_schedule() -> None:
    """Every schedule field is frozen; arithmetic equivalence cannot change the study."""
    schedule = ExposureSchedule()

    assert schedule == ExposureSchedule(
        consumed_examples=700_000,
        full_steps=1_368,
        nominal_global_batch_size=512,
        final_global_batch_size=96,
        canary_steps=20,
    )
    assert schedule.examples_for_step(1_367) == 512
    assert schedule.examples_for_step(1_368) == 96
    schedule.validate(trainer_ranks=32)

    with pytest.raises(ValueError, match="approved exact 700K schedule"):
        replace(schedule, full_steps=1_369).validate(trainer_ranks=32)
    with pytest.raises(ValueError, match="exact integers"):
        replace(schedule, canary_steps=cast("int", True)).validate(trainer_ranks=32)


def test_q30_dataset_contract_is_the_reviewed_swe_heavy_700k_arm() -> None:
    """Continuation must authenticate the same SWE-heavy identity built upstream."""
    validator = getattr(continuation, "_require_dataset_identity", None)
    assert callable(validator), "SWE-heavy dataset identity validator is missing"
    swe_heavy_quotas = {
        "ptv2_stem": 200_000,
        "ptv2_multilingual_ja": 25_000,
        "ptv2_multilingual_es": 25_000,
        "ptv2_multilingual_fr": 25_000,
        "ptv2_multilingual_it": 25_000,
        "ptv3_swe_v3": 180_000,
        "ptv3_interactive_agentic_swe": 60_000,
        "ptv3_general_tool_trajectories": 160_000,
    }
    manifest = {
        "scientific_identity": ("ptv2-ptv3-complement-700k-qwen3-30ba3b-thinking-swe-heavy-v1"),
        "row_count": 700_000,
        "quotas": swe_heavy_quotas,
        "category_counts": swe_heavy_quotas,
    }

    validator(manifest)
    assert (
        continuation._run_identity(  # pyright: ignore[reportPrivateUsage]
            "DFlash", "canary"
        )
        == "q30t-dflash-b8-ptv23-swe-heavy-700k-canary"
    )

    balanced = dict(manifest)
    balanced["scientific_identity"] = "ptv2-ptv3-complement-700k-qwen3-30ba3b-thinking-v1"
    with pytest.raises(ValueError, match="SWE-heavy"):
        validator(balanced)


@pytest.mark.parametrize("method", ["DFlash", "DSpark"])
def test_q30_canary_executes_the_real_streaming_launcher_interface(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    method: Method,
) -> None:
    """The generated invocation is accepted by a launcher that rejects invented flags."""
    contract = _contract(tmp_path, method=method)
    _authorize_contract(monkeypatch, contract)
    fake_runner = (
        contract.source_checkout_path
        / "tools/launcher/common/specdec/run_q30t_ptv23_continuation.sbatch"
    )
    fake_runner.write_text(
        """#!/bin/bash
set -euo pipefail
[[ "${STAGE:?}" == canary ]]
[[ "${METHOD:?}" == "$EXPECTED_METHOD" ]]
[[ "${TARGET_ARCHIVE:?}" == "$EXPECTED_TARGET_ARCHIVE" ]]
[[ "${PARENT_ARCHIVE:?}" == "$EXPECTED_PARENT_ARCHIVE" ]]
[[ "${SOURCE_ARCHIVE:?}" == "$EXPECTED_SOURCE_ARCHIVE" ]]
[[ "${TRAINING_ENTRYPOINT_RELATIVE:?}" == tools/launcher/common/eagle3/train_eagle_streaming.sh ]]
[[ "$#" == 0 ]]
"""
    )
    subprocess.run(("git", "add", "."), cwd=contract.source_checkout_path, check=True)
    subprocess.run(
        (
            "git",
            "-c",
            "user.name=Q30 test",
            "-c",
            "user.email=q30@example.com",
            "-c",
            "commit.gpgsign=false",
            "commit",
            "-qm",
            "fake launcher",
        ),
        cwd=contract.source_checkout_path,
        check=True,
    )
    source_commit = subprocess.check_output(
        ("git", "rev-parse", "HEAD"), cwd=contract.source_checkout_path, text=True
    ).strip()
    contract = contract.replace(
        source_commit=source_commit,
    )
    _authorize_contract(monkeypatch, contract)
    launch = build_q30_training_command(contract, stage="canary")
    environment = (
        os.environ
        | launch.environment
        | {
            "EXPECTED_METHOD": method,
            "EXPECTED_TARGET_ARCHIVE": str(contract.target_archive_path),
            "EXPECTED_PARENT_ARCHIVE": str(contract.parent_archive_path),
            "EXPECTED_SOURCE_ARCHIVE": str(contract.source_archive_path),
        }
    )
    result = subprocess.run(
        launch.command,
        cwd=launch.working_directory,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert launch.nodes == 16
    assert launch.gpus_per_node == 4
    assert launch.container_image == contract.runtime_image_path
    assert launch.command == ("/bin/bash", str(fake_runner))
    assert launch.environment["TARGET_TREE_SHA256"] == contract.target_tree_sha256
    assert launch.environment["PARENT_TREE_SHA256"] == contract.parent_checkpoint_tree_sha256
    assert launch.environment["RUNTIME_TREE_SHA256"] == contract.runtime_tree_sha256
    assert "HF_MODEL_CKPT" not in launch.environment
    assert "MODELOPT_RUNTIME" not in launch.environment
    assert "PYTHONPATH" not in launch.environment


@pytest.mark.parametrize("method", ["DFlash", "DSpark"])
def test_q30_full_launch_uses_exact_1368_step_700k_schedule(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, method: Method
) -> None:
    """The authenticated full runner must expose 700K once and finish at step 1,368."""
    contract = _contract(tmp_path, method=method)
    full_name = f"q30t-{method.lower()}-b8-ptv23-swe-heavy-700k-full"
    contract = contract.replace(
        output_root=contract.output_root.parent / full_name,
        run_name=full_name,
    )
    _authorize_contract(monkeypatch, contract)

    launch = build_q30_training_command(contract, stage="full")

    assert launch.environment["STAGE"] == "full"
    assert launch.environment["EXPECTED_STEPS"] == "1368"
    assert launch.environment["EXPECTED_EXPOSURE_COUNT"] == "700000"
    assert launch.environment["EXPECTED_FINAL_GLOBAL_BATCH_SIZE"] == "96"


def test_q30_contract_authenticates_parent_tokenizer_and_input_bytes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Every launch-time file identity and both external receipts are consumed."""
    contract = _contract(tmp_path)
    _authorize_contract(monkeypatch, contract)
    calls: list[tuple[Path, str]] = []

    def load_parent(path: Path, expected_sha256: str) -> Q30TParentReceipt:
        calls.append((path, expected_sha256))
        return _parent(contract)

    monkeypatch.setattr(continuation, "load_q30t_parent_receipt", load_parent)
    monkeypatch.setattr(
        continuation,
        "verify_q30t_tokenizer_receipt",
        lambda raw: {
            "repository": contract.target_model,
            "revision": contract.target_revision,
            "snapshot_path": str(contract.target_checkpoint_path),
            "snapshot_tree_sha256": contract.target_tree_sha256,
        },
    )
    monkeypatch.setattr(continuation, "_authenticate_dataset_bundle", lambda contract: None)

    build_q30_training_command(contract, stage="canary")
    assert calls == [(contract.parent_receipt_path, contract.parent_receipt_file_sha256)]

    contract.dataset_path.write_bytes(b"tampered\n")
    with pytest.raises(ValueError, match="dataset SHA-256"):
        build_q30_training_command(contract, stage="canary")


def test_q30_contract_rejects_caller_forged_one_row_dataset(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A caller-supplied 700K count cannot make a one-row DATA file admissible."""
    contract = _contract(tmp_path)
    _authorize_contract(monkeypatch, contract, authorize_dataset=False)

    with pytest.raises(ValueError, match="exact 700K dataset"):
        build_q30_training_command(contract, stage="canary")


def test_q30_large_artifact_hashing_does_not_retain_the_artifact(tmp_path: Path) -> None:
    """Dataset, image, and runtime hashing must not allocate their full size in memory."""
    artifact = _write(tmp_path / "large.sqsh", b"one-block")

    raw, digest = continuation._stable_file(  # pyright: ignore[reportPrivateUsage]
        artifact, label="large artifact", retain=False
    )

    assert raw is None
    assert digest == _sha(artifact)


def test_q30_canary_requires_external_production_runtime_image_approval(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A matching caller-supplied image digest is not an immutable production trust root."""
    contract = _contract(tmp_path)
    _authorize_contract(monkeypatch, contract, authorize_runtime_image=False)

    with pytest.raises(ValueError, match="production runtime image"):
        build_q30_training_command(contract, stage="canary")


def test_q30_model_archive_tree_uses_parent_receipt_canonical_digest(tmp_path: Path) -> None:
    """Target/parent archive trees use regular-file evidence, not runtime-tree semantics."""
    model_tree_sha256 = getattr(continuation, "_model_tree_sha256", None)
    assert callable(model_tree_sha256), "canonical Q30 model-tree digest is missing"
    root = tmp_path / "model"
    _write(root / "config.json", b"{}\n")
    _write(root / "weights/model.safetensors", b"weights")
    entries = [
        {
            "path": "config.json",
            "sha256": hashlib.sha256(b"{}\n").hexdigest(),
            "size": 3,
            "type": "regular",
        },
        {
            "path": "weights/model.safetensors",
            "sha256": hashlib.sha256(b"weights").hexdigest(),
            "size": 7,
            "type": "regular",
        },
    ]
    expected = hashlib.sha256(
        json.dumps(entries, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()

    assert model_tree_sha256(root) == expected


@pytest.mark.parametrize("method", ["DFlash", "DSpark"])
def test_q30_node_local_invocation_is_accepted_by_strict_real_launcher_interface(
    tmp_path: Path, method: Method
) -> None:
    """The post-attestation command has only --config plus supported dotlist overrides."""
    local_root = tmp_path / "node-local"
    launcher = _write(
        local_root / "source/tools/launcher/common/eagle3/train_eagle_streaming.sh",
        b"""#!/bin/bash
set -euo pipefail
[[ "$1" == --config && -f "$2" ]]
shift 2
for argument in "$@"; do
    [[ "$argument" != --* && "$argument" == *=* ]] || exit 41
done
""",
    )
    launcher.chmod(0o755)
    _write(
        local_root / f"source/modelopt_recipes/general/speculative_decoding/{method.lower()}.yaml",
        b"metadata: {}\n",
    )
    output = tmp_path / f"q30t-{method.lower()}-b8-ptv23-swe-heavy-700k-canary"

    command = build_q30_node_local_training_command(
        method=method,
        local_root=local_root,
        output_root=output,
        run_name=output.name,
    )
    result = subprocess.run(command, check=False, capture_output=True, text=True)

    assert result.returncode == 0, result.stderr
    overrides = dict(item.split("=", 1) for item in command[4:])
    assert overrides["model.model_name_or_path"] == str(local_root / "parent")
    assert overrides["model.tokenizer_name_or_path"] == str(local_root / "target")
    assert overrides["data.data_path"] == str(local_root / "DATA.jsonl")
    assert overrides["training.max_steps"] == "20"
    assert overrides["training.data_seed"] == "42"
    assert overrides["training.per_device_train_batch_size"] == "4"
    assert overrides["training.gradient_accumulation_steps"] == "4"
    assert overrides["training.dataloader_num_workers"] == "0"


def test_q30_contract_authenticates_the_source_checkout_head(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A different checkout cannot run even when selected launcher bytes happen to match."""
    contract = _contract(tmp_path)
    _authorize_contract(monkeypatch, contract)

    with pytest.raises(ValueError, match="source checkout commit"):
        build_q30_training_command(contract.replace(source_commit="0" * 40), stage="canary")


def test_q30_source_archive_requires_independent_commit_bound_approval(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Caller-supplied archive/commit/tree identities cannot approve one another."""
    contract = _contract(tmp_path)
    _authorize_contract(monkeypatch, contract, authorize_source=False)

    with pytest.raises(ValueError, match="approved source archive"):
        build_q30_training_command(contract, stage="canary")


@pytest.mark.parametrize("index_flag", ["--assume-unchanged", "--skip-worktree"])
def test_q30_source_checkout_rejects_git_index_concealment_flags(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, index_flag: str
) -> None:
    """Git index flags cannot hide source bytes from launch authentication."""
    contract = _contract(tmp_path)
    _authorize_contract(monkeypatch, contract)
    relative_entrypoint = contract.training_entrypoint.relative_to(contract.source_checkout_path)
    subprocess.run(
        ("git", "update-index", index_flag, str(relative_entrypoint)),
        cwd=contract.source_checkout_path,
        check=True,
    )

    with pytest.raises(ValueError, match=r"source checkout.*index flags"):
        build_q30_training_command(contract, stage="canary")


def test_q30_canonical_launch_descriptor_roundtrips_and_binds_source_head(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The only scheduler input is a canonical descriptor replayed through full authentication."""
    contract = _contract(tmp_path)
    _authorize_contract(monkeypatch, contract)
    writer = getattr(continuation, "write_q30_launch_descriptor", None)
    loader = getattr(continuation, "load_q30_launch_descriptor", None)
    assert callable(writer), "canonical Q30 launch descriptor writer is missing"
    assert callable(loader), "canonical Q30 launch descriptor replay is missing"
    descriptor = tmp_path / "launch.json"

    descriptor_sha256 = writer(contract, stage="canary", path=descriptor)
    launch = loader(
        descriptor,
        expected_sha256=descriptor_sha256,
        expected_source_checkout=contract.source_checkout_path,
    )

    assert launch == build_q30_training_command(contract, stage="canary")
    payload = json.loads(descriptor.read_bytes())
    assert descriptor.read_bytes() == continuation._canonical_json(payload) + b"\n"  # pyright: ignore[reportPrivateUsage]
    assert payload["contract"]["source_commit"] == contract.source_commit

    payload["contract"]["source_commit"] = "0" * 40
    descriptor.write_bytes(continuation._canonical_json(payload) + b"\n")  # pyright: ignore[reportPrivateUsage]
    forged_sha256 = _sha(descriptor)
    with pytest.raises(ValueError, match="source checkout commit"):
        loader(
            descriptor,
            expected_sha256=forged_sha256,
            expected_source_checkout=contract.source_checkout_path,
        )


def test_q30_launch_descriptor_cli_rejects_noncanonical_action(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Descriptor replay exposes one exact authenticated action, not a generic CLI."""
    with pytest.raises(SystemExit):
        continuation._main(  # pyright: ignore[reportPrivateUsage]
            [
                "substituted-action",
                "--descriptor",
                str(tmp_path / "descriptor.json"),
                "--sha256",
                "0" * 64,
                "--source-checkout",
                str(tmp_path),
                "--environment-output",
                str(tmp_path / "environment.bin"),
            ]
        )

    assert "invalid choice" in capsys.readouterr().err


def test_q30_contract_rejects_mutated_staged_runtime_tree(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The activated runtime cannot gain unreviewed package bytes after staging."""
    contract = _contract(tmp_path)
    _authorize_contract(monkeypatch, contract)
    _write(contract.runtime_path / "injected.py", b"UNREVIEWED = True\n")

    with pytest.raises(ValueError, match="runtime tree SHA-256"):
        build_q30_training_command(contract, stage="canary")


def test_q30_contract_rejects_caller_forged_runtime_tree_and_digest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The parent-approved archive must map to one reviewed extracted runtime tree."""
    contract = _contract(tmp_path)
    _authorize_contract(monkeypatch, contract)
    _write(contract.runtime_path / "injected.py", b"UNREVIEWED = True\n")
    contract = contract.replace(
        runtime_tree_sha256=continuation._tree_sha256(  # pyright: ignore[reportPrivateUsage]
            contract.runtime_path
        )
    )

    with pytest.raises(ValueError, match="reviewed extracted runtime"):
        build_q30_training_command(contract, stage="canary")


def test_q30_contract_rejects_a_dirty_source_checkout(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """HEAD alone cannot authorize working-tree bytes changed outside the pinned commit."""
    contract = _contract(tmp_path)
    _authorize_contract(monkeypatch, contract)
    _write(contract.source_checkout_path / "untracked.py", b"MUTATION = True\n")

    with pytest.raises(ValueError, match="authenticated commit inventory"):
        build_q30_training_command(contract, stage="canary")


def test_q30_contract_rejects_cross_bound_parent_or_tokenizer_substitution(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A valid receipt for another method or target snapshot cannot authorize the run."""
    contract = _contract(tmp_path)
    _authorize_contract(monkeypatch, contract)
    wrong_parent = replace(_parent(contract), method="DSpark")
    monkeypatch.setattr(
        continuation,
        "load_q30t_parent_receipt",
        lambda path, expected_sha256: wrong_parent,
    )
    monkeypatch.setattr(
        continuation,
        "verify_q30t_tokenizer_receipt",
        lambda raw: {
            "repository": contract.target_model,
            "revision": contract.target_revision,
            "snapshot_path": str(contract.target_checkpoint_path),
            "snapshot_tree_sha256": contract.target_tree_sha256,
        },
    )

    with pytest.raises(ValueError, match="parent receipt does not match"):
        build_q30_training_command(contract, stage="canary")


def test_q30_parent_pair_requires_one_shared_historical_lineage(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Both methods must exclude the same authenticated 1.3M ordered exposure."""
    dflash = _contract(tmp_path / "df", method="DFlash")
    dspark = _contract(tmp_path / "ds", method="DSpark")
    parents = {
        dflash.parent_receipt_path: _parent(dflash),
        dspark.parent_receipt_path: _parent(dspark),
    }
    monkeypatch.setattr(
        continuation,
        "load_q30t_parent_receipt",
        lambda path, expected_sha256: parents[path],
    )

    require_q30_parent_pair_for_shared_dataset(dflash, dspark)

    dspark = dspark.replace(historical_ordered_prompt_uuids_sha256="9" * 64)
    parents[dspark.parent_receipt_path] = _parent(dspark)
    with pytest.raises(ValueError, match="historical lineage"):
        require_q30_parent_pair_for_shared_dataset(dflash, dspark)


def test_q30_contract_rejects_non_exact_types_and_overlapping_output(tmp_path: Path) -> None:
    """Booleans cannot pass integer fields and output cannot overlap immutable inputs."""
    contract = _contract(tmp_path)
    with pytest.raises(ValueError, match="proven 16-node topology"):
        contract.replace(nodes=cast("int", True))
    with pytest.raises(ValueError, match="output root"):
        contract.replace(output_root=contract.parent_checkpoint_path / "new-run")


def test_q30_contract_rejects_rw_mount_parent_containing_dataset(tmp_path: Path) -> None:
    """The controller cannot RW-mount an output parent that also contains immutable DATA."""
    contract = _contract(tmp_path)
    sibling_output = contract.dataset_path.parent / contract.run_name

    with pytest.raises(ValueError, match=r"mount source|immutable input"):
        contract.replace(output_root=sibling_output)


def test_q30_contract_rejects_noncanonical_and_symlink_aliased_output(
    tmp_path: Path,
) -> None:
    """Dot-dot and no-follow alias paths cannot evade immutable-input disjointness."""
    contract = _contract(tmp_path)
    with pytest.raises(ValueError, match="canonical"):
        contract.replace(output_root=contract.parent_checkpoint_path / ".." / "DFlash" / "new-run")

    alias = tmp_path / "parent-alias"
    alias.symlink_to(contract.parent_checkpoint_path.parent, target_is_directory=True)
    with pytest.raises(ValueError, match=r"symlink|disjoint|output root"):
        contract.replace(output_root=alias / contract.parent_checkpoint_path.name / "new-run")


@pytest.mark.parametrize("delimiter", [",", ":", "\n", "\x7f"])
def test_q30_contract_rejects_pyxis_delimiters_in_output_mount_parent(
    tmp_path: Path, delimiter: str
) -> None:
    """Caller-controlled output parents cannot inject Pyxis mount-list delimiters."""
    contract = _contract(tmp_path / "fixture")
    output_root = tmp_path / f"output{delimiter}injection" / contract.run_name

    with pytest.raises(ValueError, match="Pyxis"):
        contract.replace(output_root=output_root)


def test_q30_launch_rejects_run_name_not_bound_to_method_and_stage(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A DFlash canary cannot publish under a DSpark or unscoped run identity."""
    contract = _contract(tmp_path)
    _authorize_contract(monkeypatch, contract)

    with pytest.raises(ValueError, match="method and stage"):
        build_q30_training_command(
            contract.replace(run_name="q30t-dspark-b8-ptv23-swe-heavy-700k-canary"),
            stage="canary",
        )
    with pytest.raises(ValueError, match="method and stage"):
        build_q30_training_command(
            contract.replace(output_root=tmp_path / "output/foreign-canary"),
            stage="canary",
        )


def test_q30_contract_rejects_wrong_target_sample_count_or_method(tmp_path: Path) -> None:
    """Scientific identity substitutions must be rejected at construction."""
    contract = _contract(tmp_path)
    with pytest.raises(ValueError, match="Thinking target"):
        contract.replace(target_model="Qwen/Qwen3-30B-A3B")
    with pytest.raises(ValueError, match="700K"):
        contract.replace(dataset_records=699_999)
    with pytest.raises(ValueError, match="method"):
        _contract(tmp_path / "bad", method=cast("Method", "DFlash2"))
