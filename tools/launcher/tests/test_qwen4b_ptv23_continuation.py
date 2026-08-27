# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Typed Qwen3-4B 1.3M-to-700K continuation launcher contracts."""

from __future__ import annotations

import ast
import hashlib
import importlib.util
import inspect
import json
import os
import subprocess
import sys
from argparse import Namespace
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import pytest

if TYPE_CHECKING:
    from collections.abc import Callable

ROOT = Path(__file__).resolve().parents[3]
MODULE_PATH = ROOT / "tools/launcher/common/specdec/qwen4b_ptv23_continuation.py"
RUNNER_PATH = ROOT / "tools/launcher/common/specdec/run_qwen4b_ptv23_continuation.sbatch"
BUILDER_PATH = ROOT / "examples/dataset/build_qwen4b_ptv23_complement.py"


def _canonical(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def _load_module():
    assert MODULE_PATH.is_file(), "the typed continuation launcher module must exist"
    spec = importlib.util.spec_from_file_location("qwen4b_ptv23_continuation", MODULE_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _load_runner_functions(stage_root: Path) -> dict[str, Any]:
    runner = RUNNER_PATH.read_text()
    source = runner.split("python3 -I - <<'PY'\n", 1)[1].rsplit("\nPY", 1)[0]
    parsed = ast.parse(source)
    functions: list[ast.stmt] = [node for node in parsed.body if isinstance(node, ast.FunctionDef)]
    namespace = {
        "hashlib": hashlib,
        "nofollow": getattr(os, "O_NOFOLLOW", 0),
        "os": os,
        "stat": __import__("stat"),
        "stage_root": str(stage_root),
        "subprocess": subprocess,
        "sys": sys,
        "descriptor_prefix": f"/proc/{os.getpid()}/fd",
    }
    exec(compile(ast.Module(body=functions, type_ignores=[]), RUNNER_PATH, "exec"), namespace)
    return namespace


def _load_runner_copy_function(stage_root: Path) -> Callable[[str | Path, str], int]:
    return cast(
        "Callable[[str | Path, str], int]",
        _load_runner_functions(stage_root)["copy_to_anonymous_file"],
    )


def _checkpoint_tree_sha256(root: Path) -> str:
    entries = [
        [path.relative_to(root).as_posix(), hashlib.sha256(path.read_bytes()).hexdigest()]
        for path in sorted(root.rglob("*"))
        if path.is_file()
    ]
    return hashlib.sha256(_canonical(entries)).hexdigest()


def _write_dataset_bundle(
    root: Path,
    *,
    historical_receipt_file_sha256: str,
    historical_ordered_prompt_uuids_sha256: str,
) -> tuple[Path, str, Path, str]:
    root.mkdir()
    data = b"fixture\n"
    (root / "DATA.jsonl").write_bytes(data)
    manifest_body = {
        "schema_version": "ptv2-ptv3-complement-bundle-v1",
        "scientific_identity": "ptv2-ptv3-complement-700k-v1",
        "row_count": 700_000,
        "quotas": {
            "ptv2_stem": 300_000,
            "ptv2_multilingual_ja": 50_000,
            "ptv2_multilingual_es": 50_000,
            "ptv2_multilingual_fr": 50_000,
            "ptv2_multilingual_it": 50_000,
            "ptv3_swe_v3": 100_000,
            "ptv3_interactive_agentic_swe": 19_000,
            "ptv3_general_tool_trajectories": 81_000,
        },
        "category_counts": {
            "ptv2_stem": 300_000,
            "ptv2_multilingual_ja": 50_000,
            "ptv2_multilingual_es": 50_000,
            "ptv2_multilingual_fr": 50_000,
            "ptv2_multilingual_it": 50_000,
            "ptv3_swe_v3": 100_000,
            "ptv3_interactive_agentic_swe": 19_000,
            "ptv3_general_tool_trajectories": 81_000,
        },
        "data": {
            "path": "DATA.jsonl",
            "bytes": len(data),
            "sha256": hashlib.sha256(data).hexdigest(),
        },
        "ordered_prompt_uuids_sha256": "1" * 64,
        "duplicate_uuid_multiplicity": {},
        "source_occurrences_sha256": "2" * 64,
        "selected_token_evidence_sha256": "3" * 64,
        "capacity_receipt_file_sha256": "",
        "capacity": None,
        "exclusions": {},
        "historical": {
            "file_sha256": historical_receipt_file_sha256,
            "receipt_sha256": "4" * 64,
            "occurrence_count": 1_300_000,
            "ordered_prompt_uuids_sha256": historical_ordered_prompt_uuids_sha256,
            "unique_prompt_uuids_sha256": "5" * 64,
            "duplicate_uuid_multiplicity": {},
        },
        "held_out": {
            "receipt_file_sha256s": [],
            "prompt_uuids_sha256": "6" * 64,
            "count": 0,
        },
        "trust": {
            "config_file_sha256": "7" * 64,
            "source_inventory_file_sha256": "8" * 64,
            "tokenizer_trust_file_sha256": "9" * 64,
            "runtime_sha256": "a" * 64,
            "source_commit": "b" * 40,
        },
    }
    manifest = manifest_body | {
        "manifest_sha256": hashlib.sha256(_canonical(manifest_body)).hexdigest()
    }
    manifest_path = root / "MANIFEST.json"
    manifest_path.write_bytes(_canonical(manifest) + b"\n")
    manifest_file_sha256 = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    receipt_body = {
        "schema_version": "ptv2-ptv3-complement-completion-v1",
        "scientific_identity": "ptv2-ptv3-complement-700k-v1",
        "output_root": str(root),
        "row_count": 700_000,
        "manifest_file_sha256": manifest_file_sha256,
        "manifest_sha256": manifest["manifest_sha256"],
        "runtime_sha256": "a" * 64,
        "source_commit": "b" * 40,
    }
    receipt = receipt_body | {
        "receipt_sha256": hashlib.sha256(_canonical(receipt_body)).hexdigest()
    }
    receipt_path = root.parent / "completion.json"
    receipt_path.write_bytes(_canonical(receipt) + b"\n")
    return (
        manifest_path,
        manifest_file_sha256,
        receipt_path,
        hashlib.sha256(receipt_path.read_bytes()).hexdigest(),
    )


@pytest.mark.parametrize("method", ["DFlash", "DSpark"])
def test_continuation_adopts_only_authenticated_weights_with_fresh_training_state(
    tmp_path: Path, method: str, monkeypatch
) -> None:
    """Only exact DFlash/DSpark weights may seed a fresh continuation."""
    module = _load_module()
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    (checkpoint / "model.safetensors").write_bytes(b"weights")
    body = {
        "schema_version": "specdec-authenticated-1.3m-checkpoint-v1",
        "method": method,
        "checkpoint_path": str(checkpoint),
        "checkpoint_tree_sha256": _checkpoint_tree_sha256(checkpoint),
        "historical_receipt_file_sha256": "1" * 64,
        "historical_ordered_prompt_uuids_sha256": "2" * 64,
        "completed_occurrences": 3,
        "runtime_sha256": "3" * 64,
        "source_commit": "4" * 40,
    }
    receipt = body | {"receipt_sha256": hashlib.sha256(_canonical(body)).hexdigest()}
    receipt_path = tmp_path / "checkpoint.json"
    receipt_path.write_bytes(_canonical(receipt) + b"\n")

    parent = module.load_authenticated_parent_checkpoint(
        receipt_path,
        expected_sha256=hashlib.sha256(receipt_path.read_bytes()).hexdigest(),
        expected_occurrences=3,
    )
    dataset_manifest, dataset_manifest_sha256, completion_receipt, completion_sha256 = (
        _write_dataset_bundle(
            tmp_path / "dataset",
            historical_receipt_file_sha256="1" * 64,
            historical_ordered_prompt_uuids_sha256="2" * 64,
        )
    )
    monkeypatch.setattr(module, "_authenticate_dataset_bundle", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(module, "_run_builder_semantic_verifier", lambda **_kwargs: None)
    monkeypatch.setattr(module, "_semantic_protected_paths", lambda **_kwargs: ())
    runtime_image = tmp_path / "runtime.sqsh"
    runtime_image.write_bytes(b"runtime")
    trainer_sha256 = hashlib.sha256(MODULE_PATH.read_bytes()).hexdigest()
    image_sha256 = hashlib.sha256(runtime_image.read_bytes()).hexdigest()
    monkeypatch.setattr(module, "TRUSTED_TRAINING_ENTRYPOINT_SHA256", trainer_sha256)
    monkeypatch.setattr(module, "TRUSTED_RUNTIME_IMAGE_SHA256", image_sha256)
    monkeypatch.setattr(
        module, "TRUSTED_PARENT_CHECKPOINT_RECEIPT_SHA256S", frozenset({parent.receipt_file_sha256})
    )
    contract = module.make_continuation_contract(
        parent,
        dataset_manifest_path=dataset_manifest,
        dataset_manifest_file_sha256=dataset_manifest_sha256,
        dataset_completion_receipt_path=completion_receipt,
        dataset_completion_receipt_file_sha256=completion_sha256,
        dataset_builder_tool_path=MODULE_PATH,
        trajectory_schema_path=MODULE_PATH,
        specdec_identity_path=MODULE_PATH,
        specdec_corpus_contracts_path=MODULE_PATH,
        historical_authenticator_path=MODULE_PATH,
        config_path=MODULE_PATH,
        source_inventory_path=MODULE_PATH,
        source_inventory_file_sha256="5" * 64,
        historical_receipt_path=receipt_path,
        held_out_receipts=tuple(
            (name, MODULE_PATH, str(index) * 64)
            for index, name in enumerate(("speed", "math", "code", "swe", "tool"), start=1)
        ),
        tokenizer_trust_path=tmp_path / "tokenizer.json",
        tokenizer_trust_file_sha256="6" * 64,
        verification_scratch_root=tmp_path / "verify",
        canary_execution_stage_root=tmp_path / "canary-stage",
        canary_checkpoint_output_root=tmp_path / "canary-output",
        canary_evidence_path=tmp_path / "canary-meta" / "evidence.json",
        canary_receipt_path=tmp_path / "canary-meta" / "receipt.json",
        full_execution_stage_root=tmp_path / "full-stage",
        full_checkpoint_output_root=tmp_path / "full-output",
        full_evidence_path=tmp_path / "full-meta" / "evidence.json",
        full_receipt_path=tmp_path / "full-meta" / "receipt.json",
        training_entrypoint_path=MODULE_PATH,
        training_entrypoint_sha256=trainer_sha256,
        runtime_image_path=runtime_image,
        runtime_image_sha256=image_sha256,
        full_steps=14_000,
        global_batch_size=50,
        dataset_identity="ptv2-ptv3-complement-700k-v1",
        run_identity=f"fixture-{method.lower()}",
    )

    assert contract.method == method
    assert contract.parent_checkpoint_tree_sha256 == body["checkpoint_tree_sha256"]
    assert contract.resume_mode == "weights-only"
    assert contract.optimizer_state == "fresh"
    assert contract.scheduler_state == "fresh"
    assert contract.canary_steps == 20
    assert contract.test_only_required is True


def test_submitter_scheduler_checks_before_20_step_canary_and_full_stage() -> None:
    """Every submission passes scheduler validation and canary remains exactly 20 steps."""
    submitter_path = ROOT / "tools/launcher/common/specdec/submit_qwen4b_ptv23_continuation.sh"
    runner_path = ROOT / "tools/launcher/common/specdec/run_qwen4b_ptv23_continuation.sbatch"
    assert submitter_path.is_file()
    assert runner_path.is_file()
    submitter = submitter_path.read_text()
    runner = runner_path.read_text()

    assert "--test-only|--submit-canary|--submit-full" in submitter
    assert 'sbatch --test-only "${args[@]}" "$RUNNER"' in submitter
    assert '[[ "$MODE" == "--test-only" ]]' in submitter
    assert "CANARY_STEPS=20" in runner
    assert 'RESUME_MODE="weights-only"' in runner
    assert 'OPTIMIZER_STATE="fresh"' in runner
    assert 'SCHEDULER_STATE="fresh"' in runner
    assert '[[ "$STAGE" == "canary" ]] && MAX_STEPS="$CANARY_STEPS"' in runner
    subprocess.run(["bash", "-n", str(submitter_path)], check=True)
    subprocess.run(["bash", "-n", str(runner_path)], check=True)


def test_submitter_rejects_export_delimiters_and_does_not_inherit_all(tmp_path: Path) -> None:
    """Caller values cannot inject SLURM exports or propagate import controls."""
    submitter_path = ROOT / "tools/launcher/common/specdec/submit_qwen4b_ptv23_continuation.sh"
    submitter = submitter_path.read_text()
    artifact = tmp_path / "artifact"
    artifact.write_bytes(b"fixture")
    result = subprocess.run(
        [
            "bash",
            str(submitter_path),
            "--test-only",
            "--account",
            "fixture",
            "--contract",
            str(artifact),
            "--contract-sha256",
            "1" * 64,
            "--contract-tool",
            str(artifact),
            "--contract-tool-sha256",
            "2" * 64,
            "--training-entrypoint",
            str(artifact),
            "--training-entrypoint-sha256",
            "3" * 64,
            "--image",
            str(artifact),
            "--image-sha256",
            "4" * 64,
            "--output",
            "/tmp/output,STAGE=full",
            "--checkpoint-output-root",
            "/tmp/checkpoint",
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode != 0
    assert "unsafe SLURM export value" in result.stderr
    assert 'exports="ALL,' not in submitter


def test_launcher_cli_exposes_contract_creation_and_authenticated_launch() -> None:
    """The CLI makes immutable contracts and consumes them through one launch boundary."""
    result = subprocess.run(
        [sys.executable, str(MODULE_PATH), "--help"],
        check=True,
        capture_output=True,
        text=True,
    )

    assert "{preflight-contract,make-contract,launch}" in result.stdout
    assert "weights-only" in result.stdout
    assert "fresh optimizer and scheduler" in result.stdout
    launch_help = subprocess.run(
        [sys.executable, str(MODULE_PATH), "launch", "--help"],
        check=True,
        capture_output=True,
        text=True,
    )
    assert "--training-entrypoint-sha256" in launch_help.stdout


def test_contract_creation_requires_authenticated_completion_and_historical_lineage() -> None:
    """A contract cannot be created from an arbitrary caller-hashed JSON file."""
    module = _load_module()
    parameters = inspect.signature(module.make_continuation_contract).parameters

    assert "dataset_completion_receipt_path" in parameters
    assert "dataset_completion_receipt_file_sha256" in parameters
    assert "dataset_builder_tool_path" in parameters
    assert "tokenizer_trust_path" in parameters


def test_launcher_trust_pin_matches_the_exact_reviewed_builder() -> None:
    """Semantic replay must execute the frozen builder bytes, not a stale revision."""
    module = _load_module()

    assert (
        hashlib.sha256(BUILDER_PATH.read_bytes()).hexdigest()
        == module.TRUSTED_DATASET_BUILDER_SHA256
    )


def test_dataset_authentication_rejects_one_line_bundle_claiming_700k(tmp_path: Path) -> None:
    """Manifest row-count assertions cannot replace physical 700K semantics."""
    module = _load_module()
    manifest, manifest_sha256, completion, completion_sha256 = _write_dataset_bundle(
        tmp_path / "dataset",
        historical_receipt_file_sha256="1" * 64,
        historical_ordered_prompt_uuids_sha256="2" * 64,
    )

    with pytest.raises(ValueError, match="700K semantic"):
        module._authenticate_dataset_bundle(
            manifest,
            manifest_file_sha256=manifest_sha256,
            completion_receipt_path=completion,
            completion_receipt_file_sha256=completion_sha256,
            historical_receipt_file_sha256="1" * 64,
            historical_ordered_prompt_uuids_sha256="2" * 64,
        )


def test_noop_canary_cannot_mint_passed_receipt(tmp_path: Path, monkeypatch) -> None:
    """Process exit zero alone is not authenticated evidence of global step 20."""
    module = _load_module()
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    (checkpoint / "model.safetensors").write_bytes(b"weights")
    manifest = tmp_path / "MANIFEST.json"
    manifest.write_bytes(b"{}\n")
    completion = tmp_path / "completion.json"
    completion.write_bytes(b"{}\n")
    noop = tmp_path / "noop.py"
    noop.write_text("raise SystemExit(0)\n")
    contract = module.ContinuationContract(
        method="DFlash",
        run_identity="fixture-dflash",
        dataset_identity="ptv2-ptv3-complement-700k-v1",
        dataset_manifest_path=manifest,
        dataset_manifest_file_sha256=hashlib.sha256(manifest.read_bytes()).hexdigest(),
        dataset_completion_receipt_path=completion,
        dataset_completion_receipt_file_sha256=hashlib.sha256(completion.read_bytes()).hexdigest(),
        dataset_builder_tool_path=MODULE_PATH,
        dataset_builder_tool_sha256="0" * 64,
        trajectory_schema_path=MODULE_PATH,
        trajectory_schema_sha256="1" * 64,
        specdec_identity_path=MODULE_PATH,
        specdec_identity_sha256="1" * 64,
        specdec_corpus_contracts_path=MODULE_PATH,
        specdec_corpus_contracts_sha256="1" * 64,
        historical_authenticator_path=MODULE_PATH,
        historical_authenticator_sha256="2" * 64,
        config_path=MODULE_PATH,
        config_file_sha256=hashlib.sha256(MODULE_PATH.read_bytes()).hexdigest(),
        source_inventory_path=MODULE_PATH,
        source_inventory_file_sha256="3" * 64,
        historical_receipt_path=completion,
        historical_receipt_file_sha256="1" * 64,
        historical_ordered_prompt_uuids_sha256="2" * 64,
        held_out_receipts=tuple(
            (name, MODULE_PATH, str(index) * 64)
            for index, name in enumerate(("speed", "math", "code", "swe", "tool"), start=1)
        ),
        tokenizer_trust_path=MODULE_PATH,
        tokenizer_trust_file_sha256="4" * 64,
        verification_scratch_root=tmp_path / "verify",
        canary_execution_stage_root=tmp_path / "canary-stage",
        canary_checkpoint_output_root=tmp_path / "canary-output",
        canary_evidence_path=tmp_path / "canary-meta" / "evidence.json",
        canary_receipt_path=tmp_path / "canary-meta" / "receipt.json",
        full_execution_stage_root=tmp_path / "full-stage",
        full_checkpoint_output_root=tmp_path / "full-output",
        full_evidence_path=tmp_path / "full-meta" / "evidence.json",
        full_receipt_path=tmp_path / "full-meta" / "receipt.json",
        parent_checkpoint_path=checkpoint,
        parent_checkpoint_tree_sha256=_checkpoint_tree_sha256(checkpoint),
        parent_receipt_path=completion,
        parent_receipt_file_sha256="1" * 64,
        training_entrypoint_path=noop,
        training_entrypoint_sha256=hashlib.sha256(noop.read_bytes()).hexdigest(),
        runtime_image_path=tmp_path / "runtime.sqsh",
        runtime_image_sha256="3" * 64,
        full_steps=14_000,
        global_batch_size=50,
    )
    canary_receipt = tmp_path / "canary.json"
    canary_evidence = tmp_path / "training-evidence.json"
    monkeypatch.setattr(module, "load_continuation_contract", lambda *_args, **_kwargs: contract)
    monkeypatch.setattr(
        module,
        "_parse_args",
        lambda _argv=None: Namespace(
            command="launch",
            contract=tmp_path / "contract.json",
            contract_sha256="2" * 64,
            training_entrypoint=noop,
            training_entrypoint_sha256=hashlib.sha256(noop.read_bytes()).hexdigest(),
            runtime_image=tmp_path / "runtime.sqsh",
            runtime_image_sha256="3" * 64,
            checkpoint_output_root=tmp_path / "checkpoints",
            stage="canary",
            max_steps=20,
            resume_mode="weights-only",
            optimizer_state="fresh",
            scheduler_state="fresh",
            canary_receipt=canary_receipt,
            canary_receipt_sha256=None,
            canary_evidence=canary_evidence,
            full_completion_evidence=None,
            full_completion_receipt=None,
        ),
    )

    with pytest.raises(ValueError, match="runtime image attestation"):
        module.main([])
    assert not canary_receipt.exists()


def test_canary_evidence_binds_runtime_parent_and_output_checkpoint(tmp_path: Path) -> None:
    """Canary evidence must bind every immutable runtime and checkpoint root."""
    module = _load_module()
    path = tmp_path / "evidence.json"
    body = {
        "schema_version": "qwen4b-ptv23-training-canary-evidence-v1",
        "contract_file_sha256": "1" * 64,
        "run_identity": "fixture",
        "observed_global_step": 20,
        "canary_status": "passed",
    }
    payload = body | {"evidence_sha256": hashlib.sha256(_canonical(body)).hexdigest()}
    path.write_bytes(_canonical(payload) + b"\n")

    with pytest.raises(ValueError, match="training canary evidence"):
        module._validate_training_canary_evidence(
            path,
            contract_sha256="1" * 64,
            run_identity="fixture",
            training_entrypoint_sha256="2" * 64,
            runtime_image_sha256="3" * 64,
            parent_checkpoint_tree_sha256="4" * 64,
        )


def test_contract_pins_exact_full_exposure_and_final_checkpoint_evidence() -> None:
    """Full training is one exact exposure with durable completion evidence."""
    module = _load_module()
    fields = module.ContinuationContract.__dataclass_fields__
    launch_help = subprocess.run(
        [sys.executable, str(MODULE_PATH), "launch", "--help"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout

    assert {"full_steps", "global_batch_size", "full_example_exposure"} <= set(fields)
    assert "--full-completion-evidence" in launch_help
    assert "--full-completion-receipt" in launch_help


def test_full_schedule_must_exceed_the_fixed_canary() -> None:
    """An authenticated one-pass contract must be runnable after its 20-step canary."""
    module = _load_module()

    module._validate_full_schedule(14_000, 50)
    with pytest.raises(ValueError, match="exceed the 20-step canary"):
        module._validate_full_schedule(20, 35_000)
    with pytest.raises(ValueError, match="exceed the 20-step canary"):
        module._validate_full_schedule(10, 70_000)


def test_runner_authenticates_and_mounts_contract_tool() -> None:
    """The no-home container must receive exactly the launcher bytes it executes."""
    runner = (
        ROOT / "tools/launcher/common/specdec/run_qwen4b_ptv23_continuation.sbatch"
    ).read_text()
    submitter = (
        ROOT / "tools/launcher/common/specdec/submit_qwen4b_ptv23_continuation.sh"
    ).read_text()

    assert "CONTRACT_TOOL_SHA256" in runner
    assert "TRAINING_ENTRYPOINT_SHA256" in runner
    assert 'os.environ["CONTRACT_TOOL"], os.environ["CONTRACT_TOOL_SHA256"]' in runner
    assert 'f"{descriptor_prefix}/{tool_fd}:{tool_target}:ro"' in runner
    assert 'plan["writable_roots"]' in runner
    assert "--contract-tool-sha256" in submitter
    assert "--training-entrypoint-sha256" in submitter


def test_stable_file_uses_one_nofollow_descriptor() -> None:
    """Receipt reads cannot cross a path-swap trust race."""
    source = MODULE_PATH.read_text()
    stable_file = source[
        source.index("def _stable_file") : source.index("def _regular_file_evidence")
    ]

    assert "os.open" in stable_file
    assert "read_bytes" not in stable_file


def test_contract_persists_every_live_semantic_trust_root() -> None:
    """Launch-time replay cannot depend on arguments that disappear after contract creation."""
    module = _load_module()
    fields = set(module.ContinuationContract.__dataclass_fields__)

    assert {
        "dataset_builder_tool_path",
        "dataset_builder_tool_sha256",
        "trajectory_schema_path",
        "trajectory_schema_sha256",
        "config_path",
        "config_file_sha256",
        "source_inventory_path",
        "source_inventory_file_sha256",
        "historical_receipt_path",
        "held_out_receipts",
        "tokenizer_trust_path",
        "tokenizer_trust_file_sha256",
        "parent_receipt_path",
    } <= fields


def test_contract_loader_replays_parent_and_dataset_semantics() -> None:
    """A caller-authored contract must not bypass the authenticated creation boundary."""
    source = MODULE_PATH.read_text()
    loader = source[
        source.index("def load_continuation_contract") : source.index("def _parse_args")
    ]

    assert "load_authenticated_parent_checkpoint" in loader
    assert "_run_builder_semantic_verifier" in loader


def test_semantic_verifier_pins_dependency_and_tokenizer_roots() -> None:
    """The reviewed builder is not immutable if its imports or tokenizer are caller-selected."""
    module = _load_module()
    parameters = inspect.signature(module._run_builder_semantic_verifier).parameters

    assert "trajectory_schema_path" in parameters
    assert module.TRUSTED_TRAJECTORY_SCHEMA_SHA256
    assert hasattr(module, "TRUSTED_TOKENIZER_TRUST_FILE_SHA256")
    assert "manifest tokenizer trust mismatch" in MODULE_PATH.read_text()


def test_canary_receipt_replays_its_live_training_evidence() -> None:
    """A receipt hash alone cannot replace the evidence bytes that proved step 20."""
    source = MODULE_PATH.read_text()
    validator = source[
        source.index("def _validate_canary_receipt") : source.index(
            "def _validate_training_canary_evidence"
        )
    ]

    assert "training_evidence_path" in validator
    assert "_validate_training_canary_evidence" in validator


def test_canary_receipt_requires_immutable_approval_and_contract_bound_paths(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Caller-self-hashed canary JSON cannot authorize full training."""
    module = _load_module()
    contract_sha256 = "1" * 64
    checkpoint = tmp_path / "attacker-checkpoint"
    checkpoint.mkdir()
    (checkpoint / "weights.bin").write_bytes(b"caller-authored")
    checkpoint_sha256 = _checkpoint_tree_sha256(checkpoint)
    evidence_path = tmp_path / "attacker-evidence.json"
    evidence_body = {
        "schema_version": "qwen4b-ptv23-training-canary-evidence-v1",
        "contract_file_sha256": contract_sha256,
        "run_identity": "fixture",
        "observed_global_step": 20,
        "canary_status": "passed",
        "training_entrypoint_sha256": "2" * 64,
        "runtime_image_sha256": "3" * 64,
        "parent_checkpoint_tree_sha256": "4" * 64,
        "output_checkpoint_path": str(checkpoint),
        "output_checkpoint_tree_sha256": checkpoint_sha256,
    }
    evidence = evidence_body | {
        "evidence_sha256": hashlib.sha256(_canonical(evidence_body)).hexdigest()
    }
    evidence_raw = _canonical(evidence) + b"\n"
    evidence_path.write_bytes(evidence_raw)
    receipt_path = tmp_path / "attacker-receipt.json"
    receipt = {
        "schema_version": "qwen4b-ptv23-continuation-canary-v1",
        "contract_file_sha256": contract_sha256,
        "completed_steps": 20,
        "status": "passed",
        "training_evidence_path": str(evidence_path),
        "training_evidence_file_sha256": hashlib.sha256(evidence_raw).hexdigest(),
        "training_evidence_sha256": evidence["evidence_sha256"],
        "training_entrypoint_sha256": "2" * 64,
        "runtime_image_sha256": "3" * 64,
        "parent_checkpoint_tree_sha256": "4" * 64,
        "output_checkpoint_path": str(checkpoint),
        "output_checkpoint_tree_sha256": checkpoint_sha256,
    }
    receipt_raw = _canonical(receipt) + b"\n"
    receipt_path.write_bytes(receipt_raw)
    receipt_sha256 = hashlib.sha256(receipt_raw).hexdigest()
    contract = Namespace(
        run_identity="fixture",
        training_entrypoint_sha256="2" * 64,
        runtime_image_sha256="3" * 64,
        parent_checkpoint_tree_sha256="4" * 64,
        canary_evidence_path=tmp_path / "approved" / "evidence.json",
        canary_receipt_path=tmp_path / "approved" / "receipt.json",
        canary_checkpoint_output_root=tmp_path / "approved" / "checkpoints",
    )

    with pytest.raises(ValueError, match="immutable-approved"):
        module._validate_canary_receipt(receipt_path, receipt_sha256, contract_sha256, contract)

    monkeypatch.setattr(module, "TRUSTED_CANARY_RECEIPT_SHA256S", frozenset({receipt_sha256}))
    with pytest.raises(ValueError, match="contract-bound paths"):
        module._validate_canary_receipt(receipt_path, receipt_sha256, contract_sha256, contract)


def test_trainer_receives_exact_one_pass_exposure_controls() -> None:
    """The trainer, not just the contract arithmetic, must enforce one exact 700K pass."""
    source = MODULE_PATH.read_text()
    launch = source[source.index("    command = [") : source.index("    subprocess.run(command")]

    assert '"--global-batch-size"' in launch
    assert '"--max-examples"' in launch
    assert '"--dataset-repeat"' in launch
    assert '"1"' in launch


def test_trainer_uses_isolated_python_and_a_reviewed_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Caller import controls cannot execute unpinned trainer dependencies."""
    module = _load_module()
    monkeypatch.setenv("PYTHONPATH", "/attacker")
    monkeypatch.setenv("PYTHONHOME", "/attacker")
    monkeypatch.setenv("LD_PRELOAD", "/attacker.so")
    monkeypatch.setenv("PATH", "/attacker/bin")
    monkeypatch.setenv("SLURM_JOB_ID", "123")

    environment = module._trainer_environment()
    assert "PYTHONPATH" not in environment
    assert "PYTHONHOME" not in environment
    assert "LD_PRELOAD" not in environment
    assert environment["PATH"] == "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
    assert environment["SLURM_JOB_ID"] == "123"

    source = MODULE_PATH.read_text()
    command_start = source.index("    command = [")
    launch = source[command_start : source.index("        subprocess.run(command", command_start)]
    assert 'sys.executable,\n        "-I"' in launch
    assert "env=_trainer_environment()" in source


def test_runner_mounts_every_contract_input_and_checkpoint_root() -> None:
    """No-home execution must not rely on cluster-global implicit mounts."""
    runner = (
        ROOT / "tools/launcher/common/specdec/run_qwen4b_ptv23_continuation.sbatch"
    ).read_text()

    assert 'plan["readonly_roots"]' in runner
    assert 'plan["writable_roots"]' in runner
    assert 'plan["checkpoint_output_root"]' in runner


def test_transitive_verifier_dependency_closure_is_contract_bound() -> None:
    """Every executable local dependency must have an immutable contract root."""
    module = _load_module()
    fields = set(module.ContinuationContract.__dataclass_fields__)
    parameters = inspect.signature(module._run_builder_semantic_verifier).parameters

    assert {"specdec_identity_path", "specdec_corpus_contracts_path"} <= fields
    assert {"specdec_identity_path", "specdec_corpus_contracts_path"} <= set(parameters)
    assert module.TRUSTED_SPECDEC_IDENTITY_SHA256
    assert module.TRUSTED_SPECDEC_CORPUS_CONTRACTS_SHA256


def test_runner_rejects_a_caller_self_attested_contract_tool() -> None:
    """The runner must compare its tool against a reviewed immutable digest."""
    runner_path = ROOT / "tools/launcher/common/specdec/run_qwen4b_ptv23_continuation.sbatch"
    runner = runner_path.read_text()

    assert "IMMUTABLE_CONTRACT_TOOL_SHA256=" in runner
    assert '[[ "$CONTRACT_TOOL_SHA256" == "$IMMUTABLE_CONTRACT_TOOL_SHA256" ]]' in runner
    result = subprocess.run(
        ["bash", str(runner_path)],
        env={
            "PATH": os.environ["PATH"],
            "STAGE": "canary",
            "CONTRACT": "/nonexistent/contract",
            "CONTRACT_SHA256": "1" * 64,
            "CONTRACT_TOOL": "/nonexistent/tool",
            "CONTRACT_TOOL_SHA256": "2" * 64,
        },
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode != 0


def test_writable_execution_roots_must_be_fresh_and_disjoint(tmp_path: Path) -> None:
    """Writable mounts cannot overlap an authenticated input tree."""
    module = _load_module()
    parent = tmp_path / "parent"
    parent.mkdir()
    output = parent / "output"

    with pytest.raises(ValueError, match="disjoint"):
        module._validate_fresh_disjoint_roots(
            protected=(parent,),
            writable=(output,),
        )


def test_direct_launch_is_fail_closed_without_runtime_attestation() -> None:
    """Direct Python invocation cannot self-claim the active container image."""
    module = _load_module()

    assert module.RUNTIME_ATTESTATION_IMPLEMENTED is False


def test_execution_staging_copies_only_authenticated_regular_files(tmp_path: Path) -> None:
    """Execution consumes a stable copy rather than a re-opened authenticated path."""
    module = _load_module()
    source = tmp_path / "source"
    source.mkdir()
    (source / "model.safetensors").write_bytes(b"weights")
    destination = tmp_path / "stage"

    digest = module._checkpoint_tree_sha256(source)
    module._stage_authenticated_tree(source, destination, expected_tree_sha256=digest)

    assert module._checkpoint_tree_sha256(destination) == digest
    assert (destination / "model.safetensors").read_bytes() == b"weights"


def test_stable_readers_reject_non_regular_files() -> None:
    """No-follow readers must also reject non-regular descriptors."""
    source = MODULE_PATH.read_text()
    stable = source[source.index("def _stable_file") : source.index("def _checkpoint_tree_sha256")]

    assert "stat.S_ISREG" in stable


def test_runner_uses_isolated_staged_tool_and_image_after_preflight() -> None:
    """Verified pathnames cannot be swapped before Python or pyxis reopens them."""
    runner = (
        ROOT / "tools/launcher/common/specdec/run_qwen4b_ptv23_continuation.sbatch"
    ).read_text()

    assert "contract_fd = copy_to_anonymous_file" in runner
    assert "tool_fd = copy_to_anonymous_file" in runner
    assert "image_fd = copy_to_anonymous_file" in runner
    assert 'descriptor_prefix = f"/proc/{os.getpid()}/fd"' in runner
    assert 'f"--container-image={descriptor_prefix}/{image_fd}"' in runner
    assert '"-I",\n    tool_target,\n    *launch' in runner


@pytest.mark.skipif(not Path("/proc/self/fd").is_dir(), reason="Linux procfs required")
def test_live_keeper_descriptor_is_visible_across_a_process_boundary(tmp_path: Path) -> None:
    """A separate daemon-like process can reopen a live keeper's descriptor."""
    payload = b"pyxis-visible-authenticated-bytes"
    source = tmp_path / "artifact"
    source.write_bytes(payload)
    descriptor = os.open(source, os.O_RDONLY)
    try:
        keeper_path = f"/proc/{os.getpid()}/fd/{descriptor}"
        observed = subprocess.run(
            [
                sys.executable,
                "-I",
                "-c",
                "import pathlib,sys;sys.stdout.buffer.write(pathlib.Path(sys.argv[1]).read_bytes())",
                keeper_path,
            ],
            check=True,
            capture_output=True,
        ).stdout
    finally:
        os.close(descriptor)

    assert observed == payload


def test_runner_copy_handles_short_writes_and_reauthenticates_staged_fd(
    tmp_path: Path, monkeypatch
) -> None:
    """The descriptor consumed by Pyxis is independently complete and authenticated."""
    runner = RUNNER_PATH.read_text()

    assert "def _write_all(" in runner
    assert "staged_digest = hashlib.sha256()" in runner
    assert "staged source copy mismatch" in runner

    tmpfile = getattr(os, "O_TMPFILE", None)
    if tmpfile is None:
        pytest.skip("O_TMPFILE is unavailable")
    try:
        probe = os.open(tmp_path, os.O_RDWR | tmpfile, 0o600)
    except OSError:
        pytest.skip("test filesystem does not support O_TMPFILE")
    else:
        os.close(probe)

    source = tmp_path / "source"
    payload = b"authenticated-stage" * 100
    source.write_bytes(payload)
    copy = _load_runner_copy_function(tmp_path)
    real_write = os.write

    def short_write(descriptor: int, data: bytes | memoryview) -> int:
        return real_write(descriptor, data[: max(1, len(data) // 3)])

    monkeypatch.setattr(os, "write", short_write)
    descriptor = copy(source, hashlib.sha256(payload).hexdigest())
    try:
        assert os.read(descriptor, len(payload) + 1) == payload
    finally:
        os.close(descriptor)


def test_runner_fails_closed_when_anonymous_staging_is_unsupported(
    tmp_path: Path, monkeypatch
) -> None:
    """Unsupported node-local filesystems fail before any untrusted execution."""
    source = tmp_path / "source"
    source.write_bytes(b"payload")
    copy = _load_runner_copy_function(tmp_path)
    monkeypatch.delattr(os, "O_TMPFILE", raising=False)

    with pytest.raises(SystemExit, match="O_TMPFILE"):
        copy(source, hashlib.sha256(source.read_bytes()).hexdigest())


def test_runner_probes_live_keeper_visibility_before_srun(tmp_path: Path, monkeypatch) -> None:
    """A real child-process probe fails closed when keeper descriptors are invisible."""
    runner = RUNNER_PATH.read_text()
    assert "def verify_pyxis_visible_descriptors(" in runner
    assert "verify_pyxis_visible_descriptors(" in runner.split("image_fd =", 1)[1]

    source = tmp_path / "source"
    source.write_bytes(b"probe")
    descriptor = os.open(source, os.O_RDONLY)
    try:
        namespace = _load_runner_functions(tmp_path)
        probe = cast(
            "Callable[[tuple[int, ...]], None]",
            namespace["verify_pyxis_visible_descriptors"],
        )
        if Path("/proc/self/fd").is_dir():
            probe((descriptor,))
        else:
            with pytest.raises(SystemExit, match="procfs"):
                probe((descriptor,))
    finally:
        os.close(descriptor)

    monkeypatch.setenv("SLURM_NNODES", "2")
    with pytest.raises(SystemExit, match="multi-node"):
        probe((descriptor,))


def test_runner_rejects_pyxis_mount_grammar_injection(tmp_path: Path) -> None:
    """Contract paths cannot add comma/colon-delimited mounts or controls."""
    runner = RUNNER_PATH.read_text()
    assert "def validate_mount_target(" in runner
    validator = cast(
        "Callable[[str], None]",
        _load_runner_functions(tmp_path)["validate_mount_target"],
    )

    validator("/safe/target")
    for malicious in ("/safe,/:/injected:rw", "/safe:rw", "/safe\n/injected"):
        with pytest.raises(SystemExit, match="mount target"):
            validator(malicious)


def test_semantic_replay_requires_isolated_import_resolution() -> None:
    """A launcher-directory sibling cannot shadow transformers or pyarrow."""
    module = _load_module()

    with pytest.raises(ValueError, match="isolated Python"):
        module._run_builder_semantic_verifier(
            builder_tool_path=MODULE_PATH,
            trajectory_schema_path=MODULE_PATH,
            specdec_identity_path=MODULE_PATH,
            specdec_corpus_contracts_path=MODULE_PATH,
            historical_authenticator_path=MODULE_PATH,
            config_path=MODULE_PATH,
            source_inventory_path=MODULE_PATH,
            source_inventory_file_sha256="1" * 64,
            historical_receipt_path=MODULE_PATH,
            historical_receipt_file_sha256="2" * 64,
            held_out_receipts=(),
            tokenizer_trust_path=MODULE_PATH,
            tokenizer_trust_file_sha256="3" * 64,
            verification_scratch_root=MODULE_PATH.parent,
            manifest_path=MODULE_PATH,
            manifest_file_sha256="4" * 64,
        )


def test_all_semantic_input_roots_are_protected_from_writable_mounts(tmp_path: Path) -> None:
    """Mount validation must expand inventory, tokenizer, held-out, and tool paths."""
    module = _load_module()
    physical_source = tmp_path / "physical.jsonl"
    snapshot = tmp_path / "tokenizer-snapshot"
    held_out = tmp_path / "held-out.json"
    inventory = tmp_path / "inventory.json"
    tokenizer = tmp_path / "tokenizer.json"
    inventory.write_bytes(
        _canonical({"sources": [{"files": [{"path": str(physical_source)}]}]}) + b"\n"
    )
    tokenizer.write_bytes(_canonical({"snapshot_path": str(snapshot)}) + b"\n")

    protected = module._semantic_protected_paths(
        source_inventory_path=inventory,
        source_inventory_file_sha256=hashlib.sha256(inventory.read_bytes()).hexdigest(),
        tokenizer_trust_path=tokenizer,
        tokenizer_trust_file_sha256=hashlib.sha256(tokenizer.read_bytes()).hexdigest(),
        held_out_receipts=(("speed", held_out, "1" * 64),),
        dataset_builder_tool_path=MODULE_PATH,
        trajectory_schema_path=MODULE_PATH,
        specdec_identity_path=MODULE_PATH,
        specdec_corpus_contracts_path=MODULE_PATH,
        historical_authenticator_path=MODULE_PATH,
    )

    assert {physical_source, snapshot, held_out, MODULE_PATH} <= set(protected)


def test_known_runtime_attestation_blocker_precedes_contract_replay(monkeypatch) -> None:
    """A known terminal blocker must fail before the 700K semantic replay."""
    module = _load_module()
    monkeypatch.setattr(module, "_parse_args", lambda _argv=None: Namespace(command="launch"))
    monkeypatch.setattr(
        module,
        "load_continuation_contract",
        lambda *_args, **_kwargs: pytest.fail("contract replay ran before known blocker"),
    )

    with pytest.raises(ValueError, match="runtime image attestation"):
        module.main([])


def test_runner_never_parses_unvalidated_canary_receipt_for_mounts() -> None:
    """Full-stage mounts come only from contract-bound canary roots."""
    runner = (
        ROOT / "tools/launcher/common/specdec/run_qwen4b_ptv23_continuation.sbatch"
    ).read_text()

    assert "json.load(open(sys.argv[3]" not in runner
    assert 'plan["readonly_roots"]' in runner
    assert 'plan["checkpoint_output_root"]' in runner
    assert 'plan["receipt_path"]' in runner


def test_runner_anchors_mounts_and_artifacts_to_open_descriptors() -> None:
    """Ancestor swaps cannot redirect staged artifacts or writable mounts."""
    runner = (
        ROOT / "tools/launcher/common/specdec/run_qwen4b_ptv23_continuation.sbatch"
    ).read_text()

    assert 'getattr(os, "O_TMPFILE", None)' in runner
    assert "O_NOFOLLOW" in runner
    assert "dir_fd=" in runner
    assert 'descriptor_prefix = f"/proc/{os.getpid()}/fd"' in runner
    assert "subprocess.run(command, check=True, env=os.environ)" in runner
    assert "os.execvpe" not in runner
    assert "/proc/self/fd" not in runner
    assert runner.index('"preflight-contract"') < runner.index(
        'copy_to_anonymous_file(os.environ["IMAGE"]'
    )
