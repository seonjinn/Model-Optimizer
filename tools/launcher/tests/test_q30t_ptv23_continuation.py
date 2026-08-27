# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Behavioral tests for the Q30 Thinking 700K continuation identity."""

from __future__ import annotations

from pathlib import Path
from typing import cast

import pytest
from common.specdec.q30t_ptv23_continuation import (
    ExposureSchedule,
    Method,
    Q30TContinuationContract,
    build_q30_training_command,
)


def _contract(*, method: Method = "DFlash") -> Q30TContinuationContract:
    return Q30TContinuationContract(
        target_model="Qwen/Qwen3-30B-A3B-Thinking-2507",
        method=method,
        block_size=8,
        dataset_path=Path("/lustre/q30t-700k/DATA.jsonl"),
        dataset_sha256="a" * 64,
        ordered_prompt_uuids_sha256="b" * 64,
        dataset_records=700_000,
        tokenizer_receipt_path=Path("/lustre/q30t-700k/TOKENIZER.json"),
        tokenizer_receipt_file_sha256="c" * 64,
        parent_checkpoint_path=Path("/lustre/q30t-parent/step-025391"),
        parent_receipt_path=Path("/lustre/q30t-parent/PARENT.json"),
        parent_receipt_file_sha256="d" * 64,
        source_commit="e" * 40,
        runtime_archive_sha256="f" * 64,
        runtime_image_sha256="1" * 64,
        training_entrypoint=Path("/run/modelopt/train_eagle_streaming.sh"),
        training_entrypoint_sha256="2" * 64,
        output_root=Path("/lustre/q30t-700k-runs/dflash-canary"),
    )


def _values(command: tuple[str, ...], flag: str) -> list[str]:
    return [command[index + 1] for index, item in enumerate(command[:-1]) if item == flag]


def test_q30_full_schedule_consumes_exact_700k_with_partial_final_batch() -> None:
    """The last partial batch must close the exact 700K exposure."""
    schedule = ExposureSchedule()

    assert schedule.full_steps == 1_368
    assert schedule.nominal_global_batch_size == 512
    assert schedule.final_global_batch_size == 96
    assert schedule.consumed_examples == 700_000
    assert schedule.examples_for_step(1_367) == 512
    assert schedule.examples_for_step(1_368) == 96
    schedule.validate(trainer_ranks=32)


def test_q30_schedule_rejects_inexact_or_rank_indivisible_exposure() -> None:
    """Invalid total exposure and rank partitioning must fail closed."""
    with pytest.raises(ValueError, match="700K"):
        ExposureSchedule(final_global_batch_size=95).validate(trainer_ranks=32)
    with pytest.raises(ValueError, match="trainer ranks"):
        ExposureSchedule(final_global_batch_size=96).validate(trainer_ranks=64)


@pytest.mark.parametrize("method", ["DFlash", "DSpark"])
def test_q30_canary_command_uses_proven_topology_and_fresh_training_state(
    method: Method,
) -> None:
    """Canaries must preserve proven topology without parent optimizer state."""
    command = build_q30_training_command(_contract(method=method), stage="canary")

    assert _values(command, "--target-model") == ["Qwen/Qwen3-30B-A3B-Thinking-2507"]
    assert _values(command, "--drafter-parent") == ["/lustre/q30t-parent/step-025391"]
    assert _values(command, "--method") == [method]
    assert _values(command, "--block-size") == ["8"]
    assert _values(command, "--nodes") == ["16"]
    assert _values(command, "--serving-nodes") == ["8"]
    assert _values(command, "--trainer-nodes") == ["8"]
    assert _values(command, "--trainer-ranks") == ["32"]
    assert _values(command, "--target-tensor-parallel-size") == ["2"]
    assert _values(command, "--max-steps") == ["20"]
    assert _values(command, "--global-batch-size") == ["512"]
    assert _values(command, "--restore") == ["weights", "modelopt-state"]
    assert _values(command, "--optimizer-state") == ["fresh"]
    assert _values(command, "--scheduler-state") == ["fresh"]
    assert _values(command, "--dataset-records") == ["700000"]
    assert "--no-cycle-dataset" in command
    assert "optimizer" not in _values(command, "--restore")
    assert "scheduler" not in _values(command, "--restore")
    assert "rng" not in _values(command, "--restore")


def test_q30_full_command_persists_exact_partial_final_batch() -> None:
    """Full training must carry explicit final-batch and exposure controls."""
    command = build_q30_training_command(_contract(), stage="full")

    assert _values(command, "--max-steps") == ["1368"]
    assert _values(command, "--global-batch-size") == ["512"]
    assert _values(command, "--final-global-batch-size") == ["96"]
    assert _values(command, "--consumed-examples") == ["700000"]
    assert "--no-cycle-dataset" in command


def test_q30_contract_rejects_wrong_target_sample_count_or_method() -> None:
    """Scientific identity substitutions must be rejected at construction."""
    with pytest.raises(ValueError, match="Thinking target"):
        _contract().replace(target_model="Qwen/Qwen3-30B-A3B")
    with pytest.raises(ValueError, match="700K"):
        _contract().replace(dataset_records=699_999)
    with pytest.raises(ValueError, match="method"):
        _contract(method=cast("Method", "DFlash2"))
