# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Qwen3-30B-A3B Thinking experimental 700K continuation contract."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Literal, Self

if TYPE_CHECKING:
    from pathlib import Path

Method = Literal["DFlash", "DSpark"]
Stage = Literal["canary", "full"]
TARGET_MODEL = "Qwen/Qwen3-30B-A3B-Thinking-2507"


def _is_lower_hex(value: object, length: int) -> bool:
    return (
        type(value) is str
        and len(value) == length
        and all(character in "0123456789abcdef" for character in value)
    )


@dataclass(frozen=True)
class ExposureSchedule:
    """Exact one-pass schedule with a rank-divisible partial final batch."""

    consumed_examples: int = 700_000
    full_steps: int = 1_368
    nominal_global_batch_size: int = 512
    final_global_batch_size: int = 96
    canary_steps: int = 20

    def validate(self, *, trainer_ranks: int) -> None:
        """Reject schedules that do not consume exactly one 700K pass."""
        if type(trainer_ranks) is not int or trainer_ranks < 1:
            raise ValueError("trainer ranks must be a positive integer")
        if any(
            type(value) is not int
            for value in (
                self.consumed_examples,
                self.full_steps,
                self.nominal_global_batch_size,
                self.final_global_batch_size,
                self.canary_steps,
            )
        ):
            raise ValueError("exposure schedule values must be exact integers")
        if self.full_steps <= self.canary_steps or self.canary_steps != 20:
            raise ValueError("full schedule must exceed the 20-step canary")
        observed = (
            self.full_steps - 1
        ) * self.nominal_global_batch_size + self.final_global_batch_size
        if self.consumed_examples != 700_000 or observed != 700_000:
            raise ValueError("exposure schedule must consume exact 700K")
        if self.nominal_global_batch_size % trainer_ranks:
            raise ValueError("nominal batch must divide across trainer ranks")
        if self.final_global_batch_size % trainer_ranks:
            raise ValueError("final batch must divide across trainer ranks")

    def examples_for_step(self, step: int) -> int:
        """Return the exact example exposure for one optimizer step."""
        if type(step) is not int or not 1 <= step <= self.full_steps:
            raise ValueError("step is outside the exposure schedule")
        return (
            self.final_global_batch_size
            if step == self.full_steps
            else self.nominal_global_batch_size
        )


@dataclass(frozen=True)
class Q30TContinuationContract:
    """Minimal reproducible identity for the trusted-cluster canary lane."""

    target_model: str
    method: Method
    block_size: int
    dataset_path: Path
    dataset_sha256: str
    ordered_prompt_uuids_sha256: str
    dataset_records: int
    tokenizer_receipt_path: Path
    tokenizer_receipt_file_sha256: str
    parent_checkpoint_path: Path
    parent_receipt_path: Path
    parent_receipt_file_sha256: str
    source_commit: str
    runtime_archive_sha256: str
    runtime_image_sha256: str
    training_entrypoint: Path
    training_entrypoint_sha256: str
    output_root: Path
    schedule: ExposureSchedule = ExposureSchedule()
    nodes: int = 16
    serving_nodes: int = 8
    trainer_nodes: int = 8
    gpus_per_node: int = 4
    trainer_ranks: int = 32
    target_tensor_parallel_size: int = 2

    def __post_init__(self) -> None:
        if self.target_model != TARGET_MODEL:
            raise ValueError("contract must use the Q30 Thinking target")
        if self.method not in ("DFlash", "DSpark"):
            raise ValueError("contract method must be DFlash or DSpark")
        if type(self.block_size) is not int or self.block_size != 8:
            raise ValueError("contract block size must be 8")
        if type(self.dataset_records) is not int or self.dataset_records != 700_000:
            raise ValueError("contract dataset must contain exact 700K records")
        if (
            self.nodes,
            self.serving_nodes,
            self.trainer_nodes,
            self.gpus_per_node,
            self.trainer_ranks,
            self.target_tensor_parallel_size,
        ) != (16, 8, 8, 4, 32, 2):
            raise ValueError("contract must preserve the proven 16-node topology")
        if any(
            not path.is_absolute()
            for path in (
                self.dataset_path,
                self.tokenizer_receipt_path,
                self.parent_checkpoint_path,
                self.parent_receipt_path,
                self.training_entrypoint,
                self.output_root,
            )
        ):
            raise ValueError("contract paths must be absolute")
        for digest in (
            self.dataset_sha256,
            self.ordered_prompt_uuids_sha256,
            self.tokenizer_receipt_file_sha256,
            self.parent_receipt_file_sha256,
            self.runtime_archive_sha256,
            self.runtime_image_sha256,
            self.training_entrypoint_sha256,
        ):
            if not _is_lower_hex(digest, 64):
                raise ValueError("contract SHA-256 values must be lowercase hexadecimal")
        if not _is_lower_hex(self.source_commit, 40):
            raise ValueError("source commit must be a lowercase 40-character Git SHA")
        self.schedule.validate(trainer_ranks=self.trainer_ranks)

    def replace(self, **changes: object) -> Self:
        """Return a validated contract with selected fields replaced."""
        return replace(self, **changes)


def _flag(name: str, value: object) -> tuple[str, str]:
    return name, str(value)


def build_q30_training_command(
    contract: Q30TContinuationContract, *, stage: Stage
) -> tuple[str, ...]:
    """Build the isolated trainer command without parent optimizer-state restore."""
    contract.__post_init__()
    if stage not in ("canary", "full"):
        raise ValueError("stage must be canary or full")
    schedule = contract.schedule
    max_steps = schedule.canary_steps if stage == "canary" else schedule.full_steps
    command = (
        str(contract.training_entrypoint),
        *_flag("--target-model", contract.target_model),
        *_flag("--drafter-parent", contract.parent_checkpoint_path),
        *_flag("--parent-receipt", contract.parent_receipt_path),
        *_flag("--parent-receipt-sha256", contract.parent_receipt_file_sha256),
        *_flag("--method", contract.method),
        *_flag("--block-size", contract.block_size),
        *_flag("--dataset", contract.dataset_path),
        *_flag("--dataset-sha256", contract.dataset_sha256),
        *_flag("--ordered-prompt-uuids-sha256", contract.ordered_prompt_uuids_sha256),
        *_flag("--dataset-records", contract.dataset_records),
        *_flag("--tokenizer-receipt", contract.tokenizer_receipt_path),
        *_flag("--tokenizer-receipt-sha256", contract.tokenizer_receipt_file_sha256),
        *_flag("--source-commit", contract.source_commit),
        *_flag("--runtime-archive-sha256", contract.runtime_archive_sha256),
        *_flag("--runtime-image-sha256", contract.runtime_image_sha256),
        *_flag("--output-root", contract.output_root),
        *_flag("--nodes", contract.nodes),
        *_flag("--serving-nodes", contract.serving_nodes),
        *_flag("--trainer-nodes", contract.trainer_nodes),
        *_flag("--gpus-per-node", contract.gpus_per_node),
        *_flag("--trainer-ranks", contract.trainer_ranks),
        *_flag("--target-tensor-parallel-size", contract.target_tensor_parallel_size),
        *_flag("--max-steps", max_steps),
        *_flag("--global-batch-size", schedule.nominal_global_batch_size),
        "--restore",
        "weights",
        "--restore",
        "modelopt-state",
        *_flag("--optimizer-state", "fresh"),
        *_flag("--scheduler-state", "fresh"),
        "--no-cycle-dataset",
    )
    if stage == "full":
        command += (
            *_flag("--final-global-batch-size", schedule.final_global_batch_size),
            *_flag("--consumed-examples", schedule.consumed_examples),
        )
    return command
