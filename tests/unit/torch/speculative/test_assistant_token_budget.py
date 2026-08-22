from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

MODULE_PATH = (
    Path(__file__).resolve().parents[4] / "modelopt/torch/speculative/assistant_token_budget.py"
)


def _load_module():
    spec = importlib.util.spec_from_file_location("assistant_token_budget", MODULE_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_rank_order_allocation_trims_only_the_global_tail() -> None:
    module = _load_module()

    assert module.local_token_allowance([3, 4, 2], rank=0, remaining=5) == 3
    assert module.local_token_allowance([3, 4, 2], rank=1, remaining=5) == 2
    assert module.local_token_allowance([3, 4, 2], rank=2, remaining=5) == 0
    assert module.trim_binary_mask([0, 1, 1, 0, 1], 2) == [0, 1, 1, 0, 0]


def test_counter_commits_only_complete_optimizer_steps() -> None:
    module = _load_module()
    controller = module.AssistantTokenBudgetController(target=10, training_fingerprint="abc")

    controller.record_microbatch(4)
    assert controller.remaining == 6
    with pytest.raises(ValueError, match="uncommitted"):
        controller.state_dict()

    controller.commit_step(1)
    assert controller.state_dict()["committed_assistant_tokens"] == 4
    controller.record_microbatch(6)
    controller.commit_step(2)
    assert controller.reached_target


def test_resume_counter_is_bound_to_fingerprint_step_and_extended_target(tmp_path: Path) -> None:
    module = _load_module()
    checkpoint = tmp_path / "assistant-token-state.json"
    checkpoint.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "training_fingerprint": "abc",
                "committed_assistant_tokens": 64,
                "global_step": 7,
            }
        )
        + "\n"
    )
    resumed = module.AssistantTokenBudgetController(target=128, training_fingerprint="abc")

    resumed.load_checkpoint(checkpoint, expected_global_step=7)

    assert resumed.committed == 64
    assert resumed.remaining == 64
    with pytest.raises(ValueError, match="identity mismatch"):
        resumed.load_checkpoint(checkpoint, expected_global_step=8)


def test_eagle_trainer_integrates_distributed_trim_stop_and_checkpoint_state() -> None:
    source = (
        Path(__file__).resolve().parents[4] / "examples/speculative_decoding/eagle_utils.py"
    ).read_text()

    assert "torch.distributed.all_gather" in source
    assert "local_token_allowance" in source
    assert 'inputs["loss_mask"] = loss_mask' in source
    assert "control.should_training_stop = True" in source
    assert 'checkpoint / "assistant-token-state.json"' in source
    assert "controller.load_checkpoint" in source
