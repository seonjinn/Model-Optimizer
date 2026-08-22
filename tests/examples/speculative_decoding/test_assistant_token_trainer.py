from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")
pytest.importorskip("datasets")
pytest.importorskip("accelerate")

REPO_ROOT = Path(__file__).resolve().parents[3]
EXAMPLE_ROOT = REPO_ROOT / "examples/speculative_decoding"
sys.path.insert(0, str(EXAMPLE_ROOT))
try:
    from eagle_utils import EagleTrainerWithAccLog, _AssistantTokenBudgetCallback
finally:
    sys.path.pop(0)

from modelopt.torch.speculative.assistant_token_budget import AssistantTokenBudgetController


def test_trainer_trims_final_local_mask_and_callback_stops_at_exact_target() -> None:
    trainer = object.__new__(EagleTrainerWithAccLog)
    trainer.assistant_token_budget = AssistantTokenBudgetController(
        target=3, training_fingerprint="fingerprint"
    )
    trainer.model = SimpleNamespace(training=True)
    inputs = {"loss_mask": torch.tensor([[1, 1, 0, 1, 1]], dtype=torch.long)}

    trimmed = trainer._apply_assistant_token_budget(inputs)

    assert trimmed["loss_mask"].tolist() == [[1, 1, 0, 1, 0]]
    callback = _AssistantTokenBudgetCallback(trainer.assistant_token_budget)
    control = SimpleNamespace(should_save=False, should_training_stop=False)
    callback.on_step_end(None, SimpleNamespace(global_step=1), control)
    assert trainer.assistant_token_budget.committed == 3
    assert control.should_save is True
    assert control.should_training_stop is True
