# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Fresh-schedule initialization contracts for converted speculative weights."""

from __future__ import annotations

import ast
import importlib.util
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from types import ModuleType

_REPO_ROOT = Path(__file__).parents[3]


def _load(name: str, relative_path: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, _REPO_ROOT / relative_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {relative_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_initialization = _load(
    "hf_weight_only_initialization_under_test",
    "modelopt/torch/speculative/plugins/hf_weight_only_initialization.py",
)
initialization_action = _initialization.initialization_action
requested_dflash_method = _initialization.requested_dflash_method
validate_converted_modelopt_state = _initialization.validate_converted_modelopt_state


def _state(*, block_size: int = 8, projector_type: str | None = None) -> dict[str, object]:
    architecture: dict[str, object] = {
        "num_hidden_layers": 5,
        "num_attention_heads": 32,
        "num_key_value_heads": 4,
        "head_dim": 128,
        "intermediate_size": 6144,
    }
    if projector_type is not None:
        architecture["projector_type"] = projector_type
    return {
        "modelopt_state_dict": [
            (
                "dflash",
                {
                    "config": {
                        "dflash_block_size": block_size,
                        "dflash_architecture_config": architecture,
                    }
                },
            )
        ]
    }


@pytest.mark.parametrize(
    ("projector_type", "expected"),
    [(None, "dflash"), ("dflash", "dflash"), ("dspark", "dspark")],
)
def test_requested_method_is_derived_from_the_converted_parent(
    projector_type: str | None, expected: str
) -> None:
    """Only the supported DFlash-family projector identities are accepted."""
    assert requested_dflash_method(projector_type) == expected


def test_weight_only_initialization_forbids_trainer_resume_state() -> None:
    """The parent supplies converted model bytes, never optimizer/scheduler/RNG state."""
    assert (
        initialization_action(
            "converted-weights-only",
            checkpoint_is_hf=False,
            explicit_resume_checkpoint=None,
        )
        == "validate-weight-only"
    )
    with pytest.raises(ValueError, match="explicit resume"):
        initialization_action(
            "converted-weights-only",
            checkpoint_is_hf=False,
            explicit_resume_checkpoint="/parent/checkpoint-25391",
        )


def test_weight_only_initialization_forbids_auto_discovered_checkpoint() -> None:
    """A stale output checkpoint cannot silently restore optimizer state."""
    with pytest.raises(ValueError, match="auto-discovered checkpoint"):
        initialization_action(
            "converted-weights-only",
            checkpoint_is_hf=False,
            explicit_resume_checkpoint=None,
            auto_discovered_checkpoint="/output/checkpoint-20",
        )


@pytest.mark.parametrize(
    ("state", "method", "message"),
    [
        ({"modelopt_state_dict": []}, "dflash", "converted"),
        (_state(block_size=16), "dflash", "block size"),
        (_state(projector_type="dspark"), "dflash", "method"),
    ],
)
def test_weight_only_initialization_rejects_wrong_parent_architecture(
    state: dict[str, object], method: str, message: str
) -> None:
    """Method and B8 are authenticated from the converted ModelOpt state."""
    with pytest.raises(ValueError, match=message):
        validate_converted_modelopt_state(state, expected_method=method, expected_block_size=8)


def test_weight_only_initialization_rejects_wrong_q30_architecture_identity() -> None:
    """Method and B8 alone cannot authorize a differently shaped parent drafter."""
    state = _state()
    state["modelopt_state_dict"][-1][1]["config"]["dflash_architecture_config"][  # type: ignore[index]
        "num_attention_heads"
    ] = 16

    with pytest.raises(ValueError, match="architecture identity"):
        validate_converted_modelopt_state(
            state,
            expected_method="dflash",
            expected_block_size=8,
            expected_architecture={
                "num_hidden_layers": 5,
                "num_attention_heads": 32,
                "num_key_value_heads": 4,
                "head_dim": 128,
                "intermediate_size": 6144,
            },
        )


def test_model_arguments_exposes_only_base_or_weight_only_initialization() -> None:
    """The recipe schema exposes no generic optimizer-resume policy."""
    source = (_REPO_ROOT / "modelopt/torch/speculative/plugins/hf_training_args.py").read_text()
    tree = ast.parse(source)
    model_arguments = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "ModelArguments"
    )
    policy = next(
        node
        for node in model_arguments.body
        if isinstance(node, ast.AnnAssign)
        and isinstance(node.target, ast.Name)
        and node.target.id == "initialization_policy"
    )

    assert ast.unparse(policy.annotation) == "Literal['base', 'converted-weights-only']"
    assert isinstance(policy.value, ast.Constant) and policy.value.value == "base"


def test_training_entrypoint_validates_converted_parent_without_reconversion() -> None:
    """The actual trainer dispatches weight-only parents to validation, not mtsp.convert."""
    source = (_REPO_ROOT / "examples/speculative_decoding/main.py").read_text()

    assert "action = initialization_action(" in source
    assert 'if action == "validate-weight-only":' in source
    assert 'elif action == "convert":' in source
    assert "validate_converted_modelopt_state(" in source
    assert "explicit_resume_checkpoint=training_args.resume_from_checkpoint" in source


def test_training_entrypoint_imports_requested_dflash_method_at_module_scope() -> None:
    """The supported initialization helper is an ordinary eager dependency."""
    source = (_REPO_ROOT / "examples/speculative_decoding/main.py").read_text()
    tree = ast.parse(source)
    requested_imports = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom)
        and node.module == "modelopt.torch.speculative.plugins.hf_weight_only_initialization"
        and any(alias.name == "requested_dflash_method" for alias in node.names)
    ]

    assert len(requested_imports) == 1
    assert requested_imports[0] in tree.body


def test_strict_exposure_is_recorded_at_trainer_consumption_boundary() -> None:
    """Q30 evidence follows consumed batches, not DataLoaderShard's prefetched batch."""
    source = (_REPO_ROOT / "examples/speculative_decoding/main.py").read_text()
    tree = ast.parse(source)
    trainer = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "_Q30ConsumptionEvidenceTrainer"
    )
    compute_loss = next(
        node
        for node in trainer.body
        if isinstance(node, ast.FunctionDef) and node.name == "compute_loss"
    )

    assert any(
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "record_q30_consumed_exposure"
        for node in ast.walk(compute_loss)
    )
    assert 'Q30StrictExposureCollator(data_module["data_collator"])' in source
    assert "trainer_class = _Q30ConsumptionEvidenceTrainer if strict_exposure" in source


def test_weight_only_parent_load_is_patched_and_uses_explicit_target_tokenizer() -> None:
    """Parent weights load under the HF5 patch while tokenization uses the target snapshot."""
    load_parent = getattr(_initialization, "load_converted_parent", None)
    tokenizer_source = getattr(_initialization, "resolve_tokenizer_name_or_path", None)
    assert callable(load_parent), "weight-only parent loader is missing"
    assert callable(tokenizer_source), "weight-only tokenizer resolver is missing"

    events: list[str] = []

    class _Patch:
        def __enter__(self):
            events.append("patch-enter")

        def __exit__(self, *_args: object) -> None:
            events.append("patch-exit")

    def fake_load(path: str, **_kwargs: object) -> object:
        events.append(f"load:{path}")
        return object()

    model = load_parent(
        fake_load,
        _Patch,
        "/node-local/parent",
        use_fake_base=False,
        use_offline_training=True,
        trust_remote_code=True,
    )

    assert model is not None
    assert events == ["patch-enter", "load:/node-local/parent", "patch-exit"]
    assert (
        tokenizer_source(
            "converted-weights-only",
            model_name_or_path="/node-local/parent",
            tokenizer_name_or_path="/node-local/target",
        )
        == "/node-local/target"
    )
    with pytest.raises(ValueError, match="explicit authenticated target tokenizer"):
        tokenizer_source(
            "converted-weights-only",
            model_name_or_path="/node-local/parent",
            tokenizer_name_or_path=None,
        )


def test_weight_only_parent_requires_exact_dflash_trainable_parameter_identity() -> None:
    """Only a nonempty DFlash module may remain trainable after parent restoration."""
    validate_trainable = getattr(
        _initialization, "validate_dflash_trainable_parameter_identity", None
    )
    assert callable(validate_trainable), "weight-only trainable-parameter validator is missing"

    class _Parameter:
        def __init__(self, requires_grad: bool) -> None:
            self.requires_grad = requires_grad

    class _Model:
        def __init__(self, parameters: list[tuple[str, _Parameter]]) -> None:
            self._parameters = parameters

        def named_parameters(self):
            return iter(self._parameters)

    validate_trainable(
        _Model(
            [
                ("model.layers.0.weight", _Parameter(False)),
                ("dflash_module.layers.0.weight", _Parameter(True)),
            ]
        )
    )
    with pytest.raises(ValueError, match="base parameter is trainable"):
        validate_trainable(
            _Model(
                [
                    ("model.layers.0.weight", _Parameter(True)),
                    ("dflash_module.layers.0.weight", _Parameter(True)),
                ]
            )
        )
    with pytest.raises(ValueError, match="draft parameter is frozen"):
        validate_trainable(
            _Model(
                [
                    ("model.layers.0.weight", _Parameter(False)),
                    ("dflash_module.layers.0.weight", _Parameter(False)),
                ]
            )
        )
