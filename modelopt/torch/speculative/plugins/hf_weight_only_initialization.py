# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Validation for fresh-schedule training from converted speculative weights."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping
    from contextlib import AbstractContextManager

__all__ = [
    "initialization_action",
    "load_converted_parent",
    "requested_dflash_method",
    "resolve_tokenizer_name_or_path",
    "validate_converted_modelopt_state",
    "validate_dflash_trainable_parameter_identity",
]


def requested_dflash_method(projector_type: object) -> Literal["dflash", "dspark", "dflash2"]:
    """Map only supported DFlash-family projector requests to an initialization method."""
    if projector_type is None or projector_type == "dflash":
        return "dflash"
    if projector_type == "dspark":
        return "dspark"
    if projector_type == "dflash2":
        return "dflash2"
    raise ValueError(f"weight-only initialization method is unsupported: {projector_type}")


def initialization_action(
    policy: str,
    *,
    checkpoint_is_hf: bool,
    explicit_resume_checkpoint: str | None,
    auto_discovered_checkpoint: str | None = None,
) -> Literal["convert", "load-converted", "validate-weight-only"]:
    """Select conversion behavior without allowing a parent optimizer-state resume."""
    if policy == "converted-weights-only":
        if explicit_resume_checkpoint is not None:
            raise ValueError(
                "converted-weights-only initialization forbids an explicit resume checkpoint"
            )
        if auto_discovered_checkpoint is not None:
            raise ValueError(
                "converted-weights-only initialization forbids an auto-discovered checkpoint"
            )
        return "validate-weight-only"
    if policy != "base":
        raise ValueError(f"unsupported initialization policy: {policy}")
    return "load-converted" if checkpoint_is_hf else "convert"


def load_converted_parent(
    load_model: Callable[..., Any],
    patch_context: Callable[[], AbstractContextManager[None]],
    model_name_or_path: str,
    *,
    use_fake_base: bool,
    use_offline_training: bool,
    trust_remote_code: bool,
) -> Any:
    """Restore a converted parent without allowing HF5 to change its freeze identity."""
    with patch_context():
        return load_model(
            model_name_or_path,
            use_fake_base=use_fake_base,
            use_offline_training=use_offline_training,
            dtype="auto",
            device_map="cpu",
            trust_remote_code=trust_remote_code,
        )


def resolve_tokenizer_name_or_path(
    initialization_policy: str,
    *,
    model_name_or_path: str,
    tokenizer_name_or_path: str | None,
) -> str:
    """Select the tokenizer source without falling back to a weight-only parent."""
    if initialization_policy == "converted-weights-only":
        if tokenizer_name_or_path is None:
            raise ValueError(
                "converted-weights-only initialization requires an explicit authenticated "
                "target tokenizer"
            )
        return tokenizer_name_or_path
    return tokenizer_name_or_path or model_name_or_path


def validate_dflash_trainable_parameter_identity(model: Any) -> None:
    """Require exactly the DFlash draft subtree to remain trainable after restoration."""
    parameters = list(model.named_parameters())
    draft_parameters = [
        (name, parameter)
        for name, parameter in parameters
        if name == "dflash_module" or name.startswith("dflash_module.") or ".dflash_module." in name
    ]
    if not draft_parameters:
        raise ValueError("weight-only initialization has no DFlash draft parameters")
    for name, parameter in draft_parameters:
        if not parameter.requires_grad:
            raise ValueError(f"weight-only draft parameter is frozen: {name}")
    draft_names = {name for name, _ in draft_parameters}
    for name, parameter in parameters:
        if name not in draft_names and parameter.requires_grad:
            raise ValueError(f"weight-only base parameter is trainable: {name}")


def validate_converted_modelopt_state(
    state: dict[str, Any],
    *,
    expected_method: str,
    expected_block_size: int,
    expected_architecture: Mapping[str, object] | None = None,
) -> None:
    """Require an already-converted DFlash-family checkpoint with exact method and block size."""
    modes = state.get("modelopt_state_dict")
    if not isinstance(modes, list) or not modes:
        raise ValueError("weight-only initialization requires a converted ModelOpt checkpoint")
    last = modes[-1]
    if not isinstance(last, (list, tuple)) or len(last) != 2 or last[0] != "dflash":
        raise ValueError("weight-only initialization requires a converted DFlash checkpoint")
    mode_state = last[1]
    if not isinstance(mode_state, dict) or not isinstance(mode_state.get("config"), dict):
        raise ValueError("weight-only initialization has invalid converted mode state")
    config = mode_state["config"]
    if (
        type(config.get("dflash_block_size")) is not int
        or config["dflash_block_size"] != expected_block_size
    ):
        raise ValueError("weight-only initialization block size mismatch")
    architecture = config.get("dflash_architecture_config")
    if not isinstance(architecture, dict):
        raise ValueError("weight-only initialization has invalid architecture state")
    actual_method = requested_dflash_method(architecture.get("projector_type"))
    if actual_method != expected_method:
        raise ValueError(
            f"weight-only initialization method mismatch: expected {expected_method}, got {actual_method}"
        )
    if expected_architecture is not None and any(
        architecture.get(name) != value for name, value in expected_architecture.items()
    ):
        raise ValueError("weight-only initialization architecture identity mismatch")
