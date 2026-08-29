# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Pin the assistant-mask template the inventory builder tokenizes with.

Qwen3-30B-A3B-Thinking-2507 ships a chat template with no ``{% generation %}``
marker. Asked for an assistant mask against such a template, transformers logs
a warning and returns an all-zero mask rather than raising, so every row scores
zero assistant tokens, every row is skipped, and a full tokenization pass over
the corpus produces an empty inventory with nothing having failed. The builder
therefore derives a masking template and proves it before use.

The derivation was already covered against a six-line synthetic template, which
is why nothing caught that the real one needed it. These tests run against the
template that actually ships.
"""

from __future__ import annotations

import ast
import importlib.util
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[3]
MODULE = ROOT / "examples/dataset/build_specdec_inventory.py"
LAUNCHER = ROOT / "tools/launcher/common/specdec/q30t_tokenizer_receipt.py"
OFFICIAL = (Path(__file__).parent / "data/q30t_official_chat_template.jinja").read_text(
    encoding="utf-8"
)


def _lift(path: Path, *names: str) -> dict[str, Any]:
    """Execute named top-level functions without importing their module.

    build_specdec_inventory imports transformers lazily but pulls a large
    dependency surface at module scope, and the launcher module is a different
    package root entirely. Both functions under test are self-contained.
    """
    tree = ast.parse(path.read_text(encoding="utf-8"))
    body = [
        node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in set(names)
    ]
    assert len(body) == len(names), f"missing {set(names) - {n.name for n in body}} in {path}"
    namespace: dict[str, Any] = {
        "Any": Any,
        "Mapping": Mapping,
        "_MASK_PROBE_MESSAGES": (
            {"role": "user", "content": "probe question"},
            {"role": "assistant", "content": "probe answer"},
        ),
    }
    exec(compile(ast.Module(body=body, type_ignores=[]), str(path), "exec"), namespace)
    return namespace


@pytest.fixture(scope="module")
def inventory() -> dict[str, Any]:
    return _lift(MODULE, "_training_chat_template", "_verify_masking_template")


def test_the_shipped_template_is_the_shape_the_derivation_expects(
    inventory: dict[str, Any],
) -> None:
    """A Qwen-side template edit has to fail here, not empty the corpus quietly.

    The derivation is anchored on exact substrings of the official template. If
    upstream reflows the assistant branch, this raises -- which is the whole
    point of anchoring it rather than pattern-matching loosely.
    """
    training = inventory["_training_chat_template"](OFFICIAL)
    assert "{%- generation %}" in training
    assert "{%- endgeneration %}" in training


def test_the_official_template_has_no_generation_marker_of_its_own(
    inventory: dict[str, Any],
) -> None:
    """The premise of the whole derivation, asserted rather than assumed."""
    assert "generation %}" not in OFFICIAL


def test_the_builder_and_the_launcher_derive_the_same_template() -> None:
    """Two copies exist because the trees cannot import each other.

    The tokenizer receipt pins training_chat_template_sha256 over the launcher's
    output, so a silent divergence would surface as an unexplainable receipt
    mismatch on the cluster. Catch it here instead.
    """
    built = _lift(MODULE, "_training_chat_template")["_training_chat_template"](OFFICIAL)
    pinned = _lift(LAUNCHER, "_training_chat_template")["_training_chat_template"](OFFICIAL)
    assert built == pinned


def test_a_template_the_derivation_does_not_recognize_is_refused(
    inventory: dict[str, Any],
) -> None:
    with pytest.raises(ValueError, match="approved assistant anchors"):
        inventory["_training_chat_template"]("{% for message in messages %}{% endfor %}")


class _StubTokenizer:
    """Answer apply_chat_template with whatever the test wants to prove is caught."""

    def __init__(self, *, rendered: dict[str, str], mask: Any) -> None:
        self._rendered = rendered
        self._mask = mask

    def apply_chat_template(self, messages: Any, **kwargs: Any) -> Any:
        template = kwargs["chat_template"]
        if not kwargs.get("tokenize"):
            return self._rendered[template]
        return {"input_ids": [1, 2, 3], "assistant_masks": self._mask}


def test_a_masking_template_that_changes_the_rendering_is_refused(
    inventory: dict[str, Any],
) -> None:
    """Rendering drift would train the drafter on text the target never emits."""
    tokenizer = _StubTokenizer(rendered={"official": "a", "training": "b"}, mask=[0, 1, 1])
    with pytest.raises(ValueError, match="render identically"):
        inventory["_verify_masking_template"](tokenizer, "official", "training")


def test_an_all_zero_mask_is_refused(inventory: dict[str, Any]) -> None:
    """This is the exact failure the shipped template produces."""
    tokenizer = _StubTokenizer(rendered={"official": "a", "training": "a"}, mask=[0, 0, 0])
    with pytest.raises(ValueError, match="no assistant-token mask"):
        inventory["_verify_masking_template"](tokenizer, "official", "training")


def test_a_non_list_mask_is_refused(inventory: dict[str, Any]) -> None:
    tokenizer = _StubTokenizer(rendered={"official": "a", "training": "a"}, mask=None)
    with pytest.raises(ValueError, match="no assistant-token mask"):
        inventory["_verify_masking_template"](tokenizer, "official", "training")


def test_a_correct_masking_template_passes(inventory: dict[str, Any]) -> None:
    tokenizer = _StubTokenizer(rendered={"official": "a", "training": "a"}, mask=[0, 1, 1])
    inventory["_verify_masking_template"](tokenizer, "official", "training")


@pytest.mark.skipif(
    importlib.util.find_spec("transformers") is None, reason="transformers not installed"
)
def test_the_derived_template_masks_assistant_turns_on_a_real_tokenizer(
    inventory: dict[str, Any], tmp_path: Path
) -> None:
    """The end-to-end claim, against a tokenizer that renders this template.

    Built from the shipped template plus a minimal word-level vocabulary rather
    than the 30B checkpoint's, because what is under test is the mask, not the
    merges.
    """
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import PreTrainedTokenizerFast

    words = [
        "<|im_start|>",
        "<|im_end|>",
        "<think>",
        "</think>",
        "\n",
        "user",
        "assistant",
        "probe",
        "question",
        "answer",
    ]
    backend = Tokenizer(models.WordLevel({w: i for i, w in enumerate(words)}, unk_token="probe"))
    backend.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend)

    training = inventory["_training_chat_template"](OFFICIAL)
    messages = [
        {"role": "user", "content": "probe question"},
        {"role": "assistant", "content": "probe answer"},
    ]
    encoded = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=False,
        return_dict=True,
        return_assistant_tokens_mask=True,
        chat_template=training,
    )
    mask = encoded["assistant_masks"]

    assert sum(mask) > 0, "derived template produced no assistant tokens"
    assert sum(mask) < len(mask), "derived template masked the prompt as well"
    # The official template renders the same text; only the mask is new.
    assert tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=False, chat_template=training
    ) == tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=False, chat_template=OFFICIAL
    )


def test_a_template_that_already_marks_its_span_is_left_alone(tmp_path: Path) -> None:
    """The loader serves the whole builder, not just the Q30 path.

    A tokenizer whose shipped template already carries a generation marker can
    mask on its own, and rewriting it would mean deriving against a shape the
    derivation never saw. The one in the shared test fixtures is exactly that
    case, so it stands in for the class.
    """
    already_tagged = (ROOT / "tests/_test_utils/torch/tokenizer/chat_template.jinja").read_text(
        encoding="utf-8"
    )
    assert "generation %}" in already_tagged or "generation -%}" in already_tagged
    # The Q30 derivation cannot read it, which is the point: the loader must
    # not attempt the rewrite at all rather than fail on a template that works.
    functions = _lift(MODULE, "_training_chat_template", "_verify_masking_template")
    with pytest.raises(ValueError, match="approved assistant anchors"):
        functions["_training_chat_template"](already_tagged)


def test_a_conversation_whose_mask_is_empty_is_refused_not_skipped() -> None:
    """This is the failure the whole fixture exists to make loud.

    An all-zero mask on a turn that has an assistant message means the template
    could not mark what to train on. The caller drops zero-token rows, so
    returning the mask here would turn a broken tokenizer into an empty corpus.
    """
    tokenize = _lift(MODULE, "_tokenize")["_tokenize"]

    class _Silent:
        def apply_chat_template(self, messages: list[Any], **_: Any) -> dict[str, Any]:
            return {"input_ids": [1, 2, 3], "assistant_masks": [0, 0, 0]}

    row = {"messages": [{"role": "user", "content": "q"}, {"role": "assistant", "content": "a"}]}
    with pytest.raises(ValueError, match="marked no assistant tokens"):
        tokenize(_Silent(), row)


def test_a_conversation_with_no_assistant_turn_keeps_its_empty_mask() -> None:
    """A prompt-only row legitimately has nothing masked.

    Refusing it too would reject the RL prompt sources the blend carries on
    purpose, so the guard keys on the presence of an assistant turn rather than
    on the mask alone.
    """
    tokenize = _lift(MODULE, "_tokenize")["_tokenize"]

    class _PromptOnly:
        def apply_chat_template(self, messages: list[Any], **_: Any) -> dict[str, Any]:
            return {"input_ids": [1, 2], "assistant_masks": [0, 0]}

    ids, mask = tokenize(_PromptOnly(), {"messages": [{"role": "user", "content": "q"}]})
    assert (ids, mask) == ([1, 2], [0, 0])
