from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]


def _load(name: str):
    path = ROOT / "examples/dataset" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    sys.path.insert(0, str(path.parent))
    try:
        spec.loader.exec_module(module)
    finally:
        sys.path.pop(0)
    return module


def test_identifier_digest_preserves_duplicate_occurrences_and_order() -> None:
    contracts = _load("specdec_corpus_contracts")
    assert (
        contracts.sha256_canonical_json(("a", "a", "b"))
        == "2819ab78cc3edbe00e0df2eed49a4a1cc7a1003c42f96fa6e24d8c7a5a63d9b6"
    )
    assert (
        contracts.sha256_canonical_json(("a", "b"))
        == "0473ef2dc0d324ab659d3580c1134e9d812035905c4781fdd6d529b0c6860e13"
    )


def test_prompt_uuid_strips_storage_fields_but_preserves_order_and_tools() -> None:
    module = _load("specdec_identity")
    messages = [{"role": "user", "content": "q"}, {"role": "assistant", "content": "a"}]
    tools = [{"type": "function", "function": {"name": "shell"}}]
    stored = [dict(messages[0], row_index=9), messages[1]]
    assert module.prompt_uuid(stored, tools) == module.prompt_uuid(messages, tools)
    assert module.prompt_uuid(list(reversed(messages)), tools) != module.prompt_uuid(
        messages, tools
    )
    assert module.prompt_uuid(messages, []) != module.prompt_uuid(messages, tools)


def test_exclusion_rejects_prior_heldout_duplicate_and_collision() -> None:
    contracts = _load("specdec_corpus_contracts")
    module = _load("specdec_identity")

    def candidate(uuid: str, payload: bytes = b"same"):
        return contracts.CanonicalPrompt(
            uuid, payload, "source", "a" * 40, "b" * 64, 0, "math", "en"
        )

    index = module.ExclusionIndex(prior={"prior"}, held_out={"held"})
    with pytest.raises(module.ContaminationError):
        index.admit(candidate("prior"))
    with pytest.raises(module.ContaminationError):
        index.admit(candidate("held"))
    index.admit(candidate("new"))
    with pytest.raises(module.DuplicatePromptError):
        index.admit(candidate("new"))
    with pytest.raises(module.UUIDCollisionError):
        index.admit(candidate("new", b"different"))
