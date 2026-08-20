# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Unit tests for FakeBaseModel and the fake-base / offline paths in load_vlm_or_llm."""

import json
from copy import deepcopy
from types import SimpleNamespace

import pytest
import safetensors.torch
import torch
from _test_utils.torch.speculative.dflash import get_dflash_config

pytest.importorskip("transformers")
import transformers

import modelopt.torch.speculative as mtsp
from modelopt.torch.speculative.config import DFLASH_DEFAULT_CFG
from modelopt.torch.speculative.plugins.modeling_fakebase import FakeBaseModel
from modelopt.torch.speculative.utils import load_vlm_or_llm

_HIDDEN_SIZE = 16
_VOCAB_SIZE = 32


def _legacy_config(rope_theta=None, rope_scaling=None):
    """Build a pre-Transformers-5 config without a rope_parameters alias."""
    return SimpleNamespace(
        model_type="llama",
        hidden_size=_HIDDEN_SIZE,
        vocab_size=_VOCAB_SIZE,
        num_hidden_layers=2,
        max_position_embeddings=128,
        tie_word_embeddings=False,
        rope_theta=rope_theta,
        rope_scaling=rope_scaling,
    )


@pytest.fixture
def fake_config(monkeypatch):
    """Monkeypatch AutoConfig.from_pretrained to return a minimal fake config."""
    cfg = transformers.PretrainedConfig()
    cfg.model_type = "llama"
    cfg.hidden_size = _HIDDEN_SIZE
    cfg.vocab_size = _VOCAB_SIZE
    cfg.num_hidden_layers = 2
    cfg.max_position_embeddings = 128
    cfg.tie_word_embeddings = False
    monkeypatch.setattr(transformers.AutoConfig, "from_pretrained", lambda *a, **kw: cfg)
    return cfg


@pytest.fixture
def fake_checkpoint(tmp_path, fake_config):
    """Minimal local safetensors checkpoint loadable by FakeBaseModel."""
    tensors = {
        "lm_head.weight": torch.zeros(_VOCAB_SIZE, _HIDDEN_SIZE),
        "embed_tokens.weight": torch.zeros(_VOCAB_SIZE, _HIDDEN_SIZE),
        # model_type "llama" is in the final-norm whitelist, so FakeBaseModel requires the norm.
        "norm.weight": torch.ones(_HIDDEN_SIZE),
    }
    shard = tmp_path / "model-00001-of-00001.safetensors"
    safetensors.torch.save_file(tensors, shard)
    index = {"weight_map": dict.fromkeys(tensors, shard.name)}
    (tmp_path / "model.safetensors.index.json").write_text(json.dumps(index))
    return tmp_path


def test_fakebase_local_happy_path(fake_checkpoint):
    model = FakeBaseModel.from_source(str(fake_checkpoint))
    assert model.lm_head.weight.shape == torch.Size([_VOCAB_SIZE, _HIDDEN_SIZE])
    assert model.embed_tokens.weight.shape == torch.Size([_VOCAB_SIZE, _HIDDEN_SIZE])


def test_fakebase_prefers_transformers5_rope_parameters(fake_checkpoint, fake_config):
    """Canonical Transformers 5 RoPE metadata wins over a flat compatibility default."""
    fake_config.rope_parameters = {"rope_type": "default", "rope_theta": 1000000.0}
    fake_config.rope_theta = 10000.0
    fake_config.num_attention_heads = 4
    fake_config.num_key_value_heads = 2
    fake_config.intermediate_size = 32

    fake_base = FakeBaseModel.from_source(str(fake_checkpoint))
    assert fake_base.config.rope_theta == 1000000.0


def test_fakebase_prefers_legacy_rope_theta(fake_checkpoint, monkeypatch):
    """A legacy top-level RoPE base remains authoritative over nested metadata."""
    legacy_config = _legacy_config(
        rope_theta=500000.0,
        rope_scaling={"rope_type": "default", "rope_theta": 1000000.0},
    )
    monkeypatch.setattr(transformers.AutoConfig, "from_pretrained", lambda *a, **kw: legacy_config)

    fake_base = FakeBaseModel.from_source(str(fake_checkpoint))

    assert fake_base.config.rope_theta == 500000.0


def test_fakebase_reads_legacy_rope_scaling(fake_checkpoint, monkeypatch):
    """Legacy rope_scaling remains a fallback when canonical metadata is absent."""
    legacy_config = _legacy_config(rope_scaling={"rope_type": "default", "rope_theta": 1000000.0})
    monkeypatch.setattr(transformers.AutoConfig, "from_pretrained", lambda *a, **kw: legacy_config)

    fake_base = FakeBaseModel.from_source(str(fake_checkpoint))

    assert fake_base.config.rope_theta == 1000000.0


def test_fakebase_reads_qwen3_vlm_text_config_rope_parameters(
    fake_checkpoint, fake_config, monkeypatch
):
    """VLM FakeBase construction reads canonical RoPE from its Qwen3-MoE text config."""
    fake_config.model_type = "qwen3_moe"
    fake_config.rope_theta = 10000.0
    fake_config.rope_parameters = {"rope_type": "default", "rope_theta": 1000000.0}
    vlm_config = transformers.PretrainedConfig()
    vlm_config.model_type = "qwen3_vl"
    vlm_config.text_config = fake_config
    monkeypatch.setattr(transformers.AutoConfig, "from_pretrained", lambda *a, **kw: vlm_config)

    fake_base = FakeBaseModel.from_source(str(fake_checkpoint))

    assert fake_base.config.rope_theta == 1000000.0


@pytest.mark.parametrize("projector_type", ["dflash", "dspark"])
def test_fakebase_nested_rope_theta_reaches_draft_rotary_and_export(
    fake_checkpoint, fake_config, projector_type, tmp_path
):
    """Nested target RoPE survives FakeBase, DFlash/DSpark conversion, and export."""
    fake_config.model_type = "qwen3_moe"
    fake_config.rope_theta = 10000.0
    fake_config.rope_parameters = {"rope_type": "default", "rope_theta": 1000000.0}
    fake_config.num_attention_heads = 4
    fake_config.num_key_value_heads = 2
    fake_config.intermediate_size = 32
    fake_base = FakeBaseModel.from_source(str(fake_checkpoint))

    config = get_dflash_config(offline=True)
    if projector_type == "dspark":
        config = deepcopy(DFLASH_DEFAULT_CFG["config"])
        config.update(
            {
                "dflash_block_size": 4,
                "dflash_use_torch_compile": False,
                "dflash_mask_token_id": 0,
                "dflash_offline": True,
                "dflash_architecture_config": {
                    "num_hidden_layers": 2,
                    "projector_type": "dspark",
                    "markov_rank": 4,
                },
            }
        )
    mtsp.convert(fake_base, [("dflash", config)])

    assert fake_base.config.rope_theta == 1000000.0
    assert fake_base.dflash_config.rope_parameters["rope_theta"] == 1000000.0
    fake_base.dflash_module._maybe_init_rotary_emb(device="cpu")
    assert fake_base.dflash_module.rotary_emb.config.rope_parameters["rope_theta"] == 1000000.0

    export_dir = tmp_path / projector_type
    fake_base.get_exporter().export(export_dir)
    with open(export_dir / "config.json") as f:
        exported_config = json.load(f)
    assert exported_config["rope_theta"] == 1000000.0


def test_fakebase_missing_index_raises(tmp_path, fake_config):
    with pytest.raises(FileNotFoundError, match="safetensors"):
        FakeBaseModel.from_source(str(tmp_path))


def test_fakebase_single_file_no_index(tmp_path, fake_config):
    """Small models often ship a single ``model.safetensors`` without an index.json."""
    tensors = {
        "lm_head.weight": torch.zeros(_VOCAB_SIZE, _HIDDEN_SIZE),
        "embed_tokens.weight": torch.zeros(_VOCAB_SIZE, _HIDDEN_SIZE),
        "norm.weight": torch.ones(_HIDDEN_SIZE),
    }
    safetensors.torch.save_file(tensors, tmp_path / "model.safetensors")
    model = FakeBaseModel.from_source(str(tmp_path))
    assert model.lm_head.weight.shape == torch.Size([_VOCAB_SIZE, _HIDDEN_SIZE])
    assert model.embed_tokens.weight.shape == torch.Size([_VOCAB_SIZE, _HIDDEN_SIZE])


def test_fakebase_tied_embeddings_falls_back_to_embed(tmp_path, fake_config):
    """Tied-embeddings models (e.g. Llama-3.2-1B) omit ``lm_head`` from safetensors;
    FakeBaseModel must reuse ``embed_tokens`` for both."""
    fake_config.tie_word_embeddings = True
    weight = torch.randn(_VOCAB_SIZE, _HIDDEN_SIZE)
    safetensors.torch.save_file(
        {"embed_tokens.weight": weight, "norm.weight": torch.ones(_HIDDEN_SIZE)},
        tmp_path / "model.safetensors",
    )
    model = FakeBaseModel.from_source(str(tmp_path))
    torch.testing.assert_close(model.lm_head.weight, weight)
    torch.testing.assert_close(model.embed_tokens.weight, weight)


def test_fakebase_missing_lm_head_without_tying_raises(tmp_path, fake_config):
    """Without ``tie_word_embeddings`` a missing ``lm_head`` is still an error."""
    safetensors.torch.save_file(
        {"embed_tokens.weight": torch.zeros(_VOCAB_SIZE, _HIDDEN_SIZE)},
        tmp_path / "model.safetensors",
    )
    with pytest.raises(RuntimeError, match="lm_head"):
        FakeBaseModel.from_source(str(tmp_path))


def test_load_vlm_or_llm_returns_fakebase(fake_checkpoint):
    model = load_vlm_or_llm(str(fake_checkpoint), use_offline_training=True, use_fake_base=True)
    assert isinstance(model, FakeBaseModel)


def test_load_vlm_or_llm_offline_zero_layers(monkeypatch):
    cfg = transformers.PretrainedConfig()
    cfg.model_type = "llama"
    cfg.num_hidden_layers = 4
    monkeypatch.setattr(transformers.AutoConfig, "from_pretrained", lambda *a, **kw: cfg)

    captured_kwargs = {}

    class _FakeModel:
        config = cfg

    def _fake_from_pretrained(*args, **kwargs):
        captured_kwargs.update(kwargs)
        return _FakeModel()

    monkeypatch.setattr(transformers.AutoModelForCausalLM, "from_pretrained", _fake_from_pretrained)

    model = load_vlm_or_llm("fake-model", use_offline_training=True, use_fake_base=False)
    assert captured_kwargs.get("num_hidden_layers") == 0
    assert model.config.num_orig_hidden_layers == 4


def test_load_vlm_or_llm_uses_transformers5_vlm_auto_class(monkeypatch):
    """Transformers 5 loads VLMs through AutoModelForImageTextToText."""
    cfg = transformers.PretrainedConfig()
    cfg.model_type = "qwen3_vl"
    cfg.text_config = object()
    monkeypatch.setattr(transformers.AutoConfig, "from_pretrained", lambda *a, **kw: cfg)

    captured = {}

    class _FakeVLM:
        @staticmethod
        def from_pretrained(*args, **kwargs):
            captured["args"] = args
            captured["kwargs"] = kwargs
            return object()

    # ``transformers`` exposes auto classes lazily, so deleting this attribute
    # lets its module-level ``__getattr__`` recreate the legacy class.  An
    # explicit ``None`` models its absence and reliably exercises the v5
    # fallback.
    monkeypatch.setattr(transformers, "AutoModelForVision2Seq", None, raising=False)
    monkeypatch.setattr(transformers, "AutoModelForImageTextToText", _FakeVLM)

    assert load_vlm_or_llm("qwen3-vl", dtype="auto") is not None
    assert captured["args"] == ("qwen3-vl",)
    assert captured["kwargs"]["torch_dtype"] == "auto"
