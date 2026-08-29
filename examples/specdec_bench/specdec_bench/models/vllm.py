# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import asyncio
import json
import os
import time

from .base import Model

try:
    from vllm import SamplingParams
    from vllm.engine.arg_utils import AsyncEngineArgs
    from vllm.inputs import TokensPrompt
    from vllm.v1.engine.async_llm import AsyncLLM
except ImportError:
    print("vllm is not installed.")
    vllm = None


# vLLM serves DFlash2 under ``method="dflash"`` and selects the V2 speculator
# from the draft checkpoint's ``architectures`` instead (vllm#52816). So a
# DFLASH2 run pointed at a plain DFlash drafter starts cleanly and silently
# benchmarks DFlash. Only the checkpoint distinguishes them.
DRAFT_ARCHITECTURES = {
    "DFLASH": "DFlashDraftModel",
    "DFLASH2": "DFlash2DraftModel",
    "DSPARK": "Qwen3DSparkModel",
}


# ``--block_size`` is B, the DFlash-family block width, but vLLM's
# ``num_speculative_tokens`` is K, the count actually proposed per block, and
# the relation between them differs by family. DFlash and DFlash2 predict
# positions 1..B-1 of each block, so K = B - 1 -- which is why the walkthrough
# serves a ``dflash_block_size=8`` checkpoint with ``num_speculative_tokens: 7``
# (examples/speculative_decoding/doc/dflash.md). DSpark's Markov head emits the
# final position too, so K = B. Forwarding B unconverted made the DFlash arms
# propose one token more than they were trained to emit, and made their
# acceptance length incomparable with DSpark's measured at the same B.
DRAFT_TOKEN_OFFSET = {"DFLASH": -1, "DFLASH2": -1, "DSPARK": 0}
DEFAULT_BLOCK_SIZE = 8


def _draft_tokens(algorithm, kwargs):
    """Convert the requested block size into vLLM's proposed-token count."""
    # ``run.py`` always passes this key, carrying None when --block_size is
    # unset, so ``kwargs.get(key, default)`` returns None rather than the
    # default and hands vLLM a null horizon.
    block_size = kwargs.get("speculative_num_draft_tokens") or DEFAULT_BLOCK_SIZE
    tokens = block_size + DRAFT_TOKEN_OFFSET[algorithm]
    if tokens < 1:
        raise ValueError(
            f"{algorithm} with block size {block_size} would propose {tokens} tokens."
        )
    return tokens


def _assert_draft_architecture(draft_model_dir, expected, algorithm):
    """Fail before serving if the drafter is not the architecture ``algorithm`` asks for."""
    if not draft_model_dir:
        raise ValueError(f"{algorithm} requires --draft_model_dir.")
    config_path = os.path.join(draft_model_dir, "config.json")
    if not os.path.isfile(config_path):
        # A repo id, or a layout we cannot inspect locally — defer to vLLM.
        return
    with open(config_path) as f:
        architectures = json.load(f).get("architectures") or []
    if expected not in architectures:
        raise ValueError(
            f"{algorithm} expects a drafter whose architectures include {expected!r}, "
            f"but {draft_model_dir} declares {architectures}. "
            "Both DFlash and DFlash2 are served as method='dflash', so a mismatch "
            "would run the wrong speculator and report a valid-looking number."
        )


# Forwarded from ``--runtime_params`` ``engine_args.<key>``; extend as needed.
PASSTHROUGH_ENGINE_ARGS = (
    "mamba_backend",
    "mamba_ssm_cache_dtype",
    "mamba_cache_mode",
    "mamba_cache_philox_rounds",
    "enable_mamba_cache_stochastic_rounding",
)


class VLLMModel(Model):
    # Cross-engine ``--max_seq_len`` (run.py) lands in kwargs under the
    # vLLM-native name ``max_model_len`` (see run.py's ``_MAX_SEQ_LEN_KEY``)
    # and is read at line ~92 below into AsyncEngineArgs.

    def __init__(self, model_dir, max_concurrent_requests, sampling_kwargs, **kwargs):
        specdec = None
        if kwargs.get("speculative_algorithm") == "EAGLE3":
            specdec = {
                "method": "eagle3",
                "model": kwargs.get("draft_model_dir"),
                "num_speculative_tokens": kwargs.get("speculative_num_steps", 3),
            }
        elif kwargs.get("speculative_algorithm") == "EAGLE":
            specdec = {
                "method": "eagle",
                "model": kwargs.get("draft_model_dir"),
                "num_speculative_tokens": kwargs.get("speculative_num_steps", 3),
            }
        elif kwargs.get("speculative_algorithm") == "NGRAM":
            specdec = {
                "method": "ngram",
                "num_speculative_tokens": kwargs.get("speculative_num_steps", 3),
                "prompt_lookup_max": kwargs.get("max_matching_ngram_size", 3),  # No idea here
            }
        elif kwargs.get("speculative_algorithm") == "DRAFT_TARGET":
            specdec = {
                "method": "draft_model",
                "model": kwargs.get("draft_model_dir"),
                "num_speculative_tokens": kwargs.get("speculative_num_steps", 3),
            }
            if kwargs.get("parallel_draft_block_sizes") is not None:
                specdec["disable_padded_drafter_batch"] = True
                specdec["parallel_draft_block_sizes"] = kwargs.get("parallel_draft_block_sizes")
        elif kwargs.get("speculative_algorithm") == "MTP":
            # vLLM's ``SpeculativeConfig.__post_init__`` (vllm/config/
            # speculative.py:529-602) does method auto-detection ONLY
            # when ``method`` is unset — when ``model`` is provided and
            # ``method`` is None, the default branch sets
            # ``method = "draft_model"`` (the generic same-architecture
            # draft path), NOT MTP. That path enforces equal num_heads
            # between target and draft and raises
            # ``AssertionError: All layers in one attention group must
            # share num_heads`` on heterogeneous-head models like
            # Gemma 4 (target=8 heads, assistant=4).
            #
            # The canonical config for ALL MTP variants is to ALWAYS
            # pass ``method="mtp"`` AND ADD ``model=<assistant>`` only
            # when the family uses a separate assistant model. vLLM's
            # own test at ``tests/v1/e2e/spec_decode/test_spec_decode.py``
            # (lines 818-823) does exactly this for the gemma4-e4b
            # parametrization:
            #
            #     speculative_config = {
            #         "method": "mtp",
            #         "num_speculative_tokens": ...,
            #     }
            #     if draft_model is not None:        # Gemma 4 case
            #         speculative_config["model"] = draft_model
            #
            # Surfaced on OMNIML-5024 pipeline #54356795: dropping the
            # ``method`` key when ``draft_model_dir`` was provided sent
            # the call into the generic draft_model path, hitting the
            # num_heads assertion. Restored both keys.
            specdec = {
                "method": "mtp",
                "num_speculative_tokens": kwargs.get("speculative_num_steps", 3),
            }
            draft_model_dir = kwargs.get("draft_model_dir")
            if draft_model_dir:
                # Gemma 4 family (E2B / E4B / 26B-A4B / 31B) uses a
                # separate assistant checkpoint as the MTP draft.
                # vLLM auto-detects Gemma4 MTP from the assistant
                # ``model_type=gemma4_assistant`` and rewrites it to
                # ``gemma4_mtp`` (speculative.py:511-522). For
                # families where the MTP layer ships inside the
                # target (Qwen 3.5 etc.), omit ``--draft_model_dir``
                # and let vLLM use the target model as its own draft
                # (handled in speculative.py:562-573).
                specdec["model"] = draft_model_dir
        elif kwargs.get("speculative_algorithm") in ("DFLASH", "DFLASH2"):
            algorithm = kwargs["speculative_algorithm"]
            _assert_draft_architecture(
                kwargs.get("draft_model_dir"), DRAFT_ARCHITECTURES[algorithm], algorithm
            )
            # Both variants are served as "dflash"; the drafter's architectures
            # pick the speculator.
            specdec = {
                "method": "dflash",
                "model": kwargs.get("draft_model_dir"),
                "num_speculative_tokens": _draft_tokens(algorithm, kwargs),
            }
        elif kwargs.get("speculative_algorithm") == "DSPARK":
            _assert_draft_architecture(
                kwargs.get("draft_model_dir"), DRAFT_ARCHITECTURES["DSPARK"], "DSPARK"
            )
            specdec = {
                "method": "dspark",
                "model": kwargs.get("draft_model_dir"),
                "num_speculative_tokens": _draft_tokens("DSPARK", kwargs),
                "draft_sample_method": kwargs.get("draft_sample_method", "greedy"),
            }
        elif kwargs.get("speculative_algorithm") == "NONE":
            specdec = None

        if specdec is None:
            num_speculative_tokens = 1
        else:
            num_speculative_tokens = specdec.get("num_speculative_tokens", 3)

        engine_args = AsyncEngineArgs(
            model=model_dir,
            tokenizer=kwargs.get("tokenizer_path"),
            trust_remote_code=kwargs.get("trust_remote_code", False),
            tensor_parallel_size=kwargs.get("tensor_parallel_size", 1),
            enable_expert_parallel=kwargs.get("moe_expert_parallel_size", 1) > 1,
            enable_prefix_caching=kwargs.get("prefix_cache", False),
            speculative_config=specdec,
            max_num_seqs=max_concurrent_requests * num_speculative_tokens,
            skip_tokenizer_init=False,
            async_scheduling=kwargs.get("async_scheduling", True),
            enforce_eager=kwargs.get("enforce_eager", False),
            max_model_len=kwargs.get("max_model_len"),
            **{key: kwargs[key] for key in PASSTHROUGH_ENGINE_ARGS if kwargs.get(key) is not None},
        )
        self.engine_args = engine_args
        self.model = AsyncLLM.from_engine_args(engine_args)
        self.sampling_kwargs = sampling_kwargs
        # https://github.com/vllm-project/vllm/blob/main/vllm/sampling_params.py
        self.sampling_config = SamplingParams(
            detokenize=False,
            temperature=sampling_kwargs.get("temperature", 1.0),
            top_p=sampling_kwargs.get("top_p", 1.0),
            top_k=sampling_kwargs.get("top_k", 0),
        )
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)

    async def run(self, prompt_ids, max_length, end_id, request_id, turn_id):  # pragma: no cover
        output_dict = {}
        self.sampling_config.max_tokens = max_length
        self.sampling_config.stop_token_ids = [end_id]
        if end_id == -1:
            self.sampling_config.ignore_eos = True

        outputs, timing, full_tokens = await self.generate(prompt_ids, request_id, turn_id)

        reformatted_output_ids = [[] for _ in range(self.sampling_kwargs.get("beam_width", 1))]
        start = 0
        timing_to_strip = []
        for i in range(len(outputs)):
            if outputs[i] == start:
                timing_to_strip.append(i)
                continue
            if i == len(outputs) - 1:
                if full_tokens[-1] == end_id:
                    if outputs[i] - start == 1:
                        timing_to_strip.append(i)
                    else:
                        reformatted_output_ids[0].append(full_tokens[start : outputs[i] - 1])
                    break
            reformatted_output_ids[0].append(full_tokens[start : outputs[i]])
            start = outputs[i]
        output_dict["output_ids"] = reformatted_output_ids
        output_dict["output_logits"] = None
        output_dict["token_times"] = [
            timing[i] for i in range(len(timing)) if i not in timing_to_strip
        ]
        return output_dict

    async def generate(self, prompt_ids, request_id, turn_id):  # pragma: no cover
        timing = []
        timing.append(time.perf_counter())
        outputs = []
        full_tokens = []
        async for output in self.model.generate(
            request_id=f"{request_id}.{turn_id}",
            prompt=TokensPrompt(prompt_token_ids=prompt_ids),
            sampling_params=self.sampling_config,
        ):
            for completion in output.outputs:
                outputs.append(len(completion.token_ids))
                timing.append(time.perf_counter())
                full_tokens = completion.token_ids
            if output.finished:
                break
        return outputs, timing, full_tokens

    def get_serving_config(self):  # pragma: no cover
        """Dump the AsyncEngineArgs dataclass plus the runtime vllm_config when available."""
        try:
            import dataclasses

            cfg = dataclasses.asdict(self.engine_args)
        except Exception:
            cfg = {}
        # vllm exposes the resolved engine config on the AsyncLLM instance once
        # initialized — capture max_model_len / kv cache / dtype defaults that
        # don't appear in AsyncEngineArgs.
        try:
            vllm_config = getattr(self.model, "vllm_config", None)
            if vllm_config is not None and hasattr(vllm_config, "to_dict"):
                cfg["vllm_config"] = vllm_config.to_dict()
        except Exception:
            pass
        return cfg

    def stop(self):  # pragma: no cover
        try:
            self.loop.run_until_complete(self.model.shutdown())
            self.loop.close()
        except Exception:
            pass
