# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Fail-fast selector range instrumentation for the one-node DFlash2 gate."""

import functools
import json

import torch
from vllm.model_executor.models.qwen3_dflash2 import CandidateSelector

_original_forward = CandidateSelector.forward


@functools.wraps(_original_forward)
def _diagnostic_forward(
    self: CandidateSelector,
    candidate_ids: torch.Tensor,
    unary_logits: torch.Tensor,
    hidden_states: torch.Tensor,
    anchor_token_ids: torch.Tensor,
) -> torch.Tensor:
    predecessor_ids = torch.cat(
        (
            anchor_token_ids[:, None, None].expand(-1, 1, self.top_k),
            candidate_ids[:, :-1],
        ),
        dim=1,
    )
    ranges = {
        "candidate_min": int(candidate_ids.min().item()),
        "candidate_max": int(candidate_ids.max().item()),
        "candidate_dtype": str(candidate_ids.dtype),
        "candidate_shape": list(candidate_ids.shape),
        "anchor_min": int(anchor_token_ids.min().item()),
        "anchor_max": int(anchor_token_ids.max().item()),
        "anchor_dtype": str(anchor_token_ids.dtype),
        "anchor_shape": list(anchor_token_ids.shape),
        "predecessor_min": int(predecessor_ids.min().item()),
        "predecessor_max": int(predecessor_ids.max().item()),
        "successor_rows": int(self.successor_codebook.shape[0]),
        "predecessor_rows": int(self.predecessor_codebook.shape[0]),
    }
    print("DFLASH2_SELECTOR_RANGES " + json.dumps(ranges, sort_keys=True), flush=True)
    if not 0 <= ranges["candidate_min"] <= ranges["candidate_max"] < ranges["successor_rows"]:
        raise RuntimeError("DFlash2 candidate IDs are outside successor-codebook rows")
    if not 0 <= ranges["predecessor_min"] <= ranges["predecessor_max"] < ranges["predecessor_rows"]:
        raise RuntimeError("DFlash2 predecessor IDs are outside predecessor-codebook rows")
    return _original_forward(
        self,
        candidate_ids,
        unary_logits,
        hidden_states,
        anchor_token_ids,
    )


CandidateSelector.forward = _diagnostic_forward
INSTALLED = True
