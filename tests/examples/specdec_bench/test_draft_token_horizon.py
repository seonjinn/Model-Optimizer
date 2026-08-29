# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""``--block_size`` is B; vLLM's ``num_speculative_tokens`` is K, and B != K.

DFlash and DFlash2 predict positions 1..B-1 of each block, so K = B - 1: the
walkthrough serves a ``dflash_block_size=8`` checkpoint with
``num_speculative_tokens: 7`` (examples/speculative_decoding/doc/dflash.md).
DSpark's Markov head emits the final position too, so K = B. Forwarding B
unconverted made the DFlash arms propose one token more than they were trained
to emit, and made their acceptance length incomparable with DSpark's measured
at the same block size -- the exact comparison this study is built on.
"""

import pytest
from specdec_bench.models.vllm import DEFAULT_BLOCK_SIZE, _draft_tokens


@pytest.mark.parametrize(
    ("algorithm", "block_size", "expected"),
    [
        ("DFLASH", 8, 7),
        ("DFLASH2", 8, 7),
        ("DSPARK", 8, 8),
        ("DFLASH", 16, 15),
        ("DFLASH2", 16, 15),
        ("DSPARK", 16, 16),
    ],
)
def test_block_size_converts_to_the_family_specific_horizon(algorithm, block_size, expected):
    assert _draft_tokens(algorithm, {"speculative_num_draft_tokens": block_size}) == expected


@pytest.mark.parametrize("algorithm", ["DFLASH", "DFLASH2", "DSPARK"])
def test_unset_block_size_falls_back_rather_than_passing_none(algorithm):
    """``run.py`` always sends the key, carrying None when --block_size is unset.

    ``dict.get(key, default)`` returns None for a present-but-None key, so the
    fallback has to treat None as absent or vLLM receives a null horizon.
    """
    for kwargs in ({}, {"speculative_num_draft_tokens": None}):
        tokens = _draft_tokens(algorithm, kwargs)
        assert tokens == DEFAULT_BLOCK_SIZE + (0 if algorithm == "DSPARK" else -1)


@pytest.mark.parametrize("algorithm", ["DFLASH", "DFLASH2"])
def test_block_size_of_one_is_rejected_rather_than_proposing_nothing(algorithm):
    with pytest.raises(ValueError, match="would propose 0 tokens"):
        _draft_tokens(algorithm, {"speculative_num_draft_tokens": 1})
