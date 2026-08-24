# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Install the source-order sampler required by the A-repair comparison."""

# pyright: reportMissingImports=false

from __future__ import annotations

import os
from typing import Any

if os.environ.get("QWEN4B_A_SEQUENTIAL_SAMPLER") == "1":
    from torch.utils.data import SequentialSampler
    from transformers import Trainer

    def _a_sequential_sampler(self: Trainer, train_dataset: Any | None = None) -> Any:
        dataset = self.train_dataset if train_dataset is None else train_dataset
        if dataset is None:
            return None
        return SequentialSampler(dataset)

    Trainer._get_train_sampler = _a_sequential_sampler
