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

"""Guard the speculative-decoding recipes against unknown ``training:`` keys.

``TrainingArguments`` in the recipe schema sets ``extra="allow"`` so HF trainer
fields pass straight through, which means a key that transformers does not accept
survives recipe validation and only raises inside
``HfTrainingArguments(**recipe.training.model_dump())`` -- after the allocation is
up and every node has staged. ``warmup_ratio`` did exactly that when transformers 5
removed it: the recipes kept loading and multi-node runs died minutes in.
"""

import dataclasses
from importlib.resources import files
from pathlib import Path

import transformers
import yaml

from modelopt.torch.speculative.plugins.hf_training_args import (
    TrainingArguments as SpecTrainingArgs,
)

SPECULATIVE_DIR = Path(str(files("modelopt_recipes"))) / "general" / "speculative_decoding"


def _accepted_training_keys() -> set[str]:
    """Return every key ``HfTrainingArguments`` can be constructed with."""
    hf_fields = {field.name for field in dataclasses.fields(transformers.TrainingArguments)}
    return hf_fields | set(SpecTrainingArgs.model_fields)


def test_speculative_recipe_training_keys_are_accepted_by_transformers():
    accepted = _accepted_training_keys()
    offenders: dict[str, list[str]] = {}
    for recipe in sorted(SPECULATIVE_DIR.glob("*.yaml")):
        training = (yaml.safe_load(recipe.read_text(encoding="utf-8")) or {}).get("training") or {}
        unknown = sorted(key for key in training if key not in accepted)
        if unknown:
            offenders[recipe.name] = unknown
    assert not offenders, (
        f"Speculative-decoding recipes set training keys that "
        f"transformers {transformers.__version__} will reject: {offenders}. "
        "These survive recipe validation because the schema allows extra keys, so they "
        "only fail once the trainer builds HfTrainingArguments on the cluster. "
        "Note transformers>=5 removed warmup_ratio; warmup_steps below 1.0 is the ratio."
    )
