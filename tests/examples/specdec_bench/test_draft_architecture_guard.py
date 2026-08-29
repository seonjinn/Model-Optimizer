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

"""The guard that stops a DFlash2 run from silently benchmarking DFlash.

vLLM serves DFlash2 under ``method="dflash"`` and picks the V2 speculator from
the draft checkpoint's ``architectures`` instead (vllm#52816). A DFLASH2 run
pointed at a plain DFlash drafter therefore starts cleanly, runs the wrong
speculator, and reports a number that reads as a valid DFlash2 measurement.
Only the checkpoint can tell the two apart, so only a pre-flight check can.
"""

import json

import pytest
from specdec_bench.models.vllm import _assert_draft_architecture

_ARCH = {
    "DFLASH": "DFlashDraftModel",
    "DFLASH2": "DFlash2DraftModel",
    "DSPARK": "Qwen3DSparkModel",
}


def _drafter(tmp_path, architectures):
    d = tmp_path / "draft"
    d.mkdir()
    (d / "config.json").write_text(json.dumps({"architectures": architectures}))
    return str(d)


@pytest.mark.parametrize("algorithm", sorted(_ARCH))
def test_matching_architecture_passes(tmp_path, algorithm):
    _assert_draft_architecture(_drafter(tmp_path, [_ARCH[algorithm]]), _ARCH[algorithm], algorithm)


def test_dflash_checkpoint_is_rejected_for_a_dflash2_run(tmp_path):
    """The case the guard exists for: both are served as method 'dflash'."""
    path = _drafter(tmp_path, ["DFlashDraftModel"])
    with pytest.raises(ValueError, match="DFlash2DraftModel"):
        _assert_draft_architecture(path, "DFlash2DraftModel", "DFLASH2")


def test_dflash2_checkpoint_is_rejected_for_a_dflash_run(tmp_path):
    path = _drafter(tmp_path, ["DFlash2DraftModel"])
    with pytest.raises(ValueError, match="DFlashDraftModel"):
        _assert_draft_architecture(path, "DFlashDraftModel", "DFLASH")


def test_missing_draft_dir_is_rejected():
    with pytest.raises(ValueError, match="requires --draft_model_dir"):
        _assert_draft_architecture(None, "DFlashDraftModel", "DFLASH")


def test_uninspectable_drafter_defers_to_vllm(tmp_path):
    """A repo id or a layout with no local config.json must not be blocked."""
    _assert_draft_architecture("z-lab/Qwen3.8-27B-DFlash2", "DFlash2DraftModel", "DFLASH2")
    _assert_draft_architecture(str(tmp_path), "DFlash2DraftModel", "DFLASH2")


def test_architectures_absent_from_config_is_rejected(tmp_path):
    d = tmp_path / "draft"
    d.mkdir()
    (d / "config.json").write_text(json.dumps({"model_type": "qwen3"}))
    with pytest.raises(ValueError, match="declares \\[\\]"):
        _assert_draft_architecture(str(d), "DFlash2DraftModel", "DFLASH2")
