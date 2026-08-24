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

"""Runtime verification for fail-closed DFlash2 serving and evaluation."""

from __future__ import annotations

import re
import shutil
import subprocess  # nosec B404 - Git is invoked with a fixed executable and argv, never a shell.
from pathlib import Path

__all__ = ["verify_vllm_checkout"]

_FULL_SHA = re.compile(r"^[0-9a-f]{40}$")
_GIT = shutil.which("git")


def _git(repo: Path, *arguments: str) -> subprocess.CompletedProcess[str]:
    if _GIT is None:
        raise ValueError("Git is required to verify the vLLM runtime")
    return subprocess.run(  # nosec B603 - arguments are passed directly without a shell.
        [_GIT, "-C", str(repo), *arguments],
        check=False,
        capture_output=True,
        text=True,
        timeout=10,
    )


def verify_vllm_checkout(package_path: Path, expected_commit: str, required_ancestor: str) -> str:
    """Verify the exact vLLM checkout and its required DFlash2 ancestor."""
    if not _FULL_SHA.fullmatch(expected_commit) or not _FULL_SHA.fullmatch(required_ancestor):
        raise ValueError("vLLM commits must be exact lowercase 40-character SHAs")

    package = package_path.resolve(strict=True)
    repo = next((path for path in (package, *package.parents) if (path / ".git").exists()), None)
    if repo is None:
        raise ValueError("vLLM package must come from a Git checkout")

    top_level = _git(repo, "rev-parse", "--show-toplevel")
    if top_level.returncode != 0:
        raise ValueError("vLLM package must come from a valid Git checkout")
    root = Path(top_level.stdout.strip()).resolve(strict=True)
    if not package.is_relative_to(root):
        raise ValueError("vLLM package is outside its reported Git checkout")

    head = _git(root, "rev-parse", "HEAD")
    actual_commit = head.stdout.strip()
    if head.returncode != 0 or actual_commit != expected_commit:
        raise ValueError(
            f"vLLM runtime does not match expected commit {expected_commit}: {actual_commit}"
        )

    ancestry = _git(root, "merge-base", "--is-ancestor", required_ancestor, actual_commit)
    if ancestry.returncode != 0:
        raise ValueError(
            f"vLLM runtime does not contain required DFlash2 commit {required_ancestor}"
        )
    return actual_commit
