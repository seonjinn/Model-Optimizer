# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Behavioral contracts for launcher service utilities."""

import os
import subprocess
from pathlib import Path

_SERVICE_UTILS = Path(__file__).resolve().parents[1] / "common" / "service_utils.sh"


def test_sourcing_service_utils_preserves_modelopt_version_source(tmp_path: Path) -> None:
    """Sourcing shared helpers must not rewrite an importable ModelOpt package."""
    package_dir = tmp_path / "modules" / "Model-Optimizer" / "modelopt"
    package_dir.mkdir(parents=True)
    init_file = package_dir / "__init__.py"
    original = b'__version__ = "9.9.9"\n'
    init_file.write_bytes(original)

    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    nvidia_smi = bin_dir / "nvidia-smi"
    nvidia_smi.write_text("#!/bin/sh\nprintf '1\\n'\n")
    nvidia_smi.chmod(0o755)

    env = {
        **os.environ,
        "PATH": f"{bin_dir}:{os.environ['PATH']}",
        "PMIX_RANK": "0",
        "PMIX_LOCAL_RANK": "0",
    }
    result = subprocess.run(
        [
            "bash",
            "-c",
            'source "$1" && source "$1" && '
            'PYTHONPATH="$PWD/modules/Model-Optimizer" python3 -c '
            "'import modelopt; assert modelopt.__version__ == \"9.9.9\"'",
            "bash",
            str(_SERVICE_UTILS),
        ],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert init_file.read_bytes() == original
