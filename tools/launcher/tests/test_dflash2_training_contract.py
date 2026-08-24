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

"""Fail-closed contracts for Q30/Q235 Nemotron DFlash2 training."""

from __future__ import annotations

import errno
import hashlib
import json
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import common.specdec.dflash2_runtime_contract as runtime_contract
import pytest
from common.specdec.build_dflash2_nemotron_manifest import build_dflash2_nemotron_manifest
from common.specdec.dflash2_runtime_contract import (
    artifact_tree_sha256,
    validate_artifact_receipt,
    validate_dflash2_source_checkout,
    verify_vllm_runtime,
    write_artifact_receipt,
    write_vllm_runtime_receipt,
)
from common.specdec.dflash2_target_contract import (
    dflash2_target_spec,
    validate_dflash2_target_snapshot,
)
from common.specdec.drafter_job_manifest import (
    DrafterExperiment,
    PinnedPaths,
    SlurmSettings,
    TargetTopology,
    load_manifest,
    validate_source_path_for_cluster,
    write_manifest,
)

_SHA256 = "d" * 64
_SOURCE_SHA = "c" * 40
_FEATURE_PATHS = (
    "modelopt/torch/export/plugins/hf_spec_export.py",
    "modelopt/torch/speculative/config.py",
    "modelopt/torch/speculative/dflash/conversion.py",
    "modelopt/torch/speculative/plugins/__init__.py",
    "modelopt/torch/speculative/plugins/hf_dflash.py",
    "modelopt/torch/speculative/plugins/hf_dflash2.py",
    "modelopt/torch/speculative/plugins/modeling_dflash.py",
    "modelopt/torch/speculative/plugins/modeling_dflash2.py",
    "modelopt_recipes/general/speculative_decoding/dflash2.yaml",
)


def _git(repo: Path, *arguments: str) -> str:
    completed = subprocess.run(
        ["git", "-c", "commit.gpgsign=false", "-C", str(repo), *arguments],
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def _vllm_checkout(tmp_path: Path) -> tuple[Path, str, str]:
    repo = tmp_path / "vllm"
    repo.mkdir()
    _git(repo, "init", "-q")
    _git(repo, "config", "user.name", "Test User")
    _git(repo, "config", "user.email", "test@example.com")
    package = repo / "vllm"
    package.mkdir()
    (package / "__init__.py").write_text('__version__ = "test"\n')
    profile_speculator = package / "v1/worker/gpu/spec_decode/dflash/speculator.py"
    profile_speculator.parent.mkdir(parents=True)
    profile_speculator.write_text(
        "".join(f"# pinned b389 line {line}\n" for line in range(1, 364))
        + "        if dummy_run and skip_attn_for_dummy_run:\n"
        "            # Memory profiling path: block_tables / kv_cache_config are not initialized.\n"
        "            # Since DFlash needs to build its own attention metadata, we must skip the\n"
        "            # preparation in this path and run a minimal forward pass.\n"
        "            self.model.precompute_and_store_context_kv(\n"
        "                self.hidden_states[:num_target_tokens],\n"
        "                self.context_positions[:num_target_tokens],\n"
        "            )\n"
        "            # DFlash processes all speculative tokens in one forward pass,\n"
        "            # so the real token count is num_query_tokens.\n"
        "            self._prepare_eplb_forward(num_query_tokens)\n"
        "            self._generate_draft(\n"
        "                num_reqs,\n"
        "                num_query_tokens,\n"
        "                attn_metadata=None,\n"
        "                slot_mappings=None,\n"
        "                num_tokens_across_dp=num_tokens_across_dp,\n"
        "                cudagraph_runtime_mode=CUDAGraphMode.NONE,\n"
        "            )\n"
        "            return self.draft_tokens[:num_reqs]\n\n"
        "        # The query slot mapping is written into the shared BlockTables slot_mappings.\n"
        "        # That buffer's address is what the captured CUDA graph reads from at replay.\n"
    )
    model_runner = package / "v1/worker/gpu/model_runner.py"
    model_runner.write_text(
        "".join(f"# pinned b389 line {line}\n" for line in range(1, 667))
        + "            num_tokens = max(num_tokens, self.decode_query_len)\n"
        "            num_reqs = num_tokens // self.decode_query_len\n"
        "            assert num_tokens % self.decode_query_len == 0\n"
        "        # Distribute the remainder evenly so no dummy request exceeds\n"
        "        # ceil(num_tokens / num_reqs) <= max_model_len tokens.\n"
        "        num_tokens_per_request = [\n"
    )
    recipe = repo / "cmake/external_projects/flashmla.cmake"
    recipe.parent.mkdir(parents=True)
    recipe.write_text("GIT_TAG test-flashmla\n")
    _git(repo, "add", "vllm", "cmake/external_projects/flashmla.cmake")
    _git(repo, "commit", "-q", "-m", "required DFlash2 runtime")
    required = _git(repo, "rev-parse", "HEAD")
    (package / "runtime.py").write_text("SUPPORTED = True\n")
    _git(repo, "add", "vllm/runtime.py")
    _git(repo, "commit", "-q", "-m", "verified descendant")
    return package, required, _git(repo, "rev-parse", "HEAD")


def _flashmla_checkout(tmp_path: Path) -> tuple[Path, str]:
    repo = tmp_path / "flashmla"
    repo.mkdir()
    _git(repo, "init", "-q")
    _git(repo, "config", "user.name", "Test User")
    _git(repo, "config", "user.email", "test@example.com")
    package = repo / "flash_mla"
    package.mkdir()
    (package / "flash_mla_interface.py").write_text(
        "import torch\nflash_mla_cuda = torch.ops._flashmla_C\n"
    )
    _git(repo, "add", "flash_mla/flash_mla_interface.py")
    _git(repo, "commit", "-q", "-m", "flashmla source")
    return package, _git(repo, "rev-parse", "HEAD")


def _cutlass_checkout(tmp_path: Path) -> tuple[Path, str, Path]:
    repo = tmp_path / "cutlass"
    repo.mkdir()
    _git(repo, "init", "-q")
    _git(repo, "config", "user.name", "Test User")
    _git(repo, "config", "user.email", "test@example.com")
    header = repo / "include/cutlass/cutlass.h"
    header.parent.mkdir(parents=True)
    header.write_text("#pragma once\n")
    _git(repo, "add", ".")
    _git(repo, "commit", "-q", "-m", "pinned vLLM CUTLASS source")
    commit = _git(repo, "rev-parse", "HEAD")
    archive = tmp_path / "vllm-cutlass-source.tar"
    with archive.open("wb") as stream:
        subprocess.run(
            ["git", "-C", str(repo), "archive", "HEAD"],
            check=True,
            stdout=stream,
        )
    return repo, commit, archive


def _dflash2_source_checkout(tmp_path: Path) -> tuple[Path, str, str]:
    repo = tmp_path / "modelopt"
    repo.mkdir()
    _git(repo, "init", "-q")
    _git(repo, "config", "user.name", "Test User")
    _git(repo, "config", "user.email", "test@example.com")
    for relative in _FEATURE_PATHS:
        path = repo / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"feature: {relative}\n")
    _git(repo, "add", ".")
    _git(repo, "commit", "-q", "-m", "DFlash2 feature base")
    feature_base = _git(repo, "rev-parse", "HEAD")
    launcher = repo / "tools/launcher/common/specdec/launcher.py"
    launcher.parent.mkdir(parents=True)
    launcher.write_text("DESCENDANT = True\n")
    _git(repo, "add", ".")
    _git(repo, "commit", "-q", "-m", "launcher-only descendant")
    return repo, feature_base, _git(repo, "rev-parse", "HEAD")


def test_feature_base_accepts_clean_launcher_descendant_and_rejects_feature_drift(
    tmp_path: Path,
) -> None:
    """A stable profile base permits launcher fixes but freezes every DFlash2 feature blob."""
    repo, feature_base, launcher_head = _dflash2_source_checkout(tmp_path)
    assert len(validate_dflash2_source_checkout(repo, launcher_head, feature_base)) == 64

    feature = repo / "modelopt/torch/speculative/plugins/modeling_dflash2.py"
    feature.write_text("MUTATED = True\n")
    _git(repo, "add", ".")
    _git(repo, "commit", "-q", "-m", "unsafe feature mutation")
    mutated_head = _git(repo, "rev-parse", "HEAD")
    with pytest.raises(ValueError, match="feature files differ"):
        validate_dflash2_source_checkout(repo, mutated_head, feature_base)
    with pytest.raises(ValueError, match="not an ancestor"):
        validate_dflash2_source_checkout(repo, mutated_head, "a" * 40)


def test_feature_base_rejects_a_dirty_launcher_checkout(tmp_path: Path) -> None:
    """Uncommitted launcher changes cannot bypass the exact manifest HEAD."""
    repo, feature_base, launcher_head = _dflash2_source_checkout(tmp_path)
    (repo / "untracked.py").write_text("DIRTY = True\n")
    with pytest.raises(ValueError, match="must be clean"):
        validate_dflash2_source_checkout(repo, launcher_head, feature_base)


def test_lyris_project_source_prefix_is_exact_and_cluster_scoped() -> None:
    """Only Lyris may consume the exact user-owned project source prefix."""
    source = "/project/coreai_dlalgo_llm/users/sna/ModelOpt-dflash2"
    validate_source_path_for_cluster(source, "lyris")
    for cluster in ("oci-hsg", "ptyche"):
        with pytest.raises(ValueError, match="Lyris project source"):
            validate_source_path_for_cluster(source, cluster)
    for unsafe in (
        "/project/coreai_dlalgo_llm/users/other/ModelOpt",
        "/project/other/users/sna/ModelOpt",
        "/project/coreai_dlalgo_llm/users/sna/../other/ModelOpt",
    ):
        with pytest.raises(ValueError):
            PinnedPaths(
                **{**_template_experiment("q30-base").paths.__dict__, "source_path": unsafe}
            )


def test_lyris_project_source_is_a_valid_pinned_path() -> None:
    """The quota-safe clean Lyris checkout remains a canonical pinned source path."""
    source = "/project/coreai_dlalgo_llm/users/sna/ModelOpt-dflash2"
    paths = PinnedPaths(
        **{**_template_experiment("q30-base").paths.__dict__, "source_path": source}
    )
    assert paths.source_path == source


def test_vllm_runtime_receipt_survives_archive_staging_without_git(tmp_path: Path) -> None:
    """A staged venv proves PR 52816 from immutable package bytes, not a copied .git tree."""
    package, required, expected = _vllm_checkout(tmp_path)
    receipt = tmp_path / "vllm-runtime.json"
    receipt_sha = write_vllm_runtime_receipt(receipt, package, expected, required)
    staged = tmp_path / "staged/vllm"
    staged.parent.mkdir()
    __import__("shutil").copytree(package, staged)

    assert verify_vllm_runtime(staged, receipt, receipt_sha, expected, required) == expected
    (staged / "injected.py").write_text("MALICIOUS = True\n")
    with pytest.raises(ValueError, match="runtime file set"):
        verify_vllm_runtime(staged, receipt, receipt_sha, expected, required)


def test_vllm_receipt_binds_tracked_source_and_compiled_runtime_extras(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A wheel may add ARM64 extensions, but every tracked PR source byte must remain exact."""
    package, required, expected = _vllm_checkout(tmp_path)
    flashmla, flashmla_commit = _flashmla_checkout(tmp_path)
    cutlass, cutlass_commit, cutlass_archive = _cutlass_checkout(tmp_path)
    runtime = tmp_path / "runtime/vllm"
    runtime.parent.mkdir()
    __import__("shutil").copytree(package, runtime)
    (runtime / "_flashmla_C.abi3.so").write_bytes(b"compiled-from-flashmla-a8f-core")
    (runtime / "_flashmla_extension_C.abi3.so").write_bytes(
        b"compiled-from-flashmla-a8f-extension"
    )
    generated = runtime / "third_party/flashmla/flash_mla_interface.py"
    generated.parent.mkdir(parents=True)
    generated.write_text(
        "import torch\nimport vllm._flashmla_C\n"
        "flash_mla_cuda = torch.ops._flashmla_C\n"
    )
    base_runtime = tmp_path / "base-runtime.tar.zst"
    base_runtime.write_bytes(b"immutable-base-runtime")
    image = tmp_path / "vllm.sqsh"
    image.write_bytes(b"immutable-container")
    builder = tmp_path / "build_dflash2_runtime.sbatch"
    builder.write_text("#!/usr/bin/env bash\ncmake --build exact-inputs\n")
    configure_log = tmp_path / "dflash2-flashmla-cmake-configure.log"
    focused_cmake = tmp_path / "focused-CMakeLists.txt"
    focused_cmake.write_text("include(cmake/external_projects/flashmla.cmake)\n")
    configure_log.write_text(
        "cmake_path=/scratch/job/runtime/bin/cmake\n"
        "cmake version 3.31.6\n"
        "ninja_path=/scratch/job/runtime/bin/ninja\n"
        "1.13.0\n"
        "-- CUDA target architectures: 10.0a\n"
        "-- FlashMLA CUDA architectures: 10.0f\n"
        "-- The VLLM_CUTLASS_SRC_DIR is set, using /scratch/vllm-cutlass-source\n"
    )
    build_manifest = tmp_path / "dflash2-flashmla-build-manifest.json"
    runtime_contract.write_flashmla_build_manifest(
        build_manifest,
        runtime,
        package,
        flashmla,
        base_runtime,
        image,
        builder,
        configure_log,
        expected,
        flashmla_commit,
        vllm_cutlass_path=cutlass,
        vllm_cutlass_archive_path=cutlass_archive,
        vllm_cutlass_commit=cutlass_commit,
        focused_cmake_path=focused_cmake,
    )
    receipt = tmp_path / "vllm-runtime-v4.json"
    receipt_sha = write_vllm_runtime_receipt(
        receipt,
        package,
        expected,
        required,
        runtime_package_path=runtime,
        flashmla_package_path=flashmla,
        flashmla_expected_commit=flashmla_commit,
        flashmla_build_manifest_path=build_manifest,
    )

    assert (
        verify_vllm_runtime(
            runtime,
            receipt,
            receipt_sha,
            expected,
            required,
            expected_flashmla_commit=flashmla_commit,
        )
        == expected
    )
    body = json.loads(receipt.read_text())
    assert body["schema_version"] == 4
    assert body["flashmla_commit"] == flashmla_commit
    assert body["flashmla_generated_interface"]["path"] == (
        "third_party/flashmla/flash_mla_interface.py"
    )
    assert {item["path"] for item in body["flashmla_extension_binaries"]} == {
        "_flashmla_C.abi3.so",
        "_flashmla_extension_C.abi3.so",
    }
    patched_root = tmp_path / "patched-runtime"
    patched_runtime = patched_root / "vllm"
    __import__("shutil").copytree(runtime, patched_runtime)
    patch_source = (
        Path(__file__).resolve().parents[1]
        / "common/specdec/patches/vllm-b389-dflash-profile-capacity.patch"
    )
    patch_path = patched_root / "dflash2-vllm-profile-capacity-patch.diff"
    base_path = patched_root / "dflash2-vllm-profile-capacity-base.py"
    __import__("shutil").copy2(patch_source, patch_path)
    __import__("shutil").copy2(
        package / "v1/worker/gpu/model_runner.py",
        base_path,
    )
    subprocess.run(
        ["git", "-C", str(patched_root), "apply", str(patch_path)],
        check=True,
    )
    __import__("shutil").copy2(build_manifest, patched_root / build_manifest.name)
    __import__("shutil").copy2(configure_log, patched_root / configure_log.name)
    patched_receipt = patched_root / "dflash2-vllm-runtime-receipt.json"
    patched_receipt_sha = write_vllm_runtime_receipt(
        patched_receipt,
        package,
        expected,
        required,
        runtime_package_path=patched_runtime,
        flashmla_package_path=flashmla,
        flashmla_expected_commit=flashmla_commit,
        flashmla_build_manifest_path=patched_root / build_manifest.name,
        runtime_source_patch_path=patch_path,
        runtime_source_patch_base_path=base_path,
    )
    assert json.loads(patched_receipt.read_text())["schema_version"] == 5
    assert (
        verify_vllm_runtime(
            patched_runtime,
            patched_receipt,
            patched_receipt_sha,
            expected,
            required,
            expected_flashmla_commit=flashmla_commit,
        )
        == expected
    )
    monkeypatch.setattr(runtime_contract, "_GIT", None)
    assert (
        verify_vllm_runtime(
            patched_runtime,
            patched_receipt,
            patched_receipt_sha,
            expected,
            required,
            expected_flashmla_commit=flashmla_commit,
        )
        == expected
    )
    child_speculator = patched_runtime / "v1/worker/gpu/spec_decode/dflash/speculator.py"
    assert child_speculator.read_bytes() == (
        package / "v1/worker/gpu/spec_decode/dflash/speculator.py"
    ).read_bytes()
    patched_file = patched_runtime / "v1/worker/gpu/model_runner.py"
    patched_file.write_text(patched_file.read_text() + "# forged\n")
    with pytest.raises(ValueError, match="profile patch output mismatch"):
        verify_vllm_runtime(
            patched_runtime,
            patched_receipt,
            patched_receipt_sha,
            expected,
            required,
            expected_flashmla_commit=flashmla_commit,
        )

    (runtime / "runtime.py").write_text("SUPPORTED = False\n")
    with pytest.raises(ValueError, match="runtime file bytes"):
        verify_vllm_runtime(
            runtime,
            receipt,
            receipt_sha,
            expected,
            required,
            expected_flashmla_commit=flashmla_commit,
        )
    (runtime / "runtime.py").write_text((package / "runtime.py").read_text())

    build_body = json.loads(build_manifest.read_text())
    build_body.pop("receipt_sha256")
    build_body.pop("focused_vllm_cmake")
    build_body["receipt_sha256"] = hashlib.sha256(
        json.dumps(build_body, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    build_manifest.write_text(
        json.dumps(build_body, sort_keys=True, separators=(",", ":")) + "\n"
    )
    body["flashmla_build_manifest"] = {
        "path": "dflash2-flashmla-build-manifest.json",
        "bytes": build_manifest.stat().st_size,
        "sha256": hashlib.sha256(build_manifest.read_bytes()).hexdigest(),
    }
    body.pop("receipt_sha256")
    body["receipt_sha256"] = hashlib.sha256(
        json.dumps(body, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    receipt.write_text(json.dumps(body, sort_keys=True, separators=(",", ":")) + "\n")
    forged_receipt_sha = hashlib.sha256(receipt.read_bytes()).hexdigest()
    with pytest.raises(ValueError, match="source-build manifest identity"):
        verify_vllm_runtime(
            runtime,
            receipt,
            forged_receipt_sha,
            expected,
            required,
            expected_flashmla_commit=flashmla_commit,
        )


def test_vllm_receipt_rejects_mutated_flashmla_build_recipe(tmp_path: Path) -> None:
    """A dirty CMake recipe cannot claim the pinned FlashMLA build provenance."""
    package, required, expected = _vllm_checkout(tmp_path)
    flashmla, flashmla_commit = _flashmla_checkout(tmp_path)
    runtime = tmp_path / "runtime/vllm"
    runtime.parent.mkdir()
    __import__("shutil").copytree(package, runtime)
    generated = runtime / "third_party/flashmla/flash_mla_interface.py"
    generated.parent.mkdir(parents=True)
    generated.write_text(
        "import torch\nimport vllm._flashmla_C\n"
        "flash_mla_cuda = torch.ops._flashmla_C\n"
    )
    (package.parent / "cmake/external_projects/flashmla.cmake").write_text(
        "GIT_TAG attacker-controlled\n"
    )

    with pytest.raises(ValueError, match="FlashMLA build recipe"):
        write_vllm_runtime_receipt(
            tmp_path / "vllm-runtime-v3.json",
            package,
            expected,
            required,
            runtime_package_path=runtime,
            flashmla_package_path=flashmla,
            flashmla_expected_commit=flashmla_commit,
        )


def test_flashmla_build_manifest_rejects_missing_extension_pair(tmp_path: Path) -> None:
    """A source-build receipt requires both exact vLLM FlashMLA extension outputs."""
    package, _, expected = _vllm_checkout(tmp_path)
    flashmla, flashmla_commit = _flashmla_checkout(tmp_path)
    cutlass, cutlass_commit, cutlass_archive = _cutlass_checkout(tmp_path)
    runtime = tmp_path / "runtime/vllm"
    runtime.mkdir(parents=True)
    (runtime / "_flashmla_C.abi3.so").write_bytes(b"only-one-extension")
    focused_cmake = tmp_path / "focused-CMakeLists.txt"
    focused_cmake.write_text("include(cmake/external_projects/flashmla.cmake)\n")
    inputs = []
    for name in (
        "base-runtime.tar.zst",
        "vllm.sqsh",
        "builder.sbatch",
        "flashmla-cmake-configure.log",
    ):
        item = tmp_path / name
        item.write_text(
            (
                "cmake_path=/scratch/job/runtime/bin/cmake\n"
                "cmake version 3.31.6\n"
                "ninja_path=/scratch/job/runtime/bin/ninja\n"
                "1.13.0\n"
                "-- CUDA target architectures: 10.0a\n"
                "-- FlashMLA CUDA architectures: 10.0f\n"
                "-- The VLLM_CUTLASS_SRC_DIR is set, using /scratch/vllm-cutlass-source\n"
            )
            if name.endswith(".log")
            else name
        )
        inputs.append(item)

    with pytest.raises(ValueError, match="exact FlashMLA extension pair"):
        runtime_contract.write_flashmla_build_manifest(
            tmp_path / "build-manifest.json",
            runtime,
            package,
            flashmla,
            *inputs,
            expected,
            flashmla_commit,
            vllm_cutlass_path=cutlass,
            vllm_cutlass_archive_path=cutlass_archive,
            vllm_cutlass_commit=cutlass_commit,
            focused_cmake_path=focused_cmake,
        )


def test_flashmla_configure_preflight_binds_isolated_toolchain(tmp_path: Path) -> None:
    """Configure-only evidence proves exact job-local tools and Blackwell architecture gates."""
    package, _, expected = _vllm_checkout(tmp_path)
    flashmla, flashmla_commit = _flashmla_checkout(tmp_path)
    cutlass, cutlass_commit, cutlass_archive = _cutlass_checkout(tmp_path)
    inputs = []
    for name in ("base-runtime.tar.zst", "vllm.sqsh", "builder.sbatch"):
        item = tmp_path / name
        item.write_bytes(name.encode())
        inputs.append(item)
    configure_log = tmp_path / "dflash2-flashmla-cmake-configure.log"
    focused_cmake = tmp_path / "focused-CMakeLists.txt"
    focused_cmake.write_text("include(cmake/external_projects/flashmla.cmake)\n")
    configure_log.write_text(
        "cmake_path=/scratch/job/runtime/bin/cmake\n"
        "cmake version 3.31.6\n"
        "ninja_path=/scratch/job/runtime/bin/ninja\n"
        "1.13.0.git.kitware.jobserver-pipe-1\n"
        "-- CUDA target architectures: 10.0a\n"
        "-- FlashMLA CUDA architectures: 10.0f\n"
        "-- The VLLM_CUTLASS_SRC_DIR is set, using /scratch/vllm-cutlass-source\n"
    )
    receipt = tmp_path / "configure-preflight.json"

    receipt_sha = runtime_contract.write_flashmla_configure_preflight(
        receipt,
        package,
        flashmla,
        *inputs,
        configure_log,
        expected,
        flashmla_commit,
        "12345",
        vllm_cutlass_path=cutlass,
        vllm_cutlass_archive_path=cutlass_archive,
        vllm_cutlass_commit=cutlass_commit,
        focused_cmake_path=focused_cmake,
    )

    body = json.loads(receipt.read_text())
    assert hashlib.sha256(receipt.read_bytes()).hexdigest() == receipt_sha
    assert body["producer"] == "dflash2-flashmla-configure-preflight-v1"
    assert body["slurm_job_id"] == "12345"
    assert body["configure_evidence"]["ninja_version"] == (
        "1.13.0.git.kitware.jobserver-pipe-1"
    )
    assert body["vllm_cutlass"]["commit"] == cutlass_commit
    assert body["vllm_cutlass"]["archive"]["sha256"] == hashlib.sha256(
        cutlass_archive.read_bytes()
    ).hexdigest()
    assert body["focused_vllm_cmake"]["sha256"] == hashlib.sha256(
        focused_cmake.read_bytes()
    ).hexdigest()
    with pytest.raises(FileExistsError):
        runtime_contract.write_flashmla_configure_preflight(
            receipt,
            package,
            flashmla,
            *inputs,
            configure_log,
            expected,
            flashmla_commit,
            "12345",
            vllm_cutlass_path=cutlass,
            vllm_cutlass_archive_path=cutlass_archive,
            vllm_cutlass_commit=cutlass_commit,
            focused_cmake_path=focused_cmake,
        )

    (cutlass / "include/cutlass/cutlass.h").write_text("mutated\n")
    with pytest.raises(ValueError, match="CUTLASS checkout must be clean"):
        runtime_contract.write_flashmla_configure_preflight(
            tmp_path / "forged-preflight.json",
            package,
            flashmla,
            *inputs,
            configure_log,
            expected,
            flashmla_commit,
            "12345",
            vllm_cutlass_path=cutlass,
            vllm_cutlass_archive_path=cutlass_archive,
            vllm_cutlass_commit=cutlass_commit,
            focused_cmake_path=focused_cmake,
        )


def test_runtime_contract_cli_builds_no_replace_receipts(tmp_path: Path) -> None:
    """Artifact staging has one auditable CLI rather than ad-hoc Python snippets."""
    package, required, expected = _vllm_checkout(tmp_path)
    receipt = tmp_path / "vllm-runtime.json"
    script = Path(__file__).resolve().parents[1] / "common/specdec/dflash2_runtime_contract.py"
    completed = subprocess.run(
        [
            sys.executable,
            str(script),
            "vllm-receipt",
            "--package",
            str(package),
            "--output",
            str(receipt),
            "--expected-commit",
            expected,
            "--required-ancestor",
            required,
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    assert (
        completed.stdout.strip() == __import__("hashlib").sha256(receipt.read_bytes()).hexdigest()
    )
    with pytest.raises(subprocess.CalledProcessError):
        subprocess.run(
            [
                sys.executable,
                str(script),
                "vllm-receipt",
                "--package",
                str(package),
                "--output",
                str(receipt),
                "--expected-commit",
                expected,
                "--required-ancestor",
                required,
            ],
            check=True,
            capture_output=True,
            text=True,
        )


def test_vllm_runtime_rejects_a_manifest_head_mismatch(tmp_path: Path) -> None:
    """A different checkout cannot reuse another image's DFlash2 attestation."""
    package, required, expected = _vllm_checkout(tmp_path)
    receipt = tmp_path / "vllm-runtime.json"
    receipt_sha = write_vllm_runtime_receipt(receipt, package, expected, required)

    with pytest.raises(ValueError, match="receipt identity"):
        verify_vllm_runtime(package, receipt, receipt_sha, "a" * 40, required)


def test_vllm_runtime_rejects_a_missing_required_ancestor(tmp_path: Path) -> None:
    """An exact runtime head without the merged DFlash2 serving change is unsafe."""
    package, _required, expected = _vllm_checkout(tmp_path)

    with pytest.raises(ValueError, match="required DFlash2 commit"):
        write_vllm_runtime_receipt(tmp_path / "receipt.json", package, expected, "b" * 40)


def test_vllm_runtime_receipt_rejects_dirty_checkout_bytes(tmp_path: Path) -> None:
    """A commit claim cannot attest modified or untracked package bytes."""
    package, required, expected = _vllm_checkout(tmp_path)
    (package / "runtime.py").write_text("SUPPORTED = False\n")

    with pytest.raises(ValueError, match="clean tracked checkout"):
        write_vllm_runtime_receipt(tmp_path / "receipt.json", package, expected, required)


def test_vllm_runtime_rejects_a_non_attested_install(tmp_path: Path) -> None:
    """Version strings alone cannot prove that PR 52816 is present."""
    package = tmp_path / "site-packages" / "vllm"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text('__version__ = "0.20.0"\n')

    with pytest.raises(ValueError, match="receipt is missing"):
        verify_vllm_runtime(package, tmp_path / "missing.json", "c" * 64, "a" * 40, "b" * 40)


def _profile(tmp_path: Path, source_sha: str = _SOURCE_SHA) -> Path:
    path = tmp_path / "profile.yaml"
    path.write_text(
        "\n".join(
            (
                "name: oci-hsg",
                f"modelopt_commit: {source_sha}",
                "ssh_host: host",
                "account: nemotron_n3_post",
                "partition: batch",
                "fallback_partition: null",
                "durable_root: /lustre/results",
                "scratch_candidates: [/raid/scratch]",
                "training_nodes: 16",
                "training_segment: 16",
                "evaluation_nodes: 1",
                "evaluation_segment: 1",
                "gpus_per_node: 4",
                "explicit_gpu_flag: true",
                'walltime: "03:55:00"',
                "",
            )
        )
    )
    return path


def _template_experiment(target: str) -> DrafterExperiment:
    is_q30 = target.startswith("q30-")
    kind = "qwen3-30b-a3b" if is_q30 else "qwen3-235b-a22b"
    nodes = 2 if is_q30 else 4
    return DrafterExperiment(
        target=target,
        dataset="nemo-direct",
        method="dflash",
        block_size=8,
        cumulative_max_steps=(4166, 14500, 25391),
        run_name=f"{target}-nemo-dflash-b8",
        topology=TargetTopology.for_kind(kind),
        paths=PinnedPaths(
            source_path="/home/user/ModelOpt",
            source_sha="a" * 40,
            image_path="/lustre/images/vllm.sqsh",
            image_sha256="b" * 64,
            runtime_archive_path="/lustre/runtimes/modelopt.tar.zst",
            runtime_archive_sha256="e" * 64,
            target_path=f"/lustre/models/{target}",
            dataset_path="/lustre/datasets/nemotron-1.3m.parquet",
            output_root=f"/lustre/results/{target}-nemo-dflash-b8",
            target_sha256="1" * 64,
            target_receipt_path=f"/lustre/models/{target}.receipt.json",
            target_receipt_sha256="2" * 64,
            dataset_sha256="3" * 64,
            dataset_receipt_path="/lustre/datasets/nemotron-1.3m.receipt.json",
            dataset_receipt_sha256="4" * 64,
        ),
        slurm=SlurmSettings(
            account="nemotron_n3_post",
            partition="batch",
            nodes=nodes,
            gpus_per_node=4,
            segment=nodes,
        ),
    )


def test_builder_emits_exact_q30_q235_base_nemotron_dflash2_matrix(tmp_path: Path) -> None:
    """One shared builder pins both Base targets, exact dims, B8/K7, and runtime proof."""
    template = tmp_path / "template.json"
    output = tmp_path / "dflash2.json"
    write_manifest(
        template,
        (_template_experiment("q30-base"), _template_experiment("q235-base")),
    )

    experiments = build_dflash2_nemotron_manifest(
        template,
        output,
        "/home/user/ModelOpt-dflash2",
        _SOURCE_SHA,
        _SHA256,
        "f" * 40,
        "e" * 64,
        cluster_profile=_profile(tmp_path),
    )

    assert load_manifest(output) == experiments
    assert {(item.target, item.dataset, item.method, item.block_size) for item in experiments} == {
        ("q30-base", "nemo-direct", "dflash2", 8),
        ("q235-base", "nemo-direct", "dflash2", 8),
    }
    expected = {
        "q30-base": (32, 4, 128, 6144, 4),
        "q235-base": (64, 4, 128, 12288, 8),
    }
    for experiment in experiments:
        topology = experiment.topology
        assert (
            topology.num_attention_heads,
            topology.num_key_value_heads,
            topology.head_dim,
            topology.intermediate_size,
            topology.gradient_accumulation_steps,
        ) == expected[experiment.target]
        assert experiment.slurm.nodes == 16
        assert experiment.num_speculative_tokens == 7
        assert experiment.dflash2 is not None
        assert experiment.dflash2.vllm_expected_commit == "f" * 40
        assert experiment.paths.source_sha == _SOURCE_SHA
        assert experiment.paths.image_sha256 == _SHA256
        assert experiment.cumulative_max_steps == (20, 4166, 14500, 25391)


def test_builder_emits_exact_q30_q235_thinking_nemotron_dflash2_matrix(
    tmp_path: Path,
) -> None:
    """Thinking targets keep the family topology and get isolated clear identities."""
    template = tmp_path / "template.json"
    output = tmp_path / "thinking-dflash2.json"
    write_manifest(
        template,
        (_template_experiment("q30-thinking"), _template_experiment("q235-thinking")),
    )

    experiments = build_dflash2_nemotron_manifest(
        template,
        output,
        "/home/user/ModelOpt-dflash2",
        _SOURCE_SHA,
        _SHA256,
        "f" * 40,
        "e" * 64,
        cluster_profile=_profile(tmp_path),
        target_variant="thinking",
    )

    assert load_manifest(output) == experiments
    assert {item.target for item in experiments} == {"q30-thinking", "q235-thinking"}
    expected = {
        "q30-thinking": ((2, 13, 24, 35, 46, 48), 2, 4, 4, 32, 4, 128, 6144),
        "q235-thinking": ((2, 25, 47, 69, 92, 94), 4, 2, 8, 64, 4, 128, 12288),
    }
    expected_names = {
        "q30-thinking": "dfl2-q30t-nemo1p3m-b8k7-n16",
        "q235-thinking": "dfl2-q235t-nemo1p3m-b8k7-n16",
    }
    for experiment in experiments:
        topology = experiment.topology
        assert (
            topology.capture_ids,
            topology.serve_tp,
            topology.per_device_train_batch_size,
            topology.gradient_accumulation_steps,
            topology.num_attention_heads,
            topology.num_key_value_heads,
            topology.head_dim,
            topology.intermediate_size,
        ) == expected[experiment.target]
        assert experiment.run_name == expected_names[experiment.target]
        assert experiment.paths.output_root.endswith("-dflash2-16n")
        assert experiment.paths.target_receipt_path is not None
        assert experiment.paths.target_receipt_sha256 is not None
        assert experiment.paths.target_sha256 is not None
        assert experiment.cumulative_max_steps == (20, 4166, 14500, 25391)
    assert len({item.paths.output_root for item in experiments}) == 2
    assert len({item.experiment_id for item in experiments}) == 2


def test_builder_rejects_base_thinking_target_mix(tmp_path: Path) -> None:
    """A target-variant manifest cannot silently mix Base and Thinking snapshots."""
    template = tmp_path / "template.json"
    write_manifest(
        template,
        (_template_experiment("q30-thinking"), _template_experiment("q235-base")),
    )

    with pytest.raises(ValueError, match="exact Q30/Q235 Thinking Nemotron DFlash B8 seeds"):
        build_dflash2_nemotron_manifest(
            template,
            tmp_path / "output.json",
            "/home/user/ModelOpt-dflash2",
            _SOURCE_SHA,
            _SHA256,
            "f" * 40,
            "e" * 64,
            cluster_profile=_profile(tmp_path),
            target_variant="thinking",
        )


def test_builder_rejects_shared_dflash2_output_roots(tmp_path: Path) -> None:
    """Two target entries can never become concurrent writers to one output root."""
    template = tmp_path / "template.json"
    shared_root = "/lustre/results/shared-nemo-dflash-b8"
    write_manifest(
        template,
        tuple(
            replace(
                _template_experiment(target),
                paths=replace(_template_experiment(target).paths, output_root=shared_root),
            )
            for target in ("q30-thinking", "q235-thinking")
        ),
    )

    with pytest.raises(ValueError, match="unique output roots"):
        build_dflash2_nemotron_manifest(
            template,
            tmp_path / "output.json",
            "/home/user/ModelOpt-dflash2",
            _SOURCE_SHA,
            _SHA256,
            "f" * 40,
            "e" * 64,
            cluster_profile=_profile(tmp_path),
            target_variant="thinking",
        )


@pytest.mark.parametrize(
    ("target_label", "revision", "dims"),
    [
        (
            "q30-thinking",
            "144afc2f379b542fdd4e85a1fcd5e1f79112d95d",
            (32, 4, 128, 6144),
        ),
        (
            "q235-thinking",
            "6cbffae6d8e28b986a6b17bd36f42f9fa0f1f0a5",
            (64, 4, 128, 12288),
        ),
    ],
)
def test_target_contract_binds_thinking_label_to_exact_snapshot(
    tmp_path: Path,
    target_label: str,
    revision: str,
    dims: tuple[int, int, int, int],
) -> None:
    """Training and serve gates share one exact label/revision/dimension contract."""
    target = tmp_path / target_label
    target.mkdir()
    (target / "snapshot-manifest.json").write_text(
        json.dumps({"source_identity": revision}) + "\n"
    )
    (target / "config.json").write_text(
        json.dumps(
            dict(
                zip(
                    (
                        "num_attention_heads",
                        "num_key_value_heads",
                        "head_dim",
                        "intermediate_size",
                    ),
                    dims,
                    strict=True,
                )
            )
        )
        + "\n"
    )

    assert validate_dflash2_target_snapshot(target, target_label) == dflash2_target_spec(
        target_label
    )
    with pytest.raises(ValueError, match="snapshot revision mismatch"):
        validate_dflash2_target_snapshot(target, target_label.replace("thinking", "base"))
    config = json.loads((target / "config.json").read_text())
    config["head_dim"] += 1
    (target / "config.json").write_text(json.dumps(config) + "\n")
    with pytest.raises(ValueError, match="dimensions mismatch"):
        validate_dflash2_target_snapshot(target, target_label)


def test_builder_rejects_profile_source_or_native_16_node_drift(tmp_path: Path) -> None:
    """The manifest cannot target a stale source commit or a four-node capacity profile."""
    template = tmp_path / "template.json"
    write_manifest(template, (_template_experiment("q30-base"), _template_experiment("q235-base")))
    with pytest.raises(ValueError, match="profile source commit"):
        build_dflash2_nemotron_manifest(
            template,
            tmp_path / "output.json",
            "/home/user/ModelOpt-dflash2",
            _SOURCE_SHA,
            _SHA256,
            "f" * 40,
            "e" * 64,
            cluster_profile=_profile(tmp_path, "a" * 40),
        )


def test_builder_rejects_a_template_without_both_base_nemotron_seeds(tmp_path: Path) -> None:
    """Thinking, OPB, or incomplete templates cannot silently redefine the study."""
    template = tmp_path / "template.json"
    write_manifest(template, (_template_experiment("q30-base"),))

    with pytest.raises(ValueError, match="exact Q30/Q235 Base Nemotron DFlash B8 seeds"):
        build_dflash2_nemotron_manifest(
            template,
            tmp_path / "output.json",
            "/home/user/ModelOpt-dflash2",
            _SOURCE_SHA,
            _SHA256,
            "f" * 40,
            "e" * 64,
            cluster_profile=_profile(tmp_path),
        )


def test_target_and_dataset_receipts_bind_exact_bytes(tmp_path: Path) -> None:
    """A path-only target or 1.3M dataset cannot enter a DFlash2 job."""
    target = tmp_path / "target"
    target.mkdir()
    (target / "config.json").write_text("{}\n")
    dataset = tmp_path / "nemotron.jsonl"
    dataset.write_bytes(b'{"messages":[]}\n' * 1_309_377)
    target_receipt = tmp_path / "target.json"
    dataset_receipt = tmp_path / "dataset.json"
    write_artifact_receipt(target_receipt, target, kind="target")
    write_artifact_receipt(dataset_receipt, dataset, kind="dataset", occurrence_count=1_300_000)
    dataset_claim = json.loads(dataset_receipt.read_text())
    assert dataset_claim["physical_occurrence_count"] == 1_309_377
    assert dataset_claim["admitted_prefix_count"] == 1_300_000
    assert dataset_claim["admitted_order_policy"] == "huggingface-streaming-take-prefix-v1"
    assert len(dataset_claim["admitted_order_sha256"]) == 64
    assert dataset_claim["ordered_sources"][0]["admitted_occurrences"] == 1_300_000
    target_sha256 = artifact_tree_sha256(target)
    dataset_sha256 = artifact_tree_sha256(dataset)
    validate_artifact_receipt(
        target_receipt,
        expected_receipt_sha256=__import__("hashlib")
        .sha256(target_receipt.read_bytes())
        .hexdigest(),
        artifact_path=target,
        expected_artifact_sha256=target_sha256,
        kind="target",
    )
    validate_artifact_receipt(
        dataset_receipt,
        expected_receipt_sha256=__import__("hashlib")
        .sha256(dataset_receipt.read_bytes())
        .hexdigest(),
        artifact_path=dataset,
        expected_artifact_sha256=dataset_sha256,
        kind="dataset",
    )
    (target / "config.json").write_text("forged\n")
    with pytest.raises(ValueError, match="artifact bytes"):
        validate_artifact_receipt(
            target_receipt,
            expected_receipt_sha256=__import__("hashlib")
            .sha256(target_receipt.read_bytes())
            .hexdigest(),
            artifact_path=target,
            expected_artifact_sha256=target_sha256,
            kind="target",
        )


def test_dataset_receipt_replay_does_not_require_pyarrow_after_attestation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The batch preflight replays exact attested bytes without its own PyArrow install."""
    dataset = tmp_path / "nemotron.jsonl"
    dataset.write_bytes(b'{"messages":[]}\n' * 1_300_000)
    receipt = tmp_path / "dataset.json"
    write_artifact_receipt(receipt, dataset, kind="dataset", occurrence_count=1_300_000)
    receipt_sha = __import__("hashlib").sha256(receipt.read_bytes()).hexdigest()
    artifact_sha = artifact_tree_sha256(dataset)
    monkeypatch.setattr(
        runtime_contract,
        "_dataset_layout",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(ImportError("no pyarrow")),
    )
    validate_artifact_receipt(
        receipt,
        expected_receipt_sha256=receipt_sha,
        artifact_path=dataset,
        expected_artifact_sha256=artifact_sha,
        kind="dataset",
        verify_dataset_rows=False,
    )


def test_dataset_receipt_rejects_a_claimed_1_3m_count_for_other_bytes(tmp_path: Path) -> None:
    """The receipt builder derives row count instead of trusting its caller."""
    dataset = tmp_path / "not-1.3m.jsonl"
    dataset.write_text('{"messages":[]}\n')

    with pytest.raises(ValueError, match="occurrence count"):
        write_artifact_receipt(
            tmp_path / "forged.json",
            dataset,
            kind="dataset",
            occurrence_count=1_300_000,
        )


def test_materialize_dataset_view_replaces_absolute_links_with_authenticated_hardlinks(
    tmp_path: Path,
) -> None:
    """A container-visible dataset view preserves exact ordered bytes without dangling links."""
    targets = tmp_path / "targets"
    targets.mkdir()
    first = targets / "chat-00000.jsonl"
    second = targets / "code-00000.jsonl"
    first.write_text('{"messages": ["first"]}\n')
    second.write_text('{"messages": ["second"]}\n')
    source = tmp_path / "source"
    source.mkdir()
    (source / first.name).symlink_to(first.resolve())
    (source / second.name).symlink_to(second.resolve())
    output = tmp_path / "physical"
    receipt = tmp_path / "materialization.json"

    receipt_sha256 = runtime_contract.materialize_dataset_view(source, output, receipt)

    assert sorted(item.name for item in output.iterdir()) == [first.name, second.name]
    assert all(item.is_file() and not item.is_symlink() for item in output.iterdir())
    assert (output / first.name).stat().st_ino == first.stat().st_ino
    assert (output / second.name).stat().st_ino == second.stat().st_ino
    body = json.loads(receipt.read_text())
    claim = body.pop("receipt_sha256")
    expected_claim = hashlib.sha256(
        json.dumps(body, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    assert claim == expected_claim
    assert body["storage"] == "hardlink"
    assert body["source_tree_sha256"] == body["output_tree_sha256"]
    assert receipt_sha256 == hashlib.sha256(receipt.read_bytes()).hexdigest()
    with pytest.raises(FileExistsError):
        runtime_contract.materialize_dataset_view(source, output, receipt)


def test_materialize_dataset_view_authenticates_cross_filesystem_copies(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """EXDEV falls back to a rehashed byte copy and records that storage identity."""
    target = tmp_path / "target.jsonl"
    target.write_text('{"messages": ["cross-device"]}\n')
    source = tmp_path / "source"
    source.mkdir()
    (source / target.name).symlink_to(target.resolve())
    output = tmp_path / "physical"
    receipt = tmp_path / "materialization.json"

    def cross_device(*_args: object, **_kwargs: object) -> None:
        raise OSError(errno.EXDEV, "cross-device test")

    monkeypatch.setattr(runtime_contract.os, "link", cross_device)
    runtime_contract.materialize_dataset_view(source, output, receipt)

    copied = output / target.name
    assert copied.read_bytes() == target.read_bytes()
    assert copied.stat().st_ino != target.stat().st_ino
    body = json.loads(receipt.read_text())
    assert body["storage"] == "copy"
    assert body["ordered_files"][0]["storage"] == "copy"


def test_dflash2_submitter_serializes_canary_and_cumulative_writers() -> None:
    """Each model uses 20→4166→14500→25391 afterok dependencies on one output root."""
    script = (
        Path(__file__).resolve().parents[1] / "common/specdec/submit_dflash2_nemotron_chain.sh"
    ).read_text()
    assert "20 4166 14500 25391" in script
    assert '--dependency "$previous_job_id"' in script
    assert "previous_job_id=" in script
    assert "submit_drafter_training_wave.sh" in script
    assert "--target) TARGET=" in script
    assert "experiment.target != sys.argv[2]" in script


def test_dflash2_cluster_profiles_are_dedicated_native_16_node_profiles() -> None:
    """Legacy four-node profiles stay unchanged while DFlash2 gets pinned 16-node profiles."""
    from common.specdec.cluster_profile import load_cluster_profile

    profile_root = Path(__file__).resolve().parents[1] / "common/specdec/profiles"
    expected = {
        "lyris-dflash2.yaml": ("lyris", "gb200"),
        "oci-hsg-dflash2.yaml": ("oci-hsg", "batch"),
        "ptyche-dflash2.yaml": ("ptyche", "36x2-a01r"),
    }
    for filename, (name, partition) in expected.items():
        profile = load_cluster_profile(profile_root / filename)
        assert profile.name == name
        assert profile.partition == partition
        assert profile.modelopt_commit == "6eda6bbf54455086a54660fe7a7b06c415b87da5"
        assert profile.modelopt_feature_base == "6eda6bbf54455086a54660fe7a7b06c415b87da5"
        assert (profile.training_nodes, profile.training_segment) == (16, 16)
    assert load_cluster_profile(profile_root / "lyris.yaml").training_nodes == 4
    legacy_oci = load_cluster_profile(profile_root / "oci-hsg.yaml")
    dflash2_oci = load_cluster_profile(profile_root / "oci-hsg-dflash2.yaml")
    assert legacy_oci.training_nodes == 4
    assert dflash2_oci.account == "nemotron_n3_post"
    assert dflash2_oci.explicit_gpu_flag is True
    assert str(dflash2_oci.durable_root).endswith("/modelopt-specdec/dflash2-oci")
    assert load_cluster_profile(profile_root / "ptyche.yaml").training_nodes == 4


def test_dflash2_runtime_stager_inserts_and_verifies_the_receipt() -> None:
    """The archive builder must place the attestation into the staged venv before tar."""
    script = (
        Path(__file__).resolve().parents[1] / "common/specdec/stage_relocatable_runtime_archive.sh"
    ).read_text()
    for required in (
        "--dflash2-vllm-package",
        "--dflash2-vllm-expected-commit",
        "--dflash2-vllm-required-ancestor",
        "dflash2-vllm-runtime-receipt.json",
        "write_vllm_runtime_receipt",
        "verify_vllm_runtime",
        'SOURCE_FOR_ARCHIVE="$prepared_runtime"',
    ):
        assert required in script
def test_dflash2_runtime_builder_smokes_exact_installed_selector() -> None:
    """The CPU staging job proves both immutable bytes and an executable selector."""
    script = (
        Path(__file__).resolve().parents[1] / "common/specdec/build_dflash2_runtime.sbatch"
    ).read_text()
    for required in (
        "b389ac29465b33f9e9c534df221ea3c129e9793f",
        "--runtime-package",
        "dflash2-vllm-runtime-receipt.json",
        "verify_vllm_runtime",
        "_score_edges",
        "DFlash2Speculator",
        "flash_mla_interface",
        "vllm.v1.attention.ops.flashmla",
        "import vllm._flashmla_C",
        "import vllm._flashmla_extension_C",
        "_is_flashmla_available() == (True, None)",
        "FLASH_MLA_SRC_DIR",
        "VLLM_CUTLASS_SRC_DIR",
        "da5e086dab31d63815acafdac9a9c5893b1c69e2",
        "vllm-cutlass-source.tar",
        "focused FlashMLA-only external project closure",
        "omit inherited non-FlashMLA extension targets",
        '"spinloop", "fs_io_C", "cumem_allocator"',
        "focused_vllm_cmake",
        'include(cmake/external_projects/flashmla.cmake)',
        "TORCH_CUDA_ARCH_LIST=10.0a",
        "cmake==3.31.6",
        "ninja==1.13.0",
        "--ignore-installed",
        "--configure-only",
        "flashmla-configure-preflight",
        "cmake_path=",
        "ninja_path=",
        '"$prepared/bin/cmake" --version',
        '"$prepared/bin/ninja" --version',
        "CUDA target architectures",
        "FlashMLA CUDA architectures",
        '"$prepared/bin/cmake" --build',
        "--component _flashmla_C",
        "--component _flashmla_extension_C",
        "flashmla-build-manifest",
        "refusing to replace existing runtime output",
    ):
        assert required in script
    assert script.index('git -C "$VLLM_CHECKOUT" archive HEAD') < script.index(
        "srun --nodes=1"
    )


def test_dflash2_zero_init_serve_gate_is_runtime_only_and_receipt_bound() -> None:
    """The pre-training serve gate must be explicit, immutable, and non-scientific."""
    script = (
        Path(__file__).resolve().parents[1]
        / "common/specdec/run_dflash2_zero_init_serve_gate.sbatch"
    ).read_text()
    target_contract = (
        Path(__file__).resolve().parents[1]
        / "common/specdec/dflash2_target_contract.py"
    ).read_text()
    for required in (
        "scientific_training_authorized",
        "training_quality_claim",
        "checkpoint-0-runtime-only",
        "torch.device(\"meta\")",
        "to_empty(device=\"cpu\")",
        "parameter.zero_()",
        "base_kernel[:, 0].fill_(1.0)",
        '"num_attention_heads": int(num_attention_heads)',
        '"num_key_value_heads": int(num_key_value_heads)',
        '"head_dim": int(head_dim)',
        '"intermediate_size": int(intermediate_size)',
        '"dflash_block_size": 8',
        '"num_speculative_tokens": 7',
        '"method": "dflash"',
        "verify_vllm_runtime",
        "validate_artifact_receipt",
        "expected_receipt_sha256=target_receipt_sha",
        'expected_artifact_sha256=target_body["artifact_sha256"]',
        'kind="target"',
        "DFlash2DraftModel",
        "vllm._flashmla_C",
        "vllm._flashmla_extension_C",
        '--tensor-parallel-size "$serve_tp"',
        "/health",
        "/v1/completions",
        'tee "$server_log"',
        "completed_requests",
        "exporter_sha256",
        "reload_loader_sha256",
        "speculator_sha256",
        "runtime_receipt_sha256",
        "target_receipt_sha256",
        "receipt_sha256",
        "SLURM_JOB_ID",
        "refusing to replace existing serve-gate output",
        "TARGET_LABEL",
        "TARGET_REVISION",
        "validate_dflash2_target_snapshot",
    ):
        assert required in script
    for required in (
        "144afc2f379b542fdd4e85a1fcd5e1f79112d95d",
        "6cbffae6d8e28b986a6b17bd36f42f9fa0f1f0a5",
        '"q30-thinking":',
        '"q235-thinking":',
        "source_identity",
    ):
        assert required in target_contract


def test_dflash2_chain_accepts_thinking_targets_and_uses_clear_job_names() -> None:
    """Thinking canaries use target-labelled names without weakening duplicate identity."""
    script = (
        Path(__file__).resolve().parents[1]
        / "common/specdec/submit_dflash2_nemotron_chain.sh"
    ).read_text()
    for required in (
        "q30-thinking",
        "q235-thinking",
        "dfl2-q30t-nemo1p3m-b8k7-n16",
        "dfl2-q235t-nemo1p3m-b8k7-n16",
        "--canary-only",
        '--job-name "$job_name"',
    ):
        assert required in script


def test_dflash2_chain_requires_canary_only_for_thinking_targets() -> None:
    """Thinking may never fall through to the cumulative production boundaries."""
    script = (
        Path(__file__).resolve().parents[1]
        / "common/specdec/submit_dflash2_nemotron_chain.sh"
    ).read_text()
    assert 'if (( manifest_has_thinking && CANARY_ONLY == 0 )); then' in script
    assert "Thinking DFlash2 requires --canary-only" in script


def test_shared_runner_validates_exact_dflash2_target_snapshot_contract() -> None:
    """The training job itself rejects a wrong target revision, not just the serve gate."""
    runner = (
        Path(__file__).resolve().parents[1] / "common/specdec/run_drafter_training.sbatch"
    ).read_text()
    assert "validate_dflash2_target_snapshot" in runner
    assert "experiment.target" in runner


def test_dflash2_serve_gate_has_a_bounded_selector_diagnostic_mode() -> None:
    """A one-node retry must identify the exact selector index before any patch."""
    root = Path(__file__).resolve().parents[1] / "common/specdec"
    script = (root / "run_dflash2_zero_init_serve_gate.sbatch").read_text()
    script += (root / "dflash2_selector_diagnostic/sitecustomize.py").read_text()
    for required in (
        "DFLASH2_SELECTOR_DIAGNOSTIC",
        "CUDA_LAUNCH_BLOCKING=1",
        "--enforce-eager",
        "DFLASH2_SELECTOR_RANGES",
        "candidate_min",
        "candidate_max",
        "anchor_min",
        "anchor_max",
        "predecessor_min",
        "predecessor_max",
        "successor_rows",
        "predecessor_rows",
        "sitecustomize.INSTALLED",
    ):
        assert required in script


def test_pinned_vllm_patch_caps_dflash_profile_request_capacity() -> None:
    """Both DFlash profile modes must fit B8 queries into the 4096-token buffer."""
    patch_path = (
        Path(__file__).resolve().parents[1]
        / "common/specdec/patches/vllm-b389-dflash-profile-capacity.patch"
    )
    patch = patch_path.read_text()
    builder = (
        Path(__file__).resolve().parents[1]
        / "common/specdec/build_dflash2_runtime.sbatch"
    ).read_text()
    assert "a/vllm/v1/worker/gpu/model_runner.py" in patch
    assert "is_profile" in patch
    assert "self.speculative_config.use_dflash()" in patch
    assert "self.max_num_tokens // self.decode_query_len" in patch
    assert "max_profile_reqs <= 0" in patch
    assert "DFlash query width exceeds the profile token capacity" in patch
    assert "uniform_decode = True" not in patch
    assert "dflash/speculator.py" not in patch
    profile_reqs = {skip_attn: min(1024, 4096 // 8) for skip_attn in (True, False)}
    assert profile_reqs == {True: 512, False: 512}
    assert min(16, 4096 // 8) == 16
    assert min(1024, 4097 // 8) == 512
    assert 7 // 8 == 0
    assert 'git -C "$build_source" apply --check "$PROFILE_PATCH"' in builder
    assert 'git -C "$build_source" apply "$PROFILE_PATCH"' in builder
    assert 'cp -a "$build_source/vllm/." "$runtime_package/"' in builder
    assert "--runtime-source-patch" in builder


def test_dflash2_artifact_receipt_builder_creates_scratch_before_pyxis() -> None:
    """Pyxis must never receive a mount source that the host job has not created."""
    script = (
        Path(__file__).resolve().parents[1]
        / "common/specdec/build_dflash2_artifact_receipt.sbatch"
    ).read_text()
    mkdir = script.index('mkdir -p "$work_root/runtime"')
    launch = script.index("srun --nodes=1")
    assert mkdir < launch
    assert 'mounts="${work_root}:${work_root}' in script
    assert '"$work_root/runtime/bin/python"' in script
    assert "--occurrence-count 1300000" in script


def test_shared_runner_enforces_and_consumes_the_dflash2_contract() -> None:
    """The manifest contract reaches runtime verification and the training recipe."""
    runner = (
        Path(__file__).resolve().parents[1] / "common/specdec/run_drafter_training.sbatch"
    ).read_text()

    for required in (
        "dflash2:8)",
        '[[ "$DFLASH2_WARMSTART_POLICY" == exact-dflash2-only ]]',
        '[[ "$DFLASH2_KERNEL_PROJECTION_INIT" == zero ]]',
        '[[ "$DFLASH2_MAX_SPECULATIVE_TOKENS" == 7 ]]',
        "verify_vllm_runtime",
        '"dflash.dflash_architecture_config.projector_type=dflash2"',
        '"dflash.dflash_architecture_config.conv_kernel_size=${DFLASH2_CONV_KERNEL_SIZE}"',
        '"dflash.dflash_architecture_config.conv_group_size=${DFLASH2_CONV_GROUP_SIZE}"',
        '"dflash.dflash_architecture_config.selector_rank=${DFLASH2_SELECTOR_RANK}"',
        '"dflash.dflash_architecture_config.selector_top_k=${DFLASH2_SELECTOR_TOP_K}"',
        'if [[ "$METHOD" == dflash2 ]]; then',
        'STAGE_TARGET_IDENTITY="$TARGET_PATH"',
        'STAGE_DATASET_IDENTITY="$DATASET_PATH"',
    ):
        assert required in runner
