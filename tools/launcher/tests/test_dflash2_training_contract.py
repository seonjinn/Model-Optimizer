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

import json
import subprocess
import sys
from pathlib import Path

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
from common.specdec.drafter_job_manifest import (
    DrafterExperiment,
    PinnedPaths,
    SlurmSettings,
    TargetTopology,
    load_manifest,
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
    _git(repo, "add", "vllm/__init__.py")
    _git(repo, "commit", "-q", "-m", "required DFlash2 runtime")
    required = _git(repo, "rev-parse", "HEAD")
    (package / "runtime.py").write_text("SUPPORTED = True\n")
    _git(repo, "add", "vllm/runtime.py")
    _git(repo, "commit", "-q", "-m", "verified descendant")
    return package, required, _git(repo, "rev-parse", "HEAD")


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


def test_vllm_receipt_binds_tracked_source_and_compiled_runtime_extras(tmp_path: Path) -> None:
    """A wheel may add ARM64 extensions, but every tracked PR source byte must remain exact."""
    package, required, expected = _vllm_checkout(tmp_path)
    runtime = tmp_path / "runtime/vllm"
    runtime.parent.mkdir()
    __import__("shutil").copytree(package, runtime)
    (runtime / "_C.abi3.so").write_bytes(b"arm64-extension")
    receipt = tmp_path / "vllm-runtime-v2.json"
    receipt_sha = write_vllm_runtime_receipt(
        receipt,
        package,
        expected,
        required,
        runtime_package_path=runtime,
    )

    assert verify_vllm_runtime(runtime, receipt, receipt_sha, expected, required) == expected
    (runtime / "runtime.py").write_text("SUPPORTED = False\n")
    with pytest.raises(ValueError, match="runtime file bytes"):
        verify_vllm_runtime(runtime, receipt, receipt_sha, expected, required)


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
    is_q30 = target == "q30-base"
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
        "refusing to replace existing runtime output",
    ):
        assert required in script


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
