# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Fail-closed contracts for the matched Q30 DFlash2 step-4166 evaluation."""

from __future__ import annotations

import argparse
import collections
import contextlib
import csv
import hashlib
import json
import math
import os
import re
import shutil
import stat
import subprocess
import tempfile
import urllib.request
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Iterator

try:
    from common.specdec.dflash2_runtime_contract import (
        artifact_tree_sha256,
        validate_artifact_receipt,
    )
    from common.specdec.dflash2_target_contract import validate_dflash2_target_snapshot
except ModuleNotFoundError:  # Direct script execution from common/specdec.
    from dflash2_runtime_contract import artifact_tree_sha256, validate_artifact_receipt
    from dflash2_target_contract import validate_dflash2_target_snapshot

DATASET_ID = "RedHatAI/speculator_benchmarks"
DATASET_REVISION = "2ae86affa2cb97a972b7fc681dd51c04fbff083e"
STANDARD_SUBSETS = (
    "HumanEval",
    "math_reasoning",
    "qa",
    "question",
    "rag",
    "summarization",
    "tool_call",
    "translation",
    "writing",
)
EVALUATION_REQUESTS_PER_SUBSET = 200
CORRECTNESS_REQUESTS_PER_SUBSET = EVALUATION_REQUESTS_PER_SUBSET
PROBE_ROWS = (
    ("HumanEval", 1),
    ("math_reasoning", 0),
    ("qa", 0),
    ("tool_call", 0),
    ("translation", 0),
)
INTERNAL_TARGET_DIAGNOSTIC_ROWS = (
    (0, "exact-control"),
    (1, "source-1-first-invalid"),
    (8, "source-8-first-tied"),
    (27, "early-divergence-position-2"),
    (41, "early-divergence-position-1"),
    (82, "maximum-target-logprob-gap"),
    (151, "second-largest-target-logprob-gap"),
    (165, "source-1-repeat-tied"),
    (172, "source-8-repeat-invalid"),
)
RCA_SOURCE_PILOT_RECEIPT_SHA256 = "4b7c6254c7d52cf8da159d1bad261255790899e36068d0c7c4ccb66ea2199cca"
EVALUATION_STEP = 4166
DFLASH2_BLOCK_SIZE = 8
DFLASH2_SPECULATIVE_TOKENS = 7
OPB_DFLASH_CONFIG_SHA256 = "d502e18b23ea01dd7f0763840cd91a4545518cf594ba925d45de94eb1a40d35a"
OPB_DFLASH_MODEL_SHA256 = "8bc3f5608d5a0db4833cf7a972f886bb81c241620ff577149d85af7a14103d15"
OPB_DFLASH_CHECKSUM_MANIFEST_SHA256 = (
    "fc6b7a7ea0c48bb0b45bb3ec68fd6d4bd75be3b7308a462aeeba0da5b74ce9c3"
)
OPB_BUNDLE_IDENTITY_SHA256 = "dc86f88b28c12e749812b291f8ef6458dab8d2ee1d7171f585e60b20c62bad02"
OPB_DFLASH_IDENTITY_FIXTURE_PATH = (
    Path(__file__).resolve().parent / "fixtures/q30_opb_dflash_s4166_identity.json"
)
_OPB_STAGE_RECEIPT_KEYS = {
    "schema_version",
    "producer",
    "source",
    "staged",
    "copy_identity_sha256",
    "live_validation_scope",
}
_OPB_LIVE_VALIDATION_SCOPE = (
    "source bytes verified during host staging and finalization; node-local staged bytes "
    "reverified immediately before vLLM; offline verification authenticates exact descriptors"
)
_OPB_ARTIFACT_DESCRIPTOR_KEYS = {
    "path",
    "tree_sha256",
    "file_sha256",
    "architecture",
    "block_size",
    "num_speculative_tokens",
    "target_layer_ids",
    "source",
    "identity_fixture",
}
_DFLASH_CONTROL_RECEIPT_V1_KEYS = {
    "schema_version",
    "producer",
    "claim_scope",
    "selection",
    "engine_mode",
    "counts",
    "classifications",
    "acceptance",
    "next_action",
    "control",
    "control_outcome",
    "source_pilot_receipt_sha256",
    "allocation_evidence_scope",
    "source_pilot_receipt",
    "artifact_identity",
    "control_artifact",
    "allocation_receipt",
    "rows",
    "manifests",
    "input_fingerprints",
    "launcher_configs",
}
_DFLASH_CONTROL_RECEIPT_V2_KEYS = _DFLASH_CONTROL_RECEIPT_V1_KEYS | {"control_stage_receipt"}
_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")
_GIT_SHA_PATTERN = re.compile(r"[0-9a-f]{40}")
_CONTROL_MANIFEST_KEYS = {
    "status",
    "recorded_at",
    "method",
    "block_size",
    "num_speculative_tokens",
    "target_model",
    "draft_model",
    "speculators_repo",
    "speculators_sha",
    "modelopt_repo",
    "modelopt_sha",
    "modelopt_dirty",
    "runtime",
    "runtimes",
    "artifact_identity",
    "container",
    "dataset",
    "provenance_error",
    "slurm_job_id",
    "launcher_config",
    "config_sha256",
    "versions",
    "server_args",
    "evaluator_args",
    "evaluation",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _is_sha256(value: object) -> bool:
    return isinstance(value, str) and _SHA256_PATTERN.fullmatch(value) is not None


def _canonical(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _sha_json(value: object) -> str:
    return hashlib.sha256(_canonical(value).encode()).hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _atomic_json(path: Path, payload: dict[str, Any], *, no_replace: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w") as stream:
            stream.write(json.dumps(payload, indent=2, sort_keys=True) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        if no_replace:
            try:
                os.link(temporary, path)
            except FileExistsError as error:
                raise FileExistsError(f"output already exists: {path}") from error
            os.unlink(temporary)
        else:
            os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _read_prompt_prefix(path: Path, limit: int) -> list[tuple[int, str]]:
    prompts: list[tuple[int, str]] = []
    with path.open() as stream:
        for source_row, line in enumerate(stream):
            if source_row >= limit:
                break
            row = json.loads(line)
            prompt = row.get("prompt") if isinstance(row, dict) else None
            if not isinstance(prompt, str) or not prompt:
                raise ValueError(f"prompt {source_row} is not a non-empty string: {path}")
            recorded_source_row = row.get("_specdec_source_row", source_row)
            if (
                isinstance(recorded_source_row, bool)
                or not isinstance(recorded_source_row, int)
                or recorded_source_row < 0
            ):
                raise ValueError(f"invalid source row {source_row}: {path}")
            prompts.append((recorded_source_row, prompt))
    if not prompts:
        raise ValueError(f"no prompts: {path}")
    return prompts


def materialize_prompt_set(
    source_manifest_path: Path,
    source_hf_home: Path,
    output_hf_home: Path,
    output_manifest_path: Path,
) -> dict[str, Any]:
    """Materialize an authenticated source-order cycle of exactly 200 rows/subset."""
    source = compute_prompt_set(source_manifest_path, source_hf_home)
    if output_manifest_path.exists() or output_hf_home.exists():
        raise FileExistsError("matched prompt output already exists")
    snapshot = (
        output_hf_home
        / "hub/datasets--RedHatAI--speculator_benchmarks/snapshots"
        / DATASET_REVISION
    )
    snapshot.mkdir(parents=True)
    source_manifest = _load_json(source_manifest_path)
    source_files = source_manifest["files"]
    files: dict[str, dict[str, object]] = {}
    for subset in STANDARD_SUBSETS:
        source_path = Path(str(source_files[subset]["path"]))
        rows: list[dict[str, Any]] = []
        with source_path.open() as stream:
            for index, line in enumerate(stream):
                if index >= EVALUATION_REQUESTS_PER_SUBSET:
                    break
                row = json.loads(line)
                if not isinstance(row, dict) or not isinstance(row.get("prompt"), str):
                    raise ValueError(f"invalid source prompt row: {subset}[{index}]")
                rows.append(row)
        if not rows:
            raise ValueError(f"empty source subset: {subset}")
        output_path = snapshot / f"{subset}.jsonl"
        with output_path.open("x") as stream:
            for ordinal in range(EVALUATION_REQUESTS_PER_SUBSET):
                source_row = ordinal % len(rows)
                derived = {**rows[source_row], "_specdec_source_row": source_row}
                stream.write(_canonical(derived) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        files[subset] = {
            "path": str(output_path.resolve(strict=True)),
            "sha256": _sha256(output_path),
        }
    payload: dict[str, Any] = {
        "dataset_id": DATASET_ID,
        "revision": DATASET_REVISION,
        "hf_home": str(output_hf_home.resolve(strict=True)),
        "files": files,
        "derivation": {
            "producer": "speculator-benchmarks-source-order-cycle-v1",
            "requests_per_subset": EVALUATION_REQUESTS_PER_SUBSET,
            "source_manifest_path": str(source_manifest_path.resolve(strict=True)),
            "source_manifest_sha256": _sha256(source_manifest_path),
            "source_hf_home": str(source_hf_home.resolve(strict=True)),
            "source_prompt_sha256": source["prompt_sha256"],
            "source_files": source["files"],
        },
    }
    _atomic_json(output_manifest_path, payload, no_replace=True)
    compute_prompt_set(output_manifest_path, output_hf_home)
    return payload


def compute_prompt_set(
    dataset_manifest_path: Path,
    hf_home: Path,
    *,
    requests_per_subset: int = EVALUATION_REQUESTS_PER_SUBSET,
) -> dict[str, Any]:
    """Bind the ordered prompt prefix consumed by every matched cell."""
    if requests_per_subset != EVALUATION_REQUESTS_PER_SUBSET:
        raise ValueError("matched evaluation requires exactly 200 requests per subset")
    manifest = _load_json(dataset_manifest_path)
    recorded_home = Path(str(manifest.get("hf_home", ""))).resolve(strict=False)
    active_home = hf_home.resolve(strict=False)
    if manifest.get("dataset_id") != DATASET_ID:
        raise ValueError("dataset identity mismatch")
    if manifest.get("revision") != DATASET_REVISION:
        raise ValueError("dataset revision mismatch")
    if recorded_home != active_home:
        raise ValueError("dataset HF_HOME mismatch")
    files = manifest.get("files")
    if not isinstance(files, dict) or set(files) != set(STANDARD_SUBSETS):
        raise ValueError("dataset manifest must contain exactly the nine subsets")
    source_prompt_set: dict[str, Any] | None = None
    derivation = manifest.get("derivation")
    if derivation is not None:
        if (
            not isinstance(derivation, dict)
            or derivation.get("producer") != "speculator-benchmarks-source-order-cycle-v1"
            or derivation.get("requests_per_subset") != requests_per_subset
        ):
            raise ValueError("invalid matched-prompt derivation")
        source_manifest = Path(str(derivation.get("source_manifest_path", "")))
        source_home = Path(str(derivation.get("source_hf_home", "")))
        if (
            not source_manifest.is_file()
            or _sha256(source_manifest) != derivation.get("source_manifest_sha256")
            or source_manifest.resolve(strict=True) == dataset_manifest_path.resolve(strict=True)
        ):
            raise ValueError("matched-prompt source manifest mismatch")
        source_prompt_set = compute_prompt_set(source_manifest, source_home)
        if (
            derivation.get("source_prompt_sha256") != source_prompt_set["prompt_sha256"]
            or derivation.get("source_files") != source_prompt_set["files"]
        ):
            raise ValueError("matched-prompt source provenance mismatch")
    snapshot = (
        active_home / "hub/datasets--RedHatAI--speculator_benchmarks/snapshots" / DATASET_REVISION
    ).resolve(strict=False)
    prompts: list[dict[str, object]] = []
    file_descriptors: dict[str, dict[str, object]] = {}
    for subset in STANDARD_SUBSETS:
        entry = files[subset]
        if not isinstance(entry, dict):
            raise ValueError(f"invalid dataset entry: {subset}")
        path = Path(str(entry.get("path", "")))
        if path.resolve(strict=False) != (snapshot / f"{subset}.jsonl").resolve(strict=False):
            raise ValueError(f"dataset path mismatch: {subset}")
        expected_sha = entry.get("sha256")
        if not path.is_file() or not isinstance(expected_sha, str) or _sha256(path) != expected_sha:
            raise ValueError(f"dataset file hash mismatch: {subset}")
        available_prompts = _read_prompt_prefix(path, requests_per_subset)
        for ordinal in range(requests_per_subset):
            selected_row, prompt = available_prompts[ordinal % len(available_prompts)]
            prompts.append(
                {
                    "subset": subset,
                    "index": ordinal,
                    "source_row": selected_row,
                    "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
                }
            )
        file_descriptors[subset] = {
            "path": str(path.resolve(strict=True)),
            "sha256": expected_sha,
            "available_rows": len(available_prompts),
        }
    if source_prompt_set is not None and prompts != source_prompt_set["ordered_prompts"]:
        raise ValueError("materialized prompt schedule does not match its source")
    return {
        "dataset_id": DATASET_ID,
        "revision": DATASET_REVISION,
        "manifest_path": str(dataset_manifest_path.resolve(strict=True)),
        "manifest_sha256": _sha256(dataset_manifest_path),
        "subsets": list(STANDARD_SUBSETS),
        "requests_per_subset": requests_per_subset,
        "total_requests": len(prompts),
        "prompt_sha256": _sha_json(prompts),
        "ordered_prompts": prompts,
        "files": file_descriptors,
    }


def _regular_file_hashes(root: Path) -> dict[str, str]:
    return {
        path.relative_to(root).as_posix(): _sha256(path)
        for path in sorted(root.rglob("*"))
        if path.is_file() and not path.is_symlink()
    }


def load_opb_dflash_identity_fixture() -> dict[str, Any]:
    """Load the checked-in trust anchor and reject production constant drift."""
    path = OPB_DFLASH_IDENTITY_FIXTURE_PATH.resolve(strict=True)
    payload = _load_json(path)
    expected_keys = {
        "schema_version",
        "producer",
        "artifact_directory_name",
        "artifact_file_sha256",
        "source_manifest_sha256",
        "config",
        "source",
    }
    if (
        set(payload) != expected_keys
        or payload.get("schema_version") != 1
        or payload.get("producer") != "q30-opb-dflash-s4166-identity-v1"
        or payload.get("artifact_directory_name") != "dflash-s4166"
        or payload.get("artifact_file_sha256")
        != {
            "config.json": OPB_DFLASH_CONFIG_SHA256,
            "model.safetensors": OPB_DFLASH_MODEL_SHA256,
        }
        or payload.get("source_manifest_sha256")
        != {
            "manifest/dflash-s4166.sha256": OPB_DFLASH_CHECKSUM_MANIFEST_SHA256,
            "manifest/identity.json": OPB_BUNDLE_IDENTITY_SHA256,
        }
        or payload.get("config")
        != {
            "architecture": "DFlashDraftModel",
            "block_size": DFLASH2_BLOCK_SIZE,
            "mask_token_id": 151669,
            "num_speculative_tokens": DFLASH2_SPECULATIVE_TOKENS,
            "target_layer_ids": [1, 12, 23, 34, 45],
        }
        or payload.get("source")
        != {
            "opb_training_milestone": EVALUATION_STEP,
            "q30_revision": "ad44e777bcd18fa416d9da3bd8f70d33ebb85d39",
            "speculators_sha": "0b08a89a83b92007be63f128e01497455b0209df",
        }
    ):
        raise ValueError("checked-in OPB DFlash identity fixture mismatch")
    return {**payload, "path": str(path), "sha256": _sha256(path)}


def validate_milestone_export(
    milestone_manifest_path: Path,
    export_path: Path,
    *,
    expected_step: int = EVALUATION_STEP,
) -> dict[str, Any]:
    """Authenticate the exact lifecycle-preserved export at step 4166."""
    if expected_step != EVALUATION_STEP:
        raise ValueError("matched DFlash2 evaluation requires step 4166")
    manifest = _load_json(milestone_manifest_path)
    milestone = milestone_manifest_path.parent
    exact_model = milestone / "exact-model"
    expected_hashes = manifest.get("exact_model_sha256")
    if (
        manifest.get("exact_model_step") != expected_step
        or not exact_model.is_symlink()
        or exact_model.resolve(strict=True) != export_path.resolve(strict=True)
        or not isinstance(expected_hashes, dict)
    ):
        raise ValueError("milestone export identity mismatch")
    actual_hashes = _regular_file_hashes(export_path.resolve(strict=True))
    if actual_hashes != expected_hashes:
        raise ValueError("milestone export file hashes mismatch")
    config = _load_json(export_path / "config.json")
    architectures = config.get("architectures")
    if not isinstance(architectures, list) or "DFlash2DraftModel" not in architectures:
        raise ValueError("step-4166 export is not DFlash2")
    dflash_config = config.get("dflash_config")
    if (
        config.get("block_size") != DFLASH2_BLOCK_SIZE
        or config.get("num_attention_heads") != 32
        or config.get("num_key_value_heads") != 4
        or config.get("head_dim") != 128
        or config.get("intermediate_size") != 6144
        or not isinstance(dflash_config, dict)
        or dflash_config.get("projector_type") != "dflash2"
    ):
        raise ValueError("step-4166 export is not the Q30 Base B8 DFlash2 architecture")
    return {
        "step": expected_step,
        "milestone_manifest_path": str(milestone_manifest_path.resolve(strict=True)),
        "milestone_manifest_sha256": _sha256(milestone_manifest_path),
        "export_path": str(export_path.resolve(strict=True)),
        "export_tree_sha256": artifact_tree_sha256(export_path),
        "export_file_sha256": actual_hashes,
    }


def _opb_artifact_tree_sha256(file_sha256: dict[str, str]) -> str:
    digest = hashlib.sha256()
    for name in ("config.json", "model.safetensors"):
        digest.update(name.encode())
        digest.update(b"\0")
        digest.update(bytes.fromhex(file_sha256[name]))
    return digest.hexdigest()


def _open_directory_componentwise_nofollow(path: Path) -> int:
    if not path.is_absolute() or ".." in path.parts:
        raise ValueError("OPB DFlash source path must be absolute and symlink-free")
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path.anchor, flags)
    try:
        for component in path.parts[1:]:
            next_descriptor = os.open(component, flags, dir_fd=descriptor)
            os.close(descriptor)
            descriptor = next_descriptor
    except OSError as error:
        os.close(descriptor)
        raise ValueError("OPB DFlash source path must be absolute and symlink-free") from error
    return descriptor


def _assert_open_directory_path(path: Path, descriptor: int) -> None:
    try:
        path_stat = path.stat(follow_symlinks=False)
    except OSError as error:
        raise ValueError("OPB DFlash authenticated directory moved during access") from error
    opened_stat = os.fstat(descriptor)
    if (
        path_stat.st_dev,
        path_stat.st_ino,
        stat.S_IFMT(path_stat.st_mode),
    ) != (
        opened_stat.st_dev,
        opened_stat.st_ino,
        stat.S_IFMT(opened_stat.st_mode),
    ) or not stat.S_ISDIR(opened_stat.st_mode):
        raise ValueError("OPB DFlash authenticated directory moved during access")


def _assert_open_file_entry(parent_descriptor: int, name: str, descriptor: int) -> None:
    try:
        path_stat = os.stat(name, dir_fd=parent_descriptor, follow_symlinks=False)
    except OSError as error:
        raise ValueError(f"OPB DFlash file entry changed during access: {name}") from error
    opened_stat = os.fstat(descriptor)
    if (
        path_stat.st_dev,
        path_stat.st_ino,
        stat.S_IFMT(path_stat.st_mode),
    ) != (
        opened_stat.st_dev,
        opened_stat.st_ino,
        stat.S_IFMT(opened_stat.st_mode),
    ) or not stat.S_ISREG(opened_stat.st_mode):
        raise ValueError(f"OPB DFlash file entry changed during access: {name}")


@contextlib.contextmanager
def _open_opb_dflash_bundle(draft_path: Path) -> Iterator[dict[str, Any]]:
    lexical_draft = Path(os.path.abspath(draft_path))
    if (
        not draft_path.is_absolute()
        or ".." in draft_path.parts
        or lexical_draft.name != "dflash-s4166"
    ):
        raise ValueError("OPB DFlash control artifact path mismatch")
    bundle_path = lexical_draft.parent
    bundle_fd = _open_directory_componentwise_nofollow(bundle_path)
    directory_flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
    file_flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    descriptors: list[int] = [bundle_fd]
    try:
        draft_fd = os.open("dflash-s4166", directory_flags, dir_fd=bundle_fd)
        manifest_fd = os.open("manifest", directory_flags, dir_fd=bundle_fd)
        descriptors.extend((draft_fd, manifest_fd))
        if set(os.listdir(draft_fd)) != {"config.json", "model.safetensors"} or set(
            os.listdir(manifest_fd)
        ) != {"dflash-s4166.sha256", "identity.json"}:
            raise ValueError("OPB DFlash bundle file set mismatch")
        files: dict[str, int] = {}
        for name, parent_fd in (
            ("config.json", draft_fd),
            ("model.safetensors", draft_fd),
            ("dflash-s4166.sha256", manifest_fd),
            ("identity.json", manifest_fd),
        ):
            descriptor = os.open(name, file_flags, dir_fd=parent_fd)
            if not stat.S_ISREG(os.fstat(descriptor).st_mode):
                os.close(descriptor)
                raise ValueError(f"OPB DFlash source is not a regular file: {name}")
            descriptors.append(descriptor)
            files[name] = descriptor
        _assert_open_directory_path(bundle_path, bundle_fd)
        _assert_open_directory_path(lexical_draft, draft_fd)
        _assert_open_directory_path(bundle_path / "manifest", manifest_fd)
        for name, descriptor in files.items():
            parent_fd = draft_fd if name in {"config.json", "model.safetensors"} else manifest_fd
            _assert_open_file_entry(parent_fd, name, descriptor)
        opened = {
            "draft_path": lexical_draft,
            "bundle_path": bundle_path,
            "bundle_fd": bundle_fd,
            "draft_fd": draft_fd,
            "manifest_fd": manifest_fd,
            "files": files,
        }
        yield opened
        _assert_open_directory_path(bundle_path, bundle_fd)
        _assert_open_directory_path(lexical_draft, draft_fd)
        _assert_open_directory_path(bundle_path / "manifest", manifest_fd)
        for name, descriptor in files.items():
            parent_fd = draft_fd if name in {"config.json", "model.safetensors"} else manifest_fd
            _assert_open_file_entry(parent_fd, name, descriptor)
    except OSError as error:
        raise ValueError("OPB DFlash bundle must be componentwise symlink-free") from error
    finally:
        for descriptor in reversed(descriptors):
            with contextlib.suppress(OSError):
                os.close(descriptor)


def _read_opb_file(descriptor: int, label: str, *, capture: bool) -> tuple[str, int, bytes]:
    os.lseek(descriptor, 0, os.SEEK_SET)
    before = os.fstat(descriptor)
    digest = hashlib.sha256()
    content = bytearray()
    while chunk := os.read(descriptor, 8 * 1024 * 1024):
        digest.update(chunk)
        if capture:
            content.extend(chunk)
    after = os.fstat(descriptor)
    os.lseek(descriptor, 0, os.SEEK_SET)
    stable_fields = ("st_dev", "st_ino", "st_size", "st_mtime_ns")
    if not stat.S_ISREG(before.st_mode) or any(
        getattr(before, field) != getattr(after, field) for field in stable_fields
    ):
        raise ValueError(f"OPB DFlash file changed during access: {label}")
    return digest.hexdigest(), before.st_size, bytes(content)


def _validate_open_opb_dflash_bundle(opened: dict[str, Any]) -> dict[str, Any]:
    files = opened["files"]
    config_hash, _config_size, config_raw = _read_opb_file(
        files["config.json"], "config.json", capture=True
    )
    model_hash, _model_size, _ = _read_opb_file(
        files["model.safetensors"], "model.safetensors", capture=False
    )
    checksum_hash, checksum_size, checksum_raw = _read_opb_file(
        files["dflash-s4166.sha256"], "dflash-s4166.sha256", capture=True
    )
    identity_hash, identity_size, identity_raw = _read_opb_file(
        files["identity.json"], "identity.json", capture=True
    )
    hashes = {"config.json": config_hash, "model.safetensors": model_hash}
    if hashes != {
        "config.json": OPB_DFLASH_CONFIG_SHA256,
        "model.safetensors": OPB_DFLASH_MODEL_SHA256,
    }:
        raise ValueError("OPB DFlash control artifact hash mismatch")
    try:
        config = json.loads(config_raw)
        source_identity = json.loads(identity_raw)
        checksum_lines = checksum_raw.decode().splitlines()
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("OPB DFlash control metadata parse mismatch") from error
    dflash_config = config.get("dflash_config")
    if (
        config.get("architectures") != ["DFlashDraftModel"]
        or config.get("block_size") != DFLASH2_BLOCK_SIZE
        or not isinstance(dflash_config, dict)
        or dflash_config.get("mask_token_id") != 151669
        or dflash_config.get("target_layer_ids") != [1, 12, 23, 34, 45]
    ):
        raise ValueError("OPB DFlash control config mismatch")
    if (
        checksum_hash != OPB_DFLASH_CHECKSUM_MANIFEST_SHA256
        or identity_hash != OPB_BUNDLE_IDENTITY_SHA256
    ):
        raise ValueError("OPB DFlash control source manifest hash mismatch")
    if checksum_lines != [
        f"{OPB_DFLASH_CONFIG_SHA256}  ./config.json",
        f"{OPB_DFLASH_MODEL_SHA256}  ./model.safetensors",
    ]:
        raise ValueError("OPB DFlash control checksum manifest mismatch")
    if (
        source_identity.get("opb_training_milestone") != EVALUATION_STEP
        or source_identity.get("q30_revision") != "ad44e777bcd18fa416d9da3bd8f70d33ebb85d39"
        or source_identity.get("speculators_sha") != "0b08a89a83b92007be63f128e01497455b0209df"
        or source_identity.get("dflash")
        != {"block_size": DFLASH2_BLOCK_SIZE, "num_speculative_tokens": DFLASH2_SPECULATIVE_TOKENS}
    ):
        raise ValueError("OPB DFlash control source identity mismatch")
    draft_path = opened["draft_path"]
    manifest_root = opened["bundle_path"] / "manifest"
    return {
        "path": str(draft_path),
        "tree_sha256": _opb_artifact_tree_sha256(hashes),
        "file_sha256": hashes,
        "architecture": "DFlashDraftModel",
        "block_size": DFLASH2_BLOCK_SIZE,
        "num_speculative_tokens": DFLASH2_SPECULATIVE_TOKENS,
        "target_layer_ids": [1, 12, 23, 34, 45],
        "source": {
            "opb_training_milestone": EVALUATION_STEP,
            "q30_revision": source_identity["q30_revision"],
            "speculators_sha": source_identity["speculators_sha"],
            "identity": {
                "path": str(manifest_root / "identity.json"),
                "bytes": identity_size,
                "sha256": identity_hash,
            },
            "checksums": {
                "path": str(manifest_root / "dflash-s4166.sha256"),
                "bytes": checksum_size,
                "sha256": checksum_hash,
            },
        },
    }


def _validate_opb_dflash_control_artifact(draft_path: Path) -> dict[str, Any]:
    """Authenticate the exact OPB control through componentwise no-follow handles."""
    with _open_opb_dflash_bundle(draft_path) as opened:
        return _validate_open_opb_dflash_bundle(opened)


def _copy_opb_regular_file(
    source_descriptor: int, source_label: str, expected_sha256: str, destination: Path
) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    digest = hashlib.sha256()
    os.lseek(source_descriptor, 0, os.SEEK_SET)
    with destination.open("xb") as output_stream:
        before = os.fstat(source_descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise ValueError(f"OPB DFlash source is not a regular file: {source_label}")
        while chunk := os.read(source_descriptor, 8 * 1024 * 1024):
            output_stream.write(chunk)
            digest.update(chunk)
        output_stream.flush()
        os.fsync(output_stream.fileno())
        after_descriptor = os.fstat(source_descriptor)
    os.lseek(source_descriptor, 0, os.SEEK_SET)
    stable_fields = ("st_dev", "st_ino", "st_size", "st_mtime_ns")
    if any(getattr(before, field) != getattr(after_descriptor, field) for field in stable_fields):
        raise ValueError(f"OPB DFlash source changed during copy: {source_label}")
    if digest.hexdigest() != expected_sha256 or digest.hexdigest() != _sha256(destination):
        raise ValueError(f"OPB DFlash copy hash mismatch: {source_label}")


def _remove_opb_stage_tree(path: Path) -> None:
    if not path.exists():
        return
    for item in path.rglob("*"):
        with contextlib.suppress(FileNotFoundError):
            os.chmod(item, 0o700 if item.is_dir() else 0o600)
    os.chmod(path, 0o700)
    shutil.rmtree(path, ignore_errors=True)


def _opb_copy_identity(source: dict[str, Any], staged: dict[str, Any]) -> dict[str, Any]:
    return {
        "source_artifact_file_sha256": source["file_sha256"],
        "staged_artifact_file_sha256": staged["file_sha256"],
        "source_manifest_sha256": {
            "checksums": source["source"]["checksums"]["sha256"],
            "identity": source["source"]["identity"]["sha256"],
        },
        "staged_manifest_sha256": {
            "checksums": staged["source"]["checksums"]["sha256"],
            "identity": staged["source"]["identity"]["sha256"],
        },
    }


def _validate_opb_artifact_descriptor_identity(
    value: object, fixture: dict[str, Any]
) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != _OPB_ARTIFACT_DESCRIPTOR_KEYS:
        raise ValueError("OPB DFlash production identity descriptor mismatch")
    raw_path = value.get("path")
    if not isinstance(raw_path, str):
        raise ValueError("OPB DFlash production identity path mismatch")
    artifact_path = Path(raw_path)
    if (
        not artifact_path.is_absolute()
        or ".." in artifact_path.parts
        or artifact_path.name != "dflash-s4166"
    ):
        raise ValueError("OPB DFlash production identity path mismatch")
    expected_files = {
        "config.json": OPB_DFLASH_CONFIG_SHA256,
        "model.safetensors": OPB_DFLASH_MODEL_SHA256,
    }
    if (
        value.get("tree_sha256") != _opb_artifact_tree_sha256(expected_files)
        or value.get("file_sha256") != expected_files
        or value.get("architecture") != "DFlashDraftModel"
        or value.get("block_size") != DFLASH2_BLOCK_SIZE
        or value.get("num_speculative_tokens") != DFLASH2_SPECULATIVE_TOKENS
        or value.get("target_layer_ids") != [1, 12, 23, 34, 45]
        or value.get("identity_fixture") != {"path": fixture["path"], "sha256": fixture["sha256"]}
    ):
        raise ValueError("OPB DFlash production identity mismatch")
    source = value.get("source")
    if not isinstance(source, dict) or set(source) != {
        "opb_training_milestone",
        "q30_revision",
        "speculators_sha",
        "identity",
        "checksums",
    }:
        raise ValueError("OPB DFlash production identity source mismatch")
    if (
        source.get("opb_training_milestone") != EVALUATION_STEP
        or source.get("q30_revision") != "ad44e777bcd18fa416d9da3bd8f70d33ebb85d39"
        or source.get("speculators_sha") != "0b08a89a83b92007be63f128e01497455b0209df"
    ):
        raise ValueError("OPB DFlash production identity source mismatch")
    manifest_root = artifact_path.parent / "manifest"
    for name, expected_path, expected_hash in (
        ("identity", manifest_root / "identity.json", OPB_BUNDLE_IDENTITY_SHA256),
        (
            "checksums",
            manifest_root / "dflash-s4166.sha256",
            OPB_DFLASH_CHECKSUM_MANIFEST_SHA256,
        ),
    ):
        descriptor = source.get(name)
        if (
            not isinstance(descriptor, dict)
            or set(descriptor) != {"path", "bytes", "sha256"}
            or descriptor.get("path") != str(expected_path)
            or not isinstance(descriptor.get("bytes"), int)
            or isinstance(descriptor.get("bytes"), bool)
            or descriptor["bytes"] < 0
            or descriptor.get("sha256") != expected_hash
        ):
            raise ValueError("OPB DFlash production identity manifest mismatch")
    return value


def stage_opb_dflash_control(
    source_draft_path: Path, stage_root: Path, receipt_path: Path
) -> dict[str, Any]:
    """Copy the exact OPB control to a private node-local immutable tree."""
    fixture = load_opb_dflash_identity_fixture()
    fixture_descriptor = {"path": fixture["path"], "sha256": fixture["sha256"]}
    stage_root = stage_root.resolve(strict=False)
    if stage_root.exists() or stage_root.is_symlink():
        raise FileExistsError(f"OPB DFlash stage already exists: {stage_root}")
    stage_root.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    os.chmod(stage_root.parent, 0o700)
    partial = Path(tempfile.mkdtemp(prefix=f".{stage_root.name}.", dir=stage_root.parent))
    try:
        with _open_opb_dflash_bundle(source_draft_path) as opened:
            source = {
                **_validate_open_opb_dflash_bundle(opened),
                "identity_fixture": fixture_descriptor,
            }
            _validate_opb_artifact_descriptor_identity(source, fixture)
            for name, relative, expected_hash in (
                (
                    "config.json",
                    Path("dflash-s4166/config.json"),
                    OPB_DFLASH_CONFIG_SHA256,
                ),
                (
                    "model.safetensors",
                    Path("dflash-s4166/model.safetensors"),
                    OPB_DFLASH_MODEL_SHA256,
                ),
                (
                    "dflash-s4166.sha256",
                    Path("manifest/dflash-s4166.sha256"),
                    OPB_DFLASH_CHECKSUM_MANIFEST_SHA256,
                ),
                (
                    "identity.json",
                    Path("manifest/identity.json"),
                    OPB_BUNDLE_IDENTITY_SHA256,
                ),
            ):
                _copy_opb_regular_file(
                    opened["files"][name], name, expected_hash, partial / relative
                )
            post_source = {
                **_validate_open_opb_dflash_bundle(opened),
                "identity_fixture": fixture_descriptor,
            }
            if post_source != source:
                raise ValueError("OPB DFlash source descriptor changed during staging")
        staged = {
            **_validate_opb_dflash_control_artifact(partial / "dflash-s4166"),
            "identity_fixture": fixture_descriptor,
        }
        _validate_opb_artifact_descriptor_identity(staged, fixture)
        copy_identity = _opb_copy_identity(source, staged)
        if (
            source["tree_sha256"] != staged["tree_sha256"]
            or source["file_sha256"] != staged["file_sha256"]
            or copy_identity["source_manifest_sha256"] != copy_identity["staged_manifest_sha256"]
        ):
            raise ValueError("OPB DFlash staged copy identity mismatch")
        for path in sorted(partial.rglob("*"), reverse=True):
            os.chmod(path, 0o500 if path.is_dir() else 0o400)
        os.chmod(partial, 0o500)
        os.rename(partial, stage_root)
        staged = {
            **_validate_opb_dflash_control_artifact(stage_root / "dflash-s4166"),
            "identity_fixture": fixture_descriptor,
        }
        payload = {
            "schema_version": 1,
            "producer": "q30-opb-dflash-s4166-node-local-stage-v1",
            "source": source,
            "staged": staged,
            "copy_identity_sha256": _sha_json(_opb_copy_identity(source, staged)),
            "live_validation_scope": _OPB_LIVE_VALIDATION_SCOPE,
        }
        payload["receipt_sha256"] = _sha_json(payload)
        _atomic_json(receipt_path, payload, no_replace=True)
        return payload
    except BaseException:
        if partial.exists():
            _remove_opb_stage_tree(partial)
        if stage_root.exists():
            _remove_opb_stage_tree(stage_root)
        raise


def validate_opb_dflash_stage_receipt(
    path: Path,
    *,
    require_live_stage: bool = False,
    require_live_source: bool = True,
    expected_staged_draft: Path | None = None,
) -> dict[str, Any]:
    """Replay the source/staged copy chain and optionally rehash both live trees."""
    payload = _load_json(path)
    claim = payload.pop("receipt_sha256", None)
    if set(payload) != _OPB_STAGE_RECEIPT_KEYS or claim != _sha_json(payload):
        raise ValueError("OPB DFlash stage receipt mismatch")
    if (
        payload.get("schema_version") != 1
        or payload.get("producer") != "q30-opb-dflash-s4166-node-local-stage-v1"
        or payload.get("live_validation_scope") != _OPB_LIVE_VALIDATION_SCOPE
    ):
        raise ValueError("OPB DFlash stage receipt schema mismatch")
    if not require_live_stage and not require_live_source:
        raise ValueError("staged-only validation requires a live staged-byte check")
    source = payload.get("source")
    staged = payload.get("staged")
    if not isinstance(source, dict) or not isinstance(staged, dict):
        raise ValueError("OPB DFlash stage descriptor mismatch")
    fixture = load_opb_dflash_identity_fixture()
    _validate_opb_artifact_descriptor_identity(source, fixture)
    _validate_opb_artifact_descriptor_identity(staged, fixture)
    if source["path"] == staged["path"]:
        raise ValueError("OPB DFlash source and staged paths must be distinct")
    if expected_staged_draft is not None:
        recorded_stage = Path(str(staged.get("path", ""))).resolve(strict=require_live_stage)
        expected_stage = expected_staged_draft.resolve(strict=require_live_stage)
        if recorded_stage != expected_stage:
            raise ValueError("OPB DFlash staged path mismatch")
    copy_identity = _opb_copy_identity(source, staged)
    if (
        payload.get("copy_identity_sha256") != _sha_json(copy_identity)
        or source.get("tree_sha256") != staged.get("tree_sha256")
        or source.get("file_sha256") != staged.get("file_sha256")
        or copy_identity["source_manifest_sha256"] != copy_identity["staged_manifest_sha256"]
    ):
        raise ValueError("OPB DFlash stage copy identity mismatch")
    if require_live_stage:
        fixture_descriptor = {"path": fixture["path"], "sha256": fixture["sha256"]}
        if require_live_source:
            try:
                live_source = {
                    **_validate_opb_dflash_control_artifact(Path(str(source["path"]))),
                    "identity_fixture": fixture_descriptor,
                }
            except (OSError, ValueError) as error:
                raise ValueError("OPB DFlash source descriptor mismatch") from error
            if live_source != source:
                raise ValueError("OPB DFlash source descriptor mismatch")
        try:
            live_staged = {
                **_validate_opb_dflash_control_artifact(Path(str(staged["path"]))),
                "identity_fixture": fixture_descriptor,
            }
        except (OSError, ValueError) as error:
            raise ValueError("OPB DFlash staged descriptor mismatch") from error
        if live_staged != staged:
            raise ValueError("OPB DFlash staged descriptor mismatch")
    return {**payload, "receipt_sha256": claim}


def build_artifact_identity(
    *,
    target_path: Path,
    target_sha256: str,
    target_receipt_path: Path,
    target_receipt_sha256: str,
    export_path: Path,
    milestone_manifest_path: Path,
    dataset_manifest_path: Path,
    hf_home: Path,
    client_runtime_archive: Path,
    client_runtime_archive_sha256: str,
    server_runtime_archive: Path,
    server_runtime_archive_sha256: str,
    server_runtime_receipt_sha256: str,
) -> dict[str, Any]:
    """Rehash every immutable scientific input shared by C1 and C32."""
    target_spec = validate_dflash2_target_snapshot(target_path, "q30-base")
    validate_artifact_receipt(
        target_receipt_path,
        expected_receipt_sha256=target_receipt_sha256,
        artifact_path=target_path,
        expected_artifact_sha256=target_sha256,
        kind="target",
    )
    export = validate_milestone_export(milestone_manifest_path, export_path)
    prompt_set = compute_prompt_set(dataset_manifest_path, hf_home)
    for path, expected, label in (
        (client_runtime_archive, client_runtime_archive_sha256, "client"),
        (server_runtime_archive, server_runtime_archive_sha256, "server"),
    ):
        if not _is_sha256(expected) or _sha256(path) != expected:
            raise ValueError(f"{label} runtime archive SHA-256 mismatch")
    if not _is_sha256(server_runtime_receipt_sha256):
        raise ValueError("server runtime receipt SHA-256 must be exact")
    payload = {
        "schema_version": 1,
        "producer": "q30-dflash2-s4166-speculators-inputs-v1",
        "target": {
            "path": str(target_path.resolve(strict=True)),
            "artifact_sha256": target_sha256,
            "receipt_path": str(target_receipt_path.resolve(strict=True)),
            "receipt_sha256": target_receipt_sha256,
            "label": "q30-base",
            "revision": target_spec.revision,
            "capture_ids": list(target_spec.capture_ids),
            "serve_tp": target_spec.serve_tp,
        },
        "draft": export,
        "dataset": prompt_set,
        "runtimes": {
            "client": {
                "archive_path": str(client_runtime_archive.resolve(strict=True)),
                "archive_sha256": client_runtime_archive_sha256,
            },
            "server": {
                "archive_path": str(server_runtime_archive.resolve(strict=True)),
                "archive_sha256": server_runtime_archive_sha256,
                "receipt_sha256": server_runtime_receipt_sha256,
                "vllm_commit": "b389ac29465b33f9e9c534df221ea3c129e9793f",
                "profile_capacity_patch_sha256": (
                    "540fbb17ff1b1f388435c924b73a56c997b88642608e1fa466e8874dd6e59212"
                ),
                "flashmla_commit": "a8f794d1251cbfd88a5011445dd5582289c727e4",
            },
        },
        "sampling": {
            "temperature": 0,
            "top_p": 1,
            "correctness": {
                "seed": 42,
                "max_tokens": 64,
                "logprobs": 0,
                "return_tokens_as_token_ids": True,
            },
        },
        "matrix": {
            "concurrencies": [1, 32],
            "requests_per_subset": 200,
            "tensor_parallel_size": {"baseline": 2, "dflash2": 2},
            "subsets": list(STANDARD_SUBSETS),
            "dflash2": {"block_size": 8, "num_speculative_tokens": 7},
        },
    }
    payload["receipt_sha256"] = _sha_json(payload)
    return payload


def _load_ledger(path: Path, requests_per_subset: int) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open() as stream:
        for line in stream:
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError("output ledger row must be an object")
            records.append(value)
    expected = [
        (subset, index) for subset in STANDARD_SUBSETS for index in range(requests_per_subset)
    ]
    actual = [(row.get("subset"), row.get("index")) for row in records]
    if actual != expected:
        raise ValueError("output ledger request order/count mismatch")
    for row in records:
        if (
            set(row)
            != {
                "subset",
                "index",
                "source_row",
                "prompt_sha256",
                "output_sha256",
                "output_text",
                "output_tokens",
                "request_sha256",
                "seed",
                "finish_reason",
            }
            or not all(
                _is_sha256(row.get(name))
                for name in ("prompt_sha256", "output_sha256", "request_sha256")
            )
            or isinstance(row.get("source_row"), bool)
            or not isinstance(row.get("source_row"), int)
            or row["source_row"] < 0
            or not isinstance(row.get("output_text"), str)
            or not isinstance(row.get("output_tokens"), list)
            or not all(isinstance(token, str) for token in row["output_tokens"])
            or row.get("seed") != 42
            or hashlib.sha256(row["output_text"].encode()).hexdigest() != row["output_sha256"]
            or row.get("finish_reason") not in {"stop", "length"}
        ):
            raise ValueError("output ledger row schema mismatch")
    return records


def _common_prefix_length(left: list[str], right: list[str]) -> int:
    length = 0
    for expected, actual in zip(left, right):
        if expected != actual:
            break
        length += 1
    return length


def _within_server_repeat_metrics(rows: list[dict[str, Any]]) -> dict[str, int | float]:
    groups: dict[tuple[str, int, str], list[dict[str, Any]]] = collections.defaultdict(list)
    for row in rows:
        groups[(row["subset"], row["source_row"], row["prompt_sha256"])].append(row)
    repeats = [values for values in groups.values() if len(values) > 1]
    token_prefixes = [
        _common_prefix_length(values[0]["output_tokens"], row["output_tokens"])
        for values in repeats
        for row in values[1:]
    ]
    character_prefixes = [
        _common_prefix_length(list(values[0]["output_text"]), list(row["output_text"]))
        for values in repeats
        for row in values[1:]
    ]
    return {
        "repeat_groups": len(repeats),
        "divergent_text_repeat_groups": sum(
            len({row["output_sha256"] for row in values}) > 1 for values in repeats
        ),
        "divergent_token_repeat_groups": sum(
            len({_canonical(row["output_tokens"]) for row in values}) > 1 for values in repeats
        ),
        "mean_common_token_prefix": (
            sum(token_prefixes) / len(token_prefixes) if token_prefixes else 0
        ),
        "mean_common_character_prefix": (
            sum(character_prefixes) / len(character_prefixes) if character_prefixes else 0
        ),
    }


def summarize_target_control(
    left_path: Path,
    right_path: Path,
    *,
    requests_per_subset: int = CORRECTNESS_REQUESTS_PER_SUBSET,
    expected_prompt_schedule: list[dict[str, object]] | None = None,
    expected_request_sha256: list[str] | None = None,
) -> dict[str, Any]:
    """Quantify target-only runtime nondeterminism without authorizing speed metrics."""
    left = _load_ledger(left_path, requests_per_subset)
    right = _load_ledger(right_path, requests_per_subset)
    left_schedule = [
        {
            "subset": row["subset"],
            "index": row["index"],
            "source_row": row["source_row"],
            "prompt_sha256": row["prompt_sha256"],
        }
        for row in left
    ]
    right_schedule = [
        {
            "subset": row["subset"],
            "index": row["index"],
            "source_row": row["source_row"],
            "prompt_sha256": row["prompt_sha256"],
        }
        for row in right
    ]
    if left_schedule != right_schedule or (
        expected_prompt_schedule is not None and left_schedule != expected_prompt_schedule
    ):
        raise ValueError("target control prompt schedule mismatch")
    left_request_sha256 = [row["request_sha256"] for row in left]
    if left_request_sha256 != [row["request_sha256"] for row in right] or (
        expected_request_sha256 is not None and left_request_sha256 != expected_request_sha256
    ):
        raise ValueError("target control request fingerprint mismatch")

    subset_metrics: dict[str, dict[str, object]] = {}
    token_prefixes: list[int] = []
    text_prefixes: list[int] = []
    exact_text_total = 0
    exact_token_total = 0
    for subset in STANDARD_SUBSETS:
        pairs = [
            (expected, actual)
            for expected, actual in zip(left, right, strict=True)
            if expected["subset"] == subset
        ]
        exact_text = sum(
            expected["output_text"] == actual["output_text"] for expected, actual in pairs
        )
        exact_tokens = sum(
            expected["output_tokens"] == actual["output_tokens"] for expected, actual in pairs
        )
        prefixes = [
            _common_prefix_length(expected["output_tokens"], actual["output_tokens"])
            for expected, actual in pairs
        ]
        char_prefixes = [
            _common_prefix_length(list(expected["output_text"]), list(actual["output_text"]))
            for expected, actual in pairs
        ]
        exact_text_total += exact_text
        exact_token_total += exact_tokens
        token_prefixes.extend(prefixes)
        text_prefixes.extend(char_prefixes)
        subset_metrics[subset] = {
            "requests": len(pairs),
            "exact_text_matches": exact_text,
            "exact_token_matches": exact_tokens,
            "mean_common_token_prefix": sum(prefixes) / len(prefixes),
            "mean_common_character_prefix": sum(char_prefixes) / len(char_prefixes),
        }
    return {
        "schema_version": 1,
        "producer": "q30-target-target-runtime-control-v1",
        "status": "diagnostic-only",
        "claim_scope": "no training-quality or speedup claim",
        "allocation_evidence_scope": (
            "live SLURM/GPU origin checked at creation; offline verification is tamper replay"
        ),
        "requests": len(left),
        "sampling": {"temperature": 0, "top_p": 1, "seed": 42},
        "request_set_sha256": _sha_json(left_request_sha256),
        "left_ledger_sha256": _sha256(left_path),
        "right_ledger_sha256": _sha256(right_path),
        "cross_server": {
            "exact_text_matches": exact_text_total,
            "exact_token_matches": exact_token_total,
            "mean_common_token_prefix": sum(token_prefixes) / len(token_prefixes),
            "mean_common_character_prefix": sum(text_prefixes) / len(text_prefixes),
            "subsets": subset_metrics,
        },
        "within_server": {
            "left": _within_server_repeat_metrics(left),
            "right": _within_server_repeat_metrics(right),
        },
        "task_correctness": {
            "status": "not-claimed",
            "reason": "performance prompts do not provide a safe uniform semantic scorer",
        },
    }


def validate_output_equivalence(
    baseline_path: Path,
    dflash2_path: Path,
    *,
    requests_per_subset: int = CORRECTNESS_REQUESTS_PER_SUBSET,
    artifact_identity_sha256: str | None = None,
    expected_prompt_schedule: list[dict[str, object]] | None = None,
) -> dict[str, Any]:
    """Require deterministic target output identity before speed metrics are publishable."""
    if requests_per_subset != CORRECTNESS_REQUESTS_PER_SUBSET:
        raise ValueError("correctness gate requires all 200 prompts per subset")
    baseline = _load_ledger(baseline_path, requests_per_subset)
    dflash2 = _load_ledger(dflash2_path, requests_per_subset)
    if expected_prompt_schedule is not None:
        actual_schedule = [
            {
                "subset": row["subset"],
                "index": row["index"],
                "source_row": row["source_row"],
                "prompt_sha256": row["prompt_sha256"],
            }
            for row in baseline
        ]
        if actual_schedule != expected_prompt_schedule:
            raise ValueError("captured prompt ledger does not match artifact identity")
    for expected, actual in zip(baseline, dflash2, strict=True):
        if expected != actual:
            raise ValueError(f"target output mismatch: {expected['subset']}[{expected['index']}]")
    if artifact_identity_sha256 is not None and not _is_sha256(artifact_identity_sha256):
        raise ValueError("artifact identity SHA-256 must be exact")
    return {
        "status": "passed",
        "matched_requests": len(baseline),
        "requests_per_subset": requests_per_subset,
        "baseline_ledger_sha256": _sha256(baseline_path),
        "dflash2_ledger_sha256": _sha256(dflash2_path),
        "output_set_sha256": _sha_json(baseline),
        "artifact_identity_sha256": artifact_identity_sha256,
    }


def _read_csv_by_subset(path: Path) -> dict[str, dict[str, str]]:
    with path.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) != len(STANDARD_SUBSETS) or {row.get("subset") for row in rows} != set(
        STANDARD_SUBSETS
    ):
        raise ValueError(f"expected exactly nine subset rows: {path}")
    return {row["subset"]: row for row in rows}


def _positive(row: dict[str, str], name: str) -> float:
    try:
        value = float(row[name])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"invalid {name}") from error
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"invalid {name}")
    return value


def _nonnegative(row: dict[str, str], name: str) -> float:
    try:
        value = float(row[name])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"invalid {name}") from error
    if not math.isfinite(value) or value < 0:
        raise ValueError(f"invalid {name}")
    return value


def _file_descriptor(path: Path) -> dict[str, object]:
    resolved = path.resolve(strict=True)
    return {"path": str(resolved), "bytes": resolved.stat().st_size, "sha256": _sha256(resolved)}


def _validate_file_descriptor(value: object) -> Path:
    if not isinstance(value, dict) or set(value) != {"path", "bytes", "sha256"}:
        raise ValueError("invalid evidence file descriptor")
    path = Path(str(value["path"])).resolve(strict=True)
    if (
        not isinstance(value["bytes"], int)
        or isinstance(value["bytes"], bool)
        or value["bytes"] != path.stat().st_size
        or not _is_sha256(value["sha256"])
        or value["sha256"] != _sha256(path)
    ):
        raise ValueError("evidence file descriptor mismatch")
    return path


def build_target_control_allocation_receipt(
    output_path: Path,
    *,
    slurm_job_id: str,
    slurm_job_num_nodes: int,
    slurm_job_nodelist: str,
    gpu_count: int,
) -> dict[str, Any]:
    """Publish job-local proof of the intended one-node disjoint TP2+TP2 allocation."""
    if (
        re.fullmatch(r"[1-9][0-9]*", slurm_job_id) is None
        or slurm_job_num_nodes != 1
        or not slurm_job_nodelist
        or gpu_count != 4
    ):
        raise ValueError("target control allocation must be one node with exactly four GPUs")
    payload: dict[str, Any] = {
        "schema_version": 1,
        "producer": "q30-target-control-allocation-v1",
        "slurm_job_id": slurm_job_id,
        "slurm_job_num_nodes": slurm_job_num_nodes,
        "slurm_job_nodelist": slurm_job_nodelist,
        "gpu_count": gpu_count,
        "cell_visible_devices": {"left": "0,1", "right": "2,3"},
    }
    payload["receipt_sha256"] = _sha_json(payload)
    _atomic_json(output_path, payload, no_replace=True)
    return payload


def validate_target_control_allocation_receipt(path: Path) -> dict[str, Any]:
    """Replay the exact one-node, four-GPU allocation claim."""
    payload = _load_json(path)
    expected_keys = {
        "schema_version",
        "producer",
        "slurm_job_id",
        "slurm_job_num_nodes",
        "slurm_job_nodelist",
        "gpu_count",
        "cell_visible_devices",
        "receipt_sha256",
    }
    unsigned = {key: value for key, value in payload.items() if key != "receipt_sha256"}
    if (
        set(payload) != expected_keys
        or payload.get("schema_version") != 1
        or payload.get("producer") != "q30-target-control-allocation-v1"
        or payload.get("receipt_sha256") != _sha_json(unsigned)
        or re.fullmatch(r"[1-9][0-9]*", str(payload.get("slurm_job_id", ""))) is None
        or payload.get("slurm_job_num_nodes") != 1
        or not isinstance(payload.get("slurm_job_nodelist"), str)
        or not payload.get("slurm_job_nodelist")
        or payload.get("gpu_count") != 4
        or payload.get("cell_visible_devices") != {"left": "0,1", "right": "2,3"}
    ):
        raise ValueError("invalid target control allocation receipt")
    return payload


def _query_current_allocation() -> dict[str, Any]:
    try:
        gpu_lines = subprocess.run(
            ["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.splitlines()
        node_count = int(os.environ["SLURM_JOB_NUM_NODES"])
        job_id = os.environ["SLURM_JOB_ID"]
        nodelist = os.environ["SLURM_JOB_NODELIST"]
    except (KeyError, ValueError, subprocess.CalledProcessError) as error:
        raise ValueError("allocation receipt must be produced inside the active GPU job") from error
    return {
        "slurm_job_id": job_id,
        "slurm_job_num_nodes": node_count,
        "slurm_job_nodelist": nodelist,
        "gpu_count": len(gpu_lines),
        "cell_visible_devices": {"left": "0,1", "right": "2,3"},
    }


def _publish_current_allocation_receipt(output_path: Path) -> dict[str, Any]:
    current = _query_current_allocation()
    return build_target_control_allocation_receipt(
        output_path,
        slurm_job_id=str(current["slurm_job_id"]),
        slurm_job_num_nodes=int(current["slurm_job_num_nodes"]),
        slurm_job_nodelist=str(current["slurm_job_nodelist"]),
        gpu_count=int(current["gpu_count"]),
    )


def _git_output(repo: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(repo), *args],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _validate_clean_repo(path_value: object, sha_value: object, name: str) -> Path:
    path = Path(str(path_value)).resolve(strict=True)
    if (
        not path.is_dir()
        or not isinstance(sha_value, str)
        or _GIT_SHA_PATTERN.fullmatch(sha_value) is None
    ):
        raise ValueError(f"invalid {name} source checkout")
    try:
        head = _git_output(path, "rev-parse", "HEAD")
        dirty = _git_output(path, "status", "--porcelain")
    except subprocess.CalledProcessError as error:
        raise ValueError(f"invalid {name} source checkout") from error
    if head != sha_value or dirty:
        raise ValueError(f"{name} source checkout mismatch")
    return path


def _validate_control_launcher(path_value: object, image: Path, expected_sha: object) -> Path:
    path = Path(str(path_value)).resolve(strict=True)
    if not _is_sha256(expected_sha) or _sha256(path) != expected_sha:
        raise ValueError("target control launcher config mismatch")
    expected = (
        f"pipeline:\n  task_0:\n    slurm_config:\n      container: {image.resolve(strict=True)}\n"
    )
    if path.read_text() != expected:
        raise ValueError("target control launcher config is not canonical")
    return path


def _validate_artifact_identity(path: Path) -> dict[str, Any]:
    identity = _load_json(path)
    if (
        identity.get("producer") != "q30-dflash2-s4166-speculators-inputs-v1"
        or identity.get("schema_version") != 1
        or identity.get("receipt_sha256")
        != _sha_json({key: value for key, value in identity.items() if key != "receipt_sha256"})
    ):
        raise ValueError("artifact identity receipt mismatch")
    target = identity.get("target")
    draft = identity.get("draft")
    dataset = identity.get("dataset")
    runtimes = identity.get("runtimes")
    if not all(isinstance(value, dict) for value in (target, draft, dataset, runtimes)):
        raise ValueError("artifact identity schema mismatch")
    assert isinstance(target, dict)
    assert isinstance(draft, dict)
    assert isinstance(dataset, dict)
    assert isinstance(runtimes, dict)
    client = runtimes.get("client")
    server = runtimes.get("server")
    if not isinstance(client, dict) or not isinstance(server, dict):
        raise ValueError("artifact identity runtime schema mismatch")
    dataset_manifest_path = Path(str(dataset.get("manifest_path", "")))
    dataset_manifest = _load_json(dataset_manifest_path)
    rebuilt = build_artifact_identity(
        target_path=Path(str(target.get("path", ""))),
        target_sha256=str(target.get("artifact_sha256", "")),
        target_receipt_path=Path(str(target.get("receipt_path", ""))),
        target_receipt_sha256=str(target.get("receipt_sha256", "")),
        export_path=Path(str(draft.get("export_path", ""))),
        milestone_manifest_path=Path(str(draft.get("milestone_manifest_path", ""))),
        dataset_manifest_path=dataset_manifest_path,
        hf_home=Path(str(dataset_manifest.get("hf_home", ""))),
        client_runtime_archive=Path(str(client.get("archive_path", ""))),
        client_runtime_archive_sha256=str(client.get("archive_sha256", "")),
        server_runtime_archive=Path(str(server.get("archive_path", ""))),
        server_runtime_archive_sha256=str(server.get("archive_sha256", "")),
        server_runtime_receipt_sha256=str(server.get("receipt_sha256", "")),
    )
    if identity != rebuilt:
        raise ValueError("artifact identity replay mismatch")
    return identity


def _expected_target_server_args(
    target_path: str, port: str, *, disable_prefix_caching: bool = False
) -> list[str]:
    args = [
        "-m",
        "vllm.entrypoints.cli.main",
        "serve",
        target_path,
        "--tensor-parallel-size",
        "2",
        "--port",
        port,
    ]
    if disable_prefix_caching:
        args.append("--no-enable-prefix-caching")
    return args


def _expected_dflash2_server_args(
    target_path: str,
    draft_path: str,
    port: str,
    *,
    detailed_metrics: bool = False,
    enforce_eager: bool = False,
    disable_prefix_caching: bool = False,
) -> list[str]:
    if disable_prefix_caching and not enforce_eager:
        raise ValueError("disabling prefix caching requires eager DFlash2 diagnosis")
    args = _expected_target_server_args(target_path, port)
    args.extend(
        [
            "--speculative-config",
            _canonical(
                {
                    "method": "dflash",
                    "model": draft_path,
                    "num_speculative_tokens": DFLASH2_SPECULATIVE_TOKENS,
                }
            ),
        ]
    )
    if detailed_metrics:
        args.extend(["--per-request-spec-decode-metrics", "detailed"])
    if enforce_eager:
        args.append("--enforce-eager")
    if disable_prefix_caching:
        args.append("--no-enable-prefix-caching")
    return args


def _validate_target_control_manifest(
    path: Path,
    artifact_identity_path: Path,
    identity: dict[str, Any],
    *,
    disable_prefix_caching: bool = False,
) -> tuple[dict[str, Any], Path, dict[str, Any], Path]:
    manifest = _load_json(path)
    evaluation = manifest.get("evaluation")
    server_args = manifest.get("server_args")
    target = identity["target"]
    dataset = identity["dataset"]
    runtimes = identity["runtimes"]
    artifact_sha256 = _sha256(artifact_identity_path)
    config_sha256 = manifest.get("config_sha256")
    artifact_binding = manifest.get("artifact_identity")
    container = manifest.get("container")
    manifest_dataset = manifest.get("dataset")
    runtimes_value = manifest.get("runtimes")
    versions = manifest.get("versions")
    evaluation_expected = {
        "dataset": DATASET_ID,
        "subsets": list(STANDARD_SUBSETS),
        "temperature": 0,
        "top_p": 1,
        "mode": "throughput",
        "max_concurrency": 1,
        "max_requests": 200,
        "tensor_parallel_size": 2,
    }
    try:
        recorded_at = datetime.fromisoformat(str(manifest.get("recorded_at", "")))
    except ValueError as error:
        raise ValueError("target control recorded_at is invalid") from error
    expected_server_args = _expected_target_server_args(
        target["path"],
        server_args[7] if isinstance(server_args, list) and len(server_args) > 7 else "",
        disable_prefix_caching=disable_prefix_caching,
    )
    if (
        set(manifest) != _CONTROL_MANIFEST_KEYS
        or manifest.get("method") != "baseline"
        or manifest.get("status") != "success"
        or manifest.get("block_size") != 0
        or manifest.get("num_speculative_tokens") != 0
        or manifest.get("draft_model") is not None
        or manifest.get("target_model") != target["path"]
        or manifest.get("modelopt_dirty") is not False
        or manifest.get("provenance_error") is not None
        or manifest.get("evaluator_args") != []
        or recorded_at.tzinfo is None
        or not isinstance(evaluation, dict)
        or evaluation != evaluation_expected
        or not isinstance(server_args, list)
        or not all(isinstance(value, str) for value in server_args)
        or any(
            value == "--speculative-config" or value.startswith("--speculative-config=")
            for value in server_args
        )
        or server_args != expected_server_args
        or server_args[7] not in {"8000", "8010"}
        or not isinstance(config_sha256, dict)
        or set(config_sha256) != {"launcher", "target"}
        or not all(_is_sha256(value) for value in config_sha256.values())
        or config_sha256["target"] != _sha256(Path(target["path"]) / "config.json")
        or artifact_binding
        != {"path": str(artifact_identity_path.resolve(strict=True)), "sha256": artifact_sha256}
        or not isinstance(container, dict)
        or not isinstance(manifest_dataset, dict)
        or not isinstance(runtimes_value, dict)
        or set(runtimes_value) != {"client", "server"}
        or manifest.get("runtime") != runtimes_value.get("client")
        or not all(isinstance(value, str) and value for value in runtimes_value.values())
        or not isinstance(versions, dict)
        or set(versions) != {"python", "vllm", "guidellm"}
        or not all(isinstance(value, str) and value for value in versions.values())
        or re.fullmatch(r"[1-9][0-9]*", str(manifest.get("slurm_job_id", ""))) is None
    ):
        raise ValueError("target control manifest is not exact baseline C1")

    _validate_clean_repo(manifest.get("modelopt_repo"), manifest.get("modelopt_sha"), "ModelOpt")
    _validate_clean_repo(
        manifest.get("speculators_repo"), manifest.get("speculators_sha"), "Speculators"
    )

    container_identity_path = Path(str(container.get("identity_path", "")))
    container_identity = _load_json(container_identity_path)
    container_path = Path(str(container.get("path", ""))).resolve(strict=True)
    if (
        container.get("identity_sha256") != _sha256(container_identity_path)
        or container_identity
        != {
            "path": container.get("path"),
            "sha256": container.get("sha256"),
            "size_bytes": container.get("size_bytes"),
        }
        or not _is_sha256(container.get("sha256"))
        or not isinstance(container.get("size_bytes"), int)
        or isinstance(container.get("size_bytes"), bool)
        or not container_path.is_file()
        or container_path.stat().st_size != container.get("size_bytes")
        or _sha256(container_path) != container.get("sha256")
    ):
        raise ValueError("target control container identity mismatch")
    launcher_path = _validate_control_launcher(
        manifest.get("launcher_config"), container_path, config_sha256["launcher"]
    )
    expected_dataset_files = {
        subset: {
            "path": dataset["files"][subset]["path"],
            "sha256": dataset["files"][subset]["sha256"],
        }
        for subset in STANDARD_SUBSETS
    }
    dataset_manifest = _load_json(Path(dataset["manifest_path"]))
    if manifest_dataset != {
        "dataset_id": dataset["dataset_id"],
        "revision": dataset["revision"],
        "hf_home": dataset_manifest["hf_home"],
        "manifest_path": dataset["manifest_path"],
        "manifest_sha256": dataset["manifest_sha256"],
        "files": expected_dataset_files,
    }:
        raise ValueError("target control dataset provenance mismatch")

    fingerprint_path = path.parent / "input-fingerprint.json"
    fingerprint = _load_json(fingerprint_path)
    inputs = fingerprint.get("inputs")
    if (
        fingerprint.get("schema_version") != 1
        or not isinstance(inputs, dict)
        or fingerprint.get("sha256") != _sha_json(inputs)
    ):
        raise ValueError("target control input fingerprint self-hash mismatch")
    expected_inputs = {
        "target_config_sha256": config_sha256["target"],
        "draft_config_sha256": None,
        "dataset": {
            "revision": dataset["revision"],
            "manifest_sha256": dataset["manifest_sha256"],
        },
        "image": {
            "sha256": container["sha256"],
            "identity_sha256": container["identity_sha256"],
        },
        "runtimes": {
            "client": {"sha256": runtimes["client"]["archive_sha256"]},
            "server": {
                "sha256": runtimes["server"]["archive_sha256"],
                "receipt_sha256": runtimes["server"]["receipt_sha256"],
            },
        },
        "artifact_identity_sha256": artifact_sha256,
        "source": {
            "modelopt_sha": manifest.get("modelopt_sha"),
            "speculators_sha": manifest.get("speculators_sha"),
        },
        "launcher_config_sha256": config_sha256["launcher"],
        "evaluation": {
            "method": "baseline",
            "block_size": 0,
            "num_speculative_tokens": 0,
            "max_concurrency": 1,
            "max_requests": 200,
            "mode": "throughput",
            "tensor_parallel_size": 2,
        },
    }
    if inputs != expected_inputs:
        raise ValueError("target control input fingerprint mismatch")
    return manifest, fingerprint_path, fingerprint, launcher_path


def build_target_control_receipt(
    left_path: Path,
    right_path: Path,
    left_manifest_path: Path,
    right_manifest_path: Path,
    artifact_identity_path: Path,
    allocation_receipt_path: Path,
    output_path: Path,
) -> dict[str, Any]:
    """Publish a self-hashed, replayable target-only nondeterminism receipt."""
    identity = _validate_artifact_identity(artifact_identity_path)
    dataset = identity.get("dataset")
    if not isinstance(dataset, dict) or not isinstance(dataset.get("ordered_prompts"), list):
        raise ValueError("artifact identity has no ordered prompt schedule")
    manifests = [left_manifest_path, right_manifest_path]
    manifest_evidence = [
        _validate_target_control_manifest(path, artifact_identity_path, identity)
        for path in manifests
    ]
    manifest_payloads = [evidence[0] for evidence in manifest_evidence]
    fingerprint_paths = [evidence[1] for evidence in manifest_evidence]
    fingerprint_payloads = [evidence[2] for evidence in manifest_evidence]
    launcher_paths = [evidence[3] for evidence in manifest_evidence]
    for name in (
        "config_sha256",
        "container",
        "dataset",
        "modelopt_sha",
        "speculators_sha",
        "target_model",
        "versions",
        "artifact_identity",
        "runtimes",
    ):
        if manifest_payloads[0].get(name) != manifest_payloads[1].get(name):
            raise ValueError(f"target control manifest mismatch: {name}")
    if fingerprint_payloads[0].get("inputs") != fingerprint_payloads[1].get("inputs"):
        raise ValueError("target control input fingerprint mismatch between cells")
    allocation = validate_target_control_allocation_receipt(allocation_receipt_path)
    current_allocation = _query_current_allocation()
    for name in (
        "slurm_job_id",
        "slurm_job_num_nodes",
        "slurm_job_nodelist",
        "gpu_count",
        "cell_visible_devices",
    ):
        if allocation.get(name) != current_allocation.get(name):
            raise ValueError(f"target control live allocation mismatch: {name}")
    if any(
        manifest.get("slurm_job_id") != allocation["slurm_job_id"] for manifest in manifest_payloads
    ):
        raise ValueError("target control allocation job mismatch")
    ports = {manifest["server_args"][-1] for manifest in manifest_payloads}
    if ports != {"8000", "8010"}:
        raise ValueError("target control server ports are not isolated")
    payload = summarize_target_control(
        left_path,
        right_path,
        expected_prompt_schedule=dataset["ordered_prompts"],
        expected_request_sha256=_expected_request_hashes(identity),
    )
    payload["artifact_identity"] = _file_descriptor(artifact_identity_path)
    payload["manifests"] = [_file_descriptor(path) for path in manifests]
    payload["input_fingerprints"] = [_file_descriptor(path) for path in fingerprint_paths]
    payload["launcher_configs"] = [_file_descriptor(path) for path in launcher_paths]
    payload["allocation_receipt"] = _file_descriptor(allocation_receipt_path)
    payload["ledgers"] = [_file_descriptor(left_path), _file_descriptor(right_path)]
    payload["receipt_sha256"] = _sha_json(payload)
    _atomic_json(output_path, payload, no_replace=True)
    return payload


def validate_target_control_receipt(path: Path) -> dict[str, Any]:
    """Rehash all target-control evidence and replay every reported metric."""
    payload = _load_json(path)
    claim = payload.pop("receipt_sha256", None)
    if claim != _sha_json(payload):
        raise ValueError("target control receipt self-hash mismatch")
    artifact_path = _validate_file_descriptor(payload.get("artifact_identity"))
    manifest_values = payload.get("manifests")
    ledger_values = payload.get("ledgers")
    if (
        not isinstance(manifest_values, list)
        or len(manifest_values) != 2
        or not isinstance(ledger_values, list)
        or len(ledger_values) != 2
    ):
        raise ValueError("target control evidence descriptor mismatch")
    manifests = [_validate_file_descriptor(value) for value in manifest_values]
    ledgers = [_validate_file_descriptor(value) for value in ledger_values]
    identity = _validate_artifact_identity(artifact_path)
    dataset = identity.get("dataset")
    if not isinstance(dataset, dict) or not isinstance(dataset.get("ordered_prompts"), list):
        raise ValueError("artifact identity has no ordered prompt schedule")
    fingerprint_values = payload.get("input_fingerprints")
    if not isinstance(fingerprint_values, list) or len(fingerprint_values) != 2:
        raise ValueError("target control input fingerprint descriptor mismatch")
    fingerprint_paths = [_validate_file_descriptor(value) for value in fingerprint_values]
    launcher_values = payload.get("launcher_configs")
    if not isinstance(launcher_values, list) or len(launcher_values) != 2:
        raise ValueError("target control launcher config descriptor mismatch")
    launcher_paths = [_validate_file_descriptor(value) for value in launcher_values]
    allocation_path = _validate_file_descriptor(payload.get("allocation_receipt"))
    allocation = validate_target_control_allocation_receipt(allocation_path)
    manifest_evidence = [
        _validate_target_control_manifest(item, artifact_path, identity) for item in manifests
    ]
    manifest_payloads = [evidence[0] for evidence in manifest_evidence]
    if [evidence[1] for evidence in manifest_evidence] != fingerprint_paths:
        raise ValueError("target control input fingerprint path mismatch")
    if [evidence[3] for evidence in manifest_evidence] != launcher_paths:
        raise ValueError("target control launcher config path mismatch")
    for name in (
        "config_sha256",
        "container",
        "dataset",
        "modelopt_sha",
        "speculators_sha",
        "target_model",
        "versions",
        "artifact_identity",
        "runtimes",
    ):
        if manifest_payloads[0].get(name) != manifest_payloads[1].get(name):
            raise ValueError(f"target control manifest mismatch: {name}")
    if manifest_evidence[0][2].get("inputs") != manifest_evidence[1][2].get("inputs"):
        raise ValueError("target control input fingerprint mismatch between cells")
    if {manifest["server_args"][-1] for manifest in manifest_payloads} != {"8000", "8010"}:
        raise ValueError("target control server ports are not isolated")
    if any(
        manifest.get("slurm_job_id") != allocation["slurm_job_id"] for manifest in manifest_payloads
    ):
        raise ValueError("target control allocation job mismatch")
    replayed = summarize_target_control(
        ledgers[0],
        ledgers[1],
        expected_prompt_schedule=dataset["ordered_prompts"],
        expected_request_sha256=_expected_request_hashes(identity),
    )
    replayed["artifact_identity"] = payload["artifact_identity"]
    replayed["manifests"] = payload["manifests"]
    replayed["input_fingerprints"] = payload["input_fingerprints"]
    replayed["launcher_configs"] = payload["launcher_configs"]
    replayed["allocation_receipt"] = payload["allocation_receipt"]
    replayed["ledgers"] = payload["ledgers"]
    if payload != replayed:
        raise ValueError("target control receipt replay mismatch")
    return {**payload, "receipt_sha256": claim}


def _metric_evidence(
    baseline_run: Path,
    dflash2_run: Path,
    *,
    concurrency: int,
    artifact_identity_sha256: str,
) -> dict[str, object]:
    if concurrency not in {1, 32} or not _is_sha256(artifact_identity_sha256):
        raise ValueError("invalid matched metric evidence request")
    result: dict[str, object] = {}
    for method, run in (("baseline", baseline_run), ("dflash2", dflash2_run)):
        manifest_path = run / "manifest.json"
        fingerprint_path = run / "input-fingerprint.json"
        manifest = _load_json(manifest_path)
        fingerprint = _load_json(fingerprint_path)
        inputs = fingerprint.get("inputs")
        evaluation = manifest.get("evaluation")
        artifact = manifest.get("artifact_identity")
        if (
            manifest.get("status") != "success"
            or manifest.get("method") != method
            or not isinstance(evaluation, dict)
            or evaluation.get("max_concurrency") != concurrency
            or evaluation.get("max_requests") != EVALUATION_REQUESTS_PER_SUBSET
            or evaluation.get("tensor_parallel_size") != 2
            or evaluation.get("temperature") != 0
            or evaluation.get("top_p") != 1
            or not isinstance(artifact, dict)
            or artifact.get("sha256") != artifact_identity_sha256
            or not isinstance(inputs, dict)
            or fingerprint.get("sha256") != _sha_json(inputs)
            or inputs.get("artifact_identity_sha256") != artifact_identity_sha256
        ):
            raise ValueError(f"invalid {method} cell evidence")
        descriptors = {
            "manifest": _file_descriptor(manifest_path),
            "input_fingerprint": _file_descriptor(fingerprint_path),
            "performance_csv": _file_descriptor(run / "perf_results.csv"),
        }
        if method == "dflash2":
            descriptors["acceptance_csv"] = _file_descriptor(run / "acceptance.csv")
        result[method] = descriptors
    return result


def summarize_pair(
    baseline_run: Path,
    dflash2_run: Path,
    *,
    tensor_parallel_size: int,
) -> dict[str, Any]:
    """Compute matched per-subset and macro metrics only after correctness passes."""
    if tensor_parallel_size != 2:
        raise ValueError("Q30 matched evaluation requires TP2+TP2")
    baseline = _read_csv_by_subset(baseline_run / "perf_results.csv")
    draft = _read_csv_by_subset(dflash2_run / "perf_results.csv")
    acceptance = _read_csv_by_subset(dflash2_run / "acceptance.csv")
    rows: list[dict[str, Any]] = []
    for subset in STANDARD_SUBSETS:
        baseline_tps = _positive(baseline[subset], "output_tps_median")
        draft_tps = _positive(draft[subset], "output_tps_median")
        baseline_latency = _positive(baseline[subset], "latency_median_s")
        draft_latency = _positive(draft[subset], "latency_median_s")
        accepted = _nonnegative(acceptance[subset], "num_accepted_tokens")
        drafted = _positive(acceptance[subset], "num_draft_tokens")
        mean_length = _positive(acceptance[subset], "acceptance_length")
        rows.append(
            {
                "subset": subset,
                "output_tps_per_gpu_baseline": baseline_tps / tensor_parallel_size,
                "output_tps_per_gpu_dflash2": draft_tps / tensor_parallel_size,
                "output_tps_speedup": draft_tps / baseline_tps,
                "latency_s_baseline": baseline_latency,
                "latency_s_dflash2": draft_latency,
                "latency_speedup": baseline_latency / draft_latency,
                "acceptance_rate": accepted / drafted,
                "mean_accepted_length": mean_length,
            }
        )

    def macro(name: str) -> float:
        return sum(float(row[name]) for row in rows) / len(rows)

    return {
        "method": "dflash2",
        "block_size": DFLASH2_BLOCK_SIZE,
        "num_speculative_tokens": DFLASH2_SPECULATIVE_TOKENS,
        "tensor_parallel_size": tensor_parallel_size,
        "subsets": rows,
        "aggregate": {
            name: macro(name)
            for name in (
                "output_tps_per_gpu_baseline",
                "output_tps_per_gpu_dflash2",
                "output_tps_speedup",
                "latency_s_baseline",
                "latency_s_dflash2",
                "latency_speedup",
                "acceptance_rate",
                "mean_accepted_length",
            )
        },
    }


def validate_report_receipt(path: Path) -> dict[str, Any]:
    """Replay a published metric receipt from its exact bound cell artifacts."""
    payload = _load_json(path)
    claim = payload.pop("receipt_sha256", None)
    if claim != _sha_json(payload):
        raise ValueError("metric report self-hash mismatch")
    artifact_path = _validate_file_descriptor(payload.get("artifact_identity"))
    correctness_path = _validate_file_descriptor(payload.get("correctness_receipt"))
    artifact_sha = _sha256(artifact_path)
    correctness = _load_json(correctness_path)
    correctness_claim = correctness.pop("receipt_sha256", None)
    if (
        correctness_claim != _sha_json(correctness)
        or correctness.get("status") != "passed"
        or correctness.get("artifact_identity_sha256") != artifact_sha
    ):
        raise ValueError("metric report correctness gate mismatch")
    evidence = payload.get("evidence")
    if not isinstance(evidence, dict) or set(evidence) != {"baseline", "dflash2"}:
        raise ValueError("metric report evidence is incomplete")
    runs: dict[str, Path] = {}
    for method in ("baseline", "dflash2"):
        descriptors = evidence[method]
        expected = {"manifest", "input_fingerprint", "performance_csv"}
        if method == "dflash2":
            expected.add("acceptance_csv")
        if not isinstance(descriptors, dict) or set(descriptors) != expected:
            raise ValueError(f"invalid {method} metric evidence")
        paths = {name: _validate_file_descriptor(value) for name, value in descriptors.items()}
        runs[method] = paths["manifest"].parent
    baseline_manifest = _load_json(runs["baseline"] / "manifest.json")
    evaluation = baseline_manifest.get("evaluation")
    if not isinstance(evaluation, dict) or evaluation.get("max_concurrency") not in {1, 32}:
        raise ValueError("metric report concurrency is invalid")
    concurrency = int(evaluation["max_concurrency"])
    replayed_evidence = _metric_evidence(
        runs["baseline"],
        runs["dflash2"],
        concurrency=concurrency,
        artifact_identity_sha256=artifact_sha,
    )
    replayed_metrics = summarize_pair(runs["baseline"], runs["dflash2"], tensor_parallel_size=2)
    for key, value in replayed_metrics.items():
        if payload.get(key) != value:
            raise ValueError(f"metric report replay mismatch: {key}")
    if payload.get("evidence") != replayed_evidence:
        raise ValueError("metric report evidence replay mismatch")
    return {**payload, "receipt_sha256": claim}


def _completion_request_body(model: str, subset: str, index: int, prompt: str) -> bytes:
    return json.dumps(
        {
            "model": model,
            "prompt": prompt,
            "max_tokens": 64,
            "temperature": 0,
            "top_p": 1,
            "seed": 42,
            "logprobs": 0,
            "return_tokens_as_token_ids": True,
            "request_id": f"specdec-correctness-{subset}-{index}",
        }
    ).encode()


def _expected_request_hashes(identity: dict[str, Any]) -> list[str]:
    target = identity.get("target")
    dataset = identity.get("dataset")
    if not isinstance(target, dict) or not isinstance(dataset, dict):
        raise ValueError("artifact identity target/dataset mismatch")
    model = target.get("path")
    files = dataset.get("files")
    if not isinstance(model, str) or not isinstance(files, dict):
        raise ValueError("artifact identity request inputs mismatch")
    hashes: list[str] = []
    for subset in STANDARD_SUBSETS:
        entry = files.get(subset)
        if not isinstance(entry, dict):
            raise ValueError("artifact identity request dataset mismatch")
        prompts = _read_prompt_prefix(Path(str(entry.get("path", ""))), 200)
        if len(prompts) != 200:
            raise ValueError("target control requires durable exact-200 prompt files")
        for index, (_, prompt) in enumerate(prompts):
            hashes.append(
                hashlib.sha256(_completion_request_body(model, subset, index, prompt)).hexdigest()
            )
    return hashes


def _post_completion(endpoint: str, body: dict[str, Any]) -> dict[str, Any]:
    request = urllib.request.Request(
        endpoint.rstrip("/") + "/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=600) as response:
        result = json.loads(response.read())
    if not isinstance(result, dict):
        raise ValueError("completion response is not an object")
    return result


def _probe_choice(result: dict[str, Any]) -> dict[str, Any]:
    choices = result.get("choices")
    if not isinstance(choices, list) or len(choices) != 1 or not isinstance(choices[0], dict):
        raise ValueError("probe response must contain exactly one choice")
    choice = choices[0]
    token_ids = choice.get("token_ids")
    prompt_token_ids = choice.get("prompt_token_ids")
    logprobs = choice.get("logprobs")
    if (
        not isinstance(choice.get("text"), str)
        or not isinstance(token_ids, list)
        or not all(isinstance(value, int) and not isinstance(value, bool) for value in token_ids)
        or not isinstance(prompt_token_ids, list)
        or not all(
            isinstance(value, int) and not isinstance(value, bool) for value in prompt_token_ids
        )
        or not isinstance(logprobs, dict)
        or not isinstance(logprobs.get("top_logprobs"), list)
    ):
        raise ValueError("probe response token/logprob schema mismatch")
    return choice


def _validate_spec_decode_metrics_payload(speculative: object) -> dict[str, Any]:
    expected = {
        "mean_acceptance_length",
        "draft_acceptance_rate",
        "acceptance_histogram",
        "num_spec_steps",
        "num_accepted_draft_tokens",
        "num_draft_tokens",
        "num_spec_tokens",
        "per_step_accepted",
        "per_step_drafted",
    }
    if not isinstance(speculative, dict) or set(speculative) != expected:
        raise ValueError("detailed speculative metrics schema mismatch")
    histogram = speculative["acceptance_histogram"]
    accepted = speculative["per_step_accepted"]
    drafted = speculative["per_step_drafted"]
    integer_fields = (
        "num_spec_steps",
        "num_accepted_draft_tokens",
        "num_draft_tokens",
        "num_spec_tokens",
    )
    if (
        any(
            isinstance(speculative[name], bool)
            or not isinstance(speculative[name], int)
            or speculative[name] < 0
            for name in integer_fields
        )
        or speculative["num_spec_tokens"] != DFLASH2_SPECULATIVE_TOKENS
        or not isinstance(histogram, list)
        or len(histogram) != DFLASH2_SPECULATIVE_TOKENS + 1
        or not all(
            isinstance(value, int) and not isinstance(value, bool) and value >= 0
            for value in histogram
        )
    ):
        raise ValueError("detailed speculative metrics values mismatch")
    steps = speculative["num_spec_steps"]
    accepted_total = speculative["num_accepted_draft_tokens"]
    drafted_total = speculative["num_draft_tokens"]
    mean = speculative["mean_acceptance_length"]
    rate = speculative["draft_acceptance_rate"]
    if steps == 0:
        if (
            accepted not in (None, [])
            or drafted not in (None, [])
            or any(histogram)
            or accepted_total != 0
            or drafted_total != 0
            or mean != 1.0
            or rate != 0.0
        ):
            raise ValueError("detailed speculative metrics accounting mismatch")
        raise ValueError("speculative diagnostic requires nonzero speculative steps")
    if (
        not isinstance(accepted, list)
        or not isinstance(drafted, list)
        or not all(
            isinstance(value, int) and not isinstance(value, bool) and value >= 0
            for value in (*accepted, *drafted)
        )
        or len(accepted) != len(drafted)
        or len(accepted) != steps
        or sum(histogram) != steps
        or sum(index * count for index, count in enumerate(histogram)) != accepted_total
        or collections.Counter(accepted)
        != collections.Counter({index: count for index, count in enumerate(histogram) if count})
        or sum(accepted) != accepted_total
        or sum(drafted) != drafted_total
        or any(
            accepted_count > drafted_count or drafted_count > DFLASH2_SPECULATIVE_TOKENS
            for accepted_count, drafted_count in zip(accepted, drafted, strict=True)
        )
        or not isinstance(mean, (int, float))
        or isinstance(mean, bool)
        or not math.isfinite(mean)
        or not math.isclose(float(mean), 1 + accepted_total / steps, rel_tol=0, abs_tol=1e-12)
        or not isinstance(rate, (int, float))
        or isinstance(rate, bool)
        or not math.isfinite(rate)
        or not math.isclose(
            float(rate),
            accepted_total / drafted_total if drafted_total else 0.0,
            rel_tol=0,
            abs_tol=1e-12,
        )
    ):
        raise ValueError("detailed speculative metrics accounting mismatch")
    return speculative


def _validated_spec_decode_metrics(result: dict[str, Any]) -> dict[str, Any]:
    metrics = result.get("metrics")
    speculative = metrics.get("speculative_decoding") if isinstance(metrics, dict) else None
    return _validate_spec_decode_metrics_payload(speculative)


def capture_divergence_probe(
    dataset_manifest_path: Path,
    hf_home: Path,
    output_path: Path,
    *,
    endpoint: str,
    model: str,
    method: str,
) -> dict[str, Any]:
    """Capture five exact prompts and replay every target prefix for next-token evidence."""
    if method not in {"baseline", "dflash2"}:
        raise ValueError("probe method must be baseline or dflash2")
    prompt_set = compute_prompt_set(dataset_manifest_path, hf_home)
    files = prompt_set.get("files")
    if not isinstance(files, dict):
        raise ValueError("probe dataset files are invalid")
    records: list[dict[str, Any]] = []
    for subset, index in PROBE_ROWS:
        entry = files.get(subset)
        if not isinstance(entry, dict):
            raise ValueError(f"probe subset is missing: {subset}")
        prompts = _read_prompt_prefix(Path(str(entry.get("path", ""))), 200)
        source_row, prompt = prompts[index]
        body: dict[str, Any] = {
            "model": model,
            "prompt": prompt,
            "max_tokens": 64,
            "temperature": 0,
            "top_p": 1,
            "seed": 42,
            "logprobs": 20,
            "return_tokens_as_token_ids": True,
            "return_token_ids": True,
            "request_id": f"specdec-divergence-{subset}-{index}",
        }
        choice = _probe_choice(_post_completion(endpoint, body))
        token_ids = choice["token_ids"]
        prompt_token_ids = choice["prompt_token_ids"]
        assert isinstance(token_ids, list)
        assert isinstance(prompt_token_ids, list)
        replays: list[dict[str, Any]] = []
        if method == "baseline":
            for position, expected_token_id in enumerate(token_ids):
                replay_body = {
                    "model": model,
                    "prompt": prompt_token_ids + token_ids[:position],
                    "max_tokens": 1,
                    "temperature": 0,
                    "top_p": 1,
                    "seed": 42,
                    "logprobs": 20,
                    "return_tokens_as_token_ids": True,
                    "return_token_ids": True,
                    "request_id": f"specdec-prefix-replay-{subset}-{index}-{position}",
                }
                replay = _probe_choice(_post_completion(endpoint, replay_body))
                replay_ids = replay["token_ids"]
                replay_logprobs = replay["logprobs"]
                assert isinstance(replay_ids, list)
                assert isinstance(replay_logprobs, dict)
                top = replay_logprobs["top_logprobs"]
                if len(replay_ids) != 1 or not isinstance(top, list) or len(top) != 1:
                    raise ValueError(
                        "target prefix replay did not return one next-token distribution"
                    )
                replays.append(
                    {
                        "position": position,
                        "expected_token_id": expected_token_id,
                        "token_id": replay_ids[0],
                        "top_logprobs": top[0],
                        "request_sha256": _sha_json(replay_body),
                    }
                )
        records.append(
            {
                "subset": subset,
                "index": index,
                "source_row": source_row,
                "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
                "request_sha256": _sha_json(body),
                "output_text": choice["text"],
                "token_ids": token_ids,
                "prompt_token_ids": prompt_token_ids,
                "replays": replays,
            }
        )
    payload: dict[str, Any] = {
        "schema_version": 1,
        "producer": "q30-dflash2-first-token-probe-v1",
        "method": method,
        "model": model,
        "dataset_prompt_sha256": prompt_set["prompt_sha256"],
        "sampling": {"temperature": 0, "top_p": 1, "seed": 42, "logprobs": 20},
        "records": records,
    }
    payload["receipt_sha256"] = _sha_json(payload)
    _atomic_json(output_path, payload, no_replace=True)
    return payload


def _validate_probe_payload(path: Path, expected_method: str) -> dict[str, Any]:
    payload = _load_json(path)
    unsigned = {key: value for key, value in payload.items() if key != "receipt_sha256"}
    if (
        set(payload)
        != {
            "schema_version",
            "producer",
            "method",
            "model",
            "dataset_prompt_sha256",
            "sampling",
            "records",
            "receipt_sha256",
        }
        or payload.get("schema_version") != 1
        or payload.get("producer") != "q30-dflash2-first-token-probe-v1"
        or payload.get("method") != expected_method
        or payload.get("receipt_sha256") != _sha_json(unsigned)
        or payload.get("sampling") != {"temperature": 0, "top_p": 1, "seed": 42, "logprobs": 20}
        or not isinstance(payload.get("records"), list)
        or len(payload["records"]) != len(PROBE_ROWS)
    ):
        raise ValueError("invalid divergence probe receipt")
    return payload


def _validate_probe_against_identity(
    payload: dict[str, Any], identity: dict[str, Any], expected_method: str
) -> None:
    target = identity.get("target")
    dataset = identity.get("dataset")
    if not isinstance(target, dict) or not isinstance(dataset, dict):
        raise ValueError("probe artifact identity mismatch")
    schedule = dataset.get("ordered_prompts")
    if (
        payload.get("model") != target.get("path")
        or payload.get("dataset_prompt_sha256") != dataset.get("prompt_sha256")
        or not isinstance(schedule, list)
    ):
        raise ValueError("probe input identity mismatch")
    expected_rows = {
        (row.get("subset"), row.get("index")): row
        for row in schedule
        if isinstance(row, dict) and (row.get("subset"), row.get("index")) in PROBE_ROWS
    }
    records = payload["records"]
    assert isinstance(records, list)
    for expected_key, record in zip(PROBE_ROWS, records, strict=True):
        if not isinstance(record, dict) or set(record) != {
            "subset",
            "index",
            "source_row",
            "prompt_sha256",
            "request_sha256",
            "output_text",
            "token_ids",
            "prompt_token_ids",
            "replays",
        }:
            raise ValueError("probe record schema mismatch")
        expected = expected_rows.get(expected_key)
        token_ids = record["token_ids"]
        prompt_token_ids = record["prompt_token_ids"]
        replays = record["replays"]
        if (
            not isinstance(expected, dict)
            or (record["subset"], record["index"]) != expected_key
            or record["source_row"] != expected.get("source_row")
            or record["prompt_sha256"] != expected.get("prompt_sha256")
            or not isinstance(record["output_text"], str)
            or not isinstance(token_ids, list)
            or not token_ids
            or not all(
                isinstance(value, int) and not isinstance(value, bool) for value in token_ids
            )
            or not isinstance(prompt_token_ids, list)
            or not prompt_token_ids
            or not all(
                isinstance(value, int) and not isinstance(value, bool) for value in prompt_token_ids
            )
            or not isinstance(replays, list)
        ):
            raise ValueError("probe record identity mismatch")
        entry = dataset["files"][expected_key[0]]
        prompt = _read_prompt_prefix(Path(str(entry["path"])), 200)[expected_key[1]][1]
        main_body = {
            "model": target["path"],
            "prompt": prompt,
            "max_tokens": 64,
            "temperature": 0,
            "top_p": 1,
            "seed": 42,
            "logprobs": 20,
            "return_tokens_as_token_ids": True,
            "return_token_ids": True,
            "request_id": f"specdec-divergence-{expected_key[0]}-{expected_key[1]}",
        }
        if record["request_sha256"] != _sha_json(main_body):
            raise ValueError("probe main request mismatch")
        if expected_method == "dflash2":
            if replays:
                raise ValueError("DFlash2 probe must not claim target replay evidence")
            continue
        if len(replays) != len(token_ids):
            raise ValueError("target replay schedule is incomplete")
        for position, replay in enumerate(replays):
            if not isinstance(replay, dict) or set(replay) != {
                "position",
                "expected_token_id",
                "token_id",
                "top_logprobs",
                "request_sha256",
            }:
                raise ValueError("target replay schema mismatch")
            top = replay["top_logprobs"]
            replay_token_id = replay["token_id"]
            replay_body = {
                "model": target["path"],
                "prompt": prompt_token_ids + token_ids[:position],
                "max_tokens": 1,
                "temperature": 0,
                "top_p": 1,
                "seed": 42,
                "logprobs": 20,
                "return_tokens_as_token_ids": True,
                "return_token_ids": True,
                "request_id": (
                    f"specdec-prefix-replay-{expected_key[0]}-{expected_key[1]}-{position}"
                ),
            }
            if (
                replay["position"] != position
                or replay["expected_token_id"] != token_ids[position]
                or not isinstance(replay_token_id, int)
                or isinstance(replay_token_id, bool)
                or replay_token_id < 0
                or replay["request_sha256"] != _sha_json(replay_body)
                or not isinstance(top, dict)
                or not top
                or any(
                    not isinstance(key, str)
                    or not key.startswith("token_id:")
                    or not isinstance(value, (int, float))
                    or isinstance(value, bool)
                    or not math.isfinite(value)
                    for key, value in top.items()
                )
                or f"token_id:{replay_token_id}" not in top
                or float(top[f"token_id:{replay_token_id}"]) != max(map(float, top.values()))
            ):
                raise ValueError("target prefix replay evidence mismatch")


def analyze_divergence_probe(baseline_path: Path, dflash2_path: Path) -> dict[str, Any]:
    """Find the first token divergence and replayed target rank for each exact prompt."""
    baseline = _validate_probe_payload(baseline_path, "baseline")
    draft = _validate_probe_payload(dflash2_path, "dflash2")
    if baseline.get("model") != draft.get("model") or baseline.get(
        "dataset_prompt_sha256"
    ) != draft.get("dataset_prompt_sha256"):
        raise ValueError("probe inputs are not matched")
    output: list[dict[str, Any]] = []
    for expected_key, target, proposed in zip(
        PROBE_ROWS, baseline["records"], draft["records"], strict=True
    ):
        if not isinstance(target, dict) or not isinstance(proposed, dict):
            raise ValueError("invalid probe record")
        key = (target.get("subset"), target.get("index"))
        if (
            key != expected_key
            or (proposed.get("subset"), proposed.get("index")) != expected_key
            or target.get("prompt_sha256") != proposed.get("prompt_sha256")
            or target.get("request_sha256") != proposed.get("request_sha256")
        ):
            raise ValueError("probe row identity mismatch")
        target_ids = target.get("token_ids")
        draft_ids = proposed.get("token_ids")
        replays = target.get("replays")
        if (
            not isinstance(target_ids, list)
            or not isinstance(draft_ids, list)
            or not isinstance(replays, list)
        ):
            raise ValueError("probe token evidence mismatch")
        common = 0
        while (
            common < min(len(target_ids), len(draft_ids))
            and target_ids[common] == draft_ids[common]
        ):
            common += 1
        if common == len(target_ids) == len(draft_ids):
            output.append({"subset": key[0], "index": key[1], "status": "exact-match"})
            continue
        if common >= len(target_ids) or common >= len(draft_ids) or common >= len(replays):
            raise ValueError("probe divergence lacks target replay evidence")
        replay = replays[common]
        target_token = target_ids[common]
        draft_token = draft_ids[common]
        if (
            not isinstance(replay, dict)
            or replay.get("position") != common
            or replay.get("expected_token_id", replay.get("token_id")) != target_token
            or not isinstance(replay.get("top_logprobs"), dict)
        ):
            raise ValueError("target replay evidence does not match the common prefix")
        top_logprobs = replay["top_logprobs"]
        assert isinstance(top_logprobs, dict)
        draft_key = f"token_id:{draft_token}"
        target_key = f"token_id:{target_token}"
        maximum_logprob = max(float(value) for value in top_logprobs.values())
        replay_key = f"token_id:{replay['token_id']}"
        replay_logprob = top_logprobs.get(replay_key)
        if (
            not isinstance(replay_logprob, (int, float))
            or isinstance(replay_logprob, bool)
            or float(replay_logprob) != maximum_logprob
        ):
            raise ValueError("target prefix replay did not emit an argmax at temperature zero")
        target_logprob = top_logprobs.get(target_key)
        target_is_argmax = (
            isinstance(target_logprob, (int, float))
            and not isinstance(target_logprob, bool)
            and float(target_logprob) == maximum_logprob
        )
        draft_logprob = top_logprobs.get(draft_key)
        draft_rank = None
        draft_is_argmax = False
        if isinstance(draft_logprob, (int, float)) and not isinstance(draft_logprob, bool):
            draft_value = float(draft_logprob)
            draft_rank = 1 + sum(float(value) > draft_value for value in top_logprobs.values())
            draft_is_argmax = draft_value == maximum_logprob
        output.append(
            {
                "subset": key[0],
                "index": key[1],
                "status": "diverged",
                "first_divergence_position": common,
                "target_token_id": target_token,
                "dflash2_token_id": draft_token,
                "replay_token_id": replay["token_id"],
                "target_token_is_replay_argmax": target_is_argmax,
                "dflash2_token_is_target_argmax": draft_is_argmax,
                "dflash2_token_target_rank": draft_rank,
                "target_top_logprobs": top_logprobs,
                "verdict": (
                    "target-valid-argmax"
                    if target_is_argmax and draft_is_argmax
                    else "target-invalid"
                    if target_is_argmax
                    else "replay-inconclusive"
                ),
            }
        )
    return {"schema_version": 1, "producer": "q30-dflash2-first-divergence-v1", "records": output}


def _validate_dflash2_probe_manifest(
    path: Path,
    baseline: dict[str, Any],
    baseline_fingerprint: dict[str, Any],
    artifact_identity_path: Path,
    identity: dict[str, Any],
    *,
    detailed_metrics: bool = False,
    enforce_eager: bool = False,
    disable_prefix_caching: bool = False,
) -> tuple[dict[str, Any], Path, Path]:
    manifest = _load_json(path)
    target = identity["target"]
    draft = identity["draft"]
    evaluation = manifest.get("evaluation")
    server_args = manifest.get("server_args")
    config = manifest.get("config_sha256")
    common = (
        "target_model",
        "speculators_repo",
        "speculators_sha",
        "modelopt_repo",
        "modelopt_sha",
        "modelopt_dirty",
        "runtime",
        "runtimes",
        "container",
        "dataset",
        "provenance_error",
        "slurm_job_id",
        "versions",
        "evaluation",
        "artifact_identity",
    )
    expected_server_args = _expected_dflash2_server_args(
        target["path"],
        draft["export_path"],
        server_args[7] if isinstance(server_args, list) and len(server_args) > 7 else "",
        detailed_metrics=detailed_metrics,
        enforce_eager=enforce_eager,
        disable_prefix_caching=disable_prefix_caching,
    )
    if (
        set(manifest) != _CONTROL_MANIFEST_KEYS
        or manifest.get("status") != "success"
        or manifest.get("method") != "dflash2"
        or manifest.get("block_size") != DFLASH2_BLOCK_SIZE
        or manifest.get("num_speculative_tokens") != DFLASH2_SPECULATIVE_TOKENS
        or manifest.get("draft_model") != draft["export_path"]
        or manifest.get("evaluator_args") != []
        or any(manifest.get(name) != baseline.get(name) for name in common)
        or not isinstance(evaluation, dict)
        or not isinstance(server_args, list)
        or server_args != expected_server_args
        or server_args[7] not in {"8000", "8010"}
        or not isinstance(config, dict)
        or set(config) != {"target", "draft", "launcher"}
        or config.get("target") != _sha256(Path(target["path"]) / "config.json")
        or config.get("draft") != _sha256(Path(draft["export_path"]) / "config.json")
        or manifest.get("artifact_identity")
        != {
            "path": str(artifact_identity_path.resolve(strict=True)),
            "sha256": _sha256(artifact_identity_path),
        }
    ):
        raise ValueError("DFlash2 probe manifest mismatch")
    launcher = _validate_control_launcher(
        manifest.get("launcher_config"),
        Path(str(manifest["container"]["path"])),
        config["launcher"],
    )
    fingerprint_path = path.parent / "input-fingerprint.json"
    fingerprint = _load_json(fingerprint_path)
    baseline_inputs = baseline_fingerprint.get("inputs")
    if not isinstance(baseline_inputs, dict):
        raise ValueError("baseline probe fingerprint mismatch")
    expected_inputs = json.loads(json.dumps(baseline_inputs))
    expected_inputs["draft_config_sha256"] = config["draft"]
    expected_inputs["evaluation"] = {
        **expected_inputs["evaluation"],
        "method": "dflash2",
        "block_size": DFLASH2_BLOCK_SIZE,
        "num_speculative_tokens": DFLASH2_SPECULATIVE_TOKENS,
    }
    if (
        fingerprint.get("schema_version") != 1
        or fingerprint.get("inputs") != expected_inputs
        or fingerprint.get("sha256") != _sha_json(expected_inputs)
    ):
        raise ValueError("DFlash2 probe input fingerprint mismatch")
    return manifest, launcher, fingerprint_path


def _validate_dflash_control_manifest(
    path: Path,
    baseline: dict[str, Any],
    baseline_fingerprint: dict[str, Any],
    artifact_identity_path: Path,
    identity: dict[str, Any],
    control_stage_receipt_path: Path | None = None,
    *,
    require_live_stage: bool = False,
) -> tuple[dict[str, Any], Path, Path, dict[str, Any]]:
    """Validate the exact OPB DFlash B8/K7 eager/no-prefix control manifest."""
    manifest = _load_json(path)
    target = identity["target"]
    draft_model = manifest.get("draft_model")
    if not isinstance(draft_model, str):
        raise ValueError("DFlash control manifest mismatch")
    if control_stage_receipt_path is None:
        control_artifact = _validate_opb_dflash_control_artifact(Path(draft_model))
        draft_artifact = control_artifact
    else:
        stage_receipt = validate_opb_dflash_stage_receipt(
            control_stage_receipt_path,
            require_live_stage=require_live_stage,
            expected_staged_draft=Path(draft_model),
        )
        draft_artifact = stage_receipt["staged"]
        control_artifact = {
            "source": stage_receipt["source"],
            "staged": stage_receipt["staged"],
            "stage_receipt": _file_descriptor(control_stage_receipt_path),
        }
    draft_config_sha256 = (
        _sha256(Path(draft_artifact["path"]) / "config.json")
        if control_stage_receipt_path is None or require_live_stage
        else draft_artifact["file_sha256"]["config.json"]
    )
    server_args = manifest.get("server_args")
    config = manifest.get("config_sha256")
    common = (
        "target_model",
        "speculators_repo",
        "speculators_sha",
        "modelopt_repo",
        "modelopt_sha",
        "modelopt_dirty",
        "runtime",
        "runtimes",
        "container",
        "dataset",
        "provenance_error",
        "slurm_job_id",
        "versions",
        "evaluation",
        "artifact_identity",
    )
    expected_server_args = _expected_dflash2_server_args(
        target["path"],
        draft_artifact["path"],
        server_args[7] if isinstance(server_args, list) and len(server_args) > 7 else "",
        detailed_metrics=True,
        enforce_eager=True,
        disable_prefix_caching=True,
    )
    if (
        set(manifest) != _CONTROL_MANIFEST_KEYS
        or manifest.get("status") != "success"
        or manifest.get("method") != "dflash"
        or manifest.get("block_size") != DFLASH2_BLOCK_SIZE
        or manifest.get("num_speculative_tokens") != DFLASH2_SPECULATIVE_TOKENS
        or manifest.get("draft_model") != draft_artifact["path"]
        or manifest.get("evaluator_args") != []
        or any(manifest.get(name) != baseline.get(name) for name in common)
        or not isinstance(server_args, list)
        or server_args != expected_server_args
        or server_args[7] not in {"8000", "8010"}
        or not isinstance(config, dict)
        or set(config) != {"target", "draft", "launcher"}
        or config.get("target") != _sha256(Path(target["path"]) / "config.json")
        or config.get("draft") != draft_config_sha256
        or manifest.get("artifact_identity")
        != {
            "path": str(artifact_identity_path.resolve(strict=True)),
            "sha256": _sha256(artifact_identity_path),
        }
    ):
        raise ValueError("DFlash control manifest mismatch")
    launcher = _validate_control_launcher(
        manifest.get("launcher_config"),
        Path(str(manifest["container"]["path"])),
        config["launcher"],
    )
    fingerprint_path = path.parent / "input-fingerprint.json"
    fingerprint = _load_json(fingerprint_path)
    baseline_inputs = baseline_fingerprint.get("inputs")
    if not isinstance(baseline_inputs, dict):
        raise ValueError("baseline control fingerprint mismatch")
    expected_inputs = json.loads(json.dumps(baseline_inputs))
    expected_inputs["draft_config_sha256"] = config["draft"]
    expected_inputs["evaluation"] = {
        **expected_inputs["evaluation"],
        "method": "dflash",
        "block_size": DFLASH2_BLOCK_SIZE,
        "num_speculative_tokens": DFLASH2_SPECULATIVE_TOKENS,
    }
    if (
        fingerprint.get("schema_version") != 1
        or fingerprint.get("inputs") != expected_inputs
        or fingerprint.get("sha256") != _sha_json(expected_inputs)
    ):
        raise ValueError("DFlash control input fingerprint mismatch")
    return manifest, launcher, fingerprint_path, control_artifact


def build_divergence_probe_receipt(
    baseline_probe_path: Path,
    dflash2_probe_path: Path,
    baseline_manifest_path: Path,
    dflash2_manifest_path: Path,
    artifact_identity_path: Path,
    allocation_receipt_path: Path,
    output_path: Path,
) -> dict[str, Any]:
    """Bind exact paired job evidence to the first-token divergence diagnosis."""
    identity = _validate_artifact_identity(artifact_identity_path)
    baseline_probe = _validate_probe_payload(baseline_probe_path, "baseline")
    dflash_probe = _validate_probe_payload(dflash2_probe_path, "dflash2")
    _validate_probe_against_identity(baseline_probe, identity, "baseline")
    _validate_probe_against_identity(dflash_probe, identity, "dflash2")
    baseline_evidence = _validate_target_control_manifest(
        baseline_manifest_path, artifact_identity_path, identity
    )
    baseline_manifest = baseline_evidence[0]
    dflash_manifest, dflash_launcher, dflash_fingerprint = _validate_dflash2_probe_manifest(
        dflash2_manifest_path,
        baseline_manifest,
        baseline_evidence[2],
        artifact_identity_path,
        identity,
    )
    allocation = validate_target_control_allocation_receipt(allocation_receipt_path)
    current = _query_current_allocation()
    for name in (
        "slurm_job_id",
        "slurm_job_num_nodes",
        "slurm_job_nodelist",
        "gpu_count",
        "cell_visible_devices",
    ):
        if allocation.get(name) != current.get(name):
            raise ValueError(f"divergence probe live allocation mismatch: {name}")
    if (
        baseline_manifest["slurm_job_id"] != allocation["slurm_job_id"]
        or dflash_manifest["slurm_job_id"] != allocation["slurm_job_id"]
    ):
        raise ValueError("divergence probe job ID mismatch")
    payload = analyze_divergence_probe(baseline_probe_path, dflash2_probe_path)
    payload["claim_scope"] = "token-correctness diagnosis only; no speedup claim"
    payload["allocation_evidence_scope"] = (
        "live SLURM/GPU origin checked at creation; offline verification is tamper replay"
    )
    payload["artifact_identity"] = _file_descriptor(artifact_identity_path)
    payload["allocation_receipt"] = _file_descriptor(allocation_receipt_path)
    payload["probes"] = {
        "baseline": _file_descriptor(baseline_probe_path),
        "dflash2": _file_descriptor(dflash2_probe_path),
    }
    payload["manifests"] = {
        "baseline": _file_descriptor(baseline_manifest_path),
        "dflash2": _file_descriptor(dflash2_manifest_path),
    }
    payload["input_fingerprints"] = {
        "baseline": _file_descriptor(baseline_evidence[1]),
        "dflash2": _file_descriptor(dflash_fingerprint),
    }
    payload["launcher_configs"] = {
        "baseline": _file_descriptor(baseline_evidence[3]),
        "dflash2": _file_descriptor(dflash_launcher),
    }
    payload["receipt_sha256"] = _sha_json(payload)
    _atomic_json(output_path, payload, no_replace=True)
    return payload


def validate_divergence_probe_receipt(path: Path) -> dict[str, Any]:
    """Replay all durable first-divergence evidence without claiming scheduler origin."""
    payload = _load_json(path)
    claim = payload.pop("receipt_sha256", None)
    if claim != _sha_json(payload):
        raise ValueError("divergence probe receipt self-hash mismatch")
    probes_value = payload.get("probes")
    manifests_value = payload.get("manifests")
    launchers_value = payload.get("launcher_configs")
    fingerprints_value = payload.get("input_fingerprints")
    if not all(
        isinstance(value, dict) and set(value) == {"baseline", "dflash2"}
        for value in (probes_value, manifests_value, launchers_value, fingerprints_value)
    ):
        raise ValueError("divergence probe evidence schema mismatch")
    assert isinstance(probes_value, dict)
    assert isinstance(manifests_value, dict)
    assert isinstance(launchers_value, dict)
    assert isinstance(fingerprints_value, dict)
    probes = {name: _validate_file_descriptor(value) for name, value in probes_value.items()}
    manifests = {name: _validate_file_descriptor(value) for name, value in manifests_value.items()}
    launchers = {name: _validate_file_descriptor(value) for name, value in launchers_value.items()}
    fingerprints = {
        name: _validate_file_descriptor(value) for name, value in fingerprints_value.items()
    }
    artifact_path = _validate_file_descriptor(payload.get("artifact_identity"))
    allocation_path = _validate_file_descriptor(payload.get("allocation_receipt"))
    allocation = validate_target_control_allocation_receipt(allocation_path)
    identity = _validate_artifact_identity(artifact_path)
    baseline_probe = _validate_probe_payload(probes["baseline"], "baseline")
    dflash_probe = _validate_probe_payload(probes["dflash2"], "dflash2")
    _validate_probe_against_identity(baseline_probe, identity, "baseline")
    _validate_probe_against_identity(dflash_probe, identity, "dflash2")
    baseline_evidence = _validate_target_control_manifest(
        manifests["baseline"], artifact_path, identity
    )
    _, dflash_launcher, dflash_fingerprint = _validate_dflash2_probe_manifest(
        manifests["dflash2"],
        baseline_evidence[0],
        baseline_evidence[2],
        artifact_path,
        identity,
    )
    if any(
        _load_json(manifest)["slurm_job_id"] != allocation["slurm_job_id"]
        for manifest in manifests.values()
    ):
        raise ValueError("divergence probe allocation replay mismatch")
    if launchers != {"baseline": baseline_evidence[3], "dflash2": dflash_launcher}:
        raise ValueError("divergence probe launcher evidence mismatch")
    if fingerprints != {
        "baseline": baseline_evidence[1],
        "dflash2": dflash_fingerprint,
    }:
        raise ValueError("divergence probe fingerprint evidence mismatch")
    replayed = analyze_divergence_probe(probes["baseline"], probes["dflash2"])
    for name in ("schema_version", "producer", "records"):
        if payload.get(name) != replayed.get(name):
            raise ValueError(f"divergence probe replay mismatch: {name}")
    if (
        payload.get("claim_scope") != "token-correctness diagnosis only; no speedup claim"
        or payload.get("allocation_evidence_scope")
        != "live SLURM/GPU origin checked at creation; offline verification is tamper replay"
    ):
        raise ValueError("divergence probe claim scope mismatch")
    return {**payload, "receipt_sha256": claim}


def classify_tie_aware_rows(target: dict[str, Any], dflash2: dict[str, Any]) -> dict[str, Any]:
    """Classify one paired row using the target's online first-divergence distribution."""
    identity_fields = (
        "subset",
        "index",
        "source_row",
        "prompt_sha256",
        "request_core_sha256",
        "prompt_token_ids",
    )
    if any(target.get(name) != dflash2.get(name) for name in identity_fields):
        raise ValueError("tie-aware paired row identity mismatch")
    target_ids = target.get("token_ids")
    dflash_ids = dflash2.get("token_ids")
    top_logprobs = target.get("top_logprobs")
    if (
        not isinstance(target_ids, list)
        or not isinstance(dflash_ids, list)
        or not isinstance(top_logprobs, list)
    ):
        return {
            "subset": target.get("subset"),
            "index": target.get("index"),
            "source_row": target.get("source_row"),
            "class": "unresolved-target-evidence",
        }
    common = 0
    while (
        common < min(len(target_ids), len(dflash_ids)) and target_ids[common] == dflash_ids[common]
    ):
        common += 1
    base = {
        "subset": target.get("subset"),
        "index": target.get("index"),
        "source_row": target.get("source_row"),
    }
    if common == len(target_ids) == len(dflash_ids):
        if len(target_ids) != len(top_logprobs):
            return {**base, "class": "unresolved-target-evidence"}
        for position, (token, distribution) in enumerate(
            zip(target_ids, top_logprobs, strict=True)
        ):
            if (
                not isinstance(distribution, dict)
                or not distribution
                or any(
                    not isinstance(key, str)
                    or re.fullmatch(r"token_id:[0-9]+", key) is None
                    or not isinstance(logprob, (int, float))
                    or isinstance(logprob, bool)
                    or not math.isfinite(logprob)
                    for key, logprob in distribution.items()
                )
            ):
                return {**base, "class": "unresolved-target-evidence", "position": position}
            emitted = distribution.get(f"token_id:{token}")
            if (
                not isinstance(emitted, (int, float))
                or isinstance(emitted, bool)
                or float(emitted) != max(map(float, distribution.values()))
            ):
                return {**base, "class": "unresolved-target-evidence", "position": position}
        if target.get("finish_reason") == dflash2.get("finish_reason"):
            return {**base, "class": "exact"}
        return {**base, "class": "unresolved-termination", "common_prefix_tokens": common}
    if common >= len(target_ids) or common >= len(dflash_ids):
        return {**base, "class": "unresolved-termination", "common_prefix_tokens": common}
    if common >= len(top_logprobs):
        return {**base, "class": "unresolved-target-evidence", "position": common}
    top = top_logprobs[common]
    target_token = target_ids[common]
    draft_token = dflash_ids[common]
    if not isinstance(top, dict) or not top:
        return {**base, "class": "unresolved-target-evidence", "position": common}
    if any(
        not isinstance(key, str)
        or not key.startswith("token_id:")
        or not isinstance(value, (int, float))
        or isinstance(value, bool)
        or not math.isfinite(value)
        for key, value in top.items()
    ):
        return {**base, "class": "unresolved-target-evidence", "position": common}
    maximum = max(float(value) for value in top.values())
    target_value = top.get(f"token_id:{target_token}")
    if (
        not isinstance(target_value, (int, float))
        or isinstance(target_value, bool)
        or float(target_value) != maximum
    ):
        return {**base, "class": "unresolved-target-evidence", "position": common}
    draft_value = top.get(f"token_id:{draft_token}")
    if not isinstance(draft_value, (int, float)) or isinstance(draft_value, bool):
        return {
            **base,
            "class": "unresolved-top20",
            "position": common,
            "target_token_id": target_token,
            "dflash2_token_id": draft_token,
        }
    draft_float = float(draft_value)
    rank = 1 + sum(float(value) > draft_float for value in top.values())
    return {
        **base,
        "class": "tied-target-valid" if draft_float == maximum else "target-invalid",
        "position": common,
        "target_token_id": target_token,
        "dflash2_token_id": draft_token,
        "dflash2_token_target_rank": rank,
        "target_max_logprob": maximum,
        "dflash2_token_logprob": draft_float,
    }


def capture_tie_aware_pilot(
    dataset_manifest_path: Path,
    hf_home: Path,
    output_path: Path,
    *,
    endpoint: str,
    model: str,
    role: str,
) -> None:
    """Stream the exact 200 HumanEval occurrences for the tie-aware pilot."""
    if role not in {"target", "dflash2"}:
        raise ValueError("tie-aware capture role must be target or dflash2")
    prompt_set = compute_prompt_set(dataset_manifest_path, hf_home)
    files = prompt_set.get("files")
    if not isinstance(files, dict) or not isinstance(files.get("HumanEval"), dict):
        raise ValueError("tie-aware HumanEval prompt evidence is missing")
    prompts = _read_prompt_prefix(Path(str(files["HumanEval"]["path"])), 200)
    if len(prompts) != 200:
        raise ValueError("tie-aware pilot requires exactly 200 HumanEval occurrences")
    if output_path.exists():
        raise FileExistsError(f"tie-aware output exists: {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{output_path.name}.", dir=output_path.parent)
    try:
        with os.fdopen(descriptor, "w") as stream:
            for index, (source_row, prompt) in enumerate(prompts):
                core = {
                    "model": model,
                    "prompt": prompt,
                    "max_tokens": 64,
                    "temperature": 0,
                    "top_p": 1,
                    "seed": 42,
                    "request_id": f"specdec-tie-pilot-HumanEval-{index}",
                }
                body = {
                    **core,
                    "logprobs": 20 if role == "target" else 0,
                    "return_tokens_as_token_ids": True,
                    "return_token_ids": True,
                }
                choice = _probe_choice(_post_completion(endpoint, body))
                finish = choice.get("finish_reason")
                logprobs = choice["logprobs"]
                assert isinstance(logprobs, dict)
                top = logprobs["top_logprobs"]
                if finish not in {"stop", "length"}:
                    raise ValueError("tie-aware completion finish reason mismatch")
                extracted = {
                    "output_text": choice["text"],
                    "token_ids": choice["token_ids"],
                    "prompt_token_ids": choice["prompt_token_ids"],
                    "finish_reason": finish,
                    "top_logprobs": top if role == "target" else [],
                }
                record = {
                    "schema_version": 1,
                    "producer": "q30-tie-aware-online-row-v1",
                    "role": role,
                    "subset": "HumanEval",
                    "index": index,
                    "source_row": source_row,
                    "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
                    "request_core_sha256": _sha_json(core),
                    "request_sha256": _sha_json(body),
                    **extracted,
                    "response_sha256": _sha_json(extracted),
                }
                stream.write(_canonical(record) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, output_path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _valid_top_logprobs(value: object) -> bool:
    return (
        isinstance(value, dict)
        and bool(value)
        and all(
            isinstance(key, str)
            and re.fullmatch(r"token_id:[0-9]+", key) is not None
            and isinstance(logprob, (int, float))
            and not isinstance(logprob, bool)
            and math.isfinite(logprob)
            for key, logprob in value.items()
        )
    )


def capture_internal_target_diagnostic(
    dataset_manifest_path: Path,
    hf_home: Path,
    output_path: Path,
    *,
    endpoint: str,
    model: str,
    role: str,
    engine_mode: str,
) -> None:
    """Capture the fixed RCA rows with online logits and detailed acceptance evidence."""
    if role not in {"target", "dflash", "dflash2"}:
        raise ValueError("internal-target role must be target, dflash, or dflash2")
    if (
        (role == "target" and engine_mode != "compiled")
        or (role == "dflash" and engine_mode != "eager-no-prefix")
        or (role == "dflash2" and engine_mode not in {"compiled", "eager", "eager-no-prefix"})
    ):
        raise ValueError("internal-target engine mode mismatch")
    prompt_set = compute_prompt_set(dataset_manifest_path, hf_home)
    files = prompt_set.get("files")
    if not isinstance(files, dict) or not isinstance(files.get("HumanEval"), dict):
        raise ValueError("internal-target HumanEval prompt evidence is missing")
    prompts = _read_prompt_prefix(Path(str(files["HumanEval"]["path"])), 200)
    if len(prompts) != 200:
        raise ValueError("internal-target diagnostic requires exact-200 HumanEval schedule")
    if output_path.exists():
        raise FileExistsError(f"internal-target output exists: {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{output_path.name}.", dir=output_path.parent)
    try:
        with os.fdopen(descriptor, "w") as stream:
            for index, reason in INTERNAL_TARGET_DIAGNOSTIC_ROWS:
                source_row, prompt = prompts[index]
                body = {
                    "model": model,
                    "prompt": prompt,
                    "max_tokens": 64,
                    "temperature": 0,
                    "top_p": 1,
                    "seed": 42,
                    "logprobs": 20,
                    "return_tokens_as_token_ids": True,
                    "return_token_ids": True,
                    "request_id": f"specdec-internal-target-HumanEval-{index}",
                }
                result = _post_completion(endpoint, body)
                choice = _probe_choice(result)
                finish = choice.get("finish_reason")
                logprobs = choice["logprobs"]
                assert isinstance(logprobs, dict)
                top = logprobs["top_logprobs"]
                token_ids = choice["token_ids"]
                assert isinstance(token_ids, list)
                if (
                    finish not in {"stop", "length"}
                    or not isinstance(top, list)
                    or len(top) != len(token_ids)
                    or not all(_valid_top_logprobs(distribution) for distribution in top)
                ):
                    raise ValueError("internal-target completion evidence mismatch")
                speculative = _validated_spec_decode_metrics(result) if role != "target" else None
                if (
                    role == "target"
                    and isinstance(result.get("metrics"), dict)
                    and result["metrics"].get("speculative_decoding") is not None
                ):
                    raise ValueError(
                        "target-only response unexpectedly contains speculative metrics"
                    )
                extracted = {
                    "output_text": choice["text"],
                    "token_ids": token_ids,
                    "prompt_token_ids": choice["prompt_token_ids"],
                    "finish_reason": finish,
                    "top_logprobs": top,
                    "speculative_decoding": speculative,
                }
                record = {
                    "schema_version": 1,
                    "producer": (
                        "q30-dflash-internal-target-row-v1"
                        if role == "dflash"
                        else "q30-dflash2-internal-target-row-v1"
                    ),
                    "role": role,
                    "engine_mode": engine_mode,
                    "subset": "HumanEval",
                    "index": index,
                    "source_row": source_row,
                    "selection_reason": reason,
                    "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
                    "request_sha256": _sha_json(body),
                    **extracted,
                    "response_sha256": _sha_json(extracted),
                }
                stream.write(_canonical(record) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, output_path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _read_internal_target_rows(path: Path, expected_role: str) -> list[dict[str, Any]]:
    if expected_role not in {"target", "dflash", "dflash2"}:
        raise ValueError("invalid internal-target role")
    expected_producer = (
        "q30-dflash-internal-target-row-v1"
        if expected_role == "dflash"
        else "q30-dflash2-internal-target-row-v1"
    )
    expected_keys = {
        "schema_version",
        "producer",
        "role",
        "engine_mode",
        "subset",
        "index",
        "source_row",
        "selection_reason",
        "prompt_sha256",
        "request_sha256",
        "output_text",
        "token_ids",
        "prompt_token_ids",
        "finish_reason",
        "top_logprobs",
        "speculative_decoding",
        "response_sha256",
    }
    rows: list[dict[str, Any]] = []
    with path.open() as stream:
        for line in stream:
            value = json.loads(line)
            if not isinstance(value, dict) or set(value) != expected_keys:
                raise ValueError("internal-target row schema mismatch")
            tokens = value.get("token_ids")
            prompt_tokens = value.get("prompt_token_ids")
            top = value.get("top_logprobs")
            extracted = {
                "output_text": value.get("output_text"),
                "token_ids": tokens,
                "prompt_token_ids": prompt_tokens,
                "finish_reason": value.get("finish_reason"),
                "top_logprobs": top,
                "speculative_decoding": value.get("speculative_decoding"),
            }
            if (
                value.get("schema_version") != 1
                or value.get("producer") != expected_producer
                or value.get("role") != expected_role
                or value.get("engine_mode") not in {"compiled", "eager", "eager-no-prefix"}
                or (expected_role == "target" and value.get("engine_mode") != "compiled")
                or (expected_role == "dflash" and value.get("engine_mode") != "eager-no-prefix")
                or value.get("subset") != "HumanEval"
                or isinstance(value.get("index"), bool)
                or not isinstance(value.get("index"), int)
                or isinstance(value.get("source_row"), bool)
                or not isinstance(value.get("source_row"), int)
                or value["source_row"] < 0
                or not isinstance(value.get("selection_reason"), str)
                or not _is_sha256(value.get("prompt_sha256"))
                or not _is_sha256(value.get("request_sha256"))
                or not _is_sha256(value.get("response_sha256"))
                or value["response_sha256"] != _sha_json(extracted)
                or not isinstance(value.get("output_text"), str)
                or value.get("finish_reason") not in {"stop", "length"}
                or not isinstance(tokens, list)
                or not tokens
                or not all(
                    isinstance(token, int) and not isinstance(token, bool) and token >= 0
                    for token in tokens
                )
                or not isinstance(prompt_tokens, list)
                or not prompt_tokens
                or not all(
                    isinstance(token, int) and not isinstance(token, bool) and token >= 0
                    for token in prompt_tokens
                )
                or not isinstance(top, list)
                or len(top) != len(tokens)
                or not all(_valid_top_logprobs(distribution) for distribution in top)
            ):
                raise ValueError("internal-target row evidence mismatch")
            if expected_role == "target":
                if value.get("speculative_decoding") is not None:
                    raise ValueError("target row must not contain speculative metrics")
                if any(
                    f"token_id:{token}" not in distribution
                    or float(distribution[f"token_id:{token}"])
                    != max(map(float, distribution.values()))
                    for token, distribution in zip(tokens, top, strict=True)
                ):
                    raise ValueError("target-only row did not emit its online argmax")
            else:
                _validate_spec_decode_metrics_payload(value.get("speculative_decoding"))
            rows.append(value)
    expected = list(INTERNAL_TARGET_DIAGNOSTIC_ROWS)
    if [(row["index"], row["selection_reason"]) for row in rows] != expected:
        raise ValueError("internal-target RCA selection mismatch")
    if expected_role != "target" and len({row["engine_mode"] for row in rows}) != 1:
        raise ValueError("internal-target speculative engine modes are mixed")
    return rows


def _emitted_token_verdict(
    row: dict[str, Any], position: int, *, token_label: str = "dflash2"
) -> dict[str, Any]:
    tokens = row["token_ids"]
    distributions = row["top_logprobs"]
    if position >= len(tokens) or position >= len(distributions):
        return {"class": "unresolved-termination", "position": position}
    token = tokens[position]
    distribution = distributions[position]
    assert isinstance(distribution, dict)
    maximum = max(map(float, distribution.values()))
    logprob = distribution.get(f"token_id:{token}")
    if not isinstance(logprob, (int, float)) or isinstance(logprob, bool):
        return {
            "class": "unresolved-internal-top20",
            "position": position,
            f"{token_label}_token_id": token,
            "internal_target_max_logprob": maximum,
        }
    value = float(logprob)
    rank = 1 + sum(float(item) > value for item in distribution.values())
    return {
        "class": (
            "speculative-internal-argmax"
            if value == maximum
            else "internal-target-consistency-mismatch"
        ),
        "position": position,
        f"{token_label}_token_id": token,
        f"{token_label}_token_internal_rank": rank,
        "internal_target_max_logprob": maximum,
        f"{token_label}_token_internal_logprob": value,
    }


def _paired_target_verdict(
    target_row: dict[str, Any],
    speculative_token_id: int,
    position: int,
    *,
    token_label: str = "dflash2",
) -> dict[str, Any]:
    distributions = target_row["top_logprobs"]
    if position >= len(distributions):
        return {"class": "unresolved-target-termination", "position": position}
    distribution = distributions[position]
    assert isinstance(distribution, dict)
    maximum = max(map(float, distribution.values()))
    logprob = distribution.get(f"token_id:{speculative_token_id}")
    if not isinstance(logprob, (int, float)) or isinstance(logprob, bool):
        return {
            "class": "unresolved-target-top20",
            "position": position,
            f"{token_label}_token_id": speculative_token_id,
            "paired_target_max_logprob": maximum,
        }
    value = float(logprob)
    rank = 1 + sum(float(item) > value for item in distribution.values())
    return {
        "class": "target-rerun-argmax" if value == maximum else "paired-target-not-argmax",
        "position": position,
        f"{token_label}_token_id": speculative_token_id,
        f"{token_label}_token_paired_target_rank": rank,
        "paired_target_max_logprob": maximum,
        f"{token_label}_token_paired_target_logprob": value,
    }


def summarize_internal_target_diagnostic(
    target_path: Path,
    dflash2_path: Path,
    *,
    speculative_role: str = "dflash2",
    token_label: str = "dflash2",
) -> dict[str, Any]:
    """Classify selected outputs against target-only and speculative-engine logits."""
    target_rows = _read_internal_target_rows(target_path, "target")
    dflash_rows = _read_internal_target_rows(dflash2_path, speculative_role)
    classifications: list[dict[str, Any]] = []
    accepted_total = 0
    drafted_total = 0
    spec_steps = 0
    for target, draft in zip(target_rows, dflash_rows, strict=True):
        identity_fields = (
            "subset",
            "index",
            "source_row",
            "selection_reason",
            "prompt_sha256",
            "request_sha256",
            "prompt_token_ids",
        )
        if any(target[name] != draft[name] for name in identity_fields):
            raise ValueError("internal-target paired row identity mismatch")
        metrics = draft["speculative_decoding"]
        assert isinstance(metrics, dict)
        accepted_total += metrics["num_accepted_draft_tokens"]
        drafted_total += metrics["num_draft_tokens"]
        spec_steps += metrics["num_spec_steps"]
        target_ids = target["token_ids"]
        dflash_ids = draft["token_ids"]
        common = 0
        while (
            common < min(len(target_ids), len(dflash_ids))
            and target_ids[common] == dflash_ids[common]
        ):
            common += 1
        base = {
            "subset": "HumanEval",
            "index": target["index"],
            "source_row": target["source_row"],
            "selection_reason": target["selection_reason"],
            "engine_mode": draft["engine_mode"],
        }
        if common == len(target_ids) == len(dflash_ids):
            if target["finish_reason"] != draft["finish_reason"]:
                classifications.append(
                    {**base, "class": "unresolved-termination", "common_prefix_tokens": common}
                )
                continue
            mismatch = None
            for position in range(len(dflash_ids)):
                verdict = _emitted_token_verdict(draft, position, token_label=token_label)
                if verdict["class"] != "speculative-internal-argmax":
                    mismatch = verdict
                    break
            classifications.append({**base, **mismatch} if mismatch else {**base, "class": "exact"})
            continue
        if common >= len(target_ids) or common >= len(dflash_ids):
            classifications.append(
                {**base, "class": "unresolved-termination", "common_prefix_tokens": common}
            )
            continue
        internal_verdict = _emitted_token_verdict(draft, common, token_label=token_label)
        if internal_verdict["class"] != "speculative-internal-argmax":
            verdict = internal_verdict
        else:
            paired_target_verdict = _paired_target_verdict(
                target, dflash_ids[common], common, token_label=token_label
            )
            verdict = (
                paired_target_verdict
                if paired_target_verdict["class"] != "paired-target-not-argmax"
                else internal_verdict
            )
        classifications.append(
            {
                **base,
                **verdict,
                "target_only_token_id": target_ids[common],
            }
        )
    counts = dict(collections.Counter(str(row["class"]) for row in classifications))
    engine_mode = dflash_rows[0]["engine_mode"]
    if counts.get("internal-target-consistency-mismatch", 0):
        next_action = "instrument-rejection-and-logprob-index-mapping"
    elif any(label.startswith("unresolved-") for label in counts):
        next_action = "capture-full-vocabulary-or-termination-evidence"
    elif counts.get("speculative-internal-argmax", 0):
        next_action = (
            "rerun-internal-target-eager"
            if engine_mode == "compiled"
            else "instrument-speculative-kv-context-and-numerical-path"
        )
    elif counts.get("target-rerun-argmax", 0):
        next_action = "treat-as-target-rerun-tie-not-engine-drift"
    else:
        next_action = "no-runtime-mismatch-reproduced"
    return {
        "schema_version": 1,
        "producer": "q30-dflash2-internal-target-diagnostic-v1",
        "claim_scope": "runtime correctness diagnosis only; no speedup or training-quality claim",
        "selection": [
            {"subset": "HumanEval", "index": index, "reason": reason}
            for index, reason in INTERNAL_TARGET_DIAGNOSTIC_ROWS
        ],
        "engine_mode": engine_mode,
        "counts": counts,
        "classifications": classifications,
        "acceptance": {
            "num_spec_steps": spec_steps,
            "num_accepted_draft_tokens": accepted_total,
            "num_draft_tokens": drafted_total,
            "draft_acceptance_rate": accepted_total / drafted_total if drafted_total else 0.0,
        },
        "next_action": next_action,
    }


def summarize_dflash_internal_target_control(
    target_path: Path, dflash_path: Path
) -> dict[str, Any]:
    """Classify the OPB DFlash B8/K7 control on the shared V2 verification path."""
    payload = summarize_internal_target_diagnostic(
        target_path,
        dflash_path,
        speculative_role="dflash",
        token_label="dflash",
    )
    rows = _read_internal_target_rows(dflash_path, "dflash")
    rows_with_steps = sum(row["speculative_decoding"]["num_spec_steps"] > 0 for row in rows)
    counts = payload["counts"]
    if counts.get("internal-target-consistency-mismatch", 0) or counts.get(
        "speculative-internal-argmax", 0
    ):
        outcome = "shared-target-rejection-path-mismatch-reproduced"
        next_action = "instrument-shared-v2-target-rejection-and-dflash-parent-path"
    elif any(label.startswith("unresolved-") for label in counts):
        outcome = "inconclusive-unresolved"
        next_action = "capture-full-vocabulary-or-termination-evidence"
    else:
        outcome = "dflash-exercised-path-exact"
        next_action = "instrument-dflash2-selector-and-draft-token-alignment"
    payload.update(
        {
            "producer": "q30-opb-dflash-s4166-internal-target-control-v1",
            "claim_scope": (
                "shared V2 target/rejection-path correctness control only; "
                "no DFlash2 speedup or training-quality claim"
            ),
            "control": {
                "method": "dflash",
                "architecture": "DFlashDraftModel",
                "training_milestone": EVALUATION_STEP,
                "block_size": DFLASH2_BLOCK_SIZE,
                "num_speculative_tokens": DFLASH2_SPECULATIVE_TOKENS,
                "engine_mode": "eager-no-prefix",
            },
            "control_outcome": outcome,
            "next_action": next_action,
        }
    )
    payload["acceptance"] = {
        **payload["acceptance"],
        "rows_with_spec_steps": rows_with_steps,
        "rows_without_spec_steps": len(rows) - rows_with_steps,
    }
    return payload


def _validate_rca_source_receipt(
    path: Path, current_identity: dict[str, Any] | None = None
) -> dict[str, Any]:
    payload = _load_json(path)
    unsigned = {key: value for key, value in payload.items() if key != "receipt_sha256"}
    classifications = payload.get("classifications")
    expected_classes = {
        0: "exact",
        1: "target-invalid",
        8: "tied-target-valid",
        27: "target-invalid",
        41: "target-invalid",
        82: "target-invalid",
        151: "target-invalid",
        165: "tied-target-valid",
        172: "target-invalid",
    }
    if (
        payload.get("receipt_sha256") != _sha_json(unsigned)
        or payload.get("receipt_sha256") != RCA_SOURCE_PILOT_RECEIPT_SHA256
        or payload.get("producer") != "q30-dflash2-tie-aware-pilot-summary-v1"
        or payload.get("status") != "failed"
        or payload.get("subset") != "HumanEval"
        or payload.get("occurrences") != 200
        or not isinstance(classifications, list)
        or len(classifications) != 200
        or [row.get("index") if isinstance(row, dict) else None for row in classifications]
        != list(range(200))
    ):
        raise ValueError("internal-target RCA source receipt mismatch")
    selected = {
        row.get("index"): row
        for row in classifications
        if isinstance(row, dict) and row.get("index") in expected_classes
    }
    for index, expected_class in expected_classes.items():
        row = selected.get(index)
        if (
            not isinstance(row, dict)
            or row.get("subset") != "HumanEval"
            or row.get("source_row") != index % 164
            or row.get("class") != expected_class
        ):
            raise ValueError("internal-target RCA classification mismatch")
    if current_identity is not None:
        source_identity_path = _validate_file_descriptor(payload.get("artifact_identity"))
        source_identity = _load_json(source_identity_path)
        source_unsigned = {
            key: value for key, value in source_identity.items() if key != "receipt_sha256"
        }
        source_dataset = source_identity.get("dataset")
        current_dataset = current_identity.get("dataset")
        if (
            source_identity.get("producer") != "q30-dflash2-s4166-speculators-inputs-v1"
            or source_identity.get("receipt_sha256") != _sha_json(source_unsigned)
            or not isinstance(source_dataset, dict)
            or not isinstance(current_dataset, dict)
            or source_dataset.get("prompt_sha256") != current_dataset.get("prompt_sha256")
            or not isinstance(source_dataset.get("ordered_prompts"), list)
            or not isinstance(current_dataset.get("ordered_prompts"), list)
        ):
            raise ValueError("internal-target RCA prompt identity mismatch")
        source_schedule = {
            row.get("index"): row
            for row in source_dataset["ordered_prompts"]
            if isinstance(row, dict) and row.get("subset") == "HumanEval"
        }
        current_schedule = {
            row.get("index"): row
            for row in current_dataset["ordered_prompts"]
            if isinstance(row, dict) and row.get("subset") == "HumanEval"
        }
        for index, _ in INTERNAL_TARGET_DIAGNOSTIC_ROWS:
            expected = source_schedule.get(index)
            current = current_schedule.get(index)
            if (
                not isinstance(expected, dict)
                or not isinstance(current, dict)
                or expected.get("source_row") != current.get("source_row")
                or expected.get("prompt_sha256") != current.get("prompt_sha256")
            ):
                raise ValueError("internal-target RCA selected prompt mismatch")
    return payload


def _validate_internal_rows_against_identity(
    path: Path, role: str, identity: dict[str, Any]
) -> list[dict[str, Any]]:
    rows = _read_internal_target_rows(path, role)
    target = identity.get("target")
    dataset = identity.get("dataset")
    if not isinstance(target, dict) or not isinstance(dataset, dict):
        raise ValueError("internal-target artifact identity mismatch")
    schedule = dataset.get("ordered_prompts")
    files = dataset.get("files")
    if not isinstance(schedule, list) or not isinstance(files, dict):
        raise ValueError("internal-target prompt identity mismatch")
    entry = files.get("HumanEval")
    if not isinstance(entry, dict):
        raise ValueError("internal-target HumanEval identity missing")
    prompts = _read_prompt_prefix(Path(str(entry.get("path", ""))), 200)
    schedule_by_index = {
        row.get("index"): row
        for row in schedule
        if isinstance(row, dict) and row.get("subset") == "HumanEval"
    }
    for row, (index, reason) in zip(rows, INTERNAL_TARGET_DIAGNOSTIC_ROWS, strict=True):
        source_row, prompt = prompts[index]
        body = {
            "model": target.get("path"),
            "prompt": prompt,
            "max_tokens": 64,
            "temperature": 0,
            "top_p": 1,
            "seed": 42,
            "logprobs": 20,
            "return_tokens_as_token_ids": True,
            "return_token_ids": True,
            "request_id": f"specdec-internal-target-HumanEval-{index}",
        }
        expected = schedule_by_index.get(index)
        if (
            not isinstance(expected, dict)
            or row["index"] != index
            or row["source_row"] != source_row
            or row["selection_reason"] != reason
            or row["prompt_sha256"] != hashlib.sha256(prompt.encode()).hexdigest()
            or row["prompt_sha256"] != expected.get("prompt_sha256")
            or source_row != expected.get("source_row")
            or row["request_sha256"] != _sha_json(body)
        ):
            raise ValueError("internal-target row does not match artifact schedule")
    return rows


def build_internal_target_diagnostic_receipt(
    target_rows_path: Path,
    dflash2_rows_path: Path,
    source_pilot_receipt_path: Path,
    target_manifest_path: Path,
    dflash2_manifest_path: Path,
    artifact_identity_path: Path,
    allocation_receipt_path: Path,
    output_path: Path,
) -> dict[str, Any]:
    """Publish live-job evidence for the bounded internal-target diagnostic."""
    identity = _validate_artifact_identity(artifact_identity_path)
    source = _validate_rca_source_receipt(source_pilot_receipt_path, identity)
    target_rows = _validate_internal_rows_against_identity(target_rows_path, "target", identity)
    dflash_rows = _validate_internal_rows_against_identity(dflash2_rows_path, "dflash2", identity)
    if any(
        target["prompt_token_ids"] != dflash["prompt_token_ids"]
        for target, dflash in zip(target_rows, dflash_rows, strict=True)
    ):
        raise ValueError("internal-target paired prompt token IDs mismatch")
    engine_mode = dflash_rows[0]["engine_mode"]
    disable_prefix_caching = engine_mode == "eager-no-prefix"
    target_evidence = _validate_target_control_manifest(
        target_manifest_path,
        artifact_identity_path,
        identity,
        disable_prefix_caching=disable_prefix_caching,
    )
    dflash_manifest, dflash_launcher, dflash_fingerprint = _validate_dflash2_probe_manifest(
        dflash2_manifest_path,
        target_evidence[0],
        target_evidence[2],
        artifact_identity_path,
        identity,
        detailed_metrics=True,
        enforce_eager=engine_mode in {"eager", "eager-no-prefix"},
        disable_prefix_caching=disable_prefix_caching,
    )
    allocation = validate_target_control_allocation_receipt(allocation_receipt_path)
    current = _query_current_allocation()
    for name in (
        "slurm_job_id",
        "slurm_job_num_nodes",
        "slurm_job_nodelist",
        "gpu_count",
        "cell_visible_devices",
    ):
        if allocation.get(name) != current.get(name):
            raise ValueError(f"internal-target live allocation mismatch: {name}")
    if any(
        manifest.get("slurm_job_id") != allocation["slurm_job_id"]
        for manifest in (target_evidence[0], dflash_manifest)
    ):
        raise ValueError("internal-target allocation job mismatch")
    if {
        _manifest_server_port(target_evidence[0]),
        _manifest_server_port(dflash_manifest),
    } != {"8000", "8010"}:
        raise ValueError("internal-target server ports are not isolated")
    payload = summarize_internal_target_diagnostic(target_rows_path, dflash2_rows_path)
    payload["source_pilot_receipt_sha256"] = source["receipt_sha256"]
    payload["allocation_evidence_scope"] = (
        "live SLURM/GPU origin checked at creation; offline verification is tamper replay"
    )
    payload["source_pilot_receipt"] = _file_descriptor(source_pilot_receipt_path)
    payload["artifact_identity"] = _file_descriptor(artifact_identity_path)
    payload["allocation_receipt"] = _file_descriptor(allocation_receipt_path)
    payload["rows"] = {
        "target": _file_descriptor(target_rows_path),
        "dflash2": _file_descriptor(dflash2_rows_path),
    }
    payload["manifests"] = {
        "target": _file_descriptor(target_manifest_path),
        "dflash2": _file_descriptor(dflash2_manifest_path),
    }
    payload["input_fingerprints"] = {
        "target": _file_descriptor(target_evidence[1]),
        "dflash2": _file_descriptor(dflash_fingerprint),
    }
    payload["launcher_configs"] = {
        "target": _file_descriptor(target_evidence[3]),
        "dflash2": _file_descriptor(dflash_launcher),
    }
    payload["receipt_sha256"] = _sha_json(payload)
    _atomic_json(output_path, payload, no_replace=True)
    return payload


def validate_internal_target_diagnostic_receipt(path: Path) -> dict[str, Any]:
    """Replay all durable internal-target evidence without scheduler-origin claims."""
    payload = _load_json(path)
    claim = payload.pop("receipt_sha256", None)
    if claim != _sha_json(payload):
        raise ValueError("internal-target receipt self-hash mismatch")
    source_path = _validate_file_descriptor(payload.get("source_pilot_receipt"))
    artifact_path = _validate_file_descriptor(payload.get("artifact_identity"))
    allocation_path = _validate_file_descriptor(payload.get("allocation_receipt"))
    evidence: dict[str, dict[str, Path]] = {}
    for group in ("rows", "manifests", "input_fingerprints", "launcher_configs"):
        value = payload.get(group)
        if not isinstance(value, dict) or set(value) != {"target", "dflash2"}:
            raise ValueError(f"internal-target {group} schema mismatch")
        evidence[group] = {name: _validate_file_descriptor(item) for name, item in value.items()}
    identity = _validate_artifact_identity(artifact_path)
    source = _validate_rca_source_receipt(source_path, identity)
    target_rows = _validate_internal_rows_against_identity(
        evidence["rows"]["target"], "target", identity
    )
    dflash_rows = _validate_internal_rows_against_identity(
        evidence["rows"]["dflash2"], "dflash2", identity
    )
    if any(
        target["prompt_token_ids"] != dflash["prompt_token_ids"]
        for target, dflash in zip(target_rows, dflash_rows, strict=True)
    ):
        raise ValueError("internal-target paired prompt token IDs mismatch")
    engine_mode = dflash_rows[0]["engine_mode"]
    disable_prefix_caching = engine_mode == "eager-no-prefix"
    target_evidence = _validate_target_control_manifest(
        evidence["manifests"]["target"],
        artifact_path,
        identity,
        disable_prefix_caching=disable_prefix_caching,
    )
    dflash_manifest, dflash_launcher, dflash_fingerprint = _validate_dflash2_probe_manifest(
        evidence["manifests"]["dflash2"],
        target_evidence[0],
        target_evidence[2],
        artifact_path,
        identity,
        detailed_metrics=True,
        enforce_eager=engine_mode in {"eager", "eager-no-prefix"},
        disable_prefix_caching=disable_prefix_caching,
    )
    allocation = validate_target_control_allocation_receipt(allocation_path)
    if any(
        manifest.get("slurm_job_id") != allocation["slurm_job_id"]
        for manifest in (target_evidence[0], dflash_manifest)
    ):
        raise ValueError("internal-target allocation replay mismatch")
    if {
        _manifest_server_port(target_evidence[0]),
        _manifest_server_port(dflash_manifest),
    } != {"8000", "8010"}:
        raise ValueError("internal-target server ports are not isolated")
    if evidence["input_fingerprints"] != {
        "target": target_evidence[1],
        "dflash2": dflash_fingerprint,
    } or evidence["launcher_configs"] != {
        "target": target_evidence[3],
        "dflash2": dflash_launcher,
    }:
        raise ValueError("internal-target provenance descriptor mismatch")
    replayed = summarize_internal_target_diagnostic(
        evidence["rows"]["target"], evidence["rows"]["dflash2"]
    )
    for name, value in replayed.items():
        if payload.get(name) != value:
            raise ValueError(f"internal-target replay mismatch: {name}")
    if (
        payload.get("source_pilot_receipt_sha256") != source["receipt_sha256"]
        or payload.get("allocation_evidence_scope")
        != "live SLURM/GPU origin checked at creation; offline verification is tamper replay"
    ):
        raise ValueError("internal-target claim scope mismatch")
    return {**payload, "receipt_sha256": claim}


def build_dflash_internal_target_control_receipt(
    target_rows_path: Path,
    dflash_rows_path: Path,
    source_pilot_receipt_path: Path,
    target_manifest_path: Path,
    dflash_manifest_path: Path,
    artifact_identity_path: Path,
    allocation_receipt_path: Path,
    control_stage_receipt_path: Path,
    output_path: Path,
) -> dict[str, Any]:
    """Publish live-job evidence for the OPB DFlash shared-path control."""
    identity = _validate_artifact_identity(artifact_identity_path)
    source = _validate_rca_source_receipt(source_pilot_receipt_path, identity)
    target_rows = _validate_internal_rows_against_identity(target_rows_path, "target", identity)
    dflash_rows = _validate_internal_rows_against_identity(dflash_rows_path, "dflash", identity)
    if any(
        target["prompt_token_ids"] != dflash["prompt_token_ids"]
        for target, dflash in zip(target_rows, dflash_rows, strict=True)
    ):
        raise ValueError("DFlash control paired prompt token IDs mismatch")
    if any(row["engine_mode"] != "eager-no-prefix" for row in dflash_rows):
        raise ValueError("DFlash control requires eager-no-prefix rows")
    target_evidence = _validate_target_control_manifest(
        target_manifest_path,
        artifact_identity_path,
        identity,
        disable_prefix_caching=True,
    )
    dflash_manifest, dflash_launcher, dflash_fingerprint, control_artifact = (
        _validate_dflash_control_manifest(
            dflash_manifest_path,
            target_evidence[0],
            target_evidence[2],
            artifact_identity_path,
            identity,
            control_stage_receipt_path,
            require_live_stage=True,
        )
    )
    allocation = validate_target_control_allocation_receipt(allocation_receipt_path)
    current = _query_current_allocation()
    for name in (
        "slurm_job_id",
        "slurm_job_num_nodes",
        "slurm_job_nodelist",
        "gpu_count",
        "cell_visible_devices",
    ):
        if allocation.get(name) != current.get(name):
            raise ValueError(f"DFlash control live allocation mismatch: {name}")
    if any(
        manifest.get("slurm_job_id") != allocation["slurm_job_id"]
        for manifest in (target_evidence[0], dflash_manifest)
    ):
        raise ValueError("DFlash control allocation job mismatch")
    if {
        _manifest_server_port(target_evidence[0]),
        _manifest_server_port(dflash_manifest),
    } != {"8000", "8010"}:
        raise ValueError("DFlash control server ports are not isolated")
    payload = summarize_dflash_internal_target_control(target_rows_path, dflash_rows_path)
    payload["schema_version"] = 2
    payload["producer"] = "q30-opb-dflash-s4166-internal-target-control-v2"
    payload["source_pilot_receipt_sha256"] = source["receipt_sha256"]
    payload["allocation_evidence_scope"] = (
        "live SLURM/GPU origin checked at creation; offline verification is tamper replay"
    )
    payload["source_pilot_receipt"] = _file_descriptor(source_pilot_receipt_path)
    payload["artifact_identity"] = _file_descriptor(artifact_identity_path)
    payload["control_artifact"] = control_artifact
    payload["control_stage_receipt"] = _file_descriptor(control_stage_receipt_path)
    payload["allocation_receipt"] = _file_descriptor(allocation_receipt_path)
    payload["rows"] = {
        "target": _file_descriptor(target_rows_path),
        "dflash": _file_descriptor(dflash_rows_path),
    }
    payload["manifests"] = {
        "target": _file_descriptor(target_manifest_path),
        "dflash": _file_descriptor(dflash_manifest_path),
    }
    payload["input_fingerprints"] = {
        "target": _file_descriptor(target_evidence[1]),
        "dflash": _file_descriptor(dflash_fingerprint),
    }
    payload["launcher_configs"] = {
        "target": _file_descriptor(target_evidence[3]),
        "dflash": _file_descriptor(dflash_launcher),
    }
    payload["receipt_sha256"] = _sha_json(payload)
    _atomic_json(output_path, payload, no_replace=True)
    return payload


def validate_dflash_internal_target_control_receipt(path: Path) -> dict[str, Any]:
    """Replay the durable OPB DFlash control evidence without scheduler-origin claims."""
    payload = _load_json(path)
    claim = payload.pop("receipt_sha256", None)
    schema_version = payload.get("schema_version")
    expected_producer = "q30-opb-dflash-s4166-internal-target-control-v2"
    if schema_version != 2:
        raise ValueError("DFlash control receipt schema v2 required; no historical v1 allowlist")
    if set(payload) != _DFLASH_CONTROL_RECEIPT_V2_KEYS:
        raise ValueError("DFlash control receipt schema mismatch")
    if payload.get("producer") != expected_producer:
        raise ValueError("DFlash control receipt producer mismatch")
    if claim != _sha_json(payload):
        raise ValueError("DFlash control receipt self-hash mismatch")
    source_path = _validate_file_descriptor(payload.get("source_pilot_receipt"))
    artifact_path = _validate_file_descriptor(payload.get("artifact_identity"))
    allocation_path = _validate_file_descriptor(payload.get("allocation_receipt"))
    control_stage_path = _validate_file_descriptor(payload.get("control_stage_receipt"))
    evidence: dict[str, dict[str, Path]] = {}
    for group in ("rows", "manifests", "input_fingerprints", "launcher_configs"):
        value = payload.get(group)
        if not isinstance(value, dict) or set(value) != {"target", "dflash"}:
            raise ValueError(f"DFlash control {group} schema mismatch")
        evidence[group] = {name: _validate_file_descriptor(item) for name, item in value.items()}
    identity = _validate_artifact_identity(artifact_path)
    source = _validate_rca_source_receipt(source_path, identity)
    target_rows = _validate_internal_rows_against_identity(
        evidence["rows"]["target"], "target", identity
    )
    dflash_rows = _validate_internal_rows_against_identity(
        evidence["rows"]["dflash"], "dflash", identity
    )
    if any(
        target["prompt_token_ids"] != dflash["prompt_token_ids"]
        for target, dflash in zip(target_rows, dflash_rows, strict=True)
    ) or any(row["engine_mode"] != "eager-no-prefix" for row in dflash_rows):
        raise ValueError("DFlash control paired row mismatch")
    target_evidence = _validate_target_control_manifest(
        evidence["manifests"]["target"],
        artifact_path,
        identity,
        disable_prefix_caching=True,
    )
    dflash_manifest, dflash_launcher, dflash_fingerprint, control_artifact = (
        _validate_dflash_control_manifest(
            evidence["manifests"]["dflash"],
            target_evidence[0],
            target_evidence[2],
            artifact_path,
            identity,
            control_stage_path,
        )
    )
    allocation = validate_target_control_allocation_receipt(allocation_path)
    if any(
        manifest.get("slurm_job_id") != allocation["slurm_job_id"]
        for manifest in (target_evidence[0], dflash_manifest)
    ):
        raise ValueError("DFlash control allocation replay mismatch")
    if {
        _manifest_server_port(target_evidence[0]),
        _manifest_server_port(dflash_manifest),
    } != {"8000", "8010"}:
        raise ValueError("DFlash control server ports are not isolated")
    if evidence["input_fingerprints"] != {
        "target": target_evidence[1],
        "dflash": dflash_fingerprint,
    } or evidence["launcher_configs"] != {
        "target": target_evidence[3],
        "dflash": dflash_launcher,
    }:
        raise ValueError("DFlash control provenance descriptor mismatch")
    if payload.get("control_artifact") != control_artifact:
        raise ValueError("DFlash control artifact replay mismatch")
    if control_artifact.get("stage_receipt") != payload.get("control_stage_receipt"):
        raise ValueError("DFlash control stage receipt replay mismatch")
    replayed = summarize_dflash_internal_target_control(
        evidence["rows"]["target"], evidence["rows"]["dflash"]
    )
    replayed["schema_version"] = schema_version
    replayed["producer"] = expected_producer
    for name, value in replayed.items():
        if payload.get(name) != value:
            raise ValueError(f"DFlash control replay mismatch: {name}")
    if (
        payload.get("source_pilot_receipt_sha256") != source["receipt_sha256"]
        or payload.get("allocation_evidence_scope")
        != "live SLURM/GPU origin checked at creation; offline verification is tamper replay"
    ):
        raise ValueError("DFlash control claim scope mismatch")
    return {**payload, "receipt_sha256": claim}


def _read_tie_aware_pilot_rows(path: Path, expected_role: str) -> list[dict[str, Any]]:
    expected_keys = {
        "schema_version",
        "producer",
        "role",
        "subset",
        "index",
        "source_row",
        "prompt_sha256",
        "request_core_sha256",
        "request_sha256",
        "output_text",
        "token_ids",
        "prompt_token_ids",
        "finish_reason",
        "top_logprobs",
        "response_sha256",
    }
    rows: list[dict[str, Any]] = []
    with path.open() as stream:
        for line in stream:
            value = json.loads(line)
            if not isinstance(value, dict) or set(value) != expected_keys:
                raise ValueError("tie-aware pilot row schema mismatch")
            tokens = value.get("token_ids")
            prompt_tokens = value.get("prompt_token_ids")
            top = value.get("top_logprobs")
            extracted = {
                "output_text": value.get("output_text"),
                "token_ids": tokens,
                "prompt_token_ids": prompt_tokens,
                "finish_reason": value.get("finish_reason"),
                "top_logprobs": top,
            }
            if (
                value.get("schema_version") != 1
                or value.get("producer") != "q30-tie-aware-online-row-v1"
                or value.get("role") != expected_role
                or value.get("subset") != "HumanEval"
                or isinstance(value.get("index"), bool)
                or not isinstance(value.get("index"), int)
                or isinstance(value.get("source_row"), bool)
                or not isinstance(value.get("source_row"), int)
                or value["source_row"] < 0
                or not _is_sha256(value.get("prompt_sha256"))
                or not _is_sha256(value.get("request_core_sha256"))
                or not _is_sha256(value.get("request_sha256"))
                or not _is_sha256(value.get("response_sha256"))
                or value["response_sha256"] != _sha_json(extracted)
                or not isinstance(value.get("output_text"), str)
                or value.get("finish_reason") not in {"stop", "length"}
                or not isinstance(tokens, list)
                or not all(
                    isinstance(token, int) and not isinstance(token, bool) and token >= 0
                    for token in tokens
                )
                or not isinstance(prompt_tokens, list)
                or not prompt_tokens
                or not all(
                    isinstance(token, int) and not isinstance(token, bool) and token >= 0
                    for token in prompt_tokens
                )
                or not isinstance(top, list)
            ):
                raise ValueError("tie-aware pilot row evidence mismatch")
            if expected_role != "target" and top:
                raise ValueError("DFlash2 pilot must not claim target logprob evidence")
            rows.append(value)
    if len(rows) != 200 or [row["index"] for row in rows] != list(range(200)):
        raise ValueError("tie-aware pilot requires exact ordered 200-row schedule")
    return rows


def summarize_tie_aware_pilot(target_path: Path, dflash2_path: Path) -> dict[str, Any]:
    """Replay the 200-row set-valued greedy correctness classification."""
    target_rows = _read_tie_aware_pilot_rows(target_path, "target")
    dflash_rows = _read_tie_aware_pilot_rows(dflash2_path, "dflash2")
    classifications = [
        classify_tie_aware_rows(target, dflash)
        for target, dflash in zip(target_rows, dflash_rows, strict=True)
    ]
    counts: dict[str, int] = {}
    for row in classifications:
        label = str(row["class"])
        counts[label] = counts.get(label, 0) + 1
    unresolved = sum(count for label, count in counts.items() if label.startswith("unresolved-"))
    return {
        "schema_version": 1,
        "producer": "q30-dflash2-tie-aware-pilot-summary-v1",
        "claim_scope": (
            "set-valued greedy correctness pilot only; transport/schema capture failures abort "
            "before receipt; no semantic correctness or speedup claim"
        ),
        "subset": "HumanEval",
        "occurrences": len(classifications),
        "unique_source_rows": len({row["source_row"] for row in target_rows}),
        "counts": counts,
        "status": (
            "passed" if counts.get("target-invalid", 0) == 0 and unresolved == 0 else "failed"
        ),
        "classifications": classifications,
    }


def _validate_tie_aware_rows_against_identity(
    path: Path, role: str, identity: dict[str, Any]
) -> list[dict[str, Any]]:
    rows = _read_tie_aware_pilot_rows(path, role)
    target = identity.get("target")
    dataset = identity.get("dataset")
    if not isinstance(target, dict) or not isinstance(dataset, dict):
        raise ValueError("tie-aware artifact identity mismatch")
    files = dataset.get("files")
    schedule = dataset.get("ordered_prompts")
    if not isinstance(files, dict) or not isinstance(schedule, list):
        raise ValueError("tie-aware prompt schedule is missing")
    entry = files.get("HumanEval")
    if not isinstance(entry, dict):
        raise ValueError("tie-aware HumanEval file is missing")
    prompts = _read_prompt_prefix(Path(str(entry.get("path", ""))), 200)
    expected_schedule = [
        item for item in schedule if isinstance(item, dict) and item.get("subset") == "HumanEval"
    ]
    if len(prompts) != 200 or len(expected_schedule) != 200:
        raise ValueError("tie-aware pilot needs the authenticated exact-200 schedule")
    for index, (row, expected, prompt_entry) in enumerate(
        zip(rows, expected_schedule, prompts, strict=True)
    ):
        source_row, prompt = prompt_entry
        core = {
            "model": target.get("path"),
            "prompt": prompt,
            "max_tokens": 64,
            "temperature": 0,
            "top_p": 1,
            "seed": 42,
            "request_id": f"specdec-tie-pilot-HumanEval-{index}",
        }
        body = {
            **core,
            "logprobs": 20 if role == "target" else 0,
            "return_tokens_as_token_ids": True,
            "return_token_ids": True,
        }
        if (
            expected
            != {
                "subset": "HumanEval",
                "index": index,
                "source_row": source_row,
                "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
            }
            or row["source_row"] != source_row
            or row["prompt_sha256"] != expected["prompt_sha256"]
            or row["request_core_sha256"] != _sha_json(core)
            or row["request_sha256"] != _sha_json(body)
        ):
            raise ValueError("tie-aware pilot row does not match artifact schedule")
    return rows


def _manifest_server_port(manifest: dict[str, Any]) -> str:
    server_args = manifest.get("server_args")
    if not isinstance(server_args, list) or server_args.count("--port") != 1:
        raise ValueError("tie-aware pilot server port schema mismatch")
    index = server_args.index("--port")
    if index + 1 >= len(server_args) or server_args[index + 1] not in {"8000", "8010"}:
        raise ValueError("tie-aware pilot server port schema mismatch")
    return str(server_args[index + 1])


def build_tie_aware_pilot_receipt(
    target_rows_path: Path,
    dflash2_rows_path: Path,
    target_manifest_path: Path,
    dflash2_manifest_path: Path,
    artifact_identity_path: Path,
    allocation_receipt_path: Path,
    output_path: Path,
) -> dict[str, Any]:
    """Publish the job-authenticated 200-row tie-aware pilot receipt."""
    identity = _validate_artifact_identity(artifact_identity_path)
    target_rows = _validate_tie_aware_rows_against_identity(target_rows_path, "target", identity)
    dflash_rows = _validate_tie_aware_rows_against_identity(dflash2_rows_path, "dflash2", identity)
    if any(
        target["prompt_token_ids"] != dflash["prompt_token_ids"]
        for target, dflash in zip(target_rows, dflash_rows, strict=True)
    ):
        raise ValueError("tie-aware paired prompt token IDs mismatch")
    target_evidence = _validate_target_control_manifest(
        target_manifest_path, artifact_identity_path, identity
    )
    dflash_manifest, dflash_launcher, dflash_fingerprint = _validate_dflash2_probe_manifest(
        dflash2_manifest_path,
        target_evidence[0],
        target_evidence[2],
        artifact_identity_path,
        identity,
    )
    allocation = validate_target_control_allocation_receipt(allocation_receipt_path)
    current = _query_current_allocation()
    for name in (
        "slurm_job_id",
        "slurm_job_num_nodes",
        "slurm_job_nodelist",
        "gpu_count",
        "cell_visible_devices",
    ):
        if allocation.get(name) != current.get(name):
            raise ValueError(f"tie-aware pilot live allocation mismatch: {name}")
    if any(
        manifest.get("slurm_job_id") != allocation["slurm_job_id"]
        for manifest in (target_evidence[0], dflash_manifest)
    ):
        raise ValueError("tie-aware pilot allocation job mismatch")
    if {
        _manifest_server_port(target_evidence[0]),
        _manifest_server_port(dflash_manifest),
    } != {"8000", "8010"}:
        raise ValueError("tie-aware pilot server ports are not isolated")
    payload = summarize_tie_aware_pilot(target_rows_path, dflash2_rows_path)
    payload["allocation_evidence_scope"] = (
        "live SLURM/GPU origin checked at creation; offline verification is tamper replay"
    )
    payload["artifact_identity"] = _file_descriptor(artifact_identity_path)
    payload["allocation_receipt"] = _file_descriptor(allocation_receipt_path)
    payload["rows"] = {
        "target": _file_descriptor(target_rows_path),
        "dflash2": _file_descriptor(dflash2_rows_path),
    }
    payload["manifests"] = {
        "target": _file_descriptor(target_manifest_path),
        "dflash2": _file_descriptor(dflash2_manifest_path),
    }
    payload["input_fingerprints"] = {
        "target": _file_descriptor(target_evidence[1]),
        "dflash2": _file_descriptor(dflash_fingerprint),
    }
    payload["launcher_configs"] = {
        "target": _file_descriptor(target_evidence[3]),
        "dflash2": _file_descriptor(dflash_launcher),
    }
    payload["receipt_sha256"] = _sha_json(payload)
    _atomic_json(output_path, payload, no_replace=True)
    return payload


def validate_tie_aware_pilot_receipt(path: Path) -> dict[str, Any]:
    """Offline tamper replay for every pilot input and classification."""
    payload = _load_json(path)
    claim = payload.pop("receipt_sha256", None)
    if claim != _sha_json(payload):
        raise ValueError("tie-aware pilot receipt self-hash mismatch")
    artifact_path = _validate_file_descriptor(payload.get("artifact_identity"))
    allocation_path = _validate_file_descriptor(payload.get("allocation_receipt"))
    evidence: dict[str, dict[str, Path]] = {}
    for group in ("rows", "manifests", "input_fingerprints", "launcher_configs"):
        value = payload.get(group)
        if not isinstance(value, dict) or set(value) != {"target", "dflash2"}:
            raise ValueError(f"tie-aware pilot {group} schema mismatch")
        evidence[group] = {name: _validate_file_descriptor(item) for name, item in value.items()}
    identity = _validate_artifact_identity(artifact_path)
    target_rows = _validate_tie_aware_rows_against_identity(
        evidence["rows"]["target"], "target", identity
    )
    dflash_rows = _validate_tie_aware_rows_against_identity(
        evidence["rows"]["dflash2"], "dflash2", identity
    )
    if any(
        target["prompt_token_ids"] != dflash["prompt_token_ids"]
        for target, dflash in zip(target_rows, dflash_rows, strict=True)
    ):
        raise ValueError("tie-aware paired prompt token IDs mismatch")
    target_evidence = _validate_target_control_manifest(
        evidence["manifests"]["target"], artifact_path, identity
    )
    dflash_manifest, dflash_launcher, dflash_fingerprint = _validate_dflash2_probe_manifest(
        evidence["manifests"]["dflash2"],
        target_evidence[0],
        target_evidence[2],
        artifact_path,
        identity,
    )
    allocation = validate_target_control_allocation_receipt(allocation_path)
    if any(
        manifest.get("slurm_job_id") != allocation["slurm_job_id"]
        for manifest in (target_evidence[0], dflash_manifest)
    ):
        raise ValueError("tie-aware pilot allocation replay mismatch")
    if {
        _manifest_server_port(target_evidence[0]),
        _manifest_server_port(dflash_manifest),
    } != {"8000", "8010"}:
        raise ValueError("tie-aware pilot server ports are not isolated")
    if evidence["input_fingerprints"] != {
        "target": target_evidence[1],
        "dflash2": dflash_fingerprint,
    } or evidence["launcher_configs"] != {
        "target": target_evidence[3],
        "dflash2": dflash_launcher,
    }:
        raise ValueError("tie-aware pilot provenance descriptor mismatch")
    replayed = summarize_tie_aware_pilot(evidence["rows"]["target"], evidence["rows"]["dflash2"])
    for name, value in replayed.items():
        if payload.get(name) != value:
            raise ValueError(f"tie-aware pilot replay mismatch: {name}")
    if payload.get("allocation_evidence_scope") != (
        "live SLURM/GPU origin checked at creation; offline verification is tamper replay"
    ):
        raise ValueError("tie-aware pilot claim scope mismatch")
    return {**payload, "receipt_sha256": claim}


def capture_outputs(
    dataset_manifest_path: Path,
    hf_home: Path,
    output_path: Path,
    *,
    endpoint: str,
    model: str,
    requests_per_subset: int = CORRECTNESS_REQUESTS_PER_SUBSET,
    max_tokens: int = 64,
) -> None:
    """Capture a small deterministic ledger using the same pinned prompt snapshot."""
    if requests_per_subset != CORRECTNESS_REQUESTS_PER_SUBSET or max_tokens != 64:
        raise ValueError("correctness capture requires 200 prompts/subset and max_tokens=64")
    provenance = compute_prompt_set(
        dataset_manifest_path,
        hf_home,
        requests_per_subset=EVALUATION_REQUESTS_PER_SUBSET,
    )
    files = provenance["files"]
    if not isinstance(files, dict):
        raise ValueError("invalid prompt provenance")
    if output_path.exists():
        raise FileExistsError(f"output ledger exists: {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{output_path.name}.", dir=output_path.parent)
    try:
        with os.fdopen(descriptor, "w") as stream:
            for subset in STANDARD_SUBSETS:
                entry = files[subset]
                assert isinstance(entry, dict)
                prompts = _read_prompt_prefix(Path(str(entry["path"])), requests_per_subset)
                for index in range(requests_per_subset):
                    source_row, prompt = prompts[index % len(prompts)]
                    body = _completion_request_body(model, subset, index, prompt)
                    request = urllib.request.Request(
                        endpoint.rstrip("/") + "/completions",
                        data=body,
                        headers={"Content-Type": "application/json"},
                        method="POST",
                    )
                    with urllib.request.urlopen(request, timeout=600) as response:
                        result = json.loads(response.read())
                    choice = result["choices"][0]
                    text = choice["text"]
                    finish = choice["finish_reason"]
                    logprobs = choice.get("logprobs")
                    tokens = logprobs.get("tokens") if isinstance(logprobs, dict) else None
                    if (
                        not isinstance(text, str)
                        or not isinstance(tokens, list)
                        or not all(isinstance(token, str) for token in tokens)
                        or finish not in {"stop", "length"}
                    ):
                        raise ValueError("invalid completion response")
                    record = {
                        "subset": subset,
                        "index": index,
                        "source_row": source_row,
                        "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
                        "output_sha256": hashlib.sha256(text.encode()).hexdigest(),
                        "output_text": text,
                        "output_tokens": tokens,
                        "request_sha256": hashlib.sha256(body).hexdigest(),
                        "seed": 42,
                        "finish_reason": finish,
                    }
                    stream.write(_canonical(record) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, output_path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def main() -> None:
    """Dispatch matched-evaluation artifact operations."""
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)

    prompts = commands.add_parser("prompt-set")
    prompts.add_argument("--dataset-manifest", required=True)
    prompts.add_argument("--hf-home", required=True)
    prompts.add_argument("--output", required=True)

    materialize = commands.add_parser("materialize-prompts")
    materialize.add_argument("--source-manifest", required=True)
    materialize.add_argument("--source-hf-home", required=True)
    materialize.add_argument("--output-hf-home", required=True)
    materialize.add_argument("--output-manifest", required=True)

    capture = commands.add_parser("capture-outputs")
    capture.add_argument("--dataset-manifest", required=True)
    capture.add_argument("--hf-home", required=True)
    capture.add_argument("--output", required=True)
    capture.add_argument("--endpoint", required=True)
    capture.add_argument("--model", required=True)

    probe = commands.add_parser("capture-probe")
    probe.add_argument("--dataset-manifest", required=True)
    probe.add_argument("--hf-home", required=True)
    probe.add_argument("--output", required=True)
    probe.add_argument("--endpoint", required=True)
    probe.add_argument("--model", required=True)
    probe.add_argument("--method", required=True, choices=("baseline", "dflash2"))

    tie_capture = commands.add_parser("capture-tie-pilot")
    tie_capture.add_argument("--dataset-manifest", required=True)
    tie_capture.add_argument("--hf-home", required=True)
    tie_capture.add_argument("--output", required=True)
    tie_capture.add_argument("--endpoint", required=True)
    tie_capture.add_argument("--model", required=True)
    tie_capture.add_argument("--role", required=True, choices=("target", "dflash2"))

    internal_capture = commands.add_parser("capture-internal-target")
    internal_capture.add_argument("--dataset-manifest", required=True)
    internal_capture.add_argument("--hf-home", required=True)
    internal_capture.add_argument("--output", required=True)
    internal_capture.add_argument("--endpoint", required=True)
    internal_capture.add_argument("--model", required=True)
    internal_capture.add_argument("--role", required=True, choices=("target", "dflash", "dflash2"))
    internal_capture.add_argument(
        "--engine-mode", required=True, choices=("compiled", "eager", "eager-no-prefix")
    )

    compare = commands.add_parser("compare-outputs")
    compare.add_argument("--baseline", required=True)
    compare.add_argument("--dflash2", required=True)
    compare.add_argument("--output", required=True)
    compare.add_argument("--artifact-identity", required=True)

    control = commands.add_parser("analyze-control")
    control.add_argument("--left", required=True)
    control.add_argument("--right", required=True)
    control.add_argument("--left-manifest", required=True)
    control.add_argument("--right-manifest", required=True)
    control.add_argument("--artifact-identity", required=True)
    control.add_argument("--allocation-receipt", required=True)
    control.add_argument("--output", required=True)

    allocation = commands.add_parser("allocation-receipt")
    allocation.add_argument("--output", required=True)

    summary = commands.add_parser("summarize")
    summary.add_argument("--baseline-run", required=True)
    summary.add_argument("--dflash2-run", required=True)
    summary.add_argument("--correctness-receipt", required=True)
    summary.add_argument("--output", required=True)
    summary.add_argument("--artifact-identity", required=True)
    summary.add_argument("--concurrency", required=True, type=int, choices=(1, 32))

    identity = commands.add_parser("artifact-identity")
    identity.add_argument("--output", required=True)
    identity.add_argument("--target", required=True)
    identity.add_argument("--target-sha256", required=True)
    identity.add_argument("--target-receipt", required=True)
    identity.add_argument("--target-receipt-sha256", required=True)
    identity.add_argument("--export", required=True)
    identity.add_argument("--milestone-manifest", required=True)
    identity.add_argument("--dataset-manifest", required=True)
    identity.add_argument("--hf-home", required=True)
    identity.add_argument("--client-runtime-archive", required=True)
    identity.add_argument("--client-runtime-archive-sha256", required=True)
    identity.add_argument("--server-runtime-archive", required=True)
    identity.add_argument("--server-runtime-archive-sha256", required=True)
    identity.add_argument("--server-runtime-receipt-sha256", required=True)

    milestone = commands.add_parser("verify-milestone")
    milestone.add_argument("--manifest", required=True)
    milestone.add_argument("--export", required=True)

    verify_report = commands.add_parser("verify-report")
    verify_report.add_argument("--report", required=True)

    verify_control = commands.add_parser("verify-control")
    verify_control.add_argument("--receipt", required=True)

    diagnose = commands.add_parser("analyze-divergence")
    diagnose.add_argument("--baseline-probe", required=True)
    diagnose.add_argument("--dflash2-probe", required=True)
    diagnose.add_argument("--baseline-manifest", required=True)
    diagnose.add_argument("--dflash2-manifest", required=True)
    diagnose.add_argument("--artifact-identity", required=True)
    diagnose.add_argument("--allocation-receipt", required=True)
    diagnose.add_argument("--output", required=True)

    verify_diagnosis = commands.add_parser("verify-divergence")
    verify_diagnosis.add_argument("--receipt", required=True)

    tie_analysis = commands.add_parser("analyze-tie-pilot")
    tie_analysis.add_argument("--target-rows", required=True)
    tie_analysis.add_argument("--dflash2-rows", required=True)
    tie_analysis.add_argument("--target-manifest", required=True)
    tie_analysis.add_argument("--dflash2-manifest", required=True)
    tie_analysis.add_argument("--artifact-identity", required=True)
    tie_analysis.add_argument("--allocation-receipt", required=True)
    tie_analysis.add_argument("--output", required=True)

    tie_verify = commands.add_parser("verify-tie-pilot")
    tie_verify.add_argument("--receipt", required=True)

    internal_analysis = commands.add_parser("analyze-internal-target")
    internal_analysis.add_argument("--target-rows", required=True)
    internal_analysis.add_argument("--dflash2-rows", required=True)
    internal_analysis.add_argument("--source-pilot-receipt", required=True)
    internal_analysis.add_argument("--target-manifest", required=True)
    internal_analysis.add_argument("--dflash2-manifest", required=True)
    internal_analysis.add_argument("--artifact-identity", required=True)
    internal_analysis.add_argument("--allocation-receipt", required=True)
    internal_analysis.add_argument("--output", required=True)

    internal_verify = commands.add_parser("verify-internal-target")
    internal_verify.add_argument("--receipt", required=True)

    opb_stage = commands.add_parser("stage-opb-dflash-control")
    opb_stage.add_argument("--source-draft", required=True)
    opb_stage.add_argument("--stage-root", required=True)
    opb_stage.add_argument("--receipt", required=True)

    opb_stage_verify = commands.add_parser("verify-opb-dflash-stage")
    opb_stage_verify.add_argument("--receipt", required=True)
    opb_stage_verify.add_argument("--require-live-stage", action="store_true")
    opb_stage_verify.add_argument("--staged-only", action="store_true")
    opb_stage_verify.add_argument("--expected-staged-draft")

    dflash_analysis = commands.add_parser("analyze-dflash-control")
    dflash_analysis.add_argument("--target-rows", required=True)
    dflash_analysis.add_argument("--dflash-rows", required=True)
    dflash_analysis.add_argument("--source-pilot-receipt", required=True)
    dflash_analysis.add_argument("--target-manifest", required=True)
    dflash_analysis.add_argument("--dflash-manifest", required=True)
    dflash_analysis.add_argument("--artifact-identity", required=True)
    dflash_analysis.add_argument("--allocation-receipt", required=True)
    dflash_analysis.add_argument("--control-stage-receipt", required=True)
    dflash_analysis.add_argument("--output", required=True)

    dflash_verify = commands.add_parser("verify-dflash-control")
    dflash_verify.add_argument("--receipt", required=True)

    args = parser.parse_args()
    if args.command == "prompt-set":
        payload = compute_prompt_set(Path(args.dataset_manifest), Path(args.hf_home))
        payload["receipt_sha256"] = _sha_json(payload)
        _atomic_json(Path(args.output), payload, no_replace=True)
    elif args.command == "materialize-prompts":
        materialize_prompt_set(
            Path(args.source_manifest),
            Path(args.source_hf_home),
            Path(args.output_hf_home),
            Path(args.output_manifest),
        )
    elif args.command == "capture-outputs":
        capture_outputs(
            Path(args.dataset_manifest),
            Path(args.hf_home),
            Path(args.output),
            endpoint=args.endpoint,
            model=args.model,
        )
    elif args.command == "capture-probe":
        capture_divergence_probe(
            Path(args.dataset_manifest),
            Path(args.hf_home),
            Path(args.output),
            endpoint=args.endpoint,
            model=args.model,
            method=args.method,
        )
    elif args.command == "capture-tie-pilot":
        capture_tie_aware_pilot(
            Path(args.dataset_manifest),
            Path(args.hf_home),
            Path(args.output),
            endpoint=args.endpoint,
            model=args.model,
            role=args.role,
        )
    elif args.command == "capture-internal-target":
        capture_internal_target_diagnostic(
            Path(args.dataset_manifest),
            Path(args.hf_home),
            Path(args.output),
            endpoint=args.endpoint,
            model=args.model,
            role=args.role,
            engine_mode=args.engine_mode,
        )
    elif args.command == "compare-outputs":
        artifact_identity_sha256 = _sha256(Path(args.artifact_identity))
        identity = _load_json(Path(args.artifact_identity))
        dataset = identity.get("dataset")
        if not isinstance(dataset, dict) or not isinstance(dataset.get("ordered_prompts"), list):
            raise ValueError("artifact identity has no ordered prompt schedule")
        payload = validate_output_equivalence(
            Path(args.baseline),
            Path(args.dflash2),
            artifact_identity_sha256=artifact_identity_sha256,
            expected_prompt_schedule=dataset["ordered_prompts"],
        )
        payload["receipt_sha256"] = _sha_json(payload)
        _atomic_json(Path(args.output), payload, no_replace=True)
    elif args.command == "analyze-control":
        build_target_control_receipt(
            Path(args.left),
            Path(args.right),
            Path(args.left_manifest),
            Path(args.right_manifest),
            Path(args.artifact_identity),
            Path(args.allocation_receipt),
            Path(args.output),
        )
    elif args.command == "allocation-receipt":
        _publish_current_allocation_receipt(Path(args.output))
    elif args.command == "analyze-divergence":
        build_divergence_probe_receipt(
            Path(args.baseline_probe),
            Path(args.dflash2_probe),
            Path(args.baseline_manifest),
            Path(args.dflash2_manifest),
            Path(args.artifact_identity),
            Path(args.allocation_receipt),
            Path(args.output),
        )
    elif args.command == "verify-divergence":
        validate_divergence_probe_receipt(Path(args.receipt))
    elif args.command == "analyze-tie-pilot":
        build_tie_aware_pilot_receipt(
            Path(args.target_rows),
            Path(args.dflash2_rows),
            Path(args.target_manifest),
            Path(args.dflash2_manifest),
            Path(args.artifact_identity),
            Path(args.allocation_receipt),
            Path(args.output),
        )
    elif args.command == "verify-tie-pilot":
        validate_tie_aware_pilot_receipt(Path(args.receipt))
    elif args.command == "analyze-internal-target":
        build_internal_target_diagnostic_receipt(
            Path(args.target_rows),
            Path(args.dflash2_rows),
            Path(args.source_pilot_receipt),
            Path(args.target_manifest),
            Path(args.dflash2_manifest),
            Path(args.artifact_identity),
            Path(args.allocation_receipt),
            Path(args.output),
        )
    elif args.command == "verify-internal-target":
        validate_internal_target_diagnostic_receipt(Path(args.receipt))
    elif args.command == "stage-opb-dflash-control":
        stage_opb_dflash_control(Path(args.source_draft), Path(args.stage_root), Path(args.receipt))
    elif args.command == "verify-opb-dflash-stage":
        validate_opb_dflash_stage_receipt(
            Path(args.receipt),
            require_live_stage=args.require_live_stage,
            require_live_source=not args.staged_only,
            expected_staged_draft=(
                Path(args.expected_staged_draft) if args.expected_staged_draft else None
            ),
        )
    elif args.command == "analyze-dflash-control":
        build_dflash_internal_target_control_receipt(
            Path(args.target_rows),
            Path(args.dflash_rows),
            Path(args.source_pilot_receipt),
            Path(args.target_manifest),
            Path(args.dflash_manifest),
            Path(args.artifact_identity),
            Path(args.allocation_receipt),
            Path(args.control_stage_receipt),
            Path(args.output),
        )
    elif args.command == "verify-dflash-control":
        validate_dflash_internal_target_control_receipt(Path(args.receipt))
    elif args.command == "summarize":
        correctness = _load_json(Path(args.correctness_receipt))
        claim = correctness.pop("receipt_sha256", None)
        artifact_identity_sha256 = _sha256(Path(args.artifact_identity))
        if (
            correctness.get("status") != "passed"
            or correctness.get("artifact_identity_sha256") != artifact_identity_sha256
            or claim != _sha_json(correctness)
        ):
            raise ValueError("correctness receipt is not a passing self-hashed receipt")
        payload = summarize_pair(
            Path(args.baseline_run), Path(args.dflash2_run), tensor_parallel_size=2
        )
        payload["evidence"] = _metric_evidence(
            Path(args.baseline_run),
            Path(args.dflash2_run),
            concurrency=args.concurrency,
            artifact_identity_sha256=artifact_identity_sha256,
        )
        payload["correctness_receipt"] = _file_descriptor(Path(args.correctness_receipt))
        payload["artifact_identity"] = _file_descriptor(Path(args.artifact_identity))
        payload["receipt_sha256"] = _sha_json(payload)
        _atomic_json(Path(args.output), payload, no_replace=True)
    elif args.command == "artifact-identity":
        payload = build_artifact_identity(
            target_path=Path(args.target),
            target_sha256=args.target_sha256,
            target_receipt_path=Path(args.target_receipt),
            target_receipt_sha256=args.target_receipt_sha256,
            export_path=Path(args.export),
            milestone_manifest_path=Path(args.milestone_manifest),
            dataset_manifest_path=Path(args.dataset_manifest),
            hf_home=Path(args.hf_home),
            client_runtime_archive=Path(args.client_runtime_archive),
            client_runtime_archive_sha256=args.client_runtime_archive_sha256,
            server_runtime_archive=Path(args.server_runtime_archive),
            server_runtime_archive_sha256=args.server_runtime_archive_sha256,
            server_runtime_receipt_sha256=args.server_runtime_receipt_sha256,
        )
        _atomic_json(Path(args.output), payload, no_replace=True)
    elif args.command == "verify-milestone":
        print(_canonical(validate_milestone_export(Path(args.manifest), Path(args.export))))
    elif args.command == "verify-report":
        print(_canonical(validate_report_receipt(Path(args.report))))
    else:
        print(_canonical(validate_target_control_receipt(Path(args.receipt))))


if __name__ == "__main__":
    main()
