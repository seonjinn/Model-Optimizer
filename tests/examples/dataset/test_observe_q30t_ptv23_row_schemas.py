# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import hashlib
import json
import os
import shutil
import sys
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any

import pyarrow as pa  # pyright: ignore[reportMissingImports]
import pyarrow.parquet as pq  # pyright: ignore[reportMissingImports]
import pytest

import examples.dataset.observe_q30t_ptv23_row_schemas as observer_module
from examples.dataset.observe_q30t_ptv23_row_schemas import (
    ObservationError,
    StageInputs,
    observe_stage_schemas,
)

if TYPE_CHECKING:
    from pathlib import Path


_TEST_RUNTIME = {
    "python_executable": "/usr/bin/python3.12",
    "python_executable_bytes": 123456,
    "python_executable_sha256": "b" * 64,
    "python_version": "3.12.0",
    "pyarrow_version": "test-pyarrow",
    "pyarrow_origin": "/usr/lib/python3.12/site-packages/pyarrow/__init__.py",
    "pyarrow_tree_file_count": 1,
    "pyarrow_tree_bytes": 1,
    "pyarrow_tree_sha256": "a" * 64,
}
_REAL_AUTHENTICATE_RUNTIME = observer_module._authenticate_runtime


@pytest.fixture(autouse=True)
def approved_runtime(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(observer_module, "_authenticate_runtime", lambda: _TEST_RUNTIME)
    monkeypatch.setattr(observer_module, "_require_imported_runtime", lambda runtime: None)


def _canonical(value: object) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode()


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def valid_row(label: str = "ok") -> dict[str, object]:
    return {
        "messages": [
            {"role": "user", "content": f"question-{label}"},
            {"role": "assistant", "content": f"answer-{label}"},
        ],
        "tools": [],
        "metadata": {"label": label},
    }


@dataclass(frozen=True)
class StageFixture:
    inputs: StageInputs
    ptv2_file: Path
    ptv3_files: tuple[Path, ...]
    ptv3_plan: dict[str, Any]
    ptv3_manifest: dict[str, Any]


def _write_parquet(path: Path, rows: tuple[dict[str, object], ...], *, raw_json: bool) -> None:
    if raw_json:
        table = pa.table(
            {"raw_json": [json.dumps(row, sort_keys=True, separators=(",", ":")) for row in rows]}
        )
    else:
        table = pa.Table.from_pylist(list(rows))
    pq.write_table(table, path)


def stage_inputs(
    tmp_path: Path,
    *,
    file_rows: tuple[tuple[dict[str, object], ...], ...] = ((valid_row("a"),), (valid_row("b"),)),
) -> StageFixture:
    if len(file_rows) < 2:
        raise ValueError("fixtures require one PTV2 file and at least one PTV3 file")

    ptv2_root = tmp_path / "ptv2"
    ptv2_data = ptv2_root / "data"
    ptv2_data.mkdir(parents=True)
    ptv2_file = ptv2_data / "train-00000-of-00001.parquet"
    _write_parquet(ptv2_file, file_rows[0], raw_json=False)
    ptv2_record = {
        "path": f"data/{ptv2_file.name}",
        "bytes": ptv2_file.stat().st_size,
        "sha256": _sha256(ptv2_file),
    }
    ptv2_plan = {
        "schema_version": 1,
        "name": "fixture-ptv2-plan",
        "sources": [
            {
                "repository_id": "nvidia/Fixture-PTV2",
                "configuration": "default",
                "split": "train",
                "revision": "1" * 40,
                "license_expression": "CC-BY-4.0",
                "approved_use": True,
                "cell": "train",
                "lane": "target-synth",
                "files": [ptv2_record],
            }
        ],
    }
    ptv2_plan_path = ptv2_root / "SOURCE_PLAN.json"
    ptv2_plan_path.write_bytes(_canonical(ptv2_plan))
    ptv2_plan_sha256 = _sha256(ptv2_plan_path)
    ptv2_completion = {
        "schema_version": 1,
        "complete": True,
        "repository_id": "nvidia/Fixture-PTV2",
        "configuration": "default",
        "revision": "1" * 40,
        "license_expression": "CC-BY-4.0",
        "approved_use": True,
        "manifest": {
            "path": "SOURCE_PLAN.json",
            "bytes": ptv2_plan_path.stat().st_size,
            "sha256": ptv2_plan_sha256,
        },
        "split_counts": {"train": 1},
        "split_bytes": {"train": ptv2_file.stat().st_size},
        "target_inventory": {"count": 1, "sha256": "2" * 64},
        "source_commit": "3" * 40,
        "source_root_realpath": str(ptv2_root),
        "approved_hao_root_realpath": str(tmp_path),
        "workers": {"requested": 1, "effective": 1},
        "readme": {
            "path": str(tmp_path / "README.md"),
            "bytes": 1,
            "sha256": "4" * 64,
            "target_device": 1,
            "target_inode": 1,
            "target_mtime_ns": 1,
        },
        "timing": {
            "started_at_utc": "2026-08-27T00:00:00Z",
            "completed_at_utc": "2026-08-27T00:00:01Z",
            "duration_seconds": 1,
        },
    }
    ptv2_completion_path = ptv2_root / "SOURCE_MANIFEST_COMPLETION.json"
    ptv2_completion_path.write_bytes(_canonical(ptv2_completion))

    ptv3_root = tmp_path / "ptv3"
    ptv3_root.mkdir()
    ptv3_plan_files: list[dict[str, object]] = []
    ptv3_manifest_files: list[dict[str, object]] = []
    ptv3_files: list[Path] = []
    for ordinal, rows in enumerate(file_rows[1:]):
        source_path = f"data/source-{ordinal:05d}.jsonl"
        physical = ptv3_root / f"stage-{ordinal:05d}.parquet"
        _write_parquet(physical, rows, raw_json=True)
        source = {
            "source_id": "nvidia/Fixture-PTV3",
            "revision": "5" * 40,
            "license": "CC-BY-4.0",
            "pool": "ptv3",
            "category": "fixture-category",
            "response_source": "trace-replay",
            "tool_lane": "recorded-trace",
            "path": source_path,
            "bytes": 100 + ordinal,
            "sha256": f"{ordinal + 6:x}" * 64,
        }
        ptv3_plan_files.append(source)
        ptv3_manifest_files.append(
            {
                "path": physical.name,
                "bytes": physical.stat().st_size,
                "sha256": _sha256(physical),
                "row_count": len(rows),
                "source_id": source["source_id"],
                "source_revision": source["revision"],
                "source_path": source_path,
                "source_bytes": source["bytes"],
                "source_sha256": source["sha256"],
                "license": source["license"],
                "pool": source["pool"],
                "category": source["category"],
                "response_source": source["response_source"],
                "tool_lane": source["tool_lane"],
            }
        )
        ptv3_files.append(physical)

    ptv3_plan = {
        "schema_version": 1,
        "name": "fixture-ptv3-plan",
        "container": {"path": str(tmp_path / "container.sqsh"), "sha256": "f" * 64},
        "files": ptv3_plan_files,
    }
    ptv3_plan_path = tmp_path / "PTV3_PLAN.json"
    ptv3_plan_path.write_bytes(_canonical(ptv3_plan))
    ptv3_plan_sha256 = _sha256(ptv3_plan_path)
    ptv3_manifest = {
        "schema_version": 1,
        "name": ptv3_plan["name"],
        "plan_sha256": ptv3_plan_sha256,
        "container": ptv3_plan["container"],
        "format": "parquet",
        "compression": "zstd",
        "file_count": len(ptv3_manifest_files),
        "row_count": sum(len(rows) for rows in file_rows[1:]),
        "files": ptv3_manifest_files,
    }
    ptv3_manifest_path = ptv3_root / "MANIFEST.json"
    ptv3_manifest_path.write_bytes(_canonical(ptv3_manifest))
    ptv3_completion = {
        "schema_version": 1,
        "complete": True,
        "plan_sha256": ptv3_plan_sha256,
        "manifest_sha256": _sha256(ptv3_manifest_path),
    }
    ptv3_completion_path = ptv3_root / "completion.json"
    ptv3_completion_path.write_bytes(_canonical(ptv3_completion))

    return StageFixture(
        inputs=StageInputs(
            ptv2_plan_path=ptv2_plan_path,
            ptv2_plan_sha256=ptv2_plan_sha256,
            ptv2_completion_path=ptv2_completion_path,
            ptv2_completion_sha256=_sha256(ptv2_completion_path),
            ptv2_root=ptv2_root,
            ptv3_plan_path=ptv3_plan_path,
            ptv3_plan_sha256=ptv3_plan_sha256,
            ptv3_completion_path=ptv3_completion_path,
            ptv3_completion_sha256=_sha256(ptv3_completion_path),
            ptv3_root=ptv3_root,
        ),
        ptv2_file=ptv2_file,
        ptv3_files=tuple(ptv3_files),
        ptv3_plan=ptv3_plan,
        ptv3_manifest=ptv3_manifest,
    )


def _rewrite_ptv3_manifest(fixture: StageFixture, manifest: dict[str, Any]) -> StageInputs:
    manifest_path = fixture.inputs.ptv3_root / "MANIFEST.json"
    manifest_path.write_bytes(_canonical(manifest))
    completion = {
        "schema_version": 1,
        "complete": True,
        "plan_sha256": fixture.inputs.ptv3_plan_sha256,
        "manifest_sha256": _sha256(manifest_path),
    }
    fixture.inputs.ptv3_completion_path.write_bytes(_canonical(completion))
    return replace(
        fixture.inputs,
        ptv3_completion_sha256=_sha256(fixture.inputs.ptv3_completion_path),
    )


def test_observer_records_every_file_and_every_row_shape(tmp_path: Path) -> None:
    fixture = stage_inputs(
        tmp_path,
        file_rows=((valid_row("a"),), (valid_row("b"), valid_row("c"))),
    )

    payload = json.loads(observe_stage_schemas(fixture.inputs))

    assert payload["schema_version"] == "q30t-ptv23-row-schema-observation-v1"
    assert len(payload["files"]) == 2
    assert [item["row_count"] for item in payload["files"]] == [1, 2]
    assert all(item["messages_field"] == "messages" for item in payload["files"])
    assert all(item["tools_field"] == "tools" for item in payload["files"])
    assert all(item["physical_format"] == "parquet" for item in payload["files"])
    assert all(item["row_schema_sha256"] for item in payload["files"])
    assert payload["source_file_counts"] == {"ptv2": 1, "ptv3": 1}
    assert payload["row_count"] == 3


def test_observer_rejects_a_late_malformed_row(tmp_path: Path) -> None:
    fixture = stage_inputs(
        tmp_path,
        file_rows=((valid_row(),), (valid_row(), {"messages": 7, "tools": []})),
    )

    with pytest.raises(ObservationError, match=r"file 2.*row 2"):
        observe_stage_schemas(fixture.inputs)


def test_observation_is_canonical_self_hashed_and_contains_no_policy_decisions(
    tmp_path: Path,
) -> None:
    fixture = stage_inputs(tmp_path)

    observed = observe_stage_schemas(fixture.inputs)
    payload = json.loads(observed)
    claimed = payload.pop("observation_sha256")

    assert observed.endswith(b"\n")
    assert claimed == hashlib.sha256(_canonical(payload)).hexdigest()
    assert observed == _canonical({**payload, "observation_sha256": claimed})
    serialized = json.dumps(payload, sort_keys=True)
    for forbidden in ("category", "license", "replay", "approval"):
        assert forbidden not in serialized


def test_observation_is_independent_of_private_plan_materialization_path(tmp_path: Path) -> None:
    fixture = stage_inputs(tmp_path)
    first = observe_stage_schemas(fixture.inputs)
    second_plan = tmp_path / "other-private-scratch" / fixture.inputs.ptv3_plan_path.name
    second_plan.parent.mkdir()
    second_plan.write_bytes(fixture.inputs.ptv3_plan_path.read_bytes())

    second = observe_stage_schemas(replace(fixture.inputs, ptv3_plan_path=second_plan))

    assert second == first


def test_observer_rejects_plan_completion_cross_pairing(tmp_path: Path) -> None:
    fixture = stage_inputs(tmp_path)
    completion = json.loads(fixture.inputs.ptv3_completion_path.read_bytes())
    completion["plan_sha256"] = fixture.inputs.ptv2_plan_sha256
    fixture.inputs.ptv3_completion_path.write_bytes(_canonical(completion))
    inputs = replace(
        fixture.inputs,
        ptv3_completion_sha256=_sha256(fixture.inputs.ptv3_completion_path),
    )

    with pytest.raises(ObservationError, match="PTV3 completion plan binding"):
        observe_stage_schemas(inputs)


def test_observer_rejects_a_plan_with_the_wrong_bound_sha(tmp_path: Path) -> None:
    fixture = stage_inputs(tmp_path)
    inputs = replace(fixture.inputs, ptv2_plan_sha256="0" * 64)

    with pytest.raises(ObservationError, match="PTV2 plan SHA-256"):
        observe_stage_schemas(inputs)


def test_observer_rejects_omitted_manifest_records_before_data_reads(tmp_path: Path) -> None:
    fixture = stage_inputs(
        tmp_path,
        file_rows=((valid_row(),), (valid_row("b"),), (valid_row("c"),)),
    )
    manifest = fixture.ptv3_manifest.copy()
    manifest["files"] = manifest["files"][:-1]
    manifest["file_count"] = 1
    manifest["row_count"] = 1
    inputs = _rewrite_ptv3_manifest(fixture, manifest)

    with pytest.raises(ObservationError, match="PTV3 plan/manifest inventory"):
        observe_stage_schemas(inputs)


def test_observer_rejects_extra_staged_paths_before_data_reads(tmp_path: Path) -> None:
    fixture = stage_inputs(tmp_path)
    (fixture.inputs.ptv3_root / "foreign.parquet").write_bytes(b"foreign")

    with pytest.raises(ObservationError, match="PTV3 staged pathname set"):
        observe_stage_schemas(fixture.inputs)


@pytest.mark.parametrize("bad_path", ["../escape.parquet", "./dot.parquet", "/abs.parquet"])
def test_observer_rejects_unsafe_manifest_paths(tmp_path: Path, bad_path: str) -> None:
    fixture = stage_inputs(tmp_path)
    manifest = fixture.ptv3_manifest.copy()
    manifest["files"] = [dict(manifest["files"][0], path=bad_path)]
    inputs = _rewrite_ptv3_manifest(fixture, manifest)

    with pytest.raises(ObservationError, match="PTV3 manifest path"):
        observe_stage_schemas(inputs)


def test_observer_rejects_duplicate_manifest_records(tmp_path: Path) -> None:
    fixture = stage_inputs(tmp_path)
    manifest = fixture.ptv3_manifest.copy()
    manifest["files"] = [manifest["files"][0], manifest["files"][0]]
    manifest["file_count"] = 2
    manifest["row_count"] = 2
    inputs = _rewrite_ptv3_manifest(fixture, manifest)

    with pytest.raises(ObservationError, match="duplicate PTV3 manifest"):
        observe_stage_schemas(inputs)


def test_observer_rejects_hardlinked_data(tmp_path: Path) -> None:
    fixture = stage_inputs(tmp_path)
    os.link(fixture.ptv2_file, tmp_path / "outside-hardlink.parquet")

    with pytest.raises(ObservationError, match="single-link regular file"):
        observe_stage_schemas(fixture.inputs)


def test_observer_rejects_a_fifo_without_blocking(tmp_path: Path) -> None:
    fixture = stage_inputs(tmp_path)
    fixture.ptv2_file.unlink()
    os.mkfifo(fixture.ptv2_file)

    with pytest.raises(ObservationError, match="single-link regular file"):
        observe_stage_schemas(fixture.inputs)


def test_observer_rejects_path_rebinding_during_iteration(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = stage_inputs(tmp_path)
    import examples.dataset.observe_q30t_ptv23_row_schemas as observer

    original = observer._iter_parquet_rows

    def rebind_after_first_row(descriptor: int):
        iterator = original(descriptor)
        yield next(iterator)
        replacement = tmp_path / "replacement.parquet"
        _write_parquet(replacement, (valid_row("replacement"),), raw_json=False)
        os.replace(replacement, fixture.ptv2_file)
        yield from iterator

    monkeypatch.setattr(observer, "_iter_parquet_rows", rebind_after_first_row)

    with pytest.raises(ObservationError, match="changed while observing"):
        observe_stage_schemas(fixture.inputs)


def test_observer_rejects_growth_during_iteration(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = stage_inputs(tmp_path)
    import examples.dataset.observe_q30t_ptv23_row_schemas as observer

    original = observer._iter_parquet_rows

    def grow_after_first_row(descriptor: int):
        iterator = original(descriptor)
        yield next(iterator)
        with fixture.ptv2_file.open("ab") as stream:
            stream.write(b"growth")
        yield from iterator

    monkeypatch.setattr(observer, "_iter_parquet_rows", grow_after_first_row)

    with pytest.raises(ObservationError, match="changed while observing"):
        observe_stage_schemas(fixture.inputs)


def test_observer_rechecks_tree_binding_after_final_metadata_reads(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = stage_inputs(tmp_path)
    replacement = tmp_path / "replacement-ptv2"
    displaced = tmp_path / "displaced-ptv2"
    shutil.copytree(fixture.inputs.ptv2_root, replacement)
    import examples.dataset.observe_q30t_ptv23_row_schemas as observer

    original = observer._read_bound_path
    calls = 0

    def replace_root_after_final_plan_read(path: Path, expected_sha256: str, label: str) -> bytes:
        nonlocal calls
        calls += 1
        raw = original(path, expected_sha256, label)
        if calls == 5:
            fixture.inputs.ptv2_root.rename(displaced)
            replacement.rename(fixture.inputs.ptv2_root)
        return raw

    monkeypatch.setattr(observer, "_read_bound_path", replace_root_after_final_plan_read)

    with pytest.raises(ObservationError, match="staged trees changed"):
        observe_stage_schemas(fixture.inputs)


def test_observer_reports_the_exact_late_invalid_jsonl_row(tmp_path: Path) -> None:
    fixture = stage_inputs(tmp_path)
    physical = fixture.ptv3_files[0]
    physical.unlink()
    physical = physical.with_suffix(".jsonl")
    physical.write_bytes(_canonical(valid_row("first")) + b"{invalid\n")
    plan = fixture.ptv3_plan.copy()
    manifest = fixture.ptv3_manifest.copy()
    manifest_record = dict(manifest["files"][0], path=physical.name)
    manifest_record["bytes"] = physical.stat().st_size
    manifest_record["sha256"] = _sha256(physical)
    manifest_record["row_count"] = 2
    manifest["files"] = [manifest_record]
    manifest["row_count"] = 2
    inputs = _rewrite_ptv3_manifest(fixture, manifest)
    assert plan["files"][0]["path"].endswith(".jsonl")

    with pytest.raises(ObservationError, match=r"file 2.*row 2"):
        observe_stage_schemas(inputs)


def test_observation_binds_the_authenticated_python_and_pyarrow_runtime(tmp_path: Path) -> None:
    fixture = stage_inputs(tmp_path)

    observation = json.loads(observe_stage_schemas(fixture.inputs))

    assert observation["runtime"] == _TEST_RUNTIME


def test_runtime_authentication_fails_closed_on_a_user_owned_python(
    tmp_path: Path,
) -> None:
    python = tmp_path / "python3.12"
    python.write_bytes(b"not an approved interpreter")
    python.chmod(0o755)
    with pytest.raises(ObservationError, match="root-owned"):
        observer_module._approved_executable(python)


def test_runtime_authentication_requires_isolated_no_site_python(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for name in tuple(observer_module.sys.modules):
        if name == "pyarrow" or name.startswith("pyarrow."):
            monkeypatch.delitem(observer_module.sys.modules, name)

    with pytest.raises(ObservationError, match=r"-I -S"):
        _REAL_AUTHENTICATE_RUNTIME()


def test_runtime_authentication_rejects_preloaded_pyarrow(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        observer_module,
        "_approved_executable",
        lambda path: observer_module.Path("/usr/bin/python3.12"),
    )
    monkeypatch.setitem(observer_module.sys.modules, "pyarrow", object())

    with pytest.raises(ObservationError, match="already loaded"):
        _REAL_AUTHENTICATE_RUNTIME()


def test_cli_publishes_exact_observation_without_clobbering(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = stage_inputs(tmp_path)
    output_parent = tmp_path / "output"
    output_parent.mkdir()
    output = output_parent / "observation.json"
    import examples.dataset.observe_q30t_ptv23_row_schemas as observer

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "observer",
            "--ptv2-plan",
            str(fixture.inputs.ptv2_plan_path),
            "--ptv2-plan-sha256",
            fixture.inputs.ptv2_plan_sha256,
            "--ptv2-completion",
            str(fixture.inputs.ptv2_completion_path),
            "--ptv2-completion-sha256",
            fixture.inputs.ptv2_completion_sha256,
            "--ptv2-root",
            str(fixture.inputs.ptv2_root),
            "--ptv3-plan",
            str(fixture.inputs.ptv3_plan_path),
            "--ptv3-plan-sha256",
            fixture.inputs.ptv3_plan_sha256,
            "--ptv3-completion",
            str(fixture.inputs.ptv3_completion_path),
            "--ptv3-completion-sha256",
            fixture.inputs.ptv3_completion_sha256,
            "--ptv3-root",
            str(fixture.inputs.ptv3_root),
            "--output",
            str(output),
        ],
    )

    assert observer.main() == 0
    assert output.read_bytes() == observe_stage_schemas(fixture.inputs)
    with pytest.raises(ObservationError, match="already exists"):
        observer.main()


def test_publication_rebinds_absolute_parent_before_mutation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output_parent = tmp_path / "output"
    output_parent.mkdir()
    displaced = tmp_path / "displaced"
    replacement = tmp_path / "replacement"
    replacement.mkdir()
    output = output_parent / "observation.json"
    import examples.dataset.observe_q30t_ptv23_row_schemas as observer

    original = observer._open_root

    def swap_after_open(path: Path, label: str) -> tuple[int, tuple[int, ...]]:
        descriptor, identity = original(path, label)
        if label == "observation output parent":
            output_parent.rename(displaced)
            replacement.rename(output_parent)
        return descriptor, identity

    monkeypatch.setattr(observer, "_open_root", swap_after_open)

    with pytest.raises(ObservationError, match=r"output parent.*changed"):
        observer._publish_observation(output, b"authenticated\n")

    assert not output.exists()
    assert not (displaced / output.name).exists()


def test_publication_failure_never_leaves_a_partial_final_name(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output_parent = tmp_path / "output"
    output_parent.mkdir()
    output = output_parent / "observation.json"
    foreign = output_parent / "foreign"
    foreign.write_text("owned elsewhere")
    import examples.dataset.observe_q30t_ptv23_row_schemas as observer

    original = observer.os.fsync

    def fail_file_fsync(descriptor: int) -> None:
        if os.path.isfile(f"/dev/fd/{descriptor}"):
            raise OSError("injected file fsync failure")
        original(descriptor)

    monkeypatch.setattr(observer.os, "fsync", fail_file_fsync)

    with pytest.raises(OSError, match="injected file fsync failure"):
        observer._publish_observation(output, b"authenticated\n")

    assert not output.exists()
    assert foreign.read_text() == "owned elsewhere"
    assert [path.name for path in output_parent.iterdir() if ".partial-" in path.name]


def test_stalled_partial_write_never_creates_the_final_name(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output_parent = tmp_path / "output"
    output_parent.mkdir()
    output = output_parent / "observation.json"
    original = observer_module.os.write
    stalled = False

    def stall_once(descriptor: int, payload: bytes) -> int:
        nonlocal stalled
        if not stalled:
            stalled = True
            return 0
        return original(descriptor, payload)

    monkeypatch.setattr(observer_module.os, "write", stall_once)

    with pytest.raises(ObservationError, match="write stalled"):
        observer_module._publish_observation(output, b"authenticated\n")

    assert not output.exists()


def test_publication_rebinds_absolute_parent_after_durability(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output_parent = tmp_path / "output"
    output_parent.mkdir()
    displaced = tmp_path / "displaced"
    replacement = tmp_path / "replacement"
    replacement.mkdir()
    output = output_parent / "observation.json"
    original = observer_module._require_absolute_parent_binding
    calls = 0

    def swap_after_durability(parent: Path, parent_fd: int) -> None:
        nonlocal calls
        calls += 1
        if calls == 3:
            output_parent.rename(displaced)
            replacement.rename(output_parent)
        original(parent, parent_fd)

    monkeypatch.setattr(observer_module, "_require_absolute_parent_binding", swap_after_durability)

    with pytest.raises(ObservationError, match=r"output parent.*changed"):
        observer_module._publish_observation(output, b"authenticated\n")

    assert not output.exists()
    assert (displaced / output.name).read_bytes() == b"authenticated\n"
