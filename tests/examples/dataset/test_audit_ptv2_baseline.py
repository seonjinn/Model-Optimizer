from __future__ import annotations

import importlib.util
import json
import sys
from dataclasses import asdict
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

ROOT = Path(__file__).resolve().parents[3]


def _load():
    path = ROOT / "examples/dataset/audit_ptv2_baseline.py"
    spec = importlib.util.spec_from_file_location("audit_ptv2_baseline", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    sys.path.insert(0, str(path.parent))
    try:
        spec.loader.exec_module(module)
    finally:
        sys.path.pop(0)
    return module


def _shard(root: Path, name: str, prompts: list[str]) -> None:
    target = root.parent / f"raw-{name}"
    pq.write_table(
        pa.table({"messages": [json.dumps([{"role": "user", "content": p}]) for p in prompts]}),
        target,
    )
    (root / name).symlink_to(target)


def test_audit_resolves_hashes_counts_and_unique_canonical_uuids(tmp_path: Path) -> None:
    module = _load()
    root = tmp_path / "data"
    root.mkdir()
    _shard(root, "chat-00000.parquet", ["a", "b"])
    _shard(root, "math-00000.parquet", ["c"])
    expected = module.BaselineExpectation(
        "5c89e01dd720ae0f4058445ed49c5fb68a03c76e", 2, {"chat": 2, "math": 1}
    )
    audit = module.audit_baseline(root, expected)
    assert audit.row_count == 3
    assert audit.split_rows == {"chat": 2, "math": 1}
    assert audit.occurrence_count == 3
    assert audit.unique_prompt_count == 3
    assert len(audit.occurrence_prompt_ids) == 3
    assert len(audit.exclusion_prompt_ids) == 3
    assert len(audit.files) == 2


def test_audit_preserves_duplicate_occurrences_and_unique_exclusion_set(tmp_path: Path) -> None:
    module = _load()
    root = tmp_path / "data"
    root.mkdir()
    _shard(root, "chat-0.parquet", ["same", "chat-only"])
    _shard(root, "math-0.parquet", ["same", "math-only"])
    expected = module.BaselineExpectation(
        "5c89e01dd720ae0f4058445ed49c5fb68a03c76e", 2, {"chat": 2, "math": 2}
    )
    audit = module.audit_baseline(root, expected)
    assert audit.occurrence_count == 4
    assert audit.unique_prompt_count == 3
    assert len(audit.occurrence_prompt_ids) == 4
    assert len(audit.exclusion_prompt_ids) == 3
    assert audit.occurrence_prompt_ids[0] == audit.occurrence_prompt_ids[2]
    assert audit.exclusion_prompt_ids == tuple(sorted(audit.exclusion_prompt_ids))
    assert audit.duplicate_uuid_multiplicity == {audit.occurrence_prompt_ids[0]: 2}
    assert audit.occurrence_prompt_ids_sha256 != audit.exclusion_prompt_ids_sha256


def test_audit_reads_top_level_nested_conversations_column(tmp_path: Path) -> None:
    module = _load()
    root = tmp_path / "data"
    root.mkdir()
    target = tmp_path / "raw-chat.parquet"
    pq.write_table(pa.table({"conversations": [[{"role": "user", "content": "nested"}]]}), target)
    (root / "chat-0.parquet").symlink_to(target)
    expected = module.BaselineExpectation(
        "5c89e01dd720ae0f4058445ed49c5fb68a03c76e", 1, {"chat": 1}
    )
    audit = module.audit_baseline(root, expected)
    assert audit.row_count == 1


def test_audit_reads_real_top_level_nested_messages_schema(tmp_path: Path) -> None:
    module = _load()
    root = tmp_path / "data"
    root.mkdir()
    target = tmp_path / "raw-chat.parquet"
    pq.write_table(pa.table({"messages": [[{"role": "user", "content": "nested"}]]}), target)
    (root / "chat-0.parquet").symlink_to(target)
    expected = module.BaselineExpectation(
        "5c89e01dd720ae0f4058445ed49c5fb68a03c76e", 1, {"chat": 1}
    )
    assert module.audit_baseline(root, expected).row_count == 1


def test_audit_uuid_excludes_source_assistant_completion(tmp_path: Path) -> None:
    module = _load()
    root = tmp_path / "data"
    root.mkdir()
    target = tmp_path / "raw-chat.parquet"
    pq.write_table(
        pa.table(
            {
                "messages": [
                    [{"role": "user", "content": "same"}, {"role": "assistant", "content": "a"}],
                    [{"role": "user", "content": "same"}, {"role": "assistant", "content": "b"}],
                ]
            }
        ),
        target,
    )
    (root / "chat-0.parquet").symlink_to(target)
    expected = module.BaselineExpectation(
        "5c89e01dd720ae0f4058445ed49c5fb68a03c76e", 1, {"chat": 2}
    )
    audit = module.audit_baseline(root, expected)
    assert audit.unique_prompt_count == 1
    assert audit.duplicate_uuid_multiplicity == {audit.occurrence_prompt_ids[0]: 2}


def test_audit_still_fails_closed_on_histogram_mismatch(tmp_path: Path) -> None:
    module = _load()
    root = tmp_path / "data"
    root.mkdir()
    _shard(root, "chat-0.parquet", ["one"])
    expected = module.BaselineExpectation(
        "5c89e01dd720ae0f4058445ed49c5fb68a03c76e", 1, {"chat": 2}
    )
    with pytest.raises(module.AuditError, match="histogram mismatch"):
        module.audit_baseline(root, expected)


def test_audit_reproduces_authoritative_sorted_stream_take_boundary(tmp_path: Path) -> None:
    module = _load()
    root = tmp_path / "data"
    root.mkdir()
    _shard(root, "chat-0.parquet", ["c0", "c1"])
    _shard(root, "math-0.parquet", ["m0", "unused"])
    expected = module.BaselineExpectation(
        "5c89e01dd720ae0f4058445ed49c5fb68a03c76e", 2, {"chat": 2, "math": 1}
    )
    audit = module.audit_baseline(root, expected)
    assert audit.row_count == 3
    assert audit.selection_policy == "hf-streaming-sorted-parquet-take"
    assert audit.selection_boundary.file == "math-0.parquet"
    assert audit.selection_boundary.rows_selected == 1
    assert audit.selection_boundary.rows_available == 2
    assert audit.selection_boundary.excluded_tail_rows == 1


def test_production_baseline_expectation_is_exact() -> None:
    module = _load()
    assert module.EXPECTED_BASELINE.source_revision == "5c89e01dd720ae0f4058445ed49c5fb68a03c76e"
    assert module.EXPECTED_BASELINE.shard_count == 26
    assert module.EXPECTED_BASELINE.split_rows == {
        "chat": 627_720,
        "code": 175_000,
        "math": 239_467,
        "multilingual_de": 257_813,
    }
    assert sum(module.EXPECTED_BASELINE.split_rows.values()) == 1_300_000
    assert module.EXPECTED_BASELINE.unique_prompt_count == 931_363


def test_audit_rejects_configured_unique_prompt_count_mismatch(tmp_path: Path) -> None:
    module = _load()
    root = tmp_path / "data"
    root.mkdir()
    _shard(root, "chat-0.parquet", ["duplicate", "duplicate", "unique"])
    expected = module.BaselineExpectation(
        "5c89e01dd720ae0f4058445ed49c5fb68a03c76e", 1, {"chat": 3}, 3
    )
    with pytest.raises(module.AuditError, match="unique prompt count mismatch"):
        module.audit_baseline(root, expected)


def test_audit_receipt_bytes_are_deterministic(tmp_path: Path) -> None:
    module = _load()
    root = tmp_path / "data"
    root.mkdir()
    _shard(root, "chat-0.parquet", ["a", "a"])
    expected = module.BaselineExpectation(
        "5c89e01dd720ae0f4058445ed49c5fb68a03c76e", 1, {"chat": 2}
    )
    first = module.canonical_json(asdict(module.audit_baseline(root, expected)))
    second = module.canonical_json(asdict(module.audit_baseline(root, expected)))
    assert first == second
