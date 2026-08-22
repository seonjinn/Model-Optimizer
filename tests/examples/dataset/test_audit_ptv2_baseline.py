from __future__ import annotations

import importlib.util
import json
import sys
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
    assert len(audit.prompt_uuids) == 3
    assert len(audit.files) == 2


def test_audit_preserves_duplicate_occurrences_and_unique_exclusion_set(tmp_path: Path) -> None:
    module = _load()
    root = tmp_path / "data"
    root.mkdir()
    _shard(root, "chat-0.parquet", ["same"])
    _shard(root, "math-0.parquet", ["same"])
    expected = module.BaselineExpectation(
        "5c89e01dd720ae0f4058445ed49c5fb68a03c76e", 2, {"chat": 1, "math": 1}
    )
    audit = module.audit_baseline(root, expected)
    assert audit.row_count == 2
    assert len(audit.prompt_uuids) == 1
    assert audit.duplicate_uuid_multiplicity == {audit.prompt_uuids[0]: 2}
    assert (
        len(audit.prompt_uuids)
        + sum(count - 1 for count in audit.duplicate_uuid_multiplicity.values())
        == audit.row_count
    )


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
    assert len(audit.prompt_uuids) == 1
    assert audit.duplicate_uuid_multiplicity == {audit.prompt_uuids[0]: 2}


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
