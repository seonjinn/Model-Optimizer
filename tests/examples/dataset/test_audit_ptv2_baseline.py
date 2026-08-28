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


def _source_manifest(root: Path, revision: str) -> Path:
    module = _load()
    files = []
    for link in sorted(path for path in root.iterdir() if path.is_symlink()):
        resolved = link.resolve(strict=True)
        files.append(
            {
                "name": link.name,
                "bytes": resolved.stat().st_size,
                "sha256": module._sha256_file(resolved),
            }
        )
    manifest = root / "SOURCE_MANIFEST.json"
    manifest.write_bytes(module.canonical_json({"source_revision": revision, "files": files}))
    return manifest


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


@pytest.mark.parametrize(
    ("messages", "tools", "match"),
    [
        (json.dumps({"role": "user", "content": "not a list"}), None, "messages must be a list"),
        (
            json.dumps([{"role": "user", "content": "valid"}]),
            json.dumps({"type": "function"}),
            "tools must be a list",
        ),
    ],
)
def test_audit_rejects_malformed_decoded_identity_inputs(
    tmp_path: Path, messages: str, tools: str | None, match: str
) -> None:
    module = _load()
    root = tmp_path / "data"
    root.mkdir()
    target = tmp_path / "raw-chat.parquet"
    columns: dict[str, list[str]] = {"messages": [messages]}
    if tools is not None:
        columns["tools"] = [tools]
    pq.write_table(pa.table(columns), target)
    (root / "chat-0.parquet").symlink_to(target)
    expected = module.BaselineExpectation(
        "5c89e01dd720ae0f4058445ed49c5fb68a03c76e", 1, {"chat": 1}
    )

    with pytest.raises(ValueError, match=match):
        module.audit_baseline(root, expected)


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
    assert module.EXPECTED_BASELINE.physical_row_count == 1_309_377
    assert module.EXPECTED_BASELINE.excluded_tail_rows == 9_377
    assert module.EXPECTED_BASELINE.excluded_tail_split == "multilingual_de"
    assert module.EXPECTED_BASELINE.source_manifest_required


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


def test_audit_rejects_truncated_physical_source(tmp_path: Path) -> None:
    module = _load()
    root = tmp_path / "data"
    root.mkdir()
    _shard(root, "chat-0.parquet", ["chat-0", "chat-1"])
    _shard(root, "multilingual_de-0.parquet", ["de-0", "de-1"])
    expected = module.BaselineExpectation(
        "5c89e01dd720ae0f4058445ed49c5fb68a03c76e",
        2,
        {"chat": 2, "multilingual_de": 1},
        physical_row_count=5,
        excluded_tail_rows=2,
        excluded_tail_split="multilingual_de",
    )
    with pytest.raises(module.AuditError, match="physical row count mismatch"):
        module.audit_baseline(root, expected)


def test_audit_rejects_wrong_excluded_tail_count(tmp_path: Path) -> None:
    module = _load()
    root = tmp_path / "data"
    root.mkdir()
    _shard(root, "chat-0.parquet", ["chat-0", "chat-1"])
    _shard(root, "multilingual_de-0.parquet", ["de-0", "de-1", "de-2", "de-3"])
    expected = module.BaselineExpectation(
        "5c89e01dd720ae0f4058445ed49c5fb68a03c76e",
        2,
        {"chat": 2, "multilingual_de": 1},
        physical_row_count=6,
        excluded_tail_rows=2,
        excluded_tail_split="multilingual_de",
    )
    with pytest.raises(module.AuditError, match="excluded tail count mismatch"):
        module.audit_baseline(root, expected)


def test_audit_rejects_non_german_excluded_tail(tmp_path: Path) -> None:
    module = _load()
    root = tmp_path / "data"
    root.mkdir()
    _shard(root, "chat-0.parquet", ["chat-0", "chat-1"])
    _shard(root, "code-0.parquet", ["code-0", "code-1", "code-2"])
    expected = module.BaselineExpectation(
        "5c89e01dd720ae0f4058445ed49c5fb68a03c76e",
        2,
        {"chat": 2, "code": 1},
        physical_row_count=5,
        excluded_tail_rows=2,
        excluded_tail_split="multilingual_de",
    )
    with pytest.raises(module.AuditError, match="excluded tail split mismatch"):
        module.audit_baseline(root, expected)


def test_audit_binds_revision_to_source_manifest_file_hashes(tmp_path: Path) -> None:
    module = _load()
    root = tmp_path / "data"
    root.mkdir()
    _shard(root, "chat-0.parquet", ["chat"])
    revision = "5c89e01dd720ae0f4058445ed49c5fb68a03c76e"
    manifest = _source_manifest(root, revision)
    expected = module.BaselineExpectation(
        revision,
        1,
        {"chat": 1},
        source_manifest_sha256=module.sha256_bytes(manifest.read_bytes()),
    )
    audit = module.audit_baseline(root, expected, source_manifest=manifest)
    assert audit.source_manifest_sha256 == module.sha256_bytes(manifest.read_bytes())
    manifest.write_bytes(
        module.canonical_json(
            {
                "source_revision": revision,
                "files": [{"name": "chat-0.parquet", "bytes": 0, "sha256": "0" * 64}],
            }
        )
    )
    with pytest.raises(module.AuditError, match="source manifest file mismatch"):
        module.audit_baseline(root, expected, source_manifest=manifest)


def test_audit_rejects_source_manifest_digest_mismatch(tmp_path: Path) -> None:
    module = _load()
    root = tmp_path / "data"
    root.mkdir()
    _shard(root, "chat-0.parquet", ["chat"])
    revision = "5c89e01dd720ae0f4058445ed49c5fb68a03c76e"
    manifest = _source_manifest(root, revision)
    expected = module.BaselineExpectation(revision, 1, {"chat": 1}, source_manifest_sha256="0" * 64)
    with pytest.raises(module.AuditError, match="source manifest SHA-256 mismatch"):
        module.audit_baseline(root, expected, source_manifest=manifest)


def test_audit_rejects_unbound_revision_without_source_manifest(tmp_path: Path) -> None:
    module = _load()
    root = tmp_path / "data"
    root.mkdir()
    _shard(root, "chat-0.parquet", ["arbitrary"])
    expected = module.BaselineExpectation(
        "5c89e01dd720ae0f4058445ed49c5fb68a03c76e", 1, {"chat": 1}, source_manifest_required=True
    )
    with pytest.raises(module.AuditError, match="source manifest is required"):
        module.audit_baseline(root, expected)


def test_receipt_publication_refuses_existing_file_and_preserves_bytes(tmp_path: Path) -> None:
    module = _load()
    root = tmp_path / "data"
    root.mkdir()
    _shard(root, "chat-0.parquet", ["a"])
    audit = module.audit_baseline(
        root,
        module.BaselineExpectation("5c89e01dd720ae0f4058445ed49c5fb68a03c76e", 1, {"chat": 1}),
    )
    output = tmp_path / "receipt.json"
    module.write_audit_receipt(audit, output)
    first = output.read_bytes()
    with pytest.raises(FileExistsError, match="immutable audit receipt already exists"):
        module.write_audit_receipt(audit, output)
    assert output.read_bytes() == first
