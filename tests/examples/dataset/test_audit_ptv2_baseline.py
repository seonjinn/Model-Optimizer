from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import stat
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


def _write_canonical_baseline_audit(tmp_path: Path):
    module = _load()
    occurrences = ("1" * 64, "2" * 64, "1" * 64)
    unique = tuple(sorted(set(occurrences)))
    expected = module.BaselineExpectation(
        "5c89e01dd720ae0f4058445ed49c5fb68a03c76e",
        1,
        {"chat": 3},
        2,
        4,
        1,
        "chat",
        True,
        "b" * 64,
    )
    audit = module.BaselineAudit(
        source_revision=expected.source_revision,
        source_manifest_sha256=expected.source_manifest_sha256,
        row_count=3,
        split_rows={"chat": 3},
        files=(module.SourceFile("/immutable/raw-chat.parquet", 123, "a" * 64),),
        occurrence_count=3,
        unique_prompt_count=2,
        occurrence_prompt_ids=occurrences,
        occurrence_prompt_ids_sha256=hashlib.sha256(module.canonical_json(occurrences)).hexdigest(),
        exclusion_prompt_ids=unique,
        exclusion_prompt_ids_sha256=hashlib.sha256(module.canonical_json(unique)).hexdigest(),
        duplicate_uuid_multiplicity={"1" * 64: 2},
        physical_row_count=4,
        selection_policy="hf-streaming-sorted-parquet-take",
        selection_boundary=module.SelectionBoundary("chat-0.parquet", 3, 4, 1),
    )
    path = tmp_path / "AUDIT.json"
    module.write_audit_receipt(audit, path)
    return module, path, audit, expected


def _rewrite_payload(path: Path, mutate, *, reconcile_self_hash: bool = True) -> str:
    payload = json.loads(path.read_bytes())
    mutate(payload)
    if reconcile_self_hash:
        body = {key: value for key, value in payload.items() if key != "receipt_sha256"}
        payload["receipt_sha256"] = hashlib.sha256(
            json.dumps(body, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
    path.write_bytes(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
        + b"\n"
    )
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_loader_authenticates_exact_canonical_baseline_audit(tmp_path: Path) -> None:
    module, path, audit, expected = _write_canonical_baseline_audit(tmp_path)
    file_sha256 = hashlib.sha256(path.read_bytes()).hexdigest()

    loaded, lineage_sha256 = module.load_baseline_audit_receipt(path, file_sha256, expected)

    assert loaded == audit
    assert lineage_sha256 == file_sha256
    assert set(tmp_path.iterdir()) == {path}


@pytest.mark.parametrize(
    ("mutate", "reconcile_self_hash", "match"),
    [
        (lambda payload: payload.__setitem__("smuggled", True), True, "key set"),
        (
            lambda payload: payload.__setitem__("receipt_sha256", "0" * 64),
            False,
            "self-hash",
        ),
        (
            lambda payload: payload["duplicate_uuid_multiplicity"].clear(),
            True,
            "multiplicity",
        ),
        (
            lambda payload: payload.__setitem__("occurrence_prompt_ids_sha256", "0" * 64),
            True,
            "occurrence prompt UUID digest",
        ),
        (
            lambda payload: payload.__setitem__("exclusion_prompt_ids", ["2" * 64]),
            True,
            "exclusion prompt UUIDs",
        ),
    ],
)
def test_loader_replays_schema_self_hash_and_uuid_invariants(
    tmp_path: Path, mutate, reconcile_self_hash: bool, match: str
) -> None:
    module, path, _, expected = _write_canonical_baseline_audit(tmp_path)
    file_sha256 = _rewrite_payload(path, mutate, reconcile_self_hash=reconcile_self_hash)

    with pytest.raises(module.AuditError, match=match):
        module.load_baseline_audit_receipt(path, file_sha256, expected)


def test_loader_rejects_wrong_whole_file_hash_and_noncanonical_bytes(tmp_path: Path) -> None:
    module, path, _, expected = _write_canonical_baseline_audit(tmp_path)

    with pytest.raises(module.AuditError, match="whole-file SHA-256"):
        module.load_baseline_audit_receipt(path, "0" * 64, expected)

    path.write_bytes(path.read_bytes().replace(b'"files":', b'"files" :', 1))
    file_sha256 = hashlib.sha256(path.read_bytes()).hexdigest()
    with pytest.raises(module.AuditError, match="canonical JSON"):
        module.load_baseline_audit_receipt(path, file_sha256, expected)


def test_loader_rejects_hardlink_fifo_and_oversized_inputs(tmp_path: Path) -> None:
    module, path, _, expected = _write_canonical_baseline_audit(tmp_path)
    file_sha256 = hashlib.sha256(path.read_bytes()).hexdigest()
    alias = tmp_path / "AUDIT.alias.json"
    os.link(path, alias)
    with pytest.raises(module.AuditError, match="single-link regular file"):
        module.load_baseline_audit_receipt(path, file_sha256, expected)

    path.unlink()
    alias.unlink()
    fifo = tmp_path / "AUDIT.fifo"
    os.mkfifo(fifo)
    with pytest.raises(module.AuditError, match="single-link regular file"):
        module.load_baseline_audit_receipt(fifo, "0" * 64, expected)

    oversized = tmp_path / "oversized.json"
    oversized.write_bytes(b"123456789")
    with pytest.raises(module.AuditError, match="too large"):
        module._stable_single_link_regular_bytes(oversized, max_bytes=8)


def test_loader_rejects_path_rebind_after_descriptor_read(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module, path, _, expected = _write_canonical_baseline_audit(tmp_path)
    file_sha256 = hashlib.sha256(path.read_bytes()).hexdigest()
    replacement = tmp_path / "replacement.json"
    replacement.write_bytes(path.read_bytes())
    displaced = tmp_path / "displaced.json"
    original_fstat = module.os.fstat
    regular_fstats = 0

    def swap_after_file_read(descriptor: int):
        nonlocal regular_fstats
        result = original_fstat(descriptor)
        if stat.S_ISREG(result.st_mode):
            regular_fstats += 1
            if regular_fstats == 2:
                path.rename(displaced)
                replacement.rename(path)
        return result

    monkeypatch.setattr(module.os, "fstat", swap_after_file_read)

    with pytest.raises(module.AuditError, match=r"changed|rebound"):
        module.load_baseline_audit_receipt(path, file_sha256, expected)


def test_loader_rejects_file_growth_at_the_read_boundary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module, path, _, expected = _write_canonical_baseline_audit(tmp_path)
    file_sha256 = hashlib.sha256(path.read_bytes()).hexdigest()
    original_read = module.os.read
    grew = False

    def read_then_grow(descriptor: int, size: int) -> bytes:
        nonlocal grew
        block = original_read(descriptor, size)
        if block and not grew:
            grew = True
            with path.open("ab") as stream:
                stream.write(b"x")
        return block

    monkeypatch.setattr(module.os, "read", read_then_grow)

    with pytest.raises(module.AuditError, match=r"changed|too large"):
        module.load_baseline_audit_receipt(path, file_sha256, expected)


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
    assert (
        module.CANONICAL_BASELINE_AUDIT_FILE_SHA256
        == "2469430c144d9b86850901df0a28cb810b437555ee894771b796b1386d1c18b5"
    )
    assert (
        module.CANONICAL_BASELINE_OCCURRENCE_PROMPT_IDS_SHA256
        == "863470b22925d74228d31b1c2433d9461d25e8d02a298cb3a08a6bc99be55060"
    )
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
