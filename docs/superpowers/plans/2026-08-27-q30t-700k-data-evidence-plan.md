# Q30 Thinking SWE-Heavy 700K Data Evidence Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Authenticate the historical, held-out, source, row-schema, tokenizer, and target-policy evidence needed to publish and independently replay the exact Qwen3-30B-A3B-Thinking-2507 SWE-heavy 700K corpus on Ptyche.

**Architecture:** Keep prompt identity and receipt authentication in small reusable modules, then make the existing complement builder consume those authenticated views without changing its selection/publication core. Receipt production, independent replay, approval-root insertion, corpus construction, and corpus replay are separate fail-closed transitions; remote work executes immutable Git objects in CPU-only Ptyche jobs and publishes only through the hardened descriptor-bound atomic boundary.

**Tech Stack:** Python 3.12+, approved Ptyche `/usr/bin/python3.12`, pytest, PyArrow, SQLite, canonical JSON and SHA-256 receipts, descriptor-relative POSIX I/O, Bash, SLURM/Pyxis, Ruff, Pyright, ShellCheck.

**Spec:** `docs/superpowers/specs/2026-08-27-q30t-swe-heavy-700k-evidence-launch-design.md`

## Global Constraints

- This plan implements Subproject A only. Q30 controller trust integration, runtime qualification, canaries, and training launch are outside this plan.
- The scientific identity is exactly `ptv2-ptv3-complement-700k-qwen3-30ba3b-thinking-swe-heavy-v1`.
- The corpus contains exactly 700,000 occurrences once, with no redistribution, replacement, padding, cycling, or duplicate exposure.
- Quotas are exactly `ptv2_stem=200000`, `ptv2_multilingual_ja=25000`, `ptv2_multilingual_es=25000`, `ptv2_multilingual_fr=25000`, `ptv2_multilingual_it=25000`, `ptv3_swe_v3=180000`, `ptv3_interactive_agentic_swe=60000`, and `ptv3_general_tool_trajectories=160000` in that order.
- Target and tokenizer repository are `Qwen/Qwen3-30B-A3B-Thinking-2507`; both revisions are `144afc2f379b542fdd4e85a1fcd5e1f79112d95d`.
- The reviewed tokenizer receipt whole-file SHA-256 is `5ba642c455e60b67eca295dce92dd7da47292fdba66c5f9d269669c14cafc509`.
- The canonical historical audit whole-file SHA-256 is `2469430c144d9b86850901df0a28cb810b437555ee894771b796b1386d1c18b5`; its ordered historical occurrence digest is `863470b22925d74228d31b1c2433d9461d25e8d02a298cb3a08a6bc99be55060`.
- The reviewed producer commit is `212d01516b32e5a81f077d7f2f26ec8de38707c6`; its target tree is `7bd176a868273ca8acf8db0ec1ab528f7972b0ae5bcf67ed4b3726368b9f7d13`.
- The five held-out names and canonical order are exactly `speed`, `math`, `code`, `swe`, `tool`; an unavailable authoritative source fails the transition and never produces an empty receipt.
- PTV2 evidence covers exactly 201 files. PTV3 evidence covers exactly 100 files and 670,399 rows.
- Existing destinations are adopted only when their stable single-link bytes are exact; foreign paths are preserved and rejected. No recovery path recursively deletes or pathname-unlinks a tree.
- Receipt generation, independent replay, approval insertion, builder execution, and corpus replay are separate commits/jobs and separate review gates.
- Every remote submission first passes `sbatch --test-only`, uses a clean pushed commit, writes bounded logs below `/lustre/fsw/coreai_dlalgo_llm/users/sna/modelopt-qwen3-drafter-training/receipts/q30t-ptv23-complement-700k-v1`, and is monitored for five minutes with one filtered scheduler query per minute.
- Every Ptyche preflight and runner invokes exactly `/usr/bin/python3.12`; the authenticated interpreter must report Python 3.12 or newer before any receipt or corpus mutation.
- Each task uses strict RED-GREEN tests and receives a fresh hostile review before its commit.
- Do not modify `uv.lock` as part of this plan.

---

## Cross-Plan Handoffs

- Controller plan: `docs/superpowers/plans/2026-08-27-q30t-700k-controller-trust-plan.md`.
- Data Tasks 1-4B have no controller dependency and may execute independently while controller Task 1 is being implemented.
- Controller Task 1 exclusively owns `tools/launcher/common/specdec/q30t_tokenizer_receipt.py`, `tools/launcher/tests/test_q30t_tokenizer_receipt.py`, the approved tokenizer hash set, and `load_q30t_tokenizer_receipt(path: Path, *, expected_sha256: str) -> dict[str, object]`.
- Controller Task 1 must land and pass its focused tests before data Task 5 begins. Data Task 5 imports and delegates to that interface; it never edits or tests the loader implementation itself.
- Data Task 5 owns only builder integration. Data Task 6 freezes policy/config, Task 8 freezes builder approval semantics, and Task 11 freezes the integrated data slice; all three gates must be complete before controller Tasks 3-9 begin. Data Tasks 9-10 then produce the remote evidence and corpus artifacts consumed by later controller authorization transitions.
- A handoff is complete only when the producer plan supplies a reviewed commit plus focused GREEN output; sharing an uncommitted worktree does not satisfy the dependency.

---

## File Structure

### Shared identity and historical lineage

- Modify `examples/dataset/specdec_identity.py`: own the one prompt-bearing-message filter and canonical prompt identity.
- Modify `examples/dataset/audit_ptv2_baseline.py`: use the shared identity and authenticate canonical `BaselineAudit` bytes directly.
- Modify `examples/dataset/build_qwen4b_ptv23_complement.py`: convert an authenticated `BaselineAudit` to `HistoricalExclusion`; do not define a second historical receipt schema.
- Modify `tests/examples/dataset/test_specdec_identity.py`, `tests/examples/dataset/test_audit_ptv2_baseline.py`, and `tests/examples/dataset/test_build_qwen4b_ptv23_complement.py`: prove historical/candidate agreement and direct lineage.

### Held-out evidence

- Create `examples/dataset/build_q30t_heldout_receipts.py`: validate five canonical UUID source artifacts, emit existing consumer receipts plus provenance receipts, and replay both schemas.
- Create `tests/examples/dataset/test_build_q30t_heldout_receipts.py`: cover malformed UUIDs, fixed ordering, provenance, stable-read attacks, and no-clobber publication.
- Modify `examples/dataset/build_qwen4b_ptv23_complement.py`: normalize held-out receipt inputs to the fixed name order.

### Source and row-schema evidence

- Create `examples/dataset/observe_q30t_ptv23_row_schemas.py`: independently observe every staged file's physical messages/tools schema before authoring the mapping policy.
- Create `tests/examples/dataset/test_observe_q30t_ptv23_row_schemas.py`: prove complete observation and fail-closed stage authentication.
- Create `tools/launcher/common/specdec/run_q30t_row_schema_observation.sbatch`, `tools/launcher/common/specdec/submit_q30t_row_schema_observation.sh`, and `tools/launcher/tests/test_q30t_row_schema_observation_runner.py`: immutable CPU-only observation boundary.
- Create `examples/dataset/qwen3_30ba3b_thinking_ptv23_swe_heavy_source_mapping_v1.json`: checked-in eight-category mapping from staged source identities to builder category, license, replay lane, and declared messages/tools fields.
- Create `examples/dataset/build_q30t_ptv23_source_inventory.py`: authenticate PTV2/PTV3 stage roots and completions, revalidate every physical file, prove every row shape, and emit the existing inventory schema.
- Create `tests/examples/dataset/test_build_q30t_ptv23_source_inventory.py`: cover source-plan/completion binding, every-file schema, heterogeneous files, counts, hashes, order, and publication attacks.

### Q30 policy and tokenizer trust

- Consume controller Task 1's `load_q30t_tokenizer_receipt(path: Path, *, expected_sha256: str) -> dict[str, object]`; this plan does not modify its module or tests.
- Modify `examples/dataset/build_qwen4b_ptv23_complement.py` and `tests/examples/dataset/test_build_qwen4b_ptv23_complement.py`: delegate the Q30 builder branch to the controller-owned loader while preserving the Q4 branch.
- Create `examples/dataset/qwen3_30ba3b_thinking_ptv23_swe_heavy_sources_v1.json`: distinct SWE-heavy source requirements.
- Modify `examples/dataset/qwen3_30ba3b_thinking_ptv23_swe_heavy_700k_v1.json`: canonical exact quotas, revision, and distinct requirements binding.
- Modify `examples/dataset/qwen3_30ba3b_thinking_ptv23_swe_heavy_target_v1.json`: canonical target/tokenizer revision and exact child hashes.
- Modify `examples/dataset/ptv23_complement_target_policy.py`: register the exact Q30 SWE-heavy policy tuple without a balanced-identity compatibility exception.
- Modify `tests/examples/dataset/test_q30t_swe_heavy_700k_config.py` and `tests/examples/dataset/test_ptv23_complement_target_policy.py`: freeze the exact bytes and policy isolation.

### Review, approval, build, and replay

- Create `examples/dataset/review_q30t_700k_data_evidence.py`: independently replay held-out, inventory, historical, tokenizer, and policy evidence and emit a canonical approval-candidate report.
- Create `tests/examples/dataset/test_review_q30t_700k_data_evidence.py`: prove report self-hash, canonical order, full tuple binding, and mutation rejection.
- Modify `examples/dataset/build_qwen4b_ptv23_complement.py`: insert only independently reviewed approval roots, use shared prompt identity, and retain full row bytes in selection evidence.
- Create `tools/launcher/common/specdec/run_q30t_700k_data_evidence.sbatch`: CPU-only immutable-Git runner with distinct `heldout`, `inventory`, `review`, `build`, and `replay` operations.
- Create `tools/launcher/common/specdec/submit_q30t_700k_data_evidence.sh`: clean-source, test-only-first submission boundary.
- Create `tools/launcher/tests/test_q30t_700k_data_evidence_runner.py`: static and behavioral runner/submitter contract tests.

---

### Task 1: Unify historical and candidate prompt identity

**Files:**

- Modify: `examples/dataset/specdec_identity.py:12-65`
- Modify: `examples/dataset/audit_ptv2_baseline.py:98-110,203-215`
- Test: `tests/examples/dataset/test_specdec_identity.py`
- Test: `tests/examples/dataset/test_audit_ptv2_baseline.py`
- Test: `tests/examples/dataset/test_trajectory_schema.py`

**Interfaces:**

- Consumes: `STORAGE_ONLY_FIELDS: frozenset[str]` already defined in `specdec_identity.py`.
- Produces: `prompt_messages(messages: object) -> list[dict[str, object]]`, `canonicalize_prompt(messages: list[dict[str, object]], tools: list[dict[str, object]] | None) -> bytes`, and `prompt_uuid(...) -> str`.
- Invariant: assistant/tool messages do not change prompt identity; `system`, `developer`, and `user` order and tool-schema bytes do.

- [ ] **Step 1: Add the failing shared-identity tests**

```python
def test_prompt_uuid_ignores_only_non_prompt_roles_and_storage_fields() -> None:
    prompt = [
        {"role": "system", "content": "policy", "row_index": 7},
        {"role": "developer", "content": "contract"},
        {"role": "user", "content": "fix it", "cache_path": "/tmp/x"},
    ]
    completed = prompt + [
        {"role": "assistant", "content": "answer A"},
        {"role": "tool", "content": "tool result"},
    ]
    tools = [{"type": "function", "function": {"name": "shell", "parameters": {}}}]
    assert prompt_uuid(completed, tools) == prompt_uuid(prompt, tools)
    assert prompt_uuid(prompt, tools) != prompt_uuid(prompt, [])
    assert prompt_uuid(prompt, tools) != prompt_uuid(list(reversed(prompt)), tools)


def test_prompt_messages_rejects_a_conversation_without_prompt_roles() -> None:
    with pytest.raises(ValueError, match="prompt-bearing"):
        prompt_messages([{"role": "assistant", "content": "only completion"}])
```

- [ ] **Step 2: Run the identity tests and verify RED**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 ../q4-ptv2-ab-integration/.venv/bin/python -m pytest -q \
  tests/examples/dataset/test_specdec_identity.py::test_prompt_uuid_ignores_only_non_prompt_roles_and_storage_fields \
  tests/examples/dataset/test_specdec_identity.py::test_prompt_messages_rejects_a_conversation_without_prompt_roles
```

Expected: FAIL because `prompt_messages` is not exported and the current `prompt_uuid` includes assistant/tool messages.

- [ ] **Step 3: Implement the single prompt-bearing filter**

```python
PROMPT_ROLES = frozenset({"system", "developer", "user"})


def prompt_messages(messages: object) -> list[dict[str, object]]:
    if not isinstance(messages, list) or any(not isinstance(item, dict) for item in messages):
        raise ValueError("messages must be a list of mappings")
    retained = [dict(item) for item in messages if item.get("role") in PROMPT_ROLES]
    if not retained:
        raise ValueError("conversation has no prompt-bearing messages")
    return retained


def canonicalize_prompt(
    messages: list[dict[str, object]], tools: list[dict[str, object]] | None
) -> bytes:
    return canonical_json(
        {"messages": _normalize(prompt_messages(messages)), "tools": _normalize(tools or [])}
    )
```

Export `PROMPT_ROLES` and `prompt_messages`. In `audit_ptv2_baseline.py`, delete its private `_prompt_messages` and call `prompt_uuid(_decode(row[message_column]), tools)` so audit and candidates cannot diverge.

- [ ] **Step 4: Run identity, audit, and trajectory GREEN tests**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 ../q4-ptv2-ab-integration/.venv/bin/python -m pytest -q \
  tests/examples/dataset/test_specdec_identity.py \
  tests/examples/dataset/test_audit_ptv2_baseline.py \
  tests/examples/dataset/test_trajectory_schema.py
```

Expected: PASS; the audit assistant-completion test and storage-field tests share the same digest implementation.

- [ ] **Step 5: Request hostile review and commit the isolated identity slice**

Review prompt: “Verify that only roles are filtered, that `_normalize` removes exactly `STORAGE_ONLY_FIELDS`, that tool schemas remain bound, and that no caller can supply a prefiltered alternative identity.”

```bash
git add examples/dataset/specdec_identity.py examples/dataset/audit_ptv2_baseline.py \
  tests/examples/dataset/test_specdec_identity.py tests/examples/dataset/test_audit_ptv2_baseline.py \
  tests/examples/dataset/test_trajectory_schema.py
git commit -S -s -m "fix(dataset): unify Q30 prompt identity"
```

### Task 2: Authenticate the canonical BaselineAudit directly

**Files:**

- Modify: `examples/dataset/audit_ptv2_baseline.py:57-73,278-317`
- Modify: `examples/dataset/build_qwen4b_ptv23_complement.py:274-285,598-667`
- Test: `tests/examples/dataset/test_audit_ptv2_baseline.py`
- Test: `tests/examples/dataset/test_build_qwen4b_ptv23_complement.py`

**Interfaces:**

- Consumes: `BaselineAudit`, `BaselineExpectation`, `EXPECTED_BASELINE`, and the stable regular-byte reader pattern already used by the builder.
- Produces: `load_baseline_audit_receipt(path: Path, expected_file_sha256: str, expected: BaselineExpectation = EXPECTED_BASELINE) -> tuple[BaselineAudit, str]`.
- Builder keeps `load_historical_exclusion(path: Path, *, expected_sha256: str, expected_occurrence_count: int = 1_300_000) -> HistoricalExclusion` as its public adapter.

- [ ] **Step 1: Add failing canonical-audit authentication tests**

```python
def test_builder_authenticates_original_baseline_audit_without_derivative_schema(
    tmp_path: Path,
) -> None:
    audit_path, audit = write_canonical_baseline_audit(tmp_path)
    file_sha = hashlib.sha256(audit_path.read_bytes()).hexdigest()
    loaded = builder.load_historical_exclusion(audit_path, expected_sha256=file_sha)
    assert loaded.file_sha256 == file_sha
    assert loaded.ordered_prompt_uuids_sha256 == audit.occurrence_prompt_ids_sha256
    assert loaded.prompt_uuids == frozenset(audit.exclusion_prompt_ids)


def test_baseline_audit_loader_rejects_rebound_and_multiply_linked_paths(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    audit_path, _ = write_canonical_baseline_audit(tmp_path)
    alias = tmp_path / "AUDIT.alias.json"
    os.link(audit_path, alias)
    with pytest.raises(AuditError, match="single-link regular file"):
        load_baseline_audit_receipt(audit_path, hashlib.sha256(audit_path.read_bytes()).hexdigest())
```

- [ ] **Step 2: Run the direct-authentication tests and verify RED**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 ../q4-ptv2-ab-integration/.venv/bin/python -m pytest -q \
  tests/examples/dataset/test_audit_ptv2_baseline.py -k 'loader or rebound' \
  tests/examples/dataset/test_build_qwen4b_ptv23_complement.py -k 'original_baseline'
```

Expected: FAIL because the audit module has no loader and the builder expects `ptv2-historical-occurrence-receipt-v1` instead of canonical `BaselineAudit` bytes.

- [ ] **Step 3: Implement exact canonical receipt reconstruction**

```python
def load_baseline_audit_receipt(
    path: Path,
    expected_file_sha256: str,
    expected: BaselineExpectation = EXPECTED_BASELINE,
) -> tuple[BaselineAudit, str]:
    raw = _stable_single_link_regular_bytes(path, max_bytes=512 * 1024 * 1024)
    if sha256(raw).hexdigest() != expected_file_sha256:
        raise AuditError("baseline audit whole-file SHA-256 mismatch")
    payload = json.loads(raw)
    if raw != canonical_json(payload) + b"\n":
        raise AuditError("baseline audit must use canonical JSON bytes")
    receipt_sha = payload.pop("receipt_sha256", None)
    if receipt_sha != sha256_bytes(canonical_json(payload)):
        raise AuditError("baseline audit self-hash does not reconcile")
    audit = _baseline_audit_from_payload(payload)
    _validate_baseline_audit(audit, expected)
    return audit, expected_file_sha256
```

`_baseline_audit_from_payload` must require the exact `asdict(BaselineAudit)` key set and reconstruct `SourceFile` and `SelectionBoundary`; `_validate_baseline_audit` must recompute occurrence count, sorted unique IDs, duplicate multiplicity, both UUID digests, row/split totals, revision, selection policy, and boundary. Stable reading uses `O_RDONLY|O_NOFOLLOW|O_NONBLOCK`, requires a regular file with `st_nlink == 1`, caps bytes, and compares descriptor identity before/after with a final pathname rebind check.

- [ ] **Step 4: Replace the builder’s derivative parser with a direct adapter**

```python
def load_historical_exclusion(
    path: Path,
    *,
    expected_sha256: str,
    expected_occurrence_count: int = 1_300_000,
) -> HistoricalExclusion:
    audit, file_sha256 = load_baseline_audit_receipt(path, expected_sha256)
    if audit.occurrence_count != expected_occurrence_count:
        raise ComplementError("historical occurrence count mismatch")
    return HistoricalExclusion(
        file_sha256=file_sha256,
        receipt_sha256=sha256(_canonical_json(asdict(audit))).hexdigest(),
        occurrence_count=audit.occurrence_count,
        ordered_prompt_uuids_sha256=audit.occurrence_prompt_ids_sha256,
        prompt_uuids=frozenset(audit.exclusion_prompt_ids),
        unique_prompt_uuids_sha256=audit.exclusion_prompt_ids_sha256,
        duplicate_uuid_multiplicity=dict(audit.duplicate_uuid_multiplicity),
    )
```

Do not write a derivative file. The lineage root remains the original file SHA `2469430c144d9b86850901df0a28cb810b437555ee894771b796b1386d1c18b5`.

- [ ] **Step 5: Run GREEN and freeze the historical lineage values**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 ../q4-ptv2-ab-integration/.venv/bin/python -m pytest -q \
  tests/examples/dataset/test_audit_ptv2_baseline.py \
  tests/examples/dataset/test_build_qwen4b_ptv23_complement.py -k historical
```

Expected: PASS, including exact occurrence digest `863470b22925d74228d31b1c2433d9461d25e8d02a298cb3a08a6bc99be55060`.

- [ ] **Step 6: Request hostile review and commit**

Review prompt: “Attempt canonical-key smuggling, self-hash substitution, duplicate multiplicity mismatch, hardlink/FIFO/rebind attacks, and any write of a derivative historical receipt.”

```bash
git add examples/dataset/audit_ptv2_baseline.py \
  examples/dataset/build_qwen4b_ptv23_complement.py \
  tests/examples/dataset/test_audit_ptv2_baseline.py \
  tests/examples/dataset/test_build_qwen4b_ptv23_complement.py
git commit -S -s -m "fix(dataset): authenticate canonical baseline audit"
```

### Task 3: Produce canonical held-out consumer and provenance receipts

**Files:**

- Create: `examples/dataset/build_q30t_heldout_receipts.py`
- Create: `tests/examples/dataset/test_build_q30t_heldout_receipts.py`
- Modify: `examples/dataset/build_qwen4b_ptv23_complement.py:110-117,670-732`
- Test: `tests/examples/dataset/test_build_qwen4b_ptv23_complement.py`

**Interfaces:**

- Consumes: `qwen4b_b_atomic.atomic_publish_bytes(destination: Path, payload: bytes, *, job_id: str) -> None`.
- Produces: `HeldOutSource`, `load_heldout_source(path: Path, expected_sha256: str) -> tuple[str, ...]`, `build_heldout_receipt(source: HeldOutSource) -> bytes`, `build_heldout_provenance(source: HeldOutSource, receipt: bytes) -> bytes`, `verify_heldout_receipt(raw: bytes) -> tuple[str, ...]`, and `verify_heldout_provenance(raw: bytes, *, receipt: bytes, source: HeldOutSource) -> dict[str, object]`.
- CLI: `build_q30t_heldout_receipts.py build --sources-manifest PATH --sources-manifest-sha256 SHA256 --output-root ABSOLUTE_PATH`; `verify` adds `--receipt-root ABSOLUTE_PATH --review-output ABSOLUTE_PATH`.

- [ ] **Step 1: Add failing source validation and canonical-order tests**

```python
@pytest.mark.parametrize(
    "uuids,error",
    [
        (["A" * 64], "lowercase"),
        (["a" * 64, "a" * 64], "duplicate"),
        (["b" * 64, "a" * 64], "sorted"),
        (["a" * 63], "64"),
        ([], "empty"),
    ],
)
def test_heldout_source_rejects_noncanonical_uuid_sets(
    tmp_path: Path, uuids: list[str], error: str
) -> None:
    source = tmp_path / "source.json"
    source.write_bytes(canonical_json({"prompt_uuids": uuids}) + b"\n")
    with pytest.raises(HeldOutError, match=error):
        load_heldout_source(source, hashlib.sha256(source.read_bytes()).hexdigest())


def test_receipts_are_emitted_in_fixed_name_order_independent_of_manifest_order(
    tmp_path: Path,
) -> None:
    manifest = heldout_manifest(tmp_path, names=["tool", "speed", "swe", "code", "math"])
    outputs = build_all(manifest, tmp_path / "receipts")
    assert tuple(outputs) == ("speed", "math", "code", "swe", "tool")
```

- [ ] **Step 2: Run held-out tests and verify RED**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 ../q4-ptv2-ab-integration/.venv/bin/python -m pytest -q \
  tests/examples/dataset/test_build_q30t_heldout_receipts.py
```

Expected: collection FAIL because `build_q30t_heldout_receipts.py` does not exist.

- [ ] **Step 3: Implement strict source and manifest types**

```python
HELD_OUT_ORDER = ("speed", "math", "code", "swe", "tool")


@dataclass(frozen=True)
class HeldOutSource:
    name: str
    path: Path
    file_sha256: str
    repository: str
    revision: str
    variant: str
    prompt_uuids: tuple[str, ...]


def load_heldout_source(path: Path, expected_sha256: str) -> tuple[str, ...]:
    raw = stable_single_link_regular_bytes(path, max_bytes=256 * 1024 * 1024)
    if sha256(raw).hexdigest() != expected_sha256:
        raise HeldOutError("held-out source whole-file SHA-256 mismatch")
    payload = json.loads(raw)
    if raw != canonical_json(payload) + b"\n" or set(payload) != {"prompt_uuids"}:
        raise HeldOutError("held-out source is not canonical")
    uuids = payload["prompt_uuids"]
    if not isinstance(uuids, list) or not uuids:
        raise HeldOutError("held-out source cannot be empty")
    if any(not is_lower_hex(value, 64) for value in uuids):
        raise HeldOutError("held-out UUID must be lowercase 64-hex")
    if uuids != sorted(uuids):
        raise HeldOutError("held-out UUIDs must be sorted")
    if len(uuids) != len(set(uuids)):
        raise HeldOutError("held-out UUIDs contain a duplicate")
    return tuple(uuids)
```

The manifest has exact root keys `schema_version` and `sources`; each source has exact keys `name`, `path`, `sha256`, `repository`, `revision`, and `variant`. Require five unique exact names, absolute source paths, nonempty repository/variant, and a revision that satisfies `_is_lower_hex(revision, 40) or _is_lower_hex(revision, 64)`.

- [ ] **Step 4: Implement existing consumer bytes and separate provenance bytes**

```python
def build_heldout_receipt(source: HeldOutSource) -> bytes:
    body = {
        "schema_version": "specdec-held-out-uuid-receipt-v1",
        "prompt_uuids": list(source.prompt_uuids),
        "prompt_uuids_sha256": sha256(canonical_json(list(source.prompt_uuids))).hexdigest(),
    }
    return canonical_json(body | {"receipt_sha256": sha256(canonical_json(body)).hexdigest()}) + b"\n"


def build_heldout_provenance(source: HeldOutSource, receipt: bytes) -> bytes:
    receipt_payload = verify_heldout_receipt(receipt)
    body = {
        "schema_version": "q30t-held-out-provenance-v1",
        "name": source.name,
        "repository": source.repository,
        "revision": source.revision,
        "variant": source.variant,
        "source_path": str(source.path),
        "source_file_sha256": source.file_sha256,
        "prompt_uuids_sha256": sha256(canonical_json(list(receipt_payload))).hexdigest(),
        "consumer_receipt_file_sha256": sha256(receipt).hexdigest(),
    }
    return canonical_json(body | {"receipt_sha256": sha256(canonical_json(body)).hexdigest()}) + b"\n"
```

Publish `NAME.json` and `NAME.provenance.json` with `atomic_publish_bytes`; after publication, reopen both via the stable reader and require byte equality. On any failure retain private scratch and preserve every existing destination.

- [ ] **Step 5: Add publication and rebind attack tests**

```python
def test_publication_never_clobbers_a_foreign_destination(tmp_path: Path) -> None:
    destination = tmp_path / "speed.json"
    destination.write_bytes(b"foreign\n")
    with pytest.raises(FileExistsError):
        publish_receipt(destination, b"canonical\n", job_id="17")
    assert destination.read_bytes() == b"foreign\n"


def test_source_reader_rejects_fifo_hardlink_and_path_rebind(tmp_path: Path) -> None:
    source = canonical_source(tmp_path)
    assert_rejects_fifo_without_blocking(load_heldout_source, tmp_path)
    assert_rejects_second_link(load_heldout_source, source)
    assert_rejects_path_replacement(load_heldout_source, source)
```

- [ ] **Step 6: Make builder held-out union order canonical**

Sort normalized triples by `HELD_OUT_ORDER.index(name)` before reading. Reject duplicate names even when set equality would otherwise hide them. Store `receipt_names` and `receipt_file_sha256s` in the fixed order.

- [ ] **Step 7: Run held-out and builder GREEN tests**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 ../q4-ptv2-ab-integration/.venv/bin/python -m pytest -q \
  tests/examples/dataset/test_build_q30t_heldout_receipts.py \
  tests/examples/dataset/test_build_qwen4b_ptv23_complement.py -k held_out
```

Expected: PASS; missing, duplicate, extra, reordered, empty, malformed, hardlinked, FIFO, and rebound inputs fail closed.

- [ ] **Step 8: Request hostile review and commit**

Review prompt: “Attempt empty synthesis, name/order ambiguity, uppercase UUIDs, provenance/consumer substitution, destination symlink/hardlink/FIFO, parent rebind, and cleanup of foreign paths.”

```bash
git add examples/dataset/build_q30t_heldout_receipts.py \
  examples/dataset/build_qwen4b_ptv23_complement.py \
  tests/examples/dataset/test_build_q30t_heldout_receipts.py \
  tests/examples/dataset/test_build_qwen4b_ptv23_complement.py
git commit -S -s -m "feat(dataset): produce Q30 heldout evidence"
```

### Task 4A: Observe staged row schemas before authoring policy

**Files:**

- Create: `examples/dataset/observe_q30t_ptv23_row_schemas.py`
- Create: `tests/examples/dataset/test_observe_q30t_ptv23_row_schemas.py`
- Create: `tools/launcher/common/specdec/run_q30t_row_schema_observation.sbatch`
- Create: `tools/launcher/common/specdec/submit_q30t_row_schema_observation.sh`
- Create: `tools/launcher/tests/test_q30t_row_schema_observation_runner.py`

**Interfaces:**

- Produces: `observe_stage_schemas(inputs: StageInputs) -> bytes`, canonical schema `q30t-ptv23-row-schema-observation-v1`, and CLI flags for exact PTV2/PTV3 plan, completion, root, and output path/SHA bindings.
- The observation records source identity, revision, split, logical path, physical SHA/bytes/row count, format, exact field names, Arrow/JSON value shapes, and per-file row-schema SHA. It contains no category, license, replay-lane, or approval decision.
- Submit CLI: `submit_q30t_row_schema_observation.sh (--test-only|--submit) --source-path PATH --output PATH --slurm-output PATH`.

- [ ] **Step 1: Write failing complete-observation tests**

```python
def test_observer_records_every_file_and_every_row_shape(tmp_path: Path) -> None:
    inputs = stage_inputs(tmp_path, file_rows=((valid_row(),), (valid_row(), valid_row())))
    payload = json.loads(observe_stage_schemas(inputs))
    assert len(payload["files"]) == 2
    assert [item["row_count"] for item in payload["files"]] == [1, 2]
    assert all(item["messages_field"] == "messages" for item in payload["files"])


def test_observer_rejects_a_late_malformed_row(tmp_path: Path) -> None:
    inputs = stage_inputs(tmp_path, file_rows=((valid_row(),), (valid_row(), {"messages": 7})))
    with pytest.raises(ObservationError, match=r"file 2.*row 2"):
        observe_stage_schemas(inputs)
```

- [ ] **Step 2: Run observer tests and verify RED**

```bash
PYTHONDONTWRITEBYTECODE=1 ../q4-ptv2-ab-integration/.venv/bin/python -m pytest -q \
  tests/examples/dataset/test_observe_q30t_ptv23_row_schemas.py
```

Expected: collection FAIL because the observer module is absent.

- [ ] **Step 3: Implement authenticated plan/completion enumeration**

Use a single `StageInputs` dataclass with explicit PTV2/PTV3 plan paths, plan SHAs, completion paths, completion SHAs, and roots. Reuse `verify_q30t_ptv2_full201_transfer.py` parsing for PTV2 and `stage_hf_subset.verify_completion` semantics for PTV3. Enumerate only manifest-bound paths; reject duplicate, absolute, dot-component, missing, or extra records before opening data.

- [ ] **Step 4: Implement every-row physical observation**

For each manifest record, use `O_RDONLY|O_NOFOLLOW|O_NONBLOCK`, require single-link regular bytes, verify staged SHA/size, inspect the Parquet schema or every JSON object, validate every messages/tools value, and finish with descriptor before/after plus final pathname identity checks. Emit canonical newline-terminated bytes with a self-hash; do not publish a policy.

- [ ] **Step 5: Add runner RED tests, then implement the CPU-only immutable runner**

```python
def test_observation_runner_uses_immutable_git_and_cpu_only() -> None:
    runner = RUNNER.read_text()
    assert "#SBATCH --nodes=1" in runner
    assert "#SBATCH --cpus-per-task=32" in runner
    assert "--gpus" not in runner
    assert "git cat-file" in runner or "git archive" in runner
    assert "PYTHONSAFEPATH=1" in runner
```

The submitter requires a clean pushed HEAD and invokes `sbatch --test-only` before `--submit`. The runner authenticates `/usr/bin/python3.12`, requires `sys.version_info >= (3, 12)`, extracts the observer from immutable Git objects into private node-local scratch, invokes it with the approved interpreter, and leaves that private tree to scheduler cleanup.

- [ ] **Step 6: Run observer, runner, and shell GREEN tests**

```bash
PYTHONDONTWRITEBYTECODE=1 ../q4-ptv2-ab-integration/.venv/bin/python -m pytest -q \
  tests/examples/dataset/test_observe_q30t_ptv23_row_schemas.py \
  tools/launcher/tests/test_q30t_row_schema_observation_runner.py
bash -n tools/launcher/common/specdec/run_q30t_row_schema_observation.sbatch \
  tools/launcher/common/specdec/submit_q30t_row_schema_observation.sh
shellcheck tools/launcher/common/specdec/run_q30t_row_schema_observation.sbatch \
  tools/launcher/common/specdec/submit_q30t_row_schema_observation.sh
```

Expected: all commands exit 0.

- [ ] **Step 7: Request hostile review, commit, and push the observer slice**

Review prompt: “Try plan/completion cross-pairing, omitted files, first-row-only inspection, malformed late rows, hardlink/FIFO/rebind, mutable checkout substitution, and policy decisions inside the observer.”

```bash
git add examples/dataset/observe_q30t_ptv23_row_schemas.py \
  tests/examples/dataset/test_observe_q30t_ptv23_row_schemas.py \
  tools/launcher/common/specdec/run_q30t_row_schema_observation.sbatch \
  tools/launcher/common/specdec/submit_q30t_row_schema_observation.sh \
  tools/launcher/tests/test_q30t_row_schema_observation_runner.py
git commit -S -s -m "feat(dataset): observe Q30 source schemas"
git push gitlab HEAD:refs/heads/sj/q30t-ptv23-complement-700k
```

- [ ] **Step 8: Test-only and submit two independent Ptyche observations**

```bash
tools/launcher/common/specdec/submit_q30t_row_schema_observation.sh --test-only \
  --source-path "$PWD" \
  --output /lustre/fsw/coreai_dlalgo_llm/users/sna/modelopt-qwen3-drafter-training/receipts/q30t-ptv23-complement-700k-v1/row-schema/observation-a.json \
  --slurm-output /lustre/fsw/coreai_dlalgo_llm/users/sna/modelopt-qwen3-drafter-training/receipts/q30t-ptv23-complement-700k-v1/logs/row-schema-a-%j.out
```

Expected: test-only succeeds without output. Repeat with `--submit` for `observation-a.json`, then repeat test-only and submit with `observation-b.json`; each runner binds PTV2 root `/lustre/fsw/coreai_dlalgo_llm/users/sna/modelopt-qwen3-drafter-training/assets/q30t-ptv2-full201-source-v1` and PTV3 root `/lustre/fsw/coreai_dlalgo_llm/users/sna/modelopt-qwen3-drafter-training/assets/q30t-ptv3-source-stage-75209087-v1` internally.

- [ ] **Step 9: Compare the two observations before authoring mapping policy**

```bash
cmp --silent \
  /lustre/fsw/coreai_dlalgo_llm/users/sna/modelopt-qwen3-drafter-training/receipts/q30t-ptv23-complement-700k-v1/row-schema/observation-a.json \
  /lustre/fsw/coreai_dlalgo_llm/users/sna/modelopt-qwen3-drafter-training/receipts/q30t-ptv23-complement-700k-v1/row-schema/observation-b.json
sha256sum \
  /lustre/fsw/coreai_dlalgo_llm/users/sna/modelopt-qwen3-drafter-training/receipts/q30t-ptv23-complement-700k-v1/row-schema/observation-a.json \
  /lustre/fsw/coreai_dlalgo_llm/users/sna/modelopt-qwen3-drafter-training/receipts/q30t-ptv23-complement-700k-v1/row-schema/observation-b.json
```

Expected: `cmp` exits 0, both hashes match, file counts are exactly 201 and 100, PTV3 rows total 670,399, and all files within each source agree on one observed descriptor. If not, stop without creating the mapping policy.

### Task 4B: Produce every-file source inventory and row-schema evidence

**Files:**

- Create: `examples/dataset/qwen3_30ba3b_thinking_ptv23_swe_heavy_source_mapping_v1.json`
- Create: `examples/dataset/build_q30t_ptv23_source_inventory.py`
- Create: `tests/examples/dataset/test_build_q30t_ptv23_source_inventory.py`
- Test: `tests/examples/dataset/test_build_qwen4b_ptv23_complement.py`

**Interfaces:**

- Consumes: exact PTV2 `SOURCE_PLAN.json` and `SOURCE_MANIFEST_COMPLETION.json`, exact PTV3 `MANIFEST.json` and completion receipt, the matched Task 4A observation path/SHA, mapping-policy SHA, target-policy SHA, quota-config SHA, and the staged physical roots.
- Produces: `SourceMapping`, `ObservedRowSchema`, `load_source_mapping(path: Path, expected_sha256: str) -> tuple[SourceMapping, ...]`, `build_source_inventory(inputs: InventoryInputs) -> bytes`, and `verify_source_inventory(inputs: InventoryInputs, raw: bytes) -> dict[str, object]`.
- CLI: `build_q30t_ptv23_source_inventory.py build|verify` with exact `--ptv2-plan`, `--ptv2-plan-sha256`, `--ptv2-completion`, `--ptv2-completion-sha256`, `--ptv2-root`, corresponding PTV3 flags, `--schema-observation`, `--schema-observation-sha256`, `--mapping-policy`, `--mapping-policy-sha256`, `--target-policy`, `--target-policy-sha256`, `--quota-config`, `--quota-config-sha256`, and `--output`.

- [ ] **Step 1: Add failing plan/completion and category-order tests**

```python
def test_inventory_requires_exact_stage_plan_completion_pair(tmp_path: Path) -> None:
    inputs = inventory_inputs(tmp_path)
    forged = replace(inputs, ptv3_completion_sha256="0" * 64)
    with pytest.raises(InventoryError, match="PTV3 completion"):
        build_source_inventory(forged)


def test_inventory_sources_follow_quota_order_not_manifest_order(tmp_path: Path) -> None:
    raw = build_source_inventory(inventory_inputs(tmp_path, reverse_sources=True))
    categories = [source["category"] for source in json.loads(raw)["sources"]]
    assert categories == [
        "ptv2_stem",
        "ptv2_multilingual_ja",
        "ptv2_multilingual_es",
        "ptv2_multilingual_fr",
        "ptv2_multilingual_it",
        "ptv3_swe_v3",
        "ptv3_interactive_agentic_swe",
        "ptv3_general_tool_trajectories",
    ]
```

- [ ] **Step 2: Add failing every-file schema and value-shape tests**

```python
def test_inventory_checks_every_row_in_every_file(tmp_path: Path) -> None:
    inputs = inventory_inputs(tmp_path, jsonl_rows=[valid_row(), valid_row(), {"messages": "bad"}])
    with pytest.raises(InventoryError, match=r"messages.*row 3"):
        build_source_inventory(inputs)


def test_inventory_rejects_heterogeneous_schema_within_one_source(tmp_path: Path) -> None:
    inputs = inventory_inputs(tmp_path, second_file_fields=("conversations", "tools"))
    with pytest.raises(InventoryError, match="source-level row schema"):
        build_source_inventory(inputs)
```

- [ ] **Step 3: Run inventory tests and verify RED**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 ../q4-ptv2-ab-integration/.venv/bin/python -m pytest -q \
  tests/examples/dataset/test_build_q30t_ptv23_source_inventory.py
```

Expected: collection FAIL because the producer and mapping policy do not exist.

- [ ] **Step 4: Define the exact checked-in mapping schema**

The canonical mapping file has root keys `schema_version`, `scientific_identity`, `observation_file_sha256`, and `sources`. Each source entry has exact keys `category`, `source_id`, `revision`, `split`, `license_expression`, `approved_use`, `replay_lane`, `format`, `messages_field`, and `tools_field`. `tools_field` is either an actual column name or `__none__`; it is copied from the matched independent observation, never inferred from a sample row. The eight categories are present in quota order, while a category may contain multiple exact source tuples. A source revision is exactly lowercase 40-hex or lowercase 64-hex; other forms fail closed. Bind the common whole-file SHA printed in Task 4A Step 9 as `observation_file_sha256`; the producer loads that observation and requires every mapping field name to match it.

Use the reviewed PTV3 identities already staged:

```python
EXPECTED_PTV3_PINS = {
    ("nvidia/Nemotron-SFT-SWE-v3", "3f73de64c1fe928a8f538fe45ccc10c228cc4c6a", "train"),
    ("nvidia/Nemotron-SFT-SWE-v2", "bd151f3f2d89c4804dda0083d912bd9f6a0a9fb7", "openhands_swe"),
    ("nvidia/Nemotron-SWE-v1", "0fe17a965b297a9c943a59050a14c42d5f0083ce", "r2e_gym"),
    ("nvidia/Nemotron-Agentic-v1", "650d590978ca35c8f1ecea2faf136e5fac421b62", "interactive_agent"),
    ("nvidia/Nemotron-Agentic-v1", "650d590978ca35c8f1ecea2faf136e5fac421b62", "tool_calling"),
}
```

Populate messages/tools field names only from the independently inspected stage schemas. If any observed field differs across files, stop at RED rather than broadening the mapping.

- [ ] **Step 5: Implement stable stage and file authentication**

```python
@dataclass(frozen=True)
class ObservedRowSchema:
    format: Literal["jsonl", "parquet"]
    messages_field: str
    tools_field: str

    def canonical(self) -> dict[str, str]:
        return {
            "format": self.format,
            "messages_field": self.messages_field,
            "tools_field": self.tools_field,
        }


def validate_messages_tools(row: dict[str, object], schema: ObservedRowSchema, row_number: int) -> None:
    messages = decode_json_value(row[schema.messages_field])
    if not isinstance(messages, list) or any(not isinstance(item, dict) for item in messages):
        raise InventoryError(f"messages value shape is invalid at row {row_number}")
    if schema.tools_field != "__none__":
        tools = decode_json_value(row[schema.tools_field])
        if tools is not None and (
            not isinstance(tools, list) or any(not isinstance(item, dict) for item in tools)
        ):
            raise InventoryError(f"tools value shape is invalid at row {row_number}")
```

For JSONL, stream bounded lines and count every decoded row. For Parquet, require the declared columns in `schema_arrow`, iterate every batch for those columns, validate every value, and compare the final count with metadata. Before scanning each file, open with `O_RDONLY|O_NOFOLLOW|O_NONBLOCK`, require single-link regular bytes, compare size and the stage-bound SHA, scan only the held descriptor, then compare before/after and final name binding. Do not run `rglob` over Lustre; enumerate only paths bound by the two authenticated stage manifests.

- [ ] **Step 6: Emit the existing consumer schema exactly**

```python
source_record = {
    "category": mapping.category,
    "source_id": mapping.source_id,
    "revision": mapping.revision,
    "split": mapping.split,
    "license_expression": mapping.license_expression,
    "approved_use": True,
    "replay_lane": mapping.replay_lane,
    "row_schema": observed.canonical(),
    "row_schema_sha256": sha256(canonical_json(observed.canonical())).hexdigest(),
    "files": [
        {
            "logical_path": file.logical_path,
            "path": str(file.physical_path),
            "bytes": file.bytes,
            "sha256": file.sha256,
            "row_count": file.row_count,
        }
        for file in files
    ],
}
```

Wrap records in `{"schema_version":"ptv2-ptv3-complement-source-inventory-v1","scientific_identity":SCIENTIFIC_IDENTITY,"sources":[...]}` and publish canonical newline-terminated bytes with `atomic_publish_bytes`. `verify` must independently rebuild bytes from live files and require byte equality.

- [ ] **Step 7: Add stable-file and completeness attack tests**

```python
@pytest.mark.parametrize("attack", ["fifo", "hardlink", "truncate", "rebind", "wrong_sha"])
def test_inventory_rejects_unstable_physical_files(tmp_path: Path, attack: str) -> None:
    with pytest.raises(InventoryError):
        build_source_inventory(inventory_inputs(tmp_path, attack=attack))


def test_inventory_covers_all_staged_files_and_rows(tmp_path: Path) -> None:
    raw = build_source_inventory(inventory_inputs(tmp_path))
    payload = json.loads(raw)
    assert sum(len(source["files"]) for source in payload["sources"]) == 4
    assert sum(file["row_count"] for source in payload["sources"] for file in source["files"]) == 6
```

Fixture constants are generated by `inventory_inputs`: the unit fixture uses four files and six rows; the Ptyche acceptance run must assert 201 PTV2 files and 100 PTV3 files, and PTV3 total 670,399.

- [ ] **Step 8: Run source-inventory and consumer GREEN tests**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 ../q4-ptv2-ab-integration/.venv/bin/python -m pytest -q \
  tests/examples/dataset/test_build_q30t_ptv23_source_inventory.py \
  tests/examples/dataset/test_build_qwen4b_ptv23_complement.py -k 'source_inventory or row_schema'
```

Expected: PASS; every physical fixture row is examined and heterogeneous files fail.

- [ ] **Step 9: Request hostile review and commit**

Review prompt: “Try manifest/completion cross-pairing, omitted files, duplicate logical paths, first-row-only validation, mixed file schemas, physical rebind/truncate, unbounded Lustre traversal, and category reordering.”

```bash
git add examples/dataset/qwen3_30ba3b_thinking_ptv23_swe_heavy_source_mapping_v1.json \
  examples/dataset/build_q30t_ptv23_source_inventory.py \
  tests/examples/dataset/test_build_q30t_ptv23_source_inventory.py \
  tests/examples/dataset/test_build_qwen4b_ptv23_complement.py
git commit -S -s -m "feat(dataset): produce Q30 source inventory evidence"
```

### Task 5: Integrate the controller-owned Q30 tokenizer loader

**Files:**

- Modify: `examples/dataset/build_qwen4b_ptv23_complement.py:756-846`
- Test: `tests/examples/dataset/test_build_qwen4b_ptv23_complement.py`

**Interfaces:**

- Consumes from controller Task 1: `load_q30t_tokenizer_receipt(path: Path, *, expected_sha256: str) -> dict[str, object]`, whose implementation owns stable reading, approval membership, whole-file SHA verification, and full receipt replay.
- Produces: builder-only adapter `_tokenizer_trust_from_q30_payload(payload: dict[str, object], file_sha256: str) -> TokenizerTrust` and Q30 delegation inside `load_tokenizer_trust(...)`.
- Invariant: the Q30 branch calls the controller-owned loader exactly once with the caller path and keyword-only SHA; the Q4 branch and its approval behavior remain byte-for-byte compatible.

- [ ] **Step 1: Confirm the controller Task 1 handoff before editing**

Run:

```bash
PYTHONPATH=tools/launcher PYTHONDONTWRITEBYTECODE=1 \
  ../q4-ptv2-ab-integration/.venv/bin/python - <<'PY'
import inspect
from common.specdec.q30t_tokenizer_receipt import load_q30t_tokenizer_receipt

parameters = inspect.signature(load_q30t_tokenizer_receipt).parameters
assert tuple(parameters) == ("path", "expected_sha256")
assert parameters["expected_sha256"].kind is inspect.Parameter.KEYWORD_ONLY
PY
```

Expected: exits 0. If import or signature assertion fails, controller Task 1 has not landed and data Task 5 remains blocked; data Tasks 1-4B may continue.

- [ ] **Step 2: Add failing builder-delegation tests**

```python
def test_q30_builder_delegates_to_controller_owned_tokenizer_loader(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    receipt_path = tmp_path / "tokenizer.json"
    receipt_path.write_bytes(b"controller-owned\n")
    expected_sha256 = "5ba642c455e60b67eca295dce92dd7da47292fdba66c5f9d269669c14cafc509"
    calls: list[tuple[Path, str]] = []

    def approved_loader(path: Path, *, expected_sha256: str) -> dict[str, object]:
        calls.append((path, expected_sha256))
        return q30_tokenizer_payload(tmp_path)

    monkeypatch.setattr(builder, "load_q30t_tokenizer_receipt", approved_loader)
    trust = builder.load_tokenizer_trust(
        receipt_path,
        expected_sha256=expected_sha256,
        policy=approved_q30_policy(),
    )
    assert calls == [(receipt_path, expected_sha256)]
    assert trust.file_sha256 == expected_sha256


def test_q30_builder_translates_loader_rejection_without_fallback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def rejected_loader(path: Path, *, expected_sha256: str) -> dict[str, object]:
        raise ValueError("Q30 tokenizer receipt is not approved")

    monkeypatch.setattr(builder, "load_q30t_tokenizer_receipt", rejected_loader)
    with pytest.raises(builder.ComplementError, match="not approved"):
        builder.load_tokenizer_trust(
            tmp_path / "receipt.json",
            expected_sha256="5ba642c455e60b67eca295dce92dd7da47292fdba66c5f9d269669c14cafc509",
            policy=approved_q30_policy(),
        )
```

- [ ] **Step 3: Run the builder tests and verify RED**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 ../q4-ptv2-ab-integration/.venv/bin/python -m pytest -q \
  tests/examples/dataset/test_build_qwen4b_ptv23_complement.py \
  -k 'controller_owned_tokenizer_loader or loader_rejection_without_fallback'
```

Expected: FAIL because the builder still duplicates Q30 parsing instead of invoking `load_q30t_tokenizer_receipt`.

- [ ] **Step 4: Import the controller-owned interface and add the builder adapter**

```python
# Add this name beside the existing verifier in both existing import branches:
from common.specdec.q30t_tokenizer_receipt import (
    load_q30t_tokenizer_receipt,
    verify_q30t_tokenizer_receipt,
)
from tools.launcher.common.specdec.q30t_tokenizer_receipt import (
    load_q30t_tokenizer_receipt,
    verify_q30t_tokenizer_receipt,
)

# Keep the existing fully unavailable fallback import-safe:
load_q30t_tokenizer_receipt = None


def _tokenizer_trust_from_q30_payload(
    payload: dict[str, object], file_sha256: str
) -> TokenizerTrust:
    def text(name: str) -> str:
        value = payload[name]
        if not isinstance(value, str) or not value:
            raise ComplementError(f"Q30 tokenizer {name} is invalid")
        return value

    def digest(name: str, length: int) -> str:
        value = text(name)
        if not _is_lower_hex(value, length):
            raise ComplementError(f"Q30 tokenizer {name} is invalid")
        return value

    def integer(name: str) -> int:
        value = payload[name]
        if type(value) is not int or value < 0:
            raise ComplementError(f"Q30 tokenizer {name} is invalid")
        return value

    return TokenizerTrust(
        file_sha256=file_sha256,
        receipt_sha256=digest("receipt_sha256", 64),
        repository=text("repository"),
        revision=digest("revision", 40),
        snapshot_path=Path(text("snapshot_path")),
        snapshot_tree_sha256=digest("snapshot_tree_sha256", 64),
        chat_template_sha256=digest("chat_template_sha256", 64),
        training_chat_template_sha256=digest("training_chat_template_sha256", 64),
        im_start_token_id=integer("im_start_token_id"),
        im_end_token_id=integer("im_end_token_id"),
    )
```

The adapter performs type narrowing only. It does not reopen the receipt, define an approval set, or call the low-level verifier.

- [ ] **Step 5: Delegate only the builder Q30 branch**

```python
if policy.tokenizer_trust_schema == "qwen3-30ba3b-thinking-tokenizer-trust-v1":
    if load_q30t_tokenizer_receipt is None:
        raise ComplementError("Q30 tokenizer loader is unavailable")
    try:
        payload = load_q30t_tokenizer_receipt(path, expected_sha256=expected_sha256)
    except ValueError as error:
        raise ComplementError(f"Q30 tokenizer trust is invalid: {error}") from error
    return _tokenizer_trust_from_q30_payload(payload, expected_sha256)
```

Delete duplicated Q30 receipt parsing and allowlisting from the builder. Keep Q4 receipt parsing and tokenizer snapshot staging/tree checks unchanged.

- [ ] **Step 6: Run builder GREEN and Q4 regression tests**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 ../q4-ptv2-ab-integration/.venv/bin/python -m pytest -q \
  tests/examples/dataset/test_build_qwen4b_ptv23_complement.py -k tokenizer \
  tests/examples/dataset/test_ptv23_complement_target_policy.py
```

Expected: PASS; Q30 delegates exactly once, loader rejection has no fallback, and Q4 behavior remains unchanged.

- [ ] **Step 7: Request hostile review and commit only builder-owned files**

Review prompt: “Verify the builder imports the controller-owned loader, passes `expected_sha256` keyword-only, defines no second approval set or receipt verifier, has no fallback after rejection, and preserves Q4 behavior.”

```bash
git add examples/dataset/build_qwen4b_ptv23_complement.py \
  tests/examples/dataset/test_build_qwen4b_ptv23_complement.py
git commit -S -s -m "fix(dataset): consume Q30 tokenizer approval"
```

### Task 6: Freeze the distinct SWE-heavy source, quota, and target policies

**Files:**

- Create: `examples/dataset/qwen3_30ba3b_thinking_ptv23_swe_heavy_sources_v1.json`
- Modify: `examples/dataset/qwen3_30ba3b_thinking_ptv23_swe_heavy_700k_v1.json`
- Modify: `examples/dataset/qwen3_30ba3b_thinking_ptv23_swe_heavy_target_v1.json`
- Modify: `examples/dataset/ptv23_complement_target_policy.py:27-85,113-134`
- Modify: `examples/dataset/build_qwen4b_ptv23_complement.py:1145-1181`
- Modify: `tests/examples/dataset/test_q30t_swe_heavy_700k_config.py`
- Modify: `tests/examples/dataset/test_ptv23_complement_target_policy.py`
- Modify: `tests/examples/dataset/test_build_qwen4b_ptv23_complement.py`

**Interfaces:**

- Consumes: exact revision `144afc2f379b542fdd4e85a1fcd5e1f79112d95d` and the checked-in source mapping from Task 4.
- Produces: one canonical requirements hash, quota-config hash, target-policy hash, and a `TargetTokenizerPolicy` registry entry whose embedded identity is SWE-heavy.

- [ ] **Step 1: Replace tests that authorize balanced-source reinterpretation with failing isolation tests**

```python
def test_swe_heavy_config_has_exact_revision_and_distinct_source_contract() -> None:
    config = load_json(Q30_SWE_HEAVY_CONFIG_PATH)
    requirements = load_json(Q30_SWE_HEAVY_SOURCE_REQUIREMENTS_PATH)
    assert config["tokenizer"]["revision"] == Q30_REVISION
    assert config["source_requirements"]["path"] == Q30_SWE_HEAVY_SOURCE_REQUIREMENTS_PATH.name
    assert requirements["scientific_identity"] == Q30_SWE_HEAVY_IDENTITY
    assert requirements["scientific_identity"] != Q30_BALANCED_IDENTITY


def test_balanced_source_contract_cannot_authorize_swe_heavy_policy() -> None:
    policy = load_target_policy(Q30_SWE_HEAVY_POLICY_PATH, expected_sha256=Q30_SWE_HEAVY_POLICY_SHA256)
    assert not source_requirements_contract_is_compatible(
        policy,
        observed_sha256=Q30_BALANCED_SOURCE_REQUIREMENTS_SHA256,
        embedded_scientific_identity=Q30_BALANCED_IDENTITY,
    )
```

- [ ] **Step 2: Run policy tests and verify RED**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 ../q4-ptv2-ab-integration/.venv/bin/python -m pytest -q \
  tests/examples/dataset/test_q30t_swe_heavy_700k_config.py \
  tests/examples/dataset/test_ptv23_complement_target_policy.py -k swe_heavy
```

Expected: FAIL because revisions are empty and the SWE-heavy policy currently reuses the balanced source identity/hash through `APPROVED_SHARED_SOURCE_CONTRACT_IDENTITIES`.

- [ ] **Step 3: Write canonical distinct source requirements**

The source requirements copy the reviewed exact `known_pinned_sources` tuples and blocker meanings from the balanced file, change `scientific_identity` to the SWE-heavy identity, and add a `source_mapping` binding whose digest is computed from the actual canonical mapping bytes. Generate the exact prospective canonical line with this read-only command, inspect it, and apply that stdout verbatim with `apply_patch`:

```bash
PYTHONDONTWRITEBYTECODE=1 ../q4-ptv2-ab-integration/.venv/bin/python - <<'PY'
import hashlib
import json
from pathlib import Path

dataset = Path("examples/dataset")
balanced_path = dataset / "qwen3_30ba3b_thinking_ptv23_complement_sources_v1.json"
mapping_path = dataset / "qwen3_30ba3b_thinking_ptv23_swe_heavy_source_mapping_v1.json"
payload = json.loads(balanced_path.read_bytes())
payload["scientific_identity"] = (
    "ptv2-ptv3-complement-700k-qwen3-30ba3b-thinking-swe-heavy-v1"
)
payload["source_mapping"] = {
    "path": mapping_path.name,
    "sha256": hashlib.sha256(mapping_path.read_bytes()).hexdigest(),
}
print(json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")))
PY
```

Expected: one canonical JSON line whose `source_mapping.sha256` is lowercase 64-hex. After applying it, `sha256sum` of the mapping must equal that embedded value.

- [ ] **Step 4: Authenticate the new mapping binding in source reconciliation**

Change the SWE-heavy requirements key set to include `source_mapping`, require its exact keys `path` and `sha256`, require `Path(path).name == path`, and call `load_source_mapping(path.with_name(path), sha256)`. Reconcile each inventory source against the authenticated mapping tuple before checking capacity:

```python
mapping_path, mapping_sha256 = _bound_file(requirements, "source_mapping")
mappings = load_source_mapping(path.with_name(mapping_path), mapping_sha256)
approved_mapping_tuples = {
    (
        item.category,
        item.source_id,
        item.revision,
        item.split,
        item.license_expression,
        item.replay_lane,
        item.format,
        item.messages_field,
        item.tools_field,
    )
    for item in mappings
}
observed_mapping_tuples = {
    (
        source.category,
        source.source_id,
        source.revision,
        source.split,
        source.license_expression,
        source.replay_lane,
        source.row_schema["format"],
        source.row_schema["messages_field"],
        source.row_schema["tools_field"],
    )
    for source in inventory.sources
}
if observed_mapping_tuples != approved_mapping_tuples:
    raise ComplementError("source inventory does not match the approved source mapping")
```

Preserve the four-key balanced requirements schema for the balanced policy; accept the five-key form only when the target policy scientific identity is the exact SWE-heavy identity.

- [ ] **Step 5: Canonicalize files bottom-up and freeze each actual digest**

Use the following deterministic command after each JSON edit:

```bash
PYTHONDONTWRITEBYTECODE=1 ../q4-ptv2-ab-integration/.venv/bin/python - <<'PY'
import hashlib
import json
from pathlib import Path

paths = [
    Path("examples/dataset/qwen3_30ba3b_thinking_ptv23_swe_heavy_sources_v1.json"),
    Path("examples/dataset/qwen3_30ba3b_thinking_ptv23_swe_heavy_700k_v1.json"),
    Path("examples/dataset/qwen3_30ba3b_thinking_ptv23_swe_heavy_target_v1.json"),
]
for path in paths:
    payload = json.loads(path.read_bytes())
    raw = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode() + b"\n"
    if raw != path.read_bytes():
        raise SystemExit(f"noncanonical JSON: {path}")
    print(hashlib.sha256(raw).hexdigest(), path)
PY
```

Update child bindings bottom-up: requirements SHA in quota config; requirements and quota SHA in target policy; then target-policy SHA in `APPROVED_TARGET_POLICY_SHA256S` and `APPROVED_TARGET_POLICIES`. Remove the SWE-heavy entry from `APPROVED_SHARED_SOURCE_CONTRACT_IDENTITIES` because exact same-identity requirements now exist.

- [ ] **Step 6: Assert exact policy values**

```python
assert policy.target_revision == "144afc2f379b542fdd4e85a1fcd5e1f79112d95d"
assert policy.tokenizer_revision == "144afc2f379b542fdd4e85a1fcd5e1f79112d95d"
assert policy.scientific_identity == Q30_SWE_HEAVY_IDENTITY
assert policy.training_sequence_length == 4096
assert sum(config["quotas"].values()) == 700_000
assert tuple(config["quotas"]) == APPROVED_QUOTA_CATEGORIES
```

- [ ] **Step 7: Run policy GREEN tests and preserve balanced bytes**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 ../q4-ptv2-ab-integration/.venv/bin/python -m pytest -q \
  tests/examples/dataset/test_q30t_swe_heavy_700k_config.py \
  tests/examples/dataset/test_ptv23_complement_target_policy.py \
  tests/examples/dataset/test_build_qwen4b_ptv23_complement.py -k 'source_requirements or mapping'
```

Expected: PASS; the balanced config still hashes to `b7436725b75f09f3557ff589f99da67b68472abfd7e328a41d6ce5ff639bb88d`, while SWE-heavy uses its own requirements and exact revision.

- [ ] **Step 8: Request hostile review and commit**

Review prompt: “Recompute all three JSON hashes bottom-up, reject the balanced embedded identity, verify quota order/total, and confirm both target and tokenizer revisions are exact.”

```bash
git add examples/dataset/qwen3_30ba3b_thinking_ptv23_swe_heavy_sources_v1.json \
  examples/dataset/qwen3_30ba3b_thinking_ptv23_swe_heavy_700k_v1.json \
  examples/dataset/qwen3_30ba3b_thinking_ptv23_swe_heavy_target_v1.json \
  examples/dataset/ptv23_complement_target_policy.py \
  examples/dataset/build_qwen4b_ptv23_complement.py \
  tests/examples/dataset/test_q30t_swe_heavy_700k_config.py \
  tests/examples/dataset/test_ptv23_complement_target_policy.py \
  tests/examples/dataset/test_build_qwen4b_ptv23_complement.py
git commit -S -s -m "feat(dataset): freeze Q30 SWE-heavy policy"
```

### Task 7: Build an independent data-evidence replay report

**Files:**

- Create: `examples/dataset/review_q30t_700k_data_evidence.py`
- Create: `tests/examples/dataset/test_review_q30t_700k_data_evidence.py`

**Interfaces:**

- Consumes: historical audit path/SHA, tokenizer path/SHA, target/quota/requirements/mapping paths and SHAs, five named consumer/provenance receipt pairs, source inventory path/SHA, and all source-stage arguments used by Task 4.
- Produces: `review_evidence(arguments: ReviewArguments) -> bytes` with schema `q30t-700k-data-evidence-review-v1` and CLI `review_q30t_700k_data_evidence.py --contract PATH --contract-sha256 SHA256 --output ABSOLUTE_PATH`.
- The contract is canonical JSON and contains every explicit path/SHA argument; it is not itself an approval root.

- [ ] **Step 1: Add failing complete-review and mutation tests**

```python
def test_review_report_binds_all_five_names_and_source_roots(tmp_path: Path) -> None:
    raw = review_evidence(review_arguments(tmp_path))
    payload = json.loads(raw)
    assert tuple(payload["held_out_receipts"]) == ("speed", "math", "code", "swe", "tool")
    assert payload["historical_audit_file_sha256"] == HISTORICAL_AUDIT_SHA256
    assert payload["tokenizer_receipt_file_sha256"] == TOKENIZER_RECEIPT_SHA256
    assert payload["ptv2_file_count"] == 201
    assert payload["ptv3_file_count"] == 100
    assert payload["ptv3_row_count"] == 670_399


@pytest.mark.parametrize(
    "mutation",
    ["historical", "tokenizer", "heldout_provenance", "inventory", "schema", "ptv3_tuple"],
)
def test_review_replays_and_rejects_each_mutated_input(tmp_path: Path, mutation: str) -> None:
    with pytest.raises(ReviewError):
        review_evidence(review_arguments(tmp_path, mutation=mutation))
```

- [ ] **Step 2: Run review tests and verify RED**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 ../q4-ptv2-ab-integration/.venv/bin/python -m pytest -q \
  tests/examples/dataset/test_review_q30t_700k_data_evidence.py
```

Expected: collection FAIL because the review module is absent.

- [ ] **Step 3: Implement independent replay composition**

```python
def review_evidence(arguments: ReviewArguments) -> bytes:
    historical, _ = load_baseline_audit_receipt(
        arguments.historical_path, arguments.historical_sha256
    )
    tokenizer = load_q30t_tokenizer_receipt(
        arguments.tokenizer_path, expected_sha256=arguments.tokenizer_sha256
    )
    held_out = tuple(replay_heldout_pair(item) for item in arguments.held_out)
    rebuilt_inventory = build_source_inventory(arguments.inventory_inputs)
    published_inventory = stable_single_link_regular_bytes(arguments.inventory_path)
    if rebuilt_inventory != published_inventory:
        raise ReviewError("source inventory replay differs")
    body = _review_body(arguments, historical, tokenizer, held_out, rebuilt_inventory)
    return canonical_json(body | {"receipt_sha256": sha256(canonical_json(body)).hexdigest()}) + b"\n"
```

The body records producer source commit, target tree, all whole-file SHAs, ordered PTV3 tuples, sorted observed row-schema hashes, exact file/row counts, and each held-out provenance SHA. It must not modify any production approval constant.

- [ ] **Step 4: Add canonical/self-hash and path-rebind tests**

```python
def test_review_report_is_canonical_and_self_hashed(tmp_path: Path) -> None:
    raw = review_evidence(review_arguments(tmp_path))
    payload = json.loads(raw)
    receipt_sha = payload.pop("receipt_sha256")
    assert raw == canonical_json(payload | {"receipt_sha256": receipt_sha}) + b"\n"
    assert receipt_sha == hashlib.sha256(canonical_json(payload)).hexdigest()


def test_review_rejects_input_rebind_between_replays(tmp_path: Path) -> None:
    with pytest.raises(ReviewError, match="changed"):
        review_evidence(review_arguments(tmp_path, mutation="rebind_during_replay"))
```

- [ ] **Step 5: Run review GREEN tests**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 ../q4-ptv2-ab-integration/.venv/bin/python -m pytest -q \
  tests/examples/dataset/test_review_q30t_700k_data_evidence.py
```

Expected: PASS; the report is a replay result, not self-authorization.

- [ ] **Step 6: Request hostile review and commit**

Review prompt: “Try omitting one held-out pair, swapping provenance, replaying different stage files, altering PTV3 tuples/schema hashes, and treating the review report as an approval without a human comparison.”

```bash
git add examples/dataset/review_q30t_700k_data_evidence.py \
  tests/examples/dataset/test_review_q30t_700k_data_evidence.py
git commit -S -s -m "feat(dataset): replay Q30 data evidence"
```

### Task 8: Integrate approved roots and exact builder/replay semantics

**Files:**

- Modify: `examples/dataset/build_qwen4b_ptv23_complement.py:100-168,466-595,670-732,775-846,1057-1142,1389-1469,2330-2400,2729-2782`
- Modify: `tests/examples/dataset/test_build_qwen4b_ptv23_complement.py`

**Interfaces:**

- Consumes: independently matched producer/reviewer evidence from Task 7.
- Produces: exact approval constants; shared candidate identity; full-message/tools binding in DATA, MANIFEST, capacity, and completion evidence; `verify_selection_bundle(...)` full replay.

- [ ] **Step 1: Add failing unresolved-root and approved-tuple tests**

```python
def test_external_roots_require_tokenizer_historical_inventory_schema_and_five_heldouts() -> None:
    with pytest.raises(ComplementError, match="authoritative external approval root"):
        _require_external_approval_roots()


def test_approved_roots_match_independently_reviewed_report(review_report: dict[str, object]) -> None:
    assert APPROVED_HISTORICAL_RECEIPT_FILE_SHA256 == review_report["historical_audit_file_sha256"]
    assert APPROVED_SOURCE_INVENTORY_FILE_SHA256 == review_report["source_inventory_file_sha256"]
    assert APPROVED_HELD_OUT_RECEIPT_FILE_SHA256S == review_report["held_out_receipts"]
    assert APPROVED_ROW_SCHEMA_SHA256S == frozenset(review_report["row_schema_sha256s"])
    assert APPROVED_PTV3_SWE_SOURCE_PINS == tuple(map(tuple, review_report["ptv3_source_pins"]))
```

- [ ] **Step 2: Run approval tests and verify RED**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 ../q4-ptv2-ab-integration/.venv/bin/python -m pytest -q \
  tests/examples/dataset/test_build_qwen4b_ptv23_complement.py -k 'external_roots or approved_roots'
```

Expected: FAIL because inventory, schema, PTV3, historical, and held-out approval roots remain empty.

- [ ] **Step 3: Compare producer and reviewer bytes before inserting roots**

On Ptyche, two different reviewers run the Task 7 `review` operation into `review-a.json` and `review-b.json`. Then run exactly:

```bash
cmp --silent \
  /lustre/fsw/coreai_dlalgo_llm/users/sna/modelopt-qwen3-drafter-training/receipts/q30t-ptv23-complement-700k-v1/data-evidence/review-a.json \
  /lustre/fsw/coreai_dlalgo_llm/users/sna/modelopt-qwen3-drafter-training/receipts/q30t-ptv23-complement-700k-v1/data-evidence/review-b.json
sha256sum \
  /lustre/fsw/coreai_dlalgo_llm/users/sna/modelopt-qwen3-drafter-training/receipts/q30t-ptv23-complement-700k-v1/data-evidence/review-a.json \
  /lustre/fsw/coreai_dlalgo_llm/users/sna/modelopt-qwen3-drafter-training/receipts/q30t-ptv23-complement-700k-v1/data-evidence/review-b.json
```

Expected: `cmp` exits 0 and `sha256sum` prints the same digest twice. If either differs, do not edit approval roots.

- [ ] **Step 4: Extract exact reviewed values without transcription ambiguity**

```bash
PYTHONDONTWRITEBYTECODE=1 /usr/bin/python3.12 -I -S - \
  /lustre/fsw/coreai_dlalgo_llm/users/sna/modelopt-qwen3-drafter-training/receipts/q30t-ptv23-complement-700k-v1/data-evidence/review-a.json <<'PY'
import json
import sys
from pathlib import Path

payload = json.loads(Path(sys.argv[1]).read_bytes())
for key in (
    "historical_audit_file_sha256",
    "source_inventory_file_sha256",
    "held_out_receipts",
    "row_schema_sha256s",
    "ptv3_source_pins",
):
    print(key, json.dumps(payload[key], sort_keys=True, separators=(",", ":")))
PY
```

Expected: historical equals `2469430c144d9b86850901df0a28cb810b437555ee894771b796b1386d1c18b5`; held-out has exactly five fixed-order keys; every printed hash is lowercase 64-hex; PTV3 pins equal the reviewed tuple set from Task 4. Insert these exact printed values into the builder constants. The plan intentionally contains no invented held-out, inventory, or row-schema hash.

- [ ] **Step 5: Add full row bytes alongside shared prompt identity**

```python
prompt_bytes = canonicalize_prompt(messages, tools)
prompt_id = sha256(prompt_bytes).hexdigest()
full_row_bytes = _canonical_json({"messages": messages, "tools": tools})
record = CandidateRecord(
    prompt_uuid=prompt_id,
    canonical_prompt=prompt_bytes,
    messages=messages,
    tools=tools,
    full_messages_tools_sha256=sha256(full_row_bytes).hexdigest(),
    full_tokens=full_tokens,
    assistant_tokens=assistant_tokens,
)
```

Persist and replay `full_messages_tools_sha256`; prompt exclusion/admission uses only `prompt_uuid`. Add a test proving two assistant completions share prompt identity but differ in full-row digest, and the second candidate is rejected as a duplicate.

- [ ] **Step 6: Add exact 700K and replay tests**

```python
def test_swe_heavy_build_has_exact_quota_order_count_and_unique_exposure(bundle: Path) -> None:
    manifest = json.loads((bundle / "MANIFEST.json").read_bytes())
    assert manifest["row_count"] == 700_000
    assert manifest["category_counts"] == Q30_SWE_HEAVY_QUOTAS
    rows = list(iter_data_rows(bundle / "DATA.jsonl"))
    assert len(rows) == len({row["prompt_uuid"] for row in rows}) == 700_000


def test_replay_from_empty_scratch_reproduces_all_bundle_hashes(bundle: Path, tmp_path: Path) -> None:
    replay = verify_selection_bundle(bundle, scratch_root=tmp_path / "fresh")
    assert replay.data_file_sha256 == sha256_file(bundle / "DATA.jsonl")
    assert replay.manifest_file_sha256 == sha256_file(bundle / "MANIFEST.json")
    assert replay.completion_file_sha256 == sha256_file(bundle.parent / "COMPLETION.json")
```

- [ ] **Step 7: Run builder GREEN tests**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 ../q4-ptv2-ab-integration/.venv/bin/python -m pytest -q \
  tests/examples/dataset/test_build_qwen4b_ptv23_complement.py \
  tests/examples/dataset/test_specdec_identity.py \
  tests/examples/dataset/test_q30t_swe_heavy_700k_config.py \
  tests/examples/dataset/test_ptv23_complement_target_policy.py
```

Expected: PASS; unresolved or substituted roots fail before candidate iteration, while the approved exact tuple reaches build/replay.

- [ ] **Step 8: Request hostile review and commit the approval transition**

Review prompt: “Independently recompute every inserted hash from review-a/review-b, verify fixed held-out order, direct historical lineage, all observed schema hashes, exact PTV3 pins, assistant-insensitive prompt UUID, full-row binding, and 700K uniqueness.”

```bash
git add examples/dataset/build_qwen4b_ptv23_complement.py \
  tests/examples/dataset/test_build_qwen4b_ptv23_complement.py
git commit -S -s -m "feat(dataset): approve Q30 data evidence"
```

### Task 9: Add immutable Ptyche evidence, build, and replay jobs

**Files:**

- Create: `tools/launcher/common/specdec/run_q30t_700k_data_evidence.sbatch`
- Create: `tools/launcher/common/specdec/submit_q30t_700k_data_evidence.sh`
- Create: `tools/launcher/tests/test_q30t_700k_data_evidence_runner.py`

**Interfaces:**

- Consumes: clean source checkout, reviewed source commit, canonical operation contract path/SHA, and exact output path.
- Produces: one SLURM job for exactly one of `heldout`, `inventory`, `review`, `build`, or `replay`; each job publishes one operation completion receipt.
- Submit CLI: `submit_q30t_700k_data_evidence.sh (--test-only|--submit) --operation OP --source-path PATH --contract PATH --contract-sha256 SHA256 --output PATH --slurm-output PATH`.

- [ ] **Step 1: Add failing submitter/runner boundary tests**

```python
def test_submitter_requires_test_only_or_submit_and_clean_pushed_head() -> None:
    text = SUBMITTER.read_text()
    assert "--test-only|--submit" in text
    assert "git status --porcelain" in text
    assert "git rev-parse @{u}" in text
    assert "sbatch --test-only" in text


def test_runner_is_cpu_only_single_operation_and_immutable_git() -> None:
    text = RUNNER.read_text()
    assert "#SBATCH --nodes=1" in text
    assert "#SBATCH --ntasks=1" in text
    assert "#SBATCH --cpus-per-task=32" in text
    assert "--gpus" not in text
    assert "git archive" in text or "git cat-file" in text
    assert "PYTHONSAFEPATH=1" in text
    assert "heldout|inventory|review|build|replay" in text
```

- [ ] **Step 2: Run runner tests and verify RED**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 ../q4-ptv2-ab-integration/.venv/bin/python -m pytest -q \
  tools/launcher/tests/test_q30t_700k_data_evidence_runner.py
```

Expected: FAIL because runner and submitter are absent.

- [ ] **Step 3: Implement the clean-source submitter gate**

```bash
source_sha="$($approved_git -C "$source_path" rev-parse HEAD)"
upstream_sha="$($approved_git -C "$source_path" rev-parse '@{u}')"
[[ "$source_sha" == "$upstream_sha" ]] || fail "reviewed source commit is not pushed"
[[ -z "$($approved_git -C "$source_path" status --porcelain --untracked-files=all)" ]] \
    || fail "source checkout is not clean"
[[ "$contract_sha256" =~ ^[0-9a-f]{64}$ ]] || fail "contract SHA-256 is invalid"
[[ "$(sha256sum "$contract" | cut -d' ' -f1)" == "$contract_sha256" ]] \
    || fail "contract SHA-256 mismatch"
```

In `--test-only`, invoke `sbatch --test-only` and exit without publishing a submission receipt. In `--submit`, run the same test-only command first, then submit the exact spooled runner and atomically publish a canonical submission receipt binding source commit, runner SHA, contract SHA, operation, output, log path, and job ID.

- [ ] **Step 4: Implement the CPU-only immutable runner**

The runner validates exactly one operation, authenticates `/usr/bin/python3.12` and requires `sys.version_info >= (3, 12)`, extracts the reviewed source commit into a fresh `/raid/scratch/$USER/q30t-data-$SLURM_JOB_ID-*` directory using the same sterile Git boundary as `run_q30t_ptv2_full201_transfer_verify.sbatch`, and invokes the operation module with `/usr/bin/env -i LC_ALL=C PYTHONDONTWRITEBYTECODE=1 PYTHONSAFEPATH=1 /usr/bin/python3.12 -I -S`. It accepts only contract-bound absolute paths and exact SHAs. Its cleanup policy intentionally leaves the private scratch directory for scheduler scratch lifecycle; it performs no recursive delete and no pathname unlink recovery.

Use exact persistent roots:

```bash
receipt_root=/lustre/fsw/coreai_dlalgo_llm/users/sna/modelopt-qwen3-drafter-training/receipts/q30t-ptv23-complement-700k-v1
ptv2_root=/lustre/fsw/coreai_dlalgo_llm/users/sna/modelopt-qwen3-drafter-training/assets/q30t-ptv2-full201-source-v1
ptv3_root=/lustre/fsw/coreai_dlalgo_llm/users/sna/modelopt-qwen3-drafter-training/assets/q30t-ptv3-source-stage-75209087-v1
historical_audit=/lustre/fsw/coreai_dlalgo_llm/users/sna/modelopt-qwen3-drafter-training/receipts/q30t-ptv2-history-audit-c644fd0f/AUDIT.json
```

- [ ] **Step 5: Add operation isolation and publication attack tests**

```python
@pytest.mark.parametrize("operation", ["heldout", "inventory", "review", "build", "replay"])
def test_runner_dispatches_only_contract_bound_operation(operation: str, tmp_path: Path) -> None:
    result = run_fixture(operation, tmp_path)
    assert result.returncode == 0
    assert operation_marker(tmp_path, operation).is_file()
    assert no_other_operation_markers(tmp_path, operation)


def test_runner_preserves_foreign_output_and_replaced_scratch(tmp_path: Path) -> None:
    result = run_fixture("inventory", tmp_path, inject="foreign_output_and_scratch_rebind")
    assert result.returncode != 0
    assert (tmp_path / "foreign-output").read_bytes() == b"foreign\n"
    assert (tmp_path / "scratch-replacement/marker").read_bytes() == b"foreign\n"
```

- [ ] **Step 6: Run runner GREEN and static shell tests**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 ../q4-ptv2-ab-integration/.venv/bin/python -m pytest -q \
  tools/launcher/tests/test_q30t_700k_data_evidence_runner.py
bash -n tools/launcher/common/specdec/run_q30t_700k_data_evidence.sbatch \
  tools/launcher/common/specdec/submit_q30t_700k_data_evidence.sh
shellcheck tools/launcher/common/specdec/run_q30t_700k_data_evidence.sbatch \
  tools/launcher/common/specdec/submit_q30t_700k_data_evidence.sh
```

Expected: all commands exit 0.

- [ ] **Step 7: Request hostile review and commit**

Review prompt: “Try dirty/unpushed source, mutable-checkout substitution, contract rebind, environment injection, operation smuggling, foreign output, parent replacement, scratch replacement, recursive cleanup, and submission without prior test-only.”

```bash
git add tools/launcher/common/specdec/run_q30t_700k_data_evidence.sbatch \
  tools/launcher/common/specdec/submit_q30t_700k_data_evidence.sh \
  tools/launcher/tests/test_q30t_700k_data_evidence_runner.py
git commit -S -s -m "feat(dataset): add Q30 data evidence jobs"
```

### Task 10: Generate, independently replay, approve, build, and replay on Ptyche

**Files:**

- Modify after evidence review: `examples/dataset/build_qwen4b_ptv23_complement.py` approval constants only
- Modify after corpus replay review: the downstream controller approval file belongs to Subproject B and is not changed in this plan

**Interfaces:**

- Consumes: clean pushed implementation commit; authoritative five-owner held-out source manifest; completed PTV2/PTV3 stage contracts; reviewed tokenizer and historical roots.
- Produces: independently matched evidence reports, approved builder roots, `DATA.jsonl`, `MANIFEST.json`, capacity receipt, completion receipt, and a replay receipt whose hashes match the published bundle.

- [ ] **Step 1: Freeze the local implementation before remote work**

Run:

```bash
git status --short
git diff --check
git rev-parse HEAD
git rev-parse '@{u}'
```

Expected: status is empty, `git diff --check` exits 0, and both commit hashes are identical. Do not submit from commit `212d01516b32e5a81f077d7f2f26ec8de38707c6`; that hash remains receipt provenance for already completed tokenizer/parent evidence, while the new data producer must bind its own reviewed pushed commit.

- [ ] **Step 2: Test-only the held-out evidence job**

```bash
tools/launcher/common/specdec/submit_q30t_700k_data_evidence.sh --test-only \
  --operation heldout \
  --source-path "$PWD" \
  --contract /lustre/fsw/coreai_dlalgo_llm/users/sna/modelopt-qwen3-drafter-training/receipts/q30t-ptv23-complement-700k-v1/contracts/heldout-sources.json \
  --contract-sha256 "$(sha256sum /lustre/fsw/coreai_dlalgo_llm/users/sna/modelopt-qwen3-drafter-training/receipts/q30t-ptv23-complement-700k-v1/contracts/heldout-sources.json | cut -d' ' -f1)" \
  --output /lustre/fsw/coreai_dlalgo_llm/users/sna/modelopt-qwen3-drafter-training/receipts/q30t-ptv23-complement-700k-v1/heldout \
  --slurm-output /lustre/fsw/coreai_dlalgo_llm/users/sna/modelopt-qwen3-drafter-training/receipts/q30t-ptv23-complement-700k-v1/logs/heldout-%j.out
```

Expected: `sbatch --test-only` accepts the request and no held-out receipt is created. If the authoritative contract lacks any one exact owner source, stop here.

- [ ] **Step 3: Submit held-out production and monitor five minutes**

Run the same command with `--submit`. Record the returned job ID as `heldout_job_id`, then run one command per minute for five minutes:

```bash
squeue -j "$heldout_job_id" -h -o '%i|%T|%M|%R'
```

Expected: one filtered line while queued/running or no line after completion; logs remain bounded beneath the receipt root. On completion require ten files: five consumer receipts and five provenance receipts.

- [ ] **Step 4: Test-only and submit source inventory production**

```bash
tools/launcher/common/specdec/submit_q30t_700k_data_evidence.sh --test-only \
  --operation inventory \
  --source-path "$PWD" \
  --contract /lustre/fsw/coreai_dlalgo_llm/users/sna/modelopt-qwen3-drafter-training/receipts/q30t-ptv23-complement-700k-v1/contracts/source-inventory.json \
  --contract-sha256 "$(sha256sum /lustre/fsw/coreai_dlalgo_llm/users/sna/modelopt-qwen3-drafter-training/receipts/q30t-ptv23-complement-700k-v1/contracts/source-inventory.json | cut -d' ' -f1)" \
  --output /lustre/fsw/coreai_dlalgo_llm/users/sna/modelopt-qwen3-drafter-training/receipts/q30t-ptv23-complement-700k-v1/source-inventory/INVENTORY.json \
  --slurm-output /lustre/fsw/coreai_dlalgo_llm/users/sna/modelopt-qwen3-drafter-training/receipts/q30t-ptv23-complement-700k-v1/logs/inventory-%j.out
```

Expected: test-only succeeds without output. Repeat with `--submit`, monitor the one returned job once per minute for five minutes, and require exactly 201 PTV2 files plus 100 PTV3 files and 670,399 PTV3 rows in the result.

- [ ] **Step 5: Run two independent evidence replays**

Use `--operation review` twice with distinct immutable outputs `data-evidence/review-a.json` and `data-evidence/review-b.json`, running the Task 8 `cmp` and `sha256sum` commands afterward. The two jobs use fresh node-local scratch and independently re-read all physical source files.

Expected: exact byte equality, identical whole-file SHA, historical root `2469430c144d9b86850901df0a28cb810b437555ee894771b796b1386d1c18b5`, tokenizer root `5ba642c455e60b67eca295dce92dd7da47292fdba66c5f9d269669c14cafc509`, five named held-out hashes, reviewed PTV3 tuples, and the same row-schema hash set.

- [ ] **Step 6: Insert reviewed roots, rerun local gates, request review, commit, and push**

Perform Task 8 Steps 4-8 using the exact review output. Then run the full gate from Task 11 below. A different reviewer recomputes the inserted values. Commit only the approval constants/tests and push that reviewed commit before build submission.

- [ ] **Step 7: Test-only and submit the one exact corpus build**

```bash
tools/launcher/common/specdec/submit_q30t_700k_data_evidence.sh --test-only \
  --operation build \
  --source-path "$PWD" \
  --contract /lustre/fsw/coreai_dlalgo_llm/users/sna/modelopt-qwen3-drafter-training/receipts/q30t-ptv23-complement-700k-v1/contracts/dataset-build.json \
  --contract-sha256 "$(sha256sum /lustre/fsw/coreai_dlalgo_llm/users/sna/modelopt-qwen3-drafter-training/receipts/q30t-ptv23-complement-700k-v1/contracts/dataset-build.json | cut -d' ' -f1)" \
  --output /lustre/fsw/coreai_dlalgo_llm/users/sna/modelopt-qwen3-drafter-training/datasets/q30t-ptv23-swe-heavy-700k-v1 \
  --slurm-output /lustre/fsw/coreai_dlalgo_llm/users/sna/modelopt-qwen3-drafter-training/receipts/q30t-ptv23-complement-700k-v1/logs/build-%j.out
```

Expected: test-only succeeds without mutating the dataset root. Repeat with `--submit`, monitor once per minute for five minutes, then require immutable DATA, MANIFEST, capacity, and completion artifacts. A second build operation against the existing root must adopt only exact bytes and otherwise fail while preserving the root.

- [ ] **Step 8: Run full replay from fresh scratch**

Use `--operation replay` with the same dataset-build contract plus the published completion SHA and output `dataset-replay/REPLAY.json`. The replay operation must select/tokenize all rows again from physical sources in fresh node-local scratch, not merely scan DATA.

Expected: exact equality for DATA SHA, MANIFEST SHA, capacity SHA, completion body/self-hash, ordered prompt UUID digest, category counts, token evidence, source inventory SHA, historical SHA, tokenizer SHA, and held-out map; row count and unique prompt count both equal 700,000.

- [ ] **Step 9: Independently approve the completion SHA**

Two reviewers run:

```bash
sha256sum \
  /lustre/fsw/coreai_dlalgo_llm/users/sna/modelopt-qwen3-drafter-training/datasets/q30t-ptv23-swe-heavy-700k-v1/MANIFEST.json \
  /lustre/fsw/coreai_dlalgo_llm/users/sna/modelopt-qwen3-drafter-training/receipts/q30t-ptv23-complement-700k-v1/dataset/COMPLETION.json \
  /lustre/fsw/coreai_dlalgo_llm/users/sna/modelopt-qwen3-drafter-training/receipts/q30t-ptv23-complement-700k-v1/dataset-replay/REPLAY.json
```

Expected: both reviewers report identical values and the replay receipt binds the observed completion whole-file SHA. Hand that reviewed SHA and path to Subproject B; do not modify controller approval roots in this plan.

### Task 11: Run the integrated local/static gate and freeze the slice

**Files:**

- Verify all Subproject A files listed in the File Structure section
- Do not modify production files during this task

**Interfaces:**

- Consumes: completed Tasks 1-10.
- Produces: test/static evidence and a content-hash freeze for independent review.

- [ ] **Step 1: Run all focused Python tests**

```bash
PYTHONDONTWRITEBYTECODE=1 ../q4-ptv2-ab-integration/.venv/bin/python -m pytest -q \
  tests/examples/dataset/test_specdec_identity.py \
  tests/examples/dataset/test_audit_ptv2_baseline.py \
  tests/examples/dataset/test_trajectory_schema.py \
  tests/examples/dataset/test_observe_q30t_ptv23_row_schemas.py \
  tests/examples/dataset/test_build_q30t_heldout_receipts.py \
  tests/examples/dataset/test_build_q30t_ptv23_source_inventory.py \
  tests/examples/dataset/test_review_q30t_700k_data_evidence.py \
  tests/examples/dataset/test_q30t_swe_heavy_700k_config.py \
  tests/examples/dataset/test_ptv23_complement_target_policy.py \
  tests/examples/dataset/test_build_qwen4b_ptv23_complement.py \
  tools/launcher/tests/test_qwen4b_b_atomic.py \
  tools/launcher/tests/test_q30t_row_schema_observation_runner.py \
  tools/launcher/tests/test_q30t_700k_data_evidence_runner.py
```

Expected: all tests PASS with no skips for trust-boundary cases.

- [ ] **Step 2: Run Ruff formatting and lint checks**

```bash
../q4-ptv2-ab-integration/.venv/bin/ruff format --check \
  examples/dataset/specdec_identity.py \
  examples/dataset/audit_ptv2_baseline.py \
  examples/dataset/observe_q30t_ptv23_row_schemas.py \
  examples/dataset/build_q30t_heldout_receipts.py \
  examples/dataset/build_q30t_ptv23_source_inventory.py \
  examples/dataset/review_q30t_700k_data_evidence.py \
  examples/dataset/build_qwen4b_ptv23_complement.py \
  examples/dataset/ptv23_complement_target_policy.py \
  tests/examples/dataset tools/launcher/tests/test_q30t_700k_data_evidence_runner.py
../q4-ptv2-ab-integration/.venv/bin/ruff check --no-fix \
  examples/dataset/specdec_identity.py \
  examples/dataset/audit_ptv2_baseline.py \
  examples/dataset/observe_q30t_ptv23_row_schemas.py \
  examples/dataset/build_q30t_heldout_receipts.py \
  examples/dataset/build_q30t_ptv23_source_inventory.py \
  examples/dataset/review_q30t_700k_data_evidence.py \
  examples/dataset/build_qwen4b_ptv23_complement.py \
  examples/dataset/ptv23_complement_target_policy.py \
  tests/examples/dataset tools/launcher/tests/test_q30t_700k_data_evidence_runner.py
```

Expected: both commands exit 0 without modifying files.

- [ ] **Step 3: Run scoped Pyright and bytecode compilation**

```bash
pyright \
  examples/dataset/specdec_identity.py \
  examples/dataset/audit_ptv2_baseline.py \
  examples/dataset/observe_q30t_ptv23_row_schemas.py \
  examples/dataset/build_q30t_heldout_receipts.py \
  examples/dataset/build_q30t_ptv23_source_inventory.py \
  examples/dataset/review_q30t_700k_data_evidence.py \
  examples/dataset/build_qwen4b_ptv23_complement.py \
  examples/dataset/ptv23_complement_target_policy.py
PYTHONDONTWRITEBYTECODE=1 ../q4-ptv2-ab-integration/.venv/bin/python -m py_compile \
  examples/dataset/specdec_identity.py \
  examples/dataset/audit_ptv2_baseline.py \
  examples/dataset/observe_q30t_ptv23_row_schemas.py \
  examples/dataset/build_q30t_heldout_receipts.py \
  examples/dataset/build_q30t_ptv23_source_inventory.py \
  examples/dataset/review_q30t_700k_data_evidence.py \
  examples/dataset/build_qwen4b_ptv23_complement.py \
  examples/dataset/ptv23_complement_target_policy.py
```

Expected: both commands exit 0.

- [ ] **Step 4: Run shell/static repository gates**

```bash
bash -n tools/launcher/common/specdec/run_q30t_700k_data_evidence.sbatch \
  tools/launcher/common/specdec/submit_q30t_700k_data_evidence.sh \
  tools/launcher/common/specdec/run_q30t_row_schema_observation.sbatch \
  tools/launcher/common/specdec/submit_q30t_row_schema_observation.sh
shellcheck tools/launcher/common/specdec/run_q30t_700k_data_evidence.sbatch \
  tools/launcher/common/specdec/submit_q30t_700k_data_evidence.sh \
  tools/launcher/common/specdec/run_q30t_row_schema_observation.sbatch \
  tools/launcher/common/specdec/submit_q30t_row_schema_observation.sh
git diff --check
```

Expected: all commands exit 0.

- [ ] **Step 5: Freeze exact content hashes for hostile review**

```bash
PYTHONDONTWRITEBYTECODE=1 ../q4-ptv2-ab-integration/.venv/bin/python - <<'PY' \
  > /tmp/q30t-data-files.sha256
from hashlib import sha256
from pathlib import Path

paths = (
    "examples/dataset/specdec_identity.py",
    "examples/dataset/audit_ptv2_baseline.py",
    "examples/dataset/observe_q30t_ptv23_row_schemas.py",
    "examples/dataset/build_q30t_heldout_receipts.py",
    "examples/dataset/build_q30t_ptv23_source_inventory.py",
    "examples/dataset/review_q30t_700k_data_evidence.py",
    "examples/dataset/build_qwen4b_ptv23_complement.py",
    "examples/dataset/ptv23_complement_target_policy.py",
    "examples/dataset/qwen3_30ba3b_thinking_ptv23_swe_heavy_source_mapping_v1.json",
    "examples/dataset/qwen3_30ba3b_thinking_ptv23_swe_heavy_sources_v1.json",
    "examples/dataset/qwen3_30ba3b_thinking_ptv23_swe_heavy_700k_v1.json",
    "examples/dataset/qwen3_30ba3b_thinking_ptv23_swe_heavy_target_v1.json",
    "tools/launcher/common/specdec/run_q30t_row_schema_observation.sbatch",
    "tools/launcher/common/specdec/submit_q30t_row_schema_observation.sh",
    "tools/launcher/common/specdec/run_q30t_700k_data_evidence.sbatch",
    "tools/launcher/common/specdec/submit_q30t_700k_data_evidence.sh",
    "tests/examples/dataset/test_specdec_identity.py",
    "tests/examples/dataset/test_audit_ptv2_baseline.py",
    "tests/examples/dataset/test_trajectory_schema.py",
    "tests/examples/dataset/test_observe_q30t_ptv23_row_schemas.py",
    "tests/examples/dataset/test_build_q30t_heldout_receipts.py",
    "tests/examples/dataset/test_build_q30t_ptv23_source_inventory.py",
    "tests/examples/dataset/test_review_q30t_700k_data_evidence.py",
    "tests/examples/dataset/test_q30t_swe_heavy_700k_config.py",
    "tests/examples/dataset/test_ptv23_complement_target_policy.py",
    "tests/examples/dataset/test_build_qwen4b_ptv23_complement.py",
    "tools/launcher/tests/test_q30t_row_schema_observation_runner.py",
    "tools/launcher/tests/test_q30t_700k_data_evidence_runner.py",
)
for name in paths:
    path = Path(name)
    print(sha256(path.read_bytes()).hexdigest(), name)
PY
sed -n '1,240p' /tmp/q30t-data-files.sha256
```

Expected: the list contains only Subproject A files from this plan and excludes `uv.lock`, controller, runtime, canary, training, and receipt artifacts.

- [ ] **Step 6: Obtain final independent hostile review**

Provide the approved spec, this plan, `git diff --stat`, `/tmp/q30t-data-files.sha256`, focused test output, and static-gate output. Reviewer must explicitly mark CLEAN for direct BaselineAudit lineage, shared identity, held-out provenance, every-file schemas, centralized tokenizer approval, distinct SWE-heavy policy, descriptor-bound publication, exact 700K build, and full replay.

- [ ] **Step 7: Commit only review corrections, rerun the full gate, and hand off**

If review requires code changes, return to the affected task’s RED step, implement the minimal correction, and repeat Tasks 11.1-11.6. When CLEAN, commit only the explicitly reviewed files with signed DCO commits and hand the reviewed dataset completion path/SHA to Subproject B.
