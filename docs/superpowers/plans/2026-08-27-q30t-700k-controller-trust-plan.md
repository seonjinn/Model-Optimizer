# Q30 Thinking SWE-Heavy 700K Controller Trust Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the Q30 DFlash/DSpark controller admit only the independently approved Q30 tokenizer and parent receipts, an authoritatively replayed exact SWE-heavy 700K dataset, and a source/runtime/completion tuple extracted from immutable Git objects before any keeper or training process starts.

**Architecture:** Receipt consumption remains Q30-native: a centralized tokenizer loader and the existing parent loader own whole-file approval and evidence replay. Dataset semantics remain owned by `build_qwen4b_ptv23_complement.py`; a small controller adapter executes its `verify` CLI and direct dependencies from blobs extracted from the contract's authenticated Git commit under isolated Python and an allowlisted sterile environment. The typed contract and canonical descriptor carry every input path/hash, while the runner consumes only the replayed environment and rechecks the same source/runtime/completion identities before canary or full execution.

**Tech Stack:** Python 3.12+, frozen dataclasses and `pathlib`, canonical JSON with SHA-256, descriptor-relative `openat`/`O_NOFOLLOW` file authentication, Git object extraction, `subprocess.run`, Bash/SLURM/Pyxis, pytest, Ruff, Pyright, ShellCheck.

**Spec:** `docs/superpowers/specs/2026-08-27-q30t-swe-heavy-700k-evidence-launch-design.md`

## Global Constraints

- Scientific identity is exactly `ptv2-ptv3-complement-700k-qwen3-30ba3b-thinking-swe-heavy-v1`.
- Exposure is exactly 700,000 occurrences once: steps 1-1,367 use global batch 512 and step 1,368 uses global batch 96; no redistribution, replacement, padding, cycling, or duplicate exposure.
- Quotas are exactly: PTV2 STEM 200,000; PTV2 Japanese/Spanish/French/Italian 25,000 each; PTV3 SWE-v3 180,000; PTV3 interactive agentic/SWE 60,000; PTV3 general tool trajectories 160,000.
- Tokenizer approval root is exactly `5ba642c455e60b67eca295dce92dd7da47292fdba66c5f9d269669c14cafc509`.
- DFlash and DSpark parent approval roots are exactly `393a2b7c5cbe2037914cbfa531d1d6fdce2bcac2f4204aa001db89d345c3a500` and `d5416b8fd9644802b42f04511791418bf00dd023f8c5df96593010be95ff2571`.
- Those receipts bind producer commit `212d01516b32e5a81f077d7f2f26ec8de38707c6`, target revision `144afc2f379b542fdd4e85a1fcd5e1f79112d95d`, and target tree `7bd176a868273ca8acf8db0ec1ab528f7972b0ae5bcf67ed4b3726368b9f7d13`.
- Parent lineage also binds runtime archive `4a20aee61f290c48bed22a84b4a0ae0cbdc54e3e3910854d253188c8854f5dc9`, historical audit `2469430c144d9b86850901df0a28cb810b437555ee894771b796b1386d1c18b5`, and ordered historical digest `863470b22925d74228d31b1c2433d9461d25e8d02a298cb3a08a6bc99be55060`.
- Required held-out receipt names and canonical order are exactly `speed`, `math`, `code`, `swe`, `tool`.
- No receipt becomes an approval root until generation and independent replay are complete; unavailable Subproject A/C hashes remain fail-closed empty production roots.
- Never adapt `qwen4b_ptv23_continuation.ContinuationContract` as a Q30 security boundary, and do not change Q4 balanced policy or behavior.
- Immutable replay never imports `examples/dataset` from the mutable checkout pathname.
- Existing or foreign destinations are preserved and rejected; controller work must not add recursive deletion or cleanup of unowned paths.
- Production/Ptyche controller replay uses exactly `/usr/bin/python3.12 -I` and rejects `sys.version_info < (3, 12)` before importing any controller or builder module.
- Every implementation slice ends with focused pytest, Ruff format/check, scoped Pyright or `py_compile`, Bash syntax/ShellCheck when shell changes, and `git diff --check`.
- Job submission is outside implementation: no canary or full job is submitted until Subprojects A and C provide independently reviewed approval roots and both test-only gates pass.

---

## File and interface map

- `tools/launcher/common/specdec/q30t_tokenizer_receipt.py`: own the single approved tokenizer whole-file set and `load_q30t_tokenizer_receipt(path: Path, *, expected_sha256: str) -> dict[str, object]`.
- `examples/dataset/build_qwen4b_ptv23_complement.py`: external Data-plan-owned authoritative verifier; this plan consumes its frozen CLI/interface but never edits it.
- `tools/launcher/common/specdec/q30t_parent_receipt.py`: insert the two reviewed parent roots; retain `load_q30t_parent_receipt` and historical pair reconciliation.
- `tools/launcher/common/specdec/q30t_dataset_replay.py`: new narrow adapter and CLI for immutable builder replay; it accepts only explicit typed paths/hashes and never chooses approvals.
- `tools/launcher/common/specdec/q30t_ptv23_continuation.py`: expand `Q30TContinuationContract`, authenticate every dataset/source/runtime field, invoke the adapter, and serialize all fields canonically.
- `tools/launcher/common/specdec/run_q30t_ptv23_continuation.sbatch`: extract the exact builder dependency closure from Git objects and execute descriptor replay under `env -i` before keeper startup.
- `tools/launcher/common/specdec/submit_q30t_ptv23_continuation.sh`: retain clean-pushed/test-only staging and reject descriptor/controller mismatches.
- `tools/launcher/tests/test_q30t_tokenizer_receipt.py`, `test_q30t_parent_receipt.py`, `test_q30t_ptv23_continuation.py`, `test_q30t_ptv23_controller.py`: focused hostile tests for each trust boundary.

## Cross-plan ownership and sequencing

- Controller Task 1 owns the centralized loader in `q30t_tokenizer_receipt.py`, its focused loader tests, and Q30 controller consumption. It exposes `load_q30t_tokenizer_receipt(path: Path, *, expected_sha256: str) -> dict[str, object]` to the Data plan; the Data plan alone changes the builder to consume it.
- Controller Tasks 1-2 may proceed against the already reviewed tokenizer/parent roots. Task 3 and every later controller task wait until the Data plan freezes the exact SWE-heavy target policy, quota config, distinct source-requirements file, and their tests.
- Task 4 and later consume the Data plan's frozen authoritative builder verifier CLI and builder bytes. This plan must not change builder logic, target-policy/config/source-requirements files, or data-policy tests.
- This plan exclusively owns `q30t_dataset_replay.py`, `q30t_ptv23_continuation.py`, `run_q30t_ptv23_continuation.sbatch`, `submit_q30t_ptv23_continuation.sh`, and their focused controller tests until Task 9's controller freeze.
- The Runtime plan may produce and independently review qualification receipts in parallel, but it waits until the Task 9 controller freeze before integrating runtime approval roots or changing controller/runner/submitter files.

### Task 1: Centralize approved Q30 tokenizer loading

**Files:**
- Modify: `tools/launcher/common/specdec/q30t_tokenizer_receipt.py:24-40,281-335`
- Modify: `tools/launcher/common/specdec/q30t_ptv23_continuation.py:20-27,902-921`
- Test: `tools/launcher/tests/test_q30t_tokenizer_receipt.py`
- Test: `tools/launcher/tests/test_q30t_ptv23_continuation.py`

**Interfaces:**
- Consumes: existing `verify_q30t_tokenizer_receipt(receipt: bytes) -> dict[str, object]` and stable descriptor helpers in `q30t_tokenizer_receipt.py`.
- Produces: `APPROVED_Q30T_TOKENIZER_RECEIPT_FILE_SHA256S: frozenset[str]` and the cross-plan interface `load_q30t_tokenizer_receipt(path: Path, *, expected_sha256: str) -> dict[str, object]`. The controller consumes it in this task; the Data plan separately changes the builder to consume the same interface.

- [ ] **Step 1: Write RED tests for caller hash, approval membership, full replay, and single-link stability**

Add tests that construct a valid fixture receipt, patch the approval set only for fixture bytes, and exercise the path loader:

```python
def test_q30t_approved_loader_hashes_membership_and_replays(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    snapshot = _make_q30_snapshot(tmp_path)
    raw = build_q30t_tokenizer_receipt(snapshot, Q30_REPOSITORY, FIXTURE_Q30_REVISION)
    receipt = tmp_path / "TOKENIZER.json"
    receipt.write_bytes(raw)
    receipt_sha256 = hashlib.sha256(raw).hexdigest()
    monkeypatch.setattr(
        receipt_module,
        "APPROVED_Q30T_TOKENIZER_RECEIPT_FILE_SHA256S",
        frozenset({receipt_sha256}),
    )

    loaded = receipt_module.load_q30t_tokenizer_receipt(
        receipt, expected_sha256=receipt_sha256
    )

    assert loaded["revision"] == FIXTURE_Q30_REVISION
    with pytest.raises(ValueError, match="caller SHA-256 mismatch"):
        receipt_module.load_q30t_tokenizer_receipt(receipt, expected_sha256="0" * 64)
```

Add `test_q30t_approved_loader_rejects_unreviewed_valid_receipt`, `test_q30t_approved_loader_rejects_hardlinked_receipt`, and `test_q30t_approved_loader_rebinds_name_after_read` beside it. The rebind test replaces the pathname after the held descriptor is read and expects `ValueError("Q30 tokenizer receipt changed while reading")`.

- [ ] **Step 2: Run the loader tests and capture the expected RED**

Run:

```bash
PYTHONPATH=tools/launcher python3 -m pytest -q \
  tools/launcher/tests/test_q30t_tokenizer_receipt.py \
  -k 'approved_loader'
```

Expected: collection succeeds and tests fail with `AttributeError: module 'common.specdec.q30t_tokenizer_receipt' has no attribute 'load_q30t_tokenizer_receipt'`.

- [ ] **Step 3: Implement the centralized loader and exact production root**

Export and define the approved set and loader in `q30t_tokenizer_receipt.py`:

```python
APPROVED_Q30T_TOKENIZER_RECEIPT_FILE_SHA256S: frozenset[str] = frozenset(
    {"5ba642c455e60b67eca295dce92dd7da47292fdba66c5f9d269669c14cafc509"}
)


def load_q30t_tokenizer_receipt(
    path: Path, *, expected_sha256: str
) -> dict[str, object]:
    """Load one independently reviewed receipt and replay its snapshot evidence."""
    if not _is_lower_hex(expected_sha256, 64):
        raise ValueError("Q30 tokenizer receipt caller SHA-256 is invalid")
    file_sha256, raw = _read_stable_receipt(path, require_single_link=True)
    if file_sha256 != expected_sha256:
        raise ValueError("Q30 tokenizer receipt caller SHA-256 mismatch")
    if expected_sha256 not in APPROVED_Q30T_TOKENIZER_RECEIPT_FILE_SHA256S:
        raise ValueError("Q30 tokenizer receipt is not independently reviewed")
    return verify_q30t_tokenizer_receipt(raw)
```

Implement `_read_stable_receipt` with a parent directory descriptor, `O_NOFOLLOW | O_NONBLOCK`, `st_nlink == 1`, streamed SHA-256, and pre/open/post/named `(dev, ino, mode, nlink, size, mtime_ns, ctime_ns)` reconciliation. Do not use `Path.read_bytes()`:

```python
_MAX_Q30T_TOKENIZER_RECEIPT_BYTES = 1024 * 1024


def _read_stable_receipt(path: Path, *, require_single_link: bool) -> tuple[str, bytes]:
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0)
    parent_fd = os.open(path.parent, flags | os.O_DIRECTORY)
    try:
        expected = os.stat(path.name, dir_fd=parent_fd, follow_symlinks=False)
        descriptor = os.open(path.name, flags, dir_fd=parent_fd)
        try:
            before = os.fstat(descriptor)
            if (
                not stat.S_ISREG(before.st_mode)
                or (require_single_link and before.st_nlink != 1)
                or before.st_size > _MAX_Q30T_TOKENIZER_RECEIPT_BYTES
            ):
                raise ValueError("Q30 tokenizer receipt must be a single-link regular file")
            digest = sha256()
            raw = bytearray()
            while block := os.read(descriptor, 1024 * 1024):
                digest.update(block)
                raw.extend(block)
            after = os.fstat(descriptor)
        finally:
            os.close(descriptor)
        named = os.stat(path.name, dir_fd=parent_fd, follow_symlinks=False)
    finally:
        os.close(parent_fd)

    def identity(value: os.stat_result) -> tuple[int, int, int, int, int, int, int]:
        return (
            value.st_dev,
            value.st_ino,
            value.st_mode,
            value.st_nlink,
            value.st_size,
            value.st_mtime_ns,
            value.st_ctime_ns,
        )

    if (
        identity(expected) != identity(before)
        or identity(before) != identity(after)
        or identity(after) != identity(named)
        or len(raw) != before.st_size
    ):
        raise ValueError("Q30 tokenizer receipt changed while reading")
    return digest.hexdigest(), bytes(raw)
```

- [ ] **Step 4: Replace the controller's raw tokenizer verifier path**

Import `load_q30t_tokenizer_receipt` instead of `verify_q30t_tokenizer_receipt`. Replace the controller's raw receipt read/replay with:

```python
tokenizer = load_q30t_tokenizer_receipt(
    contract.tokenizer_receipt_path,
    expected_sha256=contract.tokenizer_receipt_file_sha256,
)
```

Retain the controller's exact repository/revision/snapshot-path/tree comparison against the contract. Do not edit the builder in this task; Data-plan tests own builder consumption and Q4 regression coverage.

- [ ] **Step 5: Run tokenizer loader and controller consumer GREEN tests**

Run:

```bash
PYTHONPATH=tools/launcher python3 -m pytest -q \
  tools/launcher/tests/test_q30t_tokenizer_receipt.py \
  -k 'approved_loader'
PYTHONPATH=tools/launcher python3 -m pytest -q \
  tools/launcher/tests/test_q30t_ptv23_continuation.py \
  -k 'authenticates_parent_tokenizer or cross_bound_parent_or_tokenizer'
```

Expected: all selected tests pass; the production-root assertion equals the exact `5ba642c455e60b67eca295dce92dd7da47292fdba66c5f9d269669c14cafc509` singleton.

- [ ] **Step 6: Review and commit the tokenizer consumer boundary**

Run:

```bash
python3 -m ruff format --check tools/launcher/common/specdec/q30t_tokenizer_receipt.py \
  tools/launcher/common/specdec/q30t_ptv23_continuation.py \
  tools/launcher/tests/test_q30t_tokenizer_receipt.py \
  tools/launcher/tests/test_q30t_ptv23_continuation.py
python3 -m ruff check tools/launcher/common/specdec/q30t_tokenizer_receipt.py \
  tools/launcher/common/specdec/q30t_ptv23_continuation.py \
  tools/launcher/tests/test_q30t_tokenizer_receipt.py \
  tools/launcher/tests/test_q30t_ptv23_continuation.py
python3 -m py_compile tools/launcher/common/specdec/q30t_tokenizer_receipt.py \
  tools/launcher/common/specdec/q30t_ptv23_continuation.py
git diff --check
git diff -- tools/launcher/common/specdec/q30t_tokenizer_receipt.py \
  tools/launcher/common/specdec/q30t_ptv23_continuation.py \
  tools/launcher/tests/test_q30t_tokenizer_receipt.py \
  tools/launcher/tests/test_q30t_ptv23_continuation.py
```

Expected: clean static gates, exact centralized loader consumption, and no builder/Q4 diff. After independent review:

```bash
git add tools/launcher/common/specdec/q30t_tokenizer_receipt.py \
  tools/launcher/common/specdec/q30t_ptv23_continuation.py \
  tools/launcher/tests/test_q30t_tokenizer_receipt.py \
  tools/launcher/tests/test_q30t_ptv23_continuation.py
git commit -s -m "feat: centralize approved Q30 tokenizer loading"
```

### Task 2: Approve and retain native Q30 parent loading

**Files:**
- Modify: `tools/launcher/common/specdec/q30t_parent_receipt.py:30-42,132-137`
- Test: `tools/launcher/tests/test_q30t_parent_receipt.py`
- Test: `tools/launcher/tests/test_q30t_ptv23_continuation.py`

**Interfaces:**
- Consumes: existing `load_q30t_parent_receipt(path: Path, expected_sha256: str) -> Q30TParentReceipt` and `require_matching_historical_lineage(first, second) -> None`.
- Produces: exact two-member `APPROVED_Q30T_PARENT_RECEIPT_FILE_SHA256S`; no Q4 adapter or reconstructed parent type.

- [ ] **Step 1: Write RED approval-root and native-loader tests**

Replace the old empty-set assertion with:

```python
def test_q30t_parent_approval_roots_are_the_independently_reviewed_pair() -> None:
    assert receipt_module.APPROVED_Q30T_PARENT_RECEIPT_FILE_SHA256S == frozenset(
        {
            "393a2b7c5cbe2037914cbfa531d1d6fdce2bcac2f4204aa001db89d345c3a500",
            "d5416b8fd9644802b42f04511791418bf00dd023f8c5df96593010be95ff2571",
        }
    )
```

In `test_q30t_ptv23_continuation.py`, add `test_q30_controller_uses_native_q30_parent_loader`, an AST/import assertion that `q30t_ptv23_continuation` imports `load_q30t_parent_receipt` from `common.specdec.q30t_parent_receipt` and never imports `AuthenticatedParentCheckpoint` or `ContinuationContract` from the Q4 module.

- [ ] **Step 2: Run the parent tests and verify RED**

Run:

```bash
PYTHONPATH=tools/launcher python3 -m pytest -q \
  tools/launcher/tests/test_q30t_parent_receipt.py::test_q30t_parent_approval_roots_are_the_independently_reviewed_pair
```

Expected: FAIL because the current production set is empty.

- [ ] **Step 3: Insert only the independently reviewed parent roots**

Use the exact constant:

```python
APPROVED_Q30T_PARENT_RECEIPT_FILE_SHA256S: frozenset[str] = frozenset(
    {
        "393a2b7c5cbe2037914cbfa531d1d6fdce2bcac2f4204aa001db89d345c3a500",
        "d5416b8fd9644802b42f04511791418bf00dd023f8c5df96593010be95ff2571",
    }
)
```

Do not change receipt generation, parent identities, checkpoint paths, historical roots, or publication behavior.

- [ ] **Step 4: Run parent loader and shared-lineage GREEN tests**

Run:

```bash
PYTHONPATH=tools/launcher python3 -m pytest -q \
  tools/launcher/tests/test_q30t_parent_receipt.py \
  -k 'approval_roots or cannot_self_authorize or parent_pair_rejects_duplicate_methods' \
  tools/launcher/tests/test_q30t_ptv23_continuation.py \
  -k 'parent_pair_requires_one_shared_historical_lineage or native_parent_loader'
```

Expected: all selected tests pass; fixture tests continue monkeypatching fixture approvals rather than relying on production roots.

- [ ] **Step 5: Review and commit the parent approval slice**

Run static gates and inspect the two-line semantic diff. Then:

```bash
git add tools/launcher/common/specdec/q30t_parent_receipt.py \
  tools/launcher/tests/test_q30t_parent_receipt.py \
  tools/launcher/tests/test_q30t_ptv23_continuation.py
git commit -s -m "feat: approve reviewed Q30 parent receipts"
```

### Task 3: Freeze the controller's SWE-heavy identity and quota contract

**Files:**
- Modify: `tools/launcher/common/specdec/q30t_ptv23_continuation.py:42-61,228-238`
- Test: `tools/launcher/tests/test_q30t_ptv23_continuation.py`

**Interfaces:**
- Consumes: the Data plan's frozen canonical target revision `144afc2f379b542fdd4e85a1fcd5e1f79112d95d`, exact SWE-heavy policy/config constants, distinct source-requirements identity, and passing data-policy/config tests. Do not start Task 3 until that freeze is present in the integration branch.
- Produces: immutable controller constants `Q30T_SWE_HEAVY_SCIENTIFIC_IDENTITY`, `Q30T_SWE_HEAVY_QUOTAS`, `Q30T_TARGET_REVISION`, and `Q30T_TARGET_TREE_SHA256`.

- [ ] **Step 1: Write RED tests for the complete target/identity/quota tuple**

Add:

```python
def test_q30_controller_freezes_complete_swe_heavy_target_tuple() -> None:
    assert continuation.Q30T_SWE_HEAVY_SCIENTIFIC_IDENTITY == (
        "ptv2-ptv3-complement-700k-qwen3-30ba3b-thinking-swe-heavy-v1"
    )
    assert continuation.Q30T_TARGET_REVISION == "144afc2f379b542fdd4e85a1fcd5e1f79112d95d"
    assert continuation.Q30T_TARGET_TREE_SHA256 == (
        "7bd176a868273ca8acf8db0ec1ab528f7972b0ae5bcf67ed4b3726368b9f7d13"
    )
    assert continuation.Q30T_SWE_HEAVY_QUOTAS == {
        "ptv2_stem": 200_000,
        "ptv2_multilingual_ja": 25_000,
        "ptv2_multilingual_es": 25_000,
        "ptv2_multilingual_fr": 25_000,
        "ptv2_multilingual_it": 25_000,
        "ptv3_swe_v3": 180_000,
        "ptv3_interactive_agentic_swe": 60_000,
        "ptv3_general_tool_trajectories": 160_000,
    }
    assert sum(continuation.Q30T_SWE_HEAVY_QUOTAS.values()) == 700_000
```

- [ ] **Step 2: Run the tuple test and verify RED**

Run the single test. Expected: FAIL because the public constant names do not exist.

- [ ] **Step 3: Rename private constants and enforce them in contract construction**

Define the exact constants above as immutable values (`MappingProxyType` for quotas). In `Q30TContinuationContract.__post_init__`, reject any target revision/tree differing from the constants before filesystem authentication. Update `_require_dataset_identity` to compare plain dictionaries with `dict(Q30T_SWE_HEAVY_QUOTAS)` and preserve exact category insertion order.

```python
Q30T_SWE_HEAVY_SCIENTIFIC_IDENTITY = (
    "ptv2-ptv3-complement-700k-qwen3-30ba3b-thinking-swe-heavy-v1"
)
Q30T_TARGET_REVISION = "144afc2f379b542fdd4e85a1fcd5e1f79112d95d"
Q30T_TARGET_TREE_SHA256 = "7bd176a868273ca8acf8db0ec1ab528f7972b0ae5bcf67ed4b3726368b9f7d13"
Q30T_SWE_HEAVY_QUOTAS: Mapping[str, int] = MappingProxyType(
    {
        "ptv2_stem": 200_000,
        "ptv2_multilingual_ja": 25_000,
        "ptv2_multilingual_es": 25_000,
        "ptv2_multilingual_fr": 25_000,
        "ptv2_multilingual_it": 25_000,
        "ptv3_swe_v3": 180_000,
        "ptv3_interactive_agentic_swe": 60_000,
        "ptv3_general_tool_trajectories": 160_000,
    }
)
```

- [ ] **Step 4: Add negative tests for balanced identity, balanced quotas, and revision substitution**

Use `contract.replace(target_revision="0" * 40)`, a manifest with the balanced Q30 identity, and the 300K/50K balanced quota split. Each must raise before `load_q30t_parent_receipt`, tokenizer loading, keeper creation, or training command construction.

- [ ] **Step 5: Run GREEN identity tests and commit**

Run:

```bash
PYTHONPATH=tools/launcher python3 -m pytest -q \
  tools/launcher/tests/test_q30t_ptv23_continuation.py \
  -k 'swe_heavy or reviewed_swe_heavy or wrong_target'
```

Expected: selected tests pass. Commit after static review:

```bash
git add tools/launcher/common/specdec/q30t_ptv23_continuation.py \
  tools/launcher/tests/test_q30t_ptv23_continuation.py
git commit -s -m "refactor: freeze Q30 SWE-heavy controller identity"
```

### Task 4: Define the immutable builder replay adapter

**Files:**
- Create: `tools/launcher/common/specdec/q30t_dataset_replay.py`
- Create: `tools/launcher/tests/test_q30t_dataset_replay.py`
- Modify: `tools/launcher/common/specdec/q30t_ptv23_continuation.py:24-27,659-812`

**Interfaces:**
- Consumes: the Data plan's frozen `build_qwen4b_ptv23_complement.py verify` CLI with exact arguments `--target-policy`, `--target-policy-sha256`, `--config`, `--source-inventory`, `--source-inventory-sha256`, repeated `--held-out-receipt`, `--historical-receipt`, `--historical-receipt-sha256`, `--output-root`, `--completion-receipt`, `--completion-receipt-sha256`, `--tokenizer-trust`, `--tokenizer-trust-sha256`, `--runtime-sha256`, `--source-commit`, and `--scratch-root`. The Data plan guarantees that successful exit means full `verify_selection_bundle` replay and exposes the final builder file SHA-256. Task 4 starts only after that interface and hash are frozen.
- Consumes: an already extracted immutable root containing that builder and its frozen dependency closure, plus explicit contract paths/hashes. This task never edits the builder or its tests.
- Produces: `Q30TDatasetReplayRequest`, `q30t_builder_dependency_paths() -> tuple[PurePosixPath, ...]`, `materialize_q30t_builder_tree(source_checkout_path: Path, source_commit: str) -> AbstractContextManager[Path]`, and `replay_q30t_dataset(request: Q30TDatasetReplayRequest, *, immutable_source_root: Path | None = None) -> None`.

- [ ] **Step 1: Write RED request-schema and command-construction tests**

Create tests around this exact frozen request:

```python
@dataclass(frozen=True)
class Q30TDatasetReplayRequest:
    source_checkout_path: Path
    builder_source_commit: str
    builder_file_sha256: str
    target_policy_relative_path: PurePosixPath
    target_policy_file_sha256: str
    quota_config_relative_path: PurePosixPath
    quota_config_file_sha256: str
    source_inventory_path: Path
    source_inventory_file_sha256: str
    historical_audit_path: Path
    historical_audit_file_sha256: str
    held_out_receipts: tuple[tuple[str, Path, str], ...]
    tokenizer_receipt_path: Path
    tokenizer_receipt_file_sha256: str
    capacity_receipt_path: Path
    capacity_receipt_file_sha256: str
    dataset_root: Path
    manifest_file_sha256: str
    completion_receipt_path: Path
    completion_receipt_file_sha256: str
    runtime_sha256: str
    verification_scratch_root: Path
```

Assert dependency paths are exactly:

```python
(
    PurePosixPath("examples/dataset/build_qwen4b_ptv23_complement.py"),
    PurePosixPath("examples/dataset/ptv23_complement_target_policy.py"),
    PurePosixPath("examples/dataset/qwen3_4b_ptv23_target_v1.json"),
    PurePosixPath("examples/dataset/qwen3_30ba3b_thinking_ptv23_swe_heavy_target_v1.json"),
    PurePosixPath("examples/dataset/qwen3_30ba3b_thinking_ptv23_swe_heavy_700k_v1.json"),
    PurePosixPath("examples/dataset/qwen3_30ba3b_thinking_ptv23_swe_heavy_sources_v1.json"),
    PurePosixPath("examples/dataset/trajectory_schema.py"),
    PurePosixPath("examples/dataset/specdec_identity.py"),
    PurePosixPath("examples/dataset/specdec_corpus_contracts.py"),
    PurePosixPath("tools/launcher/common/specdec/q30t_tokenizer_receipt.py"),
    PurePosixPath("tools/launcher/common/specdec/q30t_tree_digest.py"),
    PurePosixPath("tools/launcher/common/specdec/qwen4b_b_atomic.py"),
)
```

This exact SWE-heavy source-requirements filename is the Data/Controller-plan interface. If it is absent, stop at RED and resolve the cross-plan contract in the approved design before implementation; never fall back to the balanced file or a wildcard dependency list.

- [ ] **Step 2: Run adapter tests and verify RED**

Run:

```bash
PYTHONPATH=tools/launcher python3 -m pytest -q tools/launcher/tests/test_q30t_dataset_replay.py
```

Expected: FAIL because `common.specdec.q30t_dataset_replay` does not exist.

- [ ] **Step 3: Implement strict request validation and sterile subprocess construction**

Validate exact lowercase hashes/commit, exact canonical relative policy/config paths, absolute canonical evidence paths, disjoint fresh scratch, fixed held-out order, and immutable-root containment. Reject relative components outside the fixed dependency inventory. Build the subprocess without a shell:

Define the fixed `bootstrap` immediately before constructing `command`:

```python
bootstrap = """\
import runpy
import sys
if sys.version_info < (3, 12):
    raise SystemExit("Q30 dataset replay requires Python 3.12+")
launcher_root, dataset_root, builder = sys.argv[1:4]
sys.path[:0] = [launcher_root, dataset_root]
sys.argv = [builder, *sys.argv[4:]]
runpy.run_path(builder, run_name="__main__")
"""

command = (
    "/usr/bin/env",
    "-i",
    "LC_ALL=C",
    "PYTHONHASHSEED=0",
    "PYTHONNOUSERSITE=1",
    "/usr/bin/python3.12",
    "-I",
    "-c",
    bootstrap,
    str(launcher_root),
    str(extracted_dataset_module_root),
    str(builder),
    "verify",
    "--target-policy", str(immutable_source_root / request.target_policy_relative_path),
    "--target-policy-sha256", request.target_policy_file_sha256,
    "--config", str(immutable_source_root / request.quota_config_relative_path),
    "--source-inventory", str(request.source_inventory_path),
    "--source-inventory-sha256", request.source_inventory_file_sha256,
    "--historical-receipt", str(request.historical_audit_path),
    "--historical-receipt-sha256", request.historical_audit_file_sha256,
    *held_out_arguments,
    "--output-root", str(request.dataset_root),
    "--completion-receipt", str(request.completion_receipt_path),
    "--completion-receipt-sha256", request.completion_receipt_file_sha256,
    "--tokenizer-trust", str(request.tokenizer_receipt_path),
    "--tokenizer-trust-sha256", request.tokenizer_receipt_file_sha256,
    "--runtime-sha256", request.runtime_sha256,
    "--source-commit", request.builder_source_commit,
    "--scratch-root", str(request.verification_scratch_root),
)
```

`-I` intentionally ignores ambient and environment-provided `PYTHONPATH`. The test must assert that neither the command nor bootstrap contains the mutable checkout path. When `immutable_source_root is None`, enter `materialize_q30t_builder_tree(request.source_checkout_path, request.builder_source_commit)` and execute from its yielded root. When the runner supplies a root, require its exact dependency inventory and Git-blob manifest before using it.

- [ ] **Step 4: Materialize local preflight dependencies from exact Git objects**

Implement `materialize_q30t_builder_tree` as a context manager backed by `tempfile.TemporaryDirectory`. Run sterile `/usr/bin/git ls-tree` and `cat-file blob` for only `q30t_builder_dependency_paths()`, require exact set equality and regular modes, recompute every Git blob OID, write each blob with `O_CREAT | O_EXCL | O_NOFOLLOW`, fsync files/directories, and yield the fresh root. On exit, allow `TemporaryDirectory` to clean up only the directory it created; never unlink a caller-provided or foreign path.

- [ ] **Step 5: Bind exact builder bytes before execution and after replay**

Stream-hash the extracted builder through a held descriptor and require `builder_file_sha256`. After subprocess success, repeat the stable hash and fail if identity or digest changed. Convert nonzero exit status into `ValueError("authoritative Q30 dataset replay failed")` while preserving bounded stderr for diagnostics.

- [ ] **Step 6: Add hostile adapter tests**

Cover: wrong builder hash; extra/reordered held-out names; mutable-checkout import injection; unknown environment variable visibility; builder replacement between prehash and exec; completion `source_commit` mismatch; and a fake builder that records argv/env and exits zero. The fake environment must equal the explicit allowlist and omit `HOME`, `PYTHONPATH` inherited from the caller, Git variables, tokens, and scheduler state.

- [ ] **Step 7: Run adapter GREEN tests and commit**

Run pytest, Ruff, Pyright/`py_compile`, and `git diff --check`. Then:

```bash
git add tools/launcher/common/specdec/q30t_dataset_replay.py \
  tools/launcher/tests/test_q30t_dataset_replay.py \
  tools/launcher/common/specdec/q30t_ptv23_continuation.py
git commit -s -m "feat: add immutable Q30 dataset replay adapter"
```

### Task 5: Expand the Q30 contract and descriptor trust surface

**Files:**
- Modify: `tools/launcher/common/specdec/q30t_ptv23_continuation.py:479-646,1147-1258`
- Modify: `tools/launcher/tests/test_q30t_ptv23_continuation.py:41-220,561-617`

**Interfaces:**
- Consumes: `Q30TDatasetReplayRequest` from Task 4.
- Produces: an expanded `Q30TContinuationContract` whose canonical descriptor carries every Subproject B trust field, plus `load_q30_launch_descriptor(path, *, expected_sha256, expected_source_checkout, immutable_source_root: Path | None = None) -> Q30TTrainingLaunch`. The execution-only immutable root is never serialized into the descriptor.

- [ ] **Step 1: Write RED contract-field and descriptor mutation tests**

Add these fields to the test fixture's expected dataclass/descriptor schema:

```python
builder_source_commit: str
dataset_builder_path: Path
dataset_builder_file_sha256: str
target_policy_path: Path
target_policy_file_sha256: str
quota_config_path: Path
quota_config_file_sha256: str
source_inventory_path: Path
source_inventory_file_sha256: str
historical_audit_path: Path
held_out_receipts: tuple[tuple[str, Path, str], ...]
capacity_receipt_path: Path
capacity_receipt_file_sha256: str
verification_scratch_root: Path
runtime_qualification_receipt_path: Path
runtime_qualification_receipt_file_sha256: str
```

Parameterize every new field: delete it from descriptor JSON, add an extra field, mutate a path, and mutate a hash/commit. `load_q30_launch_descriptor` must reject each mutation before `build_q30_training_command` returns.

- [ ] **Step 2: Run descriptor tests and verify RED**

Expected failures are missing dataclass keyword arguments and descriptor schema mismatches, not filesystem errors.

- [ ] **Step 3: Add fields with exact type/path/hash validation**

Add path fields to `_CONTRACT_PATH_FIELDS`, serialize `held_out_receipts` as canonical records:

```json
[
  {"file_sha256":"1111111111111111111111111111111111111111111111111111111111111111","name":"speed","path":"/fixtures/heldout/speed.json"},
  {"file_sha256":"2222222222222222222222222222222222222222222222222222222222222222","name":"math","path":"/fixtures/heldout/math.json"},
  {"file_sha256":"3333333333333333333333333333333333333333333333333333333333333333","name":"code","path":"/fixtures/heldout/code.json"},
  {"file_sha256":"4444444444444444444444444444444444444444444444444444444444444444","name":"swe","path":"/fixtures/heldout/swe.json"},
  {"file_sha256":"5555555555555555555555555555555555555555555555555555555555555555","name":"tool","path":"/fixtures/heldout/tool.json"}
]
```

Reject non-list encodings, duplicate/reordered names, non-absolute paths, booleans masquerading as integers, symlink aliases, path overlap with writable roots, and any field count mismatch. Bump descriptor schema to `q30t-ptv23-launch-descriptor-v2`; reject v1 rather than silently defaulting security fields.

Thread `immutable_source_root` only through descriptor replay and `_authenticate_contract`; local preflight passes `None` so Task 4 creates a fresh Git-object tree, while the spooled runner passes its already authenticated bootstrap root. Do not add this ephemeral path to `_CONTRACT_FIELD_NAMES`.

- [ ] **Step 4: Ensure builder and parent commits remain distinct**

Add a test where `parent_source_commit` remains `212d01516b32e5a81f077d7f2f26ec8de38707c6` while `builder_source_commit` equals the later authenticated controller `source_commit`. Require `builder_source_commit == source_commit`, assert the parent tuple checks the former and dataset completion checks the latter, and never overwrite one with the other.

- [ ] **Step 5: Run round-trip GREEN tests and commit**

Run:

```bash
PYTHONPATH=tools/launcher python3 -m pytest -q \
  tools/launcher/tests/test_q30t_ptv23_continuation.py \
  -k 'descriptor or contract_rejects or builder_and_parent_commits'
```

Expected: all selected tests pass. Commit the independently reviewable schema break:

```bash
git add tools/launcher/common/specdec/q30t_ptv23_continuation.py \
  tools/launcher/tests/test_q30t_ptv23_continuation.py
git commit -s -m "feat: bind Q30 dataset trust in launch descriptors"
```

### Task 6: Compare every dataset completion and provenance field

**Files:**
- Modify: `tools/launcher/common/specdec/q30t_ptv23_continuation.py:659-923`
- Test: `tools/launcher/tests/test_q30t_ptv23_continuation.py`

**Interfaces:**
- Consumes: expanded contract from Task 5 and `replay_q30t_dataset` from Task 4.
- Produces: `_authenticate_dataset_bundle(contract) -> None` that performs complete comparison and authoritative replay before returning.

- [ ] **Step 1: Write RED mutations for every manifest trust and completion field**

Build one canonical fixture bundle, then parameterize mutations of:

```python
(
    ("trust.config_file_sha256", "0" * 64),
    ("trust.source_inventory_file_sha256", "0" * 64),
    ("trust.tokenizer_trust_file_sha256", "0" * 64),
    ("trust.target_policy_file_sha256", "0" * 64),
    ("trust.runtime_sha256", "0" * 64),
    ("trust.source_commit", "0" * 40),
    ("completion.runtime_sha256", "0" * 64),
    ("completion.source_commit", "0" * 40),
)
```

Recanonicalize and self-hash each malicious fixture so rejection proves field comparison rather than stale self-hash detection. Add held-out, capacity, historical ordered digest, DATA, MANIFEST, and completion whole-file substitution cases.

- [ ] **Step 2: Run mutation tests and verify RED**

Expected: the current controller wrongly accepts at least completion `runtime_sha256` and `source_commit`, demonstrating the trust gap.

- [ ] **Step 3: Compare manifest trust and completion tuples exactly**

Require:

```python
expected_trust = {
    "config_file_sha256": contract.quota_config_file_sha256,
    "source_inventory_file_sha256": contract.source_inventory_file_sha256,
    "tokenizer_trust_file_sha256": contract.tokenizer_receipt_file_sha256,
    "target_policy_file_sha256": contract.target_policy_file_sha256,
    "runtime_sha256": contract.runtime_archive_sha256,
    "source_commit": contract.builder_source_commit,
}
```

Require completion `runtime_sha256 == contract.runtime_archive_sha256` and `source_commit == contract.builder_source_commit`. Compare the exact ordered held-out names/hashes, capacity path/hash, historical audit hash/order digest, manifest/data hashes, output root, row count, identity, quotas, and category order.

- [ ] **Step 4: Preserve Task 1's centralized tokenizer boundary during provenance refactoring**

Keep this exact Task 1 call in `_authenticate_contract` while moving the surrounding dataset logic:

```python
tokenizer = load_q30t_tokenizer_receipt(
    contract.tokenizer_receipt_path,
    expected_sha256=contract.tokenizer_receipt_file_sha256,
)
```

Retain target repository/revision/snapshot/tree comparison against the contract and approved parent. Add a regression assertion that neither `verify_q30t_tokenizer_receipt` nor a second raw tokenizer receipt read reappears in the controller.

- [ ] **Step 5: Invoke authoritative replay after cheap structural checks**

Construct `Q30TDatasetReplayRequest` from the contract and invoke `replay_q30t_dataset(request, immutable_source_root=immutable_source_root)`. Derive the two relative paths exactly:

```python
target_policy_relative = PurePosixPath(
    contract.target_policy_path.relative_to(contract.source_checkout_path).as_posix()
)
quota_config_relative = PurePosixPath(
    contract.quota_config_path.relative_to(contract.source_checkout_path).as_posix()
)
if target_policy_relative != PurePosixPath(
    "examples/dataset/qwen3_30ba3b_thinking_ptv23_swe_heavy_target_v1.json"
) or quota_config_relative != PurePosixPath(
    "examples/dataset/qwen3_30ba3b_thinking_ptv23_swe_heavy_700k_v1.json"
):
    raise ValueError("Q30 target policy or quota config path is not canonical")
```

The adapter must complete before parent archive/runtime staging or keeper-derived paths are created. Keep the 700K row scan as a bounded defense-in-depth check, but no comment may describe it as the authoritative semantic verifier.

- [ ] **Step 6: Run provenance GREEN tests and commit**

Run all focused continuation tests plus the adapter suite. Expected: every rehashed substitution fails, and the canonical fixture calls the adapter exactly once. Commit:

```bash
git add tools/launcher/common/specdec/q30t_ptv23_continuation.py \
  tools/launcher/tests/test_q30t_ptv23_continuation.py
git commit -s -m "fix: authenticate complete Q30 dataset provenance"
```

### Task 7: Extract the immutable builder dependency closure in the runner

**Files:**
- Modify: `tools/launcher/common/specdec/run_q30t_ptv23_continuation.sbatch:33-255`
- Modify: `tools/launcher/tests/test_q30t_ptv23_controller.py:34-256`

**Interfaces:**
- Consumes: `q30t_builder_dependency_paths()` exact inventory and descriptor `builder_source_commit`/builder hash.
- Produces: a bootstrap tree containing exact regular Git blobs for the controller, publisher, dataset adapter, builder, policy/config, schemas, tokenizer verifier, and digest helpers.

- [ ] **Step 1: Write RED runner extraction tests**

Extend `test_spooled_runner_bootstraps_replay_from_descriptor_commit_not_mutable_checkout` so the committed builder writes `committed-builder` and a post-spool checkout mutation writes `mutable-builder`. The replay adapter must observe only `committed-builder`. Add replace-ref, symlink, executable-mode, duplicate-record, missing-dependency, and extra-dependency tests.

- [ ] **Step 2: Run runner bootstrap tests and verify RED**

Run:

```bash
PYTHONPATH=tools/launcher python3 -m pytest -q \
  tools/launcher/tests/test_q30t_ptv23_controller.py \
  -k 'spooled_runner and builder'
```

Expected: FAIL because current `git ls-tree` extracts only `tools/launcher/common/specdec`.

- [ ] **Step 3: Request only exact Git paths under a sterile Git boundary**

Replace the broad subtree query with an explicit array matching Task 4's dependency inventory. Invoke Git as:

```bash
/usr/bin/env -i LC_ALL=C PATH=/usr/bin:/bin GIT_CONFIG_GLOBAL=/dev/null \
  GIT_CONFIG_NOSYSTEM=1 GIT_NO_REPLACE_OBJECTS=1 \
  /usr/bin/git -c core.fsmonitor=false -c core.hooksPath=/dev/null \
  -C "$source_checkout" ls-tree -z "$bootstrap_commit" -- "${bootstrap_paths[@]}"
```

For every record require regular blob mode `100644` or `100755`, exact requested path membership, uniqueness, object ID shape, blob OID recomputation, and complete set equality before any module execution.

- [ ] **Step 4: Execute descriptor replay under `env -i`**

Replace ambient invocation with absolute `/usr/bin/env -i`, exact `LC_ALL=C`, `PATH=/usr/bin:/bin`, and isolated `/usr/bin/python3.12 -I`. Before altering `sys.path` or importing controller code, require:

```python
if sys.version_info < (3, 12):
    raise SystemExit("Q30 controller requires Python 3.12+")
```

Insert only the extracted launcher root in the bootstrap script; dataset roots are introduced later by the adapter from the extracted tree. Assert the runner contains `/usr/bin/python3.12 -I`, contains no unversioned production interpreter invocation, and hides unknown `Q30_*`, Python, Git, credential, and scheduler variables.

Pass `--immutable-source-root "$bootstrap_root"` to the descriptor replay CLI. Add the same option to `_main`, validate it as a canonical no-follow directory, and forward it to `load_q30_launch_descriptor`. The submitter intentionally omits the option so local preflight uses Task 4's temporary Git-object materializer; neither path imports builder code from the mutable checkout.

- [ ] **Step 5: Extend runner required environment names**

Require nonempty authenticated values for builder path/hash/commit, policy/config paths/hashes, inventory, historical, five held-outs, capacity, manifest/completion, and runtime qualification. Do not allow shell defaults. Validate each hash/commit with anchored lowercase regexes before keeper start.

- [ ] **Step 6: Run shell and hostile GREEN gates**

Run:

```bash
bash -n tools/launcher/common/specdec/run_q30t_ptv23_continuation.sbatch
shellcheck tools/launcher/common/specdec/run_q30t_ptv23_continuation.sbatch
PYTHONPATH=tools/launcher python3 -m pytest -q \
  tools/launcher/tests/test_q30t_ptv23_controller.py \
  -k 'spooled_runner or fixed_trusted_bootstrap or authenticated_positional'
git diff --check
```

Expected: clean shell/static gates and all hostile runner tests pass.

- [ ] **Step 7: Review and commit immutable extraction separately**

Confirm the diff contains no `eval`, mutable checkout import, unfiltered `env`, wildcard extraction, recursive deletion, or fallback to local refs. Commit:

```bash
git add tools/launcher/common/specdec/run_q30t_ptv23_continuation.sbatch \
  tools/launcher/tests/test_q30t_ptv23_controller.py
git commit -s -m "feat: replay Q30 dataset verifier from Git objects"
```

### Task 8: Gate canary and full execution on authenticated data/runtime evidence

**Files:**
- Modify: `tools/launcher/common/specdec/q30t_ptv23_continuation.py:1054-1144`
- Modify: `tools/launcher/common/specdec/run_q30t_ptv23_continuation.sbatch:255-780`
- Modify: `tools/launcher/common/specdec/submit_q30t_ptv23_continuation.sh:24-80`
- Test: `tools/launcher/tests/test_q30t_ptv23_continuation.py`
- Test: `tools/launcher/tests/test_q30t_ptv23_controller.py`

**Interfaces:**
- Consumes: fully authenticated contract/descriptor and the Runtime plan's independently reviewed qualification path/hash.
- Produces: canary/full launch environments that cannot start keeper/training until all required evidence passes.

- [ ] **Step 1: Write RED phase-order tests**

Instrument mocked adapter, keeper, and trainer markers. Assert exact order:

```python
assert phases == [
    "descriptor-replay",
    "source-authentication",
    "dataset-authoritative-replay",
    "runtime-qualification-authentication",
    "keeper-start",
    "keeper-readiness",
    "training",
    "exposure-reconciliation",
    "completion-publication",
]
```

For every dataset/runtime failure, assert keeper and training markers do not exist. For keeper death during training, probe, or reconciliation, assert no final receipt.

- [ ] **Step 2: Add RED canary/full admission tests**

Canary requires stage `canary`, method-bound canary run name, exact 20 steps, and no preexisting canary receipt. Full requires stage `full`, a reviewed method-matching canary receipt, exact 1,368 steps, exposure 700,000, final GBS96, and no preexisting full receipt. Cross-method canary receipts fail.

- [ ] **Step 3: Export only authenticated evidence to the runner**

Extend `build_q30_training_command`'s `MappingProxyType` with explicit variables for all new fields. Encode held-outs as five separate `HELD_OUT_SPEED_PATH/SHA256` through `HELD_OUT_TOOL_PATH/SHA256` pairs, not a delimiter-parsed aggregate. Include `BUILDER_SOURCE_COMMIT`, `DATASET_BUILDER_SHA256`, `TARGET_POLICY_SHA256`, `QUOTA_CONFIG_SHA256`, `SOURCE_INVENTORY_SHA256`, `CAPACITY_RECEIPT_SHA256`, `DATASET_COMPLETION_RECEIPT_SHA256`, and `RUNTIME_QUALIFICATION_RECEIPT_SHA256`.

```python
def _dataset_trust_environment(contract: Q30TContinuationContract) -> dict[str, str]:
    held_out_by_name = {
        name: (path, file_sha256) for name, path, file_sha256 in contract.held_out_receipts
    }
    return {
        "BUILDER_SOURCE_COMMIT": contract.builder_source_commit,
        "DATASET_BUILDER_SHA256": contract.dataset_builder_file_sha256,
        "TARGET_POLICY_SHA256": contract.target_policy_file_sha256,
        "QUOTA_CONFIG_SHA256": contract.quota_config_file_sha256,
        "SOURCE_INVENTORY_SHA256": contract.source_inventory_file_sha256,
        "CAPACITY_RECEIPT_SHA256": contract.capacity_receipt_file_sha256,
        "DATASET_COMPLETION_RECEIPT_SHA256": (
            contract.dataset_completion_receipt_file_sha256
        ),
        "RUNTIME_QUALIFICATION_RECEIPT_SHA256": (
            contract.runtime_qualification_receipt_file_sha256
        ),
        **{
            f"HELD_OUT_{name.upper()}_PATH": str(held_out_by_name[name][0])
            for name in ("speed", "math", "code", "swe", "tool")
        },
        **{
            f"HELD_OUT_{name.upper()}_SHA256": held_out_by_name[name][1]
            for name in ("speed", "math", "code", "swe", "tool")
        },
    }
```

Add `**_dataset_trust_environment(contract)` as the final mapping-unpack entry in the existing `MappingProxyType` constructor immediately before it is frozen; do not mutate a mapping proxy after construction.

- [ ] **Step 4: Authenticate canary evidence before full admission**

Add an approved canary receipt mapping keyed by method and parent receipt hash. Leave production mappings empty until the two independently replayed canary receipts exist. Unit tests monkeypatch exact fixture mappings; production full launch must fail with `ValueError("reviewed Q30 canary receipt is unavailable")` while mappings are empty.

```python
APPROVED_Q30T_CANARY_RECEIPT_BY_METHOD_AND_PARENT: Mapping[tuple[Method, str], str] = (
    MappingProxyType({})
)
```

- [ ] **Step 5: Preserve exact exposure behavior**

Run existing tests proving canary nominal GBS512 and 20 steps, and full steps 1-1,367 GBS512 plus final GBS96. Assert no dataset iterator cycling or padding flags appear in runner or training overrides.

- [ ] **Step 6: Run canary/full GREEN gates and commit**

Run:

```bash
PYTHONPATH=tools/launcher python3 -m pytest -q \
  tools/launcher/tests/test_q30t_ptv23_continuation.py \
  tools/launcher/tests/test_q30t_ptv23_controller.py \
  -k 'canary or full or phase or keeper'
bash -n tools/launcher/common/specdec/run_q30t_ptv23_continuation.sbatch \
  tools/launcher/common/specdec/submit_q30t_ptv23_continuation.sh
shellcheck tools/launcher/common/specdec/run_q30t_ptv23_continuation.sbatch \
  tools/launcher/common/specdec/submit_q30t_ptv23_continuation.sh
```

Expected: logic tests pass using reviewed fixtures; production full admission remains intentionally fail-closed until real canary approvals are inserted. Commit:

```bash
git add tools/launcher/common/specdec/q30t_ptv23_continuation.py \
  tools/launcher/common/specdec/run_q30t_ptv23_continuation.sbatch \
  tools/launcher/common/specdec/submit_q30t_ptv23_continuation.sh \
  tools/launcher/tests/test_q30t_ptv23_continuation.py \
  tools/launcher/tests/test_q30t_ptv23_controller.py
git commit -s -m "feat: gate Q30 canary and full continuation evidence"
```

### Task 9: Freeze builder dependency hashes only after all trust edits

**Files:**
- Modify: `tools/launcher/common/specdec/q30t_ptv23_continuation.py:42-65`
- Modify: `tools/launcher/common/specdec/run_q30t_ptv23_continuation.sbatch:33-255`
- Test: `tools/launcher/tests/test_q30t_ptv23_continuation.py`
- Test: `tools/launcher/tests/test_q30t_ptv23_controller.py`

**Interfaces:**
- Consumes: final reviewed bytes from Tasks 1-8.
- Produces: an exact non-stale builder identity, Git-object-bound controller execution, and a fail-closed production approval registry.

Do not start this task until the Data plan has frozen its final builder consumer changes in the integration branch. If the Data plan changes `build_qwen4b_ptv23_complement.py` after this task, discard this hash-only commit and repeat Task 9 from RED.

- [ ] **Step 1: Write RED source-hash consistency tests**

Add tests that compute hashes from current bytes rather than copying an earlier value:

```python
def test_q30_controller_builder_pin_matches_reviewed_builder_bytes() -> None:
    assert continuation.TRUSTED_Q30T_DATASET_BUILDER_SHA256 == hashlib.sha256(
        BUILDER_PATH.read_bytes()
    ).hexdigest()


def test_q30_runner_authenticates_controller_by_commit_blob_oid() -> None:
    runner = RUNNER_PATH.read_text()
    assert 'relative_text == "tools/launcher/common/specdec/q30t_ptv23_continuation.py"' in runner
    assert "observed_oid != raw_oid" in runner
    assert "GIT_NO_REPLACE_OBJECTS=1" in runner
```

The tests must fail if the old pre-integration builder hash `8f452ca20bd22ddaabb5709288a42ba83d697e5593eca55c01504d69d9d82394` or Q4 hash `ad42d169b31679dc014f68a54abe5297755d2ae688436c8b9bd597e62c155f53` remains.

- [ ] **Step 2: Run hash tests and verify RED**

Expected: FAIL because the Q30 builder pin is absent or stale; the runner test passes only when controller execution remains bound by the authenticated commit blob OID rather than a self-referential embedded file hash.

- [ ] **Step 3: Compute and insert hashes from the frozen reviewed slice**

Run:

```bash
sha256sum examples/dataset/build_qwen4b_ptv23_complement.py \
  tools/launcher/common/specdec/q30t_ptv23_continuation.py \
  tools/launcher/common/specdec/q30t_dataset_replay.py \
  tools/launcher/common/specdec/q30t_tokenizer_receipt.py \
  tools/launcher/common/specdec/q30t_parent_receipt.py
```

Copy the command's lowercase 64-hex output directly into the corresponding reviewed constants in the same patch. Do not reuse hashes recorded before Tasks 1-8. Re-run `sha256sum` immediately after formatting; if any value differs, update the pin and repeat before review.

- [ ] **Step 4: Keep unavailable approval roots explicitly empty**

Assert these production roots remain empty until their separately reviewed artifacts exist: dataset completion receipts, runtime archive-to-tree map, runtime qualification receipts, and method-specific canary receipts. Tests must patch fixture approvals locally and assert unpatched production launch fails before keeper start.

- [ ] **Step 5: Run full Subproject B verification**

Run:

```bash
PYTHONPATH=tools/launcher python3 -m pytest -q \
  tools/launcher/tests/test_q30t_tokenizer_receipt.py \
  tools/launcher/tests/test_q30t_parent_receipt.py \
  tools/launcher/tests/test_q30t_dataset_replay.py \
  tools/launcher/tests/test_q30t_ptv23_continuation.py \
  tools/launcher/tests/test_q30t_ptv23_controller.py \
  tests/examples/dataset/test_q30t_swe_heavy_700k_config.py
python3 -m ruff format --check tools/launcher/common/specdec/q30t_*.py \
  examples/dataset/build_qwen4b_ptv23_complement.py tools/launcher/tests/test_q30t_*.py \
  tests/examples/dataset/test_q30t_swe_heavy_700k_config.py
python3 -m ruff check tools/launcher/common/specdec/q30t_*.py \
  examples/dataset/build_qwen4b_ptv23_complement.py tools/launcher/tests/test_q30t_*.py \
  tests/examples/dataset/test_q30t_swe_heavy_700k_config.py
python3 -m py_compile tools/launcher/common/specdec/q30t_tokenizer_receipt.py \
  tools/launcher/common/specdec/q30t_parent_receipt.py \
  tools/launcher/common/specdec/q30t_dataset_replay.py \
  tools/launcher/common/specdec/q30t_ptv23_continuation.py
bash -n tools/launcher/common/specdec/run_q30t_ptv23_continuation.sbatch \
  tools/launcher/common/specdec/submit_q30t_ptv23_continuation.sh
shellcheck tools/launcher/common/specdec/run_q30t_ptv23_continuation.sbatch \
  tools/launcher/common/specdec/submit_q30t_ptv23_continuation.sh
git diff --check
```

Expected: all tests and static gates pass; production canary/full remain blocked only by the explicitly empty external approval roots owned by Subprojects A/C.

- [ ] **Step 6: Run the independent hostile review gate**

Give a fresh reviewer the exact Subproject B diff and ask them to verify: no Q4 adapter; exact three receipt roots; immutable Git-object execution; all manifest/completion comparisons; held-out order; builder hash freshness; no foreign deletion; no keeper start on authentication failure. Resolve every critical/warning with a new RED-GREEN cycle before freezing hashes again.

- [ ] **Step 7: Commit the frozen trust slice**

```bash
git add tools/launcher/common/specdec/q30t_ptv23_continuation.py \
  tools/launcher/common/specdec/run_q30t_ptv23_continuation.sbatch \
  tools/launcher/tests/test_q30t_ptv23_continuation.py \
  tools/launcher/tests/test_q30t_ptv23_controller.py
git commit -s -m "chore: freeze Q30 controller trust hashes"
```

Record the signed commit SHA and all final file hashes in the review report. Pushing, `sbatch --test-only`, and job submission are separate authorized operational transitions, not part of this implementation plan.

## External-evidence admission boundary

The Controller plan is code-complete when Task 9 passes with external production roots still empty. After the Data plan independently replays the exact bundle and the Runtime plan independently replays runtime qualification, a separate minimal approval-only change may populate only:

- `APPROVED_Q30T_DATASET_COMPLETION_RECEIPT_FILE_SHA256S`;
- `APPROVED_Q30T_RUNTIME_TREE_SHA256_BY_ARCHIVE_SHA256`;
- `APPROVED_Q30T_PRODUCTION_RUNTIME_IMAGE_SHA256S` with the reviewed image `e7be53f2754097c88f7c801da92f6d94794ec4d78d9df937fcd315a6994297f0`;
- the exact runtime qualification whole-file set;
- later, method-specific reviewed canary receipt roots required for full launch.

That approval-only patch must include canonical/self-hash replay evidence, exact whole-file `sha256sum` output, negative one-nibble mutation tests, focused controller replay, fresh hostile review, and its own signed commit. It must not regenerate receipts, modify producer logic, submit jobs, or combine unrelated runtime/data/controller changes.
