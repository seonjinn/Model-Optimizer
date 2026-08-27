# Q30 Thinking 700K Data and Continuation Contract Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Produce a Qwen3-30B-A3B-Thinking-specific authenticated 700K dataset and two continuation contracts rooted in the completed Ptyche DFlash/DSpark B8 step-25391 parents.

**Architecture:** Keep one security-sensitive selection/replay implementation and move target-specific tokenizer and model requirements into immutable target policies. Extend the continuation contract with separate target and drafter-parent identities, exact historical-lineage reconciliation, fresh-state restoration, and an exact 700K schedule with a 96-example final batch.

**Tech Stack:** Python 3.13, pytest, SQLite, Hugging Face tokenizer receipts, canonical JSON/SHA-256 receipts, Ruff, Pyright, Bash.

**Spec:** `docs/superpowers/specs/2026-08-26-q30t-ptv23-complement-700k-design.md`

## Global Constraints

- Dataset quotas total exactly 700,000 and retain the eight category counts in the spec.
- Q30 validation uses `Qwen/Qwen3-30B-A3B-Thinking-2507` and an externally authenticated exact revision, tokenizer snapshot, and training template.
- The prior 1.3M occurrence receipt and exact named `speed`, `math`, `code`, `swe`, and `tool` held-out receipts are mandatory.
- DFlash and DSpark parents must be B8 Thinking step 25391 and must prove identical historical lineage.
- Continuation restores weights and ModelOpt state only; optimizer, scheduler, trainer step, and RNG are fresh.
- Full exposure is 1,368 optimizer steps: 1,367 × 512 plus one final global batch of 96, for exactly 700,000 unique examples.
- Production trust roots remain empty and fail closed until their actual receipts are independently reviewed.
- Every commit is signed and includes a DCO sign-off.

---

### Task 1: Make the dataset builder target-policy aware

**Files:**
- Create: `examples/dataset/ptv23_complement_target_policy.py`
- Create: `examples/dataset/qwen3_4b_ptv23_target_v1.json`
- Create: `examples/dataset/qwen3_30ba3b_thinking_ptv23_target_v1.json`
- Create: `examples/dataset/qwen3_30ba3b_thinking_ptv23_complement_700k_v1.json`
- Create: `examples/dataset/qwen3_30ba3b_thinking_ptv23_complement_sources_v1.json`
- Modify: `examples/dataset/build_qwen4b_ptv23_complement.py`
- Test: `tests/examples/dataset/test_build_qwen4b_ptv23_complement.py`
- Create: `tests/examples/dataset/test_ptv23_complement_target_policy.py`

**Interfaces:**
- Produces: `TargetTokenizerPolicy`, `load_target_policy(path: Path, expected_sha256: str) -> TargetTokenizerPolicy`.
- Consumes: existing source inventory, historical exclusion, held-out union, selection, publication, and replay functions.

- [ ] **Step 1: Write failing policy-isolation tests**

```python
def test_q30_policy_rejects_q4_tokenizer_receipt(tmp_path: Path) -> None:
    policy = load_target_policy(Q30_POLICY_PATH, expected_sha256=Q30_POLICY_SHA256)
    receipt = canonical_tokenizer_receipt(repository="Qwen/Qwen3-4B")
    with pytest.raises(ComplementError, match="target tokenizer identity"):
        load_tokenizer_trust(write_receipt(tmp_path, receipt), expected_sha256=sha(receipt), policy=policy)


def test_target_policy_cannot_self_authorize_an_unknown_file(tmp_path: Path) -> None:
    forged = write_policy(tmp_path, repository="Qwen/forged")
    with pytest.raises(ComplementError, match="target policy is not approved"):
        load_target_policy(forged, expected_sha256=file_sha(forged))
```

- [ ] **Step 2: Run the tests and observe RED**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 ../q4-ptv2-ab-integration/.venv/bin/python -m pytest -q \
  tests/examples/dataset/test_ptv23_complement_target_policy.py
```

Expected: collection fails because the policy module and policy-aware tokenizer API do not exist.

- [ ] **Step 3: Implement the immutable policy boundary**

```python
@dataclass(frozen=True)
class TargetTokenizerPolicy:
    schema_version: str
    scientific_identity: str
    target_repository: str
    target_revision: str
    tokenizer_repository: str
    tokenizer_revision: str
    tokenizer_trust_schema: str
    training_sequence_length: int
    file_sha256: str


APPROVED_TARGET_POLICY_SHA256S: frozenset[str] = frozenset()
```

Parse canonical JSON through one no-follow regular descriptor. Require the expected file SHA to belong to `APPROVED_TARGET_POLICY_SHA256S`; initially include only the reviewed Q4 policy and leave the Q30 entry absent until Task 2 authenticates the real tokenizer receipt. Each policy binds its exact quota-config and source-requirements path/SHA. Pass the policy explicitly into config loading, tokenizer trust loading, selection replay, manifest creation, and verification. Preserve Q4 behavior through its exact checked-in policy rather than a mutable default.

- [ ] **Step 4: Run focused and existing builder tests**

```bash
PYTHONDONTWRITEBYTECODE=1 ../q4-ptv2-ab-integration/.venv/bin/python -m pytest -q \
  tests/examples/dataset/test_ptv23_complement_target_policy.py \
  tests/examples/dataset/test_build_qwen4b_ptv23_complement.py \
  tests/examples/dataset/test_trajectory_schema.py
```

Expected: all pass; unresolved Q30 approval roots fail before source selection.

- [ ] **Step 5: Commit**

```bash
git add examples/dataset/ptv23_complement_target_policy.py \
  examples/dataset/qwen3_4b_ptv23_target_v1.json \
  examples/dataset/qwen3_30ba3b_thinking_ptv23_target_v1.json \
  examples/dataset/qwen3_30ba3b_thinking_ptv23_complement_700k_v1.json \
  examples/dataset/qwen3_30ba3b_thinking_ptv23_complement_sources_v1.json \
  examples/dataset/build_qwen4b_ptv23_complement.py \
  tests/examples/dataset/test_build_qwen4b_ptv23_complement.py \
  tests/examples/dataset/test_ptv23_complement_target_policy.py
git commit -S -s -m "refactor(dataset): add target-bound complement policy"
```

### Task 2: Authenticate the Q30 Thinking tokenizer and rebuild/replay DATA

**Files:**
- Create: `tools/launcher/common/specdec/q30t_tokenizer_receipt.py`
- Create: `tools/launcher/tests/test_q30t_tokenizer_receipt.py`
- Modify: `examples/dataset/qwen3_30ba3b_thinking_ptv23_target_v1.json`
- Modify: `examples/dataset/build_qwen4b_ptv23_complement.py`

**Interfaces:**
- Produces: `build_q30t_tokenizer_receipt(snapshot: Path, repository: str, revision: str) -> bytes` and a reviewed receipt file SHA.
- Consumes: exact Q30 snapshot, official and training chat templates, special-token IDs, target-policy loader.

- [ ] **Step 1: Write failing tokenizer receipt tests**

```python
def test_q30t_receipt_binds_snapshot_template_and_special_tokens(tmp_path: Path) -> None:
    receipt = build_q30t_tokenizer_receipt(
        snapshot=make_q30_snapshot(tmp_path),
        repository="Qwen/Qwen3-30B-A3B-Thinking-2507",
        revision=Q30_REVISION,
    )
    parsed = json.loads(receipt)
    assert parsed["snapshot_tree_sha256"] == tree_sha(parsed["snapshot_path"])
    assert parsed["training_chat_template_sha256"] == expected_training_template_sha()


@pytest.mark.parametrize("field", ["revision", "snapshot_tree_sha256", "training_chat_template_sha256"])
def test_q30t_receipt_rejects_tampering(tmp_path: Path, field: str) -> None:
    with pytest.raises(ValueError):
        verify_tampered_receipt(tmp_path, field)
```

- [ ] **Step 2: Run RED**

```bash
PYTHONDONTWRITEBYTECODE=1 ../q4-ptv2-ab-integration/.venv/bin/python -m pytest -q \
  tools/launcher/tests/test_q30t_tokenizer_receipt.py
```

Expected: import failure for the missing receipt builder.

- [ ] **Step 3: Implement descriptor-stable receipt generation and verification**

Generate canonical JSON from a descriptor-stable snapshot walk. Bind repository, exact revision, snapshot tree SHA, official template SHA, training template SHA, special-token IDs, and Q30 target identity. Never accept caller-provided derived hashes without recomputation.

- [ ] **Step 4: Authenticate the actual snapshot before populating the allowlist**

Run the receipt builder on the staged Q30 Thinking snapshot. Verify the receipt twice from independent fresh reads. Update the checked-in target-policy SHA and `APPROVED_TARGET_POLICY_SHA256S` only after the receipt and target revision are independently reviewed. If no snapshot/receipt is available, retain the empty root and record the external blocker without inventing a digest.

- [ ] **Step 5: Rebuild and replay the exact 700K bundle under Q30**

Use the existing authenticated inventory, historical receipt, and five named held-out receipts. Require every selected row to pass the Q30 4,096-token gate. Compare physical UUID/category order with the Q4 selection; identical rows are allowed, but Q30 token evidence and the resulting manifest must be newly computed.

- [ ] **Step 6: Run focused tests and commit**

```bash
PYTHONDONTWRITEBYTECODE=1 ../q4-ptv2-ab-integration/.venv/bin/python -m pytest -q \
  tools/launcher/tests/test_q30t_tokenizer_receipt.py \
  tests/examples/dataset/test_ptv23_complement_target_policy.py \
  tests/examples/dataset/test_build_qwen4b_ptv23_complement.py
git add tools/launcher/common/specdec/q30t_tokenizer_receipt.py \
  tools/launcher/tests/test_q30t_tokenizer_receipt.py \
  examples/dataset/qwen3_30ba3b_thinking_ptv23_target_v1.json \
  examples/dataset/build_qwen4b_ptv23_complement.py
git commit -S -s -m "feat(dataset): authenticate Q30 Thinking tokenizer"
```

### Task 3: Create canonical Q30 parent receipts and reconcile lineage

**Files:**
- Create: `tools/launcher/common/specdec/q30t_parent_receipt.py`
- Create: `tools/launcher/tests/test_q30t_parent_receipt.py`
- Modify: `tools/launcher/common/specdec/qwen4b_ptv23_continuation.py`
- Modify: `tools/launcher/tests/test_qwen4b_ptv23_continuation.py`

**Interfaces:**
- Produces: `Q30TParentReceipt`, `load_q30t_parent_receipt(path: Path, expected_sha256: str) -> Q30TParentReceipt`, and `require_matching_historical_lineage(dflash, dspark) -> None`.
- Consumes: exact Ptyche milestone manifests and the two step-25391 parent directories named in the spec.

- [ ] **Step 1: Write failing parent-identity tests**

```python
@pytest.mark.parametrize("field,value", [
    ("variant", "Base"),
    ("block_size", 16),
    ("global_step", 14500),
    ("method", "DFlash2"),
])
def test_q30t_parent_rejects_wrong_identity(tmp_path: Path, field: str, value: object) -> None:
    with pytest.raises(ValueError, match="Q30 Thinking parent identity"):
        load_q30t_parent_receipt(forge_parent(tmp_path, field, value), expected_sha256=forged_sha(tmp_path))


def test_dflash_and_dspark_must_share_historical_lineage() -> None:
    with pytest.raises(ValueError, match="historical lineage"):
        require_matching_historical_lineage(parent("DFlash", history="a"), parent("DSpark", history="b"))
```

- [ ] **Step 2: Run RED**

```bash
PYTHONDONTWRITEBYTECODE=1 ../q4-ptv2-ab-integration/.venv/bin/python -m pytest -q \
  tools/launcher/tests/test_q30t_parent_receipt.py
```

- [ ] **Step 3: Implement the parent receipt schema**

```python
@dataclass(frozen=True)
class Q30TParentReceipt:
    method: Literal["DFlash", "DSpark"]
    target_repository: str
    target_revision: str
    target_tree_sha256: str
    variant: Literal["Thinking-2507"]
    block_size: Literal[8]
    global_step: Literal[25391]
    checkpoint_path: Path
    checkpoint_tree_sha256: str
    modelopt_state_sha256: str
    historical_receipt_file_sha256: str
    historical_ordered_prompt_uuids_sha256: str
    source_commit: str
    runtime_sha256: str
    receipt_file_sha256: str
```

Authenticate exact file set/type/size/hash through the milestone manifest. Reject the OCI partial and Lyris failed roots by identity and missing completion evidence, not by string matching.

- [ ] **Step 4: Generate and independently verify both Ptyche receipts**

Generate receipts only from the two exact roots in the spec. Require matching historical receipt and ordered UUID hashes. Add both final receipt file SHAs to the immutable parent allowlist only after rereview.

- [ ] **Step 5: Run tests and commit**

```bash
PYTHONDONTWRITEBYTECODE=1 ../q4-ptv2-ab-integration/.venv/bin/python -m pytest -q \
  tools/launcher/tests/test_q30t_parent_receipt.py \
  tools/launcher/tests/test_qwen4b_ptv23_continuation.py
git add tools/launcher/common/specdec/q30t_parent_receipt.py \
  tools/launcher/tests/test_q30t_parent_receipt.py \
  tools/launcher/common/specdec/qwen4b_ptv23_continuation.py \
  tools/launcher/tests/test_qwen4b_ptv23_continuation.py
git commit -S -s -m "feat(specdec): authenticate Q30 Thinking parents"
```

### Task 4: Extend the continuation contract for target, method, and partial exposure

**Files:**
- Create: `tools/launcher/common/specdec/q30t_ptv23_continuation.py`
- Create: `tools/launcher/tests/test_q30t_ptv23_continuation.py`
- Modify: `tools/launcher/common/specdec/qwen4b_ptv23_continuation.py`

**Interfaces:**
- Produces: target-aware `ContinuationContract`, `ExposureSchedule`, and Q30-specific CLI wrapper.
- Consumes: authenticated target, tokenizer, dataset, parent receipt, trainer, and runtime identities.

- [ ] **Step 1: Write failing schedule and restore tests**

```python
def test_q30_full_schedule_consumes_exact_700k_with_partial_final_batch() -> None:
    schedule = ExposureSchedule(full_steps=1368, nominal_global_batch_size=512, final_global_batch_size=96)
    assert schedule.consumed_examples == 700_000
    assert schedule.examples_for_step(1368) == 96


def test_q30_command_separates_target_and_drafter_parent() -> None:
    command = build_q30_training_command(contract())
    assert command_value(command, "--target-model") == "Qwen/Qwen3-30B-A3B-Thinking-2507"
    assert command_value(command, "--drafter-parent") == str(PARENT_ROOT)
    assert command_values(command, "--restore") == ["weights", "modelopt-state"]
    assert "optimizer" not in command_values(command, "--restore")
```

- [ ] **Step 2: Run RED**

```bash
PYTHONDONTWRITEBYTECODE=1 ../q4-ptv2-ab-integration/.venv/bin/python -m pytest -q \
  tools/launcher/tests/test_q30t_ptv23_continuation.py
```

- [ ] **Step 3: Implement exact exposure types and validation**

```python
@dataclass(frozen=True)
class ExposureSchedule:
    consumed_examples: int = 700_000
    full_steps: int = 1_368
    nominal_global_batch_size: int = 512
    final_global_batch_size: int = 96

    def validate(self, trainer_ranks: int) -> None:
        if self.full_steps <= 20 or self.final_global_batch_size % trainer_ranks:
            raise ValueError("invalid exact exposure schedule")
        if (self.full_steps - 1) * self.nominal_global_batch_size + self.final_global_batch_size != self.consumed_examples:
            raise ValueError("exposure does not equal 700K")
```

Persist all fields in make/load/evidence/receipt paths. Remove every use of `max_steps * global_batch_size` as the dataset exposure identity.

- [ ] **Step 4: Implement target-aware fresh-state command construction**

Bind Thinking target, DFlash/DSpark method, B8, 16 nodes, 32 trainer ranks, target TP, method recipe, no-cycle dataset semantics, and final-partial-batch behavior. Add a behavioral fake trainer that fails if optimizer/scheduler/RNG files from the parent are consumed.

- [ ] **Step 5: Run tests and commit**

```bash
PYTHONDONTWRITEBYTECODE=1 ../q4-ptv2-ab-integration/.venv/bin/python -m pytest -q \
  tools/launcher/tests/test_q30t_ptv23_continuation.py \
  tools/launcher/tests/test_qwen4b_ptv23_continuation.py
git add tools/launcher/common/specdec/q30t_ptv23_continuation.py \
  tools/launcher/common/specdec/qwen4b_ptv23_continuation.py \
  tools/launcher/tests/test_q30t_ptv23_continuation.py
git commit -S -s -m "feat(specdec): add exact Q30T continuation contract"
```

### Task 5: Freeze and independently review the data/contract track

**Files:**
- Modify: `.superpowers/sdd/2026-08-26-q30t-ptv23-complement-700k/task-report.md`

**Interfaces:**
- Produces: frozen file scope, content hashes, test evidence, and external blocker list for the runtime track.

- [ ] **Step 1: Run full relevant tests**

```bash
PYTHONDONTWRITEBYTECODE=1 ../q4-ptv2-ab-integration/.venv/bin/python -m pytest -q \
  tests/examples/dataset/test_build_qwen4b_ptv23_complement.py \
  tests/examples/dataset/test_ptv23_complement_target_policy.py \
  tests/examples/dataset/test_trajectory_schema.py \
  tools/launcher/tests/test_q30t_tokenizer_receipt.py \
  tools/launcher/tests/test_q30t_parent_receipt.py \
  tools/launcher/tests/test_q30t_ptv23_continuation.py \
  tools/launcher/tests/test_qwen4b_ptv23_continuation.py
```

- [ ] **Step 2: Run static gates**

```bash
../q4-ptv2-ab-integration/.venv/bin/ruff check --no-fix examples/dataset tools/launcher/common/specdec tools/launcher/tests
../q4-ptv2-ab-integration/.venv/bin/ruff format --check examples/dataset tools/launcher/common/specdec tools/launcher/tests
pyright examples/dataset/build_qwen4b_ptv23_complement.py \
  examples/dataset/ptv23_complement_target_policy.py \
  tools/launcher/common/specdec/q30t_tokenizer_receipt.py \
  tools/launcher/common/specdec/q30t_parent_receipt.py \
  tools/launcher/common/specdec/q30t_ptv23_continuation.py
git diff --check
```

- [ ] **Step 3: Record hashes and request hostile review**

Record exact path+content SHA-256s and external trust roots in the task report. Freeze the worktree and dispatch an independent read-only review focused on tokenizer isolation, parent swapping, lineage, fresh-state restore, and partial-batch accounting.

- [ ] **Step 4: Commit the verified report**

```bash
git add .superpowers/sdd/2026-08-26-q30t-ptv23-complement-700k/task-report.md
git commit -S -s -m "docs(specdec): record Q30T data contract evidence"
```
