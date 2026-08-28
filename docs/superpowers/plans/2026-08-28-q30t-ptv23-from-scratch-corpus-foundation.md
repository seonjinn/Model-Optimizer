# Q30 Thinking PTV2/PTV3 From-Scratch Corpus Foundation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and independently replay the authenticated PTV2/PTV3 prompt reserves for the `H-native`, `H-synth`, and Math/SWE/agentic `M-synth` arms without loading any historical drafter checkpoint.

**Architecture:** Add a Q30 from-scratch policy boundary and focused modules for generation-prefix identity, source authentication, row adaptation, held-out exclusion, and deterministic lane reserves. Reuse existing canonical JSON, trajectory validation, tokenizer receipt, and atomic-publication primitives, but keep the continuation contract and parent-checkpoint loaders outside the dependency graph. This plan ends at immutable prompt reserves and capacity evidence; target generation, assistant-token views, drafter training, and evaluation are separate follow-on plans.

**Tech Stack:** Python 3.12+, dataclasses, canonical JSON and SHA-256, SQLite, PyArrow, pytest, Ruff, Pyright, Bash, SLURM/Pyxis.

**Spec:** `docs/superpowers/specs/2026-08-28-q30t-ptv23-from-scratch-drafter-study-design.md`

## Global Constraints

- The target is exactly `Qwen/Qwen3-30B-A3B-Thinking-2507` at revision `144afc2f379b542fdd4e85a1fcd5e1f79112d95d`; the reviewed tokenizer receipt, chat-template digest, special-token IDs, and reasoning mode are mandatory inputs.
- “From scratch” applies to drafter parameters. No code in this plan may import, load, or accept `q30t_parent_receipt`, `AuthenticatedParentCheckpoint`, `dflash_init_checkpoint`, optimizer state, scheduler state, trainer step, RNG state, or historical W&B identity.
- `H-native` and `H-synth` contain the exact historical 1,300,000 prompt occurrences in the authenticated historical order. `H-native` retains the source response; `H-synth` removes it before generation.
- `M-synth` lane shares are exactly agentless SWE 15%, interactive SWE 10%, general tool 10%, Math 25%, Code 15%, STEM 15%, instruction 5%, and multilingual 5%.
- The `M-synth` canary row counts are exactly 1,536, 1,024, 1,024, 2,560, 1,536, 1,536, 512, and 512 in that lane order, totaling 10,240.
- Multilingual token capacity is divided equally among DE, JA, ES, FR, and IT. Its 512-row canary uses deterministic largest-remainder order `DE=103`, `JA=103`, `ES=102`, `FR=102`, and `IT=102`. A short language or lane blocks the reserve; it is never backfilled from another lane.
- The canonical prompt UUID is computed before target generation and binds the immutable message prefix, tool declarations, source repository/revision/file/split/row, reasoning mode, and response lane. It excludes target output and selection rank.
- The five held-out names and order are exactly `speed`, `math`, `code`, `swe`, and `tool`. This plan consumes only an independently approved `q30t-held-out-sources-v2` bundle; state `P`, missing catalogs, empty catalogs, or producer self-approval fail closed.
- Candidate selection uses SHA-256 ordering over policy, seed, lane, source identity, source row, and prompt UUID. No source iteration order, filesystem order, or worker count may change selected bytes.
- Reserve publication is immutable and no-clobber. Foreign existing paths are preserved and rejected; cleanup never recursively removes an untrusted tree.
- High-churn staging uses node-local `/raid/scratch`; durable source snapshots, reserves, receipts, and bounded logs use `/lustre`.
- Every remote submission runs `sbatch --test-only`, uses a clean signed and pushed commit, queries only the exact job or the user's jobs, and monitors a newly running job for five minutes.
- Each implementation task follows RED-GREEN tests, runs its focused regression set, receives review, and commits with `git commit -S -s`.
- Do not modify `uv.lock`.

---

## Cross-Plan Handoffs

- Shared prerequisite: `docs/superpowers/specs/2026-08-28-q30t-conservative-heldout-v2-design.md` must produce an independently approved V2 bundle. This plan owns only its typed consumer and collision application; it does not duplicate the held-out producer.
- This plan produces `FromScratchStudyPolicy`, `GenerationPromptIdentity`, `AuthenticatedSourceRegistry`, `PromptReserve`, and `ReservePublicationReceipt` for the synthesis plan.
- The synthesis plan will consume the immutable H/M prompt reserves, generate or execute Q30 responses, and produce accepted assistant-token capacity plus nested 256M/1B/4B schedules.
- The training plan will consume only published corpus-view digests and will separately merge the reviewed DFlash2/speed/DSpark integration branch. It must not add a parent-checkpoint input.
- The evaluation/operations plan will consume exported checkpoints and the same five held-out catalogs; it owns 1-node, 2-node, and 16-node GB200 gates.

## File Structure

### Policy and identity

- Create `examples/dataset/q30t_from_scratch_policy.py`: typed policy loader, exact lanes, shares, canary counts, and downstream gate declarations.
- Create `examples/dataset/qwen3_30ba3b_thinking_ptv23_from_scratch_v1.json`: checked-in scientific policy with empty production approval roots until observed evidence is reviewed.
- Create `examples/dataset/qwen3_30ba3b_thinking_ptv23_from_scratch_sources_v1.json`: exact source-family requirements and field-map identities.
- Create `examples/dataset/q30t_generation_prompt_identity.py`: provenance-bound immutable-prefix identity without changing legacy `specdec_identity.prompt_uuid` semantics.

### Source and candidate normalization

- Create `examples/dataset/q30t_from_scratch_sources.py`: authenticate source-registry bytes, staged files, schemas, license expressions, and approved-use classifications.
- Create `examples/dataset/q30t_from_scratch_candidates.py`: map physical PTV2/PTV3 rows into response-stripped generation prompts or source-native controls.
- Create `examples/dataset/q30t_from_scratch_agentic.py`: cut valid next-action prefixes and compare canonical target/source calls without attaching mismatched tool results.
- Create `examples/dataset/q30t_from_scratch_exclusions.py`: load the approved V2 held-out union and apply UUID plus exact containment exclusions.

### Selection and publication

- Create `examples/dataset/q30t_from_scratch_reserve.py`: deterministic per-lane ordering, canary selection, historical 1.3M reconciliation, capacity reports, and immutable reserve publication.
- Create `examples/dataset/build_q30t_from_scratch_reserve.py`: CLI composition root; no scientific logic.
- Create `tools/launcher/common/specdec/run_q30t_from_scratch_reserve.sbatch`: immutable Git CPU builder/replay boundary.
- Create `tools/launcher/common/specdec/submit_q30t_from_scratch_reserve.sh`: clean-source and `--test-only` submission boundary.

### Tests

- Create one focused test module per Python module under `tests/examples/dataset/`.
- Create `tools/launcher/tests/test_q30t_from_scratch_reserve_runner.py` for the SLURM runner and submitter.

---

### Task 1: Freeze the from-scratch scientific policy

**Files:**

- Create: `examples/dataset/q30t_from_scratch_policy.py`
- Create: `examples/dataset/qwen3_30ba3b_thinking_ptv23_from_scratch_v1.json`
- Create: `examples/dataset/qwen3_30ba3b_thinking_ptv23_from_scratch_sources_v1.json`
- Create: `tests/examples/dataset/test_q30t_from_scratch_policy.py`

**Interfaces:**

- Produces: `LanePolicy`, `TokenGate`, `FromScratchStudyPolicy`, and `load_from_scratch_policy(path: Path, *, expected_sha256: str) -> FromScratchStudyPolicy`.
- Produces constants: `LANE_ORDER`, `CANARY_ROWS`, `TOKEN_SHARES`, and `APPROVED_FROM_SCRATCH_POLICY_SHA256S`.
- Invariant: a policy file cannot authorize itself; the allowlist initially contains no new production digest.

- [ ] **Step 1: Write failing exact-arithmetic and isolation tests**

```python
def test_policy_freezes_lane_order_and_arithmetic() -> None:
    policy = load_fixture_policy_for_test()
    assert tuple(lane.name for lane in policy.lanes) == LANE_ORDER
    assert sum(lane.canary_rows for lane in policy.lanes) == 10_240
    assert sum((lane.token_share for lane in policy.lanes), Fraction()) == Fraction(1, 1)
    assert policy.minimum_unique_prompts == 1_300_000
    assert tuple(gate.assistant_loss_tokens for gate in policy.gates) == (
        256_000_000,
        1_000_000_000,
        4_000_000_000,
    )


def test_policy_has_no_parent_checkpoint_field() -> None:
    payload = json.loads(POLICY_PATH.read_text())
    forbidden = {"parent_checkpoint", "dflash_init_checkpoint", "resume_from_checkpoint"}
    assert forbidden.isdisjoint(payload)
```

- [ ] **Step 2: Run the policy tests and verify RED**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest -q \
  tests/examples/dataset/test_q30t_from_scratch_policy.py
```

Expected: collection fails because the module and policy files do not exist.

- [ ] **Step 3: Implement the typed exact-rational policy boundary**

```python
@dataclass(frozen=True)
class LanePolicy:
    name: str
    token_numerator: int
    token_denominator: int
    canary_rows: int
    languages: tuple[str, ...]

    @property
    def token_share(self) -> Fraction:
        return Fraction(self.token_numerator, self.token_denominator)


@dataclass(frozen=True)
class TokenGate:
    name: Literal["pilot", "primary", "scale"]
    assistant_loss_tokens: int


@dataclass(frozen=True)
class FromScratchStudyPolicy:
    schema_version: Literal["q30t-ptv23-from-scratch-policy-v1"]
    scientific_identity: str
    target_repository: str
    target_revision: str
    minimum_unique_prompts: int
    seed: int
    lanes: tuple[LanePolicy, ...]
    gates: tuple[TokenGate, ...]
    source_requirements_path: str
    source_requirements_sha256: str
    file_sha256: str
```

Load one stable, no-follow, single-link regular file; require canonical JSON bytes, exact fields, exact lane order, rational sum one, canary sum 10,240, and source-requirements path/SHA binding. Production loading requires `expected_sha256` in `APPROVED_FROM_SCRATCH_POLICY_SHA256S`; expose a test-only parser for fixtures rather than weakening the production loader.

- [ ] **Step 4: Run focused GREEN tests and static checks**

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest -q \
  tests/examples/dataset/test_q30t_from_scratch_policy.py
uv run ruff check examples/dataset/q30t_from_scratch_policy.py \
  tests/examples/dataset/test_q30t_from_scratch_policy.py
uv run pyright examples/dataset/q30t_from_scratch_policy.py
```

Expected: all pass; the production loader rejects the checked-in unapproved digest.

- [ ] **Step 5: Commit the policy slice**

```bash
git add examples/dataset/q30t_from_scratch_policy.py \
  examples/dataset/qwen3_30ba3b_thinking_ptv23_from_scratch_v1.json \
  examples/dataset/qwen3_30ba3b_thinking_ptv23_from_scratch_sources_v1.json \
  tests/examples/dataset/test_q30t_from_scratch_policy.py
git commit -S -s -m "feat(dataset): define Q30 from-scratch study policy"
```

### Task 2: Add immutable generation-prefix identity

**Files:**

- Create: `examples/dataset/q30t_generation_prompt_identity.py`
- Create: `tests/examples/dataset/test_q30t_generation_prompt_identity.py`
- Test: `tests/examples/dataset/test_specdec_identity.py`

**Interfaces:**

- Consumes: `canonical_json`, `sha256_bytes`, and `normalize_storage_fields`.
- Produces: `PromptProvenance`, `GenerationPromptIdentity`, and `generation_prompt_identity(messages: object, tools: object, provenance: PromptProvenance) -> GenerationPromptIdentity`.
- Invariant: prior assistant actions and resolved tool results in the immutable prefix affect identity; the response being generated and storage metadata do not.

- [ ] **Step 1: Write failing prompt-identity tests**

```python
def test_agentic_prefix_binds_prior_action_and_tool_result() -> None:
    prefix = [
        {"role": "user", "content": "inspect the repo"},
        {"role": "assistant", "tool_calls": [call("1", "shell", {"cmd": "ls"})]},
        {"role": "tool", "tool_call_id": "1", "name": "shell", "content": "a.py"},
    ]
    identity = generation_prompt_identity(prefix, [shell_tool()], provenance())
    changed = deepcopy(prefix)
    changed[-1]["content"] = "b.py"
    assert identity.prompt_uuid != generation_prompt_identity(
        changed, [shell_tool()], provenance()
    ).prompt_uuid


def test_identity_excludes_generated_response_and_selection_rank() -> None:
    base = generation_prompt_identity(
        [{"role": "user", "content": "solve"}], [], provenance()
    )
    row = {"identity": base, "generated_response": "answer", "selection_rank": 7}
    assert row["identity"].prompt_uuid == base.prompt_uuid
```

- [ ] **Step 2: Run the new and legacy identity tests and verify RED**

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest -q \
  tests/examples/dataset/test_q30t_generation_prompt_identity.py \
  tests/examples/dataset/test_specdec_identity.py
```

Expected: the new module import fails; legacy tests remain unchanged.

- [ ] **Step 3: Implement provenance-bound canonical identity**

```python
@dataclass(frozen=True)
class PromptProvenance:
    source_repository: str
    source_revision: str
    source_file_sha256: str
    source_split: str
    source_row_index: int
    reasoning_mode: Literal["reasoning_on", "reasoning_off"]
    response_lane: str


@dataclass(frozen=True)
class GenerationPromptIdentity:
    prompt_uuid: str
    canonical_bytes: bytes
    provenance: PromptProvenance


def generation_prompt_identity(
    messages: object,
    tools: object,
    provenance: PromptProvenance,
) -> GenerationPromptIdentity:
    canonical_bytes = canonical_json(
        {
            "messages": _validated_immutable_prefix(messages),
            "tools": _validated_tools(tools),
            "provenance": asdict(provenance),
        }
    )
    return GenerationPromptIdentity(
        prompt_uuid=sha256_bytes(canonical_bytes),
        canonical_bytes=canonical_bytes,
        provenance=provenance,
    )
```

Validate message roles and tool-call referential integrity by calling the existing trajectory primitives on completed transactions. Reject an unresolved prior call. Keep `specdec_identity.prompt_uuid` unchanged for the continuation experiment.

- [ ] **Step 4: Run focused and trajectory GREEN tests**

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest -q \
  tests/examples/dataset/test_q30t_generation_prompt_identity.py \
  tests/examples/dataset/test_specdec_identity.py \
  tests/examples/dataset/test_trajectory_schema.py
```

Expected: all pass, including UUID collision and storage-field cases.

- [ ] **Step 5: Commit the identity slice**

```bash
git add examples/dataset/q30t_generation_prompt_identity.py \
  tests/examples/dataset/test_q30t_generation_prompt_identity.py
git commit -S -s -m "feat(dataset): bind Q30 generation prompt identity"
```

### Task 3: Authenticate the PTV2/PTV3 source registry

**Files:**

- Create: `examples/dataset/q30t_from_scratch_sources.py`
- Create: `tests/examples/dataset/test_q30t_from_scratch_sources.py`
- Consume: `examples/dataset/qwen3_30ba3b_thinking_ptv23_from_scratch_sources_v1.json`
- Test: `tests/examples/dataset/test_stage_ptv23_sources.py`

**Interfaces:**

- Consumes: `SourceInventory`, `SourceIdentity`, and existing descriptor-stable file hashing from `stage_ptv23_sources.py`.
- Produces: `SourceRegistryEntry`, `AuthenticatedSourceRegistry`, and `load_authenticated_source_registry(requirements_path: Path, *, expected_sha256: str, stage_receipts: Mapping[str, Path]) -> AuthenticatedSourceRegistry`.
- Invariant: every physical file and row schema is assigned to exactly one declared lane or is explicitly rejected.

- [ ] **Step 1: Write failing source-authentication tests**

```python
def test_registry_binds_every_physical_file_and_license(tmp_path: Path) -> None:
    registry = load_test_registry(tmp_path, entries=[source_entry("math")])
    entry = registry.entries[0]
    assert entry.file_sha256 == sha256_file(entry.path)
    assert entry.license_expression
    assert entry.approved_use == "specdec-drafter-training"


@pytest.mark.parametrize(
    "mutation",
    ["floating_revision", "unmapped_file", "row_schema_mismatch", "missing_license"],
)
def test_registry_fails_closed_on_unbound_source_facts(
    tmp_path: Path, mutation: str
) -> None:
    with pytest.raises(SourceRegistryError):
        load_mutated_registry(tmp_path, mutation)
```

- [ ] **Step 2: Run source tests and verify RED**

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest -q \
  tests/examples/dataset/test_q30t_from_scratch_sources.py \
  tests/examples/dataset/test_stage_ptv23_sources.py
```

Expected: the new source module import fails.

- [ ] **Step 3: Implement exact source entries and registry reconciliation**

```python
@dataclass(frozen=True)
class SourceRegistryEntry:
    source_id: str
    repository: str
    revision: str
    configuration: str
    split: str
    relative_path: str
    bytes: int
    file_sha256: str
    row_count: int
    row_schema_sha256: str
    license_expression: str
    approved_use: Literal["specdec-drafter-training"]
    lane: str
    adapter: str


@dataclass(frozen=True)
class AuthenticatedSourceRegistry:
    entries: tuple[SourceRegistryEntry, ...]
    requirements_sha256: str
    stage_receipt_sha256s: tuple[str, ...]
    registry_sha256: str
```

Require 40-character revisions, relative regular-file paths, exact size/SHA/count/schema reconciliation, unique `(repository, configuration, split, relative_path)` identities, approved licenses, and complete staged-file coverage. Do not infer a lane from a repository name.

- [ ] **Step 4: Run source, staging, and schema-observation GREEN tests**

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest -q \
  tests/examples/dataset/test_q30t_from_scratch_sources.py \
  tests/examples/dataset/test_stage_ptv23_sources.py \
  tests/examples/dataset/test_observe_q30t_ptv23_row_schemas.py
```

Expected: all pass; source order and worker count do not change the registry digest.

- [ ] **Step 5: Commit the source-registry slice**

```bash
git add examples/dataset/q30t_from_scratch_sources.py \
  tests/examples/dataset/test_q30t_from_scratch_sources.py
git commit -S -s -m "feat(dataset): authenticate Q30 from-scratch sources"
```

### Task 4: Normalize non-agentic and historical-control rows

**Files:**

- Create: `examples/dataset/q30t_from_scratch_candidates.py`
- Create: `tests/examples/dataset/test_q30t_from_scratch_candidates.py`
- Test: `tests/examples/dataset/test_audit_ptv2_baseline.py`

**Interfaces:**

- Consumes: `AuthenticatedSourceRegistry`, `GenerationPromptIdentity`, and the authenticated historical `BaselineAudit`.
- Produces: `PromptCandidate`, `adapt_target_synthesis_row(row: Mapping[str, object], source: SourceRegistryEntry) -> PromptCandidate`, and `adapt_historical_occurrence(row: Mapping[str, object], source: SourceRegistryEntry, *, expected_prompt_uuid: str) -> tuple[PromptCandidate, PromptCandidate]` returning H-native then H-synth.
- Invariant: only H-native contains `source_assistant`; H-synth and M-synth request bytes cannot contain the removed answer.

- [ ] **Step 1: Write failing response-separation and lineage tests**

```python
def test_h_native_and_h_synth_share_prompt_identity_but_not_response() -> None:
    native, synth = adapt_historical_occurrence(historical_row(), source_entry(), audit_row())
    assert native.identity == synth.identity
    assert native.source_assistant == {"role": "assistant", "content": "source answer"}
    assert synth.source_assistant is None
    assert b"source answer" not in synth.request_bytes


def test_m_synth_strips_source_completion_before_identity_and_request() -> None:
    candidate = adapt_target_synthesis_row(row_with_secret_answer(), source_entry())
    assert candidate.arm == "M-synth"
    assert candidate.source_assistant is None
    assert b"secret answer" not in candidate.identity.canonical_bytes
    assert b"secret answer" not in candidate.request_bytes
```

- [ ] **Step 2: Run candidate tests and verify RED**

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest -q \
  tests/examples/dataset/test_q30t_from_scratch_candidates.py \
  tests/examples/dataset/test_audit_ptv2_baseline.py
```

Expected: the new candidate module import fails.

- [ ] **Step 3: Implement the arm-safe candidate contract**

```python
@dataclass(frozen=True)
class PromptCandidate:
    arm: Literal["H-native", "H-synth", "M-synth"]
    lane: str
    language: str
    identity: GenerationPromptIdentity
    request_bytes: bytes
    source_assistant: dict[str, object] | None
    normalized_row_sha256: str
```

Parse row fields only through the registry-named adapter. For H rows, reconcile each source occurrence with the ordered historical audit before returning either arm. Serialize synthesis request bytes from the identity prefix and tools only. Store H-native responses in an authenticated sidecar so no generic generation request can accidentally include them.

- [ ] **Step 4: Run candidate, baseline, and identity GREEN tests**

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest -q \
  tests/examples/dataset/test_q30t_from_scratch_candidates.py \
  tests/examples/dataset/test_audit_ptv2_baseline.py \
  tests/examples/dataset/test_q30t_generation_prompt_identity.py
```

Expected: all pass; exact historical order and 1,300,000 count are required.

- [ ] **Step 5: Commit the row-adapter slice**

```bash
git add examples/dataset/q30t_from_scratch_candidates.py \
  tests/examples/dataset/test_q30t_from_scratch_candidates.py
git commit -S -s -m "feat(dataset): construct Q30 from-scratch prompt arms"
```

### Task 5: Construct safe agentic next-action prefixes

**Files:**

- Create: `examples/dataset/q30t_from_scratch_agentic.py`
- Create: `tests/examples/dataset/test_q30t_from_scratch_agentic.py`
- Modify: `examples/dataset/trajectory_schema.py`
- Modify: `tests/examples/dataset/test_trajectory_schema.py`

**Interfaces:**

- Consumes: `validate_trajectory` and `PromptCandidate`.
- Produces: `CanonicalToolCall`, `AgenticGenerationPrefix`, `canonical_tool_call(call: object) -> CanonicalToolCall`, `agentic_generation_prefixes(candidate: PromptCandidate, source_trajectory: Mapping[str, object], *, executable_environment: bool) -> tuple[AgenticGenerationPrefix, ...]`, `recorded_result_matches(target_call: CanonicalToolCall, source_call: CanonicalToolCall) -> bool`, and `accept_target_action(target_call: CanonicalToolCall, source_call: CanonicalToolCall, *, executable: bool) -> Literal["execute", "reuse-recorded"]`.
- Invariant: JSON-string and JSON-object arguments canonicalize identically; a divergent non-executable action rejects that continuation.

- [ ] **Step 1: Write failing tool-call and prefix tests**

```python
def test_json_string_and_object_arguments_canonicalize_identically() -> None:
    left = call("1", "shell", '{"cmd":"ls","timeout":10}')
    right = call("x", "shell", {"timeout": 10, "cmd": "ls"})
    assert canonical_tool_call(left).function_bytes == canonical_tool_call(right).function_bytes


def test_divergent_call_cannot_reuse_recorded_tool_result() -> None:
    source = canonical_tool_call(call("1", "shell", {"cmd": "ls"}))
    target = canonical_tool_call(call("7", "shell", {"cmd": "rm file"}))
    assert not recorded_result_matches(target, source)
    with pytest.raises(AgenticPrefixError, match="executable environment"):
        accept_target_action(target, source, executable=False)
```

- [ ] **Step 2: Run agentic and trajectory tests and verify RED**

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest -q \
  tests/examples/dataset/test_q30t_from_scratch_agentic.py \
  tests/examples/dataset/test_trajectory_schema.py
```

Expected: imports or object-argument cases fail because the existing validator accepts only JSON strings.

- [ ] **Step 3: Implement canonical calls and transaction-safe prefixes**

```python
@dataclass(frozen=True)
class CanonicalToolCall:
    call_id: str
    function_name: str
    arguments_bytes: bytes

    @property
    def function_bytes(self) -> bytes:
        return canonical_json(
            {"name": self.function_name, "arguments": json.loads(self.arguments_bytes)}
        )


@dataclass(frozen=True)
class AgenticGenerationPrefix:
    candidate: PromptCandidate
    source_next_call: CanonicalToolCall | None
    recorded_result: dict[str, object] | None
    executable_environment: bool
```

Extend the shared trajectory validator to accept a mapping or JSON string for `function.arguments`, always emitting a canonical JSON string. Slice only after complete prior transactions. Preserve the recorded next call/result as comparison evidence outside generation request bytes.

- [ ] **Step 4: Run all trajectory and candidate GREEN tests**

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest -q \
  tests/examples/dataset/test_q30t_from_scratch_agentic.py \
  tests/examples/dataset/test_trajectory_schema.py \
  tests/examples/dataset/test_build_specdec_inventory.py \
  tests/examples/dataset/test_q30t_from_scratch_candidates.py
```

Expected: all pass; legacy replay serialization remains byte-compatible for already canonical JSON-string inputs.

- [ ] **Step 5: Commit the agentic-prefix slice**

```bash
git add examples/dataset/q30t_from_scratch_agentic.py \
  examples/dataset/trajectory_schema.py \
  tests/examples/dataset/test_q30t_from_scratch_agentic.py \
  tests/examples/dataset/test_trajectory_schema.py
git commit -S -s -m "feat(dataset): build safe Q30 agentic prefixes"
```

### Task 6: Consume the approved held-out V2 union

**Files:**

- Create: `examples/dataset/q30t_from_scratch_exclusions.py`
- Create: `tests/examples/dataset/test_q30t_from_scratch_exclusions.py`
- Consume: held-out V2 loader and sidecars produced from `docs/superpowers/specs/2026-08-28-q30t-conservative-heldout-v2-design.md`

**Interfaces:**

- Consumes: canonical V2 receipt and index sidecars produced according to `docs/superpowers/specs/2026-08-28-q30t-conservative-heldout-v2-design.md`.
- Produces: `HeldoutV2Bundle`, `load_approved_heldout_v2(root: Path, *, expected_receipt_sha256: str) -> HeldoutV2Bundle`, `CandidateExclusion`, `ExclusionLedger`, and `apply_heldout_union(candidate: PromptCandidate, heldout: HeldoutV2Bundle) -> CandidateExclusion | None`.
- Invariant: all five catalogs are approved and nonempty before the first candidate is admitted.

- [ ] **Step 1: Write failing five-catalog and containment tests**

```python
def test_consumer_requires_five_approved_nonempty_catalogs() -> None:
    with pytest.raises(FromScratchExclusionError, match="speed,math,code,swe,tool"):
        load_from_scratch_heldout_bundle(bundle_without("tool"))


def test_answer_or_patch_containment_excludes_a_longer_candidate() -> None:
    heldout = heldout_bundle(answer_fragments=["return sorted(values)"])
    candidate = prompt_candidate("Implement it.\nreturn sorted(values)\nThen explain.")
    exclusion = apply_heldout_union(candidate, heldout)
    assert exclusion is not None
    assert exclusion.reason == "heldout_content_containment"
    assert exclusion.catalog == "code"
```

- [ ] **Step 2: Run exclusion tests and verify RED**

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest -q \
  tests/examples/dataset/test_q30t_from_scratch_exclusions.py
```

Expected: the consumer module is absent. If the shared held-out loader is not yet implemented or approved, record that exact cross-plan blocker and do not add a permissive fallback.

- [ ] **Step 3: Implement the typed consumer and ledger**

```python
@dataclass(frozen=True)
class HeldoutV2Bundle:
    receipt_sha256: str
    catalog_names: tuple[Literal["speed", "math", "code", "swe", "tool"], ...]
    prompt_uuids: frozenset[str]
    exact_content_sha256s: frozenset[str]
    containment_index: Mapping[str, tuple[str, ...]]
    approval_identity_sha256: str


@dataclass(frozen=True)
class CandidateExclusion:
    prompt_uuid: str
    catalog: Literal["speed", "math", "code", "swe", "tool"]
    reason: Literal["prompt_uuid", "exact_content", "heldout_content_containment"]
    matched_identity: str


@dataclass(frozen=True)
class ExclusionLedger:
    heldout_receipt_sha256: str
    admitted_prompt_ids_sha256: str
    excluded: tuple[CandidateExclusion, ...]
    counts_by_catalog: Mapping[str, int]
```

Authenticate the consumer receipt and content-index sidecars before use. Apply prompt UUID, normalized full-field identity, then deterministic containment rules. Record only stable identities and bounded snippets; do not copy benchmark answers into logs.

- [ ] **Step 4: Run exclusion and held-out producer GREEN tests**

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest -q \
  tests/examples/dataset/test_q30t_from_scratch_exclusions.py \
  tests/examples/dataset/test_build_q30t_heldout_receipts.py
```

Expected: all pass against an approved fixture; state `P` and empty components fail closed.

- [ ] **Step 5: Commit the held-out consumer slice**

```bash
git add examples/dataset/q30t_from_scratch_exclusions.py \
  tests/examples/dataset/test_q30t_from_scratch_exclusions.py
git commit -S -s -m "feat(dataset): exclude Q30 held-out V2 content"
```

### Task 7: Build deterministic H and M prompt reserves

**Files:**

- Create: `examples/dataset/q30t_from_scratch_reserve.py`
- Create: `tests/examples/dataset/test_q30t_from_scratch_reserve.py`
- Test: `tests/examples/dataset/test_build_specdec_inventory.py`

**Interfaces:**

- Consumes: `FromScratchStudyPolicy`, authenticated candidates, `ExclusionLedger`, and the historical audit.
- Produces: `ReserveRow`, `LaneCapacity`, `PromptReserve`, `ReserveBundle`, `build_historical_reserves(candidates: Iterable[tuple[PromptCandidate, PromptCandidate]], *, historical_audit_sha256: str) -> tuple[PromptReserve, PromptReserve]`, and `build_m_synth_reserve(policy: FromScratchStudyPolicy, candidates: Iterable[PromptCandidate], exclusion_ledger: ExclusionLedger, *, workers: int = 1) -> PromptReserve`.
- Invariant: source iteration order and worker count cannot change reserve bytes; no padding, cycling, duplication, or cross-lane redistribution exists.

- [ ] **Step 1: Write failing deterministic-selection and capacity tests**

```python
def test_m_reserve_canary_is_exact_and_order_independent() -> None:
    first = build_m_synth_reserve(policy(), candidates(), exclusions(), workers=1)
    second = build_m_synth_reserve(
        policy(), reversed(candidates()), exclusions(), workers=8
    )
    assert first.canary_bytes == second.canary_bytes
    assert first.canary_counts == {
        "agentless_swe": 1536,
        "interactive_swe": 1024,
        "general_tool": 1024,
        "math": 2560,
        "code": 1536,
        "stem": 1536,
        "instruction": 512,
        "multilingual": 512,
    }


def test_short_lane_emits_capacity_blocker_without_redistribution() -> None:
    available = candidates_without("interactive_swe")
    reserve = build_m_synth_reserve(policy(), available, exclusions())
    assert reserve.publishable is False
    blocker = next(item for item in reserve.blockers if item.lane == "interactive_swe")
    assert blocker.required_canary_rows == 1024
    assert reserve.count("general_tool") == sum(row.lane == "general_tool" for row in available)


def test_m_reserve_requires_the_policy_minimum_unique_prompts() -> None:
    small_policy = replace(policy(), minimum_unique_prompts=13)
    reserve = build_m_synth_reserve(
        small_policy, candidates(total_unique=12), exclusions()
    )
    assert reserve.publishable is False
    blocker = next(item for item in reserve.blockers if item.lane == "__all__")
    assert blocker.required_unique_rows == 13
```

- [ ] **Step 2: Run reserve tests and verify RED**

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest -q \
  tests/examples/dataset/test_q30t_from_scratch_reserve.py
```

Expected: the reserve module is absent.

- [ ] **Step 3: Implement rank keys, per-lane reserves, and blockers**

```python
@dataclass(frozen=True)
class ReserveRow:
    rank_sha256: str
    prompt_uuid: str
    lane: str
    language: str
    request_sha256: str
    candidate: PromptCandidate


@dataclass(frozen=True)
class LaneCapacity:
    lane: str
    language: str
    eligible_rows: int
    excluded_rows: int
    required_canary_rows: int
    required_unique_rows: int = 0


@dataclass(frozen=True)
class PromptReserve:
    arm: Literal["H-native", "H-synth", "M-synth"]
    rows: tuple[ReserveRow, ...]
    canary_rows: tuple[ReserveRow, ...]
    capacities: tuple[LaneCapacity, ...]
    blockers: tuple[LaneCapacity, ...]
    reserve_sha256: str

    @property
    def publishable(self) -> bool:
        return not self.blockers

    @property
    def canary_bytes(self) -> bytes:
        return canonical_json([row.prompt_uuid for row in self.canary_rows])

    @property
    def canary_counts(self) -> Mapping[str, int]:
        return Counter(row.lane for row in self.canary_rows)

    def count(self, lane: str) -> int:
        return sum(row.lane == lane for row in self.rows)


@dataclass(frozen=True)
class ReserveBundle:
    h_native: PromptReserve
    h_synth: PromptReserve
    m_synth: PromptReserve
    exclusion_ledger: ExclusionLedger


def reserve_rank(policy_sha256: str, seed: int, candidate: PromptCandidate) -> str:
    provenance = candidate.identity.provenance
    return sha256_bytes(
        canonical_json(
            [
                policy_sha256,
                seed,
                candidate.lane,
                provenance.source_repository,
                provenance.source_revision,
                provenance.source_file_sha256,
                provenance.source_row_index,
                candidate.identity.prompt_uuid,
            ]
        )
    )
```

Sort M-synth independently per lane by `(rank_sha256, prompt_uuid)`. Deduplicate its prompt UUID and normalized row identity before admission, and require at least 1,300,000 unique eligible prompts across lanes. For multilingual, enforce the exact per-language canary allocation in the global constraints. H reserves do not deduplicate historical occurrences: they retain the exact authenticated order, require exactly 1,300,000 occurrences, and report both occurrence and unique-prompt counts.

- [ ] **Step 4: Run reserve and inventory GREEN tests**

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest -q \
  tests/examples/dataset/test_q30t_from_scratch_reserve.py \
  tests/examples/dataset/test_build_specdec_inventory.py \
  tests/examples/dataset/test_q30t_from_scratch_candidates.py
```

Expected: all pass; canary arithmetic is exact and shuffled inputs reproduce identical bytes.

- [ ] **Step 5: Commit the deterministic reserve slice**

```bash
git add examples/dataset/q30t_from_scratch_reserve.py \
  tests/examples/dataset/test_q30t_from_scratch_reserve.py
git commit -S -s -m "feat(dataset): select Q30 from-scratch prompt reserves"
```

### Task 8: Publish and replay immutable prompt reserves

**Files:**

- Modify: `examples/dataset/q30t_from_scratch_reserve.py`
- Modify: `tests/examples/dataset/test_q30t_from_scratch_reserve.py`
- Test: `tests/examples/dataset/test_specdec_publication.py`

**Interfaces:**

- Produces: `ReservePublicationReceipt`, `publish_prompt_reserves(bundle: ReserveBundle, destination: Path) -> ReservePublicationReceipt`, and `replay_prompt_reserves(root: Path, receipt_sha256: str) -> ReservePublicationReceipt`.
- Invariant: publication uses a job-unique partial directory, descriptor-stable rehash, fsync, and rename-no-replace into an absent destination.

- [ ] **Step 1: Write failing no-clobber and byte-replay tests**

```python
def test_publication_preserves_foreign_destination(tmp_path: Path) -> None:
    destination = tmp_path / "reserve-v1"
    destination.mkdir()
    (destination / "foreign").write_text("keep")
    with pytest.raises(ReservePublicationError, match="already exists"):
        publish_prompt_reserves(bundle(), destination)
    assert (destination / "foreign").read_text() == "keep"


def test_clean_replay_reproduces_every_payload_digest(tmp_path: Path) -> None:
    receipt = publish_prompt_reserves(bundle(), tmp_path / "first")
    replayed = replay_prompt_reserves(tmp_path / "first", receipt.receipt_sha256)
    assert replayed.payload_files == receipt.payload_files
    assert replayed.reserve_identity_sha256 == receipt.reserve_identity_sha256
```

- [ ] **Step 2: Run publication tests and verify RED**

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest -q \
  tests/examples/dataset/test_q30t_from_scratch_reserve.py -k 'publication or replay' \
  tests/examples/dataset/test_specdec_publication.py
```

Expected: reserve publication functions do not exist.

- [ ] **Step 3: Implement bounded atomic publication**

```python
@dataclass(frozen=True)
class ReservePublicationReceipt:
    schema_version: Literal["q30t-from-scratch-prompt-reserve-receipt-v1"]
    policy_sha256: str
    source_registry_sha256: str
    heldout_receipt_sha256: str
    historical_audit_sha256: str
    reserve_identity_sha256: str
    payload_files: tuple[SourceFile, ...]
    arm_counts: Mapping[str, int]
    lane_counts: Mapping[str, int]
    receipt_sha256: str
```

Write H-native response sidecars, H-synth requests, M-synth requests, canary views, exclusion ledger, capacity report, and manifest as canonical bounded shards. Reuse the tested descriptor-relative publication helpers rather than pathname-recursive cleanup. Replay revalidates every source input and rebuilds selected bytes into a fresh destination.

- [ ] **Step 4: Run publication, reserve, and recovery GREEN tests**

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest -q \
  tests/examples/dataset/test_q30t_from_scratch_reserve.py \
  tests/examples/dataset/test_specdec_publication.py
```

Expected: all pass, including crash injection and foreign-partial recovery cases.

- [ ] **Step 5: Commit the publication slice**

```bash
git add examples/dataset/q30t_from_scratch_reserve.py \
  tests/examples/dataset/test_q30t_from_scratch_reserve.py
git commit -S -s -m "feat(dataset): publish Q30 prompt reserves atomically"
```

### Task 9: Add the local CLI and immutable SLURM boundary

**Files:**

- Create: `examples/dataset/build_q30t_from_scratch_reserve.py`
- Create: `tests/examples/dataset/test_build_q30t_from_scratch_reserve.py`
- Create: `tools/launcher/common/specdec/run_q30t_from_scratch_reserve.sbatch`
- Create: `tools/launcher/common/specdec/submit_q30t_from_scratch_reserve.sh`
- Create: `tools/launcher/tests/test_q30t_from_scratch_reserve_runner.py`

**Interfaces:**

- Consumes: all Task 1-8 typed loaders and builders.
- Produces CLI operations: `audit`, `build`, and `replay`.
- Invariant: CLI arguments carry expected digests for every policy, source, tokenizer, held-out, historical, runtime, and output identity; no path alone confers trust.

- [ ] **Step 1: Write failing CLI, runner, and submitter tests**

```python
def test_build_command_requires_every_trust_root() -> None:
    result = run_cli(["build", "--policy", "policy.json"])
    assert result.returncode != 0
    for option in (
        "--policy-sha256",
        "--source-registry-sha256",
        "--heldout-receipt-sha256",
        "--historical-audit-sha256",
        "--tokenizer-receipt-sha256",
    ):
        assert option in result.stderr


def test_submitter_runs_test_only_before_real_submission() -> None:
    script = SUBMITTER.read_text()
    assert script.index("sbatch --test-only") < script.index("sbatch --parsable")
    assert "git diff --quiet" in script
    assert "git diff --cached --quiet" in script
```

- [ ] **Step 2: Run CLI and launcher tests and verify RED**

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest -q \
  tests/examples/dataset/test_build_q30t_from_scratch_reserve.py \
  tools/launcher/tests/test_q30t_from_scratch_reserve_runner.py
```

Expected: the CLI, runner, and submitter do not exist.

- [ ] **Step 3: Implement a thin CLI and fail-closed runner**

```python
def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    inputs = load_authenticated_inputs(args)
    if args.operation == "audit":
        write_capacity_candidate(inputs, args.output)
    elif args.operation == "build":
        publish_prompt_reserves(build_reserves(inputs), args.output)
    else:
        replay_prompt_reserves(args.input, args.receipt_sha256)
    return 0
```

The SBATCH script checks out the exact pushed commit into node-local `/raid/scratch`, verifies `HEAD`, stages bounded source inputs locally, runs the CLI, and copies only immutable final artifacts to `/lustre`. The submitter rejects a dirty/unpushed commit, runs `sbatch --test-only`, then submits once with explicit account/partition/profile arguments.

- [ ] **Step 4: Run the full corpus-foundation verification set**

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest -q \
  tests/examples/dataset/test_q30t_from_scratch_policy.py \
  tests/examples/dataset/test_q30t_generation_prompt_identity.py \
  tests/examples/dataset/test_q30t_from_scratch_sources.py \
  tests/examples/dataset/test_q30t_from_scratch_candidates.py \
  tests/examples/dataset/test_q30t_from_scratch_agentic.py \
  tests/examples/dataset/test_q30t_from_scratch_exclusions.py \
  tests/examples/dataset/test_q30t_from_scratch_reserve.py \
  tests/examples/dataset/test_build_q30t_from_scratch_reserve.py \
  tools/launcher/tests/test_q30t_from_scratch_reserve_runner.py
uvx pre-commit run --from-ref HEAD~9 --to-ref HEAD
```

Expected: all tests and hooks pass; no test or command imports a parent-checkpoint or continuation module.

- [ ] **Step 5: Commit the executable boundary**

```bash
git add examples/dataset/build_q30t_from_scratch_reserve.py \
  tests/examples/dataset/test_build_q30t_from_scratch_reserve.py \
  tools/launcher/common/specdec/run_q30t_from_scratch_reserve.sbatch \
  tools/launcher/common/specdec/submit_q30t_from_scratch_reserve.sh \
  tools/launcher/tests/test_q30t_from_scratch_reserve_runner.py
git commit -S -s -m "feat(dataset): add Q30 prompt reserve workflow"
```

## Completion Gate

This plan is complete only when all focused tests and pre-commit hooks pass, every commit is signed and DCO-compliant, and a clean local replay reproduces fixture reserve bytes. Production execution remains blocked until the Q30 target/tokenizer policy, all source receipts, the historical audit, the held-out V2 bundle, and the runtime profile have independently reviewed approval roots.

The production output of this plan is an immutable prompt reserve, not a training corpus. It makes no claim about assistant-token capacity, domain learning, DFlash/DSpark/DFlash2 ranking, or rollout speed. Those claims require the synthesis/token-view, training, and evaluation plans.
