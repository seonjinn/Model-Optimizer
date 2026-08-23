# B-prime, C, and D Prompt Preparation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Prepare immutable B′ 700K and paired C/D 2M prompt views, target-specific response corpora, exact assistant-token exposure views, and fail-closed cutover/evaluation receipts without disturbing the drafter jobs that still train on the historical 1.3M PTV2 prefix.

**Architecture:** Extend the existing canonical audit, inventory, trajectory, selection, and transfer tools with a prompt-count policy layer. Freeze exact prompt UUID views before target regeneration, promote successful responses from a 20% per-cell reserve, then derive exact assistant-loss-token exposure views from the immutable response corpus. Keep source staging, replay validation, publication, cluster readiness, weight-only checkpoint adoption, training, evaluation, and report refresh as separately testable receipt boundaries.

**Tech Stack:** Python 3.12+, dataclasses and `TypedDict`, PyArrow/Parquet Zstd, Hugging Face `datasets` and `transformers`, YAML, Bash/SLURM/Pyxis, PDX/PBSS `rclone`, ModelOpt DFlash/DSpark training, Speculators and SPEED-Bench evaluators, pytest, Pyright, Ruff, ShellCheck, pre-commit.

**Spec:** `docs/superpowers/specs/2026-08-22-bprime-cd-prompt-preparation-design.md`

## Global Constraints

- The historical source is exactly 1,309,377 row occurrences, and the training prefix is exactly the first 1,300,000 occurrences in Hugging Face streaming order; the excluded tail is exactly 9,377 German occurrences.
- The historical prefix histogram is Chat 627,720, Code 175,000, Math 239,467, German 257,813, and zero for STEM, Japanese, Spanish, French, Italian, and explicit SWE/Agentic/Tool.
- The historical prefix contains exactly 931,363 unique canonical prompt UUIDs. Preserve all duplicate occurrences in lineage, but exclude the 931,363 unique UUIDs from new prompt views.
- PTV2 revision is exactly `5c89e01dd720ae0f4058445ed49c5fb68a03c76e`; every PTV2/PTV3 source also requires its exact file path, byte size, SHA-256, license expression, configuration, and split.
- B′ contains exactly 700,000 unique prompts after historical, held-out, and within-view exclusion: STEM 200,000; Japanese, Spanish, French, and Italian 125,000 each; German, Chat, Math, Code, and SWE zero.
- C and D each contain exactly 2,000,000 unique prompts: SWE/Agentic/Tool 600,000; Math 400,000; Code 200,000; STEM/Science 400,000; Multilingual 300,000; Instruction/Chat 100,000.
- D's 600,000 SWE/Agentic/Tool prompts are exactly 200,000 agentless-SWE target synthesis, 200,000 validated interactive-SWE replay, and 200,000 validated generic-agentic/tool replay.
- C and D share all eligible non-agentic prompt UUIDs and the same frozen context-bucket floors. Only D's replay lanes intentionally differ.
- Every populated prompt cell has at least 20% reserve after all UUID, held-out, source, schema, language, and context exclusions; a shortfall is fatal and never triggers borrowing or renormalization.
- Canonical prompt identity preserves roles, content, message order, top-level tools, tool calls, and call IDs. It removes only storage fields and hashes deterministic JSON with SHA-256.
- Held-out SPEED, Math, Code, SWE, multilingual, and tool/agentic UUID manifests are content-addressed inputs to selection identity.
- Target-synthesis lanes strip source completions and regenerate with an exact target, tokenizer, chat template, thinking mode, generation configuration, runtime digest, and retry identity.
- Replay preserves recorded schemas, calls, IDs, results, order, and reasoning. It never flattens a trajectory, invents execution, or manufactures a thinking-mode label.
- Context buckets derive from full serialized length: `le4k` through 4,096, `4k_16k` through 16,384, and `16k_32k` through 32,768. A 4K view that splits a tool transaction is ineligible.
- Prompt counts measure coverage. Scientific comparisons stop at exact 256M, 1B, and largest-common assistant-loss-token boundaries; only assistant-owned unmasked tokens count.
- Source, selection, response, tokenized corpus, exposure, transfer, cutover, canary, and evaluator receipts are immutable canonical JSON bound by SHA-256.
- Publication writes and verifies a job-unique sibling partial tree, `fsync`s it, and atomically renames into a previously absent final namespace. It never replaces an existing final artifact.
- Existing drafter jobs continue independently. Every B′/C/D family adopts one identical verified parent checkpoint by weights only, resets optimizer/scheduler/scaler/RNG/loader/step/W&B state, and writes a new experiment namespace.
- No production training submission occurs until source, prompt, response, exposure, publication, destination, checkpoint-adoption, GPU-canary, and evaluator-readiness gates all pass.
- Before submission, focused tests, Ruff, Pyright, Bash syntax, ShellCheck, pre-commit, manifest dry runs, and `sbatch --test-only` must pass. GPU canaries must show finite loss, a complete checkpoint, all allocated GPUs active, and evaluator output.

## File Map and Stable Interfaces

- `examples/dataset/audit_ptv2_baseline.py` — preserve occurrence lineage and emit the exact 931,363-UUID exclusion receipt.
- `examples/dataset/specdec_corpus_contracts.py` — typed canonical receipt, prompt-cell, selection, response, and exposure contracts.
- `examples/dataset/specdec_identity.py` — canonical UUID and historical/held-out/candidate exclusion.
- `examples/dataset/bprime_cd_policy.py` — load and validate the exact B′/C/D prompt-count policy.
- `examples/dataset/bprime_cd_prompt_policy.yaml` — seed, exact counts, reserve factor, languages, lanes, context policy, and exposure milestones.
- `examples/dataset/stage_ptv23_sources.py` — validate and stage pinned source files into a source inventory.
- `examples/dataset/bprime_cd_sources.json` — exact revisions, licenses, splits, files, bytes, and SHA-256 values approved for production use.
- `examples/dataset/trajectory_schema.py` — canonical interactive-SWE and generic-tool replay validation and quarantine reason codes.
- `examples/dataset/build_specdec_inventory.py` — full-context tokenization, 4K derivation, domain/lane labels, and per-cell capacity.
- `examples/dataset/select_bprime_cd_prompts.py` — reserve selection, exact B′ and C/D prompt selection, paired UUID proofs, and context floors.
- `examples/dataset/promote_synthesis_reserve.py` — validate target outputs and deterministically promote successful reserve rows.
- `examples/dataset/build_assistant_token_views.py` — exact 256M/1B/common token views and one-pass receipts.
- `examples/dataset/specdec_publication.py` — verified resumable shard writing and no-replace atomic publication.
- `examples/dataset/ptv23_builder_readiness.py` — aggregate source, count, reserve, replay, generation, exposure, and publication blockers.
- `tools/launcher/common/specdec/run_ptv23_source_stage.sbatch` — CPU source staging and digest validation.
- `tools/launcher/common/specdec/run_bprime_cd_builder.sbatch` — CPU prompt selection/publication and bounded GPU synthesis entrypoint.
- `tools/launcher/common/specdec/bprime_cd_study_manifest.py` — destination, cutover, training, canary, and evaluator identity checks.
- `tools/launcher/common/specdec/run_bprime_cd_canary.sbatch` — bounded response/training/evaluator canary.
- `tools/launcher/common/specdec/submit_bprime_cd_study.sh` — test-only and dependency-safe canary submission.
- `examples/speculative_decoding/checkpoint_adoption.py` — weight-only checkpoint validation and reset receipt.
- `modelopt/torch/speculative/assistant_token_budget.py` — distributed exact-token stopping and resume fingerprint.
- `tools/launcher/common/specdec/collect_bprime_cd_results.py` — validate evaluator bundles and emit canonical report data.
- `reports/qwen3-4b-hybrid-study/render_status.py` — render concise HTML only from validated report data.
- `reports/qwen3-4b-hybrid-study/data/bprime_cd_progress.json` — machine-readable page input.
- `reports/qwen3-4b-hybrid-study/index.html` — concise human-readable progress and results page.

Type ownership is fixed: `bprime_cd_policy.py` defines `PromptCell`,
`ArmPolicy`, and `PromptPolicy`; `stage_ptv23_sources.py` defines
`SourceIdentity` and `SourceInventory`; `build_specdec_inventory.py` defines
`CandidateInventory`; `select_bprime_cd_prompts.py` defines `PromptView` and
`PromptViewBundle`; `promote_synthesis_reserve.py` defines
`GenerationIdentity`, `GenerationAttempt`, and `ResponseCorpus`;
`build_assistant_token_views.py` defines `ExposureView`;
`specdec_publication.py` defines `CorpusBundle` and `PublicationReceipt`;
`ptv23_builder_readiness.py` defines `ReadinessInputs` and `ReadinessReceipt`;
`bprime_cd_study_manifest.py` defines `ExperimentIdentity`;
`checkpoint_adoption.py` defines `AdoptionReceipt`; and
`collect_bprime_cd_results.py` defines `ResultInputs` and `ProgressReport`.

The implementation keeps these signatures stable across tasks:

- `canonical_json(value: object) -> bytes`
- `prompt_uuid(messages: list[dict[str, object]], tools: list[dict[str, object]] | None) -> str`
- `load_prompt_policy(path: Path) -> PromptPolicy`
- `load_source_inventory(path: Path) -> SourceInventory`
- `build_candidate_inventory(sources: SourceInventory, exclusions: ExclusionIndex, tokenizer: object) -> CandidateInventory`
- `validate_trajectory(example: dict[str, object], source: SourceIdentity) -> ValidatedTrajectory`
- `select_prompt_views(inventory: CandidateInventory, policy: PromptPolicy) -> PromptViewBundle`
- `promote_responses(view: PromptView, attempts: Iterable[GenerationAttempt], identity: GenerationIdentity) -> ResponseCorpus`
- `build_exposure_view(corpus: ResponseCorpus, target_tokens: int) -> ExposureView`
- `publish_bundle(bundle: CorpusBundle, destination: Path, job_id: str) -> PublicationReceipt`
- `assess_readiness(inputs: ReadinessInputs) -> ReadinessReceipt`
- `adopt_weights(parent: Path, model: torch.nn.Module, identity: ExperimentIdentity) -> AdoptionReceipt`
- `collect_results(inputs: ResultInputs) -> ProgressReport`

---

### Task 1: Preserve the historical occurrence and exclusion contract

**Files:**
- Modify: `examples/dataset/specdec_corpus_contracts.py`
- Modify: `examples/dataset/audit_ptv2_baseline.py`
- Modify: `tests/examples/dataset/test_audit_ptv2_baseline.py`
- Modify: `tests/examples/dataset/test_specdec_identity.py`

**Interfaces:**
- Consumes: the 26 resolved historical shards and exact PTV2 revision.
- Produces: `BaselineAudit.occurrence_count`, `BaselineAudit.unique_prompt_count`, ordered occurrence digest, and unique UUID exclusion digest for all later tasks.

- [ ] **Step 1: Add RED tests for occurrence-versus-UUID accounting.** Build a small fixture with duplicated canonical prompts and assert that the receipt preserves all occurrences while the exclusion file contains each UUID once. Add a production expectation test for `1_300_000`, `931_363`, and the exact split histogram.

```python
def test_audit_preserves_occurrences_and_deduplicates_exclusions(tmp_path: Path) -> None:
    audit = audit_baseline(duplicate_occurrence_fixture(tmp_path))
    assert audit.occurrence_count == 4
    assert audit.unique_prompt_count == 3
    assert len(audit.occurrence_prompt_ids) == 4
    assert len(audit.exclusion_prompt_ids) == 3
```

- [ ] **Step 2: Run RED.**

Run: `PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest -q tests/examples/dataset/test_audit_ptv2_baseline.py tests/examples/dataset/test_specdec_identity.py`

Expected: the new fields and exclusion serialization assertions fail.

- [ ] **Step 3: Extend the typed receipt and audit.** Add explicit occurrence and unique counts, keep duplicate occurrence IDs in source order, sort the unique exclusion set for its independent digest, and reject any production receipt that does not match the exact boundary, histogram, and unique count.

```python
@dataclass(frozen=True)
class BaselineAudit:
    occurrence_count: int
    unique_prompt_count: int
    occurrence_prompt_ids_sha256: str
    exclusion_prompt_ids: Sequence[str]
    exclusion_prompt_ids_sha256: str
    selection_boundary: SelectionBoundary
```

- [ ] **Step 4: Run GREEN and deterministic-byte checks.** Run the two focused files twice and assert byte-identical canonical receipts from identical inputs.

- [ ] **Step 5: Commit the audit contract.**

```bash
git add examples/dataset/specdec_corpus_contracts.py examples/dataset/audit_ptv2_baseline.py tests/examples/dataset/test_audit_ptv2_baseline.py tests/examples/dataset/test_specdec_identity.py
git commit -s -S -m "fix(specdec): distinguish PTV2 occurrences from prompt UUIDs"
```

### Task 2: Encode and validate the exact prompt-count policy

**Files:**
- Create: `examples/dataset/bprime_cd_policy.py`
- Create: `examples/dataset/bprime_cd_prompt_policy.yaml`
- Create: `tests/examples/dataset/test_bprime_cd_policy.py`

**Interfaces:**
- Consumes: policy YAML.
- Produces: immutable `PromptPolicy`, `PromptCell`, and exact count/reserve lookup used by inventory, selection, readiness, and launch manifests.

- [ ] **Step 1: Add RED tests for exact quotas and invalid policies.** Assert B′ totals 700K, C/D total 2M, D lanes total 600K, reserve counts use `ceil(quota * 1.2)`, German is denied, and floating weights or mismatched totals are rejected.

```python
def test_approved_policy_has_exact_integer_counts() -> None:
    policy = load_prompt_policy(POLICY)
    assert policy.arms["B-prime"].prompt_count == 700_000
    assert policy.arms["C"].prompt_count == 2_000_000
    assert policy.arms["D"].lanes["interactive-swe-replay"] == 200_000
    assert policy.reserve_count("D", "interactive-swe-replay") == 240_000
```

- [ ] **Step 2: Run RED.**

Run: `PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest -q tests/examples/dataset/test_bprime_cd_policy.py`

Expected: import fails because the policy module is absent.

- [ ] **Step 3: Implement strict integer policy parsing.** Use frozen dataclasses, reject unknown keys, require positive integer counts, require reserve numerator/denominator `6/5`, verify each parent total, and derive a canonical policy SHA-256.

```python
@dataclass(frozen=True)
class PromptPolicy:
    schema_version: int
    seed: int
    reserve_numerator: int
    reserve_denominator: int
    arms: Mapping[str, ArmPolicy]
    exposure_tokens: Sequence[int]
    policy_sha256: str
```

- [ ] **Step 4: Write the approved YAML and run GREEN.** Store B′ `200000/125000/125000/125000/125000`, C/D `600000/400000/200000/400000/300000/100000`, D lanes `200000/200000/200000`, milestones `256000000/1000000000`, allowed languages, denied German, sequence length 4096, and full-context maximum 32768.

- [ ] **Step 5: Commit the policy boundary.**

```bash
git add examples/dataset/bprime_cd_policy.py examples/dataset/bprime_cd_prompt_policy.yaml tests/examples/dataset/test_bprime_cd_policy.py
git commit -s -S -m "feat(specdec): define B-prime C D prompt quotas"
```

### Task 3: Pin and stage the complete PTV2/PTV3 source inventory

**Files:**
- Create: `examples/dataset/stage_ptv23_sources.py`
- Create: `examples/dataset/bprime_cd_sources.json`
- Create: `tests/examples/dataset/test_stage_ptv23_sources.py`
- Create: `tools/launcher/common/specdec/run_ptv23_source_stage.sbatch`
- Create: `tools/launcher/common/specdec/submit_ptv23_source_stage.sh`

**Interfaces:**
- Consumes: exact source descriptors, Hugging Face cache or local files, approved-use flag, durable root.
- Produces: `SourceInventory` with repository/configuration/split/revision/license/file digests and per-cell raw counts.

- [ ] **Step 1: Add RED source-manifest tests.** Cover exact 40-character revisions, non-empty license expressions, approved-use booleans, byte/SHA verification, duplicate logical file rejection, stale file detection, and refusal to consume unpinned entries from `nemotron_ptv3_datasets.yaml`.

```python
def test_source_inventory_requires_pin_license_digest_and_approval(tmp_path: Path) -> None:
    with pytest.raises(SourceManifestError, match="approved use"):
        load_source_inventory(write_manifest(tmp_path, approved_use=False))
```

- [ ] **Step 2: Run RED.**

Run: `PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest -q tests/examples/dataset/test_stage_ptv23_sources.py`

Expected: the new module is absent.

- [ ] **Step 3: Implement fail-closed staging.** Reuse the existing staged Qwen3-4B source pins. Inventory interactive SWE from `nvidia/Nemotron-SFT-SWE-v2:openhands_swe` and `nvidia/Nemotron-SWE-v1:r2e_gym`; inventory generic tool data from the `interactive_agent` and `tool_calling` splits of `nvidia/Nemotron-Agentic-v1` and, if required for the exact reserve, `nvidia/Nemotron-SFT-Agentic-v2`. Resolve and record each exact remote commit and license before downloading; omit any entry that cannot produce a complete approved descriptor. Stream-copy to a job-local partial tree, verify bytes and SHA-256, and atomically publish the inventory receipt.

```python
@dataclass(frozen=True)
class SourceIdentity:
    repository_id: str
    configuration: str
    split: str
    revision: str
    license_expression: str
    approved_use: bool
    files: Sequence[SourceFile]
```

- [ ] **Step 4: Add the CPU SLURM wrapper.** Require `--cpus-per-task=144` on GB200 CPU data-mover nodes when available, use no GPU allocation, bind output to source-manifest SHA, call `git pull --ff-only` before submission, and make the submitter run `sbatch --test-only` before the real `sbatch` path.

- [ ] **Step 5: Run GREEN plus shell checks.**

Run: `PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest -q tests/examples/dataset/test_stage_ptv23_sources.py`

Run: `bash -n tools/launcher/common/specdec/run_ptv23_source_stage.sbatch tools/launcher/common/specdec/submit_ptv23_source_stage.sh`

Run: `shellcheck -S warning tools/launcher/common/specdec/run_ptv23_source_stage.sbatch tools/launcher/common/specdec/submit_ptv23_source_stage.sh`

- [ ] **Step 6: Commit source staging.**

```bash
git add examples/dataset/stage_ptv23_sources.py examples/dataset/bprime_cd_sources.json tests/examples/dataset/test_stage_ptv23_sources.py tools/launcher/common/specdec/run_ptv23_source_stage.sbatch tools/launcher/common/specdec/submit_ptv23_source_stage.sh
git commit -s -S -m "feat(specdec): stage pinned PTV23 prompt sources"
```

### Task 4: Build canonical candidates and validate replay lanes

**Files:**
- Modify: `examples/dataset/build_specdec_inventory.py`
- Modify: `examples/dataset/trajectory_schema.py`
- Modify: `tests/examples/dataset/test_build_specdec_inventory.py`
- Modify: `tests/examples/dataset/test_trajectory_schema.py`
- Create: `tests/examples/dataset/test_bprime_cd_candidate_inventory.py`

**Interfaces:**
- Consumes: `SourceInventory`, baseline exclusion, held-out manifests, pinned tokenizer/template.
- Produces: `CandidateInventory` with unique prompt IDs, domain/lane/language, full-context bucket, source provenance, replay validity, capacity counts, and quarantine receipts.

- [ ] **Step 1: Add RED candidate tests.** Cover historical and held-out exclusion before counting, cross-source duplicates, UUID collision, language normalization, full-context-derived buckets, source completion stripping, and stable per-cell capacity.

- [ ] **Step 2: Add RED replay tests.** Include valid multi-call interactive-SWE and generic-tool traces plus declarations-only, undeclared function, duplicate call ID, orphan result, unresolved call, malformed arguments, unsupported role, reasoning-mode mismatch, and a 4K cut that splits a transaction.

```python
@pytest.mark.parametrize(
    ("mutation", "reason"),
    [("declarations_only", "no_assistant_tool_call"), ("orphan_result", "orphan_tool_result"), ("open_call", "unresolved_tool_call")],
)
def test_invalid_replay_is_quarantined(mutation: str, reason: str) -> None:
    assert quarantine_reason(replay_fixture(mutation)) == reason
```

- [ ] **Step 3: Run RED.**

Run: `PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest -q tests/examples/dataset/test_build_specdec_inventory.py tests/examples/dataset/test_trajectory_schema.py tests/examples/dataset/test_bprime_cd_candidate_inventory.py`

Expected: missing candidate-capacity and quarantine interfaces fail.

- [ ] **Step 4: Implement one-pass canonical inventory.** Validate source identity before row parsing, canonicalize once, apply historical then held-out then admitted-candidate exclusion, derive context buckets from exact tokenizer output, strip target-synthesis completions, implement `validate_trajectory(...)` as the typed entrypoint around canonical replay validation, validate replay referential integrity, and count every rejection by stable code.

```python
@dataclass(frozen=True)
class CandidateCell:
    arm_domain: str
    lane: str
    language: str
    context_bucket: str

@dataclass(frozen=True)
class CandidateInventory:
    rows: Sequence[CanonicalPrompt]
    capacity: Mapping[CandidateCell, int]
    quarantine_counts: Mapping[str, int]
    inventory_sha256: str
```

- [ ] **Step 5: Run GREEN and source-order invariance.** Run the three focused files, then shuffle fixture source order and prove identical inventory bytes and digest.

- [ ] **Step 6: Commit canonical candidate construction.**

```bash
git add examples/dataset/build_specdec_inventory.py examples/dataset/trajectory_schema.py tests/examples/dataset/test_build_specdec_inventory.py tests/examples/dataset/test_trajectory_schema.py tests/examples/dataset/test_bprime_cd_candidate_inventory.py
git commit -s -S -m "feat(specdec): inventory validated B-prime C D candidates"
```

### Task 5: Select exact B′ and paired C/D prompt views

**Files:**
- Create: `examples/dataset/select_bprime_cd_prompts.py`
- Create: `tests/examples/dataset/test_select_bprime_cd_prompts.py`
- Modify: `examples/dataset/build_specdec_study_corpus.py`
- Modify: `tests/examples/dataset/test_ptv23_complement_selection.py`

**Interfaces:**
- Consumes: `CandidateInventory`, `PromptPolicy`, baseline and held-out receipt digests.
- Produces: `PromptViewBundle` containing reserve and final ordered UUIDs, exact count proofs, paired C/D proofs, frozen context floors, and selection identity.

- [ ] **Step 1: Add RED B′ tests.** Scale quotas down by a common fixture factor and assert exact STEM/JA/ES/FR/IT counts, zero German, global UUID uniqueness, 20% reserves, deterministic order, and fatal one-row shortfall.

- [ ] **Step 2: Add RED paired C/D tests.** Assert exact top-level and D lane counts, identical non-agentic UUID sets, identical bucket floors, distinct agentic response lanes, and a different selection digest after changing seed, held-out digest, source digest, or quota.

```python
def test_cd_views_are_paired_outside_agentic_lanes() -> None:
    bundle = select_prompt_views(INVENTORY, POLICY)
    assert bundle.C.non_agentic_prompt_ids == bundle.D.non_agentic_prompt_ids
    assert bundle.D.lane_counts == {
        "agentless-swe": 200_000,
        "interactive-swe-replay": 200_000,
        "generic-tool-replay": 200_000,
    }
```

- [ ] **Step 3: Run RED.**

Run: `PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest -q tests/examples/dataset/test_select_bprime_cd_prompts.py tests/examples/dataset/test_ptv23_complement_selection.py`

Expected: prompt-count selector and paired proof are absent.

- [ ] **Step 4: Implement deterministic count selection.** Rank by SHA-256 of policy digest, seed, domain, lane, language, bucket, source identity, source row, and prompt UUID. For every paired C/D non-agentic cell, compute the common per-bucket capacity as the minimum of both eligible inventories after exclusions. Allocate the cell quota proportionally to that common vector with integer largest-remainder rounding, then require `ceil(bucket_quota * 6 / 5)` candidates in each populated bucket. Freeze that vector before arm selection, allocate reserve first, select exact final counts from valid candidates, and emit a blocker receipt rather than partial output on shortfall.

- [ ] **Step 5: Run GREEN and inspect canonical receipts.** Assert B′ count 700,000, C/D counts 2,000,000, all per-cell sums, reserve ratio, and paired digest in a large-count arithmetic test that does not allocate two million fixture rows.

- [ ] **Step 6: Commit prompt views.**

```bash
git add examples/dataset/select_bprime_cd_prompts.py examples/dataset/build_specdec_study_corpus.py tests/examples/dataset/test_select_bprime_cd_prompts.py tests/examples/dataset/test_ptv23_complement_selection.py
git commit -s -S -m "feat(specdec): select exact B-prime C D prompt views"
```

### Task 6: Regenerate targets and promote the 20% reserve

**Files:**
- Create: `examples/dataset/promote_synthesis_reserve.py`
- Create: `tests/examples/dataset/test_promote_synthesis_reserve.py`
- Modify: `tools/launcher/common/specdec/build_qwen4b_synthesis_manifest.py`
- Modify: `tools/launcher/common/specdec/run_qwen4b_synthesis.sbatch`
- Modify: `tools/launcher/tests/test_qwen4b_dataset_study.py`

**Interfaces:**
- Consumes: frozen prompt reserve, exact target/runtime generation identity, immutable generation attempts.
- Produces: `ResponseCorpus` with exact promoted prompt counts, successful/failed/unused receipts, and per-cell response-token histograms.

- [ ] **Step 1: Add RED generation-identity tests.** Reject source completions, wrong target/tokenizer/template/runtime, mixed thinking mode, partial finish, empty assistant output, conflicting successes, missing attempt identity, and malformed tool output.

- [ ] **Step 2: Add RED deterministic-promotion tests.** Prove that failures in the primary quota are replaced from the same cell's reserve in frozen order, successful retries are idempotent, unused reserve remains reported, and a cell with fewer successes than its exact quota fails atomically.

```python
def test_promotion_uses_only_same_cell_reserve() -> None:
    corpus = promote_responses(VIEW, ATTEMPTS, IDENTITY)
    assert corpus.cell_counts == VIEW.required_cell_counts
    assert corpus.promoted_ids == EXPECTED_PRIMARY_PLUS_SAME_CELL_REPLACEMENTS
```

- [ ] **Step 3: Run RED.**

Run: `PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest -q tests/examples/dataset/test_promote_synthesis_reserve.py tools/launcher/tests/test_qwen4b_dataset_study.py`

Expected: reserve-promotion interfaces and manifest fields fail.

- [ ] **Step 4: Implement response validation and promotion.** Define frozen `GenerationIdentity`, `GenerationAttempt`, and `ResponseCorpus` dataclasses. Hash every request and response, preserve retry history, choose the first canonical successful response per identity, reject conflicting successes, promote by frozen cell rank, join only already-validated D replay records after target promotion, and bind output to target, mode, generation configuration, container, and source selection digest.

- [ ] **Step 5: Update the synthesis launcher.** Shard by prompt-cell and frozen rank, pass no source assistant completion, stage only immutable inputs, use unique partial output namespaces, publish no final corpus until promotion succeeds, and emit per-replica GPU utilization metadata.

- [ ] **Step 6: Run GREEN plus shell checks and commit.**

Run: `PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest -q tests/examples/dataset/test_promote_synthesis_reserve.py tools/launcher/tests/test_qwen4b_dataset_study.py`

Run: `bash -n tools/launcher/common/specdec/run_qwen4b_synthesis.sbatch`

```bash
git add examples/dataset/promote_synthesis_reserve.py tests/examples/dataset/test_promote_synthesis_reserve.py tools/launcher/common/specdec/build_qwen4b_synthesis_manifest.py tools/launcher/common/specdec/run_qwen4b_synthesis.sbatch tools/launcher/tests/test_qwen4b_dataset_study.py
git commit -s -S -m "feat(specdec): promote exact target-synthesis prompt views"
```

### Task 7: Build exact assistant-token exposure views

**Files:**
- Create: `examples/dataset/build_assistant_token_views.py`
- Create: `tests/examples/dataset/test_build_assistant_token_views.py`
- Modify: `examples/dataset/build_specdec_study_corpus.py`
- Modify: `tests/examples/dataset/test_build_specdec_study_corpus.py`

**Interfaces:**
- Consumes: target-specific `ResponseCorpus` for B′, C, or D.
- Produces: exact 256M, 1B, largest-common, and one-pass `ExposureView` manifests with aligned token IDs/loss masks.

- [ ] **Step 1: Add RED exact-token tests.** Cover final assistant-token masking across interleaved zero/one loss masks, rejection of zero-token records, deterministic epoch shuffle, no count from padding/tool/user tokens, and resume fingerprint mismatch.

- [ ] **Step 2: Add RED common-boundary tests.** Assert that `min(C.total_assistant_tokens, D.total_assistant_tokens)` is the full common boundary, that C/D 256M and 1B views have identical total exposure, and that full-prompt totals remain separately reported.

```python
def test_common_boundary_is_exact_and_preserves_full_totals() -> None:
    views = build_paired_exposure_views(CORPUS_C, CORPUS_D, (256_000_000, 1_000_000_000))
    assert views.common_boundary == min(CORPUS_C.total_tokens, CORPUS_D.total_tokens)
    assert views.C["common"].assistant_tokens == views.D["common"].assistant_tokens
```

- [ ] **Step 3: Run RED.**

Run: `PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest -q tests/examples/dataset/test_build_assistant_token_views.py tests/examples/dataset/test_build_specdec_study_corpus.py`

Expected: exposure-view module and paired boundary contract are absent.

- [ ] **Step 4: Implement deterministic views.** Define frozen `ExposureView` and paired-view receipt types. Order by response-corpus digest, epoch, domain/lane, prompt UUID, and seed; consume full records until the boundary; copy and mask only the final record's remaining assistant positions; bind input IDs, loss mask, row order, unique prompts, repeated prompts, and cumulative tokens into the receipt.

- [ ] **Step 5: Run GREEN and commit.**

```bash
git add examples/dataset/build_assistant_token_views.py examples/dataset/build_specdec_study_corpus.py tests/examples/dataset/test_build_assistant_token_views.py tests/examples/dataset/test_build_specdec_study_corpus.py
git commit -s -S -m "feat(specdec): materialize exact assistant-token exposure views"
```

### Task 8: Publish and resume immutable corpus bundles

**Files:**
- Create: `examples/dataset/specdec_publication.py`
- Create: `tests/examples/dataset/test_specdec_publication.py`
- Modify: `tools/launcher/common/specdec/transfers/bundle_manifest.py`
- Create: `tools/launcher/tests/test_specdec_bundle_manifest.py`

**Interfaces:**
- Consumes: source, selection, response, tokenized shard, exposure, and rejection receipts.
- Produces: no-replace `PublicationReceipt` and transferable content-addressed bundle.

- [ ] **Step 1: Add RED atomicity tests.** Simulate interruption before shard close, after shard close, before manifest, before directory `fsync`, and before rename. Assert no final namespace appears and a retry resumes only immutable verified shards with matching input identity.

- [ ] **Step 2: Add RED corruption and collision tests.** Reject missing/extra files, changed bytes, stale partial input digest, symlink entries, existing final destination, separate source/selection publication, and a transfer completion marker with the wrong artifact commit.

- [ ] **Step 3: Run RED.**

Run: `PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest -q tests/examples/dataset/test_specdec_publication.py tools/launcher/tests/test_specdec_bundle_manifest.py`

Expected: publication module and corpus-specific receipt binding are absent.

- [ ] **Step 4: Implement publication.** Define `CorpusBundle` as the complete source/selection/response/exposure/rejection input set. Write explicit union-schema Parquet shards to a job-unique sibling partial directory, hash and reread all files, reconcile prompt/token/quarantine counts, `fsync` files and directories, use no-replace rename semantics, and retain mismatched partials under a quarantine name.

```python
@dataclass(frozen=True)
class PublicationReceipt:
    artifact_id: str
    corpus_manifest_sha256: str
    selection_manifest_sha256: str
    file_count: int
    total_bytes: int
    published_path: str
```

- [ ] **Step 5: Run GREEN and commit.**

```bash
git add examples/dataset/specdec_publication.py tests/examples/dataset/test_specdec_publication.py tools/launcher/common/specdec/transfers/bundle_manifest.py tools/launcher/tests/test_specdec_bundle_manifest.py
git commit -s -S -m "feat(specdec): publish immutable prompt corpus bundles"
```

### Task 9: Aggregate readiness and run bounded cluster canaries

**Files:**
- Modify: `examples/dataset/ptv23_builder_readiness.py`
- Modify: `tests/examples/dataset/test_ptv23_builder_readiness.py`
- Create: `tools/launcher/common/specdec/bprime_cd_study_manifest.py`
- Create: `tools/launcher/tests/test_bprime_cd_study_manifest.py`
- Create: `tools/launcher/common/specdec/run_bprime_cd_builder.sbatch`
- Create: `tools/launcher/common/specdec/run_bprime_cd_canary.sbatch`
- Create: `tools/launcher/common/specdec/submit_bprime_cd_study.sh`

**Interfaces:**
- Consumes: all upstream receipts, cluster profile, destination receipt, bounded target and training inputs.
- Produces: per-arm `ReadinessReceipt`, synthesis-canary receipt, training-canary receipt, evaluator-canary receipt, and dependency-safe submission IDs.

- [ ] **Step 1: Add RED readiness tests.** Require exact unique counts, reserve counts, source approvals, file hashes, replay lane counts, paired C/D proof, context floors, generation identities, exposure views, atomic publication, and destination receipt. Assert every missing gate has a stable blocker code and no weights are renormalized.

- [ ] **Step 2: Add RED launcher tests.** Verify cluster/account/partition policy, `--gpus-per-node` rather than job-wide `--gpus`, all allocated GPU ranks have work, unique output/W&B identity, immutable receipt arguments, no login-node bulk copy, dependency wiring, and refusal to submit production from a canary-only command.

- [ ] **Step 3: Run RED.**

Run: `PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest -q tests/examples/dataset/test_ptv23_builder_readiness.py tools/launcher/tests/test_bprime_cd_study_manifest.py`

Expected: new prompt-count gates and launcher contract fail.

- [ ] **Step 4: Implement aggregate readiness.** Define `ReadinessInputs` as the complete upstream receipt set and `ReadinessReceipt` as per-arm readiness plus stable blocker codes. Replace nonzero-token checks with exact per-cell unique-count and 20%-reserve checks, require pinned interactive-SWE and generic-tool inventories, verify all receipt digests transitively, and emit `ready` independently for B′, C, and D.

- [ ] **Step 5: Implement bounded canaries.** The builder canary uses scaled exact quotas in every cell. The synthesis canary exercises one primary failure and same-cell reserve promotion. The training canary runs 20 optimizer steps, saves and reloads a checkpoint, exports the drafter, records finite loss and per-GPU DCGM activity, and runs a small evaluator bundle.

- [ ] **Step 6: Run GREEN and static checks.**

Run: `PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest -q tests/examples/dataset/test_ptv23_builder_readiness.py tools/launcher/tests/test_bprime_cd_study_manifest.py`

Run: `bash -n tools/launcher/common/specdec/run_bprime_cd_builder.sbatch tools/launcher/common/specdec/run_bprime_cd_canary.sbatch tools/launcher/common/specdec/submit_bprime_cd_study.sh`

Run: `shellcheck -S warning tools/launcher/common/specdec/run_bprime_cd_builder.sbatch tools/launcher/common/specdec/run_bprime_cd_canary.sbatch tools/launcher/common/specdec/submit_bprime_cd_study.sh`

- [ ] **Step 7: Commit readiness and canaries.**

```bash
git add examples/dataset/ptv23_builder_readiness.py tests/examples/dataset/test_ptv23_builder_readiness.py tools/launcher/common/specdec/bprime_cd_study_manifest.py tools/launcher/tests/test_bprime_cd_study_manifest.py tools/launcher/common/specdec/run_bprime_cd_builder.sbatch tools/launcher/common/specdec/run_bprime_cd_canary.sbatch tools/launcher/common/specdec/submit_bprime_cd_study.sh
git commit -s -S -m "feat(specdec): gate B-prime C D cluster canaries"
```

### Task 10: Adopt one parent checkpoint and enforce exact-token training

**Files:**
- Create: `examples/speculative_decoding/checkpoint_adoption.py`
- Create: `tests/examples/speculative_decoding/test_checkpoint_adoption.py`
- Create: `modelopt/torch/speculative/assistant_token_budget.py`
- Create: `tests/unit/torch/speculative/test_assistant_token_budget.py`
- Modify: `examples/speculative_decoding/main.py`
- Modify: `examples/speculative_decoding/eagle_utils.py`
- Modify: `tools/launcher/common/specdec/run_qwen4b_study_training.sbatch`

**Interfaces:**
- Consumes: verified parent checkpoint, experiment identity, exposure view, destination readiness, canary receipt.
- Produces: `AdoptionReceipt`, exact cumulative token state, milestone checkpoint receipts, and a new isolated training namespace.

- [ ] **Step 1: Add RED adoption tests.** Accept identical model/drafter tensors and reject name, shape, dtype, or hash mismatch. Prove optimizer, scheduler, scaler, RNG, loader cursor, epoch, global step, and W&B resume state are absent from the new run and the parent hash remains recorded as lineage.

- [ ] **Step 2: Add RED token-budget tests.** Simulate distributed ranks with different local loss-mask counts, require all-reduced cumulative assistant tokens, stop exactly at 256M/1B/common boundaries, checkpoint the counter and fingerprint, and reject resume with another corpus, seed, target, mode, method, block size, or parent.

```python
@dataclass(frozen=True)
class ExposureFingerprint:
    corpus_sha256: str
    exposure_view_sha256: str
    parent_checkpoint_sha256: str
    target_revision: str
    arm: str
    method: str
    block_size: int
    seed: int
```

- [ ] **Step 3: Run RED.**

Run: `PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest -q tests/examples/speculative_decoding/test_checkpoint_adoption.py tests/unit/torch/speculative/test_assistant_token_budget.py`

Expected: new adoption and token-budget modules are absent.

- [ ] **Step 4: Implement weight-only adoption.** Define `ExperimentIdentity` in the launcher manifest and `AdoptionReceipt` in the adoption module. Inventory parent tensors, compare against the initialized child, load model/drafter weights only, assert fresh training state, and write an atomic adoption receipt before the first optimizer step.

- [ ] **Step 5: Integrate exact-token state.** Count the actual loss mask used in each distributed batch, all-reduce the increment, persist cumulative tokens and fingerprint with checkpoints, stop on the exact masked boundary, and keep historical source occurrence exposure as report metadata rather than optimizer state.

- [ ] **Step 6: Run GREEN, related trainer tests, and commit.**

Run: `PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest -q tests/examples/speculative_decoding/test_checkpoint_adoption.py tests/unit/torch/speculative/test_assistant_token_budget.py tests/examples/speculative_decoding`

```bash
git add examples/speculative_decoding/checkpoint_adoption.py tests/examples/speculative_decoding/test_checkpoint_adoption.py modelopt/torch/speculative/assistant_token_budget.py tests/unit/torch/speculative/test_assistant_token_budget.py examples/speculative_decoding/main.py examples/speculative_decoding/eagle_utils.py tools/launcher/common/specdec/run_qwen4b_study_training.sbatch
git commit -s -S -m "feat(specdec): cut over with exact assistant-token state"
```

### Task 11: Validate evaluators and refresh the concise HTML progress page

**Files:**
- Create: `tools/launcher/common/specdec/collect_bprime_cd_results.py`
- Create: `tools/launcher/tests/test_collect_bprime_cd_results.py`
- Create: `reports/qwen3-4b-hybrid-study/render_status.py`
- Create: `tests/reports/test_qwen3_4b_hybrid_status.py`
- Create: `reports/qwen3-4b-hybrid-study/data/bprime_cd_progress.json`
- Modify: `reports/qwen3-4b-hybrid-study/index.html`
- Modify: `tools/launcher/common/specdec/speculators_eval_artifacts.py`
- Modify: `tools/launcher/tests/test_speculators_eval.py`

**Interfaces:**
- Consumes: parent/B′/C/D checkpoint receipts, evaluator manifests and CSV/JSON, corpus/readiness receipts, job/log/W&B links.
- Produces: canonical `ProgressReport` JSON and deterministic concise HTML showing measured, running, blocked, and planned states without converting planned values into results.

- [ ] **Step 1: Add RED evaluator tests.** Require identical target/runtime/hardware/K/CUDA-graph/concurrency/request identities within comparisons; validate SPEED-Bench 1K/4K/8K/16K/32K buckets, acceptance counters, latency/throughput, Math/Code/STEM/multilingual/Chat/SWE/tool category coverage, tool-schema validity, and lossless target equivalence.

- [ ] **Step 2: Add RED progress-page tests.** Provide one completed, one running, one blocked, and one planned fixture. Assert escaped output, no credentials or raw environment values, measured-result provenance links, correct historical `1.3M occurrences / 931,363 unique`, exact B′/C/D tables, blocker codes, checkpoint step/hash, and no numeric result rendered for a non-completed evaluator.

```python
def test_page_never_labels_planned_metrics_as_measured(tmp_path: Path) -> None:
    html = render_status(progress_fixture(status="planned", metric=1.23), tmp_path)
    assert "1.23" not in html
    assert "Planned" in html
```

- [ ] **Step 3: Run RED.**

Run: `PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest -q tools/launcher/tests/test_collect_bprime_cd_results.py tools/launcher/tests/test_speculators_eval.py tests/reports/test_qwen3_4b_hybrid_status.py`

Expected: result collector and renderer are absent, and the evaluator lacks the expanded category/context contract.

- [ ] **Step 4: Implement result collection.** Define `ResultInputs` as validated corpus/training/evaluator receipt paths and `ProgressReport` as immutable status plus measured metrics. Bind every metric row to evaluator revision, prompt digest, target, drafter checkpoint, K, concurrency, hardware/runtime, and timestamp. Compute paired deltas only when identities match. Record parent, B′ 256M, B′ 700K, C/D 256M, C/D 1B, and common-boundary status independently.

- [ ] **Step 5: Implement deterministic page rendering.** Render a single compact page with four sections: current cluster/checkpoint status, prompt/corpus readiness, training milestones, and measured evaluator comparisons. Read only validated `bprime_cd_progress.json`, include source/job/log/W&B links when present, and label missing source capacity as blocked rather than estimated.

- [ ] **Step 6: Run GREEN and render verification.**

Run: `PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest -q tools/launcher/tests/test_collect_bprime_cd_results.py tools/launcher/tests/test_speculators_eval.py tests/reports/test_qwen3_4b_hybrid_status.py`

Run: `.venv/bin/python reports/qwen3-4b-hybrid-study/render_status.py --input reports/qwen3-4b-hybrid-study/data/bprime_cd_progress.json --output reports/qwen3-4b-hybrid-study/index.html --check`

- [ ] **Step 7: Commit the report path.**

```bash
git add tools/launcher/common/specdec/collect_bprime_cd_results.py tools/launcher/tests/test_collect_bprime_cd_results.py reports/qwen3-4b-hybrid-study/render_status.py tests/reports/test_qwen3_4b_hybrid_status.py reports/qwen3-4b-hybrid-study/data/bprime_cd_progress.json reports/qwen3-4b-hybrid-study/index.html tools/launcher/common/specdec/speculators_eval_artifacts.py tools/launcher/tests/test_speculators_eval.py
git commit -s -S -m "feat(specdec): report B-prime C D progress and results"
```

### Task 12: Run the integrated dry run and prepare authorized submissions

**Files:**
- Modify only if verification exposes a defect in a file owned by Tasks 1–11.

**Interfaces:**
- Consumes: the complete implementation and fixture manifests.
- Produces: local verification evidence, canonical dry-run receipts, and test-only SLURM evidence. It does not submit production jobs or push a branch.

- [ ] **Step 1: Run all focused dataset and launcher suites.**

Run: `PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest -q tests/examples/dataset tests/examples/speculative_decoding tests/unit/torch/speculative/test_assistant_token_budget.py tools/launcher/tests tests/reports/test_qwen3_4b_hybrid_status.py`

- [ ] **Step 2: Run static validation.**

Run: `.venv/bin/ruff check examples/dataset examples/speculative_decoding tools/launcher/common/specdec reports/qwen3-4b-hybrid-study tests/examples/dataset tests/examples/speculative_decoding tools/launcher/tests tests/reports`

Run: `.venv/bin/pyright examples/dataset examples/speculative_decoding tools/launcher/common/specdec reports/qwen3-4b-hybrid-study`

Run: `bash -n tools/launcher/common/specdec/*.sh tools/launcher/common/specdec/*.sbatch`

Run: `shellcheck -S warning tools/launcher/common/specdec/*.sh tools/launcher/common/specdec/*.sbatch`

- [ ] **Step 3: Run canonical fixture dry runs.** Build a scaled B′/C/D source inventory, candidate reserve, exact prompt views, response promotion, exposure views, atomic corpus, readiness receipt, transfer bundle, adoption receipt, evaluator bundle, and HTML report twice; compare all canonical JSON and HTML bytes between runs.

- [ ] **Step 4: Run cluster test-only checks.** After `git pull --ff-only` on the selected cluster checkout, run the source-stage, builder, and canary submitters with `--test-only`. Record resolved account, partition, node/GPU topology, command, source commit, and receipt digests. Do not submit a real job from this step.

- [ ] **Step 5: Run pre-commit and inspect the final diff.**

Run: `.venv/bin/pre-commit run --all-files`

Run: `git diff --check`

Run: `git status --short`

- [ ] **Step 6: Commit only verification-driven fixes.** If Steps 1–5 changed tracked files, stage only those files and create one signed+DCO commit with message `fix(specdec): close B-prime C D verification gaps`. If no files changed, record the verification commands and outputs without creating an empty commit.

## Execution Decomposition

Use dependency-aware parallel execution while keeping shared-file ownership
exclusive:

1. Run Task 1 and Task 2 first; they establish receipt and policy types.
2. Run Task 3 and Task 4 in parallel after Task 1; Task 3 owns staging files,
   while Task 4 owns inventory/trajectory files.
3. Run Task 5 after Tasks 2–4; it owns prompt selection.
4. Run Task 6 after Task 5. Task 7 may begin its token-view unit layer after
   Task 2, but integrate it only after Task 6 freezes `ResponseCorpus`.
5. Run Task 8 after Task 7; it owns publication and transfer binding.
6. Run Task 9 after Tasks 3–8; it owns readiness and canary launchers.
7. Run Task 10 after Tasks 7 and 9; it owns checkpoint adoption and training
   exposure state.
8. Run Task 11 after Tasks 8–10; it owns evaluator aggregation and report files.
9. Run Task 12 serially as the final repository and cluster test-only gate.

Workers are not alone in the codebase. Each worker must preserve unrelated
changes, modify only its assigned files, and stop for coordination if another
task has changed a shared file since the worker started.
