# Task 9 Report: Qwen3-4B PTV2 A/B occurrence preparation

## Delivered

- Added the exact, fail-closed PTV2 policy: revision
  `5c89e01dd720ae0f4058445ed49c5fb68a03c76e`, seed `20260822`, one trainer
  epoch, GBS 512, 2M occurrence pass, and nominal milestones with the aligned
  cursors `500224`, `1000448`, `1300480`, and `2000000` (final 128 rows).
- Added disk-backed A-prefix and independently runnable B-balanced selectors.
  B has the required rank preimage
  `policy_sha256 || cell || source_identity || row || uuid`, SQL-streamed
  ranked selection, deterministic round-robin cycling, and no cross-cell
  borrowing. Its multilingual cell is five independent ranked/cycled subcells:
  DE/JA/ES/FR/IT at 40K each.
- Added the B operational CLI. It either consumes Task 3's published
  `SOURCE_INVENTORY.json` or stages a `SOURCE_PLAN.json` with Task 3's
  `stage_source_inventory`; it defaults to requiring exactly 201 declared
  Parquet shards. The physical tree must contain exactly those declarations,
  so the known 202nd orphan
  `multilingual-00000-of-00001.parquet` is rejected.
- Source iteration requires the staged Task 3 receipt and preserves its
  canonical source/file order. It rechecks file byte length/SHA-256, streams
  Parquet batches, requires source-native assistants, and records canonical
  conversation and response hashes per selected occurrence.
- SQLite indices are created as private `O_EXCL|O_NOFOLLOW` partials, fsynced,
  published through an atomic hard-link no-replace operation, and directory
  fsynced. Existing immutable indices are never overwritten. Interrupted
  partials remain intact and produce typed `PTV2StudyRecoveryError` recovery
  failures.
- Added the one-pass exposure adapter and PTV2 selection publication receipt
  schema authentication.

## TDD evidence

1. RED: the no-replace retry test initially failed because the old B selector
   deleted and rebuilt `ptv2-b-balanced-index.sqlite3` rather than raising
   `FileExistsError`.
2. GREEN: the same test passed after private no-follow creation, fsync/link
   publication, and immutable destination handling were implemented.
3. Earlier REDs covered the initially missing study module, the missing policy
   milestone cursor field, the missing B-only entry point, the missing
   one-pass exposure adapter, and the missing selection-receipt schema-v3
   descriptors. The corresponding focused tests were green before extending
   the next behavior.

## Verification

- `PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest -q --confcutdir=tests/examples/dataset tests/examples/dataset/test_qwen3_4b_ptv2_study.py tests/examples/dataset/test_build_assistant_token_views.py tests/examples/dataset/test_specdec_publication.py`
  - PASS: 60 tests. Pytest emitted 336 pre-existing macOS temporary-directory
    cleanup warnings; none are test failures.
- `.venv/bin/ruff check` on all Task 9 Python files: PASS.
- `.venv/bin/ruff format --check` on all Task 9 Python files: PASS.
- `python examples/dataset/qwen3_4b_ptv2_study.py --help`: PASS.
- `git diff --check`: PASS.

## Static/pre-commit availability

- `pre-commit` is not installed in this worktree (`command not found`).
- Pyright is not installed (`No module named pyright`).
- Mypy runs but cannot complete in this environment: its Python lacks `pip`,
  required PyYAML/PyArrow stubs are absent, and direct script invocation does
  not resolve sibling `examples/dataset` modules. No tooling was installed or
  bootstrapped, per the task direction.

## Self-review

- B readiness is independent of A source-order readiness.
- The selector keeps row and occurrence processing in SQLite cursors; it does
  not materialize source multiplicity/counter lists. Held-out UUIDs remain a
  supplied evaluator set by design.
- The full OCI source inventory was not staged or run locally. The executable
  path is intentionally prepared for the authorized cluster run and fails
  closed unless its Task 3 receipt has exactly the 201 declared shards and no
  physical orphan.

## Review round 1 corrective pass

- Replaced the obsolete A-prefix policy with A-repair: exact historical 1.3M,
  then the explicit STEM200K/JA125K/ES125K/FR125K/IT125K/DE0 complement. The
  selector refuses historical and held-out UUID overlap and no longer extends
  the next source-order tail.
- Replaced the continuous 3,907-step arithmetic with the required two-segment
  receipt: occurrence segments `1300000/700000`, segment steps `2540/1368`,
  cumulative steps `2540/3908`, and valid terminal counts `32/96`.
- Moved the executable module guard to EOF. A direct policy-loading CLI probe
  now reaches the intended Task 3 missing-receipt failure and does not raise
  `NameError`.
- Staged Parquet UUIDs now use only prompt-bearing system/developer/user
  messages plus top-level tools, matching `BaselineAudit`; full conversations
  and native assistant responses remain separately content-hashed.
- The production B wrapper requires a staged Task 3 receipt and typed
  `ExclusionIndex`; full paired selection requires typed source, baseline, and
  exclusion roots. The `201` declared-shard contract is a fixed code constant,
  with revision equality to the policy; it is no longer an operator CLI knob.
- Added persisted/public view summaries for per-cell/per-language counts and
  unique counts, repair composition, segment counts, and maximum multiplicity.
- Added a materialized Task 7 adapter which streams the immutable selection
  SQLite join, rechecks occurrence/conversation/response hashes, derives the
  response root and one-pass token count through the supplied assistant-mask
  counter, and rejects source-response mutation.
- Tightened Task 8 selection-v3 receipts to exact keys, SHA-256 syntax, and
  policy-byte/policy-digest equality; malformed lineage claims or added fields
  fail closed.

Corrective verification: focused Task 9 suite PASS (`63 passed`, 336
pre-existing pytest cleanup warnings); ruff check/format PASS; direct CLI probe
reaches the expected Task 3 receipt error; `git diff --check` PASS.

## Review round 2 corrective pass

- The paired full-policy API now routes A through the sole A-repair builder,
  checks the historical `BaselineAudit`, then derives complement composition
  from selected SQLite rows. Complement UUIDs are rejected if they overlap the
  historical, evaluator-held-out, or already-selected complement set.
- Full paired selection now requires typed `SourceInventory`, `BaselineAudit`,
  `ExclusionIndex`, and typed Task 5 `ExclusionReceipt` values plus a
  nonzero immutable complement selection digest. These roots are reconciled
  and carried into both immutable selection identities. B-only remains
  independently runnable through its authenticated Task 3 wrapper.
- The cluster CLI now requires a content-addressed JSON held-out UUID artifact
  and invokes that authenticated B wrapper rather than the low-level selector.
- Task 7 no longer accepts an arbitrary token-count callback. It invokes the
  supplied chat tokenizer for complete source-native conversations, validates
  aligned IDs/assistant masks, applies the final sequence boundary, materializes
  SQLite token/mask rows, and writes an fsynced immutable token receipt with
  token/packing/multiplicity/milestone summaries.
- Task 8 now carries distinct physical-policy and semantic-policy SHA-256
  fields. It authenticates policy bytes first, parses the authenticated YAML,
  and reconciles its canonical semantic digest with the selector identity.
  Schema-v3 Task 9 publications also reconcile selection, inventory, baseline,
  and held-out roots across every role receipt.
- The duplicate summaries now distinguish per-UUID natural/total multiplicity
  from per-source-occurrence constructed reuse, with matching histograms and
  correct maximum aggregation. B ranking retains exactly
  `policy_sha || cell || source_identity || row || uuid`.

### Round 2 TDD evidence

1. RED: the policy semantic-hash publication regression failed after adding
   distinct semantic and physical SHA fields because schema-v3 rejected the
   additional field before it could authenticate the canonical YAML identity.
2. GREEN: the regression passes after schema-v3 validates both hashes in the
   correct phases: descriptor bytes during artifact authentication and parsed
   canonical YAML during semantic reconciliation.
3. RED: the B CLI smoke test failed when the new required held-out receipt
   input was introduced; it passed after the test supplied the explicit JSON
   artifact and verified dispatch to `select_authenticated_b_balanced_view`.

### Round 2 verification

- Focused Task 9 pytest command: PASS, `63 passed` (336 pre-existing macOS
  temporary-directory cleanup warnings).
- Ruff check on changed Task 9 files: PASS.
- Ruff format applied then checked on changed Task 9 files: PASS.
- `git diff --check`: PASS.
- `pre-commit` and Pyright remain unavailable; no environment bootstrapping
  was performed.

### Follow-up response-binding regression

- RED: an authenticated SQLite row whose separately hashed response was not
  the final assistant message in its canonical conversation was accepted by
  the new tokenizer path.
- GREEN: `test_ptv2_derivation_rejects_a_response_not_in_the_tokenized_conversation`
  now passes after Task 7 verifies that exact final assistant payload before
  applying the chat template and assistant mask.
