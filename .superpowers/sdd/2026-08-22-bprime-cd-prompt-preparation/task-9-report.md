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
