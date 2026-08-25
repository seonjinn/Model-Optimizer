# Task 1 Report: Deterministic paired PTV2 sampler

## Delivered

- Added the immutable quota/config/result API and deterministic paired selector.
- Fixed the phase-1 scientific identity at seed `20260822`, with the exact
  historical `48286/18420/13462/19832` and balanced
  `25000/20000/25000/20000/10000` quotas; multilingual selection is exactly
  2,000 rows per DE/ES/FR/IT/JA split.
- Canonical prompt UUIDs remove exactly the terminal assistant message; tool,
  malformed, nonterminal-assistant, duplicate, and held-out rows are excluded
  before bounded per-quota refill.
- Added JSON held-out UUID array loading, SHA-256 rank ordering, and
  input-order/worker-count deterministic selection.
- Added selected-row assistant-mask token accounting, private-scratch receipt
  creation, manifest/execution/aggregate completion replay, tamper detection,
  no-final-root failure handling, no-replace publication checks, and parent
  directory fsync after install.

## TDD evidence

1. Selection RED: the requested focused pytest command was started while
   `build_qwen4b_balanced_pilot.py` was absent. Direct module loading produced
   `FileNotFoundError` for that exact path. The repository pytest fixture
   startup subsequently stalled before test output, so it was terminated after
   more than three minutes.
2. Selection GREEN: the same five selection tests passed with isolated pytest:
   `5 passed, 5 deselected in 12.91s` using `--noconftest`; direct execution of
   the five cases also passed.
3. Publication RED: `test_token_count_uses_exact_assistant_mask` failed with
   `AttributeError: ... has no attribute 'count_assistant_tokens'`.
4. Held-out loader RED: its new test failed with
   `AttributeError: ... has no attribute 'load_held_out_prompt_uuids'`.
5. GREEN: direct execution reported `all 11 pilot tests passed`; isolated
   pytest also reported `2 passed in 5.46s` for held-out receipt and manifest
   replay, and `1 passed in 3.73s` for manifest replay before the final
   held-out-loader addition.

## Verification

- `ruff check`: passed.
- `ruff format --check`: passed.
- `pyright`: `0 errors, 0 warnings, 0 informations`.
- `python -m compileall -q`: passed.
- `git diff --check`: passed.

## Concern

The exact requested pytest invocations load the repository root test fixtures
(including Torch/ModelOpt) and stalled before collecting this standalone test
file in this environment. `--noconftest` and direct execution exercise the
same test bodies without changing them; no test coverage was removed.

## Review-fix follow-up

- Added RED/GREEN checks for public production-identity rejection, lowercase
  held-out IDs, boolean quota counts, and an `EEXIST` racing final-install
  simulation. The isolated race test passed: `1 passed in 0.91s`.
- The public builder now rejects non-frozen scientific identities, verifies the
  tokenizer artifact SHA before loading the local tokenizer, records the chat
  template digest, uses Linux `renameat2(..., RENAME_NOREPLACE)`, protects
  execution worker evidence from metadata overwrite, validates exact split
  quotas/global UUID uniqueness/source/tokenizer identities during replay, and
  keeps test-sized builds behind a private helper.
- Remaining review concern: selection still materializes normalized candidates;
  resolved in follow-up commit: source rows are now streamed into a private
  mode-0700 SQLite spool, indexed by split/rank, and read back only through
  deterministic per-quota scans. The spool is deleted in all success/failure
  paths.

## Bounded-selection follow-up evidence

- Added a one-shot iterator/private-scratch test, a bounded in-flight
  tokenization executor test, worker-receipt overwrite protection, and
  independent execution-receipt tamper rejection.
- Replaced the 100K in-memory synthetic selector fixtures with a scaled quota
  fixture while retaining literal assertions for the frozen production quota
  constants. This makes serial/reversed/96-worker determinism checks quick.
- Isolated pytest completed the full test file after the tool's 30-second
  output yield (ten progress dots were emitted before completion); direct
  streaming selection coverage reported `streaming selection tests passed` in
  28.2 seconds. Ruff, Pyright, compileall, and diff checks passed.

## Second-review corrective round

The frozen `96ff59b31..e4a01b635` rereview reported 6 critical findings,
4 warnings, and 2 nits. This round addressed each finding through a failing
regression before the implementation change:

- Replaced quota-order greedy cross-split dedup with deterministic
  quota-capacity matching and canonical per-split response ownership. The
  adversarial feasible-quota and reversed conflicting-response tests changed
  from RED to GREEN.
- Bound assistant-token accounting to the trainer-visible first 4,096 token
  positions. The outside-window mask test changed from 5 to the exact expected
  count of 1.
- Added strict receipt schemas, exact scalar types, exact arm/seed/quota/row
  production identity, per-row token evidence, cross-arm pins, and semantic
  tamper tests. Public replay now rejects authenticated tiny fixtures; tests
  use the explicit non-production verifier only for scaled fixtures.
- Staged tokenizer trees into a mode-0700 private snapshot using componentwise
  no-follow descriptors, stable source-file identity checks, and mode-0600
  files; hashing and `AutoTokenizer` loading use that same staged path. A real
  locally serialized Qwen/ChatML generation-tag template verifies exactly one
  assistant token at the response boundary.
- Replaced the architecture-specific syscall number with libc `renameat2` and
  `RENAME_NOREPLACE`. The real two-publisher Linux race test is present and is
  skipped only on this Darwin host; the existing simulated EEXIST race remains
  GREEN locally.
- The bundle builder now spools selection and scalar token evidence in private
  SQLite, keeps at most `2 * workers` token futures in flight, and streams both
  JSONL publication and replay without retaining selected message bodies.
- Moved requested/effective worker evidence to a separately self-hashed
  `EXECUTION.json`. Serial and 96-worker builds now produce byte-identical
  `data.jsonl`, `MANIFEST.json`, and `COMPLETE.json` while execution receipts
  differ.
- Normalized public filesystem, malformed mapping, JSON, worker, and verifier
  exceptions to `PilotError`; removed the obsolete materialized selector and
  writer helpers and removed the ignored public tokenizer object argument.

Fresh verification on 2026-08-25:

- Full isolated test file: `34 passed, 1 skipped in 53.35s`; the sole skip is
  the real Linux `renameat2` race on Darwin.
- Genuine local HF Qwen boundary test: passed in 52.11s with no network access.
- Ruff check and format check: passed.
- Pyright: `0 errors, 0 warnings, 0 informations`.
- Compileall and `git diff --check`: passed.

The repository-root pytest fixture remains unsuitable for this standalone test
on this host, so the canonical bounded equivalent is the full file with
`--noconftest`. No test body or parameter is omitted.

Corrective commit: `66d2ebc9d2a1a5155bc5e0e138e48948c3bf3029`.
The commit has a verified ED25519 Git signature and DCO sign-off. The worktree
is clean. Frozen full binary diff `96ff59b31..66d2ebc9d` SHA-256:
`fa2ff258b93db60a91f9771e1b77c36aef8a7838376fc9877452491e0f998dc5`.

## Third-review corrective round

The fresh review of `96ff59b31..66d2ebc9d` reported two critical findings and
two warnings. Each was reproduced before changing production behavior:

- The 12,000-candidate, 2,000-row-per-split adversarial matcher executed 1,308
  `SELECT` statements in the smaller RED reproduction and the reviewer measured
  11.7 seconds at the requested scale. The replacement streams all ranked edges
  in one SQLite statement, maintains deterministic heap-backed residual edges,
  and applies a capacitated augmenting path over at most the declared split
  count. The exact 12K/2K regression now completes matching in 0.37 seconds and
  executes at most three `SELECT` statements, including the final integrity
  count. Existing quota-feasibility and input/worker byte-identity tests remain
  GREEN.
- A fully rehashed 6-to-1 per-row assistant-token forgery reached the old replay
  path without an authenticated tokenizer recomputation. Replay now stages the
  manifest-pinned tokenizer through the same componentwise no-follow copier,
  authenticates the copied tree, loads only that offline snapshot, and
  recomputes every post-4,096 assistant mask before accepting the declared row,
  digest, or aggregate token counts. The exact attacker rehash is rejected.
- Added an explicit OCI pre-submission API that authenticates an offline
  tokenizer snapshot, checks the Qwen3 `<|im_start|>`/`<|im_end|>` and EOS IDs,
  exercises the assistant-mask probe, and returns the tree/template identities.
  The contract rejects a non-Qwen identity and removes every private snapshot
  on success or failure; the generic local WordLevel boundary test remains.
- The durable copied publication tree is now recursively fsynced file-first and
  directory-bottom-up immediately before `renameat2`. An injected inode-level
  regression proves every copied regular file and directory has been synced
  before the atomic rename callback runs.

Fresh local verification on 2026-08-25:

- Full isolated Task 1 file: `38 passed, 1 skipped in 10.87s`; the sole skip is
  the real Linux-only `renameat2` race on Darwin.
- Exact 12K-candidate/2K-per-split matcher: `1 passed in 0.40s`, with the matcher
  itself reported at 0.31--0.37 seconds across fresh runs.
- Ruff check and format check: passed.
- Pyright: `0 errors, 0 warnings, 0 informations`.
- Compileall and `git diff --check`: passed.

This round remains local and unpushed pending an independent rereview.
