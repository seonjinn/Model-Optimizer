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

## Claude hostile-review corrective round

Claude Code Opus found no critical issue in the frozen third-review diff but
identified four actionable verifier/test warnings. Technical reproduction
confirmed the two production concerns before implementation:

- RED path-swap and same-inode-rewrite tests both showed that the verifier could
  hash `data.jsonl` through one open and then accept rows read through another
  open while the published path changed. Replay now opens the arm directory and
  data leaf componentwise with no-follow descriptors, hashes and parses the same
  file descriptor, recomputes token evidence, and compares the file inode plus
  file/parent metadata before and after. Both deterministic attacks now fail
  with `data file changed during verification`.
- RED caller-scratch coverage showed the verifier had no way to route its
  authenticated tokenizer copy into a private allocation. Public completion
  replay now accepts a caller-owned mode-0700 scratch root. If omitted, it owns
  and cleans an unpredictable mode-0700 wrapper created by `mkdtemp`; the
  tokenizer snapshot is nested inside rather than staged directly in a public
  temp directory. Tests reject mode-0755 caller scratch and prove both caller
  and fallback scratch trees are empty after replay.
- Renamed the input-reversal test so it makes no false worker-fanout claim. Real
  1-versus-96 token-worker byte identity remains covered by the existing bundle
  test.
- Replaced the unused-function monkeypatch streaming test with a cursor proxy
  whose weakly tracked rows fail if the production writer materializes more
  than two selected rows. The real writer stays within the bound and emits all
  expected records.
- Clarified matching failure as global quota infeasibility after prompt dedup;
  the non-Linux rename path remains test-only because the public production
  identity already requires Linux.

Fresh local evidence on 2026-08-25:

- New focused gates: `6 passed, 36 deselected in 5.06s`.
- Full isolated Task 1 file: `41 passed, 1 skipped in 209.90s`; the sole skip is
  Linux-only `renameat2`. The 190.84-second outlier was the unchanged local HF
  tokenizer fixture; all new tests completed in less than one second each.
- Ruff check/format and Pyright (`0 errors, 0 warnings, 0 informations`) pass.

This round is local and unpushed; an independent non-Claude rereview is still
required after the signed follow-up commit.

## Final trust-publication transaction corrective round

The independent review of signed HEAD `a954d2151` found that the external trust
receipt could lose its requested lexical pathname during publication, that a
restrictive caller umask could weaken its exact mode contract, and that the
bundle was installed before the required external trust receipt. All production
changes in this round were preceded by deterministic failing regressions:

- Two receipt-path races renamed or replaced the lexical parent after file or
  directory fsync. The old writer returned success; the new writer retains a
  no-follow descriptor for every path component, retains the created file
  descriptor, and rebinds the complete root-to-leaf directory chain plus the
  named receipt inode before and after the final parent fsync. A forged receipt
  at a replacement pathname is neither accepted nor deleted.
- A mode test under umask `0777` observed mode `000` before the fix. Receipt
  publication now applies `fchmod(0600)` through the held file descriptor and
  verifies exact regular-file mode, link count, bytes, and stable identity.
- An injected failure before bundle rename previously left no trust receipt.
  Publication is now trust-first and create-or-authenticate: an exact existing
  receipt is retryable, while a foreign or mismatched receipt is never replaced.
- An injected failure after no-replace bundle installation now leaves an exact
  trust/bundle pair that a rerun authenticates and adopts without replacing the
  installed inode. A destination that does not match freshly recomputed bundle
  evidence is rejected and is never deleted or overwritten.
- A sixth RED replaced the receipt immediately after bundle rename. The builder
  now reopens and authenticates the external receipt after installed bundle and
  COMPLETE verification, before it can report success.

Fresh local evidence on 2026-08-25:

- Original five RED cases: `5 failed, 47 deselected`; after implementation:
  `5 passed, 47 deselected`.
- Post-install receipt replacement RED: `1 failed, 52 deselected`; after the
  final reauthentication: GREEN.
- Final publication/trust focused gate: `7 passed, 46 deselected in 0.52s`.
- Dependency-available local suite: `51 passed, 1 skipped, 1 deselected in
  2.67s`. The deselected genuine HF Qwen tokenizer integration is unchanged and
  could not run because this host currently has neither `tokenizers` nor
  `transformers`; the unfiltered command reported exactly that one
  `ModuleNotFoundError`, alongside `51 passed, 1 skipped`.
- Ruff check and format check: passed.
- Pyright: `0 errors, 0 warnings, 0 informations`.
- `py_compile` and `git diff --check`: passed.

The final signed corrective HEAD and the frozen full binary diff SHA-256 are
recorded in the independent-review handoff; a commit cannot embed its own object
ID. No push, SSH session, build, or cluster job was performed.
