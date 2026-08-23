# Task 6 report: exact response promotion

Status: completed

Implemented an immutable response-validation and reserve-promotion boundary:

- Frozen `GenerationIdentity`, `GenerationAttempt`, `ValidatedReplayRecord`, and
  `ResponseCorpus` records bind target revision, tokenizer, chat template,
  runtime, container, source selection, thinking mode, and generation settings.
- Canonical request and response bytes are independently rehashed. Promotion
  rejects missing attempt identities, mismatched generation identities, source
  assistant completions, wrong requests, partial finishes, empty outputs,
  malformed tool outputs, and conflicting successful retries.
- The first canonical success per prompt wins. Identical retries are idempotent,
  while complete attempt history is retained for distinct immutable attempts.
- Failed primaries are filled only from the same frozen cell's reserve order.
  Exact per-cell shortfalls fail before a `ResponseCorpus` exists, and unused
  reserve IDs remain explicit.
- D replay rows join after target-native promotion only when supplied with a
  canonical record hash and a nonzero prior trajectory-validation digest.
- Corpus receipts include exact promoted IDs, successful and failed attempts,
  failed prompt IDs, unused reserve IDs, per-cell response-token histograms,
  complete generation identity, and a canonical corpus digest.
- Synthesis manifests now expose a complete generation-identity digest and an
  independent chat-template digest. Completion reuse checks the generation
  identity and successful promotion status.
- The launcher retains immutable attempt namespaces across retries, uses a
  unique promotion partial namespace, verifies the exact chat template, binds
  completion receipts to all generation inputs, and emits per-replica GPU
  utilization metadata. Final publication still uses a no-clobber rename only
  after the promotion/receipt validation block completes.

## TDD evidence

- The exact brief command first stopped at repository test bootstrap because
  this worktree's `.venv` lacks `torch` (`ModuleNotFoundError` from
  `tests/conftest.py`).
- The isolated Task 6 RED run then failed during collection because
  `promote_synthesis_reserve` did not exist.
- The launcher RED run failed on the absent immutable-attempt, promotion-partial,
  frozen-order, and GPU-utilization contracts.
- The D replay RED run failed because `ValidatedReplayRecord` was absent.
- GREEN dataset suite:
  `PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest -q -o addopts='' --disable-warnings --confcutdir=tests/examples/dataset tests/examples/dataset/test_promote_synthesis_reserve.py --basetemp=/tmp/task6-dataset-tests`
  — `11 passed in 0.51s`.
- GREEN launcher suite:
  `PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=tools/launcher .venv/bin/python -m pytest -q -o addopts='' --disable-warnings --confcutdir=tools/launcher/tests tools/launcher/tests/test_qwen4b_dataset_study.py --basetemp=/tmp/task6-launcher-tests`
  — `30 passed in 8.50s`.

## Verification

- `bash -n tools/launcher/common/specdec/run_qwen4b_synthesis.sbatch` — passed.
- Full five-file pre-commit invocation — passed all applicable hooks, including
  Ruff, Ruff format, repository-configured mypy, license validation, and Bandit.
- `git diff --check` — passed.

## Self-review

- Confirmed all target prompt rows are preflighted before attempts are selected,
  so a tainted source completion cannot be hidden by a successful reserve.
- Confirmed reserve lookup never crosses a cell boundary and corpus construction
  happens only after every exact quota is satisfied.
- Confirmed response success requires canonical hashed content, exact `stop`
  completion, a nonempty assistant message, positive exact response tokens, and
  no tool-call fields.
- Confirmed distinct successes for one prompt cannot silently choose a response,
  while byte-identical retries remain deterministic.
- Confirmed replay records cannot join B-prime/C, target attempts cannot claim a
  replay row, and missing D replay validation fails closed.

## Concern

The brief's literal combined pytest command remains unavailable until `torch` is
installed in this worktree's `.venv`. The same complete Task 6 files pass in the
two isolated commands above; the isolation excludes only the unrelated global
conftest bootstrap.
