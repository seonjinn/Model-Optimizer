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

## Review fix round 1

### Root causes and corrections

- The initial launcher still used `query.py`, copied its raw JSONL shards into
  the publication partial, and unconditionally wrote `promotion_status=passed`.
  It now calls the Task 6 generator directly. Each API response becomes a
  canonical `GenerationAttempt` with exact request/response hashes, attempt and
  generation identities, finish reason, assistant message, and API token count.
  A separate production promotion command authenticates the Task 5 manifest,
  shards, and SQLite index, calls the same attempt/replay validation contract,
  enforces same-cell reserve replacement and atomic exact quotas, and only then
  creates the promoted corpus receipt.
- Replay receipts previously accepted any lowercase 64-hex string. Promotion
  now calls `validate_trajectory(...)` independently, requires a real declared
  tool call/result transaction, reconciles its canonical bytes with the frozen
  selected row, and recomputes a proof over prompt UUID, lane, source identity,
  and canonical trajectory digest. A plain user/assistant chat is rejected.
- `mv -T --no-clobber` can exit successfully without moving when its destination
  exists. Publication now uses the repository's native Linux `renameat2` /
  Darwin `renamex_np` no-replace pattern. It authenticates the completion and
  every promoted file immediately before rename, verifies the installed inode,
  reauthenticates every installed file, and fsyncs the parent directory.
- Target responses now use closed outer and assistant schemas. Any extra
  assistant key, legacy `function_call`, or present `tool_calls` /
  `tool_call_id` field is a failed attempt even when the tool field is empty.
- The corpus root now binds every complete immutable attempt payload in arrival
  order, including attempt ID, request hash, response hash, retries, failures,
  and every replay validation receipt. It also binds the promoted per-record
  digest stream, exact cell counts, token histograms, and unused reserve proof.

### Task 7 handoff interface

- Production selection and response processing is disk-backed. Task 5 rows are
  read lazily from its authenticated SQLite byte-offset index with bounded file
  handles; attempts, successes, failures, reserve receipts, and promoted rows
  are spooled into a file-backed SQLite response index. No 2M-row Python list,
  tuple, dictionary, or aggregate JSON array is created.
- Every promoted JSONL row is a closed `PromotedResponseRecord` containing the
  prompt UUID, arm, domain, lane, language, frozen context bucket, final
  canonical messages/tools or replay, request and response hashes, exact
  selection and paired-C/D digests, generation identity, attempt ID or replay
  validation digest, and a per-record digest. `ResponseCorpus.records`, promoted
  IDs, attempt receipts, failed IDs, unused reserves, and attempt-history mapping
  have lazy SQLite-backed production views, so Task 7 can retokenize the final
  canonical content without Task 6 owning `input_ids` or `loss_mask`.
- All four per-replica GPU-utilization receipts are validated, copied into the
  promoted corpus, and included in both the promotion file manifest and final
  completion identity.

### RED evidence

- The expanded core tests first failed at collection because the authenticated
  replay factory did not exist. The launcher tests independently failed at
  collection because the native authenticated publisher did not exist.
- The new mutations cover arbitrary replay proof plus plain chat, extra
  assistant keys, legacy `function_call`, empty-but-present tool fields,
  complete retry/failed-attempt corpus-root binding, no-replace collision, bad
  completion identity, and the SQLite-backed Task 7 record artifact.

### GREEN verification

- `PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest -q -o addopts='' --disable-warnings --confcutdir=tests/examples/dataset tests/examples/dataset/test_promote_synthesis_reserve.py --basetemp=/tmp/task6-r1-commit-core`
  — final rerun: `18 passed in 0.41s`.
- `PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=tools/launcher .venv/bin/python -m pytest -q -o addopts='' --disable-warnings --confcutdir=tools/launcher/tests tools/launcher/tests/test_qwen4b_dataset_study.py --basetemp=/tmp/task6-r1-commit-launcher`
  — final rerun: `31 passed in 3.97s`.
- `PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest -q tests/examples/dataset/test_promote_synthesis_reserve.py tools/launcher/tests/test_qwen4b_dataset_study.py`
  — blocked before collection by the pre-existing
  `ModuleNotFoundError: No module named 'torch'` in `tests/conftest.py`.
- `bash -n tools/launcher/common/specdec/run_qwen4b_synthesis.sbatch` — passed.
- Six-file pre-commit invocation — passed Ruff, Ruff format, repository mypy,
  license validation, and Bandit.
- `git diff --check` — passed.

### Round-1 self-review

- Confirmed no final namespace is created until all immutable attempts, D replay
  trajectories, exact quotas, record digests, histograms, and corpus files have
  authenticated successfully under the unique promotion partial.
- Confirmed collision publication preserves both the existing winner and the
  publisher's authenticated partial; no shell no-clobber success ambiguity
  remains.
- Confirmed production iteration remains disk-backed through selection,
  attempt reconciliation, promotion, receipt generation, and Task 7 handoff.

## Review fix round 2

### Root cause and correction

- The API boundary persisted the vLLM `choice.message` object verbatim, while
  promotion intentionally accepts only the closed assistant schema
  `{role, content}`. Real vLLM responses add empty protocol defaults such as
  `reasoning_content: null`, `reasoning: ""`, and `tool_calls: []`, so otherwise
  valid completions were classified as failed and could exhaust every cell.
- The generator now closes the wire response before hashing or persistence.
  Exact assistant role/content is retained; only known semantically empty
  reasoning/tool/function defaults are removed. Unknown fields or nonempty
  reasoning, tool-call, or legacy function-call values fail closed into the
  canonical error-attempt envelope. Thus equivalent wire defaults produce the
  same deterministic response identity without weakening promotion validation.

### RED and GREEN evidence

- `PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest -q -o addopts='' --disable-warnings --confcutdir=tests/examples/dataset tests/examples/dataset/test_promote_synthesis_reserve.py -k 'vllm_empty_protocol_defaults_normalize or vllm_nonempty_protocol_fields' --basetemp=/tmp/task6-r2-red-all`
  — RED: `5 failed, 18 deselected in 0.06s`; the successful wire response kept
  the three empty defaults, and every nonempty protocol-field mutation remained
  a nominal success.
- The same focused command with `--basetemp=/tmp/task6-r2-green-focused` —
  GREEN: `5 passed, 18 deselected in 0.03s`.
- `PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest -q -o addopts='' --disable-warnings --confcutdir=tests/examples/dataset tests/examples/dataset/test_promote_synthesis_reserve.py --basetemp=/tmp/task6-r2-core`
  — `23 passed in 0.08s`.
- `PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=tools/launcher .venv/bin/python -m pytest -q -o addopts='' --disable-warnings --confcutdir=tools/launcher/tests tools/launcher/tests/test_qwen4b_dataset_study.py --basetemp=/tmp/task6-r2-launcher`
  — `31 passed in 1.82s`.
- `bash -n tools/launcher/common/specdec/run_qwen4b_synthesis.sbatch` — passed.
- `git diff --check` — passed.
- Three-file pre-commit invocation — passed Ruff, Ruff format, repository mypy,
  license validation, Bandit, and markdownlint.

## Review fix round 3

### Root cause and correction

- Round 2 used one generic JSON-emptiness predicate for every optional wire
  field. That erased malformed cross-type values such as `reasoning: []`,
  `tool_calls: ""`, and `function_call: {}` before the closed-schema validator
  could reject them.
- Wire defaults now have field-specific contracts: `reasoning` and
  `reasoning_content` accept only null or the empty string, `tool_calls` accepts
  only null or the empty list, and `function_call` accepts only null. All other
  values become the deterministic canonical failed-attempt envelope; valid
  defaults still normalize to identical role/content response bytes and hash.

### RED and GREEN evidence

- `PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest -q -o addopts='' --disable-warnings --confcutdir=tests/examples/dataset tests/examples/dataset/test_promote_synthesis_reserve.py -k 'vllm_invalid_protocol_fields' --basetemp=/tmp/task6-r3-red`
  — RED: `7 failed, 4 passed, 19 deselected in 0.08s`; every newly added
  cross-type mutation was incorrectly normalized as a success.
- The focused wire suite with
  `-k 'vllm_invalid_protocol_fields or vllm_empty_protocol_defaults_normalize' --basetemp=/tmp/task6-r3-green`
  — GREEN: `12 passed, 18 deselected in 0.05s`.
- `PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest -q -o addopts='' --disable-warnings --confcutdir=tests/examples/dataset tests/examples/dataset/test_promote_synthesis_reserve.py --basetemp=/tmp/task6-r3-core`
  — `30 passed in 0.09s`.
- `PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=tools/launcher .venv/bin/python -m pytest -q -o addopts='' --disable-warnings --confcutdir=tools/launcher/tests tools/launcher/tests/test_qwen4b_dataset_study.py --basetemp=/tmp/task6-r3-launcher`
  — `31 passed in 1.97s`.
- `bash -n tools/launcher/common/specdec/run_qwen4b_synthesis.sbatch` — passed.
- `git diff --check` — passed.
- Three-file pre-commit invocation — passed Ruff, Ruff format, repository mypy,
  license validation, Bandit, and markdownlint.
