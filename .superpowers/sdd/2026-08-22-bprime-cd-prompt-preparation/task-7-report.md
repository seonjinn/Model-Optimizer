# Task 7 report: exact assistant-token exposure views

Status: completed

Implemented disk-backed, authenticated assistant-token exposure staging:

- Final canonical Task 6 records are retokenized with the authenticated local
  trainer tokenizer snapshot and exact chat template. Production does not trust
  API usage fields or a caller-supplied tokenizer object.
- Trainer sequence truncation is applied before assistant-token counting. Only
  final assistant-owned mask positions are trimmed at an exposure boundary;
  zero-assistant-token records and unsafe replay transaction cuts fail closed.
- SQLite-backed token staging and streamed JSONL materialization avoid retaining
  a production corpus in memory. Each view is a consumable deterministic epoch
  content stream with exact order, cumulative counts, unique/repeated record
  counts, and full-corpus totals in its receipt.
- Resume reauthenticates staged databases, JSONL bytes, semantic rows, ordering,
  cumulative counts, and identity digests before returning an existing view.
- Production construction authenticates Task 5 manifests, indexes, shards,
  selection and paired C/D bucket digests; Task 6 promotion files, indexes,
  record streams, cell counts, histograms, histories, and corpus root; and the
  tokenizer snapshot files and chat template.
- Paired C/D construction proves identical selected UUIDs and bucket assignments
  before generating fixed 256M/1B, common-exposure, and one-pass views. Its
  receipt binds both arms' independent full-corpus token totals.
- Production policy permits only exact 256M and 1B fixed views plus a true
  one-pass view. The 64M value is exposed only as a runtime-screen milestone,
  and a common exposure can only be requested through the paired builder.
- Task 7 writes authenticated staging files exclusively and performs no final
  rename, deletion, or publication; Task 8 retains bundle-publication ownership.

## TDD evidence

- The isolated initial RED run collected the new tests and reported
  `7 failed, 19 passed`; the view module and policy API did not yet exist.
- Focused RED mutations subsequently covered unsafe tool-transaction truncation,
  transitive Task 5/6 authentication, production policy bypass, mutable
  tokenizer identity, resume semantic tampering, agentless lane handling, and
  public one-pass/common-label bypasses.
- Final isolated GREEN command:
  `PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest -q -o addopts='' --disable-warnings --confcutdir=tests/examples/dataset tests/examples/dataset/test_build_assistant_token_views.py tests/examples/dataset/test_build_specdec_study_corpus.py --basetemp=/tmp/task7-final-green-all`
  — `36 passed in 2.62s`.

## Verification

- Ruff format — all four implementation/test files unchanged.
- Ruff check — passed.
- Four-file pre-commit invocation — all applicable hooks passed, including
  Ruff, Ruff format, repository mypy, license validation, and Bandit.
- `git diff --check` — passed.

## Self-review and independent review

- Confirmed receipts bind input IDs and exact loss masks, so equal token counts
  cannot conceal different trainer examples.
- Confirmed deterministic ordering is represented as an actual consumable JSONL
  content view; the code does not claim a loader's later realized order.
- Confirmed the final partial record preserves earlier assistant-owned mask
  positions and trims only the later assistant-owned positions needed to reach
  the boundary.
- Confirmed the production tokenizer is reloaded local-only after authenticating
  the snapshot; fake or mutable caller objects cannot determine production
  counts.
- Three independent review rounds identified and drove fixes for sequence/tool
  cut safety, transitive artifact authentication, closed production policy,
  tokenizer snapshot binding, semantic resume checks, lane classification,
  and duplicate retokenization/performance concerns.

## Concern

The brief's literal pytest command remains unavailable because this worktree's
`.venv` lacks `torch`; repository `tests/conftest.py` stops before collection
with `ModuleNotFoundError: No module named 'torch'`. The isolated command above
excludes only that unrelated global bootstrap and passes the complete Task 7
test files.
