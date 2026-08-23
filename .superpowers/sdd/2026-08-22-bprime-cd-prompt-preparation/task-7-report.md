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
  selection and paired C/D bucket digests; externally pinned Task 6 completion
  identities, promotion files, indexes, record streams, cell counts,
  histograms, histories, and corpus roots; and the tokenizer snapshot files and
  chat template.
- Paired C/D construction streams the authenticated final promotion indexes and
  proves identical non-agentic UUIDs and context buckets after reserve
  replacement, before generating fixed 256M/1B, common-exposure, and one-pass
  views. Its receipt binds both arms' completion identities, independent
  full-corpus token totals, and the final paired-row digest.
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
  `PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest -q -o addopts='' --disable-warnings --confcutdir=tests/examples/dataset tests/examples/dataset/test_build_assistant_token_views.py tests/examples/dataset/test_build_specdec_study_corpus.py --basetemp=/tmp/task7-r1-full2`
  — correction-round final rerun: `41 passed in 121.67s`; collection and
  execution were unusually slow under concurrent host CPU contention.

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
- Independent review identified and drove fixes for sequence/tool cut safety,
  transitive artifact authentication, closed production policy, tokenizer
  snapshot binding, semantic resume checks, lane classification, duplicate
  retokenization/performance concerns, and the correction round below.

## Review fix round 1

### Root causes and corrections

- B's promoted records carry the study-global Task 5 paired digest, while the
  original tokenized staging logic intentionally populated its record-scoped
  paired field only for C/D. Production incorrectly compared those differently
  scoped fields. Receipts now retain a zero record-scoped pair for B and bind the
  separately named authenticated Task 5 paired proof for every arm.
- Production previously trusted the claimed generation-identity digest after
  checking only tokenizer/template members. It now requires the exact Task 6
  dataclass schema, recomputes the canonical `asdict` digest across target,
  tokenizer, template, runtime, container, source selection, thinking mode,
  temperature, and generation limits, and reconciles it with the corpus claim.
- A self-consistent `PROMOTION.json` tree had no external trust root. Every
  production arm now requires the caller's expected Task 6 `completion.json`
  SHA-256, authenticates that published file and its declared files, reconciles
  the promotion receipt and full generation identity, and carries the trusted
  completion SHA through staging, view, and paired receipts.
- Task 5 proved the selected C/D candidates, but reserve replacement could make
  the final promoted corpora differ. The paired builder now merge-streams both
  authenticated SQLite indexes in UUID order, filters only the agentic-only
  domain, requires exact `(prompt_uuid, context_bucket)` equality, and binds the
  resulting final-row digest without materializing the corpus.

### RED and GREEN evidence

- Focused RED: `5 failed, 15 deselected`; B production rejected the new trusted
  completion API, and the generation, completion, and final-pair authenticators
  were absent.
- Focused GREEN: `6 passed, 15 deselected`, including separate reserve-ID and
  context-bucket drift mutations plus a deterministic final stream digest.
- Full isolated Task 7 suite: initial correction run `41 passed in 3.65s`;
  final post-review rerun `41 passed in 121.67s` under host contention.

## Concern

The brief's literal pytest command remains unavailable because this worktree's
`.venv` lacks `torch`; repository `tests/conftest.py` stops before collection
with `ModuleNotFoundError: No module named 'torch'`. The isolated command above
excludes only that unrelated global bootstrap and passes the complete Task 7
test files.
