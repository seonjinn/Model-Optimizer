# Tasks 1--4 implementation report

## Delivered

- Added the typed canonical job manifest, topology validation, B/K mapping,
  pinned path/SHA validation, duplicate-tuple rejection, and atomic writes.
- Added one-node pinned Hugging Face staging with node-local download/caches,
  bounded local-to-Lustre publication, completion markers, test-only submission,
  duplicate detection, and scheduler-confirmed receipts.
- Added four-node wave rendering and the four-node runner. It enforces
  `-N4 --segment=4`, the two-serve/two-trainer split, GBS 512 arithmetic, clean
  source/SHA verification, per-node runtime/input staging, and node-local
  mutable paths. Durable model/data/checkpoint/result paths remain on Lustre.
- Added cumulative resume chaining: each below-four-hour training wave is
  followed by a one-node `--segment=1` public evaluator job. The next wave is
  gated by `afterok` and a non-empty acceptance CSV.
- Updated the Qwen3 30B documentation with Q30/Q235 Base/Thinking B/K mappings,
  segment semantics, `/raid/scratch` cache policy, and receipt locations.

## TDD evidence

1. Task 1 RED: `test_drafter_submission.py` failed collection with
   `ModuleNotFoundError: common.specdec.drafter_job_manifest`.
2. Task 2 RED: its staging contract failed because `stage_hf_model.sh` was
   absent.
3. Tasks 3--4 RED: their contracts failed because the wave and resume-chain
   entrypoints were absent.
4. GREEN: the focused contract suite passed after each implementation stage.

## Verification

- `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 pytest -q tools/launcher/tests/test_drafter_submission.py tools/launcher/tests/test_qwen_drafter_examples.py tools/launcher/tests/test_speculators_eval.py` — passed.
- `ruff check` and `ruff format --check` for the manifest and its tests — passed.
- `pyright` for the manifest and its tests — 0 errors, warnings, or information messages.
- `shellcheck` and `bash -n` for all four new shell/SBATCH entrypoints — passed.
- `git diff --check` — passed.

## Scope and operational notes

- No jobs were submitted and no branch was pushed.
- The wrapper contracts issue only filtered scheduler-name queries; periodic
  monitoring remains an operator action and must stay at least 60 seconds apart.
- The production scripts intentionally require `/home` source/configuration,
  `/lustre` durable large artifacts, and `/raid/scratch` mutable runtime/cache/
  lock/database/temp paths.
