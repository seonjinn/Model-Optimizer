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

## Fast-review fix round

- Added pinned target topology to every manifest: Q30 uses capture IDs
  `[2,13,24,35,46,48]`, serve TP 2, per-device batch 4, and accumulation 16;
  Q235 uses `[2,25,47,69,92,94]`, serve TP 4, per-device batch 2, and
  accumulation 32. Both explicitly assert `per-device * accumulation * 8 =
  512`.
- The training runner now consumes the manifest-pinned image and capture IDs,
  copies `/home` source plus immutable inputs/runtime to per-node
  `/raid/scratch`, and launches training in the pinned image without cloning or
  building on Lustre.
- Added normalized realpath containment for pinned paths, so `..` traversal and
  symlink-resolved escapes outside `/home` or `/lustre` are rejected.
- Receipts, job names/comments, evaluation outputs, and exports now carry a
  stable experiment identity. Resume chains process each manifest index
  independently even when experiments share cumulative boundaries.
- Evaluator submissions pass method, B, and K through exported environment
  values and use a fixed self-reentry script rather than an interpolated
  `--wrap` command. Duplicate checks consult receipts, filtered `squeue`, and
  filtered `sacct`, including blank-`sbatch` fallback.

## Controller integration fix round

- Added RED/GREEN contract coverage for runtime relocation, pinned-container
  imports, output mounting, role-specific staging, W&B configuration, and the
  reusable runtime probe.
- The runner now rewrites archived virtual-environment activation/wrapper paths
  after extracting the runtime on each node's `/raid/scratch` directory, then
  verifies `accelerate`, `datasets`, `modelopt`, and `wandb` with the relocated
  `MODELOPT_RUNTIME/bin/python` and staged-source `PYTHONPATH`.
- Added `probe_relocatable_runtime.sh`, a one-node pinned-image preflight that
  requires `/home` source, Lustre image/archive, and `/raid/scratch` extraction
  before asserting the same imports. It neither clones source nor builds on
  Lustre.
- Both Pyxis invocations use `--no-container-mount-home`; the training Pyxis
  invocation explicitly mounts the Lustre output root. Serve nodes stage the
  full target, trainers stage the dataset and only fake-base target sidecars.
- Removed the invalid recipe override `training.global_batch_size=512`; GBS is
  retained solely as the manifest arithmetic preflight. Production submitters
  reject all extra dotlist arguments, so no unquoted override expansion remains.
  The runner configures `training.report_to=wandb`, `WANDB_PROJECT`, and
  node-local W&B cache/run paths.
- Node-local source, target, dataset, and trainer sidecar copies use
  `cp -aL`, materializing Q30/OPB Hugging Face blob symlinks before the
  container runs without Lustre input mounts. The relocated runtime probe also
  imports `modelopt` and rejects any import origin outside the staged source.
- Every cumulative training boundary now passes `training.save_steps=$MAX_STEPS`
  and `training.save_total_limit=2`, guaranteeing a real
  `checkpoint-$MAX_STEPS` under the stable output root for the next wave's
  automatic resume. Long-lived final archival remains a separate policy.
