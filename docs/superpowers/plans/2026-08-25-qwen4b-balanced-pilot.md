# Qwen3-4B Balanced Dataset Pilot Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and launch a deterministic 100K-row historical-proportion versus balanced PTV2 pilot for Qwen3-4B DFlash B8.

**Architecture:** A single authenticated CPU producer reads pinned PTV2 splits, filters and globally deduplicates prompt identities, applies deterministic per-category quotas, token-counts only the selected rows, and atomically publishes two raw-JSONL bundles. The existing Qwen3-4B study trainer consumes those bundles after a minimal manifest-label extension.

**Tech Stack:** Python 3.13, Hugging Face `datasets` and tokenizer APIs, PyArrow-backed dataset reads, pytest, Bash/SLURM, OCI-HSG GB200.

**Spec:** `docs/superpowers/specs/2026-08-25-qwen4b-balanced-pilot-design.md`

## Global Constraints

- Source repository is exactly `nvidia/Nemotron-Post-Training-Dataset-v2` at one caller-pinned immutable revision.
- Seed is exactly `20260822`; each phase-1 arm has exactly `100000` rows.
- Historical-proportion quotas are chat 48286, math 18420, code 13462, stem 19832.
- Balanced quotas are math 25000, code 20000, stem 25000, chat 20000, multilingual 10000; multilingual is 2000 per de/es/fr/it/ja split.
- The approved later arms remain B=100% PTV2, C=75% PTV2+25% PTV3, and D=50% PTV2+50% PTV3; PTV3 rows are never substituted with PTV2 rows.
- The scaled PTV3 component keeps 30% SWE/agentic/tool, 20% math, 10% code, 20% stem/science, 15% multilingual, and 5% instruction/chat; D splits its 30% lane equally among agentless SWE, interactive SWE replay, and generic tool replay.
- Selection uses global exact prompt dedup, held-out exclusion, deterministic hash ranking, and no silent quota redistribution.
- Tool/tool-call rows and rows without a terminal assistant response are excluded from this raw-streaming pilot.
- Dataset building uses one `nemotron_n3_post/cpu_datamover` node with exactly 96 requested CPUs; scratch is under `/raid/scratch`, durable output under `/lustre`.
- Training reuses the existing two-node Qwen3-4B DFlash B8 runner, 20-step canary gate, seed 42, exact 16,000,000 assistant-token budget, and pilot-only `nemotron_n3_post` account.
- No remote job may be submitted before a signed+DCO commit is pushed and remote SHA is verified.

---

### Task 1: Deterministic paired PTV2 sampler

**Files:**
- Create: `examples/dataset/build_qwen4b_balanced_pilot.py`
- Create: `examples/dataset/qwen3_4b_balanced_pilot.yaml`
- Create: `tests/examples/dataset/test_build_qwen4b_balanced_pilot.py`

**Interfaces:**
- Consumes: pinned PTV2 split iterables, a JSON held-out UUID array, tokenizer path and SHA-256, seed, worker count, output root.
- Produces: `build_pilot_bundles(...) -> PilotCompletion`, per-arm `data.jsonl`, `MANIFEST.json`, `EXECUTION.json`, and aggregate `COMPLETE.json`.

- [x] **Step 1: Write RED tests for the public selection API**

  Add genuine synthetic rows that assert the exact 1/20-scaled historical and approved B-balanced quotas, multilingual language quotas, stable output under reversed input and worker counts 1/96, global duplicate refill, and different selected IDs for a changed seed.

- [x] **Step 2: Run the selection tests and record the expected missing-module/API failures**

  Run: `/Users/sna/ModelOpt_SpecDec/worktrees/q4-ptv2-ab-integration/.venv/bin/python -m pytest -q tests/examples/dataset/test_build_qwen4b_balanced_pilot.py -k 'quota or seed or duplicate or parallel'`

- [x] **Step 3: Implement canonical prompt identity, validation, and deterministic bounded selection**

  Implement typed immutable quota/config/result dataclasses. Canonicalize messages, remove only the terminal assistant message for prompt identity, reject tool/tool-call rows, rank by SHA-256 of canonical `[seed, split, prompt_uuid]`, and refill each quota after global dedup/held-out filtering.

- [x] **Step 4: Run the focused selection tests to GREEN**

  Run the Step 2 command and require all selected tests to pass.

- [x] **Step 5: Write RED tests for token counting and atomic publication**

  Cover exact assistant-mask counting using the pinned Qwen3-4B chat template, manifest/file/self-hash replay, fewer-than-16M failure, tampered JSONL rejection, worker failure with no final root, and existing-destination no-replace behavior.

- [x] **Step 6: Run the publication tests and record the expected failures**

  Run: `/Users/sna/ModelOpt_SpecDec/worktrees/q4-ptv2-ab-integration/.venv/bin/python -m pytest -q tests/examples/dataset/test_build_qwen4b_balanced_pilot.py -k 'token or manifest or publication or failure or no_replace'`

- [x] **Step 7: Implement selected-row token counting and receipt-bound publication**

  Tokenize only selected rows in bounded worker batches, compute exact assistant tokens, write into a mode-0700 private scratch bundle, replay every descriptor and receipt, then install the final root atomically without replacement and fsync the destination parent.

- [x] **Step 8: Run the full new module test file**

  Run: `/Users/sna/ModelOpt_SpecDec/worktrees/q4-ptv2-ab-integration/.venv/bin/python -m pytest -q tests/examples/dataset/test_build_qwen4b_balanced_pilot.py`

#### Task 1 implementation report

- Review baseline: signed+DCO commit `fb0d0b48bd723a7d9d041b572ed5c7f9c96119be`.
- Frozen bundle schema: `qwen3-4b-balanced-pilot-v2`. Both arm manifests and aggregate `COMPLETE.json` bind the canonical sorted held-out evaluator UUID digest and exact count.
- External verifier anchor: `qwen3-4b-balanced-pilot-verification-trust-v1`, written outside the bundle with canonical self-hash, exclusive no-follow creation, mode `0600`, file fsync, and parent-directory fsync. Public verification requires this persisted receipt path.
- Fresh dependency-equipped test evidence: `46 passed, 1 skipped`; the single skip is the Linux-only kernel `renameat2` race test on Darwin.
- Static evidence: Ruff check/format, Pyright, `py_compile`, and `git diff --check` are required clean at the frozen review handoff.
- Final signed commit and full binary diff SHA-256 are recorded in the immutable review handoff because a commit cannot contain its own Git object ID.

### Task 2: OCI-HSG 96-CPU producer contract

**Files:**
- Create: `tools/launcher/common/specdec/run_qwen4b_balanced_pilot_dataset.sbatch`
- Create: `tools/launcher/common/specdec/submit_qwen4b_balanced_pilot_dataset.sh`
- Modify: `tests/examples/dataset/test_build_qwen4b_balanced_pilot.py`

**Interfaces:**
- Consumes: source commit/path, pinned dataset revision, tokenizer path/SHA, held-out receipt, image path/SHA, output root.
- Produces: one tested SLURM submission and the Task 1 immutable completion root.

- [ ] **Step 1: Write RED launcher tests**

  Assert `nemotron_n3_post`, `cpu_datamover`, one node, `--cpus-per-task=96`, `/raid/scratch` working state, `/lustre` output, duplicate-writer rejection from one filtered scheduler snapshot, required caller pins, and `sbatch --test-only` before real submission.

- [ ] **Step 2: Run launcher RED**

  Run: `/Users/sna/ModelOpt_SpecDec/worktrees/q4-ptv2-ab-integration/.venv/bin/python -m pytest -q tests/examples/dataset/test_build_qwen4b_balanced_pilot.py -k launcher`

- [ ] **Step 3: Implement the runner and submitter**

  Stage the clean pushed source under `/home`, use container and caches in node-local scratch, pass exactly 96 workers to the producer, preserve per-job logs under the experiment root, and publish only Task 1 durable artifacts.

- [ ] **Step 4: Run launcher and shell checks**

  Run the Step 2 pytest command, `bash -n` on both scripts, and `shellcheck` on both scripts.

### Task 3: Reuse the Qwen3-4B study trainer with descriptive pilot arms

**Files:**
- Modify: `tools/launcher/common/specdec/qwen4b_study_manifest.py`
- Modify: `tools/launcher/common/specdec/run_qwen4b_study_training.sbatch`
- Modify: `tools/launcher/tests/test_qwen4b_dataset_study.py`

**Interfaces:**
- Consumes: verified pilot `MANIFEST.json` descriptors.
- Produces: accepted `historical-proportion` and `balanced-pilot` study experiments with unchanged topology and schedule gates.

- [ ] **Step 1: Write RED manifest tests**

  Construct both descriptive arms and assert stable experiment/job identities, exact 100K corpus rows, exact 16M budget, pilot-only `nemotron_n3_post`, DFlash B8 only for the first pilot, and rejection of any other pilot arm, account, or budget. Add a runner test proving pilot corpus tokens may exceed the common target while non-pilot corpus tokens must still equal their target.

- [ ] **Step 2: Run manifest RED**

  Run: `/Users/sna/ModelOpt_SpecDec/worktrees/q4-ptv2-ab-integration/.venv/bin/python -m pytest -q tools/launcher/tests/test_qwen4b_dataset_study.py -k pilot`

- [ ] **Step 3: Implement the minimal typed manifest extension**

  Add typed `selection_mode` and corpus-token evidence to the study manifest, extend arm/account validation and readable naming only for `pilot-row-quota-v1`, and make the runner accept `corpus_unique_assistant_tokens >= assistant_token_budget` only for that mode. Do not alter node topology, checkpoint lifecycle, W&B receipts, or the existing B/C/D equality/account contract.

- [ ] **Step 4: Run focused and full study tests**

  Run the Step 2 command, then `/Users/sna/ModelOpt_SpecDec/worktrees/q4-ptv2-ab-integration/.venv/bin/python -m pytest -q tools/launcher/tests/test_qwen4b_dataset_study.py`.

### Task 4: Verification and local commit

**Files:**
- Verify all files from Tasks 1-3.

**Interfaces:**
- Consumes: completed implementation and test evidence.
- Produces: a frozen reviewed diff and one signed+DCO local commit; no push.

- [ ] **Step 1: Run combined regression tests**

  Run both changed test files plus `tests/examples/dataset/test_build_specdec_study_corpus.py`.

- [ ] **Step 2: Run static checks**

  Run Ruff check/format-check, Pyright for changed Python files, `bash -n`, ShellCheck, `python -m compileall`, and `git diff --check`.

- [ ] **Step 3: Freeze and independently review the exact diff**

  Record a binary-safe SHA-256 and obtain a read-only review covering quotas, dedup/held-out logic, deterministic parallelism, atomic publication, and unchanged trainer topology.

- [ ] **Step 4: Fix any findings through RED-GREEN and repeat review**

  Keep the worktree unpushed until the independent verdict is clean.

- [ ] **Step 5: Create one signed+DCO local commit**

  Stage only the listed implementation, test, design, and plan files and run `git commit -s -S -m "Add Qwen4B balanced dataset pilot"`. Stop before push and request explicit current-turn push approval.

### Task 5: Remote build and paired training after push approval

**Files:**
- No local source changes expected.

**Interfaces:**
- Consumes: pushed commit with remote SHA match and authenticated OCI-HSG inputs.
- Produces: one CPU dataset job, two 20-step canaries, two matched 16M-token DFlash B8 training runs, and monitored receipts.

- [ ] **Step 1: Verify the pushed SHA in a clean OCI-HSG `/home` checkout**

  Require local/remote/cluster HEAD equality and a clean worktree.

- [ ] **Step 2: Submit the dataset producer safely**

  Take one filtered `squeue --me` snapshot, run submitter dry-run and scheduler `--test-only`, submit once, and monitor for five minutes at intervals of at least 60 seconds.

- [ ] **Step 3: Authenticate both dataset bundles**

  Recompute both 100K row counts, quotas, prompt UUID roots, file hashes, assistant-token counts, and the aggregate completion receipt before allocating GPUs.

- [ ] **Step 4: Submit and monitor both 20-step canaries**

  Use the existing study submitter with exact manifest/receipt hashes. Submit only if no duplicate tuple exists and monitor each combined scheduler pass for at least five minutes.

- [ ] **Step 5: Submit the matched 16M-token runs after both canaries pass**

  Require both current-job canary receipts, then submit identical DFlash B8 topology and exact token budgets. Record W&B, checkpoint, export, and milestone receipt paths.

- [ ] **Step 6: Evaluate and report the directional result**

  Compare matched assistant-token training traces and fixed held-out drafter acceptance. Report balanced/control deltas and explicitly avoid a broader PTV3 or DSpark conclusion until replicated.
