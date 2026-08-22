# Qwen3-4B Hybrid Dataset Study Implementation Plan

> **For Codex:** Use `superpowers:executing-plans` to implement this plan task by task. Do not submit production GPU jobs until the readiness and 20-step canary gates pass.

**Goal:** Build and run a reproducible Qwen3-4B thinking-on dataset-mixture screen comparing balanced PTV2 (B), 75/25 PTV2/PTV3 (C), and 50/50 PTV2/PTV3 (D), with exact assistant-token budgets and no idle GPUs.

**Architecture:** Add a deterministic corpus-planning layer that inventories pinned PTV2/PTV3 sources, validates tool trajectories, samples by assistant-token quotas and context buckets, and emits immutable Parquet manifests. Extend response synthesis with an explicit thinking mode and schema-preserving outputs. Add an OCI-HSG study manifest/launcher that stages Qwen3-4B and data into node-local scratch, launches four TP1 target replicas per serving node, uses every trainer GPU, and gates 256M-token arms on a successful 20-step canary. Keep short evaluator jobs separate on `nemotron_n3_post`; run study training on the verified `nemotron_sw_post` profile.

**Tech Stack:** Python 3.12+, Hugging Face `datasets`/`transformers`, PyArrow/Parquet Zstd, vLLM OpenAI server, ModelOpt DFlash/DSpark trainer, Bash/Slurm/Pyxis, pytest, Ruff, ShellCheck.

---

## Task 1: Make thinking mode explicit and reproducible

**Files:**
- Modify: `tools/launcher/common/query.py`
- Create: `tests/tools/launcher/test_query_synthesis.py`

- [ ] Write subprocess/unit tests proving `--thinking-mode on` never adds `/no_think`, `off` always adds it, and `source` honors a row's explicit field.
- [ ] Write a regression test proving shard parity no longer changes thinking mode.
- [ ] Add `--thinking-mode {on,off,source}` with `source` as the compatibility default.
- [ ] Remove the even-shard `disable_thinking_column` mutation.
- [ ] Record mode, source ID, target revision, generation parameters, and output checksum in a shard sidecar; publish `.done` only after both data and sidecar are durable.
- [ ] Run: `pytest -q tests/tools/launcher/test_query_synthesis.py`.

## Task 2: Preserve tool/agentic schemas and validate trajectories

**Files:**
- Modify: `examples/dataset/conversation_utils.py`
- Create: `examples/dataset/trajectory_schema.py`
- Create: `tests/examples/dataset/test_trajectory_schema.py`

- [ ] Add golden examples for OpenAI `tool_calls`/`tool_call_id`, R2E-Gym `id`, top-level `tools`, reasoning content, and multi-turn tool results.
- [ ] Implement canonicalization without dropping top-level tools, assistant calls, tool names/results, IDs, or reasoning.
- [ ] Reject dangling/duplicate tool-call IDs and invalid role transitions with source/row diagnostics.
- [ ] Produce a stable trajectory/content digest independent of input column order.
- [ ] Run: `pytest -q tests/examples/dataset/test_trajectory_schema.py`.

## Task 3: Build deterministic token-budgeted B/C/D corpus plans

**Files:**
- Create: `examples/dataset/build_specdec_study_corpus.py`
- Create: `examples/dataset/qwen3_4b_hybrid_study.yaml`
- Create: `tests/examples/dataset/test_build_specdec_study_corpus.py`

- [ ] Define pinned source revisions and B/C/D weights from the approved design.
- [ ] Tokenize with the exact pinned Qwen3-4B tokenizer and count trainable assistant tokens, total context tokens, prompts, and tool turns.
- [ ] Deterministically shuffle within each source before quota selection; never use lexical `.take()` prefixes.
- [ ] Enforce category minimums, context buckets (`<=4K`, `4-16K`, `16-32K`), train/eval task deduplication, and at most two epochs of exposure.
- [ ] Emit a plan-only JSON manifest before materialization, including revisions, licenses, seed, hashes, counts, token histograms, and unique-vs-exposure tokens.
- [ ] Materialize a small number of 1--4 GiB Parquet/Zstd shards via atomic sibling directories and per-file SHA-256 manifests.
- [ ] Add fixtures proving exact quota arithmetic, deterministic output, no source-order bias, and contamination rejection.
- [ ] Run: `pytest -q tests/examples/dataset/test_build_specdec_study_corpus.py`.

## Task 4: Add target-synthesis and trace-replay manifests

Before Slurm integration, extend the streaming loader to accept validated
pretokenized `input_ids`/`token_ids` plus a binary, length-aligned `loss_mask`.
The loader must truncate IDs and masks together, derive a stable sample ID, skip
samples whose supervised span is fully truncated, and never call the tokenizer
for pretokenized rows. This is required to preserve tool-aware target
serialization and make the assistant-token budget measurable.

**Files:**
- Create: `tools/launcher/common/specdec/build_qwen4b_synthesis_manifest.py`
- Create: `tools/launcher/common/specdec/run_qwen4b_synthesis.sbatch`
- Create: `tools/launcher/tests/test_qwen4b_dataset_study.py`

- [ ] Model separate immutable `thinking-on` and `thinking-off` synthesis namespaces over identical prompt IDs.
- [ ] Reference non-tool prompts for target synthesis and validated tool trajectories for explicit `trace-replay`; never silently treat replay as target-native execution.
- [ ] Pin target snapshot/revision, tokenizer hash, source manifest hash, container hash, ModelOpt commit, generation parameters, and output token budget.
- [ ] Launch four TP1 Qwen3-4B replicas on each four-GPU serving node and shard requests deterministically across replicas.
- [ ] Use node-local `/raid/scratch/$USER/$SLURM_JOB_ID` for caches/incomplete shards and atomic Lustre publication for completed outputs.
- [ ] Add `sbatch --test-only`/fake-Slurm tests for GPU ownership, ports, cleanup, requeue safety, and content-addressed resume.
- [ ] Run: `pytest -q tools/launcher/tests/test_qwen4b_dataset_study.py` and `shellcheck -S warning tools/launcher/common/specdec/run_qwen4b_synthesis.sbatch`.

## Task 5: Add study training manifests and full-GPU topology

**Files:**
- Create: `tools/launcher/common/specdec/qwen4b_study_manifest.py`
- Create: `tools/launcher/common/specdec/run_qwen4b_study_training.sbatch`
- Extend: `tools/launcher/tests/test_qwen4b_dataset_study.py`

- [ ] Encode three thinking-on DFlash-B8 arms with identical initialization, optimizer, schedule, seed, sequence length 4096, and exact assistant-token stopping criteria.
- [ ] Encode Phase A as two nodes per arm: one serve node with four TP1 replicas, one trainer node with four DDP ranks, per-device batch four, accumulation eight, GBS128.
- [ ] Encode 20-step canary and checkpoints at 64M/128M/256M assistant tokens; preserve restart checkpoints and immutable milestone exports separately.
- [ ] Require cluster profile/readiness receipts and the selected account; reject `nemotron_n3_post` for study training.
- [ ] Supervise all child processes, fail on any replica/rank death, atomically publish rendezvous files, and clean stale state before requeue.
- [ ] Verify all four GPUs per node have an owning process and capture DCGM SM activity for the canary gate.
- [ ] Add tests for batch arithmetic, all-GPU ownership, no Q30/Q235 production identity collision, save/requeue behavior, and readable job names containing target/mode/arm/method/B/topology.
- [ ] Run the focused launcher suite plus Bash/ShellCheck/Ruff checks.

## Task 6: Add evaluation and selection reporting

**Files:**
- Create: `tools/launcher/common/specdec/build_qwen4b_study_eval_manifest.py`
- Create: `tools/launcher/common/specdec/run_qwen4b_study_eval.sbatch`
- Create: `reports/qwen3-4b-hybrid-study/index.html`
- Extend: `tools/launcher/tests/test_qwen4b_dataset_study.py`

- [ ] Evaluate B/C/D with identical target, TP, CUDA Graph, temperature, request budgets, and concurrency 1/8/32.
- [ ] Report acceptance length/rate, throughput and latency speedup for SPEED categories, math/code suites, held-out SWE/tool traces, and context buckets.
- [ ] Separate native-K results from any same-K K5/K7 comparisons.
- [ ] Generate a self-contained HTML report with corpus composition, provenance, learning curves, GPU utilization, W&B/job/log links, and explicit preliminary/final decision gates.
- [ ] Validate JSON/CSV schemas and render the HTML locally before publication.

## Task 7: Verify, commit, push, and run the OCI canary

- [ ] Run all new focused tests, related existing dataset/launcher tests, Ruff check/format, Bash syntax, ShellCheck, and `git diff --check`.
- [ ] Request independent code review and resolve all Critical/Important findings.
- [ ] Create a signed DCO commit containing only the study changes and push the feature branch.
- [ ] On OCI-HSG, pull the exact pushed SHA into a clean checkout.
- [ ] Verify `nemotron_sw_post` association/FairShare and backfill permission; generate an immutable external cluster profile/readiness receipt.
- [ ] Stage pinned Qwen3-4B and PTV3 assets with manifests and checksums.
- [ ] Run `sbatch --test-only`, submit the 20-step canary, monitor at least five minutes, and require checkpoint/export plus non-idle DCGM evidence.
- [ ] Use measured synthesis/training throughput to render exact 256M projections.
- [ ] Only after the gate passes, submit B/C/D 256M arms and publish the initial report.
