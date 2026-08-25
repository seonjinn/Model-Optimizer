# Qwen3-4B Balanced Dataset Pilot Design

## Goal

Produce a fast, reproducible PTV2-only dataset comparison for Qwen3-4B while the larger authenticated PTV2/PTV3 study continues independently. The pilot compares a deduplicated historical-proportion control with a deduplicated category-balanced arm using the same source revision, seed, row count, target model, DFlash configuration, and assistant-token training budget.

## Scope

The first pilot uses only `nvidia/Nemotron-Post-Training-Dataset-v2`. It does not wait for PTV3 source staging, Task9 A/B completion, target-response regeneration, or the exact-256M A/B/H materializer. It does not admit tool trajectories because the existing raw streaming loader does not preserve the complete tool schema.

Both arms contain exactly 100,000 rows and use the previously approved seed `20260822`:

| Arm | Category quotas |
|---|---|
| `historical-proportion` | chat 48,286; math 18,420; code 13,462; stem 19,832 |
| `balanced` | math 25,000; code 20,000; stem 25,000; chat 20,000; multilingual 10,000 |

The balanced quota is an exact 1/20 scale of the approved 2M PTV2 B-balanced policy. Its multilingual quota is exactly 2,000 rows from each of `multilingual_de`, `multilingual_es`, `multilingual_fr`, `multilingual_it`, and `multilingual_ja`.

The historical-proportion quotas are the deterministic largest-remainder projection of the authenticated 1.3M historical counts: chat 627,720; math 239,467; code 175,000; stem 257,813. This arm is a deduplicated control with historical proportions, not a byte-for-byte replay of the historical occurrence stream.

## Relationship to the Approved PTV3 Study

The simplified PTV2 pilot does not redefine or replace the approved B/C/D study:

- B uses 100% balanced PTV2.
- C uses 75% balanced PTV2 and 25% PTV3.
- D uses 50% balanced PTV2 and 50% PTV3.

PTV3 SWE rows must come from the authenticated PTV3 sources; PTV2 rows are never relabeled or substituted as SWE. For the scaled PTV3 component, the latest strict category policy is retained: 30% `swe-agentic-tool`, 20% math, 10% code, 20% stem/science, 15% multilingual, and 5% instruction/chat. D divides its `swe-agentic-tool` allocation equally among agentless SWE, interactive SWE replay, and generic tool replay.

Phase 1 builds and trains the 100K PTV2 historical-proportion and B-balanced arms immediately. Phase 2 adds scaled C/D arms only after the PTV3 source inventory and selection receipts pass; the fast path must not weaken or bypass that source gate.

## Selection Contract

Each source split is pinned to one exact Hugging Face repository revision. Rows are normalized without augmentation. Rows without a terminal assistant response, rows containing tool roles or tool calls, evaluator-held-out prompt UUIDs, and malformed conversations are excluded before selection.

The canonical prompt UUID is computed from the prompt context before the terminal assistant response. Exact duplicates are removed globally within an arm. Selection order is the ascending SHA-256 rank of canonical JSON containing `seed`, source split, and prompt UUID, with prompt UUID as the deterministic tie break. After filtering, each declared category must fill its exact quota; shortages fail closed and are never silently redistributed.

Where the two arms draw from the same split, the same deterministic ranking is used. This makes the smaller quota a prefix of the larger quota and maximizes paired overlap without sacrificing either arm's declared composition.

## Output Contract

One 96-CPU OCI-HSG `cpu_datamover` job builds both arms. Source and scripts remain under `/home`; Hugging Face cache, sorting state, and tokenization scratch remain under `/raid/scratch`; only the immutable final bundles are published under `/lustre`.

Each arm publishes:

- `data.jsonl`, containing normalized `messages` and no tool records;
- `MANIFEST.json`, binding source repository/revision, seed, exact quotas, row count, category counts, prompt UUID digest, data file size/SHA-256, tokenizer identity, exact assistant-token count, exclusion counts, and producer source commit;
- `EXECUTION.json`, recording the SLURM allocation, requested/effective CPU workers, timing, and runtime image identity without changing scientific identity.

The parent publishes `COMPLETE.json`, binding the two arm manifests. Publication is private-scratch-first, replay-validated, atomic no-replace, and fails without a durable final root on any worker or validation error.

## Training Contract

The existing Qwen3-4B study runner and submitter are reused. The study manifest gains descriptive pilot arm names but retains the existing two-node topology: one GB200 node with four TP1 target servers, one GB200 node with four trainer ranks, DFlash B8/K7, sequence length 4096, answer-only loss, and training seed 42.

Both arms first run the existing 20-step canary gate. Passing canaries advance to the same exact assistant-token budget of 16,000,000 tokens. Dataset publication fails if either arm contains fewer than 16,000,000 trainable assistant tokens. A typed `pilot-row-quota-v1` selection mode permits a pilot corpus to contain more tokens than the common target while the existing B/C/D equality rule remains unchanged. Pilot GPU jobs use the user-selected OCI-HSG account `nemotron_n3_post`; existing non-pilot study account validation remains unchanged. Checkpoints, exports, W&B identity, source commit, dataset manifest, target revision, container, and runtime archive remain receipt-bound by the existing study workflow.

The first conclusion is directional: matched training-loss/accuracy traces at the same assistant-token milestones plus a fixed held-out acceptance evaluation of both exported drafters. DSpark replication and larger PTV3 mixtures are follow-ups only if the balanced DFlash arm shows a meaningful signal.

## Validation

Tests must prove exact quotas and total count, deterministic seed behavior independent of input order and worker count, global exact dedup with quota refill, held-out exclusion, malformed/tool rejection, quota shortage failure, exact assistant-token accounting, output tamper rejection, atomic failure cleanup, and no-replace behavior. Launcher tests must prove the 96-CPU CPU job contract and that the existing trainer receives the exact pilot arm identity and 16M budget.
