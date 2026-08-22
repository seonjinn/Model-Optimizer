# Qwen3-4B Hybrid Dataset Study Design

## Goal

Determine a reproducible Nemotron Post-Training v2/v3 mixture for speculative
drafter training before paying the cost of target-specific synthesis and full
training for Qwen3-30B-A3B and Qwen3-235B-A22B.

The study uses the hybrid `Qwen/Qwen3-4B` target so the same weights can produce
separate thinking-on and thinking-off corpora. The primary decision concerns
the data mixture, not the final drafter architecture or production token
budget.

## Scope

The first decision compares three dataset constructions:

- **B:** balanced PTV2 only
- **C:** 75% balanced PTV2 and 25% balanced PTV3
- **D:** 50% balanced PTV2 and 50% balanced PTV3

The percentages refer to trainable assistant tokens, not rows, serialized
bytes, prompts, or total context tokens. The existing lexically truncated PTV2
1.3M view remains an external reference and is not retrained as a fourth arm.

Phase A uses thinking-on target responses and DFlash B8. Only the strongest
mixtures continue to thinking-off and DSpark confirmation. B16 and large-model
training are outside the screening phase.

## Primary Cluster

OCI-HSG is the canonical data-construction and screening cluster.

- The complete PTV2 mirror is already available locally: 44.42 GB across 613
  files.
- The user has approximately 249.3 TiB of byte quota and 27.3 million inodes
  remaining.
- PTV3 and `Qwen/Qwen3-4B` are not staged yet and must be pinned before the
  first canary.
- Durable data lives under:

  ```text
  /lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sna/
    modelopt-specdec/dataset-studies/qwen3-4b-hybrid-v1
  ```

Source and small configuration files live under `/home`. Hugging Face caches,
temporary databases, shuffle files, locks, and incomplete shards live under
`/raid/scratch/$USER/$SLURM_JOB_ID`. The durable output uses a small number of
1--4 GiB Parquet/Zstd shards to avoid inode pressure.

Only the selected corpus and its manifest are copied to Lyris or Ptyche. Those
clusters do not independently regenerate the same responses.

## Data Construction

### Balanced PTV2 pool

The PTV2 pool uses these assistant-token quotas:

| Category | Weight |
|---|---:|
| Math | 25% |
| Code | 25% |
| STEM | 20% |
| Chat and instruction following | 20% |
| Multilingual | 10% |

Sampling is deterministic within each source split before applying its token
quota. It must not use a lexical prefix or cap rows before shuffling.

### Balanced PTV3 pool

The PTV3 pool uses these assistant-token quotas:

| Category | Weight |
|---|---:|
| SWE | 20% |
| Agentic and tool use | 25% |
| General and competitive code | 15% |
| Math | 20% |
| Science and reasoning | 10% |
| Chat and instruction following | 5% |
| Multilingual | 5% |

Each top-level category contributes at least 2,000 unique training prompts when
the source contains enough valid examples. Each populated category/context
cell contributes at least 500 prompts. Context buckets are at most 4K, 4--16K,
and 16--32K tokens. The initial trainer still consumes 4K sequences; longer
examples are retained as explicit later-context buckets rather than silently
right-truncated into the 4K pool.

### Response sources

Non-tool prompts are regenerated with the exact hybrid Qwen3-4B target.
Thinking-on and thinking-off are separate immutable generation runs over the
same prompt IDs.

The balanced PTV2 and PTV3 response pools are generated once per thinking mode.
B, C, and D reference deterministic subsets of those pools; they do not
regenerate identical responses for each arm.

PTV3 tool trajectories require a separate path. The current generic generator
filters tool-role examples and cannot execute tools. The first study therefore
uses two clearly labelled sources:

- target-synthesized responses for non-tool prompts;
- schema-preserving trace replay for valid recorded tool trajectories.

The trace representation preserves top-level tool schemas, assistant
`tool_calls`, tool-call IDs, tool results, message order, and reasoning content.
It never converts a tool trajectory to role/content-only chat. A later
environment-backed NeMo-Gym or OpenHands collector replaces trace replay with
target-native executed trajectories; that later collector is not part of the
four-hour pilot.

## Experiment Funnel

### Phase 0: readiness and canary

1. Pin exact revisions for Qwen3-4B and every PTV3 source.
2. Build immutable prompt-pool manifests with per-source counts, assistant-token
   quotas, context histograms, licenses, revisions, and checksums.
3. Generate bounded thinking-on and thinking-off smoke shards.
4. Run one 20-step single-node DFlash B8 canary.
5. Measure every GPU's utilization, peak memory, target-server queueing,
   training throughput, and projected 256M-token runtime.

The production study is submitted only if the canary has no idle GPU for the
observed training window, creates a complete checkpoint and export, and projects
an acceptable wall time.

The current generator cannot satisfy this contract unchanged. It alternates
thinking mode by shard parity, excludes tool-role rows, limits total length to
8K by default, and writes row-oriented JSONL without the required token-budget
manifest. The implementation replaces parity with an explicit `on` or `off`
mode and adds the schema-preserving, token-budgeted output contract before any
study job is submitted.

### Phase A: thinking-on mixture comparison

Train B, C, and D concurrently with identical settings:

- target: `Qwen/Qwen3-4B`
- `enable_thinking=true`
- method: DFlash
- block size: 8
- sequence length: 4096
- global batch size: 128
- identical optimizer, learning-rate schedule, seed, and initialization
- checkpoints at 64M, 128M, and 256M trainable assistant tokens

All three arms reach 256M tokens. The intermediate checkpoints provide learning
curves and reveal ranking crossovers; 64M is not treated as the final result.

At sequence length 4096, GBS 128 provides roughly 488 full-sequence-equivalent
optimizer updates over 256M tokens. GBS 512 would provide only roughly 122 and
is therefore not used for the primary mixture screen.

### Phase A2: multi-billion-token comparison

Continue B, C, and D to 4B trainable assistant tokens with GBS 512. Evaluate at
1B, 2B, and 4B so the report can distinguish an early advantage from a stable
learning-curve advantage. No arm is eliminated at 256M solely because it ranks
third there.

An arm may be eliminated after 4B only when it trails consistently at 1B, 2B,
and 4B and the acceptance gap at 4B is at least 5% relative. Otherwise all
three remain eligible for the final stage.

### Phase A3: final mixture comparison

Continue the top two Phase A2 mixtures to 20B trainable assistant tokens with
GBS 512. Evaluate at 8B, 12B, and 20B. The final mixture decision uses the
endpoint and per-category learning curves, not only an aggregate score.

The report separates unique target-synthesized tokens from cumulative training
exposure. Repeating a small pool does not count as additional coverage. The
default maximum is two epochs over a pool, and any repetition is explicit.

### Phase B: thinking-off transfer

Regenerate the same prompt IDs with `enable_thinking=false`. Train only the top
two Phase A mixtures to 128M tokens. Extend both to 256M if their ranking differs
from the thinking-on result or the confidence interval overlaps.

### Phase C: method transfer

Train DSpark B8 on the top two mixtures for 128M tokens in thinking-on mode.
Extend to 256M only if the data ranking is unclear. The 4B and 20B DFlash stages
already use production-like GBS 512, so no separate DFlash batch-size
confirmation is required.

### Escalation rule

The 256M stage provides a preliminary signal only. All three arms continue to
4B. At 4B, eliminate only an arm that meets the stable-margin rule above. The
remaining top two continue to 20B for the final small-model mixture decision.

## GPU Topology and Batch Arithmetic

The 20-step canary may use one four-GPU GB200 node per arm:

- GPUs 0--1: two Qwen3-4B TP1 serving replicas;
- GPUs 2--3: two DDP trainer ranks;
- per-device batch size 4;
- gradient accumulation 16;
- `4 * 16 * 2 = 128` global batch size.

Every allocated GPU has an owning process. This avoids the previous idle-reaper
failure where a single TP2 server used only two GPUs of a four-GPU serving node.
The canary must still confirm actual SM activity because process placement alone
does not prove useful work.

The target server starts with BF16, `max_model_len=8192`, and a conservative
GPU-memory utilization near 0.50. If the canary shows adequate headroom,
per-device batch size may increase to 8 while accumulation decreases to 8,
preserving GBS 128.

Phase A uses two nodes per arm, six nodes total:

- one serving node with four TP1 replicas;
- one trainer node with four DDP ranks;
- per-device batch size 4 and accumulation 8;
- `4 * 8 * 4 = 128` global batch size.

Phase A2 uses four nodes per arm, twelve nodes total. Two serving and two trainer
nodes give eight trainer ranks; per-device batch size 4 and accumulation 16 give
GBS 512. Phase A3 uses eight nodes per arm for the top two mixtures, sixteen
nodes total. Four serving and four trainer nodes give sixteen trainer ranks;
per-device batch size 4 and accumulation 8 preserve GBS 512.

Each serving node launches four TP1 replicas and each trainer node launches four
ranks, so no allocated GPU is intentionally unused at any scale. Actual
utilization remains a canary gate.

## Time Budget

Two to four hours is a target for an already-staged Phase A training and bounded
evaluation wave, not for downloading PTV3, generating every possible response,
and training over the complete collections. The study scans source metadata but
synthesizes only the deterministic token-budgeted pools needed by B, C, and D.

The canary measures generation and training tokens per second and projects each
stage independently. The first preliminary result, including initial staging
and bounded target synthesis, has a roughly one-day planning horizon. Phase A2
and Phase A3 are multi-day work. Their exact duration is not promised until
measured synthesis and training throughput are available; response synthesis
may dominate training time. The study never truncates a stage merely to meet a
four-hour headline.

## Evaluation

The report includes:

- SPEED-Bench acceptance length and acceptance rate by category;
- GSM8K, MATH-500, and AIME trajectory acceptance;
- HumanEval, MBPP, and LiveCodeBench trajectory acceptance;
- held-out SWE and agentic trace acceptance;
- results by context bucket;
- accepted tokens per proposed token and verifier calls per output token;
- CUDA-graph-enabled end-to-end throughput at concurrency 1, 8, and 32;
- peak memory and GPU utilization for every allocated GPU.

DFlash and DSpark are compared with the same proposed draft-token count. A K
sweep of 3, 5, and 7 is run only on the final winner.

Speculative decoding is lossless, so Math or SWE correctness must match the
target within deterministic evaluation tolerance. The selection signal is
acceptance and latency, not a claimed improvement in target solve rate.

The PTV3 mixture is selected when it improves held-out SWE acceptance by at
least 10% over balanced PTV2, reduces no major Math or SPEED category by more
than 3%, and maintains the advantage through the 4B and 20B checkpoints.

## Large-Model Transfer

The winning prompt selection, source weights, context buckets, deduplication,
and split manifests transfer to Qwen3-30B-A3B and Qwen3-235B-A22B. Generated
assistant responses do not transfer.

Each exact target checkpoint receives its own corpus:

- Q30 Base;
- Q30 Thinking;
- Q235 Base;
- Q235 Thinking.

DFlash and DSpark share the corpus generated by the same target. B8 and B16 can
also reuse it. Before a tens-of-billions-token run, the winner and runner-up are
rechecked on Q30 Thinking for 64--128M tokens. This guards against a mixture
ranking that does not transfer from 4B scale.

## Storage and Publication

The 256M study reserves 200--350 GB including working headroom. The 4B and 20B
stages reserve up to 1 TiB for immutable pools, checkpoints, atomic partials,
and reports. Exact projections come from smoke shards before any large
generation submission. OCI has sufficient byte quota, but the implementation
still uses few large shards and enforces a bounded inode budget.

The durable hierarchy contains `manifests`, `prompt-pools`, separate
`target-synth/thinking-on` and `target-synth/thinking-off`, `training`, `eval`,
and `report`. Job-unique partial directories are atomically published only after
row counts, token counts, and checksums pass. Completed immutable data is never
overwritten.

A GitLab Pages report records:

- every source revision, split, license, and preprocessing decision;
- exact category and context distributions by rows and assistant tokens;
- thinking-on/off response generation settings;
- tool-trace limitations and provenance;
- model, runtime, container, source, and config hashes;
- job IDs, W&B links, checkpoints, and failure receipts;
- learning curves and benchmark tables for B/C/D;
- the selection decision and its thresholds.

The report links to manifests and small evaluation artifacts. Models, datasets,
checkpoints, credentials, and large logs are not committed to Git.

## Failure Handling

- Missing source revision, model, runtime, or checksum: stop before GPU launch.
- Thinking mode mixed within one corpus: reject the manifest.
- Tool-call or tool-result referential-integrity failure: quarantine the row and
  report counts; never silently flatten it.
- Empty category or insufficient context bucket: fail the mixture build.
- Incomplete shard: leave it outside the immutable final namespace and resume
  only from its explicit receipt.
- Any idle GPU or fatal training error in the canary: do not submit Phase A.
- Scheduler timeout: save a complete checkpoint, self-requeue within the
  configured limit, and retain the same experiment identity.

## Verification

Local tests must cover thinking-mode isolation, deterministic token quotas,
tool-schema preservation, prompt deduplication, context bucketing, manifest
hashing, atomic Parquet publication, topology arithmetic, and report generation.
Ruff, ShellCheck, formatting, focused pytest, and diff checks must pass before a
signed/DCO commit is pushed.

OCI execution then requires, in order:

1. immutable asset and profile validation;
2. `sbatch --test-only` for staging, synthesis, canary, training, and evaluation;
3. a 20-step canary with complete checkpoint/export and per-GPU telemetry;
4. a fresh test-only pass for all three Phase A jobs;
5. at least five minutes of monitored healthy execution after they start.
