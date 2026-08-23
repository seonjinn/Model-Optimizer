# B-prime, C, and D Prompt Preparation Design

## Decision and purpose

Prepare the prompts now, while the existing Nemotron-v2 drafter jobs continue.
The prepared artifacts support two related but distinct experiments:

1. **B-prime (`B′`)** continues an existing drafter with 700,000 previously
   unseen PTV2 prompts, bringing the named source-row exposure from 1.3M to
   2.0M.
2. **C and D** are alternative 2,000,000-prompt balanced views. They use the
   same top-level domain counts; D changes only the response source within the
   SWE/Agentic/Tool lane.

This document supersedes the B/C/D row counts and domain weights in
`2026-08-22-ptv2-complement-ptv3-agentic-design.md`. The older document remains
the source for the canonical record, trajectory-integrity, atomic publication,
and cross-cluster transfer contracts unless this document explicitly changes
them. In particular, the checked-in `25/25/20/20/10` B configuration and
`35/25/15/15/5/5` C/D configuration are not production-ready for this decision.

Approval of this design authorizes prompt preparation and validation only. It
does not itself authorize a production training submission, cancellation, or
replacement of a running job.

## Audited 1.3M lineage

The existing OCI-HSG, Ptyche, and Lyris `nemo` drafter families consume the
same prefix-selected PTV2-compatible corpus. The audited physical source has
1,309,377 row occurrences. The actual training selection is the first
1,300,000 occurrences in Hugging Face streaming order; 9,377 German-tail
occurrences lie beyond that boundary.

| Split | Selected occurrences | Share |
|---|---:|---:|
| Chat | 627,720 | 48.2862% |
| Code | 175,000 | 13.4615% |
| Math | 239,467 | 18.4205% |
| German | 257,813 | 19.8318% |
| STEM | 0 | 0% |
| Japanese, Spanish, French, Italian | 0 | 0% |
| Explicit SWE/Agentic/Tool | 0 | 0% |
| **Total** | **1,300,000** | **100%** |

The 1,300,000 occurrences contain 931,363 unique canonical prompts. Duplicate
occurrences are part of the historical training stream and must remain in the
lineage receipt; they are not an audit error. The exclusion set is the 931,363
unique canonical prompt UUIDs. Reports must therefore say “1.3M source-row
occurrences,” never “1.3M unique prompts.”

The baseline receipt binds:

- PTV2 revision `5c89e01dd720ae0f4058445ed49c5fb68a03c76e`;
- the ordered physical files, resolved symlink targets, byte sizes, and SHA-256
  digests;
- the streaming selection policy and exact boundary row;
- all 1,300,000 occurrences and the 931,363-UUID exclusion set;
- the split histogram above and the excluded 9,377-row tail.

A complement build stops if any of these values fails to reproduce.

## B-prime: exact 700K PTV2 complement

`B′` is the immediate continuation dataset for the existing drafter families.
It contains exactly 700,000 unique prompts after baseline, held-out, and
within-build UUID exclusion.

| PTV2 source lane | Prompt quota |
|---|---:|
| STEM | 200,000 |
| Japanese | 125,000 |
| Spanish | 125,000 |
| French | 125,000 |
| Italian | 125,000 |
| German | 0 |
| **Total** | **700,000** |

This quota deliberately does not add more Chat, Math, or Code. The audited
streaming boundary proves that the available PTV2 files for those three
domains were already fully selected by the 1.3M prefix. It also does not add
German: German already accounts for 257,813 occurrences, while the four other
audited languages account for zero.

Each cell is selected independently from a deterministic hash order. A
shortfall is fatal: it is not filled from German, another language, or another
domain, and the weights are never renormalized. The selected source completion
is stripped and the exact target checkpoint regenerates the assistant response.
Qwen3-4B, Qwen3-30B-A3B, and Qwen3-235B-A22B, and thinking-on versus
thinking-off, remain separate immutable response families.

The phrase “2.0M” means historical 1.3M row occurrences plus a new 700K-prompt
continuation. Because the historical prefix contains duplicates, it does not
mean two million unique prompts. The report presents both occurrence exposure
and unique-UUID coverage.

## C and D: exact 2M prompt views

C and D each contain exactly 2,000,000 unique prompt UUIDs and have identical
top-level prompt counts.

| Domain | Share | C prompts | D prompts |
|---|---:|---:|---:|
| SWE / Agentic / Tool | 30% | 600,000 | 600,000 |
| Math / Reasoning | 20% | 400,000 | 400,000 |
| Code | 10% | 200,000 | 200,000 |
| STEM / Science | 20% | 400,000 | 400,000 |
| Multilingual | 15% | 300,000 | 300,000 |
| Instruction / Chat | 5% | 100,000 | 100,000 |
| **Total** | **100%** | **2,000,000** | **2,000,000** |

C uses target synthesis for all 600,000 SWE/Agentic/Tool prompts. D keeps the
same 600,000-prompt domain quota but divides it equally by response lane:

| D lane | Share of all prompts | Prompt quota |
|---|---:|---:|
| Agentless SWE target synthesis | 10% | 200,000 |
| Interactive SWE recorded replay | 10% | 200,000 |
| Generic agentic/tool recorded replay | 10% | 200,000 |
| **SWE/Agentic/Tool total** | **30%** | **600,000** |

The other 1.4M prompt UUIDs are shared between C and D whenever lane semantics
allow it. This paired construction controls prompt coverage; only the D replay
lanes intentionally differ. A trajectory is a prompt view plus its complete
recorded interaction, not a flattened collection of turns.

PTV2 and PTV3 may both supply a domain, but there is no implicit “take the
first rows” policy. Within every cell, candidates are deduplicated, assigned a
full-context bucket, and ordered by a seeded hash over the dataset policy,
source identity, source row, and canonical prompt UUID.

## Prompt counts and assistant-token exposure are separate contracts

The 700K and 2M values are exact prompt-view counts. They are inventory and
coverage targets, not substitutes for training-token accounting. Response
lengths differ substantially across Chat, Math, SWE, and replay trajectories.

The pipeline publishes three layers:

1. **Candidate reserve:** at least 20% more candidates than the requested count
   in every populated cell, after UUID and held-out exclusion.
2. **Prompt view:** exactly 700K B′ prompts or exactly 2M C/D prompts with the
   row quotas above. Prompt selection is frozen before generation.
3. **Training exposure views:** target-specific responses ordered
   deterministically within each domain/lane. Each evaluation view stops at an
   exact cumulative assistant-loss-token boundary by masking only the final
   assistant-owned tokens. User, system, developer, and tool-result tokens,
   padding, and masked truncation do not count.

The fixed evaluation boundaries are 256M and 1B assistant-loss tokens, followed
by the largest common exact-token boundary supported by both C and D. A “full
2M-prompt pass” is reported separately because C and D can have different
total assistant-token counts. It is not treated as an equal-exposure comparison
unless the common exact-token boundary is used.

For B′, publish both a one-pass 700K-prompt receipt and 256M/one-billion-token
receipts when the regenerated corpus is large enough. This makes the practical
“2.0M samples” milestone visible without confusing it with equal-token science.

## Context coverage

Context buckets are derived from the complete serialized conversation before
any training truncation:

- `le4k`: at most 4,096 tokens;
- `4k_16k`: 4,097 through 16,384 tokens;
- `16k_32k`: 16,385 through 32,768 tokens.

Every populated domain/lane must have a deterministic bucket manifest. The
inventory records prompt and assistant-token histograms by bucket. It does not
trust a source-provided “long context” label. A separate 4K training view is
derived from full context. Any truncation that splits a tool-call/result
transaction is rejected rather than repaired. Examples beyond 32K are counted
in a rejection receipt and are not silently moved into another bucket.

No fixed bucket percentages are asserted before the source inventory proves
capacity. The readiness report proposes bucket floors from available data; the
same floors are then frozen for C and D. A missing bucket stops publication if
it would make the paired views differ.

## Canonical exclusion and held-outs

The canonical prompt UUID is SHA-256 over deterministic JSON of the
prompt-bearing conversation. Canonicalization preserves roles, content,
message order, top-level tool schemas, tool calls, and tool-call IDs, while
removing storage-only fields.

Selection order is:

1. validate the pinned source row and its license metadata;
2. compute the canonical prompt bytes and UUID;
3. exclude the historical 931,363-UUID set;
4. exclude the union of SPEED-Bench, Math, Code, SWE, multilingual, and
   tool/agentic evaluator UUIDs;
5. reject cross-source and within-view duplicates;
6. assign domain, lane, language, and full-context bucket;
7. apply deterministic seeded selection to exact prompt quotas.

A repeated UUID with identical canonical bytes is a counted duplicate. The
same UUID with different canonical bytes is a fatal collision. Held-out
manifests are content-addressed inputs to dataset identity; an evaluator change
therefore cannot silently reuse a contaminated corpus.

## Target regeneration and replay semantics

Target-synthesis rows contain prompt messages only at selection time. The
source assistant completion is never used as the drafter label. The exact
target model and tokenizer regenerate the response with pinned chat template,
thinking mode, generation parameters, maximum output length, stop policy,
runtime/container digest, and retry identity.

Generation uses a 20% per-cell reserve. Only complete, schema-valid responses
enter the frozen prompt view. A retry keeps the same prompt and generation
identity. Two different successful payloads for the same identity are fatal.
The final prompt view is a deterministic selection from successful outputs;
failed generations and unused reserve rows remain in receipts but not in the
training corpus.

Replay is recorded source-native behavior. It must preserve top-level tool
schemas, assistant tool calls, function names and arguments, call IDs, tool
results, message order, and recorded reasoning. Every tool result resolves one
preceding open call, every called function is declared, and no call remains
unresolved. Declarations-only rows, orphan results, duplicated IDs, malformed
arguments, flattened conversations, and invented thinking-mode labels are
quarantined and contribute neither prompt nor token quota.

## Pinned source inventory and license gates

The currently staged Qwen3-4B PTV3 subset provides these exact source pins.
They are candidate inputs, not evidence that the 2M quotas are satisfiable.

| Source | Revision | Recorded license expression | Intended use |
|---|---|---|---|
| `nvidia/Nemotron-SFT-SWE-v2` | `bd151f3f2d89c4804dda0083d912bd9f6a0a9fb7` | CC-BY-4.0; Apache-2.0; MIT; BSD-3-Clause; BSD-2-Clause | Agentless SWE prompts |
| `nvidia/Nemotron-Agentic-v1` | `650d590978ca35c8f1ecea2faf136e5fac421b62` | CC-BY-4.0; Apache-2.0 for Glaive-derived records | Generic agentic/tool candidates |
| `nvidia/Nemotron-SFT-Competitive-Programming-v2` | `778afc98a9e027e10b3cd78020c120e93e142ef2` | CC-BY-4.0; ODC-By; MIT | Code |
| `nvidia/Nemotron-Math-v2` | `8e793210e175b6406c752a870f585f62de98c0d3` | CC-BY-4.0; CC-BY-SA-4.0 | Math |
| `nvidia/Nemotron-Science-v1` | `82e1af468197076b4f0f392c239274eac032adc7` | CC-BY-4.0 | Science |
| `nvidia/Nemotron-SFT-Instruction-Following-Chat-v2` | `1a9454ed054b8544503ab8d8c0a519d141a44c5b` | ODC-By | Chat |
| `nvidia/Nemotron-SFT-Multilingual-v1` | `22c86505762a7c595abee309d720084351c9f4ba` | CC-BY-4.0; CC-BY-SA-4.0 | Multilingual |

Every admitted physical file additionally requires its own SHA-256 and byte
count. Composite license expressions are preserved per record and per shard;
they are not simplified by the builder. A source without an exact revision,
file digest, license expression, or allowed-use review remains ineligible.

The production inventory is still missing or unproven in these areas:

- no pinned, validated interactive-SWE replay source currently supplies the
  required 200,000 D prompts;
- the currently selected generic tool replay has only a small validated token
  inventory and has not proved 200,000 unique valid prompt views;
- the currently pinned one-file Math, Code, Science, Chat, and Multilingual
  subset has not proved the new per-cell unique-count quotas after exclusions;
- the broader repositories listed in `nemotron_ptv3_datasets.yaml` do not pin
  revisions or licenses and cannot be used directly as a production manifest;
- per-target 20%-reserve generation capacity and full-context bucket floors
  have not yet been proved.

Candidate expansion may use additional Nemotron v2/v3 sources only after
recording repository, configuration, split, exact commit, license expression,
physical-file SHA-256, and held-out policy. A new source changes dataset
identity. Missing capacity is never filled by an unpinned dataset.

## Reusable implementation and required changes

The existing work is reusable:

- `audit_ptv2_baseline.py` reconstructs the streaming `take(1_300_000)`
  boundary, preserves duplicate occurrences, resolves nested message schemas,
  and emits a content-bound receipt;
- `specdec_identity.py` provides deterministic prompt UUIDs and exclusion
  indexes;
- `trajectory_schema.py` provides fail-closed tool-trajectory validation;
- `build_specdec_inventory.py` pins source identity, licenses, context, token
  IDs, and loss masks;
- `build_specdec_study_corpus.py` already supports deterministic ordering,
  exact assistant-token selection, final-token masking, and atomic corpus
  publication;
- `ptv23_builder_readiness.py` binds the audit and PTV3 manifests and reports
  missing source capacity without renormalization.

The current implementation does not yet implement this approved prompt design.
Before building production artifacts it must add:

- exact prompt-count contracts for B′ `200K/125K/125K/125K/125K`;
- exact 2M C/D count contracts `600K/400K/200K/400K/300K/100K`;
- D's `200K/200K/200K` lane contract;
- a distinction between historical occurrences, unique UUIDs, prompt-view
  counts, and assistant-loss-token exposure;
- 20% reserve accounting and deterministic promotion of successful responses;
- paired C/D non-agentic UUID manifests and shared context-bucket floors;
- readiness proofs for unique counts, not only nonzero token inventory;
- updated dataset identities and receipts that encode the new prompt policy.

Until those changes pass focused tests and a dry run, the old YAML weights are
stale and must not label a generated artifact as B′, C, or D under this design.

## Immutable manifest and atomic publication

Each layer has a separate content-addressed receipt:

- baseline occurrence and exclusion receipt;
- source inventory and license receipt;
- held-out UUID receipt;
- candidate reserve receipt;
- exact prompt-selection receipt;
- target-generation or replay-validation receipt;
- tokenized corpus and exposure-view receipt;
- transfer and destination-readiness receipt.

The prompt-selection digest covers all source manifests, source file hashes,
licenses, baseline and held-out exclusions, seed, domain/lane/language quotas,
context policy, and ordered selected UUIDs. The response digest additionally
covers target/tokenizer/template/runtime identities, generation parameters,
messages, token IDs, and loss masks.

Publication writes corpus shards and receipts to a job-unique sibling partial
directory. It rereads and hashes every artifact, reconciles all counts and
token totals, calls `fsync` on files and directories, and atomically renames
the complete directory into a previously absent final namespace. It never
overwrites a final namespace. A source manifest and selection manifest become
visible together; neither is published alone.

## Current-checkpoint cutover

Existing jobs may continue on the old 1.3M corpus while prompts are prepared.
Preparation does not mutate their dataloader or output namespaces.

For each target, thinking mode, method, and block size, choose one verified
complete parent checkpoint and record its model hash, step, cumulative old-data
exposure, dataset identity, and evaluator baseline. All B′/C/D forks within
that family must adopt the exact same parent weights. A later checkpoint cannot
replace the parent for one arm only.

Cutover is weight-only adoption into a new experiment identity. Model names,
shapes, dtypes, and tensor hashes must match. Optimizer, scheduler, scaler, RNG,
dataloader cursor, epoch, global step, and W&B resume identity reset. The new
run records the parent checkpoint and its prior exposure as lineage. This is
not a same-run resume and must not write into the old output directory.

Because many current jobs have already repeated portions of the 1.3M corpus,
the result is “complement after parent checkpoint X,” not a claim that the
model saw every historical row exactly once. If a verified checkpoint closest
to the first 1.3M-equivalent exposure is still retained, it is the preferred
parent for the cleanest 1.3M-to-2.0M interpretation. Otherwise the latest
common verified parent is acceptable, but its repeated exposure must be shown
in the report.

## Training and evaluator milestones

No arm begins until its immutable corpus, destination receipt, weight-adoption
receipt, and bounded training canary all pass. Evaluation uses the same target,
runtime, hardware class, CUDA-graph policy, concurrency, request set, and
proposed K within a paired comparison.

Milestones are:

1. **Parent:** evaluate the frozen cutover checkpoint before new data.
2. **B′ 256M tokens:** early gap-fill signal.
3. **B′ 700K prompts:** one complete complement pass; report cumulative source
   occurrences, unique UUIDs, and assistant tokens.
4. **C/D 256M tokens:** pipeline and early ranking; no arm is eliminated solely
   for placing third.
5. **C/D 1B tokens:** first substantive equal-exposure comparison.
6. **C/D common full boundary:** evaluate at the largest exact assistant-token
   boundary supported by both 2M prompt views, and separately report each
   arm's full-prompt token total.

The evaluator bundle includes SPEED-Bench throughput-1K through 32K by context
bucket; acceptance length/rate and accepted tokens per proposal; end-to-end
latency and throughput at controlled concurrency; Math, Code, STEM/Science,
multilingual, Chat, agentless SWE, interactive SWE, and generic tool subsets;
tool-schema validity; and deterministic lossless-equivalence checks against the
target. D is useful only if it improves SWE/agentic/tool acceptance over C
without a material regression in Math, Code, SPEED, or correctness.

## Readiness and failure gates

Publication and training fail closed when any of the following occurs:

- baseline count, boundary, unique UUID count, or file digest does not match;
- any source lacks an exact revision, license, physical-file hash, or approved
  use;
- an historical or held-out UUID is admitted;
- a B′, C, D, language, lane, or context cell misses its exact prompt quota;
- the 20% reserve is absent before synthesis begins;
- C and D non-agentic paired UUIDs or context floors differ;
- a target response is partial, conflicting, mixed-mode, or generated by the
  wrong target/runtime identity;
- a replay trajectory fails schema or referential-integrity validation;
- token IDs and loss masks are unaligned or use the wrong tokenizer/template;
- an exact exposure view misses its assistant-token quota;
- a partial tree, corrupt shard, or mismatched manifest is observed;
- checkpoint adoption includes stale optimizer, scheduler, RNG, loader, step,
  or W&B state;
- the bounded canary lacks finite loss, a complete checkpoint, non-idle GPU
  telemetry, or evaluator output.

Quota failure produces a machine-readable blocker receipt. It never triggers
silent row borrowing, percentage renormalization, lexical-prefix selection, or
publication under a misleading dataset name.

## Final decision

- Prepare B′ now as the exact 700K PTV2 gap fill: STEM 200K and
  Japanese/Spanish/French/Italian 125K each, with German excluded.
- Prepare C and D as paired 2M-prompt views with exact
  `30/20/10/20/15/5` domain counts.
- Keep D's 30% agentic area equally split among target-synthesized agentless
  SWE, recorded interactive SWE, and recorded generic tool trajectories.
- Preserve row/sample counts for coverage reporting, but make every scientific
  comparison at an exact assistant-loss-token boundary.
- Continue current drafter jobs independently until immutable prompt and
  cutover gates identify a common parent checkpoint for each experiment family.
