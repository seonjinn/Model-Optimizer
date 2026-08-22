# PTV2 Complement and PTV3 Agentic Dataset Design

## Goal

Build three reproducible speculative-drafter training corpora that measure the
value of unused Nemotron Post-Training v2 coverage, target-synthesized Nemotron
Post-Training v3 data, and schema-preserving software-engineering and tool-use
trajectories. The design replaces the biased 1.3M-row PTV2 view with explicit
assistant-loss-token quotas while preserving the existing trained weights as
the initialization for a new experiment family.

The three arms are:

- **B — PTV2 complement:** balanced coverage selected only from PTV2 examples
  whose UUIDs were not used by the existing 1.3M corpus.
- **C — target-synthesized hybrid:** B's eligible pool plus PTV3 Math, Code,
  Science, and agentless SWE prompts synthesized by the exact target model.
- **D — agentic hybrid:** C's eligible pool plus schema-preserving interactive
  SWE and generic agentic/tool trajectories.

The letters describe eligible data constructions, not literal set inclusion.
Because B, C, and D receive the same training-token exposure, C is rebuilt from
the B and PTV3 pools and D replaces part of C's agentless SWE quota with replay
trajectories. C is not B with extra training tokens, and D is not C with extra
training tokens.

## Scope

This design covers corpus inventory, exclusion, synthesis, trajectory replay,
mixture construction, immutable publication, cross-cluster staging, checkpoint
adoption, training gates, and evaluation gates. It does not authorize a corpus
build, production-code change, SLURM submission, cancellation, or replacement
of an existing job.

The first implementation plan may cover the shared corpus and experiment
contracts as one project because all components meet at one immutable inventory
and selection manifest. Environment-backed trajectory collection is a later
project; D uses valid recorded trajectories only.

## Existing Biased Baseline

The current direct Nemotron corpus selected exactly 1,300,000 rows from 26
symlinked shards under:

```text
/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sna/datasets/nemotron-post-training-v2-compatible-1p3m/data
```

Its audited row composition is:

| PTV2 split/domain | Rows | Share |
|---|---:|---:|
| chat | 627,720 | 48.2862% |
| code | 175,000 | 13.4615% |
| math | 239,467 | 18.4205% |
| multilingual_de | 257,813 | 19.8318% |
| stem | 0 | 0% |
| multilingual_ja/es/fr/it | 0 | 0% |
| explicit SWE/agentic/tool | 0 | 0% |
| **Total** | **1,300,000** | **100%** |

The raw PTV2 mirror is associated with revision metadata
`5c89e01dd720ae0f4058445ed49c5fb68a03c76e`. These counts document why the
corpus is a biased reference; they are not accepted as the machine-readable
exclusion record. Before any complement is built, the inventory stage must
resolve every symlink, hash every resolved shard, import its per-split row and
assistant-token histogram, recover the 1,300,000 selected UUIDs, and prove that
the imported histogram matches the table. A missing UUID, duplicate UUID,
broken symlink, count mismatch, or file-hash mismatch stops the build.

The original 1.3M corpus and its training jobs retain their existing dataset
and experiment identities. They are never relabelled as B.

## Source Identity and UUID Exclusion

Every source is immutable and content-addressed. A source manifest records:

- Hugging Face repository ID, configuration, split, license, and exact commit
  revision;
- every physical input file's relative path, byte size, and SHA-256;
- source row index, source-native ID, and canonical prompt UUID;
- source, domain, language, context, and response-lane labels;
- tokenizer snapshot digest and chat-template digest;
- row, full-context-token, 4K-view-token, and assistant-loss-token histograms.

PTV3 repository-level revisions are insufficient on their own. Each manifest
must contain the exact PTV3 revision and each downloaded file's SHA-256. A file
whose content changes under the same logical source name creates a new source
identity and cannot resume into an existing corpus namespace.

The canonical prompt UUID is the SHA-256 of the normalized prompt-bearing
conversation before target response generation. Normalization preserves role,
content, top-level tool definitions, tool-call structure, and message order;
it removes storage-only fields and uses deterministic JSON serialization. The
same normalization code is used to reconstruct the prior 1.3M exclusion set
and to inventory new candidates.

Selection rejects a candidate when its UUID appears in any of these sets:

1. the audited prior 1.3M training corpus;
2. a held-out SPEED, Math, Code, SWE, or tool-use evaluation set;
3. another source already admitted to the candidate inventory.

Deduplication occurs before quotas. A UUID collision with different canonical
content is a hard error rather than an arbitrary winner.

## PTV2 Remaining-Source Policy

Arm B uses only UUIDs absent from the prior 1.3M set. Its assistant-loss-token
quotas are the approved balanced PTV2 complement:

| B category | Assistant-loss-token quota |
|---|---:|
| Math | 25% |
| Code | 25% |
| STEM | 20% |
| Chat and instruction following | 20% |
| Multilingual | 10% |

All percentages in this document refer to assistant-loss tokens after applying
the target tokenizer and training loss mask, never rows, bytes, prompts, or
total context tokens.

The PTV2 policy deliberately repairs the audited gaps:

- remaining `stem` rows are eligible for STEM;
- `multilingual_ja`, `multilingual_es`, `multilingual_fr`, and
  `multilingual_it` are eligible for Multilingual;
- `multilingual_de` is excluded from the complement because German already
  supplied 19.8318% of the biased baseline;
- remaining Math, Code, and Chat rows are eligible only after UUID exclusion;
- PTV2 rows are never relabelled as explicit SWE, agentic, or tool-use data;
- source-native assistant answers are not used for target-synthesized lanes;
- B strips the source completion and synthesizes a fresh response with the
  exact target checkpoint, just like C's non-replay records.

Within a category, candidates are ordered by a seeded hash of
`dataset_identity`, category, context bucket, source ID, and prompt UUID. The
builder consumes that order to the exact token quota and masks the final row at
an assistant-token boundary if necessary. It never selects a lexical prefix or
caps rows before deterministic shuffling.

## Arm C and D Mixtures

C and D use the same top-level assistant-loss-token distribution so that their
difference isolates schema-preserving trajectories:

| Domain | C | D |
|---|---:|---:|
| SWE / Agentic / Tool | 35% | 35% |
| Math / Reasoning | 25% | 25% |
| Code | 15% | 15% |
| STEM / Science | 15% | 15% |
| Instruction / Chat | 5% | 5% |
| Multilingual | 5% | 5% |
| **Total** | **100%** | **100%** |

For C, the full 35% SWE/Agentic/Tool quota is agentless SWE target synthesis.
For D, that 35% is subdivided exactly as follows:

| D sub-lane | Assistant-loss-token quota |
|---|---:|
| Agentless SWE target synthesis | 15% |
| Interactive SWE replay, including OpenHands/R2E-style traces | 10% |
| Generic agentic/tool replay | 10% |

C's Math, Code, and STEM/Science candidates are the union of eligible PTV2
complement rows and pinned PTV3 Math, competitive/general Code, and Science
sources. Its Instruction/Chat and Multilingual candidates use the eligible
complement pool; adding another PTV3 source requires a new dataset identity.
D uses the same non-agentic candidate pools as C.

No hidden PTV2/PTV3 row ratio is introduced inside an overlapping domain.
Candidates enter one domain-specific seeded order after UUID exclusion, and the
quota is filled from that order. The resulting per-source share is therefore a
reproducible consequence of the pinned candidate inventory and is recorded in
the selection manifest. Changing a source, revision, seed, tokenizer, or
candidate set creates a new dataset identity rather than silently changing the
share.

## Two-Lane Data Architecture

### Target-synthesis lane

This lane accepts prompt-only records with no executable tool dependency. Each
record contains at least:

```text
schema_version
prompt_uuid
source_id, source_revision, source_file_sha256, source_row_index
domain, language, full_context_bucket
target_model_id, target_revision, tokenizer_sha256, chat_template_sha256
thinking_mode
prompt_messages
generated_assistant_message
generation_parameters
full_token_count, full_assistant_token_count
training_input_ids, training_loss_mask, training_assistant_token_count
record_sha256
```

The exact target checkpoint generates the response. Qwen3-4B, Qwen3-30B-A3B,
and Qwen3-235B-A22B are three independent synthesis families; a response from
one family is never reused by another. Thinking-on and thinking-off are also
separate immutable generation runs. Within a target and arm, they use the same
selected target-synthesis prompt UUIDs so mode comparisons do not change prompt
coverage. Each manifest pins `thinking_mode` to one value and rejects a mixed
shard.

Target-native reasoning is preserved according to the pinned chat template.
Generation settings, stop conditions, maximum output length, server/runtime
digest, and failure status are part of provenance. A partial or failed response
does not enter the candidate inventory.

### Schema-preserving replay lane

This lane accepts recorded SWE and tool-use trajectories. Each record contains
the target-independent source provenance and full-context metadata above plus:

```text
tools
messages
assistant.tool_calls[].id
assistant.tool_calls[].type
assistant.tool_calls[].function.name
assistant.tool_calls[].function.arguments
tool.tool_call_id
tool.name, tool.content
reasoning_content when present
trajectory_sha256
```

Replay preserves top-level tool schemas, assistant `tool_calls`, call IDs, tool
results, message order, and recorded reasoning. It never flattens a trajectory
to role/content-only chat and never claims that the current target executed the
tools.

Replay thinking mode is source-native rather than synthesized. A trace with
recorded reasoning is eligible only for the thinking-on replay manifest; a
trace explicitly marked reasoning-off is eligible only for the thinking-off
manifest. The builder never strips reasoning to manufacture an off record or
adds reasoning to manufacture an on record. Unlabelled traces are quarantined,
and insufficient mode-compatible replay tokens cause a quota shortfall.

Every tool result must reference one preceding unresolved assistant call ID.
Call IDs must be non-empty and unique within the trajectory; each call must name
a function declared by the top-level tool schema; all calls must be resolved
before the next non-tool message or the end of the trajectory. Orphan results,
duplicate calls, missing schemas, unresolved calls, unsupported roles, and
malformed arguments quarantine the entire trajectory with a reason code.
Quarantined rows contribute no quota.

The target tokenizer and chat template serialize both lanes. The loss mask is
one only for assistant-owned tokens, including assistant reasoning and
serialized tool calls when the template assigns them to the assistant. System,
user, developer, and tool-result tokens have loss mask zero. The manifest
records the assistant-token total derived from the mask rather than trusting
source metadata.

## Full Context and 4K Training View

The canonical record retains the complete validated conversation up to 32K
tokens and labels it `le4k`, `4k_16k`, or `16k_32k` from the full token count.
A record above 32K is excluded with a counted `context_gt_32k` reason.
It also materializes a separate sequence-length-4096 training view with aligned
token IDs and loss mask. The 4K view is derived, never treated as the original
conversation.

The manifest records full token count, full assistant-token count, 4K-view
token count, 4K-view assistant-loss-token count, truncation location, and
whether a tool transaction crosses the 4K boundary. A 4K view that cuts between
a tool call and its result is excluded from 4K training rather than producing a
referentially invalid example. Long examples remain available for later
long-context training and evaluation; they are not silently reassigned to the
`le4k` bucket.

Quota accounting for the initial study uses the 4K-view assistant loss mask.
Full-context token statistics remain mandatory metadata.

## Dataset Identity and Immutable Publication

Each materialized corpus receives a new identity of the form:

```text
ptv23-complement-v1-<target>-<thinking>-<arm>-<selection_sha256_12>
```

`selection_sha256` covers the source manifests, prior-UUID exclusion manifest,
held-out UUID manifest, target and tokenizer revisions, thinking mode, domain
weights, lane weights, seed, context policy, ordered selected UUIDs, token IDs,
and loss masks. B, C, and D therefore cannot alias the old `nemo-direct`
dataset or one another.

Publication uses a job-unique partial directory on the destination filesystem.
The builder writes shards, per-shard hashes, rejection receipts, histograms,
and the top-level manifest; re-reads and validates all of them; calls `fsync` on
files and directories; then atomically renames the partial directory into a
previously absent immutable final namespace. An existing final namespace is
never overwritten. Failed work remains outside the final namespace and may be
resumed only from a receipt whose inputs and hashes exactly match.

A completed manifest must prove:

- no prior-training or held-out UUID is selected;
- no UUID appears twice;
- category and D sub-lane assistant-token totals equal their quotas exactly;
- every shard and source file hash matches;
- thinking mode and target identity are uniform;
- token IDs and loss masks are aligned and use the pinned tokenizer;
- every replay trajectory passes referential-integrity validation;
- full-context and 4K-view histograms reconcile with selected records.

## Equal Exposure and Training Identity

At every comparison checkpoint, B, C, and D consume exactly the same cumulative
assistant-loss-token exposure. The sampler stops by summed loss-mask tokens,
not row count or optimizer step count. Batch padding, user/system/tool tokens,
and tokens masked out at the exact quota boundary do not count as exposure.

Every run receives a new experiment identity containing target revision,
thinking mode, arm, method, block size, sequence length, dataset identity,
initialization checkpoint hash, seed, planned maximum token budget, and the
256M/1B/4B gate schedule. The checkpoint's observed cumulative exposure is
state within that identity, so continuation from one gate to the next is a
same-run resume. Resume is allowed only within the same identity.

The new runs may adopt model weights from the completed first-1.3M checkpoint.
They load only model/drafter weights after strict name, shape, dtype, and hash
validation. Optimizer state, learning-rate scheduler state, gradient scaler,
RNG state, data-loader cursor, epoch, global step, and W&B resume ID are reset.
Step zero of the new experiment records the parent checkpoint as lineage, not as
a same-run resume.

## Training and Evaluation Gates

B, C, and D use identical initialization, optimizer configuration, learning-rate
schedule, seed, effective batch size, sequence length, method, block size, and
target serving configuration. Only the dataset identity differs.

All three arms pass the following cumulative assistant-loss-token gates:

1. **256M:** canary-scale comparison and pipeline validation. No arm is removed
   solely for ranking third here.
2. **1B:** first multi-billion-token learning-curve and evaluator gate.
3. **4B:** stable-ranking decision gate. Elimination requires a consistent
   disadvantage at both 1B and 4B plus a material endpoint gap recorded before
   the run starts.

Each gate requires complete checkpoints for all eligible arms and the same
evaluator revision, prompts, decoding settings, proposed draft-token count, and
hardware/runtime class. The report separates unique corpus tokens from repeated
training exposure and records epochs explicitly.

The mandatory evaluator bundle includes:

- SPEED-Bench acceptance length, acceptance rate, accepted tokens per proposed
  token, verifier calls per output token, and end-to-end latency/throughput by
  category and concurrency;
- GSM8K, MATH-500, and AIME trajectory acceptance;
- HumanEval, MBPP, and LiveCodeBench trajectory acceptance;
- held-out SWE acceptance split into agentless and interactive traces;
- held-out generic agentic/tool acceptance with tool-schema validity;
- results by full-context bucket and 4K training-view status.

Speculative decoding remains lossless: Math, Code, SWE, and tool outputs must
match the target within the evaluator's deterministic tolerance. Correctness
regression is a stop condition, not a tradeable speed metric. D advances only
if it improves held-out SWE/agentic acceptance over C without reducing any
major Math or SPEED category by more than 3%. C or D must improve held-out SWE
acceptance over B by at least 10% to justify the additional data path.

## Cross-Cluster Staging

One canonical cluster builds and publishes each corpus. Other clusters never
regenerate an allegedly identical corpus. Cross-cluster movement uses the
existing PDX/PBSS data-mover workflow with a content-addressed object prefix.

The source cluster uploads immutable shards and manifest, verifies uploaded
SHA-256 values, and publishes a transfer receipt. The destination downloads to
a job-local partial namespace, verifies every byte count and hash against the
top-level manifest, then atomically publishes the local immutable namespace.
Login nodes do not perform bulk copies. Credentials do not appear in arguments,
manifests, logs, or Git.

Training submission requires a destination readiness receipt binding cluster,
filesystem path, dataset identity, manifest SHA-256, tokenizer digest, and
timestamp. A local directory with the expected name but no matching receipt is
not considered staged.

## Failure Handling

- Missing revision, source file hash, license, UUID, tokenizer digest, target
  digest, or prior-corpus exclusion manifest stops before GPU synthesis.
- A quota shortfall stops arm publication; weights are never silently
  renormalized.
- Thinking-on and thinking-off rows cannot share a manifest or output namespace.
- Invalid tool trajectories are quarantined with counted reason codes and never
  flattened into the synthesis lane.
- Synthesis retries preserve prompt UUID and generation identity. Conflicting
  successful responses for one identity stop publication.
- A preempted writer cannot expose a final namespace. Resume requires matching
  input hashes and continues only missing immutable shards.
- Training cannot resume across dataset, target, thinking-mode, arm, seed, or
  initialization identities.
- A cross-cluster checksum mismatch deletes or quarantines only the job-local
  partial copy and leaves the canonical artifact untouched.

## Pending Production-Job Policy

The pending `s25391` replacement is not submitted merely because this design is
approved. Replacement becomes eligible only after the selected target corpus
has a complete immutable manifest, prior-UUID exclusion proof, exact quota and
schema validation, destination staging receipt, evaluator readiness receipt,
and successful training canary.

Until those gates pass, existing pending jobs remain independent of this
dataset study. Scheduler-account or partition changes do not alter corpus
readiness, and this design does not authorize cancellation or resubmission.

## Verification Requirements

The implementation must add focused tests for:

- reconstruction of the exact 1,300,000-row audited histogram from the 26
  resolved shards and rejection of any mismatch;
- canonical UUID stability, prior-UUID exclusion, collision detection, and
  held-out contamination rejection;
- deterministic assistant-token quotas for B, C, and D, including D's
  15%/10%/10% subdivision and exact final-token masking;
- PTV2 language policy, especially exclusion of `multilingual_de`;
- target, thinking-mode, tokenizer, revision, and file-SHA isolation;
- tool-call/result referential integrity and quarantine reason counts;
- full-context bucket retention, 4K-view derivation, and rejection of a 4K view
  that splits a tool transaction;
- atomic publish, interrupted resume, immutable-name collision, and corrupt
  shard detection;
- equal exposure at 256M, 1B, and 4B;
- weight-only checkpoint adoption with optimizer and scheduler reset;
- cross-cluster transfer receipt validation and credential redaction;
- evaluator thresholds and prevention of a premature `s25391` replacement.

Before any job submission, formatting, lint, type checking, focused unit tests,
manifest dry runs, and `sbatch --test-only` must pass. A bounded synthesis and
training canary must then produce valid shards, a complete checkpoint, per-GPU
telemetry, and evaluator output before B, C, or D advances to 256M.

## Design Decisions

- The prior 1.3M corpus is lineage and an exclusion set, not Arm B.
- Assistant-loss tokens are the only mixture and exposure unit.
- B balances unused PTV2 coverage; C and D share the approved
  35/25/15/15/5/5 top-level mix.
- C uses agentless target synthesis for its full 35% SWE quota; D replaces 20
  percentage points with 10% interactive SWE and 10% generic agentic/tool
  replay.
- Q4, Q30, and Q235 responses and thinking modes have separate identities.
- Full context is preserved even when the initial trainer consumes a 4K view.
- Recorded tool trajectories preserve schemas and referential integrity; they
  are not represented as target-executed interactions.
- New training starts from adopted weights with fresh optimizer and scheduler
  state.
- No pending production replacement occurs before corpus readiness is proven.
