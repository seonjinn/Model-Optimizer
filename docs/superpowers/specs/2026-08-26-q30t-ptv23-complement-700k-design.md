# Qwen3-30B-A3B Thinking PTV2/PTV3 700K Continuation Design

## Goal

Continue the fully trained Qwen3-30B-A3B-Thinking-2507 B8 DFlash and
DSpark drafters from their authenticated Nemotron Post-Training Dataset v2
1.3M step-25391 parents on one disjoint 700K complement. Run an exact 20-step
canary for each method and then one full, no-cycle pass on 16 GB200 nodes.

The continuation is intended to add STEM, multilingual, SWE, interactive
agentic, and general tool-use capability without replaying the original 1.3M
examples or admitting SWE-bench evaluation leakage.

## Approved scientific identity

The immutable dataset identity is `ptv2-ptv3-complement-700k-v1`:

| Category | Rows |
|---|---:|
| PTV2 STEM | 300,000 |
| PTV2 Japanese | 50,000 |
| PTV2 Spanish | 50,000 |
| PTV2 French | 50,000 |
| PTV2 Italian | 50,000 |
| PTV3 SWE-v3 | 100,000 |
| PTV3 interactive agentic/SWE | 19,000 |
| PTV3 general tool trajectories | 81,000 |
| Total | 700,000 |

Selection is deterministic source order with refill only inside the same
category. There is no redistribution, replacement, padding, or cycling.
Global UUID and normalized-content deduplication occur before quota admission.
The union of historical 1.3M identities and the exact named held-out receipts
is excluded. SWE-bench Verified, Test, Lite, answer, patch, and held-out
identities are excluded from both inputs and derived records.

Tool trajectories preserve declarations, call IDs, results, order, and
reasoning. Malformed, duplicate, orphaned, or unresolved calls are rejected.
Every selected row must have at least one assistant-loss token and must fit the
Qwen3-30B-A3B-Thinking-2507 tokenizer and training chat template at 4,096
tokens.

## Authenticated parents

Only the following Ptyche parents are eligible. The Lyris attempts failed
during staging and OCI-HSG contains only partial checkpoints, so neither is a
valid source for this continuation.

### DFlash

- Completion job: `2638009`, 16 nodes, step 25391, completed.
- Parent root:
  `/lustre/fsw/coreai_dlalgo_llm/users/sna/modelopt-qwen3-drafter-training/training/q30t-nemo-dflash-b8-16n-922609729/milestones/step-025391/resume-checkpoint-025391`

### DSpark

- Completion job: `2638015`, 16 nodes, step 25391, completed.
- Parent root:
  `/lustre/fsw/coreai_dlalgo_llm/users/sna/modelopt-qwen3-drafter-training/training/q30t-nemo-dspark-b8-16n-922609729/milestones/step-025391/resume-checkpoint-025391`

One canonical external receipt is produced for each parent. Each receipt binds
the target repository and revision, Thinking variant, target tree hash,
method, block size 8, global step 25391, parent run identity, checkpoint file
set and tree digest, ModelOpt state, source commit, runtime identity, historical
1.3M receipt, and ordered historical UUID digest. The two parents may share the
same 700K publication only if their authenticated historical receipt digest and
ordered historical UUID digest are identical.

The child restores drafter weights and required ModelOpt state only. It never
restores the parent optimizer, scheduler, trainer step, or RNG state. Both
methods start a fresh optimizer and fresh linear schedule under new experiment,
output, W&B, and completion identities.

## Architecture choice

### Recommended: shared selection core, target policy, and per-node keepers

Extract the target-independent selection, replay, and publication logic from
the Qwen3-4B builder into a shared complement core. Keep thin Qwen3-4B and
Qwen3-30B-A3B-Thinking policies that bind exact tokenizer repository,
revision, trust schema, template, special-token IDs, and source-requirement
files. This avoids copying more than two thousand lines of security-sensitive
selection and publication code while preventing a Q4 trust receipt from being
accepted as Q30 trust.

Extend the continuation contract rather than overloading the parent path as
the target model. The contract independently authenticates:

- the Qwen3-30B-A3B-Thinking-2507 target model;
- the DFlash or DSpark B8 parent checkpoint;
- the Q30 tokenizer and template receipt;
- the 700K dataset completion receipt and complete semantic trust chain;
- method-specific training recipe and 16-node distributed topology;
- fresh optimizer/scheduler identity and exact exposure schedule;
- trainer entrypoint and runtime image;
- canary and full evidence, checkpoint, and completion roots.

For multi-node execution, launch one long-lived host-side keeper task per
allocated node using an overlapping one-task-per-node `srun`. Each keeper
copies and independently rehashes the contract, reviewed tool, runtime image,
dataset, parent, target inputs, and required receipts into node-local
`/raid/scratch`. It holds no-follow descriptors for the complete lifetime of
the training step and publishes identical node-local anchor names that resolve
to that node's keeper descriptors. Because `/raid/scratch` is node-local, the
same Pyxis arguments resolve to the correct local keeper on every node.

The training `srun` begins only after all 16 keepers have published
authenticated readiness evidence. Keeper death terminates the training process
group and invalidates the run. The controller traps signals, waits for all
training descendants, then stops keepers and persists only bounded evidence to
durable storage.

While each keeper is alive, a one-task-per-node creation/probe step imports the
verified local image exactly once under the per-job name
`q30t-700k-${SLURM_JOB_ID}`. The training step reuses that named container
instead of reopening the image. The creation/probe step also mounts the exact
image anchor read-only at `/run/q30t/runtime.sqsh` and hashes the whole image
from inside the container, establishing the reviewed Pyxis/enroot boundary
before any trainer import.

Distributed execution separates control-plane publication from data-plane
work. Every node stages into the same pathname on its own node-local `/raid`;
no two nodes share an execution-stage directory. All ranks verify their local
stage, then enter a distributed barrier. Only global rank 0 may publish durable
canary/full evidence or receipts after collecting the per-node identities.
Other ranks never attempt exclusive publication.

### Rejected alternatives

- Copying the Q4 builder into a separate Q30 file is faster initially but
  duplicates security-sensitive replay and publication logic and allows the two
  implementations to drift.
- Plain node-local named copies without lifetime descriptor binding leave a
  same-UID pathname replacement boundary between verification and Pyxis use.
- A single batch-node `/proc/<pid>/fd` keeper cannot be resolved on remote
  nodes and is explicitly ineligible for multi-node launch.
- Self-asserted runtime hashes from inside Python do not attest the image that
  Pyxis actually launched. Pyxis/enroot must be an explicit trusted computing
  boundary and pass a real platform probe.

## Exact exposure schedule

The proven topology has 32 trainer ranks and nominal global batch size 512,
but 700,000 is not divisible by 512. The full run therefore has 1,368 optimizer
steps:

- steps 1 through 1,367 consume 512 examples each, for 699,904 examples;
- step 1,368 consumes the remaining 96 examples, exactly 3 per trainer rank;
- `dataloader_drop_last` is false;
- distributed sampling must not pad or repeat records;
- the receipt records `consumed_examples=700000`, `full_steps=1368`,
  `nominal_global_batch_size=512`, and `final_global_batch_size=96`.

The contract validates those four values together instead of using the invalid
identity `steps * global_batch_size == 700000`. Trainer evidence must reconcile
physical dataset occurrences, per-rank consumption, optimizer steps, and the
final partial batch. A 20-step canary uses an isolated output root and never
authorizes itself; its receipt must belong to an immutable approval root before
the full run is admitted.

## Runtime-image attestation

User-space Python cannot independently prove which image Pyxis used. The
runtime contract therefore treats the reviewed Pyxis/enroot deployment as a
trusted computing boundary and remains fail-closed until a platform probe
establishes the following on both Lyris and Ptyche:

1. The same node-local anchor pathname resolves to different local keeper PIDs
   and inodes on two allocated nodes.
2. Pyxis can use a keeper-backed anchor for both `--container-image` and
   directory mounts under the site's `/proc` and Yama policies.
3. `srun --overlap` supports concurrent keeper and training steps.
4. The first process in the same training container hashes the complete mounted
   runtime image, validates pinned in-image sentinels, Python, architecture,
   CUDA, and GPU count, emits per-node evidence, and then `exec`s the reviewed
   trainer without reopening mutable shared paths.
5. Killing any keeper makes its anchor unreadable and causes the training step
   to fail closed.
6. The job-specific named container is created once per node from the verified
   local anchor and is reused by training without a second image import.

Run this as a one-node probe and then a two-node probe on each cluster. Do not
set `RUNTIME_ATTESTATION_IMPLEMENTED` or permit a 16-node canary until all probe
evidence is reviewed and allowlisted.

## Data and training flow

1. Authenticate the PTV2/PTV3 physical source inventory, historical 1.3M
   receipt, exact held-out receipt set, source allowlists, and Q30 tokenizer
   receipt.
2. Select the eight quotas through the shared core and publish one exact 700K
   bundle with an external completion receipt.
3. Replay semantic verification under the Q30 tokenizer and template.
4. Authenticate both parent receipts and prove identical historical lineage.
5. Create separate DFlash and DSpark continuation contracts with isolated
   canary/full roots.
6. Run the per-node keeper/runtime probe. Admission stays fail-closed if any
   node differs.
7. Run DFlash and DSpark 20-step canaries on the same 16-node topology as full
   training. Stage execution inputs independently on every node, synchronize
   after verification, and let only global rank 0 publish durable evidence.
8. Independently review and allowlist the canary receipts.
9. Run each 1,368-step no-cycle continuation and require exact final exposure,
   checkpoint, and completion receipts.

## Failure handling and scheduler policy

- Submission always runs Slurm `--test-only` before a real `sbatch`.
- A submission identity includes method, parent receipt, dataset receipt,
  target, tokenizer, trainer, runtime, topology, and stage.
- Existing exact scheduler identities are adopted; ambiguous or conflicting
  jobs fail closed rather than duplicate.
- Requeue reuses only complete authenticated node-local publications and
  durable checkpoints. Partials use unique names and atomic no-replace
  publication; foreign names are never recursively deleted.
- Scheduler checks are filtered to the exact job or current user and are spaced
  by at least 60 seconds. A new real job is monitored for five minutes.

## Verification strategy

### Dataset and trust tests

- Q30 Thinking tokenizer repository, revision, snapshot tree, template, and
  token IDs reject Q4, Base, mutable, or self-authored receipts.
- The 700K publication is rebuilt and replayed under Q30; reuse of Q4 bytes is
  allowed only with an explicit exact tokenizer-equivalence receipt.
- Historical and held-out exclusions, exact quotas, no-cycle behavior,
  assistant-token count, source order, tool schemas, and SWE-bench exclusion
  are replayed from authenticated physical inputs.

### Parent and schedule tests

- DFlash/DSpark receipt swap, Base/Thinking swap, B8/B16 swap, wrong target,
  wrong step, and partial OCI/Lyris roots are rejected.
- Both parents must prove the same historical lineage before sharing DATA.
- Only weights and ModelOpt state transfer; optimizer, scheduler, RNG, and old
  step restoration are rejected behaviorally.
- Step 1,368 consumes exactly 96 unique remaining records and total physical
  consumption is exactly 700,000.

### Multi-node tests and probes

- Unit tests execute keeper short-write, post-copy digest, readiness, death,
  signal, cleanup, delimiter, and identity failures.
- Linux integration tests validate local keeper descriptor visibility across a
  separate process boundary.
- One-node and two-node Lyris/Ptyche probes validate actual Slurmstepd, Pyxis,
  enroot, named-container reuse, node-local anchor, runtime hash, and
  keeper-loss behavior.
- A two-node behavioral test proves that execution stages are node-local, all
  ranks reach the verification barrier, and exactly global rank 0 performs
  durable no-replace receipt publication.
- Only after those probes pass may 16-node 20-step canaries be submitted.

## Production gates

No build or training job is authorized until all of these external identities
are nonempty, immutable, and independently reviewed:

- complete PTV2 and PTV3 source inventories and schema allowlists;
- the historical 1.3M receipt and exact named held-out receipts;
- the Q30 Thinking tokenizer/template receipt;
- DFlash and DSpark parent receipts with identical historical lineage;
- target-model receipt;
- trainer entrypoint and runtime-image digests;
- successful one-node and two-node keeper/Pyxis/runtime probes on the selected
  cluster;
- immutable canary receipt approval roots for the full stage.

Until then the launcher must produce an actionable fail-closed error and must
not call `sbatch`.
