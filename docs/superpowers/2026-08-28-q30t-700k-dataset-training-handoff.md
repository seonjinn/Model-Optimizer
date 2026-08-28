# Q30T SWE-Heavy 700K Dataset and Training Handoff

**Last updated:** 2026-08-28 America/Los_Angeles

This document is the durable cross-session handoff for the Qwen3-30B-A3B
Thinking continuation from the completed PTV2 1.3M parents. It records only
verified state. It does not authorize dataset publication or a training job.

## Objective

Build one exact SWE-heavy 700,000-occurrence continuation dataset from the
remaining authenticated PTV2 and PTV3 sources, then train both block-size-8
DFlash and DSpark drafters from the completed PTV2 1.3M step-25391 parents.
Both methods must consume byte-identical dataset bytes in identical row order.

## Frozen Scientific Contract

Scientific identity:

```text
ptv2-ptv3-complement-700k-qwen3-30ba3b-thinking-swe-heavy-v1
```

Exact quotas:

| Category | Occurrences |
| --- | ---: |
| PTV2 STEM | 200,000 |
| PTV2 Japanese | 25,000 |
| PTV2 Spanish | 25,000 |
| PTV2 French | 25,000 |
| PTV2 Italian | 25,000 |
| PTV2 subtotal | 300,000 |
| PTV3 SWE-v3 | 180,000 |
| PTV3 interactive agentic/SWE | 60,000 |
| PTV3 general tool | 160,000 |
| PTV3 subtotal | 400,000 |
| Total | 700,000 |

Selection is category-local and deterministic. A rejection advances to the
next authenticated occurrence in the same category. Quotas may not be
redistributed. Padding, cycling, duplicate prompt identity, duplicate
normalized row identity, and repeated source occurrence are forbidden.

Tokenizer/target revision:

```text
144afc2f379b542fdd4e85a1fcd5e1f79112d95d
```

Target tree SHA-256:

```text
7bd176a868273ca8acf8db0ec1ab528f7972b0ae5bcf67ed4b3726368b9f7d13
```

Reviewed tokenizer receipt physical SHA-256:

```text
5ba642c455e60b67eca295dce92dd7da47292fdba66c5f9d269669c14cafc509
```

Authenticated historical audit physical SHA-256:

```text
2469430c144d9b86850901df0a28cb810b437555ee894771b796b1386d1c18b5
```

Ordered historical occurrence digest:

```text
863470b22925d74228d31b1c2433d9461d25e8d02a298cb3a08a6bc99be55060
```

## Training Schedule

- Block size: 8.
- Methods: DFlash and DSpark.
- Total occurrences: 700,000 exactly once.
- Total optimizer steps: 1,368.
- Steps 1-1,367: global batch size 512.
- Final step: global batch size 96.
- Distributed shape: 32 ranks, local batch 4 except local batch 3 for the
  final exact batch.
- No padding, cycling, duplication, or fallback rows.
- First gate: independent 20-step canary for DFlash and DSpark.
- Full training starts only after both canaries and their receipts pass.

## Source Evidence Already Present on Ptyche

PTV2 full source asset:

```text
/lustre/fsw/coreai_dlalgo_llm/users/sna/modelopt-qwen3-drafter-training/
  assets/q30t-ptv2-full201-source-v1
```

- 201 files.
- 44,423,886,661 bytes.
- `SOURCE_PLAN.json` SHA-256 prefix `96970541`.
- Completion SHA-256 prefix `018d6591`.
- Completion is true.

PTV3 staged asset:

```text
/lustre/fsw/coreai_dlalgo_llm/users/sna/modelopt-qwen3-drafter-training/
  assets/q30t-ptv3-source-stage-75209087-v1
```

- 100 files.
- 670,399 rows.
- SWE-v3: 237,970 rows.
- Interactive SWE: 97,307 rows.
- General tool: 335,122 rows.
- `MANIFEST.json` SHA-256:
  `384fdc3bda32f06ac8f54b970008a8c91c0e52b99f380b2079719408fcc5808a`.
- `completion.json` SHA-256:
  `e29ab22c9ea85ddd1b7a7106974ce24e787fb121bc1f3282daa96deb30843b68`.
- Completion is true.

The staged source assets are ready. The final 700K `DATA.jsonl`, manifest,
completion receipt, row-observation A/B evidence, canaries, and full jobs do
not exist yet.

## Parent and Tokenizer Trust

Reviewed Q30 tokenizer receipt approval root:

```python
frozenset({
    "5ba642c455e60b67eca295dce92dd7da47292fdba66c5f9d269669c14cafc509"
})
```

Reviewed Q30 parent receipt physical SHA-256 values:

```text
393a2b7c5cbe2037914cbfa531d1d6fdce2bcac2f4204aa001db89d345c3a500
d5416b8fd9644802b42f04511791418bf00dd023f8c5df96593010be95ff2571
```

All three reviewed receipts bind source commit
`212d01516b32e5a81f077d7f2f26ec8de38707c6`, target revision `144afc...`,
target tree `7bd176...`, runtime archive `4a20aee6...`, historical audit
`246943...`, and ordered history `863470...`.

## Conservative Held-Out v2

The user approved a conservative public-benchmark-superset exclusion policy.
The fixed catalog order is:

```text
speed, math, code, swe, tool
```

The policy excludes exact prompt identities and exact embedded benchmark
problem, answer, patch, test patch, code, tool-schema, and policy content.
It uses deterministic exact matching only, not fuzzy or embedding similarity.
Rejected rows are replaced only from the same quota category.

Current design file:

```text
docs/superpowers/specs/2026-08-28-q30t-conservative-heldout-v2-design.md
```

Current fix-round-2 frozen SHA-256 under scoped rereview:

```text
dfea1f49e005615ed840e182b1b1c33a82d54389f8b84f31b469429a1cc0b851
```

The design is state `P`, not approved. It does not authorize training. Main
remaining external source concerns include deterministic SPEED placeholder
resolution, LiveCodeBench durable acquisition, Terminal-Bench selected-tree
manifesting, and independent reviewer/key approval.

## Runtime Immutability

The prior Task-5 runtime was rejected because it extracted into a same-UID
writable node-local path, imported from a mutable checkout, used named Pyxis
reuse, and had unsafe cleanup. It cannot authorize row observation or training.

The replacement design and implementation plan are committed and pushed:

```text
commit 9fed9e6b56140da706039ce2151ee51b98d082a8
branch sj/q30t-ptv23-complement-700k
remote gitlab
```

Files:

```text
docs/superpowers/specs/2026-08-28-q30t-runtime-immutability-amendment-design.md
docs/superpowers/plans/2026-08-28-q30t-runtime-immutability-amendment-plan.md
```

Independent review result: 0 critical, 0 warnings, 0 nits.

The mandatory live chain is feasibility-only F test-only+real
`1 -> 2 -> 16` nodes. The 16-node receipt binds the exact segment-16 training
topology. The design requires a real administrator-owned gateway/build service,
same-OFD Linux lease proof, direct unnamed Pyxis SquashFUSE, distributed
READY/GO/ABORT, exact remote-death proof, bounded node-local logs, independently
signed review, producer P with empty roots, and constants-only approval A.

Runtime Task 0 implementation is active. Its SDD ledger is:

```text
.superpowers/sdd/2026-08-28-q30t-runtime-immutability-amendment-plan/progress.md
```

Baseline before Task 0:

```text
142 passed, 12 skipped
```

Important blocker: no authoritative public key currently exists for
`q30t-runtime-independent-reviewer` / namespace `q30t-runtime-review-v1`.
Synthetic test keys are not acceptable. Trust freeze must fail closed until an
independent reviewer public key and byte-frozen allowed-signers file are
provided.

The root-owned Ptyche gateway/build service is also a real external gate. A
mock or same-UID directory cannot substitute for it. If absent, Phase F must
record `BLOCKED` and stop before P.

## Row Observation

Completed Task 1 commit:

```text
8e195c81a086bc8e26e10f84d5bf8ab0b73ea429
```

Owned files:

```text
tools/launcher/common/specdec/q30t_row_observation_runtime.py
tools/launcher/tests/test_q30t_row_observation_runtime.py
examples/dataset/observe_q30t_ptv23_row_schemas.py
tests/examples/dataset/test_observe_q30t_ptv23_row_schemas.py
```

The blocking parent/root rebind and container-side output publication findings
were fixed and independently reviewed clean. Row Tasks 2-9 remain blocked on
the replacement runtime boundary.

## Why No Training Job Is Running

The source assets exist, but these exact gates are still open:

1. Held-out v2 design rereview, implementation, source acquisition, replay,
   semantic review, and approval.
2. Distinct Q30 SWE-heavy source-requirements and source-inventory artifacts.
3. Runtime Task 0 local F implementation and review.
4. Real Ptyche F 1-/2-/16-node feasibility and independent signature.
5. Protected derived runtime image build/review and one-/two-node qualification.
6. Row observation A/B and source-inventory/row-schema mapping.
7. Exact 700K capacity pass and dataset build.
8. DFlash and DSpark 20-step canaries.

Starting training before these gates would either admit benchmark leakage,
freeze a scientifically incomplete dataset, or execute unreviewed mutable
runtime bytes.

## Required Execution Order

1. Finish and independently rereview held-out v2 design.
2. Write/review its implementation plan and implement typed producers/loaders.
3. Finish Runtime Task 0 and independent local review.
4. Supply an authoritative independent reviewer public key and allowed-signers
   trust root.
5. Confirm administrator gateway/build service exists; run live F test-only
   then real 1 -> 2 -> 16 jobs and independent replay.
6. Complete runtime P/A chain and row-observation Tasks 2-9.
7. Produce/review historical migration, held-out, source-inventory, and
   row-schema evidence.
8. Run the exact 700K capacity pass; build and replay exact dataset bytes.
9. Run DFlash and DSpark 20-step canaries.
10. Submit both full 1,368-step jobs only after both canaries pass.

## Git and Worktree State

Primary implementation worktree:

```text
/Users/sna/ModelOpt_SpecDec/worktrees/q30t-ptv23-complement-700k
```

Branch:

```text
sj/q30t-ptv23-complement-700k
```

Preserve these unrelated user-owned dirty paths. Never stage, clean, revert,
or overwrite them unless a later task explicitly assumes ownership:

```text
tools/launcher/common/specdec/qwen4b_ptv23_continuation.py
tools/launcher/common/specdec/run_qwen4b_ptv23_continuation.sbatch
tools/launcher/tests/test_qwen4b_ptv23_continuation.py
uv.lock
docs/superpowers/specs/2026-08-27-q30t-swe-heavy-700k-companion-arm.md
examples/dataset/qwen3_30ba3b_thinking_ptv23_swe_heavy_700k_v1.json
examples/dataset/qwen3_30ba3b_thinking_ptv23_swe_heavy_target_v1.json
tests/examples/dataset/test_q30t_swe_heavy_700k_config.py
```

Use exact-path staging for every commit. Do not run `git clean`, recursive
deletion, or broad worktree reset.

## Separate Q235 Transfer Completed

This is not an input to the Q30 700K dataset, but it was completed in the same
session.

Qwen3-235B Base DSpark B8 evaluator export:

```text
/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sna/
  modelopt-specdec/checkpoints/qwen3-235ba22b-base-nemotron-b8-s25391/dspark
```

- 2 files / 2,546,452,960 bytes.
- Manifest:
  `f01d6228d52dc7855df56d9630375da06edf191422c5381eeff9789677d1ddc0`.

Full resume:

```text
/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sna/
  modelopt-specdec/checkpoints/qwen3-235ba22b-base-nemotron-b8-s25391/
  dspark-resume
```

- 42 files / 10,140,747,374 bytes.
- Manifest:
  `b48f8cfc91dcdc364817a6ecf141e55be531bbea97dd77be438acb7738c8425c`.
- Ptyche uploads `2675396`/`2675398` completed.
- OCI downloads `6657365`/`6657367` completed.
- Canonical receipts, source milestone parity, names/sizes, and symlink absence
  were verified.

## Rules for a New Session

- Read this file first, then the runtime and held-out design files.
- Read the active SDD ledger before dispatching or redoing any task.
- Trust Git commits and ledger entries over conversational memory.
- Do not claim the 700K dataset or training exists until physical receipts are
  replayed.
- Do not hand-author approval JSON or invent a reviewer key.
- Commit and push before any Slurm submission.
- Run `sbatch --test-only` before each real submission.
- On MARS clusters, keep source under `/home`, temporary work under
  `/raid/scratch`, and durable datasets/checkpoints/receipts under `/lustre`.
- Query only the relevant job or `squeue --me`, no more than once per minute.
