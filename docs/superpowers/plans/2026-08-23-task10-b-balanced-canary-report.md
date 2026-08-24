# Task 10 B-balanced canary implementation report

## Scope and scientific order

Task10 prepares only the B-balanced data artifact and its bounded 200-step
runtime canary. It does not authorize B as the first scientific training arm.
The fixed execution order remains A-repair, B-balanced, B-balanced-target, then
PTV3 SWE/agentic/tool. B artifact preparation may run concurrently, but the
submission receipt records `b_preparation_only: true` and
`scientific_training_authorized: false`.

## Authenticated producer chain

The builder no longer accepts launcher-invented
`uuid/cell/language/source_native` rows or a public `--input` escape hatch. It
requires a digest-pinned Task8 publication, its copied genuine Task9 schema-v3
selection receipt, the exact Task8 source commit, and the Task9 B projection.
Task8 rows are consumed in their producer schema:
`prompt_uuid/domain/lane/context_bucket/input_ids/loss_mask/assistant_tokens/`
`rejection_reason/record_json`.

The loader authenticates `PUBLICATION.json`, `CORPUS_MANIFEST.json`, the copied
Task9 selection receipt, exactly 201 declared shard descriptors, the manifest
file records, and missing/orphan shard state. Production semantics remain the
2M `500K/400K/500K/400K/200K` B mix with DE/JA/ES/FR/IT at 40K each.

## CPU materialization and durability

The `cpu_datamover` runner requests one task with 96 CPUs and zero GPUs. The
worker count defaults from `SLURM_CPUS_PER_TASK`, caps at the 201 declared
shards, and fixes Arrow/OMP/BLAS/MKL/NumExpr threads to one per worker. Each
worker opens its Task8 shard once with `O_NOFOLLOW`, hashes and parses through
that same descriptor, and rejects inode, namespace, size, timestamp, row-count,
or manifest-digest changes. A disk-backed SQLite merge preserves deterministic
rank order, bounded memory, and byte-identical one-worker/multi-worker output.

The output is one immutable directory containing `canary.jsonl` and
`BUILD_RECEIPT.json`. A job-unique sibling partial is fsynced, reread, and
published with an OS no-replace rename. No failure path deletes pathname state
or claims best-effort cleanup. Rename and parent-fsync ambiguity carries typed
partial/destination inode observations for explicit recovery.

## Transitive runtime authorization

The build receipt makes its source, output SHA, worker/CPU/thread settings, and
per-shard timing provenance mandatory. Readiness rehashes both builder receipt
and output and binds their exact SHA-256 identities. The 16-node OCI-HSG GPU
manifest binds those same identities plus readiness, source commit, GBS512,
8 serve + 8 train nodes, 4 GPUs per node, PDB4, GA4, 200 steps, and
`sna-qwen3-4b-dataset-study`.

The submitter schedules the 96-core builder first and submits only the bounded
GPU canary with `afterok:<builder_job_id>`. The GPU runner revalidates readiness,
builder receipt, and builder output immediately before any `srun`. Runtime
evidence still requires finite loss, checkpoint reload, drafter export,
evaluator completion, and all 64 GPU ranks; it cannot become a scientific
milestone.

## Verification

- Task10 launcher suite: 17 passed, including a genuine Task9 writer → Task8
  201-shard publisher → CPU builder → readiness → GPU manifest/runner-preflight
  integration test, deterministic 1-worker/parallel bytes, shard namespace
  race rejection, typed fsync recovery, worker-failure propagation, and real
  fake-Slurm `afterok` job wiring.
- Task9/Task8 regression suite (`qwen3_4b_ptv2_study`,
  `build_assistant_token_views`, and `specdec_publication`): 76 passed.
- Ruff check/format, Pyright, Bash syntax, ShellCheck, diff check, and the final
  combined regression commands are recorded in the signed commit handoff.

No cluster job was submitted and no branch was pushed by this implementation.
