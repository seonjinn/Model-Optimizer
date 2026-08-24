# Task 10 B-balanced canary implementation report

## Scope

This change implements only the B-balanced readiness and bounded-canary path.
It intentionally does not import or modify Task9 core code, and it has no
A-repair input or dependency.

The B builder and readiness artifacts are reusable and may be prepared while
other work runs. They do not authorize B as the first production training arm.
Scientific execution remains A-repair, then B-balanced, then
B-balanced-target, then PTV3 SWE/agentic/tool unless a later explicit ruling
changes that order.

## Task9 integration seam

`Task9BalancedView` in
`tools/launcher/common/specdec/qwen4b_b_readiness.py` is the sole adapter
contract. Task9 must publish a JSON projection with the exact fields accepted
by `load_task9_balanced_view`: B strategy, occurrence/cell/language counts,
declared shard names and root, trainer/schedule fields, source-native flag, and
publication/destination/tokenizer/template/loss-mask SHA-256 identities.
The Task10 code does not assume a Task9 implementation type or import it.

## Enforced gates

- B only: 2M occurrences; `500K/400K/500K/400K/200K` cells; five `40K`
  language cells; source-native responses; one trainer epoch; GBS512; and the
  `2540 + 1368 = 3908` schedule.
- Exactly 201 declared JSONL/Parquet shards. Missing, duplicate, unsafe, and
  orphan data shards produce stable blocker codes.
- The materialized canary has exactly 102,400 occurrences: Math 25,600, Code
  20,480, STEM 25,600, Chat 20,480, Multilingual 10,240, with 2,048 each for
  DE/JA/ES/FR/IT.
- OCI-HSG topology is fixed at segment 16: 8 serve and 8 train nodes, 4 GPUs
  per node, cpu_datamover 96, 32 trainer ranks, PDB4, GA4, GBS512, and 200
  steps. The W&B project is `sna-qwen3-4b-dataset-study`.
- Runtime evidence must show finite loss, checkpoint reload, drafter export,
  evaluator completion, and activity on ranks 0 through 63. It always records
  `scientific_milestone: false`.
- The submitter permits only `sbatch --test-only` on approved accounts and
  rejects any production submission request.

## CPU datamover materialization

`run_qwen4b_b_builder.sbatch` requests one `cpu_datamover` node with 96 CPUs
and passes the full `SLURM_CPUS_PER_TASK` allocation to the builder. The
builder caps its process count by the CPU allocation and the 201 declared
shards. Every worker fixes Arrow, OMP, BLAS, MKL, and NumExpr threads to one,
streams one shard into one job-local spool, and reports source identity and
timing provenance. The coordinator consumes those spools in declared-shard
order through a disk-backed SQLite selection index, so open files and memory
stay bounded and one-worker and multi-worker output bytes remain identical.
Worker failures remove scratch state and prevent either output or receipt
publication.

## Validation

The original red test run failed because the new B builder module was absent.
The CPU-parallel extension also recorded red gates for missing worker
resolution, missing materialization, and the missing sbatch runner. A cleanup
mutation then proved the failure-propagation assertion fails if scratch state
is retained.

Fresh focused verification reports 9 passing Task10 tests and 31 passing
Task9 regression tests. The Task10 suite includes exact one-worker versus
multi-worker output-byte identity, 96-CPU allocation propagation, provenance,
and failure cleanup. `ruff check`, `ruff format --check`, `pyright`, `bash -n`,
`shellcheck -S warning`, and `git diff --check` complete without findings.
