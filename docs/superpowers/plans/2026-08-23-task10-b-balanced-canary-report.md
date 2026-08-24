# Task 10 B-balanced canary implementation report

## Scope

This change implements only the B-balanced readiness and bounded-canary path.
It intentionally does not import or modify Task9 core code, and it has no
A-repair input or dependency.

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

## Validation

The red test run failed because the new B builder module was absent. Fresh
verification executed `PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=tools/launcher
python3 -m pytest -q --disable-warnings tools/launcher/tests/test_qwen4b_b_canary.py
tools/launcher/tests/test_qwen4b_dataset_study.py` and reported 36 passing
tests. `bash -n` and `shellcheck -S warning` completed for both shell launchers.
`ruff` was not available in the local Python environment.
