# Qwen3 DFlash and DSpark training

These launchers stream target-model hidden states into a five-layer ModelOpt
drafter. Run them from `tools/launcher`. The 30B and 235B launchers also work
for their Thinking variants because those variants retain the corresponding
model topology; override `pipeline.global_vars.hf_model` with the staged
checkpoint.

| Target family | DFlash launcher | DSpark launcher | Example target override |
| --- | --- | --- | --- |
| 30B-A3B | `examples/Qwen/Qwen3-30B-A3B/hf_streaming_dflash_multi_node.yaml` | `examples/Qwen/Qwen3-30B-A3B/hf_streaming_dspark_multi_node.yaml` | `/hf-local/Qwen/Qwen3-30B-A3B-Thinking-2507` |
| 235B-A22B | `examples/Qwen/Qwen3-235B-A22B/hf_streaming_dflash_multi_node.yaml` | `examples/Qwen/Qwen3-235B-A22B/hf_streaming_dspark_multi_node.yaml` | `/hf-local/Qwen/Qwen3-235B-A22B-Thinking-2507` |

Use the 30B launcher only with a 30B base or Thinking target, and likewise for
235B. The checked-in defaults select the base models.

## Experiment controls

Both methods default to 50,000 source rows, seed 42, one epoch capped at 2,000
steps, and W&B logging. Give every dataset/method/block-size run a unique
`run_name`. W&B model upload and parameter watching are disabled, and the run
directory is node-local at `/tmp/wandb`.

| Method | Block `B` | Training objective | vLLM draft tokens `K` |
| --- | ---: | --- | ---: |
| DFlash | 8 | D-PACE, alpha 0.5, decay gamma 0 | 7 |
| DFlash | 16 | D-PACE, alpha 0.5, decay gamma 0 | 15 |
| DSpark | 8 | decay, gamma 4 | 8 |
| DSpark | 16 | decay, gamma 7 | 16 |

DFlash predicts `K = B - 1` speculative tokens, whereas DSpark predicts
`K = B`. Keep `dflash_block_size`, the method-specific loss parameter, and
`num_spec_tokens` coupled as shown above.

## Direct OpenPerfectBlend or Nemotron data

For a staged JSON, JSONL, Parquet file, or directory of such files, skip the
dataset-building task and point `hf_data` directly at the container-visible
path. For example, set one of these paths at the shell:

```bash
export DATA_PATH="${SHARED_DATA_ROOT}/open-perfectblend/train-00000-of-00006.parquet"
# Or:
export DATA_PATH="${SHARED_DATA_ROOT}/nemotron-first50k-compatible/train.parquet"
```

`sample_size=50000` takes a bounded, deterministic prefix before training; use
the same source layout and seed for comparable OpenPerfectBlend and Nemotron
runs. Use a prevalidated, schema-compatible first-50K view for Nemotron. Do not
point a launcher at the full heterogeneous Nemotron directory: schema outliers
are not a valid training sample, and revisions without bounded streaming also
prepare the entire directory before applying `sample_size`. Set
`pipeline.task_0.skip=true` whenever using `DATA_PATH` directly.

## Dry run, then submit

Stage one immutable ARM64 vLLM `.sqsh` and one shared ModelOpt Python runtime.
Do not use the YAMLs' mutable image defaults for a production run. This complete
30B base, OpenPerfectBlend, DFlash-B8 example supplies both artifacts as typed
overrides:

```bash
export PINNED_ARM64_SQSH=/path/to/shared/vllm-openai-aarch64.sqsh
export MODELOPT_RUNTIME_PATH=/path/to/shared/modelopt-runtime
export DATA_PATH="${SHARED_DATA_ROOT}/open-perfectblend/train-00000-of-00006.parquet"
export TARGET_MODEL=/hf-local/Qwen/Qwen3-30B-A3B
export RUN_NAME=qwen3-30ba3b-dflash-b8-opb-seed42

uv run --frozen python launch.py \
  --yaml examples/Qwen/Qwen3-30B-A3B/hf_streaming_dflash_multi_node.yaml \
  pipeline.task_0.skip=true \
  pipeline.global_vars.hf_data="${DATA_PATH}" \
  pipeline.global_vars.hf_model="${TARGET_MODEL}" \
  pipeline.global_vars.sample_size=50000 \
  pipeline.global_vars.run_name="${RUN_NAME}" \
  pipeline.global_vars.modelopt_runtime="${MODELOPT_RUNTIME_PATH}" \
  pipeline.global_vars.dflash_block_size=8 \
  pipeline.global_vars.dflash_dpace_alpha=0.5 \
  pipeline.global_vars.num_spec_tokens=7 \
  pipeline.task_1.slurm_config.container="${PINNED_ARM64_SQSH}" \
  pipeline.task_2.slurm_config.container="${PINNED_ARM64_SQSH}" \
  --dryrun --yes -v
```

Inspect the resolved tasks and Slurm request. Only after that succeeds, submit
the identical command without `--dryrun` and `-v`:

```bash
uv run --frozen python launch.py \
  --yaml examples/Qwen/Qwen3-30B-A3B/hf_streaming_dflash_multi_node.yaml \
  pipeline.task_0.skip=true \
  pipeline.global_vars.hf_data="${DATA_PATH}" \
  pipeline.global_vars.hf_model="${TARGET_MODEL}" \
  pipeline.global_vars.sample_size=50000 \
  pipeline.global_vars.run_name="${RUN_NAME}" \
  pipeline.global_vars.modelopt_runtime="${MODELOPT_RUNTIME_PATH}" \
  pipeline.global_vars.dflash_block_size=8 \
  pipeline.global_vars.dflash_dpace_alpha=0.5 \
  pipeline.global_vars.num_spec_tokens=7 \
  pipeline.task_1.slurm_config.container="${PINNED_ARM64_SQSH}" \
  pipeline.task_2.slurm_config.container="${PINNED_ARM64_SQSH}" \
  --yes
```

To run 30B DSpark-B16, select the DSpark YAML and replace the three coupled
method overrides with:

```text
pipeline.global_vars.dflash_block_size=16
pipeline.global_vars.dflash_loss_decay_factor=7
pipeline.global_vars.num_spec_tokens=16
```

For DFlash-B16 use block size 16 and `num_spec_tokens=15`; alpha remains 0.5.
For DSpark-B8 use block size 8, gamma 4, and `num_spec_tokens=8`. To run 235B,
select the matching 235B YAML and 235B base or Thinking target path. The direct
Nemotron experiment changes only `DATA_PATH` and `RUN_NAME`.

## Cache and inode policy

- Keep `HF_HOME`, `UV_CACHE_DIR`, and `PIP_CACHE_DIR` in stable shared cache
  directories that do not contain a Slurm job ID. Reuse them across launches.
- Reuse the immutable `.sqsh` and `MODELOPT_RUNTIME_PATH`; do not create a new
  environment or install packages for every job or node.
- Keep writable W&B and Triton state node-local. The launchers set
  `WANDB_DIR=/tmp/wandb`; configure `WANDB_CACHE_DIR` and `TRITON_CACHE_DIR`
  under node-local `/tmp` as well. Do not place per-job W&B, Triton, Ray, or
  temporary-environment trees in a shared Lustre `.cache` directory.
- Never delete the shared content caches from a training job.

## Benchmark output

The launcher's final task is only a load/generation smoke test. Use the pinned
speculator-evaluation launcher maintained with the experiment workflow for the
publication run; do not infer per-position rates from an engine's aggregate
acceptance histogram. Preserve its normalized acceptance CSV, raw evaluator
output, resolved benchmark config, timings, target/drafter commit IDs, training
YAML, and W&B URL.

The report table groups benchmark categories into `Pos 0` through `Pos 7` and
`Avg. Length`. DSpark-B8 supplies all eight positions. DFlash-B8 has only seven
draft tokens, so its `Pos 7` value must be `N/A`, not zero. Use the evaluator's
explicit position counters as the source of truth for every position.
