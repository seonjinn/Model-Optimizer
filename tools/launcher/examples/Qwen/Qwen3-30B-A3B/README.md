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

## Reusable OCI-HSG waves

For production runs, create the canonical JSON manifest under `/home` with the
typed `drafter_job_manifest.py` API, then render each four-node wave from that
manifest. The manifest pins the clean `/home` source SHA, image/runtime archive,
model, dataset, stable training output root, and cumulative boundaries.

```bash
export MANIFEST=/home/$USER/drafter-manifests/qwen3-30b-base-dflash-b8.json
export RECEIPT=/lustre/$USER/specdec/receipts/qwen3-30b-base-dflash-b8.jsonl
bash common/specdec/submit_drafter_training_wave.sh \
  --manifest "$MANIFEST" --receipt "$RECEIPT" --dry-run
```

Remove `--dry-run` only after the command's built-in `sbatch --test-only`
passes. Every production training allocation is exactly `-N4 --segment=4`:
nodes 0--1 are the two four-GPU vLLM serve replicas and nodes 2--3 are the two
four-GPU trainer nodes. The pinned global batch settings are:

- Q30 uses per-device batch 4 with gradient accumulation 16.
- Q235 uses per-device batch 2 with gradient accumulation 32.

Both satisfy `per-device batch * accumulation * 8 trainer GPUs = 512`.
`segment=4` keeps the allocation within one OCI-HSG NVL72 segment; staging and
public evaluation use `--segment=1`.

Use the matching target and manifest identity for each public case:

| Case | Target path | Method mapping |
| --- | --- | --- |
| Q30 Base | `/lustre/models/Qwen/Qwen3-30B-A3B` | DFlash B8/K7, B16/K15; DSpark B8/K8, B16/K16 |
| Q30 Thinking | `/lustre/models/Qwen/Qwen3-30B-A3B-Thinking-2507` | DFlash B8/K7, B16/K15; DSpark B8/K8, B16/K16 |
| Q235 Base | `/lustre/models/Qwen/Qwen3-235B-A22B` | DFlash B8/K7, B16/K15; DSpark B8/K8, DSpark B16/K16 |
| Q235 Thinking | `/lustre/models/Qwen/Qwen3-235B-A22B-Thinking-2507` | DFlash B8/K7, B16/K15; DSpark B8/K8, DSpark B16/K16 |

The resume-chain entrypoint retains the same Lustre training `output_root`; it
submits `train -> 9-subset acceptance evaluation -> next train` with `afterok`
links. Each chunk requests `--time=03:55:00`, exports only its corresponding
`exported-checkpoint-<cumulative-step>`, and requires an `acceptance.csv`
before the next chunk becomes eligible. Store chain receipts beneath
`/lustre/$USER/specdec/receipts/`; they are the authoritative scheduler job-ID
record, including scheduler lookup when `sbatch` returns blank output.

## Cache and inode policy

- Keep source and JSON configuration under `/home`. Keep only immutable images,
  staged datasets/models, checkpoints, acceptance CSVs, and durable receipts on
  Lustre.
- Every mutable runtime extraction, Hugging Face cache, W&B/Triton/TorchInductor
  cache, SQLite database and locks, and temporary file belongs under
  `/raid/scratch/$SLURM_JOB_ID` on each node. A shared Lustre virtualenv or
  cache is forbidden.
- Inputs are copied once per node to `/raid/scratch` before ranks start. Do not
  recursively scan a Lustre tree, clone source, or build/install packages in a
  job.
- The staging and submission scripts query only the relevant scheduler job
  names/IDs and are intended to be monitored no more often than once every 60
  seconds.

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
