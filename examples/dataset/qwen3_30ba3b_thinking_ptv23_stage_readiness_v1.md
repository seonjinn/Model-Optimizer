# Qwen3-30B-A3B Thinking PTV2/PTV3 staging readiness

This record freezes the source-staging inputs for the simplified 700K continuation corpus. It prepares source material only; it does not assert that exclusion, deduplication, tokenization, corpus publication, or training has completed.

## PTV3 stage-ready inventory

- Stage plan: `qwen3_30ba3b_thinking_ptv3_stage_subset_v1.json`
- Plan SHA256: `752090878ed2c5fce47683b2939f9d857be5549311b158476fce33bb79f52eac`
- Stage container: `/lustre/fsw/coreai_dlalgo_llm/users/sna/containers/nemo2606/nemo_rl_nightly_nemo2606_20260812_2574659.sqsh`
- Stage container SHA256: `ab3380e548e5c62aa0bbaeaba3d1b47896151868f74e5859ac4eb311f1a069ab`
- Container provenance source commit: `6ede0dc763c77fb6c26349b0a617a8abe2985e95`
- Ptyche dependency probe: job `2665866`, `COMPLETED 0:0`, `pyarrow 24.0.0`
- `nvidia/Nemotron-SFT-SWE-v3` revision: `3f73de64c1fe928a8f538fe45ccc10c228cc4c6a`
- SWE-v3 inventory: all 96 `data/train-xxxxx-of-00096.parquet` files in source order
- SWE-v3 aggregate source bytes: `11,677,930,655`
- Interactive agentic/SWE lane: `nvidia/Nemotron-SFT-SWE-v2` `data/swe.jsonl` at revision `bd151f3f2d89c4804dda0083d912bd9f6a0a9fb7`, plus `nvidia/Nemotron-SWE-v1` `data/r2e_gym.jsonl` at revision `0fe17a965b297a9c943a59050a14c42d5f0083ce`
- General tool lane: `nvidia/Nemotron-Agentic-v1` `data/interactive_agent.jsonl` and `data/tool_calling.jsonl` at revision `650d590978ca35c8f1ecea2faf136e5fac421b62`
- Complete PTV3 plan: 100 files and `39,956,713,421` source bytes
- Every PTV3 file has an exact official LFS SHA256 and byte size. The plan is accepted directly by `tools/launcher/common/specdec/stage_hf_subset.py`.

The SWE-v3 inventory was obtained mechanically from the official Hugging Face tree API at the pinned revision. Regenerating at the same revision must reproduce the exact file paths, LFS SHA256 values, sizes, aggregate, and plan digest.

The first staging attempt, Ptyche job `2665853`, used the vLLM 0.27.1 runtime image and failed before publication because that image does not contain `pyarrow`. The replacement image above was selected only after the exact 128 GiB probe job `2665866` imported `pyarrow 24.0.0` successfully. A preliminary 16 GiB probe (`2665863`) was rejected as inconclusive because enroot extraction of the 29.8 GB image was OOM-killed before Python started.

## PTV2 full-201 bootstrap inventory

- Bootstrap inventory: `qwen3_30ba3b_thinking_ptv2_full201_stage_inventory_v1.json`
- Inventory SHA256: `af7983a73284fd0145209a2e8ca390717477538fd2e40de86b1756137ceb78b5`
- Repository: `nvidia/Nemotron-Post-Training-Dataset-v2`
- Revision: `5c89e01dd720ae0f4058445ed49c5fb68a03c76e`
- Configuration/license: `default` / `CC-BY-4.0`
- Approved production inventory: 201 Parquet files and `44,423,886,661` source bytes
- Split counts: chat 12, math 2, code 2, stem 2, multilingual_de 38, multilingual_ja 37, multilingual_es 33, multilingual_fr 37, multilingual_it 38

The pinned repository also exposes one generic `data/multilingual-00000-of-00001.parquet` file. It is intentionally excluded because the approved production bootstrap contract is the category-specific 201-file view above.

The unauthenticated official tree API returns exact paths and sizes but redacts all gated PTV2 LFS object IDs. Consequently, the inventory records `sha256: null` and `source_sha256_status: gated-redacted-require-bootstrap`; these are not guessed hashes. Actual staging remains blocked until the existing immutable cluster symlink view is hashed by `examples/dataset/bootstrap_ptv2_source_manifest.py`, which publishes the final per-file SHA256 source plan and completion receipt.

## Reproduction

Generate both plans from official metadata:

```bash
python3 examples/dataset/generate_q30t_ptv23_stage_plans.py \
  --sources examples/dataset/qwen3_30ba3b_thinking_ptv23_complement_sources_v1.json \
  --ptv3-output examples/dataset/qwen3_30ba3b_thinking_ptv3_stage_subset_v1.json \
  --ptv2-output examples/dataset/qwen3_30ba3b_thinking_ptv2_full201_stage_inventory_v1.json
```

If `HF_TOKEN` grants access to the gated PTV2 dataset, the generator records official LFS SHA256 values and changes the PTV2 hash status to `official-lfs-pinned`. Without that credential, the explicit bootstrap blocker is preserved.

Validation used for this revision:

```bash
python3 -m pytest -q -o addopts='' --confcutdir=tests/examples/dataset \
  tests/examples/dataset/test_generate_q30t_ptv23_stage_plans.py
ruff check examples/dataset/generate_q30t_ptv23_stage_plans.py \
  tests/examples/dataset/test_generate_q30t_ptv23_stage_plans.py
pyright examples/dataset/generate_q30t_ptv23_stage_plans.py \
  tests/examples/dataset/test_generate_q30t_ptv23_stage_plans.py
```
