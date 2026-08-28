# Q30 Thinking Conservative Held-Out V2 Design

## Decision

Use a conservative, pinned public benchmark superset as the held-out boundary
for the Qwen3-30B-A3B-Thinking SWE-heavy 700K complement. The builder excludes
only candidate rows that match authenticated evaluation material, then scans
forward inside the same quota category until the exact quota is filled.

This removes the dependency on missing internal evaluation-run artifacts
without weakening the exclusion boundary. It does not select DFlash or DSpark,
change block size 8, alter either step-25,391 parent, or authorize training. The
same approved 700K dataset must be consumable by both model methods. Model
choice, runtime qualification, canaries, and training launch remain separate
decisions.

The design introduces the manifest schema `q30t-held-out-sources-v2`, the
consumer receipt schema `specdec-held-out-receipt-v2`, and a content containment
index. The five canonical held-out names and their union order are fixed as:

1. `speed`
2. `math`
3. `code`
4. `swe`
5. `tool`

No missing name, empty receipt, unpinned source, floating branch, or unsigned
review can be interpreted as an empty exclusion set.

## Scope and immutable training contract

The scientific identity remains
`ptv2-ptv3-complement-700k-qwen3-30ba3b-thinking-swe-heavy-v1`.
This design changes only the evidence used to exclude held-out evaluation
material and the builder identity implementation required to consume it.

| Parent pool and category | Required rows |
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

The builder may not redistribute across categories, replace categories, pad,
cycle, replay a source occurrence, or admit a duplicate. Held-out filtering is
performed before quota admission. A rejected row causes deterministic forward
selection from the same category only.

The target tokenizer revision remains
`144afc2f379b542fdd4e85a1fcd5e1f79112d95d`. Its approved receipt SHA-256 is
`5ba642c455e60b67eca295dce92dd7da47292fdba66c5f9d269669c14cafc509`.
The authenticated historical audit SHA-256 remains
`2469430c144d9b86850901df0a28cb810b437555ee894771b796b1386d1c18b5`,
with ordered occurrence digest
`863470b22925d74228d31b1c2433d9461d25e8d02a298cb3a08a6bc99be55060`.

## Threat model

The defended failure modes are:

- an evaluation prompt appears exactly in a training row;
- an answer, solution, code body, patch, test patch, test, policy, or tool schema
  is embedded inside a longer training message;
- whitespace or line-ending changes defeat an otherwise exact comparison;
- a whole-message identity includes assistant completions and therefore differs
  from the historical prompt identity;
- a public repository moves while a floating revision is still accepted;
- a renderer, field map, split, or time window changes under the same source
  name;
- a source is substituted after it was hashed or between validation and use;
- a missing or failed component silently produces an empty receipt;
- a producer approves its own output or an approval is copied between catalogs;
- a large, malformed, linked, or rebound input bypasses resource or path checks;
- publication overwrites a foreign artifact or cleanup deletes foreign data.

This boundary is intentionally conservative. A deterministic exact-content
collision excludes a candidate even when it might be coincidental. The design
does not use edit distance, semantic embeddings, MinHash, stemming, case
folding, language-model judgment, or any other fuzzy rule whose false-positive
and false-negative behavior cannot be replayed byte-for-byte.

Out of scope are undisclosed private evaluations, future benchmark releases,
memorization that shares meaning but no reproducible exact content, parent
checkpoint validation, and runtime row A/B qualification. An evaluation owner
who wants an additional private set must publish a separately authenticated
component; absence of that set never permits the public catalog to claim that
it covers the private run.

## Success criteria

The held-out design succeeds only when all of the following are true:

1. all five canonical catalogs are nonempty and validate with outcome `A`;
2. every component is bound to immutable source bytes, a split or selector,
   field extraction, and renderer identities;
3. two independent replays reproduce each component and catalog digest;
4. the producer and reviewers are distinct identities trusted by an approved
   signing keyring;
5. candidate prompt identity is exactly `specdec_identity.prompt_uuid` at every
   historical, held-out, selection, deduplication, and replay boundary;
6. auxiliary exact and containment checks report zero admitted collisions;
7. the final receipt proves 300,000 PTV2 rows, 400,000 PTV3 rows, 700,000 unique
   prompt identities, and 700,000 unique normalized full-row identities;
8. a clean replay reproduces the data file, manifest, capacity evidence,
   exclusion ledger, and completion receipt byte-for-byte; and
9. no approval field is inferred from a filename, directory, job exit code, or
   producer assertion.

Meeting these criteria authorizes only the dataset evidence transition. It
does not authorize a canary or training job.

## Source evidence and status

All sources initially validate with outcome `P`. The hashes below freeze public
source facts for acquisition. They are not approval roots. A catalog validates
as `A` only with the separate approval root defined later.

### Speculators compatibility catalog

The evaluator is pinned to `vllm-project/speculators` revision
`0b08a89a83b92007be63f128e01497455b0209df`. Its
`scripts/evaluate/evaluate.py` declares exactly nine subsets from
`RedHatAI/speculator_benchmarks`. The source dataset revision is
`2ae86affa2cb97a972b7fc681dd51c04fbff083e`.

| File | Rows | Bytes | SHA-256 | Canonical catalog |
| --- | ---: | ---: | --- | --- |
| `HumanEval.jsonl` | 164 | 214,438 | `1d49078ba3e2b196b9344535bef34a43021f038fad9561d6ee7c53450609a6a2` | `code` |
| `math_reasoning.jsonl` | 80 | 48,051 | `b73b686f1c585eef7db1306564688da8668ef633ff8e58d55949aac04db8c9b0` | `math` |
| `qa.jsonl` | 80 | 8,127 | `17c7c2fb9f36abd16e3e2a8ab5487ae678e9762d96cb9736eec442cbaba9f506` | `speed` |
| `question.jsonl` | 80 | 40,017 | `9db8ea4678e058c26f21a9cc0bf79f95a9750e370cafb4f90a8e3b058c2a2aa4` | `speed` |
| `rag.jsonl` | 80 | 258,268 | `d1644a209c444e1dd7ff65bf3ce9e8a4daa5d609ac93cbf4f6bebc10400d8d91` | `speed` |
| `summarization.jsonl` | 80 | 302,603 | `e7d2945062f77aae25553f7f7531deefc358f5d0b87b9c623e6ba8ba555ff1be` | `speed` |
| `tool_call.jsonl` | 200 | 483,081 | `0f9ae4d22a55fee9f33c80cf223c37974826a2997b515d0a55cfa778852979eb` | `tool` |
| `translation.jsonl` | 80 | 28,915 | `bd27352726c318b72a263158672b0ff0c15feb459ae735f723b27623a0ba595d` | `speed` |
| `writing.jsonl` | 80 | 40,017 | `9db8ea4678e058c26f21a9cc0bf79f95a9750e370cafb4f90a8e3b058c2a2aa4` | `speed` |

The frozen total is 924 records and 1,423,517 bytes. The duplicated bytes for
`question.jsonl` and `writing.jsonl` are retained as two provenance components
because both evaluator subset names exist, while their derived identities are
set-deduplicated. The measured `HumanEval.jsonl` count is 164; a prose claim of
168 must not override the pinned bytes.

The compatibility renderer is the pinned evaluator's
`generative_column_mapper` contract: one user message containing the exact
decoded `prompt` string and no synthesized system message. `tool_call` remains
the exact evaluator prompt, including its embedded available-functions block;
the JSON within that block is also extracted into the `tool` content domain.
References and code fields never enter prompt identity, but do enter auxiliary
content identities.

### `math` catalog

In addition to Red Hat `math_reasoning`, `math` is the ordered union below.

| Component | Immutable selection | Rows | Bytes | SHA-256 |
| --- | --- | ---: | ---: | --- |
| GSM8K | `openai/gsm8k` revision `740312add88f781978c0658806c59bc2815b9866`, config `main`, split `test`, `main/test-00000-of-00001.parquet` | 1,319 | 419,088 | `ee7b8da9e381df27b9e3f7758a159ab2bdaa4dbaa910546cbbc47e0cb44e4f59` |
| MATH-500 | `HuggingFaceH4/MATH-500` revision `6e4ed1a2a79af7d8630a6b768ec859cb5af4d3be`, split `test`, `test.jsonl` | 500 | 446,564 | `35dc41080a3680858b27fa7e0533d2d547825316fc5dafe5d316f4ccc5a06132` |
| AIME 2022-2025 | `allenai/aime-2022-2025` revision `73e1eba765ad5847cdb5d1e2e7aaf7b22b585798`, split `train`, `data/train-00000-of-00001.parquet`, filter `year` in `[2022,2023,2024,2025]` | 120 | 285,002 | `cc1d76180fa27dc06cc0060ff03010c5f41331ca1c70cc9247e87e8eef0c5a0d` |

The AIME filter must yield exactly 30 records for each named year. GSM8K maps
`question` to problem and `answer` to answer. MATH-500 maps `problem` to
problem and both `solution` and `answer` to answer domains. AIME maps `problem`
to problem and `solution` plus `answer` to answer. Prompt rendering is one user
message containing the problem string exactly; no answer is rendered into the
prompt.

### `code` catalog

In addition to Red Hat `HumanEval`, `code` is the ordered union below.

| Component | Immutable selection | Rows | Bytes | SHA-256 |
| --- | --- | ---: | ---: | --- |
| HumanEval | `openai/openai_humaneval` revision `7dce6050a7d6d172f3cc5c32aa97f52fa1a2e544`, split `test`, `openai_humaneval/test-00000-of-00001.parquet` | 164 | 83,920 | `2f2871a15fbc95b6c683043359f4ed8e144c5a1c4f24f25f66bc51f598dfcfb6` |
| MBPP prompt | `google-research-datasets/mbpp` revision `4bb6404fdc6cacfda99d4ac4205087b89d32030c`, config `sanitized`, split `prompt`, `sanitized/prompt-00000-of-00001.parquet` | 7 | 6,717 | `73c623309b7b5d65fd5661204b35f779f8e66301aa9832d1ad4b8fc3b21151fd` |
| MBPP test | same revision and config, split `test`, `sanitized/test-00000-of-00001.parquet` | 257 | 60,864 | `e9e9efa2c0d59ef5e55537a9d126b8f875d5ac010a8d75628d76824884e15850` |
| MBPP train | same revision and config, split `train`, `sanitized/train-00000-of-00001.parquet` | 120 | 33,854 | `d95f8ad6d2fff08fe4826122d6e3e31f75716825d0c5c340d297aca5e9e0de0e` |
| MBPP validation | same revision and config, split `validation`, `sanitized/validation-00000-of-00001.parquet` | 43 | 13,987 | `27e065fcab3c863959933328a7fdbf404e1bcb5464b1be6fe0dcd9530e420204` |

The official sanitized MBPP union is therefore 427 rows, not a remembered
round number. HumanEval extracts `prompt`, `canonical_solution`, `test`, and
`entry_point`. MBPP extracts `prompt`, `code`, `test_imports`, and every
`test_list` element. The prompt renderer creates one user message from the
exact prompt only.

LiveCodeBench is the exact named component
`livecodebench/code_generation_lite`, revision
`0fe84c3912ea0c4d4a78037083943e8f0c4dd505`, release selector `release_v6`,
covering May 2023 through April 2025 and exactly 1,055 problems. Its six source
files are:

| File | Bytes | SHA-256 |
| --- | ---: | --- |
| `test.jsonl` | 1,252,609,773 | `2bd02b38beb48e8c46b5b9987095d999ff38cd8efc255ea5d58974317c48f63f` |
| `test2.jsonl` | 713,377,060 | `095df7c5daf15f882c51a9deb84085cff1e073495a5dbcf95015a564d485f3a3` |
| `test3.jsonl` | 623,360,766 | `28ed26cc83363ce3f1fe2d5fad9f8393077beb1907b167a31bd3b32f80801b79` |
| `test4.jsonl` | 1,204,644,685 | `d711138ddaebfcf5f8ec6a4283ee677298c0f5c5d374a235af92aaf0584510da` |
| `test5.jsonl` | 557,699,297 | `7f77571c2a6df0c2a72a3277650309f67e01e0008e18117e624633df53f81214` |
| `test6.jsonl` | 134,303,240 | `bb4c364f71921c4495a6ad15abe1a927350b720009f4933e2e71f8af0f6fd1f5` |

Release selection must be replayed from the pinned release metadata, not by
taking the first 1,055 rows. The renderer extracts the exact public problem,
starter code, public and private test representations present in the source,
reference code, and metadata-selected prompt. Because these files have not yet
been independently staged and decoded in the durable environment, the
LiveCodeBench remains outcome `P`; `code` cannot validate as `A` until its
acquisition receipt proves all six hashes, the `release_v6` membership digest,
the 1,055 count, and the renderer result.

### `swe` catalog

Use the full SWE-bench Test set, which conservatively covers Lite and Verified:

- repository: `princeton-nlp/SWE-bench`;
- revision: `e48e2bd1e9fecd5bbd641e9414ac59da9f2e69f6`;
- split: `test`;
- file: `data/test-00000-of-00001.parquet`;
- rows: 2,294;
- bytes: 12,097,227;
- SHA-256:
  `db4f70ef735b3162c74801ddcdf8d7bae8d704193788c6d844f898c20b571cbb`.

For every row the producer records identities for `problem_statement`,
`patch`, `test_patch`, and the complete canonical raw record. It also binds
`instance_id`, `repo`, and `base_commit` as provenance context. Prompt identity
is one user message containing `problem_statement`; answer-domain material is
never added to that prompt. Full Test is one component; Lite and Verified are
not separately unioned or double-counted.

### `speed` catalog

Use `nvidia/SPEED-Bench` revision
`487aa718444e816458d1a0a52bfce7a454285cf4`, config `qualitative`, split
`test`, file `qualitative/test-00000-of-00001.parquet`, exactly 880 rows,
364,138 bytes, SHA-256
`4f76bc45bdffb38712a5c26d7a7c9a9a791484b392a7b64a6ffb9abe48e02e26`.

The pinned bytes contain 494 deferred-source marker rows. They are not usable
as literal prompts. Deterministic resolution must reproduce all 880 questions
and bind every upstream dependency by immutable repository revision, path,
file SHA-256, selected record key, and renderer identity. The observed source
counts are:

| Upstream family | Rows |
| --- | ---: |
| `cais/hle` | 208 |
| `Spec-Bench` | 201 |
| `cnn_dailymail` | 80 |
| `WritingBench` | 66 |
| `humanevalpack` | 61 |
| `mtbench101` | 58 |
| `RoleBench` | 56 |
| `MMATH` | 45 |
| `opus100` | 33 |
| `MMLU-Pro` | 18 |
| `ChatRAG` | 17 |
| `CoSER` | 14 |
| `EQ-bench` creative writing | 10 |
| `code_contests` | 5 |
| LiveCodeBench | 5 |
| `MCIF` | 3 |

The observed EQ-bench dependency uses floating `refs/heads/main`; therefore
the public SPEED component cannot validate as `A` as observed. The resolver
must either bind those ten rows to reviewed immutable upstream bytes or consume
a reviewed internal prepared artifact containing all 880 resolved rows. That
artifact must itself carry the complete dependency manifest and may not claim
an unpinned internal evaluation run.

The current reference resolver is
`examples/specdec_bench/specdec_bench/datasets/speed.py` at ModelOpt revision
`8e195c81a086bc8e26e10f84d5bf8ab0b73ea429`, Git blob
`552a537a176f266e3f2a4dcb3de23cdb05575b8d`, and file SHA-256
`c0b24776b7ceb7f5eba74b945a132c2a2c860121d08599ac791d0715b513602a`.
An approved acquisition must additionally bind the final renderer code and
environment lock. Failure to pin or resolve one SPEED dependency fails only
the `speed` validation and emits no receipt for that catalog.

The six general Red Hat compatibility subsets assigned to `speed` are unioned
after the resolved 880-row SPEED component in stable component-ID order.

### `tool` catalog

The tool/agentic catalog owner explicitly selects the Red Hat `tool_call`
component above plus these public components. No internal run is implied.

Tau2 is pinned to `sierra-research/tau2-bench` revision
`a2c024725189473d2d7cea3a5cfdbcc67478e41f`. Select the `base` task IDs from
`split_tasks.json` for exactly three domains:

| Domain | Base rows | Bound source files and SHA-256 |
| --- | ---: | --- |
| airline | 50 | `tasks.json` `ccd8ba737b4cc371415af70151187788f728d6108d0916e73bb4317b40542052`; `split_tasks.json` `b22ced4d9a9850ac9aea31c53bdcb6d6009058140bd9acc7db37c1d36222ba8b`; `policy.md` `10dc0525421521208be39cee235bba84a16e2bcba9899eb93d92cd81d2f62fc4`; `src/tau2/domains/airline/tools.py` `3987f21c286314cb48764f97052934ff4fd60e27a0b2e4a43423911adf8eaa26` |
| retail | 114 | `tasks.json` `8e03ebce7901bd6218e7a7dc3105faa9324091a68058f7fe61c65262868812e8`; `split_tasks.json` `ed0580ec52575b63fbf76568af42490da6ee7783ecb4aa81af46961291358f20`; `policy.md` `2c9652afbce57d6e087768d37cda64d31c53d50b3e3225cfdb791bac66466467`; `src/tau2/domains/retail/tools.py` `34f4fe9702d36f0dd4697700fbcd63d0cf5386e6eb8e7499aaa336243bad8750` |
| telecom | 114 | `tasks.json` `37e562e1ae3242577407e1303b1548bc64e7ea68e37d36173e6747990ceaf8a4`; `split_tasks.json` `605b488bb9a6acb3c7f4505240a855fdc8681d09aadb16a8f38b2efcfc5c3aec`; `main_policy.md` `95943844a0cf11fdf0e2842b483b81d9f9338aa53235712ed131db075d086ed7`; `tech_support_manual.md` `015a3ee49ec8c199c5e1a059c922081937b92cf3a5ea74fedbc4357816f389ba`; `tech_support_workflow.md` `64f3ccd13875d14f358b21b9af221dab09e0e7bf5690d0e4322fb91855a43a4d`; `src/tau2/domains/telecom/tools.py` `388552434d1b8225e33e7958bdb6efccfe92b49e11d1838568d9b559514e9d01` |

The Tau2 renderer selects tasks by exact base ID, retaining original task
order only after membership validation. It extracts every string from the user
scenario, evaluation criteria, expected actions, policy, tool declaration,
tool documentation, and tool arguments. The prompt renderer is the benchmark's
canonical policy plus user scenario and canonicalized tool definitions. Its
implementation file and dependency lock hashes are mandatory acquisition
fields.

Terminal-Bench is the public registry component
`terminal-bench-core==0.1.1`. The registry repository revision is
`d28711d0da2675d0bb1d56de45ae5df6082438a3`; `registry.json` has SHA-256
`c485cf43b9a023cebade374bf13421d7d71a82b4426f1e6d8b81f56a3f44b96a`.
The registry row pins dataset revision
`91e10457b5410f16c44364da1a34cb6de8c488a5`, path `tasks`, and exactly 80
unique task IDs. Canonical JSON for the sorted task-ID list has SHA-256
`505611949dc4424bb1094ea202e920fa063434e68304ed8a5cb006caf8370f9e`;
the pinned `tasks` Git tree is
`0cbb16e9f84c60d32b9d7edf72f952a72145f314`.

The selected Terminal-Bench tree has not yet been fully acquired and hashed by
the producer and an independent reviewer. It remains outcome `P` until an
acquisition receipt enumerates every regular file beneath each of the 80
selected task directories with relative path, byte size, SHA-256, task ID, and
semantic role. The renderer extracts task instructions, solution material,
tests, fixtures, and executable script text without executing any task file.
The absence of that selected-file manifest fails only `tool` and cannot emit an
empty Terminal-Bench or `tool` receipt.

## Manifest architecture

### `q30t-held-out-sources-v2`

One canonical UTF-8 JSON manifest with a trailing newline owns the complete
source graph. It is parsed with duplicate-key rejection, finite JSON numbers
only, and exact unknown-field rejection. Required top-level fields are:

- `schema`: exact string `q30t-held-out-sources-v2`;
- `scientific_identity`: exact training identity from this design;
- `catalog_order`: exact list `speed,math,code,swe,tool`;
- `producer`: repository, 40-lowercase-hex commit, clean tree digest, producer
  path, producer file SHA-256, Python identity, and lockfile SHA-256;
- `identity`: prompt, text normalization, JSON normalization, containment, and
  binary-index algorithm names and implementation file hashes;
- `renderers`: typed renderer records keyed by stable renderer ID;
- `components`: typed component records ordered by catalog order and then
  lowercase ASCII `component_id` byte order;
- `catalogs`: exactly five catalog aggregation records;
- `storage_projection`: exact authenticated path-scoped storage schema;
- `scan_limits`: exact candidate traversal and view-work limits; and
- `proposal_sha256`: SHA-256 of the proposal hash view.

The proposal hash view is the complete top-level object with only
`proposal_sha256` omitted, serialized by the canonical JSON rules in this
design. The proposal contains no state, approval, signature, acquisition,
replay, receipt, or mutable status field. The same proposal bytes and
`proposal_sha256` are used before and after approval. A producer never rewrites
the proposal to express `P` or `A`.

Nested proposal schemas are also exact. `producer` is
`ImplementationIdentity`. `identity` contains exactly `prompt_algorithm`,
`prompt_implementation`, `text_normalization`, `json_normalization`,
`containment_algorithm`, `index_algorithm`, and `attribution_algorithm`.
Each renderer contains exactly `renderer_id`, `implementation`, `environment`,
`field_map_sha256`, `input_schema_sha256`, and `output_schema_sha256`.

Each component contains exactly `component_id`, `catalog`, `owner`,
`source_repository`, `source_revision`, `license_evidence`, `configuration`,
`split`, `selector`, `source_files`, `expected_source_rows`,
`expected_selected_rows`, `expected_selected_key_sha256`, `field_map`,
`prompt_renderer_id`, `auxiliary_renderer_id`, `identity_rules`, and
`error_disposition`. A source file contains exactly `path`, `bytes`, `sha256`,
`media_type`, `git_oid`, and `lfs_oid`; unavailable Git or LFS identities are
JSON null, never omitted. A field-map entry contains exactly
`source_json_pointer`, `domain`, `shape_id`, `field_key`, `required`, and
`cardinality`, ordered by source pointer, domain ID, shape ID, and UTF-8 field
key. `field_key` is the exact decoded object key used by key-conditioned matching
or the empty string for all other shapes. A source pointer has one separate
field-map entry for every shape it is allowed to emit, including shape 0; the
producer emits only the entries whose frozen shape predicate the normalized
value satisfies. Each catalog contains exactly `name`,
`component_ids`, and `union_algorithm`. `storage_projection` contains exactly
`schema`, `paths`, and `projection_verifier`. `scan_limits` contains exactly
`max_nesting_depth`, `max_structured_nodes`, `max_string_leaves`,
`max_scalar_utf8_bytes`, `max_decoded_row_bytes`, `max_generated_view_count`,
`max_generated_view_bytes`, `max_unique_scanned_view_bytes`, and
`max_matcher_work_bytes`.

`scan_limits_sha256` is SHA-256 of canonical JSON for the exact `scan_limits`
object. `match_semantics_sha256` is SHA-256 of canonical JSON containing exactly
the domain, rule, and shape tables, the content-digest preimage schema, the
containment algorithm identity, and all proposal field-map entries in proposal
order. Both hashes use no file-boundary newline.

Catalog component IDs retain proposal order and the union algorithm is the
sorted unique digest union with component-scoped attribution. The proposal does
not bind artifacts derived later. Components are never inferred from paths or
glob order. Derived receipts bind the proposal, not the reverse. Provenance
preserves every contributing component and its original ordered selected-record
digest.

Canonical component routing is fixed:

- `speed`: resolved SPEED qualitative, Red Hat `qa`, `question`, `rag`,
  `summarization`, `translation`, and `writing`;
- `math`: Red Hat `math_reasoning`, GSM8K test, MATH-500 test, and AIME
  2022-2025;
- `code`: Red Hat `HumanEval`, official HumanEval, all four official sanitized
  MBPP splits, and LiveCodeBench `release_v6`;
- `swe`: SWE-bench full Test; and
- `tool`: Red Hat `tool_call`, Tau2 airline/retail/telecom base, and
  Terminal-Bench core 0.1.1.

### Producer provenance

`examples/dataset/build_q30t_heldout_receipts.py` remains the single owner of
source-proposal validation and receipt production. Its current v1 file
SHA-256, observed at repository revision
`8e195c81a086bc8e26e10f84d5bf8ab0b73ea429`, is
`6666315c2a4a23f417246e88781d50012523d578c679c7e558471cd4c963295d`.
That hash is baseline evidence, not a v2 producer approval. A v2 candidate must
bind the post-review producer commit, file SHA-256, clean tree, Python binary,
dependency lock, command arguments, environment allowlist, source descriptors,
output descriptors, start time, finish time, and host/job identity.

The producer consumes the immutable proposal and writes one acquisition receipt
per component, one catalog receipt per fixed name, index and attribution
sidecars, atom stores, and one bundle receipt. It cannot write an approval root
or mutate the proposal. It returns nonzero and publishes no catalog receipt if
any component in that catalog fails acquisition, parsing, count, renderer, or
index validation. Other independent catalog candidates may still be produced
in their own scratch roots, but the builder cannot consume the bundle until a
separate valid approval root covers all five catalogs.

### Consumer receipt and index sidecars

`specdec-held-out-receipt-v2` is canonical JSON with a trailing newline. It
contains schema, scientific identity, fixed name, proposal SHA-256,
ordered component IDs, prompt identity algorithm, match-semantics SHA-256,
scan-limits SHA-256, prompt index descriptor, auxiliary index and atom-store
descriptors, raw-record digest, selected-record count, acquisition receipt
SHA-256 values, attribution descriptor, and self SHA-256. It contains no state,
approval, signature, or approval-root digest.

Large identity sets are descriptor-bound binary sidecars rather than unbounded
JSON arrays. A sidecar consists of one canonical JSON header line, followed
immediately by `count` lexicographically sorted, unique 32-byte SHA-256 values.
The header contains exactly `schema`, `scientific_identity`, `proposal_sha256`,
`match_semantics_sha256`, `catalog`, `domain_id`, `rule_id`, `shape_id`, `count`,
and `payload_sha256`; `schema` is `q30t-held-out-index-v2`. One index contains
one exact domain/rule/shape tuple. No second newline or padding occurs after the
binary payload. The receipt binds the sidecar byte size and whole-file SHA-256.
Readers verify header, exact length, strict ordering, uniqueness, payload hash,
and whole-file hash before lookup.

An atom store uses schema `q30t-held-out-atoms-v2`: one canonical JSON header
line followed by records sorted by
`(domain_id,rule_id,shape_id,field_key,digest,value)`. Each record is a one-byte
domain ID, one-byte rule ID, one-byte shape ID, unsigned two-byte big-endian
field-key length, exact normalized UTF-8 field-key bytes, 32-byte digest,
unsigned eight-byte big-endian value length, and exact normalized UTF-8 value.
The header contains exactly `schema`, `scientific_identity`, `proposal_sha256`,
`match_semantics_sha256`, `catalog`, `domain_table`, `rule_table`, `shape_table`,
`record_count`, `payload_bytes`, and `payload_sha256`; `schema` is
`q30t-held-out-atoms-v2`. Duplicate tuples, digest/preimage mismatch, length
mismatch, unknown shape, or noncanonical field key/value is fatal. The receipt
binds each atom store's whole-file SHA-256 and size.

Domain IDs are `0=problem`, `1=answer`, `2=patch`, `3=test_patch`, `4=code`,
`5=tool`, `6=policy`, and `7=raw_record`; `8=prompt` is allowed only in the
prompt index and attribution sidecar, never an atom store. Rule IDs are
`0=full`, `1=token32`, and `2=line8`. Shape IDs are `0=equality`,
`1=lexical_boundary`, `2=answer_context`, `3=substring`, `4=whole_line`,
`5=tool_canonical_json`, `6=tool_description`, `7=tool_key_value`,
`8=token_window`, `9=line_window`, and `10=raw_equality`. Any other ID is
fatal. Only shape 7 permits a nonempty `field_key`; every other shape requires
the empty string.

Each catalog also publishes `q30t-held-out-attribution-v2`. Its canonical JSON
header line contains exactly `schema`, `scientific_identity`, `proposal_sha256`,
`match_semantics_sha256`, `catalog`, `component_table`, `record_count`,
`payload_bytes`, and `payload_sha256`; `schema` is
`q30t-held-out-attribution-v2`, and the component table follows proposal order.
Payload records are sorted by
`(domain_id,rule_id,shape_id,digest)` and contain those three one-byte IDs, the
32-byte digest, an unsigned two-byte big-endian contributor count, and that many
unique contributor pairs. A pair is an unsigned two-byte component ordinal and
an unsigned two-byte field-map-entry ordinal, sorted by both ordinals. Component
and field-map ordinals index the immutable proposal; ordinal 65,535 is reserved
only for a renderer-derived prompt with no single field-map entry. The prompt
index uses domain ID 8, rule ID 0, and shape ID 0 only in this sidecar. Every
index digest must have exactly one attribution record; every attribution digest
must exist in the matching domain/rule/shape index. A collision ledger records
all unique contributing component IDs, never an arbitrary first match. More
than 65,535 contributors, 65,535 components, or 65,535 field-map entries in one
component is fatal.

The consumer reconstructs matching solely from the proposal, index header, atom
record, and attribution pairs. It dispatches each atom only to its authenticated
`shape_id` and, for shape 7, its exact `field_key`; it never infers a shape from
value bytes, reloads benchmark source files, or tries every shape. Each
attribution pair must resolve to a proposal field-map entry with the same domain,
shape ID, and field key. Its field-map-entry ordinal is the canonical provenance
for that entry's exact authenticated source JSON pointer; a copied pointer string
is never trusted as a substitute. This check occurs before any candidate is
scanned.

The five receipts are consumed in fixed catalog order alongside, never inside,
one separately supplied approval root. Their prompt indexes are merged as
sorted sets. Auxiliary indexes remain domain-separated. A v1 receipt, missing
sidecar, proposal mismatch, invalid approval root, or different scientific
identity is fatal.

### Frozen artifact schemas and loader graph

Every canonical JSON artifact below has exactly the listed fields, sorted-key
serialization, and one trailing newline. Unless a different hash view is stated,
`self_sha256` is SHA-256 of the full object with only `self_sha256` omitted.
Unknown fields are fatal. `ArtifactDescriptor` is exactly `path`, `bytes`,
`sha256`, and `media_type`. `ImplementationIdentity` is exactly `repository`,
`commit`, `tree_sha256`, `path`, `file_sha256`, `python_sha256`, and
`lockfile_sha256`. `ExecutionIdentity` is exactly `command_argv`,
`environment_allowlist`, `host`, `job_id`, `started_utc`, and `finished_utc`.
`command_argv` is an ordered string array and `environment_allowlist` is a
sorted mapping of explicitly permitted variable names to nonsecret values.
`NamedHash` is exactly `name` and `sha256`; `NamedCount` is exactly `name` and
`count`; `CategoryCount` is exactly `category` and `count`; `ByteComparison` is
exactly `path`, `expected_sha256`, `observed_sha256`, and `match`.
`ErrorRecord` is exactly `code`, `stage`, `source_path`, `expected`, and
`observed`; the last three fields are JSON null when inapplicable. Hash arrays
contain lowercase 64-hex strings. Named arrays sort by lowercase ASCII name,
except component, catalog, and category arrays that explicitly retain proposal
or training-contract order. Count values are nonnegative integers, never JSON
booleans.

The schemas and field order semantics are:

- `q30t-held-out-acquisition-v2`: `schema`, `scientific_identity`,
  `proposal_sha256`, `match_semantics_sha256`, `component_id`, `producer`,
  `execution`, `source_files`, `selected_record_count`,
  `ordered_selected_key_sha256`, `rendered_prompt_count`, `ordered_prompt_sha256`,
  `atom_counts`, `index_descriptors`,
  `atom_store_descriptors`, `attribution_descriptor`, `errors`, and
  `self_sha256`. `producer` is `ImplementationIdentity`; `execution` is
  `ExecutionIdentity`; source and output descriptors are sorted by relative
  path; atom counts use domain-ID, rule-ID, then shape-ID order; `errors` must be
  empty for publication.
- `q30t-held-out-semantic-review-v2`: `schema`, `scientific_identity`,
  `proposal_sha256`, `component_id`, `source_revision`, `source_file_sha256s`,
  `split_selector`, `field_map_sha256`, `prompt_renderer_sha256`,
  `auxiliary_renderer_sha256`, `expected_selected_count`, `license_evidence`,
  `reviewer_key_id`, `decision`, `reviewed_utc`, and `self_sha256`.
  `decision` is exactly `approve` for use; file hashes retain proposal order.
- `q30t-held-out-replay-v2`: `schema`, `scientific_identity`,
  `proposal_sha256`, `match_semantics_sha256`, `scope_kind`, `scope_id`,
  `acquisition_sha256s`, `verifier`, `execution`,
  `reacquired_source_descriptors`, `selected_record_count`,
  `ordered_selected_key_sha256`, `ordered_prompt_sha256`, `index_descriptors`,
  `atom_store_descriptors`, `attribution_descriptor`, `byte_comparison`,
  `decision`, and `self_sha256`. `scope_kind` is `component` or `catalog`;
  `decision` is `match`; `byte_comparison` is a path-sorted array of
  `ByteComparison` whose `match` values are all true.
- `specdec-held-out-receipt-v2`: `schema`, `scientific_identity`, `name`,
  `proposal_sha256`, `component_ids`, `prompt_identity`,
  `match_semantics_sha256`, `scan_limits_sha256`, `acquisition_sha256s`,
  `prompt_index_descriptor`,
  `auxiliary_index_descriptors`, `atom_store_descriptors`,
  `attribution_descriptor`, `selected_record_count`, `raw_record_sha256`, and
  `self_sha256`. `name` follows fixed catalog order; component IDs retain
  proposal order; auxiliary descriptors use domain-ID, rule-ID, then shape-ID
  order.
- `q30t-held-out-bundle-v2`: `schema`, `scientific_identity`,
  `proposal_sha256`, `catalog_receipt_descriptors`, `catalog_order`,
  `union_prompt_count`, `union_prompt_sha256`, `union_attribution_sha256`, and
  `self_sha256`. Catalog descriptors and `catalog_order` are exactly
  `speed,math,code,swe,tool`.
- `q30t-held-out-approval-v2`: `schema`, `statement`, `statement_sha256`,
  `signatures`, and `self_sha256`. `statement` contains exactly
  `scientific_identity`, `proposal_sha256`, `bundle_receipt_sha256`,
  `catalog_receipt_sha256s`, `semantic_review_sha256s`, `replay_sha256s`,
  `catalog_scope_sha256s`, `trusted_keyring_sha256`,
  `consumer_implementation`, and `decision`.
  `statement_sha256` hashes exactly `statement`; `decision` is `approve`.
  `self_sha256` omits only itself and therefore covers the statement hash and
  signatures without being signed itself. `consumer_implementation` binds code
  that accepts an external approval-root descriptor; that code and its hashed
  environment contain no embedded approval-root hash.
- `q30t-capacity-v2`: `schema`, `scientific_identity`, `builder`, `execution`,
  `tokenizer_receipt_sha256`, `historical_replay_sha256`,
  `held_out_proposal_sha256`, `held_out_bundle_sha256`,
  `held_out_approval_sha256`, `source_descriptors`, `category_order`, `quotas`,
  `scan_limits_sha256`, `counts_by_category`, `observed_scan_maxima`,
  `ordered_eligible_sha256`, `decision`, and
  `self_sha256`. Category arrays use the immutable training-contract order;
  each `counts_by_category` entry contains exactly `category`, `raw`,
  `historical_rejected`, `held_out_prompt_rejected`, `auxiliary_rejected`,
  `duplicate_rejected`, `schema_rejected`, `tokenizer_rejected`, `eligible`, and
  `selected`. `decision` is `sufficient` only when every eligible count meets
  its quota. `observed_scan_maxima` contains exactly `nesting_depth`,
  `structured_nodes`, `string_leaves`, `scalar_utf8_bytes`,
  `decoded_row_bytes`, `generated_view_count`, `generated_view_bytes`,
  `unique_scanned_view_bytes`, and `matcher_work_bytes`; each value is at most
  its proposal limit.
- `q30t-completion-v2`: `schema`, `scientific_identity`, `builder`, `execution`,
  `capacity_sha256`, `historical_replay_sha256`, `held_out_proposal_sha256`,
  `held_out_bundle_sha256`, `held_out_approval_sha256`, `tokenizer_sha256`,
  `data_descriptor`, `manifest_descriptor`, `ledger_descriptor`,
  `scan_limits_sha256`, `observed_scan_maxima`, `category_counts`,
  `unique_prompt_count`, `unique_row_count`,
  `unique_occurrence_count`, `ordered_occurrence_sha256`,
  `ordered_prompt_sha256`, `ordered_row_sha256`, `exclusion_counts`,
  `padding_count`, `cycling_count`, `redistribution_count`, `decision`, and
  `self_sha256`. `category_counts` is an ordered `CategoryCount` array and
  `exclusion_counts` is a name-sorted `NamedCount` array. `decision` is
  `complete`; the three prohibited counts are zero. `observed_scan_maxima` has
  the exact capacity-receipt field set, is independently recomputed for the
  completion run, and contains no value above the proposal limit.

Across these schemas, acquisition hashes retain proposal component order.
Approval catalog-receipt and catalog-scope hashes use fixed catalog order;
semantic-review hashes use proposal component order. Replay hashes use fixed
catalog order, then component scopes in proposal order, then the catalog scope;
the required pair within each scope is lexicographically sorted by receipt
SHA-256. Capacity and completion category arrays use the exact eight-category
training-contract order shown at the start of this design. Any permutation is
fatal even when set contents and aggregate digests match.

`q30t-exclusion-ledger-v2` is canonical JSONL and has no embedded self hash. Its
first line has exactly `schema`, `scientific_identity`, `builder_sha256`,
`capacity_sha256`, and `event_count`. Each following event line has exactly
`sequence`, `source_component`, `source_file_sha256`, `physical_row`,
`category`, `disposition`, `candidate_prompt_sha256`, `candidate_json_pointer`,
`protected_catalog`, `protected_domain`, `protected_rule`, `protected_shape`,
`protected_field_key`, `protected_digest`, and `contributing_component_ids`.
Events are ordered by monotonically increasing `sequence`, which follows
authenticated candidate source order. Events without a held-out match use JSON
null for `candidate_json_pointer` and all six protected scalar fields, plus an
empty contributor list. Prompt matches use `protected_domain="prompt"`,
`protected_rule="full"`, `protected_shape="equality"`, an empty protected field
key, a null candidate pointer, and prompt attribution. Auxiliary matches use the
authenticated source domain, rule, shape, field key, and matched leaf pointer.
Contributor IDs retain proposal order. The completion receipt binds the
ledger's byte size and whole-file hash.

Loader relationships are one-way. The acquisition loader accepts only the
proposal. The semantic reviewer accepts the proposal plus immutable source
bytes. The independent verifier accepts the proposal, source bytes, acquisition
artifacts, and semantic review, but not producer caches. The approval loader
accepts the unchanged proposal, five catalog receipts, bundle receipt, semantic
reviews, replay receipts, keyring, and signatures. The held-out consumer accepts
only that same proposal plus the five receipts, bundle, and separate approval
root. The capacity loader additionally accepts historical replay, tokenizer,
and candidate-source receipts. The builder accepts the exact capacity inputs
and writes data, manifest, ledger, and completion. The completion verifier
reopens all of them by descriptor and independently recomputes every bound hash.
The held-out consumer recomputes `match_semantics_sha256` and
`scan_limits_sha256` before loading an index. Capacity and completion loaders
allocate streaming counters from the proposal values, reject any override, and
bind observed maxima in their receipts. A counter overflow or cap exceedance is
a typed fatal error, never a candidate rejection or truncation.

## One prompt identity everywhere

The current builder computes candidate identity from the complete `messages`
and `tools` object. That includes assistant and tool-response content. The
shared `examples/dataset/specdec_identity.py` implementation instead retains
only `system`, `developer`, and `user` messages, retains tool declarations,
removes keys named by `STORAGE_ONLY_FIELDS` recursively, canonicalizes the
prompt, and hashes it. Full-message hashing and recursive key-name removal are
both release blockers, not compatibility options.

V2 replaces recursive removal with proposal-bound `STORAGE_ONLY_PATHS_V2`.
Paths use RFC 6901 escaping; `*` matches exactly one list index and never an
object key or multiple levels. The complete optional path set is:

- top level: `/row_index`, `/shard_index`, `/download_path`, `/cache_path`,
  `/ingested_at`, `/source_completion`, and `/generated_assistant_message`;
- message storage metadata: `/messages/*/row_index`,
  `/messages/*/shard_index`, `/messages/*/download_path`,
  `/messages/*/cache_path`, `/messages/*/ingested_at`, and the same five leaf
  names directly under `/messages/*/metadata`;
- message call storage metadata: `/messages/*/tool_calls/*/download_path` and
  `/messages/*/tool_calls/*/cache_path`; and
- tool storage metadata: the five message-storage leaf names directly under
  `/tools/*` or `/tools/*/metadata`.

No `source_completion` or `generated_assistant_message` path below the top
level is storage-only. A matching optional path may be removed only when its
proposal field-map declares it absent from the rendered prompt, published
training row, tool structure, tokenizer input, and loss mask. The capacity and
completion verifier prove those projection claims. A same-named key at any
other path is semantic, retained, and scanned. Adding a path requires a new
proposal hash and full review.

The migration is mandatory RED/GREEN:

1. RED tests show that the current builder identity changes when only an
   assistant completion changes and disagrees with
   `specdec_identity.prompt_uuid`.
2. GREEN changes the builder to import and call
   `specdec_identity.prompt_uuid`; local prompt hashing is removed, and that
   function uses only the frozen path projection above.
3. Historical audit, held-out rendering, global admission, candidate
   deduplication, exclusion ledger, and replay all call the same function.
4. Tests require identical prompts with different assistant completions to
   have one identity and require tool-schema changes to change identity.

No hand-authored identity values are permitted. The existing historical audit
is not grandfathered merely because it names the shared module. Before use, an
independent verifier reacquires every authenticated historical source byte,
re-renders prompts, and emits `q30t-historical-identity-replay-v2` with exactly
`schema`, `scientific_identity`, `historical_audit_sha256`,
`historical_source_descriptors`, `prompt_identity_implementation`,
`prompt_identity_environment`, `storage_projection_sha256`, `occurrence_count`,
`ordered_occurrence_sha256`, `ordered_prompt_sha256`, `comparison`, `decision`,
and `self_sha256`. Its ordinary self-hash view omits only `self_sha256`.
`prompt_identity_implementation` is `ImplementationIdentity`.
`prompt_identity_environment` contains exactly `python_sha256`,
`lockfile_sha256`, `unicode_version`, `locale`, and `timezone`; the last three
must be `15.1.0`, `C.UTF-8`, and `UTC`. `comparison` contains exactly
`audit_sha256_match`, `occurrence_count_match`,
`ordered_occurrence_sha256_match`, and `ordered_prompt_sha256_match`, all true.
It must reproduce the authenticated audit file SHA-256 and ordered occurrence
digest; `decision` must be `match`. The verifier implementation and environment
hashes are independently versioned and bound in the receipt.

An existing v1 receipt cannot satisfy a Q30 v2 catalog. A receipt made with the
old full-message or recursive-key-removal identity is recorded in an explicit
`legacy_full_message_v1` diagnostic lane, never translated by a lookup table,
and regenerated from authenticated source bytes. Until historical replay passes,
the historical audit and the 700K build remain blocked.

## Auxiliary leakage identity

Prompt equality alone cannot catch an answer or patch embedded in a longer
training trajectory. V2 therefore derives domain-separated identities from
both benchmark sources and every candidate row.

### Strict decoding and normalization

Every JSON input is UTF-8 without BOM, parsed with duplicate-key rejection and
finite numbers only. JSON strings are decoded before text normalization. Every
string value and object key is normalized with Unicode 15.1.0 NFC. The runtime
must report `unicodedata.unidata_version == "15.1.0"`; any other version is
fatal. A duplicate object key created by NFC normalization is fatal. Canonical
structured JSON uses sorted keys, UTF-8 characters without ASCII escaping,
separators `,` and `:`, and a trailing newline only at file boundaries.

Text normalization is exactly:

1. reject non-strings, NUL, invalid UTF-8, and unpaired surrogates;
2. replace CRLF with LF, then remaining CR with LF;
3. normalize Unicode to NFC; and
4. preserve every other code point, including case and punctuation.

The containment view additionally replaces every maximal run of Unicode 15.1
White_Space code points with one ASCII space and removes a leading or trailing
ASCII space. The pinned set is U+0009-U+000D, U+0020, U+0085, U+00A0, U+1680,
U+2000-U+200A, U+2028, U+2029, U+202F, U+205F, and U+3000. No other character is
deleted or rewritten.

Every digest is
`SHA256(canonical_json(["q30t-held-out-content-v2",domain_id,rule_id,shape_id,field_key,value]))`,
where canonical JSON follows the rules above without a file-boundary newline.
IDs and `field_key` follow the frozen tables above. The producer emits a separate
digest for every applicable shape of a source atom; equality and embedded-match
digests are therefore distinct even when `value` is identical. This preimage
prevents different source domains, shapes, or authenticated tool keys from
collapsing during union.

### Source field extraction

Typed field maps must enumerate source paths; recursive extraction is not used
as a substitute for a reviewed map. Required mappings include:

- prompts, questions, problem statements, task instructions, and user scenarios
  to `problem`;
- references, answers, solutions, canonical solutions, and expected action
  text to `answer`;
- SWE `patch` to `patch` and `test_patch` only to `test_patch`;
- starter code, reference code, tests, test lists, fixtures, and executable
  task scripts to `code`, except SWE test patches remain `test_patch`;
- tool declarations, schemas, names, descriptions, arguments, call examples,
  and embedded available-functions JSON to `tool`;
- benchmark policies and manuals to `policy`; and
- the complete canonical parsed record to `raw_record`.

Empty strings produce no content atom but remain visible in the raw-record
identity. Missing required fields, unexpected types, ambiguous embedded JSON,
or a field-map path that matches a different count than declared is fatal.

### Candidate extraction

For every candidate, the scanner recursively visits every string leaf in the
entire decoded row, regardless of top-level field name, nesting depth, message
role, or recognized schema. The only omitted leaves are values at exact
proposal-authenticated `STORAGE_ONLY_PATHS_V2` paths whose no-training
projection has been verified. System, developer, user, assistant, tool,
metadata, unknown extensions, and nested tool-call structures are all scanned;
role filtering applies only to prompt identity.

The scanner records each leaf's exact RFC 6901 path and also scans canonical
JSON for every containing mapping or list, LF-joined content for each message,
each tool declaration, the whole conversation, and the complete projected row.
This catches benchmark content inside a longer message, tool result, assistant
completion, unknown field, or structured declaration. A loader may reject an
unsupported value type, but it may never ignore a supported string-bearing
subtree.

Traversal is iterative depth-first and streaming. It increments depth, mapping
or list node count, string-leaf count, and decoded UTF-8 bytes with checked
integer addition before visiting or allocating a child. Canonical structured
views are written once per container by a streaming serializer to
producer-owned spill; a parent is never formed by concatenating materialized
child views. Joined views use the same streaming writer. Before a generated
view byte is written, the cumulative generated-view counter is charged, so
deduplication cannot hide construction work.

Generated views are deduplicated before matching by exact
`(view_kind,byte_length,SHA256(bytes))`; a digest hit is byte-rechecked from the
descriptor-bound spill. Only the first byte-identical view is scanned. The
unique-scanned counter charges its UTF-8 length once. Matcher-work bytes charge
that length once for each shape-specific matcher pass actually selected by the
authenticated atom records. A scalar above 16 MiB, a decoded row above 64 MiB,
depth above 64, more than 65,536 structured nodes, string leaves, or generated
views, more than 256 MiB generated-view bytes, more than 256 MiB unique
scanned-view bytes, or more than 1 GiB matcher-work bytes is fatal to the build.
These proposal-bound caps cannot be raised by command-line or environment input.

Every candidate text view is queried separately against each source domain
`problem`, `answer`, `patch`, `test_patch`, `code`, `tool`, and `policy`, using
that source domain in the digest prefix. Candidate field names do not restrict
which source domain may match: an answer pasted into a user message or a patch
pasted into a tool result must still reject. `raw_record` is the only exception;
it is an additional equality proof against the complete canonical projected
candidate row and never replaces leaf or containment scanning. Exclusion logs
contain the candidate source coordinate, candidate JSON pointer, catalog,
domain, rule, shape, authenticated field key, digest, and all contributing
component IDs, but never copy protected benchmark text.

### Exact containment rules

Three deterministic rules are evaluated:

1. **Full and shape containment.** Every nonempty non-raw atom gets a rule-0,
   shape-0 equality digest, and complete candidate-leaf equality always rejects.
   A raw-record atom gets rule 0, shape 10 only. Each permitted embedded match
   produces an additional digest with the exact shape below. No consumer applies
   a shape absent from that atom's authenticated records. Embedded containment
   is exact and domain-specific; no atom is discarded solely for being shorter
   than a universal byte or token threshold. A Unicode lexical boundary is
   start/end or a neighboring code point whose General_Category does not begin
   with `L` or `N` and is not underscore.

   - `problem` and nonnumeric `answer` atoms with at least two `L`/`N` code
     points emit shape 1 and reject on a boundary-delimited exact occurrence.
     Shorter answers emit shape 2 and reject only in the answer-shaped contexts
     below; shorter problems retain shape 0 equality only.
   - `patch`, `test_patch`, and `code` atoms with at least four non-White_Space
     code points emit shape 3 and reject on an exact occurrence. Shorter atoms
     emit shape 4 and reject on exact whole-nonempty-line equality in addition
     to shape 0, so compact lines such as `x=1` are still protected without
     treating a lone brace as universal text.
   - `policy` atoms with at least two `L`/`N` code points reject on a
     boundary-delimited exact occurrence under shape 1. Shorter policy atoms
     emit shape 4 for whole-nonempty-line equality in addition to shape 0.
   - canonical structured `tool` atoms emit shape 5 and reject on exact
     canonical-JSON containment. A free-text tool description of at least two
     tokens emits shape 6 and rejects on exact containment. A shorter tool name,
     argument, enum, or description emits shape 7 and rejects on exact canonical
     key/value containment, such as `"name":"get"`, in addition to shape 0;
     its authenticated field-map entry supplies the exact nonempty `field_key`.

   An answer is numeric-shaped only when its containment view matches the full
   ASCII regular expression
   `[+-]?(?:[0-9]+(?:\.[0-9]+)?|[0-9]+/[0-9]+|[0-9]+(?:,[0-9]{3})+)(?:%|[A-Za-z]+)?`.
   A numeric-shaped or shorter-than-two-alphanumeric answer emits shape 2 and,
   when embedded in a larger leaf, rejects only at lexical boundaries in these
   exact containment templates: `#### VALUE`, `answer VALUE`, `answer: VALUE`,
   `answer is VALUE`, the same three `Answer` forms, `final answer VALUE`,
   `final answer: VALUE`, the same two `Final` forms, or `\\boxed{VALUE}`.
   `VALUE` is the exact normalized source atom; there is no numeric parsing or
   value equivalence. Thus a bare `42` elsewhere in prose or code does not
   reject, while `Answer: 42` does. This numeric-answer exception is the only
   false-positive relaxation for an otherwise exact atom.

   Byte-oriented Aho-Corasick automata are built in sorted atom order for each
   shape class. Candidate hits are rechecked for exact bytes, boundaries,
   source field key where required, and authenticated atom attribution.
2. **Token windows.** For atoms of at least 48 tokens, rule 1 and shape 8 split
   the containment view on ASCII spaces. Reference 32-token windows begin at token offsets 0,
   16, 32, and so on. Candidate windows begin at every token offset. Reject only
   when two adjacent reference windows, starting 16 tokens apart, match windows
   starting 16 tokens apart in the same candidate view. The proof is an exact
   48-token span; one isolated window is insufficient.
3. **Line windows.** For `patch`, `test_patch`, and `code`, rule 2 and shape 9
   split normalized text on LF, apply the containment whitespace transform to
   each line, and remove empty lines. Reference eight-line windows start every four lines; candidate
   windows start at every line. Reject only when two adjacent reference windows
   match at the corresponding four-line displacement, proving an exact
   12-nonempty-line span.

Multiple fields from one source record are not concatenated to manufacture a
shape match, except for the explicitly defined canonical structured-tool and
raw-record atoms. Digest hits are always verified against source atom bytes and
the digest-to-all-contributors sidecar, eliminating hash-only false positives
and lost attribution.

There are no warnings that allow admission. A confirmed hit rejects. A digest
hit without retrievable source bytes, resource-cap exceedance, malformed atom,
index inconsistency, or scanner error fails the build closed. Rejected rows do
not change the category quota.

## Deterministic selection and proof of 700K

Selection follows the source and category order already fixed by the 700K data
plan. Within a category, rows are visited by authenticated source-file order
and physical row number. For each occurrence:

1. derive the shared prompt identity;
2. reject a historical prompt collision;
3. reject a held-out prompt collision;
4. derive and scan auxiliary content identities;
5. reject a global prompt or normalized full-row duplicate;
6. perform existing schema, tool-trajectory, tokenizer, length, and
   assistant-loss-token validation; and
7. admit the row only if the category still needs capacity.

SQLite tables with primary keys on prompt digest and normalized full-row digest
make global uniqueness an enforced invariant, not a final count assertion.
Source occurrence coordinates have a separate unique constraint, proving that
no input occurrence was cycled. Each rejection is appended to an ordered
ledger. Once a category reaches its exact quota, later rows in that category
are not eligible and cannot refill another category.

Before data publication, a capacity pass runs the complete exclusion and
eligibility pipeline without writing selected data. Its receipt records
raw, historical-rejected, held-out-prompt-rejected, auxiliary-rejected,
duplicate-rejected, schema-rejected, tokenizer-rejected, eligible, and selected
counts per category. Every eligible count must be at least its fixed quota.

The completion receipt must prove:

- each exact category count from the contract table;
- PTV2 subtotal 300,000 and PTV3 subtotal 400,000;
- total occurrences 700,000;
- 700,000 distinct prompt identities;
- 700,000 distinct normalized full-row identities;
- 700,000 distinct source occurrence coordinates;
- zero historical, held-out prompt, or auxiliary content matches among admitted
  rows;
- no padding, cycling, quota redistribution, or fallback source; and
- identical ordered occurrence, prompt, row-content, data-file, manifest,
  exclusion-ledger, capacity, and completion digests on independent replay.

## Typed loaders, limits, and filesystem safety

All proposals, receipts, source descriptors, field maps, renderer records,
index headers, acquisition receipts, replay receipts, approval roots,
capacity receipts, and completion receipts use frozen typed schemas. Unknown,
missing, duplicate, incorrectly cased, or incorrectly typed fields are fatal.
Booleans are not accepted as integers. Counts are nonnegative decimal integers;
hashes are lowercase 64-hex; Git commits are lowercase 40-hex.

Exact v2 limits are:

- source proposal or JSON receipt: 16 MiB each;
- approval root or semantic/replay review receipt: 64 MiB;
- one source file: 2 GiB;
- aggregate source bytes per catalog: 8 GiB;
- files per component: 4,096;
- records per component: 5,000,000;
- decoded record: 64 MiB;
- decoded scalar string: 16 MiB;
- candidate nesting depth: 64, with the root at depth zero;
- candidate structured mapping/list nodes per row: 65,536;
- candidate string leaves per row: 65,536;
- generated candidate views per row: 65,536;
- cumulative generated candidate-view bytes per row: 256 MiB;
- cumulative unique scanned-view bytes per row: 256 MiB;
- cumulative matcher-work bytes per row: 1 GiB;
- binary index sidecar: 1 GiB;
- atom store per catalog: 8 GiB;
- exclusion ledger: 4 GiB;
- producer or consumer resident working set: 8 GiB;
- content atoms per catalog, rule, and shape: 20,000,000.

The exact proposal `scan_limits` integer values are respectively 64, 65,536,
65,536, 16,777,216, 67,108,864, 65,536, 268,435,456, 268,435,456, and
1,073,741,824 in the field order frozen above. Byte counters count UTF-8 bytes,
not Unicode scalar values or allocator capacity.

Declared and observed values are checked before allocation and while streaming.
Compression expansion is counted against decoded record and aggregate limits.
Index construction uses bounded external sort and merge in producer-owned
scratch; it never loads an entire source, index, or atom store into memory.
Containment automata are partitioned deterministically by catalog, domain, and
power-of-two atom-byte-length bucket; each partition is checked against the
resident limit and every candidate view is streamed through every applicable
partition in that fixed order.
Archives are never extracted by the receipt producer. Terminal-Bench inputs are
read as individually enumerated files; path traversal, absolute paths, device
nodes, sockets, FIFOs, and links are rejected.

Every local input is opened with no-follow semantics, required to be a regular
single-link file, and bound to its descriptor. The reader hashes and decodes
from that descriptor, then verifies device, inode, size, and timestamps against
the pre-open snapshot; any rebind or mutation fails. Remote acquisition writes
to a fresh scratch file, verifies response identity and expected SHA-256, fsyncs
it, closes it, and reopens it through the same stable reader. Redirects are
permitted only when the final immutable object URL and certificate-validated
HTTPS origin are recorded; credentials and ambient cache files are not sources.

Publication uses a fresh same-filesystem scratch directory, mode 0700, with
files created by `O_CREAT|O_EXCL|O_NOFOLLOW`. Each file is fsynced, the scratch
directory is fsynced, and publication uses no-replace semantics. If the target
exists, exact adoption is allowed only after complete descriptor verification
of every expected artifact; otherwise publication fails. After either a
successful no-replace publish or exact adoption, the destination parent
directory descriptor is fsynced before success is reported. For a directory
bundle, both the published directory and its destination parent are fsynced.
Producers never unlink or overwrite a foreign target and never recursively
delete a caller-supplied directory. On failure, the producer reports its own
scratch root; deletion is a separate owner or scheduler action constrained to
that exact recorded root.

## Independent replay and approval

### State transition

`P` and `A` are validation outcomes, not fields in the immutable proposal or
catalog receipts. `P` means the unchanged proposal and derived artifacts exist
without a valid separate approval root and are not consumable. `A` means the
same proposal bytes, the exact proposal-bound receipts, and a separately
supplied `q30t-held-out-approval-v2` root pass all acquisition, semantic,
independent replay, signature, and consumer checks. Promotion creates only the
approval root; it never rewrites or rehashes the proposal or receipts. There is
no implicit transition and no state derived from process exit.

For each component and catalog:

1. the producer acquires source bytes and emits proposal-bound acquisitions,
   indexes, atom stores, attributions, catalog receipts, and bundle receipt;
2. the named evaluation-domain owner reviews the actual immutable source bytes,
   split/release selector, field map, prompt renderer, auxiliary renderer, and
   expected count, emits `q30t-held-out-semantic-review-v2`, and signs its
   receipt hash;
3. reviewer one independently reacquires into a new root and replays with an
   independently versioned verifier and no producer cache;
4. reviewer two repeats with a distinct key and fresh root;
5. each verifier compares source files, selected keys, rendered prompts, atoms,
   indexes, attribution, receipts, and counts byte-for-byte and signs its replay
   receipt hash;
6. each catalog owner signs its catalog-scope hash; and
7. the bundle owner signs the approval statement, after which a reviewed change
   may pin the separate approval-root whole-file SHA-256 in controller input
   configuration outside the consumer implementation hash view.

The verifier identity is a distinct repository revision, path, file SHA-256,
Python SHA-256, and lockfile SHA-256 from the producer. It may share only the
frozen canonical JSON, cryptographic digest, and descriptor-reader primitives;
it independently implements source selection, record-key extraction, field
mapping, prompt rendering, auxiliary extraction, normalization, and index
construction. Importing the producer renderer or using a producer cache fails
the replay. Both verifier identities are recorded even if they use the same
reviewed verifier version.

Each approval `signatures` record contains exactly `role`, `key_id`,
`scope_kind`, `scope_id`, `signed_sha256`, and base64 `signature`. Records are
ordered by role rank `semantic_owner`, `replay_reviewer`, `catalog_owner`, then
`bundle_owner`; within a rank they follow catalog/component proposal order and
then key ID byte order. The Ed25519 message is SHA-256 of canonical JSON
`["q30t-held-out-signature-v2", role, scope_kind, scope_id, signed_sha256]`.
Signatures use RFC 4648 standard base64 with required padding and decode to
exactly 64 bytes; alternate alphabets or noncanonical encodings are fatal.
Semantic owners sign the corresponding semantic-review self hash. Two distinct
replay reviewers sign the two replay self hashes for every component and
catalog. Catalog owners sign the matching `catalog_scope_sha256`; the bundle
owner signs `statement_sha256`. Role, scope, and signed hash must agree with the
keyring authorization and cannot be substituted.

The catalog-scope hash is SHA-256 of canonical JSON containing exactly the
scientific identity, proposal SHA-256, named catalog receipt SHA-256, that
catalog's ordered semantic-review SHA-256 values, its two ordered component and
catalog replay SHA-256 sets, and trusted keyring SHA-256. The approval statement
contains the five catalog-scope hashes in fixed catalog order. Keys must be
present in a separately reviewed keyring receipt whose whole-file SHA-256 is
supplied as a trusted controller input outside the consumer implementation hash.
No key IDs or signatures are invented in this design. Until the keyring and
separate approval root exist, every catalog validates as `P`.

Producer identities may not satisfy any semantic-owner, replay-reviewer,
catalog-owner, or bundle-owner role. A signature for one scope, proposal,
scientific identity, receipt, or keyring cannot be replayed for another.
Revoked or expired keys, duplicate replay reviewers, an untrusted owner, a
different keyring digest, or any replay mismatch fails closed.

### Failure isolation

Failure to acquire or pin a source prevents only its containing catalog from
validating as `A`. Successfully reviewed catalogs remain valid evidence, but
the five-catalog bundle remains nonconsumable. The producer never substitutes a
smaller public set, an internal filename, an empty identity set, the previous
v1 receipt, or a prose waiver. Retrying acquisition or replay retains the same
proposal and produces new derived receipt hashes; changing source semantics
requires a new proposal hash.

## Error reporting

Errors are stable typed codes with catalog, component, source descriptor, and
stage. Required codes cover source hash/count mismatch, split mismatch,
renderer drift, malformed JSON, duplicate normalized key, resource cap,
unstable descriptor, link or file-type violation, record-key mismatch, empty
component, content-index inconsistency, shape/field-map attribution mismatch,
nesting-depth cap, structured-node cap, generated-view cap, scanned-byte cap,
matcher-work cap, prompt-identity mismatch, untrusted signature, replay mismatch,
quota insufficiency, duplicate admission, and publication collision.

Reports include expected and observed nonsecret hashes, sizes, and counts but
do not print benchmark answers, patches, test patches, tool policies, training
messages, credentials, or signing material. A component error terminates that
catalog before publication. A candidate-row collision rejects the row and
records only its coordinate and match metadata. An infrastructure or index
error terminates the entire dataset build.

## Test and review gates

### Identity RED/GREEN

- RED: two rows differing only in assistant completion produce different
  identities under the current local builder helper.
- GREEN: both produce the same shared prompt identity.
- Changing a system, developer, user, or tool declaration changes identity.
- Storage-only metadata does not change identity; arbitrary content metadata
  does.
- A storage-named key at an unlisted path remains semantic and changes identity;
  every listed omission is proved absent from tokenizer and loss-mask inputs.
- Historical, held-out, candidate, deduplication, and replay callers have no
  local prompt-hash implementation.
- A legacy full-message receipt is rejected by every Q30 v2 consumer.
- Historical source replay with a different identity implementation or
  environment hash is rejected even if an unordered prompt set matches.

### Normalization and containment

- CRLF, CR, and LF variants normalize identically; NFC-equivalent text matches.
- Case and punctuation changes do not match unless the unchanged exact span
  independently satisfies a rule.
- A complete benchmark problem embedded before, inside, or after unrelated
  candidate text is rejected.
- A 48-token benchmark span with changed surrounding text is rejected by two
  adjacent token windows.
- A 12-nonempty-line patch or test span embedded in a larger assistant or tool
  message is rejected by line windows.
- Compact answer, patch, test, code, policy, and tool fixtures exercise every
  shape-specific containment branch.
- A bare short numeric answer elsewhere in prose does not reject; the same value
  in every exact answer-shaped template does reject.
- A protected atom hidden in an unknown nested candidate field rejects, while
  only a verified exact storage path is omitted.
- One isolated 32-token window and one isolated eight-line window do not reject.
- The same digest in a different domain does not reject.
- JSON duplicate keys, NFC-created duplicate keys, nonfinite numbers, invalid
  Unicode, embedded NUL, malformed tool JSON, and cap overruns fail closed.
- A digest hit is rechecked against authenticated atom bytes; missing atom bytes
  fail closed.
- One digest contributed by multiple components reports every component ID in
  stable proposal order.
- Identical value bytes under different shape IDs or tool field keys produce
  different digests; identical semantics retain all component/field-map pairs.
- Corrupt shape ID, field key, component ordinal, or field-map-entry ordinal is
  rejected before candidate scanning.
- An exclusion-ledger event with an altered shape or field key fails replay even
  when its protected digest is unchanged.
- With benchmark source files absent, the consumer reconstructs the exact
  matcher dispatch from proposal-bound atom and attribution records and never
  applies an unlisted shape.
- Depth 64 passes and depth 65 fails. Each node, leaf, scalar-byte, decoded-row,
  view-count, generated-byte, unique-scanned-byte, and matcher-work cap has
  exact-limit and limit-plus-one fixtures.
- Repeated identical nested views charge every generated byte but only one
  unique scanned view; adversarial deep mappings and lists terminate at a bound
  without quadratic memory growth.
- Capacity and completion observed maxima are independently recomputed, remain
  at or below the proposal limits, and reject altered receipt values.

### Catalog and source fixtures

- Freeze all counts, sizes, hashes, component routing, and order in this design.
- Test the nine exact Speculators subset names and 924-row frozen total.
- Test GSM8K 1,319, MATH-500 500, AIME 30 per year, HumanEval 164, MBPP
  7/257/120/43, LiveCodeBench release_v6 1,055, SWE-bench Test 2,294,
  SPEED qualitative 880, Tau2 base 50/114/114, and Terminal-Bench 80.
- A floating revision, missing file, changed byte, extra selected record,
  renderer hash mismatch, unresolved SPEED marker, or incomplete
  Terminal-Bench tree keeps only the containing catalog in `P`.
- A failed component emits no empty component or catalog receipt.
- Catalog order and stable component order are invariant under filesystem and
  locale order.

### Reader, publication, and approval attacks

- Reject symlink, hardlink, FIFO, socket, device, directory, path traversal,
  oversize, truncation, append, replace-before-open, and rebind-during-read.
- Reject duplicate or unsorted binary digest records and inconsistent headers.
- Prove no-clobber under two concurrent producers.
- Adopt only a completely identical existing artifact set.
- Inject failure at every fsync and publication boundary and prove foreign files
  are preserved.
- Prove the destination parent directory is fsynced after both publish and
  exact adoption before success is returned.
- Reject self-approval, one reviewer, duplicate reviewer keys, unknown or
  revoked keys, altered envelope bytes, wrong scientific identity, wrong
  catalog, and copied signatures.
- Prove proposal and receipt bytes are identical before and after approval and
  that approval-root removal returns the validation outcome to `P`.
- Reject producer-renderer imports in the independent verifier and require a
  signed semantic review for every component.
- Round-trip every frozen schema, reject one added or missing field per schema,
  and test each specified hash view and ordering rule.

### Quota and replay

- Place a held-out collision before every category boundary and prove refill
  occurs only from the same category.
- Exhaust one category after exclusions and prove the build fails instead of
  redistributing or padding.
- Inject duplicate prompt, duplicate row content, and duplicate occurrence
  coordinates and prove each is rejected.
- Prove exact category, subtotal, total, and unique counts.
- Replay with different directory names, process counts, and locale settings;
  all output bytes and ordered digests must match.

## Local-first and Ptyche evidence sequence

The implementation sequence is exact:

1. add failing local identity, schema, containment, adversarial reader,
   publication, signature, quota, and replay tests;
2. make the identity RED/GREEN migration and v2 producer/consumer changes;
3. run focused tests, full dataset unit tests, formatting, lint, type checking,
   Python compilation, and repository diff checks locally;
4. freeze reviewed clean producer and independent-verifier commits and their
   implementation/environment hashes;
5. run each Ptyche acquisition, producer, reviewer, capacity, and dataset script
   with `sbatch --test-only` before any real submission;
6. acquire each public component into a fresh durable input root, leaving the
   unchanged proposal outcome at `P`;
7. obtain signed semantic reviews and perform two independent replay jobs per
   component and catalog;
8. obtain scoped owner signatures and pin the separate approval-root hash,
   producing validation outcome `A` without changing proposal or receipt bytes;
9. run the complete five-catalog bundle verification;
10. run the capacity pass, then build the exact 700K dataset from fresh scratch;
11. independently replay and approve the dataset publication; and
12. hand the approved dataset roots to the separate runtime and launch gates.

Test-only acceptance proves resolved script/config paths, immutable commit and
container identities, nonempty approval-root arguments, output isolation, and
resource geometry. It does not prove source acquisition or dataset validity.
Real jobs are monitored under the existing Ptyche scheduling policy and record
durable logs and receipts in the experiment root.

## Migration and file ownership

This document owns architecture only. A later reviewed implementation plan
assigns nonoverlapping ownership as follows:

- `examples/dataset/specdec_identity.py`: shared prompt and auxiliary identity
  primitives;
- `examples/dataset/build_q30t_heldout_receipts.py`: v2 source, renderer,
  component, catalog, and proposal-bound receipt production;
- a separately versioned held-out verifier module: independent acquisition,
  renderer, index, attribution, replay, and approval verification without
  importing the producer implementation;
- `examples/dataset/build_qwen4b_ptv23_complement.py` or its extracted shared
  core: v2 consumption, candidate scanning, exclusion ledger, deterministic
  refill, and proof receipts;
- dataset tests: RED/GREEN identity, source fixtures, containment, security,
  quotas, and replay;
- Q30 target policy/controller: exact approved v2 roots and no v1 fallback; and
- runtime qualification plan: row A/B evidence and later DFlash/DSpark canaries.

The current relevant file SHA-256 values are baseline observations:

- `specdec_identity.py`:
  `1e0262af3192f61fa8fe4f6ace337e95b65780e7f5dc58baa3f8d290950c9447`;
- `build_q30t_heldout_receipts.py`:
  `6666315c2a4a23f417246e88781d50012523d578c679c7e558471cd4c963295d`;
- `build_qwen4b_ptv23_complement.py`:
  `99b1841a55b23f12fc71a9bbd5c55fb4846e9169a2577f45926e576d31e348b0`.

They must change through reviewed implementation; none is a v2 approval root.
The historical v1 producer remains readable for audit and regression fixtures,
but the Q30 consumer has no production fallback from v2 to v1.

Cross-plan handoffs are hash-only and directional. This data plan supplies the
immutable source proposal, five proposal-bound receipts and indexes, separate
approval root, capacity receipt, dataset manifest, data file, exclusion ledger,
and completion receipt. The controller plan consumes those exact hashes. The
runtime plan supplies row A/B qualification independently. The launch plan
consumes both sets; it may not modify or infer either.

## Alternatives rejected

### Wait indefinitely for internal evaluator-run artifacts

Rejected because the artifacts are not currently available and their absence
blocks useful public evidence. A future internal catalog can be added through
the same authenticated component path, but it is not silently claimed now.

### Exclude only SWE-bench prompt IDs

Rejected because answers, patches, tests, code, and tool material can appear in
larger messages while whole-prompt UUIDs remain different.

### Use only exact whole-field hashes

Rejected because concatenating benchmark text with unrelated text evades field
equality. Exact normalized containment and chunk proofs cover embedding without
introducing fuzzy semantics.

### Use fuzzy or semantic similarity

Rejected because thresholds and model versions are hard to authenticate,
replay, and audit, and can create broad false positives. Such analysis may be a
nonblocking report but cannot admit or reject production rows.

### Use latest branches or dataset aliases

Rejected because source, split, and renderer membership can change without an
observable manifest change. Every production byte must trace to an immutable
revision and file hash.

### Treat source failure as an empty exclusion set

Rejected because it turns an infrastructure failure into evaluator leakage.
Only the affected catalog fails, but the five-catalog bundle cannot approve.

### Shrink, redistribute, or pad the 700K corpus after exclusions

Rejected because it changes the approved scientific mix or repeats training
occurrences. Deterministic same-category forward selection preserves both the
quota and one-pass contract.

## Present blockers and authorization boundary

The following remain blocking after this design:

- the shared prompt identity RED/GREEN implementation is not complete;
- the path-scoped historical identity replay has not independently reproduced
  the authenticated audit;
- the v2 producer, independently versioned verifier, consumer, indexes,
  attribution, semantic reviews, approval root, and keyring are not implemented
  or reviewed;
- all five catalogs validate as `P` until semantic review, independent
  acquisition/replay, scoped signatures, and the separate approval root pass;
- LiveCodeBench release_v6 source bytes still require durable acquisition and
  renderer replay;
- SPEED still requires all 880 rows to be resolved with immutable dependencies,
  including removal of the observed floating EQ-bench reference;
- the selected Terminal-Bench file tree still requires a complete acquisition
  manifest and two replays;
- Tau2 and every other public component still require durable independent
  acquisition receipts despite their frozen source facts above;
- the exact five-name bundle approval root, capacity evidence, and final 700K
  publication do not yet exist; and
- runtime row A/B qualification remains blocked in the separate runtime plan.

Therefore this design does not authorize Qwen3-30B-A3B-Thinking DFlash or
DSpark training, canaries, job submission, or launch. The next authorized step
is review of this architecture, followed by a separate implementation plan and
local tests.

## Primary source coordinates

- Speculators evaluator:
  `https://github.com/vllm-project/speculators/tree/0b08a89a83b92007be63f128e01497455b0209df`
- Red Hat compatibility data:
  `https://huggingface.co/datasets/RedHatAI/speculator_benchmarks/tree/2ae86affa2cb97a972b7fc681dd51c04fbff083e`
- GSM8K:
  `https://huggingface.co/datasets/openai/gsm8k/tree/740312add88f781978c0658806c59bc2815b9866`
- MATH-500:
  `https://huggingface.co/datasets/HuggingFaceH4/MATH-500/tree/6e4ed1a2a79af7d8630a6b768ec859cb5af4d3be`
- AIME 2022-2025:
  `https://huggingface.co/datasets/allenai/aime-2022-2025/tree/73e1eba765ad5847cdb5d1e2e7aaf7b22b585798`
- HumanEval:
  `https://huggingface.co/datasets/openai/openai_humaneval/tree/7dce6050a7d6d172f3cc5c32aa97f52fa1a2e544`
- sanitized MBPP:
  `https://huggingface.co/datasets/google-research-datasets/mbpp/tree/4bb6404fdc6cacfda99d4ac4205087b89d32030c`
- LiveCodeBench:
  `https://huggingface.co/datasets/livecodebench/code_generation_lite/tree/0fe84c3912ea0c4d4a78037083943e8f0c4dd505`
- SWE-bench:
  `https://huggingface.co/datasets/princeton-nlp/SWE-bench/tree/e48e2bd1e9fecd5bbd641e9414ac59da9f2e69f6`
- SPEED-Bench:
  `https://huggingface.co/datasets/nvidia/SPEED-Bench/tree/487aa718444e816458d1a0a52bfce7a454285cf4`
- Tau2:
  `https://github.com/sierra-research/tau2-bench/tree/a2c024725189473d2d7cea3a5cfdbcc67478e41f`
- Terminal-Bench registry:
  `https://github.com/laude-institute/terminal-bench/tree/d28711d0da2675d0bb1d56de45ae5df6082438a3`
