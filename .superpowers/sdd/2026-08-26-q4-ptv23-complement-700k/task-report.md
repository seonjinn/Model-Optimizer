# Q4 PTV2/PTV3 700K Continuation Implementation Report

## Frozen scope

- Base commit: `2ecf4900afb92723a168dcd5dbf681b4583b26c0`
- Branch: `sj/q4-ptv2-ptv3-complement-700k`
- Scientific identity: `ptv2-ptv3-complement-700k-v1`
- No commit, push, SSH, SLURM submission, or job query was performed.
- The implementation consulted the uncommitted `qwen4b-balanced-pilot` worktree only after recording the first RED and treated it as non-authoritative reference material.

The immutable quota is 700,000 rows: PTV2 STEM 300,000; PTV2 multilingual JA, ES, FR, and IT 50,000 each; PTV3 SWE-v3 100,000; PTV3 interactive agentic/SWE 19,000; and PTV3 general tool trajectories 81,000.

## Implemented boundaries

- Strict config and external source-inventory authentication, including exact repository revision, license approval, physical file bytes/SHA-256/row count, ordered file topology, and row-schema SHA-256.
- Caller-pinned historical 1.3M occurrence receipt authentication, including support for the repository's genuine baseline-audit producer, ordered UUID digest, unique exclusion digest, and duplicate multiplicity.
- Individually caller-pinned held-out receipts followed by a deterministic union.
- Category and source occurrence order, exact same-category refill, global deduplication, collision-fatal behavior, and fail-closed capacity receipts without redistribution, replacement, or cycling.
- Full tool declaration/message/call-ID/result/reasoning preservation through the existing trajectory validator; malformed, orphaned, duplicate, or unresolved calls are rejected and refilled.
- Caller-pinned official `Qwen/Qwen3-4B` revision `1cfa9a7208912126459214e8b04321603b3df60c`, snapshot tree, official/template digests, special-token IDs, selected-row-only token evidence, and the 4,096-token gate.
- SQLite selection spooling under `/raid`, canonical JSONL bytes, quota/category counts, ordered UUID and source-occurrence digests, token evidence, exclusions, capacity evidence, historical/held-out roots, runtime SHA, and source commit bound into the manifest.
- Atomic no-replace `/lustre` publication with exact-match retry adoption and no deletion of foreign paths; build and replaying verify CLI modes plus an external no-replace completion receipt.
- Typed DFlash/DSpark parent-checkpoint authentication and continuation contracts that allow weights-only adoption, require fresh optimizer and scheduler identities, run scheduler `--test-only` first, and gate full continuation behind an exact passed 20-step canary receipt.

## Exact external blockers

Production build and training remain fail-closed until these externally produced identities exist. No digest was inferred from the prior uncommitted worktree.

1. The exact PTV2 candidate file inventory and every file's byte count, SHA-256, row count, and source row-schema SHA-256 for STEM and multilingual JA/ES/FR/IT.
2. An authoritative PTV3 SWE-v3 repository revision plus its complete physical file inventory, byte counts, SHA-256 values, row counts, and row-schema SHA-256. The candidate revision and 96-file list seen only in the prior uncommitted worktree were not promoted to trusted pins.
3. Post-historical-exclusion, post-held-out-union, post-global-dedup, post-trace-validation, and post-Qwen-tokenization capacity receipts proving each of the eight exact category quotas.
4. The externally pinned exact historical 1.3M baseline-audit file SHA-256 and all held-out receipt SHA-256 values.
5. The externally pinned official tokenizer trust receipt, runtime SHA-256, source commit, and existing DFlash/DSpark 1.3M checkpoint receipts needed for a real continuation.

The four already authenticated agentic source files and their exact revisions/bytes/SHA-256 values are recorded in `examples/dataset/qwen3_4b_ptv23_complement_sources_v1.json`; they do not by themselves prove the 19,000 or 81,000 valid unique capacities.

## Frozen artifact hashes

| Artifact | SHA-256 |
|---|---|
| `examples/dataset/build_qwen4b_ptv23_complement.py` | `e4e2abeff8c70cef700585486b5a395ed306a059f70c6ef2a14a63d2fdbc4003` |
| `examples/dataset/qwen3_4b_ptv23_complement_700k_v1.json` | `9d70203bb00623d8e2c7bdfb15dfbb3b0d717779a238fc2e87292f88fe1e7ed9` |
| `examples/dataset/qwen3_4b_ptv23_complement_sources_v1.json` | `e61ec87c2aba19c67c4a4549dba11f33abe8a3126f76232df0d72da43e8c81e1` |
| `tests/examples/dataset/test_build_qwen4b_ptv23_complement.py` | `0babf62eaa9c6d5103d39c400a714732078ad79ac8899df7f09a35a629169b82` |
| `tools/launcher/common/specdec/qwen4b_ptv23_continuation.py` | `0e7fcc3c539f2cc40f771de94bb77833f03c9c09b8f1c4d4e5132ba633cd19ab` |
| `tools/launcher/common/specdec/run_qwen4b_ptv23_continuation.sbatch` | `d3713c8060de6e390a6b95187c4711e3cc1e62cbd6baef6d021181df6b6ce2a1` |
| `tools/launcher/common/specdec/submit_qwen4b_ptv23_continuation.sh` | `4452933ca74ff38b85003c360aaff50e73157d15466e200b7b3c766991c74ea4` |
| `tools/launcher/tests/test_qwen4b_ptv23_continuation.py` | `2a7b6620447db434b6442b09958255d72d50388b5edbada8a04d8c0ecf8f2e4f` |

## Verification evidence

- Relevant pytest: `129 passed, 1 skipped`; the skip is the existing macOS `renameat2` test.
- Ruff check: clean.
- Ruff format check: clean.
- Pyright: `0 errors, 0 warnings, 0 informations`.
- `py_compile`: clean.
- `bash -n`: clean.
- ShellCheck: clean after replacing the ambiguous boolean chain.
- `git diff --check`: clean.

The first independent-review dispatch was rejected because all four shared agent slots were occupied. The root agent was notified and asked to reuse the released slot for a fresh independent review/transfer audit. This handoff is frozen pending that review.

## First hostile-review corrective round

The fresh review reported four critical findings and six warnings. Before production edits, each finding was reproduced independently against the exact frozen bytes. The corrective round then applied one RED/GREEN fix at a time:

- The executable config loader now precedes the `__main__` guard.
- Replay verification enforces the exact approved 700K quotas, total row count, manifest schema, and category block order.
- Continuation contracts authenticate the completed dataset manifest and caller-pinned completion receipt, then bind the exact historical receipt and ordered historical UUID digest to the parent checkpoint.
- A zero-exit training process can no longer mint a passed canary receipt. The digest-pinned training entrypoint must emit canonical evidence of observed global step 20 and canary success.
- Replay verification reruns trajectory validation for both agentic categories.
- Checked source requirements are parsed and reconciled against exact inventory pins and the sufficient capacity receipt; unresolved blocker vocabulary fails closed.
- Source iteration uses a no-follow descriptor with pre- and post-iteration physical identity validation.
- Publication retry matching and replay verification hash `DATA.jsonl` as a stream. Replay verification also incrementally hashes ordered manifest evidence instead of materializing whole-file arrays.
- Selection stops before requesting the next source occurrence once a category quota is full.
- The no-home runner authenticates and mounts both the contract tool and training entrypoint explicitly.

Post-correction focused evidence is `57 passed` for builder and trajectory contracts plus `7 passed` for launcher contracts. Ruff, Pyright, shell, compilation, and frozen-diff checks are rerun at the final handoff below.

## Fresh rereview verdict

The exact corrective bytes were independently rereviewed and remain **NOT CLEAN** with five new critical findings and three warnings. No commit, push, SSH, or job action is authorized from this freeze.

The remaining critical boundaries are architectural rather than regressions in the original ten fixes: the launcher must independently replay or trust-root the builder's full 700K semantic verification; the authoritative PTV3 SWE revision and row schemas require an immutable external approval root; the exact named held-out receipt set must be mandatory; canary evidence must bind immutable trainer, image, and output-checkpoint identities; and full training must bind an exact exposure schedule plus final checkpoint/completion evidence. The warnings require mounting a distinct canary-evidence directory, streaming the three publication evidence hashes, and replacing `_stable_file`'s path-based read with a no-follow descriptor.

These findings require new externally approved identities and a wider contract design. They were not papered over with additional caller-self-attested JSON fields. The branch remains local and is explicitly ineligible for build or submission pending that next design round and another independent review.

## Second hostile-review corrective round

All five critical and three warning findings were reproduced before edits and received dedicated RED tests. The exact builder verifier is now digest-pinned and executed during contract creation after a physical 700K/category-order precheck. The builder itself replays token and trajectory semantics, and its manifest verifier requires the exact named `speed`, `math`, `code`, `swe`, and `tool` held-out receipt set.

Authoritative PTV3 SWE source/revision/schema and trainer/container allowlists are explicit immutable trust roots. They intentionally remain empty, so production build, contract creation, contract loading, and launch fail closed until those external approvals are supplied; arbitrary caller hashes cannot resolve them. The exact reviewed builder digest is populated.

The continuation contract now pins trainer, image, parent checkpoint, global batch, exact one-pass 700K exposure, and the unique full-step ceiling. Canary evidence binds those identities plus a live output checkpoint tree. Full launch must equal the contract step count and emits a final receipt only after validating 700K consumption and the durable final checkpoint tree. Separate canary/full evidence directories are mounted, publication evidence hashes are incremental, and receipt reads use one no-follow descriptor.

Post-round focused evidence is `60 passed` for builder/trajectory contracts and `11 passed` for launcher contracts. This freeze still requires fresh independent rereview before any commit, push, build, SSH, or job action.

## Third hostile-review corrective round

The second rereview returned seven critical findings and five warnings. Every finding was reproduced against the frozen bytes before production edits and received a dedicated RED contract. This round closes the caller-self-attestation paths rather than adding more unchecked receipt fields.

Both contract creation and contract loading now execute the same digest-pinned builder semantic verifier. Its `trajectory_schema.py` and historical-audit authenticator dependency are independently pinned and loaded under controlled module identities. The external tokenizer receipt is immutable-allowlisted, checked against the manifest, and its live snapshot tree is replayed. The contract persists the builder, dependency, config, inventory, historical, exact five held-out, tokenizer, parent-receipt, and scratch roots needed to repeat verification at launch.

Builder verification now authenticates the externally approved inventory, historical receipt, and exact named held-out receipts; reconciles PTV3 SWE and every row-schema allowlist; replays the front-filled selection from physical source rows with historical/held-out exclusion, deduplication, trajectory validation, tokenization, category quotas, capacity, and exact row equality; and requires nonzero assistant supervision. All trusted byte readers use one no-follow descriptor. Empty approval roots fail before the 700K selection, and copied shared-storage publication bytes are rehashed before atomic rename.

The parent receipt is live-replayed and must belong to an immutable parent-receipt allowlist. A full-stage canary receipt now names and reauthenticates its live training evidence and checkpoint tree. The trainer receives the contract global batch, exact example limit, one dataset pass, and explicit no-cycling control; the schedule requires `full_steps * global_batch_size == 700000`. The no-home runner derives and mounts the authenticated dataset, completion, builder dependencies, source files, tokenizer snapshot, historical/held-out receipts, parent checkpoint/receipt, canary evidence/checkpoint, verification scratch, and fresh output roots explicitly.

The final local GREEN evidence before rereview is `65 passed` for builder/trajectory contracts and `17 passed` for launcher contracts. Ruff format/check, Pyright (`0 errors, 0 warnings, 0 informations`), `py_compile`, `bash -n`, ShellCheck, and whitespace checks are clean. The known macOS pytest temporary-directory cleanup warnings are external to these tests and do not change the pass result.

Real build or canary execution intentionally remains fail-closed until externally reviewed values populate the source-inventory, PTV3 SWE, row-schema, historical receipt, exact five held-out receipt, official tokenizer receipt, parent checkpoint receipt, trainer, and runtime-image approval roots.

Current artifact hashes before the fresh rereview:

| Artifact | SHA-256 |
|---|---|
| `examples/dataset/build_qwen4b_ptv23_complement.py` | `9007452b7d015ab61fbfe8bc5351b7d4ede242d45fd8da281669c4fde193c9f7` |
| `examples/dataset/qwen3_4b_ptv23_complement_700k_v1.json` | `9d70203bb00623d8e2c7bdfb15dfbb3b0d717779a238fc2e87292f88fe1e7ed9` |
| `examples/dataset/qwen3_4b_ptv23_complement_sources_v1.json` | `e61ec87c2aba19c67c4a4549dba11f33abe8a3126f76232df0d72da43e8c81e1` |
| `tests/examples/dataset/test_build_qwen4b_ptv23_complement.py` | `61f001e484291f5cd73e28c6260fafe351072cbe311e5413aaeba8a686eac2d4` |
| `tools/launcher/common/specdec/qwen4b_ptv23_continuation.py` | `9a105776751c195d441dabb539a59a52106a2509a540af1f0206ebf69521327f` |
| `tools/launcher/common/specdec/run_qwen4b_ptv23_continuation.sbatch` | `e0fbd4389b258ebb6198f00c3ae98a50d679300a8e73ce16620965123cc66ebd` |
| `tools/launcher/common/specdec/submit_qwen4b_ptv23_continuation.sh` | `188ddcf4461ba8ae7f9ff65695e12492ca4807d40ff8aa0e8aa6f2244e36ece4` |
| `tools/launcher/tests/test_qwen4b_ptv23_continuation.py` | `125d25bbc64263a2c5980fe7fff1137b71988520138539d76af634d3b119f455` |

## Fourth hostile-review corrective round

The third rereview found six critical issues, four warnings, and one help-text nit. Each finding was first reproduced with a RED contract. The executable verifier closure now pins and controlled-loads `specdec_corpus_contracts.py`, `specdec_identity.py`, and `trajectory_schema.py`; the obsolete broad historical fallback is no longer executable. The runner compares the launcher against a reviewed built-in digest instead of trusting a caller-provided digest.

Contracts now bind the trainer and runtime-image paths as well as their immutable hashes. Canary and full execution-stage, checkpoint, evidence, and receipt roots are distinct contract fields. Effective writable mount roots must be fresh, no-follow, disjoint from every authenticated input and verification scratch root, and isolated between canary and full stages. The runner derives those mounts from the contract and no longer remounts canary evidence or checkpoints read-write during the full stage.

The tokenizer is loaded only from a fresh staged snapshot whose final tree matches the approved digest. Dataset, trainer, and parent-checkpoint inputs are streamed into fresh authenticated execution stages and revalidated before the trainer sees them, closing verification-to-use path races. Stable readers reject non-regular descriptors, publication and staging remain streaming, replay scratch uses a managed temporary directory, and held-out help documents the required `NAME=PATH:SHA256` syntax.

Direct launch deliberately remains fail-closed through `RUNTIME_ATTESTATION_IMPLEMENTED = False`: a Python process cannot prove which container image is executing it. Production launch therefore still needs an independently reviewed platform runtime-attestation mechanism in addition to populating the already-empty external allowlists. No build, canary, submission, SSH, commit, or push was performed.

Fourth-round local GREEN evidence is 69 passed for builder/trajectory contracts and 23 passed for launcher contracts. Ruff format/check, Pyright (`0 errors, 0 warnings, 0 informations`), `py_compile`, `bash -n`, ShellCheck, and no-index whitespace checks are clean. Pytest still reports the known macOS temporary-directory cleanup warning external to these tests.

| Fourth-round artifact | SHA-256 |
|---|---|
| `examples/dataset/build_qwen4b_ptv23_complement.py` | `615d3977c4e9ba7edfdb3c3219e9fbbb218fd0025a33e26055bcd185d984be8d` |
| `examples/dataset/qwen3_4b_ptv23_complement_700k_v1.json` | `9d70203bb00623d8e2c7bdfb15dfbb3b0d717779a238fc2e87292f88fe1e7ed9` |
| `examples/dataset/qwen3_4b_ptv23_complement_sources_v1.json` | `e61ec87c2aba19c67c4a4549dba11f33abe8a3126f76232df0d72da43e8c81e1` |
| `tests/examples/dataset/test_build_qwen4b_ptv23_complement.py` | `7241f6d8943a0294de1ca48c479e2dc4db42ad9abc410466ec03727affea14a9` |
| `tools/launcher/common/specdec/qwen4b_ptv23_continuation.py` | `25511178aa6a5790bc80365e99042ffc4b8b3d845a0cc8aedf2b01fbe8b94e62` |
| `tools/launcher/common/specdec/run_qwen4b_ptv23_continuation.sbatch` | `b5735ef769d4134b997659380e2ed09b439fe026d75a09ac5f44104100ddf604` |
| `tools/launcher/common/specdec/submit_qwen4b_ptv23_continuation.sh` | `188ddcf4461ba8ae7f9ff65695e12492ca4807d40ff8aa0e8aa6f2244e36ece4` |
| `tools/launcher/tests/test_qwen4b_ptv23_continuation.py` | `3176adcb70e7e07ab7c988ad5aed7e54ffa310597b58c2161e52b18b0287f13d` |

## Fifth hostile-review corrective round

The fourth rereview returned five critical findings and four warnings. RED tests reproduced the import-shadow, incomplete protected-root, hash-to-use, pre-validation host mutation, descriptor row-count, replay cleanup, fail-fast, and source-only-test gaps before this round changed production bytes.

Semantic replay now requires isolated Python resolution, and the runner invokes both preflight and launch with `python -I`. Contract, exact launcher, and runtime image are copied through one no-follow regular descriptor into a fresh runner-owned stage, hashed while copying, and only the staged bytes are reopened. An isolated lightweight contract preflight authenticates the contract and immutable live roots before any contract-derived host path is created or mounted. The full semantic replay still runs at launch inside the isolated staged tool.

Writable-root validation now expands the exact held-out receipts, physical inventory files, tokenizer snapshot, builder, every pinned local dependency, historical authenticator, and staged contract path. The same closure is rechecked immediately before execution staging. Runtime attestation is fail-fast in launcher, runner, and submitter; no expensive semantic replay or submission occurs while that known external capability is absent.

Physical row counting uses one no-follow regular descriptor, including `/dev/fd` for Parquet metadata, with post-read descriptor stability validation. Replay scratch is owned by a `with tempfile.TemporaryDirectory(...)` boundary, so traceback retention cannot defer cleanup. New tests behaviorally exercise isolated replay rejection, expanded semantic roots, fail-fast ordering, staged runner boundaries, and non-regular row-count rejection.

Fifth-round local GREEN evidence is 71 passed for builder/trajectory contracts and 27 passed for launcher contracts. Ruff format/check, Pyright (`0 errors, 0 warnings, 0 informations`), `py_compile`, `bash -n`, ShellCheck, and no-index whitespace checks are clean. No build, canary, submission, SSH, commit, or push was performed.

| Fifth-round artifact | SHA-256 |
|---|---|
| `examples/dataset/build_qwen4b_ptv23_complement.py` | `6323bb447607351e342c314c2f8bd74e0783d193666060ac1306d11bfa9773d5` |
| `examples/dataset/qwen3_4b_ptv23_complement_700k_v1.json` | `9d70203bb00623d8e2c7bdfb15dfbb3b0d717779a238fc2e87292f88fe1e7ed9` |
| `examples/dataset/qwen3_4b_ptv23_complement_sources_v1.json` | `e61ec87c2aba19c67c4a4549dba11f33abe8a3126f76232df0d72da43e8c81e1` |
| `tests/examples/dataset/test_build_qwen4b_ptv23_complement.py` | `3233f570fdaee42c423e01bd53a0f2d74b26a72bae3e45d72a25407a6dbb2a8b` |
| `tools/launcher/common/specdec/qwen4b_ptv23_continuation.py` | `adff2f8e1ee6132e27222679e94e3a78c66c33eaf3dce318318d127e01bb5031` |
| `tools/launcher/common/specdec/run_qwen4b_ptv23_continuation.sbatch` | `b99c4bf3173f6ed7ce4f336647544edd7ad2784295f0eb7945d3e4c005474dff` |
| `tools/launcher/common/specdec/submit_qwen4b_ptv23_continuation.sh` | `6d7a68bede00b7daa605bcf74def82152baa8a5e3740ac89971d6ef7cf3d29fb` |
| `tools/launcher/tests/test_qwen4b_ptv23_continuation.py` | `7e02294012431ab63ed7d631e70b7b9cff5cd56e09863387b8252742f9d0bb65` |

## Sixth hostile-review corrective round

The fifth rereview returned two critical findings and two warnings. RED tests reproduced unvalidated canary-receipt mount injection and ancestor-symlink redirection before production changes.

The trusted isolated preflight now emits a canonical mount plan exclusively from authenticated contract fields. Full-stage read-only mounts use the contract-bound canary checkpoint, evidence, and receipt roots and never parse caller receipt JSON before `_validate_canary_receipt`. The shell no longer interprets any contract or receipt paths.

A runner-owned isolated Python boundary copies contract, launcher, and—only after successful preflight—the runtime image into anonymous `O_TMPFILE` objects while hashing a no-follow source descriptor. It retains inheritable descriptors through `exec` and supplies `/proc/self/fd` sources to pyxis and Python, so pathname replacement cannot change the bytes used. Every read-only directory is opened component-by-component with `O_DIRECTORY|O_NOFOLLOW`. Writable directories are created descriptor-relative from `/`, opened the same way, and mounted from retained descriptors, closing the preflight-to-mkdir/mount ancestor race. The inline runner Python is separately syntax-compiled in verification.

Sixth-round local GREEN evidence is 71 passed for builder/trajectory contracts and 29 passed for launcher contracts. Ruff format/check, Pyright (`0 errors, 0 warnings, 0 informations`), `py_compile`, `bash -n`, ShellCheck, inline-Python compilation, and no-index whitespace checks are clean. No build, canary, submission, SSH, commit, or push was performed.

| Sixth-round artifact | SHA-256 |
|---|---|
| `examples/dataset/build_qwen4b_ptv23_complement.py` | `6323bb447607351e342c314c2f8bd74e0783d193666060ac1306d11bfa9773d5` |
| `examples/dataset/qwen3_4b_ptv23_complement_700k_v1.json` | `9d70203bb00623d8e2c7bdfb15dfbb3b0d717779a238fc2e87292f88fe1e7ed9` |
| `examples/dataset/qwen3_4b_ptv23_complement_sources_v1.json` | `e61ec87c2aba19c67c4a4549dba11f33abe8a3126f76232df0d72da43e8c81e1` |
| `tests/examples/dataset/test_build_qwen4b_ptv23_complement.py` | `3233f570fdaee42c423e01bd53a0f2d74b26a72bae3e45d72a25407a6dbb2a8b` |
| `tools/launcher/common/specdec/qwen4b_ptv23_continuation.py` | `19bc4843813f499d4a5930928ff9c30dd7efab95a32cfdb9802d20f6d568277a` |
| `tools/launcher/common/specdec/run_qwen4b_ptv23_continuation.sbatch` | `1b54391200fd381290c7960fca7ff5361dc0ce33638ed41c5a2f345f7b6601be` |
| `tools/launcher/common/specdec/submit_qwen4b_ptv23_continuation.sh` | `6d7a68bede00b7daa605bcf74def82152baa8a5e3740ac89971d6ef7cf3d29fb` |
| `tools/launcher/tests/test_qwen4b_ptv23_continuation.py` | `5c4000c393d66e61afba1c0eca0c4b009864a248804b7976ac2762d94e0356ed` |

## Seventh hostile-review corrective round

The sixth rereview returned two critical findings and two warnings. RED tests first
reproduced the invalid `/proc/self/fd` Pyxis boundary, short-write corruption,
missing staged-output authentication, unsupported `O_TMPFILE`, and source-only
runner coverage.

The runner now keeps its descriptor-owning Python process alive through `srun` and
uses `/proc/<keeper-pid>/fd` rather than daemon-relative `/proc/self/fd`. A real
child-process probe verifies that the retained descriptors are reopenable before
launch. Because one keeper PID cannot exist on remote nodes, the runner explicitly
fails closed for multi-node allocations instead of claiming a false Pyxis contract;
a reviewed per-node staging/keeper mechanism remains a platform prerequisite for
the intended multi-node canary.

Anonymous copies now loop over short writes, fsync, independently re-read and hash
the exact staged descriptor, and compare its size and digest before it can be
consumed. Missing Python `O_TMPFILE` support or an unsupported node-local filesystem
produces an actionable fail-closed error. Tests extract and execute the actual
embedded runner functions for unsupported-capability behavior, and Linux-capable
environments additionally exercise short writes and a cross-process procfs reopen.

Seventh-round local GREEN evidence is 71 passed for builder/trajectory contracts
and 31 passed, 2 platform skips for launcher contracts. The skips are the Linux
procfs and `O_TMPFILE` positive-path probes on macOS; their fail-closed paths and
source contracts pass locally. Ruff format/check, Pyright (`0 errors, 0 warnings,
0 informations`), `py_compile`, inline-runner compilation, `bash -n`, and ShellCheck
are clean. No build, canary, submission, SSH, commit, or push was performed.

Real build/canary execution still requires the externally reviewed source inventory,
PTV3 SWE and row-schema allowlists, historical receipt, exact five held-out receipts,
official tokenizer receipt, parent receipt, trainer, runtime image, genuine platform
runtime-image attestation, and a multi-node Pyxis staging contract.

| Seventh-round artifact | SHA-256 |
|---|---|
| `examples/dataset/build_qwen4b_ptv23_complement.py` | `6323bb447607351e342c314c2f8bd74e0783d193666060ac1306d11bfa9773d5` |
| `examples/dataset/qwen3_4b_ptv23_complement_700k_v1.json` | `9d70203bb00623d8e2c7bdfb15dfbb3b0d717779a238fc2e87292f88fe1e7ed9` |
| `examples/dataset/qwen3_4b_ptv23_complement_sources_v1.json` | `e61ec87c2aba19c67c4a4549dba11f33abe8a3126f76232df0d72da43e8c81e1` |
| `tests/examples/dataset/test_build_qwen4b_ptv23_complement.py` | `3233f570fdaee42c423e01bd53a0f2d74b26a72bae3e45d72a25407a6dbb2a8b` |
| `tools/launcher/common/specdec/qwen4b_ptv23_continuation.py` | `19bc4843813f499d4a5930928ff9c30dd7efab95a32cfdb9802d20f6d568277a` |
| `tools/launcher/common/specdec/run_qwen4b_ptv23_continuation.sbatch` | `5273e1be753e8d4376cb6beb01948dd2b2912a911e7ddcc02525b4cf80ad3ff9` |
| `tools/launcher/common/specdec/submit_qwen4b_ptv23_continuation.sh` | `6d7a68bede00b7daa605bcf74def82152baa8a5e3740ac89971d6ef7cf3d29fb` |
| `tools/launcher/tests/test_qwen4b_ptv23_continuation.py` | `922c0be937be7e340452999b22e55074442e9a40d83c1c7d8a4858348ebae0eb` |

## Eighth hostile-review corrective round

The seventh rereview returned three critical findings and three warnings. A
behavioral RED fixture reproduced that fully caller-authored checkpoint bytes,
step-20 evidence, and a self-hashed receipt were accepted. Dedicated REDs also
reproduced Pyxis mount-grammar injection, trainer import-environment inheritance,
SLURM export injection, and destination replacement during publication.

Full-stage canary admission now requires the exact receipt digest to belong to an
immutable `TRUSTED_CANARY_RECEIPT_SHA256S` approval root. That root is intentionally
empty, so a canary cannot self-authorize its own full run. Approved receipts must
also use the exact contract canary receipt/evidence paths and a non-root checkpoint
strictly below the contract canary checkpoint root.

Every contract-derived Pyxis mount target rejects comma, colon, NUL-equivalent
controls, and other control characters before mount assembly. The trainer runs as
isolated Python and receives a reviewed environment that excludes `PYTHONPATH`,
`PYTHONHOME`, `LD_PRELOAD`, and caller `PATH`; the `srun` boundary uses
`--export=NONE`. The submitter no longer exports `ALL`, validates every explicit
SLURM export value as printable and comma-free before the runtime blocker, and
therefore cannot inject or overwrite exported variables through a path.

Publication unconditionally reauthenticates the installed destination after the
atomic no-replace rename and before returning its completion identity. A fault
injection test that substitutes a foreign destination at rename time is rejected.

Eighth-round local GREEN evidence is 72 passed for builder/trajectory contracts
and 35 passed, 2 platform skips for launcher contracts. Ruff format/check, Pyright
(`0 errors, 0 warnings, 0 informations`), `py_compile`, inline-runner compilation,
`bash -n`, ShellCheck, embedded-launcher digest equality, and whitespace checks are
clean. The remaining Pyxis warning is an intentionally explicit external contract:
single-node `/proc/<keeper>/fd` consumption still needs a real Linux
slurmstepd/Pyxis probe, while multi-node and all runtime launch remain fail-closed.
No build, canary, submission, SSH, commit, or push was performed.

| Eighth-round artifact | SHA-256 |
|---|---|
| `examples/dataset/build_qwen4b_ptv23_complement.py` | `399f84297c387bb4429569279eb7dfe259ec83aa1a10b08590013f9cb678c263` |
| `examples/dataset/qwen3_4b_ptv23_complement_700k_v1.json` | `9d70203bb00623d8e2c7bdfb15dfbb3b0d717779a238fc2e87292f88fe1e7ed9` |
| `examples/dataset/qwen3_4b_ptv23_complement_sources_v1.json` | `e61ec87c2aba19c67c4a4549dba11f33abe8a3126f76232df0d72da43e8c81e1` |
| `tests/examples/dataset/test_build_qwen4b_ptv23_complement.py` | `42544abca3684832980a7cd122290b7b047ea86c26f09f6809ef7a09c7b37788` |
| `tools/launcher/common/specdec/qwen4b_ptv23_continuation.py` | `9cd5ea0c0a01c46197fa73abc8a9d9f43fe2838e7f422497180f293e377e8273` |
| `tools/launcher/common/specdec/run_qwen4b_ptv23_continuation.sbatch` | `c671821388ccc2a1612bec0fa8080f33d0c0b3c0e77b0474b0d2b424aa833198` |
| `tools/launcher/common/specdec/submit_qwen4b_ptv23_continuation.sh` | `b8fad6fb1ce728c64a36edb0182cf4a01f84001d88ed5323cf35e7700e2014db` |
| `tools/launcher/tests/test_qwen4b_ptv23_continuation.py` | `858915a21499a9367632bad2b86216b7efe226dbf3beed00fff98275eaa146d2` |

## Ninth hostile-review corrective round

The eighth rereview found one critical stale-digest defect and no warnings or nits.
The post-rename builder hardening changed the reviewed builder bytes, but the
launcher still pinned the prior builder digest, making semantic replay permanently
fail closed even after external approval roots were populated.

A RED regression first proved the launcher trust pin did not equal the live frozen
builder SHA-256. The pin now matches the exact builder bytes, and the resulting
launcher SHA-256 is re-embedded in the runner. The regression prevents future
builder changes from silently leaving an unusable semantic replay closure.

Ninth-round local GREEN evidence is 72 passed for builder/trajectory contracts and
36 passed, 2 platform skips for launcher contracts. Ruff format/check, Pyright
(`0 errors, 0 warnings, 0 informations`), `py_compile`, inline-runner compilation,
`bash -n`, ShellCheck, launcher/runner digest equality, and whitespace checks are
clean. No build, canary, submission, SSH, commit, or push was performed.

| Ninth-round artifact | SHA-256 |
|---|---|
| `examples/dataset/build_qwen4b_ptv23_complement.py` | `399f84297c387bb4429569279eb7dfe259ec83aa1a10b08590013f9cb678c263` |
| `examples/dataset/qwen3_4b_ptv23_complement_700k_v1.json` | `9d70203bb00623d8e2c7bdfb15dfbb3b0d717779a238fc2e87292f88fe1e7ed9` |
| `examples/dataset/qwen3_4b_ptv23_complement_sources_v1.json` | `e61ec87c2aba19c67c4a4549dba11f33abe8a3126f76232df0d72da43e8c81e1` |
| `tests/examples/dataset/test_build_qwen4b_ptv23_complement.py` | `42544abca3684832980a7cd122290b7b047ea86c26f09f6809ef7a09c7b37788` |
| `tools/launcher/common/specdec/qwen4b_ptv23_continuation.py` | `f43eaa65cae214de9fd9d8e859bb5e9f51484bae8035d6f9985ce0bb9137f241` |
| `tools/launcher/common/specdec/run_qwen4b_ptv23_continuation.sbatch` | `e0464ae31e4a6f2adc85b724be1c518ac8f8a98d478bfc29a0ededa4bae45457` |
| `tools/launcher/common/specdec/submit_qwen4b_ptv23_continuation.sh` | `b8fad6fb1ce728c64a36edb0182cf4a01f84001d88ed5323cf35e7700e2014db` |
| `tools/launcher/tests/test_qwen4b_ptv23_continuation.py` | `10f47d1b4bfb9bb025378eb393fea48f8acd5a83f807bf3756c31df686321a8c` |

## Tenth hostile-review corrective round

The ninth rereview returned zero critical findings, one warning, and zero nits.
Contract creation/loading accepted an exact-700K schedule whose full step count was
at or below the fixed 20-step canary, while the runner correctly refused such a
full launch. A behavioral RED test reproduced this inconsistent contract state.

One shared schedule validator now requires integer positive batch arithmetic, exact
700K exposure, and `full_steps > 20`. Both contract creation and loading call the
same validator, so every authenticated contract is consistent with the runner's
canary/full ordering. The launcher digest was re-embedded in the runner.

Tenth-round local GREEN evidence is 72 passed for builder/trajectory contracts and
37 passed, 2 platform skips for launcher contracts. Ruff format/check, Pyright
(`0 errors, 0 warnings, 0 informations`), `py_compile`, inline-runner compilation,
`bash -n`, ShellCheck, and launcher/runner digest equality are clean. No build,
canary, submission, SSH, commit, or push was performed.

| Tenth-round artifact | SHA-256 |
|---|---|
| `examples/dataset/build_qwen4b_ptv23_complement.py` | `399f84297c387bb4429569279eb7dfe259ec83aa1a10b08590013f9cb678c263` |
| `examples/dataset/qwen3_4b_ptv23_complement_700k_v1.json` | `9d70203bb00623d8e2c7bdfb15dfbb3b0d717779a238fc2e87292f88fe1e7ed9` |
| `examples/dataset/qwen3_4b_ptv23_complement_sources_v1.json` | `e61ec87c2aba19c67c4a4549dba11f33abe8a3126f76232df0d72da43e8c81e1` |
| `tests/examples/dataset/test_build_qwen4b_ptv23_complement.py` | `42544abca3684832980a7cd122290b7b047ea86c26f09f6809ef7a09c7b37788` |
| `tools/launcher/common/specdec/qwen4b_ptv23_continuation.py` | `6751546191dedda988ba04b00db8fef8c30b9f6f431cd1c93e9e530e1dbd94de` |
| `tools/launcher/common/specdec/run_qwen4b_ptv23_continuation.sbatch` | `1ad37c23196d60053526d238890f1ff3c38de08a0c5e9927ccf2d56961dec440` |
| `tools/launcher/common/specdec/submit_qwen4b_ptv23_continuation.sh` | `b8fad6fb1ce728c64a36edb0182cf4a01f84001d88ed5323cf35e7700e2014db` |
| `tools/launcher/tests/test_qwen4b_ptv23_continuation.py` | `31a22fff4c2f23f2ab4dbeceafb11c909ff87870668736f479841da733fcd8e7` |
