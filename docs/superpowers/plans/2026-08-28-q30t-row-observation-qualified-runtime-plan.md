# Q30T Qualified-Runtime Row Observation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the invalid host-PyArrow Q30T row-schema observation path with a one-node keeper-backed Pyxis operation using one immutable runtime image shared with training, then produce byte-identical Ptyche observations A and B for all 201 PTV2 and 100 PTV3 files.

**Architecture:** The runtime prerequisite seals the reviewed base SQSH plus exact runtime archive into a content-addressed derived SQSH with `/opt/q30t-runtime`; neither observation nor training executes a mutable extracted runtime. One canonical row-runtime module owns approval roots, qualification/profile/archive/derived-image replay, deterministic observation evidence, and job-specific operation receipts. The observer consumes that module's deterministic evidence and lazily imports PyArrow through a held `-I -S` bootstrap. The Slurm runner stages the derived SQSH and committed tools through keeper descriptors, runs a named Pyxis container under bounded supervision, and follows one explicit publication state machine. Scientific observations omit job-specific evidence, so A and B remain byte-identical; separate operation receipts bind each job to the common observation SHA.

**Tech Stack:** Python 3.12+, Bash, SLURM, Pyxis/enroot, derived SQSH images, `/raid/scratch`, keeper descriptor anchors, canonical JSON/SHA-256 receipts, pytest, Ruff, Pyright, ShellCheck

**Spec:** `docs/superpowers/specs/2026-08-28-q30t-row-observation-qualified-runtime-design.md`

## Global Constraints

- Training is not authorized by this plan. It produces row-schema evidence for later data tasks.
- Reuse the exact runtime archive SHA-256 `4a20aee61f290c48bed22a84b4a0ae0cbdc54e3e3910854d253188c8854f5dc9` and base-image SHA-256 `e7be53f2754097c88f7c801da92f6d94794ec4d78d9df937fcd315a6994297f0`; never alter either source asset.
- Produce one immutable derived SQSH that installs the exact archive tree at `/opt/q30t-runtime`. The same derived image is the only runtime admitted for observation and later training.
- `APPROVED_Q30T_ROW_RUNTIME_QUALIFICATION_RECEIPT_FILE_SHA256S`, `APPROVED_Q30T_ROW_DERIVED_IMAGE_RECEIPT_FILE_SHA256S`, and `APPROVED_Q30T_PTV3_COMPLETION_RECEIPT_FILE_SHA256S` are empty in producer commit P. Only independent receipt review may populate them in a separate signed approval-only descendant commit A.
- Runtime qualification owns the derived-image producer/receipt, archive-tree receipt, attestation, profile, keeper, probe, submitter, and focused tests until a signed hostile-review-clean handoff. The derived SQSH passes two isolated byte-identical builds with normalized metadata. The frozen keeper stages it to an anonymous `O_TMPFILE`, closes all writers, holds a read lease, and treats lease-break notification as fatal. The probe contains no mutable runtime extraction, mutable-checkout import, `SLURM_EXPORT_ENV=ALL`, or pathname deletion.
- Every host Python command uses approved `/usr/bin/python3.12 -I -S`. Every container Python command uses `/opt/q30t-runtime/bin/python -I -S` through a held authenticated bootstrap.
- The bootstrap explicitly inserts only `/opt/q30t-runtime/lib/python3.12/site-packages` after receipt/tree replay. It never uses `PYTHONPATH`, `site`, `.pth`, caller `PATH`, or a mutable checkout.
- Git and SSH use absolute approved executables, sterile configuration, `GIT_NO_REPLACE_OBJECTS=1`, and `/usr/bin/ssh -F /dev/null`.
- No phase sets `SLURM_EXPORT_ENV=ALL`. Child environments are explicit allowlists.
- Keeper or child failure follows Task 4's state machine and never produces an operation receipt.
- Existing exact output is adopted only after canonical and semantic replay. Foreign outputs and retained private scratch are preserved; no failure recursively deletes or pathname-unlinks possibly foreign data.
- Ptyche operations use a clean pushed signed commit, `sbatch --test-only` before each real submission, one filtered scheduler query per minute, and at least five minutes of early monitoring.
- All Python interfaces are fully typed. Canonical loaders reject missing, extra, reordered, duplicate, non-canonical, oversized, symlinked, multiply linked, rebound, or self-hash-mismatched evidence.
- Do not modify or stage unrelated Q4 continuation files, companion-arm files, `uv.lock`, or existing user changes in the integration worktree.

## Cross-Plan Handoffs

| Phase | Owner | Gate |
|---|---|---|
| Runtime immutability repair | Runtime qualification plan | Derived-image receipt/loader; repaired probe/keeper/submitter; signed full GREEN; fresh hostile review CLEAN |
| Task 1 | This plan | May begin with fixture evidence while runtime repair proceeds |
| Tasks 2-6 | This plan | Begin only after repaired runtime interfaces are frozen and integrated |
| Task 7 | Independent reviewer | Exact row-observation range is CLEAN before push |
| Task 8 producer | Ptyche operations | Push producer P; produce/review derived image and one-/two-node qualification from P |
| Task 8 approval | This plan | Add only reviewed receipt hashes in approval-only descendant A; review and push A |
| Task 8 A/B | Ptyche operations | Run both observations from A with the same approved runtime tuple |
| Task 9 | Data Task 4B handoff | Requires byte-identical A/B plus two valid operation receipts and independent review |

## File Responsibility Map

- `tools/launcher/common/specdec/q30t_row_observation_runtime.py`: sole runtime-evidence types/loaders, immutable approval roots, cross-replay, deterministic evidence, and operation receipts.
- `tools/launcher/common/specdec/q30t_row_observation_bootstrap.py`: held `-I -S` control-module loader; it never adds the control directory to `sys.path`.
- `tools/launcher/tests/test_q30t_row_observation_runtime.py`: canonical records, approval roots, cross-replay, provenance, and path attacks.
- `examples/dataset/observe_q30t_ptv23_row_schemas.py`: semantic scan, shared runtime loader consumption, v2 observation schema, and private staged payload.
- `tests/examples/dataset/test_observe_q30t_ptv23_row_schemas.py`: observer runtime binding, file/row counts, determinism, and mutations.
- `tools/launcher/common/specdec/run_q30t_row_schema_observation.sbatch`: immutable bootstrap, keeper/Pyxis execution, supervision, reconciliation, and publication state machine.
- `tools/launcher/common/specdec/submit_q30t_row_schema_observation.sh`: clean pushed source, exact runtime arguments, test-only-first submission.
- `tools/launcher/tests/test_q30t_row_schema_observation_runner.py`: static and executable fake-Slurm/Pyxis/keeper behavior.
- `.superpowers/sdd/2026-08-28-q30t-row-observation-qualified-runtime/`: plan ledger, briefs, reports, and review packages.

---

### Task 1: Define One Canonical Runtime Evidence Loader and Observer v2

**Files:**

- Create: `tools/launcher/common/specdec/q30t_row_observation_runtime.py`
- Create: `tools/launcher/tests/test_q30t_row_observation_runtime.py`
- Modify: `examples/dataset/observe_q30t_ptv23_row_schemas.py`
- Modify: `tests/examples/dataset/test_observe_q30t_ptv23_row_schemas.py`

**Interfaces:**

- `RowObservationRuntimeEvidence`: deterministic runtime evidence embedded in the scientific observation.
- `load_row_observation_runtime_evidence(path: Path, *, expected_sha256: str) -> RowObservationRuntimeEvidence`.
- Extend `StageInputs` with `runtime_evidence_path: Path` and `runtime_evidence_sha256: str`.
- Keep `observe_stage_schemas(inputs: StageInputs) -> bytes` as the public scanner.
- Advance schema to `q30t-ptv23-row-schema-observation-v2`; reject host-runtime v1.

- [ ] **Step 1: Add RED canonical-loader tests**

Require one canonical newline-terminated record no larger than 1 MiB with exact qualification-file/embedded, profile, archive-tree, archive, base-image, derived-image-receipt, derived-image, runtime-tree, qualification-producer, approved-tool, Python, and PyArrow fields plus self-hash. Forbid job ID, node, row producer commit, scratch root, absolute container path, keeper hash, and container evidence.

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest -q -p no:cacheprovider \
  tools/launcher/tests/test_q30t_row_observation_runtime.py \
  -k 'runtime_evidence'
```

Expected RED: module is absent.

- [ ] **Step 2: Implement the sole stable loader**

Use descriptor-stable bounded reads, exact key sets, duplicate-key rejection, physical SHA, canonical bytes, and self-hash replay. Store Python as `bin/python` and PyArrow origin relative to `/opt/q30t-runtime`; reject absolute paths and controls.

- [ ] **Step 3: Add observer RED tests**

Require `StageInputs` runtime path/hash, v2 schema, lazy PyArrow import only after loader success, and exact inclusion of the loader's serialized runtime object. Two fixtures with different job IDs, nodes, scratch paths, and container names must yield identical observation bytes because those values never enter runtime evidence.

- [ ] **Step 4: Remove host runtime discovery and implement GREEN**

Delete host `sysconfig` site discovery and duplicated runtime parsing. Import PyArrow lazily after the shared loader succeeds and verify its version and runtime-relative origin against the record. Return canonical bytes to the held bootstrap; Task 4 owns stdout capture and final publication ordering.

- [ ] **Step 5: Run focused and static gates**

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest -q -p no:cacheprovider \
  tools/launcher/tests/test_q30t_row_observation_runtime.py \
  tests/examples/dataset/test_observe_q30t_ptv23_row_schemas.py
ruff format --check tools/launcher/common/specdec/q30t_row_observation_runtime.py \
  tools/launcher/tests/test_q30t_row_observation_runtime.py \
  examples/dataset/observe_q30t_ptv23_row_schemas.py \
  tests/examples/dataset/test_observe_q30t_ptv23_row_schemas.py
ruff check tools/launcher/common/specdec/q30t_row_observation_runtime.py \
  tools/launcher/tests/test_q30t_row_observation_runtime.py \
  examples/dataset/observe_q30t_ptv23_row_schemas.py \
  tests/examples/dataset/test_observe_q30t_ptv23_row_schemas.py
pyright tools/launcher/common/specdec/q30t_row_observation_runtime.py \
  tools/launcher/tests/test_q30t_row_observation_runtime.py \
  examples/dataset/observe_q30t_ptv23_row_schemas.py \
  tests/examples/dataset/test_observe_q30t_ptv23_row_schemas.py
git diff --check
```

- [ ] **Step 6: Review and signed commit**

Review exact four-file scope for pre-auth imports, duplicated schema, nondeterministic paths, runtime substitutions, bounded reads, and staged-publication behavior. Fix each valid finding with RED-GREEN, then commit only these files signed+DCO.

---

### Task 2: Freeze and Import the Repaired Runtime Handoff

**Files:** read-only runtime handoff, then non-destructive integration

- [ ] **Step 1: Require a signed clean handoff**

Record runtime base/head, every changed file SHA-256, commands/results, and fresh `0 critical / 0 warnings / 0 nits`. Require source scans proving no mutable runtime extraction, mutable-checkout `PYTHONPATH`, `SLURM_EXPORT_ENV=ALL`, or pathname deletion remains in the derived-image producer, keeper, probe, and submitter.

- [ ] **Step 2: Require exact public loaders**

The handoff provides safe typed loaders for archive-tree receipt, derived-image receipt, profile, node receipt, and aggregate qualification receipt. The derived-image receipt binds base image, archive, runtime tree, final SQSH, producer commit, normalized SquashFS metadata, builder executable/version/argv, two isolated build hashes, and the one retained artifact hash.

- [ ] **Step 3: Integrate reviewed runtime commits**

Cherry-pick or merge non-destructively; verify every signature/DCO and exact scope. Rerun repaired runtime suites in the integration worktree. Require behavioral evidence that same-UID write-open against the anonymous leased image causes prompt cancellation and no qualification receipt.

---

### Task 3: Authenticate Qualification Lineage and Operation Receipts

**Files:**

- Modify: `tools/launcher/common/specdec/q30t_row_observation_runtime.py`
- Modify: `tools/launcher/tests/test_q30t_row_observation_runtime.py`

**Interfaces:**

- `RowObservationRuntimeInputs`: paths and physical SHA-256s for qualification, profile, archive-tree receipt, derived-image receipt, and PTV3 completion, plus current row source commit and exact row tool hashes.
- `RowObservationRuntimeContext`: fully authenticated common lineage and row-producer identities.
- `RowObservationOperationReceipt`: job-specific evidence binding the durable observation physical SHA-256.
- `authenticate_row_observation_runtime(inputs: RowObservationRuntimeInputs) -> RowObservationRuntimeContext`.
- `build_row_observation_runtime_evidence(context: RowObservationRuntimeContext, *, python_version: str, pyarrow_version: str, python_relative_path: str, pyarrow_relative_path: str, pyarrow_tree_sha256: str) -> bytes`.
- `build_row_observation_operation_receipt(context: RowObservationRuntimeContext, *, job_id: str, node_name: str, keeper_receipt_file_sha256: str, container_evidence_sha256: str, observation_file_sha256: str) -> bytes`.
- `load_row_observation_operation_receipt(path: Path, *, expected_sha256: str) -> RowObservationOperationReceipt`.

- [ ] **Step 1: Add approval-root RED tests**

Require qualification, derived-image receipt, and PTV3 completion physical SHA values in their three immutable approval sets. All sets are empty in producer P. Caller-selected self-consistent receipts fail before keeper/container work.

- [ ] **Step 2: Implement exact cross-replay**

Use frozen public loaders for qualification, profile, archive tree, and derived image. Require two-node phase, exact prerequisite, node count, reuse/keeper-loss/lease-break proofs, cluster, archive/base/derived-image/tree identities, producer commit, deterministic double-build evidence, and every tool hash. Never reparse their JSON. Authenticate the fixed PTV3 completion through its approved physical SHA.

- [ ] **Step 3: Bind producer ancestry without weakening bytes**

Preserve `qualification_source_commit` in deterministic evidence. Require it to be an ancestor of `row_source_commit` and rehash every qualification-bound Git blob at both commits to prove byte identity. Put `row_source_commit` only in the operation receipt, never the observation.

- [ ] **Step 4: Add canonical operation receipt tests**

Two receipts with different job/node/keeper/container evidence may differ but must bind the same observation SHA and deterministic runtime tuple. Cover symlink, hardlink, FIFO, oversize, same-size mutation, path/root rebind, changed prerequisite/profile/tool/completion, and unsafe adoption.

- [ ] **Step 5: Run GREEN, static gates, review, and commit**

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest -q -p no:cacheprovider \
  tools/launcher/tests/test_q30t_row_observation_runtime.py \
  tools/launcher/tests/test_ptv23_runtime_attestation.py
ruff format --check tools/launcher/common/specdec/q30t_row_observation_runtime.py \
  tools/launcher/tests/test_q30t_row_observation_runtime.py
ruff check tools/launcher/common/specdec/q30t_row_observation_runtime.py \
  tools/launcher/tests/test_q30t_row_observation_runtime.py
pyright tools/launcher/common/specdec/q30t_row_observation_runtime.py \
  tools/launcher/tests/test_q30t_row_observation_runtime.py
git diff --check
```

Request independent review, fix RED-GREEN, and commit the exact two-file delta signed+DCO.

---

### Task 4: Rewrite the Runner Around the Derived Image

**Files:**

- Create: `tools/launcher/common/specdec/q30t_row_observation_bootstrap.py`
- Modify: `tools/launcher/common/specdec/run_q30t_row_schema_observation.sbatch`
- Modify: `tools/launcher/tests/test_q30t_row_schema_observation_runner.py`

**Positional contract:**

```text
SOURCE_PATH SOURCE_COMMIT OUTPUT OPERATION_RECEIPT
QUALIFICATION_RECEIPT QUALIFICATION_RECEIPT_FILE_SHA256
PROFILE PROFILE_FILE_SHA256
ARCHIVE_TREE_RECEIPT ARCHIVE_TREE_RECEIPT_FILE_SHA256
DERIVED_IMAGE_RECEIPT DERIVED_IMAGE_RECEIPT_FILE_SHA256
DERIVED_IMAGE DERIVED_IMAGE_SHA256
```

- [ ] **Step 1: Add an executable fake-Slurm RED harness**

Fake `srun` executes embedded keeper/container/reuse/reconciliation commands, not only argv. Record phase, PID/PGID, exit/watchdog, mounts, container image/name, sterile environment, and receipt paths. Require one exact successful trace.

- [ ] **Step 2: Add bounded hostile RED cases**

Inject receipt/image/tool substitution; keeper death during readiness/import/scan/staging/reconciliation/publication/shutdown; FIFO child; named reuse mismatch; wrong mount; observer failure; foreign output; partial write; operation-publication failure; and result streams at `N`, `N+1`, partial, empty, duplicate-key, trailing-byte, and noncanonical boundaries. Every case is bounded and obeys the state machine below.

- [ ] **Step 3: Preserve immutable source and scratch boundaries**

Keep reviewed commit-blob materialization, descriptor-relative no-follow/O_EXCL handling, safe Ptyche `/raid/scratch` policy, and held descriptor execution. Remove host PyArrow discovery and every mutable archive extraction.

- [ ] **Step 4: Authenticate runtime before keeper launch**

Call `authenticate_row_observation_runtime` with all positional receipts/image values. Build an exact `KeeperPlan` for derived image, observer, row module, attestation closure, atomic publisher, receipts, and held bootstrap. Validate that the image item was reopened read-only while its anonymous writer still existed, after which every writer was closed and an `F_RDLCK` lease was acquired. Concurrent same-UID write-open or lease-break notification disables reuse, kills and fully reaps the container/process group within less than half the observed kernel lease-break timeout, and suppresses every operation receipt before the writer can proceed.

- [ ] **Step 5: Launch the named derived-image container**

Use the keeper derived-image descriptor target directly as `--container-image`; never resolve it through a replaceable scratch symlink. Mount only keeper-backed controls plus dataset inputs read-only. Do not mount a mutable checkout or extracted runtime. Invoke the keeper-backed committed `q30t_row_observation_bootstrap.py` with `/opt/q30t-runtime/bin/python -I -S` and the exact CLI below. It opens/retains/hashes the staged row-runtime and observer blobs with `O_NOFOLLOW`, loads them with explicit `importlib` module names, never adds the control directory to `sys.path`, inserts exactly `/opt/q30t-runtime/lib/python3.12/site-packages`, replays the installed tree, and records runtime-relative Python/PyArrow paths.

```text
q30t_row_observation_bootstrap.py observe
  --row-runtime PATH --row-runtime-sha256 SHA256
  --observer PATH --observer-sha256 SHA256
  --runtime-evidence PATH --runtime-evidence-sha256 SHA256
  --ptv2-plan PATH --ptv2-plan-sha256 SHA256
  --ptv2-completion PATH --ptv2-completion-sha256 SHA256
  --ptv2-root PATH
  --ptv3-plan PATH --ptv3-plan-sha256 SHA256
  --ptv3-completion PATH --ptv3-completion-sha256 SHA256
  --ptv3-root PATH
```

The bootstrap returns no output pathname. On success stdout contains exactly one canonical newline-terminated `q30t-row-observation-container-result-v1` record with base64 observation bytes, observation SHA-256, deterministic runtime-evidence SHA-256, container evidence, and self-hash; diagnostics use stderr. `_MAX_CONTAINER_RESULT_BYTES` is exactly `64 * 1024 * 1024`. A host-side bounded pump streams stdout into its already held anonymous `O_TMPFILE`, cancels and fully reaps the srun process group immediately upon byte `N+1`, then replays the envelope. Tests cover exactly `N`, `N+1`, partial, empty, duplicate-key, trailing-byte, and noncanonical streams. No container-created pathname is trusted.

- [ ] **Step 6: Freeze exact dataset inputs in the immutable runner**

Use no caller dataset arguments. Require PTV2 root `/lustre/fsw/coreai_dlalgo_llm/users/sna/modelopt-qwen3-drafter-training/assets/q30t-ptv2-full201-source-v1`, plan SHA `96970541d0c6f5c99e74b9222b805d4a0bd2ac682837b0ad92b8bf54f7d71a3a`, completion SHA `018d659170834b17967dd6c1b066e3b03e7859386eceefd9d4409dc3bc8f48c1`; PTV3 root `/lustre/fsw/coreai_dlalgo_llm/users/sna/modelopt-qwen3-drafter-training/assets/q30t-ptv3-source-stage-75209087-v1`, plan SHA `752090878ed2c5fce47683b2939f9d857be5549311b158476fce33bb79f52eac`, completion path `completion.json`, and independently reviewed physical completion SHA `e29ab22c9ea85ddd1b7a7106974ce24e787fb121bc1f3282daa96deb30843b68`. Admission requires that SHA in `APPROVED_Q30T_PTV3_COMPLETION_RECEIPT_FILE_SHA256S`; live hashing alone is insufficient.

- [ ] **Step 7: Implement one publication state machine**

The only allowed sequence is:

```text
AUTHENTICATED
KEEPER_LIVE
CONTAINER_SCANNED
STAGED_PAYLOAD_HELD
RECONCILED
OBSERVATION_DURABLE
KEEPER_STOPPED
OPERATION_RECEIPT_DURABLE
```

The bootstrap writes its bounded canonical result only to stdout, which the host redirects into an anonymous `O_TMPFILE` it already holds. The host replays that descriptor, performs named-container reuse and reconciliation while the keeper remains live, then publishes/adopts the observation decoded from that descriptor. It rechecks keeper liveness, stops/reaps the keeper boundedly, and publishes the operation receipt last. Failure before `OBSERVATION_DURABLE` leaves no final observation. Failure afterward retains an immutable unauthenticated observation and no operation receipt. Only `OPERATION_RECEIPT_DURABLE` is success.

- [ ] **Step 8: Run focused and static gates**

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest -q -p no:cacheprovider \
  tools/launcher/tests/test_q30t_row_schema_observation_runner.py
bash -n tools/launcher/common/specdec/run_q30t_row_schema_observation.sbatch
shellcheck tools/launcher/common/specdec/run_q30t_row_schema_observation.sbatch
ruff format --check tools/launcher/common/specdec/q30t_row_observation_bootstrap.py \
  tools/launcher/tests/test_q30t_row_schema_observation_runner.py
ruff check tools/launcher/common/specdec/q30t_row_observation_bootstrap.py \
  tools/launcher/tests/test_q30t_row_schema_observation_runner.py
pyright tools/launcher/common/specdec/q30t_row_observation_bootstrap.py \
  tools/launcher/tests/test_q30t_row_schema_observation_runner.py
python -m py_compile tools/launcher/common/specdec/q30t_row_observation_bootstrap.py \
  tools/launcher/tests/test_q30t_row_schema_observation_runner.py
git diff --check
```

- [ ] **Step 9: Review and signed commit**

Review same-UID substitution, environment isolation, mounted anchors, process groups, keeper liveness, exact code execution, bootstrap envelope, state transitions, no-clobber behavior, and fake-harness fidelity. Fix RED-GREEN and commit exactly bootstrap/runner/test signed+DCO.

---

### Task 5: Extend the Test-Only-First Submitter

**Files:**

- Modify: `tools/launcher/common/specdec/submit_q30t_row_schema_observation.sh`
- Modify: `tools/launcher/tests/test_q30t_row_schema_observation_runner.py`

- [ ] **Step 1: Add RED CLI propagation tests**

Require every positional receipt/image value from Task 4 and prove exact propagation through both `sbatch --test-only` and `sbatch --parsable`. Reject duplicates, missing values, controls, relative/symlink paths, wrong hashes, and nonmatching observation/operation variants.

- [ ] **Step 2: Freeze approved absolute output pairs**

Only accept these pairs beneath `/lustre/fsw/coreai_dlalgo_llm/users/sna/modelopt-qwen3-drafter-training/receipts/q30t-ptv23-complement-700k-v1/row-schema/`: `observation-a.json` with `observation-a-operation.json`, or `observation-b.json` with `observation-b-operation.json`. Logs use the existing approved absolute log root.

- [ ] **Step 3: Preserve source provenance and sterile execution**

Require clean HEAD, approved GitLab remote, exact live remote tip, no replacement objects/attributes/hooks, exact spooled runner, privileged Bash, and absolute sterile Git/SSH/sbatch. Add no environment test hooks.

- [ ] **Step 4: Run gates, review, and commit**

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest -q -p no:cacheprovider \
  tools/launcher/tests/test_q30t_row_schema_observation_runner.py \
  -k 'submitter or source or sbatch'
bash -n tools/launcher/common/specdec/submit_q30t_row_schema_observation.sh
shellcheck tools/launcher/common/specdec/submit_q30t_row_schema_observation.sh
git diff --check
```

Request independent review, fix RED-GREEN, and commit exact submitter/test scope signed+DCO.

---

### Task 6: Run the Integrated Local Suite

- [ ] **Step 1: Use a private safe basetemp**

```bash
Q30T_BASETEMP="$(mktemp -d /Users/sna/q30t-row-qualified.XXXXXXXX)"
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest -q -p no:cacheprovider \
  --basetemp "$Q30T_BASETEMP" \
  tests/examples/dataset/test_observe_q30t_ptv23_row_schemas.py \
  tools/launcher/tests/test_q30t_row_observation_runtime.py \
  tools/launcher/tests/test_q30t_row_schema_observation_runner.py \
  tools/launcher/tests/test_ptv23_runtime_attestation.py \
  tools/launcher/tests/test_q30t_runtime_probe_runner.py
```

Do not use `/tmp`; scratch-policy tests intentionally reject world-writable ancestors.

- [ ] **Step 2: Run static and provenance gates**

Run Ruff format/check, scoped Pyright, `py_compile`, `bash -n`, ShellCheck, `git diff --check`, exact scope, commit signatures/DCO, and scans for forbidden mutable extraction, checkout import, `SLURM_EXPORT_ENV=ALL`, and deletion.

- [ ] **Step 3: Freeze a review package**

Record base/head, file SHA-256s, commands/results, expected platform skips, spec link, and every pending external transition.

---

### Task 7: Obtain a Fresh Independent Hostile Review

- [ ] **Step 1: Review the exact frozen range**

Inspect production and tests for receipt approval, immutable derived-image lineage, bootstrap imports, keeper/Pyxis mounts, state machine, A/B determinism, and no-foreign-mutation semantics.

- [ ] **Step 2: Resolve findings with RED-GREEN**

No warning is accepted merely because remote validation is planned. Re-freeze after every fix and require final rereview.

- [ ] **Step 3: Require CLEAN**

Proceed only with `0 critical, 0 warnings, 0 nits`, fresh passing gates, matching hashes, preserved unrelated changes, and good signatures/DCO.

---

### Task 8: Push Producer P, Approve Its Runtime, and Run A/B

- [ ] **Step 1: Push exact producer commit P**

Push normally and verify live GitLab tip equals P. Do not force push.

- [ ] **Step 2: Produce and review runtime evidence from P**

Run derived-image test-only and real production, independent image-receipt review, one-node test-only/real probe and review, then two-node test-only/real probe and review. Replay aggregate, profile, archive-tree, and derived-image loaders together. Independently stable-read/replay the fixed PTV3 `completion.json`, require physical SHA `e29ab22c9ea85ddd1b7a7106974ce24e787fb121bc1f3282daa96deb30843b68`, and reconcile its PTV3 plan binding before approval A. No receipt approves itself.

- [ ] **Step 3: Create approval-only descendant A**

Insert only the reviewed qualification, derived-image receipt, and PTV3 completion physical SHA-256s into the three approval sets. Generic empty-root/known-root/unknown-root transition tests already land in P; A adds no test or behavior changes. Run the full local suite, obtain an independent constants-only review, commit signed+DCO, push normally, prove P is an ancestor of A, and prove every qualification-bound Git blob has identical SHA at P and A.

- [ ] **Step 4: Submit observation A test-only then real**

Use commit A and exact approved inputs. Output is `/lustre/fsw/coreai_dlalgo_llm/users/sna/modelopt-qwen3-drafter-training/receipts/q30t-ptv23-complement-700k-v1/row-schema/observation-a.json`; operation receipt is its approved `observation-a-operation.json` sibling.

- [ ] **Step 5: Monitor A for at least five minutes**

Use one filtered `squeue -j "$JOB_ID"` query per minute and bounded `sacct -j "$JOB_ID"`/log reads. On failure, collect evidence and return to RED-GREEN/review/push; never patch remote code.

- [ ] **Step 6: Submit and monitor B identically**

Use the same A commit and runtime inputs with absolute `observation-b.json` and `observation-b-operation.json` siblings.

- [ ] **Step 7: Independently replay A/B**

Require canonical/self hashes, 201 PTV2 files, 100 PTV3 files, 670,399 PTV3 rows, matching sources, matching deterministic runtime, and byte-identical observation files. Require two valid operation receipts with distinct job identities but the same observation SHA/runtime tuple.

---

### Task 9: Approve and Hand Off to Data Task 4B

- [ ] **Step 1: Independently review physical evidence**

Review stable single-link bytes, physical/self hashes, source bindings/counts, qualification tuple, operation receipts, and A/B identity.

- [ ] **Step 2: Record the common approved observation SHA**

Record common SHA, both paths/job IDs, producer P, approval A, and exact runtime receipt roots. Do not hand-author scientific mapping fields.

- [ ] **Step 3: Start Data Task 4B**

Use reviewed observation evidence to author source mapping/inventory. Final 700K build remains blocked until authoritative `speed`, `math`, `code`, `swe`, and `tool` held-out UUID sources are independently available and approved.

## Completion Criteria

- Derived runtime image and qualification are independently reviewed and approved.
- Local row implementation and final review are CLEAN.
- Ptyche A/B complete from approval commit A with one exact runtime tuple.
- A/B observations are byte-identical and replay to 201 PTV2 files, 100 PTV3 files, and 670,399 PTV3 rows.
- Two valid operation receipts bind the common observation SHA, which is independently approved and handed to Data Task 4B.
- No training claim is made until remaining data/controller/held-out/canary gates are complete.
