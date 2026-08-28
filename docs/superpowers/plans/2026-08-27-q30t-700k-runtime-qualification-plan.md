# Q30 Thinking 700K Runtime Qualification Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Qualify the existing Q30 Thinking runtime archive and SQSH on Ptyche through independently reviewed archive/tree, one-node, and two-node keeper/Pyxis receipts, then require that qualification before any 16-node canary can be submitted.

**Architecture:** A shared runtime-tree identity module gives the archive producer and Q30 controller one canonical digest implementation. A dedicated attestation module emits strict canonical per-node and aggregate receipts, while a one-/two-node Slurm probe proves descriptor-backed Pyxis import, named-container reuse, and keeper-loss failure on the actual Ptyche platform. External profile and submitter code run test-only first; the production contract admits runtime bytes only after independent receipt replay and explicit immutable approval insertion.

**Tech Stack:** Python 3.12+, Ptyche `/usr/bin/python3.12`, Bash, SLURM, Pyxis, enroot, GNU tar/zstd, `/raid/scratch`, canonical JSON/SHA-256 receipts, pytest, Ruff, Pyright, ShellCheck

**Spec:** `docs/superpowers/specs/2026-08-27-q30t-swe-heavy-700k-evidence-launch-design.md`

## Global Constraints

- Reuse archive SHA `4a20aee61f290c48bed22a84b4a0ae0cbdc54e3e3910854d253188c8854f5dc9`; do not rebuild it.
- Reuse image SHA `e7be53f2754097c88f7c801da92f6d94794ec4d78d9df937fcd315a6994297f0`; do not rebuild it.
- The archive path is `/lustre/fsw/coreai_dlalgo_llm/users/sna/modelopt-qwen3-drafter-training/assets/q235-training-prereqs-vllm0271-v1/runtime/modelopt-vllm-0.27.1-py312-aarch64-symlinks.tar.zst`.
- The image path is `/lustre/fsw/coreai_dlalgo_llm/users/sna/modelopt-qwen3-drafter-training/assets/q235-training-prereqs-vllm0271-v1/image/vllm_openai_v0271_aarch64_20260813_2688476.sqsh`.
- The Ptyche account is `coreai_dlalgo_llm`, the primary partition is `36x2-a01r`, and every allocated node exposes exactly four GPUs.
- Every Ptyche host-side producer, profile, attestation, and replay command invokes approved `/usr/bin/python3.12` and rejects a runtime below Python 3.12 before reading qualification inputs.
- Durable qualification evidence lives under `/lustre/fsw/coreai_dlalgo_llm/users/sna/modelopt-qwen3-drafter-training/receipts/q30t-swe-heavy-700k-v1/runtime-qualification`; high-churn staging lives under `/raid/scratch/$USER`.
- Source code and the Git checkout stay under `/home`; no source checkout, build, cache, database, or lock tree is created on Lustre.
- Every producer and probe uses a clean pushed commit, `sbatch --test-only` before real submission, bounded logs, and at least five one-query-per-minute monitoring checks.
- Archive/tree production, one-node qualification, two-node qualification, independent replay, approval insertion, and 16-node consumer execution are distinct transitions.
- A produced receipt never approves itself. All approval sets and mappings remain empty until a fresh independent hostile review reports the exact whole-file SHA.
- Existing destinations are adopted only when exact canonical bytes and full metadata match; foreign existing paths are preserved and rejected.
- Keeper death cancels the active probe or reconciliation step and suppresses its final receipt.
- All new Python interfaces are fully typed and pass scoped Pyright; all canonical receipt loaders reject missing, extra, duplicate, reordered, non-canonical, or self-hash-mismatched evidence.
- Runtime Tasks 1-6 own only their new runtime-specific modules, scripts, and focused tests. They may run in parallel with the Data and Controller plans wherever those plans do not own the same file.
- Until Controller Task 9 in `docs/superpowers/plans/2026-08-27-q30t-700k-controller-trust-plan.md` freezes and commits the controller slice, the Controller plan exclusively owns `q30t_ptv23_continuation.py`, `run_q30t_ptv23_continuation.sbatch`, `submit_q30t_ptv23_continuation.sh`, and their controller tests.
- Runtime Task 7 starts only after an explicit Controller Task 9 handoff naming the frozen controller commit and hashes. From that handoff through Runtime Task 7 review/commit, this plan owns only the qualification integration changes in those handed-off controller/runner/submitter files; it must preserve all frozen controller-trust behavior.

---

## Cross-Plan Ownership and Sequencing

| Phase | Runtime work | File ownership | Parallelism / gate |
|---|---|---|---|
| 1 | Tasks 1-6 | New runtime-only modules, scripts, and runtime-only tests | May run in parallel with Data and Controller implementation because no controller-owned file is touched |
| 2 | Controller Task 9 freeze | Controller plan retains `q30t_ptv23_continuation.py`, runner, training submitter, and focused controller tests | Runtime Task 7 is blocked until the signed controller freeze commit and exact file hashes are handed off |
| 3 | Runtime Task 7 | Only the qualification integration delta in the handed-off controller, runner, submitter, and tests | Runtime plan owns these files for this slice; preserve every frozen controller-trust invariant |
| 4 | Runtime Task 8 | Integrated verification only | Requires Tasks 1-7 and the recorded controller handoff to be clean |
| 5 | Runtime Tasks 9-11 | External Ptyche receipts/reviews, then one approval-only source transition | Receipt jobs use one frozen pushed producer commit; no controller code changes occur between archive, one-node, and two-node probes |

## File Responsibility Map

- `tools/launcher/common/specdec/q30t_runtime_identity.py`: shared canonical runtime-tree entry enumeration and SHA-256 implementation.
- `tools/launcher/common/specdec/q30t_runtime_archive_receipt.py`: archive/tree receipt model, producer, loader, replay, and CLI.
- `tools/launcher/common/specdec/run_q30t_runtime_archive_receipt.sbatch`: CPU-only Ptyche archive extraction producer.
- `tools/launcher/common/specdec/ptv23_runtime_attestation.py`: per-node and aggregate runtime qualification models, attester, reconciliation, replay, and CLI.
- `tools/launcher/common/specdec/q30t_ptv23_cluster_profile.py`: canonical external Ptyche runtime-qualification profile model, finalizer, loader, and replay.
- `tools/launcher/common/specdec/probe_ptv23_pyxis_keeper.sbatch`: allocation-side one-/two-node keeper, Pyxis, reuse, and keeper-loss probe.
- `tools/launcher/common/specdec/submit_q30t_runtime_qualification.sh`: clean-pushed-source, test-only-first, idempotent archive/one-node/two-node submitter.
- `tools/launcher/common/specdec/q30t_ptv23_continuation.py`: Task 7 only, after Controller Task 9 handoff; production approval constants, shared runtime-tree alias, and contract authentication of profile and two-node qualification receipts.
- `tools/launcher/common/specdec/run_q30t_ptv23_continuation.sbatch`: Task 7 only, after Controller Task 9 handoff; replay the qualification-bound profile/runtime identities before starting keepers.
- `tools/launcher/common/specdec/submit_q30t_ptv23_continuation.sh`: Task 7 only, after Controller Task 9 handoff; require the qualified descriptor fields for canary and full submission.
- `tools/launcher/tests/test_q30t_runtime_archive_receipt.py`: archive extraction, tree identity, canonical receipt, publication, and attack tests.
- `tools/launcher/tests/test_ptv23_runtime_attestation.py`: node/aggregate receipt, exact reconciliation, canonical replay, and mutation tests.
- `tools/launcher/tests/test_q30t_runtime_probe_runner.py`: static and mocked behavioral probe-runner tests.
- `tools/launcher/tests/test_q30t_runtime_qualification_submitter.py`: Tasks 4 and 6 only; external profile and qualification-submitter tests with no controller-file ownership.
- `tools/launcher/tests/test_q30t_ptv23_submitter.py`: Task 7 only after controller handoff; qualification/training integration tests.
- `tools/launcher/tests/test_q30t_ptv23_continuation.py`: Task 7 only after controller handoff; production contract approval and descriptor round-trip tests.
- `tools/launcher/tests/test_q30t_ptv23_controller.py`: Task 7 only after controller handoff; allocation-side pre-keeper qualification replay and keeper-death tests.
- `.superpowers/sdd/2026-08-27-q30t-swe-heavy-700k/runtime-qualification-report.md`: exact test-only commands, job IDs, receipt hashes, independent review verdicts, and approval transition.

### Task 1: Create the Canonical Runtime-Tree Identity Module

**Files:**
- Create: `tools/launcher/common/specdec/q30t_runtime_identity.py`
- Create: `tools/launcher/tests/test_q30t_runtime_identity.py`

**Interfaces:**
- Consumes: an absolute no-follow runtime directory containing only directories, regular files, and symlinks.
- Produces: `RuntimeTreeIdentity`, `runtime_tree_entries(root: Path) -> tuple[RuntimeTreeEntry, ...]`, and `runtime_tree_identity(root: Path) -> RuntimeTreeIdentity`.
- Defers: importing this function into `q30t_ptv23_continuation.py` until Task 7 receives the frozen Controller Task 9 handoff.

- [ ] **Step 1: Add the RED equivalence and unsupported-entry tests**

```python
def test_shared_runtime_identity_matches_frozen_legacy_digest(tmp_path: Path) -> None:
    runtime = tmp_path / "runtime"
    (runtime / "bin").mkdir(parents=True)
    (runtime / "bin/python3.12").write_bytes(b"python")
    (runtime / "bin/python").symlink_to("python3.12")
    observed = runtime_tree_identity(runtime)
    assert observed.sha256 == "3992ed625fc893f39e1a30148e9d9df90ab23cbe2bb0a4ed97d15842267ae673"
    assert observed.file_count == 1
    assert observed.symlink_count == 1
    assert observed.total_regular_bytes == 6


def test_runtime_identity_rejects_fifo(tmp_path: Path) -> None:
    runtime = tmp_path / "runtime"
    runtime.mkdir()
    os.mkfifo(runtime / "controller.fifo")
    with pytest.raises(ValueError, match="unsupported entry"):
        runtime_tree_identity(runtime)
```

- [ ] **Step 2: Run the focused RED tests**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 ../q4-ptv2-ab-integration/.venv/bin/python -m pytest -q -p no:cacheprovider \
  tools/launcher/tests/test_q30t_runtime_identity.py
```

Expected: collection fails with `ModuleNotFoundError: No module named 'common.specdec.q30t_runtime_identity'`.

- [ ] **Step 3: Implement the typed shared identity records**

```python
@dataclass(frozen=True)
class RuntimeTreeEntry:
    path: str
    type: Literal["regular", "symlink"]
    size: int | None
    sha256: str | None
    target: str | None


@dataclass(frozen=True)
class RuntimeTreeIdentity:
    entries: tuple[RuntimeTreeEntry, ...]
    file_count: int
    symlink_count: int
    total_regular_bytes: int
    sha256: str


def runtime_tree_identity(root: Path) -> RuntimeTreeIdentity:
    entries = runtime_tree_entries(root)
    legacy = [
        [entry.path, "regular", entry.size, entry.sha256]
        if entry.type == "regular"
        else [entry.path, "symlink", entry.target]
        for entry in entries
    ]
    digest = hashlib.sha256(
        json.dumps(legacy, ensure_ascii=False, separators=(",", ":")).encode()
    ).hexdigest()
    return RuntimeTreeIdentity(
        entries=entries,
        file_count=sum(entry.type == "regular" for entry in entries),
        symlink_count=sum(entry.type == "symlink" for entry in entries),
        total_regular_bytes=sum(entry.size or 0 for entry in entries),
        sha256=digest,
    )
```

Use descriptor-stable regular-file hashing with `O_NOFOLLOW`, compare before/after/named identities, reject non-canonical relative paths, and reject devices, sockets, FIFOs, and hard-linked regular files.

- [ ] **Step 4: Add descriptor-rebind and symlink-target mutation tests**

```python
def test_runtime_identity_rejects_regular_file_rebind(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runtime = tmp_path / "runtime"
    runtime.mkdir()
    entry = runtime / "pyvenv.cfg"
    entry.write_text("home=/runtime\n")
    monkeypatch.setattr(module, "_POST_FILE_HASH_HOOK", lambda: entry.write_text("changed\n"))
    with pytest.raises(ValueError, match="changed while hashing"):
        runtime_tree_identity(runtime)


def test_runtime_identity_changes_when_symlink_target_changes(tmp_path: Path) -> None:
    runtime = tmp_path / "runtime"
    runtime.mkdir()
    link = runtime / "python"
    link.symlink_to("python3.12")
    before = runtime_tree_identity(runtime).sha256
    link.unlink()
    link.symlink_to("python3")
    assert runtime_tree_identity(runtime).sha256 != before
```

- [ ] **Step 5: Run the identity GREEN and mutation tests**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 ../q4-ptv2-ab-integration/.venv/bin/python -m pytest -q -p no:cacheprovider \
  tools/launcher/tests/test_q30t_runtime_identity.py
```

Expected: all tests pass and the frozen fixture digest remains byte-for-byte identical to the controller algorithm captured before either plan edits controller files.

- [ ] **Step 6: Run static checks for this slice**

```bash
../q4-ptv2-ab-integration/.venv/bin/ruff format --check \
  tools/launcher/common/specdec/q30t_runtime_identity.py \
  tools/launcher/tests/test_q30t_runtime_identity.py
../q4-ptv2-ab-integration/.venv/bin/ruff check \
  tools/launcher/common/specdec/q30t_runtime_identity.py \
  tools/launcher/tests/test_q30t_runtime_identity.py
PYTHONPATH=tools/launcher ../q4-ptv2-ab-integration/.venv/bin/pyright \
  tools/launcher/common/specdec/q30t_runtime_identity.py
PYTHONDONTWRITEBYTECODE=1 ../q4-ptv2-ab-integration/.venv/bin/python -m py_compile \
  tools/launcher/common/specdec/q30t_runtime_identity.py
git diff --check
```

Expected: every command exits zero.

- [ ] **Step 7: Request an independent hostile review of Task 1**

The reviewer must verify exact legacy digest compatibility, descriptor rebind detection, symlink-target hashing, hard-link rejection, unsupported-entry rejection, and absence of full-file buffering. Do not continue until the review reports no critical or warning findings.

- [ ] **Step 8: Commit the reviewed slice**

```bash
git add tools/launcher/common/specdec/q30t_runtime_identity.py \
  tools/launcher/tests/test_q30t_runtime_identity.py
git commit -S -s -m "feat(specdec): add Q30 runtime tree identity"
```

### Task 2: Produce the Canonical Archive-to-Tree Receipt

**Files:**
- Create: `tools/launcher/common/specdec/q30t_runtime_archive_receipt.py`
- Create: `tools/launcher/common/specdec/run_q30t_runtime_archive_receipt.sbatch`
- Create: `tools/launcher/tests/test_q30t_runtime_archive_receipt.py`

**Interfaces:**
- Consumes: `runtime_tree_identity`, the exact archive path/SHA, a fresh `/raid/scratch/$USER/q30t-runtime-archive-$SLURM_JOB_ID` root, source commit, producer path/SHA, and immutable output path.
- Produces: `RuntimeArchiveTreeReceipt`, `produce_runtime_archive_tree_receipt(*, archive_path: Path, expected_archive_sha256: str, extraction_root: Path, output_path: Path, source_commit: str, producer_path: Path, producer_sha256: str, job_id: str) -> RuntimeArchiveTreeReceipt`, `load_runtime_archive_tree_receipt(path: Path, expected_file_sha256: str) -> RuntimeArchiveTreeReceipt`, and CLI commands `produce` and `verify`.

- [ ] **Step 1: Write RED tests for the exact receipt schema**

```python
def test_archive_receipt_binds_archive_tree_and_producer(tmp_path: Path) -> None:
    archive = make_runtime_archive(tmp_path, {"bin/python": b"python", "pyvenv.cfg": b"home=x\n"})
    receipt = produce_runtime_archive_tree_receipt(
        archive_path=archive,
        expected_archive_sha256=file_sha256(archive),
        extraction_root=tmp_path / "scratch",
        output_path=tmp_path / "receipt.json",
        source_commit="2" * 40,
        producer_path=Path(__file__).resolve(),
        producer_sha256=file_sha256(Path(__file__).resolve()),
        job_id="unit-1",
    )
    assert receipt.schema_version == "q30t-runtime-archive-tree-v1"
    assert receipt.archive_sha256 == file_sha256(archive)
    assert receipt.runtime_tree_sha256 == runtime_tree_identity(tmp_path / "scratch/runtime").sha256
    assert receipt.sentinel_paths == ("bin/python", "pyvenv.cfg")
    assert load_runtime_archive_tree_receipt(tmp_path / "receipt.json", file_sha256(tmp_path / "receipt.json")) == receipt


def test_archive_receipt_rejects_rebound_archive(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    archive = make_runtime_archive(tmp_path, {"bin/python": b"python", "pyvenv.cfg": b"home=x\n"})
    monkeypatch.setattr(module, "_POST_HASH_HOOK", lambda: archive.write_bytes(b"rebound"))
    with pytest.raises(ValueError, match="changed while hashing"):
        produce_runtime_archive_tree_receipt(**exact_arguments(tmp_path, archive))
```

Define this concrete helper in the same test file:

```python
def exact_arguments(tmp_path: Path, archive: Path) -> dict[str, object]:
    producer = Path(__file__).resolve()
    return {
        "archive_path": archive,
        "expected_archive_sha256": file_sha256(archive),
        "extraction_root": tmp_path / "scratch",
        "output_path": tmp_path / "receipt.json",
        "source_commit": "2" * 40,
        "producer_path": producer,
        "producer_sha256": file_sha256(producer),
        "job_id": "unit-1",
    }
```

- [ ] **Step 2: Add RED publication and canonical replay tests**

Test that the loader rejects an extra key, reordered/non-canonical JSON, wrong self-hash, wrong whole-file SHA, non-single-link input, non-fresh extraction root, foreign output bytes, and a changed producer file. Test exact-byte adoption separately from foreign-path rejection.

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 ../q4-ptv2-ab-integration/.venv/bin/python -m pytest -q -p no:cacheprovider \
  tools/launcher/tests/test_q30t_runtime_archive_receipt.py
```

Expected: collection fails because `q30t_runtime_archive_receipt` does not exist.

- [ ] **Step 3: Implement the canonical receipt model**

```python
@dataclass(frozen=True)
class RuntimeArchiveTreeReceipt:
    schema_version: Literal["q30t-runtime-archive-tree-v1"]
    archive_path: str
    archive_size: int
    archive_sha256: str
    runtime_tree_sha256: str
    runtime_file_count: int
    runtime_symlink_count: int
    runtime_total_regular_bytes: int
    sentinel_paths: tuple[str, ...]
    sentinel_inventory_sha256: str
    source_commit: str
    producer_path: str
    producer_sha256: str
    receipt_sha256: str
```

The self-hash body contains every field except `receipt_sha256`. The canonical sentinel inventory is the exact sorted subset of runtime-tree entries for `bin/python`, `bin/activate`, and `pyvenv.cfg`; require `bin/python` and `pyvenv.cfg`, include `bin/activate` only when present, and reject any other sentinel key.

- [ ] **Step 4: Implement stable extraction and publication**

Stable-hash the archive through one no-follow descriptor, verify SHA `4a20aee61f290c48bed22a84b4a0ae0cbdc54e3e3910854d253188c8854f5dc9` at the caller boundary, create the exact absent extraction root with mode `0700`, run GNU tar with `--zstd --extract --no-same-owner --no-same-permissions`, compute `runtime_tree_identity`, and publish canonical bytes with `qwen4b_b_atomic.atomic_publish_bytes`. Do not unlink, replace, or recursively delete a foreign extraction or output tree.

```python
archive_size, archive_sha256 = stable_regular_file_sha256(archive_path)
if archive_sha256 != expected_archive_sha256:
    raise ValueError("runtime archive SHA-256 mismatch")
extraction_root.mkdir(mode=0o700, parents=False, exist_ok=False)
subprocess.run(
    (
        "/usr/bin/tar", "--zstd", "--extract", "--no-same-owner",
        "--no-same-permissions", "--file", str(archive_path),
        "--directory", str(extraction_root),
    ),
    check=True,
    env={"LC_ALL": "C", "PATH": "/usr/bin:/bin"},
)
identity = runtime_tree_identity(extraction_root)
atomic_publish_bytes(output_path, receipt.canonical_bytes() + b"\n", job_id=job_id)
```

- [ ] **Step 5: Implement the CPU-only sbatch producer**

The script must require seven positional arguments in this order: archive path, archive SHA, source checkout, source commit, producer SHA, output receipt, output receipt parent. It must enforce one node, no GPU flag, `--export=NONE`, sterile environment replay, `/raid/scratch/$USER/q30t-runtime-archive-$SLURM_JOB_ID`, and invoke:

```bash
env -i PATH=/usr/bin:/bin PYTHONPATH="$source_checkout/tools/launcher" \
  /usr/bin/python3.12 -m common.specdec.q30t_runtime_archive_receipt produce \
  --archive "$archive" \
  --archive-sha256 "$archive_sha256" \
  --extraction-root "$scratch_root/runtime" \
  --source-commit "$source_commit" \
  --producer "$source_checkout/tools/launcher/common/specdec/q30t_runtime_archive_receipt.py" \
  --producer-sha256 "$producer_sha256" \
  --output "$output_receipt" \
  --job-id "$SLURM_JOB_ID"
```

- [ ] **Step 6: Run GREEN and shell checks**

```bash
PYTHONDONTWRITEBYTECODE=1 ../q4-ptv2-ab-integration/.venv/bin/python -m pytest -q -p no:cacheprovider \
  tools/launcher/tests/test_q30t_runtime_archive_receipt.py
bash -n tools/launcher/common/specdec/run_q30t_runtime_archive_receipt.sbatch
shellcheck -x tools/launcher/common/specdec/run_q30t_runtime_archive_receipt.sbatch
```

Expected: all receipt tests pass; Bash and ShellCheck exit zero.

- [ ] **Step 7: Run typed/static checks and independent review**

```bash
../q4-ptv2-ab-integration/.venv/bin/ruff format --check \
  tools/launcher/common/specdec/q30t_runtime_archive_receipt.py \
  tools/launcher/tests/test_q30t_runtime_archive_receipt.py
../q4-ptv2-ab-integration/.venv/bin/ruff check \
  tools/launcher/common/specdec/q30t_runtime_archive_receipt.py \
  tools/launcher/tests/test_q30t_runtime_archive_receipt.py
PYTHONPATH=tools/launcher ../q4-ptv2-ab-integration/.venv/bin/pyright \
  tools/launcher/common/specdec/q30t_runtime_archive_receipt.py
PYTHONDONTWRITEBYTECODE=1 ../q4-ptv2-ab-integration/.venv/bin/python -m py_compile \
  tools/launcher/common/specdec/q30t_runtime_archive_receipt.py
git diff --check
```

The independent reviewer must replay canonical bytes and self-hash, inspect tar extraction confinement, inject descriptor/path rebinding, verify stable producer identity, and confirm that the output cannot approve itself.

- [ ] **Step 8: Commit the reviewed archive producer**

```bash
git add tools/launcher/common/specdec/q30t_runtime_archive_receipt.py \
  tools/launcher/common/specdec/run_q30t_runtime_archive_receipt.sbatch \
  tools/launcher/tests/test_q30t_runtime_archive_receipt.py
git commit -S -s -m "feat(specdec): produce Q30 runtime archive receipt"
```

### Task 3: Define Canonical Per-Node and Aggregate Attestation

**Files:**
- Create: `tools/launcher/common/specdec/ptv23_runtime_attestation.py`
- Create: `tools/launcher/tests/test_ptv23_runtime_attestation.py`

**Interfaces:**
- Consumes: reviewed archive/tree receipt bytes, exact profile bytes, keeper receipt, mounted SQSH path, extracted runtime path, source checkout, runner/tool paths and hashes, expected node set, and optional reviewed one-node aggregate receipt.
- Produces: `RuntimeNodeObservation`, `RuntimeNodeReuseEvidence`, `RuntimeKeeperLossEvidence`, `RuntimeNodeAttestationInput`, `RuntimeNodeAttestationReceipt`, `RuntimeQualificationContext`, `RuntimeQualificationReceipt`, `attest_runtime_node(inputs: RuntimeNodeAttestationInput, *, output_path: Path, publication_job_id: str) -> RuntimeNodeAttestationReceipt`, `reconcile_runtime_receipts(*, receipts: tuple[RuntimeNodeAttestationReceipt, ...], receipt_file_sha256s: tuple[str, ...], expected_nodes: tuple[str, ...], context: RuntimeQualificationContext, output_path: Path, publication_job_id: str) -> RuntimeQualificationReceipt`, `load_runtime_node_receipt(path: Path, expected_file_sha256: str) -> RuntimeNodeAttestationReceipt`, `load_runtime_qualification_receipt(path: Path, expected_file_sha256: str) -> RuntimeQualificationReceipt`, and CLI commands `attest-node`, `reconcile`, and `verify`.

- [ ] **Step 1: Write RED node-attestation tests**

```python
def test_node_attestation_binds_observed_runtime_and_keeper(tmp_path: Path) -> None:
    fixture = runtime_fixture(tmp_path, node_name="ptyche-n001")
    receipt = attest_runtime_node(
        fixture.inputs,
        output_path=tmp_path / "node.json",
        publication_job_id="unit-1",
    )
    assert receipt.schema_version == "q30t-runtime-node-attestation-v1"
    assert receipt.node_name == "ptyche-n001"
    assert receipt.observed_image_sha256 == fixture.image_sha256
    assert receipt.archive_sha256 == fixture.archive_sha256
    assert receipt.anchor_inode > 0
    assert receipt.visible_gpu_count == 4
    assert receipt.machine == "aarch64"
    assert receipt.container_name == "q30t-700k-unit-1"
    assert receipt.named_container_reused is True
    assert receipt.keeper_loss_fresh_use_failed is True
```

The fixture executes a fake runtime Python that emits exact JSON for executable, version, and import origins; fake `nvidia-smi`, Pyxis, and enroot version outputs are passed as already captured bounded strings. The production function validates values rather than invoking mutable shell commands internally.

- [ ] **Step 2: Write RED two-node reconciliation tests**

```python
def test_two_node_reconciliation_requires_reviewed_one_node_receipt(tmp_path: Path) -> None:
    one = make_aggregate(tmp_path, nodes=("ptyche-n001",), phase="one-node")
    materialized = (
        make_node(tmp_path, node="ptyche-n011", keeper_pid=101, anchor_inode=201),
        make_node(tmp_path, node="ptyche-n012", keeper_pid=102, anchor_inode=202),
    )
    result = reconcile_runtime_receipts(
        receipts=tuple(item.receipt for item in materialized),
        receipt_file_sha256s=tuple(item.file_sha256 for item in materialized),
        expected_nodes=("ptyche-n011", "ptyche-n012"),
        context=two_node_context(one),
        output_path=tmp_path / "two-node.json",
        publication_job_id="unit-2",
    )
    assert result.prerequisite_one_node_receipt_file_sha256 == one.file_sha256
    assert result.ordered_nodes == ("ptyche-n011", "ptyche-n012")
```

Add parameterized failures for missing/extra/duplicate nodes, duplicate keeper PID, duplicate `(anchor_device, anchor_inode)`, common archive/image/tree/profile/source/runner/tool mismatch, false reuse, false keeper-loss failure, wrong prerequisite one-node SHA, non-canonical node receipt, and mutated self-hash.

- [ ] **Step 3: Run the RED suite**

```bash
PYTHONDONTWRITEBYTECODE=1 ../q4-ptv2-ab-integration/.venv/bin/python -m pytest -q -p no:cacheprovider \
  tools/launcher/tests/test_ptv23_runtime_attestation.py
```

Expected: collection fails with `ModuleNotFoundError: No module named 'common.specdec.ptv23_runtime_attestation'`.

- [ ] **Step 4: Implement the exact node receipt schema**

Define `RuntimeNodeObservation` as canonical self-hashed `q30t-runtime-node-observation-v1` evidence carrying all node receipt fields below except the two proof booleans and final receipt SHA. Define `RuntimeNodeReuseEvidence` as canonical self-hashed `q30t-runtime-node-reuse-v1` evidence with exact job, phase, node, container name, container sentinel SHA, original observation SHA, and `named_container_reused: Literal[True]`. Define `RuntimeKeeperLossEvidence` as canonical self-hashed `q30t-runtime-node-keeper-loss-v1` evidence with exact job, phase, node, keeper PID/start ticks, anchor path/device/inode, bounded Pyxis error classification `anchor-unreadable`, and `keeper_loss_fresh_use_failed: Literal[True]`.

Define `RuntimeNodeAttestationInput` as a frozen dataclass carrying the expected common identities plus one replayed observation, reuse evidence, and keeper-loss evidence. `attest_runtime_node` requires all three job/phase/node/container/sentinel/keeper/anchor identities to match before it publishes a final receipt. Define `RuntimeQualificationContext` as a frozen dataclass carrying phase, cluster/profile, archive receipt, common runtime/source/tool identities, and the optional one-node prerequisite path/SHA. Tests construct both input dataclasses directly; the CLI constructs them only after replaying every referenced file.

```python
@dataclass(frozen=True)
class RuntimeNodeAttestationReceipt:
    schema_version: Literal["q30t-runtime-node-attestation-v1"]
    job_id: str
    phase: Literal["one-node", "two-node"]
    cluster: Literal["ptyche"]
    profile_file_sha256: str
    node_name: str
    keeper_pid: int
    keeper_start_ticks: int
    keeper_receipt_sha256: str
    anchor_path: str
    anchor_target: str
    anchor_device: int
    anchor_inode: int
    archive_sha256: str
    archive_tree_receipt_file_sha256: str
    runtime_tree_sha256: str
    expected_image_sha256: str
    observed_image_sha256: str
    sentinel_inventory_sha256: str
    source_commit: str
    contract_file_sha256: str
    runner_sha256: str
    keeper_tool_sha256: str
    attestation_tool_sha256: str
    machine: Literal["aarch64"]
    python_executable: str
    python_version: str
    python_import_origins: tuple[tuple[str, str], ...]
    cuda_visible_devices: str
    visible_gpu_identities: tuple[str, str, str, str]
    visible_gpu_count: Literal[4]
    pyxis_version: str
    enroot_version: str
    container_name: str
    container_root_device: int
    container_root_inode: int
    container_sentinel_sha256: str
    named_container_reused: Literal[True]
    keeper_loss_fresh_use_failed: Literal[True]
    receipt_sha256: str
```

Require imports for `accelerate`, `datasets`, `wandb`, and `modelopt`; require `modelopt.__file__` under the exact authenticated source checkout and the other imports under the extracted runtime or base image. Require exactly four unique non-empty GPU UUID/name records and bounded version strings without control characters.

- [ ] **Step 5: Implement the aggregate receipt schema and reconciliation**

```python
@dataclass(frozen=True)
class RuntimeQualificationReceipt:
    schema_version: Literal["q30t-runtime-qualification-v1"]
    job_id: str
    phase: Literal["one-node", "two-node"]
    cluster: Literal["ptyche"]
    profile_path: str
    profile_file_sha256: str
    archive_tree_receipt_path: str
    archive_tree_receipt_file_sha256: str
    archive_sha256: str
    image_sha256: str
    runtime_tree_sha256: str
    source_commit: str
    contract_file_sha256: str
    runner_sha256: str
    keeper_tool_sha256: str
    attestation_tool_sha256: str
    expected_node_count: Literal[1, 2]
    ordered_nodes: tuple[str, ...]
    ordered_node_receipt_file_sha256s: tuple[str, ...]
    ordered_node_receipts_sha256: str
    prerequisite_one_node_receipt_path: str | None
    prerequisite_one_node_receipt_file_sha256: str | None
    named_container_reuse_proven: Literal[True]
    keeper_loss_failure_proven: Literal[True]
    receipt_sha256: str
```

One-node receipts require both prerequisite fields to be `None`. Two-node receipts require both fields and replay the exact one-node receipt before reconciling current nodes. Sort by node name once; reject any input order that contains duplicate identities rather than silently deduplicating. Require common machine, Python version/executable/import origins, Pyxis version, enroot version, archive/image/tree/profile/source/tool identities, and container name across nodes. Require each node's four GPU identities to be internally unique and all eight two-node GPU identities to be globally unique. Require the reuse evidence to report the same `(container_root_device, container_root_inode)` as the initial observation on that node.

- [ ] **Step 6: Implement canonical loaders, CLI, and immutable publication**

Each loader stable-reads a single-link regular file, checks the caller SHA, exact key set, canonical newline-terminated JSON, lowercase fixed-width hashes, semantic invariants, and self-hash. `attest-node` writes one node receipt through `atomic_publish_bytes`; `reconcile` loads node receipts by exact explicit paths and publishes one aggregate; `verify` replays either schema without writing.

```python
def load_runtime_qualification_receipt(
    path: Path, expected_file_sha256: str
) -> RuntimeQualificationReceipt:
    raw, observed = stable_single_link_bytes(path, maximum_bytes=4 * 1024 * 1024)
    if observed != expected_file_sha256:
        raise AttestationError("qualification receipt whole-file SHA-256 mismatch")
    receipt = RuntimeQualificationReceipt.from_dict(json.loads(raw))
    if raw != receipt.canonical_bytes() + b"\n":
        raise AttestationError("qualification receipt is not canonical")
    receipt.verify_self_hash()
    return receipt
```

- [ ] **Step 7: Run GREEN, type, format, and compile gates**

```bash
PYTHONDONTWRITEBYTECODE=1 ../q4-ptv2-ab-integration/.venv/bin/python -m pytest -q -p no:cacheprovider \
  tools/launcher/tests/test_ptv23_runtime_attestation.py
../q4-ptv2-ab-integration/.venv/bin/ruff format --check \
  tools/launcher/common/specdec/ptv23_runtime_attestation.py \
  tools/launcher/tests/test_ptv23_runtime_attestation.py
../q4-ptv2-ab-integration/.venv/bin/ruff check \
  tools/launcher/common/specdec/ptv23_runtime_attestation.py \
  tools/launcher/tests/test_ptv23_runtime_attestation.py
PYTHONPATH=tools/launcher ../q4-ptv2-ab-integration/.venv/bin/pyright \
  tools/launcher/common/specdec/ptv23_runtime_attestation.py
PYTHONDONTWRITEBYTECODE=1 ../q4-ptv2-ab-integration/.venv/bin/python -m py_compile \
  tools/launcher/common/specdec/ptv23_runtime_attestation.py
git diff --check
```

Expected: every command exits zero.

- [ ] **Step 8: Independently review and commit the receipt layer**

The reviewer must attack canonical JSON, self-hash replay, field omission/addition, file/path rebinding, duplicate nodes/PIDs/inodes, one-node prerequisite substitution, common-identity mismatch, false positive booleans, and oversized metadata. Commit only after a clean verdict:

```bash
git add tools/launcher/common/specdec/ptv23_runtime_attestation.py \
  tools/launcher/tests/test_ptv23_runtime_attestation.py
git commit -S -s -m "feat(specdec): define Q30 runtime attestation receipts"
```

### Task 4: Finalize the External Ptyche Qualification Profile

**Files:**
- Create: `tools/launcher/common/specdec/q30t_ptv23_cluster_profile.py`
- Create: `tools/launcher/tests/test_q30t_runtime_qualification_submitter.py`

**Interfaces:**
- Consumes: clean source commit, exact source checkout, archive/image paths and SHAs, archive producer/keeper/attester/probe-runner paths and SHAs, and durable receipt root.
- Produces: `Q30TRuntimeClusterProfile`, `finalize_ptyche_runtime_profile(*, source_checkout: Path, source_commit: str, durable_receipt_root: Path, output_path: Path, job_id: str) -> Q30TRuntimeClusterProfile`, `load_ptyche_runtime_profile(path: Path, expected_file_sha256: str) -> Q30TRuntimeClusterProfile`, and CLI commands `finalize` and `verify`. The finalizer derives fixed asset paths/hashes and all source-tool paths from the authenticated checkout, then hashes the tool bytes itself.

- [ ] **Step 1: Write the RED exact-profile test**

```python
def test_ptyche_profile_binds_exact_runtime_and_topologies(tmp_path: Path) -> None:
    profile = finalize_ptyche_runtime_profile(**profile_arguments(tmp_path))
    assert profile.cluster == "ptyche"
    assert profile.account == "coreai_dlalgo_llm"
    assert profile.partition == "36x2-a01r"
    assert profile.probe_node_counts == (1, 2)
    assert profile.training_nodes == 16
    assert profile.training_segment == 16
    assert profile.gpus_per_node == 4
    assert profile.scratch_root == "/raid/scratch"
    assert profile.archive_sha256 == "4a20aee61f290c48bed22a84b4a0ae0cbdc54e3e3910854d253188c8854f5dc9"
    assert profile.image_sha256 == "e7be53f2754097c88f7c801da92f6d94794ec4d78d9df937fcd315a6994297f0"
```

Add RED cases for wrong account/partition, fallback partition use, wrong nodes/segment/GPU count, non-`/home` source, non-`/lustre` assets/receipts, non-`/raid/scratch` scratch, dirty or mismatched source commit, changed tool bytes, unknown profile key, and non-canonical receipt bytes.

- [ ] **Step 2: Run RED**

```bash
PYTHONDONTWRITEBYTECODE=1 ../q4-ptv2-ab-integration/.venv/bin/python -m pytest -q -p no:cacheprovider \
  tools/launcher/tests/test_q30t_runtime_qualification_submitter.py -k profile
```

Expected: collection fails because `q30t_ptv23_cluster_profile` does not exist.

- [ ] **Step 3: Implement the exact external profile schema**

```python
@dataclass(frozen=True)
class Q30TRuntimeClusterProfile:
    schema_version: Literal["q30t-runtime-cluster-profile-v1"]
    cluster: Literal["ptyche"]
    account: Literal["coreai_dlalgo_llm"]
    partition: Literal["36x2-a01r"]
    source_checkout: str
    source_commit: str
    durable_receipt_root: str
    scratch_root: Literal["/raid/scratch"]
    archive_path: str
    archive_sha256: str
    image_path: str
    image_sha256: str
    archive_producer_sha256: str
    keeper_tool_sha256: str
    attestation_tool_sha256: str
    probe_runner_sha256: str
    probe_node_counts: tuple[Literal[1], Literal[2]]
    training_nodes: Literal[16]
    training_segment: Literal[16]
    gpus_per_node: Literal[4]
    one_node_walltime: Literal["00:20:00"]
    two_node_walltime: Literal["00:30:00"]
    receipt_sha256: str
```

The output is external canonical JSON under the durable receipt root, not a source-tree profile. The finalizer verifies exact tool bytes and clean `HEAD == source_commit`; it does not assert approval.

- [ ] **Step 4: Run GREEN and static gates**

```bash
PYTHONDONTWRITEBYTECODE=1 ../q4-ptv2-ab-integration/.venv/bin/python -m pytest -q -p no:cacheprovider \
  tools/launcher/tests/test_q30t_runtime_qualification_submitter.py -k profile
../q4-ptv2-ab-integration/.venv/bin/ruff format --check \
  tools/launcher/common/specdec/q30t_ptv23_cluster_profile.py \
  tools/launcher/tests/test_q30t_runtime_qualification_submitter.py
../q4-ptv2-ab-integration/.venv/bin/ruff check \
  tools/launcher/common/specdec/q30t_ptv23_cluster_profile.py \
  tools/launcher/tests/test_q30t_runtime_qualification_submitter.py
PYTHONPATH=tools/launcher ../q4-ptv2-ab-integration/.venv/bin/pyright \
  tools/launcher/common/specdec/q30t_ptv23_cluster_profile.py
PYTHONDONTWRITEBYTECODE=1 ../q4-ptv2-ab-integration/.venv/bin/python -m py_compile \
  tools/launcher/common/specdec/q30t_ptv23_cluster_profile.py
git diff --check
```

Expected: every command exits zero.

- [ ] **Step 5: Independently review and commit the profile layer**

Review exact paths, topology, tool hash replay, source commit cleanliness, canonical publication, and the absence of any approval side effect. Then commit:

```bash
git add tools/launcher/common/specdec/q30t_ptv23_cluster_profile.py \
  tools/launcher/tests/test_q30t_runtime_qualification_submitter.py
git commit -S -s -m "feat(specdec): finalize Q30 Ptyche runtime profile"
```

### Task 5: Implement the One-/Two-Node Keeper and Pyxis Probe

**Files:**
- Create: `tools/launcher/common/specdec/probe_ptv23_pyxis_keeper.sbatch`
- Create: `tools/launcher/tests/test_q30t_runtime_probe_runner.py`
- Modify: `tools/launcher/tests/test_ptv23_node_keeper.py`

**Interfaces:**
- Consumes: exact profile path/SHA, archive-tree receipt path/SHA, optional one-node qualification path/SHA, output path, source checkout, and source commit.
- Produces: one canonical `q30t-runtime-node-attestation-v1` file per allocated node and one `q30t-runtime-qualification-v1` aggregate at the explicit output path.
- Uses: `ptv23_node_keeper.KeeperPlan`, `validate_keeper_receipt`, `ptv23_runtime_attestation attest-node/reconcile`, one keeper task per node, and one job-specific named container `q30t-700k-${SLURM_JOB_ID}` per node.

- [ ] **Step 1: Write the RED static boundary test**

```python
def test_probe_uses_one_local_anchor_for_import_and_mount() -> None:
    source = PROBE.read_text()
    assert '--container-image="${anchor_root}/runtime-image"' in source
    assert '${anchor_root}/runtime-image:/run/q30t/runtime.sqsh:ro' in source
    assert '--container-name="$container_name"' in source
    assert "srun --overlap" in source
    assert "validate_keeper_receipt" in source
    assert "attest-node" in source
    assert "reconcile" in source
    assert "--export=NONE" not in source
    assert "--container-name=\"$container_name\" --container-image" in source
    assert "--container-name=\"$container_name\" --container-mounts" in source
```

The last two assertions distinguish initial import from named reuse. The allocation script inherits only descriptor-replayed authenticated variables from its `--export=NONE` parent submission.

- [ ] **Step 2: Write mocked behavioral RED tests**

Implement a fake `srun` that assigns one or two distinct `SLURMD_NODENAME` values and records keeper, import, reuse, negative, and reconcile phases. Assert this exact trace for two nodes:

```python
assert trace == [
    "keeper:ptyche-n011",
    "keeper:ptyche-n012",
    "import:ptyche-n011",
    "import:ptyche-n012",
    "reuse:ptyche-n011",
    "reuse:ptyche-n012",
    "negative:ptyche-n011:failed",
    "negative:ptyche-n012:failed",
    "reconcile:ptyche-n011,ptyche-n012",
]
```

Add failure tests for keeper death before readiness, during import, during reuse, during negative proof, and during reconciliation. In every case assert nonzero exit and absent aggregate receipt.

- [ ] **Step 3: Run RED**

```bash
PYTHONDONTWRITEBYTECODE=1 ../q4-ptv2-ab-integration/.venv/bin/python -m pytest -q -p no:cacheprovider \
  tools/launcher/tests/test_q30t_runtime_probe_runner.py
```

Expected: failure because `probe_ptv23_pyxis_keeper.sbatch` is absent.

- [ ] **Step 4: Implement strict argument and phase validation**

The script accepts exactly:

```text
probe_ptv23_pyxis_keeper.sbatch MODE PROFILE PROFILE_SHA ARCHIVE_RECEIPT ARCHIVE_RECEIPT_SHA ONE_NODE_RECEIPT ONE_NODE_RECEIPT_SHA OUTPUT SOURCE_CHECKOUT SOURCE_COMMIT
```

`MODE` is `one-node` or `two-node`. One-node requires both prerequisite arguments to be the literal `NONE`; two-node requires canonical absolute paths and lowercase SHA-256. Require `SLURM_NNODES=1` or `2` to match mode, `SLURM_GPUS_ON_NODE=4`, exact source commit, fresh control/output paths, and a sterile descriptor replay before setting `SLURM_EXPORT_ENV=ALL` for nested steps.

```bash
[[ -x /usr/bin/python3.12 ]]
/usr/bin/python3.12 -I - <<'PY'
import sys
if sys.version_info < (3, 12):
    raise SystemExit("Q30 runtime qualification requires Python 3.12+")
PY
```

- [ ] **Step 5: Implement one keeper per node**

Each keeper stages exactly these single-link regular inputs through anonymous node-local files:

```python
sources = (
    KeeperSource("runtime-archive", profile.archive_path, profile.archive_sha256),
    KeeperSource("runtime-image", profile.image_path, profile.image_sha256),
    KeeperSource("qualification-profile", profile_path, profile_file_sha256),
    KeeperSource("archive-tree-receipt", archive_receipt_path, archive_receipt_file_sha256),
    KeeperSource("attestation-tool", attestation_tool, profile.attestation_tool_sha256),
    KeeperSource("keeper-tool", keeper_tool, profile.keeper_tool_sha256),
    KeeperSource("probe-runner", probe_runner, profile.probe_runner_sha256),
)
```

For two-node mode add the exact reviewed one-node receipt. Publish readiness only after `validate_keeper_receipt` succeeds; record keeper PID/start ticks plus `stat -Lc '%d %i %N'` evidence for the image anchor while the keeper remains alive.

- [ ] **Step 6: Implement import, attestation, and named reuse**

Start the creation step with the same image anchor in both positions:

```bash
srun --overlap --nodes="$expected_nodes" --ntasks="$expected_nodes" --ntasks-per-node=1 \
  --gpus-per-task=4 --kill-on-bad-exit=1 --no-container-mount-home \
  --container-name="$container_name" \
  --container-image="${anchor_root}/runtime-image" \
  --container-mounts="${anchor_root}/runtime-image:/run/q30t/runtime.sqsh:ro,${anchor_root}/runtime-archive:/run/q30t/runtime.tar.zst:ro,${control_root}:${control_root}" \
  /bin/bash -c "$attest_command"
```

The first in-container command hashes the complete `/run/q30t/runtime.sqsh`, extracts the archive into fresh node-local scratch, validates the archive/tree receipt and sentinel inventory, captures exact Python/import/GPU/Pyxis/enroot evidence, writes one bounded `q30t-runtime-node-observation-v1` file, and exits. It does not publish the final node receipt yet because reuse and keeper-loss evidence do not exist. A second `srun` uses only `--container-name="$container_name"` plus mounts, verifies the sentinel and observation, and writes `q30t-runtime-node-reuse-v1` evidence without reopening the image.

- [ ] **Step 7: Implement the keeper-loss negative phase**

After successful reuse, stop each keeper through its FIFO, confirm its PID/start identity is gone, and launch a fresh descriptor-backed use with a new container name `q30t-700k-negative-${SLURM_JOB_ID}`. The step must fail because the anchor is unreadable. Treat success, timeout, or an unrelated failure before anchor open as probe failure; only the expected unreadable-anchor classification writes `q30t-runtime-node-keeper-loss-v1` evidence. The verified host-side attestation tool then consumes the previously validated keeper snapshot, node observation, reuse evidence, and keeper-loss evidence and publishes the immutable final node receipt exactly once.

```bash
printf 'stop\n' >"$keeper_fifo"
wait "$keeper_pid"
[[ ! -r "${anchor_root}/runtime-image" ]] || {
  echo "keeper-loss image anchor remained readable" >&2
  exit 2
}
if srun --overlap --nodes="$expected_nodes" --ntasks="$expected_nodes" --ntasks-per-node=1 \
  --kill-on-bad-exit=1 --no-container-mount-home \
  --container-name="q30t-700k-negative-${SLURM_JOB_ID}" \
  --container-image="${anchor_root}/runtime-image" /bin/true \
  >"$negative_log" 2>&1; then
    echo "keeper-loss fresh descriptor use unexpectedly succeeded" >&2
    exit 2
fi
grep -F "runtime-image" "$negative_log" >/dev/null || {
  echo "keeper-loss failure did not classify the image anchor" >&2
  exit 2
}
```

- [ ] **Step 8: Reconcile exact node receipts**

One-node mode reconciles one explicit node receipt with no prerequisite. Two-node mode replays the exact one-node receipt and reconciles two unique Slurm nodes, PIDs, and `(device,inode)` pairs. The controller writes no aggregate if any keeper/probe child fails. Trap cleanup may stop owned processes and remove only the exact job-owned `/raid/scratch/$USER/q30t-runtime-probe-$SLURM_JOB_ID` tree; it must never remove durable output or foreign recovery paths.

```bash
env -i PATH=/usr/bin:/bin PYTHONPATH="$source_checkout/tools/launcher" \
  /usr/bin/python3.12 -m common.specdec.ptv23_runtime_attestation reconcile \
  --context "$control_root/reconciliation-context.json" \
  --node-receipt-list "$control_root/node-receipts.json" \
  --output "$output_receipt" \
  --job-id "$SLURM_JOB_ID"
```

- [ ] **Step 9: Run GREEN and shell checks**

```bash
PYTHONDONTWRITEBYTECODE=1 ../q4-ptv2-ab-integration/.venv/bin/python -m pytest -q -p no:cacheprovider \
  tools/launcher/tests/test_q30t_runtime_probe_runner.py \
  tools/launcher/tests/test_ptv23_node_keeper.py
bash -n tools/launcher/common/specdec/probe_ptv23_pyxis_keeper.sbatch
shellcheck -x tools/launcher/common/specdec/probe_ptv23_pyxis_keeper.sbatch
git diff --check
```

Expected: all tests and static checks pass.

- [ ] **Step 10: Independently review and commit the probe runner**

Review same-anchor import/mount, node-local paths, keeper lifetime, named reuse without image reimport, exact negative classification, process-group cancellation, receipt suppression, duplicate-node rejection, and cleanup scope. Then commit:

```bash
git add tools/launcher/common/specdec/probe_ptv23_pyxis_keeper.sbatch \
  tools/launcher/tests/test_q30t_runtime_probe_runner.py \
  tools/launcher/tests/test_ptv23_node_keeper.py
git commit -S -s -m "feat(specdec): probe Q30 keeper-backed Pyxis runtime"
```

### Task 6: Add a Test-Only-First Idempotent Qualification Submitter

**Files:**
- Create: `tools/launcher/common/specdec/submit_q30t_runtime_qualification.sh`
- Modify: `tools/launcher/tests/test_q30t_runtime_qualification_submitter.py`

**Interfaces:**
- Consumes: mode, profile path/SHA, archive receipt path/SHA, optional one-node receipt path/SHA, output receipt path, and bounded log path.
- Produces: exactly one archive, one-node, or two-node Slurm job ID after a successful test-only call, or adopts one exact active/completed identity without duplicate submission.

- [ ] **Step 1: Write RED submitter ordering and identity tests**

```python
@pytest.mark.parametrize("mode", ["--archive-tree", "--one-node", "--two-node"])
def test_runtime_submitter_runs_test_only_before_real(mode: str, tmp_path: Path) -> None:
    trace = run_fake_runtime_submitter(tmp_path, mode)
    assert trace.sbatch_calls[0].test_only is True
    assert sum(not call.test_only for call in trace.sbatch_calls) == 1
    assert trace.real_calls[0].export == "NONE"


def test_two_node_submission_requires_exact_one_node_receipt(tmp_path: Path) -> None:
    result = run_fake_runtime_submitter(tmp_path, "--two-node", omit_one_node=True)
    assert result.returncode == 2
    assert "two-node qualification requires reviewed one-node receipt" in result.stderr
    assert result.real_submissions == 0
```

Add tests for dirty/unpushed source, profile/source commit mismatch, wrong receipt SHA, existing foreign output, ambiguous duplicate scheduler comments, exact active-job adoption, exact completed-receipt adoption, and test-only failure suppressing real submission.

- [ ] **Step 2: Run RED**

```bash
PYTHONDONTWRITEBYTECODE=1 ../q4-ptv2-ab-integration/.venv/bin/python -m pytest -q -p no:cacheprovider \
  tools/launcher/tests/test_q30t_runtime_qualification_submitter.py -k runtime_submitter
```

Expected: failure because `submit_q30t_runtime_qualification.sh` is absent.

- [ ] **Step 3: Implement exact modes and scheduler arguments**

Use these fixed node/time identities:

```bash
case "$mode" in
  --archive-tree) nodes=1; time=00:20:00; job_name=q30t-runtime-archive-tree ;;
  --one-node) nodes=1; time=00:20:00; job_name=q30t-runtime-one-node ;;
  --two-node) nodes=2; time=00:30:00; job_name=q30t-runtime-two-node ;;
  *) usage ;;
esac
args=(--account=coreai_dlalgo_llm --partition=36x2-a01r --nodes="$nodes" \
  --ntasks-per-node=1 --time="$time" --job-name="$job_name" \
  --output="$log_path" --error="$log_path" --export=NONE)
```

Do not pass an explicit GPU option because the finalized Ptyche profile records exclusive four-GPU nodes and the real profile probe must verify visibility.

- [ ] **Step 4: Implement clean-source and exact-identity preflight**

Require a clean worktree, a valid signed `HEAD`, `HEAD == @{upstream}`, exact profile replay, stable receipt hashes, fresh output, and a canonical scheduler comment equal to SHA-256 over mode, source commit, profile SHA, archive receipt SHA, prerequisite receipt SHA or `NONE`, and output path. Query only the current user's exact job name/comment; reject more than one match.

```bash
[[ -z "$(git -C "$repo_root" status --porcelain --untracked-files=all)" ]]
git -C "$repo_root" verify-commit HEAD
head_commit="$(git -C "$repo_root" rev-parse HEAD)"
[[ "$head_commit" == "$(git -C "$repo_root" rev-parse '@{upstream}')" ]]
submission_comment="q30t-runtime-$(printf '%s\0' "$mode" "$head_commit" "$profile_sha" \
  "$archive_receipt_sha" "$one_node_receipt_sha" "$output_receipt" | \
  sha256sum | cut -d' ' -f1)"
```

- [ ] **Step 5: Implement test-only-first and exact adoption**

Always run `sbatch --test-only` before checking or creating the real reservation. If canonical output exists, verify and adopt its exact bytes. If one exact active job exists, return its numeric job ID. Otherwise call one `sbatch --parsable`; never fall back to an unfiltered queue scan and never inherit `ALL`, `PYTHONPATH`, `BASH_ENV`, or `ENV`.

```bash
sbatch --test-only "${args[@]}" --comment="$submission_comment" "${job_args[@]}" >/dev/null
mapfile -t matches < <(squeue --me -h -n "$job_name" -o '%A %k' | \
  awk -v expected="$submission_comment" '$2 == expected {print $1}')
(( ${#matches[@]} <= 1 )) || { echo "ambiguous exact runtime submissions" >&2; exit 2; }
if (( ${#matches[@]} == 1 )); then printf '%s\n' "${matches[0]}"; exit 0; fi
sbatch --parsable "${args[@]}" --comment="$submission_comment" "${job_args[@]}"
```

- [ ] **Step 6: Run GREEN and static checks**

```bash
PYTHONDONTWRITEBYTECODE=1 ../q4-ptv2-ab-integration/.venv/bin/python -m pytest -q -p no:cacheprovider \
  tools/launcher/tests/test_q30t_runtime_qualification_submitter.py -k runtime_submitter
bash -n tools/launcher/common/specdec/submit_q30t_runtime_qualification.sh
shellcheck -x tools/launcher/common/specdec/submit_q30t_runtime_qualification.sh
git diff --check
```

Expected: every command exits zero.

- [ ] **Step 7: Independently review and commit the submitter**

The reviewer must inspect test-only ordering, clean/pushed/signature checks, environment isolation, exact duplicate/adoption semantics, scheduler query bounds, and absence of real submission after any failed preflight. Then commit:

```bash
git add tools/launcher/common/specdec/submit_q30t_runtime_qualification.sh \
  tools/launcher/tests/test_q30t_runtime_qualification_submitter.py
git commit -S -s -m "feat(specdec): submit Q30 runtime qualification"
```

### Task 7: Integrate Qualification into the Q30 Production Contract

**Entry gate:** Do not begin this task while the Controller plan owns its integration files. Controller Task 9 in `docs/superpowers/plans/2026-08-27-q30t-700k-controller-trust-plan.md` must first pass, receive its independent review, and commit the frozen controller slice. The controller owner must explicitly hand off the signed freeze commit plus whole-file hashes for the controller, runner, submitter, and three focused controller test files. Runtime Tasks 1-6 may finish before this gate, but none may edit these handed-off files.

**Files:**
- Modify: `tools/launcher/common/specdec/q30t_ptv23_continuation.py`
- Modify: `tools/launcher/common/specdec/run_q30t_ptv23_continuation.sbatch`
- Modify: `tools/launcher/common/specdec/submit_q30t_ptv23_continuation.sh`
- Modify: `tools/launcher/tests/test_q30t_ptv23_continuation.py`
- Modify: `tools/launcher/tests/test_q30t_ptv23_controller.py`
- Modify: `tools/launcher/tests/test_q30t_ptv23_submitter.py`

**Interfaces:**
- Consumes: the signed Controller Task 9 freeze commit/file hashes, Task 1 `runtime_tree_identity`, and exact external profile path/SHA plus exact reviewed two-node qualification path/SHA in every canary/full `Q30TContinuationContract`.
- Produces: fail-closed `_authenticate_runtime_qualification(contract) -> RuntimeQualificationReceipt` and descriptor/environment fields `RUNTIME_CLUSTER_PROFILE`, `RUNTIME_CLUSTER_PROFILE_SHA256`, `RUNTIME_QUALIFICATION_RECEIPT`, and `RUNTIME_QUALIFICATION_RECEIPT_SHA256`.

- [ ] **Step 1: Verify and accept the explicit controller handoff**

```bash
CONTROLLER_FREEZE_COMMIT="$(git log -1 --format=%H \
  --grep='chore: freeze Q30 controller trust hashes')"
test -n "$CONTROLLER_FREEZE_COMMIT"
git verify-commit "$CONTROLLER_FREEZE_COMMIT"
git merge-base --is-ancestor "$CONTROLLER_FREEZE_COMMIT" HEAD
git show "$CONTROLLER_FREEZE_COMMIT:tools/launcher/common/specdec/q30t_ptv23_continuation.py" | sha256sum
git show "$CONTROLLER_FREEZE_COMMIT:tools/launcher/common/specdec/run_q30t_ptv23_continuation.sbatch" | sha256sum
git show "$CONTROLLER_FREEZE_COMMIT:tools/launcher/common/specdec/submit_q30t_ptv23_continuation.sh" | sha256sum
git show "$CONTROLLER_FREEZE_COMMIT:tools/launcher/tests/test_q30t_ptv23_continuation.py" | sha256sum
git show "$CONTROLLER_FREEZE_COMMIT:tools/launcher/tests/test_q30t_ptv23_controller.py" | sha256sum
git show "$CONTROLLER_FREEZE_COMMIT:tools/launcher/tests/test_q30t_ptv23_submitter.py" | sha256sum
```

Expected: the signed controller freeze exists and is an ancestor; all six printed hashes exactly equal the controller owner's explicit handoff. If any differs, return ownership to the Controller plan and do not edit integration files.

- [ ] **Step 2: Write RED shared-tree and approval-root tests**

```python
def test_q30_canary_requires_reviewed_two_node_runtime_qualification(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    contract = contract_with_runtime_qualification(tmp_path, phase="two-node")
    authorize_all_except_runtime_qualification(monkeypatch, contract)
    with pytest.raises(ValueError, match="runtime qualification receipt is not reviewed"):
        build_q30_training_command(contract, stage="canary")


def test_one_node_receipt_never_authorizes_canary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    contract = contract_with_runtime_qualification(tmp_path, phase="one-node")
    authorize_runtime_qualification(monkeypatch, contract)
    with pytest.raises(ValueError, match="exact reviewed two-node"):
        build_q30_training_command(contract, stage="canary")


def test_controller_runtime_tree_uses_shared_runtime_identity(tmp_path: Path) -> None:
    runtime = make_runtime_tree(tmp_path)
    assert continuation._tree_sha256(runtime) == runtime_tree_identity(runtime).sha256
```

Add descriptor round-trip mutation tests for each new path and SHA field; profile, archive, image, tree, source commit, runner, keeper, and attester substitution; unknown/extra receipt keys; and profile path outside the durable receipt root.

- [ ] **Step 3: Run RED**

```bash
PYTHONDONTWRITEBYTECODE=1 ../q4-ptv2-ab-integration/.venv/bin/python -m pytest -q -p no:cacheprovider \
  tools/launcher/tests/test_q30t_ptv23_continuation.py -k runtime_qualification
```

Expected: fixture construction fails because the contract has no qualification fields or approval constant, and the shared-tree test fails because the controller still owns its pre-handoff local implementation.

- [ ] **Step 4: Adopt the shared identity and add empty approval roots and contract fields**

Replace only the frozen controller's local runtime-tree body with this Task 1 delegation; do not alter any other controller-trust logic:

```python
from common.specdec.q30t_runtime_identity import runtime_tree_identity


def _tree_sha256(root: Path) -> str:
    return runtime_tree_identity(root).sha256
```

Then add the empty approval roots:

```python
APPROVED_Q30T_RUNTIME_ARCHIVE_TREE_RECEIPT_FILE_SHA256S: frozenset[str] = frozenset()
APPROVED_Q30T_RUNTIME_TREE_SHA256_BY_ARCHIVE_SHA256: Mapping[str, str] = MappingProxyType({})
APPROVED_Q30T_PRODUCTION_RUNTIME_IMAGE_SHA256S: frozenset[str] = frozenset()
APPROVED_Q30T_RUNTIME_CLUSTER_PROFILE_FILE_SHA256S: frozenset[str] = frozenset()
APPROVED_Q30T_RUNTIME_QUALIFICATION_RECEIPT_FILE_SHA256S: frozenset[str] = frozenset()
```

Add these fields to `Q30TContinuationContract` and descriptor path/SHA sets:

```python
runtime_archive_tree_receipt_path: Path
runtime_archive_tree_receipt_file_sha256: str
runtime_cluster_profile_path: Path
runtime_cluster_profile_file_sha256: str
runtime_qualification_receipt_path: Path
runtime_qualification_receipt_file_sha256: str
```

- [ ] **Step 5: Implement exact qualification authentication**

Stable-load the archive receipt, profile, and aggregate qualification. Require approval membership, `phase == "two-node"`, `expected_node_count == 2`, exact archive/image/tree/tool/runner identities, exact profile path/SHA, exact prerequisite one-node binding, and both proof booleans true. Require `qualification.source_commit == profile.source_commit`; the approved qualification producer commit may differ from the later approval-consumer commit because inserting approval literals necessarily creates a descendant commit. Runtime authentication verifies that the producer commit exists locally, is an ancestor of the consumer source commit, and that every qualification-bound runner/tool byte hash is unchanged at the consumer commit. The independent approval review, not a runtime network query, verifies that the producer commit was signed and pushed. Recompute every whole-file SHA and self-hash; do not trust descriptor strings alone.

```python
def _authenticate_runtime_qualification(
    contract: Q30TContinuationContract,
) -> RuntimeQualificationReceipt:
    qualification = load_runtime_qualification_receipt(
        contract.runtime_qualification_receipt_path,
        contract.runtime_qualification_receipt_file_sha256,
    )
    if contract.runtime_qualification_receipt_file_sha256 not in (
        APPROVED_Q30T_RUNTIME_QUALIFICATION_RECEIPT_FILE_SHA256S
    ):
        raise ValueError("runtime qualification receipt is not reviewed")
    if qualification.phase != "two-node" or qualification.expected_node_count != 2:
        raise ValueError("runtime qualification is not exact reviewed two-node evidence")
    return qualification
```

- [ ] **Step 6: Bind qualification through the launch descriptor and controller replay**

Add the four environment fields above. Before the existing runner sets `SLURM_EXPORT_ENV=ALL`, replay the profile and two-node receipt with isolated Python from authenticated source objects. Require `SLURM_CLUSTER_NAME` or the profile-confirmed site identity to match Ptyche, then require the allocation topology to remain exactly 16 nodes and four GPUs/node.

```python
environment |= {
    "RUNTIME_CLUSTER_PROFILE": str(contract.runtime_cluster_profile_path),
    "RUNTIME_CLUSTER_PROFILE_SHA256": contract.runtime_cluster_profile_file_sha256,
    "RUNTIME_QUALIFICATION_RECEIPT": str(contract.runtime_qualification_receipt_path),
    "RUNTIME_QUALIFICATION_RECEIPT_SHA256": (
        contract.runtime_qualification_receipt_file_sha256
    ),
}
```

- [ ] **Step 7: Require qualified descriptors in the training submitter**

The existing `submit_q30t_ptv23_continuation.sh` must replay the descriptor and reject missing qualification variables before `sbatch --test-only`. Preserve `--export=NONE`, clean-pushed-source checks, and test-only-first behavior. Do not add any caller-controlled bypass or boolean implementation flag.

```bash
for name in RUNTIME_CLUSTER_PROFILE RUNTIME_CLUSTER_PROFILE_SHA256 \
  RUNTIME_QUALIFICATION_RECEIPT RUNTIME_QUALIFICATION_RECEIPT_SHA256; do
  [[ -n "${authenticated_environment[$name]:-}" ]] || {
    echo "authenticated Q30 descriptor lacks runtime qualification: $name" >&2
    exit 2
  }
done
sbatch --test-only "${args[@]}" "${job_args[@]}" >/dev/null
```

- [ ] **Step 8: Run GREEN focused behavior**

```bash
PYTHONDONTWRITEBYTECODE=1 ../q4-ptv2-ab-integration/.venv/bin/python -m pytest -q -p no:cacheprovider \
  tools/launcher/tests/test_q30t_runtime_identity.py \
  tools/launcher/tests/test_q30t_ptv23_continuation.py \
  tools/launcher/tests/test_q30t_ptv23_controller.py \
  tools/launcher/tests/test_q30t_ptv23_submitter.py
bash -n tools/launcher/common/specdec/run_q30t_ptv23_continuation.sbatch \
  tools/launcher/common/specdec/submit_q30t_ptv23_continuation.sh
shellcheck -x tools/launcher/common/specdec/run_q30t_ptv23_continuation.sbatch \
  tools/launcher/common/specdec/submit_q30t_ptv23_continuation.sh
```

Expected: all tests and shell checks pass while approval roots remain empty by default.

- [ ] **Step 9: Run static and descriptor-attack gates**

```bash
../q4-ptv2-ab-integration/.venv/bin/ruff format --check \
  tools/launcher/common/specdec/q30t_ptv23_continuation.py \
  tools/launcher/tests/test_q30t_ptv23_continuation.py \
  tools/launcher/tests/test_q30t_ptv23_controller.py \
  tools/launcher/tests/test_q30t_ptv23_submitter.py
../q4-ptv2-ab-integration/.venv/bin/ruff check \
  tools/launcher/common/specdec/q30t_ptv23_continuation.py \
  tools/launcher/tests/test_q30t_ptv23_continuation.py \
  tools/launcher/tests/test_q30t_ptv23_controller.py \
  tools/launcher/tests/test_q30t_ptv23_submitter.py
PYTHONPATH=tools/launcher ../q4-ptv2-ab-integration/.venv/bin/pyright \
  tools/launcher/common/specdec/q30t_ptv23_continuation.py
PYTHONDONTWRITEBYTECODE=1 ../q4-ptv2-ab-integration/.venv/bin/python -m py_compile \
  tools/launcher/common/specdec/q30t_ptv23_continuation.py
git diff --check
```

Expected: every command exits zero.

- [ ] **Step 10: Independently review and commit fail-closed integration**

The reviewer must prove that content allowlisting alone cannot bypass platform qualification, a one-node receipt cannot authorize training, every descriptor field is round-trip mutation tested, receipt/profile paths are rebound-safe, and missing approval values stop before keeper startup. Then commit:

```bash
git add tools/launcher/common/specdec/q30t_ptv23_continuation.py \
  tools/launcher/common/specdec/run_q30t_ptv23_continuation.sbatch \
  tools/launcher/common/specdec/submit_q30t_ptv23_continuation.sh \
  tools/launcher/tests/test_q30t_ptv23_continuation.py \
  tools/launcher/tests/test_q30t_ptv23_controller.py \
  tools/launcher/tests/test_q30t_ptv23_submitter.py
git commit -S -s -m "feat(specdec): require Q30 runtime qualification"
```

### Task 8: Freeze and Verify the Complete Local Qualification Slice

**Files:**
- Modify only when a failing gate identifies a defect in its owning task.

**Interfaces:**
- Consumes: Tasks 1-7 on one clean branch, including the verified Controller Task 9 freeze/handoff inherited by Task 7.
- Produces: a clean signed pushed commit eligible for Ptyche test-only submission; no approval constants are populated in this task.

- [ ] **Step 1: Run the complete focused pytest gate**

```bash
PYTHONDONTWRITEBYTECODE=1 ../q4-ptv2-ab-integration/.venv/bin/python -m pytest -q -p no:cacheprovider \
  tools/launcher/tests/test_q30t_runtime_identity.py \
  tools/launcher/tests/test_q30t_runtime_archive_receipt.py \
  tools/launcher/tests/test_ptv23_node_keeper.py \
  tools/launcher/tests/test_ptv23_runtime_attestation.py \
  tools/launcher/tests/test_q30t_runtime_probe_runner.py \
  tools/launcher/tests/test_q30t_runtime_qualification_submitter.py \
  tools/launcher/tests/test_q30t_ptv23_continuation.py \
  tools/launcher/tests/test_q30t_ptv23_controller.py \
  tools/launcher/tests/test_q30t_ptv23_submitter.py
```

Expected: all tests pass with no xfail or skipped trust-boundary cases.

- [ ] **Step 2: Run all formatting, lint, type, compile, and shell gates**

```bash
../q4-ptv2-ab-integration/.venv/bin/ruff format --check \
  tools/launcher/common/specdec/q30t_runtime_identity.py \
  tools/launcher/common/specdec/q30t_runtime_archive_receipt.py \
  tools/launcher/common/specdec/ptv23_runtime_attestation.py \
  tools/launcher/common/specdec/q30t_ptv23_cluster_profile.py \
  tools/launcher/common/specdec/q30t_ptv23_continuation.py \
  tools/launcher/tests/test_q30t_runtime_identity.py \
  tools/launcher/tests/test_q30t_runtime_archive_receipt.py \
  tools/launcher/tests/test_ptv23_runtime_attestation.py \
  tools/launcher/tests/test_q30t_runtime_probe_runner.py \
  tools/launcher/tests/test_q30t_runtime_qualification_submitter.py \
  tools/launcher/tests/test_q30t_ptv23_submitter.py
../q4-ptv2-ab-integration/.venv/bin/ruff check tools/launcher/common/specdec tools/launcher/tests
PYTHONPATH=tools/launcher ../q4-ptv2-ab-integration/.venv/bin/pyright \
  tools/launcher/common/specdec/q30t_runtime_identity.py \
  tools/launcher/common/specdec/q30t_runtime_archive_receipt.py \
  tools/launcher/common/specdec/ptv23_runtime_attestation.py \
  tools/launcher/common/specdec/q30t_ptv23_cluster_profile.py \
  tools/launcher/common/specdec/q30t_ptv23_continuation.py
PYTHONDONTWRITEBYTECODE=1 ../q4-ptv2-ab-integration/.venv/bin/python -m py_compile \
  tools/launcher/common/specdec/q30t_runtime_identity.py \
  tools/launcher/common/specdec/q30t_runtime_archive_receipt.py \
  tools/launcher/common/specdec/ptv23_runtime_attestation.py \
  tools/launcher/common/specdec/q30t_ptv23_cluster_profile.py \
  tools/launcher/common/specdec/q30t_ptv23_continuation.py
bash -n tools/launcher/common/specdec/run_q30t_runtime_archive_receipt.sbatch \
  tools/launcher/common/specdec/probe_ptv23_pyxis_keeper.sbatch \
  tools/launcher/common/specdec/submit_q30t_runtime_qualification.sh \
  tools/launcher/common/specdec/run_q30t_ptv23_continuation.sbatch \
  tools/launcher/common/specdec/submit_q30t_ptv23_continuation.sh
shellcheck -x tools/launcher/common/specdec/run_q30t_runtime_archive_receipt.sbatch \
  tools/launcher/common/specdec/probe_ptv23_pyxis_keeper.sbatch \
  tools/launcher/common/specdec/submit_q30t_runtime_qualification.sh \
  tools/launcher/common/specdec/run_q30t_ptv23_continuation.sbatch \
  tools/launcher/common/specdec/submit_q30t_ptv23_continuation.sh
git diff --check
```

Expected: every command exits zero.

- [ ] **Step 3: Run a fresh independent hostile review of the frozen slice**

Give the reviewer the exact commit and only the files listed in the File Responsibility Map. Require explicit review of archive extraction confinement, receipt canonicality, same-anchor Pyxis use, named reuse, keeper-loss classification, Slurm environment isolation, duplicate submission prevention, and approval/contract fail-closed behavior. Resolve every critical and warning finding with a new RED/GREEN cycle.

- [ ] **Step 4: Verify clean signed source and push once**

```bash
git status --short
git log -1 --show-signature --format=fuller
git push -u gitlab sj/q30t-ptv23-complement-700k
test "$(git rev-parse HEAD)" = "$(git rev-parse @{upstream})"
```

Expected: status is empty, the signature is valid, push succeeds, and local/upstream commits are identical.

### Task 9: Produce and Independently Review the Archive/Tree Receipt on Ptyche

**Files:**
- No repository files change while the frozen producer commit is executing.
- Produce external immutable review evidence: `/lustre/fsw/coreai_dlalgo_llm/users/sna/modelopt-qwen3-drafter-training/receipts/q30t-swe-heavy-700k-v1/runtime-qualification/archive-tree.review.json`.

**Interfaces:**
- Consumes: the clean pushed Task 8 commit and exact archive.
- Produces: one independently reviewed archive/tree receipt whole-file SHA and runtime-tree SHA; approval insertion remains deferred until the two-node receipt is also reviewed.

- [ ] **Step 1: Verify the remote source and finalize the external profile**

In one authenticated Ptyche session:

```bash
cd /home/sna/ModelOpt_SpecDec
git pull --ff-only
git status --short
git log -1 --show-signature --format=fuller
SOURCE_COMMIT="$(git rev-parse HEAD)"
RECEIPT_ROOT=/lustre/fsw/coreai_dlalgo_llm/users/sna/modelopt-qwen3-drafter-training/receipts/q30t-swe-heavy-700k-v1/runtime-qualification
mkdir -p "$RECEIPT_ROOT/logs"
env -i PATH=/usr/bin:/bin PYTHONPATH=/home/sna/ModelOpt_SpecDec/tools/launcher \
  /usr/bin/python3.12 -m common.specdec.q30t_ptv23_cluster_profile finalize \
  --source-checkout /home/sna/ModelOpt_SpecDec \
  --source-commit "$SOURCE_COMMIT" \
  --durable-receipt-root "$RECEIPT_ROOT" \
  --output "$RECEIPT_ROOT/ptyche-profile.json" \
  --job-id profile-finalize
sha256sum "$RECEIPT_ROOT/ptyche-profile.json"
```

Expected: clean status, valid signature, canonical profile creation, and one lowercase profile SHA.

- [ ] **Step 2: Run archive producer test-only**

```bash
PROFILE_SHA="$(sha256sum "$RECEIPT_ROOT/ptyche-profile.json" | cut -d' ' -f1)"
/home/sna/ModelOpt_SpecDec/tools/launcher/common/specdec/submit_q30t_runtime_qualification.sh \
  --test-only --archive-tree \
  --profile "$RECEIPT_ROOT/ptyche-profile.json" \
  --profile-sha256 "$PROFILE_SHA" \
  --output-receipt "$RECEIPT_ROOT/archive-tree.json" \
  --log "$RECEIPT_ROOT/logs/archive-tree-%j.out"
```

Expected: `sbatch --test-only` succeeds and no real job is created.

- [ ] **Step 3: Submit the exact archive producer**

```bash
ARCHIVE_JOB_ID="$(/home/sna/ModelOpt_SpecDec/tools/launcher/common/specdec/submit_q30t_runtime_qualification.sh \
  --archive-tree \
  --profile "$RECEIPT_ROOT/ptyche-profile.json" \
  --profile-sha256 "$PROFILE_SHA" \
  --output-receipt "$RECEIPT_ROOT/archive-tree.json" \
  --log "$RECEIPT_ROOT/logs/archive-tree-%j.out")"
case "$ARCHIVE_JOB_ID" in (*[!0-9]*|'') exit 2;; esac
printf '%s\n' "$ARCHIVE_JOB_ID"
```

Expected: exactly one numeric job ID.

- [ ] **Step 4: Monitor once per minute for at least five minutes**

Run each command at least 60 seconds apart; do not issue per-step or unfiltered scheduler queries:

```bash
squeue -j "$ARCHIVE_JOB_ID" -h -o '%i %T %r %S'
sleep 60
squeue -j "$ARCHIVE_JOB_ID" -h -o '%i %T %r %S'
sleep 60
squeue -j "$ARCHIVE_JOB_ID" -h -o '%i %T %r %S'
sleep 60
squeue -j "$ARCHIVE_JOB_ID" -h -o '%i %T %r %S'
sleep 60
squeue -j "$ARCHIVE_JOB_ID" -h -o '%i %T %r %S'
```

Expected: PENDING or RUNNING is classified without error; if the job finishes early, replace remaining checks with one final `sacct -j "$ARCHIVE_JOB_ID" --format=JobID,State,ExitCode,Elapsed -n` and bounded `tail -n 100` of its exact log.

- [ ] **Step 5: Verify the completed archive receipt without approving it**

```bash
sacct -j "$ARCHIVE_JOB_ID" --format=JobID,State,ExitCode,Elapsed -n
ARCHIVE_RECEIPT_SHA="$(sha256sum "$RECEIPT_ROOT/archive-tree.json" | cut -d' ' -f1)"
env -i PATH=/usr/bin:/bin PYTHONPATH=/home/sna/ModelOpt_SpecDec/tools/launcher \
  /usr/bin/python3.12 -m common.specdec.q30t_runtime_archive_receipt verify \
  --receipt "$RECEIPT_ROOT/archive-tree.json" \
  --sha256 "$ARCHIVE_RECEIPT_SHA"
```

Expected: job state `COMPLETED`, exit code `0:0`, and verifier success.

- [ ] **Step 6: Perform independent archive/tree replay**

The independent reviewer receives only the pushed commit, profile, archive, and produced receipt. They stable-hash the archive, extract to a distinct fresh `/raid/scratch/$USER/q30t-runtime-archive-review-$SLURM_JOB_ID` root in a separate CPU job, recompute `runtime_tree_identity`, replay sentinel inventory and canonical self-hash, and return the exact `ARCHIVE_RECEIPT_SHA` plus `runtime_tree_sha256` with a clean verdict.

- [ ] **Step 7: Publish external independent-review evidence without changing source**

The independent reviewer publishes canonical `archive-tree.review.json` through `atomic_publish_bytes`. It binds the reviewed producer commit, profile SHA, archive receipt whole-file SHA, archive SHA, recomputed runtime-tree SHA, review-tool SHA, and verdict `clean`. The review receipt is external evidence and does not modify or approve the source checkout.

- [ ] **Step 8: Reconfirm the frozen source before the one-node probe**

```bash
git status --short
test "$(git rev-parse HEAD)" = "$SOURCE_COMMIT"
test "$(git rev-parse HEAD)" = "$(git rev-parse @{upstream})"
sha256sum "$RECEIPT_ROOT/archive-tree.json" "$RECEIPT_ROOT/archive-tree.review.json"
```

Expected: status remains empty, source/upstream/producer commits are identical, and both external evidence files are stable. Do not commit or populate any approval constant between the archive, one-node, and two-node probes; all three runtime receipts must bind this one frozen producer commit and profile.

### Task 10: Run and Review the One-Node Ptyche Qualification

**Files:**
- No repository files change while the frozen producer commit is executing.
- Produce external immutable review evidence: `/lustre/fsw/coreai_dlalgo_llm/users/sna/modelopt-qwen3-drafter-training/receipts/q30t-swe-heavy-700k-v1/runtime-qualification/one-node.review.json`.

**Interfaces:**
- Consumes: the unchanged frozen producer commit, exact profile, independently reviewed archive receipt, and archive review receipt.
- Produces: a reviewed but not controller-approved one-node qualification receipt.

- [ ] **Step 1: Reconfirm the unchanged frozen commit and recompute input hashes**

```bash
cd /home/sna/ModelOpt_SpecDec
git pull --ff-only
git status --short
git log -1 --show-signature --format=fuller
SOURCE_COMMIT="$(git rev-parse HEAD)"
RECEIPT_ROOT=/lustre/fsw/coreai_dlalgo_llm/users/sna/modelopt-qwen3-drafter-training/receipts/q30t-swe-heavy-700k-v1/runtime-qualification
PROFILE_SHA="$(sha256sum "$RECEIPT_ROOT/ptyche-profile.json" | cut -d' ' -f1)"
ARCHIVE_RECEIPT_SHA="$(sha256sum "$RECEIPT_ROOT/archive-tree.json" | cut -d' ' -f1)"
```

Expected: clean source, valid pushed signature, and exact stable hashes.

- [ ] **Step 2: Run one-node test-only**

```bash
/home/sna/ModelOpt_SpecDec/tools/launcher/common/specdec/submit_q30t_runtime_qualification.sh \
  --test-only --one-node \
  --profile "$RECEIPT_ROOT/ptyche-profile.json" \
  --profile-sha256 "$PROFILE_SHA" \
  --archive-tree-receipt "$RECEIPT_ROOT/archive-tree.json" \
  --archive-tree-receipt-sha256 "$ARCHIVE_RECEIPT_SHA" \
  --output-receipt "$RECEIPT_ROOT/one-node.json" \
  --log "$RECEIPT_ROOT/logs/one-node-%j.out"
```

Expected: test-only succeeds with no job creation.

- [ ] **Step 3: Submit one exact one-node job**

```bash
ONE_NODE_JOB_ID="$(/home/sna/ModelOpt_SpecDec/tools/launcher/common/specdec/submit_q30t_runtime_qualification.sh \
  --one-node \
  --profile "$RECEIPT_ROOT/ptyche-profile.json" \
  --profile-sha256 "$PROFILE_SHA" \
  --archive-tree-receipt "$RECEIPT_ROOT/archive-tree.json" \
  --archive-tree-receipt-sha256 "$ARCHIVE_RECEIPT_SHA" \
  --output-receipt "$RECEIPT_ROOT/one-node.json" \
  --log "$RECEIPT_ROOT/logs/one-node-%j.out")"
case "$ONE_NODE_JOB_ID" in (*[!0-9]*|'') exit 2;; esac
printf '%s\n' "$ONE_NODE_JOB_ID"
```

Expected: exactly one numeric job ID.

- [ ] **Step 4: Monitor the exact job five times at one-minute intervals**

```bash
squeue -j "$ONE_NODE_JOB_ID" -h -o '%i %T %r %S'
sleep 60
squeue -j "$ONE_NODE_JOB_ID" -h -o '%i %T %r %S'
sleep 60
squeue -j "$ONE_NODE_JOB_ID" -h -o '%i %T %r %S'
sleep 60
squeue -j "$ONE_NODE_JOB_ID" -h -o '%i %T %r %S'
sleep 60
squeue -j "$ONE_NODE_JOB_ID" -h -o '%i %T %r %S'
```

Expected: checks are at least 60 seconds apart. On early completion, use one exact `sacct` query and bounded log tail instead of further `squeue` calls.

- [ ] **Step 5: Replay the one-node receipt**

```bash
sacct -j "$ONE_NODE_JOB_ID" --format=JobID,State,ExitCode,Elapsed -n
ONE_NODE_RECEIPT_SHA="$(sha256sum "$RECEIPT_ROOT/one-node.json" | cut -d' ' -f1)"
env -i PATH=/usr/bin:/bin PYTHONPATH=/home/sna/ModelOpt_SpecDec/tools/launcher \
  /usr/bin/python3.12 -m common.specdec.ptv23_runtime_attestation verify \
  --receipt "$RECEIPT_ROOT/one-node.json" \
  --sha256 "$ONE_NODE_RECEIPT_SHA"
```

Expected: `COMPLETED`, `0:0`, `phase=one-node`, exactly one node, observed image SHA `e7be53f2754097c88f7c801da92f6d94794ec4d78d9df937fcd315a6994297f0`, four unique GPUs, named reuse true, and keeper-loss failure true.

- [ ] **Step 6: Independently review the one-node receipt and evidence**

The reviewer validates exact canonical bytes, whole-file/self-hash, archive/profile/source/tool/runner identities, keeper PID/start/inode evidence, same image anchor for import and mount, complete observed image hash, Python/import origins, aarch64, four GPUs, Pyxis/enroot versions, named reuse without image import, and expected keeper-loss failure. Publish canonical external `one-node.review.json` binding the frozen producer commit, profile SHA, archive review SHA, exact `ONE_NODE_RECEIPT_SHA`, review-tool SHA, and verdict `clean`. Do not populate any controller approval set from this receipt.

- [ ] **Step 7: Reconfirm that the source did not change before two-node test-only**

```bash
git status --short
test "$(git rev-parse HEAD)" = "$SOURCE_COMMIT"
test "$(git rev-parse HEAD)" = "$(git rev-parse @{upstream})"
sha256sum "$RECEIPT_ROOT/one-node.json" "$RECEIPT_ROOT/one-node.review.json"
```

Expected: status is empty, source remains the exact frozen producer commit, both evidence files are stable, and no production approval constant changed.

### Task 11: Run, Review, and Approve the Two-Node Ptyche Qualification

**Files:**
- Create after all external reviews finish: `.superpowers/sdd/2026-08-27-q30t-swe-heavy-700k/runtime-qualification-report.md`
- Modify after independent review only: `tools/launcher/common/specdec/q30t_ptv23_continuation.py`

**Interfaces:**
- Consumes: exact reviewed one-node receipt path/SHA plus all common profile/archive identities.
- Produces: independently reviewed image, profile, and two-node qualification approvals that unlock contract construction but do not submit a canary.

- [ ] **Step 1: Verify the unchanged producer commit and run two-node test-only**

```bash
cd /home/sna/ModelOpt_SpecDec
git pull --ff-only
git status --short
git log -1 --show-signature --format=fuller
SOURCE_COMMIT="$(git rev-parse HEAD)"
RECEIPT_ROOT=/lustre/fsw/coreai_dlalgo_llm/users/sna/modelopt-qwen3-drafter-training/receipts/q30t-swe-heavy-700k-v1/runtime-qualification
PROFILE_SHA="$(sha256sum "$RECEIPT_ROOT/ptyche-profile.json" | cut -d' ' -f1)"
ARCHIVE_RECEIPT_SHA="$(sha256sum "$RECEIPT_ROOT/archive-tree.json" | cut -d' ' -f1)"
ONE_NODE_RECEIPT_SHA="$(sha256sum "$RECEIPT_ROOT/one-node.json" | cut -d' ' -f1)"
/home/sna/ModelOpt_SpecDec/tools/launcher/common/specdec/submit_q30t_runtime_qualification.sh \
  --test-only --two-node \
  --profile "$RECEIPT_ROOT/ptyche-profile.json" \
  --profile-sha256 "$PROFILE_SHA" \
  --archive-tree-receipt "$RECEIPT_ROOT/archive-tree.json" \
  --archive-tree-receipt-sha256 "$ARCHIVE_RECEIPT_SHA" \
  --one-node-receipt "$RECEIPT_ROOT/one-node.json" \
  --one-node-receipt-sha256 "$ONE_NODE_RECEIPT_SHA" \
  --output-receipt "$RECEIPT_ROOT/two-node.json" \
  --log "$RECEIPT_ROOT/logs/two-node-%j.out"
```

Expected: test-only succeeds and no real job is created.

- [ ] **Step 2: Submit one exact two-node job**

```bash
TWO_NODE_JOB_ID="$(/home/sna/ModelOpt_SpecDec/tools/launcher/common/specdec/submit_q30t_runtime_qualification.sh \
  --two-node \
  --profile "$RECEIPT_ROOT/ptyche-profile.json" \
  --profile-sha256 "$PROFILE_SHA" \
  --archive-tree-receipt "$RECEIPT_ROOT/archive-tree.json" \
  --archive-tree-receipt-sha256 "$ARCHIVE_RECEIPT_SHA" \
  --one-node-receipt "$RECEIPT_ROOT/one-node.json" \
  --one-node-receipt-sha256 "$ONE_NODE_RECEIPT_SHA" \
  --output-receipt "$RECEIPT_ROOT/two-node.json" \
  --log "$RECEIPT_ROOT/logs/two-node-%j.out")"
case "$TWO_NODE_JOB_ID" in (*[!0-9]*|'') exit 2;; esac
printf '%s\n' "$TWO_NODE_JOB_ID"
```

Expected: exactly one numeric job ID.

- [ ] **Step 3: Monitor the exact job five times at one-minute intervals**

```bash
squeue -j "$TWO_NODE_JOB_ID" -h -o '%i %T %r %S'
sleep 60
squeue -j "$TWO_NODE_JOB_ID" -h -o '%i %T %r %S'
sleep 60
squeue -j "$TWO_NODE_JOB_ID" -h -o '%i %T %r %S'
sleep 60
squeue -j "$TWO_NODE_JOB_ID" -h -o '%i %T %r %S'
sleep 60
squeue -j "$TWO_NODE_JOB_ID" -h -o '%i %T %r %S'
```

Expected: queries are at least 60 seconds apart; early completion switches to one exact `sacct` query and bounded exact-log tail.

- [ ] **Step 4: Replay the two-node receipt**

```bash
sacct -j "$TWO_NODE_JOB_ID" --format=JobID,State,ExitCode,Elapsed -n
TWO_NODE_RECEIPT_SHA="$(sha256sum "$RECEIPT_ROOT/two-node.json" | cut -d' ' -f1)"
env -i PATH=/usr/bin:/bin PYTHONPATH=/home/sna/ModelOpt_SpecDec/tools/launcher \
  /usr/bin/python3.12 -m common.specdec.ptv23_runtime_attestation verify \
  --receipt "$RECEIPT_ROOT/two-node.json" \
  --sha256 "$TWO_NODE_RECEIPT_SHA"
```

Expected: `COMPLETED`, `0:0`, `phase=two-node`, exactly two unique nodes/PIDs/inodes, exact one-node prerequisite SHA, common runtime identities, named reuse true, and keeper-loss failure true.

- [ ] **Step 5: Perform independent two-node replay and hostile review**

The reviewer verifies both explicit node files and aggregate bytes; rejects missing/extra/duplicate nodes; checks distinct Slurm nodes, keeper PIDs/start ticks, and `(device,inode)` pairs; recomputes ordered node receipt hashes; replays the exact one-node prerequisite and review receipt; confirms all common profile/source/archive/image/tree/runner/tool identities; and classifies named reuse and keeper-loss failures from bounded logs. Publish canonical external `two-node.review.json` binding all three receipt/review roots, the frozen producer commit, exact profile SHA, exact `TWO_NODE_RECEIPT_SHA`, review-tool SHA, and verdict `clean`.

- [ ] **Step 6: Record all external evidence in the durable source report**

Create `runtime-qualification-report.md` with the exact frozen producer commit; profile/archive/image paths and hashes; archive, one-node, and two-node job IDs; their five monitoring observations; exact receipt and review-receipt whole-file hashes; exact runtime-tree SHA; bounded log paths; and independent clean verdicts. Do not abbreviate a hash or copy a value from console scrollback: obtain each value again with stable-file verification.

- [ ] **Step 7: Insert all exact independently reviewed runtime approvals atomically**

Use `apply_patch` to add:

- the reviewed archive receipt whole-file SHA to `APPROVED_Q30T_RUNTIME_ARCHIVE_TREE_RECEIPT_FILE_SHA256S`;
- the mapping from archive SHA `4a20aee61f290c48bed22a84b4a0ae0cbdc54e3e3910854d253188c8854f5dc9` to the reviewed exact runtime-tree SHA in `APPROVED_Q30T_RUNTIME_TREE_SHA256_BY_ARCHIVE_SHA256`;
- image SHA `e7be53f2754097c88f7c801da92f6d94794ec4d78d9df937fcd315a6994297f0` to `APPROVED_Q30T_PRODUCTION_RUNTIME_IMAGE_SHA256S`;
- the reviewer's literal `PROFILE_SHA` to `APPROVED_Q30T_RUNTIME_CLUSTER_PROFILE_FILE_SHA256S`;
- the reviewer's literal `TWO_NODE_RECEIPT_SHA` to `APPROVED_Q30T_RUNTIME_QUALIFICATION_RECEIPT_FILE_SHA256S`.

Do not add the one-node receipt SHA. Do not submit a canary in the approval patch.

- [ ] **Step 8: Run the post-approval consumer gate**

```bash
PYTHONDONTWRITEBYTECODE=1 ../q4-ptv2-ab-integration/.venv/bin/python -m pytest -q -p no:cacheprovider \
  tools/launcher/tests/test_q30t_runtime_archive_receipt.py \
  tools/launcher/tests/test_ptv23_runtime_attestation.py \
  tools/launcher/tests/test_q30t_ptv23_continuation.py \
  tools/launcher/tests/test_q30t_ptv23_controller.py \
  tools/launcher/tests/test_q30t_ptv23_submitter.py
bash -n tools/launcher/common/specdec/run_q30t_ptv23_continuation.sbatch \
  tools/launcher/common/specdec/submit_q30t_ptv23_continuation.sh
shellcheck -x tools/launcher/common/specdec/run_q30t_ptv23_continuation.sbatch \
  tools/launcher/common/specdec/submit_q30t_ptv23_continuation.sh
git diff --check
```

Expected: all tests/static checks pass, exact approved contract construction succeeds, one-node/unapproved/mutated receipts remain rejected, and no Slurm job is submitted.

- [ ] **Step 9: Independently review, commit, sign, and push the approval transition**

```bash
git add tools/launcher/common/specdec/q30t_ptv23_continuation.py \
  .superpowers/sdd/2026-08-27-q30t-swe-heavy-700k/runtime-qualification-report.md
git commit -S -s -m "chore(specdec): approve Q30 Ptyche runtime qualification"
git push
git status --short
test "$(git rev-parse HEAD)" = "$(git rev-parse @{upstream})"
```

Expected: independent review reports no critical/warning findings; the diff contains only exact reviewed literals and report evidence; source is clean and pushed.

Task 11 ends at the runtime approval transition. Canary test-only and real DFlash/DSpark submission remain blocked until Subprojects A and B have independently approved dataset and controller evidence and have produced their exact launch descriptors.
