# Q30 Thinking 700K Multi-Node Runtime Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the single-node descriptor runner with a verified 16-node per-node keeper, named Pyxis container, runtime-attestation probe, and rank-safe evidence path suitable for Q30 Thinking canary and full training.

**Architecture:** One long-lived keeper task per node stages immutable inputs into node-local anonymous files, retains descriptors, and publishes identical node-local anchors. Pyxis imports a per-job named container once per node from those anchors; the first in-container probe hashes the exact mounted SQSH and emits per-node evidence. Training begins only after all node receipts reconcile, stages data locally on each node, and lets only global rank 0 publish durable receipts.

**Tech Stack:** Bash, Python 3.13, SLURM, Pyxis, enroot, `/raid/scratch`, canonical JSON/SHA-256 receipts, pytest, ShellCheck.

**Spec:** `docs/superpowers/specs/2026-08-26-q30t-ptv23-complement-700k-design.md`

## Global Constraints

- Keepers and high-churn staging use node-local `/raid/scratch`; durable receipts and checkpoints use `/lustre`.
- One keeper is required per allocated node and remains alive until every training descendant exits.
- The same anchor pathname must resolve to that node's keeper; a batch-node-only `/proc/PID/fd` path is forbidden.
- Pyxis/enroot is an explicit trusted computing boundary. Runtime launch remains disabled until one-node and two-node probes pass on the selected cluster.
- Container, contract, tool, target, dataset, parent, tokenizer, and receipts are independently copied and rehashed after staging.
- All ranks stage and verify node-local inputs; only global rank 0 publishes durable canary/full evidence after a barrier.
- Scheduler queries are filtered and at least 60 seconds apart; `sbatch --test-only` precedes every real submission; new jobs are monitored for five minutes.
- Every repository commit is signed and includes a DCO sign-off.

---

### Task 1: Implement the per-node keeper protocol

**Files:**
- Create: `tools/launcher/common/specdec/ptv23_node_keeper.py`
- Create: `tools/launcher/tests/test_ptv23_node_keeper.py`

**Interfaces:**
- Produces: `KeeperPlan`, `KeeperReceipt`, `stage_regular_to_tmpfile`, `open_tree_root`, and CLI `serve --plan PATH --receipt PATH`.
- Consumes: canonical plan containing node-local scratch root, exact source paths/SHA-256s, directory roots, anchor root, and keeper identity.

- [ ] **Step 1: Write failing keeper tests**

```python
def test_keeper_stages_every_file_with_postcopy_digest(tmp_path: Path) -> None:
    receipt = run_keeper_once(tmp_path, inputs={"image": b"image", "contract": b"contract"})
    assert receipt.names == ("contract", "image")
    assert all(item.source_sha256 == item.staged_sha256 for item in receipt.items)


def test_keeper_short_write_and_dead_process_fail_closed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(os, "write", short_write)
    keeper = start_keeper(tmp_path)
    wait_ready(keeper)
    keeper.kill()
    with pytest.raises(KeeperError, match="keeper is not alive"):
        validate_keeper_receipt(keeper.receipt)
```

- [ ] **Step 2: Run RED**

```bash
PYTHONDONTWRITEBYTECODE=1 ../q4-ptv2-ab-integration/.venv/bin/python -m pytest -q \
  tools/launcher/tests/test_ptv23_node_keeper.py
```

Expected: missing module import.

- [ ] **Step 3: Implement canonical plan and receipt types**

```python
@dataclass(frozen=True)
class KeeperItem:
    name: str
    source_path: Path
    expected_sha256: str
    staged_size: int
    staged_sha256: str
    anchor_path: Path


@dataclass(frozen=True)
class KeeperReceipt:
    schema_version: Literal["ptv23-node-keeper-v1"]
    job_id: str
    node_name: str
    keeper_pid: int
    keeper_start_ticks: int
    items: tuple[KeeperItem, ...]
    receipt_sha256: str
```

Use `O_RDONLY|O_NOFOLLOW` for sources, loop over short writes into `O_TMPFILE`, fsync, rewind, and independently hash the staged descriptor. Open directory roots component-by-component. Create anchors atomically inside a private `0700` node-local directory and bind each anchor to `/proc/<keeper-pid>/fd/<fd>`.

- [ ] **Step 4: Implement lifetime and cleanup semantics**

The keeper holds all descriptors while reading a controller-owned FIFO. EOF or an explicit stop closes descriptors, removes only exact owned anchors after inode/name rechecks, and exits. Signals produce a failed receipt and never delete foreign replacements.

- [ ] **Step 5: Run tests and commit**

```bash
PYTHONDONTWRITEBYTECODE=1 ../q4-ptv2-ab-integration/.venv/bin/python -m pytest -q \
  tools/launcher/tests/test_ptv23_node_keeper.py
git add tools/launcher/common/specdec/ptv23_node_keeper.py \
  tools/launcher/tests/test_ptv23_node_keeper.py
git commit -S -s -m "feat(specdec): add per-node input keeper"
```

### Task 2: Add per-node runtime attestation and named-container evidence

**Files:**
- Create: `tools/launcher/common/specdec/ptv23_runtime_attestation.py`
- Create: `tools/launcher/tests/test_ptv23_runtime_attestation.py`
- Create: `tools/launcher/common/specdec/probe_ptv23_pyxis_keeper.sbatch`

**Interfaces:**
- Produces: `RuntimeAttestationReceipt`, CLI `attest --image /run/q30t/runtime.sqsh --contract ...`, and a one-/two-node platform probe.
- Consumes: keeper image anchor, expected SQSH SHA, contract/tool hashes, pinned sentinel inventory, Pyxis/enroot runtime metadata.

- [ ] **Step 1: Write failing attestation tests**

```python
def test_attestation_hashes_the_same_anchor_used_for_container_import() -> None:
    command = build_container_probe(plan())
    assert command.container_image == command.runtime_sqsh_mount_source


def test_missing_node_or_digest_mismatch_blocks_training() -> None:
    receipts = [receipt(node="n0"), receipt(node="n1", image_sha="0" * 64)]
    with pytest.raises(AttestationError, match="node runtime identity"):
        reconcile_runtime_receipts(receipts, expected_nodes=("n0", "n1"))
```

- [ ] **Step 2: Run RED**

```bash
PYTHONDONTWRITEBYTECODE=1 ../q4-ptv2-ab-integration/.venv/bin/python -m pytest -q \
  tools/launcher/tests/test_ptv23_runtime_attestation.py
```

- [ ] **Step 3: Implement in-container receipt generation**

Hash the complete mounted SQSH, contract, and tool; verify the exact sentinel inventory; record node name, architecture, Python identity, CUDA visibility, GPU count, Pyxis/enroot versions, named-container identity, and canonical receipt digest. Reject missing/extra nodes and duplicate node receipts.

- [ ] **Step 4: Implement the platform probe script**

The probe starts one keeper per node with `srun --overlap`, creates `--container-name=q30t-700k-${SLURM_JOB_ID}` on every node from the local anchor, mounts the same anchor at `/run/q30t/runtime.sqsh`, executes attestation, reconciles all node receipts, kills one keeper in the negative phase, and proves the new descriptor-backed launch fails.

- [ ] **Step 5: Run local tests and static checks**

```bash
PYTHONDONTWRITEBYTECODE=1 ../q4-ptv2-ab-integration/.venv/bin/python -m pytest -q \
  tools/launcher/tests/test_ptv23_runtime_attestation.py
bash -n tools/launcher/common/specdec/probe_ptv23_pyxis_keeper.sbatch
shellcheck -x tools/launcher/common/specdec/probe_ptv23_pyxis_keeper.sbatch
```

- [ ] **Step 6: Commit**

```bash
git add tools/launcher/common/specdec/ptv23_runtime_attestation.py \
  tools/launcher/tests/test_ptv23_runtime_attestation.py \
  tools/launcher/common/specdec/probe_ptv23_pyxis_keeper.sbatch
git commit -S -s -m "feat(specdec): attest per-node Pyxis runtime"
```

### Task 3: Make the runner rank-safe and multi-node

**Files:**
- Create: `tools/launcher/common/specdec/run_q30t_ptv23_continuation.sbatch`
- Create: `tools/launcher/tests/test_q30t_multinode_runner.py`
- Modify: `tools/launcher/common/specdec/q30t_ptv23_continuation.py`

**Interfaces:**
- Produces: 16-node runner using per-node keepers and named containers.
- Consumes: Task 1 keeper, Task 2 attestation, and data-contract `ContinuationContract` from the companion plan.

- [ ] **Step 1: Write failing rank-safety tests**

```python
def test_runner_launches_exactly_one_keeper_per_node() -> None:
    runner = parse_runner(RUNNER_PATH)
    assert runner.keeper_step.nodes == 16
    assert runner.keeper_step.tasks == 16
    assert runner.keeper_step.tasks_per_node == 1
    assert runner.keeper_step.overlap is True


def test_only_global_rank_zero_publishes_after_barrier() -> None:
    result = simulate_two_node_run()
    assert result.barrier_nodes == {"n0", "n1"}
    assert result.receipt_publishers == [0]
```

- [ ] **Step 2: Run RED**

```bash
PYTHONDONTWRITEBYTECODE=1 ../q4-ptv2-ab-integration/.venv/bin/python -m pytest -q \
  tools/launcher/tests/test_q30t_multinode_runner.py
```

- [ ] **Step 3: Implement controller, keeper, container, and training phases**

The runner creates one node-local plan per node, launches keepers, waits for 16 exact readiness receipts, creates/probes the named container on every node, reconciles runtime receipts, then launches training with the named container. Every node uses the same `/raid/scratch/$USER/q30t-700k-$SLURM_JOB_ID` pathname but receives its own filesystem. Contract and stage identities are explicit environment variables passed with `--export=NONE` plus an allowlist.

- [ ] **Step 4: Implement rank-safe training evidence**

Extend the training command with global rank/world size, local execution-stage root, and a rank-0 publication flag. All ranks verify local staged input digests and barrier before model construction. Global rank 0 reconciles per-node stage and exposure evidence and performs the exclusive durable receipt write; every other rank exits without publishing.

- [ ] **Step 5: Implement cleanup barriers**

Controller signal handlers terminate and wait for the training process group, close the keeper FIFO, wait for all keepers, and remove only exact private node-local anchors. A keeper or attestation failure terminates the whole allocation before trainer import.

- [ ] **Step 6: Run tests and commit**

```bash
PYTHONDONTWRITEBYTECODE=1 ../q4-ptv2-ab-integration/.venv/bin/python -m pytest -q \
  tools/launcher/tests/test_q30t_multinode_runner.py \
  tools/launcher/tests/test_q30t_ptv23_continuation.py \
  tools/launcher/tests/test_ptv23_node_keeper.py \
  tools/launcher/tests/test_ptv23_runtime_attestation.py
bash -n tools/launcher/common/specdec/run_q30t_ptv23_continuation.sbatch
shellcheck -x tools/launcher/common/specdec/run_q30t_ptv23_continuation.sbatch
git add tools/launcher/common/specdec/run_q30t_ptv23_continuation.sbatch \
  tools/launcher/common/specdec/q30t_ptv23_continuation.py \
  tools/launcher/tests/test_q30t_multinode_runner.py
git commit -S -s -m "feat(specdec): run Q30T continuation on 16 nodes"
```

### Task 4: Add cluster profiles and an idempotent submitter

**Files:**
- Create: `tools/launcher/common/specdec/q30t_ptv23_cluster_profile.py`
- Create: `tools/launcher/common/specdec/submit_q30t_ptv23_continuation.sh`
- Create: `tools/launcher/tests/test_q30t_ptv23_submitter.py`

**Interfaces:**
- Produces: canonical external profile finalizer and `--test-only`, `--submit-canary`, and `--submit-full` modes.
- Consumes: clean signed source commit, contract, reviewed runtime receipts, account/partition identity, and exact output roots.

- [ ] **Step 1: Write failing profile and submission tests**

```python
@pytest.mark.parametrize("cluster,partition", [("lyris", "gb200"), ("ptyche", "36x2-a01r")])
def test_profile_is_16_nodes_with_one_keeper_per_node(cluster: str, partition: str) -> None:
    profile = finalize_profile(cluster=cluster, source_commit=SIGNED_COMMIT)
    assert (profile.partition, profile.nodes, profile.segment, profile.gpus_per_node) == (partition, 16, 16, 4)


def test_submitter_test_only_precedes_real_and_reuses_exact_identity() -> None:
    trace = run_fake_submitter("--submit-canary")
    assert trace.calls[0].kind == "test-only"
    assert trace.real_submissions == 1
    assert trace.duplicate_submissions == 0
```

- [ ] **Step 2: Run RED**

```bash
PYTHONDONTWRITEBYTECODE=1 ../q4-ptv2-ab-integration/.venv/bin/python -m pytest -q \
  tools/launcher/tests/test_q30t_ptv23_submitter.py
```

- [ ] **Step 3: Implement external profile finalization**

Profiles bind cluster, signed source commit, account `coreai_dlalgo_llm`, exact partition, 16 nodes, segment 16, four GPUs/node, walltime, durable root, node-local scratch policy, runner SHA, and runtime probe receipt root. Reject profiles inside the source tree and reject source/profile commit mismatch.

- [ ] **Step 4: Implement idempotent submission**

Preflight all files through no-follow descriptors before durable mutation. Run `sbatch --test-only`, then reserve one canonical submission identity. Adopt an exact existing job; reject ambiguous comments or conflicting receipts. Emit explicit environment allowlist and never inherit `ALL`, `PYTHONPATH`, `BASH_ENV`, or `ENV`.

- [ ] **Step 5: Run tests and commit**

```bash
PYTHONDONTWRITEBYTECODE=1 ../q4-ptv2-ab-integration/.venv/bin/python -m pytest -q \
  tools/launcher/tests/test_q30t_ptv23_submitter.py
bash -n tools/launcher/common/specdec/submit_q30t_ptv23_continuation.sh
shellcheck -x tools/launcher/common/specdec/submit_q30t_ptv23_continuation.sh
git add tools/launcher/common/specdec/q30t_ptv23_cluster_profile.py \
  tools/launcher/common/specdec/submit_q30t_ptv23_continuation.sh \
  tools/launcher/tests/test_q30t_ptv23_submitter.py
git commit -S -s -m "feat(specdec): submit Q30T 700K continuation"
```

### Task 5: Run real platform probes without enabling training

**Files:**
- Modify: `.superpowers/sdd/2026-08-26-q30t-ptv23-complement-700k/runtime-probe-report.md`

**Interfaces:**
- Produces: reviewed one-node and two-node Lyris/Ptyche runtime receipts.
- Consumes: pushed signed branch, finalized external profiles, exact runtime image and SHA, probe script.

- [ ] **Step 1: Verify, sign, and push before cluster access**

```bash
git status --short
git log -1 --show-signature
git push -u gitlab sj/q30t-ptv23-complement-700k
```

The worktree must be clean and the remote commit must equal local HEAD.

- [ ] **Step 2: Stage source locally on each cluster**

Use one SSH session per cluster. Run `git pull --ff-only`, verify the signed commit, and keep the repository under `/home`. Store probe image and durable receipts under the profile's `/lustre` root; use `/raid/scratch` only inside jobs.

- [ ] **Step 3: Run scheduler test-only for one-node probe**

Use the finalized profile and exact probe identity. Submit no real job unless test-only succeeds and there is no active exact duplicate.

- [ ] **Step 4: Submit and monitor the one-node probe**

Submit one probe on Lyris, monitor the exact job at intervals of at least 60 seconds for five minutes, and validate the receipt. Repeat on Ptyche only after the Lyris result is classified.

- [ ] **Step 5: Submit and monitor the two-node probe**

Require the one-node receipt as an input. Prove distinct node-local keepers, named-container reuse, complete SQSH digest, and keeper-loss fail-closed behavior. Repeat for the other cluster.

- [ ] **Step 6: Review the runtime evidence**

If any cluster fails `/proc/PID/fd`, Yama/hidepid, `--overlap`, named-container, or full-image hash requirements, leave runtime attestation disabled for that cluster. Do not substitute named files while claiming descriptor equivalence.

- [ ] **Step 7: Commit the probe report**

```bash
git add .superpowers/sdd/2026-08-26-q30t-ptv23-complement-700k/runtime-probe-report.md
git commit -S -s -m "docs(specdec): record Q30T runtime probe evidence"
```

### Task 6: Submit 20-step canaries only after both tracks are clean

**Files:**
- Modify: `.superpowers/sdd/2026-08-26-q30t-ptv23-complement-700k/task-report.md`

**Interfaces:**
- Produces: one DFlash and one DSpark 20-step canary receipt, or a precise fail-closed blocker.
- Consumes: independently reviewed data/contract freeze, runtime probe receipt, parent receipts, Q30 tokenizer receipt, target receipt, trainer, runtime image, and signed pushed source.

- [ ] **Step 1: Run full local and static verification**

```bash
PYTHONDONTWRITEBYTECODE=1 ../q4-ptv2-ab-integration/.venv/bin/python -m pytest -q \
  tests/examples/dataset/test_build_qwen4b_ptv23_complement.py \
  tests/examples/dataset/test_ptv23_complement_target_policy.py \
  tests/examples/dataset/test_trajectory_schema.py \
  tools/launcher/tests/test_q30t_tokenizer_receipt.py \
  tools/launcher/tests/test_q30t_parent_receipt.py \
  tools/launcher/tests/test_q30t_ptv23_continuation.py \
  tools/launcher/tests/test_ptv23_node_keeper.py \
  tools/launcher/tests/test_ptv23_runtime_attestation.py \
  tools/launcher/tests/test_q30t_multinode_runner.py \
  tools/launcher/tests/test_q30t_ptv23_submitter.py
../q4-ptv2-ab-integration/.venv/bin/ruff check --no-fix examples/dataset tools/launcher/common/specdec tools/launcher/tests
pyright examples/dataset tools/launcher/common/specdec
bash -n tools/launcher/common/specdec/run_q30t_ptv23_continuation.sbatch \
  tools/launcher/common/specdec/submit_q30t_ptv23_continuation.sh
git diff --check
```

- [ ] **Step 2: Obtain a fresh hostile review**

Freeze exact file hashes and request a reviewer who did not implement either track. Production eligibility requires zero critical and zero warning findings.

- [ ] **Step 3: Push the reviewed signed commit**

Push only after the frozen hash survives verification and review unchanged.

- [ ] **Step 4: Submit DFlash canary**

Run test-only, verify no duplicate, submit the exact DFlash contract, and monitor for five minutes. Validate step-20 exposure, output checkpoint, per-node runtime identities, fresh optimizer/scheduler, and receipt.

- [ ] **Step 5: Submit DSpark canary**

Repeat with a distinct run/output identity and the authenticated DSpark parent. Never share writable roots with DFlash.

- [ ] **Step 6: Stop at the canary approval gate**

Record both receipts and request independent result review. Do not submit 1,368-step full jobs until both canary receipts are added to the immutable approval root and the resulting source bytes are reverified, signed, pushed, and test-only checked.
