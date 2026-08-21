# Multi-Cluster Drafter Standby Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the existing ModelOpt Qwen3 drafter training and evaluation workflow reproducibly stageable and testable on Lyris and Pre-Tyche without changing OCI-HSG behavior.

**Architecture:** Introduce one typed cluster profile consumed by the existing staging, training, and evaluation entrypoints. Keep source under `/home`, immutable large artifacts under each cluster's durable root, and all mutable runtime state under a compute-node-discovered scratch root. Transfer large artifacts through checksum-verified PDX data-mover jobs, then qualify each cluster with provenance, evaluator, and 20-step training canaries before allowing production activation.

**Tech Stack:** Python 3.12, Bash, SLURM, Pyxis/Enroot, pytest, ModelOpt, vLLM 0.27.1, PDX/PBSS rclone.

**Spec:** `docs/superpowers/specs/2026-08-20-multicluster-drafter-standby-design.md`

## Global Constraints

- Preserve ModelOpt source commit `e3febcbe1319f018eea81fa4d42e2e36cb54494e` as the initial standby pin.
- Preserve the existing 32-experiment matrix, DFlash/DSpark B/K mappings, GBS 512, `save_steps=50`, `save_total_limit=2`, maximum 50 same-ID requeues, step-4166 milestone, and step-25391 final artifact.
- Training uses four exclusive 4-GPU nodes with `--segment=4`; staging and evaluation use one exclusive 4-GPU node with `--segment=1`.
- Lyris uses partition `gb200`; Pre-Tyche prefers `36x2-a01r` with `batch` fallback; both use account `coreai_dlalgo_llm` after live verification.
- Lyris and Pre-Tyche do not receive `--gres` or `--gpus-per-node` flags.
- Source and Git metadata live under `/home`; durable images, models, datasets, checkpoints, and results live under the profile's Lustre/project root.
- Mutable runtime, cache, lock, database, and temporary log paths use the compute-node-discovered scratch root.
- Cross-cluster data moves run through one-node checksum-verified PDX data-mover jobs; login nodes never copy large artifacts inline.
- A standby qualification stops after test-only and canaries; production activation remains explicit and cluster-isolated.
- Every code commit is SSH-signed and includes `Signed-off-by`.

---

### Task 1: Typed Cluster Profiles

**Files:**
- Create: `tools/launcher/common/specdec/cluster_profile.py`
- Create: `tools/launcher/common/specdec/profiles/oci-hsg.yaml`
- Create: `tools/launcher/common/specdec/profiles/lyris.yaml`
- Create: `tools/launcher/common/specdec/profiles/ptyche.yaml`
- Test: `tools/launcher/tests/test_drafter_cluster_profiles.py`

**Interfaces:**
- Consumes: a profile YAML path.
- Produces: `ClusterProfile`, `load_cluster_profile(path: Path) -> ClusterProfile`, `scheduler_gpu_args(profile: ClusterProfile) -> tuple[str, ...]`, and `validate_scratch_root(profile: ClusterProfile, path: Path) -> None`.

- [ ] **Step 1: Write failing profile tests**

```python
def test_ptyche_profile_uses_exclusive_four_gpu_nodes() -> None:
    profile = load_cluster_profile(PROFILES / "ptyche.yaml")
    assert profile.account == "coreai_dlalgo_llm"
    assert profile.partition == "36x2-a01r"
    assert profile.training_nodes == 4
    assert profile.training_segment == 4
    assert scheduler_gpu_args(profile) == ()


def test_oci_profile_preserves_gpus_per_node() -> None:
    profile = load_cluster_profile(PROFILES / "oci-hsg.yaml")
    assert scheduler_gpu_args(profile) == ("--gpus-per-node=4",)
```

- [ ] **Step 2: Run RED**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest -q \
  tools/launcher/tests/test_drafter_cluster_profiles.py
```

Expected: collection fails because `cluster_profile` and profile files do not exist.

- [ ] **Step 3: Implement immutable typed profiles**

```python
@dataclass(frozen=True)
class ClusterProfile:
    name: str
    ssh_host: str
    account: str
    partition: str
    fallback_partition: str | None
    durable_root: Path
    scratch_candidates: tuple[Path, ...]
    training_nodes: int
    training_segment: int
    evaluation_nodes: int
    evaluation_segment: int
    gpus_per_node: int
    explicit_gpu_flag: bool
    walltime: str
```

Validate absolute paths, positive topology values, segment divisibility, the 18-node maximum segment, and exact exclusive-node GPU semantics.

- [ ] **Step 4: Run GREEN and static checks**

Run the focused pytest command, then:

```bash
.venv/bin/ruff check tools/launcher/common/specdec/cluster_profile.py \
  tools/launcher/tests/test_drafter_cluster_profiles.py
.venv/bin/ruff format --check tools/launcher/common/specdec/cluster_profile.py \
  tools/launcher/tests/test_drafter_cluster_profiles.py
```

- [ ] **Step 5: Commit Task 1**

```bash
git add tools/launcher/common/specdec/cluster_profile.py \
  tools/launcher/common/specdec/profiles tools/launcher/tests/test_drafter_cluster_profiles.py
git commit -s -S -m "feat(launcher): add drafter cluster profiles"
```

### Task 2: Compute-Node Scratch and Scheduler Rendering

**Files:**
- Create: `tools/launcher/common/specdec/probe_cluster_profile.sh`
- Modify: `tools/launcher/common/specdec/cluster_profile.py`
- Modify: `tools/launcher/tests/test_drafter_cluster_profiles.py`

**Interfaces:**
- Consumes: `--profile PATH --output PATH`.
- Produces: an atomic readiness JSON with resolved scratch root, scheduler account/partition, Pyxis availability, GPU count, architecture, hostname, and timestamp.

- [ ] **Step 1: Add RED tests for scratch selection and rendered flags**

```python
def test_select_scratch_prefers_slurm_tmpdir(tmp_path: Path) -> None:
    selected = select_scratch_root(
        candidates=(Path("$SLURM_TMPDIR"), Path("/raid/scratch"), Path("/tmp")),
        environ={"SLURM_TMPDIR": str(tmp_path)},
        writable=lambda path: path == tmp_path,
    )
    assert selected == tmp_path


def test_lyris_render_has_no_gres_or_gpu_flag() -> None:
    argv = render_probe_sbatch(load_cluster_profile(PROFILES / "lyris.yaml"))
    assert "--segment=1" in argv
    assert not any(arg.startswith("--gres") or arg.startswith("--gpus-per-node") for arg in argv)
```

- [ ] **Step 2: Run RED** and confirm the missing functions/script are the cause.

- [ ] **Step 3: Implement the probe** so the outer login-node command performs only filtered account/partition checks and `sbatch --test-only`; the compute-node payload chooses and creates scratch, checks four GPUs and ARM64, and writes only the small readiness receipt to durable storage.

- [ ] **Step 4: Run GREEN**, `bash -n`, ShellCheck at warning severity, Ruff, and `git diff --check`.

- [ ] **Step 5: Commit Task 2**

```bash
git add tools/launcher/common/specdec/probe_cluster_profile.sh \
  tools/launcher/common/specdec/cluster_profile.py \
  tools/launcher/tests/test_drafter_cluster_profiles.py
git commit -s -S -m "feat(launcher): probe drafter cluster readiness"
```

### Task 3: Profile-Aware Staging and Runtime Probes

**Files:**
- Modify: `tools/launcher/common/specdec/stage_hf_model.sh`
- Modify: `tools/launcher/common/specdec/stage_relocatable_runtime_archive.sh`
- Modify: `tools/launcher/common/specdec/stage_speculators_eval_runtime.sh`
- Modify: `tools/launcher/common/specdec/probe_relocatable_runtime.sh`
- Test: `tools/launcher/tests/test_drafter_submission.py`
- Test: `tools/launcher/tests/test_speculators_eval.py`

**Interfaces:**
- Consumes: existing CLI plus `--cluster-profile PATH` and a readiness receipt from Task 2.
- Produces: the same immutable archives/artifacts and manifests as OCI, with scheduler flags and scratch roots derived from the profile.

- [ ] **Step 1: Add RED behavioral tests** proving OCI output is unchanged, Lyris/Pre-Tyche omit GPU request flags, every mutable path uses the resolved compute scratch root, and `/raid/scratch` is not hard-coded outside the OCI profile.

- [ ] **Step 2: Run the focused tests** and verify only the new profile cases fail.

- [ ] **Step 3: Parameterize the four scripts** while preserving all existing SHA, clean-source, atomic publication, and `sbatch --test-only` gates. Reject a missing/mismatched readiness receipt before submission.

- [ ] **Step 4: Run GREEN** for both test files plus Bash syntax, ShellCheck, Ruff, and forbidden-pattern checks for Lustre caches/builds and login-node copies.

- [ ] **Step 5: Commit Task 3**

```bash
git add tools/launcher/common/specdec/stage_hf_model.sh \
  tools/launcher/common/specdec/stage_relocatable_runtime_archive.sh \
  tools/launcher/common/specdec/stage_speculators_eval_runtime.sh \
  tools/launcher/common/specdec/probe_relocatable_runtime.sh \
  tools/launcher/tests/test_drafter_submission.py \
  tools/launcher/tests/test_speculators_eval.py
git commit -s -S -m "feat(launcher): make drafter staging profile aware"
```

### Task 4: Profile-Aware Training and Evaluation

**Files:**
- Modify: `tools/launcher/common/specdec/run_drafter_training.sbatch`
- Modify: `tools/launcher/common/specdec/submit_drafter_training_wave.sh`
- Modify: `tools/launcher/common/specdec/submit_drafter_full_chain.sh`
- Modify: `tools/launcher/common/specdec/run_speculators_eval.sh`
- Modify: `tools/launcher/tests/test_drafter_submission.py`
- Modify: `tools/launcher/tests/test_speculators_eval.py`

**Interfaces:**
- Consumes: `CLUSTER_PROFILE`, the Task 2 readiness receipt, canonical training/evaluation manifests, and existing immutable artifact paths.
- Produces: cluster-isolated scheduler jobs, receipts, result roots, and W&B identities without changing OCI defaults.

- [ ] **Step 1: Add RED tests** for Lyris and Pre-Tyche scheduler argv, cluster-suffixed W&B/run IDs, cluster-specific receipt/output roots, Q30 dual-TP2 and Q235 TP4 full utilization, and refusal to share an active OCI output namespace.

- [ ] **Step 2: Add RED tests** proving `save_steps=50`, `save_total_limit=2`, maximum 50 requeues, permanent step 4166/final 25391, and existing 112-stage boundaries remain identical under all profiles.

- [ ] **Step 3: Run RED** and record only profile-related failures.

- [ ] **Step 4: Implement profile consumption** in the runner and submitters. Resolve scratch from the signed readiness receipt, render GPU flags conditionally, preserve `--segment=4/1`, and include cluster name in training fingerprint, W&B group, output root, comment, and receipt identity.

- [ ] **Step 5: Run GREEN** for the launcher/evaluator suites, then Bash syntax, ShellCheck, Ruff, formatting, and `git diff --check`.

- [ ] **Step 6: Commit Task 4**

```bash
git add tools/launcher/common/specdec/run_drafter_training.sbatch \
  tools/launcher/common/specdec/submit_drafter_training_wave.sh \
  tools/launcher/common/specdec/submit_drafter_full_chain.sh \
  tools/launcher/common/specdec/run_speculators_eval.sh \
  tools/launcher/tests/test_drafter_submission.py \
  tools/launcher/tests/test_speculators_eval.py
git commit -s -S -m "feat(launcher): run drafter workflows across GB200 clusters"
```

### Task 5: Checksum-Verified Cross-Cluster Artifact Transfer

**Files:**
- Create: `tools/launcher/common/specdec/drafter_transfer_manifest.py`
- Create: `tools/launcher/common/specdec/stage_drafter_artifacts_pdx.sh`
- Test: `tools/launcher/tests/test_drafter_transfer.py`

**Interfaces:**
- Consumes: `TransferManifest` JSON entries with `name`, `source_path`, `bucket_path`, `destination_path`, `size_bytes`, `sha256`, and `artifact_kind`.
- Produces: atomic destination artifacts, per-entry completion records, and one aggregate readiness manifest.

- [ ] **Step 1: Write RED tests** for canonical manifests, path containment, duplicate destinations, checksum mismatch, partial publication, existing matching artifacts, and conflict refusal.

```python
@dataclass(frozen=True)
class TransferEntry:
    name: str
    source_path: Path
    bucket_path: str
    destination_path: Path
    size_bytes: int
    sha256: str
    artifact_kind: Literal["container", "runtime", "model", "dataset", "checkpoint"]
```

- [ ] **Step 2: Run RED** and confirm missing transfer interfaces.

- [ ] **Step 3: Implement manifest validation and the wrapper** around the bundled PDX upload/download scripts. The wrapper submits one-node data-mover jobs, uses a content-addressed bucket prefix, publishes only after checksum success, and never stores credentials in argv, manifests, logs, or Git.

- [ ] **Step 4: Run GREEN**, Bash syntax, ShellCheck, Ruff, formatting, secret-pattern scans, and `git diff --check`.

- [ ] **Step 5: Commit Task 5**

```bash
git add tools/launcher/common/specdec/drafter_transfer_manifest.py \
  tools/launcher/common/specdec/stage_drafter_artifacts_pdx.sh \
  tools/launcher/tests/test_drafter_transfer.py
git commit -s -S -m "feat(launcher): stage drafter artifacts through PDX"
```

### Task 6: Standby Qualification and Activation Guard

**Files:**
- Create: `tools/launcher/common/specdec/qualify_drafter_standby.sh`
- Create: `tools/launcher/common/specdec/activate_drafter_standby.sh`
- Modify: `tools/launcher/examples/Qwen/Qwen3-30B-A3B/README.md`
- Test: `tools/launcher/tests/test_drafter_standby.py`

**Interfaces:**
- Consumes: cluster profile, probe receipt, transfer readiness manifest, canonical 32-experiment manifest, and `--activate` for production submission.
- Produces: one readiness receipt after provenance/evaluator/20-step training canaries and, only with explicit activation, a cluster-isolated full-chain submission receipt.

- [ ] **Step 1: Write RED orchestration tests** for gate order, failure propagation, `sbatch --test-only`, five-minute monitor metadata, full-32 render, no default production submission, and active-output split-brain refusal.

- [ ] **Step 2: Run RED** and confirm both scripts are absent.

- [ ] **Step 3: Implement qualification** in this exact order: live profile probe, artifact verification, runtime smoke, evaluator CSV smoke, four-node `max_steps=20/save_steps=20` training canary, and full manifest test-only render.

- [ ] **Step 4: Implement guarded activation** that requires the completed readiness receipt, a single filtered scheduler snapshot, a distinct cluster output namespace, and an explicit `--activate` flag. It must never cancel OCI jobs or transfer a mutable checkpoint implicitly.

- [ ] **Step 5: Document exact Lyris and Pre-Tyche commands**, MFA/Kerberos prerequisites, account/partition differences, PDX transfer flow, readiness paths, and disaster-recovery activation.

- [ ] **Step 6: Run GREEN** plus all launcher/evaluator regression suites and static checks.

- [ ] **Step 7: Commit Task 6**

```bash
git add tools/launcher/common/specdec/qualify_drafter_standby.sh \
  tools/launcher/common/specdec/activate_drafter_standby.sh \
  tools/launcher/examples/Qwen/Qwen3-30B-A3B/README.md \
  tools/launcher/tests/test_drafter_standby.py
git commit -s -S -m "feat(launcher): qualify and activate drafter standby"
```

### Task 7: Review, Publish, and Cluster Qualification

**Files:**
- Review: all files from Tasks 1-6.

**Interfaces:**
- Consumes: reviewed implementation and cluster credentials.
- Produces: exact pushed ModelOpt SHA plus Lyris and Pre-Tyche readiness receipts; no production standby training jobs.

- [ ] **Step 1: Run fresh local verification**

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest -q \
  tools/launcher/tests/test_drafter_cluster_profiles.py \
  tools/launcher/tests/test_drafter_transfer.py \
  tools/launcher/tests/test_drafter_standby.py \
  tools/launcher/tests/test_drafter_submission.py \
  tools/launcher/tests/test_speculators_eval.py
```

Run pre-commit on every owned file, Bash syntax, ShellCheck warning severity,
Ruff lint/format, and `git diff --check`.

- [ ] **Step 2: Request independent reviews** for scheduler safety, scratch/storage compliance, transfer atomicity, credential redaction, resume isolation, and split-brain prevention. Fix Critical/Important findings with RED/GREEN tests and repeat review.

- [ ] **Step 3: Create a final signed/DCO commit**, verify its signature, and push only after current-turn user authorization.

- [ ] **Step 4: Pre-Tyche qualification**

Use `login-ptyche`, `coreai_dlalgo_llm`, and `36x2-a01r`. Pull the exact commit,
run the live probe, transfer missing artifacts, verify hashes, run all
`sbatch --test-only` gates, submit the evaluator and 20-step training canaries,
and monitor each running GPU job for at least five minutes.

- [ ] **Step 5: Lyris qualification**

After the user establishes an MS MFA ControlMaster to `login-lyris`, verify the
live account/partition/FairShare, then repeat the exact source, transfer,
test-only, evaluator, and 20-step training gates using partition `gb200`.

- [ ] **Step 6: Report readiness** with exact commits, artifact SHAs, job IDs,
W&B/evaluator links, result paths, scheduler observations, and any remaining
authentication or asset blockers. Do not activate production standby jobs.
