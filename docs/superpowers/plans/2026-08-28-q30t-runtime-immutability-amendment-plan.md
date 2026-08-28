# Q30T Runtime Immutability Amendment Implementation Plan

> **For Codex:** Execute this plan task by task with
> `superpowers:executing-plans`. Use `superpowers:test-driven-development` for
> every behavior change and stop at every external or review gate.

**Goal:** Produce, qualify, independently review, and approve one reproducible
derived Q30T SQSH installed at `/opt/q30t-runtime`, executed by one global
Slurm step from per-node protected leased inodes without mutable extraction,
checkout imports, named Pyxis roots, deletion, or unbounded shared logs, with
live reviewed 16-node/segment-16 admission for Q30 canary/full.

**Architecture:** A feasibility-only commit F first proves a real root-owned
Ptyche gateway/build service, exact Pyxis direct SquashFUSE behavior, per-node
same-path routing, distributed GO/ABORT, kernel leases, remote-death evidence,
site config, and bounded logs through a prerequisite-ordered live 1 -> 2 -> 16
sequence. The exact reviewed sixteen-node tuple is required for segment-16 Q30
canary/full admission. If F fails, record BLOCKED and do not create P.
P then pins the reviewed F receipt, exact trust roots, and a literal path/blob
map. A protected hermetic service performs two isolated builds and emits full
content manifests. Per-node agents create local anonymous keeper inodes, pass
same-OFD fds with `SCM_RIGHTS`, and have the gateway link them at the same
absolute protected path. One global Pyxis `srun` mounts those local paths
read-only in unnamed direct mode. A gateway barrier holds every wrapper after
READY until global GO or ABORT. P has empty review approvals; signed independent
review precedes constants-only descendant A.

Every committed Python control blob is separately published by the gateway at
an identical root-owned per-node path for the global step. The bootstrap
stable-opens that protected tree and explicitly compiles/imports only the
declared module graph; anonymous per-node control descriptors never appear in
global argv.

**Tech stack:** Python 3.12+, Linux `O_TMPFILE`/`linkat`/file leases/
`SCM_RIGHTS`/pidfd/cgroup v2, Ed25519 gateway signatures, OpenSSH allowed
signers, Bash, Slurm, Pyxis/enroot/SquashFUSE, SquashFS, canonical JSON/SHA-256,
pytest, Ruff, Pyright, ShellCheck.

## Non-Negotiable Gates

- Base SHA is exactly
  `e7be53f2754097c88f7c801da92f6d94794ec4d78d9df937fcd315a6994297f0`.
- Archive SHA is exactly
  `4a20aee61f290c48bed22a84b4a0ae0cbdc54e3e3910854d253188c8854f5dc9`.
- The archive installs only at `/opt/q30t-runtime`; row and later training use
  the same approved SQSH.
- F is a separately reviewed, feasibility-only pushed commit. It has no import
  or write path to runtime approvals and cannot emit runtime approval receipts.
- F uses three separate test-only-first real jobs in exact 1 -> 2 -> 16 order;
  the signed v3 aggregate and independent review bind the exact sixteen nodes,
  per-node gateway endpoints, one global 16-task `srun`, and 16-rank collective.
- A real administrator-owned gateway/build service is mandatory. Missing
  service, key, protected ownership, path identity, or capability yields
  BLOCKED before P. A mock, `0700` same-UID path, or remote assumption is not a
  substitute.
- Build A/B and review A/B use distinct service protection domains and complete
  entry-content manifests. No host SquashFS executable is trusted.
- Every keeper starts with an anonymous mode-0400 inode, closes writers,
  acquires `F_RDLCK`, sends the same OFD by `SCM_RIGHTS`, and retains the lease
  through global step exit proof.
- Pyxis receives one identical absolute protected regular pathname across
  nodes. It uses one global execution step, direct unnamed `squashfuse`,
  `--container-readonly`, no home/remap/entrypoint, and audited system config.
  There is no procfd image, named rootfs, nested execution, or extracted reuse.
- Every wrapper waits after authenticated READY. Positive operations get one
  global GO only after all nodes are ready. Attack operations use the literal
  policy `ABORT_ONLY`, start same-UID blocking writes while held, and never get
  GO.
- Lease break cancels the exact global step and proves every node's
  `cgroup.events` reports `populated 0`,
  global `srun` reaped, exact step terminal, and allocation state before the
  lease deadline. Accepted `scancel` or local process-group death is not proof.
- Live logs remain node-local and bounded before stable durable publication.
  No `%j.out` or Pyxis/application stream is unbounded on Lustre.
- Every external mode runs the identical `sbatch --test-only` argv before one
  possible `sbatch --parsable`. This planning work performs neither call.
- Producer P has empty approval roots. A changes only the two frozenset
  literals of `(subject hash, review hash, signature hash)` tuples.

## Dependency Order

| Gate | Produces | Unlocks | Required evidence |
| --- | --- | --- | --- |
| Task 0 | local F implementation | Task 1 | focused RED/GREEN, F isolation scan |
| Task 1 | signed reviewed live F receipt or BLOCKED | Task 2 | 1-/2-/16-node capability artifacts and review signature |
| Task 2 | blob-map schema and trust loaders, no production selector values | Tasks 3-7 | fixture signature/replay attacks GREEN |
| Task 3 | protected build + v2 receipt | Tasks 6-7 | A/B byte/content equality and service attestations |
| Task 4 | same-OFD keeper/gateway channel | Tasks 5-7 | Linux SCM/event attacks GREEN |
| Task 5 | global Pyxis barrier/supervisor | Tasks 6-7 | one-global-step and remote-death attacks GREEN |
| Task 6 | v3 node/aggregate attestation | Task 7 | loader/cardinality/lineage attacks GREEN |
| Task 7 | integrated operations and consumers | Task 8 | fake Slurm/Pyxis and source-policy GREEN |
| Task 8 | signed reviewed P with empty roots | Task 9 | full gates and hostile review CLEAN |
| Task 9 | physical reviews and constants-only A | consumers | exact signed review chain and P→A diff |

## Hostile-Review Closure Matrix

Each finding has a design owner, a RED command that must first fail, and a
GREEN command that must later pass:

| Finding | Structural resolution | RED/GREEN owner |
| --- | --- | --- |
| C1 | SCM_RIGHTS duplicate is the sole same-OFD lease proof; peer credential, pidfd/start, and Yama gates authenticate it | Task 0 Steps 1/7; Task 4 Steps 2-5 |
| C2 | root-owned gateway links the held inode to one protected regular path; Pyxis direct unnamed mode only | Task 0 Steps 2-3/7; Task 5 Steps 1/5 |
| C3 | per-node agents stage distinct inodes at identical path text; one later global execution step uses them | Task 0 Steps 3/7; Task 5 Steps 1-5; Task 7 Steps 1/6 |
| C4 | exact read-only/no-home/no-remap/no-entrypoint argv, constructed mounts, exact env, recursive system-config manifest, protected empty user config | Task 0 Steps 3/7; Task 5 Steps 1/5 |
| C5 | build A/B run in service protection domains and replay complete base/archive/derived content manifests | Task 3 Steps 1-5 |
| C6 | private credentialed seqpacket carries same-OFD and event packets; Yama denies sibling duplication; drain/withhold attacks fail closed | Task 0 Steps 1/7; Task 4 Steps 3/5 |
| C7 | exact global step cancellation plus every cgroup empty, global reap, scheduler terminal, and allocation state before lease release | Task 0 Steps 4/7; Task 5 Steps 3/5 |
| C8 | installed distributed barrier holds all wrappers after READY; attack registration is permanently ABORT-only and never GO | Task 0 Steps 4/7; Task 5 Steps 2/5 |
| W1 | F-pinned hermetic builder image binds entrypoint, interpreter, DT_NEEDED libraries, config, and tools | Task 3 Steps 2/5 |
| W2 | source literal path-to-profile-field schema is fixed in Task 2; final values are reconstructed from Task 7's exact index at P and replayed again at A | Task 2 Steps 1/4; Task 7 Steps 7-8; Task 9 Step 5 |
| W3 | pinned allowed-signers trust root verifies signed exact subject/rebuild/prerequisite bindings | Task 2 Steps 2/4; Task 9 Steps 1-4 |
| W4 | separately pushed F performs live one-/two-/sixteen-node platform proofs before P; absent admin service is BLOCKED | Tasks 0-1 |
| W5 | all live streams are node-local and capped before stable bounded publication; no Lustre `%j.out` | Task 0 Steps 5/7; Task 7 Steps 2/6 |

### Round-2 closure matrix

| Finding | Structural resolution | RED/GREEN owner |
| --- | --- | --- |
| R2-C1 | F's one-/two-/sixteen-node run receipts record gateway/build socket device/inode/path, expected root UID and service GID, systemd identity, executable, public key, and hashes; clients reconstruct both endpoints and trust roots only from reviewed F | Task 0 Steps 2/6-7; Task 2 Steps 2/4 |
| R2-C2 | gateway copies exact committed control fds into fresh root-owned 0555/0444 identical per-node trees; receipts bind every local device/inode and common path/content manifest; the bootstrap uses no-follow `openat2`, retained fds, `compile`, and explicit `importlib` | Task 0 Steps 2-3/6-7; Task 4 Steps 1/4-5; Task 7 Steps 1/6 |
| R2-W1 | pre-spawn socketpairs use `SO_PASSCRED`; every READY/event `recvmsg` contains kernel `SCM_CREDENTIALS`, joined to a post-spawn pidfd/start-ticks identity; cached socketpair `SO_PEERCRED` is rejected | Task 4 Steps 2-5 |
| R2-W2 | canonical keeper-loss v3 evidence joins gateway pidfd exit, credentialed agent EOF, signed ABORT, exact cancellation, remote death, no GO, and no attacked receipt in fixed operation order | Task 5 Steps 2-5; Task 6 Steps 1-4 |
| R2-W3 | lease-break v2 carries ordered physical/self hashes for one persisted packet per node and per-node joins across keeper/image/attacker/timestamps/one remote-exit proof | Task 5 Steps 2-5; Task 6 Steps 1-4 |
| R2-W4 | F CLI requires node count; test-only/real submissions follow exact 1 -> 2 -> 16 receipt prerequisites with three distinct job IDs and typed run/aggregate loaders | Task 0 Steps 5-7; Task 1 Steps 1-6 |
| R2-W5 | Task 7's signed implementation commit is P; Task 8 reviews/pushes that unchanged HEAD; Task 9 stages only the approval constants before A | Task 7 Step 8; Task 8 Steps 2-4; Task 9 Steps 5-6 |
| R2-W6 | keeper blocks SIGIO before creating threads, sets `F_SETOWN_EX`/`F_SETSIG`, arms a `sigwaitinfo` thread, proves all masks, then acquires the lease; boundary injections fail closed | Task 4 Steps 2-5 |

### Round-3 closure matrix

| Finding | Structural resolution | RED/GREEN owner |
| --- | --- | --- |
| R3-C1 | initial F runs and independently reviews exact live 1 -> 2 -> 16 receipts; v3 binds the 16-node tuple/endpoints/global 16-task step, and admission requires it for segment-16 canary/full | Task 0 Steps 3/5-7; Task 1 Steps 1-6; Task 7 Steps 3/6 |
| R3-W1 | Task 2 freezes only the literal schema; after every mapped Task 4-7 source byte is final, Task 7 computes production selector hashes, writes the profile once, stages it, and reruns lineage tests before P | Task 2 Steps 1-5; Task 7 Steps 6-8 |
| R3-W2 | Task 7 stages an explicit P-owned path array, requires cached names equal that array, asserts no unstaged P-path change, and leaves unrelated dirty paths untouched before signed+DCO P | Task 7 Step 8 |

## Planned Files

Create or modify only these production paths while executing the future plan:

- `tools/launcher/common/specdec/q30t_runtime_platform_feasibility.py`
- `tools/launcher/common/specdec/q30t_runtime_gateway_client.py`
- `tools/launcher/common/specdec/q30t_runtime_content_manifest.py`
- `tools/launcher/common/specdec/q30t_anonymous_publish.py`
- `tools/launcher/common/specdec/q30t_derived_runtime_image.py`
- `tools/launcher/common/specdec/q30t_runtime_independent_review.py`
- `tools/launcher/common/specdec/q30t_runtime_review_allowed_signers`
- `tools/launcher/common/specdec/q30t_runtime_admission.py`
- `tools/launcher/common/specdec/q30t_runtime_bootstrap.py`
- `tools/launcher/common/specdec/q30t_ptv23_cluster_profile.py`
- `tools/launcher/common/specdec/ptv23_node_keeper.py`
- `tools/launcher/common/specdec/ptv23_keeper_supervisor.py`
- `tools/launcher/common/specdec/ptv23_runtime_attestation.py`
- `tools/launcher/common/specdec/q30t_runtime_approvals.py`
- `tools/launcher/common/specdec/probe_ptv23_pyxis_keeper.sbatch`
- `tools/launcher/common/specdec/run_q30t_runtime_platform_feasibility.sbatch`
- `tools/launcher/common/specdec/submit_q30t_runtime_platform_feasibility.sh`
- `tools/launcher/common/specdec/run_q30t_derived_runtime_image.sbatch`
- `tools/launcher/common/specdec/submit_q30t_runtime_qualification.sh`
- `tools/launcher/common/specdec/q30t_ptv23_continuation.py`
- `tools/launcher/common/specdec/run_q30t_ptv23_continuation.sbatch`
- `tools/launcher/common/specdec/submit_q30t_ptv23_continuation.sh`
- `tools/launcher/common/specdec/q30t_row_observation_runtime.py`
- `tools/launcher/common/specdec/q30t_row_observation_bootstrap.py`
- `tools/launcher/common/specdec/run_q30t_row_schema_observation.sbatch`
- `tools/launcher/common/specdec/submit_q30t_row_schema_observation.sh`
- `examples/dataset/observe_q30t_ptv23_row_schemas.py`

Each module has one matching `tools/launcher/tests/test_*.py`; shell paths have
submitter/runner tests. Existing row-runtime tests are extended, not bypassed.

## Task 0: Implement the Feasibility-Only F Producer Locally

**Files:**

- Create the feasibility/gateway runner and submitter paths listed above.
- Create the anonymous publisher, platform-feasibility review kind, and exact
  allowed-signers trust-root paths; F has no non-feasibility review kind.
- Create `tools/launcher/tests/test_q30t_runtime_platform_feasibility.py`.
- Create `tools/launcher/tests/test_q30t_runtime_gateway_client.py`.
- Create `tools/launcher/tests/test_q30t_runtime_platform_feasibility_runner.py`.
- Create `tools/launcher/tests/test_q30t_runtime_platform_feasibility_submitter.py`.

### Step 1: Write RED same-OFD and process-identity tests

Add Linux tests using a real `SOCK_SEQPACKET` socketpair and `SCM_RIGHTS`:

```python
def test_scm_rights_duplicate_observes_keeper_lease(tmp_path: Path) -> None:
    fixture = make_leased_tmpfile_fixture(tmp_path)
    received_fd = receive_one_fd(fixture.receiver)
    assert fcntl.fcntl(received_fd, fcntl.F_GETLEASE) == fcntl.F_RDLCK


def test_procfd_reopen_is_rejected_as_same_ofd_proof(tmp_path: Path) -> None:
    fixture = make_leased_tmpfile_fixture(tmp_path)
    reopened_fd = os.open(f"/proc/{fixture.pid}/fd/{fixture.fd}", os.O_RDONLY)
    with pytest.raises(FeasibilityError, match="same open file description"):
        prove_keeper_same_ofd(reopened_fd, fixture.ready)
```

Add real pidfd/start-ticks cases for PID mismatch/reuse and a sibling
`pidfd_getfd` attack. The test skips only when the host kernel lacks the syscall;
the live F probe never skips and reports unsupported.
Use a real post-spawn child to show cached pre-spawn socketpair `SO_PEERCRED`
is the creator PID, while `SO_PASSCRED` supplies the child's kernel
`SCM_CREDENTIALS`; require UID/GID and PID to join the child's pidfd/start
ticks. Missing, duplicate, creator-only, or mismatched credentials fail.
Also prove endpoint replay: construct one gateway endpoint identity per ordered
node plus the build-service coordinator identity, with exact socket
path/device/inode/uid/gid/mode, systemd identity,
executable hash, public-key path/hash, and identity self hash. Reject a
caller-selected socket/key/root, swapped service identities, changed live
socket inode, service restart, wrong `SO_PEERCRED`, `/proc/{pid}/exe`, cgroup,
or expected UID/GID. Require the gateway reconstruct/connect functions to take
an F-approved node name, and both service clients to consume only the
stable-loaded reviewed F receipt. Reject an unlisted node or any tuple other
than the exact reviewed 1-, 2-, or 16-node tuple.

Run RED:

```bash
python -m pytest -q \
  tools/launcher/tests/test_q30t_runtime_platform_feasibility.py \
  tools/launcher/tests/test_q30t_runtime_gateway_client.py \
  -k 'same_ofd or procfd or pidfd or start_ticks or scm_credentials or cached_peercred'
```

Expected: FAIL because interfaces do not exist.

### Step 2: Write RED mandatory-service and protected-link tests

Fake the exact Unix seqpacket gateway protocol and service signatures. Cover:

- missing socket/executable/key returns `BLOCKED`, never `unsupported-success`;
- wrong uid/gid/mode, producer-owned service, bad `SO_PEERCRED`, bad signature,
  changed public key, different device/inode, copy instead of link, mode other
  than 0400, nlink other than one, or writable/replacable parent fails;
- the same operation path reconstructed beneath the F-bound protected root is
  returned on every node, and a path reused from another operation is rejected;
- fchmod/write attack receives lease event before write completion; and
- gateway never exposes rename, unlink, overwrite, or reuse RPCs.

Trace the live keeper's signal boundary: block SIGIO in the initial thread
before any thread, set `F_SETOWN_EX(F_OWNER_PID)` and `F_SETSIG`, arm the only
`sigwaitinfo` waiter, verify every task mask, and only then call `F_SETLEASE`.
Inject the writer at every boundary and require either pre-lease handling or a
kernel break packet; no signal is lost to an unblocked thread or default
handler.

Add committed-control publication attacks. Require a fresh root-owned 0555
tree with 0444 single-link regular files at the identical absolute control
root on every 2-/16-node run node, identical bytewise path/hash/size/mode manifest, distinct
local root/file device-inode identities, and signatures from the F-bound
gateway key. Reject an omitted/extra/reordered blob, source/Git hash mismatch,
same-UID ownership, symlink, hardlink, copy after publication, cross-operation
reuse, different path text, shared cross-node inode claim, or changed file.

Run RED:

```bash
python -m pytest -q \
  tools/launcher/tests/test_q30t_runtime_gateway_client.py \
  -k 'blocked or endpoint or control_publication or protected or signature or same_inode or write_break or sigio'
```

Expected: FAIL.

### Step 3: Write RED direct-Pyxis, config, and 2-/16-node routing tests

Feed captured fixture logs to the probe. Require the exact positive direct-mode
line and reject container creation, named root, persistent `ENROOT_DATA_PATH`,
procfd image, different absolute paths, one PID/fd reused across nodes, and
nested execution.
Require a gateway-signed live SquashFUSE PID/start/cgroup and open backing fd
whose device/inode equals the publication. Reject pre-launch stat or log-only
claims, a closed/different fd, and a reused SquashFUSE process or mount.
Replay the exact `PyxisDirectMountAttestation` type from the design.

Require separate positive traces for N=2 and N=16. Each has N staging agents,
N distinct local keeper/fd/inode/endpoint identities, one identical protected
path text, and exactly one global `srun --nodes=N --ntasks=N
--ntasks-per-node=1 --gpus-per-node=4` with N global ranks and a collective
token. The N=16 trace is labeled segment-16 Q30 canary/full feasibility. Reject
a nested/per-node execution, 15/17 nodes, rank mismatch, endpoint reuse, or a
2-node trace relabeled as sixteen-node proof.

Add recursive manifest attacks for extra/changed `/etc/enroot/hooks.d`,
`mounts.d`, `environ.d`, Pyxis SPANK config/plugin, enroot, and SquashFUSE.
Poison HOME/XDG user config and require the gateway-owned empty root identity.

Run RED:

```bash
python -m pytest -q \
  tools/launcher/tests/test_q30t_runtime_platform_feasibility.py \
  -k 'direct or container_creation or multinode or global_step or config'
```

Expected: FAIL.

### Step 4: Write RED distributed barrier and remote-death tests

Model the installed gateway's registered-controller protocol. Required tests:

```python
def test_attack_session_can_never_receive_go() -> None:
    session = ready_gateway_session(nodes=("ptyche-n011", "ptyche-n012"), attack=True)
    with pytest.raises(GatewayProtocolError, match="ABORT-only"):
        session.decide("GO")


def test_global_go_waits_for_every_authenticated_ready() -> None:
    session = ready_gateway_session(nodes=("ptyche-n011", "ptyche-n012"), attack=False)
    session.accept_ready(frame_for("ptyche-n011"))
    with pytest.raises(GatewayProtocolError, match="not all nodes ready"):
        session.decide("GO")
```

Cover forged controller/wrapper PID, start ticks, cgroup, node, step, and
executable; early GO; controller EOF; duplicate/drained/withheld barrier event;
exact-step `scancel` failure; one cgroup still populated; `srun` not reaped;
nonterminal scheduler step; missing allocation state; and deadline equality.

Run RED:

```bash
python -m pytest -q \
  tools/launcher/tests/test_q30t_runtime_gateway_client.py \
  tools/launcher/tests/test_q30t_runtime_platform_feasibility.py \
  -k 'barrier or ready or abort or remote_exit or cgroup or allocation'
```

Expected: FAIL.

### Step 5: Write RED test-only-first and bounded-log tests

For each explicit `--node-count 1|2|16`, the F submitter trace must be exactly
test-only then parsable, with no second call after failure. Reject omitted or
other node counts, one-node prerequisite arguments, two-node without both the
one-node absolute receipt and 64-lower-hex physical hash, sixteen-node without
both the two-node absolute receipt and physical hash, test-only output used as
a job ID, non-distinct real job IDs, or any execution outside 1 -> 2 -> 16
prerequisite order. Require three canonical
`RuntimePlatformFeasibilityRunReceipt` objects and a v3 aggregate that reloads
all three, proves both prerequisite joins, preserves shared endpoint identities,
and exposes the exact ordered sixteen-node tuple. Require each run receipt's
one-global-step literal, exact argv hash, rank count equal node count, physical
collective artifact, null segment for 1/2, and segment 16 for node count 16.
The runner must place
batch/control/Pyxis/task logs under
`/raid/scratch/$USER/q30t-runtime-logs`, cap control at 1 MiB and each data log
at 8 MiB, close/fsync/stable-hash before durable publication, and never use a
Lustre `%j.out`.

Add platform-review RED cases for a missing/changed allowed-signers blob, wrong
principal/namespace, producer signing key reused by the reviewer, bad detached
signature, wrong F subject/rebuild physical hash, and any non-feasibility review
kind in F. Require typed `RuntimePlatformFeasibilityReviewBinding`; reject
missing/reordered producer or reviewer run hashes/job IDs, overlapping job IDs,
different sixteen-node tuples/endpoints, wrong global argv hashes, missing
collective artifacts, or a feasibility binding on another review kind.

Run RED:

```bash
python -m pytest -q \
  tools/launcher/tests/test_q30t_runtime_platform_feasibility_runner.py \
  tools/launcher/tests/test_q30t_runtime_platform_feasibility_submitter.py \
  tools/launcher/tests/test_q30t_runtime_independent_review.py
```

Expected: FAIL.

### Step 6: Implement only the F boundary

Implement the frozen feasibility run/aggregate receipts and loaders, exact
gateway/build endpoint identity reconstruction and post-connect process
checks, protected control-tree RPC, recursive config manifests, bounded
artifact collector, F runner, and F-only submitter. Implement canonical
platform review serialization, detached
allowed-signers verification, and anonymous no-clobber publication so Task 1
does not depend on future P code. Keep service operations as clients of the installed administrator
service; do not emulate protection with same-UID directories. Encode an enum
`PASS | BLOCKED`, where any missing mandatory primitive returns `BLOCKED` and
suppresses the receipt.

For node count one, the submitter executes the same constructed `sbatch` tuple
first with `--test-only`, then with `--parsable`, and publishes its run receipt.
Only after that receipt reloads may the node-count-two invocation run its own
test-only then parsable tuple with the exact prerequisite path/hash. Only after
the two-node receipt reloads may node-count sixteen do the same with that exact
path/hash. The runner performs only the requested explicit node count and
cannot import derived, attestation, consumer, or approval modules. The v3
aggregate is a separate local join over the three successful run receipts, not
another Slurm job.

### Step 7: Run GREEN and the F isolation gate

```bash
python -m pytest -q \
  tools/launcher/tests/test_q30t_runtime_platform_feasibility.py \
  tools/launcher/tests/test_q30t_runtime_gateway_client.py \
  tools/launcher/tests/test_q30t_runtime_platform_feasibility_runner.py \
  tools/launcher/tests/test_q30t_runtime_platform_feasibility_submitter.py \
  tools/launcher/tests/test_q30t_runtime_independent_review.py
python -m ruff check \
  tools/launcher/common/specdec/q30t_runtime_platform_feasibility.py \
  tools/launcher/common/specdec/q30t_runtime_gateway_client.py \
  tools/launcher/tests/test_q30t_runtime_platform_feasibility.py \
  tools/launcher/tests/test_q30t_runtime_gateway_client.py
python -m pyright \
  tools/launcher/common/specdec/q30t_runtime_platform_feasibility.py \
  tools/launcher/common/specdec/q30t_runtime_gateway_client.py
bash -n tools/launcher/common/specdec/run_q30t_runtime_platform_feasibility.sbatch
bash -n tools/launcher/common/specdec/submit_q30t_runtime_platform_feasibility.sh
shellcheck tools/launcher/common/specdec/run_q30t_runtime_platform_feasibility.sbatch \
  tools/launcher/common/specdec/submit_q30t_runtime_platform_feasibility.sh
! rg -n 'q30t_runtime_approvals|APPROVED_Q30T|derived-image|qualification-v3' \
  tools/launcher/common/specdec/q30t_runtime_platform_feasibility.py \
  tools/launcher/common/specdec/run_q30t_runtime_platform_feasibility.sbatch \
  tools/launcher/common/specdec/submit_q30t_runtime_platform_feasibility.sh
```

Expected: all GREEN; final scan has no matches.

### Step 8: Review and commit F only

Review syscall identity, real service dependency, no approval path, global
2-/16-node semantics, test-only ordering, and log bounds. Stage only F-owned
files, then:

```bash
git diff --cached --check
git commit -S -s -m "feat(specdec): probe immutable runtime platform boundary"
```

Do not combine F with P code.

### Step 9: Push F only after explicit integration approval

```bash
git push --porcelain origin HEAD
```

Record the pushed commit and remote ref. Do not submit from this step.

## Task 1: Run and Independently Review Phase F

This task is external and may start only after F is signed, clean, pushed, and
reviewed. The current documentation change does not execute it.

### Step 1: Prove submission construction without submitting

```bash
tools/launcher/common/specdec/submit_q30t_runtime_platform_feasibility.sh \
  --platform-feasibility --node-count 1 --dry-run
```

Assert the printed sequence contains the exact `sbatch --test-only` command
before the byte-identical `sbatch --parsable` command except for that one flag.

### Step 2: Submit and finish the one-node prerequisite

Run exactly:

```bash
tools/launcher/common/specdec/submit_q30t_runtime_platform_feasibility.sh \
  --platform-feasibility --node-count 1
```

The submitter performs a one-node `sbatch --test-only` and then a separate real
`sbatch --parsable`. Record only the parsable real job ID. Monitor only that
job with >=60 seconds between scheduler queries. Stable-load the canonical
one-node run receipt and its physical SHA-256. Do not continue if the service
is absent or any proof is BLOCKED.

### Step 3: Test and submit the dependent two-node probe

First verify the exact dependent argv without submission:

```bash
tools/launcher/common/specdec/submit_q30t_runtime_platform_feasibility.sh \
  --platform-feasibility --node-count 2 \
  --one-node-receipt "$one_node_receipt" \
  --one-node-receipt-sha256 "$one_node_receipt_sha256" --dry-run
```

Then remove only `--dry-run`. The submitter stable-loads that path/hash before
its two-node `sbatch --test-only`, then performs one real `sbatch --parsable`.
Require the second real job ID to differ from the first. A test-only response
is diagnostic output, never a job ID. Only after the two-node run receipt joins
the exact one-node prerequisite and replays the same gateway endpoint identity
for any node repeated across runs may the two-node run receipt
become a sixteen-node prerequisite. Record its canonical path and physical hash
as `two_node_receipt` and `two_node_receipt_sha256`.

### Step 4: Test and submit the dependent sixteen-node probe

First verify the exact dependent argv without submission:

```bash
tools/launcher/common/specdec/submit_q30t_runtime_platform_feasibility.sh \
  --platform-feasibility --node-count 16 \
  --two-node-receipt "$two_node_receipt" \
  --two-node-receipt-sha256 "$two_node_receipt_sha256" --dry-run
```

Then remove only `--dry-run`. Stable-load the exact two-node prerequisite
before sixteen-node test-only, then submit one real parsable job. Require a
third distinct real job ID, all sixteen unique nodes/endpoints, identity
preservation for any node repeated from an earlier run, one global
16-task/16-rank step, and the segment-16 collective token. Only then publish
the v3 aggregate.

### Step 5: Independently rerun and sign feasibility

The trusted reviewer checks out F independently, reruns all three node counts
in 1 -> 2 -> 16 order, loads every bounded physical artifact, compares
site/config/tool manifests, and signs
the canonical `platform-feasibility` review with namespace
`q30t-runtime-review-v1`. Its three real job IDs are pairwise distinct and
disjoint from the producer's three IDs.

### Step 6: Bind or stop

If producer and review are clean, record exact feasibility receipt, review
receipt, signature, allowed-signers, and public-service-key hashes for P. If
not, write a BLOCKED report and stop: Tasks 2-9 are forbidden.

## Task 2: Implement Trust and the Literal Blob-Map Schema

**Files:** runtime-identity/blob-map helpers, extensions to the F review module,
frozen allowed-signers blob and anonymous publisher, approvals module, and
matching fixture-based tests. Do not modify the production cluster profile in
this task.

### Step 1: Write RED exact-map/profile tests

Require the spec's literal F and qualification path/selector tuples and a
fixture profile with one SHA per `profile.*` selector, independently supplied
physical profile hash, canonical tuple hashes, F v3 receipt/review/signature
physical hashes, real gateway/build keys,
both exact service socket path/device/inode/uid/gid/mode identities, expected
service UID/GID and systemd identities, executable/public-key paths and hashes,
protected pathname, Pyxis/enroot/config manifests, log caps, and
`lease_break_time >= 15`. Reject receipt-supplied paths, extra/missing tuple
members, a profile self-hash field, empty keys, or a feasibility BLOCKED result.
Reload the one-/two-/sixteen-node run receipts from the v3 aggregate, require
three distinct real job IDs and the exact 1 -> 2 -> 16 prerequisite joins, and
reconstruct/connect both live endpoints from those F-bound values. Reject any
profile or caller override of a socket, key, peer identity, or protected root.
Add a regression test proving Task 2's commit leaves every production
`q30t_ptv23_cluster_profile.py` selector byte unchanged; fixture values are not
P's production freeze.

```bash
python -m pytest -q tools/launcher/tests/test_q30t_runtime_identity.py \
  -k 'feasibility or literal_blob or gateway or config or log_cap'
```

Expected: FAIL.

### Step 2: Write RED signed-review attacks

Cover wrong principal/namespace/key/fingerprint, changed allowed-signers blob,
producer key reused as reviewer, bad detached signature, noncanonical payload,
wrong subject schema/self/physical hash, absent or wrong rebuild receipt,
changed image/content manifest, wrong prerequisite signature, tuple cardinality,
and unbounded log.

```bash
python -m pytest -q \
  tools/launcher/tests/test_q30t_runtime_independent_review.py \
  -k 'signature or trust or subject or rebuild or prerequisite'
```

Expected: FAIL.

### Step 3: Implement the schema and review loader without freezing P values

Implement the literal path/selector map as source data, accept an explicit
fixture profile in unit tests, and verify Git blobs at claimed commits. Load the
F v3 aggregate/run physical hashes and service/reviewer endpoint identities
from typed inputs; do not write them into the production profile yet. Implement
`P_TASK7_CACHED_PATHS_V1` with the exact Task 7 Step 8 list and reject any
caller-supplied map or cached-name set. Implement
`SignedRuntimeReview`, canonical review v2, sterile `/usr/bin/ssh-keygen -Y
verify`, exact kind cardinalities, prerequisite chain, and physical replay.
Require the allowed-signers blob to be byte-identical at F and P.

Create two empty approval literals of tuple triples. The producer cannot load a
subject unless its signed review is supplied; F review is mandatory for build.

### Step 4: GREEN and static gates

```bash
python -m pytest -q \
  tools/launcher/tests/test_q30t_runtime_identity.py \
  tools/launcher/tests/test_q30t_runtime_independent_review.py \
  tools/launcher/tests/test_q30t_runtime_approvals.py \
  tools/launcher/tests/test_q30t_anonymous_publish.py
python -m ruff check tools/launcher/common/specdec tools/launcher/tests
python -m pyright \
  tools/launcher/common/specdec/q30t_runtime_identity.py \
  tools/launcher/common/specdec/q30t_runtime_independent_review.py
git diff --exit-code HEAD -- \
  tools/launcher/common/specdec/q30t_ptv23_cluster_profile.py
```

Expected: GREEN, both approval roots empty, and no production profile diff.

### Step 5: Hostile review and signed commit

Review the actual allowed-signers key, independent identity, F-v3 loader,
literal-map non-circularity, canonical signature input, anonymous no-clobber
publisher, empty approvals, and absence of a production selector freeze.

```bash
git diff --cached --check
git commit -S -s -m "feat(specdec): freeze runtime platform trust roots"
```

## Task 3: Implement Protected Hermetic Build and Full Manifests

**Files:** content-manifest module, derived-image module/runner, and tests.

### Step 1: Write RED complete-manifest tests

Create fixtures containing regulars, empty files, directories, symlinks, and
hardlinks. Every regular entry has exact `content_sha256`; reject one-byte
content changes with unchanged metadata, omitted
base entry, wrong size/link target/hardlink group/mode/uid/gid/mtime, xattr,
device/FIFO/socket, duplicate normalized path, traversal, and archive overlay
outside `/opt/q30t-runtime`. Reject a base that already contains the runtime
root rather than replacing it.

```bash
python -m pytest -q \
  tools/launcher/tests/test_q30t_runtime_content_manifest.py \
  -k 'content or symlink or hardlink or overlay or unsafe'
```

Expected: FAIL.

### Step 2: Write RED protection-domain and hermetic-closure tests

Reject build A/B sharing worker UID/identity, namespace, private root, candidate,
or request nonce; producer-owned/writable roots; host tool paths; missing ELF
interpreter/`DT_NEEDED`; unlisted library/config load; wrong builder image or
argv/environment; bad service signature; input snapshot mismatch; output
writer left open; or review worker aliasing producer workers.

```bash
python -m pytest -q tools/launcher/tests/test_q30t_derived_runtime_image.py \
  -k 'protection or hermetic or closure or service or writer'
```

Expected: FAIL.

### Step 3: Write RED A/B and publication tests

Require exact known base/archive hashes, install root, two output hashes equal,
three full manifests equal across A/B/replay, physical retained-image hash,
digest-derived path, root-owned mode-0444 inode and parent, signed service
publication, no-clobber adoption, and no execution. Reject producer-owned
retained bytes, same-UID chmod/write success, or local extraction even when
mode 0700. Replay the exact `ProtectedDerivedImagePublication` type.

```bash
python -m pytest -q tools/launcher/tests/test_q30t_derived_runtime_image.py \
  -k 'double_build or manifest or retained or no_clobber or no_execute'
```

Expected: FAIL.

### Step 4: Implement service client, manifests, receipt, and runner

Implement descriptor-only build requests to the F-pinned service. The client
does not accept roots/tools/env/argv. Load and validate signed A/B attestations,
reconstruct complete manifests, and compare image bytes. The service publishes
the retained image root-owned mode 0444 under a root-owned Lustre parent and
signs that publication; no candidate fd crosses to the producer. The producer
uses anonymous no-clobber publication only for bounded manifests and the v2
receipt. No local extraction or cleanup path exists.

### Step 5: GREEN and source gates

```bash
python -m pytest -q \
  tools/launcher/tests/test_q30t_runtime_content_manifest.py \
  tools/launcher/tests/test_q30t_derived_runtime_image.py \
  tools/launcher/tests/test_q30t_derived_runtime_image_runner.py
python -m ruff check \
  tools/launcher/common/specdec/q30t_runtime_content_manifest.py \
  tools/launcher/common/specdec/q30t_derived_runtime_image.py
python -m pyright \
  tools/launcher/common/specdec/q30t_runtime_content_manifest.py \
  tools/launcher/common/specdec/q30t_derived_runtime_image.py
bash -n tools/launcher/common/specdec/run_q30t_derived_runtime_image.sbatch
! rg -n '/usr/bin/(mksquashfs|unsquashfs|tar|zstd)|extractall|rmtree|unlink' \
  tools/launcher/common/specdec/q30t_derived_runtime_image.py \
  tools/launcher/common/specdec/run_q30t_derived_runtime_image.sbatch
```

Expected: GREEN; scan has no matches.

### Step 6: Hostile review and signed commit

Require full-content coverage, service-domain separation, hermetic closure,
input snapshots, A/B equality, no execution, and no-clobber publication.

```bash
git diff --cached --check
git commit -S -s -m "feat(specdec): build protected derived runtime image"
```

## Task 4: Implement Same-OFD Keeper and Protected Event Channel

**Files:** bootstrap, keeper, gateway client extensions, and tests.

### Step 1: Write RED protected-control bootstrap attacks

Construct two node-local gateway publications with the same protected root
text and committed relative path/hash/size/mode manifest but distinct local
device/inode identities. Poison `PYTHONPATH`, `sitecustomize`, `usercustomize`,
a same-named checkout package, and `sys.path`. Mutate a source checkout after
publication and try symlink traversal, absolute paths, `..`, late pathname
replacement, wrong owner/mode/nlink, one missing/extra/reordered module, and a
different cross-node manifest. Require root open with
`O_PATH|O_DIRECTORY|O_NOFOLLOW`, every module open with `openat2` beneath that
root and no symlink/magiclink, all fds retained before compile, exact manifest
hash, declared dependency order, `compile` plus explicit
`importlib.util.module_from_spec`, and runtime Python `-I -S`. Reject
descriptor/procfd text in global argv and any `SourceFileLoader`, `sys.path`,
checkout, `PYTHONPATH`, or absolute module reopen.
Require exactly one gateway-generated `held-modules.json`, reconstructed
byte-for-byte from the ordered committed specs; reject producer-supplied
manifest bytes, a second generated entry, or any other entry without an exact
Git blob path/hash.

```bash
python -m pytest -q tools/launcher/tests/test_q30t_runtime_bootstrap.py
```

Expected: FAIL.

### Step 2: Write RED keeper syscall, credential, and signal-order tests

Trace copy → fsync/hash → `fchmod(0400)` → read-only reopen → writer close →
main-thread `pthread_sigmask(SIG_BLOCK, SIGIO)` before any thread → inherited
blocked mask for every `/proc/self/task` entry →
`F_SETOWN_EX(F_OWNER_PID, keeper_pid)` → `F_SETSIG(SIGIO)` → dedicated
`sigwaitinfo` waiter ARMED → `F_SETLEASE(F_RDLCK)` → `F_GETLEASE` →
`sendmsg(SCM_CREDENTIALS, SCM_RIGHTS)`. Inject a blocking write after every
boundary. Reject reordered steps, any thread that can receive SIGIO
asynchronously, an asynchronous handler, owner PID mismatch, waiter not armed,
extant writer, `EINVAL`/`EAGAIN`, lease timeout below 15, changed inode,
nonzero pre-gateway nlink, or readiness before lease.
Require both agent and gateway SCM duplicates closed before publication returns,
leaving the keeper as the sole lease holder.

```bash
python -m pytest -q tools/launcher/tests/test_ptv23_node_keeper.py \
  -k 'order or sigio or signal_mask or writer or lease or scm_rights or ready'
```

Expected: FAIL.

### Step 3: Write RED same-OFD and event-channel attacks

Require `F_GETLEASE` on the SCM duplicate. Demonstrate a procfd reopen is a new
OFD and cannot satisfy the validator. Attack the seqpacket by procfd open,
sibling `pidfd_getfd`, duplicated/drained endpoint, forged credentials, wrong
nonce/PID/start, buffer fill, withheld lease event, keeper EOF/death, truncated
and extra packets. A user-sent `SIGIO` must abort but cannot set kernel
lease-break proof. Every attack must abort before global GO and publish nothing.
Because the socketpair predates spawn, assert that cached socketpair
`SO_PEERCRED` reports the creator and is rejected as keeper identity. Enable
`SO_PASSCRED`; require exactly one kernel `SCM_CREDENTIALS` on the keeper's
READY and every event packet, and join its PID/UID/GID to the child pidfd,
start ticks, expected job UID/GID, executable, and cgroup. Reject missing,
multiple, cached-only, or mismatched credentials.

```bash
python -m pytest -q \
  tools/launcher/tests/test_ptv23_node_keeper.py \
  tools/launcher/tests/test_q30t_runtime_gateway_client.py \
  -k 'same_ofd or scm_credentials or cached_peercred or drain or withhold or peercred or eof or duplicate'
```

Expected: FAIL.

### Step 4: Implement keeper/bootstrap protocol

Publish the committed control graph through the F-bound gateway, reconcile its
signed per-node file identities and identical manifest, and implement the
protected path-based bootstrap loader. Use an inherited close-on-exec
seqpacket socketpair with `SO_PASSCRED`; authenticate fresh kernel message
credentials against pidfd/start rather than cached socketpair `SO_PEERCRED`,
receive one fd by SCM, validate same-OFD lease, ask gateway to link the exact
inode, verify signed path/inode, and close the agent's duplicate. Keep the
keeper as sole lease holder. Block SIGIO in the original thread before any
thread exists, configure owner/signal, arm the dedicated `sigwaitinfo(SIGIO)`
thread, and only then acquire the lease. The waiter sends one credentialed
canonical break packet without releasing the lease. No stdout pipe, anonymous
control claim, path unlink, rename, recursive cleanup, or consumer inheritance
is allowed.

### Step 5: GREEN and Linux integration

```bash
python -m pytest -q \
  tools/launcher/tests/test_q30t_runtime_bootstrap.py \
  tools/launcher/tests/test_ptv23_node_keeper.py \
  tools/launcher/tests/test_q30t_runtime_gateway_client.py
python -m ruff check \
  tools/launcher/common/specdec/q30t_runtime_bootstrap.py \
  tools/launcher/common/specdec/ptv23_node_keeper.py
python -m pyright \
  tools/launcher/common/specdec/q30t_runtime_bootstrap.py \
  tools/launcher/common/specdec/ptv23_node_keeper.py
! rg -n 'os\.unlink|Path\.unlink|shutil\.rmtree|PYTHONPATH|SourceFileLoader' \
  tools/launcher/common/specdec/q30t_runtime_bootstrap.py \
  tools/launcher/common/specdec/ptv23_node_keeper.py
```

Expected: GREEN; scan has no matches.

### Step 6: Hostile review and signed commit

Review OFD semantics, credentials/Yama assumptions, writer closure, signal
ordering, event availability, gateway inode identity, keeper loss, and no
pathname deletion.

```bash
git diff --cached --check
git commit -S -s -m "feat(specdec): hold leased runtime inode by same OFD"
```

## Task 5: Implement Global Barrier, Exact Pyxis, and Remote-Death Proof

**Files:** keeper supervisor, gateway client extensions, global runner/probe,
and tests.

### Step 1: Write RED exact argv/environment tests

Require one global execution `srun` with node/task count N, tasks-per-node 1,
same absolute image path, and exact flags:

```text
--container-readonly
--no-container-mount-home
--no-container-remap-root
--no-container-entrypoint
```

Require constructed `:ro+rprivate` mounts, fixed workdir, runtime Python `-I
-S`, gateway-owned empty HOME/XDG config, and F-approved `/etc/enroot`/Pyxis
manifests. Reject container name/save/writable, procfd, caller mount text,
unapproved env/hooks, different per-node paths, nested srun, and creation logs.
The first constructed mount maps the identical per-node protected control root
to `/run/q30t/control`; reconcile its signed complete file identities and
manifest before constructing argv. Reject anonymous control fds,
`/proc/*/fd` control paths, different control roots/manifests, or a mutable
checkout mount.

```bash
python -m pytest -q tools/launcher/tests/test_ptv23_keeper_supervisor.py \
  -k 'argv or environment or pyxis or direct or global'
```

Expected: FAIL.

### Step 2: Write RED global barrier tests

Cover exact node cardinality/order, forged/duplicate/missing READY, wrong
job/step/node/PID/start/cgroup/executable/image inode, controller replacement,
controller EOF, early GO, partial-node GO, stale transaction, and barrier
timeout. The positive case invokes the application only after all READY and one
signed GO. Attack sessions reject GO at API and storage layers, start writers
only after all READY, emit ABORT, and never invoke the attacked operation.

Add a distinct keeper-loss trace after successful observation and before
lease-break. Kill one registered keeper only while every task is held READY;
require the gateway's pidfd-exit observation and the agent's credentialed
channel EOF, permanently mark the session ABORT-only, issue the signed ABORT,
cancel the exact step, and prove remote exit. Reject EOF without the pidfd
join, pidfd exit without EOF, any GO transaction, application invocation,
missing cancel/remote exit, a keeper-loss operation receipt, or ordering other
than positive → keeper-loss → lease-break.
Independently drop each authenticated controller/agent/wrapper/inter-gateway
connection and kill the gateway service before GO. Every case must become
ABORT-only, broadcast or locally synthesize ABORT, cancel the exact step, and
reject a replacement gateway endpoint and every later GO/receipt.

```bash
python -m pytest -q tools/launcher/tests/test_ptv23_keeper_supervisor.py \
  -k 'ready or barrier or early_go or abort_only or controller'
```

Expected: FAIL.

### Step 3: Write RED exact cancellation/death tests

Trace exact `scancel --signal=KILL job.step`. Reject only local process-group
death, accepted scancel without proof, one node's cgroup populated, spoofed
cgroup inode, unauthenticated EXIT, unreaped global srun, nonterminal/wrong
step, missing/wrong allocation, writer completion before proof, 5000 ms, or
`lease_break_time - 5` equality. Failure escalates to allocation cancellation
and never releases a success receipt.

For lease-break, persist exactly one canonical `KeeperLeaseBreakPacket` per
ordered node. Test parallel ordered physical-hash and self-hash tuples plus one
`RuntimeLeaseBreakNodeJoin` per node. Reject a packet not stable-published,
missing/duplicate/reordered packet, physical/self mismatch, cross-node
keeper/image/attacker identity, different remote-exit proof, or timestamps that
do not satisfy blocking-open-started ≤ packet-received ≤ abort-decision <
remote-exit-proof-complete < writer-open-complete. Writer completion before
the single joined remote-exit proof always fails.

```bash
python -m pytest -q tools/launcher/tests/test_ptv23_keeper_supervisor.py \
  -k 'cancel or remote_exit or cgroup or scheduler or allocation or deadline'
```

Expected: FAIL.

### Step 4: Implement global supervisor

Run a staging agent per node, reconcile signed identical-path publications,
then start one global Pyxis step. Register its exact step and wrappers with the
installed gateway. For positive operations request global GO only after every
READY. Run the fresh canonical keeper-loss operation and preserve its typed
EOF/pidfd/ABORT/no-GO/death evidence. For attack register ABORT-only, command
local same-UID writers, persist credentialed per-node break packets, broadcast
ABORT, cancel exact step, and gather the joined per-node cgroup plus scheduler/
allocation proof while every lease remains held. Publish neither negative
operation as a successful operation receipt.

### Step 5: GREEN and fake-Slurm integration

```bash
python -m pytest -q \
  tools/launcher/tests/test_ptv23_keeper_supervisor.py \
  tools/launcher/tests/test_q30t_runtime_probe_runner.py \
  -k 'global or pyxis or barrier or keeper_loss or lease_break or remote_exit'
python -m ruff check \
  tools/launcher/common/specdec/ptv23_keeper_supervisor.py
python -m pyright tools/launcher/common/specdec/ptv23_keeper_supervisor.py
bash -n tools/launcher/common/specdec/probe_ptv23_pyxis_keeper.sbatch
shellcheck tools/launcher/common/specdec/probe_ptv23_pyxis_keeper.sbatch
```

Expected: one global execution step in every two-node trace; GREEN.

### Step 6: Hostile review and signed commit

Review global rank semantics, exact flags/config, barrier authentication,
ABORT-only enforcement, same-UID attack ordering, exact step/allocation proof,
deadline, and lease lifetime.

```bash
git diff --cached --check
git commit -S -s -m "feat(specdec): gate global runtime step on protected barrier"
```

## Task 6: Implement v3 Attestation and Physical Replay

**Files:** attestation module and tests.

### Step 1: Write RED v3 schema/loader tests

Require exact identity block, F receipt/review, derived v2 receipt, full derived
manifest, literal blob tuple SHA, per-node signed publications, one global step,
direct-mount evidence, keeper-loss, lease-break, exit proof, relative imports,
four GPU identities, and bounded log hashes. Reject every missing/extra/reordered
field and noncanonical JSON.

Replay `RuntimeKeeperLossEvidence` before lease-break and require its ordered
nodes, lost keeper PID/start, READY-before-loss, gateway pidfd exit,
credentialed channel EOF, absent GO, signed ABORT physical/self hashes, exact
step cancellation, remote-exit physical/self hashes, absent operation receipt,
and self hash. Reject either loss signal alone or any later/earlier ordering.

Replay `RuntimeLeaseBreakEvidence` by first stable-loading exactly one persisted
`KeeperLeaseBreakPacket` for each ordered node from the ordered physical-hash
tuple. Require the packet self-hash tuple and corresponding
`RuntimeLeaseBreakNodeJoin` to match node/keeper/image/attacker/times, and every
join to name the same embedded `RemoteStepExitProof` self hash. Reject a
physical/self hash swap, missing/duplicate/reordered packet, cross-node join,
different remote proof, or completion-order violation.

```bash
python -m pytest -q tools/launcher/tests/test_ptv23_runtime_attestation.py \
  -k 'v3 or identity or direct_mount or physical_replay'
```

Expected: FAIL.

### Step 2: Write RED cross-evidence and cardinality attacks

Cover wrong device/inode between keeper/publication/Pyxis mount, named reuse,
container creation, one shared keeper/PID/fd/cgroup across nodes, different
protected path, nested step, mismatched global step ID, early/partial GO,
attacked receipt present, writer-before-exit, one populated cgroup, missing
allocation, unbounded log, duplicate node, and two-node wrong prerequisite.
Also replay every `ProtectedControlPublication`: require identical absolute
root text and path/hash/size/mode manifests across nodes, root/file uid 0,
root 0555, files 0444/nlink1, distinct node-local identities, F-bound gateway
signature, and the aggregate's ordered publication physical hashes. Reject a
descriptor-only control claim, missing committed module, reused tree, or
cross-node content difference.

```bash
python -m pytest -q tools/launcher/tests/test_ptv23_runtime_attestation.py \
  -k 'mismatch or duplicate or global_step or lease_break or log'
```

Expected: FAIL.

### Step 3: Implement v3 publishers/loaders

Remove named-reuse fields/types. Implement direct-mount v1, lease-break v2,
keeper-loss v3, remote-exit v1, protected-control publication v1, node v3,
aggregate v3, frozen typed inputs, canonical <=1 MiB loaders, stable physical
replay, exact cardinality, canonical negative-operation order, and source/image
lineage.
Each node receipt references the shared global evidence plus its own keeper,
mount identity, observation, and bounded logs.

### Step 4: GREEN and static gates

```bash
python -m pytest -q tools/launcher/tests/test_ptv23_runtime_attestation.py
python -m ruff check \
  tools/launcher/common/specdec/ptv23_runtime_attestation.py \
  tools/launcher/tests/test_ptv23_runtime_attestation.py
python -m pyright tools/launcher/common/specdec/ptv23_runtime_attestation.py
! rg -n 'named_container|reuse_evidence|extracted_runtime_path|source_checkout' \
  tools/launcher/common/specdec/ptv23_runtime_attestation.py
```

Expected: GREEN; scan has no matches.

### Step 5: Hostile review and signed commit

Review canonical bounds, complete lineage, per-node/global identity joins,
negative evidence ordering, physical hashes, and two-node prerequisite.

```bash
git diff --cached --check
git commit -S -s -m "feat(specdec): attest global immutable runtime boundary"
```

## Task 7: Integrate Operations, Logs, Submitter, and Consumers

**Files:** final cluster profile, probe, qualification submitter, row
loader/tests, continuation controller/runner/tests,
`q30t_runtime_admission.py`, and `test_q30t_runtime_admission.py`.

### Step 1: Write RED end-to-end fake traces

The two-node trace must show:

```python
def test_two_node_uses_local_staging_and_one_global_execution(trace: Trace) -> None:
    assert trace.staging_nodes == ("ptyche-n011", "ptyche-n012")
    assert trace.protected_paths == (trace.operation.protected_runtime_path,) * 2
    assert trace.local_inode_count == 2
    assert trace.global_execution_steps == 1
    assert trace.nested_execution_steps == 0
    assert trace.global_rank_count == 2
```

The same trace must contain six fresh gateway-signed protected control
publications: one per node for each positive, keeper-loss, and lease-break
operation. Within an operation they have identical control root text and
complete manifest but distinct per-node device/inode identities; across
operations no root/file inode is reused. Each operation's first read-only mount
is to `/run/q30t/control`. The global argv contains neither anonymous
descriptors nor procfd strings for image or Python controls.

Add operation sequence: stage → validate same OFDs/publications/config → global
positive READY/all-node GO → observation → fresh keeper-loss negative → fresh
ABORT-only READY/write/break/ABORT/cancel/remote death → node receipts →
aggregate. Any phase failure prevents later success publication.

```bash
python -m pytest -q tools/launcher/tests/test_q30t_runtime_probe_runner.py \
  -k 'two_node or sequence or fail_closed'
```

Expected: FAIL.

### Step 2: Write RED log and submitter attacks

Require node-local batch/control/Pyxis/stdout/stderr paths, 1/8 MiB caps,
O_EXCL/no-follow, close/fsync/stable hash before anonymous durable publication,
and no unbounded shared output. Cover overflow, sparse/truncated/changed log,
live writer, Lustre `%j.out`, test-only argv mismatch, preflight failure,
duplicate submission, dirty/unpushed/unsigned source, and scheduler polling under
60 seconds.

```bash
python -m pytest -q \
  tools/launcher/tests/test_q30t_runtime_probe_runner.py \
  tools/launcher/tests/test_q30t_runtime_qualification_submitter.py \
  -k 'log or overflow or test_only or submit'
```

Expected: FAIL.

### Step 3: Write RED row/training consumer authentication tests

Require subject+review+signature triples for derived and two-node receipts,
signed F-v3 1 -> 2 -> 16 review chain, same image/manifest/F/profile/blob map,
P ancestor, and all mapped blobs equal at consumer commit. Call admission with
typed `Q30TRuntimeAdmissionRequest` values for each exact reviewed node tuple.
Require segment-16 canary/full to
accept only the exact sixteen-node tuple and construct one global 16-task,
16-rank step. Reject an archive/extraction/checkout
path, direct Lustre image to Pyxis, old receipt schema, one-node-only approval,
missing/reordered/subset sixteen-node tuple, endpoint mismatch, v2 feasibility,
unknown review, changed non-approval blob, row/training image divergence, or
missing non-runtime gates.

```bash
python -m pytest -q \
  tools/launcher/tests/test_q30t_runtime_admission.py \
  tools/launcher/tests/test_q30t_row_observation_runtime.py \
  tools/launcher/tests/test_q30t_row_schema_observation_runner.py \
  tools/launcher/tests/test_q30t_ptv23_controller.py \
  tools/launcher/tests/test_q30t_ptv23_submitter.py \
  -k 'runtime or review or image or blob or extraction'
```

Expected: FAIL.

### Step 4: Implement integrated probe and bounded logs

Implement exact operation sequencing and fresh keeper/publication per
operation, while retaining one global execution step per operation. Capture
all live streams node-locally with caps before publishing stable bounded
artifacts. Submitter modes are exactly derived-image/one-node/two-node and use
test-only-first. There is no cleanup/deletion handler.

### Step 5: Implement shared consumer loader

Implement `authenticate_q30t_runtime` once in
`q30t_runtime_admission.py` and call it from row and continuation
paths. Both restage the durable image into fresh per-node keeper/gateway paths
and run one global step. Load and return the signed feasibility v3 subject and
review, match requested nodes to an exact reviewed 1-/2-/16-node run tuple, and
require the 16 tuple for segment-16 canary/full. Neither sees archive or
checkout inputs. Preserve row A/B, data, parent, canary, and training gates as
separate requirements.

### Step 6: GREEN integration and source policy

Run integration behavior with the explicit typed fixture profile from Task 2;
do not treat existing production selector values as final in this step.

```bash
python -m pytest -q \
  tools/launcher/tests/test_q30t_runtime_probe_runner.py \
  tools/launcher/tests/test_q30t_runtime_qualification_submitter.py \
  tools/launcher/tests/test_q30t_runtime_admission.py \
  tools/launcher/tests/test_q30t_row_observation_runtime.py \
  tools/launcher/tests/test_q30t_ptv23_controller.py \
  tools/launcher/tests/test_q30t_ptv23_submitter.py
bash -n tools/launcher/common/specdec/submit_q30t_runtime_qualification.sh
bash -n tools/launcher/common/specdec/probe_ptv23_pyxis_keeper.sbatch
bash -n tools/launcher/common/specdec/run_q30t_row_schema_observation.sbatch
bash -n tools/launcher/common/specdec/submit_q30t_row_schema_observation.sh
bash -n tools/launcher/common/specdec/run_q30t_ptv23_continuation.sbatch
bash -n tools/launcher/common/specdec/submit_q30t_ptv23_continuation.sh
shellcheck tools/launcher/common/specdec/submit_q30t_runtime_qualification.sh \
  tools/launcher/common/specdec/probe_ptv23_pyxis_keeper.sbatch \
  tools/launcher/common/specdec/run_q30t_row_schema_observation.sbatch \
  tools/launcher/common/specdec/submit_q30t_row_schema_observation.sh \
  tools/launcher/common/specdec/run_q30t_ptv23_continuation.sbatch \
  tools/launcher/common/specdec/submit_q30t_ptv23_continuation.sh
! rg -n 'PYTHONPATH|SLURM_EXPORT_ENV=ALL|--container-name|--container-save|--container-writable' \
  tools/launcher/common/specdec/probe_ptv23_pyxis_keeper.sbatch \
  tools/launcher/common/specdec/q30t_row_observation_runtime.py \
  tools/launcher/common/specdec/q30t_row_observation_bootstrap.py \
  tools/launcher/common/specdec/run_q30t_row_schema_observation.sbatch \
  tools/launcher/common/specdec/q30t_ptv23_continuation.py \
  tools/launcher/common/specdec/run_q30t_ptv23_continuation.sbatch
! rg -n 'extractall|unsquashfs|os\.unlink|Path\.unlink|shutil\.rmtree|rm -r' \
  tools/launcher/common/specdec/probe_ptv23_pyxis_keeper.sbatch \
  tools/launcher/common/specdec/q30t_row_observation_runtime.py \
  tools/launcher/common/specdec/q30t_row_observation_bootstrap.py \
  tools/launcher/common/specdec/run_q30t_row_schema_observation.sbatch \
  examples/dataset/observe_q30t_ptv23_row_schemas.py \
  tools/launcher/common/specdec/q30t_ptv23_continuation.py \
  tools/launcher/common/specdec/run_q30t_ptv23_continuation.sbatch
! rg -n '/lustre.*(%j|\.out|stdout|stderr)' \
  tools/launcher/common/specdec/submit_q30t_runtime_qualification.sh \
  tools/launcher/common/specdec/submit_q30t_row_schema_observation.sh \
  tools/launcher/common/specdec/run_q30t_ptv23_continuation.sbatch \
  tools/launcher/common/specdec/submit_q30t_ptv23_continuation.sh
```

Expected: all GREEN; policy scans have no matches.

### Step 7: Freeze final production selectors and rerun lineage tests

Only after Step 6 leaves every mapped Task 4-7 source byte final, stage the
mapped source paths owned by Task 7 except the profile. Invoke the Task 2
identity tool to compute each production `profile.*` SHA from exact index blobs,
bind the reviewed F-v3 aggregate/review/signature values recorded by Task 1,
write the final profile once, and stage it. The tool stable-reads each index
blob twice and fails if a mapped worktree path differs from its staged bytes.

```bash
p_paths=(
  tools/launcher/common/specdec/q30t_ptv23_cluster_profile.py
  tools/launcher/common/specdec/q30t_runtime_admission.py
  tools/launcher/common/specdec/probe_ptv23_pyxis_keeper.sbatch
  tools/launcher/common/specdec/submit_q30t_runtime_qualification.sh
  tools/launcher/common/specdec/q30t_ptv23_continuation.py
  tools/launcher/common/specdec/run_q30t_ptv23_continuation.sbatch
  tools/launcher/common/specdec/submit_q30t_ptv23_continuation.sh
  tools/launcher/common/specdec/q30t_row_observation_runtime.py
  tools/launcher/common/specdec/q30t_row_observation_bootstrap.py
  tools/launcher/common/specdec/run_q30t_row_schema_observation.sbatch
  tools/launcher/common/specdec/submit_q30t_row_schema_observation.sh
  examples/dataset/observe_q30t_ptv23_row_schemas.py
  tools/launcher/tests/test_q30t_ptv23_cluster_profile.py
  tools/launcher/tests/test_q30t_runtime_probe_runner.py
  tools/launcher/tests/test_q30t_runtime_qualification_submitter.py
  tools/launcher/tests/test_q30t_runtime_admission.py
  tools/launcher/tests/test_q30t_row_observation_runtime.py
  tools/launcher/tests/test_q30t_row_schema_observation_runner.py
  tools/launcher/tests/test_q30t_ptv23_controller.py
  tools/launcher/tests/test_q30t_ptv23_submitter.py
)
git add -- "${p_paths[@]:1}"
python -m tools.launcher.common.specdec.q30t_runtime_identity freeze-profile \
  --profile tools/launcher/common/specdec/q30t_ptv23_cluster_profile.py \
  --feasibility-receipt "$feasibility_receipt" \
  --feasibility-receipt-sha256 "$feasibility_receipt_sha256" \
  --feasibility-review "$feasibility_review_receipt" \
  --feasibility-review-sha256 "$feasibility_review_receipt_sha256" \
  --feasibility-review-signature "$feasibility_review_signature" \
  --feasibility-review-signature-sha256 "$feasibility_review_signature_sha256"
git add -- tools/launcher/common/specdec/q30t_ptv23_cluster_profile.py
python -m tools.launcher.common.specdec.q30t_runtime_identity verify-index-profile \
  --profile tools/launcher/common/specdec/q30t_ptv23_cluster_profile.py
python -m pytest -q \
  tools/launcher/tests/test_q30t_ptv23_cluster_profile.py \
  tools/launcher/tests/test_q30t_runtime_identity.py \
  tools/launcher/tests/test_q30t_runtime_admission.py \
  -k 'literal_blob or selector or lineage or feasibility_v3 or sixteen_node'
```

Expected: every literal selector equals the exact staged Git blob, F v3 and its
signed 16-node review reload, and no mapped worktree/index difference exists.
Any later mapped edit invalidates this result and restarts Steps 6-7.

### Step 8: Explicitly stage, scope-check, review, and sign P

Review global semantics, consumer identity, source policy, log caps, submitter
ordering, final selector lineage, and preservation of all independent gates.
Use this exact Task 7 ownership list; unrelated dirty worktree paths are neither
staged nor cleaned:

```bash
p_paths=(
  tools/launcher/common/specdec/q30t_ptv23_cluster_profile.py
  tools/launcher/common/specdec/q30t_runtime_admission.py
  tools/launcher/common/specdec/probe_ptv23_pyxis_keeper.sbatch
  tools/launcher/common/specdec/submit_q30t_runtime_qualification.sh
  tools/launcher/common/specdec/q30t_ptv23_continuation.py
  tools/launcher/common/specdec/run_q30t_ptv23_continuation.sbatch
  tools/launcher/common/specdec/submit_q30t_ptv23_continuation.sh
  tools/launcher/common/specdec/q30t_row_observation_runtime.py
  tools/launcher/common/specdec/q30t_row_observation_bootstrap.py
  tools/launcher/common/specdec/run_q30t_row_schema_observation.sbatch
  tools/launcher/common/specdec/submit_q30t_row_schema_observation.sh
  examples/dataset/observe_q30t_ptv23_row_schemas.py
  tools/launcher/tests/test_q30t_ptv23_cluster_profile.py
  tools/launcher/tests/test_q30t_runtime_probe_runner.py
  tools/launcher/tests/test_q30t_runtime_qualification_submitter.py
  tools/launcher/tests/test_q30t_runtime_admission.py
  tools/launcher/tests/test_q30t_row_observation_runtime.py
  tools/launcher/tests/test_q30t_row_schema_observation_runner.py
  tools/launcher/tests/test_q30t_ptv23_controller.py
  tools/launcher/tests/test_q30t_ptv23_submitter.py
)
git add -- "${p_paths[@]}"
diff -u \
  <(printf '%s\n' "${p_paths[@]}" | LC_ALL=C sort) \
  <(git diff --cached --name-only | LC_ALL=C sort)
test -z "$(git diff --name-only -- "${p_paths[@]}")"
git diff --cached --check
python -m tools.launcher.common.specdec.q30t_runtime_identity verify-index-profile \
  --profile tools/launcher/common/specdec/q30t_ptv23_cluster_profile.py
git commit -S -s -m "feat(specdec): use reviewed immutable runtime globally"
git show --show-signature --format=fuller --stat HEAD
test -z "$(git status --porcelain -- "${p_paths[@]}")"
```

This signed+DCO implementation commit, with both approval roots empty, is P.
Record its commit ID and require no staged or unstaged P-owned change. Unrelated
dirty paths remain untouched and outside P. Do not stage an equivalent
freeze-only change and do not create a second P commit in Task 8.

## Task 8: Freeze and Independently Hostile-Review Producer P

### Step 1: Run the complete local gate

```bash
python -m pytest -q tools/launcher/tests
python -m ruff check tools/launcher
python -m pyright tools/launcher/common/specdec
bash -n tools/launcher/common/specdec/run_q30t_runtime_platform_feasibility.sbatch
bash -n tools/launcher/common/specdec/submit_q30t_runtime_platform_feasibility.sh
bash -n tools/launcher/common/specdec/run_q30t_derived_runtime_image.sbatch
bash -n tools/launcher/common/specdec/submit_q30t_runtime_qualification.sh
bash -n tools/launcher/common/specdec/run_q30t_ptv23_continuation.sbatch
shellcheck tools/launcher/common/specdec/run_q30t_runtime_platform_feasibility.sbatch \
  tools/launcher/common/specdec/submit_q30t_runtime_platform_feasibility.sh \
  tools/launcher/common/specdec/run_q30t_derived_runtime_image.sbatch \
  tools/launcher/common/specdec/probe_ptv23_pyxis_keeper.sbatch \
  tools/launcher/common/specdec/submit_q30t_runtime_qualification.sh \
  tools/launcher/common/specdec/run_q30t_row_schema_observation.sbatch \
  tools/launcher/common/specdec/submit_q30t_row_schema_observation.sh \
  tools/launcher/common/specdec/run_q30t_ptv23_continuation.sbatch \
  tools/launcher/common/specdec/submit_q30t_ptv23_continuation.sh
git diff --check
```

Expected: all exit 0.

### Step 2: Verify P invariants

```bash
python -m pytest -q tools/launcher/tests/test_q30t_runtime_approvals.py \
  -k 'empty_producer_roots or literal_blob_map or unknown_review'
git grep -n 'APPROVED_Q30T_.*frozenset()' -- \
  tools/launcher/common/specdec/q30t_runtime_approvals.py
git status --short
```

Expected: two empty roots and literal map exact at P. Report but do not stage,
clean, or otherwise alter unrelated dirty paths.

### Step 3: Independent hostile review

Require explicit verdicts for all of these: same-OFD semantics; procfd rejection;
real gateway/build service binding; direct Pyxis regular path; one global
multi-node step; exact flags/config/user neutralization; protected build roots
and full manifests; protected seqpacket events; distributed GO/ABORT; exact
remote death/allocation deadline; hermetic library closure; literal blob map;
post-Task7 selector recomputation from exact staged blobs; signed reviewer
trust/subject/rebuild; live F 1 -> 2 -> 16 before P; exact segment-16 admission;
bounded node-local logs; explicit P cached-name/unstaged assertions; consumer
equality; and P empty roots.

Any critical, warning, or nit reopens the owning task. P freezes only at
`0 critical / 0 warnings / 0 nits`.

### Step 4: Confirm reviewed HEAD as P and push only after approval

```bash
git diff --cached --quiet
git show --show-signature --format=fuller --stat HEAD
git rev-parse HEAD
```

Require `HEAD` to equal the P commit recorded in Task 7 and both approval roots
to remain empty. Push that unchanged HEAD only after explicit integration-owner
approval. Do not commit or submit a job in this step. If review required a
change, reopen the owning implementation task; its new final signed commit
becomes P and Task 8 restarts. Unrelated unstaged/untracked paths do not enter
P and remain untouched.

## Task 9: Produce Physical Evidence, Review, and Constants-Only A

This task starts only from clean pushed P and the reviewed F binding.

### Step 1: Build and review the derived image

Run the derived-image submitter, which performs test-only first. Load the v2
receipt and all full manifests. The independent reviewer runs protected review
build A/B in distinct service domains, verifies byte/content equality, and
signs a `derived-image` review bound to the exact subject and F review.

### Step 2: Qualify and review one node

Run one-node test-only first. Require fresh local keeper/publication, unnamed
direct Pyxis, positive all-ready/GO, keeper-loss failure, ABORT-only same-UID
lease attack, exact global-step death proof, bounded logs, and v3 receipt. The
reviewer reruns and signs the exact one-node subject.

### Step 3: Qualify and review two nodes

Run two-node test-only first using the reviewed one-node prerequisite. Require
different local keepers/fds/inodes/cgroups at one identical path, one global
two-task execution step, tested collective, global barrier, per-node lease
attacks, cgroup empty on both nodes, exact step/allocation state, and aggregate
v3 receipt. The reviewer independently reruns and signs it.

### Step 4: Reconcile exact physical hashes

Run public loaders from P over every subject/review/signature, full manifest,
gateway/service attestation, node evidence, remote-exit proof, and bounded log.
Require review verdict `clean`, F's signed live 1 -> 2 -> 16 prerequisite
chain, and exact subject chain F → image → one-node → two-node.

### Step 5: Create constants-only A

Edit only the two approval frozensets. Add exactly one derived tuple and one
two-node qualification tuple, each containing subject, review receipt, and
review signature physical SHA-256.

```bash
git diff --name-only HEAD
git diff --check
python -m pytest -q tools/launcher/tests/test_q30t_runtime_approvals.py
git diff --exit-code HEAD -- \
  tools/launcher/common/specdec/q30t_runtime_platform_feasibility.py \
  tools/launcher/common/specdec/q30t_runtime_gateway_client.py \
  tools/launcher/common/specdec/q30t_runtime_content_manifest.py \
  tools/launcher/common/specdec/q30t_anonymous_publish.py \
  tools/launcher/common/specdec/q30t_runtime_archive_receipt.py \
  tools/launcher/common/specdec/q30t_runtime_identity.py \
  tools/launcher/common/specdec/q30t_tree_digest.py \
  tools/launcher/common/specdec/q30t_derived_runtime_image.py \
  tools/launcher/common/specdec/q30t_runtime_independent_review.py \
  tools/launcher/common/specdec/q30t_runtime_review_allowed_signers \
  tools/launcher/common/specdec/q30t_runtime_admission.py \
  tools/launcher/common/specdec/q30t_runtime_bootstrap.py \
  tools/launcher/common/specdec/q30t_ptv23_cluster_profile.py \
  tools/launcher/common/specdec/ptv23_node_keeper.py \
  tools/launcher/common/specdec/ptv23_keeper_supervisor.py \
  tools/launcher/common/specdec/ptv23_runtime_attestation.py \
  tools/launcher/common/specdec/probe_ptv23_pyxis_keeper.sbatch \
  tools/launcher/common/specdec/run_q30t_derived_runtime_image.sbatch \
  tools/launcher/common/specdec/run_q30t_runtime_platform_feasibility.sbatch \
  tools/launcher/common/specdec/submit_q30t_runtime_qualification.sh \
  tools/launcher/common/specdec/submit_q30t_runtime_platform_feasibility.sh
git add -- tools/launcher/common/specdec/q30t_runtime_approvals.py
test "$(git diff --cached --name-only)" = \
  "tools/launcher/common/specdec/q30t_runtime_approvals.py"
git diff --cached --check
```

Expected: only `q30t_runtime_approvals.py` differs from P and all mapped blobs
are byte-identical. The index contains exactly that path; no other staged or
unstaged change exists.

### Step 6: Independently review and sign A

Re-run admission from both row and continuation consumers, inspect P→A diff,
require continuation canary/full admission to replay the exact signed 16-node F
tuple, and require `0 critical / 0 warnings / 0 nits`.

```bash
git diff --cached --check
git commit -S -s -m "chore(specdec): approve reviewed immutable q30t runtime"
git show --show-signature --format=fuller --stat HEAD
```

Do not submit row observation, canary, or training from this approval step.

## Final Verification

Implementation is complete only when:

- F live one-/two-/sixteen-node feasibility and signed review are clean, or the design
  has explicitly stopped BLOCKED before P;
- two producer builds and two independent rebuilds are byte/content identical;
- same-OFD SCM validation, Yama protection, gateway same-inode publication,
  direct unnamed Pyxis, exact config, and no persistent rootfs are proved;
- node agents only stage/control and every distributed operation is one global
  Slurm step at an identical local pathname;
- all tasks wait after READY, attack tasks never GO, and exact global step death
  plus allocation state precede lease release;
- live logs are node-local/bounded and durable artifacts are stable/bounded;
- one-/two-node v3 receipts and signed reviews bind the same image;
- segment-16 canary/full admission binds F's exact reviewed sixteen-node tuple,
  per-node endpoints, and one global 16-task/16-rank step;
- source-policy scans are clean;
- final selector values are computed after all mapped source is final and
  replayed from P's exact staged blobs;
- P is signed+DCO from the explicit verified cached path set with empty roots;
  A is signed+DCO and changes only the two tuple literals; and
- row and continuation authenticate the same approved runtime while separate
  data, parent, canary, and training approvals remain enforced.
