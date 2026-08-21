# Multi-Cluster Drafter Standby Design

## Goal

Prepare Lyris and Pre-Tyche as reproducible GB200 standby environments for the
ModelOpt Qwen3 drafter workflow currently running on OCI-HSG. A standby cluster
must be able to run the same 32-experiment training matrix and checkpoint
evaluations without sharing mutable state with OCI-HSG.

The standby setup does not submit the 32 production experiments automatically.
It stops after source, asset, runtime, and short GPU canaries pass. Production
activation is an explicit operation so two clusters never write the same
checkpoint or W&B run concurrently.

## Selected Approach

Use one cluster-independent workflow with typed cluster profiles. The existing
training manifest, checkpoint validation, milestone preservation, self-requeue,
and evaluator code remain authoritative. Profiles supply only scheduler,
filesystem, and container-placement differences.

This is preferred over copying OCI scripts and editing them per cluster because
independent copies would drift in requeue limits, B/K mappings, provenance
checks, and checkpoint semantics. A container-per-cluster approach is also
unnecessary: all three clusters are ARM64 GB200 and should consume an immutable
image with the same SHA256.

## Reproducibility Contract

All profiles bind the following immutable inputs:

- ModelOpt commit `e3febcbe1319f018eea81fa4d42e2e36cb54494e`
- vLLM 0.27.1 ARM64 container filename and SHA256
- relocatable ModelOpt training runtime archive and SHA256
- relocatable Speculators evaluator runtime archive and SHA256
- Qwen3-30B-A3B Base and Thinking model revisions
- Qwen3-235B-A22B Base and Thinking model revisions
- OpenPerfectBlend revision
  `af60f3c18201652a83a93f46fcfee1b646ba3df7`
- materialized Nemotron Post-Training v2 training view and its file manifest
- canonical 32-experiment training manifest and evaluator dataset manifest

Every durable artifact has a sidecar containing source identity, byte size, and
SHA256. A job fails before GPU initialization when the profile, source commit,
or artifact identity disagrees with the manifest.

## Storage Layout

Source, scripts, Git metadata, and configuration live under `/home`. Container
images, model snapshots, datasets, checkpoints, and large results live under a
cluster-specific durable Lustre or project root. Runtime extraction, compiler
caches, databases, locks, W&B temporary files, and logs that do not need to
survive use compute-node-local scratch.

The scratch path is a profile value discovered by a one-node probe. OCI uses
`/raid/scratch`; Lyris and Pre-Tyche use the first verified path among
`$SLURM_TMPDIR`, `/raid/scratch`, and `/tmp`. Login-node path existence is not
accepted as compute-node evidence.

Each cluster uses a distinct output namespace and W&B group suffix. Resume is
allowed only within the same cluster namespace unless a checkpoint is copied as
an immutable transfer artifact and revalidated at the destination.

## Cluster Profiles

### Lyris

- SSH: `login-lyris` as `sna-mfa`
- Account: `coreai_dlalgo_llm`, pending live MFA verification
- Partition: `gb200`, pending live verification
- Training: 4 exclusive nodes, 4 GPUs per node, `--segment=4`
- Staging/evaluation: 1 exclusive node, `--segment=1`
- Do not pass `--gres` or `--gpus-per-node`
- Maximum walltime: 5 hours; use the existing 3:55 training chunks
- Durable root:
  `/lustre/fsw/coreai_dlalgo_llm/users/sna/modelopt-specdec`

The current blocker is an expired Microsoft MFA session. Account, FairShare,
filesystem permissions, Pyxis, and artifact presence must be checked after the
user establishes a ControlMaster session.

### Pre-Tyche

- SSH: `login-ptyche`
- Account: `coreai_dlalgo_llm`
- Preferred partition: `36x2-a01r`; fallback: `batch`
- Observed FairShare at design time: `0.737931`
- Training: 4 exclusive nodes, 4 GPUs per node, `--segment=4`
- Staging/evaluation: 1 exclusive node, `--segment=1`
- Do not pass `--gres` or `--gpus-per-node`
- Maximum walltime: 5 hours; use the existing 3:55 training chunks
- Durable root:
  `/lustre/fsw/coreai_dlalgo_llm/users/sna/modelopt-qwen3-drafter-training`

Pre-Tyche already has pinned Q30 Base, Q235 Base, and Q235 Thinking model
caches. It lacks Q30 Thinking, the exact vLLM image, current runtime archives,
the evaluator manifest, and exact source checkouts. Its node-local scratch path
must be discovered on a compute node.

## Artifact Transfer

Use Git over SSH for source and PDX/PBSS object storage for large immutable
artifacts. Transfers run as one-node Slurm data-mover jobs, never inline on a
login node. Upload OCI artifacts once under a content-addressed bucket prefix;
download them independently on Lyris and Pre-Tyche with checksum verification.

The transfer manifest is the source of truth. Download jobs publish to a
job-unique sibling partial directory and atomically rename only after all
checksums pass. Existing completed artifacts with matching identities are
reused. Conflicting destinations fail closed and are not overwritten or
recursively deleted.

Models and datasets are transferred as their existing bounded shard sets, not
as millions of files and not as per-job copies. Checkpoints transferred for
disaster recovery retain the permanent step-4166 and step-25391 exports plus
the selected complete resume checkpoint.

## Workflow

1. Establish SSH authentication and verify the live account, partition,
   FairShare, Pyxis support, and compute-node scratch path.
2. Clone ModelOpt and pinned Speculators source into `/home`; verify clean exact
   commits and signatures where available.
3. Transfer the immutable container, runtime archives, missing models,
   datasets, and evaluator manifest through PDX data-mover jobs.
4. Verify every size/SHA/revision sidecar at the destination.
5. Run a one-node CPU/container provenance probe.
6. Run a one-node four-GPU vLLM generation and evaluator smoke.
7. Run a four-node `max_steps=20`, `save_steps=20` training canary with the
   cluster profile, then verify checkpoint, export, W&B, and GPU utilization.
8. Render the full 32-experiment matrix and run `sbatch --test-only` for every
   initial and successor stage without submitting production training.
9. Write a readiness receipt containing all paths, hashes, test job IDs, and
   observed scheduler settings.

## Activation and Failover

Standby activation is an explicit command requiring a destination cluster and
manifest. It first takes a single filtered scheduler snapshot and refuses to
submit when the same experiment boundary is active or successfully completed
on that cluster. It uses a cluster-specific receipt directory and W&B suffix.

OCI jobs are not cancelled automatically. Cross-cluster continuation requires
an immutable checkpoint transfer and destination-side provenance validation.
This prevents split-brain training where two clusters update the same logical
run from different checkpoints.

## Error Handling

- Authentication, Kerberos, account, partition, or scratch discovery failure:
  stop before transfer or submission.
- Missing or mismatched artifact identity: fail closed and preserve evidence.
- Partial transfer: preserve a bounded receipt, retry through the data-mover;
  never publish the partial directory.
- GPU/container smoke failure: do not render or submit production jobs.
- Idle GPU detection: the Q30 profile launches two TP2 serve replicas per serve
  node; Q235 launches one TP4 replica. All four GPUs per allocated node must be
  visible in the canary telemetry.
- Requeue/resume: retain the existing `save_steps=50`, `save_total_limit=2`,
  maximum 50 same-ID requeues, permanent step-4166 milestone, and final
  step-25391 artifacts.

## Verification

Local tests cover profile validation, scheduler flag rendering, scratch-path
selection, cluster-specific output identity, transfer manifest validation, and
forbidden OCI path leakage. ShellCheck, Ruff, formatting, and focused pytest
must pass before a signed/DCO commit.

Each cluster must then pass, in order:

- filtered live scheduler/account probe
- `sbatch --test-only` for stage, smoke, evaluation, and training
- immutable artifact checksum verification
- one-node runtime/container smoke
- one-node evaluation smoke with a valid CSV
- four-node 20-step training canary monitored for at least five minutes
- full 32-experiment render and test-only receipt

No production standby matrix is submitted until all gates pass and the user
explicitly activates that cluster.
