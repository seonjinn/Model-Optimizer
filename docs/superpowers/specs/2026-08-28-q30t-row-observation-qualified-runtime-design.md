# Q30T Row-Schema Observation in a Qualified Runtime

**Date:** 2026-08-28

## Context

The Q30T SWE-heavy 700K dataset builder needs two byte-identical observations of
all 201 PTV2 and 100 PTV3 source files before it can freeze source mappings and
row-schema approvals. The first Ptyche implementation ran the observer directly
with `/usr/bin/python3.12 -I -S`.

Production evidence invalidated that design:

- Ptyche does not export `SLURM_TMPDIR`; the runner now safely allocates private
  node-local scratch under the reviewed `/raid/scratch` contract.
- Ptyche's host `/usr/bin/python3.12` and the reviewed base SQSH do not contain
  PyArrow.
- The reviewed runtime archive with SHA-256
  `4a20aee61f290c48bed22a84b4a0ae0cbdc54e3e3910854d253188c8854f5dc9`
  does contain PyArrow under `lib/python3.12/site-packages`.
- Extracting that archive into a mutable user-owned tree and importing it from
  the host would violate the same-UID substitution model already enforced by
  the row observer and runtime qualification work.
- Extracting it into an ordinary user-owned directory mounted into a container
  has the same defect; Pyxis does not make the backing directory immutable.

The row observation must therefore consume the same keeper-backed Pyxis runtime
that will qualify and run the Q30T canary, rather than inventing a weaker host
runtime boundary.

## Decision

First derive one immutable, content-addressed runtime SQSH from the reviewed
base SQSH plus the exact reviewed runtime archive. The derived image installs
the archive tree at the stable container path `/opt/q30t-runtime`; its producer
receipt binds the base-image SHA-256, archive SHA-256, canonical runtime-tree
SHA-256, producer commit/tool hashes, and final SQSH SHA-256. A separate
one-node then two-node runtime qualification must replay that receipt and use
the derived SQSH through a keeper descriptor anchor.

Before exposing that anchor, the keeper copies the reviewed SQSH into a fresh
anonymous `O_TMPFILE`, verifies and fsyncs its complete bytes, changes it to
mode `0400`, and, while the writer is still open, reopens
`/proc/self/fd/{writer_fd}` with `O_RDONLY`. It then closes every writable
descriptor, acquires and verifies an `F_RDLCK` lease on the read descriptor,
and only then exposes that leased descriptor. The backing inode has no
pathname. The keeper monitors lease-break notification as a fatal event. On
notification, supervision disables named-container reuse, kills and fully
reaps the active container/process group within a bound shorter than the
observed kernel lease-break timeout, and only then permits lease release; no
runtime code remains alive when a blocked writer can proceed. Runtime
qualification injects a same-UID write-open attempt and proves prompt
cancellation/reaping/no receipt. A platform that cannot provide this
anonymous-inode/read-lease boundary is unsupported and must fail before Pyxis
launch.

Run each row-schema observation as a one-node keeper-backed Pyxis operation
from that independently reviewed derived SQSH. The host runner performs only
source authentication, keeper supervision, container launch, receipt
reconciliation, and no-clobber publication. No mutable runtime archive is
extracted during observation.

The observer executes inside the named container with the stable runtime
executable `/opt/q30t-runtime/bin/python`. A held authenticated bootstrap runs
that executable with `-I -S` and explicitly inserts only
`/opt/q30t-runtime/lib/python3.12/site-packages` before importing PyArrow. The
observer records runtime-relative executable and module paths, never job- or
scratch-specific absolute paths. It does not search host site-packages, enable
`site`, execute `.pth` files, or fall back to a caller-selected Python
environment.

The bootstrap is the committed
`tools/launcher/common/specdec/q30t_row_observation_bootstrap.py` blob staged by
the keeper. It opens the staged row-runtime and observer blobs with
`O_NOFOLLOW`, retains and hashes their exact descriptors, then loads them in
dependency order with explicit `importlib` module names. The control directory
is never added to `sys.path`; only the authenticated runtime site-packages path
is inserted for PyArrow. The atomic publisher stays host-side and never
executes inside the container.

## Alternatives Considered

### Keep the base SQSH plus a mutable extracted runtime directory

This avoids producing a second image, but neither host mode `0700` nor a Pyxis
mount prevents the same UID from mutating the backing tree. Rehashing after an
import detects the attack only after untrusted code can execute. This option is
rejected.

### Build an observation-only PyArrow image

An observation-only image would protect PyArrow but create scientific drift
from training. The selected derived SQSH is instead the single runtime image
qualified for both observation and later training.

### Use a different Parquet implementation

No separately reviewed Parquet implementation is already available on Ptyche.
Adding one would create another dependency and identity contract without
removing the runtime qualification requirement.

## Components and Ownership

### Runtime qualification producer

The existing runtime qualification subproject owns:

- archive/image staging through `ptv23_node_keeper.py`;
- deterministic production and review of the derived runtime SQSH;
- named-container creation and reuse;
- archive-tree and derived-image lineage replay;
- exact Python, Pyxis control-plane, enroot, CUDA, and keeper evidence;
- one-node and two-node canonical qualification receipts.

The derived-image producer normalizes entry order, uid/gid, modes, timestamps,
and SquashFS build metadata. Two isolated builds from the same reviewed inputs
must be byte-identical before the retained artifact can be reviewed. The
receipt binds both build digests, the exact builder executable/version/argv,
and the one retained SQSH physical SHA-256.

The row observer consumes these receipts. It does not duplicate or weaken their
validation.

### Row observation submitter

`submit_q30t_row_schema_observation.sh` gains required arguments for:

- the reviewed runtime qualification receipt path and physical file SHA-256;
- the reviewed profile and archive-tree receipt paths and physical SHA-256s;
- the reviewed derived-image receipt path and physical SHA-256;
- the derived runtime image path and SHA-256;
- the matching observation and operation-receipt output paths.

It continues to require a clean pushed signed commit, approved Git remote,
sterile SSH/Git execution, exact approved output/log paths, and successful
`sbatch --test-only` before real submission. Caller-provided runtime values must
equal the replayed qualification receipt and an immutable production approval
set; they are not independent trust roots. Receipt production occurs from one
reviewed producer commit. After independent review, a separate approval-only
descendant commit inserts the exact qualification, derived-image receipt, and
PTV3 completion physical SHA-256 values. Admission proves the qualification
producer is an ancestor and every qualification-bound tool blob remains
byte-identical.

### Row observation runner

`run_q30t_row_schema_observation.sbatch` performs these phases:

1. Authenticate the spooled runner, committed observer, atomic publisher,
   keeper, runtime attestation tool, and their explicit import closure from the
   pushed Git commit.
2. Replay the qualification receipt and require the exact cluster/profile,
   archive, image, runtime tree, tool, and source-commit identities.
3. Allocate descriptor-bound node-local scratch using the reviewed Ptyche
   `/raid/scratch` contract.
4. Start one keeper that stages the derived image and committed tools by exact
   SHA-256 and publishes descriptor anchors.
5. Start one named Pyxis container from the keeper-backed derived-image anchor
   and mount immutable control inputs.
6. Replay the installed `/opt/q30t-runtime` canonical tree digest, then invoke
   its Python with `-I -S` through a held bootstrap that explicitly adds only
   its authenticated site-packages root.
7. Execute the held observer bytes over the exact PTV2/PTV3 inputs and retain
   its canonical staged payload through a host-held descriptor.
8. Reuse the named container without another image argument, reconcile
   container/runtime/keeper evidence, and require the keeper to remain live.
9. Publish or adopt the observation from the held payload, recheck keeper
   liveness, stop/reap it with bounded supervision, and publish the job-specific
   operation receipt last. Only the durable operation receipt marks success.

Every child process receives an explicit sterile environment. No phase uses
`SLURM_EXPORT_ENV=ALL`, inherited `BASH_ENV`, inherited `PYTHONPATH`, or a
PATH-selected security-critical executable.

### Observation schema

The observation retains its existing source-file and row-schema evidence and
adds a deterministic `runtime` object containing:

- qualification receipt physical file SHA-256 and embedded receipt SHA-256;
- archive, base-image, derived-image, and canonical runtime-tree SHA-256;
- runtime profile physical file SHA-256;
- qualification-producer commit and exact approved runtime tool SHA-256s;
- observed Python executable/version and PyArrow version/origin/tree digest,
  with paths normalized relative to `/opt/q30t-runtime`;

Job-specific evidence is deliberately excluded from the scientific observation.
Each run publishes a separate canonical operation receipt containing the row
producer commit, exact runner/observer/keeper/attestation tool SHA-256s, Slurm
job and node identity, keeper receipt digest, named-container evidence digest,
and the physical SHA-256 of the observation it produced or adopted.

The deterministic runtime object participates in the observation self-hash.
Observation A and B must be byte-identical, so they must use the same reviewed
runtime qualification receipt and immutable inputs. Their operation receipts
are expected to differ because they bind distinct jobs, but both must bind the
same observation physical SHA-256 and immutable runtime tuple.

## Failure Semantics

- Missing or unapproved runtime qualification evidence fails before keeper or
  container launch.
- Archive/image/tree, source, tool, or profile mismatch fails closed.
- Keeper death cancels the active container/process group. Before observation
  publication it leaves no final output. After no-clobber observation
  publication it leaves that immutable file explicitly unauthenticated and
  suppresses the operation receipt; no consumer may adopt it without a later
  successful full replay.
- Container launch, import, scan, staging, reconciliation, or shutdown failure follows
  the same absent-or-unauthenticated rule and never produces an operation
  receipt.
- Existing exact observation is adopted only after full semantic replay; a
  foreign or different output is preserved and rejected. An operation receipt
  is published only after that exact observation is durably bound.
- No failure path recursively deletes shared or possibly foreign data. Private
  node-local scratch may be retained for scheduler cleanup.

## Verification

Local behavioral tests must cover:

- missing/mismatched qualification receipt and runtime identities;
- dirty or unpushed source and mutable checkout substitution;
- exact fake-`srun` traces for keeper, named container, observer, publication,
  and shutdown phases;
- keeper death during import, scan, staging, reconciliation, publication, and
  shutdown;
- archive/tree, PyArrow origin, tool blob, mount, and output substitutions;
- bounded FIFO/process-group cancellation; failures before observation
  publication leave no final output, while later failures leave only an
  unauthenticated immutable observation without an operation receipt;
- exact existing-output adoption, job-specific operation receipts, A/B
  observation determinism, and foreign-output preservation.

Ptyche operations must run in this order:

1. produce and independently review the derived-image receipt;
2. review one-node, then two-node runtime qualification receipts;
3. independently replay and review the fixed PTV3 completion physical SHA-256
   together with its staged source-plan binding;
4. insert only those three reviewed physical hashes in an approval-only
   descendant commit and independently review that source transition;
5. run row observation A through `--test-only`, then real submission;
6. run row observation B through `--test-only`, then real submission;
7. require canonical JSON, valid self-hashes, 201 PTV2 files, 100 PTV3 files,
   670,399 PTV3 rows, and byte-identical A/B outputs;
8. independently review the observation physical file SHA-256 before creating
   the Task 4B source mapping and inventory producer.

## Downstream Gate

This design enables row-schema evidence and the source-inventory path. It does
not remove the separate requirement for authoritative held-out UUID sources for
`speed`, `math`, `code`, `swe`, and `tool`. The final 700K dataset build and
training remain blocked until those five evidence roots are produced and
independently approved.
