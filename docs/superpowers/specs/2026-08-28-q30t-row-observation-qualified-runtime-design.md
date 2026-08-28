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

The row observation must therefore consume the same keeper-backed Pyxis runtime
that will qualify and run the Q30T canary, rather than inventing a weaker host
runtime boundary.

## Decision

Run each row-schema observation as a one-node keeper-backed Pyxis operation.
The operation consumes an independently reviewed Q30T runtime qualification
receipt and the exact archive/image/tree identities bound by that receipt.
The host runner performs only source authentication, keeper supervision,
container launch, receipt reconciliation, and no-clobber publication.

The observer executes inside the named container after the qualified runtime
archive has been extracted and its canonical tree digest has been replayed.
The container imports PyArrow only from that authenticated runtime tree. It
does not search host site-packages, enable `site`, execute `.pth` files, or
fall back to a caller-selected Python environment.

## Alternatives Considered

### Build a dedicated PyArrow container

This would provide an immutable dependency tree, but it creates a second image,
qualification path, and approval root solely for dataset observation. It also
risks scientific drift between observation and training runtimes.

### Extract the runtime archive into private scratch and import it on the host

This is operationally simple, but mode `0700` does not protect against a
same-UID process. Rehashing after import detects mutation only after untrusted
code may already have executed. This option is rejected.

### Use a different Parquet implementation

No separately reviewed Parquet implementation is already available on Ptyche.
Adding one would create another dependency and identity contract without
removing the runtime qualification requirement.

## Components and Ownership

### Runtime qualification producer

The existing runtime qualification subproject owns:

- archive/image staging through `ptv23_node_keeper.py`;
- named-container creation and reuse;
- archive-tree replay;
- exact Python, Pyxis control-plane, enroot, CUDA, and keeper evidence;
- one-node and two-node canonical qualification receipts.

The row observer consumes these receipts. It does not duplicate or weaken their
validation.

### Row observation submitter

`submit_q30t_row_schema_observation.sh` gains required arguments for:

- the reviewed runtime qualification receipt path and physical file SHA-256;
- the runtime archive path and SHA-256;
- the runtime image path and SHA-256;
- the expected canonical runtime-tree SHA-256.

It continues to require a clean pushed signed commit, approved Git remote,
sterile SSH/Git execution, exact approved output/log paths, and successful
`sbatch --test-only` before real submission. Caller-provided runtime values must
equal the replayed qualification receipt; they are not independent trust roots.

### Row observation runner

`run_q30t_row_schema_observation.sbatch` performs these phases:

1. Authenticate the spooled runner, committed observer, atomic publisher,
   keeper, runtime attestation tool, and their explicit import closure from the
   pushed Git commit.
2. Replay the qualification receipt and require the exact cluster/profile,
   archive, image, runtime tree, tool, and source-commit identities.
3. Allocate descriptor-bound node-local scratch using the reviewed Ptyche
   `/raid/scratch` contract.
4. Start one keeper that stages the archive, image, and committed tools by exact
   SHA-256 and publishes descriptor anchors.
5. Start one named Pyxis container from the keeper-backed image anchor and mount
   the keeper-backed archive and immutable control inputs.
6. Extract the archive inside container-private node scratch, replay the shared
   canonical runtime-tree digest, and invoke its Python with `-I -S` plus an
   explicit authenticated site-packages root.
7. Execute the held observer bytes over the exact PTV2/PTV3 inputs and publish
   the canonical observation through the held-parent no-clobber publisher.
8. Reconcile container/keeper evidence, require the keeper to remain live
   through publication, then stop it with bounded process-group supervision.

Every child process receives an explicit sterile environment. No phase uses
`SLURM_EXPORT_ENV=ALL`, inherited `BASH_ENV`, inherited `PYTHONPATH`, or a
PATH-selected security-critical executable.

### Observation schema

The observation retains its existing source-file and row-schema evidence and
adds a `runtime` object containing:

- qualification receipt physical file SHA-256 and embedded receipt SHA-256;
- archive, image, and canonical runtime-tree SHA-256;
- runtime profile physical file SHA-256;
- source commit and exact runner/observer/keeper/attestation tool SHA-256s;
- observed Python executable/version and PyArrow version/origin/tree digest;
- named-container and keeper evidence digests.

The runtime object participates in the observation self-hash. Observation A and
B must be byte-identical, so they must use the same reviewed runtime
qualification receipt and immutable inputs.

## Failure Semantics

- Missing or unapproved runtime qualification evidence fails before keeper or
  container launch.
- Archive/image/tree, source, tool, or profile mismatch fails closed.
- Keeper death cancels the active container/process group and suppresses the
  final observation.
- Container launch, extraction, import, scan, or reconciliation failure
  suppresses the final observation.
- Existing exact output is adopted only after full semantic replay; a foreign
  or different output is preserved and rejected.
- No failure path recursively deletes shared or possibly foreign data. Private
  node-local scratch may be retained for scheduler cleanup.

## Verification

Local behavioral tests must cover:

- missing/mismatched qualification receipt and runtime identities;
- dirty or unpushed source and mutable checkout substitution;
- exact fake-`srun` traces for keeper, named container, observer, publication,
  and shutdown phases;
- keeper death during extraction, import, scan, and publication;
- archive/tree, PyArrow origin, tool blob, mount, and output substitutions;
- bounded FIFO/process-group cancellation and no final output on any failure;
- exact existing-output adoption and foreign-output preservation.

Ptyche operations must run in this order:

1. review and approve one-node, then two-node runtime qualification receipts;
2. run row observation A through `--test-only`, then real submission;
3. run row observation B through `--test-only`, then real submission;
4. require canonical JSON, valid self-hashes, 201 PTV2 files, 100 PTV3 files,
   670,399 PTV3 rows, and byte-identical A/B outputs;
5. independently review the observation physical file SHA-256 before creating
   the Task 4B source mapping and inventory producer.

## Downstream Gate

This design enables row-schema evidence and the source-inventory path. It does
not remove the separate requirement for authoritative held-out UUID sources for
`speed`, `math`, `code`, `swe`, and `tool`. The final 700K dataset build and
training remain blocked until those five evidence roots are produced and
independently approved.
