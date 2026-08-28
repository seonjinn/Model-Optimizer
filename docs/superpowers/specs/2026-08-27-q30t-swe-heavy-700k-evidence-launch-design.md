# Q30 Thinking SWE-Heavy 700K Evidence and Launch Design

## Goal

Produce and authenticate the remaining evidence required to build the exact
Qwen3-30B-A3B-Thinking-2507 SWE-heavy 700K continuation corpus on Ptyche,
qualify the existing runtime on the target platform, and then launch matched
DFlash and DSpark B8 training.

The immutable training contract is:

- scientific identity:
  `ptv2-ptv3-complement-700k-qwen3-30ba3b-thinking-swe-heavy-v1`;
- 700,000 occurrences exactly once;
- 1,368 optimizer steps;
- steps 1-1,367 use global batch 512;
- step 1,368 uses global batch 96;
- no redistribution, replacement, padding, cycling, or duplicate exposure;
- DFlash and DSpark both use block size 8 and their independently reviewed
  step-25,391 parent receipts.

## Current trusted state

The following Ptyche jobs completed successfully:

- tokenizer receipt job `2671354`;
- DFlash/DSpark parent receipt job `2671355`.

Their reviewed whole-file approval roots are:

- tokenizer:
  `5ba642c455e60b67eca295dce92dd7da47292fdba66c5f9d269669c14cafc509`;
- DFlash parent:
  `393a2b7c5cbe2037914cbfa531d1d6fdce2bcac2f4204aa001db89d345c3a500`;
- DSpark parent:
  `d5416b8fd9644802b42f04511791418bf00dd023f8c5df96593010be95ff2571`.

All three bind producer commit
`212d01516b32e5a81f077d7f2f26ec8de38707c6`, target revision
`144afc2f379b542fdd4e85a1fcd5e1f79112d95d`, and target tree
`7bd176a868273ca8acf8db0ec1ab528f7972b0ae5bcf67ed4b3726368b9f7d13`.
The parent pair additionally binds runtime archive
`4a20aee61f290c48bed22a84b4a0ae0cbdc54e3e3910854d253188c8854f5dc9`,
historical audit
`2469430c144d9b86850901df0a28cb810b437555ee894771b796b1386d1c18b5`,
and ordered historical occurrence digest
`863470b22925d74228d31b1c2433d9461d25e8d02a298cb3a08a6bc99be55060`.

The physical PTV2 and PTV3 source stages are complete. PTV2 contains 201
approved files. PTV3 contains 100 files and 670,399 rows, including sufficient
raw counts for the exact 180K SWE-v3, 60K interactive, and 160K tool quotas.

## Decomposition

The remaining work is split into three independently reviewable subprojects.
They may be implemented in parallel, but the final dataset build depends on
Subproject A and the 16-node canary depends on all three.

1. **Data evidence and corpus publication**
2. **Q30 controller trust integration**
3. **Ptyche runtime qualification**

No subproject may populate an approval root from a self-authored or unreviewed
receipt. Receipt generation, independent replay, approval insertion, and
consumer execution are separate transitions.

## Subproject A: Data evidence and corpus publication

### Historical lineage

The builder must authenticate the existing canonical `BaselineAudit` directly.
It must not publish a derivative historical receipt because the Q30 parent
receipts bind the original audit's whole-file SHA. The builder returns its
internal `HistoricalExclusion` view while retaining the original audit file
SHA as the lineage root.

Historical and candidate rows use one shared prompt identity implementation:

1. retain only messages whose role is `system`, `developer`, or `user`;
2. retain the tool schema;
3. remove only the storage-only fields already defined by
   `specdec_identity.STORAGE_ONLY_FIELDS`;
4. canonicalize with `specdec_identity.canonicalize_prompt`;
5. hash with SHA-256.

This is the existing baseline audit identity and becomes the exclusion and
global-admission identity for new candidates. Assistant completions do not
change prompt identity. The full messages/tools bytes remain independently
bound in the selected-row output and token evidence.

### Held-out evidence

The required names remain exactly `speed`, `math`, `code`, `swe`, and `tool`.
Each evaluation owner supplies one immutable canonical source artifact whose
UUIDs use the shared prompt identity above. The producer:

- requires the exact name, source path, source whole-file SHA, repository or
  dataset identity, revision, and variant identity;
- validates stable single-link regular input bytes;
- rejects uppercase, duplicate, unsorted, malformed, or non-64-hex UUIDs;
- emits the existing canonical `specdec-held-out-uuid-receipt-v1` consumer
  bytes;
- emits a separate canonical provenance receipt binding the consumed receipt
  to its source identity and source SHA;
- publishes with the reviewed no-clobber atomic primitive;
- never synthesizes an empty receipt when an authoritative held-out source is
  unavailable.

Receipt ordering is canonical by the fixed name order above, independent of
CLI argument order. The five whole-file hashes are reviewed together before
entering `APPROVED_HELD_OUT_RECEIPT_FILE_SHA256S`.

### Source inventory and row-schema evidence

One authenticated CPU producer consumes:

- the exact PTV2 `SOURCE_PLAN.json` and completion receipt;
- the exact PTV3 `MANIFEST.json` and completion receipt;
- a checked-in eight-category source mapping policy;
- the exact SWE-heavy target policy and quota config.

The producer emits the existing
`ptv2-ptv3-complement-source-inventory-v1` schema in exact quota-category
order. It records each source repository, revision, split, license, replay
lane, physical path, byte count, SHA-256, and row count.

For every physical JSONL or Parquet file, not merely the first row, the
producer proves the declared messages/tools columns and value shapes. Files
within one source must agree on the canonical source-level row-schema
descriptor. Heterogeneous or malformed files fail closed. The reviewed output
whole-file SHA, PTV3 source tuples, and observed row-schema hashes become the
builder approval roots.

The producer reuses source hashes already bound by the staged PTV2/PTV3
receipts and performs live stable-file revalidation on the compute node. It
does not run a new unbounded login-node scan.

### Target policy and dataset build

The SWE-heavy quota config and target policy are canonicalized with target and
tokenizer revision
`144afc2f379b542fdd4e85a1fcd5e1f79112d95d`. A distinct SWE-heavy source
requirements file carries the SWE-heavy scientific identity; the balanced
file's embedded identity is not silently reinterpreted.

The builder obtains tokenizer trust through one centralized approved Q30
tokenizer loader. After the five held-out receipts and source inventory pass
independent review, their exact hashes, the original historical audit hash,
and the tokenizer receipt hash are inserted into the approval roots. The
builder then runs once on a compute node and publishes DATA, MANIFEST,
capacity evidence, and completion receipt using the hardened held-descriptor
publication boundary.

The completed bundle is replayed from scratch before its whole-file completion
SHA can be approved.

## Subproject B: Q30 controller trust integration

The production controller uses native Q30 loaders. It must not adapt the Q4
continuation contract.

### Tokenizer and parents

One centralized `load_q30t_tokenizer_receipt(path, expected_sha256)` performs:

- stable single-link file reading;
- caller SHA verification;
- membership in the exact tokenizer approval set;
- full `verify_q30t_tokenizer_receipt` replay.

Both builder and controller use that loader. The Q30 controller continues to
use `load_q30t_parent_receipt` directly, validates the full contract tuple,
and requires the DFlash/DSpark pair to share the same historical lineage.

### Dataset semantic boundary

The controller's current row scan remains defense-in-depth, not the source of
truth. A narrow dataset-replay adapter invokes the authoritative builder
semantic verifier from immutable Git objects extracted from the approved
source commit under isolated Python and a sterile environment. It never
imports `examples/dataset` from the mutable checkout pathname.

The training descriptor and completion authentication bind at least:

- builder source commit and exact builder file hash;
- target-policy and quota-config hashes;
- source inventory path and whole-file hash;
- original historical audit hash and ordered occurrence digest;
- tokenizer receipt path and whole-file hash;
- canonical ordered five-name held-out receipt map;
- capacity receipt path and whole-file hash;
- DATA and MANIFEST paths and hashes;
- completion receipt path and whole-file hash;
- completion `runtime_sha256` and `source_commit`.

Every field is compared to the contract. Missing, extra, or substituted fields
fail before any keeper or training step starts.

The controller's immutable scientific identity and quota tuple become the
SWE-heavy values. Q4 balanced policy and continuation behavior remain
unchanged.

## Subproject C: Ptyche runtime qualification

The existing runtime archive and image are reused; they are not rebuilt:

- archive SHA:
  `4a20aee61f290c48bed22a84b4a0ae0cbdc54e3e3910854d253188c8854f5dc9`;
- image SHA:
  `e7be53f2754097c88f7c801da92f6d94794ec4d78d9df937fcd315a6994297f0`.

Qualification has three sequential receipts.

1. **Archive/tree receipt**: stable-hash the archive, extract into fresh
   node-local scratch, compute the controller's canonical runtime tree hash,
   and publish a canonical archive-to-tree receipt.
2. **One-node keeper/Pyxis receipt**: stage archive, image, contract, and tools
   through the keeper; start a named container; verify architecture, exact
   Python identity/version and import origins, four visible GPU identities,
   Pyxis/enroot versions, anchor inode identity, sentinel inventory, and named
   container reuse. Kill the keeper and require a fresh descriptor-backed use
   to fail.
3. **Two-node aggregate receipt**: require the exact reviewed one-node receipt,
   repeat the phases on two unique nodes, and reconcile distinct node,
   keeper-PID, and anchor-inode evidence plus common identities.

Per-node and aggregate receipts are canonical and self-hashed. The aggregate
binds the cluster profile, source commit, runner/tool hashes, archive/image/tree
hashes, exact node set, ordered per-node receipt hashes, named-container reuse,
and keeper-loss negative result.

Only after independent review are the archive-to-tree mapping, image SHA, and
two-node qualification whole-file SHA inserted into controller approval roots.
The 16-node canary contract carries the qualification receipt path/hash and
the exact cluster-profile identity.

## Operational sequence

1. Implement and review Subprojects A-C using strict RED-GREEN cycles.
2. Commit and push exact reviewed source before every Ptyche submission.
3. Generate historical/source/held-out evidence with CPU-only jobs where
   authoritative inputs exist.
4. Independently replay and approve the evidence.
5. Build and fully replay the exact 700K dataset.
6. Qualify runtime with test-only, one-node, then two-node jobs.
7. Run DFlash and DSpark 20-step canaries in parallel.
8. Reconcile canary exposure, runtime, parent, and dataset receipts.
9. Launch both 1,368-step full continuations in parallel.

Every job uses `sbatch --test-only` first, a clean pushed source commit, bounded
logs under the experiment receipt root, and at least five minutes of
one-query-per-minute monitoring.

## Failure behavior

- Unknown scientific identities, revisions, receipt hashes, source files,
  schemas, or runtime evidence fail before training.
- Existing destinations are adopted only when exact bytes and full metadata
  match; foreign paths are preserved and rejected.
- Publication never recursively deletes or pathname-unlinks recovery trees.
- Unsupported held-out provenance blocks only the dataset build; it is never
  replaced with an empty exclusion set.
- A keeper death cancels the active probe, reconciliation, or training step
  and suppresses the final receipt.
- A canary failure does not authorize full training.

## Verification requirements

Each subproject has focused behavioral RED tests for its trust boundary and a
fresh independent hostile review before commit. The integrated gate includes:

- canonical JSON and self-hash replay;
- stable single-link and descriptor-rebind attacks;
- prompt identity agreement between historical audit and new candidates;
- missing/duplicate/reordered held-out roots;
- every-file row-schema and physical count/hash checks;
- immutable Git-object execution and mutable-checkout substitution attacks;
- descriptor round-trip mutation for every new path/hash;
- runtime keeper death during all probe/training phases;
- exact 700K exposure and final GBS96 behavior;
- DFlash and DSpark method isolation with shared dataset/runtime lineage.

Ruff format/check, scoped Pyright, `py_compile`, `bash -n`, ShellCheck, and
`git diff --check` must be clean on every frozen slice.

## Non-goals

- Rebuilding the already proven runtime archive or image.
- Reusing the Q4 continuation contract as a Q30 security boundary.
- Guessing held-out sources, source licenses, row schemas, or approval hashes.
- Treating successful receipt production as approval without independent
  replay.
- Starting full training before both canaries and the two-node runtime probe
  are complete.
