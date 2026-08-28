# Q30T Runtime Immutability Amendment

**Date:** 2026-08-28

**Amends:**

- `docs/superpowers/specs/2026-08-27-q30t-swe-heavy-700k-evidence-launch-design.md`
- `docs/superpowers/plans/2026-08-27-q30t-700k-runtime-qualification-plan.md`
- `docs/superpowers/specs/2026-08-28-q30t-row-observation-qualified-runtime-design.md`
- `docs/superpowers/plans/2026-08-28-q30t-row-observation-qualified-runtime-plan.md`

## Finding

Runtime Qualification Task 5 does not establish an immutable execution
boundary. It extracts the reviewed archive into same-UID-writable
`$NODE_SCRATCH`, hashes that tree, executes its Python, and adds a mutable
checkout to `PYTHONPATH`. Its keeper cleanup rechecks a pathname and then
unlinks it, leaving a replacement race. A same-UID process can change code
after the last hash and before or during import. The existing receipt cannot
authorize row observation, canary, or training.

The first amendment draft also assumed Pyxis could consume an unlinked
`O_TMPFILE` through `/proc/{pid}/fd/{fd}` and that one such scalar worked on
multiple nodes. Upstream Pyxis/enroot defeats both assumptions: enroot
canonicalizes the image path, and a procfd is local to one node. A named Pyxis
container creates an extracted, same-UID-writable enroot rootfs. All three
behaviors are now explicit qualification failures.

## Decision and Boundary

Build one reproducible, content-addressed SquashFS from:

- base SQSH SHA-256
  `e7be53f2754097c88f7c801da92f6d94794ec4d78d9df937fcd315a6994297f0`;
- runtime archive SHA-256
  `4a20aee61f290c48bed22a84b4a0ae0cbdc54e3e3910854d253188c8854f5dc9`.

The archive tree is installed at `/opt/q30t-runtime`. Two isolated normalized
builds must produce byte-identical SQSH bytes and identical complete content
manifests. The retained file is:

```text
/lustre/fsw/coreai_dlalgo_llm/users/sna/modelopt-qwen3-drafter-training/
  assets/q30t-runtime-derived-v1/sha256/{derived_image_sha256}.sqsh
```

No qualification, row, canary, or training consumer extracts the archive or
imports control code from a checkout. The same approved image is the only
runtime image eligible for all four consumers. This amendment authorizes no
training job.

Two mechanisms are intentionally separate:

1. An administrator-owned platform gateway and build service provide pathname
   and build protection domains that a same-UID process cannot create.
2. A keeper-owned Linux read lease on the exact inode Pyxis opens is the
   same-UID write tripwire during an operation.

The gateway receives the keeper's read-only descriptor with `SCM_RIGHTS`,
validates the lease on that same open file description, and links the initially
anonymous inode without copying into a root-owned node-local directory on the
same `/raid` filesystem. The file remains owned by the experiment UID and mode
`0400`: Pyxis can read it, the UID cannot replace or unlink the root-owned
directory entry, and an owner `fchmod` plus `O_WRONLY` reaches the lease and
triggers the required break. No producer path is reused or deleted.

For each operation, the gateway constructs
`{protected_runtime_root}/{job_id}/{operation_id}/runtime.sqsh` and publishes
that same absolute pathname on every allocated node. Job ID is canonical
decimal and operation ID is 64 lowercase hex. The path text and bytes are
identical, while each node has its own inode,
keeper PID, descriptor, lease, gateway attestation, and staging agent. Node
agents stage and hold only. Qualification, row observation, and training use
one global Slurm step so distributed rank and collective semantics are not
changed. Pyxis receives the regular protected pathname, never a procfd, and
must use unnamed temporary-rootfs direct SquashFUSE mode. `--container-name`,
`--container-save`, and extracted rootfs reuse are forbidden.

## Threat Model

The adversary can run arbitrary processes as the experiment UID during a job.
It can inspect readable names, chmod owned files, try procfd/pidfd duplication,
signal processes, drain unprotected pipes, and race mutable data. It cannot
modify reviewed Git blobs without a source change, enter an administrator
protection domain, write a root-owned directory, forge a gateway/reviewer
signature, or bypass kernel Yama and cgroup permissions proved by feasibility.

Same-UID secrecy is never a premise. Nonces prevent accidental cross-operation
binding but do not replace peer credentials, Yama, cgroup identity, exact
executable hashes, or signatures. Killing any trusted process is denial of
service and must never yield an accepted receipt.

## Mandatory Phase F Feasibility

Producer `P` may not freeze until a separately committed and pushed
feasibility-only producer `F` has run on Ptyche and its signed receipt has been
independently reviewed. `F` cannot import the runtime approval module, emit a
derived-image or qualification receipt, or change approval roots. Its only
external mode is `--platform-feasibility`, and that mode requires an explicit
`--node-count 1`, `--node-count 2`, or `--node-count 16`.

The required root-owned gateway/build service is not presumed to exist. Phase
F must prove the real installed service, ownership, key, executable, path
identity, site configuration, and direct-mount behavior. If the service is
absent or any proof below fails, the normative result is **BLOCKED**: P is not
frozen, no derived image is approved, and no row or training runtime may use
this design. A document, local mock, or `sbatch --test-only` cannot satisfy F.

The one-node, then two-node, then sixteen-node Phase F jobs prove:

1. `/raid` supports `O_TMPFILE`, read-only procfd reopen, `F_SETLEASE`,
   `F_GETLEASE`, `SCM_RIGHTS`, and same-UID blocking write-open behavior.
2. The `SCM_RIGHTS` receiver observes `F_RDLCK`, proving the same open file
   description. A separately reopened procfd is explicitly rejected as proof.
3. `/proc/sys/kernel/yama/ptrace_scope >= 1`; a sibling same-UID process gets
   `EPERM` from `pidfd_getfd` against keeper, agent, gateway, and barrier
   control fds. Authenticated parent-child fd passing works. `pidfd_open` plus
   `/proc/{pid}/stat` start ticks rejects PID reuse.
4. The gateway accepts the held anonymous inode, links it without copying on
   the same device, preserves device/inode/lease identity, protects the name,
   and signs an attestation. Same-UID `fchmod(0600)` plus blocking `O_WRONLY`
   generates the break event before a byte can change.
5. An unnamed Pyxis operation accepts the protected regular path, logs
   `skipping container creation (squashfuse enabled)`, mounts its exact inode
   read-only, and leaves no persistent rootfs. F rejects `creating container
   filesystem`, an `ENROOT_DATA_PATH` root, procfd realpath failure, or named
   reuse. The root-owned gateway inspects the live SquashFUSE process and signs
   its PID/start/cgroup plus the exact open backing fd device/inode; argv/logs
   or a pre-launch `stat` alone are not actual-open proof.
6. One staging dispatch places identical bytes at the identical absolute path
   on two and then all sixteen nodes with distinct keeper PIDs/fds/inodes. One
   subsequent global execution step uses exact `--nodes=N --ntasks=N
   --ntasks-per-node=1 --gpus-per-node=4` for `N=2` and `N=16`, mounts every
   node-local inode, and completes a collective with exactly N global ranks.
   The 16-node run proves the segment-16 Q30 canary/full data-plane shape.
   Per-node procfd argv and nested execution steps are explicitly rejected.
7. The installed gateway's distributed barrier authenticates one registered
   controller, every local agent, and every global-step wrapper by
   `SO_PEERCRED`, pidfd/start ticks, job/step cgroup, node, and exact executable
   blob. All wrappers remain held after READY until one global GO/ABORT
   transaction. Attack sessions are permanently ABORT-only.
8. Cgroup v2 exposes each global task's step directory. Exact-step
   cancellation, authenticated per-node exit or scheduler terminal state,
   `cgroup.events` `populated 0` on every node, local `srun` reap, and exact
   allocation state are observable within the lease deadline.
9. The root-owned build service runs two workers in protection domains whose
   UID, mount namespace, scratch root, and candidate fds are inaccessible to
   the producer UID. It executes the pinned hermetic builder image, not host
   SquashFS tools.
10. Lustre supports receipt publication by `O_TMPFILE`,
    `linkat(AT_EMPTY_PATH)`, file/parent fsync, and exact-existing adoption.
11. Pyxis, enroot, Slurm SPANK, SquashFUSE, gateway, build service, and every
    `/etc/enroot` entry form exact recursive path-to-SHA manifests. Site
    `hooks.d`, `mounts.d`, and `environ.d` are enumerated; user equivalents are
    neutralized by gateway-owned empty HOME/XDG config roots.
12. Batch, staging, Pyxis, task stdout, and task stderr remain node-local and
    within byte limits before bounded durable publication.

### Feasibility receipt

```python
@dataclass(frozen=True)
class RuntimeServiceEndpointIdentity:
    service_kind: Literal["gateway", "build-service"]
    endpoint_node_name: str
    service_identity: Literal[
        "systemd:q30t-runtime-gateway.service",
        "systemd:q30t-runtime-build.service",
    ]
    executable_path: str
    executable_sha256: str
    expected_service_uid: Literal[0]
    expected_service_gid: int
    control_socket_path: str
    control_socket_device: int
    control_socket_inode: int
    control_socket_uid: Literal[0]
    control_socket_gid: int
    control_socket_mode: Literal[432]  # 0o660
    public_key_path: str
    public_key_sha256: str
    identity_sha256: str


@dataclass(frozen=True)
class RuntimePlatformFeasibilityRunReceipt:
    schema_version: Literal["q30t-runtime-platform-feasibility-run-v1"]
    cluster: Literal["ptyche"]
    node_count: Literal[1, 2, 16]
    producer_commit: str
    feasibility_bound_blobs_sha256: str
    job_id: str
    ordered_nodes: tuple[str, ...]
    prerequisite_node_count: Literal[1, 2] | None
    prerequisite_receipt_path: str | None
    prerequisite_receipt_file_sha256: str | None
    producer_uid: int
    yama_ptrace_scope: int
    raid_filesystem_type: str
    lustre_filesystem_type: Literal["lustre"]
    lease_break_time_seconds: int
    scm_rights_same_ofd_proven: Literal[True]
    sibling_pidfd_getfd_denied: Literal[True]
    protected_link_same_inode_proven: Literal[True]
    same_uid_write_break_proven: Literal[True]
    sigio_setup_order_proven: Literal[True]
    unnamed_direct_squashfuse_proven: Literal[True]
    live_backing_fd_same_inode_proven: Literal[True]
    persistent_rootfs_absent: Literal[True]
    global_multinode_path_routing_proven: bool
    global_go_abort_barrier_proven: Literal[True]
    one_global_execution_step_proven: Literal[True]
    global_srun_argv_sha256: str
    global_rank_count: int
    segment_count: Literal[16] | None
    collective_artifact_file_sha256: str
    remote_exit_proof_proven: Literal[True]
    ordered_gateway_endpoint_identities: tuple[RuntimeServiceEndpointIdentity, ...]
    protected_runtime_root: str
    build_service_endpoint_identity: RuntimeServiceEndpointIdentity
    hermetic_builder_image_path: str
    hermetic_builder_image_sha256: str
    hermetic_builder_closure_manifest_sha256: str
    pyxis_plugin_path: str
    pyxis_plugin_sha256: str
    pyxis_version: str
    enroot_path: str
    enroot_sha256: str
    enroot_version: str
    squashfuse_path: str
    squashfuse_sha256: str
    slurm_spank_manifest_sha256: str
    enroot_system_config_manifest_sha256: str
    empty_user_config_root_device: int
    empty_user_config_root_inode: int
    ordered_probe_artifact_file_sha256s: tuple[str, ...]
    ordered_bounded_log_file_sha256s: tuple[str, ...]
    receipt_sha256: str


@dataclass(frozen=True)
class RuntimePlatformFeasibilityReceipt:
    schema_version: Literal["q30t-runtime-platform-feasibility-v3"]
    cluster: Literal["ptyche"]
    producer_commit: str
    feasibility_bound_blobs_sha256: str
    one_node_receipt_path: str
    one_node_receipt_file_sha256: str
    one_node_job_id: str
    two_node_receipt_path: str
    two_node_receipt_file_sha256: str
    two_node_job_id: str
    sixteen_node_receipt_path: str
    sixteen_node_receipt_file_sha256: str
    sixteen_node_job_id: str
    ordered_nodes: tuple[str, ...]
    ordered_gateway_endpoint_identities: tuple[RuntimeServiceEndpointIdentity, ...]
    protected_runtime_root: str
    build_service_endpoint_identity: RuntimeServiceEndpointIdentity
    hermetic_builder_image_path: str
    hermetic_builder_image_sha256: str
    hermetic_builder_closure_manifest_sha256: str
    ordered_run_receipt_sha256s: tuple[str, str, str]
    receipt_sha256: str


def publish_platform_feasibility_receipt(
    receipt: RuntimePlatformFeasibilityReceipt,
    *, output_path: Path,
    publication_job_id: str,
) -> RuntimePlatformFeasibilityReceipt: ...


def load_platform_feasibility_receipt(
    path: Path, expected_file_sha256: str
) -> RuntimePlatformFeasibilityReceipt: ...


def load_platform_feasibility_run_receipt(
    path: Path,
    expected_file_sha256: str,
    *,
    expected_node_count: Literal[1, 2, 16],
) -> RuntimePlatformFeasibilityRunReceipt: ...
```

Each run loader stable-reads canonical JSON, requires exactly `node_count`
unique nodes and all applicable proof literals, verifies every physical
artifact/log, and rejects a producer UID equal to either service protection
identity. The one-node run requires all prerequisite fields null; the two-node
run requires prerequisite count 1 and the exact one-node path/hash; the
sixteen-node run requires prerequisite count 2 and the exact two-node
path/hash. The aggregate loader requires all three separately submitted job
IDs to be pairwise distinct, reloads all three run receipts, and enforces the
exact 1 -> 2 -> 16 prerequisite chain. Every run requires one global step,
`global_rank_count == node_count`, an exact argv hash, and a replayed collective
artifact. Segment count is null for node counts 1/2 and exactly 16 for node
count 16. Gateway endpoint order/cardinality must
equal each run's node tuple. Node-set inclusion is not required across runs,
but every node repeated in a later run must have a byte-identical endpoint
identity. The build coordinator identity must be byte-identical in all three
runs. The approved
sixteen-node tuple is the exact node set for segment-16 Q30 canary/full
admission; it cannot be replaced, reordered, shortened, enlarged, or deferred
to a later feasibility chain. It requires gateway and build `service_kind`,
exact systemd identity
literals, uid 0, nonnegative expected gid, root-owned socket uid, socket gid
equal expected service gid, mode 0660, root-owned no-follow socket parents,
and distinct device/inode identities. It re-lstats the live sockets and
stable-hashes both executable and public-key files whenever it reconstructs
an endpoint.

### Protected gateway client interface

The installed service protocol is frozen rather than inferred from a mock:

```python
@dataclass(frozen=True)
class RuntimeGatewayEndpoint:
    identity: RuntimeServiceEndpointIdentity
    protected_runtime_root: Path


@dataclass(frozen=True)
class ProtectedBuildServiceEndpoint:
    identity: RuntimeServiceEndpointIdentity
    hermetic_builder_image_path: Path
    hermetic_builder_image_sha256: str
    hermetic_builder_closure_manifest_sha256: str


def reconstruct_runtime_gateway_endpoint(
    feasibility: RuntimePlatformFeasibilityReceipt,
    *, node_name: str,
) -> RuntimeGatewayEndpoint: ...


def reconstruct_protected_build_service_endpoint(
    feasibility: RuntimePlatformFeasibilityReceipt,
) -> ProtectedBuildServiceEndpoint: ...


def connect_runtime_gateway(
    feasibility: RuntimePlatformFeasibilityReceipt,
    *, node_name: str,
) -> "RuntimeGatewayClient": ...


@dataclass(frozen=True)
class GlobalOperationRegistration:
    schema_version: Literal["q30t-global-operation-registration-v1"]
    job_id: str
    operation_id: str
    expected_nodes: tuple[str, ...]
    protected_runtime_path: str
    derived_image_sha256: str
    policy: Literal["GO_AFTER_ALL_READY", "ABORT_ONLY"]
    controller_pid: int
    controller_start_ticks: int
    registration_sha256: str
    gateway_signature: str


class RuntimeGatewayClient(Protocol):
    def publish_leased_image(
        self, *, leased_image_fd: int, ready: "KeeperSameOFDReady"
    ) -> "ProtectedImagePublication": ...

    def publish_control_tree(
        self,
        *,
        request: "ProtectedControlPublishRequest",
        ordered_source_fds: tuple[int, ...],
    ) -> "ProtectedControlPublication": ...

    def register_global_operation(
        self,
        *,
        job_id: str,
        operation_id: str,
        expected_nodes: tuple[str, ...],
        protected_runtime_path: Path,
        derived_image_sha256: str,
        policy: Literal["GO_AFTER_ALL_READY", "ABORT_ONLY"],
        controller_pidfd: int,
        controller_start_ticks: int,
    ) -> GlobalOperationRegistration: ...

    def await_all_ready(
        self, registration: GlobalOperationRegistration, *, timeout_seconds: float
    ) -> tuple["SlurmTaskReady", ...]: ...

    def decide_global_operation(
        self,
        registration: GlobalOperationRegistration,
        *,
        decision: Literal["GO", "ABORT"],
    ) -> "OperationBarrierDecision": ...
```

The client accepts no caller-selected socket, key, protected root, peer UID, or
service executable outside the F-bound endpoint. It independently reconstructs
the operation path from root/job/operation and rejects a caller path mismatch.
The service refuses GO for an `ABORT_ONLY` registration even from the
authenticated controller. After `connect`, it requires `SO_PEERCRED` uid/gid to
equal the endpoint, opens a pidfd for the returned PID, binds its start ticks,
requires `/proc/{pid}/exe` to match executable identity, and requires the PID's
systemd cgroup to equal `service_identity`. Any socket inode/key/process restart
requires a new reviewed feasibility receipt; callers cannot adopt it.

## Protected Reproducible Build

The producer sends stable-open base, archive, reviewed archive-tree receipt,
and expected hashes to the root-owned build service. It cannot supply a build
root, tool path, environment variable, output path, or arbitrary argv. The
service authenticates `SO_PEERCRED`, job cgroup, and F-bound source commit.

Build A and B use distinct service workers, identities/user namespaces, mount
namespaces, private roots, and candidates. The experiment UID cannot access
those roots or writable fds. Each worker snapshots/hashes inputs, builds,
replays, signs an attestation, and closes all writers. After byte/content
equality, the service, not the producer UID, copies one candidate into a
service-owned Lustre `O_TMPFILE`, fsyncs/hashes it, sets owner `0:0` and mode
`0444`, links it no-clobber at the digest-derived path, fsyncs its root-owned
parent, and signs the publication. Exact existing root-owned bytes are adopted;
foreign or user-owned targets are preserved and rejected. Failure preserves
service-private evidence and publishes no image receipt.

The descriptor-only service call is exact:

```python
@dataclass(frozen=True)
class ProtectedBuildRequest:
    schema_version: Literal["q30t-protected-build-request-v1"]
    build_label: Literal["A", "B", "review-A", "review-B"]
    source_commit: str
    base_image_sha256: str
    runtime_archive_sha256: str
    archive_tree_receipt_file_sha256: str
    runtime_install_root: Literal["/opt/q30t-runtime"]
    hermetic_builder_image_sha256: str
    request_nonce: str
    request_sha256: str


class ProtectedBuildServiceClient(Protocol):
    def build(
        self,
        request: ProtectedBuildRequest,
        *,
        base_image_fd: int,
        runtime_archive_fd: int,
        archive_tree_receipt_fd: int,
    ) -> "ProtectedBuildAttestation": ...

    def publish_verified_pair(
        self,
        *,
        build_a: "ProtectedBuildAttestation",
        build_b: "ProtectedBuildAttestation",
    ) -> "ProtectedDerivedImagePublication": ...


def connect_protected_build_service(
    feasibility: RuntimePlatformFeasibilityReceipt,
) -> ProtectedBuildServiceClient: ...


def request_protected_build(
    *,
    request: ProtectedBuildRequest,
    base_image_fd: int,
    runtime_archive_fd: int,
    archive_tree_receipt_fd: int,
    service: ProtectedBuildServiceClient,
) -> "ProtectedBuildAttestation": ...
```

No writable or read-only candidate fd crosses into the producer domain. The
service attestation proves every writable candidate fd was closed before its
candidate entered the final comparison/publication stage.

The hermetic builder-image SHA binds the complete dynamic-library closure.
Host `mksquashfs`, `unsquashfs`, `tar`, or `zstd` is never invoked. Exact argv:

```text
/usr/libexec/q30t-build-runtime --base-fd 3 --archive-fd 4
  --archive-tree-receipt-fd 5 --output-fd 6
  --runtime-root /opt/q30t-runtime --source-date-epoch 0
  --compression zstd --compression-level 19 --processors 1
```

The image-pinned tool uses `-noappend -all-root -all-time 0 -root-time 0
-mkfs-time 0 -no-xattrs -no-exports -comp zstd -Xcompression-level 19
-processors 1 -no-progress`, `env -i`, `SOURCE_DATE_EPOCH=0`, `TZ=UTC`,
`LC_ALL=C`, and `umask 077`. Its closure manifest enumerates entrypoint, ELF
interpreter, every `DT_NEEDED` object, config, and tool by image-relative path,
mode, size, and SHA-256. Any unlisted dynamic load fails.

Both workers apply one normalization policy: bytewise C-locale path order;
uid/gid `0:0`; all mtimes, root time, and mkfs time `0`; no xattrs/export
table; original base/archive permission bits with setuid/setgid/sticky cleared;
canonical symlink mode `0777`; and newly created `/opt/q30t-runtime`
directories mode `0755`. Archive modes must also equal the reviewed archive
tree receipt after normalization. One worker process and one compressor process
are permitted; no host-dependent parallel ordering is accepted.

### Complete content manifests

Metadata-only manifests are insufficient. Base, archive, and derived manifests
contain every entry in bytewise C-locale path order:

```python
@dataclass(frozen=True)
class RuntimeContentEntry:
    relative_path: str
    entry_type: Literal["directory", "regular", "symlink", "hardlink"]
    mode: int
    uid: Literal[0]
    gid: Literal[0]
    mtime_ns: Literal[0]
    size: int
    content_sha256: str | None
    symlink_target: str | None
    hardlink_group: str | None
    xattrs: tuple[tuple[str, str], ...]


@dataclass(frozen=True)
class RuntimeContentManifest:
    schema_version: Literal["q30t-runtime-content-manifest-v1"]
    subject_kind: Literal["base", "archive", "derived"]
    entries: tuple[RuntimeContentEntry, ...]
    manifest_sha256: str
```

Regulars require content SHA/size; symlinks require exact target; hardlinks
require canonical group and target content SHA; directories have no content
SHA. Xattrs are empty. Devices, FIFOs, sockets, absolute/traversal/duplicate
paths, and unsafe links fail. The derived manifest equals the base manifest
overlaid only by normalized `/opt/q30t-runtime` directories and the reviewed
archive manifest. The normalized base must not already contain
`/opt/q30t-runtime`; a collision fails instead of replacing base content.
Post-build replay reconstructs the whole manifest. Build A, B, and review
rebuilds must match.

### Derived receipt and loader

```python
@dataclass(frozen=True)
class ProtectedBuildAttestation:
    schema_version: Literal["q30t-protected-build-attestation-v1"]
    build_label: Literal["A", "B", "review-A", "review-B"]
    request_sha256: str
    worker_identity: str
    worker_uid: int
    mount_namespace_inode: int
    private_root_device: int
    private_root_inode: int
    candidate_device: int
    candidate_inode: int
    base_snapshot_sha256: str
    archive_snapshot_sha256: str
    base_content_manifest_sha256: str
    archive_content_manifest_sha256: str
    derived_content_manifest_sha256: str
    builder_image_sha256: str
    builder_closure_manifest_sha256: str
    builder_argv: tuple[str, ...]
    output_size: int
    output_sha256: str
    service_signature: str


@dataclass(frozen=True)
class ProtectedDerivedImagePublication:
    schema_version: Literal["q30t-protected-derived-publication-v1"]
    build_a_request_sha256: str
    build_b_request_sha256: str
    derived_image_path: str
    device: int
    inode: int
    nlink: Literal[1]
    uid: Literal[0]
    gid: Literal[0]
    mode: Literal[292]
    size: int
    sha256: str
    protected_parent_device: int
    protected_parent_inode: int
    service_signature: str


@dataclass(frozen=True)
class RuntimeDerivedImageReceipt:
    schema_version: Literal["q30t-runtime-derived-image-v2"]
    profile_path: str
    profile_file_sha256: str
    feasibility_receipt_path: str
    feasibility_receipt_file_sha256: str
    feasibility_review_receipt_file_sha256: str
    base_image_path: str
    base_image_size: int
    base_image_sha256: str
    runtime_archive_path: str
    runtime_archive_size: int
    runtime_archive_sha256: str
    archive_tree_receipt_path: str
    archive_tree_receipt_file_sha256: str
    runtime_tree_sha256: str
    runtime_install_root: Literal["/opt/q30t-runtime"]
    base_content_manifest_path: str
    base_content_manifest_file_sha256: str
    archive_content_manifest_path: str
    archive_content_manifest_file_sha256: str
    derived_content_manifest_path: str
    derived_content_manifest_file_sha256: str
    derived_content_manifest_sha256: str
    build_a: ProtectedBuildAttestation
    build_b: ProtectedBuildAttestation
    hermetic_builder_image_sha256: str
    hermetic_builder_closure_manifest_sha256: str
    source_commit: str
    qualification_bound_blobs_sha256: str
    derived_image_path: str
    derived_image_size: int
    derived_image_sha256: str
    protected_publication: ProtectedDerivedImagePublication
    receipt_sha256: str


@dataclass(frozen=True)
class RuntimeDerivedImageBuildInputs:
    profile_path: Path
    profile_file_sha256: str
    feasibility_receipt_path: Path
    feasibility_receipt_file_sha256: str
    feasibility_review: "SignedRuntimeReview"
    base_image_path: Path
    base_image_sha256: str
    runtime_archive_path: Path
    runtime_archive_sha256: str
    archive_tree_receipt_path: Path
    archive_tree_receipt_file_sha256: str
    source_commit: str
    derived_image_root: Path
    output_receipt_path: Path


def build_runtime_derived_image(
    inputs: RuntimeDerivedImageBuildInputs,
) -> RuntimeDerivedImageReceipt: ...


def load_runtime_derived_image_receipt(
    path: Path, expected_file_sha256: str
) -> RuntimeDerivedImageReceipt: ...
```

The loader validates both service signatures against F's pinned key; requires
distinct workers, protection domains, roots, and candidates; reconstructs all
manifests; requires byte-identical outputs; stable-hashes the retained SQSH;
requires root ownership, mode `0444`, protected-parent identity, and the signed
service publication; and rejects host-tool identity or caller-selected build
arguments.

### Anonymous durable publication

```python
@dataclass(frozen=True)
class PublishedRegularFile:
    path: Path
    size: int
    sha256: str
    device: int
    inode: int
    mode: int


def publish_anonymous_regular_file(
    *,
    source_fd: int,
    destination: Path,
    expected_size: int,
    expected_sha256: str,
    mode: int,
) -> PublishedRegularFile: ...


def publish_anonymous_bytes(
    *, destination: Path,
    payload: bytes,
    mode: int = 0o400,
) -> PublishedRegularFile: ...
```

These functions publish bounded receipts, manifests, and logs, not the retained
derived image. Both require destination-filesystem `O_TMPFILE` and
`linkat(AT_EMPTY_PATH)`, fsync the file and held no-follow parent fd, and use
no-clobber publication. Exact existing bytes are adopted; foreign bytes remain
untouched and fail. There is no named temporary, overwrite, rename, or unlink.

## Exact Qualification-Bound Blob Map

The profile and receipt reconstruct this literal map. A receipt-supplied list
or circular “every listed blob” rule is not accepted:

```python
FEASIBILITY_BOUND_BLOB_FIELDS_V1: tuple[tuple[str, str], ...] = (
    ("tools/launcher/common/specdec/q30t_runtime_platform_feasibility.py", "commit-blob"),
    ("tools/launcher/common/specdec/q30t_runtime_gateway_client.py", "commit-blob"),
    ("tools/launcher/common/specdec/q30t_anonymous_publish.py", "commit-blob"),
    ("tools/launcher/common/specdec/q30t_runtime_independent_review.py", "commit-blob"),
    ("tools/launcher/common/specdec/q30t_runtime_review_allowed_signers", "commit-blob"),
    ("tools/launcher/common/specdec/run_q30t_runtime_platform_feasibility.sbatch", "commit-blob"),
    ("tools/launcher/common/specdec/submit_q30t_runtime_platform_feasibility.sh", "commit-blob"),
)

QUALIFICATION_BOUND_BLOB_FIELDS_V1: tuple[tuple[str, str], ...] = (
    ("tools/launcher/common/specdec/q30t_ptv23_cluster_profile.py", "receipt.profile_file_sha256"),
    ("tools/launcher/common/specdec/q30t_runtime_platform_feasibility.py", "profile.platform_feasibility_tool_sha256"),
    ("tools/launcher/common/specdec/q30t_runtime_gateway_client.py", "profile.gateway_client_sha256"),
    ("tools/launcher/common/specdec/q30t_runtime_content_manifest.py", "profile.content_manifest_tool_sha256"),
    ("tools/launcher/common/specdec/q30t_anonymous_publish.py", "profile.anonymous_publisher_sha256"),
    ("tools/launcher/common/specdec/q30t_runtime_archive_receipt.py", "profile.archive_receipt_tool_sha256"),
    ("tools/launcher/common/specdec/q30t_runtime_identity.py", "profile.runtime_identity_tool_sha256"),
    ("tools/launcher/common/specdec/q30t_tree_digest.py", "profile.tree_digest_tool_sha256"),
    ("tools/launcher/common/specdec/q30t_derived_runtime_image.py", "profile.derived_image_builder_sha256"),
    ("tools/launcher/common/specdec/q30t_runtime_independent_review.py", "profile.independent_review_tool_sha256"),
    ("tools/launcher/common/specdec/q30t_runtime_review_allowed_signers", "profile.review_allowed_signers_sha256"),
    ("tools/launcher/common/specdec/q30t_runtime_admission.py", "profile.runtime_admission_tool_sha256"),
    ("tools/launcher/common/specdec/q30t_runtime_bootstrap.py", "profile.runtime_bootstrap_sha256"),
    ("tools/launcher/common/specdec/ptv23_node_keeper.py", "profile.keeper_tool_sha256"),
    ("tools/launcher/common/specdec/ptv23_keeper_supervisor.py", "profile.keeper_supervisor_sha256"),
    ("tools/launcher/common/specdec/ptv23_runtime_attestation.py", "profile.attestation_tool_sha256"),
    ("tools/launcher/common/specdec/probe_ptv23_pyxis_keeper.sbatch", "profile.probe_runner_sha256"),
    ("tools/launcher/common/specdec/run_q30t_derived_runtime_image.sbatch", "profile.derived_image_runner_sha256"),
    ("tools/launcher/common/specdec/submit_q30t_runtime_qualification.sh", "profile.qualification_submitter_sha256"),
)


@dataclass(frozen=True)
class QualificationProfileSelectorFreeze:
    schema_version: Literal["q30t-qualification-profile-selector-freeze-v1"]
    feasibility_receipt_file_sha256: str
    feasibility_review_receipt_file_sha256: str
    feasibility_review_signature_file_sha256: str
    ordered_path_selector_git_blob_sha256s: tuple[tuple[str, str, str], ...]
    profile_path: str
    profile_file_sha256: str
    freeze_sha256: str


def freeze_qualification_profile_selectors(
    *,
    repo_root: Path,
    profile_path: Path,
    feasibility_receipt: RuntimePlatformFeasibilityReceipt,
    feasibility_review: "SignedRuntimeReview",
) -> QualificationProfileSelectorFreeze: ...


def verify_index_profile_selectors(
    *, repo_root: Path, freeze: QualificationProfileSelectorFreeze
) -> QualificationProfileSelectorFreeze: ...
```

F reconstructs its seven-entry tuple directly from `producer_commit`; its
receipt stores only the resulting tuple SHA. P's profile supplies exactly one
64-lower-hex value for every `profile.*` selector, while each subject receipt
supplies its independently expected physical profile hash. This avoids asking
the profile to contain its own hash. Build, node, aggregate, and non-feasibility
review receipts store the qualification tuple SHA; feasibility review stores
the F tuple SHA in `subject_bound_blobs_sha256`. Loaders read every path at the
claimed commit, resolve only the literal selector, hash the Git blob, and
require ordered tuple equality. The approvals module is deliberately absent
because A changes only its two constant literals.

Task 2 may implement the literal path/selector schema and exercise it against
test fixtures, but it must not create or freeze P's production selector values.
After every mapped runtime file has reached its final Task 7 bytes, the producer
computes each Git blob ID from those exact bytes, writes all production
`profile.*` values once, stages the exact P-owned paths, and reruns the complete
lineage/map suite against the index. Any subsequent mapped-file change clears
the freeze and requires recomputation before P. Thus P's selector values cannot
describe a Task 2 predecessor of the code P actually commits.
`freeze_qualification_profile_selectors` accepts no caller path/selector map;
it uses the literal tuple above, requires every non-profile mapped path already
in the index, and writes only the profile. The verifier recomputes all index
blob IDs, the profile physical hash, the source-defined exact Task 7 cached-name
set `P_TASK7_CACHED_PATHS_V1`, and every mapped worktree/index equality; an
unstaged mapped delta or changed index invalidates the freeze. Neither set is
caller-supplied.

## Held Bootstrap and Same-OFD Keeper

Anonymous keeper descriptors are used only for the derived image. They cannot
name Python controls consistently inside one global multi-node `srun`.
Instead, every node agent stable-opens the exact committed control blobs named
by the literal qualification map and sends their read-only fds to the gateway.
The gateway copies them to `O_TMPFILE`s in its protection domain, verifies the
expected Git blob and physical hashes, fsyncs them, sets owner `0:0` and mode
`0444`, and no-clobber links one complete tree at
`{protected_runtime_root}/{job_id}/{operation_id}/control`. Every node has the
same absolute root, relative paths, sizes, modes, and bytes; device/inode pairs
are node-local. No control tree or file is reused, renamed, overwritten, or
deleted by receipt-producing code. The gateway itself derives canonical
`held-modules.json` from the ordered committed specs and publishes it as the
sole generated entry; the producer cannot supply its bytes. Every other entry
must be an exact commit blob.

```python
@dataclass(frozen=True)
class CommittedControlBlobSpec:
    module_name: str
    repo_relative_path: str
    control_relative_path: str
    sha256: str
    is_package: bool


@dataclass(frozen=True)
class ProtectedControlPublishRequest:
    schema_version: Literal["q30t-protected-control-publish-request-v1"]
    job_id: str
    node_name: str
    operation_id: str
    source_commit: str
    qualification_bound_blobs_sha256: str
    ordered_blobs: tuple[CommittedControlBlobSpec, ...]
    request_sha256: str


@dataclass(frozen=True)
class ProtectedControlEntry:
    entry_kind: Literal["commit-blob", "generated-manifest"]
    repo_relative_path: str | None
    control_relative_path: str
    device: int
    inode: int
    nlink: Literal[1]
    uid: Literal[0]
    gid: Literal[0]
    mode: Literal[292]  # 0o444
    size: int
    sha256: str


@dataclass(frozen=True)
class ProtectedControlPublication:
    schema_version: Literal["q30t-protected-control-publication-v1"]
    job_id: str
    node_name: str
    operation_id: str
    source_commit: str
    protected_control_root: str
    root_device: int
    root_inode: int
    root_uid: Literal[0]
    root_gid: Literal[0]
    root_mode: Literal[365]  # 0o555
    ordered_entries: tuple[ProtectedControlEntry, ...]
    control_manifest_sha256: str
    publication_sha256: str
    gateway_signature: str


@dataclass(frozen=True)
class HeldModuleSpec:
    module_name: str
    control_relative_path: str
    sha256: str
    is_package: bool


def load_protected_modules(
    *,
    control_root_fd: int,
    expected_control_manifest_sha256: str,
    modules: tuple[HeldModuleSpec, ...],
) -> Mapping[str, ModuleType]: ...


def run_protected_module(
    *,
    control_root_fd: int,
    expected_control_manifest_sha256: str,
    modules: tuple[HeldModuleSpec, ...],
    entry_module: str,
    argv: tuple[str, ...],
) -> int: ...
```

The host loader verifies every gateway signature and publication self/physical
hash before global `srun`, then requires the ordered node publications to have
identical commit, absolute root text, relative-path/hash/size/mode manifest,
and distinct node/device/inode identities. The container bootstrap opens the
already-mounted root once with `O_PATH|O_DIRECTORY|O_NOFOLLOW`, opens every
relative file with `openat2(RESOLVE_BENEATH|RESOLVE_NO_SYMLINKS|
RESOLVE_NO_MAGICLINKS)`, and requires root/file owner `0:0`, modes 0555/0444,
single links, regular files, size, content SHA, and the expected manifest SHA.
It requires exactly one `generated-manifest` entry at `held-modules.json`,
reconstructs its canonical bytes from the remaining commit-blob entries, and
requires every other non-null repo path to match the literal Git map at the
publication's source commit.
It opens and retains all fds before compiling any, then uses `compile` and
`importlib.util.module_from_spec` in declared dependency order. It never uses
`PYTHONPATH`, control-code `sys.path`, `SourceFileLoader`, `site`, `.pth`, a
checkout path, absolute control reopen, or import by pathname. A same-UID
process can read but cannot alter or replace the gateway-owned tree.

Each local agent creates an inherited
`AF_UNIX SOCK_SEQPACKET|SOCK_CLOEXEC` socketpair, enables `SO_PASSCRED` on the
receiving endpoint, and spawns its keeper. Cached `SO_PEERCRED` on a socketpair
created before spawn is not a keeper-PID proof and is never accepted. The
keeper sends one canonical READY packet plus its leased image fd in one
`sendmsg`; Linux attaches `SCM_CREDENTIALS` and the keeper attaches
`SCM_RIGHTS`. `SCM_RIGHTS` duplicates the same open file description; the
receiver's `F_GETLEASE` is therefore normative. Reopening
`/proc/{pid}/fd/{fd}` creates a different open file description and never
validates the keeper's lease.

```python
@dataclass(frozen=True)
class KeeperSameOFDReady:
    schema_version: Literal["ptv23-keeper-same-ofd-ready-v1"]
    job_id: str
    node_name: str
    operation_id: str
    keeper_pid: int
    keeper_start_ticks: int
    keeper_uid: int
    keeper_gid: int
    staged_device: int
    staged_inode: int
    staged_size: int
    staged_sha256: str
    staged_mode: Literal[256]
    staged_nlink_before_gateway: Literal[0]
    all_writer_descriptors_closed: Literal[True]
    all_threads_sigio_blocked: Literal[True]
    signal_owner_type: Literal["F_OWNER_PID"]
    signal_owner_pid: int
    sigwait_thread_armed_before_lease: Literal[True]
    lease_type: Literal["F_RDLCK"]
    lease_signal: Literal["SIGIO"]
    lease_break_time_seconds: int
    packet_sha256: str


@dataclass(frozen=True)
class ProtectedImagePublication:
    schema_version: Literal["q30t-protected-image-publication-v1"]
    job_id: str
    node_name: str
    operation_id: str
    producer_uid: int
    protected_path: str
    device: int
    inode: int
    nlink: Literal[1]
    mode: Literal[256]
    size: int
    sha256: str
    same_ofd_lease_validated: Literal[True]
    gateway_signature: str


@dataclass(frozen=True)
class KeeperLeaseBreakPacket:
    schema_version: Literal["ptv23-keeper-lease-break-packet-v1"]
    job_id: str
    node_name: str
    operation_id: str
    keeper_pid: int
    keeper_start_ticks: int
    staged_device: int
    staged_inode: int
    signal_name: Literal["SIGIO"]
    siginfo_code: int
    signal_monotonic_ns: int
    kernel_lease_break_proven: Literal[True]
    packet_sha256: str


def receive_keeper_same_ofd(
    control: socket.socket,
    *, expected_keeper_pidfd: int,
    expected_keeper_start_ticks: int,
) -> tuple[KeeperSameOFDReady, int]: ...


def publish_held_image(
    leased_image_fd: int,
    ready: KeeperSameOFDReady,
    *, gateway: RuntimeGatewayClient,
) -> ProtectedImagePublication: ...
```

The agent uses one `recvmsg` and requires exactly one kernel
`SCM_CREDENTIALS`, exactly one `SCM_RIGHTS`, and no unknown ancillary record.
It requires credential PID/UID/GID to equal the READY fields and expected job
UID/GID, then joins that PID to the pre-opened pidfd and start ticks. It checks
device/inode, size/hash, `O_RDONLY`, and `F_GETLEASE` on the received fd. It passes that fd to the
gateway, verifies the signed same-inode publication, and closes its validation
duplicate. The gateway must close its SCM duplicate before returning the signed
publication; F verifies that neither gateway nor agent retains an OFD
duplicate. The keeper remains the sole lease holder whose loss is fatal.

The keeper's signal/lease order is fixed. Before it creates any thread, its
main thread calls `pthread_sigmask(SIG_BLOCK, {SIGIO})`; every later thread
inherits the blocked mask and no code unblocks it. On the final read-only image
OFD it then sets `F_SETOWN_EX(F_OWNER_PID, keeper_pid)` and
`F_SETSIG(SIGIO)`, starts one dedicated waiter, and waits for the waiter's
ARMED synchronization after it has verified SIGIO is blocked. Only then may it
call `F_SETLEASE(F_RDLCK)` and `F_GETLEASE`. The waiter uses only
`sigwaitinfo({SIGIO})`; there is no asynchronous handler. Every
`/proc/self/task/{tid}/status` must show SIGIO blocked immediately before
`F_SETLEASE`. A write injected after every boundary either occurs before a
lease exists or yields a pending signal consumed by the armed waiter; there is
no acquired-lease/unowned-signal or acquired-lease/unblocked-thread window.

The same private seqpacket carries lease-break packets, each with fresh kernel
`SCM_CREDENTIALS` joined to the same pidfd/start identity. Neither endpoint is
inherited by Pyxis, wrapper, attacker, or child. A socket cannot be reopened as
a readable fd through procfd; sibling `pidfd_getfd` is denied by F's Yama gate.
Peer credentials, nonce, PID/start, and canonical hash are checked. EOF,
timeout, truncation, extra/malformed packets, or killed endpoint aborts. Tests
must attempt to open, duplicate, drain, and withhold events; none may permit GO
or a receipt.

A same-UID forged `SIGIO` is denial of service, never negative qualification
evidence. The keeper accepts a break packet only for kernel-origin siginfo and
a lease transition on its held OFD; any user-origin signal aborts without
setting `kernel_lease_break_proven`.

## Global Dispatch, Barrier, and Remote Death

One staging `srun` starts exactly one local agent per allocated node. Agents
create local keeper/publication sessions and register them with the installed
gateway. Once all signed publications name the identical protected path and
SHA, the trusted controller starts one global execution `srun` over all nodes.
No global argv contains a keeper PID, fd, or procfd. No nested execution step
is permitted.

Each global-step wrapper registers with its local gateway and emits READY. It
then blocks in the gateway's distributed barrier. The gateway authenticates
the controller, agents, and wrappers by persistent peer connection,
`SO_PEERCRED`, pidfd/start, exact job/step cgroup, node, and executable blob. It
releases a positive operation only after every expected node is READY and the
registered controller submits one global GO transaction. EOF, timeout,
malformed input, or ABORT exits without invoking application code. EOF on any
authenticated controller, agent, wrapper, or inter-gateway control connection
before remote exit irreversibly changes the transaction to ABORT-only, signs
and broadcasts ABORT, cancels the exact global step, and forbids every later GO
request. If the gateway process/socket itself disappears, the local agents and
controller apply the same cancel/no-GO rule; no receipt can depend on a
replacement service because its endpoint identity differs from reviewed F.

An attack session is registered as permanently ABORT-only. After every wrapper
is held READY, the controller commands each local agent to start its same-UID
blocking write. The gateway can distribute only ABORT for that session; no
caller, including the registered controller, can convert it to GO. The break
event triggers global ABORT and exact-step cancellation.

Qualification order is canonical and uses fresh operation IDs, keepers,
protected image paths, control trees, and global steps: positive observation,
then keeper-loss, then lease-break. In keeper-loss, every wrapper first reaches
authenticated READY and remains held. The selected node's keeper is killed;
the local gateway must independently observe its registered pidfd exit and the
agent must observe credentialed keeper-channel EOF. Either observation makes
the distributed session irreversibly ABORT-only, broadcasts ABORT, and cancels
the exact global step. GO is forbidden before and after loss. A loss after GO
in any non-qualification consumer also cancels and suppresses success, but
cannot produce canonical keeper-loss qualification evidence.

```python
@dataclass(frozen=True)
class SlurmTaskReady:
    schema_version: Literal["q30t-slurm-task-ready-v2"]
    job_id: str
    step_id: str
    node_name: str
    operation_id: str
    task_pid: int
    task_start_ticks: int
    cgroup_relative_path: str
    cgroup_device: int
    cgroup_inode: int
    image_device: int
    image_inode: int
    root_mount_readonly: Literal[True]
    frame_sha256: str


@dataclass(frozen=True)
class OperationBarrierDecision:
    schema_version: Literal["q30t-operation-barrier-decision-v1"]
    operation_id: str
    expected_nodes: tuple[str, ...]
    ordered_ready_frame_sha256s: tuple[str, ...]
    decision: Literal["GO", "ABORT"]
    attack_session: bool
    gateway_transaction_sha256: str
    gateway_signature: str


@dataclass(frozen=True)
class RemoteStepExitProof:
    schema_version: Literal["q30t-remote-step-exit-proof-v1"]
    job_id: str
    step_id: str
    ordered_nodes: tuple[str, ...]
    ordered_wrapper_pid_start: tuple[tuple[str, int, int], ...]
    authenticated_exit_nodes: tuple[str, ...]
    ordered_cgroup_identities: tuple[tuple[str, int, int], ...]
    every_cgroup_populated_zero: Literal[True]
    local_srun_returncode: int
    exact_step_terminal_state: Literal[
        "COMPLETED", "CANCELLED", "FAILED", "TIMEOUT", "NODE_FAIL"
    ]
    allocation_state: Literal[
        "RUNNING", "COMPLETING", "COMPLETED", "CANCELLED", "FAILED", "NODE_FAIL"
    ]
    cancel_elapsed_ms: int
    proof_completed_monotonic_ns: int
    proof_sha256: str
```

On break/control failure, the controller issues
`scancel --signal=KILL {job_id}.{step_id}` for the exact global step. Every
agent keeps its keeper alive and watches the authenticated step cgroup until
`cgroup.events` is `populated 0`; the controller reaps global `srun` and gets
exact scheduler step terminal and allocation states. An authenticated wrapper
EXIT may contribute, but per-node empty cgroups, global reap, and scheduler/
allocation confirmation are mandatory. All occur in less than five seconds
and before `lease_break_time - 5`; otherwise the allocation is cancelled and
no evidence/receipt is published. Local process-group death or accepted
`scancel` alone is never proof.

## Exact Pyxis Boundary

The single global execution argv, substituting only validated values, is:

```text
/usr/bin/srun --overlap --nodes={node_count} --ntasks={node_count}
  --ntasks-per-node=1 --gpus-per-node=4
  --container-image={identical_protected_absolute_path}
  --container-readonly --no-container-mount-home
  --no-container-remap-root --no-container-entrypoint
  --container-mounts={canonical_ro_rprivate_mounts}
  --container-workdir=/run/q30t/work
  /opt/q30t-runtime/bin/python -I -S
  /run/q30t/control/q30t_runtime_bootstrap.py
  --manifest /run/q30t/control/held-modules.json
  --manifest-sha256={validated_control_manifest_sha256}
  --entry-module common.specdec.ptv23_runtime_attestation
  --operation {operation}
```

There is no container name/save, writable flag, home mount, entrypoint,
remapped root, caller mount text, or procfd image. Every mount is a constructed
`source:destination:ro+rprivate`. The first mount is the identical per-node
protected control root to exactly `/run/q30t/control`; remaining destinations
are exact children of the bounded node-log directory. Individual anonymous
control fds or per-node procfd strings never appear in global argv.

The environment contains the exact constant entries `HOME`, `XDG_CONFIG_HOME`,
`PATH=/opt/q30t-runtime/bin:/usr/bin:/bin`, `PYTHONNOUSERSITE=1`,
`PYTHONDONTWRITEBYTECODE=1`, `LC_ALL=C`, `LANG=C`, and `TZ=UTC`; HOME and XDG
config point at the gateway-owned empty config root. The only
operation-specific entries are `XDG_DATA_HOME`, `XDG_CACHE_HOME`,
`XDG_RUNTIME_DIR`, and `Q30T_OPERATION_ID`. The only inherited entries are
validated `SLURM_JOB_ID`, `SLURM_STEP_ID`, `SLURMD_NODENAME`, `SLURM_LOCALID`,
`SLURM_PROCID`, `SLURM_NTASKS`, and `CUDA_VISIBLE_DEVICES`. The XDG data/cache/
runtime roots are distinct node-local operation roots and cannot contain
`hooks.d`, `mounts.d`, or `environ.d`. No `ENROOT_*`, `PYTHONPATH`,
`PYTHONHOME`, `LD_PRELOAD`, or `LD_LIBRARY_PATH` caller value survives.

Pyxis strips system-path overrides, so the loader recursively hashes/approves
the real `/etc/enroot` closure and Pyxis SPANK plugin/config. Its canonical
recursive manifest records bytewise-ordered relative path, type, mode, uid,
gid, size, and regular-file SHA-256; symlinks and unlisted types fail. Every
system hook, mount, and environment file is explicit; an extra/change fails
before srun.

Success requires Pyxis evidence for direct SquashFUSE mode, absence of
container creation/persistent rootfs, exact protected device/inode per node,
read-only root, no home, and no unexpected mount/environment.

The gateway-signed actual-open record is:

```python
@dataclass(frozen=True)
class PyxisDirectMountAttestation:
    schema_version: Literal["q30t-pyxis-direct-mount-attestation-v1"]
    job_id: str
    step_id: str
    node_name: str
    operation_id: str
    protected_path: str
    image_device: int
    image_inode: int
    squashfuse_pid: int
    squashfuse_start_ticks: int
    squashfuse_cgroup_relative_path: str
    backing_fd_number: int
    backing_fd_access_mode: Literal["O_RDONLY"]
    backing_fd_device: int
    backing_fd_inode: int
    root_mount_id: int
    root_mount_readonly: Literal[True]
    temporary_rootfs: Literal[True]
    container_creation_absent: Literal[True]
    persistent_rootfs_absent: Literal[True]
    gateway_signature: str
```

The loader requires backing fd device/inode to equal the protected publication
and requires a fresh SquashFUSE PID/start and root mount ID per operation.

## Attestation Interfaces

Named reuse is removed. Every operation gets fresh local keepers/publications
and one fresh unnamed global mount.

```python
@dataclass(frozen=True)
class RuntimeKeeperLossEvidence:
    schema_version: Literal["q30t-runtime-keeper-loss-v3"]
    job_id: str
    phase: Literal["one-node", "two-node"]
    operation_id: str
    ordered_nodes: tuple[str, ...]
    lost_node: str
    lost_keeper_pid: int
    lost_keeper_start_ticks: int
    every_task_ready_before_loss: Literal[True]
    every_task_ready_monotonic_ns: int
    keeper_loss_triggered_monotonic_ns: int
    gateway_keeper_pidfd_exit_observed: Literal[True]
    gateway_keeper_pidfd_exit_monotonic_ns: int
    agent_keeper_channel_eof_observed: Literal[True]
    agent_keeper_channel_eof_monotonic_ns: int
    gateway_eof_report_accepted: Literal[True]
    go_transaction_absent: Literal[True]
    abort_decision_file_sha256: str
    abort_decision_self_sha256: str
    abort_decision_monotonic_ns: int
    exact_step_cancelled: Literal[True]
    remote_exit_proof_file_sha256: str
    remote_exit_proof: RemoteStepExitProof
    keeper_loss_operation_receipt_absent: Literal[True]
    evidence_sha256: str


@dataclass(frozen=True)
class RuntimeLeaseBreakNodeJoin:
    node_name: str
    keeper_pid: int
    keeper_start_ticks: int
    image_device: int
    image_inode: int
    attack_pid: int
    attack_start_ticks: int
    packet_file_sha256: str
    packet_self_sha256: str
    blocking_open_started_monotonic_ns: int
    packet_received_monotonic_ns: int
    abort_decision_monotonic_ns: int
    remote_exit_proof_self_sha256: str
    writer_open_completed_monotonic_ns: int
    writer_open_completed_after_remote_exit: Literal[True]


@dataclass(frozen=True)
class RuntimeLeaseBreakEvidence:
    schema_version: Literal["q30t-runtime-lease-break-v2"]
    job_id: str
    phase: Literal["one-node", "two-node"]
    operation_id: str
    ordered_nodes: tuple[str, ...]
    protected_path: str
    ordered_break_packet_file_sha256s: tuple[str, ...]
    ordered_break_packet_self_sha256s: tuple[str, ...]
    ordered_node_joins: tuple[RuntimeLeaseBreakNodeJoin, ...]
    attack_open_flags: Literal["O_WRONLY"]
    every_task_ready_before_attack: Literal[True]
    go_transaction_absent: Literal[True]
    abort_transaction_seen: Literal[True]
    exact_step_cancelled: Literal[True]
    remote_exit_proof_file_sha256: str
    remote_exit_proof: RemoteStepExitProof
    every_writer_open_completed_after_remote_exit: Literal[True]
    attacked_operation_receipt_absent: Literal[True]
    lease_break_cancellation_proven: Literal[True]
    evidence_sha256: str


@dataclass(frozen=True)
class RuntimeDirectMountEvidence:
    schema_version: Literal["q30t-runtime-direct-mount-v1"]
    job_id: str
    phase: Literal["one-node", "two-node"]
    operation_id: str
    ordered_publications: tuple[ProtectedImagePublication, ...]
    ordered_actual_open_attestations: tuple[PyxisDirectMountAttestation, ...]
    global_pyxis_argv_sha256: str
    pyxis_config_manifest_sha256: str
    ordered_root_mount_identities: tuple[tuple[str, int, int], ...]
    every_root_mount_readonly: Literal[True]
    squashfuse_direct_mode: Literal[True]
    container_creation_absent: Literal[True]
    persistent_rootfs_absent: Literal[True]
    home_mount_absent: Literal[True]
    user_hooks_absent: Literal[True]
    evidence_sha256: str


@dataclass(frozen=True)
class RuntimeIdentityBlock:
    cluster: Literal["ptyche"]
    profile_file_sha256: str
    feasibility_receipt_file_sha256: str
    feasibility_review_receipt_file_sha256: str
    derived_image_receipt_file_sha256: str
    base_image_sha256: str
    runtime_archive_sha256: str
    archive_tree_receipt_file_sha256: str
    derived_image_path: str
    derived_image_sha256: str
    derived_content_manifest_sha256: str
    runtime_tree_sha256: str
    runtime_install_root: Literal["/opt/q30t-runtime"]
    source_commit: str
    qualification_bound_blobs_sha256: str
    control_manifest_sha256: str
    pyxis_config_manifest_sha256: str


@dataclass(frozen=True)
class RuntimeNodeAttestationReceipt:
    schema_version: Literal["q30t-runtime-node-attestation-v3"]
    job_id: str
    phase: Literal["one-node", "two-node"]
    node_name: str
    identity: RuntimeIdentityBlock
    keeper_receipt_file_sha256: str
    ordered_control_publication_file_sha256s: tuple[str, str, str]
    observation_file_sha256: str
    direct_mount_evidence_file_sha256: str
    keeper_loss_evidence_file_sha256: str
    lease_break_evidence_file_sha256: str
    remote_exit_proof_file_sha256: str
    bounded_log_file_sha256s: tuple[str, ...]
    python_executable_relative: Literal["bin/python"]
    python_import_origins_relative: tuple[tuple[str, str], ...]
    visible_gpu_identities: tuple[str, str, str, str]
    direct_unnamed_mount_proven: Literal[True]
    keeper_loss_failure_proven: Literal[True]
    lease_break_cancellation_proven: Literal[True]
    receipt_sha256: str


@dataclass(frozen=True)
class RuntimeQualificationReceipt:
    schema_version: Literal["q30t-runtime-qualification-v3"]
    job_id: str
    phase: Literal["one-node", "two-node"]
    identity: RuntimeIdentityBlock
    expected_node_count: Literal[1, 2]
    ordered_nodes: tuple[str, ...]
    ordered_operation_kinds: tuple[
        Literal["positive"], Literal["keeper-loss"], Literal["lease-break"]
    ]
    ordered_operation_ids: tuple[str, str, str]
    ordered_node_receipt_file_sha256s: tuple[str, ...]
    ordered_control_publication_file_sha256s: tuple[str, ...]
    ordered_node_receipts_sha256: str
    prerequisite_one_node_receipt_path: str | None
    prerequisite_one_node_receipt_file_sha256: str | None
    every_node_local_handoff_proven: Literal[True]
    one_global_execution_step_proven: Literal[True]
    every_node_direct_unnamed_mount_proven: Literal[True]
    every_node_keeper_loss_failure_proven: Literal[True]
    every_node_lease_break_cancellation_proven: Literal[True]
    receipt_sha256: str
```

```python
def load_runtime_node_receipt(
    path: Path, expected_file_sha256: str
) -> RuntimeNodeAttestationReceipt: ...


def load_runtime_qualification_receipt(
    path: Path, expected_file_sha256: str
) -> RuntimeQualificationReceipt: ...


def load_runtime_lease_break_evidence(
    path: Path, expected_file_sha256: str
) -> RuntimeLeaseBreakEvidence: ...


def load_runtime_keeper_loss_evidence(
    path: Path, expected_file_sha256: str
) -> RuntimeKeeperLossEvidence: ...


def load_runtime_direct_mount_evidence(
    path: Path, expected_file_sha256: str
) -> RuntimeDirectMountEvidence: ...
```

Loaders stable-read canonical single-link files <=1 MiB, reject duplicate,
missing, extra, or reordered keys, replay self-hashes and physical references,
and require one/two unique nodes. Two-node requires the exact reviewed one-node
receipt and rejects shared PID/fd/inode/cgroup evidence or nested execution.
Control-publication references are exactly three per node in positive,
keeper-loss, lease-break order. The aggregate flattens them operation-major,
then bytewise node order; it reloads every fresh tree and rejects reuse of any
root/file inode across operations.
The keeper-loss loader requires bytewise-sorted unique nodes, `lost_node`
membership, exact READY-before-loss ordering, gateway pidfd plus credentialed
channel EOF, no GO, the same signed ABORT decision and remote-exit proof
replayed by the node/aggregate receipts, and absence of an operation receipt.
It requires both observations, not either one, with
`every_ready <= loss_triggered <= pidfd_exit`,
`loss_triggered <= agent_eof <= abort_decision <
remote_exit_proof.proof_completed_monotonic_ns`; the gateway must have accepted
the credentialed EOF report before ABORT. The aggregate requires the literal
operation-kind tuple `("positive", "keeper-loss", "lease-break")`, three
distinct operation IDs, a completed positive receipt before loss, and the
keeper-loss remote-exit proof before the lease-break operation starts.
The lease-break loader requires exactly one canonical persisted
`KeeperLeaseBreakPacket` per ordered node. It stable-loads every packet by the
parallel physical-hash tuple, requires its self hash to equal the parallel
self-hash tuple and the corresponding node join, and joins packet
job/node/operation/keeper/device/inode to the signed publication and attack
PID/start. It requires
`blocking_open_started <= packet_received <= abort_decision <
remote_exit_proof.proof_completed_monotonic_ns < writer_open_completed` for
every node,
requires every join to bind the one global remote-exit self hash, and rejects a
missing, duplicate, reordered, cross-node, or unpersisted packet.

## Independent Review and Approval

P contains exact OpenSSH allowed signers at
`tools/launcher/common/specdec/q30t_runtime_review_allowed_signers`. Its Git
blob SHA, physical SHA, namespace `q30t-runtime-review-v1`, principal
`q30t-runtime-independent-reviewer`, and key fingerprint are profile constants.
The blob must be byte-identical to F's reviewed trust root. P cannot freeze with
an empty key or the producer's signing key. Exact verify argv is:

```text
/usr/bin/ssh-keygen -Y verify -f {allowed_signers_path}
  -I q30t-runtime-independent-reviewer
  -n q30t-runtime-review-v1 -s {signature_path}
```

Canonical receipt bytes are stdin. The receipt omits its detached-signature
hash, avoiding a circular digest.

```python
@dataclass(frozen=True)
class SignedRuntimeReview:
    receipt_path: Path
    receipt_file_sha256: str
    signature_path: Path
    signature_file_sha256: str
    allowed_signers_path: Path
    allowed_signers_file_sha256: str


@dataclass(frozen=True)
class RuntimePlatformFeasibilityReviewBinding:
    schema_version: Literal["q30t-platform-feasibility-review-binding-v1"]
    producer_aggregate_receipt_file_sha256: str
    ordered_producer_run_receipt_file_sha256s: tuple[str, str, str]
    producer_job_ids: tuple[str, str, str]
    reviewer_aggregate_receipt_file_sha256: str
    ordered_reviewer_run_receipt_file_sha256s: tuple[str, str, str]
    reviewer_job_ids: tuple[str, str, str]
    ordered_sixteen_nodes: tuple[str, ...]
    ordered_gateway_endpoint_identity_sha256s: tuple[str, ...]
    producer_global_srun_argv_sha256: str
    reviewer_global_srun_argv_sha256: str
    producer_collective_artifact_file_sha256: str
    reviewer_collective_artifact_file_sha256: str
    binding_sha256: str


@dataclass(frozen=True)
class RuntimeIndependentReviewReceipt:
    schema_version: Literal["q30t-runtime-independent-review-v2"]
    review_kind: Literal[
        "platform-feasibility", "derived-image", "one-node", "two-node"
    ]
    producer_commit: str
    reviewer_commit: str
    reviewer_principal: Literal["q30t-runtime-independent-reviewer"]
    reviewer_key_fingerprint: str
    profile_path: str | None
    profile_file_sha256: str | None
    subject_bound_blobs_sha256: str
    subject_receipt_path: str
    subject_receipt_file_sha256: str
    subject_receipt_schema: str
    subject_self_sha256: str
    prerequisite_review_receipt_file_sha256: str | None
    prerequisite_review_signature_file_sha256: str | None
    review_rebuild_receipt_path: str | None
    review_rebuild_receipt_file_sha256: str | None
    review_build_a_sha256: str | None
    review_build_b_sha256: str | None
    reviewed_derived_image_sha256: str | None
    platform_feasibility_binding: RuntimePlatformFeasibilityReviewBinding | None
    ordered_node_receipt_file_sha256s: tuple[str, ...]
    ordered_control_publication_file_sha256s: tuple[str, ...]
    ordered_direct_mount_evidence_file_sha256s: tuple[str, ...]
    ordered_keeper_loss_evidence_file_sha256s: tuple[str, ...]
    ordered_lease_break_evidence_file_sha256s: tuple[str, ...]
    ordered_remote_exit_proof_file_sha256s: tuple[str, ...]
    producer_job_ids: tuple[str, ...]
    reviewer_job_ids: tuple[str, ...]
    bounded_log_file_sha256s: tuple[str, ...]
    verdict: Literal["clean"]
    receipt_sha256: str


def load_signed_runtime_review(
    review: SignedRuntimeReview,
    *, expected_kind: Literal[
        "platform-feasibility", "derived-image", "one-node", "two-node"
    ],
) -> RuntimeIndependentReviewReceipt: ...


def publish_signed_runtime_review(
    receipt: RuntimeIndependentReviewReceipt,
    *,
    detached_signature: bytes,
    receipt_output_path: Path,
    signature_output_path: Path,
    allowed_signers_path: Path,
    allowed_signers_file_sha256: str,
) -> SignedRuntimeReview: ...
```

The loader verifies trust root/signature and exact subject schema, self hash,
and physical hash. Feasibility review independently repeats the ordered live
1 -> 2 -> 16 jobs, binds three pairwise-distinct producer job IDs and three
pairwise-distinct reviewer job IDs disjoint from the producer IDs, replays
every prerequisite receipt/hash, and requires the same ordered sixteen-node
endpoint tuple. Its
`platform_feasibility_binding` is mandatory and self-hashed; it binds both
aggregate/run receipt sets, both exact global argv hashes, endpoint identity
hashes, and both collective artifacts. That field is null for every other
review kind. Image review invokes two independent protected rebuilds and binds
their receipt,
complete manifests, and hashes. One-/two-node review reruns exact operations
and binds every node/direct-mount/keeper-loss/lease-break/exit/log hash. Exact
cardinalities and the preceding signed review package are mandatory.
Profile path/hash are null only for platform-feasibility review and non-null for
all later kinds.
The publisher first verifies the detached signature over the exact canonical
receipt bytes, then publishes receipt and signature by the anonymous
no-clobber publisher, reloads both physical hashes, and returns the signed
reference. It never writes a named temporary or handles a reviewer private key.

P has two empty roots:

```python
APPROVED_Q30T_DERIVED_IMAGE_REVIEWS: frozenset[
    tuple[str, str, str]
] = frozenset()
APPROVED_Q30T_RUNTIME_QUALIFICATION_REVIEWS: frozenset[
    tuple[str, str, str]
] = frozenset()
```

Each tuple is `(subject_receipt_file_sha256, review_receipt_file_sha256,
review_signature_file_sha256)`. Descendant A changes only these literals.
Admission proves P is A's ancestor, reconstructs the literal blob map at both
commits, requires mapped blob byte equality, and loads the signed review chain.
The implementation commit at the end of the producer task is P itself and
must remain the pushed, reviewed `HEAD` with no P-owned staged/unstaged delta
while evidence is produced. Unrelated dirty worktree paths are preserved and
excluded from P rather than cleaned or staged. The hostile-review task does
not create a second equivalent producer commit. A is
created from clean P by editing and staging only the approval module; before
the A commit, the index contains exactly that path and its diff contains only
the two frozenset literals. No mapped runtime blob is staged or modified.

```text
feasibility-only F -> independently signed feasibility review
  [live one-node -> live two-node -> live sixteen-node]
  -> reviewed signed P with empty roots
  -> protected build A/B -> independent protected rebuild A/B
  -> one-node qualification -> independent one-node qualification
  -> two-node qualification -> independent two-node qualification
  -> constants-only signed+DCO descendant A
  -> row observation or later training admission
```

## Consumer Interface

```python
@dataclass(frozen=True)
class Q30TRuntimeAdmissionRequest:
    schema_version: Literal["q30t-runtime-admission-request-v1"]
    consumer_kind: Literal["row-observation", "q30-canary", "q30-full"]
    node_count: Literal[1, 2, 16]
    segment_count: Literal[16] | None
    requested_nodes: tuple[str, ...]
    request_sha256: str


@dataclass(frozen=True)
class AuthenticatedQ30TRuntime:
    platform_feasibility: RuntimePlatformFeasibilityReceipt
    platform_feasibility_review: RuntimeIndependentReviewReceipt
    derived_image: RuntimeDerivedImageReceipt
    qualification: RuntimeQualificationReceipt
    derived_image_review: RuntimeIndependentReviewReceipt
    qualification_review: RuntimeIndependentReviewReceipt
    producer_commit: str
    derived_image_path: Path
    derived_image_sha256: str
    admitted_nodes: tuple[str, ...]
    qualification_bound_blob_sha256s: tuple[tuple[str, str], ...]


def authenticate_q30t_runtime(
    *,
    platform_feasibility_receipt_path: Path,
    platform_feasibility_receipt_file_sha256: str,
    platform_feasibility_review: SignedRuntimeReview,
    derived_image_receipt_path: Path,
    derived_image_receipt_file_sha256: str,
    derived_image_review: SignedRuntimeReview,
    qualification_receipt_path: Path,
    qualification_receipt_file_sha256: str,
    qualification_review: SignedRuntimeReview,
    request: Q30TRuntimeAdmissionRequest,
    consumer_commit: str,
) -> AuthenticatedQ30TRuntime: ...
```

The admission loader requires the reviewed feasibility schema v3 and its exact
signed 1 -> 2 -> 16 review in the initial F -> P -> A ancestry. Requested nodes
must equal the loaded one-node, two-node, or sixteen-node run tuple exactly;
segment-16 Q30 canary/full requires the sixteen-node tuple and constructs one
global `srun` with 16 tasks and 16 ranks. A subset, permutation, unreviewed
endpoint, missing 16-node receipt/review, or v2 feasibility receipt fails
before staging. Canary/full additionally requires `node_count=16` and
`segment_count=16`; row observation requires `segment_count=None` and the exact
reviewed one- or two-node tuple. Row and continuation use this same loader and
image. Each operation creates fresh node-local keepers/publications before one
global execution step.
Consumers receive no archive extraction path or mutable source path.

## Bounded Logs and Test-Only-First

External modes are exactly `--platform-feasibility`, `--derived-image`,
`--one-node`, and `--two-node`. Feasibility has its own F-only runner/submitter.
Its public CLI is exactly:

```text
submit_q30t_runtime_platform_feasibility.sh --platform-feasibility
  --node-count 1
submit_q30t_runtime_platform_feasibility.sh --platform-feasibility
  --node-count 2 --one-node-receipt {absolute_path}
  --one-node-receipt-sha256 {64-lower-hex}
submit_q30t_runtime_platform_feasibility.sh --platform-feasibility
  --node-count 16 --two-node-receipt {absolute_path}
  --two-node-receipt-sha256 {64-lower-hex}
```

The one-node invocation has no prerequisite arguments. Its test-only and real
submission complete before the two-node test-only invocation may load the
canonical one-node receipt. The two-node test-only and real submission use the
same one-node path/hash and complete before the sixteen-node test-only
invocation may load the canonical two-node receipt. The sixteen-node test-only
and real submission use that same two-node path/hash. All three real job IDs
must be pairwise distinct. The aggregate v3 receipt is created only after all
three run receipts reload and join. Each invocation runs its exact
`sbatch --test-only` argv first and,
only after success, one `sbatch --parsable`; test-only output is never treated
as a job ID. Every submitter requires a clean pushed signed commit, approved
remote, sterile absolute tools, `--export=NONE`, and one comment. This design
task performs no submission.

No Slurm output points at Lustre. Batch output is node-local
`/raid/scratch/$USER/q30t-runtime-logs/q30t-runtime-%j.batch.log`; staging/global srun
stdout contains canonical frames only and is capped at 1 MiB per step. Each
wrapper captures child stdout/stderr in `O_CREAT|O_EXCL|O_NOFOLLOW` mode-0600
node-local files, kills/fails at either 8 MiB cap, fsyncs, and hashes them.
Pyxis/enroot diagnostics have the same cap. Only closed, stable, size-checked
artifacts publish anonymously to Lustre. Overflow, truncation, sparse content,
extra writer, or publish-before-close suppresses receipts. No unbounded
`%j.out` exists on shared storage.

## Failure Semantics

- Missing real gateway/build service or failed Phase F proof makes the design
  BLOCKED; P and A do not exist.
- Mutable build root/candidate, host tool, incomplete content manifest,
  changed closure, or A/B mismatch publishes no image.
- Procfd image, named/extracted Pyxis root, nested execution, writable root,
  home, unapproved hook/config, or persistent rootfs fails before GO.
- Keeper/control EOF, PID reuse, same-OFD failure, drained/withheld event,
  malformed barrier frame, early GO, lease break, or gateway failure cancels
  the exact global step and suppresses success.
- Accepted `scancel` or dead local process group is not remote-death proof.
  Missing per-node cgroup-empty, global reap, scheduler terminal, or allocation
  state before deadline cancels the allocation and yields no evidence.
- Producer/runtime code never unlinks, renames, overwrites, or recursively
  deletes a build root, protected publication, receipt, or log. Site lifecycle
  reclamation is outside receipt-producing code.
- Runtime approval never authorizes data, parent checkpoint, canary, or
  training without separate gates.

## Rejected Alternatives

- **Unlinked procfd image:** enroot canonicalization may resolve a deleted path,
  and one procfd cannot address different nodes.
- **Named container reuse:** Pyxis creates a same-UID-writable extracted rootfs.
- **Same-UID `0700` build/path:** the owner can chmod, replace, or mutate it.
- **Procfd reopen for `F_GETLEASE`:** it creates a new open file description;
  only the `SCM_RIGHTS` duplicate validates the keeper OFD.
- **Notification pipe/local reap:** pipes can be drained, and launcher death
  does not prove remote tasks are gone.
- **Per-node nested execution:** it changes global distributed rank/step
  semantics; agents stage/control, while one global step executes.

## Completion Criteria

- Phase F live one-/two-/sixteen-node feasibility and signed independent review pass before
  P, or the work records BLOCKED and stops.
- Two protected builds and two review rebuilds produce identical bytes and
  complete manifests.
- Linux/Ptyche prove same-OFD validation, protected publication, unnamed
  read-only SquashFUSE, one global multi-node step, distributed GO/ABORT,
  protected event channels, same-UID write break, and remote death before the
  lease deadline.
- One-/two-node qualification v3 receipts plus the feasibility v3 sixteen-node
  receipt/review replay and bind one image and the exact segment-16 node tuple.
- Scans find no consumer extraction, checkout/PYTHONPATH import, named
  container, procfd image, nested execution, writable root, user hook,
  pathname deletion, recursive cleanup, or unbounded shared log.
- P has empty roots. A is signed+DCO and constants-only; its tuples bind exact
  subject/review/signature hashes and every mapped blob is unchanged.
- Row and later training use the same loader/image with independent non-runtime
  gates still required.
