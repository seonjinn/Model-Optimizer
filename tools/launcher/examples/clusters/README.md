# SpecDec Cluster Entrypoints

The cluster wrappers select scheduler and storage defaults from one profile and delegate all work to shared scripts under `common/specdec`. Keep source and external profiles in `/home`, durable artifacts in `/lustre`, and temporary transfer state in the profile's scratch candidates.

Use an external profile when a run pins a newer exact ModelOpt commit:

```bash
tools/launcher/examples/clusters/lyris/specdec.sh \
  --cluster-profile /home/user/lyris-q235.yaml \
  full-chain \
  --manifest /home/user/q235-manifest.json \
  --receipt /lustre/project/receipts/q235.jsonl \
  --readiness-receipt /lustre/project/readiness/lyris.json
```

The portable action support matrix is deliberately narrow:

| Action | OCI-HSG | Lyris | Ptyche |
| --- | --- | --- | --- |
| `probe`, `training-wave`, `full-chain`, `stage-runtime` | supported | supported | supported |
| `upload`, `download` | `cpu_datamover`, one CPU node | blocked: no approved partition | blocked: no approved partition |
| evaluator pair submission | not exposed | not exposed | not exposed |

The `training-wave`, `full-chain`, and `stage-runtime` actions require a readiness receipt. Add `--print-command` before the action to inspect the delegated command without submitting anything. The current evaluator pair sbatch contains OCI-specific static GPU directives. In particular, the Lyris profile has `explicit_gpu_flag: false`, so selecting the Lyris profile alone does not make that evaluator job portable. Use a profile-aware evaluator submitter after one is implemented; this wrapper fails closed for evaluator actions today.

Bundle transfers use an artifact ID and an rclone remote supplied at runtime; credentials are never stored in the repository:

```bash
tools/launcher/examples/clusters/oci-hsg/specdec.sh upload \
  --artifact-id q30-opb-dflash-b8-s4166 \
  --artifact-source-commit 92487e90620830922fa21dedc583dda4a519adb0 \
  --source /lustre/project/modelopt-specdec/bundles/q30-opb-dflash-b8-s4166 \
  --remote-root pbss-team:project/modelopt-specdec
```

The wrapper first runs `sbatch --test-only`, then submits one CPU-only OCI `cpu_datamover` node with durable logs under `<durable-root>/transfers/logs`. Use `--dry-run` after the action to stop after scheduler validation. Lyris and Ptyche fail closed until their transfer partitions are reviewed and added to `common/specdec/transfers/profiles`; they never fall back to a GPU training partition.

The submitter pins the clean transfer-tool checkout separately from `--artifact-source-commit`. The latter is the producer/source revision recorded in the immutable manifest, so moving an older artifact never relabels it as the current launcher revision. Uploads reject symlinks, write a stable per-file SHA-256/size manifest, verify the payload, and publish `completion.json` only after the manifest is durable remotely. The resulting remote path is `<remote-root>/<artifact-id>/<manifest-sha256>`, so the same artifact ID can safely identify multiple immutable contents.

Pass the printed digest to a download submission:

```bash
tools/launcher/examples/clusters/oci-hsg/specdec.sh download \
  --artifact-id q30-opb-dflash-b8-s4166 \
  --manifest-sha256 MANIFEST_SHA256_FROM_UPLOAD \
  --destination /lustre/project/modelopt-specdec/bundles/q30-opb-dflash-b8-s4166 \
  --remote-root pbss-team:project/modelopt-specdec
```

Downloads verify the completion marker, the bound manifest digest, and every downloaded file before an atomic no-clobber rename. An exact completed destination is reused; conflicting contents fail closed. Keep code and profiles in `/home`, durable inputs and results in `/lustre`, and transfer scratch/cache on node-local `/raid/scratch` where the cluster provides it.
