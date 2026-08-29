#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=96
#SBATCH --mem=0

# Import a pinned vLLM release into a squashfs on Lustre and record its
# SHA-256. The training runner takes the image by path and re-checks that
# digest, so this job is what gives the pin an identity: the tag names what we
# asked for, the digest names what we got.
#
# Runs on a CPU node. The import is a download and a squash - no container
# execution - so it needs no GPU, and the draft-architecture check reads the
# squashfs from the host rather than running Python inside an image that would
# want CUDA. The node must still be aarch64: enroot resolves the multi-arch
# manifest against the host, and an x86 pull here would stage an image the
# Grace compute nodes cannot run.

set -euo pipefail

required=(IMAGE_REF IMAGE_TAG DEST_ROOT)
for name in "${required[@]}"; do
    [[ -n "${!name:-}" ]] || { echo "missing required environment: $name" >&2; exit 2; }
done
[[ "${SLURM_JOB_ID:-}" =~ ^[0-9]+$ ]] || { echo "invalid Slurm job identity" >&2; exit 2; }
[[ "$DEST_ROOT" == /lustre/* ]] || { echo "container images belong on Lustre" >&2; exit 2; }
[[ "$IMAGE_TAG" =~ ^[0-9a-zA-Z._-]+$ ]] || { echo "malformed image tag" >&2; exit 2; }
[[ "$(uname -m)" == aarch64 ]] || { echo "image staging must run on an aarch64 node" >&2; exit 2; }
command -v enroot >/dev/null || { echo "enroot is absent" >&2; exit 2; }
command -v unsquashfs >/dev/null || { echo "unsquashfs is absent" >&2; exit 2; }

# Node-local scratch for the enroot cache and the in-progress squashfs.
# Streaming a multi-GB image onto Lustre in small writes is the pattern the
# shared-filesystem guidance warns against; publish once, at the end.
scratch_parent=""
for candidate in "${SLURM_TMPDIR:-}" /raid/scratch /raid /tmp; do
    [[ -n "$candidate" && -d "$candidate" && -w "$candidate" ]] || continue
    [[ "$candidate" != /lustre* && "$candidate" != /home* ]] || continue
    scratch_parent="$candidate"
    break
done
[[ -n "$scratch_parent" ]] || { echo "no node-local scratch available" >&2; exit 2; }

available_kib="$(df -Pk "$scratch_parent" | awk 'NR==2 {print $4}')"
(( available_kib >= 120 * 1024 * 1024 )) || {
    echo "node-local scratch below 120G: $scratch_parent has ${available_kib}K" >&2
    exit 2
}

scratch="$scratch_parent/runtime-image-$SLURM_JOB_ID"
mkdir -p "$scratch"
trap 'rm -rf "$scratch"' EXIT
export ENROOT_CACHE_PATH="$scratch/cache"
export ENROOT_DATA_PATH="$scratch/data"
export ENROOT_TEMP_PATH="$scratch/tmp"
mkdir -p "$ENROOT_CACHE_PATH" "$ENROOT_DATA_PATH" "$ENROOT_TEMP_PATH"

# Resolve what the tag points at *before* importing. Each cluster squashes its
# own image, so the resulting file digests differ by construction and cannot be
# compared across sites. The upstream manifest digest is the thing that can:
# it makes "every cluster imported the same vLLM" a checkable claim rather than
# an assumption, and the arms are only comparable if it holds.
# These nodes carry a node-level pull-through registry cache in the proxy
# variables, and it is not uniform across the partition: cpu-dm-0004 imported
# cleanly while cpu-dm-0007 failed six curl retries on an unresolvable
# container-cache host. The asymmetry hides well -- curl honours all_proxy for
# every scheme, Python urllib does not -- so a digest lookup can reach the
# registry on exactly the node where the import cannot. The proxy is a cache,
# not an authorisation gateway, so an unusable one is a reason to go direct
# rather than to fail. Prefer it whenever it actually answers.
#
# This runs before the digest lookup, not just before the import: urllib does
# honour https_proxy and http_proxy, so a broken one of those would otherwise
# fail the lookup before the import is ever reached.
command -v getent >/dev/null || { echo "getent is absent" >&2; exit 2; }
for proxy_var in all_proxy ALL_PROXY https_proxy HTTPS_PROXY http_proxy HTTP_PROXY; do
    proxy_value="${!proxy_var:-}"
    [[ -n "$proxy_value" ]] || continue
    proxy_authority="${proxy_value#*://}"
    proxy_authority="${proxy_authority%%/*}"
    proxy_authority="${proxy_authority##*@}"
    if [[ "$proxy_authority" == \[*\]* ]]; then
        # Bracketed IPv6 literal: the brackets delimit the host, so the port
        # separator is the colon after the closing bracket, not the ones inside.
        proxy_host="${proxy_authority%%\]*}"
        proxy_host="${proxy_host#\[}"
        proxy_port="${proxy_authority##*\]}"
        proxy_port="${proxy_port#:}"
    else
        proxy_host="${proxy_authority%%:*}"
        proxy_port="${proxy_authority##*:}"
        [[ "$proxy_port" != "$proxy_authority" ]] || proxy_port=""
    fi
    [[ -n "$proxy_host" ]] || continue
    reason=""
    if ! getent hosts "$proxy_host" >/dev/null 2>&1; then
        reason="does not resolve"
    elif [[ -n "$proxy_port" ]] && ! timeout 5 bash -c \
            "exec 3<>/dev/tcp/${proxy_host}/${proxy_port}" 2>/dev/null; then
        # A drained cache node still resolves. Resolution alone would leave the
        # import retrying against a host that will never answer.
        reason="does not accept connections on port $proxy_port"
    fi
    if [[ -n "$reason" ]]; then
        echo "proxy $proxy_var host $proxy_host $reason; going direct"
        unset "$proxy_var"
    fi
done

upstream_digest="$(python3 - "$IMAGE_REF" <<'PY_DIGEST'
import json
import sys
import urllib.request

_ACCEPT = ",".join(
    (
        "application/vnd.oci.image.index.v1+json",
        "application/vnd.docker.distribution.manifest.list.v2+json",
    )
)


def _get(url, headers):
    request = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(request, timeout=120) as response:
        return json.load(response)


reference = sys.argv[1]
if reference.startswith("docker://"):
    reference = reference[len("docker://") :]
repository, _, tag = reference.partition(":")
if "/" not in repository:
    repository = "library/" + repository

token = _get(
    "https://auth.docker.io/token?service=registry.docker.io"
    f"&scope=repository:{repository}:pull",
    {},
)["token"]
index = _get(
    f"https://registry-1.docker.io/v2/{repository}/manifests/{tag}",
    {"Authorization": f"Bearer {token}", "Accept": _ACCEPT},
)

for manifest in index.get("manifests", []):
    platform = manifest.get("platform", {})
    if platform.get("architecture") == "arm64" and platform.get("os") == "linux":
        print(manifest["digest"])
        break
else:
    raise SystemExit("tag has no linux/arm64 manifest")
PY_DIGEST
)"
[[ "$upstream_digest" =~ ^sha256:[0-9a-f]{64}$ ]] || {
    echo "could not resolve the upstream arm64 digest" >&2
    exit 2
}
if [[ -n "${EXPECTED_IMAGE_DIGEST:-}" && "$EXPECTED_IMAGE_DIGEST" != "$upstream_digest" ]]; then
    echo "tag moved: expected $EXPECTED_IMAGE_DIGEST, resolved $upstream_digest" >&2
    exit 2
fi

staged="$scratch/image.sqsh"
enroot import -o "$staged" "$IMAGE_REF"
[[ -s "$staged" ]] || { echo "import produced no image" >&2; exit 2; }

# Read the draft-architecture registry straight out of the squashfs. vLLM has
# no dflash2 method string: DFlash2 is served as method "dflash" and selected
# by the exported architecture name, so an image missing that registry entry
# would serve the DFlash2 arm as plain DFlash and be measured as a duplicate
# rather than failing. Catch it here, not in the results table.
listing="$scratch/listing.txt"
unsquashfs -l "$staged" > "$listing"
grep -q "/lib/aarch64-linux-gnu" "$listing" || {
    echo "staged image is not aarch64" >&2
    exit 2
}
grep -q "/vllm/model_executor/models/qwen3_dflash2.py$" "$listing" || {
    echo "staged image lacks qwen3_dflash2.py" >&2
    exit 2
}
registry_entry="$(grep -m1 "/vllm/model_executor/models/registry.py$" "$listing" | sed 's|^squashfs-root||')"
[[ -n "$registry_entry" ]] || { echo "staged image lacks a model registry" >&2; exit 2; }
unsquashfs -d "$scratch/extract" -no-progress "$staged" "$registry_entry" >/dev/null
for architecture in DFlashDraftModel DFlash2DraftModel Qwen3DSparkModel; do
    grep -q "\"$architecture\"" "$scratch/extract/$registry_entry" || {
        echo "staged image registry lacks $architecture" >&2
        exit 2
    }
done

digest="$(sha256sum "$staged" | cut -d' ' -f1)"
stamp="$(date -u +%Y%m%d)"
name="vllm_openai_${IMAGE_TAG//./}_aarch64_${stamp}_${SLURM_JOB_ID}.sqsh"

dest_dir="$DEST_ROOT/image"
mkdir -p "$dest_dir"
dest="$dest_dir/$name"
[[ -e "$dest" ]] && { echo "destination image already exists: $dest" >&2; exit 2; }

cp "$staged" "$dest.part"
[[ "$(sha256sum "$dest.part" | cut -d' ' -f1)" == "$digest" ]] || {
    echo "image digest changed in transit to Lustre" >&2
    rm -f "$dest.part"
    exit 2
}
mv "$dest.part" "$dest"
chmod 0444 "$dest"

receipt="$dest_dir/${name%.sqsh}.json"
python3 - "$receipt" "$IMAGE_REF" "$IMAGE_TAG" "$dest" "$digest" "$upstream_digest" <<'PY_RECEIPT'
import json
import sys

receipt, image_ref, image_tag, image_path, image_sha256, upstream_digest = sys.argv[1:7]
with open(receipt, "w") as handle:
    json.dump(
        {
            "schema": "q30t-runtime-image-receipt-v1",
            "image_ref": image_ref,
            "image_tag": image_tag,
            "image_path": image_path,
            "image_sha256": image_sha256,
            "upstream_digest": upstream_digest,
            "draft_architectures": [
                "DFlashDraftModel",
                "DFlash2DraftModel",
                "Qwen3DSparkModel",
            ],
        },
        handle,
        indent=2,
        sort_keys=True,
    )
    handle.write("\n")
PY_RECEIPT

echo "image_path=$dest"
echo "image_sha256=$digest"
echo "upstream_digest=$upstream_digest"
