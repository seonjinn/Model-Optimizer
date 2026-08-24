#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Build a relocatable venv whose vLLM Python sources are exact PR 52816 bytes.
set -euo pipefail

[[ $# == 7 ]] || {
    echo "usage: $0 BASE_RUNTIME OUTPUT VLLM_CHECKOUT FLASHMLA_CHECKOUT IMAGE LAUNCHER_ROOT SCRATCH_ROOT" >&2
    exit 2
}
BASE_RUNTIME="$(realpath "$1")"
OUTPUT_ARCHIVE="$(realpath -m "$2")"
VLLM_CHECKOUT="$(realpath "$3")"
FLASHMLA_CHECKOUT="$(realpath "$4")"
IMAGE="$(realpath "$5")"
LAUNCHER_ROOT="$(realpath "$6")"
SCRATCH_ROOT="$(realpath -m "$7")"
EXPECTED_COMMIT=b389ac29465b33f9e9c534df221ea3c129e9793f
REQUIRED_COMMIT=b389ac29465b33f9e9c534df221ea3c129e9793f
FLASHMLA_COMMIT=a8f794d1251cbfd88a5011445dd5582289c727e4

[[ -f "$BASE_RUNTIME" && -f "$IMAGE" && -d "$VLLM_CHECKOUT/vllm" && -d "$FLASHMLA_CHECKOUT/flash_mla" ]] || exit 2
[[ ! -e "$OUTPUT_ARCHIVE" && ! -e "${OUTPUT_ARCHIVE}.sha256" ]] || {
    echo "refusing to replace existing runtime output" >&2
    exit 2
}
[[ "$(git -C "$VLLM_CHECKOUT" rev-parse HEAD)" == "$EXPECTED_COMMIT" ]]
[[ -z "$(git -C "$VLLM_CHECKOUT" status --porcelain --untracked-files=all)" ]]
[[ "$(git -C "$FLASHMLA_CHECKOUT" rev-parse HEAD)" == "$FLASHMLA_COMMIT" ]]
[[ -z "$(git -C "$FLASHMLA_CHECKOUT" status --porcelain --untracked-files=all)" ]]

work_root="${SCRATCH_ROOT%/}/${SLURM_JOB_ID:?SLURM_JOB_ID is required}/dflash2-runtime"
prepared="$work_root/runtime"
mkdir -p "$prepared" "$(dirname "$OUTPUT_ARCHIVE")"
tar --extract --file="$BASE_RUNTIME" --directory="$prepared"

mounts="${work_root}:${work_root},${VLLM_CHECKOUT}:${VLLM_CHECKOUT}:ro,${FLASHMLA_CHECKOUT}:${FLASHMLA_CHECKOUT}:ro,${LAUNCHER_ROOT}:${LAUNCHER_ROOT}:ro"
srun --nodes=1 --ntasks=1 --ntasks-per-node=1 --kill-on-bad-exit=1 \
    --no-container-mount-home --container-image="$IMAGE" --container-mounts="$mounts" \
    bash -s -- "$prepared" "$VLLM_CHECKOUT" "$FLASHMLA_CHECKOUT" "$LAUNCHER_ROOT" \
    "$EXPECTED_COMMIT" "$REQUIRED_COMMIT" <<'INNER'
set -euo pipefail
prepared="$1"
checkout="$2"
flashmla_checkout="$3"
launcher="$4"
expected="$5"
required="$6"

system_package="$($prepared/bin/python - <<'PY'
from pathlib import Path
import vllm
print(Path(vllm.__file__).resolve().parent)
PY
)"
site_package="$($prepared/bin/python - <<'PY'
import sysconfig
print(sysconfig.get_paths()["purelib"])
PY
)"
runtime_package="$site_package/vllm"
printf '%s\n' "$runtime_package" >"$prepared/dflash2-vllm-runtime-package.path"
mkdir -p "$runtime_package"
cp -a "$checkout/vllm/." "$runtime_package/"
while IFS= read -r -d '' extension; do
    relative="${extension#"$system_package/"}"
    mkdir -p "$runtime_package/$(dirname "$relative")"
    cp -a "$extension" "$runtime_package/$relative"
done < <(find "$system_package" -type f \( -name '*.so' -o -name '_version.py' \) -print0)
"$prepared/bin/python" - "$flashmla_checkout" "$runtime_package" <<'PY'
import sys
from pathlib import Path

source = Path(sys.argv[1]) / "flash_mla/flash_mla_interface.py"
output = Path(sys.argv[2]) / "third_party/flashmla/flash_mla_interface.py"
needle = "flash_mla_cuda = torch.ops._flashmla_C"
raw = source.read_text()
if raw.count(needle) != 1:
    raise SystemExit("FlashMLA interface cannot be transformed deterministically")
output.parent.mkdir(parents=True, exist_ok=True)
output.write_text(raw.replace(needle, f"import vllm._flashmla_C\n{needle}"))
PY
INNER

runtime_package="$(<"$prepared/dflash2-vllm-runtime-package.path")"
receipt="$prepared/dflash2-vllm-runtime-receipt.json"
PYTHONPATH="$LAUNCHER_ROOT/tools/launcher" python3 \
    "$LAUNCHER_ROOT/tools/launcher/common/specdec/dflash2_runtime_contract.py" vllm-receipt \
    --package "$VLLM_CHECKOUT/vllm" --runtime-package "$runtime_package" --output "$receipt" \
    --expected-commit "$EXPECTED_COMMIT" --required-ancestor "$REQUIRED_COMMIT" \
    --flashmla-package "$FLASHMLA_CHECKOUT/flash_mla" \
    --flashmla-expected-commit "$FLASHMLA_COMMIT" >"$prepared/dflash2-vllm-runtime-receipt.sha256"
receipt_sha="$(<"$prepared/dflash2-vllm-runtime-receipt.sha256")"
srun --nodes=1 --ntasks=1 --ntasks-per-node=1 --kill-on-bad-exit=1 \
    --no-container-mount-home --container-image="$IMAGE" --container-mounts="$mounts" \
    bash -s -- "$prepared" "$runtime_package" "$receipt" "$receipt_sha" \
    "$EXPECTED_COMMIT" "$REQUIRED_COMMIT" "$LAUNCHER_ROOT" <<'INNER'
set -euo pipefail
prepared="$1"
runtime_package="$2"
receipt="$3"
receipt_sha="$4"
expected="$5"
required="$6"
launcher="$7"
PYTHONPATH="$launcher" "$prepared/bin/python" - "$runtime_package" "$receipt" \
    "$receipt_sha" "$expected" "$required" <<'PY'
import sys
from pathlib import Path
import torch
from tools.launcher.common.specdec.dflash2_runtime_contract import verify_vllm_runtime

verify_vllm_runtime(Path(sys.argv[1]), Path(sys.argv[2]), *sys.argv[3:])
from vllm.model_executor.models.qwen3_dflash2 import _score_edges
from vllm.v1.worker.gpu.spec_decode.dflash2.speculator import DFlash2Speculator
from vllm.v1.attention.ops.flashmla import is_flashmla_dense_supported

scores = _score_edges(
    torch.ones(8, 4), torch.ones(8, 4), torch.tensor([[[1, 2], [3, 4]]]),
    torch.zeros(1, 2, 2), torch.ones(1, 2, 4), torch.tensor([0]), 2,
)
assert scores.shape == (1, 2, 2, 2)
assert DFlash2Speculator.__name__ == "DFlash2Speculator"
assert callable(is_flashmla_dense_supported)
PY
INNER

archive="$work_root/runtime.tar.zst"
tar --create --use-compress-program=zstd --file="$archive" -C "$prepared" .
archive_sha="$(sha256sum "$archive" | cut -d' ' -f1)"
temporary="${OUTPUT_ARCHIVE}.partial-${SLURM_JOB_ID}"
cp "$archive" "$temporary"
mv "$temporary" "$OUTPUT_ARCHIVE"
checksum="${OUTPUT_ARCHIVE}.sha256.partial-${SLURM_JOB_ID}"
printf '%s  %s\n' "$archive_sha" "$(basename "$OUTPUT_ARCHIVE")" >"$checksum"
mv "$checksum" "${OUTPUT_ARCHIVE}.sha256"
(cd "$(dirname "$OUTPUT_ARCHIVE")" && sha256sum -c "$(basename "${OUTPUT_ARCHIVE}.sha256")")
