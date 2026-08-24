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
build_source="$work_root/vllm-source"
build_directory="$work_root/vllm-build"
mkdir -p "$prepared" "$build_source" "$build_directory" "$(dirname "$OUTPUT_ARCHIVE")"
tar --extract --file="$BASE_RUNTIME" --directory="$prepared"
git -C "$VLLM_CHECKOUT" archive HEAD | tar -x -C "$build_source"

mounts="${work_root}:${work_root},${VLLM_CHECKOUT}:${VLLM_CHECKOUT}:ro,${FLASHMLA_CHECKOUT}:${FLASHMLA_CHECKOUT}:ro,${LAUNCHER_ROOT}:${LAUNCHER_ROOT}:ro"
srun --nodes=1 --ntasks=1 --ntasks-per-node=1 --kill-on-bad-exit=1 \
    --no-container-mount-home --container-image="$IMAGE" --container-mounts="$mounts" \
    bash -s -- "$prepared" "$VLLM_CHECKOUT" "$FLASHMLA_CHECKOUT" "$build_source" \
    "$build_directory" <<'INNER'
set -euo pipefail
prepared="$1"
checkout="$2"
flashmla_checkout="$3"
build_source="$4"
build_directory="$5"
configure_log="$prepared/dflash2-flashmla-cmake-configure.log"
"$prepared/bin/python" -m pip install --disable-pip-version-check --no-cache-dir \
    --only-binary=:all: --no-deps --ignore-installed cmake==3.31.6 ninja==1.13.0
{
    "$prepared/bin/cmake" --version
    "$prepared/bin/ninja" --version
} >"$configure_log"

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
rm -f "$runtime_package"/_flashmla_C*.so "$runtime_package"/_flashmla_extension_C*.so
cp -a "$checkout/vllm/." "$runtime_package/"
while IFS= read -r -d '' extension; do
    relative="${extension#"$system_package/"}"
    case "$relative" in
        _flashmla_C*.so|_flashmla_extension_C*.so) continue ;;
    esac
    mkdir -p "$runtime_package/$(dirname "$relative")"
    cp -a "$extension" "$runtime_package/$relative"
done < <(find "$system_package" -type f \( -name '*.so' -o -name '_version.py' \) -print0)
TORCH_CUDA_ARCH_LIST=10.0a FLASH_MLA_SRC_DIR="$flashmla_checkout" \
    "$prepared/bin/cmake" -S "$build_source" -B "$build_directory" \
    -G Ninja -DVLLM_TARGET_DEVICE=cuda -DVLLM_PYTHON_EXECUTABLE="$prepared/bin/python" \
    -DCMAKE_INSTALL_PREFIX="$site_package" -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_MAKE_PROGRAM="$prepared/bin/ninja" -DNVCC_THREADS=8 2>&1 | tee -a "$configure_log"
grep -Eq 'CUDA target architectures:.*10\.0[af]' "$configure_log"
grep -Eq 'FlashMLA CUDA architectures:.*10\.0[af]' "$configure_log"
FLASH_MLA_SRC_DIR="$flashmla_checkout" "$prepared/bin/cmake" --build "$build_directory" \
    --target _flashmla_C _flashmla_extension_C --parallel 16
"$prepared/bin/cmake" --install "$build_directory" --component _flashmla_C
"$prepared/bin/cmake" --install "$build_directory" --component _flashmla_extension_C
INNER

runtime_package="$(<"$prepared/dflash2-vllm-runtime-package.path")"
build_manifest="$prepared/dflash2-flashmla-build-manifest.json"
PYTHONPATH="$LAUNCHER_ROOT/tools/launcher" python3 \
    "$LAUNCHER_ROOT/tools/launcher/common/specdec/dflash2_runtime_contract.py" \
    flashmla-build-manifest --output "$build_manifest" --runtime-package "$runtime_package" \
    --vllm-package "$VLLM_CHECKOUT/vllm" --flashmla-package "$FLASHMLA_CHECKOUT/flash_mla" \
    --base-runtime "$BASE_RUNTIME" --image "$IMAGE" --builder "${BASH_SOURCE[0]}" \
    --configure-log "$prepared/dflash2-flashmla-cmake-configure.log" \
    --vllm-commit "$EXPECTED_COMMIT" --flashmla-commit "$FLASHMLA_COMMIT" \
    >"$prepared/dflash2-flashmla-build-manifest.sha256"
receipt="$prepared/dflash2-vllm-runtime-receipt.json"
PYTHONPATH="$LAUNCHER_ROOT/tools/launcher" python3 \
    "$LAUNCHER_ROOT/tools/launcher/common/specdec/dflash2_runtime_contract.py" vllm-receipt \
    --package "$VLLM_CHECKOUT/vllm" --runtime-package "$runtime_package" --output "$receipt" \
    --expected-commit "$EXPECTED_COMMIT" --required-ancestor "$REQUIRED_COMMIT" \
    --flashmla-package "$FLASHMLA_CHECKOUT/flash_mla" \
    --flashmla-expected-commit "$FLASHMLA_COMMIT" --flashmla-build-manifest "$build_manifest" \
    >"$prepared/dflash2-vllm-runtime-receipt.sha256"
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
import vllm._flashmla_C
import vllm._flashmla_extension_C
from vllm.third_party.flashmla import flash_mla_interface
from vllm.v1.attention.ops.flashmla import _is_flashmla_available

scores = _score_edges(
    torch.ones(8, 4), torch.ones(8, 4), torch.tensor([[[1, 2], [3, 4]]]),
    torch.zeros(1, 2, 2), torch.ones(1, 2, 4), torch.tensor([0]), 2,
)
assert scores.shape == (1, 2, 2, 2)
assert DFlash2Speculator.__name__ == "DFlash2Speculator"
assert flash_mla_interface.flash_mla_cuda is torch.ops._flashmla_C
assert _is_flashmla_available() == (True, None)
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
