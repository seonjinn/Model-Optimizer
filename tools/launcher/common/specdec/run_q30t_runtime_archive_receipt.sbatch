#!/bin/bash -p
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

#SBATCH --account=coreai_dlalgo_llm
#SBATCH --partition=36x2-a01r
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=400G
#SBATCH --time=00:45:00
#SBATCH --export=NONE

set -euo pipefail
umask 077

[[ "$#" == 7 ]] || {
    builtin printf '%s\n' "usage: $0 ARCHIVE ARCHIVE_SHA256 SOURCE_CHECKOUT SOURCE_COMMIT PRODUCER_SHA256 OUTPUT_RECEIPT OUTPUT_RECEIPT_PARENT" >&2
    builtin exit 2
}
readonly archive="$1"
readonly archive_sha256="$2"
readonly source_checkout="$3"
readonly source_commit="$4"
readonly producer_sha256="$5"
readonly output_receipt="$6"
readonly output_receipt_parent="$7"

[[ "$-" == *p* ]] || {
    builtin printf '%s\n' "Q30 runtime archive receipt requires Bash privileged mode" >&2
    builtin exit 2
}
if builtin compgen -A function >/dev/null; then
    builtin printf '%s\n' "imported shell functions are prohibited" >&2
    builtin exit 2
fi

reject_prohibited_environment() {
    local exported_name
    while IFS= builtin read -r exported_name; do
        case "$exported_name" in
            BASH_ENV|ENV|BASH_FUNC_*|CDPATH|GLOBIGNORE|SHELLOPTS|BASHOPTS|LD_*|DYLD_*|PYTHON*|PERL5LIB|RUBYLIB)
                builtin printf 'prohibited exported environment variable: %s\n' "$exported_name" >&2
                return 2
                ;;
        esac
    done < <(builtin compgen -e)
}

clear_exported_environment() {
    local exported_name
    while IFS= builtin read -r exported_name; do
        builtin unset -v "$exported_name" || {
            builtin printf 'cannot clear exported environment variable: %s\n' "$exported_name" >&2
            return 2
        }
    done < <(builtin compgen -e)
    if builtin compgen -e >/dev/null; then
        builtin printf '%s\n' "exported environment remains after clearing" >&2
        return 2
    fi
}

reject_prohibited_environment
readonly slurm_export_env="${SLURM_EXPORT_ENV:-}"
readonly slurm_job_id="${SLURM_JOB_ID:-}"
readonly slurm_nnodes="${SLURM_NNODES:-}"
readonly slurm_job_gpus="${SLURM_JOB_GPUS:-}"
readonly slurm_step_gpus="${SLURM_STEP_GPUS:-}"
readonly slurm_gpus_on_node="${SLURM_GPUS_ON_NODE:-}"
readonly slurm_gpus="${SLURM_GPUS:-}"
readonly slurm_gpus_per_node="${SLURM_GPUS_PER_NODE:-}"
readonly slurm_gpus_per_task="${SLURM_GPUS_PER_TASK:-}"
readonly slurm_tres_per_node="${SLURM_TRES_PER_NODE:-}"
readonly runtime_user="${USER:-}"
readonly shell_ostype="${OSTYPE:-}"
readonly q30t_test_allow_system_executables="${Q30T_TEST_ALLOW_SYSTEM_EXECUTABLES:-}"
readonly q30t_test_bash="${Q30T_TEST_BASH:-}"
readonly q30t_test_env="${Q30T_TEST_ENV:-}"
readonly q30t_test_python="${Q30T_TEST_PYTHON:-}"
readonly q30t_test_tar="${Q30T_TEST_TAR:-}"
readonly q30t_test_zstd="${Q30T_TEST_ZSTD:-}"
readonly q30t_test_dirname="${Q30T_TEST_DIRNAME:-}"
readonly q30t_test_mkdir="${Q30T_TEST_MKDIR:-}"
clear_exported_environment

readonly approved_bash="/bin/bash"
readonly approved_env="/usr/bin/env"
readonly approved_python="/usr/bin/python3.12"
readonly approved_tar="/usr/bin/tar"
readonly approved_zstd="/usr/bin/zstd"
readonly approved_dirname="/usr/bin/dirname"
readonly approved_mkdir="/bin/mkdir"
system_bash="$approved_bash"
system_env="$approved_env"
system_python="$approved_python"
system_tar="$approved_tar"
system_zstd="$approved_zstd"
system_dirname="$approved_dirname"
system_mkdir="$approved_mkdir"
test_executables=0

readonly validate_executable='import os,stat,sys
path=sys.argv[1]
entry=os.lstat(path)
resolved=os.path.realpath(path)
target=os.stat(resolved)
entry_type=stat.S_ISREG(entry.st_mode) or stat.S_ISLNK(entry.st_mode)
entry_permissions=stat.S_ISLNK(entry.st_mode) or not entry.st_mode & 0o022
valid_entry=entry.st_uid == 0 and entry_type and entry_permissions
valid_target=(os.path.isabs(resolved) and target.st_uid == 0
              and not target.st_mode & 0o022 and stat.S_ISREG(target.st_mode)
              and os.access(resolved, os.X_OK))
if not (valid_entry and valid_target):
    raise SystemExit(2)
print(resolved)'

if [[ -n "$q30t_test_allow_system_executables" || -n "$q30t_test_bash" \
    || -n "$q30t_test_env" || -n "$q30t_test_python" || -n "$q30t_test_tar" \
    || -n "$q30t_test_zstd" || -n "$q30t_test_dirname" || -n "$q30t_test_mkdir" ]]; then
    [[ "$shell_ostype" != linux* \
        && "$q30t_test_allow_system_executables" == non-linux-test \
        && "$q30t_test_bash" == /* && "$q30t_test_env" == /* \
        && "$q30t_test_python" == /* && "$q30t_test_tar" == /* \
        && "$q30t_test_zstd" == /* && "$q30t_test_dirname" == /* \
        && "$q30t_test_mkdir" == /* ]] || {
        builtin printf '%s\n' "test executable configuration is invalid" >&2
        builtin exit 2
    }
    system_bash="$q30t_test_bash"
    system_env="$q30t_test_env"
    system_python="$q30t_test_python"
    system_tar="$q30t_test_tar"
    system_zstd="$q30t_test_zstd"
    system_dirname="$q30t_test_dirname"
    system_mkdir="$q30t_test_mkdir"
    for executable in "$system_bash" "$system_env" "$system_python" "$system_tar" \
        "$system_zstd" "$system_dirname" "$system_mkdir"; do
        [[ -f "$executable" && -x "$executable" && ! -L "$executable" ]] || {
            builtin printf '%s\n' "approved test executable is unavailable" >&2
            builtin exit 2
        }
    done
    test_executables=1
else
    [[ -f "$approved_python" && -x "$approved_python" ]] || {
        builtin printf '%s\n' "approved Python validator is unavailable" >&2
        builtin exit 2
    }
    system_python="$($approved_python -S -I -c "$validate_executable" "$approved_python")" || {
        builtin printf '%s\n' "approved Python executable failed ownership validation" >&2
        builtin exit 2
    }
    system_bash="$($system_python -S -I -c "$validate_executable" "$approved_bash")" || {
        builtin printf '%s\n' "approved Bash executable failed ownership validation" >&2
        builtin exit 2
    }
    system_env="$($system_python -S -I -c "$validate_executable" "$approved_env")" || {
        builtin printf '%s\n' "approved env executable failed ownership validation" >&2
        builtin exit 2
    }
    system_tar="$($system_python -S -I -c "$validate_executable" "$approved_tar")" || {
        builtin printf '%s\n' "approved tar executable failed ownership validation" >&2
        builtin exit 2
    }
    system_zstd="$($system_python -S -I -c "$validate_executable" "$approved_zstd")" || {
        builtin printf '%s\n' "approved zstd executable failed ownership validation" >&2
        builtin exit 2
    }
    system_dirname="$($system_python -S -I -c "$validate_executable" "$approved_dirname")" || {
        builtin printf '%s\n' "approved dirname executable failed ownership validation" >&2
        builtin exit 2
    }
    system_mkdir="$($system_python -S -I -c "$validate_executable" "$approved_mkdir")" || {
        builtin printf '%s\n' "approved mkdir executable failed ownership validation" >&2
        builtin exit 2
    }
fi
readonly system_bash system_env system_python system_tar system_zstd system_dirname system_mkdir
readonly test_executables

readonly expected_archive="/lustre/fsw/coreai_dlalgo_llm/users/sna/modelopt-qwen3-drafter-training/assets/q235-training-prereqs-vllm0271-v1/runtime/modelopt-vllm-0.27.1-py312-aarch64-symlinks.tar.zst"
readonly expected_archive_sha256="4a20aee61f290c48bed22a84b4a0ae0cbdc54e3e3910854d253188c8854f5dc9"

[[ "$slurm_export_env" == NONE ]] || {
    builtin printf '%s\n' "Q30 runtime archive receipt requires --export=NONE" >&2
    builtin exit 2
}
[[ "$slurm_job_id" =~ ^[0-9]+$ && "$slurm_nnodes" == 1 && "$runtime_user" =~ ^[A-Za-z0-9._-]+$ ]] || {
    builtin printf '%s\n' "Q30 runtime archive receipt requires exactly one Slurm node" >&2
    builtin exit 2
}
[[ -z "$slurm_job_gpus" && -z "$slurm_step_gpus" \
    && ( -z "$slurm_gpus_on_node" || "$slurm_gpus_on_node" == 0 ) \
    && ( -z "$slurm_gpus" || "$slurm_gpus" == 0 ) \
    && ( -z "$slurm_gpus_per_node" || "$slurm_gpus_per_node" == 0 ) \
    && ( -z "$slurm_gpus_per_task" || "$slurm_gpus_per_task" == 0 ) \
    && "$slurm_tres_per_node" != *gpu* ]] || {
    builtin printf '%s\n' "Q30 runtime archive receipt requires a CPU-only allocation" >&2
    builtin exit 2
}
[[ "$archive" == "$expected_archive" && "$archive_sha256" == "$expected_archive_sha256" ]] || {
    builtin printf '%s\n' "Q30 runtime archive identity is not approved" >&2
    builtin exit 2
}
[[ "$source_checkout" == /* && -d "$source_checkout" && ! -L "$source_checkout" \
    && ( "$test_executables" == 1 || "$source_checkout" == /home/* ) \
    && "$source_commit" =~ ^[0-9a-f]{40}$ && "$producer_sha256" =~ ^[0-9a-f]{64}$ ]] || {
    builtin printf '%s\n' "Q30 runtime archive source identity is invalid" >&2
    builtin exit 2
}
[[ "$output_receipt" == /* && "$output_receipt_parent" == /* \
    && "$("$system_dirname" -- "$output_receipt")" == "$output_receipt_parent" ]] || {
    builtin printf '%s\n' "Q30 runtime archive receipt destination is invalid" >&2
    builtin exit 2
}

USER="$runtime_user"
SLURM_JOB_ID="$slurm_job_id"
readonly USER SLURM_JOB_ID
scratch_root="/raid/scratch/$USER/q30t-runtime-archive-$SLURM_JOB_ID"
readonly scratch_root
[[ ! -e "$scratch_root" && ! -L "$scratch_root" ]] || {
    builtin printf '%s\n' "Q30 runtime archive scratch root is not fresh" >&2
    builtin exit 2
}
"$system_mkdir" -m 0700 -- "$scratch_root"
"$system_mkdir" -p -- "$output_receipt_parent"

"$system_env" -i PATH=/usr/bin:/bin PYTHONPATH="$source_checkout/tools/launcher" \
    "$system_python" -m common.specdec.q30t_runtime_archive_receipt produce \
    --archive "$archive" \
    --archive-sha256 "$archive_sha256" \
    --extraction-root "$scratch_root/runtime" \
    --source-commit "$source_commit" \
    --producer "$source_checkout/tools/launcher/common/specdec/q30t_runtime_archive_receipt.py" \
    --producer-sha256 "$producer_sha256" \
    --output "$output_receipt" \
    --job-id "$SLURM_JOB_ID"
