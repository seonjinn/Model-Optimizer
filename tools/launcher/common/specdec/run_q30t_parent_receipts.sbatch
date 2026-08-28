#!/bin/bash -p
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

#SBATCH --account=coreai_dlalgo_llm
#SBATCH --partition=36x2-a01r
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=0
#SBATCH --segment=1
#SBATCH --time=01:00:00
#SBATCH --export=NONE

set -euo pipefail
LC_ALL=C

[[ "$-" == *p* ]] || {
    builtin printf '%s\n' "Q30 parent receipt requires Bash privileged mode" >&2
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
            BASH_ENV|ENV|BASH_FUNC_*|CDPATH|GLOBIGNORE|SHELLOPTS|BASHOPTS|LD_*|DYLD_*|PYTHON*|PERL5LIB|RUBYLIB|GIT_*|SSH_ASKPASS|SSH_ASKPASS_REQUIRE)
                echo "prohibited exported environment variable: $exported_name" >&2
                return 2
                ;;
        esac
    done < <(builtin compgen -e)
}

clear_exported_environment() {
    local exported_name
    while IFS= builtin read -r exported_name; do
        builtin unset -v "$exported_name" || {
            echo "cannot clear exported environment variable: $exported_name" >&2
            return 2
        }
    done < <(builtin compgen -e)
    if builtin compgen -e >/dev/null; then
        echo "exported environment remains after clearing" >&2
        return 2
    fi
}

[[ "${SLURM_EXPORT_ENV:-}" == NONE ]] || {
    echo "Q30 parent receipt requires the Slurm --export=NONE boundary" >&2
    exit 2
}
reject_prohibited_environment
readonly slurm_job_id="${SLURM_JOB_ID:-}"
readonly slurm_nnodes="${SLURM_NNODES:-}"
readonly slurm_cpus_per_task="${SLURM_CPUS_PER_TASK:-}"
readonly slurm_job_gpus="${SLURM_JOB_GPUS:-}"
readonly slurm_step_gpus="${SLURM_STEP_GPUS:-}"
readonly slurm_gpus_on_node="${SLURM_GPUS_ON_NODE:-}"
readonly slurm_gpus="${SLURM_GPUS:-}"
readonly slurm_gpus_per_node="${SLURM_GPUS_PER_NODE:-}"
readonly slurm_gpus_per_task="${SLURM_GPUS_PER_TASK:-}"
readonly slurm_tres_per_node="${SLURM_TRES_PER_NODE:-}"
readonly q30t_test_allow_system_executables="${Q30T_TEST_ALLOW_SYSTEM_EXECUTABLES:-}"
readonly q30t_test_git="${Q30T_TEST_GIT:-}"
readonly q30t_test_python="${Q30T_TEST_PYTHON:-}"
readonly q30t_test_git_evidence="${Q30T_TEST_GIT_EVIDENCE:-}"
readonly q30t_test_real_git="${Q30T_TEST_REAL_GIT:-}"
clear_exported_environment

readonly approved_git="/usr/bin/git"
readonly approved_python="/usr/bin/python3"
readonly approved_ssh="/usr/bin/ssh"
readonly approved_env="/usr/bin/env"
system_git="$approved_git"
system_python="$approved_python"
system_ssh="$approved_ssh"
system_env="$approved_env"
test_executables=0
git_test_arguments=()
if [[ -n "$q30t_test_allow_system_executables" \
    || -n "$q30t_test_git" || -n "$q30t_test_python" \
    || -n "$q30t_test_git_evidence" || -n "$q30t_test_real_git" ]]; then
    [[ "$OSTYPE" != linux* \
        && "$q30t_test_allow_system_executables" == non-linux-test \
        && "$q30t_test_git" == /* && "$q30t_test_python" == /* \
        && "$q30t_test_git_evidence" == /* && "$q30t_test_real_git" == /* ]] || {
        echo "system executable test override is forbidden" >&2
        exit 2
    }
    system_git="$q30t_test_git"
    system_python="$q30t_test_python"
    git_test_arguments=(
        --q30t-test-exec "$q30t_test_git_evidence" "$q30t_test_real_git"
    )
    test_executables=1
fi

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

if ((test_executables)); then
    for executable in "$system_git" "$system_python" "$system_ssh" "$system_env" \
        "$q30t_test_real_git"; do
        [[ -f "$executable" && -x "$executable" && ! -L "$executable" ]] || {
            echo "approved test executable is unavailable" >&2
            exit 2
        }
    done
else
    system_python="$($approved_python -S -I -c "$validate_executable" "$approved_python")" || {
        echo "approved Python executable failed ownership validation" >&2
        exit 2
    }
    system_git="$($system_python -S -I -c "$validate_executable" "$approved_git")" || {
        echo "approved Git executable failed ownership validation" >&2
        exit 2
    }
    system_ssh="$($system_python -S -I -c "$validate_executable" "$approved_ssh")" || {
        echo "approved SSH executable failed ownership validation" >&2
        exit 2
    }
    system_env="$($system_python -S -I -c "$validate_executable" "$approved_env")" || {
        echo "approved env executable failed ownership validation" >&2
        exit 2
    }
fi
readonly system_git system_python system_ssh system_env test_executables

[[ "$#" == 14 ]] || {
    echo "usage: $0 REPO_ROOT SOURCE_COMMIT SOURCE_REMOTE_REF HISTORICAL_AUDIT TARGET_MODEL_ROOT TARGET_REVISION TARGET_IDENTITY_PATH TARGET_IDENTITY_SHA256 TARGET_MODEL_SHA256_PATH TARGET_MODEL_SHA256_SHA256 DFLASH_PARENT_ROOT DSPARK_PARENT_ROOT RUNTIME_ARCHIVE OUTPUT_ROOT" >&2
    exit 2
}
readonly REPO_ROOT="$1"
readonly SOURCE_COMMIT="$2"
readonly SOURCE_REMOTE_REF="$3"
readonly HISTORICAL_AUDIT="$4"
readonly TARGET_MODEL_ROOT="$5"
readonly TARGET_REVISION="$6"
readonly TARGET_IDENTITY_PATH="$7"
readonly TARGET_IDENTITY_SHA256="$8"
readonly TARGET_MODEL_SHA256_PATH="$9"
readonly TARGET_MODEL_SHA256_SHA256="${10}"
readonly DFLASH_PARENT_ROOT="${11}"
readonly DSPARK_PARENT_ROOT="${12}"
readonly RUNTIME_ARCHIVE="${13}"
readonly OUTPUT_ROOT="${14}"

[[ "$slurm_job_id" =~ ^[0-9]+$ ]] || {
    echo "invalid Slurm job identity" >&2
    exit 2
}
[[ "$slurm_nnodes" == 1 && "$slurm_cpus_per_task" == 64 ]] || {
    echo "Q30 parent receipt generation requires exactly one node and 64 CPUs" >&2
    exit 2
}
[[ -z "$slurm_job_gpus" && -z "$slurm_step_gpus" \
    && ( -z "$slurm_gpus_on_node" || "$slurm_gpus_on_node" == 0 ) \
    && ( -z "$slurm_gpus" || "$slurm_gpus" == 0 ) \
    && ( -z "$slurm_gpus_per_node" || "$slurm_gpus_per_node" == 0 ) \
    && ( -z "$slurm_gpus_per_task" || "$slurm_gpus_per_task" == 0 ) \
    && "$slurm_tres_per_node" != *gpu* ]] || {
    echo "Q30 parent receipt generation requires a CPU-only allocation" >&2
    exit 2
}
[[ "$SOURCE_COMMIT" =~ ^[0-9a-f]{40}$ && "$TARGET_REVISION" =~ ^[0-9a-f]{40}$ ]] || {
    echo "invalid source or target revision" >&2
    exit 2
}
[[ "$TARGET_IDENTITY_SHA256" =~ ^[0-9a-f]{64}$ \
    && "$TARGET_MODEL_SHA256_SHA256" =~ ^[0-9a-f]{64}$ ]] || {
    echo "invalid target model manifest identity" >&2
    exit 2
}

for path in "$REPO_ROOT" "$HISTORICAL_AUDIT" "$TARGET_MODEL_ROOT" \
    "$TARGET_IDENTITY_PATH" "$TARGET_MODEL_SHA256_PATH" "$DFLASH_PARENT_ROOT" \
    "$DSPARK_PARENT_ROOT" "$RUNTIME_ARCHIVE" "$OUTPUT_ROOT"; do
    [[ "$path" == /* ]] || {
        echo "Q30 parent paths must be absolute" >&2
        exit 2
    }
done
[[ -d "$REPO_ROOT" && ! -L "$REPO_ROOT" ]] || exit 2
[[ -f "$HISTORICAL_AUDIT" && ! -L "$HISTORICAL_AUDIT" ]] || exit 2
[[ -f "$RUNTIME_ARCHIVE" && ! -L "$RUNTIME_ARCHIVE" ]] || exit 2
[[ -f "$TARGET_IDENTITY_PATH" && ! -L "$TARGET_IDENTITY_PATH" ]] || exit 2
[[ -f "$TARGET_MODEL_SHA256_PATH" && ! -L "$TARGET_MODEL_SHA256_PATH" ]] || exit 2
for path in "$TARGET_MODEL_ROOT" "$DFLASH_PARENT_ROOT" "$DSPARK_PARENT_ROOT"; do
    [[ -d "$path" && ! -L "$path" ]] || exit 2
done
[[ ! -L "$OUTPUT_ROOT" && ( ! -e "$OUTPUT_ROOT" || -d "$OUTPUT_ROOT" ) ]] || {
    echo "Q30 parent retry output root is not a directory" >&2
    exit 2
}

run_git() (
    exec "$system_env" -i \
        LC_ALL=C \
        GIT_CONFIG_NOSYSTEM=1 \
        GIT_CONFIG_GLOBAL=/dev/null \
        GIT_NO_REPLACE_OBJECTS=1 \
        GIT_TERMINAL_PROMPT=0 \
        "$system_git" "${git_test_arguments[@]}" \
        -c core.fsmonitor=false \
        -c core.hooksPath=/dev/null \
        -c protocol.file.allow=never \
        -C "$REPO_ROOT" "$@"
)

run_remote_git() (
    exec "$system_env" -i \
        LC_ALL=C \
        GIT_CONFIG_NOSYSTEM=1 \
        GIT_CONFIG_GLOBAL=/dev/null \
        GIT_NO_REPLACE_OBJECTS=1 \
        GIT_TERMINAL_PROMPT=0 \
        GIT_SSH_COMMAND="$system_ssh -F /dev/null -oBatchMode=yes" \
        GIT_SSH_VARIANT=ssh \
        "$system_git" "${git_test_arguments[@]}" "$@"
)

[[ "$(run_git rev-parse HEAD)" == "$SOURCE_COMMIT" \
    && -z "$(run_git status --porcelain=v1 --untracked-files=all)" ]] || {
    echo "source checkout is not clean at the requested commit" >&2
    exit 2
}
if ((test_executables)); then
    readonly expected_remote_ref="origin/main"
    readonly remote_branch="refs/heads/main"
    remote_url="$(run_git config --get remote.origin.url)" || exit 2
else
    readonly expected_remote_ref="refs/remotes/gitlab/sj/q30t-ptv23-complement-700k"
    readonly remote_branch="refs/heads/sj/q30t-ptv23-complement-700k"
    readonly remote_url="ssh://git@gitlab-master.nvidia.com:12051/sna/modelopt.git"
fi
[[ "$SOURCE_REMOTE_REF" == "$expected_remote_ref" ]] || {
    echo "source reference is not the approved remote-tracking ref" >&2
    exit 2
}
remote_evidence="$(run_remote_git ls-remote --exit-code "$remote_url" "$remote_branch")" || {
    echo "pushed source reference is unavailable" >&2
    exit 2
}
[[ "$remote_evidence" != *$'\n'* ]] || exit 2
IFS=$'\t' builtin read -r remote_commit remote_name remote_extra <<<"$remote_evidence"
[[ "$remote_commit" == "$SOURCE_COMMIT" && "$remote_name" == "$remote_branch" \
    && -z "$remote_extra" ]] || {
    echo "requested commit is not the pushed branch tip" >&2
    exit 2
}

readonly module_bootstrap='import importlib.abc,importlib.util,io,runpy,sys,tarfile
target=sys.argv.pop(1)
commit=sys.argv.pop(1)
raw=sys.stdin.buffer.read(16*1024*1024+1)
if len(raw)>16*1024*1024:
    raise SystemExit("authenticated Python archive is too large")
sources={}
with tarfile.open(fileobj=io.BytesIO(raw),mode="r:") as archive:
    for member in archive:
        if member.isdir():
            continue
        if not member.isreg() or member.size>4*1024*1024:
            raise SystemExit("authenticated Python archive has an invalid member")
        prefix="tools/launcher/"
        if not member.name.startswith(prefix) or not member.name.endswith(".py"):
            continue
        stream=archive.extractfile(member)
        if stream is None:
            raise SystemExit("authenticated Python member is unreadable")
        name=member.name[len(prefix):-3].replace("/",".")
        if name in sources:
            raise SystemExit("authenticated Python archive has a duplicate module")
        sources[name]=stream.read()
if target not in sources:
    raise SystemExit("authenticated receipt module is absent")
packages={"common","common.specdec"}
class ArchiveLoader(importlib.abc.MetaPathFinder,importlib.abc.Loader):
    def find_spec(self,fullname,path=None,target=None):
        if fullname in packages or fullname in sources:
            return importlib.util.spec_from_loader(fullname,self,is_package=fullname in packages)
        return None
    def create_module(self,spec):
        return None
    def get_code(self,fullname):
        if fullname in packages:
            return compile("",f"<git:{commit}:{fullname}>","exec")
        return compile(sources[fullname],f"<git:{commit}:{fullname}>","exec")
    def exec_module(self,module):
        if module.__name__ in packages:
            module.__path__=[]
            return
        exec(self.get_code(module.__name__),module.__dict__)
sys.meta_path.insert(0,ArchiveLoader())
runpy.run_module(target,run_name="__main__")'
run_git archive --format=tar "$SOURCE_COMMIT" -- tools/launcher/common/specdec \
    | exec -c "$system_python" -S -I -c "$module_bootstrap" \
    "common.specdec.q30t_parent_receipt" "$SOURCE_COMMIT" \
    --historical-audit "$HISTORICAL_AUDIT" \
    --target-model-root "$TARGET_MODEL_ROOT" \
    --target-revision "$TARGET_REVISION" \
    --target-identity-path "$TARGET_IDENTITY_PATH" \
    --target-identity-sha256 "$TARGET_IDENTITY_SHA256" \
    --target-model-sha256-path "$TARGET_MODEL_SHA256_PATH" \
    --target-model-sha256-sha256 "$TARGET_MODEL_SHA256_SHA256" \
    --dflash-parent-root "$DFLASH_PARENT_ROOT" \
    --dspark-parent-root "$DSPARK_PARENT_ROOT" \
    --runtime-archive "$RUNTIME_ARCHIVE" \
    --source-commit "$SOURCE_COMMIT" \
    --output-root "$OUTPUT_ROOT"
