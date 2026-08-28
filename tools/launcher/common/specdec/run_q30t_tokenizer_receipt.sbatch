#!/bin/bash
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
LC_ALL=C

reject_prohibited_environment() {
    local exported_name
    while IFS= read -r exported_name; do
        case "$exported_name" in
            BASH_ENV|ENV|CDPATH|GLOBIGNORE|SHELLOPTS|BASHOPTS|LD_*|DYLD_*|PYTHON*|PERL5LIB|RUBYLIB|GIT_*|SSH_ASKPASS|SSH_ASKPASS_REQUIRE)
                echo "prohibited exported environment variable: $exported_name" >&2
                return 2
                ;;
        esac
    done < <(compgen -e)
}

require_reserved_environment() {
    local exported_name
    while IFS= read -r exported_name; do
        case "$exported_name" in
            PWD|SHLVL|_|SLURM_*|SPANK_*) ;;
            *)
                echo "unexpected exported environment variable under --export=NONE: $exported_name" >&2
                return 2
                ;;
        esac
    done < <(compgen -e)
}

[[ "${SLURM_EXPORT_ENV:-}" == NONE ]] || {
    echo "Q30 tokenizer receipt requires the Slurm --export=NONE boundary" >&2
    exit 2
}
reject_prohibited_environment

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
if [[ -n "${Q30T_TEST_ALLOW_SYSTEM_EXECUTABLES:-}" \
    || -n "${Q30T_TEST_GIT:-}" || -n "${Q30T_TEST_PYTHON:-}" \
    || -n "${Q30T_TEST_GIT_EVIDENCE:-}" || -n "${Q30T_TEST_REAL_GIT:-}" ]]; then
    [[ "$OSTYPE" != linux* \
        && "${Q30T_TEST_ALLOW_SYSTEM_EXECUTABLES:-}" == non-linux-test \
        && "${Q30T_TEST_GIT:-}" == /* && "${Q30T_TEST_PYTHON:-}" == /* \
        && "${Q30T_TEST_GIT_EVIDENCE:-}" == /* && "${Q30T_TEST_REAL_GIT:-}" == /* ]] || {
        echo "system executable test override is forbidden" >&2
        exit 2
    }
    system_git="$Q30T_TEST_GIT"
    system_python="$Q30T_TEST_PYTHON"
    git_test_arguments=(
        --q30t-test-exec "$Q30T_TEST_GIT_EVIDENCE" "$Q30T_TEST_REAL_GIT"
    )
    test_executables=1
else
    require_reserved_environment
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
        "$Q30T_TEST_REAL_GIT"; do
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

[[ "$#" == 11 ]] || {
    echo "usage: $0 SOURCE_PATH SOURCE_SHA SOURCE_REMOTE_REF TOKENIZER_REPOSITORY TOKENIZER_REVISION TOKENIZER_SNAPSHOT IDENTITY_PATH IDENTITY_SHA256 MODEL_SHA256_PATH MODEL_SHA256_SHA256 OUTPUT_PATH" >&2
    exit 2
}
readonly SOURCE_PATH="$1"
readonly SOURCE_SHA="$2"
readonly SOURCE_REMOTE_REF="$3"
readonly TOKENIZER_REPOSITORY="$4"
readonly TOKENIZER_REVISION="$5"
readonly TOKENIZER_SNAPSHOT="$6"
readonly IDENTITY_PATH="$7"
readonly IDENTITY_SHA256="$8"
readonly MODEL_SHA256_PATH="$9"
readonly MODEL_SHA256_SHA256="${10}"
readonly OUTPUT_PATH="${11}"

[[ "${SLURM_NNODES:-}" == 1 && "${SLURM_CPUS_PER_TASK:-}" == 64 \
    && "${SLURM_JOB_ID:-}" =~ ^[0-9]+$ ]] || {
    echo "invalid one-node Slurm CPU identity" >&2
    exit 2
}
[[ -z "${SLURM_JOB_GPUS:-}" && -z "${SLURM_STEP_GPUS:-}" \
    && ( -z "${SLURM_GPUS_ON_NODE:-}" || "${SLURM_GPUS_ON_NODE:-}" == 0 ) \
    && ( -z "${SLURM_GPUS:-}" || "${SLURM_GPUS:-}" == 0 ) \
    && ( -z "${SLURM_GPUS_PER_NODE:-}" || "${SLURM_GPUS_PER_NODE:-}" == 0 ) \
    && ( -z "${SLURM_GPUS_PER_TASK:-}" || "${SLURM_GPUS_PER_TASK:-}" == 0 ) \
    && "${SLURM_TRES_PER_NODE:-}" != *gpu* ]] || {
    echo "Q30 tokenizer receipt generation requires a CPU-only allocation" >&2
    exit 2
}
[[ "$SOURCE_SHA" =~ ^[0-9a-f]{40}$ && "$TOKENIZER_REVISION" =~ ^[0-9a-f]{40}$ ]] || {
    echo "invalid source or tokenizer revision" >&2
    exit 2
}
[[ "$IDENTITY_SHA256" =~ ^[0-9a-f]{64}$ && "$MODEL_SHA256_SHA256" =~ ^[0-9a-f]{64}$ ]] || {
    echo "invalid manifest identity" >&2
    exit 2
}
for path in "$SOURCE_PATH" "$TOKENIZER_SNAPSHOT" "$IDENTITY_PATH" "$MODEL_SHA256_PATH" \
    "$OUTPUT_PATH"; do
    [[ "$path" == /* ]] || {
        echo "Q30 tokenizer paths must be absolute" >&2
        exit 2
    }
done
[[ -d "$SOURCE_PATH" && ! -L "$SOURCE_PATH" ]] || exit 2
[[ -d "$TOKENIZER_SNAPSHOT" && ! -L "$TOKENIZER_SNAPSHOT" \
    && -f "$IDENTITY_PATH" && ! -L "$IDENTITY_PATH" \
    && -f "$MODEL_SHA256_PATH" && ! -L "$MODEL_SHA256_PATH" ]] || {
    echo "tokenizer snapshot or manifest is unavailable" >&2
    exit 2
}
[[ ! -L "$OUTPUT_PATH" && ( ! -e "$OUTPUT_PATH" || -f "$OUTPUT_PATH" ) ]] || {
    echo "tokenizer receipt retry path is not a regular file" >&2
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
        -C "$SOURCE_PATH" "$@"
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

[[ "$(run_git rev-parse HEAD)" == "$SOURCE_SHA" ]] || {
    echo "source checkout commit mismatch" >&2
    exit 2
}
[[ -z "$(run_git status --porcelain=v1 --untracked-files=all)" ]] || {
    echo "source checkout is dirty" >&2
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
IFS=$'\t' read -r remote_commit remote_name remote_extra <<<"$remote_evidence"
[[ "$remote_commit" == "$SOURCE_SHA" && "$remote_name" == "$remote_branch" \
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
run_git archive --format=tar "$SOURCE_SHA" -- tools/launcher/common/specdec \
    | exec -c "$system_python" -S -I -c "$module_bootstrap" \
    "common.specdec.q30t_tokenizer_receipt" "$SOURCE_SHA" \
    --repository "$TOKENIZER_REPOSITORY" \
    --revision "$TOKENIZER_REVISION" \
    --snapshot "$TOKENIZER_SNAPSHOT" \
    --identity-path "$IDENTITY_PATH" \
    --identity-sha256 "$IDENTITY_SHA256" \
    --model-sha256-path "$MODEL_SHA256_PATH" \
    --model-sha256-sha256 "$MODEL_SHA256_SHA256" \
    --output "$OUTPUT_PATH"
