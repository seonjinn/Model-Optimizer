#!/bin/bash -p
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail
export LC_ALL=C
umask 077

usage() {
    echo "usage: $0 (--test-only|--submit) --source-path PATH --output PATH --slurm-output PATH" >&2
    exit 2
}

fail() {
    echo "$1" >&2
    exit 2
}

[[ $# -ge 1 ]] || usage
readonly mode="$1"
shift
[[ "$mode" == "--test-only" || "$mode" == "--submit" ]] || usage

source_path=""
output=""
slurm_output=""
while (($#)); do
    [[ $# -ge 2 ]] || usage
    case "$1" in
        --source-path)
            [[ -z "$source_path" ]] || usage
            source_path="$2"
            ;;
        --output)
            [[ -z "$output" ]] || usage
            output="$2"
            ;;
        --slurm-output)
            [[ -z "$slurm_output" ]] || usage
            slurm_output="$2"
            ;;
        *) usage ;;
    esac
    shift 2
done

[[ "$source_path" == /* && "$source_path" != *"/../"* && "$source_path" != *"/./"* \
    && -d "$source_path" && ! -L "$source_path" ]] || fail "source path is unsafe"
receipt_root="/lustre/fsw/coreai_dlalgo_llm/users/sna/modelopt-qwen3-drafter-training/receipts/q30t-ptv23-complement-700k-v1"
approved_git="/usr/bin/git"
approved_sbatch="/usr/bin/sbatch"
approved_ssh="/usr/bin/ssh"
approved_env="/usr/bin/env"
approved_python="/usr/bin/python3.12"
approved_fetch_url="ssh://git@gitlab-master.nvidia.com:12051/sna/modelopt.git"
test_executables=0
if [[ -n "${Q30T_TEST_ALLOW_SYSTEM_EXECUTABLES:-}" \
    || -n "${Q30T_TEST_PYTHON:-}" || -n "${Q30T_TEST_GIT:-}" \
    || -n "${Q30T_TEST_SBATCH:-}" || -n "${Q30T_TEST_SSH:-}" ]]; then
    [[ "$OSTYPE" != linux* \
        && "${Q30T_TEST_ALLOW_SYSTEM_EXECUTABLES:-}" == non-linux-test \
        && "${Q30T_TEST_PYTHON:-}" == /* && "${Q30T_TEST_GIT:-}" == /* \
        && "${Q30T_TEST_SBATCH:-}" == /* && "${Q30T_TEST_SSH:-}" == /* \
        && "${Q30T_TEST_RECEIPT_ROOT:-}" == /* \
        && "${Q30T_TEST_FETCH_URL:-}" == /* ]] \
        || fail "system executable test override is forbidden"
    approved_python="$Q30T_TEST_PYTHON"
    approved_git="$Q30T_TEST_GIT"
    approved_sbatch="$Q30T_TEST_SBATCH"
    approved_ssh="$Q30T_TEST_SSH"
    receipt_root="$Q30T_TEST_RECEIPT_ROOT"
    approved_fetch_url="$Q30T_TEST_FETCH_URL"
    test_executables=1
fi
readonly receipt_root approved_git approved_sbatch approved_ssh approved_env approved_python
readonly approved_fetch_url test_executables
[[ "$output" =~ ^${receipt_root}/row-schema/observation-[ab]\.json$ ]] \
    || fail "output is not an approved row-schema observation path"
[[ "$slurm_output" =~ ^${receipt_root}/logs/row-schema-[ab]-%j\.out$ ]] \
    || fail "SLURM output is not an approved row-schema log path"

[[ "$-" == *p* ]] || fail "privileged Bash mode is required"
for name in $(compgen -e); do
    [[ "$name" != BASH_FUNC_* && "$name" != LD_* && "$name" != DYLD_* \
        && "$name" != BASH_ENV && "$name" != ENV && "$name" != CDPATH \
        && "$name" != PYTHONPATH && "$name" != PYTHONHOME ]] \
        || fail "unsafe inherited environment variable: $name"
done
unset PATH
PATH=/usr/bin:/bin
export PATH
authenticator_arguments=(/bin/bash "$approved_env" "$approved_python" \
    "$approved_git" "$approved_sbatch" "$approved_ssh")
((test_executables)) && authenticator_arguments=(--test-owner "${authenticator_arguments[@]}")
"$approved_python" -I -S - "${authenticator_arguments[@]}" <<'PY'
# Q30T_TOOL_AUTHENTICATOR
import os
import stat
import sys

arguments = sys.argv[1:]
test_owner = arguments[:1] == ["--test-owner"]
if test_owner:
    arguments = arguments[1:]
required_uids = {0, os.getuid()} if test_owner else {0}
for value in arguments:
    path = value
    if not os.path.isabs(path):
        raise SystemExit("approved executable path is not absolute")
    for _ in range(16):
        metadata = os.lstat(path)
        if metadata.st_uid not in required_uids or stat.S_IMODE(metadata.st_mode) & 0o022:
            raise SystemExit(f"approved executable is not root-owned or is writable: {value}")
        if not stat.S_ISLNK(metadata.st_mode):
            if not stat.S_ISREG(metadata.st_mode) or not os.access(path, os.X_OK):
                raise SystemExit(f"approved executable is not regular and executable: {value}")
            break
        target = os.readlink(path)
        path = os.path.normpath(target if os.path.isabs(target) else os.path.join(os.path.dirname(path), target))
    else:
        raise SystemExit(f"approved executable symlink chain is too deep: {value}")
if not test_owner and sys.version_info < (3, 12):
    raise SystemExit("approved Python must be Python 3.12 or newer")
PY

readonly runner_path="tools/launcher/common/specdec/run_q30t_row_schema_observation.sbatch"
readonly -a git_command=(
    /usr/bin/env -i LC_ALL=C GIT_CONFIG_NOSYSTEM=1 GIT_CONFIG_SYSTEM=/dev/null
    GIT_CONFIG_GLOBAL=/dev/null GIT_NO_REPLACE_OBJECTS=1 "$approved_git"
    -c core.attributesFile=/dev/null -c core.hooksPath=/dev/null
    -c core.fsmonitor=false -c core.untrackedCache=false -C "$source_path"
)

worktree_root="$("${git_command[@]}" rev-parse --show-toplevel)"
[[ "$worktree_root" == "$source_path" ]] || fail "source path is not the worktree root"
source_commit="$("${git_command[@]}" rev-parse --verify 'HEAD^{commit}')"
[[ "$source_commit" =~ ^[0-9a-f]{40}$ ]] || fail "source HEAD is invalid"
upstream_ref="$("${git_command[@]}" rev-parse --symbolic-full-name '@{upstream}')"
[[ "$upstream_ref" == refs/remotes/gitlab/* ]] \
    || fail "source upstream is not the approved GitLab remote"
upstream_commit="$("${git_command[@]}" rev-parse --verify '@{upstream}^{commit}')"
[[ "$source_commit" == "$upstream_commit" ]] || fail "reviewed source commit is not pushed"
remote_url="$("${git_command[@]}" config --local --get remote.gitlab.url)"
[[ "$remote_url" == "$approved_fetch_url" ]] || fail "GitLab remote URL is not approved"
remote_branch="${upstream_ref#refs/remotes/gitlab/}"
[[ "$remote_branch" =~ ^[A-Za-z0-9._/-]+$ ]] || fail "source upstream branch is invalid"
remote_record="$(/usr/bin/env -i LC_ALL=C HOME="${HOME:-/}" USER="${USER:-}" \
    LOGNAME="${LOGNAME:-}" SSH_AUTH_SOCK="${SSH_AUTH_SOCK:-}" \
    GIT_CONFIG_NOSYSTEM=1 GIT_CONFIG_SYSTEM=/dev/null GIT_CONFIG_GLOBAL=/dev/null \
    GIT_NO_REPLACE_OBJECTS=1 \
    GIT_SSH_COMMAND="$approved_ssh -F /dev/null" GIT_SSH_VARIANT=ssh \
    "$approved_git" ls-remote --exit-code "$approved_fetch_url" \
    "refs/heads/$remote_branch")"
remote_commit="${remote_record%%$'\t'*}"
remote_ref="${remote_record#*$'\t'}"
[[ "$remote_commit" == "$source_commit" \
    && "$remote_ref" == "refs/heads/$remote_branch" ]] \
    || fail "reviewed source commit is not pushed"
[[ -z "$("${git_command[@]}" status --porcelain=v1 --untracked-files=all)" ]] \
    || fail "source checkout is not clean"
final_commit="$("${git_command[@]}" rev-parse --verify 'HEAD^{commit}')"
[[ "$final_commit" == "$source_commit" ]] || fail "source HEAD changed during authentication"
"${git_command[@]}" cat-file -e "$source_commit:$runner_path"

run_sbatch() {
    local scheduler_mode="$1"
    "${git_command[@]}" cat-file blob "$source_commit:$runner_path" \
        | /usr/bin/env -i LC_ALL=C "$approved_sbatch" "$scheduler_mode" \
            --nodes=1 \
            --ntasks=1 \
            --cpus-per-task=32 \
            --job-name=q30t-row-schema-observation \
            --output="$slurm_output" \
            --export=NONE \
            /dev/stdin "$source_path" "$source_commit" "$output"
}

if [[ "$mode" == "--test-only" ]]; then
    run_sbatch --test-only
    exit 0
fi

if [[ "$mode" == "--submit" ]]; then
    run_sbatch --test-only >/dev/null
    run_sbatch --parsable
    exit 0
fi

usage
