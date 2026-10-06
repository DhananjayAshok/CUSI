#!/usr/bin/env bash
# Commit and push the android_world and WebVoyager submodules on their
# current branch (cusi), then commit those submodule pointers (and
# .gitmodules) in CUSI and push it. GameBoyRL is not touched.
# Run from the project root.

source scripts/utils.sh || { echo "Could not source utils"; exit 1; }

declare -A ARGS
populate_dict COMMIT_ALL_DEFAULTS ARGS
REQUIRED_ARGS=()
populate_array COMMIT_ALL_ESSENTIALS REQUIRED_ARGS

# --- Argument parsing (copy verbatim) ---
ALLOWED_FLAGS=("${REQUIRED_ARGS[@]}" "${!ARGS[@]}")
USAGE_STR="Usage: $0"
for req in "${REQUIRED_ARGS[@]}"; do
    USAGE_STR+=" --$req <value>"
done
for opt in "${!ARGS[@]}"; do
    if [[ ! " ${REQUIRED_ARGS[*]} " =~ " ${opt} " ]]; then
        if [[ -z "${ARGS[$opt]}" ]]; then
            echo "DEFAULT VALUE OF KEY \"$opt\" CANNOT BE BLANK"; exit 1
        fi
        USAGE_STR+=" [--$opt <value> (default: ${ARGS[$opt]})]"
    fi
done
function usage() { echo "$USAGE_STR"; exit 1; }

while [[ $# -gt 0 ]]; do
    case "$1" in
        --*)
            FLAG=${1#--}
            VALID=false
            for allowed in "${ALLOWED_FLAGS[@]}"; do
                if [[ "$FLAG" == "$allowed" ]]; then VALID=true; break; fi
            done
            if [ "$VALID" = false ]; then echo "Error: Unknown flag --$FLAG"; usage; fi
            ARGS["$FLAG"]="$2"; shift 2 ;;
        -h|--help) usage ;;
        *) echo "Unknown argument: $1"; usage ;;
    esac
done

for req in "${REQUIRED_ARGS[@]}"; do
    if [[ -z "${ARGS[$req]}" ]]; then echo "Error: --$req is required."; FAILED=true; fi
done
if [ "$FAILED" = true ]; then usage; fi
# --- End argument parsing ---

echo "Script: $0 Active variables:"
for key in "${!ARGS[@]}"; do
    echo "  -$key = ${ARGS[$key]}"
done

if [[ "${ARGS["push"]}" == "true" || "${ARGS["push"]}" == "yes" || "${ARGS["push"]}" == "y" ]]; then
    do_push=true
else
    do_push=false
fi

REPOS=("android_world" "WebVoyager")

# commit_repo <repo_dir>
# Commits all changes in repo_dir (except private_vars.yaml) and pushes its
# current branch. Refuses to run on a detached HEAD.
function commit_repo() {
    local repo="$1"
    echo "=== $repo"
    local branch
    branch=$(git -C "$repo" symbolic-ref --short -q HEAD)
    if [[ -z "$branch" ]]; then
        echo "[error] $repo is on a detached HEAD; check out its branch (cusi) first"
        return 1
    fi
    git -C "$repo" add -A || return 1
    # private_vars.yaml is committed with PLACEHOLDER values only; never push
    # machine-specific paths or credentials.
    local private
    private=$(git -C "$repo" diff --cached --name-only | grep '\(^\|/\)private_vars\.yaml$')
    if [[ -n "$private" ]]; then
        git -C "$repo" reset -q -- $private
        echo "[keep local] $private"
    fi
    if ! git -C "$repo" diff --cached --quiet; then
        git -C "$repo" commit -q -m "${ARGS["message"]}" || return 1
        echo "[commit] $(git -C "$repo" log --oneline -1)"
    else
        echo "[clean] nothing to commit"
    fi
    if [[ "$do_push" == true ]]; then
        git -C "$repo" push -u origin "$branch" || { echo "[error] push failed in $repo"; return 1; }
    fi
}

for repo in "${REPOS[@]}"; do
    commit_repo "$repo" || exit 1
done

# Record the new submodule commits in CUSI. Only these paths are committed,
# so anything else staged (e.g. GameBoyRL) stays staged and uncommitted.
echo "=== CUSI"
cusi_branch=$(git symbolic-ref --short -q HEAD)
if [[ -z "$cusi_branch" ]]; then
    echo "[error] CUSI is on a detached HEAD"; exit 1
fi
if git diff HEAD --quiet -- .gitmodules "${REPOS[@]}"; then
    echo "[clean] nothing to commit"
else
    git commit -q -m "${ARGS["message"]}" -- .gitmodules "${REPOS[@]}" || exit 1
    echo "[commit] $(git log --oneline -1)"
fi
if [[ "$do_push" == true ]]; then
    git push origin "$cusi_branch" || { echo "[error] push failed in CUSI"; exit 1; }
fi
echo "Done."
