#!/usr/bin/env bash
# Run a command inside the CUSI benchmark container ($storage_dir/containers/cusi.sif)
# with the CUSI venv and config active. Run from the CUSI root:
#
#   bash scripts/container.sh <command> [args...]
#   bash scripts/container.sh python android_world/run.py --agent_name m3a_cusi ...
#
# Takes the command verbatim rather than --flags, so it does not use the ARGS template.
#
# Writes only to /tmp, the CUSI root and storage_dir: the container gets its own home
# ($storage_dir/container_home), and Android state (AVDs, adb keys, emulator config)
# lives under $storage_dir/android.

source configs/config.env || { echo "configs/config.env not found"; exit 1; }
set -euo pipefail
if [[ $# -eq 0 ]]; then
    echo "Usage: bash $0 <command> [args...]"; exit 1
fi

PROJECT_ROOT=$(pwd)
STORAGE=$(realpath -m "$storage_dir")
SIF="$STORAGE/containers/cusi.sif"
if [[ ! -f "$SIF" ]]; then
    echo "$SIF not found; build it with: bash setup/container/build.sh"; exit 1
fi

CONTAINER_HOME="$STORAGE/container_home"
ANDROID_STATE="$STORAGE/android"
mkdir -p "$CONTAINER_HOME" "$ANDROID_STATE/avd" "$ANDROID_STATE/user"

# The venv's python is a symlink into uv's Python install, which may live outside
# the CUSI root (e.g. under the real $HOME); bind that install so it resolves.
VENV_PYTHON=$(realpath "$PROJECT_ROOT/setup/.venv/bin/python")
PYTHON_INSTALL=$(dirname "$(dirname "$VENV_PYTHON")")

BINDS=("$PROJECT_ROOT" "$STORAGE" "$PYTHON_INSTALL")
BIND_ARGS=()
for b in "${BINDS[@]}"; do
    BIND_ARGS+=(--bind "$b")
done

# /dev (including /dev/kvm) and /tmp come from the host by default.
exec apptainer exec \
    --home "$CONTAINER_HOME" \
    "${BIND_ARGS[@]}" \
    --env ANDROID_AVD_HOME="$ANDROID_STATE/avd" \
    --env ANDROID_USER_HOME="$ANDROID_STATE/user" \
    --env ANDROID_EMULATOR_HOME="$ANDROID_STATE/user" \
    "$SIF" \
    bash -c 'cd "$1" && shift && source scripts/utils.sh > /dev/null && exec "$@"' _ "$PROJECT_ROOT" "$@"
