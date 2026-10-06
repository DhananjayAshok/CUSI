#!/usr/bin/env bash
# Run a command inside the CUSI benchmark container ($storage_dir/containers/cusi.sif)
# with the CUSI venv and config active. Run from the CUSI root:
#
#   bash scripts/container.sh <command> [args...]
#   bash scripts/container.sh python android_world/run.py --agent_name <agent_name> ...
#
# Takes the command verbatim rather than --flags, so it does not use the ARGS template.
#
# Writes only to /tmp, the CUSI root and storage_dir: the container gets its own home
# ($storage_dir/container_home), and Android state (AVDs, adb keys, emulator config)
# lives under $storage_dir/android.

source configs/config.env || { echo "configs/config.env not found"; exit 1; }
[[ -n "${HF_HOME:-}" ]] || { echo "HF_HOME is not set; set it (e.g. in ~/.bashrc) to your Hugging Face cache."; exit 1; }
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
# The host's Hugging Face cache (SigLIP 2, Qwen3-VL for exploration), shared with the container.
mkdir -p "$HF_HOME"
BINDS+=("$HF_HOME")
HF_ARGS=(--env HF_HOME="$HF_HOME")
BIND_ARGS=()
for b in "${BINDS[@]}"; do
    BIND_ARGS+=(--bind "$b")
done

# XDG config and cache go on node-local /tmp, private to this call, not in the NAS home.
# Chromium keeps a locked crash database under $XDG_CONFIG_HOME (even with --user-data-dir)
# and rewrites its fontconfig cache under $XDG_CACHE_HOME on every launch; on the NAS these
# are shared by every job on every node, and one job's Chrome can make all others hang
# at startup. Deleted on exit.
XDG_TMP=$(mktemp -d "/tmp/${USER}-cusi-xdg-XXXXXX")
trap 'rm -rf "$XDG_TMP"' EXIT
mkdir -p "$XDG_TMP/config" "$XDG_TMP/cache"
XDG_ARGS=(--env XDG_CONFIG_HOME="$XDG_TMP/config" --env XDG_CACHE_HOME="$XDG_TMP/cache")

# /dev (including /dev/kvm) and /tmp come from the host by default.
# CUSI_CONTAINER_NV=1 also exposes the host's NVIDIA GPUs (apptainer --nv), for GPU work
# (e.g. exploration PPO) inside the container.
NV_ARGS=()
if [[ "${CUSI_CONTAINER_NV:-0}" == "1" ]]; then NV_ARGS=(--nv); fi
# Not exec'd, so the trap above can clean up; the command's exit code is passed through.
apptainer exec "${NV_ARGS[@]}" "${HF_ARGS[@]}" "${XDG_ARGS[@]}" \
    --home "$CONTAINER_HOME" \
    "${BIND_ARGS[@]}" \
    --env ANDROID_AVD_HOME="$ANDROID_STATE/avd" \
    --env ANDROID_USER_HOME="$ANDROID_STATE/user" \
    --env ANDROID_EMULATOR_HOME="$ANDROID_STATE/user" \
    "$SIF" \
    bash -c 'cd "$1" && shift && source scripts/utils.sh > /dev/null && exec "$@"' _ "$PROJECT_ROOT" "$@"
