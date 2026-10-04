#!/usr/bin/env bash
# Install the vLLM helper scripts (setup/vllm_scripts/: serve_vllm.sh, stop_vllm.sh, env.sh)
# into the shared vLLM directory ($shared_vllm_dir from configs/private_vars.yaml), then check
# that the directory has a .venv with vLLM. Safe to rerun: it overwrites the three scripts
# with this checkout's versions and leaves everything else (.venv, cache/, state/) alone.
#
#   bash setup/move_vllm_scripts.sh        (from the project root, after configs/config.env exists)
source scripts/utils.sh || { echo "Could not source utils"; exit 1; }
declare -A ARGS
REQUIRED_ARGS=()
parse_args ARGS REQUIRED_ARGS "$@"

[[ -n "${shared_vllm_dir:-}" ]] || { echo "ERROR: shared_vllm_dir is not set in configs/private_vars.yaml"; exit 1; }
VLLM_DIR="${shared_vllm_dir%/}"
mkdir -p "$VLLM_DIR" || { echo "ERROR: cannot create $VLLM_DIR"; exit 1; }
for f in serve_vllm.sh stop_vllm.sh env.sh; do
    cp setup/vllm_scripts/"$f" "$VLLM_DIR/$f" || { echo "ERROR: could not copy $f to $VLLM_DIR"; exit 1; }
    echo "copied $f -> $VLLM_DIR/$f"
done

if [[ ! -x "$VLLM_DIR/.venv/bin/python" ]]; then
    echo "WARNING: no venv at $VLLM_DIR/.venv. Either:"
    echo "  - create one there and install vLLM:  uv venv $VLLM_DIR/.venv && uv pip install --python $VLLM_DIR/.venv/bin/python vllm"
    echo "  - or symlink it to an existing venv that has vLLM:  ln -s <venv with vllm> $VLLM_DIR/.venv"
elif [[ ! -x "$VLLM_DIR/.venv/bin/vllm" ]]; then
    echo "WARNING: $VLLM_DIR/.venv exists but has no vLLM. Install it:"
    echo "  uv pip install --python $VLLM_DIR/.venv/bin/python vllm"
    echo "  (or point $VLLM_DIR/.venv at a venv that has vLLM)"
else
    echo "Found vLLM in $VLLM_DIR/.venv: $("$VLLM_DIR/.venv/bin/python" -c 'import importlib.metadata as m; print("vllm", m.version("vllm"))' 2>/dev/null)"
fi
