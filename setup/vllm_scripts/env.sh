# Sourced by serve_vllm.sh; returns non-zero if the install is not usable.
# Source of truth: <project>/setup/vllm_scripts/ (copied here by setup/move_vllm_scripts.sh).
#
# Expects, in this directory, a .venv (or a symlink to one) with vLLM installed; weights come
# from the caller's $HF_HOME.
VLLM_ROOT="$(cd "$(dirname "$(realpath "${BASH_SOURCE[0]}")")" && pwd)"
[[ -n "${HF_HOME:-}" ]] || { echo "ERROR: HF_HOME is not set; set it to your Hugging Face cache."; return 1; }
if [[ ! -x "$VLLM_ROOT/.venv/bin/python" ]]; then
    echo "ERROR: no venv at $VLLM_ROOT/.venv. Create one there, or symlink .venv to a venv that has vLLM."
    return 1
fi
if [[ ! -x "$VLLM_ROOT/.venv/bin/vllm" ]]; then
    echo "ERROR: $VLLM_ROOT/.venv has no vLLM. Install it: uv pip install --python $VLLM_ROOT/.venv/bin/python vllm"
    return 1
fi
# FlashInfer's top-k/top-p sampler is JIT-compiled at startup and needs a CUDA toolkit
# matching its bundled headers; the nodes have none (and the pip nvcc in .venv is
# incompatible with them), so use vLLM's PyTorch sampler instead.
export VLLM_USE_FLASHINFER_SAMPLER=0
# Keep JIT/compile caches with the shared install rather than in $HOME.
export FLASHINFER_WORKSPACE_BASE="$VLLM_ROOT/cache"
export TRITON_CACHE_DIR="$VLLM_ROOT/cache/triton"
export VLLM_CACHE_ROOT="$VLLM_ROOT/cache/vllm"
