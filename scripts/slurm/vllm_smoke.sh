#!/usr/bin/env bash
# Part 0 verification: start the shared vLLM server with CUSI's model, run tests/vllm_smoke.py,
# stop the server. Run from the CUSI root (Slurm wrapper: slurm/vllm_smoke.sh).
#
# GPUs: vLLM uses every visible GPU (-tp = visible count); they must hold --model.
#
#   bash scripts/slurm/vllm_smoke.sh --model google/gemma-4-26b-a4b-it
source scripts/utils.sh || { echo "Could not source utils"; exit 1; }
declare -A ARGS
REQUIRED_ARGS=("model")
parse_args ARGS REQUIRED_ARGS "$@"

echo "Node: $(hostname)  GPUs: ${CUDA_VISIBLE_DEVICES:-none}"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
bash scripts/serve_vllm.sh --model "${ARGS["model"]}" || exit 1
trap "bash scripts/stop_vllm.sh" EXIT
python -u tests/vllm_smoke.py --model_name "${ARGS["model"]}"
