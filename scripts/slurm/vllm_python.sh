#!/usr/bin/env bash
# Serve --model with vLLM on this job's own port, then run a Python script against it with
# --model_name <model> --vllm_base_url <url> appended to its --args. Run from the CUSI root
# (Slurm wrapper: slurm/vllm_python.sh). Used for model-side analyses (e.g. failure labelling).
#
# GPUs: vLLM uses every visible GPU (-tp = visible count); they must hold --model.
#
#   bash scripts/slurm/vllm_python.sh --model <model> --script <path.py> [--args "<script args>"] [--vllm_port <value>]
source scripts/utils.sh || { echo "Could not source utils"; exit 1; }
declare -A ARGS
populate_dict VLLM_PYTHON_DEFAULTS ARGS
REQUIRED_ARGS=()
populate_array VLLM_PYTHON_ESSENTIALS REQUIRED_ARGS
parse_args ARGS REQUIRED_ARGS "$@"

JOB="${SLURM_JOB_ID:-local$$}"
PORT="${ARGS["vllm_port"]}"
if [[ "$PORT" == "none" ]]; then
    PORT=$(( 8100 + $(echo "$JOB" | tr -cd '0-9' | tail -c 6 | sed 's/^0*//;s/^$/0/') % 400 ))
fi
URL="http://127.0.0.1:$PORT/v1/"
echo "Node: $(hostname)  GPUs: ${CUDA_VISIBLE_DEVICES:-none}  model: ${ARGS["model"]}  port: $PORT"
trap "bash scripts/stop_vllm.sh --port $PORT" EXIT
bash scripts/serve_vllm.sh --model "${ARGS["model"]}" --port "$PORT" --extra "--no-enable-prefix-caching" || exit 1
read -r -a SCRIPT_ARGS <<< "$([[ "${ARGS["args"]}" == "none" ]] || echo "${ARGS["args"]}")"
python -u "${ARGS["script"]}" "${SCRIPT_ARGS[@]}" --model_name "${ARGS["model"]}" --vllm_base_url "$URL"
