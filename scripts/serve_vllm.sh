#!/usr/bin/env bash
# Start CUSI's model on the shared vLLM install and return once it is healthy.
# CUSI's venv stays active: the shared vLLM venv is only activated inside the shared script.
#
#   bash scripts/serve_vllm.sh --model <HF id or path> [--served_model_name <value>] [--port <value>] [--tp <value>]
#       [--gpu_memory_utilization <value>] [--extra <value>]
#
# --served_model_name <served_model_name> clients pass as --model_name) defaults to --model.
# Healthy means the server answers on --port <port> lists that served name.
# For a second server on the same node, pass another --port <port> and the same port to the
# client (run_practice.py / run_explore.py --vllm_port), and give each server its own GPUs.
source scripts/utils.sh || { echo "Could not source utils"; exit 1; }
declare -A ARGS
populate_dict SERVE_VLLM_DEFAULTS ARGS
REQUIRED_ARGS=()
populate_array SERVE_VLLM_ESSENTIALS REQUIRED_ARGS
parse_args ARGS REQUIRED_ARGS "$@"

# The scripts in shared_vllm_dir are copies of this repo's setup/vllm_scripts/; warn loudly if stale.
stale=()
for f in serve_vllm.sh stop_vllm.sh env.sh; do
    cmp -s "$PROJECT_ROOT/setup/vllm_scripts/$f" "$shared_vllm_dir/$f" || stale+=("$f")
done
if [[ ${#stale[@]} -gt 0 ]]; then
    {
        echo "################################################################################"
        echo "# WARNING: the vLLM scripts in $shared_vllm_dir are out of date"
        echo "# (differ from $PROJECT_ROOT/setup/vllm_scripts/): ${stale[*]}"
        echo "# Serving with the OLD copies. To update them, run from the project root:"
        echo "#     bash setup/move_vllm_scripts.sh"
        echo "################################################################################"
    } >&2
fi

served_name="${ARGS["served_model_name"]}"
[[ "$served_name" == "none" ]] && served_name="${ARGS["model"]}"
tp_args=()
if [[ "${ARGS["tp"]}" != "none" ]]; then tp_args=(-tp "${ARGS["tp"]}"); fi
# --extra: further `vllm serve` arguments, word-split (e.g. --extra <extra>).
extra_args=()
if [[ "${ARGS["extra"]}" != "none" ]]; then read -r -a extra_args <<< "${ARGS["extra"]}"; fi
bash "$shared_vllm_dir/serve_vllm.sh" "${ARGS["model"]}" --served-model-name "$served_name" \
    --port "${ARGS["port"]}" "${tp_args[@]}" \
    --max-model-len "${ARGS["max_model_len"]}" \
    --limit-mm-per-prompt "{\"image\": ${ARGS["max_images"]}}" \
    --gpu-memory-utilization "${ARGS["gpu_memory_utilization"]}" "${extra_args[@]}"
