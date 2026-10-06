#!/usr/bin/env bash
# Stop the vLLM server scripts/serve_vllm.sh started on this node.
#   bash scripts/stop_vllm.sh [--port <value>]
source scripts/utils.sh || { echo "Could not source utils"; exit 1; }
declare -A ARGS
populate_dict STOP_VLLM_DEFAULTS ARGS
REQUIRED_ARGS=()
populate_array STOP_VLLM_ESSENTIALS REQUIRED_ARGS
parse_args ARGS REQUIRED_ARGS "$@"
bash "$shared_vllm_dir/stop_vllm.sh" "${ARGS["port"]}"
