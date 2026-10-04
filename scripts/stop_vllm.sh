#!/usr/bin/env bash
# Stop the vLLM server scripts/serve_vllm.sh started on this node.
#   bash scripts/stop_vllm.sh [--port <p>]
source scripts/utils.sh || { echo "Could not source utils"; exit 1; }
declare -A ARGS
ARGS["port"]="$vllm_port"
REQUIRED_ARGS=()
parse_args ARGS REQUIRED_ARGS "$@"
bash "$shared_vllm_dir/stop_vllm.sh" "${ARGS["port"]}"
