#!/usr/bin/env bash
# Task diversity of an annotated search run (cusi.eval.task_diversity): embed its proposed tasks with
# --embedder_model on the first visible GPU, then serve --judge_model with vLLM on every visible GPU and judge each
# task's nearest other tasks for the same functionality. Run from the CUSI root (Slurm wrapper:
# slurm/task_diversity.sh); rerunning continues where it stopped.
#
#   bash scripts/slurm/task_diversity.sh --env <env> --run <run> [--judge_model <value>] [--embedder_model <value>] \
#       [--k <value>] [--vllm_port <value>]
source scripts/utils.sh || { echo "Could not source utils"; exit 1; }
declare -A ARGS
populate_dict TASK_DIVERSITY_DEFAULTS ARGS
REQUIRED_ARGS=()
populate_array TASK_DIVERSITY_ESSENTIALS REQUIRED_ARGS
parse_args ARGS REQUIRED_ARGS "$@"

JOB="${SLURM_JOB_ID:-local$$}"
PORT="${ARGS["vllm_port"]}"
if [[ "$PORT" == "none" ]]; then
    PORT=$(( 8100 + $(echo "$JOB" | tr -cd '0-9' | tail -c 6 | sed 's/^0*//;s/^$/0/') % 800 ))
fi
echo "Node: $(hostname)  GPUs: ${CUDA_VISIBLE_DEVICES:-none}  env=${ARGS["env"]} run=${ARGS["run"]}"
COMMON=(--env "${ARGS["env"]}" --run "${ARGS["run"]}" --embedder_model "${ARGS["embedder_model"]}" --k "${ARGS["k"]}")
python -u run_eval.py task_diversity "${COMMON[@]}" --stage embed || exit 1
bash scripts/serve_vllm.sh --model "${ARGS["judge_model"]}" --port "$PORT" || exit 1
trap "bash scripts/stop_vllm.sh --port $PORT" EXIT
CUDA_VISIBLE_DEVICES="" python -u run_eval.py task_diversity "${COMMON[@]}" --stage judge \
    --judge_model "${ARGS["judge_model"]}" --judge_backend vllm --vllm_base_url "http://127.0.0.1:$PORT/v1/"
