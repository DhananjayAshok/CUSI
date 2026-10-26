#!/usr/bin/env bash
# The practice pipeline (run_practice.py all) on one env: attempt every task of --source (an annotated search run),
# judged with critique-hint retries, then build the
# success / failure datasets. Run from the CUSI root (Slurm wrapper: slurm/practice.sh); resubmitting continues with
# the tasks not yet attempted.
#
# GPUs: vLLM serves --model on every visible GPU. GameBoy runs on the node; Web and Android in the CUSI container
# (Android starts one emulator, which needs /dev/kvm).
#
#   bash scripts/slurm/practice.sh --env <env> --model <model> --source <source> [--max_tasks <value>] \
#       [--vllm_port <value>]
source scripts/utils.sh || { echo "Could not source utils"; exit 1; }
declare -A ARGS
populate_dict PRACTICE_DEFAULTS ARGS
REQUIRED_ARGS=()
populate_array PRACTICE_ESSENTIALS REQUIRED_ARGS
parse_args ARGS REQUIRED_ARGS "$@"

ENV_NAME="${ARGS["env"]}"
MODEL="${ARGS["model"]}"
JOB="${SLURM_JOB_ID:-local$$}"
PORT="${ARGS["vllm_port"]}"
if [[ "$PORT" == "none" ]]; then
    PORT=$(( 8100 + $(echo "$JOB" | tr -cd '0-9' | tail -c 6 | sed 's/^0*//;s/^$/0/') % 800 ))
fi
echo "Node: $(hostname)  GPUs: ${CUDA_VISIBLE_DEVICES:-none}  env=$ENV_NAME source=${ARGS["source"]} model=$MODEL"
bash scripts/serve_vllm.sh --model "$MODEL" --port "$PORT" || exit 1
trap "bash scripts/stop_vllm.sh --port $PORT" EXIT
# The container gets the job's GPUs (NVML accounts for the vLLM server's work); the client itself uses none.
[[ -n "${CUDA_VISIBLE_DEVICES:-}" ]] && export CUSI_CONTAINER_NV=1
export CUDA_VISIBLE_DEVICES=""

FLAGS=(--env "$ENV_NAME" --model_name "$MODEL" --vllm_port "$PORT" --source "${ARGS["source"]}" all)
[[ "${ARGS["max_tasks"]}" != "none" ]] && FLAGS+=(--max_tasks "${ARGS["max_tasks"]}")
CMD="python -u run_practice.py ${FLAGS[*]@Q}"
case "$ENV_NAME" in
    gameboy) bash -c "$CMD" ;;
    web) bash scripts/container.sh bash -c "$CMD" ;;
    android)
        bash scripts/container.sh bash -c "
            bash scripts/android_emulator.sh --action start --snapshots true || exit 1
            trap 'bash scripts/android_emulator.sh --action stop' EXIT
            $CMD
        " ;;
    *) echo "Unknown --env $ENV_NAME"; exit 1 ;;
esac
