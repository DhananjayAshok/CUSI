#!/usr/bin/env bash
# Pre-exploration tree search (run_search.py, plans/archive/tree_search_plan.md) on one env. Run from the CUSI root
# (Slurm wrapper: slurm/search.sh). Resubmitting the same command resumes the run; a larger --max_expansions
# extends it.
#
# GPUs: the search process needs one only for a cnn / siglip image embedder, the world_model novelty scorer or the
# curiosity expander's policy; it then takes the first visible GPU. With --expander vlm, vLLM serves --model on the
# other visible GPUs (on the only one if just one is visible), or on all of them when the search needs none. GameBoy runs on the node; Web and
# Android in the CUSI container (Android starts one emulator, which needs /dev/kvm).
#
#   bash scripts/slurm/search.sh --env <env> --run <run> --max_expansions <max_expansions> \
#       --image_embedder <image_embedder> --text_embedder <text_embedder> --novelty_scorer <novelty_scorer> \
#       [--expander <value>] [--model <value>] [--vllm_port <value>] \
#       [--extra <value>] [--overwrite <value>] [--ignore_config_violation <value>]
#
# --extra passes further run_search.py flags verbatim, e.g. --extra "--random_steps 20 --scenes viridian".
# --overwrite deletes the run dir first; --ignore_config_violation resumes a run despite a changed config.
source scripts/utils.sh || { echo "Could not source utils"; exit 1; }
declare -A ARGS
populate_dict SEARCH_DEFAULTS ARGS
REQUIRED_ARGS=()
populate_array SEARCH_ESSENTIALS REQUIRED_ARGS
parse_args ARGS REQUIRED_ARGS "$@"

ENV_NAME="${ARGS["env"]}"
MODEL="${ARGS["model"]}"
IS_VLM=false; [[ "${ARGS["expander"]}" == "vlm" ]] && IS_VLM=true
HAS_MODEL=false; [[ "$MODEL" != "none" ]] && HAS_MODEL=true
if [[ "$IS_VLM" != "$HAS_MODEL" ]]; then
    echo "Error: --model goes with --expander vlm, and only with it."; exit 1
fi
JOB="${SLURM_JOB_ID:-local$$}"
PORT="${ARGS["vllm_port"]}"
if [[ "$PORT" == "none" ]]; then
    PORT=$(( 8100 + $(echo "$JOB" | tr -cd '0-9' | tail -c 6 | sed 's/^0*//;s/^$/0/') % 800 ))
fi
echo "Node: $(hostname)  GPUs: ${CUDA_VISIBLE_DEVICES:-none}  env=$ENV_NAME run=${ARGS["run"]} max_expansions=${ARGS["max_expansions"]} model=$MODEL"

FLAGS=(--env "$ENV_NAME" --run_name "${ARGS["run"]}" --max_expansions "${ARGS["max_expansions"]}"
       --image_embedder "${ARGS["image_embedder"]}" --text_embedder "${ARGS["text_embedder"]}"
       --novelty_scorer "${ARGS["novelty_scorer"]}" --expander "${ARGS["expander"]}")
[[ "$MODEL" != "none" ]] && FLAGS+=(--model_name "$MODEL" --model_backend vllm --vllm_base_url "http://127.0.0.1:$PORT/v1/")
[[ "${ARGS["overwrite"]}" == "true" ]] && FLAGS+=(--overwrite)
[[ "${ARGS["ignore_config_violation"]}" == "true" ]] && FLAGS+=(--ignore_config_violation)
EXTRA=()
[[ "${ARGS["extra"]}" != "none" ]] && read -r -a EXTRA <<< "${ARGS["extra"]}"

SEARCH_NEEDS_GPU=false
if [[ "${ARGS["image_embedder"]}" == "cnn" || "${ARGS["image_embedder"]}" == "siglip" \
      || "${ARGS["novelty_scorer"]}" == "world_model" || "${ARGS["expander"]}" == "curiosity" ]]; then
    SEARCH_NEEDS_GPU=true
fi
SEARCH_GPU=""
if [[ -n "${CUDA_VISIBLE_DEVICES:-}" && "$SEARCH_NEEDS_GPU" == "true" ]]; then
    SEARCH_GPU=$(echo "$CUDA_VISIBLE_DEVICES" | cut -d, -f1)
fi
echo "Search process GPU: ${SEARCH_GPU:-none}"
if [[ "$MODEL" != "none" ]]; then
    if [[ -n "$SEARCH_GPU" ]]; then
        VLLM_GPUS=$(echo "$CUDA_VISIBLE_DEVICES" | cut -d, -f2- -s)
        VLLM_GPUS="${VLLM_GPUS:-$CUDA_VISIBLE_DEVICES}"
    else
        VLLM_GPUS="$CUDA_VISIBLE_DEVICES"
    fi
    CUDA_VISIBLE_DEVICES="$VLLM_GPUS" bash scripts/serve_vllm.sh --model "$MODEL" --port "$PORT" || exit 1
    trap "bash scripts/stop_vllm.sh --port $PORT" EXIT
fi
JOB_GPUS="${CUDA_VISIBLE_DEVICES:-}"
export CUDA_VISIBLE_DEVICES="$SEARCH_GPU"
# The container gets the job's GPUs whenever it has any, so NVML can account for them (a vLLM server's included);
# CUDA_VISIBLE_DEVICES still decides what the search process itself uses.
NV=0
[[ -n "$JOB_GPUS" ]] && NV=1

CMD="python -u run_search.py ${FLAGS[*]@Q} ${EXTRA[*]@Q}"
case "$ENV_NAME" in
    gameboy) bash -c "$CMD" ;;
    web) CUSI_CONTAINER_NV=$NV bash scripts/container.sh bash -c "$CMD" ;;
    android)
        CUSI_CONTAINER_NV=$NV bash scripts/container.sh bash -c "
            bash scripts/android_emulator.sh --action start --snapshots true || exit 1
            trap 'bash scripts/android_emulator.sh --action stop' EXIT
            $CMD
        " ;;
    *) echo "Unknown --env $ENV_NAME"; exit 1 ;;
esac
