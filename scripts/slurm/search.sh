#!/usr/bin/env bash
# Pre-exploration tree search (run_search.py, plans/tree_search_plan.md) on one env. Run from the CUSI root
# (Slurm wrapper: slurm/search.sh). Resubmitting the same command resumes the run; a larger --max_expansions
# extends it.
#
# GPUs: none needed with --image_embedder random_patch and the random expander. The search process uses the first
# visible GPU (SigLIP etc.); with --model <model> (VLM explorer or prior), vLLM serves it on the remaining visible
# GPUs, or on the only one if just one is visible. GameBoy runs on the node; Web and Android in the CUSI container
# (Android starts one emulator, which needs /dev/kvm).
#
#   bash scripts/slurm/search.sh --env <env> --run <run> --max_expansions <max_expansions> \
#       --image_embedder <image_embedder> --text_embedder <text_embedder> --novelty_scorer <novelty_scorer> \
#       [--model <value>] [--expander <value>] [--prior <value>] [--vllm_port <value>] [--extra <value>] \
#       [--overwrite <value>] [--ignore_config_violation <value>]
#
# --extra passes further run_search.py flags verbatim, e.g. --extra "--k_steps 20 --scenes viridian".
# --overwrite deletes the run dir first; --ignore_config_violation resumes a run despite a changed config.
source scripts/utils.sh || { echo "Could not source utils"; exit 1; }
declare -A ARGS
populate_dict SEARCH_DEFAULTS ARGS
REQUIRED_ARGS=()
populate_array SEARCH_ESSENTIALS REQUIRED_ARGS
parse_args ARGS REQUIRED_ARGS "$@"

ENV_NAME="${ARGS["env"]}"
MODEL="${ARGS["model"]}"
if [[ ( "${ARGS["expander"]}" == "vlm" || "${ARGS["prior"]}" == "true" ) && "$MODEL" == "none" ]]; then
    echo "Error: --expander vlm and --prior true need --model."; exit 1
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
[[ "${ARGS["prior"]}" == "true" ]] && FLAGS+=(--prior)
[[ "${ARGS["overwrite"]}" == "true" ]] && FLAGS+=(--overwrite)
[[ "${ARGS["ignore_config_violation"]}" == "true" ]] && FLAGS+=(--ignore_config_violation)
if [[ "$MODEL" != "none" ]]; then
    FLAGS+=(--model_name "$MODEL" --model_backend vllm --vllm_base_url "http://127.0.0.1:$PORT/v1/")
fi
EXTRA=()
[[ "${ARGS["extra"]}" != "none" ]] && read -r -a EXTRA <<< "${ARGS["extra"]}"

SEARCH_GPU=""
if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
    SEARCH_GPU=$(echo "$CUDA_VISIBLE_DEVICES" | cut -d, -f1)
fi
if [[ "$MODEL" != "none" ]]; then
    VLLM_GPUS=$(echo "$CUDA_VISIBLE_DEVICES" | cut -d, -f2- -s)
    VLLM_GPUS="${VLLM_GPUS:-$CUDA_VISIBLE_DEVICES}"
    CUDA_VISIBLE_DEVICES="$VLLM_GPUS" bash scripts/serve_vllm.sh --model "$MODEL" --port "$PORT" || exit 1
    trap "bash scripts/stop_vllm.sh --port $PORT" EXIT
fi
export CUDA_VISIBLE_DEVICES="$SEARCH_GPU"
NV=0
[[ -n "$SEARCH_GPU" ]] && NV=1

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
