#!/usr/bin/env bash
# plans/plan.md Part 2: curiosity PPO with the VLM policy on one scene, then (optionally) the world
# model + decoder and curiosity tasks from its replay. Run from the CUSI root
# (Slurm wrapper: slurm/explore.sh).
#
# GPUs: PPO uses the first visible GPU (Qwen3.5-0.8B / Qwen3-VL-2B + LoRA and SigLIP 2 fit on one 48 GB card).
# With --tasks <tasks> vLLM then serves --model <model> the remaining visible GPUs, which must hold it
# (so request 1 + however many vLLM needs). Android needs /dev/kvm on the node.
# gameboy runs on the host; android/web inside the container (GPU via CUSI_CONTAINER_NV=1).
#
#   bash scripts/slurm/explore.sh --env <env> --scene <scene> --run <run> \
#       --policy_model <policy_model> --curiosity_module <curiosity_module> \
#       --image_embedder <image_embedder> --text_embedder <text_embedder> \
#       [--total_steps <value>] [--wm <value>] [--tasks <value>] \
#       [--extra <value>]
source scripts/utils.sh || { echo "Could not source utils"; exit 1; }
declare -A ARGS
populate_dict EXPLORE_DEFAULTS ARGS
REQUIRED_ARGS=()
populate_array EXPLORE_ESSENTIALS REQUIRED_ARGS
parse_args ARGS REQUIRED_ARGS "$@"

ENV="${ARGS["env"]}"
RUN="${ARGS["run"]}"
if [[ "${ARGS["tasks"]}" == "true" && "${ARGS["model"]}" == "none" ]]; then
    echo "Error: --tasks true needs --model (the model vLLM serves for task inference)."; exit 1
fi
EXTRA=""
[[ "${ARGS["extra"]}" != "none" ]] && EXTRA="${ARGS["extra"]}"
echo "Node: $(hostname)  GPUs: ${CUDA_VISIBLE_DEVICES:-none}  env=$ENV scene=${ARGS["scene"]} run=$RUN steps=${ARGS["total_steps"]}"
nvidia-smi --query-gpu=name,memory.used,memory.total --format=csv

PPO_CMD="python -u run_explore.py --env $ENV ppo --scene ${ARGS["scene"]} --run_name $RUN --total_steps ${ARGS["total_steps"]} --policy_model ${ARGS["policy_model"]} --curiosity_module ${ARGS["curiosity_module"]} --image_embedder ${ARGS["image_embedder"]} --text_embedder ${ARGS["text_embedder"]} $EXTRA"
case "$ENV" in
    gameboy) $PPO_CMD || exit 1 ;;
    web) CUSI_CONTAINER_NV=1 bash scripts/container.sh $PPO_CMD || exit 1 ;;
    android)
        CUSI_CONTAINER_NV=1 bash scripts/container.sh bash -c "
            bash scripts/android_emulator.sh --action start --snapshots true || exit 1
            trap 'bash scripts/android_emulator.sh --action stop' EXIT
            $PPO_CMD
        " || exit 1 ;;
    *) echo "Unknown env: $ENV"; exit 1 ;;
esac

if [[ "${ARGS["wm"]}" == "true" ]]; then
    python -u run_explore.py --env "$ENV" world_model --run_names "$RUN" || exit 1
    python -u run_explore.py --env "$ENV" decoder --run_names "$RUN" || exit 1
    python -u run_explore.py --env "$ENV" wm_eval --name "$RUN" || exit 1
fi
if [[ "$ENV" != "gameboy" ]]; then
    python -u run_explore.py --env "$ENV" elements --run_names "$RUN"
fi
if [[ "${ARGS["tasks"]}" == "true" ]]; then
    export CUDA_VISIBLE_DEVICES=$(echo "$CUDA_VISIBLE_DEVICES" | cut -d, -f2-)
    bash scripts/serve_vllm.sh --model "${ARGS["model"]}" || exit 1
    trap "bash scripts/stop_vllm.sh" EXIT
    python -u run_explore.py --env "$ENV" tasks --run_name "$RUN" --model_name "${ARGS["model"]}" || exit 1
fi
