#!/usr/bin/env bash
# TOY (skill_discovery.md §1.5): throwaway first-build scaffolding; strip it with cusi_search/toy/,
# run_preexplore_toy.py, slurm/preexplore_toy.sh and tests/*toy*.
#
# Pre-exploration toy search (run_preexplore_toy.py) on one environment. Run from the CUSI root
# (Slurm wrapper: slurm/preexplore_toy.sh).
#
# GPUs: with --model, vLLM serves it on every visible GPU (-tp = visible count); the search itself
# needs no GPU (random_patch / tfidf). Without --model no GPU is used. Android needs /dev/kvm.
# gameboy runs on the host; web/android inside the container.
#
#   bash scripts/slurm/preexplore_toy.sh --env web --scenes arxiv --run toy_vlm --model Qwen/Qwen3-VL-2B-Instruct \
#       --extra "--expander vlm --prior --k_steps 4 --max_expansions 6 --image_embedder random_patch \
#                --text_embedder tfidf --novelty_scorer embedding"
source scripts/utils.sh || { echo "Could not source utils"; exit 1; }
declare -A ARGS
ARGS["model"]="none"
ARGS["extra"]="none"
REQUIRED_ARGS=("env" "scenes" "run")
parse_args ARGS REQUIRED_ARGS "$@"

ENV="${ARGS["env"]}"
EXTRA=""
[[ "${ARGS["extra"]}" != "none" ]] && EXTRA="${ARGS["extra"]}"
echo "Node: $(hostname)  GPUs: ${CUDA_VISIBLE_DEVICES:-none}  env=$ENV scenes=${ARGS["scenes"]} run=${ARGS["run"]}"
MODEL_FLAG=""
if [[ "${ARGS["model"]}" != "none" ]]; then
    bash scripts/serve_vllm.sh --model "${ARGS["model"]}" || exit 1
    trap "bash scripts/stop_vllm.sh" EXIT
    MODEL_FLAG="--model_name ${ARGS["model"]}"
fi

CMD="python -u run_preexplore_toy.py --env $ENV --scenes ${ARGS["scenes"]} --run_name ${ARGS["run"]} $MODEL_FLAG $EXTRA"
case "$ENV" in
    gameboy) $CMD || exit 1 ;;
    web) bash scripts/container.sh $CMD || exit 1 ;;
    android)
        ls -la /dev/kvm
        bash scripts/container.sh bash -c "
            bash scripts/android_emulator.sh --action start --snapshots true || exit 1
            trap 'bash scripts/android_emulator.sh --action stop' EXIT
            $CMD
        " || exit 1 ;;
    *) echo "Unknown env: $ENV"; exit 1 ;;
esac
