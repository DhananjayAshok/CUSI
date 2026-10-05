#!/usr/bin/env bash
# debug_curiosity.py (curiosity_plan §5) on one env: random actions or a human action list, with any
# curiosity module + embedder. Run from the CUSI root (Slurm wrapper: slurm/debug_curiosity.sh).
#
# GPUs: the first visible GPU if the embedder uses one (siglip / dense); random_patch needs none.
# gameboy runs on the host; android/web inside the container. Android needs /dev/kvm on the node.
#
#   bash scripts/slurm/debug_curiosity.sh --env android --args "human --scene contacts --run_name h1 \
#       --actions_file tests/fixtures/android_human_actions.txt --curiosity_module combinationbuffer \
#       --image_embedder random_patch --text_embedder none"
source scripts/utils.sh || { echo "Could not source utils"; exit 1; }
declare -A ARGS
REQUIRED_ARGS=("env" "args")
parse_args ARGS REQUIRED_ARGS "$@"

ENV="${ARGS["env"]}"
echo "Node: $(hostname)  GPUs: ${CUDA_VISIBLE_DEVICES:-none}  env=$ENV"
CMD="python -u debug_curiosity.py --env $ENV ${ARGS["args"]}"
case "$ENV" in
    gameboy) $CMD ;;
    web) CUSI_CONTAINER_NV=1 bash scripts/container.sh $CMD ;;
    android)
        CUSI_CONTAINER_NV=1 bash scripts/container.sh bash -c "
            bash scripts/android_emulator.sh --action start --snapshots true || exit 1
            trap 'bash scripts/android_emulator.sh --action stop' EXIT
            $CMD
        " ;;
    *) echo "Unknown env: $ENV"; exit 1 ;;
esac
