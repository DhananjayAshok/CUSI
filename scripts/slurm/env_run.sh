#!/usr/bin/env bash
# Run one command where an env lives: GameBoy on the node, Web in the CUSI container, Android in the container with
# one emulator started (needs /dev/kvm). For live checks (e.g. tests.text_boxes_live). Run from the CUSI root
# (Slurm wrapper: slurm/env_run.sh). --nv true exposes the visible GPUs inside the container.
#
#   bash scripts/slurm/env_run.sh --env <env> --cmd <cmd> [--nv <value>]
source scripts/utils.sh || { echo "Could not source utils"; exit 1; }
declare -A ARGS
populate_dict ENV_RUN_DEFAULTS ARGS
REQUIRED_ARGS=()
populate_array ENV_RUN_ESSENTIALS REQUIRED_ARGS
parse_args ARGS REQUIRED_ARGS "$@"

ENV_NAME="${ARGS["env"]}"
CMD="${ARGS["cmd"]}"
NV=0
[[ "${ARGS["nv"]}" == "true" ]] && NV=1
echo "Node: $(hostname)  GPUs: ${CUDA_VISIBLE_DEVICES:-none}  env=$ENV_NAME cmd=$CMD"
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
