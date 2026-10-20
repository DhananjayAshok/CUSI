#!/usr/bin/env bash
# Training and evaluation on finished exploration / search runs (run_explore.py): the SigLIP / cnn embedder, the
# embedding decoder, the world model and its decoded predictions. Run from the CUSI root (Slurm wrapper:
# slurm/explore_train.sh). No env is started, so it runs on the node (no container).
#
# GPUs: uses the first visible GPU; CPU if none is visible.
#
#   bash scripts/slurm/explore_train.sh --env <env> --command <train_embedder|decoder|world_model|wm_eval> \
#       [--extra <value>]
#
# --extra passes the command's run_explore.py flags verbatim, e.g.
#   --command train_embedder --extra "--embedder siglip --encoder_model <id> --run_names search:a,search:b --steps 2000"
#   --command world_model --extra "--run_names search:a,search:b --image_embedder siglip --encoder_model <id>
#                                  --embedder_load_path <dir>"
source scripts/utils.sh || { echo "Could not source utils"; exit 1; }
declare -A ARGS
populate_dict EXPLORE_TRAIN_DEFAULTS ARGS
REQUIRED_ARGS=()
populate_array EXPLORE_TRAIN_ESSENTIALS REQUIRED_ARGS
parse_args ARGS REQUIRED_ARGS "$@"

case "${ARGS["command"]}" in
    train_embedder|decoder|world_model|wm_eval) ;;
    *) echo "Error: --command must be train_embedder, decoder, world_model or wm_eval"; exit 1 ;;
esac
if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
    export CUDA_VISIBLE_DEVICES=$(echo "$CUDA_VISIBLE_DEVICES" | cut -d, -f1)
fi
echo "Node: $(hostname)  GPU: ${CUDA_VISIBLE_DEVICES:-none}  env=${ARGS["env"]} command=${ARGS["command"]}"

EXTRA=()
[[ "${ARGS["extra"]}" != "none" ]] && read -r -a EXTRA <<< "${ARGS["extra"]}"
python -u run_explore.py --env "${ARGS["env"]}" "${ARGS["command"]}" "${EXTRA[@]}"
