#!/usr/bin/env bash
# Tree world model training (run_world_model.py: dataset, SigLIP fine-tune, world model) for one env and model name.
# Run from the CUSI root (Slurm wrapper: slurm/world_model.sh). No env is started, so it runs on the node.
#
# GPUs: uses the first visible GPU; CPU if none is visible. Frames are decoded by --workers processes (default: the
# CPUs Slurm gave the job, else 8).
#
#   bash scripts/slurm/world_model.sh --env <env> --name <name> --command <data|embedder|wm|all> \
#       [--extra <value>]
#
# --extra passes the command's run_world_model.py flags verbatim, e.g.
#   --command all --extra "--runs sat_random_s0,sat_random_s1 --encoder_model <SigLIP 2 id>"
source scripts/utils.sh || { echo "Could not source utils"; exit 1; }
WORLD_MODEL_ESSENTIALS=("env" "name" "command")
declare -A WORLD_MODEL_DEFAULTS=(
    ["extra"]="none"
    ["workers"]="none"
)
declare -A ARGS
populate_dict WORLD_MODEL_DEFAULTS ARGS
REQUIRED_ARGS=()
populate_array WORLD_MODEL_ESSENTIALS REQUIRED_ARGS
parse_args ARGS REQUIRED_ARGS "$@"

case "${ARGS["command"]}" in
    data|embedder|wm|all) ;;
    *) echo "Error: --command must be data, embedder, wm or all"; exit 1 ;;
esac
if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
    export CUDA_VISIBLE_DEVICES=$(echo "$CUDA_VISIBLE_DEVICES" | cut -d, -f1)
fi
WORKERS="${ARGS["workers"]}"
[[ "$WORKERS" == "none" ]] && WORKERS="${SLURM_CPUS_PER_TASK:-8}"
echo "Node: $(hostname)  GPU: ${CUDA_VISIBLE_DEVICES:-none}  env=${ARGS["env"]} name=${ARGS["name"]} command=${ARGS["command"]} workers=$WORKERS"

EXTRA=()
[[ "${ARGS["extra"]}" != "none" ]] && read -r -a EXTRA <<< "${ARGS["extra"]}"
python -u run_world_model.py --env "${ARGS["env"]}" "${ARGS["command"]}" --name "${ARGS["name"]}" \
    --workers "$WORKERS" "${EXTRA[@]}"
