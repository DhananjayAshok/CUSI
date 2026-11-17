#!/usr/bin/env bash
# Seed study for tree world models (run_world_model.py seeds_data / seeds_wm / seeds_compare): one world model per
# tree seed, tested on real task-attempt transitions, policies compared across seeds. Run from the CUSI root (Slurm
# wrapper: slurm/world_model_seeds.sh).
#
# GPUs: seeds_wm uses the first visible GPU (CPU if none); seeds_data and seeds_compare need none. Frames are decoded
# by --workers processes (default: the CPUs Slurm gave the job, else 8).
#
#   bash scripts/slurm/world_model_seeds.sh --env <env> --study <study> --command <data|wm|compare> \
#       [--extra <value>]
#
# --extra passes the command's run_world_model.py flags verbatim, e.g.
#   --command data --extra "--group gemma=<tree>,<tree> --group random=<tree>,<tree>"
#   --command wm --extra "--embedder <model name> --eval_runs <run>,<run> --tag <tag>"
#   --command compare --extra "--tag <tag>"
source scripts/utils.sh || { echo "Could not source utils"; exit 1; }
WORLD_MODEL_SEEDS_ESSENTIALS=("env" "study" "command")
declare -A WORLD_MODEL_SEEDS_DEFAULTS=(
    ["extra"]="none"
    ["workers"]="none"
)
declare -A ARGS
populate_dict WORLD_MODEL_SEEDS_DEFAULTS ARGS
REQUIRED_ARGS=()
populate_array WORLD_MODEL_SEEDS_ESSENTIALS REQUIRED_ARGS
parse_args ARGS REQUIRED_ARGS "$@"

case "${ARGS["command"]}" in
    data|wm|compare) ;;
    *) echo "Error: --command must be data, wm or compare"; exit 1 ;;
esac
if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
    export CUDA_VISIBLE_DEVICES=$(echo "$CUDA_VISIBLE_DEVICES" | cut -d, -f1)
fi
WORKERS="${ARGS["workers"]}"
[[ "$WORKERS" == "none" ]] && WORKERS="${SLURM_CPUS_PER_TASK:-8}"
echo "Node: $(hostname)  GPU: ${CUDA_VISIBLE_DEVICES:-none}  env=${ARGS["env"]} study=${ARGS["study"]} command=${ARGS["command"]}"

EXTRA=()
[[ "${ARGS["extra"]}" != "none" ]] && read -r -a EXTRA <<< "${ARGS["extra"]}"
WORKER_FLAG=(--workers "$WORKERS")
[[ "${ARGS["command"]}" == "compare" ]] && WORKER_FLAG=()
python -u run_world_model.py --env "${ARGS["env"]}" "seeds_${ARGS["command"]}" --study "${ARGS["study"]}" \
    "${WORKER_FLAG[@]}" "${EXTRA[@]}"
