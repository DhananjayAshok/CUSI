#!/usr/bin/env bash
# Tree world models tested on real task-attempt transitions (run_world_model.py test): the steps of eval runs'
# episodes. One or more trained models of one env, one after the other. Run from the CUSI root (Slurm wrapper:
# slurm/world_model_test.sh).
#
# GPUs: uses the first visible GPU; CPU if none is visible. Frames are decoded by --workers processes (default: the
# CPUs Slurm gave the job, else 8).
#
#   bash scripts/slurm/world_model_test.sh --env <env> --names <name>,<name>,... --eval_runs <run>,<run>,... \
#       --tag <tag> [--extra <value>]
source scripts/utils.sh || { echo "Could not source utils"; exit 1; }
WORLD_MODEL_TEST_ESSENTIALS=("env" "names" "eval_runs" "tag")
declare -A WORLD_MODEL_TEST_DEFAULTS=(
    ["extra"]="none"
    ["workers"]="none"
)
declare -A ARGS
populate_dict WORLD_MODEL_TEST_DEFAULTS ARGS
REQUIRED_ARGS=()
populate_array WORLD_MODEL_TEST_ESSENTIALS REQUIRED_ARGS
parse_args ARGS REQUIRED_ARGS "$@"

if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
    export CUDA_VISIBLE_DEVICES=$(echo "$CUDA_VISIBLE_DEVICES" | cut -d, -f1)
fi
WORKERS="${ARGS["workers"]}"
[[ "$WORKERS" == "none" ]] && WORKERS="${SLURM_CPUS_PER_TASK:-8}"
echo "Node: $(hostname)  GPU: ${CUDA_VISIBLE_DEVICES:-none}  env=${ARGS["env"]} names=${ARGS["names"]} eval_runs=${ARGS["eval_runs"]}"

EXTRA=()
[[ "${ARGS["extra"]}" != "none" ]] && read -r -a EXTRA <<< "${ARGS["extra"]}"
IFS=, read -r -a NAMES <<< "${ARGS["names"]}"
STATUS=0
for name in "${NAMES[@]}"; do
    python -u run_world_model.py --env "${ARGS["env"]}" test --name "$name" --eval_runs "${ARGS["eval_runs"]}" \
        --tag "${ARGS["tag"]}" --workers "$WORKERS" "${EXTRA[@]}" || STATUS=1
done
exit $STATUS
