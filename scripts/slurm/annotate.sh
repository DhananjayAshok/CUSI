#!/usr/bin/env bash
# Annotate finished search runs (cusi.search.annotate): transition reports for the edges the explorer did not report,
# then the tasks actually done along each tree's paths, in the form of the benchmark's train tasks. Needs no env: it
# reads the runs' frames. vLLM serves --model on every visible GPU (--vllm_extra passes flags to it, e.g. "--dtype
# half" on GPUs without bf16); --run is one run or a comma-separated list of runs of --env, annotated --parallel at a
# time against that one server. Run from the CUSI root (Slurm wrapper: slurm/annotate.sh); rerunning continues where
# it stopped (reports already written, cached verdicts, finished runs skipped).
#
#   bash scripts/slurm/annotate.sh --env <env> --run <run>[,<run>...] --model <model> [--parallel <value>] \
#       [--max_steps <value>] [--n_per_flavour <value>] [--sample_seed <value>] [--workers <value>] \
#       [--vllm_extra "<flags>"] [--vllm_port <value>]
source scripts/utils.sh || { echo "Could not source utils"; exit 1; }
declare -A ARGS
populate_dict ANNOTATE_DEFAULTS ARGS
REQUIRED_ARGS=()
populate_array ANNOTATE_ESSENTIALS REQUIRED_ARGS
parse_args ARGS REQUIRED_ARGS "$@"

JOB="${SLURM_JOB_ID:-local$$}"
PORT="${ARGS["vllm_port"]}"
if [[ "$PORT" == "none" ]]; then
    PORT=$(( 8100 + $(echo "$JOB" | tr -cd '0-9' | tail -c 6 | sed 's/^0*//;s/^$/0/') % 800 ))
fi
echo "Node: $(hostname)  GPUs: ${CUDA_VISIBLE_DEVICES:-none}  env=${ARGS["env"]} runs=${ARGS["run"]} model=${ARGS["model"]}"
VEXTRA=(); [[ "${ARGS["vllm_extra"]}" != "none" ]] && VEXTRA=(--extra "${ARGS["vllm_extra"]}")
bash scripts/serve_vllm.sh --model "${ARGS["model"]}" --port "$PORT" "${VEXTRA[@]}" || exit 1
trap "bash scripts/stop_vllm.sh --port $PORT" EXIT

annotate_one() {
    CUDA_VISIBLE_DEVICES="" python -u -m cusi.search.annotate --env "${ARGS["env"]}" --run "$1" \
        --model_name "${ARGS["model"]}" --model_backend vllm --vllm_base_url "http://127.0.0.1:$PORT/v1/" \
        --max_steps "${ARGS["max_steps"]}" --n_per_flavour "${ARGS["n_per_flavour"]}" \
        --sample_seed "${ARGS["sample_seed"]}" --workers "${ARGS["workers"]}" > "$LOG_DIR/annotate_${JOB}_$1.log" 2>&1
    echo "$(date '+%F %T') $1 exit $?"
}
LOG_DIR="$results_dir/logs"
mkdir -p "$LOG_DIR"
IFS=',' read -r -a RUNS <<< "${ARGS["run"]}"
RUNNING=0
for run in "${RUNS[@]}"; do
    echo "$(date '+%F %T') start $run"
    annotate_one "$run" &
    RUNNING=$((RUNNING + 1))
    if (( RUNNING >= ${ARGS["parallel"]} )); then
        wait -n
        RUNNING=$((RUNNING - 1))
    fi
done
wait
echo "$(date '+%F %T') all ${#RUNS[@]} runs done"
