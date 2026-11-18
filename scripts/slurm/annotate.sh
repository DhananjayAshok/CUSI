#!/usr/bin/env bash
# Annotate a finished search run (cusi.search.annotate): a transition report for every edge that has none, then the
# tasks actually done along the tree's paths, in the form of the benchmark's train tasks. Needs no env: it reads the
# run's frames. vLLM serves --model on every visible GPU. Run from the CUSI root (Slurm wrapper: slurm/annotate.sh);
# rerunning continues where it stopped (reports already written, cached verdicts).
#
#   bash scripts/slurm/annotate.sh --env <env> --run <run> --model <model> [--max_steps <value>] \
#       [--n_per_flavour <value>] [--sample_seed <value>] [--workers <value>] [--vllm_port <value>]
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
echo "Node: $(hostname)  GPUs: ${CUDA_VISIBLE_DEVICES:-none}  env=${ARGS["env"]} run=${ARGS["run"]} model=${ARGS["model"]}"
bash scripts/serve_vllm.sh --model "${ARGS["model"]}" --port "$PORT" || exit 1
trap "bash scripts/stop_vllm.sh --port $PORT" EXIT
CUDA_VISIBLE_DEVICES="" python -u -m cusi.search.annotate --env "${ARGS["env"]}" --run "${ARGS["run"]}" \
    --model_name "${ARGS["model"]}" --model_backend vllm --vllm_base_url "http://127.0.0.1:$PORT/v1/" \
    --max_steps "${ARGS["max_steps"]}" --n_per_flavour "${ARGS["n_per_flavour"]}" \
    --sample_seed "${ARGS["sample_seed"]}" --workers "${ARGS["workers"]}"
