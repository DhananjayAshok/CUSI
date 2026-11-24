#!/usr/bin/env bash
# How hot can --model be sampled before its agent replies break? Serves --model on every visible GPU as the eval
# does (scripts/slurm/eval.sh: prefix caching off, the model's generation config), then run_diagnostics.py
# temperature_sweep replays recorded test-set calls of each env's --<env>_runs at every temperature through that
# env's parsers, one process per env, into <out_dir>/<env>.jsonl (+ .summary.json; resumable). Run from the CUSI root
# (Slurm wrapper: slurm/temperature_sweep.sh).
#
#   bash scripts/slurm/temperature_sweep.sh --model <model> --out_dir <dir> \
#       [--gameboy_runs <run dir,...>] [--web_runs <run dir,...>] [--android_runs <run dir,...>] \
#       [--temperatures <t,t,...>] [--n_per_tag <n>] [--samples <n>]
#   (an env whose runs are "none" is skipped)
#
# The arg tables live here, not in scripts/utils.sh, which pending jobs still source (move them there later).
source scripts/utils.sh || { echo "Could not source utils"; exit 1; }
TEMPERATURE_SWEEP_ESSENTIALS=("model" "out_dir")
declare -A TEMPERATURE_SWEEP_DEFAULTS=(
    ["gameboy_runs"]="none"
    ["web_runs"]="none"
    ["android_runs"]="none"
    ["temperatures"]="0,0.5,1,1.3,1.6,2,2.5,3"
    ["n_per_tag"]="40"
    ["samples"]="2"
)
declare -A ARGS
populate_dict TEMPERATURE_SWEEP_DEFAULTS ARGS
REQUIRED_ARGS=()
populate_array TEMPERATURE_SWEEP_ESSENTIALS REQUIRED_ARGS
parse_args ARGS REQUIRED_ARGS "$@"

MODEL="${ARGS["model"]}"
JOB="${SLURM_JOB_ID:-local$$}"
PORT=$(( 8100 + $(echo "$JOB" | tr -cd '0-9' | tail -c 6 | sed 's/^0*//;s/^$/0/') % 400 ))
mkdir -p "${ARGS["out_dir"]}"
echo "Node: $(hostname)  GPUs: ${CUDA_VISIBLE_DEVICES:-none}  model=$MODEL port=$PORT -> ${ARGS["out_dir"]}"
bash scripts/serve_vllm.sh --model "$MODEL" --port "$PORT" --extra "--no-enable-prefix-caching" || exit 1
trap "bash scripts/stop_vllm.sh --port $PORT" EXIT
STATUS=0
for ENV_NAME in gameboy web android; do
    RUNS="${ARGS["${ENV_NAME}_runs"]}"
    [[ "$RUNS" == "none" ]] && continue
    python -u run_diagnostics.py temperature_sweep --env "$ENV_NAME" --runs "$RUNS" \
        --base_url "http://127.0.0.1:$PORT/v1/" --model "$MODEL" --out "${ARGS["out_dir"]}/$ENV_NAME.jsonl" \
        --temperatures "${ARGS["temperatures"]}" --n_per_tag "${ARGS["n_per_tag"]}" \
        --samples "${ARGS["samples"]}" || STATUS=1
done
exit $STATUS
