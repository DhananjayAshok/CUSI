#!/usr/bin/env bash
# Hold this job's GPUs as one vLLM server for --model and run the eval commands queued for it, so a run can start on
# held GPUs without waiting in the Slurm queue. Run from the CUSI root (Slurm wrapper: slurm/eval_server.sh).
#
# GPUs: vLLM uses every visible GPU (-tp = visible count); they must hold --model.
#
#   bash scripts/slurm/eval_server.sh --model <model> [--queue <dir>] [--vllm_port <value>]
#
# Queue: <dir> (default $tmp_dir/eval_queue/<job id>). Each <name>.sh dropped there is run once, in the background,
# with VLLM_PORT exported (e.g. `bash scripts/slurm/eval.sh ... --vllm_port $VLLM_PORT --keep_vllm true`); it is
# renamed <name>.sh.started and its output goes to <name>.log. A file named `stop` ends the job once every started
# command has finished. Two Android evals must not run in one job (one adb server port per node).
source scripts/utils.sh || { echo "Could not source utils"; exit 1; }
declare -A ARGS
populate_dict EVAL_SERVER_DEFAULTS ARGS
REQUIRED_ARGS=()
populate_array EVAL_SERVER_ESSENTIALS REQUIRED_ARGS
parse_args ARGS REQUIRED_ARGS "$@"

JOB="${SLURM_JOB_ID:-local$$}"
PORT="${ARGS["vllm_port"]}"
if [[ "$PORT" == "none" ]]; then
    PORT=$(( 8100 + $(echo "$JOB" | tr -cd '0-9' | tail -c 6 | sed 's/^0*//;s/^$/0/') % 400 ))
fi
QUEUE="${ARGS["queue"]}"
[[ "$QUEUE" == "none" ]] && QUEUE="$tmp_dir/eval_queue/$JOB"
mkdir -p "$QUEUE"
echo "Node: $(hostname)  GPUs: ${CUDA_VISIBLE_DEVICES:-none}  model: ${ARGS["model"]}  port: $PORT  queue: $QUEUE"
trap "bash scripts/stop_vllm.sh --port $PORT" EXIT
bash scripts/serve_vllm.sh --model "${ARGS["model"]}" --port "$PORT" --extra "--no-enable-prefix-caching" || exit 1
export VLLM_PORT="$PORT"
# eval.sh's Web / Android runs ask the container for the GPUs (CUSI_CONTAINER_NV); the server already holds them.
unset CUDA_VISIBLE_DEVICES

PIDS=()
while true; do
    for f in "$QUEUE"/*.sh; do
        [[ -e "$f" ]] || continue
        mv "$f" "$f.started"
        echo "$(date '+%F %T') start ${f##*/}"
        bash "$f.started" > "${f%.sh}.log" 2>&1 &
        PIDS+=($!)
    done
    if [[ -e "$QUEUE/stop" ]]; then
        echo "$(date '+%F %T') stop requested; waiting for ${#PIDS[@]} command(s)"
        wait
        break
    fi
    sleep 20
done
