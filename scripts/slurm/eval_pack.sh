#!/usr/bin/env bash
# Several evaluations of the final agent (run_eval.py <env>) sharing ONE vLLM server this job starts (prefix
# caching off), run concurrently; then, for Web, each run is judged by the same server. GameBoy and Web only
# (Android needs one adb server per node: use eval.sh). Run from the CUSI root (Slurm wrapper: slurm/eval_pack.sh).
#
# GPUs: vLLM uses every visible GPU (-tp = visible count). Each run uses --workers workers.
# Runs: space-separated "<seed>@<split>" (split names $storage_dir/eval/design/<split>_<env>.txt).
# Run name: final_<split>_r<seed>.
#
#   bash scripts/slurm/eval_pack.sh --env <gameboy|web> --model <model> --arms "<seed>@<split> <seed>@<split> ..."
#       [--workers <value>] [--vllm_port <value>]
#
# Timing: each run's timing.jsonl gets one line with this job's vLLM start-up, its own agent and judge seconds,
# and "packed": the number of arms sharing the server (GPU-hours are split evenly by the tables).
source scripts/utils.sh || { echo "Could not source utils"; exit 1; }
declare -A ARGS
populate_dict EVAL_PACK_DEFAULTS ARGS
REQUIRED_ARGS=()
populate_array EVAL_PACK_ESSENTIALS REQUIRED_ARGS
parse_args ARGS REQUIRED_ARGS "$@"

ENV_NAME="${ARGS["env"]}"
MODEL="${ARGS["model"]}"
JOB="${SLURM_JOB_ID:-local$$}"
PORT="${ARGS["vllm_port"]}"
if [[ "$PORT" == "none" ]]; then
    PORT=$(( 8100 + $(echo "$JOB" | tr -cd '0-9' | tail -c 6 | sed 's/^0*//;s/^$/0/') % 400 ))
fi
URL="http://127.0.0.1:$PORT/v1/"
[[ "$ENV_NAME" == "gameboy" || "$ENV_NAME" == "web" ]] || { echo "eval_pack: gameboy or web only"; exit 1; }
read -r -a ARMS <<< "${ARGS["arms"]}"
N=${#ARMS[@]}
GPU_NAMES=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | sort | uniq -c | sed 's/^ *//' | paste -sd ';')
echo "Node: $(hostname)  GPUs: ${CUDA_VISIBLE_DEVICES:-none} ($GPU_NAMES)  env: $ENV_NAME  arms: $N  port: $PORT"
trap "bash scripts/stop_vllm.sh --port $PORT" EXIT

t0=$(date +%s)
bash scripts/serve_vllm.sh --model "$MODEL" --port "$PORT" --extra "--no-enable-prefix-caching" || exit 1
STARTUP=$(( $(date +%s) - t0 ))

run_arm() {   # run_arm <seed@split>: agent (and Web judge) for one run; writes its timing line
    local spec=$1 seed split run tasks status=0 a0 a1 j1
    IFS=@ read -r seed split <<< "$spec"
    run="final_${split}_r${seed}"
    tasks="${storage_dir%/}/eval/design/${split}_${ENV_NAME}.txt"
    local flags=(--model_name "$MODEL" --model_backend vllm --vllm_base_url "$URL" --run_name "$run"
                 --tasks "$tasks" --workers "${ARGS["workers"]}")
    [[ "$seed" != "0" ]] && flags+=(--seed "$seed")
    local run_dir; run_dir=$(python -m cusi.utils.paths eval_run --env "$ENV_NAME" --run "$run")
    mkdir -p "$run_dir"
    a0=$(date +%s)
    if [[ "$ENV_NAME" == "gameboy" ]]; then
        python -u run_eval.py gameboy "${flags[@]}" > "$run_dir/pack_$JOB.log" 2>&1 || status=1
        a1=$(date +%s); j1=$a1
    else
        flags+=(--task_file cusi)
        bash scripts/container.sh python -u run_eval.py web "${flags[@]}" --judge_model_name none \
            --judge_model_backend vllm --skip_judge > "$run_dir/pack_$JOB.log" 2>&1 || status=1
        a1=$(date +%s)
        python -u run_eval.py web "${flags[@]}" --judge_model_name "$MODEL" --judge_model_backend vllm \
            --judge_vllm_base_url "$URL" >> "$run_dir/pack_$JOB.log" 2>&1 || status=1
        j1=$(date +%s)
    fi
    python - "$run_dir/timing.jsonl" <<EOF
import json, sys, time
row = {"job": "$JOB", "node": "$(hostname)", "gpus": "${CUDA_VISIBLE_DEVICES:-none}", "gpu_names": "$GPU_NAMES",
       "env": "$ENV_NAME", "model": "$MODEL", "seed": "$seed", "split": "$split",
       "workers": "${ARGS["workers"]}", "packed": $N, "vllm_startup_seconds": $STARTUP,
       "agent_seconds": $((a1 - a0)), "judge_vllm_startup_seconds": 0, "judge_seconds": $((j1 - a1)),
       "exit_status": $status, "finished": time.strftime("%Y-%m-%d %H:%M:%S")}
with open(sys.argv[1], "a") as f:
    f.write(json.dumps(row) + "\n")
print("timing:", row)
EOF
    echo "arm $run finished with status $status"
    return $status
}

PIDS=()
for spec in "${ARMS[@]}"; do
    run_arm "$spec" & PIDS+=($!)
done
STATUS=0
for p in "${PIDS[@]}"; do wait "$p" || STATUS=1; done
exit $STATUS
