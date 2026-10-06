#!/usr/bin/env bash
# One test-set evaluation (run_eval.py <env>) under one supervisor arm, against a vLLM server this
# job starts fresh on its own port (prefix caching off), with the job's timing recorded
# (agents.md section 7). Run from the CUSI root (Slurm wrapper: slurm/eval.sh).
#
# GPUs: vLLM uses every visible GPU (-tp = visible count); they must hold --model (and
# --judge_model). GameBoy runs on the node; Web and Android run in the CUSI container (Chromium;
# the emulator, which needs /dev/kvm). The supervisor uses the executor's model and server.
#
#   bash scripts/slurm/eval.sh --env gameboy --model google/gemma-4-26b-a4b-it --run abl_26b_baseline \
#       [--supervisor baseline|revision|subgoal|...] [--workers 1] [--n_tasks N] [--tasks a,b] [--vllm_port P]
#   bash scripts/slurm/eval.sh --env web ... [--task_file cusi] [--judge_model M]   # no --judge_model: agent only
#   bash scripts/slurm/eval.sh --env android ... [--n_emulators K]   # K emulators, shard i/K -> run <run>_s<i>
#   bash scripts/slurm/eval.sh --env web --model M --run R --judge_only true --judge_model J   # score a finished run
#
# --vllm_port defaults to 8100 + (job id mod 800): serve_vllm.sh reuses a healthy server on its port
# and stop_vllm.sh stops whatever serves there, so two jobs on a node must not share one.
# Timing: <run dir>/timing.jsonl gets one line per job (node, GPUs, vLLM start-up, agent and judge
# wall-clock).
source scripts/utils.sh || { echo "Could not source utils"; exit 1; }
declare -A ARGS
ARGS["supervisor"]="baseline"
ARGS["workers"]="1"
ARGS["n_tasks"]="none"
ARGS["tasks"]="none"
ARGS["task_file"]="cusi"
ARGS["judge_model"]="none"
ARGS["judge_only"]="false"
ARGS["n_emulators"]="1"
ARGS["vllm_port"]="none"
REQUIRED_ARGS=("env" "model" "run")
parse_args ARGS REQUIRED_ARGS "$@"

ENV_NAME="${ARGS["env"]}"
MODEL="${ARGS["model"]}"
RUN="${ARGS["run"]}"
JOB="${SLURM_JOB_ID:-local$$}"
PORT="${ARGS["vllm_port"]}"
if [[ "$PORT" == "none" ]]; then
    PORT=$(( 8100 + $(echo "$JOB" | tr -cd '0-9' | tail -c 6 | sed 's/^0*//;s/^$/0/') % 800 ))
fi
URL="http://127.0.0.1:$PORT/v1/"
RUN_DIR="${storage_dir%/}/eval/$ENV_NAME/$RUN"
mkdir -p "$RUN_DIR"
GPU_NAMES=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | sort | uniq -c | sed 's/^ *//' | paste -sd ';')
echo "Node: $(hostname)  GPUs: ${CUDA_VISIBLE_DEVICES:-none} ($GPU_NAMES)  env: $ENV_NAME  model: $MODEL  supervisor: ${ARGS["supervisor"]}  port: $PORT  -> $RUN_DIR"
trap "bash scripts/stop_vllm.sh --port $PORT" EXIT

serve() {   # serve <model>; prints the start-up seconds
    local t0=$(date +%s)
    bash scripts/serve_vllm.sh --model "$1" --port "$PORT" --extra "--no-enable-prefix-caching" >&2 || return 1
    echo $(( $(date +%s) - t0 ))
}

FLAGS=(--model_name "$MODEL" --model_backend vllm --vllm_base_url "$URL" --run_name "$RUN" --supervisor "${ARGS["supervisor"]}")
[[ "${ARGS["tasks"]}" != "none" ]] && FLAGS+=(--tasks "${ARGS["tasks"]}")
case "$ENV_NAME" in
    gameboy)
        FLAGS+=(--workers "${ARGS["workers"]}")
        [[ "${ARGS["n_tasks"]}" != "none" ]] && FLAGS+=(--n_tasks "${ARGS["n_tasks"]}") ;;
    web)
        FLAGS+=(--workers "${ARGS["workers"]}" --task_file "${ARGS["task_file"]}") ;;
    android) ;;
    *) echo "Unknown --env $ENV_NAME"; exit 1 ;;
esac

STARTUP=0; AGENT=0; JUDGE_STARTUP=0; JUDGE=0; STATUS=0
if [[ "${ARGS["judge_only"]}" != "true" ]]; then
    STARTUP=$(serve "$MODEL") || exit 1
    t0=$(date +%s)
    case "$ENV_NAME" in
        gameboy)
            python -u run_eval.py gameboy "${FLAGS[@]}"; STATUS=$? ;;
        web)
            bash scripts/container.sh python -u run_eval.py web "${FLAGS[@]}" --judge_model_name none \
                --judge_model_backend vllm --skip_judge; STATUS=$? ;;
        android)
            # K emulators in one container (one adb server: AndroidWorld always uses adb's default
            # server port, so two Android jobs must not share a node: submit with --exclusive).
            # Emulator i: console 5554+2i, gRPC 8554+i, shard i/K into run <run>_s<i> (K=1: <run>).
            K="${ARGS["n_emulators"]}"
            SCRIPT="set -u; adb start-server > /dev/null; STOP=''"
            for i in $(seq 0 $((K - 1))); do
                CP=$((5554 + 2 * i)); GP=$((8554 + i))
                SCRIPT+="; bash scripts/android_emulator.sh --action start --port $CP --grpc_port $GP || exit 1"
                SCRIPT+="; STOP=\"\$STOP bash scripts/android_emulator.sh --action stop --port $CP;\""
                SCRIPT+="; trap \"\$STOP\" EXIT; adb -s emulator-$CP wait-for-device"
            done
            SCRIPT+="; sleep 30; PIDS=''"
            for i in $(seq 0 $((K - 1))); do
                CP=$((5554 + 2 * i)); GP=$((8554 + i))
                F=("${FLAGS[@]}")
                if [[ "$K" -gt 1 ]]; then
                    F=("${F[@]/#$RUN/${RUN}_s$i}"); F+=(--shard "$i/$K")
                fi
                F+=(--console_port "$CP" --grpc_port "$GP")
                SCRIPT+="; python -u run_eval.py android ${F[*]@Q} > ${RUN_DIR@Q}/shard$i.log 2>&1 & PIDS=\"\$PIDS \$!\""
            done
            SCRIPT+="; S=0; for p in \$PIDS; do wait \$p || S=1; done; exit \$S"
            bash scripts/container.sh bash -c "$SCRIPT"; STATUS=$? ;;
    esac
    AGENT=$(( $(date +%s) - t0 ))
fi

if [[ "$ENV_NAME" == "web" && "${ARGS["judge_model"]}" != "none" ]]; then
    if [[ "${ARGS["judge_model"]}" != "$MODEL" || "${ARGS["judge_only"]}" == "true" ]]; then
        bash scripts/stop_vllm.sh --port "$PORT"
        JUDGE_STARTUP=$(serve "${ARGS["judge_model"]}") || exit 1
    fi
    t0=$(date +%s)
    python -u run_eval.py web "${FLAGS[@]}" --judge_model_name "${ARGS["judge_model"]}" --judge_model_backend vllm \
        --judge_vllm_base_url "$URL"; STATUS=$?
    JUDGE=$(( $(date +%s) - t0 ))
fi

python - "$RUN_DIR/timing.jsonl" <<EOF
import json, sys, time
row = {"job": "$JOB", "node": "$(hostname)", "gpus": "${CUDA_VISIBLE_DEVICES:-none}", "gpu_names": "$GPU_NAMES",
       "env": "$ENV_NAME", "model": "$MODEL", "supervisor": "${ARGS["supervisor"]}", "workers": "${ARGS["workers"]}",
       "n_emulators": "${ARGS["n_emulators"]}", "judge_model": "${ARGS["judge_model"]}", "vllm_startup_seconds": $STARTUP,
       "agent_seconds": $AGENT, "judge_vllm_startup_seconds": $JUDGE_STARTUP, "judge_seconds": $JUDGE,
       "exit_status": $STATUS, "finished": time.strftime("%Y-%m-%d %H:%M:%S")}
with open(sys.argv[1], "a") as f:
    f.write(json.dumps(row) + "\n")
print("timing:", row)
EOF
exit $STATUS
