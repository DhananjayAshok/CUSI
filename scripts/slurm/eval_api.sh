#!/usr/bin/env bash
# One test-set evaluation (run_eval.py <env>) of the env's final agent with an API model (OpenRouter, OpenAI or
# Anthropic) as the agent: no vLLM for the agent, so --stage agent needs no GPU. Web is scored in a second job,
# --stage judge, which serves --judge_model on every visible GPU (vLLM) and runs the judge over the finished episodes
# with the same agent flags (so the run's config matches and any unfinished task is run first). Run from the CUSI
# root (Slurm wrapper: slurm/eval_api.sh). The backend's key (e.g. OPENROUTER_API_KEY) must be in the environment.
#
#   bash scripts/slurm/eval_api.sh --env <env> --model <model> --model_backend <model_backend> --run <run> \
#       [--tasks <task ids, comma-separated or a file>] [--workers <value>] [--n_emulators <value>]
#   bash scripts/slurm/eval_api.sh --env web ... --stage judge --judge_model <judge_model> [--vllm_port <value>]
#
# GameBoy runs on the node; Web and Android in the CUSI container (Chromium; the emulators, which need /dev/kvm).
# Android: K emulators in one container, shard i/K into run <run>_s<i> (K=1: <run>); one adb server per node, so two
# Android jobs must not share a node (submit with --exclusive).
source scripts/utils.sh || { echo "Could not source utils"; exit 1; }
declare -A ARGS
populate_dict EVAL_API_DEFAULTS ARGS
REQUIRED_ARGS=()
populate_array EVAL_API_ESSENTIALS REQUIRED_ARGS
parse_args ARGS REQUIRED_ARGS "$@"

ENV_NAME="${ARGS["env"]}"
MODEL="${ARGS["model"]}"
BACKEND="${ARGS["model_backend"]}"
RUN="${ARGS["run"]}"
STAGE="${ARGS["stage"]}"
JOB="${SLURM_JOB_ID:-local$$}"
case "$BACKEND" in
    openrouter) KEY_VAR=OPENROUTER_API_KEY ;;
    openai) KEY_VAR=OPENAI_API_KEY ;;
    anthropic) KEY_VAR=ANTHROPIC_API_KEY ;;
    *) echo "--model_backend must be an API backend (openrouter, openai, anthropic), not $BACKEND"; exit 1 ;;
esac
[[ -n "${!KEY_VAR:-}" ]] || { echo "$KEY_VAR is not set"; exit 1; }
[[ "$STAGE" == "judge" && "$ENV_NAME" != "web" ]] && { echo "--stage judge is Web only"; exit 1; }
[[ "$STAGE" == "judge" && "${ARGS["judge_model"]}" == "none" ]] && { echo "--stage judge needs --judge_model"; exit 1; }
[[ -n "${CUDA_VISIBLE_DEVICES:-}" ]] && export CUSI_CONTAINER_NV=1
RUN_DIR=$(python -m cusi.utils.paths eval_run --env "$ENV_NAME" --run "$RUN")
mkdir -p "$RUN_DIR"
echo "Node: $(hostname)  GPUs: ${CUDA_VISIBLE_DEVICES:-none}  env: $ENV_NAME  model: $MODEL ($BACKEND)  stage: $STAGE -> $RUN_DIR"

FLAGS=(--model_name "$MODEL" --model_backend "$BACKEND" --run_name "$RUN")
[[ "${ARGS["tasks"]}" != "none" ]] && FLAGS+=(--tasks "${ARGS["tasks"]}")
case "$ENV_NAME" in
    gameboy) FLAGS+=(--workers "${ARGS["workers"]}") ;;
    web) FLAGS+=(--workers "${ARGS["workers"]}" --task_file "${ARGS["task_file"]}") ;;
    android) ;;
    *) echo "Unknown --env $ENV_NAME"; exit 1 ;;
esac

STATUS=0; t0=$(date +%s)
if [[ "$STAGE" == "agent" ]]; then
    case "$ENV_NAME" in
        gameboy)
            python -u run_eval.py gameboy "${FLAGS[@]}"; STATUS=$? ;;
        web)
            bash scripts/container.sh python -u run_eval.py web "${FLAGS[@]}" --judge_model_name none \
                --judge_model_backend vllm --skip_judge; STATUS=$? ;;
        android)
            # as scripts/slurm/eval.sh: emulator i on console 5554+2i, gRPC 8554+i
            K="${ARGS["n_emulators"]}"
            SCRIPT="set -u; adb start-server > /dev/null; STOP=''"
            for i in $(seq 0 $((K - 1))); do
                CP=$((5554 + 2 * i)); GP=$((8554 + i))
                SCRIPT+="; bash scripts/android_emulator.sh --action start --port $CP --grpc_port $GP --snapshots false || exit 1"
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
else
    PORT="${ARGS["vllm_port"]}"
    if [[ "$PORT" == "none" ]]; then
        PORT=$(( 8100 + $(echo "$JOB" | tr -cd '0-9' | tail -c 6 | sed 's/^0*//;s/^$/0/') % 400 ))
    fi
    URL="http://127.0.0.1:$PORT/v1/"
    trap "bash scripts/stop_vllm.sh --port $PORT" EXIT
    bash scripts/serve_vllm.sh --model "${ARGS["judge_model"]}" --port "$PORT" --extra "--no-enable-prefix-caching" || exit 1
    # the judge runs where the base runs' judge ran (on the host); unfinished tasks would need the container's browser
    python -u run_eval.py web "${FLAGS[@]}" --judge_model_name "${ARGS["judge_model"]}" --judge_model_backend vllm \
        --judge_vllm_base_url "$URL"; STATUS=$?
fi
SECONDS_TAKEN=$(( $(date +%s) - t0 ))

python - "$RUN_DIR/timing.jsonl" <<EOF
import json, sys, time
row = {"job": "$JOB", "node": "$(hostname)", "gpus": "${CUDA_VISIBLE_DEVICES:-none}", "env": "$ENV_NAME",
       "model": "$MODEL", "model_backend": "$BACKEND", "stage": "$STAGE", "workers": "${ARGS["workers"]}",
       "n_emulators": "${ARGS["n_emulators"]}", "judge_model": "${ARGS["judge_model"]}", "seconds": $SECONDS_TAKEN,
       "exit_status": $STATUS, "finished": time.strftime("%Y-%m-%d %H:%M:%S")}
with open(sys.argv[1], "a") as f:
    f.write(json.dumps(row) + "\n")
print("timing:", row)
EOF
exit $STATUS
