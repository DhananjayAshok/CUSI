#!/usr/bin/env bash
# Start a vLLM server from the shared venv, wait until it is healthy, and return.
# Source of truth: <project>/setup/vllm_scripts/ (copied here by setup/move_vllm_scripts.sh).
#
#   bash <vllm dir>/serve_vllm.sh <model> [vllm serve args...]
#
# - Needs <vllm dir>/.venv with vLLM and $HF_HOME set (checked by env.sh).
# - Binds to 127.0.0.1 (this node only) unless the caller passes --host explicitly.
# - The shared venv is activated only inside this script's process, so the caller's
#   environment (e.g. a project's own venv) is untouched. Run it with `bash`, not `source`.
# - The server runs under `setsid nohup` in its own process group; stop_vllm.sh kills the
#   whole group (API server + engine workers holding VRAM).
# - -tp defaults to the number of visible GPUs; an explicit -tp/--tensor-parallel-size wins.
# - --port defaults to 8000. Several servers can share a node on different ports (give each
#   its own GPUs via CUDA_VISIBLE_DEVICES).
# - --served-model-name defaults to <model>. Healthy = /health answers on the port AND
#   /v1/models lists that exact name; a server already healthy on the port with that name is
#   reused, one serving a different name is an error.
# - Log + PID: state/<host>_<port>.{log,pid}.
#
# Env: VLLM_START_TIMEOUT (seconds, default 2400; first load of a big model from NAS is slow).
set -uo pipefail
VLLM_ROOT="$(cd "$(dirname "$(realpath "$0")")" && pwd)"
[[ $# -ge 1 ]] || { echo "Usage: bash $0 <model> [vllm serve args...]"; exit 1; }
source "$VLLM_ROOT/env.sh" || exit 1

MODEL="$1"; shift
ARGS=("$@")
PORT=8000
HAS_TP=0
HAS_PORT=0
HAS_HOST=0
HOST_ADDR=127.0.0.1
SERVED_NAME=""
for ((i=0; i<${#ARGS[@]}; i++)); do
    case "${ARGS[i]}" in
        --port) PORT="${ARGS[i+1]}"; HAS_PORT=1 ;;
        --port=*) PORT="${ARGS[i]#--port=}"; HAS_PORT=1 ;;
        --host) HOST_ADDR="${ARGS[i+1]}"; HAS_HOST=1 ;;
        --host=*) HOST_ADDR="${ARGS[i]#--host=}"; HAS_HOST=1 ;;
        --served-model-name) SERVED_NAME="${ARGS[i+1]}" ;;
        --served-model-name=*) SERVED_NAME="${ARGS[i]#--served-model-name=}" ;;
        -tp|--tensor-parallel-size|-tp=*|--tensor-parallel-size=*) HAS_TP=1 ;;
    esac
done
[[ $HAS_PORT -eq 0 ]] && ARGS+=(--port "$PORT")
[[ $HAS_HOST -eq 0 ]] && ARGS+=(--host 127.0.0.1)
[[ -z "$SERVED_NAME" ]] && { SERVED_NAME="$MODEL"; ARGS+=(--served-model-name "$SERVED_NAME"); }
# Where to probe: a wildcard bind is reachable on loopback.
case "$HOST_ADDR" in 0.0.0.0|::|"[::]") HOST_ADDR=127.0.0.1 ;; esac
URL="http://${HOST_ADDR}:${PORT}"

# 0: healthy and serving $SERVED_NAME; 1: not answering; 2: answering with other models.
check_server() {
    curl -s -f "$URL/health" > /dev/null 2>&1 || return 1
    local models
    models=$(curl -s -f "$URL/v1/models" 2>/dev/null) || return 1
    if grep -qF "\"id\":\"$SERVED_NAME\"" <<< "$models" || grep -qF "\"id\": \"$SERVED_NAME\"" <<< "$models"; then
        return 0
    fi
    return 2
}

if [[ $HAS_TP -eq 0 ]]; then
    NUM_GPUS=""
    if [[ -n "${CUDA_VISIBLE_DEVICES:-}" && "${CUDA_VISIBLE_DEVICES}" != "-1" ]]; then
        NUM_GPUS=$(echo "$CUDA_VISIBLE_DEVICES" | tr ',' '\n' | grep -c .)
    elif command -v nvidia-smi > /dev/null 2>&1; then
        NUM_GPUS=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | grep -c .)
    fi
    if [[ -n "$NUM_GPUS" && "$NUM_GPUS" -gt 0 ]]; then
        ARGS+=(-tp "$NUM_GPUS")
        echo "No -tp given; using the visible GPU count: $NUM_GPUS"
    fi
fi

HOST=$(hostname -s)
mkdir -p "$VLLM_ROOT/state"
LOG_FILE="$VLLM_ROOT/state/${HOST}_${PORT}.log"
PID_FILE="$VLLM_ROOT/state/${HOST}_${PORT}.pid"

check_server; rc=$?
if [[ $rc -eq 0 ]]; then
    echo "A server is already healthy on $HOST:$PORT serving '$SERVED_NAME'; not starting another."
    exit 0
elif [[ $rc -eq 2 ]]; then
    echo "ERROR: port $PORT on $HOST already serves other models (not '$SERVED_NAME'):"
    curl -s "$URL/v1/models"; echo
    exit 1
fi

# Activate the shared venv in a subshell only, and launch from there.
(
    source "$VLLM_ROOT/.venv/bin/activate"
    echo "Launching: vllm serve $MODEL ${ARGS[*]}"
    echo "  log: $LOG_FILE"
    setsid nohup vllm serve "$MODEL" "${ARGS[@]}" > "$LOG_FILE" 2>&1 < /dev/null &
    echo $! > "$PID_FILE"
)
VLLM_PID=$(cat "$PID_FILE")

TIMEOUT="${VLLM_START_TIMEOUT:-2400}"
END_TIME=$((SECONDS + TIMEOUT))
while [[ $SECONDS -lt $END_TIME ]]; do
    if ! kill -0 "$VLLM_PID" 2>/dev/null; then
        echo "ERROR: vLLM exited during startup. Last log lines ($LOG_FILE):"
        tail -30 "$LOG_FILE"
        rm -f "$PID_FILE"
        exit 1
    fi
    check_server; rc=$?
    if [[ $rc -eq 2 ]]; then
        echo "ERROR: the server on port $PORT is up but does not list '$SERVED_NAME':"
        curl -s "$URL/v1/models"; echo
        bash "$VLLM_ROOT/stop_vllm.sh" "$PORT"
        exit 1
    fi
    if [[ $rc -eq 0 ]]; then
        echo "vLLM healthy: $MODEL as '$SERVED_NAME' on $HOST:$PORT (pid $VLLM_PID, ${SECONDS}s)"
        echo "Stop with: bash $VLLM_ROOT/stop_vllm.sh $PORT"
        exit 0
    fi
    sleep 5
done

echo "ERROR: timed out after ${TIMEOUT}s waiting for vLLM on port $PORT. Stopping it."
bash "$VLLM_ROOT/stop_vllm.sh" "$PORT"
exit 1
