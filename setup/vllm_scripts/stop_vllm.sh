#!/usr/bin/env bash
# Stop the vLLM server started by serve_vllm.sh on this host:  bash stop_vllm.sh [port=8000]
# Source of truth: <project>/setup/vllm_scripts/ (copied here by setup/move_vllm_scripts.sh).
# Kills the server's whole process group (API server + engine workers).
VLLM_ROOT="$(cd "$(dirname "$(realpath "$0")")" && pwd)"
PORT="${1:-8000}"
HOST=$(hostname -s)
PID_FILE="$VLLM_ROOT/state/${HOST}_${PORT}.pid"
if [[ ! -f "$PID_FILE" ]]; then
    echo "No PID file for $HOST:$PORT ($PID_FILE); nothing to stop."
    exit 0
fi
PID=$(cat "$PID_FILE")
PGID=$(ps -o pgid= -p "$PID" 2>/dev/null | tr -d ' ')
MYPGID=$(ps -o pgid= -p $$ | tr -d ' ')
if [[ -n "$PGID" && "$PGID" != "$MYPGID" ]]; then
    kill -TERM -"$PGID" 2>/dev/null
    for _ in $(seq 1 20); do kill -0 "$PID" 2>/dev/null || break; sleep 1; done
    kill -KILL -"$PGID" 2>/dev/null
elif [[ -n "$PGID" ]]; then
    kill -TERM "$PID" 2>/dev/null; sleep 5; kill -KILL "$PID" 2>/dev/null
fi
rm -f "$PID_FILE"
echo "Stopped vLLM on $HOST:$PORT (pid $PID)."
