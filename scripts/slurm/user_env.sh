#!/usr/bin/env bash
# The User's environments (plans/user_sandbox_plan.md): a vLLM server for the Everything Model, one env server per
# env (cusi/env_server; GameBoy on the node, Web and Android in the CUSI container, Android with one emulator), then
# --cmd <cmd> with them up; everything stops when --cmd exits (no --cmd: they stay up until the job ends).
# Run from the CUSI root (Slurm wrapper: slurm/user_env.sh).
#
# GPUs: vLLM uses every visible GPU (-tp = visible count). Android: one emulator (needs /dev/kvm) and adb's default
# server port, so a job with android must not share its node with another Android job (submit with --exclusive).
#
#   bash scripts/slurm/user_env.sh --model <model> --run <run> [--envs <value>] [--kind <value>] [--cmd <cmd>]
#   bash scripts/slurm/user_env.sh ... [--record_dir <record_dir>]   # record every session's steps (off by default)
#   bash scripts/slurm/user_env.sh ... [--vllm_port <value>] [--port_base <value>]
#   bash scripts/slurm/user_env.sh ... [--vllm_extra "<more vllm serve flags>"]   # e.g. "--dtype half" before Ampere
#
# --model none: no vLLM (and Web success unjudged): the servers alone.
# --kind <kind> (dev | test) picks the servers' policy and token: dev uses $storage_dir/env_server/token, test
# $storage_dir/env_server/test_token (each made on first use). Web servers judge success with --model.
# --cmd <cmd> sees: RUN_DIR (storage_dir/env_server/runs/<run>), SERVERS_JSON ({env: url}), TOKEN_FILE, VLLM_URL,
# MODEL. The ports: env server i on --port_base <port_base> + i, default 9000 + 3 * (job id mod 300); vLLM on
# 8100 + (job id mod 400), as scripts/slurm/eval.sh.
source scripts/utils.sh || { echo "Could not source utils"; exit 1; }
declare -A ARGS
populate_dict USER_ENV_DEFAULTS ARGS
REQUIRED_ARGS=()
populate_array USER_ENV_ESSENTIALS REQUIRED_ARGS
parse_args ARGS REQUIRED_ARGS "$@"

MODEL="${ARGS["model"]}"
KIND="${ARGS["kind"]}"
JOB="${SLURM_JOB_ID:-local$$}"
JOBN=$(echo "$JOB" | tr -cd '0-9' | tail -c 6 | sed 's/^0*//;s/^$/0/')
VPORT="${ARGS["vllm_port"]}"; [[ "$VPORT" == "none" ]] && VPORT=$(( 8100 + JOBN % 400 ))
PBASE="${ARGS["port_base"]}"; [[ "$PBASE" == "none" ]] && PBASE=$(( 9000 + 3 * (JOBN % 300) ))
VLLM_URL="http://127.0.0.1:$VPORT/v1/"
RUN_DIR="$storage_dir/env_server/runs/${ARGS["run"]}"
mkdir -p "$RUN_DIR"
TOKEN_FILE="$storage_dir/env_server/token"
[[ "$KIND" == "test" ]] && TOKEN_FILE="$storage_dir/env_server/test_token"
if [[ ! -f "$TOKEN_FILE" ]]; then
    python -c "import secrets; print(secrets.token_hex(24))" > "$TOKEN_FILE"; chmod 600 "$TOKEN_FILE"
fi
echo "Node: $(hostname)  GPUs: ${CUDA_VISIBLE_DEVICES:-none}  model: $MODEL  kind: $KIND  envs: ${ARGS["envs"]}  -> $RUN_DIR"

PIDS=()
cleanup() {
    for p in "${PIDS[@]}"; do kill "$p" 2>/dev/null; done
    sleep 5
    for p in "${PIDS[@]}"; do kill -9 "$p" 2>/dev/null; done
    bash scripts/stop_vllm.sh --port "$VPORT"
}
trap cleanup EXIT

if [[ "$MODEL" != "none" ]]; then
    VEXTRA="--no-enable-prefix-caching"
    [[ "${ARGS["vllm_extra"]}" != "none" ]] && VEXTRA+=" ${ARGS["vllm_extra"]}"
    bash scripts/serve_vllm.sh --model "$MODEL" --port "$VPORT" --extra "$VEXTRA" || exit 1
fi

COMMON=(--kind "$KIND" --token_file "$TOKEN_FILE")
[[ "${ARGS["record_dir"]}" != "none" ]] && COMMON+=(--record_dir "${ARGS["record_dir"]}")
JUDGE=(--judge_model_name "$MODEL" --judge_model_backend vllm --judge_vllm_base_url "$VLLM_URL")
[[ "$MODEL" == "none" ]] && JUDGE=()
SERVERS="{"
i=0
for ENV_NAME in ${ARGS["envs"]//,/ }; do
    PORT=$(( PBASE + i )); i=$((i + 1))
    LOG="$RUN_DIR/server_$ENV_NAME.log"
    case "$ENV_NAME" in
        gameboy)
            python -u run_env_server.py --env gameboy --port "$PORT" "${COMMON[@]}" > "$LOG" 2>&1 & PIDS+=($!) ;;
        web)
            bash scripts/container.sh python -u run_env_server.py --env web --port "$PORT" "${COMMON[@]}" \
                "${JUDGE[@]}" --max_sessions 2 > "$LOG" 2>&1 & PIDS+=($!) ;;
        android)
            CMD="set -u; adb start-server > /dev/null; bash scripts/android_emulator.sh --action start || exit 1"
            CMD+="; trap 'bash scripts/android_emulator.sh --action stop' EXIT; adb -s emulator-5554 wait-for-device"
            CMD+="; sleep 30; python -u run_env_server.py --env android --port $PORT --max_sessions 1 ${COMMON[*]@Q}"
            bash scripts/container.sh bash -c "$CMD" > "$LOG" 2>&1 & PIDS+=($!) ;;
        *) echo "Unknown env $ENV_NAME"; exit 1 ;;
    esac
    [[ "$SERVERS" != "{" ]] && SERVERS+=", "
    SERVERS+="\"$ENV_NAME\": \"http://127.0.0.1:$PORT\""
done
SERVERS+="}"
echo "$SERVERS" > "$RUN_DIR/servers.json"

# wait until every server answers /health (Android: the emulator boots first)
TOKEN=$(cat "$TOKEN_FILE")
for URL in $(python -c "import json, sys; print(' '.join(json.loads(sys.argv[1]).values()))" "$SERVERS"); do
    for _ in $(seq 1 180); do
        curl -sf -H "Authorization: Bearer $TOKEN" "$URL/health" > /dev/null && break
        sleep 10
    done
    curl -sf -H "Authorization: Bearer $TOKEN" "$URL/health" || { echo "server $URL did not come up"; exit 1; }
    echo
done
echo "Servers up: $SERVERS"

export RUN_DIR SERVERS_JSON="$SERVERS" TOKEN_FILE VLLM_URL MODEL
if [[ "${ARGS["cmd"]}" != "none" ]]; then
    bash -c "${ARGS["cmd"]}"
else
    wait
fi
