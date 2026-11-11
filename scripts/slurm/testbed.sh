#!/usr/bin/env bash
# The VLM testbed (storage/tmp/testbed/testbed.json: easy and hard test-split tasks of every env): for each --models
# entry, one vLLM server, and the GameBoy, Web and Android evals of the env's final agent run against it in
# parallel (run <run_prefix>_<model short name>); then one server of --judge_model scores every Web run. Run from the
# CUSI root (Slurm wrapper: slurm/testbed.sh).
#
# GPUs: vLLM uses every visible GPU (-tp = visible count unless a model needs another split, see tp_for). GameBoy runs
# on the node; Web and Android in the CUSI container (one emulator: one Android job per node).
#
#   bash scripts/slurm/testbed.sh --models <m1,m2,...> --judge_model <judge_model> [--run_prefix <value>]
#       [--testbed <value>] [--vllm_port <value>] [--overwrite <value>]
#       [--gameboy_max_steps <value>]   # GameBoy emulator step budget per task (default 75; the benchmark's is 175)
source scripts/utils.sh || { echo "Could not source utils"; exit 1; }
declare -A ARGS
populate_dict TESTBED_DEFAULTS ARGS
REQUIRED_ARGS=()
populate_array TESTBED_ESSENTIALS REQUIRED_ARGS
parse_args ARGS REQUIRED_ARGS "$@"

JOB="${SLURM_JOB_ID:-local$$}"
PORT="${ARGS["vllm_port"]}"
[[ "$PORT" == "none" ]] && PORT=$(( 8100 + $(echo "$JOB" | tr -cd '0-9' | tail -c 6 | sed 's/^0*//;s/^$/0/') % 400 ))
URL="http://127.0.0.1:$PORT/v1/"
TESTBED="${ARGS["testbed"]}"
[[ "$TESTBED" == "none" ]] && TESTBED="$storage_dir/tmp/testbed/testbed.json"
OW=(); [[ "${ARGS["overwrite"]}" == "true" ]] && OW=(--overwrite)
tasks() {   # tasks <gb|web|aw>: the testbed's ids for that env, as run_eval takes them
    python - "$TESTBED" "$1" <<'EOF'
import json, sys
ids = [t for t in json.load(open(sys.argv[1])) if t.startswith(sys.argv[2] + ".")]
conv = {"gb": lambda t: "_".join(t.split(".")[1:]), "web": lambda t: t.split(".", 1)[1], "aw": lambda t: t.split(".", 1)[1]}
print(",".join(conv[sys.argv[2]](t) for t in ids))
EOF
}
GB_TASKS=$(tasks gb); WEB_TASKS=$(tasks web); AW_TASKS=$(tasks aw)
echo "Node: $(hostname)  GPUs: ${CUDA_VISIBLE_DEVICES:-none}  port: $PORT  testbed: $TESTBED"
echo "gameboy: $GB_TASKS"; echo "web: $WEB_TASKS"; echo "android: $AW_TASKS"
trap "bash scripts/stop_vllm.sh --port $PORT" EXIT

vllm_extra_for() {   # vLLM flags a model needs beyond the defaults
    case "$1" in
        openbmb/*|OpenGVLab/*|moonshotai/*) echo "--trust-remote-code" ;;
        mistralai/*) echo "--tokenizer-mode mistral --config-format mistral --load-format mistral" ;;
        *) echo "" ;;
    esac
}
short() { echo "${1##*/}" | tr 'A-Z' 'a-z'; }

for MODEL in ${ARGS["models"]//,/ }; do
    RUN="${ARGS["run_prefix"]}_$(short "$MODEL")"
    echo "===== $MODEL -> $RUN ($(date))"
    EXTRA="--no-enable-prefix-caching $(vllm_extra_for "$MODEL")"
    if ! bash scripts/serve_vllm.sh --model "$MODEL" --port "$PORT" --extra "$EXTRA"; then
        echo "===== $MODEL: vLLM failed to start, skipped"; continue
    fi
    F=(--model_name "$MODEL" --model_backend vllm --vllm_base_url "$URL" --run_name "$RUN" "${OW[@]}")
    LOGDIR="$storage_dir/tmp/testbed/logs/$RUN"; mkdir -p "$LOGDIR"
    python -u run_eval.py gameboy "${F[@]}" --tasks "$GB_TASKS" --workers 4 \
        --max_steps "${ARGS["gameboy_max_steps"]}" > "$LOGDIR/gameboy.log" 2>&1 &
    P1=$!
    bash scripts/container.sh python -u run_eval.py web "${F[@]}" --tasks "$WEB_TASKS" --workers 2 \
        --judge_model_name none --judge_model_backend vllm --skip_judge > "$LOGDIR/web.log" 2>&1 &
    P2=$!
    bash scripts/container.sh bash -c "set -u; adb start-server > /dev/null
        bash scripts/android_emulator.sh --action start || exit 1
        trap 'bash scripts/android_emulator.sh --action stop' EXIT
        adb -s emulator-5554 wait-for-device; sleep 30
        python -u run_eval.py android ${F[*]@Q} --tasks ${AW_TASKS@Q}" > "$LOGDIR/android.log" 2>&1 &
    P3=$!
    wait $P1 $P2 $P3
    bash scripts/stop_vllm.sh --port "$PORT"
    echo "===== $MODEL done ($(date))"
done

# Web scores: one judge server for every run
JUDGE="${ARGS["judge_model"]}"
bash scripts/serve_vllm.sh --model "$JUDGE" --port "$PORT" --extra "--no-enable-prefix-caching" || exit 1
for MODEL in ${ARGS["models"]//,/ }; do
    RUN="${ARGS["run_prefix"]}_$(short "$MODEL")"
    python -u run_eval.py web --model_name "$MODEL" --model_backend vllm --vllm_base_url "$URL" --run_name "$RUN" \
        --tasks "$WEB_TASKS" --judge_model_name "$JUDGE" --judge_model_backend vllm --judge_vllm_base_url "$URL" \
        > "$storage_dir/tmp/testbed/logs/$RUN/web_judge.log" 2>&1 || echo "judge failed for $RUN"
done
echo "===== all done ($(date))"
