#!/usr/bin/env bash
# End-to-end check of the benchmarks inside the container, with mock models
# (tests/mock_vllm.py) instead of a real vLLM server:
#   WebVoyager:   one task in headless Chromium, then auto_eval scoring.
#   AndroidWorld: boot the AVD read-only and run one task with m3a_cusi (needs KVM).
# Mock models give fixed answers, so tasks "fail"; the check is that every stage runs.
#
#   bash scripts/container.sh bash setup/verify/e2e.sh [--webvoyager true] [--android_world true]

source scripts/utils.sh || { echo "Could not source utils"; exit 1; }

declare -A ARGS
ARGS["webvoyager"]="true"
ARGS["android_world"]="true"
REQUIRED_ARGS=()

# --- Argument parsing (copy verbatim) ---
ALLOWED_FLAGS=("${REQUIRED_ARGS[@]}" "${!ARGS[@]}")
USAGE_STR="Usage: $0"
for req in "${REQUIRED_ARGS[@]}"; do
    USAGE_STR+=" --$req <value>"
done
for opt in "${!ARGS[@]}"; do
    if [[ ! " ${REQUIRED_ARGS[*]} " =~ " ${opt} " ]]; then
        if [[ -z "${ARGS[$opt]}" ]]; then
            echo "DEFAULT VALUE OF KEY \"$opt\" CANNOT BE BLANK"; exit 1
        fi
        USAGE_STR+=" [--$opt <value> (default: ${ARGS[$opt]})]"
    fi
done
function usage() { echo "$USAGE_STR"; exit 1; }

while [[ $# -gt 0 ]]; do
    case "$1" in
        --*)
            FLAG=${1#--}
            VALID=false
            for allowed in "${ALLOWED_FLAGS[@]}"; do
                if [[ "$FLAG" == "$allowed" ]]; then VALID=true; break; fi
            done
            if [ "$VALID" = false ]; then echo "Error: Unknown flag --$FLAG"; usage; fi
            ARGS["$FLAG"]="$2"; shift 2 ;;
        -h|--help) usage ;;
        *) echo "Unknown argument: $1"; usage ;;
    esac
done

for req in "${REQUIRED_ARGS[@]}"; do
    if [[ -z "${ARGS[$req]}" ]]; then echo "Error: --$req is required."; FAILED=true; fi
done
if [ "$FAILED" = true ]; then usage; fi
# --- End argument parsing ---

echo "Script: $0 Active variables:"
for key in "${!ARGS[@]}"; do
    echo "  -$key = ${ARGS[$key]}"
done

set -euo pipefail
[[ "${CUSI_CONTAINER:-}" == "1" ]] || { echo "Run this inside the container: bash scripts/container.sh bash $0"; exit 1; }
WORK=$(mktemp -d "/tmp/${USER}-cusi-e2e-XXXX")
PIDS=()
trap 'kill "${PIDS[@]}" 2> /dev/null || true' EXIT
echo "Work dir: $WORK"

# start_mock <name> <reply>: start a mock server; sets MOCK_URL.
function start_mock() {
    python -m tests.mock_vllm "$2" > "$WORK/$1.url" 2> "$WORK/$1.log" &
    PIDS+=($!)
    for _ in $(seq 50); do
        [[ -s "$WORK/$1.url" ]] && break
        sleep 0.2
    done
    MOCK_URL=$(head -1 "$WORK/$1.url")
    [[ -n "$MOCK_URL" ]] || { echo "mock server $1 did not start"; cat "$WORK/$1.log"; exit 1; }
}

if [[ "${ARGS["webvoyager"]}" == "true" ]]; then
    echo; echo "=== WebVoyager (headless Chromium)"
    # One reply serves both the agent (Thought/Action) and the judge (SUCCESS).
    start_mock webvoyager "Thought: The page is open.
Action: ANSWER; SUCCESS"
    WV_URL=$MOCK_URL
    mkdir -p "$WORK/wv_downloads"
    echo '{"web_name": "Google Search", "id": "Google Search--0", "ques": "What is the title of this page?", "web": "https://www.example.com/"}' > "$WORK/wv_task.jsonl"
    (cd WebVoyager && python -u run.py --test_file "$WORK/wv_task.jsonl" \
        --model_backend vllm --model_name mock --vllm_base_url "$WV_URL" \
        --output_dir "$WORK/wv_results" --run_subdir e2e --download_dir "$WORK/wv_downloads" \
        --headless --max_iter 3 --fix_box_color \
        --chrome_binary "$CUSI_CHROME_BINARY" --chromedriver "$CUSI_CHROMEDRIVER" \
        --chrome_profile_dir "$WORK/wv_profile")
    TASK_DIR="$WORK/wv_results/e2e/taskGoogle Search--0"
    [[ -f "$TASK_DIR/interact_messages.json" ]] || { echo "FAIL: WebVoyager wrote no trajectory"; ls -la "$TASK_DIR"; exit 1; }
    ls "$TASK_DIR"/screenshot*.png > /dev/null || { echo "FAIL: WebVoyager took no screenshot"; exit 1; }
    (cd WebVoyager/evaluation && python -u auto_eval.py --model_backend vllm --model_name mock-judge \
        --vllm_base_url "$WV_URL" --process_dir "$WORK/wv_results/e2e" --max_attached_imgs 1) > "$WORK/wv_eval.log"
    python -c "import json,sys; s=json.load(open(sys.argv[1])); assert s['scored']==1, s; print('auto_eval scores:', {k: s[k] for k in ('scored','success','unjudged','browser_error')})" "$WORK/wv_results/e2e/scores.json"
    echo "PASS: WebVoyager"
fi

if [[ "${ARGS["android_world"]}" == "true" ]]; then
    echo; echo "=== AndroidWorld (emulator)"
    MARKER="$ANDROID_AVD_HOME/AndroidWorldAvd.avd/cusi_apps_installed"
    [[ -f "$MARKER" ]] || { echo "FAIL: apps not installed on the AVD; run setup/android/app_setup.sh first"; exit 1; }
    start_mock android_world 'Reason: done.
Action: {"action_type": "status", "goal_status": "complete"}'
    AW_URL=$MOCK_URL
    bash scripts/android_emulator.sh --action start --read_only true
    trap 'bash scripts/android_emulator.sh --action stop; kill "${PIDS[@]}" 2> /dev/null || true' EXIT
    python -u android_world/run.py --suite_family=android_world --tasks=ContactsAddContact \
        --n_task_combinations=1 --agent_name=m3a_cusi --model_backend=vllm --model_name=mock \
        --vllm_base_url="$AW_URL" --adb_path="$(command -v adb)" --output_path="$WORK/aw_results" \
        2>&1 | tee "$WORK/aw_run.log" | grep -E "Running task|Task (Successful|Failed)|Finished running|Error"
    ls "$WORK"/aw_results/*/ContactsAddContact*.pkl.gz > /dev/null || { echo "FAIL: AndroidWorld wrote no checkpoint"; exit 1; }
    echo "PASS: AndroidWorld"
fi

echo; echo "E2E OK (outputs in $WORK)"
