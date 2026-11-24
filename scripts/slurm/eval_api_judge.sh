#!/usr/bin/env bash
# Score a finished Web eval run (run_eval.py web, auto_eval) with an API judge (OpenRouter, OpenAI): no vLLM, no GPU.
# The agent flags must be the ones the run was started with (its config is checked; the judge model is filled into
# it). Run from the CUSI root (Slurm wrapper: slurm/eval_api_judge.sh); the backends' keys must be in the environment.
# A task the agent never finished would be run first, on the host (no container browser): judge finished runs only.
#
#   bash scripts/slurm/eval_api_judge.sh --model <model> --model_backend <model_backend> --run <run> \
#       --judge_model <judge_model> --judge_model_backend <judge_model_backend> [--tasks <ids or file>] [--workers <n>]
#   ... --from_run <agent run>   # judge a copy: <run> is made from <agent run>'s episodes (its verdicts dropped), so
#       # the agent run itself stays unjudged (e.g. for a second judge) and <run>'s name can say which judge scored it
source scripts/utils.sh || { echo "Could not source utils"; exit 1; }
declare -A ARGS
populate_dict EVAL_API_JUDGE_DEFAULTS ARGS
REQUIRED_ARGS=()
populate_array EVAL_API_JUDGE_ESSENTIALS REQUIRED_ARGS
parse_args ARGS REQUIRED_ARGS "$@"

[[ "${ARGS["judge_model_backend"]}" == "vllm" ]] && { echo "vLLM judges go through scripts/slurm/eval.sh"; exit 1; }
RUN="${ARGS["run"]}"
RUN_DIR=$(python -m cusi.utils.paths eval_run --env web --run "$RUN")
if [[ "${ARGS["from_run"]}" != "none" && ! -f "$RUN_DIR/config.json" ]]; then
    SRC_DIR=$(python -m cusi.utils.paths eval_run --env web --run "${ARGS["from_run"]}")
    [[ -f "$SRC_DIR/config.json" ]] || { echo "No Web run at $SRC_DIR"; exit 1; }
    cp -r "${SRC_DIR%/}" "${RUN_DIR%/}" || exit 1
    rm -f "$RUN_DIR/results.jsonl" "$RUN_DIR/scores.json" "$RUN_DIR/auto_eval.log" "$RUN_DIR/timing.jsonl"
    python - "$RUN_DIR/config.json" <<'EOF'
import json, sys
config = json.load(open(sys.argv[1]))
config["judge_model_name"] = "none"     # filled in by this judge pass
json.dump(config, open(sys.argv[1], "w"), indent=1)
EOF
    echo "Copied ${ARGS["from_run"]} -> $RUN"
fi
[[ -f "$RUN_DIR/config.json" ]] || { echo "No Web run at $RUN_DIR"; exit 1; }
echo "Node: $(hostname)  run: $RUN  agent: ${ARGS["model"]} (${ARGS["model_backend"]})  judge: ${ARGS["judge_model"]} (${ARGS["judge_model_backend"]})"
FLAGS=(--model_name "${ARGS["model"]}" --model_backend "${ARGS["model_backend"]}" --run_name "$RUN"
       --workers "${ARGS["workers"]}" --task_file "${ARGS["task_file"]}")
[[ "${ARGS["tasks"]}" != "none" ]] && FLAGS+=(--tasks "${ARGS["tasks"]}")
t0=$(date +%s)
python -u run_eval.py web "${FLAGS[@]}" --judge_model_name "${ARGS["judge_model"]}" \
    --judge_model_backend "${ARGS["judge_model_backend"]}"; STATUS=$?
python - "$RUN_DIR/timing.jsonl" <<EOF
import json, sys, time
row = {"job": "${SLURM_JOB_ID:-local$$}", "node": "$(hostname)", "env": "web", "model": "${ARGS["model"]}",
       "model_backend": "${ARGS["model_backend"]}", "stage": "judge", "judge_model": "${ARGS["judge_model"]}",
       "judge_model_backend": "${ARGS["judge_model_backend"]}", "seconds": $(( $(date +%s) - t0 )),
       "exit_status": $STATUS, "finished": time.strftime("%Y-%m-%d %H:%M:%S")}
with open(sys.argv[1], "a") as f:
    f.write(json.dumps(row) + "\n")
print("timing:", row)
EOF
exit $STATUS
