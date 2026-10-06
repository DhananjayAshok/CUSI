#!/usr/bin/env bash
# plans/plan.md Part 1 small-scale test: start vLLM, then run all six practice stages (2 scenes x 3
# proposed tasks, normal step budgets) for each environment, the environments in parallel
# against the one server. Run from the CUSI root (Slurm wrapper: slurm/practice_small.sh).
#
# GPUs: vLLM uses every visible GPU (-tp = visible count); they must hold --model.
# Android needs /dev/kvm on the node.
#
#   bash scripts/slurm/practice_small.sh --model <model> [--envs <value>]
# Per-environment logs: $results_dir/logs/practice_small_<jobid>_<env>.log
source scripts/utils.sh || { echo "Could not source utils"; exit 1; }
declare -A ARGS
populate_dict PRACTICE_SMALL_DEFAULTS ARGS
REQUIRED_ARGS=()
populate_array PRACTICE_SMALL_ESSENTIALS REQUIRED_ARGS
parse_args ARGS REQUIRED_ARGS "$@"

ENVS="${ARGS["envs"]}"
MODEL="${ARGS["model"]}"
JOB="${SLURM_JOB_ID:-local}"
LOGS="${results_dir%/}/logs"
mkdir -p "$LOGS"
echo "Node: $(hostname)  GPUs: ${CUDA_VISIBLE_DEVICES:-none}  envs: $ENVS"
ls -la /dev/kvm 2>&1

bash scripts/serve_vllm.sh --model "$MODEL" || exit 1
trap "bash scripts/stop_vllm.sh" EXIT

declare -A PIDS
for env in $ENVS; do
    log="$LOGS/practice_small_${JOB}_${env}.log"
    case "$env" in
        gameboy)
            python -u run_practice.py --env gameboy --model_name "$MODEL" all > "$log" 2>&1 & ;;
        web)
            bash scripts/container.sh python -u run_practice.py --env web --model_name "$MODEL" all > "$log" 2>&1 & ;;
        android)
            bash scripts/container.sh bash -c "
                bash scripts/android_emulator.sh --action start --snapshots true || exit 1
                trap 'bash scripts/android_emulator.sh --action stop' EXIT
                python -u run_practice.py --env android --model_name '$MODEL' all
            " > "$log" 2>&1 & ;;
        *) echo "Unknown env: $env"; exit 1 ;;
    esac
    PIDS[$env]=$!
    echo "started $env (pid ${PIDS[$env]}) -> $log"
done

status=0
for env in "${!PIDS[@]}"; do
    wait "${PIDS[$env]}"; rc=$?
    echo "$env finished with exit code $rc"
    [[ $rc -ne 0 ]] && status=1
done

python -u scripts/practice_report.py --envs "$ENVS" --model_name "$MODEL" --out "$results_dir/practice_small_${JOB}.md"
exit $status
