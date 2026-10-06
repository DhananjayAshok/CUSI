#!/usr/bin/env bash
# plans/plan.md Part 2.6 -> Part 1: infer tasks from an exploration run's high-novelty trajectories,
# then guidance -> practice -> clean -> dataset on them (run_practice.py --source <source>
# Run from the CUSI root (Slurm wrapper: slurm/curiosity_practice.sh).
#
# GPUs: vLLM uses every visible GPU (-tp = visible count); they must hold --model.
# Android needs /dev/kvm on the node.
#
#   bash scripts/slurm/curiosity_practice.sh --env <env> --run <run> --model <model> \
#       [--max_groups <value>] [--z_min <value>] [--outlier <value>] [--rescore <value>]
# --z_min <z_min> --outlier <outlier> the outlier thresholds (lower them for short runs); --rescore
# recomputes the replay's intrinsic rewards with the current curiosity code.
source scripts/utils.sh || { echo "Could not source utils"; exit 1; }
declare -A ARGS
populate_dict CURIOSITY_PRACTICE_DEFAULTS ARGS
REQUIRED_ARGS=()
populate_array CURIOSITY_PRACTICE_ESSENTIALS REQUIRED_ARGS
parse_args ARGS REQUIRED_ARGS "$@"

ENV="${ARGS["env"]}"
RUN="${ARGS["run"]}"
MODEL="${ARGS["model"]}"
echo "Node: $(hostname)  GPUs: ${CUDA_VISIBLE_DEVICES:-none}  env=$ENV run=$RUN"
bash scripts/serve_vllm.sh --model "$MODEL" || exit 1
trap "bash scripts/stop_vllm.sh" EXIT
RESCORE_ARGS=()
[[ "${ARGS["rescore"]}" != "none" ]] && RESCORE_ARGS=(--rescore_text_alpha "${ARGS["rescore"]}")
python -u run_explore.py --env "$ENV" tasks --run_name "$RUN" --model_name "$MODEL" --max_groups "${ARGS["max_groups"]}" \
    --z_min "${ARGS["z_min"]}" --outlier_threshold "${ARGS["outlier"]}" "${RESCORE_ARGS[@]}" || exit 1
case "$ENV" in
    gameboy) python -u run_practice.py --env gameboy --model_name "$MODEL" --source curiosity --overwrite all ;;
    web) bash scripts/container.sh python -u run_practice.py --env web --model_name "$MODEL" --source curiosity --overwrite all ;;
    android)
        bash scripts/container.sh bash -c "
            bash scripts/android_emulator.sh --action start --snapshots true || exit 1
            trap 'bash scripts/android_emulator.sh --action stop' EXIT
            python -u run_practice.py --env android --model_name '$MODEL' --source curiosity --overwrite all
        " ;;
    *) echo "Unknown env: $ENV"; exit 1 ;;
esac
python -u scripts/practice_report.py --envs "$ENV" --model_name "$MODEL" --source curiosity --out "$results_dir/curiosity_practice_${ENV}_${RUN}.md"
