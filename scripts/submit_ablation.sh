#!/usr/bin/env bash
# Submits the supervisor ablation: env x model x supervisor, one slurm/eval.sh job each, runs named
# <prefix>_<model>_<supervisor>. Rerunning continues unfinished runs. Defaults: scripts/utils.sh.
#   bash scripts/submit_ablation.sh --models <models> --judge_model <model> (--help lists the rest)
source scripts/utils.sh || { echo "Could not source utils"; exit 1; }
declare -A ARGS
populate_dict SUBMIT_ABLATION_DEFAULTS ARGS
REQUIRED_ARGS=()
populate_array SUBMIT_ABLATION_ESSENTIALS REQUIRED_ARGS
parse_args ARGS REQUIRED_ARGS "$@"

mkdir -p "${results_dir%/}/logs"
for env in ${ARGS["envs"]}; do
    for model in ${ARGS["models"]}; do
        short=$(echo "${model##*/}" | tr '[:upper:]' '[:lower:]')
        for sup in ${ARGS["supervisors"]}; do
            run="${ARGS["prefix"]}_${short}_${sup}"
            extra=(); sbatch_extra=()
            case "$env" in
                gameboy) extra=(--workers "${ARGS["gameboy_workers"]}") ;;
                web)     extra=(--workers "${ARGS["web_workers"]}" --task_file cusi --judge_model "${ARGS["judge_model"]}") ;;
                android) extra=(--n_emulators "${ARGS["android_emulators"]}"); sbatch_extra=(--exclusive) ;;
            esac
            echo -n "$env $run: "
            sbatch --partition="${ARGS["partition"]}" --time="${ARGS["time"]}" --time-min="${ARGS["time_min"]}" --job-name="abl-$env-$short-$sup" "${sbatch_extra[@]}" \
                --output="${results_dir%/}/logs/eval_%j.out" slurm/eval.sh \
                --env "$env" --model "$model" --run "$run" --supervisor "$sup" "${extra[@]}"
        done
    done
done
