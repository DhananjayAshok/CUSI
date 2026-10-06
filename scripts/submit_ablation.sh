#!/usr/bin/env bash
# Submits the supervisor ablation (agents.md section 7): env x model x supervisor, one Slurm job each
# (slurm/eval.sh), runs named <prefix>_<model>_<supervisor>. Every runner resumes, so rerunning this
# script continues unfinished runs (finished tasks are skipped; a finished Web run only re-judges).
# Run from the CUSI root.
#
#   bash scripts/submit_ablation.sh [--prefix abl] [--time 24:00:00] [--time_min 2:00:00] [--envs "gameboy android web"]
#       [--models "google/gemma-4-26b-a4b-it google/gemma-4-31b-it"] [--supervisors "baseline revision subgoal"]
#       [--judge_model google/gemma-4-31b-it] [--gameboy_workers 8] [--web_workers 4] [--android_emulators 3]
#       [--partition medium-lg]   # e.g. scavenge-lg (6 h limit, preemptible) when medium-lg is full
# GameBoy: pokemon_red, all 49 tasks. Android: the android_world family, 1 instance each, seed 30, on
# an exclusive node (AndroidWorld uses adb's default server port). Web: --task_file cusi (471 tasks),
# judged by --judge_model through the unchanged auto_eval.
source scripts/utils.sh || { echo "Could not source utils"; exit 1; }
declare -A ARGS
ARGS["prefix"]="abl"
ARGS["time"]="24:00:00"
ARGS["time_min"]="2:00:00"   # Slurm may shorten a job's limit to this, e.g. to fit before a maintenance window
ARGS["envs"]="gameboy android web"
ARGS["models"]="google/gemma-4-26b-a4b-it google/gemma-4-31b-it"
ARGS["supervisors"]="baseline revision subgoal"
ARGS["judge_model"]="google/gemma-4-31b-it"
ARGS["gameboy_workers"]="8"
ARGS["web_workers"]="4"
ARGS["android_emulators"]="3"
ARGS["partition"]="medium-lg"
REQUIRED_ARGS=()
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
