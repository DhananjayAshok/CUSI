#!/usr/bin/env bash
# Slurm job for scripts/slurm/debug_curiosity.sh (all logic lives there). Submit from the CUSI root:
#   sbatch --output="$results_dir/logs/debug_curiosity_%j.out" slurm/debug_curiosity.sh --env android \
#       --args "human --scene contacts --run_name h1 --actions_file tests/fixtures/android_human_actions.txt \
#       --curiosity_module combinationbuffer --image_embedder random_patch --text_embedder none"
# Arguments after the script name are passed through to scripts/slurm/debug_curiosity.sh.
#SBATCH --job-name=cusi-debug-curiosity
#SBATCH --partition=medium-lg
#SBATCH --gres=gpu:rtxa6000:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=02:00:00
[[ -f scripts/slurm/debug_curiosity.sh ]] || { echo "Submit from the CUSI root"; exit 1; }
bash scripts/slurm/debug_curiosity.sh "$@"
