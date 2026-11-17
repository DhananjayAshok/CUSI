#!/usr/bin/env bash
# Slurm job for scripts/slurm/world_model_test.sh (all logic lives there). Submit from the CUSI root:
#   sbatch --output="$results_dir/logs/world_model_test_%j.out" slurm/world_model_test.sh \
#       --env <env> --names <name>,... --eval_runs <run>,... --tag <tag>
# Arguments after the script name are passed through to scripts/slurm/world_model_test.sh.
#SBATCH --job-name=cusi-wm-test
#SBATCH --partition=medium-lg
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=6:00:00
[[ -f scripts/slurm/world_model_test.sh ]] || { echo "Submit from the CUSI root"; exit 1; }
bash scripts/slurm/world_model_test.sh "$@"
