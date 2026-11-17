#!/usr/bin/env bash
# Slurm job for scripts/slurm/world_model_seeds.sh (all logic lives there). Submit from the CUSI root:
#   sbatch --output="$results_dir/logs/world_model_seeds_%j.out" slurm/world_model_seeds.sh \
#       --env <env> --study <study> --command <data|wm|compare> --extra "<run_world_model.py flags>"
# The data and compare steps need no GPU: submit them with --gres=gpu:0 (and a CPU partition).
# Arguments after the script name are passed through to scripts/slurm/world_model_seeds.sh.
#SBATCH --job-name=cusi-wm-seeds
#SBATCH --partition=medium-lg
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=23:00:00
[[ -f scripts/slurm/world_model_seeds.sh ]] || { echo "Submit from the CUSI root"; exit 1; }
bash scripts/slurm/world_model_seeds.sh "$@"
