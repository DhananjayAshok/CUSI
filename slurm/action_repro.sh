#!/usr/bin/env bash
# Slurm job for scripts/slurm/action_repro.sh (all logic lives there). Submit from the CUSI root:
#   sbatch --output="$results_dir/logs/action_repro_%j.out" slurm/action_repro.sh \
#       --model <base model> --lora <adapter dir> --out <json>
#SBATCH --job-name=cusi-action-repro
#SBATCH --partition=medium-lg
#SBATCH --gres=gpu:rtxa6000:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=06:00:00
[[ -f scripts/slurm/action_repro.sh ]] || { echo "Submit from the CUSI root"; exit 1; }
bash scripts/slurm/action_repro.sh "$@"
