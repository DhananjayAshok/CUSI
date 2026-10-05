#!/usr/bin/env bash
# Slurm job for scripts/slurm/practice_small.sh (all logic lives there). Submit from the CUSI root:
#   sbatch --output="$results_dir/logs/practice_small_%j.out" slurm/practice_small.sh --model google/gemma-4-26b-a4b-it [--envs "gameboy web"]
# Arguments after the script name are passed through to scripts/slurm/practice_small.sh.
#SBATCH --job-name=cusi-practice-small
#SBATCH --partition=medium-lg
#SBATCH --gres=gpu:rtxa6000:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=20:00:00
[[ -f scripts/slurm/practice_small.sh ]] || { echo "Submit from the CUSI root"; exit 1; }
bash scripts/slurm/practice_small.sh "$@"
