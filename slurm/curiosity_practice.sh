#!/usr/bin/env bash
# Slurm job for scripts/slurm/curiosity_practice.sh (all logic lives there). Submit from the CUSI root:
#   sbatch --output="$results_dir/logs/curiosity_practice_%j.out" slurm/curiosity_practice.sh --env gameboy --run dev_viridian --model google/gemma-4-26b-a4b-it
# Arguments after the script name are passed through to scripts/slurm/curiosity_practice.sh.
#SBATCH --job-name=cusi-curiosity-practice
#SBATCH --partition=medium-lg
#SBATCH --gres=gpu:rtxa6000:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=12:00:00
[[ -f scripts/slurm/curiosity_practice.sh ]] || { echo "Submit from the CUSI root"; exit 1; }
bash scripts/slurm/curiosity_practice.sh "$@"
