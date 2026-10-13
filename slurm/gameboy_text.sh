#!/usr/bin/env bash
# Slurm job for scripts/slurm/gameboy_text.sh (all logic lives there). Submit from the CUSI root:
#   sbatch --output="$results_dir/logs/gameboy_text_%j.out" slurm/gameboy_text.sh --variants <variants> --command <command>
# Arguments after the script name are passed through to scripts/slurm/gameboy_text.sh.
#SBATCH --job-name=cusi-gameboy-text
#SBATCH --partition=medium-lg
#SBATCH --cpus-per-task=24
#SBATCH --mem=64G
#SBATCH --time=04:00:00
[[ -f scripts/slurm/gameboy_text.sh ]] || { echo "Submit from the CUSI root"; exit 1; }
bash scripts/slurm/gameboy_text.sh "$@"
