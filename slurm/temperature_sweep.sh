#!/usr/bin/env bash
# Slurm job for scripts/slurm/temperature_sweep.sh (all logic lives there). Submit from the CUSI root:
#   sbatch --output="$results_dir/logs/temperature_sweep_%j.out" slurm/temperature_sweep.sh \
#       --model <model> --out_dir <dir> [--gameboy_runs <dirs>] [--web_runs <dirs>] [--android_runs <dirs>]
#SBATCH --job-name=cusi-temp-sweep
#SBATCH --partition=medium-lg
#SBATCH --gres=gpu:rtxa6000:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=12:00:00
[[ -f scripts/slurm/temperature_sweep.sh ]] || { echo "Submit from the CUSI root"; exit 1; }
bash scripts/slurm/temperature_sweep.sh "$@"
