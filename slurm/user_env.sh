#!/usr/bin/env bash
# Slurm job for scripts/slurm/user_env.sh (all logic lives there). Submit from the CUSI root:
#   sbatch --exclusive --output="$results_dir/logs/user_env_%j.out" slurm/user_env.sh \
#       --model <model> --run <run> [--envs gameboy,web,android] [--kind dev|test] [--cmd "<cmd>"]
# (--exclusive when --envs includes android: one adb server per node.)
# Arguments after the script name are passed through to scripts/slurm/user_env.sh.
#SBATCH --job-name=cusi-user-env
#SBATCH --partition=medium-lg
#SBATCH --gres=gpu:rtxa6000:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=24:00:00
[[ -f scripts/slurm/user_env.sh ]] || { echo "Submit from the CUSI root"; exit 1; }
bash scripts/slurm/user_env.sh "$@"
