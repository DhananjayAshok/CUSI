#!/usr/bin/env bash
# Slurm job for scripts/slurm/eval_server.sh (all logic lives there). Submit from the CUSI root:
#   sbatch --output="$results_dir/logs/eval_server_%j.out" slurm/eval_server.sh --model <model>
# Arguments after the script name are passed through to scripts/slurm/eval_server.sh.
#SBATCH --job-name=cusi-eval-server
#SBATCH --partition=medium-lg
#SBATCH --gres=gpu:rtxa6000:2
#SBATCH --cpus-per-task=32
#SBATCH --mem=160G
#SBATCH --time=24:00:00
# (medium-lg's limit)
[[ -f scripts/slurm/eval_server.sh ]] || { echo "Submit from the CUSI root"; exit 1; }
bash scripts/slurm/eval_server.sh "$@"
