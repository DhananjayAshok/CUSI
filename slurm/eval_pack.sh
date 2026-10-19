#!/usr/bin/env bash
# Slurm job for scripts/slurm/eval_pack.sh (all logic lives there). Submit from the CUSI root:
#   sbatch --output="$results_dir/logs/eval_%j.out" slurm/eval_pack.sh \
#       --env <gameboy|web> --model <model> --arms "<seed>@<split> ..."
# Arguments after the script name are passed through to scripts/slurm/eval_pack.sh.
#SBATCH --job-name=cusi-eval-pack
#SBATCH --partition=medium-lg
#SBATCH --gres=gpu:rtxa6000:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=24:00:00
[[ -f scripts/slurm/eval_pack.sh ]] || { echo "Submit from the CUSI root"; exit 1; }
bash scripts/slurm/eval_pack.sh "$@"
