#!/usr/bin/env bash
# Slurm job for scripts/slurm/eval.sh (all logic lives there). Submit from the CUSI root:
#   sbatch --output="$results_dir/logs/eval_%j.out" slurm/eval.sh \
#       --env <env> --model <model> --run <run> --supervisor <supervisor>
# Arguments after the script name are passed through to scripts/slurm/eval.sh.
#SBATCH --job-name=cusi-eval
#SBATCH --partition=medium-lg
#SBATCH --gres=gpu:rtxa6000:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=24:00:00
# (medium-lg's limit; every runner resumes, so resubmit the same command to continue a run.)
[[ -f scripts/slurm/eval.sh ]] || { echo "Submit from the CUSI root"; exit 1; }
bash scripts/slurm/eval.sh "$@"
