#!/usr/bin/env bash
# Slurm job for scripts/slurm/practice.sh (all logic lives there). Submit from the CUSI root:
#   sbatch --output="$results_dir/logs/practice_%j.out" slurm/practice.sh --env <env> --model <model> \
#       --source <annotated search run>
# Arguments after the script name are passed through to scripts/slurm/practice.sh.
# vLLM serves --model on every visible GPU (Gemma-4-31B needs 2).
#SBATCH --job-name=cusi-practice
#SBATCH --partition=medium-lg
#SBATCH --gres=gpu:rtxa6000:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=23:00:00
# (Resubmitting the same command continues with the tasks not yet attempted.)
[[ -f scripts/slurm/practice.sh ]] || { echo "Submit from the CUSI root"; exit 1; }
bash scripts/slurm/practice.sh "$@"
