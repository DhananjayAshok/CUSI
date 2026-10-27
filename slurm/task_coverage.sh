#!/usr/bin/env bash
# Slurm job for scripts/slurm/task_coverage.sh (all logic lives there). Submit from the CUSI root:
#   sbatch --output="$results_dir/logs/task_coverage_%j.out" slurm/task_coverage.sh --env <env> --run <run>
# Arguments after the script name are passed through to scripts/slurm/task_coverage.sh.
# The embedder runs on the first visible GPU, then vLLM serves the judge on all of them (Gemma-4-31B needs 2).
#SBATCH --job-name=cusi-task-coverage
#SBATCH --partition=medium-lg
#SBATCH --gres=gpu:rtxa6000:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=6:00:00
# (Resubmitting the same command continues where it stopped.)
[[ -f scripts/slurm/task_coverage.sh ]] || { echo "Submit from the CUSI root"; exit 1; }
bash scripts/slurm/task_coverage.sh "$@"
