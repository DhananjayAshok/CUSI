#!/usr/bin/env bash
# Slurm job for scripts/slurm/screen_coverage.sh (all logic lives there). Submit from the CUSI root:
#   sbatch --output="$results_dir/logs/screen_coverage_%j.out" slurm/screen_coverage.sh --env <env> --runs <r1,r2>
#   (add --gres=none to sbatch for --embedder random_patch, which needs no GPU)
# Arguments after the script name are passed through to scripts/slurm/screen_coverage.sh.
#SBATCH --job-name=cusi-screen-coverage
#SBATCH --partition=medium-lg
#SBATCH --gres=gpu:quadrortx8000:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
# (Resubmitting the same command continues where it stopped.)
[[ -f scripts/slurm/screen_coverage.sh ]] || { echo "Submit from the CUSI root"; exit 1; }
bash scripts/slurm/screen_coverage.sh "$@"
