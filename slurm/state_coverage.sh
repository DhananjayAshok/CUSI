#!/usr/bin/env bash
# Slurm job for scripts/slurm/state_coverage.sh (all logic lives there). Submit from the CUSI root:
#   sbatch --gres=none --output="$results_dir/logs/state_coverage_%j.out" slurm/state_coverage.sh --env <env> \
#       --groups "<name>=<r1,r2>;<name>=<r3,r4>"     (or --runs <r1,r2>; --stage counts)
# Arguments after the script name are passed through to scripts/slurm/state_coverage.sh. CPU only (no GPU).
#SBATCH --job-name=cusi-state-coverage
#SBATCH --partition=medium-lg
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=12:00:00
[[ -f scripts/slurm/state_coverage.sh ]] || { echo "Submit from the CUSI root"; exit 1; }
bash scripts/slurm/state_coverage.sh "$@"
