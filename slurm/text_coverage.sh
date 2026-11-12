#!/usr/bin/env bash
# Slurm job for scripts/slurm/text_coverage.sh (all logic lives there). Submit from the CUSI root:
#   sbatch --gres=none --output="$results_dir/logs/text_coverage_%j.out" slurm/text_coverage.sh --env <env> --runs <r1,r2>
#   (or --groups "<name>=<r1,r2>;<name>=<r3,r4>")
# Arguments after the script name are passed through to scripts/slurm/text_coverage.sh. CPU only (no GPU).
#SBATCH --job-name=cusi-text-coverage
#SBATCH --partition=medium-lg
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=06:00:00
[[ -f scripts/slurm/text_coverage.sh ]] || { echo "Submit from the CUSI root"; exit 1; }
bash scripts/slurm/text_coverage.sh "$@"
