#!/usr/bin/env bash
# Slurm job for scripts/slurm/env_run.sh (all logic lives there). Submit from the CUSI root:
#   sbatch --output="$results_dir/logs/env_run_%j.out" slurm/env_run.sh --env <env> --cmd <cmd>
# Arguments after the script name are passed through to scripts/slurm/env_run.sh.
#SBATCH --job-name=cusi-env-run
#SBATCH --partition=medium-lg
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
[[ -f scripts/slurm/env_run.sh ]] || { echo "Submit from the CUSI root"; exit 1; }
bash scripts/slurm/env_run.sh "$@"
