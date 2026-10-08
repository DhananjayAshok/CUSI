#!/usr/bin/env bash
# Slurm job for scripts/slurm/search_env_test.sh (all logic lives there). Submit from the CUSI root:
#   sbatch --output="$results_dir/logs/search_env_test_%j.out" slurm/search_env_test.sh --env <env>
# Arguments after the script name are passed through to scripts/slurm/search_env_test.sh.
#SBATCH --job-name=cusi-search-env-test
#SBATCH --partition=medium-lg
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
[[ -f scripts/slurm/search_env_test.sh ]] || { echo "Submit from the CUSI root"; exit 1; }
bash scripts/slurm/search_env_test.sh "$@"
