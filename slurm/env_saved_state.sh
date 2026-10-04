#!/usr/bin/env bash
# Slurm job for scripts/slurm/env_saved_state.sh (all logic lives there). Submit from the CUSI root:
#   sbatch --output="$results_dir/logs/env_saved_state_%j.out" slurm/env_saved_state.sh
# Arguments after the script name are passed through to scripts/slurm/env_saved_state.sh.
#SBATCH --job-name=cusi-env-saved-state
#SBATCH --partition=medium-lg
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00
[[ -f scripts/slurm/env_saved_state.sh ]] || { echo "Submit from the CUSI root"; exit 1; }
bash scripts/slurm/env_saved_state.sh "$@"
