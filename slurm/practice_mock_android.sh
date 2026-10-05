#!/usr/bin/env bash
# Slurm job for scripts/slurm/practice_mock_android.sh (all logic lives there). Submit from the CUSI root:
#   sbatch --output="$results_dir/logs/practice_mock_android_%j.out" slurm/practice_mock_android.sh
# Arguments after the script name are passed through to scripts/slurm/practice_mock_android.sh.
#SBATCH --job-name=cusi-mock-android
#SBATCH --partition=medium-lg
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
[[ -f scripts/slurm/practice_mock_android.sh ]] || { echo "Submit from the CUSI root"; exit 1; }
bash scripts/slurm/practice_mock_android.sh "$@"
