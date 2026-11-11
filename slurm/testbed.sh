#!/usr/bin/env bash
# Slurm job for scripts/slurm/testbed.sh (all logic lives there). Submit from the CUSI root:
#   sbatch --output="$results_dir/logs/testbed_%j.out" slurm/testbed.sh --models <m1,m2> --judge_model <judge_model>
# One Android emulator: never two Android jobs on one node.
# Arguments after the script name are passed through to scripts/slurm/testbed.sh.
#SBATCH --job-name=cusi-testbed
#SBATCH --partition=medium-lg
#SBATCH --gres=gpu:rtxa5000:4
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=24:00:00
[[ -f scripts/slurm/testbed.sh ]] || { echo "Submit from the CUSI root"; exit 1; }
bash scripts/slurm/testbed.sh "$@"
