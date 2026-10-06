#!/usr/bin/env bash
# Slurm job for scripts/slurm/policy_model_check.sh (all logic lives there). Submit from the CUSI root:
#   sbatch --output="$results_dir/logs/policy_model_check_%j.out" slurm/policy_model_check.sh --models <models>
# Arguments after the script name are passed through to scripts/slurm/policy_model_check.sh.
#SBATCH --job-name=cusi-policy-check
#SBATCH --partition=medium-lg
#SBATCH --gres=gpu:rtxa6000:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=01:00:00
[[ -f scripts/slurm/policy_model_check.sh ]] || { echo "Submit from the CUSI root"; exit 1; }
bash scripts/slurm/policy_model_check.sh "$@"
