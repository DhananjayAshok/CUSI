#!/usr/bin/env bash
# Slurm job for scripts/slurm/curiosity_parity.sh (all logic lives there). Submit from the CUSI root:
#   sbatch --output="$results_dir/logs/curiosity_parity_%j.out" slurm/curiosity_parity.sh \
#       --policy_text_encoder <policy_text_encoder>
# Arguments after the script name are passed through to scripts/slurm/curiosity_parity.sh.
#SBATCH --job-name=cusi-curiosity-parity
#SBATCH --partition=medium-lg
#SBATCH --gres=gpu:rtxa6000:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=08:00:00
[[ -f scripts/slurm/curiosity_parity.sh ]] || { echo "Submit from the CUSI root"; exit 1; }
bash scripts/slurm/curiosity_parity.sh "$@"
