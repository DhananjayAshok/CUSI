#!/usr/bin/env bash
# Slurm job for scripts/slurm/base_disagree.sh (all logic lives there). Submit from the CUSI root:
#   sbatch --output="$results_dir/logs/base_disagree_%j.out" slurm/base_disagree.sh \
#       --model <base model> --dataset_dir <dir> --out <jsonl> [--out_dataset <new dataset dir>]
#SBATCH --job-name=cusi-base-disagree
#SBATCH --partition=medium-lg
#SBATCH --gres=gpu:rtxa6000:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=06:00:00
[[ -f scripts/slurm/base_disagree.sh ]] || { echo "Submit from the CUSI root"; exit 1; }
bash scripts/slurm/base_disagree.sh "$@"
