#!/usr/bin/env bash
# Slurm job for scripts/slurm/explore_train.sh (all logic lives there). Submit from the CUSI root:
#   sbatch --output="$results_dir/logs/explore_train_%j.out" slurm/explore_train.sh \
#       --env <env> --command <train_embedder|decoder|world_model|wm_eval> --extra "<run_explore.py flags>"
# Arguments after the script name are passed through to scripts/slurm/explore_train.sh.
#SBATCH --job-name=cusi-explore-train
#SBATCH --partition=medium-lg
#SBATCH --gres=gpu:rtxa6000:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
[[ -f scripts/slurm/explore_train.sh ]] || { echo "Submit from the CUSI root"; exit 1; }
bash scripts/slurm/explore_train.sh "$@"
