#!/usr/bin/env bash
# Slurm job for scripts/slurm/explore.sh (all logic lives there). Submit from the CUSI root:
#   sbatch --output="$results_dir/logs/explore_%j.out" slurm/explore.sh --env <env> --scene <scene> --run <run> \
#       --policy_model <policy_model> --curiosity_module <curiosity_module> --image_embedder <image_embedder> \
#       --text_embedder <text_embedder> [--extra <value>] [--wm <value>]
#   (--tasks <tasks> --model <id> also needs GPUs for vLLM: submit with --gres=gpu:rtxa6000:3)
# Arguments after the script name are passed through to scripts/slurm/explore.sh.
#SBATCH --job-name=cusi-explore
#SBATCH --partition=medium-lg
#SBATCH --gres=gpu:rtxa6000:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=23:00:00
[[ -f scripts/slurm/explore.sh ]] || { echo "Submit from the CUSI root"; exit 1; }
bash scripts/slurm/explore.sh "$@"
