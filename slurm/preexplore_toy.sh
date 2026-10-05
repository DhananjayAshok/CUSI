#!/usr/bin/env bash
# TOY (skill_discovery.md §1.5): Slurm job for scripts/slurm/preexplore_toy.sh (all logic lives there).
# Submit from the CUSI root:
#   sbatch --output="$results_dir/logs/preexplore_toy_%j.out" slurm/preexplore_toy.sh --env web --scenes arxiv \
#       --run toy_vlm --model Qwen/Qwen3-VL-2B-Instruct --extra "--expander vlm --prior ..."
# Arguments after the script name are passed through to scripts/slurm/preexplore_toy.sh.
#SBATCH --job-name=cusi-preexplore-toy
#SBATCH --partition=medium-lg
#SBATCH --gres=gpu:rtxa6000:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
[[ -f scripts/slurm/preexplore_toy.sh ]] || { echo "Submit from the CUSI root"; exit 1; }
bash scripts/slurm/preexplore_toy.sh "$@"
