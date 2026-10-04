#!/usr/bin/env bash
# Slurm job for scripts/slurm/vllm_smoke.sh (all logic lives there). Submit from the CUSI root:
#   sbatch --output="$results_dir/logs/vllm_smoke_%j.out" slurm/vllm_smoke.sh --model google/gemma-4-26b-a4b-it
# Arguments after the script name are passed through to scripts/slurm/vllm_smoke.sh.
#SBATCH --job-name=cusi-vllm-smoke
#SBATCH --partition=medium-lg
#SBATCH --gres=gpu:rtxa6000:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
[[ -f scripts/slurm/vllm_smoke.sh ]] || { echo "Submit from the CUSI root"; exit 1; }
bash scripts/slurm/vllm_smoke.sh "$@"
