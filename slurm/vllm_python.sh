#!/usr/bin/env bash
# Slurm job for scripts/slurm/vllm_python.sh (all logic lives there). Submit from the CUSI root:
#   sbatch --output="$results_dir/logs/vllm_python_%j.out" slurm/vllm_python.sh \
#       --model <model> --script <path.py> --args "<script args>"
# Arguments after the script name are passed through to scripts/slurm/vllm_python.sh.
#SBATCH --job-name=cusi-vllm-python
#SBATCH --partition=medium-lg
#SBATCH --gres=gpu:rtxa6000:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00
[[ -f scripts/slurm/vllm_python.sh ]] || { echo "Submit from the CUSI root"; exit 1; }
bash scripts/slurm/vllm_python.sh "$@"
