#!/usr/bin/env bash
# Slurm job for scripts/slurm/eval_api_judge.sh (all logic lives there). Submit from the CUSI root, with the API keys
# in the environment (--export=ALL):
#   sbatch --output="$results_dir/logs/eval_api_judge_%j.out" slurm/eval_api_judge.sh --model <model> \
#       --model_backend openrouter --run <run> --judge_model <judge_model> --judge_model_backend openrouter
# Arguments after the script name are passed through to scripts/slurm/eval_api_judge.sh.
# The judge is an API model: no GPU.
#SBATCH --job-name=cusi-eval-api-judge
#SBATCH --partition=medium
#SBATCH --gres=gpu:0
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=12:00:00
[[ -f scripts/slurm/eval_api_judge.sh ]] || { echo "Submit from the CUSI root"; exit 1; }
bash scripts/slurm/eval_api_judge.sh "$@"
