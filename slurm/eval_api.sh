#!/usr/bin/env bash
# Slurm job for scripts/slurm/eval_api.sh (all logic lives there). Submit from the CUSI root, with the API key in the
# environment (--export=ALL):
#   sbatch --output="$results_dir/logs/eval_api_%j.out" slurm/eval_api.sh --env <env> --model <model> \
#       --model_backend openrouter --run <run> --tasks <file>
#   (Android: add --exclusive and --n_emulators 3 for one adb server per node; Web judge: --stage judge with
#    --partition=medium-lg --gres=gpu:rtxa6000:2 on the sbatch line, it serves the judge with vLLM)
# Arguments after the script name are passed through to scripts/slurm/eval_api.sh.
# The agent calls an API: no GPU for --stage agent.
#SBATCH --job-name=cusi-eval-api
#SBATCH --partition=medium
#SBATCH --gres=gpu:0
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=24:00:00
# (medium's limit)
[[ -f scripts/slurm/eval_api.sh ]] || { echo "Submit from the CUSI root"; exit 1; }
bash scripts/slurm/eval_api.sh "$@"
