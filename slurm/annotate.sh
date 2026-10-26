#!/usr/bin/env bash
# Slurm job for scripts/slurm/annotate.sh (all logic lives there). Submit from the CUSI root:
#   sbatch --output="$results_dir/logs/annotate_%j.out" slurm/annotate.sh --env <env> --run <run> --model <model>
# Arguments after the script name are passed through to scripts/slurm/annotate.sh.
# vLLM serves --model on every visible GPU (Gemma-4-31B needs 2).
#SBATCH --job-name=cusi-annotate
#SBATCH --partition=medium-lg
#SBATCH --gres=gpu:rtxa6000:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=23:00:00
# (Resubmitting the same command continues from the last annotated node.)
[[ -f scripts/slurm/annotate.sh ]] || { echo "Submit from the CUSI root"; exit 1; }
bash scripts/slurm/annotate.sh "$@"
