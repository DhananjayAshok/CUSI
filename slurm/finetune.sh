#!/usr/bin/env bash
# Slurm job for scripts/slurm/finetune.sh (all logic lives there). Submit from the CUSI root:
#   sbatch --output="$results_dir/logs/finetune_%j.out" slurm/finetune.sh \
#       --env <env> --model <model> --source <source> --run <run>
# Arguments after the script name are passed through to scripts/slurm/finetune.sh.
# FSDP2 over the job's GPUs (Gemma-4-31B: 4x A6000); rank 0 holds one host copy of the weights while loading.
#SBATCH --job-name=cusi-finetune
#SBATCH --partition=medium-lg
#SBATCH --gres=gpu:rtxa6000:4
#SBATCH --cpus-per-task=24
#SBATCH --mem=200G
#SBATCH --time=23:00:00
# (Resubmitting the same command continues from the last checkpoint.)
[[ -f scripts/slurm/finetune.sh ]] || { echo "Submit from the CUSI root"; exit 1; }
bash scripts/slurm/finetune.sh "$@"
