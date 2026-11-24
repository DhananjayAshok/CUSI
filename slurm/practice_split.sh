#!/usr/bin/env bash
# Slurm job for scripts/slurm/practice_split.sh (all logic lives there). Submit from the CUSI root:
#   sbatch --output="$results_dir/logs/practice_split_%j.out" slurm/practice_split.sh --env <env> --model <model> \
#       --model_backend <backend> --source <source> --split <split>
# Arguments after the script name are passed through to scripts/slurm/practice_split.sh.
# No GPU (the model is an API); Android needs /dev/kvm and a node of its own (submit with --exclusive).
#SBATCH --job-name=cusi-practice-split
#SBATCH --partition=medium-lg
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=23:00:00
# (Resubmitting the same command continues with the tasks not yet attempted.)
[[ -f scripts/slurm/practice_split.sh ]] || { echo "Submit from the CUSI root"; exit 1; }
bash scripts/slurm/practice_split.sh "$@"
