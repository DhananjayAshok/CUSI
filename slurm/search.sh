#!/usr/bin/env bash
# Slurm job for scripts/slurm/search.sh (all logic lives there). Submit from the CUSI root:
#   sbatch --output="$results_dir/logs/search_%j.out" slurm/search.sh \
#       --env <env> --run <run> --max_expansions <max_expansions> \
#       --image_embedder <image_embedder> --text_embedder <text_embedder> --novelty_scorer <novelty_scorer>
# Arguments after the script name are passed through to scripts/slurm/search.sh.
# One GPU covers SigLIP; add GPUs (--gres=gpu:rtxa6000:2) when vLLM serves --model.
#SBATCH --job-name=cusi-search
#SBATCH --partition=medium-lg
#SBATCH --gres=gpu:rtxa6000:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=23:00:00
# (Every launch resumes the run, so resubmit the same command to continue.)
[[ -f scripts/slurm/search.sh ]] || { echo "Submit from the CUSI root"; exit 1; }
bash scripts/slurm/search.sh "$@"
