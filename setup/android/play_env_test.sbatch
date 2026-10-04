#!/usr/bin/env bash
# Slurm job: end-to-end test of cusi_envs' AndroidPlayEnv on a real emulator
# (tests/android_play_env_e2e.py). Needs a KVM node. Submit from the CUSI root:
#
#   sbatch --partition=medium-lg --chdir="$(pwd)" --output="$results_dir/logs/play_env_test_%j.out" \
#       setup/android/play_env_test.sbatch
#
#SBATCH --job-name=cusi-play-env-test
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00

set -euo pipefail
echo "Node: $(hostname)"
bash scripts/container.sh bash -c '
    bash scripts/android_emulator.sh --action start --read_only true --snapshots true
    trap "bash scripts/android_emulator.sh --action stop" EXIT
    python -u tests/android_play_env_e2e.py
'
