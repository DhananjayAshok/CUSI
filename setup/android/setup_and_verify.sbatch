#!/usr/bin/env bash
# Slurm job: AndroidWorld's one-time app setup on the AVD, then an end-to-end check
# (emulator + m3a_cusi against a mock model). Needs a node with /dev/kvm (the GPU
# nodes); no GPU is used. Submit from the CUSI root:
#
#   mkdir -p "$results_dir/logs"
#   sbatch --partition=medium-lg --chdir="$(pwd)" --output="$results_dir/logs/android_setup_%j.out" \
#       setup/android/setup_and_verify.sbatch
#
#SBATCH --job-name=cusi-android-setup
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00

set -euo pipefail
echo "Node: $(hostname)"; ls -la /dev/kvm
bash scripts/container.sh bash setup/android/app_setup.sh
# Apps whose first-run setup is fiddly must open straight into the app, not onto a
# welcome screen or permission prompt.
bash scripts/container.sh bash setup/android/inspect_apps.sh --apps "chrome contacts markor" \
    --fail_on "Welcome to Chrome|Use without an account|Back up .* contacts|send you notifications|Storage permission required"
bash scripts/container.sh bash setup/verify/e2e.sh --webvoyager false --android_world true
