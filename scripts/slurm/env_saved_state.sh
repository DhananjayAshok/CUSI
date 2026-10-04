#!/usr/bin/env bash
# save_state / load_state / delete_state test on AndroidWorld (tests/env_saved_state_test.py):
# real emulator with snapshots enabled. Run from the CUSI root (Slurm wrapper: slurm/env_saved_state.sh).
#
# GPUs: none. Needs /dev/kvm on the node.
#
#   bash scripts/slurm/env_saved_state.sh
source scripts/utils.sh || { echo "Could not source utils"; exit 1; }
declare -A ARGS
REQUIRED_ARGS=()
parse_args ARGS REQUIRED_ARGS "$@"

echo "Node: $(hostname)"; ls -la /dev/kvm
bash scripts/container.sh bash -c '
    bash scripts/android_emulator.sh --action start --snapshots true
    trap "bash scripts/android_emulator.sh --action stop" EXIT
    python -u tests/env_saved_state_test.py --env android
'
