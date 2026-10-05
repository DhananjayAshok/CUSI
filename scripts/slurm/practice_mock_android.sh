#!/usr/bin/env bash
# Mock-model run of all six practice stages on AndroidWorld (tests/practice_pipeline_mock_test.py):
# real emulator, scripted model. Run from the CUSI root (Slurm wrapper: slurm/practice_mock_android.sh).
#
# GPUs: none. Needs /dev/kvm on the node.
#
#   bash scripts/slurm/practice_mock_android.sh
source scripts/utils.sh || { echo "Could not source utils"; exit 1; }
declare -A ARGS
REQUIRED_ARGS=()
parse_args ARGS REQUIRED_ARGS "$@"

echo "Node: $(hostname)"; ls -la /dev/kvm
bash scripts/container.sh bash -c '
    bash scripts/android_emulator.sh --action start --snapshots true
    trap "bash scripts/android_emulator.sh --action stop" EXIT
    python -u tests/practice_pipeline_mock_test.py --env android
'
