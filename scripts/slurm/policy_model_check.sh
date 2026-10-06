#!/usr/bin/env bash
# curiosity_plan §3.5 step 0: load each --policy_model <policy_model> VLMPolicy, generate, backward, time it
# (tests/policy_model_check.py). Run from the CUSI root (Slurm wrapper: slurm/policy_model_check.sh).
#
# GPUs: uses the first visible GPU.
#
#   bash scripts/slurm/policy_model_check.sh --models <models>
source scripts/utils.sh || { echo "Could not source utils"; exit 1; }
declare -A ARGS
populate_dict POLICY_MODEL_CHECK_DEFAULTS ARGS
REQUIRED_ARGS=()
populate_array POLICY_MODEL_CHECK_ESSENTIALS REQUIRED_ARGS
parse_args ARGS REQUIRED_ARGS "$@"

echo "Node: $(hostname)  GPUs: ${CUDA_VISIBLE_DEVICES:-none}"
nvidia-smi --query-gpu=name,memory.total --format=csv
FLAGS=""
for m in ${ARGS["models"]}; do FLAGS+=" --policy_model $m"; done
python -u tests/policy_model_check.py $FLAGS
