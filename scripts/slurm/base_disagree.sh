#!/usr/bin/env bash
# Which practice action rows does the base model already get right? Serves --model on every visible GPU, then
# run_diagnostics.py base_disagree decodes every success/train action row of --dataset_dir as the eval agent would
# (greedy, 1000 tokens) into --out (resumable), and, with --out_dataset, writes the base-disagreement rows as a
# dataset there (run_diagnostics.py disagree_dataset). Run from the CUSI root (Slurm wrapper: slurm/base_disagree.sh).
#
#   bash scripts/slurm/base_disagree.sh --model <base model> --dataset_dir <dir> --out <jsonl> \
#       [--out_dataset <new dataset dir>] [--val_copy <n>] [--max_model_len <n>]
source scripts/utils.sh || { echo "Could not source utils"; exit 1; }
declare -A ARGS
populate_dict BASE_DISAGREE_DEFAULTS ARGS
REQUIRED_ARGS=()
populate_array BASE_DISAGREE_ESSENTIALS REQUIRED_ARGS
parse_args ARGS REQUIRED_ARGS "$@"

MODEL="${ARGS["model"]}"
JOB="${SLURM_JOB_ID:-local$$}"
PORT=$(( 8100 + $(echo "$JOB" | tr -cd '0-9' | tail -c 6 | sed 's/^0*//;s/^$/0/') % 400 ))
echo "Node: $(hostname)  GPUs: ${CUDA_VISIBLE_DEVICES:-none}  model=$MODEL port=$PORT dataset=${ARGS["dataset_dir"]}"
bash scripts/serve_vllm.sh --model "$MODEL" --port "$PORT" --max_model_len "${ARGS["max_model_len"]}" \
    --extra "--no-enable-prefix-caching" || exit 1
trap "bash scripts/stop_vllm.sh --port $PORT" EXIT
python -u run_diagnostics.py base_disagree --base_url "http://127.0.0.1:$PORT/v1/" --model "$MODEL" \
    --dataset_dir "${ARGS["dataset_dir"]}" --out "${ARGS["out"]}" || exit 1
if [[ "${ARGS["out_dataset"]}" != "none" ]]; then
    python -u run_diagnostics.py disagree_dataset --src_dir "${ARGS["dataset_dir"]}" --records "${ARGS["out"]}" \
        --out_dir "${ARGS["out_dataset"]}" --val_copy "${ARGS["val_copy"]}" || exit 1
fi
