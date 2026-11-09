#!/usr/bin/env bash
# Does a practice adapter reproduce its training data? Serves --model with the adapter as a LoRA module on every visible
# GPU, then run_diagnostics.py action_repro decodes the adapter's own training and held-out practice rows greedily with
# both. Run from the CUSI root (Slurm wrapper: slurm/action_repro.sh).
#
#   bash scripts/slurm/action_repro.sh --model <base model> --lora <adapter dir> --out <json> [--n <value>]
source scripts/utils.sh || { echo "Could not source utils"; exit 1; }
declare -A ARGS
populate_dict ACTION_REPRO_DEFAULTS ARGS
REQUIRED_ARGS=()
populate_array ACTION_REPRO_ESSENTIALS REQUIRED_ARGS
parse_args ARGS REQUIRED_ARGS "$@"

MODEL="${ARGS["model"]}"
LORA="${ARGS["lora"]%/}"
JOB="${SLURM_JOB_ID:-local$$}"
PORT=$(( 8100 + $(echo "$JOB" | tr -cd '0-9' | tail -c 6 | sed 's/^0*//;s/^$/0/') % 400 ))
[[ -f "$LORA/adapter_config.json" ]] || { echo "No LoRA adapter at $LORA"; exit 1; }
NAME="repro-$(basename "$(dirname "$(dirname "$LORA")")")"
RANK=$(python -c "import json, sys; print(json.load(open(sys.argv[1]))['r'])" "$LORA/adapter_config.json")
echo "Node: $(hostname)  GPUs: ${CUDA_VISIBLE_DEVICES:-none}  model=$MODEL lora=$LORA (r$RANK) port=$PORT"
bash scripts/serve_vllm.sh --model "$MODEL" --port "$PORT" \
    --extra "--no-enable-prefix-caching --enable-lora --max-loras 1 --max-lora-rank $RANK --lora-modules $NAME=$LORA" || exit 1
trap "bash scripts/stop_vllm.sh --port $PORT" EXIT
python -u run_diagnostics.py action_repro --base_url "http://127.0.0.1:$PORT/v1/" --base_model "$MODEL" \
    --lora_model "$NAME" --adapter "$LORA" --out "${ARGS["out"]}" --n "${ARGS["n"]}"
