#!/usr/bin/env bash
# LoRA fine-tuning (run_finetune.py) of --model <model> on the practice success rows of --source <source>, into
# <model_dir>/finetune/<env>/<source>/<run>/ (its adapter/ is what eval serves: scripts/slurm/eval.sh --lora).
# Run from the CUSI root (Slurm wrapper: slurm/finetune.sh); rerunning continues from the last checkpoint.
#
# GPUs: one torchrun process per visible GPU, the model FSDP2-sharded across them (together they must hold it:
# Gemma-4-31B is ~62 GB in bf16, e.g. 4x 48 GB). Rank 0 reads the checkpoint into host memory once.
#
#   bash scripts/slurm/finetune.sh --env <env> --model <model> --source <source> --run <run> \
#       [--extra "<more run_finetune.py flags>"]
source scripts/utils.sh || { echo "Could not source utils"; exit 1; }
declare -A ARGS
populate_dict FINETUNE_DEFAULTS ARGS
REQUIRED_ARGS=()
populate_array FINETUNE_ESSENTIALS REQUIRED_ARGS
parse_args ARGS REQUIRED_ARGS "$@"

GPU_NAMES=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | sort | uniq -c | sed 's/^ *//' | paste -sd ';')
echo "Node: $(hostname)  GPUs: ${CUDA_VISIBLE_DEVICES:-none} ($GPU_NAMES)  env=${ARGS["env"]} source=${ARGS["source"]} run=${ARGS["run"]}"
FLAGS=(--env "${ARGS["env"]}" --model_name "${ARGS["model"]}" --source "${ARGS["source"]}" --run_name "${ARGS["run"]}")
EXTRA=()
[[ "${ARGS["extra"]}" != "none" ]] && read -r -a EXTRA <<< "${ARGS["extra"]}"
NGPU=$(python -c "import torch; print(torch.cuda.device_count())")
[[ "$NGPU" -ge 1 ]] || { echo "No GPU visible"; exit 1; }
export OMP_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
torchrun --standalone --nproc_per_node "$NGPU" run_finetune.py "${FLAGS[@]}" "${EXTRA[@]}"
