#!/usr/bin/env bash
# Fine-tune, then deploy and evaluate, on one env: submits slurm/finetune.sh (LoRA on the practice success rows of
# --source) and, once it succeeds, slurm/eval.sh serving the base model with that adapter, into eval run
# lora_<run>. With --base_run <name>, also evaluates the base model alone into eval run <name> unless that run
# already has a summary (the baseline to compare with). Run from the CUSI root on the login node.
#
#   bash scripts/finetune_eval.sh --env <env> --model <model> --source <source> --run <run> \
#       [--extra "<run_finetune.py flags>"] [--eval_extra "<eval.sh flags>"] [--base_run <eval run>] \
#       [--after <job id the fine-tune waits for>]
#   e.g. --env gameboy --model google/gemma-4-31b-it --source tree_random_v1 --run lora_r16 \
#        --eval_extra "--workers 8" --base_run base_gemma-4-31b-it
# The base run must use the same agent code as the fine-tuned one (eval runs made before the executor's no-change
# note, e.g. full_gemma-4-31b-it_r0, do not).
source scripts/utils.sh || { echo "Could not source utils"; exit 1; }
declare -A ARGS
populate_dict FINETUNE_EVAL_DEFAULTS ARGS
REQUIRED_ARGS=()
populate_array FINETUNE_EVAL_ESSENTIALS REQUIRED_ARGS
parse_args ARGS REQUIRED_ARGS "$@"

ENV_NAME="${ARGS["env"]}"
MODEL="${ARGS["model"]}"
RUN="${ARGS["run"]}"
LOGS="$results_dir/logs"
EVAL_EXTRA=()
[[ "${ARGS["eval_extra"]}" != "none" ]] && read -r -a EVAL_EXTRA <<< "${ARGS["eval_extra"]}"

FT_FLAGS=(--env "$ENV_NAME" --model "$MODEL" --source "${ARGS["source"]}" --run "$RUN")
[[ "${ARGS["extra"]}" != "none" ]] && FT_FLAGS+=(--extra "${ARGS["extra"]}")
DEP=()
[[ "${ARGS["after"]}" != "none" ]] && DEP=(--dependency="afterok:${ARGS["after"]}")
FT=$(sbatch --parsable "${DEP[@]}" --job-name="ft-$ENV_NAME-$RUN" --output="$LOGS/finetune_%j.out" \
    slurm/finetune.sh "${FT_FLAGS[@]}") || exit 1
echo "fine-tune: job $FT"

ADAPTER="$(python -m cusi.utils.paths finetune_run --env "$ENV_NAME" --source "${ARGS["source"]}" --run "$RUN")/adapter"
EV=$(sbatch --parsable --dependency="afterok:$FT" --job-name="eval-$ENV_NAME-lora_$RUN" \
    --output="$LOGS/eval_%j.out" slurm/eval.sh --env "$ENV_NAME" --model "$MODEL" --run "lora_$RUN" \
    --lora "$ADAPTER" "${EVAL_EXTRA[@]}") || exit 1
echo "eval with the adapter ($ADAPTER): job $EV -> eval run lora_$RUN"

if [[ "${ARGS["base_run"]}" != "none" ]]; then
    BASE_DIR=$(python -m cusi.utils.paths eval_run --env "$ENV_NAME" --run "${ARGS["base_run"]}")
    if [[ -f "${BASE_DIR%/}/summary.json" ]]; then
        echo "base: eval run ${ARGS["base_run"]} already has a summary"
    else
        BE=$(sbatch --parsable --job-name="eval-$ENV_NAME-${ARGS["base_run"]}" --output="$LOGS/eval_%j.out" \
            slurm/eval.sh --env "$ENV_NAME" --model "$MODEL" --run "${ARGS["base_run"]}" "${EVAL_EXTRA[@]}") || exit 1
        echo "base eval: job $BE -> eval run ${ARGS["base_run"]}"
    fi
fi
