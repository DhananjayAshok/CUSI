#!/usr/bin/env bash
# Practice on a benchmark split (run_practice.py attempt_split) with an API model (no vLLM server): each task of the
# split is attempted as the env's eval runs it, judged by the same model with critique-hint retries. With --split
# test this trains on the test tasks themselves (an oracle). Run from the CUSI root (Slurm wrapper:
# slurm/practice_split.sh); resubmitting continues with the tasks not yet attempted.
#
# Uses no GPU. GameBoy runs on the node; Web and Android in the CUSI container. Android starts --n_emulators
# emulators (each needs /dev/kvm; one adb server per node, so two Android jobs must not share a node: --exclusive),
# emulator i attempting shard i/K. The model's API key comes from the environment (~/.bashrc is sourced).
#
#   bash scripts/slurm/practice_split.sh --env <env> --model <model> --model_backend <backend> --source <source> \
#       --split <split> [--repeats <value>] [--n_workers <value>] [--max_tries <value>] [--tasks <value>] \
#       [--n_emulators <value>] [--max_new_tokens <value>]   # judge / critique budget (a thinking model needs more)
#   bash scripts/slurm/practice_split.sh ... --command dataset --extra "<run_practice.py dataset flags>"
source scripts/utils.sh || { echo "Could not source utils"; exit 1; }
# The model and its backend are always given: never defaulted.
PRACTICE_SPLIT_ESSENTIALS=("env" "model" "model_backend" "source" "split")
declare -A PRACTICE_SPLIT_DEFAULTS=(
    ["repeats"]="0"
    ["n_workers"]="8"
    ["max_tries"]="3"
    ["tasks"]="none"
    ["n_emulators"]="1"
    ["command"]="attempt_split"
    ["extra"]="none"
    ["max_new_tokens"]="2000"
)
declare -A ARGS
populate_dict PRACTICE_SPLIT_DEFAULTS ARGS
REQUIRED_ARGS=()
populate_array PRACTICE_SPLIT_ESSENTIALS REQUIRED_ARGS
parse_args ARGS REQUIRED_ARGS "$@"
source ~/.bashrc > /dev/null 2>&1 || true     # the API keys
source setup/.venv/bin/activate || { echo "Virtual environment not found."; exit 1; }

ENV_NAME="${ARGS["env"]}"
echo "Node: $(hostname)  env=$ENV_NAME model=${ARGS["model"]} (${ARGS["model_backend"]}) source=${ARGS["source"]} split=${ARGS["split"]}"
BASE=(--env "$ENV_NAME" --model_name "${ARGS["model"]}" --model_backend "${ARGS["model_backend"]}"
      --source "${ARGS["source"]}" --max_new_tokens "${ARGS["max_new_tokens"]}" "${ARGS["command"]}")
FLAGS=()
if [[ "${ARGS["command"]}" == "attempt_split" ]]; then
    FLAGS+=(--split "${ARGS["split"]}" --repeats "${ARGS["repeats"]}" --n_workers "${ARGS["n_workers"]}"
            --max_tries "${ARGS["max_tries"]}")
    [[ "${ARGS["tasks"]}" != "none" ]] && FLAGS+=(--tasks "${ARGS["tasks"]}")
fi
if [[ "${ARGS["extra"]}" != "none" ]]; then
    read -r -a EXTRA <<< "${ARGS["extra"]}"
    FLAGS+=("${EXTRA[@]}")
fi
if [[ "$ENV_NAME" != "android" || "${ARGS["command"]}" != "attempt_split" ]]; then
    CMD="python -u run_practice.py ${BASE[*]@Q} ${FLAGS[*]@Q}"
    case "$ENV_NAME" in
        gameboy) bash -c "$CMD" ;;
        web|android) bash scripts/container.sh bash -c "$CMD" ;;
        *) echo "Unknown --env $ENV_NAME"; exit 1 ;;
    esac
    exit $?
fi
# Android: K emulators in one container, emulator i (console 5554+2i, gRPC 8554+i) attempting shard i/K
ROOT=$(python -c "from cusi.utils.parameter_handling import load_parameters; from cusi.utils.paths import practice_root; import sys; print(practice_root(parameters=load_parameters(), env='android', model_name=sys.argv[1], source=sys.argv[2]))" "${ARGS["model"]}" "${ARGS["source"]}")
mkdir -p "$ROOT"
K="${ARGS["n_emulators"]}"
SCRIPT="set -u; adb start-server > /dev/null; STOP=''"
for i in $(seq 0 $((K - 1))); do
    CP=$((5554 + 2 * i)); GP=$((8554 + i))
    SCRIPT+="; bash scripts/android_emulator.sh --action start --port $CP --grpc_port $GP --snapshots false || exit 1"
    SCRIPT+="; STOP=\"\$STOP bash scripts/android_emulator.sh --action stop --port $CP;\""
    SCRIPT+="; trap \"\$STOP\" EXIT; adb -s emulator-$CP wait-for-device"
done
SCRIPT+="; sleep 30; PIDS=''"
for i in $(seq 0 $((K - 1))); do
    CP=$((5554 + 2 * i)); GP=$((8554 + i))
    F=("${FLAGS[@]}" --shard "$i/$K" --console_port "$CP" --grpc_port "$GP")
    SCRIPT+="; python -u run_practice.py ${BASE[*]@Q} ${F[*]@Q} > ${ROOT@Q}/shard$i.log 2>&1 & PIDS=\"\$PIDS \$!\""
done
SCRIPT+="; S=0; for p in \$PIDS; do wait \$p || S=1; done; exit \$S"
bash scripts/container.sh bash -c "$SCRIPT"
