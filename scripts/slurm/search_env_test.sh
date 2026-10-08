#!/usr/bin/env bash
# Tree-search check on a real env: a short search, then a second process that resumes it, restores and compares
# every node (--verify_restores) and extends it; on Web first the live challenge-page detector test
# (tests/web_challenge_test.py --live). random_patch embeddings and the random expander, so no GPU or vLLM.
# Run from the CUSI root (Slurm wrapper: slurm/search_env_test.sh). Android needs /dev/kvm on the node.
#
#   bash scripts/slurm/search_env_test.sh --env <env> [--run <value>] [--first <value>] [--second <value>] \
#       [--k_steps <value>]
source scripts/utils.sh || { echo "Could not source utils"; exit 1; }
declare -A ARGS
populate_dict SEARCH_ENV_TEST_DEFAULTS ARGS
REQUIRED_ARGS=()
populate_array SEARCH_ENV_TEST_ESSENTIALS REQUIRED_ARGS
parse_args ARGS REQUIRED_ARGS "$@"

ENV_NAME="${ARGS["env"]}"
RUN="${ARGS["run"]}"
[[ "$RUN" == "none" ]] && RUN="search_test_$ENV_NAME"
echo "Node: $(hostname)  env=$ENV_NAME run=$RUN"
BASE="python -u run_search.py --env $ENV_NAME --run_name $RUN --image_embedder random_patch --text_embedder none --novelty_scorer embedding --k_steps ${ARGS["k_steps"]}"
STEPS="$BASE --max_expansions ${ARGS["first"]} --overwrite && $BASE --max_expansions ${ARGS["second"]} --verify_restores 1000"
RUN_DIR=$(python -m cusi.utils.paths search_run --env "$ENV_NAME" --run "$RUN")
export CUDA_VISIBLE_DEVICES=""

case "$ENV_NAME" in
    gameboy) bash -c "$STEPS"; STATUS=$? ;;
    web) bash scripts/container.sh bash -c "python -u -m tests.web_challenge_test --live && $STEPS"; STATUS=$? ;;
    android)
        bash scripts/container.sh bash -c "
            bash scripts/android_emulator.sh --action start --snapshots true || exit 1
            trap 'bash scripts/android_emulator.sh --action stop' EXIT
            $STEPS
        "; STATUS=$? ;;
    *) echo "Unknown --env $ENV_NAME"; exit 1 ;;
esac
echo "== $RUN_DIR/summary.json"
cat "$RUN_DIR/summary.json"
exit $STATUS
