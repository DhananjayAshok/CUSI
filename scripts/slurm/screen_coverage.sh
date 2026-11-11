#!/usr/bin/env bash
# Screen coverage of search runs of one env (cusi.search.screen_coverage): embed each run's frames (resumable; a run
# that grew is embedded from where it stopped), then compare the runs frame by frame. Run from the CUSI root (Slurm
# wrapper: slurm/screen_coverage.sh).
#
# GPUs: siglip embeds on the first visible GPU (CPU without one); random_patch needs none; compare uses the first
# visible GPU if any.
#
#   bash scripts/slurm/screen_coverage.sh --env <env> --runs <run1,run2,...> [--embedder <value>] \
#       [--encoder_model <value>] [--stage <value>] [--tau <value>] [--at_steps <value>]
#
# --stage embed | compare | both | distinct (default both; compare needs 2+ runs and reports each run's distinct
# screens too; distinct: each run's distinct screens alone, after embedding). --embedder siglip (default) | random_patch;
# --encoder_model applies to siglip (default google/siglip2-base-patch16-224). --tau defaults to
# configs/screen_coverage.yaml.
source scripts/utils.sh || { echo "Could not source utils"; exit 1; }
declare -A ARGS
populate_dict SCREEN_COVERAGE_DEFAULTS ARGS
REQUIRED_ARGS=()
populate_array SCREEN_COVERAGE_ESSENTIALS REQUIRED_ARGS
parse_args ARGS REQUIRED_ARGS "$@"

echo "Node: $(hostname)  GPUs: ${CUDA_VISIBLE_DEVICES:-none}  env=${ARGS["env"]} runs=${ARGS["runs"]} embedder=${ARGS["embedder"]}"
KEY=(--embedder "${ARGS["embedder"]}")
[[ "${ARGS["encoder_model"]}" != "none" ]] && KEY+=(--encoder_model "${ARGS["encoder_model"]}")
if [[ "${ARGS["stage"]}" == "embed" || "${ARGS["stage"]}" == "both" || "${ARGS["stage"]}" == "distinct" ]]; then
    RUNS=()
    for R in ${ARGS["runs"]//,/ }; do RUNS+=(--run "$R"); done
    python -u -m cusi.search.screen_coverage embed --env "${ARGS["env"]}" "${RUNS[@]}" "${KEY[@]}" || exit 1
fi
if [[ "${ARGS["stage"]}" == "compare" || "${ARGS["stage"]}" == "both" ]]; then
    C=(--env "${ARGS["env"]}" --runs "${ARGS["runs"]}" "${KEY[@]}")
    [[ "${ARGS["tau"]}" != "none" ]] && C+=(--tau "${ARGS["tau"]}")
    [[ "${ARGS["at_steps"]}" != "none" ]] && C+=(--at_steps "${ARGS["at_steps"]}")
    python -u -m cusi.search.screen_coverage compare "${C[@]}"
fi
if [[ "${ARGS["stage"]}" == "distinct" ]]; then
    D=(--env "${ARGS["env"]}" "${KEY[@]}")
    for R in ${ARGS["runs"]//,/ }; do D+=(--run "$R"); done
    [[ "${ARGS["tau"]}" != "none" ]] && D+=(--tau "${ARGS["tau"]}")
    [[ "${ARGS["at_steps"]}" != "none" ]] && D+=(--at_steps "${ARGS["at_steps"]}")
    python -u -m cusi.search.screen_coverage distinct "${D[@]}"
fi
