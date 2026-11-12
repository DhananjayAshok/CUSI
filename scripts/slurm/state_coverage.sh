#!/usr/bin/env bash
# State coverage of search runs of one env (cusi.search.state_coverage): each run's distinct env states (its counts and
# their curve over env steps), or, compared in groups (cusi.search.groups), each run's, each group's and each group's
# unshared distinct states per measure. Run from the CUSI root (Slurm wrapper: slurm/state_coverage.sh).
#
# CPU only: it replays the runs' recorded true states (no GPU).
#
#   bash scripts/slurm/state_coverage.sh --env <env> --runs <run1,run2,...> [--stage <value>] [--at_steps <value>]
#   bash scripts/slurm/state_coverage.sh --env <env> --groups "<name>=<run1,run2>;<name>=<run3,run4>" [--at_steps <value>]
#
# --stage compare (default: --runs makes each run its own group; --groups: 2+ named groups) | counts (each of the runs'
# measures, written to the run's state_coverage.json; --at_steps also prints them at that many steps).
source scripts/utils.sh || { echo "Could not source utils"; exit 1; }
declare -A ARGS
populate_dict STATE_COVERAGE_DEFAULTS ARGS
REQUIRED_ARGS=()
populate_array STATE_COVERAGE_ESSENTIALS REQUIRED_ARGS
parse_args ARGS REQUIRED_ARGS "$@"

echo "Node: $(hostname)  env=${ARGS["env"]} stage=${ARGS["stage"]} runs=${ARGS["runs"]} groups=${ARGS["groups"]}"
if [[ "${ARGS["stage"]}" == "counts" ]]; then
    [[ "${ARGS["runs"]}" == "none" ]] && { echo "Error: --stage counts needs --runs"; exit 1; }
    C=(--env "${ARGS["env"]}")
    for R in ${ARGS["runs"]//,/ }; do C+=(--run "$R"); done
    [[ "${ARGS["at_steps"]}" != "none" ]] && C+=(--at_steps "${ARGS["at_steps"]}")
    python -u -m cusi.search.state_coverage counts "${C[@]}"
else
    C=(--env "${ARGS["env"]}")
    [[ "${ARGS["runs"]}" != "none" ]] && C+=(--runs "${ARGS["runs"]}")
    [[ "${ARGS["groups"]}" != "none" ]] && C+=(--groups "${ARGS["groups"]}")
    [[ "${ARGS["at_steps"]}" != "none" ]] && C+=(--at_steps "${ARGS["at_steps"]}")
    python -u -m cusi.search.state_coverage compare "${C[@]}"
fi
