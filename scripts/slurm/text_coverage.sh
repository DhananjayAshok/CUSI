#!/usr/bin/env bash
# Text coverage of search runs of one env (cusi.search.text_coverage): each run's distinct screen-text items and, for
# 2+ runs or groups of runs, the items only some of them saw (cusi.search.groups). Run from the CUSI root (Slurm wrapper: slurm/text_coverage.sh).
#
# CPU only: it reads the runs' recorded text (no embedding, no GPU).
#
#   bash scripts/slurm/text_coverage.sh --env <env> --runs <run1,run2,...> [--at_steps <value>]
#   bash scripts/slurm/text_coverage.sh --env <env> --groups "<name>=<run1,run2>;<name>=<run3,run4>" [--at_steps <value>]
#
# --runs: each run its own group (one run: its distinct items only); --groups: 2+ named groups of runs (a run in one
# group only). --at_steps: only each run's first N env steps.
source scripts/utils.sh || { echo "Could not source utils"; exit 1; }
declare -A ARGS
populate_dict TEXT_COVERAGE_DEFAULTS ARGS
REQUIRED_ARGS=()
populate_array TEXT_COVERAGE_ESSENTIALS REQUIRED_ARGS
parse_args ARGS REQUIRED_ARGS "$@"

echo "Node: $(hostname)  env=${ARGS["env"]} runs=${ARGS["runs"]} groups=${ARGS["groups"]}"
C=(--env "${ARGS["env"]}")
[[ "${ARGS["runs"]}" != "none" ]] && C+=(--runs "${ARGS["runs"]}")
[[ "${ARGS["groups"]}" != "none" ]] && C+=(--groups "${ARGS["groups"]}")
[[ "${ARGS["at_steps"]}" != "none" ]] && C+=(--at_steps "${ARGS["at_steps"]}")
python -u -m cusi.search.text_coverage compare "${C[@]}"
