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
#   bash scripts/slurm/screen_coverage.sh --env <env> --groups "<name>=<run1,run2>;<name>=<run3,run4>" ...
#
# --runs: each run its own group; --groups: 2+ named groups of runs (cusi.search.groups; a run in one group only).
#
# --stage embed | compare | both | distinct (default both; compare needs 2+ runs or groups and reports each run's
# distinct screens too; distinct: each run's distinct screens alone, after embedding). --embedder siglip (default) | random_patch;
# --encoder_model applies to siglip (default google/siglip2-base-patch16-224). --tau defaults to
# configs/screen_coverage.yaml.
source scripts/utils.sh || { echo "Could not source utils"; exit 1; }
declare -A ARGS
populate_dict SCREEN_COVERAGE_DEFAULTS ARGS
REQUIRED_ARGS=()
populate_array SCREEN_COVERAGE_ESSENTIALS REQUIRED_ARGS
parse_args ARGS REQUIRED_ARGS "$@"

echo "Node: $(hostname)  GPUs: ${CUDA_VISIBLE_DEVICES:-none}  env=${ARGS["env"]} runs=${ARGS["runs"]} groups=${ARGS["groups"]} embedder=${ARGS["embedder"]}"
# every run named, from --runs or --groups ("name=r1,r2;name=r3")
ALL_RUNS="${ARGS["runs"]}"
if [[ "${ARGS["groups"]}" != "none" ]]; then
    ALL_RUNS=$(python -c "import sys; print(','.join(r for g in sys.argv[1].split(';') if '=' in g for r in g.split('=', 1)[1].split(',') if r.strip()))" "${ARGS["groups"]}")
fi
[[ "$ALL_RUNS" == "none" || -z "$ALL_RUNS" ]] && { echo "Error: --runs or --groups is required"; exit 1; }
KEY=(--embedder "${ARGS["embedder"]}")
[[ "${ARGS["encoder_model"]}" != "none" ]] && KEY+=(--encoder_model "${ARGS["encoder_model"]}")
if [[ "${ARGS["stage"]}" == "embed" || "${ARGS["stage"]}" == "both" || "${ARGS["stage"]}" == "distinct" ]]; then
    RUNS=()
    for R in ${ALL_RUNS//,/ }; do RUNS+=(--run "$R"); done
    python -u -m cusi.search.screen_coverage embed --env "${ARGS["env"]}" "${RUNS[@]}" "${KEY[@]}" || exit 1
fi
if [[ "${ARGS["stage"]}" == "compare" || "${ARGS["stage"]}" == "both" ]]; then
    C=(--env "${ARGS["env"]}" "${KEY[@]}")
    [[ "${ARGS["runs"]}" != "none" ]] && C+=(--runs "${ARGS["runs"]}")
    [[ "${ARGS["groups"]}" != "none" ]] && C+=(--groups "${ARGS["groups"]}")
    [[ "${ARGS["tau"]}" != "none" ]] && C+=(--tau "${ARGS["tau"]}")
    [[ "${ARGS["at_steps"]}" != "none" ]] && C+=(--at_steps "${ARGS["at_steps"]}")
    python -u -m cusi.search.screen_coverage compare "${C[@]}"
fi
if [[ "${ARGS["stage"]}" == "distinct" ]]; then
    D=(--env "${ARGS["env"]}" "${KEY[@]}")
    for R in ${ALL_RUNS//,/ }; do D+=(--run "$R"); done
    [[ "${ARGS["tau"]}" != "none" ]] && D+=(--tau "${ARGS["tau"]}")
    [[ "${ARGS["at_steps"]}" != "none" ]] && D+=(--at_steps "${ARGS["at_steps"]}")
    python -u -m cusi.search.screen_coverage distinct "${D[@]}"
fi
