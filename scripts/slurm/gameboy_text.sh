#!/usr/bin/env bash
# One cusi.envs.gameboy_text.harvest command (collect, seed-vram, check, ...) on several GameBoy variants at once, one
# process per variant, in parallel on the CPUs given. Run from the CUSI root (Slurm wrapper: slurm/gameboy_text.sh).
#
#   bash scripts/slurm/gameboy_text.sh --variants <comma-separated variants> --command <command> [--extra <value>]
source scripts/utils.sh || { echo "Could not source utils"; exit 1; }
declare -A ARGS
populate_dict GAMEBOY_TEXT_DEFAULTS ARGS
REQUIRED_ARGS=()
populate_array GAMEBOY_TEXT_ESSENTIALS REQUIRED_ARGS
parse_args ARGS REQUIRED_ARGS "$@"

EXTRA=()
[[ "${ARGS["extra"]}" != "none" ]] && read -r -a EXTRA <<< "${ARGS["extra"]}"
echo "Node: $(hostname)  command=${ARGS["command"]} variants=${ARGS["variants"]}"
STATUS=0
PIDS=()
IFS=',' read -r -a VARIANTS <<< "${ARGS["variants"]}"
for v in "${VARIANTS[@]}"; do
    python -u -m cusi.envs.gameboy_text.harvest "${ARGS["command"]}" --variant "$v" "${EXTRA[@]}" 2>&1 \
        | grep -v "UserWarning\|INFO" | sed "s/^/[$v] /" &
    PIDS+=($!)
done
for pid in "${PIDS[@]}"; do wait "$pid" || STATUS=1; done
exit $STATUS
