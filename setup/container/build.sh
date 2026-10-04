#!/usr/bin/env bash
# Build the CUSI benchmark container (setup/container/cusi.def) into
# $storage_dir/containers/cusi.sif. Needs internet; no root (uses --fakeroot).
# Run from the CUSI root: bash setup/container/build.sh [--force true]

source scripts/utils.sh || { echo "Could not source utils"; exit 1; }

declare -A ARGS
ARGS["force"]="false"
REQUIRED_ARGS=()

# --- Argument parsing (copy verbatim) ---
ALLOWED_FLAGS=("${REQUIRED_ARGS[@]}" "${!ARGS[@]}")
USAGE_STR="Usage: $0"
for req in "${REQUIRED_ARGS[@]}"; do
    USAGE_STR+=" --$req <value>"
done
for opt in "${!ARGS[@]}"; do
    if [[ ! " ${REQUIRED_ARGS[*]} " =~ " ${opt} " ]]; then
        if [[ -z "${ARGS[$opt]}" ]]; then
            echo "DEFAULT VALUE OF KEY \"$opt\" CANNOT BE BLANK"; exit 1
        fi
        USAGE_STR+=" [--$opt <value> (default: ${ARGS[$opt]})]"
    fi
done
function usage() { echo "$USAGE_STR"; exit 1; }

while [[ $# -gt 0 ]]; do
    case "$1" in
        --*)
            FLAG=${1#--}
            VALID=false
            for allowed in "${ALLOWED_FLAGS[@]}"; do
                if [[ "$FLAG" == "$allowed" ]]; then VALID=true; break; fi
            done
            if [ "$VALID" = false ]; then echo "Error: Unknown flag --$FLAG"; usage; fi
            ARGS["$FLAG"]="$2"; shift 2 ;;
        -h|--help) usage ;;
        *) echo "Unknown argument: $1"; usage ;;
    esac
done

for req in "${REQUIRED_ARGS[@]}"; do
    if [[ -z "${ARGS[$req]}" ]]; then echo "Error: --$req is required."; FAILED=true; fi
done
if [ "$FAILED" = true ]; then usage; fi
# --- End argument parsing ---

echo "Script: $0 Active variables:"
for key in "${!ARGS[@]}"; do
    echo "  -$key = ${ARGS[$key]}"
done

set -euo pipefail
SIF="${storage_dir%/}/containers/cusi.sif"
if [[ -f "$SIF" && "${ARGS["force"]}" != "true" ]]; then
    echo "$SIF already exists; pass --force true to rebuild."
    exit 0
fi
mkdir -p "$(dirname "$SIF")" "${storage_dir%/}/apptainer/cache"
export APPTAINER_CACHEDIR="${storage_dir%/}/apptainer/cache"
# The build sandbox must be on local disk: fakeroot builds fail on NFS.
export APPTAINER_TMPDIR="/tmp/${USER}-cusi-apptainer-build"
mkdir -p "$APPTAINER_TMPDIR"
trap 'rm -rf "$APPTAINER_TMPDIR"' EXIT

apptainer build --fakeroot --force "$SIF.partial" setup/container/cusi.def
mv "$SIF.partial" "$SIF"
echo "Built $SIF"
apptainer exec "$SIF" cat /opt/cusi-versions.txt
