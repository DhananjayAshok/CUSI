#!/usr/bin/env bash
# Create the AndroidWorld AVD (Pixel 6, API 33) under $ANDROID_AVD_HOME
# ($storage_dir/android/avd). Runs inside the container; needs no KVM.
#
#   bash scripts/container.sh bash setup/android/create_avd.sh [--avd AndroidWorldAvd] [--force false]

source scripts/utils.sh || { echo "Could not source utils"; exit 1; }

declare -A ARGS
ARGS["avd"]="AndroidWorldAvd"
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
[[ "${CUSI_CONTAINER:-}" == "1" ]] || { echo "Run this inside the container: bash scripts/container.sh bash $0"; exit 1; }
AVD="${ARGS["avd"]}"
if [[ -f "$ANDROID_AVD_HOME/$AVD.ini" && "${ARGS["force"]}" != "true" ]]; then
    echo "[skip] AVD $AVD already exists in $ANDROID_AVD_HOME"
    exit 0
fi
# avdmanager asks whether to create a custom hardware profile; answer no.
echo "no" | avdmanager create avd --force --name "$AVD" --device pixel_6 \
    --package "system-images;android-33;google_apis;x86_64"
# avdmanager's defaults (800M data partition, 1.5G RAM) are too small for
# AndroidWorld's ~20 apps. Keep user data in the AVD folder (not <temp>) so the
# one-time app setup persists.
CONFIG="$ANDROID_AVD_HOME/$AVD.avd/config.ini"
sed -i -e '/^disk\.dataPartition\.path=/d' \
       -e 's/^disk\.dataPartition\.size=.*/disk.dataPartition.size=8G/' \
       -e 's/^hw\.ramSize=.*/hw.ramSize=4096M/' \
       -e 's/^hw\.keyboard=.*/hw.keyboard=yes/' "$CONFIG"
grep -q '^hw\.keyboard=' "$CONFIG" || echo "hw.keyboard=yes" >> "$CONFIG"
echo "Created AVD $AVD in $ANDROID_AVD_HOME"
grep -E '^(disk\.dataPartition|hw\.ramSize|hw\.keyboard|image\.sysdir)' "$CONFIG"
emulator -list-avds
