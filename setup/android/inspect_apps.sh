#!/usr/bin/env bash
# Diagnostic: boot the AVD read-only (no changes kept), open each app, and save a
# screenshot and the on-screen UI text, to see what an agent would face on launch.
# Runs inside the container on a node with KVM.
#
#   bash scripts/container.sh bash setup/android/inspect_apps.sh --apps <apps>

source scripts/utils.sh || { echo "Could not source utils"; exit 1; }

declare -A ARGS
populate_dict ANDROID_INSPECT_APPS_DEFAULTS ARGS
REQUIRED_ARGS=()
populate_array ANDROID_INSPECT_APPS_ESSENTIALS REQUIRED_ARGS

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
OUT="${ARGS["out_dir"]}"
[[ "$OUT" != "none" ]] || OUT="${storage_dir%/}/tmp/app_inspect/$(date +%Y%m%dT%H%M%S)"
mkdir -p "$OUT"

bash scripts/android_emulator.sh --action start --read_only true
trap 'bash scripts/android_emulator.sh --action stop' EXIT

for app in ${ARGS["apps"]}; do
    activity=$(python -c "from android_world.env import adb_utils; print(adb_utils.get_adb_activity('$app'))")
    echo; echo "=== $app ($activity)"
    adb -s emulator-5554 shell am start -n "$activity" > /dev/null
    sleep "${ARGS["wait"]}"
    adb -s emulator-5554 exec-out screencap -p > "$OUT/$app.png"
    adb -s emulator-5554 shell uiautomator dump /sdcard/ui.xml > /dev/null
    adb -s emulator-5554 exec-out cat /sdcard/ui.xml > "$OUT/$app.xml"
    # Visible text and content descriptions, one per line.
    { grep -o -E '(text|content-desc)="[^"]+"' "$OUT/$app.xml" || true; } | sed -E 's/^[a-z-]+="(.*)"$/  \1/' | sort -u > "$OUT/$app.txt"
    head -30 "$OUT/$app.txt"
    if [[ ! -s "$OUT/$app.txt" ]]; then
        echo "  FAIL: no UI text captured for $app"; FOUND_BAD=true
    elif [[ "${ARGS["fail_on"]}" != "none" ]] && grep -q -E "${ARGS["fail_on"]}" "$OUT/$app.txt"; then
        echo "  FAIL: $app shows: $(grep -E "${ARGS["fail_on"]}" "$OUT/$app.txt" | head -3 | tr '\n' '|')"; FOUND_BAD=true
    fi
    adb -s emulator-5554 shell am force-stop "${activity%%/*}"
done
echo; echo "Screenshots and UI dumps: $OUT"

# AndroidWorld restores each task's apps from per-app data snapshots taken at app
# setup. They must survive a reboot (this emulator is a fresh boot), or every task
# silently runs without resetting its apps' data.
SNAPSHOT_DATA=$(python -c "from android_world.env import device_constants; print(device_constants.SNAPSHOT_DATA)")
adb -s emulator-5554 root > /dev/null; sleep 2
N_SNAPSHOTS=$(adb -s emulator-5554 shell "ls $SNAPSHOT_DATA 2>/dev/null | wc -l" | tr -d '\r ')
echo "Per-app snapshots in $SNAPSHOT_DATA after boot: $N_SNAPSHOTS"
if [[ "${N_SNAPSHOTS:-0}" -lt 1 ]]; then
    echo "  FAIL: no per-app snapshots survived the boot"; FOUND_BAD=true
fi
if [[ "${FOUND_BAD:-false}" == "true" ]]; then
    exit 1
fi
