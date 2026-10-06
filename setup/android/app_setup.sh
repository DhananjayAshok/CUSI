#!/usr/bin/env bash
# One-time AndroidWorld app setup on the AVD (AndroidWorld's --perform_emulator_setup):
# boots the AVD writable, installs and configures the apps, then shuts it down cleanly
# so they persist. Runs inside the container on a node with KVM; needs internet.
#
#   bash scripts/container.sh bash setup/android/app_setup.sh [--force <value>]

source scripts/utils.sh || { echo "Could not source utils"; exit 1; }

declare -A ARGS
populate_dict ANDROID_APP_SETUP_DEFAULTS ARGS
REQUIRED_ARGS=()
populate_array ANDROID_APP_SETUP_ESSENTIALS REQUIRED_ARGS

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
# Bump whenever app setup changes (here or in android_world's setup_device/apps.py);
# setup/setup.sh rebuilds any AVD whose marker records an older version.
APP_SETUP_VERSION=4
MARKER="$ANDROID_AVD_HOME/${ARGS["avd"]}.avd/cusi_apps_installed"
if [[ -f "$MARKER" && "${ARGS["force"]}" != "true" ]]; then
    if grep -q "^version=$APP_SETUP_VERSION$" "$MARKER"; then
        echo "[skip] apps already installed on ${ARGS["avd"]} ($(tr '\n' ' ' < "$MARKER"))"
        exit 0
    fi
    echo "Apps were installed by an older app setup; recreate the AVD first:"
    echo "  bash scripts/container.sh bash setup/android/create_avd.sh --force true"
    exit 1
fi

bash scripts/android_emulator.sh --action start --avd "${ARGS["avd"]}" --read_only false
trap 'bash scripts/android_emulator.sh --action stop' EXIT

python - <<'EOF'
import shutil
from absl import logging
from android_world.env import env_launcher
from android_world.env.setup_device import setup
from android_world.utils import app_snapshot

# AndroidWorld's setup_app, plus a record of which apps' onboarding failed. On a
# freshly created AVD the first boot can be slow enough that several apps' first-run
# screens are not dismissed in time; retrying just those apps fixes it.
failed = []

def recording_setup_app(app, env):
    try:
        app.setup(env)
    except ValueError as e:
        logging.warning("Failed to automatically setup app %s: %s.", app.app_name, e)
        failed.append(app)
    app_snapshot.save_snapshot(app.app_name, env.controller)

setup.setup_app = recording_setup_app
env = env_launcher.load_and_setup_env(
    console_port=5554, emulator_setup=True, adb_path=shutil.which("adb"), grpc_port=8554)
for attempt in (2, 3):
    if not failed:
        break
    retry = tuple(dict.fromkeys(failed))
    failed.clear()
    print(f"Attempt {attempt}: retrying setup of {[a.app_name for a in retry]}", flush=True)
    setup.setup_apps(env, app_list=retry)
env.close()
if failed:
    print(f"WARNING: setup still failed after retries for {[a.app_name for a in dict.fromkeys(failed)]}")
print("AndroidWorld emulator setup complete")
EOF

bash scripts/android_emulator.sh --action stop
trap - EXIT
printf 'version=%s\ndate=%s\n' "$APP_SETUP_VERSION" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" > "$MARKER"
echo "Apps installed on ${ARGS["avd"]}; marker: $MARKER"
