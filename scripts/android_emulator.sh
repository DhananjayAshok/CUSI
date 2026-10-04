#!/usr/bin/env bash
# Start or stop a headless Android emulator. Runs inside the container (see
# scripts/container.sh) on a node with KVM.
#
#   bash scripts/android_emulator.sh --action start [--port 5554] [--grpc_port 8554] [--read_only true]
#   bash scripts/android_emulator.sh --action stop  [--port 5554]
#
# start returns once Android has booted; the emulator keeps running in the
# background of the same container session. --read_only true discards all changes
# on exit and lets several emulators share one AVD (use distinct --port and
# --grpc_port each). Use --read_only false only for one-time setup of the AVD.
#
# --snapshots true allows manual snapshots (`adb emu avd snapshot save/load`), which
# cusi_envs' AndroidPlayEnv uses for fast full resets. The emulator refuses snapshots
# with -read-only, so instead it runs on a private writable copy of the AVD in /tmp
# (--read_only is then ignored); the shared AVD is never modified, the copy is
# removed on stop, and several such emulators can still run side by side.

source scripts/utils.sh || { echo "Could not source utils"; exit 1; }

declare -A ARGS
ARGS["avd"]="AndroidWorldAvd"
ARGS["port"]="5554"
ARGS["grpc_port"]="8554"
ARGS["read_only"]="true"
ARGS["snapshots"]="false"
ARGS["gpu"]="off"
ARGS["boot_timeout"]="600"
REQUIRED_ARGS=("action")

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
[[ "${CUSI_CONTAINER:-}" == "1" ]] || { echo "Run this inside the container: bash scripts/container.sh bash $0 ..."; exit 1; }
PORT="${ARGS["port"]}"
SERIAL="emulator-$PORT"
RUN_DIR="/tmp/${USER}-cusi-emulator-$PORT"
mkdir -p "$RUN_DIR"

STAGE_DIR="$RUN_DIR/avd"

# stage_avd: private writable copy of the AVD (sparse) in $STAGE_DIR, without the
# shared AVD's lock files and snapshots.
function stage_avd() {
    local name="${ARGS["avd"]}"
    rm -rf "$STAGE_DIR"
    mkdir -p "$STAGE_DIR"
    cp -a --sparse=always "$ANDROID_AVD_HOME/$name.avd" "$STAGE_DIR/"
    rm -rf "$STAGE_DIR/$name.avd/snapshots" "$STAGE_DIR/$name.avd"/*.lock
    sed "s|^path=.*|path=$STAGE_DIR/$name.avd|" "$ANDROID_AVD_HOME/$name.ini" > "$STAGE_DIR/$name.ini"
}

function emulator_start() {
    [[ -r /dev/kvm && -w /dev/kvm ]] || { echo "No usable /dev/kvm on $(hostname); run on a GPU node (Slurm)."; exit 1; }
    local args=(-avd "${ARGS["avd"]}" -port "$PORT" -grpc "${ARGS["grpc_port"]}"
                -no-window -no-audio -no-boot-anim -accel on -gpu "${ARGS["gpu"]}")
    local avd_home="$ANDROID_AVD_HOME"
    if [[ "${ARGS["snapshots"]}" == "true" ]]; then
        local t=$SECONDS
        stage_avd
        avd_home="$STAGE_DIR"
        echo "Staged a private copy of ${ARGS["avd"]} in $STAGE_DIR ($((SECONDS - t))s)"
        # The emulator refuses to snapshot while a guest app renders with Vulkan
        # ("Snapshot save is skipped. Reason: UNSUPPORTED_VK_APP"); hide Vulkan from
        # the guest so apps use OpenGL ES instead.
        args+=(-no-snapshot-load -no-snapshot-save -feature -Vulkan)
    else
        args+=(-no-snapshot)
        if [[ "${ARGS["read_only"]}" == "true" ]]; then
            args+=(-read-only)
        fi
    fi
    adb start-server > /dev/null
    echo "Starting: emulator ${args[*]} (log: $RUN_DIR/emulator.log)"
    ANDROID_AVD_HOME="$avd_home" nohup emulator "${args[@]}" > "$RUN_DIR/emulator.log" 2>&1 &
    echo $! > "$RUN_DIR/emulator.pid"

    local start=$SECONDS
    while true; do
        if ! kill -0 "$(cat "$RUN_DIR/emulator.pid")" 2> /dev/null; then
            echo "Emulator exited during boot. Last log lines:"; tail -20 "$RUN_DIR/emulator.log"; exit 1
        fi
        if [[ "$(adb -s "$SERIAL" shell getprop sys.boot_completed 2> /dev/null | tr -d '\r')" == "1" ]]; then
            break
        fi
        if (( SECONDS - start > ${ARGS["boot_timeout"]} )); then
            echo "Timed out after ${ARGS["boot_timeout"]}s waiting for boot. Last log lines:"
            tail -20 "$RUN_DIR/emulator.log"; emulator_stop; exit 1
        fi
        sleep 5
    done
    # Same post-boot settings as android_world/docker_setup/start_emu_headless.sh.
    adb -s "$SERIAL" shell settings put global window_animation_scale 0.0
    adb -s "$SERIAL" shell settings put global transition_animation_scale 0.0
    adb -s "$SERIAL" shell settings put global animator_duration_scale 0.0
    echo "Booted $SERIAL in $((SECONDS - start))s"
}

function emulator_stop() {
    local pid_file="$RUN_DIR/emulator.pid"
    local pid=""
    [[ -f "$pid_file" ]] && pid=$(cat "$pid_file")
    # Power Android off properly first: it persists some state (app-ops, runtime
    # permissions) lazily, and `emu kill` alone skips that, so a writable AVD
    # can lose changes made just before stopping.
    if [[ -n "$pid" ]] && kill -0 "$pid" 2> /dev/null; then
        adb -s "$SERIAL" shell reboot -p > /dev/null 2>&1 || true
        for _ in $(seq 60); do
            kill -0 "$pid" 2> /dev/null || break
            sleep 1
        done
    fi
    adb -s "$SERIAL" emu kill > /dev/null 2>&1 || true
    if [[ -n "$pid" ]]; then
        # A clean exit (not a SIGKILL) is what makes a writable AVD keep its changes.
        for _ in $(seq 60); do
            kill -0 "$pid" 2> /dev/null || break
            sleep 1
        done
        if kill -0 "$pid" 2> /dev/null; then
            echo "Emulator $pid did not exit; killing it."
            kill -9 "$pid" || true
        fi
        rm -f "$pid_file"
    fi
    rm -rf "$STAGE_DIR"   # a --snapshots emulator's private AVD copy, if any
    echo "Stopped $SERIAL"
}

case "${ARGS["action"]}" in
    start) emulator_start ;;
    stop) emulator_stop ;;
    *) echo "Unknown --action ${ARGS["action"]}; use start or stop"; exit 1 ;;
esac
