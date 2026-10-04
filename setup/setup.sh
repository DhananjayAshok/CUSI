#!/usr/bin/env bash
# One-command CUSI setup, starting from a plain (non-recursive) git clone.
# Safe to rerun: every step skips work that is already done.
#
#   bash setup/setup.sh            (from the CUSI root, or anywhere: it cds there)
#
# Writes only to /tmp, the CUSI root and storage_dir.
#
# Steps:
#   1. Submodules        clone any that are missing (existing checkouts are left alone)
#   2. Python venv       uv sync in setup/
#   3. Config            configs/config.env from configs/*.yaml
#   4. Container         Android emulator + Chromium image  (setup/container/)
#   5. Android AVD       create the AndroidWorld AVD under storage_dir
#   6. AndroidWorld apps one-time app install on the AVD; needs KVM (else via Slurm)
#   7. GameBoyRL         private_vars from CUSI's config; link the ROMs (rom_dir)
#   8. Verify            smoke tests + end-to-end browser (and emulator) checks
#
# Environment: CUSI_SLURM_PARTITION picks the Slurm partition for step 6 when this
# node has no /dev/kvm (default: medium-lg).
#
# Steps 1-3 cannot use scripts/utils.sh: it needs configs/config.env and the venv,
# which these steps create.

set -euo pipefail
cd "$(dirname "$(realpath "$0")")/.."
PROJECT_ROOT=$(pwd)

function step() { echo; echo "=== $*"; }
function fail() { echo "ERROR: $*" >&2; exit 1; }

# --------------------------------------------------------------------- 1. Submodules
step "1. Submodules"
while read -r _key sub_path; do
    if [[ -e "$sub_path/.git" ]]; then
        echo "[skip] $sub_path (already checked out)"
    else
        echo "[clone] $sub_path"
        git submodule update --init --recursive -- "$sub_path"
    fi
done < <(git config -f .gitmodules --get-regexp '^submodule\..*\.path$')

# --------------------------------------------------------------------- 2. Python venv
step "2. Python venv (uv sync)"
# storage_dir is needed before the venv exists (uv's cache and Python go there), so
# read it straight from the YAML.
STORAGE=$(sed -n 's/^storage_dir:[[:space:]]*"\{0,1\}\([^"#]*\)"\{0,1\}.*/\1/p' configs/private_vars.yaml | sed 's/[[:space:]]*$//')
STORAGE="${STORAGE%/}"
[[ -n "$STORAGE" && "$STORAGE" != "PLACEHOLDER" && "$STORAGE" == /* ]] \
    || fail "Set storage_dir (an absolute path) in configs/private_vars.yaml, then rerun. Got: '${STORAGE}'"
mkdir -p "$STORAGE"
# Keep uv's downloads and Python installs under storage_dir rather than $HOME.
export UV_CACHE_DIR="$STORAGE/uv/cache"
export UV_PYTHON_INSTALL_DIR="$STORAGE/uv/python"
if [[ -x "$STORAGE/uv/bin/uv" ]]; then
    export PATH="$STORAGE/uv/bin:$PATH"
fi
if ! command -v uv > /dev/null; then
    echo "uv not found; installing a standalone uv into $STORAGE/uv/bin"
    curl -LsSf https://astral.sh/uv/install.sh \
        | env UV_INSTALL_DIR="$STORAGE/uv/bin" UV_NO_MODIFY_PATH=1 sh
    export PATH="$STORAGE/uv/bin:$PATH"
fi
echo "uv: $(command -v uv) ($(uv --version))"
(cd setup && uv sync)

# --------------------------------------------------------------------- 3. Config
step "3. Config (configs/config.env)"
source setup/.venv/bin/activate
python configs/create_env_file.py > /dev/null || fail "create_env_file.py failed. Fill every PLACEHOLDER in configs/private_vars.yaml (storage_dir, results_dir, ...) and rerun."
[[ -s configs/config.env ]] || fail "configs/config.env was not written."
for key in storage_dir results_dir; do
    grep -q "^export ${key}=" configs/config.env || fail "configs/config.env has no ${key}."
done
echo "configs/config.env: $(grep -c '^export ' configs/config.env) variables"
# From here on the normal script environment works.
source scripts/utils.sh
echo "storage_dir=$storage_dir"
echo "results_dir=$results_dir"

# --------------------------------------------------------------------- 4. Container
step "4. Container ($CUSI_SIF)"
command -v apptainer > /dev/null || fail "apptainer not found; it is needed for the Android emulator and Chromium."
bash setup/container/build.sh

[[ "${storage_dir%/}" == "$STORAGE" ]] || fail "storage_dir in config.env ($storage_dir) differs from private_vars.yaml ($STORAGE)."
HAVE_KVM=false
[[ -r /dev/kvm && -w /dev/kvm ]] && HAVE_KVM=true

# --------------------------------------------------------------------- 5. Android AVD
step "5. Android AVD"
# app_setup.sh records its APP_SETUP_VERSION in this marker. An AVD set up by an
# older version is rebuilt from scratch, so setup fixes always reach the AVD.
APPS_MARKER="$STORAGE/android/avd/AndroidWorldAvd.avd/cusi_apps_installed"
WANT_VERSION=$(sed -n 's/^APP_SETUP_VERSION=\([0-9]*\).*/\1/p' setup/android/app_setup.sh)
if [[ -f "$APPS_MARKER" ]] && ! grep -q "^version=$WANT_VERSION$" "$APPS_MARKER"; then
    echo "AVD apps were set up by an older app setup (want version $WANT_VERSION); rebuilding the AVD."
    bash scripts/container.sh bash setup/android/create_avd.sh --force true
else
    bash scripts/container.sh bash setup/android/create_avd.sh
fi

# --------------------------------------------------------------------- 6. AndroidWorld apps
step "6. AndroidWorld app setup (needs KVM)"
ANDROID_E2E_DONE=false
if [[ -f "$APPS_MARKER" ]]; then
    echo "[skip] apps already installed ($(tr '\n' ' ' < "$APPS_MARKER"))"
elif [[ "$HAVE_KVM" == "true" ]]; then
    bash scripts/container.sh bash setup/android/app_setup.sh
elif command -v sbatch > /dev/null; then
    # No KVM here (e.g. a login node): run app setup + the AndroidWorld e2e check on a
    # GPU node and wait for it. CUSI_SLURM_PARTITION picks the partition.
    mkdir -p "${results_dir%/}/logs"
    echo "No /dev/kvm here; running in Slurm (log: ${results_dir%/}/logs/android_setup_<jobid>.out)"
    sbatch --wait --partition="${CUSI_SLURM_PARTITION:-medium-lg}" --chdir="$PROJECT_ROOT" \
        --output="${results_dir%/}/logs/android_setup_%j.out" setup/android/setup_and_verify.sbatch \
        || fail "Slurm Android setup job failed; see ${results_dir%/}/logs/"
    ANDROID_E2E_DONE=true
else
    fail "AndroidWorld app setup needs /dev/kvm (or Slurm to reach a node with it)."
fi
[[ -f "$APPS_MARKER" ]] || fail "App setup did not finish (no $APPS_MARKER)."

# --------------------------------------------------------------------- 7. GameBoyRL
step "7. GameBoyRL"
# Fill GameBoyRL's and GameBoyWorlds' private_vars.yaml from CUSI's config, but only
# while they still hold their committed defaults; hand-edited values are kept.
GBRL_VARS=GameBoyRL/configs/private_vars.yaml
if grep -q '^storage_dir: "PLACEHOLDER"' "$GBRL_VARS"; then
    sed -i -e "s|^storage_dir: \"PLACEHOLDER\"|storage_dir: \"$STORAGE/GameBoyRL\"|" \
           -e "s|^env_dir: .*|env_dir: \"$PROJECT_ROOT/setup/.venv/\" # CUSI's shared venv|" "$GBRL_VARS"
    echo "[write] $GBRL_VARS"
else
    echo "[skip] $GBRL_VARS (already set)"
fi
GBW_VARS=GameBoyRL/GameBoyWorlds/configs/private_vars.yaml
if grep -q '^storage_dir: "storage"' "$GBW_VARS"; then
    sed -i "s|^storage_dir: \"storage\"|storage_dir: \"$STORAGE/GameBoyWorlds\"|" "$GBW_VARS"
    echo "[write] $GBW_VARS"
else
    echo "[skip] $GBW_VARS (already set)"
fi
# ROMs cannot be downloaded. If CUSI's config sets rom_dir, link it in as GameBoyWorlds'
# rom_data; otherwise say where the ROMs go.
GBW_STORAGE=$(sed -n 's/^storage_dir: "\(.*\)".*/\1/p' "$GBW_VARS")
[[ "$GBW_STORAGE" == /* ]] || GBW_STORAGE="GameBoyRL/GameBoyWorlds/$GBW_STORAGE"
if [[ -e "$GBW_STORAGE/rom_data" ]]; then
    echo "[skip] $GBW_STORAGE/rom_data exists"
elif [[ -n "${rom_dir:-}" ]]; then
    mkdir -p "$GBW_STORAGE"
    ln -s "$rom_dir" "$GBW_STORAGE/rom_data"
    echo "[link] $GBW_STORAGE/rom_data -> $rom_dir"
else
    echo "[warn] No ROMs: put them under $GBW_STORAGE/rom_data (layout: GameBoyRL/GameBoyWorlds/configs/rom_data_path_vars.yaml),"
    echo "       e.g. with GameBoyRL/GameBoyWorlds/download_roms.sh (fill in your own Drive links),"
    echo "       or set rom_dir in configs/private_vars.yaml and rerun."
fi

# --------------------------------------------------------------------- 8. Verify
step "8. Verify"
python tests/smoke_android_world_cusi.py | tail -1
python tests/smoke_webvoyager_cusi.py | tail -1
if [[ "$HAVE_KVM" == "true" ]]; then
    bash scripts/container.sh bash setup/verify/e2e.sh --webvoyager true --android_world true
else
    bash scripts/container.sh bash setup/verify/e2e.sh --webvoyager true --android_world false
    if [[ "$ANDROID_E2E_DONE" == "true" ]]; then
        echo "AndroidWorld e2e ran in the Slurm job of step 6."
    else
        echo "AndroidWorld e2e skipped (no KVM here). To run it on a GPU node:"
        echo "  sbatch --partition=medium-lg --chdir=$PROJECT_ROOT --output=${results_dir%/}/logs/android_setup_%j.out setup/android/setup_and_verify.sbatch"
    fi
fi

echo
echo "Setup finished."
