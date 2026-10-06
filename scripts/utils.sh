_env_files=(configs/*.env)
if [[ ! -f "${_env_files[0]}" ]]; then
    echo "No configs/*.env found; run: python configs/create_env_file.py"; exit 1
fi
for _env_file in "${_env_files[@]}"; do
    source "$_env_file" || { echo "Could not source $_env_file"; exit 1; }
done
source setup/.venv/bin/activate || { echo "Virtual environment not found."; exit 1; }
# Model weights come from the user's Hugging Face cache; it must already be configured.
[[ -n "${HF_HOME:-}" ]] || { echo "HF_HOME is not set; set it (e.g. in ~/.bashrc) to your Hugging Face cache."; exit 1; }
PROJECT_ROOT=$(pwd) # expects to be run from root, always.
# Makes CUSI's packages (cusi.utils, benchmark_adapters) importable from submodule
# scripts such as android_world/run.py.
export PYTHONPATH="${PROJECT_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
# uv's standalone Python looks for CA certificates at /etc/ssl/cert.pem, which some
# distros (e.g. AlmaLinux) lack, so HTTPS downloads fail. Use certifi's bundle.
SSL_CERT_FILE=$(python -c "import certifi; print(certifi.where())") || { echo "certifi not found in the venv."; exit 1; }
export SSL_CERT_FILE
export REQUESTS_CA_BUNDLE="$SSL_CERT_FILE"
# The CUSI benchmark container (Android emulator + Chromium); see scripts/container.sh.
export CUSI_SIF="${storage_dir%/}/containers/cusi.sif"
# uv's downloads and Python installs live under storage_dir (see setup/setup.sh).
export UV_CACHE_DIR="${storage_dir%/}/uv/cache"
export UV_PYTHON_INSTALL_DIR="${storage_dir%/}/uv/python"
if [[ -x "${storage_dir%/}/uv/bin/uv" ]]; then
    export PATH="${storage_dir%/}/uv/bin:$PATH"
fi

# args_to_flags <assoc_array_name>
#
# Converts a bash associative array into a flat --key value string suitable
# for passing to a Python click command or another bash script.
# Empty values ("") are emitted as --key none.
#
# Usage (capture-safe — all diagnostics go to stderr):
#   declare -A ARGS=( ["lr"]="0.001" ["dataset"]="" )
#   flags=$(args_to_flags ARGS)
#   python get_strings.py <string_kind> $flags
function args_to_flags() {
    local -n _dict="$1"
    local result=""
    for key in "${!_dict[@]}"; do
        local val="${_dict[$key]}"
        if [[ -z "$val" ]]; then
            val="none"
        fi
        result+="--${key} ${val} "
    done
    echo "${result% }"  # trim trailing space
}

# get_string_from_args <string_kind> <assoc_array_name>
#
# Utility function to get a string from get_strings.py by passing an associative array of args.
# Usage:
#   string=$(get_string_from_args <string_kind> ARGS)
function get_string_from_args() {
    local string_kind="$1"
    shift
    local flags=$(args_to_flags "$1")
    python ${PROJECT_ROOT}/scripts/get_strings.py "$string_kind" $flags
}


# args_to_flags_subset <assoc_array_name> "${KEY_LIST[@]}"
#
# Like args_to_flags, but only emits flags for the specified keys.
# Keys not present in the array are silently skipped.
# Use this when calling a subscript that doesn't accept all of the caller's ARGS.
#
# Usage:
#   subset=$(args_to_flags_subset ARGS REQUESTED_KEYS_ARRAY)
#   bash scripts/a.sh $subset
function args_to_flags_subset() {
    local -n _dict="$1"
    local -n _keys="$2"
    local result=""
    for key in "${_keys[@]}"; do
        if [[ -v _dict["$key"] ]]; then
            local val="${_dict[$key]}"
            if [[ -z "$val" ]]; then val="none"; fi
            result+="--${key} ${val} "
        fi
    done
    echo "${result% }"
}

function populate_dict(){
    local -n _source_dict="$1"
    local -n _target_dict="$2"
    for key in "${!_source_dict[@]}"; do
        _target_dict["$key"]="${_source_dict[$key]}"
    done
}

function populate_array(){
    local -n _source_arr="$1"
    local -n _target_arr="$2"
    _target_arr+=("${_source_arr[@]}")
}

function populate_array_subset(){
    local -n _source_arr="$1"
    local -n _target_arr="$2"
    local -n _subset_keys="$3"
    for key in "${_subset_keys[@]}"; do
        for val in "${_source_arr[@]}"; do
            if [[ "$val" == "$key" ]]; then
                _target_arr+=("$val")
                break
            fi
        done
    done
}

function populate_dict_subset(){
    local -n _source_dict="$1"
    local -n _target_dict="$2"
    local -n _subset_keys="$3"
    for key in "${_subset_keys[@]}"; do
        if [[ -v _source_dict["$key"] ]]; then
            _target_dict["$key"]="${_source_dict[$key]}"
        fi
    done
}


############################################ Example Usage Below ###################################
# Delete this and replace with your own stuff for a new project

declare -A COMMON_OPTIONAL_TRAINING_ARGS_DEFAULTS=(["num_epochs"]="10")
function populate_common_optional_training_args() {
    populate_dict COMMON_OPTIONAL_TRAINING_ARGS_DEFAULTS "$1"
}

COMMON_REQUIRED_TRAINING_ARGS=("dataset" "model")
function populate_common_required_training_args() {
    populate_array COMMON_REQUIRED_TRAINING_ARGS "$1"
}

# Combined key list for use with args_to_flags_subset when calling a subscript
COMMON_TRAINING_ARGS_KEYS=("${COMMON_REQUIRED_TRAINING_ARGS[@]}" "${!COMMON_OPTIONAL_TRAINING_ARGS_DEFAULTS[@]}")
# parse_args <assoc_array_name> <required_array_name> "$@"
#
# The BASH_TEMPLATE.md argument parser as one function: validates that optional
# defaults are non-blank, accepts only --key value pairs for known keys, checks the
# required ones, and prints the active variables. Exits with the usage string on error.
#   declare -A ARGS=( ["port"]="8000" ); REQUIRED_ARGS=("env")
#   parse_args ARGS REQUIRED_ARGS "$@"
function parse_args() {
    local -n _pa_args="$1"
    local -n _pa_req="$2"
    shift 2
    local usage_str="Usage: $0"
    local req opt
    for req in "${_pa_req[@]}"; do usage_str+=" --$req <value>"; done
    for opt in "${!_pa_args[@]}"; do
        if [[ ! " ${_pa_req[*]} " =~ " ${opt} " ]]; then
            if [[ -z "${_pa_args[$opt]}" ]]; then
                echo "DEFAULT VALUE OF KEY \"$opt\" CANNOT BE BLANK"; exit 1
            fi
            usage_str+=" [--$opt <value> (default: ${_pa_args[$opt]})]"
        fi
    done
    while [[ $# -gt 0 ]]; do
        case "$1" in
            -h|--help) echo "$usage_str"; exit 1 ;;
            --*)
                local flag=${1#--}
                if [[ ! " ${_pa_req[*]} ${!_pa_args[*]} " =~ " ${flag} " ]]; then
                    echo "Error: Unknown flag --$flag"; echo "$usage_str"; exit 1
                fi
                _pa_args["$flag"]="$2"
                shift 2
                ;;
            *) echo "Unknown argument: $1"; echo "$usage_str"; exit 1 ;;
        esac
    done
    local failed=false
    for req in "${_pa_req[@]}"; do
        if [[ -z "${_pa_args[$req]}" ]]; then echo "Error: Argument --$req is required."; failed=true; fi
    done
    if [[ "$failed" == true ]]; then echo "$usage_str"; exit 1; fi
    echo "Script: $0 Active variables:"
    for opt in "${!_pa_args[@]}"; do echo "  -$opt = ${_pa_args[$opt]}"; done
}

################################################################################
# Script defaults: every bash script's optional arguments (<NAME>_DEFAULTS) and required ones
# (<NAME>_ESSENTIALS). Scripts load them with populate_dict / populate_array before parse_args.
################################################################################

# scripts/android_emulator.sh
ANDROID_EMULATOR_ESSENTIALS=("action")
declare -A ANDROID_EMULATOR_DEFAULTS=(
    ["avd"]="AndroidWorldAvd"
    ["port"]="5554"
    ["grpc_port"]="8554"
    ["read_only"]="true"
    ["snapshots"]="false"
    ["gpu"]="off"
    ["boot_timeout"]="600"
)

# scripts/commit_all.sh
COMMIT_ALL_ESSENTIALS=()
declare -A COMMIT_ALL_DEFAULTS=(
    ["message"]="Update"
    ["push"]="true"
)

# scripts/serve_vllm.sh
SERVE_VLLM_ESSENTIALS=("model")
declare -A SERVE_VLLM_DEFAULTS=(
    ["served_model_name"]="none"
    ["port"]="$vllm_port"
    ["tp"]="none"
    ["max_model_len"]="$vllm_max_model_len"
    ["max_images"]="$vllm_max_images"
    ["gpu_memory_utilization"]="0.90"
    ["extra"]="none"
)

# scripts/stop_vllm.sh
STOP_VLLM_ESSENTIALS=()
declare -A STOP_VLLM_DEFAULTS=(
    ["port"]="$vllm_port"
)

# scripts/submit_ablation.sh
SUBMIT_ABLATION_ESSENTIALS=("models" "judge_model")
declare -A SUBMIT_ABLATION_DEFAULTS=(
    ["prefix"]="abl"
    ["time"]="24:00:00"
    ["time_min"]="2:00:00"   # Slurm may shorten a job's limit to this, e.g. to fit before a maintenance window
    ["envs"]="gameboy android web"
    ["supervisors"]="baseline revision subgoal"
    ["gameboy_workers"]="8"
    ["web_workers"]="4"
    ["android_emulators"]="3"
    ["partition"]="medium-lg"
)

# scripts/slurm/curiosity_practice.sh
CURIOSITY_PRACTICE_ESSENTIALS=("env" "run" "model")
declare -A CURIOSITY_PRACTICE_DEFAULTS=(
    ["max_groups"]="6"
    ["z_min"]="3.0"
    ["outlier"]="2.5"
    ["rescore"]="none"
)

# scripts/slurm/eval.sh
EVAL_ESSENTIALS=("env" "model" "run")
declare -A EVAL_DEFAULTS=(
    ["supervisor"]="baseline"
    ["workers"]="1"
    ["n_tasks"]="none"
    ["tasks"]="none"
    ["task_file"]="cusi"
    ["judge_model"]="none"
    ["judge_only"]="false"
    ["n_emulators"]="1"
    ["vllm_port"]="none"
    ["overwrite"]="false"
    ["ignore_config_violation"]="false"
)

# scripts/slurm/eval_parity_android.sh
EVAL_PARITY_ANDROID_ESSENTIALS=("model" "run")
declare -A EVAL_PARITY_ANDROID_DEFAULTS=(
    ["tasks"]="CameraTakePhoto,ClockStopWatchRunning,ClockTimerEntry,ContactsAddContact,ExpenseDeleteSingle,MarkorCreateFolder,MarkorDeleteNewestNote,OpenAppTaskEval,RecipeDeleteSingleRecipe,SimpleCalendarNextEvent,SystemBluetoothTurnOff,SystemBluetoothTurnOn,SystemBrightnessMax,SystemWifiTurnOff,SystemWifiTurnOn,TasksDueOnDate"
    ["runs"]="native ours native2"
    ["vllm_port"]="8037"
)

# scripts/slurm/eval_parity_gameboy.sh
EVAL_PARITY_GAMEBOY_ESSENTIALS=("model")
declare -A EVAL_PARITY_GAMEBOY_DEFAULTS=(
    ["n_tasks"]="20"
    ["max_steps"]="50"
    ["run"]="parity"
    ["greedy"]="true"
    ["self_check"]="false"
)

# scripts/slurm/eval_parity_web.sh
EVAL_PARITY_WEB_ESSENTIALS=("model")
declare -A EVAL_PARITY_WEB_DEFAULTS=(
    ["sites"]="ArXiv,GitHub,Huggingface"
    ["per_site"]="8"
    ["native_runs"]="2"
    ["port"]="8011"
)

# scripts/slurm/explore.sh
EXPLORE_ESSENTIALS=("env" "scene" "run" "policy_model" "curiosity_module" "image_embedder" "text_embedder")
declare -A EXPLORE_DEFAULTS=(
    ["total_steps"]="1536"
    ["wm"]="false"
    ["tasks"]="false"
    ["extra"]="none"
    ["model"]="none"
)

# scripts/slurm/policy_model_check.sh
POLICY_MODEL_CHECK_ESSENTIALS=("models")
declare -A POLICY_MODEL_CHECK_DEFAULTS=(
)

# scripts/slurm/practice_small.sh
PRACTICE_SMALL_ESSENTIALS=("model")
declare -A PRACTICE_SMALL_DEFAULTS=(
    ["envs"]="gameboy web android"
)

# scripts/slurm/vllm_smoke.sh
VLLM_SMOKE_ESSENTIALS=("model")
declare -A VLLM_SMOKE_DEFAULTS=(
)

# setup/android/app_setup.sh
ANDROID_APP_SETUP_ESSENTIALS=()
declare -A ANDROID_APP_SETUP_DEFAULTS=(
    ["avd"]="AndroidWorldAvd"
    ["force"]="false"
)

# setup/android/create_avd.sh
ANDROID_CREATE_AVD_ESSENTIALS=()
declare -A ANDROID_CREATE_AVD_DEFAULTS=(
    ["avd"]="AndroidWorldAvd"
    ["force"]="false"
)

# setup/android/inspect_apps.sh
ANDROID_INSPECT_APPS_ESSENTIALS=()
declare -A ANDROID_INSPECT_APPS_DEFAULTS=(
    ["apps"]="chrome contacts markor"
    ["wait"]="20"
    ["out_dir"]="none"
    # Extended regex; if any app's screen text matches it, exit 1 (after inspecting all).
    ["fail_on"]="none"
)

# setup/container/build.sh
CONTAINER_BUILD_ESSENTIALS=()
declare -A CONTAINER_BUILD_DEFAULTS=(
    ["force"]="false"
)

# setup/verify/e2e.sh
VERIFY_E2E_ESSENTIALS=()
declare -A VERIFY_E2E_DEFAULTS=(
    ["webvoyager"]="true"
    ["android_world"]="true"
)
