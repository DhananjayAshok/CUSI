source configs/config.env || { echo "configs/config.env not found"; exit 1; }
source setup/.venv/bin/activate || { echo "Virtual environment not found."; exit 1; }
PROJECT_ROOT=$(pwd) # expects to be run from root, always. 

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