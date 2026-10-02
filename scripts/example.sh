#!/usr/bin/env bash

source scripts/utils.sh || { echo "Could not source utils"; exit 1; }

# Define defaults and required args. 
# These should be specific to this script and not shared across scripts (that is handled below).
declare -A ARGS
ARGS["batch_size"]="32"
ARGS["lr"]="0.001"
ARGS["scheduler"]="none" 
ARGS["train_only"]="none"

REQUIRED_ARGS=("third")

# extend it to include the shared arguments (if any) (place the function in utils)
populate_common_optional_training_args ARGS # don't pass in $, we want to pass the name of the variable for it to modify
populate_common_required_training_args REQUIRED_ARGS # same here. Make sure to do this before allowed flags is computed. 

# Handle parsing and input errors below:
ALLOWED_FLAGS=("${REQUIRED_ARGS[@]}" "${!ARGS[@]}")

USAGE_STR="Usage: $0"

# Add Required to string
for req in "${REQUIRED_ARGS[@]}"; do
    USAGE_STR+=" --$req <value>"
done

# Add Optionals to string
for opt in "${!ARGS[@]}"; do
    # Only list if NOT in required (to avoid double listing)
    if [[ ! " ${REQUIRED_ARGS[*]} " =~ " ${opt} " ]]; then
        if [[ -z "${ARGS[$opt]}" ]]; then
            echo "DEFAULT VALUE OF KEY \"$opt\" CANNOT BE BLANK"
            exit 1
        fi
        USAGE_STR+=" [--$opt <value> (default: ${ARGS[$opt]})]"
    fi
done

function usage() {
    echo "$USAGE_STR"
    exit 1
}

# Parser
while [[ $# -gt 0 ]]; do
    case "$1" in
        --*)
            FLAG=${1#--}
            VALID=false
            for allowed in "${ALLOWED_FLAGS[@]}"; do
                if [[ "$FLAG" == "$allowed" ]]; then
                    VALID=true
                    break
                fi
            done
            if [ "$VALID" = false ]; then
                echo "Error: Unknown flag --$FLAG"
                usage
            fi
            ARGS["$FLAG"]="$2"
            shift 2
            ;;
        -h|--help)
            usage
            ;;
        *)
            echo "Unknown argument: $1"
            usage
            ;;
    esac
done

# Validation
for req in "${REQUIRED_ARGS[@]}"; do
    echo $req : "${ARGS[$req]}"
    if [[ -z "${ARGS[$req]}" ]]; then
        echo "Error: Argument --$req is required."
        FAILED=true
    fi
done

if [ "$FAILED" = true ]; then usage; fi

# Print active variables
echo "Script: $0 Active variables:"
for key in "${!ARGS[@]}"; do
    echo "  -$key = ${ARGS[$key]}"
done

# keep in mind also, if you use "none" as a default argument, then any python scripts taking this flag in will need to know to process it to a None. If thats not the case, then handle it explicitly in bash with :

# place this in the section below, not here
if [[ "${ARGS["scheduler"]}" == "none" ]]; then
    scheduler_arg=""
else
    scheduler_arg="--scheduler ${ARGS["scheduler"]}"
fi


# Put your script code below:
exp_name=$(get_string_from_args exp_name ARGS)
echo "Do something with exp: $exp_name"

# When calling a subscript that doesn't accept all of this script's ARGS,
# use args_to_flags_subset with the companion key arrays from utils.sh.
# This avoids the subscript erroring on unknown flags.
subset=$(args_to_flags_subset ARGS COMMON_TRAINING_ARGS_KEYS)
# bash scripts/other_script.sh $subset
