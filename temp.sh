#!/bin/bash

# --- Configuration ---
# Base path where the result directories for each model are stored.
RESULTS_BASE="/work/10608/aadharsh_aadhithya/vista/RepresentationLearning/results_20250730_220537_7"
# Base path where activations are stored.
ACTIVATIONS_BASE="/scratch/10608/aadharsh_aadhithya/data/rep_lr/activations_exps"
# Path to the attack script.
ATTACK_SCRIPT="code/dra_1/robust_attack_1.py"
# Timestamp for creating unique attack result directories.
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

# --- Helper Functions ---

# Maps experiment names to activation directory names
map_exp_to_activation_dir() {
    local exp_name=$1
    case "$exp_name" in
        "cfo_estimation") echo "cfo" ;;
        "cfo_estimation_channel_estimation") echo "cfo_channel" ;;
        "channel_estimation") echo "channel" ;;
        "rf_fingerprinting") echo "rf" ;;
        "rf_fingerprinting_cfo_estimation") echo "rf_cfo" ;;
        "rf_fingerprinting_cfo_estimation_channel_estimation") echo "rf_cfo_channel" ;;
        "rf_fingerprinting_channel_estimation") echo "rf_channel" ;;
        *) echo "$exp_name" ;; # fallback to original name
    esac
}

# Finds the first subdirectory within a given path, which is assumed to be the timestamped experiment directory.
find_experiment_dir() {
    local task_path=$1
    for item in "$task_path"/*; do
        if [ -d "$item" ]; then
            echo "$item"
            return
        fi
    done
    echo "" # Return empty string if no directory is found.
}

# Calculates the latent dimension by reading the model's training arguments from args.json.
# This avoids having to hardcode the latent dimension for each model.
calculate_latent_dim() {
    local exp_path=$1
    local args_file="$exp_path/args.json"
    
    if [ ! -f "$args_file" ]; then
        echo "512" # Fallback to a default value if args.json is not found.
        return
    fi
    
    # Use jq to safely parse JSON and provide default values if keys are missing.
    local d2=$(jq '.d2 // 128' "$args_file")
    # The latent dimension is consistently 2 * d2 across the models.
    echo $((2 * d2))
}

# --- Main Attack Function ---
# This function assembles and launches a single attack configuration.
run_attack() {
    local exp_name=$1
    local task=$2
    local noise_type=$3
    local noise_level=$4
    local leaked_fraction=$5
    local lambda_factor=$6

    echo "--- Preparing attack for: Experiment=${exp_name}, Task=${task}, Noise=${noise_type}, Level=${noise_level}, Lambda=${lambda_factor} ---"

    local task_path="${RESULTS_BASE}/${exp_name}"
    local full_experiment_path=$(find_experiment_dir "$task_path")

    if [ -z "$full_experiment_path" ]; then
        echo "ERROR: Could not find experiment directory in ${task_path}. Skipping."
        return
    fi

    local latent_dim=$(calculate_latent_dim "$full_experiment_path")
    local results_name=$(basename "$RESULTS_BASE")
    local activation_dir_name=$(map_exp_to_activation_dir "$exp_name")
    local activations_path="${ACTIVATIONS_BASE}/${results_name}/${activation_dir_name}"
    
    # Check if activations exist
    if [ ! -d "$activations_path" ]; then
        echo "ERROR: Activations path ${activations_path} does not exist. Skipping this attack."
        return
    fi
    
    echo "Using activations from: ${activations_path}"

    # Construct the nested output directory path.
    local frac_str=$(echo "$leaked_fraction" | tr '.' '_')
    local level_str=$(echo "$noise_level" | tr '.' '_')
    local base_attack_dir="${full_experiment_path}/attack_results_robust/attack_${TIMESTAMP}"
    local results_dir="${base_attack_dir}/${task}/${noise_type}/leaked_frac_${frac_str}/noise_level_${level_str}"

    # Assemble the command arguments in an array for robustness.
    local cmd_args=(
        "python" "$ATTACK_SCRIPT"
        "--experiment_path" "$full_experiment_path"
        "--activations_path" "$activations_path"
        "--task" "$task"
        "--noise_type" "$noise_type"
        "--noise_level" "$noise_level"
        "--leaked_fraction" "$leaked_fraction"
        "--latent_dim" "$latent_dim"
        "--epochs" "40"
        "--patience" "30"
        "--lr" "1e-3"
        "--batch_size" "128"
        "--optimizer" "adamw"
        "--clip_grad_norm" "1.0"
        "--use_lr_scheduler"
        "--fim_samples" "8000"
    )

    if [ "$noise_type" = "nonisotropic" ]; then
        local lambda_str="lambda_$(echo "$lambda_factor" | tr '.' '_')"
        results_dir="${results_dir}/${lambda_str}"
        cmd_args+=("--lambda_factor" "$lambda_factor")
    fi
    
    cmd_args+=("--output_dir" "$results_dir")

    # Create the results directory and run the command in the background.
    mkdir -p "$results_dir"
    local log_file="${results_dir}/attack.log"

    echo "COMMAND: ${cmd_args[@]}"
    echo "LOG FILE: $log_file"
    
    "${cmd_args[@]}" > "$log_file" 2>&1 &
}


# --- Define and Run Attacks in Parallel ---
echo "Launching all specified attack configurations in parallel..."

# 1. cfo_estimation_channel_estimation, leaked fraction 1, isotropic, level 15
run_attack "cfo_estimation_channel_estimation" "cfo" "isotropic" 15 1.0 ""
run_attack "cfo_estimation_channel_estimation" "channel" "isotropic" 15 1.0 ""

# 2. rf_fingerprinting, leaked fraction 1, nonisotropic, level 15, lambda 0.01
run_attack "rf_fingerprinting" "rf" "nonisotropic" 15 1.0 "0.01"

# 3. rf_fingerprinting, leaked fraction 1, nonisotropic, level 10, lambda 0.01
run_attack "rf_fingerprinting" "rf" "nonisotropic" 10 1.0 "0.01"

# 4. rf_fingerprinting, leaked fraction 1, isotropic, level 20
run_attack "rf_fingerprinting" "rf" "isotropic" 20 1.0 ""

# 5. rf_fingerprinting, leaked fraction 1, isotropic, level 5
run_attack "rf_fingerprinting" "rf" "isotropic" 5 1.0 ""

# 6. rf_channel, leaked fraction 1, nonisotropic, level 10, lambda 0.01
run_attack "rf_channel" "rf" "nonisotropic" 10 1.0 "0.01"
run_attack "rf_channel" "channel" "nonisotropic" 10 1.0 "0.01"

# Wait for all background jobs to complete before exiting the script.
echo "All attack jobs launched. Waiting for completion..."
wait
echo "All attacks have finished."

