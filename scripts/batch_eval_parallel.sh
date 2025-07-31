#!/bin/bash

# =================================================================================
# Parallel Batch Evaluation Script for Representation Learning Models
# =================================================================================
#
# This script automatically finds and evaluates all trained models within a
# top-level results directory, running the evaluations in parallel.
#
# Usage:
# 1. Set the `RESULTS_DIR` variable to the main results folder (e.g., 'results_20250616_190950').
# 2. Adjust `GPU_ID` and other parameters as needed. All evaluations will be
#    launched in parallel on the specified GPU.
# 3. Evaluation logs for each model will be saved as `eval.log` in their respective
#    experiment directories.
#
# The script will search for the '*_best.pt' file in each subdirectory and run
# `code/rep_lr/eval.py` on it in the background.
#
# =================================================================================

# --- Configuration ---

if [ -f .env ]; then
    echo "Loading environment variables from .env file"
    set -o allexport
    source .env
    set +o allexport
fi

# IMPORTANT: SET THIS PATH to the top-level directory containing all your experiment results.
RESULTS_DIR="/work/10608/aadharsh_aadhithya/vista/RepresentationLearning/results_20250730_220537" # <-- CHANGE THIS

# IMPORTANT: SET THIS to the path of the .pkl dataset to use for all evaluations.
EVAL_DATASET_PATH="$PKL_FILE_PATH/rf_partition_dict_0.5.pkl" 
#EVAL_DATASET_PATH="/home/hofmann/Documents/projects/RepresentationLearning/dataset/rf_partition_dict_0.5.pkl" # <-- CHANGE THIS

GPU_ID=0
TEST_FRACTION=1

# =================================================================================
# Run Batch Evaluation
# =================================================================================

# Helper function to run evaluation in background with logging
run_evaluation() {
    CMD="$1"
    EVAL_DIR="$2"
    LOG_FILE="$EVAL_DIR/eval.log"
    echo "Launching evaluation for model in: $EVAL_DIR. See log at $LOG_FILE"
    bash -c "$CMD" > "$LOG_FILE" 2>&1 &
}


if [ ! -d "$RESULTS_DIR" ]; then
    echo "Error: Results directory not found at '$RESULTS_DIR'"
    echo "Please update the RESULTS_DIR variable in this script."
    exit 1
fi

if [ ! -f "$EVAL_DATASET_PATH" ]; then
    echo "Error: Evaluation dataset not found at '$EVAL_DATASET_PATH'"
    echo "Please update the EVAL_DATASET_PATH variable in this script."
    exit 1
fi

echo "Starting parallel batch evaluation for models in: $RESULTS_DIR"
echo "============================================================"

# Loop through each subdirectory in the results directory
for experiment_dir in "$RESULTS_DIR"/*/; do
    if [ -d "$experiment_dir" ]; then
        echo "Searching for model in: $experiment_dir"

        # Find the model checkpoint file (e.g., *_best.pt)
        # Using -quit to stop after the first match
        MODEL_PATH=$(find "$experiment_dir" -name "*_best.pt" -print -quit)

        if [ -n "$MODEL_PATH" ] && [ -f "$MODEL_PATH" ]; then
            CMD="python code/rep_lr/eval.py \
                --model_path \"$MODEL_PATH\" \
                --eval_pkl_dataset_path \"$EVAL_DATASET_PATH\" \
                --gpu_id $GPU_ID \
                --test_fraction $TEST_FRACTION"
            
            run_evaluation "$CMD" "$experiment_dir"
        else
            echo "Warning: No '*_best.pt' model found in $experiment_dir. Skipping."
            echo "============================================================"
        fi
    fi
done

# Wait for all background jobs to finish
wait

echo "All evaluation jobs launched. Monitor log files in respective experiment directories for progress."
echo "Batch evaluation finished."
