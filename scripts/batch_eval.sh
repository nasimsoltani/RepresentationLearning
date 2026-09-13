#!/bin/bash

# --- resolved from .env (see .env.example) ---
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [ -f "$REPO_ROOT/.env" ]; then set -a; . "$REPO_ROOT/.env"; set +a; fi
: "${DATA_BASE_PATH:?set DATA_BASE_PATH in .env (see .env.example)}"
: "${PKL_FILE_PATH:?set PKL_FILE_PATH in .env (see .env.example)}"
DATASET_PATH="${DATASET_PATH:-$PKL_FILE_PATH/rf_partition_dict_${PORTION_TO_USE:-0.5}.pkl}"
# ---------------------------------------------


# =================================================================================
# Batch Evaluation Script for Representation Learning Models
# =================================================================================
#
# This script automatically finds and evaluates all trained models within a
# top-level results directory.
#
# Usage:
# 1. Set the `RESULTS_DIR` variable to the main results folder (e.g., 'results_20250616_190950').
# 2. Adjust GPU ID and test fraction as needed.
#
# The script will search for the '*_best.pt' file in each subdirectory and run
# `code/rep_lr/eval.py` on it.
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
RESULTS_DIR="$REPO_ROOT/results_20250702_140113" # <-- CHANGE THIS

# IMPORTANT: SET THIS to the path of the .pkl dataset to use for all evaluations.
EVAL_DATASET_PATH="${EVAL_DATASET_PATH:-$DATASET_PATH}"

GPU_ID=0
TEST_FRACTION=1

# =================================================================================
# Run Batch Evaluation
# =================================================================================

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

echo "Starting batch evaluation for models in: $RESULTS_DIR"
echo "============================================================"

# Loop through each subdirectory in the results directory
for experiment_dir in "$RESULTS_DIR"/*/; do
    if [ -d "$experiment_dir" ]; then
        echo "Searching for model in: $experiment_dir"

        # Find the model checkpoint file (e.g., *_best.pt)
        # Using -quit to stop after the first match
        MODEL_PATH=$(find "$experiment_dir" -name "*_best.pt" -print -quit)

        if [ -n "$MODEL_PATH" ] && [ -f "$MODEL_PATH" ]; then
            echo "--- Found model. Evaluating: $MODEL_PATH ---"

            # Run the evaluation script
            python code/rep_lr/eval.py \
                --model_path "$MODEL_PATH" \
                --eval_pkl_dataset_path "$EVAL_DATASET_PATH" \
                --gpu_id $GPU_ID \
                --test_fraction $TEST_FRACTION

            echo "--- Finished evaluation for $MODEL_PATH ---"
            echo "============================================================"
        else
            echo "Warning: No '*_best.pt' model found in $experiment_dir. Skipping."
            echo "============================================================"
        fi
    fi
done

echo "Batch evaluation finished." 