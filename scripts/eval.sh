#!/bin/bash

# --- resolved from .env (see .env.example) ---
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [ -f "$REPO_ROOT/.env" ]; then set -a; . "$REPO_ROOT/.env"; set +a; fi
: "${DATA_BASE_PATH:?set DATA_BASE_PATH in .env (see .env.example)}"
: "${PKL_FILE_PATH:?set PKL_FILE_PATH in .env (see .env.example)}"
DATASET_PATH="${DATASET_PATH:-$PKL_FILE_PATH/rf_partition_dict_${PORTION_TO_USE:-0.5}.pkl}"
# ---------------------------------------------


# =================================================================================
# Evaluation Script for Representation Learning Models
# =================================================================================
#
# This script runs evaluation using `code/rep_lr/eval.py`.
# It automatically detects the task(s) the model was trained on from the
# `args.json` file in the model's directory.
#
# Usage:
# 1. Set the `MODEL_PATH` variable to the checkpoint file (.pt) you want to evaluate.
# 2. Set the `EVAL_DATASET_PATH` to the .pkl file you want to use for evaluation.
# 3. Adjust GPU ID and test fraction as needed.

# --- Configuration ---
# IMPORTANT: SET THIS PATH to the '.pt' file of your trained model.
# This should be a file like `..._best.pt` or `..._epoch_xx.pt` from your results folder.
MODEL_PATH="$REPO_ROOT/results_20250720_134858" # <-- CHANGE THIS

# IMPORTANT: SET THIS to the path of the .pkl dataset to use for this evaluation.
EVAL_DATASET_PATH="${EVAL_DATASET_PATH:-$DATASET_PATH}"

# =================================================================================
# Run Evaluation
# =================================================================================
# The script will automatically run evaluation for all tasks the model was
# trained on.
# =================================================================================
if [ ! -f "$MODEL_PATH" ]; then
    echo "Error: Model file not found at '$MODEL_PATH'"
    echo "Please update the MODEL_PATH variable in this script."
    exit 1
fi

if [ ! -f "$EVAL_DATASET_PATH" ]; then
    echo "Error: Evaluation dataset not found at '$EVAL_DATASET_PATH'"
    echo "Please update the EVAL_DATASET_PATH variable in this script."
    exit 1
fi

echo "Evaluating model: $MODEL_PATH"
uv run python code/rep_lr/eval.py \
    --model_path "$MODEL_PATH" \
    --eval_pkl_dataset_path "$EVAL_DATASET_PATH" \
    --gpu_id 0 \
    --test_fraction 0.1