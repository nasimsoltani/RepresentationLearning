#!/bin/bash

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
MODEL_PATH="/home/hofmann/Documents/projects/RepresentationLearning/results_20250720_134858" # <-- CHANGE THIS

# IMPORTANT: SET THIS to the path of the .pkl dataset to use for this evaluation.
EVAL_DATASET_PATH="/home/hofmann/Documents/projects/RepresentationLearning/dataset/rf_partition_dict_0.5.pkl" # <-- CHANGE THIS

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