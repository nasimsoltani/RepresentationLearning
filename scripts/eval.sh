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
# 2. Adjust GPU ID and test fraction as needed.

# --- Configuration ---
# IMPORTANT: SET THIS PATH to the '.pt' file of your trained model.
# This should be a file like `..._best.pt` or `..._epoch_xx.pt` from your results folder.
MODEL_PATH="/home/hofmann/Documents/projects/RepresentationLearning/results/single_task/cfo_light/cfo_estimation_20250611_030632/cfo_estimation_best.pt" # <-- CHANGE THIS

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

echo "Evaluating model: $MODEL_PATH"
uv run python code/rep_lr/eval.py \
    --model_path "$MODEL_PATH" \
    --gpu_id 0 \
    --test_fraction 0.1