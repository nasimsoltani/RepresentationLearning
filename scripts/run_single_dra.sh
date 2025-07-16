#!/bin/bash

# =================================================================================
# Single Data Reconstruction Attack (DRA) Script
# =================================================================================
#
# This script runs a single data reconstruction attack experiment.
# 
# Usage:
#   bash scripts/run_single_dra.sh <noise_type> <noise_level>
#
# Examples:
#   bash scripts/run_single_dra.sh none 0.0
#   bash scripts/run_single_dra.sh isotropic 0.1
#   bash scripts/run_single_dra.sh anisotropic 0.05
#
# =================================================================================

# Parse command line arguments
NOISE_TYPE=${1:-"none"}
NOISE_LEVEL=${2:-"0.0"}

# Validate noise type
if [[ ! "$NOISE_TYPE" =~ ^(none|isotropic|anisotropic)$ ]]; then
    echo "Error: Invalid noise type '$NOISE_TYPE'"
    echo "Valid options: none, isotropic, anisotropic"
    echo ""
    echo "Usage: bash scripts/run_single_dra.sh <noise_type> <noise_level>"
    echo "Examples:"
    echo "  bash scripts/run_single_dra.sh none 0.0"
    echo "  bash scripts/run_single_dra.sh isotropic 0.1"
    echo "  bash scripts/run_single_dra.sh anisotropic 0.05"
    exit 1
fi

# --- Configuration ---
# IMPORTANT: SET THESE PATHS to your trained model and activation files
MODEL_PATH="results_20250715_202527/rf_cfo_channel/rf_fingerprinting_cfo_estimation_channel_estimation_20250715_202529/rf_fingerprinting_cfo_estimation_channel_estimation_best.pt"
ACTIVATIONS_DIR="dataset/activations/train"

# Experiment parameters
GPU_ID=0
TARGET_DATA="RF_X"  # Options: RF_X, CFO_X, Channel_X
ADVERSARY_TYPE="simple"  # Options: simple, conv, residual
EPOCHS=50
BATCH_SIZE=32
LEARNING_RATE=1e-3
TRAIN_RATIO=0.7

# Create timestamped results directory
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RESULTS_DIR="attack_results_single_${NOISE_TYPE}_${NOISE_LEVEL}_${TIMESTAMP}"

# Validate paths
if [ ! -f "$MODEL_PATH" ]; then
    echo "Error: Model file not found at '$MODEL_PATH'"
    echo "Please update the MODEL_PATH variable in this script."
    exit 1
fi

if [ ! -d "$ACTIVATIONS_DIR" ]; then
    echo "Error: Activations directory not found at '$ACTIVATIONS_DIR'"
    echo "Please update the ACTIVATIONS_DIR variable in this script."
    exit 1
fi

# =================================================================================
# Run Single DRA Experiment
# =================================================================================
echo "=============================================================="
echo "Starting DRA Experiment"
echo "=============================================================="
echo "Configuration:"
echo "  Noise Type: $NOISE_TYPE"
echo "  Noise Level: $NOISE_LEVEL"
echo "  Target Data: $TARGET_DATA"
echo "  Adversary Type: $ADVERSARY_TYPE"
echo "  Epochs: $EPOCHS"
echo "  Batch Size: $BATCH_SIZE"
echo "  Results Directory: $RESULTS_DIR"
echo "=============================================================="

uv run python code/dra/attack.py \
    --model_path "$MODEL_PATH" \
    --activations_dir "$ACTIVATIONS_DIR" \
    --target_data "$TARGET_DATA" \
    --noise_type "$NOISE_TYPE" \
    --noise_level $NOISE_LEVEL \
    --adversary_type "$ADVERSARY_TYPE" \
    --epochs $EPOCHS \
    --batch_size $BATCH_SIZE \
    --learning_rate $LEARNING_RATE \
    --train_ratio $TRAIN_RATIO \
    --gpu_id $GPU_ID \
    --save_path "$RESULTS_DIR" \
    --wandb_project "rf-dra-experiments" \
    --wandb_run_name "dra_${TARGET_DATA}_${NOISE_TYPE}_${NOISE_LEVEL}_${TIMESTAMP}" \
    --save_plots

# =================================================================================
# Results Summary
# =================================================================================
echo ""
echo "=============================================================="
echo "DRA Experiment Completed!"
echo "=============================================================="
echo "Configuration:"
echo "  Noise Type: $NOISE_TYPE"
echo "  Noise Level: $NOISE_LEVEL"
echo "  Results saved in: $RESULTS_DIR"
echo ""
echo "Check the result directory for:"
echo "  - results.json: Metrics and configuration"
echo "  - detailed_results.pkl: Full reconstruction data"
echo "  - reconstruction_plots_*.png: Visualization plots"
echo "  - best_adversary.pt: Trained adversary model"
echo ""
echo "View results on Weights & Biases:"
echo "  Project: rf-dra-experiments"
echo "  Run: dra_${TARGET_DATA}_${NOISE_TYPE}_${NOISE_LEVEL}_${TIMESTAMP}" 