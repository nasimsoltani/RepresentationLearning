#!/bin/bash

# =================================================================================
# Data Reconstruction Attack (DRA) Experiment Script
# =================================================================================
#
# This script runs data reconstruction attacks with different noise configurations:
# 1. No noise (baseline)
# 2. Isotropic noise with various strength levels
# 3. Anisotropic (FIM-based) noise with various strength levels
#
# Usage:
# 1. Set the MODEL_PATH and ACTIVATIONS_DIR variables below
# 2. Adjust GPU ID and other parameters as needed
# 3. Run: bash scripts/run_dra_experiments.sh
#
# =================================================================================

# --- Configuration ---
# IMPORTANT: SET THESE PATHS to your trained model and activation files
MODEL_PATH="results_20250715_202527/rf_cfo_channel/rf_fingerprinting_cfo_estimation_channel_estimation_20250715_202529/rf_fingerprinting_cfo_estimation_channel_estimation_best.pt"
ACTIVATIONS_DIR="dataset/activations/train"

# Experiment parameters
GPU_ID=0
TARGET_DATA="RF_X"  # Options: RF_X, CFO_X, Channel_X
ADVERSARY_TYPE="simple"  # Options: simple, conv, residual
EPOCHS=100  # Increased since model now converges properly
BATCH_SIZE=32
LEARNING_RATE=5e-4  # Reduced for more stable training (was working well in test)
TRAIN_RATIO=0.7

# Noise levels to test
NOISE_LEVELS=(0.01 0.05 0.1 0.2 0.5)

# Adversary types to test (optional - set to test multiple types)
ADVERSARY_TYPES=("conv")  # Add "conv" "residual" to test all types
TEST_MULTIPLE_ADVERSARIES=false  # Set to true to test all adversary types

# Create timestamped results directory
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RESULTS_DIR="attack_results_${TIMESTAMP}"
echo "Results for this DRA experiment batch will be saved in: ${RESULTS_DIR}"

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
# Experiment 1: No Noise (Baseline)
# =================================================================================
echo "=============================================================="
echo "Starting Experiment 1: No Noise (Baseline)"
echo "=============================================================="

uv run python code/dra/attack.py \
    --model_path "$MODEL_PATH" \
    --activations_dir "$ACTIVATIONS_DIR" \
    --target_data "$TARGET_DATA" \
    --noise_type "none" \
    --noise_level 0.0 \
    --adversary_type "$ADVERSARY_TYPE" \
    --epochs $EPOCHS \
    --batch_size $BATCH_SIZE \
    --learning_rate $LEARNING_RATE \
    --train_ratio $TRAIN_RATIO \
    --gpu_id $GPU_ID \
    --save_path "${RESULTS_DIR}/no_noise" \
    --wandb_project "rf-dra-experiments" \
    --wandb_run_name "dra_${TARGET_DATA}_no_noise_${TIMESTAMP}" \
    --save_plots

echo "Completed Experiment 1: No Noise"
echo ""

# =================================================================================
# Experiment 2: Isotropic Noise with Various Strengths
# =================================================================================
echo "=============================================================="
echo "Starting Experiment 2: Isotropic Noise"
echo "=============================================================="

for noise_level in "${NOISE_LEVELS[@]}"; do
    echo "--- Running with isotropic noise level: $noise_level ---"
    
    uv run python code/dra/attack.py \
        --model_path "$MODEL_PATH" \
        --activations_dir "$ACTIVATIONS_DIR" \
        --target_data "$TARGET_DATA" \
        --noise_type "isotropic" \
        --noise_level $noise_level \
        --adversary_type "$ADVERSARY_TYPE" \
        --epochs $EPOCHS \
        --batch_size $BATCH_SIZE \
        --learning_rate $LEARNING_RATE \
        --train_ratio $TRAIN_RATIO \
        --gpu_id $GPU_ID \
        --save_path "${RESULTS_DIR}/isotropic_${noise_level}" \
        --wandb_project "rf-dra-experiments" \
        --wandb_run_name "dra_${TARGET_DATA}_iso_${noise_level}_${TIMESTAMP}" \
        --save_plots
    
    echo "Completed isotropic noise level: $noise_level"
    echo ""
done

echo "Completed Experiment 2: Isotropic Noise"
echo ""

# =================================================================================
# Experiment 3: Anisotropic (FIM-based) Noise with Various Strengths
# =================================================================================
echo "=============================================================="
echo "Starting Experiment 3: Anisotropic (FIM-based) Noise"
echo "=============================================================="

for noise_level in "${NOISE_LEVELS[@]}"; do
    echo "--- Running with anisotropic noise level: $noise_level ---"
    
    uv run python code/dra/attack.py \
        --model_path "$MODEL_PATH" \
        --activations_dir "$ACTIVATIONS_DIR" \
        --target_data "$TARGET_DATA" \
        --noise_type "anisotropic" \
        --noise_level $noise_level \
        --adversary_type "$ADVERSARY_TYPE" \
        --epochs $EPOCHS \
        --batch_size $BATCH_SIZE \
        --learning_rate $LEARNING_RATE \
        --train_ratio $TRAIN_RATIO \
        --gpu_id $GPU_ID \
        --save_path "${RESULTS_DIR}/anisotropic_${noise_level}" \
        --wandb_project "rf-dra-experiments" \
        --wandb_run_name "dra_${TARGET_DATA}_aniso_${noise_level}_${TIMESTAMP}" \
        --save_plots
    
    echo "Completed anisotropic noise level: $noise_level"
    echo ""
done

echo "Completed Experiment 3: Anisotropic Noise"
echo ""

# =================================================================================
# Summary
# =================================================================================
echo "=============================================================="
echo "All DRA experiments completed!"
echo "=============================================================="
echo "Results saved in: $RESULTS_DIR"
echo ""
echo "Summary of experiments run:"
echo "  1. No noise (baseline)"
echo "  2. Isotropic noise with levels: ${NOISE_LEVELS[*]}"
echo "  3. Anisotropic noise with levels: ${NOISE_LEVELS[*]}"
echo ""
echo "Total experiments: $((1 + 2 * ${#NOISE_LEVELS[@]}))"
echo ""
echo "Check the individual result directories for:"
echo "  - results.json: Metrics and configuration"
echo "  - detailed_results.pkl: Full reconstruction data"
echo "  - reconstruction_plots_*.png: Visualization plots"
echo "  - best_adversary.pt: Trained adversary model"
echo ""
echo "View results on Weights & Biases:"
echo "  Project: rf-dra-experiments"
echo "  Runs tagged with timestamp: $TIMESTAMP" 