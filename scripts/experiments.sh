#!/bin/bash

if [ -f .env ]; then
    echo "Loading environment variables from .env file"
    set -o allexport
    source .env
    set +o allexport
fi

# =================================================================================
# Batch Experiment Script for Representation Learning Models
# =================================================================================
#
# This script runs a series of single-task and multi-task training experiments
# sequentially using `code/rep_lr/main.py`.
#
# A main results directory with a timestamp is created for each run of this script,
# and each experiment's outputs are saved in a separate sub-directory.
#
# =================================================================================

# --- Configuration ---
DATASET_PATH="$PKL_FILE_PATH/rf_partition_dict_0.5.pkl" # <-- IMPORTANT: SET THIS PATH IF DIFFERENT
GPU_ID=0 # <-- Set the GPU to use

# Generate a timestamped directory for this run
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RESULTS_DIR="results_${TIMESTAMP}"
echo "Results for this run will be saved in: ${RESULTS_DIR}"

# Common model parameters
PROJ_SEQ_LEN=256
PROJ_HIDDEN_DIM=512
D2=256
HEAD_HIDDEN_DIM=256

# # =================================================================================
# # 1. Single-Task: RF Fingerprinting
# # =================================================================================
# echo "--- Starting Single-Task: RF Fingerprinting ---"
# python code/rep_lr/main.py \
#     --task rf_fingerprinting \
#     --pkl_dataset_path $DATASET_PATH \
#     --save_path "${RESULTS_DIR}/rf" \
#     --epochs 300 \
#     --batch_size 512 \
#     --lr 1e-3 \
#     --proj_seq_len $PROJ_SEQ_LEN \
#     --proj_hidden_dim $PROJ_HIDDEN_DIM \
#     --d2 $D2 \
#     --head_hidden_dim $HEAD_HIDDEN_DIM \
#     --gpu_id $GPU_ID \
#     --dropout 0.1

# # =================================================================================
# # 2. Single-Task: CFO Estimation
# # =================================================================================
# echo "--- Starting Single-Task: CFO Estimation ---"
# uv run python code/rep_lr/main.py \
#     --task cfo_estimation \
#     --pkl_dataset_path $DATASET_PATH \
#     --save_path "${RESULTS_DIR}/cfo" \
#     --epochs 300 \
#     --batch_size 64 \
#     --lr 1e-3 \
#     --proj_seq_len $PROJ_SEQ_LEN \
#     --proj_hidden_dim $PROJ_HIDDEN_DIM \
#     --d2 $D2 \
#     --head_hidden_dim $HEAD_HIDDEN_DIM \
#     --gpu_id $GPU_ID \
#     --task_adaptive_encoder \
#     --encoder_num_blocks 2 

# # =================================================================================
# # 3. Single-Task: Channel Estimation
# # =================================================================================
# echo "--- Starting Single-Task: Channel Estimation ---"
# python code/rep_lr/main.py \
#     --task channel_estimation \
#     --pkl_dataset_path $DATASET_PATH \
#     --save_path "${RESULTS_DIR}/channel" \
#     --epochs 300 \
#     --batch_size 256 \
#     --lr 1e-3 \
#     --proj_seq_len $PROJ_SEQ_LEN \
#     --proj_hidden_dim $PROJ_HIDDEN_DIM \
#     --d2 $D2 \
#     --head_hidden_dim $HEAD_HIDDEN_DIM \
#     --gpu_id $GPU_ID

# # # =================================================================================
# # # 4. Multi-Task (MTL): RF + CFO
# # # =================================================================================
# echo "--- Starting MTL: RF Fingerprinting + CFO Estimation ---"
# python code/rep_lr/main.py \
#     --mtl \
#     --task rf_fingerprinting cfo_estimation \
#     --pkl_dataset_path $DATASET_PATH \
#     --save_path "${RESULTS_DIR}/rf_cfo" \
#     --epochs 300 \
#     --batch_size 256 \
#     --lr 1e-4 \
#     --w_rf 1.0 \
#     --w_cfo 1.5 \
#     --proj_seq_len $PROJ_SEQ_LEN \
#     --proj_hidden_dim $PROJ_HIDDEN_DIM \
#     --d2 $D2 \
#     --head_hidden_dim $HEAD_HIDDEN_DIM \
#     --gpu_id $GPU_ID
#     --task_adaptive_encoder \
#     --encoder_num_blocks 2


# # # =================================================================================
# # # 5. Multi-Task (MTL): RF + Channel
# # # =================================================================================
# echo "--- Starting MTL: RF Fingerprinting + Channel Estimation ---"
# python code/rep_lr/main.py \
#     --mtl \
#     --task rf_fingerprinting channel_estimation \
#     --pkl_dataset_path $DATASET_PATH \
#     --save_path "${RESULTS_DIR}/rf_channel" \
#     --epochs 300 \
#     --batch_size 256 \
#     --lr 1e-4 \
#     --w_rf 1.0 \
#     --w_channel 1.0 \
#     --proj_seq_len $PROJ_SEQ_LEN \
#     --proj_hidden_dim $PROJ_HIDDEN_DIM \
#     --d2 $D2 \
#     --head_hidden_dim $HEAD_HIDDEN_DIM \
#     --gpu_id $GPU_ID

# # =================================================================================
# # 6. Multi-Task (MTL): CFO + Channel
# # =================================================================================
# echo "--- Starting MTL: CFO Estimation + Channel Estimation ---"
# python code/rep_lr/main.py \
#     --mtl \
#     --task cfo_estimation channel_estimation \
#     --pkl_dataset_path $DATASET_PATH \
#     --save_path "${RESULTS_DIR}/cfo_channel" \
#     --epochs 300 \
#     --batch_size 256 \
#     --lr 1e-4 \
#     --w_cfo 1.0 \
#     --w_channel 1.0 \
#     --proj_seq_len $PROJ_SEQ_LEN \
#     --proj_hidden_dim $PROJ_HIDDEN_DIM \
#     --d2 $D2 \
#     --head_hidden_dim $HEAD_HIDDEN_DIM \
#     --gpu_id $GPU_ID
#     --task_adaptive_encoder \
#     --encoder_num_blocks 2



# # =================================================================================
# # 7. Multi-Task (MTL): RF + CFO + Channel
# # =================================================================================
echo "--- Starting MTL: RF Fingerprinting + CFO Estimation + Channel Estimation ---"
uv run python code/rep_lr/main.py \
    --mtl \
    --task rf_fingerprinting cfo_estimation channel_estimation \
    --pkl_dataset_path $DATASET_PATH \
    --save_path "${RESULTS_DIR}/rf_cfo_channel" \
    --epochs 300 \
    --batch_size 256 \
    --lr 1e-4 \
    --w_rf 1.0 \
    --w_cfo 1.0 \
    --w_channel 1.0 \
    --proj_seq_len $PROJ_SEQ_LEN \
    --proj_hidden_dim $PROJ_HIDDEN_DIM \
    --d2 $D2 \
    --head_hidden_dim $HEAD_HIDDEN_DIM \
    --gpu_id $GPU_ID
    --task_adaptive_encoder \
    --encoder_num_blocks 2

echo "--- All experiments finished ---" 