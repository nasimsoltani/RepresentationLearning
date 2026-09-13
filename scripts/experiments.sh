#!/bin/bash

# --- resolved from .env (see .env.example) ---
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [ -f "$REPO_ROOT/.env" ]; then set -a; . "$REPO_ROOT/.env"; set +a; fi
: "${DATA_BASE_PATH:?set DATA_BASE_PATH in .env (see .env.example)}"
: "${PKL_FILE_PATH:?set PKL_FILE_PATH in .env (see .env.example)}"
DATASET_PATH="${DATASET_PATH:-$PKL_FILE_PATH/rf_partition_dict_${PORTION_TO_USE:-0.5}.pkl}"
# ---------------------------------------------


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
GPU_ID=0

# --- Create a Unique Directory ---
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# Check if a run ID was passed as the first argument ($1)
if [ -n "$1" ]; then
    # If an argument is provided, create a subdirectory for this specific run.
    RUN_ID=$1
    RESULTS_DIR="results_${TIMESTAMP}/run_${RUN_ID}"
    echo "Run ID '$RUN_ID' provided. Results will be saved in: ${RESULTS_DIR}"
else
    # If no argument is provided, use the original behavior.
    RESULTS_DIR="results_${TIMESTAMP}"
    echo "No Run ID provided. Results will be saved in: ${RESULTS_DIR}"
fi

# The -p flag is crucial here. It creates parent directories as needed.
mkdir -p "${RESULTS_DIR}"

# Common model parameters
PROJ_SEQ_LEN=256
PROJ_HIDDEN_DIM=512
D2=256
HEAD_HIDDEN_DIM=256
BATCH_SIZE=512
ENCODER_NUM_BLOCKS=2
FUSION_TYPE=depth_concat

echo "Debug: ENCODER_NUM_BLOCKS=$ENCODER_NUM_BLOCKS"
echo "Debug: FUSION_TYPE=$FUSION_TYPE"

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
#     --dropout 0.1 \
#     --task_adaptive_encoder \
#     --encoder_num_blocks $ENCODER_NUM_BLOCKS \
#     --rf_fixed \
#     --fusion_type $FUSION_TYPE \ 
    # --rf_begin_idx 0

# # =================================================================================
# # 2. Single-Task: CFO Estimation
# # =================================================================================
# echo "--- Starting Single-Task: CFO Estimation ---"
# python code/rep_lr/main.py \
#     --task cfo_estimation \
#     --pkl_dataset_path $DATASET_PATH \
#     --save_path "${RESULTS_DIR}/cfo" \
#     --epochs 300 \
#     --batch_size $BATCH_SIZE \
#     --lr 1e-3 \
#     --proj_seq_len $PROJ_SEQ_LEN \
#     --proj_hidden_dim $PROJ_HIDDEN_DIM \
#     --d2 $D2 \
#     --head_hidden_dim $HEAD_HIDDEN_DIM \
#     --gpu_id $GPU_ID \
#     --task_adaptive_encoder \
#     --encoder_num_blocks $ENCODER_NUM_BLOCKS \
#     --fusion_type $FUSION_TYPE \
#     --rf_fixed \

# # =================================================================================
# # 3. Single-Task: Channel Estimation
# # =================================================================================
# echo "--- Starting Single-Task: Channel Estimation ---"
# uv run python code/rep_lr/main.py \
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
#     --gpu_id $GPU_ID \
#     --task_adaptive_encoder \
#     --encoder_num_blocks $ENCODER_NUM_BLOCKS \
#     --fusion_type $FUSION_TYPE \
#     --rf_fixed \

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
#     --batch_size $BATCH_SIZE \
#     --lr 1e-4 \
#     --w_rf 1.0 \
#     --w_cfo 1.5 \
#     --proj_seq_len $PROJ_SEQ_LEN \
#     --proj_hidden_dim $PROJ_HIDDEN_DIM \
#     --d2 $D2 \
#     --head_hidden_dim $HEAD_HIDDEN_DIM \
#     --gpu_id $GPU_ID \
#     --task_adaptive_encoder \
#     --encoder_num_blocks $ENCODER_NUM_BLOCKS \
#     --fusion_type $FUSION_TYPE \
#     --rf_fixed \


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
#     --gpu_id $GPU_ID \
#     --task_adaptive_encoder \
#     --encoder_num_blocks $ENCODER_NUM_BLOCKS \
#     --fusion_type $FUSION_TYPE \
#     --rf_fixed \

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
#     --gpu_id $GPU_ID \
#     --task_adaptive_encoder \
#     --encoder_num_blocks $ENCODER_NUM_BLOCKS \
#     --fusion_type $FUSION_TYPE \
#     --rf_fixed \



# # =================================================================================
# # 7. Multi-Task (MTL): RF + CFO + Channel
# # =================================================================================
echo "--- Starting MTL: RF Fingerprinting + CFO Estimation + Channel Estimation ---"
python code/rep_lr/main.py \
    --mtl \
    --task rf_fingerprinting cfo_estimation channel_estimation \
    --pkl_dataset_path $DATASET_PATH \
    --data_root "$DATA_BASE_PATH" \
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
    --gpu_id $GPU_ID \
    --task_adaptive_encoder \
    --encoder_num_blocks $ENCODER_NUM_BLOCKS \
    --fusion_type $FUSION_TYPE \
    --rf_fixed \

echo "--- All experiments finished ---" 