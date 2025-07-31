#!/bin/bash

# Load environment if present
if [ -f .env ]; then
    echo "Loading environment variables from .env file"
    set -o allexport
    source .env
    set +o allexport
fi

# --- Configuration ---
DATASET_PATH="$PKL_FILE_PATH/rf_partition_dict_0.5.pkl"
GPU_ID=0
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

if [ -n "$1" ]; then
    RUN_ID=$1
    RESULTS_DIR="results_${TIMESTAMP}/run_${RUN_ID}"
    echo "Run ID '$RUN_ID' provided. Results will be saved in: ${RESULTS_DIR}"
else
    RESULTS_DIR="results_${TIMESTAMP}"
    echo "No Run ID provided. Results will be saved in: ${RESULTS_DIR}"
fi

mkdir -p "${RESULTS_DIR}"

PROJ_SEQ_LEN=256
PROJ_HIDDEN_DIM=512
D2=256
HEAD_HIDDEN_DIM=256
BATCH_SIZE=1024
ENCODER_NUM_BLOCKS=2
FUSION_TYPE=concat

# Helper function to run a command in background with logging
run_experiment() {
    CMD="$1"
    SAVE_PATH="$2"
    mkdir -p "$SAVE_PATH"
    LOG_FILE="$SAVE_PATH/command.log"
    echo "Launching experiment at $SAVE_PATH"
    bash -c "$CMD" > "$LOG_FILE" 2>&1 &
}

# Define each experiment
run_experiment "python code/rep_lr/main.py \
    --task rf_fingerprinting \
    --pkl_dataset_path $DATASET_PATH \
    --save_path ${RESULTS_DIR}/rf \
    --epochs 300 --batch_size 512 --lr 1e-3 \
    --proj_seq_len $PROJ_SEQ_LEN --proj_hidden_dim $PROJ_HIDDEN_DIM --d2 $D2 \
    --head_hidden_dim $HEAD_HIDDEN_DIM --gpu_id $GPU_ID --dropout 0.1 \
    --task_adaptive_encoder --encoder_num_blocks 2 --rf_fixed \
    --fusion_type $FUSION_TYPE" \
    "${RESULTS_DIR}/rf"

run_experiment "python code/rep_lr/main.py \
    --task cfo_estimation \
    --pkl_dataset_path $DATASET_PATH \
    --save_path ${RESULTS_DIR}/cfo \
    --epochs 300 --batch_size $BATCH_SIZE --lr 1e-3 \
    --proj_seq_len $PROJ_SEQ_LEN --proj_hidden_dim $PROJ_HIDDEN_DIM --d2 $D2 \
    --head_hidden_dim $HEAD_HIDDEN_DIM --gpu_id $GPU_ID \
    --task_adaptive_encoder --encoder_num_blocks $ENCODER_NUM_BLOCKS \
    --fusion_type $FUSION_TYPE --rf_fixed" \
    "${RESULTS_DIR}/cfo"

run_experiment "python code/rep_lr/main.py \
    --task channel_estimation \
    --pkl_dataset_path $DATASET_PATH \
    --save_path ${RESULTS_DIR}/channel \
    --epochs 300 --batch_size 256 --lr 1e-3 \
    --proj_seq_len $PROJ_SEQ_LEN --proj_hidden_dim $PROJ_HIDDEN_DIM --d2 $D2 \
    --head_hidden_dim $HEAD_HIDDEN_DIM --gpu_id $GPU_ID \
    --task_adaptive_encoder --encoder_num_blocks 2 --fusion_type $FUSION_TYPE \
    --rf_fixed" \
    "${RESULTS_DIR}/channel"

run_experiment "python code/rep_lr/main.py \
    --mtl --task rf_fingerprinting cfo_estimation \
    --pkl_dataset_path $DATASET_PATH \
    --save_path ${RESULTS_DIR}/rf_cfo \
    --epochs 300 --batch_size $BATCH_SIZE --lr 1e-4 \
    --w_rf 1.0 --w_cfo 1.5 \
    --proj_seq_len $PROJ_SEQ_LEN --proj_hidden_dim $PROJ_HIDDEN_DIM --d2 $D2 \
    --head_hidden_dim $HEAD_HIDDEN_DIM --gpu_id $GPU_ID \
    --task_adaptive_encoder --encoder_num_blocks $ENCODER_NUM_BLOCKS \
    --fusion_type $FUSION_TYPE --rf_fixed" \
    "${RESULTS_DIR}/rf_cfo"

run_experiment "python code/rep_lr/main.py \
    --mtl --task rf_fingerprinting channel_estimation \
    --pkl_dataset_path $DATASET_PATH \
    --save_path ${RESULTS_DIR}/rf_channel \
    --epochs 300 --batch_size 256 --lr 1e-4 \
    --w_rf 1.0 --w_channel 1.0 \
    --proj_seq_len $PROJ_SEQ_LEN --proj_hidden_dim $PROJ_HIDDEN_DIM --d2 $D2 \
    --head_hidden_dim $HEAD_HIDDEN_DIM --gpu_id $GPU_ID \
    --task_adaptive_encoder --encoder_num_blocks 2 \
    --fusion_type $FUSION_TYPE --rf_fixed" \
    "${RESULTS_DIR}/rf_channel"

run_experiment "python code/rep_lr/main.py \
    --mtl --task cfo_estimation channel_estimation \
    --pkl_dataset_path $DATASET_PATH \
    --save_path ${RESULTS_DIR}/cfo_channel \
    --epochs 300 --batch_size 256 --lr 1e-4 \
    --w_cfo 1.0 --w_channel 1.0 \
    --proj_seq_len $PROJ_SEQ_LEN --proj_hidden_dim $PROJ_HIDDEN_DIM --d2 $D2 \
    --head_hidden_dim $HEAD_HIDDEN_DIM --gpu_id $GPU_ID \
    --task_adaptive_encoder --encoder_num_blocks 2 \
    --fusion_type $FUSION_TYPE --rf_fixed" \
    "${RESULTS_DIR}/cfo_channel"

run_experiment "python code/rep_lr/main.py \
    --mtl --task rf_fingerprinting cfo_estimation channel_estimation \
    --pkl_dataset_path $DATASET_PATH \
    --save_path ${RESULTS_DIR}/rf_cfo_channel \
    --epochs 300 --batch_size 256 --lr 1e-4 \
    --w_rf 1.0 --w_cfo 1.0 --w_channel 1.0 \
    --proj_seq_len $PROJ_SEQ_LEN --proj_hidden_dim $PROJ_HIDDEN_DIM --d2 $D2 \
    --head_hidden_dim $HEAD_HIDDEN_DIM --gpu_id $GPU_ID \
    --task_adaptive_encoder --encoder_num_blocks 2 \
    --fusion_type $FUSION_TYPE --rf_fixed" \
    "${RESULTS_DIR}/rf_cfo_channel"

# Wait for all jobs to finish
wait

echo "--- All parallel experiments completed ---"
