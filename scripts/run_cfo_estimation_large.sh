#!/bin/bash

# --- resolved from .env (see .env.example) ---
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [ -f "$REPO_ROOT/.env" ]; then set -a; . "$REPO_ROOT/.env"; set +a; fi
: "${DATA_BASE_PATH:?set DATA_BASE_PATH in .env (see .env.example)}"
: "${PKL_FILE_PATH:?set PKL_FILE_PATH in .env (see .env.example)}"
DATASET_PATH="${DATASET_PATH:-$PKL_FILE_PATH/rf_partition_dict_${PORTION_TO_USE:-0.5}.pkl}"
# ---------------------------------------------


# This script runs the training for the Large CFO Estimation task.
# IMPORTANT: You must specify the path to your dataset pickle file below.

uv run python $REPO_ROOT/code/rep_lr/main.py \
    --task cfo_estimation_large \
    --pkl_dataset_path "$DATASET_PATH" \
    --data_root "$DATA_BASE_PATH" \
    --epochs 300 \
    --batch_size 8 \
    --lr 1e-4 \
    --gpu_id 0 \
    --save_epochs 40 \
    --patience 20
    # --resume_from "/path/to/your/checkpoint.pt" # Optional: uncomment to resume training 