#!/bin/bash

# --- resolved from .env (see .env.example) ---
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [ -f "$REPO_ROOT/.env" ]; then set -a; . "$REPO_ROOT/.env"; set +a; fi
: "${DATA_BASE_PATH:?set DATA_BASE_PATH in .env (see .env.example)}"
: "${PKL_FILE_PATH:?set PKL_FILE_PATH in .env (see .env.example)}"
DATASET_PATH="${DATASET_PATH:-$PKL_FILE_PATH/rf_partition_dict_${PORTION_TO_USE:-0.5}.pkl}"
# ---------------------------------------------

python $REPO_ROOT/code/ML/top_test.py \
--gpu_id $1 \
--chop_size 4096 \
--slice_size 2048 \
--pkl_dataset_path "$DATASET_PATH" \
--data_root "$DATA_BASE_PATH" \
--weight_path /home/ns38942/AiR/results/IQ/weights-OOD_CW.pt \
