
# --- resolved from .env (see .env.example) ---
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [ -f "$REPO_ROOT/.env" ]; then set -a; . "$REPO_ROOT/.env"; set +a; fi
: "${DATA_BASE_PATH:?set DATA_BASE_PATH in .env (see .env.example)}"
: "${PKL_FILE_PATH:?set PKL_FILE_PATH in .env (see .env.example)}"
DATASET_PATH="${DATASET_PATH:-$PKL_FILE_PATH/rf_partition_dict_${PORTION_TO_USE:-0.5}.pkl}"
# ---------------------------------------------

# Example 1: RF Fingerprinting with Light Projection Layer
#tasks cfo_estimation rf_fingerprinting \  
uv run python code/rep_lr/main.py \
    --task channel_estimation \
    --pkl_dataset_path "$DATASET_PATH" \
    --data_root "$DATA_BASE_PATH" \
    --epochs 300 \
    --batch_size 64 \
    --lr 1e-4 \
    --slice_len 1024 \
    --projection_type light \
    --proj_channels 4 \
    --d1 256 \
    --d2 128 \
    --encoder_hidden_dims "256,128" \
    --dropout 0.1 \
    --head_hidden_dim 128 \
    --gpu_id 0 \
    --save_path results/channel_estimation_mlp_1 \
    --save_epochs 10 \
    --patience 20 \
    --wandb_project "representation-learning-lab" \
