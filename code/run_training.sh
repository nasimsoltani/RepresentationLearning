
# --- resolved from .env (see .env.example) ---
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [ -f "$REPO_ROOT/.env" ]; then set -a; . "$REPO_ROOT/.env"; set +a; fi
: "${DATA_BASE_PATH:?set DATA_BASE_PATH in .env (see .env.example)}"
: "${PKL_FILE_PATH:?set PKL_FILE_PATH in .env (see .env.example)}"
DATASET_PATH="${DATASET_PATH:-$PKL_FILE_PATH/rf_partition_dict_${PORTION_TO_USE:-0.5}.pkl}"
# ---------------------------------------------

#bin/bash/!
uv run python $REPO_ROOT/code/rep_lr/main.py \
--gpu_id 0 \
--slice_len 1024 \
--pkl_dataset_path "$DATASET_PATH" \
--data_root "$DATA_BASE_PATH" \
--save_path $REPO_ROOT/results \
