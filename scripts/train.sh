#!/bin/bash

# =================================================================================
# Training Script for Representation Learning Models
# =================================================================================
#
# This script provides examples for running single-task and multi-task training
# using `code/rep_lr/main.py`.
#
# Usage:
# 1. Set the `DATASET_PATH` variable to the location of your .pkl dataset.
# 2. Uncomment the training configuration you want to run.
# 3. Adjust hyperparameters as needed.
#
# Common arguments:
# --pkl_dataset_path: Path to the dataset file.
# --epochs: Number of training epochs.
# --batch_size: Batch size.
# --lr: Learning rate.
# --gpu_id: GPU to use.
# --save_path: Directory to save results and checkpoints.
# --wandb_project: Name for your Weights & Biases project.

# --- Configuration ---
DATASET_PATH="/home/hofmann/Documents/projects/RepresentationLearning/dataset/rf_partition_dict_0.5.pkl" # <-- IMPORTANT: SET THIS PATH

# =================================================================================
# Example 1: Single-Task - RF Fingerprinting (Light Projection)
# =================================================================================
# uv run python code/rep_lr/main.py \
#     --task rf_fingerprinting \
#     --pkl_dataset_path $DATASET_PATH \
#     --save_path results/single_task/rf_light \
#     --projection_type light \
#     --epochs 100 \
#     --batch_size 128 \
#     --d1 512 --d2 256 \
#     --encoder_hidden_dims "512,384" \
#     --head_hidden_dim 256 \
#     --gpu_id 0

# =================================================================================
# Example 2: Single-Task - Channel Estimation (MLP Projection)
# =================================================================================
# uv run python code/rep_lr/main.py \
#     --task channel_estimation \
#     --pkl_dataset_path $DATASET_PATH \
#     --save_path results/single_task/channel_mlp \
#     --projection_type mlp \
#     --epochs 150 \
#     --batch_size 128 \
#     --d1 512 --d2 256 \
#     --encoder_hidden_dims "512,384" \
#     --head_hidden_dim 256 \
#     --gpu_id 0

# =================================================================================
# Example 3: Single-Task - CFO Estimation (Light Projection)
# =================================================================================
uv run python code/rep_lr/main.py \
    --task cfo_estimation \
    --pkl_dataset_path $DATASET_PATH \
    --save_path results/single_task/cfo_light \
    --projection_type light \
    --epochs 150 \
    --batch_size 128 \
    --d1 512 --d2 256 \
    --encoder_hidden_dims "512,384" \
    --head_hidden_dim 256 \
    --gpu_id 0

# =================================================================================
# Example 4: Multi-Task (MTL) - RF + CFO
# Here, we train two tasks together. The --mtl flag is required.
# We can also assign different weights to the task losses.
# =================================================================================
# uv run python code/rep_lr/main.py \
#     --mtl \
#     --task rf_fingerprinting cfo_estimation \
#     --pkl_dataset_path $DATASET_PATH \
#     --save_path results/mtl/rf_cfo \
#     --projection_type light \
#     --epochs 200 \
#     --batch_size 64 \
#     --lr 1e-4 \
#     --w_rf 1.0 \
#     --w_cfo 1.5 \
#     --d1 512 --d2 256 \
#     --encoder_hidden_dims "512,384" \
#     --head_hidden_dim 256 \
#     --gpu_id 0

# =================================================================================
# Example 5: Multi-Task (MTL) - All three tasks (ACTIVE BY DEFAULT)
# This configuration trains all three tasks jointly.
# =================================================================================
# echo "Running MTL training for all three tasks..."
# uv run python code/rep_lr/main.py \
#     --mtl \
#     --task rf_fingerprinting cfo_estimation channel_estimation \
#     --pkl_dataset_path $DATASET_PATH \
#     --save_path results/mtl/all_tasks \
#     --projection_type light \
#     --epochs 300 \
#     --batch_size 64 \
#     --lr 1e-4 \
#     --w_rf 1.0 \
#     --w_cfo 1.0 \
#     --w_channel 1.0 \
#     --d1 512 \
#     --d2 256 \
#     --encoder_hidden_dims "512,384" \
#     --head_hidden_dim 256 \
#     --gpu_id 0 \
#     --save_epochs 10 \
#     --patience 20 \
#     --wandb_project "representation-learning-lab" 