#!/bin/bash

# Example 1: RF Fingerprinting with Light Projection Layer
python code/rep_lr/main.py \
    --task rf_fingerprinting \
    --pkl_dataset_path data/dataset.pkl \
    --epochs 300 \
    --batch_size 256 \
    --lr 1e-4 \
    --slice_len 1024 \
    --projection_type light \
    --d1 512 \
    --d2 256 \
    --encoder_hidden_dims "512,384" \
    --dropout 0.1 \
    --proj_channels 64 \
    --head_hidden_dim 256 \
    --gpu_id 0 \
    --save_path results/rf_fingerprinting_light \
    --save_epochs 10 \
    --patience 20 \
    --wandb_project "representation-learning-lab" \
    --wandb_entity "your-wandb-username"

# Example 2: RF Fingerprinting with MLP Projection Layer
python code/rep_lr/main.py \
    --task rf_fingerprinting \
    --pkl_dataset_path data/dataset.pkl \
    --epochs 300 \
    --batch_size 256 \
    --lr 1e-4 \
    --slice_len 1024 \
    --projection_type mlp \
    --d1 1024 \
    --d2 512 \
    --encoder_hidden_dims "1024,768,512" \
    --dropout 0.15 \
    --proj_channels 128 \
    --head_hidden_dim 512 \
    --gpu_id 0 \
    --save_path results/rf_fingerprinting_mlp \
    --save_epochs 10 \
    --patience 20 \
    --wandb_project "representation-learning-lab" \
    --wandb_entity "your-wandb-username"

# Example 3: Channel Estimation with Light Projection Layer
python code/rep_lr/main.py \
    --task channel_estimation \
    --pkl_dataset_path data/dataset.pkl \
    --epochs 300 \
    --batch_size 256 \
    --lr 1e-4 \
    --slice_len 160 \
    --projection_type light \
    --d1 512 \
    --d2 256 \
    --encoder_hidden_dims "512,384" \
    --dropout 0.1 \
    --proj_channels 64 \
    --head_hidden_dim 256 \
    --gpu_id 0 \
    --save_path results/channel_estimation_light \
    --save_epochs 10 \
    --patience 20 \
    --wandb_project "representation-learning-lab" \
    --wandb_entity "your-wandb-username"

# Example 4: CFO Estimation with Light Projection Layer
python code/rep_lr/main.py \
    --task cfo_estimation \
    --pkl_dataset_path data/dataset.pkl \
    --epochs 300 \
    --batch_size 256 \
    --lr 1e-4 \
    --slice_len 160 \
    --projection_type light \
    --d1 512 \
    --d2 256 \
    --encoder_hidden_dims "512,384" \
    --dropout 0.1 \
    --proj_channels 64 \
    --head_hidden_dim 256 \
    --gpu_id 0 \
    --save_path results/cfo_estimation_light \
    --save_epochs 10 \
    --patience 20 \
    --wandb_project "representation-learning-lab" \
    --wandb_entity "your-wandb-username"

# Note: Make the script executable with:
# chmod +x run_training.sh

# Run a specific example by commenting out the others
# or run them sequentially 