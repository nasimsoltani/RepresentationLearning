#!/bin/bash

# This script runs the training for the Large CFO Estimation task.
# IMPORTANT: You must specify the path to your dataset pickle file below.

uv run python /home/hofmann/Documents/projects/RepresentationLearning/code/rep_lr/main.py \
    --task cfo_estimation_large \
    --pkl_dataset_path "/home/hofmann/Documents/projects/RepresentationLearning/dataset/rf_partition_dict_0.5.pkl" \
    --epochs 300 \
    --batch_size 32 \
    --lr 1e-4 \
    --gpu_id 0 \
    --save_epochs 40 \
    --patience 20
    # --resume_from "/path/to/your/checkpoint.pt" # Optional: uncomment to resume training 