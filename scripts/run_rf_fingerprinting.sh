#!/bin/bash

# This script runs the training for the RF Fingerprinting task.
# IMPORTANT: You must specify the path to your dataset pickle file below.

# python code/rep_lr/main.py \
#     --task rf_fingerprinting \
#     --pkl_dataset_path "/path/to/your/dataset.pkl" \
#     --epochs 300 \
#     --batch_size 256 \
#     --lr 1e-4 \
#     --gpu_id 0 \
#     --save_epochs 10 \
#     --patience 20
#     # --resume_from "/path/to/your/checkpoint.pt" # Optional: uncomment to resume training 


#!/bin/bash

# This script runs the training for the Large CFO Estimation task.
# IMPORTANT: You must specify the path to your dataset pickle file below.

uv run python /home/hofmann/Documents/projects/RepresentationLearning/code/rep_lr/main.py \
    --task rf_fingerprinting \
    --pkl_dataset_path "/home/hofmann/Documents/projects/RepresentationLearning/dataset/rf_partition_dict_0.5.pkl" \
    --epochs 300 \
    --batch_size 64 \
    --lr 1e-3 \
    --gpu_id 0 \
    --save_epochs 40 \
    --patience 20
    # --resume_from "/path/to/your/checkpoint.pt" # Optional: uncomment to resume training 