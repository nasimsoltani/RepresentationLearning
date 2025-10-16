import sys
import os
import torch
import torch.nn as nn
from argparse import Namespace

# --- Setup Paths ---
# Add the specified directories to the Python path to allow importing modules
rep_lr_path = '/scratch/10608/aadharsh_aadhithya/repos/RepresentationLearning/code/rep_lr'
dra_1_path = '/scratch/10608/aadharsh_aadhithya/repos/RepresentationLearning/code/dra_1'

if not os.path.isdir(rep_lr_path):
    raise FileNotFoundError(f"Directory not found: {rep_lr_path}. Please ensure the path is correct.")
if not os.path.isdir(dra_1_path):
    raise FileNotFoundError(f"Directory not found: {dra_1_path}. Please ensure the path is correct.")

sys.path.append(os.path.dirname(rep_lr_path)) # Add 'code' directory
sys.path.append(os.path.dirname(dra_1_path))  # Add 'code' directory

# --- Import Custom Models ---
try:
    from rep_lr.models import (ComplexSequenceProjector, Encoder,
                               RFClassificationHead, ChannelEstimationHead, CFOEstimationHead)
    from dra_1.models import Decoder
except ImportError as e:
    print(f"Error importing models: {e}")
    print("Please check that the paths are correct and the model files exist.")
    sys.exit(1)

# --- Helper Function ---
def count_parameters(model):
    """Counts the number of trainable parameters in a PyTorch model."""
    if model is None:
        return 0
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# --- Model Configuration (Corrected based on shell script) ---
args = Namespace(
    proj_seq_len=256,
    proj_hidden_dim=512,
    d2=256,
    dropout=0.1,
    head_hidden_dim=256,
    encoder_num_blocks=2,
    slice_len=1024,
    num_classes=16,
    fusion_type='sum',
    latent_dim=512
)

# --- Model Creation Functions ---
def create_encoder_model(tasks):
    """Creates a full encoder model for a given list of tasks."""
    projections = nn.ModuleDict()
    heads = nn.ModuleDict()
    task_map = {'RF': 'rf_fingerprinting', 'CFO': 'cfo_estimation', 'Channel': 'channel_estimation'}
    internal_tasks = [task_map[t] for t in tasks]

    for task in internal_tasks:
        if task == 'rf_fingerprinting':
            projections[task] = ComplexSequenceProjector(1024, args.proj_seq_len, args.proj_hidden_dim)
            heads[task] = RFClassificationHead(2*args.d2, args.num_classes, args.head_hidden_dim, args.dropout)
        elif task == 'channel_estimation':
            projections[task] = ComplexSequenceProjector(160, args.proj_seq_len, args.proj_hidden_dim)
            heads[task] = ChannelEstimationHead(2*args.d2, args.head_hidden_dim, 52, args.dropout)
        elif task == 'cfo_estimation':
            projections[task] = ComplexSequenceProjector(160, args.proj_seq_len, args.proj_hidden_dim)
            heads[task] = CFOEstimationHead(2*args.d2, args.head_hidden_dim, args.dropout)

    # Handle fusion_type correctly
    input_channels = 2
    if args.fusion_type == 'depth_concat' and len(internal_tasks) > 1:
        input_channels = 2 * len(internal_tasks)

    encoder = Encoder(args.proj_seq_len, args.d2, args.dropout, args.encoder_num_blocks, input_channels)
    return {'projections': projections, 'encoder': encoder, 'heads': heads}

# --- Main Execution ---
if __name__ == "__main__":
    print("=" * 110)
    print("Unified Model Parameter Decomposition (Complete)")
    print("=" * 110)

    header = f"{'Model Type':<12} | {'Config':<18} | {'Projection Params':>18} | {'Task Head Params':>18} | {'Shared Params':>15} | {'Total Params':>15}"
    print(header)
    print("-" * len(header))

    # --- ENCODER CALCULATIONS ---
    # ADDED missing combinations
    encoder_task_configs = [
        ['RF'],
        ['CFO'],
        ['Channel'],
        ['RF', 'CFO'],
        ['RF', 'Channel'],
        ['CFO', 'Channel'],
        ['RF', 'CFO', 'Channel']
    ]
    for tasks in encoder_task_configs:
        components = create_encoder_model(tasks)
        
        proj_params = count_parameters(components['projections'])
        head_params = count_parameters(components['heads'])
        shared_params = count_parameters(components['encoder'])
        total_params = proj_params + head_params + shared_params
        
        config_name = '+'.join(tasks)
        print(f"{'Encoder':<12} | {config_name:<18} | {proj_params:>18,} | {head_params:>18,} | {shared_params:>15,} | {total_params:>15,}")

    print("-" * len(header))

    # --- DECODER CALCULATIONS ---
    latent_dim_for_decoder = args.d2 * 2
    decoder = Decoder(latent_dim=latent_dim_for_decoder)

    shared_decoder_params = count_parameters(decoder.shared_decoder) + count_parameters(decoder.shared_conv)
    rf_decoder_head_params = count_parameters(decoder.rf_head)
    cfo_decoder_head_params = count_parameters(decoder.cfo_head)
    channel_decoder_head_params = count_parameters(decoder.channel_head)

    # Print decoder rows
    # RF Decoder
    total_rf_dec = shared_decoder_params + rf_decoder_head_params
    print(f"{'Decoder':<12} | {'RF Reconstruction':<18} | {'-':>18} | {rf_decoder_head_params:>18,} | {shared_decoder_params:>15,} | {total_rf_dec:>15,}")
    
    # CFO Decoder
    total_cfo_dec = shared_decoder_params + cfo_decoder_head_params
    print(f"{'Decoder':<12} | {'CFO Reconstruction':<18} | {'-':>18} | {cfo_decoder_head_params:>18,} | {shared_decoder_params:>15,} | {total_cfo_dec:>15,}")

    # Channel Decoder
    total_chan_dec = shared_decoder_params + channel_decoder_head_params
    print(f"{'Decoder':<12} | {'Channel Reconstruction':<18} | {'-':>18} | {channel_decoder_head_params:>18,} | {shared_decoder_params:>15,} | {total_chan_dec:>15,}")

    print("=" * 110)
