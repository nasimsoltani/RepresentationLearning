"""
Count parameters for encoder and adversary (Decoder) using the exact
experiment configs from args.json files.
"""
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'code'))

import torch
from rep_lr.models import TaskAdaptiveEncoder, Encoder
from dra_1.models import Decoder


def count_params(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def print_param_breakdown(model, name):
    print(f"\n{'='*60}")
    print(f"  {name}")
    print(f"{'='*60}")
    for n, m in model.named_modules():
        if len(list(m.children())) == 0:  # leaf modules only
            params = sum(p.numel() for p in m.parameters())
            if params > 0:
                print(f"  {n:<50s}  {params:>10,}")
    total, trainable = count_params(model)
    print(f"  {'─'*62}")
    print(f"  {'TOTAL':<50s}  {total:>10,}")
    print(f"  {'TRAINABLE':<50s}  {trainable:>10,}")


# ── Experiment config (from args.json) ──────────────────────────────────────
SLICE_LEN       = 1024    # slice_len
D2              = 256     # d2  (encoder output dim per channel)
NUM_BLOCKS      = 2       # encoder_num_blocks
LATENT_DIM      = 2 * D2  # = 512  (flattened encoder output fed to decoder)
DROPOUT         = 0.1

print("\n" + "="*60)
print("  Experiment configuration")
print("="*60)
print(f"  slice_len          = {SLICE_LEN}")
print(f"  d2                 = {D2}")
print(f"  encoder_num_blocks = {NUM_BLOCKS}")
print(f"  task_adaptive_enc  = True")
print(f"  latent_dim (2*d2)  = {LATENT_DIM}")

# ── 1. Encoder (TaskAdaptiveEncoder) ────────────────────────────────────────
encoder = TaskAdaptiveEncoder(
    slice_size=SLICE_LEN,
    output_dim=D2,
    dropout=DROPOUT,
    num_blocks=NUM_BLOCKS,
    input_channels=2
)
print_param_breakdown(encoder, "Encoder  (TaskAdaptiveEncoder, num_blocks=2)")

# ── 2. Adversary / Decoder ───────────────────────────────────────────────────
decoder = Decoder(latent_dim=LATENT_DIM, dropout=DROPOUT)
print_param_breakdown(decoder, "Adversary / Decoder  (latent_dim=512)")

# ── Summary ──────────────────────────────────────────────────────────────────
enc_total, _  = count_params(encoder)
dec_total, _  = count_params(decoder)

# break decoder into sub-sections
shared_dec_params  = count_params(decoder.shared_decoder)[0]
shared_conv_params = count_params(decoder.shared_conv)[0]
rf_head_params     = count_params(decoder.rf_head)[0]
cfo_head_params    = count_params(decoder.cfo_head)[0]
ch_head_params     = count_params(decoder.channel_head)[0]

print("\n" + "="*60)
print("  SUMMARY")
print("="*60)
print(f"  Encoder (TaskAdaptiveEncoder)       : {enc_total:>10,}")
print(f"  Adversary/Decoder (total)           : {dec_total:>10,}")
print(f"    ├─ shared_decoder (MLP)           : {shared_dec_params:>10,}")
print(f"    ├─ shared_conv (ConvTranspose)    : {shared_conv_params:>10,}")
print(f"    ├─ rf_head                        : {rf_head_params:>10,}")
print(f"    ├─ cfo_head                       : {cfo_head_params:>10,}")
print(f"    └─ channel_head                  : {ch_head_params:>10,}")
print()
