#!/usr/bin/env python3
"""
RDP (Rényi Differential Privacy) instance-encoding baseline.

Implements (2, ε)-RDP Gaussian mechanism (FIL paper Appendix A.5) applied
to the rep_lr MTL encoder output.  Each task gets its OWN noise level,
derived from that task's target adversary reconstruction MSE.

Math (FIL Appendix A.5 + Mironov 2017):
────────────────────────────────────────
  MSE bound (Eq. 7, diam_i = 1 ∀i):
      target_mse  ≥  1 / (4·(e^ε − 1))

  Solve for ε:  ε  =  log(1  +  1 / (4·target_mse))

  (2,ε)-RDP Gaussian noise std (Mironov 2017, L2-sensitivity = 2C):
      σ  =  2C / √ε       [i.e. σ² = (2C)²/ε]

  Enc_RDP(x)  =  clip_flat(Enc(x), C)  +  N(0, σ²·I)   on vec(Enc) ∈ ℝ^{2·d2}

Split point: shared rep_lr encoder output (B, 2·d2), before task heads.

Output format matches your existing table:
          final_mse    utility(RDP)    utility(rep_lr)
task
rf        0.500941         ?              0.518564
cfo       0.345215         ?              0.307801
channel   0.337255         ?              0.405239

Usage (from repo root):
    python baselines/rdp/run_rdp_benchmark.py \\
        --ckpt_path   results/<run>/best.pt   \\
        --pkl_dataset_path  dataset/rf_partition_dict_0.5.pkl  \\
        --target_mse_rf       0.500941 \\
        --target_mse_cfo      0.345215 \\
        --target_mse_channel  0.337255 \\
        --rep_lr_utility_rf       0.518564 \\
        --rep_lr_utility_cfo      0.307801 \\
        --rep_lr_utility_channel  0.405239 \\
        --gpu_id 0
"""

import argparse
import json
import math
import os
import pickle
import sys

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import r2_score, accuracy_score
from torch.utils.data import DataLoader
from tqdm import tqdm

# ── path setup ───────────────────────────────────────────────────────────────
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.join(REPO_ROOT, 'code', 'rep_lr'))

from models import (
    ComplexSequenceProjector,
    Encoder, TaskAdaptiveEncoder,
    RFClassificationHead, ChannelEstimationHead,
    CFOEstimationHead, SimpleCFOEstimationHead, CFOAdaptiveHead,
)
from py_datasets import TrainDataset


# ══════════════════════════════════════════════════════════════════════════════
# RDP math
# ══════════════════════════════════════════════════════════════════════════════

def mse_to_epsilon(target_mse: float, diam: float = 1.0) -> float:
    """ε = log(1 + diam² / (4·target_mse))  — solving Eq. 7."""
    return math.log(1.0 + diam ** 2 / (4.0 * target_mse))


def epsilon_to_sigma(eps: float, C: float) -> float:
    """σ = 2C/√ε  — (2,ε)-RDP Gaussian noise std (Mironov 2017)."""
    return 2.0 * C / math.sqrt(eps)


@torch.no_grad()
def apply_rdp(enc: torch.Tensor, C: float, sigma: float) -> torch.Tensor:
    """L2-clip the flattened encoder output to norm C, then add isotropic Gaussian noise.

    Encoder output is (B, 2, d2); RDP is applied on vec(enc) ∈ ℝ^{2·d2} so C and σ
    match the FIL / Mironov sensitivity on the full representation vector.
    """
    flat    = enc.reshape(enc.size(0), -1)
    norms   = flat.norm(dim=-1, keepdim=True).clamp(min=1e-9)
    clipped = flat / torch.clamp(norms / C, min=1.0)
    noisy   = clipped + torch.randn_like(clipped) * sigma
    return noisy.view_as(enc)


# ══════════════════════════════════════════════════════════════════════════════
# Model loading
# ══════════════════════════════════════════════════════════════════════════════

def load_train_args(ckpt_path: str, args_path: str = None) -> argparse.Namespace:
    path = args_path or os.path.join(os.path.dirname(ckpt_path), 'args.json')
    if not os.path.exists(path):
        raise FileNotFoundError(f'args.json not found: {path}')
    with open(path) as f:
        return argparse.Namespace(**json.load(f))


def build_model(ta) -> nn.ModuleDict:
    """Reconstruct rep_lr MTL model from training args (ta)."""
    projections = nn.ModuleDict()
    heads       = nn.ModuleDict()

    for task in ta.task:
        if task == 'rf_fingerprinting':
            projections[task] = ComplexSequenceProjector(
                ta.slice_len, ta.proj_seq_len, ta.proj_hidden_dim)
            heads[task] = RFClassificationHead(
                2 * ta.d2, getattr(ta, 'num_classes', 16),
                ta.head_hidden_dim, ta.dropout)

        elif task == 'channel_estimation':
            projections[task] = ComplexSequenceProjector(
                160, ta.proj_seq_len, ta.proj_hidden_dim)
            heads[task] = ChannelEstimationHead(
                2 * ta.d2, ta.head_hidden_dim, 52, ta.dropout)

        elif task == 'cfo_estimation':
            projections[task] = ComplexSequenceProjector(
                160, ta.proj_seq_len, ta.proj_hidden_dim)
            if getattr(ta, 'adaptive_cfo', False):
                heads[task] = CFOAdaptiveHead(2 * ta.d2, 128, 0.1)
            elif getattr(ta, 'simple_cfo', False):
                heads[task] = SimpleCFOEstimationHead(2 * ta.d2, 64, 0.1)
            else:
                heads[task] = CFOEstimationHead(
                    2 * ta.d2, ta.head_hidden_dim, ta.dropout)

    enc_dim  = ta.proj_seq_len
    enc_chan = 2
    fusion   = getattr(ta, 'fusion_type', 'sum')
    if fusion == 'concat':
        enc_dim  = ta.proj_seq_len * len(ta.task)
    elif fusion == 'depth_concat':
        enc_chan = 2 * len(ta.task)

    EncoderCls = TaskAdaptiveEncoder \
        if getattr(ta, 'task_adaptive_encoder', False) else Encoder
    encoder = EncoderCls(
        slice_size=enc_dim, output_dim=ta.d2,
        dropout=ta.dropout, num_blocks=ta.encoder_num_blocks,
        input_channels=enc_chan)

    return nn.ModuleDict({'projections': projections,
                          'encoder':     encoder,
                          'heads':       heads})


def load_checkpoint(model, ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location=device)
    model['projections'].load_state_dict(ckpt['projections_state_dict'])
    model['encoder'].load_state_dict(ckpt['encoder_state_dict'])
    model['heads'].load_state_dict(ckpt['heads_state_dict'])


# ══════════════════════════════════════════════════════════════════════════════
# Per-task clip norm estimation
# ══════════════════════════════════════════════════════════════════════════════

# Batch indices: RF_X=0, RF_y=1, CFO_X=2, CFO_y=3, CH_X=4, CH_y=5, path=6
TASK_X_IDX = {
    'rf_fingerprinting': 0,
    'cfo_estimation':    2,
    'channel_estimation': 4,
}

@torch.no_grad()
def estimate_clip_norms(model, loader, device,
                        percentile=95.0, fusion='sum') -> dict:
    """Return {task: C} where C is the p-th percentile of ‖vec(enc)‖₂ (full vector)."""
    model.eval()
    norms_by_task = {t: [] for t in TASK_X_IDX if t in model['projections']}

    for batch in loader:
        projs = {}
        for task, idx in TASK_X_IDX.items():
            if task not in model['projections']:
                continue
            projs[task] = model['projections'][task](
                batch[idx].to(device).float())

        for task, p in projs.items():
            fused = _fuse(projs, task, fusion)
            enc   = model['encoder'](fused)
            flat  = enc.reshape(enc.size(0), -1)
            norms_by_task[task].append(flat.norm(dim=-1).cpu())

    result = {}
    for task, ns in norms_by_task.items():
        arr  = torch.cat(ns).numpy()
        C    = float(np.percentile(arr, percentile))
        result[task] = C
        print(f'  {task:<22}  p{percentile:.0f}={C:.4f}  '
              f'mean={arr.mean():.4f}  max={arr.max():.4f}')
    return result


def _fuse(projs: dict, task: str, fusion: str) -> torch.Tensor:
    p = projs[task]
    others = [projs[t] for t in projs if t != task]
    if fusion == 'sum':
        return p + sum(others) if others else p
    if fusion == 'concat':
        return torch.cat([projs[t] for t in projs], dim=2)
    if fusion == 'depth_concat':
        return torch.cat([projs[t] for t in projs], dim=1)
    return p


def _cfo_bn_disable_running_stats(head: nn.Module):
    """Use per-batch BN stats without updating running buffers (for noisy enc inputs)."""
    saved = []
    for m in head.modules():
        if isinstance(m, nn.BatchNorm1d):
            saved.append((m, m.track_running_stats))
            m.track_running_stats = False
    return saved


def _cfo_bn_restore_running_stats(saved: list) -> None:
    for m, prev in saved:
        m.track_running_stats = prev


# ══════════════════════════════════════════════════════════════════════════════
# Evaluation — one noise level per task
# ══════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def evaluate(model, loader, device, ta,
             sigmas: dict,    # {task: sigma}
             clip_norms: dict,  # {task: C}
             mean_cfo: float,
             std_cfo: float,
             n_noise: int = 5) -> dict:
    """Evaluate all three tasks, each with its own RDP noise level."""
    model.eval()
    # CFO head may use BatchNorm1d.  Eval mode + running stats from clean training
    # mis-normalizes noisy encoder inputs; train() fixes batch stats but corrupts
    # running buffers.  Use batch statistics only (track_running_stats=False)
    # without updating stored running mean/var.
    bn_saved = []
    if 'cfo_estimation' in model['heads']:
        bn_saved = _cfo_bn_disable_running_stats(
            model['heads']['cfo_estimation'])
    fusion = getattr(ta, 'fusion_type', 'sum')

    rf_preds, rf_golds   = [], []
    cfo_preds, cfo_golds = [], []
    ch_preds,  ch_golds  = [], []

    try:
        for batch in tqdm(loader, desc='  evaluating', leave=False):
            RF_X  = batch[0].to(device).float()
            RF_y  = batch[1]
            CFO_X = batch[2].to(device).float()
            CFO_y = batch[3]
            CH_X  = batch[4].to(device).float()
            CH_y  = batch[5]

            projs = {}
            for task, x in [('rf_fingerprinting', RF_X),
                            ('cfo_estimation',    CFO_X),
                            ('channel_estimation', CH_X)]:
                if task in model['projections']:
                    projs[task] = model['projections'][task](x)

            # Accumulate predictions over n_noise draws
            rf_acc  = torch.zeros(RF_X.size(0), 16, device=device)
            cfo_acc = torch.zeros(CFO_X.size(0), 1,  device=device)
            ch_acc  = None

            for _ in range(n_noise):
                for task, p in projs.items():
                    fused     = _fuse(projs, task, fusion)
                    enc       = model['encoder'](fused)
                    C         = clip_norms[task]
                    sigma     = sigmas[task]
                    enc_noisy = apply_rdp(enc, C, sigma)
                    out       = model['heads'][task](enc_noisy)

                    if task == 'rf_fingerprinting':
                        rf_acc  += out
                    elif task == 'cfo_estimation':
                        cfo_acc += out
                    elif task == 'channel_estimation':
                        if ch_acc is None:
                            ch_acc = torch.zeros_like(out)
                        ch_acc += out

            rf_acc  /= n_noise
            cfo_acc /= n_noise
            if ch_acc is not None:
                ch_acc /= n_noise

            rf_preds.append(rf_acc.argmax(1).cpu().numpy())
            rf_golds.append(RF_y.numpy())

            cfo_pred_d = cfo_acc.cpu().numpy().ravel() * std_cfo + mean_cfo
            cfo_true_d = CFO_y.numpy().ravel()          * std_cfo + mean_cfo
            cfo_preds.append(cfo_pred_d)
            cfo_golds.append(cfo_true_d)

            if ch_acc is not None:
                ch_preds.append(ch_acc.cpu().numpy().reshape(len(CH_X), -1))
                ch_golds.append(CH_y.numpy().reshape(len(CH_X), -1))

        return {
            'rf_accuracy': accuracy_score(
                np.concatenate(rf_golds), np.concatenate(rf_preds)),
            'cfo_r2': r2_score(
                np.concatenate(cfo_golds), np.concatenate(cfo_preds)),
            'channel_r2': r2_score(
                np.concatenate(ch_golds), np.concatenate(ch_preds),
                multioutput='uniform_average') if ch_preds else float('nan'),
        }
    finally:
        _cfo_bn_restore_running_stats(bn_saved)


# ══════════════════════════════════════════════════════════════════════════════
# CLI & main
# ══════════════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(
        description='RDP instance-encoding baseline (FIL Appendix A.5)')
    p.add_argument('--ckpt_path',          required=True)
    p.add_argument('--args_path',          default=None,
                   help='Path to args.json (default: same dir as ckpt_path)')
    p.add_argument('--pkl_dataset_path',   required=True)

    # Per-task target adversary MSEs (from your rep_lr runs)
    p.add_argument('--target_mse_rf',      type=float, default=0.500941)
    p.add_argument('--target_mse_cfo',     type=float, default=0.345215)
    p.add_argument('--target_mse_channel', type=float, default=0.337255)

    # rep_lr reference utility (printed in the comparison table)
    p.add_argument('--rep_lr_utility_rf',      type=float, default=0.518564)
    p.add_argument('--rep_lr_utility_cfo',     type=float, default=0.307801)
    p.add_argument('--rep_lr_utility_channel', type=float, default=0.405239)

    p.add_argument('--diam',             type=float, default=1.0,
                   help='Per-dim diameter of input space (1.0 = FIL convention)')
    p.add_argument('--clip_percentile',  type=float, default=95.0)
    p.add_argument('--n_noise_samples',  type=int,   default=5,
                   help='Noise draws to average per batch (reduces variance)')
    p.add_argument('--batch_size',       type=int,   default=256)
    p.add_argument('--num_workers',      type=int,   default=8)
    p.add_argument('--gpu_id',           type=int,   default=0)
    p.add_argument('--results_dir',      default='baselines/rdp/results')
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.results_dir, exist_ok=True)

    device = torch.device(
        f'cuda:{args.gpu_id}' if torch.cuda.is_available() else 'cpu')

    # ── load model ───────────────────────────────────────────────────────────
    print('Loading model...')
    ta    = load_train_args(args.ckpt_path, args.args_path)
    model = build_model(ta).to(device)
    load_checkpoint(model, args.ckpt_path, device)
    model.eval()

    # ── dataset ──────────────────────────────────────────────────────────────
    print('Loading dataset...')
    with open(args.pkl_dataset_path, 'rb') as f:
        pkl = pickle.load(f)

    val_list  = pkl['val']
    test_list = pkl['test']
    max_cfo   = pkl['max_cfo']
    mean_cfo  = pkl['mean_cfo']
    std_cfo   = pkl['std_cfo']

    # Build class dict (Radio0...Radio15)
    class_ids = {f'Radio{i}': i for i in range(16)}

    # Create dataset args with slice_len
    ds_args = argparse.Namespace(slice_len=getattr(ta, 'slice_len', 1024))

    val_ds  = TrainDataset(val_list,  class_ids, ds_args, max_cfo, mean_cfo, std_cfo)
    test_ds = TrainDataset(test_list, class_ids, ds_args, max_cfo, mean_cfo, std_cfo)
    val_ld  = DataLoader(val_ds,  batch_size=args.batch_size, shuffle=False,
                         num_workers=args.num_workers, pin_memory=True)
    test_ld = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                         num_workers=args.num_workers, pin_memory=True)

    # ── per-task RDP parameters ──────────────────────────────────────────────
    task_mse = {
        'rf_fingerprinting':  args.target_mse_rf,
        'cfo_estimation':     args.target_mse_cfo,
        'channel_estimation': args.target_mse_channel,
    }

    print(f'\nEstimating clip norms (p{args.clip_percentile:.0f} on val set)...')
    clip_norms = estimate_clip_norms(
        model, val_ld, device,
        percentile=args.clip_percentile,
        fusion=getattr(ta, 'fusion_type', 'sum'))

    print('\nPer-task RDP parameters:')
    epsilons = {}
    sigmas   = {}
    for task, mse in task_mse.items():
        eps  = mse_to_epsilon(mse, args.diam)
        sigma = epsilon_to_sigma(eps, clip_norms[task])
        epsilons[task] = eps
        sigmas[task]   = sigma
        print(f'  {task:<22}  target_mse={mse:.6f}  '
              f'ε={eps:.4f}  C={clip_norms[task]:.4f}  σ={sigma:.4f}')

    # ── baseline (no noise) ──────────────────────────────────────────────────
    print('\nBaseline (σ=0, no noise on any task):')
    zero_sigmas = {t: 0.0 for t in sigmas}
    baseline = evaluate(model, test_ld, device, ta,
                        sigmas=zero_sigmas,
                        clip_norms=clip_norms,
                        mean_cfo=mean_cfo,
                        std_cfo=std_cfo,
                        n_noise=1)
    print(f'  RF acc={baseline["rf_accuracy"]:.6f}  '
          f'CFO R²={baseline["cfo_r2"]:.6f}  '
          f'Channel R²={baseline["channel_r2"]:.6f}')

    # ── RDP evaluation ───────────────────────────────────────────────────────
    print('\nEvaluating with per-task RDP noise...')
    rdp = evaluate(model, test_ld, device, ta,
                   sigmas=sigmas,
                   clip_norms=clip_norms,
                   mean_cfo=mean_cfo,
                   std_cfo=std_cfo,
                   n_noise=args.n_noise_samples)

    # ── print table (matches your existing format) ────────────────────────────
    task_rows = [
        ('rf',      'rf_fingerprinting',  args.target_mse_rf,
         args.rep_lr_utility_rf,      rdp['rf_accuracy']),
        ('cfo',     'cfo_estimation',     args.target_mse_cfo,
         args.rep_lr_utility_cfo,     rdp['cfo_r2']),
        ('channel', 'channel_estimation', args.target_mse_channel,
         args.rep_lr_utility_channel, rdp['channel_r2']),
    ]

    print('\n' + '='*72)
    print('RDP Baseline — Privacy-Utility Tradeoff')
    print('(Per-task noise derived from target adversary reconstruction MSE)')
    print('='*72)
    print(f'{"task":<10}  {"final_mse":>12}  {"epsilon":>9}  '
          f'{"sigma":>9}  {"utility(RDP)":>13}  {"utility(rep_lr)":>15}')
    print('─'*72)
    for name, task, mse, util_ref, util_rdp in task_rows:
        print(f'{name:<10}  {mse:>12.6f}  {epsilons[task]:>9.4f}  '
              f'{sigmas[task]:>9.4f}  {util_rdp:>13.6f}  {util_ref:>15.6f}')

    print('\nBaseline utility (same model, no noise):')
    print(f'  rf acc     = {baseline["rf_accuracy"]:.6f}')
    print(f'  cfo R²     = {baseline["cfo_r2"]:.6f}')
    print(f'  channel R² = {baseline["channel_r2"]:.6f}')

    # ── save ─────────────────────────────────────────────────────────────────
    out = {
        'rdp_params': {
            t: {'target_mse': task_mse[t], 'epsilon': epsilons[t],
                'sigma': sigmas[t], 'C': clip_norms[t]}
            for t in task_mse
        },
        'baseline': baseline,
        'rdp': rdp,
        'table': [
            {'task': n, 'final_mse': mse,
             'utility_rdp': ur, 'utility_rep_lr': uref}
            for n, _, mse, uref, ur in task_rows
        ],
    }
    out_path = os.path.join(args.results_dir, 'rdp_results.json')
    with open(out_path, 'w') as f:
        json.dump(out, f, indent=2)
    print(f'\nResults → {out_path}')


if __name__ == '__main__':
    main()
