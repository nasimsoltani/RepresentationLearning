#!/usr/bin/env python3
"""NeuraCrypt IQ benchmark — fair MTL comparison against rep_lr.

Architecture (mirrors rep_lr's shared-encoder MTL setup)
─────────────────────────────────────────────────────────
  rep_lr:       task projections → shared trained encoder  → task heads
  NeuraCrypt:   (same image size) → shared FROZEN encoder  → shared ViT → task heads

One shared frozen PrivateEncoder processes inputs from all three tasks.
One shared trainable ViT backbone is trained jointly on:
    L = L_rf (cross-entropy) + L_cfo (MSE) + L_channel (MSE)
Three task-specific output heads complete the model.

This is a FAIR comparison because:
  - Both methods share a single encoder across all tasks
  - Both methods have task-specific heads
  - The only difference is rep_lr's encoder is *trained* while NeuraCrypt's
    is *frozen random* (NeuraCrypt's privacy guarantee)

Usage (from repo root)::

    python baselines/NeuraCrypt/scripts/run_iq_benchmark.py \\
        --pkl_dataset_path dataset/rf_partition_dict_0.5.pkl \\
        --gpu_id 0 \\
        --epochs 50

    # Single-task ablation (optional, not the primary comparison):
    python baselines/NeuraCrypt/scripts/run_iq_benchmark.py \\
        --mode single_task --pkl_dataset_path ...
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, r2_score
from torch.utils.data import DataLoader
from tqdm import tqdm

NEURACRYPT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(NEURACRYPT_ROOT))

from sandstone.datasets.iq_dataset import (
    IQChannelDataset,
    IQCFODataset,
    IQMTLDataset,
    IQRFDataset,
)

# ══════════════════════════════════════════════════════════════════════════════
# NeuraCrypt core: PrivateEncoder
# Copied verbatim from sandstone/learn/lightning/private.py
# Only change: num_chan is now set to 2 via args (not hard-coded to 1)
# ══════════════════════════════════════════════════════════════════════════════

class PrivateEncoder(nn.Module):
    """NeuraCrypt private encoder (frozen random network + pixel shuffle).

    Source: baselines/NeuraCrypt/sandstone/learn/lightning/private.py
    The only architectural change vs the original is accepting num_chan=2.
    """

    def __init__(self, args, width_factor: int = 1):
        super().__init__()
        self.args = args
        input_dim  = args.num_chan
        patch_size = args.private_kernel_size
        output_dim = args.hidden_dim
        num_patches = (args.img_size[0] // patch_size) ** 2

        args.input_dim = args.hidden_dim

        layers = [
            nn.Conv2d(input_dim, output_dim * width_factor,
                      kernel_size=patch_size, dilation=1, stride=patch_size),
            nn.ReLU(),
        ]
        for _ in range(args.private_depth):
            layers += [
                nn.Conv2d(output_dim * width_factor, output_dim * width_factor,
                          kernel_size=1, dilation=1, stride=1),
                nn.BatchNorm2d(output_dim * width_factor, track_running_stats=False),
                nn.ReLU(),
            ]
        self.image_encoder = nn.Sequential(*layers)
        self.pos_embedding  = nn.Parameter(
            torch.randn(1, num_patches, output_dim * width_factor))
        self.mixer = nn.Sequential(
            nn.ReLU(),
            nn.Linear(output_dim * width_factor, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        enc = self.image_encoder(x)
        B, C, H, W = enc.size()
        enc = enc.view(B, C, H * W).transpose(1, 2)   # (B, N, C)
        enc = enc + self.pos_embedding
        enc = self.mixer(enc)                           # (B, N, hidden_dim)

        # Per-sample random pixel shuffle — the key NeuraCrypt privacy mechanism
        if not getattr(self.args, 'remove_pixel_shuffle', False):
            shuffled = torch.zeros_like(enc)
            for i in range(B):
                idx = torch.randperm(H * W, device=enc.device)
                shuffled[i] = enc[i][idx]
            enc = shuffled
        return enc


# ══════════════════════════════════════════════════════════════════════════════
# Minimal ViT — mirrors dvit.py without einops dependency
# ══════════════════════════════════════════════════════════════════════════════

class _Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.0):
        super().__init__()
        inner = dim_head * heads
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.to_qkv = nn.Linear(dim, inner * 3, bias=False)
        self.reattn_weights = nn.Parameter(torch.randn(heads, heads))
        self.reattn_norm = nn.LayerNorm(heads)
        self.to_out = nn.Sequential(nn.Linear(inner, dim), nn.Dropout(dropout))

    def forward(self, x):
        B, N, _ = x.shape
        h = self.heads
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = [t.view(B, N, h, -1).transpose(1, 2) for t in qkv]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        # Deep-ViT re-attention
        attn = torch.einsum('bhij,hg->bgij', attn, self.reattn_weights)
        attn = self.reattn_norm(attn.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        out = (attn @ v).transpose(1, 2).reshape(B, N, -1)
        return self.to_out(out)


class _FeedForward(nn.Module):
    def __init__(self, dim, hidden, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden, dim), nn.Dropout(dropout))

    def forward(self, x):
        return self.net(x)


class _Block(nn.Module):
    def __init__(self, dim, heads, dim_head, mlp_dim, dropout):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn  = _Attention(dim, heads, dim_head, dropout)
        self.norm2 = nn.LayerNorm(dim)
        self.ff    = _FeedForward(dim, mlp_dim, dropout)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ff(self.norm2(x))
        return x


class ViTBackbone(nn.Module):
    """Shared ViT body: CLS token + transformer layers.

    Accepts pre-tokenised sequences (B, N, dim) from PrivateEncoder.
    Returns CLS token embedding (B, dim).
    """

    def __init__(self, dim, depth, heads, mlp_dim, dropout=0.0):
        super().__init__()
        dim_head = max(dim // heads, 64)
        self.cls_token   = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout     = nn.Dropout(dropout)
        self.transformer = nn.Sequential(*[
            _Block(dim, heads, dim_head, mlp_dim, dropout) for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        B = x.size(0)
        cls = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls, x], dim=1)
        x = self.dropout(x)
        x = self.transformer(x)
        return self.norm(x[:, 0])   # (B, dim)


# ══════════════════════════════════════════════════════════════════════════════
# MTL NeuraCrypt model
#   frozen PrivateEncoder  →  shared ViTBackbone  →  3 task heads
# ══════════════════════════════════════════════════════════════════════════════

class MTLNeuraCrypt(nn.Module):
    """Multi-task NeuraCrypt — the fair analog of rep_lr's MTL setup.

    Frozen PrivateEncoder (random, NeuraCrypt's privacy guarantee) +
    shared trainable ViT backbone + three task-specific linear heads.
    """

    def __init__(self, encoder_args, dim, depth, heads, mlp_dim, dropout=0.0):
        super().__init__()
        # Frozen private encoder (shared across all tasks)
        self.encoder = PrivateEncoder(encoder_args)
        for p in self.encoder.parameters():
            p.requires_grad_(False)

        # Shared trainable backbone
        self.backbone = ViTBackbone(dim, depth, heads, mlp_dim, dropout)

        # Task-specific heads
        self.rf_head      = nn.Linear(dim, 16)     # classification
        self.cfo_head     = nn.Linear(dim, 1)      # scalar regression
        self.channel_head = nn.Linear(dim, 104)    # vector regression (2×52)

    @torch.no_grad()
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def forward_task(self, x: torch.Tensor, task: str) -> torch.Tensor:
        tokens = self.encode(x)
        feat   = self.backbone(tokens)
        if task == 'rf':
            return self.rf_head(feat)
        elif task == 'cfo':
            return self.cfo_head(feat)
        else:
            return self.channel_head(feat)


# ══════════════════════════════════════════════════════════════════════════════
# Single-task model (for single-task baseline / ablation only)
# ══════════════════════════════════════════════════════════════════════════════

class SingleTaskNeuraCrypt(nn.Module):
    def __init__(self, encoder_args, num_classes, dim, depth, heads, mlp_dim,
                 dropout=0.0):
        super().__init__()
        self.encoder  = PrivateEncoder(encoder_args)
        for p in self.encoder.parameters():
            p.requires_grad_(False)
        self.backbone = ViTBackbone(dim, depth, heads, mlp_dim, dropout)
        self.head     = nn.Linear(dim, num_classes)

    def forward(self, x):
        with torch.no_grad():
            tokens = self.encoder(x)
        feat = self.backbone(tokens)
        return self.head(feat)


# ══════════════════════════════════════════════════════════════════════════════
# Training — MTL (primary, fair comparison)
# ══════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def evaluate_mtl(model: MTLNeuraCrypt, loader, device):
    model.backbone.eval()
    model.rf_head.eval(); model.cfo_head.eval(); model.channel_head.eval()

    rf_preds, rf_golds  = [], []
    cfo_preds, cfo_golds = [], []
    ch_preds,  ch_golds  = [], []
    total_loss = 0.0

    for batch in loader:
        rf_x  = batch['rf_x'].to(device)
        rf_y  = batch['rf_y'].to(device)
        cfo_x = batch['cfo_x'].to(device)
        cfo_y = batch['cfo_y'].to(device)
        ch_x  = batch['channel_x'].to(device)
        ch_y  = batch['channel_y'].to(device)

        rf_logit  = model.forward_task(rf_x,  'rf')
        cfo_logit = model.forward_task(cfo_x, 'cfo')
        ch_logit  = model.forward_task(ch_x,  'channel')

        loss = (F.cross_entropy(rf_logit, rf_y.long())
                + F.mse_loss(cfo_logit, cfo_y)
                + F.mse_loss(ch_logit,  ch_y))
        total_loss += loss.item()

        rf_preds.append(rf_logit.argmax(1).cpu().numpy())
        rf_golds.append(rf_y.cpu().numpy())
        cfo_preds.append(cfo_logit.cpu().numpy())
        cfo_golds.append(cfo_y.cpu().numpy())
        ch_preds.append(ch_logit.cpu().numpy())
        ch_golds.append(ch_y.cpu().numpy())

    rf_acc = accuracy_score(
        np.concatenate(rf_golds), np.concatenate(rf_preds))
    cfo_r2 = r2_score(
        np.concatenate(cfo_golds).ravel(), np.concatenate(cfo_preds).ravel())
    ch_golds_all = np.concatenate(ch_golds)
    ch_preds_all = np.concatenate(ch_preds)
    ch_r2 = r2_score(ch_golds_all, ch_preds_all, multioutput='uniform_average')

    avg_loss = total_loss / max(len(loader), 1)
    return avg_loss, {'rf_accuracy': rf_acc, 'cfo_r2': cfo_r2, 'channel_r2': ch_r2}


def train_mtl(args) -> dict:
    """Train MTL NeuraCrypt — the fair analog of rep_lr MTL."""
    print('\n' + '='*60)
    print('  NeuraCrypt MTL  (shared encoder + ViT + 3 heads)')
    print('  — directly comparable to rep_lr MTL —')
    print('='*60)

    from argparse import Namespace
    device = torch.device(
        f'cuda:{args.gpu_id}' if torch.cuda.is_available() else 'cpu')

    ds_args = Namespace(
        pkl_dataset_path=args.pkl_dataset_path,
        num_chan=2, img_size=[256, 256],
        private_kernel_size=args.private_kernel_size,
        private_depth=args.private_depth,
        hidden_dim=args.hidden_dim,
        remove_pixel_shuffle=False,
        load_data_from_encoded_dir=False,
        exam_to_year_dict={},
    )
    IQMTLDataset.set_args(ds_args)

    train_ds = IQMTLDataset(ds_args, None, 'train')
    dev_ds   = IQMTLDataset(ds_args, None, 'dev')
    test_ds  = IQMTLDataset(ds_args, None, 'test')

    train_ld = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                          num_workers=args.num_workers, pin_memory=True,
                          drop_last=True)
    dev_ld   = DataLoader(dev_ds,  batch_size=args.batch_size, shuffle=False,
                          num_workers=args.num_workers, pin_memory=True)
    test_ld  = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                          num_workers=args.num_workers, pin_memory=True)

    model = MTLNeuraCrypt(
        encoder_args=ds_args,
        dim=args.hidden_dim,
        depth=args.num_layers,
        heads=args.num_heads,
        mlp_dim=args.hidden_dim * 2,
    ).to(device)

    # Only backbone + heads are trainable (encoder stays frozen)
    trainable = [p for p in model.parameters() if p.requires_grad]
    print(f'Trainable params: {sum(p.numel() for p in trainable):,}')
    print(f'Frozen  params:   {sum(p.numel() for p in model.encoder.parameters()):,}')

    optimizer = torch.optim.Adam(trainable, lr=1e-3)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', patience=5, factor=0.5)

    os.makedirs(args.save_dir,    exist_ok=True)
    os.makedirs(args.results_dir, exist_ok=True)
    ckpt_path = os.path.join(args.save_dir, 'neuracrypt_mtl_best.pt')

    best_rf_acc = -1.0
    best_metrics = {}

    for epoch in range(1, args.epochs + 1):
        model.backbone.train()
        model.rf_head.train(); model.cfo_head.train(); model.channel_head.train()

        epoch_loss = 0.0
        for batch in tqdm(train_ld, desc=f'Epoch {epoch}/{args.epochs}',
                          leave=False):
            rf_x  = batch['rf_x'].to(device)
            rf_y  = batch['rf_y'].to(device)
            cfo_x = batch['cfo_x'].to(device)
            cfo_y = batch['cfo_y'].to(device)
            ch_x  = batch['channel_x'].to(device)
            ch_y  = batch['channel_y'].to(device)

            rf_logit  = model.forward_task(rf_x,  'rf')
            cfo_logit = model.forward_task(cfo_x, 'cfo')
            ch_logit  = model.forward_task(ch_x,  'channel')

            loss = (F.cross_entropy(rf_logit, rf_y.long())
                    + F.mse_loss(cfo_logit, cfo_y)
                    + F.mse_loss(ch_logit,  ch_y))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        dev_loss, dev_m = evaluate_mtl(model, dev_ld, device)
        improved = dev_m['rf_accuracy'] > best_rf_acc
        if improved:
            best_rf_acc = dev_m['rf_accuracy']
            torch.save({'backbone': model.backbone.state_dict(),
                        'rf_head':  model.rf_head.state_dict(),
                        'cfo_head': model.cfo_head.state_dict(),
                        'ch_head':  model.channel_head.state_dict()},
                       ckpt_path)
            best_metrics = dev_m

        print(f'  Epoch {epoch:3d}  '
              f'train_loss={epoch_loss/len(train_ld):.4f}  '
              f'dev_loss={dev_loss:.4f}  '
              f'rf_acc={dev_m["rf_accuracy"]:.4f}  '
              f'cfo_r2={dev_m["cfo_r2"]:.4f}  '
              f'ch_r2={dev_m["channel_r2"]:.4f}'
              + ('  ✓' if improved else ''))

        scheduler.step(dev_m['rf_accuracy'])

    # Evaluate best checkpoint on test set
    ckpt = torch.load(ckpt_path, map_location=device)
    model.backbone.load_state_dict(ckpt['backbone'])
    model.rf_head.load_state_dict(ckpt['rf_head'])
    model.cfo_head.load_state_dict(ckpt['cfo_head'])
    model.channel_head.load_state_dict(ckpt['ch_head'])

    test_loss, test_m = evaluate_mtl(model, test_ld, device)
    print(f'\nTest  rf_acc={test_m["rf_accuracy"]:.4f}  '
          f'cfo_r2={test_m["cfo_r2"]:.4f}  ch_r2={test_m["channel_r2"]:.4f}')

    return {'test_loss': test_loss, **test_m}


# ══════════════════════════════════════════════════════════════════════════════
# Training — single-task (optional ablation)
# ══════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def evaluate_single(model, loader, device, task):
    model.eval()
    preds, golds, total_loss = [], [], 0.0
    for batch in loader:
        x = batch['x'].to(device)
        y = batch['y'].to(device)
        logit = model(x)
        if task == 'rf_fingerprinting':
            loss = F.cross_entropy(logit, y.long())
            preds.append(logit.argmax(1).cpu().numpy())
        else:
            loss = F.mse_loss(logit, y.view_as(logit))
            preds.append(logit.cpu().numpy())
        golds.append(y.cpu().numpy())
        total_loss += loss.item()

    preds = np.concatenate(preds)
    golds = np.concatenate(golds)
    avg_loss = total_loss / max(len(loader), 1)

    extra = {}
    if task == 'rf_fingerprinting':
        extra['accuracy'] = accuracy_score(golds.ravel(), preds.ravel())
    else:
        g = golds.reshape(len(golds), -1)
        p = preds.reshape(len(preds), -1)
        extra['r2'] = r2_score(g.ravel() if p.shape[1] == 1 else g,
                               p.ravel() if p.shape[1] == 1 else p,
                               multioutput='uniform_average'
                               if p.shape[1] > 1 else None)
    return avg_loss, extra


def train_single_task(task: str, args) -> dict:
    """Train one NeuraCrypt model per task (single-task ablation)."""
    print(f'\n{"="*60}\n  NeuraCrypt single-task: {task}\n{"="*60}')

    from argparse import Namespace
    task_to_cls = {'rf_fingerprinting': IQRFDataset,
                   'cfo_estimation':    IQCFODataset,
                   'channel_estimation': IQChannelDataset}
    task_to_nc  = {'rf_fingerprinting': 16,
                   'cfo_estimation':     1,
                   'channel_estimation': 104}

    device = torch.device(
        f'cuda:{args.gpu_id}' if torch.cuda.is_available() else 'cpu')
    ds_cls = task_to_cls[task]
    nc     = task_to_nc[task]

    ds_args = Namespace(
        pkl_dataset_path=args.pkl_dataset_path,
        num_chan=2, img_size=[256, 256],
        num_classes=nc,
        private_kernel_size=args.private_kernel_size,
        private_depth=args.private_depth,
        hidden_dim=args.hidden_dim,
        remove_pixel_shuffle=False,
        load_data_from_encoded_dir=False,
        exam_to_year_dict={},
    )
    ds_cls.set_args(ds_args)

    def make_loader(split, shuffle):
        ds = ds_cls(ds_args, None, split)
        return DataLoader(ds, batch_size=args.batch_size, shuffle=shuffle,
                          num_workers=args.num_workers, pin_memory=True,
                          drop_last=shuffle)

    train_ld = make_loader('train', True)
    dev_ld   = make_loader('dev',   False)
    test_ld  = make_loader('test',  False)

    model = SingleTaskNeuraCrypt(ds_args, nc, args.hidden_dim, args.num_layers,
                                 args.num_heads, args.hidden_dim * 2).to(device)
    opt   = torch.optim.Adam(
        [p for p in model.parameters() if p.requires_grad], lr=1e-3)
    is_cls = task == 'rf_fingerprinting'
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode='max' if is_cls else 'min', patience=5, factor=0.5)

    ckpt = os.path.join(args.save_dir, f'neuracrypt_{task}_best.pt')
    best, best_m = (-1 if is_cls else np.inf), {}

    for epoch in range(1, args.epochs + 1):
        model.backbone.train(); model.head.train()
        ep_loss = 0.0
        for batch in tqdm(train_ld, desc=f'Ep {epoch}', leave=False):
            x = batch['x'].to(device); y = batch['y'].to(device)
            logit = model(x)
            loss = (F.cross_entropy(logit, y.long()) if is_cls
                    else F.mse_loss(logit, y.view_as(logit)))
            opt.zero_grad(); loss.backward(); opt.step()
            ep_loss += loss.item()
        dev_loss, dev_m = evaluate_single(model, dev_ld, device, task)
        score = dev_m.get('accuracy', -dev_loss)
        if (is_cls and score > best) or (not is_cls and -dev_loss > best):
            best = score if is_cls else -dev_loss
            torch.save(model.state_dict(), ckpt)
            best_m = dev_m
        print(f'  Ep {epoch:3d}  loss={ep_loss/len(train_ld):.4f}  '
              + '  '.join(f'{k}={v:.4f}' for k, v in dev_m.items()))
        sched.step(score)

    model.load_state_dict(torch.load(ckpt, map_location=device))
    test_loss, test_m = evaluate_single(model, test_ld, device, task)
    print(f'Test  {test_m}')
    return {'test_loss': test_loss, **test_m}


# ══════════════════════════════════════════════════════════════════════════════
# Adversary reconstruction attack
#
# Architecture mirrors code/dra/models.py::SimpleAdversary exactly:
#   3-layer MLP  latent → 512 → 512 → 512 → output_flat
#   Kaiming init, LeakyReLU(0.01), Dropout(0.2), no final activation
#   MSE loss, Adam(lr=1e-3, wd=1e-5), CosineAnnealingLR
#
# The only NeuraCrypt-specific adaptation:
#   • input  = mean-pooled tokens  (B, 256, 2048) → (B, 2048)
#   • target = the image fed to the encoder (B, 2, 256, 256)
#     because that is NeuraCrypt's actual "encoded input"
# ══════════════════════════════════════════════════════════════════════════════

class _SimpleAdversary(nn.Module):
    """Identical to code/dra/models.py::SimpleAdversary.

    latent_dim → [512 → 512 → 512] → output_flat → reshape to output_shape
    """

    def __init__(self, latent_dim: int, output_shape: tuple,
                 hidden_dim: int = 512, dropout: float = 0.2):
        super().__init__()
        self.output_shape = output_shape
        out_dim = 1
        for d in output_shape:
            out_dim *= d

        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
        )
        self.apply(self._init)

    def _init(self, m):
        if isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight, mode='fan_in',
                                    nonlinearity='leaky_relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z).view(-1, *self.output_shape)


def run_adversary_attack(model: MTLNeuraCrypt,
                         train_ld: DataLoader,
                         test_ld:  DataLoader,
                         device:   torch.device,
                         mtl_metrics: dict,
                         adv_epochs: int = 50,
                         adv_hidden:  int = 512,
                         adv_dropout: float = 0.2) -> dict:
    """Train one SimpleAdversary per task and report reconstruction MSE.

    For each task the adversary sees mean-pooled NeuraCrypt tokens and must
    reconstruct the image that was passed into the private encoder.
    This matches the rep_lr adversary setup (same architecture, same loss,
    same optimizer/scheduler) while targeting NeuraCrypt's actual input.

    Returns
    -------
    dict  keyed by task name, each with 'recon_mse' and 'utility':
        {
          'rf':      {'recon_mse': float, 'utility': rf_accuracy},
          'cfo':     {'recon_mse': float, 'utility': cfo_r2},
          'channel': {'recon_mse': float, 'utility': channel_r2},
        }
    """
    print('\n' + '='*60)
    print('  Adversary Reconstruction Attack  (mirrors code/dra setup)')
    print('='*60)

    # image shape fed into the private encoder
    img_shape = (2, 256, 256)    # (C, H, W)
    # pooled token dim after mean over patch dim
    token_dim = model.encoder.pos_embedding.shape[-1]  # hidden_dim

    task_keys = {
        'rf':      ('rf_x',      'rf_accuracy'),
        'cfo':     ('cfo_x',     'cfo_r2'),
        'channel': ('channel_x', 'channel_r2'),
    }

    results = {}

    for task_name, (img_key, util_key) in task_keys.items():
        print(f'\n  Task: {task_name}')

        # ── collect (pooled_token, image) pairs ─────────────────────────────
        @torch.no_grad()
        def _collect(loader, key=img_key):
            zs, imgs = [], []
            for batch in loader:
                x = batch[key].to(device)               # (B, 2, 256, 256)
                tokens = model.encode(x)                 # (B, 256, hidden_dim)
                z = tokens.mean(dim=1)                   # (B, hidden_dim)
                zs.append(z.cpu());  imgs.append(x.cpu())
            return torch.cat(zs), torch.cat(imgs)

        z_train, img_train = _collect(train_ld)
        z_test,  img_test  = _collect(test_ld)

        from torch.utils.data import TensorDataset
        tr_ds = TensorDataset(z_train, img_train)
        te_ds = TensorDataset(z_test,  img_test)
        tr_ld_adv = DataLoader(tr_ds, batch_size=64, shuffle=True,  pin_memory=True)
        te_ld_adv = DataLoader(te_ds, batch_size=64, shuffle=False, pin_memory=True)

        # ── build adversary ─────────────────────────────────────────────────
        adv = _SimpleAdversary(token_dim, img_shape,
                               hidden_dim=adv_hidden,
                               dropout=adv_dropout).to(device)
        n_params = sum(p.numel() for p in adv.parameters())
        print(f'    Adversary params: {n_params:,}')

        opt   = torch.optim.Adam(adv.parameters(), lr=1e-3, weight_decay=1e-5)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=adv_epochs)
        crit  = nn.MSELoss()

        best_mse = float('inf')
        for epoch in range(1, adv_epochs + 1):
            adv.train()
            for z_b, img_b in tr_ld_adv:
                z_b   = z_b.to(device);  img_b = img_b.to(device)
                loss  = crit(adv(z_b), img_b)
                opt.zero_grad(); loss.backward(); opt.step()
            sched.step()

            if epoch % 10 == 0 or epoch == adv_epochs:
                adv.eval()
                with torch.no_grad():
                    mse_sum, n = 0.0, 0
                    for z_b, img_b in te_ld_adv:
                        z_b = z_b.to(device); img_b = img_b.to(device)
                        mse_sum += crit(adv(z_b), img_b).item() * z_b.size(0)
                        n += z_b.size(0)
                test_mse = mse_sum / n
                best_mse = min(best_mse, test_mse)
                print(f'    Ep {epoch:3d}/{adv_epochs}  test_mse={test_mse:.6f}')

        results[task_name] = {
            'recon_mse': best_mse,
            'utility':   mtl_metrics.get(util_key, float('nan')),
        }
        print(f'    → recon_mse={best_mse:.6f}  utility={results[task_name]["utility"]:.6f}')

    # pretty-print the tradeoff table
    print('\n' + '='*60)
    print('  Privacy–Utility Tradeoff (NeuraCrypt)')
    print(f'  {"task":<10} {"recon_mse":>12}   {"utility":>10}')
    print('  ' + '-'*36)
    for t, v in results.items():
        print(f'  {t:<10} {v["recon_mse"]:>12.6f}   {v["utility"]:>10.6f}')
    print('='*60)

    return results


# ══════════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--mode', choices=['mtl', 'single_task', 'both'],
                   default='mtl',
                   help='mtl = fair comparison (default); '
                        'single_task = ablation; both = run both')
    p.add_argument('--pkl_dataset_path',
                   default='dataset/rf_partition_dict_0.5.pkl')
    p.add_argument('--save_dir',    default='baselines/NeuraCrypt/snapshots_iq')
    p.add_argument('--results_dir', default='baselines/NeuraCrypt/results_iq')
    p.add_argument('--gpu_id',      type=int, default=0)
    p.add_argument('--epochs',      type=int, default=50)
    p.add_argument('--batch_size',  type=int, default=128)
    p.add_argument('--num_workers', type=int, default=8)
    p.add_argument('--hidden_dim',  type=int, default=2048)
    p.add_argument('--num_layers',  type=int, default=1)
    p.add_argument('--num_heads',   type=int, default=8)
    p.add_argument('--private_kernel_size', type=int, default=16)
    p.add_argument('--private_depth',       type=int, default=4)
    p.add_argument('--tasks', nargs='+',
                   default=['rf_fingerprinting', 'cfo_estimation',
                            'channel_estimation'],
                   help='Which tasks for single_task mode')
    p.add_argument('--adv_epochs', type=int, default=50,
                   help='Epochs to train adversary reconstruction attack '
                        '(matches rep_lr dra default of 50)')
    p.add_argument('--skip_adversary', action='store_true',
                   help='Skip adversary attack (faster, utility metrics only)')
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.save_dir,    exist_ok=True)
    os.makedirs(args.results_dir, exist_ok=True)

    out = {}

    if args.mode in ('mtl', 'both'):
        mtl_result = train_mtl(args)
        out['mtl'] = mtl_result
        # Build summary matching the rep_lr format exactly
        summary = {
            'rf_accuracy': mtl_result.get('rf_accuracy',  float('nan')),
            'cfo_r2':      mtl_result.get('cfo_r2',        float('nan')),
            'channel_r2':  mtl_result.get('channel_r2',    float('nan')),
            'final_mse':   mtl_result.get('test_loss',      float('nan')),
        }
        out['summary'] = summary

        # ── Adversary reconstruction attack ──────────────────────────────
        if not args.skip_adversary:
            # Re-build the model and load the best checkpoint to run attack
            from argparse import Namespace
            device = torch.device(
                f'cuda:{args.gpu_id}' if torch.cuda.is_available() else 'cpu')
            ds_args = Namespace(
                pkl_dataset_path=args.pkl_dataset_path,
                num_chan=2, img_size=[256, 256],
                private_kernel_size=args.private_kernel_size,
                private_depth=args.private_depth,
                hidden_dim=args.hidden_dim,
                remove_pixel_shuffle=False,
                load_data_from_encoded_dir=False,
                exam_to_year_dict={},
            )
            IQMTLDataset.set_args(ds_args)
            adv_model = MTLNeuraCrypt(
                encoder_args=ds_args,
                dim=args.hidden_dim,
                depth=args.num_layers,
                heads=args.num_heads,
                mlp_dim=args.hidden_dim * 2,
            ).to(device)
            ckpt_path = os.path.join(args.save_dir, 'neuracrypt_mtl_best.pt')
            ckpt = torch.load(ckpt_path, map_location=device)
            adv_model.backbone.load_state_dict(ckpt['backbone'])
            adv_model.rf_head.load_state_dict(ckpt['rf_head'])
            adv_model.cfo_head.load_state_dict(ckpt['cfo_head'])
            adv_model.channel_head.load_state_dict(ckpt['ch_head'])

            train_ds = IQMTLDataset(ds_args, None, 'train')
            test_ds  = IQMTLDataset(ds_args, None, 'test')
            adv_train_ld = DataLoader(train_ds, batch_size=args.batch_size,
                                      shuffle=True, num_workers=args.num_workers,
                                      pin_memory=True, drop_last=True)
            adv_test_ld  = DataLoader(test_ds,  batch_size=args.batch_size,
                                      shuffle=False, num_workers=args.num_workers,
                                      pin_memory=True)

            adv_results = run_adversary_attack(
                adv_model, adv_train_ld, adv_test_ld, device,
                mtl_metrics=mtl_result,
                adv_epochs=args.adv_epochs,
            )
            out['privacy_utility'] = adv_results

    if args.mode in ('single_task', 'both'):
        st_results = {}
        for task in args.tasks:
            st_results[task] = train_single_task(task, args)
        out['single_task'] = st_results

    out_path = os.path.join(args.results_dir, 'neuracrypt_results.json')
    with open(out_path, 'w') as f:
        json.dump(out, f, indent=2)

    print('\n' + '='*60)
    print('NeuraCrypt IQ Benchmark – Summary')
    print('='*60)
    if 'summary' in out:
        for k, v in out['summary'].items():
            print(f'  {k:<20s}: {v:.4f}')
    if 'privacy_utility' in out:
        print('\n  Privacy–Utility Tradeoff (compare with rep_lr):')
        print(f'  {"task":<10} {"recon_mse (↑=private)":>22}   {"utility":>10}')
        print('  ' + '-'*46)
        for t, v in out['privacy_utility'].items():
            print(f'  {t:<10} {v["recon_mse"]:>22.6f}   {v["utility"]:>10.6f}')
        print()
        print('  rep_lr reference:')
        print('  rf         0.500941              0.518564')
        print('  cfo        0.345215              0.307801')
        print('  channel    0.337255              0.405239')
    print(f'\nResults → {out_path}')


if __name__ == '__main__':
    main()
