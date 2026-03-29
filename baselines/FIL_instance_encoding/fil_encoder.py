"""
FIL-compatible joint encoder for RF/CFO/Channel multi-task learning.

Key differences from the main codebase encoder:
  - GELU activations (smooth everywhere, required for valid FIM)
  - AvgPool1d instead of MaxPool1d (differentiable)

This module is SELF-CONTAINED: it defines its own encoder/projector classes so
the main codebase (code/rep_lr/models.py) is untouched.  Task heads are reused
from the main codebase (they do not affect FIM validity since FIM is defined at
the encoder output, before heads).
"""

import os
import sys
import math
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.func import vmap, jvp

# ── Path setup so we can reuse task heads from the main codebase ───────────────
_BASELINE_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_BASELINE_DIR, '..', '..'))
_CODE = os.path.join(_ROOT, 'code')
for p in [_CODE, os.path.join(_CODE, 'rep_lr')]:
    if p not in sys.path:
        sys.path.insert(0, p)

from rep_lr.models import (
    RFClassificationHead,
    ChannelEstimationHead,
    CFOEstimationHead,
    SimpleCFOEstimationHead,
    CFOAdaptiveHead,
)


# ──────────────────────────────────────────────────────────────────────────────
# Projector  (same structure as ComplexSequenceProjector in main codebase)
# ──────────────────────────────────────────────────────────────────────────────

class FILComplexSequenceProjector(nn.Module):
    """
    Projects I/Q sequences to a common length using separate MLPs.
    Uses GELU (no non-smooth ops) for FIL compatibility.

    Input:  (B, 2, input_seq_len)
    Output: (B, 2, output_seq_len)
    """
    def __init__(self, input_seq_len: int, output_seq_len: int, hidden_dim: int = 512):
        super().__init__()
        self.real_proj = nn.Sequential(
            nn.Linear(input_seq_len, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, output_seq_len),
        )
        self.imag_proj = nn.Sequential(
            nn.Linear(input_seq_len, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, output_seq_len),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        real = self.real_proj(x[:, 0, :]).unsqueeze(1)   # (B,1,L')
        imag = self.imag_proj(x[:, 1, :]).unsqueeze(1)   # (B,1,L')
        return torch.cat([real, imag], dim=1)             # (B,2,L')


# ──────────────────────────────────────────────────────────────────────────────
# FIL-compatible Encoder
# ──────────────────────────────────────────────────────────────────────────────

class FILEncoder(nn.Module):
    """
    Convolutional encoder with GELU activations and AvgPool.
    Drop-in replacement for the main codebase Encoder class.

    Input:  (B, input_channels, slice_size)
    Output: (B, 2, output_dim)
    """
    def __init__(
        self,
        slice_size: int,
        output_dim: int = 128,
        dropout: float = 0.25,
        num_blocks: int = 3,
        input_channels: int = 2,
    ):
        super().__init__()
        self.output_dim = output_dim
        self.num_blocks = num_blocks
        channel = 64

        self.conv_layers = nn.ModuleList()
        self.bn_layers = nn.ModuleList()
        for i in range(num_blocks):
            in_ch = input_channels if i == 0 else channel
            self.conv_layers.append(nn.Conv1d(in_ch,  channel, kernel_size=7, padding="same"))
            self.bn_layers.append(nn.BatchNorm1d(channel))
            self.conv_layers.append(nn.Conv1d(channel, channel, kernel_size=5, padding="same"))
            self.bn_layers.append(nn.BatchNorm1d(channel))

        # AvgPool instead of MaxPool (smooth, differentiable)
        self.pool = nn.AvgPool1d(2, 2)
        self.flatten = nn.Flatten()
        self.act = nn.GELU()

        conv_output_size = channel * (slice_size // (2 ** num_blocks))

        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(conv_output_size, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, 2 * output_dim),
        )

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for i in range(self.num_blocks):
            idx = i * 2
            residual = x  # save block input (same shape, used as skip if channels match)
            x = self.act(self.bn_layers[idx](self.conv_layers[idx](x)))
            x = self.bn_layers[idx + 1](self.conv_layers[idx + 1](x))
            # Skip connection only from block 1 onward (block 0 changes channel dim 2→64)
            if i > 0:
                x = self.act(x + residual)
            else:
                x = self.act(x)
            x = self.pool(x)
        out = self.classifier(self.flatten(x))
        return out.view(out.size(0), 2, self.output_dim)


# ──────────────────────────────────────────────────────────────────────────────
# Joint MTL encoder (wraps projectors + encoder + task heads)
# ──────────────────────────────────────────────────────────────────────────────

class FILJointEncoder(nn.Module):
    """
    Joint RF+CFO+Channel encoder following the FIL split-inference interface.

      forward_first(rf_x, cfo_x, channel_x)  →  latent z  (B, latent_dim)
      forward_second(z, sigma=None)           →  {'rf': logits, 'cfo': pred, 'channel': pred}

    Noise is injected inside forward_second when sigma is not None, matching
    the FIL paper's noisy-inference pattern.

    Args:
        slice_len       : RF input length (default 1024)
        proj_seq_len    : common length after projection (default 256)
        proj_hidden_dim : hidden dim of the projector MLP (default 512)
        d2              : half of encoder output dim; latent_dim = 2*d2 (default 128)
        num_classes     : number of RF devices (default 16)
        dropout         : dropout probability (default 0.1)
        head_hidden_dim : hidden dim of task heads (default 256)
        encoder_num_blocks: convolutional blocks in encoder (default 3)
        cfo_head_type   : 'default' | 'simple' | 'adaptive' (default 'simple')
    """

    def __init__(
        self,
        slice_len: int = 1024,
        proj_seq_len: int = 256,
        proj_hidden_dim: int = 512,
        d2: int = 128,
        num_classes: int = 16,
        dropout: float = 0.1,
        head_hidden_dim: int = 256,
        encoder_num_blocks: int = 3,
        cfo_head_type: str = 'simple',
    ):
        super().__init__()
        self.latent_dim = 2 * d2

        # Projectors
        self.proj_rf = FILComplexSequenceProjector(slice_len, proj_seq_len, proj_hidden_dim)
        self.proj_cfo = FILComplexSequenceProjector(160, proj_seq_len, proj_hidden_dim)
        self.proj_channel = FILComplexSequenceProjector(160, proj_seq_len, proj_hidden_dim)

        # Shared encoder (FIL-compatible: GELU + AvgPool)
        self.encoder = FILEncoder(
            slice_size=proj_seq_len,
            output_dim=d2,
            dropout=dropout,
            num_blocks=encoder_num_blocks,
            input_channels=2,
        )

        # Task heads (reused from main codebase)
        self.head_rf = RFClassificationHead(
            input_dim=self.latent_dim, num_classes=num_classes,
            hidden_dim=head_hidden_dim, dropout=dropout,
        )
        self.head_channel = ChannelEstimationHead(
            input_dim=self.latent_dim, hidden_dim=head_hidden_dim,
            output_length=52, dropout=dropout,
        )
        if cfo_head_type == 'simple':
            self.head_cfo = SimpleCFOEstimationHead(
                input_dim=self.latent_dim, hidden_dim=64, dropout=0.1
            )
        elif cfo_head_type == 'adaptive':
            self.head_cfo = CFOAdaptiveHead(
                input_dim=self.latent_dim, hidden_dim=128, dropout=0.1
            )
        else:
            self.head_cfo = CFOEstimationHead(
                input_dim=self.latent_dim, hidden_dim=head_hidden_dim, dropout=dropout
            )

    # ── FIL-style interface ───────────────────────────────────────────────────

    def forward_first(
        self,
        rf_x: torch.Tensor,
        cfo_x: torch.Tensor,
        channel_x: torch.Tensor,
    ) -> torch.Tensor:
        """Encode inputs → latent z of shape (B, latent_dim)."""
        projected = (
            self.proj_rf(rf_x.float())
            + self.proj_cfo(cfo_x.float())
            + self.proj_channel(channel_x.float())
        )                                                 # (B,2,proj_seq_len)
        z = self.encoder(projected)                       # (B,2,d2)
        return z.view(z.size(0), -1)                      # (B, latent_dim)

    def forward_second(
        self,
        z: torch.Tensor,
        sigma: float | torch.Tensor | None = None,
    ) -> dict:
        """
        Optionally add isotropic Gaussian noise (std=sigma per dim) then run heads.

        Returns dict: {'rf': (B,C), 'cfo': (B,1), 'channel': (B,2,52)}
        """
        if sigma is not None:
            noise = torch.randn_like(z) * sigma
            z = z + noise
        z2d = z.view(z.size(0), 2, -1)                   # (B,2,d2)
        return {
            'rf':      self.head_rf(z2d),
            'cfo':     self.head_cfo(z2d),
            'channel': self.head_channel(z2d),
        }

    def forward(self, rf_x, cfo_x, channel_x):
        """Full forward pass without noise."""
        z = self.forward_first(rf_x, cfo_x, channel_x)
        return self.forward_second(z)


# ──────────────────────────────────────────────────────────────────────────────
# Jacobian trace utilities (adapted from FIL baseline util.py)
# ──────────────────────────────────────────────────────────────────────────────

def calc_jacobian_trace_rf(
    model: FILJointEncoder,
    rf_x: torch.Tensor,
    cfo_x: torch.Tensor,
    channel_x: torch.Tensor,
    device: torch.device,
    jvp_parallelism: int = 10,
    subsample: int = 20,
) -> torch.Tensor:
    """
    Compute Tr(J^T J) where J = d(forward_first) / d(rf_x).

    Uses JVP (Jacobian-vector products) without materialising the full Jacobian,
    following the FIL codebase approach (util.py::calc_tr).

    Args:
        model          : FILJointEncoder (eval mode)
        rf_x           : (B,2,L) RF input on device
        cfo_x          : (B,2,160) CFO input on device (held fixed)
        channel_x      : (B,2,160) Channel input on device (held fixed)
        jvp_parallelism: number of tangent vectors to process in parallel
        subsample      : number of input dimensions to probe (< 2*L for speed)

    Returns:
        tr : (B,) per-sample trace estimates
    """
    rf_x = rf_x.float()
    cfo_x = cfo_x.float()
    channel_x = channel_x.float()

    d = rf_x[0].flatten().shape[0]   # 2 * L

    def forward_rf_only(rf):
        return model.forward_first(rf, cfo_x, channel_x)

    def jvp_func(rf, tgt):
        return jvp(forward_rf_only, (rf,), (tgt,))

    tr = torch.zeros(rf_x.shape[0], dtype=rf_x.dtype, device=device)

    dims = random.sample(range(d), min(d, subsample)) if subsample > 0 else range(d)

    for j in range(math.ceil(len(dims) / jvp_parallelism)):
        chunk = list(dims)[j * jvp_parallelism: (j + 1) * jvp_parallelism]
        tgts = []
        for k in chunk:
            tgt = torch.zeros_like(rf_x).reshape(rf_x.shape[0], -1)
            tgt[:, k] = 1.0
            tgts.append(tgt.reshape(rf_x.shape))
        tgts = torch.stack(tgts)   # (chunk, B, 2, L)

        def helper(tgt):
            _, grad = vmap(jvp_func, randomness='same')(rf_x, tgt)
            return torch.sum(grad * grad, dim=tuple(range(1, len(grad.shape))))

        trs = vmap(helper, randomness='same')(tgts)    # (chunk, B)
        tr += trs.sum(dim=0)

    if subsample > 0:
        tr *= d / len(dims)

    return tr


def compute_mean_trace(
    model: FILJointEncoder,
    data_loader,
    device: torch.device,
    jvp_parallelism: int = 10,
    subsample: int = 20,
    num_batches: int = 20,
) -> tuple[float, int]:
    """
    Compute the mean Jacobian trace over the dataset.

    Returns:
        mean_tr : average Tr(J^T J) per sample
        d       : input dimension (2 * RF_slice_len)
    """
    model.eval()
    tr_sum = 0.0
    cnt = 0
    d = None

    with torch.no_grad():
        for i, batch in enumerate(data_loader):
            if i >= num_batches:
                break
            rf_x, _, cfo_x, _, channel_x, _, _ = batch
            rf_x     = rf_x.to(device).float()
            cfo_x    = cfo_x.to(device).float()
            channel_x = channel_x.to(device).float()

            if d is None:
                d = rf_x[0].flatten().shape[0]

            tr = calc_jacobian_trace_rf(
                model, rf_x, cfo_x, channel_x, device,
                jvp_parallelism=jvp_parallelism, subsample=subsample,
            )
            tr_sum += tr.sum().item()
            cnt += rf_x.shape[0]

    return (tr_sum / cnt) if cnt > 0 else 0.0, d
