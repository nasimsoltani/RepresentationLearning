"""
Unbiased reconstruction attack for the FIL baseline.

Attack-ub from the FIL paper (Section 3.3):
    x̂ = argmin_{x₀} || e_noisy - Enc(x₀) ||²
         (no regulariser → unbiased)

This is a white-box optimization attack on a single task input at a time,
keeping the other two task inputs fixed.  It directly lower-bounds the
Cramér-Rao bound and is the attack that FIL's dFIL metric is calibrated to.

The attack is run per-sample (batch_size=1 style inside the function) because
each sample needs its own optimisation trajectory.

Public API
----------
run_unbiased_attack(model, data_loader, beta, task, device, ...) -> dict
    Runs the attack over a test set and returns mean reconstruction MSE.
"""

import os
import sys

import torch
import torch.optim as optim
from tqdm import tqdm

# ── Path setup ────────────────────────────────────────────────────────────────
_BASELINE_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_BASELINE_DIR, '..', '..'))
_CODE = os.path.join(_ROOT, 'code')
for p in [_CODE, os.path.join(_CODE, 'rep_lr')]:
    if p not in sys.path:
        sys.path.insert(0, p)

from fil_encoder import FILJointEncoder


# ──────────────────────────────────────────────────────────────────────────────

def _inject_noise(z: torch.Tensor, beta: float) -> torch.Tensor:
    """
    Inject isotropic Gaussian noise with total variance beta on the
    unit-norm activation — exactly matching inject_isotropic_noise() in the
    main codebase so the comparison is apples-to-apples.

    Per-dim std = sqrt(beta / d),  applied on the L2-normalised vector.
    """
    if beta <= 0.0:
        return z
    z_norm = torch.linalg.norm(z, dim=-1, keepdim=True).clamp(min=1e-7)
    z_n = z / z_norm
    d = z.shape[-1]
    std = (beta / d) ** 0.5
    z_n_noisy = z_n + torch.randn_like(z_n) * std
    return z_n_noisy * z_norm


def reconstruct_single(
    model: FILJointEncoder,
    rf_x: torch.Tensor,
    cfo_x: torch.Tensor,
    channel_x: torch.Tensor,
    e_noisy: torch.Tensor,
    task: str,
    device: torch.device,
    lr: float = 1e-2,
    niters: int = 2000,
    stop_tol: float = 1e-5,
    window: int = 200,
) -> torch.Tensor:
    """
    Optimise x̂ for a single sample to minimise ||e_noisy - Enc(..., x̂, ...)||².

    Args:
        model        : FILJointEncoder (eval mode, on device)
        rf_x         : (1,2,L) RF input — used as fixed context or optimisation target
        cfo_x        : (1,2,160) CFO input
        channel_x    : (1,2,160) Channel input
        e_noisy      : (1,latent_dim) noisy encoding for this sample
        task         : 'rf' | 'cfo' | 'channel'  — which input to reconstruct
        lr           : Adam learning rate
        niters       : max optimisation steps
        stop_tol     : relative loss-change convergence criterion (over `window` steps)

    Returns:
        x_recon : reconstructed input (1, 2, L_task), detached on CPU
    """
    model.eval()

    # Initialise the target input from Gaussian noise (same shape as truth)
    if task == 'rf':
        x_init = torch.randn_like(rf_x) * 0.1
    elif task == 'cfo':
        x_init = torch.randn_like(cfo_x) * 0.1
    else:
        x_init = torch.randn_like(channel_x) * 0.1

    x_recon = x_init.clone().requires_grad_(True).to(device)
    optimizer = optim.Adam([x_recon], lr=lr)

    e_noisy_det = e_noisy.detach()
    a_ref_sq = (e_noisy_det ** 2).mean().clamp(min=1e-8)

    prev_window_loss = None
    window_loss = 0.0

    for i in range(niters):
        optimizer.zero_grad()

        if task == 'rf':
            e_hat = model.forward_first(x_recon, cfo_x, channel_x)
        elif task == 'cfo':
            e_hat = model.forward_first(rf_x, x_recon, channel_x)
        else:
            e_hat = model.forward_first(rf_x, cfo_x, x_recon)

        # Normalised MSE in feature space (same as FIL recon.py)
        loss = ((e_noisy_det - e_hat) ** 2).mean() / a_ref_sq

        loss.backward()
        optimizer.step()
        window_loss += loss.item()

        if (i + 1) % window == 0:
            window_loss /= window
            if prev_window_loss is not None:
                rel_change = abs(prev_window_loss - window_loss) / (prev_window_loss + 1e-12)
                if rel_change < stop_tol:
                    break
            prev_window_loss = window_loss
            window_loss = 0.0

    return x_recon.detach().cpu()


def run_unbiased_attack(
    model: FILJointEncoder,
    data_loader,
    beta: float,
    task: str,
    device: torch.device,
    lr: float = 1e-2,
    niters: int = 2000,
    stop_tol: float = 1e-5,
    max_samples: int = 200,
) -> dict:
    """
    Run the unbiased reconstruction attack over a test set.

    For each sample:
      1. Encode (rf_x, cfo_x, channel_x) → z
      2. Add noise(beta) → z_noisy
      3. Optimise x̂ = argmin ||z_noisy - Enc(..., x̂, ...)||²
      4. Compute MSE(x̂, x_true) in input space

    Args:
        model       : FILJointEncoder (eval mode)
        data_loader : yields (rf_x, rf_y, cfo_x, cfo_y, ch_x, ch_y, path)
        beta        : noise level (total variance on unit-norm activation)
        task        : 'rf' | 'cfo' | 'channel'
        max_samples : cap to keep runtime manageable

    Returns:
        dict with keys:
            'mse'  : mean per-sample per-element MSE in input space
            'nmse' : normalised MSE (divided by signal power)
    """
    model.eval()
    mse_list  = []
    nmse_list = []
    n_done = 0

    for batch in tqdm(data_loader, desc=f'Attack-ub [{task}] beta={beta}'):
        rf_x, _, cfo_x, _, ch_x, _, _ = batch

        # Process sample-by-sample (optimisation is per-sample)
        for s in range(rf_x.size(0)):
            if n_done >= max_samples:
                break

            rf_s  = rf_x[s:s+1].to(device).float()
            cfo_s = cfo_x[s:s+1].to(device).float()
            ch_s  = ch_x[s:s+1].to(device).float()

            # Ground truth input for the attacked task
            if task == 'rf':
                x_true = rf_s
            elif task == 'cfo':
                x_true = cfo_s
            else:
                x_true = ch_s

            # Encode + inject noise
            with torch.no_grad():
                z = model.forward_first(rf_s, cfo_s, ch_s)
            z_noisy = _inject_noise(z, beta)

            # Reconstruct
            x_hat = reconstruct_single(
                model, rf_s, cfo_s, ch_s, z_noisy, task, device,
                lr=lr, niters=niters, stop_tol=stop_tol,
            ).to(device)

            # Compute MSE in input space
            mse  = ((x_hat - x_true) ** 2).mean().item()
            sig2 = (x_true ** 2).mean().item()
            nmse = mse / (sig2 + 1e-8)

            mse_list.append(mse)
            nmse_list.append(nmse)
            n_done += 1

        if n_done >= max_samples:
            break

    import numpy as np
    return {
        'mse':  float(np.mean(mse_list)),
        'nmse': float(np.mean(nmse_list)),
        'n_samples': n_done,
    }
