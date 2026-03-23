"""
HCR (Hammersley-Chapman-Robbins) bounds adapted for signal domain.

Adapted from baselines/hcrbounds/codes/testers.py (Meta, 2024).
Computes theoretical lower bounds on MSE of input reconstruction
from noisy latent features for our signal encoder pipeline.

Key adaptations from the original image code:
  - Encoder pipeline: projection -> encoder (instead of ResNet/CIFAR net)
  - Signal domain MSE bounds (no DCT) — numerator = ||input_perturbation||^2
  - noise_level -> sigma mapping: sigma = sqrt(noise_level / latent_dim)
  - Returns per-task numerical MSE lower bounds (no image saving)
"""

import math
import os
import sys

import numpy as np
import scipy.sparse.linalg
import torch
import torch.nn as nn

# Allow imports from the project root and code directory
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
_CODE = os.path.join(_ROOT, 'code')
if _CODE not in sys.path:
    sys.path.insert(0, _CODE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from dra_1.model_loader import load_model_from_experiment


def build_encoder_pipeline(experiment_path, device):
    """
    Load trained projection + encoder and return model_data dict.

    Reuses load_model_from_experiment() from model_loader.
    The returned dict contains 'model', 'train_args', 'is_mtl'.
    """
    model_data = load_model_from_experiment(
        experiment_path, device=device, load_heads=False
    )
    model_data['model'].eval()
    return model_data


def _make_task_forward_fn(model_data, task, batch, device):
    """
    Build a callable  x_task -> flat_features  for a specific task.

    For MTL models the other tasks' projections are computed from the
    batch and held fixed (detached), so that torch.func.jvp / vjp only
    differentiates through the target task's branch.

    Args:
        model_data : dict returned by build_encoder_pipeline / load_model_from_experiment
        task       : task name, e.g. 'rf_fingerprinting'
        batch      : one batch from ActivationDataset
        device     : torch.device

    Returns:
        Callable (x_task: Tensor) -> Tensor  [batch, latent_dim]
    """
    model = model_data['model']
    train_args = model_data['train_args']
    is_mtl = model_data['is_mtl']
    fusion_type = getattr(train_args, 'fusion_type', 'sum')

    rf_x, _, cfo_x, _, channel_x, _, _, _ = batch

    task_to_input = {
        'rf_fingerprinting': rf_x.squeeze(1).to(device),
        'cfo_estimation':    cfo_x.squeeze(1).to(device),
        'channel_estimation': channel_x.squeeze(1).to(device),
    }

    if is_mtl:
        projections = model['projections']
        encoder = model['encoder']
        tasks = list(train_args.task)

        # Pre-compute fixed projections for the other tasks (no grad needed)
        fixed_projs = {}
        with torch.no_grad():
            for t in tasks:
                if t != task:
                    fixed_projs[t] = projections[t](task_to_input[t])

        def forward_fn(x_input):
            proj = projections[task](x_input)
            all_projs = [proj] + [fixed_projs[t] for t in tasks if t != task]

            if fusion_type == 'sum':
                fused = sum(all_projs)
            elif fusion_type == 'concat':
                fused = torch.cat(all_projs, dim=-1)
            elif fusion_type == 'depth_concat':
                fused = torch.cat(all_projs, dim=1)
            else:
                fused = sum(all_projs)

            features = encoder(fused)
            return features.reshape(x_input.size(0), -1)

    else:
        projection = model['projection']
        encoder = model['encoder']

        def forward_fn(x_input):
            proj = projection(x_input)
            features = encoder(proj)
            return features.reshape(x_input.size(0), -1)

    return forward_fn


def _iterate_hcr(forward_fn, x_input, features, sigma, num_pits, diffdiv):
    """
    One random-start HCR power iteration (adapted from testers.py iterate()).

    Finds the input perturbation ε that minimises the directional derivative
    norm via LSQR, maximising the HCR ratio  ||ε||² / (exp(||zε||²/σ²) - 1).

    Args:
        forward_fn : callable x -> flat_features
        x_input    : (batch, 2, seq_len) on device, used as the base point
        features   : (batch, latent_dim) pre-computed features at x_input
        sigma      : noise std per feature element
        num_pits   : number of power-iteration steps
        diffdiv    : divisor for initial direction (from original HCR code)

    Returns:
        in_diff  : numpy (batch, 2, seq_len) — input perturbation ε
        out_diff : numpy (batch, latent_dim) — resulting feature perturbation zε
    """
    batch_size = x_input.size(0)
    out_flat_size = features.numel() // batch_size

    features_np = features.cpu().detach().numpy()

    # Random perturbed features as starting direction
    outputsa = features_np + np.random.normal(scale=sigma, size=features_np.shape)

    diff = (outputsa - features_np).reshape(batch_size, out_flat_size)
    diff = diff / math.sqrt(out_flat_size) / diffdiv

    perturbed = x_input.clone()
    cdiff = diff.copy()

    outputpert = features  # fallback if loop body never executes

    for _ in range(num_pits):
        # Re-normalise cdiff to match the scale of diff
        row_norms = np.linalg.norm(cdiff, axis=1, keepdims=True)
        row_norms = np.where(row_norms < 1e-12, 1.0, row_norms)
        cdiff = cdiff / row_norms * np.linalg.norm(diff, axis=1, keepdims=True)

        atol = 2e-2 * float(np.min(np.linalg.norm(cdiff, axis=1) + 1e-12))
        cdiff_flat = cdiff.flatten().astype(np.float32)

        # Build the Jacobian as a sparse linear operator
        (_, vjpfunc) = torch.func.vjp(forward_fn, perturbed)

        def rmatvec(v, _vjpfunc=vjpfunc, _feat_shape=list(features.shape)):
            vt = torch.from_numpy(v.reshape(_feat_shape)).to(x_input.device).float()
            return _vjpfunc(vt)[0].cpu().detach().numpy().flatten().astype(np.float32)

        def matvec(v, _fn=forward_fn, _p=perturbed, _xshape=list(x_input.shape)):
            vt = torch.from_numpy(v.reshape(_xshape)).to(x_input.device).float()
            return torch.func.jvp(_fn, (_p,), (vt,))[1].cpu().detach().numpy().flatten().astype(np.float32)

        linear_op = scipy.sparse.linalg.LinearOperator(
            shape=(features.numel(), x_input.numel()),
            matvec=matvec,
            rmatvec=rmatvec,
            matmat=matvec,
            rmatmat=rmatvec,
            dtype=np.float32,
        )
        delta = scipy.sparse.linalg.lsqr(
            linear_op, cdiff_flat, atol=atol, damp=0
        )[0]

        perturbed = x_input + torch.from_numpy(
            delta.reshape(list(x_input.shape))
        ).float().to(x_input.device)

        with torch.no_grad():
            outputpert = forward_fn(perturbed)

        cdiff = (outputpert - features).cpu().detach().numpy().reshape(
            batch_size, out_flat_size
        )

    in_diff = (perturbed - x_input).cpu().detach().numpy()
    out_diff = (outputpert - features).cpu().detach().numpy()
    return in_diff, out_diff


def compute_hcr_bounds(forward_fn, x_input, sigma, device,
                       num_iter=25, num_pits=6, diffdiv=500):
    """
    Compute HCR lower bounds on MSE for a single input batch.

    Runs num_iter random restarts of the power iteration and returns the
    per-sample maximum bound:
        HCR_MSE ≥ ||ε||² / (exp(||zε||² / σ²) - 1)

    where ε is the optimised input perturbation and zε = f(x+ε) - f(x).

    Args:
        forward_fn : callable x -> flat_features
        x_input    : (batch, 2, seq_len) tensor (will be moved to device)
        sigma      : noise std per feature element (scalar)
        device     : torch.device
        num_iter   : number of random restarts
        num_pits   : LSQR power-iteration steps per restart
        diffdiv    : initial direction divisor (500 by default, as in HCR paper)

    Returns:
        numpy array of shape (batch_size,) with per-sample MSE lower bounds
    """
    x_input = x_input.to(device)

    with torch.no_grad():
        features = forward_fn(x_input)

    batch_size = x_input.size(0)
    best_bounds = np.zeros(batch_size, dtype=np.float64)

    for it in range(num_iter):
        try:
            in_diff, out_diff = _iterate_hcr(
                forward_fn, x_input, features, sigma, num_pits, diffdiv
            )
        except Exception as exc:
            print(f"    [HCR] iteration {it} failed: {exc}")
            continue

        in_diff_flat = in_diff.reshape(batch_size, -1)
        out_diff_flat = out_diff.reshape(batch_size, -1)

        numerator = np.sum(in_diff_flat ** 2, axis=1)  # ||ε||² per sample

        out_norms_sq = np.sum(out_diff_flat ** 2, axis=1)
        # Clip exponent to avoid overflow (exp > e^700 ≈ inf)
        exp_arg = np.clip(out_norms_sq / (sigma ** 2 + 1e-30), 0.0, 700.0)
        denominator = np.exp(exp_arg) - 1.0
        denominator = np.where(denominator < 1e-12, 1e-12, denominator)

        bounds = numerator / denominator
        best_bounds = np.maximum(best_bounds, bounds)

    return best_bounds


def compute_hcr_for_noise_levels(model_data, data_loader, noise_levels, latent_dim, device,
                                  tasks=None, num_iter=25, num_pits=6, diffdiv=500,
                                  max_batches=None):
    """
    Compute HCR bounds for every (noise_level, task) combination.

    Args:
        model_data   : dict from build_encoder_pipeline / load_model_from_experiment
        data_loader  : DataLoader yielding ActivationDataset batches
        noise_levels : list/array of noise level values to sweep
        latent_dim   : total latent dimension (used for sigma = sqrt(NL/D))
        device       : torch.device
        tasks        : list of task names (default: all tasks in the model)
        num_iter     : HCR random restarts per batch
        num_pits     : power-iteration steps per restart
        diffdiv      : initial direction normalisation divisor
        max_batches  : cap on number of batches per (noise_level, task) (None = all)

    Returns:
        dict  { noise_level (float) : { task_name (str) : mean_mse_lower_bound (float|None) } }
    """
    model = model_data['model']
    train_args = model_data['train_args']
    is_mtl = model_data['is_mtl']

    if tasks is None:
        tasks = list(train_args.task) if is_mtl else [train_args.task]

    model.eval()
    results = {}

    for noise_level in noise_levels:
        # Map noise_level to per-feature-dim sigma
        if noise_level > 0:
            sigma = math.sqrt(float(noise_level) / latent_dim)
        else:
            # Zero noise: use a tiny sigma for numerical approximation
            sigma = 1e-6

        print(f"\n[HCR] noise_level={noise_level}, sigma={sigma:.6f}")
        task_bounds = {t: [] for t in tasks}

        for batch_idx, batch in enumerate(data_loader):
            if max_batches is not None and batch_idx >= max_batches:
                break

            rf_x, _, cfo_x, _, channel_x, _, _, _ = batch
            task_to_input = {
                'rf_fingerprinting': rf_x.squeeze(1).to(device),
                'cfo_estimation':    cfo_x.squeeze(1).to(device),
                'channel_estimation': channel_x.squeeze(1).to(device),
            }

            for task in tasks:
                if task not in task_to_input:
                    continue

                x_input = task_to_input[task]
                forward_fn = _make_task_forward_fn(model_data, task, batch, device)

                try:
                    bounds = compute_hcr_bounds(
                        forward_fn, x_input, sigma, device,
                        num_iter=num_iter, num_pits=num_pits, diffdiv=diffdiv,
                    )
                    task_bounds[task].extend(bounds.tolist())
                except Exception as exc:
                    print(f"  [HCR] task={task} batch={batch_idx} error: {exc}")

            print(f"  [HCR] processed batch {batch_idx + 1}")

        results[float(noise_level)] = {}
        for task in tasks:
            if task_bounds[task]:
                results[float(noise_level)][task] = float(np.mean(task_bounds[task]))
            else:
                results[float(noise_level)][task] = None
            print(f"  [HCR] {task}: {results[float(noise_level)][task]}")

    return results
