"""
FIL Baseline Evaluation: Privacy-Utility Tradeoff.

For the joint RF+CFO+Channel encoder (FIL-compatible), sweeps over noise levels
β and measures:
  - Utility  : RF accuracy, CFO MSE/R², Channel MSE/R²
  - Privacy  : reconstruction MSE via the unbiased attack (Attack-ub)
  - Bound    : Cramér-Rao lower bound  d / Tr(I_e(x))  = β / mean_Tr

The β parameter is shared with the main codebase (inject_isotropic_noise) so
results are directly comparable.

Relationship β ↔ 1/dFIL
------------------------
  noise std per dim  σ = sqrt(β / d)        (on unit-norm activation)
  dFIL(x) = Tr(J^T J) / (d · σ²) = Tr(J^T J) / β   (per-sample)
  mean dFIL = mean_Tr / β
  → 1/dFIL = β / mean_Tr

The Cramér-Rao MSE lower bound (for unbiased attacks):
  MSE_lb = d / Tr(I_e(x)) = d·σ² / Tr(J^T J) = β / mean_Tr × d/d = β/mean_Tr
  (per-dim: MSE_lb = 1/dFIL)

Usage:
    python baselines/FIL_instance_encoding/fil_eval.py \
        --model_path baselines/FIL_instance_encoding/models/fil_joint \
        --pkl_dataset_path dataset/rf_partition_dict_0.5.pkl \
        --betas 0,5,10,20,50 \
        --output_path baselines/FIL_instance_encoding/results/fil_comparison.json \
        --gpu_id 0

Outputs a JSON file with structure:
{
  "betas": [0, 5, 10, 20, 50],
  "per_beta": {
    "0": {
      "utility": {"rf_acc": 0.87, "cfo_mse": 0.01, "cfo_r2": 0.99,
                  "channel_mse": 0.02, "channel_r2": 0.98},
      "privacy": {"rf_mse": ..., "cfo_mse": ..., "channel_mse": ...},
      "cr_bound": {"1_over_dfil": ..., "mse_lower_bound": ...}
    },
    ...
  },
  "metadata": {...}
}
"""

import argparse
import json
import os
import pickle
import random
import sys
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import r2_score, mean_squared_error
from torch.utils.data import DataLoader
from tqdm import tqdm

# ── Path setup ────────────────────────────────────────────────────────────────
_BASELINE_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_BASELINE_DIR, '..', '..'))
_CODE = os.path.join(_ROOT, 'code')
for p in [_CODE, os.path.join(_CODE, 'rep_lr')]:
    if p not in sys.path:
        sys.path.insert(0, p)

from rep_lr.py_datasets import TrainDataset
from fil_encoder import FILJointEncoder, compute_mean_trace
from fil_attack import run_unbiased_attack, _inject_noise


# ──────────────────────────────────────────────────────────────────────────────

def get_args():
    p = argparse.ArgumentParser(
        description='Evaluate FIL baseline for privacy-utility tradeoff',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument('--model_path', required=True,
                   help='Directory with fil_args.json and fil_best.pt')
    p.add_argument('--pkl_dataset_path', required=True)
    p.add_argument('--betas', default='',
                   help='Comma-separated noise levels (total variance on unit-norm activation). '
                        'Leave empty if using --target_mses.')
    p.add_argument('--target_mses', default='',
                   help='Comma-separated target reconstruction MSEs. If specified, β is derived '
                        'from MSE via: β = target_MSE × mean_Tr. Overrides --betas.')
    p.add_argument('--output_path',
                   default='baselines/FIL_instance_encoding/results/fil_comparison.json')
    p.add_argument('--gpu_id',      type=int, default=0)
    p.add_argument('--batch_size',  type=int, default=64)
    p.add_argument('--num_workers', type=int, default=4)
    p.add_argument('--max_attack_samples', type=int, default=200,
                   help='Max test samples to run the unbiased attack on (per task).')
    p.add_argument('--attack_niters', type=int, default=2000,
                   help='Max optimisation steps for each Attack-ub sample.')
    p.add_argument('--attack_lr',    type=float, default=1e-2)
    p.add_argument('--trace_batches', type=int, default=20,
                   help='Number of batches to estimate mean Jacobian trace.')
    p.add_argument('--trace_subsample', type=int, default=20,
                   help='Input dims to probe per sample for trace estimation.')
    p.add_argument('--skip_attack', action='store_true',
                   help='Skip reconstruction attack (faster, no privacy result).')
    p.add_argument('--seed', type=int, default=42)
    return p.parse_args()


# ──────────────────────────────────────────────────────────────────────────────
# Model loading
# ──────────────────────────────────────────────────────────────────────────────

def load_fil_model(model_path: str, device: torch.device) -> FILJointEncoder:
    args_file = os.path.join(model_path, 'fil_args.json')
    ckpt_file = os.path.join(model_path, 'fil_best.pt')

    if not os.path.exists(args_file):
        raise FileNotFoundError(f'fil_args.json not found in {model_path}')
    if not os.path.exists(ckpt_file):
        # Fall back to final checkpoint
        ckpt_file = os.path.join(model_path, 'fil_final.pt')
        if not os.path.exists(ckpt_file):
            raise FileNotFoundError(f'No checkpoint found in {model_path}')

    with open(args_file) as f:
        args = json.load(f)

    model = FILJointEncoder(
        slice_len=args.get('slice_len', 1024),
        proj_seq_len=args.get('proj_seq_len', 256),
        proj_hidden_dim=args.get('proj_hidden_dim', 512),
        d2=args.get('d2', 128),
        num_classes=16,
        dropout=args.get('dropout', 0.1),
        head_hidden_dim=args.get('head_hidden_dim', 256),
        encoder_num_blocks=args.get('encoder_num_blocks', 3),
        cfo_head_type=args.get('cfo_head_type', 'simple'),
    )
    state = torch.load(ckpt_file, map_location=device)
    model.load_state_dict(state)
    model = model.to(device)
    model.eval()
    print(f'Loaded FIL model from {ckpt_file}')
    return model, args


# ──────────────────────────────────────────────────────────────────────────────
# Utility evaluation
# ──────────────────────────────────────────────────────────────────────────────

def evaluate_utility(
    model: FILJointEncoder,
    data_loader,
    beta: float,
    device: torch.device,
) -> dict:
    """
    Evaluate task utility at noise level beta.
    - RF  : top-1 accuracy
    - CFO : MSE, R²
    - Channel: MSE, R²
    """
    ce_loss  = nn.CrossEntropyLoss()
    mse_loss = nn.MSELoss()

    rf_correct = 0
    rf_total   = 0
    cfo_preds, cfo_true = [], []
    ch_preds, ch_true   = [], []

    model.eval()
    with torch.no_grad():
        for batch in tqdm(data_loader, desc=f'Utility beta={beta}', leave=False):
            rf_x, rf_y, cfo_x, cfo_y, ch_x, ch_y, _ = batch

            rf_x  = rf_x.to(device).float()
            cfo_x = cfo_x.to(device).float()
            ch_x  = ch_x.to(device).float()
            rf_y  = rf_y.to(device).long()
            cfo_y = cfo_y.to(device).float().view(-1, 1)
            ch_y  = ch_y.to(device).float()

            # Encode + inject noise (same mechanism as main codebase)
            z = model.forward_first(rf_x, cfo_x, ch_x)
            z_noisy = _inject_noise(z, beta)

            # Get σ for forward_second (noise already applied; pass sigma=0)
            preds = model.forward_second(z_noisy, sigma=None)

            # RF accuracy
            rf_pred = preds['rf'].argmax(1)
            rf_correct += (rf_pred == rf_y).sum().item()
            rf_total   += rf_y.size(0)

            # CFO metrics
            cfo_preds.append(preds['cfo'].squeeze(-1).cpu().numpy())
            cfo_true.append(cfo_y.squeeze(-1).cpu().numpy())

            # Channel metrics
            ch_p = preds['channel'].view(ch_y.size(0), -1).cpu().numpy()
            ch_t = ch_y.view(ch_y.size(0), -1).cpu().numpy()
            ch_preds.append(ch_p)
            ch_true.append(ch_t)

    cfo_preds = np.concatenate(cfo_preds)
    cfo_true  = np.concatenate(cfo_true)
    ch_preds  = np.concatenate(ch_preds, axis=0)
    ch_true   = np.concatenate(ch_true,  axis=0)

    rf_acc     = rf_correct / rf_total
    cfo_mse    = float(mean_squared_error(cfo_true, cfo_preds))
    cfo_r2     = float(r2_score(cfo_true, cfo_preds))
    ch_mse     = float(mean_squared_error(ch_true.ravel(), ch_preds.ravel()))
    ch_r2      = float(r2_score(ch_true.ravel(), ch_preds.ravel()))

    return {
        'rf_acc':       rf_acc,
        'cfo_mse':      cfo_mse,
        'cfo_r2':       cfo_r2,
        'channel_mse':  ch_mse,
        'channel_r2':   ch_r2,
    }


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    args = get_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    os.makedirs(os.path.dirname(os.path.abspath(args.output_path)), exist_ok=True)

    device = torch.device(f'cuda:{args.gpu_id}' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')

    # ── Load model ─────────────────────────────────────────────────────────────
    model, model_args = load_fil_model(args.model_path, device)
    latent_dim = model.latent_dim
    slice_len  = model_args.get('slice_len', 1024)
    d_rf = 2 * slice_len   # input dimension for RF

    # ── Load dataset ───────────────────────────────────────────────────────────
    with open(args.pkl_dataset_path, 'rb') as f:
        content = pickle.load(f)

    test_list = content['test']
    max_cfo   = content['max_cfo']
    mean_cfo  = content['mean_cfo']
    std_cfo   = content['std_cfo']

    ID_class_dict = {f'Radio{i}': i for i in range(16)}
    import argparse as _ap
    ds_args = _ap.Namespace(slice_len=slice_len)

    test_ds = TrainDataset(
        test_list, ID_class_dict, ds_args, max_cfo, mean_cfo, std_cfo,
        test_mode=False,   # single-slice mode for clean per-sample evaluation
    )
    test_dl = DataLoader(
        test_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True,
    )

    # ── Jacobian trace (computed once, independent of beta) ────────────────────
    # We use train data to be consistent with FIL paper (they use full test set,
    # but train split is more conservative / avoids data leakage).
    train_list = content['train']
    train_ds = TrainDataset(
        train_list, ID_class_dict, ds_args, max_cfo, mean_cfo, std_cfo,
        test_mode=False,
    )
    trace_dl = DataLoader(
        train_ds, batch_size=32, shuffle=True,
        num_workers=args.num_workers, pin_memory=True,
    )

    print('Computing mean Jacobian trace (Tr(J^T J)) ...')
    mean_tr, d_input = compute_mean_trace(
        model, trace_dl, device,
        jvp_parallelism=10,
        subsample=args.trace_subsample,
        num_batches=args.trace_batches,
    )
    print(f'  mean_Tr = {mean_tr:.4f},  d_input (RF) = {d_input}')

    # ── Determine betas (either from --betas or derived from --target_mses) ────
    if args.target_mses:
        # User provided target MSEs; derive equivalent β values
        target_mses = [float(m) for m in args.target_mses.split(',')]
        betas = [mse * mean_tr for mse in target_mses]
        print(f'\nDerived betas from target MSEs:')
        for mse, beta in zip(target_mses, betas):
            print(f'  Target MSE={mse:.6f} → β={beta:.4f}')
    elif args.betas:
        betas = [float(b) for b in args.betas.split(',')]
        print(f'\nUsing specified betas: {betas}')
    else:
        # Default
        betas = [0, 5, 10, 20, 50]
        print(f'\nUsing default betas: {betas}')

    # ── Per-beta evaluation ────────────────────────────────────────────────────
    per_beta = {}

    for beta in betas:
        print(f'\n{"="*60}\nbeta = {beta}\n{"="*60}')

        # --- dFIL and Cramér-Rao bound ----------------------------------------
        # σ²_per_dim = beta / latent_dim (on unit-norm activation)
        # dFIL = Tr(J^T J) / beta   (where J is w.r.t. RF input)
        # 1/dFIL = beta / mean_Tr
        # MSE lower bound (per element) = 1/dFIL
        one_over_dfil = beta / mean_tr if mean_tr > 0 else float('inf')
        cr_bound_mse  = one_over_dfil   # per-element MSE lower bound

        print(f'  1/dFIL = {one_over_dfil:.4f},  CR bound MSE = {cr_bound_mse:.6f}')

        # --- Utility -----------------------------------------------------------
        utility = evaluate_utility(model, test_dl, beta, device)
        print(f'  Utility: RF acc={utility["rf_acc"]:.3f}, '
              f'CFO MSE={utility["cfo_mse"]:.4f}, '
              f'Channel MSE={utility["channel_mse"]:.4f}')

        # --- Privacy (unbiased attack) -----------------------------------------
        privacy = {}
        if not args.skip_attack:
            for task in ['rf', 'cfo', 'channel']:
                res = run_unbiased_attack(
                    model, test_dl, beta, task, device,
                    lr=args.attack_lr,
                    niters=args.attack_niters,
                    max_samples=args.max_attack_samples,
                )
                privacy[f'{task}_recon_mse']  = res['mse']
                privacy[f'{task}_recon_nmse'] = res['nmse']
                print(f'  Attack-ub [{task}]: MSE={res["mse"]:.6f}, '
                      f'NMSE={res["nmse"]:.4f}  (n={res["n_samples"]})')
        else:
            print('  [Skipped attack]')

        per_beta[str(int(beta) if beta == int(beta) else beta)] = {
            'cr_bound': {
                '1_over_dfil':   one_over_dfil,
                'mse_lower_bound': cr_bound_mse,
                'mean_tr':       mean_tr,
            },
            'utility': utility,
            'privacy': privacy,
        }

    # ── Assemble and save output ───────────────────────────────────────────────
    output = {
        'betas':    betas,
        'per_beta': per_beta,
        'metadata': {
            'model_path':  args.model_path,
            'latent_dim':  latent_dim,
            'mean_tr':     mean_tr,
            'd_input_rf':  d_input,
            'timestamp':   datetime.now().isoformat(),
            'input_mode':  'target_mses' if args.target_mses else 'betas',
            'target_mses': args.target_mses if args.target_mses else None,
            'note': (
                'beta is total noise variance on unit-norm activation (same as '
                'inject_isotropic_noise in main codebase). '
                '1/dFIL = beta / mean_Tr(J^T J). '
                'CR bound is the Cramér-Rao lower bound on per-element MSE for '
                'any unbiased attack. '
                'If input_mode=target_mses, betas were derived via: β = target_MSE × mean_Tr.'
            ),
        },
    }

    with open(args.output_path, 'w') as f:
        json.dump(output, f, indent=2)

    print(f'\nResults saved to: {args.output_path}')


if __name__ == '__main__':
    main()
