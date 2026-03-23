#!/usr/bin/env python3
"""
Comparison script: Our method (isotropic / anisotropic FIM noise) vs HCR bounds.

Evaluates privacy-utility tradeoff for the joint MTL encoder (RF + CFO + Channel)
across a sweep of noise levels and writes a unified JSON for later plotting.

Usage example:
    python compare_methods.py \
        --experiment_path results_20250720_172807/rf_cfo_channel/... \
        --noise_levels 0,5,10,15,20 \
        --output_path comparison_results.json \
        --gpu_id 0

    python compare_methods.py \
    --experiment_path /path/to/rf_cfo_channel/experiment \
    --noise_levels 0,5,10,15,20 \
    --output_path comparison_results.json \
    --gpu_id 0 \
    --leaked_fraction 0.1 \
    --decoder_epochs 30 \
    --hcr_samples 100

    To skip expensive parts during development: --skip_attack, --skip_hcr, or --skip_fim
"""

import argparse
import datetime
import json
import os
import pickle
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from dotenv import load_dotenv

load_dotenv()

# ── Project path setup ─────────────────────────────────────────────────────────
_ROOT = os.path.dirname(os.path.abspath(__file__))
_CODE = os.path.join(_ROOT, 'code')
if _CODE not in sys.path:
    sys.path.insert(0, _CODE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from dra_1.model_loader import load_model_for_utility_check, load_training_args
from dra_1.py_datasets import ActivationDataset
from dra_1.robust_attack import run_reconstruction_attack
from dra_1.plot_utility import get_empirical_fim, evaluate_utility_for_task
from baselines.hcrbounds.hcr_bounds import build_encoder_pipeline, compute_hcr_for_noise_levels

# ── Constants ──────────────────────────────────────────────────────────────────
TASK_SHORT_TO_FULL = {
    'rf':      'rf_fingerprinting',
    'cfo':     'cfo_estimation',
    'channel': 'channel_estimation',
}
TASK_FULL_TO_SHORT = {v: k for k, v in TASK_SHORT_TO_FULL.items()}


# ── FIM utilities ──────────────────────────────────────────────────────────────

def compute_joint_fim(model_data, fim_loader, device, latent_dim, lambda_factor=1e-5):
    """
    Compute the joint FIM by summing task-specific FIMs and return (L, V).

    For MTL models the FIM is summed across all task heads, then normalised
    by its trace, and eigendecomposed.  Returns None, None on failure.
    """
    model = model_data['model']
    train_args = model_data['train_args']
    is_mtl = model_data['is_mtl']

    fim = torch.zeros((latent_dim, latent_dim), device=device)

    if is_mtl:
        print("Computing joint FIM (sum over all MTL tasks)...")
        for task_name in train_args.task:
            print(f"  - FIM for {task_name}")
            head = model_data['model']['heads'][task_name]
            fim += get_empirical_fim(head, fim_loader, device, latent_dim)
    else:
        task_name = train_args.task
        print(f"Computing FIM for single task: {task_name}")
        head = model_data['model']['head']
        fim = get_empirical_fim(head, fim_loader, device, latent_dim)

    trace_fim = torch.trace(fim)
    if trace_fim > 1e-10:
        fim = fim / trace_fim
        print(f"  FIM normalised by trace: {trace_fim:.3e}")
    else:
        print("  Warning: FIM trace near zero; skipping normalisation.")

    L_e, V = torch.linalg.eigh(fim)
    L = torch.relu(L_e)
    print("  Eigendecomposition complete.")
    return L, V


# ── Utility evaluation ─────────────────────────────────────────────────────────

def run_utility_evaluation(model_data, test_dl, noise_levels, device,
                            L=None, V=None, lambda_factor=1e-5):
    """
    Evaluate utility for both isotropic and anisotropic noise.

    Returns:
        dict keyed by noise_type ('isotropic', 'anisotropic_fim'), then
        metric name ('rf_accuracy', 'cfo_mse', 'cfo_r2', 'channel_mse', 'channel_r2'),
        each holding a list with one value per noise level.
    """
    model = model_data['model']
    train_args = model_data['train_args']
    is_mtl = model_data['is_mtl']

    heads = model['heads'] if is_mtl else {train_args.task: model['head']}

    noise_types = ['isotropic']
    if L is not None and V is not None:
        noise_types.append('anisotropic_fim')

    results = {nt: {} for nt in noise_types}
    # Initialise metric lists
    for nt in noise_types:
        results[nt] = {
            'rf_accuracy':   [],
            'cfo_mse':       [],
            'cfo_r2':        [],
            'channel_mse':   [],
            'channel_r2':    [],
        }

    for noise_level in noise_levels:
        print(f"\n[Utility] noise_level={noise_level}")
        for nt in noise_types:
            nl = noise_level
            l_arg = L if nt == 'anisotropic_fim' else None
            v_arg = V if nt == 'anisotropic_fim' else None
            nt_param = 'nonisotropic' if nt == 'anisotropic_fim' else 'isotropic'

            for full_task, head in heads.items():
                metrics = evaluate_utility_for_task(
                    full_task, head, test_dl, device, nl,
                    L=l_arg, V=v_arg, noise_type=nt_param,
                )
                if full_task == 'rf_fingerprinting':
                    results[nt]['rf_accuracy'].append(float(metrics['accuracy']))
                elif full_task == 'cfo_estimation':
                    results[nt]['cfo_mse'].append(float(metrics['mse']))
                    results[nt]['cfo_r2'].append(float(metrics['r2']))
                elif full_task == 'channel_estimation':
                    results[nt]['channel_mse'].append(float(metrics['mse']))
                    results[nt]['channel_r2'].append(float(metrics['r2']))

    return results


# ── Privacy evaluation ─────────────────────────────────────────────────────────

def run_privacy_evaluation(
    activation_dir, train_files, val_files, test_files,
    noise_levels, latent_dim, device, args,
    L=None, V=None,
):
    """
    Run the decoder reconstruction attack for each (noise_level, task, noise_type).

    Returns:
        dict keyed by noise_type, then 'rf_recon_mse' / 'cfo_recon_mse' /
        'channel_recon_mse', each a list over noise_levels.
    """
    noise_types = ['isotropic']
    if L is not None and V is not None:
        noise_types.append('anisotropic_fim')

    results = {nt: {'rf_recon_mse': [], 'cfo_recon_mse': [], 'channel_recon_mse': []}
               for nt in noise_types}

    short_tasks = ['rf', 'cfo', 'channel']
    task_result_keys = {
        'rf':      'rf_recon_mse',
        'cfo':     'cfo_recon_mse',
        'channel': 'channel_recon_mse',
    }

    for nt in noise_types:
        print(f"\n[Privacy] noise_type={nt}")
        # Collect per-task per-level MSEs then insert in order
        per_task = {t: [] for t in short_tasks}

        for noise_level in noise_levels:
            for task in short_tasks:
                attack_dir = os.path.join(
                    args.experiment_path, 'comparison_attacks',
                    nt, task,
                    f"noise_{str(noise_level).replace('.', '_')}",
                )

                # Reuse cached results if the attack has already been run
                cached_result_path = os.path.join(attack_dir, f"attack_{task}_results.json")
                if args.skip_attack and os.path.isfile(cached_result_path):
                    with open(cached_result_path) as f:
                        cached = json.load(f)
                    avg_mse = float(np.mean([r['MSE'] for r in cached]))
                    print(f"  [Cached] {nt}/{task}/noise={noise_level}: MSE={avg_mse:.6f}")
                    per_task[task].append(avg_mse)
                    continue

                print(f"  Running attack: {nt}/{task}/noise={noise_level}")
                l_arg = L if nt == 'anisotropic_fim' else None
                v_arg = V if nt == 'anisotropic_fim' else None

                try:
                    metrics = run_reconstruction_attack(
                        activation_dir=activation_dir,
                        train_files=train_files,
                        val_files=val_files,
                        test_files=test_files,
                        noise_type='nonisotropic' if nt == 'anisotropic_fim' else 'isotropic',
                        noise_level=float(noise_level),
                        task=task,
                        latent_dim=latent_dim,
                        device=device,
                        save_dir=attack_dir,
                        L=l_arg,
                        V=v_arg,
                        lambda_factor=args.lambda_factor,
                        clip_value=None,
                        epochs=args.decoder_epochs,
                        lr=1e-3,
                        batch_size=64,
                        patience=args.decoder_epochs,
                        leaked_fraction=args.leaked_fraction,
                        use_wandb=False,
                    )
                    per_task[task].append(metrics['mse'])
                except Exception as exc:
                    print(f"  [Warning] Attack failed ({exc}); inserting None.")
                    per_task[task].append(None)

        for task in short_tasks:
            results[nt][task_result_keys[task]] = per_task[task]

    return results


# ── Main ───────────────────────────────────────────────────────────────────────

def main(args):
    device = torch.device(f'cuda:{args.gpu_id}' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    noise_levels = [float(x.strip()) for x in args.noise_levels.split(',')]
    print(f"Noise levels: {noise_levels}")

    # ── Load model ────────────────────────────────────────────────────────────
    print("\nLoading model...")
    model_data = load_model_for_utility_check(args.experiment_path, device=device)
    train_args = model_data['train_args']
    is_mtl = model_data['is_mtl']
    latent_dim = int(getattr(train_args, 'd2', 256)) * 2  # activations are (batch, 2, d2)
    print(f"  latent_dim={latent_dim}  MTL={is_mtl}")

    # ── Activation directory ──────────────────────────────────────────────────
    activation_dir = os.path.join(args.experiment_path, 'activations')
    if not os.path.isdir(activation_dir):
        raise FileNotFoundError(f"Activations directory not found: {activation_dir}")

    # ── Partition file ────────────────────────────────────────────────────────
    pkl_file_path = os.environ.get('PKL_FILE_PATH')
    if not pkl_file_path:
        raise ValueError("PKL_FILE_PATH environment variable not set.")
    partition_file = os.path.join(pkl_file_path, 'rf_partition_dict_0.5.pkl')
    with open(partition_file, 'rb') as f:
        partitions = pickle.load(f)
    train_files = partitions['train']
    val_files   = partitions['val']
    test_files  = partitions['test']
    print(f"  Dataset splits: train={len(train_files)}, val={len(val_files)}, test={len(test_files)}")

    # ── Data loaders for utility evaluation ──────────────────────────────────
    test_dataset = ActivationDataset(
        activation_dir=activation_dir, file_list=test_files, test_mode=True
    )
    test_dl = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=4)

    # ── FIM computation (once, shared by anisotropic utility + attack) ────────
    L, V = None, None
    if not args.skip_fim:
        print("\nComputing joint FIM...")
        train_full_dataset = ActivationDataset(
            activation_dir=activation_dir, file_list=train_files
        )
        fim_indices = np.random.choice(
            len(train_full_dataset), size=min(args.fim_samples, len(train_full_dataset)),
            replace=False
        )
        fim_dataset = Subset(train_full_dataset, fim_indices)
        fim_loader  = DataLoader(fim_dataset, batch_size=64, shuffle=False, num_workers=4)
        L, V = compute_joint_fim(model_data, fim_loader, device, latent_dim, args.lambda_factor)
    else:
        print("Skipping FIM computation (--skip_fim set).")

    # ── Utility evaluation ────────────────────────────────────────────────────
    print("\n" + "="*60)
    print("UTILITY EVALUATION")
    print("="*60)
    utility_results = run_utility_evaluation(
        model_data, test_dl, noise_levels, device,
        L=L, V=V, lambda_factor=args.lambda_factor,
    )

    # ── Privacy evaluation (decoder attack) ───────────────────────────────────
    privacy_results = {}
    if not args.skip_attack:
        print("\n" + "="*60)
        print("PRIVACY EVALUATION (DECODER ATTACK)")
        print("="*60)
        privacy_results = run_privacy_evaluation(
            activation_dir=activation_dir,
            train_files=train_files,
            val_files=val_files,
            test_files=test_files,
            noise_levels=noise_levels,
            latent_dim=latent_dim,
            device=device,
            args=args,
            L=L,
            V=V,
        )
    else:
        print("\nSkipping decoder attack (--skip_attack set).")
        # Fill with None placeholders so JSON structure is consistent
        for nt in ['isotropic', 'anisotropic_fim']:
            privacy_results[nt] = {
                'rf_recon_mse':      [None] * len(noise_levels),
                'cfo_recon_mse':     [None] * len(noise_levels),
                'channel_recon_mse': [None] * len(noise_levels),
            }

    # ── HCR bounds ────────────────────────────────────────────────────────────
    hcr_results = {'mse_lower_bound': {'rf': [], 'cfo': [], 'channel': []}}
    if not args.skip_hcr:
        print("\n" + "="*60)
        print("HCR BOUNDS")
        print("="*60)
        encoder_data = build_encoder_pipeline(args.experiment_path, device)

        # Use a small subset of the test set for HCR (computationally expensive)
        hcr_test_files = test_files[:min(args.hcr_samples, len(test_files))]
        hcr_dataset = ActivationDataset(
            activation_dir=activation_dir, file_list=hcr_test_files, test_mode=True
        )
        hcr_loader = DataLoader(hcr_dataset, batch_size=args.hcr_batch_size,
                                shuffle=False, num_workers=2)

        tasks_for_hcr = list(train_args.task) if is_mtl else [train_args.task]

        hcr_raw = compute_hcr_for_noise_levels(
            model_data=encoder_data,
            data_loader=hcr_loader,
            noise_levels=noise_levels,
            latent_dim=latent_dim,
            device=device,
            tasks=tasks_for_hcr,
            num_iter=args.hcr_num_iter,
            num_pits=args.hcr_num_pits,
            diffdiv=args.hcr_diffdiv,
            max_batches=args.hcr_max_batches,
        )

        task_short_map = TASK_FULL_TO_SHORT  # full -> short
        for nl in noise_levels:
            for full_task, short in task_short_map.items():
                val = hcr_raw.get(float(nl), {}).get(full_task, None)
                hcr_results['mse_lower_bound'][short].append(val)
    else:
        print("\nSkipping HCR bounds (--skip_hcr set).")
        for short in ['rf', 'cfo', 'channel']:
            hcr_results['mse_lower_bound'][short] = [None] * len(noise_levels)

    # ── Assemble output JSON ──────────────────────────────────────────────────
    output = {
        'noise_levels': noise_levels,
        'methods': {
            'isotropic': {
                'privacy': privacy_results.get('isotropic', {}),
                'utility': {
                    k: utility_results['isotropic'][k]
                    for k in utility_results['isotropic']
                },
            },
            'anisotropic_fim': {
                'privacy': privacy_results.get('anisotropic_fim', {}),
                'utility': {
                    k: utility_results.get('anisotropic_fim', {}).get(k, [])
                    for k in ['rf_accuracy', 'cfo_mse', 'cfo_r2', 'channel_mse', 'channel_r2']
                },
            },
            'hcr_bound': hcr_results,
        },
        'metadata': {
            'experiment_path': args.experiment_path,
            'latent_dim':      latent_dim,
            'lambda_factor':   args.lambda_factor,
            'fim_samples':     args.fim_samples,
            'leaked_fraction': args.leaked_fraction,
            'decoder_epochs':  args.decoder_epochs,
            'timestamp':       datetime.datetime.now().isoformat(),
        },
    }

    os.makedirs(os.path.dirname(os.path.abspath(args.output_path)), exist_ok=True)
    with open(args.output_path, 'w') as f:
        json.dump(output, f, indent=4)
    print(f"\nResults saved to: {args.output_path}")

    # Quick sanity print
    print("\n=== Summary ===")
    print(f"Noise levels: {noise_levels}")
    iso_rf_acc = output['methods']['isotropic']['utility'].get('rf_accuracy', [])
    if iso_rf_acc:
        print(f"Isotropic RF accuracy:    {iso_rf_acc}")
    iso_rf_priv = output['methods']['isotropic']['privacy'].get('rf_recon_mse', [])
    if iso_rf_priv:
        print(f"Isotropic RF recon MSE:   {iso_rf_priv}")
    hcr_rf = output['methods']['hcr_bound']['mse_lower_bound'].get('rf', [])
    if hcr_rf:
        print(f"HCR RF MSE lower bound:   {hcr_rf}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Compare isotropic, anisotropic FIM noise, and HCR bounds.'
    )
    parser.add_argument('--experiment_path', type=str, required=True,
                        help='Path to trained MTL experiment directory.')
    parser.add_argument('--noise_levels', type=str, default='0,5,10,15,20',
                        help='Comma-separated noise levels to sweep.')
    parser.add_argument('--output_path', type=str, default='comparison_results.json',
                        help='Output JSON path.')
    parser.add_argument('--gpu_id', type=int, default=0,
                        help='GPU id to use.')
    parser.add_argument('--leaked_fraction', type=float, default=0.1,
                        help='Fraction of training data available to the adversary.')
    parser.add_argument('--decoder_epochs', type=int, default=30,
                        help='Max epochs for adversary decoder training.')
    parser.add_argument('--fim_samples', type=int, default=1000,
                        help='Number of samples used to compute the FIM.')
    parser.add_argument('--lambda_factor', type=float, default=1e-5,
                        help='FIM regularisation for anisotropic noise.')
    parser.add_argument('--skip_fim', action='store_true',
                        help='Skip FIM computation (also disables anisotropic results).')
    parser.add_argument('--skip_hcr', action='store_true',
                        help='Skip HCR bound computation.')
    parser.add_argument('--skip_attack', action='store_true',
                        help='Skip decoder attack training (uses cached results if present).')
    # HCR-specific tuning knobs
    parser.add_argument('--hcr_samples', type=int, default=100,
                        help='Number of test samples to use for HCR bound computation.')
    parser.add_argument('--hcr_batch_size', type=int, default=4,
                        help='Batch size for HCR computation (keep small due to LSQR cost).')
    parser.add_argument('--hcr_num_iter', type=int, default=25,
                        help='HCR random restarts per batch.')
    parser.add_argument('--hcr_num_pits', type=int, default=6,
                        help='Power-iteration steps per HCR restart.')
    parser.add_argument('--hcr_diffdiv', type=int, default=500,
                        help='Initial direction normalisation divisor for HCR.')
    parser.add_argument('--hcr_max_batches', type=int, default=None,
                        help='Cap on number of batches for HCR (None = all).')

    main(parser.parse_args())
