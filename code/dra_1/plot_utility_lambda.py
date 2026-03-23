#!/usr/bin/env python3
"""
Evaluate model utility for a single noise configuration.

This script is a simplified version of plot_utility.py, designed to be called
for a single combination of noise type, noise level, and (for non-isotropic noise)
a lambda factor. It calculates the utility metrics for all available tasks
and saves the results to a JSON file for each task.

Usage:
    python plot_utility_lambda.py --experiment_path /path/to/exp --noise_type nonisotropic --noise_level 10 --lambda_factor 0.001
"""

import os
import sys
import json
import argparse
import torch
import numpy as np
import pickle
import random
from tqdm import tqdm
import torch.nn as nn
from torch.utils.data import DataLoader

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from dra_1.py_datasets import ActivationDataset
from dra_1.inject_noise import inject_isotropic_noise, inject_nonisotropic_noise
from dra_1.model_loader import load_model_for_utility_check
from sklearn.metrics import mean_squared_error, r2_score


def get_empirical_fim(head, data_loader, device, latent_dim):
    """Calculates the empirical Fisher Information Matrix for a given task head."""
    fim = torch.zeros((latent_dim, latent_dim), device=device)
    num_samples = 0
    head.to(device).eval()

    from rep_lr.models import RFClassificationHead
    is_classification = isinstance(head, RFClassificationHead)
    criterion = nn.CrossEntropyLoss() if is_classification else nn.MSELoss()

    if not is_classification:
        mse_loss = nn.MSELoss()
        def complex_mse_loss(pred, target):
            pred_flat = pred.view(pred.size(0), -1)
            target_flat = target.view(target.size(0), -1)
            return mse_loss(pred_flat, target_flat)
        criterion = complex_mse_loss

    for batch in tqdm(data_loader, desc="Calculating FIM", leave=False, ncols=80):
        _, rf_labels, _, cfo_labels, _, ch_labels, activations, _ = batch
        
        activations_squeezed = activations.squeeze(1).to(device)
        original_shape = activations_squeezed.shape
        activations_flat = activations_squeezed.view(activations_squeezed.size(0), -1)
        activations_flat.requires_grad = True

        head.zero_grad()
        
        y_target = None
        if is_classification:
            y_target = rf_labels.to(device).view(-1)
        else:
            from rep_lr.models import CFOEstimationHead, ChannelEstimationHead
            if isinstance(head, CFOEstimationHead):
                y_target = cfo_labels.to(device).float().view(-1)
            elif isinstance(head, ChannelEstimationHead):
                y_target = ch_labels.to(device)
                if y_target.dim() == 4: y_target = y_target.squeeze(1)

        if y_target is None: continue

        y_hat = head(activations_flat.view(original_shape))
        if y_hat.dim() > 1 and y_target.dim() == 1 and not is_classification:
             y_hat = y_hat.squeeze(-1)

        loss = criterion(y_hat, y_target)
        loss.backward()
        
        if activations_flat.grad is not None:
            J = activations_flat.grad
            fim += J.T @ J
            num_samples += activations_flat.size(0)

    return fim / num_samples if num_samples > 0 else fim


def evaluate_utility(head, test_dl, device, noise_type, noise_level, L, V, lambda_factor, clip_value, task_name):
    """Unified utility evaluation function."""
    head.to(device).eval()
    y_true, y_pred = [], []

    is_classification = task_name == 'rf_fingerprinting'
    
    with torch.no_grad():
        for batch in test_dl:
            _, rf_labels, _, cfo_labels, _, ch_labels, activations, _ = batch

            activations = activations.squeeze(1).to(device)
            activations_flat = activations.view(activations.size(0), -1)

            if noise_type == 'isotropic':
                noisy_activations = inject_isotropic_noise(
                    activations_flat, noise_level, clip_value=clip_value
                )
            elif noise_type == 'nonisotropic':
                noisy_activations = inject_nonisotropic_noise(
                    activations_flat,
                    noise_level,
                    L,
                    V,
                    lambda_factor=lambda_factor,
                    clip_value=clip_value,
                )
            else: # 'none'
                if clip_value is not None:
                    # Treat as isotropic with zero noise to apply clipping behavior
                    noisy_activations = inject_isotropic_noise(
                        activations_flat, 0.0, clip_value=clip_value
                    )
                else:
                    noisy_activations = activations_flat

            # Reshape for head if needed
            if task_name in ['rf_fingerprinting', 'cfo_estimation']:
                noisy_activations = noisy_activations.view(noisy_activations.size(0), 2, -1)

            outputs = head(noisy_activations)
            
            # Process outputs and labels based on task
            if is_classification:
                labels = rf_labels.to(device)
                _, predicted = torch.max(outputs, 1)
                y_true.extend(labels.cpu().numpy().flatten())
                y_pred.extend(predicted.cpu().numpy())
            else: # Regression
                if task_name == 'cfo_estimation':
                    labels = cfo_labels.to(device).float()
                    outputs = outputs.squeeze(-1)
                elif task_name == 'channel_estimation':
                    labels = ch_labels.to(device)
                    if labels.dim() == 4: labels = labels.squeeze(1)
                
                y_true.extend(labels.cpu().numpy())
                y_pred.extend(outputs.cpu().numpy())

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # Calculate metrics
    if is_classification:
        return {'accuracy': np.mean(y_true == y_pred)}
    
    if task_name == 'channel_estimation':
        # Handle complex channel data
        r2 = r2_score(y_true.reshape(len(y_true), -1), y_pred.reshape(len(y_pred), -1))
        mse = mean_squared_error(y_true.reshape(len(y_true), -1), y_pred.reshape(len(y_pred), -1))
    else: # CFO
        r2 = r2_score(y_true, y_pred)
        mse = mean_squared_error(y_true, y_pred)
        
    return {'mse': mse, 'r2': r2}


def main():
    parser = argparse.ArgumentParser(description='Evaluate utility for a single noise configuration.')
    parser.add_argument('--experiment_path', type=str, required=True, help='Path to experiment results dir.')
    parser.add_argument('--activation_dir', type=str, default=None, help='Path to pre-computed activations.')
    parser.add_argument('--output_dir', type=str, required=True, help='Directory to save the output JSON file.')
    parser.add_argument('--noise_type', type=str, required=True, choices=['none', 'isotropic', 'nonisotropic'], help='Type of noise to inject.')
    parser.add_argument('--noise_level', type=float, required=True, help='Noise level (variance).')
    parser.add_argument('--lambda_factor', type=float, default=None, help='Lambda factor for non-isotropic noise.')
    parser.add_argument('--clip_value', type=float, default=None, help='Optional norm clip value for activations.')
    parser.add_argument('--fim_samples', type=int, default=8000, help='Number of samples for FIM calculation.')
    parser.add_argument('--gpu_id', default=0, type=int, help='GPU ID to use.')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size for evaluation.')
    cli_args = parser.parse_args()

    os.makedirs(cli_args.output_dir, exist_ok=True)
    device = torch.device(f'cuda:{cli_args.gpu_id}' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    print("Loading model...")
    model_data = load_model_for_utility_check(cli_args.experiment_path, device)
    model = model_data['model']
    train_args = model_data['train_args']
    heads = model['heads'] if 'heads' in model else {train_args.task: model['head']}
    print(f"Tasks to evaluate: {list(heads.keys())}")

    activation_dir = cli_args.activation_dir or os.path.join(cli_args.experiment_path, 'activations')
    with open(model_data['train_args'].pkl_dataset_path, 'rb') as f:
        partitions = pickle.load(f)
    
    test_files = partitions['test']
    random.shuffle(test_files)
    test_dataset = ActivationDataset(activation_dir=activation_dir, file_list=test_files, test_mode=True)
    test_dl = DataLoader(test_dataset, batch_size=cli_args.batch_size, shuffle=False)

    for task_name, head in heads.items():
        print(f"\n--- Evaluating task: {task_name} ---")
        L, V = None, None
        
        if cli_args.noise_type == 'nonisotropic':
            print("Calculating FIM for non-isotropic noise...")
            train_files = partitions['train']
            random.shuffle(train_files)
            fim_files = train_files[:cli_args.fim_samples]
            fim_dataset = ActivationDataset(activation_dir=activation_dir, file_list=fim_files, test_mode=False)
            fim_dl = DataLoader(fim_dataset, batch_size=cli_args.batch_size, shuffle=True)
            
            latent_dim = head.input_dim
            fim = get_empirical_fim(head, fim_dl, device, latent_dim)
            
            trace_fim = torch.trace(fim)
            if trace_fim > 1e-10:
                fim = fim / trace_fim
            
            L_e, V = torch.linalg.eigh(fim)
            L = torch.relu(L_e)
            print("FIM calculation complete.")

        metrics = evaluate_utility(
            head,
            test_dl,
            device,
            cli_args.noise_type,
            cli_args.noise_level,
            L,
            V,
            cli_args.lambda_factor,
            cli_args.clip_value,
            task_name,
        )
        
        # Save results to a JSON file for this task
        task_output_dir = os.path.join(cli_args.output_dir, task_name)
        os.makedirs(task_output_dir, exist_ok=True)
        results_path = os.path.join(task_output_dir, 'results.json')
        
        # Convert tensors to lists for JSON serialization
        for key, value in metrics.items():
            if isinstance(value, np.generic):
                metrics[key] = value.item()

        with open(results_path, 'w') as f:
            json.dump(metrics, f, indent=4)
            
        print(f"Results for {task_name} saved to {results_path}")
        print(f"Metrics: {metrics}")

    print("\nUtility analysis complete for this configuration.")

if __name__ == '__main__':
    main()