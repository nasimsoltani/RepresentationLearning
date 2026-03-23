#!/usr/bin/env python3
"""
Calculate and plot the eigenspectrum of the Fisher Information Matrix
for test set examples.

This script computes the empirical FIM for each example in the test set.
For Multi-Task Learning (MTL) models, it calculates the combined FIM:
F_bar = F_task1 + F_task2 + ...
Then it normalizes by the trace: F_norm = F_bar / Tr(F_bar)
Finally, it extracts the eigenvalues and plots the eigenspectrum.

Usage:
    python eigenspectrum.py --experiment_path /path/to/exp --output_path eigenspec.png
"""

import os
import sys
import argparse
import torch
import numpy as np
import pickle
import random
from tqdm import tqdm
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from dra_1.py_datasets import ActivationDataset
from dra_1.model_loader import load_model_for_utility_check


def get_empirical_fim_per_sample(head, activations, labels, device, latent_dim):
    """
    Calculates the empirical Fisher Information Matrix for a single sample and a specific head.
    
    Args:
        head: Task head model
        activations: Single sample activations (batch_size=1)
        labels: Labels for the sample
        device: Computing device
        latent_dim: Dimension of the latent space
    
    Returns:
        fim: Fisher Information Matrix for this sample (or None if invalid)
    """
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

    # Prepare activations
    activations_squeezed = activations.squeeze(1).to(device)
    original_shape = activations_squeezed.shape
    activations_flat = activations_squeezed.view(activations_squeezed.size(0), -1)
    
    # Detach and require grad for FIM calculation w.r.t activations
    activations_flat = activations_flat.detach().clone()
    activations_flat.requires_grad = True

    head.zero_grad()
    
    y_target = None
    if is_classification:
        y_target = labels.to(device).view(-1)
    else:
        from rep_lr.models import CFOEstimationHead, ChannelEstimationHead
        if isinstance(head, CFOEstimationHead):
            y_target = labels.to(device).float().view(-1)
        elif isinstance(head, ChannelEstimationHead):
            y_target = labels.to(device)
            if y_target.dim() == 4: 
                y_target = y_target.squeeze(1)

    if y_target is None:
        return None

    y_hat = head(activations_flat.view(original_shape))
    if y_hat.dim() > 1 and y_target.dim() == 1 and not is_classification:
        y_hat = y_hat.squeeze(-1)

    loss = criterion(y_hat, y_target)
    loss.backward()
    
    fim = None
    if activations_flat.grad is not None:
        J = activations_flat.grad
        fim = J.T @ J
    
    return fim


def collect_eigenvalues_from_test_set(heads, test_dl, device, latent_dim, num_samples=None):
    """
    Collect eigenvalues from test set examples using the combined FIM from all heads.
    F_bar = Sum(F_task)
    
    Args:
        heads: Dictionary of task heads {task_name: head}
        test_dl: Test data loader
        device: Computing device
        latent_dim: Dimension of the latent space
        num_samples: Maximum number of samples to process (None for all)
    
    Returns:
        all_eigenvalues: List of eigenvalue arrays, one per sample
    """
    all_eigenvalues = []
    sample_count = 0
    
    print(f"Calculating combined FIM for tasks: {list(heads.keys())}")
    
    # We don't need 'enable_grad' for the outer loop, but we need it inside get_empirical_fim_per_sample
    # where we explicitly set requires_grad=True on the input.
    
    for batch in tqdm(test_dl, desc="Processing test samples", ncols=80):
        _, rf_labels, _, cfo_labels, _, ch_labels, activations, _ = batch
        
        batch_size = activations.size(0)
        for i in range(batch_size):
            if num_samples is not None and sample_count >= num_samples:
                break
            
            # Extract single sample
            sample_activations = activations[i:i+1]
            
            fim_total = torch.zeros((latent_dim, latent_dim), device=device)
            valid_tasks_count = 0
            
            # Sum FIM across all available tasks
            for task_name, head in heads.items():
                # Get appropriate label for this sample and task
                if task_name == 'rf_fingerprinting':
                    sample_label = rf_labels[i:i+1]
                elif task_name == 'cfo_estimation':
                    sample_label = cfo_labels[i:i+1]
                elif task_name == 'channel_estimation':
                    sample_label = ch_labels[i:i+1]
                else:
                    continue
                
                # Calculate FIM for this task
                fim_task = get_empirical_fim_per_sample(
                    head, 
                    sample_activations, 
                    sample_label, 
                    device, 
                    latent_dim
                )
                
                if fim_task is not None:
                    fim_total += fim_task
                    valid_tasks_count += 1
            
            if valid_tasks_count > 0:
                # Normalize FIM by trace of the combined FIM
                trace_fim = torch.trace(fim_total)
                if trace_fim > 1e-10:
                    fim_total = fim_total / trace_fim
                
                # Compute eigenvalues
                eigenvalues, _ = torch.linalg.eigh(fim_total)
                eigenvalues = torch.relu(eigenvalues)  # Ensure non-negative
                
                # Convert to numpy and store
                all_eigenvalues.append(eigenvalues.cpu().numpy())
                sample_count += 1
        
        if num_samples is not None and sample_count >= num_samples:
            break
    
    return all_eigenvalues


def calculate_k_metric(eigenvalues, gamma=1.0):
    """
    Calculates the K metric for a set of eigenvalues.
    Formula: K = (1/d) * (lambda_max + gamma) * Sum(1 / (lambda_i + gamma))
    """
    d = len(eigenvalues)
    if d == 0:
        return 0.0
    lambda_max = np.max(eigenvalues)
    sum_term = np.sum(1.0 / (eigenvalues + gamma))
    k = (1.0 / d) * (lambda_max + gamma) * sum_term
    return k


def plot_eigenspectrum(all_eigenvalues, output_path, gamma=1.0):
    """
    Plot the eigenspectrum from collected eigenvalues and the K-metric histogram.
    """
    if not all_eigenvalues:
        print("No eigenvalues to plot!")
        return
    
    # Convert to numpy array for easier manipulation
    eigenvalues_array = np.array(all_eigenvalues)  # Shape: (num_samples, latent_dim)
    
    # Calculate K values for each sample
    k_values = np.array([calculate_k_metric(ev, gamma) for ev in eigenvalues_array])
    mean_k = np.mean(k_values)
    std_k = np.std(k_values)
    median_k = np.median(k_values)
    
    # Calculate statistics for eigenspectrum
    mean_eigenvalues = np.mean(eigenvalues_array, axis=0)
    std_eigenvalues = np.std(eigenvalues_array, axis=0)
    
    # Sort in descending order
    sorted_indices = np.argsort(mean_eigenvalues)[::-1]
    mean_eigenvalues = mean_eigenvalues[sorted_indices]
    std_eigenvalues = std_eigenvalues[sorted_indices]
    
    # Create figure with multiple subplots
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Eigenspectrum Analysis (Combined FIM)', fontsize=16)
    
    # Plot 1: Mean eigenspectrum with error bars
    ax1 = axes[0, 0]
    indices = np.arange(len(mean_eigenvalues))
    ax1.errorbar(indices, mean_eigenvalues, yerr=std_eigenvalues, 
                 fmt='o-', markersize=3, linewidth=1, capsize=2, alpha=0.7)
    ax1.set_xlabel('Eigenvalue Index (sorted)', fontsize=10)
    ax1.set_ylabel('Eigenvalue Magnitude', fontsize=10)
    ax1.set_title('Mean Eigenspectrum with Std Dev', fontsize=11)
    ax1.grid(True, alpha=0.3)
    ax1.set_yscale('log')
    
    # Plot 2: Log-scale eigenspectrum
    ax2 = axes[0, 1]
    ax2.semilogy(indices, mean_eigenvalues, 'o-', markersize=3, linewidth=1)
    ax2.fill_between(indices, 
                     mean_eigenvalues - std_eigenvalues, 
                     mean_eigenvalues + std_eigenvalues, 
                     alpha=0.3)
    ax2.set_xlabel('Eigenvalue Index (sorted)', fontsize=10)
    ax2.set_ylabel('Eigenvalue Magnitude (log scale)', fontsize=10)
    ax2.set_title('Mean Eigenspectrum (Log Scale)', fontsize=11)
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Cumulative sum of eigenvalues
    ax3 = axes[1, 0]
    cumsum_eigenvalues = np.cumsum(mean_eigenvalues)
    # Avoid division by zero
    if cumsum_eigenvalues[-1] > 0:
        cumsum_eigenvalues_normalized = cumsum_eigenvalues / cumsum_eigenvalues[-1]
    else:
        cumsum_eigenvalues_normalized = cumsum_eigenvalues
        
    ax3.plot(indices, cumsum_eigenvalues_normalized, 'o-', markersize=3, linewidth=2)
    ax3.axhline(y=0.9, color='r', linestyle='--', label='90% variance')
    ax3.axhline(y=0.95, color='g', linestyle='--', label='95% variance')
    ax3.axhline(y=0.99, color='b', linestyle='--', label='99% variance')
    ax3.set_xlabel('Number of Components', fontsize=10)
    ax3.set_ylabel('Cumulative Variance Explained', fontsize=10)
    ax3.set_title('Cumulative Eigenspectrum', fontsize=11)
    ax3.legend(fontsize=8)
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Histogram of K values
    ax4 = axes[1, 1]
    ax4.hist(k_values, bins=50, alpha=0.7, edgecolor='black', color='orange')
    ax4.axvline(x=mean_k, color='r', linestyle='--', 
                linewidth=2, label=f'Mean K: {mean_k:.4f}')
    ax4.set_xlabel(f'K Metric (gamma={gamma})', fontsize=10)
    ax4.set_ylabel('Frequency', fontsize=10)
    ax4.set_title(f'Distribution of K Metric\nMean: {mean_k:.4f} +/- {std_k:.4f}', fontsize=11)
    ax4.legend(fontsize=8)
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust for suptitle
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Eigenspectrum plot saved to: {output_path}")
    
    # Print summary statistics
    print(f"\n=== Eigenspectrum Summary ===")
    print(f"Number of samples analyzed: {len(all_eigenvalues)}")
    print(f"Latent dimension: {len(mean_eigenvalues)}")
    print(f"Top 10 mean eigenvalues: {mean_eigenvalues[:10]}")
    if cumsum_eigenvalues_normalized[-1] > 0:
        print(f"Effective rank (90% variance): {np.argmax(cumsum_eigenvalues_normalized >= 0.9) + 1}")
        print(f"Effective rank (95% variance): {np.argmax(cumsum_eigenvalues_normalized >= 0.95) + 1}")
    
    # Print K metric stats
    print(f"\n=== K Metric Statistics (gamma={gamma}) ===")
    print(f"Average K: {mean_k:.6f}")
    print(f"Median K: {median_k:.6f}")
    print(f"Std Dev K: {std_k:.6f}")
    print(f"Min K: {np.min(k_values):.6f}")
    print(f"Max K: {np.max(k_values):.6f}")


def main():
    parser = argparse.ArgumentParser(description='Calculate and plot eigenspectrum of Combined FIM from test set.')
    parser.add_argument('--experiment_path', type=str, required=True, 
                        help='Path to experiment results directory.')
    parser.add_argument('--activation_dir', type=str, default=None, 
                        help='Path to pre-computed activations.')
    parser.add_argument('--output_path', type=str, default='eigenspec.png', 
                        help='Path to save the eigenspectrum plot.')
    parser.add_argument('--num_samples', type=int, default=None, 
                        help='Number of test samples to process (None for all).')
    parser.add_argument('--gpu_id', default=0, type=int, help='GPU ID to use.')
    parser.add_argument('--batch_size', type=int, default=16, 
                        help='Batch size for data loading.')
    cli_args = parser.parse_args()

    device = torch.device(f'cuda:{cli_args.gpu_id}' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    print("Loading model...")
    model_data = load_model_for_utility_check(cli_args.experiment_path, device)
    model = model_data['model']
    train_args = model_data['train_args']
    
    # Load all available heads
    heads = model['heads'] if 'heads' in model else {train_args.task: model['head']}
    print(f"Available tasks for FIM calculation: {list(heads.keys())}")
    
    # Determine latent dim from one of the heads
    first_head = next(iter(heads.values()))
    latent_dim = first_head.input_dim
    print(f"Latent dimension: {latent_dim}")

    # Load test dataset
    # Use explicit activation_dir if provided, else try to find it in experiment_path
    activation_dir = cli_args.activation_dir or os.path.join(cli_args.experiment_path, 'activations')
    
    # Ensure activation directory exists
    if not os.path.exists(activation_dir):
        print(f"Error: Activation directory not found at {activation_dir}")
        # Try a fallback or check if user provided the alternative path manually
        return

    try:
        with open(train_args.pkl_dataset_path, 'rb') as f:
            partitions = pickle.load(f)
    except FileNotFoundError:
         # Fallback for dataset path if environment variables are different
         pkl_path = os.environ.get('PKL_FILE_PATH')
         if pkl_path:
             dataset_name = os.path.basename(train_args.pkl_dataset_path)
             alternative_path = os.path.join(pkl_path, dataset_name)
             if os.path.exists(alternative_path):
                 print(f"Using alternative dataset path: {alternative_path}")
                 with open(alternative_path, 'rb') as f:
                    partitions = pickle.load(f)
             else:
                 raise
         else:
             raise
    
    test_files = partitions['test']
    random.shuffle(test_files)
    test_dataset = ActivationDataset(activation_dir=activation_dir, file_list=test_files, test_mode=True)
    test_dl = DataLoader(test_dataset, batch_size=cli_args.batch_size, shuffle=False)
    print(f"Loaded {len(test_dataset)} test samples")

    # Collect eigenvalues from test set using combined FIM
    print(f"\nCalculating eigenvalues for test samples (Summing FIM across tasks)...")
    all_eigenvalues = collect_eigenvalues_from_test_set(
        heads, test_dl, device, latent_dim, cli_args.num_samples
    )
    
    # Plot eigenspectrum
    print(f"\nPlotting eigenspectrum...")
    plot_eigenspectrum(all_eigenvalues, cli_args.output_path)
    
    print("\n✅ Eigenspectrum analysis complete!")


if __name__ == '__main__':
    main()
