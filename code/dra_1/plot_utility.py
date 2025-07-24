#!/usr/bin/env python3
"""
Plot utility vs noise level for different tasks and noise types.

This script evaluates model utility (performance) across varying noise levels
for both isotropic and non-isotropic noise injection into latent activations.

Usage:
    python plot_utility.py --experiment_path /path/to/experiment/directory
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
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, mean_absolute_error, mean_squared_error, r2_score
from torch.utils.data import DataLoader
import torch.nn as nn

# Add project root to path to allow importing from other directories
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from dra_1.py_datasets import ActivationDataset
from dra_1.inject_noise import inject_isotropic_noise, inject_nonisotropic_noise
from dra_1.model_loader import load_model_for_utility_check


def get_empirical_fim(head, data_loader, device, latent_dim):
    """Calculates the empirical Fisher Information Matrix for a given task head."""
    fim = torch.zeros((latent_dim, latent_dim), device=device)
    criterion = nn.MSELoss()
    num_samples = 0
    head.to(device).eval()

    # Determine if the task is classification or regression
    from rep_lr.models import RFClassificationHead
    is_classification = isinstance(head, RFClassificationHead)
    if is_classification:
        criterion = nn.CrossEntropyLoss()
    else:
        criterion = nn.MSELoss()

    for batch in tqdm(data_loader, desc="Calculating FIM", leave=False):
        # Unpack based on what ActivationDataset yields in train mode
        _, rf_labels, _, cfo_labels, _, ch_labels, activations, _ = batch
        
        activations = activations.squeeze(1).to(device)
        activations = activations.view(activations.size(0), -1)
        activations.requires_grad = True

        head.zero_grad()
        
        # Select target based on task type
        if is_classification:
            y_target = rf_labels.to(device)
            if y_target.dim() > 1:
                y_target = y_target.view(-1)
        else:
            # For regression tasks, determine which labels to use based on head type
            from rep_lr.models import CFOEstimationHead, ChannelEstimationHead
            if isinstance(head, CFOEstimationHead):
                y_target = cfo_labels.to(device).float()
                if y_target.dim() > 1:
                    y_target = y_target.view(-1)  # Flatten CFO labels
            elif isinstance(head, ChannelEstimationHead):
                y_target = ch_labels.to(device)
                # Fix channel label shape: [batch, 1, 2, 52] -> [batch, 2, 52]
                if y_target.dim() == 4 and y_target.size(1) == 1:
                    y_target = y_target.squeeze(1)
            else:
                continue

        y_hat = head(activations)
        if y_hat.dim() > 1 and y_target.dim() == 1 and not is_classification:
             y_hat = y_hat.squeeze(-1)

        loss = criterion(y_hat, y_target)
        loss.backward()
        
        if activations.grad is not None:
            J = activations.grad
            fim += J.T @ J
            num_samples += activations.size(0)

    if num_samples == 0:
        return fim
    return fim / num_samples


def evaluate_rf_utility_metric(head, test_dl, device, noise_level, L=None, V=None, noise_type='isotropic'):
    """Evaluate RF fingerprinting utility and return accuracy metric."""
    head.to(device).eval()
    y_true, y_pred = [], []

    with torch.no_grad():
        for _, rf_labels, _, _, _, _, activations, _ in test_dl:
            activations = activations.squeeze(1).to(device)
            rf_labels = rf_labels.to(device)
            
            # Flatten for noise injection
            activations_flat = activations.view(activations.size(0), -1)

            # Apply noise
            if noise_type == 'isotropic':
                noisy_activations = inject_isotropic_noise(activations_flat, noise_level)
            elif noise_type == 'nonisotropic':
                noisy_activations = inject_nonisotropic_noise(activations_flat, noise_level, L, V)
            else:
                noisy_activations = activations_flat
            
            # Reshape back to (batch, 2, dim) for the head
            noisy_activations = noisy_activations.view(noisy_activations.size(0), 2, -1)

            # Get predictions
            outputs = head(noisy_activations)
            _, predicted = torch.max(outputs, 1)

            y_true.extend(rf_labels.cpu().numpy().flatten())
            y_pred.extend(predicted.cpu().numpy())

    # Calculate accuracy
    accuracy = np.mean(np.array(y_true) == np.array(y_pred))
    return accuracy


def evaluate_cfo_utility_metric(head, test_dl, device, noise_level, L=None, V=None, noise_type='isotropic'):
    """Evaluate CFO estimation utility and return R2 score metric."""
    head.to(device).eval()
    y_true, y_pred = [], []
    
    with torch.no_grad():
        for _, _, _, labels, _, _, activations, _ in test_dl:
            activations = activations.squeeze(1).to(device)
            labels = labels.to(device).float()

            activations = activations.view(activations.size(0), -1)

            if noise_type == 'isotropic':
                noisy_activations = inject_isotropic_noise(activations, noise_level)
            elif noise_type == 'nonisotropic':
                noisy_activations = inject_nonisotropic_noise(activations, noise_level, L, V)
            else:
                noisy_activations = activations

            # Reshape back to (batch, 2, dim) for the head
            noisy_activations = noisy_activations.view(noisy_activations.size(0), 2, -1)
            
            outputs = head(noisy_activations).squeeze(-1)

            y_true.extend(labels.cpu().numpy())
            y_pred.extend(outputs.cpu().numpy())

    # Calculate R2 score (higher is better, like accuracy)
    r2 = r2_score(y_true, y_pred)
    return r2


def evaluate_channel_utility_metric(head, test_dl, device, noise_level, L=None, V=None, noise_type='isotropic'):
    """Evaluate channel estimation utility and return R² score."""
    head.to(device).eval()
    all_y_true, all_y_pred = [], []

    with torch.no_grad():
        for _, _, _, _, _, labels, activations, _ in test_dl:
            activations = activations.squeeze(1).to(device)
            labels = labels.to(device)
            
            # Flatten for noise injection (same as working check_utility.py)
            activations = activations.view(activations.size(0), -1)

            # Apply noise (exact same logic as working check_utility.py)
            if noise_type == 'isotropic':
                noisy_activations = inject_isotropic_noise(activations, noise_level)
            elif noise_type == 'nonisotropic':
                noisy_activations = inject_nonisotropic_noise(activations, noise_level, L, V)
            else:
                noisy_activations = activations
            
            # Reshape back to (batch, 2, dim) for the head
            noisy_activations = noisy_activations.view(noisy_activations.size(0), 2, -1)

            outputs = head(noisy_activations)
            
            # Store results exactly like working check_utility.py
            for i in range(labels.size(0)):
                true_val = labels[i].cpu().numpy()
                pred_val = outputs[i].cpu().numpy()
                all_y_true.append(true_val)
                all_y_pred.append(pred_val)

    y_true = np.array(all_y_true)
    y_pred = np.array(all_y_pred)

    # Reshape exactly like working check_utility.py
    if y_true.shape[1] == 1 and y_true.shape[2] == 104:
        y_true = y_true.reshape(y_true.shape[0], 2, 52)
    if y_pred.shape[1] == 1 and y_pred.shape[2] == 104:
        y_pred = y_pred.reshape(y_pred.shape[0], 2, 52)

    if len(y_true.shape) == 4 and y_true.shape[1] == 1:
        y_true = np.squeeze(y_true, axis=1)
    
    # Calculate R² exactly like working check_utility.py
    y_true_real = y_true[:, 0, :]
    y_pred_real = y_pred[:, 0, :]
    y_true_imag = y_true[:, 1, :]
    y_pred_imag = y_pred[:, 1, :]

    r2_real = r2_score(y_true_real, y_pred_real)
    r2_imag = r2_score(y_true_imag, y_pred_imag)
    
    # Return average R² score like working check_utility.py
    r2_avg = (r2_real + r2_imag) / 2.0
    
    return r2_avg


def evaluate_utility_for_task(task_name, head, test_dl, device, noise_level, L=None, V=None, noise_type='isotropic'):
    """Helper function to call the correct utility metric function based on task name."""
    if task_name == 'rf_fingerprinting':
        return evaluate_rf_utility_metric(head, test_dl, device, noise_level, L, V, noise_type)
    elif task_name == 'cfo_estimation':
        return evaluate_cfo_utility_metric(head, test_dl, device, noise_level, L, V, noise_type)
    elif task_name == 'channel_estimation':
        return evaluate_channel_utility_metric(head, test_dl, device, noise_level, L, V, noise_type)
    else:
        raise ValueError(f"Unknown task: {task_name}")


def evaluate_utility_across_noise(heads, test_dl, device, noise_levels, train_files, activation_dir, cli_args):
    """
    Evaluate utility across different noise levels for all tasks.
    
    Returns:
        results (dict): Dictionary with task names as keys, each containing
                       'isotropic' and 'nonisotropic' arrays of utility values
    """
    results = {}
    
    for task_name, head in heads.items():
        print(f"\nEvaluating {task_name} across noise levels...")
        
        results[task_name] = {
            'isotropic': [],
            'nonisotropic': []
        }
        
        # --- FIM Calculation (Task-Specific) ---
        task_fim_data = None
        if not cli_args.skip_nonisotropic:
            print(f"  Calculating FIM for task: {task_name}")
            if not train_files:
                print("  Warning: Training files not available, skipping non-isotropic noise for this task.")
            else:
                # Use a fresh FIM dataloader each time to ensure consistency
                train_files_shuffled = train_files.copy()
                random.shuffle(train_files_shuffled)
                fim_files = train_files_shuffled[:cli_args.fim_samples]
                fim_dataset = ActivationDataset(activation_dir=activation_dir, file_list=fim_files, test_mode=False)
                fim_dl = DataLoader(fim_dataset, batch_size=cli_args.batch_size, shuffle=True)
                
                latent_dim = head.input_dim
                fim = get_empirical_fim(head, fim_dl, device, latent_dim)
                
                # Perform eigendecomposition for this specific task's FIM
                L_e, V = torch.linalg.eigh(fim)
                L = torch.relu(L_e)
                task_fim_data = (L, V)
                print(f"  FIM calculation for {task_name} complete.")
        
        # --- Utility Evaluation ---
        for noise_level in tqdm(noise_levels, desc=f"{task_name} evaluation"):
            # Evaluate with isotropic noise
            iso_utility = evaluate_utility_for_task(task_name, head, test_dl, device, noise_level, noise_type='isotropic')
            results[task_name]['isotropic'].append(iso_utility)
            
            # Evaluate with non-isotropic noise using the task-specific FIM
            if task_fim_data is not None:
                L, V = task_fim_data
                noniso_utility = evaluate_utility_for_task(task_name, head, test_dl, device, noise_level, L, V, 'nonisotropic')
                results[task_name]['nonisotropic'].append(noniso_utility)
            else:
                results[task_name]['nonisotropic'].append(None)
    
    return results


def plot_fim_heatmaps(individual_fims, total_fim, output_dir):
    """Create and save heatmaps of Fisher Information Matrices."""
    
    # Set style for heatmaps
    plt.style.use('default')
    
    # Plot individual FIMs
    for task_name, fim in individual_fims.items():
        plt.figure(figsize=(10, 8))
        
        # Convert to numpy and use log scale for better visualization
        fim_np = fim.cpu().numpy()
        fim_log = np.log10(fim_np + 1e-10)  # Add small epsilon to avoid log(0)
        
        # Create heatmap
        im = plt.imshow(fim_log, cmap='viridis', aspect='auto')
        plt.colorbar(im, label='log₁₀(Fisher Information)')
        
        plt.title(f'Fisher Information Matrix - {task_name.replace("_", " ").title()}', 
                 fontsize=14, fontweight='bold')
        plt.xlabel('Latent Dimension', fontsize=12)
        plt.ylabel('Latent Dimension', fontsize=12)
        
        # Save plot
        plot_filename = f'fim_heatmap_{task_name}.png'
        plot_path = os.path.join(output_dir, plot_filename)
        plt.tight_layout()
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"Saved FIM heatmap: {plot_path}")
        plt.close()
    
    # Plot summed FIM
    plt.figure(figsize=(10, 8))
    
    # Convert to numpy and use log scale
    total_fim_np = total_fim.cpu().numpy()
    total_fim_log = np.log10(total_fim_np + 1e-10)
    
    # Create heatmap
    im = plt.imshow(total_fim_log, cmap='viridis', aspect='auto')
    plt.colorbar(im, label='log₁₀(Fisher Information)')
    
    plt.title('Summed Fisher Information Matrix (All Tasks)', 
             fontsize=14, fontweight='bold')
    plt.xlabel('Latent Dimension', fontsize=12)
    plt.ylabel('Latent Dimension', fontsize=12)
    
    # Save plot
    plot_filename = 'fim_heatmap_summed.png'
    plot_path = os.path.join(output_dir, plot_filename)
    plt.tight_layout()
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"Saved summed FIM heatmap: {plot_path}")
    plt.close()
    
    # Create comparison plot with all FIMs as subplots
    num_tasks = len(individual_fims)
    fig, axes = plt.subplots(1, num_tasks + 1, figsize=(5 * (num_tasks + 1), 4))
    
    # Plot individual FIMs
    for idx, (task_name, fim) in enumerate(individual_fims.items()):
        fim_np = fim.cpu().numpy()
        fim_log = np.log10(fim_np + 1e-10)
        
        im = axes[idx].imshow(fim_log, cmap='viridis', aspect='auto')
        axes[idx].set_title(f'{task_name.replace("_", " ").title()}', fontsize=10)
        axes[idx].set_xlabel('Latent Dim', fontsize=9)
        if idx == 0:
            axes[idx].set_ylabel('Latent Dim', fontsize=9)
        
        # Add colorbar for each subplot
        cbar = plt.colorbar(im, ax=axes[idx], fraction=0.046, pad=0.04)
        cbar.set_label('log₁₀(FI)', fontsize=8)
    
    # Plot summed FIM
    total_fim_np = total_fim.cpu().numpy()
    total_fim_log = np.log10(total_fim_np + 1e-10)
    
    im = axes[-1].imshow(total_fim_log, cmap='viridis', aspect='auto')
    axes[-1].set_title('Summed FIM', fontsize=10)
    axes[-1].set_xlabel('Latent Dim', fontsize=9)
    
    # Add colorbar for summed FIM
    cbar = plt.colorbar(im, ax=axes[-1], fraction=0.046, pad=0.04)
    cbar.set_label('log₁₀(FI)', fontsize=8)
    
    plt.suptitle('Fisher Information Matrix Comparison', fontsize=12, fontweight='bold')
    plt.tight_layout()
    
    # Save comparison plot
    comparison_plot_path = os.path.join(output_dir, 'fim_heatmaps_comparison.png')
    plt.savefig(comparison_plot_path, dpi=300, bbox_inches='tight')
    print(f"Saved FIM comparison plot: {comparison_plot_path}")
    plt.close()


def plot_utility_results(results, noise_levels, output_dir):
    """Create and save utility vs noise level plots for each task."""
    
    # Define metric names and labels for each task
    task_info = {
        'rf_fingerprinting': {
            'metric_name': 'Accuracy',
            'title': 'RF Fingerprinting Utility vs Noise Level'
        },
        'cfo_estimation': {
            'metric_name': 'R² Score', 
            'title': 'CFO Estimation Utility vs Noise Level'
        },
        'channel_estimation': {
            'metric_name': 'R² Score',
            'title': 'Channel Estimation Utility vs Noise Level'
        }
    }
    
    # Set style
    plt.style.use('seaborn-v0_8')
    colors = ['#2E86AB', '#A23B72']  # Blue for isotropic, Pink for non-isotropic
    
    for task_name, task_results in results.items():
        if task_name not in task_info:
            continue
            
        plt.figure(figsize=(10, 6))
        
        # Plot isotropic results
        isotropic_values = task_results['isotropic']
        plt.plot(noise_levels, isotropic_values, 'o-', color=colors[0], 
                linewidth=2, markersize=6, label='Isotropic Noise', alpha=0.8)
        
        # Plot non-isotropic results if available
        nonisotropic_values = task_results['nonisotropic']
        if nonisotropic_values and all(v is not None for v in nonisotropic_values):
            plt.plot(noise_levels, nonisotropic_values, 's--', color=colors[1], 
                    linewidth=2, markersize=6, label='Non-isotropic Noise', alpha=0.8)
        
        # Formatting
        plt.xlabel('Noise Level', fontsize=12, fontweight='bold')
        plt.ylabel(f'{task_info[task_name]["metric_name"]}', fontsize=12, fontweight='bold')
        plt.title(task_info[task_name]["title"], fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3)
        plt.legend(fontsize=11)
        
        # Add some padding to y-axis
        y_min, y_max = plt.ylim()
        y_range = y_max - y_min
        plt.ylim(y_min - 0.05 * y_range, y_max + 0.05 * y_range)
        
        # Save plot
        plot_filename = f'{task_name}_utility_vs_noise.png'
        plot_path = os.path.join(output_dir, plot_filename)
        plt.tight_layout()
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"Saved plot: {plot_path}")
        plt.close()
    
    # Create a summary plot with all tasks
    fig, axes = plt.subplots(1, len(results), figsize=(5 * len(results), 5))
    if len(results) == 1:
        axes = [axes]
    
    for idx, (task_name, task_results) in enumerate(results.items()):
        if task_name not in task_info:
            continue
            
        ax = axes[idx]
        
        # Plot isotropic results
        isotropic_values = task_results['isotropic']
        ax.plot(noise_levels, isotropic_values, 'o-', color=colors[0], 
               linewidth=2, markersize=4, label='Isotropic', alpha=0.8)
        
        # Plot non-isotropic results if available
        nonisotropic_values = task_results['nonisotropic']
        if nonisotropic_values and all(v is not None for v in nonisotropic_values):
            ax.plot(noise_levels, nonisotropic_values, 's--', color=colors[1], 
                   linewidth=2, markersize=4, label='Non-isotropic', alpha=0.8)
        
        ax.set_xlabel('Noise Level', fontsize=10)
        ax.set_ylabel(task_info[task_name]["metric_name"], fontsize=10)
        ax.set_title(task_name.replace('_', ' ').title(), fontsize=11, fontweight='bold')
        ax.grid(True, alpha=0.3)
        if idx == 0:  # Only add legend to first subplot
            ax.legend(fontsize=9)
    
    plt.tight_layout()
    summary_plot_path = os.path.join(output_dir, 'all_tasks_utility_vs_noise.png')
    plt.savefig(summary_plot_path, dpi=300, bbox_inches='tight')
    print(f"Saved summary plot: {summary_plot_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Plot utility vs noise level for different tasks and noise types.')
    parser.add_argument('--experiment_path', type=str, required=True, 
                       help='Path to the experiment results directory.')
    parser.add_argument('--output_dir', type=str, default=None, 
                       help='Directory to save plots. Defaults to [experiment_path]/utility_plots.')
    parser.add_argument('--noise_min', type=float, default=0.0, 
                       help='Minimum noise level.')
    parser.add_argument('--noise_max', type=float, default=10.0, 
                       help='Maximum noise level.')
    parser.add_argument('--noise_steps', type=int, default=21, 
                       help='Number of noise levels to test.')
    parser.add_argument('--fim_samples', type=int, default=8000, 
                       help='Number of samples to use for FIM calculation.')
    parser.add_argument('--gpu_id', default=0, type=int, 
                       help='ID of GPU to be used.')
    parser.add_argument('--batch_size', type=int, default=64, 
                       help='Batch size for evaluation.')
    parser.add_argument('--skip_nonisotropic', action='store_true',
                       help='Skip non-isotropic noise evaluation (faster).')
    parser.add_argument('--load_fim_data', type=str, default=None,
                       help='Path to previously saved FIM data (.pt file) to load instead of recalculating.')
    parser.add_argument('--plot_fim_only', action='store_true',
                       help='Only plot FIM heatmaps (requires --load_fim_data), skip utility evaluation.')
    
    cli_args = parser.parse_args()
    
    # Setup output directory
    if cli_args.output_dir is None:
        cli_args.output_dir = os.path.join(cli_args.experiment_path, 'utility_plots')
    os.makedirs(cli_args.output_dir, exist_ok=True)
    print(f"Saving plots to: {cli_args.output_dir}")
    
    # Setup device
    device = torch.device(f'cuda:{cli_args.gpu_id}' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Generate noise levels
    noise_levels = np.linspace(cli_args.noise_min, cli_args.noise_max, cli_args.noise_steps)
    print(f"Testing noise levels: {noise_levels}")
    
    # Load model and setup
    print("Loading model...")
    model_data = load_model_for_utility_check(cli_args.experiment_path, device)
    model = model_data['model']
    train_args = model_data['train_args']
    is_mtl = model_data['is_mtl']
    
    # Extract heads from the loaded model
    heads = model['heads'] if 'heads' in model else {train_args.task: model['head']}
    print(f"Model loaded successfully. MTL: {is_mtl}")
    print(f"Tasks to evaluate: {list(heads.keys())}")
    
    # Setup data
    activation_dir = os.path.join(cli_args.experiment_path, 'activations')
    if not os.path.isdir(activation_dir):
        raise FileNotFoundError(f"'activations' directory not found at: {activation_dir}")
    
    partition_file = train_args.pkl_dataset_path
    with open(partition_file, 'rb') as f:
        partitions = pickle.load(f)
    
    train_files = partitions.get('train')
    test_files = partitions.get('test')
    if not test_files:
        raise ValueError("Could not find 'test' key in the partition file.")
    
    # Shuffle test files to ensure we get samples from all classes
    # This is important because files are often ordered by class/radio ID
    import random
    test_files_shuffled = test_files.copy()
    random.shuffle(test_files_shuffled)
    
    test_dataset = ActivationDataset(activation_dir=activation_dir, file_list=test_files_shuffled, test_mode=True)
    test_dl = DataLoader(test_dataset, batch_size=cli_args.batch_size, shuffle=True, num_workers=4)
    
    # Evaluate utility across noise levels
    print("\nEvaluating utility across noise levels...")
    results = evaluate_utility_across_noise(heads, test_dl, device, noise_levels, train_files, activation_dir, cli_args)
    
    # Save results to JSON
    results_json = {}
    for task_name, task_results in results.items():
        results_json[task_name] = {
            'isotropic': [float(x) for x in task_results['isotropic']],
            'nonisotropic': [float(x) if x is not None else None for x in task_results['nonisotropic']]
        }
    
    results_path = os.path.join(cli_args.output_dir, 'utility_results.json')
    with open(results_path, 'w') as f:
        json.dump({
            'noise_levels': noise_levels.tolist(),
            'results': results_json
        }, f, indent=4)
    print(f"Results saved to: {results_path}")
    
    # Create plots
    print("\nGenerating plots...")
    plot_utility_results(results, noise_levels, cli_args.output_dir)
    
    print(f"Utility analysis complete! Check {cli_args.output_dir} for plots and results.")


if __name__ == '__main__':
    main() 