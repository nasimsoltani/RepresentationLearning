import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import random

# Add necessary paths to import other modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'code')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'code', 'dra_1')))

from dra_1.py_datasets import ActivationDataset


def main():
    """Load activations, compute sensitivity ||Z - Z'||_2, and plot histogram."""
    
    # Activation directory
    activation_dir = "/scratch/10608/aadharsh_aadhithya/data/rep_lr/activations_exps/run_1/rf_cfo_channel"
    
    if not os.path.isdir(activation_dir):
        raise FileNotFoundError(f"Activations directory not found at {activation_dir}")
    
    # List all .pth files in the directory
    print(f"Scanning activation files in {activation_dir}...")
    all_pth_files = [f for f in os.listdir(activation_dir) if f.endswith('.pth')]
    print(f"Found {len(all_pth_files)} .pth files")
    
    if len(all_pth_files) == 0:
        raise ValueError(f"No .pth files found in {activation_dir}")
    
    # Randomly sample files (or all if less than the target)
    num_samples = min(10000, len(all_pth_files))
    sampled_pth_files = random.sample(all_pth_files, num_samples)
    print(f"Randomly sampled {num_samples} files")
    
    # Convert .pth filenames to .mat filenames for the file_list
    # Handle both regular files and slice files
    file_list = []
    for pth_file in sampled_pth_files:
        # Remove .pth extension and add .mat
        if '_slice' in pth_file:
            # For slice files like "file_slice0.pth", convert to "file_slice0.mat"
            mat_file = pth_file.replace('.pth', '.mat')
        else:
            # For regular files, convert .pth to .mat
            mat_file = pth_file.replace('.pth', '.mat')
        file_list.append(mat_file)
    
    # Create dataset using ActivationDataset
    print("Creating ActivationDataset...")
    dataset = ActivationDataset(activation_dir=activation_dir, file_list=file_list, test_mode=True)
    
    # Create DataLoader
    batch_size = 64
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    
    # Collect all activations (flattened)
    print("Loading activations...")
    all_activations = []
    
    for batch in data_loader:
        rf_x, rf_y, cfo_x, cfo_y, channel_x, channel_y, activations, filenames = batch
        
        # Flatten activations: (batch_size, ...) -> (batch_size, -1)
        activations = activations.squeeze(1)  # Remove any extra dimension
        activations = activations.view(activations.size(0), -1)
        
        # Store flattened activations
        all_activations.append(activations.cpu())

    print(f"Loaded {len(all_activations)} activations")
    
    # Concatenate all activations into a single tensor
    all_activations = torch.cat(all_activations, dim=0)
    print(f"Loaded {all_activations.shape[0]} activations of dimension {all_activations.shape[1]}")
    
    # Sample random pairs and compute ||Z - Z'||_2
    num_pairs = min(5000, all_activations.shape[0] * (all_activations.shape[0] - 1) // 2)
    print(f"Computing sensitivity ||Z - Z'||_2 for {num_pairs} random pairs...")
    
    sensitivities = []
    indices = list(range(all_activations.shape[0]))
    
    for _ in range(num_pairs):
        # Sample two different random indices
        i, j = random.sample(indices, 2)
        Z = all_activations[i]
        Z_prime = all_activations[j]
        
        # Compute ||Z - Z'||_2
        diff = Z - Z_prime
        sensitivity = torch.norm(diff, p=2).item()
        sensitivities.append(sensitivity)
    
    sensitivities = np.array(sensitivities)
    print(f"Computed sensitivities for {len(sensitivities)} pairs")
    print(f"Sensitivity statistics:")
    print(f"  Mean: {np.mean(sensitivities):.6f}")
    print(f"  Std: {np.std(sensitivities):.6f}")
    print(f"  Min: {np.min(sensitivities):.6f}")
    print(f"  Max: {np.max(sensitivities):.6f}")
    print(f"  Median: {np.median(sensitivities):.6f}")
    
    # Plot histogram
    print("Plotting histogram...")
    plt.figure(figsize=(10, 6))
    plt.hist(sensitivities, bins=50, edgecolor='black', alpha=0.7)
    plt.xlabel('Sensitivity ||Z - Z\'||₂', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.title(f'Histogram of Activation Sensitivity (n={len(sensitivities)} pairs)', fontsize=14)
    plt.grid(True, alpha=0.3)
    
    # Add statistics text
    stats_text = f'Mean: {np.mean(sensitivities):.4f}\nStd: {np.std(sensitivities):.4f}\nMedian: {np.median(sensitivities):.4f}'
    plt.text(0.7, 0.95, stats_text, transform=plt.gca().transAxes,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # Save the plot
    output_path = 'activation_sensitivity_histogram.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Histogram saved to {output_path}")
    plt.close()
    
    print("Done!")


if __name__ == '__main__':
    # Set random seed for reproducibility
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    
    main()

