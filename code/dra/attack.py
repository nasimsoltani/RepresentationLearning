#!/usr/bin/env python3

import os
import sys
import argparse
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import wandb
from tqdm import tqdm
import pickle

# Add parent directories to path for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from dra.py_datasets import ActivationDataset
from dra.models import get_adversary
from rep_lr.models import ComplexSequenceProjector, Encoder


def parse_args():
    """Parse command line arguments for the reconstruction attack."""
    parser = argparse.ArgumentParser(description='Data Reconstruction Attack on RF Representations')
    
    # Model and data paths
    parser.add_argument('--model_path', type=str, required=True, 
                        help='Path to the trained model checkpoint')
    parser.add_argument('--activations_dir', type=str, required=True,
                        help='Directory containing saved activations (.pth files)')
    
    # Attack parameters
    parser.add_argument('--noise_type', type=str, choices=['none', 'isotropic', 'anisotropic'], 
                        default='none', help='Type of noise to add to activations')
    parser.add_argument('--noise_level', type=float, default=0.0,
                        help='Level of noise to add (variance for isotropic, base level for anisotropic)')
    parser.add_argument('--target_data', type=str, choices=['RF_X', 'CFO_X', 'Channel_X'], 
                        default='RF_X', help='Which data type to reconstruct')
    
    # Adversary model parameters
    parser.add_argument('--adversary_type', type=str, choices=['simple', 'conv', 'residual'], 
                        default='simple', help='Type of adversary model to use')
    parser.add_argument('--adv_hidden_dim', type=int, default=512,
                        help='Hidden dimension for adversary model')
    parser.add_argument('--adv_dropout', type=float, default=0.2,
                        help='Dropout probability for adversary model')
    
    # Training parameters
    parser.add_argument('--epochs', type=int, default=50, 
                        help='Number of epochs to train the adversary')
    parser.add_argument('--batch_size', type=int, default=32, 
                        help='Batch size for training')
    parser.add_argument('--learning_rate', type=float, default=1e-3,
                        help='Learning rate for adversary training')
    parser.add_argument('--train_ratio', type=float, default=0.7,
                        help='Ratio of data to use for training (rest for testing)')
    
    # Hardware and logging
    parser.add_argument('--gpu_id', type=int, default=0, 
                        help='GPU ID to use')
    parser.add_argument('--save_path', type=str, default='./attack_results',
                        help='Directory to save attack results')
    parser.add_argument('--wandb_project', type=str, default='rf-dra',
                        help='Wandb project name')
    parser.add_argument('--wandb_run_name', type=str, default=None,
                        help='Wandb run name (if None, auto-generated)')
    
    # Debug options
    parser.add_argument('--debug', action='store_true',
                        help='Enable debug mode with smaller dataset')
    parser.add_argument('--save_plots', action='store_true', default=True,
                        help='Save reconstruction plots')
    
    return parser.parse_args()


def get_data_shape(target_data):
    """Get the expected shape for different data types."""
    # These are the typical shapes based on the dataset
    if target_data == 'RF_X':
        return (2, 1024)  # Most common RF data shape
    elif target_data == 'CFO_X':
        return (2, 160)   # CFO data shape
    elif target_data == 'Channel_X':
        return (2, 160)   # Channel data shape
    else:
        raise ValueError(f"Unknown target data type: {target_data}")


def load_model_info(model_path):
    """Load model information from checkpoint and args."""
    model_dir = os.path.dirname(model_path)
    args_path = os.path.join(model_dir, 'args.json')
    
    if not os.path.exists(args_path):
        raise FileNotFoundError(f"args.json not found in {model_dir}")
    
    with open(args_path, 'r') as f:
        train_args = argparse.Namespace(**json.load(f))
    
    checkpoint = torch.load(model_path, map_location='cpu')
    
    return train_args, checkpoint


def create_dataset_and_loader(args, train_args):
    """Create the activation dataset and data loaders."""
    # Create a modified args object for the dataset
    dataset_args = argparse.Namespace()
    dataset_args.model_path = args.model_path
    dataset_args.noise_type = args.noise_type
    dataset_args.noise_level = args.noise_level
    dataset_args.gpu_id = args.gpu_id
    
    # Create dataset
    dataset = ActivationDataset(args.activations_dir, dataset_args)
    
    if args.debug:
        # Use only a small subset for debugging
        dataset = Subset(dataset, range(min(100, len(dataset))))
    
    # Split into train and test
    n_samples = len(dataset)
    n_train = int(args.train_ratio * n_samples)
    indices = torch.randperm(n_samples)
    train_indices = indices[:n_train]
    test_indices = indices[n_train:]
    
    train_dataset = Subset(dataset, train_indices)
    test_dataset = Subset(dataset, test_indices)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, 
                             shuffle=True, num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, 
                            shuffle=False, num_workers=2, pin_memory=True)
    
    return train_loader, test_loader, dataset


def train_adversary(adversary, train_loader, criterion, optimizer, device, epoch, target_data_key):
    """Train the adversary for one epoch."""
    adversary.train()
    total_loss = 0.0
    num_batches = 0
    
    pbar = tqdm(train_loader, desc=f"Epoch {epoch+1} [Train]")
    for batch_idx, batch in enumerate(pbar):
        # Get activations and target data
        activations = batch['activation'].to(device, non_blocking=True).float()
        target_data = batch[target_data_key].to(device, non_blocking=True).float()
        
        # Flatten activations: (B, 2, 256) -> (B, 512)
        z = activations.view(activations.size(0), -1)
        
        # Forward pass
        optimizer.zero_grad()
        reconstructed = adversary(z)
        loss = criterion(reconstructed, target_data)
        
        # Backward pass
        loss.backward()
        torch.nn.utils.clip_grad_norm_(adversary.parameters(), max_norm=5.0)
        optimizer.step()
        
        total_loss += loss.item()
        num_batches += 1
        
        pbar.set_postfix({'Loss': f'{loss.item():.4f}'})
    
    return total_loss / num_batches


def evaluate_adversary(adversary, test_loader, criterion, device, target_data_key):
    """Evaluate the adversary on test data."""
    adversary.eval()
    total_loss = 0.0
    num_batches = 0
    all_reconstructed = []
    all_targets = []
    all_activations = []
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating"):
            # Get activations and target data
            activations = batch['activation'].to(device, non_blocking=True).float()
            target_data = batch[target_data_key].to(device, non_blocking=True).float()
            
            # Flatten activations
            z = activations.view(activations.size(0), -1)
            
            # Forward pass
            reconstructed = adversary(z)
            loss = criterion(reconstructed, target_data)
            
            total_loss += loss.item()
            num_batches += 1
            
            # Store for analysis
            all_reconstructed.append(reconstructed.cpu())
            all_targets.append(target_data.cpu())
            all_activations.append(activations.cpu())
    
    avg_loss = total_loss / num_batches
    
    # Concatenate all results
    all_reconstructed = torch.cat(all_reconstructed, dim=0)
    all_targets = torch.cat(all_targets, dim=0)
    all_activations = torch.cat(all_activations, dim=0)
    
    return avg_loss, all_reconstructed, all_targets, all_activations


def calculate_metrics(reconstructed, targets):
    """Calculate various reconstruction quality metrics."""
    # Mean Squared Error
    mse = torch.mean((reconstructed - targets) ** 2).item()
    
    # Mean Absolute Error
    mae = torch.mean(torch.abs(reconstructed - targets)).item()
    
    # Normalized MSE (NMSE)
    target_power = torch.mean(targets ** 2).item()
    nmse = mse / (target_power + 1e-8)
    
    # Pearson correlation coefficient
    def pearson_corr(x, y):
        x_flat = x.view(-1)
        y_flat = y.view(-1)
        x_mean = torch.mean(x_flat)
        y_mean = torch.mean(y_flat)
        num = torch.sum((x_flat - x_mean) * (y_flat - y_mean))
        den = torch.sqrt(torch.sum((x_flat - x_mean) ** 2) * torch.sum((y_flat - y_mean) ** 2))
        return (num / (den + 1e-8)).item()
    
    correlation = pearson_corr(reconstructed, targets)
    
    # Signal-to-Noise Ratio (higher is better)
    signal_power = torch.mean(targets ** 2).item()
    noise_power = torch.mean((reconstructed - targets) ** 2).item()
    snr = 10 * np.log10(signal_power / (noise_power + 1e-8))
    
    return {
        'mse': mse,
        'mae': mae,
        'nmse': nmse,
        'correlation': correlation,
        'snr': snr
    }


def save_reconstruction_plots(reconstructed, targets, save_dir, target_data, noise_type, noise_level):
    """Save plots comparing reconstructed and target data."""
    os.makedirs(save_dir, exist_ok=True)
    
    # Select a few samples to plot
    n_samples = min(4, len(reconstructed))
    indices = np.random.choice(len(reconstructed), n_samples, replace=False)
    
    fig, axes = plt.subplots(n_samples, 4, figsize=(16, 4*n_samples))
    if n_samples == 1:
        axes = axes.reshape(1, -1)
    
    for i, idx in enumerate(indices):
        recon = reconstructed[idx].cpu().numpy()  # Shape: (2, seq_len)
        target = targets[idx].cpu().numpy()       # Shape: (2, seq_len)
        
        # Real part
        axes[i, 0].plot(target[0], label='Target (Real)', alpha=0.7)
        axes[i, 0].plot(recon[0], label='Reconstructed (Real)', alpha=0.7)
        axes[i, 0].set_title(f'Sample {idx+1} - Real Part')
        axes[i, 0].legend()
        axes[i, 0].grid(True)
        
        # Imaginary part
        axes[i, 1].plot(target[1], label='Target (Imag)', alpha=0.7)
        axes[i, 1].plot(recon[1], label='Reconstructed (Imag)', alpha=0.7)
        axes[i, 1].set_title(f'Sample {idx+1} - Imaginary Part')
        axes[i, 1].legend()
        axes[i, 1].grid(True)
        
        # Magnitude
        target_mag = np.sqrt(target[0]**2 + target[1]**2)
        recon_mag = np.sqrt(recon[0]**2 + recon[1]**2)
        axes[i, 2].plot(target_mag, label='Target', alpha=0.7)
        axes[i, 2].plot(recon_mag, label='Reconstructed', alpha=0.7)
        axes[i, 2].set_title(f'Sample {idx+1} - Magnitude')
        axes[i, 2].legend()
        axes[i, 2].grid(True)
        
        # Scatter plot
        axes[i, 3].scatter(target.flatten(), recon.flatten(), alpha=0.5)
        axes[i, 3].plot([target.min(), target.max()], [target.min(), target.max()], 'r--')
        axes[i, 3].set_xlabel('Target')
        axes[i, 3].set_ylabel('Reconstructed')
        axes[i, 3].set_title(f'Sample {idx+1} - Scatter')
        axes[i, 3].grid(True)
    
    plt.tight_layout()
    plot_path = os.path.join(save_dir, f'reconstruction_plots_{target_data}_{noise_type}_{noise_level}.png')
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Saved reconstruction plots to: {plot_path}")


def main():
    """Main function to run the reconstruction attack."""
    args = parse_args()
    
    # Set up device
    device = torch.device(f'cuda:{args.gpu_id}' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create save directory with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = os.path.join(args.save_path, f"attack_results_{timestamp}")
    os.makedirs(save_dir, exist_ok=True)
    
    # Save arguments
    with open(os.path.join(save_dir, 'attack_args.json'), 'w') as f:
        json.dump(vars(args), f, indent=2)
    
    # Initialize wandb
    wandb_run_name = args.wandb_run_name or f"dra_{args.target_data}_{args.noise_type}_{args.noise_level}_{timestamp}"
    wandb.init(
        project=args.wandb_project,
        name=wandb_run_name,
        config=vars(args)
    )
    
    try:
        # Load model information
        print("Loading model information...")
        train_args, checkpoint = load_model_info(args.model_path)
        latent_dim = 2 * train_args.d2  # Flattened activation dimension
        
        # Get output shape for target data
        output_shape = get_data_shape(args.target_data)
        print(f"Target data shape: {output_shape}")
        print(f"Latent dimension: {latent_dim}")
        
        # Create dataset and data loaders
        print("Creating dataset and data loaders...")
        train_loader, test_loader, dataset = create_dataset_and_loader(args, train_args)
        print(f"Train samples: {len(train_loader.dataset)}")
        print(f"Test samples: {len(test_loader.dataset)}")
        
        # Create adversary model
        print(f"Creating {args.adversary_type} adversary model...")
        adversary = get_adversary(
            args.adversary_type,
            latent_dim,
            output_shape,
            hidden_dim=args.adv_hidden_dim,
            dropout=args.adv_dropout
        ).to(device)
        
        # Print model info
        total_params = sum(p.numel() for p in adversary.parameters())
        trainable_params = sum(p.numel() for p in adversary.parameters() if p.requires_grad)
        print(f"Adversary parameters: {trainable_params:,} (total: {total_params:,})")
        
        # Set up training
        criterion = nn.MSELoss()
        optimizer = optim.Adam(adversary.parameters(), lr=args.learning_rate, weight_decay=1e-5)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
        
        # Training loop
        print("Starting adversary training...")
        best_test_loss = float('inf')
        train_losses = []
        test_losses = []
        
        for epoch in range(args.epochs):
            # Train
            train_loss = train_adversary(adversary, train_loader, criterion, optimizer, device, epoch, args.target_data)
            train_losses.append(train_loss)
            
            # Evaluate
            test_loss, _, _, _ = evaluate_adversary(adversary, test_loader, criterion, device, args.target_data)
            test_losses.append(test_loss)
            
            # Update learning rate
            scheduler.step()
            
            # Log to wandb
            wandb.log({
                'epoch': epoch,
                'train_loss': train_loss,
                'test_loss': test_loss,
                'learning_rate': optimizer.param_groups[0]['lr']
            })
            
            print(f"Epoch {epoch+1}/{args.epochs}: Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}")
            
            # Save best model
            if test_loss < best_test_loss:
                best_test_loss = test_loss
                torch.save({
                    'adversary_state_dict': adversary.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'epoch': epoch,
                    'test_loss': test_loss,
                    'args': vars(args)
                }, os.path.join(save_dir, 'best_adversary.pt'))
        
        # Final evaluation
        print("Performing final evaluation...")
        final_test_loss, reconstructed, targets, activations = evaluate_adversary(
            adversary, test_loader, criterion, device, args.target_data
        )
        
        # Calculate metrics
        metrics = calculate_metrics(reconstructed, targets)
        print("Final Reconstruction Metrics:")
        for key, value in metrics.items():
            print(f"  {key}: {value:.6f}")
        
        # Log final metrics to wandb
        wandb.log(metrics)
        
        # Save results
        results = {
            'args': vars(args),
            'metrics': metrics,
            'train_losses': train_losses,
            'test_losses': test_losses,
            'final_test_loss': final_test_loss,
            'model_info': {
                'latent_dim': latent_dim,
                'output_shape': output_shape,
                'adversary_params': trainable_params
            }
        }
        
        with open(os.path.join(save_dir, 'results.json'), 'w') as f:
            json.dump(results, f, indent=2)
        
        # Save detailed results as pickle
        detailed_results = {
            'reconstructed': reconstructed.cpu(),
            'targets': targets.cpu(),
            'activations': activations.cpu(),
            'metrics': metrics
        }
        
        with open(os.path.join(save_dir, 'detailed_results.pkl'), 'wb') as f:
            pickle.dump(detailed_results, f)
        
        # Save reconstruction plots
        if args.save_plots:
            save_reconstruction_plots(reconstructed, targets, save_dir, args.target_data, args.noise_type, args.noise_level)
        
        print(f"Attack completed! Results saved to: {save_dir}")
        print(f"Final reconstruction MSE: {metrics['mse']:.6f}")
        print(f"Final correlation: {metrics['correlation']:.6f}")
        
    except Exception as e:
        print(f"Error during attack: {str(e)}")
        raise
    finally:
        wandb.finish()


if __name__ == "__main__":
    main() 