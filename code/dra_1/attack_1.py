#!/usr/bin/env python3

import os
import sys
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import pickle
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import json
from dotenv import load_dotenv
load_dotenv()

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from dra_1.models import Decoder
from dra_1.py_datasets import ActivationDataset
from dra_1.inject_noise import inject_isotropic_noise, inject_nonisotropic_noise


def plot_sample_reconstruction(decoder, train_batch, val_batch, epoch, save_dir, device):
    """Plot sample reconstructions for train and val data at each epoch."""
    decoder.eval()
    
    fig, axes = plt.subplots(2, 6, figsize=(18, 8))
    fig.suptitle(f'Epoch {epoch} - Sample Reconstructions', fontsize=16)
    
    datasets = [('Train', train_batch), ('Val', val_batch)]
    tasks = ['rf', 'cfo', 'channel']
    
    with torch.no_grad():
        for row_idx, (dataset_name, batch) in enumerate(datasets):
            rf_x, _, cfo_x, _, channel_x, _, activation_batch, _ = batch
            
            # Use first sample only
            activation = activation_batch[:1].squeeze(1).to(device)
            true_rf = rf_x[:1].squeeze(1).to(device)
            true_cfo = cfo_x[:1].squeeze(1).to(device)
            true_channel = channel_x[:1].squeeze(1).to(device)
            
            # Get reconstructions
            reconstructed = decoder(activation)
            
            true_data = [true_rf, true_cfo, true_channel]
            pred_data = [reconstructed['rf'], reconstructed['cfo'], reconstructed['channel']]
            
            for task_idx, (task, true, pred) in enumerate(zip(tasks, true_data, pred_data)):
                col_idx = task_idx * 2
                
                true_np = true[0].cpu().numpy()  # First sample
                pred_np = pred[0].cpu().numpy()
                
                # Real part
                axes[row_idx, col_idx].plot(true_np[0][:100], 'b-', label='True', alpha=0.8)
                axes[row_idx, col_idx].plot(pred_np[0][:100], 'r--', label='Pred', alpha=0.8)
                axes[row_idx, col_idx].set_title(f'{dataset_name} {task.upper()} Real')
                axes[row_idx, col_idx].legend(fontsize=8)
                axes[row_idx, col_idx].grid(True, alpha=0.3)
                
                # Imaginary part
                axes[row_idx, col_idx+1].plot(true_np[1][:100], 'b-', label='True', alpha=0.8)
                axes[row_idx, col_idx+1].plot(pred_np[1][:100], 'r--', label='Pred', alpha=0.8)
                axes[row_idx, col_idx+1].set_title(f'{dataset_name} {task.upper()} Imag')
                axes[row_idx, col_idx+1].legend(fontsize=8)
                axes[row_idx, col_idx+1].grid(True, alpha=0.3)
                
                # Calculate and display correlation
                corr_real = np.corrcoef(true_np[0], pred_np[0])[0, 1]
                corr_imag = np.corrcoef(true_np[1], pred_np[1])[0, 1]
                mse = np.mean((true_np - pred_np)**2)
                
                # Add metrics text
                metrics_text = f'Corr: {corr_real:.3f}/{corr_imag:.3f}\nMSE: {mse:.3f}'
                axes[row_idx, col_idx].text(0.02, 0.98, metrics_text, 
                                          transform=axes[row_idx, col_idx].transAxes,
                                          verticalalignment='top', fontsize=7,
                                          bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    plot_path = os.path.join(save_dir, f'epoch_{epoch:03d}_samples.png')
    plt.savefig(plot_path, dpi=100, bbox_inches='tight')
    plt.close()
    
    # Return metrics for tracking
    with torch.no_grad():
        # Calculate metrics for val data
        rf_x, _, cfo_x, _, channel_x, _, activation_batch, _ = val_batch
        activation = activation_batch[:1].squeeze(1).to(device)
        true_rf = rf_x[:1].squeeze(1).to(device)
        reconstructed = decoder(activation)
        
        true_np = true_rf[0].cpu().numpy()
        pred_np = reconstructed['rf'][0].cpu().numpy()
        
        rf_corr = np.corrcoef(true_np.flatten(), pred_np.flatten())[0, 1]
        rf_mse = np.mean((true_np - pred_np)**2)
        
        return {
            'rf_correlation': rf_corr,
            'rf_mse': rf_mse,
            'rf_pred_mean': pred_np.mean(),
            'rf_pred_std': pred_np.std()
        }


def train_decoder(args):
    """Main training function."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Setup paths
    activation_dir = os.path.join(args.experiment_path, 'activations')
    if not os.path.isdir(activation_dir):
        raise FileNotFoundError(f"Activations directory not found: {activation_dir}")
    
    pkl_file_path = os.environ.get('PKL_FILE_PATH')
    if not pkl_file_path:
        raise ValueError("PKL_FILE_PATH environment variable not set")
    
    partition_file = os.path.join(pkl_file_path, 'rf_partition_dict_0.5.pkl')
    
    # Create save directory
    save_dir = os.path.join(args.experiment_path, 'attack_results', args.noise_type, 
                           f"level_{str(args.noise_level).replace('.', '_')}")
    os.makedirs(save_dir, exist_ok=True)
    
    # Save arguments
    with open(os.path.join(save_dir, 'attack_args.json'), 'w') as f:
        json.dump(vars(args), f, indent=4)
    
    # Load datasets
    with open(partition_file, 'rb') as f:
        partitions = pickle.load(f)
    
    train_files = partitions.get('train')
    val_files = partitions.get('val')
    test_files = partitions.get('test')
    
    train_dataset = ActivationDataset(activation_dir=activation_dir, file_list=train_files)
    val_dataset = ActivationDataset(activation_dir=activation_dir, file_list=val_files, test_mode=True)
    test_dataset = ActivationDataset(activation_dir=activation_dir, file_list=test_files, test_mode=True)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
    
    print(f"Dataset sizes - Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")
    
    # Get fixed batches for plotting
    train_plot_batch = next(iter(train_loader))
    val_plot_batch = next(iter(val_loader))
    
    # Initialize model
    decoder = Decoder(latent_dim=args.latent_dim).to(device)
    optimizer = optim.Adam(decoder.parameters(), lr=args.lr)
    criterion = nn.L1Loss()
    
    # Training loop
    best_val_loss = float('inf')
    patience_counter = 0
    epoch_metrics = []
    
    print(f"Starting training for {args.epochs} epochs...")
    
    for epoch in range(args.epochs):
        # Training phase
        decoder.train()
        train_losses = {'rf': 0, 'cfo': 0, 'channel': 0}
        
        for batch_data in tqdm(train_loader, desc=f"Epoch {epoch+1} [Train]", leave=False):
            rf_x, _, cfo_x, _, channel_x, _, activation_batch, _ = batch_data
            
            activation_batch = activation_batch.squeeze(1).to(device)
            true_rf = rf_x.squeeze(1).to(device)
            true_cfo = cfo_x.squeeze(1).to(device)
            true_channel = channel_x.squeeze(1).to(device)
            
            # Apply noise
            if args.noise_type == 'isotropic':
                activation_batch = inject_isotropic_noise(activation_batch, args.noise_level)
            
            # Forward pass
            optimizer.zero_grad()
            reconstructed = decoder(activation_batch)

           
            
            loss_rf = criterion(reconstructed['rf'], true_rf)
            loss_cfo = criterion(reconstructed['cfo'], true_cfo)
            loss_channel = criterion(reconstructed['channel'], true_channel)
            
            total_loss = (args.rf_loss_weight * loss_rf) + loss_cfo + loss_channel
            
            total_loss.backward()
            optimizer.step()
            
            train_losses['rf'] += loss_rf.item()
            train_losses['cfo'] += loss_cfo.item()
            train_losses['channel'] += loss_channel.item()

        #plot final reconstructed
        plt.plot(reconstructed['rf'][0, 0, :100].cpu().detach().numpy(), label="reconstructed")
        plt.plot(true_rf[0, 0, :100].cpu().detach().numpy(), label="ture")
        plt.legend()
        plt.savefig(os.path.join(save_dir, f'train_{epoch}_rf.png'))
        plt.close()

        
        
        # Validation phase
        decoder.eval()
        val_losses = {'rf': 0, 'cfo': 0, 'channel': 0}
        
        with torch.no_grad():
            for batch_data in tqdm(val_loader, desc=f"Epoch {epoch+1} [Val]", leave=False):
                rf_x, _, cfo_x, _, channel_x, _, activation_batch, _ = batch_data
                
                activation_batch = activation_batch.squeeze(1).to(device)
                true_rf = rf_x.squeeze(1).to(device)
                true_cfo = cfo_x.squeeze(1).to(device)
                true_channel = channel_x.squeeze(1).to(device)
                

                #plot activation
                plt.plot(activation_batch[0, 0, :100].cpu().detach().numpy(), label="activation")
                plt.legend()
                plt.savefig(os.path.join(save_dir, f'val_{epoch}_activation.png'))
                plt.close()

                #plot reconstructed
                if args.noise_type == 'isotropic':
                    activation_batch = inject_isotropic_noise(activation_batch, args.noise_level)
                
                reconstructed = decoder(activation_batch)
                
                val_losses['rf'] += criterion(reconstructed['rf'], true_rf).item()
                val_losses['cfo'] += criterion(reconstructed['cfo'], true_cfo).item()
                val_losses['channel'] += criterion(reconstructed['channel'], true_channel).item()

                plt.plot(reconstructed['rf'][0, 0, :100].cpu().detach().numpy(), label="reconstructed")
                plt.plot(true_rf[0, 0, :100].cpu().detach().numpy(), label="ture")
                plt.legend()
                plt.savefig(os.path.join(save_dir, f'val_{epoch}_rf.png'))
                plt.close()

                import pdb; pdb.set_trace()
        
        # Calculate average losses
        avg_train_losses = {k: v / len(train_loader) for k, v in train_losses.items()}
        avg_val_losses = {k: v / len(val_loader) for k, v in val_losses.items()}
        total_val_loss = (args.rf_loss_weight * avg_val_losses['rf']) + avg_val_losses['cfo'] + avg_val_losses['channel']
        
        # Plot sample reconstructions
        sample_metrics = plot_sample_reconstruction(decoder, train_plot_batch, val_plot_batch, 
                                                   epoch + 1, save_dir, device)
        epoch_metrics.append(sample_metrics)
        
        # Print progress
        print(f"Epoch {epoch+1}/{args.epochs}:")
        print(f"  Train - RF: {avg_train_losses['rf']:.4f}, CFO: {avg_train_losses['cfo']:.4f}, Channel: {avg_train_losses['channel']:.4f}")
        print(f"  Val   - RF: {avg_val_losses['rf']:.4f}, CFO: {avg_val_losses['cfo']:.4f}, Channel: {avg_val_losses['channel']:.4f}")
        print(f"  RF Correlation: {sample_metrics['rf_correlation']:.4f}, MSE: {sample_metrics['rf_mse']:.4f}")
        
        # Early stopping
        if total_val_loss < best_val_loss:
            best_val_loss = total_val_loss
            patience_counter = 0
            torch.save(decoder.state_dict(), os.path.join(save_dir, 'best_decoder.pt'))
            print(f"  New best model saved (Val Loss: {total_val_loss:.6f})")
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"  Early stopping triggered after {patience_counter} epochs without improvement")
                break
    
    # Final evaluation
    print("\nEvaluating on test set...")
    decoder.load_state_dict(torch.load(os.path.join(save_dir, 'best_decoder.pt')))
    decoder.eval()
    
    test_losses = {'rf': 0, 'cfo': 0, 'channel': 0}
    original_samples = {'rf': [], 'cfo': [], 'channel': []}
    reconstructed_samples = {'rf': [], 'cfo': [], 'channel': []}
    
    with torch.no_grad():
        for batch_data in tqdm(test_loader, desc="Testing"):
            rf_x, _, cfo_x, _, channel_x, _, activation_batch, _ = batch_data
            
            activation_batch = activation_batch.squeeze(1).to(device)
            true_rf = rf_x.squeeze(1).to(device)
            true_cfo = cfo_x.squeeze(1).to(device)
            true_channel = channel_x.squeeze(1).to(device)
            
            if args.noise_type == 'isotropic':
                activation_batch = inject_isotropic_noise(activation_batch, args.noise_level)
            
            reconstructed = decoder(activation_batch)
            
            test_losses['rf'] += criterion(reconstructed['rf'], true_rf).item()
            test_losses['cfo'] += criterion(reconstructed['cfo'], true_cfo).item()
            test_losses['channel'] += criterion(reconstructed['channel'], true_channel).item()
            
            # Save samples for final visualization
            if len(original_samples['rf']) < 5:  # First 5 batches
                original_samples['rf'].extend(list(torch.split(true_rf.cpu(), 1))[:3])
                original_samples['cfo'].extend(list(torch.split(true_cfo.cpu(), 1))[:3])
                original_samples['channel'].extend(list(torch.split(true_channel.cpu(), 1))[:3])
                reconstructed_samples['rf'].extend(list(torch.split(reconstructed['rf'].cpu(), 1))[:3])
                reconstructed_samples['cfo'].extend(list(torch.split(reconstructed['cfo'].cpu(), 1))[:3])
                reconstructed_samples['channel'].extend(list(torch.split(reconstructed['channel'].cpu(), 1))[:3])
    
    avg_test_losses = {k: v / len(test_loader) for k, v in test_losses.items()}
    
    print("Final test results:")
    for task in ['rf', 'cfo', 'channel']:
        print(f"  {task.upper()} L1 Loss: {avg_test_losses[task]:.6f}")
    
    # Create final visualization
    create_final_visualization(original_samples, reconstructed_samples, save_dir)
    
    # Plot training trends
    plot_training_trends(epoch_metrics, save_dir)
    
    # Save results
    results = {
        'args': vars(args),
        'test_losses': avg_test_losses,
        'epoch_metrics': epoch_metrics,
        'total_epochs': epoch + 1
    }
    
    with open(os.path.join(save_dir, 'results.json'), 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nTraining completed! Results saved to: {save_dir}")


def create_final_visualization(originals, reconstructions, save_dir):
    """Create final test set visualization."""
    tasks = ['rf', 'cfo', 'channel']
    num_samples = min(3, len(originals['rf']))
    
    fig, axes = plt.subplots(num_samples * len(tasks), 3, 
                             figsize=(15, 3 * num_samples * len(tasks)))
    fig.suptitle("Final Test Reconstruction Results", fontsize=16)
    
    for i in range(num_samples):
        for j, task in enumerate(tasks):
            row_idx = i * len(tasks) + j
            
            original_np = originals[task][i].squeeze(0).numpy()
            recon_np = reconstructions[task][i].squeeze(0).numpy()
            
            # Real part
            axes[row_idx, 0].plot(original_np[0][:100], label='Original')
            axes[row_idx, 0].plot(recon_np[0][:100], label='Reconstructed', linestyle='--')
            axes[row_idx, 0].set_ylabel(f"Sample {i+1}\n{task.upper()}")
            axes[row_idx, 0].set_title("Real Part")
            axes[row_idx, 0].legend()
            axes[row_idx, 0].grid(True, alpha=0.3)
            
            # Imaginary part
            axes[row_idx, 1].plot(original_np[1][:100], label='Original')
            axes[row_idx, 1].plot(recon_np[1][:100], label='Reconstructed', linestyle='--')
            axes[row_idx, 1].set_title("Imaginary Part")
            axes[row_idx, 1].legend()
            axes[row_idx, 1].grid(True, alpha=0.3)
            
            # Magnitude
            original_mag = np.sqrt(original_np[0]**2 + original_np[1]**2)
            recon_mag = np.sqrt(recon_np[0]**2 + recon_np[1]**2)
            axes[row_idx, 2].plot(original_mag[:100], label='Original')
            axes[row_idx, 2].plot(recon_mag[:100], label='Reconstructed', linestyle='--')
            axes[row_idx, 2].set_title("Magnitude")
            axes[row_idx, 2].legend()
            axes[row_idx, 2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'final_reconstruction_results.png'), dpi=150, bbox_inches='tight')
    plt.close()


def plot_training_trends(epoch_metrics, save_dir):
    """Plot RF reconstruction quality trends over epochs."""
    if not epoch_metrics:
        return
    
    epochs = list(range(1, len(epoch_metrics) + 1))
    correlations = [m['rf_correlation'] for m in epoch_metrics]
    mses = [m['rf_mse'] for m in epoch_metrics]
    pred_means = [m['rf_pred_mean'] for m in epoch_metrics]
    pred_stds = [m['rf_pred_std'] for m in epoch_metrics]
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle('RF Reconstruction Training Progress', fontsize=16)
    
    axes[0, 0].plot(epochs, correlations, 'o-', linewidth=2)
    axes[0, 0].set_title('RF Correlation vs Epoch')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Correlation')
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].axhline(y=0, color='r', linestyle='--', alpha=0.5)
    
    axes[0, 1].plot(epochs, mses, 'o-', linewidth=2, color='red')
    axes[0, 1].set_title('RF MSE vs Epoch')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('MSE')
    axes[0, 1].set_yscale('log')
    axes[0, 1].grid(True, alpha=0.3)
    
    axes[1, 0].plot(epochs, pred_means, 'o-', linewidth=2, color='green')
    axes[1, 0].set_title('RF Prediction Mean vs Epoch')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Mean')
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].axhline(y=0, color='r', linestyle='--', alpha=0.5)
    
    axes[1, 1].plot(epochs, pred_stds, 'o-', linewidth=2, color='purple')
    axes[1, 1].set_title('RF Prediction Std vs Epoch')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Standard Deviation')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'training_trends.png'), dpi=150, bbox_inches='tight')
    plt.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Clean Data Reconstruction Attack")
    
    parser.add_argument('--experiment_path', type=str, required=True,
                        help="Path to experiment directory containing activations")
    parser.add_argument('--noise_type', type=str, default='none', choices=['isotropic', 'none'],
                        help="Type of noise to inject")
    parser.add_argument('--noise_level', type=float, default=0.0,
                        help="Noise level")
    parser.add_argument('--epochs', type=int, default=15,
                        help="Training epochs")
    parser.add_argument('--lr', type=float, default=1e-3,
                        help="Learning rate")
    parser.add_argument('--batch_size', type=int, default=128,
                        help="Batch size")
    parser.add_argument('--patience', type=int, default=10,
                        help="Early stopping patience")
    parser.add_argument('--rf_loss_weight', type=float, default=4.0,
                        help="RF loss weight")
    parser.add_argument('--latent_dim', type=int, default=512,
                        help="Latent dimension")
    
    args = parser.parse_args()
    train_decoder(args) 