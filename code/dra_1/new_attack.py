"""
New Reconstruction Attack Implementation

This script implements a privacy attack where an adversary attempts to reconstruct 
original training data (RF signals, CFO estimates, Channel estimates) from encoder 
activations alone.

The attack scenario:
1. Adversary has access to some (activation, original_data) pairs for training
2. Adversary trains a decoder model to reconstruct original data from activations
3. Attack is tested on unseen activations to measure reconstruction quality
4. Experiments include noise robustness testing (z + epsilon)

Key features:
- Uses the Decoder model from models.py as the adversary
- Implements early stopping with patience parameter
- Comprehensive metrics (MSE, MAE, SNR)
- Noise robustness experiments (isotropic and non-isotropic)
- Wandb logging for experiment tracking
- Proper train/test split for attack evaluation
- Support for both isotropic and FIM-based non-isotropic noise injection

Usage:
    # Basic attack with isotropic noise
    python new_attack.py --experiment_path /path/to/results/experiment --epochs 100
    
    # Attack with non-isotropic noise (requires FIM eigendecomposition)
    python new_attack.py --experiment_path /path/to/results/experiment \
                        --noise_type nonisotropic --fim_path /path/to/fim_eigen.pt
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import wandb
import argparse
import json
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
from datetime import datetime

from models import Decoder
from py_datasets import ActivationDataset
from inject_noise import inject_isotropic_noise, inject_nonisotropic_noise


class ReconstructionAttack:
    """
    Implements a reconstruction attack where an adversary tries to reconstruct
    original data (RF, CFO, Channel) from encoder activations.
    """
    
    def __init__(self, config):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # Initialize model
        self.model = Decoder(latent_dim=config['latent_dim']).to(self.device)
        self._initialize_model_weights()
        print(f"Initialized Decoder with latent_dim={config['latent_dim']}")
        
        # Initialize optimizer with weight decay for stability
        self.optimizer = optim.Adam(
            self.model.parameters(), 
            lr=config['learning_rate'],
            weight_decay=config.get('weight_decay', 1e-5),
            eps=1e-8
        )
        
        # Learning rate scheduler for stability
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.7, patience=8
        )
        
        # Loss functions for different tasks
        self.criterion_rf = nn.MSELoss()
        self.criterion_cfo = nn.MSELoss()
        self.criterion_channel = nn.MSELoss()
        
        # Early stopping parameters
        self.best_loss = float('inf')
        self.patience_counter = 0
        
        # Metrics tracking
        self.train_losses = []
        self.val_losses = []
        
        # Noise injection parameters
        self.noise_type = config.get('noise_type', 'isotropic')  # 'isotropic' or 'nonisotropic'
        self.fim_eigenvalues = None
        self.fim_eigenvectors = None
        
        # Setup organized output directories
        self._setup_output_directories()
        
        # Load FIM eigendecomposition if using non-isotropic noise
        if self.noise_type == 'nonisotropic':
            self._load_fim_eigendecomposition(config.get('fim_path', None))
        
        print(f"Noise type: {self.noise_type}")
        print(f"Results will be saved to: {self.results_dir}")
    
    def _initialize_model_weights(self):
        """Initialize model weights for training stability"""
        for module in self.model.modules():
            if isinstance(module, nn.Linear):
                # Xavier/Glorot initialization for linear layers
                nn.init.xavier_normal_(module.weight, gain=0.5)  # Moderate gain for balance
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.ConvTranspose1d):
                # He initialization for transposed convolutions  
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                # Moderate scaling for stability
                with torch.no_grad():
                    module.weight.data *= 0.5
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.BatchNorm1d):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)
        
        print("Applied conservative weight initialization for training stability")
    
    def _setup_output_directories(self):
        """Create organized output directory structure based on noise type and level"""
        base_output_dir = self.config['output_dir']
        
        # Determine noise subdirectory
        if self.config['noise_level'] == 0.0:
            noise_subdir = 'none'
        else:
            noise_subdir = self.noise_type
        
        # Create noise level subdirectory
        noise_level_str = f"level_{self.config['noise_level']:.1f}".replace('.', '_')
        
        # Full results directory path
        self.results_dir = os.path.join(base_output_dir, noise_subdir, noise_level_str)
        
        # Create directories
        os.makedirs(self.results_dir, exist_ok=True)
        os.makedirs(os.path.join(self.results_dir, 'plots'), exist_ok=True)
        os.makedirs(os.path.join(self.results_dir, 'checkpoints'), exist_ok=True)
        
        # Update config to use the organized directory
        self.config['output_dir'] = self.results_dir
        
        print(f"Created output structure:")
        print(f"  Base: {base_output_dir}")
        print(f"  Noise type: {noise_subdir}")
        print(f"  Noise level: {noise_level_str}")
    
    def _load_fim_eigendecomposition(self, fim_path):
        """Load FIM eigenvalues and eigenvectors for non-isotropic noise"""
        if fim_path is None:
            print("Warning: No FIM path provided for non-isotropic noise. Using isotropic noise instead.")
            self.noise_type = 'isotropic'
            return
        
        try:
            if fim_path.endswith('.pt') or fim_path.endswith('.pth'):
                fim_data = torch.load(fim_path, map_location=self.device)
            elif fim_path.endswith('.npz'):
                fim_data = np.load(fim_path)
                fim_data = {k: torch.tensor(v, device=self.device) for k, v in fim_data.items()}
            else:
                raise ValueError(f"Unsupported FIM file format: {fim_path}")
            
            # Extract eigenvalues and eigenvectors
            if 'eigenvalues' in fim_data and 'eigenvectors' in fim_data:
                self.fim_eigenvalues = fim_data['eigenvalues'].to(self.device)
                self.fim_eigenvectors = fim_data['eigenvectors'].to(self.device)
            elif 'L' in fim_data and 'V' in fim_data:
                self.fim_eigenvalues = fim_data['L'].to(self.device)
                self.fim_eigenvectors = fim_data['V'].to(self.device)
            else:
                raise KeyError("FIM data must contain 'eigenvalues'/'eigenvectors' or 'L'/'V' keys")
            
            print(f"Loaded FIM eigendecomposition: {self.fim_eigenvalues.shape[0]} dimensions")
            print(f"Eigenvalue range: [{self.fim_eigenvalues.min():.6f}, {self.fim_eigenvalues.max():.6f}]")
            
        except Exception as e:
            print(f"Error loading FIM eigendecomposition: {e}")
            print("Falling back to isotropic noise.")
            self.noise_type = 'isotropic'
            self.fim_eigenvalues = None
            self.fim_eigenvectors = None
        
    def load_dataset(self, experiment_path):
        """Load dataset from experiment activations folder"""
        activations_dir = os.path.join(experiment_path, 'activations')
        
        if not os.path.exists(activations_dir):
            raise ValueError(f"Activations directory not found: {activations_dir}")
        
        # Get all .pth files in the activations directory
        activation_files = [f for f in os.listdir(activations_dir) if f.endswith('.pth')]
        if not activation_files:
            raise ValueError(f"No .pth files found in {activations_dir}")
        
        # Convert to .mat file paths for ActivationDataset compatibility
        mat_file_list = [f.replace('.pth', '.mat') for f in activation_files]
        
        print(f"Found {len(mat_file_list)} activation files")
        
        # Create dataset
        full_dataset = ActivationDataset(activations_dir, mat_file_list, test_mode=False)
        
        # Split dataset for attack train/test
        train_size = int(self.config['train_split'] * len(full_dataset))
        test_size = len(full_dataset) - train_size
        
        train_dataset, test_dataset = random_split(
            full_dataset, [train_size, test_size],
            generator=torch.Generator().manual_seed(self.config['seed'])
        )
        
        print(f"Attack dataset split: {len(train_dataset)} train, {len(test_dataset)} test")
        
        return train_dataset, test_dataset
    
    def create_dataloaders(self, train_dataset, test_dataset):
        """Create data loaders"""
        train_loader = DataLoader(
            train_dataset, 
            batch_size=self.config['batch_size'], 
            shuffle=True,
            num_workers=4,
            pin_memory=True if self.device.type == 'cuda' else False
        )
        
        test_loader = DataLoader(
            test_dataset, 
            batch_size=self.config['batch_size'], 
            shuffle=False,
            num_workers=4,
            pin_memory=True if self.device.type == 'cuda' else False
        )
        
        return train_loader, test_loader
    
    def add_noise(self, activations, noise_level):
        """Add noise to activations based on configured noise type"""
        if noise_level == 0:
            return activations
        
        # Flatten activations for noise injection (the functions expect 2D tensors)
        original_shape = activations.shape
        batch_size = activations.shape[0]
        flattened_activations = activations.view(batch_size, -1)
        
        if self.noise_type == 'isotropic':
            noisy_activations = inject_isotropic_noise(flattened_activations, noise_level)
        elif self.noise_type == 'nonisotropic':
            if self.fim_eigenvalues is not None and self.fim_eigenvectors is not None:
                noisy_activations = inject_nonisotropic_noise(
                    flattened_activations, 
                    noise_level, 
                    self.fim_eigenvalues, 
                    self.fim_eigenvectors
                )
            else:
                print("Warning: FIM data not available. Using isotropic noise.")
                noisy_activations = inject_isotropic_noise(flattened_activations, noise_level)
        else:
            raise ValueError(f"Unknown noise type: {self.noise_type}")
        
        # Reshape back to original shape
        return noisy_activations.view(original_shape)
    
    def compute_metrics(self, predictions, targets, task_name):
        """Compute reconstruction metrics"""
        pred_np = predictions.cpu().detach().numpy()
        target_np = targets.cpu().detach().numpy()
        
        mse = mean_squared_error(target_np.flatten(), pred_np.flatten())
        mae = mean_absolute_error(target_np.flatten(), pred_np.flatten())
        
        # Signal-to-Noise Ratio
        signal_power = np.mean(target_np ** 2)
        noise_power = np.mean((target_np - pred_np) ** 2)
        snr = 10 * np.log10(signal_power / (noise_power + 1e-8))
        
        return {
            f'{task_name}_mse': mse,
            f'{task_name}_mae': mae,
            f'{task_name}_snr': snr
        }
    
    def train_epoch(self, train_loader, noise_level=0.0):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0.0
        rf_loss_sum = 0.0
        cfo_loss_sum = 0.0
        channel_loss_sum = 0.0
        
        for batch_idx, batch in enumerate(train_loader):
            rf_x, rf_y, cfo_x, cfo_y, channel_x, channel_y, activation, filename = batch
            
            # Move to device and fix shapes
            activation = activation.to(self.device)
            rf_x = rf_x.to(self.device)
            cfo_x = cfo_x.to(self.device)
            channel_x = channel_x.to(self.device)
            
            # Debug: Print original shapes for first batch
            if batch_idx == 0:
                print(f"Original shapes - RF: {rf_x.shape}, CFO: {cfo_x.shape}, Channel: {channel_x.shape}, Activation: {activation.shape}")
            
            # Fix tensor shapes to match model output
            # Handle 4D tensors: [batch, 1, channels, length] -> [batch, channels, length]
            if len(rf_x.shape) == 4:
                if rf_x.shape[1] == 1:
                    rf_x = rf_x.squeeze(1)
                else:
                    rf_x = rf_x.view(rf_x.shape[0], rf_x.shape[2], rf_x.shape[3])
            
            if len(cfo_x.shape) == 4:
                if cfo_x.shape[1] == 1:
                    cfo_x = cfo_x.squeeze(1)
                else:
                    cfo_x = cfo_x.view(cfo_x.shape[0], cfo_x.shape[2], cfo_x.shape[3])
                    
            if len(channel_x.shape) == 4:
                if channel_x.shape[1] == 1:
                    channel_x = channel_x.squeeze(1)
                else:
                    channel_x = channel_x.view(channel_x.shape[0], channel_x.shape[2], channel_x.shape[3])
            
            # Debug: Print final shapes for first batch
            if batch_idx == 0:
                print(f"After reshape - RF: {rf_x.shape}, CFO: {cfo_x.shape}, Channel: {channel_x.shape}")
                print(f"Target value ranges - RF: [{rf_x.min():.4f}, {rf_x.max():.4f}], "
                      f"CFO: [{cfo_x.min():.4f}, {cfo_x.max():.4f}], "
                      f"Channel: [{channel_x.min():.4f}, {channel_x.max():.4f}]")
            
            # Add noise to activations
            noisy_activation = self.add_noise(activation, noise_level)
            
            # Ensure activation has the right shape for decoder
            # Expected: [batch, 2, 256] -> flattened to [batch, 512] in decoder
            if len(noisy_activation.shape) == 2:  # Already flattened [batch, 512]
                # Reshape to [batch, 2, 256] 
                expected_channels = 2
                expected_length = noisy_activation.shape[1] // expected_channels
                noisy_activation = noisy_activation.view(noisy_activation.shape[0], expected_channels, expected_length)
            
            # Forward pass
            self.optimizer.zero_grad()
            reconstructions = self.model(noisy_activation)
            
            # Check for NaN/Inf in reconstructions
            if torch.isnan(reconstructions['rf']).any() or torch.isinf(reconstructions['rf']).any():
                print(f"Warning: NaN/Inf detected in RF reconstruction at batch {batch_idx}")
                continue
            
            # Compute losses with stability checks
            rf_loss = self.criterion_rf(reconstructions['rf'], rf_x)
            cfo_loss = self.criterion_cfo(reconstructions['cfo'], cfo_x)
            channel_loss = self.criterion_channel(reconstructions['channel'], channel_x)
            
            # Check for exploding losses (only skip if really extreme)
            if rf_loss > 1e8 or cfo_loss > 1e8 or channel_loss > 1e8:
                print(f"Warning: Extreme loss detected at batch {batch_idx}: "
                      f"RF={rf_loss:.2f}, CFO={cfo_loss:.2f}, Channel={channel_loss:.2f}")
                # Skip this batch to prevent gradient explosion
                continue
            
            # Combined loss
            total_batch_loss = (
                self.config['rf_weight'] * rf_loss + 
                self.config['cfo_weight'] * cfo_loss + 
                self.config['channel_weight'] * channel_loss
            )
            
            # Backward pass with gradient clipping
            total_batch_loss.backward()
            
            # Gradient clipping for stability (relaxed)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5.0)
            
            # Check gradient norms
            total_grad_norm = 0
            for param in self.model.parameters():
                if param.grad is not None:
                    total_grad_norm += param.grad.norm().item() ** 2
            total_grad_norm = total_grad_norm ** 0.5
            
            if total_grad_norm > 10.0:
                print(f"Warning: Large gradient norm {total_grad_norm:.2f} at batch {batch_idx}")
            
            self.optimizer.step()
            
            total_loss += total_batch_loss.item()
            rf_loss_sum += rf_loss.item()
            cfo_loss_sum += cfo_loss.item()
            channel_loss_sum += channel_loss.item()
            
            if batch_idx % self.config['log_interval'] == 0:
                print(f'Batch {batch_idx}/{len(train_loader)}, Loss: {total_batch_loss.item():.6f}')
        
        avg_loss = total_loss / len(train_loader)
        avg_rf_loss = rf_loss_sum / len(train_loader)
        avg_cfo_loss = cfo_loss_sum / len(train_loader)
        avg_channel_loss = channel_loss_sum / len(train_loader)
        
        return avg_loss, avg_rf_loss, avg_cfo_loss, avg_channel_loss
    
    def validate(self, test_loader, noise_level=0.0):
        """Validate the model"""
        self.model.eval()
        total_loss = 0.0
        all_metrics = {}
        
        with torch.no_grad():
            for batch in test_loader:
                rf_x, rf_y, cfo_x, cfo_y, channel_x, channel_y, activation, filename = batch
                
                # Move to device and fix shapes
                activation = activation.to(self.device)
                rf_x = rf_x.to(self.device)
                cfo_x = cfo_x.to(self.device)
                channel_x = channel_x.to(self.device)
                
                # Fix tensor shapes to match model output
                # Handle 4D tensors: [batch, 1, channels, length] -> [batch, channels, length]
                if len(rf_x.shape) == 4:
                    if rf_x.shape[1] == 1:
                        rf_x = rf_x.squeeze(1)
                    else:
                        rf_x = rf_x.view(rf_x.shape[0], rf_x.shape[2], rf_x.shape[3])
                
                if len(cfo_x.shape) == 4:
                    if cfo_x.shape[1] == 1:
                        cfo_x = cfo_x.squeeze(1)
                    else:
                        cfo_x = cfo_x.view(cfo_x.shape[0], cfo_x.shape[2], cfo_x.shape[3])
                        
                if len(channel_x.shape) == 4:
                    if channel_x.shape[1] == 1:
                        channel_x = channel_x.squeeze(1)
                    else:
                        channel_x = channel_x.view(channel_x.shape[0], channel_x.shape[2], channel_x.shape[3])
                
                # Add noise to activations
                noisy_activation = self.add_noise(activation, noise_level)
                
                # Ensure activation has the right shape for decoder
                if len(noisy_activation.shape) == 2:  # Already flattened [batch, 512]
                    # Reshape to [batch, 2, 256] 
                    expected_channels = 2
                    expected_length = noisy_activation.shape[1] // expected_channels
                    noisy_activation = noisy_activation.view(noisy_activation.shape[0], expected_channels, expected_length)
                
                # Forward pass
                reconstructions = self.model(noisy_activation)
                
                # Compute losses
                rf_loss = self.criterion_rf(reconstructions['rf'], rf_x)
                cfo_loss = self.criterion_cfo(reconstructions['cfo'], cfo_x)
                channel_loss = self.criterion_channel(reconstructions['channel'], channel_x)
                
                batch_loss = (
                    self.config['rf_weight'] * rf_loss + 
                    self.config['cfo_weight'] * cfo_loss + 
                    self.config['channel_weight'] * channel_loss
                )
                
                total_loss += batch_loss.item()
                
                # Compute detailed metrics for the first batch
                if len(all_metrics) == 0:
                    rf_metrics = self.compute_metrics(reconstructions['rf'], rf_x, 'rf')
                    cfo_metrics = self.compute_metrics(reconstructions['cfo'], cfo_x, 'cfo')
                    channel_metrics = self.compute_metrics(reconstructions['channel'], channel_x, 'channel')
                    
                    all_metrics.update(rf_metrics)
                    all_metrics.update(cfo_metrics)
                    all_metrics.update(channel_metrics)
        
        avg_loss = total_loss / len(test_loader)
        all_metrics['val_loss'] = avg_loss
        
        return avg_loss, all_metrics
    
    def save_checkpoint(self, epoch, loss, filepath):
        """Save model checkpoint"""
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'loss': loss,
            'config': self.config
        }, filepath)
    
    def plot_reconstructions(self, test_loader, epoch, num_samples=3):
        """Plot reconstruction vs true values for qualitative assessment"""
        self.model.eval()
        
        with torch.no_grad():
            # Get one batch for plotting
            batch = next(iter(test_loader))
            rf_x, rf_y, cfo_x, cfo_y, channel_x, channel_y, activation, filename = batch
            
            # Move to device and fix shapes
            activation = activation.to(self.device)
            rf_x = rf_x.to(self.device)
            cfo_x = cfo_x.to(self.device)
            channel_x = channel_x.to(self.device)
            
            # Fix tensor shapes
            if len(rf_x.shape) == 4:
                if rf_x.shape[1] == 1:
                    rf_x = rf_x.squeeze(1)
                else:
                    rf_x = rf_x.view(rf_x.shape[0], rf_x.shape[2], rf_x.shape[3])
            
            if len(cfo_x.shape) == 4:
                if cfo_x.shape[1] == 1:
                    cfo_x = cfo_x.squeeze(1)
                else:
                    cfo_x = cfo_x.view(cfo_x.shape[0], cfo_x.shape[2], cfo_x.shape[3])
                    
            if len(channel_x.shape) == 4:
                if channel_x.shape[1] == 1:
                    channel_x = channel_x.squeeze(1)
                else:
                    channel_x = channel_x.view(channel_x.shape[0], channel_x.shape[2], channel_x.shape[3])
            
            # Ensure activation has the right shape
            if len(activation.shape) == 2:
                expected_channels = 2
                expected_length = activation.shape[1] // expected_channels
                activation = activation.view(activation.shape[0], expected_channels, expected_length)
            
            # Get reconstructions
            reconstructions = self.model(activation)
            
            # Move to CPU for plotting
            rf_true = rf_x[:num_samples].cpu().numpy()
            rf_pred = reconstructions['rf'][:num_samples].cpu().numpy()
            cfo_true = cfo_x[:num_samples].cpu().numpy()
            cfo_pred = reconstructions['cfo'][:num_samples].cpu().numpy()
            channel_true = channel_x[:num_samples].cpu().numpy()
            channel_pred = reconstructions['channel'][:num_samples].cpu().numpy()
            
            # Create a cleaner plot with separate I/Q subplots
            fig, axes = plt.subplots(6, num_samples, figsize=(4*num_samples, 16))
            noise_info = f" ({self.noise_type.title()} Noise)" if self.config['noise_level'] > 0 else ""
            fig.suptitle(f'Reconstruction Results - Epoch {epoch}{noise_info}', fontsize=16, y=0.98)
            
            tasks_data = [
                ('RF I-component', rf_true, rf_pred, 0),
                ('RF Q-component', rf_true, rf_pred, 1),
                ('CFO I-component', cfo_true, cfo_pred, 0),
                ('CFO Q-component', cfo_true, cfo_pred, 1),
                ('Channel I-component', channel_true, channel_pred, 0),
                ('Channel Q-component', channel_true, channel_pred, 1),
            ]
            
            for row_idx, (task_name, true_data, pred_data, channel_idx) in enumerate(tasks_data):
                for sample_idx in range(num_samples):
                    ax = axes[row_idx, sample_idx]
                    
                    # Get the specific channel data
                    true_signal = true_data[sample_idx, channel_idx, :]
                    pred_signal = pred_data[sample_idx, channel_idx, :]
                    
                    # Plot true vs predicted
                    ax.plot(true_signal, 'b-', linewidth=1.5, label='True', alpha=0.8)
                    ax.plot(pred_signal, 'r--', linewidth=1.5, label='Pred', alpha=0.8)
                    
                    # Compute and display correlation
                    correlation = np.corrcoef(true_signal, pred_signal)[0, 1]
                    mse = np.mean((true_signal - pred_signal) ** 2)
                    
                    ax.set_title(f'{task_name} - Sample {sample_idx + 1}\nCorr: {correlation:.3f}, MSE: {mse:.4f}', 
                               fontsize=10)
                    
                    if sample_idx == 0:
                        ax.set_ylabel('Amplitude', fontsize=9)
                    if row_idx == 5:  # Last row
                        ax.set_xlabel('Time/Frequency', fontsize=9)
                    
                    ax.legend(fontsize=8)
                    ax.grid(True, alpha=0.3)
                    ax.tick_params(labelsize=8)
            
            plt.tight_layout()
            plt.subplots_adjust(top=0.95)
            
            # Save plot
            plot_filename = f'reconstruction_samples_epoch_{epoch}.png'
            plot_path = os.path.join(self.config['output_dir'], 'plots', plot_filename)
            plt.savefig(plot_path, dpi=150, bbox_inches='tight')
            plt.close()
            
            print(f"Saved reconstruction plot: {plot_path}")
            
            # Log to wandb if enabled
            if self.config['use_wandb']:
                wandb.log({f"reconstruction_plot_epoch_{epoch}": wandb.Image(plot_path)})
            
            return plot_path
    
    def plot_reconstructions_compact(self, test_loader, epoch):
        """Plot compact reconstruction comparison - one sample per task"""
        self.model.eval()
        
        with torch.no_grad():
            # Get one batch for plotting
            batch = next(iter(test_loader))
            rf_x, rf_y, cfo_x, cfo_y, channel_x, channel_y, activation, filename = batch
            
            # Move to device and fix shapes
            activation = activation.to(self.device)
            rf_x = rf_x.to(self.device)
            cfo_x = cfo_x.to(self.device)
            channel_x = channel_x.to(self.device)
            
            # Fix tensor shapes
            if len(rf_x.shape) == 4:
                if rf_x.shape[1] == 1:
                    rf_x = rf_x.squeeze(1)
                else:
                    rf_x = rf_x.view(rf_x.shape[0], rf_x.shape[2], rf_x.shape[3])
            
            if len(cfo_x.shape) == 4:
                if cfo_x.shape[1] == 1:
                    cfo_x = cfo_x.squeeze(1)
                else:
                    cfo_x = cfo_x.view(cfo_x.shape[0], cfo_x.shape[2], cfo_x.shape[3])
                    
            if len(channel_x.shape) == 4:
                if channel_x.shape[1] == 1:
                    channel_x = channel_x.squeeze(1)
                else:
                    channel_x = channel_x.view(channel_x.shape[0], channel_x.shape[2], channel_x.shape[3])
            
            # Ensure activation has the right shape
            if len(activation.shape) == 2:
                expected_channels = 2
                expected_length = activation.shape[1] // expected_channels
                activation = activation.view(activation.shape[0], expected_channels, expected_length)
            
            # Get reconstructions
            reconstructions = self.model(activation)
            
            # Take just the first sample
            rf_true = rf_x[0].cpu().numpy()
            rf_pred = reconstructions['rf'][0].cpu().numpy()
            cfo_true = cfo_x[0].cpu().numpy()
            cfo_pred = reconstructions['cfo'][0].cpu().numpy()
            channel_true = channel_x[0].cpu().numpy()
            channel_pred = reconstructions['channel'][0].cpu().numpy()
            
            # Create compact 3x2 plot (3 tasks, 2 channels each)
            fig, axes = plt.subplots(3, 2, figsize=(12, 10))
            noise_info = f" ({self.noise_type.title()} Noise)" if self.config['noise_level'] > 0 else ""
            fig.suptitle(f'Reconstruction Quality - Epoch {epoch}{noise_info}', fontsize=14, y=0.98)
            
            tasks_data = [
                ('RF Signal', rf_true, rf_pred),
                ('CFO Estimate', cfo_true, cfo_pred),
                ('Channel Estimate', channel_true, channel_pred),
            ]
            
            for row_idx, (task_name, true_data, pred_data) in enumerate(tasks_data):
                for col_idx, channel_name in enumerate(['I-component', 'Q-component']):
                    ax = axes[row_idx, col_idx]
                    
                    # Get the channel data
                    true_signal = true_data[col_idx, :]
                    pred_signal = pred_data[col_idx, :]
                    
                    # Plot true vs predicted
                    ax.plot(true_signal, 'b-', linewidth=2, label='True', alpha=0.8)
                    ax.plot(pred_signal, 'r--', linewidth=2, label='Predicted', alpha=0.8)
                    
                    # Compute metrics
                    correlation = np.corrcoef(true_signal, pred_signal)[0, 1]
                    mse = np.mean((true_signal - pred_signal) ** 2)
                    mae = np.mean(np.abs(true_signal - pred_signal))
                    
                    ax.set_title(f'{task_name} - {channel_name}\nCorr: {correlation:.3f} | MSE: {mse:.4f} | MAE: {mae:.4f}', 
                               fontsize=11)
                    
                    ax.set_ylabel('Amplitude', fontsize=10)
                    if row_idx == 2:  # Last row
                        ax.set_xlabel('Sample Index', fontsize=10)
                    
                    ax.legend(fontsize=9)
                    ax.grid(True, alpha=0.3)
                    ax.tick_params(labelsize=9)
            
            plt.tight_layout()
            plt.subplots_adjust(top=0.93)
            
            # Save plot
            plot_filename = f'reconstruction_compact_epoch_{epoch}.png'
            plot_path = os.path.join(self.config['output_dir'], 'plots', plot_filename)
            plt.savefig(plot_path, dpi=150, bbox_inches='tight')
            plt.close()
            
            print(f"Saved compact reconstruction plot: {plot_path}")
            
            # Log to wandb if enabled
            if self.config['use_wandb']:
                wandb.log({f"reconstruction_compact_epoch_{epoch}": wandb.Image(plot_path)})
            
            return plot_path
    
    def run_attack(self, experiment_path):
        """Run the complete attack"""
        print(f"Starting reconstruction attack on: {experiment_path}")
        
        # Load dataset
        train_dataset, test_dataset = self.load_dataset(experiment_path)
        train_loader, test_loader = self.create_dataloaders(train_dataset, test_dataset)
        
        # Initialize wandb
        if self.config['use_wandb']:
            wandb.init(
                project="reconstruction-attack",
                config=self.config,
                name=f"attack_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            )
        
        print("Starting training...")
        for epoch in range(self.config['epochs']):
            print(f"\nEpoch {epoch+1}/{self.config['epochs']}")
            
            # Optional warmup only for first epoch if needed
            if epoch == 0:
                warmup_lr = self.config['learning_rate'] * 0.5  # 50% of normal LR
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = warmup_lr
                print(f"First epoch warmup: LR = {warmup_lr:.2e}")
            elif epoch == 1:
                # Restore normal learning rate after brief warmup
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = self.config['learning_rate']
                print(f"Normal training: LR = {self.config['learning_rate']:.2e}")
            
            # Training
            train_loss, train_rf_loss, train_cfo_loss, train_channel_loss = self.train_epoch(
                train_loader, self.config['noise_level']
            )
            
            # Validation
            val_loss, val_metrics = self.validate(test_loader, self.config['noise_level'])
            
            # Step learning rate scheduler
            self.scheduler.step(val_loss)
            
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            
            # Get current learning rate
            current_lr = self.optimizer.param_groups[0]['lr']
            
            print(f"Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}, LR: {current_lr:.2e}")
            print(f"RF SNR: {val_metrics.get('rf_snr', 0):.2f} dB, "
                  f"CFO SNR: {val_metrics.get('cfo_snr', 0):.2f} dB, "
                  f"Channel SNR: {val_metrics.get('channel_snr', 0):.2f} dB")
            
            # Log to wandb
            if self.config['use_wandb']:
                log_dict = {
                    'epoch': epoch,
                    'train_loss': train_loss,
                    'train_rf_loss': train_rf_loss,
                    'train_cfo_loss': train_cfo_loss,
                    'train_channel_loss': train_channel_loss,
                }
                log_dict.update(val_metrics)
                wandb.log(log_dict)
            
            # Early stopping
            if val_loss < self.best_loss:
                self.best_loss = val_loss
                self.patience_counter = 0
                
                # Save best model
                checkpoint_path = os.path.join(self.config['output_dir'], 'checkpoints', 'best_adversary.pt')
                self.save_checkpoint(epoch, val_loss, checkpoint_path)
                print(f"New best model saved with loss: {val_loss:.6f}")
                
                # Plot reconstructions for the best model
                self.plot_reconstructions_compact(test_loader, epoch)
            else:
                self.patience_counter += 1
                
            # Plot reconstructions periodically (compact version)
            if epoch % self.config.get('plot_interval', 10) == 0 or epoch == 0:
                self.plot_reconstructions_compact(test_loader, epoch)
                
            if self.patience_counter >= self.config['patience']:
                print(f"Early stopping triggered after {epoch+1} epochs")
                break
        
        # Final reconstruction plots (both compact and detailed)
        print("Generating final reconstruction plots...")
        self.plot_reconstructions_compact(test_loader, epoch)
        self.plot_reconstructions(test_loader, epoch, num_samples=3)  # Detailed view with multiple samples
        
        # Run noise experiments
        if self.config['run_noise_experiments']:
            self.run_noise_experiments(test_loader)
        
        if self.config['use_wandb']:
            wandb.finish()
        

        # Generate summary
        self._generate_attack_summary()
        
        print("Attack completed!")
        print(f"\nResults saved to: {self.results_dir}")
        print(f"  📁 Plots: {os.path.join(self.results_dir, 'plots')}")
        print(f"  💾 Models: {os.path.join(self.results_dir, 'checkpoints')}")
        print(f"  📊 Config: {os.path.join(self.results_dir, 'attack_config.json')}")
        if self.config['run_noise_experiments']:
            print(f"  🔊 Noise Results: {os.path.join(self.results_dir, 'noise_experiments.json')}")
    
    def _generate_attack_summary(self):
        """Generate a summary of the attack results"""
        summary = {
            'attack_info': {
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'noise_type': self.noise_type,
                'noise_level': self.config['noise_level'],
                'epochs_completed': len(self.train_losses),
                'best_validation_loss': self.best_loss,
                'early_stopping_triggered': self.patience_counter >= self.config['patience']
            },
            'final_metrics': {
                'final_train_loss': self.train_losses[-1] if self.train_losses else None,
                'final_val_loss': self.val_losses[-1] if self.val_losses else None,
                'best_val_loss': self.best_loss,
                'training_epochs': len(self.train_losses)
            },
            'configuration': {
                'latent_dim': self.config['latent_dim'],
                'batch_size': self.config['batch_size'],
                'learning_rate': self.config['learning_rate'],
                'patience': self.config['patience'],
                'train_split': self.config['train_split']
            },
            'directory_structure': {
                'base_dir': self.results_dir,
                'plots_dir': os.path.join(self.results_dir, 'plots'),
                'checkpoints_dir': os.path.join(self.results_dir, 'checkpoints'),
                'config_file': os.path.join(self.results_dir, 'attack_config.json'),
                'noise_results': os.path.join(self.results_dir, 'noise_experiments.json')
            }
        }
        
        # Save summary
        summary_path = os.path.join(self.results_dir, 'attack_summary.json')
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"📋 Attack summary saved to: {summary_path}")
        
    def run_noise_experiments(self, test_loader):
        """Run experiments with different noise levels"""
        print("\nRunning noise robustness experiments...")
        
        noise_levels = [0.0, 0.01, 0.05, 0.1, 0.2, 0.5]
        results = {}
        
        for noise_level in noise_levels:
            print(f"Testing with noise level: {noise_level}")
            val_loss, val_metrics = self.validate(test_loader, noise_level)
            results[noise_level] = val_metrics
            
            # Plot reconstructions with this noise level
            self.plot_reconstructions_with_noise(test_loader, noise_level)
            
            if self.config['use_wandb']:
                wandb.log({
                    f'noise_{noise_level}_val_loss': val_loss,
                    f'noise_{noise_level}_rf_snr': val_metrics.get('rf_snr', 0),
                    f'noise_{noise_level}_cfo_snr': val_metrics.get('cfo_snr', 0),
                    f'noise_{noise_level}_channel_snr': val_metrics.get('channel_snr', 0),
                })
        
        # Save noise experiment results
        results_path = os.path.join(self.config['output_dir'], 'noise_experiments.json')
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"Saved noise experiment results to: {results_path}")
    
    def plot_reconstructions_with_noise(self, test_loader, noise_level, num_samples=3):
        """Plot reconstructions with specific noise level for qualitative assessment"""
        self.model.eval()
        
        with torch.no_grad():
            # Get one batch for plotting
            batch = next(iter(test_loader))
            rf_x, rf_y, cfo_x, cfo_y, channel_x, channel_y, activation, filename = batch
            
            # Move to device and fix shapes
            activation = activation.to(self.device)
            rf_x = rf_x.to(self.device)
            cfo_x = cfo_x.to(self.device)
            channel_x = channel_x.to(self.device)
            
            # Fix tensor shapes
            if len(rf_x.shape) == 4:
                if rf_x.shape[1] == 1:
                    rf_x = rf_x.squeeze(1)
                else:
                    rf_x = rf_x.view(rf_x.shape[0], rf_x.shape[2], rf_x.shape[3])
            
            if len(cfo_x.shape) == 4:
                if cfo_x.shape[1] == 1:
                    cfo_x = cfo_x.squeeze(1)
                else:
                    cfo_x = cfo_x.view(cfo_x.shape[0], cfo_x.shape[2], cfo_x.shape[3])
                    
            if len(channel_x.shape) == 4:
                if channel_x.shape[1] == 1:
                    channel_x = channel_x.squeeze(1)
                else:
                    channel_x = channel_x.view(channel_x.shape[0], channel_x.shape[2], channel_x.shape[3])
            
            # Ensure activation has the right shape
            if len(activation.shape) == 2:
                expected_channels = 2
                expected_length = activation.shape[1] // expected_channels
                activation = activation.view(activation.shape[0], expected_channels, expected_length)
            
            # Add noise to activations
            noisy_activation = self.add_noise(activation, noise_level)
            
            # Get reconstructions
            reconstructions = self.model(noisy_activation)
            
            # Move to CPU for plotting
            rf_true = rf_x[:num_samples].cpu().numpy()
            rf_pred = reconstructions['rf'][:num_samples].cpu().numpy()
            cfo_true = cfo_x[:num_samples].cpu().numpy()
            cfo_pred = reconstructions['cfo'][:num_samples].cpu().numpy()
            channel_true = channel_x[:num_samples].cpu().numpy()
            channel_pred = reconstructions['channel'][:num_samples].cpu().numpy()
            
            # Create a cleaner plot with separate I/Q subplots
            fig, axes = plt.subplots(6, num_samples, figsize=(4*num_samples, 16))
            fig.suptitle(f'Reconstruction with {self.noise_type.title()} Noise Level {noise_level}', fontsize=16, y=0.98)
            
            tasks_data = [
                ('RF I-component', rf_true, rf_pred, 0),
                ('RF Q-component', rf_true, rf_pred, 1),
                ('CFO I-component', cfo_true, cfo_pred, 0),
                ('CFO Q-component', cfo_true, cfo_pred, 1),
                ('Channel I-component', channel_true, channel_pred, 0),
                ('Channel Q-component', channel_true, channel_pred, 1),
            ]
            
            for row_idx, (task_name, true_data, pred_data, channel_idx) in enumerate(tasks_data):
                for sample_idx in range(num_samples):
                    ax = axes[row_idx, sample_idx]
                    
                    # Get the specific channel data
                    true_signal = true_data[sample_idx, channel_idx, :]
                    pred_signal = pred_data[sample_idx, channel_idx, :]
                    
                    # Plot true vs predicted
                    ax.plot(true_signal, 'b-', linewidth=1.5, label='True', alpha=0.8)
                    ax.plot(pred_signal, 'r--', linewidth=1.5, label='Pred', alpha=0.8)
                    
                    # Compute and display correlation
                    correlation = np.corrcoef(true_signal, pred_signal)[0, 1]
                    mse = np.mean((true_signal - pred_signal) ** 2)
                    
                    ax.set_title(f'{task_name} - Sample {sample_idx + 1}\nCorr: {correlation:.3f}, MSE: {mse:.4f}', 
                               fontsize=10)
                    
                    if sample_idx == 0:
                        ax.set_ylabel('Amplitude', fontsize=9)
                    if row_idx == 5:  # Last row
                        ax.set_xlabel('Time/Frequency', fontsize=9)
                    
                    ax.legend(fontsize=8)
                    ax.grid(True, alpha=0.3)
                    ax.tick_params(labelsize=8)
            
            plt.tight_layout()
            plt.subplots_adjust(top=0.95)
            
            # Save plot
            plot_filename = f'reconstruction_noise_{noise_level:.3f}.png'
            plot_path = os.path.join(self.config['output_dir'], 'plots', plot_filename)
            plt.savefig(plot_path, dpi=150, bbox_inches='tight')
            plt.close()
            
            print(f"Saved noise experiment plot: {plot_path}")
            
            # Log to wandb if enabled
            if self.config['use_wandb']:
                wandb.log({f"reconstruction_noise_{noise_level}": wandb.Image(plot_path)})
            
            return plot_path


def main():
    parser = argparse.ArgumentParser(description='Reconstruction Attack')
    parser.add_argument('--experiment_path', type=str, required=True,
                       help='Path to experiment directory containing activations')
    parser.add_argument('--output_dir', type=str, default='./attack_results',
                       help='Output directory for results')
    parser.add_argument('--epochs', type=int, default=100,
                       help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=1e-4,
                       help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-6,
                       help='Weight decay for regularization')
    parser.add_argument('--patience', type=int, default=10,
                       help='Early stopping patience')
    parser.add_argument('--noise_level', type=float, default=0.0,
                       help='Noise level for training')
    parser.add_argument('--noise_type', type=str, default='isotropic', 
                       choices=['isotropic', 'nonisotropic'],
                       help='Type of noise to inject (isotropic or nonisotropic)')
    parser.add_argument('--fim_path', type=str, default=None,
                       help='Path to FIM eigendecomposition file (.pt, .pth, or .npz) for non-isotropic noise')
    parser.add_argument('--no_wandb', action='store_true',
                       help='Disable wandb logging')
    parser.add_argument('--plot_interval', type=int, default=10,
                       help='Plot reconstructions every N epochs')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    
    args = parser.parse_args()
    
    # Configuration
    config = {
        'latent_dim': 512,  # Should match the encoder output
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'learning_rate': args.learning_rate,
        'weight_decay': args.weight_decay,
        'patience': args.patience,
        'noise_level': args.noise_level,
        'noise_type': args.noise_type,
        'fim_path': args.fim_path,
        'train_split': 0.8,
        'rf_weight': 1.0,
        'cfo_weight': 1.0,
        'channel_weight': 1.0,
        'log_interval': 50,
        'plot_interval': args.plot_interval,  # Plot reconstructions every N epochs
        'use_wandb': not args.no_wandb,
        'run_noise_experiments': True,
        'output_dir': args.output_dir,
        'seed': args.seed
    }
    
    # Initialize the attack (this will create the organized directory structure)
    attack = ReconstructionAttack(config)
    
    # Save configuration in the organized directory
    config_path = os.path.join(config['output_dir'], 'attack_config.json')
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    # Set random seeds
    torch.manual_seed(config['seed'])
    np.random.seed(config['seed'])
    
    # Run attack
    attack.run_attack(args.experiment_path)


if __name__ == '__main__':
    main() 